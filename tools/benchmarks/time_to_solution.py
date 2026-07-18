"""Time-to-solution table: Fortran v3 MPI ranks vs one dkx process.

Runs the same production namelist through (a) the SFINCS Fortran v3 binary
under ``mpirun -n {ranks}`` and (b) a single ``dkx`` process (cold and
warm, all cores or one GPU), and reports end-to-end seconds per configuration.
End-to-end wall time is the honest cross-code metric here: the Fortran run is
one linear RHSMode=1 solve whose cost is dominated by the preconditioner
factorization plus a handful of Krylov applications, while the dkx
auto tier solves the same system directly; internal phase timers are not
comparable one-to-one.

This is the harness behind the measured table in ``docs/performance.rst``.
The community convention for kinetic-solver comparisons is time-to-solution
tables plus throughput (coefficients/second for database scans — see
``tools/benchmarks/batched_scan.py`` for that axis), not classic speedup
curves, so that is what this tool emits.

Usage (Fortran sweep + JAX timing, JSON to stdout)::

    python tools/benchmarks/time_to_solution.py \
        --input /path/to/input.namelist \
        --fortran-binary /path/to/sfincs \
        --ranks 1 2 4 8 --reps 2

    python tools/benchmarks/time_to_solution.py \
        --input /path/to/input.namelist --jax-only   # no Fortran binary needed

The Fortran binary must be an MPI build; ``--mpirun`` selects the launcher
(default ``mpirun``; pass the one matching the toolchain the binary was
compiled with). Every run executes in a fresh scratch directory so outputs
never collide; the namelist is copied verbatim (equilibrium paths inside it
must be absolute or resolvable from the scratch directory).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SOLVE_RE = re.compile(r"Elapsed time in solve driver=\s*([0-9.]+)")
_FACTOR_RE = re.compile(r"Elapsed time in factorization driver\s*=\s*([0-9.]+)")


def run_fortran(
    binary: Path, deck: Path, ranks: int, mpirun: str, timeout_s: float
) -> dict:
    """One ``mpirun -n ranks`` Fortran run in a scratch dir; returns timings."""
    with tempfile.TemporaryDirectory(prefix="sfincs_ttl_") as tmp:
        work = Path(tmp)
        shutil.copy(deck, work / "input.namelist")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [mpirun, "-n", str(ranks), str(binary)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        total = time.perf_counter() - t0
        log = proc.stdout + proc.stderr
        solves = [float(x) for x in _SOLVE_RE.findall(log)]
        factor = _FACTOR_RE.findall(log)
        return {
            "ranks": ranks,
            "returncode": proc.returncode,
            "total_s": round(total, 1),
            "factorization_s": round(float(factor[-1]), 1) if factor else None,
            "solve_driver_s_sum": round(sum(solves), 1) if solves else None,
            "n_solve_driver_calls": len(solves),
        }


def run_jax(deck: Path) -> dict:
    """Cold (includes JIT) + warm end-to-end dkx timings on one process."""
    import jax

    from dkx.run import run_profile

    t0 = time.perf_counter()
    first = run_profile(str(deck), emit=None)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_profile(str(deck), emit=None)
    warm = time.perf_counter() - t0
    return {
        "backend": jax.default_backend(),
        "cold_s": round(cold, 1),
        "warm_s": round(warm, 1),
        "method": first.solve_result.method,
        "converged": bool(first.solve_result.converged),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="input.namelist")
    parser.add_argument("--fortran-binary", type=Path, default=None)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--reps", type=int, default=2, help="keep the best of N")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--jax-only", action="store_true")
    parser.add_argument("--fortran-only", action="store_true")
    args = parser.parse_args(argv)

    deck = args.input.resolve()
    if not deck.exists():
        raise SystemExit(f"input namelist not found: {deck}")

    report: dict = {"input": str(deck), "fortran": [], "dkx": None}

    if not args.jax_only:
        if args.fortran_binary is None or not args.fortran_binary.exists():
            raise SystemExit("--fortran-binary is required unless --jax-only")
        for ranks in args.ranks:
            best: dict | None = None
            for _ in range(args.reps):
                result = run_fortran(
                    args.fortran_binary, deck, ranks, args.mpirun, args.timeout_s
                )
                if result["returncode"] != 0:
                    best = result
                    break
                if best is None or result["total_s"] < best["total_s"]:
                    best = result
            report["fortran"].append(best)
            print(f"  fortran n={ranks}: {best}", file=sys.stderr)

    if not args.fortran_only:
        report["dkx"] = run_jax(deck)
        print(f"  dkx: {report['dkx']}", file=sys.stderr)

    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
