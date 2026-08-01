"""Does the fill-reducing tier-2 route make the largest decks runnable?

The classical tier-2 preconditioner allocates dense ``(Ntheta*Nzeta)`` blocks
over every Legendre index, which on the biggest production decks is tens of
gigabytes (``tools/benchmarks/tier2_sparse_fill.py`` measures it).  On a 24 GB
machine the five largest upstream decks are killed mid-solve at 11-14 GB
resident.  :mod:`dkx.sparse_precond` inverts the *same* operator exactly in a
fill-reducing order, so the question this script answers is not "how much
faster" but the blunter one:

    does the same deck, on the same machine, now finish?

That is a better-posed question than a speed ratio, because it has a
machine-independent component -- the factor memory -- and because a deck that
does not run has no runtime to compare.

Each case runs in its own subprocess so that a kill is observed rather than
inherited, and under ``/usr/bin/time`` so peak RSS comes from the OS rather
than from a process reporting on itself.

Reproduce (from the repo root):

  python tools/benchmarks/tier2_sparse_vs_coarse.py \\
      --examples /path/to/sfincs/fortran/version3/examples \\
      --out sparse_vs_coarse.json

Add ``--decks`` to restrict the set; the default is the five decks measured to
fail with the classical route.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: The decks killed mid-solve with the classical preconditioner on a 24 GB
#: machine (campaign of 2026-08-01).  All five route to tier 2: Fokker-Planck
#: collisions, tangential magnetic drifts, or full trajectories.
DEFAULT_DECKS = (
    "HSX_FPCollisions_DKESTrajectories",
    "HSX_FPCollisions_fullTrajectories",
    "HSX_PASCollisions_fullTrajectories",
    "filteredW7XNetCDF_2species_magneticDrifts_noEr",
    "filteredW7XNetCDF_2species_magneticDrifts_withEr",
)

_DRIVER = """
import json, sys, time
import jax
jax.config.update("jax_enable_x64", True)
from dkx.api import SolverOptions
from dkx.inputs import load_sfincs_input
from dkx.run import run_profile, run_transport_matrix

deck, out, kind = sys.argv[1], sys.argv[2], sys.argv[3]
general = load_sfincs_input(deck).raw.group("general")
rhs_mode = int(next((v for k, v in general.items() if k.lower() == "rhsmode"), 1))
driver = run_profile if rhs_mode == 1 else run_transport_matrix
options = SolverOptions(preconditioner=None if kind == "coarse" else kind)

t0 = time.perf_counter()
run = driver(deck, solver=options, out_path=None, emit=None)
json.dump({
    "wall_s": round(time.perf_counter() - t0, 3),
    "method": str(run.solve_result.method),
    "converged": bool(run.solve_result.converged),
    "residual": [float(r) for r in run.solve_result.residual_norms],
}, open(out, "w"))
"""

_PEAK_RE = re.compile(r"(\d+)\s+maximum resident set size")


def run_one(deck: Path, kind: str, timeout_s: float, equilibria: str | None) -> dict:
    """One deck under one preconditioner, in its own subprocess."""
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        result_path = work / "result.json"
        env = dict(os.environ)
        if equilibria:
            env["DKX_EQUILIBRIA_DIRS"] = equilibria
        command = [
            "/usr/bin/time", "-l", sys.executable, "-c", _DRIVER,
            str(deck), str(result_path), kind,
        ]
        try:
            proc = subprocess.run(
                command, cwd=work, capture_output=True, text=True,
                timeout=timeout_s, env=env, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"kind": kind, "outcome": "timeout", "timeout_s": timeout_s}

        peak_gb = None
        match = _PEAK_RE.search(proc.stderr or "")
        if match:
            peak_gb = int(match.group(1)) / 2**30
        if not result_path.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:]
            return {
                "kind": kind,
                "outcome": "killed" if proc.returncode != 0 else "no output",
                "returncode": proc.returncode,
                "peak_rss_gb": peak_gb,
                "reason": tail[0] if tail else "",
            }
        record = json.loads(result_path.read_text())
        record.update({"kind": kind, "outcome": "ok", "peak_rss_gb": peak_gb})
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--decks", nargs="+", default=list(DEFAULT_DECKS))
    parser.add_argument("--kinds", nargs="+", default=["coarse", "sparse"])
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--equilibria", default=os.environ.get("DKX_EQUILIBRIA_DIRS"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    print(f"{'case':<50}{'route':>10}{'outcome':>10}{'wall [s]':>11}{'peak [GB]':>11}")
    for name in args.decks:
        deck = args.examples / name / "input.namelist"
        for kind in args.kinds:
            record = run_one(deck, kind, args.timeout_s, args.equilibria)
            record["case"] = name
            rows.append(record)
            print(
                f"{name[:49]:<50}{kind:>10}{record['outcome']:>10}"
                f"{(record.get('wall_s') or float('nan')):>11.1f}"
                f"{(record.get('peak_rss_gb') or float('nan')):>11.2f}",
                flush=True,
            )
            if args.out:
                args.out.write_text(json.dumps(rows, indent=2) + "\n")

    ok = {(r["case"], r["kind"]) for r in rows if r["outcome"] == "ok"}
    rescued = [c for c in args.decks
               if (c, "sparse") in ok and (c, "coarse") not in ok]
    print(f"\n{len(ok)} of {len(rows)} runs completed")
    if rescued:
        print("decks the fill-reducing route makes runnable: " + ", ".join(rescued))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
