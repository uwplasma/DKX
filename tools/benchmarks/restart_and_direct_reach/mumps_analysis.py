"""MUMPS symbolic analysis (JOB=1 only) of an assembled DKX operator.

Plan section 13 step 6a. Assembles the pinned operator with
``dkx.assembly.assemble_operator``, then runs MUMPS's analysis phase -- and
never its numerical factorization -- once per ordering, reporting the
estimated factor entries, flops, and in-core and out-of-core memory.

SOLVAX 0.25.0's MUMPS adapter (``solvax.native.SpluFactorization(...,
backend="mumps")``) runs this same analysis before admitting a factorization,
but it offers no way to choose the ordering or to stop after analysis. It is
exercised once as a cross-check with a budget far below any possible estimate,
so it refuses after JOB=1 and reports its INFOG(16); the ordering sweep drives
PyMUMPS directly with the adapter's own context settings (``par=1, sym=0``,
``MPI.COMM_SELF``, centralized assembled input).

Usage::

    python mumps_analysis.py BASE.namelist OUT.json --nxi 120 --nx 16 \
        --orderings metis,pair,amd --rss-limit-gib 16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import threading
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")

_ORDERINGS = {"amd": 0, "pair": 1, "metis": 5, "auto": 7}
# INFOG(7) reports the ordering MUMPS actually used.
_ORDERING_NAMES = {0: "AMD", 1: "user", 2: "AMF", 3: "SCOTCH", 4: "PORD", 5: "METIS", 6: "QAMD"}


def _rss_bytes() -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _entries(value: int) -> int:
    """INFOG entry counts are stored negated, in millions, when they overflow."""
    return int(value) if value >= 0 else -int(value) * 1_000_000


def pair_major_order(op) -> "np.ndarray":
    """Pivot order that eliminates Legendre pairs from the highest ``l`` down.

    The state is laid out ``(species, x, l, theta, zeta)`` in C order followed
    by the bordered source unknowns. Unknowns are grouped by ``l // 2``, highest
    pair first, so the elimination walks the Legendre chain the way a
    MONKES-style block solve does; the bordered unknowns go last.
    """
    import numpy as np

    ns, nx, nl, nt, nz = op.f_shape
    s, x, l, t, z = np.meshgrid(
        np.arange(ns), np.arange(nx), np.arange(nl), np.arange(nt), np.arange(nz), indexing="ij"
    )
    key_f = np.lexsort((z.ravel(), t.ravel(), l.ravel(), x.ravel(), s.ravel(), -(l.ravel() // 2)))
    extra = np.arange(op.f_size, op.total_size)
    return np.concatenate([key_f, extra])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("out")
    parser.add_argument("--nxi", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--orderings", default="metis,pair")
    parser.add_argument("--rss-limit-gib", type=float, default=16.0)
    args = parser.parse_args()

    out = Path(args.out)
    report: dict = {
        "Nxi": args.nxi, "Nx": args.nx, "host": platform.node(),
        "cpus": sorted(os.sched_getaffinity(0)), "load_start": os.getloadavg()[0],
        "runs": [],
    }

    def save() -> None:
        report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        out.write_text(json.dumps(report, indent=1))

    limit = int(args.rss_limit_gib * 1024**3)

    def watchdog() -> None:
        while True:
            rss = _rss_bytes()
            if rss > limit:
                report["status"] = f"killed: RSS {rss / 1024**3:.2f} GiB > guard"
                save()
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()

    import numpy as np
    import solvax
    from mpi4py import MPI

    import mumps
    from dkx.assembly import assemble_operator
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    text = Path(args.base).read_text()
    for key, value in (("Nxi", args.nxi), ("Nx", args.nx)):
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^\s!]+", rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise SystemExit(f"{key} not found in {args.base}")
    deck = out.parent / f"mumps_deck_{args.nxi}_{args.nx}.namelist"
    deck.write_text(text)

    t0 = time.perf_counter()
    op = kinetic_operator_from_namelist(load_sfincs_input(deck).raw)
    t1 = time.perf_counter()
    assembled = assemble_operator(op)
    t2 = time.perf_counter()
    matrix = assembled.matrix.tocoo()
    report.update(
        solvax=solvax.__version__, f_shape=list(op.f_shape), unknowns=int(op.total_size),
        nonzeros=int(matrix.nnz), products=int(assembled.products),
        assembly_relative_error=float(assembled.relative_error),
        operator_build_s=t1 - t0, assembly_s=t2 - t1,
    )
    save()

    # Cross-check: the SOLVAX adapter's own analysis-before-factor admission.
    # A 1 MB budget cannot admit any factorization of this matrix, so the
    # adapter runs JOB=1, reads INFOG(16), and refuses before JOB=2.
    from solvax.native import SpluFactorization

    ta = time.perf_counter()
    try:
        SpluFactorization(assembled.matrix.tocsc(), backend="mumps", memory_limit_bytes=1_000_000)
        report["adapter"] = {"status": "UNEXPECTED: factorization admitted"}
    except MemoryError as refusal:
        report["adapter"] = {"status": "refused after analysis", "message": str(refusal)}
    report["adapter"]["seconds"] = time.perf_counter() - ta
    save()

    for name in [value.strip() for value in args.orderings.split(",") if value.strip()]:
        code = _ORDERINGS[name]
        run: dict = {"ordering": name, "ICNTL7": code}
        context = mumps.DMumpsContext(par=1, sym=0, comm=MPI.COMM_SELF)
        try:
            context.set_centralized_sparse(matrix)
            context.set_icntl(7, code)
            if name == "pair":
                order = pair_major_order(op)
                perm_in = np.empty(op.total_size, dtype=np.int32)
                perm_in[order] = np.arange(1, op.total_size + 1, dtype=np.int32)
                context._refs.update(perm_in=perm_in)  # keep the buffer alive
                context.id.perm_in = context.cast_array(perm_in)
            ts = time.perf_counter()
            context.run(job=1)
            run["analysis_s"] = time.perf_counter() - ts
            infog = {i: int(context.get_infog(i)) for i in (1, 2, 3, 7, 16, 17, 20, 23, 26, 27)}
            run.update(
                status="ok" if infog[1] >= 0 else f"MUMPS error INFOG(1)={infog[1]}",
                infog=infog,
                ordering_used=_ORDERING_NAMES.get(infog[7], str(infog[7])),
                factor_real_space_entries=_entries(infog[3]),
                factor_entries_estimate=_entries(infog[20]),
                flops_estimate=float(context.get_rinfog(1)),
                incore_mb_max=infog[16], incore_mb_total=infog[17],
                ooc_mb_max=infog[26], ooc_mb_total=infog[27],
                factor_gib_float64=_entries(infog[3]) * 8 / 1024**3,
            )
        except Exception as error:
            run["status"] = f"failed: {type(error).__name__}: {str(error)[:300]}"
        finally:
            context.destroy()
        report["runs"].append(run)
        save()
        print(json.dumps(run), flush=True)

    report["load_end"] = os.getloadavg()[0]
    report["status"] = "done"
    save()


if __name__ == "__main__":
    main()
