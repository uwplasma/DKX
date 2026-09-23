"""One point of the GCROT restart ladder, recorded even when the solve fails.

Plan section 13 step 5. Rebuilds a deck at the requested ``(Nxi, Nx)``, solves
it on the recycled Krylov route with the ``coarse`` preconditioner at the
requested restart length, and appends one JSON record: iterations, the true
relative residual recomputed from the operator, convergence, wall time, peak
RSS and the memory the restart basis costs. ``dkx.solve.solve`` is called
directly (not ``run_from_namelist``) so a rejected solve still reports its
iteration count.

A watchdog thread ends the process if its resident set exceeds
``--rss-limit-gib``, after writing a record that says so.

Usage::

    python restart_point.py BASE.namelist OUT.jsonl --nxi 20 --nx 16 \
        --restart 1000 --recycle-dim 8 --max-restarts 20 --rss-limit-gib 6

The base namelist names its equilibrium; the rewritten deck is written next to
``OUT.jsonl``. Neither is meant to be committed.
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


def _rss_bytes() -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("out")
    parser.add_argument("--nxi", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--recycle-dim", type=int, default=8)
    parser.add_argument("--max-restarts", type=int, required=True)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--rss-limit-gib", type=float, default=16.0)
    args = parser.parse_args()

    out = Path(args.out)
    record: dict = {
        "Nxi": args.nxi, "Nx": args.nx, "restart": args.restart,
        "recycle_dim": args.recycle_dim, "max_restarts": args.max_restarts,
        "budget_iterations": args.restart * args.max_restarts, "tol": args.tol,
        "host": platform.node(), "cpus": sorted(os.sched_getaffinity(0)),
        "load_start": os.getloadavg()[0],
    }

    def write(rec: dict) -> None:
        with out.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")

    limit = int(args.rss_limit_gib * 1024**3)

    def watchdog() -> None:
        while True:
            rss = _rss_bytes()
            if rss > limit:
                write({**record, "status": f"killed: RSS {rss / 1024**3:.2f} GiB > guard",
                       "peak_rss_gib": _peak_rss_bytes() / 1024**3})
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import solvax

    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input
    from dkx.solve import solve

    record.update(jax=jax.__version__, solvax=solvax.__version__)
    text = Path(args.base).read_text()
    for key, value in (("Nxi", args.nxi), ("Nx", args.nx)):
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^\s!]+", rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise SystemExit(f"{key} not found in {args.base}")
    deck = out.parent / f"restart_deck_{args.nxi}_{args.nx}.namelist"
    deck.write_text(text)

    t0 = time.perf_counter()
    op = kinetic_operator_from_namelist(load_sfincs_input(deck).raw)
    rhs = op.rhs()
    n = int(op.total_size)
    record.update(
        unknowns=n,
        # What the brief budgets (restart x n x 8 B), and what SOLVAX's FGMRES
        # cycle stores: the Arnoldi basis V (m+1, n) and the flexible
        # preconditioned basis Z (m, n), plus the (C, U) recycle pair.
        restart_basis_gib_nominal=args.restart * n * 8 / 1024**3,
        restart_basis_gib_fgmres=((2 * args.restart + 1) + 2 * args.recycle_dim) * n * 8 / 1024**3,
    )
    t_build = time.perf_counter()
    try:
        result = solve(
            op, rhs, method="iterative", tol=args.tol, atol=0.0,
            restart=args.restart, recycle_dim=args.recycle_dim,
            max_restarts=args.max_restarts, preconditioner="coarse", emit=None,
        )
        t_solve = time.perf_counter()
        x = jnp.asarray(result.x).reshape(-1)
        b = jnp.asarray(rhs).reshape(-1)
        mask = op.active_dof_mask()
        if mask is None:
            residual = b - op.apply(x)
        else:
            mask = jnp.asarray(mask, dtype=x.dtype)
            residual = b - (op.apply(mask * x) + (1.0 - mask) * x)
        b_norm = float(np.linalg.norm(np.asarray(b)))
        true_rel = float(np.linalg.norm(np.asarray(residual))) / b_norm
        reported = np.asarray(result.residual_norms, dtype=float).ravel()
        record.update(
            status="ok",
            iterations=int(result.iterations),
            converged=bool(result.converged),
            true_relative_residual=true_rel,
            solver_relative_residual=float(reported[-1]) / b_norm if reported.size else None,
            meets_tol=bool(true_rel <= args.tol),
            preconditioner_build_s=float(result.timings.get("build", float("nan"))),
            krylov_s=float(result.timings.get("solve", float("nan"))),
            solve_call_s=t_solve - t_build,
        )
    except Exception as error:  # a failed point is a result, not a crash
        record["status"] = f"failed: {type(error).__name__}: {str(error)[:300]}"
    record.update(
        operator_build_s=t_build - t0,
        wall_s=time.perf_counter() - t0,
        peak_rss_gib=_peak_rss_bytes() / 1024**3,
        load_end=os.getloadavg()[0],
    )
    write(record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
