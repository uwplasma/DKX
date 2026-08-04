"""What does generating the coarse preconditioner's rows cost, against storing them?

:func:`dkx.solve.build_coarse_preconditioner` has two routes to the same
inverse.  The default materializes three dense ``(Ntheta*Nzeta)`` bands over
every Legendre index and factors them once, so an application is a pair of
triangular sweeps.  Where those bands exceed physical RAM -- 42.9-53.3 GB on
five upstream decks -- ``solvax.direct.block_thomas_checkpointed_fn``
eliminates the same pinned operator from an on-demand row generator and stores
no band at all, at the cost of repeating the elimination on every application.

This script measures that cost on a deck where *both* routes fit, which is the
only place the comparison is meaningful: on the decks the generated route
exists for there is nothing to compare it against.  It reports build, warm
per-application time, and how far the two routes' outputs differ, forward and
transposed.

Reproduce (from the repo root):

  python tools/benchmarks/tier2_generated_coarse.py \\
      --examples /path/to/sfincs/fortran/version3/examples \\
      --deck geometryScheme4_2species_withEr_fullTrajectories

The generated route is forced by patching the host-memory probe rather than by
shrinking the machine, so the routing under test is the production one.  Both
routes are timed under ``jax.jit`` so the number is the elimination and not
Python dispatch.  Run it on a quiet machine: these are wall times.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dkx.solve as solve_module
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input


def _time_route(op: KineticOperator, vector: jnp.ndarray, *, generated: bool, repeats: int):
    """``(build_and_first_s, warm_apply_s, forward, transposed)`` for one route."""
    real_probe = solve_module._host_memory_bytes
    if generated:
        solve_module._host_memory_bytes = lambda: 1.0
    try:
        start = time.perf_counter()
        # Both routes are jitted so the comparison is of the two eliminations
        # rather than of Python dispatch: the generated route jits its own
        # application (an application otherwise re-traces every generated block
        # row), the dense route does not need to, and leaving that asymmetry in
        # would flatter the generated one.
        precond, precond_t = (
            jax.jit(f) for f in solve_module.build_coarse_preconditioner(op)
        )
        forward = jax.block_until_ready(precond(vector))
        build_and_first = time.perf_counter() - start

        jax.block_until_ready(precond(vector))  # warm the compilation cache
        start = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(precond(vector))
        warm = (time.perf_counter() - start) / repeats
        transposed = jax.block_until_ready(precond_t(vector))
    finally:
        solve_module._host_memory_bytes = real_probe
    return build_and_first, warm, np.asarray(forward), np.asarray(transposed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--deck", default="geometryScheme4_2species_withEr_fullTrajectories")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    namelist = args.examples / args.deck / "input.namelist"
    op = KineticOperator.from_namelist(read_sfincs_input(namelist))
    bands = solve_module.coarse_preconditioner_band_bytes(op)
    if bands > (solve_module._host_memory_bytes() or float("inf")):
        print(
            f"warning: {args.deck} needs {bands / 2**30:.1f} GB of bands, which does not "
            "fit here, so the dense route below will not be a fair reference."
        )
    vector = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))

    record = {
        "deck": args.deck,
        "f_shape": list(op.f_shape),
        "band_bytes": bands,
        "generated_peak_bytes_per_subsystem": solve_module._coarse_generated_peak_bytes(op),
    }
    print(
        f"{args.deck}: Nxi={op.n_xi} Ntheta*Nzeta={op.n_theta * op.n_zeta} "
        f"subsystems={op.n_species * op.n_x}\n"
        f"  bands {bands / 2**30:.2f} GB   generated "
        f"{record['generated_peak_bytes_per_subsystem'] / 2**30:.2f} GB/subsystem",
        flush=True,
    )

    results = {}
    for label, generated in (("dense", False), ("generated", True)):
        build, warm, forward, transposed = _time_route(
            op, vector, generated=generated, repeats=args.repeats
        )
        results[label] = (forward, transposed)
        record[label] = {"build_and_first_s": round(build, 3), "warm_apply_s": round(warm, 4)}
        print(
            f"  {label:<10} build+first {build:7.3f} s   warm apply {warm:7.3f} s",
            flush=True,
        )

    for name, index in (("forward", 0), ("transposed", 1)):
        exact, cheap = results["dense"][index], results["generated"][index]
        difference = float(np.linalg.norm(exact - cheap) / np.linalg.norm(exact))
        record[f"{name}_relative_difference"] = difference
        print(f"  {name} relative difference {difference:.3e}", flush=True)

    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
