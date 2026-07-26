#!/usr/bin/env python
"""Tier-2 preconditioner ladder: coarse block-Thomas vs multigrid V-cycle.

Measures what the tier-2 Krylov solve actually costs on a resolution ladder of
one full-Fokker-Planck, full-trajectory stellarator deck -- the physics that
has no block-tridiagonal-in-L structure and therefore *cannot* use the tier-1
direct solver.  For each grid it runs :func:`dkx.solve.solve` with
``preconditioner="coarse"`` (the classical exact block-Thomas factorization of
the SFINCS-simplified operator, cubic in ``Ntheta*Nzeta``) and with
``preconditioner="multigrid"`` (the same operator inverted by the semicoarsened
V-cycle of :mod:`dkx.multigrid`), and reports GCROT iterations, preconditioner
build time, solve time and peak resident memory for both.

Usage::

    DKX_EQUILIBRIA_DIRS=/path/to/equilibria \
    python tools/benchmarks/tier2_multigrid_ladder.py \
        --equilibrium wout_stellarator.nc \
        --ladder 11x21x41x5 15x31x61x6 21x41x81x7

``--only`` restricts the run to one preconditioner (useful when the coarse
route is known not to fit in memory at the largest grid), and ``--parity``
additionally checks that the two routes return the same solution.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import parse_sfincs_input_text  # noqa: E402
from dkx.solve import solve  # noqa: E402

TEMPLATE = """
&general
/
&geometryParameters
  geometryScheme = {scheme}
  inputRadialCoordinate = 3
  rN_wish = {rn}
  equilibriumFile = "{equilibrium}"
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
  dNHatdrHats = -0.5d+0
  dTHatdrHats = -2.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1d+0
  nu_n = {nu}
  Er = {er}
  collisionOperator = {collision_operator}
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = {nzeta}
  Nxi = {nxi}
  Nx = {nx}
  solverTolerance = {tol}
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
  preconditioner_species = 1
  preconditioner_x = 1
  preconditioner_xi = 0
/
&export_f
/
"""


def build_operator(args, grid):
    n_theta, n_zeta, n_xi, n_x = grid
    text = TEMPLATE.format(
        scheme=args.geometry_scheme,
        rn=args.rn,
        equilibrium=args.equilibrium,
        nu=args.nu,
        er=args.er,
        collision_operator=args.collision_operator,
        ntheta=n_theta,
        nzeta=n_zeta,
        nxi=n_xi,
        nx=n_x,
        tol=args.tol,
    )
    return kinetic_operator_from_namelist(parse_sfincs_input_text(text))


def peak_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes.
    return usage / (1e6 if sys.platform.startswith("linux") else 1e9)


def run(op, rhs, kind, args):
    t0 = time.perf_counter()
    try:
        result = solve(
            op,
            rhs,
            method="gmres",
            preconditioner=kind,
            tol=args.tol,
            max_restarts=args.max_restarts,
        )
    except Exception as exc:  # noqa: BLE001
        return {"kind": kind, "failed": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - t0
    return {
        "kind": kind,
        "iterations": result.iterations,
        "converged": bool(result.converged),
        "build_s": result.timings.get("build", float("nan")),
        "solve_s": result.timings.get("solve", float("nan")),
        "wall_s": wall,
        "residual": float(jnp.max(result.residual_norms)),
        "peak_gb": peak_gb(),
        "x": result.x,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equilibrium", required=True)
    parser.add_argument("--geometry-scheme", type=int, default=5)
    parser.add_argument("--rn", type=float, default=0.5)
    parser.add_argument("--nu", type=float, default=0.00831565)
    parser.add_argument("--er", type=float, default=-2.0)
    parser.add_argument(
        "--collision-operator",
        type=int,
        default=0,
        choices=[0, 1, 3],
        help="0 = full Fokker-Planck (default), 1 = pitch-angle scattering, 3 = improved Sugama",
    )
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-restarts", type=int, default=200)
    parser.add_argument("--ladder", nargs="+", default=["11x21x41x5", "15x31x61x6", "21x41x81x7"])
    parser.add_argument("--only", choices=["coarse", "multigrid"], default=None)
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()

    kinds = [args.only] if args.only else ["coarse", "multigrid"]
    print(f"backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"{'grid':>16s} {'unknowns':>9s} {'route':>10s} {'iters':>6s} "
          f"{'build s':>9s} {'solve s':>9s} {'wall s':>9s} {'peak GB':>8s} {'res':>9s}")
    for spec in args.ladder:
        grid = tuple(int(v) for v in spec.split("x"))
        op = build_operator(args, grid)
        rhs = op.rhs()
        results = {}
        for kind in kinds:
            out = run(op, rhs, kind, args)
            results[kind] = out
            if "failed" in out:
                print(f"{spec:>16s} {op.total_size:9d} {kind:>10s}   FAILED  {out['failed'][:70]}")
                continue
            print(f"{spec:>16s} {op.total_size:9d} {kind:>10s} {str(out['iterations']):>6s} "
                  f"{out['build_s']:9.2f} {out['solve_s']:9.2f} {out['wall_s']:9.2f} "
                  f"{out['peak_gb']:8.2f} {out['residual']:9.2e}")
        if args.parity and len(results) == 2 and all("failed" not in r for r in results.values()):
            a, b = (np.asarray(results[k]["x"]) for k in ("coarse", "multigrid"))
            rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)
            print(f"{'':>16s} {'':>9s} {'parity':>10s} relative difference {rel:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
