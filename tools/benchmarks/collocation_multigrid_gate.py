#!/usr/bin/env python
"""Does the pitch-collocation backend always converge, and does it scale?

Three measurements decide whether :mod:`dkx.collocation` is worth building out,
all on real geometry and the same physics the modal path solves under
``collisionOperator = 1`` with DKES-like trajectories:

``--table convergence``
    Multigrid-preconditioned GCROT to a *true* relative residual ``<= 1e-10``
    on a ladder of grid sizes, with wall time and peak resident memory.  The
    classical route (exact block-Thomas of the simplified operator) is the
    reference: it needs 86 and 149 iterations at 47k and 170k unknowns and runs
    out of memory near 488k.

``--table hindependence``
    The same iteration counts read as a function of resolution, plus the
    measured multigrid cycle rate (the average residual reduction per V-cycle
    applied as a stationary iteration).  Flat counts and a resolution-independent
    cycle rate are the property that makes the route scale; they are also what
    the Legendre-modal basis provably cannot have, because no relaxation
    converges there at all.

``--table physics``
    Refines *both* discretizations and reports the radial particle flux and
    ``FSABFlow``.  They are different discretizations, so they agree only in
    the continuum limit: what has to shrink is the gap, not the difference at
    fixed resolution.

Usage::

    DKX_EQUILIBRIA_DIRS=/path/to/equilibria \
    python tools/benchmarks/collocation_multigrid_gate.py --table all

References
----------
- M. Landreman et al., Phys. Plasmas **21**, 042503 (2014) -- the equation.
- A. Brandt, Math. Comp. **31**, 333 (1977); U. Trottenberg, C. W. Oosterlee &
  A. Schueller, *Multigrid*, Academic Press (2001) -- the cycle and the
  h-independence criterion.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dkx.api import SolverOptions  # noqa: E402
from dkx.collocation import (  # noqa: E402
    CollocationOptions,
    collocation_operator_from_namelist,
    solve_collocation,
)
from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw  # noqa: E402
from dkx.run import run_profile  # noqa: E402

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
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = {nzeta}
  Nxi = {nxi}
  Nx = {nx}
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
&export_f
/
"""

#: ``(n_alpha, n_theta, n_zeta)`` ladder, refining every coarsened axis by the
#: same factor so the mesh spacing halves from end to end.  With ``Nx = 7``
#: these are 48k, 115k, 224k, 387k and 615k unknowns, bracketing the 47k / 170k
#: / 488k sizes at which the classical route needs 86 iterations, 149
#: iterations, and more memory than the machine has.  ``Nzeta = 2 Ntheta``
#: throughout: W7-X needs roughly twice the toroidal resolution of the poloidal
#: one, and the modal reference below is converged at the same aspect ratio.
LADDER = ((24, 12, 24), (32, 16, 32), (40, 20, 40), (48, 24, 48), (56, 28, 56))

#: ``(Ntheta, Nzeta, Nxi)`` modal ladder for the physics comparison.  The modal
#: solve factors ``Nxi Nx`` dense ``(Ntheta Nzeta)`` blocks exactly, so its cost
#: is cubic in the angular product; this ladder is where that stays affordable.
MODAL_LADDER = ((11, 25, 32), (15, 31, 40), (19, 37, 48))


def namelist(args, *, ntheta=13, nzeta=15, nxi=24):
    text = TEMPLATE.format(
        scheme=args.geometry_scheme,
        rn=args.rn,
        equilibrium=args.equilibrium,
        nu=args.nu,
        er=args.er,
        ntheta=ntheta,
        nzeta=nzeta,
        nxi=nxi,
        nx=args.nx,
    )
    return parse_sfincs_input_text(text)


def peak_memory_gb() -> float:
    """Peak resident set size of this process, in GB (macOS reports bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0**3 if sys.platform == "darwin" else 1024.0**2)


def cycle_rate(operator, precond, *, cycles: int = 6, seed: int = 0) -> float:
    """Average residual reduction per V-cycle used as a stationary iteration.

    This is the multigrid convergence factor itself, stripped of the outer
    Krylov acceleration -- the number that must not grow with resolution.
    """
    rhs = jnp.asarray(np.random.default_rng(seed).standard_normal(operator.shape))
    step = jax.jit(lambda x: x + precond(rhs - operator.apply(x)))
    x = jnp.zeros(operator.shape)
    initial = float(jnp.linalg.norm(rhs))
    for _ in range(cycles):
        x = step(x)
    final = float(jnp.linalg.norm(rhs - operator.apply(x)))
    return (final / initial) ** (1.0 / cycles)


def run_grid(args, grid, *, measure_rate: bool) -> dict:
    """One collocation solve; returns the row of every table it feeds."""
    from dkx.collocation import _preconditioner

    nml = namelist(args)
    options = CollocationOptions(
        n_alpha=grid[0],
        n_theta=grid[1],
        n_zeta=grid[2],
        stencil=args.stencil,
        relaxation_stencil=args.relaxation_stencil,
        smoother=args.smoother,
        cycle=args.cycle,
        levels=args.levels,
        preconditioner=args.preconditioner,
    )
    solver = SolverOptions(tol=args.tol, restart=args.restart, recycle_dim=args.recycle, max_restarts=200)

    before = peak_memory_gb()
    start = time.perf_counter()
    solution = solve_collocation(nml, options, solver)
    elapsed = time.perf_counter() - start
    peak = peak_memory_gb()
    # A repeat call reuses XLA's compilation cache, so the difference between
    # the two is the one-time compile of the unrolled cycle and Arnoldi loop.
    warm = float("nan")
    if args.repeat:
        start = time.perf_counter()
        solve_collocation(nml, options, solver)
        warm = time.perf_counter() - start

    row = {
        "grid": list(grid),
        "unknowns": solution.operator.size,
        "iterations": solution.iterations,
        "residual": solution.residual,
        "converged": bool(solution.converged and solution.residual <= args.tol),
        "seconds": elapsed,
        "seconds_warm": warm,
        "peak_gb": peak,
        "peak_delta_gb": max(peak - before, 0.0),
        "hierarchy": [list(shape) for shape in solution.hierarchy],
    }
    if measure_rate and options.preconditioner == "multigrid":
        operator = collocation_operator_from_namelist(nml, options)
        precond, _ = _preconditioner(
            operator,
            options,
            lambda coarse, g: collocation_operator_from_namelist(nml, coarse, grid=tuple(g)),
        )
        row["cycle_rate"] = cycle_rate(operator, precond)
    moments = solution.flux_moments()
    row["particle_flux"] = float(np.asarray(moments.particle_flux_vm_psi_hat)[0])
    row["heat_flux"] = float(np.asarray(moments.heat_flux_vm_psi_hat)[0])
    row["fsab_flow"] = float(np.asarray(moments.fsab_flow)[0])
    return row


def run_modal(args, resolution) -> dict:
    """One modal reference solve at ``(Ntheta, Nzeta, Nxi)``."""
    ntheta, nzeta, nxi = resolution
    start = time.perf_counter()
    run = run_profile(
        sfincs_input_from_raw(namelist(args, ntheta=ntheta, nzeta=nzeta, nxi=nxi)), emit=None
    )
    return {
        "resolution": list(resolution),
        "unknowns": int(run.operator.total_size),
        "seconds": time.perf_counter() - start,
        "particle_flux": float(np.asarray(run.moments["particleFlux_vm_psiHat"])[0]),
        "heat_flux": float(np.asarray(run.moments["heatFlux_vm_psiHat"])[0]),
        "fsab_flow": float(np.asarray(run.moments["FSABFlow"])[0]),
    }


def table_convergence(rows) -> str:
    lines = [
        "",
        "convergence and cost (multigrid-preconditioned GCROT)",
        "  unknowns    grid            iters   true rel. residual   converged   cold (s)   warm (s)   peak RSS (GB)",
    ]
    for row in rows:
        grid = "x".join(str(n) for n in row["grid"])
        lines.append(
            f"  {row['unknowns']:>9,}  {grid:<14}  {row['iterations']:>5}   {row['residual']:>18.3e}"
            f"   {str(row['converged']):>9}   {row['seconds']:>8.1f}   "
            f"{row.get('seconds_warm', float('nan')):>8.1f}   {row['peak_gb']:>13.2f}"
        )
    return "\n".join(lines)


def table_hindependence(rows) -> str:
    lines = [
        "",
        "h-independence (same tolerance, refined grid)",
        "  unknowns    grid            levels   coarsest        iters   cycle rate   iters/decade",
    ]
    for row in rows:
        grid = "x".join(str(n) for n in row["grid"])
        hierarchy = row["hierarchy"]
        coarsest = "x".join(str(n) for n in hierarchy[-1][1:]) if hierarchy else "-"
        rate = row.get("cycle_rate")
        per_decade = row["iterations"] / max(-np.log10(max(row["residual"], 1e-300)), 1e-9)
        lines.append(
            f"  {row['unknowns']:>9,}  {grid:<14}  {len(hierarchy) - 1 if hierarchy else 0:>6}"
            f"   {coarsest:<12}  {row['iterations']:>5}   "
            f"{('%.3f' % rate) if rate is not None else '-':>10}   {per_decade:>12.1f}"
        )
    return "\n".join(lines)


def table_physics(colloc_rows, modal_rows) -> str:
    reference = modal_rows[-1]
    lines = [
        "",
        "physics agreement with the Legendre-modal path",
        f"  modal reference ({'x'.join(str(n) for n in reference['resolution'])}): "
        f"particle flux {reference['particle_flux']:+.6e}, FSABFlow {reference['fsab_flow']:+.6e}",
        "",
        "  modal ladder (self-convergence)",
        "    Ntheta x Nzeta x Nxi    particle flux        rel. gap to finest    FSABFlow           rel. gap",
    ]
    for row in modal_rows:
        label = "x".join(str(n) for n in row["resolution"])
        gap_pf = abs(row["particle_flux"] / reference["particle_flux"] - 1.0)
        gap_fl = abs(row["fsab_flow"] / reference["fsab_flow"] - 1.0)
        lines.append(
            f"    {label:<22}  {row['particle_flux']:+.8e}   {gap_pf:>18.3e}"
            f"    {row['fsab_flow']:+.8e}   {gap_fl:>9.3e}"
        )
    lines += [
        "",
        "  collocation ladder (gap to the modal reference)",
        "    Nalpha x Ntheta x Nzeta  particle flux        rel. gap              FSABFlow           rel. gap",
    ]
    for row in colloc_rows:
        label = "x".join(str(n) for n in row["grid"])
        gap_pf = abs(row["particle_flux"] / reference["particle_flux"] - 1.0)
        gap_fl = abs(row["fsab_flow"] / reference["fsab_flow"] - 1.0)
        lines.append(
            f"    {label:<22}  {row['particle_flux']:+.8e}   {gap_pf:>18.3e}"
            f"    {row['fsab_flow']:+.8e}   {gap_fl:>9.3e}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default="all", choices=("convergence", "hindependence", "physics", "all"))
    parser.add_argument("--equilibrium", default="w7x_standardConfig.bc")
    parser.add_argument("--geometry-scheme", type=int, default=11)
    parser.add_argument("--rn", type=float, default=0.5)
    parser.add_argument("--nu", type=float, default=8.3e-3)
    parser.add_argument("--er", type=float, default=1.0e-2)
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--stencil", default="up1")
    parser.add_argument("--relaxation-stencil", default="up1")
    parser.add_argument("--smoother", default="line", choices=("line", "upwind"))
    parser.add_argument("--cycle", default="v", choices=("v", "w", "f"))
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--preconditioner", default="multigrid", choices=("multigrid", "none"))
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--recycle", type=int, default=8)
    parser.add_argument("--grids", default=None, help="semicolon-separated 'a,t,z' triples")
    parser.add_argument("--modal-grids", default=None, help="semicolon-separated 'theta,zeta,xi' triples")
    parser.add_argument("--checkpoint", type=Path, default=None, help="JSON file to accumulate rows in")
    parser.add_argument("--no-repeat", dest="repeat", action="store_false",
                        help="skip the warm (compilation-cached) repeat of each solve")
    args = parser.parse_args(argv)

    def parse(spec, default):
        if spec is None:
            return default
        return tuple(tuple(int(v) for v in part.split(",")) for part in spec.split(";"))

    store = {}
    if args.checkpoint and args.checkpoint.exists():
        store = json.loads(args.checkpoint.read_text())

    def remember(key, value):
        store[key] = value
        if args.checkpoint:
            args.checkpoint.write_text(json.dumps(store, indent=1))

    def wanted(name: str) -> bool:
        return args.table in (name, "all")

    grids = parse(args.grids, LADDER)
    rows = store.get("collocation", [])
    done = {tuple(row["grid"]) for row in rows}
    for grid in grids:
        if tuple(grid) in done:
            continue
        rows.append(run_grid(args, tuple(grid), measure_rate=wanted("hindependence")))
        remember("collocation", rows)
        print(f"[collocation] {rows[-1]}", flush=True)

    if wanted("convergence"):
        print(table_convergence(store["collocation"]))
    if wanted("hindependence"):
        print(table_hindependence(store["collocation"]))
    if wanted("physics"):
        modal = store.get("modal", [])
        done = {tuple(row["resolution"]) for row in modal}
        for resolution in parse(args.modal_grids, MODAL_LADDER):
            if tuple(resolution) in done:
                continue
            modal.append(run_modal(args, tuple(resolution)))
            remember("modal", modal)
            print(f"[modal] {modal[-1]}", flush=True)
        print(table_physics(store["collocation"], modal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
