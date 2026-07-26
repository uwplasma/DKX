#!/usr/bin/env python
"""Why the tier-2 multigrid route stalls: a pitch-basis study.

Reproduces the measurements the negative result in :mod:`dkx.multigrid` rests
on.  Everything here works on **one ``(species, speed)`` block** of the
SFINCS-simplified operator -- the thing
:func:`dkx.solve.build_coarse_preconditioner` factors exactly -- because that
block is the whole preconditioning problem (the simplified operator is
uncoupled over ``(species, x)``), and it is small enough to attack with dense
linear algebra.

Three tables:

``--table stencils``
    Diagonal dominance ``d`` of each first-derivative stencil.  ``d`` is what
    decides whether a damped block-Jacobi relaxation built from the stencil
    smooths; the widened stencils keep it at a given formal order.

``--table smoother``
    Spectral radius of an alternating line block-Jacobi error propagator, and
    the two-grid convergence factor of a ``V(1,1)`` cycle around it, for dkx's
    Legendre-modal discretization (coarsening ``theta``/``zeta``) and for the
    same continuum operator on a pitch-angle collocation grid (coarsening
    ``alpha``/``theta``/``zeta``).  ``rho(S) > 1`` is fatal.

``--table precond``
    How good each surrogate is *as a preconditioner* for the modal operator:
    GMRES iterations for ``K x = b`` with the surrogate's exact inverse
    applied through the Legendre transform.  This is the half that upwinding
    destroys.

Usage::

    DKX_EQUILIBRIA_DIRS=/path/to/equilibria \
    python tools/benchmarks/tier2_pitch_basis_study.py \
        --equilibrium wout_stellarator.nc --table all
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import parse_sfincs_input_text  # noqa: E402
from dkx.multigrid import (  # noqa: E402
    UPWIND_STENCILS,
    dense_simplified_block,
    line_diagonal_dominance,
    line_smoother_spectral_radius,
    pitch_collocation_surrogate,
    stencil_matrices,
)

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
  collisionOperator = 0
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
  solverTolerance = 1d-8
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


def build(args, grid, nu):
    n_theta, n_zeta, n_xi, n_x = grid
    return kinetic_operator_from_namelist(
        parse_sfincs_input_text(
            TEMPLATE.format(
                scheme=args.geometry_scheme,
                rn=args.rn,
                equilibrium=args.equilibrium,
                nu=nu,
                er=args.er,
                ntheta=n_theta,
                nzeta=n_zeta,
                nxi=n_xi,
                nx=n_x,
            )
        )
    )


def periodic_linear_interp(n_fine: int, n_coarse: int) -> np.ndarray:
    coarse = 2 * np.pi * np.arange(n_coarse) / n_coarse
    fine = 2 * np.pi * np.arange(n_fine) / n_fine
    step = 2 * np.pi / n_coarse
    out = np.zeros((n_fine, n_coarse))
    for i, x in enumerate(fine):
        t = (x - coarse[0]) / step
        k = int(np.floor(t)) % n_coarse
        w = t - np.floor(t)
        out[i, k] += 1 - w
        out[i, (k + 1) % n_coarse] += w
    return out


def clamped_linear_interp(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    out = np.zeros((fine.size, coarse.size))
    for i, x in enumerate(fine):
        k = int(np.clip(np.searchsorted(coarse, x) - 1, 0, coarse.size - 2))
        w = float(np.clip((x - coarse[k]) / (coarse[k + 1] - coarse[k]), 0.0, 1.0))
        out[i, k] += 1 - w
        out[i, k + 1] += w
    return out


def pin(matrix: np.ndarray) -> np.ndarray:
    """Regularize the surface-constant null direction, as the tier-1 solve does."""
    out = matrix.copy()
    out[0, 0] += np.abs(np.diag(matrix)).mean()
    return out


def line_propagator(matrix, shape, omega=1.0, floor=1e-12):
    """Dense ``prod_axes (I - omega M_axis^-1 A)``; duplicated from the module so
    the two-grid operator below can reuse the intermediate."""
    size = matrix.shape[0]
    prop = np.eye(size)
    for axis in range(len(shape)):
        m = shape[axis]
        rest = tuple(s for i, s in enumerate(shape) if i != axis)
        n_rest = int(np.prod(rest))
        blocks = np.moveaxis(
            matrix.reshape(tuple(shape) + tuple(shape)), (axis, len(shape) + axis), (0, 1)
        ).reshape((m, m, n_rest, n_rest))
        blocks = np.einsum("abrr->rab", blocks)
        blocks = blocks + floor * np.abs(blocks).sum(axis=(1, 2)).max() * np.eye(m)[None]
        rhs = (matrix @ prop).reshape(tuple(shape) + (size,))
        rhs = np.moveaxis(rhs, axis, 0).reshape(m, n_rest, size).transpose(1, 0, 2)
        step = np.linalg.solve(blocks, rhs).transpose(1, 0, 2)
        step = np.moveaxis(step.reshape((m,) + rest + (size,)), 0, axis)
        prop = prop - omega * step.reshape(size, size)
    return prop


def two_grid_factor(fine, coarse, prolong, shape, omega=1.0):
    size = fine.shape[0]
    restrict = prolong.T / prolong.sum(axis=0)[:, None]
    correction = np.eye(size) - prolong @ np.linalg.solve(coarse, restrict @ fine)
    smoother = line_propagator(fine, shape, omega)
    return float(np.abs(np.linalg.eigvals(smoother @ correction @ smoother)).max())


def gmres_iterations(matvec, b, precond=None, tol=1e-8, maxiter=400):
    apply_m = precond if precond is not None else (lambda v: v)
    n = b.size
    r0 = b - matvec(np.zeros(n))
    beta = float(np.linalg.norm(r0))
    basis = np.zeros((n, maxiter + 1))
    hess = np.zeros((maxiter + 1, maxiter))
    basis[:, 0] = r0 / beta
    for k in range(maxiter):
        w = matvec(apply_m(basis[:, k]))
        for i in range(k + 1):
            hess[i, k] = basis[:, i] @ w
            w = w - hess[i, k] * basis[:, i]
        hess[k + 1, k] = np.linalg.norm(w)
        if hess[k + 1, k] > 1e-14:
            basis[:, k + 1] = w / hess[k + 1, k]
        rhs = np.zeros(k + 2)
        rhs[0] = beta
        y, *_ = np.linalg.lstsq(hess[: k + 2, : k + 1], rhs, rcond=None)
        rel = np.linalg.norm(hess[: k + 2, : k + 1] @ y - rhs) / beta
        if rel < tol or hess[k + 1, k] <= 1e-14:
            return k + 1
    return maxiter


def table_stencils() -> None:
    print("first-derivative stencils (offsets are backward-biased, in units of h)")
    print(f"{'name':>8s} {'offsets':>22s} {'order':>6s} {'diag dominance d':>17s}")
    x = 2 * np.pi * np.arange(41) / 41
    for name, offsets in UPWIND_STENCILS.items():
        mat, _ = stencil_matrices(41, 2 * np.pi / 41, name, periodic=True)
        diag = np.abs(np.diag(mat))
        off = np.abs(mat).sum(axis=1) - diag
        d = float(np.min(diag / np.maximum(off, 1e-300)))
        err = np.linalg.norm(mat @ np.sin(x) - np.cos(x)) / np.sqrt(41)
        order = len(offsets) - 1
        print(f"{name:>8s} {str(list(offsets)):>22s} {order:6d} {d:17.3f}   "
              f"(rms error on sin: {err:.2e})")


def table_smoother(args) -> None:
    grid = args.grid
    coarse_grid = args.coarse
    print(f"alternating line block-Jacobi, omega={args.omega}; "
          f"fine {grid[0]}x{grid[1]}x{grid[2]} -> coarse {coarse_grid[0]}x{coarse_grid[1]}")
    print(f"{'nu_n':>9s} {'discretization':>28s} {'rho(S)':>12s} {'rho(TG)':>12s}")
    for nu in args.nu:
        op = build(args, (grid[0], grid[1], grid[2], args.nx), nu)
        op_c = build(args, (coarse_grid[0], coarse_grid[1], grid[2], args.nx), nu)
        modal = pin(dense_simplified_block(op, species=0, speed=args.speed))
        modal_c = pin(dense_simplified_block(op_c, species=0, speed=args.speed))
        shape = (op.n_xi, op.n_theta, op.n_zeta)
        p_theta = periodic_linear_interp(op.n_theta, coarse_grid[0])
        p_zeta = periodic_linear_interp(op.n_zeta, coarse_grid[1])
        prolong = np.kron(np.eye(op.n_xi), np.kron(p_theta, p_zeta))
        print(f"{nu:9.2e} {'legendre-modal, (t,z)':>28s} "
              f"{line_smoother_spectral_radius(modal, shape, omega=args.omega):12.4g} "
              f"{two_grid_factor(modal, modal_c, prolong, shape, args.omega):12.4g}")

        op_cc = build(args, (coarse_grid[0], coarse_grid[1], coarse_grid[2], args.nx), nu)
        for name in args.stencils:
            fine = pitch_collocation_surrogate(op, speed=args.speed, angular_stencil=name)
            coarse = pitch_collocation_surrogate(
                op_cc, speed=args.speed, angular_stencil=name
            )
            p_alpha = clamped_linear_interp(fine.alpha, coarse.alpha)
            prolong_c = np.kron(p_alpha, np.kron(p_theta, p_zeta))
            a_f, a_c = pin(fine.matrix), pin(coarse.matrix)
            d_pitch = line_diagonal_dominance(a_f, fine.shape, 0)[0]
            print(f"{nu:9.2e} {f'collocation {name}, (a,t,z)':>28s} "
                  f"{line_smoother_spectral_radius(a_f, fine.shape, omega=args.omega):12.4g} "
                  f"{two_grid_factor(a_f, a_c, prolong_c, fine.shape, args.omega):12.4g}"
                  f"   (pitch-line d = {d_pitch:.3f})")
        print()


def table_precond(args) -> None:
    import scipy.linalg as sla

    grid = args.grid
    print("GMRES on the Legendre-modal simplified operator, preconditioned by the")
    print("exact inverse of each surrogate applied through the Legendre transform")
    print(f"{'nu_n':>9s} {'surrogate (angles/pitch)':>32s} {'GMRES':>7s}")
    for nu in args.nu:
        op = build(args, (grid[0], grid[1], grid[2], args.nx), nu)
        modal = dense_simplified_block(op, species=0, speed=args.speed)
        rng = np.random.default_rng(1)
        rhs = rng.standard_normal(modal.shape[0])
        print(f"{nu:9.2e} {'(none)':>32s} "
              f"{gmres_iterations(lambda v: modal @ v, rhs):7d}")
        for angular in args.stencils:
            for pitch in dict.fromkeys((angular, "up1")):
                sur = pitch_collocation_surrogate(
                    op, speed=args.speed, angular_stencil=angular, pitch_stencil=pitch
                )
                lu = sla.lu_factor(sur.matrix)

                def apply_m(r, sur=sur, lu=lu):
                    return sur.modal(sla.lu_solve(lu, sur.nodal(r)))

                its = gmres_iterations(lambda v: modal @ v, rhs, precond=apply_m)
                print(f"{'':>9s} {f'{angular} / {pitch}':>32s} {its:7d}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equilibrium", default="wout_w7x_standardConfig.nc")
    parser.add_argument("--geometry-scheme", type=int, default=5)
    parser.add_argument("--rn", type=float, default=0.5)
    parser.add_argument("--er", type=float, default=-2.0)
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--grid", type=int, nargs=3, default=[9, 11, 13],
                        metavar=("NTHETA", "NZETA", "NXI"))
    parser.add_argument("--coarse", type=int, nargs=3, default=[5, 5, 7],
                        metavar=("NTHETA", "NZETA", "NALPHA"))
    parser.add_argument("--nu", type=float, nargs="+",
                        default=[1e-1, 8.31565e-3, 1e-4])
    parser.add_argument("--stencils", nargs="+",
                        default=["up1", "wide2", "wide4", "up3", "ctr2"])
    parser.add_argument("--table", choices=["stencils", "smoother", "precond", "all"],
                        default="all")
    args = parser.parse_args()

    if args.table in ("stencils", "all"):
        table_stencils()
        print()
    if args.table in ("smoother", "all"):
        table_smoother(args)
    if args.table in ("precond", "all"):
        table_precond(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
