"""Physics and numerics of the pitch-collocation drift-kinetic backend.

The evidence pinned here is what a *second* discretization has to earn:

* it discretizes the same continuum operator -- the constant is exactly in its
  kernel, the collision term has the Lorentz eigenvalues, the collisionless
  streaming/mirror pair annihilates functions of the pitch-angle invariant, and
  the drive is the modal drive projected onto pitch;
* its stencils reach their formal order and its relaxation bands are the exact
  line blocks of the operator they claim to relax;
* the multigrid cycle it enables converges at a rate that does not degrade with
  resolution -- the property the Legendre-modal basis structurally cannot have;
* it agrees with the modal path in the continuum limit;
* and it stays jit-able and differentiable.

Geometry is the built-in analytic W7-X standard model (``geometryScheme = 4``),
so nothing here needs an equilibrium file.
"""

from __future__ import annotations

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from dkx.api import SolverOptions  # noqa: E402
from dkx.collocation import (  # noqa: E402
    COLLOCATION_STENCILS,
    CollocationOptions,
    _diff_matrices,
    _preconditioner,
    collocation_operator_from_namelist,
    solve_collocation,
)
from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw  # noqa: E402
from dkx.phase_space import stencil_diagonal_dominance  # noqa: E402
from dkx.run import run_profile  # noqa: E402

TEMPLATE = """
&general
/
&geometryParameters
  geometryScheme = 4
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
  dNHatdrHats = {dn}
  dTHatdrHats = {dt}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1d+0
  nu_n = {nu}
  Er = {er}
  collisionOperator = {collisions}
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
  solverTolerance = 1d-11
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
&export_f
/
"""


def deck(*, nu=8.3e-3, er=1.0e-2, nx=3, ntheta=13, nzeta=15, nxi=24, collisions=1, dn=-0.5, dt=-2.0):
    return parse_sfincs_input_text(
        TEMPLATE.format(
            nu=nu, er=er, nx=nx, ntheta=ntheta, nzeta=nzeta, nxi=nxi,
            collisions=collisions, dn=dn, dt=dt,
        )
    )


def operator(grid=(12, 8, 8), **kwargs):
    """Collocation operator on a small grid; ``kwargs`` split between deck and options."""
    deck_keys = {"nu", "er", "nx", "collisions"}
    options = CollocationOptions(
        n_alpha=grid[0],
        n_theta=grid[1],
        n_zeta=grid[2],
        **{k: v for k, v in kwargs.items() if k not in deck_keys},
    )
    return collocation_operator_from_namelist(
        deck(**{k: v for k, v in kwargs.items() if k in deck_keys}), options
    )


def dense_blocks(op, *, kinetic=False):
    """The operator as one dense matrix per speed node (it is block diagonal in x)."""
    cell = op.n_alpha * op.n_theta * op.n_zeta
    basis = jnp.eye(cell).reshape((cell, op.n_alpha, op.n_theta, op.n_zeta))
    apply = op.apply_kinetic if kinetic else op.apply
    columns = jax.vmap(lambda u: apply(jnp.broadcast_to(u, op.shape)).reshape(op.n_x, cell))(basis)
    return np.transpose(np.asarray(columns), (1, 2, 0))  # (X, row, column)


def dense_solve(op):
    """Exact projected solution, for tests that must not depend on the Krylov path."""
    blocks = dense_blocks(op)
    rhs = np.asarray(op.rhs()).reshape(op.n_x, -1)
    solution = np.linalg.solve(blocks, rhs[..., None])[..., 0].reshape(op.shape)
    return op.project(jnp.asarray(solution))


# ---------------------------------------------------------------------------
# physics: the discretized operator is the continuum operator
# ---------------------------------------------------------------------------


def test_the_constant_state_is_exactly_in_the_kernel():
    """Every stencil row sums to zero, so the discrete null space is the continuum one.

    This is what makes the rank-one closure of the module docstring legitimate:
    ``e`` is a true null vector of ``K``, not an approximate one.
    """
    op = operator()
    ones = jnp.ones(op.shape)
    scale = float(jnp.max(jnp.abs(op.diagonal())))
    assert float(jnp.max(jnp.abs(op.apply_kinetic(ones)))) < 1e-12 * scale
    # ...and the constraint functional is normalized so a constant maps to itself.
    np.testing.assert_allclose(np.asarray(op.constraint(ones)), 1.0, rtol=1e-13)
    # The shifted operator is therefore nonsingular in exactly that direction.
    np.testing.assert_allclose(
        np.asarray(op.apply(ones)),
        np.asarray(jnp.broadcast_to(op.pin_sigma[:, None, None, None], op.shape)),
        rtol=1e-10,
        atol=1e-12 * scale,
    )


@pytest.mark.parametrize("ell", (2, 3, 5))
def test_collision_operator_has_the_lorentz_eigenvalues(ell):
    """``C P_l = nu_D l(l+1)/2 P_l``: the Legendre modes diagonalize the pitch operator.

    Measured as the Rayleigh quotient in the grid's own pitch quadrature and
    normalized by the ``l = 1`` value, so ``nu_D`` cancels and what is left is
    the pure eigenvalue ratio ``l(l+1)/2``.  ``nu_D l(l+1)/2`` is exactly the
    coefficient :func:`dkx.collisions.make_pitch_angle_scattering_v3_operator`
    stores, so this pins the two collision operators to each other.
    """
    ratios = []
    for n_alpha in (16, 32, 64):
        op = operator((n_alpha, 4, 4))

        def quotient(order, op=op):
            mode = np.polynomial.legendre.legval(np.asarray(op.xi), np.eye(order + 1)[order])
            state = jnp.broadcast_to(jnp.asarray(mode)[None, :, None, None], op.shape)
            weight = op.pitch_weights * jnp.asarray(mode)
            numerator = jnp.einsum("a,xatz->x", weight, op._collide(state))
            return numerator / jnp.einsum("a,xatz->x", weight, state)

        ratios.append(float(jnp.max(jnp.abs(quotient(ell) / quotient(1) - 0.5 * ell * (ell + 1)))))
    exact = 0.5 * ell * (ell + 1)
    assert ratios[-1] / exact < 0.01, ratios
    assert ratios[1] < 0.4 * ratios[0] and ratios[2] < 0.4 * ratios[1], ratios  # second order


def test_collisionless_streaming_annihilates_the_pitch_angle_invariant():
    r"""``xi b.grad - (1-xi^2)/2 (b.grad lnB) d_xi`` kills any ``F((1-xi^2)/B)``.

    ``lambda = (1-xi^2)/B`` is the pitch-angle invariant of the collisionless,
    ``ExB``-free trajectories (Landreman et al. 2014), so the exact operator
    annihilates every function of it.  The discrete operator does so only to the
    order of its stencil; the measured rate is what this checks.
    """
    residuals = {}
    for n in (8, 16, 32):
        op = operator((2 * n, n, n), er=0.0)
        lam = (1.0 - op.xi[None, :, None, None] ** 2) / op.surface.b_hat[None, None, :, :]
        state = jnp.broadcast_to(jnp.exp(-2.0 * lam), op.shape)
        # streaming + mirror only: no ExB (Er = 0) and no collisions
        streaming = sum(op._advect(state, axis) for axis in (1, 2, 3))
        scale = float(jnp.max(jnp.abs(op.w_theta))) * float(jnp.max(jnp.abs(state)))
        residuals[n] = float(jnp.max(jnp.abs(streaming))) / scale
    # first-order upwinding: the residual halves for every halving of h
    assert residuals[32] < 0.6 * residuals[16] < 0.36 * residuals[8], residuals
    assert residuals[32] < 0.01, residuals


def test_drive_is_the_modal_drive_projected_onto_pitch():
    """``4/3 P_0 + 2/3 P_2`` in pitch is ``1 + xi^2``; the inductive drive is ``P_1``.

    Compared against :meth:`dkx.drift_kinetic.KineticOperator.rhs` on the *same*
    angular grid, so any disagreement is the drive, not the geometry.
    """
    nml = deck(ntheta=13, nzeta=15)
    modal = kinetic_operator_from_namelist(nml)
    reference = np.asarray(modal.rhs()[: modal.f_size]).reshape(modal.f_shape)[0]
    gaps = []
    for n_alpha in (16, 64):
        options = CollocationOptions(n_alpha=n_alpha, n_theta=14, n_zeta=16)
        op = collocation_operator_from_namelist(nml, options, grid=(n_alpha, 13, 15))
        moments = np.asarray(op.legendre_moments(op.rhs(), 3))
        scale = np.max(np.abs(reference[:, :3]))
        gaps.append(np.max(np.abs(moments - reference[:, :3])) / scale)
    assert gaps[-1] < 1e-3, gaps
    assert gaps[1] < 0.1 * gaps[0], gaps  # the midpoint pitch rule is second order


def test_projected_solution_solves_the_dke_up_to_a_pitch_constant_source():
    """The rank-one closure reproduces ``constraintScheme = 2`` exactly.

    ``K f = b - e s`` with ``s`` one scalar per speed node -- the particle source
    SFINCS borders the matrix with -- and ``<f> = 0``.  Both are checked here on
    the residual of the *unshifted* operator.
    """
    op = operator((12, 8, 8))
    f = dense_solve(op)
    np.testing.assert_allclose(np.asarray(op.constraint(f)), 0.0, atol=1e-14)
    per_node = np.asarray(op.apply_kinetic(f) - op.rhs()).reshape(op.n_x, -1)
    source = per_node.mean(axis=1)
    spread = np.max(np.abs(per_node - source[:, None]), axis=1)
    assert np.all(spread < 1e-6 * np.abs(source)), (spread, source)
    assert np.all(np.abs(source) > 0.0)  # the source is genuinely needed


def test_radial_transport_is_down_gradient_and_linear_in_the_drive():
    """``D11 > 0``: a density gradient alone drives flux down the gradient.

    Positive-definiteness of the diagonal transport coefficient follows from the
    positive-semidefinite collision operator (the entropy-production
    inequality), and holds at any collisionality; an upwind discretization can
    only add dissipation, so it must hold discretely too.  Sign conventions
    cancel because the test contracts the flux with the *operator's own*
    converted gradient, and the second half checks that the response is exactly
    linear in that gradient.
    """
    fluxes = {}
    for sign in (-1.0, +1.0):
        nml = deck(er=0.0, dn=sign * 0.5, dt=0.0)
        modal = kinetic_operator_from_namelist(nml)
        gradient = float(np.asarray(modal.dn_hat_dpsi_hat)[0])
        options = CollocationOptions(n_alpha=16, n_theta=8, n_zeta=16)
        op = collocation_operator_from_namelist(nml, options)
        flux = float(np.asarray(op.flux_moments(dense_solve(op)).particle_flux_vm_psi_hat)[0])
        assert flux * gradient < 0.0, (sign, flux, gradient)  # down-gradient
        fluxes[sign] = flux
    assert fluxes[+1.0] == pytest.approx(-fluxes[-1.0], rel=1e-8), fluxes


# ---------------------------------------------------------------------------
# numerics: stencils, bands, transfers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "dominance"), (("up1", 1.0), ("up3", 5.0 / 7.0), ("up4", 13.0 / 21.0))
)
def test_stencils_have_the_documented_diagonal_dominance(key, dominance):
    """``|c_0| / sum |c_j|`` is the number the two-grid factor tracks."""
    assert stencil_diagonal_dominance(*COLLOCATION_STENCILS[key]) == pytest.approx(dominance)


@pytest.mark.parametrize(("key", "order"), (("up1", 1), ("up3", 3), ("up4", 4)))
def test_stencils_reach_their_formal_order(key, order):
    """Measured convergence rate of ``d/dx`` on a smooth periodic function."""
    errors = []
    for n in (64, 128, 256):
        grid = 2.0 * math.pi * np.arange(n) / n
        forward, backward = _diff_matrices(n=n, spacing=2 * math.pi / n, stencil=key, periodic=True)
        exact = np.cos(grid)
        errors.append(
            max(
                np.max(np.abs(forward @ np.sin(grid) - exact)),
                np.max(np.abs(backward @ np.sin(grid) - exact)),
            )
        )
    rate = math.log2(errors[0] / errors[-1]) / 2.0
    assert rate == pytest.approx(order, abs=0.25), (rate, errors)


def test_every_stencil_annihilates_a_constant_under_both_closures():
    """Zero row sums are what puts the continuum null space in the discrete operator."""
    for key in COLLOCATION_STENCILS:
        for periodic in (True, False):
            for matrix in _diff_matrices(n=16, spacing=0.3, stencil=key, periodic=periodic):
                np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=1e-12)


def test_line_bands_reassemble_the_first_order_operator_exactly():
    """What the smoother inverts must be what the operator does along that line.

    With ``"up1"`` everywhere, ``K`` is exactly the sum over axes of its three
    bands plus one shared diagonal, so summing
    :meth:`~dkx.collocation.CollocationOperator.line_bands` over the axes has to
    reproduce the assembled operator to machine precision.  If it does not, the
    "exact line block" the relaxation claims to solve is a different matrix.
    """
    op = operator((8, 6, 6))
    blocks = dense_blocks(op, kinetic=True)
    probe = np.random.default_rng(0).standard_normal(op.shape)
    exact = np.einsum("xij,xj->xi", blocks, probe.reshape(op.n_x, -1)).reshape(op.shape)

    rebuilt = np.asarray(op.diagonal()) * probe
    for axis in (1, 2, 3):
        lower, _, upper = (np.asarray(band) for band in op.line_bands(axis))
        down, up = np.roll(probe, 1, axis=axis), np.roll(probe, -1, axis=axis)
        rebuilt = rebuilt + lower * down + upper * up
    np.testing.assert_allclose(rebuilt, exact, rtol=1e-10, atol=1e-12 * np.abs(exact).max())


def test_matvec_matches_the_assembled_operator():
    """The matrix-free apply is the matrix it claims to be, and it is x-block diagonal."""
    op = operator((8, 6, 6))
    blocks = dense_blocks(op)
    probe = np.random.default_rng(0).standard_normal(op.shape)
    expected = np.einsum("xij,xj->xi", blocks, probe.reshape(op.n_x, -1)).reshape(op.shape)
    np.testing.assert_allclose(np.asarray(op.apply(jnp.asarray(probe))), expected, atol=1e-11)


# ---------------------------------------------------------------------------
# the multigrid claim
# ---------------------------------------------------------------------------


def cycle_rate(op, precond, *, cycles=5, block=None):
    """Average residual reduction per cycle of the stationary iteration.

    ``block`` restricts the right-hand side to one speed node; the operator is
    block diagonal there, so the iterate stays inside that block.
    """
    rhs = np.random.default_rng(0).standard_normal(op.shape)
    if block is not None:
        rhs = rhs * (np.arange(op.n_x)[:, None, None, None] == block)
    rhs = jnp.asarray(rhs)
    step = jax.jit(lambda x: x + precond(rhs - op.apply(x)))
    x = jnp.zeros(op.shape)
    for _ in range(cycles):
        x = step(x)
    return float(jnp.linalg.norm(rhs - op.apply(x)) / jnp.linalg.norm(rhs)) ** (1.0 / cycles)


def build_cycle(grid, **kwargs):
    nml = deck(nx=2)
    options = CollocationOptions(n_alpha=grid[0], n_theta=grid[1], n_zeta=grid[2], **kwargs)
    op = collocation_operator_from_namelist(nml, options)
    precond, shapes = _preconditioner(
        op, options, lambda coarse, g: collocation_operator_from_namelist(nml, coarse, grid=tuple(g))
    )
    return op, precond, shapes


@pytest.mark.parametrize("grid", ((16, 8, 16), (32, 16, 32)))
def test_multigrid_cycle_rate_does_not_degrade_with_resolution(grid):
    """h-independence: the same cycle contracts at the same rate on a finer grid.

    This is the whole point of the pitch grid.  In the Legendre-modal basis the
    corresponding relaxation *diverges* (spectral radius 5.9e6 measured on the
    same physics), so no rate exists to compare.
    """
    op, precond, shapes = build_cycle(grid, levels=2)
    assert all(shape[0] == op.n_x for shape in shapes)  # x is never coarsened
    # Every speed node separately: the blocks never mix and they span the whole
    # range from collision-dominated to essentially collisionless, so an
    # aggregate rate would hide a divergent block behind a converging one.
    for ix in range(op.n_x):
        assert cycle_rate(op, precond, block=ix) < 0.7, ix


def test_multigrid_preconditioning_is_what_makes_the_solve_converge():
    """Same tolerance, same operator: with the cycle it converges, without it does not."""
    nml = deck(nx=2)
    base = dict(n_alpha=16, n_theta=8, n_zeta=16, levels=2)
    solver = SolverOptions(tol=1e-10, restart=20, recycle_dim=4, max_restarts=6)
    with_mg = solve_collocation(nml, CollocationOptions(**base), solver)
    without = solve_collocation(
        nml, CollocationOptions(**base, preconditioner="none"), solver
    )
    assert with_mg.converged and with_mg.residual <= 1e-10, with_mg
    assert without.residual > 10.0 * with_mg.residual, without
    assert with_mg.hierarchy and not without.hierarchy


@pytest.mark.parametrize("smoother", ("line", "plane", "upwind"))
@pytest.mark.parametrize("cycle", ("v", "f"))
def test_every_smoother_and_cycle_shape_contracts(smoother, cycle):
    op, precond, _ = build_cycle((16, 8, 16), levels=2, smoother=smoother, cycle=cycle)
    assert cycle_rate(op, precond) < 0.9


def test_widened_stencils_run_with_a_first_order_relaxation():
    """Double discretization: an accurate operator, an upwinded smoother (Brandt 1981)."""
    op, precond, _ = build_cycle((16, 8, 16), levels=2, stencil="up3", relaxation_stencil="up1")
    assert op.stencil == "up3" and op.relaxation_stencil == "up1"
    assert cycle_rate(op, precond) < 1.0


# ---------------------------------------------------------------------------
# agreement with the modal path
# ---------------------------------------------------------------------------


def test_agrees_with_the_modal_path_in_the_continuum_limit():
    """Different discretizations of one equation: the *gap* must shrink, not vanish.

    The modal reference is dkx's own :func:`dkx.run.run_profile` at a resolution
    where its own answer has stopped moving; the collocation ladder then has to
    walk towards it.
    """
    modal = [
        run_profile(sfincs_input_from_raw(deck(nx=4, ntheta=nt, nzeta=nz, nxi=nxi)), emit=None)
        for nt, nz, nxi in ((13, 15, 30), (17, 21, 40))
    ]
    fluxes = [float(np.asarray(run.moments["particleFlux_vm_psiHat"])[0]) for run in modal]
    assert abs(fluxes[1] / fluxes[0] - 1.0) < 0.05, fluxes  # the reference is converged
    reference = fluxes[-1]

    gaps = []
    for grid in ((16, 12, 12), (32, 24, 24)):
        solution = solve_collocation(
            deck(nx=4),
            CollocationOptions(n_alpha=grid[0], n_theta=grid[1], n_zeta=grid[2], levels=2),
            SolverOptions(tol=1e-10, restart=25, recycle_dim=5, max_restarts=40),
        )
        assert solution.residual <= 1e-10, solution
        flux = float(np.asarray(solution.flux_moments().particle_flux_vm_psi_hat)[0])
        gaps.append(abs(flux / reference - 1.0))
    assert gaps[1] < 0.6 * gaps[0], gaps
    assert gaps[1] < 0.25, gaps


# ---------------------------------------------------------------------------
# jit, gradients, and the option surface
# ---------------------------------------------------------------------------


def test_apply_is_jit_transparent_and_differentiable_against_finite_differences():
    """The operator is a pytree of arrays, so gradients flow through its coefficients."""
    op = operator((8, 6, 6))
    state = jnp.asarray(np.random.default_rng(0).standard_normal(op.shape))
    np.testing.assert_allclose(jax.jit(op.apply)(state), op.apply(state), rtol=1e-12)

    def objective(scale):
        scaled = replace(op, w_theta=op.w_theta * scale, collision=op.collision * scale)
        return jnp.sum(scaled.apply(state) ** 2)

    step = 1e-6
    finite = (objective(1.0 + step) - objective(1.0 - step)) / (2.0 * step)
    assert float(jax.grad(objective)(1.0)) == pytest.approx(float(finite), rel=1e-6)


def test_flux_moments_are_dkx_moments_of_the_projected_state():
    """The diagnostics are dkx's own functionals, which is what makes them comparable."""
    op = operator((16, 8, 8))
    moments = op.flux_moments(dense_solve(op))
    assert np.isfinite(np.asarray(moments.particle_flux_vm_psi_hat)).all()
    assert np.asarray(moments.fsab_flow).shape == (1,)
    # Legendre projection of a pure P_1 state recovers a unit l = 1 coefficient.
    dipole = jnp.broadcast_to(op.xi[None, :, None, None], op.shape)
    projected = np.asarray(op.legendre_moments(dipole, 3))
    np.testing.assert_allclose(projected[:, 1], 1.0, rtol=2e-2)
    np.testing.assert_allclose(projected[:, 0], 0.0, atol=1e-12)


def test_options_reject_grids_and_names_the_backend_cannot_honour():
    with pytest.raises(ValueError, match="even integer"):
        CollocationOptions(n_alpha=15)
    with pytest.raises(ValueError, match="even integer"):
        CollocationOptions(n_theta=2)
    with pytest.raises(ValueError, match="unknown stencil"):
        CollocationOptions(stencil="centered")
    with pytest.raises(ValueError, match="unknown relaxation_stencil"):
        CollocationOptions(relaxation_stencil="centered")
    with pytest.raises(ValueError, match="unknown smoother"):
        CollocationOptions(smoother="zebra")
    with pytest.raises(ValueError, match="unknown preconditioner"):
        CollocationOptions(preconditioner="ilu")
    assert CollocationOptions(n_alpha=16, n_theta=8, n_zeta=8).refined(2).grid == (32, 16, 16)


def test_unsupported_physics_is_refused_rather_than_silently_approximated():
    options = CollocationOptions(n_alpha=8, n_theta=6, n_zeta=6)
    with pytest.raises(NotImplementedError, match="collisionOperator=1"):
        collocation_operator_from_namelist(deck(collisions=0), options)
    xdot = parse_sfincs_input_text(
        TEMPLATE.format(nu=1e-2, er=1e-2, nx=3, ntheta=13, nzeta=15, nxi=8, collisions=1, dn=-0.5, dt=-2.0).replace(
            "includeXDotTerm = .false.", "includeXDotTerm = .true."
        )
    )
    with pytest.raises(NotImplementedError, match="with_er_xdot"):
        collocation_operator_from_namelist(xdot, options)
    rhs3 = parse_sfincs_input_text(
        TEMPLATE.format(nu=1e-2, er=1e-2, nx=3, ntheta=13, nzeta=15, nxi=8, collisions=1, dn=-0.5, dt=-2.0).replace(
            "&general", "&general\n  RHSMode = 3"
        )
    )
    with pytest.raises(NotImplementedError, match="RHSMode=1"):
        collocation_operator_from_namelist(rhs3, options)


def test_periodic_axes_reject_a_stencil_wider_than_the_grid():
    with pytest.raises(ValueError, match="periodic axis needs"):
        _diff_matrices(n=4, spacing=0.1, stencil="up4", periodic=True)


def test_apply_rejects_a_state_of_the_wrong_shape():
    op = operator((8, 6, 6))
    with pytest.raises(ValueError, match="must have shape"):
        op.apply(jnp.zeros((op.n_x, 3)))


def test_operator_round_trips_through_the_pytree_flatten():
    op = operator((8, 6, 6))
    leaves, treedef = jax.tree.flatten(op)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    state = jnp.ones(op.shape)
    np.testing.assert_allclose(np.asarray(rebuilt.apply(state)), np.asarray(op.apply(state)))
    assert rebuilt.shape == op.shape and rebuilt.size == op.size
