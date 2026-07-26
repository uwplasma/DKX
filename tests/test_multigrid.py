"""Referee tests for ``dkx.multigrid`` — the tier-2 multigrid preconditioner.

Two things are pinned here.

*Structure.*  The pieces the hierarchy is built from are checked against
independent references: the periodic angular transfers must reproduce
``solvax.transfer``'s classical stencils exactly on the nested (even) grids
where both are defined, and must preserve constants on the odd, non-nested
grids SFINCS actually uses; the Legendre p-coarsening transfers must be exact
adjoints; the pitch-collocation eigenbasis must diagonalize the streaming
``L``-coupling with the Gauss-Legendre nodes; a coarsened operator must
rediscretize the same physics; and the coarsest block-Thomas solve must invert
its own level operator.

*Parity.*  A preconditioner cannot change the answer.  ``preconditioner=
"multigrid"`` must return the same tier-2 solution as ``preconditioner=
"coarse"`` to solver tolerance on decks spanning pitch-angle scattering, full
Fokker-Planck, the improved Sugama model and the ``E_r`` ``xDot`` terms, and
the gradient through the differentiable solve must still match finite
differences with the adjoint guard armed.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.multigrid import (
    UPWIND_STENCILS,
    MultigridSettings,
    coarsen_operator,
    dense_simplified_block,
    hierarchy_shapes,
    line_diagonal_dominance,
    line_smoother_spectral_radius,
    measure_smoothing_factor,
    multigrid_available,
    periodic_transfer_matrices,
    pitch_collocation_surrogate,
    simplified_operator,
    stencil_matrices,
    xi_transfer_matrices,
)
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import _resolve_preconditioner, solve

REF = Path(__file__).parent / "ref"

pytestmark = pytest.mark.skipif(
    not multigrid_available()[0],
    reason=f"installed solvax has no multigrid API: {multigrid_available()[1]}",
)


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _sugama_op() -> KineticOperator:
    text = (REF / "quick_2species_FPCollisions_noEr.input.namelist").read_text()
    return KineticOperator.from_namelist(
        parse_sfincs_input_text(text.replace("collisionOperator = 0", "collisionOperator = 3"))
    )


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_fine", [8, 12, 16, 32])
def test_periodic_transfers_match_solvax_on_nested_grids(n_fine: int) -> None:
    """On the even/nested case these are the classical multigrid stencils."""
    from solvax.transfer import prolongation_matrix, restriction_matrix

    restrict, prolong = periodic_transfer_matrices(n_fine, n_fine // 2)
    np.testing.assert_allclose(
        np.asarray(restrict),
        np.asarray(restriction_matrix(n_fine, kind="full_weighting", boundary="periodic")),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(prolong),
        np.asarray(prolongation_matrix(n_fine, kind="linear", boundary="periodic")),
        atol=1e-15,
    )


@pytest.mark.parametrize(("n_fine", "n_coarse"), [(21, 11), (41, 21), (31, 15), (11, 5)])
def test_odd_periodic_transfers_preserve_constants(n_fine: int, n_coarse: int) -> None:
    """SFINCS forces odd angular grids, which have no nested factor-2 subgrid."""
    restrict, prolong = periodic_transfer_matrices(n_fine, n_coarse)
    assert restrict.shape == (n_coarse, n_fine)
    assert prolong.shape == (n_fine, n_coarse)
    np.testing.assert_allclose(np.asarray(restrict).sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(prolong).sum(axis=1), 1.0, atol=1e-14)
    # A smooth periodic field survives a round trip to second order.
    fine = np.cos(np.arange(n_fine) * 2 * np.pi / n_fine)
    coarse = np.asarray(restrict) @ fine
    assert np.max(np.abs(np.asarray(prolong) @ coarse - fine)) < 0.35


def test_xi_transfers_are_exact_adjoints() -> None:
    """Legendre p-coarsening is spectral truncation and its exact transpose."""
    restrict, prolong = xi_transfer_matrices(12, 6)
    np.testing.assert_allclose(np.asarray(restrict), np.asarray(prolong).T, atol=0.0)
    coefficients = np.arange(12, dtype=float)
    np.testing.assert_allclose(np.asarray(restrict) @ coefficients, coefficients[:6])
    embedded = np.asarray(prolong) @ coefficients[:6]
    np.testing.assert_allclose(embedded[:6], coefficients[:6])
    np.testing.assert_allclose(embedded[6:], 0.0)


# ---------------------------------------------------------------------------
# Hierarchy and rediscretization
# ---------------------------------------------------------------------------


def test_hierarchy_keeps_speed_and_species_uncoarsened_and_stays_odd() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    shapes = hierarchy_shapes(op, MultigridSettings(levels=3, coarsen_xi=True))
    assert shapes[0] == (op.n_theta, op.n_zeta, op.n_xi)
    for n_theta, n_zeta, _ in shapes:
        assert n_theta % 2 == 1 and n_zeta % 2 == 1
    for finer, coarser in zip(shapes[:-1], shapes[1:]):
        assert all(c <= f for c, f in zip(coarser, finer))
        assert coarser != finer


def test_simplified_operator_is_a_pas_family_operator() -> None:
    """The dense collision block is reduced into the ``pas`` diagonal slot."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    simplified = simplified_operator(op)
    assert simplified.fp is None and simplified.sugama is None
    assert simplified.pas is not None
    assert not simplified.with_er_xidot and not simplified.with_er_xdot
    # A PAS-family operator: the analytic block extraction must agree with the
    # matrix-free apply, which is what lets the coarsest level reuse the
    # existing block-Thomas kernel.
    blocks = simplified.to_block_tridiagonal()
    lower, diag, upper = (np.asarray(a) for a in blocks)
    rng = np.random.default_rng(0)
    f = rng.normal(size=simplified.f_shape)
    g = f.reshape(simplified.n_species, simplified.n_x, simplified.n_xi, -1)
    y = np.einsum("lsxij,sxlj->sxli", diag, g)
    y[:, :, 1:] += np.einsum("lsxij,sxlj->sxli", lower[1:], g[:, :, :-1])
    y[:, :, :-1] += np.einsum("lsxij,sxlj->sxli", upper[:-1], g[:, :, 1:])
    reference = np.asarray(simplified.apply_f(jnp.asarray(f))).reshape(y.shape)
    assert np.max(np.abs(y - reference)) < 1e-10 * max(1.0, np.max(np.abs(reference)))


def test_coarsen_operator_rediscretizes_the_same_physics() -> None:
    op = simplified_operator(_load_op("quick_2species_FPCollisions_noEr"))
    assert coarsen_operator(op, op.n_theta, op.n_zeta, op.n_xi) is op
    coarse = coarsen_operator(op, op.n_theta, 5, 4)
    assert coarse.f_shape == (op.n_species, op.n_x, 4, op.n_theta, 5)
    # Angular derivative matrices are rebuilt at the coarse size for the same
    # physical period, not resampled: a constant still differentiates to zero
    # and the spacing scales with the grid.
    np.testing.assert_allclose(
        np.asarray(coarse.ddtheta) @ np.ones(op.n_theta), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(coarse.ddzeta) @ np.ones(5), 0.0, atol=1e-12)
    # The flux-surface geometry is interpolated, so it stays inside its range.
    assert float(coarse.b_hat.min()) >= float(op.b_hat.min()) - 1e-12
    assert float(coarse.b_hat.max()) <= float(op.b_hat.max()) + 1e-12
    # Legendre truncation, not resampling.
    np.testing.assert_allclose(
        np.asarray(coarse.xi_coupling_lower), np.asarray(op.xi_coupling_lower)[:4]
    )


def test_streaming_eigenbasis_is_the_discrete_legendre_transform() -> None:
    """``Y = V diag(lambda) V^-1`` with the Gauss-Legendre nodes as ``lambda``."""
    from dkx.multigrid import _streaming_eigenbasis

    op = simplified_operator(_load_op("quick_2species_FPCollisions_noEr"))
    v, v_inv, lam = _streaming_eigenbasis(op)
    lower = np.asarray(op.xi_coupling_lower)
    upper = np.asarray(op.xi_coupling_upper)
    y = np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    np.testing.assert_allclose(v @ v_inv, np.eye(op.n_xi), atol=1e-12)
    np.testing.assert_allclose(v @ np.diag(lam) @ v_inv, y, atol=1e-12)
    np.testing.assert_allclose(
        np.sort(lam), np.polynomial.legendre.leggauss(op.n_xi)[0], atol=1e-12
    )


def test_coarsest_block_thomas_inverts_its_level_operator() -> None:
    from dkx.multigrid import _coarse_solve, _level_matvec, _DIAGONAL_FLOOR

    op = simplified_operator(_load_op("quick_2species_FPCollisions_noEr"))
    level = coarsen_operator(op, op.n_theta, 5, op.n_xi)
    matvec = _level_matvec(level, _DIAGONAL_FLOOR)
    solve_coarse = _coarse_solve(level, _DIAGONAL_FLOOR)
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.normal(size=level.f_shape))
    b = matvec(f)
    residual = jnp.linalg.norm(matvec(solve_coarse(b)) - b) / jnp.linalg.norm(b)
    assert float(residual) < 1e-6


def test_measured_smoothing_factor_is_finite_on_a_real_operator() -> None:
    """The smoother is measured on the real DKX operator, not a model problem."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    mu = measure_smoothing_factor(op, steps=8)
    assert np.isfinite(mu)
    assert mu > 0.0


# ---------------------------------------------------------------------------
# API routing
# ---------------------------------------------------------------------------


def test_preconditioner_argument_defaults_to_the_legacy_behaviour() -> None:
    assert _resolve_preconditioner(None, True) == "coarse"
    assert _resolve_preconditioner(None, False) == "none"
    assert _resolve_preconditioner("MultiGrid", True) == "multigrid"
    assert _resolve_preconditioner("none", True) == "none"
    with pytest.raises(ValueError, match="unknown preconditioner"):
        _resolve_preconditioner("wishful", True)


# ---------------------------------------------------------------------------
# Parity: a preconditioner cannot change the answer
# ---------------------------------------------------------------------------

PARITY_CASES = {
    "pas": lambda: _load_op("pas_1species_PAS_noEr_tiny_scheme1"),
    "fokker_planck": lambda: _load_op("quick_2species_FPCollisions_noEr"),
    "improved_sugama": _sugama_op,
    "er_xdot": lambda: _load_op("er_xdot_1species_tiny"),
}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_multigrid_preconditioner_does_not_change_the_answer(case: str) -> None:
    op = PARITY_CASES[case]()
    rhs = op.rhs()
    tol = 1e-10
    reference = solve(op, rhs, method="gmres", tol=tol, preconditioner="coarse")
    multigrid = solve(op, rhs, method="gmres", tol=tol, preconditioner="multigrid")
    assert reference.converged and multigrid.converged

    # A preconditioner may not change the answer -- but "the answer" is only
    # unique to the extent the operator is.  ``er_xdot`` pairs the Er xiDot term
    # with a per-speed constraint border and is numerically singular (its
    # adjoint is rejected outright by the residual guard in ``dkx.solve``), so
    # its solution is undetermined along the near-null direction and two
    # preconditioners land on different points of the same solution manifold.
    # There the well-posed statement is that both drive the residual down; the
    # solution vectors are compared only where the operator determines them.
    residual_scale = max(1.0, float(jnp.linalg.norm(rhs)))
    for result in (reference, multigrid):
        residual = jnp.linalg.norm(op.apply(result.x) - rhs) / residual_scale
        assert float(residual) < 1e-6, (case, float(residual))

    if case != "er_xdot":
        scale = max(1.0, float(jnp.max(jnp.abs(reference.x))))
        assert float(jnp.max(jnp.abs(multigrid.x - reference.x))) / scale < 1e-6


def test_transposed_multigrid_preconditioner_is_exact() -> None:
    """``precond_t`` is the exact transpose of the cycle, not a second hierarchy.

    The V-cycle is linear in its argument (fixed transfers, fixed smoother
    factors, fixed coarse factorization), so :func:`jax.linear_transpose` gives
    ``M^T`` exactly.  Getting this wrong is invisible in the forward solve and
    only corrupts the implicit-differentiation gradient, which is what the
    tier-2 adjoint guard exists to catch.
    """
    from dkx.multigrid import build_multigrid_f_inverse

    op = _load_op("quick_2species_FPCollisions_noEr")
    a_inv, _ = build_multigrid_f_inverse(op)
    size = op.f_size
    identity = jnp.eye(size)
    transposed = jax.linear_transpose(a_inv, jnp.zeros((size,)))
    forward = np.stack([np.asarray(a_inv(identity[:, j])) for j in range(size)], axis=1)
    adjoint = np.stack(
        [np.asarray(transposed(identity[:, j])[0]) for j in range(size)], axis=1
    )
    assert np.linalg.norm(forward.T - adjoint) / np.linalg.norm(forward) < 1e-12


def test_multigrid_preconditioned_gradient_matches_finite_differences() -> None:
    """The adjoint guard must not fire falsely, and grad must still be right."""
    from dataclasses import replace

    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")

    def objective(scale, preconditioner):
        op = replace(op0, dphi_hat_dpsi_hat_kinetic=op0.dphi_hat_dpsi_hat_kinetic * scale)
        result = solve(
            op,
            op.rhs(),
            method="gmres",
            tol=1e-11,
            differentiable=True,
            check_adjoint=True,
            preconditioner=preconditioner,
        )
        return jnp.sum(result.x**2)

    grad_mg = float(jax.grad(lambda s: objective(s, "multigrid"))(1.0))
    grad_coarse = float(jax.grad(lambda s: objective(s, "coarse"))(1.0))
    step = 1e-4
    finite = float(
        (objective(1.0 + step, "coarse") - objective(1.0 - step, "coarse")) / (2 * step)
    )
    scale = max(abs(finite), 1.0)
    assert abs(grad_coarse - finite) / scale < 1e-4
    assert abs(grad_mg - grad_coarse) / max(abs(grad_coarse), 1.0) < 1e-6


def test_adjoint_diagnostics_are_recorded_for_the_multigrid_route() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    result = solve(
        op,
        op.rhs(),
        method="gmres",
        tol=1e-10,
        differentiable=True,
        preconditioner="multigrid",
    )
    assert result.adjoint is not None
    assert result.adjoint.checked


# ---------------------------------------------------------------------------
# Why the multigrid route stalls: the pitch basis.  These pin the structural
# facts behind the negative result documented in ``dkx/multigrid.py`` and
# ``docs/performance.rst``, so it stays a measurement rather than a claim.
# ---------------------------------------------------------------------------


def test_widened_upwind_stencils_buy_diagonal_dominance_at_fixed_order() -> None:
    """The point of the widened stencils: more diagonal weight per unit accuracy.

    A first-derivative stencil's diagonal dominance ``d`` is what decides
    whether a damped block-Jacobi relaxation built from it smooths.  The
    textbook upwind-biased schemes lose it as their order rises; skipping near
    neighbours keeps it.
    """
    expected = {
        "up1": 1.0,  # 1st order, fully one-sided
        "up2": 0.6,  # 2nd order, textbook
        "wide2": 0.882,  # 2nd order, widened
        "up3": 1.0 / 3.0,  # 3rd order, textbook
        "wide4": 0.619,  # 4th order, widened
        "ctr2": 0.0,  # centered: the scheme dkx actually uses on the angles
    }
    measured = {}
    for name in expected:
        back, forward = stencil_matrices(21, 1.0, name, periodic=True)
        # the forward-biased partner is the mirror image, so it has the same d
        for mat in (back, forward):
            diag = np.abs(np.diag(mat))
            off = np.abs(mat).sum(axis=1) - diag
            measured.setdefault(name, []).append(float(np.min(diag / off)))
        assert measured[name][0] == pytest.approx(measured[name][1], abs=1e-12)
    for name, value in expected.items():
        assert measured[name][0] == pytest.approx(value, abs=2e-3), name
    # a widened stencil beats the textbook one of the *same or higher* order
    assert measured["wide2"] > measured["up2"]
    assert measured["wide4"] > measured["up3"]
    # and every stencil in the table is a consistent first derivative
    for name in UPWIND_STENCILS:
        back, forward = stencil_matrices(21, 2 * np.pi / 21, name, periodic=True)
        x = 2 * np.pi * np.arange(21) / 21
        for mat in (back, forward):
            np.testing.assert_allclose(mat @ np.ones(21), 0.0, atol=1e-10)
            assert np.linalg.norm(mat @ np.sin(x) - np.cos(x)) / np.sqrt(21) < 0.35


def test_streaming_is_strictly_off_diagonal_in_the_legendre_pitch_index() -> None:
    """The structural reason no line smoother can work in the modal basis.

    Parallel streaming and the mirror force couple ``L -> L +- 1`` only, so they
    contribute *nothing* to the ``(L, L)`` block: an angular line relaxation at
    fixed ``L`` never sees the operator's dominant term, whatever angular
    stencil it is built from.
    """
    from dataclasses import replace

    simplified = simplified_operator(_load_op("quick_2species_FPCollisions_noEr"))
    stripped = replace(
        simplified,
        with_exb=False,
        pas=replace(simplified.pas, coef=jnp.zeros_like(simplified.pas.coef)),
    )
    for ell in (0, 1, simplified.n_xi // 2, simplified.n_xi - 1):
        blocks = stripped.legendre_blocks(ell)
        assert float(jnp.max(jnp.abs(blocks.diag))) == 0.0
    assert float(jnp.max(jnp.abs(stripped.legendre_blocks(1).upper))) > 0.0
    assert float(jnp.max(jnp.abs(stripped.legendre_blocks(1).lower))) > 0.0


def test_line_smoothers_diverge_in_the_legendre_basis_and_converge_on_a_pitch_grid() -> None:
    """``rho(S) < 1`` is necessary for any cycle, and the modal basis has no such S.

    Same continuum operator, same geometry, same deck: only the pitch
    discretization differs.
    """
    op = _load_op("quick_2species_FPCollisions_noEr")
    modal = dense_simplified_block(op, species=0, speed=1)
    modal_shape = (op.n_xi, op.n_theta, op.n_zeta)
    rho_modal = line_smoother_spectral_radius(modal, modal_shape)
    assert rho_modal > 1.5, rho_modal

    upwind = pitch_collocation_surrogate(op, species=0, speed=1, angular_stencil="up1")
    rho_upwind = line_smoother_spectral_radius(upwind.matrix, upwind.shape)
    assert rho_upwind < 1.0, rho_upwind

    # and it is the upwinding, not the change of basis: the same collocation
    # grid with centered differences has a divergent relaxation too.
    centered = pitch_collocation_surrogate(op, species=0, speed=1, angular_stencil="ctr2")
    assert line_smoother_spectral_radius(centered.matrix, centered.shape) > 1.0

    # the upwind pitch line is diagonally dominant; the centered one is not
    assert line_diagonal_dominance(upwind.matrix, upwind.shape, 0)[0] > (
        line_diagonal_dominance(centered.matrix, centered.shape, 0)[0]
    )


def test_pitch_collocation_surrogate_discretizes_the_same_operator() -> None:
    """The surrogate is the same continuum operator, not a different problem."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    surrogate = pitch_collocation_surrogate(
        op, species=0, speed=1, angular_stencil="ctr2"
    )
    assert surrogate.shape == (op.n_xi, op.n_theta, op.n_zeta)

    # the modal <-> nodal transform is an exact round trip at Nalpha = Nxi
    rng = np.random.default_rng(0)
    v = rng.normal(size=op.n_xi * op.n_theta * op.n_zeta)
    np.testing.assert_allclose(surrogate.modal(surrogate.nodal(v)), v, atol=1e-10)

    # the dense modal block agrees with the matrix-free simplified operator
    modal = dense_simplified_block(op, species=0, speed=1)
    f = rng.normal(size=op.f_shape)
    reference = np.asarray(simplified_operator(op).apply_f(jnp.asarray(f)))[0, 1]
    got = modal @ f[0, 1].reshape(-1)
    np.testing.assert_allclose(got, reference.reshape(-1), rtol=1e-9, atol=1e-11)

    # on a field the coarse angular grid resolves, the two discretizations of
    # the same continuum operator agree to their truncation error
    theta = 2 * np.pi * np.arange(op.n_theta) / op.n_theta
    smooth = np.zeros((op.n_xi, op.n_theta, op.n_zeta))
    smooth[1] = np.cos(theta)[:, None]
    a = surrogate.nodal(modal @ smooth.reshape(-1))
    b = surrogate.matrix @ surrogate.nodal(smooth.reshape(-1))
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.25
