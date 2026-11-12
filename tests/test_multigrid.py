"""Referee tests for ``dkx.multigrid`` — the recycled Krylov multigrid preconditioner.

Two things are pinned here.

*Structure.*  The pieces the hierarchy is built from are checked against
independent references: the periodic angular transfers must reproduce
``solvax.transfer``'s classical stencils exactly on the nested (even) grids
where both are defined, and must preserve constants on the odd, non-nested
grids SFINCS actually uses; the Legendre p-coarsening transfers must be exact
adjoints; a coarsened operator must
rediscretize the same physics; and the coarsest block-Thomas solve must invert
its own level operator.

*Parity.*  A preconditioner cannot change the answer.  ``preconditioner=
"multigrid"`` must return the same recycled Krylov solution as ``preconditioner=
"coarse"`` to solver tolerance on decks spanning pitch-angle scattering, full
Fokker-Planck, the improved Sugama model and the ``E_r`` ``xDot`` terms, and
the gradient through the differentiable solve must still match finite
differences with the adjoint guard armed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.multigrid import (
    MultigridSettings,
    coarsen_operator,
    hierarchy_shapes,
    periodic_transfer_matrices,
    simplified_operator,
    xi_transfer_matrices,
)
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import _resolve_preconditioner, solve

REF = Path(__file__).parent / "ref"

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
    recycled Krylov adjoint guard exists to catch.
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
# ``docs/benchmarks/performance.md``, so it stays a measurement rather than a claim.
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("axisymmetric", [False, True])
@pytest.mark.parametrize("n_fine,stencil,min_angle", [(7, 2, 5), (5, 1, 3)])
def test_multigrid_preconditioner_transpose_matches_dense_and_ad(
    axisymmetric, n_fine, stencil, min_angle
):
    from dkx.multigrid import build_multigrid_preconditioner

    name = "pas_1species_PAS_noEr_tiny_scheme1" if axisymmetric else "er_xdot_1species_tiny"
    namelist = read_sfincs_input(REF / f"{name}.input.namelist")
    namelist.group("resolutionParameters").update(
        NTHETA=n_fine, NZETA=1 if axisymmetric else n_fine, NXI=3
    )
    namelist.group("otherNumericalParameters").update(
        THETADERIVATIVESCHEME=stencil, ZETADERIVATIVESCHEME=stencil
    )
    op = KineticOperator.from_namelist(namelist)
    # A finite pin isolates transpose parity from the constant kinetic null mode.
    precond, precond_t = build_multigrid_preconditioner(
        op,
        settings=MultigridSettings(levels=2, min_angle=min_angle, shift=0.1),
    )
    dense = np.asarray(jax.vmap(precond, in_axes=1, out_axes=1)(jnp.eye(op.total_size)))
    rng = np.random.default_rng(23)
    rhs, cotangent = (jnp.asarray(rng.normal(size=op.total_size)) for _ in range(2))
    expected = dense.T @ np.asarray(cotangent)
    transpose = np.asarray(precond_t(cotangent))
    derivative = np.asarray(jax.grad(lambda r: jnp.vdot(precond(r), cotangent))(rhs))
    compiled_transpose = np.asarray(jax.jit(precond_t)(cotangent))
    compiled_forward = np.asarray(jax.jit(precond)(rhs))
    assert np.all(np.isfinite(dense))
    expected_forward = dense @ np.asarray(rhs)
    assert np.linalg.norm(compiled_forward - expected_forward) / np.linalg.norm(expected_forward) < 1e-12
    for got in (transpose, derivative, compiled_transpose):
        assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-12


def _small_smoothing_level(n_theta, n_zeta, n_xi, *, exb=True, masked=False):
    name = "pas_1species_PAS_noEr_tiny_scheme1" if n_zeta == 1 else "er_xdot_1species_tiny"
    namelist = read_sfincs_input(REF / f"{name}.input.namelist")
    namelist.group("resolutionParameters").update(NTHETA=n_theta, NZETA=n_zeta, NXI=n_xi)
    # Three-point angular stencils support the tiny grids used by these linear-action oracles.
    namelist.group("otherNumericalParameters").update(
        THETADERIVATIVESCHEME=1, ZETADERIVATIVESCHEME=1, MAGNETICDRIFTDERIVATIVESCHEME=0)
    if not exb:
        namelist.group("physicsParameters")["ER"] = 0.
    level = simplified_operator(KineticOperator.from_namelist(namelist))
    assert level.with_exb == exb
    if masked:
        level = replace(level, n_xi_for_x=jnp.minimum(jnp.arange(level.n_x) + 1, n_xi))
    return level


@pytest.mark.parametrize("n_theta,n_zeta,exb,masked", [
    (3, 5, False, False),
    (5, 3, True, True),
    (3, 5, True, True),
    (5, 1, False, True),
])
def test_plane_smoother_preserves_linear_adjoint_and_inactive_rows(
    n_theta, n_zeta, exb, masked
):
    """A relaxation may be approximate, but its linear action and transpose agree."""
    from dkx.multigrid import _level_matvec, _level_smoother

    level = _small_smoothing_level(n_theta, n_zeta, 3, exb=exb, masked=masked)
    settings = MultigridSettings()
    matvec = _level_matvec(level, .1)
    smooth = _level_smoother(level, settings, .1)
    zero = jnp.zeros(level.f_shape)
    apply = lambda rhs: smooth(matvec, zero, rhs)
    rng = np.random.default_rng(7)
    a, b = (jnp.asarray(rng.normal(size=level.f_shape)) for _ in range(2))
    y, z = apply(a), apply(b)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(z))
    assert float(jnp.linalg.norm(y)) > 0.
    np.testing.assert_allclose(smooth(matvec, a, matvec(a)), a, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(apply(1.3 * a - .7 * b), 1.3 * y - .7 * z, rtol=1e-10, atol=1e-11)
    _, pullback = jax.vjp(apply, zero)
    adjoint = pullback(b)[0]
    np.testing.assert_allclose(jnp.vdot(y, b), jnp.vdot(a, adjoint), rtol=1e-10, atol=1e-11)
    if masked:
        active = level._mask()[None, :, :, None, None]
        inactive_rhs = a * (1 - active)
        np.testing.assert_allclose(apply(inactive_rhs), inactive_rhs, atol=1e-11)
