from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_qi_coarse import (
    RHS1QICoarseBlockLayout,
    apply_rhs1_qi_galerkin_correction,
    apply_rhs1_qi_coarse_correction,
    build_rhs1_qi_galerkin_preconditioner,
    build_rhs1_qi_coarse_basis,
    build_rhs1_qi_coarse_candidates,
    build_rhs1_qi_xblock_hard_seed_basis,
    build_rhs1_qi_xblock_hard_seed_candidates,
    orthonormalize_rhs1_qi_coarse_basis,
)


def test_build_qi_coarse_basis_is_deterministic_and_rank_gated() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(4, 4, 4),
        n_theta=2,
        n_zeta=2,
        block_species=(0, 1, 0),
        block_x=(0, 0, 1),
    )

    candidates, labels = build_rhs1_qi_coarse_candidates(layout)
    basis = orthonormalize_rhs1_qi_coarse_basis(candidates, labels=labels)
    repeated = build_rhs1_qi_coarse_basis(
        layout,
        include_global=True,
        include_species=True,
        include_x_ramp=True,
        include_angular=True,
        include_blocks=True,
    )

    assert labels[:4] == ("global", "species:0", "species:1", "x_ramp")
    assert basis.metadata.total_size == 12
    assert basis.metadata.candidate_count == candidates.shape[1]
    assert basis.metadata.rank < basis.metadata.candidate_count
    assert basis.metadata.discarded_count > 0
    assert basis.metadata.accepted_labels[0] == "global"
    np.testing.assert_allclose(basis.vectors, repeated.vectors, atol=1.0e-14)
    np.testing.assert_allclose(basis.vectors.T @ basis.vectors, np.eye(basis.metadata.rank), atol=1.0e-13)


def test_xblock_hard_seed_basis_is_deterministic_bounded_and_rank_gated() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(45, 45, 45, 45, 45, 45),
        n_theta=5,
        n_zeta=3,
        block_species=(0, 0, 0, 1, 1, 1),
        block_x=(0, 1, 2, 0, 1, 2),
    )

    candidates, labels = build_rhs1_qi_xblock_hard_seed_candidates(
        layout,
        max_candidates=96,
        max_angular_mode=2,
    )
    repeated_candidates, repeated_labels = build_rhs1_qi_xblock_hard_seed_candidates(
        layout,
        max_candidates=96,
        max_angular_mode=2,
    )
    limited_basis = build_rhs1_qi_xblock_hard_seed_basis(
        layout,
        max_candidates=18,
        max_rank=5,
        max_angular_mode=2,
    )
    repeated_limited_basis = build_rhs1_qi_xblock_hard_seed_basis(
        layout,
        max_candidates=18,
        max_rank=5,
        max_angular_mode=2,
    )

    assert candidates.shape == (layout.total_size, len(labels))
    assert len(labels) <= 96
    assert labels == repeated_labels
    np.testing.assert_allclose(candidates, repeated_candidates, atol=0.0)
    assert labels[:14] == (
        "global",
        "species:0",
        "species:1",
        "radial:x_ramp",
        "radial:x_quad",
        "radial:species:0:x_ramp",
        "radial:species:0:x_quad",
        "radial:species:1:x_ramp",
        "radial:species:1:x_quad",
        "constraint:xi_ramp",
        "constraint:xi_quad",
        "constraint:radial_x_ramp*xi_ramp",
        "constraint:species:0:xi_ramp",
        "constraint:species:1:xi_ramp",
    )
    assert "theta_cos2" in labels
    assert "radial:x_ramp*theta_cos1" in labels
    assert "schur:x_diff:s0:0->1" in labels
    assert "schur:species_diff:x1:s0->1" in labels
    assert limited_basis.metadata.candidate_count == 18
    assert limited_basis.metadata.rank <= 5
    assert limited_basis.metadata.discarded_count >= 13
    assert limited_basis.metadata.accepted_labels == repeated_limited_basis.metadata.accepted_labels
    np.testing.assert_allclose(limited_basis.vectors, repeated_limited_basis.vectors, atol=1.0e-14)


def test_qi_coarse_correction_reduces_residual_for_block_slow_mode() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(3, 3, 3), block_x=(0, 1, 2))
    basis = build_rhs1_qi_coarse_basis(layout, include_angular=False)
    operator = jnp.diag(jnp.asarray([2.0, 2.5, 3.0, 1.5, 1.7, 1.9, 2.2, 2.4, 2.6]))
    exact = jnp.asarray([1.0, 1.0, 1.0, -0.5, -0.5, -0.5, 0.25, 0.25, 0.25])
    rhs = operator @ exact

    result = apply_rhs1_qi_coarse_correction(operator, rhs, basis=basis)

    assert result.applied
    assert result.reason == "residual_reduced"
    assert result.basis_metadata.rank == basis.metadata.rank
    assert result.residual_after_norm < 1.0e-12
    assert result.improvement_ratio < 1.0e-12
    np.testing.assert_allclose(result.solution, exact, atol=1.0e-12)


def test_xblock_hard_seed_basis_reduces_synthetic_constraint_moment_residual() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(45, 45, 45, 45, 45, 45),
        n_theta=5,
        n_zeta=3,
        block_species=(0, 0, 0, 1, 1, 1),
        block_x=(0, 1, 2, 0, 1, 2),
    )
    candidates, labels = build_rhs1_qi_xblock_hard_seed_candidates(
        layout,
        max_candidates=96,
        max_angular_mode=2,
    )
    basis = build_rhs1_qi_xblock_hard_seed_basis(
        layout,
        max_candidates=96,
        max_rank=32,
        max_angular_mode=2,
    )
    exact = 0.25 * candidates[:, labels.index("constraint:xi_ramp")]
    diagonal = jnp.linspace(1.0, 2.0, layout.total_size)
    def operator(value: jnp.ndarray) -> jnp.ndarray:
        return diagonal * value

    rhs = operator(exact)

    result = apply_rhs1_qi_coarse_correction(operator, rhs, basis=basis)

    assert "constraint:xi_ramp" in basis.metadata.accepted_labels
    assert result.applied
    assert result.reason == "residual_reduced"
    assert result.residual_after_norm < result.residual_before_norm * 1.0e-10
    np.testing.assert_allclose(result.solution, exact, atol=1.0e-10)


def test_qi_coarse_correction_supports_callable_operator_and_existing_seed() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(2, 2), block_x=(0, 1))
    basis = build_rhs1_qi_coarse_basis(layout, include_angular=False)
    matrix = jnp.asarray(
        [
            [4.0, 0.5, 0.0, 0.0],
            [0.5, 3.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.25],
            [0.0, 0.0, 0.25, 2.5],
        ]
    )
    exact = jnp.asarray([0.75, 0.75, -0.25, -0.25])
    current = jnp.asarray([0.50, 0.50, 0.0, 0.0])
    rhs = matrix @ exact

    result = apply_rhs1_qi_coarse_correction(lambda value: matrix @ value, rhs, current=current, basis=basis)

    assert result.applied
    assert result.residual_after_norm < result.residual_before_norm
    np.testing.assert_allclose(result.solution, exact, atol=1.0e-12)


def test_qi_coarse_correction_guards_empty_basis_and_zero_residual() -> None:
    candidates = jnp.zeros((4, 2))
    basis = orthonormalize_rhs1_qi_coarse_basis(candidates, labels=("zero_a", "zero_b"))
    operator = jnp.eye(4)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    empty_result = apply_rhs1_qi_coarse_correction(operator, rhs, basis=basis)

    assert not empty_result.applied
    assert empty_result.reason == "empty_basis"
    assert empty_result.basis_metadata.rank == 0
    np.testing.assert_allclose(empty_result.solution, jnp.zeros_like(rhs), atol=0.0)

    zero_result = apply_rhs1_qi_coarse_correction(operator, rhs, current=rhs, basis=basis)

    assert not zero_result.applied
    assert zero_result.reason == "zero_residual"
    assert zero_result.residual_before_norm == pytest.approx(0.0)


def test_xblock_hard_seed_basis_rejects_when_residual_is_not_reduced() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(2, 2), block_x=(0, 1))
    basis = build_rhs1_qi_xblock_hard_seed_basis(layout, max_candidates=1, max_rank=1)
    operator = jnp.eye(4)
    rhs = jnp.asarray([1.0, -1.0, 1.0, -1.0])

    result = apply_rhs1_qi_coarse_correction(operator, rhs, basis=basis)

    assert basis.metadata.candidate_labels == ("global",)
    assert basis.metadata.rank == 1
    assert not result.applied
    assert result.reason == "not_reduced"
    assert result.residual_after_norm == pytest.approx(result.residual_before_norm)
    np.testing.assert_allclose(result.solution, jnp.zeros_like(rhs), atol=0.0)


def test_qi_galerkin_preconditioner_reuses_coarse_operator_and_eliminates_coarse_residual() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(3, 3, 3), block_x=(0, 1, 2))
    basis = build_rhs1_qi_coarse_basis(layout, include_angular=False)
    q = basis.vectors
    coarse_matrix = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.5, 0.3],
            [0.1, -0.2, 1.75],
        ],
        dtype=jnp.float64,
    )[: basis.metadata.rank, : basis.metadata.rank]
    projector = q @ q.T
    operator = q @ coarse_matrix @ q.T + 5.0 * (jnp.eye(layout.total_size, dtype=jnp.float64) - projector)
    coefficients = jnp.linspace(0.25, 0.75, basis.metadata.rank, dtype=jnp.float64)
    exact = q @ coefficients
    rhs = operator @ exact

    preconditioner = build_rhs1_qi_galerkin_preconditioner(operator, basis=basis)
    result = apply_rhs1_qi_galerkin_correction(operator, rhs, preconditioner=preconditioner)
    jitted_correction = jax.jit(preconditioner.as_preconditioner())(rhs)

    assert preconditioner.metadata.rank == basis.metadata.rank
    assert preconditioner.metadata.coarse_operator_shape == (basis.metadata.rank, basis.metadata.rank)
    np.testing.assert_allclose(preconditioner.coarse_operator, q.T @ operator @ q, atol=1.0e-12)
    assert result.applied
    assert result.reason == "galerkin_residual_reduced"
    assert result.residual_after_norm < result.residual_before_norm * 1.0e-10
    np.testing.assert_allclose(result.solution, exact, atol=1.0e-10)
    np.testing.assert_allclose(jitted_correction, exact, atol=1.0e-10)


def test_qi_galerkin_preconditioner_can_wrap_base_preconditioner_multiplicatively() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(3, 3, 3), block_x=(0, 1, 2))
    basis = build_rhs1_qi_coarse_basis(layout, include_angular=False)
    q = basis.vectors
    coarse_matrix = jnp.asarray(
        [
            [2.5, 0.4, -0.2],
            [0.1, 3.0, 0.3],
            [-0.2, 0.2, 2.0],
        ],
        dtype=jnp.float64,
    )[: basis.metadata.rank, : basis.metadata.rank]
    projector = q @ q.T
    operator = q @ coarse_matrix @ q.T + 4.0 * (jnp.eye(layout.total_size, dtype=jnp.float64) - projector)
    coefficients = jnp.asarray([0.5, -0.25, 0.75], dtype=jnp.float64)[: basis.metadata.rank]
    exact = q @ coefficients
    rhs = operator @ exact
    preconditioner = build_rhs1_qi_galerkin_preconditioner(operator, basis=basis)

    def base_preconditioner(value: jnp.ndarray) -> jnp.ndarray:
        return 0.25 * jnp.asarray(value, dtype=jnp.float64)

    base_solution = base_preconditioner(rhs)
    base_residual = rhs - operator @ base_solution
    corrected_solution = base_solution + preconditioner.apply(base_residual)
    corrected_residual = rhs - operator @ corrected_solution

    assert float(jnp.linalg.norm(corrected_residual)) < float(jnp.linalg.norm(base_residual)) * 1.0e-10
    np.testing.assert_allclose(corrected_solution, exact, atol=1.0e-10)
