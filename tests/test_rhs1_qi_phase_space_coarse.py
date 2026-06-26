from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_qi_basis import RHS1QICoarseBlockLayout
from sfincs_jax.solvers.preconditioner_qi_basis import (
    RHS1QIPhaseSpaceCoarseConfig,
    build_rhs1_qi_phase_space_coarse_basis,
    build_rhs1_qi_phase_space_coarse_candidates,
)


def _layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(10, 10, 10, 10),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, 0, 1),
        block_species=(0, 0, 1, 1),
    )


def _project_residual(residual: jnp.ndarray, basis_vectors: jnp.ndarray) -> jnp.ndarray:
    return basis_vectors @ (basis_vectors.T @ residual)


def test_phase_space_coarse_basis_shape_finite_and_deterministic() -> None:
    layout = _layout()
    config = RHS1QIPhaseSpaceCoarseConfig(max_rank=24)

    basis = build_rhs1_qi_phase_space_coarse_basis(layout, config=config)
    repeated = build_rhs1_qi_phase_space_coarse_basis(layout, config=config)

    assert basis.vectors.shape[0] == layout.total_size
    assert 0 < basis.metadata.rank <= config.max_rank
    assert basis.vectors.shape[1] == basis.metadata.rank
    assert basis.metadata.candidate_count >= basis.metadata.rank
    assert all(label.startswith("phase_space:") for label in basis.metadata.candidate_labels)
    assert all(label.startswith("phase_space:") for label in basis.metadata.accepted_labels)
    assert bool(jnp.all(jnp.isfinite(basis.vectors)))
    np.testing.assert_allclose(basis.vectors, repeated.vectors, atol=0.0)
    np.testing.assert_allclose(
        basis.vectors.T @ basis.vectors,
        np.eye(basis.metadata.rank),
        atol=1.0e-13,
    )


def test_phase_space_coarse_basis_rank_limit_is_enforced() -> None:
    layout = _layout()
    full = build_rhs1_qi_phase_space_coarse_basis(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(max_rank=24),
    )
    limited = build_rhs1_qi_phase_space_coarse_basis(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(max_rank=3),
    )

    assert full.metadata.candidate_count > 3
    assert limited.metadata.rank <= 3
    assert limited.metadata.discarded_count == limited.metadata.candidate_count - limited.metadata.rank
    assert limited.metadata.accepted_labels == full.metadata.accepted_labels[: limited.metadata.rank]


def test_phase_space_candidates_respect_disabled_groups() -> None:
    layout = _layout()
    candidates, labels = build_rhs1_qi_phase_space_coarse_candidates(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(
            include_trapped=False,
            include_passing=False,
            include_boundary=False,
            include_radial=False,
            include_species=False,
        ),
    )

    assert candidates.shape == (layout.total_size, 3)
    assert labels == (
        "phase_space:pitch:even_abs",
        "phase_space:pitch:odd",
        "phase_space:pitch:sign",
    )


def test_phase_space_projection_reduces_trapped_passing_residual_mode() -> None:
    layout = _layout()
    candidates, labels = build_rhs1_qi_phase_space_coarse_candidates(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(include_radial=False, include_species=False),
    )
    basis = build_rhs1_qi_phase_space_coarse_basis(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(include_radial=False, include_species=False),
    )
    trapped = candidates[:, labels.index("phase_space:pitch_band:trapped")]
    passing = candidates[:, labels.index("phase_space:pitch_band:passing")]
    residual = (2.0 * trapped) - (0.5 * passing)

    projected = _project_residual(residual, basis.vectors)
    residual_after = residual - projected
    jitted_projected = jax.jit(_project_residual)(residual, basis.vectors)

    assert float(jnp.linalg.norm(residual_after)) < float(jnp.linalg.norm(residual)) * 1.0e-12
    np.testing.assert_allclose(projected, residual, atol=1.0e-12)
    np.testing.assert_allclose(jitted_projected, projected, atol=1.0e-12)


def test_phase_space_coarse_basis_fails_closed_for_unsupported_block_shape() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(5, 5),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1),
        block_species=(0, 1),
    )

    basis = build_rhs1_qi_phase_space_coarse_basis(layout)

    assert basis.vectors.shape == (layout.total_size, 0)
    assert basis.metadata.rank == 0
    assert basis.metadata.candidate_count == 0


def test_phase_space_coarse_basis_ignores_explicit_nonphysical_tail_block() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(10, 10, 3),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, -1),
        block_species=(0, 0, -1),
    )

    basis = build_rhs1_qi_phase_space_coarse_basis(layout)

    assert basis.metadata.rank > 0
    np.testing.assert_allclose(basis.vectors[-3:, :], 0.0, atol=0.0)
