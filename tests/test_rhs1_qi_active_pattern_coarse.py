from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.qi.basis import (
    RHS1QIActivePatternCoarseConfig,
    build_rhs1_qi_active_pattern_coarse_basis,
    build_rhs1_qi_active_pattern_coarse_candidates,
    project_rhs1_qi_active_pattern_correction,
)
from sfincs_jax.solvers.preconditioners.qi.basis import RHS1QICoarseBlockLayout


def _layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12, 12),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1, 0, 1),
        block_species=(0, 0, 1, 1),
    )


def _block_pitch_indices(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    pitch_index: int,
) -> jnp.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    start = int(layout.block_offsets[int(block_index)]) + int(pitch_index) * n_angular
    return jnp.arange(start, start + n_angular, dtype=jnp.int32)


def _block_angular_indices(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    angular_index: int,
) -> jnp.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    n_pitch = int(layout.block_sizes[int(block_index)]) // n_angular
    start = int(layout.block_offsets[int(block_index)]) + int(angular_index)
    return start + n_angular * jnp.arange(n_pitch, dtype=jnp.int32)


def _project(residual: jnp.ndarray, basis_vectors: jnp.ndarray) -> jnp.ndarray:
    return basis_vectors @ (jnp.conjugate(basis_vectors).T @ residual)


def test_active_pattern_basis_shape_rank_finite_and_deterministic() -> None:
    layout = _layout()
    residual = jnp.arange(1, layout.total_size + 1, dtype=jnp.float64)
    config = RHS1QIActivePatternCoarseConfig(
        max_rank=6,
        max_candidates=10,
        min_chunk_energy_fraction=0.0,
    )

    basis = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)
    repeated = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)

    assert basis.vectors.shape == (layout.total_size, basis.metadata.rank)
    assert 0 < basis.metadata.rank <= config.max_rank
    assert basis.metadata.candidate_count <= config.max_candidates
    assert basis.metadata.candidate_count >= basis.metadata.rank
    assert all(label.startswith("active_pattern:") for label in basis.metadata.candidate_labels)
    assert bool(jnp.all(jnp.isfinite(basis.vectors)))
    np.testing.assert_allclose(basis.vectors, repeated.vectors, atol=0.0)
    np.testing.assert_allclose(
        basis.vectors.T @ basis.vectors,
        np.eye(basis.metadata.rank),
        atol=1.0e-6,
    )


def test_active_pattern_projection_captures_high_energy_pitch_chunk() -> None:
    layout = _layout()
    residual = jnp.zeros((layout.total_size,), dtype=jnp.float64)
    chunk_indices = _block_pitch_indices(layout, block_index=2, pitch_index=1)
    residual = residual.at[chunk_indices].set(jnp.asarray([4.0, -3.0, 2.0, -1.0]))
    config = RHS1QIActivePatternCoarseConfig(
        max_rank=1,
        max_candidates=1,
        min_chunk_energy_fraction=0.0,
    )

    basis = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)
    projected = project_rhs1_qi_active_pattern_correction(residual, basis)
    remaining = residual - projected

    assert basis.metadata.rank == 1
    assert basis.metadata.accepted_labels == ("active_pattern:block:2*pitch:1",)
    assert float(jnp.linalg.norm(remaining)) < float(jnp.linalg.norm(residual)) * 1.0e-6
    np.testing.assert_allclose(projected, residual, atol=1.0e-6)


def test_active_pattern_max_rank_is_enforced() -> None:
    layout = _layout()
    residual = jnp.arange(1, layout.total_size + 1, dtype=jnp.float64)
    config = RHS1QIActivePatternCoarseConfig(
        max_rank=2,
        max_candidates=12,
        min_chunk_energy_fraction=0.0,
    )

    basis = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)

    assert basis.metadata.candidate_count > config.max_rank
    assert basis.metadata.rank == config.max_rank
    assert basis.metadata.discarded_count == basis.metadata.candidate_count - basis.metadata.rank


def test_active_pattern_fails_closed_for_unsupported_and_tail_only_residuals() -> None:
    unsupported = RHS1QICoarseBlockLayout(
        block_sizes=(5, 8),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1),
        block_species=(0, 0),
    )

    unsupported_basis = build_rhs1_qi_active_pattern_coarse_basis(
        unsupported,
        jnp.ones((unsupported.total_size,), dtype=jnp.float64),
    )

    assert unsupported_basis.vectors.shape == (unsupported.total_size, 0)
    assert unsupported_basis.metadata.rank == 0
    assert unsupported_basis.metadata.candidate_count == 0

    tail_layout = RHS1QICoarseBlockLayout(
        block_sizes=(12, 5),
        n_theta=2,
        n_zeta=2,
        block_x=(0, -1),
        block_species=(0, -1),
    )
    tail_residual = jnp.zeros((tail_layout.total_size,), dtype=jnp.float64)
    tail_residual = tail_residual.at[-5:].set(jnp.asarray([20.0, -10.0, 5.0, -2.0, 1.0]))

    tail_basis = build_rhs1_qi_active_pattern_coarse_basis(tail_layout, tail_residual)

    assert tail_basis.vectors.shape == (tail_layout.total_size, 0)
    assert tail_basis.metadata.rank == 0
    assert tail_basis.metadata.candidate_count == 0


def test_active_pattern_selected_metadata_labels_include_radial_angular_chunks() -> None:
    layout = _layout()
    residual = jnp.zeros((layout.total_size,), dtype=jnp.float64)
    angular_index = 3
    for block_index in (1, 3):
        indices = _block_angular_indices(layout, block_index, angular_index)
        values = jnp.linspace(1.0, 3.0, int(indices.shape[0]))
        residual = residual.at[indices].set(values)
    config = RHS1QIActivePatternCoarseConfig(
        max_rank=2,
        max_candidates=4,
        min_chunk_energy_fraction=0.0,
        include_block_pitch_chunks=False,
        include_block_angular_chunks=False,
        include_radial_pitch_chunks=False,
        include_block_chunks=False,
        include_species_chunks=False,
    )

    candidates, labels = build_rhs1_qi_active_pattern_coarse_candidates(
        layout,
        residual,
        config=config,
    )
    basis = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)

    assert candidates.shape == (layout.total_size, len(labels))
    assert "active_pattern:radial:1*angular:theta:1:zeta:1" in labels
    assert basis.metadata.accepted_labels[0] == "active_pattern:radial:1*angular:theta:1:zeta:1"
    assert all(label.startswith("active_pattern:radial:") for label in labels)


def test_active_pattern_projection_is_jit_compatible() -> None:
    layout = _layout()
    residual = jnp.arange(1, layout.total_size + 1, dtype=jnp.float64)
    config = RHS1QIActivePatternCoarseConfig(
        max_rank=4,
        max_candidates=8,
        min_chunk_energy_fraction=0.0,
    )
    basis = build_rhs1_qi_active_pattern_coarse_basis(layout, residual, config=config)

    def project(value: jnp.ndarray) -> jnp.ndarray:
        return project_rhs1_qi_active_pattern_correction(value, basis)

    eager = project(residual)
    compiled = jax.jit(project)(residual)
    manual = _project(residual, basis.vectors)

    np.testing.assert_allclose(compiled, eager, atol=1.0e-6)
    np.testing.assert_allclose(eager, manual, atol=1.0e-6)
