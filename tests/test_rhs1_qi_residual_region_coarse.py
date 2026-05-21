from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_qi_coarse import RHS1QICoarseBlockLayout
from sfincs_jax.rhs1_qi_residual_region_coarse import (
    RHS1QIResidualRegionCoarseConfig,
    build_rhs1_qi_residual_region_coarse_basis,
    build_rhs1_qi_residual_region_coarse_candidates,
    project_rhs1_qi_residual_region_correction,
)


def _layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12, 12),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, 0, 1),
        block_species=(0, 0, 1, 1),
    )


def _localized_trapped_residual(layout: RHS1QICoarseBlockLayout, block_index: int) -> jnp.ndarray:
    residual = jnp.zeros((layout.total_size,), dtype=jnp.float64)
    start = layout.block_offsets[block_index]
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    n_pitch = int(layout.block_sizes[block_index]) // n_angular
    pitch = np.linspace(-1.0, 1.0, n_pitch)
    trapped_pitch = np.flatnonzero(np.abs(pitch) <= 0.35)
    trapped_offsets = np.concatenate(
        [np.arange(index * n_angular, (index + 1) * n_angular) for index in trapped_pitch]
    )
    values = jnp.asarray([2.0, -1.0, 1.5, -0.5], dtype=jnp.float64)
    return residual.at[start + jnp.asarray(trapped_offsets)].set(values)


def test_residual_region_basis_reduces_synthetic_localized_bounce_mode() -> None:
    layout = _layout()
    residual = _localized_trapped_residual(layout, block_index=2)
    config = RHS1QIResidualRegionCoarseConfig(
        max_rank=4,
        min_region_energy_fraction=0.0,
        include_global_active_region=False,
        region_bands="trapped",
    )

    basis = build_rhs1_qi_residual_region_coarse_basis(layout, residual, config=config)
    projected = project_rhs1_qi_residual_region_correction(residual, basis)
    remaining = residual - projected

    assert basis.metadata.rank > 0
    assert "residual_region:block:2*bounce:trapped" in basis.metadata.accepted_labels
    assert float(jnp.linalg.norm(remaining)) < float(jnp.linalg.norm(residual)) * 1.0e-12
    np.testing.assert_allclose(projected, residual, atol=1.0e-12)


def test_residual_region_rejects_zero_and_unsupported_layouts() -> None:
    layout = _layout()
    zero_basis = build_rhs1_qi_residual_region_coarse_basis(
        layout,
        jnp.zeros((layout.total_size,), dtype=jnp.float64),
    )

    assert zero_basis.vectors.shape == (layout.total_size, 0)
    assert zero_basis.metadata.rank == 0
    assert zero_basis.metadata.candidate_count == 0

    unsupported = RHS1QICoarseBlockLayout(
        block_sizes=(5, 8),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1),
        block_species=(0, 0),
    )
    unsupported_basis = build_rhs1_qi_residual_region_coarse_basis(
        unsupported,
        jnp.ones((unsupported.total_size,), dtype=jnp.float64),
    )

    assert unsupported_basis.vectors.shape == (unsupported.total_size, 0)
    assert unsupported_basis.metadata.rank == 0
    assert unsupported_basis.metadata.candidate_count == 0


def test_residual_region_ignores_nonphysical_tail_blocks() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 5),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, -1),
        block_species=(0, 0, -1),
    )
    residual = _localized_trapped_residual(layout, block_index=1)
    residual = residual.at[-5:].set(jnp.asarray([100.0, -100.0, 50.0, -50.0, 25.0]))

    candidates, labels = build_rhs1_qi_residual_region_coarse_candidates(layout, residual)
    basis = build_rhs1_qi_residual_region_coarse_basis(layout, residual)

    assert candidates.shape[0] == layout.total_size
    assert labels
    assert basis.metadata.rank > 0
    assert not any("block:2" in label for label in basis.metadata.candidate_labels)
    np.testing.assert_allclose(candidates[-5:, :], 0.0, atol=0.0)
    np.testing.assert_allclose(basis.vectors[-5:, :], 0.0, atol=0.0)


def test_residual_region_projection_is_jit_compatible() -> None:
    layout = _layout()
    residual = _localized_trapped_residual(layout, block_index=0)
    basis = build_rhs1_qi_residual_region_coarse_basis(
        layout,
        residual,
        config=RHS1QIResidualRegionCoarseConfig(max_rank=3, min_region_energy_fraction=0.0),
    )

    def project(value: jnp.ndarray) -> jnp.ndarray:
        return project_rhs1_qi_residual_region_correction(value, basis)

    eager = project(residual)
    compiled = jax.jit(project)(residual)

    np.testing.assert_allclose(compiled, eager, atol=1.0e-12)


def test_residual_region_shape_rank_and_label_metadata_are_preserved() -> None:
    layout = _layout()
    residual = jnp.arange(1, layout.total_size + 1, dtype=jnp.float64)
    config = RHS1QIResidualRegionCoarseConfig(
        max_rank=2,
        max_candidates=6,
        min_region_energy_fraction=0.0,
    )

    basis = build_rhs1_qi_residual_region_coarse_basis(layout, residual, config=config)

    assert basis.vectors.shape == (layout.total_size, basis.metadata.rank)
    assert basis.metadata.total_size == layout.total_size
    assert basis.metadata.candidate_count == 6
    assert basis.metadata.rank == 2
    assert basis.metadata.discarded_count == basis.metadata.candidate_count - basis.metadata.rank
    assert len(basis.metadata.candidate_labels) == basis.metadata.candidate_count
    assert len(basis.metadata.accepted_labels) == basis.metadata.rank
    assert all(label.startswith("residual_region:") for label in basis.metadata.candidate_labels)
    np.testing.assert_allclose(
        basis.vectors.T @ basis.vectors,
        np.eye(basis.metadata.rank),
        atol=1.0e-12,
    )


def test_residual_region_band_selection_controls_candidate_labels() -> None:
    layout = _layout()
    residual = jnp.arange(1, layout.total_size + 1, dtype=jnp.float64)

    _, trapped_labels = build_rhs1_qi_residual_region_coarse_candidates(
        layout,
        residual,
        config=RHS1QIResidualRegionCoarseConfig(
            max_candidates=32,
            min_region_energy_fraction=0.0,
            region_bands="trapped",
        ),
    )
    _, passing_labels = build_rhs1_qi_residual_region_coarse_candidates(
        layout,
        residual,
        config=RHS1QIResidualRegionCoarseConfig(
            max_candidates=32,
            min_region_energy_fraction=0.0,
            region_bands="passing",
        ),
    )

    assert any("bounce:trapped" in label for label in trapped_labels)
    assert not any("passing" in label or "boundary" in label for label in trapped_labels)
    assert any("passing_negative" in label for label in passing_labels)
    assert any("passing_positive" in label for label in passing_labels)
    assert not any("bounce:trapped" in label or "boundary" in label for label in passing_labels)


def test_residual_region_rejects_residual_length_mismatch() -> None:
    layout = _layout()

    with pytest.raises(ValueError, match="residual length"):
        build_rhs1_qi_residual_region_coarse_basis(
            layout,
            jnp.ones((layout.total_size + 1,), dtype=jnp.float64),
        )
