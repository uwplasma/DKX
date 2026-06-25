from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.operators.profile_response.compressed_layout import (
    build_rhs1_compressed_pitch_layout,
    infer_rhs1_compressed_pitch_layout_from_active_indices,
)


def _op(*, nxi_for_x: list[int] | None = None, total_tail: int = 5) -> SimpleNamespace:
    n_species = 2
    n_x = 3
    n_xi = 4
    n_theta = 2
    n_zeta = 3
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    if nxi_for_x is None:
        nxi_for_x = [4, 2, 1]
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=f_size + total_tail,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray(nxi_for_x, dtype=np.int32)),
        ),
    )


def _mask_order_active_indices(op: SimpleNamespace) -> np.ndarray:
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int64)
    l_idx = np.arange(int(op.n_xi), dtype=np.int64)[None, :]
    xl_mask = l_idx < nxi_for_x[:, None]
    f_mask = np.broadcast_to(
        xl_mask[None, :, :, None, None],
        (int(op.n_species), int(op.n_x), int(op.n_xi), int(op.n_theta), int(op.n_zeta)),
    )
    f_active = np.flatnonzero(f_mask.reshape((-1,)))
    tail = np.arange(int(op.f_size), int(op.total_size), dtype=np.int64)
    return np.concatenate([f_active, tail])


def test_compressed_pitch_layout_matches_fortran_active_order_and_tail() -> None:
    op = _op()
    layout = build_rhs1_compressed_pitch_layout(op)

    assert layout.first_index_for_x.tolist() == [0, 4, 6]
    assert layout.active_pitch_count_per_species == 7
    assert layout.kinetic_active_size_per_species == 42
    assert layout.kinetic_active_size == 84
    assert layout.tail_reduced_start == 84
    assert layout.tail_size == 5
    assert layout.reduced_size == 89
    np.testing.assert_array_equal(layout.active_full_indices, _mask_order_active_indices(op))
    np.testing.assert_array_equal(layout.tail_full_indices, np.arange(op.f_size, op.total_size))

    assert layout.full_kinetic_index(1, 1, 1, 1, 2) == (((1 * 3 + 1) * 4 + 1) * 2 + 1) * 3 + 2
    assert layout.reduced_kinetic_index(1, 1, 1, 1, 2) == 42 + (4 + 1) * 6 + 1 * 3 + 2
    assert layout.species_x_reduced_slice(1, 1) == slice(42 + 4 * 6, 42 + 6 * 6)

    reduced = layout.reduced_kinetic_index(0, 2, 0, 1, 2)
    full = layout.full_kinetic_index(0, 2, 0, 1, 2)
    assert layout.active_full_indices[reduced] == full
    assert layout.full_to_reduced_index[full] == reduced
    assert layout.full_to_reduced_index[layout.full_kinetic_index(0, 2, 1, 0, 0)] == -1
    assert layout.full_to_reduced_index[op.f_size + 3] == layout.tail_reduced_start + 3

    with pytest.raises(IndexError, match="inactive"):
        layout.reduced_kinetic_index(0, 2, 1, 0, 0)


def test_compressed_pitch_layout_defaults_to_rectangular_when_no_reduction() -> None:
    op = _op(nxi_for_x=None, total_tail=0)
    op.fblock.collisionless.n_xi_for_x = np.asarray([op.n_xi] * op.n_x, dtype=np.int32)
    layout = build_rhs1_compressed_pitch_layout(op)

    assert layout.kinetic_active_size == op.f_size
    assert layout.reduced_size == op.f_size
    np.testing.assert_array_equal(layout.active_full_indices, np.arange(op.f_size, dtype=np.int64))
    np.testing.assert_array_equal(layout.full_to_reduced_index, np.arange(op.f_size, dtype=np.int64))


def test_compressed_pitch_layout_rejects_invalid_operator_sizes() -> None:
    bad_nxi = _op(nxi_for_x=[4, 5, 1])
    with pytest.raises(ValueError, match="Nxi_for_x"):
        build_rhs1_compressed_pitch_layout(bad_nxi)

    bad_f_size = _op()
    bad_f_size.f_size += 1
    bad_f_size.total_size += 1
    with pytest.raises(ValueError, match="f_size"):
        build_rhs1_compressed_pitch_layout(bad_f_size)

    bad_total = _op()
    bad_total.total_size = bad_total.f_size - 1
    with pytest.raises(ValueError, match="total_size"):
        build_rhs1_compressed_pitch_layout(bad_total)


def test_infer_compressed_pitch_layout_from_active_indices_matches_builder() -> None:
    op = _op()
    expected = build_rhs1_compressed_pitch_layout(op)
    inferred = infer_rhs1_compressed_pitch_layout_from_active_indices(op, expected.active_full_indices)

    np.testing.assert_array_equal(inferred.nxi_for_x, expected.nxi_for_x)
    np.testing.assert_array_equal(inferred.first_index_for_x, expected.first_index_for_x)
    np.testing.assert_array_equal(inferred.active_full_indices, expected.active_full_indices)
    np.testing.assert_array_equal(inferred.full_to_reduced_index, expected.full_to_reduced_index)

    kinetic_only = infer_rhs1_compressed_pitch_layout_from_active_indices(op, expected.kinetic_active_full_indices)
    assert kinetic_only.total_size == op.f_size
    assert kinetic_only.tail_size == 0
    np.testing.assert_array_equal(kinetic_only.active_full_indices, expected.kinetic_active_full_indices)


def test_infer_compressed_pitch_layout_rejects_non_fortran_active_patterns() -> None:
    op = _op()
    expected = build_rhs1_compressed_pitch_layout(op)

    non_prefix = expected.active_full_indices.copy()
    ell0_plane = np.asarray(
        [
            expected.full_kinetic_index(species, 0, 0, theta, zeta)
            for species in range(op.n_species)
            for theta in range(op.n_theta)
            for zeta in range(op.n_zeta)
        ],
        dtype=np.int64,
    )
    non_prefix = non_prefix[~np.isin(non_prefix, ell0_plane)]
    with pytest.raises(ValueError, match="contiguous ell prefix"):
        infer_rhs1_compressed_pitch_layout_from_active_indices(op, non_prefix)

    incomplete_plane = expected.active_full_indices.copy()
    plane_point = expected.full_kinetic_index(0, 1, 1, 1, 2)
    incomplete_plane = incomplete_plane[incomplete_plane != plane_point]
    with pytest.raises(ValueError, match="complete species/theta/zeta planes"):
        infer_rhs1_compressed_pitch_layout_from_active_indices(op, incomplete_plane)

    with pytest.raises(ValueError, match="duplicates"):
        infer_rhs1_compressed_pitch_layout_from_active_indices(op, np.concatenate([expected.active_full_indices, expected.active_full_indices[:1]]))

    bad_tail = expected.active_full_indices[expected.active_full_indices != op.f_size + 2]
    with pytest.raises(ValueError, match="contiguous tail"):
        infer_rhs1_compressed_pitch_layout_from_active_indices(op, bad_tail)
