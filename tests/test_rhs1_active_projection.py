from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_setup import (
    expand_reduced_with_map,
    fp_pitch_mode_active_indices,
    project_pas_constraint_f,
    reduce_full_with_indices,
)


def test_full_reduced_index_helpers_round_trip_active_entries() -> None:
    active_idx = jnp.asarray([0, 2, 5], dtype=jnp.int32)
    full_to_active = jnp.asarray([1, 0, 2, 0, 0, 3], dtype=jnp.int32)
    v_full = jnp.asarray([10.0, -1.0, 20.0, -2.0, -3.0, 30.0])

    reduced = reduce_full_with_indices(v_full, active_idx)
    expanded = expand_reduced_with_map(reduced, full_to_active)

    assert np.allclose(np.asarray(reduced), [10.0, 20.0, 30.0])
    assert np.allclose(np.asarray(expanded), [10.0, 0.0, 20.0, 0.0, 0.0, 30.0])
    assert expanded.dtype == reduced.dtype


def test_pas_constraint_projection_removes_l0_flux_surface_average() -> None:
    # Shape is (species, x, l, theta, zeta). Only l=0 is constrained.
    f = jnp.asarray(
        np.arange(1, 1 + 1 * 3 * 2 * 2 * 2, dtype=np.float64).reshape((1, 3, 2, 2, 2))
    )
    fs_factor = jnp.ones((2, 2), dtype=jnp.float64)
    fs_sum_safe = jnp.asarray(4.0, dtype=jnp.float64)
    mask_x = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float64)

    projected = project_pas_constraint_f(
        f.reshape((-1,)),
        f_shape=(1, 3, 2, 2, 2),
        fs_factor=fs_factor,
        fs_sum_safe=fs_sum_safe,
        mask_x=mask_x,
    ).reshape((1, 3, 2, 2, 2))

    l0_average = np.asarray(jnp.einsum("tz,sxtz->sx", fs_factor, projected[:, :, 0]) / fs_sum_safe)
    assert np.isclose(l0_average[0, 0], np.asarray(f[:, 0, 0]).mean())
    assert np.allclose(l0_average[0, 1:], 0.0, atol=1.0e-14)
    assert np.allclose(np.asarray(projected[:, :, 1]), np.asarray(f[:, :, 1]))


def test_fp_pitch_mode_active_indices_selects_full_low_l_band() -> None:
    indices = fp_pitch_mode_active_indices(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([2, 1], dtype=np.int32),
        l_min=0,
        l_max=1,
    )

    np.testing.assert_array_equal(
        indices,
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15], dtype=np.int32),
    )


def test_fp_pitch_mode_active_indices_maps_to_reduced_active_order() -> None:
    full_to_active = np.zeros((16,), dtype=np.int32)
    full_to_active[[4, 6, 12]] = np.asarray([1, 2, 3], dtype=np.int32)

    l1_indices = fp_pitch_mode_active_indices(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([2, 1], dtype=np.int32),
        l_min=1,
        l_max=1,
        full_to_active=full_to_active,
    )
    low_l_indices = fp_pitch_mode_active_indices(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([2, 1], dtype=np.int32),
        l_min=0,
        l_max=1,
        full_to_active=full_to_active,
    )

    np.testing.assert_array_equal(l1_indices, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(low_l_indices, np.asarray([0, 1, 2], dtype=np.int32))
