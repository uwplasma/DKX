from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.profile_setup import (
    build_rhs1_active_dof_state,
    resolve_rhs1_active_dof_mode,
)


def test_rhs1_active_dof_respects_explicit_env_overrides() -> None:
    forced_on = resolve_rhs1_active_dof_mode(
        active_dof_env="1",
        dkes_active_env="0",
        rhs_mode=1,
        include_phi1=True,
        has_reduced_modes=False,
        sparse_host_like_requested=True,
        xblock_active_dof_requested=False,
        has_pas=True,
        use_dkes=True,
    )
    forced_off = resolve_rhs1_active_dof_mode(
        active_dof_env="off",
        dkes_active_env="",
        rhs_mode=2,
        include_phi1=False,
        has_reduced_modes=True,
        sparse_host_like_requested=False,
        xblock_active_dof_requested=False,
        has_pas=False,
        use_dkes=False,
    )

    assert forced_on.use_active_dof_mode is True
    assert forced_on.reason == "env"
    assert forced_off.use_active_dof_mode is False
    assert forced_off.reason == "env"


def test_rhs1_active_dof_auto_matches_driver_sparse_and_dkes_guards() -> None:
    auto = resolve_rhs1_active_dof_mode(
        active_dof_env="",
        dkes_active_env="",
        rhs_mode=1,
        include_phi1=False,
        has_reduced_modes=True,
        sparse_host_like_requested=False,
        xblock_active_dof_requested=False,
        has_pas=True,
        use_dkes=True,
    )
    sparse_host_guard = resolve_rhs1_active_dof_mode(
        active_dof_env="",
        dkes_active_env="",
        rhs_mode=1,
        include_phi1=False,
        has_reduced_modes=True,
        sparse_host_like_requested=True,
        xblock_active_dof_requested=False,
        has_pas=True,
        use_dkes=True,
    )
    dkes_guard = resolve_rhs1_active_dof_mode(
        active_dof_env="",
        dkes_active_env="false",
        rhs_mode=1,
        include_phi1=False,
        has_reduced_modes=True,
        sparse_host_like_requested=False,
        xblock_active_dof_requested=False,
        has_pas=True,
        use_dkes=True,
    )

    assert auto.use_active_dof_mode is True
    assert auto.reason == "auto"
    assert sparse_host_guard.use_active_dof_mode is False
    assert sparse_host_guard.reason == "sparse_host"
    assert dkes_guard.use_active_dof_mode is False
    assert dkes_guard.reason == "dkes_env"


def test_rhs1_active_dof_state_builds_full_and_pas_projection_maps() -> None:
    op = SimpleNamespace(total_size=8, f_size=5)

    def active_indices(_op):
        return np.asarray([0, 2, 4, 6], dtype=np.int32)

    full_state = build_rhs1_active_dof_state(
        op=op,
        use_active_dof_mode=True,
        use_pas_projection=False,
        active_dof_indices=active_indices,
    )
    pas_state = build_rhs1_active_dof_state(
        op=op,
        use_active_dof_mode=True,
        use_pas_projection=True,
        active_dof_indices=active_indices,
    )
    disabled_state = build_rhs1_active_dof_state(
        op=op,
        use_active_dof_mode=False,
        use_pas_projection=True,
        active_dof_indices=active_indices,
    )

    assert full_state.active_size == 4
    assert full_state.active_idx_np.tolist() == [0, 2, 4, 6]
    assert np.asarray(full_state.full_to_active_jnp).tolist() == [1, 0, 2, 0, 3, 0, 4, 0]

    assert pas_state.active_size == 3
    assert pas_state.active_idx_np.tolist() == [0, 2, 4]
    assert np.asarray(pas_state.full_to_active_jnp).tolist() == [1, 0, 2, 0, 3]

    assert disabled_state.active_idx_np is None
    assert disabled_state.active_idx_jnp is None
    assert disabled_state.full_to_active_jnp is None
    assert disabled_state.active_size == 8
