from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.transport_solve_policy import (
    build_transport_active_dof_state,
    resolve_transport_active_dof_mode,
    resolve_transport_dense_policy,
    transport_geometry5_mono_low_memory_preferred,
)


def _op(*, rhs_mode: int = 2, n_xi: int = 4, nxi_for_x=(4, 2), total_size: int = 40):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        n_xi=n_xi,
        total_size=total_size,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=np.asarray(nxi_for_x, dtype=np.int32))),
    )


def test_resolve_transport_active_dof_mode_respects_env_and_auto() -> None:
    forced = resolve_transport_active_dof_mode(
        op=_op(),
        rhs_mode=2,
        solve_method_use="dense",
        solve_method="auto",
        active_dof_env="1",
    )
    assert forced.use_active_dof_mode
    assert forced.reason == "env"
    assert forced.solve_method_use == "auto"

    auto = resolve_transport_active_dof_mode(
        op=_op(nxi_for_x=(4, 3)),
        rhs_mode=2,
        solve_method_use="auto",
        solve_method="auto",
        active_dof_env="",
    )
    assert auto.use_active_dof_mode
    assert auto.reason == "auto"

    disabled = resolve_transport_active_dof_mode(
        op=_op(nxi_for_x=(4, 4)),
        rhs_mode=2,
        solve_method_use="auto",
        solve_method="auto",
        active_dof_env="",
    )
    assert not disabled.use_active_dof_mode
    assert disabled.emit_disabled_hint


def test_build_transport_active_dof_state_builds_inverse_index_map() -> None:
    state = build_transport_active_dof_state(
        op=_op(total_size=6),
        use_active_dof_mode=True,
        active_dof_indices=lambda _op: np.asarray([1, 4], dtype=np.int32),
    )
    assert state.active_size == 2
    assert state.active_idx_np.tolist() == [1, 4]
    assert state.full_to_active_jnp.tolist() == [0, 1, 0, 0, 2, 0]


def test_resolve_transport_dense_policy_handles_memory_caps_and_auto_dense(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB", raising=False)
    policy = resolve_transport_dense_policy(
        rhs_mode=2,
        n_rhs=2,
        total_size=400,
        active_size=400,
        solve_method_use="auto",
        force_krylov=False,
        force_dense=False,
        dense_fallback=True,
        dense_retry_max=6000,
        dense_mem_max_mb=128.0,
        dense_mem_block=False,
        dense_use_mixed=False,
        low_memory_outputs=False,
        dense_backend_allowed=True,
        dense_precond_default=True,
    )
    assert policy.solve_method_use == "dense"
    assert policy.dense_precond_enabled is False

    capped = resolve_transport_dense_policy(
        rhs_mode=2,
        n_rhs=2,
        total_size=6000,
        active_size=6000,
        solve_method_use="dense",
        force_krylov=False,
        force_dense=True,
        dense_fallback=True,
        dense_retry_max=6000,
        dense_mem_max_mb=10.0,
        dense_mem_block=False,
        dense_use_mixed=False,
        low_memory_outputs=False,
        dense_backend_allowed=True,
        dense_precond_default=True,
    )
    assert capped.dense_mem_block
    assert capped.solve_method_use == "incremental"
    assert not capped.force_dense
    assert not capped.dense_fallback


def test_geometry5_mono_low_memory_policy_is_guarded(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", raising=False)
    assert transport_geometry5_mono_low_memory_preferred(
        rhs_mode=3,
        geometry_scheme=5,
        backend="cpu",
        has_fp=False,
        n_x=1,
        total_size=3697,
    )
    assert not transport_geometry5_mono_low_memory_preferred(
        rhs_mode=3,
        geometry_scheme=11,
        backend="cpu",
        has_fp=False,
        n_x=1,
        total_size=3697,
    )
    assert not transport_geometry5_mono_low_memory_preferred(
        rhs_mode=3,
        geometry_scheme=5,
        backend="gpu",
        has_fp=False,
        n_x=1,
        total_size=3697,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", "0")
    assert not transport_geometry5_mono_low_memory_preferred(
        rhs_mode=3,
        geometry_scheme=5,
        backend="cpu",
        has_fp=False,
        n_x=1,
        total_size=3697,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", "1")
    assert transport_geometry5_mono_low_memory_preferred(
        rhs_mode=3,
        geometry_scheme=5,
        backend="gpu",
        has_fp=False,
        n_x=1,
        total_size=1,
    )
    assert not transport_geometry5_mono_low_memory_preferred(
        rhs_mode=2,
        geometry_scheme=1,
        backend="gpu",
        has_fp=True,
        n_x=16,
        total_size=1,
    )
