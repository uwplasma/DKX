from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.problems.transport_parallel_runtime import (
    audit_multi_gpu_case_throughput_summary,
    audit_parallel_scaling_claim_scope,
    audit_sharded_solve_scaling_summary,
    audit_transport_parallel_scaling_summary,
    rewrite_xla_flags,
    transport_parallel_backend,
    transport_parallel_gpu_worker_env,
    transport_parallel_persistent_pool_enabled,
    transport_parallel_pool_executor_kwargs,
    transport_parallel_pool_key,
    transport_parallel_start_method,
    transport_parallel_visible_gpu_ids,
    transport_parallel_worker_env,
)
from sfincs_jax.problems.transport_policies import (
    TransportDDConfig,
    TransportPreconditionerContext,
    TransportPreconditionerDispatchBuilders,
    TransportRuntimePolicy,
    TransportSparseJaxConfig,
    TransportStrongPreconditionerCache,
    auto_transport_preconditioner_choice,
    build_transport_active_dof_state,
    build_transport_preconditioner_from_kind,
    build_transport_strong_preconditioner_from_kind,
    normalize_transport_preconditioner_kind,
    resolve_transport_active_dof_mode,
    resolve_transport_dense_policy,
    resolve_transport_initial_solve_policy,
    resolve_transport_per_rhs_loop_policy,
    resolve_transport_precondition_side_for_kind,
    resolve_transport_preconditioner_choice,
    transport_candidate_is_better,
    transport_dense_accelerator_auto_allowed,
    transport_dense_backend_allowed,
    transport_disable_auto_recycle,
    transport_dd_config_from_env,
    transport_geometry_scheme_from_namelist,
    transport_host_gmres_accepts_preconditioned_residual,
    transport_host_gmres_first_attempt_allowed,
    transport_precondition_side,
    transport_sparse_jax_config_from_env,
    transport_sparse_direct_first_attempt_allowed,
    transport_sparse_direct_needs_float64_retry,
    transport_sparse_direct_rescue_allowed,
    transport_sparse_direct_rescue_first,
    transport_sparse_direct_use_explicit_helper,
    transport_sparse_factor_dtype,
    transport_polish_config_from_env,
    transport_tzfft_accelerator_auto_allowed,
    transport_tzfft_backend_allowed,
    transport_residual_gate_failure,
    transport_residual_gate_failures_from_arrays,
    transport_residual_gate_thresholds_from_env,
    transport_residual_value,
    transport_result_needs_retry,
)


def _op(
    *,
    rhs_mode: int = 2,
    include_phi1: bool = False,
    has_fp: bool = False,
    n_x: int = 2,
    n_theta: int = 8,
    n_zeta: int = 8,
    total_size: int = 2400,
    constraint_scheme: int = 2,
    n_species: int = 1,
    n_xi: int = 6,
    nxi_for_x: tuple[int, ...] | None = None,
    phi1_size: int = 0,
    extra_size: int = 2,
) -> SimpleNamespace:
    if nxi_for_x is None:
        nxi_for_x = tuple([n_xi] * n_x)
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray(nxi_for_x, dtype=np.int32)),
        ),
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_species=n_species,
        total_size=total_size,
        constraint_scheme=constraint_scheme,
        phi1_size=phi1_size,
        extra_size=extra_size,
    )


def test_transport_backend_flags_share_boolean_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", raising=False)

    assert transport_dense_backend_allowed(backend="cpu")
    assert not transport_dense_backend_allowed(backend="gpu")
    assert transport_tzfft_backend_allowed(backend="cpu")
    assert not transport_tzfft_backend_allowed(backend="cuda")

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "yes")
    assert transport_dense_backend_allowed(backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "off")
    assert not transport_dense_backend_allowed(backend="cpu")

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "on")
    assert transport_tzfft_backend_allowed(backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "no")
    assert not transport_tzfft_backend_allowed(backend="cpu")


def test_transport_accelerator_auto_guards_parse_geometry_and_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _op(rhs_mode=3)
    assert transport_dense_accelerator_auto_allowed(op, backend="gpu", geometry_scheme=1)
    assert not transport_dense_accelerator_auto_allowed(op, backend="cpu", geometry_scheme=1)
    assert not transport_dense_accelerator_auto_allowed(op, backend="tpu", geometry_scheme=1)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_GEOMETRIES", "bad, 11")
    assert not transport_dense_accelerator_auto_allowed(op, backend="gpu", geometry_scheme=1)
    assert transport_dense_accelerator_auto_allowed(op, backend="gpu", geometry_scheme=11)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO_MAX", "bad")
    assert not transport_dense_accelerator_auto_allowed(
        _op(rhs_mode=3, total_size=2501),
        backend="gpu",
        geometry_scheme=11,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO", "0")
    assert not transport_dense_accelerator_auto_allowed(op, backend="gpu", geometry_scheme=11)

    assert transport_tzfft_accelerator_auto_allowed(op, backend="cpu")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", raising=False)
    assert transport_tzfft_accelerator_auto_allowed(op, backend="gpu")
    assert not transport_tzfft_accelerator_auto_allowed(_op(rhs_mode=1), backend="gpu")
    assert not transport_tzfft_accelerator_auto_allowed(_op(rhs_mode=3, include_phi1=True), backend="gpu")
    assert not transport_tzfft_accelerator_auto_allowed(_op(rhs_mode=3, has_fp=True), backend="gpu")
    assert not transport_tzfft_accelerator_auto_allowed(_op(rhs_mode=3, n_x=3), backend="gpu")
    assert not transport_tzfft_accelerator_auto_allowed(
        _op(rhs_mode=3, n_theta=4, n_zeta=8),
        backend="gpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", "bad")
    assert not transport_tzfft_accelerator_auto_allowed(
        _op(rhs_mode=3, total_size=5001),
        backend="gpu",
    )


def test_transport_accelerator_auto_rejects_unbounded_dense_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ACCELERATOR_AUTO", raising=False)

    assert not transport_dense_accelerator_auto_allowed(
        _op(rhs_mode=3, include_phi1=True),
        backend="gpu",
        geometry_scheme=1,
    )
    assert not transport_dense_accelerator_auto_allowed(
        _op(rhs_mode=3, has_fp=True),
        backend="gpu",
        geometry_scheme=1,
    )
    assert not transport_dense_accelerator_auto_allowed(
        _op(rhs_mode=3, n_x=3),
        backend="gpu",
        geometry_scheme=1,
    )
    assert not transport_dense_accelerator_auto_allowed(
        _op(rhs_mode=3, n_theta=4, n_zeta=4),
        backend="gpu",
        geometry_scheme=1,
    )


def test_transport_sparse_direct_and_host_gmres_first_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _op(rhs_mode=2)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", "bad")
    assert transport_sparse_direct_rescue_allowed(
        op=op,
        size=40_000,
        residual_norm=200.0,
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_rescue_allowed(
        op=op,
        size=40_001,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert transport_sparse_direct_rescue_allowed(
        op=op,
        size=100,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert transport_sparse_direct_rescue_allowed(
        op=op,
        size=100,
        residual_norm=1.0,
        target=0.0,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "0")
    assert not transport_sparse_direct_rescue_allowed(
        op=op,
        size=100,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", "0")
    assert not transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", raising=False)
    assert transport_sparse_direct_rescue_first(sparse_direct_rescue=True)

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", "bad")
    assert not transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=11_999,
        use_implicit=False,
        backend="cpu",
    )
    assert transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MAX", "10")
    assert not transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MAX", "bad")
    assert transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_first_attempt_allowed(
        op=op,
        size=12_000,
        use_implicit=True,
        backend="cpu",
    )
    assert not transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=1),
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2, include_phi1=True),
        size=12_000,
        use_implicit=False,
        backend="cpu",
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "bad")
    assert transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=80_000,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=80_001,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", "off")
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=1,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", "1")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", raising=False)
    assert transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=2, has_fp=True, n_x=4, total_size=500_000),
        size=500_000,
        use_implicit=False,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "499999")
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=2, has_fp=True, n_x=4, total_size=500_000),
        size=500_000,
        use_implicit=False,
        backend="cpu",
    )


def test_transport_sparse_and_host_gmres_reject_ineligible_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)

    assert not transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=100,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=True,
        backend="cpu",
    )
    assert not transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=1),
        size=100,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2, include_phi1=True),
        size=100,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3),
        size=80_000,
        residual_norm=1.1e4,
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3),
        size=80_001,
        residual_norm=float("nan"),
        target=1.0,
        use_implicit=False,
        backend="cpu",
    )

    assert transport_sparse_direct_rescue_first(sparse_direct_rescue=False) is False
    assert transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3, n_theta=4, n_zeta=4),
        size=100,
        use_implicit=False,
        backend="gpu",
    )

    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=1,
        use_implicit=True,
        backend="cpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3),
        size=1,
        use_implicit=False,
        backend="gpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=1),
        size=1,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, include_phi1=True),
        size=1,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=True),
        size=1,
        use_implicit=False,
        backend="cpu",
    )
    assert not transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, n_x=3),
        size=1,
        use_implicit=False,
        backend="cpu",
    )


def test_transport_residual_dtype_recycle_and_helper_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mono = _op(rhs_mode=3, constraint_scheme=2)
    assert transport_host_gmres_accepts_preconditioned_residual(
        op=mono,
        true_residual_norm=1.0e3,
        target_true=1.0,
    )
    assert not transport_host_gmres_accepts_preconditioned_residual(
        op=mono,
        true_residual_norm=float("inf"),
        target_true=1.0,
    )
    assert not transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=2),
        true_residual_norm=11.0,
        target_true=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "right")
    assert transport_precondition_side(op=mono, use_implicit=False) == "right"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "invalid")
    assert transport_precondition_side(op=mono, use_implicit=False) == "left"

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", raising=False)
    assert transport_disable_auto_recycle(op=mono, use_implicit=False, backend="cpu")
    assert not transport_disable_auto_recycle(op=mono, use_implicit=True, backend="cpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "0")
    assert not transport_disable_auto_recycle(op=mono, use_implicit=False, backend="cpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "1")
    assert transport_disable_auto_recycle(op=_op(rhs_mode=2), use_implicit=True, backend="gpu")

    assert not transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.float64,
        residual_norm=float("inf"),
        target_true=1.0,
    )
    assert transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.float32,
        residual_norm=float("inf"),
        target_true=1.0,
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FLOAT64_RETRY_RATIO", "bad")
    assert transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.float32,
        residual_norm=11.0,
        target_true=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float64")
    assert transport_sparse_factor_dtype(
        size=1,
        use_implicit=False,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    ) == np.dtype(np.float64)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float32")
    assert transport_sparse_factor_dtype(
        size=1,
        use_implicit=False,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
    ) == np.dtype(np.float32)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FLOAT64_MIN", "bad")
    assert transport_sparse_factor_dtype(
        size=30_000,
        use_implicit=False,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    ) == np.dtype(np.float64)
    assert transport_sparse_factor_dtype(
        size=30_000,
        use_implicit=True,
        backend="cpu",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    ) == np.dtype(np.float32)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "0")
    assert not transport_sparse_direct_use_explicit_helper(size=1, backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", "1")
    assert transport_sparse_direct_use_explicit_helper(size=1, backend="cpu")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CPU_MIN", "bad")
    assert transport_sparse_direct_use_explicit_helper(size=1, backend="gpu")
    assert not transport_sparse_direct_use_explicit_helper(size=11_999, backend="cpu")
    assert transport_sparse_direct_use_explicit_helper(size=12_000, backend="cpu")


def test_transport_runtime_policy_binds_backend_and_dtype_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = "cpu"

    def current_backend() -> str:
        return backend

    policy = TransportRuntimePolicy(
        backend=current_backend,
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    )

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    assert policy.dense_backend_allowed()
    backend = "gpu"
    assert not policy.dense_backend_allowed()
    assert policy.sparse_direct_use_explicit_helper(size=1)

    backend = "cpu"
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_FLOAT64_MIN", raising=False)
    assert policy.sparse_factor_dtype(size=30_000, use_implicit=False) == np.dtype(
        np.float64
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_PROGRESS_EVERY", "bad")
    assert policy.host_gmres_progress_every() == 10
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_PROGRESS_EVERY", "0")
    assert policy.host_gmres_progress_every() == 0


def test_transport_initial_active_and_dense_policies_cover_production_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _op(rhs_mode=3, total_size=2_000, nxi_for_x=(6, 4))
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", raising=False)
    policy = resolve_transport_initial_solve_policy(
        op=op,
        rhs_mode=3,
        n_rhs=3,
        solve_method="auto",
        restart=30,
        maxiter=None,
        backend="cpu",
        geometry_scheme=5,
        dense_accelerator_auto_allowed=False,
        dense_backend_policy_allowed=True,
        state_out_requested=False,
        force_stream_diagnostics=None,
        force_store_state=None,
        subset_mode=False,
    )
    assert policy.low_memory_outputs
    assert policy.force_krylov
    assert policy.stream_diagnostics
    assert not policy.store_state_vectors
    assert policy.dense_retry_max == 0
    assert policy.maxiter is None
    assert any("low-memory Krylov" in message for _, message in policy.notes)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_STORE_STATE", "0")
    forced_stream = resolve_transport_initial_solve_policy(
        op=_op(rhs_mode=2, total_size=100),
        rhs_mode=2,
        n_rhs=1,
        solve_method="dense",
        restart=20,
        maxiter=10,
        backend="gpu",
        geometry_scheme=1,
        dense_accelerator_auto_allowed=False,
        dense_backend_policy_allowed=False,
        state_out_requested=False,
        force_stream_diagnostics=None,
        force_store_state=None,
        subset_mode=True,
    )
    assert forced_stream.solve_method_use == "incremental"
    assert forced_stream.stream_diagnostics
    assert forced_stream.store_state_vectors
    assert not forced_stream.dense_backend_allowed
    assert any("streaming diagnostics forced" in message for _, message in forced_stream.notes)
    assert any("dense transport path disabled" in message for _, message in forced_stream.notes)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_MAX_MB", "0.05")
    memory_blocked = resolve_transport_initial_solve_policy(
        op=_op(rhs_mode=2, total_size=200),
        rhs_mode=2,
        n_rhs=1,
        solve_method="auto",
        restart=20,
        maxiter=None,
        backend="cpu",
        geometry_scheme=1,
        dense_accelerator_auto_allowed=False,
        dense_backend_policy_allowed=True,
        state_out_requested=False,
        force_stream_diagnostics=False,
        force_store_state=True,
        subset_mode=False,
    )
    assert memory_blocked.dense_mem_block
    assert memory_blocked.solve_method_use == "incremental"
    assert memory_blocked.gmres_restart >= 80
    assert memory_blocked.maxiter == 800

    decision = resolve_transport_active_dof_mode(
        op=op,
        rhs_mode=3,
        solve_method_use="dense",
        solve_method="incremental",
        active_dof_env="auto",
    )
    assert decision.use_active_dof_mode
    assert decision.reason == "auto"
    assert decision.solve_method_use == "incremental"
    state = build_transport_active_dof_state(
        op=_op(total_size=6),
        use_active_dof_mode=True,
        active_dof_indices=lambda _op: np.asarray([0, 2, 5], dtype=np.int32),
    )
    assert state.active_size == 3
    assert state.active_idx_np.tolist() == [0, 2, 5]
    assert np.asarray(state.full_to_active_jnp).tolist() == [1, 0, 2, 0, 0, 3]
    inactive = build_transport_active_dof_state(
        op=_op(total_size=6),
        use_active_dof_mode=False,
        active_dof_indices=lambda _op: np.asarray([], dtype=np.int32),
    )
    assert inactive.active_idx_np is None
    assert inactive.active_size == 6

    dense_policy = resolve_transport_dense_policy(
        rhs_mode=2,
        n_rhs=2,
        total_size=200,
        active_size=200,
        solve_method_use="auto",
        force_krylov=False,
        force_dense=False,
        dense_fallback=True,
        dense_retry_max=2500,
        dense_mem_max_mb=0.05,
        dense_mem_block=False,
        dense_use_mixed=False,
        low_memory_outputs=False,
        dense_backend_allowed=True,
        dense_precond_default=True,
    )
    assert dense_policy.dense_mem_block
    assert not dense_policy.dense_precond_enabled
    assert dense_policy.solve_method_use == "auto"

    mixed_dense_policy = resolve_transport_dense_policy(
        rhs_mode=3,
        n_rhs=1,
        total_size=70,
        active_size=70,
        solve_method_use="incremental",
        force_krylov=False,
        force_dense=False,
        dense_fallback=True,
        dense_retry_max=500,
        dense_mem_max_mb=0.03,
        dense_mem_block=False,
        dense_use_mixed=False,
        low_memory_outputs=False,
        dense_backend_allowed=True,
        dense_precond_default=True,
    )
    assert mixed_dense_policy.dense_use_mixed
    assert mixed_dense_policy.dense_precond_enabled


def test_transport_loop_polish_and_residual_gate_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _op(rhs_mode=2, constraint_scheme=1, phi1_size=0, extra_size=3)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_EPAR_LOOSE", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_EPAR_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "1")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "123")
    loop = resolve_transport_per_rhs_loop_policy(op=op, rhs_mode=2)
    assert loop.project_nullspace_enabled
    assert loop.projection_candidate(3)
    assert loop.projection_needed(3)
    assert loop.rhs3_krylov_flags(3) == (True, True)
    assert loop.iter_stats_enabled
    assert loop.iter_stats_max_size == 123

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_ABS", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", "bad")
    polish = transport_polish_config_from_env(
        rhs_mode=3,
        residual_norm=1.0,
        target=1.0e-9,
        gmres_restart=20,
        maxiter=None,
    )
    assert polish.enabled
    assert polish.threshold == pytest.approx(1.0e-8)
    assert polish.restart == 80
    assert polish.maxiter == 1600

    better = SimpleNamespace(residual_norm=0.5)
    worse = SimpleNamespace(residual_norm=2.0)
    assert transport_residual_value(SimpleNamespace(residual_norm=float("nan"))) == float("inf")
    assert transport_result_needs_retry(worse, 1.0, result_is_finite=lambda _r: True)
    assert transport_result_needs_retry(better, 1.0, result_is_finite=lambda _r: False)
    assert transport_candidate_is_better(candidate=better, current=worse)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL", "-2")
    assert transport_residual_gate_thresholds_from_env() == (0.0, 0.0)
    failure = transport_residual_gate_failure(
        which_rhs=2,
        residual_norm=5.0,
        rhs_norm=10.0,
        max_abs=4.0,
        max_relative=0.9,
    )
    assert failure is not None
    assert "whichRHS=2" in failure
    failures = transport_residual_gate_failures_from_arrays(
        which_rhs_values=[1, 2],
        residual_norms=[0.1, float("inf")],
        rhs_norms=[1.0, 2.0],
        max_abs=1.0,
        max_relative=0.4,
    )
    assert len(failures) == 1
    assert "whichRHS=2" in failures[0]


def test_transport_preconditioner_selection_and_dispatch_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert normalize_transport_preconditioner_kind(env_value="fp_petsc_like_lu") == "fp_fortran_reduced_lu"
    assert normalize_transport_preconditioner_kind(env_value="fp_active_true_block") == "fp_direct_active_block_schur"
    assert normalize_transport_preconditioner_kind(env_value="fp_line") == "fp_tzfft_line"
    assert normalize_transport_preconditioner_kind(env_value="dd_t") == "theta_dd"
    assert normalize_transport_preconditioner_kind(env_value="unknown-kind") == "auto"
    assert resolve_transport_precondition_side_for_kind(
        kind="fp_tzfft_line",
        requested_side="right",
    ) == ("left", True)
    assert resolve_transport_precondition_side_for_kind(kind="collision", requested_side="bad") == ("left", False)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SXBLOCK_MAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_AUTO_MIN", "100")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=3, has_fp=False, total_size=200, n_species=4, n_x=20),
        default_solver_kind="gmres",
        parallel_workers=2,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis="theta",
    ) == ("theta_schwarz", "theta_schwarz")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=3, has_fp=False, total_size=200, n_species=4, n_x=20),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    ) == ("block", "block")

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO", "1")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=2, has_fp=True, total_size=10_000, n_theta=9, n_zeta=9),
        default_solver_kind="bicgstab",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis=None,
    ) == ("fp_xblock_tz_lu_schur", "fp_xblock_tz_lu_schur")

    messages: list[str] = []
    assert resolve_transport_preconditioner_choice(
        op=_op(rhs_mode=3, has_fp=False, total_size=200),
        transport_precond_kind="tzfft",
        default_solver_kind="bicgstab",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
        backend="gpu",
        emit=lambda _level, message: messages.append(message),
    ) == ("collision", None)
    assert any("tzfft preconditioner disabled" in message for message in messages)


def test_transport_preconditioner_aliases_configs_and_auto_fp_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = {
        "stream_fft": "tzfft",
        "fp_streaming_fft": "fp_tzfft",
        "fp_block_thomas_schur": "fp_tzfft_line_schur",
        "fp_nonavg_line": "fp_local_geom_line",
        "fp_angular_xblock_lu_schur": "fp_xblock_tz_lu_schur",
        "fp_angular_xblock_lu": "fp_xblock_tz_lu",
        "fp_kinetic_lu": "fp_structured_fblock_lu",
        "fp_reduced_pmat_lu": "fp_fortran_reduced_lu",
        "fp_true_block_lu": "fp_direct_active_block_schur",
        "schwarz_theta": "theta_schwarz",
        "dd_z": "zeta_dd",
        "schwarz_zeta": "zeta_schwarz",
        "sparse_jax": "sparse_jax",
    }
    for raw, canonical in aliases.items():
        assert normalize_transport_preconditioner_kind(env_value=raw) == canonical

    class _NamelistLike:
        def __init__(self, value):
            self.value = value

        def group(self, _name):
            return {"geometryScheme": self.value}

    assert transport_geometry_scheme_from_namelist(_NamelistLike("5")) == 5
    assert transport_geometry_scheme_from_namelist(_NamelistLike("bad")) == -1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "999")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_Z", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_OVERLAP", "999")
    dd = transport_dd_config_from_env(op=_op(n_theta=5, n_zeta=7))
    assert dd.block_theta == 5
    assert dd.block_zeta == 7
    assert dd.overlap_theta == 4
    assert dd.overlap_zeta == 6

    for name in (
        "SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL",
        "SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL",
        "SFINCS_JAX_TRANSPORT_SPARSE_JAX_REG",
        "SFINCS_JAX_TRANSPORT_SPARSE_JAX_OMEGA",
        "SFINCS_JAX_TRANSPORT_SPARSE_JAX_SWEEPS",
        "SFINCS_JAX_TRANSPORT_SPARSE_JAX_MAX_MB",
    ):
        monkeypatch.setenv(name, "bad")
    sparse = transport_sparse_jax_config_from_env()
    assert sparse.drop_tol == 0.0
    assert sparse.drop_rel == pytest.approx(1.0e-6)
    assert sparse.reg == pytest.approx(1.0e-10)
    assert sparse.omega == pytest.approx(0.8)
    assert sparse.sweeps == 2
    assert sparse.max_mb == pytest.approx(128.0)

    fp_auto_envs = (
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO",
        "SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO",
    )

    def _disable_all_fp_auto() -> None:
        for env_name in fp_auto_envs:
            monkeypatch.setenv(env_name, "0")

    forced_cases = {
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO": "fp_fortran_reduced_lu",
        "SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_AUTO": "fp_xblock_tz_lu_schur",
        "SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_AUTO": "fp_xblock_tz_lu",
        "SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_AUTO": "fp_structured_fblock_lu",
        "SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_AUTO": "fp_local_geom_line",
        "SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_AUTO": "fp_tzfft_line_schur",
        "SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_AUTO": "fp_tzfft_line",
    }
    for env_name, expected in forced_cases.items():
        _disable_all_fp_auto()
        monkeypatch.setenv(env_name, "1")
        assert auto_transport_preconditioner_choice(
            op=_op(rhs_mode=2, has_fp=True, total_size=10_000, n_theta=9, n_zeta=9),
            default_solver_kind="bicgstab",
            parallel_workers=1,
            dense_mem_block=False,
            tzfft_backend_allowed=True,
            shard_axis=None,
        ) == (expected, expected)

    _disable_all_fp_auto()
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=2, has_fp=True, total_size=100, n_species=1, n_x=2),
        default_solver_kind="bicgstab",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis=None,
    ) == ("sxblock", "sxblock")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=2, has_fp=True, total_size=100, n_species=20, n_x=4),
        default_solver_kind="gmres",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis=None,
    ) == ("sxblock", "block")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=2, has_fp=True, total_size=100_000, n_species=20, n_x=4),
        default_solver_kind="bicgstab",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=True,
        shard_axis=None,
    ) == ("collision", "xmg")

    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=3, has_fp=False, n_species=20, n_x=4, total_size=100_000),
        default_solver_kind="bicgstab",
        parallel_workers=1,
        dense_mem_block=False,
        tzfft_backend_allowed=False,
        shard_axis=None,
    ) == ("collision", "collision")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_AUTO_MIN", "100")
    assert auto_transport_preconditioner_choice(
        op=_op(rhs_mode=3, has_fp=False, total_size=100_000),
        default_solver_kind="bicgstab",
        parallel_workers=2,
        dense_mem_block=True,
        tzfft_backend_allowed=True,
        shard_axis="zeta",
    ) == ("zeta_schwarz", "zeta_schwarz")


def test_transport_preconditioner_builders_and_strong_cache_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _builder(name: str):
        def _build(**kwargs):
            calls.append(name)
            return lambda x, _name=name: (_name, x)

        return _build

    builders = TransportPreconditionerDispatchBuilders(
        collision_builder=_builder("collision"),
        sxblock_builder=_builder("sxblock"),
        block_builder=_builder("block"),
        xmg_builder=_builder("xmg"),
        theta_dd_builder=_builder("theta_dd"),
        theta_schwarz_builder=_builder("theta_schwarz"),
        zeta_dd_builder=_builder("zeta_dd"),
        zeta_schwarz_builder=_builder("zeta_schwarz"),
        tzfft_builder=_builder("tzfft"),
        sparse_jax_builder=_builder("sparse_jax"),
        sparse_jax_cache_key=lambda _op, suffix: ("cache", suffix),
        apply_operator_cached=lambda _op, x: x,
        precond_dtype=lambda _size: np.float64,
        fp_fortran_reduced_lu_builder=_builder("fp_fortran_reduced_lu"),
        fp_direct_active_block_schur_builder=_builder("fp_direct_active_block_schur"),
    )
    context = TransportPreconditionerContext(
        op=_op(rhs_mode=2, has_fp=True),
        active_size=4,
        use_active_dof_mode=True,
        reduce_full=lambda x: x[:4],
        expand_reduced=lambda x: x,
        active_indices_np=np.asarray([0, 1, 2, 3], dtype=np.int32),
        emit=lambda *_args: None,
    )
    dd_config = TransportDDConfig(block_theta=2, overlap_theta=1, block_zeta=2, overlap_zeta=1)
    sparse_config = TransportSparseJaxConfig(
        drop_tol=0.0,
        drop_rel=1.0e-6,
        reg=1.0e-10,
        omega=0.8,
        sweeps=2,
        max_mb=128.0,
    )

    pc = build_transport_preconditioner_from_kind(
        kind="theta_schwarz",
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_config,
        use_reduced=True,
    )
    assert pc("x") == ("theta_schwarz", "x")

    fallback = build_transport_preconditioner_from_kind(
        kind="fp_fortran_reduced_lu",
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_config,
        use_reduced=False,
    )
    assert fallback("y") == ("sxblock", "y")

    reduced = build_transport_preconditioner_from_kind(
        kind="fp_direct_active_block_schur",
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_config,
        use_reduced=True,
    )
    assert reduced("z") == ("fp_direct_active_block_schur", "z")

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "bad")
    emitted: list[str] = []
    too_large_context = TransportPreconditionerContext(
        op=_op(rhs_mode=2, has_fp=False),
        active_size=10_000,
        use_active_dof_mode=False,
        emit=lambda _level, message: emitted.append(message),
    )
    too_large = build_transport_preconditioner_from_kind(
        kind="sparse_jax",
        context=too_large_context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=TransportSparseJaxConfig(
            drop_tol=0.0,
            drop_rel=1.0e-6,
            reg=1.0e-10,
            omega=0.8,
            sweeps=2,
            max_mb=0.001,
        ),
        use_reduced=False,
    )
    assert too_large("q") == ("collision", "q")
    assert any("sparse_jax preconditioner disabled" in message for message in emitted)

    def reused(x):
        return "reused", x

    assert (
        build_transport_strong_preconditioner_from_kind(
            kind="collision",
            use_reduced=False,
            precond_kind_used="collision",
            preconditioner_full=reused,
            preconditioner_reduced=None,
            context=context,
            builders=builders,
            dd_config=dd_config,
            sparse_jax_config=sparse_config,
        )
        is reused
    )

    cache = TransportStrongPreconditionerCache(
        kind="block",
        precond_kind_used=None,
        preconditioner_full=None,
        preconditioner_reduced=None,
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_config,
    )
    first = cache.get(use_reduced=False)
    second = cache.get(use_reduced=False)
    assert first is second
    assert first is not None
    assert first("r") == ("block", "r")
    reduced_cached = cache.get(use_reduced=True)
    assert reduced_cached is not None
    assert reduced_cached("s") == ("block", "s")


def test_transport_preconditioner_builder_dispatch_covers_primary_branches() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _builder(name: str):
        def _build(**kwargs):
            calls.append((name, kwargs))
            return lambda x, _name=name: (_name, x)

        return _build

    def _builders_with_optionals(**optional_builders):
        return TransportPreconditionerDispatchBuilders(
            collision_builder=_builder("collision"),
            sxblock_builder=_builder("sxblock"),
            block_builder=_builder("block"),
            xmg_builder=_builder("xmg"),
            theta_dd_builder=_builder("theta_dd"),
            theta_schwarz_builder=_builder("theta_schwarz"),
            zeta_dd_builder=_builder("zeta_dd"),
            zeta_schwarz_builder=_builder("zeta_schwarz"),
            tzfft_builder=_builder("tzfft"),
            sparse_jax_builder=_builder("sparse_jax"),
            sparse_jax_cache_key=lambda _op, suffix: ("cache", suffix),
            apply_operator_cached=lambda _op, x: x,
            precond_dtype=lambda _size: np.float64,
            **optional_builders,
        )

    context = TransportPreconditionerContext(
        op=_op(rhs_mode=2, has_fp=True),
        active_size=4,
        use_active_dof_mode=False,
        reduce_full=lambda x: x[:4],
        expand_reduced=lambda x: x,
        active_indices_np=np.asarray([0, 1, 2, 3], dtype=np.int32),
        emit=lambda *_args: None,
    )
    dd_config = TransportDDConfig(block_theta=2, overlap_theta=1, block_zeta=3, overlap_zeta=1)
    sparse_config = TransportSparseJaxConfig(
        drop_tol=0.0,
        drop_rel=1.0e-6,
        reg=1.0e-10,
        omega=0.8,
        sweeps=2,
        max_mb=128.0,
    )
    builders = _builders_with_optionals(
        fp_tzfft_builder=_builder("fp_tzfft"),
        fp_tzfft_line_builder=_builder("fp_tzfft_line"),
        fp_tzfft_line_schur_builder=_builder("fp_tzfft_line_schur"),
        fp_local_geom_line_builder=_builder("fp_local_geom_line"),
        fp_xblock_tz_lu_builder=_builder("fp_xblock_tz_lu"),
        fp_xblock_tz_lu_schur_builder=_builder("fp_xblock_tz_lu_schur"),
        fp_structured_fblock_lu_builder=_builder("fp_structured_fblock_lu"),
    )
    expected_by_kind = {
        "xmg": "xmg",
        "theta_dd": "theta_dd",
        "theta_schwarz": "theta_schwarz",
        "zeta_dd": "zeta_dd",
        "zeta_schwarz": "zeta_schwarz",
        "tzfft": "tzfft",
        "fp_tzfft": "fp_tzfft",
        "fp_tzfft_line": "fp_tzfft_line",
        "fp_tzfft_line_schur": "fp_tzfft_line_schur",
        "fp_local_geom_line": "fp_local_geom_line",
        "fp_xblock_tz_lu": "fp_xblock_tz_lu",
        "fp_xblock_tz_lu_schur": "fp_xblock_tz_lu_schur",
        "fp_structured_fblock_lu": "fp_structured_fblock_lu",
        "sxblock": "sxblock",
        "block": "block",
        "unknown": "collision",
    }
    for kind, expected in expected_by_kind.items():
        pc = build_transport_preconditioner_from_kind(
            kind=kind,
            context=context,
            builders=builders,
            dd_config=dd_config,
            sparse_jax_config=sparse_config,
            use_reduced=True,
        )
        assert pc(kind) == (expected, kind)

    fallback_builders = _builders_with_optionals()
    fallback_expected = {
        "fp_tzfft": "tzfft",
        "fp_tzfft_line": "sxblock",
        "fp_tzfft_line_schur": "sxblock",
        "fp_local_geom_line": "sxblock",
        "fp_xblock_tz_lu": "sxblock",
        "fp_xblock_tz_lu_schur": "sxblock",
        "fp_structured_fblock_lu": "sxblock",
    }
    for kind, expected in fallback_expected.items():
        pc = build_transport_preconditioner_from_kind(
            kind=kind,
            context=context,
            builders=fallback_builders,
            dd_config=dd_config,
            sparse_jax_config=sparse_config,
            use_reduced=True,
        )
        assert pc(kind) == (expected, kind)

    sparse_pc = build_transport_preconditioner_from_kind(
        kind="sparse_jax",
        context=context,
        builders=builders,
        dd_config=dd_config,
        sparse_jax_config=sparse_config,
        use_reduced=True,
    )
    assert sparse_pc("u") == ("sparse_jax", "u")
    sparse_calls = [kwargs for name, kwargs in calls if name == "sparse_jax"]
    assert sparse_calls
    assert sparse_calls[-1]["cache_key"] == ("cache", "sparse_jax_active_4")


def test_transport_parallel_scaling_audit_accepts_warm_deterministic_summary() -> None:
    audit = audit_transport_parallel_scaling_summary(
        {
            "benchmark_kind": "transport-worker-scaling",
            "backend": "gpu",
            "rhs_count": 4,
            "device_count": 2,
            "timing_semantics": "warm",
            "results": [
                {"workers": 1, "mean_s": 4.0},
                {"workers": 2, "mean_s": 2.0},
            ],
            "payloads_by_workers": {
                "2": [
                    {"which_rhs_values": [1, 3]},
                    {"which_rhs_values": [2, 4]},
                ],
            },
            "deterministic_output_check": True,
        }
    )

    assert audit.release_scaling_claim
    assert audit.claim_workers == 2
    assert audit.claim_speedup == pytest.approx(2.0)
    assert audit.deterministic_payload_coverage
    assert audit.deterministic_output_check
    assert audit.failures == ()


def test_transport_parallel_scaling_audit_fails_closed_on_missing_provenance() -> None:
    audit = audit_transport_parallel_scaling_summary(
        {
            "backend": "gpu",
            "which_rhs_values": [10, 20],
            "visible_gpu_ids": [],
            "timing_semantics": "mixed",
            "results": [
                {"workers": 1, "mean_s": 10.0},
                {"workers": 3, "mean_s": 1.0},
            ],
            "payloads": [{"which_rhs_values": [10, 10]}],
            "deterministic_output_check": False,
        },
        min_speedup=1.1,
        min_efficiency=0.1,
    )

    assert not audit.release_scaling_claim
    assert any("GPU device count" in failure for failure in audit.failures)
    assert any("cannot be claimed" in failure for failure in audit.failures)
    assert any("mixed warm/cold" in failure for failure in audit.failures)
    assert any("deterministic payload coverage" in failure for failure in audit.failures)
    assert any("deterministic output check" in failure for failure in audit.failures)
    assert any("finite-task ideal" in failure for failure in audit.failures)
    assert any("payload RHS coverage" in note for note in audit.notes)


def test_sharded_scaling_audit_stays_non_release_but_ci_gateable() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "sharded_solve",
            "backend": "gpu",
            "devices": [1, 2],
            "global_warmup": 0,
            "gpu_device_count": 2,
            "results": [
                {"devices": 1, "mean_s": 4.0},
                {"devices": 2, "mean_s": 3.0},
            ],
            "scaling_status": "regression-snapshot",
            "timing_semantics": "hot_solve",
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "hot_solve",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "persistent_compile_cache": True,
                "compile_cache_dir": "examples/performance/output/cache",
            },
            "deterministic_output_check": False,
        }
    )

    assert audit.release_scaling_claim is False
    assert audit.experimental_single_case_scaling
    assert audit.ci_gate_pass
    assert audit.timing_semantics == "hot_solve"
    assert audit.operator_reuse_gate
    assert any("not a release scaling claim" in note for note in audit.notes)


def test_parallel_claim_scope_rejects_plan_only_release_and_measured_mix() -> None:
    audit = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "cpu",
            "rhs_count": 2,
            "workers": [1, 2],
            "artifact_kind": "benchmark_plan",
            "launches_solves": False,
            "release_scaling_claim": True,
            "results": [{"workers": 1, "mean_s": 4.0}],
        }
    )

    assert not audit.release_scaling_supported
    assert audit.plan_only_scope_evidence
    assert any("plan-only" in failure for failure in audit.failures)
    assert any("transport-worker scaling is independent" in note for note in audit.notes)


def test_parallel_claim_scope_keeps_single_case_sharding_experimental() -> None:
    audit = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "devices": [1, 2],
            "experimental_single_case_scaling": True,
            "timing_semantics": "warm",
            "results": [{"devices": 1, "mean_s": 3.0}, {"devices": 2, "mean_s": 2.2}],
        }
    )

    assert audit.claim_scope == "single_case_sharded_solve_experimental"
    assert audit.claim_scope_release_eligible
    assert audit.release_scaling_supported is False
    assert audit.unsupported_single_case_strong_scaling
    assert any("never promotes single-case" in note for note in audit.notes)


def test_multi_gpu_case_throughput_audit_accepts_warm_nonrelease_evidence() -> None:
    audit = audit_multi_gpu_case_throughput_summary(
        {
            "benchmark_kind": "gpu_case_throughput",
            "backend": "gpu",
            "timing_semantics": "cache_warm",
            "required_gpu_count": 2,
            "sequential_one_gpu": {"wall_s": 10.0},
            "parallel_two_gpu": {"wall_s": 5.0},
        },
        min_throughput_speedup=1.5,
    )

    assert audit.ci_gate_pass
    assert audit.release_scaling_claim is False
    assert audit.throughput_speedup == pytest.approx(2.0)
    assert any("not single-case strong scaling" in note for note in audit.notes)


def test_multi_gpu_case_throughput_audit_fails_closed_on_release_and_bad_ratio() -> None:
    audit = audit_multi_gpu_case_throughput_summary(
        {
            "benchmark_kind": "multi_gpu_case_throughput",
            "backend": "cpu",
            "timing_semantics": "cold_start",
            "required_gpu_count": 1,
            "release_scaling_claim": True,
            "sequential_one_gpu": {"wall_s": 10.0},
            "parallel_two_gpu": {"wall_s": 8.0},
            "throughput_speedup": 3.0,
        },
        min_throughput_speedup=1.5,
    )

    assert not audit.ci_gate_pass
    assert any("must not set release_scaling_claim=true" in failure for failure in audit.failures)
    assert any("backend must be 'gpu'" in failure for failure in audit.failures)
    assert any("required_gpu_count=1" in failure for failure in audit.failures)
    assert any("timing semantics 'cold_start'" in failure for failure in audit.failures)
    assert any("does not match wall-time ratio" in failure for failure in audit.failures)


def test_multi_gpu_case_throughput_audit_infers_warm_timing_from_warmup_metadata() -> None:
    audit = audit_multi_gpu_case_throughput_summary(
        {
            "benchmark_kind": "multi_gpu_case_throughput",
            "backend": "gpu",
            "warmup": 1,
            "required_gpu_count": 2,
            "sequential_one_gpu": {"wall_s": 9.0},
            "parallel_two_gpu": {"wall_s": 4.5},
            "throughput_speedup": 2.0,
        },
        min_throughput_speedup=1.5,
    )

    assert audit.ci_gate_pass
    assert audit.timing_semantics == "cache_warm"
    assert audit.throughput_speedup == pytest.approx(2.0)
    assert any("inferred from recorded warmup counts" in note for note in audit.notes)


def test_sharded_scaling_deterministic_gate_rejects_missing_digest() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "device_count": 2,
            "timing_semantics": "warm",
            "experimental_single_case_scaling": True,
            "results": [{"devices": 1, "mean_s": 6.0}, {"devices": 2, "mean_s": 4.0}],
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "warm",
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "timed_repeats": 2,
            },
            "deterministic_output_probe_requested": True,
            "deterministic_output_gate": {
                "passes": True,
                "residual_tolerance": 1.0e-9,
                "max_relative_residual_norm": 1.0e-12,
            },
        }
    )

    assert not audit.ci_gate_pass
    assert audit.deterministic_output_gate is False
    assert any("requires output_digest" in failure for failure in audit.failures)
    assert any("requested deterministic output probe did not pass" in failure for failure in audit.failures)


def test_rewrite_xla_flags_replaces_stale_thread_and_device_caps() -> None:
    rewritten = rewrite_xla_flags(
        " --keep=this "
        "--xla_cpu_parallelism_threads=99 "
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_cpu_multi_thread_eigen_num_threads=32 "
        "--xla_force_host_platform_device_count=8 ",
        6,
        2,
    )

    assert rewritten == (
        "--keep=this "
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_cpu_multi_thread_eigen_num_threads=6 "
        "--xla_force_host_platform_device_count=2"
    )


def test_rewrite_xla_flags_drops_managed_tokens_when_caps_are_unset() -> None:
    rewritten = rewrite_xla_flags(
        "--xla_force_host_platform_device_count=4 "
        "--xla_cpu_multi_thread_eigen_num_threads=3 "
        "--other_flag=yes",
        None,
        None,
    )

    assert rewritten == "--other_flag=yes"


def test_transport_parallel_env_helpers_restore_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "forkserver")
    assert transport_parallel_start_method() == "forkserver"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "bogus")
    assert transport_parallel_start_method() == "spawn"

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND", "gpu_process")
    assert transport_parallel_backend() == "gpu"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND", "process")
    assert transport_parallel_backend() == "cpu"

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "off")
    assert not transport_parallel_persistent_pool_enabled()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "yes")
    assert transport_parallel_persistent_pool_enabled()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 0,1, ,2")
    assert transport_parallel_visible_gpu_ids(3) == ["1", "0", "2"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert transport_parallel_visible_gpu_ids(2) == ["0", "1"]

    worker_env = transport_parallel_gpu_worker_env(gpu_id="7")
    assert worker_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert worker_env["SFINCS_JAX_TRANSPORT_PARALLEL"] == "off"
    assert worker_env["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] == "1"
    assert worker_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert worker_env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"

    monkeypatch.setenv("SFINCS_JAX_CORES", "8")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PIN_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "old")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    calls: list[tuple[str, int | None, int | None]] = []

    def _rewrite(flags: str, cpu_devices: int | None, threads: int | None) -> str:
        calls.append((flags, cpu_devices, threads))
        return "rewritten"

    with transport_parallel_worker_env(parallel_workers=4, rewrite_xla_flags=_rewrite):
        assert calls == [("", None, 1)]
        assert transport_parallel_pool_key(4)[0] == 4
        assert transport_parallel_pool_key(4)[1] == "cpu"
        assert transport_parallel_pool_key(4)[3] is True
        assert transport_parallel_pool_key(4)[4] == "8"
        assert transport_parallel_pool_key(4)[2] == "spawn"
        assert os.environ["XLA_FLAGS"] == "rewritten"
        assert os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ["SFINCS_JAX_SHARD"] == "0"

    assert os.environ["OMP_NUM_THREADS"] == "old"
    assert "XLA_FLAGS" not in os.environ

    emitted: list[tuple[int, str]] = []

    def _get_context(method: str) -> str:
        if method == "fork":
            raise ValueError("unsupported")
        return f"ctx:{method}"

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "fork")
    kwargs = transport_parallel_pool_executor_kwargs(
        parallel_workers=2,
        get_context=_get_context,
        emit=lambda level, message: emitted.append((level, message)),
    )

    assert kwargs == {"max_workers": 2, "mp_context": "ctx:spawn"}
    assert emitted and "using 'spawn'" in emitted[0][1]
