from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from sfincs_jax.problems.transport_matrix.parallel.runtime import (
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
from sfincs_jax.problems.transport_matrix.policies import (
    TransportRuntimePolicy,
    transport_dense_accelerator_auto_allowed,
    transport_dense_backend_allowed,
    transport_disable_auto_recycle,
    transport_host_gmres_accepts_preconditioned_residual,
    transport_host_gmres_first_attempt_allowed,
    transport_precondition_side,
    transport_sparse_direct_first_attempt_allowed,
    transport_sparse_direct_needs_float64_retry,
    transport_sparse_direct_rescue_allowed,
    transport_sparse_direct_rescue_first,
    transport_sparse_direct_use_explicit_helper,
    transport_sparse_factor_dtype,
    transport_tzfft_accelerator_auto_allowed,
    transport_tzfft_backend_allowed,
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
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
        n_x=n_x,
        n_theta=n_theta,
        n_zeta=n_zeta,
        total_size=total_size,
        constraint_scheme=constraint_scheme,
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
        assert transport_parallel_pool_key(4)[3] == "1"
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
