from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

import sfincs_jax.v3_driver as v3_driver


def test_use_solver_jit_respects_boolean_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "1")
    assert v3_driver._use_solver_jit(size_hint=10_000_000)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "off")
    assert not v3_driver._use_solver_jit(size_hint=1)


def test_use_solver_jit_uses_threshold_and_invalid_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "256")
    assert v3_driver._use_solver_jit(size_hint=128)
    assert not v3_driver._use_solver_jit(size_hint=512)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "not-an-int")
    assert v3_driver._use_solver_jit(size_hint=1)
    assert not v3_driver._use_solver_jit(size_hint=100_001)


def test_use_solver_jit_falls_back_to_cached_size_hint(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "500")
    v3_driver._set_precond_size_hint(400)
    try:
        assert v3_driver._use_solver_jit() is True
        v3_driver._set_precond_size_hint(600)
        assert v3_driver._use_solver_jit() is False
    finally:
        v3_driver._set_precond_size_hint(None)


def test_auto_pas_geom4_fp32_precond_allowed_policy_boundaries(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")

    v3_driver._set_precond_policy_hints(
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )
    v3_driver._set_precond_size_hint(20_000)
    try:
        assert v3_driver._auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)

        monkeypatch.setenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", "off")
        assert not v3_driver._auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)
        monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)

        monkeypatch.setenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", "1e-14")
        v3_driver._set_precond_policy_hints(
            geom_scheme=4,
            use_dkes=False,
            rhs1_precond_kind="schur",
            has_pas=True,
            has_fp=False,
            include_phi1=False,
            rhs_mode=1,
            er_abs=1e-12,
        )
        assert not v3_driver._auto_pas_geom4_fp32_precond_allowed(size_hint=20_000)
    finally:
        v3_driver._set_precond_policy_hints()
        v3_driver._set_precond_size_hint(None)


def test_precond_dtype_respects_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PRECOND_DTYPE", "fp32")
    assert v3_driver._precond_dtype(size_hint=1) == jnp.float32

    monkeypatch.setenv("SFINCS_JAX_PRECOND_DTYPE", "float64")
    assert v3_driver._precond_dtype(size_hint=10_000_000) == jnp.float64


def test_dense_backend_allowed_defaults_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert v3_driver._rhsmode1_dense_backend_allowed()

    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not v3_driver._rhsmode1_dense_backend_allowed()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "on")
    assert v3_driver._rhsmode1_dense_backend_allowed()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "0")
    assert not v3_driver._rhsmode1_dense_backend_allowed()


def test_resource_exhausted_error_detection_includes_causes() -> None:
    exc = RuntimeError("top level")
    exc.__cause__ = MemoryError("resource_exhausted during allocation")
    assert v3_driver._is_resource_exhausted_error(exc)
    assert not v3_driver._is_resource_exhausted_error(RuntimeError("solver diverged"))


def test_rhs1_sharded_line_override_allowed_whitelist() -> None:
    assert v3_driver._rhs1_sharded_line_override_allowed(None)
    assert v3_driver._rhs1_sharded_line_override_allowed("theta_line")
    assert not v3_driver._rhs1_sharded_line_override_allowed("schur")


def test_rhs1_pas_dkes_xblock_rejects_invalid_backend_and_zero_limits() -> None:
    assert not v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="metal",
        n_theta=9,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=1000,
    )
    assert not v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=1,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=1000,
    )
    assert not v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=4,
        xblock_tz_limit=0,
    )


def test_pas_tokamak_gpu_policy_handles_invalid_env_bounds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", "bad")
    assert v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MIN", "1000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", "bad")
    assert not v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", "12000")
    assert v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=1500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )


def test_rhs1_gpu_sparse_fallback_skip_invalid_ratio_and_nonpositive_ratio(monkeypatch) -> None:
    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        fblock=SimpleNamespace(pas=object()),
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "bad")
    assert v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "0")
    assert not v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )


def test_sparse_structural_tol_handles_invalid_and_negative_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", raising=False)
    default_tol = v3_driver._sparse_structural_tol()
    assert default_tol >= 0.0

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "bad")
    assert v3_driver._sparse_structural_tol() == default_tol

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "-1.0")
    assert v3_driver._sparse_structural_tol() == 0.0


def test_transport_tzfft_accelerator_auto_allowed_boundary_cases(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    cpu_op = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=1, n_theta=2, n_zeta=2, total_size=10, fblock=SimpleNamespace(fp=None))
    assert v3_driver._transport_tzfft_accelerator_auto_allowed(cpu_op)

    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ACCELERATOR_AUTO_MAX", "bad")
    reject_phi1 = SimpleNamespace(rhs_mode=3, include_phi1=True, n_x=1, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_rhs = SimpleNamespace(rhs_mode=1, include_phi1=False, n_x=1, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_nx = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=3, n_theta=37, n_zeta=5, total_size=1000, fblock=SimpleNamespace(fp=None))
    reject_grid = SimpleNamespace(rhs_mode=3, include_phi1=False, n_x=1, n_theta=7, n_zeta=7, total_size=1000, fblock=SimpleNamespace(fp=None))
    assert not v3_driver._transport_tzfft_accelerator_auto_allowed(reject_phi1)
    assert not v3_driver._transport_tzfft_accelerator_auto_allowed(reject_rhs)
    assert not v3_driver._transport_tzfft_accelerator_auto_allowed(reject_nx)
    assert not v3_driver._transport_tzfft_accelerator_auto_allowed(reject_grid)


def test_rhsmode1_dense_and_host_dense_policy_envs(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", "0")
    assert not v3_driver._rhsmode1_dense_krylov_allowed()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", "1")
    assert v3_driver._rhsmode1_dense_krylov_allowed()
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", raising=False)
    assert v3_driver._rhsmode1_dense_krylov_allowed()

    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert v3_driver._rhsmode1_host_dense_fallback_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "on")
    assert v3_driver._rhsmode1_host_dense_fallback_allowed()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "off")
    assert not v3_driver._rhsmode1_host_dense_fallback_allowed()
