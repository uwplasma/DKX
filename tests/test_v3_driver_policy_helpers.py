from __future__ import annotations

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
