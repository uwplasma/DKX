from __future__ import annotations

import jax.numpy as jnp

from sfincs_jax import preconditioner_context as pc


def teardown_function() -> None:
    pc.set_precond_size_hint(None)
    pc.set_precond_policy_hints()


def test_preconditioner_context_tracks_size_and_policy_hints() -> None:
    pc.set_precond_size_hint(123)
    pc.set_precond_policy_hints(
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )

    hints = pc.precond_policy_hints()

    assert hints.size_hint == 123
    assert hints.geom_scheme == 4
    assert hints.rhs1_precond_kind == "schur"
    assert hints.has_pas is True
    assert hints.has_fp is False


def test_context_solver_jit_uses_cached_size_hint(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "10")

    pc.set_precond_size_hint(9)
    assert pc.use_solver_jit()

    pc.set_precond_size_hint(11)
    assert not pc.use_solver_jit()


def test_context_preconditioner_dtype_uses_geom4_pas_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PRECOND_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)
    monkeypatch.setenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE", "100")
    monkeypatch.setattr("sfincs_jax.preconditioner_context.jax.default_backend", lambda: "cpu")
    pc.set_precond_policy_hints(
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="pas_schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )

    assert pc.auto_pas_geom4_fp32_precond_allowed(size_hint=100)
    assert pc.precond_dtype(size_hint=100) == jnp.float32


def test_context_sparse_structural_tol_falls_back_for_invalid_env(monkeypatch) -> None:
    default_tol = pc.sparse_structural_tol()

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "not-a-number")
    assert pc.sparse_structural_tol() == default_tol

    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "-1")
    assert pc.sparse_structural_tol() == 0.0
