from __future__ import annotations

from sfincs_jax.phi1_newton_policy import (
    phi1_frozen_jacobian_policy,
    phi1_gmres_restart,
    phi1_line_search_policy,
    phi1_use_active_dof_mode,
)


def test_phi1_use_active_dof_mode_auto_and_env_override() -> None:
    assert phi1_use_active_dof_mode(
        rhs_mode=1,
        include_phi1=True,
        has_reduced_modes=True,
        env_value="",
    )
    assert not phi1_use_active_dof_mode(
        rhs_mode=1,
        include_phi1=False,
        has_reduced_modes=True,
        env_value="",
    )
    assert phi1_use_active_dof_mode(
        rhs_mode=2,
        include_phi1=False,
        has_reduced_modes=False,
        env_value="1",
    )
    assert not phi1_use_active_dof_mode(
        rhs_mode=1,
        include_phi1=True,
        has_reduced_modes=True,
        env_value="0",
    )


def test_phi1_gmres_restart_caps_small_active_systems() -> None:
    assert phi1_gmres_restart(active_size=800, gmres_restart=400) == 200
    assert phi1_gmres_restart(active_size=5000, gmres_restart=80) == 80
    assert phi1_gmres_restart(active_size=50, gmres_restart=0) == 1


def test_phi1_frozen_jacobian_policy_defaults_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY", raising=False)
    pol = phi1_frozen_jacobian_policy(include_phi1=True)
    assert pol.mode == "frozen"
    assert pol.use_cache is True
    assert pol.every == 1

    monkeypatch.setenv("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "frozen_op")
    monkeypatch.setenv("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE", "0")
    monkeypatch.setenv("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY", "5")
    pol = phi1_frozen_jacobian_policy(include_phi1=False)
    assert pol.mode == "frozen_op"
    assert pol.use_cache is False
    assert pol.every == 5


def test_phi1_line_search_policy_defaults_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PHI1_STEP_SCALE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_LINESEARCH_FACTOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_LINESEARCH_C1", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_LINESEARCH_MODE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PHI1_LINESEARCH_MAXITER", raising=False)
    pol = phi1_line_search_policy(use_frozen_linearization=True, include_phi1=True)
    assert pol.mode == "petsc"
    assert pol.maxiter == 40
    assert pol.factor is None
    assert pol.step_scale == 1.0

    monkeypatch.setenv("SFINCS_JAX_PHI1_LINESEARCH_MODE", "best")
    monkeypatch.setenv("SFINCS_JAX_PHI1_LINESEARCH_MAXITER", "9")
    monkeypatch.setenv("SFINCS_JAX_PHI1_LINESEARCH_FACTOR", "0.5")
    monkeypatch.setenv("SFINCS_JAX_PHI1_LINESEARCH_C1", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_PHI1_STEP_SCALE", "2.0")
    pol = phi1_line_search_policy(use_frozen_linearization=False, include_phi1=False)
    assert pol.mode == "best"
    assert pol.maxiter == 9
    assert pol.factor == 0.5
    assert pol.c1 == 1.0e-3
    assert pol.step_scale == 2.0
