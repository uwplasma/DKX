from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.problems.transport_matrix.policies import (
    transport_candidate_is_better,
    transport_polish_config_from_env,
    transport_residual_value,
    transport_result_needs_retry,
)


def _residual(value: float):
    return SimpleNamespace(residual_norm=value)


def test_transport_residual_value_maps_nonfinite_to_inf() -> None:
    assert transport_residual_value(_residual(1.25)) == 1.25
    assert transport_residual_value(_residual(float("nan"))) == float("inf")


def test_transport_result_needs_retry_respects_finite_gate() -> None:
    assert transport_result_needs_retry(_residual(0.5), 1.0, result_is_finite=lambda _res: True) is False
    assert transport_result_needs_retry(_residual(2.0), 1.0, result_is_finite=lambda _res: True) is True
    assert transport_result_needs_retry(_residual(0.5), 1.0, result_is_finite=lambda _res: False) is True


def test_transport_candidate_is_better_uses_residual_metric() -> None:
    assert transport_candidate_is_better(candidate=_residual(0.1), current=_residual(0.2))
    assert not transport_candidate_is_better(candidate=_residual(0.3), current=_residual(0.2))


def test_transport_polish_config_defaults_and_rhsmode_gate(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_POLISH_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_POLISH_ABS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_POLISH_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", raising=False)
    config = transport_polish_config_from_env(
        rhs_mode=3,
        residual_norm=3.0e-6,
        target=1.0e-6,
        gmres_restart=40,
        maxiter=400,
    )
    assert config.enabled
    assert config.threshold == 2.0e-6
    assert config.restart == 80
    assert config.maxiter == 1200
    assert not transport_polish_config_from_env(
        rhs_mode=2,
        residual_norm=3.0e-6,
        target=1.0e-6,
        gmres_restart=40,
        maxiter=400,
    ).enabled


def test_transport_polish_config_env_overrides_and_invalid_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_ABS", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", "bad")
    config = transport_polish_config_from_env(
        rhs_mode=3,
        residual_norm=1.0e-7,
        target=1.0e-9,
        gmres_restart=20,
        maxiter=None,
    )
    assert config.enabled
    assert config.ratio == 2.0
    assert config.abs_tol == 1.0e-8
    assert config.restart == 80
    assert config.maxiter == 1600

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RATIO", "5")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_ABS", "1e-5")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_RESTART", "96")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", "321")
    config = transport_polish_config_from_env(
        rhs_mode=3,
        residual_norm=2.0e-5,
        target=1.0e-7,
        gmres_restart=20,
        maxiter=100,
    )
    assert config.enabled
    assert config.threshold == 1.0e-5
    assert config.restart == 96
    assert config.maxiter == 321
