from __future__ import annotations

from sfincs_jax.rhs1_sparse_polish_policy import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
)


def test_rhs1_polish_enabled_defaults_on_and_honors_false_values(monkeypatch) -> None:
    monkeypatch.delenv("MY_POLISH_FLAG", raising=False)
    assert rhs1_polish_enabled(env_name="MY_POLISH_FLAG")
    monkeypatch.setenv("MY_POLISH_FLAG", "off")
    assert not rhs1_polish_enabled(env_name="MY_POLISH_FLAG")


def test_rhs1_parse_accept_ratio_clamps_and_handles_invalid_env(monkeypatch) -> None:
    monkeypatch.delenv("MY_ACCEPT_RATIO", raising=False)
    assert rhs1_parse_accept_ratio(env_name="MY_ACCEPT_RATIO", default=10.0) == 10.0
    monkeypatch.setenv("MY_ACCEPT_RATIO", "bad")
    assert rhs1_parse_accept_ratio(env_name="MY_ACCEPT_RATIO", default=10.0) == 10.0
    monkeypatch.setenv("MY_ACCEPT_RATIO", "0.25")
    assert rhs1_parse_accept_ratio(env_name="MY_ACCEPT_RATIO", default=10.0) == 1.0


def test_rhs1_parse_polish_gmres_config_uses_defaults_and_bounds(monkeypatch) -> None:
    monkeypatch.delenv("MY_RESTART", raising=False)
    monkeypatch.delenv("MY_MAXITER", raising=False)
    assert rhs1_parse_polish_gmres_config(
        restart_env_name="MY_RESTART",
        maxiter_env_name="MY_MAXITER",
        default_restart=40,
        default_maxiter=80,
    ) == (40, 80)

    monkeypatch.setenv("MY_RESTART", "bad")
    monkeypatch.setenv("MY_MAXITER", "-1")
    assert rhs1_parse_polish_gmres_config(
        restart_env_name="MY_RESTART",
        maxiter_env_name="MY_MAXITER",
        default_restart=40,
        default_maxiter=80,
    ) == (40, 5)

    monkeypatch.setenv("MY_RESTART", "2")
    monkeypatch.setenv("MY_MAXITER", "3")
    assert rhs1_parse_polish_gmres_config(
        restart_env_name="MY_RESTART",
        maxiter_env_name="MY_MAXITER",
        default_restart=40,
        default_maxiter=80,
        min_restart=7,
        min_maxiter=11,
    ) == (7, 11)
