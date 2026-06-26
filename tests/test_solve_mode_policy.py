from __future__ import annotations

from sfincs_jax.problems.profile_policies import resolve_use_implicit


def test_resolve_use_implicit_honors_explicit_differentiable_flag(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "0")
    assert resolve_use_implicit(differentiable=True) is True
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "1")
    assert resolve_use_implicit(differentiable=False) is False


def test_resolve_use_implicit_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_IMPLICIT_SOLVE", raising=False)
    assert resolve_use_implicit(differentiable=None) is True


def test_resolve_use_implicit_respects_false_env_aliases(monkeypatch) -> None:
    for value in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", value)
        assert resolve_use_implicit(differentiable=None) is False

    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "1")
    assert resolve_use_implicit(differentiable=None) is True
