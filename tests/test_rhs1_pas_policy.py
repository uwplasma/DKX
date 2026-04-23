from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_pas_policy import rhs1_pas_adaptive_smoother_allowed


def _op(*, has_pas: bool = True, has_phi1: bool = False):
    return SimpleNamespace(
        include_phi1=has_phi1,
        fblock=SimpleNamespace(pas=object() if has_pas else None),
    )


def test_pas_adaptive_smoother_allowed_for_large_explicit_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", raising=False)
    assert rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )


def test_pas_adaptive_smoother_respects_problem_and_residual_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "op": _op(has_phi1=True)})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "op": _op(has_pas=False)})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "residual_norm": 1.0e-10})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "active_size": 1000})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "use_implicit": True})


def test_pas_adaptive_smoother_env_controls_and_invalid_min(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "0")
    assert not rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "1")
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", "bad")
    assert rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )
