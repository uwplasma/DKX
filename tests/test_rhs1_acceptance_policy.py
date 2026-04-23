from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.rhs1_acceptance_policy import rhs1_host_factor_probe_ok, rhs1_pas_fast_accept


def _op(*, has_pas: bool = True, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        fblock=SimpleNamespace(pas=object() if has_pas else None),
    )


def test_pas_fast_accept_allows_large_explicit_cpu_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", raising=False)
    assert rhs1_pas_fast_accept(
        op=_op(),
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
        backend="cpu",
    )


def test_pas_fast_accept_env_overrides_and_invalid_parsing(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", "bad")
    assert rhs1_pas_fast_accept(
        op=_op(),
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
        backend="cpu",
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", "1000")
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", "1.0")
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", "1e-6")
    assert rhs1_pas_fast_accept(
        op=_op(),
        active_size=1000,
        residual_norm=5.0e-7,
        target=1.0e-10,
        use_implicit=False,
        backend="cpu",
    )


def test_pas_fast_accept_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_pas_fast_accept(**{**kwargs, "active_size": 5000})
    assert not rhs1_pas_fast_accept(**{**kwargs, "residual_norm": 1.0e-5})
    assert not rhs1_pas_fast_accept(**{**kwargs, "residual_norm": float("nan")})
    assert not rhs1_pas_fast_accept(**{**kwargs, "use_implicit": True})
    assert not rhs1_pas_fast_accept(**{**kwargs, "backend": "gpu"})
    assert not rhs1_pas_fast_accept(**{**kwargs, "op": _op(has_pas=False)})
    assert not rhs1_pas_fast_accept(**{**kwargs, "op": _op(has_phi1=True)})
    monkeypatch.setenv("SFINCS_JAX_PAS_FAST_ACCEPT", "0")
    assert not rhs1_pas_fast_accept(**kwargs)


class _Factor:
    def __init__(self, value) -> None:
        self.value = value

    def solve(self, x):
        if self.value == "raise":
            raise RuntimeError("factor failed")
        if self.value == "short":
            return np.ones((max(0, x.size - 1),), dtype=np.float64)
        if self.value == "nan":
            return np.full_like(x, np.nan)
        return float(self.value) * x


def test_host_factor_probe_accepts_bounded_factor(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", raising=False)
    assert rhs1_host_factor_probe_ok(factor=_Factor(10.0), block_size=8)


def test_host_factor_probe_rejects_invalid_factor_outputs(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", raising=False)
    assert not rhs1_host_factor_probe_ok(factor=None, block_size=8)
    assert not rhs1_host_factor_probe_ok(factor=_Factor(1.0), block_size=0)
    assert not rhs1_host_factor_probe_ok(factor=_Factor("raise"), block_size=8)
    assert not rhs1_host_factor_probe_ok(factor=_Factor("short"), block_size=8)
    assert not rhs1_host_factor_probe_ok(factor=_Factor("nan"), block_size=8)


def test_host_factor_probe_rejects_unbounded_factor(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", "100")
    assert not rhs1_host_factor_probe_ok(factor=_Factor(1.0e6), block_size=8)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", "bad")
    assert rhs1_host_factor_probe_ok(factor=_Factor(10.0), block_size=8)
