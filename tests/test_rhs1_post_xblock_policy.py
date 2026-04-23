from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_post_xblock_policy import (
    rhs1_fast_post_xblock_polish_allowed,
    rhs1_fp_targeted_polish_allowed,
    rhs1_skip_global_sparse_after_xblock_allowed,
)


def _op(*, has_fp: bool = True, has_pas: bool = False, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            pas=object() if has_pas else None,
        ),
    )


def test_fast_post_xblock_polish_triggers_for_bad_large_cpu_seed(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_ABS", raising=False)
    assert rhs1_fast_post_xblock_polish_allowed(
        op=_op(),
        active_size=68670,
        residual_norm=2.8e-4,
        target=1.0e-8,
        used_large_cpu_xblock_shortcut=True,
        use_implicit=False,
        backend="cpu",
    )


def test_fast_post_xblock_polish_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=68670,
        residual_norm=2.8e-4,
        target=1.0e-8,
        used_large_cpu_xblock_shortcut=True,
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "residual_norm": 1.0e-7})
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "active_size": 8000})
    assert not rhs1_fast_post_xblock_polish_allowed(
        **{**kwargs, "used_large_cpu_xblock_shortcut": False},
    )
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "use_implicit": True})
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "op": _op(has_fp=False)})
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "backend": "gpu"})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH", "0")
    assert not rhs1_fast_post_xblock_polish_allowed(**kwargs)


def test_fp_targeted_polish_triggers_for_medium_large_cpu_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_ABS", raising=False)
    assert rhs1_fp_targeted_polish_allowed(
        op=_op(),
        active_size=68670,
        residual_norm=4.8e-5,
        target=1.0e-8,
        rhs1_precond_kind="xmg",
        use_implicit=False,
        backend="cpu",
    )


def test_fp_targeted_polish_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=68670,
        residual_norm=4.8e-5,
        target=1.0e-8,
        rhs1_precond_kind="xmg",
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_fp_targeted_polish_allowed(**{**kwargs, "active_size": 8000})
    assert not rhs1_fp_targeted_polish_allowed(**{**kwargs, "residual_norm": 1.0e-8})
    assert not rhs1_fp_targeted_polish_allowed(**{**kwargs, "rhs1_precond_kind": "schur"})
    assert not rhs1_fp_targeted_polish_allowed(**{**kwargs, "use_implicit": True})
    assert not rhs1_fp_targeted_polish_allowed(**{**kwargs, "backend": "gpu"})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH", "0")
    assert not rhs1_fp_targeted_polish_allowed(**kwargs)


def test_skip_global_sparse_after_xblock_requires_explicit_good_seed(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", "1")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_ABS", raising=False)
    assert rhs1_skip_global_sparse_after_xblock_allowed(
        op=_op(),
        active_size=68670,
        residual_norm=4.1e-4,
        target=1.0e-8,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    )


def test_skip_global_sparse_after_xblock_respects_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", "1")
    kwargs = dict(
        op=_op(),
        active_size=68670,
        residual_norm=4.1e-4,
        target=1.0e-8,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**{**kwargs, "active_size": 8000})
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**{**kwargs, "residual_norm": 1.0e-3})
    assert not rhs1_skip_global_sparse_after_xblock_allowed(
        **{**kwargs, "used_large_cpu_xblock_shortcut": False},
    )
    assert not rhs1_skip_global_sparse_after_xblock_allowed(
        **{**kwargs, "used_explicit_fp_xblock_seed": False},
    )
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**{**kwargs, "use_implicit": True})
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**{**kwargs, "backend": "gpu"})
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", raising=False)
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**kwargs)
