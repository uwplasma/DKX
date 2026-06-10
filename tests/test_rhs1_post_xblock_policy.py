from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_post_xblock_policy import (
    rhs1_fast_post_xblock_polish_allowed,
    rhs1_fp_xblock_global_correction_allowed,
    rhs1_fp_targeted_polish_allowed,
    rhs1_scipy_rescue_abs_floor_after_xblock,
    rhs1_scipy_rescue_active_size_allowed,
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
    assert not rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "active_size": 250000})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAX", "300000")
    assert rhs1_fast_post_xblock_polish_allowed(**{**kwargs, "active_size": 250000})
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAX", raising=False)
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
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", raising=False)
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
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", raising=False)
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
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK", "0")
    assert not rhs1_skip_global_sparse_after_xblock_allowed(**kwargs)


def test_scipy_rescue_abs_floor_after_xblock_defaults_for_large_cpu_fp_seed(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS_MIN", raising=False)
    assert rhs1_scipy_rescue_abs_floor_after_xblock(
        op=_op(),
        active_size=81377,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    ) == 1.0e-9


def test_scipy_rescue_abs_floor_after_xblock_respects_guards_and_override(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=81377,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
        backend="cpu",
    )
    assert rhs1_scipy_rescue_abs_floor_after_xblock(**{**kwargs, "active_size": 8000}) == 0.0
    assert (
        rhs1_scipy_rescue_abs_floor_after_xblock(
            **{**kwargs, "used_large_cpu_xblock_shortcut": False},
        )
        == 0.0
    )
    assert (
        rhs1_scipy_rescue_abs_floor_after_xblock(
            **{**kwargs, "used_explicit_fp_xblock_seed": False},
        )
        == 0.0
    )
    assert rhs1_scipy_rescue_abs_floor_after_xblock(**{**kwargs, "use_implicit": True}) == 0.0
    assert rhs1_scipy_rescue_abs_floor_after_xblock(**{**kwargs, "backend": "gpu"}) == 0.0
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS", "2e-10")
    assert rhs1_scipy_rescue_abs_floor_after_xblock(**{**kwargs, "backend": "gpu"}) == 2.0e-10


def test_scipy_rescue_active_size_cap_blocks_no_seed_large_cpu_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=507004,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=False,
        use_implicit=False,
        backend="cpu",
    )

    assert not rhs1_scipy_rescue_active_size_allowed(**kwargs)
    assert rhs1_scipy_rescue_active_size_allowed(**{**kwargs, "active_size": 120000})
    assert rhs1_scipy_rescue_active_size_allowed(
        **{**kwargs, "used_explicit_fp_xblock_seed": True},
    )
    assert rhs1_scipy_rescue_active_size_allowed(
        **{**kwargs, "used_large_cpu_xblock_shortcut": False},
    )
    assert rhs1_scipy_rescue_active_size_allowed(**{**kwargs, "use_implicit": True})
    assert rhs1_scipy_rescue_active_size_allowed(**{**kwargs, "backend": "gpu"})
    assert rhs1_scipy_rescue_active_size_allowed(**{**kwargs, "op": _op(has_pas=True)})


def test_scipy_rescue_active_size_cap_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE", "0")

    assert rhs1_scipy_rescue_active_size_allowed(
        op=_op(),
        active_size=507004,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=False,
        use_implicit=False,
        backend="cpu",
    )


def test_fp_xblock_global_correction_is_opt_in_and_bounded(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=507004,
        residual_norm=4.96e-5,
        target=4.77e-10,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        sparse_xblock_candidate_accepted=True,
        use_implicit=False,
        backend="cpu",
    )

    assert not rhs1_fp_xblock_global_correction_allowed(**kwargs)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION", "1")
    assert rhs1_fp_xblock_global_correction_allowed(**kwargs)
    assert not rhs1_fp_xblock_global_correction_allowed(**{**kwargs, "residual_norm": 1.0e-11})
    assert not rhs1_fp_xblock_global_correction_allowed(
        **{**kwargs, "used_explicit_fp_xblock_seed": False},
    )
    assert not rhs1_fp_xblock_global_correction_allowed(
        **{**kwargs, "sparse_xblock_candidate_accepted": False},
    )
    assert not rhs1_fp_xblock_global_correction_allowed(**{**kwargs, "active_size": 8000})
    assert not rhs1_fp_xblock_global_correction_allowed(**{**kwargs, "active_size": 700000})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_MAX", "0")
    assert rhs1_fp_xblock_global_correction_allowed(**{**kwargs, "active_size": 700000})
    assert not rhs1_fp_xblock_global_correction_allowed(**{**kwargs, "backend": "gpu"})
