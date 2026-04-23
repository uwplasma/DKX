from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_large_cpu_policy import (
    rhs1_fp_xblock_assembled_host_allowed,
    rhs1_large_cpu_sparse_exact_lu_allowed,
    rhs1_large_cpu_sparse_exact_lu_xblock_allowed,
    rhs1_large_cpu_sparse_rescue_allowed,
    rhs1_large_cpu_sparse_rescue_first,
    rhs1_large_cpu_xblock_skip_primary_allowed,
    rhs1_sparse_sxblock_rescue_allowed,
    rhs1_sparse_xblock_rescue_allowed,
)


def _op(
    *,
    has_fp: bool = True,
    has_pas: bool = False,
    has_phi1: bool = False,
    rhs_mode: int = 1,
    n_species: int = 1,
    point_at_x0: bool = False,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        point_at_x0=point_at_x0,
        n_species=n_species,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            pas=object() if has_pas else None,
        ),
    )


def test_large_cpu_sparse_rescue_defaults_for_large_fullx_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    assert rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )


def test_large_cpu_sparse_rescue_respects_backend_size_and_residual_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", "12000")
    assert not rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=5000,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )
    assert not rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=1,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )
    assert not rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(has_phi1=True),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )
    assert not rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
        backend="gpu",
    )
    assert not rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0e-10,
        target=1.0e-6,
        backend="cpu",
    )


def test_large_cpu_sparse_rescue_allows_moderate_fullx_with_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", raising=False)
    assert rhs1_large_cpu_sparse_rescue_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=1,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )


def test_large_cpu_sparse_rescue_first_and_exact_lu_cap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FIRST", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", raising=False)
    assert rhs1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="")
    assert rhs1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="auto")
    assert not rhs1_large_cpu_sparse_rescue_first(
        large_cpu_sparse_rescue=False,
        strong_precond_env="",
    )
    assert not rhs1_large_cpu_sparse_rescue_first(
        large_cpu_sparse_rescue=True,
        strong_precond_env="theta_line",
    )
    assert rhs1_large_cpu_sparse_exact_lu_allowed(active_size=18366)
    assert not rhs1_large_cpu_sparse_exact_lu_allowed(active_size=68670)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FIRST", "0")
    assert not rhs1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="")


def test_exact_lu_xblock_promotion_requires_good_explicit_cpu_seed(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_ABS", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=68670,
        preconditioner_x=1,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        xblock_seed_residual=4.0e-4,
        xblock_seed_improvement_ratio=212.1,
        use_implicit=False,
        backend="cpu",
    )
    assert rhs1_large_cpu_sparse_exact_lu_xblock_allowed(**kwargs)
    assert not rhs1_large_cpu_sparse_exact_lu_xblock_allowed(**{**kwargs, "backend": "gpu"})
    assert not rhs1_large_cpu_sparse_exact_lu_xblock_allowed(
        **{**kwargs, "used_large_cpu_xblock_shortcut": False},
    )
    assert not rhs1_large_cpu_sparse_exact_lu_xblock_allowed(
        **{**kwargs, "xblock_seed_residual": 1.0e-3},
    )
    assert not rhs1_large_cpu_sparse_exact_lu_xblock_allowed(
        **{**kwargs, "xblock_seed_improvement_ratio": 10.0},
    )


def test_sparse_xblock_rescue_defaults_and_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_RATIO", raising=False)
    kwargs = dict(
        op=_op(),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        residual_norm=1.0,
        target=1.0e-6,
        backend="cpu",
    )
    assert rhs1_sparse_xblock_rescue_allowed(**kwargs)
    assert not rhs1_sparse_xblock_rescue_allowed(**{**kwargs, "preconditioner_x": 0})
    assert not rhs1_sparse_xblock_rescue_allowed(**{**kwargs, "pre_theta": 1})
    assert not rhs1_sparse_xblock_rescue_allowed(**{**kwargs, "backend": "gpu"})


def test_fp_xblock_host_assembly_and_primary_skip(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY", raising=False)
    assert rhs1_fp_xblock_assembled_host_allowed(
        op=_op(),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_fp_xblock_assembled_host_allowed(
        op=_op(point_at_x0=True),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
        backend="cpu",
    )
    assert rhs1_large_cpu_xblock_skip_primary_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
        backend="cpu",
    )
    assert not rhs1_large_cpu_xblock_skip_primary_allowed(
        op=_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="schur",
        backend="cpu",
    )


def test_sparse_sxblock_rescue_requires_explicit_multispecies_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE", raising=False)
    kwargs = dict(
        op=_op(n_species=2),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        backend="cpu",
    )
    assert not rhs1_sparse_sxblock_rescue_allowed(**kwargs)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE", "1")
    assert rhs1_sparse_sxblock_rescue_allowed(**kwargs)
    assert not rhs1_sparse_sxblock_rescue_allowed(**{**kwargs, "op": _op(n_species=1)})
    assert not rhs1_sparse_sxblock_rescue_allowed(**{**kwargs, "solve_method_kind": "dense"})
    assert not rhs1_sparse_sxblock_rescue_allowed(**{**kwargs, "preconditioner_x": 0})
    assert not rhs1_sparse_sxblock_rescue_allowed(**{**kwargs, "active_size": 5000})
