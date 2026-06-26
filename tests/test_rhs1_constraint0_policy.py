from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.problems.profile_policies import (
    rhs1_constraint0_dense_fallback_allowed,
    rhs1_constraint0_petsc_compat,
    rhs1_constraint0_petsc_compat_config_from_env,
    rhs1_constraint0_petsc_compat_regularization,
    rhs1_constraint0_sparse_first,
)


def _op(*, constraint_scheme: int, has_fp: bool = True, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        constraint_scheme=constraint_scheme,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_constraint0_sparse_first_defaults_to_gpu_only(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)
    kwargs = dict(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert rhs1_constraint0_sparse_first(**kwargs, backend="gpu")
    assert not rhs1_constraint0_sparse_first(**kwargs, backend="cpu")


def test_constraint0_sparse_first_env_can_enable_cpu_or_disable_all(monkeypatch) -> None:
    kwargs = dict(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", "1")
    assert rhs1_constraint0_sparse_first(**kwargs)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", "off")
    assert not rhs1_constraint0_sparse_first(**kwargs)


def test_constraint0_sparse_first_respects_problem_and_solver_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)
    base = dict(
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
        backend="gpu",
    )
    assert not rhs1_constraint0_sparse_first(op=_op(constraint_scheme=1), **base)
    assert not rhs1_constraint0_sparse_first(op=_op(constraint_scheme=0, has_fp=False), **base)
    assert not rhs1_constraint0_sparse_first(op=_op(constraint_scheme=0, has_phi1=True), **base)
    assert not rhs1_constraint0_sparse_first(op=_op(constraint_scheme=0, rhs_mode=2), **base)
    assert not rhs1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="dense",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
        backend="gpu",
    )
    assert not rhs1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=3276,
        sparse_max_size=6000,
        backend="gpu",
    )
    assert not rhs1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=8000,
        sparse_max_size=6000,
        backend="gpu",
    )


def test_constraint0_petsc_compat_is_explicit_only(monkeypatch) -> None:
    kwargs = dict(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", raising=False)
    assert not rhs1_constraint0_petsc_compat(**kwargs)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "yes")
    assert rhs1_constraint0_petsc_compat(**kwargs)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "0")
    assert not rhs1_constraint0_petsc_compat(**kwargs)


def test_constraint0_petsc_compat_respects_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "1")
    base = dict(
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not rhs1_constraint0_petsc_compat(op=_op(constraint_scheme=1), **base)
    assert not rhs1_constraint0_petsc_compat(op=_op(constraint_scheme=0, has_fp=False), **base)
    assert not rhs1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="dense_ksp",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not rhs1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not rhs1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=8000,
        sparse_max_size=6000,
    )


def test_constraint0_dense_fallback_is_disabled_unless_explicit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK", raising=False)
    assert not rhs1_constraint0_dense_fallback_allowed(_op(constraint_scheme=0))
    assert rhs1_constraint0_dense_fallback_allowed(_op(constraint_scheme=1))
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK", "on")
    assert rhs1_constraint0_dense_fallback_allowed(_op(constraint_scheme=0))


def test_constraint0_petsc_compat_config_uses_legacy_defaults(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DROP_TOL",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_FILL",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DIAG_PIVOT",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_RESTART",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_MAXITER",
    ):
        monkeypatch.delenv(name, raising=False)

    config = rhs1_constraint0_petsc_compat_config_from_env(
        restart=80,
        maxiter=None,
    )
    assert config.drop_tol == 1.0e-4
    assert config.fill == 10.0
    assert config.diag_pivot == 0.0
    assert config.restart == 2000
    assert config.maxiter == 1

    config = rhs1_constraint0_petsc_compat_config_from_env(
        restart=3000,
        maxiter=4,
    )
    assert config.restart == 3000
    assert config.maxiter == 4


def test_constraint0_petsc_compat_config_parses_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_FILL", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DIAG_PIVOT", "0.25")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_RESTART", "333")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_MAXITER", "7")

    config = rhs1_constraint0_petsc_compat_config_from_env(
        restart=80,
        maxiter=None,
    )
    assert config.drop_tol == 1.0e-3
    assert config.fill == 12.0
    assert config.diag_pivot == 0.25
    assert config.restart == 333
    assert config.maxiter == 7


def test_constraint0_petsc_compat_config_ignores_invalid_values(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DROP_TOL",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_FILL",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DIAG_PIVOT",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_RESTART",
        "SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_MAXITER",
    ):
        monkeypatch.setenv(name, "bad")

    config = rhs1_constraint0_petsc_compat_config_from_env(
        restart=120,
        maxiter=3,
    )
    assert config.drop_tol == 1.0e-4
    assert config.fill == 10.0
    assert config.diag_pivot == 0.0
    assert config.restart == 2000
    assert config.maxiter == 3


def test_constraint0_petsc_compat_regularization_uses_default_override_and_floor(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", raising=False)
    assert rhs1_constraint0_petsc_compat_regularization(max_abs=5.0) == 5.0e-12

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", "1e-8")
    assert rhs1_constraint0_petsc_compat_regularization(max_abs=5.0) == 1.0e-8

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", "bad")
    assert rhs1_constraint0_petsc_compat_regularization(max_abs=5.0) == 5.0e-12

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", "-1")
    assert rhs1_constraint0_petsc_compat_regularization(max_abs=5.0) == 0.0
