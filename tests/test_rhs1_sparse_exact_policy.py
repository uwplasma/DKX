from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.problems.profile_policies import (
    rhs1_prefer_sparse_over_dense_shortcut,
    rhs1_sparse_exact_lu_requested,
    rhs1_sparse_prefer_skips_stage2,
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


def test_sparse_exact_lu_auto_enables_full_x_cpu_and_gpu_dkes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    assert rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )
    assert rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=6302,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=True,
        backend="gpu",
    )


def test_sparse_exact_lu_auto_enables_small_accelerator_fp_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", raising=False)
    assert rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=2804,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
        backend="gpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", "2000")
    assert not rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=2804,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
        backend="gpu",
    )


def test_sparse_exact_lu_allows_pas_only_when_full_or_forced(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    kwargs = dict(
        op=_op(has_fp=False, has_pas=True),
        solve_method_kind="incremental",
        active_size=3284,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )
    assert not rhs1_sparse_exact_lu_requested(**kwargs, full_precond_requested=False)
    assert rhs1_sparse_exact_lu_requested(**kwargs, full_precond_requested=True)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    assert rhs1_sparse_exact_lu_requested(**kwargs, full_precond_requested=False)


def test_sparse_exact_lu_respects_disable_dense_phi1_and_size_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "off")
    assert not rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", "bad")
    assert not rhs1_sparse_exact_lu_requested(
        op=_op(has_phi1=True),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )
    assert not rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="dense_ksp",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )
    assert not rhs1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=7000,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
        backend="cpu",
    )


def test_prefer_sparse_over_dense_shortcut_defaults_for_moderate_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", raising=False)
    assert rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )


def test_prefer_sparse_over_dense_shortcut_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", raising=False)
    assert not rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=1000,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=4288,
        sparse_max_size=4000,
        use_implicit=False,
    )
    assert not rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(has_fp=False, has_pas=True),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=True,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", "0")
    assert not rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", "bad")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", raising=False)
    assert rhs1_prefer_sparse_over_dense_shortcut(
        op=_op(),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )


def test_sparse_prefer_skips_stage2_defaults_and_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2", raising=False)
    assert rhs1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="auto",
    )
    assert not rhs1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=False,
        sparse_precond_mode="auto",
    )
    assert not rhs1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="off",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2", "0")
    assert not rhs1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="auto",
    )
