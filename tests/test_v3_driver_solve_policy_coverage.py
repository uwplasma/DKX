from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.v3_driver as vd


def _rhs1_fp_op(*, constraint_scheme: int = 1, include_phi1: bool = False, point_at_x0: bool = False):
    return SimpleNamespace(
        rhs_mode=1,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        point_at_x0=point_at_x0,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )


def _rhs1_pas_op(*, constraint_scheme: int = 1, include_phi1: bool = False):
    return SimpleNamespace(
        rhs_mode=1,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        fblock=SimpleNamespace(fp=None, pas=object()),
    )


def _transport_op(
    *,
    rhs_mode: int = 3,
    has_fp: bool = False,
    include_phi1: bool = False,
    n_x: int = 1,
    constraint_scheme: int = 2,
    n_theta: int = 9,
    n_zeta: int = 5,
    total_size: int = 1024,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        n_x=n_x,
        n_theta=n_theta,
        n_zeta=n_zeta,
        total_size=total_size,
        fblock=SimpleNamespace(fp=object() if has_fp else None, pas=None),
    )


def test_constraint0_petsc_compat_can_be_enabled_and_respects_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "1")
    assert vd._rhsmode1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not vd._rhsmode1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=1),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not vd._rhsmode1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="dense",
        sparse_precond_mode="auto",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not vd._rhsmode1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=4096,
        sparse_max_size=6000,
    )
    assert not vd._rhsmode1_constraint0_petsc_compat(
        op=_rhs1_fp_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=7000,
        sparse_max_size=6000,
    )


def test_constraint0_dense_fallback_policy() -> None:
    assert vd._rhsmode1_constraint0_dense_fallback_allowed(_rhs1_fp_op(constraint_scheme=1))
    assert not vd._rhsmode1_constraint0_dense_fallback_allowed(_rhs1_fp_op(constraint_scheme=0))


def test_sparse_pc_default_permc_spec_targets_pas_er_rows() -> None:
    assert (
        vd._rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=True,
            n_species=2,
        )
        == "MMD_AT_PLUS_A"
    )
    assert (
        vd._rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == "MMD_AT_PLUS_A"
    )
    assert (
        vd._rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=True,
            tokamak_pas_er_pc=False,
            n_species=2,
        )
        == "MMD_ATA"
    )
    assert (
        vd._rhsmode1_sparse_pc_default_permc_spec(
            constrained_pas_pc=False,
            tokamak_pas_er_pc=False,
            n_species=2,
        )
        == "COLAMD"
    )


def test_sparse_pc_default_restart_caps_one_species_pas_er_without_env() -> None:
    assert (
        vd._rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 40
    )
    assert (
        vd._rhsmode1_sparse_pc_default_restart(
            requested_restart=20,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 20
    )
    assert (
        vd._rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=True,
            n_species=2,
        )
        == 80
    )
    assert (
        vd._rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="80",
            tokamak_pas_er_pc=True,
            n_species=1,
        )
        == 80
    )
    assert (
        vd._rhsmode1_sparse_pc_default_restart(
            requested_restart=80,
            restart_env_value="",
            tokamak_pas_er_pc=False,
            n_species=1,
        )
        == 80
    )


def test_sparse_exact_lu_requested_covers_pas_full_and_accelerator_small_case(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", raising=False)

    assert vd._rhsmode1_sparse_exact_lu_requested(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    assert vd._rhsmode1_sparse_exact_lu_requested(
        op=_rhs1_pas_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        full_precond_requested=False,
        preconditioner_x=1,
        use_dkes=False,
    )


def test_sparse_exact_lu_requested_respects_off_dense_and_size_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "off")
    assert not vd._rhsmode1_sparse_exact_lu_requested(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", "2000")
    assert not vd._rhsmode1_sparse_exact_lu_requested(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=3500,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )
    assert not vd._rhsmode1_sparse_exact_lu_requested(
        op=_rhs1_fp_op(),
        solve_method_kind="dense",
        active_size=1000,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )


def test_large_cpu_xblock_skip_primary_allowed_positive_and_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert vd._rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="auto",
    )
    assert not vd._rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_rhs1_fp_op(point_at_x0=True),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="auto",
    )
    assert not vd._rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_rhs1_fp_op(),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="xblock_tz",
    )


def test_transport_sparse_direct_first_attempt_handles_invalid_cpu_max_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MAX", "bad")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert vd._transport_sparse_direct_first_attempt_allowed(
        op=_transport_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
    )


def test_transport_host_gmres_first_attempt_respects_disable_and_invalid_max(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", "off")
    assert not vd._transport_host_gmres_first_attempt_allowed(
        op=_transport_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST_MAX", "bad")
    assert vd._transport_host_gmres_first_attempt_allowed(
        op=_transport_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_host_gmres_accepts_preconditioned_residual_handles_invalid_env_and_nonfinite(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", "bad")
    assert not vd._transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=float("nan"),
        target_true=1.0e-6,
    )
    assert vd._transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=8.0e-6,
        target_true=1.0e-6,
    )
    assert not vd._transport_host_gmres_accepts_preconditioned_residual(
        op=_transport_op(rhs_mode=2, has_fp=False, n_x=3),
        true_residual_norm=2.0e-5,
        target_true=1.0e-6,
    )


def test_transport_precondition_side_accepts_none_override(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECONDITION_SIDE", "none")
    assert vd._transport_precondition_side(
        op=_transport_op(),
        use_implicit=False,
    ) == "none"


def test_transport_disable_auto_recycle_forced_on(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DISABLE_AUTO_RECYCLE", "1")
    assert vd._transport_disable_auto_recycle(
        op=_transport_op(rhs_mode=2, has_fp=True, n_x=4, constraint_scheme=1),
        use_implicit=True,
    )


def test_transport_sparse_direct_needs_float64_retry_nonfinite_and_invalid_ratio(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FLOAT64_RETRY_RATIO", "bad")
    assert vd._transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(np.float32),
        residual_norm=float("nan"),
        target_true=1.0e-6,
    )
    assert vd._transport_sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(np.float32),
        residual_norm=2.0e-5,
        target_true=1.0e-6,
    )


def test_transport_sparse_factor_dtype_respects_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float32")
    assert vd._transport_sparse_factor_dtype(size=99999, use_implicit=False) == np.dtype(np.float32)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE", "float64")
    assert vd._transport_sparse_factor_dtype(size=1, use_implicit=False) == np.dtype(np.float64)


def test_host_scipy_krylov_requested_and_dispatch_host_only(monkeypatch) -> None:
    assert vd._host_scipy_krylov_requested("lgmres")
    assert vd._host_scipy_krylov_requested(" LGMRES_SCIPY ")
    assert not vd._host_scipy_krylov_requested("incremental")

    sentinel = object()
    monkeypatch.setattr(vd, "gmres_solve", lambda **kwargs: ("host", kwargs["solve_method"], sentinel))
    monkeypatch.setattr(vd, "distributed_gmres_enabled", lambda: False)
    out = vd._gmres_solve_dispatch(solve_method="lgmres", distributed_axis=None, size_hint=10, rhs=jnp.ones(1))
    assert out == ("host", "lgmres", sentinel)


def test_gmres_dispatch_rejects_host_only_method_with_distributed_axis(monkeypatch) -> None:
    monkeypatch.setattr(vd, "distributed_gmres_enabled", lambda: False)
    with pytest.raises(ValueError, match="host-only"):
        vd._gmres_solve_dispatch(solve_method="lgmres", distributed_axis="theta", size_hint=10, rhs=jnp.ones(1))


def test_gmres_dispatch_selects_jit_and_nonjit_paths(monkeypatch) -> None:
    monkeypatch.setattr(vd, "distributed_gmres_enabled", lambda: False)
    monkeypatch.setattr(vd, "_use_solver_jit", lambda size_hint=None: False)
    monkeypatch.setattr(vd, "gmres_solve", lambda **kwargs: ("plain", kwargs.get("solve_method")))
    monkeypatch.setattr(vd, "gmres_solve_jit", lambda **kwargs: ("jit", kwargs.get("solve_method")))
    assert vd._gmres_solve_dispatch(solve_method="incremental", distributed_axis=None, size_hint=10) == (
        "plain",
        "incremental",
    )
    monkeypatch.setattr(vd, "_use_solver_jit", lambda size_hint=None: True)
    assert vd._gmres_solve_dispatch(solve_method="incremental", distributed_axis=None, size_hint=10) == (
        "jit",
        "incremental",
    )


def test_gmres_with_residual_dispatch_host_only_and_distributed_paths(monkeypatch) -> None:
    monkeypatch.setattr(vd, "distributed_gmres_enabled", lambda: False)
    monkeypatch.setattr(vd, "gmres_solve_with_residual", lambda **kwargs: ("host_residual", kwargs["solve_method"]))
    monkeypatch.setattr(
        vd,
        "gmres_solve_with_residual_distributed",
        lambda **kwargs: ("distributed_residual", kwargs.get("axis_name")),
    )
    assert vd._gmres_solve_with_residual_dispatch(
        solve_method="lgmres_scipy",
        distributed_axis=None,
        size_hint=10,
    ) == ("host_residual", "lgmres_scipy")
    assert vd._gmres_solve_with_residual_dispatch(
        solve_method="incremental",
        distributed_axis="zeta",
        size_hint=10,
    ) == ("distributed_residual", "zeta")
