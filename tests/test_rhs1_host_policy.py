from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.profile_policies import (
    host_sparse_direct_refine_steps,
    host_sparse_factor_dtype_current_backend,
    host_sparse_factor_dtype,
    rhsmode1_constraint0_sparse_first_current_backend,
    rhsmode1_dense_backend_allowed_current_backend,
    rhsmode1_fast_post_xblock_polish_allowed_current_backend,
    rhsmode1_fp_targeted_polish_allowed_current_backend,
    rhsmode1_fp_xblock_global_correction_allowed_current_backend,
    rhsmode1_host_dense_fallback_allowed_current_backend,
    rhsmode1_host_dense_shortcut_allowed_current_backend,
    rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend,
    rhsmode1_large_cpu_sparse_rescue_allowed_current_backend,
    rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend,
    rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend,
    rhsmode1_pas_fast_accept_current_backend,
    rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend,
    rhsmode1_scipy_rescue_active_size_allowed_current_backend,
    rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend,
    rhsmode1_sparse_exact_lu_requested_current_backend,
    rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend,
    rhsmode1_sparse_sxblock_rescue_allowed_current_backend,
    rhsmode1_sparse_xblock_rescue_allowed_current_backend,
    rhs1_dense_auto_fp_allowed,
    rhs1_constrained_pas_sparse_pc_auto_allowed,
    rhs1_dense_auto_fp_accelerator_min,
    rhs1_dense_auto_fp_cutoff,
    rhs1_dense_backend_allowed,
    rhs1_dense_fallback_max,
    rhs1_dense_krylov_allowed,
    rhs1_fp_3d_sparse_pc_auto_allowed,
    rhs1_fp_3d_xblock_sparse_pc_auto_allowed,
    rhs1_structured_full_csr_auto_allowed,
    rhs1_explicit_sparse_host_direct_allowed,
    rhs1_host_dense_fallback_allowed,
    rhs1_host_dense_shortcut_allowed,
    rhs1_host_sparse_direct_allowed,
    rhs1_host_sparse_skip_dense_ratio,
    rhs1_sparse_operator_preconditioned_rescue_allowed,
    rhs1_tokamak_er_dense_auto_allowed,
    rhs1_tokamak_fp_er_sparse_pc_auto_allowed,
    rhs1_tokamak_fp_noer_sparse_pc_auto_allowed,
    rhs1_tokamak_pas_er_sparse_pc_auto_allowed,
    rhs1_tokamak_pas_noer_sparse_pc_auto_allowed,
)


def _op(
    *,
    has_fp: bool = True,
    has_pas: bool = False,
    rhs_mode: int = 1,
    include_phi1: bool = False,
    constraint_scheme: int = 1,
    n_xi: int = 100,
    n_species: int = 1,
    point_at_x0: bool = False,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        n_zeta=5,
        n_xi=n_xi,
        n_species=n_species,
        point_at_x0=point_at_x0,
        fblock=SimpleNamespace(fp=object() if has_fp else None, pas=object() if has_pas else None),
    )


def test_rhs1_structured_full_csr_auto_policy_targets_3d_full_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", raising=False)

    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "1")

    assert rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
    )

    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21),
        active_size=20_000,
        use_implicit=True,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, has_pas=True, n_xi=21),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21, include_phi1=True),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21, n_species=3),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "1")
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=21, n_species=3),
        active_size=20_000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )


def test_rhs1_structured_full_csr_auto_policy_respects_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", "3000")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", "4")

    assert rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=8),
        active_size=3000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=3),
        active_size=3000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=8),
        active_size=2999,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=60),
        active_size=507_004,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MAX_SIZE", "0")
    assert rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=60),
        active_size=507_004,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "0")
    assert not rhs1_structured_full_csr_auto_allowed(
        op=_op(has_fp=True, n_xi=8),
        active_size=3000,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )


def test_rhs1_dense_backend_policy_respects_backend_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", raising=False)
    assert rhs1_dense_backend_allowed(backend="cpu")
    assert not rhs1_dense_backend_allowed(backend="gpu")

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "on")
    assert rhs1_dense_backend_allowed(backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR", "0")
    assert not rhs1_dense_backend_allowed(backend="cpu")


def test_rhs1_host_dense_fallback_and_krylov_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", raising=False)
    assert rhs1_host_dense_fallback_allowed(backend="cpu")
    assert not rhs1_host_dense_fallback_allowed(backend="gpu")
    assert rhs1_dense_krylov_allowed()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "yes")
    assert rhs1_host_dense_fallback_allowed(backend="gpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV", "off")
    assert not rhs1_dense_krylov_allowed()


def test_rhs1_host_dense_shortcut_guards_small_accelerator_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", raising=False)

    assert rhs1_host_dense_shortcut_allowed(
        op=_op(has_fp=True),
        active_size=600,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="gpu",
        dense_fallback_max=5000,
    )
    assert not rhs1_host_dense_shortcut_allowed(
        op=_op(has_fp=True),
        active_size=600,
        use_implicit=True,
        solve_method_kind="incremental",
        backend="gpu",
        dense_fallback_max=5000,
    )
    assert not rhs1_host_dense_shortcut_allowed(
        op=_op(has_fp=False, has_pas=True),
        active_size=600,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="gpu",
        dense_fallback_max=5000,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", "off")
    assert not rhs1_host_dense_shortcut_allowed(
        op=_op(has_fp=True),
        active_size=600,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="gpu",
        dense_fallback_max=5000,
    )


def test_rhs1_dense_fallback_max_respects_fp_pas_and_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", raising=False)

    assert rhs1_dense_fallback_max(_op(has_fp=True)) == 8000
    assert rhs1_dense_fallback_max(_op(has_fp=False, has_pas=True, constraint_scheme=1)) == 0
    assert rhs1_dense_fallback_max(_op(has_fp=False, has_pas=True, constraint_scheme=0)) == 5000

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "800")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", "1200")
    assert rhs1_dense_fallback_max(_op(has_fp=True)) == 1200

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", "0")
    assert rhs1_dense_fallback_max(_op(has_fp=True)) == 800

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", "1600")
    assert rhs1_dense_fallback_max(_op(has_fp=False, has_pas=True, constraint_scheme=2)) == 1600

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", "0")
    assert rhs1_dense_fallback_max(_op(has_fp=False, has_pas=True, constraint_scheme=2)) == 0


def test_rhs1_dense_auto_fp_cutoff_matches_fallback_budget(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", raising=False)

    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=5000) == 5000
    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=3200) == 3200
    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=9000) == 8000

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "2628")
    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=5000) == 2628

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "0")
    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=5000) == 0

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "not-an-int")
    assert rhs1_dense_auto_fp_cutoff(dense_active_cutoff=5000) == 5000


def test_rhs1_dense_auto_fp_accelerator_min_is_tunable(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", raising=False)
    assert rhs1_dense_auto_fp_accelerator_min() == 1000

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", "2400")
    assert rhs1_dense_auto_fp_accelerator_min() == 2400

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", "-1")
    assert rhs1_dense_auto_fp_accelerator_min() == 0

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", "bad")
    assert rhs1_dense_auto_fp_accelerator_min() == 1000


def test_rhs1_dense_auto_fp_allowed_keeps_tiny_accelerator_off_dense(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", raising=False)

    assert rhs1_dense_auto_fp_allowed(backend="cpu", active_size=500, dense_active_cutoff=8000)
    assert not rhs1_dense_auto_fp_allowed(backend="gpu", active_size=500, dense_active_cutoff=8000)
    assert rhs1_dense_auto_fp_allowed(backend="gpu", active_size=5007, dense_active_cutoff=8000)
    assert rhs1_dense_auto_fp_allowed(backend="cpu", active_size=7264, dense_active_cutoff=8000)
    assert rhs1_dense_auto_fp_allowed(backend="gpu", active_size=7264, dense_active_cutoff=8000)
    assert not rhs1_dense_auto_fp_allowed(backend="gpu", active_size=9000, dense_active_cutoff=8000)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", "6000")
    assert not rhs1_dense_auto_fp_allowed(backend="gpu", active_size=5007, dense_active_cutoff=8000)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "0")
    assert not rhs1_dense_auto_fp_allowed(backend="cpu", active_size=500, dense_active_cutoff=8000)


def test_rhs1_host_sparse_direct_and_pc_rescue_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", raising=False)

    assert not rhs1_host_sparse_direct_allowed(sparse_exact_lu=False)
    assert not rhs1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=True)
    assert rhs1_host_sparse_direct_allowed(sparse_exact_lu=True)

    assert rhs1_sparse_operator_preconditioned_rescue_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        sparse_exact_lu=True,
        host_sparse_direct_wanted=True,
        backend="cpu",
    )
    assert not rhs1_sparse_operator_preconditioned_rescue_allowed(
        op=_op(has_fp=True, constraint_scheme=0),
        sparse_exact_lu=True,
        host_sparse_direct_wanted=True,
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", "off")
    assert not rhs1_sparse_operator_preconditioned_rescue_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        sparse_exact_lu=True,
        host_sparse_direct_wanted=True,
        backend="cpu",
    )


def test_rhs1_constrained_pas_sparse_pc_auto_targets_large_nondiff_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MAX", raising=False)

    assert rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=42_850,
        use_implicit=False,
        solve_method_kind="auto",
    )
    assert not rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=15_610,
        use_implicit=False,
        solve_method_kind="auto",
    )
    assert not rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=42_850,
        use_implicit=True,
        solve_method_kind="auto",
    )
    assert not rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=42_850,
        use_implicit=False,
        solve_method_kind="dense",
    )
    assert not rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=True, has_pas=False, constraint_scheme=2),
        active_size=42_850,
        use_implicit=False,
        solve_method_kind="auto",
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC", "off")
    assert not rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=42_850,
        use_implicit=False,
        solve_method_kind="auto",
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC", "on")
    assert rhs1_constrained_pas_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=2),
        active_size=1,
        use_implicit=False,
        solve_method_kind="auto",
    )


def test_rhs1_fp_3d_sparse_pc_auto_targets_measured_cpu_fp_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MAX", raising=False)

    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=404,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=6516,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=404,
        use_implicit=True,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=404,
        use_implicit=False,
        solve_method_kind="dense",
        backend="cpu",
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=404,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
    )
    tokamak = _op(has_fp=True, constraint_scheme=1)
    tokamak.n_zeta = 1
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=tokamak,
        active_size=404,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=False, has_pas=True, constraint_scheme=1),
        active_size=404,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=404,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        eparallel_abs=1.0e-4,
    )
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1, n_xi=25),
        active_size=6516,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN_NXI", "20")
    assert rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1, n_xi=25),
        active_size=6516,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN_NXI", raising=False)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC", "off")
    assert not rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=6516,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC", "on")
    assert rhs1_fp_3d_sparse_pc_auto_allowed(
        op=_op(has_fp=True, constraint_scheme=1),
        active_size=1,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
    )


def test_rhs1_fp_3d_xblock_sparse_pc_auto_targets_qi_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN_NXI", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN_NXI", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX_NXI", raising=False)

    common = {
        "op": _op(has_fp=True, constraint_scheme=1, n_xi=50),
        "active_size": 39_314,
        "use_implicit": False,
        "solve_method_kind": "auto",
        "backend": "cpu",
        "eparallel_abs": 0.0,
    }

    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**common)
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "backend": "gpu"})
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "backend": "cuda"})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 20_000})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 70_000})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "use_implicit": True})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "solve_method_kind": "dense"})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "eparallel_abs": 1.0e-4})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, has_pas=True, constraint_scheme=1, n_xi=50)}
    )
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=50, n_species=2)}
    )
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=50, point_at_x0=True)}
    )
    low_pitch = _op(has_fp=True, constraint_scheme=1, n_xi=25)
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "op": low_pitch})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN_NXI", "20")
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "op": low_pitch})
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN_NXI", raising=False)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC", "off")
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC", "on")
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 1})


def test_rhs1_fp_3d_xblock_sparse_pc_auto_targets_finite_beta_multispecies_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN_NXI", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX_NXI", raising=False)

    finite_beta_like = _op(has_fp=True, constraint_scheme=1, n_xi=12, n_species=2)
    common = {
        "op": finite_beta_like,
        "active_size": 34_276,
        "use_implicit": False,
        "solve_method_kind": "auto",
        "backend": "cpu",
        "eparallel_abs": 0.0,
    }

    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**common)
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "backend": "gpu"})
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=14, n_species=2), "active_size": 58_804}
    )
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=16, n_species=2), "active_size": 99_204}
    )
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{
            **common,
            "backend": "gpu",
            "op": _op(has_fp=True, constraint_scheme=1, n_xi=16, n_species=2),
            "active_size": 99_204,
        }
    )
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 4_540})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 100_001})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 1_020_004})
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=7, n_species=2)}
    )
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=17, n_species=2)}
    )
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
        **{**common, "op": _op(has_fp=True, constraint_scheme=1, n_xi=12, n_species=3)}
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES", "off")
    assert not rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES", "on")
    assert rhs1_fp_3d_xblock_sparse_pc_auto_allowed(**{**common, "active_size": 1})


def test_rhs1_tokamak_er_dense_auto_targets_bounded_cpu_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX_BYTES", raising=False)

    fp_tokamak = _op(has_fp=True, constraint_scheme=1)
    fp_tokamak.n_zeta = 1
    pas_tokamak = _op(has_fp=False, has_pas=True, constraint_scheme=2)
    pas_tokamak.n_zeta = 1

    assert rhs1_tokamak_er_dense_auto_allowed(
        op=fp_tokamak,
        active_size=5677,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )

    assert rhs1_tokamak_er_dense_auto_allowed(
        op=pas_tokamak,
        active_size=5686,
        use_implicit=False,
        solve_method_kind="incremental",
        backend="cpu",
        use_dkes=False,
        er_abs=30.0,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=pas_tokamak,
        active_size=5686,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        use_dkes=False,
        er_abs=30.0,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=pas_tokamak,
        active_size=5686,
        use_implicit=True,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=False,
        er_abs=30.0,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=pas_tokamak,
        active_size=5686,
        use_implicit=False,
        solve_method_kind="dense",
        backend="cpu",
        use_dkes=False,
        er_abs=30.0,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=pas_tokamak,
        active_size=5686,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=False,
        er_abs=0.0,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    three_d = _op(has_fp=True, constraint_scheme=1)
    three_d.n_zeta = 5
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=three_d,
        active_size=5677,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE", "off")
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=fp_tokamak,
        active_size=5677,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE", "on")
    assert rhs1_tokamak_er_dense_auto_allowed(
        op=fp_tokamak,
        active_size=1,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX", "5000")
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=fp_tokamak,
        active_size=5677,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX", "6500")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX_BYTES", "1")
    assert not rhs1_tokamak_er_dense_auto_allowed(
        op=fp_tokamak,
        active_size=5677,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        use_dkes=True,
        er_abs=30.0,
        include_xdot=False,
        include_electric_field_xi=False,
    )


def test_rhs1_tokamak_pas_er_sparse_pc_targets_production_floor_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MAX", raising=False)

    pas_tokamak = _op(has_fp=False, has_pas=True, constraint_scheme=2)
    pas_tokamak.n_zeta = 1
    common = dict(
        op=pas_tokamak,
        active_size=12_733,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        er_abs=1.0e-2,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**common)
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "active_size": 2_000})
    assert rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "backend": "gpu"})
    assert rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "backend": "cuda"})
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "backend": "tpu"})
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "use_implicit": True})
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "solve_method_kind": "dense"})
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "er_abs": 0.0})
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
        **{
            **common,
            "include_xdot": False,
            "include_electric_field_xi": False,
            "use_dkes": False,
        }
    )

    non_tokamak = _op(has_fp=False, has_pas=True, constraint_scheme=2)
    non_tokamak.n_zeta = 5
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "op": non_tokamak})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC", "off")
    assert not rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC", "on")
    assert rhs1_tokamak_pas_er_sparse_pc_auto_allowed(**{**common, "active_size": 1})


def test_rhs1_tokamak_pas_noer_sparse_pc_targets_production_floor_window(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MAX", raising=False)

    pas_tokamak = _op(has_fp=False, has_pas=True, constraint_scheme=2)
    pas_tokamak.n_zeta = 1
    common = dict(
        op=pas_tokamak,
        active_size=25_466,
        use_implicit=False,
        solve_method_kind="auto",
        backend="cpu",
        er_abs=0.0,
    )
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**common)
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "active_size": 5_604})
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "active_size": 469_321})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "active_size": 900_000})
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "backend": "gpu"})
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "backend": "cuda"})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "backend": "tpu"})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "active_size": 2_000})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "use_implicit": True})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "solve_method_kind": "dense"})
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "er_abs": 1.0e-2})

    non_tokamak = _op(has_fp=False, has_pas=True, constraint_scheme=2)
    non_tokamak.n_zeta = 5
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "op": non_tokamak})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC", "off")
    assert not rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC", "on")
    assert rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(**{**common, "active_size": 1})


def test_rhs1_tokamak_fp_er_sparse_pc_targets_gpu_production_floor(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MAX", raising=False)

    fp_tokamak = _op(has_fp=True, has_pas=False, constraint_scheme=1)
    fp_tokamak.n_zeta = 1
    common = dict(
        op=fp_tokamak,
        active_size=12_727,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        er_abs=1.0e-2,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**common)
    assert rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "backend": "cuda"})
    assert rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "backend": "cpu"})
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "active_size": 2_000})
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "use_implicit": True})
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "solve_method_kind": "dense"})
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "er_abs": 0.0})

    non_tokamak = _op(has_fp=True, has_pas=False, constraint_scheme=1)
    non_tokamak.n_zeta = 5
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "op": non_tokamak})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC", "off")
    assert not rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC", "on")
    assert rhs1_tokamak_fp_er_sparse_pc_auto_allowed(**{**common, "backend": "cpu", "active_size": 1})


def test_rhs1_tokamak_fp_noer_sparse_pc_targets_gpu_production_floor(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MAX", raising=False)

    fp_tokamak = _op(has_fp=True, has_pas=False, constraint_scheme=0)
    fp_tokamak.n_zeta = 1
    common = dict(
        op=fp_tokamak,
        active_size=12_725,
        use_implicit=False,
        solve_method_kind="auto",
        backend="gpu",
        er_abs=0.0,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**common)
    assert rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "backend": "cuda"})
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "backend": "cpu"})
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "active_size": 2_000})
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "use_implicit": True})
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "solve_method_kind": "dense"})
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "er_abs": 1.0e-2})

    er_scheme = _op(has_fp=True, has_pas=False, constraint_scheme=1)
    er_scheme.n_zeta = 1
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "op": er_scheme})

    non_tokamak = _op(has_fp=True, has_pas=False, constraint_scheme=0)
    non_tokamak.n_zeta = 5
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "op": non_tokamak})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC", "off")
    assert not rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC", "on")
    assert rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(**{**common, "backend": "cpu", "active_size": 1})


def test_host_sparse_factor_dtype_and_refinement_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)

    assert host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=False, backend="cpu") == np.dtype(np.float32)
    assert host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=False, backend="gpu") == np.dtype(np.float64)
    assert host_sparse_factor_dtype(size=20_000, factorization="ilu", use_implicit=False, backend="cpu") == np.dtype(np.float64)
    assert host_sparse_factor_dtype(size=20_000, factorization="lu", use_implicit=True, backend="cpu") == np.dtype(np.float64)

    monkeypatch.setenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "32")
    assert host_sparse_factor_dtype(size=1, factorization="lu", use_implicit=True, backend="gpu") == np.dtype(np.float32)

    monkeypatch.delenv("MY_REFINE_STEPS", raising=False)
    assert host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=3) == 3
    monkeypatch.setenv("MY_REFINE_STEPS", "bad")
    assert host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 2
    monkeypatch.setenv("MY_REFINE_STEPS", "-4")
    assert host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 0


def test_rhs1_sparse_helper_bounds_and_skip_dense_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", raising=False)

    assert rhs1_host_sparse_skip_dense_ratio() == 1.0e4
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", "bad")
    assert rhs1_host_sparse_skip_dense_ratio() == 1.0e4

    assert rhs1_explicit_sparse_host_direct_allowed(sparse_exact_lu=True, use_implicit=False, active_size=20_000)
    assert not rhs1_explicit_sparse_host_direct_allowed(sparse_exact_lu=True, use_implicit=False, active_size=20_001)
    assert not rhs1_explicit_sparse_host_direct_allowed(sparse_exact_lu=False, use_implicit=False, active_size=10)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", "off")
    assert not rhs1_explicit_sparse_host_direct_allowed(sparse_exact_lu=True, use_implicit=False, active_size=10)


def test_current_backend_rhs1_policy_wrappers_delegate_cpu_defaults(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR",
        "SFINCS_JAX_RHSMODE1_DENSE_HOST_LU",
        "SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE",
        "SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN",
        "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO",
        "SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "sfincs_jax.problems.profile_policies.jax.default_backend",
        lambda: "cpu",
    )

    op_fp = _op(has_fp=True, has_pas=False, constraint_scheme=1)
    op_pas = _op(has_fp=False, has_pas=True, constraint_scheme=2)

    assert rhsmode1_dense_backend_allowed_current_backend()
    assert rhsmode1_host_dense_fallback_allowed_current_backend()
    assert not rhsmode1_host_dense_shortcut_allowed_current_backend(
        op=op_fp,
        active_size=100,
        use_implicit=False,
        solve_method_kind="incremental",
    )
    assert rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend(
        op=op_fp,
        sparse_exact_lu=True,
        host_sparse_direct_wanted=True,
    )
    assert host_sparse_factor_dtype_current_backend(
        size=20_000,
        factorization="lu",
        use_implicit=False,
    ) == np.dtype(np.float32)

    assert not rhsmode1_pas_fast_accept_current_backend(
        op=op_pas,
        active_size=1000,
        residual_norm=1.0e-10,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not rhsmode1_constraint0_sparse_first_current_backend(
        op=_op(has_fp=True, has_pas=False, constraint_scheme=0),
        solve_method_kind="auto",
        sparse_precond_mode="auto",
        active_size=1000,
        sparse_max_size=2000,
    )
    assert not rhsmode1_sparse_exact_lu_requested_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=1000,
        sparse_max_size=2000,
        full_precond_requested=False,
        preconditioner_x=1,
        use_dkes=False,
    )
    assert not rhsmode1_large_cpu_sparse_rescue_allowed_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=15_000,
        sparse_max_size=20_000,
        preconditioner_x=1,
        residual_norm=1.0e-2,
        target=1.0e-9,
    )
    assert not rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=15_000,
        sparse_max_size=20_000,
        use_implicit=False,
    )
    assert rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        preconditioner_x=1,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        xblock_seed_residual=1.0e-6,
        xblock_seed_improvement_ratio=100.0,
        use_implicit=False,
    )
    assert not rhsmode1_sparse_xblock_rescue_allowed_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=15_000,
        sparse_max_size=20_000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        residual_norm=1.0e-2,
        target=1.0e-9,
    )
    assert not rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=15_000,
        sparse_max_size=20_000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )
    assert not rhsmode1_fast_post_xblock_polish_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        residual_norm=1.0e-7,
        target=1.0e-9,
        used_large_cpu_xblock_shortcut=True,
        use_implicit=False,
    )
    assert not rhsmode1_fp_targeted_polish_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        residual_norm=1.0e-7,
        target=1.0e-9,
        rhs1_precond_kind="xblock_tz",
        use_implicit=False,
    )
    assert rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        residual_norm=1.0e-10,
        target=1.0e-9,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
    )
    assert not rhsmode1_fp_xblock_global_correction_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        residual_norm=1.0e-7,
        target=1.0e-9,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        sparse_xblock_candidate_accepted=True,
        use_implicit=False,
    )
    assert rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend(
        op=op_fp,
        active_size=15_000,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
    ) == 1.0e-9
    assert rhsmode1_scipy_rescue_active_size_allowed_current_backend(
        op=op_fp,
        active_size=15_000,
        used_large_cpu_xblock_shortcut=True,
        used_explicit_fp_xblock_seed=True,
        use_implicit=False,
    )
    assert not rhsmode1_sparse_sxblock_rescue_allowed_current_backend(
        op=op_fp,
        solve_method_kind="auto",
        active_size=15_000,
        sparse_max_size=20_000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )
