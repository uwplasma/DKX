from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.profile_policies import (
    RHS1DefaultPreconditionerSelectionContext,
    canonical_rhs1_preconditioner_kind,
    pas_auto_skip_strong_retry,
    rhs1_gpu_sparse_fallback_skip_allowed,
    rhs1_fp_dkes_default_kind,
    rhs1_fp_dkes_env_preconditioner_kind,
    rhs1_geometry4_pas_memory_pas_tz_preferred,
    rhs1_large_fp_near_zero_er_override_kind,
    rhs1_measured_auto_promotion_allowed,
    rhs1_measured_auto_promotion_gate,
    rhs1_pas_auto_large_base_kind,
    rhs1_pas_dkes_cpu_pas_tz_preferred,
    rhs1_pas_dkes_pas_tz_preferred,
    rhs1_pas_dkes_xblock_allowed,
    rhs1_pas_family_refinement_kind,
    rhs1_pas_full_pas_tz_preferred,
    rhs1_pas_full_cpu_pas_tz_preferred,
    rhs1_pas_tokamak_cpu_xblock_preferred,
    rhs1_pas_tokamak_gpu_theta_allowed,
    rhs1_pas_tokamak_gpu_tight_tol,
    rhs1_pas_tokamak_gpu_xblock_preferred,
    rhs1_pas_weak_auto_override_kind,
    rhs1_sharded_line_override_allowed,
    resolve_rhs1_default_preconditioner_selection,
)
from sfincs_jax.solvers.selection_policy import SolverCandidateMetrics


def test_canonical_rhs1_preconditioner_kind_preserves_driver_aliases() -> None:
    cases = {
        "": None,
        "off": None,
        "theta": "theta_line",
        "line_theta": "theta_line",
        "theta_block": "theta_dd",
        "ras_theta": "theta_schwarz",
        "theta_xdiag": "theta_line_xdiag",
        "xdiag": "point_xdiag",
        "species": "species_block",
        "species_x": "sxblock",
        "sx_tz": "sxblock_tz",
        "xblock_tz_cut": "xblock_tz_lmax",
        "xtz": "xblock_tz",
        "multigrid": "xmg",
        "pas_light": "pas_lite",
        "pas_line_xcoarse": "pas_hybrid",
        "pas_block_schur": "pas_schur",
        "pas_3d": "pas_tz",
        "pas_fft": "pas_tzfft",
        "pas_streaming_fft": "pas_tzfft",
        "block_ilu": "pas_ilu",
        "tz": "theta_zeta",
        "line_zeta": "zeta_line",
        "dd_z": "zeta_dd",
        "ras_zeta": "zeta_schwarz",
        "zeta_theta": "adi",
        "yes": "point",
        "constraint_schur": "schur",
        "diag": "collision",
        "structured_fblock": "structured_fblock_jacobi",
        "fblock_jacobi": "structured_fblock_jacobi",
        "structured_fblock_angular": "structured_fblock_angular_jacobi",
        "fblock_angular_jacobi": "structured_fblock_angular_jacobi",
        "structured_fblock_xi_angular": "structured_fblock_xi_angular_jacobi",
        "fblock_xi_angular_jacobi": "structured_fblock_xi_angular_jacobi",
        "structured_fblock_fp_radial": "structured_fblock_fp_radial_jacobi",
        "fblock_species_x_jacobi": "structured_fblock_fp_radial_jacobi",
        "structured_fblock_fp_lowmode_schur": "structured_fblock_fp_lowmode_schur",
        "fblock_fp_galerkin": "structured_fblock_fp_lowmode_schur",
        "structured_fblock_fp_moment_schur": "structured_fblock_fp_moment_schur",
        "fblock_fp_moment_galerkin": "structured_fblock_fp_moment_schur",
        "structured_fblock_fp_coupled_moment_schur": "structured_fblock_fp_coupled_moment_schur",
        "fblock_fp_coupled_galerkin": "structured_fblock_fp_coupled_moment_schur",
        "structured_fblock_fp_tail_coupled_schur": "structured_fblock_fp_tail_coupled_schur",
        "fblock_fp_tail_minres": "structured_fblock_fp_tail_coupled_schur",
        "unknown": None,
    }
    for raw, expected in cases.items():
        assert canonical_rhs1_preconditioner_kind(raw) == expected

    assert canonical_rhs1_preconditioner_kind(" THETA_ZETA ") == "theta_zeta"


def test_pas_auto_large_base_kind_respects_threshold(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_LITE_MIN", raising=False)
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_lite"
    assert rhs1_pas_auto_large_base_kind(active_size=5_000) == "pas_hybrid"

    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_MIN", "bad")
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_lite"

    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_MIN", "30_000")
    assert rhs1_pas_auto_large_base_kind(active_size=25_000) == "pas_hybrid"


def test_pas_weak_auto_override_promotes_default_weak_kinds(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_LITE_MIN", raising=False)

    assert (
        rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            current_kind="collision",
            active_size=500,
            n_theta=8,
            n_zeta=3,
            max_l=10,
        )
        == "xblock_tz"
    )

    assert (
        rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env="manual",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            current_kind="collision",
            active_size=500,
            n_theta=8,
            n_zeta=3,
            max_l=10,
        )
        == "collision"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "100")
    assert (
        rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            current_kind=None,
            active_size=30_000,
            n_theta=16,
            n_zeta=9,
            max_l=12,
        )
        == "pas_lite"
    )


def test_pas_weak_auto_override_rejects_measured_regression(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", raising=False)

    baseline = SolverCandidateMetrics(
        name="collision",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=0.2,
        solve_s=0.8,
        peak_rss_mb=400.0,
    )
    candidate = SolverCandidateMetrics(
        name="xblock_tz",
        residual_norm=1.0e-8,
        target=1.0e-9,
        setup_s=2.0,
        solve_s=7.0,
        peak_rss_mb=900.0,
    )

    assert (
        rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            current_kind="collision",
            active_size=500,
            n_theta=8,
            n_zeta=3,
            max_l=10,
            candidate_metrics=candidate,
            baseline_metrics=baseline,
        )
        == "collision"
    )


def test_measured_auto_promotion_gate_accepts_clean_runtime_win() -> None:
    baseline = SolverCandidateMetrics(
        name="collision",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=3.0,
        peak_rss_mb=800.0,
    )
    candidate = SolverCandidateMetrics(
        name="xblock_tz",
        residual_norm=2.0e-12,
        target=1.0e-9,
        setup_s=0.5,
        solve_s=1.0,
        peak_rss_mb=760.0,
    )

    gate = rhs1_measured_auto_promotion_gate(
        current_kind="collision",
        candidate_kind="xblock_tz",
        candidate_metrics=candidate,
        baseline_metrics=baseline,
    )

    assert gate.accepted
    assert rhs1_measured_auto_promotion_allowed(
        current_kind="collision",
        candidate_kind="xblock_tz",
        candidate_metrics=candidate,
        baseline_metrics=baseline,
    )


def test_measured_auto_promotion_preserves_unmeasured_historical_policy() -> None:
    gate = rhs1_measured_auto_promotion_gate(
        current_kind="collision",
        candidate_kind="xblock_tz",
    )

    assert gate.accepted
    assert gate.reasons == ("unmeasured_historical_policy",)


def test_pas_family_refinement_preserves_specialized_routing(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_ILU_MIN", raising=False)

    assert (
        rhs1_pas_family_refinement_kind(
            rhs1_precond_env="",
            has_pas=True,
            has_fp=False,
            current_kind="pas_lite",
            active_size=500,
            n_zeta=1,
            geom_scheme=1,
            pas_tz_applicable=False,
            pas_tokamak_theta_applicable=False,
        )
        == "pas_hybrid"
    )
    assert (
        rhs1_pas_family_refinement_kind(
            rhs1_precond_env="auto",
            has_pas=True,
            has_fp=False,
            current_kind="pas_hybrid",
            active_size=500,
            n_zeta=1,
            geom_scheme=1,
            pas_tz_applicable=True,
            pas_tokamak_theta_applicable=True,
        )
        == "pas_tokamak_theta"
    )
    assert (
        rhs1_pas_family_refinement_kind(
            rhs1_precond_env="auto",
            has_pas=True,
            has_fp=False,
            current_kind="pas_hybrid",
            active_size=500,
            n_zeta=7,
            geom_scheme=5,
            pas_tz_applicable=True,
            pas_tokamak_theta_applicable=False,
        )
        == "pas_tz"
    )
    assert (
        rhs1_pas_family_refinement_kind(
            rhs1_precond_env="",
            has_pas=True,
            has_fp=False,
            current_kind="pas_hybrid",
            active_size=20_000,
            n_zeta=3,
            geom_scheme=5,
            pas_tz_applicable=False,
            pas_tokamak_theta_applicable=False,
        )
        == "pas_ilu"
    )


def test_fp_dkes_env_override_only_promotes_bounded_xblock(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", raising=False)

    assert (
        rhs1_fp_dkes_env_preconditioner_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            use_dkes=True,
            total_size=500,
            n_theta=7,
            n_zeta=3,
            max_l=10,
        )
        == "xblock_tz"
    )
    assert (
        rhs1_fp_dkes_env_preconditioner_kind(
            rhs1_precond_env="theta_line",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            use_dkes=True,
            total_size=500,
            n_theta=7,
            n_zeta=3,
            max_l=10,
        )
        == "theta_line"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "100")
    assert (
        rhs1_fp_dkes_env_preconditioner_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            use_dkes=True,
            total_size=500,
            n_theta=7,
            n_zeta=3,
            max_l=10,
        )
        == ""
    )


def test_fp_dkes_default_kind_bounds_strong_preconditioners(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", raising=False)

    assert (
        rhs1_fp_dkes_default_kind(
            active_size=500,
            n_theta=7,
            n_zeta=3,
            max_l=10,
            xblock_tz_limit=500,
        )
        == "xblock_tz"
    )
    assert (
        rhs1_fp_dkes_default_kind(
            active_size=500,
            n_theta=7,
            n_zeta=3,
            max_l=10,
            xblock_tz_limit=100,
        )
        == "xmg"
    )
    assert (
        rhs1_fp_dkes_default_kind(
            active_size=50_000,
            n_theta=7,
            n_zeta=3,
            max_l=10,
            xblock_tz_limit=500,
        )
        == "collision"
    )


def test_large_fp_near_zero_er_override_forces_xmg_only_for_weak_kinds(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_FORCE_XMG_MIN", raising=False)

    assert (
        rhs1_large_fp_near_zero_er_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            has_pas=False,
            current_kind="collision",
            total_size=150_000,
            er_abs=0.0,
            schur_er_min=1.0e-12,
        )
        == "xmg"
    )
    assert (
        rhs1_large_fp_near_zero_er_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            has_pas=False,
            current_kind="schur",
            total_size=150_000,
            er_abs=0.0,
            schur_er_min=1.0e-12,
        )
        == "schur"
    )
    assert (
        rhs1_large_fp_near_zero_er_override_kind(
            rhs1_precond_env="",
            rhs_mode=1,
            include_phi1=False,
            has_fp=True,
            has_pas=False,
            current_kind="collision",
            total_size=150_000,
            er_abs=1.0e-8,
            schur_er_min=1.0e-12,
        )
        == "collision"
    )


def test_pas_auto_skip_strong_retry_requires_pas_auto_and_strong_base() -> None:
    assert pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="theta_line",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="pas_tz",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not pas_auto_skip_strong_retry(
        has_pas=False,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )


def test_dkes_xblock_policy_bounds_backend_and_block_size() -> None:
    assert rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert not rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="metal",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert not rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=17,
        n_zeta=23,
        max_l=36,
        xblock_tz_limit=2500,
    )


def test_dkes_cpu_pas_tz_policy_targets_large_cpu_angular_blocks(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_ACTIVE_MAX", raising=False)

    assert rhs1_pas_dkes_pas_tz_preferred(
        has_pas=True,
        use_dkes=True,
        backend="cpu",
        n_theta=5,
        n_zeta=15,
        max_l=20,
        active_size=7000,
    )
    assert rhs1_pas_dkes_pas_tz_preferred(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=5,
        n_zeta=15,
        max_l=20,
        active_size=7000,
    )
    assert rhs1_pas_dkes_cpu_pas_tz_preferred(
        has_pas=True,
        use_dkes=True,
        backend="cpu",
        n_theta=5,
        n_zeta=15,
        max_l=20,
        active_size=7000,
    )
    assert not rhs1_pas_dkes_pas_tz_preferred(
        has_pas=True,
        use_dkes=True,
        backend="cpu",
        n_theta=5,
        n_zeta=15,
        max_l=10,
        active_size=7000,
    )


def test_dkes_cpu_pas_tz_policy_env_bounds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN", "2000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_ACTIVE_MAX", "6000")

    kwargs = dict(
        has_pas=True,
        use_dkes=True,
        backend="cpu",
        n_theta=5,
        n_zeta=15,
        max_l=20,
    )
    assert not rhs1_pas_dkes_cpu_pas_tz_preferred(**kwargs, active_size=7000)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_CPU_PAS_TZ_MIN", "1200")
    assert rhs1_pas_dkes_cpu_pas_tz_preferred(**kwargs, active_size=6000)


def test_dkes_gpu_pas_tz_policy_uses_gpu_env_bounds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_MIN", "2000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_ACTIVE_MAX", "6000")

    kwargs = dict(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=5,
        n_zeta=15,
        max_l=20,
    )
    assert not rhs1_pas_dkes_pas_tz_preferred(**kwargs, active_size=7000)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_PAS_TZ_MIN", "950")
    assert rhs1_pas_dkes_pas_tz_preferred(**kwargs, active_size=6000)


def test_full_cpu_pas_tz_policy_targets_bounded_geometry11_cpu_cases(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_NZETA_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_ACTIVE_MAX", raising=False)

    assert rhs1_pas_full_cpu_pas_tz_preferred(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="cpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=15,
        max_l=20,
        active_size=7000,
        pas_tz_applicable=True,
    )
    assert not rhs1_pas_full_cpu_pas_tz_preferred(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="gpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=15,
        max_l=20,
        active_size=7000,
        pas_tz_applicable=True,
    )
    assert rhs1_pas_full_cpu_pas_tz_preferred(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="cpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=19,
        max_l=20,
        active_size=7000,
        pas_tz_applicable=True,
    )
    assert not rhs1_pas_full_cpu_pas_tz_preferred(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="cpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=23,
        max_l=20,
        active_size=7000,
        pas_tz_applicable=True,
    )
    assert not rhs1_pas_full_cpu_pas_tz_preferred(
        has_pas=True,
        has_fp=False,
        use_dkes=True,
        backend="cpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=15,
        max_l=20,
        active_size=7000,
        pas_tz_applicable=True,
    )


def test_full_cpu_pas_tz_policy_env_bounds(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_NZETA_MAX", "20")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_MIN", "2000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_ACTIVE_MAX", "6000")

    kwargs = dict(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="cpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=15,
        max_l=20,
        pas_tz_applicable=True,
    )
    assert not rhs1_pas_full_cpu_pas_tz_preferred(**kwargs, active_size=7000)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_MIN", "950")
    assert rhs1_pas_full_cpu_pas_tz_preferred(**kwargs, active_size=6000)


def test_full_gpu_pas_tz_policy_targets_bounded_geometry11_gpu_cases(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_NZETA_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_ACTIVE_MAX", raising=False)

    common = dict(
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        backend="gpu",
        geom_scheme=11,
        n_theta=6,
        n_zeta=19,
        max_l=20,
        active_size=4526,
        pas_tz_applicable=True,
    )
    assert rhs1_pas_full_pas_tz_preferred(**common)
    assert not rhs1_pas_full_pas_tz_preferred(**{**common, "geom_scheme": 1})
    assert not rhs1_pas_full_pas_tz_preferred(**{**common, "n_zeta": 23})

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_ACTIVE_MAX", "4000")
    assert not rhs1_pas_full_pas_tz_preferred(**common)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_ACTIVE_MAX", "15000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ", "0")
    assert not rhs1_pas_full_pas_tz_preferred(**common)


def test_geometry4_pas_memory_pas_tz_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MAX", raising=False)
    common = dict(
        rhs1_precond_env="",
        current_kind="schur",
        has_pas=True,
        has_fp=False,
        use_dkes=False,
        geom_scheme=4,
        n_theta=8,
        n_zeta=11,
        max_l=25,
        active_size=10898,
        er_abs=0.0,
        schur_er_min=1.0e-12,
        pas_tz_applicable=True,
    )
    assert rhs1_geometry4_pas_memory_pas_tz_preferred(**common)
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**{**common, "current_kind": "pas_tz"})
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**{**common, "rhs1_precond_env": "schur"})
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**{**common, "geom_scheme": 11})
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**{**common, "active_size": 1000})
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**{**common, "er_abs": 1.0e-3})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ", "0")
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**common)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MAX", "9000")
    assert not rhs1_geometry4_pas_memory_pas_tz_preferred(**common)


def test_tokamak_pas_gpu_and_cpu_xblock_policies(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_CPU_XBLOCK_ACTIVE_MAX", raising=False)

    common = dict(
        has_pas=True,
        has_fp=False,
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )
    assert rhs1_pas_tokamak_gpu_theta_allowed(backend="gpu", **common)
    assert not rhs1_pas_tokamak_gpu_xblock_preferred(
        backend="gpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    medium_case = {**common, "active_size": 2650}
    assert rhs1_pas_tokamak_gpu_xblock_preferred(
        backend="gpu",
        n_theta=15,
        n_zeta=1,
        max_l=31,
        xblock_tz_limit=1200,
        **medium_case,
    )
    assert not rhs1_pas_tokamak_gpu_xblock_preferred(
        backend="cpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    assert rhs1_pas_tokamak_cpu_xblock_preferred(
        backend="cpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )
    assert not rhs1_pas_tokamak_cpu_xblock_preferred(
        backend="gpu",
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
        **common,
    )


def test_tokamak_pas_gpu_tight_tol_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_TOL", raising=False)
    common = dict(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=500,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )
    assert rhs1_pas_tokamak_gpu_tight_tol(enabled=True, **common) == 1.0e-8
    assert rhs1_pas_tokamak_gpu_tight_tol(enabled=False, **common) is None
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL", "1e-9")
    assert rhs1_pas_tokamak_gpu_tight_tol(enabled=True, **common) == 1.0e-9
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL", "0")
    assert rhs1_pas_tokamak_gpu_tight_tol(enabled=True, **common) is None
    assert rhs1_pas_tokamak_gpu_tight_tol(
        enabled=True,
        **{**common, "backend": "cpu"},
    ) is None


def test_gpu_sparse_fallback_skip_policy(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", raising=False)
    assert rhs1_gpu_sparse_fallback_skip_allowed(
        backend="gpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )
    assert not rhs1_gpu_sparse_fallback_skip_allowed(
        backend="cpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", "0")
    assert not rhs1_gpu_sparse_fallback_skip_allowed(
        backend="gpu",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=5.0,
        target=1.0,
    )


def test_sharded_line_override_policy_preserves_dedicated_pas_preconditioners() -> None:
    assert rhs1_sharded_line_override_allowed(None)
    assert rhs1_sharded_line_override_allowed("theta_line")
    assert rhs1_sharded_line_override_allowed("pas_hybrid")
    assert not rhs1_sharded_line_override_allowed("schur")
    assert not rhs1_sharded_line_override_allowed("pas_tz")
    assert not rhs1_sharded_line_override_allowed("pas_tokamak_theta")
    assert not rhs1_sharded_line_override_allowed("pas_ilu")


def _default_selection_op(
    *,
    has_fp: bool = False,
    has_pas: bool = False,
    rhs_mode: int = 1,
    include_phi1: bool = False,
    constraint_scheme: int = 2,
    extra_size: int = 2,
    n_theta: int = 8,
    n_zeta: int = 1,
    total_size: int = 500,
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=include_phi1,
        constraint_scheme=constraint_scheme,
        extra_size=extra_size,
        total_size=total_size,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_species=1,
        n_x=2,
        fblock=SimpleNamespace(
            fp=object() if has_fp else None,
            pas=object() if has_pas else None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([6, 8], dtype=np.int32)),
            magdrift_theta=None,
            magdrift_zeta=None,
            magdrift_xidot=None,
        ),
    )


def _default_selection_context(**updates) -> RHS1DefaultPreconditionerSelectionContext:
    op = updates.pop("op", _default_selection_op())
    physics = updates.pop("physics", {})
    values = {
        "_canonical_rhs1_preconditioner_kind": canonical_rhs1_preconditioner_kind,
        "_matvec_shard_axis": lambda _op: None,
        "_pas_tz_preconditioner_applicable": lambda _op: True,
        "_rhs1_fp_dkes_default_kind": lambda **_kwargs: "fp_dkes_default",
        "_rhs1_pas_auto_large_base_kind": rhs1_pas_auto_large_base_kind,
        "_rhs1_pas_dkes_pas_tz_preferred": lambda **_kwargs: False,
        "_rhs1_pas_dkes_xblock_allowed": lambda **_kwargs: False,
        "_rhs1_pas_small_near_zero_er_kind": lambda **_kwargs: "pas_small_near_zero",
        "_rhs1_pas_tokamak_gpu_theta_allowed": lambda **_kwargs: False,
        "_rhs1_pas_tokamak_gpu_xblock_preferred": lambda **_kwargs: False,
        "active_size": 500,
        "emit": None,
        "full_precond_requested": False,
        "geom_scheme": 2,
        "jax": SimpleNamespace(default_backend=lambda: "cpu", device_count=lambda: 1),
        "nml": SimpleNamespace(group=lambda name: physics if name.lower() == "physicsparameters" else {}),
        "np": np,
        "op": op,
        "os": __import__("os"),
        "pre_theta": 0,
        "pre_zeta": 0,
        "er_abs": 0.0,
        "rhs1_gpu_tokamak_pas_tight_gmres": False,
        "rhs1_precond_env": "",
        "rhs1_xblock_tz_lmax": 0,
        "schur_er_min": 1.0e-12,
        "use_dkes": False,
    }
    values.update(updates)
    return RHS1DefaultPreconditionerSelectionContext(values=values)


def test_default_preconditioner_selection_honors_explicit_and_theta_zeta_controls() -> None:
    explicit = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(rhs1_precond_env="theta")
    )
    assert explicit["rhs1_precond_kind"] == "theta_line"

    adi = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(pre_theta=1, pre_zeta=1)
    )
    assert adi["rhs1_precond_kind"] == "adi"

    theta = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(pre_theta=1, pre_zeta=0)
    )
    assert theta["rhs1_precond_kind"] == "theta_line"

    zeta = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(pre_theta=0, pre_zeta=1)
    )
    assert zeta["rhs1_precond_kind"] == "zeta_line"


def test_default_preconditioner_selection_prefers_xmg_for_moderate_fp_near_zero_er() -> None:
    result = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(
            op=_default_selection_op(has_fp=True, has_pas=False, n_theta=7, n_zeta=5, total_size=2000),
            active_size=2000,
        )
    )

    assert result["rhs1_precond_kind"] == "xmg"
    assert result["er_abs"] == 0.0
    assert result["max_l"] == 8


def test_default_preconditioner_selection_bounds_tokamak_pas_full_preconditioner_by_size() -> None:
    small = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(
            op=_default_selection_op(has_pas=True, total_size=500, n_theta=8, n_zeta=1),
            full_precond_requested=True,
            geom_scheme=1,
        )
    )
    large = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(
            op=_default_selection_op(has_pas=True, total_size=9000, n_theta=8, n_zeta=1),
            active_size=9000,
            full_precond_requested=True,
            geom_scheme=1,
        )
    )

    assert small["rhs1_precond_kind"] == "xblock_tz"
    assert small["tokamak_like"] is True
    assert large["rhs1_precond_kind"] == "pas_hybrid"


def test_default_preconditioner_selection_delegates_fp_dkes_default() -> None:
    result = resolve_rhs1_default_preconditioner_selection(
        _default_selection_context(
            op=_default_selection_op(has_fp=True, n_theta=7, n_zeta=5, total_size=3000),
            active_size=3000,
            use_dkes=True,
        )
    )

    assert result["rhs1_precond_kind"] == "fp_dkes_default"
