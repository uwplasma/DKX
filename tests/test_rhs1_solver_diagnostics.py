from __future__ import annotations

from sfincs_jax.rhs1_solver_diagnostics import (
    RHS1PostMinresDiagnostics,
    RHS1PreflightDiagnostics,
    RHS1SubspaceCorrectionDiagnostics,
    build_rhs1_xblock_correction_metadata,
)


def test_rhs1_xblock_correction_metadata_preserves_historical_keys() -> None:
    metadata = build_rhs1_xblock_correction_metadata(
        probe_coarse=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=2,
            direction_counts=(3, 4),
            direction_names=("fsavg", "angular"),
            residual_before=5.0,
            residual_after=2.0,
            history=(4.0, 2.0),
            fsavg_lmax=3,
            angular_lmax=2,
            angular_residual=True,
            seed_initialized=True,
            setup_s=0.125,
        ),
        preflight=RHS1PreflightDiagnostics(
            min_improvement=1.0e-3,
            required=True,
            residual_norm=3.0,
            improvement=0.2,
            passed=True,
        ),
        post_minres=RHS1PostMinresDiagnostics(
            steps_requested=3,
            alphas=(0.5, -0.25),
            history=(1.7, 1.2),
            residual_before=2.0,
            residual_after=1.2,
        ),
        post_coarse=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=4,
            direction_counts=(2,),
            direction_names=("post-fsavg",),
            residual_before=1.2,
            residual_after=0.8,
            history=(1.0, 0.8),
            fsavg_lmax=1,
            angular_lmax=0,
            angular_residual=False,
        ),
        post_residual_equation=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=1,
            direction_counts=(5, 6),
            direction_names=("qi", "residual"),
            residual_before=0.8,
            residual_after=0.3,
            history=(0.6, 0.3),
            fsavg_lmax=4,
            angular_lmax=3,
            angular_residual=True,
            include_qi_basis=True,
        ),
    )

    assert metadata["xblock_probe_coarse_steps_requested"] == 2
    assert metadata["xblock_probe_coarse_steps_accepted"] == 2
    assert metadata["xblock_probe_coarse_direction_count"] == 7
    assert metadata["xblock_probe_coarse_residual_before"] == 5.0
    assert metadata["xblock_probe_coarse_residual_after"] == 2.0
    assert metadata["xblock_probe_coarse_seed_initialized"] is True
    assert metadata["xblock_probe_coarse_s"] == 0.125
    assert metadata["xblock_probe_coarse_history"] == (4.0, 2.0)
    assert metadata["xblock_probe_coarse_direction_counts"] == (3, 4)
    assert metadata["xblock_probe_coarse_direction_names"] == ("fsavg", "angular")
    assert metadata["xblock_probe_coarse_fsavg_lmax"] == 3
    assert metadata["xblock_probe_coarse_angular_lmax"] == 2
    assert metadata["xblock_probe_coarse_angular_residual"] is True

    assert metadata["xblock_preflight_min_improvement"] == 1.0e-3
    assert metadata["xblock_preflight_required"] is True
    assert metadata["xblock_preflight_residual_norm"] == 3.0
    assert metadata["xblock_preflight_improvement"] == 0.2
    assert metadata["xblock_preflight_passed"] is True

    assert metadata["xblock_post_minres_steps_requested"] == 3
    assert metadata["xblock_post_minres_steps_accepted"] == 2
    assert metadata["xblock_post_minres_residual_before"] == 2.0
    assert metadata["xblock_post_minres_residual_after"] == 1.2
    assert metadata["xblock_post_minres_alphas"] == (0.5, -0.25)
    assert metadata["xblock_post_minres_history"] == (1.7, 1.2)

    assert metadata["xblock_post_coarse_steps_requested"] == 4
    assert metadata["xblock_post_coarse_steps_accepted"] == 1
    assert metadata["xblock_post_coarse_direction_count"] == 2
    assert metadata["xblock_post_coarse_residual_before"] == 1.2
    assert metadata["xblock_post_coarse_residual_after"] == 0.8
    assert metadata["xblock_post_coarse_history"] == (1.0, 0.8)
    assert metadata["xblock_post_coarse_direction_counts"] == (2,)
    assert metadata["xblock_post_coarse_direction_names"] == ("post-fsavg",)
    assert metadata["xblock_post_coarse_fsavg_lmax"] == 1
    assert metadata["xblock_post_coarse_angular_lmax"] == 0
    assert metadata["xblock_post_coarse_angular_residual"] is False

    assert metadata["xblock_post_residual_equation_steps_requested"] == 1
    assert metadata["xblock_post_residual_equation_steps_accepted"] == 2
    assert metadata["xblock_post_residual_equation_direction_count"] == 11
    assert metadata["xblock_post_residual_equation_residual_before"] == 0.8
    assert metadata["xblock_post_residual_equation_residual_after"] == 0.3
    assert metadata["xblock_post_residual_equation_history"] == (0.6, 0.3)
    assert metadata["xblock_post_residual_equation_direction_counts"] == (5, 6)
    assert metadata["xblock_post_residual_equation_direction_names"] == (
        "qi",
        "residual",
    )
    assert metadata["xblock_post_residual_equation_fsavg_lmax"] == 4
    assert metadata["xblock_post_residual_equation_angular_lmax"] == 3
    assert metadata["xblock_post_residual_equation_angular_residual"] is True
    assert metadata["xblock_post_residual_equation_include_qi_basis"] is True


def test_rhs1_xblock_correction_metadata_defaults_are_output_safe() -> None:
    metadata = build_rhs1_xblock_correction_metadata(
        probe_coarse=RHS1SubspaceCorrectionDiagnostics(steps_requested=0),
        preflight=RHS1PreflightDiagnostics(min_improvement=0.0, required=False),
        post_minres=RHS1PostMinresDiagnostics(steps_requested=0),
        post_coarse=RHS1SubspaceCorrectionDiagnostics(steps_requested=0),
        post_residual_equation=RHS1SubspaceCorrectionDiagnostics(steps_requested=0),
    )

    assert metadata["xblock_probe_coarse_steps_accepted"] == 0
    assert metadata["xblock_probe_coarse_direction_count"] == 0
    assert metadata["xblock_probe_coarse_seed_initialized"] is False
    assert metadata["xblock_probe_coarse_s"] == 0.0
    assert metadata["xblock_probe_coarse_history"] == ()
    assert metadata["xblock_preflight_required"] is False
    assert metadata["xblock_preflight_passed"] is None
    assert metadata["xblock_post_minres_steps_accepted"] == 0
    assert metadata["xblock_post_minres_alphas"] == ()
    assert metadata["xblock_post_coarse_direction_names"] == ()
    assert metadata["xblock_post_residual_equation_direction_count"] == 0
    assert metadata["xblock_post_residual_equation_include_qi_basis"] is False
