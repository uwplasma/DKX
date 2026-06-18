from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.profile_response.diagnostics import (
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockSparsePCCoreDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    fp_xblock_global_correction_metadata,
    fp_xblock_highx_residual_correction_metadata,
    sparse_pc_direct_tail_result_metadata,
    sparse_rescue_tail_metadata,
    sparse_xblock_rescue_metadata,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_device_krylov_diagnostics,
    xblock_qi_deflated_preconditioner_diagnostics,
    xblock_qi_device_preconditioner_diagnostics,
    xblock_qi_seed_preconditioner_diagnostics,
    xblock_sparse_pc_core_diagnostics,
    xblock_side_probe_diagnostics,
)


def _sparse_rescue_scope() -> dict[str, object]:
    return {
        "sparse_xblock_rescue_active": 1,
        "sparse_xblock_rescue_attempted": True,
        "sparse_xblock_rescue_built": False,
        "sparse_xblock_rescue_error": None,
        "sparse_xblock_rescue_reason": "seed_rejected",
        "sparse_xblock_rescue_assembled_host_fp": True,
        "sparse_xblock_rescue_preconditioner_xi": 2,
        "sparse_xblock_rescue_seed_residual": 1.25,
        "sparse_xblock_rescue_seed_improvement_ratio": 4.0,
        "sparse_xblock_rescue_seed_accept_ratio": 0.2,
        "sparse_xblock_rescue_seed_refine_steps": 3,
        "sparse_xblock_rescue_seed_refines_performed": 2,
        "sparse_xblock_rescue_candidate_residual": 0.5,
        "sparse_xblock_rescue_candidate_accepted": 1,
        "fp_xblock_global_correction_allowed": True,
        "fp_xblock_global_correction_attempted": True,
        "fp_xblock_global_correction_accepted": False,
        "fp_xblock_global_correction_reason": "no_improvement",
        "fp_xblock_global_correction_error": "none",
        "fp_xblock_global_correction_preconditioner": "xblock",
        "fp_xblock_global_correction_steps": 4,
        "fp_xblock_global_correction_accepted_steps": 1,
        "fp_xblock_global_correction_residual_before": 2.0,
        "fp_xblock_global_correction_residual_after": 1.5,
        "fp_xblock_global_correction_improvement_ratio": 1.25,
        "fp_xblock_global_correction_elapsed_s": 0.125,
        "fp_xblock_highx_residual_correction_allowed": True,
        "fp_xblock_highx_residual_correction_attempted": False,
        "fp_xblock_highx_residual_correction_accepted": False,
        "fp_xblock_highx_residual_correction_reason": "policy_guard",
        "fp_xblock_highx_residual_correction_error": None,
        "fp_xblock_highx_residual_correction_residual_before": None,
        "fp_xblock_highx_residual_correction_residual_after": None,
        "fp_xblock_highx_residual_correction_improvement_ratio": None,
        "fp_xblock_highx_residual_correction_elapsed_s": None,
        "fp_xblock_highx_residual_correction_direction_count": 0,
        "fp_xblock_highx_residual_correction_direction_names": ["a", "b"],
    }


def test_sparse_rescue_metadata_helpers_preserve_driver_keys_and_types() -> None:
    scope = _sparse_rescue_scope()

    sparse_meta = sparse_xblock_rescue_metadata(scope)
    global_meta = fp_xblock_global_correction_metadata(scope)
    highx_meta = fp_xblock_highx_residual_correction_metadata(scope)
    combined = sparse_rescue_tail_metadata(scope)

    assert sparse_meta["sparse_xblock_rescue_active"] is True
    assert sparse_meta["sparse_xblock_rescue_reason"] == "seed_rejected"
    assert sparse_meta["sparse_xblock_rescue_candidate_accepted"] is True
    assert global_meta["fp_xblock_global_correction_reason"] == "no_improvement"
    assert global_meta["fp_xblock_global_correction_accepted"] is False
    assert highx_meta["fp_xblock_highx_residual_correction_allowed"] is True
    assert highx_meta["fp_xblock_highx_residual_correction_direction_names"] == (
        "a",
        "b",
    )
    assert combined == {**sparse_meta, **global_meta, **highx_meta}


def _qi_device_preconditioner_scope() -> dict[str, object]:
    return {
        "qi_device_preconditioner_enabled": 1,
        "qi_device_preconditioner_built": True,
        "qi_device_preconditioner_used": False,
        "qi_device_preconditioner_used_in_krylov": True,
        "qi_device_preconditioner_reason": "probe_reject",
        "qi_device_preconditioner_rank": 7,
        "qi_device_preconditioner_candidate_count": 11,
        "qi_device_preconditioner_coarse_shape": (7, 7),
        "qi_device_preconditioner_operator_on_basis_shape": (13, 7),
        "qi_device_preconditioner_coarse_norm": 2.5,
        "qi_device_preconditioner_operator_on_basis_norm": 3.5,
        "qi_device_preconditioner_residual_before": 4.0,
        "qi_device_preconditioner_residual_after": 1.0,
        "qi_device_preconditioner_improvement_ratio": 4.0,
        "qi_device_preconditioner_setup_s": 0.25,
        "qi_device_preconditioner_min_improvement": 0.05,
        "qi_device_preconditioner_use_in_krylov": True,
        "qi_device_augmented_krylov_requested": True,
        "qi_device_augmented_krylov_used": False,
        "qi_device_augmented_krylov_rank": 3,
        "qi_device_augmented_krylov_reason": "seed_only",
        "qi_device_augmented_krylov_mode": "right",
        "qi_device_augmented_seed_requested": True,
        "qi_device_augmented_seed_available": True,
        "qi_device_augmented_seed_used": False,
        "qi_device_augmented_seed_rank": 2,
        "qi_device_augmented_seed_max_rank": 5,
        "qi_device_augmented_seed_reason": "accepted",
        "qi_device_augmented_seed_projection_residual": 1.0e-3,
        "qi_device_augmented_seed_labels": ("constant", "current"),
        "qi_device_stats": {"applies": 9},
        "qi_device_preconditioner_metadata": {
            "operator_krylov_enrichment_enabled": True,
            "multilevel_coarse_enabled": True,
            "global_moment_residual_equation_enabled": True,
            "global_moment_residual_equation_solver": "galerkin",
            "global_moment_residual_equation_rank": 4,
            "global_moment_residual_equation_candidate_count": 6,
            "global_moment_residual_equation_condition_estimate": 12.0,
            "phase_space_residual_equation_enabled": True,
            "phase_space_residual_equation_max_rank_requested": 8,
            "phase_space_residual_equation_solver": "action_lstsq",
            "phase_space_residual_equation_rank": 5,
            "phase_space_residual_equation_candidate_count": 10,
            "phase_space_residual_equation_stage_count": 1,
            "phase_space_residual_equation_condition_estimate": 6.0,
            "phase_space_residual_equation_include_global": True,
            "phase_space_residual_equation_trapped_boundary_fraction": 0.25,
            "residual_region_bounce_coarse_enabled": True,
            "residual_region_bounce_coarse_max_rank": 9,
            "residual_region_bounce_coarse_rank": 3,
            "residual_region_bounce_coarse_candidate_count": 4,
            "active_pattern_coarse_enabled": True,
            "active_pattern_coarse_max_rank_requested": 12,
            "active_pattern_coarse_candidate_count": 7,
            "coupled_residual_equation_enabled": True,
            "coupled_residual_equation_max_rank_requested": 14,
            "coupled_residual_equation_rank": 6,
            "coupled_residual_equation_candidate_count": 16,
            "coupled_residual_equation_source_stage_ranks": (1, 2, 3),
            "coupled_residual_equation_solver": "action_lstsq",
            "coupled_residual_equation_include_flat": True,
            "coupled_residual_equation_min_relative_improvement_requested": 0.1,
            "coupled_residual_equation_install_in_krylov_on_reject_requested": True,
            "seed_probe_accepted": False,
            "installed_in_krylov_after_seed_reject": True,
            "coupled_residual_equation_accepted": False,
            "coupled_residual_equation_reason": "insufficient_improvement",
            "block_schur_residual_enrichment_enabled": True,
        },
    }


def test_xblock_qi_device_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_device_preconditioner_diagnostics(
        _qi_device_preconditioner_scope()
    )

    assert metadata["xblock_qi_device_preconditioner_enabled"] is True
    assert metadata["xblock_qi_device_preconditioner_used"] is False
    assert metadata["xblock_qi_device_preconditioner_rank"] == 7
    assert metadata["xblock_qi_device_preconditioner_applies"] == 9
    assert metadata["xblock_qi_device_preconditioner_augmented_seed_labels"] == (
        "constant",
        "current",
    )
    assert (
        metadata["xblock_qi_device_preconditioner_global_moment_residual_equation"]
        is True
    )
    assert (
        metadata["xblock_qi_device_preconditioner_global_moment_residual_equation_rank"]
        == 4
    )
    assert (
        metadata[
            "xblock_qi_device_preconditioner_phase_space_residual_equation_max_rank"
        ]
        == 8
    )
    assert (
        metadata[
            "xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank"
        ]
        == 9
    )
    assert (
        metadata["xblock_qi_device_preconditioner_active_pattern_coarse_max_rank"] == 12
    )
    assert (
        metadata["xblock_qi_device_preconditioner_coupled_residual_equation_rank"] == 6
    )
    assert (
        metadata[
            "xblock_qi_device_preconditioner_coupled_residual_equation_install_in_krylov_on_reject"
        ]
        is True
    )
    assert (
        metadata["xblock_qi_device_preconditioner_block_schur_residual_enrichment"]
        is True
    )


def _qi_deflated_preconditioner_scope() -> dict[str, object]:
    return {
        "qi_deflated_preconditioner_enabled": 1,
        "qi_deflated_preconditioner_built": True,
        "qi_deflated_preconditioner_used": False,
        "qi_deflated_preconditioner_used_in_krylov": True,
        "qi_deflated_preconditioner_reason": "seed_rejected",
        "qi_deflated_preconditioner_rank": 5,
        "qi_deflated_preconditioner_candidate_count": 8,
        "qi_deflated_preconditioner_residual_before": 3.0,
        "qi_deflated_preconditioner_residual_after": 1.5,
        "qi_deflated_preconditioner_improvement_ratio": 2.0,
        "qi_deflated_preconditioner_setup_s": 0.125,
        "qi_deflated_stats": {"applies": 4, "local_applies": 6},
        "qi_deflated_preconditioner_metadata": {
            "correction_cycles": 3,
            "seed_solver": "minres",
            "cycle_residual_history": (3.0, 2.0, 1.5),
            "cycle_coefficients": (0.25, 0.5),
        },
    }


def test_xblock_qi_deflated_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_deflated_preconditioner_diagnostics(
        _qi_deflated_preconditioner_scope()
    )

    assert metadata["xblock_qi_deflated_preconditioner_enabled"] is True
    assert metadata["xblock_qi_deflated_preconditioner_used"] is False
    assert metadata["xblock_qi_deflated_preconditioner_use_in_krylov"] is True
    assert metadata["xblock_qi_deflated_preconditioner_rank"] == 5
    assert metadata["xblock_qi_deflated_preconditioner_candidate_count"] == 8
    assert metadata["xblock_qi_deflated_preconditioner_setup_s"] == 0.125
    assert metadata["xblock_qi_deflated_preconditioner_applies"] == 4
    assert metadata["xblock_qi_deflated_preconditioner_local_applies"] == 6
    assert metadata["xblock_qi_deflated_preconditioner_cycles"] == 3
    assert metadata["xblock_qi_deflated_preconditioner_seed_solver"] == "minres"
    assert metadata["xblock_qi_deflated_preconditioner_cycle_residual_history"] == (
        3.0,
        2.0,
        1.5,
    )
    assert metadata["xblock_qi_deflated_preconditioner_cycle_coefficients"] == (
        0.25,
        0.5,
    )


def _xblock_side_probe_scope() -> dict[str, object]:
    return {
        "xblock_side_probe_enabled": 1,
        "xblock_side_probe_used": True,
        "xblock_side_probe_switched": False,
        "xblock_side_probe_switch_suppressed_by_global_coupling": True,
        "xblock_side_probe_switch_suppressed_by_explicit_side": False,
        "xblock_side_probe_physical_seed_preserved_after_switch": True,
        "xblock_side_probe_seed_used": 1,
        "xblock_side_probe_seed_residual_norm": 1.0e-4,
        "xblock_side_probe_initial_side": "left",
        "xblock_side_probe_selected_side": "right",
        "xblock_side_probe_initial_method": "gmres",
        "xblock_side_probe_selected_method": "lgmres",
        "xblock_side_probe_lgmres_rescue": True,
        "xblock_lgmres_rescue_maxiter_capped": 1,
        "xblock_lgmres_rescue_outer_k": 12,
        "xblock_side_probe_residual_norm": 2.0e-5,
        "xblock_side_probe_residual_ratio": 3.5,
        "xblock_side_probe_iterations": np.int64(9),
        "xblock_side_probe_matvecs": np.int64(14),
        "xblock_side_probe_s": np.float64(0.375),
    }


def test_xblock_side_probe_diagnostics_preserve_payload() -> None:
    scope = _xblock_side_probe_scope()
    metadata = xblock_side_probe_diagnostics(
        XBlockSideProbeDiagnosticsContext(
            enabled=scope["xblock_side_probe_enabled"],
            used=scope["xblock_side_probe_used"],
            switched=scope["xblock_side_probe_switched"],
            switch_suppressed_by_global_coupling=scope[
                "xblock_side_probe_switch_suppressed_by_global_coupling"
            ],
            switch_suppressed_by_explicit_side=scope[
                "xblock_side_probe_switch_suppressed_by_explicit_side"
            ],
            physical_seed_preserved_after_switch=scope[
                "xblock_side_probe_physical_seed_preserved_after_switch"
            ],
            seed_used=scope["xblock_side_probe_seed_used"],
            seed_residual_norm=scope["xblock_side_probe_seed_residual_norm"],
            initial_side=scope["xblock_side_probe_initial_side"],
            selected_side=scope["xblock_side_probe_selected_side"],
            initial_method=scope["xblock_side_probe_initial_method"],
            selected_method=scope["xblock_side_probe_selected_method"],
            lgmres_rescue=scope["xblock_side_probe_lgmres_rescue"],
            lgmres_rescue_maxiter_capped=scope[
                "xblock_lgmres_rescue_maxiter_capped"
            ],
            lgmres_rescue_outer_k=scope["xblock_lgmres_rescue_outer_k"],
            residual_norm=scope["xblock_side_probe_residual_norm"],
            residual_ratio=scope["xblock_side_probe_residual_ratio"],
            iterations=scope["xblock_side_probe_iterations"],
            matvecs=scope["xblock_side_probe_matvecs"],
            elapsed_s=scope["xblock_side_probe_s"],
        )
    )

    assert metadata["xblock_side_probe_enabled"] is True
    assert metadata["xblock_side_probe_switched"] is False
    assert metadata["xblock_side_probe_switch_suppressed_by_global_coupling"] is True
    assert metadata["xblock_side_probe_physical_seed_preserved_after_switch"] is True
    assert metadata["xblock_side_probe_seed_used"] is True
    assert metadata["xblock_side_probe_initial_side"] == "left"
    assert metadata["xblock_side_probe_selected_side"] == "right"
    assert metadata["xblock_side_probe_selected_method"] == "lgmres"
    assert metadata["xblock_side_probe_lgmres_rescue"] is True
    assert metadata["xblock_lgmres_rescue_maxiter_capped"] is True
    assert metadata["xblock_lgmres_rescue_outer_k"] == 12
    assert metadata["xblock_side_probe_residual_ratio"] == 3.5
    assert metadata["xblock_side_probe_iterations"] == 9
    assert metadata["xblock_side_probe_matvecs"] == 14
    assert metadata["xblock_side_probe_s"] == 0.375


def test_xblock_assembled_operator_diagnostics_preserve_payload() -> None:
    metadata = xblock_assembled_operator_diagnostics(
        XBlockAssembledOperatorDiagnosticsContext(
            enabled=1,
            built=True,
            metadata={
                "active_dof": True,
                "preflight_scope": "active",
                "setup_s": 1.25,
                "preflight_rejected": False,
                "preflight_pattern_nnz_estimate": 50,
                "preflight_peak_nbytes_estimate": 2048,
                "preflight_full_csr_nbytes_estimate": 4096,
                "preflight_active_csr_nbytes_estimate": 1024,
                "pattern_nnz": 45,
                "matrix_nnz": 43,
                "csr_nbytes_estimate": 512,
                "device_enabled": True,
                "device_required": False,
                "device_resident": True,
                "device_nnz": 43,
                "device_csr_nbytes_estimate": 400,
                "device_validation_rel_errors": (1.0e-12,),
                "device_error": None,
                "max_colors": 7,
                "validation_rel_errors": (2.0e-12,),
                "error": None,
            },
            row_equilibration_enabled=1,
            row_equilibration_built=True,
            row_equilibration_metadata={
                "norm": "inf",
                "setup_s": 0.1,
                "zero_or_tiny_rows": 2,
                "row_norm_min": 0.5,
                "row_norm_max": 4.0,
                "row_scale_min": 0.25,
                "row_scale_max": 2.0,
            },
            col_equilibration_enabled=1,
            col_equilibration_built=False,
            col_equilibration_metadata={
                "norm": "two",
                "setup_s": 0.2,
                "zero_or_tiny_columns": 3,
                "col_norm_min": 0.25,
                "col_norm_max": 5.0,
                "col_scale_min": 0.2,
                "col_scale_max": 4.0,
            },
        )
    )

    assert metadata["xblock_assembled_operator_enabled"] is True
    assert metadata["xblock_assembled_operator_built"] is True
    assert metadata["xblock_assembled_operator_active_dof"] is True
    assert metadata["xblock_assembled_operator_preflight_scope"] == "active"
    assert metadata["xblock_assembled_operator_preflight_pattern_nnz_estimate"] == 50
    assert metadata["xblock_assembled_operator_matrix_nnz"] == 43
    assert metadata["xblock_assembled_operator_device_enabled"] is True
    assert metadata["xblock_assembled_operator_device_resident"] is True
    assert metadata["xblock_assembled_operator_device_validation_rel_errors"] == (
        1.0e-12,
    )
    assert metadata["xblock_assembled_operator_row_equilibration_enabled"] is True
    assert metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert metadata["xblock_assembled_operator_row_equilibration_norm"] == "inf"
    assert (
        metadata["xblock_assembled_operator_row_equilibration_zero_or_tiny_rows"] == 2
    )
    assert metadata["xblock_assembled_operator_col_equilibration_enabled"] is True
    assert metadata["xblock_assembled_operator_col_equilibration_built"] is False
    assert metadata["xblock_assembled_operator_col_equilibration_norm"] == "two"
    assert (
        metadata["xblock_assembled_operator_col_equilibration_zero_or_tiny_columns"]
        == 3
    )
    assert metadata["xblock_assembled_operator_max_colors"] == 7
    assert metadata["xblock_assembled_operator_validation_rel_errors"] == (2.0e-12,)


def test_xblock_coarse_correction_diagnostics_preserve_payload() -> None:
    metadata = xblock_coarse_correction_diagnostics(
        {
            "moment_schur_enabled": 1,
            "moment_schur_built": True,
            "moment_schur_used": False,
            "moment_schur_reason": "compact_factor_guard",
            "moment_schur_default_blocked_by_compact_factors": 1,
            "moment_schur_probe_residual_before": 2.0,
            "moment_schur_probe_residual_after": 0.5,
            "moment_schur_probe_improvement_ratio": 4.0,
            "moment_schur_metadata": {
                "mode": "constraint",
                "rank": 3,
                "extra_size": 2,
                "setup_s": 0.1,
                "expected_size": 5,
                "rcond": 1.0e-8,
                "singular_value_proxy": (1.0, 0.1),
                "device_resident": True,
                "error": None,
            },
            "moment_schur_stats": {"applies": 6, "base_applies": 4},
            "two_level_enabled": 1,
            "two_level_built": True,
            "two_level_metadata": {
                "mode": "seed",
                "basis_size": 4,
                "rank": 4,
                "setup_s": 0.2,
                "rcond": 1.0e-7,
                "basis_names": ("density", "flow"),
                "active_projected": True,
                "expected_size": 8,
                "error": None,
            },
            "two_level_stats": {"applies": 8, "coarse_applies": 3},
            "global_coupling_enabled": 1,
            "global_coupling_built": True,
            "global_coupling_metadata": {
                "mode": "load",
                "load_basis_size": 5,
                "basis_size": 7,
                "rank": 6,
                "setup_s": 0.3,
                "setup_budget_s": 2.0,
                "setup_budget_reached": False,
                "rcond": 1.0e-6,
                "coarse_solver": "pinv",
                "smoother": "block",
                "ridge": 1.0e-10,
                "singular_values": (3.0, 1.0),
                "device_resident": True,
                "fsavg_lmax": 2,
                "angular_lmax": 3,
                "basis_names": ("source", "constraint"),
                "error": None,
            },
            "global_coupling_stats": {"applies": 9, "coarse_applies": 5},
        }
    )

    assert metadata["xblock_moment_schur_enabled"] is True
    assert metadata["xblock_moment_schur_used"] is False
    assert metadata["xblock_moment_schur_default_blocked_by_compact_factors"] is True
    assert metadata["xblock_moment_schur_rank"] == 3
    assert metadata["xblock_moment_schur_device_resident"] is True
    assert metadata["xblock_moment_schur_applies"] == 6
    assert metadata["xblock_two_level_enabled"] is True
    assert metadata["xblock_two_level_basis_names"] == ("density", "flow")
    assert metadata["xblock_two_level_active_projected"] is True
    assert metadata["xblock_two_level_coarse_applies"] == 3
    assert metadata["xblock_global_coupling_enabled"] is True
    assert metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert metadata["xblock_global_coupling_coarse_solver"] == "pinv"
    assert metadata["xblock_global_coupling_singular_values"] == (3.0, 1.0)
    assert metadata["xblock_global_coupling_device_resident"] is True
    assert metadata["xblock_global_coupling_coarse_applies"] == 5


def test_xblock_qi_seed_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_seed_preconditioner_diagnostics(
        {
            "xblock_initial_seed_used": 1,
            "xblock_initial_seed_residual_norm": 1.0e-3,
            "xblock_initial_seed_residual_ratio": 0.5,
            "moment_schur_seed_enabled": 1,
            "moment_schur_seed_used": False,
            "moment_schur_seed_residual_norm": 2.0e-3,
            "moment_schur_seed_residual_ratio": 0.75,
            "qi_coarse_seed_enabled": 1,
            "qi_coarse_seed_used": True,
            "qi_coarse_seed_residual_before": 3.0,
            "qi_coarse_seed_residual_after": 1.0,
            "qi_coarse_seed_improvement_ratio": 3.0,
            "qi_coarse_seed_rank": 4,
            "qi_coarse_seed_candidate_count": 9,
            "qi_coarse_seed_reason": "accepted",
            "qi_coarse_seed_labels": ("flat", "current"),
            "qi_coarse_seed_s": 0.25,
            "qi_seed_basis_kind": "load",
            "qi_seed_max_candidates": 12,
            "qi_seed_max_angular_mode": 3,
            "qi_galerkin_preconditioner_enabled": 1,
            "qi_galerkin_preconditioner_built": True,
            "qi_galerkin_preconditioner_used": False,
            "qi_galerkin_preconditioner_reason": "probe_rejected",
            "qi_galerkin_preconditioner_mode": "coarse",
            "qi_galerkin_preconditioner_rank": 5,
            "qi_galerkin_preconditioner_candidate_count": 10,
            "qi_galerkin_preconditioner_coarse_shape": (5, 5),
            "qi_galerkin_preconditioner_coarse_norm": 2.5,
            "qi_galerkin_preconditioner_rcond": 1.0e-6,
            "qi_galerkin_preconditioner_damping": 1.0e-4,
            "qi_galerkin_preconditioner_basis_reused_from_seed": True,
            "qi_galerkin_preconditioner_residual_before": 4.0,
            "qi_galerkin_preconditioner_residual_after": 2.0,
            "qi_galerkin_preconditioner_improvement_ratio": 2.0,
            "qi_galerkin_preconditioner_probe_reduced": True,
            "qi_galerkin_preconditioner_probe_candidates": (0, 2),
            "qi_galerkin_preconditioner_selected_index": 1,
            "qi_galerkin_preconditioner_setup_s": 0.5,
            "qi_galerkin_stats": {
                "applies": 6,
                "coarse_applies": 4,
                "base_applies": 2,
            },
            "qi_two_level_preconditioner_enabled": 1,
            "qi_two_level_preconditioner_built": True,
            "qi_two_level_preconditioner_used": True,
            "qi_two_level_preconditioner_reason": "accepted",
            "qi_two_level_preconditioner_rank": 6,
            "qi_two_level_preconditioner_candidate_count": 11,
            "qi_two_level_preconditioner_coarse_shape": (6, 6),
            "qi_two_level_preconditioner_coarse_norm": 3.5,
            "qi_two_level_preconditioner_operator_on_basis_shape": (20, 6),
            "qi_two_level_preconditioner_operator_on_basis_norm": 4.5,
            "qi_two_level_preconditioner_coarse_solver": "pinv",
            "qi_two_level_preconditioner_residual_augmented": True,
            "qi_two_level_preconditioner_rank_before_augmentation": 4,
            "qi_two_level_preconditioner_augmentation_labels": ("r0", "r1"),
            "qi_two_level_preconditioner_residual_augment_max_extra": 3,
            "qi_two_level_preconditioner_residual_augment_steps": 2,
            "qi_two_level_preconditioner_residual_augment_include_residuals": True,
            "qi_two_level_preconditioner_smoothed_load_basis": True,
            "qi_two_level_preconditioner_smoothed_load_metadata": {"rank": 2},
            "qi_two_level_preconditioner_rcond": 1.0e-7,
            "qi_two_level_preconditioner_damping": 1.0e-5,
            "qi_two_level_preconditioner_basis_reused_from_seed": True,
            "qi_two_level_preconditioner_residual_before": 5.0,
            "qi_two_level_preconditioner_residual_after": 1.0,
            "qi_two_level_preconditioner_improvement_ratio": 5.0,
            "qi_two_level_preconditioner_probe_candidates": (1, 3),
            "qi_two_level_preconditioner_selected_index": 0,
            "qi_two_level_preconditioner_setup_s": 0.75,
            "qi_two_level_stats": {"applies": 8, "local_applies": 5},
        }
    )

    assert metadata["xblock_initial_seed_used"] is True
    assert metadata["xblock_moment_schur_seed_used"] is False
    assert metadata["xblock_qi_coarse_seed_used"] is True
    assert metadata["xblock_qi_coarse_seed_rank"] == 4
    assert metadata["xblock_qi_coarse_seed_labels"] == ("flat", "current")
    assert metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert metadata["xblock_qi_galerkin_preconditioner_used"] is False
    assert metadata["xblock_qi_galerkin_preconditioner_rank"] == 5
    assert metadata["xblock_qi_galerkin_preconditioner_applies"] == 6
    assert metadata["xblock_qi_galerkin_preconditioner_base_applies"] == 2
    assert metadata["xblock_qi_two_level_preconditioner_enabled"] is True
    assert metadata["xblock_qi_two_level_preconditioner_used"] is True
    assert metadata["xblock_qi_two_level_preconditioner_rank"] == 6
    assert metadata["xblock_qi_two_level_preconditioner_residual_augmented"] is True
    assert metadata["xblock_qi_two_level_preconditioner_augmentation_labels"] == (
        "r0",
        "r1",
    )
    assert metadata["xblock_qi_two_level_preconditioner_smoothed_load_metadata"] == {
        "rank": 2
    }
    assert metadata["xblock_qi_two_level_preconditioner_local_applies"] == 5


def test_xblock_device_krylov_diagnostics_preserve_transfer_free_logic() -> None:
    fallback = SimpleNamespace(
        mode="auto",
        used=False,
        reason="device_ok",
        requested_method="fgmres_jax",
        effective_krylov_env_value="fgmres_jax",
        min_active_size=128,
        qi_like_full_fp_3d=True,
        ignored_env=False,
        non_autodiff=False,
    )
    operator_reuse = SimpleNamespace(
        enabled=True,
        reason="shape_reusable",
        skip_xblock_factors=True,
        to_metadata=lambda: {"enabled": True, "reason": "shape_reusable"},
    )

    metadata = xblock_device_krylov_diagnostics(
        {
            "xblock_krylov_method": "fgmres_jax",
            "xblock_device_krylov_methods": {"gmres_jax", "fgmres_jax"},
            "xblock_jax_factors": True,
            "assembled_operator_built": True,
            "assembled_operator_device_resident": True,
            "two_level_built": False,
            "global_coupling_built": True,
            "global_coupling_metadata": {"device_resident": True},
            "xblock_device_host_fallback_decision": fallback,
            "xblock_krylov_env_requested": "auto",
            "xblock_device_host_fallback_auto_disabled_by_qi_device": True,
            "xblock_qi_device_operator_reuse_decision": operator_reuse,
            "xblock_device_fgmres_jit": True,
            "xblock_device_fgmres_jit_mode": "cycle",
            "xblock_device_fgmres_jit_outer_k": 7,
            "qi_device_augmented_krylov_requested": True,
            "qi_device_augmented_krylov_used": False,
            "qi_device_augmented_krylov_rank": 3,
            "qi_device_augmented_krylov_reason": "seed_only",
            "qi_device_augmented_krylov_mode": "right",
            "qi_device_augmented_seed_requested": True,
            "qi_device_augmented_seed_available": True,
            "qi_device_augmented_seed_used": False,
            "qi_device_augmented_seed_rank": 2,
            "qi_device_augmented_seed_max_rank": 5,
            "qi_device_augmented_seed_reason": "accepted",
            "qi_device_augmented_seed_projection_residual": 1.0e-4,
            "qi_device_augmented_seed_labels": ("constant", "current"),
            "tfqmr_replacement_interval": 11,
            "xblock_device_krylov_forced_jax_factors": True,
            "xblock_device_fgmres_forced_right_pc": True,
            "fgmres_block_between_cycles": True,
            "xblock_estimated_gmres_basis_nbytes": 100,
            "xblock_estimated_bicgstab_work_nbytes": 200,
            "xblock_estimated_tfqmr_work_nbytes": 300,
        }
    )

    assert metadata["xblock_device_krylov_method"] == "fgmres_jax"
    assert metadata["xblock_device_host_fallback_mode"] == "auto"
    assert metadata["xblock_device_host_fallback_used"] is False
    assert metadata["xblock_device_host_fallback_auto_disabled_by_qi_device"] is True
    assert metadata["xblock_qi_device_operator_reuse"] == {
        "enabled": True,
        "reason": "shape_reusable",
    }
    assert metadata["xblock_qi_device_operator_reuse_skip_xblock_factors"] is True
    assert metadata["xblock_device_fgmres_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_mode"] == "cycle"
    assert metadata["xblock_device_fgmres_jit_outer_k"] == 7
    assert metadata["xblock_device_fgmres_qi_augmented_seed_labels"] == (
        "constant",
        "current",
    )
    assert metadata["xblock_device_tfqmr_replacement_interval"] == 11
    assert metadata["xblock_device_fgmres_forced_jax_factors"] is True
    assert metadata["xblock_estimated_tfqmr_work_nbytes"] == 300
    assert metadata["xblock_device_krylov_host_transfer_free"] is True
    assert metadata["xblock_device_fgmres_host_transfer_free"] is True
    assert metadata["xblock_device_bicgstab_host_transfer_free"] is False
    assert metadata["xblock_device_tfqmr_host_transfer_free"] is False


class _DefaultDirectTailState(dict):
    def __missing__(self, key: str) -> object:
        self[key] = 1
        return self[key]


def test_sparse_pc_direct_tail_result_metadata_preserves_driver_conversions() -> None:
    structured_metadata = {"kind": "native"}
    support_metadata = {"accepted": True}
    coupled_metadata = {"rank": 5}
    state = _DefaultDirectTailState(
        {
            "direct_tail_operator_bundle": SimpleNamespace(
                metadata=SimpleNamespace(
                    reason="direct_pmat",
                    nnz_estimate=np.int64(123),
                    csr_nbytes_estimate=np.int64(456),
                )
            ),
            "direct_tail_structured_max_nbytes": 2 * 1024 * 1024,
            "direct_tail_true_window_specs": ((np.int64(1), np.int64(2)),),
            "direct_tail_true_active_block_species_count": None,
            "direct_tail_structured_pc_metadata": structured_metadata,
            "direct_tail_support_mode_preflight_metadata": support_metadata,
            "direct_tail_true_coupled_coarse_metadata": coupled_metadata,
            "direct_tail_residual_window_coefficient_mode": "normal",
            "direct_tail_residual_window_combine_mode": "additive",
            "direct_tail_error": "not_selected",
            "direct_tail_structured_pc_requested": "auto",
            "direct_tail_structured_pc_reason": "admitted",
            "direct_tail_structured_pc_error": None,
            "direct_tail_support_mode_preflight_error": None,
        }
    )

    metadata = sparse_pc_direct_tail_result_metadata(state)

    assert metadata["sparse_pc_direct_tail_true_window_specs"] == ((1, 2),)
    assert metadata["sparse_pc_direct_tail_true_active_block_species_count"] is None
    assert (
        metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb"]
        == 2.0
    )
    assert (
        metadata["sparse_pc_fortran_reduced_direct_tail_operator_reason"]
        == "direct_pmat"
    )
    assert metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] == 123
    assert (
        metadata["sparse_pc_fortran_reduced_direct_tail_csr_nbytes_estimate"]
        == 456
    )
    assert (
        metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
        is structured_metadata
    )
    assert (
        metadata[
            "sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_metadata"
        ]
        is support_metadata
    )
    assert (
        metadata["sparse_pc_direct_tail_true_coupled_coarse_metadata"]
        is coupled_metadata
    )
    assert metadata["sparse_pc_direct_tail_residual_window_coefficient_mode"] == "normal"
    assert metadata["sparse_pc_direct_tail_residual_window_combine_mode"] == "additive"


def test_xblock_sparse_pc_core_diagnostics_preserve_payload() -> None:
    metadata = xblock_sparse_pc_core_diagnostics(
        XBlockSparsePCCoreDiagnosticsContext(
            solver_kind="sparse_pc_gmres",
            accepted_converged=True,
            reported_iterations=np.int64(12),
            reported_matvecs=np.int64(34),
            python_matvecs=np.int64(56),
            device_cycle_estimated_matvecs=np.int64(78),
            krylov_method="fgmres_jax",
            candidate_krylov_method="gmres_jax",
            candidate_iterations=9,
            candidate_matvecs=10,
            candidate_residual_norm=1.0e-5,
            fallback_started_from_candidate=True,
            fallback_candidate_improved_rhs=False,
            precondition_side="right",
            default_right_preconditioned=True,
            default_short_restart_capped=False,
            gmres_restart=40,
            gmres_maxiter=80,
            setup_s=0.5,
            solve_s=1.5,
            elapsed_s=12.5,
            sparse_pc_factor_s=0.25,
            preconditioner_xi=3,
            preconditioner_built=True,
            assembled_host=False,
            jax_factors=True,
            jax_factor_format="padded",
            jax_factor_apply="vmap",
            lower_fill_mode="probe",
            lower_fill_ignored_env=True,
        )
    )

    assert metadata["solver_kind"] == "sparse_pc_gmres"
    assert metadata["residual_kind"] == "true_residual"
    assert metadata["accepted_converged"] is True
    assert metadata["iterations"] == 12
    assert metadata["python_matvecs"] == 56
    assert metadata["device_cycle_estimated_matvecs"] == 78
    assert metadata["fallback_from_krylov_method"] == "gmres_jax"
    assert metadata["fallback_started_from_candidate"] is True
    assert metadata["precondition_side"] == "right"
    assert metadata["elapsed_s"] == 12.5
    assert metadata["sparse_pc_xblock_preconditioner_xi"] == 3
    assert metadata["sparse_pc_xblock_jax_factor_format"] == "padded"
    assert metadata["xblock_lower_fill_requested"] is True
    assert metadata["xblock_lower_fill_ignored_env"] is True
