"""Diagnostics metadata builders for profile-response sparse-PC solves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def sparse_xblock_rescue_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return stable diagnostics for the sparse x-block rescue tail."""

    return {
        "sparse_xblock_rescue_active": bool(scope["sparse_xblock_rescue_active"]),
        "sparse_xblock_rescue_attempted": bool(scope["sparse_xblock_rescue_attempted"]),
        "sparse_xblock_rescue_built": bool(scope["sparse_xblock_rescue_built"]),
        "sparse_xblock_rescue_error": scope["sparse_xblock_rescue_error"],
        "sparse_xblock_rescue_reason": str(scope["sparse_xblock_rescue_reason"]),
        "sparse_xblock_rescue_assembled_host_fp": bool(
            scope["sparse_xblock_rescue_assembled_host_fp"]
        ),
        "sparse_xblock_rescue_preconditioner_xi": scope[
            "sparse_xblock_rescue_preconditioner_xi"
        ],
        "sparse_xblock_rescue_seed_residual": scope[
            "sparse_xblock_rescue_seed_residual"
        ],
        "sparse_xblock_rescue_seed_improvement_ratio": scope[
            "sparse_xblock_rescue_seed_improvement_ratio"
        ],
        "sparse_xblock_rescue_seed_accept_ratio": scope[
            "sparse_xblock_rescue_seed_accept_ratio"
        ],
        "sparse_xblock_rescue_seed_refine_steps": scope[
            "sparse_xblock_rescue_seed_refine_steps"
        ],
        "sparse_xblock_rescue_seed_refines_performed": scope[
            "sparse_xblock_rescue_seed_refines_performed"
        ],
        "sparse_xblock_rescue_candidate_residual": scope[
            "sparse_xblock_rescue_candidate_residual"
        ],
        "sparse_xblock_rescue_candidate_accepted": bool(
            scope["sparse_xblock_rescue_candidate_accepted"]
        ),
    }


def fp_xblock_global_correction_metadata(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return stable diagnostics for FP x-block global correction."""

    return {
        "fp_xblock_global_correction_allowed": bool(
            scope["fp_xblock_global_correction_allowed"]
        ),
        "fp_xblock_global_correction_attempted": bool(
            scope["fp_xblock_global_correction_attempted"]
        ),
        "fp_xblock_global_correction_accepted": bool(
            scope["fp_xblock_global_correction_accepted"]
        ),
        "fp_xblock_global_correction_reason": str(
            scope["fp_xblock_global_correction_reason"]
        ),
        "fp_xblock_global_correction_error": scope["fp_xblock_global_correction_error"],
        "fp_xblock_global_correction_preconditioner": scope[
            "fp_xblock_global_correction_preconditioner"
        ],
        "fp_xblock_global_correction_steps": scope["fp_xblock_global_correction_steps"],
        "fp_xblock_global_correction_accepted_steps": scope[
            "fp_xblock_global_correction_accepted_steps"
        ],
        "fp_xblock_global_correction_residual_before": scope[
            "fp_xblock_global_correction_residual_before"
        ],
        "fp_xblock_global_correction_residual_after": scope[
            "fp_xblock_global_correction_residual_after"
        ],
        "fp_xblock_global_correction_improvement_ratio": scope[
            "fp_xblock_global_correction_improvement_ratio"
        ],
        "fp_xblock_global_correction_elapsed_s": scope[
            "fp_xblock_global_correction_elapsed_s"
        ],
    }


def fp_xblock_highx_residual_correction_metadata(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return stable diagnostics for FP high-x residual correction."""

    return {
        "fp_xblock_highx_residual_correction_allowed": bool(
            scope["fp_xblock_highx_residual_correction_allowed"]
        ),
        "fp_xblock_highx_residual_correction_attempted": bool(
            scope["fp_xblock_highx_residual_correction_attempted"]
        ),
        "fp_xblock_highx_residual_correction_accepted": bool(
            scope["fp_xblock_highx_residual_correction_accepted"]
        ),
        "fp_xblock_highx_residual_correction_reason": str(
            scope["fp_xblock_highx_residual_correction_reason"]
        ),
        "fp_xblock_highx_residual_correction_error": scope[
            "fp_xblock_highx_residual_correction_error"
        ],
        "fp_xblock_highx_residual_correction_residual_before": scope[
            "fp_xblock_highx_residual_correction_residual_before"
        ],
        "fp_xblock_highx_residual_correction_residual_after": scope[
            "fp_xblock_highx_residual_correction_residual_after"
        ],
        "fp_xblock_highx_residual_correction_improvement_ratio": scope[
            "fp_xblock_highx_residual_correction_improvement_ratio"
        ],
        "fp_xblock_highx_residual_correction_elapsed_s": scope[
            "fp_xblock_highx_residual_correction_elapsed_s"
        ],
        "fp_xblock_highx_residual_correction_direction_count": scope[
            "fp_xblock_highx_residual_correction_direction_count"
        ],
        "fp_xblock_highx_residual_correction_direction_names": tuple(
            scope["fp_xblock_highx_residual_correction_direction_names"]
        ),
    }


def sparse_rescue_tail_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return the combined sparse-rescue tail diagnostics for final metadata."""

    return {
        **sparse_xblock_rescue_metadata(scope),
        **fp_xblock_global_correction_metadata(scope),
        **fp_xblock_highx_residual_correction_metadata(scope),
    }


def xblock_qi_device_preconditioner_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return the x-block QI-device preconditioner diagnostics payload."""

    metadata = scope["qi_device_preconditioner_metadata"]
    stats = scope["qi_device_stats"]
    if not isinstance(metadata, Mapping):
        raise TypeError("qi_device_preconditioner_metadata must be a mapping")
    if not isinstance(stats, Mapping):
        raise TypeError("qi_device_stats must be a mapping")

    out: dict[str, object] = {
        "xblock_qi_device_preconditioner_enabled": bool(
            scope["qi_device_preconditioner_enabled"]
        ),
        "xblock_qi_device_preconditioner_built": bool(
            scope["qi_device_preconditioner_built"]
        ),
        "xblock_qi_device_preconditioner_used": bool(
            scope["qi_device_preconditioner_used"]
        ),
        "xblock_qi_device_preconditioner_used_in_krylov": bool(
            scope["qi_device_preconditioner_used_in_krylov"]
        ),
        "xblock_qi_device_preconditioner_reason": scope[
            "qi_device_preconditioner_reason"
        ],
        "xblock_qi_device_preconditioner_rank": int(
            scope["qi_device_preconditioner_rank"]
        ),
        "xblock_qi_device_preconditioner_candidate_count": int(
            scope["qi_device_preconditioner_candidate_count"]
        ),
        "xblock_qi_device_preconditioner_coarse_operator_shape": scope[
            "qi_device_preconditioner_coarse_shape"
        ],
        "xblock_qi_device_preconditioner_operator_on_basis_shape": scope[
            "qi_device_preconditioner_operator_on_basis_shape"
        ],
        "xblock_qi_device_preconditioner_coarse_operator_norm": float(
            scope["qi_device_preconditioner_coarse_norm"]
        ),
        "xblock_qi_device_preconditioner_operator_on_basis_norm": float(
            scope["qi_device_preconditioner_operator_on_basis_norm"]
        ),
        "xblock_qi_device_preconditioner_residual_before": scope[
            "qi_device_preconditioner_residual_before"
        ],
        "xblock_qi_device_preconditioner_residual_after": scope[
            "qi_device_preconditioner_residual_after"
        ],
        "xblock_qi_device_preconditioner_improvement_ratio": scope[
            "qi_device_preconditioner_improvement_ratio"
        ],
        "xblock_qi_device_preconditioner_metadata": metadata,
        "xblock_qi_device_preconditioner_setup_s": float(
            scope["qi_device_preconditioner_setup_s"]
        ),
        "xblock_qi_device_preconditioner_min_improvement": float(
            scope["qi_device_preconditioner_min_improvement"]
        ),
        "xblock_qi_device_preconditioner_use_in_krylov": bool(
            scope["qi_device_preconditioner_use_in_krylov"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_requested": bool(
            scope["qi_device_augmented_krylov_requested"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_used": bool(
            scope["qi_device_augmented_krylov_used"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_rank": int(
            scope["qi_device_augmented_krylov_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_reason": scope[
            "qi_device_augmented_krylov_reason"
        ],
        "xblock_qi_device_preconditioner_augmented_krylov_mode": scope[
            "qi_device_augmented_krylov_mode"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_requested": bool(
            scope["qi_device_augmented_seed_requested"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_available": bool(
            scope["qi_device_augmented_seed_available"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_used": bool(
            scope["qi_device_augmented_seed_used"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_rank": int(
            scope["qi_device_augmented_seed_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_max_rank": int(
            scope["qi_device_augmented_seed_max_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_reason": scope[
            "qi_device_augmented_seed_reason"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_projection_residual_norm": scope[
            "qi_device_augmented_seed_projection_residual"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_labels": scope[
            "qi_device_augmented_seed_labels"
        ],
        "xblock_qi_device_preconditioner_applies": int(stats.get("applies", 0)),
        "xblock_qi_device_preconditioner_operator_krylov_enrichment": bool(
            metadata.get("operator_krylov_enrichment_enabled", False)
        ),
        "xblock_qi_device_preconditioner_coarse_reuse": bool(
            metadata.get("multilevel_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_enrichment": bool(
            metadata.get("residual_snapshot_enrichment_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation": bool(
            metadata.get("residual_snapshot_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_rank": int(
            metadata.get("residual_snapshot_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_candidate_count": int(
            metadata.get("residual_snapshot_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_group_count": int(
            metadata.get("residual_snapshot_residual_equation_group_count", 0)
        ),
        "xblock_qi_device_preconditioner_multilevel_residual_equation": bool(
            metadata.get("multilevel_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_multilevel_residual_equation_solver": metadata.get(
            "multilevel_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation": bool(
            metadata.get("global_moment_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_solver": metadata.get(
            "global_moment_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_rank": int(
            metadata.get("global_moment_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count": int(
            metadata.get("global_moment_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_condition_estimate": float(
            metadata.get(
                "global_moment_residual_equation_condition_estimate", float("inf")
            )
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation": bool(
            metadata.get("residual_galerkin_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_solver": metadata.get(
            "residual_galerkin_equation_solver"
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_rank": int(
            metadata.get("residual_galerkin_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count": int(
            metadata.get("residual_galerkin_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_stage_count": int(
            metadata.get("residual_galerkin_equation_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_condition_estimate": float(
            metadata.get("residual_galerkin_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation": bool(
            metadata.get("phase_space_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_max_rank": int(
            metadata.get(
                "phase_space_residual_equation_max_rank_requested",
                metadata.get("phase_space_residual_equation_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_solver": metadata.get(
            "phase_space_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_rank": int(
            metadata.get("phase_space_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_candidate_count": int(
            metadata.get("phase_space_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_stage_count": int(
            metadata.get("phase_space_residual_equation_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_condition_estimate": float(
            metadata.get(
                "phase_space_residual_equation_condition_estimate", float("inf")
            )
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_residual_before": float(
            metadata.get("phase_space_residual_equation_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_residual_after": float(
            metadata.get("phase_space_residual_equation_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_global": bool(
            metadata.get("phase_space_residual_equation_include_global", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_trapped_boundary_fraction": float(
            metadata.get(
                "phase_space_residual_equation_trapped_boundary_fraction",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_radial": bool(
            metadata.get("phase_space_residual_equation_include_radial", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_species": bool(
            metadata.get("phase_space_residual_equation_include_species", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse": bool(
            metadata.get("residual_region_bounce_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank": int(
            metadata.get(
                "residual_region_bounce_coarse_max_rank_requested",
                metadata.get("residual_region_bounce_coarse_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_solver": metadata.get(
            "residual_region_bounce_coarse_solver"
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_rank": int(
            metadata.get("residual_region_bounce_coarse_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_candidate_count": int(
            metadata.get("residual_region_bounce_coarse_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_stage_count": int(
            metadata.get("residual_region_bounce_coarse_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_condition_estimate": float(
            metadata.get(
                "residual_region_bounce_coarse_condition_estimate", float("inf")
            )
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_before": float(
            metadata.get("residual_region_bounce_coarse_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_after": float(
            metadata.get("residual_region_bounce_coarse_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_global": bool(
            metadata.get("residual_region_bounce_coarse_include_global", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_radial": bool(
            metadata.get("residual_region_bounce_coarse_include_radial", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_species": bool(
            metadata.get("residual_region_bounce_coarse_include_species", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_bounce_boundary": float(
            metadata.get("residual_region_bounce_coarse_bounce_boundary", float("nan"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_min_region_energy_fraction": float(
            metadata.get(
                "residual_region_bounce_coarse_min_region_energy_fraction",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_region_bands": metadata.get(
            "residual_region_bounce_coarse_region_bands"
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse": bool(
            metadata.get("active_pattern_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_max_rank": int(
            metadata.get(
                "active_pattern_coarse_max_rank_requested",
                metadata.get("active_pattern_coarse_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_max_candidates": int(
            metadata.get("active_pattern_coarse_max_candidates_requested", 0) or 0
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_solver": metadata.get(
            "active_pattern_coarse_solver"
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_rank": int(
            metadata.get("active_pattern_coarse_rank", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_candidate_count": int(
            metadata.get("active_pattern_coarse_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_stage_count": int(
            metadata.get("active_pattern_coarse_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_condition_estimate": float(
            metadata.get("active_pattern_coarse_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_residual_before": float(
            metadata.get("active_pattern_coarse_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_residual_after": float(
            metadata.get("active_pattern_coarse_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_include_global": bool(
            metadata.get("active_pattern_coarse_include_global", False)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_min_chunk_energy_fraction": float(
            metadata.get(
                "active_pattern_coarse_min_chunk_energy_fraction", float("nan")
            )
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation": bool(
            metadata.get("block_schur_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_rank": int(
            metadata.get("block_schur_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_candidate_count": int(
            metadata.get("block_schur_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_group_count": int(
            metadata.get("block_schur_residual_equation_group_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation": bool(
            metadata.get("coupled_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_max_rank": int(
            metadata.get(
                "coupled_residual_equation_max_rank_requested",
                metadata.get("coupled_residual_equation_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_rank": int(
            metadata.get("coupled_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_candidate_count": int(
            metadata.get("coupled_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_source_stage_count": int(
            metadata.get("coupled_residual_equation_source_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_source_stage_ranks": metadata.get(
            "coupled_residual_equation_source_stage_ranks"
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_solver": metadata.get(
            "coupled_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_include_flat": bool(
            metadata.get("coupled_residual_equation_include_flat", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_min_relative_improvement": float(
            metadata.get(
                "coupled_residual_equation_min_relative_improvement_requested",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_install_in_krylov_on_reject": bool(
            metadata.get(
                "coupled_residual_equation_install_in_krylov_on_reject_requested",
                False,
            )
        ),
        "xblock_qi_device_preconditioner_seed_probe_accepted": bool(
            metadata.get("seed_probe_accepted", False)
        ),
        "xblock_qi_device_preconditioner_installed_in_krylov_after_seed_reject": bool(
            metadata.get("installed_in_krylov_after_seed_reject", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_condition_estimate": float(
            metadata.get("coupled_residual_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_residual_before": float(
            metadata.get("coupled_residual_equation_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_residual_after": float(
            metadata.get("coupled_residual_equation_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_accepted": bool(
            metadata.get("coupled_residual_equation_accepted", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_reason": metadata.get(
            "coupled_residual_equation_reason"
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_enrichment": bool(
            metadata.get("block_schur_residual_enrichment_enabled", False)
        ),
    }
    return out


def xblock_qi_deflated_preconditioner_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return the x-block QI residual-deflation preconditioner diagnostics."""

    metadata = scope["qi_deflated_preconditioner_metadata"]
    stats = scope["qi_deflated_stats"]
    if not isinstance(metadata, Mapping):
        raise TypeError("qi_deflated_preconditioner_metadata must be a mapping")
    if not isinstance(stats, Mapping):
        raise TypeError("qi_deflated_stats must be a mapping")

    return {
        "xblock_qi_deflated_preconditioner_enabled": bool(
            scope["qi_deflated_preconditioner_enabled"]
        ),
        "xblock_qi_deflated_preconditioner_built": bool(
            scope["qi_deflated_preconditioner_built"]
        ),
        "xblock_qi_deflated_preconditioner_used": bool(
            scope["qi_deflated_preconditioner_used"]
        ),
        "xblock_qi_deflated_preconditioner_reason": scope[
            "qi_deflated_preconditioner_reason"
        ],
        "xblock_qi_deflated_preconditioner_rank": int(
            scope["qi_deflated_preconditioner_rank"]
        ),
        "xblock_qi_deflated_preconditioner_candidate_count": int(
            scope["qi_deflated_preconditioner_candidate_count"]
        ),
        "xblock_qi_deflated_preconditioner_residual_before": scope[
            "qi_deflated_preconditioner_residual_before"
        ],
        "xblock_qi_deflated_preconditioner_residual_after": scope[
            "qi_deflated_preconditioner_residual_after"
        ],
        "xblock_qi_deflated_preconditioner_improvement_ratio": scope[
            "qi_deflated_preconditioner_improvement_ratio"
        ],
        "xblock_qi_deflated_preconditioner_metadata": metadata,
        "xblock_qi_deflated_preconditioner_setup_s": float(
            scope["qi_deflated_preconditioner_setup_s"]
        ),
        "xblock_qi_deflated_preconditioner_applies": int(stats.get("applies", 0)),
        "xblock_qi_deflated_preconditioner_local_applies": int(
            stats.get("local_applies", 0)
        ),
        "xblock_qi_deflated_preconditioner_cycles": int(
            metadata.get("correction_cycles", 0)
        ),
        "xblock_qi_deflated_preconditioner_seed_solver": metadata.get("seed_solver"),
        "xblock_qi_deflated_preconditioner_cycle_residual_history": metadata.get(
            "cycle_residual_history",
            (),
        ),
        "xblock_qi_deflated_preconditioner_cycle_coefficients": metadata.get(
            "cycle_coefficients",
            (),
        ),
        "xblock_qi_deflated_preconditioner_use_in_krylov": bool(
            scope["qi_deflated_preconditioner_used_in_krylov"]
        ),
    }


@dataclass(frozen=True, slots=True)
class XBlockSideProbeDiagnosticsContext:
    """Explicit inputs for side-probe and LGMRES-rescue diagnostics."""

    enabled: object
    used: object
    switched: object
    switch_suppressed_by_global_coupling: object
    switch_suppressed_by_explicit_side: object
    physical_seed_preserved_after_switch: object
    seed_used: object
    seed_residual_norm: object
    initial_side: object
    selected_side: object
    initial_method: object
    selected_method: object
    lgmres_rescue: object
    lgmres_rescue_maxiter_capped: object
    lgmres_rescue_outer_k: object
    residual_norm: object
    residual_ratio: object
    iterations: object
    matvecs: object
    elapsed_s: object


def xblock_side_probe_diagnostics(
    context: XBlockSideProbeDiagnosticsContext,
) -> dict[str, object]:
    """Return side-probe and LGMRES-rescue diagnostics for x-block solves."""

    return {
        "xblock_side_probe_enabled": bool(context.enabled),
        "xblock_side_probe_used": bool(context.used),
        "xblock_side_probe_switched": bool(context.switched),
        "xblock_side_probe_switch_suppressed_by_global_coupling": bool(
            context.switch_suppressed_by_global_coupling
        ),
        "xblock_side_probe_switch_suppressed_by_explicit_side": bool(
            context.switch_suppressed_by_explicit_side
        ),
        "xblock_side_probe_physical_seed_preserved_after_switch": bool(
            context.physical_seed_preserved_after_switch
        ),
        "xblock_side_probe_seed_used": bool(context.seed_used),
        "xblock_side_probe_seed_residual_norm": context.seed_residual_norm,
        "xblock_side_probe_initial_side": context.initial_side,
        "xblock_side_probe_selected_side": context.selected_side,
        "xblock_side_probe_initial_method": context.initial_method,
        "xblock_side_probe_selected_method": context.selected_method,
        "xblock_side_probe_lgmres_rescue": bool(context.lgmres_rescue),
        "xblock_lgmres_rescue_maxiter_capped": bool(
            context.lgmres_rescue_maxiter_capped
        ),
        "xblock_lgmres_rescue_outer_k": context.lgmres_rescue_outer_k,
        "xblock_side_probe_residual_norm": context.residual_norm,
        "xblock_side_probe_residual_ratio": context.residual_ratio,
        "xblock_side_probe_iterations": int(context.iterations),
        "xblock_side_probe_matvecs": int(context.matvecs),
        "xblock_side_probe_s": float(context.elapsed_s),
    }


@dataclass(frozen=True, slots=True)
class XBlockAssembledOperatorDiagnosticsContext:
    """Explicit inputs for assembled-operator and equilibration diagnostics."""

    enabled: object
    built: object
    metadata: Mapping[str, object]
    row_equilibration_enabled: object
    row_equilibration_built: object
    row_equilibration_metadata: Mapping[str, object]
    col_equilibration_enabled: object
    col_equilibration_built: object
    col_equilibration_metadata: Mapping[str, object]


def xblock_assembled_operator_diagnostics(
    context: XBlockAssembledOperatorDiagnosticsContext,
) -> dict[str, object]:
    """Return assembled-operator and equilibration diagnostics for x-block solves."""

    metadata = context.metadata
    row_metadata = context.row_equilibration_metadata
    col_metadata = context.col_equilibration_metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("assembled_operator_metadata must be a mapping")
    if not isinstance(row_metadata, Mapping):
        raise TypeError("xblock_row_equilibration_metadata must be a mapping")
    if not isinstance(col_metadata, Mapping):
        raise TypeError("xblock_col_equilibration_metadata must be a mapping")

    return {
        "xblock_assembled_operator_enabled": bool(context.enabled),
        "xblock_assembled_operator_built": bool(context.built),
        "xblock_assembled_operator_active_dof": metadata.get("active_dof", False),
        "xblock_assembled_operator_preflight_scope": metadata.get("preflight_scope"),
        "xblock_assembled_operator_setup_s": metadata.get("setup_s"),
        "xblock_assembled_operator_preflight_rejected": metadata.get(
            "preflight_rejected",
            False,
        ),
        "xblock_assembled_operator_preflight_pattern_nnz_estimate": metadata.get(
            "preflight_pattern_nnz_estimate"
        ),
        "xblock_assembled_operator_preflight_peak_nbytes_estimate": metadata.get(
            "preflight_peak_nbytes_estimate"
        ),
        "xblock_assembled_operator_preflight_full_csr_nbytes_estimate": metadata.get(
            "preflight_full_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_preflight_active_csr_nbytes_estimate": metadata.get(
            "preflight_active_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_pattern_nnz": metadata.get("pattern_nnz"),
        "xblock_assembled_operator_matrix_nnz": metadata.get("matrix_nnz"),
        "xblock_assembled_operator_csr_nbytes_estimate": metadata.get(
            "csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_device_enabled": bool(
            metadata.get("device_enabled", False)
        ),
        "xblock_assembled_operator_device_required": bool(
            metadata.get("device_required", False)
        ),
        "xblock_assembled_operator_device_resident": bool(
            metadata.get("device_resident", False)
        ),
        "xblock_assembled_operator_device_nnz": metadata.get("device_nnz"),
        "xblock_assembled_operator_device_csr_nbytes_estimate": metadata.get(
            "device_csr_nbytes_estimate"
        ),
        "xblock_assembled_operator_device_validation_rel_errors": metadata.get(
            "device_validation_rel_errors",
            (),
        ),
        "xblock_assembled_operator_device_error": metadata.get("device_error"),
        "xblock_assembled_operator_row_equilibration_enabled": bool(
            context.row_equilibration_enabled
        ),
        "xblock_assembled_operator_row_equilibration_built": bool(
            context.row_equilibration_built
        ),
        "xblock_assembled_operator_row_equilibration_norm": row_metadata.get("norm"),
        "xblock_assembled_operator_row_equilibration_setup_s": row_metadata.get(
            "setup_s"
        ),
        "xblock_assembled_operator_row_equilibration_zero_or_tiny_rows": (
            row_metadata.get("zero_or_tiny_rows")
        ),
        "xblock_assembled_operator_row_equilibration_row_norm_min": row_metadata.get(
            "row_norm_min"
        ),
        "xblock_assembled_operator_row_equilibration_row_norm_max": row_metadata.get(
            "row_norm_max"
        ),
        "xblock_assembled_operator_row_equilibration_scale_min": row_metadata.get(
            "row_scale_min"
        ),
        "xblock_assembled_operator_row_equilibration_scale_max": row_metadata.get(
            "row_scale_max"
        ),
        "xblock_assembled_operator_col_equilibration_enabled": bool(
            context.col_equilibration_enabled
        ),
        "xblock_assembled_operator_col_equilibration_built": bool(
            context.col_equilibration_built
        ),
        "xblock_assembled_operator_col_equilibration_norm": col_metadata.get("norm"),
        "xblock_assembled_operator_col_equilibration_setup_s": col_metadata.get(
            "setup_s"
        ),
        "xblock_assembled_operator_col_equilibration_zero_or_tiny_columns": (
            col_metadata.get("zero_or_tiny_columns")
        ),
        "xblock_assembled_operator_col_equilibration_col_norm_min": col_metadata.get(
            "col_norm_min"
        ),
        "xblock_assembled_operator_col_equilibration_col_norm_max": col_metadata.get(
            "col_norm_max"
        ),
        "xblock_assembled_operator_col_equilibration_scale_min": col_metadata.get(
            "col_scale_min"
        ),
        "xblock_assembled_operator_col_equilibration_scale_max": col_metadata.get(
            "col_scale_max"
        ),
        "xblock_assembled_operator_max_colors": metadata.get("max_colors"),
        "xblock_assembled_operator_validation_rel_errors": metadata.get(
            "validation_rel_errors",
            (),
        ),
        "xblock_assembled_operator_error": metadata.get("error"),
    }


def xblock_coarse_correction_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return moment-Schur, two-level, and global-coupling diagnostics."""

    moment_metadata = scope["moment_schur_metadata"]
    moment_stats = scope["moment_schur_stats"]
    two_level_metadata = scope["two_level_metadata"]
    two_level_stats = scope["two_level_stats"]
    global_metadata = scope["global_coupling_metadata"]
    global_stats = scope["global_coupling_stats"]
    for name, value in (
        ("moment_schur_metadata", moment_metadata),
        ("moment_schur_stats", moment_stats),
        ("two_level_metadata", two_level_metadata),
        ("two_level_stats", two_level_stats),
        ("global_coupling_metadata", global_metadata),
        ("global_coupling_stats", global_stats),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")

    return {
        "xblock_moment_schur_enabled": bool(scope["moment_schur_enabled"]),
        "xblock_moment_schur_built": bool(scope["moment_schur_built"]),
        "xblock_moment_schur_used": bool(scope["moment_schur_used"]),
        "xblock_moment_schur_reason": scope["moment_schur_reason"],
        "xblock_moment_schur_default_blocked_by_compact_factors": bool(
            scope["moment_schur_default_blocked_by_compact_factors"]
        ),
        "xblock_moment_schur_mode": moment_metadata.get("mode"),
        "xblock_moment_schur_rank": moment_metadata.get("rank"),
        "xblock_moment_schur_extra_size": moment_metadata.get("extra_size"),
        "xblock_moment_schur_setup_s": moment_metadata.get("setup_s"),
        "xblock_moment_schur_expected_size": moment_metadata.get("expected_size"),
        "xblock_moment_schur_rcond": moment_metadata.get("rcond"),
        "xblock_moment_schur_singular_value_proxy": moment_metadata.get(
            "singular_value_proxy",
            (),
        ),
        "xblock_moment_schur_device_resident": bool(
            moment_metadata.get("device_resident", False)
        ),
        "xblock_moment_schur_probe_residual_before": scope[
            "moment_schur_probe_residual_before"
        ],
        "xblock_moment_schur_probe_residual_after": scope[
            "moment_schur_probe_residual_after"
        ],
        "xblock_moment_schur_probe_improvement_ratio": scope[
            "moment_schur_probe_improvement_ratio"
        ],
        "xblock_moment_schur_error": moment_metadata.get("error"),
        "xblock_moment_schur_applies": int(moment_stats.get("applies", 0)),
        "xblock_moment_schur_base_applies": int(moment_stats.get("base_applies", 0)),
        "xblock_two_level_enabled": bool(scope["two_level_enabled"]),
        "xblock_two_level_built": bool(scope["two_level_built"]),
        "xblock_two_level_mode": two_level_metadata.get("mode"),
        "xblock_two_level_basis_size": two_level_metadata.get("basis_size"),
        "xblock_two_level_rank": two_level_metadata.get("rank"),
        "xblock_two_level_setup_s": two_level_metadata.get("setup_s"),
        "xblock_two_level_rcond": two_level_metadata.get("rcond"),
        "xblock_two_level_basis_names": two_level_metadata.get("basis_names", ()),
        "xblock_two_level_active_projected": bool(
            two_level_metadata.get("active_projected", False)
        ),
        "xblock_two_level_expected_size": two_level_metadata.get("expected_size"),
        "xblock_two_level_error": two_level_metadata.get("error"),
        "xblock_two_level_applies": int(two_level_stats.get("applies", 0)),
        "xblock_two_level_coarse_applies": int(
            two_level_stats.get("coarse_applies", 0)
        ),
        "xblock_global_coupling_enabled": bool(scope["global_coupling_enabled"]),
        "xblock_global_coupling_built": bool(scope["global_coupling_built"]),
        "xblock_global_coupling_mode": global_metadata.get("mode"),
        "xblock_global_coupling_load_basis_size": global_metadata.get(
            "load_basis_size"
        ),
        "xblock_global_coupling_basis_size": global_metadata.get("basis_size"),
        "xblock_global_coupling_rank": global_metadata.get("rank"),
        "xblock_global_coupling_setup_s": global_metadata.get("setup_s"),
        "xblock_global_coupling_setup_budget_s": global_metadata.get("setup_budget_s"),
        "xblock_global_coupling_setup_budget_reached": bool(
            global_metadata.get("setup_budget_reached", False)
        ),
        "xblock_global_coupling_rcond": global_metadata.get("rcond"),
        "xblock_global_coupling_coarse_solver": global_metadata.get("coarse_solver"),
        "xblock_global_coupling_smoother": global_metadata.get("smoother"),
        "xblock_global_coupling_ridge": global_metadata.get("ridge"),
        "xblock_global_coupling_singular_values": global_metadata.get(
            "singular_values",
            (),
        ),
        "xblock_global_coupling_device_resident": bool(
            global_metadata.get("device_resident", False)
        ),
        "xblock_global_coupling_fsavg_lmax": global_metadata.get("fsavg_lmax"),
        "xblock_global_coupling_angular_lmax": global_metadata.get("angular_lmax"),
        "xblock_global_coupling_basis_names": global_metadata.get("basis_names", ()),
        "xblock_global_coupling_error": global_metadata.get("error"),
        "xblock_global_coupling_applies": int(global_stats.get("applies", 0)),
        "xblock_global_coupling_coarse_applies": int(
            global_stats.get("coarse_applies", 0)
        ),
    }


def xblock_qi_seed_preconditioner_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return QI seed, Galerkin, and two-level preconditioner diagnostics."""

    galerkin_stats = scope["qi_galerkin_stats"]
    two_level_stats = scope["qi_two_level_stats"]
    if not isinstance(galerkin_stats, Mapping):
        raise TypeError("qi_galerkin_stats must be a mapping")
    if not isinstance(two_level_stats, Mapping):
        raise TypeError("qi_two_level_stats must be a mapping")

    return {
        "xblock_initial_seed_used": bool(scope["xblock_initial_seed_used"]),
        "xblock_initial_seed_residual_norm": scope["xblock_initial_seed_residual_norm"],
        "xblock_initial_seed_residual_ratio": scope[
            "xblock_initial_seed_residual_ratio"
        ],
        "xblock_moment_schur_seed_enabled": bool(scope["moment_schur_seed_enabled"]),
        "xblock_moment_schur_seed_used": bool(scope["moment_schur_seed_used"]),
        "xblock_moment_schur_seed_residual_norm": scope[
            "moment_schur_seed_residual_norm"
        ],
        "xblock_moment_schur_seed_residual_ratio": scope[
            "moment_schur_seed_residual_ratio"
        ],
        "xblock_qi_coarse_seed_enabled": bool(scope["qi_coarse_seed_enabled"]),
        "xblock_qi_coarse_seed_used": bool(scope["qi_coarse_seed_used"]),
        "xblock_qi_coarse_seed_residual_before": scope[
            "qi_coarse_seed_residual_before"
        ],
        "xblock_qi_coarse_seed_residual_after": scope["qi_coarse_seed_residual_after"],
        "xblock_qi_coarse_seed_improvement_ratio": scope[
            "qi_coarse_seed_improvement_ratio"
        ],
        "xblock_qi_coarse_seed_rank": int(scope["qi_coarse_seed_rank"]),
        "xblock_qi_coarse_seed_candidate_count": int(
            scope["qi_coarse_seed_candidate_count"]
        ),
        "xblock_qi_coarse_seed_reason": scope["qi_coarse_seed_reason"],
        "xblock_qi_coarse_seed_labels": scope["qi_coarse_seed_labels"],
        "xblock_qi_coarse_seed_s": float(scope["qi_coarse_seed_s"]),
        "xblock_qi_coarse_seed_basis": scope["qi_seed_basis_kind"],
        "xblock_qi_coarse_seed_max_candidates": int(scope["qi_seed_max_candidates"]),
        "xblock_qi_coarse_seed_max_angular_mode": int(
            scope["qi_seed_max_angular_mode"]
        ),
        "xblock_qi_galerkin_preconditioner_enabled": bool(
            scope["qi_galerkin_preconditioner_enabled"]
        ),
        "xblock_qi_galerkin_preconditioner_built": bool(
            scope["qi_galerkin_preconditioner_built"]
        ),
        "xblock_qi_galerkin_preconditioner_used": bool(
            scope["qi_galerkin_preconditioner_used"]
        ),
        "xblock_qi_galerkin_preconditioner_reason": scope[
            "qi_galerkin_preconditioner_reason"
        ],
        "xblock_qi_galerkin_preconditioner_mode": scope[
            "qi_galerkin_preconditioner_mode"
        ],
        "xblock_qi_galerkin_preconditioner_rank": int(
            scope["qi_galerkin_preconditioner_rank"]
        ),
        "xblock_qi_galerkin_preconditioner_candidate_count": int(
            scope["qi_galerkin_preconditioner_candidate_count"]
        ),
        "xblock_qi_galerkin_preconditioner_coarse_operator_shape": scope[
            "qi_galerkin_preconditioner_coarse_shape"
        ],
        "xblock_qi_galerkin_preconditioner_coarse_operator_norm": float(
            scope["qi_galerkin_preconditioner_coarse_norm"]
        ),
        "xblock_qi_galerkin_preconditioner_rcond": float(
            scope["qi_galerkin_preconditioner_rcond"]
        ),
        "xblock_qi_galerkin_preconditioner_damping": float(
            scope["qi_galerkin_preconditioner_damping"]
        ),
        "xblock_qi_galerkin_preconditioner_basis_reused_from_seed": bool(
            scope["qi_galerkin_preconditioner_basis_reused_from_seed"]
        ),
        "xblock_qi_galerkin_preconditioner_residual_before": scope[
            "qi_galerkin_preconditioner_residual_before"
        ],
        "xblock_qi_galerkin_preconditioner_residual_after": scope[
            "qi_galerkin_preconditioner_residual_after"
        ],
        "xblock_qi_galerkin_preconditioner_improvement_ratio": scope[
            "qi_galerkin_preconditioner_improvement_ratio"
        ],
        "xblock_qi_galerkin_preconditioner_probe_reduced": bool(
            scope["qi_galerkin_preconditioner_probe_reduced"]
        ),
        "xblock_qi_galerkin_preconditioner_probe_candidates": scope[
            "qi_galerkin_preconditioner_probe_candidates"
        ],
        "xblock_qi_galerkin_preconditioner_selected_index": scope[
            "qi_galerkin_preconditioner_selected_index"
        ],
        "xblock_qi_galerkin_preconditioner_setup_s": float(
            scope["qi_galerkin_preconditioner_setup_s"]
        ),
        "xblock_qi_galerkin_preconditioner_applies": int(
            galerkin_stats.get("applies", 0)
        ),
        "xblock_qi_galerkin_preconditioner_coarse_applies": int(
            galerkin_stats.get("coarse_applies", 0)
        ),
        "xblock_qi_galerkin_preconditioner_base_applies": int(
            galerkin_stats.get("base_applies", 0)
        ),
        "xblock_qi_two_level_preconditioner_enabled": bool(
            scope["qi_two_level_preconditioner_enabled"]
        ),
        "xblock_qi_two_level_preconditioner_built": bool(
            scope["qi_two_level_preconditioner_built"]
        ),
        "xblock_qi_two_level_preconditioner_used": bool(
            scope["qi_two_level_preconditioner_used"]
        ),
        "xblock_qi_two_level_preconditioner_reason": scope[
            "qi_two_level_preconditioner_reason"
        ],
        "xblock_qi_two_level_preconditioner_rank": int(
            scope["qi_two_level_preconditioner_rank"]
        ),
        "xblock_qi_two_level_preconditioner_candidate_count": int(
            scope["qi_two_level_preconditioner_candidate_count"]
        ),
        "xblock_qi_two_level_preconditioner_coarse_operator_shape": scope[
            "qi_two_level_preconditioner_coarse_shape"
        ],
        "xblock_qi_two_level_preconditioner_coarse_operator_norm": float(
            scope["qi_two_level_preconditioner_coarse_norm"]
        ),
        "xblock_qi_two_level_preconditioner_operator_on_basis_shape": scope[
            "qi_two_level_preconditioner_operator_on_basis_shape"
        ],
        "xblock_qi_two_level_preconditioner_operator_on_basis_norm": float(
            scope["qi_two_level_preconditioner_operator_on_basis_norm"]
        ),
        "xblock_qi_two_level_preconditioner_coarse_solver": scope[
            "qi_two_level_preconditioner_coarse_solver"
        ],
        "xblock_qi_two_level_preconditioner_residual_augmented": bool(
            scope["qi_two_level_preconditioner_residual_augmented"]
        ),
        "xblock_qi_two_level_preconditioner_rank_before_augmentation": int(
            scope["qi_two_level_preconditioner_rank_before_augmentation"]
        ),
        "xblock_qi_two_level_preconditioner_augmentation_labels": scope[
            "qi_two_level_preconditioner_augmentation_labels"
        ],
        "xblock_qi_two_level_preconditioner_residual_augment_max_extra": int(
            scope["qi_two_level_preconditioner_residual_augment_max_extra"]
        ),
        "xblock_qi_two_level_preconditioner_residual_augment_steps": int(
            scope["qi_two_level_preconditioner_residual_augment_steps"]
        ),
        "xblock_qi_two_level_preconditioner_residual_augment_include_residuals": bool(
            scope["qi_two_level_preconditioner_residual_augment_include_residuals"]
        ),
        "xblock_qi_two_level_preconditioner_smoothed_load_basis": bool(
            scope["qi_two_level_preconditioner_smoothed_load_basis"]
        ),
        "xblock_qi_two_level_preconditioner_smoothed_load_metadata": scope[
            "qi_two_level_preconditioner_smoothed_load_metadata"
        ],
        "xblock_qi_two_level_preconditioner_rcond": float(
            scope["qi_two_level_preconditioner_rcond"]
        ),
        "xblock_qi_two_level_preconditioner_damping": float(
            scope["qi_two_level_preconditioner_damping"]
        ),
        "xblock_qi_two_level_preconditioner_basis_reused_from_seed": bool(
            scope["qi_two_level_preconditioner_basis_reused_from_seed"]
        ),
        "xblock_qi_two_level_preconditioner_residual_before": scope[
            "qi_two_level_preconditioner_residual_before"
        ],
        "xblock_qi_two_level_preconditioner_residual_after": scope[
            "qi_two_level_preconditioner_residual_after"
        ],
        "xblock_qi_two_level_preconditioner_improvement_ratio": scope[
            "qi_two_level_preconditioner_improvement_ratio"
        ],
        "xblock_qi_two_level_preconditioner_probe_candidates": scope[
            "qi_two_level_preconditioner_probe_candidates"
        ],
        "xblock_qi_two_level_preconditioner_selected_index": scope[
            "qi_two_level_preconditioner_selected_index"
        ],
        "xblock_qi_two_level_preconditioner_setup_s": float(
            scope["qi_two_level_preconditioner_setup_s"]
        ),
        "xblock_qi_two_level_preconditioner_applies": int(
            two_level_stats.get("applies", 0)
        ),
        "xblock_qi_two_level_preconditioner_local_applies": int(
            two_level_stats.get("local_applies", 0)
        ),
    }


def xblock_device_krylov_diagnostics(scope: Mapping[str, object]) -> dict[str, object]:
    """Return device-Krylov, host-fallback, and transfer-free diagnostics."""

    method = str(scope["xblock_krylov_method"])
    device_methods = scope["xblock_device_krylov_methods"]
    global_metadata = scope["global_coupling_metadata"]
    if not isinstance(global_metadata, Mapping):
        raise TypeError("global_coupling_metadata must be a mapping")

    host_transfer_free_base = bool(
        method in device_methods
        and bool(scope["xblock_jax_factors"])
        and (
            not bool(scope["assembled_operator_built"])
            or bool(scope["assembled_operator_device_resident"])
        )
        and not bool(scope["two_level_built"])
        and (
            not bool(scope["global_coupling_built"])
            or bool(global_metadata.get("device_resident", False))
        )
    )
    fallback_decision = scope["xblock_device_host_fallback_decision"]
    operator_reuse_decision = scope["xblock_qi_device_operator_reuse_decision"]
    fgmres_jit_active = bool(
        method in {"fgmres_jax", "gmres_jax"}
        and bool(scope["xblock_device_fgmres_jit"])
    )

    return {
        "xblock_device_krylov_method": method if method in device_methods else None,
        "xblock_device_host_fallback_mode": str(fallback_decision.mode),
        "xblock_device_host_fallback_used": bool(fallback_decision.used),
        "xblock_device_host_fallback_reason": str(fallback_decision.reason),
        "xblock_device_host_fallback_requested_method": str(
            fallback_decision.requested_method
        ),
        "xblock_device_host_fallback_requested_env": str(
            scope["xblock_krylov_env_requested"]
        ),
        "xblock_device_host_fallback_effective_krylov_env_value": str(
            fallback_decision.effective_krylov_env_value
        ),
        "xblock_device_host_fallback_min_active_size": int(
            fallback_decision.min_active_size
        ),
        "xblock_device_host_fallback_qi_like_full_fp_3d": bool(
            fallback_decision.qi_like_full_fp_3d
        ),
        "xblock_device_host_fallback_ignored_env": bool(fallback_decision.ignored_env),
        "xblock_device_host_fallback_auto_disabled_by_qi_device": bool(
            scope["xblock_device_host_fallback_auto_disabled_by_qi_device"]
        ),
        "xblock_device_host_fallback_non_autodiff": bool(
            fallback_decision.non_autodiff
        ),
        "xblock_qi_device_operator_reuse": operator_reuse_decision.to_metadata(),
        "xblock_qi_device_operator_reuse_enabled": bool(
            operator_reuse_decision.enabled
        ),
        "xblock_qi_device_operator_reuse_reason": str(operator_reuse_decision.reason),
        "xblock_qi_device_operator_reuse_skip_xblock_factors": bool(
            operator_reuse_decision.skip_xblock_factors
        ),
        "xblock_device_gmres_enabled": bool(method == "gmres_jax"),
        "xblock_device_fgmres_enabled": bool(method == "fgmres_jax"),
        "xblock_device_fgmres_jit_enabled": fgmres_jit_active,
        "xblock_device_fgmres_jit_mode": (
            scope["xblock_device_fgmres_jit_mode"] if fgmres_jit_active else None
        ),
        "xblock_device_fgmres_jit_outer_k": (
            int(scope["xblock_device_fgmres_jit_outer_k"])
            if fgmres_jit_active and scope["xblock_device_fgmres_jit_mode"] == "cycle"
            else 0
        ),
        "xblock_device_fgmres_qi_augmented_krylov_requested": bool(
            scope["qi_device_augmented_krylov_requested"]
        ),
        "xblock_device_fgmres_qi_augmented_krylov_used": bool(
            scope["qi_device_augmented_krylov_used"]
        ),
        "xblock_device_fgmres_qi_augmented_krylov_rank": int(
            scope["qi_device_augmented_krylov_rank"]
        ),
        "xblock_device_fgmres_qi_augmented_krylov_reason": scope[
            "qi_device_augmented_krylov_reason"
        ],
        "xblock_device_fgmres_qi_augmented_krylov_mode": scope[
            "qi_device_augmented_krylov_mode"
        ],
        "xblock_device_fgmres_qi_augmented_seed_requested": bool(
            scope["qi_device_augmented_seed_requested"]
        ),
        "xblock_device_fgmres_qi_augmented_seed_available": bool(
            scope["qi_device_augmented_seed_available"]
        ),
        "xblock_device_fgmres_qi_augmented_seed_used": bool(
            scope["qi_device_augmented_seed_used"]
        ),
        "xblock_device_fgmres_qi_augmented_seed_rank": int(
            scope["qi_device_augmented_seed_rank"]
        ),
        "xblock_device_fgmres_qi_augmented_seed_max_rank": int(
            scope["qi_device_augmented_seed_max_rank"]
        ),
        "xblock_device_fgmres_qi_augmented_seed_reason": scope[
            "qi_device_augmented_seed_reason"
        ],
        "xblock_device_fgmres_qi_augmented_seed_projection_residual_norm": scope[
            "qi_device_augmented_seed_projection_residual"
        ],
        "xblock_device_fgmres_qi_augmented_seed_labels": scope[
            "qi_device_augmented_seed_labels"
        ],
        "xblock_device_bicgstab_enabled": bool(method == "bicgstab_jax"),
        "xblock_device_tfqmr_enabled": bool(method == "tfqmr_jax"),
        "xblock_device_tfqmr_replacement_interval": int(
            scope["tfqmr_replacement_interval"]
        ),
        "xblock_device_krylov_forced_jax_factors": bool(
            scope["xblock_device_krylov_forced_jax_factors"]
        ),
        "xblock_device_fgmres_forced_jax_factors": bool(
            scope["xblock_device_krylov_forced_jax_factors"]
        ),
        "xblock_device_fgmres_forced_right_pc": bool(
            scope["xblock_device_fgmres_forced_right_pc"]
        ),
        "xblock_device_fgmres_block_between_cycles": bool(
            scope["fgmres_block_between_cycles"]
        ),
        "xblock_estimated_gmres_basis_nbytes": int(
            scope["xblock_estimated_gmres_basis_nbytes"]
        ),
        "xblock_estimated_bicgstab_work_nbytes": int(
            scope["xblock_estimated_bicgstab_work_nbytes"]
        ),
        "xblock_estimated_tfqmr_work_nbytes": int(
            scope["xblock_estimated_tfqmr_work_nbytes"]
        ),
        "xblock_device_krylov_host_transfer_free": host_transfer_free_base,
        "xblock_device_fgmres_host_transfer_free": bool(
            method == "fgmres_jax" and host_transfer_free_base
        ),
        "xblock_device_bicgstab_host_transfer_free": bool(
            method == "bicgstab_jax" and host_transfer_free_base
        ),
        "xblock_device_tfqmr_host_transfer_free": bool(
            method == "tfqmr_jax" and host_transfer_free_base
        ),
    }


@dataclass(frozen=True, slots=True)
class XBlockSparsePCCoreDiagnosticsContext:
    """Explicit inputs for top-level x-block sparse-PC diagnostics."""

    solver_kind: object
    accepted_converged: object
    reported_iterations: object
    reported_matvecs: object
    python_matvecs: object
    device_cycle_estimated_matvecs: object
    krylov_method: object
    candidate_krylov_method: object
    candidate_iterations: object
    candidate_matvecs: object
    candidate_residual_norm: object
    fallback_started_from_candidate: object
    fallback_candidate_improved_rhs: object
    precondition_side: object
    default_right_preconditioned: object
    default_short_restart_capped: object
    gmres_restart: object
    gmres_maxiter: object
    setup_s: object
    solve_s: object
    elapsed_s: object
    sparse_pc_factor_s: object
    preconditioner_xi: object
    preconditioner_built: object
    assembled_host: object
    jax_factors: object
    jax_factor_format: object
    jax_factor_apply: object
    lower_fill_mode: object
    lower_fill_ignored_env: object


def xblock_sparse_pc_core_diagnostics(
    context: XBlockSparsePCCoreDiagnosticsContext,
) -> dict[str, object]:
    """Return top-level x-block sparse-PC solve diagnostics."""

    method = str(context.krylov_method)
    candidate_method = str(context.candidate_krylov_method)
    device_estimated_matvecs = context.device_cycle_estimated_matvecs
    jax_factors = bool(context.jax_factors)
    lower_fill_mode = str(context.lower_fill_mode)

    return {
        "solver_kind": context.solver_kind,
        "residual_kind": "true_residual",
        "accepted_converged": bool(context.accepted_converged),
        "acceptance_criterion": "true_residual",
        "iterations": int(context.reported_iterations),
        "matvecs": int(context.reported_matvecs),
        "python_matvecs": int(context.python_matvecs),
        "device_cycle_estimated_matvecs": (
            None if device_estimated_matvecs is None else int(device_estimated_matvecs)
        ),
        "krylov_method": method,
        "candidate_krylov_method": candidate_method,
        "candidate_iterations": int(context.candidate_iterations),
        "candidate_matvecs": int(context.candidate_matvecs),
        "candidate_residual_norm": float(context.candidate_residual_norm),
        "fallback_from_krylov_method": (
            candidate_method if candidate_method != method else None
        ),
        "fallback_started_from_candidate": bool(
            context.fallback_started_from_candidate
        ),
        "fallback_candidate_improved_rhs": bool(
            context.fallback_candidate_improved_rhs
        ),
        "precondition_side": str(context.precondition_side),
        "default_right_preconditioned": bool(context.default_right_preconditioned),
        "default_short_restart_capped": bool(context.default_short_restart_capped),
        "gmres_restart": int(context.gmres_restart),
        "gmres_maxiter": int(context.gmres_maxiter),
        "setup_s": float(context.setup_s),
        "solve_s": float(context.solve_s),
        "elapsed_s": float(context.elapsed_s),
        "sparse_pc_factor_s": float(context.sparse_pc_factor_s),
        "sparse_pc_xblock_preconditioner_xi": int(context.preconditioner_xi),
        "sparse_pc_xblock_preconditioner_built": bool(context.preconditioner_built),
        "sparse_pc_xblock_assembled_host": bool(context.assembled_host),
        "sparse_pc_xblock_jax_factors": jax_factors,
        "sparse_pc_xblock_jax_factor_format": (
            str(context.jax_factor_format) if jax_factors else None
        ),
        "sparse_pc_xblock_jax_factor_apply": (
            str(context.jax_factor_apply) if jax_factors else None
        ),
        "xblock_lower_fill_mode": lower_fill_mode,
        "xblock_lower_fill_requested": lower_fill_mode in {"probe", "force"},
        "xblock_lower_fill_ignored_env": bool(context.lower_fill_ignored_env),
    }


def xblock_sparse_pc_result_diagnostics_from_driver_state(
    state: Mapping[str, object],
    *,
    full_size: object,
) -> dict[str, object]:
    """Build final x-block sparse-PC diagnostics from the driver solve state.

    This is a transitional boundary: the driver now has one explicit metadata
    handoff instead of many scattered ``locals()`` calls, while the component
    helpers keep the stable public keys and the remaining coarse/QI payloads
    can be typed incrementally.
    """

    return {
        **xblock_sparse_pc_core_diagnostics(
            XBlockSparsePCCoreDiagnosticsContext(
                solver_kind=state["xblock_solver_kind"],
                accepted_converged=state["accepted_converged_xblock"],
                reported_iterations=state["reported_iterations"],
                reported_matvecs=state["reported_matvecs"],
                python_matvecs=state["mv_count"],
                device_cycle_estimated_matvecs=state[
                    "device_krylov_estimated_matvecs"
                ],
                krylov_method=state["xblock_krylov_method"],
                candidate_krylov_method=state["candidate_krylov_method"],
                candidate_iterations=state["candidate_iterations"],
                candidate_matvecs=state["candidate_matvecs"],
                candidate_residual_norm=state["candidate_residual_norm"],
                fallback_started_from_candidate=state[
                    "fallback_started_from_candidate"
                ],
                fallback_candidate_improved_rhs=state[
                    "fallback_candidate_improved_rhs"
                ],
                precondition_side=state["precondition_side"],
                default_right_preconditioned=state["xblock_default_right_pc"],
                default_short_restart_capped=state["xblock_default_restart_capped"],
                gmres_restart=state["pc_restart"],
                gmres_maxiter=state["pc_maxiter"],
                setup_s=state["setup_s"],
                solve_s=state["solve_s"],
                elapsed_s=state["sparse_timer"].elapsed_s(),
                sparse_pc_factor_s=state["pc_factor_s"],
                preconditioner_xi=state["xblock_preconditioner_xi"],
                preconditioner_built=state["xblock_preconditioner_built"],
                assembled_host=state["xblock_assembled_host_fp"],
                jax_factors=state["xblock_jax_factors"],
                jax_factor_format=state["xblock_jax_factor_format"],
                jax_factor_apply=state["xblock_jax_factor_apply"],
                lower_fill_mode=state["xblock_lower_fill_mode"],
                lower_fill_ignored_env=state["xblock_lower_fill_ignored_env"],
            )
        ),
        **xblock_device_krylov_diagnostics(state),
        "xblock_active_dof": bool(state["xblock_use_active_dof"]),
        "xblock_linear_size": int(state["xblock_linear_size"]),
        "xblock_full_size": int(full_size),
        **xblock_assembled_operator_diagnostics(
            XBlockAssembledOperatorDiagnosticsContext(
                enabled=state["assembled_operator_enabled"],
                built=state["assembled_operator_built"],
                metadata=state["assembled_operator_metadata"],
                row_equilibration_enabled=state[
                    "xblock_row_equilibration_enabled"
                ],
                row_equilibration_built=state["xblock_row_equilibration_built"],
                row_equilibration_metadata=state[
                    "xblock_row_equilibration_metadata"
                ],
                col_equilibration_enabled=state[
                    "xblock_col_equilibration_enabled"
                ],
                col_equilibration_built=state["xblock_col_equilibration_built"],
                col_equilibration_metadata=state[
                    "xblock_col_equilibration_metadata"
                ],
            )
        ),
        **xblock_coarse_correction_diagnostics(state),
        **xblock_qi_seed_preconditioner_diagnostics(state),
        **xblock_qi_device_preconditioner_diagnostics(state),
        **xblock_qi_deflated_preconditioner_diagnostics(state),
        **xblock_side_probe_diagnostics(
            XBlockSideProbeDiagnosticsContext(
                enabled=state["xblock_side_probe_enabled"],
                used=state["xblock_side_probe_used"],
                switched=state["xblock_side_probe_switched"],
                switch_suppressed_by_global_coupling=state[
                    "xblock_side_probe_switch_suppressed_by_global_coupling"
                ],
                switch_suppressed_by_explicit_side=state[
                    "xblock_side_probe_switch_suppressed_by_explicit_side"
                ],
                physical_seed_preserved_after_switch=state[
                    "xblock_side_probe_physical_seed_preserved_after_switch"
                ],
                seed_used=state["xblock_side_probe_seed_used"],
                seed_residual_norm=state["xblock_side_probe_seed_residual_norm"],
                initial_side=state["xblock_side_probe_initial_side"],
                selected_side=state["xblock_side_probe_selected_side"],
                initial_method=state["xblock_side_probe_initial_method"],
                selected_method=state["xblock_side_probe_selected_method"],
                lgmres_rescue=state["xblock_side_probe_lgmres_rescue"],
                lgmres_rescue_maxiter_capped=state[
                    "xblock_lgmres_rescue_maxiter_capped"
                ],
                lgmres_rescue_outer_k=state["xblock_lgmres_rescue_outer_k"],
                residual_norm=state["xblock_side_probe_residual_norm"],
                residual_ratio=state["xblock_side_probe_residual_ratio"],
                iterations=state["xblock_side_probe_iterations"],
                matvecs=state["xblock_side_probe_matvecs"],
                elapsed_s=state["xblock_side_probe_s"],
            )
        ),
    }
