from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import sfincs_jax.problems.profile_diagnostics as profile_diagnostics
from sfincs_jax.problems.profile_diagnostics import (
    SparsePCDirectTailMetadataContext,
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    SparseRescueTailMetadataContext,
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockCoarseCorrectionDiagnosticsContext,
    XBlockSparsePCCoreDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    fortran_reduced_xblock_result_metadata,
    record_structured_fblock_preconditioner_metadata,
    sparse_pc_direct_tail_result_metadata,
    sparse_pc_direct_tail_result_metadata_from_context,
    sparse_pc_factor_preflight_result_metadata,
    sparse_pc_factor_preflight_result_metadata_from_context,
    sparse_pc_gmres_result_metadata,
    sparse_pc_gmres_static_metadata,
    sparse_pc_gmres_static_metadata_from_context,
    sparse_pc_pattern_result_metadata,
    sparse_pc_pattern_result_metadata_from_context,
    sparse_rescue_tail_metadata,
    sparse_rescue_tail_metadata_from_context,
    sparse_xblock_rescue_metadata,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_coarse_correction_diagnostics_from_context,
    xblock_device_krylov_diagnostics,
    xblock_sparse_pc_core_diagnostics,
    xblock_sparse_pc_result_diagnostics_from_solve_state,
    xblock_side_probe_diagnostics,
)


def test_record_structured_fblock_preconditioner_metadata_ignores_missing_metadata() -> None:
    metadata: dict[str, object] = {"existing": True}

    record_structured_fblock_preconditioner_metadata(
        target=metadata,
        preconditioner=object(),
    )

    assert metadata == {"existing": True}


def test_record_structured_fblock_preconditioner_metadata_records_assembly_summary() -> None:
    metadata: dict[str, object] = {}

    def preconditioner(x):
        return x

    preconditioner._sfincs_jax_structured_fblock_metadata = {
        "selected": True,
        "reason": "unit-test",
        "assembly": {"nnz_blocks": 7, "data_nbytes": 128},
    }

    record_structured_fblock_preconditioner_metadata(
        target=metadata,
        preconditioner=preconditioner,
    )

    assert metadata["structured_fblock_preconditioner_enabled"] is True
    assert metadata["structured_fblock_preconditioner_selected"] is True
    assert metadata["structured_fblock_preconditioner_reason"] == "unit-test"
    assert metadata["structured_fblock_preconditioner_nnz_blocks"] == 7
    assert metadata["structured_fblock_preconditioner_data_nbytes"] == 128
    assert metadata["structured_fblock_preconditioner_metadata"] is preconditioner._sfincs_jax_structured_fblock_metadata


def test_record_structured_fblock_preconditioner_metadata_handles_malformed_assembly() -> None:
    metadata: dict[str, object] = {}

    def preconditioner(x):
        return x

    preconditioner._sfincs_jax_structured_fblock_metadata = {
        "selected": 0,
        "reason": 123,
        "assembly": "not-a-dict",
    }

    record_structured_fblock_preconditioner_metadata(
        target=metadata,
        preconditioner=preconditioner,
    )

    assert metadata["structured_fblock_preconditioner_enabled"] is True
    assert metadata["structured_fblock_preconditioner_selected"] is False
    assert metadata["structured_fblock_preconditioner_reason"] == "123"
    assert metadata["structured_fblock_preconditioner_nnz_blocks"] == 0
    assert metadata["structured_fblock_preconditioner_data_nbytes"] == 0


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
    }


def _sparse_rescue_context() -> SparseRescueTailMetadataContext:
    scope = _sparse_rescue_scope()
    return SparseRescueTailMetadataContext(
        sparse_xblock_rescue_active=scope["sparse_xblock_rescue_active"],
        sparse_xblock_rescue_attempted=scope["sparse_xblock_rescue_attempted"],
        sparse_xblock_rescue_built=scope["sparse_xblock_rescue_built"],
        sparse_xblock_rescue_error=scope["sparse_xblock_rescue_error"],
        sparse_xblock_rescue_reason=scope["sparse_xblock_rescue_reason"],
        sparse_xblock_rescue_assembled_host_fp=scope[
            "sparse_xblock_rescue_assembled_host_fp"
        ],
        sparse_xblock_rescue_preconditioner_xi=scope[
            "sparse_xblock_rescue_preconditioner_xi"
        ],
        sparse_xblock_rescue_seed_residual=scope[
            "sparse_xblock_rescue_seed_residual"
        ],
        sparse_xblock_rescue_seed_improvement_ratio=scope[
            "sparse_xblock_rescue_seed_improvement_ratio"
        ],
        sparse_xblock_rescue_seed_accept_ratio=scope[
            "sparse_xblock_rescue_seed_accept_ratio"
        ],
        sparse_xblock_rescue_seed_refine_steps=scope[
            "sparse_xblock_rescue_seed_refine_steps"
        ],
        sparse_xblock_rescue_seed_refines_performed=scope[
            "sparse_xblock_rescue_seed_refines_performed"
        ],
        sparse_xblock_rescue_candidate_residual=scope[
            "sparse_xblock_rescue_candidate_residual"
        ],
        sparse_xblock_rescue_candidate_accepted=scope[
            "sparse_xblock_rescue_candidate_accepted"
        ],
    )


def test_sparse_rescue_metadata_helpers_preserve_driver_keys_and_types() -> None:
    scope = _sparse_rescue_scope()
    context = _sparse_rescue_context()

    sparse_meta = sparse_xblock_rescue_metadata(scope)
    combined = sparse_rescue_tail_metadata(scope)
    combined_context = sparse_rescue_tail_metadata_from_context(context)

    assert sparse_meta["sparse_xblock_rescue_active"] is True
    assert sparse_meta["sparse_xblock_rescue_reason"] == "seed_rejected"
    assert sparse_meta["sparse_xblock_rescue_candidate_accepted"] is True
    assert combined == sparse_meta
    assert combined_context == combined


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
    scope = {
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
    metadata = xblock_coarse_correction_diagnostics(scope)
    context_metadata = xblock_coarse_correction_diagnostics_from_context(
        XBlockCoarseCorrectionDiagnosticsContext(
            moment_schur_enabled=scope["moment_schur_enabled"],
            moment_schur_built=scope["moment_schur_built"],
            moment_schur_used=scope["moment_schur_used"],
            moment_schur_reason=scope["moment_schur_reason"],
            moment_schur_default_blocked_by_compact_factors=scope[
                "moment_schur_default_blocked_by_compact_factors"
            ],
            moment_schur_probe_residual_before=scope[
                "moment_schur_probe_residual_before"
            ],
            moment_schur_probe_residual_after=scope[
                "moment_schur_probe_residual_after"
            ],
            moment_schur_probe_improvement_ratio=scope[
                "moment_schur_probe_improvement_ratio"
            ],
            moment_schur_metadata=scope["moment_schur_metadata"],
            moment_schur_stats=scope["moment_schur_stats"],
            two_level_enabled=scope["two_level_enabled"],
            two_level_built=scope["two_level_built"],
            two_level_metadata=scope["two_level_metadata"],
            two_level_stats=scope["two_level_stats"],
            global_coupling_enabled=scope["global_coupling_enabled"],
            global_coupling_built=scope["global_coupling_built"],
            global_coupling_metadata=scope["global_coupling_metadata"],
            global_coupling_stats=scope["global_coupling_stats"],
        )
    )

    assert context_metadata == metadata
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


def test_xblock_device_krylov_diagnostics_preserve_transfer_free_logic() -> None:
    fallback = SimpleNamespace(
        mode="auto",
        used=False,
        reason="device_ok",
        requested_method="fgmres_jax",
        effective_krylov_env_value="fgmres_jax",
        min_active_size=128,
        large_full_fp_3d=True,
        ignored_env=False,
        non_autodiff=False,
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
            "xblock_device_fgmres_jit": True,
            "xblock_device_fgmres_jit_mode": "cycle",
            "xblock_device_fgmres_jit_outer_k": 7,
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
    assert metadata["xblock_device_fgmres_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_mode"] == "cycle"
    assert metadata["xblock_device_fgmres_jit_outer_k"] == 7
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


class _DefaultDirectTailSuffixes(dict):
    def __missing__(self, key: str) -> object:
        self[key] = 1
        return self[key]


def test_sparse_pc_direct_tail_result_metadata_preserves_driver_conversions() -> None:
    structured_metadata = {"kind": "native"}
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
            "direct_tail_structured_pc_metadata": structured_metadata,
            "direct_tail_error": "not_selected",
            "direct_tail_structured_pc_requested": "auto",
            "direct_tail_structured_pc_reason": "admitted",
            "direct_tail_structured_pc_error": None,
        }
    )

    metadata = sparse_pc_direct_tail_result_metadata(state)
    suffix_values = _DefaultDirectTailSuffixes(
        {
            key.removeprefix("direct_tail_"): value
            for key, value in state.items()
            if key.startswith("direct_tail_")
        }
    )
    context_metadata = sparse_pc_direct_tail_result_metadata_from_context(
        SparsePCDirectTailMetadataContext(
            structured_pc_preflight_required=state[
                "structured_pc_preflight_required"
            ],
            structured_pc_preflight_required_min_size=state[
                "structured_pc_preflight_required_min_size"
            ],
            suffix_values=suffix_values,
            operator_bundle=state["direct_tail_operator_bundle"],
            structured_max_nbytes=state["direct_tail_structured_max_nbytes"],
            enabled=state["direct_tail_enabled"],
            direct_reduced_pmat_requested=state[
                "direct_tail_direct_reduced_pmat_requested"
            ],
            built=state["direct_tail_built"],
            error=state["direct_tail_error"],
            structured_pc_requested=state["direct_tail_structured_pc_requested"],
            structured_pc_required=state["direct_tail_structured_pc_required"],
            structured_pc_selected=state["direct_tail_structured_pc_selected"],
            structured_pc_reason=state["direct_tail_structured_pc_reason"],
            structured_pc_error=state["direct_tail_structured_pc_error"],
            structured_pc_max_mb_auto=state[
                "direct_tail_structured_pc_max_mb_auto"
            ],
            structured_pc_metadata=state["direct_tail_structured_pc_metadata"],
        )
    )

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
    assert not any("true_coupled" in key for key in metadata)
    assert not any("true_active" in key for key in metadata)
    assert not any("residual_window" in key for key in metadata)
    assert context_metadata == metadata


def test_sparse_pc_gmres_result_metadata_preserves_driver_schema() -> None:
    state = _DefaultDirectTailState(
        {
            "op": SimpleNamespace(total_size=np.int64(11)),
            "history": (1.0, 0.1, 0.01),
            "mv_count": np.int64(9),
            "pc_restart": np.int64(7),
            "pc_maxiter": np.int64(25),
            "sparse_pc_first_attempt_maxiter": np.int64(5),
            "sparse_pc_post_minres_steps": np.int64(3),
            "sparse_pc_post_minres_alphas": (0.5, 0.25),
            "sparse_pc_post_minres_alpha_clip": 4.0,
            "sparse_pc_post_minres_min_improvement": 0.2,
            "sparse_pc_post_minres_residual_before": 3.0,
            "sparse_pc_post_minres_residual_after": 1.0,
            "sparse_pc_post_minres_history": (3.0, 1.0),
            "sparse_pc_post_minres_error": None,
            "pc_shift": 1.0e-8,
            "sparse_pc_factor_dtype_used": np.dtype(np.float32),
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_factor_dtype_retry": "float64",
            "factor_preflight_enabled": True,
            "factor_preflight_required": False,
            "factor_preflight_seed_enabled": True,
            "factor_preflight_seed_used": False,
            "factor_preflight_passed": True,
            "factor_preflight_error": None,
            "factor_preflight_residual_before": 4.0,
            "factor_preflight_residual_after": 2.0,
            "factor_preflight_improvement_ratio": 2.0,
            "factor_preflight_target_ratio": 5.0,
            "factor_preflight_max_target_ratio": 10.0,
            "factor_preflight_residual_diagnostics": {"tail": 0.2},
            "fortran_reduced_sparse_pc": True,
            "fortran_reduced_sparse_pc_backend": "global",
            "fortran_reduced_sparse_pc_backend_reason": "direct_tail",
            "fortran_reduced_xblock_min_size": np.int64(13),
            "sparse_pc_preconditioner_operator": "full",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "ilu",
            "sparse_pc_default_ilu_fill_factor": 6.0,
            "sparse_pc_default_ilu_drop_tol": 1.0e-5,
            "sparse_pc_default_pattern_color_batch": np.int64(8),
            "preconditioner_x": np.int64(1),
            "preconditioner_x_min_l": np.int64(2),
            "preconditioner_xi": np.int64(3),
            "preconditioner_species": np.int64(4),
            "sparse_pc_permc_spec": "MMD_AT_PLUS_A",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": np.int64(10),
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 1.25,
            "solve_s": 2.5,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 9.0),
            "summary": SimpleNamespace(nnz=np.int64(30), avg_row_nnz=3.0, max_row_nnz=np.int64(5)),
            "sparse_pattern_scope": "fortran_reduced_active_dof",
            "pattern_build_s": 0.125,
            "pc_factor_s": 0.75,
            "factor_bundle_pc": SimpleNamespace(
                factor_s=0.5,
                factor_nbytes_estimate=np.int64(100),
                factor_nnz_estimate=np.int64(20),
            ),
            "_operator_bundle_pc": SimpleNamespace(
                metadata=SimpleNamespace(
                    nnz_estimate=np.int64(123),
                    csr_nbytes_estimate=np.int64(456),
                )
            ),
            "target": 0.25,
            "residual_norm_sparse_pc": 0.5,
            "sparse_pc_accepted_converged": False,
            "sparse_pc_factor_quality_rejected": True,
            "direct_tail_operator_bundle": None,
            "direct_tail_structured_max_nbytes": None,
            "direct_tail_structured_pc_metadata": {"kind": "active"},
            "direct_tail_error": None,
            "direct_tail_structured_pc_requested": "auto",
            "direct_tail_structured_pc_reason": "selected",
            "direct_tail_structured_pc_error": None,
        }
    )

    metadata = sparse_pc_gmres_result_metadata(state)

    assert metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert metadata["accepted_converged"] is False
    assert metadata["iterations"] == 3
    assert metadata["matvecs"] == 9
    assert metadata["sparse_pc_factor_dtype"] == "float32"
    assert metadata["sparse_pc_initial_factor_dtype"] == "float64"
    assert metadata["sparse_pc_factor_dtype_retry"] == "float64"
    assert metadata["sparse_pc_post_minres_steps_accepted"] == 2
    assert metadata["sparse_pc_post_minres_history"] == (3.0, 1.0)
    assert metadata["sparse_pc_backend"] == "global"
    assert metadata["sparse_pc_xblock_min_size"] == 13
    assert metadata["sparse_pc_full_size"] == 11
    assert metadata["sparse_pc_fp_dense_velocity_block"] is None
    assert metadata["elapsed_s"] == 9.0
    assert metadata["sparse_pattern_nnz"] == 30
    assert metadata["sparse_pc_factor_elapsed_s"] == 0.5
    assert metadata["sparse_pc_factor_nbytes_estimate"] == 100
    assert metadata["sparse_pc_operator_csr_nbytes_estimate"] == 456
    assert metadata["sparse_pc_residual_ratio_to_target"] == 2.0
    assert metadata["sparse_pc_factor_quality_rejected"] is True
    assert metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"] == {"kind": "active"}

    precomputed_tail_metadata = dict(sparse_pc_direct_tail_result_metadata(state))
    precomputed_tail_metadata[
        "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"
    ] = {"kind": "precomputed"}
    precomputed_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("direct_tail_")
    }
    precomputed_state["sparse_pc_direct_tail_metadata"] = precomputed_tail_metadata

    precomputed_metadata = sparse_pc_gmres_result_metadata(precomputed_state)

    assert (
        precomputed_metadata[
            "sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"
        ]
        == {"kind": "precomputed"}
    )


def test_sparse_pc_gmres_static_metadata_covers_global_branch() -> None:
    metadata = sparse_pc_gmres_static_metadata_from_context(
        SparsePCGMRESStaticMetadataContext(
            op=SimpleNamespace(total_size=np.int64(17)),
            fortran_reduced_sparse_pc=False,
            fortran_reduced_sparse_pc_backend="unused",
            fortran_reduced_sparse_pc_backend_reason="unused",
            fortran_reduced_xblock_min_size=np.int64(999),
            pc_restart=np.int64(12),
            pc_maxiter=np.int64(34),
            sparse_pc_first_attempt_maxiter=np.int64(5),
            pc_shift=np.float64(1.0e-9),
            sparse_pc_factor_dtype_initial=np.dtype(np.float32),
            sparse_pc_preconditioner_operator="full",
            sparse_pc_factorization="ilu",
            sparse_pc_default_factor_kind="ilu",
            sparse_pc_default_ilu_fill_factor=np.float64(4.0),
            sparse_pc_default_ilu_drop_tol=np.float64(1.0e-4),
            sparse_pc_default_pattern_color_batch=np.int64(6),
            preconditioner_x=np.int64(1),
            preconditioner_x_min_l=np.int64(2),
            preconditioner_xi=np.int64(3),
            preconditioner_species=np.int64(4),
            sparse_pc_permc_spec="COLAMD",
            sparse_pc_default_permc_spec="MMD_AT_PLUS_A",
            sparse_pc_use_active_dof=1,
            sparse_pc_linear_size=np.int64(13),
            sparse_pc_fp_dense_velocity_block=np.bool_(True),
        )
    )

    assert metadata["solver_kind"] == "sparse_pc_gmres"
    assert metadata["sparse_pc_backend"] == "global"
    assert metadata["sparse_pc_backend_reason"] == "not_fortran_reduced"
    assert metadata["sparse_pc_xblock_min_size"] is None
    assert metadata["sparse_pc_fortran_reduced"] is False
    assert metadata["sparse_pc_full_size"] == 17
    assert metadata["sparse_pc_fp_dense_velocity_block"] is True


def test_sparse_pc_gmres_static_metadata_solve_state_wrapper() -> None:
    metadata = sparse_pc_gmres_static_metadata(
        {
            "op": SimpleNamespace(total_size=19),
            "fortran_reduced_sparse_pc": True,
            "fortran_reduced_sparse_pc_backend": "xblock",
            "fortran_reduced_sparse_pc_backend_reason": "unit",
            "fortran_reduced_xblock_min_size": 10,
            "pc_restart": 4,
            "pc_maxiter": 8,
            "sparse_pc_first_attempt_maxiter": 3,
            "pc_shift": 1.0e-8,
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_preconditioner_operator": "fortran_reduced",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "lu",
            "sparse_pc_default_ilu_fill_factor": 1.0,
            "sparse_pc_default_ilu_drop_tol": 0.0,
            "sparse_pc_default_pattern_color_batch": 1,
            "preconditioner_x": 0,
            "preconditioner_x_min_l": 0,
            "preconditioner_xi": 0,
            "preconditioner_species": 1,
            "sparse_pc_permc_spec": "COLAMD",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": False,
            "sparse_pc_linear_size": 11,
            "sparse_pc_fp_dense_velocity_block": None,
        }
    )

    assert metadata["sparse_pc_fortran_reduced"]
    assert metadata["sparse_pc_backend"] == "xblock"
    assert metadata["sparse_pc_linear_size"] == 11


def test_xblock_sparse_pc_result_diagnostics_solve_state_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_diagnostics,
        "xblock_device_krylov_diagnostics",
        lambda _state: {"xblock_device_krylov_method": "unit"},
    )
    state = {
        "xblock_assembled_operator_result_metadata": {"xblock_assembled_operator_built": False},
        "xblock_coarse_correction_metadata": {"xblock_coarse_correction_selected": False},
        "xblock_side_probe_metadata": {"xblock_side_probe_enabled": False},
        "xblock_solver_kind": "xblock_sparse_pc",
        "accepted_converged_xblock": True,
        "reported_iterations": 2,
        "reported_matvecs": 3,
        "mv_count": 4,
        "device_krylov_estimated_matvecs": 0,
        "xblock_krylov_method": "gmres",
        "candidate_krylov_method": "none",
        "candidate_iterations": 0,
        "candidate_matvecs": 0,
        "candidate_residual_norm": 0.0,
        "fallback_started_from_candidate": False,
        "fallback_candidate_improved_rhs": False,
        "precondition_side": "left",
        "xblock_default_right_pc": False,
        "xblock_default_restart_capped": False,
        "pc_restart": 5,
        "pc_maxiter": 6,
        "setup_s": 0.1,
        "solve_s": 0.2,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: 0.3),
        "pc_factor_s": 0.4,
        "xblock_preconditioner_xi": 1,
        "xblock_preconditioner_built": True,
        "xblock_assembled_host_fp": False,
        "xblock_jax_factors": False,
        "xblock_jax_factor_format": None,
        "xblock_jax_factor_apply": None,
        "xblock_lower_fill_mode": "off",
        "xblock_lower_fill_ignored_env": False,
        "xblock_use_active_dof": True,
        "xblock_linear_size": 7,
    }

    metadata = xblock_sparse_pc_result_diagnostics_from_solve_state(state, full_size=17)

    assert metadata["solver_kind"] == "xblock_sparse_pc"
    assert metadata["xblock_full_size"] == 17
    assert metadata["xblock_linear_size"] == 7
    assert metadata["xblock_device_krylov_method"] == "unit"


def test_sparse_pc_gmres_result_metadata_accepts_precomputed_sections_and_zero_target() -> None:
    state = {
        "history": (),
        "mv_count": np.int64(0),
        "sparse_pc_post_minres_steps": np.int64(0),
        "sparse_pc_post_minres_alphas": (),
        "sparse_pc_post_minres_alpha_clip": np.float64(2.0),
        "sparse_pc_post_minres_min_improvement": np.float64(0.1),
        "sparse_pc_post_minres_residual_before": None,
        "sparse_pc_post_minres_residual_after": None,
        "sparse_pc_post_minres_history": (),
        "sparse_pc_post_minres_error": "not_attempted",
        "sparse_pc_factor_dtype_used": "float64",
        "sparse_pc_factor_dtype_retry": None,
        "sparse_pc_factor_preflight_metadata": {
            "sparse_pc_factor_preflight_enabled": False,
        },
        "sparse_pc_direct_tail_metadata": {
            "sparse_pc_fortran_reduced_direct_tail_enabled": False,
        },
        "sparse_pc_pattern_metadata": {
            "sparse_pattern_nnz": 0,
            "sparse_pattern_scope": "precomputed",
        },
        "sparse_pc_static_metadata": {
            "solver_kind": "sparse_pc_gmres",
            "sparse_pc_backend": "global",
        },
        "setup_s": np.float64(0.1),
        "solve_s": np.float64(0.2),
        "sparse_pc_elapsed_s": np.float64(0.3),
        "pc_factor_s": np.float64(0.0),
        "factor_bundle_pc": None,
        "_operator_bundle_pc": None,
        "target": 0.0,
        "residual_norm_sparse_pc": 1.0,
        "sparse_pc_accepted_converged": False,
        "sparse_pc_factor_quality_rejected": True,
    }

    metadata = sparse_pc_gmres_result_metadata(state)

    assert metadata["solver_kind"] == "sparse_pc_gmres"
    assert metadata["iterations"] == 0
    assert metadata["matvecs"] == 0
    assert metadata["elapsed_s"] == 0.3
    assert metadata["sparse_pc_factor_elapsed_s"] is None
    assert metadata["sparse_pc_operator_csr_nbytes_estimate"] is None
    assert math.isinf(metadata["sparse_pc_residual_ratio_to_target"])
    assert metadata["sparse_pc_factor_quality_rejected"] is True
    assert metadata["sparse_pattern_scope"] == "precomputed"
    assert metadata["sparse_pc_factor_preflight_enabled"] is False
    assert metadata["sparse_pc_fortran_reduced_direct_tail_enabled"] is False


@pytest.mark.parametrize(
    ("bad_key", "message"),
    [
        ("moment_schur_metadata", "moment_schur_metadata must be a mapping"),
        ("moment_schur_stats", "moment_schur_stats must be a mapping"),
        ("global_coupling_metadata", "global_coupling_metadata must be a mapping"),
        ("global_coupling_stats", "global_coupling_stats must be a mapping"),
    ],
)
def test_fortran_reduced_xblock_result_metadata_rejects_non_mapping_sections(
    bad_key: str,
    message: str,
) -> None:
    state = {
        "moment_schur_metadata": {},
        "moment_schur_stats": {},
        "global_coupling_metadata": {},
        "global_coupling_stats": {},
    }
    state[bad_key] = object()

    with pytest.raises(TypeError, match=message):
        fortran_reduced_xblock_result_metadata(state)


def test_sparse_pc_factor_preflight_result_metadata_context_matches_state() -> None:
    state = {
        "factor_preflight_enabled": True,
        "factor_preflight_required": False,
        "factor_preflight_seed_enabled": True,
        "factor_preflight_seed_used": False,
        "factor_preflight_passed": True,
        "factor_preflight_error": None,
        "factor_preflight_residual_before": 4.0,
        "factor_preflight_residual_after": 2.0,
        "factor_preflight_improvement_ratio": 2.0,
        "factor_preflight_target_ratio": 5.0,
        "factor_preflight_max_target_ratio": np.float64(10.0),
        "factor_preflight_residual_diagnostics": {"tail": 0.2},
    }

    metadata = sparse_pc_factor_preflight_result_metadata(state)
    context_metadata = sparse_pc_factor_preflight_result_metadata_from_context(
        SparsePCFactorPreflightMetadataContext(
            enabled=state["factor_preflight_enabled"],
            required=state["factor_preflight_required"],
            seed_enabled=state["factor_preflight_seed_enabled"],
            seed_used=state["factor_preflight_seed_used"],
            passed=state["factor_preflight_passed"],
            error=state["factor_preflight_error"],
            residual_before=state["factor_preflight_residual_before"],
            residual_after=state["factor_preflight_residual_after"],
            improvement_ratio=state["factor_preflight_improvement_ratio"],
            target_ratio=state["factor_preflight_target_ratio"],
            max_target_ratio=state["factor_preflight_max_target_ratio"],
            residual_diagnostics=state["factor_preflight_residual_diagnostics"],
        )
    )

    assert context_metadata == metadata
    assert metadata["sparse_pc_factor_preflight_enabled"] is True
    assert metadata["sparse_pc_factor_preflight_max_target_ratio"] == 10.0
    assert metadata["sparse_pc_factor_preflight_residual_diagnostics"] == {
        "tail": 0.2
    }


def test_sparse_pc_pattern_result_metadata_context_matches_state() -> None:
    state = {
        "summary": SimpleNamespace(
            nnz=np.int64(30),
            avg_row_nnz=np.float64(3.0),
            max_row_nnz=np.int64(5),
        ),
        "sparse_pattern_scope": "fortran_reduced_active_dof",
        "pattern_build_s": np.float64(0.125),
    }

    metadata = sparse_pc_pattern_result_metadata(state)
    context_metadata = sparse_pc_pattern_result_metadata_from_context(
        SparsePCPatternMetadataContext(
            summary=state["summary"],
            scope=state["sparse_pattern_scope"],
            build_s=state["pattern_build_s"],
        )
    )

    assert context_metadata == metadata
    assert metadata["sparse_pattern_nnz"] == 30
    assert metadata["sparse_pattern_avg_row_nnz"] == 3.0
    assert metadata["sparse_pattern_max_row_nnz"] == 5
    assert metadata["sparse_pattern_scope"] == "fortran_reduced_active_dof"
    assert metadata["sparse_pattern_build_s"] == 0.125


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
