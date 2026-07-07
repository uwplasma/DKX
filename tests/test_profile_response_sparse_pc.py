from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp
from scipy import sparse as scipy_sparse

import sfincs_jax.problems.profile_sparse_solve as sparse_pc_module
import sfincs_jax.problems.profile_sparse_direct as sparse_direct_module
import sfincs_jax.problems.profile_sparse_finalization as sparse_finalization_module
import sfincs_jax.problems.profile_sparse_fortran_reduced as sparse_fortran_reduced_module
import sfincs_jax.problems.profile_sparse_policy as sparse_policy_module
import sfincs_jax.problems.profile_sparse_xblock as sparse_xblock_module
from sfincs_jax.problems.profile_setup import (
    expand_reduced_with_map,
    reduce_full_with_indices,
)
from sfincs_jax.problems.profile_diagnostics import (
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    fortran_reduced_xblock_result_metadata,
)
from sfincs_jax.solver import GMRESSolveResult
from sfincs_jax.problems.profile_sparse_direct import (
    DirectTailMaterializationContext,
    DirectTailMaterializationResult,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredAdmissionResult,
    DirectTailStructuredBuildContext,
    DirectTailSupportModePreflightContext,
    DirectTailResidualRescuePolicy,
    DirectTailTrueActiveRescuePolicy,
    DirectTailCoupledCoarseRescuePolicy,
    SparseHostDirectPayload,
    SparseHostDirectFactorSolvePayload,
    SparseHostDirectPolishPayload,
    SparseHostDirectFallbackPayload,
    ExplicitSparseMinimumNormBranchContext,
    ExplicitSparseHostDirectBranchContext,
    SparseHostOrILUFactorBuildContext,
    SparseHostOrILUFactorBuildResult,
    SparseHostOrILUFactorControls,
    SparseILUPreconditionerBuildContext,
    SparseHostScipyPreconditionerBuildContext,
    SparseHostScipyGMRESContext,
    SparseHostRetryCandidateContext,
    SparseHostRetryCandidateResult,
    SparseJAXRetryPreconditionerBuildContext,
    ExplicitSparseOperatorBuildPolicy,
    ExplicitSparseOperatorBuildResult,
    SparsePCDirectTailFinalMetadataContext,
    SparseMinimumNormPayload,
    SparseMinimumNormPolicy,
    build_explicit_sparse_operator_from_pattern,
    build_direct_tail_materialization_setup,
    build_direct_tail_structured_preconditioner_setup,
    explicit_sparse_pattern_progress_messages,
    resolve_explicit_sparse_operator_build_policy,
    resolve_sparse_minimum_norm_policy,
    resolve_sparse_host_or_ilu_factor_controls,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    run_direct_tail_support_mode_preflight,
    sparse_pc_direct_tail_final_metadata,
    sparse_host_direct_solve_payload,
    sparse_host_direct_solve_from_pattern,
    solve_explicit_sparse_host_direct_branch,
    solve_sparse_host_direct_from_available_factor,
    apply_sparse_host_direct_polish_if_needed,
    sparse_host_direct_fallback_payload,
    build_sparse_host_or_ilu_factor,
    build_sparse_ilu_preconditioner_from_cache,
    build_sparse_host_scipy_preconditioner,
    run_sparse_host_scipy_gmres,
    run_sparse_host_retry_candidate,
    build_sparse_jax_retry_preconditioner,
    sparse_minimum_norm_solve_payload,
    sparse_minimum_norm_solve_from_pattern,
    sparse_minimum_norm_start_message,
    solve_explicit_sparse_minimum_norm_branch,
    validate_explicit_sparse_host_request,
)
from sfincs_jax.problems.profile_sparse_finalization import (
    SparsePCFactorDtypeRetryContext,
    SparsePCFactorDtypeRetryResult,
    SparsePCGMRESContext,
    SparsePCGMRESCompletionMessageContext,
    SparsePCGMRESFinalizationBundleContext,
    SparsePCGMRESFinalPayload,
    SparsePCFactorDtypeRetryFinalizationContext,
    SparsePCPostMinresFinalizationContext,
    SparsePCPostMinresContext,
    SparsePCPostMinresUpdateContext,
    SparsePCGMRESFinalizationContext,
    SparsePCGMRESFinalizationStateContext,
    SparsePCGMRESFinalResultContext,
    SparsePCGMRESResult,
    apply_sparse_pc_post_minres,
    apply_sparse_pc_post_minres_if_needed,
    apply_sparse_pc_post_minres_from_solve_state,
    emit_sparse_pc_gmres_completion_from_solve_state,
    evaluate_sparse_pc_factor_dtype_retry,
    finalize_sparse_pc_gmres_from_solve_state,
    finalize_sparse_pc_gmres_bundle,
    finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state,
    finalize_sparse_pc_gmres_with_dtype_retry,
    sparse_pc_gmres_finalization_solve_scope_keys,
    sparse_pc_gmres_finalization_solve_state_keys,
    sparse_pc_gmres_finalization_bundle_from_solve_result,
    sparse_pc_gmres_finalization_bundle_from_solve_scope,
    sparse_pc_gmres_finalization_state_from_context,
    sparse_pc_gmres_finalization_state_from_solve_scope,
    sparse_pc_factor_dtype_retry_initial_guess,
    retry_sparse_pc_factor_dtype_from_finalization_context,
    run_sparse_pc_gmres_once,
    run_sparse_pc_gmres_once_for_retry,
    retry_sparse_pc_factor_dtype_from_solve_state,
    retry_sparse_pc_factor_dtype_if_needed,
    sparse_pc_gmres_completion_message,
    sparse_pc_gmres_final_payload_from_solve_state,
)
from sfincs_jax.problems.profile_sparse_fortran_reduced import (
    FortranReducedXBlockFactorBuildContext,
    FortranReducedXBlockFinalPayloadContext,
    FortranReducedXBlockGlobalCouplingStageContext,
    FortranReducedXBlockKrylovSetupContext,
    FortranReducedXBlockKrylovSolveContext,
    FortranReducedXBlockMomentSchurStageContext,
    apply_fortran_reduced_xblock_global_coupling_stage,
    apply_fortran_reduced_xblock_initial_seed,
    apply_fortran_reduced_xblock_moment_schur_stage,
    build_fortran_reduced_xblock_factor_stage,
    build_fortran_reduced_xblock_krylov_setup,
    prepare_fortran_reduced_xblock_initial_guess,
    resolve_fortran_reduced_sparse_pc_backend,
    resolve_fortran_reduced_xblock_factor_policy,
    resolve_fortran_reduced_xblock_global_coupling_policy,
    resolve_fortran_reduced_xblock_initial_seed_policy,
    resolve_fortran_reduced_xblock_krylov_policy,
    resolve_fortran_reduced_xblock_moment_schur_policy,
    fortran_reduced_xblock_final_payload_from_solve_state,
    fortran_reduced_xblock_final_payload,
    run_fortran_reduced_xblock_krylov_solve,
)
from sfincs_jax.problems.profile_sparse_policy import (
    SparsePCFactorPreflightPolicyContext,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCResidualCandidateAcceptanceContext,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCAutoPreflightRetryEvaluationContext,
    SparsePCGMRESControlPolicy,
    SparsePCMemoryBudgetPreflightContext,
    SparsePCPatternSetupContext,
    build_sparse_pc_active_dof_setup,
    build_sparse_pc_pattern_setup,
    enforce_sparse_pc_memory_budget,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    select_sparse_pc_auto_preflight_retry_candidates,
    evaluate_sparse_pc_auto_preflight_retry,
    resolve_sparse_pc_gmres_control_policy,
    resolve_sparse_pc_entry_policy,
    resolve_sparse_pc_factor_policy,
    resolve_sparse_pc_factor_preflight_policy,
)
from sfincs_jax.problems.profile_sparse_xblock import (
    FPXBlockGlobalCorrectionContext,
    FPXBlockHighXCorrectionContext,
    MatvecCounter,
    XBlockSubspaceCorrectionContext,
    SparseSXBlockRescueContext,
    SparseXBlockExplicitSeedContext,
    SparseXBlockRescueAcceptanceContext,
    SparseXBlockRescueBuildContext,
    SparseXBlockRescueSolveContext,
    XBlockFirstKrylovAttemptContext,
    XBlockFirstKrylovAttemptResult,
    XBlockSideProbeStageContext,
    XBlockSideProbeStageResult,
    XBlockGMRESFallbackDecision,
    XBlockGMRESFallbackContext,
    XBlockGMRESFallbackResult,
    XBlockGlobalCouplingStageContext,
    XBlockKrylovControlSetup,
    XBlockKrylovControlSetupContext,
    XBlockMomentSchurStageContext,
    XBlockPostKrylovCompletionContext,
    XBlockPostKrylovCompletionResult,
    XBlockPostSolveCorrectionContext,
    XBlockPostSolveCorrectionResult,
    XBlockPhysicalResidual,
    XBlockPreflightGateContext,
    XBlockProbeCoarseStageContext,
    XBlockProbeCoarseStageResult,
    XBlockSparsePCCompletionContext,
    XBlockSparsePCFinalCoreState,
    XBlockSparsePCFinalDeviceState,
    XBlockSparsePCFinalMetadataStateContext,
    XBlockSparsePCFinalNestedMetadata,
    XBlockSparsePCFinalPayloadContext,
    XBlockSparsePCFinalPreflightState,
    XBlockSparsePCWorkEstimates,
    XBlockTwoLevelStageContext,
    XBlockAssembledPreflightError,
    apply_xblock_global_coupling_stage,
    apply_xblock_moment_schur_stage,
    apply_xblock_probe_coarse_stage,
    apply_xblock_side_probe_stage,
    apply_xblock_two_level_stage,
    apply_sparse_xblock_explicit_seed,
    accept_sparse_xblock_rescue_candidate,
    apply_xblock_subspace_correction_if_needed,
    build_sparse_xblock_rescue_preconditioner,
    build_xblock_assembled_equilibration_setup,
    build_xblock_assembled_device_setup,
    build_xblock_assembled_matvec_setup,
    build_xblock_assembled_operator_if_requested,
    build_xblock_assembled_operator_preflight_setup,
    build_xblock_krylov_matvec_setup,
    build_xblock_local_preconditioner,
    emit_xblock_sparse_pc_completion_from_solve_state,
    emit_xblock_sparse_pc_completion,
    evaluate_xblock_moment_schur_probe_result,
    evaluate_xblock_preflight_gate,
    failed_xblock_global_coupling_metadata,
    failed_xblock_two_level_metadata,
    failed_xblock_moment_schur_metadata,
    finalize_xblock_global_coupling_metadata,
    finalize_xblock_two_level_metadata,
    finalize_xblock_moment_schur_metadata,
    prepare_xblock_initial_guess,
    resolve_xblock_krylov_control_setup,
    resolve_xblock_global_coupling_policy_setup,
    resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup,
    resolve_xblock_sparse_pc_setup,
    resolve_xblock_sparse_pc_branch_setup,
    resolve_xblock_sparse_pc_side_policy_setup,
    resolve_xblock_two_level_policy_setup,
    complete_xblock_post_krylov_stage,
    run_xblock_first_krylov_attempt,
    run_fp_xblock_global_correction_stage,
    run_fp_xblock_highx_residual_correction_stage,
    run_sparse_sxblock_rescue_stage,
    run_sparse_xblock_rescue_solve_stage,
    run_xblock_gmres_fallback_if_needed,
    run_xblock_post_solve_corrections,
    xblock_sparse_pc_final_metadata_solve_scope_keys,
    xblock_sparse_pc_final_metadata_solve_state_keys,
    xblock_sparse_pc_final_metadata_state_from_context,
    xblock_sparse_pc_final_metadata_state_from_solve_scope,
    xblock_sparse_pc_completion_message,
    xblock_gmres_fallback_decision,
    xblock_physical_solution_and_residual,
    xblock_sparse_pc_final_payload,
    xblock_sparse_pc_work_estimates,
    finalize_xblock_assembled_operator_metadata,
    xblock_sparse_pc_final_metadata_from_solve_state,
    xblock_sparse_pc_final_payload_from_solve_state,
)
from sfincs_jax.problems.profile_sparse_solve import (
    RHS1FullSparseRetryStageContext,
    run_rhs1_full_sparse_retry_stage,
)


def test_sparse_xblock_module_exposes_canonical_public_contract() -> None:
    """The split sparse x-block module owns its public sparse-PC helpers."""

    moved_names = (
        "MatvecCounter",
        "SparseXBlockRescueBuildContext",
        "SparseXBlockExplicitSeedContext",
        "SparseXBlockRescueSolveContext",
        "SparseXBlockRescueAcceptanceContext",
        "SparseSXBlockRescueContext",
        "FPXBlockGlobalCorrectionContext",
        "FPXBlockHighXCorrectionContext",
        "XBlockGlobalCouplingPolicySetup",
        "XBlockInitialGuessSetup",
        "XBlockKrylovMatvecSetup",
        "XBlockMomentSchurPolicySetup",
        "XBlockSparsePCSetup",
        "XBlockSparsePCSidePolicySetup",
        "XBlockSparsePCBranchSetup",
        "XBlockLocalPreconditionerBuildResult",
        "XBlockAssembledEquilibrationSetup",
        "XBlockAssembledPreflightMemoryError",
        "XBlockAssembledPreflightError",
        "XBlockAssembledOperatorPreflightSetup",
        "XBlockAssembledDeviceSetup",
        "XBlockAssembledMatvecSetup",
        "XBlockAssembledOperatorBuildResult",
        "XBlockMomentSchurProbeResult",
        "XBlockTwoLevelPolicySetup",
        "XBlockMomentSchurStageContext",
        "XBlockMomentSchurStageResult",
        "XBlockTwoLevelStageContext",
        "XBlockTwoLevelStageResult",
        "XBlockGlobalCouplingStageContext",
        "XBlockGlobalCouplingStageResult",
        "XBlockSeedPolicySetup",
        "XBlockSparsePCBranchContext",
        "XBlockKrylovReport",
        "XBlockSparsePCCompletionContext",
        "XBlockSparsePCFinalPayloadContext",
        "XBlockGMRESFallbackDecision",
        "XBlockGMRESFallbackContext",
        "XBlockGMRESFallbackResult",
        "XBlockDeviceKrylovState",
        "XBlockFirstKrylovAttemptContext",
        "XBlockFirstKrylovAttemptResult",
        "XBlockSideProbeStageContext",
        "XBlockSideProbeStageResult",
        "XBlockProbeCoarseStageContext",
        "XBlockProbeCoarseStageResult",
        "XBlockPreflightGateContext",
        "XBlockPreflightGateResult",
        "XBlockKrylovControlSetupContext",
        "XBlockKrylovControlSetup",
        "XBlockKrylovProgressCallbacksContext",
        "XBlockKrylovProgressCallbacks",
        "XBlockKrylovSolveState",
        "XBlockFirstKrylovSolveStateContext",
        "XBlockKrylovSolveStageContext",
        "XBlockKrylovSolveStageResult",
        "XBlockKrylovSolveSpaceContext",
        "XBlockKrylovSolveSpace",
        "XBlockSparsePCWorkEstimates",
        "XBlockPhysicalResidual",
        "build_sparse_xblock_rescue_preconditioner",
        "build_xblock_krylov_matvec_setup",
        "apply_sparse_xblock_explicit_seed",
        "prepare_xblock_initial_guess",
        "run_sparse_xblock_rescue_solve_stage",
        "accept_sparse_xblock_rescue_candidate",
        "run_sparse_sxblock_rescue_stage",
        "run_fp_xblock_global_correction_stage",
        "run_fp_xblock_highx_residual_correction_stage",
        "xblock_krylov_report",
        "apply_xblock_side_probe_stage",
        "apply_xblock_probe_coarse_stage",
        "evaluate_xblock_preflight_gate",
        "resolve_xblock_krylov_control_setup",
        "xblock_krylov_state_from_first_attempt",
        "xblock_krylov_state_from_gmres_fallback",
        "run_xblock_krylov_solve_stage",
        "xblock_device_cycle_progress_message",
        "xblock_host_krylov_progress_message",
        "build_xblock_krylov_progress_callbacks",
        "xblock_device_krylov_state",
        "prepare_xblock_krylov_solve_space",
        "run_xblock_first_krylov_attempt",
        "xblock_gmres_fallback_decision",
        "run_xblock_gmres_fallback_if_needed",
        "xblock_sparse_pc_work_estimates",
        "xblock_sparse_pc_completion_message",
        "emit_xblock_sparse_pc_completion",
        "emit_xblock_sparse_pc_completion_from_solve_state",
        "xblock_physical_solution_and_residual",
        "XBlockSparsePCFinalCoreState",
        "XBlockSparsePCFinalDeviceState",
        "XBlockSparsePCFinalPreflightState",
        "XBlockSparsePCFinalNestedMetadata",
        "XBlockSparsePCFinalMetadataStateContext",
        "xblock_sparse_pc_final_metadata_solve_state_keys",
        "xblock_sparse_pc_final_metadata_solve_scope_keys",
        "xblock_sparse_pc_final_metadata_state_from_context",
        "xblock_sparse_pc_final_metadata_state_from_solve_scope",
        "xblock_sparse_pc_final_metadata_from_solve_state",
        "xblock_sparse_pc_final_payload_from_solve_state",
        "xblock_sparse_pc_final_payload",
        "XBlockSubspaceCorrectionContext",
        "XBlockSubspaceCorrectionResult",
        "XBlockPostSolveCorrectionContext",
        "XBlockPostSolveCorrectionResult",
        "XBlockPostKrylovCompletionContext",
        "XBlockPostKrylovCompletionResult",
        "apply_xblock_subspace_correction_if_needed",
        "run_xblock_post_solve_corrections",
        "complete_xblock_post_krylov_stage",
        "run_xblock_sparse_pc_branch",
        "resolve_xblock_sparse_pc_setup",
        "resolve_xblock_sparse_pc_side_policy_setup",
        "resolve_xblock_sparse_pc_branch_setup",
        "build_xblock_local_preconditioner",
        "build_xblock_assembled_equilibration_setup",
        "build_xblock_assembled_operator_preflight_setup",
        "build_xblock_assembled_device_setup",
        "build_xblock_assembled_matvec_setup",
        "build_xblock_assembled_operator_if_requested",
        "finalize_xblock_assembled_operator_metadata",
        "resolve_xblock_moment_schur_policy_setup",
        "evaluate_xblock_moment_schur_probe_result",
        "finalize_xblock_moment_schur_metadata",
        "failed_xblock_moment_schur_metadata",
        "resolve_xblock_two_level_policy_setup",
        "finalize_xblock_two_level_metadata",
        "failed_xblock_two_level_metadata",
        "resolve_xblock_global_coupling_policy_setup",
        "finalize_xblock_global_coupling_metadata",
        "failed_xblock_global_coupling_metadata",
        "apply_xblock_moment_schur_stage",
        "apply_xblock_two_level_stage",
        "apply_xblock_global_coupling_stage",
        "resolve_xblock_seed_policy_setup",
    )
    for name in moved_names:
        assert hasattr(sparse_xblock_module, name)
        assert name in sparse_xblock_module.__all__


def test_sparse_xblock_result_containers_preserve_orchestration_contract() -> None:
    """Sparse x-block stages exchange small result dataclasses, not solve scope."""

    result = GMRESSolveResult(x=jnp.asarray([1.0]), residual_norm=jnp.asarray(0.25))
    residual_vec = jnp.asarray([0.5])

    build = sparse_xblock_module.SparseXBlockRescueBuildResult(
        preconditioner=lambda value: value,
        preconditioner_xi=2,
        force_assembled_host_fp=True,
    )
    explicit_seed = sparse_xblock_module.SparseXBlockExplicitSeedResult(
        result=result,
        seed_residual=0.2,
        seed_improvement_ratio=0.5,
        seed_accept_ratio=0.25,
        refine_steps=3,
        refines_performed=2,
        reason="accepted",
    )
    solve = sparse_xblock_module.SparseXBlockRescueSolveResult(
        result=result,
        reason="candidate",
        candidate_residual=0.25,
        seed_residual=0.2,
        seed_improvement_ratio=0.5,
        seed_accept_ratio=0.25,
        seed_refine_steps=3,
        seed_refines_performed=2,
    )
    acceptance = sparse_xblock_module.SparseXBlockRescueAcceptanceResult(
        result=result,
        accepted=True,
        reason="accepted",
        candidate_residual=0.25,
        explicit_seed_used=True,
    )
    sxblock = sparse_xblock_module.SparseSXBlockRescueResult(
        result=result,
        accepted=True,
        polished=False,
        error=None,
        seed_residual=0.3,
        polish_residual=None,
        polish_restart=None,
        polish_maxiter=None,
    )
    global_correction = sparse_xblock_module.FPXBlockGlobalCorrectionResult(
        result=result,
        residual_vec=residual_vec,
        accepted=True,
        reason="accepted",
        error=None,
        preconditioner_label="unit",
        steps=2,
        accepted_steps=1,
        residual_before=1.0,
        residual_after=0.25,
        improvement_ratio=0.25,
        elapsed_s=0.1,
    )
    highx = sparse_xblock_module.FPXBlockHighXCorrectionResult(
        result=result,
        residual_vec=residual_vec,
        accepted=True,
        reason="accepted",
        error=None,
        residual_before=1.0,
        residual_after=0.5,
        improvement_ratio=0.5,
        elapsed_s=0.2,
        direction_count=2,
        direction_names=("x0", "x1"),
    )

    assert build.preconditioner_xi == 2
    assert build.force_assembled_host_fp
    assert explicit_seed.refines_performed == 2
    assert solve.seed_accept_ratio == pytest.approx(0.25)
    assert acceptance.explicit_seed_used
    assert sxblock.seed_residual == pytest.approx(0.3)
    assert global_correction.accepted_steps == 1
    assert highx.direction_names == ("x0", "x1")


def test_run_xblock_sparse_pc_branch_disabled_context_returns_none() -> None:
    values = {
        dataclass_field.name: None
        for dataclass_field in fields(sparse_xblock_module.XBlockSparsePCBranchContext)
    }
    values["xblock_sparse_pc"] = False

    result = sparse_xblock_module.run_xblock_sparse_pc_branch(
        sparse_xblock_module.XBlockSparsePCBranchContext(**values)
    )

    assert result is None


class _DefaultSparseSolveValues(dict):
    def __getitem__(self, key):
        return self.get(key, None)


def _base_residual_correction_values() -> dict[str, object]:
    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu", factor_s=0.0)
    residual = jnp.asarray([1.0], dtype=jnp.float64)
    values = {
        dataclass_field.name: None
        for dataclass_field in fields(sparse_pc_module.SparsePCResidualCorrectionStageContext)
    }
    values.update(
        factor_bundle_pc=factor,
        operator_bundle_pc=SimpleNamespace(kind="operator"),
        structured_pc_ready=True,
        pc_factor_s=1.0,
        setup_s=2.0,
        sparse_pc_rhs=residual,
        sparse_pc_linear_size=1,
        target=1.0,
        factor_preflight_residual_before=4.0,
        factor_preflight_residual_after=2.0,
        factor_preflight_residual_diagnostics=None,
        factor_preflight_improvement_ratio=2.0,
        factor_preflight_target_ratio=2.0,
        factor_preflight_passed=False,
        factor_preflight_seed_enabled=True,
        factor_preflight_seed_used=False,
        factor_preflight_max_target_ratio=1.0,
        residual_vec_current=jnp.asarray([2.0], dtype=jnp.float64),
        x0_sparse=None,
        matvec=lambda vec: vec,
        matvec_no_count=lambda vec: vec,
        matmat=lambda mat: mat,
        diagnostics=lambda **_kwargs: {"selected": True},
        layout=None,
        active_indices=None,
        elapsed_s=lambda: 3.0,
        emit=None,
        additive_rescue_nbytes=lambda _bundle, _mb: 1024,
        true_action_column_cache_factory=lambda **_kwargs: None,
        true_active_submatrix_builder=lambda **_kwargs: None,
        true_active_block_builder=lambda **_kwargs: None,
        true_active_residual_block_builder=lambda **_kwargs: None,
        true_window_builder=lambda **_kwargs: None,
        residual_coarse_builder=lambda **_kwargs: None,
        residual_window_builder=lambda **_kwargs: None,
        continue_after_base_improvement=False,
        true_coupled_base_improvement_override_used=False,
        true_active_submatrix_requested=False,
        true_active_block_requested=False,
        true_active_residual_block_requested=False,
        true_window_requested=False,
        residual_coarse_requested=False,
        residual_window_requested=False,
        true_active_column_cache_requested=False,
        true_active_column_cache_max_mb=0.0,
        true_active_block_x_count=1,
        true_active_block_ell_count=1,
        true_active_block_species_count=None,
        true_active_block_theta_stride=1,
        true_active_block_zeta_stride=1,
        true_active_block_max_mb=1.0,
        true_active_block_regularization=0.0,
        true_active_block_max_size=1,
        true_active_block_column_batch=1,
        true_active_block_drop_tol=0.0,
        true_active_block_include_tail=False,
        true_active_block_max_tail=0,
        true_active_block_damping=False,
        true_active_block_beta_max=1.0,
        true_active_submatrix_damping=False,
        true_active_submatrix_alpha_clip=1.0,
        true_active_submatrix_min_improvement=0.0,
        true_active_residual_block_max_mb=1.0,
        true_active_residual_block_regularization=0.0,
        true_active_residual_block_max_size=1,
        true_active_residual_block_column_batch=1,
        true_active_residual_block_drop_tol=0.0,
        true_active_residual_block_include_tail=False,
        true_active_residual_block_max_tail=0,
        true_active_residual_block_kinetic_only=True,
        true_active_residual_block_damping=False,
        true_active_residual_block_beta_max=1.0,
        true_active_residual_block_min_improvement=0.0,
        true_active_residual_block_accept_base_improvement=False,
        true_window_max_windows=1,
        true_window_x_radius=0,
        true_window_ell_radius=0,
        true_window_max_mb=1.0,
        true_window_regularization=0.0,
        true_window_max_size=1,
        true_window_column_batch=1,
        true_window_drop_tol=0.0,
        true_window_include_tail=False,
        true_window_specs=(),
        true_window_damping=False,
        true_window_beta_max=1.0,
        residual_coarse_rank=1,
        residual_coarse_max_mb=1.0,
        residual_coarse_regularization=0.0,
        residual_window_max_windows=1,
        residual_window_x_radius=0,
        residual_window_ell_radius=0,
        residual_window_max_mb=1.0,
        residual_window_regularization=0.0,
        residual_window_coefficient_mode="identity",
        residual_window_combine_mode="additive",
        residual_window_interface_depth=0,
        residual_window_max_size=1,
    )
    return values


def test_requested_sparse_pc_gmres_branch_returns_none_when_method_not_requested() -> None:
    values = _DefaultSparseSolveValues(
        {
            "_SPARSE_HOST_PC_GMRES_SOLVE_METHODS": frozenset({"sparse_pc_gmres"}),
            "_SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS": frozenset({"xblock_sparse_pc_gmres"}),
            "solve_method_kind_explicit": "gmres",
        }
    )

    result = sparse_pc_module.try_run_requested_sparse_pc_gmres_branch(
        sparse_pc_module.RequestedSparsePCGMRESBranchContext(values=values)
    )

    assert result is None


def test_sparse_pc_factor_preflight_reports_real_residual_and_seed() -> None:
    emits: list[str] = []
    rhs = jnp.asarray([2.0, 0.0], dtype=jnp.float64)

    result = sparse_pc_module.run_sparse_pc_factor_preflight(
        sparse_pc_module.SparsePCFactorPreflightRunContext(
            rhs=rhs,
            rhs_norm=2.0,
            target=1.5,
            preconditioner=lambda vec: 0.5 * vec,
            matvec=lambda vec: vec,
            diagnostics=lambda **_kwargs: {"selected": False},
            layout=None,
            active_indices=None,
            seed_enabled=True,
            max_target_ratio=1.0,
            emit=lambda _level, message: emits.append(message),
        )
    )

    assert result.residual_before == 2.0
    assert result.residual_after == pytest.approx(1.0)
    assert result.improvement_ratio == pytest.approx(2.0)
    assert result.target_ratio == pytest.approx(2.0 / 3.0)
    assert result.passed
    assert result.seed_used
    assert result.x0_seed is not None
    assert any("factor preflight" in message for message in emits)


def test_sparse_pc_factor_preflight_emits_residual_diagnostics() -> None:
    emits: list[str] = []

    result = sparse_pc_module.run_sparse_pc_factor_preflight(
        sparse_pc_module.SparsePCFactorPreflightRunContext(
            rhs=jnp.asarray([2.0, 0.0], dtype=jnp.float64),
            rhs_norm=2.0,
            target=1.0,
            preconditioner=lambda vec: 0.5 * vec,
            matvec=lambda vec: vec,
            diagnostics=lambda **_kwargs: {
                "selected": True,
                "component_norms": {
                    "kinetic": {"energy_fraction": 0.75},
                    "extra": {"energy_fraction": 0.25},
                },
                "top_species_x": [{"label": "s0x0", "energy_fraction": 0.5}],
                "top_x": [{"label": "x0", "energy_fraction": 0.4}],
                "top_ell": [{"label": "ell1", "energy_fraction": 0.3}],
            },
            layout=None,
            active_indices=None,
            seed_enabled=False,
            max_target_ratio=10.0,
            emit=lambda _level, message: emits.append(message),
        )
    )

    assert result.residual_after == pytest.approx(1.0)
    assert not result.seed_used
    assert any("preflight residual diagnostics" in message for message in emits)
    assert any("top_species_x=s0x0" in message for message in emits)


def test_sparse_pc_auto_preflight_retry_accepts_better_candidate() -> None:
    class RetryPreconditioner:
        selected = True
        reason = "selected"
        operator = object()
        kind = "active_dense"
        setup_s = 0.25
        metadata = {"factor_nbytes_actual": 8}

        def to_dict(self) -> dict[str, object]:
            return {
                "kind": self.kind,
                "reason": self.reason,
                "setup_s": self.setup_s,
                "metadata": dict(self.metadata),
            }

    residual = jnp.asarray([1.0], dtype=jnp.float64)
    original_factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu")
    structured_metadata: dict[str, object] = {
        "metadata": {
            "auto_candidates": (
                "active_fortran_v3_reduced_lu",
                "active_dense",
            ),
            "auto_selected_kind": "active_fortran_v3_reduced_lu",
        }
    }
    emits: list[str] = []

    result = sparse_pc_module.run_sparse_pc_auto_preflight_retry_stage(
        sparse_pc_module.SparsePCAutoPreflightRetryStageContext(
            env={},
            structured_pc_ready=True,
            structured_pc_preflight_required=True,
            factor_preflight_passed=False,
            direct_tail_structured_pc_requested="auto",
            direct_tail_operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(1)),
            direct_tail_structured_layout=object(),
            direct_tail_structured_active_indices=np.asarray([0], dtype=np.int64),
            direct_tail_structured_max_nbytes=1024,
            direct_tail_structured_pc_selected=False,
            direct_tail_structured_pc_reason="failed_preflight",
            direct_tail_structured_pc_metadata=structured_metadata,
            operator_bundle_pc=None,
            factor_bundle_pc=original_factor,
            pc_factor_s=1.0,
            pc_reg=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            sparse_pc_rhs=residual,
            sparse_pc_linear_size=1,
            structured_pc_preflight_required_min_size=10,
            factor_preflight_max_target_ratio=1.0,
            factor_preflight_residual_before=2.0,
            factor_preflight_residual_after=1.0,
            factor_preflight_residual_diagnostics=None,
            factor_preflight_improvement_ratio=2.0,
            factor_preflight_target_ratio=1.0,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=False,
            residual_vec_current=residual,
            target=1.0,
            matvec_no_count=lambda vec: vec,
            diagnostics=lambda **_kwargs: {"selected": True},
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=lambda _level, message: emits.append(message),
            structured_preconditioner_builder=lambda **_kwargs: RetryPreconditioner(),
            factor_bundle_factory=lambda **kwargs: SimpleNamespace(
                kind=kwargs["kind"],
                factor_s=kwargs["factor_s"],
                solve=lambda rhs: rhs,
            ),
        )
    )

    assert result.selected
    assert result.factor_bundle_pc is not original_factor
    assert result.factor_preflight_residual_after == pytest.approx(0.0)
    assert result.factor_preflight_passed
    assert result.factor_preflight_seed_used
    assert result.x0_sparse is not None
    assert result.attempts[0]["kind"] == "active_dense"
    assert any("auto preflight retry accepted" in message for message in emits)


def test_sparse_pc_auto_preflight_retry_noop_preserves_state() -> None:
    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu")
    residual = jnp.asarray([1.0], dtype=jnp.float64)

    result = sparse_pc_module.run_sparse_pc_auto_preflight_retry_stage(
        sparse_pc_module.SparsePCAutoPreflightRetryStageContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_RETRY": "0"},
            structured_pc_ready=True,
            structured_pc_preflight_required=True,
            factor_preflight_passed=False,
            direct_tail_structured_pc_requested="auto",
            direct_tail_operator_bundle=SimpleNamespace(matrix=None),
            direct_tail_structured_layout=object(),
            direct_tail_structured_active_indices=np.asarray([0], dtype=np.int64),
            direct_tail_structured_max_nbytes=1024,
            direct_tail_structured_pc_selected=False,
            direct_tail_structured_pc_reason=None,
            direct_tail_structured_pc_metadata={},
            operator_bundle_pc=None,
            factor_bundle_pc=factor,
            pc_factor_s=1.0,
            pc_reg=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            sparse_pc_rhs=residual,
            sparse_pc_linear_size=1,
            structured_pc_preflight_required_min_size=1,
            factor_preflight_max_target_ratio=10.0,
            factor_preflight_residual_before=2.0,
            factor_preflight_residual_after=1.0,
            factor_preflight_residual_diagnostics=None,
            factor_preflight_improvement_ratio=2.0,
            factor_preflight_target_ratio=1.0,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=False,
            residual_vec_current=residual,
            target=1.0,
            matvec_no_count=lambda vec: vec,
            diagnostics=lambda **_kwargs: {},
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=None,
            structured_preconditioner_builder=lambda **_kwargs: None,
            factor_bundle_factory=lambda **_kwargs: None,
        )
    )

    assert not result.selected
    assert result.attempts == ()
    assert result.factor_bundle_pc is factor
    assert result.pc_factor_s == pytest.approx(1.0)
    assert result.setup_s == pytest.approx(3.0)
    assert result.residual_vec_current is residual


def test_sparse_pc_residual_candidate_update_accepts_improving_candidate() -> None:
    emits: list[str] = []
    old_factor = SimpleNamespace(kind="old", factor_s=0.0)
    new_factor = SimpleNamespace(kind="new", factor_s=0.25, metadata={"coarse_size": 2})
    candidate_x = jnp.asarray([1.0], dtype=jnp.float64)
    residual_vec = jnp.asarray([0.5], dtype=jnp.float64)
    diagnostics_calls: list[jnp.ndarray] = []

    result = sparse_pc_module.apply_sparse_pc_residual_candidate_update(
        sparse_pc_module.SparsePCResidualCandidateUpdateContext(
            label="candidate",
            metadata_count_key="coarse_size",
            metadata_count_label="coarse_size",
            bundle=new_factor,
            candidate_x=candidate_x,
            candidate_residual_vec=residual_vec,
            candidate_residual_after=0.5,
            candidate_metadata=dict(new_factor.metadata),
            factor_bundle_pc=old_factor,
            pc_factor_s=1.0,
            setup_s=2.0,
            factor_preflight_residual_before=4.0,
            factor_preflight_residual_after=2.0,
            factor_preflight_residual_diagnostics=None,
            factor_preflight_improvement_ratio=2.0,
            factor_preflight_target_ratio=2.0,
            factor_preflight_passed=False,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=False,
            target=1.0,
            max_target_ratio=1.0,
            residual_vec_current=jnp.asarray([2.0], dtype=jnp.float64),
            x0_sparse=None,
            diagnostics=lambda **kwargs: diagnostics_calls.append(kwargs["residual"]) or {"ok": True},
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=lambda _level, message: emits.append(message),
        )
    )

    assert result.accepted
    assert result.factor_bundle_pc is new_factor
    assert result.pc_factor_s == pytest.approx(1.25)
    assert result.setup_s == pytest.approx(3.0)
    assert result.factor_preflight_residual_after == pytest.approx(0.5)
    assert result.factor_preflight_target_ratio == pytest.approx(0.5)
    assert result.factor_preflight_passed
    assert result.factor_preflight_seed_used
    assert result.x0_sparse is candidate_x
    assert diagnostics_calls == [residual_vec]
    assert result.metadata["base_residual_after"] == pytest.approx(2.0)
    assert any("candidate accepted coarse_size=2" in message for message in emits)


def test_sparse_pc_residual_candidate_update_rejects_regression() -> None:
    emits: list[str] = []
    old_factor = SimpleNamespace(kind="old", factor_s=0.0)
    new_factor = SimpleNamespace(kind="new", factor_s=0.25, metadata={"coarse_size": 2})
    current_residual = jnp.asarray([1.0], dtype=jnp.float64)

    result = sparse_pc_module.apply_sparse_pc_residual_candidate_update(
        sparse_pc_module.SparsePCResidualCandidateUpdateContext(
            label="candidate",
            metadata_count_key="coarse_size",
            metadata_count_label="coarse_size",
            bundle=new_factor,
            candidate_x=jnp.asarray([2.0], dtype=jnp.float64),
            candidate_residual_vec=jnp.asarray([2.0], dtype=jnp.float64),
            candidate_residual_after=2.0,
            candidate_metadata=dict(new_factor.metadata),
            factor_bundle_pc=old_factor,
            pc_factor_s=1.0,
            setup_s=2.0,
            factor_preflight_residual_before=4.0,
            factor_preflight_residual_after=1.0,
            factor_preflight_residual_diagnostics={"base": True},
            factor_preflight_improvement_ratio=4.0,
            factor_preflight_target_ratio=1.0,
            factor_preflight_passed=True,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=True,
            target=1.0,
            max_target_ratio=1.0,
            residual_vec_current=current_residual,
            x0_sparse=jnp.asarray([1.0], dtype=jnp.float64),
            diagnostics=lambda **_kwargs: {"unused": True},
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=lambda _level, message: emits.append(message),
        )
    )

    assert not result.accepted
    assert result.factor_bundle_pc is old_factor
    assert result.factor_preflight_residual_after == pytest.approx(1.0)
    assert result.residual_vec_current is current_residual
    assert result.metadata["base_residual_after"] == pytest.approx(1.0)
    assert any("candidate rejected" in message for message in emits)


def test_sparse_pc_residual_correction_stage_accepts_true_active_submatrix() -> None:
    values = _base_residual_correction_values()
    bundle = SimpleNamespace(
        factor_s=0.5,
        metadata={"block_size": 1},
        solve=lambda rhs: rhs,
    )
    values.update(
        true_active_submatrix_requested=True,
        true_active_submatrix_builder=lambda **_kwargs: bundle,
    )

    result = sparse_pc_module.run_sparse_pc_residual_correction_stage(
        sparse_pc_module.SparsePCResidualCorrectionStageContext(**values)
    )

    assert result.true_active_submatrix_selected
    assert result.true_active_submatrix_residual_after == pytest.approx(0.0)
    assert result.factor_bundle_pc is bundle
    assert result.factor_preflight_passed
    assert result.factor_preflight_seed_used
    assert result.x0_sparse is not None


@pytest.mark.parametrize(
    (
        "request_field",
        "builder_field",
        "selected_field",
        "residual_field",
        "metadata_key",
    ),
    (
        (
            "true_active_block_requested",
            "true_active_block_builder",
            "true_active_block_selected",
            "true_active_block_residual_after",
            "block_size",
        ),
        (
            "true_active_residual_block_requested",
            "true_active_residual_block_builder",
            "true_active_residual_block_selected",
            "true_active_residual_block_residual_after",
            "block_size",
        ),
        (
            "true_window_requested",
            "true_window_builder",
            "true_window_selected",
            "true_window_residual_after",
            "window_size",
        ),
        (
            "residual_coarse_requested",
            "residual_coarse_builder",
            "residual_coarse_selected",
            "residual_coarse_residual_after",
            "rank",
        ),
        (
            "residual_window_requested",
            "residual_window_builder",
            "residual_window_selected",
            "residual_window_residual_after",
            "window_count",
        ),
    ),
)
def test_sparse_pc_residual_correction_stage_accepts_rescue_family(
    request_field: str,
    builder_field: str,
    selected_field: str,
    residual_field: str,
    metadata_key: str,
) -> None:
    values = _base_residual_correction_values()
    bundle = SimpleNamespace(
        factor_s=0.5,
        metadata={metadata_key: 1},
        solve=lambda rhs: rhs,
    )
    values.update(
        {
            request_field: True,
            builder_field: lambda **_kwargs: bundle,
        }
    )

    result = sparse_pc_module.run_sparse_pc_residual_correction_stage(
        sparse_pc_module.SparsePCResidualCorrectionStageContext(**values)
    )

    assert getattr(result, selected_field)
    assert getattr(result, residual_field) == pytest.approx(0.0)
    assert result.factor_bundle_pc is bundle
    assert result.factor_preflight_passed


def test_sparse_pc_residual_correction_stage_reports_true_column_cache() -> None:
    class Cache:
        def matvec(self, vec):
            return np.asarray(vec, dtype=np.float64)

        def matmat(self, mat):
            return np.asarray(mat, dtype=np.float64)

        def metadata(self) -> dict[str, int]:
            return {
                "hits": 2,
                "misses": 1,
                "stored_columns": 1,
                "stored_nbytes": 8,
            }

    def window_builder(**kwargs):
        np.testing.assert_allclose(kwargs["true_matvec"](np.asarray([1.0])), [1.0])
        np.testing.assert_allclose(kwargs["true_matmat"](np.eye(1)), np.eye(1))
        return SimpleNamespace(
            factor_s=0.5,
            metadata={"window_size": 1},
            solve=lambda rhs: rhs,
        )

    emits: list[str] = []
    values = _base_residual_correction_values()
    values.update(
        true_window_requested=True,
        true_active_column_cache_requested=True,
        true_action_column_cache_factory=lambda **_kwargs: Cache(),
        true_window_builder=window_builder,
        emit=lambda _level, message: emits.append(message),
    )

    result = sparse_pc_module.run_sparse_pc_residual_correction_stage(
        sparse_pc_module.SparsePCResidualCorrectionStageContext(**values)
    )

    assert result.true_window_selected
    assert result.true_active_column_cache_metadata == {
        "hits": 2,
        "misses": 1,
        "stored_columns": 1,
        "stored_nbytes": 8,
    }
    assert any("true active column cache" in message for message in emits)


def test_sparse_pc_residual_correction_stage_records_builder_errors() -> None:
    def raising_builder(**_kwargs):
        raise RuntimeError("boom")

    values = _base_residual_correction_values()
    values.update(
        true_active_block_requested=True,
        true_active_block_builder=raising_builder,
        emit=lambda _level, _message: None,
    )

    result = sparse_pc_module.run_sparse_pc_residual_correction_stage(
        sparse_pc_module.SparsePCResidualCorrectionStageContext(**values)
    )

    assert not result.true_active_block_selected
    assert result.true_active_block_error == "RuntimeError: boom"


def test_sparse_pc_residual_correction_stage_noop_preserves_state() -> None:
    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu")
    residual = jnp.asarray([1.0], dtype=jnp.float64)
    values = {
        dataclass_field.name: None
        for dataclass_field in fields(sparse_pc_module.SparsePCResidualCorrectionStageContext)
    }
    values.update(
        factor_bundle_pc=factor,
        operator_bundle_pc=None,
        structured_pc_ready=False,
        pc_factor_s=1.0,
        setup_s=2.0,
        sparse_pc_rhs=residual,
        sparse_pc_linear_size=1,
        target=1.0,
        factor_preflight_residual_before=2.0,
        factor_preflight_residual_after=1.0,
        factor_preflight_residual_diagnostics=None,
        factor_preflight_improvement_ratio=2.0,
        factor_preflight_target_ratio=1.0,
        factor_preflight_passed=True,
        factor_preflight_seed_enabled=True,
        factor_preflight_seed_used=False,
        factor_preflight_max_target_ratio=10.0,
        residual_vec_current=residual,
        x0_sparse=None,
        matvec=lambda vec: vec,
        matvec_no_count=lambda vec: vec,
        matmat=lambda mat: mat,
        diagnostics=lambda **_kwargs: {},
        layout=None,
        active_indices=None,
        elapsed_s=lambda: 3.0,
        emit=None,
        additive_rescue_nbytes=lambda _bundle, _mb: 0,
        true_action_column_cache_factory=lambda **_kwargs: None,
        true_active_submatrix_builder=lambda **_kwargs: None,
        true_active_block_builder=lambda **_kwargs: None,
        true_active_residual_block_builder=lambda **_kwargs: None,
        true_window_builder=lambda **_kwargs: None,
        residual_coarse_builder=lambda **_kwargs: None,
        residual_window_builder=lambda **_kwargs: None,
        continue_after_base_improvement=False,
        true_coupled_base_improvement_override_used=False,
        true_active_submatrix_requested=False,
        true_active_block_requested=False,
        true_active_residual_block_requested=False,
        true_window_requested=False,
        residual_coarse_requested=False,
        residual_window_requested=False,
        true_active_column_cache_requested=False,
        true_active_column_cache_max_mb=0.0,
    )

    result = sparse_pc_module.run_sparse_pc_residual_correction_stage(
        sparse_pc_module.SparsePCResidualCorrectionStageContext(**values)
    )

    assert result.factor_bundle_pc is factor
    assert result.pc_factor_s == pytest.approx(1.0)
    assert result.setup_s == pytest.approx(2.0)
    assert not result.true_active_submatrix_selected
    assert not result.true_active_block_selected
    assert not result.true_active_residual_block_selected
    assert not result.true_window_selected
    assert not result.residual_coarse_selected
    assert not result.residual_window_selected


def test_sparse_pc_true_coupled_coarse_stage_noop_preserves_state() -> None:
    factor = SimpleNamespace(kind="point", factor_s=0.0)
    residual = jnp.asarray([1.0], dtype=jnp.float64)

    result = sparse_pc_module.run_sparse_pc_true_coupled_coarse_stage(
        sparse_pc_module.SparsePCTrueCoupledCoarseStageContext(
            factor_bundle_pc=factor,
            pc_factor_s=1.0,
            setup_s=2.0,
            sparse_pc_rhs=residual,
            sparse_pc_linear_size=1,
            target=1.0,
            factor_preflight_residual_before=2.0,
            factor_preflight_residual_after=1.0,
            factor_preflight_residual_diagnostics=None,
            factor_preflight_improvement_ratio=2.0,
            factor_preflight_target_ratio=1.0,
            factor_preflight_passed=True,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=False,
            factor_preflight_max_target_ratio=10.0,
            residual_vec_current=residual,
            x0_sparse=None,
            matvec_no_count=lambda vec: vec,
            matmat=lambda mat: mat,
            diagnostics=lambda **_kwargs: {},
            op=SimpleNamespace(),
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=None,
            builder=lambda **_kwargs: None,
            additive_rescue_nbytes=lambda _bundle, _mb: 0,
            explicit_requested=False,
            auto_enabled=False,
            auto_native_enabled=False,
            auto_target_ratio=1.0,
            auto_min_size=1,
            max_windows=0,
            x_radius=0,
            ell_radius=0,
            max_mb=0.0,
            regularization=0.0,
            max_size=0,
            column_batch=1,
            drop_tol=0.0,
            low_lmax=0,
            profile_moment_count=0,
            angular_lmax=0,
            angular_mode_max=0,
            max_tail_units=0,
            include_tail=False,
            include_constraint_sources=False,
            include_fsavg=False,
            include_window_residual=False,
            include_profile_moments=False,
            include_angular_residual=False,
            include_angular_basis=False,
            include_preconditioned_loads=False,
            preconditioned_load_max_columns=0,
            preconditioned_load_max_nnz=0,
            preconditioned_load_drop_tol=0.0,
            damping=False,
            beta_max=1.0,
            accept_base_improvement=False,
        )
    )

    assert not result.requested
    assert not result.auto_selected
    assert not result.selected
    assert result.factor_bundle_pc is factor
    assert result.pc_factor_s == pytest.approx(1.0)
    assert result.residual_vec_current is residual


def test_sparse_pc_true_coupled_coarse_stage_accepts_improving_bundle() -> None:
    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu", factor_s=0.0)
    bundle = SimpleNamespace(
        factor_s=0.4,
        metadata={"coarse_size": 1},
        solve=lambda rhs: rhs,
    )
    emits: list[str] = []

    result = sparse_pc_module.run_sparse_pc_true_coupled_coarse_stage(
        sparse_pc_module.SparsePCTrueCoupledCoarseStageContext(
            factor_bundle_pc=factor,
            pc_factor_s=1.0,
            setup_s=2.0,
            sparse_pc_rhs=jnp.asarray([1.0], dtype=jnp.float64),
            sparse_pc_linear_size=1,
            target=1.0,
            factor_preflight_residual_before=4.0,
            factor_preflight_residual_after=2.0,
            factor_preflight_residual_diagnostics=None,
            factor_preflight_improvement_ratio=2.0,
            factor_preflight_target_ratio=2.0,
            factor_preflight_passed=False,
            factor_preflight_seed_enabled=True,
            factor_preflight_seed_used=False,
            factor_preflight_max_target_ratio=1.0,
            residual_vec_current=jnp.asarray([2.0], dtype=jnp.float64),
            x0_sparse=None,
            matvec_no_count=lambda vec: vec,
            matmat=lambda mat: mat,
            diagnostics=lambda **_kwargs: {"selected": True},
            op=SimpleNamespace(),
            layout=None,
            active_indices=None,
            elapsed_s=lambda: 3.0,
            emit=lambda _level, message: emits.append(message),
            builder=lambda **_kwargs: bundle,
            additive_rescue_nbytes=lambda _bundle, _mb: 1024,
            explicit_requested=True,
            auto_enabled=False,
            auto_native_enabled=False,
            auto_target_ratio=1.0,
            auto_min_size=1,
            max_windows=1,
            x_radius=0,
            ell_radius=0,
            max_mb=1.0,
            regularization=0.0,
            max_size=1,
            column_batch=1,
            drop_tol=0.0,
            low_lmax=0,
            profile_moment_count=0,
            angular_lmax=0,
            angular_mode_max=0,
            max_tail_units=0,
            include_tail=False,
            include_constraint_sources=False,
            include_fsavg=False,
            include_window_residual=False,
            include_profile_moments=False,
            include_angular_residual=False,
            include_angular_basis=False,
            include_preconditioned_loads=False,
            preconditioned_load_max_columns=0,
            preconditioned_load_max_nnz=0,
            preconditioned_load_drop_tol=0.0,
            damping=False,
            beta_max=1.0,
            accept_base_improvement=False,
        )
    )

    assert result.requested
    assert result.selected
    assert result.factor_bundle_pc is bundle
    assert result.residual_after == pytest.approx(0.0)
    assert result.factor_preflight_passed
    assert result.x0_sparse is not None
    assert any("true coupled coarse accepted" in message for message in emits)


def test_sparse_pc_generic_branch_setup_defers_xblock_backend_pattern_build() -> None:
    op = SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=1,
        phi1_size=0,
        include_phi1=False,
        total_size=2,
        n_species=1,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    result = sparse_pc_module.build_sparse_pc_generic_branch_setup(
        sparse_pc_module.SparsePCGenericBranchSetupContext(
            op=op,
            rhs=rhs,
            sparse_pc_use_active_dof=False,
            active_dof_indices=lambda _op: np.asarray([0], dtype=np.int32),
            reduce_full_with_indices=lambda vec, idx: vec[idx],
            expand_reduced_with_map=lambda vec, _map: vec,
            fortran_reduced_sparse_pc=True,
            preconditioner_x=0,
            preconditioner_x_min_l=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            sparse_pc_fp_dense_velocity_block=None,
            constrained_pas_pc=False,
            tokamak_pas_er_pc=False,
            tokamak_fp_pc=False,
            pc_maxiter=4,
            pc_restart=4,
            host_sparse_factor_dtype=np.dtype(np.float64),
            sparse_timer=SimpleNamespace(elapsed_s=lambda: 0.0),
            emit=None,
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "xblock"},
            default_permc_spec=lambda **_kwargs: "COLAMD",
            build_fortran_reduced_operator=lambda op_in, **_kwargs: SimpleNamespace(source=op_in),
            build_point_operator=lambda op_in: op_in,
            fortran_reduced_pattern_for_indices=lambda **_kwargs: None,
            fortran_reduced_pattern=lambda **_kwargs: None,
            conservative_pattern_for_indices=lambda **_kwargs: None,
            conservative_pattern=lambda **_kwargs: None,
            summarize_pattern=lambda _pattern: None,
            estimate_sparse_pc_memory=lambda **_kwargs: 0,
            device_count=1,
        )
    )

    assert result.linear_size == 2
    assert result.preconditioner_operator == "fortran_reduced_global"
    assert result.fortran_reduced_sparse_pc_backend == "xblock"
    assert result.sparse_pattern_scope == "fortran_reduced_xblock_deferred"
    assert result.pattern is None
    assert result.factor_policy is None
    np.testing.assert_allclose(np.asarray(result.reduce_full(rhs)), np.asarray(rhs))


def test_sparse_pc_generic_branch_setup_builds_active_global_pattern() -> None:
    op = SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=1,
        phi1_size=0,
        include_phi1=False,
        total_size=3,
        n_species=1,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    pattern = scipy_sparse.eye(2, format="csr")
    emits: list[str] = []

    result = sparse_pc_module.build_sparse_pc_generic_branch_setup(
        sparse_pc_module.SparsePCGenericBranchSetupContext(
            op=op,
            rhs=jnp.asarray([1.0, 9.0, -1.0], dtype=jnp.float64),
            sparse_pc_use_active_dof=True,
            active_dof_indices=lambda _op: np.asarray([0, 2], dtype=np.int32),
            reduce_full_with_indices=reduce_full_with_indices,
            expand_reduced_with_map=expand_reduced_with_map,
            fortran_reduced_sparse_pc=True,
            preconditioner_x=1,
            preconditioner_x_min_l=0,
            preconditioner_xi=1,
            preconditioner_species=1,
            sparse_pc_fp_dense_velocity_block=None,
            constrained_pas_pc=False,
            tokamak_pas_er_pc=False,
            tokamak_fp_pc=False,
            pc_maxiter=4,
            pc_restart=4,
            host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
            sparse_timer=SimpleNamespace(elapsed_s=lambda: 1.0),
            emit=lambda level, message: emits.append(f"{level}:{message}"),
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "global"},
            default_permc_spec=lambda **_kwargs: "COLAMD",
            build_fortran_reduced_operator=lambda op_in, **_kwargs: SimpleNamespace(source=op_in),
            build_point_operator=lambda op_in: op_in,
            fortran_reduced_pattern_for_indices=lambda *_args, **_kwargs: pattern,
            fortran_reduced_pattern=lambda *_args, **_kwargs: pattern,
            conservative_pattern_for_indices=lambda *_args, **_kwargs: pattern,
            conservative_pattern=lambda *_args, **_kwargs: pattern,
            summarize_pattern=lambda _op, _pattern: SimpleNamespace(
                nnz=2,
                avg_row_nnz=1.0,
                max_row_nnz=1,
            ),
            estimate_sparse_pc_memory=lambda **_kwargs: 0,
            device_count=1,
        )
    )

    assert result.linear_size == 2
    assert result.pattern is pattern
    assert result.sparse_pattern_scope == "fortran_reduced_active_dof"
    assert result.factor_policy is not None
    np.testing.assert_allclose(np.asarray(result.rhs), [1.0, -1.0])
    assert any("active-DOF reduction enabled" in message for message in emits)
    assert any("using global angular-coupled" in message for message in emits)


def test_sparse_pc_direct_tail_factor_setup_falls_back_to_host_factor(monkeypatch) -> None:
    materialization = sparse_pc_module.DirectTailMaterializationResult(
        direct_tail_default=False,
        enabled=False,
        built=False,
        error=None,
        operator_bundle=None,
        pc_env="off",
        direct_reduced_pmat_requested=False,
    )
    admission = sparse_pc_module.DirectTailStructuredAdmissionResult(
        pc_env="off",
        requested=None,
        auto_default=False,
        fail_closed_size=0,
        auto_large_fail_closed=False,
        required=False,
        setup_allowed=False,
        max_mb_auto=False,
        max_mb=1.0,
        regularization=1.0e-12,
    )
    operator_bundle = SimpleNamespace(kind="operator")
    factor_bundle = SimpleNamespace(kind="host_factor")

    monkeypatch.setattr(
        sparse_pc_module,
        "build_direct_tail_materialization_setup",
        lambda _context: materialization,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "resolve_direct_tail_structured_admission",
        lambda _context: admission,
    )

    result = sparse_pc_module.build_sparse_pc_direct_tail_factor_setup(
        sparse_pc_module.SparsePCDirectTailFactorSetupContext(
            env={},
            op=SimpleNamespace(),
            op_pc=SimpleNamespace(),
            rhs_dtype=jnp.float64,
            pattern=SimpleNamespace(nnz=1),
            active_indices=None,
            sparse_pc_use_active_dof=False,
            reduce_full=lambda vec: vec,
            expand_reduced=lambda vec: vec,
            factor_matvec=lambda vec: vec,
            pc_shift=0.0,
            factor_dtype_initial=np.dtype(np.float64),
            factorization="lu",
            default_factor_kind="lu",
            default_ilu_fill_factor=1.0,
            default_ilu_drop_tol=0.0,
            default_pattern_color_batch=1,
            default_permc_spec="COLAMD",
            permc_spec="COLAMD",
            sparse_pc_linear_size=1,
            constrained_pas_pc=False,
            tokamak_fp_pc=False,
            fortran_reduced_sparse_pc=False,
            preconditioner_x=0,
            preconditioner_x_min_l=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            sparse_timer=SimpleNamespace(elapsed_s=lambda: 2.0),
            emit=None,
            default_direct_tail_max_mb=lambda **_kwargs: 1.0,
            is_direct_reduced_pmat_pc_kind=lambda _kind: False,
            build_direct_tail_bundle=lambda **_kwargs: None,
            build_structured_full_csr_operator_bundle=lambda **_kwargs: None,
            layout_from_operator=lambda _op: None,
            build_direct_active_preconditioner=lambda **_kwargs: None,
            build_active_projected_preconditioner=lambda **_kwargs: None,
            structured_cache={},
            structured_cache_key=lambda **_kwargs: (),
            structured_cache_metadata=lambda _value: None,
            structured_factor_bundle_factory=lambda **_kwargs: None,
            host_factor_builder=lambda **_kwargs: (operator_bundle, factor_bundle),
        )
    )

    assert result.materialization is materialization
    assert not result.structured_pc_ready
    assert result.operator_bundle_pc is operator_bundle
    assert result.factor_bundle_pc is factor_bundle
    assert result.direct_tail_pc_env == "off"
    assert result.pc_max_mb == pytest.approx(1.0)


def test_sparse_pc_direct_tail_factor_setup_selects_structured_path(monkeypatch) -> None:
    operator_bundle = SimpleNamespace(kind="operator", matrix=scipy_sparse.eye(1))
    factor_bundle = SimpleNamespace(kind="structured_factor")
    materialization = sparse_pc_module.DirectTailMaterializationResult(
        direct_tail_default=True,
        enabled=True,
        built=True,
        error=None,
        operator_bundle=operator_bundle,
        pc_env="auto",
        direct_reduced_pmat_requested=True,
    )
    admission = sparse_pc_module.DirectTailStructuredAdmissionResult(
        pc_env="auto",
        requested="active_dense",
        auto_default=True,
        fail_closed_size=0,
        auto_large_fail_closed=False,
        required=False,
        setup_allowed=True,
        max_mb_auto=False,
        max_mb=1.0,
        regularization=1.0e-12,
    )
    structured_build = SimpleNamespace(
        layout=object(),
        active_indices=np.asarray([0], dtype=np.int64),
        max_nbytes=1024,
        preconditioner=SimpleNamespace(kind="active_dense"),
        factor_bundle=factor_bundle,
        operator_bundle_pc=operator_bundle,
        ready=True,
        selected=True,
        reason="selected",
        metadata={
            "kind": "active_dense",
            "setup_s": 0.1,
            "metadata": {
                "factor_nbytes_actual": 8,
                "permc_spec": "COLAMD",
                "superlu_permc_spec": "COLAMD",
            },
        },
        error=None,
        cache_hit=False,
        cache_key=("key",),
    )
    emits: list[str] = []

    monkeypatch.setattr(
        sparse_pc_module,
        "build_direct_tail_materialization_setup",
        lambda _context: materialization,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "resolve_direct_tail_structured_admission",
        lambda _context: admission,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "build_direct_tail_structured_preconditioner_setup",
        lambda _context: structured_build,
    )

    result = sparse_pc_module.build_sparse_pc_direct_tail_factor_setup(
        sparse_pc_module.SparsePCDirectTailFactorSetupContext(
            env={},
            op=SimpleNamespace(),
            op_pc=SimpleNamespace(),
            rhs_dtype=jnp.float64,
            pattern=SimpleNamespace(nnz=1),
            active_indices=None,
            sparse_pc_use_active_dof=False,
            reduce_full=lambda vec: vec,
            expand_reduced=lambda vec: vec,
            factor_matvec=lambda vec: vec,
            pc_shift=1.0e-8,
            factor_dtype_initial=np.dtype(np.float64),
            factorization="lu",
            default_factor_kind="lu",
            default_ilu_fill_factor=1.0,
            default_ilu_drop_tol=0.0,
            default_pattern_color_batch=1,
            default_permc_spec="COLAMD",
            permc_spec="COLAMD",
            sparse_pc_linear_size=1,
            constrained_pas_pc=False,
            tokamak_fp_pc=False,
            fortran_reduced_sparse_pc=True,
            preconditioner_x=0,
            preconditioner_x_min_l=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            sparse_timer=SimpleNamespace(elapsed_s=lambda: 2.0),
            emit=lambda level, message: emits.append(f"{level}:{message}"),
            default_direct_tail_max_mb=lambda **_kwargs: 1.0,
            is_direct_reduced_pmat_pc_kind=lambda _kind: True,
            build_direct_tail_bundle=lambda **_kwargs: None,
            build_structured_full_csr_operator_bundle=lambda **_kwargs: None,
            layout_from_operator=lambda _op: None,
            build_direct_active_preconditioner=lambda **_kwargs: None,
            build_active_projected_preconditioner=lambda **_kwargs: None,
            structured_cache={},
            structured_cache_key=lambda **_kwargs: (),
            structured_cache_metadata=lambda _value: None,
            structured_factor_bundle_factory=lambda **_kwargs: None,
            host_factor_builder=lambda **_kwargs: pytest.fail("host factor fallback was not expected"),
        )
    )

    assert result.structured_pc_ready
    assert result.factor_bundle_pc is factor_bundle
    assert result.operator_bundle_pc is operator_bundle
    assert result.direct_tail_structured_pc_selected
    assert result.pc_max_mb == pytest.approx(1.0)
    assert any("structured preconditioner selected" in message for message in emits)


def test_sparse_pc_direct_tail_factor_setup_required_structured_path_fails_fast(
    monkeypatch,
) -> None:
    operator_bundle = SimpleNamespace(kind="operator", matrix=scipy_sparse.eye(1))
    materialization = sparse_pc_module.DirectTailMaterializationResult(
        direct_tail_default=False,
        enabled=True,
        built=True,
        error=None,
        operator_bundle=operator_bundle,
        pc_env="active_dense!",
        direct_reduced_pmat_requested=True,
    )
    admission = sparse_pc_module.DirectTailStructuredAdmissionResult(
        pc_env="active_dense!",
        requested="active_dense",
        auto_default=False,
        fail_closed_size=0,
        auto_large_fail_closed=False,
        required=True,
        setup_allowed=True,
        max_mb_auto=False,
        max_mb=1.0,
        regularization=1.0e-12,
    )
    structured_build = SimpleNamespace(
        layout=object(),
        active_indices=np.asarray([0], dtype=np.int64),
        max_nbytes=1024,
        preconditioner=None,
        factor_bundle=None,
        operator_bundle_pc=None,
        ready=False,
        selected=False,
        reason="memory_gate",
        metadata={"kind": "active_dense", "reason": "memory_gate"},
        error="factor estimate exceeds budget",
        cache_hit=False,
        cache_key=None,
    )
    emits: list[str] = []

    monkeypatch.setattr(
        sparse_pc_module,
        "build_direct_tail_materialization_setup",
        lambda _context: materialization,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "resolve_direct_tail_structured_admission",
        lambda _context: admission,
    )
    monkeypatch.setattr(
        sparse_pc_module,
        "build_direct_tail_structured_preconditioner_setup",
        lambda _context: structured_build,
    )

    with pytest.raises(RuntimeError, match="explicitly requested but not selected"):
        sparse_pc_module.build_sparse_pc_direct_tail_factor_setup(
            sparse_pc_module.SparsePCDirectTailFactorSetupContext(
                env={},
                op=SimpleNamespace(),
                op_pc=SimpleNamespace(),
                rhs_dtype=jnp.float64,
                pattern=SimpleNamespace(nnz=1),
                active_indices=None,
                sparse_pc_use_active_dof=False,
                reduce_full=lambda vec: vec,
                expand_reduced=lambda vec: vec,
                factor_matvec=lambda vec: vec,
                pc_shift=0.0,
                factor_dtype_initial=np.dtype(np.float64),
                factorization="lu",
                default_factor_kind="lu",
                default_ilu_fill_factor=1.0,
                default_ilu_drop_tol=0.0,
                default_pattern_color_batch=1,
                default_permc_spec="COLAMD",
                permc_spec="COLAMD",
                sparse_pc_linear_size=1,
                constrained_pas_pc=False,
                tokamak_fp_pc=False,
                fortran_reduced_sparse_pc=True,
                preconditioner_x=0,
                preconditioner_x_min_l=0,
                preconditioner_xi=0,
                preconditioner_species=0,
                sparse_timer=SimpleNamespace(elapsed_s=lambda: 2.0),
                emit=lambda level, message: emits.append(f"{level}:{message}"),
                default_direct_tail_max_mb=lambda **_kwargs: 1.0,
                is_direct_reduced_pmat_pc_kind=lambda _kind: True,
                build_direct_tail_bundle=lambda **_kwargs: None,
                build_structured_full_csr_operator_bundle=lambda **_kwargs: None,
                layout_from_operator=lambda _op: None,
                build_direct_active_preconditioner=lambda **_kwargs: None,
                build_active_projected_preconditioner=lambda **_kwargs: None,
                structured_cache={},
                structured_cache_key=lambda **_kwargs: (),
                structured_cache_metadata=lambda _value: None,
                structured_factor_bundle_factory=lambda **_kwargs: None,
                host_factor_builder=lambda **_kwargs: pytest.fail("required structured path must not fall back"),
            )
        )

    assert any("required path will fail fast" in message for message in emits)


def test_sparse_pc_direct_tail_rescue_policy_setup_defaults_without_support_preflight() -> None:
    factor = SimpleNamespace(kind="host_factor")
    result = sparse_pc_module.build_sparse_pc_direct_tail_rescue_policy_setup(
        sparse_pc_module.SparsePCDirectTailRescuePolicySetupContext(
            env={},
            op=SimpleNamespace(),
            rhs_dtype=jnp.float64,
            sparse_pc_rhs=jnp.asarray([1.0], dtype=jnp.float64),
            sparse_pc_linear_size=1,
            fortran_reduced_sparse_pc=False,
            factor_bundle_pc=factor,
            structured_pc_ready=False,
            direct_tail_operator_bundle=None,
            direct_tail_structured_layout=None,
            direct_tail_structured_active_indices=None,
            direct_tail_structured_max_nbytes=None,
            direct_tail_structured_pc_selected=False,
            direct_tail_structured_pc_reason=None,
            direct_tail_structured_pc_metadata=None,
            pc_reg=0.0,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            emit=None,
            true_matvec=lambda vec: vec,
            support_mode_selector=lambda **_kwargs: None,
            structured_factor_bundle_factory=lambda **_kwargs: None,
        )
    )

    assert result.factor_bundle_pc is factor
    assert not result.direct_tail_support_mode_preflight_requested
    assert not result.direct_tail_support_mode_preflight_selected
    assert result.factor_preflight_passed is None
    assert "direct_tail_residual_coarse_requested" in result.rescue_values


def test_sparse_pc_direct_tail_rescue_policy_setup_promotes_support_mode(monkeypatch) -> None:
    class SupportPreconditioner:
        kind = "active_dense"
        reason = "support_mode_selected"
        metadata = {"factor_nbytes_actual": 8}

        def to_dict(self) -> dict[str, object]:
            return {
                "kind": self.kind,
                "reason": self.reason,
                "metadata": dict(self.metadata),
            }

    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu")
    support_factor = SimpleNamespace(kind="active_dense")
    emits: list[str] = []
    monkeypatch.setattr(
        sparse_pc_module,
        "run_direct_tail_support_mode_preflight",
        lambda _context: SimpleNamespace(
            metadata={
                "selected_candidate": "active_dense",
                "baseline_residual_after": 2.0,
                "best_residual_after": 0.5,
                "accepted_nonbaseline": True,
            },
            error=None,
            selected=True,
            preconditioner=SupportPreconditioner(),
            factor_bundle=support_factor,
        ),
    )

    result = sparse_pc_module.build_sparse_pc_direct_tail_rescue_policy_setup(
        sparse_pc_module.SparsePCDirectTailRescuePolicySetupContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1"},
            op=SimpleNamespace(),
            rhs_dtype=jnp.float64,
            sparse_pc_rhs=jnp.asarray([1.0], dtype=jnp.float64),
            sparse_pc_linear_size=1,
            fortran_reduced_sparse_pc=True,
            factor_bundle_pc=factor,
            structured_pc_ready=True,
            direct_tail_operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(1)),
            direct_tail_structured_layout=object(),
            direct_tail_structured_active_indices=np.asarray([0], dtype=np.int64),
            direct_tail_structured_max_nbytes=1024,
            direct_tail_structured_pc_selected=False,
            direct_tail_structured_pc_reason=None,
            direct_tail_structured_pc_metadata=None,
            pc_reg=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            emit=lambda level, message: emits.append(f"{level}:{message}"),
            true_matvec=lambda vec: vec,
            support_mode_selector=lambda **_kwargs: None,
            structured_factor_bundle_factory=lambda **_kwargs: None,
        )
    )

    assert result.factor_bundle_pc is support_factor
    assert result.direct_tail_structured_pc_selected
    assert result.direct_tail_support_mode_preflight_selected
    assert result.direct_tail_structured_pc_metadata["kind"] == "active_dense"
    assert any("support-mode preflight selected" in message for message in emits)


def test_sparse_pc_direct_tail_rescue_policy_setup_reports_support_failure(
    monkeypatch,
) -> None:
    factor = SimpleNamespace(kind="active_fortran_v3_reduced_lu")
    emits: list[str] = []
    monkeypatch.setattr(
        sparse_pc_module,
        "run_direct_tail_support_mode_preflight",
        lambda _context: SimpleNamespace(
            metadata={"attempted": True},
            error="support candidates exhausted",
            selected=False,
            preconditioner=None,
            factor_bundle=None,
        ),
    )

    result = sparse_pc_module.build_sparse_pc_direct_tail_rescue_policy_setup(
        sparse_pc_module.SparsePCDirectTailRescuePolicySetupContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1",
            },
            op=SimpleNamespace(),
            rhs_dtype=jnp.float64,
            sparse_pc_rhs=jnp.asarray([1.0], dtype=jnp.float64),
            sparse_pc_linear_size=1,
            fortran_reduced_sparse_pc=True,
            factor_bundle_pc=factor,
            structured_pc_ready=True,
            direct_tail_operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(1)),
            direct_tail_structured_layout=object(),
            direct_tail_structured_active_indices=np.asarray([0], dtype=np.int64),
            direct_tail_structured_max_nbytes=1024,
            direct_tail_structured_pc_selected=True,
            direct_tail_structured_pc_reason="initial",
            direct_tail_structured_pc_metadata={"kind": "active_dense"},
            pc_reg=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            emit=lambda level, message: emits.append(f"{level}:{message}"),
            true_matvec=lambda vec: vec,
            support_mode_selector=lambda **_kwargs: None,
            structured_factor_bundle_factory=lambda **_kwargs: None,
        )
    )

    assert result.factor_bundle_pc is factor
    assert result.direct_tail_support_mode_preflight_requested
    assert not result.direct_tail_support_mode_preflight_selected
    assert result.direct_tail_support_mode_preflight_error == "support candidates exhausted"
    assert result.direct_tail_structured_pc_selected
    assert result.direct_tail_structured_pc_reason == "initial"
    assert result.direct_tail_structured_pc_metadata == {"kind": "active_dense"}
    assert any("support-mode preflight failed" in message for message in emits)


def test_sparse_minimum_norm_lsqr_and_host_direct_residual_fallback() -> None:
    matrix = scipy_sparse.eye(2, format="csr")
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    policy = SparseMinimumNormPolicy(
        solver_name="lsqr",
        atol=1.0e-12,
        btol=1.0e-12,
        conlim=1.0e8,
        damp=0.0,
        maxiter=20,
        show=False,
        petsc_compat_requested=True,
    )

    payload = sparse_minimum_norm_solve_payload(
        matrix=matrix,
        rhs=rhs,
        policy=policy,
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm(np.asarray(rhs))),
        elapsed_s=lambda: 0.25,
    )

    np.testing.assert_allclose(np.asarray(payload.x), np.asarray(rhs), rtol=1.0e-10)
    assert payload.metadata["solver_kind"] == "sparse_lsmr"
    assert payload.metadata["petsc_compat_requested"] is True
    assert "solver=lsqr" in payload.start_message

    direct_payload = sparse_host_direct_solve_payload(
        factor_solve=lambda vec: np.asarray(vec, dtype=np.float64),
        operator_matrix=matrix,
        rhs=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=0,
        matvec=lambda _vec: (_ for _ in ()).throw(RuntimeError("matvec failed")),
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=1.0,
        elapsed_s=lambda: 0.1,
        direct_solve_with_refinement=lambda **_kwargs: (np.asarray(rhs), 7.0),
    )

    assert float(direct_payload.residual_norm) == pytest.approx(7.0)
    assert not direct_payload.metadata["accepted_converged"]


def test_sparse_host_direct_polish_disabled_and_factor_guard_errors() -> None:
    polish = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([1.0, 0.0], dtype=np.float64),
        residual_norm=10.0,
        factor_dtype=np.dtype(np.float32),
        target=1.0,
        matvec=lambda vec: jnp.asarray(vec, dtype=jnp.float64),
        rhs=jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-9,
        atol=1.0e-12,
        restart=40,
        maxiter=80,
        precondition_side="left",
        emit=None,
        polish_enabled=lambda **_kwargs: False,
        parse_polish_gmres_config=lambda **_kwargs: (10, 20),
        host_sparse_direct_polish=lambda **_kwargs: (np.zeros(2), 0.0),
    )
    assert not polish.attempted
    assert not polish.accepted
    assert float(polish.residual_norm) == pytest.approx(10.0)

    base_kwargs = dict(
        matvec=lambda vec: vec,
        n=2,
        dtype=jnp.float64,
        cache_key=("case",),
        factor_dtype=np.dtype(np.float64),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
        build_dense_factors=False,
        build_jax_factors=False,
        store_dense=False,
        factorization="lu",
        emit=None,
    )
    with pytest.raises(ValueError, match="explicit sparse host factor"):
        build_sparse_host_or_ilu_factor(
            SparseHostOrILUFactorBuildContext(
                **base_kwargs,
                host_sparse_direct_wanted=True,
                explicit_sparse_allowed=True,
            )
        )
    with pytest.raises(ValueError, match="ILU factor"):
        build_sparse_host_or_ilu_factor(
            SparseHostOrILUFactorBuildContext(
                **base_kwargs,
                host_sparse_direct_wanted=False,
                explicit_sparse_allowed=False,
            )
        )


def test_sparse_factor_controls_force_host_direct_and_cache_dtype() -> None:
    calls: list[dict[str, object]] = []

    controls = resolve_sparse_host_or_ilu_factor_controls(
        n=9,
        cache_key=("base",),
        sparse_exact_lu=True,
        use_implicit=False,
        force_host_sparse_direct=True,
        sparse_ilu_dense_max=4,
        sparse_dense_cache_max=16,
        host_sparse_direct_wanted=False,
        host_sparse_direct_allowed=lambda **_kwargs: False,
        host_sparse_factor_dtype=lambda **kwargs: calls.append(kwargs) or np.dtype(np.float32),
        sparse_factor_cache_key=lambda cache_key, factor_dtype: (
            *cache_key,
            np.dtype(factor_dtype).str,
        ),
        explicit_sparse_host_direct_allowed=lambda **_kwargs: True,
    )

    assert controls.host_sparse_direct_wanted
    assert controls.factor_dtype == np.dtype(np.float32)
    assert controls.cache_key_use == ("base", "<f4")
    assert controls.explicit_sparse_allowed
    assert not controls.build_dense_factors
    assert controls.store_dense
    assert calls == [{"size": 9, "factorization": "lu", "use_implicit": False}]


def test_sparse_direct_wrapper_pattern_cache_and_sparse_jax(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_build_host_sparse_direct_factor_from_matvec_impl(**kwargs):
        seen.update(kwargs)
        return "operator", "factor"

    monkeypatch.setattr(
        sparse_direct_module,
        "_build_host_sparse_direct_factor_from_matvec_impl",
        fake_build_host_sparse_direct_factor_from_matvec_impl,
    )
    operator, factor = sparse_direct_module.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda vec: vec,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        pattern=None,
        emit=None,
    )
    assert (operator, factor) == ("operator", "factor")
    assert seen["default_backend_callback"] is sparse_direct_module.jax.default_backend

    emits: list[tuple[int, str]] = []
    monkeypatch.setattr(
        sparse_direct_module,
        "v3_full_system_conservative_sparsity_pattern",
        lambda op: "pattern",
    )
    monkeypatch.setattr(
        sparse_direct_module,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(shape=(2, 2), nnz=3, avg_row_nnz=1.5, max_row_nnz=2),
    )
    pattern = sparse_direct_module.maybe_rhsmode1_full_sparse_pattern(
        op=object(),
        emit=lambda level, message: emits.append((level, message)),
        env={"SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_PATTERN": "pattern"},
    )
    assert pattern == "pattern"
    assert "avg_row_nnz=1.5" in emits[0][1]

    monkeypatch.setattr(
        sparse_direct_module,
        "_rhsmode1_precond_cache_key",
        lambda _op, kind: ("operator-key", kind),
    )
    cache_key = sparse_direct_module.rhsmode1_sparse_cache_key(
        SimpleNamespace(total_size=2, rhs_mode=1),
        kind="jacobi",
        active_size=2,
        use_active_dof_mode=True,
        use_pas_projection=False,
        drop_tol=1.0e-4,
        drop_rel=1.0e-3,
        ilu_drop_tol=1.0e-2,
        fill_factor=3.0,
    )
    assert cache_key[-5:] == (1, 0, 1.0e-4, 1.0e-3, 1.0e-2, 3.0)[-5:]

    sparse_direct_module._RHSMODE1_SPARSE_JAX_CACHE.clear()
    sparse_emits: list[str] = []
    preconditioner = sparse_direct_module.build_sparse_jax_preconditioner_from_matvec(
        matvec=lambda vec: jnp.asarray(vec, dtype=jnp.float64),
        n=2,
        dtype=jnp.float64,
        cache_key=("sparse-jax",),
        drop_tol=0.0,
        drop_rel=0.0,
        reg=1.0,
        omega=0.5,
        sweeps=2,
        emit=lambda _level, message: sparse_emits.append(message),
    )
    result = preconditioner(jnp.asarray([2.0, 0.0], dtype=jnp.float64))
    assert np.all(np.isfinite(np.asarray(result)))
    assert any("assembling dense operator" in message for message in sparse_emits)

    cached = sparse_direct_module.build_sparse_jax_preconditioner_from_matvec(
        matvec=lambda vec: jnp.asarray(vec, dtype=jnp.float64),
        n=2,
        dtype=jnp.float64,
        cache_key=("sparse-jax",),
        drop_tol=1.0,
        drop_rel=1.0,
        reg=9.0,
        omega=9.0,
        sweeps=9,
        emit=lambda _level, message: sparse_emits.append(message),
    )
    np.testing.assert_allclose(np.asarray(cached(jnp.asarray([2.0, 0.0]))), np.asarray(result))


def test_solve_fortran_reduced_xblock_backend_rejects_non_full_fp_pc() -> None:
    values = {
        dataclass_field.name: None
        for dataclass_field in fields(sparse_pc_module.FortranReducedXBlockBackendContext)
    }
    values.update(
        op=SimpleNamespace(),
        op_pc=SimpleNamespace(fblock=SimpleNamespace(fp=None, pas=None)),
        sparse_timer=SimpleNamespace(elapsed_s=lambda: 0.0),
    )

    with pytest.raises(NotImplementedError, match="full-FP RHSMode=1"):
        sparse_pc_module.solve_fortran_reduced_xblock_backend(
            sparse_pc_module.FortranReducedXBlockBackendContext(**values)
        )


def test_sparse_direct_module_exposes_canonical_public_contract() -> None:
    """The split sparse direct module owns its public sparse-PC helpers."""

    moved_names = (
        "DirectTailCoupledCoarseRescuePolicy",
        "DirectTailMaterializationContext",
        "DirectTailMaterializationResult",
        "DirectTailResidualRescuePolicy",
        "DirectTailStructuredAdmissionContext",
        "DirectTailStructuredAdmissionResult",
        "DirectTailStructuredBuildContext",
        "DirectTailStructuredBuildResult",
        "DirectTailSupportModePreflightContext",
        "DirectTailSupportModePreflightResult",
        "DirectTailTrueActiveRescuePolicy",
        "ExplicitSparseHostDirectBranchContext",
        "ExplicitSparseMinimumNormBranchContext",
        "ExplicitSparseOperatorBuildPolicy",
        "ExplicitSparseOperatorBuildResult",
        "SparsePCDirectTailFinalMetadataContext",
        "SparseHostDirectFactorSolvePayload",
        "SparseHostDirectFallbackPayload",
        "SparseHostDirectPayload",
        "SparseHostDirectPolishPayload",
        "SparseHostOrILUFactorBuildContext",
        "SparseHostOrILUFactorBuildResult",
        "SparseHostOrILUFactorControls",
        "SparseHostRetryCandidateContext",
        "SparseHostRetryCandidateResult",
        "SparseHostScipyGMRESContext",
        "SparseHostScipyPreconditionerBuildContext",
        "SparseHostScipyPreconditionerBuildResult",
        "SparseILUPreconditionerBuildContext",
        "SparseILUPreconditionerBuildResult",
        "SparseJAXRetryPreconditionerBuildContext",
        "SparseMinimumNormPayload",
        "SparseMinimumNormPolicy",
        "apply_sparse_host_direct_polish_if_needed",
        "build_direct_tail_materialization_setup",
        "build_direct_tail_structured_preconditioner_setup",
        "build_explicit_sparse_operator_from_pattern",
        "build_sparse_host_or_ilu_factor",
        "build_sparse_host_scipy_preconditioner",
        "build_sparse_ilu_preconditioner_from_cache",
        "build_sparse_jax_retry_preconditioner",
        "explicit_sparse_pattern_progress_messages",
        "resolve_direct_tail_coupled_coarse_rescue_policy",
        "resolve_direct_tail_residual_rescue_policy",
        "resolve_direct_tail_structured_admission",
        "resolve_direct_tail_true_active_rescue_policy",
        "resolve_explicit_sparse_operator_build_policy",
        "resolve_sparse_host_or_ilu_factor_controls",
        "resolve_sparse_minimum_norm_policy",
        "run_direct_tail_support_mode_preflight",
        "run_sparse_host_retry_candidate",
        "run_sparse_host_scipy_gmres",
        "solve_explicit_sparse_host_direct_branch",
        "solve_explicit_sparse_minimum_norm_branch",
        "solve_sparse_host_direct_from_available_factor",
        "sparse_host_direct_fallback_payload",
        "sparse_host_direct_solve_from_pattern",
        "sparse_host_direct_solve_payload",
        "sparse_pc_direct_tail_final_metadata",
        "sparse_minimum_norm_solve_from_pattern",
        "sparse_minimum_norm_solve_payload",
        "sparse_minimum_norm_start_message",
        "validate_explicit_sparse_host_request",
    )
    for name in moved_names:
        assert hasattr(sparse_direct_module, name)
        assert name in sparse_direct_module.__all__


def test_sparse_finalization_module_exposes_canonical_public_contract() -> None:
    """The split sparse finalization module owns its public sparse-PC helpers."""

    moved_names = (
        "SparsePCFactorDtypeRetryContext",
        "SparsePCFactorDtypeRetryDecision",
        "SparsePCFactorDtypeRetryFinalizationContext",
        "SparsePCFactorDtypeRetryResult",
        "SparsePCGMRESCompletionMessageContext",
        "SparsePCGMRESContext",
        "SparsePCGMRESFinalPayload",
        "SparsePCGMRESFinalResultContext",
        "SparsePCGMRESFinalizationBundleContext",
        "SparsePCGMRESFinalizationContext",
        "SparsePCGMRESFinalizationStateContext",
        "SparsePCGMRESResult",
        "SparsePCPostMinresContext",
        "SparsePCPostMinresFinalizationContext",
        "SparsePCPostMinresResult",
        "SparsePCPostMinresUpdateContext",
        "SparsePCPostMinresUpdateResult",
        "apply_sparse_pc_post_minres",
        "apply_sparse_pc_post_minres_from_solve_state",
        "apply_sparse_pc_post_minres_if_needed",
        "emit_sparse_pc_gmres_completion_from_solve_state",
        "evaluate_sparse_pc_factor_dtype_retry",
        "finalize_sparse_pc_gmres_bundle",
        "finalize_sparse_pc_gmres_from_solve_state",
        "finalize_sparse_pc_gmres_with_dtype_retry",
        "finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state",
        "run_sparse_pc_gmres_once",
        "run_sparse_pc_gmres_once_for_retry",
        "retry_sparse_pc_factor_dtype_from_solve_state",
        "retry_sparse_pc_factor_dtype_from_finalization_context",
        "retry_sparse_pc_factor_dtype_if_needed",
        "sparse_pc_factor_dtype_retry_initial_guess",
        "sparse_pc_gmres_completion_message",
        "sparse_pc_gmres_finalization_bundle_from_solve_result",
        "sparse_pc_gmres_finalization_bundle_from_solve_scope",
        "sparse_pc_gmres_finalization_solve_scope_keys",
        "sparse_pc_gmres_finalization_solve_state_keys",
        "sparse_pc_gmres_finalization_state_from_context",
        "sparse_pc_gmres_finalization_state_from_solve_scope",
        "sparse_pc_gmres_final_payload_from_solve_state",
    )
    for name in moved_names:
        assert hasattr(sparse_finalization_module, name)
        assert name in sparse_finalization_module.__all__


def test_sparse_krylov_helpers_live_on_sparse_pc_owner() -> None:
    """Sparse-PC Krylov helpers live directly on the finalization owner."""

    moved_names = (
        "SparsePCGMRESContext",
        "run_sparse_pc_gmres_once",
        "run_sparse_pc_gmres_once_for_retry",
    )
    for name in moved_names:
        assert hasattr(sparse_finalization_module, name)
        assert name in sparse_finalization_module.__all__


def test_sparse_policy_module_exposes_canonical_public_contract() -> None:
    """The split sparse policy module owns its public sparse-PC helpers."""

    moved_names = (
        "SparsePCActiveDOFSetup",
        "SparsePCEntryPolicySetup",
        "SparsePCFactorPolicySetup",
        "SparsePCPatternSetupContext",
        "SparsePCPatternSetupResult",
        "SparsePCMemoryBudgetPreflightContext",
        "SparsePCFactorPreflightPolicyContext",
        "SparsePCFactorPreflightPolicy",
        "SparsePCFactorPreflightEvaluationContext",
        "SparsePCFactorPreflightEvaluationResult",
        "SparsePCResidualCandidateAcceptanceContext",
        "SparsePCResidualCandidateAcceptanceResult",
        "SparsePCAutoPreflightRetrySelectionContext",
        "SparsePCAutoPreflightRetrySelectionResult",
        "SparsePCAutoPreflightRetryEvaluationContext",
        "SparsePCAutoPreflightRetryEvaluationResult",
        "SparsePCGMRESControlPolicy",
        "build_sparse_pc_active_dof_setup",
        "build_sparse_pc_pattern_setup",
        "resolve_sparse_pc_entry_policy",
        "resolve_sparse_pc_factor_policy",
        "enforce_sparse_pc_memory_budget",
        "resolve_sparse_pc_factor_preflight_policy",
        "evaluate_sparse_pc_factor_preflight",
        "evaluate_sparse_pc_residual_candidate_acceptance",
        "select_sparse_pc_auto_preflight_retry_candidates",
        "evaluate_sparse_pc_auto_preflight_retry",
        "resolve_sparse_pc_gmres_control_policy",
    )
    for name in moved_names:
        assert hasattr(sparse_policy_module, name)
        assert name in sparse_policy_module.__all__


def test_sparse_fortran_reduced_module_exposes_canonical_public_contract() -> None:
    """The split fortran-reduced module owns its public sparse-PC helpers."""

    moved_names = (
        "FortranReducedSparsePCBackendSetup",
        "FortranReducedXBlockFactorBuildContext",
        "FortranReducedXBlockFactorBuildResult",
        "FortranReducedXBlockFactorPolicySetup",
        "FortranReducedXBlockFinalPayloadContext",
        "FortranReducedXBlockGlobalCouplingStageContext",
        "FortranReducedXBlockGlobalCouplingStageResult",
        "FortranReducedXBlockInitialSeedPolicySetup",
        "FortranReducedXBlockInitialSeedResult",
        "FortranReducedXBlockKrylovPolicySetup",
        "FortranReducedXBlockKrylovSetupContext",
        "FortranReducedXBlockKrylovSetupResult",
        "FortranReducedXBlockKrylovSolveContext",
        "FortranReducedXBlockMomentSchurStageContext",
        "FortranReducedXBlockMomentSchurStageResult",
        "apply_fortran_reduced_xblock_global_coupling_stage",
        "apply_fortran_reduced_xblock_initial_seed",
        "apply_fortran_reduced_xblock_moment_schur_stage",
        "build_fortran_reduced_xblock_factor_stage",
        "build_fortran_reduced_xblock_krylov_setup",
        "fortran_reduced_xblock_final_payload",
        "fortran_reduced_xblock_final_payload_from_solve_state",
        "prepare_fortran_reduced_xblock_initial_guess",
        "resolve_fortran_reduced_sparse_pc_backend",
        "resolve_fortran_reduced_xblock_factor_policy",
        "resolve_fortran_reduced_xblock_global_coupling_policy",
        "resolve_fortran_reduced_xblock_initial_seed_policy",
        "resolve_fortran_reduced_xblock_krylov_policy",
        "resolve_fortran_reduced_xblock_moment_schur_policy",
        "run_fortran_reduced_xblock_krylov_solve",
    )
    for name in moved_names:
        assert hasattr(sparse_fortran_reduced_module, name)
        assert name in sparse_fortran_reduced_module.__all__


def _identity(v: jnp.ndarray) -> jnp.ndarray:
    return v


def _unused_solver(**_kwargs: object) -> object:
    raise AssertionError("unexpected solver call")


def _device_result(
    *,
    x: list[float],
    residual_norm: float,
    history: list[float],
    n_iterations: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=jnp.asarray(x, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        residual_history=jnp.asarray(history, dtype=jnp.float64),
        n_iterations=jnp.asarray(n_iterations),
    )


def _first_krylov_context(**overrides: object) -> XBlockFirstKrylovAttemptContext:
    values: dict[str, object] = {
        "krylov_method": "gmres",
        "matvec": _identity,
        "rhs": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "preconditioner": None,
        "x0": None,
        "tol": 1e-8,
        "atol": 1e-12,
        "restart": 5,
        "maxiter": 7,
        "precondition_side": "right",
        "lgmres_outer_k": None,
        "fgmres_block_between_cycles": False,
        "skip_inactive_work": True,
        "device_fgmres_jit": False,
        "device_fgmres_jit_mode": "cycle",
        "device_fgmres_jit_outer_k": 0,
        "augmented_krylov_used": False,
        "augmentation_basis": None,
        "operator_on_augmentation": None,
        "augmentation_mode": "combined",
        "tfqmr_replacement_interval": 0,
        "mv_count": 3,
        "host_progress_callback": None,
        "device_cycle_progress_callback": None,
        "gmres_solver": _unused_solver,
        "lgmres_solver": _unused_solver,
        "gcrotmk_solver": _unused_solver,
        "bicgstab_solver": _unused_solver,
        "fgmres_solver": _unused_solver,
        "fgmres_jit_solver": _unused_solver,
        "fgmres_cycle_jit_solver": _unused_solver,
        "bicgstab_jax_solver": _unused_solver,
        "tfqmr_jax_solver": _unused_solver,
    }
    values.update(overrides)
    return XBlockFirstKrylovAttemptContext(**values)  # type: ignore[arg-type]


def _qi_device_state(*, rank: int = 2) -> SimpleNamespace:
    """Minimal augmented-Krylov state fixture for extracted-QI compatibility."""

    return SimpleNamespace(
        metadata=SimpleNamespace(rank=rank),
        basis=SimpleNamespace(vectors=jnp.eye(2, max(rank, 1), dtype=jnp.float64)[:, :rank]),
        operator_on_basis=jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))[:, :rank],
    )


def _krylov_control_context(
    *,
    env: dict[str, str] | None = None,
    krylov_method: str = "gmres",
    emitted: list[tuple[int, str]] | None = None,
) -> XBlockKrylovControlSetupContext:
    emit_log = emitted if emitted is not None else []
    return XBlockKrylovControlSetupContext(
        env={} if env is None else env,
        krylov_method=krylov_method,
        pc_restart=8,
        pc_maxiter=13,
        precondition_side="right",
        emit=lambda level, message: emit_log.append((level, message)),
    )


def test_resolve_xblock_krylov_control_setup_defaults_emit_solve_start() -> None:
    emitted: list[tuple[int, str]] = []
    setup = resolve_xblock_krylov_control_setup(
        _krylov_control_context(emitted=emitted)
    )

    assert isinstance(setup, XBlockKrylovControlSetup)
    assert setup.fgmres_block_between_cycles is False
    assert setup.tfqmr_replacement_interval == 0
    assert setup.device_fgmres_jit is False
    assert setup.device_fgmres_jit_mode == "cycle"
    assert setup.device_fgmres_jit_outer_k == 0
    assert len(emitted) == 1
    assert "solve start method=gmres restart=8 maxiter=13" in emitted[0][1]


def test_resolve_xblock_krylov_control_setup_normalizes_modes_and_emits_device_lines() -> None:
    emitted: list[tuple[int, str]] = []
    setup = resolve_xblock_krylov_control_setup(
        _krylov_control_context(
            krylov_method="fgmres_jax",
            emitted=emitted,
            env={
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES": "1",
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT": "1",
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE": "bad-mode",
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K": "5",
            },
        )
    )

    assert setup.fgmres_block_between_cycles is True
    assert setup.device_fgmres_jit is True
    assert setup.device_fgmres_jit_mode == "cycle"
    assert setup.device_fgmres_jit_outer_k == 5
    messages = [message for _, message in emitted]
    assert any("FGMRES cycle-boundary synchronization enabled" in m for m in messages)
    assert any("JIT-compiled device FGMRES enabled mode=cycle" in m for m in messages)


def test_resolve_xblock_krylov_control_setup_tfqmr_note_and_clamps_interval() -> None:
    emitted: list[tuple[int, str]] = []
    setup = resolve_xblock_krylov_control_setup(
        _krylov_control_context(
            krylov_method="tfqmr_jax",
            emitted=emitted,
            env={
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TFQMR_REPLACE_INTERVAL": "-4",
            },
        )
    )

    assert setup.tfqmr_replacement_interval == 0
    assert "tfqmr_replacement_interval=0" in emitted[0][1]


def test_run_xblock_first_krylov_attempt_dispatches_host_gmres() -> None:
    calls: list[dict[str, object]] = []

    def gmres_solver(**kwargs: object) -> tuple[np.ndarray, float, list[float]]:
        calls.append(kwargs)
        return np.asarray([3.0, 4.0]), 0.25, [1.0, 0.25]

    def progress_callback(iteration: int, residual_norm: float) -> None:
        del iteration, residual_norm

    result = run_xblock_first_krylov_attempt(
        _first_krylov_context(
            krylov_method="unknown_defaults_to_gmres",
            host_progress_callback=progress_callback,
            gmres_solver=gmres_solver,
        )
    )

    assert isinstance(result, XBlockFirstKrylovAttemptResult)
    np.testing.assert_allclose(result.x, np.asarray([3.0, 4.0]))
    assert result.residual_norm == pytest.approx(0.25)
    assert result.history == (1.0, 0.25)
    assert result.device_iterations is None
    assert result.device_estimated_matvecs is None
    assert calls[0]["progress_callback"] is progress_callback


def test_run_xblock_first_krylov_attempt_dispatches_cycle_jit_fgmres() -> None:
    calls: list[dict[str, object]] = []
    basis = jnp.eye(2, dtype=jnp.float64)
    action = 2.0 * basis

    def cycle_solver(**kwargs: object) -> tuple[SimpleNamespace, None]:
        calls.append(kwargs)
        return _device_result(
            x=[5.0, 6.0],
            residual_norm=0.125,
            history=[1.0, 0.5, 0.125],
            n_iterations=2,
        ), None

    result = run_xblock_first_krylov_attempt(
        _first_krylov_context(
            krylov_method="fgmres_jax",
            device_fgmres_jit=True,
            device_fgmres_jit_mode="cycle",
            device_fgmres_jit_outer_k=4,
            augmented_krylov_used=True,
            augmentation_basis=basis,
            operator_on_augmentation=action,
            augmentation_mode="projected",
            mv_count=9,
            fgmres_cycle_jit_solver=cycle_solver,
        )
    )

    np.testing.assert_allclose(result.x, np.asarray([5.0, 6.0]))
    assert result.residual_norm == pytest.approx(0.125)
    assert result.history == (1.0, 0.5, 0.125)
    assert result.device_iterations == 2
    assert result.device_estimated_matvecs == 9
    assert calls[0]["outer_k"] == 4
    assert calls[0]["augmentation_mode"] == "projected"
    assert calls[0]["augmentation_basis"] is basis
    assert calls[0]["operator_on_augmentation"] is action


def test_run_xblock_first_krylov_attempt_dispatches_tfqmr_replacement_interval() -> None:
    calls: list[dict[str, object]] = []

    def tfqmr_solver(**kwargs: object) -> tuple[SimpleNamespace, None]:
        calls.append(kwargs)
        return _device_result(
            x=[7.0, 8.0],
            residual_norm=0.0625,
            history=[1.0, 0.25, 0.0625],
            n_iterations=2,
        ), None

    result = run_xblock_first_krylov_attempt(
        _first_krylov_context(
            krylov_method="tfqmr_jax",
            tfqmr_replacement_interval=5,
            tfqmr_jax_solver=tfqmr_solver,
        )
    )

    np.testing.assert_allclose(result.x, np.asarray([7.0, 8.0]))
    assert result.residual_norm == pytest.approx(0.0625)
    assert result.history == (1.0, 0.25, 0.0625)
    assert result.device_iterations == 2
    assert calls[0]["residual_replacement_interval"] == 5


def _side_probe_controls(
    *,
    enabled: bool = True,
    switch: bool = False,
    lgmres_rescue_enabled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        restart=3,
        maxiter=4,
        should_switch=lambda _ratio: bool(switch),
        lgmres_rescue_enabled=bool(lgmres_rescue_enabled),
        global_coupling_keep_left_ratio=100.0,
        lgmres_rescue_maxiter=11,
        lgmres_rescue_maxiter_capped=True,
        lgmres_rescue_outer_k=7,
    )


def _side_probe_context(
    *,
    controls: object,
    precondition_side: str = "right",
    krylov_method: str = "gmres",
    pc_maxiter: int | None = 9,
    side_env: str = "",
    global_coupling_built: bool = False,
    x0: jnp.ndarray | None = None,
    gmres_solver=None,
    emitted: list[tuple[int, str]] | None = None,
) -> XBlockSideProbeStageContext:
    time_values = iter((0.0, 0.5))
    matvec_values = iter((2, 7))
    emit_log = emitted if emitted is not None else []

    if gmres_solver is None:
        def default_gmres_solver(
            **_kwargs: object,
        ) -> tuple[np.ndarray, float, list[float]]:
            return np.asarray([2.0, -1.0]), 0.25, [1.0, 0.25]

        gmres_solver = default_gmres_solver

    return XBlockSideProbeStageContext(
        controls=controls,
        precondition_side=precondition_side,
        krylov_method=krylov_method,
        pc_maxiter=pc_maxiter,
        side_env=side_env,
        global_coupling_built=global_coupling_built,
        matvec=lambda values: values,
        true_matvec_no_count=lambda values: values,
        rhs=jnp.asarray([1.0, 1.0]),
        rhs_norm=10.0,
        target=1.0,
        preconditioner=lambda values: values,
        x0=x0,
        tol=1.0e-8,
        atol=1.0e-10,
        elapsed_s=lambda: next(time_values),
        matvec_count=lambda: next(matvec_values),
        emit=lambda level, message: emit_log.append((level, message)),
        gmres_solver=gmres_solver,
    )


def test_apply_xblock_side_probe_stage_keeps_side_and_uses_better_seed() -> None:
    emitted: list[tuple[int, str]] = []
    result = apply_xblock_side_probe_stage(
        _side_probe_context(
            controls=_side_probe_controls(switch=False),
            precondition_side="right",
            x0=jnp.asarray([5.0, 5.0]),
            emitted=emitted,
        )
    )

    assert isinstance(result, XBlockSideProbeStageResult)
    assert result.enabled is True
    assert result.used is True
    assert result.switched is False
    assert result.precondition_side == "right"
    assert result.krylov_method == "gmres"
    assert result.iterations == 2
    assert result.matvecs == 5
    assert result.elapsed_s == pytest.approx(0.5)
    assert result.residual_norm == pytest.approx(0.25)
    assert result.residual_ratio == pytest.approx(0.25)
    assert result.seed_used is True
    assert jnp.allclose(result.x0, jnp.asarray([2.0, -1.0]))
    assert "side probe keep side=right->right" in emitted[-1][1]


def test_apply_xblock_side_probe_stage_switches_left_to_right_with_seed() -> None:
    emitted: list[tuple[int, str]] = []
    result = apply_xblock_side_probe_stage(
        _side_probe_context(
            controls=_side_probe_controls(switch=True),
            precondition_side="left",
            x0=None,
            emitted=emitted,
        )
    )

    assert result.precondition_side == "right"
    assert result.switched is True
    assert result.seed_used is True
    assert result.physical_seed_preserved_after_switch is True
    assert jnp.allclose(result.x0, jnp.asarray([2.0, -1.0]))
    assert "side probe switch side=left->right" in emitted[-1][1]
    assert "preserved_physical_seed=1" in emitted[-1][1]


def test_apply_xblock_side_probe_stage_lgmres_rescue_updates_method() -> None:
    emitted: list[tuple[int, str]] = []
    result = apply_xblock_side_probe_stage(
        _side_probe_context(
            controls=_side_probe_controls(
                switch=True,
                lgmres_rescue_enabled=True,
            ),
            precondition_side="left",
            pc_maxiter=5,
            emitted=emitted,
        )
    )

    assert result.precondition_side == "left"
    assert result.krylov_method == "lgmres"
    assert result.pc_maxiter == 11
    assert result.lgmres_rescue is True
    assert result.lgmres_rescue_maxiter_capped is True
    assert result.lgmres_rescue_outer_k == 7
    assert "side probe method_rescue side=left->left" in emitted[-1][1]


def test_apply_xblock_side_probe_stage_reports_solver_failure() -> None:
    emitted: list[tuple[int, str]] = []

    def failing_solver(**_kwargs: object) -> tuple[np.ndarray, float, list[float]]:
        raise RuntimeError("probe failed")

    result = apply_xblock_side_probe_stage(
        _side_probe_context(
            controls=_side_probe_controls(switch=True),
            precondition_side="left",
            gmres_solver=failing_solver,
            emitted=emitted,
        )
    )

    assert result.failed is True
    assert result.failure_reason == "RuntimeError: probe failed"
    assert result.precondition_side == "left"
    assert result.krylov_method == "gmres"
    assert result.selected_side == "left"
    assert result.selected_method == "gmres"
    assert result.elapsed_s == pytest.approx(0.5)
    assert "side probe failed (RuntimeError: probe failed)" in emitted[-1][1]


def _probe_coarse_policy(**overrides: object) -> SimpleNamespace:
    values = {
        "steps_requested": 2,
        "max_directions": 3,
        "max_extra_units": 4,
        "fsavg_lmax": 1,
        "angular_lmax": 2,
        "include_angular_residual": True,
        "include_raw": False,
        "alpha_clip": 10.0,
        "rcond": 1.0e-10,
        "min_improvement": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _probe_coarse_context(
    *,
    policy: object | None = None,
    x0: jnp.ndarray | None = None,
    correction=None,
    emitted: list[tuple[int, str]] | None = None,
) -> XBlockProbeCoarseStageContext:
    time_values = iter((1.0, 1.75))
    emit_log = emitted if emitted is not None else []

    def direction_builder(
        residual_vec: jnp.ndarray,
        **kwargs: object,
    ) -> tuple[tuple[str, jnp.ndarray], ...]:
        assert "max_directions" in kwargs
        assert residual_vec.shape == (2,)
        return (("raw", jnp.asarray([1.0, 0.0])),)

    if correction is None:
        def correction_default(**kwargs: object):
            directions = kwargs["direction_builder"](jnp.asarray([1.0, 0.0]))
            assert directions[0][0] == "raw"
            return (
                jnp.asarray([0.9, 0.0]),
                jnp.asarray([0.1, 0.0]),
                (1.0, 0.1),
                (len(directions),),
                tuple(name for name, _ in directions),
            )

        correction = correction_default

    return XBlockProbeCoarseStageContext(
        policy=_probe_coarse_policy() if policy is None else policy,
        rhs=jnp.asarray([1.0, 0.0]),
        x0=x0,
        matvec=lambda values: values,
        target=1.0e-3,
        direction_builder=direction_builder,
        correction=correction,
        elapsed_s=lambda: next(time_values),
        emit=lambda level, message: emit_log.append((level, message)),
    )


def test_apply_xblock_probe_coarse_stage_disabled_keeps_seed() -> None:
    result = apply_xblock_probe_coarse_stage(
        _probe_coarse_context(
            policy=_probe_coarse_policy(steps_requested=0),
            x0=jnp.asarray([0.25, 0.5]),
        )
    )

    assert isinstance(result, XBlockProbeCoarseStageResult)
    assert result.steps_requested == 0
    assert result.elapsed_s == pytest.approx(0.0)
    assert result.seed_initialized is False
    assert result.improved is False
    assert jnp.allclose(result.x0, jnp.asarray([0.25, 0.5]))


def test_apply_xblock_probe_coarse_stage_initializes_and_accepts_seed() -> None:
    emitted: list[tuple[int, str]] = []
    result = apply_xblock_probe_coarse_stage(
        _probe_coarse_context(emitted=emitted)
    )

    assert result.seed_initialized is True
    assert result.improved is True
    assert result.failed is False
    assert result.residual_before == pytest.approx(1.0)
    assert result.residual_after == pytest.approx(0.1)
    assert result.history == (1.0, 0.1)
    assert result.direction_counts == (1,)
    assert result.direction_names == ("raw",)
    assert result.elapsed_s == pytest.approx(0.75)
    assert result.include_angular_residual is True
    assert result.include_raw is False
    assert jnp.allclose(result.x0, jnp.asarray([0.9, 0.0]))
    assert "probe-coarse improved seed residual" in emitted[-1][1]


def test_apply_xblock_probe_coarse_stage_rejects_non_improving_seed() -> None:
    emitted: list[tuple[int, str]] = []

    def correction_reject(**_kwargs: object):
        return (
            jnp.asarray([9.0, 9.0]),
            jnp.asarray([2.0, 0.0]),
            (1.0, 2.0),
            (1,),
            ("raw",),
        )

    result = apply_xblock_probe_coarse_stage(
        _probe_coarse_context(
            x0=jnp.asarray([0.0, 0.0]),
            correction=correction_reject,
            emitted=emitted,
        )
    )

    assert result.seed_initialized is False
    assert result.improved is False
    assert result.residual_before == pytest.approx(1.0)
    assert result.residual_after == pytest.approx(2.0)
    assert jnp.allclose(result.x0, jnp.asarray([0.0, 0.0]))
    assert "probe-coarse rejected seed residual" in emitted[-1][1]


def test_apply_xblock_probe_coarse_stage_reports_failure() -> None:
    emitted: list[tuple[int, str]] = []

    def correction_failure(**_kwargs: object):
        raise RuntimeError("coarse failed")

    result = apply_xblock_probe_coarse_stage(
        _probe_coarse_context(
            x0=jnp.asarray([0.0, 0.0]),
            correction=correction_failure,
            emitted=emitted,
        )
    )

    assert result.failed is True
    assert result.failure_reason == "RuntimeError: coarse failed"
    assert result.improved is False
    assert result.residual_before == pytest.approx(1.0)
    assert result.residual_after is None
    assert "probe-coarse failed (RuntimeError: coarse failed)" in emitted[-1][1]


def _preflight_context(
    *,
    min_improvement: float = 0.0,
    required: bool = False,
    rhs: jnp.ndarray | None = None,
    rhs_norm: float = 10.0,
    x0: jnp.ndarray | None = None,
    matvec=None,
    target: float = 1.0e-3,
    emitted: list[tuple[int, str]] | None = None,
) -> XBlockPreflightGateContext:
    emit_log = emitted if emitted is not None else []
    if rhs is None:
        rhs = jnp.asarray([1.0, 0.0])
    if matvec is None:
        def identity_matvec(values: jnp.ndarray) -> jnp.ndarray:
            return values

        matvec = identity_matvec
    return XBlockPreflightGateContext(
        min_improvement=float(min_improvement),
        required=bool(required),
        rhs=rhs,
        rhs_norm=float(rhs_norm),
        x0=x0,
        matvec=matvec,
        target=float(target),
        emit=lambda level, message: emit_log.append((level, message)),
    )


def test_evaluate_xblock_preflight_gate_inactive_skips_evaluation() -> None:
    result = evaluate_xblock_preflight_gate(_preflight_context())

    assert result.evaluated is False
    assert result.passed is None
    assert result.improvement is None
    assert result.residual_norm is None


def test_evaluate_xblock_preflight_gate_passes_on_residual_target() -> None:
    emitted: list[tuple[int, str]] = []
    result = evaluate_xblock_preflight_gate(
        _preflight_context(
            min_improvement=0.95,
            x0=jnp.asarray([1.0, 0.0]),
            target=1.0e-2,
            emitted=emitted,
        )
    )

    assert result.evaluated is True
    assert result.failed is False
    assert result.passed is True
    assert result.residual_norm == pytest.approx(0.0)
    assert result.improvement == pytest.approx(1.0)
    assert "preflight residual=" in emitted[-1][1]
    assert "passed=1" in emitted[-1][1]


def test_evaluate_xblock_preflight_gate_passes_on_improvement() -> None:
    emitted: list[tuple[int, str]] = []
    result = evaluate_xblock_preflight_gate(
        _preflight_context(
            min_improvement=0.2,
            rhs=jnp.asarray([10.0, 0.0]),
            rhs_norm=10.0,
            x0=jnp.asarray([8.0, 0.0]),
            target=1.0e-6,
            emitted=emitted,
        )
    )

    assert result.passed is True
    assert result.residual_norm == pytest.approx(2.0)
    assert result.improvement == pytest.approx(0.8)
    assert "passed=1" in emitted[-1][1]


def test_evaluate_xblock_preflight_gate_nonrequired_failure_emits_warning() -> None:
    emitted: list[tuple[int, str]] = []
    result = evaluate_xblock_preflight_gate(
        _preflight_context(
            min_improvement=0.9,
            x0=jnp.asarray([0.0, 0.0]),
            matvec=lambda _values: (_ for _ in ()).throw(RuntimeError("bad matvec")),
            emitted=emitted,
        )
    )

    assert result.evaluated is True
    assert result.failed is True
    assert result.failure_reason == "RuntimeError: bad matvec"
    assert result.passed is None
    assert "preflight failed (RuntimeError: bad matvec)" in emitted[-1][1]


def test_evaluate_xblock_preflight_gate_required_failure_raises() -> None:
    emitted: list[tuple[int, str]] = []
    with pytest.raises(RuntimeError, match="preflight gate failed"):
        evaluate_xblock_preflight_gate(
            _preflight_context(
                min_improvement=0.9,
                required=True,
                rhs=jnp.asarray([10.0, 0.0]),
                rhs_norm=10.0,
                x0=jnp.asarray([0.0, 0.0]),
                emitted=emitted,
            )
        )

    assert "passed=0" in emitted[-1][1]


def test_evaluate_xblock_preflight_gate_required_missing_seed_raises() -> None:
    with pytest.raises(RuntimeError, match="required an initial seed"):
        evaluate_xblock_preflight_gate(
            _preflight_context(min_improvement=0.0, required=True, x0=None)
        )


@pytest.mark.parametrize("residual_norm", [1.1, np.nan])
def test_xblock_gmres_fallback_decision_retries_failed_non_gmres(
    residual_norm: float,
) -> None:
    assert xblock_gmres_fallback_decision(
        krylov_method="lgmres",
        fallback_enabled=True,
        residual_norm=residual_norm,
        target=1.0,
    ) == XBlockGMRESFallbackDecision(run=True)


def test_xblock_gmres_fallback_decision_skips_gmres_method() -> None:
    assert not xblock_gmres_fallback_decision(
        krylov_method="gmres",
        fallback_enabled=True,
        residual_norm=2.0,
        target=1.0,
    ).run


def test_xblock_gmres_fallback_decision_skips_disabled_policy() -> None:
    assert not xblock_gmres_fallback_decision(
        krylov_method="bicgstab",
        fallback_enabled=False,
        residual_norm=2.0,
        target=1.0,
    ).run


def test_xblock_gmres_fallback_decision_skips_converged_non_gmres() -> None:
    assert not xblock_gmres_fallback_decision(
        krylov_method="tfqmr",
        fallback_enabled=True,
        residual_norm=0.5,
        target=1.0,
    ).run


def test_run_xblock_gmres_fallback_if_needed_preserves_converged_candidate() -> None:
    def fail_gmres(**_kwargs):
        raise AssertionError("gmres fallback should not run")

    def fail_initial_guess(**_kwargs):
        raise AssertionError("fallback seed should not run")

    result = run_xblock_gmres_fallback_if_needed(
        XBlockGMRESFallbackContext(
            krylov_method="tfqmr",
            fallback_enabled=True,
            x_solution=np.asarray([1.0, 2.0]),
            x_physical=np.asarray([3.0, 4.0]),
            residual_norm=0.25,
            history=(1.0, 0.25),
            solve_s=5.0,
            target=1.0,
            rhs_norm=10.0,
            original_x0=jnp.zeros(2),
            solve_rhs=jnp.zeros(2),
            solve_matvec=_identity,
            solve_preconditioner=_identity,
            precondition_side="right",
            tol=1.0e-8,
            atol=1.0e-12,
            restart=10,
            maxiter=20,
            progress_callback=None,
            emit=None,
            elapsed_s=lambda: 99.0,
            gmres_solver=fail_gmres,
            initial_guess_builder=fail_initial_guess,
            solution_to_physical=_identity,
            physical_rhs=jnp.zeros(2),
            physical_matvec=_identity,
            device_iterations=7,
            device_estimated_matvecs=11,
        )
    )

    assert isinstance(result, XBlockGMRESFallbackResult)
    assert result.krylov_method == "tfqmr"
    np.testing.assert_allclose(result.x_solution, np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(result.x_physical, np.asarray([3.0, 4.0]))
    assert result.residual_norm == pytest.approx(0.25)
    assert result.history == (1.0, 0.25)
    assert result.solve_s == pytest.approx(5.0)
    assert result.device_iterations == 7
    assert result.device_estimated_matvecs == 11
    assert result.fallback_started_from_candidate is False
    assert result.fallback_candidate_improved_rhs is False


def test_run_xblock_gmres_fallback_if_needed_retries_and_recomputes_residual() -> None:
    emitted: list[tuple[int, str]] = []
    calls: dict[str, object] = {}
    times = iter((4.0, 4.25))

    def initial_guess_builder(**kwargs):
        calls["initial_candidate"] = np.asarray(kwargs["candidate"], dtype=np.float64)
        calls["initial_rhs_shape"] = kwargs["rhs_shape"]
        calls["initial_residual"] = kwargs["candidate_residual_norm"]
        calls["initial_rhs_norm"] = kwargs["rhs_norm"]
        calls["initial_side"] = kwargs["precondition_side"]
        return jnp.asarray([0.1, 0.2]), True, False

    def gmres_solver(**kwargs):
        calls["gmres_x0"] = np.asarray(kwargs["x0"], dtype=np.float64)
        calls["gmres_restart"] = kwargs["restart"]
        calls["gmres_maxiter"] = kwargs["maxiter"]
        calls["gmres_side"] = kwargs["precondition_side"]
        assert kwargs["progress_callback"] is None
        return np.asarray([3.0, 4.0]), 99.0, (2.0, 0.5)

    result = run_xblock_gmres_fallback_if_needed(
        XBlockGMRESFallbackContext(
            krylov_method="bicgstab",
            fallback_enabled=True,
            x_solution=np.asarray([1.0, 2.0]),
            x_physical=np.asarray([1.0, 2.0]),
            residual_norm=2.0,
            history=(3.0, 2.0),
            solve_s=5.0,
            target=1.0,
            rhs_norm=10.0,
            original_x0=jnp.zeros(2),
            solve_rhs=jnp.asarray([1.0, 2.0]),
            solve_matvec=_identity,
            solve_preconditioner=_identity,
            precondition_side="left",
            tol=1.0e-8,
            atol=1.0e-12,
            restart=8,
            maxiter=13,
            progress_callback=None,
            emit=lambda level, message: emitted.append((level, message)),
            elapsed_s=lambda: next(times),
            gmres_solver=gmres_solver,
            initial_guess_builder=initial_guess_builder,
            solution_to_physical=lambda value: 2.0 * value,
            physical_rhs=jnp.asarray([6.0, 8.0]),
            physical_matvec=_identity,
            device_iterations=5,
            device_estimated_matvecs=9,
        )
    )

    assert result.krylov_method == "gmres"
    np.testing.assert_allclose(result.x_solution, np.asarray([3.0, 4.0]))
    np.testing.assert_allclose(result.x_physical, np.asarray([6.0, 8.0]))
    assert result.residual_norm == pytest.approx(0.0)
    assert result.history == (2.0, 0.5)
    assert result.solve_s == pytest.approx(5.25)
    assert result.device_iterations is None
    assert result.device_estimated_matvecs is None
    assert result.fallback_started_from_candidate is True
    assert result.fallback_candidate_improved_rhs is False
    assert emitted == [
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "bicgstab residual=2.000000e+00 > target=1.000000e+00; "
            "falling back to gmres",
        )
    ]
    np.testing.assert_allclose(calls["initial_candidate"], np.asarray([1.0, 2.0]))
    assert calls["initial_rhs_shape"] == (2,)
    assert calls["initial_residual"] == pytest.approx(2.0)
    assert calls["initial_rhs_norm"] == pytest.approx(10.0)
    assert calls["initial_side"] == "left"
    np.testing.assert_allclose(calls["gmres_x0"], np.asarray([0.1, 0.2]))
    assert calls["gmres_restart"] == 8
    assert calls["gmres_maxiter"] == 13
    assert calls["gmres_side"] == "left"


def test_xblock_sparse_pc_work_estimates_report_gmres_metadata() -> None:
    result = xblock_sparse_pc_work_estimates(
        krylov_method="gmres",
        linear_size=100,
        restart=7,
        dtype=np.float64,
    )

    assert result == XBlockSparsePCWorkEstimates(
        solver_kind="xblock_sparse_pc_gmres",
        device_krylov_methods=frozenset(
            {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
        ),
        gmres_basis_nbytes=100 * (7 + 1 + 4) * 8,
        bicgstab_work_nbytes=100 * 8 * 8,
        tfqmr_work_nbytes=100 * 10 * 8,
    )


def test_xblock_sparse_pc_work_estimates_report_non_gmres_solver_kind() -> None:
    result = xblock_sparse_pc_work_estimates(
        krylov_method="tfqmr_jax",
        linear_size=10,
        restart=3,
        dtype=np.float32,
    )

    assert result.solver_kind == "xblock_sparse_pc_tfqmr_jax"
    assert result.gmres_basis_nbytes == 10 * (3 + 1 + 4) * 4
    assert result.bicgstab_work_nbytes == 10 * 8 * 4
    assert result.tfqmr_work_nbytes == 10 * 10 * 4


def test_xblock_sparse_pc_completion_message_includes_ksp_residual() -> None:
    assert xblock_sparse_pc_completion_message(
        krylov_method="tfqmr_jax",
        elapsed_s=12.34567,
        iterations=9,
        matvecs=31,
        residual_norm=2.0e-8,
        target=1.0e-9,
        history=(1.0, 3.0e-7),
    ) == (
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
        "method=tfqmr_jax elapsed_s=12.346 iters=9 matvecs=31 "
        "residual=2.000000e-08 target=1.000000e-09 ksp_residual=3.000000e-07"
    )


def test_xblock_sparse_pc_completion_message_omits_empty_ksp_residual() -> None:
    assert xblock_sparse_pc_completion_message(
        krylov_method="gmres",
        elapsed_s=1.0,
        iterations=2,
        matvecs=3,
        residual_norm=4.0,
        target=5.0,
        history=(),
    ) == (
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
        "method=gmres elapsed_s=1.000 iters=2 matvecs=3 "
        "residual=4.000000e+00 target=5.000000e+00"
    )


def test_emit_xblock_sparse_pc_completion_uses_explicit_context() -> None:
    emitted: list[tuple[int, str]] = []

    emit_xblock_sparse_pc_completion(
        XBlockSparsePCCompletionContext(
            emit=lambda level, message: emitted.append((level, message)),
            krylov_method="gmres_jax",
            elapsed_s=3.5,
            iterations=6,
            matvecs=14,
            residual_norm=2.0e-9,
            target=1.0e-8,
            history=(1.0e-3, 2.0e-9),
        )
    )

    assert emitted == [
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
            "method=gmres_jax elapsed_s=3.500 iters=6 matvecs=14 "
            "residual=2.000000e-09 target=1.000000e-08 ksp_residual=2.000000e-09",
        )
    ]


def test_emit_xblock_sparse_pc_completion_skips_missing_emit() -> None:
    emit_xblock_sparse_pc_completion(
        XBlockSparsePCCompletionContext(
            emit=None,
            krylov_method="gmres",
            elapsed_s=0.0,
            iterations=0,
            matvecs=0,
            residual_norm=0.0,
            target=1.0,
            history=(),
        )
    )


def test_emit_xblock_sparse_pc_completion_from_solve_state_emits_message() -> None:
    emitted: list[tuple[int, str]] = []

    emit_xblock_sparse_pc_completion_from_solve_state(
        {
            "emit": lambda level, message: emitted.append((level, message)),
            "xblock_krylov_method": "bicgstab",
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.25),
            "reported_iterations": np.int64(4),
            "reported_matvecs": np.int64(12),
            "residual_norm_xblock_pc": 6.0e-7,
            "target_xblock": 1.0e-8,
            "history": (2.0, 1.0e-6),
        }
    )

    assert emitted == [
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
            "method=bicgstab elapsed_s=2.250 iters=4 matvecs=12 "
            "residual=6.000000e-07 target=1.000000e-08 ksp_residual=1.000000e-06",
        )
    ]


def test_emit_xblock_sparse_pc_completion_from_solve_state_skips_missing_emit() -> None:
    emit_xblock_sparse_pc_completion_from_solve_state({"emit": None})


def _xblock_post_policy(
    *,
    post_minres_steps: int,
    post_coarse_steps: int,
    post_residual_steps: int,
    include_post_coarse: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        post_minres=SimpleNamespace(
            steps_requested=post_minres_steps,
            alpha_clip=4.0,
            min_improvement=0.0,
        ),
        post_coarse=SimpleNamespace(
            steps_requested=post_coarse_steps,
            max_directions=5,
            max_extra_units=2,
            fsavg_lmax=1,
            angular_lmax=3,
            include_angular_residual=True,
            include_raw=False,
            alpha_clip=6.0,
            rcond=1.0e-10,
            min_improvement=0.0,
        ),
        post_residual_equation=SimpleNamespace(
            steps_requested=post_residual_steps,
            max_directions=7,
            max_extra_units=4,
            fsavg_lmax=2,
            angular_lmax=5,
            include_angular_residual=False,
            include_raw=True,
            include_post_coarse=include_post_coarse,
            alpha_clip=8.0,
            rcond=1.0e-9,
            min_improvement=0.0,
        ),
    )


def test_run_xblock_post_solve_corrections_applies_ordered_stages() -> None:
    calls: list[str] = []
    direction_kwargs: list[dict[str, object]] = []
    messages: list[tuple[int, str]] = []
    clock = {"t": 0.0}

    def elapsed_s() -> float:
        clock["t"] += 0.5
        return clock["t"]

    def coarse_direction_builder(residual_vec: jnp.ndarray, **kwargs):
        direction_kwargs.append(dict(kwargs))
        return (("coarse", residual_vec),)

    def residual_equation_correction(**kwargs):
        calls.append("post_residual_equation")
        directions = kwargs["direction_builder"](jnp.asarray([1.0, 0.0]))
        assert directions[0][0] == "coarse"
        return (
            jnp.asarray([0.8, 0.0]),
            jnp.asarray([0.4, 0.0]),
            (1.0, 0.4),
            (1,),
            ("residual-space",),
        )

    def minres_correction(**_kwargs):
        calls.append("post_minres")
        return (
            jnp.asarray([0.6, 0.0]),
            jnp.asarray([0.2, 0.0]),
            (0.4, 0.2),
            (0.5,),
        )

    def coarse_correction(**kwargs):
        calls.append("post_coarse")
        directions = kwargs["direction_builder"](jnp.asarray([1.0, 0.0]))
        assert directions[0][0] == "coarse"
        return (
            jnp.asarray([0.5, 0.0]),
            jnp.asarray([0.1, 0.0]),
            (0.2, 0.1),
            (2,),
            ("coarse-space",),
        )

    result = run_xblock_post_solve_corrections(
        XBlockPostSolveCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.asarray([1.0, 0.0]),
            residual_norm=1.0,
            target=1.0e-6,
            solve_s=10.0,
            preconditioner=_identity,
            precondition_side="right",
            post_solve_policy=_xblock_post_policy(
                post_minres_steps=1,
                post_coarse_steps=1,
                post_residual_steps=1,
            ),
            coarse_direction_builder=coarse_direction_builder,
            emit=lambda level, message: messages.append((level, message)),
            elapsed_s=elapsed_s,
            minres_correction=minres_correction,
            residual_equation_correction=residual_equation_correction,
            coarse_correction=coarse_correction,
        )
    )

    assert isinstance(result, XBlockPostSolveCorrectionResult)
    assert calls == ["post_residual_equation", "post_minres", "post_coarse"]
    np.testing.assert_allclose(result.x, np.asarray([0.5, 0.0]))
    assert result.residual_norm == pytest.approx(0.1)
    assert result.solve_s == pytest.approx(11.5)
    assert result.post_residual_equation_history == (1.0, 0.4)
    assert result.post_minres_history == (0.4, 0.2)
    assert result.post_minres_alphas == (0.5,)
    assert result.post_coarse_direction_counts == (2,)
    assert result.post_coarse_direction_names == ("coarse-space",)
    assert direction_kwargs[0]["include_raw"] is True
    assert direction_kwargs[0]["max_directions"] == 7
    assert direction_kwargs[1]["include_raw"] is False
    assert direction_kwargs[1]["max_directions"] == 5
    assert any("post-residual-equation improved" in message for _, message in messages)
    assert any("post-minres improved" in message for _, message in messages)
    assert any("post-coarse improved" in message for _, message in messages)

    state = result.metadata_state()
    assert state["post_residual_equation_direction_counts"] == (1,)
    assert state["post_minres_alphas"] == (0.5,)
    assert state["post_coarse_direction_names"] == ("coarse-space",)


def test_run_xblock_post_solve_corrections_preserves_state_when_inactive() -> None:
    def fail_correction(**_kwargs):
        raise AssertionError("correction should not run")

    result = run_xblock_post_solve_corrections(
        XBlockPostSolveCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.asarray([1.0, 2.0]),
            residual_norm=0.25,
            target=1.0,
            solve_s=3.0,
            preconditioner=_identity,
            precondition_side="none",
            post_solve_policy=_xblock_post_policy(
                post_minres_steps=0,
                post_coarse_steps=0,
                post_residual_steps=0,
                include_post_coarse=False,
            ),
            coarse_direction_builder=lambda residual_vec, **_kwargs: (("raw", residual_vec),),
            emit=None,
            elapsed_s=lambda: 0.0,
            minres_correction=fail_correction,
            residual_equation_correction=fail_correction,
            coarse_correction=fail_correction,
        )
    )

    np.testing.assert_allclose(result.x, np.asarray([1.0, 2.0]))
    assert result.residual_norm == pytest.approx(0.25)
    assert result.solve_s == pytest.approx(3.0)
    assert result.post_minres_history == ()
    assert result.post_coarse_direction_counts == ()
    assert result.post_residual_equation_direction_counts == ()
    assert result.post_residual_equation_include_post_coarse is False


def test_complete_xblock_post_krylov_stage_emits_completion_and_returns_state() -> None:
    emitted: list[tuple[int, str]] = []

    def fail_correction(**_kwargs):
        raise AssertionError("correction should not run")

    result = complete_xblock_post_krylov_stage(
        XBlockPostKrylovCompletionContext(
            corrections=XBlockPostSolveCorrectionContext(
                matvec=_identity,
                rhs=jnp.zeros(2),
                x=np.asarray([1.0, 2.0]),
                residual_norm=0.25,
                target=1.0,
                solve_s=3.0,
                preconditioner=_identity,
                precondition_side="none",
                post_solve_policy=_xblock_post_policy(
                    post_minres_steps=0,
                    post_coarse_steps=0,
                    post_residual_steps=0,
                    include_post_coarse=False,
                ),
                coarse_direction_builder=lambda residual_vec, **_kwargs: (
                    ("raw", residual_vec),
                ),
                emit=lambda level, message: emitted.append((level, message)),
                elapsed_s=lambda: 2.25,
                minres_correction=fail_correction,
                residual_equation_correction=fail_correction,
                coarse_correction=fail_correction,
            ),
            krylov_method="gmres",
            elapsed_s=lambda: 9.0,
            iterations=7,
            matvecs=11,
            target=1.0,
            history=(0.5, 0.25),
        )
    )

    assert isinstance(result, XBlockPostKrylovCompletionResult)
    np.testing.assert_allclose(result.x, np.asarray([1.0, 2.0]))
    assert result.residual_norm == pytest.approx(0.25)
    assert result.solve_s == pytest.approx(3.0)
    assert result.corrections.post_minres_history == ()
    assert emitted == [
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
            "method=gmres elapsed_s=9.000 iters=7 matvecs=11 "
            "residual=2.500000e-01 target=1.000000e+00 "
            "ksp_residual=2.500000e-01",
        )
    ]


def test_xblock_physical_solution_and_residual_measures_true_residual() -> None:
    result = xblock_physical_solution_and_residual(
        x=np.asarray([1.0, 2.0]),
        solution_to_physical=lambda value: 2.0 * value,
        rhs=jnp.asarray([2.0, 4.0]),
        matvec=lambda value: value,
        fallback_residual_norm=99.0,
    )

    assert isinstance(result, XBlockPhysicalResidual)
    np.testing.assert_allclose(result.x_physical, np.asarray([2.0, 4.0]))
    assert result.residual_norm == 0.0


def test_xblock_physical_solution_and_residual_keeps_fallback_on_matvec_error() -> None:
    def _raise(_value):
        raise RuntimeError("boom")

    result = xblock_physical_solution_and_residual(
        x=np.asarray([1.0, 2.0]),
        solution_to_physical=lambda value: 2.0 * value,
        rhs=jnp.asarray([2.0, 4.0]),
        matvec=_raise,
        fallback_residual_norm=99.0,
    )

    np.testing.assert_allclose(result.x_physical, np.asarray([2.0, 4.0]))
    assert result.residual_norm == 99.0


class _DefaultSparsePCDriverState(dict):
    def __missing__(self, key: str) -> object:
        self[key] = 1
        return self[key]


def _op(
    *, fp=False, pas=False, constraint_scheme=1, n_zeta=1, n_species=1
) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        include_phi1=False,
        n_zeta=n_zeta,
        n_species=n_species,
        point_at_x0=False,
        fblock=SimpleNamespace(
            fp=object() if fp else None,
            pas=object() if pas else None,
        ),
    )


def test_sparse_pc_active_dof_setup_disabled_uses_full_system_vectors() -> None:
    rhs = jnp.arange(6.0)
    setup = build_sparse_pc_active_dof_setup(
        op=SimpleNamespace(total_size=6),
        rhs=rhs,
        sparse_pc_use_active_dof=False,
        active_dof_indices=lambda _op: np.asarray([0, 2, 5]),
        reduce_full_with_indices=reduce_full_with_indices,
        expand_reduced_with_map=expand_reduced_with_map,
    )

    assert setup.active_idx_np is None
    assert setup.active_idx_jnp is None
    assert setup.full_to_active_jnp is None
    assert setup.linear_size == 6
    assert setup.messages == ()
    np.testing.assert_allclose(np.asarray(setup.rhs), np.asarray(rhs))
    np.testing.assert_allclose(np.asarray(setup.reduce_full(rhs + 10)), np.arange(6) + 10)
    np.testing.assert_allclose(np.asarray(setup.expand_reduced(rhs + 20)), np.arange(6) + 20)


def test_sparse_pc_active_dof_setup_builds_reduction_maps_and_message() -> None:
    rhs = jnp.arange(6.0)
    setup = build_sparse_pc_active_dof_setup(
        op=SimpleNamespace(total_size=6),
        rhs=rhs,
        sparse_pc_use_active_dof=True,
        active_dof_indices=lambda _op: np.asarray([0, 2, 5], dtype=np.int64),
        reduce_full_with_indices=reduce_full_with_indices,
        expand_reduced_with_map=expand_reduced_with_map,
    )

    assert setup.linear_size == 3
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres active-DOF reduction "
            "enabled (size=3/6)",
        ),
    )
    np.testing.assert_array_equal(setup.active_idx_np, np.asarray([0, 2, 5]))
    np.testing.assert_array_equal(np.asarray(setup.active_idx_jnp), np.asarray([0, 2, 5]))
    np.testing.assert_array_equal(
        np.asarray(setup.full_to_active_jnp),
        np.asarray([1, 0, 2, 0, 0, 3], dtype=np.int32),
    )
    np.testing.assert_allclose(np.asarray(setup.rhs), np.asarray([0.0, 2.0, 5.0]))
    np.testing.assert_allclose(
        np.asarray(setup.reduce_full(jnp.arange(6.0) + 10.0)),
        np.asarray([10.0, 12.0, 15.0]),
    )
    np.testing.assert_allclose(
        np.asarray(setup.expand_reduced(jnp.asarray([1.0, 2.0, 3.0]))),
        np.asarray([1.0, 0.0, 2.0, 0.0, 0.0, 3.0]),
    )


@pytest.mark.parametrize(
    ("fortran_reduced", "active", "expected_scope", "expected_pattern"),
    (
        (False, False, "full", "generic_full"),
        (False, True, "active_dof", "generic_active"),
        (True, False, "fortran_reduced_full", "fortran_full"),
        (True, True, "fortran_reduced_active_dof", "fortran_active"),
    ),
)
def test_sparse_pc_pattern_setup_selects_scope_and_preserves_callbacks(
    fortran_reduced: bool,
    active: bool,
    expected_scope: str,
    expected_pattern: str,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    messages: list[tuple[int, str]] = []
    elapsed_values = iter((10.0, 10.25))

    def _record(name: str, pattern: str):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return pattern

        return _inner

    result = build_sparse_pc_pattern_setup(
        SparsePCPatternSetupContext(
            op=SimpleNamespace(total_size=4),
            pattern_source_op="source-op",
            fortran_reduced_sparse_pc=fortran_reduced,
            sparse_pc_use_active_dof=active,
            active_idx_np=np.asarray([3, 1], dtype=np.int64) if active else None,
            preconditioner_x=2,
            preconditioner_xi=3,
            preconditioner_species=4,
            preconditioner_x_min_l=5,
            fp_dense_velocity_block=True,
            elapsed_s=lambda: next(elapsed_values),
            emit=lambda level, message: messages.append((level, message)),
            fortran_reduced_pattern_for_indices=_record(
                "fortran_active",
                "fortran_active",
            ),
            fortran_reduced_pattern=_record("fortran_full", "fortran_full"),
            conservative_pattern_for_indices=_record(
                "generic_active",
                "generic_active",
            ),
            conservative_pattern=_record("generic_full", "generic_full"),
            summarize_pattern=lambda _op, pattern: SimpleNamespace(
                nnz=7,
                avg_row_nnz=1.75,
                max_row_nnz=3,
                pattern=pattern,
            ),
        )
    )

    assert result.pattern == expected_pattern
    assert result.scope == expected_scope
    assert result.build_s == 0.25
    assert result.summary.nnz == 7
    assert calls[0][0] == expected_pattern
    if active:
        np.testing.assert_array_equal(calls[0][1][1], np.asarray([3, 1], dtype=np.int32))
    if fortran_reduced:
        assert calls[0][2] == {
            "preconditioner_x": 2,
            "preconditioner_xi": 3,
            "preconditioner_species": 4,
            "preconditioner_x_min_l": 5,
        }
    else:
        assert calls[0][2] == {"fp_dense_velocity_block": True}
    assert messages[0] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres building conservative pattern",
    )
    assert messages[1] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres pattern "
        f"scope={expected_scope} nnz=7 avg_row_nnz=1.75 max_row_nnz=3",
    )


def test_fortran_reduced_backend_policy_honors_explicit_backend_alias() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "local-xblock"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "xblock"
    assert setup.reason == "env"
    assert setup.backend_raw == "local_xblock"
    assert setup.xblock_min_size == 100000
    assert setup.messages == ()


def test_fortran_reduced_backend_policy_auto_selects_large_full_fp_xblock() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE": "10"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "xblock"
    assert setup.reason == "auto_large_full_fp_size>=10"
    assert not setup.backend_ignored_env


def test_fortran_reduced_backend_policy_direct_tail_required_forces_global() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3),
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER": (
                "active-fortran-v3-reduced-lu"
            ),
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED": "1",
        },
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "global"
    assert setup.reason == "required_direct_tail_structured_pc"
    assert setup.direct_tail_pc_env == "active_fortran_v3_reduced_lu"
    assert setup.direct_tail_pc_explicit
    assert setup.direct_tail_structured_pc_required
    assert setup.direct_tail_structured_pc_forces_global


def test_fortran_reduced_backend_policy_ignored_env_reports_message() -> None:
    setup = resolve_fortran_reduced_sparse_pc_backend(
        op=_op(fp=False, constraint_scheme=1, n_zeta=1),
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND": "unknown-backend"},
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=12,
    )

    assert setup.backend == "global"
    assert setup.reason == "auto_global"
    assert setup.backend_ignored_env
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: ignoring unknown "
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND="
            "'unknown_backend'; using global",
        ),
    )


def test_sparse_pc_factor_policy_uses_large_fortran_reduced_defaults() -> None:
    setup = resolve_sparse_pc_factor_policy(
        env={},
        constrained_pas_pc=False,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=True,
        sparse_pc_linear_size=100000,
        pc_maxiter=120,
        default_permc_spec="MMD_ATA",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
    )

    assert setup.pc_shift == 1.0e-8
    assert setup.factorization == "ilu"
    assert setup.default_factor_kind == "ilu"
    assert setup.default_ilu_fill_factor == 2.0
    assert setup.default_ilu_drop_tol == 1.0e-3
    assert setup.default_pattern_color_batch == 16
    assert setup.factor_dtype_initial == np.dtype(np.float64)
    assert setup.factor_dtype_used == np.dtype(np.float64)
    assert setup.factor_dtype_retry is None
    assert setup.default_permc_spec == "MMD_ATA"
    assert setup.permc_spec == "MMD_ATA"
    assert setup.fp32_probe_maxiter == 2
    assert setup.first_attempt_maxiter == 120


def test_sparse_pc_factor_policy_honors_env_overrides_and_fp32_probe() -> None:
    setup = resolve_sparse_pc_factor_policy(
        env={
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT": "2e-4",
            "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "diagonal",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_DTYPE": "fp32",
            "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "COLAMD",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER": "7",
        },
        constrained_pas_pc=True,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=9,
        pc_maxiter=20,
        default_permc_spec="MMD_AT_PLUS_A",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
    )

    assert setup.pc_shift == 2.0e-4
    assert setup.factorization == "jacobi"
    assert setup.factor_dtype_initial == np.dtype(np.float32)
    assert setup.factor_dtype_used == np.dtype(np.float32)
    assert setup.permc_spec == "COLAMD"
    assert setup.fp32_probe_maxiter == 7
    assert setup.first_attempt_maxiter == 7


def test_sparse_pc_factor_policy_can_defer_dtype_to_host_policy() -> None:
    calls: list[dict[str, object]] = []

    def host_dtype(**kwargs):
        calls.append(kwargs)
        return np.dtype(np.float32)

    setup = resolve_sparse_pc_factor_policy(
        env={
            "SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE": "auto",
            "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "ilu",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT": "bad",
            "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "bad",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER": "bad",
        },
        constrained_pas_pc=False,
        tokamak_fp_pc=True,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=33,
        pc_maxiter=5,
        default_permc_spec="NATURAL",
        host_sparse_factor_dtype=host_dtype,
    )

    assert setup.pc_shift == 1.0e-8
    assert setup.factorization == "ilu"
    assert setup.factor_dtype_initial == np.dtype(np.float32)
    assert setup.permc_spec == "NATURAL"
    assert setup.fp32_probe_maxiter == 2
    assert setup.first_attempt_maxiter == 2
    assert calls == [
        {
            "size": 33,
            "factorization": "ilu",
            "use_implicit": False,
        }
    ]


def test_sparse_pc_factor_dtype_retry_promotes_failed_fp32_probe() -> None:
    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=2.0,
        target=1.0,
    )

    assert decision.retry is True
    assert decision.factor_dtype_used == np.dtype(np.float64)
    assert decision.factor_dtype_retry == "float64"


def test_sparse_pc_factor_dtype_retry_promotes_nonfinite_fp32_probe() -> None:
    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=float("nan"),
        target=1.0,
    )

    assert decision.retry is True
    assert decision.factor_dtype_used == np.dtype(np.float64)
    assert decision.factor_dtype_retry == "float64"


def test_sparse_pc_factor_dtype_retry_keeps_successful_or_fp64_probe() -> None:
    fp32_success = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float32),
        residual_norm=0.5,
        target=1.0,
    )
    fp64_failure = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=np.dtype(np.float64),
        residual_norm=2.0,
        target=1.0,
    )

    assert fp32_success.retry is False
    assert fp32_success.factor_dtype_used == np.dtype(np.float32)
    assert fp32_success.factor_dtype_retry is None
    assert fp64_failure.retry is False
    assert fp64_failure.factor_dtype_used == np.dtype(np.float64)
    assert fp64_failure.factor_dtype_retry is None


def test_sparse_pc_factor_dtype_retry_initial_guess_uses_finite_candidate() -> None:
    fallback = jnp.asarray([9.0, 9.0])
    finite = sparse_pc_factor_dtype_retry_initial_guess(
        np.asarray([1.0, 2.0]),
        fallback,
    )
    nonfinite = sparse_pc_factor_dtype_retry_initial_guess(
        np.asarray([1.0, np.nan]),
        fallback,
    )

    np.testing.assert_allclose(np.asarray(finite), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(nonfinite), np.asarray([9.0, 9.0]))


def test_retry_sparse_pc_factor_dtype_if_needed_preserves_successful_probe_state() -> None:
    build_calls: list[np.dtype] = []
    run_calls: list[tuple[jnp.ndarray, int]] = []

    result = retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(np.float32),
            factor_dtype_retry=None,
            residual_norm=0.5,
            preconditioned_residual_norm=0.25,
            history=(1.0, 0.5),
            target=1.0,
            x=np.asarray([1.0, 2.0]),
            x0_fallback=jnp.asarray([0.0, 0.0]),
            solve_s=3.0,
            pc_maxiter=20,
            operator_bundle="operator0",
            factor_bundle="factor0",
            elapsed_s=lambda: 0.0,
            emit=None,
            build_factor=lambda dtype: build_calls.append(dtype) or ("operator1", "factor1"),
            run_gmres_once=lambda x0, maxiter: run_calls.append((x0, maxiter))
            or (np.zeros(2), 0.0, 0.0, (), 0.0),
        )
    )

    assert result.retried is False
    assert result.factor_dtype_used == np.dtype(np.float32)
    assert result.factor_dtype_retry is None
    assert result.operator_bundle == "operator0"
    assert result.factor_bundle == "factor0"
    assert result.factor_s_increment == 0.0
    assert result.setup_s is None
    np.testing.assert_allclose(result.x, np.asarray([1.0, 2.0]))
    assert result.residual_norm == 0.5
    assert result.preconditioned_residual_norm == 0.25
    assert result.history == (1.0, 0.5)
    assert result.solve_s == 3.0
    assert build_calls == []
    assert run_calls == []


def test_retry_sparse_pc_factor_dtype_if_needed_rebuilds_and_reruns_failed_fp32_probe() -> None:
    messages: list[str] = []
    times = iter((10.0, 10.4, 10.5))
    build_calls: list[np.dtype] = []
    run_calls: list[tuple[np.ndarray, int]] = []

    def build_factor(dtype: np.dtype):
        build_calls.append(np.dtype(dtype))
        return "operator64", "factor64"

    def run_gmres_once(x0: jnp.ndarray, maxiter: int):
        run_calls.append((np.asarray(x0), int(maxiter)))
        return np.asarray([3.0, 4.0]), 0.1, 0.05, (0.5, 0.1), 7.0

    result = retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(np.float32),
            factor_dtype_retry=None,
            residual_norm=2.0,
            preconditioned_residual_norm=1.0,
            history=(2.0,),
            target=1.0,
            x=np.asarray([1.0, 2.0]),
            x0_fallback=jnp.asarray([9.0, 9.0]),
            solve_s=3.0,
            pc_maxiter=20,
            operator_bundle="operator0",
            factor_bundle="factor0",
            elapsed_s=lambda: next(times),
            emit=lambda _level, msg: messages.append(msg),
            build_factor=build_factor,
            run_gmres_once=run_gmres_once,
        )
    )

    assert result.retried is True
    assert result.factor_dtype_used == np.dtype(np.float64)
    assert result.factor_dtype_retry == "float64"
    assert result.operator_bundle == "operator64"
    assert result.factor_bundle == "factor64"
    assert result.factor_s_increment == pytest.approx(0.4)
    assert result.setup_s == pytest.approx(10.5)
    np.testing.assert_allclose(result.x, np.asarray([3.0, 4.0]))
    assert result.residual_norm == 0.1
    assert result.preconditioned_residual_norm == 0.05
    assert result.history == (0.5, 0.1)
    assert result.solve_s == 10.0
    assert build_calls == [np.dtype(np.float64)]
    np.testing.assert_allclose(run_calls[0][0], np.asarray([1.0, 2.0]))
    assert run_calls[0][1] == 20
    assert any("factor_dtype=float64" in msg for msg in messages)


def test_retry_sparse_pc_factor_dtype_from_finalization_context_rebuilds_and_reruns() -> None:
    messages: list[str] = []
    times = iter((20.0, 20.2, 20.3))
    build_calls: list[dict[str, object]] = []
    run_calls: list[tuple[np.ndarray, int]] = []

    def factor_matvec(x):
        return x

    context = SparsePCFactorDtypeRetryFinalizationContext(
        factor_matvec=factor_matvec,
        linear_size=2,
        rhs_dtype=np.dtype(np.float64),
        pattern="pattern",
        emit=lambda _level, msg: messages.append(str(msg)),
        constrained_pas_pc=False,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=False,
        default_permc_spec="COLAMD",
        default_factor_kind="ilu",
        default_ilu_fill_factor=3.0,
        default_ilu_drop_tol=1.0e-3,
        default_pattern_color_batch=4,
        x0_fallback=jnp.asarray([9.0, 9.0], dtype=jnp.float64),
        pc_maxiter=12,
        elapsed_s=lambda: next(times),
    )

    def build_host_sparse_direct_factor_from_matvec(**kwargs):
        build_calls.append(kwargs)
        return "operator64", "factor64"

    def run_sparse_pc_gmres_once_callback(x0, *, maxiter_arg: int):
        run_calls.append((np.asarray(x0), int(maxiter_arg)))
        return np.asarray([4.0, 5.0]), 0.2, 0.1, (0.4, 0.2), 6.0

    result = retry_sparse_pc_factor_dtype_from_finalization_context(
        context,
        factor_dtype_used=np.dtype(np.float32),
        factor_dtype_retry=None,
        residual_norm=3.0,
        preconditioned_residual_norm=2.0,
        history=(3.0,),
        target=1.0,
        x=np.asarray([1.0, 2.0]),
        solve_s=5.0,
        operator_bundle="operator0",
        factor_bundle="factor0",
        build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
    )

    assert result.retried is True
    assert result.factor_dtype_used == np.dtype(np.float64)
    assert result.factor_s_increment == pytest.approx(0.2)
    assert result.setup_s == pytest.approx(20.3)
    assert result.solve_s == pytest.approx(11.0)
    np.testing.assert_allclose(result.x, np.asarray([4.0, 5.0]))
    assert build_calls[0]["matvec"] is factor_matvec
    assert build_calls[0]["factor_dtype"] == np.dtype(np.float64)
    assert build_calls[0]["default_permc_spec"] == "COLAMD"
    assert build_calls[0]["default_factor_kind"] == "ilu"
    assert run_calls[0][1] == 12
    np.testing.assert_allclose(run_calls[0][0], np.asarray([1.0, 2.0]))
    assert any("factor_dtype=float64" in msg for msg in messages)


def test_retry_sparse_pc_factor_dtype_from_solve_state_forwards_build_policy() -> None:
    times = iter((2.0, 2.25, 2.5))
    build_kwargs: list[dict[str, object]] = []
    run_calls: list[tuple[np.ndarray, int]] = []

    def build_factor(**kwargs):
        build_kwargs.append(kwargs)
        return "operator64", "factor64"

    def run_gmres_once(x0: jnp.ndarray, *, maxiter_arg: int):
        run_calls.append((np.asarray(x0), int(maxiter_arg)))
        return np.asarray([4.0, 5.0]), 0.2, 0.1, (0.4, 0.2), 6.0

    state = {
        "_sparse_pc_factor_mv": "matvec",
        "sparse_pc_linear_size": 12,
        "rhs": jnp.ones(3),
        "pattern": "pattern",
        "emit": None,
        "constrained_pas_pc": False,
        "tokamak_fp_pc": True,
        "fortran_reduced_sparse_pc": False,
        "sparse_pc_default_permc_spec": "COLAMD",
        "sparse_pc_default_factor_kind": "ilu",
        "sparse_pc_default_ilu_fill_factor": 3.0,
        "sparse_pc_default_ilu_drop_tol": 1.0e-4,
        "sparse_pc_default_pattern_color_batch": 5,
        "sparse_pc_factor_dtype_used": np.dtype(np.float32),
        "sparse_pc_factor_dtype_retry": None,
        "residual_norm_sparse_pc": 2.0,
        "rn_pc": 1.0,
        "history": (2.0,),
        "target": 1.0,
        "x_np": np.asarray([1.0, 2.0]),
        "x0_sparse": jnp.asarray([9.0, 9.0]),
        "solve_s": 4.0,
        "pc_maxiter": 11,
        "_operator_bundle_pc": "operator0",
        "factor_bundle_pc": "factor0",
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: next(times)),
    }

    result = retry_sparse_pc_factor_dtype_from_solve_state(
        state,
        build_host_sparse_direct_factor_from_matvec=build_factor,
        run_sparse_pc_gmres_once_callback=run_gmres_once,
    )

    assert result.retried is True
    assert result.factor_dtype_used == np.dtype(np.float64)
    assert result.factor_dtype_retry == "float64"
    assert build_kwargs[0]["matvec"] == "matvec"
    assert build_kwargs[0]["n"] == 12
    assert build_kwargs[0]["factor_dtype"] == np.dtype(np.float64)
    assert build_kwargs[0]["default_diag_pivot_thresh"] == 0.0
    assert build_kwargs[0]["default_permc_spec"] == "COLAMD"
    assert build_kwargs[0]["default_factor_kind"] == "ilu"
    assert build_kwargs[0]["default_pattern_color_batch"] == 5
    np.testing.assert_allclose(run_calls[0][0], np.asarray([1.0, 2.0]))
    assert run_calls[0][1] == 11
    assert result.factor_s_increment == pytest.approx(0.25)
    assert result.setup_s == pytest.approx(2.5)
    assert result.solve_s == pytest.approx(10.0)


def test_sparse_pc_memory_budget_preflight_is_noop_without_positive_budget() -> None:
    calls = 0

    def estimate(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(csr_total_nbytes=10**9, dense_total_nbytes=0)

    enforce_sparse_pc_memory_budget(
        SparsePCMemoryBudgetPreflightContext(
            env={"SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "bad"},
            unknowns=10,
            gmres_restart=20,
            csr_nnz=30,
            dtype=np.dtype(np.float64),
            device_count=1,
            estimate_sparse_pc_memory=estimate,
        )
    )

    assert calls == 0


def test_sparse_pc_memory_budget_preflight_passes_estimator_inputs() -> None:
    calls: list[dict[str, object]] = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(csr_total_nbytes=2_000_000, dense_total_nbytes=0)

    enforce_sparse_pc_memory_budget(
        SparsePCMemoryBudgetPreflightContext(
            env={
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "3",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE": "4.5",
            },
            unknowns=10,
            gmres_restart=20,
            csr_nnz=30,
            dtype=np.dtype(np.float32),
            device_count=0,
            estimate_sparse_pc_memory=estimate,
        )
    )

    assert calls == [
        {
            "unknowns": 10,
            "gmres_restart": 20,
            "csr_nnz": 30,
            "dtype": np.dtype(np.float32),
            "factor_fill_estimate": 4.5,
            "device_count": 1,
        }
    ]


def test_sparse_pc_memory_budget_preflight_raises_same_budget_error() -> None:
    def estimate(**_kwargs):
        return SimpleNamespace(csr_total_nbytes=0, dense_total_nbytes=5_500_000)

    with pytest.raises(MemoryError, match="estimated=5.500 MB budget=5.000 MB"):
        enforce_sparse_pc_memory_budget(
            SparsePCMemoryBudgetPreflightContext(
                env={
                    "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB": "5",
                    "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE": "bad",
                },
                unknowns=11,
                gmres_restart=22,
                csr_nnz=33,
                dtype=np.dtype(np.float64),
                device_count=2,
                estimate_sparse_pc_memory=estimate,
            )
        )


def _direct_tail_context(
    *,
    env: dict[str, str] | None,
    sparse_pc_linear_size: int = 100000,
    active_indices: np.ndarray | None = None,
    build_direct_tail_bundle=None,
    elapsed_s=None,
    messages: list[tuple[int, str]] | None = None,
) -> DirectTailMaterializationContext:
    if build_direct_tail_bundle is None:
        def build_direct_tail_bundle(**_kwargs):
            return None

    if elapsed_s is None:
        def elapsed_s():
            return 0.0

    return DirectTailMaterializationContext(
        env=env or {},
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, phi1_size=0),
        op_pc="op-pc",
        pattern="pattern",
        active_indices=active_indices,
        sparse_pc_use_active_dof=active_indices is not None,
        reduce_full=_identity,
        expand_reduced=_identity,
        pc_shift=1.0e-8,
        dtype=np.dtype(np.float64),
        factor_dtype=np.dtype(np.float64),
        sparse_pc_linear_size=sparse_pc_linear_size,
        default_pattern_color_batch=9,
        elapsed_s=elapsed_s,
        emit=None if messages is None else lambda level, msg: messages.append((level, msg)),
        is_direct_reduced_pmat_pc_kind=lambda kind: kind == "direct_pmat",
        build_direct_tail_bundle=build_direct_tail_bundle,
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **kwargs: kwargs,
    )


def test_direct_tail_materialization_respects_disabled_env_without_builder_call() -> None:
    calls = 0

    def builder(**_kwargs):
        nonlocal calls
        calls += 1
        return object()

    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "0"},
            build_direct_tail_bundle=builder,
        )
    )

    assert result.direct_tail_default is True
    assert result.enabled is False
    assert result.built is False
    assert result.operator_bundle is None
    assert result.error is None
    assert calls == 0


def test_direct_tail_materialization_skips_direct_reduced_pmat_request() -> None:
    messages: list[tuple[int, str]] = []
    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER": (
                    "direct-pmat"
                ),
            },
            build_direct_tail_bundle=lambda **_kwargs: pytest.fail("unexpected build"),
            messages=messages,
        )
    )

    assert result.enabled is True
    assert result.direct_reduced_pmat_requested is True
    assert result.built is False
    assert result.pc_env == "direct_pmat"
    assert "materialization skipped" in messages[0][1]


def test_direct_tail_materialization_forwards_builder_args_and_emits_complete() -> None:
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(matrix="csr")
    elapsed_values = iter((1.0, 1.75))
    messages: list[tuple[int, str]] = []

    def builder(**kwargs):
        calls.append(kwargs)
        return bundle

    result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1",
                "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "bad",
                "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "bad",
                "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH": "bad",
            },
            active_indices=np.asarray([2, 0], dtype=np.int64),
            build_direct_tail_bundle=builder,
            elapsed_s=lambda: next(elapsed_values),
            messages=messages,
        )
    )

    assert result.built is True
    assert result.operator_bundle is bundle
    assert result.error is None
    assert calls[0]["active_indices"].tolist() == [2, 0]
    assert calls[0]["csr_max_mb"] == 512.0
    assert calls[0]["drop_tol"] == 0.0
    assert calls[0]["color_batch"] == 9
    assert calls[0]["pc_shift"] == 1.0e-8
    assert "materialization start" in messages[0][1]
    assert "materialization complete elapsed_s=0.750" in messages[1][1]


def test_direct_tail_materialization_records_not_selected_and_exception() -> None:
    none_messages: list[tuple[int, str]] = []
    none_elapsed = iter((3.0, 3.5))
    none_result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1"},
            build_direct_tail_bundle=lambda **_kwargs: None,
            elapsed_s=lambda: next(none_elapsed),
            messages=none_messages,
        )
    )

    assert none_result.built is False
    assert none_result.error is None
    assert "materialization not selected elapsed_s=0.500" in none_messages[1][1]

    err_messages: list[tuple[int, str]] = []
    err_elapsed = iter((4.0, 4.25))

    def broken_builder(**_kwargs):
        raise RuntimeError("boom")

    err_result = build_direct_tail_materialization_setup(
        _direct_tail_context(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL": "1"},
            build_direct_tail_bundle=broken_builder,
            elapsed_s=lambda: next(err_elapsed),
            messages=err_messages,
        )
    )

    assert err_result.built is False
    assert err_result.operator_bundle is None
    assert err_result.error == "RuntimeError: boom"
    assert "materialization disabled after failure elapsed_s=0.250" in err_messages[1][1]


@dataclass(frozen=True)
class _FakeStructuredPreconditioner:
    operator: object | None = object()
    selected: bool = True
    kind: str = "fake_lu"
    reason: str = "ok"
    setup_s: float = 0.25
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": bool(self.selected),
            "kind": str(self.kind),
            "reason": str(self.reason),
            "setup_s": float(self.setup_s),
            "metadata": dict(self.metadata),
        }


class _FakeLayout:
    total_size = 3
    f_size = 2

    def to_dict(self) -> dict[str, int]:
        return {"total_size": 3, "f_size": 2}


def _fake_cache_metadata(
    preconditioner: _FakeStructuredPreconditioner,
    *,
    cache_hit: bool,
    cache_key: tuple[object, ...],
) -> _FakeStructuredPreconditioner:
    metadata = dict(preconditioner.metadata)
    metadata["cache_hit"] = bool(cache_hit)
    metadata["cache_key"] = cache_key
    return replace(preconditioner, metadata=metadata)


def _fake_factor_bundle(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def test_direct_tail_structured_admission_auto_defaults_when_large_bundle_exists() -> None:
    calls: list[dict[str, object]] = []

    def default_max_mb(**kwargs):
        calls.append(kwargs)
        return 768.0

    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={},
            pc_env="",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=100000,
            default_max_mb=default_max_mb,
        )
    )

    assert result.pc_env == ""
    assert result.requested == "auto"
    assert result.auto_default is True
    assert result.required is True
    assert result.setup_allowed is True
    assert result.max_mb_auto is True
    assert result.max_mb == 768.0
    assert result.regularization == 1.0e-12
    assert calls == [{"requested_kind": "auto", "active_size": 100000}]


def test_direct_tail_structured_admission_explicit_kind_is_required_and_uses_env_caps() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB": "12.5",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_REGULARIZATION": "4e-9",
            },
            pc_env="active-fortran-v3-reduced-lu",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=9,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.requested == "active_fortran_v3_reduced_lu"
    assert result.required is True
    assert result.setup_allowed is True
    assert result.max_mb_auto is False
    assert result.max_mb == 12.5
    assert result.regularization == 4.0e-9


def test_direct_tail_structured_admission_fail_closed_and_overrides() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_FAIL_CLOSED_SIZE": "10",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED": "0",
            },
            pc_env="auto",
            operator_bundle=object(),
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=12,
            default_max_mb=lambda **_kwargs: 512.0,
        )
    )

    assert result.requested == "auto"
    assert result.fail_closed_size == 10
    assert result.auto_large_fail_closed is True
    assert result.required is False
    assert result.setup_allowed is True


def test_direct_tail_structured_admission_allows_direct_reduced_pmat_without_bundle() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB": "bad",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_REGULARIZATION": "bad",
            },
            pc_env="direct-pmat",
            operator_bundle=None,
            direct_reduced_pmat_requested=True,
            sparse_pc_linear_size=5,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.requested == "direct_pmat"
    assert result.setup_allowed is True
    assert result.required is True
    assert result.max_mb_auto is False
    assert result.max_mb == 512.0
    assert result.regularization == 1.0e-12


def test_direct_tail_structured_admission_no_bundle_blocks_default_setup() -> None:
    result = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env={},
            pc_env="",
            operator_bundle=None,
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=100000,
            default_max_mb=lambda **_kwargs: pytest.fail("unexpected default cap"),
        )
    )

    assert result.auto_default is False
    assert result.requested is None
    assert result.setup_allowed is False
    assert result.max_mb_auto is False
    assert result.max_mb == 0.0


def test_direct_tail_structured_build_uses_direct_reduced_pmat_builder() -> None:
    calls: dict[str, object] = {}
    active_indices = np.array([0, 2], dtype=np.int64)

    def direct_builder(**kwargs) -> _FakeStructuredPreconditioner:
        calls.update(kwargs)
        return _FakeStructuredPreconditioner(metadata={"factor_nbytes_actual": 123})

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=None,
            active_indices=active_indices,
            requested_kind="direct_reduced_pmat_lu",
            direct_reduced_pmat_requested=True,
            sparse_pc_linear_size=9,
            max_mb=2.0,
            regularization=1.0e-9,
            preconditioner_x=1,
            preconditioner_xi=2,
            preconditioner_species=3,
            preconditioner_x_min_l=4,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=direct_builder,
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache={},
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is True
    assert result.selected is True
    assert result.error is None
    assert result.max_nbytes == 2 * 1024 * 1024
    assert result.cache_hit is False
    assert result.cache_key == ("direct_reduced_pmat_pc_cache_disabled", "direct_reduced_pmat_lu", 9, (1, 2, 3, 4))
    assert result.factor_bundle.factor_nbytes_estimate == 123
    assert calls["active_indices"] is active_indices
    assert calls["max_factor_nbytes"] == 2 * 1024 * 1024
    assert calls["max_csr_nbytes"] == 2 * 1024 * 1024
    assert calls["include_jacobian_terms"] is True


def test_direct_tail_structured_build_reuses_cached_active_preconditioner() -> None:
    cache: dict[tuple[object, ...], object] = {
        ("cached",): _FakeStructuredPreconditioner(metadata={"factor_nbytes_estimate": 77})
    }
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(3, format="csr"))

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=bundle,
            active_indices=None,
            requested_kind="active_fortran_v3_reduced_lu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=1.0,
            regularization=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=1,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache=cache,
            cache_key=lambda **_kwargs: ("cached",),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is True
    assert result.cache_hit is True
    assert result.factor_bundle.operator is bundle
    assert result.factor_bundle.factor_nbytes_estimate == 77
    assert result.preconditioner.metadata["cache_hit"] is True


def test_direct_tail_structured_build_can_disable_active_cache() -> None:
    calls: dict[str, object] = {}
    cache: dict[tuple[object, ...], object] = {}
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(3, format="csr"))

    def active_builder(**kwargs) -> _FakeStructuredPreconditioner:
        calls.update(kwargs)
        return _FakeStructuredPreconditioner(selected=False, operator=None, reason="memory_cap")

    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_CACHE": "0"},
            op=object(),
            operator_bundle=bundle,
            active_indices=np.array([1], dtype=np.int64),
            requested_kind="active_fortran_v3_reduced_ilu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=0.5,
            regularization=2.0e-8,
            preconditioner_x=0,
            preconditioner_xi=1,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=active_builder,
            cache=cache,
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is False
    assert result.selected is False
    assert result.reason == "memory_cap"
    assert result.error is None
    assert result.cache_key == ("direct_tail_structured_pc_cache_disabled", "active_fortran_v3_reduced_ilu", (0, 1, 0, 0))
    assert cache == {}
    assert calls["regularization"] == 2.0e-8


def test_direct_tail_structured_build_reports_missing_matrix_exception() -> None:
    result = build_direct_tail_structured_preconditioner_setup(
        DirectTailStructuredBuildContext(
            env={},
            op=object(),
            operator_bundle=None,
            active_indices=None,
            requested_kind="active_fortran_v3_reduced_lu",
            direct_reduced_pmat_requested=False,
            sparse_pc_linear_size=3,
            max_mb=1.0,
            regularization=1.0e-12,
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            layout_from_operator=lambda _op: _FakeLayout(),
            build_direct_active_preconditioner=lambda **_kwargs: pytest.fail("unexpected direct builder"),
            build_active_projected_preconditioner=lambda **_kwargs: pytest.fail("unexpected active builder"),
            cache={},
            cache_key=lambda **_kwargs: pytest.fail("unexpected cache key"),
            with_cache_metadata=_fake_cache_metadata,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.ready is False
    assert result.selected is False
    assert result.reason == "structured_pc_exception"
    assert result.error == "RuntimeError: direct-tail structured cache requested without a direct-tail matrix"


def test_direct_tail_support_mode_preflight_reports_not_applicable() -> None:
    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1"},
            factor_kind="other",
            structured_pc_ready=True,
            operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr")),
            layout=_FakeLayout(),
            active_indices=None,
            max_nbytes=1024,
            regularization=1.0e-12,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            selector=lambda **_kwargs: pytest.fail("unexpected selector"),
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.requested is True
    assert result.applicable is False
    assert result.selected is False
    assert result.metadata == {
        "selected": False,
        "reason": "support_mode_preflight_not_applicable",
        "structured_pc_ready": True,
        "factor_kind": "other",
    }


def test_direct_tail_support_mode_preflight_selects_factor_bundle() -> None:
    calls: dict[str, object] = {}
    bundle = SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr"))

    def selector(**kwargs):
        calls.update(kwargs)
        return (
            _FakeStructuredPreconditioner(
                kind="active_fortran_v3_reduced_lu",
                reason="support_mode",
                setup_s=0.75,
                metadata={"factor_nbytes_actual": 55},
            ),
            {
                "selected_candidate": "xmin_l2",
                "baseline_residual_after": 2.0,
                "best_residual_after": 1.0,
            },
        )

    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_CANDIDATES": "current,xmin_l2",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MAX_CANDIDATES": "2",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MIN_IMPROVEMENT": "1.25",
            },
            factor_kind="active-fortran-v3-reduced-lu",
            structured_pc_ready=True,
            operator_bundle=bundle,
            layout=_FakeLayout(),
            active_indices=np.array([0, 1], dtype=np.int64),
            max_nbytes=2048,
            regularization=1.0e-9,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=1,
            preconditioner_xi=2,
            preconditioner_species=3,
            preconditioner_x_min_l=4,
            selector=selector,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.applicable is True
    assert result.selected is True
    assert result.factor_bundle.operator is bundle
    assert result.factor_bundle.factor_nbytes_estimate == 55
    assert result.metadata["selected_candidate"] == "xmin_l2"
    assert calls["requested_kind"] == "active_fortran_v3_reduced_lu"
    assert calls["candidates"] == "current,xmin_l2"
    assert calls["max_candidates"] == 2
    assert calls["min_improvement_ratio"] == 1.25


def test_direct_tail_support_mode_preflight_reports_selector_exception() -> None:
    def selector(**_kwargs):
        raise RuntimeError("selector failed")

    result = run_direct_tail_support_mode_preflight(
        DirectTailSupportModePreflightContext(
            env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT": "1"},
            factor_kind="active_fortran_v3_reduced_ilu",
            structured_pc_ready=True,
            operator_bundle=SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr")),
            layout=_FakeLayout(),
            active_indices=None,
            max_nbytes=1024,
            regularization=1.0e-12,
            rhs=np.ones(2),
            true_matvec=lambda v: np.asarray(v),
            preconditioner_x=0,
            preconditioner_xi=0,
            preconditioner_species=0,
            preconditioner_x_min_l=0,
            selector=selector,
            factor_bundle=_fake_factor_bundle,
        )
    )

    assert result.applicable is True
    assert result.selected is False
    assert result.metadata is None
    assert result.error == "RuntimeError: selector failed"


def test_sparse_pc_factor_preflight_policy_uses_metadata_trigger() -> None:
    policy = resolve_sparse_pc_factor_preflight_policy(
        SparsePCFactorPreflightPolicyContext(
            env={},
            fortran_reduced_sparse_pc=True,
            structured_pc_ready=True,
            structured_pc_metadata={
                "kind": "",
                "metadata": {
                    "requested_kind": "active-fortran-v3-reduced-ilu",
                    "requires_preflight": True,
                },
            },
            sparse_pc_linear_size=10,
        )
    )

    assert policy.factor_preflight_enabled is True
    assert policy.factor_preflight_required is False
    assert policy.factor_preflight_seed_enabled is True
    assert policy.direct_tail_structured_pc_requires_preflight is True
    assert policy.direct_tail_structured_pc_kind_for_preflight == "active_fortran_v3_reduced_ilu"
    assert policy.direct_tail_structured_pc_size_requires_preflight is False
    assert policy.structured_pc_preflight_required is True
    assert policy.factor_preflight_max_target_ratio == 1.0e6


def test_sparse_pc_factor_preflight_policy_uses_size_trigger_and_overrides() -> None:
    policy = resolve_sparse_pc_factor_preflight_policy(
        SparsePCFactorPreflightPolicyContext(
            env={
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT": "0",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_REQUIRED": "1",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_SEED": "0",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE": "20",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED": "0",
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_MAX_TARGET_RATIO": "0.5",
            },
            fortran_reduced_sparse_pc=True,
            structured_pc_ready=True,
            structured_pc_metadata={
                "kind": "active-fortran-v3-reduced-ilu",
                "metadata": {},
            },
            sparse_pc_linear_size=25,
        )
    )

    assert policy.factor_preflight_enabled is False
    assert policy.factor_preflight_required is True
    assert policy.factor_preflight_seed_enabled is False
    assert policy.structured_pc_preflight_required_min_size == 20
    assert policy.direct_tail_structured_pc_kind_for_preflight == "active_fortran_v3_reduced_ilu"
    assert policy.direct_tail_structured_pc_size_requires_preflight is True
    assert policy.structured_pc_preflight_required is False
    assert policy.factor_preflight_max_target_ratio == 1.0


def test_sparse_pc_factor_preflight_evaluation_passes_and_seeds() -> None:
    diagnostics_calls: list[dict[str, object]] = []

    def diagnostics(**kwargs):
        diagnostics_calls.append(kwargs)
        return {"selected": True}

    result = evaluate_sparse_pc_factor_preflight(
        SparsePCFactorPreflightEvaluationContext(
            rhs=jnp.asarray([2.0]),
            rhs_norm=2.0,
            target=0.1,
            preconditioner=lambda _rhs: jnp.asarray([1.0]),
            matvec=lambda _x: jnp.asarray([1.5]),
            diagnostics=diagnostics,
            layout=_FakeLayout(),
            active_indices=np.array([0], dtype=np.int64),
            seed_enabled=True,
            max_target_ratio=10.0,
        )
    )

    assert result.residual_before == 2.0
    assert result.residual_after == 0.5
    assert result.improvement_ratio == 4.0
    assert result.target_ratio == 5.0
    assert result.passed is True
    assert result.seed_used is True
    np.testing.assert_allclose(np.asarray(result.x0_seed), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(result.residual_vec), np.asarray([0.5]))
    assert result.diagnostics == {"selected": True}
    assert diagnostics_calls[0]["layout"].to_dict() == {"total_size": 3, "f_size": 2}
    np.testing.assert_array_equal(diagnostics_calls[0]["active_indices"], np.array([0]))


def test_sparse_pc_factor_preflight_evaluation_rejects_large_target_ratio() -> None:
    result = evaluate_sparse_pc_factor_preflight(
        SparsePCFactorPreflightEvaluationContext(
            rhs=jnp.asarray([2.0]),
            rhs_norm=2.0,
            target=0.1,
            preconditioner=lambda _rhs: jnp.asarray([1.0]),
            matvec=lambda _x: jnp.asarray([1.5]),
            diagnostics=lambda **_kwargs: {},
            layout=_FakeLayout(),
            active_indices=None,
            seed_enabled=False,
            max_target_ratio=2.0,
        )
    )

    assert result.residual_after == 0.5
    assert result.target_ratio == 5.0
    assert result.passed is False
    assert result.seed_used is False
    assert result.x0_seed is None


def test_sparse_pc_residual_candidate_acceptance_strict_passes_and_seeds() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=1.0,
            current_residual_after=2.0,
            original_residual_before=4.0,
            target=0.5,
            max_target_ratio=3.0,
            seed_enabled=True,
        )
    )

    assert result.finite_candidate is True
    assert result.improves_current_residual is True
    assert result.improves_original_residual is True
    assert result.strict_accept is True
    assert result.base_improvement_accept is False
    assert result.accepted is True
    assert result.base_improvement_override_used is False
    assert result.improvement_ratio == 4.0
    assert result.target_ratio == 2.0
    assert result.passed is True
    assert result.seed_used is True


def test_sparse_pc_residual_candidate_acceptance_current_only_can_select_without_pass() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=2.5,
            current_residual_after=3.0,
            original_residual_before=2.0,
            target=1.0,
            max_target_ratio=10.0,
            seed_enabled=True,
            require_original_improvement=False,
            current_min_improvement=0.1,
        )
    )

    assert result.improves_current_residual is True
    assert result.improves_original_residual is False
    assert result.strict_accept is True
    assert result.accepted is True
    assert result.improvement_ratio == pytest.approx(0.8)
    assert result.passed is False
    assert result.seed_used is False


def test_sparse_pc_residual_candidate_acceptance_base_improvement_override_sets_passed() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=2.5,
            current_residual_after=3.0,
            original_residual_before=2.0,
            target=1.0,
            max_target_ratio=1.0,
            seed_enabled=False,
            accept_base_improvement=True,
            base_improvement_requires_original_miss=False,
            base_improvement_sets_passed=True,
        )
    )

    assert result.strict_accept is False
    assert result.base_improvement_accept is True
    assert result.accepted is True
    assert result.base_improvement_override_used is True
    assert result.target_ratio == 2.5
    assert result.passed is True


def test_sparse_pc_residual_candidate_acceptance_rejects_nonfinite_candidate() -> None:
    result = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=float("nan"),
            current_residual_after=1.0,
            original_residual_before=2.0,
            target=0.1,
            max_target_ratio=10.0,
            seed_enabled=True,
        )
    )

    assert result.finite_candidate is False
    assert result.improves_current_residual is False
    assert result.accepted is False
    assert result.target_ratio == float("inf")
    assert result.passed is False
    assert result.seed_used is False


def test_sparse_pc_auto_preflight_retry_selection_filters_after_selected_kind() -> None:
    result = select_sparse_pc_auto_preflight_retry_candidates(
        SparsePCAutoPreflightRetrySelectionContext(
            metadata={
                "auto_selected_kind": "active-spilu",
                "auto_candidates": [
                    "active-spilu",
                    "active-ilu",
                    "active-fortran-v3-reduced-lu",
                    "structured",
                    "jacobi",
                ],
                "auto_rejected_candidates": [{"kind": "active_ilu"}],
            },
            current_kind="fallback",
            sparse_pc_linear_size=25,
            preflight_required_min_size=20,
            skip_large_kinds_raw="jacobi,diagonal",
            max_candidates=3,
        )
    )

    assert result.selected_kind == "active_spilu"
    assert result.auto_candidates == (
        "active_spilu",
        "active_ilu",
        "active_fortran_v3_reduced_lu",
        "structured",
        "jacobi",
    )
    assert result.rejected_kinds == frozenset({"active_ilu"})
    assert result.retry_candidates == ("active_fortran_v3_reduced_lu",)


def test_sparse_pc_auto_preflight_retry_selection_uses_current_kind_when_metadata_missing() -> None:
    result = select_sparse_pc_auto_preflight_retry_candidates(
        SparsePCAutoPreflightRetrySelectionContext(
            metadata={
                "auto_candidates": ["active_global_sparse_lu", "active_fortran_v3_reduced_lu"],
            },
            current_kind="active-global-sparse-lu",
            sparse_pc_linear_size=3,
            preflight_required_min_size=20,
            skip_large_kinds_raw="",
            max_candidates=1,
        )
    )

    assert result.selected_kind == "active_global_sparse_lu"
    assert result.retry_candidates == ("active_fortran_v3_reduced_lu",)


def test_sparse_pc_auto_preflight_retry_evaluation_required_candidate_must_pass_gate() -> None:
    result = evaluate_sparse_pc_auto_preflight_retry(
        SparsePCAutoPreflightRetryEvaluationContext(
            residual_after=0.5,
            target=0.25,
            max_target_ratio=3.0,
            residual_before=2.0,
            sparse_pc_linear_size=30,
            preflight_required_min_size=20,
            retry_kind="active-global-sparse-lu",
            retry_metadata={"requires_preflight": False},
        )
    )

    assert result.target_ratio == 2.0
    assert result.requires_metadata is False
    assert result.requires_size is True
    assert result.required is True
    assert result.preflight_passed is True
    assert result.policy_passed is True


def test_sparse_pc_auto_preflight_retry_evaluation_lu_can_pass_policy_without_required_preflight() -> None:
    result = evaluate_sparse_pc_auto_preflight_retry(
        SparsePCAutoPreflightRetryEvaluationContext(
            residual_after=10.0,
            target=1.0,
            max_target_ratio=1.0,
            residual_before=1.0,
            sparse_pc_linear_size=30,
            preflight_required_min_size=20,
            retry_kind="active-fortran-v3-reduced-lu",
            retry_metadata={},
        )
    )

    assert result.target_ratio == 10.0
    assert result.requires_metadata is False
    assert result.requires_size is False
    assert result.required is False
    assert result.preflight_passed is False
    assert result.policy_passed is True


def test_direct_tail_residual_rescue_policy_defaults() -> None:
    policy = resolve_direct_tail_residual_rescue_policy({})

    assert isinstance(policy, DirectTailResidualRescuePolicy)
    assert policy.residual_coarse_requested is False
    assert policy.residual_coarse_rank == 4
    assert policy.residual_coarse_max_mb == 512.0
    assert policy.residual_window_requested is False
    assert policy.residual_window_max_windows == 2
    assert policy.residual_window_coefficient_mode == "additive"
    assert policy.residual_window_combine_mode == "independent"
    assert policy.true_window_requested is False
    assert policy.true_window_max_windows == 1
    assert policy.true_window_column_batch == 4
    assert policy.true_window_include_tail is True
    assert policy.true_coupled_coarse_explicit_requested is False
    assert policy.true_coupled_coarse_auto_enabled is True
    assert policy.true_coupled_coarse_auto_native_enabled is False
    assert policy.true_coupled_coarse_auto_target_ratio == 10.0
    assert policy.true_coupled_coarse_auto_min_size == 300000


def test_direct_tail_residual_rescue_policy_normalizes_modes_and_clamps() -> None:
    policy = resolve_direct_tail_residual_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_RANK": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_MAX_MB": "-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_MAX_WINDOWS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_X_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_ELL_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS": "NORMAL-EQUATIONS",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE": "graph-interface",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DROP_TOL": "-1e-4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_BETA_MAX": "-2",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_NATIVE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE": "0",
        }
    )

    assert policy.residual_coarse_requested is True
    assert policy.residual_coarse_rank == 1
    assert policy.residual_coarse_max_mb == 0.0
    assert policy.residual_window_requested is True
    assert policy.residual_window_max_windows == 1
    assert policy.residual_window_x_radius == 0
    assert policy.residual_window_ell_radius == 0
    assert policy.residual_window_coefficient_mode == "normal_equations"
    assert policy.residual_window_combine_mode == "graph_interface"
    assert policy.true_window_requested is True
    assert policy.true_window_drop_tol == 0.0
    assert policy.true_window_include_tail is False
    assert policy.true_window_damping is True
    assert policy.true_window_beta_max == 0.0
    assert policy.true_coupled_coarse_explicit_requested is True
    assert policy.true_coupled_coarse_auto_enabled is False
    assert policy.true_coupled_coarse_auto_native_enabled is True
    assert policy.true_coupled_coarse_auto_target_ratio == 1.0
    assert policy.true_coupled_coarse_auto_min_size == 1


def test_direct_tail_residual_rescue_policy_falls_back_for_bad_modes() -> None:
    policy = resolve_direct_tail_residual_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE": "bad",
        }
    )

    assert policy.residual_window_coefficient_mode == "additive"
    assert policy.residual_window_combine_mode == "independent"


def test_direct_tail_true_active_rescue_policy_defaults_and_inheritance() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy({})

    assert isinstance(policy, DirectTailTrueActiveRescuePolicy)
    assert policy.active_block_requested is False
    assert policy.active_residual_block_requested is False
    assert policy.active_submatrix_requested is False
    assert policy.active_column_cache_requested is True
    assert policy.active_column_cache_max_mb == 512.0
    assert policy.active_block_x_count == 1
    assert policy.active_block_ell_count == 8
    assert policy.active_block_species_count is None
    assert policy.active_block_theta_stride == 1
    assert policy.active_block_zeta_stride == 1
    assert policy.active_block_max_mb == 1024.0
    assert policy.active_block_regularization == 1.0e-12
    assert policy.active_block_max_size == 4096
    assert policy.active_block_column_batch == 8
    assert policy.active_block_drop_tol == 1.0e-14
    assert policy.active_block_include_tail is True
    assert policy.active_block_max_tail == 512
    assert policy.active_block_damping is False
    assert policy.active_block_beta_max == 10.0
    assert policy.active_residual_block_max_mb == policy.active_block_max_mb
    assert policy.active_residual_block_regularization == policy.active_block_regularization
    assert policy.active_residual_block_max_size == policy.active_block_max_size
    assert policy.active_residual_block_column_batch == policy.active_block_column_batch
    assert policy.active_residual_block_drop_tol == policy.active_block_drop_tol
    assert policy.active_residual_block_include_tail == policy.active_block_include_tail
    assert policy.active_residual_block_max_tail == policy.active_block_max_tail
    assert policy.active_residual_block_damping == policy.active_block_damping
    assert policy.active_residual_block_beta_max == policy.active_block_beta_max
    assert policy.active_residual_block_kinetic_only is True
    assert policy.active_residual_block_min_improvement == 1.0e-6
    assert policy.active_residual_block_accept_base_improvement is False
    assert policy.active_submatrix_damping is True
    assert policy.active_submatrix_alpha_clip == 10.0
    assert policy.active_submatrix_min_improvement == 1.0e-6


def test_direct_tail_true_active_rescue_policy_clamps_and_overrides() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE_MAX_MB": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_X_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ELL_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT": "3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_THETA_STRIDE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ZETA_STRIDE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_MB": "-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_REGULARIZATION": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_SIZE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_COLUMN_BATCH": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DROP_TOL": "-1e-5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_TAIL": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_BETA_MAX": "-10",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_MB": "9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_REGULARIZATION": "2e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_SIZE": "11",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_COLUMN_BATCH": "12",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DROP_TOL": "3e-4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_INCLUDE_TAIL": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_TAIL": "13",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_KINETIC_ONLY": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DAMPING": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_BETA_MAX": "14",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MIN_IMPROVEMENT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_ACCEPT_BASE_IMPROVEMENT": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_DAMPING": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_ALPHA_CLIP": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_MIN_IMPROVEMENT": "-1",
        }
    )

    assert policy.active_block_requested is True
    assert policy.active_residual_block_requested is True
    assert policy.active_submatrix_requested is True
    assert policy.active_column_cache_requested is False
    assert policy.active_column_cache_max_mb == 0.0
    assert policy.active_block_x_count == 0
    assert policy.active_block_ell_count == 0
    assert policy.active_block_species_count == 3
    assert policy.active_block_theta_stride == 1
    assert policy.active_block_zeta_stride == 1
    assert policy.active_block_max_mb == 0.0
    assert policy.active_block_regularization == 0.0
    assert policy.active_block_max_size == 1
    assert policy.active_block_column_batch == 1
    assert policy.active_block_drop_tol == 0.0
    assert policy.active_block_include_tail is False
    assert policy.active_block_max_tail == 0
    assert policy.active_block_damping is True
    assert policy.active_block_beta_max == 0.0
    assert policy.active_residual_block_max_mb == 9.0
    assert policy.active_residual_block_regularization == 2.0e-8
    assert policy.active_residual_block_max_size == 11
    assert policy.active_residual_block_column_batch == 12
    assert policy.active_residual_block_drop_tol == 3.0e-4
    assert policy.active_residual_block_include_tail is True
    assert policy.active_residual_block_max_tail == 13
    assert policy.active_residual_block_kinetic_only is False
    assert policy.active_residual_block_damping is False
    assert policy.active_residual_block_beta_max == 14.0
    assert policy.active_residual_block_min_improvement == 0.0
    assert policy.active_residual_block_accept_base_improvement is True
    assert policy.active_submatrix_damping is False
    assert policy.active_submatrix_alpha_clip == 0.0
    assert policy.active_submatrix_min_improvement == 0.0


def test_direct_tail_true_active_rescue_policy_bad_species_count_is_none() -> None:
    policy = resolve_direct_tail_true_active_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT": "bad",
        }
    )

    assert policy.active_block_species_count is None


def test_direct_tail_coupled_coarse_rescue_policy_defaults() -> None:
    policy = resolve_direct_tail_coupled_coarse_rescue_policy({})

    assert isinstance(policy, DirectTailCoupledCoarseRescuePolicy)
    assert policy.max_windows == 2
    assert policy.x_radius == 0
    assert policy.ell_radius == 1
    assert policy.max_mb == 512.0
    assert policy.regularization == 1.0e-12
    assert policy.max_size == 64
    assert policy.column_batch == 4
    assert policy.drop_tol == 1.0e-14
    assert policy.low_lmax == 3
    assert policy.profile_moment_count == 4
    assert policy.angular_lmax == 2
    assert policy.angular_mode_max == 1
    assert policy.max_tail_units == 16
    assert policy.include_tail is True
    assert policy.include_constraint_sources is True
    assert policy.include_fsavg is True
    assert policy.include_window_residual is True
    assert policy.include_profile_moments is True
    assert policy.include_angular_residual is True
    assert policy.include_angular_basis is False
    assert policy.include_preconditioned_loads is False
    assert policy.preconditioned_load_max_columns == 16
    assert policy.preconditioned_load_max_nnz == 50000
    assert policy.preconditioned_load_drop_tol == 1.0e-12
    assert policy.damping is False
    assert policy.beta_max == 10.0
    assert policy.accept_base_improvement is False


def test_direct_tail_coupled_coarse_rescue_policy_clamps_and_overrides() -> None:
    policy = resolve_direct_tail_coupled_coarse_rescue_policy(
        {
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_WINDOWS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_X_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ELL_RADIUS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_MB": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_REGULARIZATION": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COLUMN_BATCH": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DROP_TOL": "-1e-3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_LOW_LMAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PROFILE_MOMENT_COUNT": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_LMAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_MODE_MAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_CONSTRAINT_SOURCES": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_FSAVG": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_WINDOW_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PROFILE_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_BASIS": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_DROP_TOL": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DAMPING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_BETA_MAX": "-1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ACCEPT_BASE_IMPROVEMENT": "1",
        }
    )

    assert policy.max_windows == 1
    assert policy.x_radius == 0
    assert policy.ell_radius == 0
    assert policy.max_mb == 0.0
    assert policy.regularization == 0.0
    assert policy.max_size == 1
    assert policy.column_batch == 1
    assert policy.drop_tol == 0.0
    assert policy.low_lmax == 0
    assert policy.profile_moment_count == 0
    assert policy.angular_lmax == 0
    assert policy.angular_mode_max == 0
    assert policy.max_tail_units == 0
    assert policy.include_tail is False
    assert policy.include_constraint_sources is False
    assert policy.include_fsavg is False
    assert policy.include_window_residual is False
    assert policy.include_profile_moments is False
    assert policy.include_angular_residual is False
    assert policy.include_angular_basis is True
    assert policy.include_preconditioned_loads is True
    assert policy.preconditioned_load_max_columns == 0
    assert policy.preconditioned_load_max_nnz == 0
    assert policy.preconditioned_load_drop_tol == 0.0
    assert policy.damping is True
    assert policy.beta_max == 0.0
    assert policy.accept_base_improvement is True


def test_fortran_reduced_xblock_factor_policy_uses_specific_env_before_generic() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL": "9.0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_TOL": "1.5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL": "2.5e-7",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_ILU_DROP_TOL": "bad",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR": "4.0",
        },
        preconditioner_xi=2,
    )

    assert setup.drop_tol == 1.5
    assert setup.drop_rel == 2.5e-7
    assert setup.ilu_drop_tol == 1.0e-4
    assert setup.fill_factor == 4.0
    assert setup.preconditioner_xi == 2
    assert setup.promote_xi
    assert setup.messages == ()


def test_fortran_reduced_xblock_factor_policy_promotes_zero_xi_by_default() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={},
        preconditioner_xi=0,
    )

    assert setup.preconditioner_xi == 1
    assert setup.promote_xi
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
            "promoting x-block backend preconditioner_xi 0 -> 1 for stronger FP block factors",
        ),
    )


def test_fortran_reduced_xblock_factor_policy_can_disable_xi_promotion() -> None:
    setup = resolve_fortran_reduced_xblock_factor_policy(
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PROMOTE_XI": "off"},
        preconditioner_xi=0,
    )

    assert setup.preconditioner_xi == 0
    assert not setup.promote_xi
    assert setup.messages == ()


def test_fortran_reduced_xblock_factor_stage_builds_with_policy_and_timing() -> None:
    messages: list[str] = []
    times = iter([2.0, 2.75])
    calls: dict[str, object] = {}

    def assembled_allowed(**kwargs) -> bool:
        calls["assembled"] = kwargs
        return True

    def builder(**kwargs):
        calls["builder"] = kwargs
        return lambda v: 3.0 * v

    result = build_fortran_reduced_xblock_factor_stage(
        context=FortranReducedXBlockFactorBuildContext(
            op_pc=SimpleNamespace(),
            reduce_full=_identity,
            expand_reduced=_identity,
            preconditioner_species=1,
            preconditioner_xi=0,
            sparse_pc_linear_size=42,
            backend_reason="auto_large_full_fp",
            elapsed_s=lambda: next(times),
            emit=lambda _level, msg: messages.append(msg),
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL": "2e-7",
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_FILL_FACTOR": "6",
            },
            assembled_host_allowed=assembled_allowed,
            builder=builder,
        )
    )

    assert result.preconditioner_xi == 1
    assert result.drop_rel == pytest.approx(2.0e-7)
    assert result.fill_factor == pytest.approx(6.0)
    assert result.force_assembled_host_fp is True
    assert result.factor_s == pytest.approx(0.75)
    assert result.preconditioner(jnp.asarray([2.0])).tolist() == [6.0]
    assert calls["assembled"]["preconditioner_xi"] == 1
    assert calls["builder"]["preconditioner_species"] == 1
    assert calls["builder"]["force_assembled_host_fp"] is True
    assert any("promoting x-block backend preconditioner_xi 0 -> 1" in message for message in messages)
    assert any("using x-block backend instead of monolithic CSR factor" in message for message in messages)


def test_sparse_xblock_rescue_build_promotes_xi_and_builds_with_markers() -> None:
    messages: list[str] = []
    calls: dict[str, object] = {}
    marks: list[str] = []
    op = SimpleNamespace(fblock=SimpleNamespace(fp=object(), pas=None))

    def assembled_allowed(**kwargs) -> bool:
        calls["assembled"] = kwargs
        return True

    def builder(**kwargs):
        calls["builder"] = kwargs
        return lambda v: 2.0 * v

    result = build_sparse_xblock_rescue_preconditioner(
        context=SparseXBlockRescueBuildContext(
            op=op,
            reduce_full=_identity,
            expand_reduced=_identity,
            active_size=42,
            preconditioner_species=1,
            preconditioner_x=3,
            preconditioner_xi=0,
            use_implicit=False,
            drop_tol=0.1,
            drop_rel=0.2,
            ilu_drop_tol=0.3,
            fill_factor=4.0,
            emit=lambda _level, msg: messages.append(msg),
            mark=marks.append,
            assembled_host_allowed=assembled_allowed,
            builder=builder,
        )
    )

    assert result.preconditioner_xi == 1
    assert result.force_assembled_host_fp is True
    assert result.preconditioner(jnp.asarray([3.0])).tolist() == [6.0]
    assert marks == ["rhs1_sparse_precond_build_start", "rhs1_sparse_precond_build_done"]
    assert calls["assembled"]["preconditioner_xi"] == 1
    assert calls["builder"]["build_jax_factors"] is False
    assert calls["builder"]["preconditioner_xi"] == 1
    assert calls["builder"]["force_assembled_host_fp"] is True
    assert any("v3-like sparse x-block rescue" in message for message in messages)
    assert any("promoting sparse x-block rescue preconditioner_xi 0 -> 1" in message for message in messages)


def test_sparse_xblock_rescue_build_keeps_xi_for_implicit_or_pas_cases() -> None:
    calls: dict[str, object] = {}
    op = SimpleNamespace(fblock=SimpleNamespace(fp=object(), pas=object()))

    def assembled_allowed(**kwargs) -> bool:
        calls["assembled"] = kwargs
        return False

    def builder(**kwargs):
        calls["builder"] = kwargs
        return _identity

    result = build_sparse_xblock_rescue_preconditioner(
        context=SparseXBlockRescueBuildContext(
            op=op,
            reduce_full=_identity,
            expand_reduced=_identity,
            active_size=42,
            preconditioner_species=1,
            preconditioner_x=3,
            preconditioner_xi=0,
            use_implicit=True,
            drop_tol=0.1,
            drop_rel=0.2,
            ilu_drop_tol=0.3,
            fill_factor=4.0,
            emit=None,
            mark=lambda _name: None,
            assembled_host_allowed=assembled_allowed,
            builder=builder,
        )
    )

    assert result.preconditioner_xi == 0
    assert result.force_assembled_host_fp is False
    assert calls["assembled"]["preconditioner_xi"] == 0
    assert calls["builder"]["build_jax_factors"] is True
    assert calls["builder"]["preconditioner_xi"] == 0


def test_sparse_xblock_explicit_seed_accepts_and_refines(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH", "0")
    messages: list[str] = []

    result = apply_sparse_xblock_explicit_seed(
        context=SparseXBlockExplicitSeedContext(
            preconditioner=lambda v: 0.5 * v,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            matvec=_identity,
            current_result=GMRESSolveResult(
                x=jnp.asarray([0.0], dtype=jnp.float64),
                residual_norm=jnp.asarray(10.0, dtype=jnp.float64),
            ),
            target=1.0e-9,
            tol=1.0e-9,
            atol=1.0e-12,
            restart=20,
            maxiter=40,
            precondition_side="left",
            active_size=10,
            emit=lambda _level, msg: messages.append(msg),
            polish_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("polish disabled")),
        )
    )

    assert result.result is not None
    assert float(result.result.residual_norm) == pytest.approx(0.125)
    assert result.seed_residual == pytest.approx(0.5)
    assert result.seed_improvement_ratio == pytest.approx(20.0)
    assert result.seed_accept_ratio == pytest.approx(10.0)
    assert result.refine_steps == 2
    assert result.refines_performed == 2
    assert result.reason == "seed_accepted"
    assert any("explicit FP x-block seed" in message for message in messages)
    assert any("explicit FP x-block refinement steps=2/2" in message for message in messages)


def test_sparse_xblock_explicit_seed_rejects_bad_seed(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_REFINES", "0")
    messages: list[str] = []

    result = apply_sparse_xblock_explicit_seed(
        context=SparseXBlockExplicitSeedContext(
            preconditioner=lambda v: jnp.zeros_like(v),
            rhs=jnp.asarray([10.0], dtype=jnp.float64),
            matvec=_identity,
            current_result=GMRESSolveResult(
                x=jnp.asarray([0.0], dtype=jnp.float64),
                residual_norm=jnp.asarray(0.1, dtype=jnp.float64),
            ),
            target=1.0e-9,
            tol=1.0e-9,
            atol=1.0e-12,
            restart=20,
            maxiter=40,
            precondition_side="left",
            active_size=10,
            emit=lambda _level, msg: messages.append(msg),
            polish_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("polish disabled")),
        )
    )

    assert result.result is None
    assert result.seed_residual == pytest.approx(10.0)
    assert result.seed_improvement_ratio == pytest.approx(0.01)
    assert result.refine_steps == 0
    assert result.refines_performed == 0
    assert result.reason == "seed_rejected_accept_gate"
    assert any("explicit FP x-block seed rejected" in message for message in messages)


def test_sparse_xblock_rescue_solve_stage_dispatches_implicit_solver() -> None:
    calls: dict[str, object] = {}
    marks: list[str] = []
    expected = GMRESSolveResult(
        x=jnp.asarray([2.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(0.25, dtype=jnp.float64),
    )

    def solve_linear(**kwargs):
        calls["solve_linear"] = kwargs
        return expected

    result = run_sparse_xblock_rescue_solve_stage(
        context=SparseXBlockRescueSolveContext(
            preconditioner=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            matvec=_identity,
            current_result=GMRESSolveResult(
                x=jnp.asarray([0.0], dtype=jnp.float64),
                residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
            ),
            target=1.0e-9,
            tol=1.0e-9,
            atol=1.0e-12,
            restart=20,
            maxiter=40,
            precondition_side="left",
            active_size=10,
            use_implicit=True,
            assembled_host_fp=False,
            emit=None,
            mark=marks.append,
            solve_linear=solve_linear,
            host_gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("host not used")),
        )
    )

    assert result.result is expected
    assert result.reason == "started"
    assert result.candidate_residual is None
    assert marks == ["rhs1_sparse_precond_solve_start", "rhs1_sparse_precond_solve_done"]
    assert calls["solve_linear"]["solve_method_val"] == "incremental"
    assert calls["solve_linear"]["precond_fn"] is _identity


def test_sparse_xblock_rescue_solve_stage_builds_host_gmres_candidate() -> None:
    marks: list[str] = []

    result = run_sparse_xblock_rescue_solve_stage(
        context=SparseXBlockRescueSolveContext(
            preconditioner=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            matvec=_identity,
            current_result=GMRESSolveResult(
                x=jnp.asarray([0.0], dtype=jnp.float64),
                residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
            ),
            target=1.0e-9,
            tol=1.0e-9,
            atol=1.0e-12,
            restart=20,
            maxiter=40,
            precondition_side="left",
            active_size=10,
            use_implicit=False,
            assembled_host_fp=False,
            emit=None,
            mark=marks.append,
            solve_linear=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("implicit not used")),
            host_gmres_solver=lambda **_kwargs: (np.asarray([0.75]), 0.0, ()),
        )
    )

    assert result.result is not None
    assert float(result.result.residual_norm) == pytest.approx(0.25)
    assert result.candidate_residual == pytest.approx(0.25)
    assert result.reason == "gmres_candidate"
    assert marks == ["rhs1_sparse_precond_solve_start", "rhs1_sparse_precond_solve_done"]


def test_sparse_xblock_rescue_solve_stage_routes_assembled_seed(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH", "0")
    marks: list[str] = []

    result = run_sparse_xblock_rescue_solve_stage(
        context=SparseXBlockRescueSolveContext(
            preconditioner=lambda v: 0.5 * v,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            matvec=_identity,
            current_result=GMRESSolveResult(
                x=jnp.asarray([0.0], dtype=jnp.float64),
                residual_norm=jnp.asarray(10.0, dtype=jnp.float64),
            ),
            target=1.0e-9,
            tol=1.0e-9,
            atol=1.0e-12,
            restart=20,
            maxiter=40,
            precondition_side="left",
            active_size=10,
            use_implicit=False,
            assembled_host_fp=True,
            emit=None,
            mark=marks.append,
            solve_linear=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("implicit not used")),
            host_gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("polish disabled")),
        )
    )

    assert result.result is not None
    assert float(result.result.residual_norm) == pytest.approx(0.125)
    assert result.reason == "seed_accepted"
    assert result.seed_residual == pytest.approx(0.5)
    assert result.seed_refines_performed == 2
    assert marks == ["rhs1_sparse_precond_solve_start", "rhs1_sparse_precond_solve_done"]


def test_sparse_xblock_rescue_acceptance_rejects_non_improving_candidate() -> None:
    replay = SimpleNamespace(x0_vec="old")
    record_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0, dtype=jnp.float64),
    )
    candidate = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = accept_sparse_xblock_rescue_candidate(
        context=SparseXBlockRescueAcceptanceContext(
            current_result=current,
            candidate_result=candidate,
            reason="gmres_candidate",
            assembled_host_fp=False,
            use_implicit=False,
            replay_state=replay,
            matvec=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=_identity,
            precondition_side="left",
            solver_kind="gmres",
            restart=20,
            maxiter=40,
            record_replay_problem=lambda *args, **kwargs: record_calls.append(
                (args, kwargs)
            ),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "gmres_candidate"
    assert result.candidate_residual is None
    assert not result.explicit_seed_used
    assert replay.x0_vec == "old"
    assert record_calls == []


def test_sparse_xblock_rescue_acceptance_records_replay_for_host_gmres() -> None:
    replay = SimpleNamespace()
    record_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    candidate = GMRESSolveResult(
        x=jnp.asarray([0.75], dtype=jnp.float64),
        residual_norm=jnp.asarray(0.25, dtype=jnp.float64),
    )
    rhs = jnp.asarray([1.0], dtype=jnp.float64)

    result = accept_sparse_xblock_rescue_candidate(
        context=SparseXBlockRescueAcceptanceContext(
            current_result=current,
            candidate_result=candidate,
            reason="gmres_candidate",
            assembled_host_fp=False,
            use_implicit=False,
            replay_state=replay,
            matvec=_identity,
            rhs=rhs,
            preconditioner=_identity,
            precondition_side="left",
            solver_kind="gmres",
            restart=20,
            maxiter=40,
            record_replay_problem=lambda *args, **kwargs: record_calls.append(
                (args, kwargs)
            ),
        )
    )

    assert result.result is candidate
    assert result.accepted
    assert result.reason == "gmres_candidate_improved"
    assert result.candidate_residual == pytest.approx(0.25)
    assert not result.explicit_seed_used
    assert len(record_calls) == 1
    args, kwargs = record_calls[0]
    assert args == (replay,)
    assert kwargs["matvec_fn"] is _identity
    assert kwargs["b_vec"] is rhs
    assert kwargs["precond_fn"] is _identity
    assert kwargs["x0_vec"] is candidate.x
    assert kwargs["precond_side"] == "left"
    assert kwargs["solver_kind"] == "gmres"
    assert kwargs["restart"] == 20
    assert kwargs["maxiter"] == 40


def test_sparse_xblock_rescue_acceptance_updates_assembled_seed_replay() -> None:
    replay = SimpleNamespace(x0_vec=None)
    record_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    candidate = GMRESSolveResult(
        x=jnp.asarray([0.75], dtype=jnp.float64),
        residual_norm=jnp.asarray(0.25, dtype=jnp.float64),
    )

    result = accept_sparse_xblock_rescue_candidate(
        context=SparseXBlockRescueAcceptanceContext(
            current_result=current,
            candidate_result=candidate,
            reason="seed_accepted",
            assembled_host_fp=True,
            use_implicit=False,
            replay_state=replay,
            matvec=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=_identity,
            precondition_side="left",
            solver_kind="gmres",
            restart=20,
            maxiter=40,
            record_replay_problem=lambda *args, **kwargs: record_calls.append(
                (args, kwargs)
            ),
        )
    )

    assert result.result is candidate
    assert result.accepted
    assert result.reason == "seed_accepted"
    assert result.candidate_residual == pytest.approx(0.25)
    assert result.explicit_seed_used
    assert replay.x0_vec is candidate.x
    assert record_calls == []


def test_fp_xblock_global_correction_stage_accepts_improvement() -> None:
    replay = SimpleNamespace(x0_vec=None)
    marks: list[str] = []
    messages: list[str] = []
    elapsed_values = iter((10.0, 12.5))
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    def safe_preconditioner(preconditioner, *, clip):
        assert preconditioner is _identity
        assert clip == pytest.approx(4.0)
        return preconditioner

    def correction(**kwargs):
        assert kwargs["matvec"] is _identity
        assert kwargs["rhs"] is rhs
        assert kwargs["x0"] is current.x
        assert kwargs["preconditioner"] is _identity
        assert kwargs["steps"] == 3
        assert kwargs["alpha_clip"] == pytest.approx(10.0)
        assert kwargs["min_improvement"] == pytest.approx(0.0)
        return (
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray([0.25], dtype=jnp.float64),
            (0.5,),
            (1.0, 0.5),
        )

    rhs = jnp.asarray([1.0], dtype=jnp.float64)
    result = run_fp_xblock_global_correction_stage(
        context=FPXBlockGlobalCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=rhs,
            preconditioner=_identity,
            preconditioner_label="base",
            steps=3,
            alpha_clip=10.0,
            min_improvement=0.0,
            preconditioner_clip=4.0,
            replay_state=replay,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(elapsed_values),
            mark=marks.append,
            safe_preconditioner=safe_preconditioner,
            correction=correction,
        )
    )

    assert result.accepted
    assert result.reason == "accepted"
    assert result.result.residual_norm == pytest.approx(0.5)
    np.testing.assert_allclose(np.asarray(result.residual_vec), np.asarray([0.25]))
    assert result.preconditioner_label == "base"
    assert result.steps == 3
    assert result.accepted_steps == 2
    assert result.residual_before == pytest.approx(2.0)
    assert result.residual_after == pytest.approx(0.5)
    assert result.improvement_ratio == pytest.approx(4.0)
    assert result.elapsed_s == pytest.approx(2.5)
    np.testing.assert_allclose(np.asarray(replay.x0_vec), np.asarray([1.0]))
    assert marks == [
        "rhs1_fp_xblock_global_correction_start",
        "rhs1_fp_xblock_global_correction_done",
    ]
    assert any("FP x-block global correction" in message for message in messages)
    assert any("accepted" in message for message in messages)


def test_fp_xblock_global_correction_stage_rejects_non_improvement() -> None:
    replay = SimpleNamespace(x0_vec="old")
    marks: list[str] = []
    elapsed_values = iter((1.0, 1.25))
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = run_fp_xblock_global_correction_stage(
        context=FPXBlockGlobalCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=_identity,
            preconditioner_label="sparse_xblock",
            steps=2,
            alpha_clip=0.0,
            min_improvement=0.0,
            preconditioner_clip=1.0e6,
            replay_state=replay,
            emit=None,
            elapsed_s=lambda: next(elapsed_values),
            mark=marks.append,
            safe_preconditioner=lambda preconditioner, **_kwargs: preconditioner,
            correction=lambda **_kwargs: (
                jnp.asarray([1.0], dtype=jnp.float64),
                jnp.asarray([3.0], dtype=jnp.float64),
                (3.0,),
                (1.0,),
            ),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "no_improvement"
    assert result.residual_vec is None
    assert result.residual_before == pytest.approx(2.0)
    assert result.residual_after == pytest.approx(3.0)
    assert result.accepted_steps == 1
    assert result.elapsed_s == pytest.approx(0.25)
    assert replay.x0_vec == "old"
    assert marks == [
        "rhs1_fp_xblock_global_correction_start",
        "rhs1_fp_xblock_global_correction_done",
    ]


def test_fp_xblock_global_correction_stage_reports_missing_preconditioner() -> None:
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    marks: list[str] = []

    result = run_fp_xblock_global_correction_stage(
        context=FPXBlockGlobalCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=None,
            preconditioner_label=None,
            steps=2,
            alpha_clip=0.0,
            min_improvement=0.0,
            preconditioner_clip=1.0,
            replay_state=SimpleNamespace(x0_vec=None),
            emit=None,
            elapsed_s=lambda: 0.0,
            mark=marks.append,
            safe_preconditioner=lambda *_args, **_kwargs: _identity,
            correction=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("correction should not run")
            ),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "missing_preconditioner"
    assert result.steps is None
    assert result.elapsed_s is None
    assert marks == []


def test_fp_xblock_global_correction_stage_reports_exception() -> None:
    replay = SimpleNamespace(x0_vec="old")
    marks: list[str] = []
    messages: list[str] = []
    elapsed_values = iter((2.0, 2.5))
    current = GMRESSolveResult(
        x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = run_fp_xblock_global_correction_stage(
        context=FPXBlockGlobalCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            preconditioner=_identity,
            preconditioner_label="base",
            steps=2,
            alpha_clip=0.0,
            min_improvement=0.0,
            preconditioner_clip=1.0,
            replay_state=replay,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(elapsed_values),
            mark=marks.append,
            safe_preconditioner=lambda preconditioner, **_kwargs: preconditioner,
            correction=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "exception"
    assert result.error == "RuntimeError: boom"
    assert result.elapsed_s == pytest.approx(0.5)
    assert replay.x0_vec == "old"
    assert marks == [
        "rhs1_fp_xblock_global_correction_start",
        "rhs1_fp_xblock_global_correction_failed",
    ]
    assert any("failed" in message for message in messages)


def test_fp_xblock_highx_residual_correction_stage_accepts_improvement() -> None:
    total_size = 30
    replay = SimpleNamespace(x0_vec=None)
    marks: list[str] = []
    messages: list[str] = []
    elapsed_values = iter((5.0, 6.25))
    current = GMRESSolveResult(
        x=jnp.zeros((total_size,), dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    rhs = jnp.ones((total_size,), dtype=jnp.float64)

    def correction(**kwargs):
        assert kwargs["matvec"] is _identity
        assert kwargs["rhs"] is rhs
        assert kwargs["x0"] is current.x
        assert kwargs["steps"] == 2
        assert kwargs["max_directions"] == 4
        assert kwargs["alpha_clip"] == pytest.approx(1.0)
        assert kwargs["rcond"] == pytest.approx(1.0e-8)
        assert kwargs["min_improvement"] == pytest.approx(0.0)
        directions = kwargs["direction_builder"](
            jnp.arange(1, total_size + 1, dtype=jnp.float64)
        )
        direction_names = tuple(name for name, _direction in directions)
        assert direction_names == ("raw_residual", "highx_all", "highx_s0_x0")
        np.testing.assert_allclose(
            np.asarray(directions[1][1])[:6],
            np.arange(1, 7, dtype=np.float64),
        )
        np.testing.assert_allclose(np.asarray(directions[1][1])[6:], 0.0)
        return (
            jnp.ones((total_size,), dtype=jnp.float64),
            jnp.full((total_size,), 0.25, dtype=jnp.float64),
            (0.5,),
            (len(directions),),
            direction_names,
        )

    result = run_fp_xblock_highx_residual_correction_stage(
        context=FPXBlockHighXCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=rhs,
            reduce_full=_identity,
            expand_reduced=_identity,
            total_size=total_size,
            n_species=1,
            n_x=1,
            n_xi=5,
            n_theta=2,
            n_zeta=3,
            n_xi_for_x=(1,),
            host_block_max_env_value="",
            include_factored_blocks=False,
            max_blocks=1,
            steps=2,
            max_directions=4,
            alpha_clip=1.0,
            rcond=1.0e-8,
            min_improvement=0.0,
            include_all=True,
            include_raw=True,
            replay_state=replay,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(elapsed_values),
            mark=marks.append,
            block_factor_allowed=lambda **_kwargs: False,
            correction=correction,
        )
    )

    assert result.accepted
    assert result.reason == "accepted"
    assert result.result.residual_norm == pytest.approx(0.5)
    np.testing.assert_allclose(np.asarray(result.residual_vec), 0.25)
    assert result.residual_before == pytest.approx(2.0)
    assert result.residual_after == pytest.approx(0.5)
    assert result.improvement_ratio == pytest.approx(4.0)
    assert result.elapsed_s == pytest.approx(1.25)
    assert result.direction_count == 3
    assert result.direction_names == ("raw_residual", "highx_all", "highx_s0_x0")
    np.testing.assert_allclose(np.asarray(replay.x0_vec), 1.0)
    assert marks == [
        "rhs1_fp_xblock_highx_residual_correction_start",
        "rhs1_fp_xblock_highx_residual_correction_done",
    ]
    assert any("FP high-x residual-equation correction" in msg for msg in messages)
    assert any("accepted" in msg for msg in messages)


def test_fp_xblock_highx_residual_correction_stage_reports_no_skipped_blocks() -> None:
    total_size = 30
    current = GMRESSolveResult(
        x=jnp.zeros((total_size,), dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    marks: list[str] = []

    result = run_fp_xblock_highx_residual_correction_stage(
        context=FPXBlockHighXCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=jnp.ones((total_size,), dtype=jnp.float64),
            reduce_full=_identity,
            expand_reduced=_identity,
            total_size=total_size,
            n_species=1,
            n_x=1,
            n_xi=5,
            n_theta=2,
            n_zeta=3,
            n_xi_for_x=(1,),
            host_block_max_env_value="",
            include_factored_blocks=False,
            max_blocks=1,
            steps=2,
            max_directions=4,
            alpha_clip=1.0,
            rcond=1.0e-8,
            min_improvement=0.0,
            include_all=True,
            include_raw=True,
            replay_state=SimpleNamespace(x0_vec=None),
            emit=None,
            elapsed_s=lambda: 0.0,
            mark=marks.append,
            block_factor_allowed=lambda **_kwargs: True,
            correction=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("correction should not run")
            ),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "no_skipped_blocks"
    assert result.residual_before is None
    assert result.elapsed_s is None
    assert result.direction_count is None
    assert result.direction_names == ()
    assert marks == [
        "rhs1_fp_xblock_highx_residual_correction_start",
        "rhs1_fp_xblock_highx_residual_correction_done",
    ]


def test_fp_xblock_highx_residual_correction_stage_preserves_exception_residual() -> None:
    total_size = 30
    current = GMRESSolveResult(
        x=jnp.zeros((total_size,), dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )
    marks: list[str] = []
    elapsed_values = iter((3.0, 3.4))

    result = run_fp_xblock_highx_residual_correction_stage(
        context=FPXBlockHighXCorrectionContext(
            current_result=current,
            matvec=_identity,
            rhs=jnp.ones((total_size,), dtype=jnp.float64),
            reduce_full=_identity,
            expand_reduced=_identity,
            total_size=total_size,
            n_species=1,
            n_x=1,
            n_xi=5,
            n_theta=2,
            n_zeta=3,
            n_xi_for_x=(1,),
            host_block_max_env_value="",
            include_factored_blocks=False,
            max_blocks=1,
            steps=2,
            max_directions=4,
            alpha_clip=1.0,
            rcond=1.0e-8,
            min_improvement=0.0,
            include_all=True,
            include_raw=True,
            replay_state=SimpleNamespace(x0_vec=None),
            emit=None,
            elapsed_s=lambda: next(elapsed_values),
            mark=marks.append,
            block_factor_allowed=lambda **_kwargs: False,
            correction=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert result.reason == "exception"
    assert result.error == "RuntimeError: boom"
    assert result.residual_before == pytest.approx(2.0)
    assert result.elapsed_s == pytest.approx(0.4)
    assert marks == [
        "rhs1_fp_xblock_highx_residual_correction_start",
        "rhs1_fp_xblock_highx_residual_correction_failed",
    ]


def _sxblock_context(**overrides: object) -> SparseSXBlockRescueContext:
    values: dict[str, object] = {
        "op": SimpleNamespace(n_species=2),
        "current_result": GMRESSolveResult(
            x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
        ),
        "matvec": _identity,
        "rhs": jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        "reduce_full": _identity,
        "expand_reduced": _identity,
        "drop_tol": 0.0,
        "drop_rel": 0.0,
        "ilu_drop_tol": 0.0,
        "fill_factor": 1.0,
        "preconditioner": _identity,
        "replay_state": SimpleNamespace(x0_vec=None),
        "tol": 1.0e-9,
        "atol": 1.0e-12,
        "restart": 20,
        "maxiter": 80,
        "target": 1.0e-9,
        "precondition_side": "left",
        "solver_kind": "gmres",
        "emit": None,
        "mark": lambda _name: None,
        "seed_builder": lambda **_kwargs: jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        "gmres_solver": lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected gmres polish")
        ),
        "parse_polish_gmres_config": lambda **_kwargs: (5, 6),
        "record_replay_problem": lambda *_args, **_kwargs: None,
    }
    values.update(overrides)
    return SparseSXBlockRescueContext(**values)


def test_sparse_sxblock_rescue_stage_rejects_non_improving_seed() -> None:
    replay = SimpleNamespace(x0_vec="old")
    marks: list[str] = []
    messages: list[str] = []

    result = run_sparse_sxblock_rescue_stage(
        context=_sxblock_context(
            rhs=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            seed_builder=lambda **_kwargs: jnp.asarray([-1.0, -1.0], dtype=jnp.float64),
            replay_state=replay,
            emit=lambda _level, msg: messages.append(msg),
            mark=marks.append,
        )
    )

    assert not result.accepted
    assert not result.polished
    assert result.error is None
    assert result.seed_residual == pytest.approx(np.sqrt(8.0))
    assert result.polish_residual is None
    assert replay.x0_vec == "old"
    assert marks == [
        "rhs1_sparse_precond_build_start",
        "rhs1_sparse_precond_build_done",
        "rhs1_sparse_precond_solve_start",
        "rhs1_sparse_precond_solve_done",
    ]
    assert any("sparse sxblock_tz rescue" in message for message in messages)
    assert any("explicit sxblock seed" in message for message in messages)


def test_sparse_sxblock_rescue_stage_accepts_seed_without_polish() -> None:
    replay = SimpleNamespace(x0_vec=None)

    result = run_sparse_sxblock_rescue_stage(
        context=_sxblock_context(
            rhs=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            target=0.75,
            seed_builder=lambda **_kwargs: jnp.asarray([0.75, 0.75], dtype=jnp.float64),
            replay_state=replay,
            preconditioner=lambda _v: pytest.fail("polish should not run"),
        )
    )

    assert result.accepted
    assert not result.polished
    assert result.seed_residual == pytest.approx(np.sqrt(0.125))
    assert result.polish_residual is None
    np.testing.assert_allclose(np.asarray(result.result.x), np.asarray([0.75, 0.75]))
    np.testing.assert_allclose(np.asarray(replay.x0_vec), np.asarray([0.75, 0.75]))


def test_sparse_sxblock_rescue_stage_accepts_polish_and_records_replay() -> None:
    replay = SimpleNamespace(x0_vec=None)
    record_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    parse_calls: list[dict[str, object]] = []
    gmres_calls: list[dict[str, object]] = []
    rhs = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    def parse_config(**kwargs):
        parse_calls.append(kwargs)
        return 7, 8

    def gmres_solver(**kwargs):
        gmres_calls.append(kwargs)
        return np.asarray([0.9, 0.9], dtype=np.float64), 0.0, (0.2,)

    result = run_sparse_sxblock_rescue_stage(
        context=_sxblock_context(
            rhs=rhs,
            target=1.0e-9,
            restart=30,
            maxiter=200,
            seed_builder=lambda **_kwargs: jnp.asarray([0.75, 0.75], dtype=jnp.float64),
            replay_state=replay,
            preconditioner=_identity,
            parse_polish_gmres_config=parse_config,
            gmres_solver=gmres_solver,
            record_replay_problem=lambda *args, **kwargs: record_calls.append(
                (args, kwargs)
            ),
        )
    )

    assert result.accepted
    assert result.polished
    assert result.seed_residual == pytest.approx(np.sqrt(0.125))
    assert result.polish_residual == pytest.approx(np.sqrt(0.02))
    assert result.polish_restart == 7
    assert result.polish_maxiter == 8
    np.testing.assert_allclose(np.asarray(result.result.x), np.asarray([0.9, 0.9]))
    assert parse_calls == [
        {
            "restart_env_name": "SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_RESTART",
            "maxiter_env_name": "SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_MAXITER",
            "default_restart": 30,
            "default_maxiter": 120,
        }
    ]
    assert len(gmres_calls) == 1
    assert gmres_calls[0]["matvec"] is _identity
    assert gmres_calls[0]["b"] is rhs
    assert gmres_calls[0]["preconditioner"] is _identity
    assert gmres_calls[0]["restart"] == 7
    assert gmres_calls[0]["maxiter"] == 8
    assert len(record_calls) == 1
    args, kwargs = record_calls[0]
    assert args == (replay,)
    assert kwargs["matvec_fn"] is _identity
    assert kwargs["b_vec"] is rhs
    assert kwargs["precond_fn"] is _identity
    assert kwargs["precond_side"] == "left"
    assert kwargs["solver_kind"] == "gmres"
    assert kwargs["restart"] == 7
    assert kwargs["maxiter"] == 8
    np.testing.assert_allclose(np.asarray(kwargs["x0_vec"]), np.asarray([0.9, 0.9]))


def test_sparse_sxblock_rescue_stage_reports_seed_exception() -> None:
    messages: list[str] = []
    current = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = run_sparse_sxblock_rescue_stage(
        context=_sxblock_context(
            current_result=current,
            seed_builder=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            emit=lambda _level, msg: messages.append(msg),
        )
    )

    assert result.result is current
    assert not result.accepted
    assert not result.polished
    assert result.error == "RuntimeError: boom"
    assert result.seed_residual is None
    assert any("sxblock_sparse: failed" in message for message in messages)


def test_fortran_reduced_xblock_krylov_policy_defaults_and_counter() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(env={})

    assert setup.side_env == ""
    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 25
    assert int(setup.mv_count) == 0
    setup.mv_count.increment()
    assert int(setup.mv_count) == 1
    assert setup.messages == ()


def test_fortran_reduced_xblock_krylov_policy_normalizes_aliases() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(
        env={
            "SFINCS_JAX_GMRES_PRECONDITION_SIDE": "right",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM": "explicit_left",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "gcrotmk-scipy",
            "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "7",
        }
    )

    assert setup.precondition_side == "right"
    assert setup.pc_form == "explicit_left"
    assert setup.krylov_method == "gcrotmk"
    assert setup.progress_every == 7
    assert setup.messages == ()


def test_fortran_reduced_xblock_krylov_policy_falls_back_invalid_values() -> None:
    setup = resolve_fortran_reduced_xblock_krylov_policy(
        env={
            "SFINCS_JAX_GMRES_PRECONDITION_SIDE": "bad-side",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM": "bad-form",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "bad-method",
            "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "-5",
        }
    )

    assert setup.side_env == "bad-side"
    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 0
    assert setup.messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "ignoring unknown SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV="
            "'bad_method'; using gmres",
        ),
    )


def test_fortran_reduced_xblock_krylov_setup_builds_active_matvec_and_wrapper() -> None:
    messages: list[str] = []
    setup = build_fortran_reduced_xblock_krylov_setup(
        context=FortranReducedXBlockKrylovSetupContext(
            op=SimpleNamespace(total_size=4),
            rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
            xblock_use_active_dof=True,
            active_idx=jnp.asarray([0, 2], dtype=jnp.int32),
            full_to_active=jnp.asarray([0, -1, 1, -1], dtype=jnp.int32),
            reduce_full_with_indices=lambda v, idx: v[idx],
            expand_reduced_with_map=lambda v, fmap: jnp.where(
                fmap >= 0,
                v[jnp.maximum(fmap, 0)],
                0.0,
            ),
            operator_matvec=lambda v: v + 10.0,
            base_preconditioner=lambda v: 2.0 * v,
            elapsed_s=lambda: 3.5,
            emit=lambda _level, msg: messages.append(msg),
            env={
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV": "bad-method",
                "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "2",
            },
        )
    )

    assert setup.precondition_side == "left"
    assert setup.pc_form == "scipy_left"
    assert setup.krylov_method == "gmres"
    assert setup.progress_every == 2
    assert setup.matvec_no_count(jnp.asarray([5.0, 7.0])).tolist() == [15.0, 17.0]
    assert setup.preconditioner(jnp.asarray([1.0, 3.0])).tolist() == [2.0, 6.0]
    assert int(setup.mv_count) == 0
    setup.matvec(jnp.asarray([0.0, 1.0]))
    setup.matvec(jnp.asarray([2.0, 3.0]))
    assert int(setup.mv_count) == 2
    assert any("using gmres" in message for message in messages)
    assert any("fortran_reduced_pc_gmres xblock matvecs=2" in message for message in messages)
    assert not any("active-DOF reduction" in message for message in messages)


def test_fortran_reduced_xblock_initial_seed_policy_parses_controls() -> None:
    setup = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_INITIAL_SEED": "off",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "-3",
        }
    )

    assert not setup.enabled
    assert setup.refine_steps == 2
    assert setup.accept_ratio == 0.0


def test_fortran_reduced_xblock_initial_seed_accepts_refined_seed() -> None:
    policy = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "2",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "0.2",
        }
    )
    times = iter([5.0, 5.75])
    result = apply_fortran_reduced_xblock_initial_seed(
        policy=policy,
        rhs=jnp.asarray([4.0, 0.0]),
        rhs_norm=4.0,
        x0=None,
        preconditioner=lambda v: 0.25 * v,
        matvec_no_count=lambda v: 2.0 * v,
        elapsed_s=lambda: next(times),
    )

    assert result.used
    assert result.refines_performed == 2
    assert result.residual_norm == pytest.approx(0.5)
    assert result.improvement_ratio == pytest.approx(8.0)
    assert result.elapsed_s == pytest.approx(0.75)
    assert result.x0 is not None
    assert result.x0.tolist() == pytest.approx([1.75, 0.0])
    assert any("accepted=True" in message for _, message in result.messages)


def test_fortran_reduced_xblock_initial_seed_rejects_weak_seed() -> None:
    policy = resolve_fortran_reduced_xblock_initial_seed_policy(
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES": "3",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO": "0.5",
        }
    )
    result = apply_fortran_reduced_xblock_initial_seed(
        policy=policy,
        rhs=jnp.asarray([2.0, 0.0]),
        rhs_norm=2.0,
        x0=None,
        preconditioner=lambda v: jnp.zeros_like(v),
        matvec_no_count=lambda v: v,
        elapsed_s=lambda: 0.0,
    )

    assert not result.used
    assert result.x0 is None
    assert result.refines_performed == 0
    assert result.residual_norm == pytest.approx(2.0)
    assert result.improvement_ratio == pytest.approx(1.0)
    assert any("accepted=False" in message for _, message in result.messages)


def test_fortran_reduced_xblock_krylov_solve_runs_gmres_and_true_residual() -> None:
    messages: list[str] = []
    counter = MatvecCounter(0)
    times = iter([1.0, 1.5, 2.0, 2.25])

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        counter.increment()
        return v

    def gmres_solver(**kwargs):
        assert kwargs["preconditioner"] is None
        kwargs["progress_callback"](2, 0.25)
        return np.asarray([1.0, 2.0]), 99.0, [0.5, 0.25]

    def unused_solver(**_kwargs):
        raise AssertionError("unused")

    result = run_fortran_reduced_xblock_krylov_solve(
        context=FortranReducedXBlockKrylovSolveContext(
            matvec=matvec,
            rhs=jnp.asarray([1.0, 2.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            method="gmres",
            pc_form="scipy_left",
            restart=8,
            maxiter=3,
            tol=1.0e-8,
            atol=1.0e-10,
            target=1.0e-9,
            precondition_side="none",
            progress_every=2,
            mv_count=counter,
            explicit_left_solver=unused_solver,
            gmres_solver=gmres_solver,
            lgmres_solver=unused_solver,
            gcrotmk_solver=unused_solver,
            bicgstab_solver=unused_solver,
        ),
        x0=None,
    )

    assert result.x.tolist() == [1.0, 2.0]
    assert result.residual_norm == pytest.approx(0.0)
    assert result.history == (0.5, 0.25)
    assert result.solve_s == pytest.approx(1.0)
    assert int(counter) == 1
    assert any("iters=2 ksp_residual=2.500000e-01" in message for message in messages)
    assert any("matvecs=1 residual=0.000000e+00" in message for message in messages)


def test_fortran_reduced_xblock_krylov_solve_explicit_left_reports_pc_residual() -> None:
    messages: list[str] = []
    counter = MatvecCounter(0)

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        counter.increment()
        return v

    def explicit_left_solver(**kwargs):
        assert kwargs["preconditioner"] is _identity
        kwargs["progress_callback"](1, 0.125)
        return np.asarray([3.0]), 4.0, 0.125, [0.125]

    def unused_solver(**_kwargs):
        raise AssertionError("unused")

    result = run_fortran_reduced_xblock_krylov_solve(
        context=FortranReducedXBlockKrylovSolveContext(
            matvec=matvec,
            rhs=jnp.asarray([3.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: 0.0,
            method="gmres",
            pc_form="explicit_left",
            restart=4,
            maxiter=2,
            tol=1.0e-8,
            atol=1.0e-10,
            target=1.0e-9,
            precondition_side="left",
            progress_every=1,
            mv_count=counter,
            explicit_left_solver=explicit_left_solver,
            gmres_solver=unused_solver,
            lgmres_solver=unused_solver,
            gcrotmk_solver=unused_solver,
            bicgstab_solver=unused_solver,
        ),
        x0=None,
    )

    assert result.residual_norm == pytest.approx(0.0)
    assert result.preconditioned_residual_norm == pytest.approx(0.125)
    assert any("preconditioned_residual=1.250000e-01" in message for message in messages)


def test_fortran_reduced_xblock_result_metadata_formats_branch_payload() -> None:
    state = {
        "op": SimpleNamespace(total_size=12),
        "fortran_reduced_xblock_accepted_converged": True,
        "history": (0.5, 0.25),
        "mv_count": MatvecCounter(7),
        "pc_restart": 8,
        "pc_maxiter": 3,
        "fortran_reduced_sparse_pc_backend_reason": "auto_large_full_fp",
        "fortran_reduced_xblock_min_size": 100,
        "preconditioner_x": 4,
        "preconditioner_x_min_l": 2,
        "preconditioner_xi": 1,
        "preconditioner_species": 0,
        "xblock_preconditioner_xi": 1,
        "force_assembled_host_fp": True,
        "xblock_krylov_method": "gmres",
        "seed_enabled": True,
        "seed_used": True,
        "seed_residual_norm": 1.0e-4,
        "seed_improvement_ratio": 10.0,
        "seed_accept_ratio": 1.0,
        "seed_refine_steps": 2,
        "seed_refines_performed": 1,
        "moment_schur_enabled": True,
        "moment_schur_built": True,
        "moment_schur_used": False,
        "moment_schur_reason": "probe_not_reduced",
        "moment_schur_metadata": {
            "mode": "additive",
            "rank": 3,
            "extra_size": 2,
            "setup_s": 0.25,
            "expected_size": 10,
            "rcond": 1.0e-12,
            "singular_value_proxy": (1.0, 0.1),
            "device_resident": False,
        },
        "moment_schur_probe_residual_before": 2.0,
        "moment_schur_probe_residual_after": 1.5,
        "moment_schur_probe_improvement_ratio": 0.75,
        "moment_schur_stats": {"applies": 4, "base_applies": 5},
        "global_coupling_enabled": True,
        "global_coupling_built": True,
        "global_coupling_metadata": {
            "mode": "multiplicative",
            "load_basis_size": 6,
            "basis_size": 5,
            "rank": 4,
            "setup_s": 0.5,
            "setup_budget_s": 1.0,
            "setup_budget_reached": False,
            "rcond": 1.0e-11,
            "smoother": "xblock",
            "basis_names": ("rhs", "fsavg"),
        },
        "global_coupling_stats": {"applies": 8, "coarse_applies": 9},
        "xblock_drop_tol": 0.0,
        "xblock_drop_rel": 1.0e-8,
        "xblock_ilu_drop_tol": 1.0e-4,
        "xblock_fill_factor": 10.0,
        "sparse_pc_use_active_dof": False,
        "sparse_pc_linear_size": 10,
        "sparse_pc_fp_dense_velocity_block": None,
        "setup_s": 0.75,
        "solve_s": 1.25,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
        "pc_factor_s": 0.5,
        "target": 0.2,
        "residual_norm_sparse_pc": 0.1,
        "fortran_reduced_xblock_factor_quality_rejected": False,
    }

    metadata = fortran_reduced_xblock_result_metadata(state)

    assert metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert metadata["accepted_converged"] is True
    assert metadata["iterations"] == 2
    assert metadata["matvecs"] == 7
    assert metadata["sparse_pc_backend"] == "xblock"
    assert metadata["sparse_pc_xblock_initial_seed_used"] is True
    assert metadata["sparse_pc_xblock_moment_schur_rank"] == 3
    assert metadata["sparse_pc_xblock_global_coupling_basis_names"] == ("rhs", "fsavg")
    assert metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.5)
    assert metadata["sparse_pc_factor_quality_rejected"] is False


def _fortran_reduced_xblock_solve_state() -> _DefaultSparsePCDriverState:
    return _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=4),
            "atol": 0.25,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "target": 0.5,
            "mv_count": MatvecCounter(4),
            "pc_restart": 8,
            "pc_maxiter": 3,
            "fortran_reduced_sparse_pc_backend_reason": "auto_large_full_fp",
            "fortran_reduced_xblock_min_size": 100,
            "preconditioner_x": 4,
            "preconditioner_x_min_l": 2,
            "preconditioner_xi": 1,
            "preconditioner_species": 0,
            "xblock_preconditioner_xi": 1,
            "force_assembled_host_fp": True,
            "xblock_krylov_method": "gmres",
            "seed_enabled": True,
            "seed_used": False,
            "seed_residual_norm": None,
            "seed_improvement_ratio": None,
            "seed_accept_ratio": 1.0,
            "seed_refine_steps": 2,
            "seed_refines_performed": 0,
            "moment_schur_enabled": True,
            "moment_schur_built": False,
            "moment_schur_used": False,
            "moment_schur_reason": "disabled",
            "moment_schur_metadata": {},
            "moment_schur_probe_residual_before": None,
            "moment_schur_probe_residual_after": None,
            "moment_schur_probe_improvement_ratio": None,
            "moment_schur_stats": {},
            "global_coupling_enabled": True,
            "global_coupling_built": False,
            "global_coupling_metadata": {},
            "global_coupling_stats": {},
            "xblock_drop_tol": 0.0,
            "xblock_drop_rel": 1.0e-8,
            "xblock_ilu_drop_tol": 1.0e-4,
            "xblock_fill_factor": 10.0,
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": 2,
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.75,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 3.0),
            "pc_factor_s": 0.5,
        }
    )


def _fortran_reduced_xblock_result() -> SparsePCGMRESResult:
    return SparsePCGMRESResult(
        x=np.asarray([1.0, 2.0]),
        residual_norm=0.2,
        preconditioned_residual_norm=np.nan,
        history=(1.0, 0.4, 0.2),
        solve_s=1.25,
    )


def test_fortran_reduced_xblock_final_payload_uses_explicit_context() -> None:
    state = _fortran_reduced_xblock_solve_state()

    payload = fortran_reduced_xblock_final_payload(
        FortranReducedXBlockFinalPayloadContext(
            diagnostic_state=state,
            result=_fortran_reduced_xblock_result(),
            atol=0.25,
            tol=0.0,
            rhs_norm=1.0,
            target=0.5,
        ),
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.2)
    assert payload.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["iterations"] == 3
    assert payload.metadata["sparse_pc_factor_quality_rejected"] is False
    assert payload.metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.4)


def test_fortran_reduced_xblock_final_payload_from_solve_state_sets_gates() -> None:
    state = _fortran_reduced_xblock_solve_state()

    payload = fortran_reduced_xblock_final_payload_from_solve_state(
        state,
        result=_fortran_reduced_xblock_result(),
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.2)
    assert payload.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["iterations"] == 3
    assert payload.metadata["sparse_pc_factor_quality_rejected"] is False
    assert payload.metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.4)


def test_fortran_reduced_xblock_moment_schur_stage_accepts_probe() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
        },
    )
    messages: list[str] = []
    stats = {"applies": 2, "base_applies": 3}

    def builder(**_kwargs):
        return (lambda v: v), {"rank": 1}, stats

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=lambda v: jnp.zeros_like(v),
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 2.0,
            emit=lambda _level, msg: messages.append(msg),
            builder=builder,
        )
    )

    assert result.built
    assert result.used
    assert result.reason == "probe_reduced"
    assert result.metadata["setup_s"] == pytest.approx(0.0)
    assert result.probe_residual_after == pytest.approx(0.0)
    assert result.stats is stats
    stats["applies"] = 7
    assert result.stats["applies"] == 7
    assert result.preconditioner(jnp.asarray([4.0])).tolist() == [4.0]
    assert any("constraint1 moment-Schur accepted" in message for message in messages)


def test_fortran_reduced_xblock_moment_schur_stage_rejects_probe() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
        },
    )
    def base(v: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * v

    def builder(**_kwargs):
        return (lambda v: jnp.zeros_like(v)), {}, {}

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=base,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert result.built
    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.preconditioner is base
    assert result.probe_improvement_ratio == pytest.approx(1.0)


def test_fortran_reduced_xblock_moment_schur_stage_records_failure() -> None:
    policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1"},
    )

    def builder(**_kwargs):
        raise RuntimeError("boom")

    result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=_identity,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="left",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert not result.used
    assert "RuntimeError: boom" in str(result.reason)
    assert result.metadata["error"] == "RuntimeError: boom"
    assert result.preconditioner is _identity


def test_fortran_reduced_xblock_global_coupling_stage_builds_and_records_stats() -> None:
    policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1"},
    )
    messages: list[str] = []
    stats = {"applies": 4, "coarse_applies": 5}

    def builder(**kwargs):
        assert kwargs["expected_size"] == 3
        assert kwargs["mode"] == "additive"
        return (lambda v: 2.0 * v), {"rank": 2}, stats

    result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0, 3.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=3,
            policy=policy,
            elapsed_s=lambda: 1.0,
            emit=lambda _level, msg: messages.append(msg),
            builder=builder,
        )
    )

    assert result.built
    assert result.metadata["rank"] == 2
    assert result.metadata["setup_s"] == pytest.approx(0.0)
    assert result.stats is stats
    stats["applies"] = 6
    assert result.stats["applies"] == 6
    assert result.preconditioner(jnp.asarray([3.0])).tolist() == [6.0]
    assert any("global-coupling build start" in message for message in messages)


def test_fortran_reduced_xblock_global_coupling_stage_without_builder_stays_disabled() -> None:
    policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS": "8",
        },
    )
    result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=2,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert "moved to research QI branch" in result.metadata["error"]


def test_fortran_reduced_xblock_global_coupling_stage_records_failure() -> None:
    policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={"SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1"},
    )

    def builder(**_kwargs):
        raise ValueError("bad basis")

    result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=1,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert result.metadata["error"] == "ValueError: bad basis"
    assert result.stats == {"applies": 0, "coarse_applies": 0}


def test_fortran_reduced_xblock_moment_schur_policy_defaults_disabled() -> None:
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="left",
        env={},
    )

    assert not setup.default_candidate
    assert not setup.default_blocked_by_compact_factors
    assert not setup.enabled
    assert setup.rcond == pytest.approx(1.0e-12)
    assert not setup.probe_enabled
    assert setup.probe_min_improvement == 0.0
    assert setup.messages == ()


def test_fortran_reduced_xblock_moment_schur_policy_uses_fortran_env_over_generic() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "3e-9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND": "2e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_MIN_IMPROVEMENT": "0.4",
        },
    )

    assert setup.enabled
    assert setup.rcond == pytest.approx(2.0e-8)
    assert setup.probe_enabled
    assert setup.probe_min_improvement == pytest.approx(0.4)
    assert any("moment-Schur build start" in message for _, message in setup.messages)


def test_fortran_reduced_xblock_moment_schur_policy_falls_back_to_generic_rcond() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side="none",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "3e-9",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND": "bad",
        },
    )

    assert setup.enabled
    assert setup.rcond == pytest.approx(3.0e-9)
    assert setup.messages == ()


def test_fortran_reduced_xblock_global_coupling_policy_defaults_off() -> None:
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="right",
        env={},
    )

    assert not setup.enabled
    assert not setup.should_build
    assert not setup.use_device_builder
    assert setup.mode == "additive"
    assert setup.max_directions == 96
    assert setup.fsavg_lmax == 12
    assert setup.angular_lmax == 2
    assert setup.max_extra_units == 8
    assert setup.rcond == pytest.approx(1.0e-11)
    assert setup.include_rhs
    assert setup.setup_max_s == 0.0


def test_fortran_reduced_xblock_global_coupling_policy_parses_controls() -> None:
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS": "11",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_FSAVG_LMAX": "4",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_ANGULAR_LMAX": "5",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_EXTRA_UNITS": "6",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_RCOND": "3e-8",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_INCLUDE_RHS": "0",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S": "9.5",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 11
    assert setup.fsavg_lmax == 4
    assert setup.angular_lmax == 5
    assert setup.max_extra_units == 6
    assert setup.rcond == pytest.approx(3.0e-8)
    assert not setup.include_rhs
    assert setup.setup_max_s == pytest.approx(9.5)


def test_fortran_reduced_xblock_global_coupling_policy_generic_mode_and_no_side() -> (
    None
):
    setup = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side="none",
        env={
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE": "right_additive",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS": "bad",
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S": "-1",
        },
    )

    assert setup.enabled
    assert not setup.should_build
    assert setup.mode == "right_additive"
    assert setup.max_directions == 96
    assert setup.setup_max_s == 0.0


def test_sparse_pc_entry_policy_classifies_pas_er_and_active_dof() -> None:
    def parse_config(**kwargs):
        assert kwargs["default_restart"] == 50
        assert kwargs["default_maxiter"] == 100
        return 50, 100

    setup = resolve_sparse_pc_entry_policy(
        op=_op(pas=True, constraint_scheme=2, n_zeta=1, n_species=1),
        solve_method_kind="sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=False,
        xblock_active_dof_requested=False,
        active_maps_available=False,
        use_dkes=True,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=True,
        er_abs_sparse_pc=0.2,
        restart=50,
        maxiter=80,
        parse_polish_gmres_config=parse_config,
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"] - 5,
        env={"SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK": "1"},
    )

    assert setup.constrained_pas_pc
    assert setup.tokamak_pas_er_pc
    assert not setup.tokamak_pas_noer_pc
    assert setup.sparse_pc_use_active_dof
    assert setup.sparse_pc_fp_dense_velocity_block is True
    assert setup.pc_restart == 45
    assert setup.pc_maxiter == 100


def test_sparse_pc_entry_policy_classifies_xblock_active_maps() -> None:
    setup = resolve_sparse_pc_entry_policy(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3, n_species=2),
        solve_method_kind="xblock_sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=True,
        xblock_active_dof_requested=True,
        active_maps_available=True,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        er_abs_sparse_pc=0.0,
        restart=10,
        maxiter=None,
        parse_polish_gmres_config=lambda **_kwargs: (20, 400),
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"],
        env={},
    )

    assert setup.xblock_sparse_pc
    assert setup.xblock_use_active_dof
    assert not setup.sparse_pc_use_active_dof
    assert setup.pc_restart == 20
    assert setup.pc_maxiter == 400


def test_xblock_sparse_pc_setup_resolves_host_assembly_and_device_fallback() -> None:
    fallback_calls: list[dict[str, object]] = []

    def fallback_decision(**kwargs):
        fallback_calls.append(kwargs)
        return SimpleNamespace(
            used=True,
            ignored_env=False,
            mode="host",
            reason="forced",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value="auto",
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=True,
        )

    setup = resolve_xblock_sparse_pc_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        preconditioner_species=0,
        preconditioner_xi=0,
        active_size=1000,
        lower_fill_mode=lambda value: ("force", value == "bad"),
        species_decoupled_for_host_assembly=lambda **_kwargs: True,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: (
            "gmres_jax" if value == "gmres_jax" else "gmres",
            False,
        ),
        device_host_fallback_decision=fallback_decision,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL": "1e-5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL": "force",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "gmres_jax",
            "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK": "host",
        },
    )

    assert setup.xblock_drop_tol == pytest.approx(1.0e-5)
    assert setup.xblock_lower_fill_mode == "force"
    assert setup.xblock_preconditioner_xi == 1
    assert setup.force_assembled_host_fp
    assert setup.xblock_assembled_host_fp
    assert setup.xblock_krylov_env_requested == "gmres_jax"
    assert setup.xblock_krylov_env == "auto"
    assert setup.xblock_krylov_requested == "gmres"
    assert not setup.xblock_device_krylov_requested
    assert setup.xblock_device_host_fallback_decision.used
    assert any(
        "non-autodiff host x-block fallback" in message for _, message in setup.messages
    )
    assert fallback_calls[0]["requested_krylov_method"] == "gmres_jax"


def test_xblock_sparse_pc_side_policy_parses_jax_factors_and_forces_fgmres_right_pc() -> (
    None
):
    def side_policy(**kwargs):
        assert kwargs["krylov_env_value"] == "fgmres_jax"
        assert kwargs["full_fp_3d_pc"] is True
        return SimpleNamespace(
            precondition_side="left",
            default_right_preconditioned=False,
            krylov_method="fgmres_jax",
            gmres_restart=33,
            restart_capped=True,
            ignored_krylov_env=True,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        xblock_krylov_env="fgmres_jax",
        pc_restart=50,
        pc_restart_env="50",
        tokamak_fp_er_pc=False,
        active_size=4000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT": "compact-csr",
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY": "jacobi",
        },
    )

    assert setup.xblock_jax_factors
    assert not setup.xblock_jax_factors_requested
    assert setup.xblock_device_krylov_forced_jax_factors
    assert setup.xblock_jax_factor_format == "csr"
    assert setup.xblock_jax_factor_apply == "diagonal"
    assert setup.precondition_side == "right"
    assert setup.xblock_device_fgmres_forced_right_pc
    assert setup.pc_restart == 33
    assert setup.xblock_default_restart_capped
    assert any("ignoring unknown" in message for _, message in setup.messages)


def test_xblock_sparse_pc_side_policy_uses_host_factors_when_fallback_is_used() -> None:
    def side_policy(**kwargs):
        return SimpleNamespace(
            precondition_side="right",
            default_right_preconditioned=True,
            krylov_method=kwargs["krylov_env_value"],
            gmres_restart=kwargs["requested_restart"],
            restart_capped=False,
            ignored_krylov_env=False,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=True),
        xblock_krylov_env="gmres",
        pc_restart=20,
        pc_restart_env="",
        tokamak_fp_er_pc=False,
        active_size=2000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS": "1"},
    )

    assert setup.xblock_jax_factors_requested
    assert not setup.xblock_jax_factors
    assert setup.xblock_jax_factor_format == "padded"
    assert setup.xblock_jax_factor_apply == "exact"
    assert any(
        "requires host sparse factors" in message for _, message in setup.messages
    )


def test_xblock_sparse_pc_branch_setup_composes_fallback_side_and_reuse() -> None:
    def fallback_decision(**kwargs):
        return SimpleNamespace(
            used=False,
            ignored_env=False,
            mode="auto",
            reason="ok",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value=kwargs["env_value"],
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=False,
        )

    def side_policy(**kwargs):
        assert kwargs["krylov_env_value"] == "fgmres_jax"
        return SimpleNamespace(
            precondition_side="left",
            default_right_preconditioned=False,
            krylov_method="fgmres_jax",
            gmres_restart=41,
            restart_capped=True,
            ignored_krylov_env=True,
        )

    setup = resolve_xblock_sparse_pc_branch_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        preconditioner_species=1,
        preconditioner_xi=0,
        active_size=4000,
        pc_restart=20,
        pc_restart_env="",
        tokamak_fp_er_pc=False,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        lower_fill_mode=lambda value: ("force", value == "bad"),
        species_decoupled_for_host_assembly=lambda **_kwargs: False,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: (
            "fgmres_jax" if value == "fgmres_jax" else "gmres",
            False,
        ),
        device_host_fallback_decision=fallback_decision,
        resolve_xblock_policy=side_policy,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "fgmres_jax",
        },
    )

    assert setup.xblock_preconditioner_xi == 1
    assert setup.xblock_device_krylov_requested
    assert setup.precondition_side == "right"
    assert setup.xblock_device_fgmres_forced_right_pc
    assert setup.pc_restart == 41
    assert setup.xblock_default_restart_capped
    assert setup.xblock_jax_factors
    assert any("ignoring unknown" in message for _, message in setup.messages)
    assert any(
        "building jax x-block preconditioner" in message
        for _, message in setup.messages
    )


def test_xblock_local_preconditioner_delegates_factor_build_with_controls() -> None:
    calls: list[dict[str, object]] = []
    times = iter([2.0, 3.5])

    def preconditioner(v):
        return 2.0 * v

    def build_preconditioner(**kwargs):
        calls.append(kwargs)
        return preconditioner

    op = object()
    result = build_xblock_local_preconditioner(
        skip_factors=False,
        elapsed_s=lambda: next(times),
        build_preconditioner=build_preconditioner,
        op=op,
        build_jax_factors=True,
        preconditioner_species=2,
        preconditioner_xi=3,
        drop_tol=1.0e-5,
        drop_rel=1.0e-6,
        ilu_drop_tol=1.0e-4,
        fill_factor=8.0,
        force_assembled_host_fp=True,
        emit=None,
    )

    assert result.built
    assert result.factor_s == pytest.approx(1.5)
    np.testing.assert_allclose(np.asarray(result.preconditioner(jnp.asarray([2.0]))), np.asarray([4.0]))
    assert calls == [
        {
            "op": op,
            "build_jax_factors": True,
            "preconditioner_species": 2,
            "preconditioner_xi": 3,
            "drop_tol": 1.0e-5,
            "drop_rel": 1.0e-6,
            "ilu_drop_tol": 1.0e-4,
            "fill_factor": 8.0,
            "force_assembled_host_fp": True,
            "emit": None,
        }
    ]


def test_xblock_krylov_matvec_setup_reduces_active_dofs_and_counts_progress() -> None:
    messages: list[str] = []
    active_idx = jnp.asarray([0, 2], dtype=jnp.int32)
    full_to_active = jnp.asarray([0, -1, 1, -1], dtype=jnp.int32)
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=4),
        rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        xblock_use_active_dof=True,
        active_idx=active_idx,
        full_to_active=full_to_active,
        reduce_full_with_indices=lambda v, idx: v[idx],
        expand_reduced_with_map=lambda v, fmap: jnp.where(
            fmap >= 0, v[jnp.maximum(fmap, 0)], 0.0
        ),
        operator_matvec=lambda v: 2.0 * v,
        elapsed_s=lambda: 12.5,
        emit=lambda _level, msg: messages.append(msg),
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "2"},
    )

    assert setup.xblock_linear_size == 2
    assert setup.xblock_active_idx_np.tolist() == [0, 2]
    assert setup.xblock_rhs.tolist() == [1.0, 3.0]
    assert setup.matvec_no_count(jnp.asarray([5.0, 7.0])).tolist() == [10.0, 14.0]
    assert int(setup.mv_count) == 0
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert int(setup.mv_count) == 1
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [4.0, 6.0]
    assert int(setup.mv_count) == 2
    assert any("active-DOF reduction" in message for _, message in setup.messages)
    assert any("matvecs=2" in message for message in messages)


def test_xblock_krylov_matvec_setup_full_space_is_identity_mapping() -> None:
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=3),
        rhs=jnp.asarray([1.0, 2.0, 3.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v + 1.0,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "bad"},
    )

    assert setup.progress_every == 25
    assert setup.xblock_linear_size == 3
    assert setup.xblock_active_idx_np is None
    assert setup.reduce_full(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.expand_reduced(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.matvec(jnp.asarray([1.0, 2.0, 3.0])).tolist() == [2.0, 3.0, 4.0]
    assert int(setup.mv_count) == 1


def test_xblock_krylov_matvec_setup_reuses_counter_and_custom_progress_label() -> None:
    messages: list[str] = []
    counter = MatvecCounter(3)
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=4),
        rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        xblock_use_active_dof=True,
        active_idx=jnp.asarray([1, 3], dtype=jnp.int32),
        full_to_active=jnp.asarray([-1, 0, -1, 1], dtype=jnp.int32),
        reduce_full_with_indices=lambda v, idx: v[idx],
        expand_reduced_with_map=lambda v, fmap: jnp.where(
            fmap >= 0,
            v[jnp.maximum(fmap, 0)],
            0.0,
        ),
        operator_matvec=lambda v: 4.0 * v,
        elapsed_s=lambda: 8.25,
        emit=lambda _level, msg: messages.append(msg),
        progress_every=2,
        mv_count=counter,
        progress_label="fortran_reduced_pc_gmres xblock",
        emit_active_message=False,
    )

    assert setup.messages == ()
    assert setup.mv_count is counter
    assert int(counter) == 3
    assert setup.xblock_rhs.tolist() == [2.0, 4.0]
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [8.0, 12.0]
    assert int(counter) == 4
    assert any(
        "fortran_reduced_pc_gmres xblock matvecs=4" in message
        for message in messages
    )


def test_xblock_assembled_equilibration_setup_builds_row_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, -1.0], [0.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 3.0,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE": "1"},
    )

    assert setup.row_enabled
    assert setup.row_built
    assert not setup.col_enabled
    assert not setup.col_built
    assert setup.row_metadata["norm"] == "linf"
    assert np.asarray(setup.row_scale).tolist() == pytest.approx([0.5, 0.25])
    assert np.asarray(setup.inv_row_scale).tolist() == pytest.approx([2.0, 4.0])
    assert any(
        "assembled row equilibration built" in message for _, message in setup.messages
    )


def test_xblock_assembled_equilibration_setup_builds_row_and_column_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, 0.0], [1.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 5.0,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM": "l1",
        },
    )

    assert setup.row_enabled
    assert setup.row_built
    assert setup.col_enabled
    assert setup.col_built
    assert setup.row_metadata["norm"] == "l1"
    assert setup.col_metadata["norm"] == "l1"
    assert setup.row_metadata["column_equilibration"] is True
    assert np.all(np.isfinite(np.asarray(setup.col_scale)))
    assert np.all(np.asarray(setup.col_scale) > 0.0)
    assert any(
        "assembled column equilibration built" in message
        for _, message in setup.messages
    )


def test_xblock_assembled_operator_preflight_uses_full_pattern_when_under_budget() -> (
    None
):
    full_pattern = object()
    full_summary = SimpleNamespace(nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0)

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres_jax",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: full_pattern,
        active_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        summarize_pattern=lambda _op, pattern: full_summary
        if pattern is full_pattern
        else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is full_pattern
    assert setup.summary is full_summary
    assert setup.device_enabled
    assert not setup.metadata["preflight_rejected"]
    assert setup.metadata["preflight_scope"] == "full"


def test_xblock_assembled_operator_preflight_uses_active_pattern_scope() -> None:
    full_summary = SimpleNamespace(
        nnz=1000, shape=(100, 100), max_row_nnz=20, avg_row_nnz=10.0
    )
    active_summary = SimpleNamespace(
        nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0
    )
    active_pattern = object()

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=np.asarray([0, 2], dtype=np.int32),
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        active_pattern=lambda *_args, **_kwargs: active_pattern,
        summarize_pattern=lambda _op, pattern: active_summary
        if pattern is active_pattern
        else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is active_pattern
    assert setup.summary is active_summary
    assert setup.metadata["preflight_scope"] == "active_dof"
    assert setup.metadata["preflight_active_pattern_nnz_estimate"] == 4
    assert not setup.device_enabled


def test_xblock_assembled_operator_preflight_rejection_carries_metadata() -> None:
    summary = SimpleNamespace(nnz=10, shape=(3, 3), max_row_nnz=4, avg_row_nnz=3.0)
    with pytest.raises(XBlockAssembledPreflightError) as excinfo:
        build_xblock_assembled_operator_preflight_setup(
            op=SimpleNamespace(),
            xblock_active_idx_np=None,
            sparse_pc_fp_dense_velocity_block=False,
            xblock_krylov_method="gmres",
            estimate_summary=lambda *_args, **_kwargs: summary,
            full_pattern=lambda *_args, **_kwargs: object(),
            active_pattern=lambda *_args, **_kwargs: object(),
            summarize_pattern=lambda _op, _pattern: summary,
            env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "0"},
        )

    assert excinfo.value.metadata["preflight_rejected"] is True
    assert excinfo.value.metadata["preflight_pattern_nnz_estimate"] == 10
    assert "non-positive CSR memory budget" in str(excinfo.value)


def test_xblock_assembled_operator_if_requested_disabled_returns_defaults() -> None:
    def default_matvec(v):
        return 2.0 * v

    counter = MatvecCounter(0)

    result = build_xblock_assembled_operator_if_requested(
        enabled=False,
        op=SimpleNamespace(),
        rhs_dtype=jnp.float64,
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        xblock_linear_size=2,
        true_matvec_no_count=lambda v: v,
        default_matvec=default_matvec,
        mv_count=counter,
        progress_every=1,
        elapsed_s=lambda: 0.0,
        emit=None,
        estimate_summary=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        full_pattern=lambda *_args, **_kwargs: object(),
        active_pattern=lambda *_args, **_kwargs: object(),
        summarize_pattern=lambda *_args, **_kwargs: object(),
        build_operator_from_pattern=lambda *_args, **_kwargs: object(),
        device_csr_from_matrix=lambda *_args, **_kwargs: object(),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        finalize_metadata=lambda **_kwargs: {},
        backend="cpu",
        env={},
    )

    assert result.matvec is default_matvec
    assert not result.built
    assert result.metadata == {}
    assert result.pc_factor_increment_s == 0.0
    assert not result.row_enabled
    assert not result.col_enabled


def test_xblock_assembled_operator_if_requested_builds_host_operator() -> None:
    messages: list[str] = []
    pattern = object()
    summary = SimpleNamespace(nnz=2, shape=(2, 2), max_row_nnz=1, avg_row_nnz=1.0)
    matrix = scipy_sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    elapsed_values = iter([1.0, 1.2, 1.5, 2.0])

    def elapsed_s() -> float:
        try:
            return next(elapsed_values)
        except StopIteration:
            return 2.0

    bundle = SimpleNamespace(
        matrix=matrix,
        matvec=lambda x: np.asarray(x, dtype=np.float64),
        metadata=SimpleNamespace(
            storage_kind="csr",
            reason="test",
            csr_nbytes_estimate=64,
        ),
    )

    result = build_xblock_assembled_operator_if_requested(
        enabled=True,
        op=SimpleNamespace(),
        rhs_dtype=jnp.float64,
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        xblock_linear_size=2,
        true_matvec_no_count=lambda v: v,
        default_matvec=lambda v: -v,
        mv_count=MatvecCounter(0),
        progress_every=0,
        elapsed_s=elapsed_s,
        emit=lambda _level, msg: messages.append(msg),
        estimate_summary=lambda *_args, **_kwargs: summary,
        full_pattern=lambda *_args, **_kwargs: pattern,
        active_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        summarize_pattern=lambda _op, used_pattern: summary
        if used_pattern is pattern
        else (_ for _ in ()).throw(AssertionError("bad pattern")),
        build_operator_from_pattern=lambda *_args, **_kwargs: bundle,
        device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("device disabled")
        ),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        finalize_metadata=finalize_xblock_assembled_operator_metadata,
        backend="cpu",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE": "1",
        },
    )

    assert result.built
    assert not result.device_resident
    assert result.metadata["matrix_nnz"] == 2
    assert result.metadata["validation_rel_errors"] == pytest.approx((0.0,))
    assert result.pc_factor_increment_s == pytest.approx(result.metadata["setup_s"])
    assert result.row_enabled
    assert result.row_built
    assert result.matvec(jnp.asarray([3.0, 4.0])).tolist() == [3.0, 4.0]
    assert any("assembled operator built" in message for message in messages)


def test_xblock_assembled_operator_if_requested_failure_returns_metadata() -> None:
    summary = SimpleNamespace(nnz=10, shape=(3, 3), max_row_nnz=4, avg_row_nnz=3.0)

    result = build_xblock_assembled_operator_if_requested(
        enabled=True,
        op=SimpleNamespace(),
        rhs_dtype=jnp.float64,
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        xblock_linear_size=3,
        true_matvec_no_count=lambda v: v,
        default_matvec=lambda v: -v,
        mv_count=MatvecCounter(0),
        progress_every=0,
        elapsed_s=lambda: 5.0,
        emit=None,
        estimate_summary=lambda *_args, **_kwargs: summary,
        full_pattern=lambda *_args, **_kwargs: object(),
        active_pattern=lambda *_args, **_kwargs: object(),
        summarize_pattern=lambda *_args, **_kwargs: summary,
        build_operator_from_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight should reject before materialization")
        ),
        device_csr_from_matrix=lambda *_args, **_kwargs: object(),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        finalize_metadata=lambda **_kwargs: {},
        backend="cpu",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "0"},
    )

    assert not result.built
    assert "XBlockAssembledPreflightMemoryError" in result.metadata["error"]
    assert result.metadata["preflight_rejected"] is True
    assert result.pc_factor_increment_s == 0.0


def test_xblock_assembled_device_setup_builds_and_validates_operator() -> None:
    device_operator = SimpleNamespace(nnz=2, nbytes_estimate=64)

    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1024,
        device_enabled=True,
        device_required=False,
        validation_samples=2,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: device_operator,
        validate_device_csr_matvec=lambda *_args, **_kwargs: (0.0, 1.0e-12),
    )

    assert setup.device_operator is device_operator
    assert setup.device_resident
    assert setup.validation_errors == (0.0, 1.0e-12)
    assert setup.error is None


def test_xblock_assembled_device_setup_optional_failure_returns_message() -> None:
    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1,
        device_enabled=True,
        device_required=False,
        validation_samples=1,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MemoryError("too large")
        ),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
    )

    assert setup.device_operator is None
    assert not setup.device_resident
    assert "MemoryError" in str(setup.error)
    assert any(
        "disabled after build failure" in message for _, message in setup.messages
    )


def test_xblock_assembled_device_setup_required_failure_raises() -> None:
    with pytest.raises(RuntimeError, match="device CSR operator failed"):
        build_xblock_assembled_device_setup(
            assembled_matrix=object(),
            assembled_matvec=lambda x: x,
            csr_cap_nbytes=1,
            device_enabled=True,
            device_required=True,
            validation_samples=1,
            validation_tol=1.0e-8,
            device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                MemoryError("too large")
            ),
            validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        )


def test_xblock_assembled_matvec_setup_host_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda x: 3.0 * x,
        device_operator=None,
        mv_count=counter,
        progress_every=2,
        elapsed_s=lambda: 4.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "host"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [3.0, 6.0]
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [6.0, 9.0]
    assert int(counter) == 2
    assert any("assembled_host_matvecs=2" in message for message in messages)


def test_xblock_assembled_matvec_setup_device_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(
            AssertionError("unused")
        ),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    device_operator = SimpleNamespace(jitted_matvec=lambda: (lambda v: 5.0 * v))
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda _x: (_ for _ in ()).throw(AssertionError("unused")),
        device_operator=device_operator,
        mv_count=counter,
        progress_every=1,
        elapsed_s=lambda: 7.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "device"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [5.0, 10.0]
    assert int(counter) == 1
    assert any("assembled_device_matvecs=1" in message for message in messages)


def test_finalize_xblock_assembled_operator_metadata_normalizes_fields() -> None:
    metadata = finalize_xblock_assembled_operator_metadata(
        metadata={"preflight_scope": "full"},
        setup_s=1.25,
        assembled_matrix=scipy_sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0]]),
        assembled_summary=SimpleNamespace(nnz=3, avg_row_nnz=1.5, max_row_nnz=2),
        assembled_bundle_metadata=SimpleNamespace(
            storage_kind="csr",
            reason="materialized",
            csr_nbytes_estimate=128,
        ),
        max_colors=4,
        validation_errors=(1.0e-12,),
        device_enabled=True,
        device_required=False,
        device_resident=True,
        device_operator=SimpleNamespace(nnz=3, nbytes_estimate=96),
        device_validation_errors=(2.0e-12,),
        device_error=None,
    )

    assert metadata["preflight_scope"] == "full"
    assert metadata["matrix_nnz"] == 3
    assert metadata["pattern_avg_row_nnz"] == pytest.approx(1.5)
    assert metadata["device_nnz"] == 3
    assert metadata["device_validation_rel_errors"] == (2.0e-12,)


def test_xblock_moment_schur_policy_defaults_on_for_constraint1_device_krylov() -> None:
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={},
    )

    assert setup.default_candidate
    assert setup.enabled
    assert not setup.default_blocked_by_compact_factors
    assert any("moment-Schur build start" in message for _, message in setup.messages)


def test_xblock_moment_schur_policy_blocks_compact_csr_default_but_allows_force() -> (
    None
):
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0)
    blocked = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={},
    )
    forced = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT": "0.25",
        },
    )

    assert blocked.default_blocked_by_compact_factors
    assert not blocked.enabled
    assert any("default disabled" in message for _, message in blocked.messages)
    assert forced.enabled
    assert forced.rcond == pytest.approx(1.0e-9)
    assert forced.probe_enabled
    assert forced.probe_min_improvement == pytest.approx(0.25)


def test_xblock_moment_schur_policy_does_not_emit_build_for_no_preconditioner_side() -> (
    None
):
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="none",
        env={},
    )

    assert setup.enabled
    assert not setup.messages


def test_xblock_moment_schur_probe_result_accepts_sufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=7.0,
        min_improvement=0.2,
    )

    assert result.used
    assert result.reason == "probe_reduced"
    assert result.improvement_ratio == pytest.approx(0.7)
    assert any("accepted" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_rejects_insufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=9.0,
        min_improvement=0.2,
    )

    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.improvement_ratio == pytest.approx(0.9)
    assert any("rejected" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_handles_zero_rhs_norm() -> None:
    zero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=0.0,
        min_improvement=0.5,
    )
    nonzero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=1.0,
        min_improvement=0.5,
    )

    assert zero.used
    assert zero.improvement_ratio == 0.0
    assert not nonzero.used
    assert np.isinf(nonzero.improvement_ratio)


def test_xblock_moment_schur_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_moment_schur_metadata(
        metadata={"rank": 3},
        setup_s=1.5,
    )
    failure = failed_xblock_moment_schur_metadata(
        exc=ValueError("bad factor"),
        setup_s=2.5,
    )

    assert success == {"rank": 3, "setup_s": 1.5}
    assert failure["setup_s"] == 2.5
    assert failure["error"] == "ValueError: bad factor"


def test_xblock_moment_schur_stage_accepts_probe_and_records_metadata() -> None:
    policy = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=1, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE": "1",
        },
    )
    messages: list[str] = []
    stats = {"applies": 2, "base_applies": 3}

    def builder(**kwargs):
        assert kwargs["rcond"] == pytest.approx(policy.rcond)
        return (lambda v: v), {"rank": 1}, stats

    result = apply_xblock_moment_schur_stage(
        context=XBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=lambda v: jnp.zeros_like(v),
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="right",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 2.0,
            emit=lambda _level, msg: messages.append(msg),
            builder=builder,
        )
    )

    assert result.built
    assert result.used
    assert result.reason == "probe_reduced"
    assert result.metadata["rank"] == 1
    assert result.metadata["setup_s"] == pytest.approx(0.0)
    assert result.probe_residual_before == pytest.approx(1.0)
    assert result.probe_residual_after == pytest.approx(0.0)
    assert result.stats is stats
    assert result.preconditioner(jnp.asarray([4.0])).tolist() == [4.0]
    assert any("moment-Schur build start" in message for message in messages)
    assert any("constraint1 moment-Schur accepted" in message for message in messages)


def test_xblock_moment_schur_stage_rejects_probe_and_restores_base() -> None:
    policy = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=1, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE": "1",
        },
    )

    def base(v: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * v

    def builder(**_kwargs):
        return (lambda v: jnp.zeros_like(v)), {}, {}

    result = apply_xblock_moment_schur_stage(
        context=XBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=base,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="right",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert result.built
    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.preconditioner is base
    assert result.probe_improvement_ratio == pytest.approx(1.0)


def test_xblock_moment_schur_stage_records_failure() -> None:
    policy = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=1, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1"},
    )

    def builder(**_kwargs):
        raise RuntimeError("bad moment")

    result = apply_xblock_moment_schur_stage(
        context=XBlockMomentSchurStageContext(
            op=SimpleNamespace(),
            base_preconditioner=_identity,
            reduce_full=None,
            expand_reduced=None,
            policy=policy,
            precondition_side="right",
            rhs=jnp.asarray([1.0]),
            matvec_no_count=lambda v: v,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert not result.used
    assert result.preconditioner is _identity
    assert result.metadata["error"] == "RuntimeError: bad moment"


def test_xblock_two_level_policy_defaults_off_and_honors_disabled_side() -> None:
    off = resolve_xblock_two_level_policy_setup(precondition_side="right", env={})
    no_side = resolve_xblock_two_level_policy_setup(
        precondition_side="none",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert off.mode == "additive"
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_two_level_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_two_level_policy_setup(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_EXTRA_UNITS": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_INCLUDE_RHS": "0",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 7
    assert setup.fsavg_lmax == 3
    assert setup.max_extra_units == 2
    assert setup.rcond == pytest.approx(1.0e-8)
    assert not setup.include_rhs


def test_xblock_two_level_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_two_level_metadata(
        metadata={"mode": "additive"}, setup_s=0.25
    )
    failure = failed_xblock_two_level_metadata(
        exc=RuntimeError("bad coarse"), setup_s=0.5
    )

    assert success == {"mode": "additive", "setup_s": 0.25}
    assert failure == {"error": "RuntimeError: bad coarse", "setup_s": 0.5}


def test_xblock_two_level_stage_builds_and_records_stats() -> None:
    policy = resolve_xblock_two_level_policy_setup(
        precondition_side="right",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1"},
    )
    stats = {"applies": 3, "coarse_applies": 4}

    def builder(**kwargs):
        assert kwargs["expected_size"] == 2
        assert kwargs["mode"] == "additive"
        return (lambda v: 2.0 * v), {"rank": 2}, stats

    result = apply_xblock_two_level_stage(
        context=XBlockTwoLevelStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=2,
            policy=policy,
            elapsed_s=lambda: 1.0,
            emit=None,
            builder=builder,
        )
    )

    assert result.built
    assert result.metadata == {"rank": 2, "setup_s": 0.0}
    assert result.stats is stats
    assert result.preconditioner(jnp.asarray([3.0])).tolist() == [6.0]


def test_xblock_two_level_stage_without_injected_builder_stays_disabled() -> None:
    policy = resolve_xblock_two_level_policy_setup(
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS": "8",
        },
    )
    result = apply_xblock_two_level_stage(
        context=XBlockTwoLevelStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=2,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert "moved to research QI branch" in result.metadata["error"]


def test_xblock_two_level_stage_records_failure() -> None:
    policy = resolve_xblock_two_level_policy_setup(
        precondition_side="right",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1"},
    )

    def builder(**_kwargs):
        raise ValueError("bad two")

    result = apply_xblock_two_level_stage(
        context=XBlockTwoLevelStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=1,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
            builder=builder,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert result.metadata["error"] == "ValueError: bad two"


def test_xblock_global_coupling_policy_defaults_off_and_selects_builder_defaults() -> (
    None
):
    off = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres",
        env={},
    )
    device = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )
    no_side = resolve_xblock_global_coupling_policy_setup(
        precondition_side="none",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert not off.use_device_builder
    assert off.setup_max_s == 0.0
    assert device.enabled
    assert device.should_build
    assert device.use_device_builder
    assert device.setup_max_s == pytest.approx(180.0)
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_global_coupling_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_global_coupling_policy_setup(
        precondition_side="left",
        xblock_krylov_method="bicgstab_jax",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS": "9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_EXTRA_UNITS": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_RCOND": "1e-7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_INCLUDE_RHS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S": "12.5",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.use_device_builder
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 9
    assert setup.fsavg_lmax == 3
    assert setup.angular_lmax == 4
    assert setup.max_extra_units == 5
    assert setup.rcond == pytest.approx(1.0e-7)
    assert not setup.include_rhs
    assert setup.setup_max_s == pytest.approx(12.5)


def test_xblock_global_coupling_metadata_helpers_normalize_success_and_failure() -> (
    None
):
    success = finalize_xblock_global_coupling_metadata(
        metadata={"mode": "additive"}, setup_s=0.75
    )
    failure = failed_xblock_global_coupling_metadata(
        exc=RuntimeError("timeout"), setup_s=1.5
    )

    assert success == {"mode": "additive", "setup_s": 0.75}
    assert failure == {"error": "RuntimeError: timeout", "setup_s": 1.5}


def test_xblock_global_coupling_stage_selects_device_builder() -> None:
    policy = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )
    stats = {"applies": 5, "coarse_applies": 6}

    def host_builder(**_kwargs):
        raise AssertionError("host builder should not be selected")

    def device_builder(**kwargs):
        assert kwargs["max_setup_s"] == pytest.approx(180.0)
        assert kwargs["expected_size"] == 2
        return (lambda v: 3.0 * v), {"rank": 4}, stats

    result = apply_xblock_global_coupling_stage(
        context=XBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0, 2.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=2,
            policy=policy,
            elapsed_s=lambda: 1.0,
            emit=None,
            host_builder=host_builder,
            device_builder=device_builder,
        )
    )

    assert result.built
    assert result.metadata == {"rank": 4, "setup_s": 0.0}
    assert result.stats is stats
    assert result.preconditioner(jnp.asarray([2.0])).tolist() == [6.0]


def test_xblock_global_coupling_stage_without_injected_builder_stays_disabled() -> None:
    for method in ("gmres", "gmres_jax"):
        policy = resolve_xblock_global_coupling_policy_setup(
            precondition_side="right",
            xblock_krylov_method=method,
            env={
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1",
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS": "8",
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S": "0",
            },
        )
        result = apply_xblock_global_coupling_stage(
            context=XBlockGlobalCouplingStageContext(
                op=SimpleNamespace(),
                rhs=jnp.asarray([1.0, 2.0]),
                matvec=lambda v: v,
                base_preconditioner=_identity,
                direction_projector=None,
                expected_size=2,
                policy=policy,
                elapsed_s=lambda: 0.0,
                emit=None,
            )
        )

        assert not result.built
        assert result.preconditioner is _identity
        assert "moved to research QI branch" in result.metadata["error"]


def test_xblock_global_coupling_stage_records_failure() -> None:
    policy = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )

    def host_builder(**_kwargs):
        raise RuntimeError("bad global")

    result = apply_xblock_global_coupling_stage(
        context=XBlockGlobalCouplingStageContext(
            op=SimpleNamespace(),
            rhs=jnp.asarray([1.0]),
            matvec=lambda v: v,
            base_preconditioner=_identity,
            direction_projector=None,
            expected_size=1,
            policy=policy,
            elapsed_s=lambda: 0.0,
            emit=None,
            host_builder=host_builder,
            device_builder=host_builder,
        )
    )

    assert not result.built
    assert result.preconditioner is _identity
    assert result.metadata["error"] == "RuntimeError: bad global"


def test_prepare_xblock_initial_guess_accepts_reduced_and_full_active_shapes() -> None:
    reduced = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    rhs_reduced = jnp.zeros(2)
    rhs_full = jnp.zeros(4)

    reduced_result = prepare_xblock_initial_guess(
        x0=reduced,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )
    full_result = prepare_xblock_initial_guess(
        x0=full,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )

    assert reduced_result.messages == ()
    assert jnp.asarray(reduced_result.x0_full).tolist() == [1.0, 2.0]
    assert full_result.messages == ()
    assert jnp.asarray(full_result.x0_full).tolist() == [10.0, 12.0]


def test_prepare_xblock_initial_guess_rejects_incompatible_shape_with_message() -> None:
    result = prepare_xblock_initial_guess(
        x0=jnp.ones(3),
        xblock_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        xblock_use_active_dof=True,
        reduce_full=lambda v: v,
    )

    assert result.x0_full is None
    assert len(result.messages) == 1
    assert "ignoring incompatible x0 shape=(3,)" in result.messages[0][1]
    assert "expected=(2,) or (4,)" in result.messages[0][1]


def test_prepare_fortran_reduced_xblock_initial_guess_routes_reduced_and_full_shapes() -> None:
    reduced = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    rhs_reduced = jnp.zeros(2)
    rhs_full = jnp.zeros(4)

    reduced_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=reduced,
        sparse_pc_rhs=rhs_reduced,
        full_rhs=rhs_full,
        reduce_full=lambda v: v[jnp.asarray([1, 3])],
    )
    full_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=full,
        sparse_pc_rhs=rhs_reduced,
        full_rhs=rhs_full,
        reduce_full=lambda v: v[jnp.asarray([1, 3])],
    )

    assert reduced_result.messages == ()
    assert jnp.asarray(reduced_result.x0_full).tolist() == [1.0, 2.0]
    assert full_result.messages == ()
    assert jnp.asarray(full_result.x0_full).tolist() == [11.0, 13.0]


def test_prepare_fortran_reduced_xblock_initial_guess_handles_none_and_bad_shape() -> None:
    none_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=None,
        sparse_pc_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        reduce_full=lambda v: v,
    )
    bad_result = prepare_fortran_reduced_xblock_initial_guess(
        x0=jnp.ones(3),
        sparse_pc_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        reduce_full=lambda v: v,
    )

    assert none_result.x0_full is None
    assert none_result.messages == ()
    assert bad_result.x0_full is None
    assert len(bad_result.messages) == 1
    assert "fortran_reduced_pc_gmres xblock ignoring incompatible x0 shape=(3,)" in (
        bad_result.messages[0][1]
    )
    assert "expected=(2,) or (4,)" in bad_result.messages[0][1]


def test_xblock_seed_policy_defaults_and_env_overrides() -> None:
    default = resolve_xblock_seed_policy_setup(moment_schur_used=True, env={})
    disabled = resolve_xblock_seed_policy_setup(
        moment_schur_used=True,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED": "0",
        },
    )

    assert not default.initial_seed_enabled
    assert default.moment_schur_seed_enabled
    assert disabled.initial_seed_enabled
    assert not disabled.moment_schur_seed_enabled


def test_sparse_pc_gmres_control_policy_defaults() -> None:
    policy = resolve_sparse_pc_gmres_control_policy({})

    assert isinstance(policy, SparsePCGMRESControlPolicy)
    assert policy.stagnation_abort is False
    assert policy.stagnation_min_iter == 500
    assert policy.stagnation_window == 500
    assert policy.stagnation_rel_improvement == pytest.approx(1.0e-3)
    assert policy.post_minres_steps == 0
    assert policy.post_minres_alpha_clip == pytest.approx(10.0)
    assert policy.post_minres_min_improvement == 0.0


def test_sparse_pc_gmres_control_policy_overrides_and_clamps() -> None:
    policy = resolve_sparse_pc_gmres_control_policy(
        {
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_ABORT": "1",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_MIN_ITER": "0",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_WINDOW": "-2",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_REL_IMPROVEMENT": "-0.1",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_STEPS": "-3",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_ALPHA_CLIP": "-1.5",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_MIN_IMPROVEMENT": "0.25",
        }
    )

    assert policy.stagnation_abort is True
    assert policy.stagnation_min_iter == 1
    assert policy.stagnation_window == 1
    assert policy.stagnation_rel_improvement == 0.0
    assert policy.post_minres_steps == 0
    assert policy.post_minres_alpha_clip == 0.0
    assert policy.post_minres_min_improvement == pytest.approx(0.25)


def test_sparse_pc_gmres_once_explicit_left_recomputes_true_residual() -> None:
    messages: list[str] = []
    times = iter((0.0, 0.25, 0.5, 0.75))

    def explicit_left_solver(**kwargs):
        kwargs["progress_callback"](2, 4.0e-1)
        return np.asarray([0.25, 0.75]), 99.0, 0.5, (1.0, 0.4)

    result = run_sparse_pc_gmres_once(
        context=SparsePCGMRESContext(
            matvec=lambda x: 2.0 * x,
            rhs=jnp.asarray([1.0, 1.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            restart=7,
            tol=1.0e-8,
            atol=0.0,
            precondition_side="left",
            factor_dtype=np.dtype(np.float32),
            progress_every=2,
            stagnation_abort=False,
            stagnation_min_iter=10,
            stagnation_window=10,
            stagnation_rel_improvement=1.0e-3,
            explicit_left_solver=explicit_left_solver,
            gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("wrong solver")
            ),
        ),
        x0=None,
        maxiter=3,
    )

    assert result.x.tolist() == [0.25, 0.75]
    assert result.preconditioned_residual_norm == pytest.approx(0.5)
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.5, -0.5]))
    assert result.history == (1.0, 0.4)
    assert any("factor_dtype=float32" in msg for msg in messages)
    assert any("iters=2" in msg for msg in messages)


def test_sparse_pc_gmres_once_for_retry_returns_dtype_retry_tuple() -> None:
    times = iter((0.0, 0.5))

    def gmres_solver(**kwargs):
        assert kwargs["maxiter"] == 4
        return np.asarray([0.25, 0.75]), 99.0, (1.0, 0.4)

    x, residual_norm, rn_pc, history, solve_s = run_sparse_pc_gmres_once_for_retry(
        context=SparsePCGMRESContext(
            matvec=lambda x_arg: 2.0 * x_arg,
            rhs=jnp.asarray([1.0, 1.0]),
            preconditioner=_identity,
            emit=None,
            elapsed_s=lambda: next(times),
            pc_form="right",
            restart=7,
            tol=1.0e-8,
            atol=0.0,
            precondition_side="right",
            factor_dtype=np.dtype(np.float64),
            progress_every=0,
            stagnation_abort=False,
            stagnation_min_iter=10,
            stagnation_window=10,
            stagnation_rel_improvement=1.0e-3,
            explicit_left_solver=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("wrong solver")
            ),
            gmres_solver=gmres_solver,
        ),
        x0=None,
        maxiter=4,
    )

    assert isinstance(x, np.ndarray)
    assert x.tolist() == [0.25, 0.75]
    assert residual_norm == pytest.approx(np.linalg.norm([0.5, -0.5]))
    assert not np.isfinite(rn_pc)
    assert history == (1.0, 0.4)
    assert solve_s == pytest.approx(0.5)


def test_sparse_pc_gmres_once_stagnation_guard_raises() -> None:
    def gmres_solver(**kwargs):
        progress = kwargs["progress_callback"]
        progress(1, 1.0)
        progress(2, 1.0)
        return np.ones(2), 1.0, (1.0,)

    with pytest.raises(RuntimeError, match="sparse_pc_gmres stagnation detected"):
        run_sparse_pc_gmres_once(
            context=SparsePCGMRESContext(
                matvec=_identity,
                rhs=jnp.ones(2),
                preconditioner=_identity,
                emit=None,
                elapsed_s=lambda: 0.0,
                pc_form="right",
                restart=5,
                tol=1.0e-8,
                atol=0.0,
                precondition_side="right",
                factor_dtype=np.dtype(np.float64),
                progress_every=0,
                stagnation_abort=True,
                stagnation_min_iter=2,
                stagnation_window=1,
                stagnation_rel_improvement=1.0e-3,
                explicit_left_solver=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("wrong solver")
                ),
                gmres_solver=gmres_solver,
            ),
            x0=None,
            maxiter=10,
        )


def test_sparse_pc_gmres_completion_message_includes_pc_and_ksp_residuals() -> None:
    message = sparse_pc_gmres_completion_message(
        SparsePCGMRESCompletionMessageContext(
            elapsed_s=12.3456,
            iterations=7,
            matvecs=13,
            residual_norm=1.25e-4,
            target=1.0e-6,
            preconditioned_residual_norm=2.5e-3,
            history=(1.0, 3.0e-3),
        )
    )

    assert message == (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        "elapsed_s=12.346 iters=7 matvecs=13 residual=1.250000e-04 "
        "target=1.000000e-06 preconditioned_residual=2.500000e-03 "
        "ksp_residual=3.000000e-03"
    )


def test_sparse_pc_gmres_completion_message_omits_nonfinite_optional_residuals() -> None:
    message = sparse_pc_gmres_completion_message(
        SparsePCGMRESCompletionMessageContext(
            elapsed_s=1.0,
            iterations=0,
            matvecs=0,
            residual_norm=float("inf"),
            target=2.0,
            preconditioned_residual_norm=float("nan"),
            history=(),
        )
    )

    assert message == (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        "elapsed_s=1.000 iters=0 matvecs=0 residual=inf target=2.000000e+00"
    )


def test_emit_sparse_pc_gmres_completion_from_solve_state_uses_current_state() -> None:
    messages: list[tuple[int, str]] = []

    emit_sparse_pc_gmres_completion_from_solve_state(
        {
            "emit": lambda level, msg: messages.append((level, msg)),
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
            "history": (1.0, 0.25),
            "mv_count": 6,
            "residual_norm_sparse_pc": 0.125,
            "target": 0.01,
            "rn_pc": 0.5,
        }
    )
    emit_sparse_pc_gmres_completion_from_solve_state(
        {
            "emit": None,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 0.0),
            "history": (),
            "mv_count": 0,
            "residual_norm_sparse_pc": 0.0,
            "target": 1.0,
            "rn_pc": float("nan"),
        }
    )

    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
            "elapsed_s=2.500 iters=2 matvecs=6 residual=1.250000e-01 "
            "target=1.000000e-02 preconditioned_residual=5.000000e-01 "
            "ksp_residual=2.500000e-01",
        )
    ]


def test_sparse_pc_gmres_final_payload_from_solve_state_expands_result_and_metadata() -> None:
    state = _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=np.int64(4)),
            "x_np": np.asarray([1.0, 2.0]),
            "residual_norm_sparse_pc": 0.25,
            "history": (1.0, 0.25),
            "mv_count": np.int64(3),
            "pc_restart": np.int64(4),
            "pc_maxiter": np.int64(5),
            "sparse_pc_first_attempt_maxiter": np.int64(2),
            "sparse_pc_post_minres_steps": np.int64(0),
            "sparse_pc_post_minres_alphas": (),
            "sparse_pc_post_minres_alpha_clip": 4.0,
            "sparse_pc_post_minres_min_improvement": 0.1,
            "sparse_pc_post_minres_residual_before": None,
            "sparse_pc_post_minres_residual_after": None,
            "sparse_pc_post_minres_history": (),
            "sparse_pc_post_minres_error": None,
            "pc_shift": 0.0,
            "sparse_pc_factor_dtype_used": np.dtype(np.float64),
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_factor_dtype_retry": None,
            "factor_preflight_enabled": False,
            "factor_preflight_required": False,
            "factor_preflight_seed_enabled": False,
            "factor_preflight_seed_used": False,
            "factor_preflight_passed": None,
            "factor_preflight_error": None,
            "factor_preflight_residual_before": None,
            "factor_preflight_residual_after": None,
            "factor_preflight_improvement_ratio": None,
            "factor_preflight_target_ratio": None,
            "factor_preflight_max_target_ratio": 8.0,
            "factor_preflight_residual_diagnostics": {},
            "fortran_reduced_sparse_pc": False,
            "sparse_pc_preconditioner_operator": "full",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "ilu",
            "sparse_pc_default_ilu_fill_factor": 6.0,
            "sparse_pc_default_ilu_drop_tol": 1.0e-5,
            "sparse_pc_default_pattern_color_batch": np.int64(8),
            "preconditioner_x": np.int64(1),
            "preconditioner_x_min_l": np.int64(0),
            "preconditioner_xi": np.int64(1),
            "preconditioner_species": np.int64(1),
            "sparse_pc_permc_spec": "COLAMD",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": np.int64(2),
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.5,
            "solve_s": 1.5,
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 3.0),
            "summary": SimpleNamespace(
                nnz=np.int64(7),
                avg_row_nnz=1.75,
                max_row_nnz=np.int64(3),
            ),
            "sparse_pattern_scope": "active_dof",
            "pattern_build_s": 0.125,
            "pc_factor_s": 0.25,
            "factor_bundle_pc": SimpleNamespace(
                factor_s=None,
                factor_nbytes_estimate=None,
                factor_nnz_estimate=None,
            ),
            "_operator_bundle_pc": None,
            "target": 0.5,
            "atol": 0.5,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "direct_tail_operator_bundle": None,
            "direct_tail_structured_max_nbytes": None,
            "direct_tail_true_window_specs": (),
            "direct_tail_true_active_block_species_count": None,
            "direct_tail_structured_pc_metadata": {},
            "direct_tail_support_mode_preflight_metadata": {},
            "direct_tail_true_coupled_coarse_metadata": {},
            "direct_tail_residual_window_coefficient_mode": "normal",
            "direct_tail_residual_window_combine_mode": "additive",
            "direct_tail_error": None,
            "direct_tail_structured_pc_requested": "auto",
            "direct_tail_structured_pc_reason": "none",
            "direct_tail_structured_pc_error": None,
            "direct_tail_support_mode_preflight_error": None,
        }
    )

    payload = sparse_pc_gmres_final_payload_from_solve_state(
        state,
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert payload.metadata["solver_kind"] == "sparse_pc_gmres"
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["sparse_pc_residual_ratio_to_target"] == pytest.approx(0.5)
    assert payload.metadata["sparse_pc_linear_size"] == 2


def test_explicit_sparse_operator_policy_and_messages_are_stable() -> None:
    policy = resolve_explicit_sparse_operator_build_policy(
        {
            "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "bad",
            "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "2e-4",
        }
    )
    messages = explicit_sparse_pattern_progress_messages(
        solver_label="sparse_host",
        summary=SimpleNamespace(nnz=np.int64(9), avg_row_nnz=2.25, max_row_nnz=np.int64(4)),
    )

    assert isinstance(policy, ExplicitSparseOperatorBuildPolicy)
    assert policy.csr_max_mb == pytest.approx(512.0)
    assert policy.drop_tol == pytest.approx(2.0e-4)
    assert messages == (
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host building conservative pattern",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host pattern "
            "nnz=9 avg_row_nnz=2.25 max_row_nnz=4",
        ),
    )


def test_build_explicit_sparse_operator_from_pattern_forwards_policy_and_reports_storage() -> None:
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="materialized"),
        matrix=scipy_sparse.eye(2, format="csr"),
    )

    def builder(matvec_np, **kwargs):
        calls.append({"matvec_np": matvec_np, **kwargs})
        return bundle

    result = build_explicit_sparse_operator_from_pattern(
        matvec_np=lambda x: np.asarray(x, dtype=np.float64),
        pattern={"row": (0,)},
        dtype=np.float64,
        backend="cpu",
        env={
            "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "64",
            "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "1e-5",
        },
        build_operator_from_pattern=builder,
        allow_operator_only=False,
    )

    assert isinstance(result, ExplicitSparseOperatorBuildResult)
    assert result.operator_bundle is bundle
    assert result.policy.csr_max_mb == pytest.approx(64.0)
    assert result.policy.drop_tol == pytest.approx(1.0e-5)
    assert calls[0]["pattern"] == {"row": (0,)}
    assert calls[0]["backend"] == "cpu"
    assert calls[0]["csr_max_mb"] == pytest.approx(64.0)
    assert calls[0]["drop_tol"] == pytest.approx(1.0e-5)
    assert calls[0]["allow_operator_only"] is False
    assert result.messages == (
        (1, "explicit_sparse: storage=csr reason=materialized"),
    )


def test_validate_explicit_sparse_host_request_preserves_user_facing_errors() -> None:
    validate_explicit_sparse_host_request(
        solve_method_label="sparse_host",
        differentiable=False,
        rhs_mode=1,
        use_active_dof=False,
        path_description="host sparse LU path",
    )

    with pytest.raises(ValueError, match="non-differentiable host sparse LU path"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_host",
            differentiable=True,
            rhs_mode=1,
            use_active_dof=False,
            path_description="host sparse LU path",
        )
    with pytest.raises(NotImplementedError, match="RHSMode=1"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_lsmr",
            differentiable=False,
            rhs_mode=2,
            use_active_dof=False,
            path_description="host sparse minimum-norm path",
        )
    with pytest.raises(NotImplementedError, match="ACTIVE_DOF=0"):
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_lsmr",
            differentiable=False,
            rhs_mode=1,
            use_active_dof=True,
            path_description="host sparse minimum-norm path",
        )


def test_sparse_minimum_norm_policy_parses_env_and_preserves_defaults() -> None:
    default_policy = resolve_sparse_minimum_norm_policy(
        {},
        solve_method_kind="sparse_lsmr",
        tol=1.0e-8,
        maxiter=None,
        emit_enabled=False,
    )

    assert isinstance(default_policy, SparseMinimumNormPolicy)
    assert default_policy.solver_name == "lsmr"
    assert default_policy.atol == pytest.approx(1.0e-8)
    assert default_policy.btol == pytest.approx(1.0e-8)
    assert default_policy.conlim == pytest.approx(1.0e8)
    assert default_policy.damp == pytest.approx(0.0)
    assert default_policy.maxiter == 1000
    assert default_policy.show is False
    assert default_policy.petsc_compat_requested is False

    parsed_policy = resolve_sparse_minimum_norm_policy(
        {
            "SFINCS_JAX_SPARSE_LSMR_ATOL": "2e-7",
            "SFINCS_JAX_SPARSE_LSMR_BTOL": "bad",
            "SFINCS_JAX_SPARSE_LSMR_CONLIM": "3e5",
            "SFINCS_JAX_SPARSE_LSMR_DAMP": "4e-3",
            "SFINCS_JAX_SPARSE_LSMR_MAXITER": "12",
            "SFINCS_JAX_SPARSE_LSMR_SHOW": "yes",
        },
        solve_method_kind="sparse_lsqr",
        tol=1.0e-6,
        maxiter=7,
        emit_enabled=True,
    )

    assert parsed_policy.solver_name == "lsqr"
    assert parsed_policy.atol == pytest.approx(2.0e-7)
    assert parsed_policy.btol == pytest.approx(1.0e-6)
    assert parsed_policy.conlim == pytest.approx(3.0e5)
    assert parsed_policy.damp == pytest.approx(4.0e-3)
    assert parsed_policy.maxiter == 12
    assert parsed_policy.show is True
    assert sparse_minimum_norm_start_message(parsed_policy) == (
        "solve_v3_full_system_linear_gmres: sparse_lsmr solve start "
        "solver=lsqr atol=2.0e-07 btol=1.0e-06 damp=4.0e-03 "
        "conlim=3.0e+05 maxiter=12"
    )


def test_sparse_minimum_norm_solve_payload_solves_tiny_identity_system() -> None:
    policy = resolve_sparse_minimum_norm_policy(
        {},
        solve_method_kind="petsc_compat",
        tol=1.0e-12,
        maxiter=20,
        emit_enabled=False,
    )

    payload = sparse_minimum_norm_solve_payload(
        matrix=scipy_sparse.eye(2, format="csr"),
        rhs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        policy=policy,
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([1.0, -2.0])),
        elapsed_s=lambda: 1.25,
    )

    assert isinstance(payload, SparseMinimumNormPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, -2.0]), atol=1.0e-10)
    assert float(payload.residual_norm) < 1.0e-10
    assert payload.metadata["solver_kind"] == "sparse_lsmr"
    assert payload.metadata["petsc_compat_requested"] is True
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["acceptance_criterion"] == "true_residual"
    assert "accepted=True criterion=true_residual" in payload.completion_message


def test_sparse_minimum_norm_solve_from_pattern_materializes_and_emits_messages() -> None:
    messages: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="materialized"),
        matrix=scipy_sparse.eye(2, format="csr"),
    )

    def builder(matvec_np, **kwargs):
        calls.append({"matvec_np": matvec_np, **kwargs})
        return bundle

    payload = sparse_minimum_norm_solve_from_pattern(
        matvec_np=lambda x: np.asarray(x, dtype=np.float64),
        pattern={"rows": (0, 1)},
        summary=SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1),
        rhs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        solve_method_kind="sparse_lsmr",
        tol=1.0e-12,
        atol=1.0e-12,
        maxiter=20,
        rhs_norm=float(np.linalg.norm([1.0, -2.0])),
        elapsed_s=lambda: 2.5,
        backend="cpu",
        env={"SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "32"},
        emit=lambda level, message: messages.append((level, message)),
        build_operator_from_pattern=builder,
    )

    assert isinstance(payload, SparseMinimumNormPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, -2.0]), atol=1.0e-10)
    assert float(payload.residual_norm) < 1.0e-10
    assert calls[0]["pattern"] == {"rows": (0, 1)}
    assert calls[0]["backend"] == "cpu"
    assert calls[0]["csr_max_mb"] == pytest.approx(32.0)
    assert calls[0]["allow_operator_only"] is False
    assert messages[0] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_lsmr building conservative pattern",
    )
    assert messages[1] == (
        1,
        "solve_v3_full_system_linear_gmres: sparse_lsmr pattern "
        "nnz=2 avg_row_nnz=1 max_row_nnz=1",
    )
    assert messages[2] == (1, "explicit_sparse: storage=csr reason=materialized")
    assert messages[3][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr solve start")
    assert messages[4][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr complete")


def test_sparse_minimum_norm_solve_from_pattern_requires_materialized_matrix() -> None:
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="operator_only", reason="too_large"),
        matrix=None,
    )

    with pytest.raises(RuntimeError, match="requires a materialized sparse matrix"):
        sparse_minimum_norm_solve_from_pattern(
            matvec_np=lambda x: np.asarray(x, dtype=np.float64),
            pattern={"rows": (0,)},
            summary=SimpleNamespace(nnz=1, avg_row_nnz=1.0, max_row_nnz=1),
            rhs=jnp.asarray([1.0], dtype=jnp.float64),
            solve_method_kind="sparse_lsmr",
            tol=1.0e-12,
            atol=1.0e-12,
            maxiter=10,
            rhs_norm=1.0,
            elapsed_s=lambda: 0.0,
            backend="cpu",
            env={},
            emit=None,
            build_operator_from_pattern=lambda *_args, **_kwargs: bundle,
        )


def test_solve_explicit_sparse_minimum_norm_branch_uses_driver_callbacks() -> None:
    messages: list[tuple[int, str]] = []
    calls: dict[str, object] = {}
    op = SimpleNamespace(rhs_mode=1, total_size=2, matrix=scipy_sparse.eye(2, format="csr"))
    bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="materialized"),
        matrix=op.matrix,
    )

    def build_pattern(arg_op):
        calls["pattern_op"] = arg_op
        return {"rows": (0, 1)}

    def summarize_pattern(arg_op, pattern):
        calls["summary_op"] = arg_op
        calls["summary_pattern"] = pattern
        return SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1)

    def apply_cached_operator(arg_op, x):
        calls["matvec_op"] = arg_op
        return jnp.asarray(arg_op.matrix @ np.asarray(x, dtype=np.float64), dtype=x.dtype)

    def build_operator_from_pattern(matvec_np, **kwargs):
        calls["operator_kwargs"] = kwargs
        calls["operator_action"] = matvec_np(np.asarray([2.0, -3.0]))
        return bundle

    payload = solve_explicit_sparse_minimum_norm_branch(
        ExplicitSparseMinimumNormBranchContext(
            op=op,
            rhs=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            solve_method_kind="sparse_lsmr",
            differentiable=False,
            use_active_dof=False,
            tol=1.0e-12,
            atol=1.0e-12,
            maxiter=20,
            rhs_norm=float(np.linalg.norm([1.0, -2.0])),
            backend="cpu",
            env={},
            emit=lambda level, message: messages.append((level, message)),
            build_pattern=build_pattern,
            summarize_pattern=summarize_pattern,
            apply_cached_operator=apply_cached_operator,
            build_operator_from_pattern=build_operator_from_pattern,
        )
    )

    assert isinstance(payload, SparseMinimumNormPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, -2.0]), atol=1.0e-10)
    assert float(payload.residual_norm) < 1.0e-10
    assert calls["pattern_op"] is op
    assert calls["summary_pattern"] == {"rows": (0, 1)}
    np.testing.assert_allclose(calls["operator_action"], np.asarray([2.0, -3.0]))
    assert calls["operator_kwargs"]["pattern"] == {"rows": (0, 1)}
    assert calls["operator_kwargs"]["backend"] == "cpu"
    assert messages[0][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr building")
    assert messages[-1][1].startswith("solve_v3_full_system_linear_gmres: sparse_lsmr complete")


def test_sparse_host_direct_solve_payload_recomputes_true_residual_and_metadata() -> None:
    calls: list[dict[str, object]] = []

    def direct_solve_with_refinement(**kwargs):
        calls.append(kwargs)
        return np.asarray([3.0, -1.0]), 99.0

    payload = sparse_host_direct_solve_payload(
        factor_solve=lambda rhs: rhs,
        operator_matrix=scipy_sparse.eye(2, format="csr"),
        rhs=jnp.asarray([3.0, -1.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=2,
        matvec=lambda x: jnp.asarray(x, dtype=jnp.float64),
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([3.0, -1.0])),
        elapsed_s=lambda: 0.75,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )

    assert isinstance(payload, SparseHostDirectPayload)
    assert calls[0]["refine_steps"] == 2
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([3.0, -1.0]))
    assert float(payload.residual_norm) == pytest.approx(0.0)
    assert payload.metadata == {
        "solver_kind": "sparse_host",
        "residual_kind": "true_residual",
        "accepted_converged": True,
        "acceptance_criterion": "true_residual",
    }
    assert payload.completion_message == (
        "solve_v3_full_system_linear_gmres: sparse_host complete "
        "elapsed_s=0.750 residual=0.000000e+00"
    )


def test_sparse_host_direct_solve_from_pattern_builds_factor_and_emits_messages() -> None:
    messages: list[tuple[int, str]] = []
    build_calls: list[dict[str, object]] = []
    direct_calls: list[dict[str, object]] = []
    operator_bundle = SimpleNamespace(matrix=scipy_sparse.diags([2.0, 4.0], format="csr"))
    factor_bundle = SimpleNamespace(solve=lambda rhs: rhs)

    def build_factor(**kwargs):
        build_calls.append(kwargs)
        return operator_bundle, factor_bundle

    def direct_solve_with_refinement(**kwargs):
        direct_calls.append(kwargs)
        return np.asarray([1.0, 2.0]), 9.0

    payload = sparse_host_direct_solve_from_pattern(
        matvec=lambda x: jnp.asarray([2.0 * x[0], 4.0 * x[1]], dtype=jnp.float64),
        pattern={"rows": (0, 1)},
        summary=SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1),
        n=2,
        dtype=jnp.float64,
        rhs=jnp.asarray([2.0, 8.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
        atol=1.0e-12,
        tol=1.0e-12,
        rhs_norm=float(np.linalg.norm([2.0, 8.0])),
        elapsed_s=lambda: 1.5,
        emit=lambda level, message: messages.append((level, message)),
        build_host_sparse_direct_factor_from_matvec=build_factor,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )

    assert isinstance(payload, SparseHostDirectPayload)
    assert build_calls[0]["pattern"] == {"rows": (0, 1)}
    assert build_calls[0]["n"] == 2
    assert build_calls[0]["factor_dtype"] == np.dtype(np.float64)
    assert build_calls[0]["emit"] is not None
    assert direct_calls[0]["factor_solve"] is factor_bundle.solve
    assert direct_calls[0]["operator_matrix"] is operator_bundle.matrix
    assert direct_calls[0]["refine_steps"] == 3
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.0)
    assert payload.metadata["solver_kind"] == "sparse_host"
    assert payload.metadata["accepted_converged"] is True
    assert messages == [
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host building conservative pattern",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: sparse_host pattern "
            "nnz=2 avg_row_nnz=1 max_row_nnz=1",
        ),
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse_host complete "
            "elapsed_s=1.500 residual=0.000000e+00",
        ),
    ]


def test_solve_explicit_sparse_host_direct_branch_uses_driver_callbacks() -> None:
    messages: list[tuple[int, str]] = []
    calls: dict[str, object] = {}
    matrix = scipy_sparse.diags([2.0, 4.0], format="csr")
    op = SimpleNamespace(rhs_mode=1, total_size=2, matrix=matrix)
    operator_bundle = SimpleNamespace(matrix=matrix)
    factor_bundle = SimpleNamespace(solve=lambda rhs: rhs)

    def build_pattern(arg_op):
        calls["pattern_op"] = arg_op
        return {"rows": (0, 1)}

    def summarize_pattern(arg_op, pattern):
        calls["summary_op"] = arg_op
        calls["summary_pattern"] = pattern
        return SimpleNamespace(nnz=2, avg_row_nnz=1.0, max_row_nnz=1)

    def apply_operator(arg_op, x):
        calls["matvec_op"] = arg_op
        return jnp.asarray(arg_op.matrix @ np.asarray(x, dtype=np.float64), dtype=x.dtype)

    def build_factor(**kwargs):
        calls["factor_kwargs"] = kwargs
        return operator_bundle, factor_bundle

    def direct_solve_with_refinement(**kwargs):
        calls["direct_kwargs"] = kwargs
        return np.asarray([1.0, 2.0]), 9.0

    payload = solve_explicit_sparse_host_direct_branch(
        ExplicitSparseHostDirectBranchContext(
            op=op,
            rhs=jnp.asarray([2.0, 8.0], dtype=jnp.float64),
            differentiable=False,
            use_active_dof=False,
            tol=1.0e-12,
            atol=1.0e-12,
            rhs_norm=float(np.linalg.norm([2.0, 8.0])),
            refine_steps=3,
            emit=lambda level, message: messages.append((level, message)),
            build_pattern=build_pattern,
            summarize_pattern=summarize_pattern,
            apply_operator=apply_operator,
            build_host_sparse_direct_factor_from_matvec=build_factor,
            direct_solve_with_refinement=direct_solve_with_refinement,
        )
    )

    assert isinstance(payload, SparseHostDirectPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.0)
    assert calls["pattern_op"] is op
    assert calls["summary_pattern"] == {"rows": (0, 1)}
    assert calls["factor_kwargs"]["pattern"] == {"rows": (0, 1)}
    assert calls["factor_kwargs"]["n"] == 2
    assert calls["direct_kwargs"]["refine_steps"] == 3
    assert messages[0][1].startswith("solve_v3_full_system_linear_gmres: sparse_host building")
    assert messages[-1] == (0, payload.completion_message)


def test_solve_sparse_host_direct_from_available_factor_prefers_explicit_factor() -> None:
    calls: list[str] = []
    explicit_factor = SimpleNamespace(solve=lambda rhs: rhs)
    explicit_operator = SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr"))

    def direct_solve(**kwargs):
        calls.append("direct")
        assert kwargs["factor_solve"] is explicit_factor.solve
        assert kwargs["operator_matrix"] is explicit_operator.matrix
        assert kwargs["refine_steps"] == 3
        return np.asarray([1.0, 2.0]), 0.125

    def ilu_solve(**_kwargs):
        calls.append("ilu")
        return np.asarray([9.0, 9.0]), 9.0

    payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=explicit_factor,
        explicit_sparse_operator=explicit_operator,
        ilu=object(),
        a_csr_full=object(),
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
    )

    assert isinstance(payload, SparseHostDirectFactorSolvePayload)
    assert calls == ["direct"]
    np.testing.assert_allclose(payload.x, np.asarray([1.0, 2.0]))
    assert payload.residual_norm == pytest.approx(0.125)
    assert payload.used_explicit_factor is True


def test_solve_sparse_host_direct_from_available_factor_uses_ilu_without_explicit_factor() -> None:
    calls: list[str] = []
    ilu = object()
    matrix = scipy_sparse.eye(2, format="csr")

    def direct_solve(**_kwargs):
        calls.append("direct")
        return np.asarray([9.0, 9.0]), 9.0

    def ilu_solve(**kwargs):
        calls.append("ilu")
        assert kwargs["ilu"] is ilu
        assert kwargs["a_csr_full"] is matrix
        assert kwargs["refine_steps"] == 1
        return np.asarray([-1.0, 3.0]), 0.25

    payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=None,
        explicit_sparse_operator=None,
        ilu=ilu,
        a_csr_full=matrix,
        rhs=jnp.asarray([-1.0, 3.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float32),
        refine_steps=1,
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
    )

    assert calls == ["ilu"]
    np.testing.assert_allclose(payload.x, np.asarray([-1.0, 3.0]))
    assert payload.residual_norm == pytest.approx(0.25)
    assert payload.used_explicit_factor is False


def test_apply_sparse_host_direct_polish_skips_non_float32_or_converged_result() -> None:
    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([1.0]),
        residual_norm=10.0,
        factor_dtype=np.dtype(np.float64),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([1.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=100,
        precondition_side="right",
        emit=lambda *_args: pytest.fail("unexpected emit"),
        polish_enabled=lambda **_kwargs: pytest.fail("unexpected policy"),
        parse_polish_gmres_config=lambda **_kwargs: pytest.fail("unexpected parse"),
        host_sparse_direct_polish=lambda **_kwargs: pytest.fail("unexpected polish"),
    )

    assert isinstance(payload, SparseHostDirectPolishPayload)
    assert payload.attempted is False
    assert payload.accepted is False
    assert payload.restart is None
    assert float(payload.residual_norm) == pytest.approx(10.0)


def test_apply_sparse_host_direct_polish_accepts_improved_float32_result() -> None:
    messages: list[tuple[int, str]] = []
    parse_calls: list[dict[str, object]] = []
    polish_calls: list[dict[str, object]] = []

    def parse_config(**kwargs):
        parse_calls.append(kwargs)
        return 17, 33

    def polish(**kwargs):
        polish_calls.append(kwargs)
        return np.asarray([0.5, -0.5]), 0.25

    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([1.0, -1.0]),
        residual_norm=2.0,
        factor_dtype=np.dtype(np.float32),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=100,
        precondition_side="left",
        emit=lambda level, message: messages.append((level, message)),
        polish_enabled=lambda **kwargs: kwargs["env_name"] == "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH",
        parse_polish_gmres_config=parse_config,
        host_sparse_direct_polish=polish,
    )

    assert payload.attempted is True
    assert payload.accepted is True
    assert payload.restart == 17
    assert payload.maxiter == 33
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.5, -0.5]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert parse_calls[0]["default_restart"] == 40
    assert parse_calls[0]["default_maxiter"] == 100
    assert polish_calls[0]["precondition_side"] == "left"
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            "restart=17 maxiter=33",
        )
    ]


def test_apply_sparse_host_direct_polish_rejects_nonimproving_result() -> None:
    payload = apply_sparse_host_direct_polish_if_needed(
        x=np.asarray([2.0]),
        residual_norm=2.0,
        factor_dtype=np.dtype(np.float32),
        target=1.0,
        matvec=_identity,
        rhs=jnp.asarray([2.0], dtype=jnp.float64),
        ilu=object(),
        tol=1.0e-8,
        atol=1.0e-8,
        restart=20,
        maxiter=None,
        precondition_side="right",
        emit=None,
        polish_enabled=lambda **_kwargs: True,
        parse_polish_gmres_config=lambda **_kwargs: (5, 6),
        host_sparse_direct_polish=lambda **_kwargs: (np.asarray([0.0]), 3.0),
    )

    assert payload.attempted is True
    assert payload.accepted is False
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([2.0]))
    assert float(payload.residual_norm) == pytest.approx(2.0)


def test_sparse_host_direct_fallback_payload_polishes_and_recomputes_residual_vector() -> None:
    messages: list[tuple[int, str]] = []
    explicit_factor = SimpleNamespace(solve=lambda rhs: rhs)
    explicit_operator = SimpleNamespace(matrix=scipy_sparse.diags([2.0, 4.0], format="csr"))

    def direct_solve(**kwargs):
        assert kwargs["factor_solve"] is explicit_factor.solve
        assert kwargs["operator_matrix"] is explicit_operator.matrix
        return np.asarray([1.0, 1.0]), 8.0

    def ilu_solve(**_kwargs):
        pytest.fail("explicit factor should be preferred")

    def polish(**kwargs):
        assert kwargs["precondition_side"] == "left"
        return np.asarray([1.0, 2.0]), 0.125

    payload = sparse_host_direct_fallback_payload(
        explicit_sparse_factor=explicit_factor,
        explicit_sparse_operator=explicit_operator,
        ilu=object(),
        a_csr_full=object(),
        rhs=jnp.asarray([3.0, 9.0], dtype=jnp.float64),
        factor_dtype=np.dtype(np.float32),
        refine_steps=2,
        matvec=lambda x: jnp.asarray([2.0 * x[0], 4.0 * x[1]], dtype=jnp.float64),
        target=1.0,
        tol=1.0e-8,
        atol=1.0e-8,
        restart=80,
        maxiter=120,
        precondition_side="left",
        emit=lambda level, msg: messages.append((level, msg)),
        backend_name="cpu",
        polish_enabled=lambda **_kwargs: True,
        parse_polish_gmres_config=lambda **_kwargs: (11, 22),
        direct_solve_with_refinement=direct_solve,
        ilu_solve_with_refinement=ilu_solve,
        host_sparse_direct_polish=polish,
    )

    assert isinstance(payload, SparseHostDirectFallbackPayload)
    assert payload.used_explicit_factor is True
    assert payload.polish_attempted is True
    assert payload.polish_accepted is True
    assert payload.polish_restart == 11
    assert payload.polish_maxiter == 22
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([1.0, 2.0]))
    assert float(payload.residual_norm) == pytest.approx(0.125)
    np.testing.assert_allclose(np.asarray(payload.residual_vec), np.asarray([1.0, 1.0]))
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse LU direct fallback "
            "on backend=cpu",
        ),
        (
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            "restart=11 maxiter=22",
        )
    ]


def test_sparse_host_or_ilu_factor_controls_use_direct_lu_path() -> None:
    controls = resolve_sparse_host_or_ilu_factor_controls(
        n=128,
        cache_key=("base",),
        sparse_exact_lu=True,
        use_implicit=False,
        force_host_sparse_direct=False,
        sparse_ilu_dense_max=32,
        sparse_dense_cache_max=64,
        host_sparse_direct_allowed=lambda **kwargs: kwargs["sparse_exact_lu"],
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
        sparse_factor_cache_key=lambda cache_key, dtype: (*cache_key, np.dtype(dtype).str),
        explicit_sparse_host_direct_allowed=lambda **kwargs: kwargs["active_size"] == 128,
    )

    assert controls == SparseHostOrILUFactorControls(
        host_sparse_direct_wanted=True,
        factor_dtype=np.dtype(np.float32),
        cache_key_use=("base", "<f4"),
        build_dense_factors=False,
        build_jax_factors=False,
        store_dense=False,
        explicit_sparse_allowed=True,
    )


def test_sparse_host_or_ilu_factor_controls_use_implicit_ilu_path() -> None:
    controls = resolve_sparse_host_or_ilu_factor_controls(
        n=24,
        cache_key=("ilu",),
        sparse_exact_lu=False,
        use_implicit=True,
        force_host_sparse_direct=True,
        sparse_ilu_dense_max=32,
        sparse_dense_cache_max=64,
        host_sparse_direct_allowed=lambda **_kwargs: False,
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
        sparse_factor_cache_key=lambda cache_key, dtype: (*cache_key, np.dtype(dtype).str),
        explicit_sparse_host_direct_allowed=lambda **_kwargs: True,
    )

    assert controls == SparseHostOrILUFactorControls(
        host_sparse_direct_wanted=False,
        factor_dtype=np.dtype(np.float64),
        cache_key_use=("ilu",),
        build_dense_factors=True,
        build_jax_factors=True,
        store_dense=True,
        explicit_sparse_allowed=False,
    )


def test_build_sparse_host_or_ilu_factor_prefers_explicit_sparse_when_allowed() -> None:
    calls: list[dict[str, object]] = []
    explicit_operator = SimpleNamespace(matrix="csr")
    explicit_factor = SimpleNamespace(factor="lu")

    def build_host(**kwargs):
        calls.append(kwargs)
        return explicit_operator, explicit_factor

    result = build_sparse_host_or_ilu_factor(
        SparseHostOrILUFactorBuildContext(
            matvec=_identity,
            n=3,
            dtype=np.float64,
            cache_key="cache",
            factor_dtype=np.dtype(np.float64),
            drop_tol=0.0,
            drop_rel=0.0,
            ilu_drop_tol=1.0e-6,
            fill_factor=10.0,
            build_dense_factors=False,
            build_jax_factors=False,
            store_dense=False,
            factorization="lu",
            emit=None,
            host_sparse_direct_wanted=True,
            explicit_sparse_allowed=True,
            explicit_sparse_pattern="pattern",
            build_host_sparse_direct_factor_from_matvec=build_host,
            build_sparse_ilu_from_matvec=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("ILU should not be built")
            ),
        )
    )

    assert result.used_explicit_sparse
    assert result.explicit_sparse_operator is explicit_operator
    assert result.explicit_sparse_factor is explicit_factor
    assert result.a_csr_full == "csr"
    assert result.ilu == "lu"
    assert calls[0]["pattern"] == "pattern"
    assert calls[0]["n"] == 3


def test_build_sparse_host_or_ilu_factor_uses_ilu_when_explicit_not_allowed() -> None:
    calls: list[dict[str, object]] = []

    def build_ilu(**kwargs):
        calls.append(kwargs)
        return "csr", "drop", "ilu", "dense", "l", "u", True

    result = build_sparse_host_or_ilu_factor(
        SparseHostOrILUFactorBuildContext(
            matvec=_identity,
            n=4,
            dtype=np.float64,
            cache_key="cache",
            factor_dtype=np.dtype(np.float32),
            drop_tol=1.0e-4,
            drop_rel=1.0e-5,
            ilu_drop_tol=1.0e-6,
            fill_factor=8.0,
            build_dense_factors=True,
            build_jax_factors=True,
            store_dense=True,
            factorization="ilu",
            emit=None,
            host_sparse_direct_wanted=True,
            explicit_sparse_allowed=False,
            build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("explicit host factor should not be built")
            ),
            build_sparse_ilu_from_matvec=build_ilu,
        )
    )

    assert not result.used_explicit_sparse
    assert result.explicit_sparse_operator is None
    assert result.explicit_sparse_factor is None
    assert result.a_csr_full == "csr"
    assert result.a_csr_drop == "drop"
    assert result.ilu == "ilu"
    assert result.a_dense_cache == "dense"
    assert result.l_dense == "l"
    assert result.u_dense == "u"
    assert result.l_unit_diag
    assert calls[0]["cache_key"] == "cache"
    assert calls[0]["build_jax_factors"] is True
    assert calls[0]["factorization"] == "ilu"


def test_build_sparse_ilu_preconditioner_from_cache_uses_dense_triangular() -> None:
    cache = SimpleNamespace(
        perm_r=jnp.asarray([0, 1], dtype=jnp.int32),
        inv_perm_c=jnp.asarray([0, 1], dtype=jnp.int32),
        lower_idx=None,
        lower_val=None,
        lower_diag=None,
        upper_idx=None,
        upper_val=None,
        upper_diag=None,
    )
    lower = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    upper = np.asarray([[3.0, 1.0], [0.0, 4.0]])

    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=cache,
            l_dense=lower,
            u_dense=upper,
            l_unit_diag=True,
        )
    )

    assert result.preconditioner is not None
    assert result.used_dense_triangular
    assert not result.used_padded_triangular
    rhs = np.asarray([7.0, 10.0])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, rhs))
    np.testing.assert_allclose(np.asarray(result.preconditioner(jnp.asarray(rhs))), expected)


def test_build_sparse_ilu_preconditioner_from_cache_uses_padded_triangular() -> None:
    cache = SimpleNamespace(
        perm_r=jnp.asarray([0, 1], dtype=jnp.int32),
        inv_perm_c=jnp.asarray([0, 1], dtype=jnp.int32),
        lower_idx=jnp.asarray([[-1], [0]], dtype=jnp.int32),
        lower_val=jnp.asarray([[0.0], [2.0]], dtype=jnp.float64),
        lower_diag=jnp.ones(2, dtype=jnp.float64),
        upper_idx=jnp.asarray([[1], [-1]], dtype=jnp.int32),
        upper_val=jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        upper_diag=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
    )

    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=cache,
            l_dense=None,
            u_dense=None,
            l_unit_diag=True,
        )
    )

    assert result.preconditioner is not None
    assert not result.used_dense_triangular
    assert result.used_padded_triangular
    lower = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    upper = np.asarray([[3.0, 1.0], [0.0, 4.0]])
    rhs = np.asarray([7.0, 10.0])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, rhs))
    np.testing.assert_allclose(np.asarray(result.preconditioner(jnp.asarray(rhs))), expected)


def test_build_sparse_ilu_preconditioner_from_cache_reports_unavailable() -> None:
    result = build_sparse_ilu_preconditioner_from_cache(
        SparseILUPreconditionerBuildContext(
            cache_entry=None,
            l_dense=None,
            u_dense=None,
            l_unit_diag=False,
        )
    )

    assert result.preconditioner is None
    assert not result.used_dense_triangular
    assert not result.used_padded_triangular


def test_build_sparse_host_scipy_preconditioner_uses_explicit_matrix_matvec() -> None:
    factor = SimpleNamespace(solve=lambda rhs: 0.25 * rhs)
    matrix = np.asarray([[2.0, 0.0], [0.0, 3.0]])

    result = build_sparse_host_scipy_preconditioner(
        SparseHostScipyPreconditionerBuildContext(
            ilu=factor,
            a_csr_full=matrix,
            base_matvec=lambda v: 10.0 * v,
            sparse_use_matvec=True,
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.preconditioner(jnp.asarray([4.0, 8.0]))),
        np.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        np.asarray(result.matvec(jnp.asarray([5.0, 7.0]))),
        np.asarray([10.0, 21.0]),
    )


def test_build_sparse_host_scipy_preconditioner_can_reuse_base_matvec() -> None:
    factor = SimpleNamespace(solve=lambda rhs: rhs)

    result = build_sparse_host_scipy_preconditioner(
        SparseHostScipyPreconditionerBuildContext(
            ilu=factor,
            a_csr_full=np.eye(2),
            base_matvec=lambda v: 3.0 * v,
            sparse_use_matvec=False,
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.matvec(jnp.asarray([2.0, 4.0]))),
        np.asarray([6.0, 12.0]),
    )


def test_build_sparse_host_scipy_preconditioner_raises_when_factor_missing() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        build_sparse_host_scipy_preconditioner(
            SparseHostScipyPreconditionerBuildContext(
                ilu=None,
                a_csr_full=np.eye(2),
                base_matvec=_identity,
                sparse_use_matvec=True,
                unavailable_message="missing",
            )
        )


def test_run_sparse_host_scipy_gmres_wraps_result_without_residual_vector() -> None:
    calls: list[dict[str, object]] = []

    def gmres_solver(**kwargs):
        calls.append(kwargs)
        return np.asarray([1.0, 2.0]), 0.125, (1.0, 0.125)

    result, residual_vec = run_sparse_host_scipy_gmres(
        SparseHostScipyGMRESContext(
            matvec=lambda v: 2.0 * v,
            rhs=jnp.asarray([2.0, 4.0]),
            preconditioner=lambda v: v,
            x0=jnp.asarray([0.0, 0.0]),
            tol=1.0e-8,
            atol=1.0e-12,
            restart=7,
            maxiter=11,
            precondition_side="left",
            gmres_solver=gmres_solver,
        )
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray([1.0, 2.0]))
    assert float(result.residual_norm) == pytest.approx(0.125)
    assert residual_vec is None
    assert calls[0]["restart"] == 7
    assert calls[0]["maxiter"] == 11
    assert calls[0]["precondition_side"] == "left"


def test_run_sparse_host_scipy_gmres_computes_requested_true_residual_vector() -> None:
    def gmres_solver(**_kwargs):
        return np.asarray([1.0, 2.0]), 9.0, ()

    result, residual_vec = run_sparse_host_scipy_gmres(
        SparseHostScipyGMRESContext(
            matvec=lambda v: v,
            rhs=jnp.asarray([3.0, 7.0]),
            preconditioner=lambda v: v,
            x0=jnp.asarray([0.0, 0.0]),
            tol=1.0e-8,
            atol=1.0e-12,
            restart=7,
            maxiter=None,
            precondition_side="none",
            gmres_solver=gmres_solver,
            residual_matvec=lambda v: 2.0 * v,
        )
    )

    assert float(result.residual_norm) == pytest.approx(9.0)
    np.testing.assert_allclose(np.asarray(residual_vec), np.asarray([1.0, 3.0]))


def test_run_sparse_host_retry_candidate_uses_host_direct_path() -> None:
    messages: list[tuple[int, str]] = []
    explicit_factor = SimpleNamespace(solve=lambda rhs: rhs, factor=object())
    explicit_operator = SimpleNamespace(matrix=scipy_sparse.eye(2, format="csr"))
    factor_build = SparseHostOrILUFactorBuildResult(
        explicit_sparse_operator=explicit_operator,
        explicit_sparse_factor=explicit_factor,
        a_csr_full=explicit_operator.matrix,
        a_csr_drop=explicit_operator.matrix,
        ilu=explicit_factor.factor,
        a_dense_cache=None,
        l_dense=None,
        u_dense=None,
        l_unit_diag=False,
        used_explicit_sparse=True,
    )

    def direct_solve(**_kwargs):
        return np.asarray([1.0, 2.0]), 5.0

    result = run_sparse_host_retry_candidate(
        SparseHostRetryCandidateContext(
            factor_build=factor_build,
            host_sparse_direct=True,
            host_direct_operator_pc=False,
            use_implicit=False,
            matvec=lambda x: jnp.asarray(x, dtype=jnp.float64),
            rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            x0=jnp.zeros(2, dtype=jnp.float64),
            factor_dtype=np.dtype(np.float64),
            refine_steps=3,
            target=1.0,
            tol=1.0e-8,
            atol=1.0e-8,
            restart=20,
            maxiter=30,
            precondition_side="right",
            emit=lambda level, message: messages.append((level, message)),
            backend_name="cpu",
            sparse_use_matvec=False,
            sparse_exact_lu=True,
            cache_entry=None,
            require_lower_diag=False,
            polish_enabled=lambda **_kwargs: False,
            parse_polish_gmres_config=lambda **_kwargs: (5, 6),
            direct_solve_with_refinement=direct_solve,
            ilu_solve_with_refinement=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("direct factor should be preferred")
            ),
            host_sparse_direct_polish=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("polish disabled")
            ),
            gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("GMRES should not run")
            ),
            implicit_solver=lambda _precond: (_ for _ in ()).throw(
                AssertionError("implicit solve should not run")
            ),
        )
    )

    assert isinstance(result, SparseHostRetryCandidateResult)
    assert result.result is not None
    assert result.host_sparse_direct_used is True
    assert result.preconditioner is None
    np.testing.assert_allclose(np.asarray(result.result.x), np.asarray([1.0, 2.0]))
    assert float(result.result.residual_norm) == pytest.approx(5.0)
    assert messages[0] == (
        0,
        "solve_v3_full_system_linear_gmres: host sparse LU direct fallback "
        "on backend=cpu",
    )


def test_run_sparse_host_retry_candidate_uses_scipy_gmres_path() -> None:
    messages: list[tuple[int, str]] = []
    ilu = SimpleNamespace(solve=lambda rhs: 0.5 * rhs)
    matrix = scipy_sparse.eye(2, format="csr")
    factor_build = SparseHostOrILUFactorBuildResult(
        explicit_sparse_operator=None,
        explicit_sparse_factor=None,
        a_csr_full=matrix,
        a_csr_drop=matrix,
        ilu=ilu,
        a_dense_cache=None,
        l_dense=None,
        u_dense=None,
        l_unit_diag=False,
        used_explicit_sparse=False,
    )

    def gmres_solver(**kwargs):
        assert kwargs["restart"] == 9
        assert kwargs["precondition_side"] == "left"
        return np.asarray([1.0, 2.0]), 0.25, (1.0, 0.25)

    result = run_sparse_host_retry_candidate(
        SparseHostRetryCandidateContext(
            factor_build=factor_build,
            host_sparse_direct=False,
            host_direct_operator_pc=False,
            use_implicit=False,
            matvec=lambda x: jnp.asarray(x, dtype=jnp.float64),
            rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            x0=jnp.zeros(2, dtype=jnp.float64),
            factor_dtype=np.dtype(np.float64),
            refine_steps=2,
            target=1.0,
            tol=1.0e-8,
            atol=1.0e-8,
            restart=9,
            maxiter=None,
            precondition_side="left",
            emit=lambda level, message: messages.append((level, message)),
            backend_name="cpu",
            sparse_use_matvec=False,
            sparse_exact_lu=False,
            cache_entry=None,
            require_lower_diag=False,
            polish_enabled=lambda **_kwargs: False,
            parse_polish_gmres_config=lambda **_kwargs: (5, 6),
            direct_solve_with_refinement=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("direct solve should not run")
            ),
            ilu_solve_with_refinement=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("direct ILU solve should not run")
            ),
            host_sparse_direct_polish=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("polish should not run")
            ),
            gmres_solver=gmres_solver,
            implicit_solver=lambda _precond: (_ for _ in ()).throw(
                AssertionError("implicit solve should not run")
            ),
            compute_scipy_residual_vec=False,
        )
    )

    assert isinstance(result, SparseHostRetryCandidateResult)
    assert result.result is not None
    assert result.host_sparse_direct_used is False
    assert result.preconditioner is not None
    assert result.residual_vec is None
    np.testing.assert_allclose(np.asarray(result.result.x), np.asarray([1.0, 2.0]))
    assert float(result.result.residual_norm) == pytest.approx(0.25)
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse ILU GMRES fallback",
        )
    ]


def test_build_sparse_jax_retry_preconditioner_calls_builder_and_emits_progress() -> None:
    calls: list[dict[str, object]] = []
    messages: list[tuple[int, str]] = []

    def builder(**kwargs):
        calls.append(kwargs)
        return lambda v: 0.5 * v

    preconditioner = build_sparse_jax_retry_preconditioner(
        SparseJAXRetryPreconditionerBuildContext(
            matvec=_identity,
            n=5,
            dtype=jnp.float64,
            cache_key=("cache",),
            drop_tol=1.0e-3,
            drop_rel=2.0e-3,
            reg=1.0e-8,
            omega=0.75,
            sweeps=4,
            emit=lambda level, message: messages.append((level, message)),
            builder=builder,
        )
    )

    np.testing.assert_allclose(np.asarray(preconditioner(jnp.asarray([2.0]))), [1.0])
    assert calls[0]["matvec"] is _identity
    assert calls[0]["n"] == 5
    assert calls[0]["dtype"] is jnp.float64
    assert calls[0]["cache_key"] == ("cache",)
    assert calls[0]["drop_tol"] == pytest.approx(1.0e-3)
    assert calls[0]["drop_rel"] == pytest.approx(2.0e-3)
    assert calls[0]["reg"] == pytest.approx(1.0e-8)
    assert calls[0]["omega"] == pytest.approx(0.75)
    assert calls[0]["sweeps"] == 4
    assert messages == [
        (
            0,
            "solve_v3_full_system_linear_gmres: sparse JAX Jacobi fallback "
            "(sweeps=4 omega=0.75)",
        )
    ]


def test_sparse_pc_post_minres_accepts_improved_residual_and_recomputes_pc_norm() -> (
    None
):
    messages: list[str] = []
    times = iter((1.0, 1.4))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.9, 0.25),
            (0.75,),
        )

    result = apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=lambda v: 0.5 * v,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            steps=2,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
        ),
        x=np.zeros(2),
        residual_norm=1.0,
        preconditioned_residual_norm=float("nan"),
    )

    assert result.x.tolist() == [0.5, 0.5]
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.1, 0.2]))
    assert result.preconditioned_residual_norm == pytest.approx(
        np.linalg.norm([-0.25, -0.25])
    )
    assert result.history == (0.9, 0.25)
    assert result.alphas == (0.75,)
    assert result.error is None
    assert result.solve_s == pytest.approx(0.4)
    assert any("post-minres improved residual" in msg for msg in messages)


def test_sparse_pc_post_minres_uses_custom_solver_label() -> None:
    messages: list[str] = []
    times = iter((1.0, 1.1))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.25,),
            (0.75,),
        )

    apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="none",
            steps=1,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            solver_label="xblock_sparse_pc_gmres",
        ),
        x=np.zeros(2),
        residual_norm=1.0,
        preconditioned_residual_norm=1.0,
    )

    assert any(
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres post-minres improved"
        in msg
        for msg in messages
    )


def test_sparse_pc_post_minres_if_needed_preserves_state_when_disabled_or_converged() -> None:
    def minres_correction(**_kwargs):
        raise AssertionError("post-minres should not run")

    disabled = apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=None,
            elapsed_s=lambda: 0.0,
            pc_form="right",
            steps=0,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            x=np.asarray([1.0, 2.0]),
            residual_norm=2.0,
            preconditioned_residual_norm=1.0,
            solve_s=3.0,
            target=1.0,
        )
    )
    converged = apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=_identity,
            emit=None,
            elapsed_s=lambda: 0.0,
            pc_form="right",
            steps=2,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
            x=np.asarray([1.0, 2.0]),
            residual_norm=0.5,
            preconditioned_residual_norm=0.25,
            solve_s=3.0,
            target=1.0,
        )
    )

    np.testing.assert_allclose(disabled.x, np.asarray([1.0, 2.0]))
    assert disabled.residual_norm == 2.0
    assert disabled.preconditioned_residual_norm == 1.0
    assert disabled.history == ()
    assert disabled.alphas == ()
    assert disabled.residual_before is None
    assert disabled.residual_after is None
    assert disabled.error is None
    assert disabled.solve_s == 3.0
    assert converged.residual_norm == 0.5
    assert converged.solve_s == 3.0


def test_xblock_subspace_correction_accepts_improved_residual() -> None:
    messages: list[str] = []
    times = iter((2.0, 2.25))

    def direction_builder(residual_vec):
        return (("fsavg", residual_vec),)

    def correction(**kwargs):
        assert kwargs["steps"] == 2
        assert kwargs["max_directions"] == 4
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.5, 0.25),
            (1, 2),
            ("fsavg", "angular"),
        )

    result = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.zeros(2),
            residual_norm=1.0,
            target=1.0e-6,
            direction_builder=direction_builder,
            steps=2,
            max_directions=4,
            alpha_clip=10.0,
            rcond=1.0e-12,
            min_improvement=0.0,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            correction=correction,
            correction_label="post-residual-equation",
        )
    )

    assert result.x.tolist() == [0.5, 0.5]
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.1, 0.2]))
    assert result.history == (0.5, 0.25)
    assert result.direction_counts == (1, 2)
    assert result.direction_names == ("fsavg", "angular")
    assert result.residual_before == 1.0
    assert result.solve_s == pytest.approx(0.25)
    assert any(
        "xblock_sparse_pc_gmres post-residual-equation improved" in msg
        for msg in messages
    )


def test_xblock_subspace_correction_rejects_nonimproving_residual() -> None:
    messages: list[str] = []
    times = iter((2.0, 2.25))

    def correction(**_kwargs):
        return (
            jnp.asarray([2.0, 2.0]),
            jnp.asarray([2.0, 0.0]),
            (2.0,),
            (1,),
            ("fsavg",),
        )

    result = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            x=np.asarray([1.0, 1.0]),
            residual_norm=1.0,
            target=1.0e-6,
            direction_builder=lambda residual_vec: (("fsavg", residual_vec),),
            steps=1,
            max_directions=4,
            alpha_clip=10.0,
            rcond=1.0e-12,
            min_improvement=0.0,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            correction=correction,
        )
    )

    assert result.x.tolist() == [1.0, 1.0]
    assert result.residual_norm == 1.0
    assert result.residual_after == 2.0
    assert any("xblock_sparse_pc_gmres post-coarse rejected" in msg for msg in messages)


def test_sparse_pc_post_minres_from_solve_state_updates_solve_state() -> None:
    times = iter((4.0, 4.6))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.25, 0.75]),
            jnp.asarray([0.1, 0.0]),
            (1.5, 0.1),
            (0.5,),
        )

    state = {
        "_mv_true": _identity,
        "sparse_pc_rhs": jnp.zeros(2),
        "_precond_sparse": lambda v: 2.0 * v,
        "emit": None,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: next(times)),
        "pc_form": "explicit_left",
        "sparse_pc_post_minres_steps": 2,
        "sparse_pc_post_minres_alpha_clip": 10.0,
        "sparse_pc_post_minres_min_improvement": 0.0,
        "x_np": np.zeros(2),
        "residual_norm_sparse_pc": 1.0,
        "rn_pc": float("nan"),
        "solve_s": 7.0,
        "target": 0.1,
    }

    result = apply_sparse_pc_post_minres_from_solve_state(
        state,
        minres_correction=minres_correction,
    )

    np.testing.assert_allclose(result.x, np.asarray([0.25, 0.75]))
    assert result.residual_norm == pytest.approx(0.1)
    assert result.preconditioned_residual_norm == pytest.approx(
        np.linalg.norm([-0.5, -1.5])
    )
    assert result.history == (1.5, 0.1)
    assert result.alphas == (0.5,)
    assert result.residual_before == 1.0
    assert result.residual_after == pytest.approx(0.1)
    assert result.error is None
    assert result.solve_s == pytest.approx(7.6)


def test_finalize_sparse_pc_gmres_from_solve_state_applies_polish_and_payload() -> None:
    messages: list[tuple[int, str]] = []

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([3.0, 4.0]),
            jnp.asarray([0.25, 0.0]),
            (1.0, 0.25),
            (0.5,),
        )

    state = _DefaultSparsePCDriverState(
        {
            "op": SimpleNamespace(total_size=np.int64(3)),
            "_mv_true": _identity,
            "sparse_pc_rhs": jnp.zeros(2),
            "_precond_sparse": _identity,
            "emit": lambda level, msg: messages.append((level, msg)),
            "sparse_timer": SimpleNamespace(elapsed_s=lambda: 2.5),
            "pc_form": "right",
            "x_np": np.asarray([1.0, 2.0]),
            "residual_norm_sparse_pc": 1.0,
            "rn_pc": 0.5,
            "solve_s": 7.0,
            "target": 0.5,
            "atol": 0.5,
            "tol": 0.0,
            "rhs_norm": 1.0,
            "history": (1.0,),
            "mv_count": np.int64(3),
            "pc_restart": np.int64(4),
            "pc_maxiter": np.int64(5),
            "sparse_pc_first_attempt_maxiter": np.int64(2),
            "sparse_pc_post_minres_steps": np.int64(2),
            "sparse_pc_post_minres_alpha_clip": 4.0,
            "sparse_pc_post_minres_min_improvement": 0.0,
            "sparse_pc_post_minres_alphas": (),
            "sparse_pc_post_minres_residual_before": None,
            "sparse_pc_post_minres_residual_after": None,
            "sparse_pc_post_minres_history": (),
            "sparse_pc_post_minres_error": None,
            "pc_shift": 0.0,
            "sparse_pc_factor_dtype_used": np.dtype(np.float64),
            "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
            "sparse_pc_factor_dtype_retry": None,
            "factor_preflight_enabled": False,
            "factor_preflight_required": False,
            "factor_preflight_seed_enabled": False,
            "factor_preflight_seed_used": False,
            "factor_preflight_passed": None,
            "factor_preflight_error": None,
            "factor_preflight_residual_before": None,
            "factor_preflight_residual_after": None,
            "factor_preflight_improvement_ratio": None,
            "factor_preflight_target_ratio": None,
            "factor_preflight_max_target_ratio": 8.0,
            "factor_preflight_residual_diagnostics": {},
            "fortran_reduced_sparse_pc": False,
            "sparse_pc_preconditioner_operator": "full",
            "sparse_pc_factorization": "lu",
            "sparse_pc_default_factor_kind": "ilu",
            "sparse_pc_default_ilu_fill_factor": 6.0,
            "sparse_pc_default_ilu_drop_tol": 1.0e-5,
            "sparse_pc_default_pattern_color_batch": np.int64(8),
            "preconditioner_x": np.int64(1),
            "preconditioner_x_min_l": np.int64(0),
            "preconditioner_xi": np.int64(1),
            "preconditioner_species": np.int64(1),
            "sparse_pc_permc_spec": "COLAMD",
            "sparse_pc_default_permc_spec": "COLAMD",
            "sparse_pc_use_active_dof": True,
            "sparse_pc_linear_size": np.int64(2),
            "sparse_pc_fp_dense_velocity_block": None,
            "setup_s": 0.5,
            "summary": SimpleNamespace(
                nnz=np.int64(7),
                avg_row_nnz=1.75,
                max_row_nnz=np.int64(3),
            ),
            "sparse_pattern_scope": "active_dof",
            "pattern_build_s": 0.125,
            "pc_factor_s": 0.25,
            "factor_bundle_pc": SimpleNamespace(
                factor_s=None,
                factor_nbytes_estimate=None,
                factor_nnz_estimate=None,
            ),
            "_operator_bundle_pc": None,
            "direct_tail_operator_bundle": None,
            "direct_tail_structured_max_nbytes": None,
            "direct_tail_true_active_block_species_count": None,
            "direct_tail_true_window_specs": (),
        }
    )

    payload = finalize_sparse_pc_gmres_from_solve_state(
        state,
        minres_correction=minres_correction,
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 3.0, 4.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert payload.metadata["accepted_converged"] is True
    assert payload.metadata["sparse_pc_post_minres_steps_accepted"] == 1
    assert payload.metadata["sparse_pc_post_minres_residual_after"] == pytest.approx(
        0.25
    )
    assert state["x_np"].tolist() == [1.0, 2.0]
    assert any("post-minres improved" in message for _, message in messages)
    assert any("sparse_pc_gmres complete" in message for _, message in messages)


def test_sparse_pc_gmres_finalization_state_from_solve_scope_filters_scope() -> None:
    keys = sparse_pc_gmres_finalization_solve_state_keys()
    scope_keys = sparse_pc_gmres_finalization_solve_scope_keys()
    scope = {key: object() for key in keys}
    direct_tail_metadata = {"kind": "precomputed"}
    factor_preflight_metadata = {"preflight": "precomputed"}
    pattern_metadata = {"pattern": "precomputed"}
    static_metadata = {"solver_kind": "precomputed"}
    scope["sparse_pc_direct_tail_metadata"] = direct_tail_metadata
    scope["sparse_pc_factor_preflight_metadata"] = factor_preflight_metadata
    scope["sparse_pc_pattern_metadata"] = pattern_metadata
    scope["sparse_pc_static_metadata"] = static_metadata
    scope["unrelated_solver_scratch"] = object()
    scope["direct_tail_structured_pc_metadata"] = {"kind": "raw"}
    scope["factor_preflight_enabled"] = True
    scope["summary"] = object()
    scope["sparse_pc_preconditioner_operator"] = "raw_operator"

    state = sparse_pc_gmres_finalization_state_from_solve_scope(scope)
    context_state = sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=scope["atol"],
            mv_count=scope["mv_count"],
            rhs_norm=scope["rhs_norm"],
            target=scope["target"],
            tol=scope["tol"],
            sparse_pc_direct_tail_metadata=direct_tail_metadata,
            sparse_pc_factor_preflight_metadata=factor_preflight_metadata,
            sparse_pc_pattern_metadata=pattern_metadata,
            sparse_pc_static_metadata=static_metadata,
        )
    )

    assert context_state == state
    assert tuple(state) == (
        *keys,
        "sparse_pc_direct_tail_metadata",
        "sparse_pc_factor_preflight_metadata",
        "sparse_pc_pattern_metadata",
        "sparse_pc_static_metadata",
    )
    assert "unrelated_solver_scratch" not in state
    assert "direct_tail_structured_pc_metadata" not in state
    assert "factor_preflight_enabled" not in state
    assert "summary" not in state
    assert "sparse_pc_preconditioner_operator" not in state
    for key in keys:
        assert state[key] is scope[key]
    assert state["sparse_pc_direct_tail_metadata"] is direct_tail_metadata
    assert state["sparse_pc_factor_preflight_metadata"] is factor_preflight_metadata
    assert state["sparse_pc_pattern_metadata"] is pattern_metadata
    assert state["sparse_pc_static_metadata"] is static_metadata
    assert len(keys) < len(scope_keys)

    incomplete_scope = dict(scope)
    missing = keys[0]
    incomplete_scope.pop(missing)
    with pytest.raises(KeyError, match=missing):
        sparse_pc_gmres_finalization_state_from_solve_scope(incomplete_scope)


def test_sparse_pc_direct_tail_final_metadata_uses_grouped_policy_state() -> None:
    env = {
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE": "1",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_RANK": "7",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW": "1",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS": "normal",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE": "union",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW": "1",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE": "1",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO": "12",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK": "1",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT": "3",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_WINDOWS": "4",
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_BASIS": "1",
    }
    operator_bundle = SimpleNamespace(
        metadata=SimpleNamespace(
            reason="direct_pmat",
            nnz_estimate=np.int64(123),
            csr_nbytes_estimate=np.int64(456),
        )
    )

    metadata = sparse_pc_direct_tail_final_metadata(
        SparsePCDirectTailFinalMetadataContext(
            structured_pc_preflight_required=True,
            structured_pc_preflight_required_min_size=1000,
            materialization=DirectTailMaterializationResult(
                direct_tail_default=True,
                enabled=True,
                built=True,
                error=None,
                operator_bundle=operator_bundle,
                pc_env="auto",
                direct_reduced_pmat_requested=True,
            ),
            structured_admission=DirectTailStructuredAdmissionResult(
                pc_env="active_auto",
                requested="active_native_stack",
                auto_default=False,
                fail_closed_size=500_000,
                auto_large_fail_closed=False,
                required=True,
                setup_allowed=True,
                max_mb_auto=False,
                max_mb=64.0,
                regularization=1.0e-9,
            ),
            residual_policy=resolve_direct_tail_residual_rescue_policy(env),
            true_active_policy=resolve_direct_tail_true_active_rescue_policy(env),
            coupled_coarse_policy=resolve_direct_tail_coupled_coarse_rescue_policy(env),
            true_window_specs=((1, 2, 3),),
            true_active_block_species_count=3,
            structured_max_nbytes=4 * 1024 * 1024,
            structured_pc_selected=True,
            structured_pc_reason="selected",
            structured_pc_error=None,
            structured_pc_metadata={"kind": "active_native_stack"},
            support_mode_preflight_requested=True,
            support_mode_preflight_selected=False,
            support_mode_preflight_error="not_requested",
            support_mode_preflight_metadata={"baseline": True},
            residual_coarse_selected=True,
            residual_coarse_residual_after=0.25,
            residual_coarse_error=None,
            residual_coarse_metadata={"rank": 7},
            true_coupled_coarse_requested=True,
            true_coupled_coarse_auto_selected=True,
            true_coupled_coarse_selected=True,
            true_coupled_coarse_residual_after=0.125,
            true_coupled_coarse_error=None,
            true_coupled_coarse_metadata={"windows": 4},
            true_coupled_coarse_base_improvement_override_used=True,
            true_active_submatrix_selected=True,
            true_active_submatrix_residual_after=0.5,
            true_active_submatrix_error=None,
            true_active_submatrix_metadata={"active": "submatrix"},
            true_active_column_cache_metadata={"hits": 2},
            true_active_block_selected=True,
            true_active_block_residual_after=0.375,
            true_active_block_error=None,
            true_active_block_metadata={"active": "block"},
            true_active_residual_block_selected=True,
            true_active_residual_block_residual_after=0.3125,
            true_active_residual_block_error=None,
            true_active_residual_block_metadata={"active": "residual_block"},
            true_active_residual_block_base_improvement_override_used=True,
            true_window_selected=True,
            true_window_residual_after=0.2,
            true_window_error=None,
            true_window_metadata={"window": "true"},
            residual_window_selected=True,
            residual_window_residual_after=0.3,
            residual_window_error=None,
            residual_window_metadata={"window": "residual"},
        )
    )

    assert metadata["sparse_pc_direct_tail_residual_coarse_requested"] is True
    assert metadata["sparse_pc_direct_tail_residual_coarse_selected"] is True
    assert metadata["sparse_pc_direct_tail_residual_coarse_rank"] == 7
    assert metadata["sparse_pc_direct_tail_residual_window_combine_mode"] == "union"
    assert metadata["sparse_pc_direct_tail_true_window_specs"] == ((1, 2, 3),)
    assert metadata["sparse_pc_direct_tail_true_active_block_species_count"] == 3
    assert metadata["sparse_pc_direct_tail_true_coupled_coarse_max_windows"] == 4
    assert metadata["sparse_pc_direct_tail_true_coupled_coarse_auto_target_ratio"] == pytest.approx(12.0)
    assert metadata["sparse_pc_direct_tail_true_coupled_coarse_include_angular_basis"] is True
    assert metadata["sparse_pc_direct_tail_true_active_column_cache_metadata"] == {"hits": 2}
    assert metadata["sparse_pc_direct_tail_true_active_residual_block_base_improvement_override_used"] is True
    assert metadata["sparse_pc_fortran_reduced_direct_tail_operator_reason"] == "direct_pmat"
    assert metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] == 123
    assert metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_required"] is True
    assert metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb"] == pytest.approx(4.0)
    assert metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"] == {
        "kind": "active_native_stack"
    }
    assert metadata["sparse_pc_fortran_reduced_direct_tail_support_mode_preflight_metadata"] == {
        "baseline": True
    }


def test_sparse_pc_gmres_finalization_bundle_from_solve_scope_groups_locals() -> None:
    materialization = DirectTailMaterializationResult(
        direct_tail_default=True,
        enabled=True,
        built=True,
        error=None,
        operator_bundle="operator",
        pc_env="auto",
        direct_reduced_pmat_requested=True,
    )
    structured_admission = DirectTailStructuredAdmissionResult(
        pc_env="active_auto",
        requested="active_native_stack",
        auto_default=True,
        fail_closed_size=1000,
        auto_large_fail_closed=False,
        required=True,
        setup_allowed=True,
        max_mb_auto=False,
        max_mb=8.0,
        regularization=1.0e-8,
    )
    result = SparsePCGMRESFinalResultContext(
        x=np.asarray([1.0, 2.0]),
        residual_norm=0.5,
        preconditioned_residual_norm=0.25,
        history=(1.0, 0.5),
        solve_s=3.0,
        factor_dtype_used=np.dtype(np.float64),
        factor_dtype_retry=None,
        operator_bundle="operator",
        factor_bundle="factor",
        pc_factor_s=0.2,
        setup_s=0.3,
    )
    post_minres = SparsePCPostMinresFinalizationContext(
        matvec=_identity,
        rhs=jnp.zeros(2, dtype=jnp.float64),
        preconditioner=_identity,
        emit=None,
        elapsed_s=lambda: 1.0,
        pc_form="right",
        steps=0,
        alpha_clip=1.0,
        min_improvement=0.0,
        target=1.0e-8,
    )
    dtype_retry = SparsePCFactorDtypeRetryFinalizationContext(
        factor_matvec=_identity,
        linear_size=2,
        rhs_dtype=np.dtype(np.float64),
        pattern=None,
        emit=None,
        constrained_pas_pc=False,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=True,
        default_permc_spec="COLAMD",
        default_factor_kind="splu",
        default_ilu_fill_factor=4.0,
        default_ilu_drop_tol=0.0,
        default_pattern_color_batch=8,
        x0_fallback=jnp.zeros(2, dtype=jnp.float64),
        pc_maxiter=5,
        elapsed_s=lambda: 1.0,
    )
    scope = {
        "atol": 1.0e-10,
        "mv_count": np.int64(7),
        "rhs_norm": 2.0,
        "target": 1.0e-8,
        "tol": 1.0e-9,
        "structured_pc_preflight_required": True,
        "structured_pc_preflight_required_min_size": np.int64(64),
        "direct_tail_materialization": materialization,
        "direct_tail_structured_admission": structured_admission,
        "direct_tail_residual_rescue_policy": resolve_direct_tail_residual_rescue_policy({}),
        "direct_tail_true_active_rescue_policy": resolve_direct_tail_true_active_rescue_policy({}),
        "direct_tail_true_coupled_coarse_policy": resolve_direct_tail_coupled_coarse_rescue_policy({}),
        "direct_tail_true_window_specs": ((1, 2),),
        "direct_tail_true_active_block_species_count": np.int64(2),
        "direct_tail_structured_max_nbytes": np.int64(8 * 1024 * 1024),
        "direct_tail_structured_pc_selected": True,
        "direct_tail_structured_pc_reason": "selected",
        "direct_tail_structured_pc_error": None,
        "direct_tail_structured_pc_metadata": {"kind": "active_native_stack"},
        "direct_tail_support_mode_preflight_requested": False,
        "direct_tail_support_mode_preflight_selected": False,
        "direct_tail_support_mode_preflight_error": None,
        "direct_tail_support_mode_preflight_metadata": None,
        "direct_tail_residual_coarse_selected": False,
        "direct_tail_residual_coarse_residual_after": None,
        "direct_tail_residual_coarse_error": None,
        "direct_tail_residual_coarse_metadata": None,
        "direct_tail_true_coupled_coarse_requested": False,
        "direct_tail_true_coupled_coarse_auto_selected": False,
        "direct_tail_true_coupled_coarse_selected": False,
        "direct_tail_true_coupled_coarse_residual_after": None,
        "direct_tail_true_coupled_coarse_error": None,
        "direct_tail_true_coupled_coarse_metadata": None,
        "direct_tail_true_coupled_coarse_base_improvement_override_used": False,
        "direct_tail_true_active_submatrix_selected": False,
        "direct_tail_true_active_submatrix_residual_after": None,
        "direct_tail_true_active_submatrix_error": None,
        "direct_tail_true_active_submatrix_metadata": None,
        "direct_tail_true_active_column_cache_metadata": None,
        "direct_tail_true_active_block_selected": False,
        "direct_tail_true_active_block_residual_after": None,
        "direct_tail_true_active_block_error": None,
        "direct_tail_true_active_block_metadata": None,
        "direct_tail_true_active_residual_block_selected": False,
        "direct_tail_true_active_residual_block_residual_after": None,
        "direct_tail_true_active_residual_block_error": None,
        "direct_tail_true_active_residual_block_metadata": None,
        "direct_tail_true_active_residual_block_base_improvement_override_used": False,
        "direct_tail_true_window_selected": False,
        "direct_tail_true_window_residual_after": None,
        "direct_tail_true_window_error": None,
        "direct_tail_true_window_metadata": None,
        "direct_tail_residual_window_selected": False,
        "direct_tail_residual_window_residual_after": None,
        "direct_tail_residual_window_error": None,
        "direct_tail_residual_window_metadata": None,
        "factor_preflight_enabled": True,
        "factor_preflight_required": True,
        "factor_preflight_seed_enabled": False,
        "factor_preflight_seed_used": False,
        "factor_preflight_passed": True,
        "factor_preflight_error": None,
        "factor_preflight_residual_before": 1.0,
        "factor_preflight_residual_after": 0.25,
        "factor_preflight_improvement_ratio": 4.0,
        "factor_preflight_target_ratio": 2.0,
        "factor_preflight_max_target_ratio": 8.0,
        "factor_preflight_residual_diagnostics": {"ok": True},
        "summary": SimpleNamespace(nnz=np.int64(5), avg_row_nnz=2.5, max_row_nnz=np.int64(3)),
        "sparse_pattern_scope": "active_dof",
        "pattern_build_s": 0.125,
        "op": SimpleNamespace(total_size=np.int64(4)),
        "fortran_reduced_sparse_pc": True,
        "fortran_reduced_sparse_pc_backend": "xblock",
        "fortran_reduced_sparse_pc_backend_reason": "test",
        "fortran_reduced_xblock_min_size": np.int64(16),
        "pc_restart": np.int64(3),
        "pc_maxiter": np.int64(5),
        "sparse_pc_first_attempt_maxiter": np.int64(4),
        "pc_shift": 0.0,
        "sparse_pc_factor_dtype_initial": np.dtype(np.float64),
        "sparse_pc_preconditioner_operator": "direct_pmat",
        "sparse_pc_factorization": "splu",
        "sparse_pc_default_factor_kind": "splu",
        "sparse_pc_default_ilu_fill_factor": 4.0,
        "sparse_pc_default_ilu_drop_tol": 0.0,
        "sparse_pc_default_pattern_color_batch": np.int64(8),
        "preconditioner_x": np.int64(1),
        "preconditioner_x_min_l": np.int64(0),
        "preconditioner_xi": np.int64(1),
        "preconditioner_species": np.int64(1),
        "sparse_pc_permc_spec": "COLAMD",
        "sparse_pc_default_permc_spec": "COLAMD",
        "sparse_pc_use_active_dof": True,
        "sparse_pc_linear_size": np.int64(2),
        "sparse_pc_fp_dense_velocity_block": False,
        "sparse_pc_factor_dtype_used": np.dtype(np.float64),
        "sparse_pc_factor_dtype_retry": None,
        "_operator_bundle_pc": "operator",
        "factor_bundle_pc": "factor",
        "pc_factor_s": 0.2,
        "setup_s": 0.3,
        "_mv_true": _identity,
        "sparse_pc_rhs": jnp.zeros(2, dtype=jnp.float64),
        "_precond_sparse": _identity,
        "emit": None,
        "sparse_timer": SimpleNamespace(elapsed_s=lambda: 1.0),
        "pc_form": "right",
        "sparse_pc_post_minres_steps": np.int64(0),
        "sparse_pc_post_minres_alpha_clip": 1.0,
        "sparse_pc_post_minres_min_improvement": 0.0,
        "_sparse_pc_factor_mv": _identity,
        "rhs": jnp.zeros(2, dtype=jnp.float64),
        "pattern": None,
        "constrained_pas_pc": False,
        "tokamak_fp_pc": False,
        "x0_sparse": jnp.zeros(2, dtype=jnp.float64),
    }

    bundle = sparse_pc_gmres_finalization_bundle_from_solve_scope(
        scope,
        result=result,
        post_minres=post_minres,
        dtype_retry=dtype_retry,
    )

    assert bundle.atol == pytest.approx(1.0e-10)
    assert bundle.direct_tail.materialization is materialization
    assert bundle.direct_tail.structured_admission is structured_admission
    assert bundle.direct_tail.true_window_specs == ((1, 2),)
    assert bundle.factor_preflight.residual_after == pytest.approx(0.25)
    assert bundle.pattern.scope == "active_dof"
    assert bundle.static.sparse_pc_linear_size == 2
    assert bundle.result is result
    assert bundle.post_minres is post_minres
    assert bundle.dtype_retry is dtype_retry

    solve_bundle = sparse_pc_gmres_finalization_bundle_from_solve_result(
        scope,
        x=np.asarray([1.0, 2.0]),
        residual_norm=0.5,
        preconditioned_residual_norm=0.25,
        history=(1.0, 0.5),
        solve_s=3.0,
    )

    np.testing.assert_allclose(solve_bundle.result.x, np.asarray([1.0, 2.0]))
    assert solve_bundle.result.factor_dtype_used == np.dtype(np.float64)
    assert solve_bundle.result.operator_bundle == "operator"
    assert solve_bundle.post_minres.rhs.shape == (2,)
    assert solve_bundle.dtype_retry.linear_size == 2
    assert solve_bundle.dtype_retry.pc_maxiter == 5


def test_finalize_sparse_pc_gmres_bundle_builds_typed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    operator_bundle = SimpleNamespace(
        metadata=SimpleNamespace(
            reason="direct_pmat",
            nnz_estimate=np.int64(11),
            csr_nbytes_estimate=np.int64(256),
        )
    )
    factor_bundle = SimpleNamespace(
        factor_s=None,
        factor_nbytes_estimate=None,
        factor_nnz_estimate=None,
    )
    post_minres = SparsePCPostMinresFinalizationContext(
        matvec=_identity,
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        preconditioner=_identity,
        emit=None,
        elapsed_s=lambda: 1.5,
        pc_form="right",
        steps=0,
        alpha_clip=1.0,
        min_improvement=0.0,
        target=1.0e-8,
    )
    dtype_retry = SparsePCFactorDtypeRetryFinalizationContext(
        factor_matvec=_identity,
        linear_size=2,
        rhs_dtype=np.dtype(np.float64),
        pattern=None,
        emit=None,
        constrained_pas_pc=False,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=True,
        default_permc_spec="COLAMD",
        default_factor_kind="splu",
        default_ilu_fill_factor=4.0,
        default_ilu_drop_tol=0.0,
        default_pattern_color_batch=8,
        x0_fallback=jnp.zeros(2, dtype=jnp.float64),
        pc_maxiter=5,
        elapsed_s=lambda: 1.5,
    )

    def fake_finalize(context, **kwargs):
        calls["context"] = context
        calls["kwargs"] = kwargs
        return SparsePCGMRESFinalPayload(
            x=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
            residual_norm=jnp.asarray(0.25, dtype=jnp.float64),
            metadata={"accepted_converged": True},
        )

    monkeypatch.setattr(
        sparse_finalization_module,
        "finalize_sparse_pc_gmres_with_dtype_retry",
        fake_finalize,
    )

    payload = finalize_sparse_pc_gmres_bundle(
        SparsePCGMRESFinalizationBundleContext(
            atol=1.0e-10,
            mv_count=np.int64(7),
            rhs_norm=2.0,
            target=1.0e-8,
            tol=1.0e-9,
            direct_tail=SparsePCDirectTailFinalMetadataContext(
                structured_pc_preflight_required=True,
                structured_pc_preflight_required_min_size=64,
                materialization=DirectTailMaterializationResult(
                    direct_tail_default=True,
                    enabled=True,
                    built=True,
                    error=None,
                    operator_bundle=operator_bundle,
                    pc_env="auto",
                    direct_reduced_pmat_requested=True,
                ),
                structured_admission=DirectTailStructuredAdmissionResult(
                    pc_env="active_auto",
                    requested="active_native_stack",
                    auto_default=True,
                    fail_closed_size=1000,
                    auto_large_fail_closed=False,
                    required=True,
                    setup_allowed=True,
                    max_mb_auto=False,
                    max_mb=8.0,
                    regularization=1.0e-8,
                ),
                residual_policy=resolve_direct_tail_residual_rescue_policy({}),
                true_active_policy=resolve_direct_tail_true_active_rescue_policy({}),
                coupled_coarse_policy=resolve_direct_tail_coupled_coarse_rescue_policy({}),
                true_window_specs=((1, 2),),
                true_active_block_species_count=2,
                structured_max_nbytes=8 * 1024 * 1024,
                structured_pc_selected=True,
                structured_pc_reason="selected",
                structured_pc_error=None,
                structured_pc_metadata={"kind": "active_native_stack"},
                support_mode_preflight_requested=False,
                support_mode_preflight_selected=False,
                support_mode_preflight_error=None,
                support_mode_preflight_metadata=None,
                residual_coarse_selected=False,
                residual_coarse_residual_after=None,
                residual_coarse_error=None,
                residual_coarse_metadata=None,
                true_coupled_coarse_requested=False,
                true_coupled_coarse_auto_selected=False,
                true_coupled_coarse_selected=False,
                true_coupled_coarse_residual_after=None,
                true_coupled_coarse_error=None,
                true_coupled_coarse_metadata=None,
                true_coupled_coarse_base_improvement_override_used=False,
                true_active_submatrix_selected=False,
                true_active_submatrix_residual_after=None,
                true_active_submatrix_error=None,
                true_active_submatrix_metadata=None,
                true_active_column_cache_metadata=None,
                true_active_block_selected=False,
                true_active_block_residual_after=None,
                true_active_block_error=None,
                true_active_block_metadata=None,
                true_active_residual_block_selected=False,
                true_active_residual_block_residual_after=None,
                true_active_residual_block_error=None,
                true_active_residual_block_metadata=None,
                true_active_residual_block_base_improvement_override_used=False,
                true_window_selected=False,
                true_window_residual_after=None,
                true_window_error=None,
                true_window_metadata=None,
                residual_window_selected=False,
                residual_window_residual_after=None,
                residual_window_error=None,
                residual_window_metadata=None,
            ),
            factor_preflight=SparsePCFactorPreflightMetadataContext(
                enabled=True,
                required=True,
                seed_enabled=False,
                seed_used=False,
                passed=True,
                error=None,
                residual_before=1.0,
                residual_after=0.25,
                improvement_ratio=4.0,
                target_ratio=2.0,
                max_target_ratio=8.0,
                residual_diagnostics={"ok": True},
            ),
            pattern=SparsePCPatternMetadataContext(
                summary=SimpleNamespace(nnz=np.int64(5), avg_row_nnz=2.5, max_row_nnz=np.int64(3)),
                scope="active_dof",
                build_s=0.125,
            ),
            static=SparsePCGMRESStaticMetadataContext(
                op=SimpleNamespace(total_size=np.int64(4)),
                fortran_reduced_sparse_pc=True,
                fortran_reduced_sparse_pc_backend="xblock",
                fortran_reduced_sparse_pc_backend_reason="test",
                fortran_reduced_xblock_min_size=np.int64(16),
                pc_restart=np.int64(3),
                pc_maxiter=np.int64(5),
                sparse_pc_first_attempt_maxiter=np.int64(4),
                pc_shift=0.0,
                sparse_pc_factor_dtype_initial=np.dtype(np.float64),
                sparse_pc_preconditioner_operator="direct_pmat",
                sparse_pc_factorization="splu",
                sparse_pc_default_factor_kind="splu",
                sparse_pc_default_ilu_fill_factor=4.0,
                sparse_pc_default_ilu_drop_tol=0.0,
                sparse_pc_default_pattern_color_batch=np.int64(8),
                preconditioner_x=np.int64(1),
                preconditioner_x_min_l=np.int64(0),
                preconditioner_xi=np.int64(1),
                preconditioner_species=np.int64(1),
                sparse_pc_permc_spec="COLAMD",
                sparse_pc_default_permc_spec="COLAMD",
                sparse_pc_use_active_dof=True,
                sparse_pc_linear_size=np.int64(2),
                sparse_pc_fp_dense_velocity_block=False,
            ),
            result=SparsePCGMRESFinalResultContext(
                x=np.asarray([1.0, 2.0]),
                residual_norm=0.5,
                preconditioned_residual_norm=0.25,
                history=(1.0, 0.5),
                solve_s=3.0,
                factor_dtype_used=np.dtype(np.float64),
                factor_dtype_retry=None,
                operator_bundle=operator_bundle,
                factor_bundle=factor_bundle,
                pc_factor_s=0.2,
                setup_s=0.3,
            ),
            post_minres=post_minres,
            dtype_retry=dtype_retry,
        ),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
        run_sparse_pc_gmres_once_callback=lambda *_args, **_kwargs: None,
        minres_correction=lambda **_kwargs: None,
        expand_reduced=lambda x: x,
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    context = calls["context"]
    assert isinstance(context, SparsePCGMRESFinalizationContext)
    assert context.post_minres is post_minres
    assert context.dtype_retry is dtype_retry
    assert context.operator_bundle is operator_bundle
    assert context.factor_bundle is factor_bundle
    assert context.pc_factor_s == pytest.approx(0.2)
    assert context.setup_s == pytest.approx(0.3)
    np.testing.assert_allclose(context.result.x, np.asarray([1.0, 2.0]))
    state = context.diagnostic_state
    assert state["mv_count"] == np.int64(7)
    assert state["sparse_pc_pattern_metadata"]["sparse_pattern_nnz"] == 5
    assert state["sparse_pc_static_metadata"]["solver_kind"] == "fortran_reduced_pc_gmres"
    assert state["sparse_pc_static_metadata"]["sparse_pc_full_size"] == 4
    assert state["sparse_pc_factor_preflight_metadata"]["sparse_pc_factor_preflight_passed"] is True
    assert state["sparse_pc_direct_tail_metadata"]["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert state["sparse_pc_direct_tail_metadata"]["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert calls["kwargs"]["expand_reduced"](jnp.asarray([1.0])).shape == (1,)


def test_finalize_sparse_pc_gmres_with_dtype_retry_updates_copied_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    state = {
        "pc_factor_s": 2.0,
        "x_np": np.asarray([1.0, 2.0]),
    }

    def fake_retry(arg_state, **kwargs):
        calls["retry_state"] = arg_state
        calls["retry_kwargs"] = kwargs
        return SparsePCFactorDtypeRetryResult(
            retried=True,
            factor_dtype_used=np.dtype(np.float64),
            factor_dtype_retry="float64",
            operator_bundle="operator64",
            factor_bundle="factor64",
            factor_s_increment=0.75,
            setup_s=4.0,
            x=np.asarray([3.0, 4.0]),
            residual_norm=0.25,
            preconditioned_residual_norm=0.125,
            history=(1.0, 0.25),
            solve_s=5.0,
        )

    def fake_finalize(arg_state, **kwargs):
        calls["final_state"] = arg_state
        calls["final_kwargs"] = kwargs
        return SparsePCGMRESFinalPayload(
            x=jnp.asarray([0.0, 3.0, 4.0]),
            residual_norm=jnp.asarray(0.25),
            metadata={"accepted_converged": True},
        )

    monkeypatch.setattr(
        sparse_finalization_module,
        "retry_sparse_pc_factor_dtype_from_solve_state",
        fake_retry,
    )
    monkeypatch.setattr(
        sparse_finalization_module,
        "finalize_sparse_pc_gmres_from_solve_state",
        fake_finalize,
    )

    payload = finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=state,
            result=SparsePCGMRESResult(
                x=np.asarray([1.0, 2.0]),
                residual_norm=1.5,
                preconditioned_residual_norm=1.25,
                history=(2.0, 1.5),
                solve_s=3.0,
            ),
            factor_dtype_used=np.dtype(np.float32),
            factor_dtype_retry=None,
            operator_bundle="operator32",
            factor_bundle="factor32",
            pc_factor_s=2.0,
            setup_s=3.5,
        ),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
        run_sparse_pc_gmres_once_callback=lambda *_args, **_kwargs: None,
        minres_correction=lambda **_kwargs: None,
        expand_reduced=lambda x: x,
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    assert calls["retry_state"] is not state
    retry_state = calls["retry_state"]
    assert retry_state["sparse_pc_factor_dtype_used"] == np.dtype(np.float32)
    assert retry_state["sparse_pc_factor_dtype_retry"] is None
    assert retry_state["_operator_bundle_pc"] == "operator32"
    assert retry_state["factor_bundle_pc"] == "factor32"
    assert retry_state["setup_s"] == pytest.approx(3.5)
    assert retry_state["residual_norm_sparse_pc"] == pytest.approx(1.5)
    assert retry_state["rn_pc"] == pytest.approx(1.25)
    assert retry_state["history"] == (2.0, 1.5)
    assert retry_state["solve_s"] == pytest.approx(3.0)
    assert calls["final_state"] is not state
    final_state = calls["final_state"]
    assert final_state["sparse_pc_factor_dtype_used"] == np.dtype(np.float64)
    assert final_state["sparse_pc_factor_dtype_retry"] == "float64"
    assert final_state["_operator_bundle_pc"] == "operator64"
    assert final_state["factor_bundle_pc"] == "factor64"
    assert final_state["pc_factor_s"] == pytest.approx(2.75)
    assert final_state["setup_s"] == pytest.approx(4.0)
    np.testing.assert_allclose(final_state["x_np"], np.asarray([3.0, 4.0]))
    assert final_state["residual_norm_sparse_pc"] == pytest.approx(0.25)
    assert final_state["rn_pc"] == pytest.approx(0.125)
    assert final_state["history"] == (1.0, 0.25)
    assert final_state["solve_s"] == pytest.approx(5.0)
    assert state["x_np"].tolist() == [1.0, 2.0]


def test_finalize_sparse_pc_gmres_with_dtype_retry_uses_explicit_finalization_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fail_legacy_retry(*_args, **_kwargs):
        raise AssertionError("explicit dtype context should bypass legacy retry")

    def fail_legacy_finalize(*_args, **_kwargs):
        raise AssertionError("explicit post-minres context should bypass legacy finalizer")

    def fake_payload(arg_state, **kwargs):
        calls["final_state"] = arg_state
        calls["payload_kwargs"] = kwargs
        return SparsePCGMRESFinalPayload(
            x=jnp.asarray(arg_state["x_np"], dtype=jnp.float64),
            residual_norm=jnp.asarray(arg_state["residual_norm_sparse_pc"]),
            metadata={"accepted_converged": True},
        )

    monkeypatch.setattr(
        sparse_finalization_module,
        "retry_sparse_pc_factor_dtype_from_solve_state",
        fail_legacy_retry,
    )
    monkeypatch.setattr(
        sparse_finalization_module,
        "finalize_sparse_pc_gmres_from_solve_state",
        fail_legacy_finalize,
    )
    monkeypatch.setattr(
        sparse_finalization_module,
        "sparse_pc_gmres_final_payload_from_solve_state",
        fake_payload,
    )

    elapsed_values = iter((10.0, 10.5, 11.0))

    def elapsed_s() -> float:
        return next(elapsed_values)

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([2.0, 3.0], dtype=jnp.float64),
            jnp.asarray([0.1, 0.0], dtype=jnp.float64),
            (0.25, 0.1),
            (0.75,),
        )

    payload = finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state={"emit": None, "target": 1e-3},
            result=SparsePCGMRESResult(
                x=np.asarray([1.0, 2.0]),
                residual_norm=1.5,
                preconditioned_residual_norm=1.25,
                history=(2.0, 1.5),
                solve_s=3.0,
            ),
            factor_dtype_used=np.dtype(np.float64),
            factor_dtype_retry=None,
            operator_bundle="operator",
            factor_bundle="factor",
            pc_factor_s=2.0,
            setup_s=3.5,
            post_minres=SparsePCPostMinresFinalizationContext(
                matvec=_identity,
                rhs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
                preconditioner=_identity,
                emit=None,
                elapsed_s=elapsed_s,
                pc_form="right",
                steps=1,
                alpha_clip=2.0,
                min_improvement=0.0,
                target=1e-3,
            ),
            dtype_retry=SparsePCFactorDtypeRetryFinalizationContext(
                factor_matvec=_identity,
                linear_size=2,
                rhs_dtype=np.dtype(np.float64),
                pattern=None,
                emit=None,
                constrained_pas_pc=False,
                tokamak_fp_pc=False,
                fortran_reduced_sparse_pc=False,
                default_permc_spec="COLAMD",
                default_factor_kind="splu",
                default_ilu_fill_factor=4.0,
                default_ilu_drop_tol=0.0,
                default_pattern_color_batch=8,
                x0_fallback=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
                pc_maxiter=5,
                elapsed_s=elapsed_s,
            ),
        ),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
        run_sparse_pc_gmres_once_callback=lambda *_args, **_kwargs: None,
        minres_correction=minres_correction,
        expand_reduced=lambda x: x,
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    assert "final_state" in calls
    final_state = calls["final_state"]
    np.testing.assert_allclose(final_state["x_np"], np.asarray([2.0, 3.0]))
    assert final_state["residual_norm_sparse_pc"] == pytest.approx(0.1)
    assert final_state["rn_pc"] == pytest.approx(1.25)
    assert final_state["sparse_pc_post_minres_steps"] == 1
    assert final_state["sparse_pc_post_minres_alpha_clip"] == pytest.approx(2.0)
    assert final_state["sparse_pc_post_minres_min_improvement"] == pytest.approx(0.0)
    assert final_state["sparse_pc_post_minres_history"] == (0.25, 0.1)
    assert final_state["sparse_pc_post_minres_alphas"] == (0.75,)
    assert final_state["sparse_pc_post_minres_residual_before"] == pytest.approx(1.5)
    assert final_state["sparse_pc_post_minres_residual_after"] == pytest.approx(0.1)
    assert final_state["sparse_pc_post_minres_error"] is None
    assert final_state["solve_s"] == pytest.approx(3.5)
    assert final_state["sparse_pc_elapsed_s"] == pytest.approx(11.0)
    assert "_sparse_pc_factor_mv" not in final_state
    assert "pattern" not in final_state
    assert "rhs" not in final_state
    assert "x0_sparse" not in final_state


def test_finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_finalize(context, **kwargs):
        calls["context"] = context
        calls["kwargs"] = kwargs
        return SparsePCGMRESFinalPayload(
            x=jnp.asarray([1.0]),
            residual_norm=jnp.asarray(0.5),
            metadata={"delegated": True},
        )

    monkeypatch.setattr(
        sparse_finalization_module,
        "finalize_sparse_pc_gmres_with_dtype_retry",
        fake_finalize,
    )
    state = {
        "x_np": np.asarray([1.0, 2.0]),
        "residual_norm_sparse_pc": 0.5,
        "rn_pc": 0.25,
        "history": (1.0, 0.5),
        "solve_s": 2.0,
        "sparse_pc_factor_dtype_used": np.dtype(np.float64),
        "sparse_pc_factor_dtype_retry": None,
        "_operator_bundle_pc": "operator",
        "factor_bundle_pc": "factor",
        "pc_factor_s": 0.75,
        "setup_s": 1.25,
    }

    payload = finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state(
        state,
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
        run_sparse_pc_gmres_once_callback=lambda *_args, **_kwargs: None,
        minres_correction=lambda **_kwargs: None,
        expand_reduced=lambda x: x,
    )

    assert payload.metadata == {"delegated": True}
    context = calls["context"]
    assert isinstance(context, SparsePCGMRESFinalizationContext)
    assert context.diagnostic_state is state
    np.testing.assert_allclose(context.result.x, np.asarray([1.0, 2.0]))
    assert context.result.residual_norm == pytest.approx(0.5)
    assert context.result.preconditioned_residual_norm == pytest.approx(0.25)
    assert context.result.history == (1.0, 0.5)
    assert context.result.solve_s == pytest.approx(2.0)
    assert context.factor_dtype_used == np.dtype(np.float64)
    assert context.factor_dtype_retry is None
    assert context.operator_bundle == "operator"
    assert context.factor_bundle == "factor"
    assert context.pc_factor_s == pytest.approx(0.75)
    assert context.setup_s == pytest.approx(1.25)


def test_xblock_sparse_pc_final_metadata_from_solve_state_merges_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    state = {"token": object()}

    def fake_result_metadata(arg_state, *, full_size):
        calls["result_state"] = arg_state
        calls["full_size"] = full_size
        return {"core": 1, "shared": "core"}

    def fake_correction_metadata(arg_state):
        calls["correction_state"] = arg_state
        return {"correction": 2, "shared": "correction"}

    monkeypatch.setattr(
        sparse_xblock_module,
        "xblock_sparse_pc_result_diagnostics_from_solve_state",
        fake_result_metadata,
    )
    monkeypatch.setattr(
        sparse_xblock_module,
        "build_rhs1_xblock_correction_metadata_from_solve_state",
        fake_correction_metadata,
    )

    metadata = xblock_sparse_pc_final_metadata_from_solve_state(
        state,
        full_size=123,
    )

    assert calls == {
        "result_state": state,
        "full_size": 123,
        "correction_state": state,
    }
    assert metadata == {"core": 1, "correction": 2, "shared": "correction"}


def test_xblock_sparse_pc_final_metadata_state_from_solve_scope_filters_scope() -> None:
    keys = xblock_sparse_pc_final_metadata_solve_state_keys()
    scope_keys = xblock_sparse_pc_final_metadata_solve_scope_keys()
    precomputed_metadata = {
        "xblock_assembled_operator_result_metadata": {"assembled": True},
        "xblock_coarse_correction_metadata": {"coarse": True},
        "xblock_side_probe_metadata": {"side": True},
    }
    scope = {key: object() for key in keys}
    scope.update(precomputed_metadata)
    scope["unrelated_xblock_scratch"] = object()
    scope["unrelated_nested_metadata"] = {"raw": True}

    state = xblock_sparse_pc_final_metadata_state_from_solve_scope(scope)

    assert tuple(state) == (*keys, *precomputed_metadata)
    assert "unrelated_xblock_scratch" not in state
    for key in keys:
        assert state[key] is scope[key]
    for key, value in precomputed_metadata.items():
        assert state[key] is value
    assert "unrelated_nested_metadata" not in state
    assert len(keys) < len(scope_keys)

    incomplete_scope = dict(scope)
    missing = keys[-1]
    incomplete_scope.pop(missing)
    with pytest.raises(KeyError, match=missing):
        xblock_sparse_pc_final_metadata_state_from_solve_scope(incomplete_scope)


def test_xblock_sparse_pc_final_metadata_state_context_matches_solve_scope() -> None:
    keys = xblock_sparse_pc_final_metadata_solve_state_keys()
    precomputed_metadata = {
        "xblock_assembled_operator_result_metadata": {"assembled": True},
        "xblock_coarse_correction_metadata": {"coarse": True},
        "xblock_side_probe_metadata": {"side": True},
    }
    scope = {key: object() for key in keys}
    scope.update(precomputed_metadata)

    def _kwargs(cls):
        return {key: scope[key] for key in cls.__dataclass_fields__}

    context_state = xblock_sparse_pc_final_metadata_state_from_context(
        XBlockSparsePCFinalMetadataStateContext(
            core=XBlockSparsePCFinalCoreState(**_kwargs(XBlockSparsePCFinalCoreState)),
            device=XBlockSparsePCFinalDeviceState(
                **_kwargs(XBlockSparsePCFinalDeviceState)
            ),
            preflight=XBlockSparsePCFinalPreflightState(
                **_kwargs(XBlockSparsePCFinalPreflightState)
            ),
            nested=XBlockSparsePCFinalNestedMetadata(
                **_kwargs(XBlockSparsePCFinalNestedMetadata)
            ),
        )
    )
    wrapper_state = xblock_sparse_pc_final_metadata_state_from_solve_scope(scope)

    assert context_state == wrapper_state
    assert tuple(context_state) == (*keys, *precomputed_metadata)


def test_xblock_sparse_pc_final_payload_uses_explicit_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_result_metadata(arg_state, *, full_size):
        calls["accepted"] = arg_state["accepted_converged_xblock"]
        calls["full_size"] = full_size
        calls["solver_kind"] = arg_state["xblock_solver_kind"]
        calls["gmres_basis_nbytes"] = arg_state["xblock_estimated_gmres_basis_nbytes"]
        calls["x_np"] = np.asarray(arg_state["x_np"], dtype=np.float64)
        calls["diagnostic_token"] = arg_state["diagnostic_token"]
        return {"core": 1}

    def fake_correction_metadata(arg_state):
        calls["post_minres_alphas"] = arg_state["post_minres_alphas"]
        return {"correction": 2}

    monkeypatch.setattr(
        sparse_xblock_module,
        "xblock_sparse_pc_result_diagnostics_from_solve_state",
        fake_result_metadata,
    )
    monkeypatch.setattr(
        sparse_xblock_module,
        "build_rhs1_xblock_correction_metadata_from_solve_state",
        fake_correction_metadata,
    )

    payload = xblock_sparse_pc_final_payload(
        XBlockSparsePCFinalPayloadContext(
            op=SimpleNamespace(total_size=9),
            x=np.asarray([5.0, 6.0]),
            residual_norm=0.125,
            target=0.25,
            krylov_method="gmres",
            linear_size=8,
            restart=2,
            diagnostic_state={"diagnostic_token": "kept"},
            post_corrections=SimpleNamespace(
                metadata_state=lambda: {"post_minres_alphas": (0.25,)}
            ),
        ),
        expand_reduced=lambda x: jnp.asarray([x[1], x[0]], dtype=x.dtype),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([6.0, 5.0]))
    assert float(payload.residual_norm) == pytest.approx(0.125)
    np.testing.assert_allclose(calls.pop("x_np"), np.asarray([5.0, 6.0]))
    assert calls == {
        "accepted": True,
        "full_size": 9,
        "solver_kind": "xblock_sparse_pc_gmres",
        "gmres_basis_nbytes": 8 * (2 + 1 + 4) * 8,
        "diagnostic_token": "kept",
        "post_minres_alphas": (0.25,),
    }
    assert payload.metadata == {"core": 1, "correction": 2}


def test_xblock_sparse_pc_final_payload_from_solve_state_sets_gate_and_expands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_result_metadata(arg_state, *, full_size):
        calls["accepted"] = arg_state["accepted_converged_xblock"]
        calls["full_size"] = full_size
        calls["solver_kind"] = arg_state["xblock_solver_kind"]
        calls["gmres_basis_nbytes"] = arg_state["xblock_estimated_gmres_basis_nbytes"]
        calls["device_methods"] = arg_state["xblock_device_krylov_methods"]
        return {"core": 1}

    def fake_correction_metadata(arg_state):
        calls["post_minres_alphas"] = arg_state["post_minres_alphas"]
        return {"correction": 2}

    monkeypatch.setattr(
        sparse_xblock_module,
        "xblock_sparse_pc_result_diagnostics_from_solve_state",
        fake_result_metadata,
    )
    monkeypatch.setattr(
        sparse_xblock_module,
        "build_rhs1_xblock_correction_metadata_from_solve_state",
        fake_correction_metadata,
    )

    payload = xblock_sparse_pc_final_payload_from_solve_state(
        {
            "op": SimpleNamespace(total_size=7),
            "x_np": np.asarray([3.0, 4.0]),
            "residual_norm_xblock_pc": 0.25,
            "target_xblock": 0.5,
            "xblock_krylov_method": "tfqmr_jax",
            "xblock_linear_size": 10,
            "pc_restart": 3,
        },
        expand_reduced=lambda x: jnp.concatenate(
            [jnp.asarray([0.0], dtype=x.dtype), x]
        ),
        post_corrections=SimpleNamespace(
            metadata_state=lambda: {"post_minres_alphas": (0.5,)}
        ),
    )

    assert isinstance(payload, SparsePCGMRESFinalPayload)
    np.testing.assert_allclose(np.asarray(payload.x), np.asarray([0.0, 3.0, 4.0]))
    assert float(payload.residual_norm) == pytest.approx(0.25)
    assert calls == {
        "accepted": True,
        "full_size": 7,
        "solver_kind": "xblock_sparse_pc_tfqmr_jax",
        "gmres_basis_nbytes": 10 * (3 + 1 + 4) * 8,
        "device_methods": {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"},
        "post_minres_alphas": (0.5,),
    }
    assert payload.metadata == {"core": 1, "correction": 2}


@pytest.mark.parametrize(
    ("scope", "returns_residual_vec"),
    [
        ("full", True),
        ("reduced", False),
    ],
)
def test_rhs1_sparse_retry_stage_uses_measured_sparse_jax_path(
    scope: str,
    returns_residual_vec: bool,
) -> None:
    calls: dict[str, object] = {}
    current = GMRESSolveResult(x=jnp.asarray([1.0, 0.0]), residual_norm=4.0)
    candidate = GMRESSolveResult(x=jnp.asarray([0.5, 0.0]), residual_norm=0.25)
    residual_vec = jnp.asarray([4.0, 0.0])
    candidate_residual_vec = jnp.asarray([0.25, 0.0])

    def builder(**kwargs):
        calls["builder_kind"] = kwargs["cache_key"]
        calls["builder_n"] = kwargs["n"]
        return lambda x: x

    def solve_linear(**kwargs):
        calls["solve_method"] = kwargs["solve_method_val"]
        calls["x0"] = np.asarray(kwargs["x0_vec"])
        if returns_residual_vec:
            return candidate, candidate_residual_vec
        return candidate

    def measured_candidate(**kwargs):
        calls["candidate_name"] = kwargs["candidate_name"]
        calls["baseline_name"] = kwargs["baseline_name"]
        calls["returns_residual_vec"] = kwargs["returns_residual_vec"]
        candidate_output = kwargs["solve_linear"](
            matvec_fn=kwargs["matvec_fn"],
            b_vec=kwargs["b_vec"],
            precond_fn=kwargs["precond_fn"],
            x0_vec=current.x,
            tol_val=kwargs["tol"],
            atol_val=kwargs["atol"],
            restart_val=kwargs["restart"],
            maxiter_val=kwargs["maxiter"],
            solve_method_val=kwargs["solve_method"],
            precond_side=kwargs["precond_side"],
        )
        if kwargs["returns_residual_vec"]:
            result, residual = candidate_output
        else:
            result = candidate_output
            residual = residual_vec
        return result, residual, True, 0.01

    stage = run_rhs1_full_sparse_retry_stage(
        RHS1FullSparseRetryStageContext(
            op=SimpleNamespace(total_size=2),
            result=current,
            residual_vec=residual_vec,
            rhs=jnp.asarray([1.0, 0.0]),
            matvec=lambda x: x,
            target=1.0,
            tol=1e-8,
            atol=0.0,
            restart=5,
            maxiter=10,
            precondition_side="left",
            sparse_kind_use="jax",
            sparse_exact_lu=False,
            sparse_drop_tol=0.0,
            sparse_drop_rel=0.0,
            sparse_ilu_drop_tol=0.0,
            sparse_ilu_fill=1.0,
            sparse_ilu_dense_max=0,
            sparse_dense_cache_max=0,
            sparse_use_matvec=False,
            sparse_jax_reg=1e-8,
            sparse_jax_omega=0.8,
            sparse_jax_sweeps=2,
            use_implicit=False,
            use_pas_projection=False,
            active_size=2,
            large_cpu_sparse_rescue=False,
            rhs1_polish_enabled=False,
            emit=None,
            mark=lambda label: calls.setdefault("marks", []).append(label),
            cache_key_builder=lambda *args, **kwargs: kwargs["kind"],
            precond_dtype=lambda _n: jnp.float64,
            build_sparse_jax_preconditioner_from_matvec=builder,
            host_sparse_direct_allowed=lambda **_kwargs: False,
            sparse_operator_preconditioned_rescue_allowed=lambda **_kwargs: False,
            build_point_preconditioner_operator=lambda _op: None,
            apply_cached_operator=lambda _op, x: x,
            host_sparse_factor_dtype=lambda: np.float64,
            sparse_factor_cache_key=lambda **_kwargs: "unused",
            explicit_sparse_host_direct_allowed=lambda **_kwargs: False,
            maybe_full_sparse_pattern=lambda *_args, **_kwargs: None,
            build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: None,
            build_sparse_ilu_from_matvec=lambda **_kwargs: None,
            host_sparse_direct_refine_steps=lambda *_args, **_kwargs: 0,
            direct_solve_with_refinement=lambda **_kwargs: None,
            ilu_solve_with_refinement=lambda **_kwargs: None,
            host_sparse_direct_polish=lambda **_kwargs: None,
            parse_polish_gmres_config=lambda **_kwargs: (5, 10),
            gmres_solver=lambda **_kwargs: None,
            solve_linear_with_residual=solve_linear,
            run_measured_linear_candidate=measured_candidate,
            accept_sparse_retry_candidate=lambda **_kwargs: (current, residual_vec, False),
            replay_state=object(),
            solver_kind="gmres",
            peak_rss_mb=lambda: 12.5,
            sparse_ilu_cache={},
            problem_size=2 if scope == "reduced" else None,
            cache_active_size=2 if scope == "reduced" else None,
            scope=scope,
            use_active_dof_mode=scope == "reduced",
            enable_operator_preconditioned_rescue=scope == "full",
            measured_returns_residual_vec=returns_residual_vec,
        )
    )

    assert stage.result is candidate
    expected_residual_vec = candidate_residual_vec if returns_residual_vec else residual_vec
    assert stage.residual_vec is expected_residual_vec
    assert stage.dense_matrix_cache is None
    assert not stage.host_sparse_direct_used
    np.testing.assert_allclose(calls.pop("x0"), np.asarray([1.0, 0.0]))
    assert calls == {
        "marks": ["rhs1_sparse_precond_build_start", "rhs1_sparse_precond_build_done"],
        "builder_kind": "sparse_jax",
        "builder_n": 2,
        "candidate_name": f"sparse_jax_{scope}",
        "baseline_name": f"current_{scope}",
        "returns_residual_vec": returns_residual_vec,
        "solve_method": "incremental",
    }
