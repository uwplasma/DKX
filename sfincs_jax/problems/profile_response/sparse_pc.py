"""Host sparse-PC Krylov helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .diagnostics import (
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockSparsePCCoreDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    fp_xblock_global_correction_metadata,
    fp_xblock_highx_residual_correction_metadata,
    sparse_pc_factor_preflight_result_metadata,
    sparse_pc_factor_preflight_result_metadata_from_context,
    sparse_pc_gmres_static_metadata,
    sparse_pc_gmres_static_metadata_from_context,
    sparse_pc_direct_tail_result_metadata,
    sparse_pc_pattern_result_metadata_from_context,
    sparse_pc_pattern_result_metadata,
    sparse_rescue_tail_metadata,
    sparse_xblock_rescue_metadata,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_device_krylov_diagnostics,
    xblock_qi_deflated_preconditioner_diagnostics,
    xblock_qi_device_preconditioner_diagnostics,
    xblock_qi_seed_preconditioner_diagnostics,
    xblock_sparse_pc_core_diagnostics,
    xblock_sparse_pc_result_diagnostics_from_driver_state,
    xblock_side_probe_diagnostics,
)
from .residual import (
    residual_converged as profile_residual_converged,
)
from .solver_diagnostics import (
    build_rhs1_xblock_correction_metadata_from_driver_state,
)
from .sparse.direct import (
    DirectTailCoupledCoarseRescuePolicy,
    DirectTailMaterializationContext,
    DirectTailMaterializationResult,
    DirectTailResidualRescuePolicy,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredAdmissionResult,
    DirectTailStructuredBuildContext,
    DirectTailStructuredBuildResult,
    DirectTailSupportModePreflightContext,
    DirectTailSupportModePreflightResult,
    DirectTailTrueActiveRescuePolicy,
    ExplicitSparseHostDirectBranchContext,
    ExplicitSparseMinimumNormBranchContext,
    ExplicitSparseOperatorBuildPolicy,
    ExplicitSparseOperatorBuildResult,
    SparsePCDirectTailFinalMetadataContext,
    SparseHostDirectFactorSolvePayload,
    SparseHostDirectFallbackPayload,
    SparseHostDirectPayload,
    SparseHostDirectPolishPayload,
    SparseHostOrILUFactorBuildContext,
    SparseHostOrILUFactorBuildResult,
    SparseHostOrILUFactorControls,
    SparseHostRetryCandidateContext,
    SparseHostRetryCandidateResult,
    SparseHostScipyGMRESContext,
    SparseHostScipyPreconditionerBuildContext,
    SparseHostScipyPreconditionerBuildResult,
    SparseILUPreconditionerBuildContext,
    SparseILUPreconditionerBuildResult,
    SparseJAXRetryPreconditionerBuildContext,
    SparseMinimumNormPayload,
    SparseMinimumNormPolicy,
    apply_sparse_host_direct_polish_if_needed,
    build_direct_tail_materialization_setup,
    build_direct_tail_structured_preconditioner_setup,
    build_explicit_sparse_operator_from_pattern,
    build_sparse_host_or_ilu_factor,
    build_sparse_host_scipy_preconditioner,
    build_sparse_ilu_preconditioner_from_cache,
    build_sparse_jax_retry_preconditioner,
    explicit_sparse_pattern_progress_messages,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_explicit_sparse_operator_build_policy,
    resolve_sparse_host_or_ilu_factor_controls,
    resolve_sparse_minimum_norm_policy,
    run_direct_tail_support_mode_preflight,
    run_sparse_host_retry_candidate,
    run_sparse_host_scipy_gmres,
    solve_explicit_sparse_host_direct_branch,
    solve_explicit_sparse_minimum_norm_branch,
    solve_sparse_host_direct_from_available_factor,
    sparse_host_direct_fallback_payload,
    sparse_host_direct_solve_from_pattern,
    sparse_host_direct_solve_payload,
    sparse_pc_direct_tail_final_metadata,
    sparse_minimum_norm_solve_from_pattern,
    sparse_minimum_norm_solve_payload,
    sparse_minimum_norm_start_message,
    validate_explicit_sparse_host_request,
)
from .sparse.finalization import (
    SparsePCFactorDtypeRetryContext,
    SparsePCFactorDtypeRetryDecision,
    SparsePCFactorDtypeRetryFinalizationContext,
    SparsePCFactorDtypeRetryResult,
    SparsePCGMRESCompletionMessageContext,
    SparsePCGMRESFinalPayload,
    SparsePCGMRESFinalResultContext,
    SparsePCGMRESFinalizationContext,
    SparsePCGMRESResult,
    SparsePCPostMinresContext,
    SparsePCPostMinresFinalizationContext,
    SparsePCPostMinresResult,
    SparsePCPostMinresUpdateContext,
    SparsePCPostMinresUpdateResult,
    apply_sparse_pc_post_minres,
    apply_sparse_pc_post_minres_from_driver_state,
    apply_sparse_pc_post_minres_if_needed,
    emit_sparse_pc_gmres_completion_from_driver_state,
    evaluate_sparse_pc_factor_dtype_retry,
    finalize_sparse_pc_gmres_from_driver_state,
    finalize_sparse_pc_gmres_with_dtype_retry,
    finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state,
    retry_sparse_pc_factor_dtype_from_driver_state,
    retry_sparse_pc_factor_dtype_from_finalization_context,
    retry_sparse_pc_factor_dtype_if_needed,
    sparse_pc_factor_dtype_retry_initial_guess,
    sparse_pc_gmres_completion_message,
    sparse_pc_gmres_final_payload_from_driver_state,
)
from .sparse.fortran_reduced import (
    FortranReducedSparsePCBackendSetup,
    FortranReducedXBlockFactorBuildContext,
    FortranReducedXBlockFactorBuildResult,
    FortranReducedXBlockFactorPolicySetup,
    FortranReducedXBlockFinalPayloadContext,
    FortranReducedXBlockGlobalCouplingStageContext,
    FortranReducedXBlockGlobalCouplingStageResult,
    FortranReducedXBlockInitialSeedPolicySetup,
    FortranReducedXBlockInitialSeedResult,
    FortranReducedXBlockKrylovPolicySetup,
    FortranReducedXBlockKrylovSetupContext,
    FortranReducedXBlockKrylovSetupResult,
    FortranReducedXBlockKrylovSolveContext,
    FortranReducedXBlockMomentSchurStageContext,
    FortranReducedXBlockMomentSchurStageResult,
    apply_fortran_reduced_xblock_global_coupling_stage,
    apply_fortran_reduced_xblock_initial_seed,
    apply_fortran_reduced_xblock_moment_schur_stage,
    build_fortran_reduced_xblock_factor_stage,
    build_fortran_reduced_xblock_krylov_setup,
    fortran_reduced_xblock_final_payload,
    fortran_reduced_xblock_final_payload_from_driver_state,
    prepare_fortran_reduced_xblock_initial_guess,
    resolve_fortran_reduced_sparse_pc_backend,
    resolve_fortran_reduced_xblock_factor_policy,
    resolve_fortran_reduced_xblock_global_coupling_policy,
    resolve_fortran_reduced_xblock_initial_seed_policy,
    resolve_fortran_reduced_xblock_krylov_policy,
    resolve_fortran_reduced_xblock_moment_schur_policy,
    run_fortran_reduced_xblock_krylov_solve,
)
from .sparse.policy import (
    SparsePCActiveDOFSetup,
    SparsePCAutoPreflightRetryEvaluationContext,
    SparsePCAutoPreflightRetryEvaluationResult,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCAutoPreflightRetrySelectionResult,
    SparsePCEntryPolicySetup,
    SparsePCFactorPolicySetup,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCFactorPreflightEvaluationResult,
    SparsePCFactorPreflightPolicy,
    SparsePCFactorPreflightPolicyContext,
    SparsePCGMRESControlPolicy,
    SparsePCMemoryBudgetPreflightContext,
    SparsePCPatternSetupContext,
    SparsePCPatternSetupResult,
    SparsePCResidualCandidateAcceptanceContext,
    SparsePCResidualCandidateAcceptanceResult,
    _env_bool,
    _env_float,
    _env_int,
    _env_value,
    build_sparse_pc_active_dof_setup,
    build_sparse_pc_pattern_setup,
    enforce_sparse_pc_memory_budget,
    evaluate_sparse_pc_auto_preflight_retry,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    resolve_sparse_pc_entry_policy,
    resolve_sparse_pc_factor_policy,
    resolve_sparse_pc_factor_preflight_policy,
    resolve_sparse_pc_gmres_control_policy,
    select_sparse_pc_auto_preflight_retry_candidates,
)
from .sparse.krylov import (
    SparsePCGMRESContext,
    run_sparse_pc_gmres_once,
    run_sparse_pc_gmres_once_for_retry,
)
from .sparse.qi import (
    XBlockQICoarseSeedStageContext,
    XBlockQICoarseSeedStageResult,
    XBlockQIDeflatedPolicySetup,
    XBlockQIDeflatedStageContext,
    XBlockQIDeflatedStageResult,
    XBlockQIDeviceAdmissionSetup,
    XBlockQIDeviceBaseConfigSetup,
    XBlockQIDeviceEnrichmentConfigSetup,
    XBlockQIDeviceMetadataContext,
    XBlockQIDeviceMultilevelConfigSetup,
    XBlockQIDeviceOperatorReuseSetup,
    XBlockQIDeviceStageContext,
    XBlockQIDeviceStageResult,
    XBlockQIDeviceSetupConfig,
    XBlockQIDeviceSetupConfigContext,
    XBlockQIGalerkinPolicySetup,
    XBlockQIGalerkinStageContext,
    XBlockQIGalerkinStageResult,
    XBlockQIStagePipelineContext,
    XBlockQIStagePipelineResult,
    XBlockQISeedPolicySetup,
    XBlockQITwoLevelPolicySetup,
    XBlockQITwoLevelStageContext,
    XBlockQITwoLevelStageResult,
    apply_xblock_qi_coarse_seed_stage,
    apply_xblock_qi_deflated_stage,
    apply_xblock_qi_device_stage,
    apply_xblock_qi_galerkin_stage,
    apply_xblock_qi_two_level_stage,
    build_xblock_qi_stage_pipeline_context,
    build_xblock_qi_device_preconditioner_metadata,
    build_xblock_qi_device_setup_config,
    resolve_xblock_qi_deflated_policy_setup,
    resolve_xblock_qi_device_admission_setup,
    resolve_xblock_qi_device_base_config_setup,
    resolve_xblock_qi_device_enrichment_config_setup,
    resolve_xblock_qi_device_multilevel_config_setup,
    resolve_xblock_qi_device_operator_reuse_setup,
    resolve_xblock_qi_galerkin_policy_setup,
    resolve_xblock_qi_seed_policy_setup,
    resolve_xblock_qi_two_level_policy_setup,
    run_xblock_qi_preconditioner_pipeline,
)
from .sparse.xblock import (
    XBlockSparsePCFinalCoreState,
    XBlockSparsePCFinalDeviceState,
    XBlockSparsePCFinalPreflightState,
    XBlockSparsePCFinalNestedMetadata,
    XBlockSparsePCFinalMetadataStateContext,
    xblock_sparse_pc_final_metadata_driver_state_keys,
    xblock_sparse_pc_final_metadata_driver_scope_keys,
    xblock_sparse_pc_final_metadata_state_from_context,
    xblock_sparse_pc_final_metadata_state_from_driver_scope,
    XBlockSubspaceCorrectionContext,
    XBlockSubspaceCorrectionResult,
    XBlockPostSolveCorrectionContext,
    XBlockPostSolveCorrectionResult,
    XBlockPostKrylovCompletionContext,
    XBlockPostKrylovCompletionResult,
    apply_xblock_subspace_correction_if_needed,
    run_xblock_post_solve_corrections,
    complete_xblock_post_krylov_stage,
    XBlockKrylovReport,
    XBlockSparsePCCompletionContext,
    XBlockSparsePCFinalPayloadContext,
    XBlockGMRESFallbackDecision,
    XBlockGMRESFallbackContext,
    XBlockGMRESFallbackResult,
    XBlockDeviceKrylovState,
    XBlockFirstKrylovAttemptContext,
    XBlockFirstKrylovAttemptResult,
    XBlockSideProbeStageContext,
    XBlockSideProbeStageResult,
    XBlockProbeCoarseStageContext,
    XBlockProbeCoarseStageResult,
    XBlockPreflightGateContext,
    XBlockPreflightGateResult,
    XBlockKrylovControlSetupContext,
    XBlockKrylovControlSetup,
    XBlockKrylovProgressCallbacksContext,
    XBlockKrylovProgressCallbacks,
    XBlockKrylovSolveState,
    XBlockFirstKrylovSolveStateContext,
    XBlockKrylovSolveStageContext,
    XBlockKrylovSolveStageResult,
    XBlockKrylovSolveSpaceContext,
    XBlockKrylovSolveSpace,
    XBlockAugmentedKrylovBasisContext,
    XBlockAugmentedKrylovBasisResult,
    XBlockAugmentedKrylovStageContext,
    XBlockAugmentedKrylovStageResult,
    XBlockSparsePCWorkEstimates,
    XBlockPhysicalResidual,
    xblock_krylov_report,
    apply_xblock_side_probe_stage,
    apply_xblock_probe_coarse_stage,
    evaluate_xblock_preflight_gate,
    resolve_xblock_krylov_control_setup,
    xblock_krylov_state_from_first_attempt,
    xblock_krylov_state_from_gmres_fallback,
    run_xblock_krylov_solve_stage,
    xblock_device_cycle_progress_message,
    xblock_host_krylov_progress_message,
    build_xblock_krylov_progress_callbacks,
    xblock_device_krylov_state,
    prepare_xblock_krylov_solve_space,
    prepare_xblock_augmented_krylov_basis,
    apply_xblock_augmented_krylov_stage,
    run_xblock_first_krylov_attempt,
    xblock_gmres_fallback_decision,
    run_xblock_gmres_fallback_if_needed,
    xblock_sparse_pc_work_estimates,
    xblock_sparse_pc_completion_message,
    emit_xblock_sparse_pc_completion,
    emit_xblock_sparse_pc_completion_from_driver_state,
    xblock_physical_solution_and_residual,
    FPXBlockGlobalCorrectionContext,
    FPXBlockGlobalCorrectionResult,
    FPXBlockHighXCorrectionContext,
    FPXBlockHighXCorrectionResult,
    MatvecCounter,
    SparseSXBlockRescueContext,
    SparseSXBlockRescueResult,
    SparseXBlockExplicitSeedContext,
    SparseXBlockExplicitSeedResult,
    SparseXBlockRescueAcceptanceContext,
    SparseXBlockRescueAcceptanceResult,
    SparseXBlockRescueBuildContext,
    SparseXBlockRescueBuildResult,
    SparseXBlockRescueSolveContext,
    SparseXBlockRescueSolveResult,
    XBlockInitialGuessSetup,
    XBlockGlobalCouplingPolicySetup,
    XBlockKrylovMatvecSetup,
    XBlockMomentSchurPolicySetup,
    accept_sparse_xblock_rescue_candidate,
    apply_sparse_xblock_explicit_seed,
    build_sparse_xblock_rescue_preconditioner,
    build_xblock_krylov_matvec_setup,
    prepare_xblock_initial_guess,
    run_fp_xblock_global_correction_stage,
    run_fp_xblock_highx_residual_correction_stage,
    run_sparse_sxblock_rescue_stage,
    run_sparse_xblock_rescue_solve_stage,
)


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]














































































@dataclass(frozen=True)
class SparsePCGMRESFinalizationBundleContext:
    """Typed sparse-PC finalization inputs that the driver passes as one bundle."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    direct_tail: "SparsePCDirectTailFinalMetadataContext"
    factor_preflight: SparsePCFactorPreflightMetadataContext
    pattern: SparsePCPatternMetadataContext
    static: SparsePCGMRESStaticMetadataContext
    result: SparsePCGMRESFinalResultContext
    post_minres: SparsePCPostMinresFinalizationContext
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext


def _unique_state_keys(*groups: Sequence[str]) -> tuple[str, ...]:
    """Return state keys in first-seen order without duplicate diagnostics."""

    keys: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for key in group:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


_SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS = (
    "atol",
    "mv_count",
    "rhs_norm",
    "target",
    "tol",
)

_SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS = (
    "fortran_reduced_sparse_pc",
    "fortran_reduced_sparse_pc_backend",
    "fortran_reduced_sparse_pc_backend_reason",
    "fortran_reduced_xblock_min_size",
    "op",
    "pc_maxiter",
    "pc_restart",
    "pc_shift",
    "preconditioner_species",
    "preconditioner_x",
    "preconditioner_x_min_l",
    "preconditioner_xi",
    "sparse_pc_default_factor_kind",
    "sparse_pc_default_ilu_drop_tol",
    "sparse_pc_default_ilu_fill_factor",
    "sparse_pc_default_pattern_color_batch",
    "sparse_pc_default_permc_spec",
    "sparse_pc_factor_dtype_initial",
    "sparse_pc_factorization",
    "sparse_pc_first_attempt_maxiter",
    "sparse_pc_fp_dense_velocity_block",
    "sparse_pc_linear_size",
    "sparse_pc_permc_spec",
    "sparse_pc_preconditioner_operator",
    "sparse_pc_use_active_dof",
)

_SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
)

_SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
    _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
)


def sparse_pc_gmres_finalization_driver_state_keys() -> tuple[str, ...]:
    """Return finalizer keys copied from driver scope before metadata injection."""

    return _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS


def sparse_pc_gmres_finalization_driver_scope_keys() -> tuple[str, ...]:
    """Return raw driver-scope keys needed to build sparse-PC finalization state."""

    return _SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS


@dataclass(frozen=True)
class SparsePCGMRESFinalizationStateContext:
    """Explicit sparse-PC final metadata state inputs."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    sparse_pc_direct_tail_metadata: object
    sparse_pc_factor_preflight_metadata: object
    sparse_pc_pattern_metadata: object
    sparse_pc_static_metadata: object


def sparse_pc_gmres_finalization_state_from_context(
    context: SparsePCGMRESFinalizationStateContext,
) -> dict[str, object]:
    """Build sparse-PC finalization metadata state from typed inputs."""

    return {
        "atol": context.atol,
        "mv_count": context.mv_count,
        "rhs_norm": context.rhs_norm,
        "target": context.target,
        "tol": context.tol,
        "sparse_pc_direct_tail_metadata": context.sparse_pc_direct_tail_metadata,
        "sparse_pc_factor_preflight_metadata": (
            context.sparse_pc_factor_preflight_metadata
        ),
        "sparse_pc_pattern_metadata": context.sparse_pc_pattern_metadata,
        "sparse_pc_static_metadata": context.sparse_pc_static_metadata,
    }


def sparse_pc_gmres_finalization_state_from_driver_scope(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Copy only sparse-PC finalizer state and precompute direct-tail metadata."""

    required_keys = _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS
    if "sparse_pc_static_metadata" not in scope:
        required_keys = _unique_state_keys(
            required_keys,
            _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
        )
    missing = tuple(key for key in required_keys if key not in scope)
    if missing:
        joined = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise KeyError(f"sparse-PC GMRES finalization state missing: {joined}{suffix}")
    state = {key: scope[key] for key in _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS}
    if "sparse_pc_direct_tail_metadata" in scope:
        direct_tail_metadata = scope["sparse_pc_direct_tail_metadata"]
    else:
        direct_tail_metadata = sparse_pc_direct_tail_result_metadata(scope)
    if "sparse_pc_factor_preflight_metadata" in scope:
        factor_preflight_metadata = scope["sparse_pc_factor_preflight_metadata"]
    else:
        factor_preflight_metadata = sparse_pc_factor_preflight_result_metadata(scope)
    if "sparse_pc_pattern_metadata" in scope:
        pattern_metadata = scope["sparse_pc_pattern_metadata"]
    else:
        pattern_metadata = sparse_pc_pattern_result_metadata(scope)
    if "sparse_pc_static_metadata" in scope:
        static_metadata = scope["sparse_pc_static_metadata"]
    else:
        static_metadata = sparse_pc_gmres_static_metadata(scope)
    return sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=state["atol"],
            mv_count=state["mv_count"],
            rhs_norm=state["rhs_norm"],
            target=state["target"],
            tol=state["tol"],
            sparse_pc_direct_tail_metadata=direct_tail_metadata,
            sparse_pc_factor_preflight_metadata=factor_preflight_metadata,
            sparse_pc_pattern_metadata=pattern_metadata,
            sparse_pc_static_metadata=static_metadata,
        )
    )


def sparse_pc_gmres_finalization_bundle_from_driver_scope(
    scope: Mapping[str, object],
    *,
    result: SparsePCGMRESFinalResultContext,
    post_minres: SparsePCPostMinresFinalizationContext,
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the typed sparse-PC finalization bundle from driver-local names."""

    return SparsePCGMRESFinalizationBundleContext(
        atol=scope["atol"],
        mv_count=scope["mv_count"],
        rhs_norm=scope["rhs_norm"],
        target=scope["target"],
        tol=scope["tol"],
        direct_tail=SparsePCDirectTailFinalMetadataContext(
            structured_pc_preflight_required=bool(
                scope["structured_pc_preflight_required"]
            ),
            structured_pc_preflight_required_min_size=int(
                scope["structured_pc_preflight_required_min_size"]
            ),
            materialization=scope["direct_tail_materialization"],
            structured_admission=scope["direct_tail_structured_admission"],
            residual_policy=scope["direct_tail_residual_rescue_policy"],
            true_active_policy=scope["direct_tail_true_active_rescue_policy"],
            coupled_coarse_policy=scope["direct_tail_true_coupled_coarse_policy"],
            true_window_specs=tuple(
                tuple(int(value) for value in spec)
                for spec in scope["direct_tail_true_window_specs"]
            ),
            true_active_block_species_count=scope[
                "direct_tail_true_active_block_species_count"
            ],
            structured_max_nbytes=scope["direct_tail_structured_max_nbytes"],
            structured_pc_selected=bool(scope["direct_tail_structured_pc_selected"]),
            structured_pc_reason=scope["direct_tail_structured_pc_reason"],
            structured_pc_error=scope["direct_tail_structured_pc_error"],
            structured_pc_metadata=scope["direct_tail_structured_pc_metadata"],
            support_mode_preflight_requested=bool(
                scope["direct_tail_support_mode_preflight_requested"]
            ),
            support_mode_preflight_selected=bool(
                scope["direct_tail_support_mode_preflight_selected"]
            ),
            support_mode_preflight_error=scope[
                "direct_tail_support_mode_preflight_error"
            ],
            support_mode_preflight_metadata=scope[
                "direct_tail_support_mode_preflight_metadata"
            ],
            residual_coarse_selected=bool(scope["direct_tail_residual_coarse_selected"]),
            residual_coarse_residual_after=scope[
                "direct_tail_residual_coarse_residual_after"
            ],
            residual_coarse_error=scope["direct_tail_residual_coarse_error"],
            residual_coarse_metadata=scope["direct_tail_residual_coarse_metadata"],
            true_coupled_coarse_requested=bool(
                scope["direct_tail_true_coupled_coarse_requested"]
            ),
            true_coupled_coarse_auto_selected=bool(
                scope["direct_tail_true_coupled_coarse_auto_selected"]
            ),
            true_coupled_coarse_selected=bool(
                scope["direct_tail_true_coupled_coarse_selected"]
            ),
            true_coupled_coarse_residual_after=scope[
                "direct_tail_true_coupled_coarse_residual_after"
            ],
            true_coupled_coarse_error=scope["direct_tail_true_coupled_coarse_error"],
            true_coupled_coarse_metadata=scope[
                "direct_tail_true_coupled_coarse_metadata"
            ],
            true_coupled_coarse_base_improvement_override_used=bool(
                scope[
                    "direct_tail_true_coupled_coarse_base_improvement_override_used"
                ]
            ),
            true_active_submatrix_selected=bool(
                scope["direct_tail_true_active_submatrix_selected"]
            ),
            true_active_submatrix_residual_after=scope[
                "direct_tail_true_active_submatrix_residual_after"
            ],
            true_active_submatrix_error=scope[
                "direct_tail_true_active_submatrix_error"
            ],
            true_active_submatrix_metadata=scope[
                "direct_tail_true_active_submatrix_metadata"
            ],
            true_active_column_cache_metadata=scope[
                "direct_tail_true_active_column_cache_metadata"
            ],
            true_active_block_selected=bool(
                scope["direct_tail_true_active_block_selected"]
            ),
            true_active_block_residual_after=scope[
                "direct_tail_true_active_block_residual_after"
            ],
            true_active_block_error=scope["direct_tail_true_active_block_error"],
            true_active_block_metadata=scope["direct_tail_true_active_block_metadata"],
            true_active_residual_block_selected=bool(
                scope["direct_tail_true_active_residual_block_selected"]
            ),
            true_active_residual_block_residual_after=scope[
                "direct_tail_true_active_residual_block_residual_after"
            ],
            true_active_residual_block_error=scope[
                "direct_tail_true_active_residual_block_error"
            ],
            true_active_residual_block_metadata=scope[
                "direct_tail_true_active_residual_block_metadata"
            ],
            true_active_residual_block_base_improvement_override_used=bool(
                scope[
                    "direct_tail_true_active_residual_block_base_improvement_override_used"
                ]
            ),
            true_window_selected=bool(scope["direct_tail_true_window_selected"]),
            true_window_residual_after=scope["direct_tail_true_window_residual_after"],
            true_window_error=scope["direct_tail_true_window_error"],
            true_window_metadata=scope["direct_tail_true_window_metadata"],
            residual_window_selected=bool(scope["direct_tail_residual_window_selected"]),
            residual_window_residual_after=scope[
                "direct_tail_residual_window_residual_after"
            ],
            residual_window_error=scope["direct_tail_residual_window_error"],
            residual_window_metadata=scope["direct_tail_residual_window_metadata"],
        ),
        factor_preflight=SparsePCFactorPreflightMetadataContext(
            enabled=bool(scope["factor_preflight_enabled"]),
            required=bool(scope["factor_preflight_required"]),
            seed_enabled=bool(scope["factor_preflight_seed_enabled"]),
            seed_used=bool(scope["factor_preflight_seed_used"]),
            passed=scope["factor_preflight_passed"],
            error=scope["factor_preflight_error"],
            residual_before=scope["factor_preflight_residual_before"],
            residual_after=scope["factor_preflight_residual_after"],
            improvement_ratio=scope["factor_preflight_improvement_ratio"],
            target_ratio=scope["factor_preflight_target_ratio"],
            max_target_ratio=float(scope["factor_preflight_max_target_ratio"]),
            residual_diagnostics=scope["factor_preflight_residual_diagnostics"],
        ),
        pattern=SparsePCPatternMetadataContext(
            summary=scope["summary"],
            scope=scope["sparse_pattern_scope"],
            build_s=float(scope["pattern_build_s"]),
        ),
        static=SparsePCGMRESStaticMetadataContext(
            op=scope["op"],
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            fortran_reduced_sparse_pc_backend=scope[
                "fortran_reduced_sparse_pc_backend"
            ],
            fortran_reduced_sparse_pc_backend_reason=scope[
                "fortran_reduced_sparse_pc_backend_reason"
            ],
            fortran_reduced_xblock_min_size=scope["fortran_reduced_xblock_min_size"],
            pc_restart=int(scope["pc_restart"]),
            pc_maxiter=int(scope["pc_maxiter"]),
            sparse_pc_first_attempt_maxiter=int(
                scope["sparse_pc_first_attempt_maxiter"]
            ),
            pc_shift=float(scope["pc_shift"]),
            sparse_pc_factor_dtype_initial=scope["sparse_pc_factor_dtype_initial"],
            sparse_pc_preconditioner_operator=scope[
                "sparse_pc_preconditioner_operator"
            ],
            sparse_pc_factorization=scope["sparse_pc_factorization"],
            sparse_pc_default_factor_kind=scope["sparse_pc_default_factor_kind"],
            sparse_pc_default_ilu_fill_factor=float(
                scope["sparse_pc_default_ilu_fill_factor"]
            ),
            sparse_pc_default_ilu_drop_tol=float(
                scope["sparse_pc_default_ilu_drop_tol"]
            ),
            sparse_pc_default_pattern_color_batch=int(
                scope["sparse_pc_default_pattern_color_batch"]
            ),
            preconditioner_x=int(scope["preconditioner_x"]),
            preconditioner_x_min_l=int(scope["preconditioner_x_min_l"]),
            preconditioner_xi=int(scope["preconditioner_xi"]),
            preconditioner_species=int(scope["preconditioner_species"]),
            sparse_pc_permc_spec=scope["sparse_pc_permc_spec"],
            sparse_pc_default_permc_spec=scope["sparse_pc_default_permc_spec"],
            sparse_pc_use_active_dof=bool(scope["sparse_pc_use_active_dof"]),
            sparse_pc_linear_size=int(scope["sparse_pc_linear_size"]),
            sparse_pc_fp_dense_velocity_block=scope[
                "sparse_pc_fp_dense_velocity_block"
            ],
        ),
        result=result,
        post_minres=post_minres,
        dtype_retry=dtype_retry,
    )


def sparse_pc_gmres_finalization_bundle_from_driver_result(
    scope: Mapping[str, object],
    *,
    x: np.ndarray,
    residual_norm: float,
    preconditioned_residual_norm: float,
    history: Sequence[float] | None,
    solve_s: float,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the full sparse-PC finalization bundle from the first GMRES result."""

    return sparse_pc_gmres_finalization_bundle_from_driver_scope(
        scope,
        result=SparsePCGMRESFinalResultContext(
            x=np.asarray(x, dtype=np.float64),
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(preconditioned_residual_norm),
            history=tuple(float(v) for v in (history or ())),
            solve_s=float(solve_s),
            factor_dtype_used=np.dtype(scope["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=scope["sparse_pc_factor_dtype_retry"],
            operator_bundle=scope["_operator_bundle_pc"],
            factor_bundle=scope["factor_bundle_pc"],
            pc_factor_s=float(scope["pc_factor_s"]),
            setup_s=float(scope["setup_s"]),
        ),
        post_minres=SparsePCPostMinresFinalizationContext(
            matvec=scope["_mv_true"],
            rhs=scope["sparse_pc_rhs"],
            preconditioner=scope["_precond_sparse"],
            emit=scope["emit"],
            elapsed_s=scope["sparse_timer"].elapsed_s,
            pc_form=scope["pc_form"],
            steps=int(scope["sparse_pc_post_minres_steps"]),
            alpha_clip=float(scope["sparse_pc_post_minres_alpha_clip"]),
            min_improvement=float(scope["sparse_pc_post_minres_min_improvement"]),
            target=float(scope["target"]),
        ),
        dtype_retry=SparsePCFactorDtypeRetryFinalizationContext(
            factor_matvec=scope["_sparse_pc_factor_mv"],
            linear_size=int(scope["sparse_pc_linear_size"]),
            rhs_dtype=np.dtype(scope["rhs"].dtype),
            pattern=scope["pattern"],
            emit=scope["emit"],
            constrained_pas_pc=bool(scope["constrained_pas_pc"]),
            tokamak_fp_pc=bool(scope["tokamak_fp_pc"]),
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            default_permc_spec=scope["sparse_pc_default_permc_spec"],
            default_factor_kind=scope["sparse_pc_default_factor_kind"],
            default_ilu_fill_factor=float(scope["sparse_pc_default_ilu_fill_factor"]),
            default_ilu_drop_tol=float(scope["sparse_pc_default_ilu_drop_tol"]),
            default_pattern_color_batch=int(
                scope["sparse_pc_default_pattern_color_batch"]
            ),
            x0_fallback=scope["x0_sparse"],
            pc_maxiter=int(scope["pc_maxiter"]),
            elapsed_s=scope["sparse_timer"].elapsed_s,
        ),
    )












@dataclass(frozen=True)
class XBlockSparsePCSetup:
    """Setup controls for RHSMode=1 x-block sparse-PC solves."""

    xblock_drop_tol: float
    xblock_drop_rel: float
    xblock_ilu_drop_tol: float
    xblock_fill_factor: float
    xblock_lower_fill_mode: str
    xblock_lower_fill_ignored_env: bool
    xblock_preconditioner_xi: int
    force_assembled_host_fp: bool
    xblock_assembled_host_fp: bool
    xblock_krylov_env_requested: str
    xblock_krylov_env: str
    xblock_krylov_requested: str
    xblock_device_fgmres_requested: bool
    xblock_device_gmres_requested: bool
    xblock_device_bicgstab_requested: bool
    xblock_device_tfqmr_requested: bool
    xblock_device_krylov_requested: bool
    xblock_device_host_fallback_decision: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: bool
    qi_device_preconditioner_requested_for_fallback: bool
    qi_device_matrix_free_requested_for_fallback: bool
    qi_device_use_in_krylov_requested_for_fallback: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockSparsePCSidePolicySetup:
    """JAX-factor and side-preconditioner policy for x-block sparse-PC solves."""

    xblock_jax_factors_env: str
    xblock_jax_factors_requested: bool
    xblock_jax_factors: bool
    xblock_jax_factor_format: str
    xblock_jax_factor_apply: str
    xblock_device_krylov_forced_jax_factors: bool
    full_fp_3d_pc: bool
    side_env: str
    precondition_side: str
    xblock_default_right_pc: bool
    xblock_krylov_method: str
    xblock_device_fgmres_forced_right_pc: bool
    pc_restart: int
    xblock_default_restart_capped: bool
    messages: tuple[tuple[int, str], ...]




@dataclass(frozen=True)
class XBlockSparsePCBranchSetup:
    """Combined x-block sparse-PC branch setup before factor construction."""

    xblock_drop_tol: float
    xblock_drop_rel: float
    xblock_ilu_drop_tol: float
    xblock_fill_factor: float
    xblock_lower_fill_mode: str
    xblock_lower_fill_ignored_env: bool
    xblock_preconditioner_xi: int
    force_assembled_host_fp: bool
    xblock_assembled_host_fp: bool
    xblock_krylov_env_requested: str
    xblock_krylov_env: str
    xblock_krylov_requested: str
    xblock_device_fgmres_requested: bool
    xblock_device_gmres_requested: bool
    xblock_device_bicgstab_requested: bool
    xblock_device_tfqmr_requested: bool
    xblock_device_krylov_requested: bool
    xblock_device_host_fallback_decision: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: bool
    qi_device_preconditioner_requested_for_fallback: bool
    qi_device_matrix_free_requested_for_fallback: bool
    qi_device_use_in_krylov_requested_for_fallback: bool
    xblock_jax_factors: bool
    xblock_jax_factor_format: str
    xblock_jax_factor_apply: str
    xblock_device_krylov_forced_jax_factors: bool
    full_fp_3d_pc: bool
    side_env: str
    precondition_side: str
    xblock_default_right_pc: bool
    xblock_krylov_method: str
    xblock_device_fgmres_forced_right_pc: bool
    pc_restart: int
    xblock_default_restart_capped: bool
    xblock_qi_device_operator_reuse_decision: object
    xblock_qi_device_operator_reuse_skip_factors: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockLocalPreconditionerBuildResult:
    """Local x-block preconditioner and timing metadata."""

    preconditioner: ArrayFn
    factor_s: float
    built: bool






@dataclass(frozen=True)
class XBlockAssembledEquilibrationSetup:
    """Row/column equilibration state for an assembled x-block operator."""

    row_enabled: bool
    row_built: bool
    row_metadata: dict[str, object]
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_enabled: bool
    col_built: bool
    col_metadata: dict[str, object]
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None
    messages: tuple[tuple[int, str], ...]


class XBlockAssembledPreflightMemoryError(MemoryError):
    """Preflight rejection that carries metadata for solver diagnostics."""

    def __init__(self, message: str, metadata: Mapping[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


XBlockAssembledPreflightError = XBlockAssembledPreflightMemoryError


@dataclass(frozen=True)
class XBlockAssembledOperatorPreflightSetup:
    """Memory-budget and structural-pattern preflight for assembled x-block operators."""

    csr_max_mb: float
    drop_tol: float
    device_enabled: bool
    device_required: bool
    max_colors: int
    csr_cap_nbytes: int
    pattern: object
    summary: object
    metadata: dict[str, object]


@dataclass(frozen=True)
class XBlockAssembledDeviceSetup:
    """Optional device-resident CSR operator setup for assembled x-block matvecs."""

    device_operator: object | None
    device_resident: bool
    validation_errors: tuple[float, ...]
    error: str | None
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockAssembledMatvecSetup:
    """Matvec closure for an assembled x-block operator."""

    matvec: ArrayFn
    location: str


@dataclass(frozen=True)
class XBlockAssembledOperatorBuildResult:
    """Optional assembled x-block operator build state."""

    matvec: ArrayFn
    built: bool
    device_resident: bool
    metadata: dict[str, object]
    device_operator: object | None
    pc_factor_increment_s: float
    row_enabled: bool
    row_built: bool
    row_metadata: dict[str, object]
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_enabled: bool
    col_built: bool
    col_metadata: dict[str, object]
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None




@dataclass(frozen=True)
class XBlockMomentSchurProbeResult:
    """Decision from probing a moment-Schur seed against the true residual."""

    used: bool
    reason: str
    residual_before: float
    residual_after: float
    improvement_ratio: float
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockTwoLevelPolicySetup:
    """Admission and build parameters for x-block two-level correction."""

    enabled: bool
    should_build: bool
    mode: str
    max_directions: int
    fsavg_lmax: int
    max_extra_units: int
    rcond: float
    include_rhs: bool




@dataclass(frozen=True)
class XBlockMomentSchurStageContext:
    """Dependencies for optional primary x-block moment-Schur setup."""

    op: object
    base_preconditioner: ArrayFn
    reduce_full: ArrayFn | None
    expand_reduced: ArrayFn | None
    policy: XBlockMomentSchurPolicySetup
    precondition_side: str
    rhs: jnp.ndarray
    matvec_no_count: ArrayFn
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]


@dataclass(frozen=True)
class XBlockMomentSchurStageResult:
    """Result from optional primary x-block moment-Schur setup."""

    preconditioner: ArrayFn
    built: bool
    used: bool
    reason: str | None
    metadata: dict[str, object]
    stats: dict[str, int]
    probe_residual_before: float | None
    probe_residual_after: float | None
    probe_improvement_ratio: float | None
    setup_s: float


@dataclass(frozen=True)
class XBlockTwoLevelStageContext:
    """Dependencies for optional primary x-block two-level setup."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockTwoLevelPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None


@dataclass(frozen=True)
class XBlockTwoLevelStageResult:
    """Result from optional primary x-block two-level setup."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float


@dataclass(frozen=True)
class XBlockGlobalCouplingStageContext:
    """Dependencies for optional primary x-block global-coupling setup."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockGlobalCouplingPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    host_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None
    device_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None


@dataclass(frozen=True)
class XBlockGlobalCouplingStageResult:
    """Result from optional primary x-block global-coupling setup."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float






























@dataclass(frozen=True)
class XBlockSeedPolicySetup:
    """Initial preconditioner seed controls for x-block Krylov solves."""

    initial_seed_enabled: bool
    moment_schur_seed_enabled: bool








def _xblock_device_flags(method: str) -> tuple[bool, bool, bool, bool, bool]:
    method_s = str(method)
    fgmres = method_s == "fgmres_jax"
    gmres = method_s == "gmres_jax"
    bicgstab = method_s == "bicgstab_jax"
    tfqmr = method_s == "tfqmr_jax"
    return fgmres, gmres, bicgstab, tfqmr, bool(fgmres or gmres or bicgstab or tfqmr)


def resolve_xblock_sparse_pc_setup(
    *,
    op: object,
    preconditioner_species: int,
    preconditioner_xi: int,
    active_size: int,
    lower_fill_mode: Callable[[str], tuple[str, bool]],
    species_decoupled_for_host_assembly: Callable[..., bool],
    assembled_host_allowed: Callable[..., bool],
    krylov_method: Callable[[str], tuple[str, bool]],
    device_host_fallback_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSetup:
    """Resolve x-block sparse-PC setup controls before factor construction."""

    if op.fblock.fp is None or op.fblock.pas is not None:
        raise NotImplementedError("solve_method='xblock_sparse_pc_gmres' currently targets full-FP RHSMode=1 systems.")

    drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL", 0.0)
    drop_rel = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_REL", 1.0e-8)
    ilu_drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ILU_DROP_TOL", 1.0e-4)
    fill_factor = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR", 10.0)
    lower_fill_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL")
    lower_fill_mode_value, lower_fill_ignored_env = lower_fill_mode(lower_fill_env)

    xblock_preconditioner_xi = int(preconditioner_xi)
    if xblock_preconditioner_xi == 0:
        xblock_preconditioner_xi = 1

    force_assembled_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ASSEMBLED_HOST").lower()
    force_assembled_host_fp = force_assembled_env not in {"0", "false", "f", "no", "off", ".false.", ".f."}
    xblock_assembled_host_fp = bool(
        (
            bool(force_assembled_host_fp)
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
            and species_decoupled_for_host_assembly(
                op=op,
                preconditioner_species=int(preconditioner_species),
            )
            and int(xblock_preconditioner_xi) == 1
            and (not bool(op.point_at_x0))
        )
        or assembled_host_allowed(
            op=op,
            preconditioner_species=int(preconditioner_species),
            preconditioner_xi=int(xblock_preconditioner_xi),
            use_implicit=False,
        )
    )

    krylov_env_requested = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV").lower()
    krylov_env = str(krylov_env_requested)
    krylov_requested, _unknown = krylov_method(krylov_env)
    (
        device_fgmres,
        device_gmres,
        device_bicgstab,
        device_tfqmr,
        device_krylov,
    ) = _xblock_device_flags(str(krylov_requested))

    fallback_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK")
    fallback_auto_disabled_by_qi_device = False
    qi_device_preconditioner = _env_bool(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", default=False)
    qi_device_matrix_free = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
        default=False,
    )
    qi_device_use_in_krylov = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV",
        default=False,
    )
    precondition_side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    fallback_env_token = fallback_env.strip().lower().replace("-", "_")
    if (
        bool(device_krylov)
        and bool(qi_device_preconditioner)
        and bool(qi_device_matrix_free)
        and bool(qi_device_use_in_krylov)
        and precondition_side_env != "none"
        and fallback_env_token in {"", "auto", "default"}
    ):
        fallback_env = "off"
        fallback_auto_disabled_by_qi_device = True

    fallback_decision = device_host_fallback_decision(
        env_value=fallback_env,
        requested_krylov_method=str(krylov_requested),
        active_size=int(active_size),
        min_active_size_env_value=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE"),
        rhs_mode=int(op.rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
        n_zeta=int(getattr(op, "n_zeta", 1)),
    )
    messages: list[tuple[int, str]] = []
    if bool(fallback_decision.used):
        krylov_env = str(fallback_decision.effective_krylov_env_value)
        krylov_requested, _unknown = krylov_method(krylov_env)
        (
            device_fgmres,
            device_gmres,
            device_bicgstab,
            device_tfqmr,
            _device_krylov_after_fallback,
        ) = _xblock_device_flags(str(krylov_requested))
        device_krylov = False
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "using non-autodiff host x-block fallback for requested device Krylov "
                f"method={fallback_decision.requested_method} "
                f"reason={fallback_decision.reason} "
                f"active_size={int(active_size)}",
            )
        )
    elif bool(fallback_decision.ignored_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK value; "
                f"using auto policy reason={fallback_decision.reason}",
            )
        )
    elif bool(fallback_auto_disabled_by_qi_device):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "automatic non-autodiff host fallback disabled by explicit matrix-free "
                "QI-device Krylov preconditioner request",
            )
        )

    return XBlockSparsePCSetup(
        xblock_drop_tol=float(drop_tol),
        xblock_drop_rel=float(drop_rel),
        xblock_ilu_drop_tol=float(ilu_drop_tol),
        xblock_fill_factor=float(fill_factor),
        xblock_lower_fill_mode=str(lower_fill_mode_value),
        xblock_lower_fill_ignored_env=bool(lower_fill_ignored_env),
        xblock_preconditioner_xi=int(xblock_preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        xblock_assembled_host_fp=bool(xblock_assembled_host_fp),
        xblock_krylov_env_requested=str(krylov_env_requested),
        xblock_krylov_env=str(krylov_env),
        xblock_krylov_requested=str(krylov_requested),
        xblock_device_fgmres_requested=bool(device_fgmres),
        xblock_device_gmres_requested=bool(device_gmres),
        xblock_device_bicgstab_requested=bool(device_bicgstab),
        xblock_device_tfqmr_requested=bool(device_tfqmr),
        xblock_device_krylov_requested=bool(device_krylov),
        xblock_device_host_fallback_decision=fallback_decision,
        xblock_device_host_fallback_auto_disabled_by_qi_device=bool(fallback_auto_disabled_by_qi_device),
        qi_device_preconditioner_requested_for_fallback=bool(qi_device_preconditioner),
        qi_device_matrix_free_requested_for_fallback=bool(qi_device_matrix_free),
        qi_device_use_in_krylov_requested_for_fallback=bool(qi_device_use_in_krylov),
        messages=tuple(messages),
    )


def _normalize_jax_factor_format(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"csr", "compact", "compact_csr", "ragged_csr"}:
        return "csr"
    return "padded"


def _normalize_jax_factor_apply(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"diag", "diagonal", "jacobi", "factor_diag", "factor_diagonal"}:
        return "diagonal"
    if token in {"identity", "none", "skip"}:
        return "identity"
    if token in {"upper", "upper_only", "u", "u_only"}:
        return "upper"
    if token in {"lower", "lower_only", "l", "l_only"}:
        return "lower"
    return "exact"


def resolve_xblock_sparse_pc_side_policy_setup(
    *,
    op: object,
    xblock_device_krylov_requested: bool,
    xblock_device_host_fallback_decision: object,
    xblock_krylov_env: str,
    pc_restart: int,
    pc_restart_env: str,
    tokamak_fp_er_pc: bool,
    active_size: int,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    resolve_xblock_policy: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSidePolicySetup:
    """Resolve x-block factor format and preconditioner-side policy."""

    jax_factors_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS").lower()
    jax_factors_requested = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS",
        default=False,
    )
    fallback_used = bool(getattr(xblock_device_host_fallback_decision, "used", False))
    jax_factors = bool(jax_factors_requested or bool(xblock_device_krylov_requested)) and not fallback_used

    messages: list[tuple[int, str]] = []
    if fallback_used and bool(jax_factors_requested):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS=1 because "
                "the non-autodiff host fallback requires host sparse factors",
            )
        )

    jax_factor_format = _normalize_jax_factor_format(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT") or "padded"
    )
    jax_factor_apply = _normalize_jax_factor_apply(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY") or "exact"
    )
    device_krylov_forced_jax_factors = bool(
        xblock_device_krylov_requested
        and jax_factors_env not in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    )

    side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    full_fp_3d_right_pc_max_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_RIGHT_PC_MAX")
    full_fp_3d_pc = bool(
        op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) > 1
    )
    xblock_policy = resolve_xblock_policy(
        precondition_side_env_value=side_env,
        krylov_env_value=str(xblock_krylov_env),
        requested_restart=int(pc_restart),
        restart_env_value=str(pc_restart_env),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        active_size=int(active_size),
        full_fp_3d_right_pc_max_env_value=str(full_fp_3d_right_pc_max_env),
        use_dkes=bool(use_dkes),
        include_xdot=bool(include_xdot_sparse_pc),
        include_electric_field_xi=bool(include_electric_field_xi_sparse_pc),
    )
    precondition_side = str(xblock_policy.precondition_side)
    xblock_default_right_pc = bool(xblock_policy.default_right_preconditioned)
    xblock_krylov_method = str(xblock_policy.krylov_method)
    device_fgmres_forced_right_pc = False
    if xblock_krylov_method == "fgmres_jax" and precondition_side == "left":
        precondition_side = "right"
        device_fgmres_forced_right_pc = True
    if bool(xblock_policy.ignored_krylov_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV={xblock_krylov_env!r}",
            )
        )

    return XBlockSparsePCSidePolicySetup(
        xblock_jax_factors_env=str(jax_factors_env),
        xblock_jax_factors_requested=bool(jax_factors_requested),
        xblock_jax_factors=bool(jax_factors),
        xblock_jax_factor_format=str(jax_factor_format),
        xblock_jax_factor_apply=str(jax_factor_apply),
        xblock_device_krylov_forced_jax_factors=bool(device_krylov_forced_jax_factors),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        side_env=str(side_env),
        precondition_side=str(precondition_side),
        xblock_default_right_pc=bool(xblock_default_right_pc),
        xblock_krylov_method=str(xblock_krylov_method),
        xblock_device_fgmres_forced_right_pc=bool(device_fgmres_forced_right_pc),
        pc_restart=int(xblock_policy.gmres_restart),
        xblock_default_restart_capped=bool(xblock_policy.restart_capped),
        messages=tuple(messages),
    )




def resolve_xblock_sparse_pc_branch_setup(
    *,
    op: object,
    preconditioner_species: int,
    preconditioner_xi: int,
    active_size: int,
    pc_restart: int,
    pc_restart_env: str,
    tokamak_fp_er_pc: bool,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    lower_fill_mode: Callable[[str], tuple[str, bool]],
    species_decoupled_for_host_assembly: Callable[..., bool],
    assembled_host_allowed: Callable[..., bool],
    krylov_method: Callable[[str], tuple[str, bool]],
    device_host_fallback_decision: Callable[..., object],
    resolve_xblock_policy: Callable[..., object],
    reuse_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCBranchSetup:
    """Resolve x-block sparse-PC branch policy as one typed setup contract."""

    setup = resolve_xblock_sparse_pc_setup(
        op=op,
        preconditioner_species=int(preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        active_size=int(active_size),
        lower_fill_mode=lower_fill_mode,
        species_decoupled_for_host_assembly=species_decoupled_for_host_assembly,
        assembled_host_allowed=assembled_host_allowed,
        krylov_method=krylov_method,
        device_host_fallback_decision=device_host_fallback_decision,
        env=env,
    )
    side = resolve_xblock_sparse_pc_side_policy_setup(
        op=op,
        xblock_device_krylov_requested=bool(setup.xblock_device_krylov_requested),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        xblock_krylov_env=str(setup.xblock_krylov_env),
        pc_restart=int(pc_restart),
        pc_restart_env=str(pc_restart_env),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        active_size=int(active_size),
        use_dkes=bool(use_dkes),
        include_xdot_sparse_pc=bool(include_xdot_sparse_pc),
        include_electric_field_xi_sparse_pc=bool(include_electric_field_xi_sparse_pc),
        resolve_xblock_policy=resolve_xblock_policy,
        env=env,
    )
    reuse = resolve_xblock_qi_device_operator_reuse_setup(
        op=op,
        xblock_krylov_method=str(side.xblock_krylov_method),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        qi_device_preconditioner_requested=bool(setup.qi_device_preconditioner_requested_for_fallback),
        qi_device_matrix_free_requested=bool(setup.qi_device_matrix_free_requested_for_fallback),
        qi_device_use_in_krylov_requested=bool(setup.qi_device_use_in_krylov_requested_for_fallback),
        precondition_side=str(side.precondition_side),
        xblock_jax_factors=bool(side.xblock_jax_factors),
        xblock_device_krylov_forced_jax_factors=bool(side.xblock_device_krylov_forced_jax_factors),
        xblock_preconditioner_xi=int(setup.xblock_preconditioner_xi),
        reuse_decision=reuse_decision,
        env=env,
    )
    return XBlockSparsePCBranchSetup(
        xblock_drop_tol=float(setup.xblock_drop_tol),
        xblock_drop_rel=float(setup.xblock_drop_rel),
        xblock_ilu_drop_tol=float(setup.xblock_ilu_drop_tol),
        xblock_fill_factor=float(setup.xblock_fill_factor),
        xblock_lower_fill_mode=str(setup.xblock_lower_fill_mode),
        xblock_lower_fill_ignored_env=bool(setup.xblock_lower_fill_ignored_env),
        xblock_preconditioner_xi=int(setup.xblock_preconditioner_xi),
        force_assembled_host_fp=bool(setup.force_assembled_host_fp),
        xblock_assembled_host_fp=bool(setup.xblock_assembled_host_fp),
        xblock_krylov_env_requested=str(setup.xblock_krylov_env_requested),
        xblock_krylov_env=str(setup.xblock_krylov_env),
        xblock_krylov_requested=str(setup.xblock_krylov_requested),
        xblock_device_fgmres_requested=bool(setup.xblock_device_fgmres_requested),
        xblock_device_gmres_requested=bool(setup.xblock_device_gmres_requested),
        xblock_device_bicgstab_requested=bool(setup.xblock_device_bicgstab_requested),
        xblock_device_tfqmr_requested=bool(setup.xblock_device_tfqmr_requested),
        xblock_device_krylov_requested=bool(setup.xblock_device_krylov_requested),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        xblock_device_host_fallback_auto_disabled_by_qi_device=bool(
            setup.xblock_device_host_fallback_auto_disabled_by_qi_device
        ),
        qi_device_preconditioner_requested_for_fallback=bool(
            setup.qi_device_preconditioner_requested_for_fallback
        ),
        qi_device_matrix_free_requested_for_fallback=bool(
            setup.qi_device_matrix_free_requested_for_fallback
        ),
        qi_device_use_in_krylov_requested_for_fallback=bool(
            setup.qi_device_use_in_krylov_requested_for_fallback
        ),
        xblock_jax_factors=bool(reuse.xblock_jax_factors),
        xblock_jax_factor_format=str(side.xblock_jax_factor_format),
        xblock_jax_factor_apply=str(side.xblock_jax_factor_apply),
        xblock_device_krylov_forced_jax_factors=bool(
            reuse.xblock_device_krylov_forced_jax_factors
        ),
        full_fp_3d_pc=bool(side.full_fp_3d_pc),
        side_env=str(side.side_env),
        precondition_side=str(side.precondition_side),
        xblock_default_right_pc=bool(side.xblock_default_right_pc),
        xblock_krylov_method=str(side.xblock_krylov_method),
        xblock_device_fgmres_forced_right_pc=bool(side.xblock_device_fgmres_forced_right_pc),
        pc_restart=int(side.pc_restart),
        xblock_default_restart_capped=bool(side.xblock_default_restart_capped),
        xblock_qi_device_operator_reuse_decision=reuse.decision,
        xblock_qi_device_operator_reuse_skip_factors=bool(reuse.skip_xblock_factors),
        messages=tuple((*setup.messages, *side.messages, *reuse.messages)),
    )


def build_xblock_local_preconditioner(
    *,
    skip_factors: bool,
    elapsed_s: Callable[[], float],
    build_preconditioner: Callable[..., ArrayFn],
    op: object,
    build_jax_factors: bool,
    preconditioner_species: int,
    preconditioner_xi: int,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    force_assembled_host_fp: bool,
    emit: EmitFn | None = None,
) -> XBlockLocalPreconditionerBuildResult:
    """Build or skip the local x-block factor preconditioner with timing."""

    factor_start_s = float(elapsed_s())
    if bool(skip_factors):

        def identity_preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(v, dtype=jnp.float64)

        return XBlockLocalPreconditionerBuildResult(
            preconditioner=identity_preconditioner,
            factor_s=float(elapsed_s()) - factor_start_s,
            built=False,
        )

    preconditioner = build_preconditioner(
        op=op,
        build_jax_factors=bool(build_jax_factors),
        preconditioner_species=int(preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        ilu_drop_tol=float(ilu_drop_tol),
        fill_factor=float(fill_factor),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        emit=emit,
    )
    return XBlockLocalPreconditionerBuildResult(
        preconditioner=preconditioner,
        factor_s=float(elapsed_s()) - factor_start_s,
        built=True,
    )




def _normalized_equilibration_norm(value: str) -> str:
    norm = str(value).strip().lower().replace("-", "_")
    if norm in {"inf", "max", "maximum"}:
        return "linf"
    if norm in {"linf", "l1", "l2"}:
        return norm
    return "linf"


def build_xblock_assembled_equilibration_setup(
    *,
    assembled_matrix: object,
    xblock_linear_size: int,
    elapsed_s: Callable[[], float],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledEquilibrationSetup:
    """Build optional row/column scaling for assembled x-block Krylov operators."""

    col_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE",
        default=False,
    )
    row_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE",
        default=bool(col_enabled),
    )
    row_metadata: dict[str, object] = {}
    col_metadata: dict[str, object] = {}
    row_scale_jnp: jnp.ndarray | None = None
    inv_row_scale_jnp: jnp.ndarray | None = None
    col_scale_jnp: jnp.ndarray | None = None
    inv_col_scale_jnp: jnp.ndarray | None = None
    messages: list[tuple[int, str]] = []
    row_built = False
    col_built = False
    if not bool(row_enabled):
        return XBlockAssembledEquilibrationSetup(
            row_enabled=bool(row_enabled),
            row_built=False,
            row_metadata=row_metadata,
            row_scale=None,
            inv_row_scale=None,
            col_enabled=bool(col_enabled),
            col_built=False,
            col_metadata=col_metadata,
            col_scale=None,
            inv_col_scale=None,
            messages=(),
        )

    row_start_s = float(elapsed_s())
    norm = _normalized_equilibration_norm(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM") or "linf"
    )
    floor = _env_float(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_FLOOR",
        default=1.0e-14,
    )
    floor = max(0.0, float(floor))
    max_scale = max(
        1.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_MAX_SCALE",
            default=1.0e8,
        ),
    )
    assembled_csr = assembled_matrix.tocsr()
    abs_csr = abs(assembled_csr)
    if norm == "l1":
        row_norm = np.asarray(abs_csr.sum(axis=1), dtype=np.float64).reshape((-1,))
    elif norm == "l2":
        squared_csr = assembled_csr.copy()
        squared_csr.data = np.asarray(np.abs(squared_csr.data) ** 2, dtype=np.float64)
        row_norm = np.sqrt(np.asarray(squared_csr.sum(axis=1), dtype=np.float64).reshape((-1,)))
    else:
        row_norm = np.asarray(abs_csr.max(axis=1).toarray(), dtype=np.float64).reshape((-1,))
    row_norm = np.asarray(row_norm, dtype=np.float64)
    finite_positive = np.isfinite(row_norm) & (row_norm > float(floor))
    raw_scale = np.ones_like(row_norm, dtype=np.float64)
    raw_scale[finite_positive] = 1.0 / row_norm[finite_positive]
    row_scale_np = np.clip(raw_scale, 1.0 / float(max_scale), float(max_scale))
    inv_row_scale_np = 1.0 / row_scale_np
    expected_shape = (int(xblock_linear_size),)
    if (
        row_scale_np.shape != expected_shape
        or not np.all(np.isfinite(row_scale_np))
        or not np.all(np.isfinite(inv_row_scale_np))
    ):
        raise RuntimeError("assembled x-block row equilibration produced invalid row scales")
    row_scale_jnp = jnp.asarray(row_scale_np, dtype=jnp.float64)
    inv_row_scale_jnp = jnp.asarray(inv_row_scale_np, dtype=jnp.float64)
    row_built = True

    if bool(col_enabled):
        col_start_s = float(elapsed_s())
        row_scaled_abs = abs_csr.multiply(row_scale_np[:, None])
        if norm == "l1":
            col_norm = np.asarray(row_scaled_abs.sum(axis=0), dtype=np.float64).reshape((-1,))
        elif norm == "l2":
            row_scaled_squared = assembled_csr.copy()
            row_scaled_squared.data = np.asarray(row_scaled_squared.data, dtype=np.float64) ** 2
            row_scaled_squared = row_scaled_squared.multiply((row_scale_np**2)[:, None])
            col_norm = np.sqrt(np.asarray(row_scaled_squared.sum(axis=0), dtype=np.float64).reshape((-1,)))
        else:
            col_norm = np.asarray(row_scaled_abs.max(axis=0).toarray(), dtype=np.float64).reshape((-1,))
        col_norm = np.asarray(col_norm, dtype=np.float64)
        col_finite_positive = np.isfinite(col_norm) & (col_norm > float(floor))
        raw_col_scale = np.ones_like(col_norm, dtype=np.float64)
        raw_col_scale[col_finite_positive] = 1.0 / col_norm[col_finite_positive]
        col_scale_np = np.clip(raw_col_scale, 1.0 / float(max_scale), float(max_scale))
        inv_col_scale_np = 1.0 / col_scale_np
        if (
            col_scale_np.shape != expected_shape
            or not np.all(np.isfinite(col_scale_np))
            or not np.all(np.isfinite(inv_col_scale_np))
        ):
            raise RuntimeError("assembled x-block column equilibration produced invalid column scales")
        col_scale_jnp = jnp.asarray(col_scale_np, dtype=jnp.float64)
        inv_col_scale_jnp = jnp.asarray(inv_col_scale_np, dtype=jnp.float64)
        col_built = True
        col_norm_positive = col_norm[col_finite_positive]
        col_metadata = {
            "enabled": True,
            "built": True,
            "norm": norm,
            "floor": float(floor),
            "max_scale": float(max_scale),
            "setup_s": float(elapsed_s()) - col_start_s,
            "zero_or_tiny_columns": int(col_norm.size - np.count_nonzero(col_finite_positive)),
            "col_norm_min": float(np.min(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_norm_max": float(np.max(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_scale_min": float(np.min(col_scale_np)) if col_scale_np.size else 0.0,
            "col_scale_max": float(np.max(col_scale_np)) if col_scale_np.size else 0.0,
        }

    row_norm_positive = row_norm[finite_positive]
    row_metadata = {
        "enabled": True,
        "built": True,
        "norm": norm,
        "floor": float(floor),
        "max_scale": float(max_scale),
        "setup_s": float(elapsed_s()) - row_start_s,
        "zero_or_tiny_rows": int(row_norm.size - np.count_nonzero(finite_positive)),
        "row_norm_min": float(np.min(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_norm_max": float(np.max(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_scale_min": float(np.min(row_scale_np)) if row_scale_np.size else 0.0,
        "row_scale_max": float(np.max(row_scale_np)) if row_scale_np.size else 0.0,
        "column_equilibration": bool(col_built),
    }
    messages.append(
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "assembled row equilibration built "
            f"norm={norm} "
            f"scale_range=[{float(np.min(row_scale_np)):.3e}, {float(np.max(row_scale_np)):.3e}]",
        )
    )
    if bool(col_built):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled column equilibration built "
                f"norm={norm} "
                f"scale_range=[{col_metadata['col_scale_min']:.3e}, {col_metadata['col_scale_max']:.3e}]",
            )
        )

    return XBlockAssembledEquilibrationSetup(
        row_enabled=bool(row_enabled),
        row_built=bool(row_built),
        row_metadata=row_metadata,
        row_scale=row_scale_jnp,
        inv_row_scale=inv_row_scale_jnp,
        col_enabled=bool(col_enabled),
        col_built=bool(col_built),
        col_metadata=col_metadata,
        col_scale=col_scale_jnp,
        inv_col_scale=inv_col_scale_jnp,
        messages=tuple(messages),
    )


def _csr_storage_nbytes(*, nnz: int, n_rows: int) -> int:
    return int(
        int(nnz) * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
        + (int(n_rows) + 1) * np.dtype(np.int32).itemsize
    )


def build_xblock_assembled_operator_preflight_setup(
    *,
    op: object,
    xblock_active_idx_np: np.ndarray | None,
    sparse_pc_fp_dense_velocity_block: bool | None,
    xblock_krylov_method: str,
    estimate_summary: Callable[..., object],
    full_pattern: Callable[..., object],
    active_pattern: Callable[..., object],
    summarize_pattern: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledOperatorPreflightSetup:
    """Resolve assembled-operator memory budget and structural pattern."""

    csr_max_mb = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB",
            default=2048.0,
        ),
    )
    drop_tol = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DROP_TOL",
            default=0.0,
        ),
    )
    device_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE",
        default=str(xblock_krylov_method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"},
    )
    device_required = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED",
        default=False,
    )
    max_colors = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS",
        default=512,
        minimum=1,
    )
    full_preflight = estimate_summary(
        op,
        fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
    )
    full_csr_nbytes = _csr_storage_nbytes(
        nnz=int(full_preflight.nnz),
        n_rows=int(full_preflight.shape[0]),
    )
    preflight_csr_nbytes = int(full_csr_nbytes)
    preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
    csr_cap_nbytes = int(float(csr_max_mb) * 1.0e6)
    pattern = None
    preflight_scope = "full"
    metadata: dict[str, object] = {
        "active_dof": bool(xblock_active_idx_np is not None),
        "preflight_scope": preflight_scope,
        "preflight_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_pattern_max_row_nnz_estimate": int(full_preflight.max_row_nnz),
        "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
        "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
        "preflight_full_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_full_csr_nbytes_estimate": int(full_csr_nbytes),
        "preflight_csr_max_mb": float(csr_max_mb),
        "preflight_rejected": False,
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": False,
    }
    if int(csr_cap_nbytes) <= 0:
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected non-positive CSR memory budget "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if xblock_active_idx_np is not None:
        pattern = active_pattern(
            op,
            xblock_active_idx_np,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
        active_preflight = summarize_pattern(op, pattern)
        preflight_scope = "active_dof"
        preflight_csr_nbytes = _csr_storage_nbytes(
            nnz=int(active_preflight.nnz),
            n_rows=int(active_preflight.shape[0]),
        )
        preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
        metadata.update(
            {
                "preflight_scope": preflight_scope,
                "preflight_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_pattern_max_row_nnz_estimate": int(active_preflight.max_row_nnz),
                "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
                "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
                "preflight_active_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_active_csr_nbytes_estimate": int(preflight_csr_nbytes),
            }
        )
    if int(preflight_csr_nbytes) > int(csr_cap_nbytes):
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected "
            f"{preflight_scope} CSR estimate "
            f"{int(preflight_csr_nbytes) / 1.0e6:.3g} MB > "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if pattern is None:
        pattern = full_pattern(
            op,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
    summary = summarize_pattern(op, pattern)
    return XBlockAssembledOperatorPreflightSetup(
        csr_max_mb=float(csr_max_mb),
        drop_tol=float(drop_tol),
        device_enabled=bool(device_enabled),
        device_required=bool(device_required),
        max_colors=int(max_colors),
        csr_cap_nbytes=int(csr_cap_nbytes),
        pattern=pattern,
        summary=summary,
        metadata=metadata,
    )


def build_xblock_assembled_device_setup(
    *,
    assembled_matrix: object,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    csr_cap_nbytes: int,
    device_enabled: bool,
    device_required: bool,
    validation_samples: int,
    validation_tol: float,
    device_csr_from_matrix: Callable[..., object],
    validate_device_csr_matvec: Callable[..., Sequence[float]],
) -> XBlockAssembledDeviceSetup:
    """Optionally build and validate a device CSR matvec for an assembled operator."""

    if not bool(device_enabled):
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=None,
            messages=(),
        )
    messages: list[tuple[int, str]] = []
    try:
        device_operator = device_csr_from_matrix(
            assembled_matrix,
            dtype=np.float64,
            max_nbytes=int(csr_cap_nbytes),
        )
        validation_errors = validate_device_csr_matvec(
            device_operator,
            assembled_matvec,
            samples=int(validation_samples),
            rtol=float(validation_tol),
            seed=1730,
        )
        return XBlockAssembledDeviceSetup(
            device_operator=device_operator,
            device_resident=True,
            validation_errors=tuple(float(v) for v in validation_errors),
            error=None,
            messages=(),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if bool(device_required):
            raise RuntimeError(f"assembled x-block device CSR operator failed ({error})") from exc
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled device operator disabled after build failure "
                f"({error})",
            )
        )
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=error,
            messages=tuple(messages),
    )


def build_xblock_assembled_matvec_setup(
    *,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    device_operator: object | None,
    mv_count: MatvecCounter,
    progress_every: int,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
) -> XBlockAssembledMatvecSetup:
    """Select host or device matvec closure for assembled x-block Krylov solves."""

    if device_operator is not None:
        device_matvec = device_operator.jitted_matvec()

        def matvec(v: jnp.ndarray) -> jnp.ndarray:
            mv_count.increment()
            if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"assembled_device_matvecs={int(mv_count)} "
                    f"elapsed_s={float(elapsed_s()):.3f}",
                )
            return device_matvec(jnp.asarray(v, dtype=jnp.float64))

        return XBlockAssembledMatvecSetup(matvec=matvec, location="device")

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        mv_count.increment()
        if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"assembled_host_matvecs={int(mv_count)} "
                f"elapsed_s={float(elapsed_s()):.3f}",
            )
        v_np = np.asarray(jax.device_get(v), dtype=np.float64).reshape((-1,))
        return jnp.asarray(assembled_matvec(v_np), dtype=jnp.float64)

    return XBlockAssembledMatvecSetup(matvec=matvec, location="host")


def build_xblock_assembled_operator_if_requested(
    *,
    enabled: bool,
    op: object,
    rhs_dtype: object,
    xblock_active_idx_np: np.ndarray | None,
    sparse_pc_fp_dense_velocity_block: bool | None,
    xblock_krylov_method: str,
    xblock_linear_size: int,
    true_matvec_no_count: ArrayFn,
    default_matvec: ArrayFn,
    mv_count: MatvecCounter,
    progress_every: int,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    estimate_summary: Callable[..., object],
    full_pattern: Callable[..., object],
    active_pattern: Callable[..., object],
    summarize_pattern: Callable[..., object],
    build_operator_from_pattern: Callable[..., object],
    device_csr_from_matrix: Callable[..., object],
    validate_device_csr_matvec: Callable[..., object],
    finalize_metadata: Callable[..., dict[str, object]],
    backend: str,
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledOperatorBuildResult:
    """Optionally assemble an x-block Krylov operator and return replacement matvec state."""

    if not bool(enabled):
        return XBlockAssembledOperatorBuildResult(
            matvec=default_matvec,
            built=False,
            device_resident=False,
            metadata={},
            device_operator=None,
            pc_factor_increment_s=0.0,
            row_enabled=False,
            row_built=False,
            row_metadata={},
            row_scale=None,
            inv_row_scale=None,
            col_enabled=False,
            col_built=False,
            col_metadata={},
            col_scale=None,
            inv_col_scale=None,
        )

    start_s = float(elapsed_s())
    metadata: dict[str, object] = {}
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "building assembled operator for Krylov matvec reuse",
        )
    try:
        try:
            preflight = build_xblock_assembled_operator_preflight_setup(
                op=op,
                xblock_active_idx_np=xblock_active_idx_np,
                sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                xblock_krylov_method=str(xblock_krylov_method),
                estimate_summary=estimate_summary,
                full_pattern=full_pattern,
                active_pattern=active_pattern,
                summarize_pattern=summarize_pattern,
                env=env,
            )
        except XBlockAssembledPreflightError as preflight_exc:
            metadata.update(preflight_exc.metadata)
            raise
        metadata.update(preflight.metadata)

        def matvec_np_no_count(x_np: np.ndarray) -> np.ndarray:
            return np.asarray(
                jax.device_get(
                    true_matvec_no_count(
                        jnp.asarray(np.asarray(x_np, dtype=np.float64), dtype=rhs_dtype)
                    )
                ),
                dtype=np.float64,
            ).reshape((-1,))

        bundle = build_operator_from_pattern(
            matvec_np_no_count,
            pattern=preflight.pattern,
            dtype=np.float64,
            backend=str(backend),
            csr_max_mb=float(preflight.csr_max_mb),
            drop_tol=float(preflight.drop_tol),
            allow_operator_only=False,
            max_colors=int(preflight.max_colors),
        )
        matrix = bundle.matrix
        if matrix is None:
            raise RuntimeError("assembled x-block operator materialization returned no matrix")

        validation_samples = _env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE",
            default=1,
            minimum=0,
        )
        validation_tol = max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE_RTOL",
                default=1.0e-8,
            ),
        )
        validation_errors: list[float] = []
        rng = np.random.default_rng(1729)
        for _ in range(int(validation_samples)):
            probe = rng.standard_normal(int(xblock_linear_size)).astype(np.float64)
            probe_norm = float(np.linalg.norm(probe))
            if np.isfinite(probe_norm) and probe_norm > 0.0:
                probe /= probe_norm
            ref = matvec_np_no_count(probe)
            got = np.asarray(bundle.matvec(probe), dtype=np.float64).reshape((-1,))
            denom = max(float(np.linalg.norm(ref)), 1.0e-300)
            validation_errors.append(float(np.linalg.norm(got - ref) / denom))
        max_validation_error = max(validation_errors, default=0.0)
        if max_validation_error > float(validation_tol):
            raise RuntimeError(
                "assembled x-block operator validation failed "
                f"max_rel_error={max_validation_error:.3e} > {float(validation_tol):.3e}"
            )

        equilibration = build_xblock_assembled_equilibration_setup(
            assembled_matrix=matrix,
            xblock_linear_size=int(xblock_linear_size),
            elapsed_s=elapsed_s,
            env=env,
        )
        if emit is not None:
            for level, message in equilibration.messages:
                emit(int(level), str(message))

        device = build_xblock_assembled_device_setup(
            assembled_matrix=matrix,
            assembled_matvec=bundle.matvec,
            csr_cap_nbytes=int(preflight.csr_cap_nbytes),
            device_enabled=bool(preflight.device_enabled),
            device_required=bool(preflight.device_required),
            validation_samples=int(validation_samples),
            validation_tol=float(validation_tol),
            device_csr_from_matrix=device_csr_from_matrix,
            validate_device_csr_matvec=validate_device_csr_matvec,
        )
        if emit is not None:
            for level, message in device.messages:
                emit(int(level), str(message))

        matvec_setup = build_xblock_assembled_matvec_setup(
            assembled_matvec=bundle.matvec,
            device_operator=device.device_operator,
            mv_count=mv_count,
            progress_every=int(progress_every),
            elapsed_s=elapsed_s,
            emit=emit,
        )
        metadata = finalize_metadata(
            metadata=metadata,
            setup_s=float(elapsed_s()) - start_s,
            assembled_matrix=matrix,
            assembled_summary=preflight.summary,
            assembled_bundle_metadata=bundle.metadata,
            max_colors=int(preflight.max_colors),
            validation_errors=validation_errors,
            device_enabled=bool(preflight.device_enabled),
            device_required=bool(preflight.device_required),
            device_resident=bool(device.device_resident),
            device_operator=device.device_operator,
            device_validation_errors=tuple(device.validation_errors),
            device_error=device.error,
        )
        if emit is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres assembled operator "
                f"built location={matvec_setup.location} nnz={metadata['matrix_nnz']} "
                f"setup_s={metadata['setup_s']:.3f}",
            )
        return XBlockAssembledOperatorBuildResult(
            matvec=matvec_setup.matvec,
            built=True,
            device_resident=bool(device.device_resident),
            metadata=metadata,
            device_operator=device.device_operator,
            pc_factor_increment_s=float(metadata["setup_s"]),
            row_enabled=bool(equilibration.row_enabled),
            row_built=bool(equilibration.row_built),
            row_metadata=dict(equilibration.row_metadata),
            row_scale=equilibration.row_scale,
            inv_row_scale=equilibration.inv_row_scale,
            col_enabled=bool(equilibration.col_enabled),
            col_built=bool(equilibration.col_built),
            col_metadata=dict(equilibration.col_metadata),
            col_scale=equilibration.col_scale,
            inv_col_scale=equilibration.inv_col_scale,
        )
    except Exception as exc:  # noqa: BLE001
        metadata = {
            **metadata,
            "error": f"{type(exc).__name__}: {exc}",
            "setup_s": float(elapsed_s()) - start_s,
        }
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"assembled operator disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockAssembledOperatorBuildResult(
            matvec=default_matvec,
            built=False,
            device_resident=False,
            metadata=metadata,
            device_operator=None,
            pc_factor_increment_s=0.0,
            row_enabled=False,
            row_built=False,
            row_metadata={},
            row_scale=None,
            inv_row_scale=None,
            col_enabled=False,
            col_built=False,
            col_metadata={},
            col_scale=None,
            inv_col_scale=None,
        )


def finalize_xblock_assembled_operator_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
    assembled_matrix: object,
    assembled_summary: object,
    assembled_bundle_metadata: object,
    max_colors: int,
    validation_errors: Sequence[float],
    device_enabled: bool,
    device_required: bool,
    device_resident: bool,
    device_operator: object | None,
    device_validation_errors: Sequence[float],
    device_error: str | None,
) -> dict[str, object]:
    """Return normalized metadata after assembled x-block operator construction."""

    if hasattr(assembled_matrix, "nnz"):
        matrix_nnz = int(assembled_matrix.nnz)
    else:
        matrix_nnz = int(np.count_nonzero(np.asarray(assembled_matrix)))
    return {
        **dict(metadata),
        "setup_s": float(setup_s),
        "pattern_nnz": int(assembled_summary.nnz),
        "pattern_avg_row_nnz": float(assembled_summary.avg_row_nnz),
        "pattern_max_row_nnz": int(assembled_summary.max_row_nnz),
        "storage_kind": assembled_bundle_metadata.storage_kind,
        "reason": assembled_bundle_metadata.reason,
        "matrix_nnz": int(matrix_nnz),
        "csr_nbytes_estimate": int(assembled_bundle_metadata.csr_nbytes_estimate),
        "max_colors": int(max_colors),
        "validation_rel_errors": tuple(float(v) for v in validation_errors),
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": bool(device_resident),
        "device_nnz": int(device_operator.nnz) if device_operator is not None else None,
        "device_csr_nbytes_estimate": (
            int(device_operator.nbytes_estimate) if device_operator is not None else None
        ),
        "device_validation_rel_errors": tuple(float(v) for v in device_validation_errors),
        "device_error": device_error,
    }


def resolve_xblock_moment_schur_policy_setup(
    *,
    op: object,
    xblock_krylov_method: str,
    xblock_jax_factors: bool,
    xblock_jax_factor_format: str,
    precondition_side: str,
    env: Mapping[str, str] | None = None,
) -> XBlockMomentSchurPolicySetup:
    """Resolve x-block moment-Schur default, force, and probe settings."""

    default_candidate = bool(
        str(xblock_krylov_method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
        and int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 1
        and int(op.extra_size) > 0
        and int(op.phi1_size) == 0
    )
    env_raw = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR").lower()
    default_blocked_by_compact_factors = bool(
        default_candidate
        and env_raw in {"", "auto", "default"}
        and bool(xblock_jax_factors)
        and str(xblock_jax_factor_format).strip().lower() == "csr"
    )
    default_enabled = bool(default_candidate and not default_blocked_by_compact_factors)
    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR",
        default=default_enabled,
    )
    rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND",
            default=1.0e-12,
        ),
    )
    probe_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE",
        default=False,
    )
    probe_min_improvement = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT",
            default=0.0,
        ),
    )
    messages: list[tuple[int, str]] = []
    if bool(default_blocked_by_compact_factors) and not bool(enabled):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "constraint1 moment-Schur default disabled for compact JAX factors "
                "(set SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR=1 to force)",
            )
        )
    if bool(enabled) and str(precondition_side) != "none":
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "constraint1 moment-Schur build start",
            )
        )
    return XBlockMomentSchurPolicySetup(
        default_candidate=bool(default_candidate),
        default_blocked_by_compact_factors=bool(default_blocked_by_compact_factors),
        enabled=bool(enabled),
        rcond=float(rcond),
        probe_enabled=bool(probe_enabled),
        probe_min_improvement=float(probe_min_improvement),
        messages=tuple(messages),
    )


def evaluate_xblock_moment_schur_probe_result(
    *,
    residual_before: float,
    residual_after: float,
    min_improvement: float,
) -> XBlockMomentSchurProbeResult:
    """Gate moment-Schur use from before/after residual norms."""

    before = float(residual_before)
    after = float(residual_after)
    if before > 0.0:
        ratio = float(after / before)
        required = before * max(0.0, 1.0 - float(min_improvement))
        used = bool(np.isfinite(after) and after < float(required))
    else:
        ratio = 0.0 if after == 0.0 else float("inf")
        used = bool(np.isfinite(after) and after <= 0.0)
    reason = "probe_reduced" if bool(used) else "probe_not_reduced"
    messages = (
        (
            0 if bool(used) else 1,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "constraint1 moment-Schur "
            f"{'accepted' if bool(used) else 'rejected'} "
            f"seed residual {before:.6e} -> {after:.6e} "
            f"(ratio={float(ratio):.6e})",
        ),
    )
    return XBlockMomentSchurProbeResult(
        used=bool(used),
        reason=str(reason),
        residual_before=float(before),
        residual_after=float(after),
        improvement_ratio=float(ratio),
        messages=messages,
    )


def finalize_xblock_moment_schur_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return moment-Schur metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_moment_schur_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized moment-Schur failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def resolve_xblock_two_level_policy_setup(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None = None,
) -> XBlockTwoLevelPolicySetup:
    """Resolve x-block two-level correction admission and build parameters."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL",
        default=False,
    )
    return XBlockTwoLevelPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        mode=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE") or "additive",
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS",
            default=48,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX",
            default=8,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_INCLUDE_RHS",
            default=True,
        ),
    )


def finalize_xblock_two_level_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return two-level metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_two_level_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized two-level failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def _xblock_device_krylov_method(method: str) -> bool:
    return str(method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}


def resolve_xblock_global_coupling_policy_setup(
    *,
    precondition_side: str,
    xblock_krylov_method: str,
    env: Mapping[str, str] | None = None,
) -> XBlockGlobalCouplingPolicySetup:
    """Resolve x-block global-coupling admission and build parameters."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING",
        default=False,
    )
    use_device_builder = _xblock_device_krylov_method(str(xblock_krylov_method))
    return XBlockGlobalCouplingPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        use_device_builder=bool(use_device_builder),
        mode=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE") or "additive",
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS",
            default=96,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX",
            default=12,
            minimum=0,
        ),
        angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX",
            default=2,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_INCLUDE_RHS",
            default=True,
        ),
        setup_max_s=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S",
                default=180.0 if bool(use_device_builder) else 0.0,
            ),
        ),
    )


def finalize_xblock_global_coupling_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return global-coupling metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_global_coupling_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized global-coupling failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def apply_xblock_moment_schur_stage(
    *,
    context: XBlockMomentSchurStageContext,
) -> XBlockMomentSchurStageResult:
    """Build and optionally probe the primary x-block moment-Schur stage."""

    if context.emit is not None:
        for level, message in context.policy.messages:
            context.emit(int(level), str(message))
    if (not bool(context.policy.enabled)) or str(context.precondition_side) == "none":
        return XBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=None,
            metadata={},
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        candidate, metadata, stats = context.builder(
            op=context.op,
            base_preconditioner=context.base_preconditioner,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            rcond=float(context.policy.rcond),
            emit=context.emit,
        )
        used = True
        reason: str | None = "built"
        probe_residual_before: float | None = None
        probe_residual_after: float | None = None
        probe_improvement_ratio: float | None = None
        if bool(context.policy.probe_enabled):
            seed_candidate = jnp.asarray(candidate(context.rhs), dtype=jnp.float64)
            seed_residual = context.rhs - jnp.asarray(
                context.matvec_no_count(seed_candidate),
                dtype=jnp.float64,
            )
            probe_result = evaluate_xblock_moment_schur_probe_result(
                residual_before=float(jnp.linalg.norm(context.rhs)),
                residual_after=float(jnp.linalg.norm(seed_residual)),
                min_improvement=float(context.policy.probe_min_improvement),
            )
            used = bool(probe_result.used)
            reason = str(probe_result.reason)
            probe_residual_before = float(probe_result.residual_before)
            probe_residual_after = float(probe_result.residual_after)
            probe_improvement_ratio = float(probe_result.improvement_ratio)
            if context.emit is not None:
                for level, message in probe_result.messages:
                    context.emit(int(level), str(message))
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockMomentSchurStageResult(
            preconditioner=candidate if bool(used) else context.base_preconditioner,
            built=True,
            used=bool(used),
            reason=reason,
            metadata=finalize_xblock_moment_schur_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            probe_residual_before=probe_residual_before,
            probe_residual_after=probe_residual_after,
            probe_improvement_ratio=probe_improvement_ratio,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"constraint1 moment-Schur disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=reason,
            metadata=failed_xblock_moment_schur_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=float(setup_s),
        )


def apply_xblock_two_level_stage(
    *,
    context: XBlockTwoLevelStageContext,
) -> XBlockTwoLevelStageResult:
    """Build the optional primary x-block two-level stage."""

    if not bool(context.policy.should_build):
        return XBlockTwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        if context.builder is None:
            from sfincs_jax.solvers.preconditioners.qi.two_level import (
                build_rhs1_xblock_two_level_preconditioner,
            )

            builder = build_rhs1_xblock_two_level_preconditioner
        else:
            builder = context.builder
        preconditioner, metadata, stats = builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=int(context.policy.fsavg_lmax),
            max_extra_units=int(context.policy.max_extra_units),
            max_directions=int(context.policy.max_directions),
            rcond=float(context.policy.rcond),
            include_rhs=bool(context.policy.include_rhs),
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockTwoLevelStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=finalize_xblock_two_level_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"two-level coarse disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockTwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata=failed_xblock_two_level_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )


def apply_xblock_global_coupling_stage(
    *,
    context: XBlockGlobalCouplingStageContext,
) -> XBlockGlobalCouplingStageResult:
    """Build the optional primary x-block global-coupling stage."""

    if not bool(context.policy.should_build):
        return XBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        if context.host_builder is None or context.device_builder is None:
            from sfincs_jax.solvers.preconditioners.qi.two_level import (
                build_rhs1_xblock_device_global_coupling_preconditioner,
                build_rhs1_xblock_smoothed_global_coupling_preconditioner,
            )

            host_builder = (
                build_rhs1_xblock_smoothed_global_coupling_preconditioner
                if context.host_builder is None
                else context.host_builder
            )
            device_builder = (
                build_rhs1_xblock_device_global_coupling_preconditioner
                if context.device_builder is None
                else context.device_builder
            )
        else:
            host_builder = context.host_builder
            device_builder = context.device_builder
        builder = device_builder if bool(context.policy.use_device_builder) else host_builder
        preconditioner, metadata, stats = builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=int(context.policy.fsavg_lmax),
            angular_lmax=int(context.policy.angular_lmax),
            max_extra_units=int(context.policy.max_extra_units),
            max_directions=int(context.policy.max_directions),
            rcond=float(context.policy.rcond),
            include_rhs=bool(context.policy.include_rhs),
            max_setup_s=float(context.policy.setup_max_s),
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockGlobalCouplingStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=finalize_xblock_global_coupling_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"global-coupling disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata=failed_xblock_global_coupling_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )






def resolve_xblock_seed_policy_setup(
    *,
    moment_schur_used: bool,
    env: Mapping[str, str] | None = None,
) -> XBlockSeedPolicySetup:
    """Resolve initial and moment-Schur x-block seed controls."""

    return XBlockSeedPolicySetup(
        initial_seed_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED",
            default=False,
        ),
        moment_schur_seed_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED",
            default=bool(moment_schur_used),
        ),
    )




































def finalize_sparse_pc_gmres_bundle(
    context: SparsePCGMRESFinalizationBundleContext,
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build typed sparse-PC final metadata, apply retry/polish, and return payload."""

    diagnostic_state = sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=context.atol,
            mv_count=context.mv_count,
            rhs_norm=context.rhs_norm,
            target=context.target,
            tol=context.tol,
            sparse_pc_direct_tail_metadata=sparse_pc_direct_tail_final_metadata(
                context.direct_tail
            ),
            sparse_pc_factor_preflight_metadata=(
                sparse_pc_factor_preflight_result_metadata_from_context(
                    context.factor_preflight
                )
            ),
            sparse_pc_pattern_metadata=sparse_pc_pattern_result_metadata_from_context(
                context.pattern
            ),
            sparse_pc_static_metadata=sparse_pc_gmres_static_metadata_from_context(
                context.static
            ),
        )
    )
    result = context.result
    return finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=diagnostic_state,
            result=SparsePCGMRESResult(
                x=np.asarray(result.x, dtype=np.float64),
                residual_norm=float(result.residual_norm),
                preconditioned_residual_norm=float(
                    result.preconditioned_residual_norm
                ),
                history=tuple(float(v) for v in (result.history or ())),
                solve_s=float(result.solve_s),
            ),
            factor_dtype_used=np.dtype(result.factor_dtype_used),
            factor_dtype_retry=result.factor_dtype_retry,
            operator_bundle=result.operator_bundle,
            factor_bundle=result.factor_bundle,
            pc_factor_s=float(result.pc_factor_s),
            setup_s=float(result.setup_s),
            post_minres=context.post_minres,
            dtype_retry=context.dtype_retry,
        ),
        build_host_sparse_direct_factor_from_matvec=(
            build_host_sparse_direct_factor_from_matvec
        ),
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
    )


















def xblock_sparse_pc_final_metadata_from_driver_state(
    state: Mapping[str, object],
    *,
    full_size: object,
) -> dict[str, object]:
    """Build final x-block sparse-PC metadata from one driver-state handoff."""
    return {
        **xblock_sparse_pc_result_diagnostics_from_driver_state(
            state,
            full_size=full_size,
        ),
        **build_rhs1_xblock_correction_metadata_from_driver_state(state),
    }


def xblock_sparse_pc_final_payload_from_driver_state(
    state: Mapping[str, object],
    *,
    expand_reduced: ArrayFn,
    post_corrections: object | None = None,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the x-block sparse-PC branch from driver state."""

    return xblock_sparse_pc_final_payload(
        XBlockSparsePCFinalPayloadContext(
            op=state["op"],
            x=np.asarray(state["x_np"], dtype=np.float64),
            residual_norm=float(state["residual_norm_xblock_pc"]),
            target=float(state["target_xblock"]),
            krylov_method=str(state["xblock_krylov_method"]),
            linear_size=(
                int(state["xblock_linear_size"])
                if "xblock_linear_size" in state
                else None
            ),
            restart=int(state["pc_restart"]) if "pc_restart" in state else None,
            diagnostic_state=state,
            post_corrections=post_corrections,
        ),
        expand_reduced=expand_reduced,
    )


def xblock_sparse_pc_final_payload(
    context: XBlockSparsePCFinalPayloadContext,
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the x-block sparse-PC branch."""

    residual_norm = float(context.residual_norm)
    metadata_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    metadata_state.update(
        {
            "op": context.op,
            "x_np": np.asarray(context.x, dtype=np.float64),
            "residual_norm_xblock_pc": residual_norm,
            "target_xblock": float(context.target),
            "xblock_krylov_method": str(context.krylov_method),
        }
    )
    if context.linear_size is not None:
        metadata_state["xblock_linear_size"] = int(context.linear_size)
    if context.restart is not None:
        metadata_state["pc_restart"] = int(context.restart)
    if context.post_corrections is not None:
        metadata_state.update(context.post_corrections.driver_state())
    if (
        "xblock_solver_kind" not in metadata_state
        and context.linear_size is not None
        and context.restart is not None
    ):
        work_estimates = xblock_sparse_pc_work_estimates(
            krylov_method=str(context.krylov_method),
            linear_size=int(context.linear_size),
            restart=int(context.restart),
            dtype=np.float64,
        )
        metadata_state.update(
            {
                "xblock_solver_kind": work_estimates.solver_kind,
                "xblock_device_krylov_methods": set(work_estimates.device_krylov_methods),
                "xblock_estimated_gmres_basis_nbytes": work_estimates.gmres_basis_nbytes,
                "xblock_estimated_bicgstab_work_nbytes": work_estimates.bicgstab_work_nbytes,
                "xblock_estimated_tfqmr_work_nbytes": work_estimates.tfqmr_work_nbytes,
            }
        )
    metadata_state["accepted_converged_xblock"] = profile_residual_converged(
        residual_norm,
        float(context.target),
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(context.x, dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=xblock_sparse_pc_final_metadata_from_driver_state(
            metadata_state,
            full_size=getattr(context.op, "total_size"),
        ),
    )






def _elapsed_since_now() -> Callable[[], float]:
    """Return a cheap elapsed-time callback for explicit host sparse branches."""

    start_s = perf_counter()
    return lambda: perf_counter() - start_s




































__all__ = [
    "FortranReducedSparsePCBackendSetup",
    "FortranReducedXBlockFactorPolicySetup",
    "FortranReducedXBlockFactorBuildContext",
    "FortranReducedXBlockFactorBuildResult",
    "FortranReducedXBlockFinalPayloadContext",
    "FortranReducedXBlockInitialSeedPolicySetup",
    "FortranReducedXBlockInitialSeedResult",
    "FortranReducedXBlockGlobalCouplingStageContext",
    "FortranReducedXBlockGlobalCouplingStageResult",
    "FortranReducedXBlockKrylovSolveContext",
    "FortranReducedXBlockKrylovPolicySetup",
    "FortranReducedXBlockKrylovSetupContext",
    "FortranReducedXBlockKrylovSetupResult",
    "FortranReducedXBlockMomentSchurStageContext",
    "FortranReducedXBlockMomentSchurStageResult",
    "DirectTailMaterializationContext",
    "DirectTailMaterializationResult",
    "DirectTailStructuredAdmissionContext",
    "DirectTailStructuredAdmissionResult",
    "DirectTailStructuredBuildContext",
    "DirectTailStructuredBuildResult",
    "DirectTailSupportModePreflightContext",
    "DirectTailSupportModePreflightResult",
    "FPXBlockGlobalCorrectionContext",
    "FPXBlockGlobalCorrectionResult",
    "FPXBlockHighXCorrectionContext",
    "FPXBlockHighXCorrectionResult",
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
    "SparsePCEntryPolicySetup",
    "DirectTailResidualRescuePolicy",
    "DirectTailTrueActiveRescuePolicy",
    "DirectTailCoupledCoarseRescuePolicy",
    "MatvecCounter",
    "XBlockInitialGuessSetup",
    "XBlockKrylovMatvecSetup",
    "XBlockAssembledOperatorDiagnosticsContext",
    "XBlockSparsePCCoreDiagnosticsContext",
    "XBlockSideProbeDiagnosticsContext",
    "SparsePCActiveDOFSetup",
    "SparsePCFactorPolicySetup",
    "SparsePCFactorDtypeRetryDecision",
    "SparsePCFactorDtypeRetryContext",
    "SparsePCFactorDtypeRetryResult",
    "SparsePCMemoryBudgetPreflightContext",
    "SparsePCPatternSetupContext",
    "SparsePCPatternSetupResult",
    "SparsePCGMRESContext",
    "SparsePCGMRESResult",
    "SparsePCGMRESFinalizationContext",
    "SparsePCGMRESFinalizationBundleContext",
    "SparsePCGMRESFinalResultContext",
    "SparsePCGMRESFinalizationStateContext",
    "SparsePCFactorDtypeRetryFinalizationContext",
    "SparsePCPostMinresFinalizationContext",
    "sparse_pc_gmres_finalization_driver_scope_keys",
    "sparse_pc_gmres_finalization_driver_state_keys",
    "sparse_pc_gmres_finalization_bundle_from_driver_scope",
    "sparse_pc_gmres_finalization_bundle_from_driver_result",
    "sparse_pc_gmres_finalization_state_from_context",
    "sparse_pc_gmres_finalization_state_from_driver_scope",
    "XBlockKrylovReport",
    "XBlockSparsePCCompletionContext",
    "XBlockSparsePCFinalPayloadContext",
    "XBlockGMRESFallbackDecision",
    "XBlockGMRESFallbackContext",
    "XBlockGMRESFallbackResult",
    "XBlockDeviceKrylovState",
    "XBlockFirstKrylovAttemptContext",
    "XBlockFirstKrylovAttemptResult",
    "XBlockFirstKrylovSolveStateContext",
    "XBlockKrylovSolveStageContext",
    "XBlockKrylovSolveStageResult",
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
    "XBlockAugmentedKrylovBasisContext",
    "XBlockAugmentedKrylovBasisResult",
    "XBlockAugmentedKrylovStageContext",
    "XBlockAugmentedKrylovStageResult",
    "XBlockKrylovSolveSpace",
    "XBlockKrylovSolveSpaceContext",
    "XBlockSparsePCFinalCoreState",
    "XBlockSparsePCFinalDeviceState",
    "XBlockSparsePCFinalPreflightState",
    "XBlockSparsePCFinalNestedMetadata",
    "XBlockSparsePCFinalMetadataStateContext",
    "XBlockSparsePCWorkEstimates",
    "XBlockSparsePCBranchSetup",
    "XBlockLocalPreconditionerBuildResult",
    "XBlockAssembledOperatorBuildResult",
    "XBlockMomentSchurStageContext",
    "XBlockMomentSchurStageResult",
    "XBlockTwoLevelStageContext",
    "XBlockTwoLevelStageResult",
    "XBlockGlobalCouplingStageContext",
    "XBlockGlobalCouplingStageResult",
    "XBlockQICoarseSeedStageContext",
    "XBlockQICoarseSeedStageResult",
    "XBlockQIDeviceAdmissionSetup",
    "XBlockQIDeviceBaseConfigSetup",
    "XBlockQIDeviceEnrichmentConfigSetup",
    "XBlockQIDeviceMetadataContext",
    "XBlockQIDeviceMultilevelConfigSetup",
    "XBlockQIDeviceOperatorReuseSetup",
    "XBlockQIDeviceStageContext",
    "XBlockQIDeviceStageResult",
    "XBlockQIDeviceSetupConfig",
    "XBlockQIDeviceSetupConfigContext",
    "XBlockQIDeflatedPolicySetup",
    "XBlockQIDeflatedStageContext",
    "XBlockQIDeflatedStageResult",
    "XBlockQIGalerkinPolicySetup",
    "XBlockQIGalerkinStageContext",
    "XBlockQIGalerkinStageResult",
    "XBlockQISeedPolicySetup",
    "XBlockQITwoLevelPolicySetup",
    "XBlockQITwoLevelStageContext",
    "XBlockQITwoLevelStageResult",
    "XBlockPhysicalResidual",
    "SparsePCGMRESFinalPayload",
    "SparseMinimumNormPolicy",
    "SparseMinimumNormPayload",
    "SparseHostDirectPayload",
    "SparseHostDirectFactorSolvePayload",
    "SparseHostDirectPolishPayload",
    "SparseHostDirectFallbackPayload",
    "ExplicitSparseMinimumNormBranchContext",
    "ExplicitSparseHostDirectBranchContext",
    "SparseHostOrILUFactorBuildContext",
    "SparseHostOrILUFactorBuildResult",
    "SparseHostOrILUFactorControls",
    "SparseILUPreconditionerBuildContext",
    "SparseILUPreconditionerBuildResult",
    "SparseHostScipyPreconditionerBuildContext",
    "SparseHostScipyPreconditionerBuildResult",
    "SparseHostScipyGMRESContext",
    "SparseHostRetryCandidateContext",
    "SparseHostRetryCandidateResult",
    "SparseJAXRetryPreconditionerBuildContext",
    "SparseSXBlockRescueContext",
    "SparseSXBlockRescueResult",
    "SparseXBlockExplicitSeedContext",
    "SparseXBlockExplicitSeedResult",
    "SparseXBlockRescueAcceptanceContext",
    "SparseXBlockRescueAcceptanceResult",
    "SparseXBlockRescueBuildContext",
    "SparseXBlockRescueBuildResult",
    "SparseXBlockRescueSolveContext",
    "SparseXBlockRescueSolveResult",
    "SparsePCDirectTailFinalMetadataContext",
    "ExplicitSparseOperatorBuildPolicy",
    "ExplicitSparseOperatorBuildResult",
    "SparsePCGMRESCompletionMessageContext",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "SparsePCPostMinresUpdateContext",
    "SparsePCPostMinresUpdateResult",
    "XBlockSubspaceCorrectionContext",
    "XBlockSubspaceCorrectionResult",
    "XBlockPostSolveCorrectionContext",
    "XBlockPostSolveCorrectionResult",
    "XBlockPostKrylovCompletionContext",
    "XBlockPostKrylovCompletionResult",
    "accept_sparse_xblock_rescue_candidate",
    "apply_fortran_reduced_xblock_global_coupling_stage",
    "apply_fortran_reduced_xblock_initial_seed",
    "apply_fortran_reduced_xblock_moment_schur_stage",
    "apply_xblock_global_coupling_stage",
    "apply_xblock_moment_schur_stage",
    "apply_xblock_qi_coarse_seed_stage",
    "apply_xblock_qi_deflated_stage",
    "apply_xblock_qi_device_stage",
    "apply_xblock_qi_galerkin_stage",
    "apply_xblock_qi_two_level_stage",
    "build_xblock_qi_stage_pipeline_context",
    "apply_xblock_side_probe_stage",
    "apply_xblock_probe_coarse_stage",
    "apply_xblock_augmented_krylov_stage",
    "apply_xblock_two_level_stage",
    "apply_sparse_pc_post_minres",
    "apply_sparse_pc_post_minres_if_needed",
    "apply_sparse_pc_post_minres_from_driver_state",
    "apply_sparse_xblock_explicit_seed",
    "apply_xblock_subspace_correction_if_needed",
    "build_fortran_reduced_xblock_factor_stage",
    "build_sparse_xblock_rescue_preconditioner",
    "build_fortran_reduced_xblock_krylov_setup",
    "build_xblock_local_preconditioner",
    "build_xblock_assembled_operator_if_requested",
    "build_xblock_qi_device_preconditioner_metadata",
    "build_xblock_qi_device_setup_config",
    "XBlockQIStagePipelineContext",
    "XBlockQIStagePipelineResult",
    "build_xblock_krylov_progress_callbacks",
    "build_xblock_krylov_matvec_setup",
    "resolve_xblock_qi_deflated_policy_setup",
    "resolve_xblock_qi_device_admission_setup",
    "resolve_xblock_qi_device_base_config_setup",
    "resolve_xblock_qi_device_enrichment_config_setup",
    "resolve_xblock_qi_device_multilevel_config_setup",
    "resolve_xblock_qi_device_operator_reuse_setup",
    "resolve_xblock_qi_galerkin_policy_setup",
    "resolve_xblock_qi_seed_policy_setup",
    "resolve_xblock_qi_two_level_policy_setup",
    "run_xblock_qi_preconditioner_pipeline",
    "resolve_xblock_krylov_control_setup",
    "build_sparse_pc_active_dof_setup",
    "build_sparse_pc_pattern_setup",
    "build_direct_tail_materialization_setup",
    "build_direct_tail_structured_preconditioner_setup",
    "enforce_sparse_pc_memory_budget",
    "emit_xblock_sparse_pc_completion",
    "emit_xblock_sparse_pc_completion_from_driver_state",
    "evaluate_sparse_pc_factor_preflight",
    "evaluate_sparse_pc_residual_candidate_acceptance",
    "evaluate_xblock_preflight_gate",
    "select_sparse_pc_auto_preflight_retry_candidates",
    "evaluate_sparse_pc_auto_preflight_retry",
    "resolve_sparse_pc_gmres_control_policy",
    "resolve_sparse_pc_factor_preflight_policy",
    "resolve_direct_tail_residual_rescue_policy",
    "resolve_direct_tail_true_active_rescue_policy",
    "resolve_direct_tail_coupled_coarse_rescue_policy",
    "run_direct_tail_support_mode_preflight",
    "sparse_pc_direct_tail_final_metadata",
    "resolve_direct_tail_structured_admission",
    "fp_xblock_global_correction_metadata",
    "fp_xblock_highx_residual_correction_metadata",
    "prepare_fortran_reduced_xblock_initial_guess",
    "prepare_xblock_initial_guess",
    "resolve_fortran_reduced_sparse_pc_backend",
    "resolve_fortran_reduced_xblock_factor_policy",
    "resolve_fortran_reduced_xblock_global_coupling_policy",
    "resolve_fortran_reduced_xblock_initial_seed_policy",
    "resolve_fortran_reduced_xblock_krylov_policy",
    "resolve_fortran_reduced_xblock_moment_schur_policy",
    "resolve_sparse_pc_entry_policy",
    "resolve_xblock_sparse_pc_branch_setup",
    "resolve_sparse_pc_factor_policy",
    "evaluate_sparse_pc_factor_dtype_retry",
    "sparse_pc_factor_dtype_retry_initial_guess",
    "retry_sparse_pc_factor_dtype_if_needed",
    "retry_sparse_pc_factor_dtype_from_driver_state",
    "retry_sparse_pc_factor_dtype_from_finalization_context",
    "run_fortran_reduced_xblock_krylov_solve",
    "run_fp_xblock_global_correction_stage",
    "run_fp_xblock_highx_residual_correction_stage",
    "run_sparse_sxblock_rescue_stage",
    "run_sparse_xblock_rescue_solve_stage",
    "run_sparse_pc_gmres_once",
    "run_sparse_pc_gmres_once_for_retry",
    "finalize_sparse_pc_gmres_bundle",
    "prepare_xblock_augmented_krylov_basis",
    "prepare_xblock_krylov_solve_space",
    "run_xblock_first_krylov_attempt",
    "run_xblock_krylov_solve_stage",
    "run_xblock_gmres_fallback_if_needed",
    "run_xblock_post_solve_corrections",
    "complete_xblock_post_krylov_stage",
    "xblock_device_krylov_state",
    "xblock_device_cycle_progress_message",
    "xblock_host_krylov_progress_message",
    "xblock_sparse_pc_final_metadata_driver_scope_keys",
    "xblock_sparse_pc_final_metadata_driver_state_keys",
    "xblock_sparse_pc_final_metadata_state_from_context",
    "xblock_sparse_pc_final_metadata_state_from_driver_scope",
    "xblock_krylov_state_from_first_attempt",
    "xblock_krylov_state_from_gmres_fallback",
    "xblock_sparse_pc_completion_message",
    "xblock_gmres_fallback_decision",
    "xblock_krylov_report",
    "xblock_physical_solution_and_residual",
    "xblock_sparse_pc_work_estimates",
    "sparse_pc_gmres_completion_message",
    "emit_sparse_pc_gmres_completion_from_driver_state",
    "sparse_pc_gmres_final_payload_from_driver_state",
    "finalize_sparse_pc_gmres_from_driver_state",
    "finalize_sparse_pc_gmres_with_dtype_retry",
    "finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state",
    "fortran_reduced_xblock_final_payload",
    "fortran_reduced_xblock_final_payload_from_driver_state",
    "xblock_sparse_pc_final_payload",
    "xblock_sparse_pc_final_payload_from_driver_state",
    "resolve_sparse_minimum_norm_policy",
    "sparse_minimum_norm_solve_payload",
    "sparse_minimum_norm_solve_from_pattern",
    "sparse_minimum_norm_start_message",
    "solve_explicit_sparse_minimum_norm_branch",
    "sparse_host_direct_solve_payload",
    "sparse_host_direct_solve_from_pattern",
    "solve_explicit_sparse_host_direct_branch",
    "solve_sparse_host_direct_from_available_factor",
    "apply_sparse_host_direct_polish_if_needed",
    "sparse_host_direct_fallback_payload",
    "build_sparse_host_or_ilu_factor",
    "resolve_sparse_host_or_ilu_factor_controls",
    "build_sparse_ilu_preconditioner_from_cache",
    "build_sparse_host_scipy_preconditioner",
    "run_sparse_host_scipy_gmres",
    "run_sparse_host_retry_candidate",
    "build_sparse_jax_retry_preconditioner",
    "build_explicit_sparse_operator_from_pattern",
    "explicit_sparse_pattern_progress_messages",
    "resolve_explicit_sparse_operator_build_policy",
    "validate_explicit_sparse_host_request",
    "sparse_rescue_tail_metadata",
    "sparse_xblock_rescue_metadata",
    "xblock_assembled_operator_diagnostics",
    "xblock_coarse_correction_diagnostics",
    "xblock_device_krylov_diagnostics",
    "xblock_qi_deflated_preconditioner_diagnostics",
    "xblock_qi_device_preconditioner_diagnostics",
    "xblock_qi_seed_preconditioner_diagnostics",
    "xblock_sparse_pc_core_diagnostics",
    "xblock_sparse_pc_result_diagnostics_from_driver_state",
    "xblock_sparse_pc_final_metadata_from_driver_state",
    "xblock_side_probe_diagnostics",
]
