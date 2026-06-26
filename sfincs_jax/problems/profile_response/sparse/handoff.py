"""Sparse-PC handoff helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..diagnostics import (
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
from .direct import (
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
from .finalization import (
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
from .fortran_reduced import (
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
from .policy import (
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
from .qi import (
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
from .xblock import (
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
    xblock_sparse_pc_final_metadata_from_driver_state,
    xblock_sparse_pc_final_payload_from_driver_state,
    xblock_sparse_pc_final_payload,
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

# Consolidated sparse-PC Krylov execution helpers

ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class SparsePCGMRESContext:
    """Solve-local dependencies for one sparse-PC GMRES attempt."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    restart: int
    tol: float
    atol: float
    precondition_side: str
    factor_dtype: np.dtype
    progress_every: int
    stagnation_abort: bool
    stagnation_min_iter: int
    stagnation_window: int
    stagnation_rel_improvement: float
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]

def run_sparse_pc_gmres_once(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> SparsePCGMRESResult:
    """Run one host sparse-PC GMRES attempt and recompute the true residual."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres solve start "
            f"form={context.pc_form} restart={int(context.restart)} maxiter={int(maxiter)} "
            f"precondition_side={context.precondition_side} "
            f"factor_dtype={np.dtype(context.factor_dtype).name}",
        )

    solve_start_s = float(context.elapsed_s())
    stagnation_best = float("inf")
    stagnation_best_iter = 0

    def _progress_callback(iteration: int, residual_norm: float) -> None:
        nonlocal stagnation_best, stagnation_best_iter
        iteration_i = int(iteration)
        residual_f = float(residual_norm)
        if np.isfinite(residual_f) and (
            not np.isfinite(stagnation_best)
            or residual_f < stagnation_best * (1.0 - float(context.stagnation_rel_improvement))
        ):
            stagnation_best = float(residual_f)
            stagnation_best_iter = int(iteration_i)
        if (
            bool(context.stagnation_abort)
            and iteration_i >= int(context.stagnation_min_iter)
            and iteration_i - int(stagnation_best_iter) >= int(context.stagnation_window)
        ):
            raise RuntimeError(
                "sparse_pc_gmres stagnation detected: "
                f"iters={iteration_i} best_iter={int(stagnation_best_iter)} "
                f"best_ksp_residual={float(stagnation_best):.6e} "
                f"current_ksp_residual={residual_f:.6e} "
                f"window={int(context.stagnation_window)} "
                f"rel_improvement={float(context.stagnation_rel_improvement):.3e}"
            )
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if iteration_i % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres "
            f"iters={iteration_i} ksp_residual={residual_f:.6e} "
            f"elapsed_s={float(context.elapsed_s()):.3f}",
        )

    preconditioned_residual_norm = float("nan")
    if context.pc_form in {"explicit_left", "petsc_left"}:
        x_np, residual_norm, preconditioned_residual_norm, history = context.explicit_left_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            progress_callback=_progress_callback,
        )
    else:
        x_np, residual_norm, history = context.gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            precondition_side=context.precondition_side,
            progress_callback=_progress_callback,
        )

    solve_s = float(context.elapsed_s()) - solve_start_s
    try:
        residual_true = np.asarray(context.rhs, dtype=np.float64) - np.asarray(
            jax.device_get(context.matvec(jnp.asarray(x_np, dtype=jnp.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(residual_norm)

    return SparsePCGMRESResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(preconditioned_residual_norm),
        history=tuple(float(v) for v in (history or ())),
        solve_s=float(solve_s),
    )

def run_sparse_pc_gmres_once_for_retry(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> tuple[np.ndarray, float, float, tuple[float, ...], float]:
    """Run sparse-PC GMRES and return the tuple contract used by dtype retry."""

    result = run_sparse_pc_gmres_once(
        context=context,
        x0=x0,
        maxiter=int(maxiter),
    )
    return (
        result.x,
        float(result.residual_norm),
        float(result.preconditioned_residual_norm),
        tuple(float(value) for value in result.history),
        float(result.solve_s),
    )


@dataclass(frozen=True)
class RHS1FullSparseRetryStageContext:
    """Inputs for the full-space sparse retry stage after primary RHSMode-1 solve."""

    op: Any
    result: Any
    residual_vec: Any
    rhs: jnp.ndarray
    matvec: ArrayFn
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    sparse_kind_use: str
    sparse_exact_lu: bool
    sparse_drop_tol: float
    sparse_drop_rel: float
    sparse_ilu_drop_tol: float
    sparse_ilu_fill: float
    sparse_ilu_dense_max: int
    sparse_dense_cache_max: int
    sparse_use_matvec: bool
    sparse_jax_reg: float
    sparse_jax_omega: float
    sparse_jax_sweeps: int
    use_implicit: bool
    use_pas_projection: bool
    active_size: int
    large_cpu_sparse_rescue: bool
    rhs1_polish_enabled: bool
    emit: EmitFn | None
    mark: Callable[[str], None]
    cache_key_builder: Callable[..., object]
    precond_dtype: Callable[[int], Any]
    build_sparse_jax_preconditioner_from_matvec: Callable[..., ArrayFn]
    host_sparse_direct_allowed: Callable[..., bool]
    sparse_operator_preconditioned_rescue_allowed: Callable[..., bool]
    build_point_preconditioner_operator: Callable[[Any], Any]
    apply_cached_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    host_sparse_factor_dtype: Callable[[], Any]
    sparse_factor_cache_key: Callable[..., object]
    explicit_sparse_host_direct_allowed: Callable[..., bool]
    maybe_full_sparse_pattern: Callable[..., Any]
    build_host_sparse_direct_factor_from_matvec: Callable[..., Any]
    build_sparse_ilu_from_matvec: Callable[..., Any]
    host_sparse_direct_refine_steps: Callable[..., int]
    direct_solve_with_refinement: Callable[..., Any]
    ilu_solve_with_refinement: Callable[..., Any]
    host_sparse_direct_polish: Callable[..., Any]
    parse_polish_gmres_config: Callable[..., tuple[int, int]]
    gmres_solver: Callable[..., Any]
    solve_linear_with_residual: Callable[..., Any]
    run_measured_linear_candidate: Callable[..., tuple[Any, Any, bool, float]]
    accept_sparse_retry_candidate: Callable[..., tuple[Any, Any, bool]]
    replay_state: Any
    solver_kind: str
    peak_rss_mb: Callable[[], float | None]
    sparse_ilu_cache: Mapping[object, Any]
    problem_size: int | None = None
    cache_active_size: int | None = None
    scope: str = "full"
    use_active_dof_mode: bool = False
    force_host_sparse_direct: bool = False
    enable_operator_preconditioned_rescue: bool = True
    require_lower_diag: bool = True
    measured_returns_residual_vec: bool = True
    implicit_solver_returns_residual_vec: bool = True
    accept_candidate_residual_vec: bool = True
    compute_scipy_residual_vec: bool = True


@dataclass(frozen=True)
class RHS1FullSparseRetryStageResult:
    """Updated solve state after attempting the full-space sparse retry."""

    result: Any
    residual_vec: Any
    dense_matrix_cache: np.ndarray | None
    host_sparse_direct_used: bool


def run_rhs1_full_sparse_retry_stage(
    context: RHS1FullSparseRetryStageContext,
) -> RHS1FullSparseRetryStageResult:
    """Run the full-space sparse-JAX or host sparse retry and update replay.

    This stage is intentionally driver-independent except for callbacks that
    own cache keys, monkeypatchable builders, and replay acceptance. Keeping the
    branch here makes the public solve entry point a phase sequencer instead of
    a sparse-factor implementation.
    """

    result = context.result
    residual_vec = context.residual_vec
    dense_matrix_cache: np.ndarray | None = None
    host_sparse_direct_used = False
    problem_size = (
        int(context.problem_size)
        if context.problem_size is not None
        else int(context.op.total_size)
    )
    cache_active_size = (
        int(context.cache_active_size)
        if context.cache_active_size is not None
        else int(problem_size)
    )
    scope = str(context.scope)

    if float(result.residual_norm) <= float(context.target):
        return RHS1FullSparseRetryStageResult(
            result=result,
            residual_vec=residual_vec,
            dense_matrix_cache=dense_matrix_cache,
            host_sparse_direct_used=host_sparse_direct_used,
        )

    if str(context.sparse_kind_use) == "jax":
        try:
            context.mark("rhs1_sparse_precond_build_start")
            cache_key = context.cache_key_builder(
                context.op,
                kind="sparse_jax",
                active_size=int(cache_active_size),
                use_active_dof_mode=bool(context.use_active_dof_mode),
                use_pas_projection=bool(context.use_pas_projection),
                drop_tol=float(context.sparse_drop_tol),
                drop_rel=float(context.sparse_drop_rel),
                ilu_drop_tol=float(context.sparse_ilu_drop_tol),
                fill_factor=float(context.sparse_ilu_fill),
            )
            precond_sparse = build_sparse_jax_retry_preconditioner(
                SparseJAXRetryPreconditionerBuildContext(
                    matvec=context.matvec,
                    n=int(problem_size),
                    dtype=context.precond_dtype(int(problem_size)),
                    cache_key=cache_key,
                    drop_tol=float(context.sparse_drop_tol),
                    drop_rel=float(context.sparse_drop_rel),
                    reg=float(context.sparse_jax_reg),
                    omega=float(context.sparse_jax_omega),
                    sweeps=int(context.sparse_jax_sweeps),
                    emit=context.emit,
                    builder=context.build_sparse_jax_preconditioner_from_matvec,
                )
            )
            context.mark("rhs1_sparse_precond_build_done")
            result, residual_vec, _accepted, _elapsed = (
                context.run_measured_linear_candidate(
                    replay_state=context.replay_state,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    matvec_fn=context.matvec,
                    b_vec=context.rhs,
                    precond_fn=precond_sparse,
                    tol=float(context.tol),
                    atol=float(context.atol),
                    restart=int(context.restart),
                    maxiter=context.maxiter,
                    solve_method="incremental",
                    precond_side=context.precondition_side,
                    solve_linear=context.solve_linear_with_residual,
                    solver_kind=context.solver_kind,
                    candidate_name=f"sparse_jax_{scope}",
                    baseline_name=f"current_{scope}",
                    target_value=float(context.target),
                    peak_rss_mb=context.peak_rss_mb(),
                    returns_residual_vec=bool(context.measured_returns_residual_vec),
                )
            )
        except Exception as exc:  # noqa: BLE001
            if context.emit is not None:
                context.emit(1, f"sparse_jax: failed ({type(exc).__name__}: {exc})")
        return RHS1FullSparseRetryStageResult(
            result=result,
            residual_vec=residual_vec,
            dense_matrix_cache=dense_matrix_cache,
            host_sparse_direct_used=host_sparse_direct_used,
        )

    try:
        context.mark("rhs1_sparse_precond_build_start")
        cache_key = context.cache_key_builder(
            context.op,
            kind="sparse_lu" if bool(context.sparse_exact_lu) else "sparse_ilu",
            active_size=int(cache_active_size),
            use_active_dof_mode=bool(context.use_active_dof_mode),
            use_pas_projection=bool(context.use_pas_projection),
            drop_tol=float(context.sparse_drop_tol),
            drop_rel=float(context.sparse_drop_rel),
            ilu_drop_tol=float(context.sparse_ilu_drop_tol),
            fill_factor=float(context.sparse_ilu_fill),
        )
        host_sparse_direct_wanted = context.host_sparse_direct_allowed(
            sparse_exact_lu=bool(context.sparse_exact_lu),
            use_implicit=bool(context.use_implicit),
        )
        if bool(context.large_cpu_sparse_rescue) and bool(context.sparse_exact_lu):
            host_sparse_direct_wanted = True
        sparse_operator_pc = bool(context.enable_operator_preconditioned_rescue) and (
            context.sparse_operator_preconditioned_rescue_allowed(
                op=context.op,
                sparse_exact_lu=bool(context.sparse_exact_lu),
                host_sparse_direct_wanted=bool(host_sparse_direct_wanted),
            )
        )
        sparse_factor_matvec = context.matvec
        if bool(sparse_operator_pc):
            op_sparse_pc = context.build_point_preconditioner_operator(context.op)

            def sparse_factor_matvec(v: jnp.ndarray, op_pc=op_sparse_pc) -> jnp.ndarray:
                return context.apply_cached_operator(op_pc, v)

        cache_key_for_factor = cache_key
        if bool(sparse_operator_pc):
            cache_key_for_factor = context.cache_key_builder(
                context.op,
                kind="sparse_lu_pc_point",
                active_size=int(context.active_size),
                use_active_dof_mode=bool(context.use_active_dof_mode),
                use_pas_projection=bool(context.use_pas_projection),
                drop_tol=float(context.sparse_drop_tol),
                drop_rel=float(context.sparse_drop_rel),
                ilu_drop_tol=float(context.sparse_ilu_drop_tol),
                fill_factor=float(context.sparse_ilu_fill),
            )
        sparse_factor_controls = resolve_sparse_host_or_ilu_factor_controls(
            n=int(problem_size),
            cache_key=cache_key_for_factor,
            sparse_exact_lu=bool(context.sparse_exact_lu),
            use_implicit=bool(context.use_implicit),
            force_host_sparse_direct=bool(context.force_host_sparse_direct),
            sparse_ilu_dense_max=int(context.sparse_ilu_dense_max),
            sparse_dense_cache_max=int(context.sparse_dense_cache_max),
            host_sparse_direct_wanted=bool(host_sparse_direct_wanted),
            host_sparse_direct_allowed=context.host_sparse_direct_allowed,
            host_sparse_factor_dtype=context.host_sparse_factor_dtype,
            sparse_factor_cache_key=context.sparse_factor_cache_key,
            explicit_sparse_host_direct_allowed=context.explicit_sparse_host_direct_allowed,
        )
        factor_dtype = sparse_factor_controls.factor_dtype
        explicit_sparse_pattern = (
            context.maybe_full_sparse_pattern(context.op, emit=context.emit)
            if sparse_factor_controls.explicit_sparse_allowed
            else None
        )
        sparse_factor_build = build_sparse_host_or_ilu_factor(
            SparseHostOrILUFactorBuildContext(
                matvec=sparse_factor_matvec,
                n=int(problem_size),
                dtype=context.rhs.dtype,
                cache_key=sparse_factor_controls.cache_key_use,
                factor_dtype=sparse_factor_controls.factor_dtype,
                drop_tol=float(context.sparse_drop_tol),
                drop_rel=float(context.sparse_drop_rel),
                ilu_drop_tol=float(context.sparse_ilu_drop_tol),
                fill_factor=float(context.sparse_ilu_fill),
                build_dense_factors=sparse_factor_controls.build_dense_factors,
                build_jax_factors=sparse_factor_controls.build_jax_factors,
                store_dense=sparse_factor_controls.store_dense,
                factorization="lu" if bool(context.sparse_exact_lu) else "ilu",
                emit=context.emit,
                host_sparse_direct_wanted=sparse_factor_controls.host_sparse_direct_wanted,
                explicit_sparse_allowed=sparse_factor_controls.explicit_sparse_allowed,
                explicit_sparse_pattern=explicit_sparse_pattern,
                build_host_sparse_direct_factor_from_matvec=(
                    context.build_host_sparse_direct_factor_from_matvec
                ),
                build_sparse_ilu_from_matvec=context.build_sparse_ilu_from_matvec,
            )
        )
        dense_matrix_cache = sparse_factor_build.a_dense_cache
        context.mark("rhs1_sparse_precond_build_done")

        sparse_pc_restart = None
        sparse_pc_maxiter = None
        if bool(sparse_operator_pc):
            sparse_pc_restart, sparse_pc_maxiter = context.parse_polish_gmres_config(
                restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART",
                maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER",
                default_restart=max(120, int(context.restart)),
                default_maxiter=max(800, int(context.maxiter or 400) * 2),
            )

        def run_full_implicit_sparse(precond_sparse):
            implicit_result = context.solve_linear_with_residual(
                matvec_fn=context.matvec,
                b_vec=context.rhs,
                precond_fn=precond_sparse,
                x0_vec=result.x,
                tol_val=float(context.tol),
                atol_val=float(context.atol),
                restart_val=int(context.restart),
                maxiter_val=context.maxiter,
                solve_method_val="incremental",
                precond_side=context.precondition_side,
            )
            if bool(context.implicit_solver_returns_residual_vec):
                return implicit_result
            return implicit_result, None

        sparse_retry_candidate = run_sparse_host_retry_candidate(
            SparseHostRetryCandidateContext(
                factor_build=sparse_factor_build,
                host_sparse_direct=bool(host_sparse_direct_wanted),
                host_direct_operator_pc=bool(sparse_operator_pc),
                use_implicit=bool(context.use_implicit),
                matvec=context.matvec,
                rhs=context.rhs,
                x0=result.x,
                factor_dtype=factor_dtype,
                refine_steps=context.host_sparse_direct_refine_steps(
                    "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_REFINE",
                    default=2,
                ),
                target=float(context.target),
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                precondition_side=context.precondition_side,
                emit=context.emit,
                backend_name=jax.default_backend(),
                sparse_use_matvec=bool(context.sparse_use_matvec),
                sparse_exact_lu=bool(context.sparse_exact_lu),
                cache_entry=context.sparse_ilu_cache.get(cache_key),
                require_lower_diag=bool(context.require_lower_diag),
                polish_enabled=bool(context.rhs1_polish_enabled),
                parse_polish_gmres_config=context.parse_polish_gmres_config,
                direct_solve_with_refinement=context.direct_solve_with_refinement,
                ilu_solve_with_refinement=context.ilu_solve_with_refinement,
                host_sparse_direct_polish=context.host_sparse_direct_polish,
                gmres_solver=context.gmres_solver,
                implicit_solver=run_full_implicit_sparse,
                operator_pc_restart=sparse_pc_restart,
                operator_pc_maxiter=sparse_pc_maxiter,
                compute_scipy_residual_vec=bool(context.compute_scipy_residual_vec),
            )
        )
        if sparse_retry_candidate.result is not None:
            host_sparse_direct_used = (
                host_sparse_direct_used
                or sparse_retry_candidate.host_sparse_direct_used
            )
            result, residual_vec, _accepted = context.accept_sparse_retry_candidate(
                replay_state=context.replay_state,
                current_result=result,
                candidate_result=sparse_retry_candidate.result,
                current_residual_vec=residual_vec,
                candidate_residual_vec=(
                    sparse_retry_candidate.residual_vec
                    if bool(context.accept_candidate_residual_vec)
                    else None
                ),
                matvec_fn=sparse_retry_candidate.matvec,
                b_vec=context.rhs,
                precond_fn=sparse_retry_candidate.preconditioner,
                restart=int(context.restart),
                maxiter=context.maxiter,
                precond_side=context.precondition_side,
                solver_kind=context.solver_kind,
                candidate_family="sparse",
                scope=scope,
                target_value=float(context.target),
                solve_s=sparse_retry_candidate.solve_s,
                peak_rss_mb=context.peak_rss_mb(),
            )
    except Exception as exc:  # noqa: BLE001
        if context.emit is not None:
            context.emit(
                1,
                f"{'sparse_lu' if bool(context.sparse_exact_lu) else 'sparse_ilu'}: "
                f"failed ({type(exc).__name__}: {exc})",
            )

    return RHS1FullSparseRetryStageResult(
        result=result,
        residual_vec=residual_vec,
        dense_matrix_cache=dense_matrix_cache,
        host_sparse_direct_used=host_sparse_direct_used,
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
class SparsePCGenericBranchSetupContext:
    """Dependencies for the generic RHSMode-1 sparse-PC setup stage."""

    op: object
    rhs: jnp.ndarray
    sparse_pc_use_active_dof: bool
    active_dof_indices: Callable[[object], np.ndarray]
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    fortran_reduced_sparse_pc: bool
    preconditioner_x: int
    preconditioner_x_min_l: int
    preconditioner_xi: int
    preconditioner_species: int
    sparse_pc_fp_dense_velocity_block: object
    constrained_pas_pc: bool
    tokamak_pas_er_pc: bool
    tokamak_fp_pc: bool
    pc_maxiter: int
    pc_restart: int
    host_sparse_factor_dtype: object
    sparse_timer: object
    emit: EmitFn | None
    env: Mapping[str, str] | None
    default_permc_spec: Callable[..., str]
    build_fortran_reduced_operator: Callable[..., object]
    build_point_operator: Callable[..., object]
    fortran_reduced_pattern_for_indices: Callable[..., object]
    fortran_reduced_pattern: Callable[..., object]
    conservative_pattern_for_indices: Callable[..., object]
    conservative_pattern: Callable[..., object]
    summarize_pattern: Callable[..., object]
    estimate_sparse_pc_memory: Callable[..., object]
    device_count: int


@dataclass(frozen=True)
class SparsePCGenericBranchSetupResult:
    """Resolved generic sparse-PC setup state used by later factor stages."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    rhs: jnp.ndarray
    linear_size: int
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    op_pc: object
    pattern_source_op: object
    preconditioner_operator: str
    fortran_reduced_xblock_min_size: int
    fortran_reduced_sparse_pc_backend: str
    fortran_reduced_sparse_pc_backend_reason: str
    pattern: object
    sparse_pattern_scope: str
    pattern_build_s: float
    summary: object
    factor_policy: SparsePCFactorPolicySetup


@dataclass(frozen=True)
class SparsePCDirectTailFactorSetupContext:
    """Inputs for direct-tail structured-PC admission and factor setup."""

    env: Mapping[str, str] | None
    op: object
    op_pc: object
    rhs_dtype: object
    pattern: object
    active_indices: np.ndarray | None
    sparse_pc_use_active_dof: bool
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    factor_matvec: Callable[[np.ndarray], jnp.ndarray]
    pc_shift: float
    factor_dtype_initial: object
    factorization: str
    default_factor_kind: str
    default_ilu_fill_factor: float
    default_ilu_drop_tol: float
    default_pattern_color_batch: int
    default_permc_spec: str
    permc_spec: str
    sparse_pc_linear_size: int
    constrained_pas_pc: bool
    tokamak_fp_pc: bool
    fortran_reduced_sparse_pc: bool
    preconditioner_x: int
    preconditioner_x_min_l: int
    preconditioner_xi: int
    preconditioner_species: int
    sparse_timer: object
    emit: EmitFn | None
    default_direct_tail_max_mb: Callable[..., float]
    is_direct_reduced_pmat_pc_kind: Callable[[str], bool]
    build_direct_tail_bundle: Callable[..., object]
    build_structured_full_csr_operator_bundle: Callable[..., object]
    layout_from_operator: Callable[[object], object]
    build_direct_active_preconditioner: Callable[..., object]
    build_active_projected_preconditioner: Callable[..., object]
    structured_cache: object
    structured_cache_key: Callable[..., tuple[object, ...]]
    structured_cache_metadata: Callable[..., object]
    structured_factor_bundle_factory: Callable[..., object]
    host_factor_builder: Callable[..., tuple[object, object]]


@dataclass(frozen=True)
class SparsePCDirectTailFactorSetupResult:
    """Resolved direct-tail factor setup state consumed by later sparse stages."""

    materialization: DirectTailMaterializationResult
    direct_tail_default: bool
    direct_tail_enabled: bool
    direct_tail_built: bool
    direct_tail_error: str | None
    direct_tail_operator_bundle: object | None
    direct_tail_structured_pc_requested: str | None
    direct_tail_structured_pc_selected: bool
    direct_tail_structured_pc_reason: str | None
    direct_tail_structured_pc_metadata: dict[str, object] | None
    direct_tail_structured_pc_error: str | None
    direct_tail_pc_env_early: str
    direct_tail_direct_reduced_pmat_requested: bool
    structured_admission: DirectTailStructuredAdmissionResult
    direct_tail_pc_env: str
    direct_tail_pc_auto_default: bool
    direct_tail_fail_closed_size: int
    direct_tail_auto_large_fail_closed: bool
    direct_tail_structured_pc_required: bool
    structured_pc_ready: bool
    direct_tail_structured_layout: object | None
    direct_tail_structured_active_indices: np.ndarray | None
    direct_tail_structured_max_nbytes: int | None
    direct_tail_support_mode_preflight_requested: bool
    direct_tail_support_mode_preflight_selected: bool
    direct_tail_support_mode_preflight_metadata: dict[str, object] | None
    direct_tail_support_mode_preflight_error: str | None
    direct_tail_structured_pc_max_mb_auto: bool
    pc_max_mb: float
    pc_reg: float
    operator_bundle_pc: object | None
    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float


def build_sparse_pc_generic_branch_setup(
    context: SparsePCGenericBranchSetupContext,
) -> SparsePCGenericBranchSetupResult:
    """Resolve active maps, operator policy, pattern, factor policy, and budget."""

    active_setup = build_sparse_pc_active_dof_setup(
        op=context.op,
        rhs=context.rhs,
        sparse_pc_use_active_dof=bool(context.sparse_pc_use_active_dof),
        active_dof_indices=context.active_dof_indices,
        reduce_full_with_indices=context.reduce_full_with_indices,
        expand_reduced_with_map=context.expand_reduced_with_map,
    )
    if context.emit is not None:
        for level, message in active_setup.messages:
            context.emit(level, message)

    if context.fortran_reduced_sparse_pc:
        op_pc = context.build_fortran_reduced_operator(
            context.op,
            preconditioner_x=int(context.preconditioner_x),
            preconditioner_xi=int(context.preconditioner_xi),
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_x_min_l=int(context.preconditioner_x_min_l),
        )
        preconditioner_operator = "fortran_reduced_global"
        pattern_source_op = op_pc
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
                "using global angular-coupled RHSMode=1 preconditioner operator "
                f"(preconditioner_x={int(context.preconditioner_x)} "
                f"preconditioner_x_min_L={int(context.preconditioner_x_min_l)} "
                f"preconditioner_xi={int(context.preconditioner_xi)} "
                f"preconditioner_species={int(context.preconditioner_species)})",
            )
    else:
        op_pc = context.build_point_operator(context.op)
        preconditioner_operator = "point"
        pattern_source_op = context.op

    backend_setup = resolve_fortran_reduced_sparse_pc_backend(
        op=context.op,
        env=context.env,
        fortran_reduced_sparse_pc=bool(context.fortran_reduced_sparse_pc),
        sparse_pc_linear_size=int(active_setup.linear_size),
    )
    if context.emit is not None:
        for level, message in backend_setup.messages:
            context.emit(level, message)

    pattern_setup = build_sparse_pc_pattern_setup(
        SparsePCPatternSetupContext(
            op=context.op,
            pattern_source_op=pattern_source_op,
            fortran_reduced_sparse_pc=bool(context.fortran_reduced_sparse_pc),
            sparse_pc_use_active_dof=bool(context.sparse_pc_use_active_dof),
            active_idx_np=active_setup.active_idx_np,
            preconditioner_x=int(context.preconditioner_x),
            preconditioner_xi=int(context.preconditioner_xi),
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_x_min_l=int(context.preconditioner_x_min_l),
            fp_dense_velocity_block=context.sparse_pc_fp_dense_velocity_block,
            elapsed_s=context.sparse_timer.elapsed_s,
            emit=context.emit,
            fortran_reduced_pattern_for_indices=(
                context.fortran_reduced_pattern_for_indices
            ),
            fortran_reduced_pattern=context.fortran_reduced_pattern,
            conservative_pattern_for_indices=context.conservative_pattern_for_indices,
            conservative_pattern=context.conservative_pattern,
            summarize_pattern=context.summarize_pattern,
        )
    )
    factor_policy = resolve_sparse_pc_factor_policy(
        env=context.env,
        constrained_pas_pc=bool(context.constrained_pas_pc),
        tokamak_fp_pc=bool(context.tokamak_fp_pc),
        fortran_reduced_sparse_pc=bool(context.fortran_reduced_sparse_pc),
        sparse_pc_linear_size=int(active_setup.linear_size),
        pc_maxiter=int(context.pc_maxiter),
        default_permc_spec=context.default_permc_spec(
            constrained_pas_pc=bool(context.constrained_pas_pc),
            tokamak_pas_er_pc=bool(context.tokamak_pas_er_pc),
            n_species=int(context.op.n_species),
        ),
        host_sparse_factor_dtype=context.host_sparse_factor_dtype,
    )
    enforce_sparse_pc_memory_budget(
        SparsePCMemoryBudgetPreflightContext(
            env=context.env,
            unknowns=int(active_setup.linear_size),
            gmres_restart=int(context.pc_restart),
            csr_nnz=int(pattern_setup.summary.nnz),
            dtype=np.dtype(factor_policy.factor_dtype_initial),
            device_count=max(1, int(context.device_count)),
            estimate_sparse_pc_memory=context.estimate_sparse_pc_memory,
        )
    )
    return SparsePCGenericBranchSetupResult(
        active_idx_np=active_setup.active_idx_np,
        active_idx_jnp=active_setup.active_idx_jnp,
        full_to_active_jnp=active_setup.full_to_active_jnp,
        rhs=active_setup.rhs,
        linear_size=int(active_setup.linear_size),
        reduce_full=active_setup.reduce_full,
        expand_reduced=active_setup.expand_reduced,
        op_pc=op_pc,
        pattern_source_op=pattern_source_op,
        preconditioner_operator=preconditioner_operator,
        fortran_reduced_xblock_min_size=int(backend_setup.xblock_min_size),
        fortran_reduced_sparse_pc_backend=str(backend_setup.backend),
        fortran_reduced_sparse_pc_backend_reason=str(backend_setup.reason),
        pattern=pattern_setup.pattern,
        sparse_pattern_scope=str(pattern_setup.scope),
        pattern_build_s=float(pattern_setup.build_s),
        summary=pattern_setup.summary,
        factor_policy=factor_policy,
    )


def build_sparse_pc_direct_tail_factor_setup(
    context: SparsePCDirectTailFactorSetupContext,
) -> SparsePCDirectTailFactorSetupResult:
    """Build the direct-tail structured PC or fall back to a host sparse factor."""

    sparse_timer = context.sparse_timer
    emit = context.emit
    if emit is not None:
        shift_note = f" shift={float(context.pc_shift):.1e}" if context.pc_shift != 0.0 else ""
        emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres factoring RHSMode=1 preconditioner"
            f"{shift_note} factor_dtype={np.dtype(context.factor_dtype_initial).name} "
            f"factor_kind={context.factorization} permc={context.permc_spec}",
        )
    materialization = build_direct_tail_materialization_setup(
        DirectTailMaterializationContext(
            env=context.env,
            op=context.op,
            op_pc=context.op_pc,
            pattern=context.pattern,
            active_indices=context.active_indices,
            sparse_pc_use_active_dof=bool(context.sparse_pc_use_active_dof),
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            pc_shift=float(context.pc_shift),
            dtype=context.rhs_dtype,
            factor_dtype=context.factor_dtype_initial,
            sparse_pc_linear_size=int(context.sparse_pc_linear_size),
            default_pattern_color_batch=int(context.default_pattern_color_batch),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            is_direct_reduced_pmat_pc_kind=context.is_direct_reduced_pmat_pc_kind,
            build_direct_tail_bundle=context.build_direct_tail_bundle,
            build_structured_rhs1_full_csr_operator_bundle_callback=(
                context.build_structured_full_csr_operator_bundle
            ),
        )
    )
    direct_tail_operator_bundle = materialization.operator_bundle
    direct_tail_direct_reduced_pmat_requested = bool(
        materialization.direct_reduced_pmat_requested
    )
    factor_start_s = sparse_timer.elapsed_s()
    admission = resolve_direct_tail_structured_admission(
        DirectTailStructuredAdmissionContext(
            env=context.env,
            pc_env=str(materialization.pc_env),
            operator_bundle=direct_tail_operator_bundle,
            direct_reduced_pmat_requested=bool(
                direct_tail_direct_reduced_pmat_requested
            ),
            sparse_pc_linear_size=int(context.sparse_pc_linear_size),
            default_max_mb=context.default_direct_tail_max_mb,
        )
    )
    direct_tail_structured_pc_requested = admission.requested
    direct_tail_structured_pc_required = bool(admission.required)
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor setup start "
            f"size={int(context.sparse_pc_linear_size)} "
            f"factor_dtype={np.dtype(context.factor_dtype_initial).name} "
            f"factor_kind={context.factorization} "
            f"direct_tail_built={bool(materialization.built)} "
            f"structured_pc_requested={direct_tail_structured_pc_requested}",
        )

    structured_pc_ready = False
    direct_tail_structured_layout = None
    direct_tail_structured_active_indices = None
    direct_tail_structured_max_nbytes = None
    direct_tail_structured_pc_selected = False
    direct_tail_structured_pc_reason = None
    direct_tail_structured_pc_metadata = None
    direct_tail_structured_pc_error = None
    operator_bundle_pc = None
    factor_bundle_pc = None
    pc_max_mb = float(admission.max_mb)
    pc_reg = float(admission.regularization)
    if bool(admission.setup_allowed):
        start_s = sparse_timer.elapsed_s()
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                "structured preconditioner setup start "
                f"kind={direct_tail_structured_pc_requested} "
                f"active_size={int(context.sparse_pc_linear_size)} "
                f"max_mb={float(pc_max_mb):.3g} "
                f"max_mb_auto={bool(admission.max_mb_auto)} "
                f"reg={float(pc_reg):.3e}",
            )
        try:
            structured_build = build_direct_tail_structured_preconditioner_setup(
                DirectTailStructuredBuildContext(
                    env=context.env,
                    op=context.op_pc,
                    operator_bundle=direct_tail_operator_bundle,
                    active_indices=(
                        context.active_indices
                        if context.sparse_pc_use_active_dof
                        else None
                    ),
                    requested_kind=direct_tail_structured_pc_requested,
                    direct_reduced_pmat_requested=bool(
                        direct_tail_direct_reduced_pmat_requested
                    ),
                    sparse_pc_linear_size=int(context.sparse_pc_linear_size),
                    max_mb=float(pc_max_mb),
                    regularization=float(pc_reg),
                    preconditioner_x=int(context.preconditioner_x),
                    preconditioner_xi=int(context.preconditioner_xi),
                    preconditioner_species=int(context.preconditioner_species),
                    preconditioner_x_min_l=int(context.preconditioner_x_min_l),
                    layout_from_operator=context.layout_from_operator,
                    build_direct_active_preconditioner=(
                        context.build_direct_active_preconditioner
                    ),
                    build_active_projected_preconditioner=(
                        context.build_active_projected_preconditioner
                    ),
                    cache=context.structured_cache,
                    cache_key=context.structured_cache_key,
                    with_cache_metadata=context.structured_cache_metadata,
                    factor_bundle=context.structured_factor_bundle_factory,
                )
            )
            direct_tail_structured_layout = structured_build.layout
            direct_tail_structured_active_indices = structured_build.active_indices
            direct_tail_structured_max_nbytes = structured_build.max_nbytes
            direct_tail_structured_pc_selected = bool(structured_build.selected)
            direct_tail_structured_pc_reason = structured_build.reason
            direct_tail_structured_pc_metadata = structured_build.metadata
            direct_tail_structured_pc_error = structured_build.error
            cache_hit = bool(structured_build.cache_hit)
            if cache_hit and emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    "structured preconditioner cache hit "
                    f"elapsed_s={sparse_timer.elapsed_s() - start_s:.3f}",
                )
            if bool(structured_build.ready):
                factor_bundle_pc = structured_build.factor_bundle
                operator_bundle_pc = structured_build.operator_bundle_pc
                structured_pc_ready = True
                if emit is not None:
                    inner = {}
                    if isinstance(direct_tail_structured_pc_metadata, dict):
                        maybe_inner = direct_tail_structured_pc_metadata.get(
                            "metadata"
                        )
                        if isinstance(maybe_inner, dict):
                            inner = maybe_inner
                    factor_nbytes = inner.get("factor_nbytes_actual")
                    if factor_nbytes is None:
                        factor_nbytes = inner.get("factor_nbytes_estimate")
                    factor_permc = inner.get("permc_spec", "na")
                    factor_superlu_permc = inner.get("superlu_permc_spec", "na")
                    pc_kind = (
                        str(
                            direct_tail_structured_pc_metadata.get(
                                "kind", direct_tail_structured_pc_requested
                            )
                        )
                        if isinstance(direct_tail_structured_pc_metadata, dict)
                        else str(direct_tail_structured_pc_requested)
                    )
                    pc_setup_s = (
                        float(
                            direct_tail_structured_pc_metadata.get("setup_s", 0.0)
                            or 0.0
                        )
                        if isinstance(direct_tail_structured_pc_metadata, dict)
                        else 0.0
                    )
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        f"structured preconditioner selected kind={pc_kind} "
                        f"setup_s={float(pc_setup_s):.3f} "
                        f"elapsed_s={sparse_timer.elapsed_s() - start_s:.3f} "
                        f"reason={direct_tail_structured_pc_reason} "
                        f"cache_hit={cache_hit} "
                        f"factor_nbytes={factor_nbytes if factor_nbytes is not None else 'na'} "
                        f"permc={factor_permc} superlu_permc={factor_superlu_permc}",
                    )
            elif emit is not None:
                tail_action = (
                    "required path will fail fast"
                    if direct_tail_structured_pc_required
                    else "falling back to host factorization"
                )
                if direct_tail_structured_pc_error is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "structured preconditioner failed "
                        f"elapsed_s={sparse_timer.elapsed_s() - start_s:.3f} "
                        f"({direct_tail_structured_pc_error}); {tail_action}",
                    )
                else:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "structured preconditioner not selected "
                        f"kind={direct_tail_structured_pc_requested} "
                        f"reason={direct_tail_structured_pc_reason}; "
                        f"elapsed_s={sparse_timer.elapsed_s() - start_s:.3f}; "
                        f"{tail_action}",
                    )
        except Exception as exc:  # noqa: BLE001
            direct_tail_structured_pc_error = f"{type(exc).__name__}: {exc}"
            direct_tail_structured_pc_selected = False
            direct_tail_structured_pc_reason = "structured_pc_exception"
            if emit is not None:
                tail_action = (
                    "required path will fail fast"
                    if direct_tail_structured_pc_required
                    else "falling back to host factorization"
                )
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    "structured preconditioner failed "
                    f"elapsed_s={sparse_timer.elapsed_s() - start_s:.3f} "
                    f"({direct_tail_structured_pc_error}); {tail_action}",
                )
    if (
        direct_tail_structured_pc_requested is not None
        and direct_tail_structured_pc_required
        and not structured_pc_ready
    ):
        raise RuntimeError(
            "direct-tail structured preconditioner was explicitly requested but not selected: "
            f"kind={direct_tail_structured_pc_requested} "
            f"reason={direct_tail_structured_pc_reason} "
            f"error={direct_tail_structured_pc_error} "
            f"direct_tail_built={bool(materialization.built)} "
            f"direct_reduced_pmat_requested={bool(direct_tail_direct_reduced_pmat_requested)}"
        )
    if not structured_pc_ready:
        operator_bundle_pc, factor_bundle_pc = context.host_factor_builder(
            matvec=context.factor_matvec,
            n=int(context.sparse_pc_linear_size),
            dtype=context.rhs_dtype,
            factor_dtype=context.factor_dtype_initial,
            pattern=context.pattern,
            operator_bundle_override=direct_tail_operator_bundle,
            emit=emit,
            default_diag_pivot_thresh=(
                0.0
                if (
                    context.constrained_pas_pc
                    or context.tokamak_fp_pc
                    or context.fortran_reduced_sparse_pc
                )
                else 1.0
            ),
            default_permc_spec=context.default_permc_spec,
            default_factor_kind=context.default_factor_kind,
            default_ilu_fill_factor=float(context.default_ilu_fill_factor),
            default_ilu_drop_tol=float(context.default_ilu_drop_tol),
            default_pattern_color_batch=int(context.default_pattern_color_batch),
        )
    pc_factor_s = sparse_timer.elapsed_s() - factor_start_s
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor setup complete "
            f"elapsed_s={float(pc_factor_s):.3f} "
            f"structured_pc_ready={bool(structured_pc_ready)} "
            f"direct_tail_built={bool(materialization.built)}",
        )
    return SparsePCDirectTailFactorSetupResult(
        materialization=materialization,
        direct_tail_default=bool(materialization.direct_tail_default),
        direct_tail_enabled=bool(materialization.enabled),
        direct_tail_built=bool(materialization.built),
        direct_tail_error=materialization.error,
        direct_tail_operator_bundle=direct_tail_operator_bundle,
        direct_tail_structured_pc_requested=direct_tail_structured_pc_requested,
        direct_tail_structured_pc_selected=bool(direct_tail_structured_pc_selected),
        direct_tail_structured_pc_reason=direct_tail_structured_pc_reason,
        direct_tail_structured_pc_metadata=direct_tail_structured_pc_metadata,
        direct_tail_structured_pc_error=direct_tail_structured_pc_error,
        direct_tail_pc_env_early=str(materialization.pc_env),
        direct_tail_direct_reduced_pmat_requested=bool(
            direct_tail_direct_reduced_pmat_requested
        ),
        structured_admission=admission,
        direct_tail_pc_env=str(admission.pc_env),
        direct_tail_pc_auto_default=bool(admission.auto_default),
        direct_tail_fail_closed_size=int(admission.fail_closed_size),
        direct_tail_auto_large_fail_closed=bool(admission.auto_large_fail_closed),
        direct_tail_structured_pc_required=bool(admission.required),
        structured_pc_ready=bool(structured_pc_ready),
        direct_tail_structured_layout=direct_tail_structured_layout,
        direct_tail_structured_active_indices=direct_tail_structured_active_indices,
        direct_tail_structured_max_nbytes=direct_tail_structured_max_nbytes,
        direct_tail_support_mode_preflight_requested=False,
        direct_tail_support_mode_preflight_selected=False,
        direct_tail_support_mode_preflight_metadata=None,
        direct_tail_support_mode_preflight_error=None,
        direct_tail_structured_pc_max_mb_auto=bool(admission.max_mb_auto),
        pc_max_mb=float(pc_max_mb),
        pc_reg=float(pc_reg),
        operator_bundle_pc=operator_bundle_pc,
        factor_bundle_pc=factor_bundle_pc,
        pc_factor_s=float(pc_factor_s),
        setup_s=float(sparse_timer.elapsed_s()),
    )


@dataclass(frozen=True)
class FortranReducedXBlockBackendContext:
    """State needed to run the fortran-reduced x-block sparse-PC backend."""

    op: object
    op_pc: object
    rhs: jnp.ndarray
    sparse_pc_rhs: jnp.ndarray
    x0: jnp.ndarray | None
    sparse_pc_use_active_dof: bool
    sparse_pc_active_idx_jnp: jnp.ndarray | None
    sparse_pc_full_to_active_jnp: jnp.ndarray | None
    sparse_pc_linear_size: int
    sparse_pc_fp_dense_velocity_block: object
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    operator_matvec: ArrayFn
    preconditioner_x: int
    preconditioner_x_min_l: int
    preconditioner_xi: int
    preconditioner_species: int
    backend_reason: str
    xblock_min_size: int
    sparse_timer: object
    pc_restart: int
    pc_maxiter: int
    atol: float
    tol: float
    rhs_norm: float
    emit: EmitFn | None
    env: Mapping[str, str] | None
    rhs1_l2_norm_float: Callable[[jnp.ndarray], float]
    rhs1_residual_target: Callable[..., float]
    assembled_host_allowed: Callable[..., bool]
    xblock_preconditioner_builder: Callable[..., ArrayFn]
    moment_schur_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    lgmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    gcrotmk_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    bicgstab_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


def solve_fortran_reduced_xblock_backend(
    context: FortranReducedXBlockBackendContext,
) -> object:
    """Run the fortran-reduced x-block backend and return a linear-solve payload."""

    op = context.op
    op_pc = context.op_pc
    if op_pc.fblock.fp is None or op_pc.fblock.pas is not None:
        raise NotImplementedError(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND=xblock currently targets "
            "full-FP RHSMode=1 systems."
        )

    sparse_timer = context.sparse_timer
    emit = context.emit
    xblock_factor_build = build_fortran_reduced_xblock_factor_stage(
        context=FortranReducedXBlockFactorBuildContext(
            op_pc=op_pc,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_xi=int(context.preconditioner_xi),
            sparse_pc_linear_size=int(context.sparse_pc_linear_size),
            backend_reason=str(context.backend_reason),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            env=context.env,
            assembled_host_allowed=context.assembled_host_allowed,
            builder=context.xblock_preconditioner_builder,
        )
    )
    precond_xblock = xblock_factor_build.preconditioner
    xblock_drop_tol = float(xblock_factor_build.drop_tol)
    xblock_drop_rel = float(xblock_factor_build.drop_rel)
    xblock_ilu_drop_tol = float(xblock_factor_build.ilu_drop_tol)
    xblock_fill_factor = float(xblock_factor_build.fill_factor)
    xblock_preconditioner_xi = int(xblock_factor_build.preconditioner_xi)
    force_assembled_host_fp = bool(xblock_factor_build.force_assembled_host_fp)
    pc_factor_s = float(xblock_factor_build.factor_s)
    setup_s = sparse_timer.elapsed_s()

    xblock_krylov_setup = build_fortran_reduced_xblock_krylov_setup(
        context=FortranReducedXBlockKrylovSetupContext(
            op=op,
            rhs=context.rhs,
            xblock_use_active_dof=bool(context.sparse_pc_use_active_dof),
            active_idx=context.sparse_pc_active_idx_jnp,
            full_to_active=context.sparse_pc_full_to_active_jnp,
            reduce_full_with_indices=context.reduce_full_with_indices,
            expand_reduced_with_map=context.expand_reduced_with_map,
            operator_matvec=context.operator_matvec,
            base_preconditioner=precond_xblock,
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            env=context.env,
        )
    )
    precondition_side = xblock_krylov_setup.precondition_side
    pc_form = xblock_krylov_setup.pc_form
    xblock_krylov_method = xblock_krylov_setup.krylov_method
    progress_every = xblock_krylov_setup.progress_every
    mv_count = xblock_krylov_setup.mv_count
    matvec_no_count = xblock_krylov_setup.matvec_no_count
    matvec = xblock_krylov_setup.matvec
    preconditioner = xblock_krylov_setup.preconditioner

    moment_schur_policy = resolve_fortran_reduced_xblock_moment_schur_policy(
        precondition_side=precondition_side,
        env=context.env,
    )
    moment_schur_result = apply_fortran_reduced_xblock_moment_schur_stage(
        context=FortranReducedXBlockMomentSchurStageContext(
            op=op,
            base_preconditioner=preconditioner,
            reduce_full=(
                context.reduce_full if context.sparse_pc_use_active_dof else None
            ),
            expand_reduced=(
                context.expand_reduced if context.sparse_pc_use_active_dof else None
            ),
            policy=moment_schur_policy,
            precondition_side=str(precondition_side),
            rhs=context.sparse_pc_rhs,
            matvec_no_count=matvec_no_count,
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            builder=context.moment_schur_builder,
        )
    )
    preconditioner = moment_schur_result.preconditioner
    pc_factor_s += float(moment_schur_result.setup_s)

    global_coupling_policy = resolve_fortran_reduced_xblock_global_coupling_policy(
        precondition_side=precondition_side,
        env=context.env,
    )
    global_coupling_result = apply_fortran_reduced_xblock_global_coupling_stage(
        context=FortranReducedXBlockGlobalCouplingStageContext(
            op=op,
            rhs=context.rhs,
            matvec=matvec_no_count,
            base_preconditioner=preconditioner,
            direction_projector=(
                context.reduce_full if context.sparse_pc_use_active_dof else None
            ),
            expected_size=int(context.sparse_pc_linear_size),
            policy=global_coupling_policy,
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
        )
    )
    preconditioner = global_coupling_result.preconditioner
    pc_factor_s += float(global_coupling_result.setup_s)

    x0_setup = prepare_fortran_reduced_xblock_initial_guess(
        x0=context.x0,
        sparse_pc_rhs=context.sparse_pc_rhs,
        full_rhs=context.rhs,
        reduce_full=context.reduce_full,
    )
    x0_sparse = x0_setup.x0_full
    if emit is not None:
        for level, message in x0_setup.messages:
            emit(level, message)

    sparse_pc_rhs_norm = context.rhs1_l2_norm_float(context.sparse_pc_rhs)
    target = context.rhs1_residual_target(
        atol=float(context.atol),
        tol=float(context.tol),
        rhs_norm=float(sparse_pc_rhs_norm),
    )
    initial_seed_policy = resolve_fortran_reduced_xblock_initial_seed_policy(
        env=context.env,
    )
    initial_seed_result = apply_fortran_reduced_xblock_initial_seed(
        policy=initial_seed_policy,
        rhs=context.sparse_pc_rhs,
        rhs_norm=float(sparse_pc_rhs_norm),
        x0=x0_sparse,
        preconditioner=preconditioner,
        matvec_no_count=matvec_no_count,
        elapsed_s=sparse_timer.elapsed_s,
    )
    x0_sparse = initial_seed_result.x0
    if emit is not None:
        for level, message in initial_seed_result.messages:
            emit(level, message)

    krylov_result = run_fortran_reduced_xblock_krylov_solve(
        context=FortranReducedXBlockKrylovSolveContext(
            matvec=matvec,
            rhs=context.sparse_pc_rhs,
            preconditioner=preconditioner,
            emit=emit,
            elapsed_s=sparse_timer.elapsed_s,
            method=str(xblock_krylov_method),
            pc_form=str(pc_form),
            restart=int(context.pc_restart),
            maxiter=int(context.pc_maxiter),
            tol=float(context.tol),
            atol=float(context.atol),
            target=float(target),
            precondition_side=str(precondition_side),
            progress_every=int(progress_every),
            mv_count=mv_count,
            explicit_left_solver=context.explicit_left_solver,
            gmres_solver=context.gmres_solver,
            lgmres_solver=context.lgmres_solver,
            gcrotmk_solver=context.gcrotmk_solver,
            bicgstab_solver=context.bicgstab_solver,
        ),
        x0=x0_sparse,
    )
    return fortran_reduced_xblock_final_payload(
        FortranReducedXBlockFinalPayloadContext(
            diagnostic_state={
                "op": op,
                "fortran_reduced_sparse_pc_backend_reason": context.backend_reason,
                "fortran_reduced_xblock_min_size": context.xblock_min_size,
                "preconditioner_x": context.preconditioner_x,
                "preconditioner_x_min_l": context.preconditioner_x_min_l,
                "preconditioner_xi": context.preconditioner_xi,
                "preconditioner_species": context.preconditioner_species,
                "xblock_preconditioner_xi": xblock_preconditioner_xi,
                "force_assembled_host_fp": force_assembled_host_fp,
                "xblock_krylov_method": xblock_krylov_method,
                "seed_enabled": bool(initial_seed_policy.enabled),
                "seed_used": bool(initial_seed_result.used),
                "seed_residual_norm": initial_seed_result.residual_norm,
                "seed_improvement_ratio": initial_seed_result.improvement_ratio,
                "seed_accept_ratio": float(initial_seed_policy.accept_ratio),
                "seed_refine_steps": int(initial_seed_policy.refine_steps),
                "seed_refines_performed": int(initial_seed_result.refines_performed),
                "moment_schur_enabled": bool(moment_schur_policy.enabled),
                "moment_schur_built": bool(moment_schur_result.built),
                "moment_schur_used": bool(moment_schur_result.used),
                "moment_schur_reason": moment_schur_result.reason,
                "moment_schur_metadata": moment_schur_result.metadata,
                "moment_schur_stats": moment_schur_result.stats,
                "moment_schur_probe_residual_before": (
                    moment_schur_result.probe_residual_before
                ),
                "moment_schur_probe_residual_after": (
                    moment_schur_result.probe_residual_after
                ),
                "moment_schur_probe_improvement_ratio": (
                    moment_schur_result.probe_improvement_ratio
                ),
                "global_coupling_enabled": bool(global_coupling_policy.enabled),
                "global_coupling_built": bool(global_coupling_result.built),
                "global_coupling_metadata": global_coupling_result.metadata,
                "global_coupling_stats": global_coupling_result.stats,
                "xblock_drop_tol": xblock_drop_tol,
                "xblock_drop_rel": xblock_drop_rel,
                "xblock_ilu_drop_tol": xblock_ilu_drop_tol,
                "xblock_fill_factor": xblock_fill_factor,
                "sparse_pc_use_active_dof": context.sparse_pc_use_active_dof,
                "sparse_pc_linear_size": context.sparse_pc_linear_size,
                "sparse_pc_fp_dense_velocity_block": (
                    context.sparse_pc_fp_dense_velocity_block
                ),
                "setup_s": setup_s,
                "solve_s": float(krylov_result.solve_s),
                "sparse_timer": sparse_timer,
                "pc_factor_s": pc_factor_s,
                "target": target,
                "mv_count": mv_count,
                "pc_restart": context.pc_restart,
                "pc_maxiter": context.pc_maxiter,
                "history": tuple(krylov_result.history),
                "residual_norm_sparse_pc": float(krylov_result.residual_norm),
            },
            result=krylov_result,
            atol=float(context.atol),
            tol=float(context.tol),
            rhs_norm=float(context.rhs_norm),
            target=float(target),
        ),
        expand_reduced=context.expand_reduced,
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








@dataclass(frozen=True)
class XBlockSparsePCBranchContext:
    """Solve-local state and callbacks for the x-block sparse-PC branch."""

    _apply_device_subspace_residual_equation_correction: object
    _apply_preconditioned_minres_correction: object
    _apply_subspace_minres_correction: object
    _rhs1_xblock_post_coarse_directions: object
    _build_rhs1_xblock_constraint1_moment_schur_preconditioner: object
    _build_rhsmode1_xblock_tz_sparse_preconditioner: object
    _read_rhs1_post_solve_correction_policy: object
    _read_rhs1_probe_coarse_policy: object
    _rhs1_bool_env: object
    _rhs1_float_env: object
    _rhs1_xblock_fallback_initial_guess: object
    _rhs1_xblock_policy: object
    _rhsmode1_fp_xblock_assembled_host_allowed: object
    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly: object
    active_idx_jnp: object
    active_size: object
    apply_v3_full_system_operator_cached: object
    atol: object
    bicgstab_solve_with_history_scipy: object
    bicgstab_solve_with_residual: object
    build_operator_from_pattern: object
    device_csr_from_matrix: object
    emit: object
    estimate_v3_full_system_conservative_sparsity_summary: object
    expand_reduced_with_map: object
    fgmres_cycle_jit_solve_with_residual: object
    fgmres_solve_with_residual: object
    fgmres_solve_with_residual_jit: object
    full_to_active_jnp: object
    gcrotmk_solve_with_history_scipy: object
    gmres_solve_with_history_scipy: object
    include_electric_field_xi_sparse_pc: object
    include_xdot_sparse_pc: object
    lgmres_solve_with_history_scipy: object
    op: object
    pc_maxiter: object
    pc_restart: object
    pc_restart_env: object
    preconditioner_species: object
    preconditioner_xi: object
    reduce_full_with_indices: object
    resolve_rhs1_xblock_sparse_pc_policy: object
    rhs: object
    rhs1_l2_norm_float: object
    rhs1_residual_target: object
    rhs1_safe_ratio: object
    sparse_pc_fp_dense_velocity_block: object
    sparse_timer: object
    summarize_v3_sparse_pattern: object
    tfqmr_solve_with_residual: object
    tokamak_fp_er_pc: object
    tol: object
    use_dkes: object
    v3_full_system_conservative_sparsity_pattern: object
    v3_full_system_conservative_sparsity_pattern_for_indices: object
    v3_linear_solve_result_from_payload: object
    validate_device_csr_matvec: object
    x0: object
    xblock_sparse_pc: object
    xblock_use_active_dof: object


def run_xblock_sparse_pc_branch(context: XBlockSparsePCBranchContext):
    """Run the RHSMode=1 x-block sparse-PC GMRES branch outside solve.py."""

    _apply_device_subspace_residual_equation_correction = context._apply_device_subspace_residual_equation_correction
    _apply_preconditioned_minres_correction = context._apply_preconditioned_minres_correction
    _apply_subspace_minres_correction = context._apply_subspace_minres_correction
    _rhs1_xblock_post_coarse_directions = context._rhs1_xblock_post_coarse_directions
    _build_rhs1_xblock_constraint1_moment_schur_preconditioner = context._build_rhs1_xblock_constraint1_moment_schur_preconditioner
    _build_rhsmode1_xblock_tz_sparse_preconditioner = context._build_rhsmode1_xblock_tz_sparse_preconditioner
    _read_rhs1_post_solve_correction_policy = context._read_rhs1_post_solve_correction_policy
    _read_rhs1_probe_coarse_policy = context._read_rhs1_probe_coarse_policy
    _rhs1_bool_env = context._rhs1_bool_env
    _rhs1_float_env = context._rhs1_float_env
    _rhs1_xblock_fallback_initial_guess = context._rhs1_xblock_fallback_initial_guess
    _rhs1_xblock_policy = context._rhs1_xblock_policy
    _rhsmode1_fp_xblock_assembled_host_allowed = context._rhsmode1_fp_xblock_assembled_host_allowed
    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly = context._rhsmode1_fp_xblock_species_decoupled_for_host_assembly
    active_idx_jnp = context.active_idx_jnp
    active_size = context.active_size
    apply_v3_full_system_operator_cached = context.apply_v3_full_system_operator_cached
    atol = context.atol
    bicgstab_solve_with_history_scipy = context.bicgstab_solve_with_history_scipy
    bicgstab_solve_with_residual = context.bicgstab_solve_with_residual
    build_operator_from_pattern = context.build_operator_from_pattern
    device_csr_from_matrix = context.device_csr_from_matrix
    emit = context.emit
    estimate_v3_full_system_conservative_sparsity_summary = context.estimate_v3_full_system_conservative_sparsity_summary
    expand_reduced_with_map = context.expand_reduced_with_map
    fgmres_cycle_jit_solve_with_residual = context.fgmres_cycle_jit_solve_with_residual
    fgmres_solve_with_residual = context.fgmres_solve_with_residual
    fgmres_solve_with_residual_jit = context.fgmres_solve_with_residual_jit
    full_to_active_jnp = context.full_to_active_jnp
    gcrotmk_solve_with_history_scipy = context.gcrotmk_solve_with_history_scipy
    gmres_solve_with_history_scipy = context.gmres_solve_with_history_scipy
    include_electric_field_xi_sparse_pc = context.include_electric_field_xi_sparse_pc
    include_xdot_sparse_pc = context.include_xdot_sparse_pc
    lgmres_solve_with_history_scipy = context.lgmres_solve_with_history_scipy
    op = context.op
    pc_maxiter = context.pc_maxiter
    pc_restart = context.pc_restart
    pc_restart_env = context.pc_restart_env
    preconditioner_species = context.preconditioner_species
    preconditioner_xi = context.preconditioner_xi
    reduce_full_with_indices = context.reduce_full_with_indices
    resolve_rhs1_xblock_sparse_pc_policy = context.resolve_rhs1_xblock_sparse_pc_policy
    rhs = context.rhs
    rhs1_l2_norm_float = context.rhs1_l2_norm_float
    rhs1_residual_target = context.rhs1_residual_target
    rhs1_safe_ratio = context.rhs1_safe_ratio
    sparse_pc_fp_dense_velocity_block = context.sparse_pc_fp_dense_velocity_block
    sparse_timer = context.sparse_timer
    summarize_v3_sparse_pattern = context.summarize_v3_sparse_pattern
    tfqmr_solve_with_residual = context.tfqmr_solve_with_residual
    tokamak_fp_er_pc = context.tokamak_fp_er_pc
    tol = context.tol
    use_dkes = context.use_dkes
    v3_full_system_conservative_sparsity_pattern = context.v3_full_system_conservative_sparsity_pattern
    v3_full_system_conservative_sparsity_pattern_for_indices = context.v3_full_system_conservative_sparsity_pattern_for_indices
    v3_linear_solve_result_from_payload = context.v3_linear_solve_result_from_payload
    validate_device_csr_matvec = context.validate_device_csr_matvec
    x0 = context.x0
    xblock_sparse_pc = context.xblock_sparse_pc
    xblock_use_active_dof = context.xblock_use_active_dof
    if xblock_sparse_pc:
        xblock_branch_setup = resolve_xblock_sparse_pc_branch_setup(
            op=op,
            preconditioner_species=int(preconditioner_species),
            preconditioner_xi=int(preconditioner_xi),
            active_size=int(active_size),
            pc_restart=int(pc_restart),
            pc_restart_env=str(pc_restart_env),
            tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
            use_dkes=bool(use_dkes),
            include_xdot_sparse_pc=bool(include_xdot_sparse_pc),
            include_electric_field_xi_sparse_pc=bool(include_electric_field_xi_sparse_pc),
            lower_fill_mode=_rhs1_xblock_policy.rhs1_xblock_lower_fill_mode,
            species_decoupled_for_host_assembly=_rhsmode1_fp_xblock_species_decoupled_for_host_assembly,
            assembled_host_allowed=_rhsmode1_fp_xblock_assembled_host_allowed,
            krylov_method=_rhs1_xblock_policy.rhs1_xblock_krylov_method,
            device_host_fallback_decision=_rhs1_xblock_policy.rhs1_xblock_device_host_fallback_decision,
            resolve_xblock_policy=resolve_rhs1_xblock_sparse_pc_policy,
            reuse_decision=_rhs1_xblock_policy.rhs1_xblock_qi_device_operator_reuse_decision,
            env=os.environ,
        )
        xblock_drop_tol = float(xblock_branch_setup.xblock_drop_tol)
        xblock_drop_rel = float(xblock_branch_setup.xblock_drop_rel)
        xblock_ilu_drop_tol = float(xblock_branch_setup.xblock_ilu_drop_tol)
        xblock_fill_factor = float(xblock_branch_setup.xblock_fill_factor)
        xblock_lower_fill_mode = str(xblock_branch_setup.xblock_lower_fill_mode)
        xblock_lower_fill_ignored_env = bool(xblock_branch_setup.xblock_lower_fill_ignored_env)
        xblock_preconditioner_xi = int(xblock_branch_setup.xblock_preconditioner_xi)
        force_assembled_host_fp = bool(xblock_branch_setup.force_assembled_host_fp)
        xblock_assembled_host_fp = bool(xblock_branch_setup.xblock_assembled_host_fp)
        xblock_krylov_env_requested = str(xblock_branch_setup.xblock_krylov_env_requested)
        xblock_krylov_env = str(xblock_branch_setup.xblock_krylov_env)
        xblock_krylov_requested = str(xblock_branch_setup.xblock_krylov_requested)
        xblock_device_fgmres_requested = bool(xblock_branch_setup.xblock_device_fgmres_requested)
        xblock_device_gmres_requested = bool(xblock_branch_setup.xblock_device_gmres_requested)
        xblock_device_bicgstab_requested = bool(xblock_branch_setup.xblock_device_bicgstab_requested)
        xblock_device_tfqmr_requested = bool(xblock_branch_setup.xblock_device_tfqmr_requested)
        xblock_device_krylov_requested = bool(xblock_branch_setup.xblock_device_krylov_requested)
        xblock_device_host_fallback_decision = xblock_branch_setup.xblock_device_host_fallback_decision
        xblock_device_host_fallback_auto_disabled_by_qi_device = bool(
            xblock_branch_setup.xblock_device_host_fallback_auto_disabled_by_qi_device
        )
        qi_device_preconditioner_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_preconditioner_requested_for_fallback
        )
        qi_device_matrix_free_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_matrix_free_requested_for_fallback
        )
        qi_device_use_in_krylov_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_use_in_krylov_requested_for_fallback
        )
        xblock_jax_factors = bool(xblock_branch_setup.xblock_jax_factors)
        xblock_jax_factor_format = str(xblock_branch_setup.xblock_jax_factor_format)
        xblock_jax_factor_apply = str(xblock_branch_setup.xblock_jax_factor_apply)
        xblock_device_krylov_forced_jax_factors = bool(
            xblock_branch_setup.xblock_device_krylov_forced_jax_factors
        )
        full_fp_3d_pc = bool(xblock_branch_setup.full_fp_3d_pc)
        side_env = str(xblock_branch_setup.side_env)
        precondition_side = str(xblock_branch_setup.precondition_side)
        xblock_default_right_pc = bool(xblock_branch_setup.xblock_default_right_pc)
        xblock_krylov_method = str(xblock_branch_setup.xblock_krylov_method)
        xblock_device_fgmres_forced_right_pc = bool(
            xblock_branch_setup.xblock_device_fgmres_forced_right_pc
        )
        pc_restart = int(xblock_branch_setup.pc_restart)
        xblock_default_restart_capped = bool(xblock_branch_setup.xblock_default_restart_capped)
        xblock_qi_device_operator_reuse_decision = xblock_branch_setup.xblock_qi_device_operator_reuse_decision
        xblock_qi_device_operator_reuse_skip_factors = bool(
            xblock_branch_setup.xblock_qi_device_operator_reuse_skip_factors
        )
        if emit is not None:
            for level, message in xblock_branch_setup.messages:
                emit(int(level), str(message))
        xblock_local_preconditioner = build_xblock_local_preconditioner(
            skip_factors=bool(xblock_qi_device_operator_reuse_skip_factors),
            elapsed_s=sparse_timer.elapsed_s,
            build_preconditioner=_build_rhsmode1_xblock_tz_sparse_preconditioner,
            op=op,
            build_jax_factors=bool(xblock_jax_factors),
            preconditioner_species=preconditioner_species,
            preconditioner_xi=xblock_preconditioner_xi,
            drop_tol=xblock_drop_tol,
            drop_rel=xblock_drop_rel,
            ilu_drop_tol=xblock_ilu_drop_tol,
            fill_factor=xblock_fill_factor,
            force_assembled_host_fp=bool(force_assembled_host_fp),
            emit=emit,
        )
        precond_xblock = xblock_local_preconditioner.preconditioner
        pc_factor_s = float(xblock_local_preconditioner.factor_s)
        xblock_preconditioner_built = bool(xblock_local_preconditioner.built)
        setup_s = sparse_timer.elapsed_s()
        xblock_matvec_setup = build_xblock_krylov_matvec_setup(
            op=op,
            rhs=rhs,
            xblock_use_active_dof=bool(xblock_use_active_dof),
            active_idx=active_idx_jnp,
            full_to_active=full_to_active_jnp,
            reduce_full_with_indices=reduce_full_with_indices,
            expand_reduced_with_map=expand_reduced_with_map,
            operator_matvec=lambda x_full: apply_v3_full_system_operator_cached(op, x_full),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            env=os.environ,
        )
        progress_every = int(xblock_matvec_setup.progress_every)
        mv_count = xblock_matvec_setup.mv_count
        xblock_linear_size = int(xblock_matvec_setup.xblock_linear_size)
        xblock_active_idx_np = xblock_matvec_setup.xblock_active_idx_np
        xblock_rhs = xblock_matvec_setup.xblock_rhs
        _xblock_reduce_full = xblock_matvec_setup.reduce_full
        _xblock_expand_reduced = xblock_matvec_setup.expand_reduced
        _mv_true_no_count = xblock_matvec_setup.matvec_no_count
        _mv_true = xblock_matvec_setup.matvec
        if emit is not None:
            for level, message in xblock_matvec_setup.messages:
                emit(int(level), str(message))

        _mv_xblock_krylov = _mv_true

        def _precond_xblock_krylov_base(v: jnp.ndarray) -> jnp.ndarray:
            if not xblock_use_active_dof:
                return precond_xblock(v)
            z_full = precond_xblock(_xblock_expand_reduced(jnp.asarray(v, dtype=rhs.dtype)))
            return _xblock_reduce_full(z_full)

        assembled_operator_enabled = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR",
            default=False,
        )
        assembled_operator = build_xblock_assembled_operator_if_requested(
            enabled=bool(assembled_operator_enabled),
            op=op,
            rhs_dtype=rhs.dtype,
            xblock_active_idx_np=xblock_active_idx_np,
            sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
            xblock_krylov_method=str(xblock_krylov_method),
            xblock_linear_size=int(xblock_linear_size),
            true_matvec_no_count=_mv_true_no_count,
            default_matvec=_mv_xblock_krylov,
            mv_count=mv_count,
            progress_every=int(progress_every),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            estimate_summary=estimate_v3_full_system_conservative_sparsity_summary,
            full_pattern=v3_full_system_conservative_sparsity_pattern,
            active_pattern=v3_full_system_conservative_sparsity_pattern_for_indices,
            summarize_pattern=summarize_v3_sparse_pattern,
            build_operator_from_pattern=build_operator_from_pattern,
            device_csr_from_matrix=device_csr_from_matrix,
            validate_device_csr_matvec=validate_device_csr_matvec,
            finalize_metadata=finalize_xblock_assembled_operator_metadata,
            backend=str(jax.default_backend()),
            env=os.environ,
        )
        _mv_xblock_krylov = assembled_operator.matvec
        assembled_operator_built = bool(assembled_operator.built)
        assembled_operator_device_resident = bool(assembled_operator.device_resident)
        assembled_operator_metadata = dict(assembled_operator.metadata)
        assembled_device_operator = assembled_operator.device_operator
        pc_factor_s += float(assembled_operator.pc_factor_increment_s)
        xblock_row_equilibration_enabled = bool(assembled_operator.row_enabled)
        xblock_row_equilibration_built = bool(assembled_operator.row_built)
        xblock_row_equilibration_metadata = dict(assembled_operator.row_metadata)
        xblock_row_scale_jnp = assembled_operator.row_scale
        xblock_inv_row_scale_jnp = assembled_operator.inv_row_scale
        xblock_col_equilibration_enabled = bool(assembled_operator.col_enabled)
        xblock_col_equilibration_built = bool(assembled_operator.col_built)
        xblock_col_equilibration_metadata = dict(assembled_operator.col_metadata)
        xblock_col_scale_jnp = assembled_operator.col_scale
        xblock_inv_col_scale_jnp = assembled_operator.inv_col_scale

        precond_xblock_krylov = _precond_xblock_krylov_base
        moment_schur_policy = resolve_xblock_moment_schur_policy_setup(
            op=op,
            xblock_krylov_method=str(xblock_krylov_method),
            xblock_jax_factors=bool(xblock_jax_factors),
            xblock_jax_factor_format=str(xblock_jax_factor_format),
            precondition_side=str(precondition_side),
            env=os.environ,
        )
        moment_schur_default_candidate = bool(moment_schur_policy.default_candidate)
        moment_schur_default_blocked_by_compact_factors = bool(
            moment_schur_policy.default_blocked_by_compact_factors
        )
        moment_schur_enabled = bool(moment_schur_policy.enabled)
        moment_schur_stage = apply_xblock_moment_schur_stage(
            context=XBlockMomentSchurStageContext(
                op=op,
                base_preconditioner=precond_xblock_krylov,
                reduce_full=_xblock_reduce_full if xblock_use_active_dof else None,
                expand_reduced=_xblock_expand_reduced if xblock_use_active_dof else None,
                policy=moment_schur_policy,
                precondition_side=str(precondition_side),
                rhs=xblock_rhs,
                matvec_no_count=_mv_true_no_count,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                builder=_build_rhs1_xblock_constraint1_moment_schur_preconditioner,
            )
        )
        precond_xblock_krylov = moment_schur_stage.preconditioner
        moment_schur_built = bool(moment_schur_stage.built)
        moment_schur_used = bool(moment_schur_stage.used)
        moment_schur_reason = moment_schur_stage.reason
        moment_schur_probe_residual_before = moment_schur_stage.probe_residual_before
        moment_schur_probe_residual_after = moment_schur_stage.probe_residual_after
        moment_schur_probe_improvement_ratio = moment_schur_stage.probe_improvement_ratio
        moment_schur_metadata = moment_schur_stage.metadata
        moment_schur_stats = moment_schur_stage.stats
        pc_factor_s += float(moment_schur_stage.setup_s)

        two_level_policy = resolve_xblock_two_level_policy_setup(
            precondition_side=str(precondition_side),
            env=os.environ,
        )
        two_level_enabled = bool(two_level_policy.enabled)
        two_level_stage = apply_xblock_two_level_stage(
            context=XBlockTwoLevelStageContext(
                op=op,
                rhs=rhs,
                matvec=_mv_xblock_krylov,
                base_preconditioner=precond_xblock_krylov,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                policy=two_level_policy,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        precond_xblock_krylov = two_level_stage.preconditioner
        two_level_built = bool(two_level_stage.built)
        two_level_metadata = two_level_stage.metadata
        two_level_stats = two_level_stage.stats
        pc_factor_s += float(two_level_stage.setup_s)

        global_coupling_policy = resolve_xblock_global_coupling_policy_setup(
            precondition_side=str(precondition_side),
            xblock_krylov_method=str(xblock_krylov_method),
            env=os.environ,
        )
        global_coupling_enabled = bool(global_coupling_policy.enabled)
        global_coupling_stage = apply_xblock_global_coupling_stage(
            context=XBlockGlobalCouplingStageContext(
                op=op,
                rhs=rhs,
                matvec=_mv_xblock_krylov,
                base_preconditioner=precond_xblock_krylov,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                policy=global_coupling_policy,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        precond_xblock_krylov = global_coupling_stage.preconditioner
        global_coupling_built = bool(global_coupling_stage.built)
        global_coupling_metadata = global_coupling_stage.metadata
        global_coupling_stats = global_coupling_stage.stats
        pc_factor_s += float(global_coupling_stage.setup_s)

        setup_s = sparse_timer.elapsed_s()
        x0_setup = prepare_xblock_initial_guess(
            x0=x0,
            xblock_rhs=xblock_rhs,
            full_rhs=rhs,
            xblock_use_active_dof=bool(xblock_use_active_dof),
            reduce_full=_xblock_reduce_full,
        )
        x0_full = x0_setup.x0_full
        for level, message in x0_setup.messages:
            if emit is not None:
                emit(level, message)
        xblock_initial_seed_used = False
        xblock_initial_seed_residual_norm: float | None = None
        xblock_initial_seed_residual_ratio: float | None = None
        seed_policy = resolve_xblock_seed_policy_setup(
            moment_schur_used=bool(moment_schur_used),
            env=os.environ,
        )
        seed_enabled = bool(seed_policy.initial_seed_enabled)
        if x0_full is None and seed_enabled:
            try:
                seed_vec = jnp.asarray(precond_xblock_krylov(xblock_rhs), dtype=jnp.float64)
                if seed_vec.shape == xblock_rhs.shape and bool(jnp.all(jnp.isfinite(seed_vec))):
                    seed_residual = xblock_rhs - _mv_true(seed_vec)
                    seed_residual_norm = rhs1_l2_norm_float(seed_residual)
                    rhs_norm_float = rhs1_l2_norm_float(xblock_rhs)
                    xblock_initial_seed_residual_norm = float(seed_residual_norm)
                    xblock_initial_seed_residual_ratio = rhs1_safe_ratio(
                        seed_residual_norm,
                        rhs_norm_float,
                    )
                    if np.isfinite(seed_residual_norm) and seed_residual_norm < rhs_norm_float:
                        x0_full = seed_vec
                        xblock_initial_seed_used = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                f"initial x-block seed residual={seed_residual_norm:.6e} "
                                f"rhs_norm={rhs_norm_float:.6e}",
                            )
                    elif emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"initial x-block seed rejected residual={seed_residual_norm:.6e} "
                            f"rhs_norm={rhs_norm_float:.6e}",
                        )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        f"initial x-block seed failed ({type(exc).__name__}: {exc})",
                    )

        xblock_rhs_norm = rhs1_l2_norm_float(xblock_rhs)
        target_xblock = rhs1_residual_target(
            atol=float(atol),
            tol=float(tol),
            rhs_norm=float(xblock_rhs_norm),
        )
        moment_schur_seed_enabled = bool(seed_policy.moment_schur_seed_enabled)
        moment_schur_seed_used = False
        moment_schur_seed_residual_norm: float | None = None
        moment_schur_seed_residual_ratio: float | None = None
        if moment_schur_seed_enabled and moment_schur_built:
            try:
                seed_vec = jnp.asarray(precond_xblock_krylov(xblock_rhs), dtype=jnp.float64)
                if seed_vec.shape == xblock_rhs.shape and bool(jnp.all(jnp.isfinite(seed_vec))):
                    seed_residual = xblock_rhs - jnp.asarray(_mv_true_no_count(seed_vec), dtype=jnp.float64)
                    seed_residual_norm = rhs1_l2_norm_float(seed_residual)
                    moment_schur_seed_residual_norm = float(seed_residual_norm)
                    moment_schur_seed_residual_ratio = rhs1_safe_ratio(
                        seed_residual_norm,
                        target_xblock,
                    )
                    incumbent_norm = float(xblock_rhs_norm)
                    if x0_full is not None:
                        incumbent_residual = xblock_rhs - jnp.asarray(
                            _mv_true_no_count(jnp.asarray(x0_full, dtype=jnp.float64)),
                            dtype=jnp.float64,
                        )
                        incumbent_norm = rhs1_l2_norm_float(incumbent_residual)
                    if np.isfinite(seed_residual_norm) and float(seed_residual_norm) < float(incumbent_norm):
                        x0_full = seed_vec
                        moment_schur_seed_used = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                f"constraint1 moment-Schur seed residual={seed_residual_norm:.6e} "
                                f"rhs_norm={float(xblock_rhs_norm):.6e}",
                            )
                    elif emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"constraint1 moment-Schur seed rejected residual={seed_residual_norm:.6e} "
                            f"incumbent={float(incumbent_norm):.6e}",
                        )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        f"constraint1 moment-Schur seed failed ({type(exc).__name__}: {exc})",
                    )
        qi_pipeline = run_xblock_qi_preconditioner_pipeline(
            build_xblock_qi_stage_pipeline_context(
                op=op,
                rhs=rhs,
                x0_full=x0_full,
                xblock_rhs=xblock_rhs,
                xblock_rhs_norm=float(xblock_rhs_norm),
                base_preconditioner=precond_xblock_krylov,
                matvec=_mv_xblock_krylov,
                true_matvec_no_count=_mv_true_no_count,
                direction_projector=(
                    _xblock_reduce_full if xblock_use_active_dof else None
                ),
                active_dof=bool(xblock_use_active_dof),
                linear_size=int(xblock_linear_size),
                host_fallback_used=bool(xblock_device_host_fallback_decision.used),
                precondition_side=str(precondition_side),
                assembled_device_operator=assembled_device_operator,
                assembled_operator_metadata=assembled_operator_metadata,
                assembled_operator_enabled=bool(assembled_operator_enabled),
                assembled_operator_built=bool(assembled_operator_built),
                assembled_operator_device_resident=bool(
                    assembled_operator_device_resident
                ),
                assembled_operator_device_error=assembled_operator_metadata.get(
                    "device_error"
                ),
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                env=os.environ,
                reduce_full=_xblock_reduce_full,
            )
        )
        precond_xblock_krylov = qi_pipeline.preconditioner
        x0_full = qi_pipeline.x0_full
        qi_device_state_for_augmented_krylov = (
            qi_pipeline.qi_device_state_for_augmented_krylov
        )
        qi_device_augmented_seed_basis_for_krylov = (
            qi_pipeline.qi_device_augmented_seed_basis_for_krylov
        )
        qi_device_augmented_seed_action_for_krylov = (
            qi_pipeline.qi_device_augmented_seed_action_for_krylov
        )
        qi_device_augmented_seed_available = (
            qi_pipeline.qi_device_augmented_seed_available
        )
        qi_device_augmented_seed_used = qi_pipeline.qi_device_augmented_seed_used
        qi_device_augmented_seed_rank = qi_pipeline.qi_device_augmented_seed_rank
        qi_device_preconditioner_metadata = (
            qi_pipeline.qi_device_preconditioner_metadata
        )
        pc_factor_s += float(qi_pipeline.pc_factor_s)
        xblock_side_probe_controls = _rhs1_xblock_policy.rhs1_xblock_side_probe_controls_from_env(
            env=os.environ,
            explicit_side_env_value=side_env,
            full_fp_3d_pc=bool(full_fp_3d_pc),
            active_size=int(active_size),
            krylov_method=str(xblock_krylov_method),
            precondition_side=str(precondition_side),
            pc_restart=int(pc_restart),
            pc_maxiter=int(pc_maxiter),
            backend=str(jax.default_backend()),
            krylov_env_value=xblock_krylov_env,
            device_host_fallback_used=bool(xblock_device_host_fallback_decision.used),
        )
        xblock_side_probe_stage = apply_xblock_side_probe_stage(
            XBlockSideProbeStageContext(
                controls=xblock_side_probe_controls,
                precondition_side=str(precondition_side),
                krylov_method=str(xblock_krylov_method),
                pc_maxiter=pc_maxiter,
                side_env=str(side_env),
                global_coupling_built=bool(global_coupling_built),
                matvec=_mv_xblock_krylov,
                true_matvec_no_count=_mv_true_no_count,
                rhs=xblock_rhs,
                rhs_norm=float(xblock_rhs_norm),
                target=float(target_xblock),
                preconditioner=precond_xblock_krylov,
                x0=x0_full,
                tol=float(tol),
                atol=float(atol),
                elapsed_s=sparse_timer.elapsed_s,
                matvec_count=lambda: int(mv_count),
                emit=emit,
                gmres_solver=gmres_solve_with_history_scipy,
            )
        )
        x0_full = xblock_side_probe_stage.x0
        precondition_side = xblock_side_probe_stage.precondition_side
        xblock_krylov_method = xblock_side_probe_stage.krylov_method
        pc_maxiter = xblock_side_probe_stage.pc_maxiter
        xblock_side_probe_enabled = bool(xblock_side_probe_stage.enabled)
        xblock_side_probe_used = bool(xblock_side_probe_stage.used)
        xblock_side_probe_switched = bool(xblock_side_probe_stage.switched)
        xblock_side_probe_initial_side = xblock_side_probe_stage.initial_side
        xblock_side_probe_selected_side = xblock_side_probe_stage.selected_side
        xblock_side_probe_initial_method = xblock_side_probe_stage.initial_method
        xblock_side_probe_selected_method = xblock_side_probe_stage.selected_method
        xblock_side_probe_lgmres_rescue = bool(xblock_side_probe_stage.lgmres_rescue)
        xblock_lgmres_rescue_maxiter_capped = bool(
            xblock_side_probe_stage.lgmres_rescue_maxiter_capped
        )
        xblock_lgmres_rescue_outer_k = xblock_side_probe_stage.lgmres_rescue_outer_k
        xblock_side_probe_residual_norm = xblock_side_probe_stage.residual_norm
        xblock_side_probe_residual_ratio = xblock_side_probe_stage.residual_ratio
        xblock_side_probe_iterations = int(xblock_side_probe_stage.iterations)
        xblock_side_probe_matvecs = int(xblock_side_probe_stage.matvecs)
        xblock_side_probe_s = float(xblock_side_probe_stage.elapsed_s)
        xblock_side_probe_switch_suppressed_by_global_coupling = bool(
            xblock_side_probe_stage.switch_suppressed_by_global_coupling
        )
        xblock_side_probe_switch_suppressed_by_explicit_side = bool(
            xblock_side_probe_stage.switch_suppressed_by_explicit_side
        )
        xblock_side_probe_physical_seed_preserved_after_switch = bool(
            xblock_side_probe_stage.physical_seed_preserved_after_switch
        )
        xblock_side_probe_seed_used = bool(xblock_side_probe_stage.seed_used)
        xblock_side_probe_seed_residual_norm = (
            xblock_side_probe_stage.seed_residual_norm
        )

        if precondition_side != "none":
            if xblock_use_active_dof:

                def _coarse_preconditioner_for_basis(v_full: jnp.ndarray) -> jnp.ndarray:
                    reduced = _xblock_reduce_full(jnp.asarray(v_full, dtype=jnp.float64))
                    return _xblock_expand_reduced(precond_xblock_krylov(reduced))

            else:
                _coarse_preconditioner_for_basis = precond_xblock_krylov
        else:

            def _coarse_preconditioner_for_basis(v_full: jnp.ndarray) -> jnp.ndarray:
                return jnp.asarray(v_full, dtype=jnp.float64)

        def _xblock_coarse_direction_builder(
            residual_vec: jnp.ndarray,
            *,
            include_raw: bool,
            fsavg_lmax: int,
            angular_lmax: int,
            max_extra_units: int,
            max_directions: int,
            include_angular_residual: bool,
        ) -> tuple[tuple[str, jnp.ndarray], ...]:
            residual_for_basis = (
                _xblock_expand_reduced(jnp.asarray(residual_vec, dtype=jnp.float64))
                if xblock_use_active_dof
                else jnp.asarray(residual_vec, dtype=jnp.float64)
            )
            return _rhs1_xblock_post_coarse_directions(
                op=op,
                residual=residual_for_basis,
                preconditioner=_coarse_preconditioner_for_basis,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                include_raw=bool(include_raw),
                fsavg_lmax=int(fsavg_lmax),
                angular_lmax=int(angular_lmax),
                max_extra_units=int(max_extra_units),
                max_directions=int(max_directions),
                include_angular_residual=bool(include_angular_residual),
            )

        probe_coarse_policy = _read_rhs1_probe_coarse_policy()
        probe_coarse_stage = apply_xblock_probe_coarse_stage(
            XBlockProbeCoarseStageContext(
                policy=probe_coarse_policy,
                rhs=xblock_rhs,
                x0=x0_full,
                matvec=_mv_true,
                target=float(target_xblock),
                direction_builder=_xblock_coarse_direction_builder,
                correction=_apply_subspace_minres_correction,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        x0_full = probe_coarse_stage.x0
        probe_coarse_steps_requested = int(probe_coarse_stage.steps_requested)
        probe_coarse_max_directions = int(probe_coarse_stage.max_directions)
        probe_coarse_max_extra_units = int(probe_coarse_stage.max_extra_units)
        probe_coarse_fsavg_lmax = int(probe_coarse_stage.fsavg_lmax)
        probe_coarse_angular_lmax = int(probe_coarse_stage.angular_lmax)
        probe_coarse_include_angular_residual = bool(
            probe_coarse_stage.include_angular_residual
        )
        probe_coarse_include_raw = bool(probe_coarse_stage.include_raw)
        probe_coarse_alpha_clip = float(probe_coarse_stage.alpha_clip)
        probe_coarse_rcond = float(probe_coarse_stage.rcond)
        probe_coarse_min_improvement = float(probe_coarse_stage.min_improvement)
        probe_coarse_s = float(probe_coarse_stage.elapsed_s)
        probe_coarse_history = probe_coarse_stage.history
        probe_coarse_direction_counts = probe_coarse_stage.direction_counts
        probe_coarse_direction_names = probe_coarse_stage.direction_names
        probe_coarse_residual_before = probe_coarse_stage.residual_before
        probe_coarse_residual_after = probe_coarse_stage.residual_after
        probe_coarse_seed_initialized = bool(probe_coarse_stage.seed_initialized)

        preflight_min_improvement = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_MIN_IMPROVEMENT",
            default=0.0,
            minimum=0.0,
        )
        preflight_required = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_REQUIRED",
            default=False,
        )
        preflight_gate = evaluate_xblock_preflight_gate(
            XBlockPreflightGateContext(
                min_improvement=float(preflight_min_improvement),
                required=bool(preflight_required),
                rhs=xblock_rhs,
                rhs_norm=float(xblock_rhs_norm),
                x0=x0_full,
                matvec=_mv_true_no_count,
                target=float(target_xblock),
                emit=emit,
            )
        )
        preflight_residual_norm = preflight_gate.residual_norm
        preflight_improvement = preflight_gate.improvement
        preflight_passed = preflight_gate.passed

        xblock_krylov_controls = resolve_xblock_krylov_control_setup(
            XBlockKrylovControlSetupContext(
                env=os.environ,
                krylov_method=str(xblock_krylov_method),
                pc_restart=int(pc_restart),
                pc_maxiter=pc_maxiter,
                precondition_side=str(precondition_side),
                emit=emit,
            )
        )
        fgmres_block_between_cycles = bool(
            xblock_krylov_controls.fgmres_block_between_cycles
        )
        tfqmr_replacement_interval = int(
            xblock_krylov_controls.tfqmr_replacement_interval
        )
        xblock_device_fgmres_jit = bool(
            xblock_krylov_controls.device_fgmres_jit
        )
        xblock_device_fgmres_jit_mode = (
            xblock_krylov_controls.device_fgmres_jit_mode
        )
        xblock_device_fgmres_jit_outer_k = int(
            xblock_krylov_controls.device_fgmres_jit_outer_k
        )
        qi_device_augmented_krylov_requested = bool(
            xblock_krylov_controls.qi_device_augmented_krylov_requested
        )
        qi_device_augmented_krylov_mode = (
            xblock_krylov_controls.qi_device_augmented_krylov_mode
        )
        solve_matvec = _mv_xblock_krylov
        solve_rhs = xblock_rhs
        solve_preconditioner = precond_xblock_krylov if precondition_side != "none" else None
        solve_x0 = x0_full
        solve_space = prepare_xblock_krylov_solve_space(
            XBlockKrylovSolveSpaceContext(
                matvec=solve_matvec,
                rhs=solve_rhs,
                preconditioner=solve_preconditioner,
                x0=solve_x0,
                precondition_side=str(precondition_side),
                row_equilibration_built=bool(xblock_row_equilibration_built),
                col_equilibration_built=bool(xblock_col_equilibration_built),
                row_scale=xblock_row_scale_jnp,
                inv_row_scale=xblock_inv_row_scale_jnp,
                col_scale=xblock_col_scale_jnp,
                inv_col_scale=xblock_inv_col_scale_jnp,
            )
        )
        solve_matvec = solve_space.matvec
        solve_rhs = solve_space.rhs
        solve_preconditioner = solve_space.preconditioner
        solve_x0 = solve_space.x0
        solve_solution_to_physical = solve_space.solution_to_physical
        if emit is not None and solve_space.transform_label is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"using {solve_space.transform_label}-equilibrated assembled operator for Krylov solve",
            )
        augmentation_basis_for_solve = None
        operator_on_augmentation_for_solve = None
        augmented_krylov_stage = apply_xblock_augmented_krylov_stage(
            XBlockAugmentedKrylovStageContext(
                requested=bool(qi_device_augmented_krylov_requested),
                krylov_method=str(xblock_krylov_method),
                qi_device_state=qi_device_state_for_augmented_krylov,
                seed_available=bool(qi_device_augmented_seed_available),
                seed_rank=int(qi_device_augmented_seed_rank),
                seed_basis=qi_device_augmented_seed_basis_for_krylov,
                seed_operator_on_basis=qi_device_augmented_seed_action_for_krylov,
                seed_used=bool(qi_device_augmented_seed_used),
                row_equilibration_built=bool(xblock_row_equilibration_built),
                col_equilibration_built=bool(xblock_col_equilibration_built),
                row_scale=xblock_row_scale_jnp,
                inv_col_scale=xblock_inv_col_scale_jnp,
                precondition_side=str(precondition_side),
                solve_preconditioner=solve_preconditioner,
                mode=str(qi_device_augmented_krylov_mode),
                metadata=qi_device_preconditioner_metadata,
                emit=emit,
                basis_builder=prepare_xblock_augmented_krylov_basis,
            )
        )
        augmentation_basis_for_solve = augmented_krylov_stage.basis
        operator_on_augmentation_for_solve = (
            augmented_krylov_stage.operator_on_basis
        )
        qi_device_augmented_krylov_used = bool(augmented_krylov_stage.used)
        qi_device_augmented_krylov_rank = int(augmented_krylov_stage.rank)
        qi_device_augmented_krylov_reason = augmented_krylov_stage.reason
        qi_device_augmented_seed_used = bool(augmented_krylov_stage.seed_used)
        qi_device_preconditioner_metadata = augmented_krylov_stage.metadata
        solve_start_s = sparse_timer.elapsed_s()
        progress_callbacks = build_xblock_krylov_progress_callbacks(
            XBlockKrylovProgressCallbacksContext(
                emit=emit,
                elapsed_s=sparse_timer.elapsed_s,
                progress_every=int(progress_every),
            )
        )

        fallback_to_gmres = _rhs1_xblock_policy.rhs1_xblock_fallback_to_gmres_enabled(
            env_value=os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_FALLBACK_GMRES", ""),
            xblock_side_probe_lgmres_rescue=bool(xblock_side_probe_lgmres_rescue),
            xblock_krylov_method=str(xblock_krylov_method),
        )
        krylov_stage = run_xblock_krylov_solve_stage(
            XBlockKrylovSolveStageContext(
                first_attempt=XBlockFirstKrylovAttemptContext(
                    krylov_method=str(xblock_krylov_method),
                    matvec=solve_matvec,
                    rhs=solve_rhs,
                    preconditioner=solve_preconditioner,
                    x0=solve_x0,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(pc_restart),
                    maxiter=pc_maxiter,
                    precondition_side=str(precondition_side),
                    lgmres_outer_k=xblock_lgmres_rescue_outer_k,
                    fgmres_block_between_cycles=bool(fgmres_block_between_cycles),
                    skip_inactive_work=not bool(two_level_built),
                    device_fgmres_jit=bool(xblock_device_fgmres_jit),
                    device_fgmres_jit_mode=str(xblock_device_fgmres_jit_mode),
                    device_fgmres_jit_outer_k=int(xblock_device_fgmres_jit_outer_k),
                    augmented_krylov_used=bool(qi_device_augmented_krylov_used),
                    augmentation_basis=augmentation_basis_for_solve,
                    operator_on_augmentation=operator_on_augmentation_for_solve,
                    augmentation_mode=str(qi_device_augmented_krylov_mode),
                    tfqmr_replacement_interval=int(tfqmr_replacement_interval),
                    mv_count=int(mv_count),
                    host_progress_callback=progress_callbacks.host_progress_callback,
                    device_cycle_progress_callback=(
                        progress_callbacks.device_cycle_progress_callback
                    ),
                    gmres_solver=gmres_solve_with_history_scipy,
                    lgmres_solver=lgmres_solve_with_history_scipy,
                    gcrotmk_solver=gcrotmk_solve_with_history_scipy,
                    bicgstab_solver=bicgstab_solve_with_history_scipy,
                    fgmres_solver=fgmres_solve_with_residual,
                    fgmres_jit_solver=fgmres_solve_with_residual_jit,
                    fgmres_cycle_jit_solver=fgmres_cycle_jit_solve_with_residual,
                    bicgstab_jax_solver=bicgstab_solve_with_residual,
                    tfqmr_jax_solver=tfqmr_solve_with_residual,
                ),
                solve_start_s=float(solve_start_s),
                side_probe_s=float(xblock_side_probe_s),
                probe_coarse_s=float(probe_coarse_s),
                elapsed_s=sparse_timer.elapsed_s,
                solution_to_physical=solve_solution_to_physical,
                physical_rhs=xblock_rhs,
                physical_matvec=_mv_true,
                target=float(target_xblock),
                rhs_norm=float(xblock_rhs_norm),
                fallback_enabled=bool(fallback_to_gmres),
                progress_callback=progress_callbacks.host_progress_callback,
                emit=emit,
                initial_guess_builder=_rhs1_xblock_fallback_initial_guess,
            )
        )
        candidate_state = krylov_stage.candidate_state
        candidate_krylov_method = str(candidate_state.krylov_method)
        candidate_residual_norm = float(candidate_state.residual_norm)
        candidate_iterations = int(candidate_state.reported_iterations)
        candidate_matvecs = int(candidate_state.reported_matvecs)
        solve_state = krylov_stage.final_state
        xblock_krylov_method = str(solve_state.krylov_method)
        x_solution_np = solve_state.x_solution
        x_physical_np = solve_state.x_physical
        residual_norm_xblock_pc = float(solve_state.residual_norm)
        history = solve_state.history
        solve_s = float(solve_state.solve_s)
        device_krylov_iterations = solve_state.device_iterations
        device_krylov_estimated_matvecs = solve_state.device_estimated_matvecs
        fallback_started_from_candidate = solve_state.fallback_started_from_candidate
        fallback_candidate_improved_rhs = solve_state.fallback_candidate_improved_rhs
        reported_iterations = int(solve_state.reported_iterations)
        reported_matvecs = int(solve_state.reported_matvecs)
        x_np = solve_state.x_physical
        post_completion = complete_xblock_post_krylov_stage(
            XBlockPostKrylovCompletionContext(
                corrections=XBlockPostSolveCorrectionContext(
                    matvec=_mv_true,
                    rhs=xblock_rhs,
                    x=np.asarray(x_np, dtype=np.float64),
                    residual_norm=float(residual_norm_xblock_pc),
                    target=float(target_xblock),
                    solve_s=float(solve_s),
                    preconditioner=precond_xblock_krylov,
                    precondition_side=str(precondition_side),
                    post_solve_policy=_read_rhs1_post_solve_correction_policy(),
                    qi_device_state=qi_device_state_for_augmented_krylov,
                    coarse_direction_builder=_xblock_coarse_direction_builder,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    minres_correction=_apply_preconditioned_minres_correction,
                    residual_equation_correction=(
                        _apply_device_subspace_residual_equation_correction
                    ),
                    coarse_correction=_apply_subspace_minres_correction,
                ),
                krylov_method=str(xblock_krylov_method),
                elapsed_s=sparse_timer.elapsed_s,
                iterations=int(reported_iterations),
                matvecs=int(reported_matvecs),
                target=float(target_xblock),
                history=history,
            )
        )
        post_corrections = post_completion.corrections
        x_np = np.asarray(post_completion.x, dtype=np.float64)
        residual_norm_xblock_pc = float(post_completion.residual_norm)
        solve_s = float(post_completion.solve_s)
        xblock_final_driver_state = {**qi_pipeline.diagnostic_scope(), **locals()}
        xblock_final_metadata_state = (
            xblock_sparse_pc_final_metadata_state_from_driver_scope(
                xblock_final_driver_state
            )
        )
        xblock_sparse_pc_final_payload = (
            xblock_sparse_pc_final_payload_from_driver_state(
                {
                    **xblock_final_metadata_state,
                    "op": op,
                    "x_np": np.asarray(x_np, dtype=np.float64),
                    "residual_norm_xblock_pc": float(residual_norm_xblock_pc),
                    "target_xblock": float(target_xblock),
                    "xblock_krylov_method": str(xblock_krylov_method),
                    "xblock_linear_size": int(xblock_linear_size),
                    "pc_restart": int(pc_restart),
                },
                expand_reduced=_xblock_expand_reduced,
                post_corrections=post_corrections,
            )
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=xblock_sparse_pc_final_payload,
        )
    return None


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


















def _elapsed_since_now() -> Callable[[], float]:
    """Return a cheap elapsed-time callback for explicit host sparse branches."""

    start_s = perf_counter()
    return lambda: perf_counter() - start_s




































__all__ = [
    "FortranReducedSparsePCBackendSetup",
    "FortranReducedXBlockBackendContext",
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
    "SparsePCDirectTailFactorSetupContext",
    "SparsePCDirectTailFactorSetupResult",
    "SparsePCGenericBranchSetupContext",
    "SparsePCGenericBranchSetupResult",
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
    "build_sparse_pc_direct_tail_factor_setup",
    "build_sparse_pc_generic_branch_setup",
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
    "run_xblock_sparse_pc_branch",
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
    "solve_fortran_reduced_xblock_backend",
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
