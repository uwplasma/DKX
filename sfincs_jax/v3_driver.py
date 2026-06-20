from __future__ import annotations

from dataclasses import replace
import atexit
import contextlib
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import time

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from collections.abc import Callable, Sequence
from typing import Any
import os
import concurrent.futures
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

from .namelist import Namelist, read_sfincs_input
from .newton_krylov_diagnostics import emit_newton_krylov_ksp_history as _emit_newton_krylov_ksp_history
from .solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_history_scipy,
    dense_krylov_solve_from_matrix_with_residual,
    dense_solve_from_matrix,
    dense_solve_from_matrix_row_scaled,
    gmres_solve,
    gmres_solve_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_jit,
    gmres_solve_distributed,
    gmres_solve_with_residual_distributed,
    distributed_gmres_enabled,
    explicit_left_preconditioned_gmres_scipy,
    fgmres_cycle_jit_solve_with_residual,
    fgmres_solve_with_residual,
    fgmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy,
    gcrotmk_solve_with_history_scipy,
    lgmres_solve_with_history_scipy,
    tfqmr_solve_with_residual,
)
from .linear_algebra import (
    recycled_initial_guess as _recycled_initial_guess,
    small_regularized_lstsq as _small_regularized_lstsq,
)
from .constraint_projection import (
    project_constraint_scheme1_nullspace_solution as _project_constraint_scheme1_nullspace_solution,
    project_constraint_scheme1_nullspace_solution_with_residual as _project_constraint_scheme1_nullspace_solution_with_residual,
)
from .structured_velocity import factor_block_tridiagonal
from .pas_smoother import adaptive_pas_smoother
from .explicit_sparse import (
    SparseDecision,
    SparseOperatorBundle,
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    build_operator_from_matvec,
    build_operator_from_pattern,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
    estimate_multifrontal_direct_lu_nbytes,
    factorize_host_sparse_operator,
    wrap_sparse_factor_with_coarse_correction,
)
from .explicit_sparse_factor_builder import (
    build_host_sparse_direct_factor_from_matvec as _build_host_sparse_direct_factor_from_matvec_impl,
)
from .explicit_sparse_factor_policy import (
    explicit_sparse_monolithic_max_size as _explicit_sparse_monolithic_max_size,
)
from .rhs1_device_operator import device_csr_from_matrix, validate_device_csr_matvec
from .rhs1_domain_decomposition import (  # compatibility exports for legacy tests/debug scripts
    _dd_core_patch_ranges,
    _rhs1_dd_auto_block_size,
    _rhs1_dd_coarse_block_size,
    _rhs1_dd_coarse_block_sizes,
    _rhs1_dd_coarse_level_count,
)
from .rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    apply_rhs1_qi_coarse_correction,
    build_rhs1_xblock_global_coarse_basis as _rhs1_xblock_global_coarse_basis,
    build_rhs1_xblock_global_coupling_load_basis as _rhs1_xblock_global_coupling_load_basis,
    build_rhs1_xblock_qi_coarse_basis as _rhs1_xblock_qi_coarse_basis,
    build_rhs1_xblock_smoothed_load_qi_basis as _rhs1_xblock_smoothed_load_qi_basis,
    build_rhs1_qi_galerkin_preconditioner,
    orthonormalize_rhs1_qi_coarse_basis,
    rhs1_xblock_qi_block_geometry_metadata,
)
from .rhs1_qi_galerkin_policy import (
    parse_rhs1_qi_galerkin_dampings,
    parse_rhs1_qi_galerkin_modes,
)
from .rhs1_qi_deflation import (
    build_rhs1_qi_residual_deflated_preconditioner,
    probe_rhs1_qi_deflated_correction,
    probe_rhs1_qi_deflated_minres_seed,
)
from .rhs1_qi_device_preconditioner import (
    probe_rhs1_qi_device_augmented_seed,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)
from .rhs1_qi_two_level import (
    build_rhs1_qi_two_level_preconditioner,
    build_rhs1_xblock_device_global_coupling_preconditioner as _build_rhs1_xblock_device_global_coupling_preconditioner,
    build_rhs1_xblock_smoothed_global_coupling_preconditioner as _build_rhs1_xblock_smoothed_global_coupling_preconditioner,
    build_rhs1_xblock_two_level_preconditioner as _build_rhs1_xblock_two_level_preconditioner,
)
from .memory_model import estimate_sparse_pc_memory
from .rhs1_pas_policy import (
    build_pas_tz_memory_fallback,
    estimate_rhs1_pas_tz_build_bytes as _estimate_rhs1_pas_tz_build_bytes,
    pas_tokamak_theta_preconditioner_applicable as _pas_tokamak_theta_preconditioner_applicable,
    pas_tz_preconditioner_applicable as _pas_tz_preconditioner_applicable,
    pas_tz_preconditioner_memory_safe as _pas_tz_preconditioner_memory_safe,
    resolve_pas_tz_guarded_correction_kind,
    rhs1_pas_adaptive_smoother_allowed as _rhs1_pas_adaptive_smoother_allowed_impl,
    rhs1_pas_adaptive_smoother_controls_from_env,
    rhs1_pas_default_preconditioner_kind as _rhs1_pas_default_preconditioner_kind,
    rhs1_pas_force_full_decision_from_env,
    rhs1_pas_preconditioner_probe_admitted as _rhs1_pas_preconditioner_probe_admitted,
    rhs1_pas_preconditioner_probe_config_from_env as _rhs1_pas_preconditioner_probe_config_from_env,
    rhs1_pas_preconditioner_probe_large_collision_skip as _rhs1_pas_preconditioner_probe_large_collision_skip,
    rhs1_pas_preconditioner_probe_uses_collision as _rhs1_pas_preconditioner_probe_uses_collision,
    rhs1_pas_schur_rescue_controls_from_env,
    rhs1_pas_tz_guarded_strong_retry_from_env,
    rhs1_pas_tz_max_bytes as _rhs1_pas_tz_max_bytes,
)
from .rhs1_preconditioner_dispatch import (
    RHS1PreconditionerDispatchBuilders,
    build_rhs1_preconditioner_from_kind as _dispatch_rhs1_preconditioner_from_kind,
)
from .rhs1_fblock_assembly import select_structured_rhs1_fblock_csr_operator, select_structured_rhs1_fblock_operator
from .rhs1_full_assembly import (
    build_active_projected_rhs1_full_csr_preconditioner,
    build_direct_active_fortran_v3_reduced_pmat_preconditioner,
    select_active_fortran_v3_reduced_support_mode_preconditioner,
    solve_structured_rhs1_full_csr,
)
from .rhs1_structured_full_csr import _try_build_structured_rhs1_full_csr_operator_bundle
from .rhs1_preconditioner_auto_policy import (
    canonical_rhs1_preconditioner_kind as _canonical_rhs1_preconditioner_kind,
    pas_auto_skip_strong_retry as _pas_auto_skip_strong_retry,
    rhs1_fp_dkes_default_kind as _rhs1_fp_dkes_default_kind,
    rhs1_fp_dkes_env_preconditioner_kind as _rhs1_fp_dkes_env_preconditioner_kind,
    rhs1_geometry4_pas_memory_pas_tz_preferred as _rhs1_geometry4_pas_memory_pas_tz_preferred,
    rhs1_large_fp_near_zero_er_override_kind as _rhs1_large_fp_near_zero_er_override_kind,
    rhs1_pas_auto_large_base_kind as _rhs1_pas_auto_large_base_kind,
    rhs1_pas_dkes_pas_tz_preferred as _rhs1_pas_dkes_pas_tz_preferred,
    rhs1_pas_dkes_xblock_allowed as _rhs1_pas_dkes_xblock_allowed,
    rhs1_pas_family_refinement_kind as _rhs1_pas_family_refinement_kind,
    rhs1_pas_full_pas_tz_preferred as _rhs1_pas_full_pas_tz_preferred,
    rhs1_pas_tokamak_cpu_xblock_preferred as _rhs1_pas_tokamak_cpu_xblock_preferred,
    rhs1_pas_tokamak_gpu_theta_allowed as _rhs1_pas_tokamak_gpu_theta_allowed,
    rhs1_pas_tokamak_gpu_tight_tol as _rhs1_pas_tokamak_gpu_tight_tol,
    rhs1_pas_tokamak_gpu_xblock_preferred as _rhs1_pas_tokamak_gpu_xblock_preferred,
    rhs1_pas_weak_auto_override_kind as _rhs1_pas_weak_auto_override_kind,
    rhs1_sharded_line_override_allowed as _rhs1_sharded_line_override_allowed,
)
from .rhs1_preconditioner_auto_policy import (
    rhs1_gpu_sparse_fallback_skip_allowed as _rhs1_gpu_sparse_fallback_skip_allowed_impl,
)
from .rhs1_schur_policy import resolve_rhs1_schur_base_kind
from .problems.profile_response.handoff import (
    RHS1KSPReplayState,
    rhs1_accept_candidate_and_update_replay,
    rhs1_accept_measured_candidate_and_update_replay,
    rhs1_accept_sparse_retry_candidate_and_update_replay,
    rhs1_accept_smoother_candidate_and_update_replay,
    rhs1_run_fast_post_xblock_polish,
    rhs1_run_linear_candidate_and_update_replay,
    rhs1_run_measured_linear_candidate_and_update_replay,
)
from .problems.profile_response.auto_solve import (
    RHS1AutoHostSolveContext,
    RHS1SparseHostSafeSolveContext,
    RHS1StructuredCSRSolveContext,
    solve_rhs1_structured_full_csr_explicit,
    try_rhs1_auto_host_solve,
    try_rhs1_sparse_host_safe_solve,
)
from .problems.profile_response.dense import (
    HostDenseFullSolveContext,
    HostDenseReducedSolveContext,
    RHS1ReducedDenseFallbackCandidateContext,
    rhs1_dense_probe_admission,
    rhs1_dense_probe_enabled_from_env,
    rhs1_dense_fallback_thresholds_from_env,
    rhs1_dense_probe_shortcut_decision,
    rhs1_dense_shortcut_setup_from_env,
    rhs1_fp_preconditioner_probe_kind_from_env,
    solve_host_dense_full,
    solve_host_dense_reduced,
    solve_rhs1_reduced_dense_fallback_candidate,
)
from .problems.profile_response.linear_solve import (
    ProfileLinearSolveContext,
    RHS1ScipyRescueContext,
    profile_solver_kind,
    rhs1_small_gmres_max_from_env,
    run_rhs1_scipy_rescue,
    solve_profile_linear,
    solve_profile_linear_with_residual,
)
from .problems.profile_response.preconditioner_build import (
    RHS1FullPreconditionerBuildContext,
    RHS1ReducedPreconditionerBuildContext,
    build_rhs1_full_preconditioner,
    build_rhs1_reduced_preconditioner_with_fallback,
)
from .problems.profile_response.qi_device_seed import (
    MatrixFreeQIDeviceSeedContext,
    attempt_matrixfree_qi_device_seed,
)
from .problems.profile_response.diagnostics import (
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    SparseRescueTailMetadataContext,
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockCoarseCorrectionDiagnosticsContext,
    XBlockQIDeflatedPreconditionerDiagnosticsContext,
    XBlockQIDevicePreconditionerDiagnosticsContext,
    XBlockQISeedPreconditionerDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    sparse_pc_factor_preflight_result_metadata_from_context,
    sparse_pc_gmres_static_metadata_from_context,
    sparse_pc_pattern_result_metadata_from_context,
    sparse_rescue_tail_metadata_from_context,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics_from_context,
    xblock_qi_deflated_preconditioner_diagnostics_from_context,
    xblock_qi_device_preconditioner_diagnostics_from_context,
    xblock_qi_seed_preconditioner_diagnostics_from_context,
    xblock_side_probe_diagnostics,
)
from .problems.profile_response.sparse_pc import (
    DirectTailMaterializationContext,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredBuildContext,
    DirectTailSupportModePreflightContext,
    FortranReducedXBlockFactorBuildContext,
    FortranReducedXBlockFinalPayloadContext,
    FortranReducedXBlockGlobalCouplingStageContext,
    FortranReducedXBlockKrylovSetupContext,
    FortranReducedXBlockKrylovSolveContext,
    FortranReducedXBlockMomentSchurStageContext,
    SparsePCMemoryBudgetPreflightContext,
    SparsePCFactorPreflightPolicyContext,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCResidualCandidateAcceptanceContext,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCAutoPreflightRetryEvaluationContext,
    SparsePCPatternSetupContext,
    SparseHostOrILUFactorBuildContext,
    SparseILUPreconditionerBuildContext,
    SparseHostScipyPreconditionerBuildContext,
    SparseHostScipyGMRESContext,
    SparseJAXRetryPreconditionerBuildContext,
    SparsePCFactorDtypeRetryFinalizationContext,
    SparsePCDirectTailFinalMetadataContext,
    SparsePCGMRESContext,
    SparsePCGMRESFinalizationContext,
    SparsePCGMRESFinalizationStateContext,
    SparsePCPostMinresFinalizationContext,
    SparsePCGMRESResult,
    XBlockAugmentedKrylovStageContext,
    XBlockFirstKrylovAttemptContext,
    XBlockFirstKrylovSolveStateContext,
    XBlockGMRESFallbackContext,
    XBlockGlobalCouplingStageContext,
    XBlockKrylovControlSetupContext,
    XBlockKrylovSolveSpaceContext,
    XBlockMomentSchurStageContext,
    XBlockPostSolveCorrectionContext,
    XBlockPreflightGateContext,
    XBlockProbeCoarseStageContext,
    XBlockQICoarseSeedStageContext,
    XBlockQIDeviceMetadataContext,
    XBlockQIDeviceSetupConfigContext,
    XBlockQIDeflatedStageContext,
    XBlockQIGalerkinStageContext,
    XBlockQITwoLevelStageContext,
    XBlockSideProbeStageContext,
    XBlockSparsePCCompletionContext,
    XBlockSparsePCFinalCoreState,
    XBlockSparsePCFinalDeviceState,
    XBlockSparsePCFinalMetadataStateContext,
    XBlockSparsePCFinalNestedMetadata,
    XBlockSparsePCFinalPayloadContext,
    XBlockSparsePCFinalPreflightState,
    XBlockTwoLevelStageContext,
    apply_fortran_reduced_xblock_global_coupling_stage,
    apply_fortran_reduced_xblock_initial_seed,
    apply_fortran_reduced_xblock_moment_schur_stage,
    apply_xblock_global_coupling_stage,
    apply_xblock_augmented_krylov_stage,
    apply_xblock_moment_schur_stage,
    apply_xblock_probe_coarse_stage,
    apply_xblock_qi_coarse_seed_stage,
    apply_xblock_qi_deflated_stage,
    apply_xblock_qi_galerkin_stage,
    apply_xblock_qi_two_level_stage,
    apply_xblock_side_probe_stage,
    apply_xblock_two_level_stage,
    build_fortran_reduced_xblock_factor_stage,
    build_fortran_reduced_xblock_krylov_setup,
    build_xblock_local_preconditioner,
    build_sparse_pc_pattern_setup,
    build_xblock_krylov_matvec_setup,
    build_direct_tail_structured_preconditioner_setup,
    build_xblock_assembled_operator_if_requested,
    build_sparse_pc_active_dof_setup,
    build_direct_tail_materialization_setup,
    build_xblock_qi_device_preconditioner_metadata,
    build_xblock_qi_device_setup_config,
    fortran_reduced_xblock_final_payload,
    xblock_sparse_pc_final_payload as build_xblock_sparse_pc_final_payload,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    evaluate_xblock_preflight_gate,
    select_sparse_pc_auto_preflight_retry_candidates,
    evaluate_sparse_pc_auto_preflight_retry,
    emit_xblock_sparse_pc_completion,
    resolve_sparse_pc_gmres_control_policy,
    enforce_sparse_pc_memory_budget,
    prepare_fortran_reduced_xblock_initial_guess,
    prepare_xblock_initial_guess,
    resolve_sparse_pc_entry_policy,
    resolve_sparse_pc_factor_policy,
    resolve_sparse_pc_factor_preflight_policy,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    resolve_fortran_reduced_sparse_pc_backend,
    run_direct_tail_support_mode_preflight,
    resolve_fortran_reduced_xblock_factor_policy,
    resolve_fortran_reduced_xblock_global_coupling_policy,
    resolve_fortran_reduced_xblock_initial_seed_policy,
    resolve_fortran_reduced_xblock_moment_schur_policy,
    prepare_xblock_augmented_krylov_basis,
    prepare_xblock_krylov_solve_space,
    resolve_xblock_qi_device_admission_setup,
    resolve_xblock_qi_device_base_config_setup,
    resolve_xblock_qi_device_enrichment_config_setup,
    resolve_xblock_qi_device_multilevel_config_setup,
    resolve_xblock_qi_deflated_policy_setup,
    resolve_xblock_krylov_control_setup,
    resolve_xblock_qi_galerkin_policy_setup,
    resolve_xblock_qi_seed_policy_setup,
    resolve_xblock_qi_two_level_policy_setup,
    resolve_xblock_global_coupling_policy_setup,
    resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup,
    resolve_xblock_sparse_pc_branch_setup,
    resolve_xblock_two_level_policy_setup,
    run_fortran_reduced_xblock_krylov_solve,
    run_sparse_pc_gmres_once,
    run_xblock_first_krylov_attempt,
    run_xblock_gmres_fallback_if_needed,
    run_xblock_post_solve_corrections,
    xblock_device_cycle_progress_message,
    xblock_krylov_state_from_first_attempt,
    xblock_krylov_state_from_gmres_fallback,
    xblock_host_krylov_progress_message,
    xblock_sparse_pc_final_metadata_state_from_context,
    build_sparse_host_or_ilu_factor,
    build_sparse_ilu_preconditioner_from_cache,
    build_sparse_host_scipy_preconditioner,
    run_sparse_host_scipy_gmres,
    build_sparse_jax_retry_preconditioner,
    resolve_sparse_host_or_ilu_factor_controls,
    finalize_sparse_pc_gmres_with_dtype_retry,
    sparse_pc_direct_tail_final_metadata,
    sparse_pc_gmres_finalization_state_from_context,
    sparse_host_direct_fallback_payload,
    sparse_host_direct_solve_from_pattern,
    sparse_minimum_norm_solve_from_pattern,
    validate_explicit_sparse_host_request,
    finalize_xblock_assembled_operator_metadata,
)
from .rhs1_strong_fallback import (
    build_rhs1_strong_preconditioner_full_from_kind,
    build_rhs1_strong_preconditioner_reduced_from_kind,
)
from .problems.profile_response.strong_preconditioning import (
    adjust_rhs1_reduced_auto_kind,
    adjust_rhs1_pas_schur_strong_kind_from_env,
    adjust_rhs1_theta_line_auto_kind,
    auto_rhs1_full_strong_kind,
    auto_rhs1_reduced_strong_kind,
    requested_rhs1_strong_preconditioner_kind,
    rhs1_collision_retry_allowed,
    rhs1_fp_strong_size_guard_from_env,
    rhs1_pas_force_strong_ratio_from_env,
    rhs1_pas_tz_guarded_minres_controls_from_env,
    rhs1_pas_weak_minres_controls_from_env,
    rhs1_pas_weak_minres_steps,
    rhs1_pas_weak_strong_retry_skip,
    rhs1_resolved_strong_preconditioner_control,
    rhs1_strong_preconditioner_env_from_env,
    rhs1_strong_retry_controls_from_env,
    rhs1_strong_trigger_controls_from_env,
)
from .problems.profile_response.policies import (
    parse_rhs1_pas_tz_guarded_structured_levels as _rhs1_pas_tz_guarded_structured_levels,
    rhs1_qi_device_coupled_install_on_reject_requested as _rhs1_qi_device_coupled_install_on_reject_requested,
    rhs1_qi_device_extra_coarse_controls as _rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_extra_coarse_metadata as _rhs1_qi_device_extra_coarse_metadata,
    rhs1_qi_device_extra_coarse_setup_kwargs as _rhs1_qi_device_extra_coarse_setup_kwargs,
    rhs1_qi_device_probe_uses_minres_step as _rhs1_qi_device_probe_uses_minres_step,
    rhs1_qi_device_progress_messages as _rhs1_qi_device_progress_messages,
    rhs1_qi_device_rank_budget as _rhs1_qi_device_rank_budget,
    rhs1_qi_device_residual_correction_controls as _rhs1_qi_device_residual_correction_controls,
    rhs1_qi_device_residual_correction_metadata as _rhs1_qi_device_residual_correction_metadata,
    rhs1_qi_device_residual_correction_setup_kwargs as _rhs1_qi_device_residual_correction_setup_kwargs,
    rhs1_qi_device_setup_summary as _rhs1_qi_device_setup_summary,
    rhs1_qi_device_status_fields as _rhs1_qi_device_status_fields,
    rhs1_qi_device_tail_block_required,
    rhs1_sparse_jax_config_from_env,
    rhs1_sparse_operator_admission,
    rhs1_sparse_preconditioner_config_from_env,
    rhs1_sparse_rescue_initial_messages,
    rhs1_sparse_rescue_policy_setup,
    rhs1_sparse_rescue_tail_skip_messages,
    rhs1_xblock_fallback_initial_guess as _rhs1_xblock_fallback_initial_guess,
)

_rhs1_xblock_qi_block_geometry_metadata = rhs1_xblock_qi_block_geometry_metadata
_rhs1_qi_device_tail_block_required = rhs1_qi_device_tail_block_required

from .problems.profile_response.setup import (
    SPARSE_HOST_DIRECT_SOLVE_METHODS as _SPARSE_HOST_DIRECT_SOLVE_METHODS,
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS as _SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS,
    SPARSE_HOST_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_SAFE_SOLVE_METHODS as _SPARSE_HOST_SAFE_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
    STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS as _STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS,
    equilibrium_name_hint_from_namelist,
    geometry_scheme_hint_from_namelist,
    resolve_rhs1_active_problem_setup,
    resolve_rhs1_domain_decomposition_setup,
    resolve_rhs1_gmres_budget_setup,
    resolve_rhs1_initial_route_setup,
    resolve_rhs1_post_active_solve_policy_setup,
    resolve_rhs1_recycle_basis_setup,
    resolve_rhs1_reduced_mode_shape_setup,
    resolve_rhs1_tolerance_setup,
)
from . import rhs1_xblock_policy as _rhs1_xblock_policy
from . import rhs1_xblock_sparse_host_policy as _rhs1_xblock_sparse_host_policy
from .rhs1_xblock_policy import (
    resolve_rhs1_xblock_sparse_pc_policy,
)
from .solvers.preconditioners.pas import (
    RHS1PasCompositeBuilders,
    build_rhs1_pas_hybrid_preconditioner,
    build_rhs1_pas_lite_preconditioner,
    build_rhs1_pas_schur_preconditioner,
    build_rhs1_pas_tokamak_theta_preconditioner,
    build_rhs1_pas_tz_preconditioner,
    build_rhs1_pas_xblock_ilu_preconditioner,
    compose_preconditioners as _compose_preconditioners,
)
from .solvers.preconditioners.full_fp import (
    build_rhs1_block_preconditioner,
    build_rhs1_block_preconditioner_xdiag,
    build_rhs1_collision_preconditioner,
    build_rhs1_species_block_preconditioner,
    build_rhs1_species_xblock_preconditioner,
    build_rhs1_structured_fblock_angular_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner,
    build_rhs1_structured_fblock_fp_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner,
    build_rhs1_structured_fblock_jacobi_preconditioner,
    build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner,
)
from .solvers.preconditioners.xblock import (
    assemble_rhsmode1_fp_xblock_tz_sparse_matrix as _assemble_rhsmode1_fp_xblock_tz_sparse_matrix,
    assemble_selected_theta_tz_operator as _assemble_selected_theta_tz_operator,
    assemble_selected_zeta_tz_operator as _assemble_selected_zeta_tz_operator,
    build_rhs1_sxblock_tz_preconditioner,
    build_rhs1_sxblock_tz_sparse_host_preconditioner,
    build_rhs1_xmg_preconditioner,
    build_rhs1_xupwind_preconditioner,
    build_rhs1_xblock_tz_lmax_preconditioner,
    build_rhs1_xblock_tz_preconditioner,
    build_rhs1_xblock_tz_sparse_preconditioner,
    compute_rhs1_sxblock_tz_sparse_host_seed,
    get_rhsmode1_fp_xblock_assembled_host_cache as _get_rhsmode1_fp_xblock_assembled_host_cache,
    rhsmode1_fp_xblock_assembled_host_allowed as _rhsmode1_fp_xblock_assembled_host_allowed,
    rhsmode1_fp_xblock_species_decoupled_for_host_assembly as _rhsmode1_fp_xblock_species_decoupled_for_host_assembly,
    rhsmode1_fp_xblock_tz_sparse_diagonal as _rhsmode1_fp_xblock_tz_sparse_diagonal,
    rhsmode1_host_factor_probe_ok as _rhsmode1_host_factor_probe_ok,
    rhsmode1_precond_cache_key as _rhsmode1_precond_cache_key,
    rhsmode1_xblock_sparse_lu_default_max as _rhsmode1_xblock_sparse_lu_default_max,
    safe_inverse_diagonal_np as _safe_inverse_diagonal_np,
)
from .solvers.preconditioners.schur import (
    RHS1SchurPreconditionerBuilders,
    build_rhs1_schur_preconditioner,
)
from .solvers.preconditioners.transport_matrix import (
    build_rhsmode23_block_preconditioner,
    build_rhsmode23_collision_preconditioner,
    build_rhsmode23_fp_local_geom_line_preconditioner,
    build_rhsmode23_fp_structured_fblock_lu_preconditioner,
    build_rhsmode23_fp_tzfft_line_preconditioner,
    build_rhsmode23_fp_tzfft_line_schur_preconditioner,
    build_rhsmode23_fp_tzfft_preconditioner,
    build_rhsmode23_fp_xblock_tz_lu_preconditioner,
    build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner,
    build_rhsmode23_sxblock_preconditioner,
    build_rhsmode23_tzfft_preconditioner,
    build_rhsmode23_xmg_preconditioner,
)
from .solvers.preconditioners.domain_decomposition import (
    build_rhs1_theta_dd_preconditioner,
    build_rhs1_theta_line_preconditioner,
    build_rhs1_theta_schwarz_preconditioner,
    build_rhs1_theta_line_xdiag_preconditioner,
    build_rhs1_theta_zeta_preconditioner,
    build_rhs1_zeta_dd_preconditioner,
    build_rhs1_zeta_line_preconditioner,
    build_rhs1_zeta_schwarz_preconditioner,
)
from .problems.profile_response.policies import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
)
from .rhs1_solver_policy import (
    read_bool_env as _rhs1_bool_env,
    read_float_env as _rhs1_float_env,
    read_int_env as _rhs1_int_env,
    read_post_solve_correction_policy as _read_rhs1_post_solve_correction_policy,
    read_probe_coarse_policy as _read_rhs1_probe_coarse_policy,
)
from .rhs1_direct_tail_policy import (
    _DIRECT_TAIL_STRUCTURED_PC_CACHE,
    _StructuredHostSparsePreconditionerBundle,
    _direct_tail_structured_pc_cache_key,
    _direct_tail_structured_pc_with_cache_metadata,
    _hash_numpy_array_for_cache,
    _is_direct_reduced_pmat_pc_kind,
    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb,
)
from .rhs1_fortran_reduced_direct_tail import _try_build_fortran_reduced_constraint1_direct_tail_bundle
from .rhs1_true_operator_rescue import (
    _ResidualCoarseHostSparsePreconditionerBundle,
    _ResidualWindowHostSparsePreconditionerBundle,
    _ReusableTrueActionColumnCache,
    _TrueOperatorActiveSubmatrixPreconditionerBundle,
    _TrueOperatorCoupledCoarseLSQPreconditionerBundle,
    _TrueOperatorWindowLSQPreconditionerBundle,
    _expand_sparse_graph_positions,
    _parse_true_operator_window_specs,
    _rhs1_additive_rescue_nbytes,
    _rhs1_active_reduced_residual_diagnostics,
    _sparse_factor_nbytes_estimate,
    _true_operator_window_positions_from_residual,
    _try_build_true_operator_active_block_lsq_preconditioner,
    _try_build_true_operator_active_residual_block_lsq_preconditioner,
    _try_build_true_operator_active_submatrix_preconditioner,
    _try_build_true_operator_coupled_coarse_lsq_preconditioner,
    _try_build_true_operator_residual_window_lsq_preconditioner,
    _try_build_residual_coarse_host_sparse_preconditioner,
    _try_build_residual_window_host_sparse_preconditioner,
)
from .rhs1_ksp_diagnostics import (
    rhs1_fortran_stdout_from_env,
    rhs1_ksp_diagnostics_controls_from_env,
    rhs1_ksp_history_limits_from_env,
)
from .problems.profile_response.solver_diagnostics import (
    RHS1KSPDiagnosticsContext,
    emit_profile_response_ksp_history,
    emit_profile_response_ksp_iter_stats,
)
from .problems.profile_response.active_dof import (
    build_rhs1_active_dof_state as _build_rhs1_active_dof_state_compat,
)
from .rhs1_compressed_layout import build_rhs1_compressed_pitch_layout
from .problems.profile_response.active_projection import (
    expand_reduced_with_map,
    fp_pitch_mode_active_indices,
    project_pas_constraint_f,
    reduce_full_with_indices,
)
from .rhs1_block_operator import (
    RHS1ActiveBlockLayout,
    RHS1ActiveFieldSplitOrdering,
    RHS1BlockLayout,
)
from .rhs1_lowmode_coarse import (
    _rhs1_cap_lowmode_features,
    _rhs1_low_legendre_index_features,
    _rhs1_lowmode_angular_features,
    _rhs1_polynomial_moment_features,
)
from .problems.profile_response.residual import (
    apply_damped_preconditioned_residual_polish as _apply_damped_preconditioned_residual_polish,
    apply_device_subspace_residual_equation_correction as _apply_device_subspace_residual_equation_correction,
    apply_preconditioned_minres_correction as _apply_preconditioned_minres_correction,
    apply_projected_residual_polish as _apply_projected_residual_polish,
    apply_subspace_minres_correction as _apply_subspace_minres_correction,
    build_rhs1_xblock_post_coarse_directions as _rhs1_xblock_post_coarse_directions,
    compose_multilevel_minres_correction_preconditioner as _compose_multilevel_minres_correction_preconditioner,
    compose_multilevel_residual_correction_preconditioner as _compose_multilevel_residual_correction_preconditioner,
    compose_residual_correction_preconditioner as _compose_residual_correction_preconditioner,
    l2_norm_float as rhs1_l2_norm_float,
    recompute_true_residual_result as rhs1_recompute_true_residual_result,
    replay_left_preconditioned_residual_norms as rhs1_replay_left_preconditioned_residual_norms,
    residual_converged as rhs1_residual_converged,
    residual_target as rhs1_residual_target,
    result_with_true_residual as rhs1_result_with_true_residual,
    safe_preconditioner as _safe_preconditioner,
    safe_ratio as rhs1_safe_ratio,
    true_residual_norm_or_inf as rhs1_true_residual_norm_or_inf,
)
from .problems.profile_response.policies import (
    rhs1_constraint0_dense_fallback_allowed as _rhs1_constraint0_dense_fallback_allowed_impl,
    rhs1_constraint0_petsc_compat as _rhs1_constraint0_petsc_compat_impl,
    rhs1_constraint0_petsc_compat_config_from_env,
    rhs1_constraint0_petsc_compat_regularization,
    rhs1_constraint0_sparse_first as _rhs1_constraint0_sparse_first_impl,
)
from .rhs1_constraint_sources import (
    build_rhs1_xblock_constraint1_moment_schur_preconditioner as _build_rhs1_xblock_constraint1_moment_schur_preconditioner,
    constraint_scheme1_inject_source as _constraint_scheme1_inject_source,
    constraint_scheme1_moments_from_f as _constraint_scheme1_moments_from_f,
    constraint_scheme2_inject_source as _constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f as _constraint_scheme2_source_from_f,
)
from .problems.profile_response.policies import (
    rhs1_prefer_sparse_over_dense_shortcut as _rhs1_prefer_sparse_over_dense_shortcut_impl,
    rhs1_sparse_exact_lu_requested as _rhs1_sparse_exact_lu_requested_impl,
    rhs1_sparse_prefer_skips_stage2 as _rhs1_sparse_prefer_skips_stage2_impl,
)
from .rhs1_large_cpu_policy import (
    rhs1_large_cpu_sparse_exact_lu_allowed as _rhs1_large_cpu_sparse_exact_lu_allowed_impl,
    rhs1_large_cpu_sparse_exact_lu_xblock_allowed as _rhs1_large_cpu_sparse_exact_lu_xblock_allowed_impl,
    rhs1_large_cpu_sparse_rescue_allowed as _rhs1_large_cpu_sparse_rescue_allowed_impl,
    rhs1_large_cpu_sparse_rescue_first as _rhs1_large_cpu_sparse_rescue_first_impl,
    rhs1_large_cpu_sparse_skip_primary_allowed as _rhs1_large_cpu_sparse_skip_primary_allowed_impl,
    rhs1_large_cpu_xblock_skip_primary_allowed as _rhs1_large_cpu_xblock_skip_primary_allowed_impl,
    rhs1_sparse_sxblock_rescue_allowed as _rhs1_sparse_sxblock_rescue_allowed_impl,
    rhs1_sparse_xblock_rescue_allowed as _rhs1_sparse_xblock_rescue_allowed_impl,
)
from .problems.profile_response.policies import (
    rhs1_bicgstab_fallback_controls_from_env,
    rhs1_bicgstab_fallback_target_from_env,
    rhs1_fast_post_xblock_polish_allowed as _rhs1_fast_post_xblock_polish_allowed_impl,
    rhs1_fast_post_xblock_polish_controls_from_env,
    rhs1_fp_bicgstab_polish_controls_from_env,
    rhs1_fp_global_low_l_polish_controls_from_env,
    rhs1_fp_l1_polish_controls_from_env,
    rhs1_fp_low_l_polish_controls_from_env,
    rhs1_fp_residual_polish_controls_from_env,
    rhs1_fp_xblock_global_correction_allowed as _rhs1_fp_xblock_global_correction_allowed_impl,
    rhs1_fp_targeted_polish_allowed as _rhs1_fp_targeted_polish_allowed_impl,
    rhs1_gmres_precondition_side_from_env,
    rhs1_krylov_routing_controls_from_env,
    rhs1_pas_source_zero_tolerance_from_env,
    rhs1_scipy_rescue_abs_floor_after_xblock as _rhs1_scipy_rescue_abs_floor_after_xblock_impl,
    rhs1_scipy_rescue_active_size_allowed as _rhs1_scipy_rescue_active_size_allowed_impl,
    rhs1_scipy_rescue_controls_from_env,
    rhs1_skip_global_sparse_after_xblock_allowed as _rhs1_skip_global_sparse_after_xblock_allowed_impl,
)
from .problems.profile_response.policies import rhs1_pas_fast_accept as _rhs1_pas_fast_accept_impl
from .problems.profile_response.policies import (
    rhs1_fp_force_stage2,
    rhs1_pas_stage2_skip,
    rhs1_pas_tz_guarded_stage2_retry,
    rhs1_stage2_admission_controls_from_env,
    rhs1_stage2_retry_controls_from_env,
    rhs1_stage2_trigger,
)
from . import solver_path_policy as _solver_path_policy
from .rhs1_host_policy import (
    host_sparse_direct_refine_steps as _host_sparse_direct_refine_steps_impl,
    host_sparse_factor_dtype as _host_sparse_factor_dtype_impl,
    rhs1_dense_auto_fp_allowed as _rhs1_dense_auto_fp_allowed_impl,
    rhs1_dense_auto_fp_cutoff as _rhs1_dense_auto_fp_cutoff_impl,
    rhs1_dense_backend_allowed as _rhs1_dense_backend_allowed_impl,
    rhs1_dense_fallback_max as _rhs1_dense_fallback_max_impl,
    rhs1_dense_krylov_allowed as _rhs1_dense_krylov_allowed_impl,
    rhs1_explicit_sparse_host_direct_allowed as _rhs1_explicit_sparse_host_direct_allowed_impl,
    rhs1_host_dense_fallback_allowed as _rhs1_host_dense_fallback_allowed_impl,
    rhs1_host_dense_shortcut_allowed as _rhs1_host_dense_shortcut_allowed_impl,
    rhs1_host_sparse_direct_allowed as _rhs1_host_sparse_direct_allowed_impl,
    rhs1_host_sparse_skip_dense_ratio as _rhs1_host_sparse_skip_dense_ratio_impl,
    rhs1_sparse_operator_preconditioned_rescue_allowed as _rhs1_sparse_operator_preconditioned_rescue_allowed_impl,
    rhs1_structured_full_csr_auto_allowed as _rhs1_structured_full_csr_auto_allowed_impl,
)
from .host_refinement import (
    host_direct_solve_with_refinement as _host_direct_solve_with_refinement_impl,
    host_sparse_direct_polish as _host_sparse_direct_polish_impl,
    host_sparse_direct_solve_with_refinement as _host_sparse_direct_solve_with_refinement_impl,
)
from .transport_policy import (
    transport_dense_accelerator_auto_allowed as _transport_dense_accelerator_auto_allowed_impl,
    transport_dense_backend_allowed as _transport_dense_backend_allowed_impl,
    transport_disable_auto_recycle as _transport_disable_auto_recycle_impl,
    transport_host_gmres_accepts_preconditioned_residual as _transport_host_gmres_accepts_preconditioned_residual_impl,
    transport_host_gmres_first_attempt_allowed as _transport_host_gmres_first_attempt_allowed_impl,
    transport_precondition_side as _transport_precondition_side_impl,
    transport_sparse_direct_first_attempt_allowed as _transport_sparse_direct_first_attempt_allowed_impl,
    transport_sparse_direct_needs_float64_retry as _transport_sparse_direct_needs_float64_retry_impl,
    transport_sparse_direct_rescue_allowed as _transport_sparse_direct_rescue_allowed_impl,
    transport_sparse_direct_rescue_first as _transport_sparse_direct_rescue_first_impl,
    transport_sparse_direct_use_explicit_helper as _transport_sparse_direct_use_explicit_helper_impl,
    transport_sparse_factor_dtype as _transport_sparse_factor_dtype_impl,
    transport_tzfft_accelerator_auto_allowed as _transport_tzfft_accelerator_auto_allowed_impl,
    transport_tzfft_backend_allowed as _transport_tzfft_backend_allowed_impl,
    transport_tzfft_first_attempt_budget as _transport_tzfft_first_attempt_budget_impl,
    transport_tzfft_structured_first_attempt_allowed as _transport_tzfft_structured_first_attempt_allowed_impl,
)
from .problems.transport_matrix.preconditioner_dispatch import (
    TransportPreconditionerContext,
    TransportPreconditionerDispatchBuilders,
    build_transport_preconditioner_from_kind,
    build_transport_strong_preconditioner_from_kind,
    normalize_transport_preconditioner_kind,
    resolve_transport_precondition_side_for_kind,
    resolve_transport_preconditioner_choice,
    transport_dd_config_from_env,
    transport_sparse_jax_config_from_env,
)
from .problems.transport_matrix.direct_block_schur import build_transport_fp_direct_active_block_schur_preconditioner
from .problems.transport_matrix.fortran_reduced_lu import build_transport_fp_fortran_reduced_lu_preconditioner
from .transport_solve_policy import resolve_transport_per_rhs_loop_policy, transport_geometry_scheme_from_namelist
from .transport_solve_setup import (
    resolve_transport_maxiter_setup,
    resolve_transport_parallel_request,
    resolve_transport_state_setup,
    resolve_transport_which_rhs_setup,
)
from .transport_active_dense_setup import resolve_transport_active_dense_setup
from .transport_handoff_policy import (
    transport_candidate_is_better,
    transport_polish_config_from_env,
    transport_residual_value,
    transport_result_needs_retry,
)
from .transport_dense_lu import (
    dense_preconditioner_for_matvec as _dense_preconditioner_for_matvec,
    dense_solver_for_matvec as _dense_solver_for_matvec,
)
from .transport_dense_batch import (
    TransportDenseBatchContext,
    solve_transport_dense_batch as _solve_transport_dense_batch,
)
from .transport_host_gmres import transport_host_gmres_solve as _transport_host_gmres_solve
from .transport_iteration_stats import emit_transport_ksp_iteration_stats as _emit_transport_ksp_iteration_stats
from .transport_linear_solve import (
    TransportLinearSolveContext,
    solve_transport_linear as _solve_transport_linear,
    solve_transport_linear_with_residual as _solve_transport_linear_with_residual,
    transport_restart_for_method as _transport_restart_for_method,
    transport_solver_kind as _transport_solver_kind,
)
from .transport_solve_finalization import (
    TransportRHSFinalizationContext,
    finalize_full_transport_rhs,
    finalize_reduced_transport_rhs,
)
from .transport_loop_support import (
    TransportLoopProgress,
    TransportMatvecCache,
    TransportRecycleState,
    resolve_transport_recycle_k,
)
from .transport_sparse_direct_solve import (
    TransportSparseDirectContext,
    transport_sparse_direct_pattern_for_solve as _transport_sparse_direct_pattern_for_context,
    transport_sparse_direct_solve as _transport_sparse_direct_solve_with_context,
)
from .problems.transport_matrix.parallel.policy import (
    rewrite_xla_flags as _rewrite_xla_flags,
    transport_parallel_backend as _transport_parallel_backend_impl,
    transport_parallel_gpu_worker_env as _transport_parallel_gpu_worker_env_impl,
    transport_parallel_persistent_pool_enabled as _transport_parallel_persistent_pool_enabled_impl,
    transport_parallel_pool_executor_kwargs as _transport_parallel_pool_executor_kwargs_impl,
    transport_parallel_pool_key as _transport_parallel_pool_key_impl,
    transport_parallel_start_method as _transport_parallel_start_method_impl,
    transport_parallel_visible_gpu_ids as _transport_parallel_visible_gpu_ids_impl,
    transport_parallel_worker_env as _transport_parallel_worker_env_impl,
)
from .problems.transport_matrix.parallel.payload import solve_transport_parallel_payload as _solve_transport_parallel_payload
from .problems.transport_matrix.parallel.runtime import (
    run_transport_parallel_gpu_subprocesses as _run_transport_parallel_gpu_subprocesses_impl,
)
from .problems.transport_matrix.parallel.solve import TransportParallelSolveRuntime, maybe_run_transport_parallel_solve
from .transport_residual_quality import transport_residual_gate_thresholds_from_env
from .problems.transport_matrix.parallel.pool import TransportParallelPoolCache
from .solve_mode_policy import resolve_use_implicit as _resolve_use_implicit_impl
from .phi1_newton_policy import (
    phi1_frozen_jacobian_policy,
    phi1_gmres_restart,
    phi1_line_search_policy,
    phi1_use_active_dof_mode,
)
from .phi1_newton_linear import (
    build_phi1_newton_preconditioner,
    solve_phi1_newton_linear_step,
)
from .phi1_line_search import advance_phi1_newton_iterate
from .solver_progress import (
    RHS1ProgressNotes,
    rhs1_large_progress_enabled,
)
from .problems.transport_matrix.diagnostics import (
    _flux_functions_from_op,
    transport_matrix_size_from_rhs_mode,
)
from .transport_postsolve_diagnostics import compute_transport_postsolve_diagnostics
from .transport_streaming_outputs import TransportStreamingOutputAccumulator
from .solver_runtime import (
    block_gmres_result_ready as _block_gmres_result_ready,
    gmres_result_is_finite as _gmres_result_is_finite,
)
from .matrix_reductions import (  # noqa: F401
    block_diagonal_only as _block_diag_only,
    diagonal_only as _diag_only,
)
from .preconditioner_operators import (
    _build_rhsmode1_preconditioner_operator_fortran_reduced,
    _build_rhsmode1_preconditioner_operator_point,
    _build_rhsmode1_preconditioner_operator_theta_dd,
    _build_rhsmode1_preconditioner_operator_theta_line,
    _build_rhsmode1_preconditioner_operator_zeta_dd,
    _build_rhsmode1_preconditioner_operator_zeta_line,
    _build_transport_preconditioner_operator_fortran_reduced,
    _build_transport_preconditioner_operator_point,
)
from .sparse_triangular import (
    inverse_permutation as _inverse_permutation,
    triangular_solve_lower_csr_rows as _triangular_solve_lower_csr_rows,
    triangular_solve_lower_padded as _triangular_solve_lower_padded,
    triangular_solve_upper_csr_rows as _triangular_solve_upper_csr_rows,
    triangular_solve_upper_padded as _triangular_solve_upper_padded,
)
from .preconditioner_setup import (
    hash_array as _hash_array,
    matvec_submatrix as _matvec_submatrix_impl,
    precond_chunk_cols as _precond_chunk_cols,
    rhs_mode1_precond_cache_key as _rhs_mode1_precond_cache_key_impl,
    rhs_mode1_structured_fblock_cache_key as _rhs_mode1_structured_fblock_cache_key_impl,
    transport_precond_cache_key as _transport_precond_cache_key_impl,
)
from .krylov_dispatch import (
    HOST_SCIPY_KRYLOV_METHODS as _HOST_SCIPY_KRYLOV_METHODS,
    gmres_solve_dispatch as _gmres_solve_dispatch_impl,
    gmres_solve_with_residual_dispatch as _gmres_solve_with_residual_dispatch_impl,
    host_scipy_krylov_requested as _host_scipy_krylov_requested,
    ksp_iteration_solver_label as _ksp_iteration_solver_label,
    resolve_distributed_gmres_axis as _resolve_distributed_gmres_axis_impl,
    rhs_krylov_method_for_context as _rhs_krylov_method_for_context,
    solver_kind_for_label as _solver_kind_for_label,
)
from .preconditioner_caches import (
    _RHSMODE1_PAS_PRECOND_PROBE_CACHE,
    _RHSMODE1_PAS_TOKAMAK_THETA_CACHE,
    _RHSMODE1_PAS_TZ_CACHE,
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_PRECOND_GLOBAL_CACHE,
    _RHSMODE1_PRECOND_IDX_CACHE,
    _RHSMODE1_PRECOND_ILU_CACHE,
    _RHSMODE1_PRECOND_LIST_CACHE,
    _RHSMODE1_SCHUR_CACHE,
    _RHSMODE1_SPARSE_ILU_CACHE,
    _RHSMODE1_SPARSE_JAX_CACHE,
    _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE,
    _RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE,
    _RHSMODE1_THETA_LINE_DIAGX_CACHE,
    _RHSMODE1_XMG_PRECOND_CACHE,
    _RHSMODE1_XUPWIND_PRECOND_CACHE,
    _RHSMODE23_PRECOND_CACHE,
    _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE,
    _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE,
    _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_PRECOND_CACHE,
    _TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE,
    _TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE,
    _TRANSPORT_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_PRECOND_CACHE,
    _TRANSPORT_TZFFT_PRECOND_CACHE,
    _TRANSPORT_XBLOCK_PRECOND_CACHE,
    _TRANSPORT_XMG_PRECOND_CACHE,
    _LowRankXBlockPrecondCache,
    _PasTokamakThetaPrecondCache,
    _PasTzPrecondCache,
    _RHSMode1ILUBlockPrecondCache,
    _RHSMode1PrecondCache,
    _RHSMode1PrecondGlobalCache,
    _RHSMode1PrecondIdxCache,
    _RHSMode1PrecondListCache,
    _RHSMode1SparseSXBlockHostPrecondCache,
    _RHSMode1SparseXBlockCSRPrecondCache,
    _RHSMode1SparseXBlockHostPrecondCache,
    _RHSMode1SparseXBlockPrecondCache,
    _RHSMode1ThetaLineDiagXCache,
    _SparseJaxPrecondCache,
    _TransportFpDirectActiveBlockSchurPrecondCache,
    _TransportFpFortranReducedLuPrecondCache,
    _TransportFpLocalGeomLinePrecondCache,
    _TransportFpStructuredFBlockLuPrecondCache,
    _TransportFpTzFftLinePrecondCache,
    _TransportFpTzFftLineSchurPrecondCache,
    _TransportFpTzFftPrecondCache,
    _TransportFpXBlockTzLuPrecondCache,
    _TransportPrecondCache,
    _TransportTzFftPrecondCache,
    _TransportXBlockPrecondCache,
    _TransportXmgPrecondCache,
    _XUpwindPrecondCache,
)
from .preconditioner_context import (
    auto_pas_geom4_fp32_precond_allowed as _auto_pas_geom4_fp32_precond_allowed,
    precond_dtype as _precond_dtype,
    precond_policy_hints as _precond_policy_hints,
    set_precond_policy_hints as _set_precond_policy_hints,
    set_precond_size_hint as _set_precond_size_hint,
    sparse_structural_tol as _sparse_structural_tol,
    use_solver_jit as _use_solver_jit,
)
from .solvers.preconditioners.symbolic_sparse import (
    RHS1FullSystemMatrixFreeOperatorAdapter as _RHS1FullSystemMatrixFreeOperatorAdapter,
    build_sparse_ilu_from_matvec as _build_sparse_ilu_from_matvec,
    factorize_sparse_matrix_csr_host as _factorize_sparse_matrix_csr_host,
)
from .problems.transport_matrix.direct_pmat import (
    _build_rhsmode23_direct_pmat_physics_coarse_basis,
    _try_build_rhsmode23_fp_direct_active_operator_bundle,
    _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle,
)
from .v3_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1, _matvec_shard_axis, sharding_constraints
from .verbose import Timer
from .v3 import geometry_from_namelist, grids_from_namelist
from .v3_system import (
    V3FullSystemOperator,
    _THRESHOLD_FOR_INCLUSION,
    _operator_signature_cached,
    apply_v3_full_system_jacobian,
    apply_v3_full_system_jacobian_jit,
    apply_v3_full_system_operator,
    apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    rhs_v3_full_system_jit,
    with_transport_rhs_settings,
)
from .v3_results import (
    V3LinearSolveResult,
    V3NewtonKrylovResult,
    V3TransportMatrixSolveResult,
    v3_linear_solve_result_from_payload,
)
from .v3_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary,
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices,
)
from .profiling import _rss_mb, maybe_profiler


_rhs1_xblock_precondition_side = _rhs1_xblock_policy.rhs1_xblock_precondition_side
_rhs1_xblock_gmres_restart = _rhs1_xblock_policy.rhs1_xblock_gmres_restart
build_rhs1_active_dof_state = _build_rhs1_active_dof_state_compat
_rhs1_dkes_gmres_budget = _solver_path_policy.rhs1_dkes_gmres_budget
_rhs1_residual_needs_rescue = _solver_path_policy.rhs1_residual_needs_rescue


def _rhs1_gpu_sparse_fallback_skip_allowed(
    *,
    op: V3FullSystemOperator,
    rhs1_precond_kind: str | None,
    use_active_dof_mode: bool,
    residual_norm: float,
    target: float,
) -> bool:
    return _rhs1_gpu_sparse_fallback_skip_allowed_impl(
        backend=jax.default_backend(),
        rhs_mode=int(getattr(op, "rhs_mode", 0) or 0),
        include_phi1=bool(getattr(op, "include_phi1", False)),
        has_pas=getattr(getattr(op, "fblock", None), "pas", None) is not None,
        rhs1_precond_kind=rhs1_precond_kind,
        use_active_dof_mode=bool(use_active_dof_mode),
        residual_norm=float(residual_norm),
        target=float(target),
    )


_is_resource_exhausted_error = _solver_path_policy.is_resource_exhausted_error
_resolve_use_implicit = _resolve_use_implicit_impl


def _rhsmode1_dense_backend_allowed() -> bool:
    return _rhs1_dense_backend_allowed_impl(backend=jax.default_backend())


def _transport_dense_backend_allowed() -> bool:
    return _transport_dense_backend_allowed_impl(backend=jax.default_backend())


def _transport_dense_accelerator_auto_allowed(
    op: V3FullSystemOperator,
    *,
    geometry_scheme: int,
) -> bool:
    return _transport_dense_accelerator_auto_allowed_impl(
        op,
        backend=jax.default_backend(),
        geometry_scheme=int(geometry_scheme),
    )


def _transport_tzfft_backend_allowed() -> bool:
    return _transport_tzfft_backend_allowed_impl(backend=jax.default_backend())


def _transport_tzfft_accelerator_auto_allowed(op: V3FullSystemOperator) -> bool:
    return _transport_tzfft_accelerator_auto_allowed_impl(op, backend=jax.default_backend())


def _transport_tzfft_structured_first_attempt_allowed(
    op: V3FullSystemOperator,
    *,
    size: int,
    use_implicit: bool,
) -> bool:
    return _transport_tzfft_structured_first_attempt_allowed_impl(
        op,
        size=size,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


_transport_tzfft_first_attempt_budget = _transport_tzfft_first_attempt_budget_impl


def _rhsmode1_host_dense_fallback_allowed() -> bool:
    return _rhs1_host_dense_fallback_allowed_impl(backend=jax.default_backend())


def _rhsmode1_host_dense_shortcut_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
) -> bool:
    return _rhs1_host_dense_shortcut_allowed_impl(
        op=op,
        active_size=active_size,
        use_implicit=use_implicit,
        solve_method_kind=solve_method_kind,
        backend=jax.default_backend(),
        dense_fallback_max=_rhsmode1_dense_fallback_max(op),
    )


_rhsmode1_dense_krylov_allowed = _rhs1_dense_krylov_allowed_impl


_rhsmode1_host_sparse_direct_allowed = _rhs1_host_sparse_direct_allowed_impl


def _rhsmode1_sparse_operator_preconditioned_rescue_allowed(
    *,
    op: V3FullSystemOperator,
    sparse_exact_lu: bool,
    host_sparse_direct_wanted: bool,
) -> bool:
    """Allow a Fortran-like sparse-preconditioned GMRES rescue before direct LU.

    For RHSMode=1 full-FP constraintScheme=1 runs on CPU, exact sparse LU of the
    true Jacobian can converge to a slightly different low-order moment branch than
    PETSc's iterative solve, even when the linear residual is tiny. In this regime,
    first using an exact sparse LU of the simplified preconditioner operator as the
    preconditioner for GMRES on the true Jacobian more closely matches the Fortran
    KSP path while keeping the strong CPU rescue.
    """
    return _rhs1_sparse_operator_preconditioned_rescue_allowed_impl(
        op=op,
        sparse_exact_lu=sparse_exact_lu,
        host_sparse_direct_wanted=host_sparse_direct_wanted,
        backend=jax.default_backend(),
    )


def _host_sparse_factor_dtype(
    *,
    size: int,
    factorization: str,
    use_implicit: bool,
) -> np.dtype:
    return _host_sparse_factor_dtype_impl(
        size=size,
        factorization=factorization,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


def _sparse_factor_cache_key(cache_key: tuple[object, ...], factor_dtype: np.dtype) -> tuple[object, ...]:
    return (*cache_key, np.dtype(factor_dtype).str)


_host_sparse_direct_refine_steps = _host_sparse_direct_refine_steps_impl


_host_sparse_direct_solve_with_refinement = (
    _host_sparse_direct_solve_with_refinement_impl
)


_host_direct_solve_with_refinement = _host_direct_solve_with_refinement_impl


def _host_sparse_direct_polish(
    *,
    matvec_fn,
    rhs_vec: jnp.ndarray,
    x0_np: np.ndarray,
    ilu,
    factor_dtype: np.dtype,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
) -> tuple[np.ndarray, float]:
    return _host_sparse_direct_polish_impl(
        matvec_fn=matvec_fn,
        rhs_vec=rhs_vec,
        x0_np=x0_np,
        ilu=ilu,
        factor_dtype=factor_dtype,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        precondition_side=precondition_side,
        gmres_solver=gmres_solve_with_history_scipy,
    )


def _host_physical_memory_mb() -> float | None:
    """Return physical host memory in MB when the platform exposes it cheaply."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        value = float(pages) * float(page_size) / 1.0e6
        if np.isfinite(value) and value > 0.0:
            return value
    except (AttributeError, OSError, ValueError):
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            value = float(result.stdout.strip()) / 1.0e6
            if np.isfinite(value) and value > 0.0:
                return value
    except Exception:
        pass
    return None


_rhsmode1_host_sparse_skip_dense_ratio = _rhs1_host_sparse_skip_dense_ratio_impl


_rhsmode1_explicit_sparse_host_direct_allowed = (
    _rhs1_explicit_sparse_host_direct_allowed_impl
)


def _build_host_sparse_direct_factor_from_matvec(
    **kwargs,
):
    """Compatibility wrapper around the explicit-sparse factor builder.

    The implementation and its full keyword schema live in
    `explicit_sparse_factor_builder.py`. This thin wrapper exists so older
    driver tests can still monkeypatch the builder dependencies through
    `v3_driver`, without duplicating the entire builder signature here.
    """
    kwargs.setdefault("build_operator_from_matvec_callback", build_operator_from_matvec)
    kwargs.setdefault("build_operator_from_pattern_callback", build_operator_from_pattern)
    kwargs.setdefault(
        "factorize_host_sparse_operator_callback",
        factorize_host_sparse_operator,
    )
    kwargs.setdefault("default_backend_callback", jax.default_backend)
    kwargs.setdefault(
        "monolithic_max_size_callback",
        _explicit_sparse_monolithic_max_size,
    )
    return _build_host_sparse_direct_factor_from_matvec_impl(
        **kwargs,
    )


def _rhsmode1_explicit_sparse_pattern_probe_enabled() -> bool:
    env = os.environ.get("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_PATTERN", "").strip().lower()
    return env in {"1", "true", "yes", "on", "pattern"}


def _rhsmode1_sparse_pc_default_permc_spec(
    *,
    constrained_pas_pc: bool,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> str:
    """Return the measured SuperLU column-ordering default for sparse-PC RHSMode=1.

    Constrained PAS systems usually benefit from ``MMD_ATA``. The measured
    exception kept as a default is the tokamak PAS+Er full-trajectory window,
    where ``MMD_AT_PLUS_A`` lowers fill and runtime on both CPU and one-GPU
    validation runs.
    """
    return _solver_path_policy.rhsmode1_sparse_pc_default_permc_spec(
        constrained_pas_pc=constrained_pas_pc,
        tokamak_pas_er_pc=tokamak_pas_er_pc,
        n_species=n_species,
    )


def _rhsmode1_sparse_pc_default_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> int:
    """Return the sparse-PC GMRES restart after scoped production caps.

    The one-species tokamak PAS+Er production row is memory dominated on GPUs.
    A restart cap of 40 preserved output parity in CPU/GPU sweeps while lowering
    GPU resident memory and slightly reducing time-to-solution. Keep all other
    sparse-PC rows on their requested restart, and always respect an explicit
    user environment override.
    """
    return _solver_path_policy.rhsmode1_sparse_pc_default_restart(
        requested_restart=requested_restart,
        restart_env_value=restart_env_value,
        tokamak_pas_er_pc=tokamak_pas_er_pc,
        n_species=n_species,
    )


def _maybe_rhsmode1_full_sparse_pattern(op: V3FullSystemOperator, emit: Callable[[int, str], None] | None = None):
    if not _rhsmode1_explicit_sparse_pattern_probe_enabled():
        return None
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    if emit is not None:
        summary = summarize_v3_sparse_pattern(op, pattern)
        emit(
            1,
            "explicit_sparse_pattern: "
            f"shape={summary.shape} nnz={summary.nnz} "
            f"avg_row_nnz={summary.avg_row_nnz:.3g} max_row_nnz={summary.max_row_nnz}",
        )
    return pattern


def _rhsmode1_pas_fast_accept(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
) -> bool:
    return _rhs1_pas_fast_accept_impl(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


_rhsmode1_pas_adaptive_smoother_allowed = (
    _rhs1_pas_adaptive_smoother_allowed_impl
)


def _rhsmode1_constraint0_sparse_first(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    return _rhs1_constraint0_sparse_first_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        backend=jax.default_backend(),
    )


_rhsmode1_constraint0_petsc_compat = _rhs1_constraint0_petsc_compat_impl


_rhsmode1_constraint0_dense_fallback_allowed = (
    _rhs1_constraint0_dense_fallback_allowed_impl
)


def _rhsmode1_sparse_exact_lu_requested(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    full_precond_requested: bool = False,
    preconditioner_x: int,
    use_dkes: bool,
) -> bool:
    return _rhs1_sparse_exact_lu_requested_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        full_precond_requested=bool(full_precond_requested),
        preconditioner_x=int(preconditioner_x),
        use_dkes=bool(use_dkes),
        backend=jax.default_backend(),
    )


_rhsmode1_prefer_sparse_over_dense_shortcut = (
    _rhs1_prefer_sparse_over_dense_shortcut_impl
)


_rhsmode1_sparse_prefer_skips_stage2 = _rhs1_sparse_prefer_skips_stage2_impl


def _rhsmode1_large_cpu_sparse_rescue_allowed(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    residual_norm: float,
    target: float,
) -> bool:
    return _rhs1_large_cpu_sparse_rescue_allowed_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_x=int(preconditioner_x),
        residual_norm=float(residual_norm),
        target=float(target),
        backend=jax.default_backend(),
    )


_rhsmode1_large_cpu_sparse_rescue_first = _rhs1_large_cpu_sparse_rescue_first_impl


def _rhsmode1_large_cpu_sparse_skip_primary_allowed(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    use_implicit: bool,
) -> bool:
    return _rhs1_large_cpu_sparse_skip_primary_allowed_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


_rhsmode1_large_cpu_sparse_exact_lu_allowed = (
    _rhs1_large_cpu_sparse_exact_lu_allowed_impl
)


def _rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    preconditioner_x: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    xblock_seed_residual: float,
    xblock_seed_improvement_ratio: float,
    use_implicit: bool,
) -> bool:
    return _rhs1_large_cpu_sparse_exact_lu_xblock_allowed_impl(
        op=op,
        active_size=int(active_size),
        preconditioner_x=int(preconditioner_x),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        xblock_seed_residual=float(xblock_seed_residual),
        xblock_seed_improvement_ratio=float(xblock_seed_improvement_ratio),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_sparse_xblock_rescue_allowed(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    pre_theta: int,
    pre_zeta: int,
    residual_norm: float,
    target: float,
) -> bool:
    return _rhs1_sparse_xblock_rescue_allowed_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_x=int(preconditioner_x),
        pre_theta=int(pre_theta),
        pre_zeta=int(pre_zeta),
        residual_norm=float(residual_norm),
        target=float(target),
        backend=jax.default_backend(),
    )


def _rhsmode1_large_cpu_xblock_skip_primary_allowed(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_species: int,
    preconditioner_x: int,
    preconditioner_xi: int,
    pre_theta: int,
    pre_zeta: int,
    use_implicit: bool,
    rhs1_precond_env: str,
) -> bool:
    return _rhs1_large_cpu_xblock_skip_primary_allowed_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        pre_theta=int(pre_theta),
        pre_zeta=int(pre_zeta),
        use_implicit=bool(use_implicit),
        rhs1_precond_env=rhs1_precond_env,
        backend=jax.default_backend(),
    )


def _rhsmode1_fast_post_xblock_polish_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    use_implicit: bool,
) -> bool:
    return _rhs1_fast_post_xblock_polish_allowed_impl(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_fp_targeted_polish_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    residual_norm: float,
    target: float,
    rhs1_precond_kind: str,
    use_implicit: bool,
) -> bool:
    return _rhs1_fp_targeted_polish_allowed_impl(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        rhs1_precond_kind=rhs1_precond_kind,
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_skip_global_sparse_after_xblock_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> bool:
    return _rhs1_skip_global_sparse_after_xblock_allowed_impl(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_fp_xblock_global_correction_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    sparse_xblock_candidate_accepted: bool,
    use_implicit: bool,
) -> bool:
    return _rhs1_fp_xblock_global_correction_allowed_impl(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        sparse_xblock_candidate_accepted=bool(sparse_xblock_candidate_accepted),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_scipy_rescue_abs_floor_after_xblock(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> float:
    return _rhs1_scipy_rescue_abs_floor_after_xblock_impl(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_scipy_rescue_active_size_allowed(
    *,
    op: V3FullSystemOperator,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> bool:
    return _rhs1_scipy_rescue_active_size_allowed_impl(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _rhsmode1_sparse_sxblock_rescue_allowed(
    *,
    op: V3FullSystemOperator,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    pre_theta: int,
    pre_zeta: int,
    use_implicit: bool,
) -> bool:
    return _rhs1_sparse_sxblock_rescue_allowed_impl(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_x=int(preconditioner_x),
        pre_theta=int(pre_theta),
        pre_zeta=int(pre_zeta),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )


def _transport_sparse_direct_rescue_allowed(
    *,
    op: V3FullSystemOperator,
    size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
) -> bool:
    return _transport_sparse_direct_rescue_allowed_impl(
        op=op,
        size=size,
        residual_norm=residual_norm,
        target=target,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


_transport_sparse_direct_rescue_first = _transport_sparse_direct_rescue_first_impl


def _transport_sparse_direct_first_attempt_allowed(
    *,
    op: V3FullSystemOperator,
    size: int,
    use_implicit: bool,
) -> bool:
    return _transport_sparse_direct_first_attempt_allowed_impl(
        op=op,
        size=size,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


def _transport_host_gmres_first_attempt_allowed(
    *,
    op: V3FullSystemOperator,
    size: int,
    use_implicit: bool,
) -> bool:
    return _transport_host_gmres_first_attempt_allowed_impl(
        op=op,
        size=size,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


_transport_host_gmres_accepts_preconditioned_residual = (
    _transport_host_gmres_accepts_preconditioned_residual_impl
)


_transport_precondition_side = _transport_precondition_side_impl


def _transport_disable_auto_recycle(
    *,
    op: V3FullSystemOperator,
    use_implicit: bool,
) -> bool:
    return _transport_disable_auto_recycle_impl(
        op=op,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )


_transport_sparse_direct_needs_float64_retry = (
    _transport_sparse_direct_needs_float64_retry_impl
)


def _transport_sparse_factor_dtype(*, size: int, use_implicit: bool) -> np.dtype:
    return _transport_sparse_factor_dtype_impl(
        size=size,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
        host_sparse_factor_dtype=_host_sparse_factor_dtype,
    )


def _transport_sparse_direct_use_explicit_helper(*, size: int) -> bool:
    return _transport_sparse_direct_use_explicit_helper_impl(
        size=size,
        backend=jax.default_backend(),
    )


def _transport_host_gmres_progress_every() -> int:
    env = os.environ.get("SFINCS_JAX_TRANSPORT_HOST_GMRES_PROGRESS_EVERY", "").strip()
    try:
        return max(0, int(env)) if env else 10
    except ValueError:
        return 10


def _gmres_solve_dispatch(*, distributed_axis: str | None = None, size_hint: int | None = None, **kwargs):
    return _gmres_solve_dispatch_impl(
        distributed_axis=distributed_axis,
        size_hint=size_hint,
        gmres_solve_fn=gmres_solve,
        gmres_solve_jit_fn=gmres_solve_jit,
        gmres_solve_distributed_fn=gmres_solve_distributed,
        distributed_gmres_enabled_fn=distributed_gmres_enabled,
        use_solver_jit_fn=_use_solver_jit,
        **kwargs,
    )


def _gmres_solve_with_residual_dispatch(*, distributed_axis: str | None = None, size_hint: int | None = None, **kwargs):
    return _gmres_solve_with_residual_dispatch_impl(
        distributed_axis=distributed_axis,
        size_hint=size_hint,
        gmres_solve_with_residual_fn=gmres_solve_with_residual,
        gmres_solve_with_residual_jit_fn=gmres_solve_with_residual_jit,
        gmres_solve_with_residual_distributed_fn=gmres_solve_with_residual_distributed,
        distributed_gmres_enabled_fn=distributed_gmres_enabled,
        use_solver_jit_fn=_use_solver_jit,
        **kwargs,
    )


def _resolve_distributed_gmres_axis(
    *, op: V3FullSystemOperator | None, emit: Callable[[int, str], None] | None = None
) -> str | None:
    return _resolve_distributed_gmres_axis_impl(
        op=op,
        emit=emit,
        matvec_shard_axis_fn=_matvec_shard_axis,
    )



def _rhsmode1_sparse_cache_key(
    op: V3FullSystemOperator,
    *,
    kind: str,
    active_size: int,
    use_active_dof_mode: bool,
    use_pas_projection: bool,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
) -> tuple[object, ...]:
    return (
        *_rhsmode1_precond_cache_key(op, kind),
        int(active_size),
        int(bool(use_active_dof_mode)),
        int(bool(use_pas_projection)),
        float(drop_tol),
        float(drop_rel),
        float(ilu_drop_tol),
        float(fill_factor),
    )

def _build_sparse_jax_preconditioner_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    drop_tol: float,
    drop_rel: float,
    reg: float,
    omega: float,
    sweeps: int,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    cached = _RHSMODE1_SPARSE_JAX_CACHE.get(cache_key)
    if cached is not None:
        a_sp = cached.a_sp
        d_inv = cached.d_inv
        omega = cached.omega
        sweeps = cached.sweeps
    else:
        if emit is not None:
            emit(1, f"sparse_jax: assembling dense operator (n={n})")
        a_dense = assemble_dense_matrix_from_matvec(matvec=matvec, n=int(n), dtype=dtype)
        a_dense = jnp.asarray(a_dense, dtype=dtype)
        max_abs = jnp.max(jnp.abs(a_dense)) if int(n) > 0 else jnp.asarray(0.0, dtype=dtype)
        thresh = jnp.maximum(jnp.asarray(drop_tol, dtype=dtype), jnp.asarray(drop_rel, dtype=dtype) * max_abs)
        if drop_tol > 0.0 or drop_rel > 0.0:
            a_drop = jnp.where(jnp.abs(a_dense) >= thresh, a_dense, jnp.zeros_like(a_dense))
        else:
            a_drop = a_dense
        diag_idx = jnp.arange(int(n), dtype=jnp.int32)
        diag = a_dense[diag_idx, diag_idx]
        diag_safe = diag + jnp.asarray(reg, dtype=dtype)
        a_drop = a_drop.at[diag_idx, diag_idx].set(diag_safe)
        d_inv = jnp.where(diag_safe != 0, 1.0 / diag_safe, jnp.asarray(0.0, dtype=dtype))
        try:
            from jax.experimental import sparse as jsparse  # noqa: PLC0415

            a_sp = jsparse.BCOO.fromdense(a_drop)
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(1, f"sparse_jax: failed to build BCOO ({type(exc).__name__}: {exc})")
            a_sp = None
        if a_sp is None:
            raise RuntimeError("sparse_jax: failed to build sparse operator")
        _RHSMODE1_SPARSE_JAX_CACHE[cache_key] = _SparseJaxPrecondCache(
            a_sp=a_sp,
            d_inv=jnp.asarray(d_inv, dtype=dtype),
            omega=float(omega),
            sweeps=int(sweeps),
        )

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=d_inv.dtype)
        x0 = jnp.zeros_like(v)

        def _body(i, x):
            r = v - a_sp @ x
            return x + omega * d_inv * r

        x = jax.lax.fori_loop(0, int(sweeps), _body, x0)
        return jnp.asarray(x, dtype=jnp.float64)

    return _apply


def _matvec_submatrix(
    op_pc: object,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
) -> np.ndarray:
    """Driver compatibility wrapper for unsharded operator-column probes.

    The production implementation lives in :mod:`sfincs_jax.preconditioner_setup`.
    Keeping this late-bound wrapper lets driver-level tests monkeypatch the
    unsharded operator apply without accidentally entering the cached/sharded
    path inside ``jax.vmap``.
    """

    return _matvec_submatrix_impl(
        op_pc,
        col_idx=col_idx,
        row_idx=row_idx,
        total_size=total_size,
        chunk_cols=chunk_cols,
        apply_operator_fn=apply_v3_full_system_operator,
    )


def _rhsmode1_dense_fallback_max(op: V3FullSystemOperator) -> int:
    return _rhs1_dense_fallback_max_impl(op)
def _rhsmode1_structured_fblock_cache_key(
    op: V3FullSystemOperator,
    kind: str,
    *,
    params: tuple[object, ...] = (),
) -> tuple[object, ...]:
    return _rhs_mode1_structured_fblock_cache_key_impl(
        op,
        kind,
        precond_dtype=_precond_dtype(),
        params=params,
    )


def _transport_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return _transport_precond_cache_key_impl(op, kind, precond_dtype=_precond_dtype())


_build_rhsmode23_collision_preconditioner = build_rhsmode23_collision_preconditioner
_build_rhsmode23_sxblock_preconditioner = build_rhsmode23_sxblock_preconditioner
_build_rhsmode23_xmg_preconditioner = build_rhsmode23_xmg_preconditioner
_build_rhsmode1_xmg_preconditioner = build_rhs1_xmg_preconditioner
_build_rhsmode1_xupwind_preconditioner = build_rhs1_xupwind_preconditioner
_build_rhsmode23_tzfft_preconditioner = build_rhsmode23_tzfft_preconditioner


def _build_rhsmode1_pas_tokamak_theta_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the PAS tokamak theta/L builder."""
    return build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        block_preconditioner_builder=_build_rhsmode1_block_preconditioner,
        pas_tokamak_theta_applicable=_pas_tokamak_theta_preconditioner_applicable,
    )


def _build_rhsmode1_pas_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the PAS theta-zeta/L builder."""
    return build_rhs1_pas_tz_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pas_tz_applicable=_pas_tz_preconditioner_applicable,
        pas_tz_memory_safe=_pas_tz_preconditioner_memory_safe,
        matvec_shard_axis=_matvec_shard_axis,
        device_count=jax.device_count,
        theta_schwarz_builder=_build_rhsmode1_theta_schwarz_preconditioner,
        zeta_schwarz_builder=_build_rhsmode1_zeta_schwarz_preconditioner,
        pas_hybrid_builder=_build_rhsmode1_pas_hybrid_preconditioner,
        collision_builder=_build_rhsmode1_collision_preconditioner,
        tzfft_builder=_build_rhsmode23_tzfft_preconditioner,
    )


_build_rhsmode1_collision_preconditioner = build_rhs1_collision_preconditioner
_build_rhsmode1_block_preconditioner_xdiag = build_rhs1_block_preconditioner_xdiag
_build_rhsmode1_block_preconditioner = build_rhs1_block_preconditioner


def _build_rhsmode23_theta_dd_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Domain-decomposition theta-block preconditioner for RHSMode=2/3 transport solves."""
    return _build_rhsmode1_theta_dd_preconditioner(
        op=op,
        block=int(block),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def _build_rhsmode23_zeta_dd_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Domain-decomposition zeta-block preconditioner for RHSMode=2/3 transport solves."""
    return _build_rhsmode1_zeta_dd_preconditioner(
        op=op,
        block=int(block),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def _build_rhsmode23_theta_schwarz_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    overlap: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Restricted additive Schwarz theta-line preconditioner for transport solves."""
    return _build_rhsmode1_theta_schwarz_preconditioner(
        op=op,
        block=int(block),
        overlap=int(overlap),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def _build_rhsmode23_zeta_schwarz_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    overlap: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Restricted additive Schwarz zeta-line preconditioner for transport solves."""
    return _build_rhsmode1_zeta_schwarz_preconditioner(
        op=op,
        block=int(block),
        overlap=int(overlap),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


_build_rhsmode23_block_preconditioner = build_rhsmode23_block_preconditioner


_build_rhsmode23_fp_tzfft_preconditioner = build_rhsmode23_fp_tzfft_preconditioner
_build_rhsmode23_fp_tzfft_line_preconditioner = (
    build_rhsmode23_fp_tzfft_line_preconditioner
)
_build_rhsmode23_fp_tzfft_line_schur_preconditioner = (
    build_rhsmode23_fp_tzfft_line_schur_preconditioner
)
_build_rhsmode23_fp_local_geom_line_preconditioner = (
    build_rhsmode23_fp_local_geom_line_preconditioner
)
_build_rhsmode23_fp_xblock_tz_lu_preconditioner = (
    build_rhsmode23_fp_xblock_tz_lu_preconditioner
)
_build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner = (
    build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner
)


def _build_rhsmode23_fp_direct_active_block_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return build_transport_fp_direct_active_block_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_indices_np=active_indices_np,
        emit=emit,
        fallback_builder=_build_rhsmode23_sxblock_preconditioner,
        transport_precond_cache_key=_transport_precond_cache_key,
    )


def _build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return build_transport_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_indices_np=active_indices_np,
        emit=emit,
        fallback_builder=_build_rhsmode23_sxblock_preconditioner,
        transport_precond_cache_key=_transport_precond_cache_key,
        build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
        host_physical_memory_mb=_host_physical_memory_mb,
    )


_build_rhsmode23_fp_structured_fblock_lu_preconditioner = (
    build_rhsmode23_fp_structured_fblock_lu_preconditioner
)


def _build_rhsmode1_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the RHSMode=1 constraint-source Schur builder."""
    builders = RHS1SchurPreconditionerBuilders(
        pas_tokamak_theta_applicable=_pas_tokamak_theta_preconditioner_applicable,
        pas_tz_applicable=_pas_tz_preconditioner_applicable,
        theta_line_builder=_build_rhsmode1_theta_line_preconditioner,
        theta_dd_builder=_build_rhsmode1_theta_dd_preconditioner,
        species_block_builder=_build_rhsmode1_species_block_preconditioner,
        sxblock_tz_builder=_build_rhsmode1_sxblock_tz_preconditioner,
        xblock_tz_builder=_build_rhsmode1_xblock_tz_preconditioner,
        xblock_tz_lmax_builder=_build_rhsmode1_xblock_tz_lmax_preconditioner,
        pas_xblock_ilu_builder=_build_rhsmode1_pas_xblock_ilu_preconditioner,
        xmg_builder=_build_rhsmode1_xmg_preconditioner,
        pas_lite_builder=_build_rhsmode1_pas_lite_preconditioner,
        pas_hybrid_builder=_build_rhsmode1_pas_hybrid_preconditioner,
        pas_schur_builder=_build_rhsmode1_pas_schur_preconditioner,
        pas_tokamak_theta_builder=_build_rhsmode1_pas_tokamak_theta_preconditioner,
        pas_tz_builder=_build_rhsmode1_pas_tz_preconditioner,
        theta_zeta_builder=_build_rhsmode1_theta_zeta_preconditioner,
        zeta_line_builder=_build_rhsmode1_zeta_line_preconditioner,
        zeta_dd_builder=_build_rhsmode1_zeta_dd_preconditioner,
        block_builder=_build_rhsmode1_block_preconditioner,
    )
    return build_rhs1_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        builders=builders,
        geom_scheme=int(_precond_policy_hints().geom_scheme or 0),
    )


_build_rhsmode1_theta_line_preconditioner = build_rhs1_theta_line_preconditioner
_build_rhsmode1_theta_dd_preconditioner = build_rhs1_theta_dd_preconditioner
_build_rhsmode1_zeta_dd_preconditioner = build_rhs1_zeta_dd_preconditioner
_build_rhsmode1_theta_schwarz_preconditioner = (
    build_rhs1_theta_schwarz_preconditioner
)
_build_rhsmode1_zeta_schwarz_preconditioner = (
    build_rhs1_zeta_schwarz_preconditioner
)
_build_rhsmode1_theta_line_xdiag_preconditioner = (
    build_rhs1_theta_line_xdiag_preconditioner
)
_build_rhsmode1_theta_zeta_preconditioner = build_rhs1_theta_zeta_preconditioner


def _rhs1_pas_composite_builders() -> RHS1PasCompositeBuilders:
    """Bind current PAS component builders for compatibility wrappers."""

    return RHS1PasCompositeBuilders(
        pas_tokamak_theta_applicable=_pas_tokamak_theta_preconditioner_applicable,
        pas_tz_applicable=_pas_tz_preconditioner_applicable,
        pas_tokamak_theta_builder=_build_rhsmode1_pas_tokamak_theta_preconditioner,
        pas_tz_builder=_build_rhsmode1_pas_tz_preconditioner,
        theta_line_builder=_build_rhsmode1_theta_line_preconditioner,
        zeta_line_builder=_build_rhsmode1_zeta_line_preconditioner,
        xblock_tz_lmax_builder=_build_rhsmode1_xblock_tz_lmax_preconditioner,
        xmg_builder=_build_rhsmode1_xmg_preconditioner,
        xupwind_builder=_build_rhsmode1_xupwind_preconditioner,
        collision_builder=_build_rhsmode1_collision_preconditioner,
    )


def _build_rhsmode1_pas_lite_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the canonical PAS-lite composite builder."""

    return build_rhs1_pas_lite_preconditioner(
        op=op,
        builders=_rhs1_pas_composite_builders(),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        safe=safe,
    )


def _build_rhsmode1_pas_hybrid_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the canonical PAS-hybrid composite builder."""

    return build_rhs1_pas_hybrid_preconditioner(
        op=op,
        builders=_rhs1_pas_composite_builders(),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        safe=safe,
    )


def _build_rhsmode1_pas_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    safe: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the canonical PAS-Schur composite builder."""

    return build_rhs1_pas_schur_preconditioner(
        op=op,
        builders=_rhs1_pas_composite_builders(),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        safe=safe,
    )


_build_rhsmode1_species_block_preconditioner = build_rhs1_species_block_preconditioner
_build_rhsmode1_species_xblock_preconditioner = (
    build_rhs1_species_xblock_preconditioner
)
_build_rhsmode1_xblock_tz_preconditioner = build_rhs1_xblock_tz_preconditioner
_build_rhsmode1_xblock_tz_lmax_preconditioner = (
    build_rhs1_xblock_tz_lmax_preconditioner
)
_build_rhsmode1_xblock_tz_sparse_preconditioner = (
    build_rhs1_xblock_tz_sparse_preconditioner
)
_build_rhsmode1_sxblock_tz_sparse_host_preconditioner = (
    build_rhs1_sxblock_tz_sparse_host_preconditioner
)
_compute_rhsmode1_sxblock_tz_sparse_host_seed = (
    compute_rhs1_sxblock_tz_sparse_host_seed
)
_build_rhsmode1_sxblock_tz_preconditioner = build_rhs1_sxblock_tz_preconditioner


def _build_rhsmode1_pas_xblock_ilu_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return build_rhs1_pas_xblock_ilu_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pas_hybrid_preconditioner=_build_rhsmode1_pas_hybrid_preconditioner,
    )


_build_rhsmode1_zeta_line_preconditioner = build_rhs1_zeta_line_preconditioner
_build_rhsmode1_structured_fblock_jacobi_preconditioner = (
    build_rhs1_structured_fblock_jacobi_preconditioner
)
_build_rhsmode1_structured_fblock_angular_jacobi_preconditioner = (
    build_rhs1_structured_fblock_angular_jacobi_preconditioner
)
_build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner = (
    build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner
)
_build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner = (
    build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner
)
_build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner = (
    build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner
)
_build_rhsmode1_structured_fblock_fp_moment_schur_preconditioner = (
    build_rhs1_structured_fblock_fp_moment_schur_preconditioner
)
_build_rhsmode1_structured_fblock_fp_coupled_moment_schur_preconditioner = (
    build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner
)
_build_rhsmode1_structured_fblock_fp_tail_coupled_schur_preconditioner = (
    build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner
)


def _build_rhs1_preconditioner_from_kind(
    *,
    op: V3FullSystemOperator,
    rhs1_precond_kind: str | None,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    preconditioner_species: int = 1,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    rhs1_xblock_tz_lmax: int | None = None,
    dd_block_theta: int = 8,
    dd_overlap_theta: int = 1,
    dd_block_zeta: int = 8,
    dd_overlap_zeta: int = 1,
    adi_sweeps: int = 2,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Resolve the RHSMode=1 preconditioner builder from the selected kind.

    The actual dispatch ladder lives in ``rhs1_preconditioner_dispatch.py``.
    This wrapper binds the current builder functions at call time so existing
    monkeypatch-based tests and local debug workflows continue to operate on
    ``sfincs_jax.v3_driver`` without behavioral changes.
    """
    return _dispatch_rhs1_preconditioner_from_kind(
        op=op,
        rhs1_precond_kind=rhs1_precond_kind,
        builders=RHS1PreconditionerDispatchBuilders(
            theta_line_builder=_build_rhsmode1_theta_line_preconditioner,
            theta_dd_builder=_build_rhsmode1_theta_dd_preconditioner,
            theta_schwarz_builder=_build_rhsmode1_theta_schwarz_preconditioner,
            theta_line_xdiag_builder=_build_rhsmode1_theta_line_xdiag_preconditioner,
            block_xdiag_builder=_build_rhsmode1_block_preconditioner_xdiag,
            species_block_builder=_build_rhsmode1_species_block_preconditioner,
            sxblock_builder=_build_rhsmode1_species_xblock_preconditioner,
            sxblock_tz_builder=_build_rhsmode1_sxblock_tz_preconditioner,
            xblock_tz_builder=_build_rhsmode1_xblock_tz_preconditioner,
            xblock_tz_lmax_builder=_build_rhsmode1_xblock_tz_lmax_preconditioner,
            theta_zeta_builder=_build_rhsmode1_theta_zeta_preconditioner,
            xmg_builder=_build_rhsmode1_xmg_preconditioner,
            pas_lite_builder=_build_rhsmode1_pas_lite_preconditioner,
            pas_hybrid_builder=_build_rhsmode1_pas_hybrid_preconditioner,
            pas_schur_builder=_build_rhsmode1_pas_schur_preconditioner,
            pas_tz_builder=_build_rhsmode1_pas_tz_preconditioner,
            pas_tzfft_builder=_build_rhsmode23_tzfft_preconditioner,
            pas_tokamak_theta_builder=_build_rhsmode1_pas_tokamak_theta_preconditioner,
            pas_ilu_builder=_build_rhsmode1_pas_xblock_ilu_preconditioner,
            zeta_line_builder=_build_rhsmode1_zeta_line_preconditioner,
            zeta_dd_builder=_build_rhsmode1_zeta_dd_preconditioner,
            zeta_schwarz_builder=_build_rhsmode1_zeta_schwarz_preconditioner,
            schur_builder=_build_rhsmode1_schur_preconditioner,
            collision_builder=_build_rhsmode1_collision_preconditioner,
            structured_fblock_jacobi_builder=_build_rhsmode1_structured_fblock_jacobi_preconditioner,
            structured_fblock_angular_jacobi_builder=_build_rhsmode1_structured_fblock_angular_jacobi_preconditioner,
            structured_fblock_xi_angular_jacobi_builder=_build_rhsmode1_structured_fblock_xi_angular_jacobi_preconditioner,
            structured_fblock_fp_radial_jacobi_builder=_build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner,
            structured_fblock_fp_lowmode_schur_builder=_build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner,
            structured_fblock_fp_moment_schur_builder=_build_rhsmode1_structured_fblock_fp_moment_schur_preconditioner,
            structured_fblock_fp_coupled_moment_schur_builder=_build_rhsmode1_structured_fblock_fp_coupled_moment_schur_preconditioner,
            structured_fblock_fp_tail_coupled_schur_builder=_build_rhsmode1_structured_fblock_fp_tail_coupled_schur_preconditioner,
            block_builder=_build_rhsmode1_block_preconditioner,
            compose_preconditioners=_compose_preconditioners,
        ),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        preconditioner_species=preconditioner_species,
        preconditioner_x=preconditioner_x,
        preconditioner_xi=preconditioner_xi,
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
        dd_block_theta=dd_block_theta,
        dd_overlap_theta=dd_overlap_theta,
        dd_block_zeta=dd_block_zeta,
        dd_overlap_zeta=dd_overlap_zeta,
        adi_sweeps=adi_sweeps,
        emit=emit,
    )


def _build_rhs1_strong_preconditioner_full_from_kind(
    *,
    op: V3FullSystemOperator,
    strong_precond_kind: str | None,
    rhs1_precond_kind: str | None,
    residual_norm: float,
    rhs1_xblock_tz_lmax: int | None = None,
    dd_block_theta: int = 8,
    dd_overlap_theta: int = 1,
    dd_block_zeta: int = 8,
    dd_overlap_zeta: int = 1,
    adi_sweeps: int | None = None,
) -> tuple[str | None, Callable[[jnp.ndarray], jnp.ndarray] | None]:
    """Build the full-system strong fallback preconditioner via shared dispatch."""
    return build_rhs1_strong_preconditioner_full_from_kind(
        op=op,
        strong_precond_kind=strong_precond_kind,
        base_preconditioner_kind=rhs1_precond_kind,
        residual_norm=float(residual_norm),
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
        dd_block_theta=dd_block_theta,
        dd_overlap_theta=dd_overlap_theta,
        dd_block_zeta=dd_block_zeta,
        dd_overlap_zeta=dd_overlap_zeta,
        dispatch_builder=_build_rhs1_preconditioner_from_kind,
        adi_sweeps=adi_sweeps,
    )


def _build_rhs1_strong_preconditioner_reduced_from_kind(
    *,
    op: V3FullSystemOperator,
    strong_precond_kind: str | None,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray],
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray],
    rhs1_xblock_tz_lmax: int | None = None,
    dd_block_theta: int = 8,
    dd_overlap_theta: int = 1,
    dd_block_zeta: int = 8,
    dd_overlap_zeta: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Build the reduced active-DOF strong fallback preconditioner via dispatch."""
    return build_rhs1_strong_preconditioner_reduced_from_kind(
        op=op,
        strong_precond_kind=strong_precond_kind,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
        dd_block_theta=int(dd_block_theta),
        dd_overlap_theta=int(dd_overlap_theta),
        dd_block_zeta=int(dd_block_zeta),
        dd_overlap_zeta=int(dd_overlap_zeta),
        dispatch_builder=_build_rhs1_preconditioner_from_kind,
    )


def solve_v3_full_system_structured_csr(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    op: V3FullSystemOperator | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    max_csr_nbytes: int | None = None,
    method: str = "gmres",
    preconditioner: str | None = "auto",
    preconditioner_max_schur_size: int = 2048,
    preconditioner_max_block_inverse_nbytes: int = 64 * 1024 * 1024,
    active_dof: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> V3LinearSolveResult:
    """Solve a supported RHSMode=1 system with explicit host CSR Krylov.

    This is an opt-in, non-autodiff route for CLI/runtime studies. It assembles
    the supported full-system CSR operator without probing, runs SciPy Krylov on
    the host CSR matrix, and returns the standard v3 solve-result wrapper.
    """

    if op is None:
        op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    if which_rhs is not None:
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
    rhs = rhs_v3_full_system(op)
    active_indices = _transport_active_dof_indices(op) if bool(active_dof) else None
    if emit is not None:
        active_msg = (
            f" active_size={int(active_indices.size)}/{int(op.total_size)}"
            if active_indices is not None
            else " full_size"
        )
        emit(
            0,
            "solve_v3_full_system_structured_csr: assembling no-probe host CSR "
            f"(size={int(op.total_size)}{active_msg} method={method} preconditioner={preconditioner})",
        )
    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        method=method,
        preconditioner=preconditioner,
        preconditioner_max_schur_size=preconditioner_max_schur_size,
        preconditioner_max_block_inverse_nbytes=preconditioner_max_block_inverse_nbytes,
        max_csr_nbytes=max_csr_nbytes,
        active_indices=active_indices,
    )
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_structured_csr: "
            f"converged={bool(result.converged)} residual={float(result.residual_norm):.3e} "
            f"solve_s={float(result.solve_s):.3f}",
        )
        pc_summary = dict(result.metadata.get("preconditioner", {}) or {})
        pc_metadata = dict(pc_summary.get("metadata", {}) or {})
        factor_nbytes = pc_metadata.get("factor_nbytes_actual")
        if factor_nbytes is None:
            factor_nbytes = pc_metadata.get("factor_nbytes_estimate")
        if pc_summary:
            emit(
                0,
                "solve_v3_full_system_structured_csr: "
                f"pc_kind={pc_summary.get('kind', 'unknown')} "
                f"pc_selected={bool(pc_summary.get('selected', False))} "
                f"pc_reason={pc_summary.get('reason', 'unknown')} "
                f"pc_setup_s={float(pc_summary.get('setup_s', 0.0) or 0.0):.3f} "
                f"pc_factor_nbytes={factor_nbytes if factor_nbytes is not None else 'na'} "
                f"pc_permc={pc_metadata.get('permc_spec', 'na')} "
                f"pc_superlu_permc={pc_metadata.get('superlu_permc_spec', 'na')}",
            )
    return V3LinearSolveResult(
        op=op,
        rhs=rhs,
        gmres=GMRESSolveResult(
            x=jnp.asarray(result.x, dtype=jnp.float64),
            residual_norm=jnp.asarray(result.residual_norm, dtype=jnp.float64),
        ),
        metadata={
            "solver_path": "structured_full_csr_host_gmres",
            "structured_full_csr": result.to_dict(),
            "active_dof": bool(active_dof),
        },
    )


def solve_v3_full_system_linear_gmres(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    op: V3FullSystemOperator | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "auto",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    differentiable: bool | None = None,
    emit: Callable[[int, str], None] | None = None,
    recycle_basis: Sequence[jnp.ndarray] | None = None,
) -> V3LinearSolveResult:
    """Solve the current v3 full-system linear problem `A x = rhs` matrix-free using GMRES.

    Notes
    -----
    This helper currently targets the linear runs exercised in the parity fixtures
    (e.g. includePhi1InKineticEquation=false). For nonlinear runs, use `residual_v3_full_system`
    and an outer Newton-Krylov iteration (not yet shipped as a stable API).
    """
    t = Timer()
    profiler = maybe_profiler(emit=emit)

    def _mark(label: str) -> None:
        if profiler is not None:
            profiler.mark(label)
    gmres_budget_setup = resolve_rhs1_gmres_budget_setup(
        restart=int(restart),
        maxiter=maxiter,
        env=os.environ,
    )
    maxiter_env = os.environ.get("SFINCS_JAX_GMRES_MAXITER", "").strip()
    restart = int(gmres_budget_setup.restart)
    maxiter = gmres_budget_setup.maxiter
    restart_env_forced = bool(gmres_budget_setup.restart_env_forced)
    maxiter_env_forced = bool(gmres_budget_setup.maxiter_env_forced)
    geom_scheme_hint = geometry_scheme_hint_from_namelist(nml)
    vmec_operator_timer: Timer | None = None
    if emit is not None:
        emit(1, "solve_v3_full_system_linear_gmres: building operator")
        if geom_scheme_hint == 5:
            eq_name = equilibrium_name_hint_from_namelist(nml)
            emit(1, f"solve_v3_full_system_linear_gmres: VMEC operator build start ({eq_name})")
            vmec_operator_timer = Timer()
    op = (
        full_system_operator_from_namelist(
            nml=nml,
            identity_shift=identity_shift,
            phi1_hat_base=phi1_hat_base,
        )
        if op is None
        else op
    )
    if emit is not None and vmec_operator_timer is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: VMEC operator build done elapsed_s={vmec_operator_timer.elapsed_s():.3f}")
    _mark("operator_built")
    _set_precond_size_hint(int(op.total_size))
    _set_precond_policy_hints(
        geom_scheme=geom_scheme_hint,
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
    )
    tolerance_setup = resolve_rhs1_tolerance_setup(op=op, tol=float(tol), env=os.environ)
    tol = float(tolerance_setup.tol)
    fp_tol = float(tolerance_setup.fp_tol)
    if emit is not None and tolerance_setup.fp_tightened:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: FP tol tightened "
            f"{float(tolerance_setup.fp_previous_tol):.1e} -> {float(tol):.1e}",
        )
    if emit is not None and tolerance_setup.pas_tightened:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: PAS tol tightened "
            f"{float(tolerance_setup.pas_previous_tol):.1e} -> {float(tol):.1e}",
        )
    if int(op.rhs_mode) in {2, 3}:
        # v3 sets (dnHatdpsiHats, dTHatdpsiHats, EParallelHat) internally based on whichRHS.
        # If the input file omits gradients (common for monoenergetic runs), callers must select whichRHS.
        if which_rhs is None:
            which_rhs = 1
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
        if emit is not None:
            emit(1, f"solve_v3_full_system_linear_gmres: applied transport RHS settings whichRHS={int(which_rhs)}")
    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: total_size={int(op.total_size)}")
        emit(1, "solve_v3_full_system_linear_gmres: assembling RHS")
    rhs = rhs_v3_full_system(op)
    _mark("rhs_assembled")
    rhs_norm = jnp.linalg.norm(rhs)
    if emit is not None:
        emit(2, f"solve_v3_full_system_linear_gmres: rhs_norm={float(rhs_norm):.6e}")
    rhs1_progress_notes = RHS1ProgressNotes(
        emit=emit,
        enabled=rhs1_large_progress_enabled(rhs_mode=int(op.rhs_mode), total_size=int(op.total_size)),
    )
    route_setup = resolve_rhs1_initial_route_setup(
        nml=nml,
        op=op,
        solve_method=str(solve_method),
        xblock_active_dof_env=os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", ""),
        use_implicit=bool(_resolve_use_implicit(differentiable=differentiable)),
        force_krylov=bool(_rhs1_bool_env("SFINCS_JAX_RHSMODE1_FORCE_KRYLOV", default=False)),
        sharded_axis=_matvec_shard_axis(op),
        backend=str(jax.default_backend()),
        device_count=int(jax.device_count()),
        structured_auto_allowed=_rhs1_structured_full_csr_auto_allowed_impl,
    )
    method_flags = route_setup.method_flags
    solve_method_kind_requested = method_flags.kind
    sparse_host_like_requested = bool(method_flags.sparse_host_like_requested)
    xblock_active_dof_requested = bool(method_flags.xblock_active_dof_requested)
    structured_full_csr_explicit_requested = bool(method_flags.structured_full_csr_explicit_requested)
    use_implicit_requested = bool(route_setup.use_implicit_requested)
    structured_auto_allowed = bool(route_setup.structured_auto_allowed)
    structured_sharded_multidevice = bool(route_setup.structured_sharded_multidevice)
    auto_host_result = try_rhs1_auto_host_solve(
        RHS1AutoHostSolveContext(
            nml=nml,
            which_rhs=which_rhs,
            op=op,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart),
            maxiter=maxiter,
            solve_method=str(solve_method),
            identity_shift=float(identity_shift),
            phi1_hat_base=phi1_hat_base,
            differentiable=differentiable,
            emit=emit,
            recycle_basis=recycle_basis,
            solve_driver=solve_v3_full_system_linear_gmres,
            solve_method_kind_requested=solve_method_kind_requested,
            structured_full_csr_explicit_requested=bool(structured_full_csr_explicit_requested),
            use_implicit=bool(use_implicit_requested),
            structured_auto_allowed=bool(structured_auto_allowed),
            structured_sharded_multidevice=bool(structured_sharded_multidevice),
        )
    )
    if auto_host_result is not None:
        return auto_host_result
    structured_full_csr_requested = bool(structured_full_csr_explicit_requested)
    if structured_full_csr_requested:
        return solve_rhs1_structured_full_csr_explicit(
            RHS1StructuredCSRSolveContext(
                nml=nml,
                op=op,
                x0=x0,
                rhs_norm=rhs_norm,
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                solve_method=str(solve_method),
                identity_shift=float(identity_shift),
                phi1_hat_base=phi1_hat_base,
                differentiable=differentiable,
                emit=emit,
                structured_solver=solve_v3_full_system_structured_csr,
            )
        )

    recycle_basis_setup = resolve_rhs1_recycle_basis_setup(
        recycle_basis=recycle_basis,
        total_size=int(op.total_size),
        recycle_k_env=os.environ.get("SFINCS_JAX_RHSMODE1_RECYCLE_K", ""),
        asarray=jnp.asarray,
    )
    recycle_basis_use = list(recycle_basis_setup.basis)

    reduced_mode_shape_setup = resolve_rhs1_reduced_mode_shape_setup(
        nxi_for_x=op.fblock.collisionless.n_xi_for_x,
        n_xi=int(op.n_xi),
    )
    nxi_for_x = reduced_mode_shape_setup.nxi_for_x
    max_l = int(reduced_mode_shape_setup.max_l)
    has_reduced_modes = bool(reduced_mode_shape_setup.has_reduced_modes)
    precond_opts = nml.group("preconditionerOptions")
    active_problem_setup = resolve_rhs1_active_problem_setup(
        nml=nml,
        op=op,
        tol=float(tol),
        fp_tol=float(fp_tol),
        restart=int(restart),
        maxiter=maxiter,
        restart_env_forced=bool(restart_env_forced),
        maxiter_env_forced=bool(maxiter_env_forced),
        has_reduced_modes=bool(has_reduced_modes),
        sparse_host_like_requested=bool(sparse_host_like_requested),
        xblock_active_dof_requested=bool(xblock_active_dof_requested),
        dkes_gmres_budget=_rhs1_dkes_gmres_budget,
        active_dof_indices=_transport_active_dof_indices,
        env=os.environ,
    )
    tol = float(active_problem_setup.tol)
    restart = int(active_problem_setup.restart)
    maxiter = active_problem_setup.maxiter
    use_dkes = bool(active_problem_setup.use_dkes)
    include_xdot_sparse_pc = bool(active_problem_setup.include_xdot_sparse_pc)
    include_electric_field_xi_sparse_pc = bool(active_problem_setup.include_electric_field_xi_sparse_pc)
    er_abs_sparse_pc = float(active_problem_setup.er_abs_sparse_pc)
    preconditioner_species = int(active_problem_setup.preconditioner_species)
    preconditioner_x = int(active_problem_setup.preconditioner_x)
    preconditioner_x_min_l = int(active_problem_setup.preconditioner_x_min_l)
    preconditioner_xi = int(active_problem_setup.preconditioner_xi)
    full_precond_requested = bool(active_problem_setup.full_preconditioner_requested)
    geom_scheme = int(active_problem_setup.geom_scheme)
    use_pas_projection = bool(active_problem_setup.use_pas_projection)
    use_active_dof_mode = bool(active_problem_setup.use_active_dof_mode)
    active_idx_jnp = active_problem_setup.active_idx_jnp
    full_to_active_jnp = active_problem_setup.full_to_active_jnp
    active_size = int(active_problem_setup.active_size)
    if emit is not None:
        for level, message in active_problem_setup.messages:
            emit(int(level), str(message))
    pas_tz_guarded_correction_metadata: dict[str, object] = {}
    rhsmode1_general_metadata: dict[str, object] = {}

    def _record_structured_fblock_preconditioner_metadata(precond: Callable[[jnp.ndarray], jnp.ndarray]) -> None:
        metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata", None)
        if not isinstance(metadata, dict):
            return
        assembly = metadata.get("assembly", {})
        if not isinstance(assembly, dict):
            assembly = {}
        rhsmode1_general_metadata.update(
            {
                "structured_fblock_preconditioner_enabled": True,
                "structured_fblock_preconditioner_selected": bool(metadata.get("selected", False)),
                "structured_fblock_preconditioner_reason": str(metadata.get("reason", "")),
                "structured_fblock_preconditioner_nnz_blocks": int(assembly.get("nnz_blocks", 0) or 0),
                "structured_fblock_preconditioner_data_nbytes": int(assembly.get("data_nbytes", 0) or 0),
                "structured_fblock_preconditioner_metadata": metadata,
            }
        )

    cpu_large_xblock_shortcut = False
    explicit_fp_xblock_seed_used = False
    if use_active_dof_mode and emit is not None:
        if use_pas_projection:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: PAS constraint projection enabled "
                f"(size={active_size}/{int(op.total_size)})",
            )
        else:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: active-DOF mode enabled "
                f"(size={active_size}/{int(op.total_size)})",
            )

    post_active_solve_policy_setup = resolve_rhs1_post_active_solve_policy_setup(
        op=op,
        restart=int(restart),
        maxiter=maxiter,
        solve_method=str(solve_method),
        active_size=int(active_size),
        use_active_dof_mode=bool(use_active_dof_mode),
        full_precond_requested=bool(full_precond_requested),
        geom_scheme=int(geom_scheme),
        dense_backend_allowed=bool(_rhsmode1_dense_backend_allowed()),
        backend=str(jax.default_backend()),
        sharded_axis_hint=_matvec_shard_axis(op),
        device_count=int(jax.device_count()),
        env=os.environ,
    )
    restart = int(post_active_solve_policy_setup.restart)
    maxiter = post_active_solve_policy_setup.maxiter
    solve_method = str(post_active_solve_policy_setup.solve_method)
    tokamak_pas = bool(post_active_solve_policy_setup.tokamak_pas)
    pas_large_bicgstab_fastpath = bool(post_active_solve_policy_setup.pas_large_bicgstab_fastpath)
    pas_large_fastpath_min = int(post_active_solve_policy_setup.pas_large_fastpath_min)
    if emit is not None:
        for level, message in post_active_solve_policy_setup.messages:
            emit(int(level), str(message))
    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}")
        emit(1, "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)")
    solve_method_kind_explicit = str(solve_method).strip().lower().replace("-", "_")
    sparse_host_safe_result = try_rhs1_sparse_host_safe_solve(
        RHS1SparseHostSafeSolveContext(
            nml=nml,
            which_rhs=which_rhs,
            op=op,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart),
            maxiter=maxiter,
            identity_shift=float(identity_shift),
            phi1_hat_base=phi1_hat_base,
            differentiable=differentiable,
            emit=emit,
            recycle_basis=recycle_basis,
            solve_driver=solve_v3_full_system_linear_gmres,
            solve_method_kind_explicit=solve_method_kind_explicit,
            requested=solve_method_kind_explicit in _SPARSE_HOST_SAFE_SOLVE_METHODS,
        )
    )
    if sparse_host_safe_result is not None:
        return sparse_host_safe_result
    if (
        solve_method_kind_explicit in _SPARSE_HOST_PC_GMRES_SOLVE_METHODS
        or solve_method_kind_explicit in _SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
    ):
        if differentiable is True:
            raise ValueError(
                "solve_method='sparse_pc_gmres'/'xblock_sparse_pc_gmres' is a non-differentiable host sparse-PC GMRES path."
            )
        if int(op.rhs_mode) != 1:
            raise NotImplementedError(
                "solve_method='sparse_pc_gmres'/'xblock_sparse_pc_gmres' is currently implemented for RHSMode=1 only."
            )
        sparse_pc_entry_policy = resolve_sparse_pc_entry_policy(
            op=op,
            solve_method_kind=solve_method_kind_explicit,
            has_reduced_modes=bool(has_reduced_modes),
            use_active_dof_mode=bool(use_active_dof_mode),
            xblock_active_dof_requested=bool(xblock_active_dof_requested),
            active_maps_available=bool(active_idx_jnp is not None and full_to_active_jnp is not None),
            use_dkes=bool(use_dkes),
            include_xdot_sparse_pc=bool(include_xdot_sparse_pc),
            include_electric_field_xi_sparse_pc=bool(include_electric_field_xi_sparse_pc),
            er_abs_sparse_pc=float(er_abs_sparse_pc),
            restart=int(restart),
            maxiter=maxiter,
            parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
            sparse_pc_default_restart=_rhsmode1_sparse_pc_default_restart,
            env=os.environ,
        )
        constrained_pas_pc = bool(sparse_pc_entry_policy.constrained_pas_pc)
        tokamak_pas_noer_pc = bool(sparse_pc_entry_policy.tokamak_pas_noer_pc)
        tokamak_pas_er_pc = bool(sparse_pc_entry_policy.tokamak_pas_er_pc)
        tokamak_fp_er_pc = bool(sparse_pc_entry_policy.tokamak_fp_er_pc)
        tokamak_fp_noer_pc = bool(sparse_pc_entry_policy.tokamak_fp_noer_pc)
        tokamak_fp_pc = bool(sparse_pc_entry_policy.tokamak_fp_pc)
        xblock_sparse_pc = bool(sparse_pc_entry_policy.xblock_sparse_pc)
        fortran_reduced_sparse_pc = bool(sparse_pc_entry_policy.fortran_reduced_sparse_pc)
        sparse_pc_use_active_dof = bool(sparse_pc_entry_policy.sparse_pc_use_active_dof)
        xblock_use_active_dof = bool(sparse_pc_entry_policy.xblock_use_active_dof)
        sparse_pc_fp_dense_velocity_block = sparse_pc_entry_policy.sparse_pc_fp_dense_velocity_block
        sparse_timer = Timer()
        pc_restart_env = sparse_pc_entry_policy.pc_restart_env
        pc_restart = int(sparse_pc_entry_policy.pc_restart)
        pc_maxiter = int(sparse_pc_entry_policy.pc_maxiter)

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
                    builder=_build_rhs1_xblock_two_level_preconditioner,
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
                    host_builder=_build_rhs1_xblock_smoothed_global_coupling_preconditioner,
                    device_builder=_build_rhs1_xblock_device_global_coupling_preconditioner,
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
            qi_seed_policy = resolve_xblock_qi_seed_policy_setup(env=os.environ)
            qi_coarse_seed_enabled = bool(qi_seed_policy.coarse_seed_enabled)
            qi_coarse_seed_used = False
            qi_coarse_seed_residual_before: float | None = None
            qi_coarse_seed_residual_after: float | None = None
            qi_coarse_seed_improvement_ratio: float | None = None
            qi_coarse_seed_rank = 0
            qi_coarse_seed_candidate_count = 0
            qi_coarse_seed_reason: str | None = None
            qi_coarse_seed_labels: tuple[str, ...] = ()
            qi_coarse_seed_s = 0.0
            qi_seed_max_rank = int(qi_seed_policy.max_rank)
            qi_seed_max_candidates = int(qi_seed_policy.max_candidates)
            qi_seed_max_angular_mode = int(qi_seed_policy.max_angular_mode)
            qi_seed_rank_rtol = float(qi_seed_policy.rank_rtol)
            qi_seed_min_improvement = float(qi_seed_policy.min_improvement)
            qi_seed_rcond = float(qi_seed_policy.rcond)
            qi_seed_include_angular = bool(qi_seed_policy.include_angular)
            qi_seed_include_blocks = bool(qi_seed_policy.include_blocks)
            qi_seed_include_radial = bool(qi_seed_policy.include_radial)
            qi_seed_include_radial_angular = bool(qi_seed_policy.include_radial_angular)
            qi_seed_include_constraint_moments = bool(qi_seed_policy.include_constraint_moments)
            qi_seed_include_schur = bool(qi_seed_policy.include_schur)
            qi_seed_basis_kind: str | None = qi_seed_policy.basis_kind
            qi_seed_basis_for_galerkin: RHS1QICoarseBasis | None = None
            qi_galerkin_preconditioner_enabled = bool(qi_seed_policy.galerkin_preconditioner_enabled)
            qi_galerkin_preconditioner_built = False
            qi_galerkin_preconditioner_used = False
            qi_galerkin_preconditioner_reason: str | None = None
            qi_galerkin_preconditioner_mode: str | None = None
            qi_galerkin_preconditioner_rank = 0
            qi_galerkin_preconditioner_candidate_count = 0
            qi_galerkin_preconditioner_coarse_shape: tuple[int, int] = (0, 0)
            qi_galerkin_preconditioner_coarse_norm = 0.0
            qi_galerkin_preconditioner_setup_s = 0.0
            qi_galerkin_preconditioner_rcond = 0.0
            qi_galerkin_preconditioner_damping = 1.0
            qi_galerkin_preconditioner_basis_reused_from_seed = False
            qi_galerkin_preconditioner_residual_before: float | None = None
            qi_galerkin_preconditioner_residual_after: float | None = None
            qi_galerkin_preconditioner_improvement_ratio: float | None = None
            qi_galerkin_preconditioner_probe_reduced = False
            qi_galerkin_preconditioner_probe_candidates: list[dict[str, object]] = []
            qi_galerkin_preconditioner_selected_index: int | None = None
            qi_galerkin_stats = {"applies": 0, "coarse_applies": 0, "base_applies": 0}
            qi_two_level_preconditioner_enabled = bool(qi_seed_policy.two_level_preconditioner_enabled)
            qi_two_level_preconditioner_built = False
            qi_two_level_preconditioner_used = False
            qi_two_level_preconditioner_reason: str | None = None
            qi_two_level_preconditioner_rank = 0
            qi_two_level_preconditioner_candidate_count = 0
            qi_two_level_preconditioner_coarse_shape: tuple[int, int] = (0, 0)
            qi_two_level_preconditioner_coarse_norm = 0.0
            qi_two_level_preconditioner_operator_on_basis_shape: tuple[int, int] = (0, 0)
            qi_two_level_preconditioner_operator_on_basis_norm = 0.0
            qi_two_level_preconditioner_coarse_solver: str | None = None
            qi_two_level_preconditioner_residual_augmented = False
            qi_two_level_preconditioner_rank_before_augmentation = 0
            qi_two_level_preconditioner_augmentation_labels: tuple[str, ...] = ()
            qi_two_level_preconditioner_residual_augment_max_extra = 0
            qi_two_level_preconditioner_residual_augment_steps = 0
            qi_two_level_preconditioner_residual_augment_include_residuals = False
            qi_two_level_preconditioner_smoothed_load_basis = False
            qi_two_level_preconditioner_smoothed_load_metadata: dict[str, object] = {}
            qi_two_level_preconditioner_setup_s = 0.0
            qi_two_level_preconditioner_rcond = 0.0
            qi_two_level_preconditioner_damping = 1.0
            qi_two_level_preconditioner_basis_reused_from_seed = False
            qi_two_level_preconditioner_residual_before: float | None = None
            qi_two_level_preconditioner_residual_after: float | None = None
            qi_two_level_preconditioner_improvement_ratio: float | None = None
            qi_two_level_preconditioner_probe_candidates: list[dict[str, object]] = []
            qi_two_level_preconditioner_selected_index: int | None = None
            qi_two_level_stats = {"applies": 0, "local_applies": 0}
            qi_device_preconditioner_enabled = bool(qi_seed_policy.device_preconditioner_enabled)
            qi_device_preconditioner_built = False
            qi_device_preconditioner_used = False
            qi_device_preconditioner_used_in_krylov = False
            qi_device_preconditioner_reason: str | None = None
            qi_device_preconditioner_rank = 0
            qi_device_preconditioner_candidate_count = 0
            qi_device_preconditioner_coarse_shape: tuple[int, int] = (0, 0)
            qi_device_preconditioner_operator_on_basis_shape: tuple[int, int] = (0, 0)
            qi_device_preconditioner_coarse_norm = 0.0
            qi_device_preconditioner_operator_on_basis_norm = 0.0
            qi_device_preconditioner_residual_before: float | None = None
            qi_device_preconditioner_residual_after: float | None = None
            qi_device_preconditioner_improvement_ratio: float | None = None
            qi_device_preconditioner_metadata: dict[str, object] = {}
            qi_device_preconditioner_setup_s = 0.0
            qi_device_preconditioner_min_improvement = 0.0
            qi_device_preconditioner_use_in_krylov = False
            qi_device_stats = {"applies": 0}
            qi_device_state_for_augmented_krylov = None
            qi_device_augmented_krylov_requested = False
            qi_device_augmented_krylov_used = False
            qi_device_augmented_krylov_rank = 0
            qi_device_augmented_krylov_reason: str | None = None
            qi_device_augmented_krylov_mode = "projected"
            qi_device_augmented_seed_requested = False
            qi_device_augmented_seed_available = False
            qi_device_augmented_seed_used = False
            qi_device_augmented_seed_rank = 0
            qi_device_augmented_seed_max_rank = 0
            qi_device_augmented_seed_reason: str | None = None
            qi_device_augmented_seed_projection_residual: float | None = None
            qi_device_augmented_seed_labels: tuple[str, ...] = ()
            qi_device_augmented_seed_basis_for_krylov = None
            qi_device_augmented_seed_action_for_krylov = None
            qi_deflated_preconditioner_enabled = bool(qi_seed_policy.deflated_preconditioner_enabled)
            qi_deflated_preconditioner_built = False
            qi_deflated_preconditioner_used = False
            qi_deflated_preconditioner_used_in_krylov = False
            qi_deflated_preconditioner_reason: str | None = None
            qi_deflated_preconditioner_rank = 0
            qi_deflated_preconditioner_candidate_count = 0
            qi_deflated_preconditioner_residual_before: float | None = None
            qi_deflated_preconditioner_residual_after: float | None = None
            qi_deflated_preconditioner_improvement_ratio: float | None = None
            qi_deflated_preconditioner_metadata: dict[str, object] = {}
            qi_deflated_preconditioner_setup_s = 0.0
            qi_deflated_stats = {"applies": 0, "local_applies": 0}
            qi_coarse_seed_stage = apply_xblock_qi_coarse_seed_stage(
                context=XBlockQICoarseSeedStageContext(
                    op=op,
                    x0_full=x0_full,
                    xblock_rhs=xblock_rhs,
                    matvec_no_count=_mv_true_no_count,
                    active_dof=bool(xblock_use_active_dof),
                    linear_size=int(xblock_linear_size),
                    policy=qi_seed_policy,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    basis_builder=_rhs1_xblock_qi_coarse_basis,
                    correction_builder=apply_rhs1_qi_coarse_correction,
                )
            )
            x0_full = qi_coarse_seed_stage.x0_full
            qi_seed_basis_for_galerkin = qi_coarse_seed_stage.basis_for_galerkin
            qi_coarse_seed_used = bool(qi_coarse_seed_stage.used)
            qi_coarse_seed_residual_before = qi_coarse_seed_stage.residual_before
            qi_coarse_seed_residual_after = qi_coarse_seed_stage.residual_after
            qi_coarse_seed_improvement_ratio = qi_coarse_seed_stage.improvement_ratio
            qi_coarse_seed_rank = int(qi_coarse_seed_stage.rank)
            qi_coarse_seed_candidate_count = int(qi_coarse_seed_stage.candidate_count)
            qi_coarse_seed_reason = qi_coarse_seed_stage.reason
            qi_coarse_seed_labels = qi_coarse_seed_stage.labels
            qi_coarse_seed_s = float(qi_coarse_seed_stage.setup_s)
            qi_galerkin_policy = resolve_xblock_qi_galerkin_policy_setup(
                enabled=bool(qi_galerkin_preconditioner_enabled),
                host_fallback_used=bool(xblock_device_host_fallback_decision.used),
                precondition_side=str(precondition_side),
                parse_modes=parse_rhs1_qi_galerkin_modes,
                parse_dampings=parse_rhs1_qi_galerkin_dampings,
                env=os.environ,
            )
            qi_galerkin_stage = apply_xblock_qi_galerkin_stage(
                context=XBlockQIGalerkinStageContext(
                    op=op,
                    base_preconditioner=precond_xblock_krylov,
                    matvec=_mv_xblock_krylov,
                    true_matvec_no_count=_mv_true_no_count,
                    xblock_rhs=xblock_rhs,
                    xblock_rhs_norm=float(xblock_rhs_norm),
                    active_dof=bool(xblock_use_active_dof),
                    linear_size=int(xblock_linear_size),
                    basis_for_galerkin=qi_seed_basis_for_galerkin,
                    seed_policy=qi_seed_policy,
                    galerkin_policy=qi_galerkin_policy,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    basis_builder=_rhs1_xblock_qi_coarse_basis,
                    preconditioner_builder=build_rhs1_qi_galerkin_preconditioner,
                )
            )
            precond_xblock_krylov = qi_galerkin_stage.preconditioner
            qi_seed_basis_for_galerkin = qi_galerkin_stage.basis_for_galerkin
            qi_galerkin_preconditioner_built = bool(qi_galerkin_stage.built)
            qi_galerkin_preconditioner_used = bool(qi_galerkin_stage.used)
            qi_galerkin_preconditioner_reason = qi_galerkin_stage.reason
            qi_galerkin_preconditioner_mode = qi_galerkin_stage.mode
            qi_galerkin_preconditioner_rank = int(qi_galerkin_stage.rank)
            qi_galerkin_preconditioner_candidate_count = int(qi_galerkin_stage.candidate_count)
            qi_galerkin_preconditioner_coarse_shape = qi_galerkin_stage.coarse_shape
            qi_galerkin_preconditioner_coarse_norm = float(qi_galerkin_stage.coarse_norm)
            qi_galerkin_preconditioner_setup_s = float(qi_galerkin_stage.setup_s)
            qi_galerkin_preconditioner_rcond = float(qi_galerkin_stage.rcond)
            qi_galerkin_preconditioner_damping = float(qi_galerkin_stage.damping)
            qi_galerkin_preconditioner_basis_reused_from_seed = bool(
                qi_galerkin_stage.basis_reused_from_seed
            )
            qi_galerkin_preconditioner_residual_before = qi_galerkin_stage.residual_before
            qi_galerkin_preconditioner_residual_after = qi_galerkin_stage.residual_after
            qi_galerkin_preconditioner_improvement_ratio = qi_galerkin_stage.improvement_ratio
            qi_galerkin_preconditioner_probe_reduced = bool(qi_galerkin_stage.probe_reduced)
            qi_galerkin_preconditioner_probe_candidates = qi_galerkin_stage.probe_candidates
            qi_galerkin_preconditioner_selected_index = qi_galerkin_stage.selected_index
            qi_galerkin_stats = qi_galerkin_stage.stats
            pc_factor_s += float(qi_galerkin_preconditioner_setup_s)
            qi_two_level_policy = resolve_xblock_qi_two_level_policy_setup(
                enabled=bool(qi_two_level_preconditioner_enabled),
                host_fallback_used=bool(xblock_device_host_fallback_decision.used),
                precondition_side=str(precondition_side),
                seed_max_rank=int(qi_seed_max_rank),
                parse_dampings=parse_rhs1_qi_galerkin_dampings,
                env=os.environ,
            )
            qi_two_level_stage = apply_xblock_qi_two_level_stage(
                context=XBlockQITwoLevelStageContext(
                    op=op,
                    rhs=rhs,
                    x0_full=x0_full,
                    xblock_rhs=xblock_rhs,
                    base_preconditioner=precond_xblock_krylov,
                    matvec=_mv_xblock_krylov,
                    true_matvec_no_count=_mv_true_no_count,
                    direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                    active_dof=bool(xblock_use_active_dof),
                    linear_size=int(xblock_linear_size),
                    basis_for_galerkin=qi_seed_basis_for_galerkin,
                    seed_policy=qi_seed_policy,
                    two_level_policy=qi_two_level_policy,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    basis_builder=_rhs1_xblock_qi_coarse_basis,
                    smoothed_load_basis_builder=_rhs1_xblock_smoothed_load_qi_basis,
                    orthonormalizer=orthonormalize_rhs1_qi_coarse_basis,
                    preconditioner_builder=build_rhs1_qi_two_level_preconditioner,
                )
            )
            precond_xblock_krylov = qi_two_level_stage.preconditioner
            x0_full = qi_two_level_stage.x0_full
            qi_seed_basis_for_galerkin = qi_two_level_stage.basis_for_galerkin
            qi_two_level_preconditioner_built = bool(qi_two_level_stage.built)
            qi_two_level_preconditioner_used = bool(qi_two_level_stage.used)
            qi_two_level_preconditioner_reason = qi_two_level_stage.reason
            qi_two_level_preconditioner_rank = int(qi_two_level_stage.rank)
            qi_two_level_preconditioner_candidate_count = int(qi_two_level_stage.candidate_count)
            qi_two_level_preconditioner_coarse_shape = qi_two_level_stage.coarse_shape
            qi_two_level_preconditioner_coarse_norm = float(qi_two_level_stage.coarse_norm)
            qi_two_level_preconditioner_operator_on_basis_shape = (
                qi_two_level_stage.operator_on_basis_shape
            )
            qi_two_level_preconditioner_operator_on_basis_norm = float(
                qi_two_level_stage.operator_on_basis_norm
            )
            qi_two_level_preconditioner_coarse_solver = qi_two_level_stage.coarse_solver
            qi_two_level_preconditioner_residual_augmented = bool(
                qi_two_level_stage.residual_augmented
            )
            qi_two_level_preconditioner_rank_before_augmentation = int(
                qi_two_level_stage.rank_before_augmentation
            )
            qi_two_level_preconditioner_augmentation_labels = qi_two_level_stage.augmentation_labels
            qi_two_level_preconditioner_residual_augment_max_extra = int(
                qi_two_level_stage.residual_augment_max_extra
            )
            qi_two_level_preconditioner_residual_augment_steps = int(
                qi_two_level_stage.residual_augment_steps
            )
            qi_two_level_preconditioner_residual_augment_include_residuals = bool(
                qi_two_level_stage.residual_augment_include_residuals
            )
            qi_two_level_preconditioner_smoothed_load_basis = bool(
                qi_two_level_stage.smoothed_load_basis
            )
            qi_two_level_preconditioner_smoothed_load_metadata = (
                qi_two_level_stage.smoothed_load_metadata
            )
            qi_two_level_preconditioner_setup_s = float(qi_two_level_stage.setup_s)
            qi_two_level_preconditioner_rcond = float(qi_two_level_stage.rcond)
            qi_two_level_preconditioner_damping = float(qi_two_level_stage.damping)
            qi_two_level_preconditioner_basis_reused_from_seed = bool(
                qi_two_level_stage.basis_reused_from_seed
            )
            qi_two_level_preconditioner_residual_before = qi_two_level_stage.residual_before
            qi_two_level_preconditioner_residual_after = qi_two_level_stage.residual_after
            qi_two_level_preconditioner_improvement_ratio = qi_two_level_stage.improvement_ratio
            qi_two_level_preconditioner_probe_candidates = qi_two_level_stage.probe_candidates
            qi_two_level_preconditioner_selected_index = qi_two_level_stage.selected_index
            qi_two_level_stats = qi_two_level_stage.stats
            pc_factor_s += float(qi_two_level_preconditioner_setup_s)
            qi_device_admission = resolve_xblock_qi_device_admission_setup(
                enabled=bool(qi_device_preconditioner_enabled),
                host_fallback_used=bool(xblock_device_host_fallback_decision.used),
                assembled_device_operator_available=assembled_device_operator is not None,
                assembled_operator_enabled=bool(assembled_operator_enabled),
                assembled_operator_built=bool(assembled_operator_built),
                assembled_operator_device_resident=bool(assembled_operator_device_resident),
                assembled_operator_device_error=assembled_operator_metadata.get("device_error"),
                env=os.environ,
            )
            if qi_device_admission.reason is not None and not qi_device_admission.should_build:
                qi_device_preconditioner_reason = qi_device_admission.reason
                qi_device_preconditioner_metadata = dict(qi_device_admission.metadata)
            for level, message in qi_device_admission.messages:
                if emit is not None:
                    emit(level, message)
            if qi_device_admission.should_build:
                qi_device_start_s = sparse_timer.elapsed_s()
                qi_device_matrix_free_enabled = bool(qi_device_admission.matrix_free_enabled)
                qi_device_base_config = resolve_xblock_qi_device_base_config_setup(
                    matrix_free_enabled=bool(qi_device_matrix_free_enabled),
                    assembled_device_operator_available=assembled_device_operator is not None,
                    precondition_side=str(precondition_side),
                    probe_uses_minres_step=_rhs1_qi_device_probe_uses_minres_step,
                    env=os.environ,
                )
                qi_device_local_smoother_kind = qi_device_base_config.local_smoother_kind
                qi_device_preconditioner_min_improvement = float(qi_device_base_config.min_improvement)
                qi_device_preconditioner_cycles = int(qi_device_base_config.cycles)
                qi_device_augmented_seed_requested = bool(qi_device_base_config.augmented_seed_requested)
                qi_device_augmented_seed_max_rank = int(qi_device_base_config.augmented_seed_max_rank)
                qi_device_preconditioner_minres_step = bool(qi_device_base_config.minres_step)
                qi_device_preconditioner_alpha_clip = float(qi_device_base_config.alpha_clip)
                qi_device_use_in_krylov_requested = bool(qi_device_base_config.use_in_krylov_requested)
                qi_device_preconditioner_use_in_krylov = bool(qi_device_base_config.use_in_krylov)
                qi_device_compose_with_base = bool(qi_device_base_config.compose_with_base)
                qi_device_compose_mode = qi_device_base_config.compose_mode
                qi_device_enrichment_config = resolve_xblock_qi_device_enrichment_config_setup(
                    matrix_free_enabled=bool(qi_device_matrix_free_enabled),
                    env=os.environ,
                )
                qi_device_operator_krylov_enrichment = bool(
                    qi_device_enrichment_config.operator_krylov_enrichment
                )
                qi_device_multilevel_config = resolve_xblock_qi_device_multilevel_config_setup(
                    env=os.environ
                )
                qi_device_multilevel_coarse = bool(
                    qi_device_multilevel_config.multilevel_coarse
                )
                qi_device_residual_correction_controls = (
                    _rhs1_qi_device_residual_correction_controls()
                )
                qi_device_residual_correction_setup_kwargs = (
                    _rhs1_qi_device_residual_correction_setup_kwargs(
                        qi_device_residual_correction_controls
                    )
                )
                qi_device_residual_correction_metadata = (
                    _rhs1_qi_device_residual_correction_metadata(
                        qi_device_residual_correction_controls
                    )
                )
                qi_device_multilevel_max_rank_env = os.environ.get(
                    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK",
                    "",
                ).strip()
                qi_device_multilevel_max_rank: int | None = None
                if qi_device_multilevel_max_rank_env:
                    try:
                        qi_device_multilevel_max_rank = max(1, int(qi_device_multilevel_max_rank_env))
                    except ValueError:
                        qi_device_multilevel_max_rank = None
                qi_device_extra_coarse_controls = _rhs1_qi_device_extra_coarse_controls()
                qi_device_extra_coarse_setup_kwargs = (
                    _rhs1_qi_device_extra_coarse_setup_kwargs(
                        qi_device_extra_coarse_controls
                    )
                )
                qi_device_extra_coarse_metadata = (
                    _rhs1_qi_device_extra_coarse_metadata(
                        qi_device_extra_coarse_controls
                    )
                )
                qi_device_setup_summary = _rhs1_qi_device_setup_summary(
                    seed_max_rank=int(qi_seed_max_rank),
                    n_species=int(getattr(op, "n_species", 1)),
                    assembled_device_operator_available=(
                        assembled_device_operator is not None
                    ),
                    enrichment_config=qi_device_enrichment_config,
                    multilevel_config=qi_device_multilevel_config,
                    multilevel_max_rank=qi_device_multilevel_max_rank,
                    extra_coarse_controls=qi_device_extra_coarse_controls,
                    residual_correction_controls=qi_device_residual_correction_controls,
                )
                qi_device_rank_budget = int(qi_device_setup_summary.rank_budget)
                qi_device_max_rank = qi_device_setup_summary.max_rank
                try:
                    if assembled_device_operator is None and not bool(qi_device_matrix_free_enabled):
                        raise RuntimeError("missing assembled device CSR operator and matrix-free fallback disabled")
                    qi_device_preconditioner_basis_reused_from_seed = qi_seed_basis_for_galerkin is not None
                    if qi_seed_basis_for_galerkin is None:
                        qi_seed_basis_for_galerkin = _rhs1_xblock_qi_coarse_basis(
                            op=op,
                            active_dof=bool(xblock_use_active_dof),
                            linear_size=int(xblock_linear_size),
                            max_rank=int(qi_seed_max_rank),
                            rank_rtol=float(qi_seed_rank_rtol),
                            include_angular=bool(qi_seed_include_angular),
                            include_blocks=bool(qi_seed_include_blocks),
                            basis_kind=qi_seed_basis_kind,
                            max_candidates=int(qi_seed_max_candidates),
                            max_angular_mode=int(qi_seed_max_angular_mode),
                            include_radial=bool(qi_seed_include_radial),
                            include_radial_angular=bool(qi_seed_include_radial_angular),
                            include_constraint_moments=bool(qi_seed_include_constraint_moments),
                            include_schur=bool(qi_seed_include_schur),
                        )
                    qi_current = (
                        jnp.zeros_like(xblock_rhs)
                        if x0_full is None
                        else jnp.asarray(x0_full, dtype=jnp.float64)
                    )
                    qi_device_residual_seed = None
                    if bool(qi_device_setup_summary.residual_seed_required):
                        qi_device_residual_seed = xblock_rhs - _mv_true_no_count(qi_current)
                    qi_operator_for_setup = (
                        assembled_device_operator if assembled_device_operator is not None else _mv_true_no_count
                    )
                    if emit is not None:
                        for qi_device_message in qi_device_setup_summary.progress_messages:
                            emit(1, qi_device_message)
                    qi_device_setup_config = build_xblock_qi_device_setup_config(
                        XBlockQIDeviceSetupConfigContext(
                            op=op,
                            active_dof=bool(xblock_use_active_dof),
                            linear_size=int(xblock_linear_size),
                            base_config=qi_device_base_config,
                            enrichment_config=qi_device_enrichment_config,
                            multilevel_config=qi_device_multilevel_config,
                            multilevel_max_rank=qi_device_multilevel_max_rank,
                            max_rank=qi_device_max_rank,
                            extra_coarse_controls=qi_device_extra_coarse_controls,
                            extra_coarse_setup_kwargs=(
                                qi_device_extra_coarse_setup_kwargs
                            ),
                            residual_correction_setup_kwargs=(
                                qi_device_residual_correction_setup_kwargs
                            ),
                        )
                    )
                    qi_device_state = setup_rhs1_qi_device_preconditioner(
                        operator=qi_operator_for_setup,
                        coarse_basis=qi_seed_basis_for_galerkin,
                        residual_seed=qi_device_residual_seed,
                        total_size=int(xblock_linear_size),
                        dtype=jnp.float64,
                        operator_metadata=assembled_operator_metadata,
                        geometry_metadata=qi_device_setup_config.geometry_metadata,
                        config=qi_device_setup_config.config,
                    )
                    qi_device_preconditioner_built = True
                    qi_device_state_for_augmented_krylov = qi_device_state
                    qi_device_preconditioner_rank = int(qi_device_state.metadata.rank)
                    qi_device_preconditioner_candidate_count = int(qi_device_state.basis.metadata.candidate_count)
                    qi_device_preconditioner_coarse_shape = tuple(
                        int(value) for value in qi_device_state.metadata.coarse_operator_shape
                    )
                    qi_device_preconditioner_operator_on_basis_shape = tuple(
                        int(value) for value in qi_device_state.metadata.operator_on_basis_shape
                    )
                    qi_device_preconditioner_coarse_norm = float(qi_device_state.metadata.coarse_operator_norm)
                    qi_device_preconditioner_operator_on_basis_norm = float(
                        qi_device_state.metadata.operator_on_basis_norm
                    )
                    if bool(qi_device_augmented_seed_requested):
                        qi_device_augmented_seed = probe_rhs1_qi_device_augmented_seed(
                            rhs=xblock_rhs,
                            x0=qi_current,
                            state=qi_device_state,
                            operator=_mv_true_no_count,
                            min_relative_improvement=float(qi_device_preconditioner_min_improvement),
                            max_cycles=int(qi_device_preconditioner_cycles),
                            residual_minimizing_step=bool(qi_device_preconditioner_minres_step),
                            alpha_clip=float(qi_device_preconditioner_alpha_clip),
                            max_rank=int(qi_device_augmented_seed_max_rank),
                        )
                        x_device_candidate = qi_device_augmented_seed.solution
                        qi_device_probe = qi_device_augmented_seed.probe
                        qi_device_augmented_seed_rank = int(qi_device_augmented_seed.rank)
                        qi_device_augmented_seed_available = bool(
                            qi_device_probe.accepted and qi_device_augmented_seed_rank > 0
                        )
                        qi_device_augmented_seed_reason = str(qi_device_augmented_seed.reason)
                        qi_device_augmented_seed_projection_residual = (
                            None
                            if qi_device_augmented_seed.projection_residual_norm is None
                            else float(qi_device_augmented_seed.projection_residual_norm)
                        )
                        qi_device_augmented_seed_labels = tuple(
                            str(label) for label in qi_device_augmented_seed.accepted_labels
                        )
                        if bool(qi_device_augmented_seed_available):
                            qi_device_augmented_seed_basis_for_krylov = jnp.asarray(
                                qi_device_augmented_seed.augmentation_basis,
                                dtype=jnp.float64,
                            )
                            qi_device_augmented_seed_action_for_krylov = jnp.asarray(
                                qi_device_augmented_seed.operator_on_augmentation,
                                dtype=jnp.float64,
                            )
                    else:
                        x_device_candidate, qi_device_probe = probe_rhs1_qi_device_preconditioner(
                            rhs=xblock_rhs,
                            x0=qi_current,
                            state=qi_device_state,
                            operator=_mv_true_no_count,
                            min_relative_improvement=float(qi_device_preconditioner_min_improvement),
                            max_cycles=int(qi_device_preconditioner_cycles),
                            residual_minimizing_step=bool(qi_device_preconditioner_minres_step),
                            alpha_clip=float(qi_device_preconditioner_alpha_clip),
                        )
                    qi_device_preconditioner_residual_before = float(qi_device_probe.residual_before_norm)
                    qi_device_preconditioner_residual_after = float(qi_device_probe.residual_after_norm)
                    qi_device_preconditioner_improvement_ratio = (
                        None
                        if qi_device_probe.improvement_ratio is None
                        else float(qi_device_probe.improvement_ratio)
                    )
                    qi_device_preconditioner_reason = str(qi_device_probe.reason)
                    qi_device_preconditioner_metadata = (
                        build_xblock_qi_device_preconditioner_metadata(
                            XBlockQIDeviceMetadataContext(
                                probe=qi_device_probe,
                                state=qi_device_state,
                                basis_reused_from_seed=(
                                    qi_device_preconditioner_basis_reused_from_seed
                                ),
                                min_improvement=qi_device_preconditioner_min_improvement,
                                cycles_requested=qi_device_preconditioner_cycles,
                                minres_step=qi_device_preconditioner_minres_step,
                                alpha_clip=qi_device_preconditioner_alpha_clip,
                                augmented_seed_requested=qi_device_augmented_seed_requested,
                                augmented_seed_available=qi_device_augmented_seed_available,
                                augmented_seed_used=qi_device_augmented_seed_used,
                                augmented_seed_rank=qi_device_augmented_seed_rank,
                                augmented_seed_max_rank=qi_device_augmented_seed_max_rank,
                                augmented_seed_reason=qi_device_augmented_seed_reason,
                                augmented_seed_projection_residual=(
                                    qi_device_augmented_seed_projection_residual
                                ),
                                augmented_seed_labels=qi_device_augmented_seed_labels,
                                use_in_krylov=qi_device_preconditioner_use_in_krylov,
                                use_in_krylov_requested=qi_device_use_in_krylov_requested,
                                precondition_side=precondition_side,
                                compose_with_base=qi_device_compose_with_base,
                                compose_mode=qi_device_compose_mode,
                                matrix_free_enabled=qi_device_matrix_free_enabled,
                                local_smoother_kind=qi_device_local_smoother_kind,
                                enrichment_config=qi_device_enrichment_config,
                                multilevel_config=qi_device_multilevel_config,
                                multilevel_max_rank=qi_device_multilevel_max_rank,
                                extra_coarse_metadata=qi_device_extra_coarse_metadata,
                                residual_correction_metadata=(
                                    qi_device_residual_correction_metadata
                                ),
                                max_rank_requested=qi_device_max_rank,
                            )
                        )
                    )
                    qi_device_preconditioner_probe_cycles = int(
                        qi_device_preconditioner_metadata.get(
                            "cycles",
                            1 if bool(qi_device_probe.accepted) else 0,
                        )
                    )
                    base_precond_before_qi_device = precond_xblock_krylov

                    def _precond_xblock_qi_device(v: jnp.ndarray) -> jnp.ndarray:
                        qi_device_stats["applies"] += 1
                        v_j = jnp.asarray(v, dtype=jnp.float64)
                        if bool(qi_device_compose_with_base):
                            base = jnp.asarray(base_precond_before_qi_device(v_j), dtype=jnp.float64)
                            if qi_device_compose_mode == "multiplicative":
                                coarse_input = v_j - jnp.asarray(_mv_true_no_count(base), dtype=jnp.float64)
                            else:
                                coarse_input = v_j
                            coarse = jnp.asarray(
                                qi_device_state.apply(jnp.asarray(coarse_input, dtype=jnp.float64)),
                                dtype=jnp.float64,
                            )
                            return base + coarse
                        return jnp.asarray(
                            qi_device_state.apply(v_j),
                            dtype=jnp.float64,
                        )

                    coupled_stage_accepted_for_krylov = (
                        bool(
                            qi_device_preconditioner_metadata.get(
                                "coupled_residual_equation_accepted", False
                            )
                        )
                        and int(
                            qi_device_preconditioner_metadata.get(
                                "coupled_residual_equation_rank", 0
                            )
                            or 0
                        )
                        > 0
                    )
                    qi_device_install_after_seed_reject = bool(
                        _rhs1_qi_device_coupled_install_on_reject_requested(
                            qi_device_residual_correction_controls
                        )
                        and qi_device_preconditioner_use_in_krylov
                        and coupled_stage_accepted_for_krylov
                        and not bool(qi_device_probe.accepted)
                    )
                    qi_device_preconditioner_metadata[
                        "seed_probe_accepted"
                    ] = bool(qi_device_probe.accepted)
                    qi_device_preconditioner_metadata[
                        "installed_in_krylov_after_seed_reject"
                    ] = bool(qi_device_install_after_seed_reject)
                    qi_device_status_fields = _rhs1_qi_device_status_fields(
                        extra_coarse_controls=qi_device_extra_coarse_controls,
                        residual_correction_controls=(
                            qi_device_residual_correction_controls
                        ),
                        metadata=qi_device_preconditioner_metadata,
                    )
                    if bool(qi_device_probe.accepted):
                        x0_full = jnp.asarray(x_device_candidate, dtype=jnp.float64)
                        qi_device_preconditioner_used = True
                        if bool(qi_device_preconditioner_use_in_krylov):
                            precond_xblock_krylov = _precond_xblock_qi_device
                            qi_device_preconditioner_used_in_krylov = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                "QI device preconditioner accepted "
                                f"residual {qi_device_preconditioner_residual_before:.6e} "
                                f"-> {qi_device_preconditioner_residual_after:.6e} "
                                f"(rank={int(qi_device_preconditioner_rank)} "
                                f"cycles={int(qi_device_preconditioner_probe_cycles)} "
                                f"ratio={float(qi_device_preconditioner_improvement_ratio):.6e} "
                                f"use_in_krylov={int(bool(qi_device_preconditioner_use_in_krylov))} "
                                f"augmented_seed_requested={int(bool(qi_device_augmented_seed_requested))} "
                                f"augmented_seed_available={int(bool(qi_device_augmented_seed_available))} "
                                f"augmented_seed_used={int(bool(qi_device_augmented_seed_used))} "
                                f"augmented_seed_rank={int(qi_device_augmented_seed_rank)} "
                                f"augmented_seed_max_rank={int(qi_device_augmented_seed_max_rank)} "
                                f"augmented_seed_reason={qi_device_augmented_seed_reason or 'none'} "
                                f"augmented_seed_projection_residual={float(qi_device_augmented_seed_projection_residual) if qi_device_augmented_seed_projection_residual is not None else float('nan'):.6e} "
                                f"operator_krylov={int(bool(qi_device_operator_krylov_enrichment))} "
                                f"coarse_reuse={int(bool(qi_device_multilevel_coarse))} "
                                f"{qi_device_status_fields} "
                                f"compose_base={int(bool(qi_device_compose_with_base))})",
                            )
                    elif bool(qi_device_install_after_seed_reject):
                        precond_xblock_krylov = _precond_xblock_qi_device
                        qi_device_preconditioner_used = True
                        qi_device_preconditioner_used_in_krylov = True
                        qi_device_preconditioner_reason = (
                            "krylov_installed_after_seed_probe_reject"
                        )
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                "QI device preconditioner installed in Krylov after seed "
                                f"probe reject (rank={int(qi_device_preconditioner_rank)} "
                                f"coupled_rank={int(qi_device_preconditioner_metadata.get('coupled_residual_equation_rank', 0))} "
                                f"coupled_candidates={int(qi_device_preconditioner_metadata.get('coupled_residual_equation_candidate_count', 0))} "
                                f"residual {float(qi_device_preconditioner_residual_before):.6e} "
                                f"-> {float(qi_device_preconditioner_residual_after):.6e})",
                            )
                    elif emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            "QI device preconditioner rejected "
                            f"reason={qi_device_preconditioner_reason} "
                            f"residual {float(qi_device_preconditioner_residual_before):.6e} "
                            f"-> {float(qi_device_preconditioner_residual_after):.6e} "
                            f"(rank={int(qi_device_preconditioner_rank)} "
                            f"cycles={int(qi_device_preconditioner_probe_cycles)} "
                            f"ratio={float(qi_device_preconditioner_improvement_ratio) if qi_device_preconditioner_improvement_ratio is not None else float('nan'):.6e} "
                            f"step_policy={qi_device_preconditioner_metadata.get('step_policy', 'fixed')} "
                            f"use_in_krylov={int(bool(qi_device_preconditioner_use_in_krylov))} "
                            f"augmented_seed_requested={int(bool(qi_device_augmented_seed_requested))} "
                            f"augmented_seed_available={int(bool(qi_device_augmented_seed_available))} "
                            f"augmented_seed_used={int(bool(qi_device_augmented_seed_used))} "
                            f"augmented_seed_rank={int(qi_device_augmented_seed_rank)} "
                            f"augmented_seed_max_rank={int(qi_device_augmented_seed_max_rank)} "
                            f"augmented_seed_reason={qi_device_augmented_seed_reason or 'none'} "
                            f"augmented_seed_projection_residual={float(qi_device_augmented_seed_projection_residual) if qi_device_augmented_seed_projection_residual is not None else float('nan'):.6e} "
                            f"operator_krylov={int(bool(qi_device_operator_krylov_enrichment))} "
                            f"coarse_reuse={int(bool(qi_device_multilevel_coarse))} "
                            f"{qi_device_status_fields} "
                            f"compose_base={int(bool(qi_device_compose_with_base))})",
                        )
                except Exception as exc:  # noqa: BLE001
                    qi_device_preconditioner_reason = f"{type(exc).__name__}: {exc}"
                    qi_device_preconditioner_metadata = {"error": qi_device_preconditioner_reason}
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"QI device preconditioner disabled after build failure ({type(exc).__name__}: {exc})",
                        )
                qi_device_preconditioner_setup_s = float(sparse_timer.elapsed_s() - qi_device_start_s)
                pc_factor_s += float(qi_device_preconditioner_setup_s)
            if qi_deflated_preconditioner_enabled and bool(xblock_device_host_fallback_decision.used):
                qi_deflated_preconditioner_reason = "disabled_by_device_host_fallback"
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        "QI residual-deflated preconditioner disabled because device-host fallback is active",
                    )
            elif qi_deflated_preconditioner_enabled and precondition_side == "none":
                qi_deflated_preconditioner_reason = "disabled_by_precondition_side_none"
            elif qi_deflated_preconditioner_enabled:
                qi_deflated_policy = resolve_xblock_qi_deflated_policy_setup(os.environ)
                qi_deflated_stage = apply_xblock_qi_deflated_stage(
                    context=XBlockQIDeflatedStageContext(
                        op=op,
                        rhs=rhs,
                        x0_full=x0_full,
                        xblock_rhs=xblock_rhs,
                        base_preconditioner=precond_xblock_krylov,
                        matvec=_mv_xblock_krylov,
                        true_matvec_no_count=_mv_true_no_count,
                        active_dof=bool(xblock_use_active_dof),
                        reduce_full=_xblock_reduce_full if bool(xblock_use_active_dof) else None,
                        policy=qi_deflated_policy,
                        elapsed_s=sparse_timer.elapsed_s,
                        emit=emit,
                        global_load_basis_builder=_rhs1_xblock_global_coupling_load_basis,
                        preconditioner_builder=build_rhs1_qi_residual_deflated_preconditioner,
                        minres_seed_probe=probe_rhs1_qi_deflated_minres_seed,
                        linear_probe=probe_rhs1_qi_deflated_correction,
                    )
                )
                precond_xblock_krylov = qi_deflated_stage.preconditioner
                x0_full = qi_deflated_stage.x0_full
                qi_deflated_preconditioner_built = bool(qi_deflated_stage.built)
                qi_deflated_preconditioner_used = bool(qi_deflated_stage.used)
                qi_deflated_preconditioner_used_in_krylov = bool(
                    qi_deflated_stage.used_in_krylov
                )
                qi_deflated_preconditioner_reason = qi_deflated_stage.reason
                qi_deflated_preconditioner_rank = int(qi_deflated_stage.rank)
                qi_deflated_preconditioner_candidate_count = int(
                    qi_deflated_stage.candidate_count
                )
                qi_deflated_preconditioner_residual_before = (
                    qi_deflated_stage.residual_before
                )
                qi_deflated_preconditioner_residual_after = (
                    qi_deflated_stage.residual_after
                )
                qi_deflated_preconditioner_improvement_ratio = (
                    qi_deflated_stage.improvement_ratio
                )
                qi_deflated_preconditioner_metadata = qi_deflated_stage.metadata
                qi_deflated_preconditioner_setup_s = float(qi_deflated_stage.setup_s)
                qi_deflated_stats = qi_deflated_stage.stats
                pc_factor_s += float(qi_deflated_preconditioner_setup_s)
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
            solve_solution_to_physical = lambda v: jnp.asarray(v, dtype=jnp.float64)
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

            def _device_cycle_progress_callback(
                *,
                cycle: int,
                iterations: int,
                residual_norm: float,
                target: float,
            ) -> None:
                if emit is None:
                    return
                emit(
                    0,
                    xblock_device_cycle_progress_message(
                        cycle=int(cycle),
                        iterations=int(iterations),
                        residual_norm=float(residual_norm),
                        target=float(target),
                        elapsed_s=sparse_timer.elapsed_s(),
                    ),
                )

            def _host_krylov_progress_callback(iteration: int, residual_norm: float) -> None:
                if emit is None or progress_every <= 0:
                    return
                if int(iteration) % int(progress_every) != 0:
                    return
                emit(
                    1,
                    xblock_host_krylov_progress_message(
                        iteration=int(iteration),
                        residual_norm=float(residual_norm),
                        elapsed_s=sparse_timer.elapsed_s(),
                    ),
                )

            device_krylov_iterations: int | None = None
            device_krylov_estimated_matvecs: int | None = None
            first_krylov = run_xblock_first_krylov_attempt(
                XBlockFirstKrylovAttemptContext(
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
                    host_progress_callback=_host_krylov_progress_callback,
                    device_cycle_progress_callback=_device_cycle_progress_callback,
                    gmres_solver=gmres_solve_with_history_scipy,
                    lgmres_solver=lgmres_solve_with_history_scipy,
                    gcrotmk_solver=gcrotmk_solve_with_history_scipy,
                    bicgstab_solver=bicgstab_solve_with_history_scipy,
                    fgmres_solver=fgmres_solve_with_residual,
                    fgmres_jit_solver=fgmres_solve_with_residual_jit,
                    fgmres_cycle_jit_solver=fgmres_cycle_jit_solve_with_residual,
                    bicgstab_jax_solver=bicgstab_solve_with_residual,
                    tfqmr_jax_solver=tfqmr_solve_with_residual,
                )
            )
            solve_s = (sparse_timer.elapsed_s() - solve_start_s) + float(xblock_side_probe_s) + float(probe_coarse_s)
            solve_state = xblock_krylov_state_from_first_attempt(
                XBlockFirstKrylovSolveStateContext(
                    krylov_method=str(xblock_krylov_method),
                    first_attempt=first_krylov,
                    solve_s=float(solve_s),
                    solution_to_physical=solve_solution_to_physical,
                    physical_rhs=xblock_rhs,
                    physical_matvec=_mv_true,
                    mv_count=int(mv_count),
                )
            )
            x_solution_np = solve_state.x_solution
            x_physical_np = solve_state.x_physical
            residual_norm_xblock_pc = float(solve_state.residual_norm)
            history = solve_state.history
            candidate_krylov_method = str(solve_state.krylov_method)
            candidate_residual_norm = float(residual_norm_xblock_pc)
            device_krylov_iterations = solve_state.device_iterations
            device_krylov_estimated_matvecs = solve_state.device_estimated_matvecs
            candidate_iterations = int(solve_state.reported_iterations)
            candidate_matvecs = int(solve_state.reported_matvecs)
            fallback_started_from_candidate = False
            fallback_candidate_improved_rhs = False
            fallback_to_gmres = _rhs1_xblock_policy.rhs1_xblock_fallback_to_gmres_enabled(
                env_value=os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_FALLBACK_GMRES", ""),
                xblock_side_probe_lgmres_rescue=bool(xblock_side_probe_lgmres_rescue),
                xblock_krylov_method=str(xblock_krylov_method),
            )
            fallback_result = run_xblock_gmres_fallback_if_needed(
                XBlockGMRESFallbackContext(
                    krylov_method=str(xblock_krylov_method),
                    fallback_enabled=bool(fallback_to_gmres),
                    x_solution=x_solution_np,
                    x_physical=x_physical_np,
                    residual_norm=float(residual_norm_xblock_pc),
                    history=history,
                    solve_s=float(solve_s),
                    target=float(target_xblock),
                    rhs_norm=float(xblock_rhs_norm),
                    original_x0=solve_x0,
                    solve_rhs=solve_rhs,
                    solve_matvec=solve_matvec,
                    solve_preconditioner=solve_preconditioner,
                    precondition_side=str(precondition_side),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(pc_restart),
                    maxiter=pc_maxiter,
                    progress_callback=_host_krylov_progress_callback,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    gmres_solver=gmres_solve_with_history_scipy,
                    initial_guess_builder=_rhs1_xblock_fallback_initial_guess,
                    solution_to_physical=solve_solution_to_physical,
                    physical_rhs=xblock_rhs,
                    physical_matvec=_mv_true,
                    device_iterations=device_krylov_iterations,
                    device_estimated_matvecs=device_krylov_estimated_matvecs,
                )
            )
            solve_state = xblock_krylov_state_from_gmres_fallback(
                fallback=fallback_result,
                mv_count=int(mv_count),
            )
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
            post_corrections = run_xblock_post_solve_corrections(
                XBlockPostSolveCorrectionContext(
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
                )
            )
            x_np = np.asarray(post_corrections.x, dtype=np.float64)
            residual_norm_xblock_pc = float(post_corrections.residual_norm)
            solve_s = float(post_corrections.solve_s)
            emit_xblock_sparse_pc_completion(
                XBlockSparsePCCompletionContext(
                    emit=emit,
                    krylov_method=str(xblock_krylov_method),
                    elapsed_s=sparse_timer.elapsed_s(),
                    iterations=int(reported_iterations),
                    matvecs=int(reported_matvecs),
                    residual_norm=float(residual_norm_xblock_pc),
                    target=float(target_xblock),
                    history=history,
                )
            )
            xblock_final_metadata_state = (
                xblock_sparse_pc_final_metadata_state_from_context(
                    XBlockSparsePCFinalMetadataStateContext(
                        core=XBlockSparsePCFinalCoreState(
                            candidate_iterations=candidate_iterations,
                            candidate_krylov_method=candidate_krylov_method,
                            candidate_matvecs=candidate_matvecs,
                            candidate_residual_norm=candidate_residual_norm,
                            device_krylov_estimated_matvecs=(
                                device_krylov_estimated_matvecs
                            ),
                            fallback_candidate_improved_rhs=(
                                fallback_candidate_improved_rhs
                            ),
                            fallback_started_from_candidate=(
                                fallback_started_from_candidate
                            ),
                            mv_count=mv_count,
                            pc_factor_s=pc_factor_s,
                            pc_maxiter=pc_maxiter,
                            pc_restart=pc_restart,
                            precondition_side=precondition_side,
                            reported_iterations=reported_iterations,
                            reported_matvecs=reported_matvecs,
                            setup_s=setup_s,
                            solve_s=solve_s,
                            sparse_timer=sparse_timer,
                            xblock_assembled_host_fp=xblock_assembled_host_fp,
                            xblock_default_restart_capped=(
                                xblock_default_restart_capped
                            ),
                            xblock_default_right_pc=xblock_default_right_pc,
                            xblock_jax_factor_apply=xblock_jax_factor_apply,
                            xblock_jax_factor_format=xblock_jax_factor_format,
                            xblock_jax_factors=xblock_jax_factors,
                            xblock_krylov_method=xblock_krylov_method,
                            xblock_linear_size=xblock_linear_size,
                            xblock_lower_fill_ignored_env=(
                                xblock_lower_fill_ignored_env
                            ),
                            xblock_lower_fill_mode=xblock_lower_fill_mode,
                            xblock_preconditioner_built=(
                                xblock_preconditioner_built
                            ),
                            xblock_preconditioner_xi=xblock_preconditioner_xi,
                            xblock_use_active_dof=xblock_use_active_dof,
                        ),
                        device=XBlockSparsePCFinalDeviceState(
                            assembled_operator_built=assembled_operator_built,
                            assembled_operator_device_resident=(
                                assembled_operator_device_resident
                            ),
                            fgmres_block_between_cycles=fgmres_block_between_cycles,
                            global_coupling_built=global_coupling_built,
                            global_coupling_metadata=global_coupling_metadata,
                            qi_device_augmented_krylov_mode=(
                                qi_device_augmented_krylov_mode
                            ),
                            qi_device_augmented_krylov_rank=(
                                qi_device_augmented_krylov_rank
                            ),
                            qi_device_augmented_krylov_reason=(
                                qi_device_augmented_krylov_reason
                            ),
                            qi_device_augmented_krylov_requested=(
                                qi_device_augmented_krylov_requested
                            ),
                            qi_device_augmented_krylov_used=(
                                qi_device_augmented_krylov_used
                            ),
                            qi_device_augmented_seed_available=(
                                qi_device_augmented_seed_available
                            ),
                            qi_device_augmented_seed_labels=(
                                qi_device_augmented_seed_labels
                            ),
                            qi_device_augmented_seed_max_rank=(
                                qi_device_augmented_seed_max_rank
                            ),
                            qi_device_augmented_seed_projection_residual=(
                                qi_device_augmented_seed_projection_residual
                            ),
                            qi_device_augmented_seed_rank=(
                                qi_device_augmented_seed_rank
                            ),
                            qi_device_augmented_seed_reason=(
                                qi_device_augmented_seed_reason
                            ),
                            qi_device_augmented_seed_requested=(
                                qi_device_augmented_seed_requested
                            ),
                            qi_device_augmented_seed_used=(
                                qi_device_augmented_seed_used
                            ),
                            tfqmr_replacement_interval=tfqmr_replacement_interval,
                            two_level_built=two_level_built,
                            xblock_device_fgmres_forced_right_pc=(
                                xblock_device_fgmres_forced_right_pc
                            ),
                            xblock_device_fgmres_jit=xblock_device_fgmres_jit,
                            xblock_device_fgmres_jit_mode=(
                                xblock_device_fgmres_jit_mode
                            ),
                            xblock_device_fgmres_jit_outer_k=(
                                xblock_device_fgmres_jit_outer_k
                            ),
                            xblock_device_host_fallback_auto_disabled_by_qi_device=(
                                xblock_device_host_fallback_auto_disabled_by_qi_device
                            ),
                            xblock_device_host_fallback_decision=(
                                xblock_device_host_fallback_decision
                            ),
                            xblock_device_krylov_forced_jax_factors=(
                                xblock_device_krylov_forced_jax_factors
                            ),
                            xblock_krylov_env_requested=xblock_krylov_env_requested,
                            xblock_qi_device_operator_reuse_decision=(
                                xblock_qi_device_operator_reuse_decision
                            ),
                        ),
                        preflight=XBlockSparsePCFinalPreflightState(
                            preflight_improvement=preflight_improvement,
                            preflight_min_improvement=preflight_min_improvement,
                            preflight_passed=preflight_passed,
                            preflight_required=preflight_required,
                            preflight_residual_norm=preflight_residual_norm,
                            probe_coarse_angular_lmax=probe_coarse_angular_lmax,
                            probe_coarse_direction_counts=(
                                probe_coarse_direction_counts
                            ),
                            probe_coarse_direction_names=(
                                probe_coarse_direction_names
                            ),
                            probe_coarse_fsavg_lmax=probe_coarse_fsavg_lmax,
                            probe_coarse_history=probe_coarse_history,
                            probe_coarse_include_angular_residual=(
                                probe_coarse_include_angular_residual
                            ),
                            probe_coarse_residual_after=probe_coarse_residual_after,
                            probe_coarse_residual_before=probe_coarse_residual_before,
                            probe_coarse_s=probe_coarse_s,
                            probe_coarse_seed_initialized=(
                                probe_coarse_seed_initialized
                            ),
                            probe_coarse_steps_requested=(
                                probe_coarse_steps_requested
                            ),
                        ),
                        nested=XBlockSparsePCFinalNestedMetadata(
                            xblock_assembled_operator_result_metadata=(
                                xblock_assembled_operator_diagnostics(
                                    XBlockAssembledOperatorDiagnosticsContext(
                                        enabled=assembled_operator_enabled,
                                        built=assembled_operator_built,
                                        metadata=assembled_operator_metadata,
                                        row_equilibration_enabled=(
                                            xblock_row_equilibration_enabled
                                        ),
                                        row_equilibration_built=(
                                            xblock_row_equilibration_built
                                        ),
                                        row_equilibration_metadata=(
                                            xblock_row_equilibration_metadata
                                        ),
                                        col_equilibration_enabled=(
                                            xblock_col_equilibration_enabled
                                        ),
                                        col_equilibration_built=(
                                            xblock_col_equilibration_built
                                        ),
                                        col_equilibration_metadata=(
                                            xblock_col_equilibration_metadata
                                        ),
                                    )
                                )
                            ),
                            xblock_coarse_correction_metadata=(
                                xblock_coarse_correction_diagnostics_from_context(
                                    XBlockCoarseCorrectionDiagnosticsContext(
                                        moment_schur_enabled=moment_schur_enabled,
                                        moment_schur_built=moment_schur_built,
                                        moment_schur_used=moment_schur_used,
                                        moment_schur_reason=moment_schur_reason,
                                        moment_schur_default_blocked_by_compact_factors=(
                                            moment_schur_default_blocked_by_compact_factors
                                        ),
                                        moment_schur_probe_residual_before=(
                                            moment_schur_probe_residual_before
                                        ),
                                        moment_schur_probe_residual_after=(
                                            moment_schur_probe_residual_after
                                        ),
                                        moment_schur_probe_improvement_ratio=(
                                            moment_schur_probe_improvement_ratio
                                        ),
                                        moment_schur_metadata=moment_schur_metadata,
                                        moment_schur_stats=moment_schur_stats,
                                        two_level_enabled=two_level_enabled,
                                        two_level_built=two_level_built,
                                        two_level_metadata=two_level_metadata,
                                        two_level_stats=two_level_stats,
                                        global_coupling_enabled=(
                                            global_coupling_enabled
                                        ),
                                        global_coupling_built=global_coupling_built,
                                        global_coupling_metadata=(
                                            global_coupling_metadata
                                        ),
                                        global_coupling_stats=global_coupling_stats,
                                    )
                                )
                            ),
                            xblock_qi_seed_preconditioner_metadata=(
                                xblock_qi_seed_preconditioner_diagnostics_from_context(
                                    XBlockQISeedPreconditionerDiagnosticsContext(
                                        qi_galerkin_stats=qi_galerkin_stats,
                                        qi_two_level_stats=qi_two_level_stats,
                                        xblock_initial_seed_residual_norm=(
                                            xblock_initial_seed_residual_norm
                                        ),
                                        xblock_initial_seed_residual_ratio=(
                                            xblock_initial_seed_residual_ratio
                                        ),
                                        moment_schur_seed_residual_norm=(
                                            moment_schur_seed_residual_norm
                                        ),
                                        moment_schur_seed_residual_ratio=(
                                            moment_schur_seed_residual_ratio
                                        ),
                                        qi_coarse_seed_residual_before=(
                                            qi_coarse_seed_residual_before
                                        ),
                                        qi_coarse_seed_residual_after=(
                                            qi_coarse_seed_residual_after
                                        ),
                                        qi_coarse_seed_improvement_ratio=(
                                            qi_coarse_seed_improvement_ratio
                                        ),
                                        qi_coarse_seed_reason=qi_coarse_seed_reason,
                                        qi_coarse_seed_labels=qi_coarse_seed_labels,
                                        qi_seed_basis_kind=qi_seed_basis_kind,
                                        qi_galerkin_preconditioner_reason=(
                                            qi_galerkin_preconditioner_reason
                                        ),
                                        qi_galerkin_preconditioner_mode=(
                                            qi_galerkin_preconditioner_mode
                                        ),
                                        qi_galerkin_preconditioner_coarse_shape=(
                                            qi_galerkin_preconditioner_coarse_shape
                                        ),
                                        qi_galerkin_preconditioner_residual_before=(
                                            qi_galerkin_preconditioner_residual_before
                                        ),
                                        qi_galerkin_preconditioner_residual_after=(
                                            qi_galerkin_preconditioner_residual_after
                                        ),
                                        qi_galerkin_preconditioner_improvement_ratio=(
                                            qi_galerkin_preconditioner_improvement_ratio
                                        ),
                                        qi_galerkin_preconditioner_probe_candidates=(
                                            qi_galerkin_preconditioner_probe_candidates
                                        ),
                                        qi_galerkin_preconditioner_selected_index=(
                                            qi_galerkin_preconditioner_selected_index
                                        ),
                                        qi_two_level_preconditioner_reason=(
                                            qi_two_level_preconditioner_reason
                                        ),
                                        qi_two_level_preconditioner_coarse_shape=(
                                            qi_two_level_preconditioner_coarse_shape
                                        ),
                                        qi_two_level_preconditioner_operator_on_basis_shape=(
                                            qi_two_level_preconditioner_operator_on_basis_shape
                                        ),
                                        qi_two_level_preconditioner_coarse_solver=(
                                            qi_two_level_preconditioner_coarse_solver
                                        ),
                                        qi_two_level_preconditioner_augmentation_labels=(
                                            qi_two_level_preconditioner_augmentation_labels
                                        ),
                                        qi_two_level_preconditioner_smoothed_load_metadata=(
                                            qi_two_level_preconditioner_smoothed_load_metadata
                                        ),
                                        qi_two_level_preconditioner_residual_before=(
                                            qi_two_level_preconditioner_residual_before
                                        ),
                                        qi_two_level_preconditioner_residual_after=(
                                            qi_two_level_preconditioner_residual_after
                                        ),
                                        qi_two_level_preconditioner_improvement_ratio=(
                                            qi_two_level_preconditioner_improvement_ratio
                                        ),
                                        qi_two_level_preconditioner_probe_candidates=(
                                            qi_two_level_preconditioner_probe_candidates
                                        ),
                                        qi_two_level_preconditioner_selected_index=(
                                            qi_two_level_preconditioner_selected_index
                                        ),
                                        xblock_initial_seed_used=(
                                            xblock_initial_seed_used
                                        ),
                                        moment_schur_seed_enabled=(
                                            moment_schur_seed_enabled
                                        ),
                                        moment_schur_seed_used=moment_schur_seed_used,
                                        qi_coarse_seed_enabled=(
                                            qi_coarse_seed_enabled
                                        ),
                                        qi_coarse_seed_used=qi_coarse_seed_used,
                                        qi_coarse_seed_rank=qi_coarse_seed_rank,
                                        qi_coarse_seed_candidate_count=(
                                            qi_coarse_seed_candidate_count
                                        ),
                                        qi_coarse_seed_s=qi_coarse_seed_s,
                                        qi_seed_max_candidates=(
                                            qi_seed_max_candidates
                                        ),
                                        qi_seed_max_angular_mode=(
                                            qi_seed_max_angular_mode
                                        ),
                                        qi_galerkin_preconditioner_enabled=(
                                            qi_galerkin_preconditioner_enabled
                                        ),
                                        qi_galerkin_preconditioner_built=(
                                            qi_galerkin_preconditioner_built
                                        ),
                                        qi_galerkin_preconditioner_used=(
                                            qi_galerkin_preconditioner_used
                                        ),
                                        qi_galerkin_preconditioner_rank=(
                                            qi_galerkin_preconditioner_rank
                                        ),
                                        qi_galerkin_preconditioner_candidate_count=(
                                            qi_galerkin_preconditioner_candidate_count
                                        ),
                                        qi_galerkin_preconditioner_coarse_norm=(
                                            qi_galerkin_preconditioner_coarse_norm
                                        ),
                                        qi_galerkin_preconditioner_rcond=(
                                            qi_galerkin_preconditioner_rcond
                                        ),
                                        qi_galerkin_preconditioner_damping=(
                                            qi_galerkin_preconditioner_damping
                                        ),
                                        qi_galerkin_preconditioner_basis_reused_from_seed=(
                                            qi_galerkin_preconditioner_basis_reused_from_seed
                                        ),
                                        qi_galerkin_preconditioner_probe_reduced=(
                                            qi_galerkin_preconditioner_probe_reduced
                                        ),
                                        qi_galerkin_preconditioner_setup_s=(
                                            qi_galerkin_preconditioner_setup_s
                                        ),
                                        qi_two_level_preconditioner_enabled=(
                                            qi_two_level_preconditioner_enabled
                                        ),
                                        qi_two_level_preconditioner_built=(
                                            qi_two_level_preconditioner_built
                                        ),
                                        qi_two_level_preconditioner_used=(
                                            qi_two_level_preconditioner_used
                                        ),
                                        qi_two_level_preconditioner_rank=(
                                            qi_two_level_preconditioner_rank
                                        ),
                                        qi_two_level_preconditioner_candidate_count=(
                                            qi_two_level_preconditioner_candidate_count
                                        ),
                                        qi_two_level_preconditioner_coarse_norm=(
                                            qi_two_level_preconditioner_coarse_norm
                                        ),
                                        qi_two_level_preconditioner_operator_on_basis_norm=(
                                            qi_two_level_preconditioner_operator_on_basis_norm
                                        ),
                                        qi_two_level_preconditioner_residual_augmented=(
                                            qi_two_level_preconditioner_residual_augmented
                                        ),
                                        qi_two_level_preconditioner_rank_before_augmentation=(
                                            qi_two_level_preconditioner_rank_before_augmentation
                                        ),
                                        qi_two_level_preconditioner_residual_augment_max_extra=(
                                            qi_two_level_preconditioner_residual_augment_max_extra
                                        ),
                                        qi_two_level_preconditioner_residual_augment_steps=(
                                            qi_two_level_preconditioner_residual_augment_steps
                                        ),
                                        qi_two_level_preconditioner_residual_augment_include_residuals=(
                                            qi_two_level_preconditioner_residual_augment_include_residuals
                                        ),
                                        qi_two_level_preconditioner_smoothed_load_basis=(
                                            qi_two_level_preconditioner_smoothed_load_basis
                                        ),
                                        qi_two_level_preconditioner_rcond=(
                                            qi_two_level_preconditioner_rcond
                                        ),
                                        qi_two_level_preconditioner_damping=(
                                            qi_two_level_preconditioner_damping
                                        ),
                                        qi_two_level_preconditioner_basis_reused_from_seed=(
                                            qi_two_level_preconditioner_basis_reused_from_seed
                                        ),
                                        qi_two_level_preconditioner_setup_s=(
                                            qi_two_level_preconditioner_setup_s
                                        ),
                                    )
                                )
                            ),
                            xblock_qi_device_preconditioner_metadata=(
                                xblock_qi_device_preconditioner_diagnostics_from_context(
                                    XBlockQIDevicePreconditionerDiagnosticsContext(
                                        qi_device_preconditioner_enabled=(
                                            qi_device_preconditioner_enabled
                                        ),
                                        qi_device_preconditioner_built=(
                                            qi_device_preconditioner_built
                                        ),
                                        qi_device_preconditioner_used=(
                                            qi_device_preconditioner_used
                                        ),
                                        qi_device_preconditioner_used_in_krylov=(
                                            qi_device_preconditioner_used_in_krylov
                                        ),
                                        qi_device_preconditioner_reason=(
                                            qi_device_preconditioner_reason
                                        ),
                                        qi_device_preconditioner_rank=(
                                            qi_device_preconditioner_rank
                                        ),
                                        qi_device_preconditioner_candidate_count=(
                                            qi_device_preconditioner_candidate_count
                                        ),
                                        qi_device_preconditioner_coarse_shape=(
                                            qi_device_preconditioner_coarse_shape
                                        ),
                                        qi_device_preconditioner_operator_on_basis_shape=(
                                            qi_device_preconditioner_operator_on_basis_shape
                                        ),
                                        qi_device_preconditioner_coarse_norm=(
                                            qi_device_preconditioner_coarse_norm
                                        ),
                                        qi_device_preconditioner_operator_on_basis_norm=(
                                            qi_device_preconditioner_operator_on_basis_norm
                                        ),
                                        qi_device_preconditioner_residual_before=(
                                            qi_device_preconditioner_residual_before
                                        ),
                                        qi_device_preconditioner_residual_after=(
                                            qi_device_preconditioner_residual_after
                                        ),
                                        qi_device_preconditioner_improvement_ratio=(
                                            qi_device_preconditioner_improvement_ratio
                                        ),
                                        qi_device_preconditioner_setup_s=(
                                            qi_device_preconditioner_setup_s
                                        ),
                                        qi_device_preconditioner_min_improvement=(
                                            qi_device_preconditioner_min_improvement
                                        ),
                                        qi_device_preconditioner_use_in_krylov=(
                                            qi_device_preconditioner_use_in_krylov
                                        ),
                                        qi_device_augmented_krylov_requested=(
                                            qi_device_augmented_krylov_requested
                                        ),
                                        qi_device_augmented_krylov_used=(
                                            qi_device_augmented_krylov_used
                                        ),
                                        qi_device_augmented_krylov_rank=(
                                            qi_device_augmented_krylov_rank
                                        ),
                                        qi_device_augmented_krylov_reason=(
                                            qi_device_augmented_krylov_reason
                                        ),
                                        qi_device_augmented_krylov_mode=(
                                            qi_device_augmented_krylov_mode
                                        ),
                                        qi_device_augmented_seed_requested=(
                                            qi_device_augmented_seed_requested
                                        ),
                                        qi_device_augmented_seed_available=(
                                            qi_device_augmented_seed_available
                                        ),
                                        qi_device_augmented_seed_used=(
                                            qi_device_augmented_seed_used
                                        ),
                                        qi_device_augmented_seed_rank=(
                                            qi_device_augmented_seed_rank
                                        ),
                                        qi_device_augmented_seed_max_rank=(
                                            qi_device_augmented_seed_max_rank
                                        ),
                                        qi_device_augmented_seed_reason=(
                                            qi_device_augmented_seed_reason
                                        ),
                                        qi_device_augmented_seed_projection_residual=(
                                            qi_device_augmented_seed_projection_residual
                                        ),
                                        qi_device_augmented_seed_labels=(
                                            qi_device_augmented_seed_labels
                                        ),
                                        qi_device_preconditioner_metadata=(
                                            qi_device_preconditioner_metadata
                                        ),
                                        qi_device_stats=qi_device_stats,
                                    )
                                )
                            ),
                            xblock_qi_deflated_preconditioner_metadata=(
                                xblock_qi_deflated_preconditioner_diagnostics_from_context(
                                    XBlockQIDeflatedPreconditionerDiagnosticsContext(
                                        qi_deflated_preconditioner_enabled=(
                                            qi_deflated_preconditioner_enabled
                                        ),
                                        qi_deflated_preconditioner_built=(
                                            qi_deflated_preconditioner_built
                                        ),
                                        qi_deflated_preconditioner_used=(
                                            qi_deflated_preconditioner_used
                                        ),
                                        qi_deflated_preconditioner_used_in_krylov=(
                                            qi_deflated_preconditioner_used_in_krylov
                                        ),
                                        qi_deflated_preconditioner_reason=(
                                            qi_deflated_preconditioner_reason
                                        ),
                                        qi_deflated_preconditioner_rank=(
                                            qi_deflated_preconditioner_rank
                                        ),
                                        qi_deflated_preconditioner_candidate_count=(
                                            qi_deflated_preconditioner_candidate_count
                                        ),
                                        qi_deflated_preconditioner_residual_before=(
                                            qi_deflated_preconditioner_residual_before
                                        ),
                                        qi_deflated_preconditioner_residual_after=(
                                            qi_deflated_preconditioner_residual_after
                                        ),
                                        qi_deflated_preconditioner_improvement_ratio=(
                                            qi_deflated_preconditioner_improvement_ratio
                                        ),
                                        qi_deflated_preconditioner_setup_s=(
                                            qi_deflated_preconditioner_setup_s
                                        ),
                                        qi_deflated_stats=qi_deflated_stats,
                                        qi_deflated_preconditioner_metadata=(
                                            qi_deflated_preconditioner_metadata
                                        ),
                                    )
                                )
                            ),
                            xblock_side_probe_metadata=xblock_side_probe_diagnostics(
                                XBlockSideProbeDiagnosticsContext(
                                    enabled=xblock_side_probe_enabled,
                                    used=xblock_side_probe_used,
                                    switched=xblock_side_probe_switched,
                                    switch_suppressed_by_global_coupling=(
                                        xblock_side_probe_switch_suppressed_by_global_coupling
                                    ),
                                    switch_suppressed_by_explicit_side=(
                                        xblock_side_probe_switch_suppressed_by_explicit_side
                                    ),
                                    physical_seed_preserved_after_switch=(
                                        xblock_side_probe_physical_seed_preserved_after_switch
                                    ),
                                    seed_used=xblock_side_probe_seed_used,
                                    seed_residual_norm=(
                                        xblock_side_probe_seed_residual_norm
                                    ),
                                    initial_side=xblock_side_probe_initial_side,
                                    selected_side=xblock_side_probe_selected_side,
                                    initial_method=xblock_side_probe_initial_method,
                                    selected_method=xblock_side_probe_selected_method,
                                    lgmres_rescue=xblock_side_probe_lgmres_rescue,
                                    lgmres_rescue_maxiter_capped=(
                                        xblock_lgmres_rescue_maxiter_capped
                                    ),
                                    lgmres_rescue_outer_k=(
                                        xblock_lgmres_rescue_outer_k
                                    ),
                                    residual_norm=xblock_side_probe_residual_norm,
                                    residual_ratio=xblock_side_probe_residual_ratio,
                                    iterations=xblock_side_probe_iterations,
                                    matvecs=xblock_side_probe_matvecs,
                                    elapsed_s=xblock_side_probe_s,
                                )
                            ),
                        ),
                    )
                )
            )
            xblock_sparse_pc_final_payload = build_xblock_sparse_pc_final_payload(
                XBlockSparsePCFinalPayloadContext(
                    op=op,
                    x=np.asarray(x_np, dtype=np.float64),
                    residual_norm=float(residual_norm_xblock_pc),
                    target=float(target_xblock),
                    krylov_method=str(xblock_krylov_method),
                    linear_size=int(xblock_linear_size),
                    restart=int(pc_restart),
                    diagnostic_state=xblock_final_metadata_state,
                    post_corrections=post_corrections,
                ),
                expand_reduced=_xblock_expand_reduced,
            )
            return v3_linear_solve_result_from_payload(
                op=op,
                rhs=rhs,
                payload=xblock_sparse_pc_final_payload,
            )

        sparse_pc_active_setup = build_sparse_pc_active_dof_setup(
            op=op,
            rhs=rhs,
            sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
            active_dof_indices=_transport_active_dof_indices,
            reduce_full_with_indices=reduce_full_with_indices,
            expand_reduced_with_map=expand_reduced_with_map,
        )
        sparse_pc_active_idx_np = sparse_pc_active_setup.active_idx_np
        sparse_pc_active_idx_jnp = sparse_pc_active_setup.active_idx_jnp
        sparse_pc_full_to_active_jnp = sparse_pc_active_setup.full_to_active_jnp
        sparse_pc_rhs = sparse_pc_active_setup.rhs
        sparse_pc_linear_size = int(sparse_pc_active_setup.linear_size)
        _sparse_pc_reduce_full = sparse_pc_active_setup.reduce_full
        _sparse_pc_expand_reduced = sparse_pc_active_setup.expand_reduced
        if emit is not None:
            for level, message in sparse_pc_active_setup.messages:
                emit(level, message)

        if fortran_reduced_sparse_pc:
            op_pc = _build_rhsmode1_preconditioner_operator_fortran_reduced(
                op,
                preconditioner_x=preconditioner_x,
                preconditioner_xi=preconditioner_xi,
                preconditioner_species=preconditioner_species,
                preconditioner_x_min_l=preconditioner_x_min_l,
            )
            sparse_pc_preconditioner_operator = "fortran_reduced_global"
            pattern_source_op = op_pc
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
                    "using global angular-coupled RHSMode=1 preconditioner operator "
                    f"(preconditioner_x={int(preconditioner_x)} "
                    f"preconditioner_x_min_L={int(preconditioner_x_min_l)} "
                    f"preconditioner_xi={int(preconditioner_xi)} "
                    f"preconditioner_species={int(preconditioner_species)})",
                )
        else:
            op_pc = _build_rhsmode1_preconditioner_operator_point(op)
            sparse_pc_preconditioner_operator = "point"
            pattern_source_op = op

        fortran_reduced_backend_setup = resolve_fortran_reduced_sparse_pc_backend(
            op=op,
            env=os.environ,
            fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
            sparse_pc_linear_size=int(sparse_pc_linear_size),
        )
        fortran_reduced_xblock_min_size = (
            fortran_reduced_backend_setup.xblock_min_size
        )
        fortran_reduced_sparse_pc_backend = fortran_reduced_backend_setup.backend
        fortran_reduced_sparse_pc_backend_reason = fortran_reduced_backend_setup.reason
        if emit is not None:
            for level, message in fortran_reduced_backend_setup.messages:
                emit(level, message)

        if bool(fortran_reduced_sparse_pc) and str(fortran_reduced_sparse_pc_backend) == "xblock":
            if op_pc.fblock.fp is None or op_pc.fblock.pas is not None:
                raise NotImplementedError(
                    "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND=xblock currently targets "
                    "full-FP RHSMode=1 systems."
                )

            xblock_factor_build = build_fortran_reduced_xblock_factor_stage(
                context=FortranReducedXBlockFactorBuildContext(
                    op_pc=op_pc,
                    reduce_full=_sparse_pc_reduce_full,
                    expand_reduced=_sparse_pc_expand_reduced,
                    preconditioner_species=int(preconditioner_species),
                    preconditioner_xi=int(preconditioner_xi),
                    sparse_pc_linear_size=int(sparse_pc_linear_size),
                    backend_reason=str(fortran_reduced_sparse_pc_backend_reason),
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    env=os.environ,
                    assembled_host_allowed=_rhsmode1_fp_xblock_assembled_host_allowed,
                    builder=_build_rhsmode1_xblock_tz_sparse_preconditioner,
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
                    rhs=rhs,
                    xblock_use_active_dof=bool(sparse_pc_use_active_dof),
                    active_idx=sparse_pc_active_idx_jnp,
                    full_to_active=sparse_pc_full_to_active_jnp,
                    reduce_full_with_indices=reduce_full_with_indices,
                    expand_reduced_with_map=expand_reduced_with_map,
                    operator_matvec=lambda x_full: apply_v3_full_system_operator_cached(op, x_full),
                    base_preconditioner=precond_xblock,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    env=os.environ,
                )
            )
            side_env = xblock_krylov_setup.side_env
            precondition_side = xblock_krylov_setup.precondition_side
            pc_form = xblock_krylov_setup.pc_form
            xblock_krylov_method = xblock_krylov_setup.krylov_method
            progress_every = xblock_krylov_setup.progress_every
            mv_count = xblock_krylov_setup.mv_count
            _mv_true_no_count = xblock_krylov_setup.matvec_no_count
            _mv_true = xblock_krylov_setup.matvec
            precond_xblock_krylov = xblock_krylov_setup.preconditioner
            moment_schur_policy = resolve_fortran_reduced_xblock_moment_schur_policy(
                precondition_side=precondition_side,
                env=os.environ,
            )
            moment_schur_enabled = bool(moment_schur_policy.enabled)
            moment_schur_result = apply_fortran_reduced_xblock_moment_schur_stage(
                context=FortranReducedXBlockMomentSchurStageContext(
                    op=op,
                    base_preconditioner=precond_xblock_krylov,
                    reduce_full=_sparse_pc_reduce_full if sparse_pc_use_active_dof else None,
                    expand_reduced=_sparse_pc_expand_reduced if sparse_pc_use_active_dof else None,
                    policy=moment_schur_policy,
                    precondition_side=str(precondition_side),
                    rhs=sparse_pc_rhs,
                    matvec_no_count=_mv_true_no_count,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    builder=_build_rhs1_xblock_constraint1_moment_schur_preconditioner,
                )
            )
            precond_xblock_krylov = moment_schur_result.preconditioner
            moment_schur_built = bool(moment_schur_result.built)
            moment_schur_used = bool(moment_schur_result.used)
            moment_schur_reason = moment_schur_result.reason
            moment_schur_metadata = moment_schur_result.metadata
            moment_schur_stats = moment_schur_result.stats
            moment_schur_probe_residual_before = moment_schur_result.probe_residual_before
            moment_schur_probe_residual_after = moment_schur_result.probe_residual_after
            moment_schur_probe_improvement_ratio = moment_schur_result.probe_improvement_ratio
            pc_factor_s += float(moment_schur_result.setup_s)

            global_coupling_policy = (
                resolve_fortran_reduced_xblock_global_coupling_policy(
                    precondition_side=precondition_side,
                    env=os.environ,
                )
            )
            global_coupling_enabled = bool(global_coupling_policy.enabled)
            global_coupling_result = apply_fortran_reduced_xblock_global_coupling_stage(
                context=FortranReducedXBlockGlobalCouplingStageContext(
                    op=op,
                    rhs=rhs,
                    matvec=_mv_true_no_count,
                    base_preconditioner=precond_xblock_krylov,
                    direction_projector=_sparse_pc_reduce_full if sparse_pc_use_active_dof else None,
                    expected_size=int(sparse_pc_linear_size),
                    policy=global_coupling_policy,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    builder=_build_rhs1_xblock_smoothed_global_coupling_preconditioner,
                )
            )
            precond_xblock_krylov = global_coupling_result.preconditioner
            global_coupling_built = bool(global_coupling_result.built)
            global_coupling_metadata = global_coupling_result.metadata
            global_coupling_stats = global_coupling_result.stats
            pc_factor_s += float(global_coupling_result.setup_s)

            fortran_x0_setup = prepare_fortran_reduced_xblock_initial_guess(
                x0=x0,
                sparse_pc_rhs=sparse_pc_rhs,
                full_rhs=rhs,
                reduce_full=_sparse_pc_reduce_full,
            )
            x0_sparse = fortran_x0_setup.x0_full
            if emit is not None:
                for level, message in fortran_x0_setup.messages:
                    emit(level, message)

            sparse_pc_rhs_norm = rhs1_l2_norm_float(sparse_pc_rhs)
            target = rhs1_residual_target(
                atol=float(atol),
                tol=float(tol),
                rhs_norm=float(sparse_pc_rhs_norm),
            )
            initial_seed_policy = resolve_fortran_reduced_xblock_initial_seed_policy(
                env=os.environ,
            )
            seed_enabled = bool(initial_seed_policy.enabled)
            seed_residual_norm: float | None = None
            seed_improvement_ratio: float | None = None
            seed_refines_performed = 0
            seed_refine_steps = int(initial_seed_policy.refine_steps)
            seed_accept_ratio = float(initial_seed_policy.accept_ratio)
            seed_used = False
            initial_seed_result = apply_fortran_reduced_xblock_initial_seed(
                policy=initial_seed_policy,
                rhs=sparse_pc_rhs,
                rhs_norm=float(sparse_pc_rhs_norm),
                x0=x0_sparse,
                preconditioner=precond_xblock_krylov,
                matvec_no_count=_mv_true_no_count,
                elapsed_s=sparse_timer.elapsed_s,
            )
            x0_sparse = initial_seed_result.x0
            seed_used = bool(initial_seed_result.used)
            seed_residual_norm = initial_seed_result.residual_norm
            seed_improvement_ratio = initial_seed_result.improvement_ratio
            seed_refines_performed = int(initial_seed_result.refines_performed)
            if emit is not None:
                for level, message in initial_seed_result.messages:
                    emit(level, message)
            xblock_krylov_result = run_fortran_reduced_xblock_krylov_solve(
                context=FortranReducedXBlockKrylovSolveContext(
                    matvec=_mv_true,
                    rhs=sparse_pc_rhs,
                    preconditioner=precond_xblock_krylov,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    method=str(xblock_krylov_method),
                    pc_form=str(pc_form),
                    restart=int(pc_restart),
                    maxiter=int(pc_maxiter),
                    tol=float(tol),
                    atol=float(atol),
                    target=float(target),
                    precondition_side=str(precondition_side),
                    progress_every=int(progress_every),
                    mv_count=mv_count,
                    explicit_left_solver=explicit_left_preconditioned_gmres_scipy,
                    gmres_solver=gmres_solve_with_history_scipy,
                    lgmres_solver=lgmres_solve_with_history_scipy,
                    gcrotmk_solver=gcrotmk_solve_with_history_scipy,
                    bicgstab_solver=bicgstab_solve_with_history_scipy,
                ),
                x0=x0_sparse,
            )
            fortran_reduced_xblock_payload = (
                fortran_reduced_xblock_final_payload(
                    FortranReducedXBlockFinalPayloadContext(
                        diagnostic_state={
                            "op": op,
                            "fortran_reduced_sparse_pc_backend_reason": fortran_reduced_sparse_pc_backend_reason,
                            "fortran_reduced_xblock_min_size": fortran_reduced_xblock_min_size,
                            "preconditioner_x": preconditioner_x,
                            "preconditioner_x_min_l": preconditioner_x_min_l,
                            "preconditioner_xi": preconditioner_xi,
                            "preconditioner_species": preconditioner_species,
                            "xblock_preconditioner_xi": xblock_preconditioner_xi,
                            "force_assembled_host_fp": force_assembled_host_fp,
                            "xblock_krylov_method": xblock_krylov_method,
                            "seed_enabled": seed_enabled,
                            "seed_used": seed_used,
                            "seed_residual_norm": seed_residual_norm,
                            "seed_improvement_ratio": seed_improvement_ratio,
                            "seed_accept_ratio": seed_accept_ratio,
                            "seed_refine_steps": seed_refine_steps,
                            "seed_refines_performed": seed_refines_performed,
                            "moment_schur_enabled": moment_schur_enabled,
                            "moment_schur_built": moment_schur_built,
                            "moment_schur_used": moment_schur_used,
                            "moment_schur_reason": moment_schur_reason,
                            "moment_schur_metadata": moment_schur_metadata,
                            "moment_schur_stats": moment_schur_stats,
                            "moment_schur_probe_residual_before": (
                                moment_schur_probe_residual_before
                            ),
                            "moment_schur_probe_residual_after": (
                                moment_schur_probe_residual_after
                            ),
                            "moment_schur_probe_improvement_ratio": (
                                moment_schur_probe_improvement_ratio
                            ),
                            "global_coupling_enabled": global_coupling_enabled,
                            "global_coupling_built": global_coupling_built,
                            "global_coupling_metadata": global_coupling_metadata,
                            "global_coupling_stats": global_coupling_stats,
                            "xblock_drop_tol": xblock_drop_tol,
                            "xblock_drop_rel": xblock_drop_rel,
                            "xblock_ilu_drop_tol": xblock_ilu_drop_tol,
                            "xblock_fill_factor": xblock_fill_factor,
                            "sparse_pc_use_active_dof": sparse_pc_use_active_dof,
                            "sparse_pc_linear_size": sparse_pc_linear_size,
                            "sparse_pc_fp_dense_velocity_block": sparse_pc_fp_dense_velocity_block,
                            "setup_s": setup_s,
                            "solve_s": float(xblock_krylov_result.solve_s),
                            "sparse_timer": sparse_timer,
                            "pc_factor_s": pc_factor_s,
                            "target": target,
                            "mv_count": mv_count,
                            "pc_restart": pc_restart,
                            "pc_maxiter": pc_maxiter,
                            "history": tuple(xblock_krylov_result.history),
                            "residual_norm_sparse_pc": float(
                                xblock_krylov_result.residual_norm
                            ),
                        },
                        result=xblock_krylov_result,
                        atol=float(atol),
                        tol=float(tol),
                        rhs_norm=float(rhs_norm),
                        target=float(target),
                    ),
                    expand_reduced=_sparse_pc_expand_reduced,
                )
            )
            return v3_linear_solve_result_from_payload(
                op=op,
                rhs=rhs,
                payload=fortran_reduced_xblock_payload,
            )

        sparse_pc_pattern_setup = build_sparse_pc_pattern_setup(
            SparsePCPatternSetupContext(
                op=op,
                pattern_source_op=pattern_source_op,
                fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                active_idx_np=sparse_pc_active_idx_np,
                preconditioner_x=int(preconditioner_x),
                preconditioner_xi=int(preconditioner_xi),
                preconditioner_species=int(preconditioner_species),
                preconditioner_x_min_l=int(preconditioner_x_min_l),
                fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                fortran_reduced_pattern_for_indices=(
                    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices
                ),
                fortran_reduced_pattern=v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
                conservative_pattern_for_indices=v3_full_system_conservative_sparsity_pattern_for_indices,
                conservative_pattern=v3_full_system_conservative_sparsity_pattern,
                summarize_pattern=summarize_v3_sparse_pattern,
            )
        )
        pattern = sparse_pc_pattern_setup.pattern
        sparse_pattern_scope = str(sparse_pc_pattern_setup.scope)
        pattern_build_s = float(sparse_pc_pattern_setup.build_s)
        summary = sparse_pc_pattern_setup.summary
        sparse_pc_factor_policy = resolve_sparse_pc_factor_policy(
            env=os.environ,
            constrained_pas_pc=bool(constrained_pas_pc),
            tokamak_fp_pc=bool(tokamak_fp_pc),
            fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
            sparse_pc_linear_size=int(sparse_pc_linear_size),
            pc_maxiter=int(pc_maxiter),
            default_permc_spec=_rhsmode1_sparse_pc_default_permc_spec(
                constrained_pas_pc=bool(constrained_pas_pc),
                tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
                n_species=int(op.n_species),
            ),
            host_sparse_factor_dtype=_host_sparse_factor_dtype,
        )
        pc_shift = float(sparse_pc_factor_policy.pc_shift)
        sparse_pc_factorization = str(sparse_pc_factor_policy.factorization)
        sparse_pc_default_factor_kind = str(sparse_pc_factor_policy.default_factor_kind)
        sparse_pc_default_ilu_fill_factor = float(sparse_pc_factor_policy.default_ilu_fill_factor)
        sparse_pc_default_ilu_drop_tol = float(sparse_pc_factor_policy.default_ilu_drop_tol)
        sparse_pc_default_pattern_color_batch = int(sparse_pc_factor_policy.default_pattern_color_batch)
        sparse_pc_factor_dtype_initial = np.dtype(sparse_pc_factor_policy.factor_dtype_initial)
        sparse_pc_factor_dtype_used = np.dtype(sparse_pc_factor_policy.factor_dtype_used)
        sparse_pc_factor_dtype_retry = sparse_pc_factor_policy.factor_dtype_retry
        sparse_pc_default_permc_spec = str(sparse_pc_factor_policy.default_permc_spec)
        sparse_pc_permc_spec = str(sparse_pc_factor_policy.permc_spec)
        fp32_probe_maxiter = int(sparse_pc_factor_policy.fp32_probe_maxiter)
        sparse_pc_first_attempt_maxiter = int(sparse_pc_factor_policy.first_attempt_maxiter)
        enforce_sparse_pc_memory_budget(
            SparsePCMemoryBudgetPreflightContext(
                env=os.environ,
                unknowns=int(sparse_pc_linear_size),
                gmres_restart=int(pc_restart),
                csr_nnz=int(summary.nnz),
                dtype=sparse_pc_factor_dtype_initial,
                device_count=max(1, int(jax.device_count())),
                estimate_sparse_pc_memory=estimate_sparse_pc_memory,
            )
        )

        def _sparse_pc_factor_mv(x_np: np.ndarray) -> jnp.ndarray:
            x_jnp = _sparse_pc_expand_reduced(jnp.asarray(x_np, dtype=rhs.dtype))
            y_jnp = apply_v3_full_system_operator_cached(op_pc, x_jnp)
            if pc_shift != 0.0:
                y_jnp = y_jnp + jnp.asarray(pc_shift, dtype=rhs.dtype) * x_jnp
            return _sparse_pc_reduce_full(y_jnp)

        if emit is not None:
            shift_note = f" shift={pc_shift:.1e}" if pc_shift != 0.0 else ""
            emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres factoring RHSMode=1 preconditioner"
                f"{shift_note} factor_dtype={sparse_pc_factor_dtype_initial.name} "
                f"factor_kind={sparse_pc_factorization} permc={sparse_pc_permc_spec}",
            )
        direct_tail_materialization = build_direct_tail_materialization_setup(
            DirectTailMaterializationContext(
                env=os.environ,
                op=op,
                op_pc=op_pc,
                pattern=pattern,
                active_indices=sparse_pc_active_idx_np,
                sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                reduce_full=_sparse_pc_reduce_full,
                expand_reduced=_sparse_pc_expand_reduced,
                pc_shift=float(pc_shift),
                dtype=rhs.dtype,
                factor_dtype=sparse_pc_factor_dtype_initial,
                sparse_pc_linear_size=int(sparse_pc_linear_size),
                default_pattern_color_batch=int(sparse_pc_default_pattern_color_batch),
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                is_direct_reduced_pmat_pc_kind=_is_direct_reduced_pmat_pc_kind,
                build_direct_tail_bundle=_try_build_fortran_reduced_constraint1_direct_tail_bundle,
                build_structured_rhs1_full_csr_operator_bundle_callback=(
                    _try_build_structured_rhs1_full_csr_operator_bundle
                ),
            )
        )
        direct_tail_default = bool(direct_tail_materialization.direct_tail_default)
        direct_tail_enabled = bool(direct_tail_materialization.enabled)
        direct_tail_built = bool(direct_tail_materialization.built)
        direct_tail_error = direct_tail_materialization.error
        direct_tail_operator_bundle = direct_tail_materialization.operator_bundle
        direct_tail_structured_pc_requested: str | None = None
        direct_tail_structured_pc_selected = False
        direct_tail_structured_pc_reason: str | None = None
        direct_tail_structured_pc_metadata: dict[str, object] | None = None
        direct_tail_structured_pc_error: str | None = None
        direct_tail_pc_env_early = str(direct_tail_materialization.pc_env)
        direct_tail_direct_reduced_pmat_requested = bool(
            direct_tail_materialization.direct_reduced_pmat_requested
        )
        factor_start_s = sparse_timer.elapsed_s()
        direct_tail_structured_admission = resolve_direct_tail_structured_admission(
            DirectTailStructuredAdmissionContext(
                env=os.environ,
                pc_env=str(direct_tail_materialization.pc_env),
                operator_bundle=direct_tail_operator_bundle,
                direct_reduced_pmat_requested=bool(direct_tail_direct_reduced_pmat_requested),
                sparse_pc_linear_size=int(sparse_pc_linear_size),
                default_max_mb=_rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb,
            )
        )
        direct_tail_pc_env = str(direct_tail_structured_admission.pc_env)
        direct_tail_pc_auto_default = bool(direct_tail_structured_admission.auto_default)
        direct_tail_structured_pc_requested = direct_tail_structured_admission.requested
        direct_tail_fail_closed_size = int(direct_tail_structured_admission.fail_closed_size)
        direct_tail_auto_large_fail_closed = bool(direct_tail_structured_admission.auto_large_fail_closed)
        direct_tail_structured_pc_required = bool(direct_tail_structured_admission.required)
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor setup start "
                f"size={int(sparse_pc_linear_size)} "
                f"factor_dtype={sparse_pc_factor_dtype_initial.name} "
                f"factor_kind={sparse_pc_factorization} direct_tail_built={bool(direct_tail_built)} "
                f"structured_pc_requested={direct_tail_structured_pc_requested}",
            )
        structured_pc_ready = False
        direct_tail_structured_layout: RHS1BlockLayout | None = None
        direct_tail_structured_active_indices: np.ndarray | None = None
        direct_tail_structured_max_nbytes: int | None = None
        direct_tail_support_mode_preflight_requested = False
        direct_tail_support_mode_preflight_selected = False
        direct_tail_support_mode_preflight_metadata: dict[str, object] | None = None
        direct_tail_support_mode_preflight_error: str | None = None
        direct_tail_structured_pc_max_mb_auto = bool(direct_tail_structured_admission.max_mb_auto)
        pc_max_mb = float(direct_tail_structured_admission.max_mb)
        pc_reg = float(direct_tail_structured_admission.regularization)
        if bool(direct_tail_structured_admission.setup_allowed):
            direct_tail_structured_pc_start_s = sparse_timer.elapsed_s()
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    "structured preconditioner setup start "
                    f"kind={direct_tail_structured_pc_requested} "
                    f"active_size={int(sparse_pc_linear_size)} "
                    f"max_mb={float(pc_max_mb):.3g} "
                    f"max_mb_auto={bool(direct_tail_structured_pc_max_mb_auto)} "
                    f"reg={float(pc_reg):.3e}",
                )
            try:
                direct_tail_structured_build = build_direct_tail_structured_preconditioner_setup(
                    DirectTailStructuredBuildContext(
                        env=os.environ,
                        op=op_pc,
                        operator_bundle=direct_tail_operator_bundle,
                        active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                        requested_kind=direct_tail_structured_pc_requested,
                        direct_reduced_pmat_requested=bool(direct_tail_direct_reduced_pmat_requested),
                        sparse_pc_linear_size=int(sparse_pc_linear_size),
                        max_mb=float(pc_max_mb),
                        regularization=float(pc_reg),
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                        layout_from_operator=RHS1BlockLayout.from_operator,
                        build_direct_active_preconditioner=(
                            build_direct_active_fortran_v3_reduced_pmat_preconditioner
                        ),
                        build_active_projected_preconditioner=(
                            build_active_projected_rhs1_full_csr_preconditioner
                        ),
                        cache=_DIRECT_TAIL_STRUCTURED_PC_CACHE,
                        cache_key=_direct_tail_structured_pc_cache_key,
                        with_cache_metadata=_direct_tail_structured_pc_with_cache_metadata,
                        factor_bundle=_StructuredHostSparsePreconditionerBundle,
                    )
                )
                direct_tail_structured_layout = direct_tail_structured_build.layout
                direct_tail_structured_active_indices = direct_tail_structured_build.active_indices
                direct_tail_structured_max_nbytes = direct_tail_structured_build.max_nbytes
                direct_tail_structured_pc_selected = bool(direct_tail_structured_build.selected)
                direct_tail_structured_pc_reason = direct_tail_structured_build.reason
                direct_tail_structured_pc_metadata = direct_tail_structured_build.metadata
                direct_tail_structured_pc_error = direct_tail_structured_build.error
                direct_tail_structured_pc_cache_hit = bool(direct_tail_structured_build.cache_hit)
                if bool(direct_tail_structured_pc_cache_hit) and emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "structured preconditioner cache hit "
                        f"elapsed_s={sparse_timer.elapsed_s() - direct_tail_structured_pc_start_s:.3f}",
                    )
                if bool(direct_tail_structured_build.ready):
                    factor_bundle_pc = direct_tail_structured_build.factor_bundle
                    _operator_bundle_pc = direct_tail_structured_build.operator_bundle_pc
                    structured_pc_ready = True
                    if emit is not None:
                        structured_pc_metadata_inner = {}
                        if isinstance(direct_tail_structured_pc_metadata, dict):
                            maybe_inner = direct_tail_structured_pc_metadata.get("metadata")
                            if isinstance(maybe_inner, dict):
                                structured_pc_metadata_inner = maybe_inner
                        factor_nbytes = structured_pc_metadata_inner.get("factor_nbytes_actual")
                        if factor_nbytes is None:
                            factor_nbytes = structured_pc_metadata_inner.get("factor_nbytes_estimate")
                        factor_permc = structured_pc_metadata_inner.get("permc_spec", "na")
                        factor_superlu_permc = structured_pc_metadata_inner.get("superlu_permc_spec", "na")
                        pc_kind = (
                            str(direct_tail_structured_pc_metadata.get("kind", direct_tail_structured_pc_requested))
                            if isinstance(direct_tail_structured_pc_metadata, dict)
                            else str(direct_tail_structured_pc_requested)
                        )
                        pc_setup_s = (
                            float(direct_tail_structured_pc_metadata.get("setup_s", 0.0) or 0.0)
                            if isinstance(direct_tail_structured_pc_metadata, dict)
                            else 0.0
                        )
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                            f"structured preconditioner selected kind={pc_kind} "
                            f"setup_s={float(pc_setup_s):.3f} "
                            f"elapsed_s={sparse_timer.elapsed_s() - direct_tail_structured_pc_start_s:.3f} "
                            f"reason={direct_tail_structured_pc_reason} "
                            f"cache_hit={bool(direct_tail_structured_pc_cache_hit)} "
                            f"factor_nbytes={factor_nbytes if factor_nbytes is not None else 'na'} "
                            f"permc={factor_permc} superlu_permc={factor_superlu_permc}",
                        )
                elif emit is not None:
                    tail_action = (
                        "required path will fail fast"
                        if bool(direct_tail_structured_pc_required)
                        else "falling back to host factorization"
                    )
                    if direct_tail_structured_pc_error is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                            "structured preconditioner failed "
                            f"elapsed_s={sparse_timer.elapsed_s() - direct_tail_structured_pc_start_s:.3f} "
                            f"({direct_tail_structured_pc_error}); {tail_action}",
                        )
                    else:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                            "structured preconditioner not selected "
                            f"kind={direct_tail_structured_pc_requested} reason={direct_tail_structured_pc_reason}; "
                            f"elapsed_s={sparse_timer.elapsed_s() - direct_tail_structured_pc_start_s:.3f}; "
                            f"{tail_action}",
                        )
            except Exception as exc:  # noqa: BLE001
                direct_tail_structured_pc_error = f"{type(exc).__name__}: {exc}"
                direct_tail_structured_pc_selected = False
                direct_tail_structured_pc_reason = "structured_pc_exception"
                if emit is not None:
                    tail_action = (
                        "required path will fail fast"
                        if bool(direct_tail_structured_pc_required)
                        else "falling back to host factorization"
                    )
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "structured preconditioner failed "
                        f"elapsed_s={sparse_timer.elapsed_s() - direct_tail_structured_pc_start_s:.3f} "
                        f"({direct_tail_structured_pc_error}); {tail_action}",
                    )
        if (
            direct_tail_structured_pc_requested is not None
            and bool(direct_tail_structured_pc_required)
            and not bool(structured_pc_ready)
        ):
            raise RuntimeError(
                "direct-tail structured preconditioner was explicitly requested but not selected: "
                f"kind={direct_tail_structured_pc_requested} "
                f"reason={direct_tail_structured_pc_reason} "
                f"error={direct_tail_structured_pc_error} "
                f"direct_tail_built={bool(direct_tail_built)} "
                f"direct_reduced_pmat_requested={bool(direct_tail_direct_reduced_pmat_requested)}"
            )
        if not structured_pc_ready:
            _operator_bundle_pc, factor_bundle_pc = _build_host_sparse_direct_factor_from_matvec(
                matvec=_sparse_pc_factor_mv,
                n=int(sparse_pc_linear_size),
                dtype=rhs.dtype,
                factor_dtype=sparse_pc_factor_dtype_initial,
                pattern=pattern,
                operator_bundle_override=direct_tail_operator_bundle,
                emit=emit,
                default_diag_pivot_thresh=0.0 if (constrained_pas_pc or tokamak_fp_pc or fortran_reduced_sparse_pc) else 1.0,
                default_permc_spec=sparse_pc_default_permc_spec,
                default_factor_kind=sparse_pc_default_factor_kind,
                default_ilu_fill_factor=float(sparse_pc_default_ilu_fill_factor),
                default_ilu_drop_tol=float(sparse_pc_default_ilu_drop_tol),
                default_pattern_color_batch=int(sparse_pc_default_pattern_color_batch),
            )
        pc_factor_s = sparse_timer.elapsed_s() - factor_start_s
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor setup complete "
                f"elapsed_s={float(pc_factor_s):.3f} structured_pc_ready={bool(structured_pc_ready)} "
                f"direct_tail_built={bool(direct_tail_built)}",
            )
        setup_s = sparse_timer.elapsed_s()

        precondition_side = rhs1_gmres_precondition_side_from_env()
        pc_form = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM", "").strip().lower()
        if pc_form not in {"", "scipy_left", "scipy", "explicit_left", "petsc_left"}:
            pc_form = ""
        pc_form = pc_form or "scipy_left"
        progress_every_env = os.environ.get("SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY", "").strip()
        try:
            progress_every = int(progress_every_env) if progress_every_env else 25
        except ValueError:
            progress_every = 25
        progress_every = max(0, int(progress_every))
        mv_count = 0

        def _mv_true_no_count(v: jnp.ndarray) -> jnp.ndarray:
            x_full = _sparse_pc_expand_reduced(jnp.asarray(v, dtype=rhs.dtype))
            y_full = apply_v3_full_system_operator_cached(op, x_full)
            return _sparse_pc_reduce_full(y_full)

        def _mv_true_matmat(cols: np.ndarray) -> np.ndarray:
            cols_np = np.asarray(cols, dtype=np.float64)
            if cols_np.ndim != 2:
                raise ValueError("true matmat columns must be a rank-2 array")
            out = jax.vmap(_mv_true_no_count, in_axes=1, out_axes=1)(jnp.asarray(cols_np, dtype=rhs.dtype))
            return np.asarray(out, dtype=np.float64)

        def _mv_true(v: jnp.ndarray) -> jnp.ndarray:
            nonlocal mv_count
            mv_count += 1
            if emit is not None and progress_every > 0 and mv_count % progress_every == 0:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: sparse_pc_gmres "
                    f"matvecs={int(mv_count)} elapsed_s={sparse_timer.elapsed_s():.3f}",
                )
            return _mv_true_no_count(v)

        def _precond_sparse(v: jnp.ndarray) -> jnp.ndarray:
            v_np = np.asarray(v, dtype=np.float64).reshape((-1,))
            y_np = factor_bundle_pc.solve(v_np)
            return jnp.asarray(y_np, dtype=jnp.float64)

        x0_sparse = None
        if x0 is not None:
            x0_arr = jnp.asarray(x0, dtype=jnp.float64)
            if x0_arr.shape == sparse_pc_rhs.shape:
                x0_sparse = x0_arr
            elif x0_arr.shape == rhs.shape:
                x0_sparse = _sparse_pc_reduce_full(x0_arr)
            elif emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: sparse_pc_gmres ignoring incompatible x0 "
                    f"shape={tuple(x0_arr.shape)} expected={tuple(sparse_pc_rhs.shape)} or {tuple(rhs.shape)}",
                )

        sparse_pc_rhs_norm = rhs1_l2_norm_float(sparse_pc_rhs)
        target = rhs1_residual_target(
            atol=float(atol),
            tol=float(tol),
            rhs_norm=float(sparse_pc_rhs_norm),
        )
        direct_tail_support_mode_preflight_requested = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT",
            default=False,
        )
        if bool(direct_tail_support_mode_preflight_requested):
            factor_kind_for_support = str(getattr(factor_bundle_pc, "kind", "")).strip().lower().replace("-", "_")

            def _support_true_matvec(v_np: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jax.device_get(_mv_true_no_count(jnp.asarray(v_np, dtype=rhs.dtype))),
                    dtype=np.float64,
                ).reshape((-1,))

            support_preflight = run_direct_tail_support_mode_preflight(
                DirectTailSupportModePreflightContext(
                    env=os.environ,
                    factor_kind=factor_kind_for_support,
                    structured_pc_ready=bool(structured_pc_ready),
                    operator_bundle=direct_tail_operator_bundle,
                    layout=direct_tail_structured_layout,
                    active_indices=direct_tail_structured_active_indices,
                    max_nbytes=direct_tail_structured_max_nbytes,
                    regularization=float(pc_reg),
                    rhs=np.asarray(sparse_pc_rhs, dtype=np.float64),
                    true_matvec=_support_true_matvec,
                    preconditioner_x=int(preconditioner_x),
                    preconditioner_xi=int(preconditioner_xi),
                    preconditioner_species=int(preconditioner_species),
                    preconditioner_x_min_l=int(preconditioner_x_min_l),
                    selector=select_active_fortran_v3_reduced_support_mode_preconditioner,
                    factor_bundle=_StructuredHostSparsePreconditionerBundle,
                )
            )
            direct_tail_support_mode_preflight_metadata = support_preflight.metadata
            direct_tail_support_mode_preflight_error = support_preflight.error
            if bool(support_preflight.selected):
                support_pc = support_preflight.preconditioner
                factor_bundle_pc = support_preflight.factor_bundle
                direct_tail_structured_pc_selected = True
                direct_tail_structured_pc_reason = str(getattr(support_pc, "reason", "support_mode_selected"))
                direct_tail_structured_pc_metadata = support_pc.to_dict()
                direct_tail_support_mode_preflight_selected = True
                if emit is not None and isinstance(direct_tail_support_mode_preflight_metadata, dict):
                    selected_candidate = direct_tail_support_mode_preflight_metadata.get("selected_candidate")
                    baseline_after = direct_tail_support_mode_preflight_metadata.get("baseline_residual_after")
                    best_after = direct_tail_support_mode_preflight_metadata.get("best_residual_after")
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "support-mode preflight selected "
                        f"candidate={selected_candidate} "
                        f"baseline_residual={float(baseline_after or float('nan')):.6e} "
                        f"best_residual={float(best_after or float('nan')):.6e} "
                        f"accepted_nonbaseline="
                        f"{bool(direct_tail_support_mode_preflight_metadata.get('accepted_nonbaseline', False))}",
                    )
            elif direct_tail_support_mode_preflight_error is not None and emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    f"support-mode preflight failed ({direct_tail_support_mode_preflight_error}); "
                    "continuing with existing structured preconditioner",
                )
        factor_preflight_policy = resolve_sparse_pc_factor_preflight_policy(
            SparsePCFactorPreflightPolicyContext(
                env=os.environ,
                fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                structured_pc_ready=bool(structured_pc_ready),
                structured_pc_metadata=(
                    direct_tail_structured_pc_metadata
                    if isinstance(direct_tail_structured_pc_metadata, dict)
                    else None
                ),
                sparse_pc_linear_size=int(sparse_pc_linear_size),
            )
        )
        factor_preflight_enabled = bool(factor_preflight_policy.factor_preflight_enabled)
        factor_preflight_required = bool(factor_preflight_policy.factor_preflight_required)
        factor_preflight_seed_enabled = bool(factor_preflight_policy.factor_preflight_seed_enabled)
        structured_pc_preflight_required_min_size = int(
            factor_preflight_policy.structured_pc_preflight_required_min_size
        )
        direct_tail_structured_pc_requires_preflight = bool(
            factor_preflight_policy.direct_tail_structured_pc_requires_preflight
        )
        direct_tail_structured_pc_kind_for_preflight = str(
            factor_preflight_policy.direct_tail_structured_pc_kind_for_preflight
        )
        direct_tail_structured_pc_size_requires_preflight = bool(
            factor_preflight_policy.direct_tail_structured_pc_size_requires_preflight
        )
        structured_pc_preflight_required = bool(factor_preflight_policy.structured_pc_preflight_required)
        factor_preflight_max_target_ratio = float(factor_preflight_policy.factor_preflight_max_target_ratio)
        factor_preflight_residual_before: float | None = None
        factor_preflight_residual_after: float | None = None
        factor_preflight_improvement_ratio: float | None = None
        factor_preflight_target_ratio: float | None = None
        factor_preflight_residual_diagnostics: dict[str, object] | None = None
        factor_preflight_seed_used = False
        factor_preflight_passed: bool | None = None
        factor_preflight_error: str | None = None
        direct_tail_residual_rescue_policy = resolve_direct_tail_residual_rescue_policy(os.environ)
        direct_tail_residual_coarse_requested = bool(
            direct_tail_residual_rescue_policy.residual_coarse_requested
        )
        direct_tail_residual_coarse_selected = False
        direct_tail_residual_coarse_metadata: dict[str, object] | None = None
        direct_tail_residual_coarse_error: str | None = None
        direct_tail_residual_coarse_residual_after: float | None = None
        direct_tail_residual_coarse_rank = int(direct_tail_residual_rescue_policy.residual_coarse_rank)
        direct_tail_residual_coarse_max_mb = float(direct_tail_residual_rescue_policy.residual_coarse_max_mb)
        direct_tail_residual_coarse_regularization = float(
            direct_tail_residual_rescue_policy.residual_coarse_regularization
        )
        direct_tail_residual_window_requested = bool(
            direct_tail_residual_rescue_policy.residual_window_requested
        )
        direct_tail_true_window_requested = bool(direct_tail_residual_rescue_policy.true_window_requested)
        direct_tail_true_coupled_coarse_explicit_requested = bool(
            direct_tail_residual_rescue_policy.true_coupled_coarse_explicit_requested
        )
        direct_tail_true_coupled_coarse_auto_enabled = bool(
            direct_tail_residual_rescue_policy.true_coupled_coarse_auto_enabled
        )
        direct_tail_true_coupled_coarse_auto_native_enabled = bool(
            direct_tail_residual_rescue_policy.true_coupled_coarse_auto_native_enabled
        )
        direct_tail_true_coupled_coarse_auto_target_ratio = float(
            direct_tail_residual_rescue_policy.true_coupled_coarse_auto_target_ratio
        )
        direct_tail_true_coupled_coarse_auto_min_size = int(
            direct_tail_residual_rescue_policy.true_coupled_coarse_auto_min_size
        )
        direct_tail_true_coupled_coarse_requested = bool(direct_tail_true_coupled_coarse_explicit_requested)
        direct_tail_true_coupled_coarse_auto_selected = False
        direct_tail_true_coupled_coarse_selected = False
        direct_tail_true_coupled_coarse_metadata: dict[str, object] | None = None
        direct_tail_true_coupled_coarse_error: str | None = None
        direct_tail_true_coupled_coarse_residual_after: float | None = None
        direct_tail_true_window_selected = False
        direct_tail_true_window_metadata: dict[str, object] | None = None
        direct_tail_true_window_error: str | None = None
        direct_tail_true_window_residual_after: float | None = None
        direct_tail_residual_window_selected = False
        direct_tail_residual_window_metadata: dict[str, object] | None = None
        direct_tail_residual_window_error: str | None = None
        direct_tail_residual_window_residual_after: float | None = None
        direct_tail_residual_window_max_windows = int(
            direct_tail_residual_rescue_policy.residual_window_max_windows
        )
        direct_tail_residual_window_x_radius = int(direct_tail_residual_rescue_policy.residual_window_x_radius)
        direct_tail_residual_window_ell_radius = int(
            direct_tail_residual_rescue_policy.residual_window_ell_radius
        )
        direct_tail_residual_window_max_mb = float(direct_tail_residual_rescue_policy.residual_window_max_mb)
        direct_tail_residual_window_regularization = float(
            direct_tail_residual_rescue_policy.residual_window_regularization
        )
        direct_tail_residual_window_coefficient_mode = str(
            direct_tail_residual_rescue_policy.residual_window_coefficient_mode
        )
        direct_tail_residual_window_combine_mode = str(
            direct_tail_residual_rescue_policy.residual_window_combine_mode
        )
        direct_tail_residual_window_interface_depth = int(
            direct_tail_residual_rescue_policy.residual_window_interface_depth
        )
        direct_tail_residual_window_max_size = int(direct_tail_residual_rescue_policy.residual_window_max_size)
        direct_tail_true_window_max_windows = int(direct_tail_residual_rescue_policy.true_window_max_windows)
        direct_tail_true_window_x_radius = int(direct_tail_residual_rescue_policy.true_window_x_radius)
        direct_tail_true_window_ell_radius = int(direct_tail_residual_rescue_policy.true_window_ell_radius)
        direct_tail_true_window_max_mb = float(direct_tail_residual_rescue_policy.true_window_max_mb)
        direct_tail_true_window_regularization = float(
            direct_tail_residual_rescue_policy.true_window_regularization
        )
        direct_tail_true_window_max_size = int(direct_tail_residual_rescue_policy.true_window_max_size)
        direct_tail_true_window_column_batch = int(direct_tail_residual_rescue_policy.true_window_column_batch)
        direct_tail_true_window_drop_tol = float(direct_tail_residual_rescue_policy.true_window_drop_tol)
        direct_tail_true_window_include_tail = bool(direct_tail_residual_rescue_policy.true_window_include_tail)
        direct_tail_true_window_damping = bool(direct_tail_residual_rescue_policy.true_window_damping)
        direct_tail_true_window_beta_max = float(direct_tail_residual_rescue_policy.true_window_beta_max)
        direct_tail_true_window_specs_env = (
            os.environ.get("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_SPECS", "").strip()
            or os.environ.get("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_SPEC", "").strip()
        )
        direct_tail_true_window_specs = ()
        if direct_tail_true_window_specs_env:
            try:
                direct_tail_true_window_specs = _parse_true_operator_window_specs(
                    direct_tail_true_window_specs_env,
                    layout=RHS1BlockLayout.from_operator(op),
                )
            except (AttributeError, TypeError, ValueError) as exc:
                if emit is not None:
                    emit(
                        1,
                        "fortran_reduced_direct_tail_true_window: "
                        f"skipped explicit specs ({type(exc).__name__}: {exc})",
                    )
        direct_tail_true_active_rescue_policy = resolve_direct_tail_true_active_rescue_policy(os.environ)
        direct_tail_true_active_block_requested = bool(direct_tail_true_active_rescue_policy.active_block_requested)
        direct_tail_true_active_residual_block_requested = bool(
            direct_tail_true_active_rescue_policy.active_residual_block_requested
        )
        direct_tail_true_active_submatrix_requested = bool(
            direct_tail_true_active_rescue_policy.active_submatrix_requested
        )
        direct_tail_true_active_submatrix_selected = False
        direct_tail_true_active_submatrix_metadata: dict[str, object] | None = None
        direct_tail_true_active_submatrix_error: str | None = None
        direct_tail_true_active_submatrix_residual_after: float | None = None
        direct_tail_true_active_block_selected = False
        direct_tail_true_active_block_metadata: dict[str, object] | None = None
        direct_tail_true_active_block_error: str | None = None
        direct_tail_true_active_block_residual_after: float | None = None
        direct_tail_true_active_residual_block_selected = False
        direct_tail_true_active_residual_block_metadata: dict[str, object] | None = None
        direct_tail_true_active_residual_block_error: str | None = None
        direct_tail_true_active_residual_block_residual_after: float | None = None
        direct_tail_true_active_column_cache_requested = bool(
            direct_tail_true_active_rescue_policy.active_column_cache_requested
        )
        direct_tail_true_active_column_cache_max_mb = float(
            direct_tail_true_active_rescue_policy.active_column_cache_max_mb
        )
        direct_tail_true_active_column_cache_metadata: dict[str, object] | None = None
        direct_tail_true_active_block_x_count = int(direct_tail_true_active_rescue_policy.active_block_x_count)
        direct_tail_true_active_block_ell_count = int(direct_tail_true_active_rescue_policy.active_block_ell_count)
        direct_tail_true_active_block_species_count = (
            None
            if direct_tail_true_active_rescue_policy.active_block_species_count is None
            else int(direct_tail_true_active_rescue_policy.active_block_species_count)
        )
        direct_tail_true_active_block_theta_stride = int(
            direct_tail_true_active_rescue_policy.active_block_theta_stride
        )
        direct_tail_true_active_block_zeta_stride = int(
            direct_tail_true_active_rescue_policy.active_block_zeta_stride
        )
        direct_tail_true_active_block_max_mb = float(direct_tail_true_active_rescue_policy.active_block_max_mb)
        direct_tail_true_active_block_regularization = float(
            direct_tail_true_active_rescue_policy.active_block_regularization
        )
        direct_tail_true_active_block_max_size = int(direct_tail_true_active_rescue_policy.active_block_max_size)
        direct_tail_true_active_block_column_batch = int(
            direct_tail_true_active_rescue_policy.active_block_column_batch
        )
        direct_tail_true_active_block_drop_tol = float(direct_tail_true_active_rescue_policy.active_block_drop_tol)
        direct_tail_true_active_block_include_tail = bool(
            direct_tail_true_active_rescue_policy.active_block_include_tail
        )
        direct_tail_true_active_block_max_tail = int(direct_tail_true_active_rescue_policy.active_block_max_tail)
        direct_tail_true_active_block_damping = bool(direct_tail_true_active_rescue_policy.active_block_damping)
        direct_tail_true_active_block_beta_max = float(direct_tail_true_active_rescue_policy.active_block_beta_max)
        direct_tail_true_active_residual_block_max_mb = float(
            direct_tail_true_active_rescue_policy.active_residual_block_max_mb
        )
        direct_tail_true_active_residual_block_regularization = float(
            direct_tail_true_active_rescue_policy.active_residual_block_regularization
        )
        direct_tail_true_active_residual_block_max_size = int(
            direct_tail_true_active_rescue_policy.active_residual_block_max_size
        )
        direct_tail_true_active_residual_block_column_batch = int(
            direct_tail_true_active_rescue_policy.active_residual_block_column_batch
        )
        direct_tail_true_active_residual_block_drop_tol = float(
            direct_tail_true_active_rescue_policy.active_residual_block_drop_tol
        )
        direct_tail_true_active_residual_block_include_tail = bool(
            direct_tail_true_active_rescue_policy.active_residual_block_include_tail
        )
        direct_tail_true_active_residual_block_max_tail = int(
            direct_tail_true_active_rescue_policy.active_residual_block_max_tail
        )
        direct_tail_true_active_residual_block_kinetic_only = bool(
            direct_tail_true_active_rescue_policy.active_residual_block_kinetic_only
        )
        direct_tail_true_active_residual_block_damping = bool(
            direct_tail_true_active_rescue_policy.active_residual_block_damping
        )
        direct_tail_true_active_residual_block_beta_max = float(
            direct_tail_true_active_rescue_policy.active_residual_block_beta_max
        )
        direct_tail_true_active_residual_block_min_improvement = float(
            direct_tail_true_active_rescue_policy.active_residual_block_min_improvement
        )
        direct_tail_true_active_residual_block_accept_base_improvement = bool(
            direct_tail_true_active_rescue_policy.active_residual_block_accept_base_improvement
        )
        direct_tail_true_active_residual_block_base_improvement_override_used = False
        direct_tail_true_active_submatrix_damping = bool(
            direct_tail_true_active_rescue_policy.active_submatrix_damping
        )
        direct_tail_true_active_submatrix_alpha_clip = float(
            direct_tail_true_active_rescue_policy.active_submatrix_alpha_clip
        )
        direct_tail_true_active_submatrix_min_improvement = float(
            direct_tail_true_active_rescue_policy.active_submatrix_min_improvement
        )
        direct_tail_true_coupled_coarse_policy = resolve_direct_tail_coupled_coarse_rescue_policy(os.environ)
        direct_tail_true_coupled_coarse_max_windows = int(direct_tail_true_coupled_coarse_policy.max_windows)
        direct_tail_true_coupled_coarse_x_radius = int(direct_tail_true_coupled_coarse_policy.x_radius)
        direct_tail_true_coupled_coarse_ell_radius = int(direct_tail_true_coupled_coarse_policy.ell_radius)
        direct_tail_true_coupled_coarse_max_mb = float(direct_tail_true_coupled_coarse_policy.max_mb)
        direct_tail_true_coupled_coarse_regularization = float(
            direct_tail_true_coupled_coarse_policy.regularization
        )
        direct_tail_true_coupled_coarse_max_size = int(direct_tail_true_coupled_coarse_policy.max_size)
        direct_tail_true_coupled_coarse_column_batch = int(direct_tail_true_coupled_coarse_policy.column_batch)
        direct_tail_true_coupled_coarse_drop_tol = float(direct_tail_true_coupled_coarse_policy.drop_tol)
        direct_tail_true_coupled_coarse_low_lmax = int(direct_tail_true_coupled_coarse_policy.low_lmax)
        direct_tail_true_coupled_coarse_profile_moment_count = int(
            direct_tail_true_coupled_coarse_policy.profile_moment_count
        )
        direct_tail_true_coupled_coarse_angular_lmax = int(direct_tail_true_coupled_coarse_policy.angular_lmax)
        direct_tail_true_coupled_coarse_angular_mode_max = int(
            direct_tail_true_coupled_coarse_policy.angular_mode_max
        )
        direct_tail_true_coupled_coarse_max_tail_units = int(
            direct_tail_true_coupled_coarse_policy.max_tail_units
        )
        direct_tail_true_coupled_coarse_include_tail = bool(direct_tail_true_coupled_coarse_policy.include_tail)
        direct_tail_true_coupled_coarse_include_constraint_sources = bool(
            direct_tail_true_coupled_coarse_policy.include_constraint_sources
        )
        direct_tail_true_coupled_coarse_include_fsavg = bool(direct_tail_true_coupled_coarse_policy.include_fsavg)
        direct_tail_true_coupled_coarse_include_window_residual = bool(
            direct_tail_true_coupled_coarse_policy.include_window_residual
        )
        direct_tail_true_coupled_coarse_include_profile_moments = bool(
            direct_tail_true_coupled_coarse_policy.include_profile_moments
        )
        direct_tail_true_coupled_coarse_include_angular_residual = bool(
            direct_tail_true_coupled_coarse_policy.include_angular_residual
        )
        direct_tail_true_coupled_coarse_include_angular_basis = bool(
            direct_tail_true_coupled_coarse_policy.include_angular_basis
        )
        direct_tail_true_coupled_coarse_include_preconditioned_loads = bool(
            direct_tail_true_coupled_coarse_policy.include_preconditioned_loads
        )
        direct_tail_true_coupled_coarse_preconditioned_load_max_columns = int(
            direct_tail_true_coupled_coarse_policy.preconditioned_load_max_columns
        )
        direct_tail_true_coupled_coarse_preconditioned_load_max_nnz = int(
            direct_tail_true_coupled_coarse_policy.preconditioned_load_max_nnz
        )
        direct_tail_true_coupled_coarse_preconditioned_load_drop_tol = float(
            direct_tail_true_coupled_coarse_policy.preconditioned_load_drop_tol
        )
        direct_tail_true_coupled_coarse_damping = bool(direct_tail_true_coupled_coarse_policy.damping)
        direct_tail_true_coupled_coarse_beta_max = float(direct_tail_true_coupled_coarse_policy.beta_max)
        direct_tail_true_coupled_coarse_accept_base_improvement = bool(
            direct_tail_true_coupled_coarse_policy.accept_base_improvement
        )
        direct_tail_true_coupled_coarse_base_improvement_override_used = False

        if bool(factor_preflight_enabled) and x0_sparse is None:
            try:
                factor_preflight_evaluation = evaluate_sparse_pc_factor_preflight(
                    SparsePCFactorPreflightEvaluationContext(
                        rhs=sparse_pc_rhs,
                        rhs_norm=float(sparse_pc_rhs_norm),
                        target=float(target),
                        preconditioner=_precond_sparse,
                        matvec=_mv_true,
                        diagnostics=_rhs1_active_reduced_residual_diagnostics,
                        layout=RHS1BlockLayout.from_operator(op),
                        active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                        seed_enabled=bool(factor_preflight_seed_enabled),
                        max_target_ratio=float(factor_preflight_max_target_ratio),
                    )
                )
                factor_preflight_residual_before = float(factor_preflight_evaluation.residual_before)
                factor_preflight_residual_after = float(factor_preflight_evaluation.residual_after)
                factor_preflight_residual_diagnostics = factor_preflight_evaluation.diagnostics
                factor_preflight_improvement_ratio = factor_preflight_evaluation.improvement_ratio
                factor_preflight_target_ratio = factor_preflight_evaluation.target_ratio
                factor_preflight_passed = bool(factor_preflight_evaluation.passed)
                factor_preflight_seed_used = bool(factor_preflight_evaluation.seed_used)
                residual_vec_current = factor_preflight_evaluation.residual_vec
                if factor_preflight_evaluation.x0_seed is not None:
                    x0_sparse = factor_preflight_evaluation.x0_seed
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor preflight "
                        f"residual={float(factor_preflight_residual_before):.6e}"
                        f"->{float(factor_preflight_residual_after):.6e} "
                        f"improvement={float(factor_preflight_improvement_ratio or 0.0):.6e} "
                        f"target_ratio={float(factor_preflight_target_ratio or float('inf')):.6e} "
                        f"seed_used={bool(factor_preflight_seed_used)} "
                        f"passed={bool(factor_preflight_passed)}",
                    )
                    if isinstance(factor_preflight_residual_diagnostics, dict) and factor_preflight_residual_diagnostics.get(
                        "selected"
                    ):
                        component_norms = factor_preflight_residual_diagnostics.get("component_norms", {})
                        kinetic_fraction = (
                            component_norms.get("kinetic", {}).get("energy_fraction", 0.0)
                            if isinstance(component_norms, dict)
                            else 0.0
                        )
                        extra_fraction = (
                            component_norms.get("extra", {}).get("energy_fraction", 0.0)
                            if isinstance(component_norms, dict)
                            else 0.0
                        )
                        top_sx = factor_preflight_residual_diagnostics.get("top_species_x", [])
                        top_sx_label = top_sx[0].get("label") if isinstance(top_sx, list) and top_sx else "none"
                        top_sx_fraction = (
                            top_sx[0].get("energy_fraction", 0.0)
                            if isinstance(top_sx, list) and top_sx
                            else 0.0
                        )
                        top_x = factor_preflight_residual_diagnostics.get("top_x", [])
                        top_x_label = top_x[0].get("label") if isinstance(top_x, list) and top_x else "none"
                        top_x_fraction = (
                            top_x[0].get("energy_fraction", 0.0)
                            if isinstance(top_x, list) and top_x
                            else 0.0
                        )
                        top_ell = factor_preflight_residual_diagnostics.get("top_ell", [])
                        top_ell_label = top_ell[0].get("label") if isinstance(top_ell, list) and top_ell else "none"
                        top_ell_fraction = (
                            top_ell[0].get("energy_fraction", 0.0)
                            if isinstance(top_ell, list) and top_ell
                            else 0.0
                        )
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: sparse_pc_gmres preflight residual diagnostics "
                            f"kinetic_energy_fraction={float(kinetic_fraction):.6e} "
                            f"extra_energy_fraction={float(extra_fraction):.6e} "
                            f"top_species_x={top_sx_label} "
                            f"top_species_x_fraction={float(top_sx_fraction):.6e} "
                            f"top_x={top_x_label} top_x_fraction={float(top_x_fraction):.6e} "
                            f"top_ell={top_ell_label} top_ell_fraction={float(top_ell_fraction):.6e}",
                        )
                true_coupled_factor_kind = str(getattr(factor_bundle_pc, "kind", "")).strip().lower().replace("-", "_")
                true_coupled_auto_reference_kind = true_coupled_factor_kind in {
                    "active_fortran_v3_reduced_lu",
                }
                true_coupled_auto_native_kind = true_coupled_factor_kind in {
                    "active_fortran_v3_reduced_native_stack",
                    "active_v3_reduced_native_stack",
                    "fortran_v3_reduced_native_stack",
                    "active_bounded_native_stack",
                    "active_native_stack",
                }
                direct_tail_true_coupled_coarse_auto_selected = bool(
                    direct_tail_true_coupled_coarse_auto_enabled
                    and (
                        bool(true_coupled_auto_reference_kind)
                        or (
                            bool(direct_tail_true_coupled_coarse_auto_native_enabled)
                            and bool(true_coupled_auto_native_kind)
                        )
                    )
                    and int(sparse_pc_linear_size) >= int(direct_tail_true_coupled_coarse_auto_min_size)
                    and factor_preflight_target_ratio is not None
                    and np.isfinite(float(factor_preflight_target_ratio))
                    and float(factor_preflight_target_ratio)
                    > float(direct_tail_true_coupled_coarse_auto_target_ratio)
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                )
                direct_tail_true_coupled_coarse_requested = bool(
                    direct_tail_true_coupled_coarse_explicit_requested
                    or direct_tail_true_coupled_coarse_auto_selected
                )
                if (
                    bool(direct_tail_true_coupled_coarse_requested)
                    and (
                        factor_preflight_passed is False
                        or bool(direct_tail_true_coupled_coarse_auto_selected)
                    )
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        true_coupled_bundle = _try_build_true_operator_coupled_coarse_lsq_preconditioner(
                            true_matvec=lambda vec: np.asarray(
                                jax.device_get(_mv_true_no_count(jnp.asarray(vec, dtype=jnp.float64))),
                                dtype=np.float64,
                            ),
                            true_matmat=lambda mat: np.asarray(_mv_true_matmat(np.asarray(mat, dtype=np.float64))),
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            op=op,
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            max_windows=int(direct_tail_true_coupled_coarse_max_windows),
                            x_radius=int(direct_tail_true_coupled_coarse_x_radius),
                            ell_radius=int(direct_tail_true_coupled_coarse_ell_radius),
                            max_nbytes=_rhs1_additive_rescue_nbytes(
                                factor_bundle_pc,
                                direct_tail_true_coupled_coarse_max_mb
                            ),
                            regularization=float(direct_tail_true_coupled_coarse_regularization),
                            max_coarse_size=int(direct_tail_true_coupled_coarse_max_size),
                            column_batch=int(direct_tail_true_coupled_coarse_column_batch),
                            drop_tol=float(direct_tail_true_coupled_coarse_drop_tol),
                            low_lmax=int(direct_tail_true_coupled_coarse_low_lmax),
                            profile_moment_count=int(direct_tail_true_coupled_coarse_profile_moment_count),
                            angular_lmax=int(direct_tail_true_coupled_coarse_angular_lmax),
                            angular_mode_max=int(direct_tail_true_coupled_coarse_angular_mode_max),
                            max_tail_units=int(direct_tail_true_coupled_coarse_max_tail_units),
                            include_tail=bool(direct_tail_true_coupled_coarse_include_tail),
                            include_constraint_sources=bool(
                                direct_tail_true_coupled_coarse_include_constraint_sources
                            ),
                            include_fsavg=bool(direct_tail_true_coupled_coarse_include_fsavg),
                            include_window_residual=bool(direct_tail_true_coupled_coarse_include_window_residual),
                            include_profile_moments=bool(
                                direct_tail_true_coupled_coarse_include_profile_moments
                            ),
                            include_angular_residual=bool(
                                direct_tail_true_coupled_coarse_include_angular_residual
                            ),
                            include_angular_basis=bool(direct_tail_true_coupled_coarse_include_angular_basis),
                            include_preconditioned_loads=bool(
                                direct_tail_true_coupled_coarse_include_preconditioned_loads
                            ),
                            preconditioned_load_max_columns=int(
                                direct_tail_true_coupled_coarse_preconditioned_load_max_columns
                            ),
                            preconditioned_load_max_nnz=int(
                                direct_tail_true_coupled_coarse_preconditioned_load_max_nnz
                            ),
                            preconditioned_load_drop_tol=float(
                                direct_tail_true_coupled_coarse_preconditioned_load_drop_tol
                            ),
                            damping=bool(direct_tail_true_coupled_coarse_damping),
                            beta_max=float(direct_tail_true_coupled_coarse_beta_max),
                            emit=emit,
                        )
                        if true_coupled_bundle is None:
                            direct_tail_true_coupled_coarse_error = "builder_returned_none"
                        else:
                            x_true_coupled_sparse = jnp.asarray(
                                true_coupled_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_true_coupled = sparse_pc_rhs - jnp.asarray(
                                _mv_true_no_count(x_true_coupled_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_true_coupled_coarse_residual_after = float(
                                jnp.linalg.norm(residual_vec_true_coupled)
                            )
                            true_coupled_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_true_coupled,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_true_coupled_coarse_metadata = dict(true_coupled_bundle.metadata or {})
                            direct_tail_true_coupled_coarse_metadata["residual_after"] = float(
                                direct_tail_true_coupled_coarse_residual_after
                            )
                            direct_tail_true_coupled_coarse_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            true_coupled_acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                                SparsePCResidualCandidateAcceptanceContext(
                                    candidate_residual_after=float(
                                        direct_tail_true_coupled_coarse_residual_after
                                    ),
                                    current_residual_after=float(factor_preflight_residual_after),
                                    original_residual_before=factor_preflight_residual_before,
                                    target=float(target),
                                    max_target_ratio=float(factor_preflight_max_target_ratio),
                                    seed_enabled=bool(factor_preflight_seed_enabled),
                                    accept_base_improvement=bool(
                                        direct_tail_true_coupled_coarse_accept_base_improvement
                                    ),
                                    base_improvement_requires_original_miss=False,
                                    base_improvement_sets_passed=True,
                                )
                            )
                            if bool(true_coupled_acceptance.accepted):
                                direct_tail_true_coupled_coarse_selected = True
                                direct_tail_true_coupled_coarse_base_improvement_override_used = bool(
                                    true_coupled_acceptance.base_improvement_override_used
                                )
                                factor_bundle_pc = true_coupled_bundle
                                pc_factor_s += float(true_coupled_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(true_coupled_acceptance.residual_after)
                                residual_vec_current = residual_vec_true_coupled
                                factor_preflight_residual_diagnostics = true_coupled_diagnostics
                                factor_preflight_improvement_ratio = true_coupled_acceptance.improvement_ratio
                                factor_preflight_target_ratio = true_coupled_acceptance.target_ratio
                                factor_preflight_passed = bool(true_coupled_acceptance.passed)
                                if bool(true_coupled_acceptance.seed_used):
                                    x0_sparse = x_true_coupled_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: true coupled coarse accepted "
                                        f"coarse_size={direct_tail_true_coupled_coarse_metadata.get('coarse_size')} "
                                        f"residual={direct_tail_true_coupled_coarse_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)} "
                                        f"base_improvement_override={bool(direct_tail_true_coupled_coarse_base_improvement_override_used)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: true coupled coarse rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_true_coupled_coarse_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_true_coupled_coarse_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: true coupled coarse failed "
                                f"({direct_tail_true_coupled_coarse_error})",
                            )

                def _direct_tail_rescue_needed_after_preflight() -> bool:
                    if factor_preflight_passed is False:
                        return True
                    if not bool(direct_tail_true_coupled_coarse_base_improvement_override_used):
                        return False
                    continue_after_override = _rhs1_bool_env(
                        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESCUE_AFTER_BASE_IMPROVEMENT",
                        default=True,
                    )
                    if not bool(continue_after_override):
                        return False
                    if factor_preflight_target_ratio is None:
                        return True
                    try:
                        return float(factor_preflight_target_ratio) > float(factor_preflight_max_target_ratio)
                    except (TypeError, ValueError):
                        return True

                direct_tail_true_active_column_cache: _ReusableTrueActionColumnCache | None = None
                if (
                    bool(direct_tail_true_active_submatrix_requested)
                    or bool(direct_tail_true_active_block_requested)
                    or bool(direct_tail_true_active_residual_block_requested)
                    or bool(direct_tail_true_window_requested)
                ):
                    direct_tail_true_active_column_cache = _ReusableTrueActionColumnCache(
                        true_matvec=lambda vec: np.asarray(
                            jax.device_get(_mv_true_no_count(jnp.asarray(vec, dtype=jnp.float64))),
                            dtype=np.float64,
                        ).reshape((-1,)),
                        true_matmat=lambda mat: np.asarray(_mv_true_matmat(np.asarray(mat, dtype=np.float64))),
                        n=int(sparse_pc_linear_size),
                        max_nbytes=int(
                            max(0.0, float(direct_tail_true_active_column_cache_max_mb)) * 1024.0 * 1024.0
                        ),
                        enabled=bool(direct_tail_true_active_column_cache_requested),
                    )

                def _true_active_cached_matvec(vec: np.ndarray) -> np.ndarray:
                    if direct_tail_true_active_column_cache is not None:
                        return direct_tail_true_active_column_cache.matvec(vec)
                    return np.asarray(
                        jax.device_get(_mv_true_no_count(jnp.asarray(vec, dtype=jnp.float64))),
                        dtype=np.float64,
                    ).reshape((-1,))

                def _true_active_cached_matmat(mat: np.ndarray) -> np.ndarray:
                    if direct_tail_true_active_column_cache is not None:
                        return direct_tail_true_active_column_cache.matmat(mat)
                    return np.asarray(_mv_true_matmat(np.asarray(mat, dtype=np.float64)))

                if (
                    bool(direct_tail_true_active_submatrix_requested)
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        true_active_submatrix_bundle = _try_build_true_operator_active_submatrix_preconditioner(
                            true_matvec=_true_active_cached_matvec,
                            true_matmat=_true_active_cached_matmat,
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            x_count=int(direct_tail_true_active_block_x_count),
                            ell_count=int(direct_tail_true_active_block_ell_count),
                            species_count=direct_tail_true_active_block_species_count,
                            theta_stride=int(direct_tail_true_active_block_theta_stride),
                            zeta_stride=int(direct_tail_true_active_block_zeta_stride),
                            max_nbytes=_rhs1_additive_rescue_nbytes(
                                factor_bundle_pc,
                                direct_tail_true_active_block_max_mb
                            ),
                            regularization=float(direct_tail_true_active_block_regularization),
                            max_block_size=int(direct_tail_true_active_block_max_size),
                            column_batch=int(direct_tail_true_active_block_column_batch),
                            drop_tol=float(direct_tail_true_active_block_drop_tol),
                            include_tail=bool(direct_tail_true_active_block_include_tail),
                            max_tail=int(direct_tail_true_active_block_max_tail),
                            damping=bool(direct_tail_true_active_submatrix_damping),
                            alpha_clip=float(direct_tail_true_active_submatrix_alpha_clip),
                            emit=emit,
                        )
                        if true_active_submatrix_bundle is None:
                            direct_tail_true_active_submatrix_error = "builder_returned_none"
                        else:
                            x_true_active_submatrix_sparse = jnp.asarray(
                                true_active_submatrix_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_true_active_submatrix = sparse_pc_rhs - jnp.asarray(
                                _mv_true_no_count(x_true_active_submatrix_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_true_active_submatrix_residual_after = float(
                                jnp.linalg.norm(residual_vec_true_active_submatrix)
                            )
                            true_active_submatrix_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_true_active_submatrix,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_true_active_submatrix_metadata = dict(
                                true_active_submatrix_bundle.metadata or {}
                            )
                            direct_tail_true_active_submatrix_metadata["residual_after"] = float(
                                direct_tail_true_active_submatrix_residual_after
                            )
                            direct_tail_true_active_submatrix_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            true_active_submatrix_acceptance = (
                                evaluate_sparse_pc_residual_candidate_acceptance(
                                    SparsePCResidualCandidateAcceptanceContext(
                                        candidate_residual_after=float(
                                            direct_tail_true_active_submatrix_residual_after
                                        ),
                                        current_residual_after=float(factor_preflight_residual_after),
                                        original_residual_before=factor_preflight_residual_before,
                                        target=float(target),
                                        max_target_ratio=float(factor_preflight_max_target_ratio),
                                        seed_enabled=bool(factor_preflight_seed_enabled),
                                        require_original_improvement=False,
                                        current_min_improvement=float(
                                            direct_tail_true_active_submatrix_min_improvement
                                        ),
                                    )
                                )
                            )
                            if bool(true_active_submatrix_acceptance.accepted):
                                direct_tail_true_active_submatrix_selected = True
                                factor_bundle_pc = true_active_submatrix_bundle
                                pc_factor_s += float(true_active_submatrix_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(
                                    true_active_submatrix_acceptance.residual_after
                                )
                                residual_vec_current = residual_vec_true_active_submatrix
                                factor_preflight_residual_diagnostics = true_active_submatrix_diagnostics
                                factor_preflight_improvement_ratio = (
                                    true_active_submatrix_acceptance.improvement_ratio
                                )
                                factor_preflight_target_ratio = true_active_submatrix_acceptance.target_ratio
                                factor_preflight_passed = bool(true_active_submatrix_acceptance.passed)
                                if bool(true_active_submatrix_acceptance.seed_used):
                                    x0_sparse = x_true_active_submatrix_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: true active submatrix accepted "
                                        f"block_size={direct_tail_true_active_submatrix_metadata.get('block_size')} "
                                        f"residual={direct_tail_true_active_submatrix_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: true active submatrix rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_true_active_submatrix_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_true_active_submatrix_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: true active submatrix failed "
                                f"({direct_tail_true_active_submatrix_error})",
                            )

                if (
                    bool(direct_tail_true_active_block_requested)
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        true_active_block_bundle = _try_build_true_operator_active_block_lsq_preconditioner(
                            true_matvec=_true_active_cached_matvec,
                            true_matmat=_true_active_cached_matmat,
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            x_count=int(direct_tail_true_active_block_x_count),
                            ell_count=int(direct_tail_true_active_block_ell_count),
                            species_count=direct_tail_true_active_block_species_count,
                            theta_stride=int(direct_tail_true_active_block_theta_stride),
                            zeta_stride=int(direct_tail_true_active_block_zeta_stride),
                            max_nbytes=_rhs1_additive_rescue_nbytes(
                                factor_bundle_pc,
                                direct_tail_true_active_block_max_mb
                            ),
                            regularization=float(direct_tail_true_active_block_regularization),
                            max_block_size=int(direct_tail_true_active_block_max_size),
                            column_batch=int(direct_tail_true_active_block_column_batch),
                            drop_tol=float(direct_tail_true_active_block_drop_tol),
                            include_tail=bool(direct_tail_true_active_block_include_tail),
                            max_tail=int(direct_tail_true_active_block_max_tail),
                            damping=bool(direct_tail_true_active_block_damping),
                            beta_max=float(direct_tail_true_active_block_beta_max),
                            emit=emit,
                        )
                        if true_active_block_bundle is None:
                            direct_tail_true_active_block_error = "builder_returned_none"
                        else:
                            x_true_active_block_sparse = jnp.asarray(
                                true_active_block_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_true_active_block = sparse_pc_rhs - jnp.asarray(
                                _mv_true_no_count(x_true_active_block_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_true_active_block_residual_after = float(
                                jnp.linalg.norm(residual_vec_true_active_block)
                            )
                            true_active_block_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_true_active_block,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_true_active_block_metadata = dict(true_active_block_bundle.metadata or {})
                            direct_tail_true_active_block_metadata["residual_after"] = float(
                                direct_tail_true_active_block_residual_after
                            )
                            direct_tail_true_active_block_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            true_active_block_acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                                SparsePCResidualCandidateAcceptanceContext(
                                    candidate_residual_after=float(
                                        direct_tail_true_active_block_residual_after
                                    ),
                                    current_residual_after=float(factor_preflight_residual_after),
                                    original_residual_before=factor_preflight_residual_before,
                                    target=float(target),
                                    max_target_ratio=float(factor_preflight_max_target_ratio),
                                    seed_enabled=bool(factor_preflight_seed_enabled),
                                    require_original_improvement=False,
                                )
                            )
                            if bool(true_active_block_acceptance.accepted):
                                direct_tail_true_active_block_selected = True
                                factor_bundle_pc = true_active_block_bundle
                                pc_factor_s += float(true_active_block_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(true_active_block_acceptance.residual_after)
                                residual_vec_current = residual_vec_true_active_block
                                factor_preflight_residual_diagnostics = true_active_block_diagnostics
                                factor_preflight_improvement_ratio = true_active_block_acceptance.improvement_ratio
                                factor_preflight_target_ratio = true_active_block_acceptance.target_ratio
                                factor_preflight_passed = bool(true_active_block_acceptance.passed)
                                if bool(true_active_block_acceptance.seed_used):
                                    x0_sparse = x_true_active_block_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: true active block accepted "
                                        f"block_size={direct_tail_true_active_block_metadata.get('block_size')} "
                                        f"residual={direct_tail_true_active_block_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: true active block rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_true_active_block_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_true_active_block_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: true active block failed "
                                f"({direct_tail_true_active_block_error})",
                            )

                if (
                    bool(direct_tail_true_active_residual_block_requested)
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        true_active_residual_block_bundle = (
                            _try_build_true_operator_active_residual_block_lsq_preconditioner(
                                true_matvec=_true_active_cached_matvec,
                                true_matmat=_true_active_cached_matmat,
                                factor_bundle=factor_bundle_pc,
                                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                                max_nbytes=_rhs1_additive_rescue_nbytes(
                                    factor_bundle_pc,
                                    direct_tail_true_active_residual_block_max_mb
                                ),
                                regularization=float(direct_tail_true_active_residual_block_regularization),
                                max_block_size=int(direct_tail_true_active_residual_block_max_size),
                                column_batch=int(direct_tail_true_active_residual_block_column_batch),
                                drop_tol=float(direct_tail_true_active_residual_block_drop_tol),
                                include_tail=bool(direct_tail_true_active_residual_block_include_tail),
                                max_tail=int(direct_tail_true_active_residual_block_max_tail),
                                kinetic_only=bool(direct_tail_true_active_residual_block_kinetic_only),
                                damping=bool(direct_tail_true_active_residual_block_damping),
                                beta_max=float(direct_tail_true_active_residual_block_beta_max),
                                emit=emit,
                            )
                        )
                        if true_active_residual_block_bundle is None:
                            direct_tail_true_active_residual_block_error = "builder_returned_none"
                        else:
                            x_true_active_residual_block_sparse = jnp.asarray(
                                true_active_residual_block_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_true_active_residual_block = sparse_pc_rhs - jnp.asarray(
                                _mv_true_no_count(x_true_active_residual_block_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_true_active_residual_block_residual_after = float(
                                jnp.linalg.norm(residual_vec_true_active_residual_block)
                            )
                            true_active_residual_block_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_true_active_residual_block,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_true_active_residual_block_metadata = dict(
                                true_active_residual_block_bundle.metadata or {}
                            )
                            direct_tail_true_active_residual_block_metadata["residual_after"] = float(
                                direct_tail_true_active_residual_block_residual_after
                            )
                            direct_tail_true_active_residual_block_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            true_active_residual_block_acceptance = (
                                evaluate_sparse_pc_residual_candidate_acceptance(
                                    SparsePCResidualCandidateAcceptanceContext(
                                        candidate_residual_after=float(
                                            direct_tail_true_active_residual_block_residual_after
                                        ),
                                        current_residual_after=float(factor_preflight_residual_after),
                                        original_residual_before=factor_preflight_residual_before,
                                        target=float(target),
                                        max_target_ratio=float(factor_preflight_max_target_ratio),
                                        seed_enabled=bool(factor_preflight_seed_enabled),
                                        current_min_improvement=float(
                                            direct_tail_true_active_residual_block_min_improvement
                                        ),
                                        accept_base_improvement=bool(
                                            direct_tail_true_active_residual_block_accept_base_improvement
                                        ),
                                        missing_original_improves=True,
                                    )
                                )
                            )
                            direct_tail_true_active_residual_block_base_improvement_override_used = bool(
                                true_active_residual_block_acceptance.base_improvement_override_used
                            )
                            direct_tail_true_active_residual_block_metadata[
                                "accept_base_improvement"
                            ] = bool(direct_tail_true_active_residual_block_accept_base_improvement)
                            direct_tail_true_active_residual_block_metadata[
                                "base_improvement_override_used"
                            ] = bool(direct_tail_true_active_residual_block_base_improvement_override_used)
                            direct_tail_true_active_residual_block_metadata[
                                "improves_current_residual"
                            ] = bool(true_active_residual_block_acceptance.improves_current_residual)
                            direct_tail_true_active_residual_block_metadata[
                                "improves_original_residual"
                            ] = bool(true_active_residual_block_acceptance.improves_original_residual)
                            if bool(true_active_residual_block_acceptance.accepted):
                                direct_tail_true_active_residual_block_selected = True
                                factor_bundle_pc = true_active_residual_block_bundle
                                pc_factor_s += float(true_active_residual_block_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(
                                    true_active_residual_block_acceptance.residual_after
                                )
                                residual_vec_current = residual_vec_true_active_residual_block
                                factor_preflight_residual_diagnostics = true_active_residual_block_diagnostics
                                factor_preflight_improvement_ratio = (
                                    true_active_residual_block_acceptance.improvement_ratio
                                )
                                factor_preflight_target_ratio = (
                                    true_active_residual_block_acceptance.target_ratio
                                )
                                factor_preflight_passed = bool(true_active_residual_block_acceptance.passed)
                                if bool(true_active_residual_block_acceptance.seed_used):
                                    x0_sparse = x_true_active_residual_block_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: true active residual block accepted "
                                        f"block_size={direct_tail_true_active_residual_block_metadata.get('block_size')} "
                                        f"residual={direct_tail_true_active_residual_block_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)} "
                                        "base_improvement_override="
                                        f"{bool(direct_tail_true_active_residual_block_base_improvement_override_used)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: true active residual block rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_true_active_residual_block_residual_after):.6e} "
                                    "improves_original="
                                    f"{bool(true_active_residual_block_acceptance.improves_original_residual)}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_true_active_residual_block_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: true active residual block failed "
                                f"({direct_tail_true_active_residual_block_error})",
                            )

                if (
                    bool(direct_tail_true_window_requested)
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        true_window_bundle = _try_build_true_operator_residual_window_lsq_preconditioner(
                            true_matvec=_true_active_cached_matvec,
                            true_matmat=_true_active_cached_matmat,
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            max_windows=int(direct_tail_true_window_max_windows),
                            x_radius=int(direct_tail_true_window_x_radius),
                            ell_radius=int(direct_tail_true_window_ell_radius),
                            max_nbytes=_rhs1_additive_rescue_nbytes(factor_bundle_pc, direct_tail_true_window_max_mb),
                            regularization=float(direct_tail_true_window_regularization),
                            max_window_size=int(direct_tail_true_window_max_size),
                            column_batch=int(direct_tail_true_window_column_batch),
                            drop_tol=float(direct_tail_true_window_drop_tol),
                            include_tail=bool(direct_tail_true_window_include_tail),
                            explicit_specs=tuple(direct_tail_true_window_specs),
                            damping=bool(direct_tail_true_window_damping),
                            beta_max=float(direct_tail_true_window_beta_max),
                            emit=emit,
                        )
                        if true_window_bundle is None:
                            direct_tail_true_window_error = "builder_returned_none"
                        else:
                            x_true_window_sparse = jnp.asarray(
                                true_window_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_true_window = sparse_pc_rhs - jnp.asarray(
                                _mv_true_no_count(x_true_window_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_true_window_residual_after = float(jnp.linalg.norm(residual_vec_true_window))
                            true_window_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_true_window,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_true_window_metadata = dict(true_window_bundle.metadata or {})
                            direct_tail_true_window_metadata["residual_after"] = float(
                                direct_tail_true_window_residual_after
                            )
                            direct_tail_true_window_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            true_window_acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                                SparsePCResidualCandidateAcceptanceContext(
                                    candidate_residual_after=float(direct_tail_true_window_residual_after),
                                    current_residual_after=float(factor_preflight_residual_after),
                                    original_residual_before=factor_preflight_residual_before,
                                    target=float(target),
                                    max_target_ratio=float(factor_preflight_max_target_ratio),
                                    seed_enabled=bool(factor_preflight_seed_enabled),
                                )
                            )
                            if bool(true_window_acceptance.accepted):
                                direct_tail_true_window_selected = True
                                factor_bundle_pc = true_window_bundle
                                pc_factor_s += float(true_window_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(true_window_acceptance.residual_after)
                                residual_vec_current = residual_vec_true_window
                                factor_preflight_residual_diagnostics = true_window_diagnostics
                                factor_preflight_improvement_ratio = true_window_acceptance.improvement_ratio
                                factor_preflight_target_ratio = true_window_acceptance.target_ratio
                                factor_preflight_passed = bool(true_window_acceptance.passed)
                                if bool(true_window_acceptance.seed_used):
                                    x0_sparse = x_true_window_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: true residual window accepted "
                                        f"window_size={direct_tail_true_window_metadata.get('window_size')} "
                                        f"residual={direct_tail_true_window_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: true residual window rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_true_window_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_true_window_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: true residual window failed "
                                f"({direct_tail_true_window_error})",
                            )
                if (
                    bool(direct_tail_residual_coarse_requested)
                    and bool(structured_pc_ready)
                    and _operator_bundle_pc is not None
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        residual_coarse_bundle = _try_build_residual_coarse_host_sparse_preconditioner(
                            operator_bundle=_operator_bundle_pc,
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            max_rank=int(direct_tail_residual_coarse_rank),
                            max_nbytes=_rhs1_additive_rescue_nbytes(
                                factor_bundle_pc, direct_tail_residual_coarse_max_mb
                            ),
                            regularization=float(direct_tail_residual_coarse_regularization),
                            emit=emit,
                        )
                        if residual_coarse_bundle is None:
                            direct_tail_residual_coarse_error = "builder_returned_none"
                        else:
                            x_rescue_sparse = jnp.asarray(
                                residual_coarse_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_rescue = sparse_pc_rhs - jnp.asarray(
                                _mv_true(x_rescue_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_residual_coarse_residual_after = float(jnp.linalg.norm(residual_vec_rescue))
                            rescue_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_rescue,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_residual_coarse_metadata = dict(residual_coarse_bundle.metadata or {})
                            direct_tail_residual_coarse_metadata["residual_after"] = float(
                                direct_tail_residual_coarse_residual_after
                            )
                            direct_tail_residual_coarse_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            residual_coarse_acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                                SparsePCResidualCandidateAcceptanceContext(
                                    candidate_residual_after=float(direct_tail_residual_coarse_residual_after),
                                    current_residual_after=float(factor_preflight_residual_after),
                                    original_residual_before=factor_preflight_residual_before,
                                    target=float(target),
                                    max_target_ratio=float(factor_preflight_max_target_ratio),
                                    seed_enabled=bool(factor_preflight_seed_enabled),
                                )
                            )
                            if bool(residual_coarse_acceptance.accepted):
                                direct_tail_residual_coarse_selected = True
                                factor_bundle_pc = residual_coarse_bundle
                                pc_factor_s += float(residual_coarse_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(residual_coarse_acceptance.residual_after)
                                residual_vec_current = residual_vec_rescue
                                factor_preflight_residual_diagnostics = rescue_diagnostics
                                factor_preflight_improvement_ratio = residual_coarse_acceptance.improvement_ratio
                                factor_preflight_target_ratio = residual_coarse_acceptance.target_ratio
                                factor_preflight_passed = bool(residual_coarse_acceptance.passed)
                                if bool(residual_coarse_acceptance.seed_used):
                                    x0_sparse = x_rescue_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: residual coarse accepted "
                                        f"rank={direct_tail_residual_coarse_metadata.get('rank')} "
                                        f"residual={direct_tail_residual_coarse_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: residual coarse rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_residual_coarse_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_residual_coarse_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: residual coarse failed "
                                f"({direct_tail_residual_coarse_error})",
                            )
                if (
                    bool(direct_tail_residual_window_requested)
                    and bool(structured_pc_ready)
                    and _operator_bundle_pc is not None
                    and _direct_tail_rescue_needed_after_preflight()
                    and factor_preflight_residual_after is not None
                    and np.isfinite(float(factor_preflight_residual_after))
                ):
                    try:
                        residual_window_bundle = _try_build_residual_window_host_sparse_preconditioner(
                            operator_bundle=_operator_bundle_pc,
                            factor_bundle=factor_bundle_pc,
                            residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            max_windows=int(direct_tail_residual_window_max_windows),
                            x_radius=int(direct_tail_residual_window_x_radius),
                            ell_radius=int(direct_tail_residual_window_ell_radius),
                            max_nbytes=_rhs1_additive_rescue_nbytes(
                                factor_bundle_pc, direct_tail_residual_window_max_mb
                            ),
                            regularization=float(direct_tail_residual_window_regularization),
                            coefficient_mode=str(direct_tail_residual_window_coefficient_mode),
                            combine_mode=str(direct_tail_residual_window_combine_mode),
                            interface_depth=int(direct_tail_residual_window_interface_depth),
                            max_window_size=int(direct_tail_residual_window_max_size),
                            emit=emit,
                        )
                        if residual_window_bundle is None:
                            direct_tail_residual_window_error = "builder_returned_none"
                        else:
                            x_window_sparse = jnp.asarray(
                                residual_window_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                                dtype=jnp.float64,
                            )
                            residual_vec_window = sparse_pc_rhs - jnp.asarray(
                                _mv_true(x_window_sparse),
                                dtype=jnp.float64,
                            )
                            direct_tail_residual_window_residual_after = float(jnp.linalg.norm(residual_vec_window))
                            window_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                                residual=residual_vec_window,
                                layout=RHS1BlockLayout.from_operator(op),
                                active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                            )
                            direct_tail_residual_window_metadata = dict(residual_window_bundle.metadata or {})
                            direct_tail_residual_window_metadata["residual_after"] = float(
                                direct_tail_residual_window_residual_after
                            )
                            direct_tail_residual_window_metadata["base_residual_after"] = float(
                                factor_preflight_residual_after
                            )
                            residual_window_acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                                SparsePCResidualCandidateAcceptanceContext(
                                    candidate_residual_after=float(direct_tail_residual_window_residual_after),
                                    current_residual_after=float(factor_preflight_residual_after),
                                    original_residual_before=factor_preflight_residual_before,
                                    target=float(target),
                                    max_target_ratio=float(factor_preflight_max_target_ratio),
                                    seed_enabled=bool(factor_preflight_seed_enabled),
                                )
                            )
                            if bool(residual_window_acceptance.accepted):
                                direct_tail_residual_window_selected = True
                                factor_bundle_pc = residual_window_bundle
                                pc_factor_s += float(residual_window_bundle.factor_s or 0.0)
                                setup_s = sparse_timer.elapsed_s()
                                factor_preflight_residual_after = float(residual_window_acceptance.residual_after)
                                residual_vec_current = residual_vec_window
                                factor_preflight_residual_diagnostics = window_diagnostics
                                factor_preflight_improvement_ratio = residual_window_acceptance.improvement_ratio
                                factor_preflight_target_ratio = residual_window_acceptance.target_ratio
                                factor_preflight_passed = bool(residual_window_acceptance.passed)
                                if bool(residual_window_acceptance.seed_used):
                                    x0_sparse = x_window_sparse
                                    factor_preflight_seed_used = True
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: residual window accepted "
                                        f"windows={direct_tail_residual_window_metadata.get('window_count')} "
                                        f"residual={direct_tail_residual_window_metadata['base_residual_after']:.6e}"
                                        f"->{float(factor_preflight_residual_after):.6e} "
                                        f"passed={bool(factor_preflight_passed)}",
                                    )
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: residual window rejected "
                                    f"residual={float(factor_preflight_residual_after):.6e}"
                                    f"->{float(direct_tail_residual_window_residual_after):.6e}",
                                )
                    except Exception as exc:  # noqa: BLE001
                        direct_tail_residual_window_error = f"{type(exc).__name__}: {exc}"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: residual window failed "
                                f"({direct_tail_residual_window_error})",
                            )
                if direct_tail_true_active_column_cache is not None:
                    direct_tail_true_active_column_cache_metadata = (
                        direct_tail_true_active_column_cache.metadata()
                    )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: true active column cache "
                            f"hits={direct_tail_true_active_column_cache_metadata['hits']} "
                            f"misses={direct_tail_true_active_column_cache_metadata['misses']} "
                            f"stored_columns={direct_tail_true_active_column_cache_metadata['stored_columns']} "
                            f"stored_mb={float(direct_tail_true_active_column_cache_metadata['stored_nbytes']) / 1.0e6:.3f}",
                        )
            except Exception as exc:  # noqa: BLE001
                factor_preflight_passed = False
                factor_preflight_error = f"{type(exc).__name__}: {exc}"
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor preflight failed "
                        f"({factor_preflight_error})",
                    )
            if bool(factor_preflight_required) and not bool(factor_preflight_passed):
                raise RuntimeError(
                    "sparse_pc_gmres factor preflight failed: "
                    f"residual_after={factor_preflight_residual_after} "
                    f"target={float(target):.6e} "
                    f"target_ratio={factor_preflight_target_ratio} "
                    f"max_target_ratio={float(factor_preflight_max_target_ratio):.6e} "
                    f"error={factor_preflight_error}"
                )
            auto_preflight_retry_selected = False
            auto_preflight_retry_attempts: list[dict[str, object]] = []
            auto_preflight_retry_enabled = _rhs1_bool_env(
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_RETRY",
                default=True,
            )
            if (
                bool(auto_preflight_retry_enabled)
                and bool(structured_pc_ready)
                and bool(structured_pc_preflight_required)
                and factor_preflight_passed is False
                and str(direct_tail_structured_pc_requested or "").strip().lower().replace("-", "_")
                in {"auto", "active_auto", "structured", "structured_auto"}
                and direct_tail_operator_bundle is not None
                and direct_tail_structured_layout is not None
                and direct_tail_structured_max_nbytes is not None
                and isinstance(direct_tail_structured_pc_metadata, dict)
            ):
                metadata_inner = direct_tail_structured_pc_metadata.get("metadata")
                max_retry_candidates = _rhs1_int_env(
                    "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_MAX_CANDIDATES",
                    default=2,
                    minimum=1,
                )
                retry_selection = select_sparse_pc_auto_preflight_retry_candidates(
                    SparsePCAutoPreflightRetrySelectionContext(
                        metadata=metadata_inner if isinstance(metadata_inner, dict) else None,
                        current_kind=str(getattr(factor_bundle_pc, "kind", "")),
                        sparse_pc_linear_size=int(sparse_pc_linear_size),
                        preflight_required_min_size=int(structured_pc_preflight_required_min_size),
                        skip_large_kinds_raw=os.environ.get(
                            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_SKIP_LARGE",
                            "active_spilu,active_ilu,active_global_sparse_ilu,jacobi,diagonal",
                        ),
                        max_candidates=int(max_retry_candidates),
                    )
                )
                for retry_candidate in retry_selection.retry_candidates:
                    retry_start_s = sparse_timer.elapsed_s()
                    try:
                        retry_pc = build_active_projected_rhs1_full_csr_preconditioner(
                            matrix=direct_tail_operator_bundle.matrix,
                            layout=direct_tail_structured_layout,
                            active_indices=direct_tail_structured_active_indices,
                            kind=str(retry_candidate),
                            max_factor_nbytes=int(direct_tail_structured_max_nbytes),
                            regularization=float(pc_reg),
                            preconditioner_x=int(preconditioner_x),
                            preconditioner_xi=int(preconditioner_xi),
                            preconditioner_species=int(preconditioner_species),
                            preconditioner_x_min_l=int(preconditioner_x_min_l),
                        )
                    except Exception as exc:  # noqa: BLE001
                        auto_preflight_retry_attempts.append(
                            {
                                "kind": str(retry_candidate),
                                "selected": False,
                                "reason": "exception",
                                "error": f"{type(exc).__name__}: {exc}",
                                "setup_s": float(sparse_timer.elapsed_s() - retry_start_s),
                            }
                        )
                        continue
                    retry_entry: dict[str, object] = {
                        "kind": str(retry_candidate),
                        "selected": bool(retry_pc.selected),
                        "reason": str(retry_pc.reason),
                        "setup_s": float(retry_pc.setup_s),
                    }
                    if not bool(retry_pc.selected) or retry_pc.operator is None:
                        retry_entry["metadata"] = dict(retry_pc.metadata)
                        auto_preflight_retry_attempts.append(retry_entry)
                        continue
                    retry_factor_nbytes = retry_pc.metadata.get("factor_nbytes_actual")
                    if retry_factor_nbytes is None:
                        retry_factor_nbytes = retry_pc.metadata.get("factor_nbytes_estimate")
                    retry_bundle = _StructuredHostSparsePreconditionerBundle(
                        preconditioner=retry_pc,
                        operator=direct_tail_operator_bundle,
                        kind=str(retry_pc.kind),
                        factor_nbytes_estimate=None if retry_factor_nbytes is None else int(retry_factor_nbytes),
                        factor_nnz_estimate=None,
                        factor_s=float(retry_pc.setup_s),
                    )
                    try:
                        retry_x = jnp.asarray(
                            retry_bundle.solve(np.asarray(sparse_pc_rhs, dtype=np.float64)),
                            dtype=jnp.float64,
                        )
                        retry_residual_vec = sparse_pc_rhs - jnp.asarray(_mv_true_no_count(retry_x), dtype=jnp.float64)
                        retry_residual = float(jnp.linalg.norm(retry_residual_vec))
                    except Exception as exc:  # noqa: BLE001
                        retry_entry.update(
                            {
                                "preflight_error": f"{type(exc).__name__}: {exc}",
                                "preflight_passed": False,
                            }
                        )
                        auto_preflight_retry_attempts.append(retry_entry)
                        continue
                    retry_metadata = retry_pc.metadata if isinstance(retry_pc.metadata, dict) else {}
                    retry_evaluation = evaluate_sparse_pc_auto_preflight_retry(
                        SparsePCAutoPreflightRetryEvaluationContext(
                            residual_after=float(retry_residual),
                            target=float(target),
                            max_target_ratio=float(factor_preflight_max_target_ratio),
                            residual_before=factor_preflight_residual_before,
                            sparse_pc_linear_size=int(sparse_pc_linear_size),
                            preflight_required_min_size=int(structured_pc_preflight_required_min_size),
                            retry_kind=str(retry_pc.kind),
                            retry_metadata=retry_metadata,
                        )
                    )
                    retry_entry.update(
                        {
                            "selected_kind": str(retry_pc.kind),
                            "preflight_required": bool(retry_evaluation.required),
                            "preflight_requires_metadata": bool(retry_evaluation.requires_metadata),
                            "preflight_requires_size": bool(retry_evaluation.requires_size),
                            "preflight_residual_after": float(retry_residual),
                            "preflight_target_ratio": float(retry_evaluation.target_ratio),
                            "preflight_passed": bool(retry_evaluation.preflight_passed),
                            "preflight_policy_passed": bool(retry_evaluation.policy_passed),
                            "factor_nbytes_estimate": (
                                None if retry_factor_nbytes is None else int(retry_factor_nbytes)
                            ),
                        }
                    )
                    auto_preflight_retry_attempts.append(retry_entry)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: auto preflight retry "
                            f"kind={retry_candidate} residual={float(retry_residual):.6e} "
                            f"target_ratio={float(retry_evaluation.target_ratio):.6e} "
                            f"required={bool(retry_evaluation.required)} "
                            f"passed={bool(retry_evaluation.preflight_passed)} "
                            f"policy_passed={bool(retry_evaluation.policy_passed)}",
                        )
                    if bool(retry_evaluation.policy_passed):
                        factor_bundle_pc = retry_bundle
                        direct_tail_structured_pc_selected = True
                        direct_tail_structured_pc_reason = (
                            f"auto_preflight_selected:{retry_pc.reason}"
                            if bool(retry_evaluation.required)
                            else f"auto_retry_selected_no_required_preflight:{retry_pc.reason}"
                        )
                        direct_tail_structured_pc_metadata = retry_pc.to_dict()
                        _operator_bundle_pc = direct_tail_operator_bundle
                        pc_factor_s += float(retry_pc.setup_s)
                        setup_s = sparse_timer.elapsed_s()
                        residual_vec_current = retry_residual_vec
                        factor_preflight_residual_after = float(retry_residual)
                        factor_preflight_residual_diagnostics = _rhs1_active_reduced_residual_diagnostics(
                            residual=retry_residual_vec,
                            layout=RHS1BlockLayout.from_operator(op),
                            active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                        )
                        if factor_preflight_residual_before is not None and float(factor_preflight_residual_before) > 0.0:
                            factor_preflight_improvement_ratio = float(factor_preflight_residual_before) / max(
                                float(factor_preflight_residual_after),
                                1.0e-300,
                        )
                        factor_preflight_target_ratio = float(retry_evaluation.target_ratio)
                        factor_preflight_passed = True
                        if bool(factor_preflight_seed_enabled):
                            x0_sparse = retry_x
                            factor_preflight_seed_used = True
                        auto_preflight_retry_selected = True
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: auto preflight retry accepted "
                                f"kind={retry_candidate} required={bool(retry_evaluation.required)}",
                            )
                        break
            if auto_preflight_retry_attempts:
                if isinstance(direct_tail_structured_pc_metadata, dict):
                    metadata_inner = direct_tail_structured_pc_metadata.setdefault("metadata", {})
                    if isinstance(metadata_inner, dict):
                        metadata_inner["auto_preflight_retry_enabled"] = bool(auto_preflight_retry_enabled)
                        metadata_inner["auto_preflight_retry_selected"] = bool(auto_preflight_retry_selected)
                        metadata_inner["auto_preflight_retry_attempts"] = tuple(
                            dict(entry) for entry in auto_preflight_retry_attempts
                        )
            if bool(structured_pc_ready) and bool(structured_pc_preflight_required) and factor_preflight_passed is False:
                raise RuntimeError(
                    "direct-tail structured preconditioner preflight failed: "
                    f"kind={getattr(factor_bundle_pc, 'kind', 'unknown')} "
                    f"residual_before={factor_preflight_residual_before} "
                    f"residual_after={factor_preflight_residual_after} "
                    f"target={float(target):.6e} "
                    f"target_ratio={factor_preflight_target_ratio} "
                    f"max_target_ratio={float(factor_preflight_max_target_ratio):.6e} "
                    f"error={factor_preflight_error}"
                )

        sparse_pc_gmres_policy = resolve_sparse_pc_gmres_control_policy(os.environ)
        sparse_pc_stagnation_abort = bool(sparse_pc_gmres_policy.stagnation_abort)
        sparse_pc_stagnation_min_iter = int(sparse_pc_gmres_policy.stagnation_min_iter)
        sparse_pc_stagnation_window = int(sparse_pc_gmres_policy.stagnation_window)
        sparse_pc_stagnation_rel_improvement = float(sparse_pc_gmres_policy.stagnation_rel_improvement)
        sparse_pc_post_minres_steps = int(sparse_pc_gmres_policy.post_minres_steps)
        sparse_pc_post_minres_alpha_clip = float(sparse_pc_gmres_policy.post_minres_alpha_clip)
        sparse_pc_post_minres_min_improvement = float(sparse_pc_gmres_policy.post_minres_min_improvement)

        def _run_sparse_pc_gmres_once(x0_arg, *, maxiter_arg: int):
            sparse_pc_gmres = run_sparse_pc_gmres_once(
                context=SparsePCGMRESContext(
                    matvec=_mv_true,
                    rhs=sparse_pc_rhs,
                    preconditioner=_precond_sparse,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    pc_form=pc_form,
                    restart=int(pc_restart),
                    tol=float(tol),
                    atol=float(atol),
                    precondition_side=precondition_side,
                    factor_dtype=np.dtype(sparse_pc_factor_dtype_used),
                    progress_every=int(progress_every),
                    stagnation_abort=bool(sparse_pc_stagnation_abort),
                    stagnation_min_iter=int(sparse_pc_stagnation_min_iter),
                    stagnation_window=int(sparse_pc_stagnation_window),
                    stagnation_rel_improvement=float(sparse_pc_stagnation_rel_improvement),
                    explicit_left_solver=explicit_left_preconditioned_gmres_scipy,
                    gmres_solver=gmres_solve_with_history_scipy,
                ),
                x0=x0_arg,
                maxiter=int(maxiter_arg),
            )
            return (
                sparse_pc_gmres.x,
                float(sparse_pc_gmres.residual_norm),
                float(sparse_pc_gmres.preconditioned_residual_norm),
                sparse_pc_gmres.history,
                float(sparse_pc_gmres.solve_s),
            )

        x_np, residual_norm_sparse_pc, rn_pc, history, solve_s = _run_sparse_pc_gmres_once(
            x0_sparse,
            maxiter_arg=sparse_pc_first_attempt_maxiter,
        )
        sparse_pc_direct_tail_metadata = sparse_pc_direct_tail_final_metadata(
            SparsePCDirectTailFinalMetadataContext(
                structured_pc_preflight_required=bool(
                    structured_pc_preflight_required
                ),
                structured_pc_preflight_required_min_size=int(
                    structured_pc_preflight_required_min_size
                ),
                materialization=direct_tail_materialization,
                structured_admission=direct_tail_structured_admission,
                residual_policy=direct_tail_residual_rescue_policy,
                true_active_policy=direct_tail_true_active_rescue_policy,
                coupled_coarse_policy=direct_tail_true_coupled_coarse_policy,
                true_window_specs=tuple(
                    tuple(int(value) for value in spec)
                    for spec in direct_tail_true_window_specs
                ),
                true_active_block_species_count=direct_tail_true_active_block_species_count,
                structured_max_nbytes=direct_tail_structured_max_nbytes,
                structured_pc_selected=bool(direct_tail_structured_pc_selected),
                structured_pc_reason=direct_tail_structured_pc_reason,
                structured_pc_error=direct_tail_structured_pc_error,
                structured_pc_metadata=direct_tail_structured_pc_metadata,
                support_mode_preflight_requested=bool(
                    direct_tail_support_mode_preflight_requested
                ),
                support_mode_preflight_selected=bool(
                    direct_tail_support_mode_preflight_selected
                ),
                support_mode_preflight_error=direct_tail_support_mode_preflight_error,
                support_mode_preflight_metadata=direct_tail_support_mode_preflight_metadata,
                residual_coarse_selected=bool(direct_tail_residual_coarse_selected),
                residual_coarse_residual_after=direct_tail_residual_coarse_residual_after,
                residual_coarse_error=direct_tail_residual_coarse_error,
                residual_coarse_metadata=direct_tail_residual_coarse_metadata,
                true_coupled_coarse_requested=bool(
                    direct_tail_true_coupled_coarse_requested
                ),
                true_coupled_coarse_auto_selected=bool(
                    direct_tail_true_coupled_coarse_auto_selected
                ),
                true_coupled_coarse_selected=bool(
                    direct_tail_true_coupled_coarse_selected
                ),
                true_coupled_coarse_residual_after=(
                    direct_tail_true_coupled_coarse_residual_after
                ),
                true_coupled_coarse_error=direct_tail_true_coupled_coarse_error,
                true_coupled_coarse_metadata=direct_tail_true_coupled_coarse_metadata,
                true_coupled_coarse_base_improvement_override_used=bool(
                    direct_tail_true_coupled_coarse_base_improvement_override_used
                ),
                true_active_submatrix_selected=bool(
                    direct_tail_true_active_submatrix_selected
                ),
                true_active_submatrix_residual_after=(
                    direct_tail_true_active_submatrix_residual_after
                ),
                true_active_submatrix_error=direct_tail_true_active_submatrix_error,
                true_active_submatrix_metadata=direct_tail_true_active_submatrix_metadata,
                true_active_column_cache_metadata=direct_tail_true_active_column_cache_metadata,
                true_active_block_selected=bool(direct_tail_true_active_block_selected),
                true_active_block_residual_after=direct_tail_true_active_block_residual_after,
                true_active_block_error=direct_tail_true_active_block_error,
                true_active_block_metadata=direct_tail_true_active_block_metadata,
                true_active_residual_block_selected=bool(
                    direct_tail_true_active_residual_block_selected
                ),
                true_active_residual_block_residual_after=(
                    direct_tail_true_active_residual_block_residual_after
                ),
                true_active_residual_block_error=(
                    direct_tail_true_active_residual_block_error
                ),
                true_active_residual_block_metadata=(
                    direct_tail_true_active_residual_block_metadata
                ),
                true_active_residual_block_base_improvement_override_used=bool(
                    direct_tail_true_active_residual_block_base_improvement_override_used
                ),
                true_window_selected=bool(direct_tail_true_window_selected),
                true_window_residual_after=direct_tail_true_window_residual_after,
                true_window_error=direct_tail_true_window_error,
                true_window_metadata=direct_tail_true_window_metadata,
                residual_window_selected=bool(direct_tail_residual_window_selected),
                residual_window_residual_after=direct_tail_residual_window_residual_after,
                residual_window_error=direct_tail_residual_window_error,
                residual_window_metadata=direct_tail_residual_window_metadata,
            )
        )
        sparse_pc_factor_preflight_metadata = (
            sparse_pc_factor_preflight_result_metadata_from_context(
                SparsePCFactorPreflightMetadataContext(
                    enabled=bool(factor_preflight_enabled),
                    required=bool(factor_preflight_required),
                    seed_enabled=bool(factor_preflight_seed_enabled),
                    seed_used=bool(factor_preflight_seed_used),
                    passed=factor_preflight_passed,
                    error=factor_preflight_error,
                    residual_before=factor_preflight_residual_before,
                    residual_after=factor_preflight_residual_after,
                    improvement_ratio=factor_preflight_improvement_ratio,
                    target_ratio=factor_preflight_target_ratio,
                    max_target_ratio=float(factor_preflight_max_target_ratio),
                    residual_diagnostics=factor_preflight_residual_diagnostics,
                )
            )
        )
        sparse_pc_pattern_metadata = sparse_pc_pattern_result_metadata_from_context(
            SparsePCPatternMetadataContext(
                summary=summary,
                scope=sparse_pattern_scope,
                build_s=float(pattern_build_s),
            )
        )
        sparse_pc_static_metadata = sparse_pc_gmres_static_metadata_from_context(
            SparsePCGMRESStaticMetadataContext(
                op=op,
                fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                fortran_reduced_sparse_pc_backend=fortran_reduced_sparse_pc_backend,
                fortran_reduced_sparse_pc_backend_reason=(
                    fortran_reduced_sparse_pc_backend_reason
                ),
                fortran_reduced_xblock_min_size=fortran_reduced_xblock_min_size,
                pc_restart=int(pc_restart),
                pc_maxiter=int(pc_maxiter),
                sparse_pc_first_attempt_maxiter=int(sparse_pc_first_attempt_maxiter),
                pc_shift=float(pc_shift),
                sparse_pc_factor_dtype_initial=sparse_pc_factor_dtype_initial,
                sparse_pc_preconditioner_operator=sparse_pc_preconditioner_operator,
                sparse_pc_factorization=sparse_pc_factorization,
                sparse_pc_default_factor_kind=sparse_pc_default_factor_kind,
                sparse_pc_default_ilu_fill_factor=float(
                    sparse_pc_default_ilu_fill_factor
                ),
                sparse_pc_default_ilu_drop_tol=float(sparse_pc_default_ilu_drop_tol),
                sparse_pc_default_pattern_color_batch=int(
                    sparse_pc_default_pattern_color_batch
                ),
                preconditioner_x=int(preconditioner_x),
                preconditioner_x_min_l=int(preconditioner_x_min_l),
                preconditioner_xi=int(preconditioner_xi),
                preconditioner_species=int(preconditioner_species),
                sparse_pc_permc_spec=sparse_pc_permc_spec,
                sparse_pc_default_permc_spec=sparse_pc_default_permc_spec,
                sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                sparse_pc_linear_size=int(sparse_pc_linear_size),
                sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
            )
        )
        sparse_pc_finalization_state = sparse_pc_gmres_finalization_state_from_context(
            SparsePCGMRESFinalizationStateContext(
                atol=atol,
                mv_count=mv_count,
                rhs_norm=rhs_norm,
                target=target,
                tol=tol,
                sparse_pc_direct_tail_metadata=sparse_pc_direct_tail_metadata,
                sparse_pc_factor_preflight_metadata=(
                    sparse_pc_factor_preflight_metadata
                ),
                sparse_pc_pattern_metadata=sparse_pc_pattern_metadata,
                sparse_pc_static_metadata=sparse_pc_static_metadata,
            )
        )
        sparse_pc_final_payload = finalize_sparse_pc_gmres_with_dtype_retry(
            SparsePCGMRESFinalizationContext(
                diagnostic_state=sparse_pc_finalization_state,
                result=SparsePCGMRESResult(
                    x=np.asarray(x_np, dtype=np.float64),
                    residual_norm=float(residual_norm_sparse_pc),
                    preconditioned_residual_norm=float(rn_pc),
                    history=tuple(float(v) for v in (history or ())),
                    solve_s=float(solve_s),
                ),
                factor_dtype_used=np.dtype(sparse_pc_factor_dtype_used),
                factor_dtype_retry=sparse_pc_factor_dtype_retry,
                operator_bundle=_operator_bundle_pc,
                factor_bundle=factor_bundle_pc,
                pc_factor_s=float(pc_factor_s),
                setup_s=float(setup_s),
                post_minres=SparsePCPostMinresFinalizationContext(
                    matvec=_mv_true,
                    rhs=sparse_pc_rhs,
                    preconditioner=_precond_sparse,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    pc_form=pc_form,
                    steps=int(sparse_pc_post_minres_steps),
                    alpha_clip=float(sparse_pc_post_minres_alpha_clip),
                    min_improvement=float(sparse_pc_post_minres_min_improvement),
                    target=float(target),
                ),
                dtype_retry=SparsePCFactorDtypeRetryFinalizationContext(
                    factor_matvec=_sparse_pc_factor_mv,
                    linear_size=int(sparse_pc_linear_size),
                    rhs_dtype=np.dtype(rhs.dtype),
                    pattern=pattern,
                    emit=emit,
                    constrained_pas_pc=bool(constrained_pas_pc),
                    tokamak_fp_pc=bool(tokamak_fp_pc),
                    fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                    default_permc_spec=sparse_pc_default_permc_spec,
                    default_factor_kind=sparse_pc_default_factor_kind,
                    default_ilu_fill_factor=float(sparse_pc_default_ilu_fill_factor),
                    default_ilu_drop_tol=float(sparse_pc_default_ilu_drop_tol),
                    default_pattern_color_batch=int(
                        sparse_pc_default_pattern_color_batch
                    ),
                    x0_fallback=x0_sparse,
                    pc_maxiter=int(pc_maxiter),
                    elapsed_s=sparse_timer.elapsed_s,
                ),
            ),
            build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
            run_sparse_pc_gmres_once_callback=_run_sparse_pc_gmres_once,
            minres_correction=_apply_preconditioned_minres_correction,
            expand_reduced=_sparse_pc_expand_reduced,
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=sparse_pc_final_payload,
        )
    if solve_method_kind_explicit in _SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS:
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_lsmr",
            differentiable=differentiable,
            rhs_mode=int(op.rhs_mode),
            use_active_dof=bool(use_active_dof_mode),
            path_description="host sparse minimum-norm path",
        )
        sparse_timer = Timer()
        pattern = v3_full_system_conservative_sparsity_pattern(op)
        summary = summarize_v3_sparse_pattern(op, pattern)

        def _sparse_min_norm_mv(x_np: np.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator_cached(op, jnp.asarray(x_np, dtype=rhs.dtype))

        def _matvec_np(x_np: np.ndarray) -> np.ndarray:
            return np.asarray(_sparse_min_norm_mv(np.asarray(x_np, dtype=np.float64)), dtype=np.float64)

        sparse_minimum_norm_payload = sparse_minimum_norm_solve_from_pattern(
            matvec_np=_matvec_np,
            pattern=pattern,
            summary=summary,
            rhs=rhs,
            solve_method_kind=solve_method_kind_explicit,
            tol=float(tol),
            atol=float(atol),
            maxiter=maxiter,
            rhs_norm=float(rhs_norm),
            elapsed_s=sparse_timer.elapsed_s,
            backend=jax.default_backend(),
            env=os.environ,
            emit=emit,
            build_operator_from_pattern=build_operator_from_pattern,
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=sparse_minimum_norm_payload,
        )
    if solve_method_kind_explicit in _SPARSE_HOST_DIRECT_SOLVE_METHODS:
        validate_explicit_sparse_host_request(
            solve_method_label="sparse_host",
            differentiable=differentiable,
            rhs_mode=int(op.rhs_mode),
            use_active_dof=bool(use_active_dof_mode),
            path_description="host sparse LU path",
        )
        sparse_timer = Timer()
        pattern = v3_full_system_conservative_sparsity_pattern(op)
        summary = summarize_v3_sparse_pattern(op, pattern)

        def _sparse_host_mv(x_np: np.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator(op, jnp.asarray(x_np, dtype=rhs.dtype))

        sparse_host_direct_payload = sparse_host_direct_solve_from_pattern(
            matvec=_sparse_host_mv,
            pattern=pattern,
            summary=summary,
            n=int(op.total_size),
            dtype=rhs.dtype,
            factor_dtype=np.dtype(np.float64),
            rhs=rhs,
            refine_steps=_host_sparse_direct_refine_steps(
                "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_REFINE",
                default=2,
            ),
            atol=float(atol),
            tol=float(tol),
            rhs_norm=float(rhs_norm),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
            direct_solve_with_refinement=_host_direct_solve_with_refinement,
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=sparse_host_direct_payload,
        )
    rhs1_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "").strip().lower()
    rhs1_precond_env_user = rhs1_precond_env
    rhs1_bicgstab_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND", "").strip().lower()
    rhs1_bicgstab_env_user = rhs1_bicgstab_env
    rhs1_precond_env = _rhs1_fp_dkes_env_preconditioner_kind(
        rhs1_precond_env=rhs1_precond_env,
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        total_size=int(op.total_size),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
    )
    try:
        pre_theta = int(precond_opts.get("PRECONDITIONER_THETA", 0) or 0)
    except (TypeError, ValueError):
        pre_theta = 0
    try:
        pre_zeta = int(precond_opts.get("PRECONDITIONER_ZETA", 0) or 0)
    except (TypeError, ValueError):
        pre_zeta = 0
    rhs1_precond_kind: str | None
    rhs1_xblock_tz_lmax: int | None = None
    rhs1_gpu_tokamak_pas_tight_gmres = False
    rhs1_dd_setup = resolve_rhs1_domain_decomposition_setup(
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        sum_nxi=int(np.sum(nxi_for_x)) if nxi_for_x.size else 1,
        distributed_env=os.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED", ""),
        device_count=int(jax.device_count()),
        auto_axis=_matvec_shard_axis(op),
        theta_block_env=os.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_T", ""),
        zeta_block_env=os.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_Z", ""),
        theta_overlap_env=os.environ.get("SFINCS_JAX_RHSMODE1_DD_OVERLAP_T", ""),
        zeta_overlap_env=os.environ.get("SFINCS_JAX_RHSMODE1_DD_OVERLAP_Z", ""),
        overlap_env=os.environ.get("SFINCS_JAX_RHSMODE1_DD_OVERLAP", ""),
        patch_dof_target_env=os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_PATCH_DOF_TARGET", ""),
    )

    pas_auto_strong_ratio_env = os.environ.get("SFINCS_JAX_PAS_AUTO_STRONG_RATIO", "").strip()
    try:
        pas_auto_strong_ratio = float(pas_auto_strong_ratio_env) if pas_auto_strong_ratio_env else 10.0
    except ValueError:
        pas_auto_strong_ratio = 10.0
    er_abs = 0.0
    schur_er_min = 1.0e-12

    if rhs1_precond_env:
        rhs1_precond_kind = _canonical_rhs1_preconditioner_kind(rhs1_precond_env)
    else:
        # Default to v3-like preconditioner options: when preconditioner_theta/zeta are 0,
        # use point-block Jacobi. Enable line preconditioning only when explicitly requested.
        if int(op.rhs_mode) == 1 and (not bool(op.include_phi1)):
            if pre_theta == 0 and pre_zeta == 0:
                tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                try:
                    tz_max = int(tz_max_env) if tz_max_env else 128
                except ValueError:
                    tz_max = 128
                xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                default_xblock_tz_max = 1200
                if op.fblock.pas is not None and geom_scheme == 1:
                    default_xblock_tz_max = 6000
                elif op.fblock.pas is not None:
                    default_xblock_tz_max = 2000
                try:
                    xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else default_xblock_tz_max
                except ValueError:
                    xblock_tz_max = default_xblock_tz_max
                xblock_tz_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "").strip()
                try:
                    xblock_tz_lmax_override = int(xblock_tz_lmax_env) if xblock_tz_lmax_env else 0
                except ValueError:
                    xblock_tz_lmax_override = 0
                species_block_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "").strip()
                try:
                    species_block_max = int(species_block_max_env) if species_block_max_env else 1600
                except ValueError:
                    species_block_max = 1600
                nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                lmax_auto = 0
                if int(op.n_theta) > 0 and int(op.n_zeta) > 0:
                    lmax_auto = int(xblock_tz_max // (int(op.n_theta) * int(op.n_zeta)))
                lmax_auto = max(0, min(max_l, lmax_auto))
                local_per_species = int(np.sum(nxi_for_x))
                dke_size = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
                line_size = int(local_per_species * int(op.n_theta))
                sxblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_MAX", "").strip()
                try:
                    sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
                except ValueError:
                    sxblock_max = 64
                sxblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_MAX", "").strip()
                try:
                    sxblock_tz_max = int(sxblock_tz_max_env) if sxblock_tz_max_env else 0
                except ValueError:
                    sxblock_tz_max = 0
                sxblock_tz_active_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_ACTIVE_MAX", "").strip()
                try:
                    sxblock_tz_active_max = int(sxblock_tz_active_max_env) if sxblock_tz_active_max_env else 20000
                except ValueError:
                    sxblock_tz_active_max = 20000
                if sxblock_tz_max == 0 and op.fblock.fp is not None and (
                    int(op.n_theta) > 1 or int(op.n_zeta) > 1
                ):
                    # Allow a modest FP sxblock_tz preconditioner in multi-angle FP cases
                    # to avoid RHSMode=1 stagnation without large dense fallbacks.
                    sxblock_tz_max = 2000
                sxblock_size = int(int(op.n_species) * local_per_species)
                sxblock_tz_size = int(int(op.n_species) * int(op.n_x) * int(op.n_theta) * int(op.n_zeta))
                schur_auto = False
                if (
                    int(op.constraint_scheme) == 2
                    and int(op.extra_size) > 0
                    and op.fblock.pas is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                ):
                    schur_auto_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN", "").strip()
                    try:
                        schur_auto_min = int(schur_auto_min_env) if schur_auto_min_env else 2500
                    except ValueError:
                        schur_auto_min = 2500
                    schur_auto = int(op.total_size) >= schur_auto_min
                phys_params = nml.group("physicsParameters")
                er_val = phys_params.get("ER", phys_params.get("Er", phys_params.get("er", None)))
                er_abs = 0.0
                if er_val is not None:
                    try:
                        er_abs = float(er_val)
                    except (TypeError, ValueError):
                        er_abs = 0.0
                er_abs = abs(er_abs)
                epar_val = phys_params.get("EPARALLELHAT", phys_params.get("EParallelHat", None))
                try:
                    epar_abs = abs(float(epar_val)) if epar_val is not None else 0.0
                except (TypeError, ValueError):
                    epar_abs = 0.0
                if epar_abs > 0.0 and sxblock_tz_max == 0:
                    sxblock_tz_max = 2000
                schur_er_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN", "").strip()
                try:
                    schur_er_min = float(schur_er_env) if schur_er_env else 1.0e-12
                except ValueError:
                    schur_er_min = 1.0e-12
                pas_dkes_gpu_xblock_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_XBLOCK_TZ_MAX", "").strip()
                try:
                    pas_dkes_gpu_xblock_max = int(pas_dkes_gpu_xblock_env) if pas_dkes_gpu_xblock_env else 2500
                except ValueError:
                    pas_dkes_gpu_xblock_max = 2500
                pas_xdiag_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_XDIAG_MIN", "").strip()
                try:
                    pas_xdiag_min = int(pas_xdiag_env) if pas_xdiag_env else 1000000000
                except ValueError:
                    pas_xdiag_min = 1000000000
                pas_xmg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", "").strip()
                try:
                    pas_xmg_min = int(pas_xmg_env) if pas_xmg_env else 80000
                except ValueError:
                    pas_xmg_min = 80000
                fp_xmg_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XMG_MAX", "").strip()
                try:
                    # Keep xmg as the default for larger FP systems as long as we are still
                    # in the matrix-free Krylov regime; this avoids expensive Schwarz builds
                    # that can dominate runtime in high-resolution single-RHS runs.
                    fp_xmg_max = int(fp_xmg_env) if fp_xmg_env else 200000
                except ValueError:
                    fp_xmg_max = 200000
                schur_tokamak_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK", "").strip().lower()
                schur_tokamak = schur_tokamak_env in {"1", "true", "yes", "on"}
                tokamak_like = int(op.n_zeta) == 1 or geom_scheme == 1
                if full_precond_requested and int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                    if tokamak_like and schur_tokamak and er_abs <= schur_er_min:
                        rhs1_precond_kind = "schur"
                    elif tokamak_like and (not schur_tokamak) and er_abs <= schur_er_min:
                        if op.fblock.pas is not None:
                            # For tiny tokamak PAS systems, prefer the xblock_tz preconditioner
                            # (matches legacy fixtures). For larger systems, keep the lighter
                            # PAS hybrid to avoid expensive dense angular blocks.
                            xblock_small_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", "").strip()
                            try:
                                xblock_small_max = int(xblock_small_env) if xblock_small_env else 4000
                            except ValueError:
                                xblock_small_max = 4000
                            if (
                                int(op.total_size) <= max(1, int(xblock_small_max))
                                and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= max(1, int(xblock_tz_max))
                            ):
                                rhs1_precond_kind = "xblock_tz"
                            else:
                                # Tokamak-like PAS systems benefit from the PAS hybrid (line + x-coarse)
                                # preconditioner; avoid the expensive global Schur/xblock path here.
                                rhs1_precond_kind = "pas_hybrid"
                        else:
                            pas_schur_small_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_SMALL_MAX", "").strip()
                            try:
                                pas_schur_small_max = int(pas_schur_small_env) if pas_schur_small_env else 20000
                            except ValueError:
                                pas_schur_small_max = 20000
                            if int(op.total_size) <= max(1, int(pas_schur_small_max)):
                                rhs1_precond_kind = "schur"
                            elif (
                                int(op.n_theta) > 1
                                and xblock_tz_max > 0
                                and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                            ):
                                rhs1_precond_kind = "xblock_tz"
                            else:
                                rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                    else:
                        if (
                            op.fblock.pas is not None
                            and er_abs <= schur_er_min
                            and (not schur_tokamak)
                            and int(op.total_size) < pas_xmg_min
                        ):
                            # For constrained PAS near-zero-Er systems below the Schur regime,
                            # prefer a lightweight PAS preconditioner when angular blocks are
                            # modest; otherwise fall back to x-coarsening to avoid expensive
                            # global Schur setup while retaining good Krylov convergence.
                            tz_size = int(op.n_theta) * int(op.n_zeta)
                            pas_lite_tz_env = os.environ.get("SFINCS_JAX_PAS_LITE_TZ_MAX", "").strip()
                            try:
                                pas_lite_tz_max = int(pas_lite_tz_env) if pas_lite_tz_env else 256
                            except ValueError:
                                pas_lite_tz_max = 256
                            if _pas_tz_preconditioner_applicable(op) and pas_lite_tz_max > 0 and tz_size <= pas_lite_tz_max:
                                pas_lite_min_env = os.environ.get("SFINCS_JAX_PAS_LITE_MIN", "").strip()
                                try:
                                    pas_lite_min = int(pas_lite_min_env) if pas_lite_min_env else 20000
                                except ValueError:
                                    pas_lite_min = 20000
                                if int(active_size) >= max(1, int(pas_lite_min)):
                                    rhs1_precond_kind = "pas_lite"
                                else:
                                    rhs1_precond_kind = "pas_hybrid"
                            else:
                                rhs1_precond_kind = "xmg"
                        elif (
                            op.fblock.fp is not None
                            and er_abs <= schur_er_min
                            and int(op.total_size) < fp_xmg_max
                        ):
                            rhs1_precond_kind = "xmg"
                        elif _rhs1_pas_tokamak_gpu_xblock_preferred(
                            has_pas=op.fblock.pas is not None,
                            has_fp=op.fblock.fp is not None,
                            backend=jax.default_backend(),
                            tokamak_like=tokamak_like,
                            active_size=int(active_size),
                            er_abs=float(er_abs),
                            schur_er_min=float(schur_er_min),
                            has_magdrift=(
                                op.fblock.magdrift_theta is not None
                                or op.fblock.magdrift_zeta is not None
                                or op.fblock.magdrift_xidot is not None
                            ),
                            has_collisionless=op.fblock.collisionless is not None,
                            n_theta=int(op.n_theta),
                            n_zeta=int(op.n_zeta),
                            max_l=int(max_l),
                            xblock_tz_limit=max(int(xblock_tz_max), int(pas_dkes_gpu_xblock_max)),
                        ):
                            rhs1_precond_kind = "xblock_tz"
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                                    "auto -> xblock_tz preconditioner",
                                )
                        elif _rhs1_pas_tokamak_gpu_theta_allowed(
                            has_pas=op.fblock.pas is not None,
                            has_fp=op.fblock.fp is not None,
                            backend=jax.default_backend(),
                            tokamak_like=tokamak_like,
                            active_size=int(active_size),
                            er_abs=float(er_abs),
                            schur_er_min=float(schur_er_min),
                            has_magdrift=(
                                op.fblock.magdrift_theta is not None
                                or op.fblock.magdrift_zeta is not None
                                or op.fblock.magdrift_xidot is not None
                            ),
                            has_collisionless=op.fblock.collisionless is not None,
                        ):
                            rhs1_precond_kind = None
                            rhs1_gpu_tokamak_pas_tight_gmres = True
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                                    "auto -> tight unpreconditioned GMRES",
                                )
                        elif op.fblock.pas is not None and int(op.total_size) >= pas_xmg_min:
                            # Large constrained PAS+Er systems need stronger x/L coupling than
                            # collision/point/xmg alone, but a global Schur setup can dominate
                            # wall time. Keep the auto path in the PAS-native family here so
                            # the later tokamak/3D refinements can promote to pas_tz/pas_ilu.
                            rhs1_precond_kind = _rhs1_pas_auto_large_base_kind(active_size=int(active_size))
                        elif op.fblock.pas is not None and int(op.total_size) >= pas_xdiag_min:
                            lmax_use = xblock_tz_lmax_override if xblock_tz_lmax_override > 0 else lmax_auto
                            if lmax_use >= 1:
                                rhs1_precond_kind = "xblock_tz_lmax"
                                rhs1_xblock_tz_lmax = int(lmax_use)
                            else:
                                rhs1_precond_kind = "point_xdiag"
                        else:
                            if op.fblock.pas is not None:
                                rhs1_precond_kind = _rhs1_pas_auto_large_base_kind(active_size=int(active_size))
                            else:
                                rhs1_precond_kind = "schur"
                elif full_precond_requested and (int(op.n_theta) > 1 or int(op.n_zeta) > 1):
                    if (
                        op.fblock.pas is not None
                        and er_abs <= schur_er_min
                        and (not schur_tokamak)
                        and int(op.total_size) < pas_xmg_min
                    ):
                        tz_size = int(op.n_theta) * int(op.n_zeta)
                        pas_lite_tz_env = os.environ.get("SFINCS_JAX_PAS_LITE_TZ_MAX", "").strip()
                        try:
                            pas_lite_tz_max = int(pas_lite_tz_env) if pas_lite_tz_env else 256
                        except ValueError:
                            pas_lite_tz_max = 256
                        if _pas_tz_preconditioner_applicable(op) and pas_lite_tz_max > 0 and tz_size <= pas_lite_tz_max:
                            pas_lite_min_env = os.environ.get("SFINCS_JAX_PAS_LITE_MIN", "").strip()
                            try:
                                pas_lite_min = int(pas_lite_min_env) if pas_lite_min_env else 20000
                            except ValueError:
                                pas_lite_min = 20000
                            if int(active_size) >= max(1, int(pas_lite_min)):
                                rhs1_precond_kind = "pas_lite"
                            else:
                                rhs1_precond_kind = "pas_hybrid"
                        else:
                            rhs1_precond_kind = "xmg"
                    elif (
                        op.fblock.fp is not None
                        and er_abs <= schur_er_min
                        and int(op.total_size) < fp_xmg_max
                    ):
                        rhs1_precond_kind = "xmg"
                    else:
                        rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                elif schur_auto:
                    # For sharded multi-device PAS near-zero-Er runs, Schur can become
                    # communication-dominated as device count increases. Prefer x-coarsening
                    # here to keep Krylov/preconditioner cost closer to shard-local.
                    shard_axis_auto = _matvec_shard_axis(op)
                    if (
                        op.fblock.pas is not None
                        and (not bool(op.include_phi1))
                        and float(er_abs) <= float(schur_er_min)
                        and shard_axis_auto in {"theta", "zeta"}
                        and jax.device_count() > 1
                        and int(op.total_size) <= max(1, int(pas_xmg_min))
                    ):
                        rhs1_precond_kind = "xmg"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: sharded PAS near-zero-Er "
                                "schur_auto -> xmg preconditioner",
                            )
                    elif _rhs1_pas_dkes_pas_tz_preferred(
                        has_pas=op.fblock.pas is not None,
                        use_dkes=bool(use_dkes),
                        backend=jax.default_backend(),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        active_size=int(active_size),
                    ):
                        rhs1_precond_kind = "pas_tz"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: PAS DKES "
                                "schur_auto -> pas_tz preconditioner",
                            )
                    elif _rhs1_pas_dkes_xblock_allowed(
                        has_pas=op.fblock.pas is not None,
                        use_dkes=bool(use_dkes),
                        backend=jax.default_backend(),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        xblock_tz_limit=max(int(xblock_tz_max), int(pas_dkes_gpu_xblock_max)),
                    ):
                        rhs1_precond_kind = "xblock_tz"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: GPU PAS DKES "
                                "schur_auto -> xblock_tz preconditioner",
                            )
                    else:
                        rhs1_precond_kind = "schur"
                elif op.fblock.fp is not None and use_dkes:
                    # DKES-trajectory FP cases can stagnate with collision-only
                    # preconditioners. Prefer a lightweight xmg/sxblock_tz path for
                    # small/medium systems, and fall back to collision for larger sizes
                    # to avoid expensive block builds.
                    max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                    rhs1_precond_kind = _rhs1_fp_dkes_default_kind(
                        active_size=int(active_size),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        xblock_tz_limit=int(xblock_tz_max),
                    )
                    if rhs1_precond_kind == "xblock_tz" and not rhs1_precond_env:
                        rhs1_precond_env = "xblock_tz"
                elif (
                    op.fblock.fp is not None
                    and er_abs <= schur_er_min
                    and int(active_size) < fp_xmg_max
                ):
                    # For moderate-size FP systems at near-zero Er, x-coarsened preconditioning
                    # is typically much cheaper than global (S,X,theta,zeta) blocks and
                    # preserves parity for RHSMode=1.
                    rhs1_precond_kind = "xmg"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and sxblock_tz_max > 0
                    and int(op.total_size) <= max(1, int(sxblock_tz_active_max))
                    and sxblock_tz_size <= sxblock_tz_max
                ):
                    rhs1_precond_kind = "sxblock_tz"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                elif op.fblock.fp is not None and sxblock_max > 0 and sxblock_size <= sxblock_max:
                    rhs1_precond_kind = "sxblock"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and species_block_max > 0
                    and dke_size <= species_block_max
                ):
                    rhs1_precond_kind = "species_block"
                elif _rhs1_pas_dkes_pas_tz_preferred(
                    has_pas=op.fblock.pas is not None,
                    use_dkes=bool(use_dkes),
                    backend=jax.default_backend(),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    max_l=int(max_l),
                    active_size=int(active_size),
                ):
                    rhs1_precond_kind = "pas_tz"
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS DKES "
                            "auto -> pas_tz preconditioner",
                        )
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and xblock_tz_max > 0
                    and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                ):
                    rhs1_precond_kind = "xblock_tz"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                elif (
                    op.fblock.pas is not None
                    and int(active_size) >= pas_xmg_min
                ):
                    # Large PAS systems tend to be x/L-coupling dominated. Prefer the
                    # x-coarsened PAS preconditioner over weak collision/point
                    # preconditioners that often trigger expensive fallback branches.
                    rhs1_precond_kind = "xmg"
                else:
                    if (
                        op.fblock.pas is not None
                        and er_abs <= schur_er_min
                        and int(active_size) < pas_xmg_min
                    ):
                        tz_size = int(op.n_theta) * int(op.n_zeta)
                        pas_lite_tz_env = os.environ.get("SFINCS_JAX_PAS_LITE_TZ_MAX", "").strip()
                        try:
                            pas_lite_tz_max = int(pas_lite_tz_env) if pas_lite_tz_env else 256
                        except ValueError:
                            pas_lite_tz_max = 256
                        if _pas_tz_preconditioner_applicable(op) and pas_lite_tz_max > 0 and tz_size <= pas_lite_tz_max:
                            pas_lite_min_env = os.environ.get("SFINCS_JAX_PAS_LITE_MIN", "").strip()
                            try:
                                pas_lite_min = int(pas_lite_min_env) if pas_lite_min_env else 20000
                            except ValueError:
                                pas_lite_min = 20000
                            if int(active_size) >= max(1, int(pas_lite_min)):
                                rhs1_precond_kind = "pas_lite"
                            else:
                                rhs1_precond_kind = "pas_hybrid"
                        else:
                            rhs1_precond_kind = "xmg"
                    else:
                        collision_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "").strip()
                        try:
                            collision_precond_min = int(collision_precond_min_env) if collision_precond_min_env else 600
                        except ValueError:
                            collision_precond_min = 600
                        use_collision_precond = (
                            (op.fblock.fp is not None or op.fblock.pas is not None)
                            and int(op.total_size) >= collision_precond_min
                        )
                        if (
                            use_collision_precond
                            and full_precond_requested
                            and op.fblock.pas is not None
                            and int(op.total_size) >= pas_xdiag_min
                        ):
                            rhs1_precond_kind = "point_xdiag"
                        else:
                            # Last-resort auto mode: collision-only preconditioning is cheap but
                            # can be too weak/unstable for FP systems at nonzero Er. Prefer xmg
                            # when the (x,theta,zeta) grid is still moderate to improve robustness.
                            if (
                                op.fblock.fp is not None
                                and op.fblock.pas is None
                                and int(active_size) < fp_xmg_max
                            ):
                                rhs1_precond_kind = "xmg"
                            else:
                                rhs1_precond_kind = "collision" if use_collision_precond else "point"
                theta_line_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", "").strip()
                try:
                    theta_line_max = int(theta_line_max_env) if theta_line_max_env else 0
                except ValueError:
                    theta_line_max = 0
                if rhs1_precond_kind == "theta_line" and theta_line_max > 0 and line_size > theta_line_max:
                    rhs1_precond_kind = "theta_line_xdiag"
            elif pre_theta > 0 and pre_zeta > 0:
                rhs1_precond_kind = "adi"
            elif pre_theta > 0:
                rhs1_precond_kind = "theta_line"
            elif pre_zeta > 0:
                rhs1_precond_kind = "zeta_line"
            else:
                rhs1_precond_kind = "point"
        else:
            rhs1_precond_kind = None
    if (
        (not rhs1_precond_env)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and rhs1_precond_kind
        in {
            None,
            "collision",
            "point",
            "xmg",
            "theta_line",
            "zeta_line",
            "theta_zeta",
            "xblock_tz",
            "xblock_tz_lmax",
            "theta_line_xdiag",
        }
        and not rhs1_gpu_tokamak_pas_tight_gmres
    ):
        # PAS runs can stagnate with weak preconditioners; use the
        # PAS hybrid (line + x-coarse) preconditioner by default. For large systems,
        # prefer a lighter PAS preconditioner to keep setup cost down. The PAS probe
        # can still downgrade to a collision preconditioner if it suffices.
        max_l_local = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
        rhs1_precond_kind = _rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env=rhs1_precond_env,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            current_kind=rhs1_precond_kind,
            active_size=int(active_size),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l_local),
        )
        if (
            rhs1_precond_kind == "xblock_tz"
            and _rhs1_pas_dkes_pas_tz_preferred(
                has_pas=op.fblock.pas is not None,
                use_dkes=bool(use_dkes),
                backend=jax.default_backend(),
                n_theta=int(op.n_theta),
                n_zeta=int(op.n_zeta),
                max_l=int(max_l_local),
                active_size=int(active_size),
            )
        ):
            rhs1_precond_kind = "pas_tz"
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS DKES "
                    "weak-auto override -> pas_tz preconditioner",
                )
    tokamak_like = bool(geom_scheme == 1 or int(op.n_zeta) <= 5)
    rhs1_precond_kind = _rhs1_pas_family_refinement_kind(
        rhs1_precond_env=rhs1_precond_env,
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        current_kind=rhs1_precond_kind,
        active_size=int(active_size),
        n_zeta=int(op.n_zeta),
        geom_scheme=int(geom_scheme),
        pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
        pas_tokamak_theta_applicable=_pas_tokamak_theta_preconditioner_applicable(op),
    )
    if (
        rhs1_precond_env in {"", "auto", "default"}
        and rhs1_precond_kind == "schur"
        and _rhs1_pas_full_pas_tz_preferred(
            has_pas=op.fblock.pas is not None,
            has_fp=op.fblock.fp is not None,
            use_dkes=bool(use_dkes),
            backend=jax.default_backend(),
            geom_scheme=int(geom_scheme),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l),
            active_size=int(active_size),
            pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
        )
    ):
        rhs1_precond_kind = "pas_tz"
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: full-trajectory PAS "
                "auto -> pas_tz preconditioner",
            )
    if _rhs1_geometry4_pas_memory_pas_tz_preferred(
        rhs1_precond_env=rhs1_precond_env,
        current_kind=rhs1_precond_kind,
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        geom_scheme=int(geom_scheme),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        max_l=int(max_l),
        active_size=int(active_size),
        er_abs=float(er_abs),
        schur_er_min=float(schur_er_min),
        pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
    ):
        rhs1_precond_kind = "pas_tz"
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: geometry4 PAS memory "
                "auto -> pas_tz preconditioner",
            )
    if (
        tokamak_like
        and rhs1_precond_env in {"", "auto", "default"}
        and rhs1_precond_kind in {"pas_lite", "pas_hybrid", "pas_tokamak_theta", "pas_tz", "xmg", "collision", "point"}
        and op.fblock.pas is not None
        and op.fblock.fp is None
        and (not _pas_tz_preconditioner_applicable(op))
        and (not _pas_tokamak_theta_preconditioner_applicable(op))
        and (not _rhs1_pas_tokamak_gpu_theta_allowed(
            has_pas=op.fblock.pas is not None,
            has_fp=op.fblock.fp is not None,
            backend=jax.default_backend(),
            tokamak_like=tokamak_like,
            active_size=int(active_size),
            er_abs=float(er_abs),
            schur_er_min=float(schur_er_min),
            has_magdrift=(
                op.fblock.magdrift_theta is not None
                or op.fblock.magdrift_zeta is not None
                or op.fblock.magdrift_xidot is not None
            ),
            has_collisionless=op.fblock.collisionless is not None,
        ))
    ):
        if _rhs1_pas_tokamak_cpu_xblock_preferred(
            has_pas=op.fblock.pas is not None,
            has_fp=op.fblock.fp is not None,
            backend=jax.default_backend(),
            tokamak_like=tokamak_like,
            active_size=int(active_size),
            er_abs=float(er_abs),
            schur_er_min=float(schur_er_min),
            has_magdrift=(
                op.fblock.magdrift_theta is not None
                or op.fblock.magdrift_zeta is not None
                or op.fblock.magdrift_xidot is not None
            ),
            has_collisionless=op.fblock.collisionless is not None,
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l),
            xblock_tz_limit=max(1, int(xblock_tz_max)),
        ):
            rhs1_precond_kind = "xblock_tz"
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: CPU PAS tokamak "
                    "auto -> xblock_tz preconditioner",
                )
        else:
            rhs1_precond_kind = "pas_schur"
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and float(er_abs) <= float(schur_er_min)
    ):
        rhs1_precond_kind_override = _rhs1_large_fp_near_zero_er_override_kind(
            rhs1_precond_env=rhs1_precond_env,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            current_kind=rhs1_precond_kind,
            total_size=int(op.total_size),
            er_abs=float(er_abs),
            schur_er_min=float(schur_er_min),
        )
        if rhs1_precond_kind_override != rhs1_precond_kind:
            rhs1_precond_kind = rhs1_precond_kind_override
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: large FP near-zero-Er "
                    "auto override -> xmg preconditioner",
                )
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and op.fblock.pas is not None
        and rhs1_precond_kind in {None, "collision", "point"}
        and not rhs1_gpu_tokamak_pas_tight_gmres
    ):
        pas_strong_max_env = os.environ.get("SFINCS_JAX_PAS_STRONG_MAX", "").strip()
        try:
            pas_strong_max = int(pas_strong_max_env) if pas_strong_max_env else 25000
        except ValueError:
            pas_strong_max = 25000
        if int(active_size) <= max(1, int(pas_strong_max)):
            rhs1_precond_kind = "pas_hybrid"
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and int(op.n_species) >= 2
        and (geom_scheme == 1 or int(op.n_zeta) <= 9)
        and rhs1_precond_kind in {"theta_line", "zeta_line", "theta_zeta"}
    ):
        # Multi-species PAS tokamak-like runs are prone to stagnation with pure line
        # preconditioners; prefer Schur by default for robustness/parity.
        rhs1_precond_kind = "schur"
    rhs1_precond_kind_requested = rhs1_precond_kind
    if rhs1_precond_env == "" and rhs1_precond_kind == "point" and use_pas_projection:
        # PAS tokamak-like cases benefit from a stronger line preconditioner by default.
        rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
    if rhs1_precond_env == "":
        shard_axis = _matvec_shard_axis(op)
        if shard_axis in {"theta", "zeta"} and jax.device_count() > 1:
            pas_tz_estimate = _estimate_rhs1_pas_tz_build_bytes(op)
            pas_tz_max_bytes = _rhs1_pas_tz_max_bytes()
            pas_shard_xmg_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SHARD_XMG_MIN", "").strip()
            try:
                pas_shard_xmg_min = int(pas_shard_xmg_min_env) if pas_shard_xmg_min_env else 80000
            except ValueError:
                pas_shard_xmg_min = 80000
            fp_shard_xmg_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_SHARD_XMG_MIN", "").strip()
            try:
                fp_shard_xmg_min = int(fp_shard_xmg_min_env) if fp_shard_xmg_min_env else 120000
            except ValueError:
                fp_shard_xmg_min = 120000
            keep_xmg_for_large_pas_er = bool(
                rhs1_precond_kind == "xmg"
                and op.fblock.pas is not None
                and int(op.total_size) >= max(1, int(pas_shard_xmg_min))
            )
            keep_xmg_for_large_fp = bool(
                rhs1_precond_kind == "xmg"
                and op.fblock.fp is not None
                and int(op.total_size) >= max(1, int(fp_shard_xmg_min))
            )
            schwarz_auto_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "").strip()
            try:
                schwarz_auto_min = int(schwarz_auto_min_env) if schwarz_auto_min_env else 120000
            except ValueError:
                schwarz_auto_min = 120000
            force_schwarz = bool(schwarz_auto_min_env) and int(schwarz_auto_min) <= 0
            if force_schwarz:
                rhs1_precond_kind = "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
            elif _rhs1_sharded_line_override_allowed(rhs1_precond_kind):
                # Preserve dedicated PAS preconditioners on sharded runs. Demoting
                # pas_tz/pas_tokamak_theta/pas_ilu to pure line blocks can turn a
                # parity-clean moderate PAS solve into a long line-preconditioned
                # Krylov run with no robustness benefit.
                if rhs1_precond_kind in {"theta_line", "zeta_line"} and rhs1_precond_kind != f"{shard_axis}_line":
                    pass
                elif keep_xmg_for_large_pas_er or keep_xmg_for_large_fp:
                    pass
                elif (
                    op.fblock.pas is not None
                    and pas_tz_estimate > max(0, int(pas_tz_max_bytes))
                ):
                    rhs1_precond_kind = "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: sharded PAS large pas_tz "
                            f"(est={pas_tz_estimate / 2**30:.2f} GiB > cap={pas_tz_max_bytes / 2**30:.2f} GiB) -> "
                            f"{rhs1_precond_kind}",
                        )
                elif int(op.total_size) >= max(1, int(schwarz_auto_min)):
                    rhs1_precond_kind = "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                else:
                    rhs1_precond_kind = "theta_line" if shard_axis == "theta" else "zeta_line"
    if (
        rhs1_precond_env == ""
        and rhs1_precond_kind == "schur"
        and op.fblock.pas is not None
        and (not bool(op.include_phi1))
    ):
        # In sharded multi-device PAS runs, the global Schur preconditioner can dominate
        # wall time for moderate-size systems. Prefer shard-local Schwarz blocks when
        # Er is near zero, where this branch is typically more than sufficient.
        shard_axis = _matvec_shard_axis(op)
        if shard_axis in {"theta", "zeta"} and jax.device_count() > 1:
            schur_shard_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_SHARD_MAX", "").strip()
            try:
                schur_shard_max = int(schur_shard_max_env) if schur_shard_max_env else 30000
            except ValueError:
                schur_shard_max = 30000
            schur_shard_er_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_SHARD_ER_MAX", "").strip()
            try:
                schur_shard_er_max = float(schur_shard_er_env) if schur_shard_er_env else 1.0e-8
            except ValueError:
                schur_shard_er_max = 1.0e-8
            if int(op.total_size) <= max(1, int(schur_shard_max)) and float(er_abs) <= max(
                0.0, float(schur_shard_er_max)
            ):
                rhs1_precond_kind = "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: sharded PAS near-zero-Er -> "
                        f"{rhs1_precond_kind} preconditioner",
                    )
    if str(solve_method).strip().lower() in {"dense", "dense_ksp", "dense_row_scaled"}:
        rhs1_precond_kind = None
    pas_tokamak_gpu_tol = _rhs1_pas_tokamak_gpu_tight_tol(
        enabled=rhs1_gpu_tokamak_pas_tight_gmres or rhs1_precond_kind == "pas_tokamak_theta",
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        backend=jax.default_backend(),
        tokamak_like=tokamak_like,
        active_size=int(active_size),
        er_abs=float(er_abs),
        schur_er_min=float(schur_er_min),
        has_magdrift=(
            op.fblock.magdrift_theta is not None
            or op.fblock.magdrift_zeta is not None
            or op.fblock.magdrift_xidot is not None
        ),
        has_collisionless=op.fblock.collisionless is not None,
    )
    if pas_tokamak_gpu_tol is not None:
        tol_old = float(tol)
        tol = min(float(tol), float(pas_tokamak_gpu_tol))
        if emit is not None and float(tol) < tol_old:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                f"tol tightened {tol_old:.1e} -> {float(tol):.1e}",
            )
    if (
        (not rhs1_precond_env)
        and maxiter_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and rhs1_precond_kind == "xmg"
        and int(op.total_size) >= 120000
    ):
        # Keep large full-trajectory FP cases inside the regression budget by
        # capping default Krylov work when xmg is selected automatically.
        #
        # Use the same cap on single- and multi-device runs. The tighter multi-device
        # cap can under-converge large FP systems.
        fp_auto_maxiter = 800
        maxiter = min(int(maxiter if maxiter is not None else 400), int(fp_auto_maxiter))
        fp_auto_restart_max = 160
        restart = max(80, min(int(restart), int(fp_auto_restart_max)))
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: large FP auto-tune "
                f"(precond=xmg restart={int(restart)} maxiter={int(maxiter)})",
            )
    structured_fblock_precond_requested = str(rhs1_precond_kind or "").startswith("structured_fblock_")
    rhs1_precond_enabled = (
        rhs1_precond_kind is not None
        and int(op.rhs_mode) == 1
        and ((not bool(op.include_phi1)) or bool(structured_fblock_precond_requested))
    )
    _set_precond_policy_hints(
        geom_scheme=geom_scheme,
        use_dkes=bool(use_dkes),
        rhs1_precond_kind=rhs1_precond_kind,
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
        er_abs=float(er_abs),
    )
    if rhs1_bicgstab_env in {"0", "false", "no", "off"}:
        rhs1_bicgstab_kind = None
    elif rhs1_bicgstab_env in {"rhs1", "same", "preconditioner"}:
        rhs1_bicgstab_kind = "rhs1"
    elif rhs1_bicgstab_env in {"", "1", "true", "yes", "on", "collision", "diag"}:
        rhs1_bicgstab_kind = "collision"
    else:
        rhs1_bicgstab_kind = None
    if tokamak_pas and rhs1_bicgstab_env in {"", "auto"}:
        # Tokamak PAS systems converge more reliably with GMRES-only.
        rhs1_bicgstab_kind = None
    if (
        rhs1_bicgstab_kind == "collision"
        and op.fblock.fp is not None
        and rhs1_precond_kind not in {None, "collision"}
    ):
        rhs1_bicgstab_kind = "rhs1"
    if (
        rhs1_bicgstab_kind == "collision"
        and op.fblock.pas is not None
        and use_dkes
        and rhs1_precond_kind not in {None, "collision"}
    ):
        # For PAS+DKES, BiCGStab only pays off if it uses the same strong RHS1
        # preconditioner as GMRES (typically Schur + sparse PAS blocks). A cheap
        # collision-only preconditioner often stagnates and just triggers a GMRES
        # fallback (wasting compile/runtime).
        rhs1_bicgstab_kind = "rhs1"
    solve_method_kind = str(solve_method).strip().lower()
    use_implicit = _resolve_use_implicit(differentiable=differentiable)
    if (
        solve_method_kind in {"auto", "default", "incremental"}
        and (not use_implicit)
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and (not bool(op.include_phi1))
        and int(op.rhs_mode) == 1
    ):
        dense_auto_cutoff = _rhs1_dense_auto_fp_cutoff_impl(
            dense_active_cutoff=_rhsmode1_dense_fallback_max(op),
        )
        if _rhs1_dense_auto_fp_allowed_impl(
            backend=jax.default_backend(),
            active_size=int(active_size),
            dense_active_cutoff=_rhsmode1_dense_fallback_max(op),
        ):
            solve_method = "dense"
            solve_method_kind = "dense"
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: auto-selected dense "
                    f"full-FP solve (size={int(active_size)} <= cutoff={int(dense_auto_cutoff)})",
                )
    if solve_method_kind == "dense_ksp":
        # `dense_ksp` uses its own PETSc-like block preconditioner on the assembled dense system.
        rhs1_precond_enabled = False
    # Upstream SFINCS v3 reports KSP residual norms for the *preconditioned* residual, matching
    # a left-preconditioned solve. Default to left to align solver-branch parity and to avoid
    # JAX transpose-rule limitations in the right-preconditioned path.
    krylov_routing_controls = rhs1_krylov_routing_controls_from_env()
    gmres_precond_side = str(krylov_routing_controls.gmres_precondition_side)

    bicgstab_fallback_controls = rhs1_bicgstab_fallback_controls_from_env(
        pas_large_bicgstab_fastpath=bool(pas_large_bicgstab_fastpath),
    )
    bicgstab_fallback_strict = bool(bicgstab_fallback_controls.strict)
    distributed_axis = _resolve_distributed_gmres_axis(op=op, emit=emit)
    use_sharded_matvec = distributed_axis in {"theta", "zeta"} and (not use_implicit)
    distributed_auto_solver = str(krylov_routing_controls.distributed_auto_solver)
    if use_sharded_matvec:
        def mv(x):
            return apply_v3_full_system_operator(op, x, allow_sharding=True)
    else:
        def mv(x):
            # Use the JIT-compiled operator application to reduce Python overhead in repeated matvecs
            # (e.g. during GMRES iterations and Er scans).
            return apply_v3_full_system_operator_cached(op, x)

    sparse_config = rhs1_sparse_preconditioner_config_from_env(
        has_pas=op.fblock.pas is not None,
        use_dkes=bool(use_dkes),
        active_size=int(active_size),
        backend=str(jax.default_backend()),
    )
    sparse_precond_mode = sparse_config.precond_mode
    sparse_precond_kind = sparse_config.precond_kind
    sparse_allow_nondiff = sparse_config.allow_nondiff
    sparse_use_matvec = sparse_config.use_matvec
    sparse_operator_mode = sparse_config.operator_mode
    sparse_max_size = sparse_config.max_size
    sparse_drop_tol = sparse_config.drop_tol
    sparse_drop_rel = sparse_config.drop_rel
    sparse_ilu_drop_tol = sparse_config.ilu_drop_tol
    sparse_ilu_fill = sparse_config.ilu_fill
    sparse_ilu_dense_max = sparse_config.ilu_dense_max
    sparse_dense_cache_max = sparse_config.dense_cache_max
    sparse_jax_config = rhs1_sparse_jax_config_from_env()
    sparse_exact_lu = _rhsmode1_sparse_exact_lu_requested(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        full_precond_requested=bool(full_precond_requested),
        preconditioner_x=int(preconditioner_x),
        use_dkes=bool(use_dkes),
    )
    sparse_exact_direct = _rhsmode1_host_sparse_direct_allowed(
        sparse_exact_lu=sparse_exact_lu,
        use_implicit=bool(use_implicit),
    )
    sparse_prefer_over_dense_shortcut = _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=op,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        use_implicit=bool(use_implicit),
    )
    sparse_prefer_skips_stage2 = _rhsmode1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=bool(sparse_prefer_over_dense_shortcut),
        sparse_precond_mode=sparse_precond_mode,
    )
    gpu_dkes_sparse_shortcut = bool(
        rhs1_precond_env_user in {"", "auto"}
        and rhs1_bicgstab_env_user in {"", "auto"}
        and solve_method_kind not in {"dense", "dense_ksp"}
        and jax.default_backend() != "cpu"
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and use_dkes
        and sparse_precond_mode != "off"
        and int(active_size) <= int(sparse_max_size)
    )
    cs0_sparse_first = _rhsmode1_constraint0_sparse_first(
        op=op,
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
    )
    cs0_petsc_compat = _rhsmode1_constraint0_petsc_compat(
        op=op,
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
    )
    cs0_dense_fallback_allowed = _rhsmode1_constraint0_dense_fallback_allowed(op)
    if cs0_petsc_compat:
        rhs1_precond_kind = None
        rhs1_precond_enabled = False
        rhs1_bicgstab_kind = None
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat auto mode "
                "-> dedicated sparse ILU path",
            )
    if gpu_dkes_sparse_shortcut:
        rhs1_precond_kind = None
        rhs1_precond_enabled = False
        rhs1_bicgstab_kind = None
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: GPU DKES auto mode -> sparse ILU shortcut "
                f"(size={int(active_size)})",
            )

    profile_linear_context = ProfileLinearSolveContext(
        rhs_mode=int(op.rhs_mode),
        total_size=int(op.total_size),
        use_implicit=bool(use_implicit),
        use_solver_jit=bool(_use_solver_jit()),
        distributed_axis=distributed_axis,
        distributed_auto_solver=distributed_auto_solver,
        small_gmres_max=rhs1_small_gmres_max_from_env(),
    )

    def _solver_kind(method: str) -> tuple[str, str]:
        return profile_solver_kind(method, context=profile_linear_context)

    stage2_admission = rhs1_stage2_admission_controls_from_env(
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        solver_kind_default=_solver_kind(solve_method)[0],
        pas_large_bicgstab_fastpath=bool(pas_large_bicgstab_fastpath),
        tokamak_pas=bool(tokamak_pas),
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        total_size=int(op.total_size),
    )
    stage2_enabled = bool(stage2_admission.enabled)
    stage2_time_cap_s = float(stage2_admission.time_cap_s)

    def _solve_linear(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        precond_side: str,
    ):
        return solve_profile_linear(
            context=profile_linear_context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            precond_side=precond_side,
        )

    def _solve_linear_with_residual(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        precond_side: str,
    ) -> tuple[GMRESSolveResult, jnp.ndarray]:
        return solve_profile_linear_with_residual(
            context=profile_linear_context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            precond_side=precond_side,
        )

    ksp_diagnostics_controls = rhs1_ksp_diagnostics_controls_from_env(emit=emit)

    ksp_replay = RHS1KSPReplayState(
        restart=restart,
        maxiter=maxiter,
        precond_side=gmres_precond_side,
        solver_kind="gmres",
    )
    residual_vec: jnp.ndarray | None = None
    rhs1_ksp_diagnostics_context = RHS1KSPDiagnosticsContext(
        emit=emit,
        fortran_stdout=bool(ksp_diagnostics_controls.fortran_stdout),
        history_max_size=ksp_diagnostics_controls.history_max_size,
        history_max_iter=ksp_diagnostics_controls.history_max_iter,
        iter_stats_enabled=bool(ksp_diagnostics_controls.iter_stats_enabled),
        iter_stats_max_size=ksp_diagnostics_controls.iter_stats_max_size,
    )

    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return reduce_full_with_indices(v_full, active_idx_jnp)

        def _wrap_pas_precond(precond_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
            return precond_fn

        if use_pas_projection:
            fs_factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)
            fs_sum = jnp.sum(fs_factor)
            fs_sum_safe = jnp.where(fs_sum != 0, fs_sum, jnp.asarray(1.0, dtype=jnp.float64))
            ix0 = _ix_min(bool(op.point_at_x0))
            mask_x = (jnp.arange(int(op.n_x)) >= ix0).astype(jnp.float64)

            def _project_pas_f(f_flat: jnp.ndarray) -> jnp.ndarray:
                return project_pas_constraint_f(
                    f_flat,
                    f_shape=op.fblock.f_shape,
                    fs_factor=fs_factor,
                    fs_sum_safe=fs_sum_safe,
                    mask_x=mask_x,
                )

            def _expand_active_f(v_reduced: jnp.ndarray) -> jnp.ndarray:
                return expand_reduced_with_map(v_reduced, full_to_active_jnp)

            def _project_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(v_reduced)
                f_proj = _project_pas_f(f_full)
                return reduce_full(f_proj)

            def _wrap_pas_precond(precond_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
                def _apply(v_reduced: jnp.ndarray) -> jnp.ndarray:
                    z_reduced = precond_fn(v_reduced)
                    return _project_reduced(z_reduced)
                return _apply

            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(v_reduced)
                if int(op.extra_size) > 0:
                    zeros_e = jnp.zeros((int(op.extra_size),), dtype=v_reduced.dtype)
                    return jnp.concatenate([f_full, zeros_e], axis=0)
                return f_full

            zeros_extra = jnp.zeros((int(op.extra_size),), dtype=jnp.float64)

            def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(x_reduced)
                f_proj = _project_pas_f(f_full)
                x_full = jnp.concatenate([f_proj, zeros_extra], axis=0) if int(op.extra_size) > 0 else f_proj
                y_full = mv(x_full)
                y_f = y_full[: op.f_size]
                y_proj = _project_pas_f(y_f)
                return reduce_full(y_proj)

            rhs_f = rhs[: op.f_size]
            rhs_proj = _project_pas_f(rhs_f)
            rhs_reduced = reduce_full(rhs_proj)
            x0_reduced = None
            if x0 is not None:
                x0_arr = jnp.asarray(x0)
                if x0_arr.shape == (active_size,):
                    x0_reduced = _project_reduced(x0_arr)
                elif x0_arr.shape == (op.total_size,):
                    f0_proj = _project_pas_f(x0_arr[: op.f_size])
                    x0_reduced = reduce_full(f0_proj)
                elif x0_arr.shape == (op.f_size,):
                    f0_proj = _project_pas_f(x0_arr)
                    x0_reduced = reduce_full(f0_proj)
            if recycle_basis_use:
                basis_reduced: list[jnp.ndarray] = []
                for vec in recycle_basis_use:
                    if vec.shape != (op.total_size,):
                        continue
                    f_proj = _project_pas_f(vec[: op.f_size])
                    basis_reduced.append(reduce_full(f_proj))
                if basis_reduced:
                    basis_au = [mv_reduced(b) for b in basis_reduced]
                    x0_recycled = _recycled_initial_guess(rhs_reduced, basis_reduced, basis_au)
                    if x0_recycled is not None:
                        if x0_reduced is None:
                            x0_reduced = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv_reduced(x0_reduced) - rhs_reduced)
                            r1 = jnp.linalg.norm(mv_reduced(x0_recycled) - rhs_reduced)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0_reduced = x0_recycled
        else:
            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                return expand_reduced_with_map(v_reduced, full_to_active_jnp)

            def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                return reduce_full(mv(expand_reduced(x_reduced)))

            rhs_reduced = reduce_full(rhs)
            x0_reduced = None
            if x0 is not None:
                x0_arr = jnp.asarray(x0)
                if x0_arr.shape == (active_size,):
                    x0_reduced = x0_arr
                elif x0_arr.shape == (op.total_size,):
                    x0_reduced = reduce_full(x0_arr)
                elif use_pas_projection and x0_arr.shape == (op.f_size,):
                    x0_reduced = reduce_full(x0_arr)
            if recycle_basis_use:
                basis_reduced = []
                for vec in recycle_basis_use:
                    if vec.shape != (op.total_size,):
                        continue
                    basis_reduced.append(reduce_full(vec))
                if basis_reduced:
                    basis_au = [mv_reduced(b) for b in basis_reduced]
                    x0_recycled = _recycled_initial_guess(rhs_reduced, basis_reduced, basis_au)
                    if x0_recycled is not None:
                        if x0_reduced is None:
                            x0_reduced = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv_reduced(x0_reduced) - rhs_reduced)
                            r1 = jnp.linalg.norm(mv_reduced(x0_recycled) - rhs_reduced)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0_reduced = x0_recycled
        target_reduced = max(float(atol), float(tol) * float(jnp.linalg.norm(rhs_reduced)))
        target_stage2 = float(target_reduced)
        if op.fblock.fp is not None and op.fblock.pas is None and (not bool(op.include_phi1)):
            # FP RHS can have large norms; enforce a stricter absolute target for
            # stage2/strong-preconditioner decisions to avoid premature convergence
            # with loose relative norms.
            target_stage2 = min(float(target_stage2), max(float(atol), float(tol)))
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        dense_backend_allowed = _rhsmode1_dense_backend_allowed()
        host_dense_fallback_allowed = _rhsmode1_host_dense_fallback_allowed()
        dense_krylov_allowed = _rhsmode1_dense_krylov_allowed()
        dense_shortcut_setup = rhs1_dense_shortcut_setup_from_env(
            has_pas=op.fblock.pas is not None,
            include_phi1=bool(op.include_phi1),
            constraint_scheme=int(op.constraint_scheme),
            active_size=int(active_size),
            dense_fallback_max=int(dense_fallback_max),
            dense_backend_allowed=bool(dense_backend_allowed),
            host_dense_fallback_allowed=bool(host_dense_fallback_allowed),
            dense_krylov_allowed=bool(dense_krylov_allowed),
            backend=str(jax.default_backend()),
        )
        dense_shortcut_ratio = float(dense_shortcut_setup.dense_shortcut_ratio)
        dense_fallback_max = int(dense_shortcut_setup.dense_fallback_max)
        disable_dense_pas = bool(dense_shortcut_setup.disable_dense_pas)
        if emit is not None:
            for _level, _message in dense_shortcut_setup.messages:
                emit(_level, _message)
        rhs1_precond_kind = rhs1_fp_preconditioner_probe_kind_from_env(
            rhs1_precond_kind=rhs1_precond_kind,
            rhs1_precond_env=str(rhs1_precond_env),
            has_fp=op.fblock.fp is not None,
            use_dkes=bool(use_dkes),
            include_phi1=bool(op.include_phi1),
            dense_fallback_max=int(dense_fallback_max),
            active_size=int(active_size),
            rhs1_precond_enabled=bool(rhs1_precond_enabled),
            solve_method_kind=solve_method_kind,
        )
        early_dense_shortcut = False
        probe_shortcut = False
        probe_x0: jnp.ndarray | None = None
        preconditioner_reduced = None
        strong_preconditioner_reduced = None
        bicgstab_preconditioner_reduced = None
        pas_precond_force_collision = False
        rhs1_pas_tz_guarded_fallback = False
        rhs1_pas_tz_guarded_axis: str | None = None
        pas_tz_guarded_stream_requested = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STREAM_UPDATE",
            default=False,
        )
        dense_matrix_cache: np.ndarray | None = None
        host_dense_shortcut = _rhsmode1_host_dense_shortcut_allowed(
            op=op,
            active_size=int(active_size),
            use_implicit=bool(use_implicit),
            solve_method_kind=str(solve_method_kind),
        )
        cpu_large_sparse_shortcut = _rhsmode1_large_cpu_sparse_skip_primary_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
            use_implicit=bool(use_implicit),
        )
        if cpu_large_sparse_shortcut:
            # This path intentionally bypasses the setup-time RHSMode=1
            # preconditioner and primary Krylov attempt. For the measured
            # mid-size QI full-FP rung, those stages only gate the exact active
            # sparse-LU rescue that actually writes the converged solution.
            rhs1_precond_enabled = False
            rhs1_precond_kind = None
            rhs1_bicgstab_kind = None
            if emit is not None:
                backend_name = str(jax.default_backend()).strip().lower()
                sparse_label = "CPU" if backend_name == "cpu" else f"{backend_name} host-sparse"
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: "
                    f"{sparse_label} sparse-LU shortcut -> skip primary preconditioner build",
                )

        def _solve_host_dense_reduced(*, x0_dense: jnp.ndarray | None = None) -> GMRESSolveResult:
            return solve_host_dense_reduced(
                context=HostDenseReducedSolveContext(
                    matvec=mv_reduced,
                    rhs=rhs_reduced,
                    active_size=int(active_size),
                    constraint_scheme=int(op.constraint_scheme),
                    has_fp=op.fblock.fp is not None,
                    dense_matrix_cache=dense_matrix_cache,
                ),
                x0=x0_dense,
            )
        if rhs1_bicgstab_kind is not None:
            if emit is not None:
                emit(1, f"solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner={rhs1_bicgstab_kind}")
            if rhs1_bicgstab_kind == "collision":
                bicgstab_preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if use_pas_projection:
                bicgstab_preconditioner_reduced = _wrap_pas_precond(bicgstab_preconditioner_reduced)

        # PAS probe shortcut: avoid expensive block/line preconditioner builds when a
        # cheap collision-based preconditioner already provides a strong residual drop.
        pas_probe_config = _rhs1_pas_preconditioner_probe_config_from_env()
        rhs1_precond_kind = _rhs1_pas_default_preconditioner_kind(
            requested_env=rhs1_precond_env,
            current_kind=rhs1_precond_kind,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            n_species=int(op.n_species),
            n_zeta=int(op.n_zeta),
            geom_scheme=int(geom_scheme),
        )
        if _rhs1_pas_preconditioner_probe_admitted(
            config=pas_probe_config,
            preconditioner_kind=rhs1_precond_kind,
            preconditioner_enabled=bool(rhs1_precond_enabled),
            solve_method_kind=solve_method_kind,
            has_pas=op.fblock.pas is not None,
            use_dkes=bool(use_dkes),
        ):
            probe_key = _rhsmode1_precond_cache_key(op, "pas_probe_decision")
            use_collision_precond = _RHSMODE1_PAS_PRECOND_PROBE_CACHE.get(probe_key)
            use_collision_precond, skip_message = _rhs1_pas_preconditioner_probe_large_collision_skip(
                config=pas_probe_config,
                cached_decision=use_collision_precond,
                total_size=int(op.total_size),
                constraint_scheme=int(op.constraint_scheme),
                extra_size=int(op.extra_size),
            )
            if skip_message is not None:
                _RHSMODE1_PAS_PRECOND_PROBE_CACHE[probe_key] = True
                if emit is not None:
                    emit(1, skip_message)
            if use_collision_precond is None:
                try:
                    probe_precond = _build_rhsmode1_collision_preconditioner(
                        op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                    if use_pas_projection:
                        probe_precond = _wrap_pas_precond(probe_precond)
                    probe_x = probe_precond(rhs_reduced)
                    probe_r = rhs_reduced - mv_reduced(probe_x)
                    rhs_norm = float(jnp.linalg.norm(rhs_reduced))
                    probe_rel = float(jnp.linalg.norm(probe_r)) / rhs_norm if rhs_norm > 0 else 0.0
                    use_collision_precond = _rhs1_pas_preconditioner_probe_uses_collision(
                        probe_rel=probe_rel,
                        rel_max=pas_probe_config.rel_max,
                    )
                    _RHSMODE1_PAS_PRECOND_PROBE_CACHE[probe_key] = bool(use_collision_precond)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS precond probe "
                            f"(rel={probe_rel:.3e}, max={pas_probe_config.rel_max:.3e}) -> "
                            f"{'collision' if use_collision_precond else 'full'}",
                        )
                except Exception as exc:  # noqa: BLE001
                    use_collision_precond = None
                    if emit is not None:
                        emit(1, f"solve_v3_full_system_linear_gmres: PAS precond probe failed ({type(exc).__name__}: {exc})")
            if use_collision_precond:
                preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                if use_pas_projection:
                    preconditioner_reduced = _wrap_pas_precond(preconditioner_reduced)
                rhs1_precond_kind = "collision"
                pas_precond_force_collision = True
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced

        rhs1_reduced_preconditioner_build_context = RHS1ReducedPreconditionerBuildContext(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            mv_reduced=mv_reduced,
            emit=emit,
            mark=_mark,
            progress_preconditioner_build=rhs1_progress_notes.preconditioner_build,
            record_structured_metadata=_record_structured_fblock_preconditioner_metadata,
            wrap_pas_preconditioner=_wrap_pas_precond,
            dd_setup=rhs1_dd_setup,
            use_pas_projection=bool(use_pas_projection),
            preconditioner_species=int(preconditioner_species),
            preconditioner_x=int(preconditioner_x),
            preconditioner_xi=int(preconditioner_xi),
            build_from_kind=_build_rhs1_preconditioner_from_kind,
            build_collision=_build_rhsmode1_collision_preconditioner,
            build_xmg=_build_rhsmode1_xmg_preconditioner,
            compose_residual_correction=_compose_residual_correction_preconditioner,
            compose_multilevel_residual_correction=_compose_multilevel_residual_correction_preconditioner,
            compose_multilevel_minres_correction=_compose_multilevel_minres_correction_preconditioner,
            parse_guarded_structured_levels=_rhs1_pas_tz_guarded_structured_levels,
            resource_exhausted_error=_is_resource_exhausted_error,
        )

        def _build_rhs1_preconditioner_reduced_with_fallback():
            nonlocal rhs1_precond_kind, pas_precond_force_collision
            nonlocal bicgstab_preconditioner_reduced, rhs1_pas_tz_guarded_axis
            nonlocal rhs1_pas_tz_guarded_fallback
            precond_build = build_rhs1_reduced_preconditioner_with_fallback(
                context=rhs1_reduced_preconditioner_build_context,
                rhs1_precond_kind=rhs1_precond_kind,
                rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
                rhs1_bicgstab_kind=rhs1_bicgstab_kind,
            )
            rhs1_precond_kind = precond_build.rhs1_precond_kind
            rhs1_pas_tz_guarded_fallback = bool(precond_build.pas_tz_guarded_fallback)
            rhs1_pas_tz_guarded_axis = precond_build.pas_tz_guarded_axis
            pas_precond_force_collision = bool(precond_build.pas_precond_force_collision)
            if precond_build.bicgstab_preconditioner is not None:
                bicgstab_preconditioner_reduced = precond_build.bicgstab_preconditioner
            return precond_build.preconditioner

        if rhs1_precond_enabled and (not host_dense_shortcut):
            solver_kind = _solver_kind(solve_method)[0]
            build_rhs1 = (
                (solver_kind != "bicgstab" and solve_method_kind != "dense")
                or (rhs1_bicgstab_kind == "rhs1" and solve_method_kind != "dense")
            )
            if build_rhs1 and preconditioner_reduced is None:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced_with_fallback()
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced
        if (not host_dense_shortcut) and preconditioner_reduced is None and bicgstab_preconditioner_reduced is not None:
            preconditioner_reduced = bicgstab_preconditioner_reduced
        if (not host_dense_shortcut) and preconditioner_reduced is not None and rhs1_precond_kind in {
            "pas_hybrid",
            "pas_lite",
            "pas_tz",
            "pas_schur",
            "pas_tokamak_theta",
            "pas_ilu",
        }:
            try:
                probe = preconditioner_reduced(rhs_reduced)
                probe_ok = bool(jnp.all(jnp.isfinite(probe)))
            except Exception as exc:  # noqa: BLE001
                probe_ok = False
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS precond probe failed "
                        f"({type(exc).__name__}: {exc}), using collision preconditioner",
                    )
            if not probe_ok:
                preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                if use_pas_projection:
                    preconditioner_reduced = _wrap_pas_precond(preconditioner_reduced)
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS precond non-finite -> collision",
                    )
        # FP-only large systems: optionally augment the base preconditioner with a
        # low-L block correction to improve flow/Mach convergence without dense fallback.
        if (
            not host_dense_shortcut
            and
            preconditioner_reduced is not None
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
        ):
            fp_l1_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_L1_HYBRID", "").strip().lower()
            fp_l1_enabled = fp_l1_env in {"1", "true", "yes", "on"}
            fp_l1_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_MIN", "").strip()
            fp_l1_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_LMAX", "").strip()
            fp_l1_block_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_BLOCK_MAX", "").strip()
            try:
                fp_l1_min = int(fp_l1_min_env) if fp_l1_min_env else 80000
            except ValueError:
                fp_l1_min = 80000
            try:
                fp_l1_lmax = int(fp_l1_lmax_env) if fp_l1_lmax_env else 1
            except ValueError:
                fp_l1_lmax = 1
            try:
                fp_l1_block_max = int(fp_l1_block_env) if fp_l1_block_env else 1500
            except ValueError:
                fp_l1_block_max = 1500
            fp_l1_lmax = max(1, min(int(fp_l1_lmax), int(op.n_xi)))
            if fp_l1_enabled and int(active_size) >= max(1, int(fp_l1_min)):
                n_theta = int(op.n_theta)
                n_zeta = int(op.n_zeta)
                block_size = int(fp_l1_lmax * n_theta * n_zeta)
                if block_size > 0 and block_size <= int(fp_l1_block_max):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP L1 hybrid preconditioner "
                            f"(lmax={fp_l1_lmax} block={block_size})",
                        )
                    try:
                        l1_precond = _build_rhsmode1_xblock_tz_lmax_preconditioner(
                            op=op,
                            lmax=int(fp_l1_lmax),
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                        base_precond = preconditioner_reduced

                        def _fp_l1_hybrid(v: jnp.ndarray) -> jnp.ndarray:
                            z0 = base_precond(v)
                            r1 = v - mv_reduced(z0)
                            z1 = l1_precond(r1)
                            return z0 + z1

                        preconditioner_reduced = _fp_l1_hybrid
                        if rhs1_bicgstab_kind == "rhs1":
                            bicgstab_preconditioner_reduced = preconditioner_reduced
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: FP L1 hybrid precond failed "
                                f"({type(exc).__name__}: {exc})",
                            )
        if host_dense_shortcut:
            _mark("rhs1_host_dense_shortcut_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: accelerator FP small system -> "
                    f"using host dense shortcut (size={int(active_size)})",
                )
            res_reduced = _solve_host_dense_reduced(x0_dense=x0_reduced)
            _mark("rhs1_host_dense_shortcut_done")
            early_dense_shortcut = True
            probe_shortcut = True
            ksp_replay.matvec_fn = mv_reduced
            ksp_replay.b_vec = rhs_reduced
            ksp_replay.precond_fn = None
            ksp_replay.x0_vec = x0_reduced
            ksp_replay.precond_side = "none"
            ksp_replay.solver_kind = _solver_kind("incremental")[0]
        sparse_operator_admission = rhs1_sparse_operator_admission(
            operator_mode=sparse_operator_mode,
            use_matvec=bool(sparse_use_matvec),
            has_fp=op.fblock.fp is not None,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            use_implicit=bool(use_implicit),
            allow_nondiff=bool(sparse_allow_nondiff),
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
        )
        sparse_operator_use = bool(sparse_operator_admission.use_sparse_operator)
        if emit is not None:
            for _level, _message in sparse_operator_admission.messages:
                emit(_level, _message)
        if sparse_operator_use:
            try:
                cache_key = _rhsmode1_sparse_cache_key(
                    op,
                    kind="sparse_operator",
                    active_size=int(active_size),
                    use_active_dof_mode=True,
                    use_pas_projection=use_pas_projection,
                    drop_tol=sparse_drop_tol,
                    drop_rel=sparse_drop_rel,
                    ilu_drop_tol=sparse_ilu_drop_tol,
                    fill_factor=sparse_ilu_fill,
                )
                a_csr_full, _a_csr_drop, _ilu, _a_dense, _l_dense, _u_dense, _l_unit = _build_sparse_ilu_from_matvec(
                    matvec=mv_reduced,
                    n=int(active_size),
                    dtype=rhs_reduced.dtype,
                    cache_key=cache_key,
                    drop_tol=sparse_drop_tol,
                    drop_rel=sparse_drop_rel,
                    ilu_drop_tol=sparse_ilu_drop_tol,
                    fill_factor=sparse_ilu_fill,
                    build_dense_factors=False,
                    build_jax_factors=False,
                    build_ilu=False,
                    store_dense=False,
                    emit=emit,
                )

                def _mv_sparse(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    y_np = a_csr_full @ x_np
                    return jnp.asarray(y_np, dtype=jnp.float64)

                mv_reduced = _mv_sparse
                if emit is not None:
                    emit(0, "solve_v3_full_system_linear_gmres: using sparse operator matvec")
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"sparse_operator: failed ({type(exc).__name__}: {exc})")
        if cs0_petsc_compat and solve_method_kind not in {"dense", "dense_ksp"}:
            try:
                import scipy.sparse as sp  # noqa: PLC0415
                from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415
                from scipy.sparse.linalg import spilu  # noqa: PLC0415

                compat_config = rhs1_constraint0_petsc_compat_config_from_env(
                    restart=int(restart),
                    maxiter=maxiter,
                )
                compat_drop_tol = compat_config.drop_tol
                compat_fill = compat_config.fill
                compat_diag_pivot = compat_config.diag_pivot
                compat_restart = compat_config.restart
                compat_maxiter = compat_config.maxiter

                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat sparse ILU solve "
                        f"(size={int(active_size)} drop_tol={compat_drop_tol:.1e} fill={compat_fill:.1f})",
                    )

                a_dense_cs0 = assemble_dense_matrix_from_matvec(
                    matvec=mv_reduced,
                    n=int(active_size),
                    dtype=rhs_reduced.dtype,
                )
                a_np_cs0 = np.asarray(a_dense_cs0, dtype=np.float64)
                max_abs_cs0 = float(np.max(np.abs(a_np_cs0))) if a_np_cs0.size else 0.0
                compat_drop_thresh = max(float(sparse_drop_tol), float(sparse_drop_rel) * max_abs_cs0)
                if compat_drop_thresh > 0.0:
                    a_np_cs0 = a_np_cs0.copy()
                    a_np_cs0[np.abs(a_np_cs0) < compat_drop_thresh] = 0.0
                a_csr_cs0 = sp.csr_matrix(a_np_cs0)
                a_csr_cs0.eliminate_zeros()
                max_abs_cs0 = float(np.max(np.abs(a_csr_cs0.data))) if int(a_csr_cs0.nnz) > 0 else 0.0
                compat_reg = rhs1_constraint0_petsc_compat_regularization(max_abs=max_abs_cs0)
                perm = np.asarray(
                    reverse_cuthill_mckee(a_csr_cs0, symmetric_mode=False),
                    dtype=np.int32,
                )
                inv_perm = np.argsort(perm).astype(np.int32, copy=False)
                a_perm = a_csr_cs0[perm][:, perm].tocsc()
                if compat_reg != 0.0:
                    diag_idx = np.arange(int(active_size), dtype=np.int32)
                    a_perm = a_perm.copy()
                    a_perm[diag_idx, diag_idx] = a_perm[diag_idx, diag_idx] + compat_reg
                ilu_cs0 = spilu(
                    a_perm,
                    drop_tol=float(compat_drop_tol),
                    fill_factor=float(compat_fill),
                    permc_spec="NATURAL",
                    diag_pivot_thresh=float(compat_diag_pivot),
                )
                rhs_perm = jnp.asarray(np.asarray(rhs_reduced, dtype=np.float64)[perm], dtype=jnp.float64)
                x0_perm = None

                def _mv_perm(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    return jnp.asarray(a_perm @ x_np, dtype=jnp.float64)

                def _precond_perm(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    return jnp.asarray(ilu_cs0.solve(x_np), dtype=jnp.float64)

                rhs_pc_perm_np = np.asarray(_precond_perm(rhs_perm), dtype=np.float64)
                rhs_pc_norm = float(np.linalg.norm(rhs_pc_perm_np))
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat rhs_pc "
                        f"norm={rhs_pc_norm:.3e} finite={bool(np.all(np.isfinite(rhs_pc_perm_np)))} "
                        f"drop={compat_drop_thresh:.3e} reg={compat_reg:.3e} nnz={int(a_csr_cs0.nnz)}",
                    )
                rhs_perm_norm = float(np.linalg.norm(np.asarray(rhs_perm, dtype=np.float64)))
                rhs_pc_zero_tol = max(float(atol), max(1.0, rhs_perm_norm) * float(tol))
                if np.isfinite(rhs_pc_norm) and rhs_pc_norm <= rhs_pc_zero_tol:
                    x_perm_np = np.zeros((int(active_size),), dtype=np.float64)
                    rn_true_cs0 = rhs_perm_norm
                    rn_pc_cs0 = rhs_pc_norm
                else:
                    x_perm_np, rn_true_cs0, rn_pc_cs0, _history = explicit_left_preconditioned_gmres_scipy(
                        matvec=_mv_perm,
                        b=rhs_perm,
                        preconditioner=_precond_perm,
                        x0=x0_perm,
                        tol=tol,
                        atol=atol,
                        restart=min(int(active_size), max(1, int(compat_restart))),
                        maxiter=max(1, int(compat_maxiter)),
                    )
                x_cs0_np = np.asarray(x_perm_np, dtype=np.float64)[inv_perm]
                rhs_pc_np = rhs_pc_perm_np[inv_perm]

                def _mv_cs0_pc_full(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    y_perm = np.asarray(a_perm @ x_np[perm], dtype=np.float64)
                    z_perm = ilu_cs0.solve(y_perm)
                    return jnp.asarray(z_perm[inv_perm], dtype=jnp.float64)

                res_reduced = GMRESSolveResult(
                    x=jnp.asarray(x_cs0_np, dtype=jnp.float64),
                    residual_norm=jnp.asarray(rn_pc_cs0, dtype=jnp.float64),
                )
                ksp_replay.matvec_fn = _mv_cs0_pc_full
                ksp_replay.b_vec = jnp.asarray(rhs_pc_np, dtype=jnp.float64)
                ksp_replay.precond_fn = None
                ksp_replay.x0_vec = None if x0_reduced is None else jnp.asarray(x0_reduced, dtype=jnp.float64)
                ksp_replay.precond_side = "none"
                ksp_replay.solver_kind = _solver_kind("incremental")[0]
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat residuals "
                        f"preconditioned={rn_pc_cs0:.3e} true={rn_true_cs0:.3e}",
                    )
            except Exception as exc:  # noqa: BLE001
                cs0_petsc_compat = False
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat solve failed "
                        f"({type(exc).__name__}: {exc})",
                    )
        dense_probe_admission = rhs1_dense_probe_admission(
            probe_enabled=rhs1_dense_probe_enabled_from_env(),
            probe_shortcut=bool(probe_shortcut),
            cs0_petsc_compat=bool(cs0_petsc_compat),
            cs0_sparse_first=bool(cs0_sparse_first),
            cs0_dense_fallback_allowed=bool(cs0_dense_fallback_allowed),
            constraint_scheme=int(op.constraint_scheme),
            has_preconditioner=preconditioner_reduced is not None,
            solve_method_kind=solve_method_kind,
        )
        if dense_probe_admission.enabled:
            try:
                probe_x0 = preconditioner_reduced(rhs_reduced)
                probe_r = rhs_reduced - mv_reduced(probe_x0)
                probe_norm = float(jnp.linalg.norm(probe_r))
                probe_ratio = probe_norm / max(float(target_reduced), 1e-300)
                dense_probe_decision = rhs1_dense_probe_shortcut_decision(
                    dense_shortcut_ratio=float(dense_shortcut_ratio),
                    probe_ratio=float(probe_ratio),
                    dense_fallback_max=int(dense_fallback_max),
                    active_size=int(active_size),
                    sparse_prefer_over_dense_shortcut=bool(sparse_prefer_over_dense_shortcut),
                )
                if dense_probe_decision.accept_shortcut:
                    early_dense_shortcut = True
                    probe_shortcut = True
                    res_reduced = GMRESSolveResult(x=probe_x0, residual_norm=jnp.asarray(probe_norm))
                    ksp_replay.matvec_fn = mv_reduced
                    ksp_replay.b_vec = rhs_reduced
                    ksp_replay.precond_fn = preconditioner_reduced
                    ksp_replay.x0_vec = probe_x0
                    ksp_replay.precond_side = gmres_precond_side
                    ksp_replay.solver_kind = _solver_kind(solve_method)[0]
                elif dense_probe_decision.seed_x0_if_missing and x0_reduced is None:
                    x0_reduced = probe_x0
                if emit is not None:
                    for _level, _message in dense_probe_decision.messages:
                        emit(_level, _message)
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: probe failed ({type(exc).__name__}: {exc})")
        cpu_large_xblock_shortcut = False
        if cs0_petsc_compat:
            pass
        elif probe_shortcut:
            pass
        elif solve_method_kind == "dense_ksp":
            if int(op.phi1_size) != 0:
                raise NotImplementedError("dense_ksp is only supported for includePhi1=false RHSMode=1 solves.")
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: assembling dense reduced matrix for dense_ksp")
            a_dense = assemble_dense_matrix_from_matvec(matvec=mv_reduced, n=active_size, dtype=rhs_reduced.dtype)

            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building PETSc-like species-block preconditioner (dense_ksp)")

            import jax.scipy.linalg as jla  # noqa: PLC0415

            n_species = int(op.n_species)
            n_theta = int(op.n_theta)
            n_zeta = int(op.n_zeta)
            local_per_species = int(np.sum(nxi_for_x))
            dke_size = int(local_per_species * n_theta * n_zeta)
            extra_size = int(op.extra_size)
            extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
            if extra_size and (extra_per_species * n_species != extra_size):
                extra_per_species = 0

            f_size = int(n_species * dke_size)
            expected_active = int(f_size + int(op.phi1_size) + extra_size)
            if int(active_size) != expected_active:
                raise RuntimeError(f"dense_ksp expects active_size={expected_active}, got {active_size}")

            lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
            idx_blocks: list[jnp.ndarray] = []
            for s in range(n_species):
                f_idx = np.arange(s * dke_size, (s + 1) * dke_size, dtype=np.int32)
                extra_idx = np.arange(f_size + s * extra_per_species, f_size + (s + 1) * extra_per_species, dtype=np.int32)
                block_idx_np = np.concatenate([f_idx, extra_idx], axis=0) if extra_per_species else f_idx
                block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
                a_block = a_dense[jnp.ix_(block_idx, block_idx)]
                lu, piv = jla.lu_factor(a_block)
                lu_factors.append((lu, piv))
                idx_blocks.append(block_idx)

            def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
                out = jnp.zeros_like(v)
                for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
                    rhs_block = v[block_idx]
                    sol_block = jla.lu_solve((lu, piv), rhs_block)
                    out = out.at[block_idx].set(sol_block, unique_indices=True)
                return out

            def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                return a_dense @ x

            # PETSc v3 uses *left* preconditioning and checks convergence in the
            # preconditioned residual norm ||M^{-1} r||. To match this behavior with
            # JAX's GMRES (which uses a SciPy-style convergence check), solve the
            # explicitly left-preconditioned system:
            #   (M^{-1} A) x = (M^{-1} b).
            rhs_pc = preconditioner_dense(rhs_reduced)

            def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
                return preconditioner_dense(mv_dense(x))

            rhs1_progress_notes.krylov_start()
            _mark("rhs1_krylov_solve_start")
            res_reduced = _solve_linear(
                matvec_fn=mv_pc,
                b_vec=rhs_pc,
                precond_fn=None,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side="none",
            )
            res_reduced = _block_gmres_result_ready(res_reduced)
            _mark("rhs1_krylov_solve_done")
            ksp_replay.matvec_fn = mv_pc
            ksp_replay.b_vec = rhs_pc
            ksp_replay.precond_fn = None
            ksp_replay.x0_vec = x0_reduced
            ksp_replay.precond_side = "none"
            ksp_replay.solver_kind = _solver_kind("incremental")[0]
        else:
            # If the probe indicates the system is far from converged but the dense
            # fallback is not allowed (e.g. medium FP DKES systems), avoid spending
            # many matvecs on a weak/expensive preconditioner. Instead, jump directly
            # to the sparse ILU branch.
            cpu_large_xblock_shortcut = _rhsmode1_large_cpu_xblock_skip_primary_allowed(
                op=op,
                solve_method_kind=solve_method_kind,
                active_size=int(active_size),
                sparse_max_size=int(sparse_max_size),
                preconditioner_species=preconditioner_species,
                preconditioner_x=preconditioner_x,
                preconditioner_xi=preconditioner_xi,
                pre_theta=pre_theta,
                pre_zeta=pre_zeta,
                use_implicit=bool(use_implicit),
                rhs1_precond_env=str(rhs1_precond_env),
            )
            skip_primary_krylov = (
                (
                    (op.fblock.fp is not None)
                    and bool(early_dense_shortcut)
                    and (not bool(probe_shortcut))
                    and sparse_precond_mode != "off"
                    and int(active_size) <= int(sparse_max_size)
                    and solve_method_kind not in {"dense", "dense_ksp"}
                )
                or gpu_dkes_sparse_shortcut
                or cpu_large_xblock_shortcut
                or cpu_large_sparse_shortcut
            )
            if skip_primary_krylov:
                if emit is not None:
                    reason = "probe ratio huge, dense disabled"
                    if gpu_dkes_sparse_shortcut:
                        reason = "GPU DKES auto sparse shortcut"
                    elif cpu_large_xblock_shortcut:
                        backend_name = str(jax.default_backend()).strip().lower()
                        reason = (
                            "CPU large FP x-block shortcut"
                            if backend_name == "cpu"
                            else f"{backend_name} host-sparse FP x-block shortcut"
                        )
                    elif cpu_large_sparse_shortcut:
                        backend_name = str(jax.default_backend()).strip().lower()
                        reason = (
                            "CPU large FP sparse-LU shortcut"
                            if backend_name == "cpu"
                            else f"{backend_name} host-sparse FP sparse-LU shortcut"
                        )
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: skipping initial Krylov "
                        f"({reason}) -> sparse ILU",
                    )
                if x0_reduced is None:
                    x0_reduced = jnp.zeros_like(rhs_reduced)
                try:
                    r0 = rhs_reduced - mv_reduced(x0_reduced)
                    rn0 = jnp.linalg.norm(r0)
                except Exception:
                    rn0 = jnp.asarray(np.inf, dtype=jnp.float64)
                res_reduced = GMRESSolveResult(x=x0_reduced, residual_norm=jnp.asarray(rn0, dtype=jnp.float64))
                ksp_replay.matvec_fn = mv_reduced
                ksp_replay.b_vec = rhs_reduced
                ksp_replay.precond_fn = preconditioner_reduced
                ksp_replay.x0_vec = x0_reduced
                ksp_replay.precond_side = gmres_precond_side
                ksp_replay.solver_kind = _solver_kind(solve_method)[0]
            else:
                rhs1_progress_notes.krylov_start()
                _mark("rhs1_krylov_solve_start")
                res_reduced = _solve_linear(
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    x0_vec=x0_reduced,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val=solve_method,
                    precond_side=gmres_precond_side,
                )
                res_reduced = _block_gmres_result_ready(res_reduced)
                _mark("rhs1_krylov_solve_done")
                ksp_replay.matvec_fn = mv_reduced
                ksp_replay.b_vec = rhs_reduced
                ksp_replay.precond_fn = preconditioner_reduced
                ksp_replay.x0_vec = x0_reduced
                ksp_replay.precond_side = gmres_precond_side
                ksp_replay.solver_kind = _solver_kind(solve_method)[0]
        if (not probe_shortcut) and preconditioner_reduced is not None and (not _gmres_result_is_finite(res_reduced)):
            if emit is not None:
                emit(0, "solve_v3_full_system_linear_gmres: preconditioned reduced GMRES returned non-finite result; retrying without preconditioner")
            res_reduced = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=None,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val=solve_method,
                precond_side=gmres_precond_side,
            )
            res_reduced = _block_gmres_result_ready(res_reduced)
            ksp_replay.matvec_fn = mv_reduced
            ksp_replay.b_vec = rhs_reduced
            ksp_replay.precond_fn = None
            ksp_replay.x0_vec = x0_reduced
            ksp_replay.precond_side = gmres_precond_side
            ksp_replay.solver_kind = _solver_kind(solve_method)[0]
        force_full_decision = rhs1_pas_force_full_decision_from_env(
            enabled=bool(pas_precond_force_collision),
            has_pas=op.fblock.pas is not None,
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            active_size=int(active_size),
            requested_kind=rhs1_precond_kind_requested,
        )
        if force_full_decision.run:
            forced_kind = str(force_full_decision.forced_kind or "xmg")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: PAS forcing full preconditioner "
                    f"(kind={forced_kind}, residual={float(res_reduced.residual_norm):.3e} > "
                    f"{force_full_decision.ratio:.1f}x target)",
                )
            rhs1_precond_kind = forced_kind
            preconditioner_reduced = _build_rhs1_preconditioner_reduced_with_fallback()
            if use_pas_projection:
                preconditioner_reduced = _wrap_pas_precond(preconditioner_reduced)
            res_reduced, residual_vec, _accepted, _forced_full_elapsed_s = (
                rhs1_run_linear_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    solve_method="incremental",
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear,
                    solver_kind=_solver_kind("incremental")[0],
                    returns_residual_vec=False,
                )
            )
        res_reduced, residual_vec, residual_norm_true = (
            rhs1_recompute_true_residual_result(
                result=res_reduced,
                rhs=rhs_reduced,
                matvec=mv_reduced,
                residual_vec=residual_vec,
                update_residual_vec=False,
            )
        )
        if (
            rhs1_pas_tz_guarded_fallback
            and preconditioner_reduced is not None
            and float(res_reduced.residual_norm) > float(target_reduced)
        ):
            correction_preconditioner = preconditioner_reduced
            correction_kind = resolve_pas_tz_guarded_correction_kind(
                requested=os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION", "")
            )
            if correction_kind is not None or pas_tz_guarded_stream_requested:
                pas_tz_guarded_correction_metadata.update(
                    {
                        "pas_tz_guarded_correction_kind": correction_kind,
                        "pas_tz_guarded_correction_stream_requested": bool(pas_tz_guarded_stream_requested),
                        "pas_tz_guarded_correction_streamed": False,
                        "pas_tz_guarded_correction_full_update_materialized": False,
                    }
                )
            if pas_tz_guarded_stream_requested:
                blocker = (
                    "production-pas-tz-minres-correction-requires-full-residual-direction"
                )
                pas_tz_guarded_correction_metadata["pas_tz_guarded_correction_stream_blocker"] = blocker
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS-TZ guarded streamed "
                        "correction requested but unavailable; using dense minres "
                        "correction because the production preconditioner requires "
                        "a full residual and full preconditioned direction",
                    )
            if correction_kind == "tzfft" and rhs1_pas_tz_guarded_axis != "tzfft":
                try:
                    correction_preconditioner = _build_rhsmode23_tzfft_preconditioner(
                        op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                    if use_pas_projection:
                        correction_preconditioner = _wrap_pas_precond(correction_preconditioner)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS-TZ guarded "
                            "matrix-free correction=tzfft",
                        )
                except Exception as exc:  # noqa: BLE001
                    correction_preconditioner = preconditioner_reduced
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS-TZ guarded "
                            f"matrix-free correction=tzfft unavailable ({type(exc).__name__}); "
                            "using base fallback",
                        )
            guarded_minres = rhs1_pas_tz_guarded_minres_controls_from_env()
            if guarded_minres.steps > 0:
                if pas_tz_guarded_correction_metadata:
                    pas_tz_guarded_correction_metadata[
                        "pas_tz_guarded_correction_full_update_materialized"
                    ] = True
                    pas_tz_guarded_correction_metadata["pas_tz_guarded_correction_minres_steps"] = int(
                        guarded_minres.steps
                    )
                x_minres, residual_minres, minres_history, minres_alphas = _apply_preconditioned_minres_correction(
                    matvec=mv_reduced,
                    rhs=rhs_reduced,
                    x0=res_reduced.x,
                    preconditioner=correction_preconditioner,
                    steps=int(guarded_minres.steps),
                    alpha_clip=float(guarded_minres.alpha_clip),
                    min_improvement=float(guarded_minres.min_improvement),
                )
                if minres_history and float(minres_history[-1]) < float(res_reduced.residual_norm):
                    old_residual = float(res_reduced.residual_norm)
                    residual_norm_true = float(minres_history[-1])
                    residual_vec = residual_minres
                    res_reduced = GMRESSolveResult(
                        x=jnp.asarray(x_minres, dtype=jnp.float64),
                        residual_norm=jnp.asarray(residual_norm_true, dtype=jnp.float64),
                    )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS-TZ guarded minres correction "
                            f"accepted {len(minres_alphas)} step(s), residual="
                            f"{old_residual:.3e}->{residual_norm_true:.3e}",
                        )
        weak_minres_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        weak_minres_steps = rhs1_pas_weak_minres_steps(
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            res_ratio=float(weak_minres_ratio),
        )
        weak_minres = rhs1_pas_weak_minres_controls_from_env(steps=int(weak_minres_steps))
        if (
            (not rhs1_pas_tz_guarded_fallback)
            and preconditioner_reduced is not None
            and weak_minres.steps > 0
            and float(res_reduced.residual_norm) > float(target_reduced)
        ):
            x_minres, residual_minres, minres_history, minres_alphas = _apply_preconditioned_minres_correction(
                matvec=mv_reduced,
                rhs=rhs_reduced,
                x0=res_reduced.x,
                preconditioner=preconditioner_reduced,
                steps=int(weak_minres.steps),
                alpha_clip=float(weak_minres.alpha_clip),
                min_improvement=float(weak_minres.min_improvement),
            )
            if minres_history and float(minres_history[-1]) < float(res_reduced.residual_norm):
                old_residual = float(res_reduced.residual_norm)
                residual_norm_true = float(minres_history[-1])
                residual_vec = residual_minres
                res_reduced = GMRESSolveResult(
                    x=jnp.asarray(x_minres, dtype=jnp.float64),
                    residual_norm=jnp.asarray(residual_norm_true, dtype=jnp.float64),
                )
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: weak PAS minres correction "
                        f"accepted {len(minres_alphas)} step(s), residual="
                        f"{old_residual:.3e}->{residual_norm_true:.3e}",
                    )
        res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        stage2_trigger = rhs1_stage2_trigger(res_ratio=res_ratio, use_dkes=bool(use_dkes))
        fp_force_stage2 = rhs1_fp_force_stage2(
            has_fp=op.fblock.fp is not None,
            include_phi1=bool(op.include_phi1),
            residual_norm=float(res_reduced.residual_norm),
        )
        if fp_force_stage2:
            stage2_trigger = True
        if cpu_large_xblock_shortcut:
            stage2_trigger = False
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: CPU large FP x-block shortcut "
                    "skipping stage2 GMRES and proceeding directly to x-block rescue",
                )
        if cpu_large_sparse_shortcut:
            stage2_trigger = False
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: CPU large FP sparse-LU shortcut "
                    "skipping stage2 GMRES and proceeding directly to sparse rescue",
                )
        if rhs1_pas_stage2_skip(
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            res_ratio=float(res_ratio),
        ):
            stage2_trigger = False
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS stage2 skipped "
                    f"(residual ratio={res_ratio:.3e}; set the relevant PAS stage2 skip ratio to 0 to retry)",
                )
        if rhs1_pas_tz_guarded_fallback and stage2_trigger and not rhs1_pas_tz_guarded_stage2_retry():
            stage2_trigger = False
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: stage2 reduced GMRES skipped "
                    "after guarded PAS-TZ fallback; set "
                    "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY=1 to retry",
                )
        if (
            (not early_dense_shortcut)
            and (not cs0_sparse_first)
            and (cs0_dense_fallback_allowed or int(op.constraint_scheme) != 0)
            and dense_shortcut_ratio > 0
            and res_ratio >= dense_shortcut_ratio
            and (not sparse_prefer_over_dense_shortcut)
        ):
            dense_thresholds = rhs1_dense_fallback_thresholds_from_env(
                dense_fallback_max=int(dense_fallback_max),
                residual_ratio=float(res_ratio),
            )
            dense_fallback_limit = int(dense_thresholds.dense_fallback_limit)
            if dense_fallback_limit > 0 and int(active_size) <= int(dense_fallback_limit):
                early_dense_shortcut = True
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: dense fallback shortcut (early) "
                        f"(ratio={res_ratio:.3e} >= {dense_shortcut_ratio:.1e})",
                    )
            elif emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: dense fallback shortcut skipped "
                    f"(size={int(active_size)} > dense_max={int(dense_fallback_limit)})",
                )
        solver_kind = _solver_kind(solve_method)[0]
        bicgstab_fallback_target = float(target_reduced)
        if bicgstab_fallback_strict:
            # In distributed PAS runs, BiCGStab often reaches parity-accurate
            # solutions while the strict relative target is tiny due to small
            # RHS norms. The policy helper applies that measured floor.
            bicgstab_fallback_target = rhs1_bicgstab_fallback_target_from_env(
                target=float(target_reduced),
                distributed_axis=distributed_axis,
                has_pas=op.fblock.pas is not None,
                include_phi1=bool(op.include_phi1),
            )
        if (not cpu_large_sparse_shortcut) and solver_kind == "bicgstab" and (
            (not _gmres_result_is_finite(res_reduced))
            or (bicgstab_fallback_strict and float(res_reduced.residual_norm) > bicgstab_fallback_target)
        ):
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: BiCGStab fallback to GMRES "
                    f"(residual={float(res_reduced.residual_norm):.3e} > target={bicgstab_fallback_target:.3e})",
                )
            if preconditioner_reduced is None and rhs1_precond_enabled:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced_with_fallback()
            res_reduced = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner_reduced,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side=gmres_precond_side,
            )
            res_reduced = _block_gmres_result_ready(res_reduced)
            ksp_replay.matvec_fn = mv_reduced
            ksp_replay.b_vec = rhs_reduced
            ksp_replay.precond_fn = preconditioner_reduced
            ksp_replay.x0_vec = x0_reduced
            ksp_replay.precond_side = gmres_precond_side
            ksp_replay.solver_kind = "gmres"
        if (
            sparse_prefer_skips_stage2
            and float(res_reduced.residual_norm) > target_reduced
            and stage2_enabled
            and stage2_trigger
            and not early_dense_shortcut
            and not gpu_dkes_sparse_shortcut
            and emit is not None
        ):
            emit(
                1,
                "solve_v3_full_system_linear_gmres: stage2 reduced GMRES skipped "
                "(preferring sparse rescue first)",
            )
        if (
            (float(res_reduced.residual_norm) > target_reduced or fp_force_stage2)
            and stage2_enabled
            and stage2_trigger
            and not early_dense_shortcut
            and not gpu_dkes_sparse_shortcut
            and not sparse_prefer_skips_stage2
            and t.elapsed_s() < stage2_time_cap_s
        ):
            if preconditioner_reduced is None and rhs1_precond_enabled:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced_with_fallback()
            stage2_controls = rhs1_stage2_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
                active_size=int(active_size),
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
                tokamak_pas=bool(tokamak_pas),
            )
            stage2_restart = int(stage2_controls.restart)
            stage2_maxiter = int(stage2_controls.maxiter)
            stage2_method = str(stage2_controls.method)
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: stage2 reduced GMRES "
                    f"(residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e}) "
                    f"restart={stage2_restart} maxiter={stage2_maxiter} method={stage2_method}",
                )
            res_reduced, residual_vec, _accepted, _stage2_elapsed_s = (
                rhs1_run_measured_linear_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(stage2_restart),
                    maxiter=int(stage2_maxiter),
                    solve_method=stage2_method,
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear,
                    solver_kind=_solver_kind(stage2_method)[0],
                    candidate_name=f"stage2_reduced:{stage2_method}",
                    baseline_name="current_reduced",
                    target_value=float(target_reduced),
                    peak_rss_mb=_rss_mb(),
                    returns_residual_vec=False,
                    result_ready=_block_gmres_result_ready,
                )
            )
        pas_fast_accept = _rhsmode1_pas_fast_accept(
            op=op,
            active_size=int(active_size),
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            use_implicit=bool(use_implicit),
        )
        strong_trigger_controls = rhs1_strong_trigger_controls_from_env(
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            has_fp=op.fblock.fp is not None,
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            delay_pas_base_retries=True,
        )
        res_ratio = float(strong_trigger_controls.res_ratio)
        strong_precond_trigger = bool(strong_trigger_controls.trigger)
        fp_force_strong = bool(strong_trigger_controls.fp_force)

        qi_device_seed_context = MatrixFreeQIDeviceSeedContext(
            op=op,
            active_size=int(active_size),
            target_reduced=float(target_reduced),
            mv_reduced=mv_reduced,
            rhs_reduced=rhs_reduced,
            emit=emit,
            timer_elapsed_s=t.elapsed_s,
            rhsmode1_general_metadata=rhsmode1_general_metadata,
        )

        qi_device_early_enabled = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_EARLY",
            default=False,
        )
        qi_device_skip_strong = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_SKIP_STRONG",
            default=False,
        )
        if bool(qi_device_early_enabled or qi_device_skip_strong):
            res_reduced = attempt_matrixfree_qi_device_seed(
                res_reduced, hook="early_active_dof", context=qi_device_seed_context
            )

        if (
            rhs1_precond_kind in {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
            and preconditioner_reduced is not None
            and _rhsmode1_pas_adaptive_smoother_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                use_implicit=bool(use_implicit),
            )
        ):
            smoother_controls = rhs1_pas_adaptive_smoother_controls_from_env()
            smoother = adaptive_pas_smoother(
                matvec=mv_reduced,
                rhs=rhs_reduced,
                preconditioner=preconditioner_reduced,
                x0=res_reduced.x,
                target=float(target_reduced),
                omega=float(smoother_controls.omega),
                max_sweeps=int(smoother_controls.max_sweeps),
            )
            res_reduced, residual_vec, _accepted = (
                rhs1_accept_smoother_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    smoother=smoother,
                    result_factory=lambda *, x, residual_norm: GMRESSolveResult(
                        x=x,
                        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
                    ),
                    candidate_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    restart=restart,
                    maxiter=maxiter,
                    precond_side=gmres_precond_side,
                    solver_kind=_solver_kind("incremental")[0],
                )
            )
        if fp_force_strong:
            strong_precond_trigger = True
        if rhs1_collision_retry_allowed(
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            rhs1_precond_kind=rhs1_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            strong_precond_trigger=bool(strong_precond_trigger),
        ):
            if bicgstab_preconditioner_reduced is None:
                bicgstab_preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if bicgstab_preconditioner_reduced is not None:
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: retry with collision preconditioner "
                        f"(residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                    )
                res_reduced, residual_vec, _accepted, _collision_elapsed_s = (
                    rhs1_run_linear_candidate_and_update_replay(
                        replay_state=ksp_replay,
                        current_result=res_reduced,
                        current_residual_vec=residual_vec,
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=bicgstab_preconditioner_reduced,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        solve_method="incremental",
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear,
                        solver_kind=_solver_kind("incremental")[0],
                        returns_residual_vec=False,
                    )
                )
        large_cpu_sparse_rescue_active = _rhsmode1_large_cpu_sparse_rescue_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
            preconditioner_x=int(preconditioner_x),
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
        )
        sparse_xblock_rescue_active = _rhsmode1_sparse_xblock_rescue_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
            preconditioner_x=int(preconditioner_x),
            pre_theta=int(pre_theta),
            pre_zeta=int(pre_zeta),
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
        )
        sparse_sxblock_rescue_active = _rhsmode1_sparse_sxblock_rescue_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
            preconditioner_x=int(preconditioner_x),
            pre_theta=int(pre_theta),
            pre_zeta=int(pre_zeta),
            use_implicit=bool(use_implicit),
        )
        strong_precond_env = rhs1_strong_preconditioner_env_from_env()
        large_cpu_sparse_rescue_first = _rhsmode1_large_cpu_sparse_rescue_first(
            large_cpu_sparse_rescue=large_cpu_sparse_rescue_active,
            strong_precond_env=strong_precond_env,
        )
        pas_auto_skip = _pas_auto_skip_strong_retry(
            has_pas=op.fblock.pas is not None,
            strong_precond_env=strong_precond_env,
            rhs1_precond_kind=rhs1_precond_kind,
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            ratio=float(pas_auto_strong_ratio),
        )
        strong_control = rhs1_resolved_strong_preconditioner_control(
            strong_precond_env=strong_precond_env,
            has_extra_constraint_block=int(op.constraint_scheme) == 2 and int(op.extra_size) > 0 and (not use_pas_projection),
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            size=int(active_size),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            pas_large_bicgstab_fastpath=bool(pas_large_bicgstab_fastpath),
            cs0_sparse_first=bool(cs0_sparse_first),
            large_cpu_sparse_rescue_first=bool(large_cpu_sparse_rescue_first),
            pas_auto_skip=bool(pas_auto_skip),
            pas_fast_accept=bool(pas_fast_accept),
            pas_precond_force_collision=bool(pas_precond_force_collision),
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
        )
        strong_precond_min = int(strong_control.min_size)
        strong_precond_disabled = bool(strong_control.disabled)
        strong_precond_auto = bool(strong_control.auto)
        if strong_control.reason_cs0_sparse_first and emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: constraintScheme=0 sparse-first "
                "auto mode -> defer strong preconditioner until after sparse ILU",
            )
        if strong_control.reason_large_cpu_sparse_first and emit is not None:
            if emit is not None:
                backend_name = str(jax.default_backend()).strip().lower()
                sparse_label = "large CPU" if backend_name == "cpu" else f"{backend_name} host-sparse"
                emit(
                    1,
                    f"solve_v3_full_system_linear_gmres: {sparse_label} rescue-first "
                    "auto mode -> defer strong preconditioner until after sparse LU",
                )
        if strong_control.reason_pas_auto_skip and emit is not None:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS auto strong preconditioner skipped "
                    f"after base={rhs1_precond_kind} "
                    f"(residual={float(res_reduced.residual_norm):.3e} <= {float(pas_auto_strong_ratio):.1f}x target)",
                )
        if strong_control.reason_pas_fast_accept and emit is not None:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS fast-accept "
                    f"(residual={float(res_reduced.residual_norm):.3e}) -> skip strong preconditioner tail",
                )
        if strong_control.reason_collision_probe_skip and emit is not None:
            emit(1, "solve_v3_full_system_linear_gmres: PAS collision probe disabled strong preconditioner auto")
        elif pas_precond_force_collision and strong_precond_env in {"", "auto"} and emit is not None:
            pas_force_strong_ratio = rhs1_pas_force_strong_ratio_from_env()
            if float(res_reduced.residual_norm) > target_reduced * pas_force_strong_ratio:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS collision probe allows strong preconditioner "
                    f"(residual={float(res_reduced.residual_norm):.3e} > {pas_force_strong_ratio:.1f}x target)",
                )
        strong_precond_kind: str | None = None
        strong_xblock_tz_lmax: int | None = None
        if strong_precond_disabled:
            strong_precond_kind = None
        else:
            strong_precond_kind = requested_rhs1_strong_preconditioner_kind(
                strong_precond_env,
                mode="reduced",
            )

        if strong_precond_kind is None and (not strong_precond_disabled) and strong_precond_auto:
            if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                if op.fblock.pas is not None:
                    auto_sel = auto_rhs1_reduced_strong_kind(
                        has_pas=True,
                        has_fp=False,
                        geom_scheme=int(geom_scheme),
                        use_dkes=bool(use_dkes),
                        active_size=int(active_size),
                        strong_precond_min=int(strong_precond_min),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
                        shard_axis=_matvec_shard_axis(op),
                        device_count=int(jax.device_count()),
                    )
                    strong_precond_kind = auto_sel.kind
                else:
                    strong_precond_kind = "schur"
            else:
                auto_sel = auto_rhs1_reduced_strong_kind(
                    has_pas=op.fblock.pas is not None,
                    has_fp=op.fblock.fp is not None,
                    geom_scheme=int(geom_scheme),
                    use_dkes=bool(use_dkes),
                    active_size=int(active_size),
                    strong_precond_min=int(strong_precond_min),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
                    shard_axis=_matvec_shard_axis(op),
                    device_count=int(jax.device_count()),
                )
                strong_precond_kind = auto_sel.kind
                strong_xblock_tz_lmax = auto_sel.xblock_tz_lmax

        auto_sel = adjust_rhs1_reduced_auto_kind(
            kind=strong_precond_kind,
            has_pas=op.fblock.pas is not None,
            geom_scheme=int(geom_scheme),
            n_zeta=int(op.n_zeta),
            strong_precond_trigger=bool(strong_precond_trigger),
            max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
            n_theta=int(op.n_theta),
        )
        strong_precond_kind = auto_sel.kind
        if auto_sel.xblock_tz_lmax is not None:
            strong_xblock_tz_lmax = auto_sel.xblock_tz_lmax

        auto_sel = adjust_rhs1_theta_line_auto_kind(
            kind=strong_precond_kind,
            n_theta=int(op.n_theta),
            nxi_for_x_sum=int(np.sum(nxi_for_x)) if nxi_for_x.size else 0,
        )
        strong_precond_kind = auto_sel.kind

        if rhs1_pas_weak_strong_retry_skip(
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            res_ratio=float(res_ratio),
        ):
            if emit is not None and strong_precond_kind is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
                    "after weak PAS base residual exceeded skip threshold; set "
                    "SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO=0 to retry",
                )
            strong_precond_kind = None
            strong_precond_trigger = False

        if rhs1_pas_tz_guarded_fallback and not rhs1_pas_tz_guarded_strong_retry_from_env():
            if emit is not None and strong_precond_kind is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
                    "after guarded PAS-TZ fallback; set "
                    "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1 to retry",
                )
            strong_precond_kind = None
            strong_precond_trigger = False

        if bool(qi_device_skip_strong) and strong_precond_kind is not None:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
                    "for QI device preconditioner experiment",
                )
            strong_precond_kind = None
            strong_precond_trigger = False

        if (
            strong_precond_kind is not None
            and _rhs1_residual_needs_rescue(
                float(res_reduced.residual_norm),
                float(target_reduced),
                force=bool(fp_force_strong),
            )
            and strong_precond_trigger
            and not early_dense_shortcut
        ):
            fp_size_guard = rhs1_fp_strong_size_guard_from_env(
                active_size=int(active_size),
                strong_precond_kind=strong_precond_kind,
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
            )
            if fp_size_guard.skip:
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
                        f"(kind={strong_precond_kind}, size={int(active_size)} "
                        f"> fp_max={int(fp_size_guard.max_active_size)})",
                    )
                strong_precond_kind = None
        if (
            strong_precond_kind is not None
            and _rhs1_residual_needs_rescue(
                float(res_reduced.residual_norm),
                float(target_reduced),
                force=bool(fp_force_strong),
            )
            and strong_precond_trigger
            and not early_dense_shortcut
        ):
            strong_precond_kind = adjust_rhs1_pas_schur_strong_kind_from_env(
                kind=strong_precond_kind,
                has_pas=op.fblock.pas is not None,
                base_kind=rhs1_precond_kind,
                residual_norm=float(res_reduced.residual_norm),
                active_size=int(active_size),
            )
            _mark("rhs1_strong_precond_build_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: strong preconditioner fallback "
                    f"kind={strong_precond_kind} (residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )

            strong_preconditioner_reduced = _build_rhs1_strong_preconditioner_reduced_from_kind(
                op=op,
                strong_precond_kind=strong_precond_kind,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                rhs1_xblock_tz_lmax=strong_xblock_tz_lmax,
                dd_block_theta=rhs1_dd_setup.block("theta"),
                dd_overlap_theta=rhs1_dd_setup.overlap("theta", default=1),
                dd_block_zeta=rhs1_dd_setup.block("zeta"),
                dd_overlap_zeta=rhs1_dd_setup.overlap("zeta", default=1),
            )
            _mark("rhs1_strong_precond_build_done")
            if use_pas_projection:
                strong_preconditioner_reduced = _wrap_pas_precond(strong_preconditioner_reduced)

            strong_retry_controls = rhs1_strong_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
            )
            res_reduced, residual_vec, _accepted, _strong_elapsed_s = (
                rhs1_run_measured_linear_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=strong_preconditioner_reduced,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(strong_retry_controls.restart),
                    maxiter=int(strong_retry_controls.maxiter),
                    solve_method="incremental",
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear,
                    solver_kind=_solver_kind("incremental")[0],
                    candidate_name="strong_reduced",
                    baseline_name="current_reduced",
                    target_value=float(target_reduced),
                    peak_rss_mb=_rss_mb(),
                    returns_residual_vec=False,
                    result_ready=_block_gmres_result_ready,
                )
            )

        # Only treat the probe as a "dense shortcut" when the dense branch is
        # actually allowed (probe_shortcut). Otherwise we still want to try
        # stronger preconditioners (e.g. sparse ILU) before giving up.
        dense_shortcut = probe_shortcut
        if not dense_shortcut and dense_shortcut_ratio > 0:
            quick_ratio = float(res_reduced.residual_norm) / max(float(target_reduced), 1e-300)
            if quick_ratio >= dense_shortcut_ratio:
                dense_fallback_max = _rhsmode1_dense_fallback_max(op)
                residual_norm_true = rhs1_true_residual_norm_or_inf(
                    rhs=rhs_reduced,
                    matvec=mv_reduced,
                    x=res_reduced.x,
                )
                res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
                dense_thresholds = rhs1_dense_fallback_thresholds_from_env(
                    dense_fallback_max=int(dense_fallback_max),
                    residual_ratio=float(res_ratio),
                )
                dense_fallback_limit = int(dense_thresholds.dense_fallback_limit)
                force_dense_cs0 = bool(int(op.constraint_scheme) == 0 and (not cs0_sparse_first))
                if force_dense_cs0:
                    dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
                dense_fallback_trigger = bool(dense_thresholds.dense_fallback_trigger)
                if (
                    dense_fallback_limit > 0
                    and int(active_size) <= dense_fallback_limit
                    and dense_fallback_trigger
                    and (float(residual_norm_true) > target_reduced or force_dense_cs0)
                    and res_ratio >= dense_shortcut_ratio
                ):
                    if sparse_prefer_over_dense_shortcut and (not sparse_exact_direct):
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: dense shortcut skipped "
                                "(preferring sparse rescue over dense shortcut)",
                            )
                    else:
                        dense_shortcut = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: dense fallback shortcut "
                                f"(ratio={res_ratio:.3e} >= {dense_shortcut_ratio:.1e})",
                            )

        sparse_policy = rhs1_sparse_rescue_policy_setup(
            sparse_precond_mode=sparse_precond_mode,
            sparse_precond_kind=sparse_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            size=int(active_size),
            sparse_max_size=int(sparse_max_size),
            precond_dtype=_precond_dtype(int(active_size)),
            dense_shortcut=bool(dense_shortcut),
            sparse_exact_direct=bool(sparse_exact_direct),
            large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue_active),
            sparse_xblock_rescue_active=bool(sparse_xblock_rescue_active),
            sparse_sxblock_rescue_active=bool(sparse_sxblock_rescue_active),
            sparse_jax_max_mb=float(sparse_jax_config.max_mb),
            pas_fast_accept=bool(pas_fast_accept),
            gpu_sparse_skip=bool(
                _rhs1_gpu_sparse_fallback_skip_allowed(
                    op=op,
                    rhs1_precond_kind=rhs1_precond_kind,
                    use_active_dof_mode=True,
                    residual_norm=float(res_reduced.residual_norm),
                    target=float(target_reduced),
                )
            ),
        )
        sparse_order = sparse_policy.ordering
        sparse_enabled = bool(sparse_policy.enabled)
        sparse_kind_use = str(sparse_policy.kind_use)
        sparse_xblock_rescue_active = bool(sparse_order.xblock_rescue_active)
        sparse_sxblock_rescue_active = bool(sparse_order.sxblock_rescue_active)
        large_cpu_sparse_label = "large CPU sparse"
        if sparse_order.reason_size_large_cpu:
            sparse_exact_lu = _rhsmode1_large_cpu_sparse_exact_lu_allowed(active_size=int(active_size))
            backend_name = str(jax.default_backend()).strip().lower()
            large_cpu_sparse_label = (
                "large CPU sparse" if backend_name == "cpu" else f"{backend_name} host-sparse"
            )
        if emit is not None:
            rescue_kind = "xblock" if sparse_xblock_rescue_active else "sxblock"
            for _level, _message in rhs1_sparse_rescue_initial_messages(
                ordering=sparse_order,
                size=int(active_size),
                sparse_max_size=int(sparse_max_size),
                sparse_jax_memory_disabled_message=sparse_policy.sparse_jax_memory_disabled_message,
                large_cpu_sparse_exact_lu=bool(sparse_exact_lu),
                large_cpu_label=large_cpu_sparse_label,
                targeted_rescue_kind=rescue_kind,
            ):
                emit(_level, _message)

        host_sparse_direct_used = False
        precond_sparse_xblock_current = None
        explicit_fp_xblock_seed_used = False
        explicit_fp_xblock_seed_residual = float("inf")
        explicit_fp_xblock_seed_improvement_ratio = 1.0
        sparse_xblock_rescue_attempted = False
        sparse_xblock_rescue_built = False
        sparse_xblock_rescue_error: str | None = None
        sparse_xblock_rescue_assembled_host_fp = False
        sparse_xblock_rescue_preconditioner_xi: int | None = None
        sparse_xblock_rescue_seed_residual: float | None = None
        sparse_xblock_rescue_seed_improvement_ratio: float | None = None
        sparse_xblock_rescue_seed_accept_ratio: float | None = None
        sparse_xblock_rescue_seed_refine_steps: int | None = None
        sparse_xblock_rescue_seed_refines_performed: int | None = None
        sparse_xblock_rescue_candidate_residual: float | None = None
        sparse_xblock_rescue_candidate_accepted = False
        sparse_xblock_rescue_reason = "not_needed" if float(res_reduced.residual_norm) <= target_reduced else "inactive"
        fp_xblock_global_correction_allowed = False
        fp_xblock_global_correction_attempted = False
        fp_xblock_global_correction_accepted = False
        fp_xblock_global_correction_reason = "not_evaluated"
        fp_xblock_global_correction_error: str | None = None
        fp_xblock_global_correction_preconditioner: str | None = None
        fp_xblock_global_correction_steps: int | None = None
        fp_xblock_global_correction_accepted_steps: int | None = None
        fp_xblock_global_correction_residual_before: float | None = None
        fp_xblock_global_correction_residual_after: float | None = None
        fp_xblock_global_correction_improvement_ratio: float | None = None
        fp_xblock_global_correction_elapsed_s: float | None = None
        fp_xblock_highx_residual_correction_allowed = False
        fp_xblock_highx_residual_correction_attempted = False
        fp_xblock_highx_residual_correction_accepted = False
        fp_xblock_highx_residual_correction_reason = "not_evaluated"
        fp_xblock_highx_residual_correction_error: str | None = None
        fp_xblock_highx_residual_correction_residual_before: float | None = None
        fp_xblock_highx_residual_correction_residual_after: float | None = None
        fp_xblock_highx_residual_correction_improvement_ratio: float | None = None
        fp_xblock_highx_residual_correction_elapsed_s: float | None = None
        fp_xblock_highx_residual_correction_direction_count: int | None = None
        fp_xblock_highx_residual_correction_direction_names: tuple[str, ...] = ()
        pre_sparse_qi_device_enabled = bool(
            int(op.rhs_mode) == 1
            and _rhs1_bool_env(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER",
                default=False,
            )
            and _rhs1_bool_env(
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
                default=False,
            )
        )
        if (
            pre_sparse_qi_device_enabled
            and sparse_enabled
            and float(res_reduced.residual_norm) > target_reduced
            and not bool(rhsmode1_general_metadata.get("xblock_qi_device_preconditioner_built", False))
        ):
            qi_device_residual_before = float(res_reduced.residual_norm)
            res_reduced = attempt_matrixfree_qi_device_seed(
                res_reduced,
                hook="pre_sparse_active_dof",
                context=qi_device_seed_context,
            )
            if float(res_reduced.residual_norm) < qi_device_residual_before:
                ksp_replay.x0_vec = res_reduced.x

        if emit is not None:
            for _level, _message in rhs1_sparse_rescue_tail_skip_messages(
                ordering=sparse_order,
                residual_norm=float(res_reduced.residual_norm),
                rhs1_precond_kind=rhs1_precond_kind,
            ):
                emit(_level, _message)
        skip_global_sparse_after_xblock = False
        if sparse_enabled and float(res_reduced.residual_norm) > target_reduced:
            if sparse_xblock_rescue_active:
                sparse_xblock_rescue_attempted = True
                sparse_xblock_rescue_reason = "started"
                try:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: v3-like sparse x-block rescue "
                            f"(size={int(active_size)} preconditioner_x={int(preconditioner_x)})",
                        )
                    sparse_xblock_preconditioner_xi = int(preconditioner_xi)
                    if (
                        sparse_xblock_preconditioner_xi == 0
                        and (not bool(use_implicit))
                        and op.fblock.fp is not None
                        and op.fblock.pas is None
                    ):
                        sparse_xblock_preconditioner_xi = 1
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: promoting sparse x-block rescue "
                                "preconditioner_xi 0 -> 1 for stronger host FP factorization",
                            )
                    assembled_host_fp = _rhsmode1_fp_xblock_assembled_host_allowed(
                        op=op,
                        preconditioner_species=preconditioner_species,
                        preconditioner_xi=sparse_xblock_preconditioner_xi,
                        use_implicit=bool(use_implicit),
                        active_size=int(active_size),
                    )
                    sparse_xblock_rescue_assembled_host_fp = bool(assembled_host_fp)
                    sparse_xblock_rescue_preconditioner_xi = int(sparse_xblock_preconditioner_xi)
                    _mark("rhs1_sparse_precond_build_start")
                    precond_sparse_xblock = _build_rhsmode1_xblock_tz_sparse_preconditioner(
                        op=op,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                        build_jax_factors=bool(use_implicit),
                        preconditioner_species=preconditioner_species,
                        preconditioner_xi=sparse_xblock_preconditioner_xi,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                        force_assembled_host_fp=bool(assembled_host_fp),
                        emit=emit,
                    )
                    precond_sparse_xblock_current = precond_sparse_xblock
                    sparse_xblock_rescue_built = True
                    _mark("rhs1_sparse_precond_build_done")
                    _mark("rhs1_sparse_precond_solve_start")
                    if use_implicit:
                        res_sparse_xblock = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            precond_fn=precond_sparse_xblock,
                            x0_vec=res_reduced.x,
                            tol_val=tol,
                            atol_val=atol,
                            restart_val=restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            precond_side=gmres_precond_side,
                        )
                    else:
                        res_sparse_xblock = None
                        if assembled_host_fp:
                            refine_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XBLOCK_REFINES", "").strip()
                            try:
                                refine_steps = int(refine_env) if refine_env else 2
                            except ValueError:
                                refine_steps = 2
                            refine_steps = max(0, int(refine_steps))
                            accept_ratio = rhs1_parse_accept_ratio(
                                env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_ACCEPT_RATIO",
                                default=10.0,
                            )
                            sparse_xblock_rescue_seed_accept_ratio = float(accept_ratio)
                            polish_enabled = rhs1_polish_enabled(
                                env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH",
                            )
                            polish_restart, polish_maxiter = rhs1_parse_polish_gmres_config(
                                restart_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_RESTART",
                                maxiter_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_MAXITER",
                                default_restart=min(int(restart), 40),
                                default_maxiter=min(int(maxiter or 80), 80),
                                active_size=int(active_size),
                                large_active_min_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_MIN",
                                large_default_restart_env_name=(
                                    "SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_RESTART_DEFAULT"
                                ),
                                large_default_maxiter_env_name=(
                                    "SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_MAXITER_DEFAULT"
                                ),
                                default_large_restart=10,
                                default_large_maxiter=1,
                                min_maxiter=1,
                            )
                            base_residual_norm = float(res_reduced.residual_norm)
                            x_trial = jnp.asarray(precond_sparse_xblock(rhs_reduced), dtype=jnp.float64)
                            residual_vec_sparse_xblock = rhs_reduced - mv_reduced(x_trial)
                            residual_norm_sparse_xblock = float(jnp.linalg.norm(residual_vec_sparse_xblock))
                            explicit_fp_xblock_seed_residual = float(residual_norm_sparse_xblock)
                            sparse_xblock_rescue_seed_residual = float(residual_norm_sparse_xblock)
                            if np.isfinite(residual_norm_sparse_xblock) and residual_norm_sparse_xblock > 0.0:
                                explicit_fp_xblock_seed_improvement_ratio = float(base_residual_norm) / float(
                                    residual_norm_sparse_xblock
                                )
                                sparse_xblock_rescue_seed_improvement_ratio = float(
                                    explicit_fp_xblock_seed_improvement_ratio
                                )
                            elif np.isfinite(residual_norm_sparse_xblock):
                                explicit_fp_xblock_seed_improvement_ratio = float("inf")
                                sparse_xblock_rescue_seed_improvement_ratio = float("inf")
                            if emit is not None:
                                emit(
                                    0,
                                    "solve_v3_full_system_linear_gmres: explicit FP x-block seed "
                                    f"(residual={residual_norm_sparse_xblock:.6e} current={base_residual_norm:.6e})",
                                )
                            performed_refines = 0
                            for refine_index in range(refine_steps):
                                if not np.isfinite(residual_norm_sparse_xblock) or residual_norm_sparse_xblock == 0.0:
                                    break
                                dx_trial = jnp.asarray(precond_sparse_xblock(residual_vec_sparse_xblock), dtype=jnp.float64)
                                x_next = x_trial + dx_trial
                                residual_vec_next = rhs_reduced - mv_reduced(x_next)
                                residual_norm_next = float(jnp.linalg.norm(residual_vec_next))
                                if not np.isfinite(residual_norm_next) or residual_norm_next >= residual_norm_sparse_xblock:
                                    break
                                x_trial = x_next
                                residual_vec_sparse_xblock = residual_vec_next
                                residual_norm_sparse_xblock = residual_norm_next
                                performed_refines = int(refine_index) + 1
                            sparse_xblock_rescue_seed_refine_steps = int(refine_steps)
                            sparse_xblock_rescue_seed_refines_performed = int(performed_refines)
                            if emit is not None and int(refine_steps) > 0:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: explicit FP x-block refinement "
                                    f"steps={int(performed_refines)}/{int(refine_steps)} "
                                    f"residual={float(residual_norm_sparse_xblock):.6e}",
                                )
                            if (
                                np.isfinite(residual_norm_sparse_xblock)
                                and residual_norm_sparse_xblock <= max(float(target_reduced), base_residual_norm * accept_ratio)
                            ):
                                sparse_xblock_rescue_reason = "seed_accepted"
                                if polish_enabled and residual_norm_sparse_xblock > float(target_reduced):
                                    polish_precond = precond_sparse_xblock if precond_sparse_xblock is not None else ksp_replay.precond_fn
                                    if emit is not None:
                                        emit(
                                            1,
                                            "solve_v3_full_system_linear_gmres: explicit FP x-block polish "
                                            f"start residual={float(residual_norm_sparse_xblock):.6e} "
                                            f"target={float(target_reduced):.3e} restart={int(polish_restart)} "
                                            f"maxiter={int(polish_maxiter)}",
                                        )
                                    x_np, _rn_sparse_xblock, _history = gmres_solve_with_history_scipy(
                                        matvec=mv_reduced,
                                        b=rhs_reduced,
                                        preconditioner=polish_precond,
                                        x0=x_trial,
                                        tol=tol,
                                        atol=atol,
                                        restart=polish_restart,
                                        maxiter=polish_maxiter,
                                        precondition_side=gmres_precond_side,
                                    )
                                    x_polish = jnp.asarray(x_np, dtype=jnp.float64)
                                    residual_vec_polish = rhs_reduced - mv_reduced(x_polish)
                                    residual_norm_polish = float(jnp.linalg.norm(residual_vec_polish))
                                    if emit is not None:
                                        emit(
                                            1,
                                            "solve_v3_full_system_linear_gmres: explicit FP x-block polish "
                                            f"done residual={float(residual_norm_polish):.6e}",
                                        )
                                    if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm_sparse_xblock:
                                        x_trial = x_polish
                                        residual_norm_sparse_xblock = residual_norm_polish
                                res_sparse_xblock = GMRESSolveResult(
                                    x=x_trial,
                                    residual_norm=jnp.asarray(residual_norm_sparse_xblock, dtype=jnp.float64),
                                )
                            elif emit is not None:
                                sparse_xblock_rescue_reason = "seed_rejected_accept_gate"
                                emit(
                                    0,
                                    "solve_v3_full_system_linear_gmres: explicit FP x-block seed rejected "
                                    f"(residual={residual_norm_sparse_xblock:.6e}, base={base_residual_norm:.6e}, "
                                    f"accept_ratio={accept_ratio:.1e})",
                                )
                        else:
                            x_np, _rn_sparse_xblock, _history = gmres_solve_with_history_scipy(
                                matvec=mv_reduced,
                                b=rhs_reduced,
                                preconditioner=precond_sparse_xblock,
                                x0=res_reduced.x,
                                tol=tol,
                                atol=atol,
                                restart=restart,
                                maxiter=maxiter,
                                precondition_side=gmres_precond_side,
                            )
                            x_sparse_xblock = jnp.asarray(x_np, dtype=jnp.float64)
                            residual_vec_sparse_xblock = rhs_reduced - mv_reduced(x_sparse_xblock)
                            res_sparse_xblock = GMRESSolveResult(
                                x=x_sparse_xblock,
                                residual_norm=jnp.asarray(jnp.linalg.norm(residual_vec_sparse_xblock), dtype=jnp.float64),
                            )
                            sparse_xblock_rescue_candidate_residual = float(res_sparse_xblock.residual_norm)
                            sparse_xblock_rescue_reason = "gmres_candidate"
                    _mark("rhs1_sparse_precond_solve_done")
                    if res_sparse_xblock is not None and float(res_sparse_xblock.residual_norm) < float(res_reduced.residual_norm):
                        sparse_xblock_rescue_candidate_accepted = True
                        sparse_xblock_rescue_candidate_residual = float(res_sparse_xblock.residual_norm)
                        if sparse_xblock_rescue_reason == "gmres_candidate":
                            sparse_xblock_rescue_reason = "gmres_candidate_improved"
                        res_reduced = res_sparse_xblock
                        explicit_fp_xblock_seed_used = bool(assembled_host_fp and (not bool(use_implicit)))
                        if assembled_host_fp:
                            ksp_replay.x0_vec = res_reduced.x
                        else:
                            ksp_replay.matvec_fn = mv_reduced
                            ksp_replay.b_vec = rhs_reduced
                            ksp_replay.precond_fn = precond_sparse_xblock
                            ksp_replay.x0_vec = res_reduced.x
                            ksp_replay.restart = restart
                            ksp_replay.maxiter = maxiter
                            ksp_replay.precond_side = gmres_precond_side
                            ksp_replay.solver_kind = _solver_kind("incremental")[0]
                except Exception as exc:  # noqa: BLE001
                    sparse_xblock_rescue_error = f"{type(exc).__name__}: {exc}"
                    sparse_xblock_rescue_reason = "exception"
                    if emit is not None:
                        emit(1, f"xblock_sparse: failed ({type(exc).__name__}: {exc})")
            else:
                sparse_xblock_rescue_reason = "inactive_by_policy"
            fp_xblock_global_correction_allowed = _rhsmode1_fp_xblock_global_correction_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                sparse_xblock_candidate_accepted=bool(sparse_xblock_rescue_candidate_accepted),
                use_implicit=bool(use_implicit),
            )
            if fp_xblock_global_correction_allowed:
                fp_xblock_global_correction_attempted = True
                fp_xblock_global_correction_reason = "started"
                correction_precond = precond_sparse_xblock_current or preconditioner_reduced
                fp_xblock_global_correction_preconditioner = (
                    "sparse_xblock" if precond_sparse_xblock_current is not None else "base"
                )
                if correction_precond is None:
                    fp_xblock_global_correction_reason = "missing_preconditioner"
                else:
                    correction_steps = _rhs1_int_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_STEPS",
                        default=3,
                        minimum=1,
                    )
                    correction_alpha_clip = _rhs1_float_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_ALPHA_CLIP",
                        default=10.0,
                        minimum=0.0,
                    )
                    correction_min_improvement = _rhs1_float_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_MIN_IMPROVEMENT",
                        default=0.0,
                        minimum=0.0,
                    )
                    correction_precond_clip = _rhs1_float_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION_PRECONDITIONER_CLIP",
                        default=1.0e100,
                        minimum=0.0,
                    )
                    fp_xblock_global_correction_steps = int(correction_steps)
                    fp_xblock_global_correction_residual_before = float(res_reduced.residual_norm)
                    correction_start_s = float(t.elapsed_s())
                    _mark("rhs1_fp_xblock_global_correction_start")
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP x-block global correction "
                            f"(steps={int(correction_steps)} "
                            f"preconditioner={fp_xblock_global_correction_preconditioner} "
                            f"residual={float(res_reduced.residual_norm):.6e})",
                        )
                    try:
                        x_corr, residual_corr, correction_history, correction_alphas = (
                            _apply_preconditioned_minres_correction(
                                matvec=mv_reduced,
                                rhs=rhs_reduced,
                                x0=res_reduced.x,
                                preconditioner=_safe_preconditioner(
                                    correction_precond,
                                    clip=float(correction_precond_clip),
                                ),
                                steps=int(correction_steps),
                                alpha_clip=float(correction_alpha_clip),
                                min_improvement=float(correction_min_improvement),
                            )
                        )
                        fp_xblock_global_correction_elapsed_s = float(t.elapsed_s() - correction_start_s)
                        fp_xblock_global_correction_accepted_steps = int(len(correction_alphas))
                        if correction_history:
                            fp_xblock_global_correction_residual_after = float(correction_history[-1])
                        if (
                            correction_history
                            and np.isfinite(float(correction_history[-1]))
                            and float(correction_history[-1]) < float(res_reduced.residual_norm)
                        ):
                            before = float(res_reduced.residual_norm)
                            after = float(correction_history[-1])
                            fp_xblock_global_correction_accepted = True
                            fp_xblock_global_correction_reason = "accepted"
                            fp_xblock_global_correction_improvement_ratio = float(before) / max(after, 1.0e-300)
                            residual_vec = residual_corr
                            res_reduced = GMRESSolveResult(
                                x=jnp.asarray(x_corr, dtype=jnp.float64),
                                residual_norm=jnp.asarray(after, dtype=jnp.float64),
                            )
                            ksp_replay.x0_vec = res_reduced.x
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP x-block global correction accepted "
                                    f"{before:.3e}->{after:.3e} "
                                    f"steps={int(len(correction_alphas))}",
                                )
                        else:
                            fp_xblock_global_correction_reason = "no_improvement"
                        _mark("rhs1_fp_xblock_global_correction_done")
                    except Exception as exc:  # noqa: BLE001
                        fp_xblock_global_correction_error = f"{type(exc).__name__}: {exc}"
                        fp_xblock_global_correction_reason = "exception"
                        fp_xblock_global_correction_elapsed_s = float(t.elapsed_s() - correction_start_s)
                        _mark("rhs1_fp_xblock_global_correction_failed")
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: FP x-block global correction failed "
                                f"({type(exc).__name__}: {exc})",
                            )
            else:
                correction_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION", "").strip().lower()
                fp_xblock_global_correction_reason = (
                    "disabled" if correction_env not in {"1", "true", "yes", "on"} else "policy_guard"
                )
            highx_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION", "").strip().lower()
            highx_enabled = highx_env in {"1", "true", "yes", "on"}
            highx_active_max = _rhs1_int_env(
                "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MAX",
                default=600000,
                minimum=0,
            )
            fp_xblock_highx_residual_correction_allowed = bool(
                highx_enabled
                and (not bool(use_implicit))
                and jax.default_backend() == "cpu"
                and int(op.rhs_mode) == 1
                and (not bool(op.include_phi1))
                and op.fblock.fp is not None
                and op.fblock.pas is None
                and bool(cpu_large_xblock_shortcut)
                and bool(explicit_fp_xblock_seed_used)
                and bool(sparse_xblock_rescue_candidate_accepted)
                and float(res_reduced.residual_norm) > float(target_reduced)
                and (int(highx_active_max) <= 0 or int(active_size) <= int(highx_active_max))
                and reduce_full is not None
                and expand_reduced is not None
            )
            if fp_xblock_highx_residual_correction_allowed:
                fp_xblock_highx_residual_correction_attempted = True
                fp_xblock_highx_residual_correction_reason = "started"
                highx_start_s = float(t.elapsed_s())
                _mark("rhs1_fp_xblock_highx_residual_correction_start")
                try:
                    highx_host_block_max_env = os.environ.get(
                        "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX",
                        "",
                    ).strip()
                    highx_include_factored = _rhs1_bool_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_FACTORED",
                        default=False,
                    )
                    highx_slices: list[tuple[str, int, int]] = []
                    nxi_for_x_highx = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                    for s_highx in range(int(op.n_species)):
                        for ix_highx in range(int(op.n_x)):
                            n_lx_highx = int(nxi_for_x_highx[int(ix_highx)])
                            block_size_highx = int(n_lx_highx * int(op.n_theta) * int(op.n_zeta))
                            if block_size_highx <= 0:
                                continue
                            block_factor_allowed = _rhs1_xblock_sparse_host_policy.rhs1_xblock_sparse_host_block_factor_allowed(
                                block_size=int(block_size_highx),
                                max_block_size_env_value=highx_host_block_max_env,
                            )
                            if block_factor_allowed and not bool(highx_include_factored):
                                continue
                            start_highx = int(
                                (int(s_highx) * int(op.n_x) + int(ix_highx))
                                * int(op.n_xi)
                                * int(op.n_theta)
                                * int(op.n_zeta)
                            )
                            highx_slices.append(
                                (
                                    f"s{int(s_highx)}_x{int(ix_highx)}",
                                    int(start_highx),
                                    int(block_size_highx),
                                )
                            )
                    max_blocks = _rhs1_int_env(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MAX_BLOCKS",
                        default=16,
                        minimum=1,
                    )
                    highx_slices = highx_slices[: int(max_blocks)]
                    if not highx_slices:
                        fp_xblock_highx_residual_correction_reason = "no_skipped_blocks"
                    else:
                        highx_steps = _rhs1_int_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_STEPS",
                            default=1,
                            minimum=1,
                        )
                        highx_max_directions = _rhs1_int_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MAX_DIRECTIONS",
                            default=12,
                            minimum=1,
                        )
                        highx_alpha_clip = _rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_ALPHA_CLIP",
                            default=0.0,
                            minimum=0.0,
                        )
                        highx_rcond = _rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_RCOND",
                            default=1.0e-12,
                            minimum=0.0,
                        )
                        highx_min_improvement = _rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MIN_IMPROVEMENT",
                            default=0.0,
                            minimum=0.0,
                        )
                        highx_include_all = _rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_ALL",
                            default=True,
                        )
                        highx_include_raw = _rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_RAW",
                            default=False,
                        )
                        fp_xblock_highx_residual_correction_residual_before = float(res_reduced.residual_norm)

                        def _highx_direction_builder(residual_reduced: jnp.ndarray) -> Sequence[tuple[str, jnp.ndarray]]:
                            residual_full_np = np.asarray(
                                jax.device_get(expand_reduced(jnp.asarray(residual_reduced, dtype=jnp.float64))),
                                dtype=np.float64,
                            ).reshape((-1,))
                            directions: list[tuple[str, jnp.ndarray]] = []
                            if bool(highx_include_raw):
                                directions.append(
                                    (
                                        "raw_residual",
                                        jnp.asarray(residual_reduced, dtype=jnp.float64),
                                    )
                                )

                            def _direction_for(blocks: Sequence[tuple[str, int, int]], name: str) -> jnp.ndarray | None:
                                full_np = np.zeros((int(op.total_size),), dtype=np.float64)
                                for _label, start, block_size in blocks:
                                    sl = slice(int(start), int(start + block_size))
                                    full_np[sl] = residual_full_np[sl]
                                if not np.any(np.isfinite(full_np) & (full_np != 0.0)):
                                    return None
                                return reduce_full(jnp.asarray(full_np, dtype=jnp.float64))

                            if bool(highx_include_all):
                                all_direction = _direction_for(highx_slices, "highx_all")
                                if all_direction is not None:
                                    directions.append(("highx_all", all_direction))
                            for label, start, block_size in highx_slices:
                                direction = _direction_for(((label, start, block_size),), f"highx_{label}")
                                if direction is not None:
                                    directions.append((f"highx_{label}", direction))
                            return tuple(directions)

                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: FP high-x residual-equation correction "
                                f"(blocks={len(highx_slices)} directions<={int(highx_max_directions)} "
                                f"residual={float(res_reduced.residual_norm):.6e})",
                            )
                        x_highx, residual_highx, highx_history, highx_counts, highx_names = (
                            _apply_subspace_minres_correction(
                                matvec=mv_reduced,
                                rhs=rhs_reduced,
                                x0=res_reduced.x,
                                direction_builder=_highx_direction_builder,
                                steps=int(highx_steps),
                                max_directions=int(highx_max_directions),
                                alpha_clip=float(highx_alpha_clip),
                                rcond=float(highx_rcond),
                                min_improvement=float(highx_min_improvement),
                            )
                        )
                        fp_xblock_highx_residual_correction_elapsed_s = float(t.elapsed_s() - highx_start_s)
                        fp_xblock_highx_residual_correction_direction_count = int(sum(highx_counts))
                        fp_xblock_highx_residual_correction_direction_names = tuple(highx_names)
                        if highx_history:
                            fp_xblock_highx_residual_correction_residual_after = float(highx_history[-1])
                        if (
                            highx_history
                            and np.isfinite(float(highx_history[-1]))
                            and float(highx_history[-1]) < float(res_reduced.residual_norm)
                        ):
                            highx_before = float(res_reduced.residual_norm)
                            highx_after = float(highx_history[-1])
                            fp_xblock_highx_residual_correction_accepted = True
                            fp_xblock_highx_residual_correction_reason = "accepted"
                            fp_xblock_highx_residual_correction_improvement_ratio = highx_before / max(
                                highx_after,
                                1.0e-300,
                            )
                            residual_vec = residual_highx
                            res_reduced = GMRESSolveResult(
                                x=jnp.asarray(x_highx, dtype=jnp.float64),
                                residual_norm=jnp.asarray(highx_after, dtype=jnp.float64),
                            )
                            ksp_replay.x0_vec = res_reduced.x
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP high-x residual-equation correction "
                                    f"accepted {highx_before:.3e}->{highx_after:.3e} "
                                    f"directions={int(sum(highx_counts))}",
                                )
                        elif fp_xblock_highx_residual_correction_reason == "started":
                            fp_xblock_highx_residual_correction_reason = "no_improvement"
                    _mark("rhs1_fp_xblock_highx_residual_correction_done")
                except Exception as exc:  # noqa: BLE001
                    fp_xblock_highx_residual_correction_error = f"{type(exc).__name__}: {exc}"
                    fp_xblock_highx_residual_correction_reason = "exception"
                    fp_xblock_highx_residual_correction_elapsed_s = float(t.elapsed_s() - highx_start_s)
                    _mark("rhs1_fp_xblock_highx_residual_correction_failed")
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP high-x residual-equation correction failed "
                            f"({type(exc).__name__}: {exc})",
                        )
            else:
                fp_xblock_highx_residual_correction_reason = (
                    "disabled" if not highx_enabled else "policy_guard"
                )
            skip_global_sparse_after_xblock = _rhsmode1_skip_global_sparse_after_xblock_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                use_implicit=bool(use_implicit),
            )
            if (
                large_cpu_sparse_rescue_active
                and (not sparse_exact_lu)
                and _rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed(
                    op=op,
                    active_size=int(active_size),
                    preconditioner_x=int(preconditioner_x),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    xblock_seed_residual=float(explicit_fp_xblock_seed_residual),
                    xblock_seed_improvement_ratio=float(explicit_fp_xblock_seed_improvement_ratio),
                    use_implicit=bool(use_implicit),
                )
            ):
                sparse_exact_lu = True
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: promoting large CPU sparse rescue to exact LU "
                        f"after x-block seed (residual={float(explicit_fp_xblock_seed_residual):.3e} "
                        f"improvement={float(explicit_fp_xblock_seed_improvement_ratio):.1f}x)",
                    )
            if skip_global_sparse_after_xblock and emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: skipping global sparse rescue after x-block seed "
                    f"(residual={float(res_reduced.residual_norm):.3e})",
                )
            if (
                float(res_reduced.residual_norm) > target_reduced
                and sparse_sxblock_rescue_active
                and (not skip_global_sparse_after_xblock)
            ):
                try:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: sparse sxblock_tz rescue "
                            f"(size={int(active_size)} n_species={int(op.n_species)})",
                        )
                    _mark("rhs1_sparse_precond_build_start")
                    x_sparse_sxblock = _compute_rhsmode1_sxblock_tz_sparse_host_seed(
                        op=op,
                        rhs_reduced=rhs_reduced,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                        emit=emit,
                    )
                    _mark("rhs1_sparse_precond_build_done")
                    _mark("rhs1_sparse_precond_solve_start")
                    residual_vec_sparse_sxblock = rhs_reduced - mv_reduced(x_sparse_sxblock)
                    res_sparse_sxblock = GMRESSolveResult(
                        x=x_sparse_sxblock,
                        residual_norm=jnp.asarray(jnp.linalg.norm(residual_vec_sparse_sxblock), dtype=jnp.float64),
                    )
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: explicit sxblock seed "
                            f"(residual={float(res_sparse_sxblock.residual_norm):.6e})",
                        )
                    _mark("rhs1_sparse_precond_solve_done")
                    if float(res_sparse_sxblock.residual_norm) < float(res_reduced.residual_norm):
                        res_reduced = res_sparse_sxblock
                        ksp_replay.x0_vec = res_reduced.x
                        if float(res_reduced.residual_norm) > target_reduced:
                            polish_precond = precond_sparse_xblock_current or preconditioner_reduced
                            if polish_precond is not None:
                                sxblock_polish_restart, sxblock_polish_maxiter = rhs1_parse_polish_gmres_config(
                                    restart_env_name="SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_RESTART",
                                    maxiter_env_name="SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_MAXITER",
                                    default_restart=min(int(restart), 40),
                                    default_maxiter=min(max(40, int(maxiter or 120)), 120),
                                )
                                if emit is not None:
                                    emit(
                                        0,
                                        "solve_v3_full_system_linear_gmres: sxblock seed polish "
                                        f"restart={sxblock_polish_restart} maxiter={sxblock_polish_maxiter}",
                                    )
                                x_np, _rn_sxpolish, _history = gmres_solve_with_history_scipy(
                                    matvec=mv_reduced,
                                    b=rhs_reduced,
                                    preconditioner=polish_precond,
                                    x0=res_reduced.x,
                                    tol=tol,
                                    atol=atol,
                                    restart=sxblock_polish_restart,
                                    maxiter=sxblock_polish_maxiter,
                                    precondition_side=gmres_precond_side,
                                )
                                x_sxpolish = jnp.asarray(x_np, dtype=jnp.float64)
                                residual_vec_sxpolish = rhs_reduced - mv_reduced(x_sxpolish)
                                res_sxpolish = GMRESSolveResult(
                                    x=x_sxpolish,
                                    residual_norm=jnp.asarray(jnp.linalg.norm(residual_vec_sxpolish), dtype=jnp.float64),
                                )
                                if float(res_sxpolish.residual_norm) < float(res_reduced.residual_norm):
                                    res_reduced = res_sxpolish
                                    ksp_replay.matvec_fn = mv_reduced
                                    ksp_replay.b_vec = rhs_reduced
                                    ksp_replay.precond_fn = polish_precond
                                    ksp_replay.x0_vec = res_reduced.x
                                    ksp_replay.restart = sxblock_polish_restart
                                    ksp_replay.maxiter = sxblock_polish_maxiter
                                    ksp_replay.precond_side = gmres_precond_side
                                    ksp_replay.solver_kind = _solver_kind("incremental")[0]
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(1, f"sxblock_sparse: failed ({type(exc).__name__}: {exc})")
            if (
                float(res_reduced.residual_norm) > target_reduced
                and sparse_kind_use == "jax"
                and (not skip_global_sparse_after_xblock)
            ):
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_jax",
                        active_size=int(active_size),
                        use_active_dof_mode=True,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    precond_dtype = _precond_dtype(int(active_size))
                    precond_sparse = build_sparse_jax_retry_preconditioner(
                        SparseJAXRetryPreconditionerBuildContext(
                            matvec=mv_reduced,
                            n=int(active_size),
                            dtype=precond_dtype,
                            cache_key=cache_key,
                            drop_tol=sparse_drop_tol,
                            drop_rel=sparse_drop_rel,
                            reg=sparse_jax_config.reg,
                            omega=sparse_jax_config.omega,
                            sweeps=sparse_jax_config.sweeps,
                            emit=emit,
                            builder=_build_sparse_jax_preconditioner_from_matvec,
                        )
                    )
                    _mark("rhs1_sparse_precond_build_done")
                    res_reduced, residual_vec, _accepted, _elapsed = (
                        rhs1_run_measured_linear_candidate_and_update_replay(
                            replay_state=ksp_replay,
                            current_result=res_reduced,
                            current_residual_vec=residual_vec,
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            precond_fn=precond_sparse,
                            tol=tol,
                            atol=atol,
                            restart=restart,
                            maxiter=maxiter,
                            solve_method="incremental",
                            precond_side=gmres_precond_side,
                            solve_linear=_solve_linear,
                            solver_kind=_solver_kind("incremental")[0],
                            candidate_name="sparse_jax_reduced",
                            baseline_name="current_reduced",
                            target_value=target_reduced,
                            peak_rss_mb=_rss_mb(),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(1, f"sparse_jax: failed ({type(exc).__name__}: {exc})")
            elif float(res_reduced.residual_norm) > target_reduced and (not skip_global_sparse_after_xblock):
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_lu" if sparse_exact_lu else "sparse_ilu",
                        active_size=int(active_size),
                        use_active_dof_mode=True,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    sparse_factor_controls = resolve_sparse_host_or_ilu_factor_controls(
                        n=int(active_size),
                        cache_key=cache_key,
                        sparse_exact_lu=bool(sparse_exact_lu),
                        use_implicit=bool(use_implicit),
                        force_host_sparse_direct=bool(large_cpu_sparse_rescue_active),
                        sparse_ilu_dense_max=int(sparse_ilu_dense_max),
                        sparse_dense_cache_max=int(sparse_dense_cache_max),
                        host_sparse_direct_allowed=_rhsmode1_host_sparse_direct_allowed,
                        host_sparse_factor_dtype=_host_sparse_factor_dtype,
                        sparse_factor_cache_key=_sparse_factor_cache_key,
                        explicit_sparse_host_direct_allowed=_rhsmode1_explicit_sparse_host_direct_allowed,
                    )
                    host_sparse_direct_wanted = sparse_factor_controls.host_sparse_direct_wanted
                    factor_dtype = sparse_factor_controls.factor_dtype
                    sparse_factor_build = build_sparse_host_or_ilu_factor(
                        SparseHostOrILUFactorBuildContext(
                            matvec=mv_reduced,
                            n=int(active_size),
                            dtype=rhs_reduced.dtype,
                            cache_key=sparse_factor_controls.cache_key_use,
                            factor_dtype=sparse_factor_controls.factor_dtype,
                            drop_tol=sparse_drop_tol,
                            drop_rel=sparse_drop_rel,
                            ilu_drop_tol=sparse_ilu_drop_tol,
                            fill_factor=sparse_ilu_fill,
                            build_dense_factors=sparse_factor_controls.build_dense_factors,
                            build_jax_factors=sparse_factor_controls.build_jax_factors,
                            store_dense=sparse_factor_controls.store_dense,
                            factorization="lu" if sparse_exact_lu else "ilu",
                            emit=emit,
                            host_sparse_direct_wanted=sparse_factor_controls.host_sparse_direct_wanted,
                            explicit_sparse_allowed=sparse_factor_controls.explicit_sparse_allowed,
                            build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
                            build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
                        )
                    )
                    explicit_sparse_operator = sparse_factor_build.explicit_sparse_operator
                    explicit_sparse_factor = sparse_factor_build.explicit_sparse_factor
                    a_csr_full = sparse_factor_build.a_csr_full
                    _a_csr_drop = sparse_factor_build.a_csr_drop
                    ilu = sparse_factor_build.ilu
                    a_dense_cache = sparse_factor_build.a_dense_cache
                    l_dense = sparse_factor_build.l_dense
                    u_dense = sparse_factor_build.u_dense
                    l_unit_diag = sparse_factor_build.l_unit_diag
                    dense_matrix_cache = a_dense_cache
                    _mark("rhs1_sparse_precond_build_done")
                    host_sparse_direct = host_sparse_direct_wanted
                    sparse_retry_timer = Timer()

                    if host_sparse_direct and ilu is not None:
                        host_sparse_direct_used = True
                        sparse_host_fallback = sparse_host_direct_fallback_payload(
                            explicit_sparse_factor=explicit_sparse_factor,
                            explicit_sparse_operator=explicit_sparse_operator,
                            ilu=ilu,
                            a_csr_full=a_csr_full,
                            rhs=rhs_reduced,
                            factor_dtype=factor_dtype,
                            refine_steps=_host_sparse_direct_refine_steps(
                                "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_REFINE",
                                default=2,
                            ),
                            matvec=mv_reduced,
                            target=float(target_reduced),
                            tol=tol,
                            atol=atol,
                            restart=restart,
                            maxiter=maxiter,
                            precondition_side=gmres_precond_side,
                            emit=emit,
                            backend_name=jax.default_backend(),
                            polish_enabled=rhs1_polish_enabled,
                            parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
                            direct_solve_with_refinement=_host_direct_solve_with_refinement,
                            ilu_solve_with_refinement=_host_sparse_direct_solve_with_refinement,
                            host_sparse_direct_polish=_host_sparse_direct_polish,
                        )
                        res_sparse = GMRESSolveResult(
                            x=sparse_host_fallback.x,
                            residual_norm=sparse_host_fallback.residual_norm,
                        )
                        residual_vec_sparse = sparse_host_fallback.residual_vec
                        _mv_sparse = mv_reduced
                        _precond_sparse = None
                    elif use_implicit:
                        ilu_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(cache_key)
                        precond_build = build_sparse_ilu_preconditioner_from_cache(
                            SparseILUPreconditionerBuildContext(
                                cache_entry=ilu_cache,
                                l_dense=l_dense,
                                u_dense=u_dense,
                                l_unit_diag=l_unit_diag,
                            )
                        )
                        precond_sparse = precond_build.preconditioner

                        if precond_sparse is None:
                            if emit is not None:
                                emit(
                                    1,
                                    f"{'sparse_lu' if sparse_exact_lu else 'sparse_ilu'}: "
                                    "implicit preconditioner factors unavailable; skipping",
                                )
                            res_sparse = None
                        else:
                            if emit is not None:
                                emit(
                                    0,
                                    "solve_v3_full_system_linear_gmres: "
                                    f"{'sparse LU' if sparse_exact_lu else 'sparse ILU'} (implicit) fallback",
                                )
                            res_sparse = _solve_linear(
                                matvec_fn=mv_reduced,
                                b_vec=rhs_reduced,
                                precond_fn=precond_sparse,
                                x0_vec=res_reduced.x,
                                tol_val=tol,
                                atol_val=atol,
                                restart_val=restart,
                                maxiter_val=maxiter,
                                solve_method_val="incremental",
                                precond_side=gmres_precond_side,
                            )
                    else:
                        scipy_sparse_build = build_sparse_host_scipy_preconditioner(
                            SparseHostScipyPreconditionerBuildContext(
                                ilu=ilu,
                                a_csr_full=a_csr_full,
                                base_matvec=mv_reduced,
                                sparse_use_matvec=sparse_use_matvec,
                            )
                        )
                        _precond_sparse = scipy_sparse_build.preconditioner
                        _mv_sparse = scipy_sparse_build.matvec

                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: "
                                f"{'sparse LU' if sparse_exact_lu else 'sparse ILU'} GMRES fallback",
                            )
                        res_sparse, _residual_vec_sparse_unused = run_sparse_host_scipy_gmres(
                            SparseHostScipyGMRESContext(
                                matvec=_mv_sparse,
                                rhs=rhs_reduced,
                                preconditioner=_precond_sparse,
                                x0=res_reduced.x,
                                tol=tol,
                                atol=atol,
                                restart=restart,
                                maxiter=maxiter,
                                precondition_side=gmres_precond_side,
                                gmres_solver=gmres_solve_with_history_scipy,
                            )
                        )
                    if res_sparse is not None:
                        sparse_retry_elapsed_s = sparse_retry_timer.elapsed_s()
                        res_reduced, residual_vec, _accepted = rhs1_accept_sparse_retry_candidate_and_update_replay(
                            replay_state=ksp_replay,
                            current_result=res_reduced,
                            candidate_result=res_sparse,
                            current_residual_vec=residual_vec,
                            candidate_residual_vec=None,
                            matvec_fn=mv_reduced if use_implicit else _mv_sparse,
                            b_vec=rhs_reduced,
                            precond_fn=_precond_sparse,
                            restart=restart,
                            maxiter=maxiter,
                            precond_side=gmres_precond_side,
                            solver_kind=_solver_kind("incremental")[0],
                            candidate_family="sparse",
                            scope="reduced",
                            target_value=target_reduced,
                            solve_s=sparse_retry_elapsed_s,
                            peak_rss_mb=_rss_mb(),
                        )
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(
                            1,
                            f"{'sparse_lu' if sparse_exact_lu else 'sparse_ilu'}: "
                            f"failed ({type(exc).__name__}: {exc})",
                        )
        residual_vec, residual_norm_true, residual_norm_check = (
            rhs1_replay_left_preconditioned_residual_norms(
                result=res_reduced,
                rhs=ksp_replay.b_vec,
                matvec=ksp_replay.matvec_fn,
                residual_vec=residual_vec,
                preconditioner=ksp_replay.precond_fn,
                precondition_side=ksp_replay.precond_side,
                update_residual_vec=False,
            )
        )
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        if not dense_backend_allowed and not host_dense_fallback_allowed and not dense_krylov_allowed:
            dense_fallback_max = 0
        res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        dense_thresholds = rhs1_dense_fallback_thresholds_from_env(
            dense_fallback_max=int(dense_fallback_max),
            residual_ratio=float(res_ratio),
        )
        dense_fallback_limit = int(dense_thresholds.dense_fallback_limit)
        dense_fallback_trigger = bool(dense_thresholds.dense_fallback_trigger)
        if host_sparse_direct_used and jax.default_backend() != "cpu":
            host_sparse_skip_ratio = _rhsmode1_host_sparse_skip_dense_ratio()
            if host_sparse_skip_ratio > 0.0 and res_ratio <= float(host_sparse_skip_ratio):
                dense_fallback_trigger = False
                dense_fallback_max = 0
                dense_fallback_limit = 0
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: skipping dense fallback after host sparse LU "
                        f"(ratio={res_ratio:.3e} <= {float(host_sparse_skip_ratio):.1e})",
                    )
        pas_force_dense = (
            (not disable_dense_pas)
            and op.fblock.fp is None
            and int(op.constraint_scheme) == 2
            and dense_fallback_limit > 0
            and int(active_size) <= dense_fallback_limit
            and float(res_reduced.residual_norm) > target_reduced
        )
        if pas_force_dense:
            dense_fallback_trigger = True
        fp_force_dense = (
            op.fblock.fp is not None
            and dense_fallback_max > 0
            and int(active_size) <= dense_fallback_max
            and float(residual_norm_true) > target_reduced
        )
        if fp_force_dense:
            dense_fallback_trigger = True
            dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
        force_dense_cs0 = bool(
            int(op.constraint_scheme) == 0
            and cs0_dense_fallback_allowed
            and (not cs0_sparse_first)
            and (not cs0_petsc_compat)
        )
        if force_dense_cs0:
            # constraintScheme=0 systems are singular; keep the dense fallback
            # available even when the residual ratio is huge.
            dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
            dense_fallback_trigger = True
        if int(op.constraint_scheme) == 0 and not cs0_dense_fallback_allowed:
            dense_fallback_limit = 0
            dense_fallback_trigger = False
        if (
            dense_fallback_limit > 0
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and int(active_size) <= dense_fallback_limit
            and dense_fallback_trigger
            and (float(residual_norm_true) > target_reduced or force_dense_cs0)
        ):
            _mark("rhs1_dense_fallback_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback "
                    f"(size={active_size} residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )
            try:
                res_dense, dense_retry_elapsed_s = (
                    solve_rhs1_reduced_dense_fallback_candidate(
                        context=RHS1ReducedDenseFallbackCandidateContext(
                            matvec=mv_reduced,
                            rhs=rhs_reduced,
                            x0=res_reduced.x,
                            active_size=int(active_size),
                            constraint_scheme=int(op.constraint_scheme),
                            has_fp=op.fblock.fp is not None,
                            has_pas=op.fblock.pas is not None,
                            dense_matrix_cache=dense_matrix_cache,
                            dense_backend_allowed=bool(dense_backend_allowed),
                            use_implicit=bool(use_implicit),
                            tol=float(tol),
                            atol=float(atol),
                            restart=int(restart),
                            maxiter=maxiter,
                            gmres_precond_side=gmres_precond_side,
                        ),
                        emit=emit,
                    )
                )
                res_reduced, residual_vec, _accepted = rhs1_accept_measured_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    candidate_result=res_dense,
                    current_residual_vec=residual_vec,
                    candidate_residual_vec=None,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=None,
                    x0_vec=res_dense.x,
                    restart=restart,
                    maxiter=maxiter,
                    precond_side="none",
                    solver_kind="dense",
                    candidate_name="dense_reduced",
                    baseline_name="current_reduced",
                    target_value=target_reduced,
                    solve_s=dense_retry_elapsed_s,
                    peak_rss_mb=_rss_mb(),
                )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: dense fallback failed ({type(exc).__name__}: {exc})")
            _mark("rhs1_dense_fallback_done")
        if (
            _rhsmode1_fast_post_xblock_polish_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                use_implicit=bool(use_implicit),
            )
            and preconditioner_reduced is not None
        ):
            polish_controls = rhs1_fast_post_xblock_polish_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
                tol=float(tol),
            )
            res_reduced, _accepted = rhs1_run_fast_post_xblock_polish(
                current_result=res_reduced,
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner_reduced,
                tol=polish_controls.tol,
                atol=atol,
                restart=polish_controls.restart,
                maxiter=polish_controls.maxiter,
                precond_side=gmres_precond_side,
                solve_linear=_solve_linear,
                emit=emit,
            )
        # Cheap post-solve polish for large FP systems:
        # Apply a few damped preconditioned-residual correction steps to improve
        # low-order moments (flow/Mach/jHat) without paying for a full second GMRES pass.
        if (
            int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
            and rhs1_precond_kind == "xmg"
            and preconditioner_reduced is not None
        ):
            fp_polish_controls = rhs1_fp_residual_polish_controls_from_env()
            fp_targeted_polish = _rhsmode1_fp_targeted_polish_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                rhs1_precond_kind=rhs1_precond_kind,
                use_implicit=bool(use_implicit),
            )
            polish_precond = preconditioner_reduced
            lmax_precond_for_l1: Callable[[jnp.ndarray], jnp.ndarray] | None = None
            need_hybrid_fp_precond = fp_targeted_polish or (
                fp_polish_controls.steps > 0 and int(active_size) >= fp_polish_controls.min_size
            )
            if fp_polish_controls.hybrid and need_hybrid_fp_precond:
                precond_collision = _build_rhsmode1_collision_preconditioner(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )

                def _hybrid_precond(v: jnp.ndarray) -> jnp.ndarray:
                    z0 = preconditioner_reduced(v)
                    r1 = v - mv_reduced(z0)
                    z1 = precond_collision(r1)
                    return z0 + z1

                polish_precond = _hybrid_precond

            if fp_polish_controls.steps > 0 and int(active_size) >= fp_polish_controls.min_size:
                polish_base_residual = float(res_reduced.residual_norm)
                res_polish, polish_improved = _apply_damped_preconditioned_residual_polish(
                    current_result=res_reduced,
                    rhs=rhs_reduced,
                    matvec=mv_reduced,
                    preconditioner=polish_precond,
                    target=float(target_reduced),
                    steps=int(fp_polish_controls.steps),
                    omega=float(fp_polish_controls.omega),
                    backtrack=int(fp_polish_controls.backtrack),
                )
                if polish_improved:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP polish improved residual "
                            f"{polish_base_residual:.3e} -> {float(res_polish.residual_norm):.3e}",
                        )
                    res_reduced = res_polish
            # Optional FP-specific angular/x preconditioner polish (low-L blocks).
            fp_lmax_controls = rhs1_fp_low_l_polish_controls_from_env(
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
                n_theta=int(op.n_theta),
                n_zeta=int(op.n_zeta),
            )
            if fp_targeted_polish and float(res_reduced.residual_norm) > target_reduced:
                nxi_for_x_np = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                max_l = int(np.max(nxi_for_x_np)) if nxi_for_x_np.size else 0
                lmax_use = max(0, min(int(max_l), int(fp_lmax_controls.lmax_default)))
                block_size = int(lmax_use) * int(op.n_theta) * int(op.n_zeta)
                if lmax_use > 0 and block_size > 0 and block_size <= int(fp_lmax_controls.block_max):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP low-L polish "
                            f"(lmax={int(lmax_use)} block={block_size} restart={int(fp_lmax_controls.restart)} "
                            f"maxiter={int(fp_lmax_controls.maxiter)})",
                        )
                    try:
                        lmax_precond = _build_rhsmode1_xblock_tz_lmax_preconditioner(
                            op=op,
                            lmax=int(lmax_use),
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                        lmax_precond_for_l1 = lmax_precond
                        res_lmax = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            precond_fn=lmax_precond,
                            x0_vec=res_reduced.x,
                            tol_val=tol,
                            atol_val=atol,
                            restart_val=int(fp_lmax_controls.restart),
                            maxiter_val=int(fp_lmax_controls.maxiter),
                            solve_method_val="incremental",
                            precond_side=gmres_precond_side,
                        )
                        if float(res_lmax.residual_norm) < float(res_reduced.residual_norm):
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP low-L polish improved residual "
                                    f"{float(res_reduced.residual_norm):.3e} -> {float(res_lmax.residual_norm):.3e}",
                                )
                            res_reduced = res_lmax
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(1, f"solve_v3_full_system_linear_gmres: FP low-L polish failed ({type(exc).__name__}: {exc})")
            # L=1 targeted polish (flow channel): solve a small projected system
            # on the L=1 active modes only. This is a cheap way to improve flow/Mach
            # parity when the full-system solve stalls above the strict target.
            l1_polish_controls = rhs1_fp_l1_polish_controls_from_env()
            if fp_targeted_polish and l1_polish_controls.enabled:
                nxi_for_x_np = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                l1_active_idx_np = fp_pitch_mode_active_indices(
                    n_species=int(op.n_species),
                    n_x=int(op.n_x),
                    n_xi=int(op.n_xi),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    nxi_for_x=nxi_for_x_np,
                    l_min=1,
                    l_max=1,
                    full_to_active=(
                        full_to_active_jnp
                        if use_active_dof_mode and full_to_active_jnp is not None
                        else None
                    ),
                )

                if int(l1_active_idx_np.size) > 0:
                    l1_idx_jnp = jnp.asarray(np.unique(l1_active_idx_np), dtype=jnp.int32)
                    l1_n = int(l1_idx_jnp.shape[0])

                    def _pre_l1_full(v: jnp.ndarray) -> jnp.ndarray:
                        if lmax_precond_for_l1 is not None:
                            return lmax_precond_for_l1(v)
                        return polish_precond(v)

                    try:
                        l1_outcome = _apply_projected_residual_polish(
                            current_result=res_reduced,
                            rhs=rhs_reduced,
                            matvec=mv_reduced,
                            projected_indices=l1_idx_jnp,
                            active_size=int(active_size),
                            solve_linear=_solve_linear,
                            preconditioner=_pre_l1_full,
                            tol=l1_polish_controls.tol,
                            restart=int(l1_polish_controls.restart),
                            maxiter=int(l1_polish_controls.maxiter),
                            precond_side=gmres_precond_side,
                            target=float(target_reduced),
                            threshold_ratio=float(l1_polish_controls.ratio),
                            abs_threshold=float(l1_polish_controls.abs_threshold),
                            full_accept_ratio=float(l1_polish_controls.full_accept_ratio),
                            require_full_improvement=False,
                        )
                        if (
                            emit is not None
                            and np.isfinite(l1_outcome.projected_residual_before)
                            and l1_outcome.projected_residual_before
                            > max(
                                float(target_reduced) * max(1.0, float(l1_polish_controls.ratio)),
                                float(l1_polish_controls.abs_threshold),
                            )
                        ):
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: FP L1 polish "
                                f"(size={l1_n} restart={l1_polish_controls.restart} "
                                f"maxiter={l1_polish_controls.maxiter} "
                                f"b_norm={l1_outcome.projected_residual_before:.3e})",
                            )
                        if l1_outcome.accepted:
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP L1 polish improved residual "
                                    f"full {l1_outcome.full_residual_before:.3e} -> "
                                    f"{float(l1_outcome.full_residual_after):.3e}; "
                                    f"L1 {l1_outcome.projected_residual_before:.3e} -> "
                                    f"{float(l1_outcome.projected_residual_after):.3e}",
                                )
                            res_reduced = l1_outcome.result
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(1, f"solve_v3_full_system_linear_gmres: FP L1 polish failed ({type(exc).__name__}: {exc})")
            # Low-L global polish: solve a projected system on the lowest L modes across all
            # species/x/(theta,zeta). This is more expensive than the L1 polish but can be
            # significantly more effective at improving flow/current parity when the full
            # solve stalls above the strict target (especially for FP+Er).
            global_low_l_controls = rhs1_fp_global_low_l_polish_controls_from_env(
                n_xi=int(op.n_xi),
            )
            if (
                fp_targeted_polish
                and global_low_l_controls.enabled
                and float(res_reduced.residual_norm) > target_reduced
            ):
                if (
                    float(res_reduced.residual_norm)
                    > float(target_reduced) * float(global_low_l_controls.ratio)
                    and global_low_l_controls.lmax > 0
                ):
                    nxi_for_x_np = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                    low_active_idx_np = fp_pitch_mode_active_indices(
                        n_species=int(op.n_species),
                        n_x=int(op.n_x),
                        n_xi=int(op.n_xi),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        nxi_for_x=nxi_for_x_np,
                        l_min=0,
                        l_max=int(global_low_l_controls.lmax),
                        full_to_active=(
                            full_to_active_jnp
                            if use_active_dof_mode and full_to_active_jnp is not None
                            else None
                        ),
                    )

                    if int(low_active_idx_np.size) > 0:
                        low_idx_jnp = jnp.asarray(np.unique(low_active_idx_np), dtype=jnp.int32)
                        low_n = int(low_idx_jnp.shape[0])
                    else:
                        low_n = 0
                        low_idx_jnp = None

                    if (
                        low_n > 0
                        and (
                            global_low_l_controls.max_size <= 0
                            or low_n <= global_low_l_controls.max_size
                        )
                    ):
                        assert low_idx_jnp is not None

                        def _pre_low_full(v: jnp.ndarray) -> jnp.ndarray:
                            if lmax_precond_for_l1 is not None:
                                return lmax_precond_for_l1(v)
                            return polish_precond(v)

                        try:
                            low_outcome = _apply_projected_residual_polish(
                                current_result=res_reduced,
                                rhs=rhs_reduced,
                                matvec=mv_reduced,
                                projected_indices=low_idx_jnp,
                                active_size=int(active_size),
                                solve_linear=_solve_linear,
                                preconditioner=_pre_low_full,
                                tol=float(global_low_l_controls.tol),
                                restart=int(global_low_l_controls.restart),
                                maxiter=int(global_low_l_controls.maxiter),
                                precond_side=gmres_precond_side,
                                target=float(target_reduced),
                                threshold_ratio=float(global_low_l_controls.threshold_ratio),
                                abs_threshold=float(global_low_l_controls.abs_threshold),
                                full_accept_ratio=float(global_low_l_controls.full_accept_ratio),
                                require_full_improvement=True,
                            )
                            if (
                                emit is not None
                                and np.isfinite(low_outcome.projected_residual_before)
                                and low_outcome.projected_residual_before
                                > max(
                                    float(target_reduced)
                                    * float(global_low_l_controls.threshold_ratio),
                                    float(global_low_l_controls.abs_threshold),
                                )
                            ):
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP global low-L polish "
                                    f"(lmax={int(global_low_l_controls.lmax)} size={int(low_n)} "
                                    f"restart={int(global_low_l_controls.restart)} "
                                    f"maxiter={int(global_low_l_controls.maxiter)} "
                                    f"b_norm={low_outcome.projected_residual_before:.3e})",
                                )
                            if low_outcome.accepted:
                                if emit is not None:
                                    emit(
                                        1,
                                        "solve_v3_full_system_linear_gmres: FP global low-L polish improved residual "
                                        f"full {low_outcome.full_residual_before:.3e} -> "
                                        f"{float(low_outcome.full_residual_after):.3e}; "
                                        f"low-L {low_outcome.projected_residual_before:.3e} -> "
                                        f"{float(low_outcome.projected_residual_after):.3e}",
                                    )
                                res_reduced = low_outcome.result
                        except Exception as exc:  # noqa: BLE001
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: FP global low-L polish failed "
                                    f"({type(exc).__name__}: {exc})",
                                )
                # Optional BiCGStab polish for large FP systems: short-recurrence Krylov
                # can reduce residuals further when restarted GMRES stagnates.
                fp_bi_controls = rhs1_fp_bicgstab_polish_controls_from_env(
                    tol=float(tol),
                    atol=float(atol),
                )
                if (
                    fp_bi_controls.enabled
                    and int(active_size) >= int(fp_bi_controls.min_size)
                    and float(res_reduced.residual_norm) > target_reduced
                ):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: FP BiCGStab polish "
                            f"(maxiter={fp_bi_controls.maxiter} tol={fp_bi_controls.tol:.1e})",
                        )
                    precond_bi = preconditioner_reduced
                    if precond_bi is None and fp_polish_controls.hybrid:
                        precond_bi = polish_precond
                    res_bi = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=precond_bi,
                        x0_vec=res_reduced.x,
                        tol_val=fp_bi_controls.tol,
                        atol_val=fp_bi_controls.atol,
                        restart_val=restart,
                        maxiter_val=fp_bi_controls.maxiter,
                        solve_method_val="bicgstab",
                        precond_side=gmres_precond_side,
                    )
                    if float(res_bi.residual_norm) < float(res_reduced.residual_norm):
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: FP BiCGStab polish improved residual "
                                f"{float(res_reduced.residual_norm):.3e} -> {float(res_bi.residual_norm):.3e}",
                            )
                        res_reduced = res_bi
        if not bool(sparse_enabled):
            sparse_xblock_rescue_reason = "sparse_disabled"
        elif float(res_reduced.residual_norm) <= float(target_reduced) and not bool(sparse_xblock_rescue_attempted):
            sparse_xblock_rescue_reason = "not_needed"
        rhsmode1_general_metadata.update(
            sparse_rescue_tail_metadata_from_context(
                SparseRescueTailMetadataContext(
                    sparse_xblock_rescue_active=sparse_xblock_rescue_active,
                    sparse_xblock_rescue_attempted=sparse_xblock_rescue_attempted,
                    sparse_xblock_rescue_built=sparse_xblock_rescue_built,
                    sparse_xblock_rescue_error=sparse_xblock_rescue_error,
                    sparse_xblock_rescue_reason=sparse_xblock_rescue_reason,
                    sparse_xblock_rescue_assembled_host_fp=sparse_xblock_rescue_assembled_host_fp,
                    sparse_xblock_rescue_preconditioner_xi=sparse_xblock_rescue_preconditioner_xi,
                    sparse_xblock_rescue_seed_residual=sparse_xblock_rescue_seed_residual,
                    sparse_xblock_rescue_seed_improvement_ratio=sparse_xblock_rescue_seed_improvement_ratio,
                    sparse_xblock_rescue_seed_accept_ratio=sparse_xblock_rescue_seed_accept_ratio,
                    sparse_xblock_rescue_seed_refine_steps=sparse_xblock_rescue_seed_refine_steps,
                    sparse_xblock_rescue_seed_refines_performed=sparse_xblock_rescue_seed_refines_performed,
                    sparse_xblock_rescue_candidate_residual=sparse_xblock_rescue_candidate_residual,
                    sparse_xblock_rescue_candidate_accepted=sparse_xblock_rescue_candidate_accepted,
                    fp_xblock_global_correction_allowed=fp_xblock_global_correction_allowed,
                    fp_xblock_global_correction_attempted=fp_xblock_global_correction_attempted,
                    fp_xblock_global_correction_accepted=fp_xblock_global_correction_accepted,
                    fp_xblock_global_correction_reason=fp_xblock_global_correction_reason,
                    fp_xblock_global_correction_error=fp_xblock_global_correction_error,
                    fp_xblock_global_correction_preconditioner=fp_xblock_global_correction_preconditioner,
                    fp_xblock_global_correction_steps=fp_xblock_global_correction_steps,
                    fp_xblock_global_correction_accepted_steps=fp_xblock_global_correction_accepted_steps,
                    fp_xblock_global_correction_residual_before=fp_xblock_global_correction_residual_before,
                    fp_xblock_global_correction_residual_after=fp_xblock_global_correction_residual_after,
                    fp_xblock_global_correction_improvement_ratio=fp_xblock_global_correction_improvement_ratio,
                    fp_xblock_global_correction_elapsed_s=fp_xblock_global_correction_elapsed_s,
                    fp_xblock_highx_residual_correction_allowed=fp_xblock_highx_residual_correction_allowed,
                    fp_xblock_highx_residual_correction_attempted=fp_xblock_highx_residual_correction_attempted,
                    fp_xblock_highx_residual_correction_accepted=fp_xblock_highx_residual_correction_accepted,
                    fp_xblock_highx_residual_correction_reason=fp_xblock_highx_residual_correction_reason,
                    fp_xblock_highx_residual_correction_error=fp_xblock_highx_residual_correction_error,
                    fp_xblock_highx_residual_correction_residual_before=fp_xblock_highx_residual_correction_residual_before,
                    fp_xblock_highx_residual_correction_residual_after=fp_xblock_highx_residual_correction_residual_after,
                    fp_xblock_highx_residual_correction_improvement_ratio=fp_xblock_highx_residual_correction_improvement_ratio,
                    fp_xblock_highx_residual_correction_elapsed_s=fp_xblock_highx_residual_correction_elapsed_s,
                    fp_xblock_highx_residual_correction_direction_count=fp_xblock_highx_residual_correction_direction_count,
                    fp_xblock_highx_residual_correction_direction_names=fp_xblock_highx_residual_correction_direction_names,
                )
            )
        )
        # As a last resort on CPU, retry the reduced solve using SciPy GMRES.
        #
        # JAX's gmres can stagnate or return a poor solution for some FP operators on CPU,
        # even after our strong-preconditioner fallbacks, while SciPy GMRES is often more
        # robust for the same matvec/preconditioner pair. This path is explicitly
        # non-differentiable and intended for CLI robustness.
        if (
            (not bool(use_implicit))
            and jax.default_backend() == "cpu"
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and float(res_reduced.residual_norm) > float(target_reduced)
            and (not skip_global_sparse_after_xblock)
        ):
            scipy_rescue_controls = rhs1_scipy_rescue_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
            )
            if scipy_rescue_controls.enabled:
                rescue_abs_floor = _rhsmode1_scipy_rescue_abs_floor_after_xblock(
                    op=op,
                    active_size=int(active_size),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    use_implicit=bool(use_implicit),
                )
                rescue_threshold = max(
                    float(target_reduced) * float(scipy_rescue_controls.ratio),
                    float(rescue_abs_floor),
                )
                scipy_rescue_size_allowed = _rhsmode1_scipy_rescue_active_size_allowed(
                    op=op,
                    active_size=int(active_size),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    use_implicit=bool(use_implicit),
                )
                if (not scipy_rescue_size_allowed) and float(res_reduced.residual_norm) > float(rescue_threshold):
                    _mark("rhs1_scipy_rescue_skipped")
                    rhsmode1_general_metadata.update(
                        {
                            "scipy_rescue_attempted": False,
                            "scipy_rescue_skipped": True,
                            "scipy_rescue_skip_reason": "active_size_cap",
                            "scipy_rescue_initial_residual": float(res_reduced.residual_norm),
                            "scipy_rescue_target": float(target_reduced),
                            "scipy_rescue_threshold": float(rescue_threshold),
                            "scipy_rescue_active_size": int(active_size),
                            "scipy_rescue_used_large_cpu_xblock_shortcut": bool(cpu_large_xblock_shortcut),
                            "scipy_rescue_used_explicit_fp_xblock_seed": bool(explicit_fp_xblock_seed_used),
                        }
                    )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: skipping SciPy rescue "
                            f"(active_size={int(active_size)} exceeds default rescue cap; "
                            "set SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE=0 to force)",
                    )
                elif float(res_reduced.residual_norm) > float(rescue_threshold):
                    rescue_preconditioner = preconditioner_reduced
                    rescue_precond_name = rhs1_precond_kind or "none"
                    if scipy_rescue_controls.use_strong and strong_preconditioner_reduced is not None:
                        rescue_preconditioner = strong_preconditioner_reduced
                        rescue_precond_name = strong_precond_kind or "strong"
                    rescue_method = scipy_rescue_controls.method
                    if rescue_method == "auto":
                        rescue_method = "bicgstab" if rescue_preconditioner is strong_preconditioner_reduced else "gmres"
                    try:
                        scipy_rescue_start_s = float(t.elapsed_s())
                        scipy_rescue_initial_residual = float(res_reduced.residual_norm)
                        _mark("rhs1_scipy_rescue_start")
                        rhsmode1_general_metadata.update(
                            {
                                "scipy_rescue_attempted": True,
                                "scipy_rescue_method": str(rescue_method),
                                "scipy_rescue_preconditioner": str(rescue_precond_name),
                                "scipy_rescue_restart": int(scipy_rescue_controls.restart),
                                "scipy_rescue_maxiter": int(scipy_rescue_controls.maxiter),
                                "scipy_rescue_initial_residual": float(scipy_rescue_initial_residual),
                                "scipy_rescue_target": float(target_reduced),
                                "scipy_rescue_threshold": float(rescue_threshold),
                                "scipy_rescue_start_s": float(scipy_rescue_start_s),
                            }
                        )
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: SciPy rescue "
                                f"(residual={float(res_reduced.residual_norm):.3e} > "
                                f"{float(scipy_rescue_controls.ratio):.1e}x target={float(target_reduced):.3e} "
                                f"method={rescue_method} restart={int(scipy_rescue_controls.restart)} "
                                f"maxiter={int(scipy_rescue_controls.maxiter)} "
                                f"preconditioner={rescue_precond_name})",
                            )
                        scipy_outcome = run_rhs1_scipy_rescue(
                            context=RHS1ScipyRescueContext(
                                matvec=mv_reduced,
                                rhs=rhs_reduced,
                                x0=res_reduced.x,
                                preconditioner=rescue_preconditioner,
                                method=rescue_method,
                                tol=float(tol),
                                atol=float(atol),
                                restart=int(scipy_rescue_controls.restart),
                                maxiter=int(scipy_rescue_controls.maxiter),
                                precond_side=gmres_precond_side,
                            ),
                            emit=emit,
                        )
                        res_scipy = scipy_outcome.result
                        r_scipy = scipy_outcome.residual_vec
                        scipy_rescue_elapsed_s = float(t.elapsed_s() - scipy_rescue_start_s)
                        scipy_rescue_final_residual = float(res_scipy.residual_norm)
                        _mark("rhs1_scipy_rescue_done")
                        rhsmode1_general_metadata.update(
                            {
                                "scipy_rescue_elapsed_s": float(scipy_rescue_elapsed_s),
                                "scipy_rescue_final_residual": float(scipy_rescue_final_residual),
                                "scipy_rescue_reported_residual": float(scipy_outcome.reported_residual),
                                "scipy_rescue_history_len": int(scipy_outcome.history_len),
                                "scipy_rescue_improved": bool(
                                    scipy_rescue_final_residual < scipy_rescue_initial_residual
                                ),
                            }
                        )
                        if float(res_scipy.residual_norm) < float(res_reduced.residual_norm):
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: SciPy rescue improved residual "
                                    f"{float(res_reduced.residual_norm):.3e} -> {float(res_scipy.residual_norm):.3e}",
                                )
                            res_reduced = res_scipy
                    except Exception as exc:  # noqa: BLE001
                        _mark("rhs1_scipy_rescue_failed")
                        rhsmode1_general_metadata.update(
                            {
                                "scipy_rescue_failed": True,
                                "scipy_rescue_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if emit is not None:
                            emit(1, f"solve_v3_full_system_linear_gmres: SciPy rescue failed ({type(exc).__name__}: {exc})")
                elif emit is not None and float(rescue_abs_floor) > 0.0:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: skipping SciPy rescue after x-block seed "
                        f"(residual={float(res_reduced.residual_norm):.3e} <= floor={float(rescue_abs_floor):.1e})",
                    )
        elif (
            (not bool(use_implicit))
            and jax.default_backend() == "cpu"
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and float(res_reduced.residual_norm) > float(target_reduced)
            and skip_global_sparse_after_xblock
            and emit is not None
        ):
            emit(
                1,
                "solve_v3_full_system_linear_gmres: skipping SciPy rescue after bounded x-block seed "
                f"(residual={float(res_reduced.residual_norm):.3e}; not accepted as converged)",
            )
        if use_pas_projection:
            f_full = _expand_active_f(res_reduced.x)
            f_full = _project_pas_f(f_full)
            if int(op.extra_size) > 0:
                zeros_extra = jnp.zeros((int(op.extra_size),), dtype=jnp.float64)
                y_full = mv(jnp.concatenate([f_full, zeros_extra], axis=0))
                r_f = rhs[: op.f_size] - y_full[: op.f_size]
                extra = _constraint_scheme2_source_from_f(op, r_f.reshape(op.fblock.f_shape)) / fs_sum_safe
                if ix0 > 0:
                    extra = extra.at[:, :ix0].set(0.0)
                zero_tol = rhs1_pas_source_zero_tolerance_from_env()
                if zero_tol > 0.0:
                    max_abs = jnp.max(jnp.abs(extra))
                    extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
                x_full = jnp.concatenate([f_full, extra.reshape((-1,))], axis=0)
            else:
                x_full = f_full
        else:
            x_full = expand_reduced(res_reduced.x)
        # Residuals in active-DOF mode are computed on the reduced system to avoid an
        # extra full matvec; this matches the reduced KSP system used upstream.
        residual_norm_full = res_reduced.residual_norm
        result = GMRESSolveResult(x=x_full, residual_norm=residual_norm_full)
    else:
        if solve_method_kind == "dense_ksp":
            if int(op.phi1_size) != 0:
                raise NotImplementedError("dense_ksp is only supported for includePhi1=false RHSMode=1 solves.")
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: assembling dense full matrix for dense_ksp")
            a_dense = assemble_dense_matrix_from_matvec(matvec=mv, n=int(op.total_size), dtype=rhs.dtype)

            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building PETSc-like species-block preconditioner (dense_ksp)")

            import jax.scipy.linalg as jla  # noqa: PLC0415

            n_species = int(op.n_species)
            n_theta = int(op.n_theta)
            n_zeta = int(op.n_zeta)
            local_per_species = int(np.sum(nxi_for_x))
            dke_size = int(local_per_species * n_theta * n_zeta)
            extra_size = int(op.extra_size)
            extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
            if extra_size and (extra_per_species * n_species != extra_size):
                extra_per_species = 0

            f_size = int(n_species * dke_size)
            expected_size = int(f_size + int(op.phi1_size) + extra_size)
            if int(op.total_size) != expected_size:
                raise RuntimeError(f"dense_ksp expects total_size={expected_size}, got {int(op.total_size)}")

            lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
            idx_blocks: list[jnp.ndarray] = []
            for s in range(n_species):
                f_idx = np.arange(s * dke_size, (s + 1) * dke_size, dtype=np.int32)
                extra_idx = np.arange(
                    f_size + s * extra_per_species,
                    f_size + (s + 1) * extra_per_species,
                    dtype=np.int32,
                )
                block_idx_np = np.concatenate([f_idx, extra_idx], axis=0) if extra_per_species else f_idx
                block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
                a_block = a_dense[jnp.ix_(block_idx, block_idx)]
                lu, piv = jla.lu_factor(a_block)
                lu_factors.append((lu, piv))
                idx_blocks.append(block_idx)

            def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
                out = jnp.zeros_like(v)
                for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
                    rhs_block = v[block_idx]
                    sol_block = jla.lu_solve((lu, piv), rhs_block)
                    out = out.at[block_idx].set(sol_block, unique_indices=True)
                return out

            def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                return a_dense @ x

            rhs_pc = preconditioner_dense(rhs)

            def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
                return preconditioner_dense(mv_dense(x))

            res_pc = _solve_linear(
                matvec_fn=mv_pc,
                b_vec=rhs_pc,
                precond_fn=None,
                x0_vec=x0,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side="none",
            )
            ksp_replay.matvec_fn = mv_pc
            ksp_replay.b_vec = rhs_pc
            ksp_replay.precond_fn = None
            ksp_replay.x0_vec = x0
            ksp_replay.precond_side = "none"
            ksp_replay.solver_kind = _solver_kind("incremental")[0]
            residual_norm_full = jnp.linalg.norm(mv(res_pc.x) - rhs)
            result = GMRESSolveResult(x=res_pc.x, residual_norm=residual_norm_full)
        else:
            preconditioner_full = None
            bicgstab_preconditioner_full = None
            host_dense_shortcut_full = _rhsmode1_host_dense_shortcut_allowed(
                op=op,
                active_size=int(op.total_size),
                use_implicit=bool(use_implicit),
                solve_method_kind=str(solve_method_kind),
            )

            def _solve_host_dense_full(*, x0_dense: jnp.ndarray | None = None) -> tuple[GMRESSolveResult, jnp.ndarray]:
                return solve_host_dense_full(
                    context=HostDenseFullSolveContext(
                        matvec=mv,
                        rhs=rhs,
                        total_size=int(op.total_size),
                    ),
                    x0=x0_dense,
                )

            if rhs1_bicgstab_kind is not None:
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner={rhs1_bicgstab_kind}")
                if rhs1_bicgstab_kind == "collision":
                    bicgstab_preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)

            def _build_rhs1_preconditioner_full():
                return build_rhs1_full_preconditioner(
                    context=RHS1FullPreconditionerBuildContext(
                        op=op,
                        emit=emit,
                        mark=_mark,
                        progress_preconditioner_build=rhs1_progress_notes.preconditioner_build,
                        record_structured_metadata=_record_structured_fblock_preconditioner_metadata,
                        dd_setup=rhs1_dd_setup,
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        build_from_kind=_build_rhs1_preconditioner_from_kind,
                    ),
                    rhs1_precond_kind=rhs1_precond_kind,
                    rhs1_xblock_tz_lmax=rhs1_xblock_tz_lmax,
                )

            if rhs1_precond_enabled and (not host_dense_shortcut_full):
                solver_kind = _solver_kind(solve_method)[0]
                build_rhs1 = (
                    (solver_kind != "bicgstab" and solve_method_kind != "dense")
                    or (rhs1_bicgstab_kind == "rhs1" and solve_method_kind != "dense")
                )
                if build_rhs1:
                    preconditioner_full = _build_rhs1_preconditioner_full()
                    if rhs1_bicgstab_kind == "rhs1":
                        bicgstab_preconditioner_full = preconditioner_full
            if (not host_dense_shortcut_full) and preconditioner_full is None and bicgstab_preconditioner_full is not None:
                preconditioner_full = bicgstab_preconditioner_full
            if (not host_dense_shortcut_full) and preconditioner_full is not None and rhs1_precond_kind in {
                "pas_hybrid",
                "pas_lite",
                "pas_tz",
                "pas_schur",
                "pas_tokamak_theta",
                "pas_ilu",
            }:
                try:
                    probe = preconditioner_full(rhs)
                    probe_ok = bool(jnp.all(jnp.isfinite(probe)))
                except Exception as exc:  # noqa: BLE001
                    probe_ok = False
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS precond probe failed "
                            f"({type(exc).__name__}: {exc}), using collision preconditioner",
                        )
                if not probe_ok:
                    preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)
                    if rhs1_bicgstab_kind == "rhs1":
                        bicgstab_preconditioner_full = preconditioner_full
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS precond non-finite -> collision",
                        )
            if host_dense_shortcut_full:
                _mark("rhs1_host_dense_shortcut_start")
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: accelerator FP small system -> "
                        f"using host dense shortcut (size={int(op.total_size)})",
                    )
                result, residual_vec = _solve_host_dense_full(x0_dense=x0)
                _mark("rhs1_host_dense_shortcut_done")
                ksp_replay.matvec_fn = mv
                ksp_replay.b_vec = rhs
                ksp_replay.precond_fn = None
                ksp_replay.x0_vec = x0
                ksp_replay.precond_side = "none"
                ksp_replay.solver_kind = _solver_kind("incremental")[0]
            if recycle_basis_use and (not host_dense_shortcut_full):
                basis_full: list[jnp.ndarray] = []
                for vec in recycle_basis_use:
                    if vec.shape == (op.total_size,):
                        basis_full.append(vec)
                if basis_full:
                    basis_au = [mv(v) for v in basis_full]
                    x0_recycled = _recycled_initial_guess(rhs, basis_full, basis_au)
                    if x0_recycled is not None:
                        if x0 is None:
                            x0 = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv(jnp.asarray(x0)) - rhs)
                            r1 = jnp.linalg.norm(mv(x0_recycled) - rhs)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0 = x0_recycled
            if not host_dense_shortcut_full:
                rhs1_progress_notes.krylov_start()
                _mark("rhs1_krylov_solve_start")
                result, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    x0_vec=x0,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val=solve_method,
                    precond_side=gmres_precond_side,
                )
                result = _block_gmres_result_ready(result)
                try:
                    jax.block_until_ready(residual_vec)
                except Exception:
                    pass
                _mark("rhs1_krylov_solve_done")
                ksp_replay.matvec_fn = mv
                ksp_replay.b_vec = rhs
                ksp_replay.precond_fn = preconditioner_full
                ksp_replay.x0_vec = x0
                ksp_replay.precond_side = gmres_precond_side
                ksp_replay.solver_kind = _solver_kind(solve_method)[0]
            if (not host_dense_shortcut_full) and preconditioner_full is not None and (not _gmres_result_is_finite(result)):
                if emit is not None:
                    emit(0, "solve_v3_full_system_linear_gmres: preconditioned GMRES returned non-finite result; retrying without preconditioner")
                _mark("rhs1_krylov_solve_retry_start")
                result, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=None,
                    x0_vec=x0,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val=solve_method,
                    precond_side=gmres_precond_side,
                )
                result = _block_gmres_result_ready(result)
                try:
                    jax.block_until_ready(residual_vec)
                except Exception:
                    pass
                _mark("rhs1_krylov_solve_retry_done")
                ksp_replay.matvec_fn = mv
                ksp_replay.b_vec = rhs
                ksp_replay.precond_fn = None
                ksp_replay.x0_vec = x0
                ksp_replay.precond_side = gmres_precond_side
                ksp_replay.solver_kind = _solver_kind(solve_method)[0]
            # If GMRES does not reach the requested tolerance (common without preconditioning),
            # retry with a larger iteration budget and the more robust incremental mode.
            target = max(float(atol), float(tol) * float(rhs_norm))
            res_ratio = float(result.residual_norm) / max(float(target), 1e-300)
            stage2_trigger = rhs1_stage2_trigger(
                res_ratio=res_ratio,
                use_dkes=bool(op.fblock.pas is not None and use_dkes),
            )
            solver_kind = _solver_kind(solve_method)[0]
            bicgstab_fallback_target = float(target)
            if bicgstab_fallback_strict:
                bicgstab_fallback_target = rhs1_bicgstab_fallback_target_from_env(
                    target=float(target),
                    distributed_axis=distributed_axis,
                    has_pas=op.fblock.pas is not None,
                    include_phi1=bool(op.include_phi1),
                )
            if solver_kind == "bicgstab" and (
                (not _gmres_result_is_finite(result))
                or (bicgstab_fallback_strict and float(result.residual_norm) > bicgstab_fallback_target)
            ):
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: BiCGStab fallback to GMRES "
                        f"(residual={float(result.residual_norm):.3e} > target={bicgstab_fallback_target:.3e})",
                    )
                if preconditioner_full is None and rhs1_precond_enabled:
                    preconditioner_full = _build_rhs1_preconditioner_full()
                result, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    x0_vec=x0,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val="incremental",
                    precond_side=gmres_precond_side,
                )
                ksp_replay.matvec_fn = mv
                ksp_replay.b_vec = rhs
                ksp_replay.precond_fn = preconditioner_full
                ksp_replay.x0_vec = x0
                ksp_replay.precond_side = gmres_precond_side
                ksp_replay.solver_kind = "gmres"
        # The full-size RHSMode=1 branch does not have the later active-DOF sparse
        # ILU rescue. On accelerators, skipping stage2 GMRES here can therefore
        # return a high-residual solution with no real recovery path.
        prefer_sparse_accel = False
        if prefer_sparse_accel and float(result.residual_norm) > target and stage2_trigger and emit is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: skipping stage2 GMRES on accelerator "
                "backend; prefer sparse fallback",
            )
        if (
            float(result.residual_norm) > target
            and stage2_enabled
            and stage2_trigger
            and (not prefer_sparse_accel)
            and t.elapsed_s() < stage2_time_cap_s
        ):
            if preconditioner_full is None and rhs1_precond_enabled:
                preconditioner_full = _build_rhs1_preconditioner_full()
            stage2_controls = rhs1_stage2_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
                active_size=int(op.total_size),
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
            )
            stage2_restart = int(stage2_controls.restart)
            stage2_maxiter = int(stage2_controls.maxiter)
            stage2_method = str(stage2_controls.method)
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: stage2 GMRES "
                    f"(residual={float(result.residual_norm):.3e} > target={target:.3e}) "
                    f"restart={stage2_restart} maxiter={stage2_maxiter} method={stage2_method}",
                )
            result, residual_vec, _accepted, _stage2_elapsed_s = (
                rhs1_run_measured_linear_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(stage2_restart),
                    maxiter=int(stage2_maxiter),
                    solve_method=stage2_method,
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear_with_residual,
                    solver_kind=_solver_kind(stage2_method)[0],
                    candidate_name=f"stage2_full:{stage2_method}",
                    baseline_name="current_full",
                    target_value=float(target),
                    peak_rss_mb=_rss_mb(),
                    returns_residual_vec=True,
                )
            )
        # Krylov solvers with left preconditioning report the preconditioned residual
        # norm. Recompute the true residual before deciding whether to escalate to a
        # stronger preconditioner or dense fallback so those decisions track the
        # printed residual and H5 parity behavior.
        result, residual_vec, _residual_norm_true = rhs1_recompute_true_residual_result(
            result=result,
            rhs=rhs,
            matvec=mv,
            residual_vec=residual_vec,
            update_residual_vec=True,
        )
        pas_fast_accept = _rhsmode1_pas_fast_accept(
            op=op,
            active_size=int(op.total_size),
            residual_norm=float(result.residual_norm),
            target=float(target),
            use_implicit=bool(use_implicit),
        )
        strong_trigger_controls = rhs1_strong_trigger_controls_from_env(
            residual_norm=float(result.residual_norm),
            target=float(target),
            has_fp=op.fblock.fp is not None,
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            delay_pas_base_retries=False,
        )
        res_ratio = float(strong_trigger_controls.res_ratio)
        strong_precond_trigger = bool(strong_trigger_controls.trigger)
        if (
            rhs1_precond_kind in {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
            and preconditioner_full is not None
            and _rhsmode1_pas_adaptive_smoother_allowed(
                op=op,
                active_size=int(op.total_size),
                residual_norm=float(result.residual_norm),
                target=float(target),
                use_implicit=bool(use_implicit),
            )
        ):
            smoother_controls = rhs1_pas_adaptive_smoother_controls_from_env()
            smoother = adaptive_pas_smoother(
                matvec=mv,
                rhs=rhs,
                preconditioner=preconditioner_full,
                x0=result.x,
                target=float(target),
                omega=float(smoother_controls.omega),
                max_sweeps=int(smoother_controls.max_sweeps),
            )
            result, residual_vec, _accepted = (
                rhs1_accept_smoother_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    smoother=smoother,
                    result_factory=lambda *, x, residual_norm: GMRESSolveResult(
                        x=x,
                        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
                    ),
                    candidate_residual_vec=lambda candidate: rhs - mv(candidate.x),
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    restart=restart,
                    maxiter=maxiter,
                    precond_side=gmres_precond_side,
                    solver_kind=_solver_kind("incremental")[0],
                )
            )
        if rhs1_collision_retry_allowed(
            residual_norm=float(result.residual_norm),
            target=float(target),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            rhs1_precond_kind=rhs1_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            strong_precond_trigger=bool(strong_precond_trigger),
        ):
            if bicgstab_preconditioner_full is None:
                bicgstab_preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)
            if bicgstab_preconditioner_full is not None:
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: retry with collision preconditioner "
                        f"(residual={float(result.residual_norm):.3e} > target={target:.3e})",
                )
                result, residual_vec, _accepted, _collision_elapsed_s = (
                    rhs1_run_linear_candidate_and_update_replay(
                        replay_state=ksp_replay,
                        current_result=result,
                        current_residual_vec=residual_vec,
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=bicgstab_preconditioner_full,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        solve_method="incremental",
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear_with_residual,
                        solver_kind=_solver_kind("incremental")[0],
                        returns_residual_vec=True,
                    )
                )
        strong_precond_env = rhs1_strong_preconditioner_env_from_env()
        cs0_sparse_first = _rhsmode1_constraint0_sparse_first(
            op=op,
            solve_method_kind=solve_method_kind,
            sparse_precond_mode=sparse_precond_mode,
            active_size=int(active_size),
            sparse_max_size=int(sparse_max_size),
        )
        strong_control = rhs1_resolved_strong_preconditioner_control(
            strong_precond_env=strong_precond_env,
            has_extra_constraint_block=int(op.constraint_scheme) == 2 and int(op.extra_size) > 0,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            size=int(op.total_size),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            cs0_sparse_first=bool(cs0_sparse_first),
            pas_auto_skip=_pas_auto_skip_strong_retry(
                has_pas=op.fblock.pas is not None,
                strong_precond_env=strong_precond_env,
                rhs1_precond_kind=rhs1_precond_kind,
                residual_norm=float(result.residual_norm),
                target=float(target),
                ratio=float(pas_auto_strong_ratio),
            ),
            pas_fast_accept=bool(pas_fast_accept),
        )
        strong_precond_min = int(strong_control.min_size)
        strong_precond_disabled = bool(strong_control.disabled)
        strong_precond_auto = bool(strong_control.auto)
        if strong_control.reason_cs0_sparse_first and emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: constraintScheme=0 sparse-first "
                "auto mode -> defer strong preconditioner until after sparse ILU",
            )
        if strong_control.reason_pas_auto_skip and emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: PAS auto strong preconditioner skipped "
                f"after base={rhs1_precond_kind} "
                f"(residual={float(result.residual_norm):.3e} <= {float(pas_auto_strong_ratio):.1f}x target)",
            )
        if strong_control.reason_pas_fast_accept and emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: PAS fast-accept "
                f"(residual={float(result.residual_norm):.3e}) -> skip strong preconditioner tail",
            )
        strong_precond_kind: str | None = None
        strong_xblock_tz_lmax: int | None = None
        if strong_precond_disabled:
            strong_precond_kind = None
        else:
            strong_precond_kind = requested_rhs1_strong_preconditioner_kind(
                strong_precond_env,
                mode="full",
            )

        if strong_precond_kind is None and (not strong_precond_disabled) and strong_precond_auto:
            if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                if op.fblock.pas is not None:
                    auto_sel = auto_rhs1_full_strong_kind(
                        has_pas=True,
                        has_fp=False,
                        rhs1_precond_kind=rhs1_precond_kind,
                        total_size=int(op.total_size),
                        strong_precond_min=int(strong_precond_min),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
                        shard_axis=_matvec_shard_axis(op),
                        device_count=int(jax.device_count()),
                    )
                    strong_precond_kind = auto_sel.kind
                else:
                    strong_precond_kind = "schur"
            else:
                auto_sel = auto_rhs1_full_strong_kind(
                    has_pas=op.fblock.pas is not None,
                    has_fp=op.fblock.fp is not None,
                    rhs1_precond_kind=rhs1_precond_kind,
                    total_size=int(op.total_size),
                    strong_precond_min=int(strong_precond_min),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
                    shard_axis=_matvec_shard_axis(op),
                    device_count=int(jax.device_count()),
                )
                strong_precond_kind = auto_sel.kind
                strong_xblock_tz_lmax = auto_sel.xblock_tz_lmax

        auto_sel = adjust_rhs1_reduced_auto_kind(
            kind=strong_precond_kind,
            has_pas=op.fblock.pas is not None,
            geom_scheme=int(geom_scheme),
            n_zeta=int(op.n_zeta),
            strong_precond_trigger=True,
            max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
            n_theta=int(op.n_theta),
        )
        strong_precond_kind = auto_sel.kind
        if auto_sel.xblock_tz_lmax is not None:
            strong_xblock_tz_lmax = auto_sel.xblock_tz_lmax

        auto_sel = adjust_rhs1_theta_line_auto_kind(
            kind=strong_precond_kind,
            n_theta=int(op.n_theta),
            nxi_for_x_sum=int(np.sum(nxi_for_x)) if nxi_for_x.size else 0,
        )
        strong_precond_kind = auto_sel.kind

        if strong_precond_kind is not None and _rhs1_residual_needs_rescue(
            float(result.residual_norm),
            float(target),
        ):
            dd_block_theta = rhs1_dd_setup.block("theta")
            dd_overlap_theta = rhs1_dd_setup.overlap("theta", default=1)
            dd_block_zeta = rhs1_dd_setup.block("zeta")
            dd_overlap_zeta = rhs1_dd_setup.overlap("zeta", default=1)
            strong_precond_kind, strong_preconditioner_full = _build_rhs1_strong_preconditioner_full_from_kind(
                op=op,
                strong_precond_kind=strong_precond_kind,
                rhs1_precond_kind=rhs1_precond_kind,
                residual_norm=float(result.residual_norm),
                dd_block_theta=dd_block_theta,
                dd_overlap_theta=dd_overlap_theta,
                dd_block_zeta=dd_block_zeta,
                dd_overlap_zeta=dd_overlap_zeta,
            )
        if strong_precond_kind is not None and _rhs1_residual_needs_rescue(
            float(result.residual_norm),
            float(target),
        ):
            _mark("rhs1_strong_precond_build_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: strong preconditioner fallback "
                    f"kind={strong_precond_kind} (residual={float(result.residual_norm):.3e} > target={target:.3e})",
                )
            _mark("rhs1_strong_precond_build_done")

            strong_retry_controls = rhs1_strong_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
            )
            result, residual_vec, _accepted, _strong_elapsed_s = (
                rhs1_run_measured_linear_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=strong_preconditioner_full,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(strong_retry_controls.restart),
                    maxiter=int(strong_retry_controls.maxiter),
                    solve_method="incremental",
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear_with_residual,
                    solver_kind=_solver_kind("incremental")[0],
                    candidate_name="strong_full",
                    baseline_name="current_full",
                    target_value=float(target),
                    peak_rss_mb=_rss_mb(),
                    returns_residual_vec=True,
                )
            )
        pas_schur_rescue_controls = rhs1_pas_schur_rescue_controls_from_env(
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            n_species=int(op.n_species),
            residual_norm=float(result.residual_norm),
            target=float(target),
            active_size=int(active_size),
            restart=int(restart),
            maxiter=maxiter,
        )
        if pas_schur_rescue_controls.run:
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: PAS Schur rescue "
                    f"(residual={float(result.residual_norm):.3e} "
                    f"> {float(target)*float(pas_schur_rescue_controls.ratio):.3e})",
                )
            try:
                schur_precond = _build_rhsmode1_schur_preconditioner(op=op)
                result, residual_vec, _accepted, _schur_elapsed_s = (
                    rhs1_run_linear_candidate_and_update_replay(
                        replay_state=ksp_replay,
                        current_result=result,
                        current_residual_vec=residual_vec,
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=schur_precond,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(pas_schur_rescue_controls.restart),
                        maxiter=int(pas_schur_rescue_controls.maxiter),
                        solve_method="incremental",
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear_with_residual,
                        solver_kind=_solver_kind("incremental")[0],
                        returns_residual_vec=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: PAS Schur rescue failed ({type(exc).__name__}: {exc})")
        large_cpu_sparse_rescue_full = _rhsmode1_large_cpu_sparse_rescue_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(op.total_size),
            sparse_max_size=int(sparse_max_size),
            preconditioner_x=int(preconditioner_x),
            residual_norm=float(result.residual_norm),
            target=float(target),
        )

        sparse_exact_direct = _rhsmode1_host_sparse_direct_allowed(
            sparse_exact_lu=sparse_exact_lu,
            use_implicit=bool(use_implicit),
        )
        sparse_policy = rhs1_sparse_rescue_policy_setup(
            sparse_precond_mode=sparse_precond_mode,
            sparse_precond_kind=sparse_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            residual_norm=float(result.residual_norm),
            target=float(target),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            size=int(op.total_size),
            sparse_max_size=int(sparse_max_size),
            precond_dtype=_precond_dtype(int(op.total_size)),
            sparse_exact_direct=bool(sparse_exact_direct),
            large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue_full),
            sparse_jax_max_mb=float(sparse_jax_config.max_mb),
            pas_fast_accept=bool(pas_fast_accept),
            gpu_sparse_skip=bool(
                _rhs1_gpu_sparse_fallback_skip_allowed(
                    op=op,
                    rhs1_precond_kind=rhs1_precond_kind,
                    use_active_dof_mode=False,
                    residual_norm=float(result.residual_norm),
                    target=float(target),
                )
            ),
        )
        sparse_order = sparse_policy.ordering
        sparse_enabled = bool(sparse_policy.enabled)
        sparse_kind_use = str(sparse_policy.kind_use)
        if sparse_order.reason_size_large_cpu:
            sparse_exact_lu = _rhsmode1_large_cpu_sparse_exact_lu_allowed(active_size=int(op.total_size))
        if emit is not None:
            for _level, _message in rhs1_sparse_rescue_initial_messages(
                ordering=sparse_order,
                size=int(op.total_size),
                sparse_max_size=int(sparse_max_size),
                sparse_jax_memory_disabled_message=sparse_policy.sparse_jax_memory_disabled_message,
                large_cpu_sparse_exact_lu=bool(sparse_exact_lu),
                large_cpu_label="large CPU sparse",
            ):
                emit(_level, _message)

        dense_matrix_cache: np.ndarray | None = None
        host_sparse_direct_used = False
        if emit is not None:
            for _level, _message in rhs1_sparse_rescue_tail_skip_messages(
                ordering=sparse_order,
                residual_norm=float(result.residual_norm),
                rhs1_precond_kind=rhs1_precond_kind,
            ):
                emit(_level, _message)
        if sparse_enabled and float(result.residual_norm) > target:
            if sparse_kind_use == "jax":
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_jax",
                        active_size=int(op.total_size),
                        use_active_dof_mode=False,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    precond_dtype = _precond_dtype(int(op.total_size))
                    precond_sparse = build_sparse_jax_retry_preconditioner(
                        SparseJAXRetryPreconditionerBuildContext(
                            matvec=mv,
                            n=int(op.total_size),
                            dtype=precond_dtype,
                            cache_key=cache_key,
                            drop_tol=sparse_drop_tol,
                            drop_rel=sparse_drop_rel,
                            reg=sparse_jax_config.reg,
                            omega=sparse_jax_config.omega,
                            sweeps=sparse_jax_config.sweeps,
                            emit=emit,
                            builder=_build_sparse_jax_preconditioner_from_matvec,
                        )
                    )
                    _mark("rhs1_sparse_precond_build_done")
                    result, residual_vec, _accepted, _elapsed = (
                        rhs1_run_measured_linear_candidate_and_update_replay(
                            replay_state=ksp_replay,
                            current_result=result,
                            current_residual_vec=residual_vec,
                            matvec_fn=mv,
                            b_vec=rhs,
                            precond_fn=precond_sparse,
                            tol=tol,
                            atol=atol,
                            restart=restart,
                            maxiter=maxiter,
                            solve_method="incremental",
                            precond_side=gmres_precond_side,
                            solve_linear=_solve_linear_with_residual,
                            solver_kind=_solver_kind("incremental")[0],
                            candidate_name="sparse_jax_full",
                            baseline_name="current_full",
                            target_value=target,
                            peak_rss_mb=_rss_mb(),
                            returns_residual_vec=True,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(1, f"sparse_jax: failed ({type(exc).__name__}: {exc})")
            else:
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_lu" if sparse_exact_lu else "sparse_ilu",
                        active_size=int(op.total_size),
                        use_active_dof_mode=False,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    host_sparse_direct_wanted = _rhsmode1_host_sparse_direct_allowed(
                        sparse_exact_lu=sparse_exact_lu,
                        use_implicit=bool(use_implicit),
                    )
                    if large_cpu_sparse_rescue_full and sparse_exact_lu:
                        host_sparse_direct_wanted = True
                    sparse_operator_preconditioned_rescue = _rhsmode1_sparse_operator_preconditioned_rescue_allowed(
                        op=op,
                        sparse_exact_lu=sparse_exact_lu,
                        host_sparse_direct_wanted=host_sparse_direct_wanted,
                    )
                    sparse_factor_matvec = mv
                    if sparse_operator_preconditioned_rescue:
                        op_sparse_pc = _build_rhsmode1_preconditioner_operator_point(op)

                        def _mv_sparse_factor(v: jnp.ndarray, op_pc=op_sparse_pc) -> jnp.ndarray:
                            return apply_v3_full_system_operator_cached(op_pc, v)

                        sparse_factor_matvec = _mv_sparse_factor
                    cache_key_for_factor = cache_key
                    if sparse_operator_preconditioned_rescue:
                        cache_key_for_factor = _rhsmode1_sparse_cache_key(
                            op,
                            kind="sparse_lu_pc_point",
                            active_size=int(active_size),
                            use_active_dof_mode=False,
                            use_pas_projection=use_pas_projection,
                            drop_tol=sparse_drop_tol,
                            drop_rel=sparse_drop_rel,
                            ilu_drop_tol=sparse_ilu_drop_tol,
                            fill_factor=sparse_ilu_fill,
                        )
                    sparse_factor_controls = resolve_sparse_host_or_ilu_factor_controls(
                        n=int(op.total_size),
                        cache_key=cache_key_for_factor,
                        sparse_exact_lu=bool(sparse_exact_lu),
                        use_implicit=bool(use_implicit),
                        force_host_sparse_direct=False,
                        sparse_ilu_dense_max=int(sparse_ilu_dense_max),
                        sparse_dense_cache_max=int(sparse_dense_cache_max),
                        host_sparse_direct_wanted=host_sparse_direct_wanted,
                        host_sparse_direct_allowed=_rhsmode1_host_sparse_direct_allowed,
                        host_sparse_factor_dtype=_host_sparse_factor_dtype,
                        sparse_factor_cache_key=_sparse_factor_cache_key,
                        explicit_sparse_host_direct_allowed=_rhsmode1_explicit_sparse_host_direct_allowed,
                    )
                    factor_dtype = sparse_factor_controls.factor_dtype
                    explicit_sparse_pattern = (
                        _maybe_rhsmode1_full_sparse_pattern(op, emit=emit)
                        if sparse_factor_controls.explicit_sparse_allowed
                        else None
                    )
                    sparse_factor_build = build_sparse_host_or_ilu_factor(
                        SparseHostOrILUFactorBuildContext(
                            matvec=sparse_factor_matvec,
                            n=int(op.total_size),
                            dtype=rhs.dtype,
                            cache_key=sparse_factor_controls.cache_key_use,
                            factor_dtype=sparse_factor_controls.factor_dtype,
                            drop_tol=sparse_drop_tol,
                            drop_rel=sparse_drop_rel,
                            ilu_drop_tol=sparse_ilu_drop_tol,
                            fill_factor=sparse_ilu_fill,
                            build_dense_factors=sparse_factor_controls.build_dense_factors,
                            build_jax_factors=sparse_factor_controls.build_jax_factors,
                            store_dense=sparse_factor_controls.store_dense,
                            factorization="lu" if sparse_exact_lu else "ilu",
                            emit=emit,
                            host_sparse_direct_wanted=sparse_factor_controls.host_sparse_direct_wanted,
                            explicit_sparse_allowed=sparse_factor_controls.explicit_sparse_allowed,
                            explicit_sparse_pattern=explicit_sparse_pattern,
                            build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
                            build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
                        )
                    )
                    explicit_sparse_operator = sparse_factor_build.explicit_sparse_operator
                    explicit_sparse_factor = sparse_factor_build.explicit_sparse_factor
                    a_csr_full = sparse_factor_build.a_csr_full
                    _a_csr_drop = sparse_factor_build.a_csr_drop
                    ilu = sparse_factor_build.ilu
                    a_dense_cache = sparse_factor_build.a_dense_cache
                    l_dense = sparse_factor_build.l_dense
                    u_dense = sparse_factor_build.u_dense
                    l_unit_diag = sparse_factor_build.l_unit_diag
                    dense_matrix_cache = a_dense_cache
                    _mark("rhs1_sparse_precond_build_done")
                    host_sparse_direct = host_sparse_direct_wanted
                    sparse_retry_timer = Timer()

                    if host_sparse_direct and ilu is not None:
                        if sparse_operator_preconditioned_rescue:
                            scipy_sparse_build = build_sparse_host_scipy_preconditioner(
                                SparseHostScipyPreconditionerBuildContext(
                                    ilu=ilu,
                                    a_csr_full=a_csr_full,
                                    base_matvec=mv,
                                    sparse_use_matvec=False,
                                )
                            )
                            _precond_sparse = scipy_sparse_build.preconditioner

                            sparse_pc_restart, sparse_pc_maxiter = rhs1_parse_polish_gmres_config(
                                restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART",
                                maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER",
                                default_restart=max(120, int(restart)),
                                default_maxiter=max(800, int(maxiter or 400) * 2),
                            )
                            if emit is not None:
                                emit(
                                    0,
                                    "solve_v3_full_system_linear_gmres: sparse LU operator-preconditioned "
                                    f"GMRES fallback restart={int(sparse_pc_restart)} maxiter={int(sparse_pc_maxiter)}",
                                )
                            res_sparse, residual_vec_sparse = run_sparse_host_scipy_gmres(
                                SparseHostScipyGMRESContext(
                                    matvec=mv,
                                    rhs=rhs,
                                    preconditioner=_precond_sparse,
                                    x0=result.x,
                                    tol=tol,
                                    atol=atol,
                                    restart=sparse_pc_restart,
                                    maxiter=sparse_pc_maxiter,
                                    precondition_side=gmres_precond_side,
                                    gmres_solver=gmres_solve_with_history_scipy,
                                    residual_matvec=mv,
                                )
                            )
                            _mv_sparse = mv
                        else:
                            host_sparse_direct_used = True
                            sparse_host_fallback = sparse_host_direct_fallback_payload(
                                explicit_sparse_factor=explicit_sparse_factor,
                                explicit_sparse_operator=explicit_sparse_operator,
                                ilu=ilu,
                                a_csr_full=a_csr_full,
                                rhs=rhs,
                                factor_dtype=factor_dtype,
                                refine_steps=_host_sparse_direct_refine_steps(
                                    "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_REFINE",
                                    default=2,
                                ),
                                matvec=mv,
                                target=float(target),
                                tol=tol,
                                atol=atol,
                                restart=restart,
                                maxiter=maxiter,
                                precondition_side=gmres_precond_side,
                                emit=emit,
                                backend_name=jax.default_backend(),
                                polish_enabled=rhs1_polish_enabled,
                                parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
                                direct_solve_with_refinement=_host_direct_solve_with_refinement,
                                ilu_solve_with_refinement=_host_sparse_direct_solve_with_refinement,
                                host_sparse_direct_polish=_host_sparse_direct_polish,
                            )
                            res_sparse = GMRESSolveResult(
                                x=sparse_host_fallback.x,
                                residual_norm=sparse_host_fallback.residual_norm,
                            )
                            residual_vec_sparse = sparse_host_fallback.residual_vec
                            _mv_sparse = mv
                            _precond_sparse = None
                    elif use_implicit:
                        ilu_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(cache_key)
                        precond_build = build_sparse_ilu_preconditioner_from_cache(
                            SparseILUPreconditionerBuildContext(
                                cache_entry=ilu_cache,
                                l_dense=l_dense,
                                u_dense=u_dense,
                                l_unit_diag=l_unit_diag,
                                require_lower_diag=True,
                            )
                        )
                        precond_sparse = precond_build.preconditioner

                        if precond_sparse is None:
                            if emit is not None:
                                emit(
                                    1,
                                    f"{'sparse_lu' if sparse_exact_lu else 'sparse_ilu'}: "
                                    "implicit preconditioner factors unavailable; skipping",
                                )
                            res_sparse = None
                            residual_vec_sparse = None
                        else:
                            if emit is not None:
                                emit(
                                    0,
                                    "solve_v3_full_system_linear_gmres: "
                                    f"{'sparse LU' if sparse_exact_lu else 'sparse ILU'} (implicit) fallback",
                                )
                            res_sparse, residual_vec_sparse = _solve_linear_with_residual(
                                matvec_fn=mv,
                                b_vec=rhs,
                                precond_fn=precond_sparse,
                                x0_vec=result.x,
                                tol_val=tol,
                                atol_val=atol,
                                restart_val=restart,
                                maxiter_val=maxiter,
                                solve_method_val="incremental",
                                precond_side=gmres_precond_side,
                            )
                    else:
                        scipy_sparse_build = build_sparse_host_scipy_preconditioner(
                            SparseHostScipyPreconditionerBuildContext(
                                ilu=ilu,
                                a_csr_full=a_csr_full,
                                base_matvec=mv,
                                sparse_use_matvec=sparse_use_matvec,
                            )
                        )
                        _precond_sparse = scipy_sparse_build.preconditioner
                        _mv_sparse = scipy_sparse_build.matvec

                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: "
                                f"{'sparse LU' if sparse_exact_lu else 'sparse ILU'} GMRES fallback",
                            )
                        res_sparse, residual_vec_sparse = run_sparse_host_scipy_gmres(
                            SparseHostScipyGMRESContext(
                                matvec=_mv_sparse,
                                rhs=rhs,
                                preconditioner=_precond_sparse,
                                x0=result.x,
                                tol=tol,
                                atol=atol,
                                restart=restart,
                                maxiter=maxiter,
                                precondition_side=gmres_precond_side,
                                gmres_solver=gmres_solve_with_history_scipy,
                                residual_matvec=_mv_sparse,
                            )
                        )
                    if res_sparse is not None:
                        sparse_retry_elapsed_s = sparse_retry_timer.elapsed_s()
                        result, residual_vec, _accepted = rhs1_accept_sparse_retry_candidate_and_update_replay(
                            replay_state=ksp_replay,
                            current_result=result,
                            candidate_result=res_sparse,
                            current_residual_vec=residual_vec,
                            candidate_residual_vec=residual_vec_sparse,
                            matvec_fn=mv if use_implicit else _mv_sparse,
                            b_vec=rhs,
                            precond_fn=_precond_sparse,
                            restart=restart,
                            maxiter=maxiter,
                            precond_side=gmres_precond_side,
                            solver_kind=_solver_kind("incremental")[0],
                            candidate_family="sparse",
                            scope="full",
                            target_value=target,
                            solve_s=sparse_retry_elapsed_s,
                            peak_rss_mb=_rss_mb(),
                        )
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(
                            1,
                            f"{'sparse_lu' if sparse_exact_lu else 'sparse_ilu'}: "
                            f"failed ({type(exc).__name__}: {exc})",
                        )
        residual_vec, residual_norm_true, residual_norm_check = (
            rhs1_replay_left_preconditioned_residual_norms(
                result=result,
                rhs=rhs,
                matvec=mv,
                residual_vec=residual_vec,
                preconditioner=ksp_replay.precond_fn,
                precondition_side=ksp_replay.precond_side,
                update_residual_vec=True,
            )
        )
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        dense_backend_allowed = _rhsmode1_dense_backend_allowed()
        host_dense_fallback_allowed = _rhsmode1_host_dense_fallback_allowed()
        dense_krylov_allowed = _rhsmode1_dense_krylov_allowed()
        if (not dense_backend_allowed) and (not host_dense_fallback_allowed) and (not dense_krylov_allowed):
            dense_fallback_max = 0
        res_ratio = float(residual_norm_true) / max(float(target), 1e-300)
        dense_thresholds = rhs1_dense_fallback_thresholds_from_env(
            dense_fallback_max=int(dense_fallback_max),
            residual_ratio=float(res_ratio),
            allow_huge_limit=False,
        )
        dense_fallback_trigger = bool(dense_thresholds.dense_fallback_trigger)
        if host_sparse_direct_used and jax.default_backend() != "cpu":
            host_sparse_skip_ratio = _rhsmode1_host_sparse_skip_dense_ratio()
            if host_sparse_skip_ratio > 0.0 and res_ratio <= float(host_sparse_skip_ratio):
                dense_fallback_trigger = False
                dense_fallback_max = 0
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: skipping dense fallback after host sparse LU "
                        f"(ratio={res_ratio:.3e} <= {float(host_sparse_skip_ratio):.1e})",
                    )
        fp_force_dense = (
            op.fblock.fp is not None
            and dense_fallback_max > 0
            and int(active_size) <= dense_fallback_max
            and float(residual_norm_true) > target
        )
        if fp_force_dense:
            dense_fallback_trigger = True
        force_dense_cs0 = bool(int(op.constraint_scheme) == 0 and (not cs0_sparse_first))
        if force_dense_cs0:
            dense_fallback_trigger = True
        if (
            dense_fallback_max > 0
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and int(op.total_size) <= dense_fallback_max
            and dense_fallback_trigger
            and float(residual_norm_true) > target
        ):
            _mark("rhs1_dense_fallback_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback "
                    f"(size={int(op.total_size)} residual={float(residual_norm_check):.3e} > target={target:.3e})",
                )
            try:
                dense_retry_timer = Timer()
                use_row_scaled = int(op.constraint_scheme) == 0
                if dense_backend_allowed:
                    dense_method = "dense_row_scaled" if use_row_scaled else "dense"
                    res_dense, residual_vec_dense = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=None,
                        x0_vec=None,
                        tol_val=tol,
                        atol_val=atol,
                        restart_val=restart,
                        maxiter_val=maxiter,
                        solve_method_val=dense_method,
                        precond_side="none",
                    )
                else:
                    if emit is not None and jax.default_backend() != "cpu":
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: dense fallback using explicit dense Krylov "
                            f"on backend={jax.default_backend()}",
                        )
                    if dense_matrix_cache is not None:
                        a_dense = jnp.asarray(dense_matrix_cache, dtype=rhs.dtype)
                    else:
                        a_dense = assemble_dense_matrix_from_matvec(
                            matvec=mv, n=int(op.total_size), dtype=rhs.dtype
                        )
                    res_dense, residual_vec_dense = dense_krylov_solve_from_matrix_with_residual(
                        a=a_dense,
                        b=rhs,
                        x0=result.x,
                        preconditioner=None,
                        tol=tol,
                        atol=atol,
                        restart=restart,
                        maxiter=maxiter,
                        solve_method="incremental",
                        precondition_side="none",
                        row_scaled=use_row_scaled,
                    )
                dense_retry_elapsed_s = dense_retry_timer.elapsed_s()
                result, residual_vec, _accepted = rhs1_accept_measured_candidate_and_update_replay(
                    replay_state=ksp_replay,
                    current_result=result,
                    candidate_result=res_dense,
                    current_residual_vec=residual_vec,
                    candidate_residual_vec=residual_vec_dense,
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=None,
                    x0_vec=res_dense.x,
                    restart=restart,
                    maxiter=maxiter,
                    precond_side="none",
                    solver_kind="dense",
                    candidate_name="dense_full",
                    baseline_name="current_full",
                    target_value=target,
                    solve_s=dense_retry_elapsed_s,
                    peak_rss_mb=_rss_mb(),
                )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: dense fallback failed ({type(exc).__name__}: {exc})")
            _mark("rhs1_dense_fallback_done")
    if int(op.rhs_mode) == 1:
        project_env = os.environ.get("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "").strip().lower()
        if project_env in {"0", "false", "no", "off"}:
            project_rhs1 = False
        elif project_env in {"1", "true", "yes", "on"}:
            project_rhs1 = True
        else:
            # Default parity-first behavior: enforce constraintScheme=1 nullspace projection
            # for linear RHSMode=1 solves without Phi1.
            project_rhs1 = bool(int(op.constraint_scheme) == 1 and (not bool(op.include_phi1)))
        if project_rhs1:
            x_projected, residual_projected = _project_constraint_scheme1_nullspace_solution_with_residual(
                op=op,
                x_vec=result.x,
                rhs_vec=rhs,
                matvec_op=op,
                enabled_env_var="SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE",
                residual_vec=residual_vec if residual_vec is not None and residual_vec.shape == rhs.shape else None,
            )
            if not bool(jnp.allclose(x_projected, result.x)):
                residual_norm_projected = jnp.linalg.norm(residual_projected)
                result = GMRESSolveResult(x=x_projected, residual_norm=residual_norm_projected)
        if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
            zero_tol = rhs1_pas_source_zero_tolerance_from_env()
            if zero_tol > 0.0:
                extra = result.x[-int(op.extra_size) :]
                max_abs = jnp.max(jnp.abs(extra))
                extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
                x_new = jnp.concatenate([result.x[: -int(op.extra_size)], extra], axis=0)
                result = GMRESSolveResult(x=x_new, residual_norm=result.residual_norm)
    if ksp_replay.matvec_fn is not None and ksp_replay.b_vec is not None:
        ksp_history = emit_profile_response_ksp_history(
            context=rhs1_ksp_diagnostics_context,
            matvec_fn=ksp_replay.matvec_fn,
            b_vec=ksp_replay.b_vec,
            precond_fn=ksp_replay.precond_fn,
            x0_vec=ksp_replay.x0_vec,
            tol_val=tol,
            atol_val=atol,
            restart_val=int(ksp_replay.restart),
            maxiter_val=ksp_replay.maxiter,
            precond_side=ksp_replay.precond_side,
            solver_kind=ksp_replay.solver_kind,
            solve_method_val=str(solve_method),
        )
        emit_profile_response_ksp_iter_stats(
            context=rhs1_ksp_diagnostics_context,
            matvec_fn=ksp_replay.matvec_fn,
            b_vec=ksp_replay.b_vec,
            precond_fn=ksp_replay.precond_fn,
            x0_vec=ksp_replay.x0_vec,
            tol_val=float(tol),
            atol_val=float(atol),
            restart_val=int(ksp_replay.restart),
            maxiter_val=ksp_replay.maxiter,
            precond_side=ksp_replay.precond_side,
            solver_kind=ksp_replay.solver_kind,
            history=ksp_history,
            solve_method_val=str(solve_method),
        )
    if emit is not None:
        emit(0, f"solve_v3_full_system_linear_gmres: residual_norm={float(result.residual_norm):.6e}")
        emit(1, f"solve_v3_full_system_linear_gmres: elapsed_s={t.elapsed_s():.3f}")
    metadata_out = {}
    metadata_out.update(pas_tz_guarded_correction_metadata)
    metadata_out.update(rhsmode1_general_metadata)
    if int(op.rhs_mode) == 1:
        post_xblock_accept_floor = _rhsmode1_scipy_rescue_abs_floor_after_xblock(
            op=op,
            active_size=int(active_size),
            used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
            used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
            use_implicit=bool(use_implicit),
        )
        if (
            float(post_xblock_accept_floor) > 0.0
            and np.isfinite(float(result.residual_norm))
            and float(result.residual_norm) <= float(post_xblock_accept_floor)
        ):
            true_residual_target = rhs1_residual_target(
                atol=float(atol),
                tol=float(tol),
                rhs_norm=rhs1_l2_norm_float(rhs),
            )
            metadata_out.update(
                {
                    "accepted_converged": True,
                    "acceptance_criterion": "post_xblock_abs_floor",
                    "true_residual_converged": rhs1_residual_converged(
                        float(result.residual_norm),
                        true_residual_target,
                    ),
                    "accepted_residual_floor": float(post_xblock_accept_floor),
                }
            )
    return V3LinearSolveResult(
        op=op,
        rhs=rhs,
        gmres=result,
        metadata=metadata_out or None,
    )


solve_v3_full_system_linear_gmres_jit = jax.jit(
    solve_v3_full_system_linear_gmres,
    static_argnames=("tol", "atol", "restart", "maxiter", "solve_method", "identity_shift"),
)


def solve_v3_full_system_newton_krylov(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
) -> V3NewtonKrylovResult:
    """Solve `residual_v3_full_system(op, x) = 0` using a basic Newton–Krylov iteration.

    This is intended for small parity fixtures and developer experimentation. It is **not**
    yet a stable API for production runs.
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    _set_precond_size_hint(int(op.total_size))
    _set_precond_policy_hints(
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
    )
    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for k in range(int(max_newton)):
        # Compute residual and a *single* linearization for this Newton step that can be reused
        # by GMRES. This avoids applying JAX's autodiff transform inside every matvec call,
        # which is a major performance bottleneck for includePhi1 solves.
        r, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
        rnorm = jnp.linalg.norm(r)
        if float(rnorm) < float(tol):
            return V3NewtonKrylovResult(
                op=op,
                x=x,
                residual_norm=rnorm,
                n_newton=k,
                last_linear_residual_norm=last_linear_resid,
            )

        # Solve J s = -r
        lin = _gmres_solve_dispatch(
            matvec=jvp,
            b=-r,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=str(solve_method),
        )
        s = lin.x
        last_linear_resid = lin.residual_norm

        # Backtracking line search on ||r|| (very simple Armijo-style criterion).
        step = 1.0
        step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
        try:
            step_scale = float(step_scale_env) if step_scale_env else 1.0
        except ValueError:
            step_scale = 1.0
        rnorm0 = float(rnorm)
        for _ in range(12):
            x_try = x + (step * step_scale) * s
            r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if rnorm_try <= 0.9 * rnorm0:
                x = x_try
                break
            step *= 0.5
        else:
            # If we fail to reduce the residual, still take a small step to avoid stalling.
            x = x + (1.0 / 64.0) * s

    r = residual_v3_full_system(op, x)
    return V3NewtonKrylovResult(
        op=op,
        x=x,
        residual_norm=jnp.linalg.norm(r),
        n_newton=int(max_newton),
        last_linear_residual_norm=last_linear_resid,
    )


def solve_v3_full_system_newton_krylov_history(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
    nonlinear_rtol: float = 0.0,
    use_frozen_linearization: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[V3NewtonKrylovResult, list[jnp.ndarray]]:
    """Newton–Krylov solve that also returns the per-iteration accepted states.

    The returned history matches v3's convention of saving diagnostics for iteration numbers
    starting at 1, i.e. it includes the sequence of *accepted* Newton iterates and excludes
    the initial guess `x0`.

    Optionally, this routine can use a v3-parity-oriented solve path with a frozen
    (`whichMatrix=1`-like) linearization and relative residual stopping
    (`||F|| <= nonlinear_rtol * ||F_0||`) in addition to absolute `tol`.
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    if emit is not None:
        emit(1, f"solve_v3_full_system_newton_krylov_history: total_size={int(op.total_size)}")
    fortran_stdout = rhs1_fortran_stdout_from_env(emit=emit)
    env_gmres_tol = os.environ.get("SFINCS_JAX_PHI1_GMRES_TOL", "").strip()
    if env_gmres_tol:
        gmres_tol = float(env_gmres_tol)

    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    active_env = os.environ.get("SFINCS_JAX_PHI1_ACTIVE_DOF", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    has_reduced_modes = bool(np.any(nxi_for_x < int(op.n_xi)))
    use_active_dof_mode = phi1_use_active_dof_mode(
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        has_reduced_modes=has_reduced_modes,
        env_value=active_env,
    )

    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op.total_size)
    if use_active_dof_mode:
        active_idx_np = _transport_active_dof_indices(op)
        active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
        full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
        full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
            1, int(active_idx_np.shape[0]) + 1, dtype=np.int32
        )
        full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
        active_size = int(active_idx_np.shape[0])
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_newton_krylov_history: active-DOF mode enabled "
                f"(size={active_size}/{int(op.total_size)})",
            )
    gmres_restart_use = phi1_gmres_restart(active_size=active_size, gmres_restart=int(gmres_restart))

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        assert active_idx_jnp is not None
        return v_full[active_idx_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        assert full_to_active_jnp is not None
        z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
        padded = jnp.concatenate([z0, v_reduced], axis=0)
        return padded[full_to_active_jnp]

    preconditioner = None
    # Only enable block preconditioning in the PETSc-like parity mode that freezes the
    # Jacobian/linearization. For autodiff-linearized Newton steps, JAX's GMRES can
    # behave differently with an approximate preconditioner, which impacts iteration
    # histories and `sfincsOutput.h5` shape parity for linear Phi1 fixtures.
    pc_env = os.environ.get("SFINCS_JAX_PHI1_USE_PRECONDITIONER", "").strip().lower()
    use_preconditioner = pc_env not in {"0", "false", "no", "off"}
    dense_cutoff_env = os.environ.get("SFINCS_JAX_PHI1_NK_DENSE_CUTOFF", "").strip()
    try:
        dense_cutoff = int(dense_cutoff_env) if dense_cutoff_env else 5000
    except ValueError:
        dense_cutoff = 5000
    linear_size = active_size if use_active_dof_mode else int(op.total_size)
    solve_method_in = str(solve_method).strip().lower()
    use_sparse_direct_linear = solve_method_in == "sparse_direct"
    use_dense_linear = solve_method_in in {"dense", "dense_row_scaled"} or (
        use_frozen_linearization and int(linear_size) <= int(dense_cutoff)
    )
    if use_dense_linear or use_sparse_direct_linear:
        use_preconditioner = False
    preconditioner = build_phi1_newton_preconditioner(
        use_preconditioner=bool(use_preconditioner),
        use_frozen_linearization=bool(use_frozen_linearization),
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        use_active_dof_mode=bool(use_active_dof_mode),
        op=op,
        reduce_full=_reduce_full if use_active_dof_mode else None,
        expand_reduced=_expand_reduced if use_active_dof_mode else None,
        preconditioner_options=nml.group("preconditionerOptions"),
        collision_builder=_build_rhsmode1_collision_preconditioner,
        block_builder=_build_rhsmode1_block_preconditioner,
        emit=emit,
    )

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)
    accepted: list[jnp.ndarray] = []
    rnorm_initial: float | None = None
    cached_jvp = None
    cached_jvp_iter = -1
    frozen_jac_policy = phi1_frozen_jacobian_policy(include_phi1=bool(op.include_phi1))
    use_frozen_jac_cache = bool(frozen_jac_policy.use_cache)
    frozen_jac_every = int(frozen_jac_policy.every)
    ksp_history_limits = rhs1_ksp_history_limits_from_env()
    ksp_history_max_size = ksp_history_limits.max_size
    ksp_history_max_iter = ksp_history_limits.max_iter

    def _emit_ksp_history_nk(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
    ) -> None:
        _emit_newton_krylov_ksp_history(
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            precond_side=precond_side,
            emit=emit,
            fortran_stdout=bool(fortran_stdout),
            max_size=ksp_history_max_size,
            max_history_iter=ksp_history_max_iter,
        )

    def _phi1_sparse_direct_solve(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        n: int,
        cache_tag: tuple[object, ...],
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
    ) -> GMRESSolveResult:
        factor_dtype = _transport_sparse_factor_dtype(size=int(n), use_implicit=False)
        cache_key_use = _sparse_factor_cache_key(("phi1_nk_sparse_direct", *cache_tag), factor_dtype)
        a_csr_full, _a_csr_drop, ilu, _a_dense, _l_dense, _u_dense, _l_unit = _build_sparse_ilu_from_matvec(
            matvec=matvec_fn,
            n=int(n),
            dtype=jnp.float64,
            cache_key=cache_key_use,
            factor_dtype=factor_dtype,
            drop_tol=0.0,
            drop_rel=0.0,
            ilu_drop_tol=0.0,
            fill_factor=1.0,
            build_dense_factors=False,
            build_jax_factors=False,
            build_ilu=True,
            store_dense=False,
            factorization="lu",
            emit=emit,
        )
        if ilu is None:
            raise RuntimeError("phi1 sparse_direct: factors unavailable")
        x_np, residual_norm = _host_sparse_direct_solve_with_refinement(
            ilu=ilu,
            a_csr_full=a_csr_full,
            rhs_vec=b_vec,
            factor_dtype=factor_dtype,
            refine_steps=_host_sparse_direct_refine_steps("SFINCS_JAX_PHI1_SPARSE_DIRECT_REFINE", default=2),
        )
        target_true = max(float(atol_val), float(tol_val) * float(jnp.linalg.norm(b_vec)))
        if factor_dtype == np.dtype(np.float32) and residual_norm > target_true:
            polish_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH", "").strip().lower()
            if polish_env not in {"0", "false", "no", "off"}:
                polish_restart_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH_RESTART", "").strip()
                polish_maxiter_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH_MAXITER", "").strip()
                try:
                    polish_restart = int(polish_restart_env) if polish_restart_env else min(int(restart_val), 40)
                except ValueError:
                    polish_restart = min(int(restart_val), 40)
                try:
                    polish_maxiter = (
                        int(polish_maxiter_env)
                        if polish_maxiter_env
                        else min(max(40, int(maxiter_val or 120)), 120)
                    )
                except ValueError:
                    polish_maxiter = min(max(40, int(maxiter_val or 120)), 120)
                x_polish, residual_norm_polish = _host_sparse_direct_polish(
                    matvec_fn=matvec_fn,
                    rhs_vec=b_vec,
                    x0_np=x_np,
                    ilu=ilu,
                    factor_dtype=factor_dtype,
                    tol=tol_val,
                    atol=atol_val,
                    restart=max(5, int(polish_restart)),
                    maxiter=max(5, int(polish_maxiter)),
                    precondition_side="left",
                )
                if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm:
                    x_np = x_polish
                    residual_norm = residual_norm_polish
        return GMRESSolveResult(
            x=jnp.asarray(x_np, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        )

    for k in range(int(max_newton)):
        if emit is not None:
            emit(1, f"newton_iter={k}: evaluateResidual called")
        op_use = op
        if bool(op.include_phi1):
            phi1_flat = x[op.f_size : op.f_size + op.n_theta * op.n_zeta]
            phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
            op_use = replace(op, phi1_hat_base=phi1)

        r = apply_v3_full_system_operator_cached(op_use, x, include_jacobian_terms=False) - rhs_v3_full_system_jit(op_use)
        rnorm = jnp.linalg.norm(r)
        rnorm_f = float(rnorm)
        if rnorm_initial is None:
            rnorm_initial = max(rnorm_f, 1e-300)
        if emit is not None:
            emit(0, f"newton_iter={k}: residual_norm={rnorm_f:.6e}")
        if emit is not None and fortran_stdout:
            emit(0, f"{k:4d} SNES Function norm {rnorm_f: .12e} ")
        if not np.isfinite(rnorm_f):
            # Keep the latest finite iterate. This mirrors PETSc's behavior of
            # stopping when the nonlinear residual becomes invalid instead of
            # continuing with NaN/Inf states.
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        converged_abs = rnorm_f < float(tol)
        converged_rel = rnorm_f <= float(nonlinear_rtol) * float(rnorm_initial)
        if converged_abs or converged_rel:
            if not accepted:
                accepted.append(x)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x,
                    residual_norm=rnorm,
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        frozen_jac_mode = None
        if use_frozen_linearization:
            jac_mode = frozen_jac_policy.mode
            frozen_jac_mode = jac_mode

            if jac_mode == "frozen_rhs":
                # Keep the kinetic/collision operator frozen at the current iterate (op_use),
                # but let the RHS keep its explicit Phi1 dependence. This is closer to v3's
                # nonlinear Jacobian path for includePhi1 while retaining robust parity behavior.
                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_rhs_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_rhs_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_use, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_rhs_x)
                    )

                reuse_cached = (
                    use_frozen_jac_cache
                    and cached_jvp is not None
                    and (k - cached_jvp_iter) < frozen_jac_every
                )
                if reuse_cached:
                    matvec = cached_jvp
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian reused (frozen_rhs cache)")
                else:
                    _r_lin, jvp = jax.linearize(residual_for_jac, x)
                    matvec = jvp
                    if use_frozen_jac_cache:
                        cached_jvp = jvp
                        cached_jvp_iter = k
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian called (frozen operator + dynamic RHS)")
            elif jac_mode == "frozen_op":
                # Keep RHS frozen at the current iterate, but let the operator
                # carry the Phi1 dependence. This emulates partial Jacobian updates
                # in upstream SNES paths.
                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_mat_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_mat_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_mat_x, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_use)
                    )

                _r_lin, jvp = jax.linearize(residual_for_jac, x)
                matvec = jvp
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (dynamic operator + frozen RHS)")
            else:
                matvec = lambda dx: apply_v3_full_system_jacobian_jit(op_use, x, dx)
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (fully frozen linearization)")
        else:
            # Optional exact mode for debugging/experimentation.
            _r_lin, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
            matvec = jvp
            if emit is not None:
                emit(1, f"newton_iter={k}: evaluateJacobian called (autodiff linearization)")

        solve_method_linear = str(solve_method)
        if use_frozen_linearization:
            if int(linear_size) <= int(dense_cutoff):
                solve_method_linear = "dense"

        lin, s, linear_resid_norm = solve_phi1_newton_linear_step(
            use_active_dof_mode=bool(use_active_dof_mode),
            solve_method_linear=solve_method_linear,
            matvec=matvec,
            residual_vec=r,
            preconditioner=preconditioner,
            gmres_tol=float(gmres_tol),
            gmres_restart=int(gmres_restart_use),
            gmres_maxiter=gmres_maxiter,
            sparse_direct_solve=_phi1_sparse_direct_solve,
            gmres_dispatch=_gmres_solve_dispatch,
            gmres_result_is_finite=_gmres_result_is_finite,
            emit_ksp_history=_emit_ksp_history_nk,
            emit=emit,
            newton_iter=int(k),
            reduce_full=_reduce_full if use_active_dof_mode else None,
            expand_reduced=_expand_reduced if use_active_dof_mode else None,
            active_size=int(active_size),
            total_size=int(op.total_size),
        )

        if emit is not None:
            emit(1, f"newton_iter={k}: gmres_residual={float(linear_resid_norm):.6e}")
        if not _gmres_result_is_finite(lin):
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )
        last_linear_resid = linear_resid_norm

        rnorm0 = float(rnorm)
        ls_policy = phi1_line_search_policy(
            use_frozen_linearization=bool(use_frozen_linearization),
            include_phi1=bool(op.include_phi1),
        )
        x = advance_phi1_newton_iterate(
            x=x,
            step_direction=s,
            residual_norm0=rnorm0,
            residual_fn=lambda x_try: residual_v3_full_system(op, x_try),
            accepted=accepted,
            mode=str(ls_policy.mode),
            step_scale=float(ls_policy.step_scale),
            factor=ls_policy.factor,
            c1=float(ls_policy.c1),
            maxiter=int(ls_policy.maxiter),
        )
        accepted.append(x)

    r = residual_v3_full_system(op, x)
    return (
        V3NewtonKrylovResult(
            op=op,
            x=x,
            residual_norm=jnp.linalg.norm(r),
            n_newton=int(max_newton),
            last_linear_residual_norm=last_linear_resid,
        ),
        accepted,
    )


@contextlib.contextmanager
def _transport_parallel_worker_env(parallel_workers: int):
    with _transport_parallel_worker_env_impl(
        parallel_workers=int(parallel_workers),
        rewrite_xla_flags=_rewrite_xla_flags,
    ):
        yield


def _transport_parallel_worker(payload: dict[str, object]) -> dict[str, object]:
    """Worker entry point for parallel whichRHS transport solves."""
    return _solve_transport_parallel_payload(
        payload,
        read_input=read_sfincs_input,
        solve_transport=solve_v3_transport_matrix_linear_gmres,
    )


_TRANSPORT_PARALLEL_POOL_CACHE = TransportParallelPoolCache()


_transport_parallel_start_method = _transport_parallel_start_method_impl


_transport_parallel_backend = _transport_parallel_backend_impl


_transport_parallel_persistent_pool_enabled = (
    _transport_parallel_persistent_pool_enabled_impl
)


_transport_parallel_pool_key = _transport_parallel_pool_key_impl


_transport_parallel_visible_gpu_ids = _transport_parallel_visible_gpu_ids_impl


_transport_parallel_gpu_worker_env = _transport_parallel_gpu_worker_env_impl


def _run_transport_parallel_gpu_subprocesses(
    *,
    payloads: list[dict[str, object]],
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    return _run_transport_parallel_gpu_subprocesses_impl(
        payloads=payloads,
        parallel_workers=int(parallel_workers),
        visible_gpu_ids=_transport_parallel_visible_gpu_ids,
        gpu_worker_env=_transport_parallel_gpu_worker_env,
        emit=emit,
    )


def _transport_parallel_pool_executor_kwargs(
    *,
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    return _transport_parallel_pool_executor_kwargs_impl(
        parallel_workers=int(parallel_workers),
        get_context=mp.get_context,
        emit=emit,
    )


def _shutdown_transport_parallel_pool() -> None:
    _TRANSPORT_PARALLEL_POOL_CACHE.shutdown()


atexit.register(_shutdown_transport_parallel_pool)


def _get_transport_parallel_pool(
    *,
    parallel_workers: int,
    emit: Callable[[int, str], None] | None = None,
) -> concurrent.futures.ProcessPoolExecutor:
    return _TRANSPORT_PARALLEL_POOL_CACHE.get(
        parallel_workers=int(parallel_workers),
        key_fn=_transport_parallel_pool_key,
        worker_env=_transport_parallel_worker_env,
        executor_kwargs=_transport_parallel_pool_executor_kwargs,
        executor_class=concurrent.futures.ProcessPoolExecutor,
        emit=emit,
    )


def _transport_active_dof_indices(op: V3FullSystemOperator) -> np.ndarray:
    """Return full-vector indices for active transport solve DOFs.

    For v3 RHSMode=2/3 transport solves, Fortran only includes active Legendre
    modes for each x (as set by `Nxi_for_x`) in the linear system unknown vector.
    This helper builds that reduced active set so matrix-free solves can mirror
    Fortran's non-singular system size.
    """
    return build_rhs1_compressed_pitch_layout(op).active_full_indices.astype(np.int32, copy=False)


def solve_v3_transport_matrix_linear_gmres(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    x0_by_rhs: dict[int, jnp.ndarray] | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "auto",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    differentiable: bool | None = None,
    emit: Callable[[int, str], None] | None = None,
    input_namelist: Path | None = None,
    which_rhs_values: Sequence[int] | None = None,
    force_stream_diagnostics: bool | None = None,
    force_store_state: bool | None = None,
    collect_transport_output_fields: bool = True,
    parallel_workers: int | None = None,
) -> V3TransportMatrixSolveResult:
    """Compute a RHSMode=2/3 transport matrix by running all `whichRHS` solves matrix-free in JAX.

    Notes
    -----
    This mirrors the v3 `solver.F90` RHSMode=2/3 path:
    - Loop `whichRHS`
    - Overwrite (dnHatdpsiHats, dTHatdpsiHats, EParallelHat)
    - Build the RHS via `evaluateResidual(f=0)`
    - Solve `A x = rhs`
    - Use `diagnostics.F90` formulas to fill `transportMatrix`
    """
    t_all = Timer()

    maxiter_setup = resolve_transport_maxiter_setup(maxiter)
    maxiter = maxiter_setup.maxiter
    if emit is not None:
        for level, message in maxiter_setup.notes:
            emit(int(level), message)

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: starting whichRHS loop")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    _set_precond_size_hint(int(op0.total_size))
    _set_precond_policy_hints(
        has_pas=getattr(op0.fblock, "pas", None) is not None,
        has_fp=getattr(op0.fblock, "fp", None) is not None,
        include_phi1=bool(op0.include_phi1),
        rhs_mode=int(op0.rhs_mode),
    )
    state_setup = resolve_transport_state_setup(op=op0, x0=x0, x0_by_rhs=x0_by_rhs)
    state_in_env = state_setup.state_in_path
    state_out_env = state_setup.state_out_path
    x0 = state_setup.x0
    x0_by_rhs = state_setup.x0_by_rhs
    state_x_by_rhs = state_setup.state_x_by_rhs
    rhs_setup = resolve_transport_which_rhs_setup(rhs_mode=int(op0.rhs_mode), which_rhs_values=which_rhs_values)
    rhs_mode = int(rhs_setup.rhs_mode)
    n = int(rhs_setup.n_rhs)
    which_rhs_values = rhs_setup.which_rhs_values
    subset_mode = bool(rhs_setup.subset_mode)
    parallel_request = resolve_transport_parallel_request(
        which_rhs_count=len(which_rhs_values),
        n_rhs=int(n),
        parallel_workers=parallel_workers,
        parallel_backend=_transport_parallel_backend(),
        visible_gpu_ids=_transport_parallel_visible_gpu_ids,
    )
    parallel_child = bool(parallel_request.parallel_child)
    parallel_workers = int(parallel_request.parallel_workers)
    parallel_backend = str(parallel_request.parallel_backend)

    parallel_result = maybe_run_transport_parallel_solve(
        nml=nml,
        op0=op0,
        rhs_mode=int(rhs_mode),
        n_rhs=int(n),
        which_rhs_values=which_rhs_values,
        parallel_child=bool(parallel_child),
        parallel_workers=int(parallel_workers),
        parallel_backend=parallel_backend,
        input_namelist=input_namelist,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=solve_method,
        identity_shift=float(identity_shift),
        collect_transport_output_fields=bool(collect_transport_output_fields),
        phi1_hat_base=phi1_hat_base,
        differentiable=differentiable,
        runtime=TransportParallelSolveRuntime(
            run_gpu_subprocesses=_run_transport_parallel_gpu_subprocesses,
            persistent_pool_enabled=_transport_parallel_persistent_pool_enabled(),
            get_pool=_get_transport_parallel_pool,
            shutdown_pool=_shutdown_transport_parallel_pool,
            worker=_transport_parallel_worker,
            worker_env=_transport_parallel_worker_env,
            executor_class=concurrent.futures.ProcessPoolExecutor,
            executor_kwargs=_transport_parallel_pool_executor_kwargs,
            elapsed_s=t_all.elapsed_s,
        ),
        emit=emit,
    )
    if parallel_result is not None:
        return parallel_result
    if emit is not None:
        emit(1, f"solve_v3_transport_matrix_linear_gmres: rhs_mode={rhs_mode} whichRHS_count={n} total_size={int(op0.total_size)}")
        emit(
            0,
            "solve_v3_transport_matrix_linear_gmres: ETA becomes available after the first completed whichRHS solve. "
            "The first solve may include one-time JIT compilation, so later solves can be faster.",
        )

    transport_geom_scheme = transport_geometry_scheme_from_namelist(nml)
    active_dense_setup = resolve_transport_active_dense_setup(
        op=op0,
        rhs_mode=int(rhs_mode),
        n_rhs=int(n),
        solve_method=str(solve_method),
        restart=int(restart),
        maxiter=maxiter,
        backend=jax.default_backend(),
        geometry_scheme=int(transport_geom_scheme),
        dense_accelerator_auto_allowed=_transport_dense_accelerator_auto_allowed(
            op0,
            geometry_scheme=int(transport_geom_scheme),
        ),
        dense_backend_policy_allowed=_transport_dense_backend_allowed(),
        state_out_requested=bool(state_out_env),
        force_stream_diagnostics=force_stream_diagnostics,
        force_store_state=force_store_state,
        subset_mode=bool(subset_mode),
        active_dof_indices=_transport_active_dof_indices,
    )
    if emit is not None:
        for level, message in active_dense_setup.initial_notes:
            emit(int(level), message)
    low_memory_outputs = bool(active_dense_setup.low_memory_outputs)
    stream_diagnostics = bool(active_dense_setup.stream_diagnostics)
    store_state_vectors = bool(active_dense_setup.store_state_vectors)
    solve_method_use = str(active_dense_setup.solve_method_use)
    dense_retry_max = int(active_dense_setup.dense_retry_max)
    dense_mem_block = bool(active_dense_setup.dense_mem_block)
    dense_use_mixed = bool(active_dense_setup.dense_use_mixed)
    dense_backend_allowed = bool(active_dense_setup.dense_backend_allowed)
    gmres_restart = int(active_dense_setup.gmres_restart)
    maxiter = active_dense_setup.maxiter

    use_implicit = _resolve_use_implicit(differentiable=differentiable)
    transport_precondition_side = _transport_precondition_side(op=op0, use_implicit=bool(use_implicit))
    if emit is not None and transport_precondition_side != "left":
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: transport preconditioner side="
            f"{transport_precondition_side}",
        )
    distributed_axis = _resolve_distributed_gmres_axis(op=op0, emit=emit)

    use_solver_jit = _use_solver_jit(int(op0.total_size))
    transport_linear_context = TransportLinearSolveContext(
        rhs_mode=int(rhs_mode),
        size_hint=int(op0.total_size),
        use_implicit=bool(use_implicit),
        use_solver_jit=bool(use_solver_jit),
        distributed_axis=distributed_axis,
    )

    def _dense_dtype(dtype_in: jnp.dtype) -> jnp.dtype:
        return jnp.float32 if dense_use_mixed else dtype_in

    def _solver_kind(method: str) -> tuple[str, str]:
        return _transport_solver_kind(method, rhs_mode=int(rhs_mode))

    def _restart_for_method(method: str) -> int:
        return _transport_restart_for_method(
            method,
            rhs_mode=int(rhs_mode),
            gmres_restart=int(gmres_restart),
            restart=int(restart),
        )

    def _solve_linear(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val=None,
        precondition_side_val: str = "left",
    ):
        return _solve_transport_linear(
            context=transport_linear_context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            preconditioner_val=preconditioner_val,
            precondition_side_val=precondition_side_val,
        )

    def _solve_linear_with_residual(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val=None,
        precondition_side_val: str = "left",
    ) -> tuple[GMRESSolveResult, jnp.ndarray]:
        return _solve_transport_linear_with_residual(
            context=transport_linear_context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            preconditioner_val=preconditioner_val,
            precondition_side_val=precondition_side_val,
        )

    if emit is not None:
        for level, message in (*active_dense_setup.active_notes, *active_dense_setup.dense_notes):
            emit(int(level), message)
    use_active_dof_mode = bool(active_dense_setup.use_active_dof_mode)
    active_idx_np = active_dense_setup.active_idx_np
    active_idx_jnp = active_dense_setup.active_idx_jnp
    full_to_active_jnp = active_dense_setup.full_to_active_jnp
    active_size = int(active_dense_setup.active_size)
    dense_mem_block = bool(active_dense_setup.dense_mem_block)
    dense_use_mixed = bool(active_dense_setup.dense_use_mixed)
    solve_method_use = str(active_dense_setup.solve_method_use)
    dense_precond_enabled = bool(active_dense_setup.dense_precond_enabled)
    dense_precond_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_precond_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}

    reduce_full = None
    expand_reduced = None
    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return v_full[active_idx_jnp]

        def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
            z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
            padded = jnp.concatenate([z0, v_reduced], axis=0)
            return padded[full_to_active_jnp]

    transport_precond_kind = normalize_transport_preconditioner_kind(
        env_value=os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND", "")
    )
    preconditioner_full = None
    preconditioner_reduced = None
    strong_precond_kind: str | None = None
    default_solver_kind = _solver_kind(solve_method_use)[0]
    precond_kind_used: str | None = None
    sparse_jax_config = transport_sparse_jax_config_from_env()
    dd_config = transport_dd_config_from_env(op=op0)
    transport_precond_context = TransportPreconditionerContext(
        op=op0,
        active_size=int(active_size),
        use_active_dof_mode=bool(use_active_dof_mode),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_indices_np=active_idx_np,
        emit=emit,
    )
    transport_precond_builders = TransportPreconditionerDispatchBuilders(
        collision_builder=_build_rhsmode23_collision_preconditioner,
        sxblock_builder=_build_rhsmode23_sxblock_preconditioner,
        block_builder=_build_rhsmode23_block_preconditioner,
        xmg_builder=_build_rhsmode23_xmg_preconditioner,
        theta_dd_builder=_build_rhsmode23_theta_dd_preconditioner,
        theta_schwarz_builder=_build_rhsmode23_theta_schwarz_preconditioner,
        zeta_dd_builder=_build_rhsmode23_zeta_dd_preconditioner,
        zeta_schwarz_builder=_build_rhsmode23_zeta_schwarz_preconditioner,
        tzfft_builder=_build_rhsmode23_tzfft_preconditioner,
        sparse_jax_builder=_build_sparse_jax_preconditioner_from_matvec,
        sparse_jax_cache_key=_transport_precond_cache_key,
        apply_operator_cached=apply_v3_full_system_operator_cached,
        precond_dtype=_precond_dtype,
        fp_tzfft_builder=_build_rhsmode23_fp_tzfft_preconditioner,
        fp_tzfft_line_builder=_build_rhsmode23_fp_tzfft_line_preconditioner,
        fp_tzfft_line_schur_builder=_build_rhsmode23_fp_tzfft_line_schur_preconditioner,
        fp_local_geom_line_builder=_build_rhsmode23_fp_local_geom_line_preconditioner,
        fp_xblock_tz_lu_builder=_build_rhsmode23_fp_xblock_tz_lu_preconditioner,
        fp_xblock_tz_lu_schur_builder=_build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner,
        fp_structured_fblock_lu_builder=_build_rhsmode23_fp_structured_fblock_lu_preconditioner,
        fp_fortran_reduced_lu_builder=_build_rhsmode23_fp_fortran_reduced_lu_preconditioner,
        fp_direct_active_block_schur_builder=_build_rhsmode23_fp_direct_active_block_schur_preconditioner,
    )
    structured_tzfft_size = int(active_size) if use_active_dof_mode else int(op0.total_size)
    structured_tzfft_first_auto = _transport_tzfft_structured_first_attempt_allowed(
        op0,
        size=int(structured_tzfft_size),
        use_implicit=bool(use_implicit),
    )
    tzfft_backend_allowed = (
        _transport_tzfft_backend_allowed()
        or _transport_tzfft_accelerator_auto_allowed(op0)
        or bool(structured_tzfft_first_auto)
    )
    if structured_tzfft_first_auto and emit is not None:
        method_tz, restart_tz, maxiter_tz = _transport_tzfft_first_attempt_budget(
            restart=int(gmres_restart),
            maxiter=maxiter,
        )
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt enabled "
            f"(size={int(structured_tzfft_size)} method={method_tz} "
            f"restart={int(restart_tz)} maxiter={int(maxiter_tz)})",
        )
    if transport_precond_kind is not None and int(rhs_mode) in {2, 3}:
        precond_kind_used, strong_precond_kind = resolve_transport_preconditioner_choice(
            op=op0,
            transport_precond_kind=transport_precond_kind,
            default_solver_kind=default_solver_kind,
            parallel_workers=int(parallel_workers),
            dense_mem_block=bool(dense_mem_block),
            tzfft_backend_allowed=bool(tzfft_backend_allowed),
            shard_axis=_matvec_shard_axis(op0),
            backend=jax.default_backend(),
            emit=emit,
        )
        transport_precondition_side, side_changed = resolve_transport_precondition_side_for_kind(
            kind=precond_kind_used,
            requested_side=transport_precondition_side,
        )
        if side_changed and emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: FP line-factor preconditioner uses left "
                "preconditioning; overriding requested right preconditioning",
            )
        if precond_kind_used is not None:
            preconditioner_full = build_transport_preconditioner_from_kind(
                kind=precond_kind_used,
                context=transport_precond_context,
                builders=transport_precond_builders,
                dd_config=dd_config,
                sparse_jax_config=sparse_jax_config,
                use_reduced=False,
            )
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = build_transport_preconditioner_from_kind(
                    kind=precond_kind_used,
                    context=transport_precond_context,
                    builders=transport_precond_builders,
                    dd_config=dd_config,
                    sparse_jax_config=sparse_jax_config,
                    use_reduced=True,
                )
        if emit is not None and precond_kind_used is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: preconditioner="
                f"{precond_kind_used} strong={strong_precond_kind}",
            )

    strong_preconditioner_full = None
    strong_preconditioner_reduced = None

    transport_sparse_drop_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "").strip()
    transport_sparse_drop_rel_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "").strip()
    try:
        transport_sparse_drop_tol = float(transport_sparse_drop_tol_env) if transport_sparse_drop_tol_env else 0.0
    except ValueError:
        transport_sparse_drop_tol = 0.0
    try:
        transport_sparse_drop_rel = float(transport_sparse_drop_rel_env) if transport_sparse_drop_rel_env else 0.0
    except ValueError:
        transport_sparse_drop_rel = 0.0

    def _get_strong_preconditioner(use_reduced: bool) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
        nonlocal strong_preconditioner_full, strong_preconditioner_reduced
        if strong_precond_kind is None:
            return None
        if use_reduced:
            if strong_preconditioner_reduced is None:
                strong_preconditioner_reduced = build_transport_strong_preconditioner_from_kind(
                    kind=strong_precond_kind,
                    use_reduced=True,
                    precond_kind_used=precond_kind_used,
                    preconditioner_full=preconditioner_full,
                    preconditioner_reduced=preconditioner_reduced,
                    context=transport_precond_context,
                    builders=transport_precond_builders,
                    dd_config=dd_config,
                    sparse_jax_config=sparse_jax_config,
                )
            return strong_preconditioner_reduced
        if strong_preconditioner_full is None:
            strong_preconditioner_full = build_transport_strong_preconditioner_from_kind(
                kind=strong_precond_kind,
                use_reduced=False,
                precond_kind_used=precond_kind_used,
                preconditioner_full=preconditioner_full,
                preconditioner_reduced=preconditioner_reduced,
                context=transport_precond_context,
                builders=transport_precond_builders,
                dd_config=dd_config,
                sparse_jax_config=sparse_jax_config,
            )
        return strong_preconditioner_full

    # RHSMode=2/3 transport reuses the same active operator for multiple drives,
    # so keep sparse-helper factors scoped to this solve and reuse them across RHS.
    transport_sparse_direct_context = TransportSparseDirectContext(
        op=op0,
        factor_cache={},
        pattern_cache={},
        sparse_drop_tol=float(transport_sparse_drop_tol),
        sparse_drop_rel=float(transport_sparse_drop_rel),
        emit=emit,
        sparse_factor_cache_key=_sparse_factor_cache_key,
        hash_numpy_array_for_cache=_hash_numpy_array_for_cache,
        build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
        build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
        try_build_direct_active_operator_bundle=_try_build_rhsmode23_fp_direct_active_operator_bundle,
        host_sparse_direct_solve_with_refinement=_host_sparse_direct_solve_with_refinement,
        host_sparse_direct_refine_steps=_host_sparse_direct_refine_steps,
        host_sparse_direct_polish=_host_sparse_direct_polish,
        sparse_factor_dtype=_transport_sparse_factor_dtype,
        sparse_direct_use_explicit_helper=_transport_sparse_direct_use_explicit_helper,
        sparse_direct_needs_float64_retry=_transport_sparse_direct_needs_float64_retry,
    )

    def _transport_sparse_direct_pattern_for_solve(*, n: int, active_indices_np: np.ndarray | None):
        return _transport_sparse_direct_pattern_for_context(
            context=transport_sparse_direct_context,
            n=int(n),
            active_indices_np=active_indices_np,
        )

    def _transport_sparse_direct_solve(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        n: int,
        dtype: jnp.dtype,
        cache_key: tuple[object, ...],
        active_indices_np: np.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precondition_side_val: str,
    ) -> GMRESSolveResult:
        return _transport_sparse_direct_solve_with_context(
            context=transport_sparse_direct_context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            active_indices_np=active_indices_np,
            tol_val=float(tol_val),
            atol_val=float(atol_val),
            restart_val=int(restart_val),
            maxiter_val=maxiter_val,
            precondition_side_val=str(precondition_side_val),
        )

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    solver_kinds_by_rhs: dict[int, str] = {}
    solve_methods_by_rhs: dict[int, str] = {}
    elapsed_s = np.zeros((n,), dtype=np.float64)
    op_rhs_by_index = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in which_rhs_values]
    rhs_by_index = [rhs_v3_full_system_jit(op_rhs) for op_rhs in op_rhs_by_index]
    rhs_norms: dict[int, jnp.ndarray] = {
        int(which_rhs): jnp.linalg.norm(rhs_by_index[idx])
        for idx, which_rhs in enumerate(which_rhs_values)
    }
    abort_max_residual, abort_max_relative_residual = transport_residual_gate_thresholds_from_env()
    transport_loop_progress = TransportLoopProgress(
        which_rhs_values=which_rhs_values,
        rhs_norms=rhs_norms,
        residual_norms=residual_norms,
        elapsed_s=elapsed_s,
        abort_max_residual=float(abort_max_residual),
        abort_max_relative_residual=float(abort_max_relative_residual),
        emit=emit,
    )

    use_op_rhs_in_matvec = bool(op0.include_phi1_in_kinetic)
    env_transport_matvec = os.environ.get("SFINCS_JAX_TRANSPORT_MATVEC_MODE", "").strip().lower()
    if env_transport_matvec == "rhs":
        use_op_rhs_in_matvec = True
    elif env_transport_matvec == "base":
        use_op_rhs_in_matvec = False
    op_matvec_by_index = [op_rhs if use_op_rhs_in_matvec else op0 for op_rhs in op_rhs_by_index]

    env_diag_op = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_OP", "").strip().lower()
    use_diag_op0 = env_diag_op != "rhs"
    diag_op_by_index = op_rhs_by_index if not use_diag_op0 else None

    transport_output_fields: dict[str, np.ndarray] | None = None
    collect_full_transport_outputs = bool(collect_transport_output_fields)
    streaming_outputs: TransportStreamingOutputAccumulator | None = None
    if stream_diagnostics:
        streaming_outputs = TransportStreamingOutputAccumulator.create(
            nml=nml,
            grids=grids,
            geom=geom,
            op0=op0,
            n_rhs=n,
            collect_full_output_fields=collect_full_transport_outputs,
        )

        def _collect_transport_outputs(which_rhs: int, x_full: jnp.ndarray) -> None:
            """Populate streaming diagnostics for a single whichRHS solve."""
            assert streaming_outputs is not None
            streaming_outputs.collect(int(which_rhs), x_full)

    transport_matvec_cache = TransportMatvecCache(
        use_active_dof_mode=bool(use_active_dof_mode),
        active_size=int(active_size),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    _get_full_matvec = transport_matvec_cache.get_full
    _get_reduced_matvec = transport_matvec_cache.get_reduced

    recycle_k = resolve_transport_recycle_k(
        op=op0,
        use_implicit=bool(use_implicit),
        op_matvec_by_index=op_matvec_by_index,
        disable_auto_recycle=_transport_disable_auto_recycle,
        emit=emit,
    )
    recycle_state = TransportRecycleState(k=int(recycle_k))
    state_recycle_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_STATE", "").strip().lower()
    state_recycle_enabled = state_recycle_env not in {"0", "false", "no", "off"}
    if recycle_k > 0 and state_recycle_enabled and state_x_by_rhs:
        recycle_state.seed_from_state(
            state_x_by_rhs=state_x_by_rhs,
            total_size=int(op0.total_size),
            active_size=int(active_size),
            matvec_cache=transport_matvec_cache,
            op_ref=op_matvec_by_index[0],
        )

    def _residual_value(res: GMRESSolveResult) -> float:
        return transport_residual_value(res)

    def _needs_retry(res: GMRESSolveResult, target: float) -> bool:
        return transport_result_needs_retry(
            res,
            float(target),
            result_is_finite=_gmres_result_is_finite,
        )

    per_rhs_loop_policy = resolve_transport_per_rhs_loop_policy(op=op0, rhs_mode=int(rhs_mode))

    def _maybe_project_constraint_nullspace(
        x_vec: jnp.ndarray,
        *,
        which_rhs: int,
        op_matvec: V3FullSystemOperator,
        rhs_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        if not per_rhs_loop_policy.projection_candidate(int(which_rhs)):
            return x_vec
        return _project_constraint_scheme1_nullspace_solution(
            op=op0,
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=op_matvec,
            enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        )

    dense_batch_done = False
    dense_batch_fallback_enabled = bool(per_rhs_loop_policy.dense_batch_fallback_enabled)
    transport_rhs_finalization_context = TransportRHSFinalizationContext(
        state_vectors=state_vectors,
        residual_norms=residual_norms,
        solver_kinds_by_rhs=solver_kinds_by_rhs,
        solve_methods_by_rhs=solve_methods_by_rhs,
        store_state_vectors=bool(store_state_vectors),
        stream_diagnostics=bool(stream_diagnostics),
        collect_transport_outputs=_collect_transport_outputs if stream_diagnostics else None,
        recycle_state=recycle_state if recycle_k > 0 else None,
        apply_operator=apply_v3_full_system_operator_cached,
        emit_iteration_stats=_emit_transport_ksp_iteration_stats,
        emit=emit,
        iter_stats_enabled=bool(per_rhs_loop_policy.iter_stats_enabled),
        iter_stats_max_size=per_rhs_loop_policy.iter_stats_max_size,
        atol=float(atol), maxiter=maxiter, precond_side=transport_precondition_side,
    )

    def _dense_batch_solve_all(*, op_probe_ref: V3FullSystemOperator, reason: str) -> bool:
        dense_batch_context = TransportDenseBatchContext(
            dense_backend_allowed=bool(dense_backend_allowed),
            dense_use_mixed=bool(dense_use_mixed),
            use_active_dof_mode=bool(use_active_dof_mode),
            active_size=int(active_size),
            op0=op0,
            op_matvec_by_index=op_matvec_by_index,
            rhs_by_index=rhs_by_index,
            which_rhs_values=which_rhs_values,
            rhs_norms=rhs_norms,
            residual_norms=residual_norms,
            solver_kinds_by_rhs=solver_kinds_by_rhs,
            solve_methods_by_rhs=solve_methods_by_rhs,
            elapsed_s=elapsed_s,
            state_vectors=state_vectors,
            store_state_vectors=bool(store_state_vectors),
            stream_diagnostics=bool(stream_diagnostics),
            rhs3_krylov_flags=per_rhs_loop_policy.rhs3_krylov_flags,
            maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
            collect_transport_outputs=_collect_transport_outputs if stream_diagnostics else None,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            emit=emit,
        )
        return _solve_transport_dense_batch(
            context=dense_batch_context,
            op_probe_ref=op_probe_ref,
            reason=reason,
        )

    if str(solve_method_use).lower() == "dense":
        op_probe_ref = op_matvec_by_index[0]
        if _dense_batch_solve_all(op_probe_ref=op_probe_ref, reason="auto dense"):
            dense_batch_done = True

    if not dense_batch_done:
        for idx, which_rhs in enumerate(which_rhs_values):
            t_rhs = Timer()
            op_rhs = op_rhs_by_index[idx]
            rhs = rhs_by_index[idx]
            op_matvec = op_matvec_by_index[idx]
            if emit is not None:
                emit(0, f"whichRHS={which_rhs}/{n}: assembling+solving (rhs_norm={float(jnp.linalg.norm(rhs)):.6e})")
                emit(1, f"whichRHS={which_rhs}/{n}: evaluateJacobian called (matrix-free)")

            use_loose_epar_krylov, force_epar_krylov = per_rhs_loop_policy.rhs3_krylov_flags(which_rhs)
            solve_method_rhs = solve_method_use
            tol_rhs = tol
            if force_epar_krylov or use_loose_epar_krylov:
                solve_method_rhs = "incremental"
                if use_loose_epar_krylov:
                    epar_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_TOL", "").strip()
                    try:
                        epar_tol = float(epar_tol_env) if epar_tol_env else 1e-8
                    except ValueError:
                        epar_tol = 1e-8
                    tol_rhs = max(float(tol), float(epar_tol))

            if use_active_dof_mode:
                assert active_idx_jnp is not None
                assert full_to_active_jnp is not None
                assert reduce_full is not None
                assert expand_reduced is not None
                mv_reduced = _get_reduced_matvec(op_matvec)

                rhs_reduced = reduce_full(rhs)
                preconditioner_use = preconditioner_reduced
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv_reduced,
                        n=active_size,
                        dtype=_dense_dtype(rhs_reduced.dtype),
                        cache=dense_precond_cache_reduced,
                        key=(sig, int(active_size)),
                    )
                x0_reduced = None
                x0_local = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if x0_local is not None:
                    x0_arr = jnp.asarray(x0_local)
                    if x0_arr.shape == (active_size,):
                        x0_reduced = x0_arr
                    elif x0_arr.shape == (op0.total_size,):
                        x0_reduced = reduce_full(x0_arr)
                if recycle_k > 0:
                    x0_recycled = recycle_state.candidate_reduced(rhs_reduced)
                    if x0_reduced is None and x0_recycled is not None:
                        x0_reduced = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_reduced
                dense_used = False
                structured_tzfft_first_attempt = False
                initial_solve_method_rhs = solve_method_rhs
                initial_restart_used = _restart_for_method(solve_method_rhs)
                initial_maxiter = maxiter
                if (
                    structured_tzfft_first_auto
                    and precond_kind_used == "tzfft"
                    and preconditioner_use is not None
                    and str(solve_method_rhs).strip().lower()
                    in {"auto", "default", "batched", "bicgstab", "bicgstab_jax", "incremental"}
                ):
                    structured_tzfft_first_attempt = True
                    initial_solve_method_rhs, initial_restart_used, initial_maxiter = (
                        _transport_tzfft_first_attempt_budget(
                            restart=int(gmres_restart),
                            maxiter=maxiter,
                        )
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = initial_solve_method_rhs
                    restart_used = int(initial_restart_used)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt "
                            f"whichRHS={int(which_rhs)} size={int(active_size)} "
                            f"restart={int(initial_restart_used)} maxiter={int(initial_maxiter)}",
                        )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs_reduced)))
                host_gmres_first_attempt = _transport_host_gmres_first_attempt_allowed(
                    op=op0,
                    size=int(active_size),
                    use_implicit=bool(use_implicit),
                )
                sparse_direct_first_attempt = _transport_sparse_direct_first_attempt_allowed(
                    op=op0,
                    size=int(active_size),
                    use_implicit=bool(use_implicit),
                )
                if host_gmres_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt "
                            f"(size={int(active_size)} backend={jax.default_backend()})",
                        )
                    try:
                        res_reduced, residual_vec = _transport_host_gmres_solve(
                            op=op0,
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            preconditioner_fn=preconditioner_use,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                            emit=emit,
                            which_rhs=int(which_rhs),
                            progress_every=_transport_host_gmres_progress_every(),
                        )
                        solver_kind_used = "gmres_scipy"
                        solve_method_used = "incremental"
                        restart_used = initial_restart_used
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res_reduced, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                elif sparse_direct_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt "
                            f"(size={int(active_size)} backend={jax.default_backend()})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_reduced = _transport_sparse_direct_solve(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            n=int(active_size),
                            dtype=rhs_reduced.dtype,
                            cache_key=("transport_sparse_lu", sig, int(active_size), "active"),
                            active_indices_np=active_idx_np,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        solver_kind_used = "sparse_lu"
                        solve_method_used = "sparse_lu"
                        restart_used = 0
                        preconditioner_used = None
                        x0_used = None
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res_reduced = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=x0_reduced,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                else:
                    res_reduced = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=initial_restart_used,
                        maxiter_val=initial_maxiter,
                        solve_method_val=initial_solve_method_rhs,
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                solver_kind = _solver_kind(initial_solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res_reduced) or float(res_reduced.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_reduced = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                sparse_direct_rescue = _transport_sparse_direct_rescue_allowed(
                    op=op0,
                    size=int(active_size),
                    residual_norm=float(res_reduced.residual_norm),
                    target=float(target_rhs),
                    use_implicit=bool(use_implicit),
                )
                if structured_tzfft_first_attempt and _needs_retry(res_reduced, target_rhs):
                    sparse_direct_rescue = sparse_direct_rescue or _transport_sparse_direct_rescue_allowed(
                        op=op0,
                        size=int(active_size),
                        residual_norm=float("nan"),
                        target=float(target_rhs),
                        use_implicit=bool(use_implicit),
                    )
                sparse_direct_rescue_first = _transport_sparse_direct_rescue_first(
                    sparse_direct_rescue=sparse_direct_rescue,
                )
                if sparse_direct_rescue_first and emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: sparse LU rescue-first "
                        "auto mode -> defer transport retry branches",
                    )
                if _needs_retry(res_reduced, target_rhs) and preconditioner_use is not None and (not sparse_direct_rescue_first):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val=transport_precondition_side,
                    )
                    if _residual_value(res_retry) < _residual_value(res_reduced):
                        res_reduced = res_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res_reduced, target_rhs) and (not sparse_direct_rescue_first):
                    strong_precond = _get_strong_preconditioner(True)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=res_reduced.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_strong) < _residual_value(res_reduced):
                            res_reduced = res_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res_reduced, target_rhs) and sparse_direct_rescue:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_sparse = _transport_sparse_direct_solve(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            n=int(active_size),
                            dtype=rhs_reduced.dtype,
                            cache_key=("transport_sparse_lu", sig, int(active_size), "active"),
                            active_indices_np=active_idx_np,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method(solve_method_rhs),
                            maxiter_val=maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_sparse) < _residual_value(res_reduced):
                            res_reduced = res_sparse
                            preconditioner_use = None
                            preconditioner_used = None
                            solver_kind_used = "sparse_lu"
                            solve_method_used = "sparse_lu"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                if _needs_retry(res_reduced, target_rhs) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv_reduced,
                            n=int(active_size),
                            dtype=_dense_dtype(rhs_reduced.dtype),
                            cache=dense_solver_cache_reduced,
                            key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                        r_dense = rhs_reduced - mv_reduced(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))
                        if _residual_value(res_dense) < _residual_value(res_reduced):
                            res_reduced = res_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                polish_config = transport_polish_config_from_env(
                    rhs_mode=int(rhs_mode),
                    residual_norm=_residual_value(res_reduced),
                    target=float(target_rhs),
                    gmres_restart=int(gmres_restart),
                    maxiter=maxiter,
                )
                if _needs_retry(res_reduced, target_rhs) and polish_config.enabled:
                    polish_precond = _get_strong_preconditioner(True)
                    if polish_precond is None:
                        polish_precond = preconditioner_use
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: polish solve for RHSMode=3 "
                            f"(residual={float(res_reduced.residual_norm):.3e} > "
                            f"max({polish_config.ratio:.1f}x target, {polish_config.abs_tol:.1e}), "
                            f"restart={polish_config.restart} maxiter={polish_config.maxiter})",
                        )
                    res_polish = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=res_reduced.x,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=int(polish_config.restart),
                        maxiter_val=int(polish_config.maxiter),
                        solve_method_val="incremental",
                        preconditioner_val=polish_precond,
                        precondition_side_val=transport_precondition_side,
                    )
                    if transport_candidate_is_better(candidate=res_polish, current=res_reduced):
                        res_reduced = res_polish
                        preconditioner_used = polish_precond
                        solver_kind_used = "gmres"
                        solve_method_used = "incremental"
                        restart_used = int(polish_config.restart)
                x_full = expand_reduced(res_reduced.x)
                x_full = _maybe_project_constraint_nullspace(
                    x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                )
                ax_full = apply_v3_full_system_operator_cached(op_matvec, x_full)
                res_norm_full = jnp.linalg.norm(ax_full - rhs)
                if (not dense_used) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    target_full = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                    if float(res_norm_full) > target_full:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback (true residual) "
                                f"(size={int(active_size)} residual={float(res_norm_full):.3e} > target={target_full:.3e})",
                            )
                        try:
                            sig = _operator_signature_cached(op_matvec)
                            dense_solver = _dense_solver_for_matvec(
                                matvec_fn=mv_reduced,
                                n=int(active_size),
                                dtype=_dense_dtype(rhs_reduced.dtype),
                                cache=dense_solver_cache_reduced,
                                key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                            )
                            rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                            x_dense = dense_solver(rhs_dense)
                            if dense_use_mixed:
                                r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                                dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                                x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                            x_full_dense = expand_reduced(x_dense)
                            x_full_dense = _maybe_project_constraint_nullspace(
                                x_full_dense, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                            )
                            ax_dense = apply_v3_full_system_operator_cached(op_matvec, x_full_dense)
                            res_dense_norm = jnp.linalg.norm(ax_dense - rhs)
                            if float(res_dense_norm) < float(res_norm_full):
                                x_full = x_full_dense
                                ax_full = ax_dense
                                res_norm_full = res_dense_norm
                                dense_used = True
                                solver_kind_used = "dense"
                                solve_method_used = "dense"
                        except Exception as exc:  # noqa: BLE001
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                    f"({type(exc).__name__}: {exc})",
                                )
                if (
                    dense_used
                    and dense_batch_fallback_enabled
                    and (not dense_batch_done)
                    and dense_retry_max > 0
                    and int(active_size) <= int(dense_retry_max)
                ):
                    if _dense_batch_solve_all(op_probe_ref=op_matvec_by_index[0], reason="dense fallback"):
                        dense_batch_done = True
                        break
                finalize_reduced_transport_rhs(
                    context=transport_rhs_finalization_context,
                    which_rhs=int(which_rhs),
                    result=res_reduced,
                    rhs_full=rhs,
                    op_matvec=op_matvec,
                    solver_kind=str(solver_kind_used),
                    solve_method=str(solve_method_used),
                    dense_used=bool(dense_used),
                    expand_reduced=expand_reduced,
                    reduce_full=reduce_full,
                    maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
                    ksp_request=transport_rhs_finalization_context.ksp_request(
                        mv_reduced,
                        rhs_reduced,
                        preconditioner_used,
                        x0_used,
                        tol_val=float(tol_rhs),
                        restart_val=int(restart_used),
                        solver_kind=str(solver_kind_used),
                    ),
                    accepted_x_full=x_full,
                    accepted_ax_full=ax_full,
                    accepted_residual_norm=res_norm_full,
                )
            else:
                mv = _get_full_matvec(op_matvec)

                preconditioner_use = preconditioner_full
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv,
                        n=int(op0.total_size),
                        dtype=_dense_dtype(rhs.dtype),
                        cache=dense_precond_cache_full,
                        key=(sig, int(op0.total_size)),
                    )
                x0_full = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if recycle_k > 0:
                    x0_recycled = recycle_state.candidate_full(rhs)
                    if x0_full is None and x0_recycled is not None:
                        x0_full = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_full
                dense_used = False
                structured_tzfft_first_attempt = False
                initial_solve_method_rhs = solve_method_rhs
                initial_restart_used = _restart_for_method(solve_method_rhs)
                initial_maxiter = maxiter
                if (
                    structured_tzfft_first_auto
                    and precond_kind_used == "tzfft"
                    and preconditioner_use is not None
                    and str(solve_method_rhs).strip().lower()
                    in {"auto", "default", "batched", "bicgstab", "bicgstab_jax", "incremental"}
                ):
                    structured_tzfft_first_attempt = True
                    initial_solve_method_rhs, initial_restart_used, initial_maxiter = (
                        _transport_tzfft_first_attempt_budget(
                            restart=int(gmres_restart),
                            maxiter=maxiter,
                        )
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = initial_solve_method_rhs
                    restart_used = int(initial_restart_used)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: structured tzfft first attempt "
                            f"whichRHS={int(which_rhs)} size={int(op0.total_size)} "
                            f"restart={int(initial_restart_used)} maxiter={int(initial_maxiter)}",
                        )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                host_gmres_first_attempt = _transport_host_gmres_first_attempt_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    use_implicit=bool(use_implicit),
                )
                sparse_direct_first_attempt = _transport_sparse_direct_first_attempt_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    use_implicit=bool(use_implicit),
                )
                if host_gmres_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt "
                            f"(size={int(op0.total_size)} backend={jax.default_backend()})",
                        )
                    try:
                        res, residual_vec = _transport_host_gmres_solve(
                            op=op0,
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            preconditioner_fn=preconditioner_use,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                            emit=emit,
                            which_rhs=int(which_rhs),
                            progress_every=_transport_host_gmres_progress_every(),
                        )
                        solver_kind_used = "gmres_scipy"
                        solve_method_used = "incremental"
                        restart_used = initial_restart_used
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host SciPy GMRES first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                elif sparse_direct_first_attempt:
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt "
                            f"(size={int(op0.total_size)} backend={jax.default_backend()})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res = _transport_sparse_direct_solve(
                            matvec_fn=mv,
                            b_vec=rhs,
                            n=int(op0.total_size),
                            dtype=rhs.dtype,
                            cache_key=("transport_sparse_lu", sig, int(op0.total_size), "full"),
                            active_indices_np=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        residual_vec = None
                        solver_kind_used = "sparse_lu"
                        solve_method_used = "sparse_lu"
                        restart_used = 0
                        preconditioner_used = None
                        x0_used = None
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: host sparse LU first attempt failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                        res, residual_vec = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=x0_full,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=initial_restart_used,
                            maxiter_val=initial_maxiter,
                            solve_method_val=initial_solve_method_rhs,
                            preconditioner_val=preconditioner_use,
                            precondition_side_val=transport_precondition_side,
                        )
                else:
                    res, residual_vec = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=initial_restart_used,
                        maxiter_val=initial_maxiter,
                        solve_method_val=initial_solve_method_rhs,
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                solver_kind = _solver_kind(initial_solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res) or float(res.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res, residual_vec = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val=transport_precondition_side,
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                sparse_direct_rescue = _transport_sparse_direct_rescue_allowed(
                    op=op0,
                    size=int(op0.total_size),
                    residual_norm=float(res.residual_norm),
                    target=float(target_rhs),
                    use_implicit=bool(use_implicit),
                )
                if structured_tzfft_first_attempt and _needs_retry(res, target_rhs):
                    sparse_direct_rescue = sparse_direct_rescue or _transport_sparse_direct_rescue_allowed(
                        op=op0,
                        size=int(op0.total_size),
                        residual_norm=float("nan"),
                        target=float(target_rhs),
                        use_implicit=bool(use_implicit),
                    )
                sparse_direct_rescue_first = _transport_sparse_direct_rescue_first(
                    sparse_direct_rescue=sparse_direct_rescue,
                )
                if sparse_direct_rescue_first and emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: sparse LU rescue-first "
                        "auto mode -> defer transport retry branches",
                    )
                if _needs_retry(res, target_rhs) and preconditioner_use is not None and (not sparse_direct_rescue_first):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry, residual_retry = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val=transport_precondition_side,
                    )
                    if _residual_value(res_retry) < _residual_value(res):
                        res = res_retry
                        residual_vec = residual_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res, target_rhs) and (not sparse_direct_rescue_first):
                    strong_precond = _get_strong_preconditioner(False)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong, residual_vec_strong = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=res.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_strong) < _residual_value(res):
                            res = res_strong
                            residual_vec = residual_vec_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res, target_rhs) and sparse_direct_rescue:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        res_sparse = _transport_sparse_direct_solve(
                            matvec_fn=mv,
                            b_vec=rhs,
                            n=int(op0.total_size),
                            dtype=rhs.dtype,
                            cache_key=("transport_sparse_lu", sig, int(op0.total_size), "full"),
                            active_indices_np=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method(solve_method_rhs),
                            maxiter_val=maxiter,
                            precondition_side_val=transport_precondition_side,
                        )
                        if _residual_value(res_sparse) < _residual_value(res):
                            res = res_sparse
                            residual_vec = None
                            preconditioner_use = None
                            preconditioner_used = None
                            solver_kind_used = "sparse_lu"
                            solve_method_used = "sparse_lu"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: sparse LU direct rescue failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                if _needs_retry(res, target_rhs) and dense_retry_max > 0 and int(op0.total_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv,
                            n=int(op0.total_size),
                            dtype=_dense_dtype(rhs.dtype),
                            cache=dense_solver_cache_full,
                            key=(sig, int(op0.total_size), str(_dense_dtype(rhs.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs, dtype=_dense_dtype(rhs.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs - mv(jnp.asarray(x_dense, dtype=rhs.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs.dtype) + jnp.asarray(dx, dtype=rhs.dtype)
                        residual_dense = rhs - mv(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_dense))
                        if _residual_value(res_dense) < _residual_value(res):
                            res = res_dense
                            residual_vec = residual_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                polish_config = transport_polish_config_from_env(
                    rhs_mode=int(rhs_mode),
                    residual_norm=_residual_value(res),
                    target=float(target_rhs),
                    gmres_restart=int(gmres_restart),
                    maxiter=maxiter,
                )
                if _needs_retry(res, target_rhs) and polish_config.enabled:
                    polish_precond = _get_strong_preconditioner(False)
                    if polish_precond is None:
                        polish_precond = preconditioner_use
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: polish solve for RHSMode=3 "
                            f"(residual={float(res.residual_norm):.3e} > "
                            f"max({polish_config.ratio:.1f}x target, {polish_config.abs_tol:.1e}), "
                            f"restart={polish_config.restart} maxiter={polish_config.maxiter})",
                        )
                    res_polish, residual_polish = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=res.x,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=int(polish_config.restart),
                        maxiter_val=int(polish_config.maxiter),
                        solve_method_val="incremental",
                        preconditioner_val=polish_precond,
                        precondition_side_val=transport_precondition_side,
                    )
                    if transport_candidate_is_better(candidate=res_polish, current=res):
                        res = res_polish
                        residual_vec = residual_polish
                        preconditioner_used = polish_precond
                        solver_kind_used = "gmres"
                        solve_method_used = "incremental"
                        restart_used = int(polish_config.restart)
                if (
                    dense_used
                    and dense_batch_fallback_enabled
                    and (not dense_batch_done)
                    and dense_retry_max > 0
                    and int(op0.total_size) <= int(dense_retry_max)
                ):
                    if _dense_batch_solve_all(op_probe_ref=op_matvec_by_index[0], reason="dense fallback"):
                        dense_batch_done = True
                        break
                projection_needed = per_rhs_loop_policy.projection_needed(which_rhs)
                finalize_full_transport_rhs(
                    context=transport_rhs_finalization_context,
                    which_rhs=int(which_rhs),
                    result=res,
                    rhs_full=rhs,
                    op_matvec=op_matvec,
                    solver_kind=str(solver_kind_used),
                    solve_method=str(solve_method_used),
                    dense_used=bool(dense_used),
                    projection_needed=bool(projection_needed),
                    residual_vec=residual_vec,
                    maybe_project_constraint_nullspace=_maybe_project_constraint_nullspace,
                    ksp_request=transport_rhs_finalization_context.ksp_request(
                        mv,
                        rhs,
                        preconditioner_used,
                        x0_used,
                        tol_val=float(tol_rhs),
                        restart_val=int(restart_used),
                        solver_kind=str(solver_kind_used),
                    ),
                )
            transport_loop_progress.finish_rhs(
                which_rhs=int(which_rhs),
                rhs_elapsed_s=float(t_rhs.elapsed_s()),
                total_elapsed_s=float(t_all.elapsed_s()),
            )

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")
    postsolve_diagnostics = compute_transport_postsolve_diagnostics(
        op0=op0,
        geom=geom,
        state_vectors=state_vectors,
        which_rhs_values=which_rhs_values,
        stream_diagnostics=bool(stream_diagnostics),
        streaming_outputs=streaming_outputs,
        use_diag_op0=bool(use_diag_op0),
        diag_op_by_index=diag_op_by_index,
        emit=None,
    )
    tm = postsolve_diagnostics.transport_matrix
    diag_pf_jnp = postsolve_diagnostics.particle_flux_vm_psi_hat
    diag_hf_jnp = postsolve_diagnostics.heat_flux_vm_psi_hat
    diag_flow_jnp = postsolve_diagnostics.fsab_flow
    transport_output_fields = postsolve_diagnostics.transport_output_fields
    if state_out_env:
        try:
            from .solver_state import save_krylov_state  # noqa: PLC0415

            save_krylov_state(path=state_out_env, op=op0, x_by_rhs=state_vectors)
        except Exception:
            if emit is not None:
                emit(1, f"solve_v3_transport_matrix_linear_gmres: failed to write state {state_out_env}")
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={t_all.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=diag_flow_jnp,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
        transport_output_fields=transport_output_fields,
        rhs_norms_by_rhs=rhs_norms,
        active_size=int(active_size),
        use_active_dof_mode=bool(use_active_dof_mode),
        solver_kinds_by_rhs=solver_kinds_by_rhs,
        solve_methods_by_rhs=solve_methods_by_rhs,
        preconditioner_kind=precond_kind_used,
        strong_preconditioner_kind=strong_precond_kind,
    )
