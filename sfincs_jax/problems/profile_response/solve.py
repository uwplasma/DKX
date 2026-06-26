from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
import tempfile
import time

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from collections.abc import Callable, Sequence
from typing import Any
import os
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

from sfincs_jax.namelist import Namelist, read_sfincs_input
from sfincs_jax.problems.profile_response.solver_diagnostics import emit_newton_krylov_ksp_history as _emit_newton_krylov_ksp_history
from sfincs_jax.solver import (
    GMRESSolveResult,
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
from sfincs_jax.solver import (
    recycled_initial_guess as _recycled_initial_guess,
    small_regularized_lstsq as _small_regularized_lstsq,
)
from sfincs_jax.structured_velocity import factor_block_tridiagonal
from sfincs_jax.pas_smoother import adaptive_pas_smoother
from sfincs_jax.solvers.explicit_sparse import (
    SparseDecision,
    SparseOperatorBundle,
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    build_operator_from_pattern,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
    estimate_multifrontal_direct_lu_nbytes,
    wrap_sparse_factor_with_coarse_correction,
)
from sfincs_jax.operators.profile_response.device_sparse import device_csr_from_matrix, validate_device_csr_matvec
from sfincs_jax.solvers.preconditioners.domain_decomposition.line_blocks import (  # compatibility exports for legacy tests/debug scripts
    _dd_core_patch_ranges,
    _rhs1_dd_auto_block_size,
    _rhs1_dd_coarse_block_size,
    _rhs1_dd_coarse_block_sizes,
    _rhs1_dd_coarse_level_count,
)
from sfincs_jax.solvers.memory_model import estimate_sparse_pc_memory
from sfincs_jax.solvers.preconditioners.pas.policy import (
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
    rhs1_pas_small_near_zero_er_kind as _rhs1_pas_small_near_zero_er_kind,
    rhs1_pas_tz_guarded_strong_retry_from_env,
    rhs1_pas_tz_max_bytes as _rhs1_pas_tz_max_bytes,
)
from sfincs_jax.solvers.preconditioners.dispatch import (
    build_rhs1_preconditioner_from_kind as _dispatch_rhs1_preconditioner_from_kind,
)
from sfincs_jax.operators.profile_response.kinetic import select_structured_rhs1_fblock_csr_operator, select_structured_rhs1_fblock_operator
from sfincs_jax.operators.profile_response.full_system import (
    build_active_projected_rhs1_full_csr_preconditioner,
    build_direct_active_fortran_v3_reduced_pmat_preconditioner,
    select_active_fortran_v3_reduced_support_mode_preconditioner,
    solve_structured_rhs1_full_csr,
)
from sfincs_jax.operators.profile_response.structured_csr import _try_build_structured_rhs1_full_csr_operator_bundle
from sfincs_jax.problems.profile_response.policies import (
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
from sfincs_jax.problems.profile_response.policies import (
    rhs1_gpu_sparse_fallback_skip_allowed_current_backend as _rhs1_gpu_sparse_fallback_skip_allowed,
)
from sfincs_jax.solvers.preconditioners.schur.profile_response import resolve_rhs1_schur_base_kind
from sfincs_jax.problems.profile_response.handoff import (
    RHS1KSPReplayState,
    RHS1SkipPrimaryKrylovSeedContext,
    rhs1_accept_candidate_and_update_replay,
    rhs1_accept_measured_candidate_and_update_replay,
    rhs1_accept_sparse_retry_candidate_and_update_replay,
    rhs1_accept_smoother_candidate_and_update_replay,
    rhs1_record_ksp_replay_problem,
    rhs1_retry_without_preconditioner_if_nonfinite,
    rhs1_run_adaptive_smoother_and_update_replay,
    rhs1_run_bicgstab_gmres_fallback_if_allowed,
    rhs1_run_collision_retry_if_allowed,
    rhs1_run_fast_post_xblock_polish,
    rhs1_run_full_pas_schur_rescue_from_env,
    rhs1_run_linear_candidate_and_update_replay,
    rhs1_run_measured_linear_candidate_and_update_replay,
    rhs1_run_primary_krylov_and_update_replay,
    rhs1_run_stage2_retry_if_allowed,
    rhs1_seed_skip_primary_krylov_and_update_replay,
    rhs1_skip_primary_krylov_reason,
)
from sfincs_jax.problems.profile_response.auto_solve import (
    RHS1AutoHostSolveContext,
    RHS1SparseHostSafeSolveContext,
    RHS1StructuredCSRSolveContext,
    solve_v3_full_system_structured_csr,
    solve_rhs1_structured_full_csr_explicit,
    try_rhs1_auto_host_solve,
    try_rhs1_sparse_host_safe_solve,
)
from sfincs_jax.problems.profile_response.dense import (
    HostDenseFullSolveContext,
    HostDenseReducedSolveContext,
    ProfileLinearSolveContext,
    RHS1Constraint0PETScCompatSolveContext,
    RHS1DenseKSPFullSolveContext,
    RHS1DenseKSPReducedSolveContext,
    RHS1FullDenseFallbackContext,
    RHS1FullDenseFallbackStageContext,
    RHS1FullHostDenseShortcutContext,
    RHS1DenseProbeStageContext,
    RHS1PostKrylovDenseShortcutEvaluationContext,
    RHS1ReducedDenseFallbackAdmissionStageContext,
    RHS1ReducedDenseFallbackCandidateContext,
    RHS1ReducedDenseFallbackStageContext,
    RHS1ReducedHostDenseShortcutContext,
    RHS1ScipyRescueStageContext,
    profile_solver_kind,
    rhs1_dense_shortcut_setup_from_env,
    rhs1_early_dense_shortcut_decision,
    rhs1_evaluate_post_krylov_dense_shortcut,
    rhs1_fp_preconditioner_probe_kind_from_env,
    rhs1_small_gmres_max_from_env,
    run_rhs1_scipy_rescue_stage,
    solve_profile_linear,
    solve_profile_linear_with_residual,
    solve_rhs1_constraint0_petsc_compat,
    solve_rhs1_dense_ksp_full,
    solve_rhs1_dense_ksp_reduced,
    run_rhs1_dense_probe_stage,
    run_rhs1_full_dense_fallback_stage,
    run_rhs1_full_host_dense_shortcut_stage,
    run_rhs1_reduced_dense_fallback_admission_stage,
    run_rhs1_reduced_host_dense_shortcut_stage,
)
from sfincs_jax.problems.profile_response.phi1_newton import (
    solve_v3_full_system_newton_krylov,
    solve_v3_full_system_newton_krylov_history,
)
from sfincs_jax.problems.profile_response.preconditioner_build import (
    RHS1FullBasePreconditionerSetupContext,
    RHS1FullPreconditionerBuildContext,
    RHS1FullStrongRetryStageContext,
    RHS1ReducedPreconditionerBuildContext,
    RHS1ReducedStrongRetryStageContext,
    _build_rhsmode1_block_preconditioner,
    _build_rhs1_preconditioner_from_kind,
    _build_rhs1_strong_preconditioner_full_from_kind,
    _build_rhs1_strong_preconditioner_reduced_from_kind,
    _build_rhsmode1_collision_preconditioner,
    _build_rhsmode1_pas_hybrid_preconditioner,
    _build_rhsmode1_pas_lite_preconditioner,
    _build_rhsmode1_pas_schur_preconditioner,
    _build_rhsmode1_pas_tokamak_theta_preconditioner,
    _build_rhsmode1_pas_tz_preconditioner,
    _build_rhsmode1_pas_xblock_ilu_preconditioner,
    _build_rhsmode1_species_block_preconditioner,
    _build_rhsmode1_sxblock_tz_preconditioner,
    _build_rhsmode1_theta_dd_preconditioner,
    _build_rhsmode1_theta_line_preconditioner,
    _build_rhsmode1_theta_zeta_preconditioner,
    _build_rhsmode1_xblock_tz_preconditioner,
    _build_rhsmode1_xblock_tz_lmax_preconditioner,
    _build_rhsmode1_xblock_tz_sparse_preconditioner,
    _build_rhsmode1_xmg_preconditioner,
    _build_rhsmode1_zeta_dd_preconditioner,
    _build_rhsmode1_zeta_line_preconditioner,
    _build_rhsmode23_tzfft_preconditioner,
    _compute_rhsmode1_sxblock_tz_sparse_host_seed,
    build_rhs1_full_preconditioner,
    build_rhs1_reduced_preconditioner_with_fallback,
    run_rhs1_full_strong_retry_stage,
    run_rhs1_reduced_strong_retry_stage,
    setup_rhs1_full_base_preconditioner,
)
from sfincs_jax.problems.profile_response.sparse.qi import (
    attempt_matrixfree_qi_device_seed_if_requested,
    build_matrixfree_qi_device_seed_setup,
)
from sfincs_jax.problems.profile_response.sparse.direct import (
    build_host_sparse_direct_factor_from_matvec as _build_host_sparse_direct_factor_from_matvec,
    build_sparse_jax_preconditioner_from_matvec as _build_sparse_jax_preconditioner_from_matvec,
    host_physical_memory_mb as _host_physical_memory_mb,
    host_sparse_direct_polish as _host_sparse_direct_polish,
    matvec_submatrix as _matvec_submatrix,
    maybe_rhsmode1_full_sparse_pattern as _maybe_rhsmode1_full_sparse_pattern,
    rhsmode1_explicit_sparse_pattern_probe_enabled as _rhsmode1_explicit_sparse_pattern_probe_enabled,
    rhsmode1_sparse_cache_key as _rhsmode1_sparse_cache_key,
    sparse_factor_cache_key as _sparse_factor_cache_key,
)
from sfincs_jax.problems.profile_response.diagnostics import (
    SparseRescueTailMetadataContext,
    sparse_rescue_tail_metadata_from_context,
)
from sfincs_jax.problems.profile_response.sparse.handoff import (
    DirectTailMaterializationContext,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredBuildContext,
    DirectTailSupportModePreflightContext,
    FortranReducedXBlockBackendContext,
    SparsePCGenericBranchSetupContext,
    SparsePCFactorPreflightPolicyContext,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCResidualCandidateAcceptanceContext,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCAutoPreflightRetryEvaluationContext,
    ExplicitSparseMinimumNormBranchContext,
    ExplicitSparseHostDirectBranchContext,
    RHS1FullSparseRetryStageContext,
    SparseHostOrILUFactorBuildContext,
    SparseHostRetryCandidateContext,
    SparseJAXRetryPreconditionerBuildContext,
    SparsePCGMRESContext,
    XBlockAugmentedKrylovStageContext,
    XBlockFirstKrylovAttemptContext,
    XBlockGlobalCouplingStageContext,
    XBlockKrylovControlSetupContext,
    XBlockKrylovProgressCallbacksContext,
    XBlockKrylovSolveStageContext,
    XBlockKrylovSolveSpaceContext,
    XBlockMomentSchurStageContext,
    XBlockPostKrylovCompletionContext,
    XBlockPostSolveCorrectionContext,
    XBlockPreflightGateContext,
    XBlockProbeCoarseStageContext,
    XBlockSideProbeStageContext,
    XBlockSparsePCBranchContext,
    XBlockTwoLevelStageContext,
    apply_xblock_global_coupling_stage,
    apply_xblock_augmented_krylov_stage,
    apply_xblock_moment_schur_stage,
    apply_xblock_probe_coarse_stage,
    run_xblock_qi_preconditioner_pipeline,
    apply_xblock_side_probe_stage,
    apply_xblock_two_level_stage,
    build_xblock_local_preconditioner,
    build_xblock_krylov_matvec_setup,
    build_direct_tail_structured_preconditioner_setup,
    build_xblock_assembled_operator_if_requested,
    build_sparse_pc_generic_branch_setup,
    build_direct_tail_materialization_setup,
    build_xblock_krylov_progress_callbacks,
    build_xblock_qi_stage_pipeline_context,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    evaluate_xblock_preflight_gate,
    select_sparse_pc_auto_preflight_retry_candidates,
    evaluate_sparse_pc_auto_preflight_retry,
    complete_xblock_post_krylov_stage,
    resolve_sparse_pc_gmres_control_policy,
    prepare_xblock_initial_guess,
    resolve_sparse_pc_entry_policy,
    resolve_sparse_pc_factor_preflight_policy,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    run_direct_tail_support_mode_preflight,
    resolve_fortran_reduced_xblock_factor_policy,
    prepare_xblock_augmented_krylov_basis,
    prepare_xblock_krylov_solve_space,
    resolve_xblock_krylov_control_setup,
    resolve_xblock_global_coupling_policy_setup,
    resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup,
    resolve_xblock_sparse_pc_branch_setup,
    resolve_xblock_two_level_policy_setup,
    run_sparse_pc_gmres_once,
    run_sparse_pc_gmres_once_for_retry,
    FPXBlockGlobalCorrectionContext,
    FPXBlockHighXCorrectionContext,
    SparseSXBlockRescueContext,
    SparseXBlockRescueAcceptanceContext,
    SparseXBlockRescueBuildContext,
    SparseXBlockRescueSolveContext,
    accept_sparse_xblock_rescue_candidate,
    build_sparse_xblock_rescue_preconditioner,
    run_fp_xblock_global_correction_stage,
    run_fp_xblock_highx_residual_correction_stage,
    run_sparse_sxblock_rescue_stage,
    run_sparse_xblock_rescue_solve_stage,
    run_rhs1_full_sparse_retry_stage,
    run_xblock_sparse_pc_branch,
    run_xblock_krylov_solve_stage,
    build_sparse_host_or_ilu_factor,
    run_sparse_host_retry_candidate,
    build_sparse_jax_retry_preconditioner,
    resolve_sparse_host_or_ilu_factor_controls,
    solve_fortran_reduced_xblock_backend,
    finalize_sparse_pc_gmres_bundle,
    sparse_pc_gmres_finalization_bundle_from_driver_result,
    solve_explicit_sparse_minimum_norm_branch,
    solve_explicit_sparse_host_direct_branch,
    finalize_xblock_assembled_operator_metadata,
)
from sfincs_jax.problems.profile_response.sparse.xblock import (
    xblock_sparse_pc_final_metadata_state_from_driver_scope,
    xblock_sparse_pc_final_payload_from_driver_state,
)
from sfincs_jax.problems.profile_response.preconditioner_build import (
    RHS1PostPrimaryMinresCorrectionContext,
    rhs1_collision_retry_allowed,
    rhs1_pas_force_strong_ratio_from_env,
    rhs1_pas_tz_guarded_minres_controls_from_env,
    rhs1_pas_weak_minres_controls_from_env,
    rhs1_pas_weak_minres_steps,
    rhs1_reduced_strong_selection_skip_messages,
    rhs1_resolved_strong_preconditioner_control,
    resolve_rhs1_reduced_strong_preconditioner_selection,
    rhs1_strong_preconditioner_env_from_env,
    rhs1_strong_preconditioner_control_messages,
    rhs1_strong_trigger_controls_from_env,
    run_rhs1_post_primary_minres_corrections,
)
from sfincs_jax.problems.profile_response.policies import (
    RHS1FullSparseRescueSetupContext,
    parse_rhs1_pas_tz_guarded_structured_levels as _rhs1_pas_tz_guarded_structured_levels,
    rhs1_full_sparse_rescue_setup,
    rhs1_sparse_jax_config_from_env,
    rhs1_sparse_operator_admission,
    rhs1_sparse_preconditioner_config_from_env,
    rhs1_sparse_rescue_initial_messages,
    rhs1_sparse_rescue_policy_setup,
    rhs1_sparse_rescue_tail_skip_messages,
    rhs1_xblock_fallback_initial_guess as _rhs1_xblock_fallback_initial_guess,
    rhsmode1_sparse_pc_default_permc_spec as _rhsmode1_sparse_pc_default_permc_spec,
    rhsmode1_sparse_pc_default_restart as _rhsmode1_sparse_pc_default_restart,
)

from sfincs_jax.problems.profile_response.setup import (
    SPARSE_HOST_DIRECT_SOLVE_METHODS as _SPARSE_HOST_DIRECT_SOLVE_METHODS,
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS as _SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS,
    SPARSE_HOST_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_SAFE_SOLVE_METHODS as _SPARSE_HOST_SAFE_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
    STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS as _STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS,
    ProfileResponseLinearProblemSetupContext,
    materialize_profile_response_linear_problem,
    resolve_rhs1_active_problem_setup,
    resolve_rhs1_domain_decomposition_setup,
    resolve_rhs1_initial_route_setup,
    resolve_rhs1_post_active_solve_policy_setup,
    resolve_rhs1_recycle_basis_setup,
    resolve_rhs1_reduced_mode_shape_setup,
)
from sfincs_jax.solvers.preconditioners.xblock import policy as _rhs1_xblock_policy
from sfincs_jax.solvers.preconditioners.xblock import policy as _rhs1_xblock_sparse_host_policy
from sfincs_jax.solvers.preconditioners.xblock.policy import (
    resolve_rhs1_xblock_sparse_pc_policy,
)
from sfincs_jax.solvers.preconditioners.pas import (
    RHS1PasFamilyBuilders,
    compose_preconditioners as _compose_preconditioners,
)
from sfincs_jax.solvers.preconditioners.full_fp import (
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
from sfincs_jax.solvers.preconditioners.xblock import (
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
from sfincs_jax.solvers.preconditioners.schur import (
    RHS1SchurPreconditionerBuilders,
    build_rhs1_schur_preconditioner,
)
from sfincs_jax.solvers.preconditioners.transport_matrix import (
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
from sfincs_jax.solvers.preconditioners.domain_decomposition import (
    build_rhs1_theta_dd_preconditioner,
    build_rhs1_theta_line_preconditioner,
    build_rhs1_theta_schwarz_preconditioner,
    build_rhs1_theta_line_xdiag_preconditioner,
    build_rhs1_theta_zeta_preconditioner,
    build_rhs1_zeta_dd_preconditioner,
    build_rhs1_zeta_line_preconditioner,
    build_rhs1_zeta_schwarz_preconditioner,
)
from sfincs_jax.problems.profile_response.policies import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
)
from sfincs_jax.problems.profile_response.policies import (
    read_bool_env as _rhs1_bool_env,
    read_float_env as _rhs1_float_env,
    read_int_env as _rhs1_int_env,
    read_post_solve_correction_policy as _read_rhs1_post_solve_correction_policy,
    read_probe_coarse_policy as _read_rhs1_probe_coarse_policy,
)
from sfincs_jax.problems.profile_response.policies import (
    _DIRECT_TAIL_STRUCTURED_PC_CACHE,
    _StructuredHostSparsePreconditionerBundle,
    _direct_tail_structured_pc_cache_key,
    _direct_tail_structured_pc_with_cache_metadata,
    _hash_numpy_array_for_cache,
    _is_direct_reduced_pmat_pc_kind,
    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb,
)
from sfincs_jax.operators.profile_response.reduced_tail import _try_build_fortran_reduced_constraint1_direct_tail_bundle
from sfincs_jax.operators.profile_response.true_operator_rescue import (
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
from sfincs_jax.problems.profile_response.solver_diagnostics import (
    rhs1_fortran_stdout_from_env,
    rhs1_ksp_diagnostics_controls_from_env,
    rhs1_ksp_history_limits_from_env,
)
from sfincs_jax.problems.profile_response.solver_diagnostics import (
    RHS1KSPDiagnosticsContext,
)
from sfincs_jax.problems.profile_response.solver_diagnostics import (
    ProfileResponseLinearFinalizationContext,
    finalize_profile_response_linear_solve,
)
from sfincs_jax.problems.profile_response.active_dof import (
    build_rhs1_active_dof_state as _build_rhs1_active_dof_state_compat,
)
from sfincs_jax.problems.profile_response.active_dof import (
    expand_reduced_with_map,
    fp_pitch_mode_active_indices,
    project_pas_constraint_f,
    reduce_full_with_indices,
)
from sfincs_jax.operators.profile_response.layout import (
    RHS1ActiveBlockLayout,
    RHS1ActiveFieldSplitOrdering,
    RHS1BlockLayout,
)
from sfincs_jax.solvers.preconditioners.xblock.coarse import (
    _rhs1_cap_lowmode_features,
    _rhs1_low_legendre_index_features,
    _rhs1_lowmode_angular_features,
    _rhs1_polynomial_moment_features,
)
from sfincs_jax.problems.profile_response.residual import (
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
from sfincs_jax.problems.profile_response.policies import (
    rhs1_constraint0_dense_fallback_allowed as _rhs1_constraint0_dense_fallback_allowed_impl,
    rhs1_constraint0_petsc_compat as _rhs1_constraint0_petsc_compat_impl,
    rhs1_constraint0_petsc_compat_config_from_env,
    rhs1_constraint0_petsc_compat_regularization,
    rhsmode1_constraint0_sparse_first_current_backend as _rhsmode1_constraint0_sparse_first,
)
from sfincs_jax.operators.profile_response.sources import (
    build_rhs1_xblock_constraint1_moment_schur_preconditioner as _build_rhs1_xblock_constraint1_moment_schur_preconditioner,
    constraint_scheme1_inject_source as _constraint_scheme1_inject_source,
    constraint_scheme1_moments_from_f as _constraint_scheme1_moments_from_f,
    constraint_scheme2_inject_source as _constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f as _constraint_scheme2_source_from_f,
)
from sfincs_jax.problems.profile_response.policies import (
    rhs1_prefer_sparse_over_dense_shortcut as _rhs1_prefer_sparse_over_dense_shortcut_impl,
    rhs1_sparse_prefer_skips_stage2 as _rhs1_sparse_prefer_skips_stage2_impl,
    rhsmode1_sparse_exact_lu_requested_current_backend as _rhsmode1_sparse_exact_lu_requested,
)
from sfincs_jax.problems.profile_response.policies import (
    rhs1_large_cpu_sparse_exact_lu_allowed as _rhs1_large_cpu_sparse_exact_lu_allowed_impl,
    rhs1_large_cpu_sparse_rescue_first as _rhs1_large_cpu_sparse_rescue_first_impl,
    rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend as _rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed,
    rhsmode1_large_cpu_sparse_rescue_allowed_current_backend as _rhsmode1_large_cpu_sparse_rescue_allowed,
    rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend as _rhsmode1_large_cpu_sparse_skip_primary_allowed,
    rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend as _rhsmode1_large_cpu_xblock_skip_primary_allowed,
    rhsmode1_sparse_sxblock_rescue_allowed_current_backend as _rhsmode1_sparse_sxblock_rescue_allowed,
    rhsmode1_sparse_xblock_rescue_allowed_current_backend as _rhsmode1_sparse_xblock_rescue_allowed,
)
from sfincs_jax.problems.profile_response.policies import (
    rhs1_bicgstab_fallback_controls_from_env,
    rhs1_bicgstab_fallback_decision,
    rhs1_bicgstab_fallback_target_from_env,
    rhs1_fast_post_xblock_polish_controls_from_env,
    rhs1_fp_bicgstab_polish_controls_from_env,
    rhs1_fp_global_low_l_polish_controls_from_env,
    rhs1_fp_l1_polish_controls_from_env,
    rhs1_fp_low_l_polish_controls_from_env,
    rhs1_fp_residual_polish_controls_from_env,
    rhs1_gmres_precondition_side_from_env,
    rhs1_krylov_routing_controls_from_env,
    rhs1_pas_source_zero_tolerance_from_env,
    rhsmode1_fast_post_xblock_polish_allowed_current_backend as _rhsmode1_fast_post_xblock_polish_allowed,
    rhsmode1_fp_targeted_polish_allowed_current_backend as _rhsmode1_fp_targeted_polish_allowed,
    rhsmode1_fp_xblock_global_correction_allowed_current_backend as _rhsmode1_fp_xblock_global_correction_allowed,
    rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend as _rhsmode1_skip_global_sparse_after_xblock_allowed,
)
from sfincs_jax.problems.profile_response.policies import (
    rhsmode1_pas_fast_accept_current_backend as _rhsmode1_pas_fast_accept,
)
from sfincs_jax.problems.profile_response.policies import (
    rhs1_pas_tz_guarded_stage2_retry,
    rhs1_stage2_admission_controls_from_env,
    rhs1_stage2_retry_admission_decision,
    rhs1_stage2_retry_controls_from_env,
    rhs1_stage2_trigger,
    rhs1_stage2_trigger_decision,
)
from sfincs_jax.solvers import path_policy as _solver_path_policy
from sfincs_jax.problems.profile_response.policies import (
    host_sparse_factor_dtype_current_backend as _host_sparse_factor_dtype,
    host_sparse_direct_refine_steps as _host_sparse_direct_refine_steps_impl,
    rhs1_dense_auto_fp_allowed as _rhs1_dense_auto_fp_allowed_impl,
    rhs1_dense_auto_fp_cutoff as _rhs1_dense_auto_fp_cutoff_impl,
    rhs1_dense_fallback_max as _rhs1_dense_fallback_max_impl,
    rhs1_dense_krylov_allowed as _rhs1_dense_krylov_allowed_impl,
    rhs1_explicit_sparse_host_direct_allowed as _rhs1_explicit_sparse_host_direct_allowed_impl,
    rhs1_host_sparse_direct_allowed as _rhs1_host_sparse_direct_allowed_impl,
    rhs1_host_sparse_skip_dense_ratio as _rhs1_host_sparse_skip_dense_ratio_impl,
    rhs1_structured_full_csr_auto_allowed as _rhs1_structured_full_csr_auto_allowed_impl,
    rhsmode1_dense_backend_allowed_current_backend as _rhsmode1_dense_backend_allowed,
    rhsmode1_host_dense_fallback_allowed_current_backend as _rhsmode1_host_dense_fallback_allowed,
    rhsmode1_host_dense_shortcut_allowed_current_backend as _rhsmode1_host_dense_shortcut_allowed,
    rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend as _rhsmode1_sparse_operator_preconditioned_rescue_allowed,
)
from sfincs_jax.host_refinement import (
    host_direct_solve_with_refinement as _host_direct_solve_with_refinement_impl,
    host_sparse_direct_solve_with_refinement as _host_sparse_direct_solve_with_refinement_impl,
)
from sfincs_jax.problems.transport_matrix.policies import (
    transport_host_gmres_accepts_preconditioned_residual as _transport_host_gmres_accepts_preconditioned_residual_impl,
    transport_precondition_side as _transport_precondition_side_impl,
    transport_sparse_direct_needs_float64_retry as _transport_sparse_direct_needs_float64_retry_impl,
    transport_sparse_direct_rescue_first as _transport_sparse_direct_rescue_first_impl,
    transport_tzfft_first_attempt_budget as _transport_tzfft_first_attempt_budget_impl,
    TransportRuntimePolicy,
)
from sfincs_jax.problems.transport_matrix.preconditioner_dispatch import (
    TransportPreconditionerContext,
    TransportPreconditionerDispatchBuilders,
    TransportStrongPreconditionerCache,
    build_transport_preconditioner_from_kind,
    normalize_transport_preconditioner_kind,
    resolve_transport_precondition_side_for_kind,
    resolve_transport_preconditioner_choice,
    transport_dd_config_from_env,
    transport_sparse_jax_config_from_env,
)
from sfincs_jax.problems.transport_matrix.direct_block_schur import build_transport_fp_direct_active_block_schur_preconditioner
from sfincs_jax.problems.transport_matrix.fortran_reduced_lu import build_transport_fp_fortran_reduced_lu_preconditioner
from sfincs_jax.problems.transport_matrix.solve_policy import resolve_transport_per_rhs_loop_policy, transport_geometry_scheme_from_namelist
from sfincs_jax.problems.transport_matrix.setup import (
    resolve_transport_maxiter_setup,
    resolve_transport_parallel_request,
    resolve_transport_state_setup,
    resolve_transport_which_rhs_setup,
)
from sfincs_jax.problems.transport_matrix.active_dense import (
    resolve_transport_active_dense_setup,
    transport_active_dof_indices as _transport_active_dof_indices,
)
from sfincs_jax.problems.transport_matrix.handoff_policy import (
    transport_candidate_is_better,
    transport_polish_config_from_env,
    transport_residual_value,
    transport_result_needs_retry,
)
from sfincs_jax.problems.transport_matrix.dense_lu import (
    dense_preconditioner_for_matvec as _dense_preconditioner_for_matvec,
    dense_solver_for_matvec as _dense_solver_for_matvec,
)
from sfincs_jax.problems.transport_matrix.dense_batch import (
    TransportDenseBatchContext,
    solve_transport_dense_batch as _solve_transport_dense_batch,
)
from sfincs_jax.problems.transport_matrix.host_gmres import transport_host_gmres_solve as _transport_host_gmres_solve
from sfincs_jax.problems.transport_matrix.iteration_stats import emit_transport_ksp_iteration_stats as _emit_transport_ksp_iteration_stats
from sfincs_jax.problems.transport_matrix.finalize import (
    TransportConstraintNullspaceProjector,
    TransportRHSFinalizationContext,
    finalize_full_transport_rhs,
    finalize_reduced_transport_rhs,
)
from sfincs_jax.problems.transport_matrix.loop import (
    TransportLoopProgress,
    TransportMatvecCache,
    TransportRecycleState,
    resolve_transport_recycle_k,
)
from sfincs_jax.problems.transport_matrix.sparse_direct_solve import (
    transport_sparse_direct_context_from_env as _transport_sparse_direct_context_from_env,
)
from sfincs_jax.problems.transport_matrix.parallel.policy import (
    transport_parallel_backend as _transport_parallel_backend,
    transport_parallel_gpu_worker_env as _transport_parallel_gpu_worker_env,
    transport_parallel_persistent_pool_enabled as _transport_parallel_persistent_pool_enabled,
    transport_parallel_start_method as _transport_parallel_start_method,
    transport_parallel_visible_gpu_ids as _transport_parallel_visible_gpu_ids,
)
from sfincs_jax.problems.transport_matrix.parallel.runtime import (
    TransportParallelSolveRuntime,
    get_transport_parallel_pool as _get_transport_parallel_pool,
    maybe_run_transport_parallel_solve,
    run_transport_parallel_gpu_subprocesses_with_policy as _run_transport_parallel_gpu_subprocesses,
    shutdown_transport_parallel_pool as _shutdown_transport_parallel_pool,
    solve_transport_parallel_payload as _solve_transport_parallel_payload,
    transport_parallel_pool_executor_kwargs as _transport_parallel_pool_executor_kwargs,
    transport_parallel_pool_key as _transport_parallel_pool_key,
    transport_parallel_process_pool_executor as _transport_parallel_process_pool_executor,
    transport_parallel_worker_env as _transport_parallel_worker_env,
)
from sfincs_jax.problems.transport_matrix.residual_quality import transport_residual_gate_thresholds_from_env
from sfincs_jax.problems.profile_response.policies import resolve_use_implicit as _resolve_use_implicit_impl
from sfincs_jax.phi1_newton_policy import (
    phi1_frozen_jacobian_policy,
    phi1_gmres_restart,
    phi1_line_search_policy,
    phi1_use_active_dof_mode,
)
from sfincs_jax.phi1_newton_linear import (
    build_phi1_newton_preconditioner,
    solve_phi1_newton_linear_step,
)
from sfincs_jax.problems.profile_response.phi1_newton import advance_phi1_newton_iterate
from sfincs_jax.solvers.progress import (
    RHS1ProgressNotes,
    rhs1_large_progress_enabled,
)
from sfincs_jax.problems.transport_matrix.diagnostics import (
    _flux_functions_from_op,
    transport_matrix_size_from_rhs_mode,
)
from sfincs_jax.problems.transport_matrix.postsolve_diagnostics import compute_transport_postsolve_diagnostics
from sfincs_jax.problems.transport_matrix.streaming_outputs import TransportStreamingOutputAccumulator
from sfincs_jax.solver import (
    block_gmres_result_ready as _block_gmres_result_ready,
    gmres_result_is_finite as _gmres_result_is_finite,
)
from sfincs_jax.solvers.preconditioner_operators import (  # noqa: F401
    block_diagonal_only as _block_diag_only,
    diagonal_only as _diag_only,
)
from sfincs_jax.solvers.preconditioner_operators import (
    _build_rhsmode1_preconditioner_operator_fortran_reduced,
    _build_rhsmode1_preconditioner_operator_point,
    _build_rhsmode1_preconditioner_operator_theta_dd,
    _build_rhsmode1_preconditioner_operator_theta_line,
    _build_rhsmode1_preconditioner_operator_zeta_dd,
    _build_rhsmode1_preconditioner_operator_zeta_line,
    _build_transport_preconditioner_operator_fortran_reduced,
    _build_transport_preconditioner_operator_point,
)
from sfincs_jax.solvers.sparse_triangular import (
    inverse_permutation as _inverse_permutation,
    triangular_solve_lower_csr_rows as _triangular_solve_lower_csr_rows,
    triangular_solve_lower_padded as _triangular_solve_lower_padded,
    triangular_solve_upper_csr_rows as _triangular_solve_upper_csr_rows,
    triangular_solve_upper_padded as _triangular_solve_upper_padded,
)
from sfincs_jax.solvers.preconditioner_setup import (
    hash_array as _hash_array,
    precond_chunk_cols as _precond_chunk_cols,
    rhs_mode1_precond_cache_key as _rhs_mode1_precond_cache_key_impl,
    rhs_mode1_structured_fblock_cache_key as _rhs_mode1_structured_fblock_cache_key_impl,
    transport_precond_cache_key as _transport_precond_cache_key_impl,
)
from sfincs_jax.solvers.krylov_dispatch import (
    HOST_SCIPY_KRYLOV_METHODS as _HOST_SCIPY_KRYLOV_METHODS,
    gmres_solve_dispatch as _gmres_solve_dispatch_impl,
    gmres_solve_with_residual_dispatch as _gmres_solve_with_residual_dispatch_impl,
    host_scipy_krylov_requested as _host_scipy_krylov_requested,
    ksp_iteration_solver_label as _ksp_iteration_solver_label,
    resolve_distributed_gmres_axis as _resolve_distributed_gmres_axis_impl,
    rhs_krylov_method_for_context as _rhs_krylov_method_for_context,
    solver_kind_for_label as _solver_kind_for_label,
)
from sfincs_jax.solvers.preconditioner_caches import (
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
from sfincs_jax.solvers.preconditioner_context import (
    auto_pas_geom4_fp32_precond_allowed as _auto_pas_geom4_fp32_precond_allowed,
    precond_dtype as _precond_dtype,
    precond_policy_hints as _precond_policy_hints,
    set_precond_policy_hints as _set_precond_policy_hints,
    set_precond_size_hint as _set_precond_size_hint,
    sparse_structural_tol as _sparse_structural_tol,
    use_solver_jit as _use_solver_jit,
)
from sfincs_jax.solvers.preconditioners.symbolic_sparse import (
    RHS1FullSystemMatrixFreeOperatorAdapter as _RHS1FullSystemMatrixFreeOperatorAdapter,
    build_sparse_ilu_from_matvec as _build_sparse_ilu_from_matvec,
    factorize_sparse_matrix_csr_host as _factorize_sparse_matrix_csr_host,
)
from sfincs_jax.problems.transport_matrix.direct_pmat import (
    _build_rhsmode23_direct_pmat_physics_coarse_basis,
    _try_build_rhsmode23_fp_direct_active_operator_bundle,
    _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle,
)
from sfincs_jax.v3_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1, _matvec_shard_axis, sharding_constraints
from sfincs_jax.profiling import Timer
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.v3_system import (
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
from sfincs_jax.v3_results import (
    V3LinearSolveResult,
    V3NewtonKrylovResult,
    V3TransportMatrixSolveResult,
    v3_linear_solve_result_from_payload,
)
from sfincs_jax.v3_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary,
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices,
)
from sfincs_jax.profiling import _rss_mb, maybe_profiler


_rhs1_xblock_precondition_side = _rhs1_xblock_policy.rhs1_xblock_precondition_side
_rhs1_xblock_gmres_restart = _rhs1_xblock_policy.rhs1_xblock_gmres_restart
build_rhs1_active_dof_state = _build_rhs1_active_dof_state_compat
_rhs1_dkes_gmres_budget = _solver_path_policy.rhs1_dkes_gmres_budget
_rhs1_residual_needs_rescue = _solver_path_policy.rhs1_residual_needs_rescue


_is_resource_exhausted_error = _solver_path_policy.is_resource_exhausted_error
_resolve_use_implicit = _resolve_use_implicit_impl


_transport_tzfft_first_attempt_budget = _transport_tzfft_first_attempt_budget_impl


_rhsmode1_dense_krylov_allowed = _rhs1_dense_krylov_allowed_impl


_rhsmode1_host_sparse_direct_allowed = _rhs1_host_sparse_direct_allowed_impl


_transport_runtime_policy = TransportRuntimePolicy(
    backend=lambda: jax.default_backend(),
    host_sparse_factor_dtype=_host_sparse_factor_dtype,
)
_transport_dense_backend_allowed = _transport_runtime_policy.dense_backend_allowed
_transport_dense_accelerator_auto_allowed = (
    _transport_runtime_policy.dense_accelerator_auto_allowed
)
_transport_tzfft_backend_allowed = _transport_runtime_policy.tzfft_backend_allowed
_transport_tzfft_accelerator_auto_allowed = (
    _transport_runtime_policy.tzfft_accelerator_auto_allowed
)
_transport_tzfft_structured_first_attempt_allowed = (
    _transport_runtime_policy.tzfft_structured_first_attempt_allowed
)
_transport_sparse_direct_rescue_allowed = (
    _transport_runtime_policy.sparse_direct_rescue_allowed
)
_transport_sparse_direct_first_attempt_allowed = (
    _transport_runtime_policy.sparse_direct_first_attempt_allowed
)
_transport_host_gmres_first_attempt_allowed = (
    _transport_runtime_policy.host_gmres_first_attempt_allowed
)
_transport_disable_auto_recycle = _transport_runtime_policy.disable_auto_recycle
_transport_sparse_factor_dtype = _transport_runtime_policy.sparse_factor_dtype
_transport_sparse_direct_use_explicit_helper = (
    _transport_runtime_policy.sparse_direct_use_explicit_helper
)
_transport_host_gmres_progress_every = (
    _transport_runtime_policy.host_gmres_progress_every
)


_host_sparse_direct_refine_steps = _host_sparse_direct_refine_steps_impl


_host_sparse_direct_solve_with_refinement = (
    _host_sparse_direct_solve_with_refinement_impl
)


_host_direct_solve_with_refinement = _host_direct_solve_with_refinement_impl


_rhsmode1_host_sparse_skip_dense_ratio = _rhs1_host_sparse_skip_dense_ratio_impl


_rhsmode1_explicit_sparse_host_direct_allowed = (
    _rhs1_explicit_sparse_host_direct_allowed_impl
)


_rhsmode1_pas_adaptive_smoother_allowed = (
    _rhs1_pas_adaptive_smoother_allowed_impl
)


_rhsmode1_constraint0_petsc_compat = _rhs1_constraint0_petsc_compat_impl


_rhsmode1_constraint0_dense_fallback_allowed = (
    _rhs1_constraint0_dense_fallback_allowed_impl
)


_rhsmode1_prefer_sparse_over_dense_shortcut = (
    _rhs1_prefer_sparse_over_dense_shortcut_impl
)


_rhsmode1_sparse_prefer_skips_stage2 = _rhs1_sparse_prefer_skips_stage2_impl


_rhsmode1_large_cpu_sparse_rescue_first = _rhs1_large_cpu_sparse_rescue_first_impl


_rhsmode1_large_cpu_sparse_exact_lu_allowed = (
    _rhs1_large_cpu_sparse_exact_lu_allowed_impl
)


_transport_sparse_direct_rescue_first = _transport_sparse_direct_rescue_first_impl


_transport_host_gmres_accepts_preconditioned_residual = (
    _transport_host_gmres_accepts_preconditioned_residual_impl
)


_transport_precondition_side = _transport_precondition_side_impl


_transport_sparse_direct_needs_float64_retry = (
    _transport_sparse_direct_needs_float64_retry_impl
)


def _build_rhsmode1_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper that binds Schur builders from this module.

    ``sfincs_jax.v3_driver`` aliases this module for legacy callers. Several
    tests and downstream scripts monkeypatch the individual builder globals
    here, so the wrapper must construct the Schur builder bundle from this
    module's globals instead of using a closed-over bundle elsewhere.
    """

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
    )


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
    linear_problem_setup = materialize_profile_response_linear_problem(
        ProfileResponseLinearProblemSetupContext(
            nml=nml,
            op=op,
            which_rhs=which_rhs,
            restart=int(restart),
            maxiter=maxiter,
            tol=float(tol),
            identity_shift=float(identity_shift),
            phi1_hat_base=phi1_hat_base,
            emit=emit,
            mark=_mark,
            env=os.environ,
            timer_factory=Timer,
            build_operator=full_system_operator_from_namelist,
            rhs_builder=rhs_v3_full_system,
            norm=jnp.linalg.norm,
            with_transport_rhs_settings=with_transport_rhs_settings,
            set_precond_size_hint=_set_precond_size_hint,
            set_precond_policy_hints=_set_precond_policy_hints,
        )
    )
    op = linear_problem_setup.op
    which_rhs = linear_problem_setup.which_rhs
    rhs = linear_problem_setup.rhs
    rhs_norm = linear_problem_setup.rhs_norm
    tol = float(linear_problem_setup.tol)
    fp_tol = float(linear_problem_setup.fp_tol)
    restart = int(linear_problem_setup.restart)
    maxiter = linear_problem_setup.maxiter
    restart_env_forced = bool(linear_problem_setup.restart_env_forced)
    maxiter_env_forced = bool(linear_problem_setup.maxiter_env_forced)
    maxiter_env = os.environ.get("SFINCS_JAX_GMRES_MAXITER", "").strip()
    geom_scheme_hint = int(linear_problem_setup.geom_scheme_hint)
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
            return run_xblock_sparse_pc_branch(
                XBlockSparsePCBranchContext(
                    _apply_device_subspace_residual_equation_correction=(
                        _apply_device_subspace_residual_equation_correction
                    ),
                    _apply_preconditioned_minres_correction=(
                        _apply_preconditioned_minres_correction
                    ),
                    _apply_subspace_minres_correction=_apply_subspace_minres_correction,
                    _rhs1_xblock_post_coarse_directions=(
                        _rhs1_xblock_post_coarse_directions
                    ),
                    _build_rhs1_xblock_constraint1_moment_schur_preconditioner=(
                        _build_rhs1_xblock_constraint1_moment_schur_preconditioner
                    ),
                    _build_rhsmode1_xblock_tz_sparse_preconditioner=(
                        _build_rhsmode1_xblock_tz_sparse_preconditioner
                    ),
                    _read_rhs1_post_solve_correction_policy=(
                        _read_rhs1_post_solve_correction_policy
                    ),
                    _read_rhs1_probe_coarse_policy=_read_rhs1_probe_coarse_policy,
                    _rhs1_bool_env=_rhs1_bool_env,
                    _rhs1_float_env=_rhs1_float_env,
                    _rhs1_xblock_fallback_initial_guess=(
                        _rhs1_xblock_fallback_initial_guess
                    ),
                    _rhs1_xblock_policy=_rhs1_xblock_policy,
                    _rhsmode1_fp_xblock_assembled_host_allowed=(
                        _rhsmode1_fp_xblock_assembled_host_allowed
                    ),
                    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly=(
                        _rhsmode1_fp_xblock_species_decoupled_for_host_assembly
                    ),
                    active_idx_jnp=active_idx_jnp,
                    active_size=active_size,
                    apply_v3_full_system_operator_cached=(
                        apply_v3_full_system_operator_cached
                    ),
                    atol=atol,
                    bicgstab_solve_with_history_scipy=bicgstab_solve_with_history_scipy,
                    bicgstab_solve_with_residual=bicgstab_solve_with_residual,
                    build_operator_from_pattern=build_operator_from_pattern,
                    device_csr_from_matrix=device_csr_from_matrix,
                    emit=emit,
                    estimate_v3_full_system_conservative_sparsity_summary=(
                        estimate_v3_full_system_conservative_sparsity_summary
                    ),
                    expand_reduced_with_map=expand_reduced_with_map,
                    fgmres_cycle_jit_solve_with_residual=(
                        fgmres_cycle_jit_solve_with_residual
                    ),
                    fgmres_solve_with_residual=fgmres_solve_with_residual,
                    fgmres_solve_with_residual_jit=fgmres_solve_with_residual_jit,
                    full_to_active_jnp=full_to_active_jnp,
                    gcrotmk_solve_with_history_scipy=gcrotmk_solve_with_history_scipy,
                    gmres_solve_with_history_scipy=gmres_solve_with_history_scipy,
                    include_electric_field_xi_sparse_pc=(
                        include_electric_field_xi_sparse_pc
                    ),
                    include_xdot_sparse_pc=include_xdot_sparse_pc,
                    lgmres_solve_with_history_scipy=lgmres_solve_with_history_scipy,
                    op=op,
                    pc_maxiter=pc_maxiter,
                    pc_restart=pc_restart,
                    pc_restart_env=pc_restart_env,
                    preconditioner_species=preconditioner_species,
                    preconditioner_xi=preconditioner_xi,
                    reduce_full_with_indices=reduce_full_with_indices,
                    resolve_rhs1_xblock_sparse_pc_policy=(
                        resolve_rhs1_xblock_sparse_pc_policy
                    ),
                    rhs=rhs,
                    rhs1_l2_norm_float=rhs1_l2_norm_float,
                    rhs1_residual_target=rhs1_residual_target,
                    rhs1_safe_ratio=rhs1_safe_ratio,
                    sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                    sparse_timer=sparse_timer,
                    summarize_v3_sparse_pattern=summarize_v3_sparse_pattern,
                    tfqmr_solve_with_residual=tfqmr_solve_with_residual,
                    tokamak_fp_er_pc=tokamak_fp_er_pc,
                    tol=tol,
                    use_dkes=use_dkes,
                    v3_full_system_conservative_sparsity_pattern=(
                        v3_full_system_conservative_sparsity_pattern
                    ),
                    v3_full_system_conservative_sparsity_pattern_for_indices=(
                        v3_full_system_conservative_sparsity_pattern_for_indices
                    ),
                    v3_linear_solve_result_from_payload=(
                        v3_linear_solve_result_from_payload
                    ),
                    validate_device_csr_matvec=validate_device_csr_matvec,
                    x0=x0,
                    xblock_sparse_pc=xblock_sparse_pc,
                    xblock_use_active_dof=xblock_use_active_dof,
                )
            )

        sparse_pc_branch_setup = build_sparse_pc_generic_branch_setup(
            SparsePCGenericBranchSetupContext(
                op=op,
                rhs=rhs,
                sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                active_dof_indices=_transport_active_dof_indices,
                reduce_full_with_indices=reduce_full_with_indices,
                expand_reduced_with_map=expand_reduced_with_map,
                fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                preconditioner_x=int(preconditioner_x),
                preconditioner_x_min_l=int(preconditioner_x_min_l),
                preconditioner_xi=int(preconditioner_xi),
                preconditioner_species=int(preconditioner_species),
                sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                constrained_pas_pc=bool(constrained_pas_pc),
                tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
                tokamak_fp_pc=bool(tokamak_fp_pc),
                pc_maxiter=int(pc_maxiter),
                pc_restart=int(pc_restart),
                host_sparse_factor_dtype=_host_sparse_factor_dtype,
                sparse_timer=sparse_timer,
                emit=emit,
                env=os.environ,
                default_permc_spec=_rhsmode1_sparse_pc_default_permc_spec,
                build_fortran_reduced_operator=(
                    _build_rhsmode1_preconditioner_operator_fortran_reduced
                ),
                build_point_operator=_build_rhsmode1_preconditioner_operator_point,
                fortran_reduced_pattern_for_indices=(
                    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices
                ),
                fortran_reduced_pattern=(
                    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern
                ),
                conservative_pattern_for_indices=(
                    v3_full_system_conservative_sparsity_pattern_for_indices
                ),
                conservative_pattern=v3_full_system_conservative_sparsity_pattern,
                summarize_pattern=summarize_v3_sparse_pattern,
                estimate_sparse_pc_memory=estimate_sparse_pc_memory,
                device_count=int(jax.device_count()),
            )
        )
        sparse_pc_active_idx_np = sparse_pc_branch_setup.active_idx_np
        sparse_pc_active_idx_jnp = sparse_pc_branch_setup.active_idx_jnp
        sparse_pc_full_to_active_jnp = sparse_pc_branch_setup.full_to_active_jnp
        sparse_pc_rhs = sparse_pc_branch_setup.rhs
        sparse_pc_linear_size = int(sparse_pc_branch_setup.linear_size)
        _sparse_pc_reduce_full = sparse_pc_branch_setup.reduce_full
        _sparse_pc_expand_reduced = sparse_pc_branch_setup.expand_reduced
        op_pc = sparse_pc_branch_setup.op_pc
        sparse_pc_preconditioner_operator = (
            sparse_pc_branch_setup.preconditioner_operator
        )
        fortran_reduced_xblock_min_size = (
            sparse_pc_branch_setup.fortran_reduced_xblock_min_size
        )
        fortran_reduced_sparse_pc_backend = (
            sparse_pc_branch_setup.fortran_reduced_sparse_pc_backend
        )
        fortran_reduced_sparse_pc_backend_reason = (
            sparse_pc_branch_setup.fortran_reduced_sparse_pc_backend_reason
        )

        if bool(fortran_reduced_sparse_pc) and str(fortran_reduced_sparse_pc_backend) == "xblock":
            fortran_reduced_xblock_payload = solve_fortran_reduced_xblock_backend(
                FortranReducedXBlockBackendContext(
                    op=op,
                    op_pc=op_pc,
                    rhs=rhs,
                    sparse_pc_rhs=sparse_pc_rhs,
                    x0=x0,
                    sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                    sparse_pc_active_idx_jnp=sparse_pc_active_idx_jnp,
                    sparse_pc_full_to_active_jnp=sparse_pc_full_to_active_jnp,
                    sparse_pc_linear_size=int(sparse_pc_linear_size),
                    sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                    reduce_full=_sparse_pc_reduce_full,
                    expand_reduced=_sparse_pc_expand_reduced,
                    reduce_full_with_indices=reduce_full_with_indices,
                    expand_reduced_with_map=expand_reduced_with_map,
                    operator_matvec=lambda x_full: apply_v3_full_system_operator_cached(op, x_full),
                    preconditioner_x=int(preconditioner_x),
                    preconditioner_x_min_l=int(preconditioner_x_min_l),
                    preconditioner_xi=int(preconditioner_xi),
                    preconditioner_species=int(preconditioner_species),
                    backend_reason=str(fortran_reduced_sparse_pc_backend_reason),
                    xblock_min_size=int(fortran_reduced_xblock_min_size),
                    sparse_timer=sparse_timer,
                    pc_restart=int(pc_restart),
                    pc_maxiter=int(pc_maxiter),
                    atol=float(atol),
                    tol=float(tol),
                    rhs_norm=float(rhs_norm),
                    emit=emit,
                    env=os.environ,
                    rhs1_l2_norm_float=rhs1_l2_norm_float,
                    rhs1_residual_target=rhs1_residual_target,
                    assembled_host_allowed=_rhsmode1_fp_xblock_assembled_host_allowed,
                    xblock_preconditioner_builder=_build_rhsmode1_xblock_tz_sparse_preconditioner,
                    moment_schur_builder=_build_rhs1_xblock_constraint1_moment_schur_preconditioner,
                    explicit_left_solver=explicit_left_preconditioned_gmres_scipy,
                    gmres_solver=gmres_solve_with_history_scipy,
                    lgmres_solver=lgmres_solve_with_history_scipy,
                    gcrotmk_solver=gcrotmk_solve_with_history_scipy,
                    bicgstab_solver=bicgstab_solve_with_history_scipy,
                )
            )
            return v3_linear_solve_result_from_payload(
                op=op,
                rhs=rhs,
                payload=fortran_reduced_xblock_payload,
            )

        pattern = sparse_pc_branch_setup.pattern
        sparse_pattern_scope = str(sparse_pc_branch_setup.sparse_pattern_scope)
        pattern_build_s = float(sparse_pc_branch_setup.pattern_build_s)
        summary = sparse_pc_branch_setup.summary
        sparse_pc_factor_policy = sparse_pc_branch_setup.factor_policy
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

        sparse_pc_gmres_context = SparsePCGMRESContext(
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
        )

        x_np, residual_norm_sparse_pc, rn_pc, history, solve_s = run_sparse_pc_gmres_once_for_retry(
            context=sparse_pc_gmres_context,
            x0=x0_sparse,
            maxiter=int(sparse_pc_first_attempt_maxiter),
        )
        sparse_pc_final_payload = finalize_sparse_pc_gmres_bundle(
            sparse_pc_gmres_finalization_bundle_from_driver_result(
                locals(),
                x=np.asarray(x_np, dtype=np.float64),
                residual_norm=float(residual_norm_sparse_pc),
                preconditioned_residual_norm=float(rn_pc),
                history=history,
                solve_s=float(solve_s),
            ),
            build_host_sparse_direct_factor_from_matvec=_build_host_sparse_direct_factor_from_matvec,
            run_sparse_pc_gmres_once_callback=lambda x0_arg, maxiter_arg: run_sparse_pc_gmres_once_for_retry(
                context=sparse_pc_gmres_context,
                x0=x0_arg,
                maxiter=int(maxiter_arg),
            ),
            minres_correction=_apply_preconditioned_minres_correction,
            expand_reduced=_sparse_pc_expand_reduced,
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=sparse_pc_final_payload,
        )
    if solve_method_kind_explicit in _SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS:
        sparse_minimum_norm_payload = solve_explicit_sparse_minimum_norm_branch(
            ExplicitSparseMinimumNormBranchContext(
                op=op,
                rhs=rhs,
                solve_method_kind=solve_method_kind_explicit,
                differentiable=differentiable,
                use_active_dof=bool(use_active_dof_mode),
                tol=float(tol),
                atol=float(atol),
                maxiter=maxiter,
                rhs_norm=float(rhs_norm),
                backend=jax.default_backend(),
                env=os.environ,
                emit=emit,
                build_pattern=v3_full_system_conservative_sparsity_pattern,
                summarize_pattern=summarize_v3_sparse_pattern,
                apply_cached_operator=apply_v3_full_system_operator_cached,
                build_operator_from_pattern=build_operator_from_pattern,
            )
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=sparse_minimum_norm_payload,
        )
    if solve_method_kind_explicit in _SPARSE_HOST_DIRECT_SOLVE_METHODS:
        sparse_host_direct_payload = solve_explicit_sparse_host_direct_branch(
            ExplicitSparseHostDirectBranchContext(
                op=op,
                rhs=rhs,
                differentiable=differentiable,
                use_active_dof=bool(use_active_dof_mode),
                tol=float(tol),
                atol=float(atol),
                rhs_norm=float(rhs_norm),
                refine_steps=_host_sparse_direct_refine_steps(
                    "SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_REFINE",
                    default=2,
                ),
                emit=emit,
                build_pattern=v3_full_system_conservative_sparsity_pattern,
                summarize_pattern=summarize_v3_sparse_pattern,
                apply_operator=apply_v3_full_system_operator,
                build_host_sparse_direct_factor_from_matvec=(
                    _build_host_sparse_direct_factor_from_matvec
                ),
                direct_solve_with_refinement=_host_direct_solve_with_refinement,
            )
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
                            rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                                pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                                tz_size=int(op.n_theta) * int(op.n_zeta),
                                active_size=int(active_size),
                            )
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
                        rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                            pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                            tz_size=int(op.n_theta) * int(op.n_zeta),
                            active_size=int(active_size),
                        )
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
                        rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                            pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                            tz_size=int(op.n_theta) * int(op.n_zeta),
                            active_size=int(active_size),
                        )
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
        res_reduced: GMRESSolveResult | None = None
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
            host_dense_shortcut_outcome = run_rhs1_reduced_host_dense_shortcut_stage(
                context=RHS1ReducedHostDenseShortcutContext(
                    enabled=True,
                    solve_context=HostDenseReducedSolveContext(
                        matvec=mv_reduced,
                        rhs=rhs_reduced,
                        active_size=int(active_size),
                        constraint_scheme=int(op.constraint_scheme),
                        has_fp=op.fblock.fp is not None,
                        dense_matrix_cache=dense_matrix_cache,
                    ),
                    current_result=None,
                    x0=x0_reduced,
                    active_size=int(active_size),
                    early_dense_shortcut=bool(early_dense_shortcut),
                    probe_shortcut=bool(probe_shortcut),
                ),
                replay_state=ksp_replay,
                record_replay_problem=rhs1_record_ksp_replay_problem,
                solver_kind=_solver_kind,
                emit=emit,
                mark=_mark,
            )
            res_reduced = host_dense_shortcut_outcome.result
            early_dense_shortcut = bool(host_dense_shortcut_outcome.early_dense_shortcut)
            probe_shortcut = bool(host_dense_shortcut_outcome.probe_shortcut)
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
                compat_config = rhs1_constraint0_petsc_compat_config_from_env(
                    restart=int(restart),
                    maxiter=maxiter,
                )
                cs0_outcome = solve_rhs1_constraint0_petsc_compat(
                    RHS1Constraint0PETScCompatSolveContext(
                        matvec=mv_reduced,
                        rhs=rhs_reduced,
                        x0=x0_reduced,
                        active_size=int(active_size),
                        tol=float(tol),
                        atol=float(atol),
                        sparse_drop_tol=float(sparse_drop_tol),
                        sparse_drop_rel=float(sparse_drop_rel),
                        config=compat_config,
                        regularization=lambda max_abs: rhs1_constraint0_petsc_compat_regularization(
                            max_abs=max_abs
                        ),
                    ),
                    emit=emit,
                )
                res_reduced = cs0_outcome.result
                rhs1_record_ksp_replay_problem(
                    ksp_replay,
                    matvec_fn=cs0_outcome.replay_matvec,
                    b_vec=cs0_outcome.replay_rhs,
                    precond_fn=None,
                    x0_vec=None if x0_reduced is None else jnp.asarray(x0_reduced, dtype=jnp.float64),
                    precond_side="none",
                    solver_kind=_solver_kind("incremental")[0],
                )
            except Exception as exc:  # noqa: BLE001
                cs0_petsc_compat = False
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat solve failed "
                        f"({type(exc).__name__}: {exc})",
                    )
        dense_probe_result = run_rhs1_dense_probe_stage(
            context=RHS1DenseProbeStageContext(
                matvec=mv_reduced,
                rhs=rhs_reduced,
                preconditioner=preconditioner_reduced,
                current_result=res_reduced,
                x0_reduced=x0_reduced,
                target=float(target_reduced),
                active_size=int(active_size),
                constraint_scheme=int(op.constraint_scheme),
                probe_shortcut=bool(probe_shortcut),
                cs0_petsc_compat=bool(cs0_petsc_compat),
                cs0_sparse_first=bool(cs0_sparse_first),
                cs0_dense_fallback_allowed=bool(cs0_dense_fallback_allowed),
                solve_method_kind=solve_method_kind,
                solve_method=str(solve_method),
                dense_shortcut_ratio=float(dense_shortcut_ratio),
                dense_fallback_max=int(dense_fallback_max),
                sparse_prefer_over_dense_shortcut=bool(sparse_prefer_over_dense_shortcut),
                gmres_precond_side=gmres_precond_side,
            ),
            replay_state=ksp_replay,
            record_replay_problem=rhs1_record_ksp_replay_problem,
            solver_kind=_solver_kind,
            emit=emit,
        )
        if dense_probe_result.result is not None:
            res_reduced = dense_probe_result.result
        x0_reduced = dense_probe_result.x0_reduced
        early_dense_shortcut = bool(dense_probe_result.early_dense_shortcut)
        probe_shortcut = bool(dense_probe_result.probe_shortcut)
        cpu_large_xblock_shortcut = False
        if cs0_petsc_compat:
            pass
        elif probe_shortcut:
            pass
        elif solve_method_kind == "dense_ksp":
            # PETSc v3 uses *left* preconditioning and checks convergence in the
            # preconditioned residual norm ||M^{-1} r||. To match this behavior with
            # JAX's GMRES (which uses a SciPy-style convergence check), solve the
            # explicitly left-preconditioned system:
            #   (M^{-1} A) x = (M^{-1} b).
            rhs1_progress_notes.krylov_start()
            _mark("rhs1_krylov_solve_start")
            dense_ksp_reduced_outcome = solve_rhs1_dense_ksp_reduced(
                RHS1DenseKSPReducedSolveContext(
                    matvec=mv_reduced,
                    rhs=rhs_reduced,
                    x0=x0_reduced,
                    active_size=int(active_size),
                    phi1_size=int(op.phi1_size),
                    n_species=int(op.n_species),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    nxi_for_x=nxi_for_x,
                    extra_size=int(op.extra_size),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    solve_linear=_solve_linear,
                    result_ready=_block_gmres_result_ready,
                ),
                emit=emit,
            )
            _mark("rhs1_krylov_solve_done")
            res_reduced = dense_ksp_reduced_outcome.result
            rhs1_record_ksp_replay_problem(
                ksp_replay,
                matvec_fn=dense_ksp_reduced_outcome.replay_matvec,
                b_vec=dense_ksp_reduced_outcome.replay_rhs,
                precond_fn=None,
                x0_vec=x0_reduced,
                precond_side="none",
                solver_kind=_solver_kind("incremental")[0],
            )
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
                    reason = rhs1_skip_primary_krylov_reason(
                        gpu_dkes_sparse_shortcut=bool(gpu_dkes_sparse_shortcut),
                        cpu_large_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                        cpu_large_sparse_shortcut=bool(cpu_large_sparse_shortcut),
                        backend_name=str(jax.default_backend()).strip().lower(),
                    )
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: skipping initial Krylov "
                        f"({reason}) -> sparse ILU",
                    )
                res_reduced, x0_reduced = rhs1_seed_skip_primary_krylov_and_update_replay(
                    replay_state=ksp_replay,
                    context=RHS1SkipPrimaryKrylovSeedContext(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=preconditioner_reduced,
                        x0_vec=x0_reduced,
                        precond_side=gmres_precond_side,
                        solver_kind=_solver_kind(solve_method)[0],
                        zero_like=jnp.zeros_like,
                        norm=jnp.linalg.norm,
                        inf_residual=lambda: jnp.asarray(np.inf, dtype=jnp.float64),
                        result_factory=lambda x, residual_norm: GMRESSolveResult(
                            x=x,
                            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
                        ),
                    ),
                )
            else:
                res_reduced, residual_vec, _primary_elapsed_s = (
                    rhs1_run_primary_krylov_and_update_replay(
                        replay_state=ksp_replay,
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=preconditioner_reduced,
                        x0_vec=x0_reduced,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        solve_method=solve_method,
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear,
                        solver_kind=_solver_kind(solve_method)[0],
                        returns_residual_vec=False,
                        current_residual_vec=residual_vec,
                        result_ready=_block_gmres_result_ready,
                        progress_start=rhs1_progress_notes.krylov_start,
                        mark=_mark,
                        mark_start="rhs1_krylov_solve_start",
                        mark_done="rhs1_krylov_solve_done",
                    )
                )
        res_reduced, residual_vec, _accepted, _retry_elapsed_s = (
            rhs1_retry_without_preconditioner_if_nonfinite(
                allowed=(not probe_shortcut) and preconditioner_reduced is not None,
                replay_state=ksp_replay,
                current_result=res_reduced,
                current_residual_vec=residual_vec,
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                x0_vec=x0_reduced,
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                solve_method=solve_method,
                precond_side=gmres_precond_side,
                solve_linear=_solve_linear,
                solver_kind=_solver_kind(solve_method)[0],
                result_is_finite=_gmres_result_is_finite,
                returns_residual_vec=False,
                result_ready=_block_gmres_result_ready,
                emit=emit,
                message=(
                    "solve_v3_full_system_linear_gmres: preconditioned reduced "
                    "GMRES returned non-finite result; retrying without preconditioner"
                ),
            )
        )
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
        minres_corrections = run_rhs1_post_primary_minres_corrections(
            RHS1PostPrimaryMinresCorrectionContext(
                result=res_reduced,
                residual_vec=residual_vec,
                residual_norm_true=float(residual_norm_true),
                target=float(target_reduced),
                matvec=mv_reduced,
                rhs=rhs_reduced,
                preconditioner=preconditioner_reduced,
                has_pas=op.fblock.pas is not None,
                rhs1_precond_kind=rhs1_precond_kind,
                pas_tz_guarded_fallback=bool(rhs1_pas_tz_guarded_fallback),
                pas_tz_guarded_axis=rhs1_pas_tz_guarded_axis,
                pas_tz_guarded_stream_requested=bool(pas_tz_guarded_stream_requested),
                use_pas_projection=bool(use_pas_projection),
                metadata=pas_tz_guarded_correction_metadata,
                requested_guarded_correction=os.environ.get(
                    "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION", ""
                ),
                build_tzfft_preconditioner=lambda: _build_rhsmode23_tzfft_preconditioner(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                ),
                wrap_pas_preconditioner=_wrap_pas_precond,
                minres_correction=_apply_preconditioned_minres_correction,
                result_factory=lambda x, residual_norm: GMRESSolveResult(
                    x=jnp.asarray(x, dtype=jnp.float64),
                    residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
                ),
                resolve_guarded_correction_kind=resolve_pas_tz_guarded_correction_kind,
                guarded_controls_factory=rhs1_pas_tz_guarded_minres_controls_from_env,
                weak_steps_policy=rhs1_pas_weak_minres_steps,
                weak_controls_factory=rhs1_pas_weak_minres_controls_from_env,
            ),
            emit=emit,
        )
        res_reduced = minres_corrections.result
        residual_vec = minres_corrections.residual_vec
        residual_norm_true = float(minres_corrections.residual_norm_true)
        res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        stage2_decision = rhs1_stage2_trigger_decision(
            res_ratio=float(res_ratio),
            use_dkes=bool(use_dkes),
            has_fp=op.fblock.fp is not None, include_phi1=bool(op.include_phi1),
            residual_norm=float(res_reduced.residual_norm),
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            pas_tz_guarded_fallback=bool(rhs1_pas_tz_guarded_fallback),
            pas_tz_guarded_retry=rhs1_pas_tz_guarded_stage2_retry(),
            cpu_large_xblock_shortcut=bool(cpu_large_xblock_shortcut), cpu_large_sparse_shortcut=bool(cpu_large_sparse_shortcut),
        )
        stage2_trigger, fp_force_stage2 = stage2_decision.stage2_trigger, stage2_decision.fp_force_stage2
        if emit is not None:
            for _level, _message in stage2_decision.messages:
                emit(_level, _message)
        early_dense_decision = rhs1_early_dense_shortcut_decision(
            early_dense_shortcut=bool(early_dense_shortcut),
            cs0_sparse_first=bool(cs0_sparse_first),
            cs0_dense_fallback_allowed=bool(cs0_dense_fallback_allowed),
            constraint_scheme=int(op.constraint_scheme),
            dense_shortcut_ratio=float(dense_shortcut_ratio),
            residual_ratio=float(res_ratio),
            sparse_prefer_over_dense_shortcut=bool(sparse_prefer_over_dense_shortcut),
            dense_fallback_max=int(dense_fallback_max),
            active_size=int(active_size),
        )
        early_dense_shortcut = bool(early_dense_decision.early_dense_shortcut)
        if emit is not None:
            for _level, _message in early_dense_decision.messages:
                emit(_level, _message)
        solver_kind = _solver_kind(solve_method)[0]
        bicgstab_fallback = rhs1_bicgstab_fallback_decision(
            solver_kind=solver_kind,
            cpu_large_sparse_shortcut=bool(cpu_large_sparse_shortcut),
            result_is_finite=_gmres_result_is_finite(res_reduced),
            residual_norm=float(res_reduced.residual_norm),
            strict=bool(bicgstab_fallback_strict),
            target=float(target_reduced),
            distributed_axis=distributed_axis,
            has_pas=op.fblock.pas is not None,
            include_phi1=bool(op.include_phi1),
        )
        if bicgstab_fallback.run_fallback:
            res_reduced, residual_vec, preconditioner_reduced, _accepted, _bicgstab_fallback_elapsed_s = (
                rhs1_run_bicgstab_gmres_fallback_if_allowed(
                    allowed=True,
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    preconditioner_enabled=bool(rhs1_precond_enabled),
                    build_preconditioner=_build_rhs1_preconditioner_reduced_with_fallback,
                    x0_vec=x0_reduced,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear,
                    target=float(bicgstab_fallback.target),
                    returns_residual_vec=False,
                    result_ready=_block_gmres_result_ready,
                    emit=emit,
                )
            )
        stage2_retry_admission = rhs1_stage2_retry_admission_decision(
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            fp_force_stage2=bool(fp_force_stage2),
            stage2_enabled=bool(stage2_enabled),
            stage2_trigger=bool(stage2_trigger),
            early_dense_shortcut=bool(early_dense_shortcut),
            gpu_dkes_sparse_shortcut=bool(gpu_dkes_sparse_shortcut),
            sparse_prefer_skips_stage2=bool(sparse_prefer_skips_stage2),
            elapsed_s=t.elapsed_s(),
            time_cap_s=float(stage2_time_cap_s),
        )
        if emit is not None:
            for _level, _message in stage2_retry_admission.messages:
                emit(_level, _message)
        if stage2_retry_admission.run_retry:
            stage2_controls = rhs1_stage2_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
                active_size=int(active_size),
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
                tokamak_pas=bool(tokamak_pas),
            )
            stage2_method = str(stage2_controls.method)
            res_reduced, residual_vec, preconditioner_reduced, _accepted, _stage2_elapsed_s = (
                rhs1_run_stage2_retry_if_allowed(
                    allowed=True,
                    replay_state=ksp_replay,
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=preconditioner_reduced,
                    preconditioner_enabled=bool(rhs1_precond_enabled),
                    build_preconditioner=_build_rhs1_preconditioner_reduced_with_fallback,
                    controls=stage2_controls,
                    tol=float(tol),
                    atol=float(atol),
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear,
                    solver_kind=_solver_kind(stage2_method)[0],
                    candidate_name=f"stage2_reduced:{stage2_method}",
                    baseline_name="current_reduced",
                    target=float(target_reduced),
                    peak_rss_mb=_rss_mb,
                    returns_residual_vec=False,
                    result_ready=_block_gmres_result_ready,
                    emit=emit,
                    label="stage2 reduced GMRES",
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

        qi_device_seed_setup = build_matrixfree_qi_device_seed_setup(
            op=op,
            active_size=int(active_size),
            target_reduced=float(target_reduced),
            mv_reduced=mv_reduced,
            rhs_reduced=rhs_reduced,
            emit=emit,
            timer_elapsed_s=t.elapsed_s,
            rhsmode1_general_metadata=rhsmode1_general_metadata,
        )

        qi_device_skip_strong = bool(qi_device_seed_setup.skip_strong)
        early_qi_attempt = attempt_matrixfree_qi_device_seed_if_requested(
            res_reduced,
            hook="early_active_dof",
            setup=qi_device_seed_setup,
            enabled=bool(qi_device_seed_setup.early_enabled or qi_device_seed_setup.skip_strong),
        )
        res_reduced = early_qi_attempt.result

        pas_smoother_allowed = (
            rhs1_precond_kind in {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
            and preconditioner_reduced is not None
            and _rhsmode1_pas_adaptive_smoother_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                use_implicit=bool(use_implicit),
            )
        )
        smoother_controls = rhs1_pas_adaptive_smoother_controls_from_env() if pas_smoother_allowed else None
        res_reduced, residual_vec, _accepted = rhs1_run_adaptive_smoother_and_update_replay(
            allowed=bool(pas_smoother_allowed),
            replay_state=ksp_replay,
            current_result=res_reduced,
            current_residual_vec=residual_vec,
            smoother_factory=lambda result: adaptive_pas_smoother(
                matvec=mv_reduced,
                rhs=rhs_reduced,
                preconditioner=preconditioner_reduced,
                x0=result.x,
                target=float(target_reduced),
                omega=float(smoother_controls.omega),
                max_sweeps=int(smoother_controls.max_sweeps),
            ),
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
        if fp_force_strong:
            strong_precond_trigger = True
        collision_retry_allowed = rhs1_collision_retry_allowed(
            residual_norm=float(res_reduced.residual_norm),
            target=float(target_reduced),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            rhs1_precond_kind=rhs1_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            strong_precond_trigger=bool(strong_precond_trigger),
        )
        res_reduced, residual_vec, bicgstab_preconditioner_reduced, _accepted, _collision_elapsed_s = (
            rhs1_run_collision_retry_if_allowed(
                allowed=bool(collision_retry_allowed),
                replay_state=ksp_replay,
                current_result=res_reduced,
                current_residual_vec=residual_vec,
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=bicgstab_preconditioner_reduced,
                build_preconditioner=lambda: _build_rhsmode1_collision_preconditioner(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                ),
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precond_side=gmres_precond_side,
                solve_linear=_solve_linear,
                solver_kind=_solver_kind("incremental")[0],
                target=float(target_reduced),
                returns_residual_vec=False,
                emit=emit,
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
        if emit is not None:
            backend_name = str(jax.default_backend()).strip().lower()
            sparse_label = "large CPU" if backend_name == "cpu" else f"{backend_name} host-sparse"
            for message in rhs1_strong_preconditioner_control_messages(
                strong_control,
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                rhs1_precond_kind=rhs1_precond_kind,
                pas_auto_strong_ratio=float(pas_auto_strong_ratio),
                pas_collision_probe_allows_strong=(
                    bool(pas_precond_force_collision)
                    and strong_precond_env in {"", "auto"}
                ),
                pas_force_strong_ratio=rhs1_pas_force_strong_ratio_from_env(),
                sparse_rescue_label=sparse_label,
            ):
                emit(1, message)
        reduced_strong_selection = resolve_rhs1_reduced_strong_preconditioner_selection(
            strong_precond_env=strong_precond_env,
            control=strong_control,
            has_extra_constraint_block=int(op.constraint_scheme) == 2 and int(op.extra_size) > 0,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            geom_scheme=int(geom_scheme),
            use_dkes=bool(use_dkes),
            active_size=int(active_size),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
            nxi_for_x_sum=int(np.sum(nxi_for_x)) if nxi_for_x.size else 0,
            shard_axis=_matvec_shard_axis(op),
            device_count=int(jax.device_count()),
            strong_precond_trigger=bool(strong_precond_trigger),
            rhs1_precond_kind=rhs1_precond_kind,
            res_ratio=float(res_ratio),
            pas_tz_guarded_fallback=bool(rhs1_pas_tz_guarded_fallback),
            pas_tz_guarded_strong_retry=rhs1_pas_tz_guarded_strong_retry_from_env(),
            qi_device_skip_strong=bool(qi_device_skip_strong),
        )
        strong_precond_kind = reduced_strong_selection.kind
        strong_xblock_tz_lmax = reduced_strong_selection.xblock_tz_lmax
        strong_precond_trigger = bool(reduced_strong_selection.trigger)

        if emit is not None:
            for message in rhs1_reduced_strong_selection_skip_messages(
                reduced_strong_selection
            ):
                emit(1, message)

        def _build_reduced_strong_candidate(kind: str, lmax: int | None):
            return _build_rhs1_strong_preconditioner_reduced_from_kind(
                op=op,
                strong_precond_kind=kind,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                rhs1_xblock_tz_lmax=lmax,
                dd_block_theta=rhs1_dd_setup.block("theta"),
                dd_overlap_theta=rhs1_dd_setup.overlap("theta", default=1),
                dd_block_zeta=rhs1_dd_setup.block("zeta"),
                dd_overlap_zeta=rhs1_dd_setup.overlap("zeta", default=1),
            )

        reduced_strong_retry = run_rhs1_reduced_strong_retry_stage(
            RHS1ReducedStrongRetryStageContext(
                strong_precond_kind=strong_precond_kind,
                strong_xblock_tz_lmax=strong_xblock_tz_lmax,
                rescue_needed=_rhs1_residual_needs_rescue(
                    float(res_reduced.residual_norm),
                    float(target_reduced),
                    force=bool(fp_force_strong),
                ),
                strong_precond_trigger=bool(strong_precond_trigger),
                early_dense_shortcut=bool(early_dense_shortcut),
                active_size=int(active_size),
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
                rhs1_precond_kind=rhs1_precond_kind,
                current_result=res_reduced,
                current_residual_vec=residual_vec,
                matvec=mv_reduced,
                rhs=rhs_reduced,
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precondition_side=gmres_precond_side,
                solver_kind=_solver_kind("incremental")[0],
                target=float(target_reduced),
                peak_rss_mb=_rss_mb(),
                emit=emit,
                mark=_mark,
                replay_state=ksp_replay,
                build_strong_preconditioner=_build_reduced_strong_candidate,
                wrap_pas_preconditioner=_wrap_pas_precond,
                use_pas_projection=bool(use_pas_projection),
                run_measured_candidate=rhs1_run_measured_linear_candidate_and_update_replay,
                solve_linear=_solve_linear,
                result_ready=_block_gmres_result_ready,
            )
        )
        res_reduced = reduced_strong_retry.result
        residual_vec = reduced_strong_retry.residual_vec

        # Only treat the probe as a "dense shortcut" when the dense branch is
        # actually allowed (probe_shortcut). Otherwise we still want to try
        # stronger preconditioners (e.g. sparse ILU) before giving up.
        dense_shortcut = probe_shortcut
        post_krylov_dense_evaluation = rhs1_evaluate_post_krylov_dense_shortcut(
            RHS1PostKrylovDenseShortcutEvaluationContext(
                dense_shortcut=bool(dense_shortcut),
                dense_shortcut_ratio=float(dense_shortcut_ratio),
                current_result=res_reduced,
                rhs=rhs_reduced,
                matvec=mv_reduced,
                target=float(target_reduced),
                dense_fallback_max=int(_rhsmode1_dense_fallback_max(op)),
                active_size=int(active_size),
                constraint_scheme=int(op.constraint_scheme),
                cs0_sparse_first=bool(cs0_sparse_first),
                sparse_prefer_over_dense_shortcut=bool(
                    sparse_prefer_over_dense_shortcut
                ),
                sparse_exact_direct=bool(sparse_exact_direct),
            )
        )
        dense_shortcut = bool(post_krylov_dense_evaluation.dense_shortcut)
        if emit is not None:
            for _level, _message in post_krylov_dense_evaluation.messages:
                emit(_level, _message)

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
        pre_sparse_qi_attempt = attempt_matrixfree_qi_device_seed_if_requested(
            res_reduced,
            hook="pre_sparse_active_dof",
            setup=qi_device_seed_setup,
            enabled=bool(
                qi_device_seed_setup.pre_sparse_enabled
                and sparse_enabled
                and float(res_reduced.residual_norm) > target_reduced
                and not bool(rhsmode1_general_metadata.get("xblock_qi_device_preconditioner_built", False))
            ),
        )
        res_reduced = pre_sparse_qi_attempt.result
        if bool(pre_sparse_qi_attempt.improved):
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
                    sparse_xblock_build = build_sparse_xblock_rescue_preconditioner(
                        context=SparseXBlockRescueBuildContext(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                            active_size=int(active_size),
                            preconditioner_species=int(preconditioner_species),
                            preconditioner_x=int(preconditioner_x),
                            preconditioner_xi=int(preconditioner_xi),
                            use_implicit=bool(use_implicit),
                            drop_tol=float(sparse_drop_tol),
                            drop_rel=float(sparse_drop_rel),
                            ilu_drop_tol=float(sparse_ilu_drop_tol),
                            fill_factor=float(sparse_ilu_fill),
                            emit=emit,
                            mark=_mark,
                            assembled_host_allowed=_rhsmode1_fp_xblock_assembled_host_allowed,
                            builder=_build_rhsmode1_xblock_tz_sparse_preconditioner,
                        )
                    )
                    precond_sparse_xblock = sparse_xblock_build.preconditioner
                    sparse_xblock_preconditioner_xi = int(sparse_xblock_build.preconditioner_xi)
                    assembled_host_fp = bool(sparse_xblock_build.force_assembled_host_fp)
                    precond_sparse_xblock_current = precond_sparse_xblock
                    sparse_xblock_rescue_built = True
                    sparse_xblock_rescue_assembled_host_fp = bool(assembled_host_fp)
                    sparse_xblock_rescue_preconditioner_xi = int(sparse_xblock_preconditioner_xi)
                    sparse_xblock_solve = run_sparse_xblock_rescue_solve_stage(
                        context=SparseXBlockRescueSolveContext(
                            preconditioner=precond_sparse_xblock,
                            rhs=rhs_reduced,
                            matvec=mv_reduced,
                            current_result=res_reduced,
                            target=float(target_reduced),
                            tol=float(tol),
                            atol=float(atol),
                            restart=int(restart),
                            maxiter=maxiter,
                            precondition_side=gmres_precond_side,
                            active_size=int(active_size),
                            use_implicit=bool(use_implicit),
                            assembled_host_fp=bool(assembled_host_fp),
                            emit=emit,
                            mark=_mark,
                            solve_linear=_solve_linear,
                            host_gmres_solver=gmres_solve_with_history_scipy,
                        )
                    )
                    res_sparse_xblock = sparse_xblock_solve.result
                    sparse_xblock_rescue_reason = str(sparse_xblock_solve.reason)
                    if sparse_xblock_solve.candidate_residual is not None:
                        sparse_xblock_rescue_candidate_residual = float(
                            sparse_xblock_solve.candidate_residual
                        )
                    if sparse_xblock_solve.seed_residual is not None:
                        explicit_fp_xblock_seed_residual = float(sparse_xblock_solve.seed_residual)
                        sparse_xblock_rescue_seed_residual = float(sparse_xblock_solve.seed_residual)
                    if sparse_xblock_solve.seed_improvement_ratio is not None:
                        explicit_fp_xblock_seed_improvement_ratio = float(
                            sparse_xblock_solve.seed_improvement_ratio
                        )
                        sparse_xblock_rescue_seed_improvement_ratio = float(
                            sparse_xblock_solve.seed_improvement_ratio
                        )
                    if sparse_xblock_solve.seed_accept_ratio is not None:
                        sparse_xblock_rescue_seed_accept_ratio = float(
                            sparse_xblock_solve.seed_accept_ratio
                        )
                    if sparse_xblock_solve.seed_refine_steps is not None:
                        sparse_xblock_rescue_seed_refine_steps = int(
                            sparse_xblock_solve.seed_refine_steps
                        )
                    if sparse_xblock_solve.seed_refines_performed is not None:
                        sparse_xblock_rescue_seed_refines_performed = int(
                            sparse_xblock_solve.seed_refines_performed
                        )
                    sparse_xblock_acceptance = accept_sparse_xblock_rescue_candidate(
                        context=SparseXBlockRescueAcceptanceContext(
                            current_result=res_reduced,
                            candidate_result=res_sparse_xblock,
                            reason=sparse_xblock_rescue_reason,
                            assembled_host_fp=bool(assembled_host_fp),
                            use_implicit=bool(use_implicit),
                            replay_state=ksp_replay,
                            matvec=mv_reduced,
                            rhs=rhs_reduced,
                            preconditioner=precond_sparse_xblock,
                            precondition_side=gmres_precond_side,
                            solver_kind=_solver_kind("incremental")[0],
                            restart=int(restart),
                            maxiter=maxiter,
                            record_replay_problem=rhs1_record_ksp_replay_problem,
                        )
                    )
                    res_reduced = sparse_xblock_acceptance.result
                    sparse_xblock_rescue_candidate_accepted = bool(
                        sparse_xblock_acceptance.accepted
                    )
                    sparse_xblock_rescue_reason = str(sparse_xblock_acceptance.reason)
                    explicit_fp_xblock_seed_used = bool(
                        sparse_xblock_acceptance.explicit_seed_used
                    )
                    if sparse_xblock_acceptance.candidate_residual is not None:
                        sparse_xblock_rescue_candidate_residual = float(
                            sparse_xblock_acceptance.candidate_residual
                        )
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
                fp_xblock_global_correction = run_fp_xblock_global_correction_stage(
                    context=FPXBlockGlobalCorrectionContext(
                        current_result=res_reduced,
                        matvec=mv_reduced,
                        rhs=rhs_reduced,
                        preconditioner=correction_precond,
                        preconditioner_label=fp_xblock_global_correction_preconditioner,
                        steps=int(correction_steps),
                        alpha_clip=float(correction_alpha_clip),
                        min_improvement=float(correction_min_improvement),
                        preconditioner_clip=float(correction_precond_clip),
                        replay_state=ksp_replay,
                        emit=emit,
                        elapsed_s=t.elapsed_s,
                        mark=_mark,
                        safe_preconditioner=_safe_preconditioner,
                        correction=_apply_preconditioned_minres_correction,
                    )
                )
                res_reduced = fp_xblock_global_correction.result
                if fp_xblock_global_correction.residual_vec is not None:
                    residual_vec = fp_xblock_global_correction.residual_vec
                fp_xblock_global_correction_accepted = bool(
                    fp_xblock_global_correction.accepted
                )
                fp_xblock_global_correction_reason = str(
                    fp_xblock_global_correction.reason
                )
                fp_xblock_global_correction_error = fp_xblock_global_correction.error
                fp_xblock_global_correction_steps = fp_xblock_global_correction.steps
                fp_xblock_global_correction_accepted_steps = (
                    fp_xblock_global_correction.accepted_steps
                )
                fp_xblock_global_correction_residual_before = (
                    fp_xblock_global_correction.residual_before
                )
                fp_xblock_global_correction_residual_after = (
                    fp_xblock_global_correction.residual_after
                )
                fp_xblock_global_correction_improvement_ratio = (
                    fp_xblock_global_correction.improvement_ratio
                )
                fp_xblock_global_correction_elapsed_s = (
                    fp_xblock_global_correction.elapsed_s
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
                highx_correction = run_fp_xblock_highx_residual_correction_stage(
                    context=FPXBlockHighXCorrectionContext(
                        current_result=res_reduced,
                        matvec=mv_reduced,
                        rhs=rhs_reduced,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                        total_size=int(op.total_size),
                        n_species=int(op.n_species),
                        n_x=int(op.n_x),
                        n_xi=int(op.n_xi),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        n_xi_for_x=tuple(
                            int(v)
                            for v in np.asarray(
                                op.fblock.collisionless.n_xi_for_x,
                                dtype=np.int32,
                            )
                        ),
                        host_block_max_env_value=os.environ.get(
                            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX",
                            "",
                        ).strip(),
                        include_factored_blocks=_rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_FACTORED",
                            default=False,
                        ),
                        max_blocks=_rhs1_int_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MAX_BLOCKS",
                            default=16,
                            minimum=1,
                        ),
                        steps=_rhs1_int_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_STEPS",
                            default=1,
                            minimum=1,
                        ),
                        max_directions=_rhs1_int_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MAX_DIRECTIONS",
                            default=12,
                            minimum=1,
                        ),
                        alpha_clip=_rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_ALPHA_CLIP",
                            default=0.0,
                            minimum=0.0,
                        ),
                        rcond=_rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_RCOND",
                            default=1.0e-12,
                            minimum=0.0,
                        ),
                        min_improvement=_rhs1_float_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_MIN_IMPROVEMENT",
                            default=0.0,
                            minimum=0.0,
                        ),
                        include_all=_rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_ALL",
                            default=True,
                        ),
                        include_raw=_rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION_INCLUDE_RAW",
                            default=False,
                        ),
                        replay_state=ksp_replay,
                        emit=emit,
                        elapsed_s=t.elapsed_s,
                        mark=_mark,
                        block_factor_allowed=(
                            _rhs1_xblock_sparse_host_policy.rhs1_xblock_sparse_host_block_factor_allowed
                        ),
                        correction=_apply_subspace_minres_correction,
                    )
                )
                res_reduced = highx_correction.result
                if highx_correction.residual_vec is not None:
                    residual_vec = highx_correction.residual_vec
                fp_xblock_highx_residual_correction_accepted = bool(
                    highx_correction.accepted
                )
                fp_xblock_highx_residual_correction_reason = str(
                    highx_correction.reason
                )
                fp_xblock_highx_residual_correction_error = highx_correction.error
                fp_xblock_highx_residual_correction_residual_before = (
                    highx_correction.residual_before
                )
                fp_xblock_highx_residual_correction_residual_after = (
                    highx_correction.residual_after
                )
                fp_xblock_highx_residual_correction_improvement_ratio = (
                    highx_correction.improvement_ratio
                )
                fp_xblock_highx_residual_correction_elapsed_s = (
                    highx_correction.elapsed_s
                )
                fp_xblock_highx_residual_correction_direction_count = (
                    highx_correction.direction_count
                )
                fp_xblock_highx_residual_correction_direction_names = (
                    highx_correction.direction_names
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
                sxblock_rescue = run_sparse_sxblock_rescue_stage(
                    context=SparseSXBlockRescueContext(
                        op=op,
                        current_result=res_reduced,
                        matvec=mv_reduced,
                        rhs=rhs_reduced,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                        drop_tol=float(sparse_drop_tol),
                        drop_rel=float(sparse_drop_rel),
                        ilu_drop_tol=float(sparse_ilu_drop_tol),
                        fill_factor=float(sparse_ilu_fill),
                        preconditioner=(
                            precond_sparse_xblock_current or preconditioner_reduced
                        ),
                        replay_state=ksp_replay,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        target=float(target_reduced),
                        precondition_side=gmres_precond_side,
                        solver_kind=_solver_kind("incremental")[0],
                        emit=emit,
                        mark=_mark,
                        seed_builder=_compute_rhsmode1_sxblock_tz_sparse_host_seed,
                        gmres_solver=gmres_solve_with_history_scipy,
                        parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
                        record_replay_problem=rhs1_record_ksp_replay_problem,
                    )
                )
                res_reduced = sxblock_rescue.result
            if (
                float(res_reduced.residual_norm) > target_reduced
                and (not skip_global_sparse_after_xblock)
            ):
                reduced_sparse_retry = run_rhs1_full_sparse_retry_stage(
                    RHS1FullSparseRetryStageContext(
                        op=op,
                        result=res_reduced,
                        residual_vec=residual_vec,
                        rhs=rhs_reduced,
                        matvec=mv_reduced,
                        target=float(target_reduced),
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        precondition_side=gmres_precond_side,
                        sparse_kind_use=sparse_kind_use,
                        sparse_exact_lu=bool(sparse_exact_lu),
                        sparse_drop_tol=float(sparse_drop_tol),
                        sparse_drop_rel=float(sparse_drop_rel),
                        sparse_ilu_drop_tol=float(sparse_ilu_drop_tol),
                        sparse_ilu_fill=float(sparse_ilu_fill),
                        sparse_ilu_dense_max=int(sparse_ilu_dense_max),
                        sparse_dense_cache_max=int(sparse_dense_cache_max),
                        sparse_use_matvec=bool(sparse_use_matvec),
                        sparse_jax_reg=float(sparse_jax_config.reg),
                        sparse_jax_omega=float(sparse_jax_config.omega),
                        sparse_jax_sweeps=int(sparse_jax_config.sweeps),
                        use_implicit=bool(use_implicit),
                        use_pas_projection=bool(use_pas_projection),
                        active_size=int(active_size),
                        large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue_active),
                        rhs1_polish_enabled=bool(rhs1_polish_enabled),
                        emit=emit,
                        mark=_mark,
                        cache_key_builder=_rhsmode1_sparse_cache_key,
                        precond_dtype=_precond_dtype,
                        build_sparse_jax_preconditioner_from_matvec=(
                            _build_sparse_jax_preconditioner_from_matvec
                        ),
                        host_sparse_direct_allowed=_rhsmode1_host_sparse_direct_allowed,
                        sparse_operator_preconditioned_rescue_allowed=(
                            _rhsmode1_sparse_operator_preconditioned_rescue_allowed
                        ),
                        build_point_preconditioner_operator=(
                            _build_rhsmode1_preconditioner_operator_point
                        ),
                        apply_cached_operator=apply_v3_full_system_operator_cached,
                        host_sparse_factor_dtype=_host_sparse_factor_dtype,
                        sparse_factor_cache_key=_sparse_factor_cache_key,
                        explicit_sparse_host_direct_allowed=(
                            _rhsmode1_explicit_sparse_host_direct_allowed
                        ),
                        maybe_full_sparse_pattern=_maybe_rhsmode1_full_sparse_pattern,
                        build_host_sparse_direct_factor_from_matvec=(
                            _build_host_sparse_direct_factor_from_matvec
                        ),
                        build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
                        host_sparse_direct_refine_steps=_host_sparse_direct_refine_steps,
                        direct_solve_with_refinement=(
                            _host_direct_solve_with_refinement
                        ),
                        ilu_solve_with_refinement=(
                            _host_sparse_direct_solve_with_refinement
                        ),
                        host_sparse_direct_polish=_host_sparse_direct_polish,
                        parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
                        gmres_solver=gmres_solve_with_history_scipy,
                        solve_linear_with_residual=_solve_linear,
                        run_measured_linear_candidate=(
                            rhs1_run_measured_linear_candidate_and_update_replay
                        ),
                        accept_sparse_retry_candidate=(
                            rhs1_accept_sparse_retry_candidate_and_update_replay
                        ),
                        replay_state=ksp_replay,
                        solver_kind=_solver_kind("incremental")[0],
                        peak_rss_mb=_rss_mb,
                        sparse_ilu_cache=_RHSMODE1_SPARSE_ILU_CACHE,
                        problem_size=int(active_size),
                        cache_active_size=int(active_size),
                        scope="reduced",
                        use_active_dof_mode=True,
                        force_host_sparse_direct=bool(large_cpu_sparse_rescue_active),
                        enable_operator_preconditioned_rescue=False,
                        require_lower_diag=False,
                        measured_returns_residual_vec=False,
                        implicit_solver_returns_residual_vec=False,
                        accept_candidate_residual_vec=False,
                        compute_scipy_residual_vec=False,
                    )
                )
                res_reduced = reduced_sparse_retry.result
                residual_vec = reduced_sparse_retry.residual_vec
                if reduced_sparse_retry.dense_matrix_cache is not None:
                    dense_matrix_cache = reduced_sparse_retry.dense_matrix_cache
                host_sparse_direct_used = (
                    host_sparse_direct_used
                    or reduced_sparse_retry.host_sparse_direct_used
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
        res_reduced, residual_vec, _accepted = run_rhs1_reduced_dense_fallback_admission_stage(
            context=RHS1ReducedDenseFallbackAdmissionStageContext(
                stage_context=RHS1ReducedDenseFallbackStageContext(
                    candidate_context=RHS1ReducedDenseFallbackCandidateContext(
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
                    current_result=res_reduced,
                    current_residual_vec=residual_vec,
                    target=float(target_reduced),
                ),
                dense_fallback_max=int(dense_fallback_max),
                residual_norm_true=float(residual_norm_true),
                reported_residual_norm=float(res_reduced.residual_norm),
                active_size=int(active_size),
                rhs_mode=int(op.rhs_mode),
                include_phi1=bool(op.include_phi1),
                has_fp=op.fblock.fp is not None,
                disable_dense_pas=bool(disable_dense_pas),
                any_dense_path_allowed=bool(
                    dense_backend_allowed
                    or host_dense_fallback_allowed
                    or dense_krylov_allowed
                ),
                host_sparse_direct_used=bool(host_sparse_direct_used),
                backend=jax.default_backend(),
                host_sparse_skip_ratio=float(_rhsmode1_host_sparse_skip_dense_ratio()),
                cs0_dense_fallback_allowed=bool(cs0_dense_fallback_allowed),
                cs0_sparse_first=bool(cs0_sparse_first),
                cs0_petsc_compat=bool(cs0_petsc_compat),
            ),
            replay_state=ksp_replay,
            accept_candidate=rhs1_accept_measured_candidate_and_update_replay,
            emit=emit,
            mark=_mark,
            peak_rss_mb=_rss_mb,
        )
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
        ):
            scipy_rescue = run_rhs1_scipy_rescue_stage(
                RHS1ScipyRescueStageContext(
                    op=op,
                    result=res_reduced,
                    residual_vec=residual_vec,
                    matvec=mv_reduced,
                    rhs=rhs_reduced,
                    preconditioner=preconditioner_reduced,
                    strong_preconditioner=strong_preconditioner_reduced,
                    preconditioner_name=rhs1_precond_kind or "none",
                    strong_preconditioner_name=strong_precond_kind or "strong",
                    target=float(target_reduced),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    precond_side=gmres_precond_side,
                    active_size=int(active_size),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    use_implicit=bool(use_implicit),
                    skip_global_sparse_after_xblock=bool(skip_global_sparse_after_xblock),
                    elapsed_s=t.elapsed_s,
                    emit=emit,
                    mark=_mark,
                )
            )
            res_reduced = scipy_rescue.result
            rhsmode1_general_metadata.update(scipy_rescue.metadata)
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
            dense_ksp_outcome = solve_rhs1_dense_ksp_full(
                RHS1DenseKSPFullSolveContext(
                    matvec=mv,
                    rhs=rhs,
                    x0=x0,
                    total_size=int(op.total_size),
                    phi1_size=int(op.phi1_size),
                    n_species=int(op.n_species),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    nxi_for_x=nxi_for_x,
                    extra_size=int(op.extra_size),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    solve_linear=_solve_linear,
                ),
                emit=emit,
            )
            rhs1_record_ksp_replay_problem(
                ksp_replay,
                matvec_fn=dense_ksp_outcome.replay_matvec,
                b_vec=dense_ksp_outcome.replay_rhs,
                precond_fn=None,
                x0_vec=x0,
                precond_side="none",
                solver_kind=_solver_kind("incremental")[0],
            )
            result = dense_ksp_outcome.result
        else:
            preconditioner_full = None
            bicgstab_preconditioner_full = None
            host_dense_shortcut_full = _rhsmode1_host_dense_shortcut_allowed(
                op=op,
                active_size=int(op.total_size),
                use_implicit=bool(use_implicit),
                solve_method_kind=str(solve_method_kind),
            )

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

            base_precond_setup = setup_rhs1_full_base_preconditioner(
                RHS1FullBasePreconditionerSetupContext(
                    rhs=rhs,
                    rhs1_precond_enabled=bool(rhs1_precond_enabled),
                    host_dense_shortcut=bool(host_dense_shortcut_full),
                    rhs1_bicgstab_kind=rhs1_bicgstab_kind,
                    rhs1_precond_kind=rhs1_precond_kind,
                    solve_method=solve_method,
                    solve_method_kind=solve_method_kind,
                    emit=emit,
                    solver_kind=_solver_kind,
                    build_rhs1_preconditioner=_build_rhs1_preconditioner_full,
                    build_collision_preconditioner=lambda: _build_rhsmode1_collision_preconditioner(
                        op=op
                    ),
                )
            )
            preconditioner_full = base_precond_setup.preconditioner
            bicgstab_preconditioner_full = base_precond_setup.bicgstab_preconditioner
            if host_dense_shortcut_full:
                host_dense_shortcut_full_outcome = run_rhs1_full_host_dense_shortcut_stage(
                    context=RHS1FullHostDenseShortcutContext(
                        enabled=True,
                        solve_context=HostDenseFullSolveContext(
                            matvec=mv,
                            rhs=rhs,
                            total_size=int(op.total_size),
                        ),
                        current_result=None,
                        current_residual_vec=residual_vec,
                        x0=x0,
                        total_size=int(op.total_size),
                    ),
                    replay_state=ksp_replay,
                    record_replay_problem=rhs1_record_ksp_replay_problem,
                    solver_kind=_solver_kind,
                    emit=emit,
                    mark=_mark,
                )
                result = host_dense_shortcut_full_outcome.result
                residual_vec = host_dense_shortcut_full_outcome.residual_vec
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
                result, residual_vec, _primary_elapsed_s = (
                    rhs1_run_primary_krylov_and_update_replay(
                        replay_state=ksp_replay,
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=preconditioner_full,
                        x0_vec=x0,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        solve_method=solve_method,
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear_with_residual,
                        solver_kind=_solver_kind(solve_method)[0],
                        returns_residual_vec=True,
                        current_residual_vec=residual_vec,
                        result_ready=_block_gmres_result_ready,
                        progress_start=rhs1_progress_notes.krylov_start,
                        mark=_mark,
                        mark_start="rhs1_krylov_solve_start",
                        mark_done="rhs1_krylov_solve_done",
                        block_residual_until_ready=True,
                    )
                )
            result, residual_vec, _accepted, _retry_elapsed_s = (
                rhs1_retry_without_preconditioner_if_nonfinite(
                    allowed=(not host_dense_shortcut_full) and preconditioner_full is not None,
                    replay_state=ksp_replay,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv,
                    b_vec=rhs,
                    x0_vec=x0,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    solve_method=solve_method,
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear_with_residual,
                    solver_kind=_solver_kind(solve_method)[0],
                    result_is_finite=_gmres_result_is_finite,
                    returns_residual_vec=True,
                    result_ready=_block_gmres_result_ready,
                    mark=_mark,
                    mark_start="rhs1_krylov_solve_retry_start",
                    mark_done="rhs1_krylov_solve_retry_done",
                    block_residual_until_ready=True,
                    emit=emit,
                )
            )
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
                result, residual_vec, preconditioner_full, _accepted, _bicgstab_fallback_elapsed_s = (
                    rhs1_run_bicgstab_gmres_fallback_if_allowed(
                        allowed=True,
                        replay_state=ksp_replay,
                        current_result=result,
                        current_residual_vec=residual_vec,
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=preconditioner_full,
                        preconditioner_enabled=bool(rhs1_precond_enabled),
                        build_preconditioner=_build_rhs1_preconditioner_full,
                        x0_vec=x0,
                        tol=float(tol),
                        atol=float(atol),
                        restart=int(restart),
                        maxiter=maxiter,
                        precond_side=gmres_precond_side,
                        solve_linear=_solve_linear_with_residual,
                        target=float(bicgstab_fallback_target),
                        returns_residual_vec=True,
                        emit=emit,
                    )
                )
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
            stage2_controls = rhs1_stage2_retry_controls_from_env(
                restart=int(restart),
                maxiter=maxiter,
                active_size=int(op.total_size),
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
            )
            stage2_method = str(stage2_controls.method)
            result, residual_vec, preconditioner_full, _accepted, _stage2_elapsed_s = (
                rhs1_run_stage2_retry_if_allowed(
                    allowed=True,
                    replay_state=ksp_replay,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    preconditioner_enabled=bool(rhs1_precond_enabled),
                    build_preconditioner=_build_rhs1_preconditioner_full,
                    controls=stage2_controls,
                    tol=float(tol),
                    atol=float(atol),
                    precond_side=gmres_precond_side,
                    solve_linear=_solve_linear_with_residual,
                    solver_kind=_solver_kind(stage2_method)[0],
                    candidate_name=f"stage2_full:{stage2_method}",
                    baseline_name="current_full",
                    target=float(target),
                    peak_rss_mb=_rss_mb,
                    returns_residual_vec=True,
                    emit=emit,
                    label="stage2 GMRES",
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
        collision_retry_allowed = rhs1_collision_retry_allowed(
            residual_norm=float(result.residual_norm),
            target=float(target),
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            rhs1_precond_kind=rhs1_precond_kind,
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            strong_precond_trigger=bool(strong_precond_trigger),
        )
        result, residual_vec, bicgstab_preconditioner_full, _accepted, _collision_elapsed_s = (
            rhs1_run_collision_retry_if_allowed(
                allowed=bool(collision_retry_allowed),
                replay_state=ksp_replay,
                current_result=result,
                current_residual_vec=residual_vec,
                matvec_fn=mv,
                b_vec=rhs,
                precond_fn=bicgstab_preconditioner_full,
                build_preconditioner=lambda: _build_rhsmode1_collision_preconditioner(op=op),
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precond_side=gmres_precond_side,
                solve_linear=_solve_linear_with_residual,
                solver_kind=_solver_kind("incremental")[0],
                target=float(target),
                returns_residual_vec=True,
                emit=emit,
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
        def _build_rhs1_full_strong_preconditioner_for_retry(strong_kind: str):
            dd_block_theta = rhs1_dd_setup.block("theta")
            dd_overlap_theta = rhs1_dd_setup.overlap("theta", default=1)
            dd_block_zeta = rhs1_dd_setup.block("zeta")
            dd_overlap_zeta = rhs1_dd_setup.overlap("zeta", default=1)
            return _build_rhs1_strong_preconditioner_full_from_kind(
                op=op,
                strong_precond_kind=strong_kind,
                rhs1_precond_kind=rhs1_precond_kind,
                residual_norm=float(result.residual_norm),
                dd_block_theta=dd_block_theta,
                dd_overlap_theta=dd_overlap_theta,
                dd_block_zeta=dd_block_zeta,
                dd_overlap_zeta=dd_overlap_zeta,
            )

        full_strong_retry = run_rhs1_full_strong_retry_stage(
            RHS1FullStrongRetryStageContext(
                strong_precond_env=strong_precond_env,
                strong_control=strong_control,
                has_extra_constraint_block=int(op.constraint_scheme) == 2 and int(op.extra_size) > 0,
                has_fp=op.fblock.fp is not None,
                has_pas=op.fblock.pas is not None,
                rhs1_precond_kind=rhs1_precond_kind,
                geom_scheme=int(geom_scheme),
                total_size=int(op.total_size),
                n_theta=int(op.n_theta),
                n_zeta=int(op.n_zeta),
                max_l=int(np.max(nxi_for_x)) if nxi_for_x.size else 0,
                nxi_for_x_sum=int(np.sum(nxi_for_x)) if nxi_for_x.size else 0,
                shard_axis=_matvec_shard_axis(op),
                device_count=int(jax.device_count()),
                pas_auto_strong_ratio=float(pas_auto_strong_ratio),
                current_result=result,
                current_residual_vec=residual_vec,
                matvec=mv,
                rhs=rhs,
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precondition_side=gmres_precond_side,
                solver_kind=_solver_kind("incremental")[0],
                target=float(target),
                peak_rss_mb=_rss_mb(),
                emit=emit,
                mark=_mark,
                replay_state=ksp_replay,
                build_strong_preconditioner=_build_rhs1_full_strong_preconditioner_for_retry,
                run_measured_candidate=rhs1_run_measured_linear_candidate_and_update_replay,
                solve_linear=_solve_linear_with_residual,
            )
        )
        result = full_strong_retry.result
        residual_vec = full_strong_retry.residual_vec
        strong_precond_kind = full_strong_retry.selected_kind
        result, residual_vec, _accepted, _schur_elapsed_s = (
            rhs1_run_full_pas_schur_rescue_from_env(
                replay_state=ksp_replay,
                current_result=result,
                current_residual_vec=residual_vec,
                matvec_fn=mv,
                b_vec=rhs,
                build_preconditioner=lambda: _build_rhsmode1_schur_preconditioner(op=op),
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precond_side=gmres_precond_side,
                solve_linear=_solve_linear_with_residual,
                solver_kind=_solver_kind("incremental")[0],
                target=float(target),
                rhs_mode=int(op.rhs_mode),
                include_phi1=bool(op.include_phi1),
                has_pas=op.fblock.pas is not None,
                n_species=int(op.n_species),
                active_size=int(active_size),
                emit=emit,
            )
        )
        large_cpu_sparse_rescue_full = _rhsmode1_large_cpu_sparse_rescue_allowed(
            op=op,
            solve_method_kind=solve_method_kind,
            active_size=int(op.total_size),
            sparse_max_size=int(sparse_max_size),
            preconditioner_x=int(preconditioner_x),
            residual_norm=float(result.residual_norm),
            target=float(target),
        )

        full_sparse_setup = rhs1_full_sparse_rescue_setup(
            RHS1FullSparseRescueSetupContext(
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
                sparse_exact_lu=bool(sparse_exact_lu),
                use_implicit=bool(use_implicit),
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
                rhs1_precond_kind=rhs1_precond_kind,
                emit=emit,
                host_sparse_direct_allowed=_rhsmode1_host_sparse_direct_allowed,
                large_cpu_sparse_exact_lu_allowed=_rhsmode1_large_cpu_sparse_exact_lu_allowed,
            )
        )
        sparse_order = full_sparse_setup.ordering
        sparse_enabled = bool(full_sparse_setup.enabled)
        sparse_kind_use = str(full_sparse_setup.kind_use)
        sparse_exact_lu = bool(full_sparse_setup.sparse_exact_lu)

        dense_matrix_cache: np.ndarray | None = None
        host_sparse_direct_used = False
        if sparse_enabled and float(result.residual_norm) > target:
            full_sparse_retry = run_rhs1_full_sparse_retry_stage(
                RHS1FullSparseRetryStageContext(
                    op=op,
                    result=result,
                    residual_vec=residual_vec,
                    rhs=rhs,
                    matvec=mv,
                    target=float(target),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    precondition_side=gmres_precond_side,
                    sparse_kind_use=sparse_kind_use,
                    sparse_exact_lu=bool(sparse_exact_lu),
                    sparse_drop_tol=float(sparse_drop_tol),
                    sparse_drop_rel=float(sparse_drop_rel),
                    sparse_ilu_drop_tol=float(sparse_ilu_drop_tol),
                    sparse_ilu_fill=float(sparse_ilu_fill),
                    sparse_ilu_dense_max=int(sparse_ilu_dense_max),
                    sparse_dense_cache_max=int(sparse_dense_cache_max),
                    sparse_use_matvec=bool(sparse_use_matvec),
                    sparse_jax_reg=float(sparse_jax_config.reg),
                    sparse_jax_omega=float(sparse_jax_config.omega),
                    sparse_jax_sweeps=int(sparse_jax_config.sweeps),
                    use_implicit=bool(use_implicit),
                    use_pas_projection=bool(use_pas_projection),
                    active_size=int(active_size),
                    large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue_full),
                    rhs1_polish_enabled=bool(rhs1_polish_enabled),
                    emit=emit,
                    mark=_mark,
                    cache_key_builder=_rhsmode1_sparse_cache_key,
                    precond_dtype=_precond_dtype,
                    build_sparse_jax_preconditioner_from_matvec=(
                        _build_sparse_jax_preconditioner_from_matvec
                    ),
                    host_sparse_direct_allowed=_rhsmode1_host_sparse_direct_allowed,
                    sparse_operator_preconditioned_rescue_allowed=(
                        _rhsmode1_sparse_operator_preconditioned_rescue_allowed
                    ),
                    build_point_preconditioner_operator=(
                        _build_rhsmode1_preconditioner_operator_point
                    ),
                    apply_cached_operator=apply_v3_full_system_operator_cached,
                    host_sparse_factor_dtype=_host_sparse_factor_dtype,
                    sparse_factor_cache_key=_sparse_factor_cache_key,
                    explicit_sparse_host_direct_allowed=(
                        _rhsmode1_explicit_sparse_host_direct_allowed
                    ),
                    maybe_full_sparse_pattern=_maybe_rhsmode1_full_sparse_pattern,
                    build_host_sparse_direct_factor_from_matvec=(
                        _build_host_sparse_direct_factor_from_matvec
                    ),
                    build_sparse_ilu_from_matvec=_build_sparse_ilu_from_matvec,
                    host_sparse_direct_refine_steps=_host_sparse_direct_refine_steps,
                    direct_solve_with_refinement=_host_direct_solve_with_refinement,
                    ilu_solve_with_refinement=(
                        _host_sparse_direct_solve_with_refinement
                    ),
                    host_sparse_direct_polish=_host_sparse_direct_polish,
                    parse_polish_gmres_config=rhs1_parse_polish_gmres_config,
                    gmres_solver=gmres_solve_with_history_scipy,
                    solve_linear_with_residual=_solve_linear_with_residual,
                    run_measured_linear_candidate=(
                        rhs1_run_measured_linear_candidate_and_update_replay
                    ),
                    accept_sparse_retry_candidate=(
                        rhs1_accept_sparse_retry_candidate_and_update_replay
                    ),
                    replay_state=ksp_replay,
                    solver_kind=_solver_kind("incremental")[0],
                    peak_rss_mb=_rss_mb,
                    sparse_ilu_cache=_RHSMODE1_SPARSE_ILU_CACHE,
                )
            )
            result = full_sparse_retry.result
            residual_vec = full_sparse_retry.residual_vec
            dense_matrix_cache = full_sparse_retry.dense_matrix_cache
            host_sparse_direct_used = bool(full_sparse_retry.host_sparse_direct_used)
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
        result, residual_vec, _accepted = run_rhs1_full_dense_fallback_stage(
            context=RHS1FullDenseFallbackStageContext(
                candidate_context=RHS1FullDenseFallbackContext(
                    matvec=mv,
                    rhs=rhs,
                    current_result=result,
                    current_residual_vec=residual_vec,
                    total_size=int(op.total_size),
                    constraint_scheme=int(op.constraint_scheme),
                    dense_matrix_cache=dense_matrix_cache,
                    dense_backend_allowed=bool(dense_backend_allowed),
                    residual_norm_check=float(residual_norm_check),
                    target=float(target),
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    backend=jax.default_backend(),
                ),
                dense_fallback_max=int(dense_fallback_max),
                residual_norm_true=float(residual_norm_true),
                active_size=int(active_size),
                rhs_mode=int(op.rhs_mode),
                include_phi1=bool(op.include_phi1),
                has_fp=op.fblock.fp is not None,
                any_dense_path_allowed=bool(
                    dense_backend_allowed
                    or host_dense_fallback_allowed
                    or dense_krylov_allowed
                ),
                host_sparse_direct_used=bool(host_sparse_direct_used),
                host_sparse_skip_ratio=float(_rhsmode1_host_sparse_skip_dense_ratio()),
                cs0_sparse_first=bool(cs0_sparse_first),
            ),
            replay_state=ksp_replay,
            accept_candidate=rhs1_accept_measured_candidate_and_update_replay,
            solve_linear_with_residual=_solve_linear_with_residual,
            emit=emit,
            mark=_mark,
            peak_rss_mb=_rss_mb,
        )
    return finalize_profile_response_linear_solve(
        ProfileResponseLinearFinalizationContext(
            op=op,
            rhs=rhs,
            result=result,
            residual_vec=residual_vec,
            ksp_replay=ksp_replay,
            ksp_diagnostics_context=rhs1_ksp_diagnostics_context,
            tol=float(tol),
            atol=float(atol),
            solve_method=str(solve_method),
            active_size=int(active_size),
            used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
            used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
            use_implicit=bool(use_implicit),
            backend=jax.default_backend(),
            metadata_parts=(
                pas_tz_guarded_correction_metadata,
                rhsmode1_general_metadata,
            ),
            emit=emit,
            elapsed_s=t.elapsed_s,
        )
    )


solve_v3_full_system_linear_gmres_jit = jax.jit(
    solve_v3_full_system_linear_gmres,
    static_argnames=("tol", "atol", "restart", "maxiter", "solve_method", "identity_shift"),
)
