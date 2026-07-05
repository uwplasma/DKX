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
from sfincs_jax.problems.profile_solver_diagnostics import (
    emit_newton_krylov_ksp_history as _emit_newton_krylov_ksp_history,
)
from sfincs_jax.solver import (
    GMRESSolveResult, bicgstab_solve_with_residual, bicgstab_solve_with_history_scipy,
    dense_krylov_solve_from_matrix_with_residual, dense_solve_from_matrix, dense_solve_from_matrix_row_scaled,
    gmres_solve, gmres_solve_jit, gmres_solve_with_residual, gmres_solve_with_residual_jit, gmres_solve_distributed,
    gmres_solve_with_residual_distributed, distributed_gmres_enabled, explicit_left_preconditioned_gmres_scipy,
    fgmres_cycle_jit_solve_with_residual, fgmres_solve_with_residual, fgmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy, gcrotmk_solve_with_history_scipy, lgmres_solve_with_history_scipy,
    tfqmr_solve_with_residual,
)
from sfincs_jax.solver import (
    recycled_initial_guess as _recycled_initial_guess, small_regularized_lstsq as _small_regularized_lstsq,
)
from sfincs_jax.discretization.structured_velocity import factor_block_tridiagonal
from sfincs_jax.solvers.preconditioner_pas_policy import adaptive_pas_smoother
from sfincs_jax.solvers.explicit_sparse import (
    SparseDecision, SparseOperatorBundle, admit_sparse_factor_against_operator, analyze_sparse_symbolic_structure,
    build_operator_from_pattern, estimate_csr_nbytes, estimate_dense_nbytes, estimate_multifrontal_direct_lu_nbytes,
    wrap_sparse_factor_with_coarse_correction,
)
from sfincs_jax.operators.profile_device_sparse import (
    device_csr_from_matrix, validate_device_csr_matvec,
)
from sfincs_jax.solvers.preconditioner_domain_decomposition import (  # compatibility exports for legacy tests/debug scripts
    _dd_core_patch_ranges,
    _rhs1_dd_auto_block_size,
    _rhs1_dd_coarse_block_size,
    _rhs1_dd_coarse_block_sizes,
    _rhs1_dd_coarse_level_count,
)
from sfincs_jax.solvers.memory_model import estimate_sparse_pc_memory
from sfincs_jax.solvers.preconditioner_pas_policy import (
    build_pas_tz_memory_fallback, estimate_rhs1_pas_tz_build_bytes as _estimate_rhs1_pas_tz_build_bytes,
    pas_tokamak_theta_preconditioner_applicable as _pas_tokamak_theta_preconditioner_applicable,
    pas_tz_preconditioner_applicable as _pas_tz_preconditioner_applicable,
    pas_tz_preconditioner_memory_safe as _pas_tz_preconditioner_memory_safe, resolve_pas_tz_guarded_correction_kind,
    rhs1_pas_adaptive_smoother_allowed as _rhs1_pas_adaptive_smoother_allowed_impl,
    rhs1_pas_adaptive_smoother_controls_from_env,
    rhs1_pas_default_preconditioner_kind as _rhs1_pas_default_preconditioner_kind,
    rhs1_pas_force_full_decision_from_env,
    rhs1_pas_preconditioner_probe_admitted as _rhs1_pas_preconditioner_probe_admitted,
    rhs1_pas_preconditioner_probe_config_from_env as _rhs1_pas_preconditioner_probe_config_from_env,
    rhs1_pas_preconditioner_probe_large_collision_skip as _rhs1_pas_preconditioner_probe_large_collision_skip,
    rhs1_pas_preconditioner_probe_uses_collision as _rhs1_pas_preconditioner_probe_uses_collision,
    rhs1_pas_small_near_zero_er_kind as _rhs1_pas_small_near_zero_er_kind, rhs1_pas_tz_guarded_strong_retry_from_env,
    rhs1_pas_tz_max_bytes as _rhs1_pas_tz_max_bytes,
)
from sfincs_jax.solvers.preconditioning import (
    build_rhs1_preconditioner_from_kind as _dispatch_rhs1_preconditioner_from_kind,
)
from sfincs_jax.operators.profile_kinetic import (
    select_structured_rhs1_fblock_csr_operator, select_structured_rhs1_fblock_operator,
)
from sfincs_jax.operators.profile_full_system import (
    build_active_projected_rhs1_full_csr_preconditioner, build_direct_active_fortran_v3_reduced_pmat_preconditioner,
    _try_build_structured_rhs1_full_csr_operator_bundle,
    select_active_fortran_v3_reduced_support_mode_preconditioner, solve_structured_rhs1_full_csr,
)
from sfincs_jax.problems.profile_policies import (
    RHS1PreconditionerRouteSetupContext,
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
    resolve_rhs1_preconditioner_route_setup,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_gpu_sparse_fallback_skip_allowed_current_backend as _rhs1_gpu_sparse_fallback_skip_allowed,
)
from sfincs_jax.solvers.preconditioner_schur_profile import (
    resolve_rhs1_schur_base_kind,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    RHS1KSPReplayState, RHS1SkipPrimaryKrylovSeedContext, rhs1_accept_candidate_and_update_replay,
    rhs1_accept_measured_candidate_and_update_replay, rhs1_accept_sparse_retry_candidate_and_update_replay,
    rhs1_accept_smoother_candidate_and_update_replay, rhs1_record_ksp_replay_problem,
    rhs1_retry_without_preconditioner_if_nonfinite, rhs1_run_adaptive_smoother_and_update_replay,
    rhs1_run_bicgstab_gmres_fallback_if_allowed, rhs1_run_collision_retry_if_allowed,
    rhs1_run_fast_post_xblock_polish, rhs1_run_full_pas_schur_rescue_from_env,
    rhs1_run_linear_candidate_and_update_replay, rhs1_run_measured_linear_candidate_and_update_replay,
    rhs1_run_primary_krylov_and_update_replay, rhs1_run_stage2_retry_if_allowed,
    rhs1_seed_skip_primary_krylov_and_update_replay, rhs1_skip_primary_krylov_reason,
)
from sfincs_jax.problems.profile_dense import (
    HostDenseFullSolveContext, HostDenseReducedSolveContext, RHS1AutoHostSolveContext,
    RHS1Constraint0PETScCompatSolveContext, RHS1DenseKSPFullSolveContext, RHS1DenseKSPReducedSolveContext,
    RHS1FullDenseFallbackContext, RHS1FullDenseFallbackStageContext, RHS1FullHostDenseShortcutContext,
    RHS1DenseProbeStageContext, RHS1PostKrylovDenseShortcutEvaluationContext,
    RHS1ReducedDenseFallbackAdmissionStageContext, RHS1ReducedDenseFallbackCandidateContext,
    RHS1ReducedDenseFallbackStageContext, RHS1ReducedHostDenseShortcutContext, RHS1ScipyRescueStageContext,
    RHS1SparseHostSafeSolveContext, RHS1StructuredCSRSolveContext, build_profile_linear_solve_dispatch,
    rhs1_dense_shortcut_setup_from_env, rhs1_early_dense_shortcut_decision, rhs1_evaluate_post_krylov_dense_shortcut,
    rhs1_fp_preconditioner_probe_kind_from_env, rhs1_small_gmres_max_from_env, run_rhs1_scipy_rescue_stage,
    solve_rhs1_structured_full_csr_explicit, solve_rhs1_constraint0_petsc_compat, solve_rhs1_dense_ksp_full, solve_rhs1_dense_ksp_reduced,
    solve_v3_full_system_structured_csr, try_rhs1_auto_host_solve, try_rhs1_sparse_host_safe_solve,
    run_rhs1_dense_probe_stage, run_rhs1_full_dense_fallback_stage, run_rhs1_full_host_dense_shortcut_stage,
    run_rhs1_reduced_dense_fallback_admission_stage, run_rhs1_reduced_host_dense_shortcut_stage,
)
from sfincs_jax.problems.profile_phi1_newton import (
    solve_v3_full_system_newton_krylov, solve_v3_full_system_newton_krylov_history,
)
from sfincs_jax.problems.profile_preconditioner_build import (
    RHS1FullBasePreconditionerSetupContext, RHS1FullPreconditionerBuildContext, RHS1FullStrongRetryStageContext,
    RHS1ReducedPreconditionerBuildContext, RHS1ReducedStrongRetryStageContext, _build_rhsmode1_block_preconditioner,
    _build_rhs1_preconditioner_from_kind, _build_rhs1_strong_preconditioner_full_from_kind,
    _build_rhs1_strong_preconditioner_reduced_from_kind, _build_rhsmode1_collision_preconditioner,
    _build_rhsmode1_pas_hybrid_preconditioner, _build_rhsmode1_pas_lite_preconditioner,
    _build_rhsmode1_pas_schur_preconditioner, _build_rhsmode1_pas_tokamak_theta_preconditioner,
    _build_rhsmode1_pas_tz_preconditioner, _build_rhsmode1_pas_xblock_ilu_preconditioner,
    _build_rhsmode1_species_block_preconditioner, _build_rhsmode1_sxblock_tz_preconditioner,
    _build_rhsmode1_theta_dd_preconditioner, _build_rhsmode1_theta_line_preconditioner,
    _build_rhsmode1_theta_zeta_preconditioner, _build_rhsmode1_xblock_tz_preconditioner,
    _build_rhsmode1_xblock_tz_lmax_preconditioner, _build_rhsmode1_xblock_tz_sparse_preconditioner,
    _build_rhsmode1_xmg_preconditioner, _build_rhsmode1_zeta_dd_preconditioner,
    _build_rhsmode1_zeta_line_preconditioner, _build_rhsmode23_tzfft_preconditioner,
    _compute_rhsmode1_sxblock_tz_sparse_host_seed, build_rhs1_full_preconditioner,
    build_rhs1_reduced_preconditioner_with_fallback, run_rhs1_full_strong_retry_stage,
    run_rhs1_reduced_strong_retry_stage, setup_rhs1_full_base_preconditioner,
)
from sfincs_jax.problems.profile_sparse_qi import (
    attempt_matrixfree_qi_device_seed_if_requested, build_matrixfree_qi_device_seed_setup,
)
from sfincs_jax.problems.profile_sparse_direct import (
    build_host_sparse_direct_factor_from_matvec as _build_host_sparse_direct_factor_from_matvec,
    build_sparse_jax_preconditioner_from_matvec as _build_sparse_jax_preconditioner_from_matvec,
    host_physical_memory_mb as _host_physical_memory_mb, host_sparse_direct_polish as _host_sparse_direct_polish,
    matvec_submatrix as _matvec_submatrix, maybe_rhsmode1_full_sparse_pattern as _maybe_rhsmode1_full_sparse_pattern,
    rhsmode1_explicit_sparse_pattern_probe_enabled as _rhsmode1_explicit_sparse_pattern_probe_enabled,
    rhsmode1_sparse_cache_key as _rhsmode1_sparse_cache_key, sparse_factor_cache_key as _sparse_factor_cache_key,
)
from sfincs_jax.problems.profile_diagnostics import (
    SparseRescueTailMetadataContext, record_structured_fblock_preconditioner_metadata,
    sparse_rescue_tail_metadata_from_context,
)
from sfincs_jax.problems.profile_sparse_solve import (
    FortranReducedXBlockBackendContext, RequestedSparsePCGMRESBranchContext, SparsePCDirectTailFactorSetupContext,
    SparsePCDirectTailRescuePolicySetupContext, SparsePCGenericBranchSetupContext, SparsePCFactorPreflightRunContext,
    SparsePCResidualCorrectionStageContext, SparsePCAutoPreflightRetryStageContext,
    SparsePCTrueCoupledCoarseStageContext, ExplicitSparseMinimumNormBranchContext,
    ExplicitSparseHostDirectBranchContext, RHS1FullSparseRetryStageContext, SparseHostOrILUFactorBuildContext,
    SparseHostRetryCandidateContext, SparseJAXRetryPreconditionerBuildContext, SparsePCGMRESContext,
    XBlockAugmentedKrylovStageContext, XBlockFirstKrylovAttemptContext, XBlockGlobalCouplingStageContext,
    XBlockKrylovControlSetupContext, XBlockKrylovProgressCallbacksContext, XBlockKrylovSolveStageContext,
    XBlockKrylovSolveSpaceContext, XBlockMomentSchurStageContext, XBlockPostKrylovCompletionContext,
    XBlockPostSolveCorrectionContext, XBlockPreflightGateContext, XBlockProbeCoarseStageContext,
    XBlockSideProbeStageContext, XBlockSparsePCBranchContext, XBlockTwoLevelStageContext,
    apply_xblock_global_coupling_stage, apply_xblock_augmented_krylov_stage, apply_xblock_moment_schur_stage,
    apply_xblock_probe_coarse_stage, run_xblock_qi_preconditioner_pipeline, apply_xblock_side_probe_stage,
    apply_xblock_two_level_stage, build_xblock_local_preconditioner, build_xblock_krylov_matvec_setup,
    build_xblock_assembled_operator_if_requested, build_sparse_pc_direct_tail_factor_setup,
    build_sparse_pc_direct_tail_rescue_policy_setup, build_sparse_pc_generic_branch_setup,
    build_xblock_krylov_progress_callbacks, build_xblock_qi_stage_pipeline_context, evaluate_xblock_preflight_gate,
    run_sparse_pc_auto_preflight_retry_stage, run_sparse_pc_factor_preflight, run_sparse_pc_residual_correction_stage,
    run_sparse_pc_true_coupled_coarse_stage, complete_xblock_post_krylov_stage,
    resolve_sparse_pc_gmres_control_policy, prepare_xblock_initial_guess, resolve_sparse_pc_entry_policy,
    resolve_fortran_reduced_xblock_factor_policy, prepare_xblock_augmented_krylov_basis,
    prepare_xblock_krylov_solve_space, resolve_xblock_krylov_control_setup,
    resolve_xblock_global_coupling_policy_setup, resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup, resolve_xblock_sparse_pc_branch_setup, resolve_xblock_two_level_policy_setup,
    run_sparse_pc_gmres_once, run_sparse_pc_gmres_once_for_retry, FPXBlockGlobalCorrectionContext,
    FPXBlockHighXCorrectionContext, SparseSXBlockRescueContext, SparseXBlockRescueAcceptanceContext,
    SparseXBlockRescueBuildContext, SparseXBlockRescueSolveContext, accept_sparse_xblock_rescue_candidate,
    build_sparse_xblock_rescue_preconditioner, run_fp_xblock_global_correction_stage,
    run_fp_xblock_highx_residual_correction_stage, run_sparse_sxblock_rescue_stage,
    run_sparse_xblock_rescue_solve_stage, run_rhs1_full_sparse_retry_stage, run_xblock_sparse_pc_branch,
    run_xblock_krylov_solve_stage, build_sparse_host_or_ilu_factor, run_sparse_host_retry_candidate,
    build_sparse_jax_retry_preconditioner, resolve_sparse_host_or_ilu_factor_controls,
    solve_fortran_reduced_xblock_backend, finalize_sparse_pc_gmres_bundle,
    sparse_pc_gmres_finalization_bundle_from_driver_result, try_run_requested_sparse_pc_gmres_branch,
    solve_explicit_sparse_minimum_norm_branch, solve_explicit_sparse_host_direct_branch,
    finalize_xblock_assembled_operator_metadata,
)
from sfincs_jax.problems.profile_sparse_xblock import (
    xblock_sparse_pc_final_metadata_state_from_driver_scope, xblock_sparse_pc_final_payload_from_driver_state,
)
from sfincs_jax.problems.profile_preconditioner_build import (
    RHS1PostPrimaryMinresCorrectionContext, rhs1_collision_retry_allowed, rhs1_pas_force_strong_ratio_from_env,
    rhs1_pas_tz_guarded_minres_controls_from_env, rhs1_pas_weak_minres_controls_from_env, rhs1_pas_weak_minres_steps,
    rhs1_reduced_strong_selection_skip_messages, rhs1_resolved_strong_preconditioner_control,
    resolve_rhs1_reduced_strong_preconditioner_selection, rhs1_strong_preconditioner_env_from_env,
    rhs1_strong_preconditioner_control_messages, rhs1_strong_trigger_controls_from_env,
    run_rhs1_post_primary_minres_corrections,
)
from sfincs_jax.problems.profile_policies import (
    RHS1FullSparseRescueSetupContext,
    parse_rhs1_pas_tz_guarded_structured_levels as _rhs1_pas_tz_guarded_structured_levels,
    rhs1_full_sparse_rescue_setup, rhs1_sparse_jax_config_from_env, rhs1_sparse_operator_admission,
    rhs1_sparse_preconditioner_config_from_env, rhs1_sparse_rescue_initial_messages, rhs1_sparse_rescue_policy_setup,
    rhs1_sparse_rescue_tail_skip_messages, rhs1_xblock_fallback_initial_guess as _rhs1_xblock_fallback_initial_guess,
    rhsmode1_sparse_pc_default_permc_spec as _rhsmode1_sparse_pc_default_permc_spec,
    rhsmode1_sparse_pc_default_restart as _rhsmode1_sparse_pc_default_restart,
)

from sfincs_jax.problems.profile_setup import (
    SPARSE_HOST_DIRECT_SOLVE_METHODS as _SPARSE_HOST_DIRECT_SOLVE_METHODS,
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS as _SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS,
    SPARSE_HOST_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_SAFE_SOLVE_METHODS as _SPARSE_HOST_SAFE_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS as _SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
    STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS as _STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS,
    ProfileResponseLinearProblemSetupContext, build_rhs1_active_dof_state as _build_rhs1_active_dof_state_compat,
    RHS1ActiveReducedSystemSetupContext, build_rhs1_active_reduced_system_setup,
    expand_reduced_with_map, fp_pitch_mode_active_indices,
    materialize_profile_response_linear_problem, reduce_full_with_indices,
    resolve_rhs1_active_problem_setup,
    resolve_rhs1_domain_decomposition_setup, resolve_rhs1_initial_route_setup,
    resolve_rhs1_post_active_solve_policy_setup, resolve_rhs1_recycle_basis_setup,
    resolve_rhs1_reduced_mode_shape_setup,
)
from sfincs_jax.solvers import preconditioner_xblock_policy as _rhs1_xblock_policy
from sfincs_jax.solvers import preconditioner_xblock_policy as _rhs1_xblock_sparse_host_policy
from sfincs_jax.solvers.preconditioner_xblock_policy import (
    resolve_rhs1_xblock_sparse_pc_policy,
)
from sfincs_jax.solvers.preconditioner_pas_composite import (
    RHS1PasFamilyBuilders, compose_preconditioners as _compose_preconditioners,
)
from sfincs_jax.solvers.preconditioner_full_fp_kinetic import (
    build_rhs1_block_preconditioner, build_rhs1_block_preconditioner_xdiag, build_rhs1_collision_preconditioner,
)
from sfincs_jax.solvers.preconditioner_full_fp_species import (
    build_rhs1_species_block_preconditioner, build_rhs1_species_xblock_preconditioner,
)
from sfincs_jax.solvers.preconditioner_full_fp_structured import (
    build_rhs1_structured_fblock_angular_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner,
    build_rhs1_structured_fblock_fp_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner,
    build_rhs1_structured_fblock_jacobi_preconditioner, build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner,
)
from sfincs_jax.solvers.preconditioner_xblock_block_jacobi import (
    build_rhs1_sxblock_tz_preconditioner, build_rhs1_xblock_tz_lmax_preconditioner,
    build_rhs1_xblock_tz_preconditioner,
)
from sfincs_jax.solvers.preconditioner_xblock_radial import (
    build_rhs1_xmg_preconditioner, build_rhs1_xupwind_preconditioner,
)
from sfincs_jax.solvers.preconditioner_xblock_tz_sparse import (
    assemble_rhsmode1_fp_xblock_tz_sparse_matrix as _assemble_rhsmode1_fp_xblock_tz_sparse_matrix,
    assemble_selected_theta_tz_operator as _assemble_selected_theta_tz_operator,
    assemble_selected_zeta_tz_operator as _assemble_selected_zeta_tz_operator,
    build_rhs1_sxblock_tz_sparse_host_preconditioner,
    build_rhs1_xblock_tz_sparse_preconditioner, compute_rhs1_sxblock_tz_sparse_host_seed,
    get_rhsmode1_fp_xblock_assembled_host_cache as _get_rhsmode1_fp_xblock_assembled_host_cache,
    rhsmode1_fp_xblock_assembled_host_allowed as _rhsmode1_fp_xblock_assembled_host_allowed,
    rhsmode1_fp_xblock_species_decoupled_for_host_assembly as _rhsmode1_fp_xblock_species_decoupled_for_host_assembly,
    rhsmode1_fp_xblock_tz_sparse_diagonal as _rhsmode1_fp_xblock_tz_sparse_diagonal,
    rhsmode1_host_factor_probe_ok as _rhsmode1_host_factor_probe_ok,
    rhsmode1_precond_cache_key as _rhsmode1_precond_cache_key,
    rhsmode1_xblock_sparse_lu_default_max as _rhsmode1_xblock_sparse_lu_default_max,
    safe_inverse_diagonal_np as _safe_inverse_diagonal_np,
)
from sfincs_jax.solvers.preconditioner_schur_profile import (
    RHS1SchurPreconditionerBuilders, build_rhs1_schur_preconditioner,
)
from sfincs_jax.solvers.preconditioner_transport_matrix import (
    build_rhsmode23_block_preconditioner, build_rhsmode23_collision_preconditioner,
    build_rhsmode23_fp_local_geom_line_preconditioner, build_rhsmode23_fp_structured_fblock_lu_preconditioner,
    build_rhsmode23_fp_tzfft_line_preconditioner, build_rhsmode23_fp_tzfft_line_schur_preconditioner,
    build_rhsmode23_fp_tzfft_preconditioner, build_rhsmode23_fp_xblock_tz_lu_preconditioner,
    build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner, build_rhsmode23_sxblock_preconditioner,
    build_rhsmode23_tzfft_preconditioner, build_rhsmode23_xmg_preconditioner,
)
from sfincs_jax.solvers.preconditioner_domain_decomposition import (
    build_rhs1_theta_dd_preconditioner, build_rhs1_theta_line_preconditioner, build_rhs1_theta_schwarz_preconditioner,
    build_rhs1_theta_line_xdiag_preconditioner, build_rhs1_theta_zeta_preconditioner,
    build_rhs1_zeta_dd_preconditioner, build_rhs1_zeta_line_preconditioner, build_rhs1_zeta_schwarz_preconditioner,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_parse_accept_ratio, rhs1_parse_polish_gmres_config, rhs1_polish_enabled,
)
from sfincs_jax.problems.profile_policies import (
    read_bool_env as _rhs1_bool_env, read_float_env as _rhs1_float_env, read_int_env as _rhs1_int_env,
    read_post_solve_correction_policy as _read_rhs1_post_solve_correction_policy,
    read_probe_coarse_policy as _read_rhs1_probe_coarse_policy,
)
from sfincs_jax.problems.profile_policies import (
    _DIRECT_TAIL_STRUCTURED_PC_CACHE, _StructuredHostSparsePreconditionerBundle, _direct_tail_structured_pc_cache_key,
    _direct_tail_structured_pc_with_cache_metadata, _hash_numpy_array_for_cache, _is_direct_reduced_pmat_pc_kind,
    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb,
)
from sfincs_jax.operators.profile_reduced_tail import (
    _try_build_fortran_reduced_constraint1_direct_tail_bundle,
)
from sfincs_jax.operators.profile_true_operator_rescue import (
    _ResidualCoarseHostSparsePreconditionerBundle, _ResidualWindowHostSparsePreconditionerBundle,
    _ReusableTrueActionColumnCache, _TrueOperatorActiveSubmatrixPreconditionerBundle,
    _TrueOperatorCoupledCoarseLSQPreconditionerBundle, _TrueOperatorWindowLSQPreconditionerBundle,
    _expand_sparse_graph_positions, _parse_true_operator_window_specs, _rhs1_additive_rescue_nbytes,
    _rhs1_active_reduced_residual_diagnostics, _sparse_factor_nbytes_estimate,
    _true_operator_window_positions_from_residual, _try_build_true_operator_active_block_lsq_preconditioner,
    _try_build_true_operator_active_residual_block_lsq_preconditioner,
    _try_build_true_operator_active_submatrix_preconditioner,
    _try_build_true_operator_coupled_coarse_lsq_preconditioner,
    _try_build_true_operator_residual_window_lsq_preconditioner,
    _try_build_residual_coarse_host_sparse_preconditioner, _try_build_residual_window_host_sparse_preconditioner,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    rhs1_fortran_stdout_from_env, rhs1_ksp_diagnostics_controls_from_env, rhs1_ksp_history_limits_from_env,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    RHS1KSPDiagnosticsContext,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    ProfileResponseLinearFinalizationContext, finalize_profile_response_linear_solve,
)
from sfincs_jax.operators.profile_layout import (
    RHS1ActiveBlockLayout, RHS1ActiveFieldSplitOrdering, RHS1BlockLayout,
)
from sfincs_jax.solvers.preconditioner_xblock_coarse import (
    _rhs1_cap_lowmode_features, _rhs1_low_legendre_index_features, _rhs1_lowmode_angular_features,
    _rhs1_polynomial_moment_features,
)
from sfincs_jax.problems.profile_residual import (
    apply_device_subspace_residual_equation_correction as _apply_device_subspace_residual_equation_correction,
    apply_preconditioned_minres_correction as _apply_preconditioned_minres_correction,
    apply_subspace_minres_correction as _apply_subspace_minres_correction,
    build_rhs1_xblock_post_coarse_directions as _rhs1_xblock_post_coarse_directions,
    compose_multilevel_minres_correction_preconditioner as _compose_multilevel_minres_correction_preconditioner,
    compose_multilevel_residual_correction_preconditioner as _compose_multilevel_residual_correction_preconditioner,
    compose_residual_correction_preconditioner as _compose_residual_correction_preconditioner,
    l2_norm_float as rhs1_l2_norm_float, recompute_true_residual_result as rhs1_recompute_true_residual_result,
    RHS1FPPostSolvePolishContext, run_rhs1_fp_post_solve_polish,
    replay_left_preconditioned_residual_norms as rhs1_replay_left_preconditioned_residual_norms,
    residual_converged as rhs1_residual_converged, residual_target as rhs1_residual_target,
    result_with_true_residual as rhs1_result_with_true_residual, safe_preconditioner as _safe_preconditioner,
    safe_ratio as rhs1_safe_ratio, true_residual_norm_or_inf as rhs1_true_residual_norm_or_inf,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_constraint0_dense_fallback_allowed as _rhs1_constraint0_dense_fallback_allowed_impl,
    rhs1_constraint0_petsc_compat as _rhs1_constraint0_petsc_compat_impl,
    rhs1_constraint0_petsc_compat_config_from_env, rhs1_constraint0_petsc_compat_regularization,
    rhsmode1_constraint0_sparse_first_current_backend as _rhsmode1_constraint0_sparse_first,
)
from sfincs_jax.operators.profile_sources import (
    build_rhs1_xblock_constraint1_moment_schur_preconditioner as _build_rhs1_xblock_constraint1_moment_schur_preconditioner,
    constraint_scheme1_inject_source as _constraint_scheme1_inject_source,
    constraint_scheme1_moments_from_f as _constraint_scheme1_moments_from_f,
    constraint_scheme2_inject_source as _constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f as _constraint_scheme2_source_from_f,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_prefer_sparse_over_dense_shortcut as _rhs1_prefer_sparse_over_dense_shortcut_impl,
    rhs1_sparse_prefer_skips_stage2 as _rhs1_sparse_prefer_skips_stage2_impl,
    rhsmode1_sparse_exact_lu_requested_current_backend as _rhsmode1_sparse_exact_lu_requested,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_large_cpu_sparse_exact_lu_allowed as _rhs1_large_cpu_sparse_exact_lu_allowed_impl,
    rhs1_large_cpu_sparse_rescue_first as _rhs1_large_cpu_sparse_rescue_first_impl,
    rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend as _rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed,
    rhsmode1_large_cpu_sparse_rescue_allowed_current_backend as _rhsmode1_large_cpu_sparse_rescue_allowed,
    rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend as _rhsmode1_large_cpu_sparse_skip_primary_allowed,
    rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend as _rhsmode1_large_cpu_xblock_skip_primary_allowed,
    rhsmode1_sparse_sxblock_rescue_allowed_current_backend as _rhsmode1_sparse_sxblock_rescue_allowed,
    rhsmode1_sparse_xblock_rescue_allowed_current_backend as _rhsmode1_sparse_xblock_rescue_allowed,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_bicgstab_fallback_controls_from_env, rhs1_bicgstab_fallback_decision, rhs1_bicgstab_fallback_target_from_env,
    rhs1_fast_post_xblock_polish_controls_from_env, rhs1_fp_bicgstab_polish_controls_from_env,
    rhs1_fp_global_low_l_polish_controls_from_env, rhs1_fp_l1_polish_controls_from_env,
    rhs1_fp_low_l_polish_controls_from_env, rhs1_fp_residual_polish_controls_from_env,
    rhs1_gmres_precondition_side_from_env, rhs1_krylov_routing_controls_from_env,
    rhs1_pas_source_zero_tolerance_from_env,
    rhsmode1_fast_post_xblock_polish_allowed_current_backend as _rhsmode1_fast_post_xblock_polish_allowed,
    rhsmode1_fp_targeted_polish_allowed_current_backend as _rhsmode1_fp_targeted_polish_allowed,
    rhsmode1_fp_xblock_global_correction_allowed_current_backend as _rhsmode1_fp_xblock_global_correction_allowed,
    rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend as _rhsmode1_skip_global_sparse_after_xblock_allowed,
)
from sfincs_jax.problems.profile_policies import (
    rhsmode1_pas_fast_accept_current_backend as _rhsmode1_pas_fast_accept,
)
from sfincs_jax.problems.profile_policies import (
    rhs1_pas_tz_guarded_stage2_retry, rhs1_stage2_admission_controls_from_env, rhs1_stage2_retry_admission_decision,
    rhs1_stage2_retry_controls_from_env, rhs1_stage2_trigger, rhs1_stage2_trigger_decision,
)
from sfincs_jax.solvers import path_policy as _solver_path_policy
from sfincs_jax.problems.profile_policies import (
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
from sfincs_jax.solvers.explicit_sparse import (
    host_direct_solve_with_refinement as _host_direct_solve_with_refinement_impl,
    host_sparse_direct_solve_with_refinement as _host_sparse_direct_solve_with_refinement_impl,
)
from sfincs_jax.problems.transport_policies import (
    TransportPreconditionerContext, TransportPreconditionerDispatchBuilders, TransportRuntimePolicy,
    TransportStrongPreconditionerCache, build_transport_preconditioner_from_kind,
    normalize_transport_preconditioner_kind, resolve_transport_precondition_side_for_kind,
    resolve_transport_preconditioner_choice, resolve_transport_per_rhs_loop_policy, transport_candidate_is_better,
    transport_dd_config_from_env, transport_geometry_scheme_from_namelist,
    transport_host_gmres_accepts_preconditioned_residual as _transport_host_gmres_accepts_preconditioned_residual_impl,
    transport_polish_config_from_env, transport_precondition_side as _transport_precondition_side_impl,
    transport_residual_value, transport_result_needs_retry, transport_sparse_jax_config_from_env,
    transport_sparse_direct_needs_float64_retry as _transport_sparse_direct_needs_float64_retry_impl,
    transport_sparse_direct_rescue_first as _transport_sparse_direct_rescue_first_impl,
    transport_tzfft_first_attempt_budget as _transport_tzfft_first_attempt_budget_impl,
)
from sfincs_jax.problems.transport_linear_system import (
    build_transport_fp_direct_active_block_schur_preconditioner,
)
from sfincs_jax.problems.transport_linear_system import (
    build_transport_fp_fortran_reduced_lu_preconditioner,
)
from sfincs_jax.problems.transport_setup import (
    resolve_transport_maxiter_setup, resolve_transport_parallel_request, resolve_transport_state_setup,
    resolve_transport_which_rhs_setup,
)
from sfincs_jax.problems.transport_linear_system import (
    resolve_transport_active_dense_setup, transport_active_dof_indices as _transport_active_dof_indices,
)
from sfincs_jax.problems.transport_finalize import (
    TransportConstraintNullspaceProjector, TransportRHSFinalizationContext, finalize_full_transport_rhs,
    finalize_reduced_transport_rhs,
)
from sfincs_jax.problems.transport_parallel_runtime import (
    transport_parallel_backend as _transport_parallel_backend,
    transport_parallel_gpu_worker_env as _transport_parallel_gpu_worker_env,
    transport_parallel_persistent_pool_enabled as _transport_parallel_persistent_pool_enabled,
    transport_parallel_start_method as _transport_parallel_start_method,
    transport_parallel_visible_gpu_ids as _transport_parallel_visible_gpu_ids,
)
from sfincs_jax.problems.transport_parallel_runtime import (
    TransportParallelSolveRuntime, get_transport_parallel_pool as _get_transport_parallel_pool,
    maybe_run_transport_parallel_solve,
    run_transport_parallel_gpu_subprocesses_with_policy as _run_transport_parallel_gpu_subprocesses,
    shutdown_transport_parallel_pool as _shutdown_transport_parallel_pool,
    solve_transport_parallel_payload as _solve_transport_parallel_payload,
    transport_parallel_pool_executor_kwargs as _transport_parallel_pool_executor_kwargs,
    transport_parallel_pool_key as _transport_parallel_pool_key,
    transport_parallel_process_pool_executor as _transport_parallel_process_pool_executor,
    transport_parallel_worker_env as _transport_parallel_worker_env,
)
from sfincs_jax.problems.profile_policies import (
    resolve_use_implicit as _resolve_use_implicit_impl,
)
from sfincs_jax.problems.profile_phi1_newton import (
    phi1_frozen_jacobian_policy, phi1_gmres_restart, phi1_line_search_policy, phi1_use_active_dof_mode,
)
from sfincs_jax.problems.profile_phi1_newton import (
    build_phi1_newton_preconditioner, solve_phi1_newton_linear_step,
)
from sfincs_jax.problems.profile_phi1_newton import advance_phi1_newton_iterate
from sfincs_jax.solvers.diagnostics import (
    RHS1ProgressNotes, rhs1_large_progress_enabled,
)
from sfincs_jax.problems.transport_diagnostics import (
    _flux_functions_from_op, transport_matrix_size_from_rhs_mode,
)
from sfincs_jax.problems.transport_finalize import (
    compute_transport_postsolve_diagnostics,
)
from sfincs_jax.outputs.transport import TransportStreamingOutputAccumulator
from sfincs_jax.solver import (
    block_gmres_result_ready as _block_gmres_result_ready, gmres_result_is_finite as _gmres_result_is_finite,
)
from sfincs_jax.solvers.preconditioning import (  # noqa: F401
    block_diagonal_only as _block_diag_only,
    diagonal_only as _diag_only,
)
from sfincs_jax.solvers.preconditioning import (
    _build_rhsmode1_preconditioner_operator_fortran_reduced, _build_rhsmode1_preconditioner_operator_point,
    _build_rhsmode1_preconditioner_operator_theta_dd, _build_rhsmode1_preconditioner_operator_theta_line,
    _build_rhsmode1_preconditioner_operator_zeta_dd, _build_rhsmode1_preconditioner_operator_zeta_line,
    _build_transport_preconditioner_operator_fortran_reduced, _build_transport_preconditioner_operator_point,
)
from sfincs_jax.solvers.explicit_sparse import (
    inverse_permutation as _inverse_permutation, triangular_solve_lower_csr_rows as _triangular_solve_lower_csr_rows,
    triangular_solve_lower_padded as _triangular_solve_lower_padded,
    triangular_solve_upper_csr_rows as _triangular_solve_upper_csr_rows,
    triangular_solve_upper_padded as _triangular_solve_upper_padded,
)
from sfincs_jax.solvers.preconditioning import (
    hash_array as _hash_array, precond_chunk_cols as _precond_chunk_cols,
    rhs_mode1_precond_cache_key as _rhs_mode1_precond_cache_key_impl,
    rhs_mode1_structured_fblock_cache_key as _rhs_mode1_structured_fblock_cache_key_impl,
    transport_precond_cache_key as _transport_precond_cache_key_impl,
)
from sfincs_jax.solvers.krylov_dispatch import (
    HOST_SCIPY_KRYLOV_METHODS as _HOST_SCIPY_KRYLOV_METHODS, gmres_solve_dispatch as _gmres_solve_dispatch_impl,
    gmres_solve_with_residual_dispatch as _gmres_solve_with_residual_dispatch_impl,
    host_scipy_krylov_requested as _host_scipy_krylov_requested,
    ksp_iteration_solver_label as _ksp_iteration_solver_label,
    resolve_distributed_gmres_axis as _resolve_distributed_gmres_axis_impl,
    rhs_krylov_method_for_context as _rhs_krylov_method_for_context, solver_kind_for_label as _solver_kind_for_label,
)
from sfincs_jax.solvers.preconditioning import (
    _RHSMODE1_PAS_PRECOND_PROBE_CACHE, _RHSMODE1_PAS_TOKAMAK_THETA_CACHE, _RHSMODE1_PAS_TZ_CACHE,
    _RHSMODE1_PRECOND_CACHE, _RHSMODE1_PRECOND_GLOBAL_CACHE, _RHSMODE1_PRECOND_IDX_CACHE, _RHSMODE1_PRECOND_ILU_CACHE,
    _RHSMODE1_PRECOND_LIST_CACHE, _RHSMODE1_SCHUR_CACHE, _RHSMODE1_SPARSE_ILU_CACHE,
    _RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE, _RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE, _RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE, _RHSMODE1_THETA_LINE_DIAGX_CACHE, _RHSMODE1_XMG_PRECOND_CACHE,
    _RHSMODE1_XUPWIND_PRECOND_CACHE, _RHSMODE23_PRECOND_CACHE, _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE,
    _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE, _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE, _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE, _TRANSPORT_FP_TZFFT_PRECOND_CACHE,
    _TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE, _TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE,
    _TRANSPORT_PRECOND_CACHE, _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE, _TRANSPORT_SXBLOCK_PRECOND_CACHE,
    _TRANSPORT_TZFFT_PRECOND_CACHE, _TRANSPORT_XBLOCK_PRECOND_CACHE, _TRANSPORT_XMG_PRECOND_CACHE,
    _LowRankXBlockPrecondCache, _PasTokamakThetaPrecondCache, _PasTzPrecondCache, _RHSMode1ILUBlockPrecondCache,
    _RHSMode1PrecondCache, _RHSMode1PrecondGlobalCache, _RHSMode1PrecondIdxCache, _RHSMode1PrecondListCache,
    _RHSMode1SparseSXBlockHostPrecondCache, _RHSMode1SparseXBlockCSRPrecondCache,
    _RHSMode1SparseXBlockHostPrecondCache, _RHSMode1SparseXBlockPrecondCache, _RHSMode1ThetaLineDiagXCache,
    _TransportFpDirectActiveBlockSchurPrecondCache, _TransportFpFortranReducedLuPrecondCache,
    _TransportFpLocalGeomLinePrecondCache, _TransportFpStructuredFBlockLuPrecondCache,
    _TransportFpTzFftLinePrecondCache, _TransportFpTzFftLineSchurPrecondCache, _TransportFpTzFftPrecondCache,
    _TransportFpXBlockTzLuPrecondCache, _TransportPrecondCache, _TransportTzFftPrecondCache,
    _TransportXBlockPrecondCache, _TransportXmgPrecondCache, _XUpwindPrecondCache,
)
from sfincs_jax.solvers.preconditioning import (
    auto_pas_geom4_fp32_precond_allowed as _auto_pas_geom4_fp32_precond_allowed, precond_dtype as _precond_dtype,
    precond_policy_hints as _precond_policy_hints, set_precond_policy_hints as _set_precond_policy_hints,
    set_precond_size_hint as _set_precond_size_hint, sparse_structural_tol as _sparse_structural_tol,
    use_solver_jit as _use_solver_jit,
)
from sfincs_jax.solvers.preconditioner_symbolic_host import (
    RHS1FullSystemMatrixFreeOperatorAdapter as _RHS1FullSystemMatrixFreeOperatorAdapter,
    build_sparse_ilu_from_matvec as _build_sparse_ilu_from_matvec,
    factorize_sparse_matrix_csr_host as _factorize_sparse_matrix_csr_host,
)
from sfincs_jax.problems.transport_linear_system import (
    _build_rhsmode23_direct_pmat_physics_coarse_basis, _try_build_rhsmode23_fp_direct_active_operator_bundle,
    _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle,
)
from sfincs_jax.operators.profile_system import (
    _source_basis_constraint_scheme_1, _matvec_shard_axis, sharding_constraints,
)
from sfincs_jax.profiling import Timer
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator, _THRESHOLD_FOR_INCLUSION, _operator_signature_cached, apply_v3_full_system_jacobian,
    apply_v3_full_system_jacobian_jit, apply_v3_full_system_operator, apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist, residual_v3_full_system, rhs_v3_full_system, rhs_v3_full_system_jit,
    with_transport_rhs_settings,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    V3LinearSolveResult, V3NewtonKrylovResult, v3_linear_solve_result_from_payload,
)
from sfincs_jax.problems.transport_finalize import (
    V3TransportMatrixSolveResult,
)
from sfincs_jax.operators.profile_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary, summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern, v3_full_system_conservative_sparsity_pattern_for_indices,
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


_rhsmode1_pas_adaptive_smoother_allowed = _rhs1_pas_adaptive_smoother_allowed_impl


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


def _gmres_solve_dispatch(
    *, distributed_axis: str | None = None, size_hint: int | None = None, **kwargs
):
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


def _gmres_solve_with_residual_dispatch(
    *, distributed_axis: str | None = None, size_hint: int | None = None, **kwargs
):
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


def _transport_precond_cache_key(
    op: V3FullSystemOperator, kind: str
) -> tuple[object, ...]:
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
        enabled=rhs1_large_progress_enabled(
            rhs_mode=int(op.rhs_mode), total_size=int(op.total_size)
        ),
    )
    route_setup = resolve_rhs1_initial_route_setup(
        nml=nml,
        op=op,
        solve_method=str(solve_method),
        xblock_active_dof_env=os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", ""
        ),
        use_implicit=bool(_resolve_use_implicit(differentiable=differentiable)),
        force_krylov=bool(
            _rhs1_bool_env("SFINCS_JAX_RHSMODE1_FORCE_KRYLOV", default=False)
        ),
        sharded_axis=_matvec_shard_axis(op),
        backend=str(jax.default_backend()),
        device_count=int(jax.device_count()),
        structured_auto_allowed=_rhs1_structured_full_csr_auto_allowed_impl,
    )
    method_flags = route_setup.method_flags
    solve_method_kind_requested = method_flags.kind
    sparse_host_like_requested = bool(method_flags.sparse_host_like_requested)
    xblock_active_dof_requested = bool(method_flags.xblock_active_dof_requested)
    structured_full_csr_explicit_requested = bool(
        method_flags.structured_full_csr_explicit_requested
    )
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
            structured_full_csr_explicit_requested=bool(
                structured_full_csr_explicit_requested
            ),
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
    include_electric_field_xi_sparse_pc = bool(
        active_problem_setup.include_electric_field_xi_sparse_pc
    )
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

    def _record_structured_fblock_preconditioner_metadata(precond: object) -> None:
        record_structured_fblock_preconditioner_metadata(
            target=rhsmode1_general_metadata,
            preconditioner=precond,
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
    pas_large_bicgstab_fastpath = bool(
        post_active_solve_policy_setup.pas_large_bicgstab_fastpath
    )
    pas_large_fastpath_min = int(post_active_solve_policy_setup.pas_large_fastpath_min)
    if emit is not None:
        for level, message in post_active_solve_policy_setup.messages:
            emit(int(level), str(message))
    if emit is not None:
        emit(
            1,
            f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}",
        )
        emit(
            1,
            "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)",
        )
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
    sparse_pc_gmres_result = try_run_requested_sparse_pc_gmres_branch(
        RequestedSparsePCGMRESBranchContext({**globals(), **locals()})
    )
    if sparse_pc_gmres_result is not None:
        return sparse_pc_gmres_result
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
    rhs1_route_setup = resolve_rhs1_preconditioner_route_setup(
        RHS1PreconditionerRouteSetupContext({**globals(), **locals()})
    )
    er_abs = rhs1_route_setup["er_abs"]
    max_l = rhs1_route_setup["max_l"]
    nxi_for_x = rhs1_route_setup["nxi_for_x"]
    pas_auto_strong_ratio = rhs1_route_setup["pas_auto_strong_ratio"]
    pre_theta = rhs1_route_setup["pre_theta"]
    pre_zeta = rhs1_route_setup["pre_zeta"]
    restart = int(rhs1_route_setup["restart"])
    maxiter = rhs1_route_setup["maxiter"]
    rhs1_bicgstab_env = rhs1_route_setup["rhs1_bicgstab_env"]
    rhs1_bicgstab_env_user = rhs1_route_setup["rhs1_bicgstab_env_user"]
    rhs1_dd_setup = rhs1_route_setup["rhs1_dd_setup"]
    rhs1_gpu_tokamak_pas_tight_gmres = rhs1_route_setup[
        "rhs1_gpu_tokamak_pas_tight_gmres"
    ]
    rhs1_precond_enabled = bool(rhs1_route_setup["rhs1_precond_enabled"])
    rhs1_precond_env = rhs1_route_setup["rhs1_precond_env"]
    rhs1_precond_env_user = rhs1_route_setup["rhs1_precond_env_user"]
    rhs1_precond_kind = rhs1_route_setup["rhs1_precond_kind"]
    rhs1_precond_kind_requested = rhs1_route_setup["rhs1_precond_kind_requested"]
    rhs1_xblock_tz_lmax = rhs1_route_setup["rhs1_xblock_tz_lmax"]
    schur_er_min = rhs1_route_setup["schur_er_min"]
    structured_fblock_precond_requested = bool(
        rhs1_route_setup["structured_fblock_precond_requested"]
    )
    tokamak_like = bool(rhs1_route_setup["tokamak_like"])
    tol = float(rhs1_route_setup["tol"])
    if "lmax_use" in rhs1_route_setup:
        lmax_use = rhs1_route_setup["lmax_use"]
    if "use_collision_precond" in rhs1_route_setup:
        use_collision_precond = rhs1_route_setup["use_collision_precond"]
    if "xblock_tz_max" in rhs1_route_setup:
        xblock_tz_max = rhs1_route_setup["xblock_tz_max"]
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

    linear_solve_dispatch = build_profile_linear_solve_dispatch(
        rhs_mode=int(op.rhs_mode),
        total_size=int(op.total_size),
        use_implicit=bool(use_implicit),
        use_solver_jit=bool(_use_solver_jit()),
        distributed_axis=distributed_axis,
        distributed_auto_solver=distributed_auto_solver,
        small_gmres_max=rhs1_small_gmres_max_from_env(),
    )

    stage2_admission = rhs1_stage2_admission_controls_from_env(
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        solver_kind_default=linear_solve_dispatch.solver_kind(solve_method)[0],
        pas_large_bicgstab_fastpath=bool(pas_large_bicgstab_fastpath),
        tokamak_pas=bool(tokamak_pas),
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        total_size=int(op.total_size),
    )
    stage2_enabled = bool(stage2_admission.enabled)
    stage2_time_cap_s = float(stage2_admission.time_cap_s)
    _solver_kind = linear_solve_dispatch.solver_kind
    _solve_linear = linear_solve_dispatch.solve
    _solve_linear_with_residual = linear_solve_dispatch.solve_with_residual

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
        reduced_system_setup = build_rhs1_active_reduced_system_setup(
            RHS1ActiveReducedSystemSetupContext(
                op=op,
                rhs=rhs,
                x0=x0,
                mv=mv,
                active_idx_jnp=active_idx_jnp,
                full_to_active_jnp=full_to_active_jnp,
                active_size=int(active_size),
                use_pas_projection=bool(use_pas_projection),
                recycle_basis=tuple(recycle_basis_use),
                tol=float(tol),
                atol=float(atol),
            )
        )
        reduce_full = reduced_system_setup.reduce_full
        expand_reduced = reduced_system_setup.expand_reduced
        mv_reduced = reduced_system_setup.mv_reduced
        _wrap_pas_precond = reduced_system_setup.wrap_pas_preconditioner
        rhs_reduced = reduced_system_setup.rhs_reduced
        x0_reduced = reduced_system_setup.x0_reduced
        target_reduced = float(reduced_system_setup.target_reduced)
        target_stage2 = float(reduced_system_setup.target_stage2)
        res_reduced: GMRESSolveResult | None = None
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
                sparse_label = (
                    "CPU" if backend_name == "cpu" else f"{backend_name} host-sparse"
                )
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: "
                    f"{sparse_label} sparse-LU shortcut -> skip primary preconditioner build",
                )

        if rhs1_bicgstab_kind is not None:
            if emit is not None:
                emit(
                    1,
                    f"solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner={rhs1_bicgstab_kind}",
                )
            if rhs1_bicgstab_kind == "collision":
                bicgstab_preconditioner_reduced = (
                    _build_rhsmode1_collision_preconditioner(
                        op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                )
            if use_pas_projection:
                bicgstab_preconditioner_reduced = _wrap_pas_precond(
                    bicgstab_preconditioner_reduced
                )

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
            use_collision_precond, skip_message = (
                _rhs1_pas_preconditioner_probe_large_collision_skip(
                    config=pas_probe_config,
                    cached_decision=use_collision_precond,
                    total_size=int(op.total_size),
                    constraint_scheme=int(op.constraint_scheme),
                    extra_size=int(op.extra_size),
                )
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
                    probe_rel = (
                        float(jnp.linalg.norm(probe_r)) / rhs_norm
                        if rhs_norm > 0
                        else 0.0
                    )
                    use_collision_precond = (
                        _rhs1_pas_preconditioner_probe_uses_collision(
                            probe_rel=probe_rel,
                            rel_max=pas_probe_config.rel_max,
                        )
                    )
                    _RHSMODE1_PAS_PRECOND_PROBE_CACHE[probe_key] = bool(
                        use_collision_precond
                    )
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
                        emit(
                            1,
                            f"solve_v3_full_system_linear_gmres: PAS precond probe failed ({type(exc).__name__}: {exc})",
                        )
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
            pas_precond_force_collision = bool(
                precond_build.pas_precond_force_collision
            )
            if precond_build.bicgstab_preconditioner is not None:
                bicgstab_preconditioner_reduced = precond_build.bicgstab_preconditioner
            return precond_build.preconditioner

        if rhs1_precond_enabled and (not host_dense_shortcut):
            solver_kind = _solver_kind(solve_method)[0]
            build_rhs1 = (
                solver_kind != "bicgstab" and solve_method_kind != "dense"
            ) or (rhs1_bicgstab_kind == "rhs1" and solve_method_kind != "dense")
            if build_rhs1 and preconditioner_reduced is None:
                preconditioner_reduced = (
                    _build_rhs1_preconditioner_reduced_with_fallback()
                )
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced
        if (
            (not host_dense_shortcut)
            and preconditioner_reduced is None
            and bicgstab_preconditioner_reduced is not None
        ):
            preconditioner_reduced = bicgstab_preconditioner_reduced
        if (
            (not host_dense_shortcut)
            and preconditioner_reduced is not None
            and rhs1_precond_kind
            in {
                "pas_hybrid",
                "pas_lite",
                "pas_tz",
                "pas_schur",
                "pas_tokamak_theta",
                "pas_ilu",
            }
        ):
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
            and preconditioner_reduced is not None
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
        ):
            fp_l1_env = (
                os.environ.get("SFINCS_JAX_RHSMODE1_FP_L1_HYBRID", "").strip().lower()
            )
            fp_l1_enabled = fp_l1_env in {"1", "true", "yes", "on"}
            fp_l1_min_env = os.environ.get(
                "SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_MIN", ""
            ).strip()
            fp_l1_lmax_env = os.environ.get(
                "SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_LMAX", ""
            ).strip()
            fp_l1_block_env = os.environ.get(
                "SFINCS_JAX_RHSMODE1_FP_L1_HYBRID_BLOCK_MAX", ""
            ).strip()
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
            early_dense_shortcut = bool(
                host_dense_shortcut_outcome.early_dense_shortcut
            )
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
                a_csr_full, _a_csr_drop, _ilu, _a_dense, _l_dense, _u_dense, _l_unit = (
                    _build_sparse_ilu_from_matvec(
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
                )

                def _mv_sparse(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    y_np = a_csr_full @ x_np
                    return jnp.asarray(y_np, dtype=jnp.float64)

                mv_reduced = _mv_sparse
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: using sparse operator matvec",
                    )
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
                    x0_vec=None
                    if x0_reduced is None
                    else jnp.asarray(x0_reduced, dtype=jnp.float64),
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
                sparse_prefer_over_dense_shortcut=bool(
                    sparse_prefer_over_dense_shortcut
                ),
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
                res_reduced, x0_reduced = (
                    rhs1_seed_skip_primary_krylov_and_update_replay(
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
                                residual_norm=jnp.asarray(
                                    residual_norm, dtype=jnp.float64
                                ),
                            ),
                        ),
                    )
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
            has_fp=op.fblock.fp is not None,
            include_phi1=bool(op.include_phi1),
            residual_norm=float(res_reduced.residual_norm),
            has_pas=op.fblock.pas is not None,
            rhs1_precond_kind=rhs1_precond_kind,
            pas_tz_guarded_fallback=bool(rhs1_pas_tz_guarded_fallback),
            pas_tz_guarded_retry=rhs1_pas_tz_guarded_stage2_retry(),
            cpu_large_xblock_shortcut=bool(cpu_large_xblock_shortcut),
            cpu_large_sparse_shortcut=bool(cpu_large_sparse_shortcut),
        )
        stage2_trigger, fp_force_stage2 = (
            stage2_decision.stage2_trigger,
            stage2_decision.fp_force_stage2,
        )
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
            (
                res_reduced,
                residual_vec,
                preconditioner_reduced,
                _accepted,
                _bicgstab_fallback_elapsed_s,
            ) = rhs1_run_bicgstab_gmres_fallback_if_allowed(
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
            (
                res_reduced,
                residual_vec,
                preconditioner_reduced,
                _accepted,
                _stage2_elapsed_s,
            ) = rhs1_run_stage2_retry_if_allowed(
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
            enabled=bool(
                qi_device_seed_setup.early_enabled or qi_device_seed_setup.skip_strong
            ),
        )
        res_reduced = early_qi_attempt.result

        pas_smoother_allowed = (
            rhs1_precond_kind
            in {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
            and preconditioner_reduced is not None
            and _rhsmode1_pas_adaptive_smoother_allowed(
                op=op,
                active_size=int(active_size),
                residual_norm=float(res_reduced.residual_norm),
                target=float(target_reduced),
                use_implicit=bool(use_implicit),
            )
        )
        smoother_controls = (
            rhs1_pas_adaptive_smoother_controls_from_env()
            if pas_smoother_allowed
            else None
        )
        res_reduced, residual_vec, _accepted = (
            rhs1_run_adaptive_smoother_and_update_replay(
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
        (
            res_reduced,
            residual_vec,
            bicgstab_preconditioner_reduced,
            _accepted,
            _collision_elapsed_s,
        ) = rhs1_run_collision_retry_if_allowed(
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
            has_extra_constraint_block=int(op.constraint_scheme) == 2
            and int(op.extra_size) > 0
            and (not use_pas_projection),
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
            sparse_label = (
                "large CPU" if backend_name == "cpu" else f"{backend_name} host-sparse"
            )
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
            has_extra_constraint_block=int(op.constraint_scheme) == 2
            and int(op.extra_size) > 0,
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
            sparse_exact_lu = _rhsmode1_large_cpu_sparse_exact_lu_allowed(
                active_size=int(active_size)
            )
            backend_name = str(jax.default_backend()).strip().lower()
            large_cpu_sparse_label = (
                "large CPU sparse"
                if backend_name == "cpu"
                else f"{backend_name} host-sparse"
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
        sparse_xblock_rescue_reason = (
            "not_needed"
            if float(res_reduced.residual_norm) <= target_reduced
            else "inactive"
        )
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
                and not bool(
                    rhsmode1_general_metadata.get(
                        "xblock_qi_device_preconditioner_built", False
                    )
                )
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
                    sparse_xblock_preconditioner_xi = int(
                        sparse_xblock_build.preconditioner_xi
                    )
                    assembled_host_fp = bool(
                        sparse_xblock_build.force_assembled_host_fp
                    )
                    precond_sparse_xblock_current = precond_sparse_xblock
                    sparse_xblock_rescue_built = True
                    sparse_xblock_rescue_assembled_host_fp = bool(assembled_host_fp)
                    sparse_xblock_rescue_preconditioner_xi = int(
                        sparse_xblock_preconditioner_xi
                    )
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
                        explicit_fp_xblock_seed_residual = float(
                            sparse_xblock_solve.seed_residual
                        )
                        sparse_xblock_rescue_seed_residual = float(
                            sparse_xblock_solve.seed_residual
                        )
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
            fp_xblock_global_correction_allowed = (
                _rhsmode1_fp_xblock_global_correction_allowed(
                    op=op,
                    active_size=int(active_size),
                    residual_norm=float(res_reduced.residual_norm),
                    target=float(target_reduced),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    sparse_xblock_candidate_accepted=bool(
                        sparse_xblock_rescue_candidate_accepted
                    ),
                    use_implicit=bool(use_implicit),
                )
            )
            if fp_xblock_global_correction_allowed:
                fp_xblock_global_correction_attempted = True
                fp_xblock_global_correction_reason = "started"
                correction_precond = (
                    precond_sparse_xblock_current or preconditioner_reduced
                )
                fp_xblock_global_correction_preconditioner = (
                    "sparse_xblock"
                    if precond_sparse_xblock_current is not None
                    else "base"
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
                correction_env = (
                    os.environ.get(
                        "SFINCS_JAX_RHSMODE1_FP_XBLOCK_GLOBAL_CORRECTION", ""
                    )
                    .strip()
                    .lower()
                )
                fp_xblock_global_correction_reason = (
                    "disabled"
                    if correction_env not in {"1", "true", "yes", "on"}
                    else "policy_guard"
                )
            highx_env = (
                os.environ.get(
                    "SFINCS_JAX_RHSMODE1_FP_XBLOCK_HIGHX_RESIDUAL_CORRECTION", ""
                )
                .strip()
                .lower()
            )
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
                and (
                    int(highx_active_max) <= 0
                    or int(active_size) <= int(highx_active_max)
                )
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
            skip_global_sparse_after_xblock = (
                _rhsmode1_skip_global_sparse_after_xblock_allowed(
                    op=op,
                    active_size=int(active_size),
                    residual_norm=float(res_reduced.residual_norm),
                    target=float(target_reduced),
                    used_large_cpu_xblock_shortcut=bool(cpu_large_xblock_shortcut),
                    used_explicit_fp_xblock_seed=bool(explicit_fp_xblock_seed_used),
                    use_implicit=bool(use_implicit),
                )
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
                    xblock_seed_improvement_ratio=float(
                        explicit_fp_xblock_seed_improvement_ratio
                    ),
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
            if float(res_reduced.residual_norm) > target_reduced and (
                not skip_global_sparse_after_xblock
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
        res_reduced, residual_vec, _accepted = (
            run_rhs1_reduced_dense_fallback_admission_stage(
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
                    host_sparse_skip_ratio=float(
                        _rhsmode1_host_sparse_skip_dense_ratio()
                    ),
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
        res_reduced = run_rhs1_fp_post_solve_polish(
            RHS1FPPostSolvePolishContext(
                op=op,
                result=res_reduced,
                rhs=rhs_reduced,
                matvec=mv_reduced,
                preconditioner=preconditioner_reduced,
                active_size=int(active_size),
                target=float(target_reduced),
                tol=float(tol),
                atol=float(atol),
                restart=int(restart),
                maxiter=maxiter,
                precondition_side=gmres_precond_side,
                rhs1_precond_kind=rhs1_precond_kind,
                use_implicit=bool(use_implicit),
                use_active_dof_mode=bool(use_active_dof_mode),
                full_to_active=full_to_active_jnp,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
                read_residual_controls=rhs1_fp_residual_polish_controls_from_env,
                read_low_l_controls=rhs1_fp_low_l_polish_controls_from_env,
                read_l1_controls=rhs1_fp_l1_polish_controls_from_env,
                read_global_low_l_controls=rhs1_fp_global_low_l_polish_controls_from_env,
                read_bicgstab_controls=rhs1_fp_bicgstab_polish_controls_from_env,
                targeted_polish_allowed=_rhsmode1_fp_targeted_polish_allowed,
                build_collision_preconditioner=_build_rhsmode1_collision_preconditioner,
                build_lmax_preconditioner=_build_rhsmode1_xblock_tz_lmax_preconditioner,
                pitch_mode_active_indices=fp_pitch_mode_active_indices,
                solve_linear=_solve_linear,
                emit=emit,
            )
        )
        if not bool(sparse_enabled):
            sparse_xblock_rescue_reason = "sparse_disabled"
        elif float(res_reduced.residual_norm) <= float(target_reduced) and not bool(
            sparse_xblock_rescue_attempted
        ):
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
                    skip_global_sparse_after_xblock=bool(
                        skip_global_sparse_after_xblock
                    ),
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
                extra = (
                    _constraint_scheme2_source_from_f(
                        op, r_f.reshape(op.fblock.f_shape)
                    )
                    / fs_sum_safe
                )
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
                host_dense_shortcut_full_outcome = (
                    run_rhs1_full_host_dense_shortcut_stage(
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
                            if jnp.isfinite(r1) and (
                                not jnp.isfinite(r0) or float(r1) < float(r0)
                            ):
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
                    allowed=(not host_dense_shortcut_full)
                    and preconditioner_full is not None,
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
                or (
                    bicgstab_fallback_strict
                    and float(result.residual_norm) > bicgstab_fallback_target
                )
            ):
                (
                    result,
                    residual_vec,
                    preconditioner_full,
                    _accepted,
                    _bicgstab_fallback_elapsed_s,
                ) = rhs1_run_bicgstab_gmres_fallback_if_allowed(
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
        # The full-size RHSMode=1 branch does not have the later active-DOF sparse
        # ILU rescue. On accelerators, skipping stage2 GMRES here can therefore
        # return a high-residual solution with no real recovery path.
        prefer_sparse_accel = False
        if (
            prefer_sparse_accel
            and float(result.residual_norm) > target
            and stage2_trigger
            and emit is not None
        ):
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
            rhs1_precond_kind
            in {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
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
        (
            result,
            residual_vec,
            bicgstab_preconditioner_full,
            _accepted,
            _collision_elapsed_s,
        ) = rhs1_run_collision_retry_if_allowed(
            allowed=bool(collision_retry_allowed),
            replay_state=ksp_replay,
            current_result=result,
            current_residual_vec=residual_vec,
            matvec_fn=mv,
            b_vec=rhs,
            precond_fn=bicgstab_preconditioner_full,
            build_preconditioner=lambda: _build_rhsmode1_collision_preconditioner(
                op=op
            ),
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
            has_extra_constraint_block=int(op.constraint_scheme) == 2
            and int(op.extra_size) > 0,
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
                has_extra_constraint_block=int(op.constraint_scheme) == 2
                and int(op.extra_size) > 0,
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
                build_preconditioner=lambda: _build_rhsmode1_schur_preconditioner(
                    op=op
                ),
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
    static_argnames=(
        "tol",
        "atol",
        "restart",
        "maxiter",
        "solve_method",
        "identity_shift",
    ),
)
