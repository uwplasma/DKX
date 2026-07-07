"""Sparse-PC solve orchestration for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .profile_sparse_direct import (
    DirectTailCoupledCoarseRescuePolicy,
    DirectTailMaterializationContext,
    DirectTailMaterializationResult,
    DirectTailResidualRescuePolicy,
    DirectTailStructuredAdmissionContext,
    DirectTailStructuredAdmissionResult,
    DirectTailStructuredBuildContext,
    DirectTailSupportModePreflightContext,
    DirectTailTrueActiveRescuePolicy,
    SparseHostOrILUFactorBuildContext,
    SparseHostRetryCandidateContext,
    SparseJAXRetryPreconditionerBuildContext,
    build_direct_tail_materialization_setup,
    build_direct_tail_structured_preconditioner_setup,
    build_sparse_host_or_ilu_factor,
    build_sparse_jax_retry_preconditioner,
    resolve_direct_tail_coupled_coarse_rescue_policy,
    resolve_direct_tail_residual_rescue_policy,
    resolve_direct_tail_structured_admission,
    resolve_direct_tail_true_active_rescue_policy,
    resolve_sparse_host_or_ilu_factor_controls,
    run_direct_tail_support_mode_preflight,
    run_sparse_host_retry_candidate,
)
from .profile_sparse_fortran_reduced import (
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
    fortran_reduced_xblock_final_payload,
    prepare_fortran_reduced_xblock_initial_guess,
    resolve_fortran_reduced_sparse_pc_backend,
    resolve_fortran_reduced_xblock_global_coupling_policy,
    resolve_fortran_reduced_xblock_initial_seed_policy,
    resolve_fortran_reduced_xblock_moment_schur_policy,
    run_fortran_reduced_xblock_krylov_solve,
)
from .profile_sparse_policy import (
    SparsePCAutoPreflightRetryEvaluationContext,
    SparsePCAutoPreflightRetrySelectionContext,
    SparsePCFactorPolicySetup,
    SparsePCFactorPreflightEvaluationContext,
    SparsePCFactorPreflightPolicy,
    SparsePCFactorPreflightPolicyContext,
    SparsePCMemoryBudgetPreflightContext,
    SparsePCPatternSetupContext,
    SparsePCResidualCandidateAcceptanceContext,
    _env_bool,
    _env_int,
    _env_value,
    build_sparse_pc_active_dof_setup,
    build_sparse_pc_pattern_setup,
    enforce_sparse_pc_memory_budget,
    evaluate_sparse_pc_auto_preflight_retry,
    evaluate_sparse_pc_factor_preflight,
    evaluate_sparse_pc_residual_candidate_acceptance,
    resolve_sparse_pc_factor_policy,
    resolve_sparse_pc_factor_preflight_policy,
    select_sparse_pc_auto_preflight_retry_candidates,
)
from .profile_sparse_xblock import (
    XBlockSparsePCSetup as XBlockSparsePCSetup,
    XBlockSparsePCSidePolicySetup as XBlockSparsePCSidePolicySetup,
    XBlockAssembledEquilibrationSetup as XBlockAssembledEquilibrationSetup,
    XBlockAssembledPreflightMemoryError as XBlockAssembledPreflightMemoryError,
    XBlockAssembledPreflightError as XBlockAssembledPreflightError,
    XBlockAssembledOperatorPreflightSetup as XBlockAssembledOperatorPreflightSetup,
    XBlockAssembledDeviceSetup as XBlockAssembledDeviceSetup,
    XBlockAssembledMatvecSetup as XBlockAssembledMatvecSetup,
    XBlockMomentSchurProbeResult as XBlockMomentSchurProbeResult,
    XBlockTwoLevelPolicySetup as XBlockTwoLevelPolicySetup,
    XBlockSeedPolicySetup as XBlockSeedPolicySetup,
    XBlockSparsePCBranchContext as XBlockSparsePCBranchContext,
    resolve_xblock_sparse_pc_setup as resolve_xblock_sparse_pc_setup,
    resolve_xblock_sparse_pc_side_policy_setup as resolve_xblock_sparse_pc_side_policy_setup,
    build_xblock_assembled_equilibration_setup as build_xblock_assembled_equilibration_setup,
    build_xblock_assembled_operator_preflight_setup as build_xblock_assembled_operator_preflight_setup,
    build_xblock_assembled_device_setup as build_xblock_assembled_device_setup,
    build_xblock_assembled_matvec_setup as build_xblock_assembled_matvec_setup,
    finalize_xblock_assembled_operator_metadata as finalize_xblock_assembled_operator_metadata,
    resolve_xblock_moment_schur_policy_setup as resolve_xblock_moment_schur_policy_setup,
    evaluate_xblock_moment_schur_probe_result as evaluate_xblock_moment_schur_probe_result,
    finalize_xblock_moment_schur_metadata as finalize_xblock_moment_schur_metadata,
    failed_xblock_moment_schur_metadata as failed_xblock_moment_schur_metadata,
    resolve_xblock_two_level_policy_setup as resolve_xblock_two_level_policy_setup,
    finalize_xblock_two_level_metadata as finalize_xblock_two_level_metadata,
    failed_xblock_two_level_metadata as failed_xblock_two_level_metadata,
    resolve_xblock_global_coupling_policy_setup as resolve_xblock_global_coupling_policy_setup,
    finalize_xblock_global_coupling_metadata as finalize_xblock_global_coupling_metadata,
    failed_xblock_global_coupling_metadata as failed_xblock_global_coupling_metadata,
    resolve_xblock_seed_policy_setup as resolve_xblock_seed_policy_setup,
    XBlockGlobalCouplingPolicySetup as XBlockGlobalCouplingPolicySetup,
    XBlockMomentSchurPolicySetup as XBlockMomentSchurPolicySetup,
)

# Consolidated sparse-PC Krylov execution helpers

ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class RequestedSparsePCGMRESBranchContext:
    """Driver scope for the explicit RHSMode-1 sparse-PC GMRES branch."""

    values: Mapping[str, Any]


def try_run_requested_sparse_pc_gmres_branch(
    context: RequestedSparsePCGMRESBranchContext,
) -> Any | None:
    """Run the requested sparse-PC GMRES branch, if the solve method selects it."""

    FortranReducedXBlockBackendContext = context.values['FortranReducedXBlockBackendContext']
    RHS1BlockLayout = context.values['RHS1BlockLayout']
    SparsePCAutoPreflightRetryStageContext = context.values['SparsePCAutoPreflightRetryStageContext']
    SparsePCDirectTailFactorSetupContext = context.values['SparsePCDirectTailFactorSetupContext']
    SparsePCDirectTailRescuePolicySetupContext = context.values['SparsePCDirectTailRescuePolicySetupContext']
    SparsePCFactorPreflightRunContext = context.values['SparsePCFactorPreflightRunContext']
    SparsePCGMRESContext = context.values['SparsePCGMRESContext']
    SparsePCGenericBranchSetupContext = context.values['SparsePCGenericBranchSetupContext']
    SparsePCResidualCorrectionStageContext = context.values['SparsePCResidualCorrectionStageContext']
    SparsePCTrueCoupledCoarseStageContext = context.values['SparsePCTrueCoupledCoarseStageContext']
    Timer = context.values['Timer']
    _DIRECT_TAIL_STRUCTURED_PC_CACHE = context.values['_DIRECT_TAIL_STRUCTURED_PC_CACHE']
    _ReusableTrueActionColumnCache = context.values['_ReusableTrueActionColumnCache']
    _SPARSE_HOST_PC_GMRES_SOLVE_METHODS = context.values['_SPARSE_HOST_PC_GMRES_SOLVE_METHODS']
    _SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS = context.values['_SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS']
    _StructuredHostSparsePreconditionerBundle = context.values['_StructuredHostSparsePreconditionerBundle']
    _apply_device_subspace_residual_equation_correction = context.values['_apply_device_subspace_residual_equation_correction']
    _apply_preconditioned_minres_correction = context.values['_apply_preconditioned_minres_correction']
    _apply_subspace_minres_correction = context.values['_apply_subspace_minres_correction']
    _build_host_sparse_direct_factor_from_matvec = context.values['_build_host_sparse_direct_factor_from_matvec']
    _build_rhs1_xblock_constraint1_moment_schur_preconditioner = context.values['_build_rhs1_xblock_constraint1_moment_schur_preconditioner']
    _build_rhsmode1_preconditioner_operator_fortran_reduced = context.values['_build_rhsmode1_preconditioner_operator_fortran_reduced']
    _build_rhsmode1_preconditioner_operator_point = context.values['_build_rhsmode1_preconditioner_operator_point']
    _build_rhsmode1_xblock_tz_sparse_preconditioner = context.values['_build_rhsmode1_xblock_tz_sparse_preconditioner']
    _direct_tail_structured_pc_cache_key = context.values['_direct_tail_structured_pc_cache_key']
    _direct_tail_structured_pc_with_cache_metadata = context.values['_direct_tail_structured_pc_with_cache_metadata']
    _host_sparse_factor_dtype = context.values['_host_sparse_factor_dtype']
    _is_direct_reduced_pmat_pc_kind = context.values['_is_direct_reduced_pmat_pc_kind']
    _parse_true_operator_window_specs = context.values['_parse_true_operator_window_specs']
    _read_rhs1_post_solve_correction_policy = context.values['_read_rhs1_post_solve_correction_policy']
    _read_rhs1_probe_coarse_policy = context.values['_read_rhs1_probe_coarse_policy']
    _rhs1_active_reduced_residual_diagnostics = context.values['_rhs1_active_reduced_residual_diagnostics']
    _rhs1_additive_rescue_nbytes = context.values['_rhs1_additive_rescue_nbytes']
    _rhs1_bool_env = context.values['_rhs1_bool_env']
    _rhs1_float_env = context.values['_rhs1_float_env']
    _rhs1_xblock_fallback_initial_guess = context.values['_rhs1_xblock_fallback_initial_guess']
    _rhs1_xblock_policy = context.values['_rhs1_xblock_policy']
    _rhs1_xblock_post_coarse_directions = context.values['_rhs1_xblock_post_coarse_directions']
    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb = context.values['_rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb']
    _rhsmode1_fp_xblock_assembled_host_allowed = context.values['_rhsmode1_fp_xblock_assembled_host_allowed']
    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly = context.values['_rhsmode1_fp_xblock_species_decoupled_for_host_assembly']
    _rhsmode1_sparse_pc_default_permc_spec = context.values['_rhsmode1_sparse_pc_default_permc_spec']
    _rhsmode1_sparse_pc_default_restart = context.values['_rhsmode1_sparse_pc_default_restart']
    _transport_active_dof_indices = context.values['_transport_active_dof_indices']
    _try_build_fortran_reduced_constraint1_direct_tail_bundle = context.values['_try_build_fortran_reduced_constraint1_direct_tail_bundle']
    _try_build_residual_coarse_host_sparse_preconditioner = context.values['_try_build_residual_coarse_host_sparse_preconditioner']
    _try_build_residual_window_host_sparse_preconditioner = context.values['_try_build_residual_window_host_sparse_preconditioner']
    _try_build_structured_rhs1_full_csr_operator_bundle = context.values['_try_build_structured_rhs1_full_csr_operator_bundle']
    _try_build_true_operator_active_block_lsq_preconditioner = context.values['_try_build_true_operator_active_block_lsq_preconditioner']
    _try_build_true_operator_active_residual_block_lsq_preconditioner = context.values['_try_build_true_operator_active_residual_block_lsq_preconditioner']
    _try_build_true_operator_active_submatrix_preconditioner = context.values['_try_build_true_operator_active_submatrix_preconditioner']
    _try_build_true_operator_coupled_coarse_lsq_preconditioner = context.values['_try_build_true_operator_coupled_coarse_lsq_preconditioner']
    _try_build_true_operator_residual_window_lsq_preconditioner = context.values['_try_build_true_operator_residual_window_lsq_preconditioner']
    active_idx_jnp = context.values['active_idx_jnp']
    active_size = context.values['active_size']
    apply_v3_full_system_operator_cached = context.values['apply_v3_full_system_operator_cached']
    atol = context.values['atol']
    bicgstab_solve_with_history_scipy = context.values['bicgstab_solve_with_history_scipy']
    bicgstab_solve_with_residual = context.values['bicgstab_solve_with_residual']
    build_active_projected_rhs1_full_csr_preconditioner = context.values['build_active_projected_rhs1_full_csr_preconditioner']
    build_direct_active_fortran_v3_reduced_pmat_preconditioner = context.values['build_direct_active_fortran_v3_reduced_pmat_preconditioner']
    build_operator_from_pattern = context.values['build_operator_from_pattern']
    build_sparse_pc_direct_tail_factor_setup = context.values['build_sparse_pc_direct_tail_factor_setup']
    build_sparse_pc_direct_tail_rescue_policy_setup = context.values['build_sparse_pc_direct_tail_rescue_policy_setup']
    build_sparse_pc_generic_branch_setup = context.values['build_sparse_pc_generic_branch_setup']
    device_csr_from_matrix = context.values['device_csr_from_matrix']
    differentiable = context.values['differentiable']
    emit = context.values['emit']
    er_abs_sparse_pc = context.values['er_abs_sparse_pc']
    estimate_sparse_pc_memory = context.values['estimate_sparse_pc_memory']
    estimate_v3_full_system_conservative_sparsity_summary = context.values['estimate_v3_full_system_conservative_sparsity_summary']
    expand_reduced_with_map = context.values['expand_reduced_with_map']
    explicit_left_preconditioned_gmres_scipy = context.values['explicit_left_preconditioned_gmres_scipy']
    fgmres_cycle_jit_solve_with_residual = context.values['fgmres_cycle_jit_solve_with_residual']
    fgmres_solve_with_residual = context.values['fgmres_solve_with_residual']
    fgmres_solve_with_residual_jit = context.values['fgmres_solve_with_residual_jit']
    finalize_sparse_pc_gmres_bundle = context.values['finalize_sparse_pc_gmres_bundle']
    full_to_active_jnp = context.values['full_to_active_jnp']
    gcrotmk_solve_with_history_scipy = context.values['gcrotmk_solve_with_history_scipy']
    gmres_solve_with_history_scipy = context.values['gmres_solve_with_history_scipy']
    has_reduced_modes = context.values['has_reduced_modes']
    include_electric_field_xi_sparse_pc = context.values['include_electric_field_xi_sparse_pc']
    include_xdot_sparse_pc = context.values['include_xdot_sparse_pc']
    jax = context.values['jax']
    jnp = context.values['jnp']
    lgmres_solve_with_history_scipy = context.values['lgmres_solve_with_history_scipy']
    maxiter = context.values['maxiter']
    np = context.values['np']
    op = context.values['op']
    os = context.values['os']
    preconditioner_species = context.values['preconditioner_species']
    preconditioner_x = context.values['preconditioner_x']
    preconditioner_x_min_l = context.values['preconditioner_x_min_l']
    preconditioner_xi = context.values['preconditioner_xi']
    reduce_full_with_indices = context.values['reduce_full_with_indices']
    resolve_rhs1_xblock_sparse_pc_policy = context.values['resolve_rhs1_xblock_sparse_pc_policy']
    resolve_sparse_pc_entry_policy = context.values['resolve_sparse_pc_entry_policy']
    resolve_sparse_pc_gmres_control_policy = context.values['resolve_sparse_pc_gmres_control_policy']
    restart = context.values['restart']
    rhs = context.values['rhs']
    rhs1_gmres_precondition_side_from_env = context.values['rhs1_gmres_precondition_side_from_env']
    rhs1_l2_norm_float = context.values['rhs1_l2_norm_float']
    rhs1_parse_polish_gmres_config = context.values['rhs1_parse_polish_gmres_config']
    rhs1_residual_target = context.values['rhs1_residual_target']
    rhs1_safe_ratio = context.values['rhs1_safe_ratio']
    rhs_norm = context.values['rhs_norm']
    run_sparse_pc_auto_preflight_retry_stage = context.values['run_sparse_pc_auto_preflight_retry_stage']
    run_sparse_pc_factor_preflight = context.values['run_sparse_pc_factor_preflight']
    run_sparse_pc_gmres_once_for_retry = context.values['run_sparse_pc_gmres_once_for_retry']
    run_sparse_pc_residual_correction_stage = context.values['run_sparse_pc_residual_correction_stage']
    run_sparse_pc_true_coupled_coarse_stage = context.values['run_sparse_pc_true_coupled_coarse_stage']
    run_xblock_sparse_pc_branch = context.values['run_xblock_sparse_pc_branch']
    select_active_fortran_v3_reduced_support_mode_preconditioner = context.values['select_active_fortran_v3_reduced_support_mode_preconditioner']
    solve_fortran_reduced_xblock_backend = context.values['solve_fortran_reduced_xblock_backend']
    solve_method_kind_explicit = context.values['solve_method_kind_explicit']
    sparse_pc_gmres_finalization_bundle_from_solve_result = context.values['sparse_pc_gmres_finalization_bundle_from_solve_result']
    summarize_v3_sparse_pattern = context.values['summarize_v3_sparse_pattern']
    tfqmr_solve_with_residual = context.values['tfqmr_solve_with_residual']
    tol = context.values['tol']
    use_active_dof_mode = context.values['use_active_dof_mode']
    use_dkes = context.values['use_dkes']
    v3_full_system_conservative_sparsity_pattern = context.values['v3_full_system_conservative_sparsity_pattern']
    v3_full_system_conservative_sparsity_pattern_for_indices = context.values['v3_full_system_conservative_sparsity_pattern_for_indices']
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern = context.values['v3_full_system_fortran_reduced_preconditioner_sparsity_pattern']
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices = context.values['v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices']
    v3_linear_solve_result_from_payload = context.values['v3_linear_solve_result_from_payload']
    validate_device_csr_matvec = context.values['validate_device_csr_matvec']
    x0 = context.values['x0']
    xblock_active_dof_requested = context.values['xblock_active_dof_requested']

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

        direct_tail_factor_setup = build_sparse_pc_direct_tail_factor_setup(
            SparsePCDirectTailFactorSetupContext(
                env=os.environ,
                op=op,
                op_pc=op_pc,
                rhs_dtype=rhs.dtype,
                pattern=pattern,
                active_indices=sparse_pc_active_idx_np,
                sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
                reduce_full=_sparse_pc_reduce_full,
                expand_reduced=_sparse_pc_expand_reduced,
                factor_matvec=_sparse_pc_factor_mv,
                pc_shift=float(pc_shift),
                factor_dtype_initial=sparse_pc_factor_dtype_initial,
                factorization=str(sparse_pc_factorization),
                default_factor_kind=str(sparse_pc_default_factor_kind),
                default_ilu_fill_factor=float(sparse_pc_default_ilu_fill_factor),
                default_ilu_drop_tol=float(sparse_pc_default_ilu_drop_tol),
                default_pattern_color_batch=int(sparse_pc_default_pattern_color_batch),
                default_permc_spec=str(sparse_pc_default_permc_spec),
                permc_spec=str(sparse_pc_permc_spec),
                sparse_pc_linear_size=int(sparse_pc_linear_size),
                constrained_pas_pc=bool(constrained_pas_pc),
                tokamak_fp_pc=bool(tokamak_fp_pc),
                fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                preconditioner_x=int(preconditioner_x),
                preconditioner_x_min_l=int(preconditioner_x_min_l),
                preconditioner_xi=int(preconditioner_xi),
                preconditioner_species=int(preconditioner_species),
                sparse_timer=sparse_timer,
                emit=emit,
                default_direct_tail_max_mb=(
                    _rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb
                ),
                is_direct_reduced_pmat_pc_kind=_is_direct_reduced_pmat_pc_kind,
                build_direct_tail_bundle=(
                    _try_build_fortran_reduced_constraint1_direct_tail_bundle
                ),
                build_structured_full_csr_operator_bundle=(
                    _try_build_structured_rhs1_full_csr_operator_bundle
                ),
                layout_from_operator=RHS1BlockLayout.from_operator,
                build_direct_active_preconditioner=(
                    build_direct_active_fortran_v3_reduced_pmat_preconditioner
                ),
                build_active_projected_preconditioner=(
                    build_active_projected_rhs1_full_csr_preconditioner
                ),
                structured_cache=_DIRECT_TAIL_STRUCTURED_PC_CACHE,
                structured_cache_key=_direct_tail_structured_pc_cache_key,
                structured_cache_metadata=(
                    _direct_tail_structured_pc_with_cache_metadata
                ),
                structured_factor_bundle_factory=(
                    _StructuredHostSparsePreconditionerBundle
                ),
                host_factor_builder=_build_host_sparse_direct_factor_from_matvec,
            )
        )
        direct_tail_materialization = direct_tail_factor_setup.materialization
        direct_tail_default = direct_tail_factor_setup.direct_tail_default
        direct_tail_enabled = direct_tail_factor_setup.direct_tail_enabled
        direct_tail_built = direct_tail_factor_setup.direct_tail_built
        direct_tail_error = direct_tail_factor_setup.direct_tail_error
        direct_tail_operator_bundle = direct_tail_factor_setup.direct_tail_operator_bundle
        direct_tail_structured_pc_requested = (
            direct_tail_factor_setup.direct_tail_structured_pc_requested
        )
        direct_tail_structured_pc_selected = (
            direct_tail_factor_setup.direct_tail_structured_pc_selected
        )
        direct_tail_structured_pc_reason = (
            direct_tail_factor_setup.direct_tail_structured_pc_reason
        )
        direct_tail_structured_pc_metadata = (
            direct_tail_factor_setup.direct_tail_structured_pc_metadata
        )
        direct_tail_structured_pc_error = (
            direct_tail_factor_setup.direct_tail_structured_pc_error
        )
        direct_tail_pc_env_early = direct_tail_factor_setup.direct_tail_pc_env_early
        direct_tail_direct_reduced_pmat_requested = (
            direct_tail_factor_setup.direct_tail_direct_reduced_pmat_requested
        )
        direct_tail_structured_admission = (
            direct_tail_factor_setup.structured_admission
        )
        direct_tail_pc_env = direct_tail_factor_setup.direct_tail_pc_env
        direct_tail_pc_auto_default = (
            direct_tail_factor_setup.direct_tail_pc_auto_default
        )
        direct_tail_fail_closed_size = (
            direct_tail_factor_setup.direct_tail_fail_closed_size
        )
        direct_tail_auto_large_fail_closed = (
            direct_tail_factor_setup.direct_tail_auto_large_fail_closed
        )
        direct_tail_structured_pc_required = (
            direct_tail_factor_setup.direct_tail_structured_pc_required
        )
        structured_pc_ready = direct_tail_factor_setup.structured_pc_ready
        direct_tail_structured_layout = (
            direct_tail_factor_setup.direct_tail_structured_layout
        )
        direct_tail_structured_active_indices = (
            direct_tail_factor_setup.direct_tail_structured_active_indices
        )
        direct_tail_structured_max_nbytes = (
            direct_tail_factor_setup.direct_tail_structured_max_nbytes
        )
        direct_tail_support_mode_preflight_requested = (
            direct_tail_factor_setup.direct_tail_support_mode_preflight_requested
        )
        direct_tail_support_mode_preflight_selected = (
            direct_tail_factor_setup.direct_tail_support_mode_preflight_selected
        )
        direct_tail_support_mode_preflight_metadata = (
            direct_tail_factor_setup.direct_tail_support_mode_preflight_metadata
        )
        direct_tail_support_mode_preflight_error = (
            direct_tail_factor_setup.direct_tail_support_mode_preflight_error
        )
        direct_tail_structured_pc_max_mb_auto = (
            direct_tail_factor_setup.direct_tail_structured_pc_max_mb_auto
        )
        pc_max_mb = direct_tail_factor_setup.pc_max_mb
        pc_reg = direct_tail_factor_setup.pc_reg
        _operator_bundle_pc = direct_tail_factor_setup.operator_bundle_pc
        factor_bundle_pc = direct_tail_factor_setup.factor_bundle_pc
        pc_factor_s = direct_tail_factor_setup.pc_factor_s
        setup_s = direct_tail_factor_setup.setup_s

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
        def _support_true_matvec(v_np: np.ndarray) -> np.ndarray:
            return np.asarray(
                jax.device_get(_mv_true_no_count(jnp.asarray(v_np, dtype=rhs.dtype))),
                dtype=np.float64,
            ).reshape((-1,))

        direct_tail_rescue_policy_setup = (
            build_sparse_pc_direct_tail_rescue_policy_setup(
                SparsePCDirectTailRescuePolicySetupContext(
                    env=os.environ,
                    op=op,
                    rhs_dtype=rhs.dtype,
                    sparse_pc_rhs=sparse_pc_rhs,
                    sparse_pc_linear_size=int(sparse_pc_linear_size),
                    fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
                    factor_bundle_pc=factor_bundle_pc,
                    structured_pc_ready=bool(structured_pc_ready),
                    direct_tail_operator_bundle=direct_tail_operator_bundle,
                    direct_tail_structured_layout=direct_tail_structured_layout,
                    direct_tail_structured_active_indices=(
                        direct_tail_structured_active_indices
                    ),
                    direct_tail_structured_max_nbytes=(
                        direct_tail_structured_max_nbytes
                    ),
                    direct_tail_structured_pc_selected=(
                        direct_tail_structured_pc_selected
                    ),
                    direct_tail_structured_pc_reason=direct_tail_structured_pc_reason,
                    direct_tail_structured_pc_metadata=(
                        direct_tail_structured_pc_metadata
                    ),
                    pc_reg=float(pc_reg),
                    preconditioner_x=int(preconditioner_x),
                    preconditioner_xi=int(preconditioner_xi),
                    preconditioner_species=int(preconditioner_species),
                    preconditioner_x_min_l=int(preconditioner_x_min_l),
                    emit=emit,
                    true_matvec=_support_true_matvec,
                    support_mode_selector=(
                        select_active_fortran_v3_reduced_support_mode_preconditioner
                    ),
                    structured_factor_bundle_factory=(
                        _StructuredHostSparsePreconditionerBundle
                    ),
                    layout_from_operator=RHS1BlockLayout.from_operator,
                    parse_true_operator_window_specs=(
                        _parse_true_operator_window_specs
                    ),
                )
            )
        )
        factor_bundle_pc = direct_tail_rescue_policy_setup.factor_bundle_pc
        direct_tail_structured_pc_selected = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_selected
        )
        direct_tail_structured_pc_reason = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_reason
        )
        direct_tail_structured_pc_metadata = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_metadata
        )
        direct_tail_support_mode_preflight_requested = (
            direct_tail_rescue_policy_setup.direct_tail_support_mode_preflight_requested
        )
        direct_tail_support_mode_preflight_selected = (
            direct_tail_rescue_policy_setup.direct_tail_support_mode_preflight_selected
        )
        direct_tail_support_mode_preflight_metadata = (
            direct_tail_rescue_policy_setup.direct_tail_support_mode_preflight_metadata
        )
        direct_tail_support_mode_preflight_error = (
            direct_tail_rescue_policy_setup.direct_tail_support_mode_preflight_error
        )
        factor_preflight_policy = (
            direct_tail_rescue_policy_setup.factor_preflight_policy
        )
        factor_preflight_enabled = (
            direct_tail_rescue_policy_setup.factor_preflight_enabled
        )
        factor_preflight_required = (
            direct_tail_rescue_policy_setup.factor_preflight_required
        )
        factor_preflight_seed_enabled = (
            direct_tail_rescue_policy_setup.factor_preflight_seed_enabled
        )
        structured_pc_preflight_required_min_size = (
            direct_tail_rescue_policy_setup.structured_pc_preflight_required_min_size
        )
        direct_tail_structured_pc_requires_preflight = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_requires_preflight
        )
        direct_tail_structured_pc_kind_for_preflight = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_kind_for_preflight
        )
        direct_tail_structured_pc_size_requires_preflight = (
            direct_tail_rescue_policy_setup.direct_tail_structured_pc_size_requires_preflight
        )
        structured_pc_preflight_required = (
            direct_tail_rescue_policy_setup.structured_pc_preflight_required
        )
        factor_preflight_max_target_ratio = (
            direct_tail_rescue_policy_setup.factor_preflight_max_target_ratio
        )
        factor_preflight_residual_before = (
            direct_tail_rescue_policy_setup.factor_preflight_residual_before
        )
        factor_preflight_residual_after = (
            direct_tail_rescue_policy_setup.factor_preflight_residual_after
        )
        factor_preflight_improvement_ratio = (
            direct_tail_rescue_policy_setup.factor_preflight_improvement_ratio
        )
        factor_preflight_target_ratio = (
            direct_tail_rescue_policy_setup.factor_preflight_target_ratio
        )
        factor_preflight_residual_diagnostics = (
            direct_tail_rescue_policy_setup.factor_preflight_residual_diagnostics
        )
        factor_preflight_seed_used = (
            direct_tail_rescue_policy_setup.factor_preflight_seed_used
        )
        factor_preflight_passed = (
            direct_tail_rescue_policy_setup.factor_preflight_passed
        )
        factor_preflight_error = (
            direct_tail_rescue_policy_setup.factor_preflight_error
        )
        direct_tail_residual_rescue_policy = (
            direct_tail_rescue_policy_setup.direct_tail_residual_rescue_policy
        )
        direct_tail_true_active_rescue_policy = (
            direct_tail_rescue_policy_setup.direct_tail_true_active_rescue_policy
        )
        direct_tail_true_coupled_coarse_policy = (
            direct_tail_rescue_policy_setup.direct_tail_true_coupled_coarse_policy
        )
        (
            direct_tail_residual_coarse_requested,
            direct_tail_residual_coarse_selected,
            direct_tail_residual_coarse_metadata,
            direct_tail_residual_coarse_error,
            direct_tail_residual_coarse_residual_after,
            direct_tail_residual_coarse_rank,
            direct_tail_residual_coarse_max_mb,
            direct_tail_residual_coarse_regularization,
            direct_tail_residual_window_requested,
            direct_tail_true_window_requested,
            direct_tail_true_coupled_coarse_explicit_requested,
            direct_tail_true_coupled_coarse_auto_enabled,
            direct_tail_true_coupled_coarse_auto_native_enabled,
            direct_tail_true_coupled_coarse_auto_target_ratio,
            direct_tail_true_coupled_coarse_auto_min_size,
            direct_tail_true_coupled_coarse_requested,
            direct_tail_true_coupled_coarse_auto_selected,
            direct_tail_true_coupled_coarse_selected,
            direct_tail_true_coupled_coarse_metadata,
            direct_tail_true_coupled_coarse_error,
            direct_tail_true_coupled_coarse_residual_after,
            direct_tail_true_window_selected,
            direct_tail_true_window_metadata,
            direct_tail_true_window_error,
            direct_tail_true_window_residual_after,
            direct_tail_residual_window_selected,
            direct_tail_residual_window_metadata,
            direct_tail_residual_window_error,
            direct_tail_residual_window_residual_after,
            direct_tail_residual_window_max_windows,
            direct_tail_residual_window_x_radius,
            direct_tail_residual_window_ell_radius,
            direct_tail_residual_window_max_mb,
            direct_tail_residual_window_regularization,
            direct_tail_residual_window_coefficient_mode,
            direct_tail_residual_window_combine_mode,
            direct_tail_residual_window_interface_depth,
            direct_tail_residual_window_max_size,
            direct_tail_true_window_max_windows,
            direct_tail_true_window_x_radius,
            direct_tail_true_window_ell_radius,
            direct_tail_true_window_max_mb,
            direct_tail_true_window_regularization,
            direct_tail_true_window_max_size,
            direct_tail_true_window_column_batch,
            direct_tail_true_window_drop_tol,
            direct_tail_true_window_include_tail,
            direct_tail_true_window_damping,
            direct_tail_true_window_beta_max,
            direct_tail_true_window_specs_env,
            direct_tail_true_window_specs,
            direct_tail_true_active_block_requested,
            direct_tail_true_active_residual_block_requested,
            direct_tail_true_active_submatrix_requested,
            direct_tail_true_active_submatrix_selected,
            direct_tail_true_active_submatrix_metadata,
            direct_tail_true_active_submatrix_error,
            direct_tail_true_active_submatrix_residual_after,
            direct_tail_true_active_block_selected,
            direct_tail_true_active_block_metadata,
            direct_tail_true_active_block_error,
            direct_tail_true_active_block_residual_after,
            direct_tail_true_active_residual_block_selected,
            direct_tail_true_active_residual_block_metadata,
            direct_tail_true_active_residual_block_error,
            direct_tail_true_active_residual_block_residual_after,
            direct_tail_true_active_column_cache_requested,
            direct_tail_true_active_column_cache_max_mb,
            direct_tail_true_active_column_cache_metadata,
            direct_tail_true_active_block_x_count,
            direct_tail_true_active_block_ell_count,
            direct_tail_true_active_block_species_count,
            direct_tail_true_active_block_theta_stride,
            direct_tail_true_active_block_zeta_stride,
            direct_tail_true_active_block_max_mb,
            direct_tail_true_active_block_regularization,
            direct_tail_true_active_block_max_size,
            direct_tail_true_active_block_column_batch,
            direct_tail_true_active_block_drop_tol,
            direct_tail_true_active_block_include_tail,
            direct_tail_true_active_block_max_tail,
            direct_tail_true_active_block_damping,
            direct_tail_true_active_block_beta_max,
            direct_tail_true_active_residual_block_max_mb,
            direct_tail_true_active_residual_block_regularization,
            direct_tail_true_active_residual_block_max_size,
            direct_tail_true_active_residual_block_column_batch,
            direct_tail_true_active_residual_block_drop_tol,
            direct_tail_true_active_residual_block_include_tail,
            direct_tail_true_active_residual_block_max_tail,
            direct_tail_true_active_residual_block_kinetic_only,
            direct_tail_true_active_residual_block_damping,
            direct_tail_true_active_residual_block_beta_max,
            direct_tail_true_active_residual_block_min_improvement,
            direct_tail_true_active_residual_block_accept_base_improvement,
            direct_tail_true_active_residual_block_base_improvement_override_used,
            direct_tail_true_active_submatrix_damping,
            direct_tail_true_active_submatrix_alpha_clip,
            direct_tail_true_active_submatrix_min_improvement,
            direct_tail_true_coupled_coarse_max_windows,
            direct_tail_true_coupled_coarse_x_radius,
            direct_tail_true_coupled_coarse_ell_radius,
            direct_tail_true_coupled_coarse_max_mb,
            direct_tail_true_coupled_coarse_regularization,
            direct_tail_true_coupled_coarse_max_size,
            direct_tail_true_coupled_coarse_column_batch,
            direct_tail_true_coupled_coarse_drop_tol,
            direct_tail_true_coupled_coarse_low_lmax,
            direct_tail_true_coupled_coarse_profile_moment_count,
            direct_tail_true_coupled_coarse_angular_lmax,
            direct_tail_true_coupled_coarse_angular_mode_max,
            direct_tail_true_coupled_coarse_max_tail_units,
            direct_tail_true_coupled_coarse_include_tail,
            direct_tail_true_coupled_coarse_include_constraint_sources,
            direct_tail_true_coupled_coarse_include_fsavg,
            direct_tail_true_coupled_coarse_include_window_residual,
            direct_tail_true_coupled_coarse_include_profile_moments,
            direct_tail_true_coupled_coarse_include_angular_residual,
            direct_tail_true_coupled_coarse_include_angular_basis,
            direct_tail_true_coupled_coarse_include_preconditioned_loads,
            direct_tail_true_coupled_coarse_preconditioned_load_max_columns,
            direct_tail_true_coupled_coarse_preconditioned_load_max_nnz,
            direct_tail_true_coupled_coarse_preconditioned_load_drop_tol,
            direct_tail_true_coupled_coarse_damping,
            direct_tail_true_coupled_coarse_beta_max,
            direct_tail_true_coupled_coarse_accept_base_improvement,
            direct_tail_true_coupled_coarse_base_improvement_override_used,
        ) = direct_tail_rescue_policy_setup.solve_rescue_tuple()
        residual_vec_current = jnp.asarray(sparse_pc_rhs, dtype=jnp.float64)
        if bool(factor_preflight_enabled) and x0_sparse is None:
            try:
                factor_preflight_run = run_sparse_pc_factor_preflight(
                    SparsePCFactorPreflightRunContext(
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
                        emit=emit,
                    )
                )
                factor_preflight_residual_before = float(factor_preflight_run.residual_before)
                factor_preflight_residual_after = float(factor_preflight_run.residual_after)
                factor_preflight_residual_diagnostics = factor_preflight_run.residual_diagnostics
                factor_preflight_improvement_ratio = factor_preflight_run.improvement_ratio
                factor_preflight_target_ratio = factor_preflight_run.target_ratio
                factor_preflight_passed = bool(factor_preflight_run.passed)
                factor_preflight_seed_used = bool(factor_preflight_run.seed_used)
                residual_vec_current = factor_preflight_run.residual_vec
                if factor_preflight_run.x0_seed is not None:
                    x0_sparse = factor_preflight_run.x0_seed
                true_coupled_stage = run_sparse_pc_true_coupled_coarse_stage(
                    SparsePCTrueCoupledCoarseStageContext(
                        factor_bundle_pc=factor_bundle_pc,
                        pc_factor_s=float(pc_factor_s),
                        setup_s=float(setup_s),
                        sparse_pc_rhs=sparse_pc_rhs,
                        sparse_pc_linear_size=int(sparse_pc_linear_size),
                        target=float(target),
                        factor_preflight_residual_before=factor_preflight_residual_before,
                        factor_preflight_residual_after=factor_preflight_residual_after,
                        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
                        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
                        factor_preflight_target_ratio=factor_preflight_target_ratio,
                        factor_preflight_passed=factor_preflight_passed,
                        factor_preflight_seed_enabled=bool(factor_preflight_seed_enabled),
                        factor_preflight_seed_used=bool(factor_preflight_seed_used),
                        factor_preflight_max_target_ratio=float(factor_preflight_max_target_ratio),
                        residual_vec_current=residual_vec_current,
                        x0_sparse=x0_sparse,
                        matvec_no_count=_mv_true_no_count,
                        matmat=_mv_true_matmat,
                        diagnostics=_rhs1_active_reduced_residual_diagnostics,
                        op=op,
                        layout=RHS1BlockLayout.from_operator(op),
                        active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                        elapsed_s=sparse_timer.elapsed_s,
                        emit=emit,
                        builder=_try_build_true_operator_coupled_coarse_lsq_preconditioner,
                        additive_rescue_nbytes=_rhs1_additive_rescue_nbytes,
                        explicit_requested=bool(direct_tail_true_coupled_coarse_explicit_requested),
                        auto_enabled=bool(direct_tail_true_coupled_coarse_auto_enabled),
                        auto_native_enabled=bool(direct_tail_true_coupled_coarse_auto_native_enabled),
                        auto_target_ratio=float(direct_tail_true_coupled_coarse_auto_target_ratio),
                        auto_min_size=int(direct_tail_true_coupled_coarse_auto_min_size),
                        max_windows=int(direct_tail_true_coupled_coarse_max_windows),
                        x_radius=int(direct_tail_true_coupled_coarse_x_radius),
                        ell_radius=int(direct_tail_true_coupled_coarse_ell_radius),
                        max_mb=float(direct_tail_true_coupled_coarse_max_mb),
                        regularization=float(direct_tail_true_coupled_coarse_regularization),
                        max_size=int(direct_tail_true_coupled_coarse_max_size),
                        column_batch=int(direct_tail_true_coupled_coarse_column_batch),
                        drop_tol=float(direct_tail_true_coupled_coarse_drop_tol),
                        low_lmax=int(direct_tail_true_coupled_coarse_low_lmax),
                        profile_moment_count=int(direct_tail_true_coupled_coarse_profile_moment_count),
                        angular_lmax=int(direct_tail_true_coupled_coarse_angular_lmax),
                        angular_mode_max=int(direct_tail_true_coupled_coarse_angular_mode_max),
                        max_tail_units=int(direct_tail_true_coupled_coarse_max_tail_units),
                        include_tail=bool(direct_tail_true_coupled_coarse_include_tail),
                        include_constraint_sources=bool(direct_tail_true_coupled_coarse_include_constraint_sources),
                        include_fsavg=bool(direct_tail_true_coupled_coarse_include_fsavg),
                        include_window_residual=bool(direct_tail_true_coupled_coarse_include_window_residual),
                        include_profile_moments=bool(direct_tail_true_coupled_coarse_include_profile_moments),
                        include_angular_residual=bool(direct_tail_true_coupled_coarse_include_angular_residual),
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
                        accept_base_improvement=bool(direct_tail_true_coupled_coarse_accept_base_improvement),
                    )
                )
                factor_bundle_pc = true_coupled_stage.factor_bundle_pc
                pc_factor_s = float(true_coupled_stage.pc_factor_s)
                setup_s = float(true_coupled_stage.setup_s)
                factor_preflight_residual_after = true_coupled_stage.factor_preflight_residual_after
                factor_preflight_residual_diagnostics = true_coupled_stage.factor_preflight_residual_diagnostics
                factor_preflight_improvement_ratio = true_coupled_stage.factor_preflight_improvement_ratio
                factor_preflight_target_ratio = true_coupled_stage.factor_preflight_target_ratio
                factor_preflight_passed = true_coupled_stage.factor_preflight_passed
                factor_preflight_seed_used = bool(true_coupled_stage.factor_preflight_seed_used)
                residual_vec_current = true_coupled_stage.residual_vec_current
                x0_sparse = true_coupled_stage.x0_sparse
                direct_tail_true_coupled_coarse_requested = bool(true_coupled_stage.requested)
                direct_tail_true_coupled_coarse_auto_selected = bool(true_coupled_stage.auto_selected)
                direct_tail_true_coupled_coarse_selected = bool(true_coupled_stage.selected)
                direct_tail_true_coupled_coarse_residual_after = true_coupled_stage.residual_after
                direct_tail_true_coupled_coarse_metadata = true_coupled_stage.metadata
                direct_tail_true_coupled_coarse_error = true_coupled_stage.error
                direct_tail_true_coupled_coarse_base_improvement_override_used = bool(
                    true_coupled_stage.base_improvement_override_used
                )

                residual_correction_stage = run_sparse_pc_residual_correction_stage(
                    SparsePCResidualCorrectionStageContext(
                        factor_bundle_pc=factor_bundle_pc,
                        operator_bundle_pc=_operator_bundle_pc,
                        structured_pc_ready=bool(structured_pc_ready),
                        pc_factor_s=float(pc_factor_s),
                        setup_s=float(setup_s),
                        sparse_pc_rhs=sparse_pc_rhs,
                        sparse_pc_linear_size=int(sparse_pc_linear_size),
                        target=float(target),
                        factor_preflight_residual_before=factor_preflight_residual_before,
                        factor_preflight_residual_after=factor_preflight_residual_after,
                        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
                        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
                        factor_preflight_target_ratio=factor_preflight_target_ratio,
                        factor_preflight_passed=factor_preflight_passed,
                        factor_preflight_seed_enabled=bool(factor_preflight_seed_enabled),
                        factor_preflight_seed_used=bool(factor_preflight_seed_used),
                        factor_preflight_max_target_ratio=float(factor_preflight_max_target_ratio),
                        residual_vec_current=residual_vec_current,
                        x0_sparse=x0_sparse,
                        matvec=_mv_true,
                        matvec_no_count=_mv_true_no_count,
                        matmat=_mv_true_matmat,
                        diagnostics=_rhs1_active_reduced_residual_diagnostics,
                        layout=RHS1BlockLayout.from_operator(op),
                        active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                        elapsed_s=sparse_timer.elapsed_s,
                        emit=emit,
                        additive_rescue_nbytes=_rhs1_additive_rescue_nbytes,
                        true_action_column_cache_factory=_ReusableTrueActionColumnCache,
                        true_active_submatrix_builder=_try_build_true_operator_active_submatrix_preconditioner,
                        true_active_block_builder=_try_build_true_operator_active_block_lsq_preconditioner,
                        true_active_residual_block_builder=(
                            _try_build_true_operator_active_residual_block_lsq_preconditioner
                        ),
                        true_window_builder=_try_build_true_operator_residual_window_lsq_preconditioner,
                        residual_coarse_builder=_try_build_residual_coarse_host_sparse_preconditioner,
                        residual_window_builder=_try_build_residual_window_host_sparse_preconditioner,
                        continue_after_base_improvement=_rhs1_bool_env(
                            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESCUE_AFTER_BASE_IMPROVEMENT",
                            default=True,
                        ),
                        true_coupled_base_improvement_override_used=bool(
                            direct_tail_true_coupled_coarse_base_improvement_override_used
                        ),
                        true_active_submatrix_requested=bool(direct_tail_true_active_submatrix_requested),
                        true_active_block_requested=bool(direct_tail_true_active_block_requested),
                        true_active_residual_block_requested=bool(direct_tail_true_active_residual_block_requested),
                        true_window_requested=bool(direct_tail_true_window_requested),
                        residual_coarse_requested=bool(direct_tail_residual_coarse_requested),
                        residual_window_requested=bool(direct_tail_residual_window_requested),
                        true_active_column_cache_requested=bool(direct_tail_true_active_column_cache_requested),
                        true_active_column_cache_max_mb=float(direct_tail_true_active_column_cache_max_mb),
                        true_active_block_x_count=int(direct_tail_true_active_block_x_count),
                        true_active_block_ell_count=int(direct_tail_true_active_block_ell_count),
                        true_active_block_species_count=direct_tail_true_active_block_species_count,
                        true_active_block_theta_stride=int(direct_tail_true_active_block_theta_stride),
                        true_active_block_zeta_stride=int(direct_tail_true_active_block_zeta_stride),
                        true_active_block_max_mb=float(direct_tail_true_active_block_max_mb),
                        true_active_block_regularization=float(direct_tail_true_active_block_regularization),
                        true_active_block_max_size=int(direct_tail_true_active_block_max_size),
                        true_active_block_column_batch=int(direct_tail_true_active_block_column_batch),
                        true_active_block_drop_tol=float(direct_tail_true_active_block_drop_tol),
                        true_active_block_include_tail=bool(direct_tail_true_active_block_include_tail),
                        true_active_block_max_tail=int(direct_tail_true_active_block_max_tail),
                        true_active_block_damping=bool(direct_tail_true_active_block_damping),
                        true_active_block_beta_max=float(direct_tail_true_active_block_beta_max),
                        true_active_submatrix_damping=bool(direct_tail_true_active_submatrix_damping),
                        true_active_submatrix_alpha_clip=float(direct_tail_true_active_submatrix_alpha_clip),
                        true_active_submatrix_min_improvement=float(
                            direct_tail_true_active_submatrix_min_improvement
                        ),
                        true_active_residual_block_max_mb=float(direct_tail_true_active_residual_block_max_mb),
                        true_active_residual_block_regularization=float(
                            direct_tail_true_active_residual_block_regularization
                        ),
                        true_active_residual_block_max_size=int(direct_tail_true_active_residual_block_max_size),
                        true_active_residual_block_column_batch=int(
                            direct_tail_true_active_residual_block_column_batch
                        ),
                        true_active_residual_block_drop_tol=float(direct_tail_true_active_residual_block_drop_tol),
                        true_active_residual_block_include_tail=bool(
                            direct_tail_true_active_residual_block_include_tail
                        ),
                        true_active_residual_block_max_tail=int(direct_tail_true_active_residual_block_max_tail),
                        true_active_residual_block_kinetic_only=bool(
                            direct_tail_true_active_residual_block_kinetic_only
                        ),
                        true_active_residual_block_damping=bool(direct_tail_true_active_residual_block_damping),
                        true_active_residual_block_beta_max=float(direct_tail_true_active_residual_block_beta_max),
                        true_active_residual_block_min_improvement=float(
                            direct_tail_true_active_residual_block_min_improvement
                        ),
                        true_active_residual_block_accept_base_improvement=bool(
                            direct_tail_true_active_residual_block_accept_base_improvement
                        ),
                        true_window_max_windows=int(direct_tail_true_window_max_windows),
                        true_window_x_radius=int(direct_tail_true_window_x_radius),
                        true_window_ell_radius=int(direct_tail_true_window_ell_radius),
                        true_window_max_mb=float(direct_tail_true_window_max_mb),
                        true_window_regularization=float(direct_tail_true_window_regularization),
                        true_window_max_size=int(direct_tail_true_window_max_size),
                        true_window_column_batch=int(direct_tail_true_window_column_batch),
                        true_window_drop_tol=float(direct_tail_true_window_drop_tol),
                        true_window_include_tail=bool(direct_tail_true_window_include_tail),
                        true_window_specs=tuple(direct_tail_true_window_specs),
                        true_window_damping=bool(direct_tail_true_window_damping),
                        true_window_beta_max=float(direct_tail_true_window_beta_max),
                        residual_coarse_rank=int(direct_tail_residual_coarse_rank),
                        residual_coarse_max_mb=float(direct_tail_residual_coarse_max_mb),
                        residual_coarse_regularization=float(direct_tail_residual_coarse_regularization),
                        residual_window_max_windows=int(direct_tail_residual_window_max_windows),
                        residual_window_x_radius=int(direct_tail_residual_window_x_radius),
                        residual_window_ell_radius=int(direct_tail_residual_window_ell_radius),
                        residual_window_max_mb=float(direct_tail_residual_window_max_mb),
                        residual_window_regularization=float(direct_tail_residual_window_regularization),
                        residual_window_coefficient_mode=str(direct_tail_residual_window_coefficient_mode),
                        residual_window_combine_mode=str(direct_tail_residual_window_combine_mode),
                        residual_window_interface_depth=int(direct_tail_residual_window_interface_depth),
                        residual_window_max_size=int(direct_tail_residual_window_max_size),
                    )
                )
                factor_bundle_pc = residual_correction_stage.factor_bundle_pc
                pc_factor_s = float(residual_correction_stage.pc_factor_s)
                setup_s = float(residual_correction_stage.setup_s)
                factor_preflight_residual_after = residual_correction_stage.factor_preflight_residual_after
                factor_preflight_residual_diagnostics = (
                    residual_correction_stage.factor_preflight_residual_diagnostics
                )
                factor_preflight_improvement_ratio = residual_correction_stage.factor_preflight_improvement_ratio
                factor_preflight_target_ratio = residual_correction_stage.factor_preflight_target_ratio
                factor_preflight_passed = residual_correction_stage.factor_preflight_passed
                factor_preflight_seed_used = bool(residual_correction_stage.factor_preflight_seed_used)
                residual_vec_current = residual_correction_stage.residual_vec_current
                x0_sparse = residual_correction_stage.x0_sparse
                direct_tail_true_active_submatrix_selected = bool(
                    residual_correction_stage.true_active_submatrix_selected
                )
                direct_tail_true_active_submatrix_residual_after = (
                    residual_correction_stage.true_active_submatrix_residual_after
                )
                direct_tail_true_active_submatrix_error = residual_correction_stage.true_active_submatrix_error
                direct_tail_true_active_submatrix_metadata = residual_correction_stage.true_active_submatrix_metadata
                direct_tail_true_active_block_selected = bool(residual_correction_stage.true_active_block_selected)
                direct_tail_true_active_block_residual_after = (
                    residual_correction_stage.true_active_block_residual_after
                )
                direct_tail_true_active_block_error = residual_correction_stage.true_active_block_error
                direct_tail_true_active_block_metadata = residual_correction_stage.true_active_block_metadata
                direct_tail_true_active_residual_block_selected = bool(
                    residual_correction_stage.true_active_residual_block_selected
                )
                direct_tail_true_active_residual_block_residual_after = (
                    residual_correction_stage.true_active_residual_block_residual_after
                )
                direct_tail_true_active_residual_block_error = (
                    residual_correction_stage.true_active_residual_block_error
                )
                direct_tail_true_active_residual_block_metadata = (
                    residual_correction_stage.true_active_residual_block_metadata
                )
                direct_tail_true_active_residual_block_base_improvement_override_used = bool(
                    residual_correction_stage.true_active_residual_block_base_improvement_override_used
                )
                direct_tail_true_window_selected = bool(residual_correction_stage.true_window_selected)
                direct_tail_true_window_residual_after = residual_correction_stage.true_window_residual_after
                direct_tail_true_window_error = residual_correction_stage.true_window_error
                direct_tail_true_window_metadata = residual_correction_stage.true_window_metadata
                direct_tail_residual_coarse_selected = bool(residual_correction_stage.residual_coarse_selected)
                direct_tail_residual_coarse_residual_after = residual_correction_stage.residual_coarse_residual_after
                direct_tail_residual_coarse_error = residual_correction_stage.residual_coarse_error
                direct_tail_residual_coarse_metadata = residual_correction_stage.residual_coarse_metadata
                direct_tail_residual_window_selected = bool(residual_correction_stage.residual_window_selected)
                direct_tail_residual_window_residual_after = residual_correction_stage.residual_window_residual_after
                direct_tail_residual_window_error = residual_correction_stage.residual_window_error
                direct_tail_residual_window_metadata = residual_correction_stage.residual_window_metadata
                direct_tail_true_active_column_cache_metadata = (
                    residual_correction_stage.true_active_column_cache_metadata
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
            auto_preflight_retry_stage = run_sparse_pc_auto_preflight_retry_stage(
                SparsePCAutoPreflightRetryStageContext(
                    env=os.environ,
                    structured_pc_ready=bool(structured_pc_ready),
                    structured_pc_preflight_required=bool(structured_pc_preflight_required),
                    factor_preflight_passed=factor_preflight_passed,
                    direct_tail_structured_pc_requested=direct_tail_structured_pc_requested,
                    direct_tail_operator_bundle=direct_tail_operator_bundle,
                    direct_tail_structured_layout=direct_tail_structured_layout,
                    direct_tail_structured_active_indices=direct_tail_structured_active_indices,
                    direct_tail_structured_max_nbytes=direct_tail_structured_max_nbytes,
                    direct_tail_structured_pc_selected=bool(direct_tail_structured_pc_selected),
                    direct_tail_structured_pc_reason=direct_tail_structured_pc_reason,
                    direct_tail_structured_pc_metadata=direct_tail_structured_pc_metadata,
                    operator_bundle_pc=_operator_bundle_pc,
                    factor_bundle_pc=factor_bundle_pc,
                    pc_factor_s=float(pc_factor_s),
                    pc_reg=float(pc_reg),
                    preconditioner_x=int(preconditioner_x),
                    preconditioner_xi=int(preconditioner_xi),
                    preconditioner_species=int(preconditioner_species),
                    preconditioner_x_min_l=int(preconditioner_x_min_l),
                    sparse_pc_rhs=sparse_pc_rhs,
                    sparse_pc_linear_size=int(sparse_pc_linear_size),
                    structured_pc_preflight_required_min_size=int(structured_pc_preflight_required_min_size),
                    factor_preflight_max_target_ratio=float(factor_preflight_max_target_ratio),
                    factor_preflight_residual_before=factor_preflight_residual_before,
                    factor_preflight_residual_after=factor_preflight_residual_after,
                    factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
                    factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
                    factor_preflight_target_ratio=factor_preflight_target_ratio,
                    factor_preflight_seed_enabled=bool(factor_preflight_seed_enabled),
                    factor_preflight_seed_used=bool(factor_preflight_seed_used),
                    residual_vec_current=residual_vec_current,
                    target=float(target),
                    matvec_no_count=_mv_true_no_count,
                    diagnostics=_rhs1_active_reduced_residual_diagnostics,
                    layout=direct_tail_structured_layout,
                    active_indices=sparse_pc_active_idx_np if sparse_pc_use_active_dof else None,
                    elapsed_s=sparse_timer.elapsed_s,
                    emit=emit,
                    structured_preconditioner_builder=build_active_projected_rhs1_full_csr_preconditioner,
                    factor_bundle_factory=_StructuredHostSparsePreconditionerBundle,
                )
            )
            factor_bundle_pc = auto_preflight_retry_stage.factor_bundle_pc
            direct_tail_structured_pc_selected = auto_preflight_retry_stage.direct_tail_structured_pc_selected
            direct_tail_structured_pc_reason = auto_preflight_retry_stage.direct_tail_structured_pc_reason
            direct_tail_structured_pc_metadata = auto_preflight_retry_stage.direct_tail_structured_pc_metadata
            _operator_bundle_pc = auto_preflight_retry_stage.operator_bundle_pc
            pc_factor_s = float(auto_preflight_retry_stage.pc_factor_s)
            setup_s = float(auto_preflight_retry_stage.setup_s)
            residual_vec_current = auto_preflight_retry_stage.residual_vec_current
            factor_preflight_residual_after = auto_preflight_retry_stage.factor_preflight_residual_after
            factor_preflight_residual_diagnostics = (
                auto_preflight_retry_stage.factor_preflight_residual_diagnostics
            )
            factor_preflight_improvement_ratio = auto_preflight_retry_stage.factor_preflight_improvement_ratio
            factor_preflight_target_ratio = auto_preflight_retry_stage.factor_preflight_target_ratio
            factor_preflight_passed = auto_preflight_retry_stage.factor_preflight_passed
            factor_preflight_seed_used = auto_preflight_retry_stage.factor_preflight_seed_used
            if auto_preflight_retry_stage.x0_sparse is not None:
                x0_sparse = auto_preflight_retry_stage.x0_sparse
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
            sparse_pc_gmres_finalization_bundle_from_solve_result(
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
    return None


@dataclass(frozen=True)
class SparsePCFactorPreflightRunContext:
    """Inputs for the solve-time sparse-PC factor preflight check."""

    rhs: jnp.ndarray
    rhs_norm: float
    target: float
    preconditioner: ArrayFn
    matvec: ArrayFn
    diagnostics: Callable[..., dict[str, object]]
    layout: object
    active_indices: object | None
    seed_enabled: bool
    max_target_ratio: float
    emit: EmitFn | None


@dataclass(frozen=True)
class SparsePCFactorPreflightRunResult:
    """State produced by one sparse-PC factor preflight execution."""

    residual_before: float
    residual_after: float
    residual_diagnostics: dict[str, object] | None
    improvement_ratio: float | None
    target_ratio: float | None
    passed: bool
    seed_used: bool
    residual_vec: jnp.ndarray
    x0_seed: jnp.ndarray | None


@dataclass(frozen=True)
class SparsePCAutoPreflightRetryStageContext:
    """Inputs for retrying structured direct-tail preconditioners after preflight."""

    env: Mapping[str, str] | None
    structured_pc_ready: bool
    structured_pc_preflight_required: bool
    factor_preflight_passed: bool | None
    direct_tail_structured_pc_requested: str | None
    direct_tail_operator_bundle: object | None
    direct_tail_structured_layout: object | None
    direct_tail_structured_active_indices: np.ndarray | None
    direct_tail_structured_max_nbytes: int | None
    direct_tail_structured_pc_selected: bool
    direct_tail_structured_pc_reason: str | None
    direct_tail_structured_pc_metadata: dict[str, object] | None
    operator_bundle_pc: object | None
    factor_bundle_pc: object
    pc_factor_s: float
    pc_reg: float
    preconditioner_x: int
    preconditioner_xi: int
    preconditioner_species: int
    preconditioner_x_min_l: int
    sparse_pc_rhs: jnp.ndarray
    sparse_pc_linear_size: int
    structured_pc_preflight_required_min_size: int
    factor_preflight_max_target_ratio: float
    factor_preflight_residual_before: float | None
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_seed_enabled: bool
    factor_preflight_seed_used: bool
    residual_vec_current: jnp.ndarray
    target: float
    matvec_no_count: ArrayFn
    diagnostics: Callable[..., dict[str, object]]
    layout: object
    active_indices: object | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    structured_preconditioner_builder: Callable[..., object]
    factor_bundle_factory: Callable[..., object]


@dataclass(frozen=True)
class SparsePCAutoPreflightRetryStageResult:
    """Updated sparse-PC state after auto preflight retry attempts."""

    selected: bool
    attempts: tuple[dict[str, object], ...]
    factor_bundle_pc: object
    direct_tail_structured_pc_selected: bool
    direct_tail_structured_pc_reason: str | None
    direct_tail_structured_pc_metadata: dict[str, object] | None
    operator_bundle_pc: object | None
    pc_factor_s: float
    setup_s: float
    residual_vec_current: jnp.ndarray
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_used: bool
    x0_sparse: jnp.ndarray | None


@dataclass(frozen=True)
class SparsePCTrueCoupledCoarseStageContext:
    """Inputs for the true-operator coupled coarse rescue stage."""

    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float
    sparse_pc_rhs: jnp.ndarray
    sparse_pc_linear_size: int
    target: float
    factor_preflight_residual_before: float | None
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_enabled: bool
    factor_preflight_seed_used: bool
    factor_preflight_max_target_ratio: float
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    matvec_no_count: ArrayFn
    matmat: Callable[[np.ndarray], np.ndarray]
    diagnostics: Callable[..., dict[str, object]]
    op: object
    layout: object
    active_indices: object | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., object]
    additive_rescue_nbytes: Callable[[object, float], int]
    explicit_requested: bool
    auto_enabled: bool
    auto_native_enabled: bool
    auto_target_ratio: float
    auto_min_size: int
    max_windows: int
    x_radius: int
    ell_radius: int
    max_mb: float
    regularization: float
    max_size: int
    column_batch: int
    drop_tol: float
    low_lmax: int
    profile_moment_count: int
    angular_lmax: int
    angular_mode_max: int
    max_tail_units: int
    include_tail: bool
    include_constraint_sources: bool
    include_fsavg: bool
    include_window_residual: bool
    include_profile_moments: bool
    include_angular_residual: bool
    include_angular_basis: bool
    include_preconditioned_loads: bool
    preconditioned_load_max_columns: int
    preconditioned_load_max_nnz: int
    preconditioned_load_drop_tol: float
    damping: bool
    beta_max: float
    accept_base_improvement: bool


@dataclass(frozen=True)
class SparsePCTrueCoupledCoarseStageResult:
    """Updated sparse-PC state after a true-operator coupled coarse attempt."""

    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_used: bool
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    requested: bool
    auto_selected: bool
    selected: bool
    residual_after: float | None
    metadata: dict[str, object] | None
    error: str | None
    base_improvement_override_used: bool


@dataclass(frozen=True)
class SparsePCResidualCandidateUpdateContext:
    """State update contract shared by sparse residual rescue candidates."""

    label: str
    metadata_count_key: str | None
    metadata_count_label: str | None
    bundle: object
    candidate_x: jnp.ndarray
    candidate_residual_vec: jnp.ndarray
    candidate_residual_after: float
    candidate_metadata: dict[str, object]
    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float
    factor_preflight_residual_before: float | None
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_enabled: bool
    factor_preflight_seed_used: bool
    target: float
    max_target_ratio: float
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    diagnostics: Callable[..., dict[str, object]]
    layout: object
    active_indices: object | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    require_original_improvement: bool = True
    current_min_improvement: float = 0.0
    accept_base_improvement: bool = False
    base_improvement_requires_original_miss: bool = True
    base_improvement_sets_passed: bool = False
    missing_original_improves: bool = False


@dataclass(frozen=True)
class SparsePCResidualCandidateUpdateResult:
    """Updated sparse-PC state after evaluating one residual candidate."""

    accepted: bool
    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_used: bool
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    metadata: dict[str, object]
    base_improvement_override_used: bool
    improves_current_residual: bool
    improves_original_residual: bool


@dataclass(frozen=True)
class SparsePCResidualCorrectionStageContext:
    """Inputs for the direct-tail residual correction candidate family."""

    factor_bundle_pc: object
    operator_bundle_pc: object | None
    structured_pc_ready: bool
    pc_factor_s: float
    setup_s: float
    sparse_pc_rhs: jnp.ndarray
    sparse_pc_linear_size: int
    target: float
    factor_preflight_residual_before: float | None
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_enabled: bool
    factor_preflight_seed_used: bool
    factor_preflight_max_target_ratio: float
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    matvec: ArrayFn
    matvec_no_count: ArrayFn
    matmat: Callable[[np.ndarray], np.ndarray]
    diagnostics: Callable[..., dict[str, object]]
    layout: object
    active_indices: object | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    additive_rescue_nbytes: Callable[[object, float], int]
    true_action_column_cache_factory: Callable[..., object]
    true_active_submatrix_builder: Callable[..., object]
    true_active_block_builder: Callable[..., object]
    true_active_residual_block_builder: Callable[..., object]
    true_window_builder: Callable[..., object]
    residual_coarse_builder: Callable[..., object]
    residual_window_builder: Callable[..., object]
    continue_after_base_improvement: bool
    true_coupled_base_improvement_override_used: bool
    true_active_submatrix_requested: bool
    true_active_block_requested: bool
    true_active_residual_block_requested: bool
    true_window_requested: bool
    residual_coarse_requested: bool
    residual_window_requested: bool
    true_active_column_cache_requested: bool
    true_active_column_cache_max_mb: float
    true_active_block_x_count: int
    true_active_block_ell_count: int
    true_active_block_species_count: object
    true_active_block_theta_stride: int
    true_active_block_zeta_stride: int
    true_active_block_max_mb: float
    true_active_block_regularization: float
    true_active_block_max_size: int
    true_active_block_column_batch: int
    true_active_block_drop_tol: float
    true_active_block_include_tail: bool
    true_active_block_max_tail: int
    true_active_block_damping: bool
    true_active_block_beta_max: float
    true_active_submatrix_damping: bool
    true_active_submatrix_alpha_clip: float
    true_active_submatrix_min_improvement: float
    true_active_residual_block_max_mb: float
    true_active_residual_block_regularization: float
    true_active_residual_block_max_size: int
    true_active_residual_block_column_batch: int
    true_active_residual_block_drop_tol: float
    true_active_residual_block_include_tail: bool
    true_active_residual_block_max_tail: int
    true_active_residual_block_kinetic_only: bool
    true_active_residual_block_damping: bool
    true_active_residual_block_beta_max: float
    true_active_residual_block_min_improvement: float
    true_active_residual_block_accept_base_improvement: bool
    true_window_max_windows: int
    true_window_x_radius: int
    true_window_ell_radius: int
    true_window_max_mb: float
    true_window_regularization: float
    true_window_max_size: int
    true_window_column_batch: int
    true_window_drop_tol: float
    true_window_include_tail: bool
    true_window_specs: tuple[object, ...]
    true_window_damping: bool
    true_window_beta_max: float
    residual_coarse_rank: int
    residual_coarse_max_mb: float
    residual_coarse_regularization: float
    residual_window_max_windows: int
    residual_window_x_radius: int
    residual_window_ell_radius: int
    residual_window_max_mb: float
    residual_window_regularization: float
    residual_window_coefficient_mode: str
    residual_window_combine_mode: str
    residual_window_interface_depth: int
    residual_window_max_size: int


@dataclass(frozen=True)
class SparsePCResidualCorrectionStageResult:
    """Updated sparse-PC state after direct-tail residual corrections."""

    factor_bundle_pc: object
    pc_factor_s: float
    setup_s: float
    factor_preflight_residual_after: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_passed: bool | None
    factor_preflight_seed_used: bool
    residual_vec_current: jnp.ndarray
    x0_sparse: jnp.ndarray | None
    true_active_submatrix_selected: bool
    true_active_submatrix_residual_after: float | None
    true_active_submatrix_error: str | None
    true_active_submatrix_metadata: dict[str, object] | None
    true_active_block_selected: bool
    true_active_block_residual_after: float | None
    true_active_block_error: str | None
    true_active_block_metadata: dict[str, object] | None
    true_active_residual_block_selected: bool
    true_active_residual_block_residual_after: float | None
    true_active_residual_block_error: str | None
    true_active_residual_block_metadata: dict[str, object] | None
    true_active_residual_block_base_improvement_override_used: bool
    true_window_selected: bool
    true_window_residual_after: float | None
    true_window_error: str | None
    true_window_metadata: dict[str, object] | None
    residual_coarse_selected: bool
    residual_coarse_residual_after: float | None
    residual_coarse_error: str | None
    residual_coarse_metadata: dict[str, object] | None
    residual_window_selected: bool
    residual_window_residual_after: float | None
    residual_window_error: str | None
    residual_window_metadata: dict[str, object] | None
    true_active_column_cache_metadata: dict[str, object] | None


def run_sparse_pc_factor_preflight(
    context: SparsePCFactorPreflightRunContext,
) -> SparsePCFactorPreflightRunResult:
    """Run sparse-PC factor preflight and emit the standard progress messages."""

    evaluation = evaluate_sparse_pc_factor_preflight(
        SparsePCFactorPreflightEvaluationContext(
            rhs=context.rhs,
            rhs_norm=float(context.rhs_norm),
            target=float(context.target),
            preconditioner=context.preconditioner,
            matvec=context.matvec,
            diagnostics=context.diagnostics,
            layout=context.layout,
            active_indices=context.active_indices,
            seed_enabled=bool(context.seed_enabled),
            max_target_ratio=float(context.max_target_ratio),
        )
    )
    residual_diagnostics = evaluation.diagnostics
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres factor preflight "
            f"residual={float(evaluation.residual_before):.6e}"
            f"->{float(evaluation.residual_after):.6e} "
            f"improvement={float(evaluation.improvement_ratio or 0.0):.6e} "
            f"target_ratio={float(evaluation.target_ratio or float('inf')):.6e} "
            f"seed_used={bool(evaluation.seed_used)} "
            f"passed={bool(evaluation.passed)}",
        )
        if isinstance(residual_diagnostics, dict) and residual_diagnostics.get("selected"):
            component_norms = residual_diagnostics.get("component_norms", {})
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
            top_sx = residual_diagnostics.get("top_species_x", [])
            top_sx_label = top_sx[0].get("label") if isinstance(top_sx, list) and top_sx else "none"
            top_sx_fraction = (
                top_sx[0].get("energy_fraction", 0.0) if isinstance(top_sx, list) and top_sx else 0.0
            )
            top_x = residual_diagnostics.get("top_x", [])
            top_x_label = top_x[0].get("label") if isinstance(top_x, list) and top_x else "none"
            top_x_fraction = top_x[0].get("energy_fraction", 0.0) if isinstance(top_x, list) and top_x else 0.0
            top_ell = residual_diagnostics.get("top_ell", [])
            top_ell_label = top_ell[0].get("label") if isinstance(top_ell, list) and top_ell else "none"
            top_ell_fraction = (
                top_ell[0].get("energy_fraction", 0.0) if isinstance(top_ell, list) and top_ell else 0.0
            )
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres preflight residual diagnostics "
                f"kinetic_energy_fraction={float(kinetic_fraction):.6e} "
                f"extra_energy_fraction={float(extra_fraction):.6e} "
                f"top_species_x={top_sx_label} "
                f"top_species_x_fraction={float(top_sx_fraction):.6e} "
                f"top_x={top_x_label} top_x_fraction={float(top_x_fraction):.6e} "
                f"top_ell={top_ell_label} top_ell_fraction={float(top_ell_fraction):.6e}",
            )

    return SparsePCFactorPreflightRunResult(
        residual_before=float(evaluation.residual_before),
        residual_after=float(evaluation.residual_after),
        residual_diagnostics=residual_diagnostics,
        improvement_ratio=evaluation.improvement_ratio,
        target_ratio=evaluation.target_ratio,
        passed=bool(evaluation.passed),
        seed_used=bool(evaluation.seed_used),
        residual_vec=evaluation.residual_vec,
        x0_seed=evaluation.x0_seed,
    )


def run_sparse_pc_auto_preflight_retry_stage(
    context: SparsePCAutoPreflightRetryStageContext,
) -> SparsePCAutoPreflightRetryStageResult:
    """Try alternate structured preconditioners when direct-tail preflight fails."""

    auto_preflight_retry_selected = False
    auto_preflight_retry_attempts: list[dict[str, object]] = []
    auto_preflight_retry_enabled = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_RETRY",
        default=True,
    )
    factor_bundle_pc = context.factor_bundle_pc
    direct_tail_structured_pc_selected = bool(context.direct_tail_structured_pc_selected)
    direct_tail_structured_pc_reason = context.direct_tail_structured_pc_reason
    direct_tail_structured_pc_metadata = context.direct_tail_structured_pc_metadata
    operator_bundle_pc = context.operator_bundle_pc
    pc_factor_s = float(context.pc_factor_s)
    setup_s = float(context.elapsed_s())
    residual_vec_current = context.residual_vec_current
    factor_preflight_residual_after = context.factor_preflight_residual_after
    factor_preflight_residual_diagnostics = context.factor_preflight_residual_diagnostics
    factor_preflight_improvement_ratio = context.factor_preflight_improvement_ratio
    factor_preflight_target_ratio = context.factor_preflight_target_ratio
    factor_preflight_passed = context.factor_preflight_passed
    factor_preflight_seed_used = bool(context.factor_preflight_seed_used)
    x0_sparse: jnp.ndarray | None = None

    retry_requested_kind = str(context.direct_tail_structured_pc_requested or "").strip().lower().replace("-", "_")
    if (
        bool(auto_preflight_retry_enabled)
        and bool(context.structured_pc_ready)
        and bool(context.structured_pc_preflight_required)
        and context.factor_preflight_passed is False
        and retry_requested_kind in {"auto", "active_auto", "structured", "structured_auto"}
        and context.direct_tail_operator_bundle is not None
        and context.direct_tail_structured_layout is not None
        and context.direct_tail_structured_max_nbytes is not None
        and isinstance(direct_tail_structured_pc_metadata, dict)
    ):
        metadata_inner = direct_tail_structured_pc_metadata.get("metadata")
        max_retry_candidates = _env_int(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_MAX_CANDIDATES",
            default=2,
            minimum=1,
        )
        retry_selection = select_sparse_pc_auto_preflight_retry_candidates(
            SparsePCAutoPreflightRetrySelectionContext(
                metadata=metadata_inner if isinstance(metadata_inner, dict) else None,
                current_kind=str(getattr(factor_bundle_pc, "kind", "")),
                sparse_pc_linear_size=int(context.sparse_pc_linear_size),
                preflight_required_min_size=int(context.structured_pc_preflight_required_min_size),
                skip_large_kinds_raw=(
                    _env_value(
                        context.env,
                        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_PREFLIGHT_SKIP_LARGE",
                    )
                    or "active_spilu,active_ilu,active_global_sparse_ilu,jacobi,diagonal"
                ),
                max_candidates=int(max_retry_candidates),
            )
        )
        for retry_candidate in retry_selection.retry_candidates:
            retry_start_s = float(context.elapsed_s())
            try:
                retry_pc = context.structured_preconditioner_builder(
                    matrix=context.direct_tail_operator_bundle.matrix,
                    layout=context.direct_tail_structured_layout,
                    active_indices=context.direct_tail_structured_active_indices,
                    kind=str(retry_candidate),
                    max_factor_nbytes=int(context.direct_tail_structured_max_nbytes),
                    regularization=float(context.pc_reg),
                    preconditioner_x=int(context.preconditioner_x),
                    preconditioner_xi=int(context.preconditioner_xi),
                    preconditioner_species=int(context.preconditioner_species),
                    preconditioner_x_min_l=int(context.preconditioner_x_min_l),
                )
            except Exception as exc:  # noqa: BLE001
                auto_preflight_retry_attempts.append(
                    {
                        "kind": str(retry_candidate),
                        "selected": False,
                        "reason": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "setup_s": float(context.elapsed_s() - retry_start_s),
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
            retry_bundle = context.factor_bundle_factory(
                preconditioner=retry_pc,
                operator=context.direct_tail_operator_bundle,
                kind=str(retry_pc.kind),
                factor_nbytes_estimate=None if retry_factor_nbytes is None else int(retry_factor_nbytes),
                factor_nnz_estimate=None,
                factor_s=float(retry_pc.setup_s),
            )
            try:
                retry_x = jnp.asarray(
                    retry_bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                retry_residual_vec = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(retry_x),
                    dtype=jnp.float64,
                )
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
                    target=float(context.target),
                    max_target_ratio=float(context.factor_preflight_max_target_ratio),
                    residual_before=context.factor_preflight_residual_before,
                    sparse_pc_linear_size=int(context.sparse_pc_linear_size),
                    preflight_required_min_size=int(context.structured_pc_preflight_required_min_size),
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
                    "factor_nbytes_estimate": None if retry_factor_nbytes is None else int(retry_factor_nbytes),
                }
            )
            auto_preflight_retry_attempts.append(retry_entry)
            if context.emit is not None:
                context.emit(
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
                operator_bundle_pc = context.direct_tail_operator_bundle
                pc_factor_s += float(retry_pc.setup_s)
                setup_s = float(context.elapsed_s())
                residual_vec_current = retry_residual_vec
                factor_preflight_residual_after = float(retry_residual)
                factor_preflight_residual_diagnostics = context.diagnostics(
                    residual=retry_residual_vec,
                    layout=context.layout,
                    active_indices=context.active_indices,
                )
                if (
                    context.factor_preflight_residual_before is not None
                    and float(context.factor_preflight_residual_before) > 0.0
                ):
                    factor_preflight_improvement_ratio = float(context.factor_preflight_residual_before) / max(
                        float(factor_preflight_residual_after),
                        1.0e-300,
                    )
                factor_preflight_target_ratio = float(retry_evaluation.target_ratio)
                factor_preflight_passed = True
                if bool(context.factor_preflight_seed_enabled):
                    x0_sparse = retry_x
                    factor_preflight_seed_used = True
                auto_preflight_retry_selected = True
                if context.emit is not None:
                    context.emit(
                        1,
                        "solve_v3_full_system_linear_gmres: auto preflight retry accepted "
                        f"kind={retry_candidate} required={bool(retry_evaluation.required)}",
                    )
                break
    if auto_preflight_retry_attempts and isinstance(direct_tail_structured_pc_metadata, dict):
        metadata_inner = direct_tail_structured_pc_metadata.setdefault("metadata", {})
        if isinstance(metadata_inner, dict):
            metadata_inner["auto_preflight_retry_enabled"] = bool(auto_preflight_retry_enabled)
            metadata_inner["auto_preflight_retry_selected"] = bool(auto_preflight_retry_selected)
            metadata_inner["auto_preflight_retry_attempts"] = tuple(
                dict(entry) for entry in auto_preflight_retry_attempts
            )

    return SparsePCAutoPreflightRetryStageResult(
        selected=bool(auto_preflight_retry_selected),
        attempts=tuple(dict(entry) for entry in auto_preflight_retry_attempts),
        factor_bundle_pc=factor_bundle_pc,
        direct_tail_structured_pc_selected=bool(direct_tail_structured_pc_selected),
        direct_tail_structured_pc_reason=direct_tail_structured_pc_reason,
        direct_tail_structured_pc_metadata=direct_tail_structured_pc_metadata,
        operator_bundle_pc=operator_bundle_pc,
        pc_factor_s=float(pc_factor_s),
        setup_s=float(setup_s),
        residual_vec_current=residual_vec_current,
        factor_preflight_residual_after=factor_preflight_residual_after,
        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
        factor_preflight_target_ratio=factor_preflight_target_ratio,
        factor_preflight_passed=factor_preflight_passed,
        factor_preflight_seed_used=bool(factor_preflight_seed_used),
        x0_sparse=x0_sparse,
    )


def apply_sparse_pc_residual_candidate_update(
    context: SparsePCResidualCandidateUpdateContext,
) -> SparsePCResidualCandidateUpdateResult:
    """Evaluate one sparse rescue candidate and update shared preflight state."""

    metadata = dict(context.candidate_metadata or {})
    metadata["residual_after"] = float(context.candidate_residual_after)
    if context.factor_preflight_residual_after is not None:
        metadata["base_residual_after"] = float(context.factor_preflight_residual_after)

    acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
        SparsePCResidualCandidateAcceptanceContext(
            candidate_residual_after=float(context.candidate_residual_after),
            current_residual_after=context.factor_preflight_residual_after,
            original_residual_before=context.factor_preflight_residual_before,
            target=float(context.target),
            max_target_ratio=float(context.max_target_ratio),
            seed_enabled=bool(context.factor_preflight_seed_enabled),
            require_original_improvement=bool(context.require_original_improvement),
            current_min_improvement=float(context.current_min_improvement),
            accept_base_improvement=bool(context.accept_base_improvement),
            base_improvement_requires_original_miss=bool(
                context.base_improvement_requires_original_miss
            ),
            base_improvement_sets_passed=bool(context.base_improvement_sets_passed),
            missing_original_improves=bool(context.missing_original_improves),
        )
    )

    factor_bundle_pc = context.factor_bundle_pc
    pc_factor_s = float(context.pc_factor_s)
    setup_s = float(context.setup_s)
    factor_preflight_residual_after = context.factor_preflight_residual_after
    factor_preflight_residual_diagnostics = context.factor_preflight_residual_diagnostics
    factor_preflight_improvement_ratio = context.factor_preflight_improvement_ratio
    factor_preflight_target_ratio = context.factor_preflight_target_ratio
    factor_preflight_passed = context.factor_preflight_passed
    factor_preflight_seed_used = bool(context.factor_preflight_seed_used)
    residual_vec_current = context.residual_vec_current
    x0_sparse = context.x0_sparse

    if bool(acceptance.accepted):
        factor_bundle_pc = context.bundle
        pc_factor_s += float(getattr(context.bundle, "factor_s", None) or 0.0)
        setup_s = float(context.elapsed_s())
        factor_preflight_residual_after = float(acceptance.residual_after)
        residual_vec_current = context.candidate_residual_vec
        factor_preflight_residual_diagnostics = context.diagnostics(
            residual=context.candidate_residual_vec,
            layout=context.layout,
            active_indices=context.active_indices,
        )
        factor_preflight_improvement_ratio = acceptance.improvement_ratio
        factor_preflight_target_ratio = acceptance.target_ratio
        factor_preflight_passed = bool(acceptance.passed)
        if bool(acceptance.seed_used):
            x0_sparse = context.candidate_x
            factor_preflight_seed_used = True
        if context.emit is not None:
            count_fragment = ""
            if context.metadata_count_key and context.metadata_count_label:
                count_fragment = (
                    f"{context.metadata_count_label}="
                    f"{metadata.get(context.metadata_count_key)} "
                )
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.label} accepted "
                f"{count_fragment}"
                f"residual={metadata.get('base_residual_after', float('nan')):.6e}"
                f"->{float(factor_preflight_residual_after):.6e} "
                f"passed={bool(factor_preflight_passed)}",
            )
    elif context.emit is not None:
        base = (
            float(context.factor_preflight_residual_after)
            if context.factor_preflight_residual_after is not None
            else float("nan")
        )
        context.emit(
            1,
            f"solve_v3_full_system_linear_gmres: {context.label} rejected "
            f"residual={base:.6e}->{float(context.candidate_residual_after):.6e}",
        )

    return SparsePCResidualCandidateUpdateResult(
        accepted=bool(acceptance.accepted),
        factor_bundle_pc=factor_bundle_pc,
        pc_factor_s=float(pc_factor_s),
        setup_s=float(setup_s),
        factor_preflight_residual_after=factor_preflight_residual_after,
        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
        factor_preflight_target_ratio=factor_preflight_target_ratio,
        factor_preflight_passed=factor_preflight_passed,
        factor_preflight_seed_used=bool(factor_preflight_seed_used),
        residual_vec_current=residual_vec_current,
        x0_sparse=x0_sparse,
        metadata=metadata,
        base_improvement_override_used=bool(acceptance.base_improvement_override_used),
        improves_current_residual=bool(acceptance.improves_current_residual),
        improves_original_residual=bool(acceptance.improves_original_residual),
    )


def run_sparse_pc_residual_correction_stage(
    context: SparsePCResidualCorrectionStageContext,
) -> SparsePCResidualCorrectionStageResult:
    """Run the direct-tail true-active and residual-window correction family."""

    factor_bundle_pc = context.factor_bundle_pc
    pc_factor_s = float(context.pc_factor_s)
    setup_s = float(context.setup_s)
    factor_preflight_residual_after = context.factor_preflight_residual_after
    factor_preflight_residual_diagnostics = context.factor_preflight_residual_diagnostics
    factor_preflight_improvement_ratio = context.factor_preflight_improvement_ratio
    factor_preflight_target_ratio = context.factor_preflight_target_ratio
    factor_preflight_passed = context.factor_preflight_passed
    factor_preflight_seed_used = bool(context.factor_preflight_seed_used)
    residual_vec_current = context.residual_vec_current
    x0_sparse = context.x0_sparse

    true_active_submatrix_selected = False
    true_active_submatrix_residual_after: float | None = None
    true_active_submatrix_error: str | None = None
    true_active_submatrix_metadata: dict[str, object] | None = None
    true_active_block_selected = False
    true_active_block_residual_after: float | None = None
    true_active_block_error: str | None = None
    true_active_block_metadata: dict[str, object] | None = None
    true_active_residual_block_selected = False
    true_active_residual_block_residual_after: float | None = None
    true_active_residual_block_error: str | None = None
    true_active_residual_block_metadata: dict[str, object] | None = None
    true_active_residual_block_base_improvement_override_used = False
    true_window_selected = False
    true_window_residual_after: float | None = None
    true_window_error: str | None = None
    true_window_metadata: dict[str, object] | None = None
    residual_coarse_selected = False
    residual_coarse_residual_after: float | None = None
    residual_coarse_error: str | None = None
    residual_coarse_metadata: dict[str, object] | None = None
    residual_window_selected = False
    residual_window_residual_after: float | None = None
    residual_window_error: str | None = None
    residual_window_metadata: dict[str, object] | None = None
    true_active_column_cache_metadata: dict[str, object] | None = None

    def _rescue_needed_after_preflight() -> bool:
        if factor_preflight_passed is False:
            return True
        if not bool(context.true_coupled_base_improvement_override_used):
            return False
        if not bool(context.continue_after_base_improvement):
            return False
        if factor_preflight_target_ratio is None:
            return True
        try:
            return float(factor_preflight_target_ratio) > float(context.factor_preflight_max_target_ratio)
        except (TypeError, ValueError):
            return True

    def _apply_candidate_update(
        *,
        label: str,
        metadata_count_key: str | None,
        metadata_count_label: str | None,
        bundle: object,
        candidate_x: jnp.ndarray,
        candidate_residual_vec: jnp.ndarray,
        candidate_residual_after: float,
        require_original_improvement: bool = True,
        current_min_improvement: float = 0.0,
        accept_base_improvement: bool = False,
        missing_original_improves: bool = False,
    ) -> SparsePCResidualCandidateUpdateResult:
        nonlocal factor_bundle_pc
        nonlocal pc_factor_s
        nonlocal setup_s
        nonlocal factor_preflight_residual_after
        nonlocal factor_preflight_residual_diagnostics
        nonlocal factor_preflight_improvement_ratio
        nonlocal factor_preflight_target_ratio
        nonlocal factor_preflight_passed
        nonlocal factor_preflight_seed_used
        nonlocal residual_vec_current
        nonlocal x0_sparse

        update = apply_sparse_pc_residual_candidate_update(
            SparsePCResidualCandidateUpdateContext(
                label=label,
                metadata_count_key=metadata_count_key,
                metadata_count_label=metadata_count_label,
                bundle=bundle,
                candidate_x=candidate_x,
                candidate_residual_vec=candidate_residual_vec,
                candidate_residual_after=float(candidate_residual_after),
                candidate_metadata=dict(getattr(bundle, "metadata", {}) or {}),
                factor_bundle_pc=factor_bundle_pc,
                pc_factor_s=float(pc_factor_s),
                setup_s=float(setup_s),
                factor_preflight_residual_before=context.factor_preflight_residual_before,
                factor_preflight_residual_after=factor_preflight_residual_after,
                factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
                factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
                factor_preflight_target_ratio=factor_preflight_target_ratio,
                factor_preflight_passed=factor_preflight_passed,
                factor_preflight_seed_enabled=bool(context.factor_preflight_seed_enabled),
                factor_preflight_seed_used=bool(factor_preflight_seed_used),
                target=float(context.target),
                max_target_ratio=float(context.factor_preflight_max_target_ratio),
                residual_vec_current=residual_vec_current,
                x0_sparse=x0_sparse,
                diagnostics=context.diagnostics,
                layout=context.layout,
                active_indices=context.active_indices,
                elapsed_s=context.elapsed_s,
                emit=context.emit,
                require_original_improvement=bool(require_original_improvement),
                current_min_improvement=float(current_min_improvement),
                accept_base_improvement=bool(accept_base_improvement),
                missing_original_improves=bool(missing_original_improves),
            )
        )
        factor_bundle_pc = update.factor_bundle_pc
        pc_factor_s = float(update.pc_factor_s)
        setup_s = float(update.setup_s)
        factor_preflight_residual_after = update.factor_preflight_residual_after
        factor_preflight_residual_diagnostics = update.factor_preflight_residual_diagnostics
        factor_preflight_improvement_ratio = update.factor_preflight_improvement_ratio
        factor_preflight_target_ratio = update.factor_preflight_target_ratio
        factor_preflight_passed = update.factor_preflight_passed
        factor_preflight_seed_used = bool(update.factor_preflight_seed_used)
        residual_vec_current = update.residual_vec_current
        x0_sparse = update.x0_sparse
        return update

    true_active_column_cache = None
    if (
        bool(context.true_active_submatrix_requested)
        or bool(context.true_active_block_requested)
        or bool(context.true_active_residual_block_requested)
        or bool(context.true_window_requested)
    ):
        true_active_column_cache = context.true_action_column_cache_factory(
            true_matvec=lambda vec: np.asarray(
                jax.device_get(context.matvec_no_count(jnp.asarray(vec, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,)),
            true_matmat=lambda mat: np.asarray(context.matmat(np.asarray(mat, dtype=np.float64))),
            n=int(context.sparse_pc_linear_size),
            max_nbytes=int(max(0.0, float(context.true_active_column_cache_max_mb)) * 1024.0 * 1024.0),
            enabled=bool(context.true_active_column_cache_requested),
        )

    def _true_active_cached_matvec(vec: np.ndarray) -> np.ndarray:
        if true_active_column_cache is not None:
            return true_active_column_cache.matvec(vec)
        return np.asarray(
            jax.device_get(context.matvec_no_count(jnp.asarray(vec, dtype=jnp.float64))),
            dtype=np.float64,
        ).reshape((-1,))

    def _true_active_cached_matmat(mat: np.ndarray) -> np.ndarray:
        if true_active_column_cache is not None:
            return true_active_column_cache.matmat(mat)
        return np.asarray(context.matmat(np.asarray(mat, dtype=np.float64)))

    if (
        bool(context.true_active_submatrix_requested)
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.true_active_submatrix_builder(
                true_matvec=_true_active_cached_matvec,
                true_matmat=_true_active_cached_matmat,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                layout=context.layout,
                active_indices=context.active_indices,
                x_count=int(context.true_active_block_x_count),
                ell_count=int(context.true_active_block_ell_count),
                species_count=context.true_active_block_species_count,
                theta_stride=int(context.true_active_block_theta_stride),
                zeta_stride=int(context.true_active_block_zeta_stride),
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.true_active_block_max_mb),
                ),
                regularization=float(context.true_active_block_regularization),
                max_block_size=int(context.true_active_block_max_size),
                column_batch=int(context.true_active_block_column_batch),
                drop_tol=float(context.true_active_block_drop_tol),
                include_tail=bool(context.true_active_block_include_tail),
                max_tail=int(context.true_active_block_max_tail),
                damping=bool(context.true_active_submatrix_damping),
                alpha_clip=float(context.true_active_submatrix_alpha_clip),
                emit=context.emit,
            )
            if bundle is None:
                true_active_submatrix_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(candidate_x),
                    dtype=jnp.float64,
                )
                true_active_submatrix_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="true active submatrix",
                    metadata_count_key="block_size",
                    metadata_count_label="block_size",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(true_active_submatrix_residual_after),
                    require_original_improvement=False,
                    current_min_improvement=float(context.true_active_submatrix_min_improvement),
                )
                true_active_submatrix_metadata = update.metadata
                true_active_submatrix_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            true_active_submatrix_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active submatrix failed "
                    f"({true_active_submatrix_error})",
                )

    if (
        bool(context.true_active_block_requested)
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.true_active_block_builder(
                true_matvec=_true_active_cached_matvec,
                true_matmat=_true_active_cached_matmat,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                layout=context.layout,
                active_indices=context.active_indices,
                x_count=int(context.true_active_block_x_count),
                ell_count=int(context.true_active_block_ell_count),
                species_count=context.true_active_block_species_count,
                theta_stride=int(context.true_active_block_theta_stride),
                zeta_stride=int(context.true_active_block_zeta_stride),
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.true_active_block_max_mb),
                ),
                regularization=float(context.true_active_block_regularization),
                max_block_size=int(context.true_active_block_max_size),
                column_batch=int(context.true_active_block_column_batch),
                drop_tol=float(context.true_active_block_drop_tol),
                include_tail=bool(context.true_active_block_include_tail),
                max_tail=int(context.true_active_block_max_tail),
                damping=bool(context.true_active_block_damping),
                beta_max=float(context.true_active_block_beta_max),
                emit=context.emit,
            )
            if bundle is None:
                true_active_block_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(candidate_x),
                    dtype=jnp.float64,
                )
                true_active_block_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="true active block",
                    metadata_count_key="block_size",
                    metadata_count_label="block_size",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(true_active_block_residual_after),
                    require_original_improvement=False,
                )
                true_active_block_metadata = update.metadata
                true_active_block_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            true_active_block_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active block failed "
                    f"({true_active_block_error})",
                )

    if (
        bool(context.true_active_residual_block_requested)
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.true_active_residual_block_builder(
                true_matvec=_true_active_cached_matvec,
                true_matmat=_true_active_cached_matmat,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                layout=context.layout,
                active_indices=context.active_indices,
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.true_active_residual_block_max_mb),
                ),
                regularization=float(context.true_active_residual_block_regularization),
                max_block_size=int(context.true_active_residual_block_max_size),
                column_batch=int(context.true_active_residual_block_column_batch),
                drop_tol=float(context.true_active_residual_block_drop_tol),
                include_tail=bool(context.true_active_residual_block_include_tail),
                max_tail=int(context.true_active_residual_block_max_tail),
                kinetic_only=bool(context.true_active_residual_block_kinetic_only),
                damping=bool(context.true_active_residual_block_damping),
                beta_max=float(context.true_active_residual_block_beta_max),
                emit=context.emit,
            )
            if bundle is None:
                true_active_residual_block_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(candidate_x),
                    dtype=jnp.float64,
                )
                true_active_residual_block_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="true active residual block",
                    metadata_count_key="block_size",
                    metadata_count_label="block_size",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(true_active_residual_block_residual_after),
                    current_min_improvement=float(context.true_active_residual_block_min_improvement),
                    accept_base_improvement=bool(context.true_active_residual_block_accept_base_improvement),
                    missing_original_improves=True,
                )
                true_active_residual_block_metadata = update.metadata
                true_active_residual_block_base_improvement_override_used = bool(
                    update.base_improvement_override_used
                )
                true_active_residual_block_metadata["accept_base_improvement"] = bool(
                    context.true_active_residual_block_accept_base_improvement
                )
                true_active_residual_block_metadata[
                    "base_improvement_override_used"
                ] = bool(true_active_residual_block_base_improvement_override_used)
                true_active_residual_block_metadata["improves_current_residual"] = bool(
                    update.improves_current_residual
                )
                true_active_residual_block_metadata["improves_original_residual"] = bool(
                    update.improves_original_residual
                )
                true_active_residual_block_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            true_active_residual_block_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active residual block failed "
                    f"({true_active_residual_block_error})",
                )

    if (
        bool(context.true_window_requested)
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.true_window_builder(
                true_matvec=_true_active_cached_matvec,
                true_matmat=_true_active_cached_matmat,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                layout=context.layout,
                active_indices=context.active_indices,
                max_windows=int(context.true_window_max_windows),
                x_radius=int(context.true_window_x_radius),
                ell_radius=int(context.true_window_ell_radius),
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.true_window_max_mb),
                ),
                regularization=float(context.true_window_regularization),
                max_window_size=int(context.true_window_max_size),
                column_batch=int(context.true_window_column_batch),
                drop_tol=float(context.true_window_drop_tol),
                include_tail=bool(context.true_window_include_tail),
                explicit_specs=tuple(context.true_window_specs),
                damping=bool(context.true_window_damping),
                beta_max=float(context.true_window_beta_max),
                emit=context.emit,
            )
            if bundle is None:
                true_window_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(candidate_x),
                    dtype=jnp.float64,
                )
                true_window_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="true residual window",
                    metadata_count_key="window_size",
                    metadata_count_label="window_size",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(true_window_residual_after),
                )
                true_window_metadata = update.metadata
                true_window_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            true_window_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true residual window failed "
                    f"({true_window_error})",
                )

    if (
        bool(context.residual_coarse_requested)
        and bool(context.structured_pc_ready)
        and context.operator_bundle_pc is not None
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.residual_coarse_builder(
                operator_bundle=context.operator_bundle_pc,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                max_rank=int(context.residual_coarse_rank),
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.residual_coarse_max_mb),
                ),
                regularization=float(context.residual_coarse_regularization),
                emit=context.emit,
            )
            if bundle is None:
                residual_coarse_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(context.matvec(candidate_x), dtype=jnp.float64)
                residual_coarse_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="residual coarse",
                    metadata_count_key="rank",
                    metadata_count_label="rank",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(residual_coarse_residual_after),
                )
                residual_coarse_metadata = update.metadata
                residual_coarse_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            residual_coarse_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: residual coarse failed "
                    f"({residual_coarse_error})",
                )

    if (
        bool(context.residual_window_requested)
        and bool(context.structured_pc_ready)
        and context.operator_bundle_pc is not None
        and _rescue_needed_after_preflight()
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            bundle = context.residual_window_builder(
                operator_bundle=context.operator_bundle_pc,
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                layout=context.layout,
                active_indices=context.active_indices,
                max_windows=int(context.residual_window_max_windows),
                x_radius=int(context.residual_window_x_radius),
                ell_radius=int(context.residual_window_ell_radius),
                max_nbytes=context.additive_rescue_nbytes(
                    factor_bundle_pc,
                    float(context.residual_window_max_mb),
                ),
                regularization=float(context.residual_window_regularization),
                coefficient_mode=str(context.residual_window_coefficient_mode),
                combine_mode=str(context.residual_window_combine_mode),
                interface_depth=int(context.residual_window_interface_depth),
                max_window_size=int(context.residual_window_max_size),
                emit=context.emit,
            )
            if bundle is None:
                residual_window_error = "builder_returned_none"
            else:
                candidate_x = jnp.asarray(
                    bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec = context.sparse_pc_rhs - jnp.asarray(context.matvec(candidate_x), dtype=jnp.float64)
                residual_window_residual_after = float(jnp.linalg.norm(residual_vec))
                update = _apply_candidate_update(
                    label="residual window",
                    metadata_count_key="window_count",
                    metadata_count_label="windows",
                    bundle=bundle,
                    candidate_x=candidate_x,
                    candidate_residual_vec=residual_vec,
                    candidate_residual_after=float(residual_window_residual_after),
                )
                residual_window_metadata = update.metadata
                residual_window_selected = bool(update.accepted)
        except Exception as exc:  # noqa: BLE001
            residual_window_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: residual window failed "
                    f"({residual_window_error})",
                )

    if true_active_column_cache is not None:
        true_active_column_cache_metadata = true_active_column_cache.metadata()
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: true active column cache "
                f"hits={true_active_column_cache_metadata['hits']} "
                f"misses={true_active_column_cache_metadata['misses']} "
                f"stored_columns={true_active_column_cache_metadata['stored_columns']} "
                f"stored_mb={float(true_active_column_cache_metadata['stored_nbytes']) / 1.0e6:.3f}",
            )

    return SparsePCResidualCorrectionStageResult(
        factor_bundle_pc=factor_bundle_pc,
        pc_factor_s=float(pc_factor_s),
        setup_s=float(setup_s),
        factor_preflight_residual_after=factor_preflight_residual_after,
        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
        factor_preflight_target_ratio=factor_preflight_target_ratio,
        factor_preflight_passed=factor_preflight_passed,
        factor_preflight_seed_used=bool(factor_preflight_seed_used),
        residual_vec_current=residual_vec_current,
        x0_sparse=x0_sparse,
        true_active_submatrix_selected=bool(true_active_submatrix_selected),
        true_active_submatrix_residual_after=true_active_submatrix_residual_after,
        true_active_submatrix_error=true_active_submatrix_error,
        true_active_submatrix_metadata=true_active_submatrix_metadata,
        true_active_block_selected=bool(true_active_block_selected),
        true_active_block_residual_after=true_active_block_residual_after,
        true_active_block_error=true_active_block_error,
        true_active_block_metadata=true_active_block_metadata,
        true_active_residual_block_selected=bool(true_active_residual_block_selected),
        true_active_residual_block_residual_after=true_active_residual_block_residual_after,
        true_active_residual_block_error=true_active_residual_block_error,
        true_active_residual_block_metadata=true_active_residual_block_metadata,
        true_active_residual_block_base_improvement_override_used=bool(
            true_active_residual_block_base_improvement_override_used
        ),
        true_window_selected=bool(true_window_selected),
        true_window_residual_after=true_window_residual_after,
        true_window_error=true_window_error,
        true_window_metadata=true_window_metadata,
        residual_coarse_selected=bool(residual_coarse_selected),
        residual_coarse_residual_after=residual_coarse_residual_after,
        residual_coarse_error=residual_coarse_error,
        residual_coarse_metadata=residual_coarse_metadata,
        residual_window_selected=bool(residual_window_selected),
        residual_window_residual_after=residual_window_residual_after,
        residual_window_error=residual_window_error,
        residual_window_metadata=residual_window_metadata,
        true_active_column_cache_metadata=true_active_column_cache_metadata,
    )


def run_sparse_pc_true_coupled_coarse_stage(
    context: SparsePCTrueCoupledCoarseStageContext,
) -> SparsePCTrueCoupledCoarseStageResult:
    """Try the bounded true-operator coupled coarse correction after preflight."""

    factor_bundle_pc = context.factor_bundle_pc
    pc_factor_s = float(context.pc_factor_s)
    setup_s = float(context.setup_s)
    residual_vec_current = context.residual_vec_current
    x0_sparse = context.x0_sparse
    factor_preflight_residual_after = context.factor_preflight_residual_after
    factor_preflight_residual_diagnostics = context.factor_preflight_residual_diagnostics
    factor_preflight_improvement_ratio = context.factor_preflight_improvement_ratio
    factor_preflight_target_ratio = context.factor_preflight_target_ratio
    factor_preflight_passed = context.factor_preflight_passed
    factor_preflight_seed_used = bool(context.factor_preflight_seed_used)
    selected = False
    residual_after: float | None = None
    metadata: dict[str, object] | None = None
    error: str | None = None
    base_improvement_override_used = False

    factor_kind = str(getattr(factor_bundle_pc, "kind", "")).strip().lower().replace("-", "_")
    auto_reference_kind = factor_kind in {"active_fortran_v3_reduced_lu"}
    auto_native_kind = factor_kind in {
        "active_fortran_v3_reduced_native_stack",
        "active_v3_reduced_native_stack",
        "fortran_v3_reduced_native_stack",
        "active_bounded_native_stack",
        "active_native_stack",
    }
    auto_selected = bool(
        context.auto_enabled
        and (
            bool(auto_reference_kind)
            or (bool(context.auto_native_enabled) and bool(auto_native_kind))
        )
        and int(context.sparse_pc_linear_size) >= int(context.auto_min_size)
        and factor_preflight_target_ratio is not None
        and np.isfinite(float(factor_preflight_target_ratio))
        and float(factor_preflight_target_ratio) > float(context.auto_target_ratio)
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    )
    requested = bool(context.explicit_requested or auto_selected)

    if (
        bool(requested)
        and (factor_preflight_passed is False or bool(auto_selected))
        and factor_preflight_residual_after is not None
        and np.isfinite(float(factor_preflight_residual_after))
    ):
        try:
            true_coupled_bundle = context.builder(
                true_matvec=lambda vec: np.asarray(
                    jax.device_get(context.matvec_no_count(jnp.asarray(vec, dtype=jnp.float64))),
                    dtype=np.float64,
                ),
                true_matmat=lambda mat: np.asarray(context.matmat(np.asarray(mat, dtype=np.float64))),
                factor_bundle=factor_bundle_pc,
                residual=np.asarray(jax.device_get(residual_vec_current), dtype=np.float64),
                op=context.op,
                layout=context.layout,
                active_indices=context.active_indices,
                max_windows=int(context.max_windows),
                x_radius=int(context.x_radius),
                ell_radius=int(context.ell_radius),
                max_nbytes=context.additive_rescue_nbytes(factor_bundle_pc, float(context.max_mb)),
                regularization=float(context.regularization),
                max_coarse_size=int(context.max_size),
                column_batch=int(context.column_batch),
                drop_tol=float(context.drop_tol),
                low_lmax=int(context.low_lmax),
                profile_moment_count=int(context.profile_moment_count),
                angular_lmax=int(context.angular_lmax),
                angular_mode_max=int(context.angular_mode_max),
                max_tail_units=int(context.max_tail_units),
                include_tail=bool(context.include_tail),
                include_constraint_sources=bool(context.include_constraint_sources),
                include_fsavg=bool(context.include_fsavg),
                include_window_residual=bool(context.include_window_residual),
                include_profile_moments=bool(context.include_profile_moments),
                include_angular_residual=bool(context.include_angular_residual),
                include_angular_basis=bool(context.include_angular_basis),
                include_preconditioned_loads=bool(context.include_preconditioned_loads),
                preconditioned_load_max_columns=int(context.preconditioned_load_max_columns),
                preconditioned_load_max_nnz=int(context.preconditioned_load_max_nnz),
                preconditioned_load_drop_tol=float(context.preconditioned_load_drop_tol),
                damping=bool(context.damping),
                beta_max=float(context.beta_max),
                emit=context.emit,
            )
            if true_coupled_bundle is None:
                error = "builder_returned_none"
            else:
                x_true_coupled_sparse = jnp.asarray(
                    true_coupled_bundle.solve(np.asarray(context.sparse_pc_rhs, dtype=np.float64)),
                    dtype=jnp.float64,
                )
                residual_vec_true_coupled = context.sparse_pc_rhs - jnp.asarray(
                    context.matvec_no_count(x_true_coupled_sparse),
                    dtype=jnp.float64,
                )
                residual_after = float(jnp.linalg.norm(residual_vec_true_coupled))
                true_coupled_diagnostics = context.diagnostics(
                    residual=residual_vec_true_coupled,
                    layout=context.layout,
                    active_indices=context.active_indices,
                )
                metadata = dict(true_coupled_bundle.metadata or {})
                metadata["residual_after"] = float(residual_after)
                metadata["base_residual_after"] = float(factor_preflight_residual_after)
                acceptance = evaluate_sparse_pc_residual_candidate_acceptance(
                    SparsePCResidualCandidateAcceptanceContext(
                        candidate_residual_after=float(residual_after),
                        current_residual_after=float(factor_preflight_residual_after),
                        original_residual_before=context.factor_preflight_residual_before,
                        target=float(context.target),
                        max_target_ratio=float(context.factor_preflight_max_target_ratio),
                        seed_enabled=bool(context.factor_preflight_seed_enabled),
                        accept_base_improvement=bool(context.accept_base_improvement),
                        base_improvement_requires_original_miss=False,
                        base_improvement_sets_passed=True,
                    )
                )
                if bool(acceptance.accepted):
                    selected = True
                    base_improvement_override_used = bool(acceptance.base_improvement_override_used)
                    factor_bundle_pc = true_coupled_bundle
                    pc_factor_s += float(true_coupled_bundle.factor_s or 0.0)
                    setup_s = float(context.elapsed_s())
                    factor_preflight_residual_after = float(acceptance.residual_after)
                    residual_vec_current = residual_vec_true_coupled
                    factor_preflight_residual_diagnostics = true_coupled_diagnostics
                    factor_preflight_improvement_ratio = acceptance.improvement_ratio
                    factor_preflight_target_ratio = acceptance.target_ratio
                    factor_preflight_passed = bool(acceptance.passed)
                    if bool(acceptance.seed_used):
                        x0_sparse = x_true_coupled_sparse
                        factor_preflight_seed_used = True
                    if context.emit is not None:
                        context.emit(
                            1,
                            "solve_v3_full_system_linear_gmres: true coupled coarse accepted "
                            f"coarse_size={metadata.get('coarse_size')} "
                            f"residual={metadata['base_residual_after']:.6e}"
                            f"->{float(factor_preflight_residual_after):.6e} "
                            f"passed={bool(factor_preflight_passed)} "
                            f"base_improvement_override={bool(base_improvement_override_used)}",
                        )
                elif context.emit is not None:
                    context.emit(
                        1,
                        "solve_v3_full_system_linear_gmres: true coupled coarse rejected "
                        f"residual={float(factor_preflight_residual_after):.6e}"
                        f"->{float(residual_after):.6e}",
                    )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true coupled coarse failed "
                    f"({error})",
                )

    return SparsePCTrueCoupledCoarseStageResult(
        factor_bundle_pc=factor_bundle_pc,
        pc_factor_s=float(pc_factor_s),
        setup_s=float(setup_s),
        factor_preflight_residual_after=factor_preflight_residual_after,
        factor_preflight_residual_diagnostics=factor_preflight_residual_diagnostics,
        factor_preflight_improvement_ratio=factor_preflight_improvement_ratio,
        factor_preflight_target_ratio=factor_preflight_target_ratio,
        factor_preflight_passed=factor_preflight_passed,
        factor_preflight_seed_used=bool(factor_preflight_seed_used),
        residual_vec_current=residual_vec_current,
        x0_sparse=x0_sparse,
        requested=bool(requested),
        auto_selected=bool(auto_selected),
        selected=bool(selected),
        residual_after=residual_after,
        metadata=metadata,
        error=error,
        base_improvement_override_used=bool(base_improvement_override_used),
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
    pattern: object | None
    sparse_pattern_scope: str
    pattern_build_s: float
    summary: object | None
    factor_policy: SparsePCFactorPolicySetup | None


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


@dataclass(frozen=True)
class SparsePCDirectTailRescuePolicySetupContext:
    """Inputs for direct-tail support preflight and rescue-policy expansion."""

    env: Mapping[str, str] | None
    op: object
    rhs_dtype: object
    sparse_pc_rhs: jnp.ndarray
    sparse_pc_linear_size: int
    fortran_reduced_sparse_pc: bool
    factor_bundle_pc: object
    structured_pc_ready: bool
    direct_tail_operator_bundle: object | None
    direct_tail_structured_layout: object | None
    direct_tail_structured_active_indices: np.ndarray | None
    direct_tail_structured_max_nbytes: int | None
    direct_tail_structured_pc_selected: bool
    direct_tail_structured_pc_reason: str | None
    direct_tail_structured_pc_metadata: dict[str, object] | None
    pc_reg: float
    preconditioner_x: int
    preconditioner_xi: int
    preconditioner_species: int
    preconditioner_x_min_l: int
    emit: EmitFn | None
    true_matvec: Callable[[np.ndarray], np.ndarray]
    support_mode_selector: Callable[..., object]
    structured_factor_bundle_factory: Callable[..., object]
    layout_from_operator: Callable[[object], object]
    parse_true_operator_window_specs: Callable[..., tuple[object, ...]]


@dataclass(frozen=True)
class SparsePCDirectTailRescuePolicySetupResult:
    """Direct-tail support/preflight/rescue policy state for sparse-PC solves."""

    factor_bundle_pc: object
    direct_tail_structured_pc_selected: bool
    direct_tail_structured_pc_reason: str | None
    direct_tail_structured_pc_metadata: dict[str, object] | None
    direct_tail_support_mode_preflight_requested: bool
    direct_tail_support_mode_preflight_selected: bool
    direct_tail_support_mode_preflight_metadata: dict[str, object] | None
    direct_tail_support_mode_preflight_error: str | None
    factor_preflight_policy: SparsePCFactorPreflightPolicy
    factor_preflight_enabled: bool
    factor_preflight_required: bool
    factor_preflight_seed_enabled: bool
    structured_pc_preflight_required_min_size: int
    direct_tail_structured_pc_requires_preflight: bool
    direct_tail_structured_pc_kind_for_preflight: str
    direct_tail_structured_pc_size_requires_preflight: bool
    structured_pc_preflight_required: bool
    factor_preflight_max_target_ratio: float
    factor_preflight_residual_before: float | None
    factor_preflight_residual_after: float | None
    factor_preflight_improvement_ratio: float | None
    factor_preflight_target_ratio: float | None
    factor_preflight_residual_diagnostics: dict[str, object] | None
    factor_preflight_seed_used: bool
    factor_preflight_passed: bool | None
    factor_preflight_error: str | None
    direct_tail_residual_rescue_policy: DirectTailResidualRescuePolicy
    direct_tail_true_active_rescue_policy: DirectTailTrueActiveRescuePolicy
    direct_tail_true_coupled_coarse_policy: DirectTailCoupledCoarseRescuePolicy
    rescue_values: Mapping[str, object]

    def solve_rescue_tuple(self) -> tuple[object, ...]:
        """Return solve-local rescue state in the order used by orchestration.

        This keeps current solve behavior stable while the surrounding
        sparse-PC rescue stages are still being collapsed into owner modules.
        """

        return tuple(
            self.rescue_values[name]
            for name in (
                "direct_tail_residual_coarse_requested",
                "direct_tail_residual_coarse_selected",
                "direct_tail_residual_coarse_metadata",
                "direct_tail_residual_coarse_error",
                "direct_tail_residual_coarse_residual_after",
                "direct_tail_residual_coarse_rank",
                "direct_tail_residual_coarse_max_mb",
                "direct_tail_residual_coarse_regularization",
                "direct_tail_residual_window_requested",
                "direct_tail_true_window_requested",
                "direct_tail_true_coupled_coarse_explicit_requested",
                "direct_tail_true_coupled_coarse_auto_enabled",
                "direct_tail_true_coupled_coarse_auto_native_enabled",
                "direct_tail_true_coupled_coarse_auto_target_ratio",
                "direct_tail_true_coupled_coarse_auto_min_size",
                "direct_tail_true_coupled_coarse_requested",
                "direct_tail_true_coupled_coarse_auto_selected",
                "direct_tail_true_coupled_coarse_selected",
                "direct_tail_true_coupled_coarse_metadata",
                "direct_tail_true_coupled_coarse_error",
                "direct_tail_true_coupled_coarse_residual_after",
                "direct_tail_true_window_selected",
                "direct_tail_true_window_metadata",
                "direct_tail_true_window_error",
                "direct_tail_true_window_residual_after",
                "direct_tail_residual_window_selected",
                "direct_tail_residual_window_metadata",
                "direct_tail_residual_window_error",
                "direct_tail_residual_window_residual_after",
                "direct_tail_residual_window_max_windows",
                "direct_tail_residual_window_x_radius",
                "direct_tail_residual_window_ell_radius",
                "direct_tail_residual_window_max_mb",
                "direct_tail_residual_window_regularization",
                "direct_tail_residual_window_coefficient_mode",
                "direct_tail_residual_window_combine_mode",
                "direct_tail_residual_window_interface_depth",
                "direct_tail_residual_window_max_size",
                "direct_tail_true_window_max_windows",
                "direct_tail_true_window_x_radius",
                "direct_tail_true_window_ell_radius",
                "direct_tail_true_window_max_mb",
                "direct_tail_true_window_regularization",
                "direct_tail_true_window_max_size",
                "direct_tail_true_window_column_batch",
                "direct_tail_true_window_drop_tol",
                "direct_tail_true_window_include_tail",
                "direct_tail_true_window_damping",
                "direct_tail_true_window_beta_max",
                "direct_tail_true_window_specs_env",
                "direct_tail_true_window_specs",
                "direct_tail_true_active_block_requested",
                "direct_tail_true_active_residual_block_requested",
                "direct_tail_true_active_submatrix_requested",
                "direct_tail_true_active_submatrix_selected",
                "direct_tail_true_active_submatrix_metadata",
                "direct_tail_true_active_submatrix_error",
                "direct_tail_true_active_submatrix_residual_after",
                "direct_tail_true_active_block_selected",
                "direct_tail_true_active_block_metadata",
                "direct_tail_true_active_block_error",
                "direct_tail_true_active_block_residual_after",
                "direct_tail_true_active_residual_block_selected",
                "direct_tail_true_active_residual_block_metadata",
                "direct_tail_true_active_residual_block_error",
                "direct_tail_true_active_residual_block_residual_after",
                "direct_tail_true_active_column_cache_requested",
                "direct_tail_true_active_column_cache_max_mb",
                "direct_tail_true_active_column_cache_metadata",
                "direct_tail_true_active_block_x_count",
                "direct_tail_true_active_block_ell_count",
                "direct_tail_true_active_block_species_count",
                "direct_tail_true_active_block_theta_stride",
                "direct_tail_true_active_block_zeta_stride",
                "direct_tail_true_active_block_max_mb",
                "direct_tail_true_active_block_regularization",
                "direct_tail_true_active_block_max_size",
                "direct_tail_true_active_block_column_batch",
                "direct_tail_true_active_block_drop_tol",
                "direct_tail_true_active_block_include_tail",
                "direct_tail_true_active_block_max_tail",
                "direct_tail_true_active_block_damping",
                "direct_tail_true_active_block_beta_max",
                "direct_tail_true_active_residual_block_max_mb",
                "direct_tail_true_active_residual_block_regularization",
                "direct_tail_true_active_residual_block_max_size",
                "direct_tail_true_active_residual_block_column_batch",
                "direct_tail_true_active_residual_block_drop_tol",
                "direct_tail_true_active_residual_block_include_tail",
                "direct_tail_true_active_residual_block_max_tail",
                "direct_tail_true_active_residual_block_kinetic_only",
                "direct_tail_true_active_residual_block_damping",
                "direct_tail_true_active_residual_block_beta_max",
                "direct_tail_true_active_residual_block_min_improvement",
                "direct_tail_true_active_residual_block_accept_base_improvement",
                "direct_tail_true_active_residual_block_base_improvement_override_used",
                "direct_tail_true_active_submatrix_damping",
                "direct_tail_true_active_submatrix_alpha_clip",
                "direct_tail_true_active_submatrix_min_improvement",
                "direct_tail_true_coupled_coarse_max_windows",
                "direct_tail_true_coupled_coarse_x_radius",
                "direct_tail_true_coupled_coarse_ell_radius",
                "direct_tail_true_coupled_coarse_max_mb",
                "direct_tail_true_coupled_coarse_regularization",
                "direct_tail_true_coupled_coarse_max_size",
                "direct_tail_true_coupled_coarse_column_batch",
                "direct_tail_true_coupled_coarse_drop_tol",
                "direct_tail_true_coupled_coarse_low_lmax",
                "direct_tail_true_coupled_coarse_profile_moment_count",
                "direct_tail_true_coupled_coarse_angular_lmax",
                "direct_tail_true_coupled_coarse_angular_mode_max",
                "direct_tail_true_coupled_coarse_max_tail_units",
                "direct_tail_true_coupled_coarse_include_tail",
                "direct_tail_true_coupled_coarse_include_constraint_sources",
                "direct_tail_true_coupled_coarse_include_fsavg",
                "direct_tail_true_coupled_coarse_include_window_residual",
                "direct_tail_true_coupled_coarse_include_profile_moments",
                "direct_tail_true_coupled_coarse_include_angular_residual",
                "direct_tail_true_coupled_coarse_include_angular_basis",
                "direct_tail_true_coupled_coarse_include_preconditioned_loads",
                "direct_tail_true_coupled_coarse_preconditioned_load_max_columns",
                "direct_tail_true_coupled_coarse_preconditioned_load_max_nnz",
                "direct_tail_true_coupled_coarse_preconditioned_load_drop_tol",
                "direct_tail_true_coupled_coarse_damping",
                "direct_tail_true_coupled_coarse_beta_max",
                "direct_tail_true_coupled_coarse_accept_base_improvement",
                "direct_tail_true_coupled_coarse_base_improvement_override_used",
            )
        )


def _direct_tail_residual_rescue_driver_values(
    policy: DirectTailResidualRescuePolicy,
    *,
    true_window_specs_env: str,
    true_window_specs: tuple[object, ...],
    true_coupled_coarse_explicit_requested: bool,
) -> dict[str, object]:
    """Return default driver state derived from residual-window policy."""

    return {
        "direct_tail_residual_coarse_requested": bool(policy.residual_coarse_requested),
        "direct_tail_residual_coarse_selected": False,
        "direct_tail_residual_coarse_metadata": None,
        "direct_tail_residual_coarse_error": None,
        "direct_tail_residual_coarse_residual_after": None,
        "direct_tail_residual_coarse_rank": int(policy.residual_coarse_rank),
        "direct_tail_residual_coarse_max_mb": float(policy.residual_coarse_max_mb),
        "direct_tail_residual_coarse_regularization": float(
            policy.residual_coarse_regularization
        ),
        "direct_tail_residual_window_requested": bool(policy.residual_window_requested),
        "direct_tail_true_window_requested": bool(policy.true_window_requested),
        "direct_tail_true_coupled_coarse_explicit_requested": bool(
            true_coupled_coarse_explicit_requested
        ),
        "direct_tail_true_coupled_coarse_auto_enabled": bool(
            policy.true_coupled_coarse_auto_enabled
        ),
        "direct_tail_true_coupled_coarse_auto_native_enabled": bool(
            policy.true_coupled_coarse_auto_native_enabled
        ),
        "direct_tail_true_coupled_coarse_auto_target_ratio": float(
            policy.true_coupled_coarse_auto_target_ratio
        ),
        "direct_tail_true_coupled_coarse_auto_min_size": int(
            policy.true_coupled_coarse_auto_min_size
        ),
        "direct_tail_true_coupled_coarse_requested": bool(
            true_coupled_coarse_explicit_requested
        ),
        "direct_tail_true_coupled_coarse_auto_selected": False,
        "direct_tail_true_coupled_coarse_selected": False,
        "direct_tail_true_coupled_coarse_metadata": None,
        "direct_tail_true_coupled_coarse_error": None,
        "direct_tail_true_coupled_coarse_residual_after": None,
        "direct_tail_true_window_selected": False,
        "direct_tail_true_window_metadata": None,
        "direct_tail_true_window_error": None,
        "direct_tail_true_window_residual_after": None,
        "direct_tail_residual_window_selected": False,
        "direct_tail_residual_window_metadata": None,
        "direct_tail_residual_window_error": None,
        "direct_tail_residual_window_residual_after": None,
        "direct_tail_residual_window_max_windows": int(
            policy.residual_window_max_windows
        ),
        "direct_tail_residual_window_x_radius": int(policy.residual_window_x_radius),
        "direct_tail_residual_window_ell_radius": int(
            policy.residual_window_ell_radius
        ),
        "direct_tail_residual_window_max_mb": float(policy.residual_window_max_mb),
        "direct_tail_residual_window_regularization": float(
            policy.residual_window_regularization
        ),
        "direct_tail_residual_window_coefficient_mode": str(
            policy.residual_window_coefficient_mode
        ),
        "direct_tail_residual_window_combine_mode": str(
            policy.residual_window_combine_mode
        ),
        "direct_tail_residual_window_interface_depth": int(
            policy.residual_window_interface_depth
        ),
        "direct_tail_residual_window_max_size": int(policy.residual_window_max_size),
        "direct_tail_true_window_max_windows": int(policy.true_window_max_windows),
        "direct_tail_true_window_x_radius": int(policy.true_window_x_radius),
        "direct_tail_true_window_ell_radius": int(policy.true_window_ell_radius),
        "direct_tail_true_window_max_mb": float(policy.true_window_max_mb),
        "direct_tail_true_window_regularization": float(
            policy.true_window_regularization
        ),
        "direct_tail_true_window_max_size": int(policy.true_window_max_size),
        "direct_tail_true_window_column_batch": int(policy.true_window_column_batch),
        "direct_tail_true_window_drop_tol": float(policy.true_window_drop_tol),
        "direct_tail_true_window_include_tail": bool(policy.true_window_include_tail),
        "direct_tail_true_window_damping": bool(policy.true_window_damping),
        "direct_tail_true_window_beta_max": float(policy.true_window_beta_max),
        "direct_tail_true_window_specs_env": str(true_window_specs_env),
        "direct_tail_true_window_specs": tuple(true_window_specs),
    }


def _direct_tail_true_active_rescue_driver_values(
    policy: DirectTailTrueActiveRescuePolicy,
) -> dict[str, object]:
    """Return default driver state derived from true-active rescue policy."""

    return {
        "direct_tail_true_active_block_requested": bool(
            policy.active_block_requested
        ),
        "direct_tail_true_active_residual_block_requested": bool(
            policy.active_residual_block_requested
        ),
        "direct_tail_true_active_submatrix_requested": bool(
            policy.active_submatrix_requested
        ),
        "direct_tail_true_active_submatrix_selected": False,
        "direct_tail_true_active_submatrix_metadata": None,
        "direct_tail_true_active_submatrix_error": None,
        "direct_tail_true_active_submatrix_residual_after": None,
        "direct_tail_true_active_block_selected": False,
        "direct_tail_true_active_block_metadata": None,
        "direct_tail_true_active_block_error": None,
        "direct_tail_true_active_block_residual_after": None,
        "direct_tail_true_active_residual_block_selected": False,
        "direct_tail_true_active_residual_block_metadata": None,
        "direct_tail_true_active_residual_block_error": None,
        "direct_tail_true_active_residual_block_residual_after": None,
        "direct_tail_true_active_column_cache_requested": bool(
            policy.active_column_cache_requested
        ),
        "direct_tail_true_active_column_cache_max_mb": float(
            policy.active_column_cache_max_mb
        ),
        "direct_tail_true_active_column_cache_metadata": None,
        "direct_tail_true_active_block_x_count": int(policy.active_block_x_count),
        "direct_tail_true_active_block_ell_count": int(policy.active_block_ell_count),
        "direct_tail_true_active_block_species_count": (
            None
            if policy.active_block_species_count is None
            else int(policy.active_block_species_count)
        ),
        "direct_tail_true_active_block_theta_stride": int(
            policy.active_block_theta_stride
        ),
        "direct_tail_true_active_block_zeta_stride": int(
            policy.active_block_zeta_stride
        ),
        "direct_tail_true_active_block_max_mb": float(policy.active_block_max_mb),
        "direct_tail_true_active_block_regularization": float(
            policy.active_block_regularization
        ),
        "direct_tail_true_active_block_max_size": int(policy.active_block_max_size),
        "direct_tail_true_active_block_column_batch": int(
            policy.active_block_column_batch
        ),
        "direct_tail_true_active_block_drop_tol": float(policy.active_block_drop_tol),
        "direct_tail_true_active_block_include_tail": bool(
            policy.active_block_include_tail
        ),
        "direct_tail_true_active_block_max_tail": int(policy.active_block_max_tail),
        "direct_tail_true_active_block_damping": bool(policy.active_block_damping),
        "direct_tail_true_active_block_beta_max": float(policy.active_block_beta_max),
        "direct_tail_true_active_residual_block_max_mb": float(
            policy.active_residual_block_max_mb
        ),
        "direct_tail_true_active_residual_block_regularization": float(
            policy.active_residual_block_regularization
        ),
        "direct_tail_true_active_residual_block_max_size": int(
            policy.active_residual_block_max_size
        ),
        "direct_tail_true_active_residual_block_column_batch": int(
            policy.active_residual_block_column_batch
        ),
        "direct_tail_true_active_residual_block_drop_tol": float(
            policy.active_residual_block_drop_tol
        ),
        "direct_tail_true_active_residual_block_include_tail": bool(
            policy.active_residual_block_include_tail
        ),
        "direct_tail_true_active_residual_block_max_tail": int(
            policy.active_residual_block_max_tail
        ),
        "direct_tail_true_active_residual_block_kinetic_only": bool(
            policy.active_residual_block_kinetic_only
        ),
        "direct_tail_true_active_residual_block_damping": bool(
            policy.active_residual_block_damping
        ),
        "direct_tail_true_active_residual_block_beta_max": float(
            policy.active_residual_block_beta_max
        ),
        "direct_tail_true_active_residual_block_min_improvement": float(
            policy.active_residual_block_min_improvement
        ),
        "direct_tail_true_active_residual_block_accept_base_improvement": bool(
            policy.active_residual_block_accept_base_improvement
        ),
        "direct_tail_true_active_residual_block_base_improvement_override_used": False,
        "direct_tail_true_active_submatrix_damping": bool(
            policy.active_submatrix_damping
        ),
        "direct_tail_true_active_submatrix_alpha_clip": float(
            policy.active_submatrix_alpha_clip
        ),
        "direct_tail_true_active_submatrix_min_improvement": float(
            policy.active_submatrix_min_improvement
        ),
    }


def _direct_tail_coupled_coarse_rescue_driver_values(
    policy: DirectTailCoupledCoarseRescuePolicy,
) -> dict[str, object]:
    """Return default driver state derived from coupled-coarse rescue policy."""

    return {
        "direct_tail_true_coupled_coarse_max_windows": int(policy.max_windows),
        "direct_tail_true_coupled_coarse_x_radius": int(policy.x_radius),
        "direct_tail_true_coupled_coarse_ell_radius": int(policy.ell_radius),
        "direct_tail_true_coupled_coarse_max_mb": float(policy.max_mb),
        "direct_tail_true_coupled_coarse_regularization": float(
            policy.regularization
        ),
        "direct_tail_true_coupled_coarse_max_size": int(policy.max_size),
        "direct_tail_true_coupled_coarse_column_batch": int(policy.column_batch),
        "direct_tail_true_coupled_coarse_drop_tol": float(policy.drop_tol),
        "direct_tail_true_coupled_coarse_low_lmax": int(policy.low_lmax),
        "direct_tail_true_coupled_coarse_profile_moment_count": int(
            policy.profile_moment_count
        ),
        "direct_tail_true_coupled_coarse_angular_lmax": int(policy.angular_lmax),
        "direct_tail_true_coupled_coarse_angular_mode_max": int(
            policy.angular_mode_max
        ),
        "direct_tail_true_coupled_coarse_max_tail_units": int(policy.max_tail_units),
        "direct_tail_true_coupled_coarse_include_tail": bool(policy.include_tail),
        "direct_tail_true_coupled_coarse_include_constraint_sources": bool(
            policy.include_constraint_sources
        ),
        "direct_tail_true_coupled_coarse_include_fsavg": bool(policy.include_fsavg),
        "direct_tail_true_coupled_coarse_include_window_residual": bool(
            policy.include_window_residual
        ),
        "direct_tail_true_coupled_coarse_include_profile_moments": bool(
            policy.include_profile_moments
        ),
        "direct_tail_true_coupled_coarse_include_angular_residual": bool(
            policy.include_angular_residual
        ),
        "direct_tail_true_coupled_coarse_include_angular_basis": bool(
            policy.include_angular_basis
        ),
        "direct_tail_true_coupled_coarse_include_preconditioned_loads": bool(
            policy.include_preconditioned_loads
        ),
        "direct_tail_true_coupled_coarse_preconditioned_load_max_columns": int(
            policy.preconditioned_load_max_columns
        ),
        "direct_tail_true_coupled_coarse_preconditioned_load_max_nnz": int(
            policy.preconditioned_load_max_nnz
        ),
        "direct_tail_true_coupled_coarse_preconditioned_load_drop_tol": float(
            policy.preconditioned_load_drop_tol
        ),
        "direct_tail_true_coupled_coarse_damping": bool(policy.damping),
        "direct_tail_true_coupled_coarse_beta_max": float(policy.beta_max),
        "direct_tail_true_coupled_coarse_accept_base_improvement": bool(
            policy.accept_base_improvement
        ),
        "direct_tail_true_coupled_coarse_base_improvement_override_used": False,
    }


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

    if bool(context.fortran_reduced_sparse_pc) and str(backend_setup.backend) == "xblock":
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
                "skipping monolithic sparse pattern for x-block backend",
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
            pattern=None,
            sparse_pattern_scope="fortran_reduced_xblock_deferred",
            pattern_build_s=0.0,
            summary=None,
            factor_policy=None,
        )

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


def build_sparse_pc_direct_tail_rescue_policy_setup(
    context: SparsePCDirectTailRescuePolicySetupContext,
) -> SparsePCDirectTailRescuePolicySetupResult:
    """Resolve direct-tail support-mode preflight and rescue policy defaults."""

    emit = context.emit
    factor_bundle_pc = context.factor_bundle_pc
    direct_tail_structured_pc_selected = bool(
        context.direct_tail_structured_pc_selected
    )
    direct_tail_structured_pc_reason = context.direct_tail_structured_pc_reason
    direct_tail_structured_pc_metadata = context.direct_tail_structured_pc_metadata
    direct_tail_support_mode_preflight_requested = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT",
        default=False,
    )
    direct_tail_support_mode_preflight_selected = False
    direct_tail_support_mode_preflight_metadata: dict[str, object] | None = None
    direct_tail_support_mode_preflight_error: str | None = None
    if bool(direct_tail_support_mode_preflight_requested):
        factor_kind_for_support = (
            str(getattr(factor_bundle_pc, "kind", "")).strip().lower().replace("-", "_")
        )
        support_preflight = run_direct_tail_support_mode_preflight(
            DirectTailSupportModePreflightContext(
                env=context.env,
                factor_kind=factor_kind_for_support,
                structured_pc_ready=bool(context.structured_pc_ready),
                operator_bundle=context.direct_tail_operator_bundle,
                layout=context.direct_tail_structured_layout,
                active_indices=context.direct_tail_structured_active_indices,
                max_nbytes=context.direct_tail_structured_max_nbytes,
                regularization=float(context.pc_reg),
                rhs=np.asarray(context.sparse_pc_rhs, dtype=np.float64),
                true_matvec=context.true_matvec,
                preconditioner_x=int(context.preconditioner_x),
                preconditioner_xi=int(context.preconditioner_xi),
                preconditioner_species=int(context.preconditioner_species),
                preconditioner_x_min_l=int(context.preconditioner_x_min_l),
                selector=context.support_mode_selector,
                factor_bundle=context.structured_factor_bundle_factory,
            )
        )
        direct_tail_support_mode_preflight_metadata = support_preflight.metadata
        direct_tail_support_mode_preflight_error = support_preflight.error
        if bool(support_preflight.selected):
            support_pc = support_preflight.preconditioner
            factor_bundle_pc = support_preflight.factor_bundle
            direct_tail_structured_pc_selected = True
            direct_tail_structured_pc_reason = str(
                getattr(support_pc, "reason", "support_mode_selected")
            )
            direct_tail_structured_pc_metadata = support_pc.to_dict()
            direct_tail_support_mode_preflight_selected = True
            if (
                emit is not None
                and isinstance(direct_tail_support_mode_preflight_metadata, dict)
            ):
                selected_candidate = direct_tail_support_mode_preflight_metadata.get(
                    "selected_candidate"
                )
                baseline_after = direct_tail_support_mode_preflight_metadata.get(
                    "baseline_residual_after"
                )
                best_after = direct_tail_support_mode_preflight_metadata.get(
                    "best_residual_after"
                )
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
            env=context.env,
            fortran_reduced_sparse_pc=bool(context.fortran_reduced_sparse_pc),
            structured_pc_ready=bool(context.structured_pc_ready),
            structured_pc_metadata=(
                direct_tail_structured_pc_metadata
                if isinstance(direct_tail_structured_pc_metadata, dict)
                else None
            ),
            sparse_pc_linear_size=int(context.sparse_pc_linear_size),
        )
    )
    direct_tail_residual_rescue_policy = resolve_direct_tail_residual_rescue_policy(
        context.env
    )
    direct_tail_true_window_specs_env = (
        _env_value(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_SPECS",
        ).strip()
        or _env_value(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_SPEC",
        ).strip()
    )
    direct_tail_true_window_specs: tuple[object, ...] = ()
    if direct_tail_true_window_specs_env:
        try:
            direct_tail_true_window_specs = context.parse_true_operator_window_specs(
                direct_tail_true_window_specs_env,
                layout=context.layout_from_operator(context.op),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            if emit is not None:
                emit(
                    1,
                    "fortran_reduced_direct_tail_true_window: "
                    f"skipped explicit specs ({type(exc).__name__}: {exc})",
                )

    direct_tail_true_active_rescue_policy = (
        resolve_direct_tail_true_active_rescue_policy(context.env)
    )
    direct_tail_true_coupled_coarse_policy = (
        resolve_direct_tail_coupled_coarse_rescue_policy(context.env)
    )
    direct_tail_true_coupled_coarse_explicit_requested = bool(
        direct_tail_residual_rescue_policy.true_coupled_coarse_explicit_requested
    )
    rescue_values = {
        **_direct_tail_residual_rescue_driver_values(
            direct_tail_residual_rescue_policy,
            true_window_specs_env=str(direct_tail_true_window_specs_env),
            true_window_specs=tuple(direct_tail_true_window_specs),
            true_coupled_coarse_explicit_requested=bool(
                direct_tail_true_coupled_coarse_explicit_requested
            ),
        ),
        **_direct_tail_true_active_rescue_driver_values(
            direct_tail_true_active_rescue_policy
        ),
        **_direct_tail_coupled_coarse_rescue_driver_values(
            direct_tail_true_coupled_coarse_policy
        ),
    }
    return SparsePCDirectTailRescuePolicySetupResult(
        factor_bundle_pc=factor_bundle_pc,
        direct_tail_structured_pc_selected=bool(direct_tail_structured_pc_selected),
        direct_tail_structured_pc_reason=direct_tail_structured_pc_reason,
        direct_tail_structured_pc_metadata=direct_tail_structured_pc_metadata,
        direct_tail_support_mode_preflight_requested=bool(
            direct_tail_support_mode_preflight_requested
        ),
        direct_tail_support_mode_preflight_selected=bool(
            direct_tail_support_mode_preflight_selected
        ),
        direct_tail_support_mode_preflight_metadata=(
            direct_tail_support_mode_preflight_metadata
        ),
        direct_tail_support_mode_preflight_error=(
            direct_tail_support_mode_preflight_error
        ),
        factor_preflight_policy=factor_preflight_policy,
        factor_preflight_enabled=bool(
            factor_preflight_policy.factor_preflight_enabled
        ),
        factor_preflight_required=bool(
            factor_preflight_policy.factor_preflight_required
        ),
        factor_preflight_seed_enabled=bool(
            factor_preflight_policy.factor_preflight_seed_enabled
        ),
        structured_pc_preflight_required_min_size=int(
            factor_preflight_policy.structured_pc_preflight_required_min_size
        ),
        direct_tail_structured_pc_requires_preflight=bool(
            factor_preflight_policy.direct_tail_structured_pc_requires_preflight
        ),
        direct_tail_structured_pc_kind_for_preflight=str(
            factor_preflight_policy.direct_tail_structured_pc_kind_for_preflight
        ),
        direct_tail_structured_pc_size_requires_preflight=bool(
            factor_preflight_policy.direct_tail_structured_pc_size_requires_preflight
        ),
        structured_pc_preflight_required=bool(
            factor_preflight_policy.structured_pc_preflight_required
        ),
        factor_preflight_max_target_ratio=float(
            factor_preflight_policy.factor_preflight_max_target_ratio
        ),
        factor_preflight_residual_before=None,
        factor_preflight_residual_after=None,
        factor_preflight_improvement_ratio=None,
        factor_preflight_target_ratio=None,
        factor_preflight_residual_diagnostics=None,
        factor_preflight_seed_used=False,
        factor_preflight_passed=None,
        factor_preflight_error=None,
        direct_tail_residual_rescue_policy=direct_tail_residual_rescue_policy,
        direct_tail_true_active_rescue_policy=direct_tail_true_active_rescue_policy,
        direct_tail_true_coupled_coarse_policy=direct_tail_true_coupled_coarse_policy,
        rescue_values=rescue_values,
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











_LOCAL_EXPORTS = (
    "FortranReducedXBlockBackendContext", "RequestedSparsePCGMRESBranchContext",
    "RHS1FullSparseRetryStageContext", "SparsePCAutoPreflightRetryStageContext",
    "SparsePCAutoPreflightRetryStageResult", "SparsePCDirectTailFactorSetupContext",
    "SparsePCDirectTailFactorSetupResult",
    "SparsePCDirectTailRescuePolicySetupContext",
    "SparsePCDirectTailRescuePolicySetupResult", "SparsePCFactorPreflightRunContext",
    "SparsePCFactorPreflightRunResult", "SparsePCGenericBranchSetupContext",
    "SparsePCGenericBranchSetupResult", "SparsePCResidualCandidateUpdateContext",
    "SparsePCResidualCandidateUpdateResult", "SparsePCResidualCorrectionStageContext",
    "SparsePCResidualCorrectionStageResult", "SparsePCTrueCoupledCoarseStageContext",
    "SparsePCTrueCoupledCoarseStageResult",
    "apply_sparse_pc_residual_candidate_update",
    "build_sparse_pc_direct_tail_factor_setup",
    "build_sparse_pc_direct_tail_rescue_policy_setup",
    "build_sparse_pc_generic_branch_setup",
    "run_sparse_pc_auto_preflight_retry_stage", "run_sparse_pc_factor_preflight",
    "run_sparse_pc_residual_correction_stage", "run_sparse_pc_true_coupled_coarse_stage",
    "run_rhs1_full_sparse_retry_stage", "run_xblock_sparse_pc_branch",
    "solve_fortran_reduced_xblock_backend", "try_run_requested_sparse_pc_gmres_branch",
)

_DIAGNOSTIC_EXPORTS = (
    "XBlockAssembledOperatorDiagnosticsContext", "XBlockSideProbeDiagnosticsContext",
    "XBlockSparsePCCoreDiagnosticsContext", "fp_xblock_global_correction_metadata",
    "fp_xblock_highx_residual_correction_metadata", "sparse_rescue_tail_metadata",
    "sparse_xblock_rescue_metadata", "xblock_assembled_operator_diagnostics",
    "xblock_coarse_correction_diagnostics", "xblock_device_krylov_diagnostics",
    "xblock_qi_deflated_preconditioner_diagnostics",
    "xblock_qi_seed_preconditioner_diagnostics", "xblock_side_probe_diagnostics",
    "xblock_sparse_pc_core_diagnostics",
    "xblock_sparse_pc_result_diagnostics_from_solve_state",
)

__all__ = tuple(dict.fromkeys((*_LOCAL_EXPORTS, *_DIAGNOSTIC_EXPORTS)))
