from __future__ import annotations

import importlib
from types import ModuleType


DOMAIN_PACKAGES = (
    "sfincs_jax.input",
    "sfincs_jax.physics",
    "sfincs_jax.discretization",
    "sfincs_jax.operators",
    "sfincs_jax.problems",
    "sfincs_jax.problems.profile_response",
    "sfincs_jax.problems.transport_matrix",
    "sfincs_jax.problems.transport_matrix.parallel",
    "sfincs_jax.solvers",
    "sfincs_jax.solvers.preconditioners",
    "sfincs_jax.solvers.preconditioners.pas",
    "sfincs_jax.solvers.preconditioners.full_fp",
    "sfincs_jax.solvers.preconditioners.qi",
    "sfincs_jax.solvers.preconditioners.schur",
    "sfincs_jax.solvers.preconditioners.domain_decomposition",
    "sfincs_jax.solvers.preconditioners.coarse_space",
    "sfincs_jax.solvers.preconditioners.xblock",
    "sfincs_jax.solvers.preconditioners.symbolic_sparse",
    "sfincs_jax.parallel",
    "sfincs_jax.outputs",
    "sfincs_jax.workflows",
    "sfincs_jax.validation",
    "sfincs_jax.benchmarks",
    "sfincs_jax.compat",
)

ACTIVE_PACKAGE_EXPORTS = {
    "sfincs_jax.problems": (
        "AmbipolarIteration",
        "AmbipolarProblem",
        "AmbipolarResult",
        "RadialCurrentDerivativeEvaluator",
        "RadialCurrentDerivativeResult",
        "RHSMode1RadialCurrentResponse",
        "SfincsJaxEvaluationRecord",
        "SfincsJaxRadialCurrentEvaluator",
        "brent_ambipolar_root",
        "dense_rhs1_vm_radial_current_linear_observable_system",
        "dphi_hat_dpsi_hat_er_derivative_from_namelist",
        "er_operator_tangent_from_dphi_hat_dpsi_hat_derivative",
        "finite_difference_radial_current_derivative",
        "implicit_linear_radial_current_derivative",
        "implicit_linear_radial_current_derivative_from_builder",
        "implicit_matrix_free_radial_current_derivative",
        "implicit_matrix_free_radial_current_derivative_from_builder",
        "matrix_free_radial_current_derivative_provider",
        "matrix_free_rhs1_vm_radial_current_linear_observable_system",
        "newton_ambipolar_root",
        "operator_tangent_from_centered_difference",
        "rhsmode1_radial_current_response_from_namelist",
        "safeguarded_newton_ambipolar_root",
        "solve_ambipolar_brent",
        "solve_ambipolar_newton",
        "solve_ambipolar_safeguarded_newton",
        "solve_rhsmode1_ambipolar_from_namelist",
        "solve_sfincs_jax_ambipolar_brent",
        "validate_fortran_v3_ambipolar_constraints",
    ),
    "sfincs_jax.solvers.preconditioners.pas": (
        "RHS1PasCompositeBuilders",
        "RHS1PasFamilyBuilders",
        "build_rhs1_pas_hybrid_preconditioner",
        "build_rhs1_pas_lite_preconditioner",
        "build_rhs1_pas_schur_preconditioner",
        "build_rhs1_pas_tokamak_theta_preconditioner",
        "build_rhs1_pas_tz_preconditioner",
        "build_rhs1_pas_xblock_ilu_preconditioner",
        "compose_preconditioners",
        "rhsmode1_pas_xblock_precond_cache_key",
    ),
    "sfincs_jax.solvers.preconditioners.full_fp": (
        "build_rhs1_block_preconditioner",
        "build_rhs1_block_preconditioner_xdiag",
        "build_rhs1_collision_preconditioner",
        "build_rhs1_species_block_preconditioner",
        "build_rhs1_species_xblock_preconditioner",
        "build_rhs1_structured_fblock_angular_jacobi_preconditioner",
        "build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner",
        "build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner",
        "build_rhs1_structured_fblock_fp_moment_schur_preconditioner",
        "build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner",
        "build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner",
        "build_rhs1_structured_fblock_jacobi_preconditioner",
        "build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner",
    ),
    "sfincs_jax.solvers.preconditioners.schur": (
        "ActiveNativeFieldSplitSparseCoarsePolicy",
        "ActiveNativeStackPolicy",
        "ActiveSparseCoarseResidualPolicy",
        "RHS1SchurPreconditionerBuilders",
        "RHS1StructuredFullCSRPreconditioner",
        "append_adaptive_residual_basis_csc",
        "build_active_native_xell_coarse_window_basis_csc",
        "build_block_schur_preconditioner",
        "build_coarse_residual_basis_csc",
        "build_diagonal_schur_preconditioner",
        "build_jacobi_preconditioner",
        "build_rhs1_schur_preconditioner",
        "build_x_xi_block_schur_preconditioner",
        "build_xi_block_schur_preconditioner",
        "canonical_schur_base_kind",
        "coarse_residual_config",
        "coarse_surface_mode_count",
        "coarse_surface_modes",
        "estimate_coarse_residual_nbytes",
        "estimate_x_xi_block_inverse_nbytes",
        "estimate_xblock_tz_low_l_factor_nbytes",
        "estimate_xi_block_inverse_nbytes",
        "estimate_zeta_block_inverse_nbytes",
        "resolve_active_native_field_split_sparse_coarse_policy",
        "resolve_active_native_stack_policy",
        "resolve_active_sparse_coarse_residual_policy",
        "safe_inverse_diagonal",
        "xblock_tz_low_l_config",
    ),
    "sfincs_jax.solvers.preconditioners.domain_decomposition": (
        "build_rhs1_theta_dd_preconditioner",
        "build_rhs1_theta_line_preconditioner",
        "build_rhs1_theta_line_xdiag_preconditioner",
        "build_rhs1_theta_schwarz_preconditioner",
        "build_rhs1_theta_zeta_preconditioner",
        "build_rhs1_zeta_dd_preconditioner",
        "build_rhs1_zeta_line_preconditioner",
        "build_rhs1_zeta_schwarz_preconditioner",
    ),
    "sfincs_jax.solvers.preconditioners.xblock": (
        "active_positions_for_full_indices",
        "assemble_rhsmode1_fp_xblock_tz_sparse_matrix",
        "assemble_selected_theta_tz_operator",
        "assemble_selected_zeta_tz_operator",
        "build_active_fortran_v3_reduced_native_stack_preconditioner",
        "build_active_projected_bounded_native_stack_preconditioner",
        "build_active_projected_global_field_split_schur_preconditioner",
        "build_active_projected_multiline_field_split_base_preconditioner",
        "build_active_projected_angular_line_preconditioner",
        "build_active_projected_diagonal_schur_preconditioner",
        "build_active_projected_native_indexed_schwarz_preconditioner",
        "build_active_projected_overlap_schwarz_preconditioner",
        "build_active_projected_xell_kinetic_line_preconditioner",
        "build_active_projected_xblock_preconditioner",
        "build_rhs1_sxblock_tz_preconditioner",
        "build_rhs1_sxblock_tz_sparse_host_preconditioner",
        "build_rhs1_xmg_preconditioner",
        "build_rhs1_xupwind_preconditioner",
        "build_rhs1_xblock_tz_lmax_preconditioner",
        "build_rhs1_xblock_tz_preconditioner",
        "build_rhs1_xblock_tz_sparse_preconditioner",
        "build_native_xell_kinetic_preconditioner",
        "build_native_xell_tail_schur_preconditioner",
        "build_xblock_tz_low_l_coarse_residual_preconditioner",
        "build_xblock_tz_low_l_schur_preconditioner",
        "compute_rhs1_sxblock_tz_sparse_host_seed",
        "get_rhsmode1_fp_xblock_assembled_host_cache",
        "rhsmode1_fp_xblock_assembled_host_allowed",
        "rhsmode1_fp_xblock_species_decoupled_for_host_assembly",
        "rhsmode1_fp_xblock_tz_sparse_diagonal",
        "rhsmode1_host_factor_probe_ok",
        "rhsmode1_precond_cache_key",
        "rhsmode1_xblock_sparse_lu_default_max",
        "safe_inverse_diagonal_np",
        "xblock_tz_low_l_indices",
    ),
    "sfincs_jax.solvers.preconditioners.symbolic_sparse": (
        "RHS1FullSystemMatrixFreeOperatorAdapter",
        "active_fortran_v3_reduced_preconditioner_matrix",
        "build_active_filtered_sparse_factor_preconditioner",
        "build_active_global_sparse_factor_preconditioner",
        "build_active_scaled_sparse_factor_preconditioner",
        "build_sparse_ilu_from_matvec",
        "build_active_fortran_v3_reduced_sparse_factor_preconditioner",
        "estimate_spilu_factor_nbytes",
        "factorize_sparse_matrix_csr_host",
        "parse_active_fortran_v3_support_mode_candidates",
        "select_active_fortran_v3_reduced_support_mode_preconditioner",
        "sparse_equilibration_scale",
        "sparse_lu_factor_nbytes",
    ),
    "sfincs_jax.outputs": (
        "conversion_factors_to_from_dpsi_hat",
        "decode_if_bytes",
        "fortran_h5_layout",
        "output_file_format",
        "read_sfincs_h5",
        "read_sfincs_output_file",
        "to_numpy_for_h5",
        "transport_solver_diagnostic_arrays",
        "write_transport_h5_streaming",
        "write_sfincs_h5",
        "write_sfincs_netcdf",
        "write_sfincs_npz",
        "write_sfincs_output_file",
    ),
}

ACTIVE_MODULE_EXPORTS = {
    "sfincs_jax.sensitivity": (
        "FluxFn",
        "JvpVjpDotProductResult",
        "LinearObservableBuilder",
        "LinearObservableDerivativeResult",
        "LinearOperatorApply",
        "LinearObservableSystem",
        "MatrixFreeLinearObservableBuilder",
        "MatrixFreeLinearObservableSystem",
        "StateObservableFn",
        "VectorSolver",
        "adjoint_dot_product_check",
        "evaluate_linear_observable",
        "evaluate_matrix_free_linear_observable",
        "fortran_v3_adjoint_sensitivity_output_fields",
        "fortran_v3_adjoint_sensitivity_output_ranks",
        "implicit_linear_observable_derivative",
        "implicit_linear_observable_derivative_from_builder",
        "implicit_matrix_free_linear_observable_derivative",
        "implicit_matrix_free_linear_observable_derivative_from_builder",
        "jvp_flux",
        "probe_linear_observable_vector",
        "validate_fortran_v3_adjoint_sensitivity_constraints",
        "validate_fortran_v3_adjoint_sensitivity_output_surface",
        "vjp_flux",
    ),
}

LEGACY_MODULES_THAT_KEEP_THEIR_IMPORT_PATHS = (
    "sfincs_jax.input_compat",
    "sfincs_jax.namelist",
    "sfincs_jax.geometry",
    "sfincs_jax.io",
    "sfincs_jax.solver",
    "sfincs_jax.v3_driver",
)

RESERVED_MODULE_NAMES_UNTIL_MIGRATION = (
    "sfincs_jax.geometry",
    "sfincs_jax.io",
)

TRANSPORT_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.problems.transport_matrix.diagnostics",
        "sfincs_jax.problems.transport_matrix.diagnostics",
        "v3_transport_matrix_from_state_vectors",
    ),
    (
        "sfincs_jax.problems.transport_matrix.setup",
        "sfincs_jax.problems.transport_matrix.setup",
        "resolve_transport_which_rhs_setup",
    ),
    (
        "sfincs_jax.problems.transport_matrix.active_dense",
        "sfincs_jax.problems.transport_matrix.active_dense",
        "resolve_transport_active_dense_setup",
    ),
    (
        "sfincs_jax.problems.transport_matrix.loop",
        "sfincs_jax.problems.transport_matrix.loop",
        "resolve_transport_recycle_k",
    ),
    (
        "sfincs_jax.problems.transport_matrix.finalize",
        "sfincs_jax.problems.transport_matrix.finalize",
        "finalize_full_transport_rhs",
    ),
    (
        "sfincs_jax.problems.transport_matrix.finalize",
        "sfincs_jax.problems.transport_matrix.finalize",
        "V3TransportMatrixSolveResult",
    ),
    (
        "sfincs_jax.problems.transport_matrix.streaming_outputs",
        "sfincs_jax.problems.transport_matrix.streaming_outputs",
        "TransportStreamingOutputAccumulator",
    ),
    (
        "sfincs_jax.problems.transport_matrix.postsolve_diagnostics",
        "sfincs_jax.problems.transport_matrix.postsolve_diagnostics",
        "compute_transport_postsolve_diagnostics",
    ),
    (
        "sfincs_jax.problems.transport_matrix.policies",
        "sfincs_jax.problems.transport_matrix.policies",
        "transport_dense_backend_allowed",
    ),
    (
        "sfincs_jax.problems.transport_matrix.solve_policy",
        "sfincs_jax.problems.transport_matrix.solve_policy",
        "resolve_transport_initial_solve_policy",
    ),
    (
        "sfincs_jax.problems.transport_matrix.solve",
        "sfincs_jax.problems.transport_matrix.solve",
        "solve_transport_linear_with_residual",
    ),
    (
        "sfincs_jax.problems.transport_matrix.host_gmres",
        "sfincs_jax.problems.transport_matrix.host_gmres",
        "transport_host_gmres_solve",
    ),
    (
        "sfincs_jax.problems.transport_matrix.handoff_policy",
        "sfincs_jax.problems.transport_matrix.handoff_policy",
        "transport_polish_config_from_env",
    ),
    (
        "sfincs_jax.problems.transport_matrix.residual_quality",
        "sfincs_jax.problems.transport_matrix.residual_quality",
        "transport_residual_gate_failure",
    ),
    (
        "sfincs_jax.problems.transport_matrix.iteration_stats",
        "sfincs_jax.problems.transport_matrix.iteration_stats",
        "emit_transport_ksp_iteration_stats",
    ),
    (
        "sfincs_jax.problems.transport_matrix.dense_lu",
        "sfincs_jax.problems.transport_matrix.dense_lu",
        "dense_solver_for_matvec",
    ),
    (
        "sfincs_jax.problems.transport_matrix.dense_batch",
        "sfincs_jax.problems.transport_matrix.dense_batch",
        "solve_transport_dense_batch",
    ),
    (
        "sfincs_jax.problems.transport_matrix.active_factor",
        "sfincs_jax.problems.transport_matrix.active_factor",
        "build_active_block_schur_factor",
    ),
    (
        "sfincs_jax.problems.transport_matrix.sparse_direct_solve",
        "sfincs_jax.problems.transport_matrix.sparse_direct_solve",
        "transport_sparse_direct_solve",
    ),
    (
        "sfincs_jax.problems.transport_matrix.preconditioner_dispatch",
        "sfincs_jax.problems.transport_matrix.preconditioner_dispatch",
        "build_transport_preconditioner_from_kind",
    ),
    (
        "sfincs_jax.problems.transport_matrix.direct_pmat",
        "sfincs_jax.problems.transport_matrix.direct_pmat",
        "_try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle",
    ),
    (
        "sfincs_jax.problems.transport_matrix.direct_block_schur",
        "sfincs_jax.problems.transport_matrix.direct_block_schur",
        "build_transport_fp_direct_active_block_schur_preconditioner",
    ),
    (
        "sfincs_jax.problems.transport_matrix.fortran_reduced_lu",
        "sfincs_jax.problems.transport_matrix.fortran_reduced_lu",
        "build_transport_fp_fortran_reduced_lu_preconditioner",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.policy",
        "sfincs_jax.problems.transport_matrix.parallel.policy",
        "transport_parallel_backend",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "merge_transport_parallel_results",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "solve_transport_parallel_payload",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "TransportParallelPoolCache",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "run_transport_parallel_payloads",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "maybe_run_transport_parallel_solve",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "sfincs_jax.problems.transport_matrix.parallel.runtime",
        "validate_transport_worker_result_payload",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.sharding",
        "sfincs_jax.problems.transport_matrix.parallel.sharding",
        "plan_single_case_operator_coarse_reuse",
    ),
    (
        "sfincs_jax.problems.transport_matrix.parallel.worker",
        "sfincs_jax.problems.transport_matrix.parallel.worker",
        "main",
    ),
)

PRECONDITIONER_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.solvers.preconditioners.pas.xblock_ilu",
        "sfincs_jax.solvers.preconditioners.pas.xblock_ilu",
        "build_rhs1_pas_xblock_ilu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.xblock.tz_sparse",
        "sfincs_jax.solvers.preconditioners.xblock.tz_sparse",
        "build_rhs1_xblock_tz_sparse_preconditioner",
    ),
)

PRECONDITIONER_IMPLEMENTATION_IMPORTS = (
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_block_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_collision_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_local_geom_line_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_structured_fblock_lu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_tzfft_line_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_tzfft_line_schur_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_tzfft_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_xblock_tz_lu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_sxblock_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_tzfft_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioners.transport_matrix",
        "build_rhsmode23_xmg_preconditioner",
    ),
)

PROFILE_RESPONSE_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_pas_fast_accept",
    ),
    (
        "sfincs_jax.problems.profile_response.active_dof",
        "sfincs_jax.problems.profile_response.active_dof",
        "resolve_rhs1_active_dof_mode",
    ),
    (
        "sfincs_jax.problems.profile_response.active_dof",
        "sfincs_jax.problems.profile_response.active_dof",
        "reduce_full_with_indices",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_constraint0_sparse_first",
    ),
    (
        "sfincs_jax.problems.profile_response.handoff",
        "sfincs_jax.problems.profile_response.handoff",
        "rhs1_accept_candidate",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_fast_post_xblock_polish_allowed",
    ),
    (
        "sfincs_jax.problems.profile_response.residual",
        "sfincs_jax.problems.profile_response.residual",
        "residual_target",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_sparse_exact_lu_requested",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_polish_enabled",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_sparse_kind_use",
    ),
    (
        "sfincs_jax.problems.profile_response.policies",
        "sfincs_jax.problems.profile_response.policies",
        "rhs1_stage2_trigger",
    ),
    (
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "build_rhs1_xblock_correction_metadata",
    ),
    (
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "V3LinearSolveResult",
    ),
    (
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "V3NewtonKrylovResult",
    ),
    (
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "sfincs_jax.problems.profile_response.solver_diagnostics",
        "v3_linear_solve_result_from_payload",
    ),
    (
        "sfincs_jax.operators.profile_response.sparse_pattern",
        "sfincs_jax.operators.profile_response.sparse_pattern",
        "v3_full_system_conservative_sparsity_pattern",
    ),
    (
        "sfincs_jax.operators.profile_response.sparse_pattern",
        "sfincs_jax.operators.profile_response.sparse_pattern",
        "v3_full_system_fortran_reduced_preconditioner_sparsity_pattern",
    ),
    (
        "sfincs_jax.operators.profile_response.fblock",
        "sfincs_jax.operators.profile_response.fblock",
        "V3FBlockOperator",
    ),
    (
        "sfincs_jax.operators.profile_response.fblock",
        "sfincs_jax.operators.profile_response.fblock",
        "fblock_operator_from_namelist",
    ),
    (
        "sfincs_jax.operators.profile_response.fblock",
        "sfincs_jax.operators.profile_response.fblock",
        "matvec_v3_fblock_flat",
    ),
    (
        "sfincs_jax.operators.profile_response.system",
        "sfincs_jax.operators.profile_response.system",
        "V3FullSystemOperator",
    ),
    (
        "sfincs_jax.operators.profile_response.system",
        "sfincs_jax.operators.profile_response.system",
        "full_system_operator_from_namelist",
    ),
    (
        "sfincs_jax.operators.profile_response.system",
        "sfincs_jax.operators.profile_response.system",
        "apply_v3_full_system_operator_cached",
    ),
    (
        "sfincs_jax.discretization.v3",
        "sfincs_jax.discretization.v3",
        "V3Grids",
    ),
    (
        "sfincs_jax.discretization.v3",
        "sfincs_jax.discretization.v3",
        "grids_from_namelist",
    ),
    (
        "sfincs_jax.discretization.v3",
        "sfincs_jax.discretization.v3",
        "geometry_from_namelist",
    ),
    (
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "auto_rhs1_full_strong_kind",
    ),
    (
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "rhs1_resolved_strong_preconditioner_control",
    ),
    (
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "sfincs_jax.problems.profile_response.preconditioner_build",
        "requested_rhs1_strong_preconditioner_kind",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "FortranReducedXBlockBackendContext",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "SparsePCDirectTailFactorSetupContext",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "SparsePCDirectTailRescuePolicySetupContext",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "SparsePCGenericBranchSetupContext",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "build_sparse_pc_direct_tail_factor_setup",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "build_sparse_pc_direct_tail_rescue_policy_setup",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "build_sparse_pc_generic_branch_setup",
    ),
    (
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "sfincs_jax.problems.profile_response.sparse.handoff",
        "solve_fortran_reduced_xblock_backend",
    ),
)


def _import_module(name: str) -> ModuleType:
    return importlib.import_module(name)


def test_domain_packages_are_importable_with_expected_facades() -> None:
    """Domain packages are importable and expose only intentional facades."""

    for module_name in DOMAIN_PACKAGES:
        module = _import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name
        assert hasattr(module, "__path__"), module_name
        expected_exports = ACTIVE_PACKAGE_EXPORTS.get(module_name, ())
        assert module.__all__ == expected_exports, module_name
        for export_name in expected_exports:
            assert hasattr(module, export_name), f"{module_name}.{export_name}"


def test_active_modules_are_importable_with_expected_exports() -> None:
    """Domain-level modules can expose small public facades without becoming packages."""

    for module_name, expected_exports in ACTIVE_MODULE_EXPORTS.items():
        module = _import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name
        assert not hasattr(module, "__path__"), module_name
        assert module.__all__ == expected_exports, module_name
        for export_name in expected_exports:
            assert hasattr(module, export_name), f"{module_name}.{export_name}"


def test_existing_legacy_modules_keep_their_import_paths() -> None:
    """The package skeleton must not break current public/internal imports."""

    for module_name in LEGACY_MODULES_THAT_KEEP_THEIR_IMPORT_PATHS:
        module = _import_module(module_name)
        if module_name == "sfincs_jax.v3_driver":
            assert module.__name__ == "sfincs_jax.problems.profile_response.solve"
            continue
        assert module.__name__ == module_name


def test_module_names_reserved_for_later_package_migration_still_load_as_modules() -> None:
    """Avoid silently shadowing large legacy modules during Phase A."""

    for module_name in RESERVED_MODULE_NAMES_UNTIL_MIGRATION:
        module = _import_module(module_name)
        assert not hasattr(module, "__path__"), module_name
        assert module.__file__ is not None
        assert module.__file__.endswith(".py"), module.__file__


def test_transport_matrix_package_moves_preserve_legacy_imports() -> None:
    """Moved implementation modules must remain reachable through old names."""

    for legacy_name, new_name, public_name in TRANSPORT_COMPATIBILITY_IMPORTS:
        legacy_module = _import_module(legacy_name)
        new_module = _import_module(new_name)
        assert getattr(legacy_module, public_name) is getattr(new_module, public_name)


def test_preconditioner_package_moves_preserve_legacy_imports() -> None:
    """Moved preconditioner modules must remain reachable through old names."""

    for legacy_name, new_name, public_name in PRECONDITIONER_COMPATIBILITY_IMPORTS:
        legacy_module = _import_module(legacy_name)
        new_module = _import_module(new_name)
        assert legacy_module is new_module
        assert getattr(legacy_module, public_name) is getattr(new_module, public_name)


def test_preconditioner_implementation_modules_expose_expected_builders() -> None:
    """Implementation modules should keep the moved numerical builders importable."""

    for module_name, public_name in PRECONDITIONER_IMPLEMENTATION_IMPORTS:
        module = _import_module(module_name)
        assert public_name in getattr(module, "__all__", ()), module_name
        assert hasattr(module, public_name), f"{module_name}.{public_name}"


def test_profile_response_package_moves_preserve_legacy_imports() -> None:
    """Moved profile-response modules must remain reachable through old names."""

    for legacy_name, new_name, public_name in PROFILE_RESPONSE_COMPATIBILITY_IMPORTS:
        legacy_module = _import_module(legacy_name)
        new_module = _import_module(new_name)
        assert legacy_module is new_module
        assert getattr(legacy_module, public_name) is getattr(new_module, public_name)
        if hasattr(legacy_module, "__all__"):
            assert public_name in legacy_module.__all__
        if hasattr(new_module, "__all__"):
            assert public_name in new_module.__all__
