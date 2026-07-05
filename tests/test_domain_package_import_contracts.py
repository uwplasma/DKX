from __future__ import annotations

import importlib
from pathlib import Path
import re
from types import ModuleType


DOMAIN_PACKAGES = (
    "sfincs_jax.physics",
    "sfincs_jax.discretization",
    "sfincs_jax.geometry",
    "sfincs_jax.operators",
    "sfincs_jax.problems",
    "sfincs_jax.solvers",
    "sfincs_jax.outputs",
    "sfincs_jax.workflows",
    "sfincs_jax.validation",
)

ACTIVE_PACKAGE_EXPORTS = {
    "sfincs_jax.workflows": (
        "mapped_xgrid",
        "optimization",
    ),
    "sfincs_jax.geometry": (
        "BoozerGeometry",
        "boozer_geometry_from_bc_file",
        "boozer_geometry_scheme1",
        "boozer_geometry_scheme2",
        "boozer_geometry_scheme4",
    ),
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
    "sfincs_jax.solvers.preconditioner_domain_decomposition": (
        "build_rhs1_theta_dd_preconditioner",
        "build_rhs1_theta_line_preconditioner",
        "build_rhs1_theta_line_xdiag_preconditioner",
        "build_rhs1_theta_schwarz_preconditioner",
        "build_rhs1_theta_zeta_preconditioner",
        "build_rhs1_zeta_dd_preconditioner",
        "build_rhs1_zeta_line_preconditioner",
        "build_rhs1_zeta_schwarz_preconditioner",
    ),
    "sfincs_jax.outputs": (
        "ExportFConfig",
        "TransportStreamingOutputAccumulator",
        "conversion_factors_to_from_dpsi_hat",
        "decode_if_bytes",
        "fortran_h5_layout",
        "localize_equilibrium_file_in_place",
        "output_file_format",
        "read_sfincs_h5",
        "read_sfincs_output_file",
        "sfincs_jax_output_dict",
        "to_numpy_for_h5",
        "transport_solver_diagnostic_arrays",
        "write_transport_h5_streaming",
        "write_sfincs_h5",
        "write_sfincs_jax_output_h5",
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
    "sfincs_jax.validation.qi_device": (
        "DEFAULT_GPU_CPU_ROOT_ATOL",
        "DEFAULT_GPU_FORTRAN_ROOT_ATOL",
        "QIDeviceArtifactCheck",
        "QILadderPromotionResult",
        "QIRunEvidence",
        "check_qi_device_artifact_file",
        "check_qi_device_artifact_files",
        "evaluate_qi_production_ladder_promotion",
        "evaluate_qi_res15_gpu_campaign",
        "evaluate_qi_res15_gpu_campaign_files",
        "load_json_object",
        "qi_device_artifact_errors",
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
    "sfincs_jax.io",
)

MOVED_ROOT_MODULE_OWNERS = {
    "sfincs_jax.physics.classical_transport": (
        "classical_flux_v3",
    ),
    "sfincs_jax.physics.collisions": (
        "FokkerPlanckV3Operator",
        "FokkerPlanckV3Phi1Operator",
        "PitchAngleScatteringV3Operator",
        "apply_pitch_angle_scattering_v3",
    ),
    "sfincs_jax.problems.profile_phi1_newton": (
        "build_phi1_newton_preconditioner",
        "phi1_frozen_jacobian_policy",
        "phi1_gmres_restart",
        "phi1_line_search_policy",
        "phi1_use_active_dof_mode",
        "solve_phi1_newton_linear_step",
    ),
    "sfincs_jax.solvers.explicit_sparse": (
        "host_direct_solve_with_refinement",
        "host_sparse_direct_polish",
        "host_sparse_direct_solve_with_refinement",
    ),
    "sfincs_jax.solvers.preconditioning": (
        "project_constraint_scheme1_nullspace_solution",
        "project_constraint_scheme1_nullspace_solution_with_residual",
    ),
    "sfincs_jax.solvers.preconditioner_pas_policy": (
        "AdaptivePassSmootherResult",
        "ConstrainedPASBranchRecord",
        "PasSmootherConfig",
        "adaptive_pas_smoother",
        "adaptive_pas_smoother_allowed",
        "summarize_constrained_pas_branches",
    ),
    "sfincs_jax.operators.profile_collisionless": (
        "CollisionlessV3Operator",
        "apply_collisionless_v3",
    ),
    "sfincs_jax.operators.profile_electric_field": (
        "ErXiDotV3Operator",
        "ErXDotV3Operator",
        "apply_er_xidot_v3",
        "apply_er_xdot_v3",
    ),
    "sfincs_jax.operators.profile_exb": (
        "ExBThetaV3Operator",
        "ExBZetaV3Operator",
        "apply_exb_theta_v3",
        "apply_exb_zeta_v3",
    ),
    "sfincs_jax.operators.profile_system": (
        "V3FBlockLinearSystem",
        "V3FullLinearSystem",
    ),
    "sfincs_jax.operators.profile_magnetic_drifts": (
        "MagneticDriftThetaV3Operator",
        "MagneticDriftXiDotV3Operator",
        "MagneticDriftZetaV3Operator",
        "apply_magnetic_drift_theta_v3",
    ),
    "sfincs_jax.discretization.adaptive_maps": (
        "AffineXMap",
        "MappedXGrid",
        "RationalTailXMap",
        "make_reference_eta_grid",
    ),
    "sfincs_jax.discretization.v3": (
        "V3Grids",
        "V3Indexing",
        "geometry_from_namelist",
        "grids_from_namelist",
    ),
    "sfincs_jax.discretization.periodic_stencil": (
        "apply_periodic_stencil_roll",
        "extract_sparse_row_stencil",
    ),
    "sfincs_jax.discretization.structured_velocity": (
        "BlockTridiagonalFactorization",
        "factor_block_tridiagonal",
        "solve_block_tridiagonal",
    ),
    "sfincs_jax.discretization.xgrid": (
        "XGrid",
        "make_x_grid",
        "make_x_polynomial_diff_matrices",
    ),
    "sfincs_jax.geometry.boozer": (
        "read_boozer_bc_header",
        "read_boozer_bc_bracketing_surfaces",
        "selected_r_n_from_bc",
    ),
    "sfincs_jax.geometry.jax_adapters": (
        "geometry_proxy_workflow_contract",
        "vmec_wout_from_wout_like",
    ),
    "sfincs_jax.geometry.vmec": (
        "vmec_geometry_from_wout",
        "vmec_geometry_from_wout_file",
    ),
    "sfincs_jax.geometry.vmec_wout": (
        "VmecWout",
        "read_vmec_wout",
        "vmec_interpolation",
    ),
    "sfincs_jax.validation.data_fetch": (
        "ensure_external_equilibrium_data",
        "external_data_manifest",
        "resolve_external_equilibrium",
    ),
    "sfincs_jax.workflows.scans": (
        "find_upstream_utils_dir",
        "ScanResult",
        "linspace_including_endpoints",
        "run_er_scan",
        "run_upstream_util",
    ),
    "sfincs_jax.workflows.optimization": (
        "AmbipolarRoot",
        "build_candidate_scan_plan",
        "evaluate_sfincs_scan_promotion",
        "qa_proxy_neoclassical_objective",
    ),
}

DELETED_ROOT_ALIASES = (
    "sfincs_jax.adaptive_maps",
    "sfincs_jax.boozer_bc",
    "sfincs_jax.data_fetch",
    "sfincs_jax.indices",
    "sfincs_jax.jax_geometry_adapters",
    "sfincs_jax.periodic_stencil",
    "sfincs_jax.postprocess_upstream",
    "sfincs_jax.scans",
    "sfincs_jax.structured_velocity",
    "sfincs_jax.vmec_geometry",
    "sfincs_jax.vmec_wout",
    "sfincs_jax.xgrid",
    "sfincs_jax.collisionless",
    "sfincs_jax.collisionless_er",
    "sfincs_jax.collisionless_exb",
    "sfincs_jax.magnetic_drifts",
    "sfincs_jax.residual",
    "sfincs_jax.classical_transport",
    "sfincs_jax.collisions",
    "sfincs_jax.constrained_pas_branch",
    "sfincs_jax.constraint_projection",
    "sfincs_jax.host_refinement",
    "sfincs_jax.pas_smoother",
    "sfincs_jax.phi1_newton_linear",
    "sfincs_jax.phi1_newton_policy",
)

ROOT_MODULE_CLASSIFICATIONS = {
    "__init__.py": "public package facade",
    "__main__.py": "public entry point",
    "ambipolar.py": "public physics API",
    "api.py": "public API",
    "cli.py": "public entry point",
    "compare.py": "public validation API",
    "diagnostics.py": "stable physics kernel",
    "grids.py": "public discretization API",
    "input_compat.py": "public compatibility API",
    "io.py": "compatibility facade",
    "namelist.py": "public input API",
    "paths.py": "stable support utility",
    "plotting.py": "public plotting API",
    "profiling.py": "stable support utility",
    "sensitivity.py": "public differentiation API",
    "solver.py": "stable solver kernel",
    "v3_driver.py": "compatibility shim",
}

ROOT_MODULE_CLOSURE_MANIFEST = {
    "__init__.py": ("package root public facade", "keep at root"),
    "__main__.py": ("package root CLI entry point", "keep at root"),
    "ambipolar.py": ("problems.ambipolar via public API facade", "keep root shim until public docs/examples migrate"),
    "api.py": ("package root public API", "keep at root"),
    "cli.py": ("package root CLI entry point", "keep at root"),
    "compare.py": (
        "package root public comparison API",
        "keep at root; it owns user-facing SFINCS comparison helpers and strict HDF5 output parity",
    ),
    "diagnostics.py": ("physics/output diagnostics owner", "defer until diagnostics API split is explicit"),
    "grids.py": ("discretization public grid owner", "keep root public helper until discretization package exports are documented"),
    "input_compat.py": ("input compatibility owner", "keep root public compatibility shim until input package exports cover callers"),
    "io.py": ("outputs writer/formats owners", "keep tiny root facade until public imports migrate"),
    "namelist.py": ("input namelist owner", "keep root public parser until input package exports are documented"),
    "paths.py": ("package root path support utility", "keep at root unless a support package is introduced with broad import rewrite"),
    "plotting.py": ("outputs/plotting public helper", "keep root public helper unless API replacement is documented"),
    "profiling.py": ("solvers/validation profiling support", "defer until profiling API boundary is explicit"),
    "sensitivity.py": ("package root differentiation API", "keep at root"),
    "solver.py": ("solvers public contracts owner", "keep root shim until solvers exports cover public contracts"),
    "v3_driver.py": (
        "compatibility shim to problem owners",
        "keep tiny shim until the compatibility deprecation window closes; public examples and scripts should not import it",
    ),
}

TRANSPORT_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.problems.transport_diagnostics",
        "sfincs_jax.problems.transport_diagnostics",
        "v3_transport_matrix_from_state_vectors",
    ),
    (
        "sfincs_jax.problems.transport_setup",
        "sfincs_jax.problems.transport_setup",
        "resolve_transport_which_rhs_setup",
    ),
    (
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_linear_system",
        "resolve_transport_active_dense_setup",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "resolve_transport_recycle_k",
    ),
    (
        "sfincs_jax.problems.transport_finalize",
        "sfincs_jax.problems.transport_finalize",
        "finalize_full_transport_rhs",
    ),
    (
        "sfincs_jax.problems.transport_finalize",
        "sfincs_jax.problems.transport_finalize",
        "V3TransportMatrixSolveResult",
    ),
    (
        "sfincs_jax.outputs.transport",
        "sfincs_jax.outputs.transport",
        "TransportStreamingOutputAccumulator",
    ),
    (
        "sfincs_jax.problems.transport_finalize",
        "sfincs_jax.problems.transport_finalize",
        "compute_transport_postsolve_diagnostics",
    ),
    (
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_policies",
        "transport_dense_backend_allowed",
    ),
    (
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_policies",
        "resolve_transport_initial_solve_policy",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "solve_transport_linear_with_residual",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "transport_host_gmres_solve",
    ),
    (
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_policies",
        "transport_polish_config_from_env",
    ),
    (
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_policies",
        "transport_residual_gate_failure",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "emit_transport_ksp_iteration_stats",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "dense_solver_for_matvec",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "solve_transport_dense_batch",
    ),
    (
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_linear_system",
        "build_active_block_schur_factor",
    ),
    (
        "sfincs_jax.problems.transport_solve",
        "sfincs_jax.problems.transport_solve",
        "transport_sparse_direct_solve",
    ),
    (
        "sfincs_jax.problems.transport_policies",
        "sfincs_jax.problems.transport_policies",
        "build_transport_preconditioner_from_kind",
    ),
    (
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_linear_system",
        "_try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle",
    ),
    (
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_linear_system",
        "build_transport_fp_direct_active_block_schur_preconditioner",
    ),
    (
        "sfincs_jax.problems.transport_linear_system",
        "sfincs_jax.problems.transport_linear_system",
        "build_transport_fp_fortran_reduced_lu_preconditioner",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "transport_parallel_backend",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "merge_transport_parallel_results",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "solve_transport_parallel_payload",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "TransportParallelPoolCache",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "run_transport_parallel_payloads",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "maybe_run_transport_parallel_solve",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "validate_transport_worker_result_payload",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "plan_single_case_operator_coarse_reuse",
    ),
    (
        "sfincs_jax.problems.transport_parallel_runtime",
        "sfincs_jax.problems.transport_parallel_runtime",
        "main",
    ),
)

PRECONDITIONER_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.solvers.preconditioner_pas_xblock_ilu",
        "sfincs_jax.solvers.preconditioner_pas_xblock_ilu",
        "build_rhs1_pas_xblock_ilu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_xblock_tz_sparse",
        "sfincs_jax.solvers.preconditioner_xblock_tz_sparse",
        "build_rhs1_xblock_tz_sparse_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_full_fp_kinetic",
        "sfincs_jax.solvers.preconditioner_full_fp_kinetic",
        "build_rhs1_collision_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_qi_basis",
        "sfincs_jax.solvers.preconditioner_qi_basis",
        "build_rhs1_qi_coarse_basis",
    ),
    (
        "sfincs_jax.solvers.preconditioner_qi_corrections",
        "sfincs_jax.solvers.preconditioner_qi_corrections",
        "build_rhs1_qi_two_level_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_qi_device",
        "sfincs_jax.solvers.preconditioner_qi_device",
        "setup_rhs1_qi_device_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_schur_profile",
        "sfincs_jax.solvers.preconditioner_schur_profile",
        "build_rhs1_schur_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_symbolic_profile",
        "sfincs_jax.solvers.preconditioner_symbolic_profile",
        "build_active_fortran_v3_reduced_sparse_factor_preconditioner",
    ),
)

PRECONDITIONER_IMPLEMENTATION_IMPORTS = (
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_block_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_collision_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_local_geom_line_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_structured_fblock_lu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_tzfft_line_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_tzfft_line_schur_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_tzfft_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_xblock_tz_lu_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_sxblock_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_tzfft_preconditioner",
    ),
    (
        "sfincs_jax.solvers.preconditioner_transport_matrix",
        "build_rhsmode23_xmg_preconditioner",
    ),
)

PROFILE_RESPONSE_COMPATIBILITY_IMPORTS = (
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_pas_fast_accept",
    ),
    (
        "sfincs_jax.problems.profile_setup",
        "sfincs_jax.problems.profile_setup",
        "resolve_rhs1_active_dof_mode",
    ),
    (
        "sfincs_jax.problems.profile_setup",
        "sfincs_jax.problems.profile_setup",
        "reduce_full_with_indices",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_constraint0_sparse_first",
    ),
    (
        "sfincs_jax.problems.profile_solver_diagnostics",
        "sfincs_jax.problems.profile_solver_diagnostics",
        "rhs1_accept_candidate",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_fast_post_xblock_polish_allowed",
    ),
    (
        "sfincs_jax.problems.profile_residual",
        "sfincs_jax.problems.profile_residual",
        "residual_target",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_sparse_exact_lu_requested",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_polish_enabled",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_sparse_kind_use",
    ),
    (
        "sfincs_jax.problems.profile_policies",
        "sfincs_jax.problems.profile_policies",
        "rhs1_stage2_trigger",
    ),
    (
        "sfincs_jax.problems.profile_solver_diagnostics",
        "sfincs_jax.problems.profile_solver_diagnostics",
        "build_rhs1_xblock_correction_metadata",
    ),
    (
        "sfincs_jax.problems.profile_solver_diagnostics",
        "sfincs_jax.problems.profile_solver_diagnostics",
        "V3LinearSolveResult",
    ),
    (
        "sfincs_jax.problems.profile_solver_diagnostics",
        "sfincs_jax.problems.profile_solver_diagnostics",
        "V3NewtonKrylovResult",
    ),
    (
        "sfincs_jax.problems.profile_solver_diagnostics",
        "sfincs_jax.problems.profile_solver_diagnostics",
        "v3_linear_solve_result_from_payload",
    ),
    (
        "sfincs_jax.operators.profile_sparse_pattern",
        "sfincs_jax.operators.profile_sparse_pattern",
        "v3_full_system_conservative_sparsity_pattern",
    ),
    (
        "sfincs_jax.operators.profile_sparse_pattern",
        "sfincs_jax.operators.profile_sparse_pattern",
        "v3_full_system_fortran_reduced_preconditioner_sparsity_pattern",
    ),
    (
        "sfincs_jax.operators.profile_fblock",
        "sfincs_jax.operators.profile_fblock",
        "V3FBlockOperator",
    ),
    (
        "sfincs_jax.operators.profile_fblock",
        "sfincs_jax.operators.profile_fblock",
        "fblock_operator_from_namelist",
    ),
    (
        "sfincs_jax.operators.profile_fblock",
        "sfincs_jax.operators.profile_fblock",
        "matvec_v3_fblock_flat",
    ),
    (
        "sfincs_jax.operators.profile_system",
        "sfincs_jax.operators.profile_system",
        "V3FullSystemOperator",
    ),
    (
        "sfincs_jax.operators.profile_system",
        "sfincs_jax.operators.profile_system",
        "full_system_operator_from_namelist",
    ),
    (
        "sfincs_jax.operators.profile_system",
        "sfincs_jax.operators.profile_system",
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
        "sfincs_jax.problems.profile_preconditioner_build",
        "sfincs_jax.problems.profile_preconditioner_build",
        "auto_rhs1_full_strong_kind",
    ),
    (
        "sfincs_jax.problems.profile_preconditioner_build",
        "sfincs_jax.problems.profile_preconditioner_build",
        "rhs1_resolved_strong_preconditioner_control",
    ),
    (
        "sfincs_jax.problems.profile_preconditioner_build",
        "sfincs_jax.problems.profile_preconditioner_build",
        "requested_rhs1_strong_preconditioner_kind",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "FortranReducedXBlockBackendContext",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "SparsePCDirectTailFactorSetupContext",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "SparsePCDirectTailRescuePolicySetupContext",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "SparsePCGenericBranchSetupContext",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "build_sparse_pc_direct_tail_factor_setup",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "build_sparse_pc_direct_tail_rescue_policy_setup",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
        "build_sparse_pc_generic_branch_setup",
    ),
    (
        "sfincs_jax.problems.profile_sparse_solve",
        "sfincs_jax.problems.profile_sparse_solve",
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
            assert module.__name__ == "sfincs_jax.problems.profile_solve"
            continue
        assert module.__name__ == module_name


def test_module_names_reserved_for_later_package_migration_still_load_as_modules() -> None:
    """Avoid silently shadowing large legacy modules during Phase A."""

    for module_name in RESERVED_MODULE_NAMES_UNTIL_MIGRATION:
        module = _import_module(module_name)
        assert not hasattr(module, "__path__"), module_name
        assert module.__file__ is not None
        assert module.__file__.endswith(".py"), module.__file__


def test_moved_root_workflow_modules_have_domain_owners() -> None:
    """Phase 2 root reductions should land in durable domain owners, not shims."""

    for module_name, expected_exports in MOVED_ROOT_MODULE_OWNERS.items():
        module = _import_module(module_name)
        assert module.__doc__ is not None
        for export_name in expected_exports:
            assert hasattr(module, export_name), f"{module_name}.{export_name}"

    for deleted_root in DELETED_ROOT_ALIASES:
        try:
            _import_module(deleted_root)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{deleted_root} should not remain as a root compatibility shim")


def test_github_workflows_do_not_use_deleted_flat_import_aliases() -> None:
    """CI jobs must follow the same domain-owner import contract as code/tests."""

    repo_root = Path(__file__).resolve().parents[1]
    workflow_dir = repo_root / ".github" / "workflows"
    for path in workflow_dir.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for deleted_alias in DELETED_ROOT_ALIASES:
            assert deleted_alias not in text, f"{path.relative_to(repo_root)} still imports {deleted_alias}"


def test_root_modules_are_explicitly_classified() -> None:
    """Batch E requires every remaining package-root module to have an owner class."""

    root = Path(__file__).resolve().parents[1] / "sfincs_jax"
    actual = {path.name for path in root.glob("*.py")}
    expected = set(ROOT_MODULE_CLASSIFICATIONS)
    assert actual == expected
    allowed_classes = {
        "compatibility facade",
        "compatibility shim",
        "public API",
        "public compatibility API",
        "public differentiation API",
        "public discretization API",
        "public entry point",
        "public geometry API",
        "public geometry workflow API",
        "public input API",
        "public package facade",
        "public physics API",
        "public plotting API",
        "public support workflow",
        "public validation API",
        "public workflow API",
        "stable discretization kernel",
        "stable geometry kernel",
        "stable numerical kernel",
        "stable operator kernel",
        "stable physics kernel",
        "stable preconditioner kernel",
        "stable solver kernel",
        "stable solver-policy kernel",
        "stable support utility",
    }
    assert set(ROOT_MODULE_CLASSIFICATIONS.values()) <= allowed_classes


def test_root_module_closure_manifest_is_complete_and_documented() -> None:
    """Closure Phase 1 requires a move/delete decision for every root module."""

    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "sfincs_jax"
    actual = {path.name for path in root.glob("*.py")}
    assert set(ROOT_MODULE_CLOSURE_MANIFEST) == actual
    assert set(ROOT_MODULE_CLOSURE_MANIFEST) == set(ROOT_MODULE_CLASSIFICATIONS)

    source_map = (repo_root / "docs" / "source_map.rst").read_text(encoding="utf-8")
    assert "Closure move/delete manifest" in source_map
    for filename, (target_owner, disposition) in ROOT_MODULE_CLOSURE_MANIFEST.items():
        assert f"``{filename}``" in source_map
        assert target_owner in source_map
        assert disposition in source_map


def test_source_map_does_not_advertise_deleted_flat_aliases() -> None:
    """Deleted flat rhs1/transport modules must appear only as historical notes."""

    repo_root = Path(__file__).resolve().parents[1]
    source_map = (repo_root / "docs" / "source_map.rst").read_text(encoding="utf-8")
    assert "legacy alias" not in source_map
    assert "The legacy ``sfincs_jax/transport_matrix.py`` path remains" not in source_map
    live_deleted_root_owner = re.compile(
        r"(?m)^- ``sfincs_jax/(?:rhs1|transport)[^`]*\.py``:"
    )
    assert live_deleted_root_owner.search(source_map) is None


def test_transport_matrix_canonical_modules_expose_expected_symbols() -> None:
    """Transport implementation modules should use only canonical flat owners."""

    for _, new_name, public_name in TRANSPORT_COMPATIBILITY_IMPORTS:
        new_module = _import_module(new_name)
        assert hasattr(new_module, public_name), f"{new_name}.{public_name}"


def test_preconditioner_canonical_modules_expose_expected_symbols() -> None:
    """Preconditioner implementation modules should use only canonical owners."""

    for _, new_name, public_name in PRECONDITIONER_COMPATIBILITY_IMPORTS:
        new_module = _import_module(new_name)
        assert hasattr(new_module, public_name), f"{new_name}.{public_name}"


def test_preconditioner_implementation_modules_expose_expected_builders() -> None:
    """Implementation modules should keep the moved numerical builders importable."""

    for module_name, public_name in PRECONDITIONER_IMPLEMENTATION_IMPORTS:
        module = _import_module(module_name)
        assert public_name in getattr(module, "__all__", ()), module_name
        assert hasattr(module, public_name), f"{module_name}.{public_name}"


def test_profile_response_canonical_modules_expose_expected_symbols() -> None:
    """Profile-response implementation modules should use canonical flat owners."""

    for _, new_name, public_name in PROFILE_RESPONSE_COMPATIBILITY_IMPORTS:
        new_module = _import_module(new_name)
        assert hasattr(new_module, public_name), f"{new_name}.{public_name}"
        if hasattr(new_module, "__all__"):
            assert public_name in new_module.__all__


def test_sparse_solve_reexport_waiver_is_documented() -> None:
    """The sparse solve lint waiver must remain a documented compatibility seam."""

    source = Path("sfincs_jax/problems/profile_sparse_solve.py").read_text()
    assert "# ruff: noqa: F401,F811" in source
    assert "dynamic re-export surface" in source
    assert "Delete this waiver" in source
    assert "solve.py and owner tests import the concrete sparse owners directly" in source
