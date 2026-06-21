API reference
=============

.. automodule:: sfincs_jax.namelist
   :members:

.. automodule:: sfincs_jax.v3
   :members:

.. automodule:: sfincs_jax.geometry
   :members:

.. automodule:: sfincs_jax.vmec_wout
   :members:

.. automodule:: sfincs_jax.vmec_geometry
   :members:

.. automodule:: sfincs_jax.jax_geometry_adapters
   :members:

.. automodule:: sfincs_jax.diagnostics
   :members:

.. automodule:: sfincs_jax.outputs.formats
   :members:

.. automodule:: sfincs_jax.grids
   :members:

.. automodule:: sfincs_jax.xgrid
   :members:

.. automodule:: sfincs_jax.collisionless
   :members:

.. automodule:: sfincs_jax.collisionless_er
   :members:

.. automodule:: sfincs_jax.collisionless_exb
   :members:

.. automodule:: sfincs_jax.magnetic_drifts
   :members:

.. automodule:: sfincs_jax.collisions
   :members:

.. automodule:: sfincs_jax.v3_fblock
   :members:

.. automodule:: sfincs_jax.v3_system
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.diagnostics
   :members:

``sfincs_jax.transport_matrix`` remains a compatibility alias for existing
scripts and notebooks.

.. automodule:: sfincs_jax.residual
   :members:

.. automodule:: sfincs_jax.validation_artifacts
   :members:

.. automodule:: sfincs_jax.memory_model
   :members:

.. automodule:: sfincs_jax.v3_results
   :members:

.. automodule:: sfincs_jax.solver_runtime
   :members:

.. automodule:: sfincs_jax.matrix_reductions
   :members:

.. automodule:: sfincs_jax.linear_algebra
   :members:

.. automodule:: sfincs_jax.sparse_triangular
   :members:

.. automodule:: sfincs_jax.preconditioner_context
   :members:

.. automodule:: sfincs_jax.preconditioner_operators
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.pas.xblock_ilu
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_policy
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_basis
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.xblock.low_l_schur
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.xblock.active_projected
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.xblock.tz_sparse
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.direct_pmat
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.direct_block_schur
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.fortran_reduced_lu
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.setup
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.active_dense
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.loop
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.finalize
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.postsolve_diagnostics
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.solve
   :members:

.. automodule:: sfincs_jax.preconditioner_setup
   :members:

.. automodule:: sfincs_jax.explicit_sparse_factor_policy
   :members:

.. automodule:: sfincs_jax.explicit_sparse_factor_builder
   :members:

.. automodule:: sfincs_jax.krylov_dispatch
   :members:

.. automodule:: sfincs_jax.preconditioner_caches
   :members:

.. automodule:: sfincs_jax.v3_driver
   :members:

Refactored solve-policy modules
-------------------------------

These modules hold small, directly tested policy and dispatch decisions that used
to live inside the large driver. They are included here because they are part of
the maintainable public source structure for debugging, testing, and downstream
research workflows.

.. automodule:: sfincs_jax.problems.profile_response.handoff
   :members:

.. automodule:: sfincs_jax.problems.profile_response.policies
   :members:

.. automodule:: sfincs_jax.problems.profile_response.phi1_newton
   :members:

.. automodule:: sfincs_jax.problems.profile_response.strong_preconditioning
   :members:

.. automodule:: sfincs_jax.rhs1_lowmode_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_domain_decomposition
   :members:

.. automodule:: sfincs_jax.problems.profile_response.active_dof
   :members:

.. automodule:: sfincs_jax.problems.profile_response.active_projection
   :members:

.. automodule:: sfincs_jax.rhs1_constraint_sources
   :members:

.. automodule:: sfincs_jax.rhs1_device_operator
   :members:

.. automodule:: sfincs_jax.rhs1_host_policy
   :members:

.. automodule:: sfincs_jax.rhs1_direct_tail_policy
   :members:

.. automodule:: sfincs_jax.rhs1_fortran_reduced_direct_tail
   :members:

.. automodule:: sfincs_jax.rhs1_structured_full_csr
   :members:

.. automodule:: sfincs_jax.rhs1_true_operator_rescue
   :members:

.. automodule:: sfincs_jax.host_refinement
   :members:

.. automodule:: sfincs_jax.rhs1_large_cpu_policy
   :members:

.. automodule:: sfincs_jax.rhs1_pas_matrixfree
   :members:

.. automodule:: sfincs_jax.rhs1_pas_policy
   :members:

.. automodule:: sfincs_jax.rhs1_preconditioner_auto_policy
   :members:

.. automodule:: sfincs_jax.rhs1_preconditioner_dispatch
   :members:

.. automodule:: sfincs_jax.rhs1_qi_active_pattern_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_qi_block_schur
   :members:

.. automodule:: sfincs_jax.rhs1_qi_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_qi_coupled_residual
   :members:

.. automodule:: sfincs_jax.rhs1_qi_deflation
   :members:

.. automodule:: sfincs_jax.rhs1_qi_device_preconditioner
   :members:

.. automodule:: sfincs_jax.rhs1_qi_device_smoother
   :members:

.. automodule:: sfincs_jax.rhs1_qi_galerkin_policy
   :members:

.. automodule:: sfincs_jax.rhs1_qi_global_moment_closure
   :members:

.. automodule:: sfincs_jax.rhs1_qi_multilevel_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_qi_phase_space_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_qi_promotion
   :members:

.. automodule:: sfincs_jax.rhs1_qi_residual_galerkin
   :members:

.. automodule:: sfincs_jax.rhs1_qi_residual_region_coarse
   :members:

.. automodule:: sfincs_jax.rhs1_qi_two_level
   :members:

.. automodule:: sfincs_jax.problems.profile_response.residual
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.direct
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.finalization
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.fortran_reduced
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.krylov
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.qi
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.xblock
   :members:

.. automodule:: sfincs_jax.rhs1_schur_policy
   :members:

.. automodule:: sfincs_jax.problems.profile_response.solver_diagnostics
   :members:

.. automodule:: sfincs_jax.rhs1_ksp_diagnostics
   :members:

.. automodule:: sfincs_jax.newton_krylov_diagnostics
   :members:

.. automodule:: sfincs_jax.rhs1_solver_policy
   :members:

.. automodule:: sfincs_jax.rhs1_strong_fallback
   :members:

.. automodule:: sfincs_jax.rhs1_xblock_policy
   :members:

.. automodule:: sfincs_jax.rhs1_xblock_sparse_host_policy
   :members:

.. automodule:: sfincs_jax.solve_mode_policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.dense_lu
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.dense_batch
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.active_factor
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.policies
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.solve_policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.handoff_policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.host_gmres
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.iteration_stats
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.linear_solve
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.payload
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.runtime
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.execution
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.pool
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.validation
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.sharding
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.worker
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.preconditioner_dispatch
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.residual_quality
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.sparse_direct_solve
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.streaming_outputs
   :members:

.. automodule:: sfincs_jax.indices
   :members:

.. automodule:: sfincs_jax.petsc_binary
   :members:

.. automodule:: sfincs_jax.sparse
   :members:

.. automodule:: sfincs_jax.solver
   :members:

.. automodule:: sfincs_jax.implicit_solve
   :members:

.. automodule:: sfincs_jax.paths
   :members:
