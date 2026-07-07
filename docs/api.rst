API reference
=============

.. automodule:: sfincs_jax.api
   :members:

.. automodule:: sfincs_jax.namelist
   :members:

.. automodule:: sfincs_jax.compare
   :members:

.. automodule:: sfincs_jax.discretization.v3
   :members:

.. automodule:: sfincs_jax.geometry
   :members:

.. automodule:: sfincs_jax.geometry.vmec_wout
   :members:

.. automodule:: sfincs_jax.geometry.vmec
   :members:

.. automodule:: sfincs_jax.geometry.jax_adapters
   :members:

.. automodule:: sfincs_jax.diagnostics
   :members:

.. automodule:: sfincs_jax.outputs.formats
   :members:

.. automodule:: sfincs_jax.grids
   :members:

.. automodule:: sfincs_jax.discretization.xgrid
   :members:

.. automodule:: sfincs_jax.operators.profile_collisionless
   :members:

.. automodule:: sfincs_jax.operators.profile_electric_field
   :members:

.. automodule:: sfincs_jax.operators.profile_exb
   :members:

.. automodule:: sfincs_jax.operators.profile_magnetic_drifts
   :members:

.. automodule:: sfincs_jax.physics.collisions
   :members:

.. automodule:: sfincs_jax.operators.profile_fblock
   :members:

.. automodule:: sfincs_jax.operators.profile_system
   :members:

.. automodule:: sfincs_jax.problems.transport_diagnostics
   :members:

Transport-matrix and profile residual helpers are documented through their
maintained domain modules.

.. automodule:: sfincs_jax.validation.artifacts
   :members:

.. automodule:: sfincs_jax.solvers.memory_model
   :members:

.. automodule:: sfincs_jax.problems.profile_solver_diagnostics
   :members:

.. automodule:: sfincs_jax.problems.transport_finalize
   :members:

.. automodule:: sfincs_jax.solvers.krylov
   :members:

.. automodule:: sfincs_jax.solvers.preconditioning
   :members:

.. automodule:: sfincs_jax.solvers.diagnostics
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_pas_xblock_ilu
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_schur_profile
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_xblock_low_l_schur
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_xblock_active
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_xblock_tz_sparse
   :members:

.. automodule:: sfincs_jax.problems.transport_linear_system
   :members:

.. automodule:: sfincs_jax.problems.transport_setup
   :members:

.. automodule:: sfincs_jax.solvers.explicit_sparse
   :members:

.. automodule:: sfincs_jax.solvers.krylov_dispatch
   :members:

Refactored solve-policy modules
-------------------------------

These modules hold small, directly tested policy and dispatch decisions that used
to live inside the large driver. They are included here because they are part of
the maintainable public source structure for debugging, testing, and downstream
research workflows.

.. automodule:: sfincs_jax.problems.profile_policies
   :members:

.. automodule:: sfincs_jax.problems.profile_setup
   :members:

.. automodule:: sfincs_jax.problems.profile_phi1_newton
   :members:

.. automodule:: sfincs_jax.problems.profile_preconditioner_build
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_xblock_coarse
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_domain_decomposition
   :members:

.. automodule:: sfincs_jax.operators.profile_device_sparse
   :members:

.. automodule:: sfincs_jax.operators.profile_reduced_tail
   :members:

.. automodule:: sfincs_jax.operators.profile_sparse_pattern
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_pas_matrix_free
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_pas_policy
   :members:

.. automodule:: sfincs_jax.problems.profile_residual
   :members:

.. automodule:: sfincs_jax.problems.profile_sparse_direct
   :members:

.. automodule:: sfincs_jax.problems.profile_sparse_finalization
   :members:

.. automodule:: sfincs_jax.problems.profile_sparse_fortran_reduced
   :members:

.. automodule:: sfincs_jax.problems.profile_sparse_solve
   :members:

.. automodule:: sfincs_jax.problems.profile_sparse_xblock
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_xblock_policy
   :members:

.. automodule:: sfincs_jax.problems.transport_policies
   :members:

.. automodule:: sfincs_jax.problems.transport_solve
   :members:

.. automodule:: sfincs_jax.problems.transport_parallel_runtime
   :members:

.. automodule:: sfincs_jax.outputs.transport
   :members:

.. automodule:: sfincs_jax.validation.fortran
   :members:

.. automodule:: sfincs_jax.solvers.implicit
   :members:

.. automodule:: sfincs_jax.sensitivity
   :members:

.. automodule:: sfincs_jax.paths
   :members:
