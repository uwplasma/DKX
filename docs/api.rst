API reference
=============

.. automodule:: sfincs_jax.api
   :members:

.. automodule:: sfincs_jax.namelist
   :members:

.. automodule:: sfincs_jax.discretization.v3
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

.. automodule:: sfincs_jax.operators.profile_response.fblock
   :members:

.. automodule:: sfincs_jax.operators.profile_response.system
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.diagnostics
   :members:

Transport-matrix helpers are documented through their maintained domain module.

.. automodule:: sfincs_jax.residual
   :members:

.. automodule:: sfincs_jax.validation.artifacts
   :members:

.. automodule:: sfincs_jax.solvers.memory_model
   :members:

.. automodule:: sfincs_jax.problems.profile_response.solver_diagnostics
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.finalize
   :members:

.. automodule:: sfincs_jax.solver
   :members:

.. automodule:: sfincs_jax.solvers.sparse_triangular
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_context
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_operators
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.pas.xblock_ilu
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.schur.profile_response
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

.. automodule:: sfincs_jax.problems.transport_matrix.finalize
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_setup
   :members:

.. automodule:: sfincs_jax.solvers.explicit_sparse_factor_policy
   :members:

.. automodule:: sfincs_jax.solvers.explicit_sparse_factor_builder
   :members:

.. automodule:: sfincs_jax.solvers.krylov_dispatch
   :members:

.. automodule:: sfincs_jax.solvers.preconditioner_caches
   :members:

.. automodule:: sfincs_jax.v3_driver

   Compatibility shim for historical imports. New code should import solve
   entry points from the domain owners documented below.

Refactored solve-policy modules
-------------------------------

These modules hold small, directly tested policy and dispatch decisions that used
to live inside the large driver. They are included here because they are part of
the maintainable public source structure for debugging, testing, and downstream
research workflows.

.. automodule:: sfincs_jax.problems.profile_response.policies
   :members:

.. automodule:: sfincs_jax.problems.profile_response.setup
   :members:

.. automodule:: sfincs_jax.problems.profile_response.phi1_newton
   :members:

.. automodule:: sfincs_jax.problems.profile_response.preconditioner_build
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.xblock.coarse
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.domain_decomposition
   :members:

.. automodule:: sfincs_jax.operators.profile_response.sources
   :members:

.. automodule:: sfincs_jax.operators.profile_response.device_sparse
   :members:

.. automodule:: sfincs_jax.operators.profile_response.reduced_tail
   :members:

.. automodule:: sfincs_jax.operators.profile_response.structured_csr
   :members:

.. automodule:: sfincs_jax.operators.profile_response.sparse_pattern
   :members:

.. automodule:: sfincs_jax.operators.profile_response.true_operator_rescue
   :members:

.. automodule:: sfincs_jax.host_refinement
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.pas.matrix_free
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.pas.policy
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.dispatch
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.qi.basis
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.qi.corrections
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.qi.device
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.qi.policy
   :members:

.. automodule:: sfincs_jax.problems.profile_response.residual
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.direct
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.finalization
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.fortran_reduced
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.handoff
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.qi
   :members:

.. automodule:: sfincs_jax.problems.profile_response.sparse.xblock
   :members:

.. automodule:: sfincs_jax.solvers.preconditioners.xblock.policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.active_factor
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.policies
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.solve
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.policy
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.runtime
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.sharding
   :members:

.. automodule:: sfincs_jax.problems.transport_matrix.parallel.worker
   :members:

.. automodule:: sfincs_jax.outputs.transport
   :members:

.. automodule:: sfincs_jax.indices
   :members:

.. automodule:: sfincs_jax.validation.petsc_binary
   :members:

.. automodule:: sfincs_jax.solvers.explicit_sparse
   :members:

.. automodule:: sfincs_jax.solvers.implicit
   :members:

.. automodule:: sfincs_jax.sensitivity
   :members:

.. automodule:: sfincs_jax.paths
   :members:
