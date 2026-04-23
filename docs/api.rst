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

.. automodule:: sfincs_jax.transport_matrix
   :members:

.. automodule:: sfincs_jax.residual
   :members:

.. automodule:: sfincs_jax.v3_driver
   :members:

Refactored solve-policy modules
-------------------------------

These modules hold small, directly tested policy and dispatch decisions that used
to live inside the large driver. They are included here because they are part of
the maintainable public source structure for debugging, testing, and downstream
research workflows.

.. automodule:: sfincs_jax.rhs1_handoff
   :members:

.. automodule:: sfincs_jax.rhs1_constraint0_policy
   :members:

.. automodule:: sfincs_jax.rhs1_host_policy
   :members:

.. automodule:: sfincs_jax.rhs1_large_cpu_policy
   :members:

.. automodule:: sfincs_jax.rhs1_pas_policy
   :members:

.. automodule:: sfincs_jax.rhs1_post_xblock_policy
   :members:

.. automodule:: sfincs_jax.rhs1_preconditioner_auto_policy
   :members:

.. automodule:: sfincs_jax.rhs1_preconditioner_dispatch
   :members:

.. automodule:: sfincs_jax.rhs1_schur_policy
   :members:

.. automodule:: sfincs_jax.rhs1_sparse_polish_policy
   :members:

.. automodule:: sfincs_jax.rhs1_sparse_exact_policy
   :members:

.. automodule:: sfincs_jax.rhs1_sparse_rescue_policy
   :members:

.. automodule:: sfincs_jax.rhs1_stage2_policy
   :members:

.. automodule:: sfincs_jax.rhs1_strong_auto_kind
   :members:

.. automodule:: sfincs_jax.rhs1_strong_control
   :members:

.. automodule:: sfincs_jax.rhs1_strong_fallback
   :members:

.. automodule:: sfincs_jax.rhs1_strong_policy
   :members:

.. automodule:: sfincs_jax.transport_dense_lu
   :members:

.. automodule:: sfincs_jax.transport_handoff_policy
   :members:

.. automodule:: sfincs_jax.transport_host_gmres
   :members:

.. automodule:: sfincs_jax.transport_preconditioner_dispatch
   :members:

.. automodule:: sfincs_jax.transport_solve_policy
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
