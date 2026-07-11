API reference
=============

The stable public facade is :mod:`sfincs_jax.api` (re-exported at the package
top level). The canonical modules that implement the physics and solver stack
are indexed below, with links to the pages that document them in depth and to
:doc:`source_map` for the full source catalogue.

Public facade
-------------

.. automodule:: sfincs_jax.api
   :members:

High-level runners live in :mod:`sfincs_jax.run` (``run_profile``,
``run_transport_matrix``) and the CLI in :mod:`sfincs_jax.cli`; see :doc:`usage`.

Canonical modules
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``sfincs_jax.run``
     - ``run_profile`` / ``run_transport_matrix`` orchestration and console flow
     - :doc:`usage`
   * - ``sfincs_jax.inputs`` / ``sfincs_jax.namelist``
     - Typed ``SfincsInput`` parser and raw namelist reader
     - :doc:`inputs`
   * - ``sfincs_jax.magnetic_geometry``
     - ``FluxSurfaceGeometry`` and all geometry schemes
     - :doc:`geometry`
   * - ``sfincs_jax.species``
     - Charges, masses, profiles, deflection frequencies
     - :doc:`physics_reference`
   * - ``sfincs_jax.phase_space``
     - Legendre coupling, Landreman--Ernst speed grid, ``Nxi_for_x`` ramp
     - :doc:`numerics`
   * - ``sfincs_jax.drift_kinetic``
     - ``KineticOperator`` — the consolidated v3 drift-kinetic operator
     - :doc:`physics_reference`, :doc:`system_equations`
   * - ``sfincs_jax.collisions``
     - Pitch-angle scattering and full Fokker--Planck (Rosenbluth) operators
     - :doc:`physics_reference`
   * - ``sfincs_jax.solve``
     - Three-tier auto solver (block-tridiagonal, recycled Krylov, host direct)
     - :doc:`numerics`
   * - ``sfincs_jax.moments``
     - Velocity-space moments, fluxes, ``FSABjHat``, transport matrix
     - :doc:`outputs`, :doc:`physics_reference`
   * - ``sfincs_jax.phi1``
     - Nonlinear :math:`\Phi_1` / quasineutrality Newton solve
     - :doc:`physics_reference`
   * - ``sfincs_jax.er``
     - Ambipolar radial-electric-field root solve (Brent + differentiable)
     - :doc:`physics_reference`
   * - ``sfincs_jax.writer`` / ``sfincs_jax.io``
     - ``sfincsOutput.h5`` / ``.nc`` / ``.npz`` writer and reader
     - :doc:`outputs`
   * - ``sfincs_jax.diagnostics``
     - Geometry-derived scalar diagnostics
     - :doc:`outputs`
   * - ``sfincs_jax.sensitivity``
     - Implicit-differentiation observable derivatives (RHSMode 4/5 spine)
     - :doc:`feature_matrix`
   * - ``sfincs_jax.compare``
     - ``compare-h5`` parity tooling against Fortran fixtures
     - :doc:`parity`

Namelist parsing and reader helpers
------------------------------------

.. automodule:: sfincs_jax.namelist
   :members:

Legacy pipeline modules
-----------------------

The retained legacy solver/operator/problem/geometry-subpackage internals are
off the canonical solve path. They are catalogued, with their canonical
replacements, in :doc:`source_map`.
