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

Reduced-model and analysis modules
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``sfincs_jax.monoenergetic``
     - Monoenergetic-database mode: ``(nuPrime, EStar)`` scans and energy
       convolution to thermal ``L_ij``
     - :doc:`capabilities`
   * - ``sfincs_jax.variational``
     - Variational upper/lower bounds on the monoenergetic :math:`D_{11}`
       (convergence certificate)
     - :doc:`capabilities`
   * - ``sfincs_jax.shaing_callen``
     - Collisionless-limit bootstrap coefficient with an analytic axisymmetric
       cross-check
     - :doc:`capabilities`

The differentiable solve, the implicit adjoint, and the
``vmec_jax -> booz_xform_jax -> sfincs_jax`` chain are documented in
:doc:`differentiability`; the full source catalogue is :doc:`source_map`.
