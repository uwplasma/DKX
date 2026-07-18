API reference
=============

The stable public facade is :mod:`dkx.api` (re-exported at the package
top level). The canonical modules that implement the physics and solver stack
are indexed below, with links to the pages that document them in depth and to
:doc:`source_map` for the full source catalogue.

Public facade
-------------

.. automodule:: dkx.api
   :members:

High-level runners live in :mod:`dkx.run` (``run_profile``,
``run_transport_matrix``) and the CLI in :mod:`dkx.cli`; see :doc:`usage`.

Canonical modules
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``dkx.run``
     - ``run_profile`` / ``run_transport_matrix`` orchestration and console flow
     - :doc:`usage`
   * - ``dkx.inputs`` / ``dkx.namelist``
     - Typed ``SfincsInput`` parser and raw namelist reader
     - :doc:`inputs`
   * - ``dkx.magnetic_geometry``
     - ``FluxSurfaceGeometry`` and all geometry schemes
     - :doc:`geometry`
   * - ``dkx.species``
     - Charges, masses, profiles, deflection frequencies
     - :doc:`physics_reference`
   * - ``dkx.phase_space``
     - Legendre coupling, Landreman--Ernst speed grid, ``Nxi_for_x`` ramp
     - :doc:`numerics`
   * - ``dkx.drift_kinetic``
     - ``KineticOperator`` — the consolidated v3 drift-kinetic operator
     - :doc:`physics_reference`, :doc:`system_equations`
   * - ``dkx.collisions``
     - Pitch-angle scattering and full Fokker--Planck (Rosenbluth) operators
     - :doc:`physics_reference`
   * - ``dkx.solve``
     - Three-tier auto solver (block-tridiagonal, recycled Krylov, host direct)
     - :doc:`numerics`
   * - ``dkx.moments``
     - Velocity-space moments, fluxes, ``FSABjHat``, transport matrix
     - :doc:`outputs`, :doc:`physics_reference`
   * - ``dkx.phi1``
     - Nonlinear :math:`\Phi_1` / quasineutrality Newton solve
     - :doc:`physics_reference`
   * - ``dkx.er``
     - Ambipolar radial-electric-field root solve (Brent + differentiable)
     - :doc:`physics_reference`
   * - ``dkx.writer`` / ``dkx.io``
     - ``sfincsOutput.h5`` / ``.nc`` / ``.npz`` writer and reader
     - :doc:`outputs`
   * - ``dkx.sensitivity``
     - Implicit-differentiation observable derivatives (RHSMode 4/5 spine)
     - :doc:`feature_matrix`
   * - ``dkx.compare``
     - ``compare-h5`` parity tooling against Fortran fixtures
     - :doc:`parity`

Namelist parsing and reader helpers
------------------------------------

.. automodule:: dkx.namelist
   :members:

Reduced-model and analysis modules
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``dkx.monoenergetic``
     - Monoenergetic-database mode: ``(nuPrime, EStar)`` scans and energy
       convolution to thermal ``L_ij``
     - :doc:`capabilities`
   * - ``dkx.variational``
     - Variational upper/lower bounds on the monoenergetic :math:`D_{11}`
       (convergence certificate)
     - :doc:`capabilities`
   * - ``dkx.shaing_callen``
     - Collisionless-limit bootstrap coefficient with an analytic axisymmetric
       cross-check
     - :doc:`capabilities`

The differentiable solve, the implicit adjoint, and the
``vmex -> booz_xform_jax -> dkx`` chain are documented in
:doc:`differentiability`; the full source catalogue is :doc:`source_map`.
