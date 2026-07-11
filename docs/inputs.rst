Inputs (namelist) reference
===========================

`sfincs_jax` reads Fortran-style namelist files, typically named
``input.namelist``. The input surface is compatible with the mature SFINCS-style
ecosystem so equilibria, scans, and case descriptions carry over, but this page
documents the public `sfincs_jax` interface in its own right and enumerates every
parameter the canonical parser recognizes.

How the namelist parser works
-----------------------------

The canonical typed parser is :class:`sfincs_jax.inputs.SfincsInput`. Group names
are matched case-insensitively; each supported parameter is a typed field whose
Python type is inferred from its default (``tuple`` :math:`\to` float array,
``bool``, ``int``, ``float``, else ``str``). Two behaviors are worth stating up
front:

- **Species arrays** accept indexed assignment (``Zs(2) = 6.0``) and are folded
  into vectors; the species count is the number of ``Zs`` entries.
- **Unknown keys are not an error.** A key that is not a typed field of a
  recognized group is dropped from the typed struct but retained in
  ``SfincsInput.raw``; an unknown *group* is likewise retained in ``raw`` but
  never typed. "Supported" below means specifically the typed fields.

The tables give the Fortran namelist name, default, and type for every typed
field, grouped by namelist.

``&general``
------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Name
     - Default
     - Type / meaning
   * - ``RHSMode``
     - ``1``
     - int; 1 = profile solve, 2 = 3×3 transport matrix, 3 = monoenergetic 2×2
   * - ``ambipolarSolve``
     - ``.false.``
     - bool; solve for the ambipolar :math:`E_r`
   * - ``ambipolarSolveOption``
     - ``2``
     - int; root-finder option (2 = Brent, canonical)
   * - ``NEr_ambipolarSolve``
     - ``20``
     - int; number of :math:`E_r` samples
   * - ``Er_search_tolerance_dx`` / ``Er_search_tolerance_f``
     - ``1e-8`` / ``1e-10``
     - float; root-find tolerances
   * - ``Er_min`` / ``Er_max``
     - ``-100`` / ``100``
     - float; :math:`E_r` search bracket
   * - ``outputFilename``
     - ``"sfincsOutput.h5"``
     - str; output file name
   * - ``solveSystem``
     - ``.true.``
     - bool; solve (vs geometry/grids only)
   * - ``saveMatlabOutput`` / ``saveMatricesAndVectorsInBinary``
     - ``.false.``
     - bool; legacy dump toggles

``&geometryParameters``
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 22 38

   * - Name
     - Default
     - Type / meaning
   * - ``geometryScheme``
     - ``1``
     - int; geometry family (:doc:`geometry`)
   * - ``equilibriumFile``
     - ``""``
     - str; ``.bc`` / ``wout`` path (aliases ``JGboozer_file``,
       ``JGboozer_file_NonStelSym``, ``fort996boozer_file``)
   * - ``GHat`` / ``IHat`` / ``iota``
     - ``3.7481`` / ``0.0`` / ``0.4542``
     - float; Boozer flux functions and rotational transform (analytic schemes)
   * - ``B0OverBBar``
     - ``1.0``
     - float; reference field ratio
   * - ``psiAHat`` / ``aHat``
     - ``0.15596`` / ``0.5585``
     - float; edge flux and minor-radius normalizations
   * - ``epsilon_t`` / ``epsilon_h`` / ``epsilon_antisymm``
     - ``-0.07053`` / ``0.05067`` / ``0.0``
     - float; scheme-1 toroidal / helical / antisymmetric ripple amplitudes
   * - ``helicity_l`` / ``helicity_n``
     - ``2`` / ``10``
     - int; scheme-1 helical mode numbers
   * - ``helicity_antisymm_l`` / ``helicity_antisymm_n``
     - ``1`` / ``0``
     - int; scheme-1 antisymmetric mode numbers
   * - ``NPeriods``
     - ``0``
     - int; field periods (0 = infer from scheme)
   * - ``min_Bmn_to_load`` / ``rippleScale``
     - ``0.0`` / ``1.0``
     - float; Boozer harmonic filter / ripple scaling
   * - ``inputRadialCoordinate``
     - ``3``
     - int; surface label (0=ψ, 1=ψ_N, 2=r̂, 3=r_N, 4=Er)
   * - ``inputRadialCoordinateForGradients``
     - ``4``
     - int; gradient label
   * - ``psiHat_wish`` / ``psiN_wish`` / ``rHat_wish`` / ``rN_wish``
     - ``-1`` / ``0.25`` / ``-1`` / ``0.5``
     - float; requested surface (``normradius_wish`` aliases ``rN_wish``)
   * - ``VMECRadialOption`` / ``VMEC_Nyquist_option``
     - ``1`` / ``1``
     - int; VMEC radial-interpolation and Nyquist conventions

``&speciesParameters``
----------------------

Species arrays default to empty tuples; give one entry per species. All arrays
are float arrays. Gradients are accepted in every radial coordinate.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Name
     - Meaning
   * - ``Zs`` / ``mHats`` / ``nHats`` / ``THats``
     - charge, mass, density, temperature (per species)
   * - ``dNHatdpsiHats`` / ``dNHatdpsiNs`` / ``dNHatdrHats`` / ``dNHatdrNs``
     - density gradient in each radial coordinate
   * - ``dTHatdpsiHats`` / ``dTHatdpsiNs`` / ``dTHatdrHats`` / ``dTHatdrNs``
     - temperature gradient in each radial coordinate
   * - ``withAdiabatic`` / ``adiabaticZ`` / ``adiabaticMHat`` / ``adiabaticNHat`` / ``adiabaticTHat``
     - adiabatic species toggle (default ``.false.``) and its Z / m / n / T
   * - ``withNBIspec`` / ``NBIspecZ`` / ``NBIspecNHat``
     - NBI fast-species toggle (default ``.false.``) and its Z / n

``&physicsParameters``
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Name
     - Default
     - Type / meaning
   * - ``Delta`` / ``alpha`` / ``nu_n``
     - ``4.5694e-3`` / ``1.0`` / ``8.330e-3``
     - float; drift-kinetic ordering parameters (:doc:`normalizations`)
   * - ``EParallelHat``
     - ``0.0``
     - float; inductive parallel field :math:`\hat E_\parallel`
   * - ``dPhiHatdpsiHat`` / ``dPhiHatdpsiN`` / ``dPhiHatdrHat`` / ``dPhiHatdrN`` / ``Er``
     - ``0.0``
     - float; radial-electric-field drive in each coordinate
   * - ``collisionOperator``
     - ``0``
     - int; 0 = full Fokker--Planck, 1 = pitch-angle scattering
   * - ``constraintScheme``
     - ``-1``
     - int; source/constraint scheme (−1 = auto by collision operator)
   * - ``includeXDotTerm``
     - ``.true.``
     - bool; :math:`E_r` energy drift :math:`\dot x\,\partial_x`
   * - ``includeElectricFieldTermInXiDot``
     - ``.true.``
     - bool; :math:`E_r` pitch drift :math:`\dot\xi\,\partial_\xi`
   * - ``useDKESExBDrift``
     - ``.false.``
     - bool; DKES (:math:`\langle B^2\rangle`) vs full (:math:`B^2`) ExB
   * - ``include_fDivVE_term``
     - ``.false.``
     - bool; compressible ExB correction
   * - ``includeTemperatureEquilibrationTerm``
     - ``.false.``
     - bool; add :math:`C_{ab}[f_{aM},f_{bM}]` to the RHS
   * - ``includePhi1``
     - ``.false.``
     - bool; enable the quasineutrality / :math:`\Phi_1` block
   * - ``includePhi1InKineticEquation``
     - ``.true.``
     - bool; :math:`\Phi_1` terms in the DKE (requires ``includePhi1``)
   * - ``includePhi1InCollisionOperator``
     - ``.false.``
     - bool; poloidal density factor in collisions
   * - ``quasineutralityOption``
     - ``1``
     - int; 1 = full Boltzmann, 2 = EUTERPE adiabatic
   * - ``readExternalPhi1``
     - ``.false.``
     - bool; **deferred** (see below)
   * - ``nuPrime`` / ``EStar``
     - ``1.0`` / ``0.0``
     - float; monoenergetic collisionality / electric field (``RHSMode=3``)
   * - ``magneticDriftScheme``
     - ``0``
     - int; tangential magnetic drift (**> 0 deferred on the canonical stack**)
   * - ``Krook``
     - ``0.0``
     - float; optional non-conserving drag

``&resolutionParameters``
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Name
     - Default
     - Meaning
   * - ``Ntheta`` / ``Nzeta``
     - ``15`` / ``15``
     - poloidal / toroidal grid points
   * - ``Nxi``
     - ``16``
     - Legendre pitch modes
   * - ``NL``
     - ``4``
     - Rosenbluth-potential Legendre modes (``Nxi_for_x`` floor)
   * - ``Nx``
     - ``5``
     - speed-grid nodes (forced to 1 for ``RHSMode=3``)
   * - ``xMax`` / ``NxPotentialsPerVth``
     - ``5.0`` / ``40.0``
     - speed-grid extent / Rosenbluth potential resolution
   * - ``solverTolerance``
     - ``1e-6``
     - linear-solve residual target
   * - ``forceOddNthetaAndNzeta``
     - ``.true.``
     - round grid sizes to odd

The Fortran convergence-scan **ramp arrays** (``NthetaNumRuns``, ``Nx_min``, ...)
have no typed fields; if present they are retained in ``raw`` only and are not
acted upon.

``&otherNumericalParameters``
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 44 14 42

   * - Name
     - Default
     - Meaning
   * - ``thetaDerivativeScheme`` / ``zetaDerivativeScheme``
     - ``2`` / ``2``
     - angular finite-difference scheme
   * - ``ExBDerivativeSchemeTheta`` / ``ExBDerivativeSchemeZeta``
     - ``0`` / ``0``
     - ExB derivative scheme
   * - ``magneticDriftDerivativeScheme``
     - ``3``
     - magnetic-drift derivative / upwinding scheme
   * - ``xDotDerivativeScheme``
     - ``0``
     - speed-derivative scheme (canonical: 0)
   * - ``xGridScheme`` / ``xPotentialsGridScheme`` / ``xGrid_k``
     - ``5`` / ``2`` / ``0.0``
     - speed-grid family and weight exponent :math:`k`
   * - ``Nxi_for_x_option``
     - ``1``
     - :math:`N_\xi`-for-:math:`x` ramp (:doc:`numerics`)
   * - ``useIterativeLinearSolver``
     - ``.true.``
     - iterative vs direct (alias ``useIterativeSolver``)
   * - ``whichParallelSolverToFactorPreconditioner`` / ``PETSCPreallocationStrategy``
     - ``1`` / ``1``
     - legacy PETSc knobs (retained for compatibility)

``&preconditionerOptions``
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 12 38

   * - Name
     - Default
     - Meaning
   * - ``preconditioner_species``
     - ``1``
     - 1 = self-collisions only in the coarse operator
   * - ``preconditioner_x`` / ``preconditioner_x_min_L``
     - ``1`` / ``0``
     - x-diagonal collisions in the preconditioner
   * - ``preconditioner_xi``
     - ``1``
     - drop L±1 streaming coupling in the preconditioner
   * - ``preconditioner_theta`` / ``preconditioner_zeta``
     - ``0`` / ``0``
     - angular coarsening
   * - ``preconditioner_theta_min_L`` / ``preconditioner_zeta_min_L``
     - ``0`` / ``0``
     - minimum L kept per angle
   * - ``preconditioner_magnetic_drifts_max_L``
     - ``2``
     - magnetic-drift truncation in the preconditioner
   * - ``reusePreconditioner``
     - ``.true.``
     - reuse the factorization across right-hand sides

These map onto the tier-2 coarse preconditioner in :doc:`numerics`.

Deferred and unsupported inputs
-------------------------------

Stated plainly:

- **``&export_f`` group** — retained in ``raw`` only, untyped. Setting
  ``export_full_f`` or ``export_delta_f`` rejects the run from the JAX pipeline
  with a clear message ("export_f output is deferred to the legacy pipeline").
- **``readExternalPhi1``** — parsed but not honored; a run with it set is routed
  to the legacy pipeline, and reaching the canonical solver with it raises
  ``NotImplementedError``.
- **Non-stellarator-symmetric VMEC** — routed to the legacy owner; Boozer
  ``.bc`` scheme 12 is the supported non-symmetric path (:doc:`geometry`).
- **``magneticDriftScheme > 0``** and **resolution-ramp arrays** — deferred /
  raw-only as noted above.

Geometry and equilibrium inputs
-------------------------------

The public geometry families are documented in :doc:`geometry`. The most common
choices are analytic tokamak (``geometryScheme = 1``), VMEC
(``geometryScheme = 5`` with ``equilibriumFile`` or ``wout_path``), and Boozer
``.bc`` (``geometryScheme = 11`` or ``12``). The CLI and Python API share one
equilibrium search order:

- absolute path from the namelist or explicit override,
- relative to the input namelist directory,
- relative to the current working directory,
- directories listed in ``SFINCS_JAX_EQUILIBRIA_DIRS``,
- small bundled test/example data directories,
- then known release-hosted public fixtures by basename, fetched into the
  ``SFINCS_JAX_DATA_DIR`` cache when needed.

Prefetch the fixture cache with
``python -m sfincs_jax.validation.data_fetch``, or set ``SFINCS_JAX_OFFLINE=1``
to require a pre-populated cache.

Runtime overrides
-----------------

Without editing the namelist, override the equilibrium source through:

- CLI: ``--equilibrium-file ...`` or ``--wout-path ...``
- Python: ``equilibrium_file=...`` or ``wout_path=...`` in
  :func:`sfincs_jax.io.write_sfincs_jax_output_h5`

When an override is used, the embedded ``input.namelist`` dataset written to
``sfincsOutput.h5`` reflects the effective configuration.

Transport-matrix modes (``RHSMode = 2/3``)
------------------------------------------

For ``RHSMode = 2`` and ``RHSMode = 3`` the solver loops over ``whichRHS`` and
overwrites the relevant drives internally before building each RHS. On the
canonical stack this is ``KineticOperator._with_rhs_settings``, so the transport-mode right-hand side reproduces the
v3 solver RHS exactly. For ``RHSMode = 3`` the speed grid collapses to the single
node at :math:`x=1` (``Nx = 1``), matching v3 ``createGrids.F90``; this is handled
by :meth:`sfincs_jax.drift_kinetic.KineticOperator.from_namelist`.

Worked example decks are in ``examples/`` (see :doc:`examples`), and the geometry
input knobs are cross-referenced from :doc:`geometry`.
