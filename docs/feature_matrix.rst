Fortran v3 And sfincs_jax Feature Matrix
=========================================

This page is the review anchor for functionality parity and intentional
extensions. It is audited from the local SFINCS Fortran v3 source tree and the
vendored version-3 manual mirror on 2026-06-25. It separates three questions that
are easy to conflate during refactoring:

- what the Fortran v3 code can do,
- where that behavior lives in ``sfincs_jax``,
- and which promotion gate is still required before making a production claim.

Status definitions
------------------

``implemented``
   The capability is available through the public CLI or Python API and has a
   focused test or checked validation artifact.

``implemented with gates``
   The capability is available, but production claims are limited by explicit
   residual, runtime, memory, grid-resolution, or backend gates.

``partial``
   The mathematical or API spine exists, but at least one Fortran-compatible
   output, branch, option, or production-resolution gate is not complete.

``deferred``
   The capability is intentionally documented as future work. It must not be
   used as release evidence until its promotion gate is closed.

Fortran v3 feature ownership
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 23 29 24 24

   * - Feature surface
     - Fortran v3 owner
     - Source/manual evidence
     - ``sfincs_jax`` owner and status
   * - Namelist schema, defaults, and validation
     - ``readInput.F90``, ``validateInput.F90``, ``globalVariables.F90``
     - Manual ``inputParameters.tex`` documents ``RHSMode``,
       ``ambipolarSolve``, geometry, collision, and drift compatibility.
     - ``sfincs_jax.namelist`` and ``sfincs_jax.input_compat`` validation
       helpers: implemented, with compatibility guards expanded as refactoring
       exposes cleaner public contracts.
   * - RHSMode 1 profile-response solve
     - ``solver.F90``, ``evaluateResidual.F90``, ``populateMatrix.F90``,
       ``diagnostics.F90``, ``writeHDF5Output.F90``
     - Fortran performs a nonlinear or linear profile-response solve, then
       writes particle fluxes, heat fluxes, flows, currents, Phi1 fields, and
       convergence diagnostics.
     - ``sfincs_jax.problems.profile_solve`` plus flat ``profile_*`` problem
       owners and legacy driver shims:
       implemented with gates. QA/QH production-grid convergence is supported
       with documented solver-policy limits; lower-memory replacement remains a
       performance lane, not a correctness blocker.
   * - RHSMode 2 and RHSMode 3 transport/monoenergetic matrices
     - ``solver.F90``, ``diagnostics.F90``, ``validateInput.F90``
     - Fortran loops over transport RHS columns; RHSMode 3 enforces
       monoenergetic constraints such as ``Nx=1`` and DKES-compatible settings.
     - ``sfincs_jax.problems.transport_matrix``: implemented with gates.
       Geometry-rich production preconditioners are bounded by residual and
       setup-time admission tests before auto promotion.
   * - Ambipolar root solve option 1
     - ``ambipolarSolver.F90``, ``solver.F90``, ``adjointDiagnostics.F90``
     - Safeguarded Newton/bisection uses an adjoint-computed
       ``dRadialCurrentdEr`` and maintains a bracket.
     - ``sfincs_jax.problems.ambipolar`` and ``sfincs_jax.sensitivity``:
       implemented with gates. Root-policy logic, dense certificates, matrix-free/JVP
       derivative-provider gates, analytic existing-branch ``E_r`` tangents,
       opt-in fixed-shape zero-field branch retention, and a namelist-backed
       RHSMode-1 derivative-response helper now run a bounded small-deck
       Fortran active-operator ``particleFlux_vm_rN`` option-1 root replay.
       Production physical replay gates remain outside normal CI.
   * - Ambipolar root solve option 2
     - ``ambipolarSolver.F90``
     - Brent method evaluates the radial current at bracket endpoints and an
       initial guess, then uses inverse interpolation or bisection.
     - ``sfincs_jax.problems.ambipolar``: implemented and tested against checked
       small helical summaries.
   * - Ambipolar root solve option 3
     - ``ambipolarSolver.F90``, ``solver.F90``, ``adjointDiagnostics.F90``
     - Pure Newton uses the adjoint-computed ``dRadialCurrentdEr`` and exits if
       a step leaves the allowed ``E_r`` bounds.
     - ``sfincs_jax.problems.ambipolar`` and ``sfincs_jax.sensitivity``:
       implemented with bounded gates. Fast option-3-style matrix-free
       derivative-provider root tests are covered, checked small helical plus
       W7-X-like option-3 currents replay with the active namelist-backed
       provider, and the helical small-deck pure-Newton root replay now runs
       through the same namelist-backed provider. Production refresh benchmarks
       remain external to normal CI.
   * - RHSMode 4 fixed-``E_r`` sensitivities
     - ``solver.F90``, ``populateAdjointRHS.F90``,
       ``populatedMatrixdLambda.F90``, ``populatedRHSdLambda.F90``,
       ``adjointDiagnostics.F90``, ``testingAdjointDiagnostics.F90``
     - Fortran solves adjoint systems for particle flux, heat flux, bootstrap,
       parallel flow, total heat flux, and radial current sensitivities.
     - ``sfincs_jax.sensitivity`` plus diagnostic observable builders:
       source contract plus derivative spine present. Linear implicit
       derivative, JVP/VJP, dot-product, small RHSMode-1 radial-current gates,
       active Fortran-style option-1 ``dJ_r/dE_r`` replay, Fortran-v3
       RHSMode-4/5 input validators, HDF5 sensitivity field-name/rank gates,
       and compact Fortran RHSMode-4 radial-current, heat-flux,
       parallel-flow, bootstrap, and debug finite-difference numerical replay
       fixtures are implemented and covered by tests. Production-grid refreshes
       are release benchmarks rather than normal CI gates.
   * - RHSMode 5 ambipolar sensitivities
     - ``ambipolarSolver.F90``, ``solver.F90``, ``adjointDiagnostics.F90``
     - Fortran first finds ambipolar ``E_r``, then evaluates derivatives at
       constant radial current with the extra ``dPhi/dPsi`` term.
     - Bounded/reference implemented. The implicit sensitivity spine is present,
       and a compact Fortran RHSMode=5 heat-flux/``dPhidPsidLambda``
       constant-current fixture pins the output surface. Full production-grid
       RHSMode-5 parity is tracked as an external release-refresh benchmark.
   * - Collision models
     - ``populateMatrix.F90`` and collision-specific helpers
     - Manual and validation checks distinguish PAS and full Fokker-Planck
       branches, plus field-particle and momentum-restoring terms.
     - ``sfincs_jax.physics.collisions`` and assembly helpers: implemented for the
       release-facing suite, with high-pitch/geometry-rich performance gates
       tracked separately.
   * - Magnetic and electric drift branches
     - ``validateInput.F90``, ``populateMatrix.F90``, ``geometry.F90``
     - Manual ``magneticDriftScheme`` notes Fortran limitations when Phi1 and
       some magnetic-drift terms are combined.
     - Implemented with gates. Release examples cover DKES-like and full
       trajectory branches; Phi1 drift-current sensitivity promotion is still
       explicit future work.
   * - Geometry schemes and radial-coordinate conversions
     - ``geometry.F90``, ``radialCoordinates.F90``, ``updateBoozerGeometry.F90``
     - Fortran supports analytic, Boozer, VMEC-derived, and related geometry
       schemes, with RHSMode>3 restricted to Boozer coordinates.
     - ``sfincs_jax.geometry``, ``sfincs_jax.geometry.vmec``,
       ``sfincs_jax.geometry.jax_adapters``: implemented with gates. VMEC and
       differentiable adapters are supported; broader QI and scheme-13
       production promotion remains documented research work.
   * - Phi1/quasineutrality
     - ``evaluateResidual.F90``, ``populateMatrix.F90``, ``diagnostics.F90``
     - Fortran solves coupled kinetic/quasineutrality systems for compatible
       RHSMode-1 settings and rejects RHSMode>3 with Phi1.
     - ``sfincs_jax.operators.profile_system`` and profile-response modules: implemented for
       documented RHSMode-1 cases. RHSMode 4/5 with Phi1 remains invalid by
       design, matching Fortran validation.
   * - Sparse solver and preconditioner backend
     - ``solver.F90`` with PETSc ``KSP``, ``PCLU``, MUMPS, SuperLU_DIST, serial
       sparse direct fallback, transpose solves, and MUMPS memory retry controls.
     - Fortran generally factors a preconditioner/direct matrix and uses the
       same infrastructure for adjoint/transpose solves.
     - Native JAX/Python solver stack: implemented with gates. ``sfincs_jax``
       intentionally does not require PETSc/MUMPS/SuperLU_DIST. Current work
       focuses on reusable operator protocols, native block/Schur factors, and
       strict true-residual admission rather than external direct-solver
       bindings.
   * - Output files and plotting
     - ``writeHDF5Output.F90`` and diagnostics writers
     - Fortran writes HDF5 fields for inputs, geometry, solution, diagnostics,
       and adjoint quantities when enabled.
     - ``sfincs_jax.outputs`` and CLI plotting: implemented for HDF5, NetCDF,
       NPZ, and PDF plot workflows. RHSMode 4/5 adjoint output contracts are
       pinned by compact Fortran-v3 sensitivity fixtures; production refreshes
       remain benchmark artifacts rather than normal CI data.
   * - Parallelism
     - PETSc/MPI distribution plus parallel sparse factorization backends
     - Fortran delegates matrix distribution, factorization, and solves to MPI
       PETSc and optional parallel direct solvers.
     - JAX CPU/GPU execution, chunking, worker scaling, and sharding paths:
       implemented with gates. Single-case multi-GPU strong scaling and true
       device-QI are deferred research lanes until strict production gates pass.

``sfincs_jax`` implementation status
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 35 25

   * - Capability
     - Status
     - Evidence in this branch
     - Remaining promotion gate
   * - Example-suite parity against SFINCS Fortran v3
     - implemented
     - Frozen CPU/GPU 39-case reports and benchmark plots referenced from
       :doc:`validation_matrix` and :doc:`fortran_comparison`.
     - Regenerate after production-floor suite changes or default solver-policy
       changes.
   * - Public CLI and Python solve workflows
     - implemented
     - CLI, output-format, plotting, and import-contract tests; API docs in
       :doc:`usage`, :doc:`outputs`, and :doc:`api`.
     - Continue moving private internals behind stable problem-domain APIs.
   * - Ambipolar Brent option
     - implemented
     - Checked ambipolar summaries and root-policy tests.
     - Add production metadata replay for solver counts, residual separation,
       and RSS bounds.
   * - Ambipolar Newton/bisection and pure Newton options
     - bounded small-deck implemented; production pending
     - Small-deck derivative certificates, the namelist-backed fixed-shape
       RHSMode-1 radial-current response helper, active Fortran-style
       option-1 current/slope replay, option-1/3 root replay, and
       radial-current observable gates in ``tests/test_sensitivity.py`` and
       ``tests/test_ambipolar_problem.py``.
     - Production sparse/matrix-free ``E_r`` derivatives, in-process RHSMode-1
       evaluator reuse, and production Fortran option-1/3 replay gates.
   * - RHSMode 4 fixed-``E_r`` sensitivities
     - source contract implemented; numerical replay pending
     - ``sfincs_jax.sensitivity`` supports implicit linear observable
       derivatives, builder probes, JVP/VJP, adjoint dot-product checks,
       Fortran-compatible input validation, and HDF5 sensitivity field-name
       gates.
     - Build numerical Fortran-compatible diagnostic/output surfaces and
       intermediate-grid gates.
   * - RHSMode 5 ambipolar sensitivities
     - source contract implemented; numerical replay deferred
     - Shared sensitivity spine exists; constant-current formulas are documented
       in this matrix and source-compatible input/output field gates are tested.
     - Close option-1/3 ambipolar derivative gates first, then add
       constant-current HDF5 diagnostics.
   * - QA/QH bootstrap-current validation
     - implemented with gates
     - README/docs figures compare ``sfincs_jax``, SFINCS Fortran v3, and Redl
       formula at matched resolutions where available.
     - More radial points and production-grid repeats after solver-policy
       changes.
   * - QI kinetic promotion
     - implemented with gates
     - Bounded CPU/GPU/Fortran promotion artifacts and documented claim
       boundaries.
     - True device-QI and production-resolution ladders remain deferred.
   * - Lower-memory production solver replacement
     - implemented with gates
     - Native block/Schur/factor infrastructure, direct-Pmat experiments, and
       true-residual admission tests.
     - Promote to auto only when production CPU/GPU residual, runtime, and RSS
       gates pass.
   * - Refactor/review-ready architecture
     - partial
     - Domain packages and import-contract tests exist; new extracted helpers
       are covered by focused tests.
     - Finish owner-boundary extraction, simplify ``v3_driver.py``, keep docs and
       source map synchronized, and avoid file-count churn.

Using this matrix during review
-------------------------------

When a PR changes defaults, solver dispatch, input compatibility, or output
fields, update this page in the same commit as the code or test that changes the
claim. A feature should move from ``partial`` to ``implemented`` only when it has
all of the following:

- a public CLI or Python entry point,
- a focused unit or numerical test,
- a physics or cross-code gate when the capability is physics-facing,
- docs that identify limitations and reproduction commands,
- and, for performance claims, fresh runtime and memory provenance.
