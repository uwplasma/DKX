Source-code map
===============

This page links the main pieces of the mathematics to the source files that implement
them. The goal is to shorten the path from an equation in the docs to the exact module
that evaluates it.

High-level flow
---------------

For a standard solve, the execution path is:

1. parse namelist and resolve equilibrium inputs,
2. build grids and geometry coefficients,
3. construct the operator / residual objects,
4. choose a solve path and preconditioner,
5. run the linear or nonlinear iteration,
6. postprocess diagnostics and write ``sfincsOutput.h5``.

Package layout
--------------

The package uses one level of domain folders below ``sfincs_jax/``. New
implementation code should be placed by physics or numerical responsibility,
not by historical helper prefixes. The importable package folders are:

- ``sfincs_jax/discretization`` for grids, quadrature, basis functions,
  indexing/layout, sparse stencils, active degrees of freedom, speed-grid
  maps, and velocity-space structure.
- ``sfincs_jax/geometry`` for analytic Boozer geometry, Boozer-file loading,
  VMEC ``wout`` loading, and JAX-native geometry adapters.
- ``sfincs_jax/operators`` for drift-kinetic operator terms, matrix-free
  actions, assembled sparse helpers, profile-response layouts, source terms,
  and SFINCS Fortran-v3 convention translation.
- ``sfincs_jax/outputs`` for output schemas, output caches, HDF5/NetCDF/NPZ
  file formats, RHSMode=1 diagnostics, and RHSMode=2/3 transport-output fields.
- ``sfincs_jax/physics`` for collision operators, classical-transport formulas,
  bootstrap-current normalization, and analytic validation formulas.
- ``sfincs_jax/problems`` for physical problem orchestration: RHSMode=1
  profile-current/bootstrap-current solves, RHSMode=2/3 transport-matrix and
  monoenergetic-response solves, ambipolar root solves, and scan-level
  diagnostics.
- ``sfincs_jax/solvers`` for Krylov dispatch, sparse/direct factors,
  residual-gate policies, memory models, implicit differentiation, and
  preconditioner families.
- ``sfincs_jax/validation`` for frozen-reference loading, parity checks,
  release-data manifests, validation artifacts, and comparison math.
- ``sfincs_jax/workflows`` for optional research workflows that combine the
  public APIs into optimization, mapped-grid, scan, QI-promotion, and upstream
  postprocessing tasks.

The root modules are the stable user-facing surface: ``api.py``, ``cli.py``,
``solver.py`` (compatibility alias), ``ambipolar.py``, ``sensitivity.py``, ``plotting.py``,
``compare.py``, ``io.py``, ``namelist.py``, ``input_compat.py``, ``grids.py``,
``diagnostics.py``, ``paths.py``, and ``profiling.py``. Historical monolithic
driver imports have been retired; use the profile and transport problem owners
directly.

The source tree does not contain nested implementation packages or non-root
one-file facades for profile-response, transport-matrix, or preconditioner
families. Use the flat canonical owners directly: ``operators/profile_*.py``,
``problems/profile_*.py``, ``problems/transport_*.py``, and
``solvers/preconditioner_*.py``.

Root public surface classification
----------------------------------

The package root is intentionally small and compatibility-aware. New
implementation code should go into the domain packages listed above. The root
files that remain after the consolidation pass are classified here so reviewers
can distinguish public entry points and stable kernels from compatibility
facades.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Root file
     - Classification
     - Reason to keep at package root
   * - ``__init__.py``
     - public package facade
     - Defines the import-time package surface and lazy public helpers.
   * - ``__main__.py``
     - public entry point
     - Supports ``python -m sfincs_jax``.
   * - ``api.py``
     - public API
     - Stable Python contracts and lazy high-level facades.
   * - ``cli.py``
     - public entry point
     - Command-line parser and dispatch surface.
   * - ``ambipolar.py``
     - public physics API
     - Lightweight ambipolar postprocessing helpers used by workflows and tests.
   * - ``compare.py``
     - public validation API
     - SFINCS output comparison, strict numeric HDF5 parity, and benchmark-summary helpers used by examples, scripts, and validation tools.
   * - ``input_compat.py``
     - public compatibility API
     - Fortran-v3-compatible input normalization and equilibrium path handling.
   * - ``io.py``
     - compatibility facade
     - Stable output read/write import path; concrete writers live in ``outputs``.
   * - ``namelist.py``
     - public input API
     - Namelist parser used directly by scripts, tests, and examples.
   * - ``plotting.py``
     - public plotting API
     - Lightweight output plotting helpers used by CLI/examples.
   * - ``sensitivity.py``
     - public differentiation API
     - JVP/VJP, implicit, and adjoint certificates used by optimization workflows.
   * - ``diagnostics.py``
     - stable physics kernel
     - Flux-surface averages, moment integrals, and output diagnostics.
   * - ``grids.py``
     - public discretization API
     - Velocity/grid helpers used directly by docs and tests.
   * - ``paths.py``
     - stable support utility
     - Repository/data path resolution helpers.
   * - ``profiling.py``
     - stable support utility
     - Phase timing and optional memory profiling helpers.
   * - ``solver.py``
     - compatibility facade
     - Historical import path that aliases the implementation in ``solvers/krylov.py``.

Closure move/delete manifest
----------------------------

This manifest locks the owner decision for every package-root file before any
more code movement. This prevents one-helper churn: a later change may move a
file only if the move follows the owner below, keeps the root file as a
documented compatibility facade, and passes the corresponding owner tests.

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Root file
     - Target owner
     - Disposition or deletion condition
   * - ``__init__.py``
     - package root public facade
     - keep at root
   * - ``__main__.py``
     - package root CLI entry point
     - keep at root
   * - ``constants.py``
     - canonical normalization/radial-coordinate owner
     - keep at root as canonical stack module
   * - ``species.py``
     - canonical species-pytree owner
     - keep at root as canonical stack module
   * - ``phase_space.py``
     - canonical grids/discretization owner
     - keep at root as canonical stack module
   * - ``magnetic_geometry.py``
     - canonical flux-surface geometry owner
     - keep at root as canonical stack module
   * - ``collisions.py``
     - canonical collision-operator owner
     - keep at root as canonical stack module
   * - ``drift_kinetic.py``
     - canonical KineticOperator owner
     - keep at root as canonical stack module
   * - ``solve.py``
     - canonical three-tier solver owner
     - keep at root as canonical stack module
   * - ``moments.py``
     - canonical velocity-space moments owner
     - keep at root as canonical stack module
   * - ``inputs.py``
     - canonical typed-namelist owner
     - keep at root as canonical stack module
   * - ``console.py``
     - canonical Fortran-parity stdout owner
     - keep at root as canonical stack module
   * - ``run.py``
     - canonical transport-run driver owner
     - keep at root as canonical stack module
   * - ``writer.py``
     - canonical sfincsOutput writer owner
     - keep at root as canonical stack module
   * - ``api.py``
     - package root public API
     - keep at root
   * - ``cli.py``
     - package root CLI entry point
     - keep at root
   * - ``ambipolar.py``
     - problems.ambipolar via public API facade
     - keep root facade until public docs/examples migrate
   * - ``compare.py``
     - package root public comparison API
     - keep at root; it owns user-facing SFINCS comparison helpers and strict HDF5 output parity
   * - ``input_compat.py``
     - input compatibility owner
     - keep root public compatibility facade until input package exports cover callers
   * - ``io.py``
     - outputs writer/formats owners
     - keep tiny root facade until public imports migrate
   * - ``namelist.py``
     - input namelist owner
     - keep root public parser until input package exports are documented
   * - ``plotting.py``
     - outputs/plotting public helper
     - keep root public helper unless API replacement is documented
   * - ``sensitivity.py``
     - package root differentiation API
     - keep at root
   * - ``diagnostics.py``
     - physics/output diagnostics owner
     - defer until diagnostics API split is explicit
   * - ``grids.py``
     - discretization public grid owner
     - keep root public helper until discretization package exports are documented
   * - ``paths.py``
     - package root path support utility
     - keep at root unless a support package is introduced with broad import rewrite
   * - ``profiling.py``
     - solvers/validation profiling support
     - defer until profiling API boundary is explicit
   * - ``solver.py``
     - solvers.krylov public contracts owner
     - root facade aliases ``sfincs_jax.solvers.krylov`` for compatibility

Core modules
------------

``sfincs_jax/api.py``
^^^^^^^^^^^^^^^^^^^^^

Stable public data contracts for high-level workflows:

- ``SolveInputs`` for normalized CLI/Python solve requests,
- ``GeometryState`` / ``GridState`` / ``OperatorState`` summaries passed across
  problem setup, validation, and solver layers,
- ``PreconditionerState`` and ``SolverResult`` for solver metadata,
- ``TransportResult`` for file-format-independent transport summaries,
- ``OutputSchema`` and ``BenchmarkReport`` for output and performance artifacts.
- ``write_output``, ``read_output``, and ``run_ambipolar_brent`` as lazy public
  facades for common Python workflows that should not import implementation
  internals directly.

These contracts are plain frozen dataclasses and intentionally avoid importing
JAX at module import. Facade functions import the heavy solve/output modules
inside the function body, so solver-specific JAX pytrees remain in the numerical
modules that need JAX transformations.

``sfincs_jax/sensitivity.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Differentiation contracts for optimization and validation workflows:

- JVP/VJP wrappers and dot-product consistency checks for JAX-transformable
  flux and diagnostic functions,
- dense implicit linear-observable certificates for small validation decks,
- matrix-free implicit linear-observable certificates for production-size
  owners that can provide operator, transpose, derivative-action, solve, and
  transpose-solve closures,
- operator-tangent helpers that let production owners use JAX ``jvp`` actions
  instead of materialized derivative matrices,
- finite-difference comparison hooks used as promotion gates before derivative
  paths are used in ambipolar Newton solves or RHSMode=4/5-style adjoint
  diagnostics.

This module is intentionally problem-agnostic. Problem packages such as
``sfincs_jax.problems.ambipolar`` adapt these certificates to physical
observables including radial-current derivatives.

``sfincs_jax/cli.py``
^^^^^^^^^^^^^^^^^^^^^

Public command-line interface:

- ``sfincs_jax input.namelist`` default solve mode,
- ``write-output``,
- ``transport-matrix-v3``,
- comparison and utility commands,
- parallel runtime/bootstrap flags.

RHSMode=2/3 transport-matrix runs (``transport-matrix-v3`` and the default
``write-output`` dispatch for RHSMode=2/3 inputs) route through the canonical
stack driver ``sfincs_jax.run.run_transport_matrix``; the legacy outputs
writer remains the owner only for RHSMode=1 runs and for options the
canonical writer does not cover yet (``.npz`` output, ``export_f``, solver
traces, ``--no-overwrite``, non-Fortran layout).

``sfincs_jax/io.py``
^^^^^^^^^^^^^^^^^^^^

Input/output orchestration:

- reads namelists,
- resolves equilibrium overrides (including ``wout_path``),
- materializes output diagnostics,
- exposes the in-memory results API,
- keeps documented compatibility aliases for flat output-format helpers owned
  by ``sfincs_jax.outputs.formats``.

``sfincs_jax/outputs/formats.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flat output file-format and output-cache helpers:

- HDF5, NetCDF, and NPZ readers/writers,
- SFINCS Fortran-compatible HDF5 layout conversion,
- output suffix dispatch,
- NetCDF-safe dataset naming,
- solver-trace attachment for HDF5 outputs.
- output-cache environment gates,
- stable cache-directory and cache-path construction,
- hashable namelist fragments for geometry/species/physics cache keys,
- content-based equilibrium identity for VMEC/Boozer files,
- geometry-output cache-key construction with an injected equilibrium resolver,
- filtered disk load/save for geometry-derived fields such as
  ``gpsiHatpsiHat``, ``uHat``, ``VPrimeHat``, ``FSABHat2``, ``BDotCurlB``,
  ``diotadpsiHat``, and classical no-``Phi1`` fluxes.

``sfincs_jax/io.py`` keeps a small compatibility wrapper for
``_output_geom_cache_key`` because equilibrium path localization is still part
of input/output orchestration.

``sfincs_jax/outputs/rhsmode1.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=1 output-safety and trace-schema helpers:

- production-size nonconverged-output refusal gates,
- residual/target extraction from solver results,
- solver metadata normalization for output and traces,
- main-file RHSMode=1 solver diagnostics, timing, memory-estimate, direct-tail,
  sparse-PC, and residual-target-ratio output fields,
- conservative solver-trace memory estimates,
- sidecar JSON trace writing when a large RHSMode=1 diagnostic output is
  intentionally refused.

Solved-field physics schema construction is exposed through the stable
``sfincs_jax/io.py`` facade and implemented with the output owners listed on
this page. Future I/O work should keep physical solved-field and provenance
schema construction behind a small output contract while preserving writer
functions and the output-cache boundary in the ``outputs`` package.

``sfincs_jax/outputs/transport.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=2/3 transport output-schema helpers:

- per-``whichRHS`` linear residual, RHS norm, and relative residual arrays,
- max residual and max relative residual summary fields,
- explicit ``NaN`` placeholders for missing RHS diagnostics in partial debug
  artifacts.
- streaming HDF5 transport diagnostics writer for large RHSMode=2/3 output
  payloads, including per-``whichRHS`` solved-field slices, flux variants,
  classical fluxes, transport matrix, elapsed time, and solver diagnostics
  without keeping every diagnostic array resident in ``io.py``.
- radial derivative conversion factors used when writing ``psiHat``,
  ``psiN``, ``rHat``, and ``rN`` transport-flux variants.

``sfincs_jax/workflows/scans.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Electric-field scan workflow owner:

- scan-directory materialization,
- scan-point input patching,
- optional scan-point parallelism,
- progress and ETA reporting,
- Krylov state recycling between adjacent scan points,
- public ``run_er_scan`` helper used by CLI, examples, and validation scripts,
- upstream-style postprocessing helpers that locate vendored or user-provided
  SFINCS Fortran-v3 ``utils`` scripts,
- non-interactive execution of plotting/postprocessing scripts with a non-GUI
  matplotlib backend.

This module is the canonical owner for scan execution and upstream-style
postprocessing workflow helpers.

``sfincs_jax/workflows/optimization.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimization-facing workflow owner. It contains the differentiable
neoclassical proxy objectives, high-fidelity scan-promotion gates,
candidate-scan plan builders, promotion evidence campaign builders,
CPU/GPU/Fortran promotion comparison gates, and finite-beta convergence-ladder
checks. Experimental QI/device-QI campaign ingestion is preserved on the
``research/qi-device-hard-seed`` branch, not in the stable core, so workflow
code stays focused on reusable execution and evidence-generation tasks.
Historical ``optimization_*`` workflow modules resolve through package-level
compatibility aliases in ``sfincs_jax.workflows`` instead of separate
implementation files.

``sfincs_jax/workflows/mapped_xgrid.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mapped speed-grid workflow owner. It contains differentiable Maxwellian moment
objectives for rational-tail speed grids, bounded transport-matrix evidence
reports, namelist patching for ``xGridScheme = 50``, CSV/JSON artifact writers,
and solve-summary/error metrics. Historical mapped-x-grid objective/evidence
modules resolve through package-level compatibility aliases.

``sfincs_jax/validation/data_fetch.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Release-hosted external-equilibrium fixture owner:

- reads the embedded equilibrium-data manifest,
- downloads and verifies release-hosted VMEC/Boozer fixture archives,
- resolves known equilibrium basenames from the user cache,
- supports offline CI and examples without committing large fixtures to git.

This module replaces the former root ``sfincs_jax/data_fetch.py``
implementation.

``sfincs_jax/input_compat.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compatibility and search-order logic for equilibrium files, input normalization,
radial-coordinate conversions, and user overrides. This is the module to
inspect first when a case fails to find a VMEC or Boozer file, or when a
Fortran-v3 radial-coordinate input such as ``psiHat_wish``, ``psiN_wish``,
``rHat_wish``, or ``rN_wish`` needs to be converted consistently.

``sfincs_jax/geometry/jax_adapters.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional structural adapters for JAX-native geometry producers such as ``vmec_jax``
and ``booz_xform_jax``. These helpers do not make either package a required
dependency; they normalize in-memory VMEC-like ``wout`` objects to the internal
``VmecWout`` layout used by ``geometryScheme=5``. The module also owns the
machine-readable VMEC/Boozer proxy workflow contract and the shared no-solve
provenance gate used by the workflow status and autodiff examples.

``sfincs_jax/grids.py`` and ``sfincs_jax/discretization/xgrid.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Velocity-space discretization:

- collocation points in :math:`x`,
- quadrature weights,
- modal transforms used by the collision operator,
- special handling for monoenergetic ``RHSMode=3``.
- mapped speed-grid research primitives in ``discretization/adaptive_maps.py``,
- Fortran-v3 active indexing in ``discretization/v3.py::V3Indexing``,
- compact periodic stencil extraction/application in
  ``discretization/periodic_stencil.py``,
- block-tridiagonal velocity factors in
  ``discretization/structured_velocity.py``.

``sfincs_jax/geometry/__init__.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry loading and normalized coefficient generation:

- analytic model fields,
- VMEC-derived coefficients,
- Boozer ``.bc`` evaluation,
- surface metrics and scalar geometry diagnostics.

``sfincs_jax/geometry/boozer.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Boozer ``.bc`` parser and geometryScheme 11/12 metric helper:

- header and surface-block parsing with SFINCS-v3 sign conventions,
- bracketing-surface selection and effective ``rN`` resolution,
- Fourier reconstruction of ``R``, ``Z``, ``Dz``, and their angular
  derivatives on the Boozer grid,
- ``gpsiHatpsiHat`` reconstruction used by output diagnostics and classical
  transport terms.

``sfincs_jax/geometry/vmec_wout.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC ``wout`` reader and radial interpolation contracts:

- netCDF ``wout`` parsing,
- mode/radius array convention normalization,
- half/full-mesh interpolation rules,
- SFINCS-compatible ``psiAHat`` and ripple-scale helpers,
- ``gpsiHatpsiHat`` reconstruction from VMEC Fourier coefficients using the
  same full/half-mesh derivative convention as the scheme-5 geometry evaluator.

``sfincs_jax/geometry/vmec.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC ``geometryScheme=5`` Fourier-sum evaluator. The public file path
``vmec_geometry_from_wout_file(...)`` reads ``wout`` data and delegates to
``vmec_geometry_from_wout(...)`` so optional in-memory producers can exercise the
same formulas and parity tests without hidden file I/O.

``sfincs_jax/discretization/v3.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SFINCS-v3-compatible grid construction, mapped speed-grid construction,
geometry loading, geometry cache keys, and the ``V3Grids`` data contract. This
replaces the historical root ``sfincs_jax/v3.py`` owner.

``sfincs_jax/operators/profile_collisionless.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streaming and mirror-force contributions in the Legendre basis.

``sfincs_jax/operators/profile_exb.py`` and ``electric_field.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`E\times B` terms in the kinetic operator, including angular advection and
the radial-electric-field contributions to :math:`\dot \xi` and :math:`\dot x` where
supported.

``sfincs_jax/operators/profile_magnetic_drifts.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Magnetic-drift coefficient construction, angular advection terms, upwinding masks, and
associated :math:`\partial_\xi` couplings.

``sfincs_jax/physics/collisions.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collision models:

- PAS,
- full linearized Fokker-Planck,
- field-particle terms,
- Phi1-modified collision coefficients.

``sfincs_jax/operators/profile_system.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

System construction:

- state-vector ordering,
- operator block composition,
- transport-RHS rewrites,
- cached operator application,
- matrix-free residual and JVP wrappers,
- constraint-source moment kernels for constraint schemes 1 and 2,
- system metadata used by the driver and diagnostics.

``sfincs_jax/operators/profile_layout.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=1 full, active, field-split, and compressed pitch-space layout
metadata. This is where Fortran-style active ``Nxi_for_x`` pitch prefixes,
tail blocks, active-to-full maps, block-COO storage, and layout preflight
helpers live before solver policies choose a numerical path.

When a solve behaves differently on CPU and GPU, inspect the problem owner
first: ``sfincs_jax.problems.profile_solve`` for RHSMode 1 and
``sfincs_jax.problems.transport_solve`` for RHSMode 2/3. The main domain
owners are:

- ``sfincs_jax/problems/profile_solver_diagnostics.py``:
  RHSMode=1 linear-solve and Newton-Krylov result dataclasses, final
  profile-response result wrapping, output-visible solver metadata, and bounded
  PETSc-style KSP residual-history replay.
- ``sfincs_jax/problems/transport_finalize.py``:
  RHSMode=2/3 transport-matrix result dataclass plus per-RHS finalization,
  constraint projection, residual bookkeeping, and KSP replay request contracts.
- ``sfincs_jax/solvers/krylov.py``:
  Krylov solve results, result finite-state checks, and XLA synchronization
  helpers around solver timing/profiling, plus small differentiable JAX-native
  linear algebra kernels such as the regularized tiny least-squares solve and
  recycled Krylov initial-guess builder used by RHSMode=1 and transport solves.
- ``sfincs_jax/solvers/preconditioning.py``:
  shared preconditioning state, setup utilities, RHSMode=1 preconditioner
  dispatch, and operator-shaping helpers.
  It owns passive cache dataclasses/global registries for RHSMode=1 and
  RHSMode=2/3 preconditioners, mutable solve-context hints for automatic
  preconditioner selection, sparse structural tolerance, factor dtype and
  solver-JIT admission, diagonal/block-diagonal reductions, point/line/
  domain-decomposition/Fortran-reduced ``Pmat`` builders, memory-bounded
  setup-column chunking, selected submatrix probing, stable array hashes, and
  RHSMode=1/transport preconditioner cache-key construction, and
  constraintScheme=1 nullspace/source-row projection used by RHSMode=1 and
  RHSMode=2/3 solves after iterative branches. The projection routines build
  the small particle/energy source correction basis, apply the roundoff skip
  gate used by transport solves, and return either the corrected state or the
  corrected residual through an injected operator action for direct numerical
  tests. The numerical setup/apply routines still live in the family owners;
  this module is the common state and shaping surface.
- ``sfincs_jax/solvers/preconditioner_pas_composite.py``:
  PAS-family RHSMode=1 composite preconditioner policy. It owns the
  ``pas_lite``, ``pas_hybrid``, and ``pas_schur`` composition rules, including
  angular/TZ applicability, line/truncated-:math:`L` selection, ``Er`` x-upwind
  routing, x-coarse correction, collision smoothing, the safety wrapper, and
  the ``RHS1PasFamilyBuilders`` dependency bundle used to build all public
  PAS-family variants. This package is the numerical owner; callers should use
  the profile-solve and solver-family modules directly.
- ``sfincs_jax/solvers/preconditioner_pas_angular.py``:
  PAS-only RHSMode=1 angular block-tridiagonal factors, including the
  tokamak-like theta/:math:`L` builder and the geometry-rich theta-zeta/:math:`L`
  builder. This module owns the block-Thomas factor setup, low-mode combined
  ``L=0,1`` singularity handling, optional structured velocity tail factors,
  PAS-TZ memory fallback invocation, cache population, and reduced/full apply
  wrappers. Fallback-builder wiring lives in the profile-response solve
  owner.
- ``sfincs_jax/solvers/preconditioner_pas_xblock_ilu.py``:
  sparse block-Jacobi ILU/LU setup for PAS-only RHSMode=1 operators. This
  module owns the per-``(species,x)`` Legendre/theta/zeta block assembly,
  PETSc-style ILU/exact-LU cutoff policy, padded triangular-factor conversion,
  threaded block factor setup, and extra-variable Schur solve. PAS-hybrid
  fallback injection lives in the profile-response solve owner.
- ``sfincs_jax/solvers/preconditioner_full_fp_kinetic.py``:
  RHSMode=1 collision-based, species-block, species-by-``(x,L)``,
  point-xdiag, and point-block kinetic preconditioners. The module owns PAS/FP
  diagonal collision inverses, FP species-``x`` and ``x`` block collision
  factors, full-species active block index maps, chunked unsharded operator
  probing, low-rank FP collision correction, PETSc-style point block probing,
  extra-variable tail solves, and
  reduced/full apply wrappers. Compatibility access is not an implementation
  owner.
- ``sfincs_jax/solvers/preconditioner_xblock_block_jacobi.py``:
  dense x-block Jacobi preconditioners for RHSMode=1, including
  per-``(species,x)`` blocks, the truncated-low-:math:`L` variant used by PAS
  and strong-fallback routes, and species/``x``-per-:math:`L` blocks. The module
  owns block slicing, active pitch-index maps, PAS chunk caps, chunked
  unsharded operator probing, extra-variable tail inversion, and identity
  passthrough for modes not covered by the truncated-:math:`L` factor.
- ``sfincs_jax/solvers/preconditioner_xblock_radial.py``:
  radial x-grid RHSMode=1 preconditioners, including the two-level additive
  x-multigrid approximation and the stable PAS+``Er`` x-upwind solve. The
  module owns coarse-x selection, Legendre-low-mode xDot coupling, upwind
  line-factor setup, cache population, and reduced/full apply wrappers.
  Compatibility access is not an implementation owner.
- ``sfincs_jax/solvers/preconditioner_domain_decomposition.py``:
  angular line-block and restricted-additive-Schwarz preconditioners for
  RHSMode=1 domain-decomposition and strong fallback paths. It owns the
  theta-line, zeta-line, theta-domain, zeta-domain, theta-Schwarz,
  zeta-Schwarz, theta-line-with-``x``-diagonal, and full theta-zeta
  angular-block setup/apply kernels used by automatic line selection,
  Schur-base construction, and explicit strong-preconditioner requests. Shared
  axis-line index maps, cache keys, regularization policy, extra-variable tail
  solves, and multi-level residual correction hooks live here. Compatibility
  access is not an implementation owner.
- ``sfincs_jax/solvers/preconditioner_schur_profile.py``:
  profile-response Schur and coarse preconditioners. It owns Schur
  base-preconditioner selection, constraint-source projection/injection,
  diagonal/full/x-coupled Schur inverse setup, active native-stack and
  sparse-coarse policy parsing, low-``L``/low-angular-mode coarse residual
  bases, targeted ``(species,x,L)`` window bases, full-CSR structured Schur
  result objects, Jacobi fallback, diagonal tail-Schur, zeta-line Schur,
  pitch-line Schur, radial-pitch Schur builders, block memory estimates, and
  regularized diagonal inversion. This flat module is the canonical owner for
  Schur and coarse profile-response preconditioners.
- ``sfincs_jax/problems/transport_linear_system.py``:
  RHSMode=2/3 transport active-system owner. It owns active-DOF and dense-path
  setup, dense-LU solver/preconditioner construction for bounded fallback
  paths, all-RHS dense-batch solves, host SciPy GMRES first-attempt/rescue
  solves, linear solve dispatch for explicit/JIT/implicit modes, active block
  ordering, bounded block-Schur factors, residual-coarse admission, direct
  reduced-``Pmat`` and exact active-operator emission, direct active
  block-Schur preconditioner setup, and the global full-FP Fortran-reduced
  sparse-factor preconditioner. The old active-dense, dense-LU, dense-batch,
  host-GMRES, active-factor, direct-``Pmat``, direct block-Schur, and
  Fortran-reduced LU implementation files are represented by this single
  linear-system owner.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 direct-tail structured-preconditioner adapter, direct reduced-Pmat
  aliases, stable cache-key hashing, cache-hit metadata tagging, and adaptive
  direct-tail memory-cap policy. Canonical debug scripts should import this
  owner directly when clearing the direct-tail cache or inspecting the policy.
- ``sfincs_jax/problems/profile_policies.py``:
  active-projected RHSMode=1 full-CSR preconditioner auto-policy. The module
  owns environment parsing for the candidate ladder, large-system fallback
  guard, skipped-fallback metadata, and progress logging default. Candidate
  dispatch and setup timing live in the profile-response sparse/solve owners.
- ``sfincs_jax/solvers/krylov_dispatch.py``:
  concrete Krylov solver routing for host-only SciPy methods, JIT/non-JIT JAX
  GMRES, distributed GMRES, diagnostic solver labels, and
  ``SFINCS_JAX_GMRES_DISTRIBUTED`` axis selection. Problem solve owners pass the
  selected solver callbacks through typed contexts; implementation code imports
  this owner directly.
- ``sfincs_jax/solvers/preconditioner_transport_matrix.py``:
  numerical builder implementations for the common RHSMode=2/3 transport
  preconditioners: collision diagonal, species/speed block, x-grid coarse
  correction, angular FFT/tridiagonal solve, point-block transport
  preconditioners, and the FP transport family: dense Fourier FP,
  block-Thomas Fourier line factors, Schur overlays, local-geometry line
  factors, x-block angular sparse LU, x-block Schur correction, and structured
  f-block LU. Orchestration-only wrapper names are exposed through the
  transport/profile-response solve owners when those owners still need
  compatibility wiring.
- ``sfincs_jax/solvers/preconditioner_pas_policy.py``:
  PAS applicability, PAS-TZ memory safety, PAS fallback routing, and PAS
  adaptive-smoother eligibility.
- ``sfincs_jax/problems/profile_policies.py``:
  typed RHSMode=1 solve-routing policy parsing for x-block probe-coarse,
  post-minres, post-coarse, post-residual-equation, bounded sparse-polish,
  host x-block factorization, current-backend dense/sparse admission wrappers,
  override semantics, and fail-closed high-resolution behavior. This keeps
  environment parsing and correction-policy defaults out of the solve entry
  point.
- ``sfincs_jax/problems/profile_setup.py``:
  pure setup helpers for RHSMode=1/profile-response solves: GMRES restart and
  max-iteration environment overrides, geometry/equilibrium progress hints,
  FP/PAS tolerance tightening, physics-flag normalization, solve-method lane
  classification, preconditioner-option admission, and domain-decomposition
  block/overlap parsing. It also owns the injected initial problem
  materialization step that builds or accepts the v3 operator, emits operator
  and RHS progress lines, installs preconditioner policy hints, applies
  transport ``whichRHS`` defaults, assembles the RHS, and returns the RHS norm.
  The driver consumes these typed setup results before entering the numerical
  solve loop.
- ``sfincs_jax/problems/profile_phi1_newton.py``:
  nonlinear Phi1 Newton-Krylov solve orchestration for RHSMode=1 profile
  response. This module owns the accepted-state history solve used by output
  writing, the small Newton-Krylov parity fixture path, active-DOF compaction,
  frozen-Jacobian mode selection, sparse-direct host rescue for non-autodiff
  runs, KSP-history replay wiring, and line-search advancement. Historical
  public names from this module are imported from the profile-response owner or
  from this module directly.
- ``sfincs_jax/problems/profile_preconditioner_build.py``:
  RHSMode=1/profile-response full and reduced preconditioner build
  orchestration. The solve owner passes solve-local builders and projection
  functions through typed contexts, and the helper returns explicit state for
  PAS-TZ guard metadata, collision fallback admission, and optional BiCGStab
  preconditioner reuse. It also owns RHSMode=1 strong-preconditioner family
  mapping, full/reduced strong fallback builders, PAS-Schur to PAS-hybrid build
  adjustment, ADI sweep parsing, and x-block TZ low-``l`` controls. It is also
  the canonical owner of the RHSMode=1 preconditioner registry and binding
  layer for dispatch, PAS-family builders, Schur binding,
  transport ``tzfft`` reuse, x-block builders, and strong fallback binding; the
  solve owner imports these names only as compatibility seams.
- ``sfincs_jax/operators/profile_fblock.py``:
  matrix-free kinetic f-block operator builder and matvec for RHSMode-1
  profile-response solves, including collisionless streaming, ExB, magnetic
  drift, Er, PAS, and Fokker-Planck terms. This replaces the historical root
  ``sfincs_jax/v3_fblock.py`` owner.
- ``sfincs_jax/problems/profile_dense.py``:
  RHSMode=1/profile-response dense and linear-solve helpers. This module owns
  Krylov routing for implicit, JIT, distributed, GMRES, and BiCGStab solve
  attempts; dense-KSP full/reduced solves; constraintScheme=0 PETSc-compatible
  sparse-ILU; host SciPy rescue; the reduced row-scaled LU path; and the
  full/reduced least-squares dense fallback used by non-differentiable host
  shortcut paths.
- ``sfincs_jax/problems/profile_solver_diagnostics.py``:
  typed RHSMode=1 x-block correction diagnostic records, solver
  metadata key assembly, and KSP replay diagnostic context forwarding. This
  keeps output-visible trace fields independently testable outside the
  monolithic solve path.
- ``sfincs_jax/problems/profile_solver_diagnostics.py``:
  final RHSMode=1/profile-response linear-solve diagnostics, output-visible solver
  metadata, bounded PETSc-style KSP residual-history replay, and iteration-count
  diagnostics. It applies cleanup projection, emits optional replay diagnostics,
  writes final residual and elapsed-time progress lines, applies post-xblock
  acceptance-floor metadata, wraps the result in ``V3LinearSolveResult``, and
  owns the bounded PETSc-style GMRES history replay for the optional
  Phi1/Newton-Krylov full-system path.
- ``sfincs_jax/solvers/preconditioner_domain_decomposition.py``:
  deterministic angular domain-decomposition patch ranges, shard-aware block
  sizing, and two-level Schwarz coarse-block heuristics. These rules are kept
  independent of the full operator so multi-device preconditioner policy can be
  tested without launching a solve.
- ``sfincs_jax/problems/profile_setup.py``:
  RHSMode=1 setup decisions, including active-degree-of-freedom routing and
  reduced-index-map construction for truncated pitch grids, x-block active-DOF
  opt-ins, and PAS constraint-projection solves. The same owner holds the
  reusable JAX primitives for full-to-reduced gathers, reduced-to-full
  one-based scatters, PAS ``l=0`` flux-surface-average projection, and final
  RHSMode=1 cleanup. These primitives are shared by RHSMode=1 sparse-PC,
  x-block active-DOF, PAS-projected reduced residual paths, and final linear
  solve normalization.
- ``sfincs_jax/problems/profile_residual.py``:
  small residual target, ratio, convergence, and host-scalar norm helpers used
  by RHSMode=1 sparse-PC and x-block diagnostics, plus the physics-aware
  x-block post-coarse direction builder and the bounded host/device subspace
  residual-equation correction kernels used after x-block solves. The module
  also owns residual-correction preconditioner composition, safe
  non-finite/clipped preconditioner wrapping, and scalar preconditioned-minres
  polish. This keeps fail-closed residual-polish algebra testable without
  entering the production driver.
- ``sfincs_jax/problems/profile_setup.py``:
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 preconditioner environment alias normalization plus bounded
  automatic preconditioner policy predicates for PAS, DKES, tokamak, GPU sparse
  fallback, weak-default PAS promotion, PAS-family refinement, FP/DKES routing,
  large-FP near-zero-Er overrides, and sharded line-overrides.
- ``sfincs_jax/solvers/preconditioner_schur_profile.py``:
  RHSMode=1 Schur base-preconditioner alias normalization and automatic
  geometry/PAS/DKES routing policy.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 profile-response solve-routing gates, including stage-2 triggers,
  sparse exact-LU admission, sparse-rescue ordering, sparse-polish budgets,
  post-x-block polish, large-PAS fast acceptance, host factor probes, and
  constraint-scheme-0 sparse/dense routing. It also owns current-backend
  wrappers used by the solve owner and small x-block/QI
  control helpers: guarded PAS-TZ
  structured-level parsing, QI device extra-coarse environment controls,
  QI probe minres-step selection, and safe x-block fallback initial-guess
  admission.
- ``sfincs_jax/problems/profile_preconditioner_build.py``:
  strong-preconditioner request mapping, enable/disable control, automatic
  strong-kind selection, and post-selection adjustment policy.
- ``sfincs_jax/problems/profile_preconditioner_build.py``:
  compatibility facade for RHSMode=1 strong-preconditioner fallback
  imports. The implementation owner is
  ``sfincs_jax/problems/profile_preconditioner_build.py``.
- ``sfincs_jax/problems/profile_solver_diagnostics.py``
  (absorbed owner for former profile-response finalization helpers):
  accepted-candidate replay, Krylov replay-state updates, final RHSMode=1
  solver diagnostics, and final linear-solve metadata. This is the
  source-mapped seam for the repeated RHSMode=1 driver pattern: compare a
  rescue/refinement candidate against the incumbent residual, apply optional
  measured solver-candidate gates, preserve the accepted residual vector, and
  update the KSP replay metadata only after a strict finite residual
  improvement.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 host dense fallback, host sparse-direct, sparse-preconditioned
  GMRES rescue, factor-dtype, explicit sparse-helper policy, and automatic
  solver/fallback admission.
- ``sfincs_jax/problems/profile_policies.py``:
  large explicit full-FP CPU sparse rescue, x-block seed, exact-LU promotion,
  host x-block assembly, and species-x-block rescue policy.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 profile-response admission, post-solve correction, solver-path,
  and implicit/differentiable solve-mode policy.
- ``sfincs_jax/solvers/path_policy.py``:
  pure solver/preconditioner path policy for JIT admission, RHSMode=1 rescue
  slack, DKES GMRES budget defaults, sparse-PC defaults, preconditioner dtype,
  backend resource-exhaustion classification, and measured candidate
  acceptance gates used by automatic solver/preconditioner promotions.
  Candidate gates require residual/parity checks and paired runtime/memory
  comparisons against an incumbent path.
- ``sfincs_jax/problems/transport_policies.py``:
  pure transport backend, sparse-direct, host-GMRES, dtype, recycle, polish,
  residual-abort threshold parsing and failure-message formatting,
  RHSMode=2/3 initial solve, active-DOF, dense fallback, low-memory output,
  streamed-diagnostic, state-vector retention, GMRES restart, per-``whichRHS``
  loop, preconditioner-kind normalization, auto-selection, DD/sparse-JAX env,
  and reduced/full preconditioner builder-dispatch policy. ``TransportRuntimePolicy``
  binds backend-sensitive decisions to the active JAX backend and host
  sparse-factor dtype provider. The former solver-replay, residual-quality,
  preconditioner-dispatch, and solve-policy relays have been deleted; tests
  import this owner directly.
- ``sfincs_jax/problems/transport_setup.py``:
  side-effect-light RHSMode=2/3 setup resolution for transport max-iteration
  overrides, optional Krylov state-file loading/merging, ``whichRHS`` subset
  normalization, CPU/GPU process-parallel worker requests, loop-local
  full/reduced matvec caches, bounded Krylov recycle bases, recycle-size
  admission, and sequential residual-gate/ETA progress bookkeeping. The driver
  emits the returned notes and keeps solve orchestration, while these setup and
  loop-state rules are covered by direct unit tests.
- ``sfincs_jax/outputs/transport.py``:
  RHSMode=2/3 transport output-schema helpers, host-side streaming
  accumulators, and streaming HDF5 writes. It owns the per-``whichRHS`` NumPy
  buffers, NTV/source handling, final output-field dictionary assembly, solver
  diagnostic arrays, derivative conversion factors, and low-memory HDF5 output
  path.
- ``sfincs_jax/problems/transport_finalize.py``:
  final RHSMode=2/3 transport solve bookkeeping and post-solve diagnostics. It
  recovers accepted full-space states, applies optional constraint projection,
  stores residual/KSP diagnostics, chooses streamed versus batched diagnostics,
  applies rematerialization/precompute/chunking policy, assembles
  species-by-``whichRHS`` flux arrays, and returns the transport matrix plus
  optional output fields.
- ``sfincs_jax/problems/transport_diagnostics.py``:
  JAX formulas for RHSMode=1 output moments, RHSMode=2/3 transport diagnostics,
  transport-matrix assembly, strict Fortran-order reductions, and cached
  geometry/species diagnostic precomputes.
- ``sfincs_jax/problems/transport_solve.py``:
  public RHSMode=2/3 transport solve orchestration. It builds one operator per
  requested ``whichRHS``, chooses active/full-space solve routing, calls the
  linear-system and preconditioner owners, applies retry/rescue policy, and
  finalizes accepted states and diagnostics. The module still owns the
  RHSMode=2/3 sparse-direct rescue implementation, including sparse-pattern
  admission, direct active FP operator factors, explicit sparse helper
  materialization, fallback sparse-ILU setup, host iterative refinement,
  float32 polish, and float64 retry. Dense-LU, host-GMRES, dense-batch, and
  loop-state support live in ``transport_linear_system.py`` and
  ``transport_setup.py``.
- ``sfincs_jax/problems/transport_finalize.py``:
  sequential RHSMode=2/3 per-``whichRHS`` finalization after a solver branch has
  accepted a candidate. It owns reduced/full state bookkeeping, optional
  constraint projection, true-residual recomputation, streamed-output
  collection, recycle-basis updates, and optional KSP iteration-stat dispatch.
  Dense fallback accepted-state overrides are explicit so the refactor preserves
  the established active-DOF branch behavior.
- ``sfincs_jax/problems/transport_parallel_runtime.py``:
  transport parallel backend policy, benchmark scaling audits, worker-count
  validation, XLA worker flag rewriting, RHS partitioning, injected-dependency
  payload normalization, child-worker guard setup, merge-ready result packing,
  GPU worker NPZ conversion, persistent process-pool caching, worker-environment
  setup, backend-specific execution/retry/fallback, GPU subprocess launch,
  parent-side merge of per-worker state/residual/elapsed-time results, and
  single-case sharded-solve planning metadata. This module absorbed the old
  policy, sharding, payload, pool, execution, solve, and validation micro-files
  so transport parallelism has one canonical runtime owner.
- ``sfincs_jax/problems/transport_parallel_runtime.py``
  (worker CLI entry point):
  command-line worker entry point used by GPU transport subprocesses. The
  maintained path is ``python -m sfincs_jax.problems.transport_parallel_runtime``,
  which reads a worker payload, runs the transport solve lazily, and writes the
  merge-ready NPZ schema.
- ``sfincs_jax/validation/artifacts.py``:
  lightweight loaders and physics metrics for checked-in publication artifacts. This
  module is independent of the heavy solver path, so documentation and CI can verify
  collisionality, high-collisionality trend, trajectory-sweep, and dashboard artifacts
  plus frozen CPU/GPU Fortran-suite benchmark summaries without rerunning large scans
  or example-suite audits. It also owns the fail-closed schema validator for the
  Fortran-v3 vs SFINCS-JAX runtime/memory benchmark summary consumed by README/docs
  plots.
- ``sfincs_jax/validation/fortran.py``:
  Fortran-v3 execution/profiling helpers and PETSc binary fixture readers used by
  parity tests, diagnostic comparison scripts, and pedagogical examples. Keeping
  ``read_petsc_vec`` and ``read_petsc_mat_aij`` beside the Fortran runner makes
  frozen-reference ownership explicit and avoids a separate tiny validation module.
- ``sfincs_jax/problems/profile_phi1_newton.py``:
  Phi1 Newton policy, bounded nonlinear linear-step orchestration, accepted-
  iterate update logic, and solve orchestration for the Newton path, including
  active-DOF mode selection, restart sizing, frozen-Jacobian cache policy,
  line-search policy, reduced/full routing, sparse-direct entry, KSP-history
  emission, retry-without-preconditioner, PETSc-like backtracking,
  fixed-candidate ``best`` search, and finite-state fallback handling.
- ``sfincs_jax/solvers/diagnostics.py``:
  solver-neutral diagnostics and observability support: user-facing duration
  formatting, coarse runtime hints, one-shot large RHSMode=1 progress messages,
  transport whichRHS ETA text, fixed-shape Krylov state signatures and
  warm-start files, portable solver-trace JSON/HDF5 records, and compact
  Fortran-v3/SFINCS-JAX solver-profile comparisons. These utilities improve
  reproducibility and CLI progress without changing numerical decisions.
- ``sfincs_jax/profiling.py``:
  opt-in coarse solver/output profiling behind ``SFINCS_JAX_PROFILE``. It owns
  phase-level timing, RSS high-water sampling, optional JAX device-memory polling,
  and the ``profile_entries`` payload written into solver traces and output metadata.
- ``sfincs_jax/validation/artifacts.py``:
  fast schema, provenance, and release-blocking classification policy for checked-in
  benchmark JSON artifacts.
- ``sfincs_jax/solvers/memory_model.py``:
  conservative dense/CSR/Krylov/preconditioner memory estimates used by solver
  restart caps, benchmark manifests, and measured solver-candidate gates. This is
  the preflight layer that keeps future memory-saving defaults testable before
  expensive operators or preconditioners are materialized.
- ``sfincs_jax/problems/profile_policies.py``:
  tested admission gates for RHSMode=1 host dense, sparse-host, constrained-PAS
  sparse-PC, CPU 3D full-FP sparse-PC, and GPU tokamak full-FP no-Er/Er
  sparse-PC auto lanes. These helpers keep solver path promotion rules explicit
  and unit-testable without assembling a kinetic operator.

``sfincs_jax/solvers/krylov.py`` and ``sfincs_jax/solvers/implicit.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear-algebra infrastructure:

- Krylov wrappers,
- host-direct and sparse rescues,
- differentiable linear solves,
- traced-safe implicit-solve method resolution that maps host-only SciPy Krylov
  requests to JAX-native incremental GMRES inside autodiff paths,
- JAX-native linear solve utilities,
- augmented FGMRES hooks that reuse a checked coarse basis and stored operator
  action ``(U, A U)`` without assembling a dense global operator.

``sfincs_jax/problems/transport_diagnostics.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=2/3 postprocessing and transport-matrix assembly. New code should
import this owner or use the public API/CLI.

``sfincs_jax/diagnostics.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Moment integrals, flux-surface-averaged outputs, classical transport diagnostics, and
other quantities that end up in ``sfincsOutput.h5``.

Where the main equations live
-----------------------------

The conceptual mapping is:

- drift-kinetic model:
  :doc:`physics_models`, :doc:`system_equations`, :doc:`physics_reference`
- discretization:
  :doc:`method`, :doc:`numerics`
- geometry coefficients:
  :doc:`geometry`
- solve stack:
  ``sfincs_jax/problems/profile_solve.py``,
  ``sfincs_jax/problems/transport_solve.py``,
  ``sfincs_jax/solvers`` and its flat ``preconditioner_*.py`` modules
- outputs and diagnostics:
  :doc:`outputs`, ``sfincs_jax/io.py``, ``sfincs_jax/diagnostics.py``

Tests that protect each layer
-----------------------------

The repository intentionally tests the code at several levels:

- unit tests for geometry, collision, and solve heuristics,
- parity/regression tests against frozen reference outputs,
- end-to-end output-writing tests,
- benchmark smoke tests for parallel and performance tooling.

See :doc:`testing` for the validation strategy and the most relevant test files.
