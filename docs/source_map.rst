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

Active package migration
------------------------

The architecture branch is migrating the historical flat module layout into
domain packages. New behavior should be placed by physics or numerical
responsibility, not by historical helper prefixes. The current importable
skeleton packages are:

- ``sfincs_jax/input`` for namelist parsing, input normalization, compatibility
  translation, defaults, and option validation.
- ``sfincs_jax/physics`` for drift-kinetic terms, collisions, ambipolarity,
  bootstrap-current normalization, and analytic validation formulas.
- ``sfincs_jax/discretization`` for grids, quadrature, basis functions,
  indexing/layout, sparse stencils, and active degrees of freedom.
- ``sfincs_jax/operators`` for matrix-free and assembled DKE operators,
  residual/source assembly, sparse patterns, and SFINCS Fortran-v3 convention
  translation.
- ``sfincs_jax/problems/profile_response`` for RHSMode=1 profile-current and
  bootstrap-current problem orchestration.
- ``sfincs_jax/problems/transport_matrix`` for RHSMode=2/3 transport-matrix and
  monoenergetic-response orchestration, diagnostics, output-field assembly, and
  compatibility shims for historical flat transport modules.
- ``sfincs_jax/solvers`` and ``sfincs_jax/solvers/preconditioners`` for reusable
  Krylov, sparse/direct, residual-gate, implicit-differentiation, and
  preconditioning machinery.
- ``sfincs_jax/parallel`` for CPU/GPU process, sharding, worker payload, and
  scaling utilities.
- ``sfincs_jax/outputs`` for output-schema-adjacent helpers and flat
  HDF5/NetCDF/NPZ file-format readers and writers.
- ``sfincs_jax/workflows`` for optimization, VMEC-JAX/Boozer/SFINCS-JAX
  pipelines, scans, and publication figures.
- ``sfincs_jax/validation`` and ``sfincs_jax/benchmarks`` for physics gates,
  parity reports, benchmark schemas, and release artifact readers.
- ``sfincs_jax/compat`` for legacy import shims and external-code comparison
  helpers.

``sfincs_jax/geometry`` is now a real package owner for analytic Boozer,
Boozer-file, VMEC, and JAX-native geometry adapters. ``sfincs_jax/io.py``
intentionally remains a module for now because that import path is already part
of the active public code; the ``io`` package name is reserved for a later
migration step that moves the implementation without shadowing or breaking
existing imports.

``sfincs_jax/discretization`` now also owns the former flat speed-grid,
indexing, sparse-stencil, and structured velocity kernels:
``adaptive_maps.py``, ``indices.py``, ``periodic_stencil.py``,
``structured_velocity.py``, and ``xgrid.py``. Public examples and tests import
these modules through the package owner, not through root compatibility shims.

``sfincs_jax/operators/profile_response`` now owns the former flat
profile-response operator kernels: collisionless streaming, radial electric
field terms, ExB terms, magnetic drifts, and matrix-free linear-system
residual wrappers. These are imported through the operator package, not through
root compatibility shims.

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
     - HDF5/output comparison helpers used by examples and validation tools.
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
   * - ``classical_transport.py``
     - stable physics kernel
     - Classical transport formulas and validation gates.
   * - ``collisions.py``
     - stable physics kernel
     - Collision-operator formulas shared by profile-response and transport solves.
   * - ``constrained_pas_branch.py``
     - stable solver-policy kernel
     - Constraint-aware PAS branch policy guarded by focused tests.
   * - ``constraint_projection.py``
     - stable numerical kernel
     - Constraint projection used by RHSMode=1 and transport solves.
   * - ``diagnostics.py``
     - stable physics kernel
     - Flux-surface averages, moment integrals, and output diagnostics.
   * - ``grids.py``
     - public discretization API
     - Velocity/grid helpers used directly by docs and tests.
   * - ``host_refinement.py``
     - stable solver-policy kernel
     - Host-side iterative refinement gates.
   * - ``pas_smoother.py``
     - stable preconditioner kernel
     - PAS smoother formulas used by preconditioner owners.
   * - ``paths.py``
     - stable support utility
     - Repository/data path resolution helpers.
   * - ``phi1_newton_linear.py``
     - stable solver kernel
     - Phi1 Newton linear-step orchestration.
   * - ``phi1_newton_policy.py``
     - stable solver-policy kernel
     - Phi1 Newton admission and line-search policy.
   * - ``profiling.py``
     - stable support utility
     - Phase timing and optional memory profiling helpers.
   * - ``solver.py``
     - stable solver kernel
     - Krylov result contracts, XLA synchronization, and linear algebra utilities.
   * - ``v3_driver.py``
     - compatibility shim
     - Historical import path; implementation lives in domain owners and this file must remain tiny.

Closure move/delete manifest
----------------------------

Closure Phase 1 locks a move/delete decision for every package-root file before
any more code movement. This prevents one-helper churn: a later phase may move a
file only if the move follows the owner below, deletes the root implementation
or keeps a documented compatibility shim, and passes the corresponding owner
tests.

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
   * - ``api.py``
     - package root public API
     - keep at root
   * - ``cli.py``
     - package root CLI entry point
     - keep at root
   * - ``ambipolar.py``
     - problems.ambipolar via public API facade
     - keep root shim until public docs/examples migrate
   * - ``compare.py``
     - validation comparison API
     - move only after examples/scripts use validation owner
   * - ``input_compat.py``
     - input compatibility owner
     - keep root public compatibility shim until input package exports cover callers
   * - ``io.py``
     - outputs writer/formats/cache owners
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
   * - ``classical_transport.py``
     - physics classical transport owner
     - move only with physics API export tests
   * - ``collisions.py``
     - physics/operators collision owner
     - move only with collision API export tests
   * - ``constrained_pas_branch.py``
     - solvers/preconditioners PAS policy owner
     - move in solver-policy group if no public shim is needed
   * - ``constraint_projection.py``
     - solvers constraint-projection owner
     - move only after transport/profile imports use solver owner
   * - ``diagnostics.py``
     - physics/output diagnostics owner
     - defer until diagnostics API split is explicit
   * - ``grids.py``
     - discretization public grid owner
     - keep root public helper until discretization package exports are documented
   * - ``host_refinement.py``
     - solvers refinement policy owner
     - move in solver-policy group if profile-response imports migrate
   * - ``pas_smoother.py``
     - solvers/preconditioners PAS smoother owner
     - move in solver-preconditioner group
   * - ``paths.py``
     - package root path support utility
     - keep at root unless a support package is introduced with broad import rewrite
   * - ``phi1_newton_linear.py``
     - problems.profile_response Phi1 Newton owner
     - move if it deletes root file without adding shim
   * - ``phi1_newton_policy.py``
     - problems.profile_response Phi1 policy owner
     - move if it deletes root file without adding shim
   * - ``profiling.py``
     - solvers/validation profiling support
     - defer until profiling API boundary is explicit
   * - ``solver.py``
     - solvers public contracts owner
     - keep root shim until solvers exports cover public contracts
   * - ``v3_driver.py``
     - compatibility shim to problem owners
     - delete after tests/examples stop importing sfincs_jax.v3_driver

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
  facades for common Python workflows that should not import legacy internals
  directly.

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

``sfincs_jax/io.py``
^^^^^^^^^^^^^^^^^^^^

Input/output orchestration:

- reads namelists,
- resolves equilibrium overrides (including ``wout_path``),
- materializes output diagnostics,
- exposes the in-memory results API,
- keeps legacy compatibility aliases for flat output-format helpers now owned by
  ``sfincs_jax.outputs.formats``.

``sfincs_jax/outputs/formats.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flat output file-format helpers:

- HDF5, NetCDF, and NPZ readers/writers,
- SFINCS Fortran-compatible HDF5 layout conversion,
- output suffix dispatch,
- NetCDF-safe dataset naming,
- solver-trace attachment for HDF5 outputs.

``sfincs_jax/outputs/cache.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry-output cache helpers:

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

The solved-field physics schema construction still lives in ``sfincs_jax/io.py``
during the current refactor tranche. The next I/O split should move physical
solved-field and provenance schema construction behind a smaller output
contract while preserving these writer functions and the output-cache boundary
in the ``outputs`` package.

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
- public ``run_er_scan`` helper used by CLI, examples, and validation scripts.

This module replaces the former root ``sfincs_jax/scans.py`` implementation.

``sfincs_jax/workflows/postprocess_upstream.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Upstream utility postprocessing wrapper:

- locates vendored or user-provided SFINCS Fortran-v3 ``utils`` scripts,
- runs plotting/postprocessing scripts in non-interactive mode,
- forces a non-GUI matplotlib backend for scripted examples.

This module replaces the former root ``sfincs_jax/postprocess_upstream.py``
implementation.

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

Compatibility and search-order logic for equilibrium files, input normalization, and
user overrides. This is the module to inspect first when a case fails to find a VMEC or
Boozer file.

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
- Fortran-v3 active indexing in ``discretization/indices.py``,
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

``sfincs_jax/geometry/vmec_wout.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC ``wout`` reader and radial interpolation contracts:

- netCDF ``wout`` parsing,
- mode/radius array convention normalization,
- half/full-mesh interpolation rules,
- SFINCS-compatible ``psiAHat`` and ripple-scale helpers.

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

``sfincs_jax/operators/profile_response/collisionless.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streaming and mirror-force contributions in the Legendre basis.

``sfincs_jax/operators/profile_response/exb.py`` and ``electric_field.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`E\times B` terms in the kinetic operator, including angular advection and
the radial-electric-field contributions to :math:`\dot \xi` and :math:`\dot x` where
supported.

``sfincs_jax/operators/profile_response/magnetic_drifts.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Magnetic-drift coefficient construction, angular advection terms, upwinding masks, and
associated :math:`\partial_\xi` couplings.

``sfincs_jax/collisions.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collision models:

- PAS,
- full linearized Fokker-Planck,
- field-particle terms,
- Phi1-modified collision coefficients.

``sfincs_jax/operators/profile_response/system.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

System construction:

- state-vector ordering,
- operator block composition,
- transport-RHS rewrites,
- cached operator application,
- system metadata used by the driver and diagnostics.

``sfincs_jax/operators/profile_response/linear_systems.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Residual and source-term helpers. This is where the thermodynamic drives and other RHS
pieces are assembled before being fed to the solve stack.

``sfincs_jax/v3_driver.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Top-level solve orchestration. This file controls:

- solver selection,
- preconditioner selection,
- bounded rescue paths,
- post-Krylov residual-equation corrections that reuse cached QI ``(U, A U)``
  columns,
- transport-worker parallelism,
- sharded experimental paths,
- output-field collection.

When a solve behaves differently on CPU and GPU, this is usually the first file to
inspect.

On the active refactor branch, the main policy layers are being split out of the
monolith into narrower modules while keeping ``v3_driver.py`` as the stable public seam
for debugging and monkeypatch-based tests. The first extracted layers are:

Refactor rule for new extractions: keep a driver wrapper only when it injects
driver-local dependencies, adapts a signature, preserves live monkeypatch behavior,
or bridges a public debugging seam. Otherwise, import the extracted function under
the historical private driver name and test the focused module directly. This keeps
``v3_driver.py`` shrinking without replacing monolithic code with wrapper clutter.

- ``sfincs_jax/problems/profile_response/solver_diagnostics.py``:
  RHSMode=1 linear-solve and Newton-Krylov result dataclasses, final
  profile-response result wrapping, output-visible solver metadata, and bounded
  PETSc-style KSP residual-history replay.
- ``sfincs_jax/problems/transport_matrix/finalize.py``:
  RHSMode=2/3 transport-matrix result dataclass plus per-RHS finalization,
  constraint projection, residual bookkeeping, and KSP replay request contracts.
- ``sfincs_jax/solver.py``:
  Krylov solve results, result finite-state checks, and XLA synchronization
  helpers around solver timing/profiling, plus small differentiable JAX-native
  linear algebra kernels such as the regularized tiny least-squares solve and
  recycled Krylov initial-guess builder used by RHSMode=1 and transport solves.
- ``sfincs_jax/solvers/preconditioning.py``:
  shared preconditioning state, setup utilities, and operator-shaping helpers.
  It owns passive cache dataclasses/global registries for RHSMode=1 and
  RHSMode=2/3 preconditioners, mutable solve-context hints for automatic
  preconditioner selection, sparse structural tolerance, factor dtype and
  solver-JIT admission, diagonal/block-diagonal reductions, point/line/
  domain-decomposition/Fortran-reduced ``Pmat`` builders, memory-bounded
  setup-column chunking, selected submatrix probing, stable array hashes, and
  RHSMode=1/transport preconditioner cache-key construction. The numerical
  setup/apply routines still live in the family owners; this module is the
  common state and shaping surface.
- ``sfincs_jax/constraint_projection.py``:
  constraintScheme=1 nullspace/source-row projection used by RHSMode=1 and
  RHSMode=2/3 solves after iterative branches. It builds the small
  particle/energy source correction basis, applies the roundoff skip gate used
  by transport solves, and returns either the corrected state or the corrected
  residual through an injected operator action for direct numerical tests.
- ``sfincs_jax/solvers/sparse_triangular.py``:
  JAX-native triangular solves for padded and compact-CSR sparse factor rows,
  plus permutation inversion. These pure kernels are used by sparse
  preconditioner apply paths and are directly tested against dense triangular
  references.
- ``sfincs_jax/solvers/preconditioners/pas/composite.py``:
  PAS-family RHSMode=1 composite preconditioner policy. It owns the
  ``pas_lite``, ``pas_hybrid``, and ``pas_schur`` composition rules, including
  angular/TZ applicability, line/truncated-:math:`L` selection, ``Er`` x-upwind
  routing, x-coarse correction, collision smoothing, the safety wrapper, and
  the ``RHS1PasFamilyBuilders`` dependency bundle used to build all public
  PAS-family variants. ``v3_driver.py`` now keeps compatibility wrappers only:
  it supplies the current low-level builders for monkeypatch-based debug
  workflows and delegates family composition back to this package.
- ``sfincs_jax/solvers/preconditioners/pas/angular.py``:
  PAS-only RHSMode=1 angular block-tridiagonal factors, including the
  tokamak-like theta/:math:`L` builder and the geometry-rich theta-zeta/:math:`L`
  builder. This module owns the block-Thomas factor setup, low-mode combined
  ``L=0,1`` singularity handling, optional structured velocity tail factors,
  PAS-TZ memory fallback invocation, cache population, and reduced/full apply
  wrappers. ``v3_driver.py`` injects fallback builders and keeps compatibility
  wrappers only.
- ``sfincs_jax/solvers/preconditioners/pas/xblock_ilu.py``
  (historical location: ``sfincs_jax/rhs1_pas_xblock_ilu.py``):
  sparse block-Jacobi ILU/LU setup for PAS-only RHSMode=1 operators. This
  module owns the per-``(species,x)`` Legendre/theta/zeta block assembly,
  PETSc-style ILU/exact-LU cutoff policy, padded triangular-factor conversion,
  threaded block factor setup, and extra-variable Schur solve. ``v3_driver.py``
  keeps a compatibility wrapper only to inject the current PAS-hybrid fallback.
- ``sfincs_jax/solvers/preconditioners/full_fp/species_blocks.py``:
  species-block and species-by-``(x,L)`` block-Jacobi preconditioners for
  RHSMode=1. The module owns the active block index maps, chunked unsharded
  operator probing, block inverse construction, extra-variable tail inverse,
  and JAX apply kernels. ``v3_driver.py`` keeps compatibility wrappers only.
- ``sfincs_jax/solvers/preconditioners/full_fp/kinetic_blocks.py``:
  RHSMode=1 collision-based, point-xdiag, and point-block kinetic
  preconditioners. The module owns PAS/FP diagonal collision inverses, FP
  species-``x`` and ``x`` block collision factors, low-rank FP collision
  correction, PETSc-style point block probing, extra-variable tail solves, and
  reduced/full apply wrappers. ``v3_driver.py`` keeps compatibility wrappers
  only.
- ``sfincs_jax/solvers/preconditioners/full_fp/structured_fblock.py``:
  structured full-Fokker-Planck RHSMode=1 f-block preconditioners. The module
  owns block-Jacobi, angular-line, pitch-angular, FP-radial grouped factors,
  and low-mode/moment/tail Schur correction builders over the structured
  f-block operator. Same-shape cache keys, metadata emission, memory guards,
  and matrix-free residual-correction composition live here; ``v3_driver.py``
  keeps compatibility wrappers only.
- ``sfincs_jax/solvers/preconditioners/xblock/block_jacobi.py``:
  dense x-block Jacobi preconditioners for RHSMode=1, including
  per-``(species,x)`` blocks, the truncated-low-:math:`L` variant used by PAS
  and strong-fallback routes, and species/``x``-per-:math:`L` blocks. The module
  owns block slicing, active pitch-index maps, PAS chunk caps, chunked
  unsharded operator probing, extra-variable tail inversion, and identity
  passthrough for modes not covered by the truncated-:math:`L` factor.
- ``sfincs_jax/solvers/preconditioners/xblock/radial.py``:
  radial x-grid RHSMode=1 preconditioners, including the two-level additive
  x-multigrid approximation and the stable PAS+``Er`` x-upwind solve. The
  module owns coarse-x selection, Legendre-low-mode xDot coupling, upwind
  line-factor setup, cache population, and reduced/full apply wrappers.
  ``v3_driver.py`` keeps compatibility wrappers only.
- ``sfincs_jax/solvers/preconditioners/xblock/tz_sparse.py``
  (historical location: ``sfincs_jax/rhs1_xblock_tz_sparse.py``):
  sparse per-``x`` RHSMode=1 full-FP preconditioner setup. This module owns the
  host/JAX x-block LU/ILU policy, compact CSR/padded triangular-factor apply,
  selected theta/zeta upwind sparse-stencil assembly, explicit FP assembled-host
  cache, host-assembly admission policy, per-block sparse matrix/diagonal
  assembly, sparse per-:math:`L` species/``x`` host rescue factors, one-shot
  sparse species/``x`` seed construction, skipped-block diagonal fallback,
  host-factor probe/cache-key policy, shared chunked unsharded matrix probing,
  and extra-variable Schur solve. ``v3_driver.py`` keeps compatibility wrappers
  only.
- ``sfincs_jax/solvers/preconditioners/xblock/low_l_schur.py``:
  low-pitch x-block Schur preconditioners for exact RHSMode=1 full-CSR systems.
  This module owns the opt-in native ``x_ell`` kinetic factor, native
  ``x_ell`` plus dense-tail Schur factor, sparse low-``ell`` ``(theta,zeta)``
  x-block factor, physics low-mode coarse residual correction, and the shared
  low-``ell`` x-block index helper. ``rhs1_full_assembly.py`` keeps only
  compatibility aliases plus dispatch/admission logic around these builders.
- ``sfincs_jax/solvers/preconditioners/xblock/active_projected.py``:
  active-projected x-block, diagonal-Schur, x-ell kinetic-line, angular-line,
  native indexed Schwarz, restricted-additive-Schwarz, global field-split,
  multiline field-split, bounded native-stack, and Fortran-v3-reduced
  native-stack preconditioners for exact RHSMode=1 active CSR systems. This
  module owns active full-index to projected-position mapping, active x-block
  sparse LU/ILU residual correction, optional block scaling, singular block
  fallback metadata, compact active kinetic/angular line inverses, native
  padded-indexed block factors, active global-tail Schur setup,
  overlap-Schwarz patch setup, local base dispatch for extracted x-block
  families, and the bounded line/patch/coarse native-stack architecture.
  ``rhs1_full_assembly.py`` keeps only compatibility aliases plus
  dispatch/admission logic around these builders.
- ``sfincs_jax/solvers/preconditioners/domain_decomposition/__init__.py``:
  angular line-block and restricted-additive-Schwarz preconditioners for
  RHSMode=1 domain-decomposition and strong fallback paths. It owns the
  theta-line, zeta-line, theta-domain, zeta-domain, theta-Schwarz,
  zeta-Schwarz, theta-line-with-``x``-diagonal, and full theta-zeta
  angular-block setup/apply kernels used by automatic line selection,
  Schur-base construction, and explicit strong-preconditioner requests. Shared
  axis-line index maps, cache keys, regularization policy, extra-variable tail
  solves, and multi-level residual correction hooks live here; ``v3_driver.py``
  keeps compatibility wrappers only.
- ``sfincs_jax/solvers/preconditioners/schur/profile_response.py``:
  profile-response Schur and coarse preconditioners. It owns Schur
  base-preconditioner selection, constraint-source projection/injection,
  diagonal/full/x-coupled Schur inverse setup, active native-stack and
  sparse-coarse policy parsing, low-``L``/low-angular-mode coarse residual
  bases, targeted ``(species,x,L)`` window bases, full-CSR structured Schur
  result objects, Jacobi fallback, diagonal tail-Schur, zeta-line Schur,
  pitch-line Schur, radial-pitch Schur builders, block memory estimates, and
  regularized diagonal inversion. The package-level
  ``sfincs_jax.solvers.preconditioners.schur`` import re-exports this stable
  owner; the historical ``rhs1*`` Schur implementation files were removed.
- ``sfincs_jax/problems/transport_matrix/linear_system.py``:
  RHSMode=2/3 transport active-system owner. It owns active-DOF and dense-path
  setup, active block ordering, bounded block-Schur factors, residual-coarse
  admission, direct reduced-``Pmat`` and exact active-operator emission,
  direct active block-Schur preconditioner setup, and the global full-FP
  Fortran-reduced sparse-factor preconditioner. The old active-dense,
  active-factor, direct-``Pmat``, direct block-Schur, and Fortran-reduced LU
  implementation files were absorbed here so transport linear-system logic has
  one review surface.
- ``sfincs_jax/solvers/explicit_sparse.py``:
  explicit-sparse host-factor environment parsing, canonical factor-kind alias
  resolution, monolithic LU/ILU guard sizing, and the typed
  ``ExplicitSparseFactorSettings`` bundle, dense/CSR storage decisions,
  pattern-color probing, symbolic Schur/frontal/ND/BLR settings, SuperLU
  pivot/permutation options, ILU options, host explicit-sparse operator
  assembly, logging, monolithic preflight guard, and factorization
  orchestration. ``v3_driver.py`` injects its current
  ``build_operator_from_matvec``, ``build_operator_from_pattern``,
  ``factorize_host_sparse_operator``, and backend callbacks so existing
  monkeypatch/debug workflows keep exercising the same runtime seam. The old
  explicit-sparse policy and builder support files were absorbed here so the
  explicit sparse host-factor lane has one review surface.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/host_factor.py``:
  RHSMode=1 host sparse ILU/LU factor setup used by non-differentiable
  CLI-oriented rescue paths. The module owns matrix-free column assembly,
  structural-threshold application, SuperLU retry/regularization policy, cached
  dense/JAX triangular-factor materialization, and the matrix-free full-system
  adapter used by coarse/Galerkin corrections. ``v3_driver.py`` imports the
  historical private helper names as aliases so existing debugging and
  monkeypatch tests keep using the same seam.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/profile_response.py``:
  Fortran-v3-style reduced active sparse factors for RHSMode=1. The module owns
  reduced active matrix construction, support-mode parsing and preflight,
  symbolic-plan permutation, sparse equilibration, LU/ILU memory admission, and
  SuperLU/RCM factor setup for the non-differentiable host CSR lane.
  ``rhs1_full_assembly.py`` now imports the historical private names as
  compatibility aliases and keeps only direct-Pmat emission plus surrounding
  active-preconditioner dispatch.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/active_factors.py``:
  active-projected RHSMode=1 sparse-factor preconditioners. The module owns the
  global active sparse factor, row/column-equilibrated active factor, and
  physics-filtered active sparse factor that retains selected off-diagonal
  kinetic couplings. These are host-side, non-differentiable preconditioner
  setup routines for explicit CSR solves; ``rhs1_full_assembly.py`` imports the
  historical private builder names as aliases and keeps candidate dispatch.
- ``sfincs_jax/problems/profile_response/policies.py``
  (historical location: ``sfincs_jax/rhs1_direct_tail_policy.py``):
  RHSMode=1 direct-tail structured-preconditioner adapter, direct reduced-Pmat
  aliases, stable cache-key hashing, cache-hit metadata tagging, and adaptive
  direct-tail memory-cap policy. ``v3_driver.py`` imports the same private
  compatibility names so existing debug scripts can still clear the direct-tail
  cache or inspect the policy through the historical driver namespace.
- ``sfincs_jax/operators/profile_response/reduced_tail.py``
  (historical location: ``sfincs_jax/rhs1_fortran_reduced_direct_tail.py``):
  RHSMode=1 Fortran-reduced constraintScheme=1 direct-tail sparse-operator
  materialization. The module emits source/tail columns and moment rows from the
  same formulas used by the matrix-free v3 operator, while ``v3_driver.py``
  injects the structured full-CSR builder callback to preserve the existing
  monkeypatch/debug seam and avoid circular imports.
- ``sfincs_jax/problems/profile_response/policies.py``
  (historical location: ``sfincs_jax/rhs1_active_preconditioner_policy.py``):
  active-projected RHSMode=1 full-CSR preconditioner auto-policy. The module
  owns environment parsing for the candidate ladder, large-system fallback
  guard, skipped-fallback metadata, and progress logging default, leaving
  ``rhs1_full_assembly.py`` to dispatch candidates and measure setup results.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/policy.py``
  (historical location: ``sfincs_jax/rhs1_fortran_reduced_factor_policy.py``):
  Fortran-v3-reduced RHSMode=1 active-Pmat factorization policy. The module
  owns factor-kind normalization, large-matrix ILU guards, LU prefill safety
  defaults, SuperLU/RCM ordering candidates, equilibration norm selection, and
  progress logging defaults; the symbolic-sparse RHSMode=1 Fortran-reduced
  module consumes this policy and performs the numerical sparse factor setup.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/policy.py``
  (historical location: ``sfincs_jax/rhs1_symbolic_frontal_policy.py``):
  symbolic frontal/Schur RHSMode=1 active-preconditioner policy. The module
  owns frontal versus nested-dissection routing, separator/block limits, dense
  Schur update budgets, admission probe thresholds, and ND residual-polish
  controls; ``rhs1_full_assembly.py`` keeps sparse symbolic analysis,
  factorization, and true residual admission.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/policy.py``
  (historical location: ``sfincs_jax/rhs1_symbolic_sparse_policy.py``):
  symbolic superblock and separator-Schur RHSMode=1 active-preconditioner
  policy. The module owns grouped-block and block-Schur size gates, ordering
  defaults, separator/coarse limits, retained-cross-fraction gates, prefill
  safety factors, and admission probe thresholds; ``rhs1_full_assembly.py``
  keeps symbolic sparse analysis, host factorization, and true residual
  admission.
- ``sfincs_jax/operators/profile_response/structured_csr.py``
  (historical location: ``sfincs_jax/rhs1_structured_full_csr.py``):
  runtime/non-autodiff wrapper that adapts analytic RHSMode=1 full-CSR assembly
  from ``rhs1_full_assembly.py`` into the ``SparseOperatorBundle`` contract used
  by sparse-PC solver paths. Unsupported or over-budget cases return ``None`` so
  callers can fall back to the established matrix-free or pattern-probed path.
- ``sfincs_jax/operators/profile_response/true_operator_rescue.py``
  (historical location: ``sfincs_jax/rhs1_true_operator_rescue.py``):
  support bundles and low-level helpers for RHSMode=1 true-operator
  residual-window, active-submatrix, coupled-coarse, and residual-coarse rescue
  preconditioners. The module owns the reusable true-action column cache,
  sparse-factor storage estimator, additive-rescue budget accounting, graph
  expansion, residual-window target parsing, and residual-driven window
  selection. It also owns the residual sparse-window/coarse builders, true-
  operator residual-window LSQ, active-block LSQ, active-residual-block LSQ,
  active-submatrix, coupled-coarse builders, and active residual diagnostic
  summaries; ``v3_driver.py`` imports the historical private names for
  compatibility.
- ``sfincs_jax/solvers/krylov_dispatch.py``:
  concrete Krylov solver routing for host-only SciPy methods, JIT/non-JIT JAX
  GMRES, distributed GMRES, diagnostic solver labels, and
  ``SFINCS_JAX_GMRES_DISTRIBUTED`` axis selection. The driver passes its current
  solver globals through compatibility wrappers so existing monkeypatch-based
  tests still exercise the same routes.
- ``sfincs_jax/solvers/preconditioners/transport_matrix.py``:
  numerical builder implementations for the common RHSMode=2/3 transport
  preconditioners: collision diagonal, species/speed block, x-grid coarse
  correction, angular FFT/tridiagonal solve, point-block transport
  preconditioners, and the FP transport family: dense Fourier FP,
  block-Thomas Fourier line factors, Schur overlays, local-geometry line
  factors, x-block angular sparse LU, x-block Schur correction, and structured
  f-block LU.  ``v3_driver.py`` keeps historical private wrapper names so
  monkeypatch-based dispatch tests and user debug scripts continue to exercise
  the same facade.
- ``sfincs_jax/solvers/preconditioners/pas/policy.py``
  (historical location: ``sfincs_jax/rhs1_pas_policy.py``):
  PAS applicability, PAS-TZ memory safety, PAS fallback routing, and PAS
  adaptive-smoother eligibility.
- ``sfincs_jax/solvers/preconditioners/pas/matrix_free.py``
  (historical location: ``sfincs_jax/rhs1_pas_matrixfree.py``):
  bounded matrix-free PAS correction probes, streaming L2 norms, candidate
  byte-budget preflights, and ``PasRuntimeChunkPlan`` metadata for keeping
  PAS-heavy residual/correction reductions inside configured memory budgets
  before a matvec is launched.
- ``sfincs_jax/problems/profile_response/policies.py``:
  typed RHSMode=1 solve-routing policy parsing for x-block probe-coarse,
  post-minres, post-coarse, post-residual-equation, bounded sparse-polish,
  host x-block factorization, current-backend dense/sparse admission wrappers,
  override semantics, and fail-closed high-resolution behavior. This keeps
  environment parsing and correction-policy defaults out of ``v3_driver.py``
  and out of the solve entry point.
- ``sfincs_jax/problems/profile_response/setup.py``:
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
- ``sfincs_jax/problems/profile_response/phi1_newton.py``:
  nonlinear Phi1 Newton-Krylov solve orchestration for RHSMode=1 profile
  response. This module owns the accepted-state history solve used by output
  writing, the small Newton-Krylov parity fixture path, active-DOF compaction,
  frozen-Jacobian mode selection, sparse-direct host rescue for non-autodiff
  runs, KSP-history replay wiring, and line-search advancement. ``v3_driver.py``
  now imports the historical public names from this module as compatibility
  facades.
- ``sfincs_jax/problems/profile_response/preconditioner_build.py``:
  RHSMode=1/profile-response full and reduced preconditioner build
  orchestration. The driver passes solve-local builders and projection
  functions through typed contexts, and the helper returns explicit state for
  PAS-TZ guard metadata, collision fallback admission, and optional BiCGStab
  preconditioner reuse. It also owns RHSMode=1 strong-preconditioner family
  mapping, full/reduced strong fallback builders, PAS-Schur to PAS-hybrid build
  adjustment, ADI sweep parsing, and x-block TZ low-``l`` controls. It is also
  the canonical owner of the current RHSMode=1 preconditioner registry and
  legacy binding layer for dispatch, PAS-family builders, Schur binding,
  transport ``tzfft`` reuse, x-block builders, and strong fallback binding; the
  solve owner imports these names only as compatibility seams while Tranche 1
  continues.
- ``sfincs_jax/problems/profile_response/sparse/handoff.py``:
  RHSMode=1/profile-response sparse-PC handoff layer. It owns the
  driver-facing sparse-PC attempt orchestration that depends on solve-local
  cache/replay/residual routing, generic sparse-PC retry execution, direct-tail
  correction admission, and the compatibility import surface used by the
  monolithic solve owner while Batch A continues. Sparse GMRES finalization now
  lives in ``sparse/finalization.py`` and x-block branch orchestration now
  lives in ``sparse/xblock.py``; ``handoff.py`` only re-exports those names for
  compatibility until the public internal imports are fully migrated. Its local
  ``F401,F811`` Ruff waiver is intentional: the module composes dynamic
  ``__all__`` lists from sparse owner modules and carries a few shadowed
  compatibility aliases. Delete the waiver only after ``solve.py`` and owner
  tests import the concrete sparse owners directly.
- ``sfincs_jax/problems/profile_response/sparse/policy.py``:
  generic sparse-PC policy and admission helpers: active-DOF map construction,
  entry classification, sparse factor policy, conservative-pattern setup,
  memory-budget preflight, factor residual-preflight gates, rescue-candidate
  acceptance, auto-retry selection, GMRES stagnation/post-MinRes controls, and
  the shared sparse env-token parser family used by direct, x-block, QI, and
  Fortran-reduced sparse owners. This module is intentionally independent of
  x-block assembled-operator and QI-device setup so it can stay reusable and
  easy to test.
- ``sfincs_jax/operators/profile_response/sparse_pattern.py``:
  conservative and Fortran-reduced sparse structural patterns for
  profile-response full-system operators, including active-index restricted
  patterns, sparse-pattern summaries, and memory-preflight estimates. This
  replaces the historical root ``sfincs_jax/v3_sparse_pattern.py`` owner.
- ``sfincs_jax/operators/profile_response/fblock.py``:
  matrix-free kinetic f-block operator builder and matvec for RHSMode-1
  profile-response solves, including collisionless streaming, ExB, magnetic
  drift, Er, PAS, and Fokker-Planck terms. This replaces the historical root
  ``sfincs_jax/v3_fblock.py`` owner.
- ``sfincs_jax/problems/profile_response/sparse/finalization.py``:
  sparse-PC GMRES result contracts, post-MinRes polish metadata, dtype-retry
  result assembly, completion messages, and final payload construction.
- ``sfincs_jax/problems/profile_response/sparse/direct.py``:
  explicit sparse operator admission, minimum-norm/direct host shortcuts,
  sparse-factor cache keys, host-memory probing, sparse-JAX preconditioner
  materialization, conservative full-pattern probing, ILU/direct-tail policy
  parsing through the shared sparse policy parser, structured direct-tail
  materialization, and final direct-tail metadata assembly.
- ``sfincs_jax/problems/profile_response/sparse/xblock.py``:
  x-block and sxblock rescue/correction helpers, shared x-block Krylov matvec
  and initial-guess policy dataclasses, x-block sparse-PC setup/side-policy
  resolution, assembled-operator setup and preflight, local factor setup,
  precondition-side/probe-coarse gates, moment-Schur/two-level/global-coupling
  stage setup, device/host Krylov control, optional augmented Krylov setup,
  GMRES fallback routing, work estimates, progress messages, physical-residual
  measurement, post-Krylov post-solve correction/completion orchestration,
  x-block branch orchestration, and final x-block sparse-PC diagnostic metadata
  assembly. This module owns generic x-block stage mechanics; QI-specific
  coarse-basis choices remain in ``sparse/qi.py``.
- ``sfincs_jax/problems/profile_response/sparse/fortran_reduced.py``:
  Fortran-reduced x-block backend policy, factor-build, Krylov setup/solve,
  optional moment/global coarse stages, and final payload construction. The
  optional global-coupling stage uses the canonical QI host builder by default
  when no test builder is injected.
- ``sfincs_jax/problems/profile_response/sparse/qi.py``:
  QI-specific x-block device/operator-reuse policy, coarse-seed, Galerkin,
  two-level, QI-device admission/build/probe/install, and residual-deflated
  stages. It also owns ``run_xblock_qi_preconditioner_pipeline()``, the
  aggregate runner that keeps QI stage ordering, setup-time accounting,
  fail-closed reasons, and seed/device/deflated diagnostic scope out of
  ``v3_driver.py``. These helpers are separated from generic sparse-PC logic
  because they encode QI-specific coarse-basis and residual-space choices;
  ``build_xblock_qi_stage_pipeline_context()`` owns production default-builder
  wiring, so ``v3_driver.py`` now injects only solve-local arrays, operators,
  timing, and active-DOF maps rather than importing each QI builder directly.
- ``sfincs_jax/problems/profile_response/dense.py``:
  RHSMode=1/profile-response dense and linear-solve helpers. This module owns
  Krylov routing for implicit, JIT, distributed, GMRES, and BiCGStab solve
  attempts; dense-KSP full/reduced solves; constraintScheme=0 PETSc-compatible
  sparse-ILU; host SciPy rescue; the reduced row-scaled LU path; and the
  full/reduced least-squares dense fallback used by non-differentiable host
  shortcut paths.
- ``sfincs_jax/problems/profile_response/sparse/qi.py``:
  matrix-free QI device seed correction for RHSMode=1 active-DOF solves. The
  driver passes solve-local state through a typed setup/context while the
  module owns env-gate resolution for early and pre-sparse seed hooks, QI
  coarse-basis setup, residual-improvement admission, metadata updates, and
  fail-closed diagnostics. The returned attempt object reports whether a hook
  was attempted and whether it improved the residual so the driver only updates
  replay state when the domain helper accepts a better seed.
- ``sfincs_jax/problems/profile_response/solver_diagnostics.py``
  (historical location: ``sfincs_jax/rhs1_solver_diagnostics.py``):
  typed RHSMode=1 x-block correction diagnostic records, historical solver
  metadata key assembly, and KSP replay diagnostic context forwarding. This
  keeps output-visible trace fields independently testable while
  ``v3_driver.py`` continues to own the solve orchestration.
- ``sfincs_jax/problems/profile_response/solver_diagnostics.py``:
  final RHSMode=1/profile-response linear-solve handoff, output-visible solver
  metadata, bounded PETSc-style KSP residual-history replay, and iteration-count
  diagnostics. It applies cleanup projection, emits optional replay diagnostics,
  writes final residual and elapsed-time progress lines, applies post-xblock
  acceptance-floor metadata, wraps the result in ``V3LinearSolveResult``, and
  owns the bounded PETSc-style GMRES history replay for the optional
  Phi1/Newton-Krylov full-system path.
- ``sfincs_jax/solvers/preconditioners/xblock/coarse.py``
  (historical location: ``sfincs_jax/rhs1_lowmode_coarse.py``):
  low-mode angular, moment, coupled f/tail, and tail-only feature construction
  plus matrix-free Galerkin/least-squares residual-correction builders for
  structured RHSMode=1 f-block preconditioners. The module keeps coarse-space
  algebra independently testable without materializing dense operator bases in
  the driver.
- ``sfincs_jax/solvers/preconditioners/domain_decomposition/__init__.py``
  (historical location: ``sfincs_jax/rhs1_domain_decomposition.py``):
  deterministic angular domain-decomposition patch ranges, shard-aware block
  sizing, and two-level Schwarz coarse-block heuristics. These rules are kept
  independent of the full operator so multi-device preconditioner policy can be
  tested without launching a solve.
- ``sfincs_jax/problems/profile_response/setup.py``:
  RHSMode=1 setup decisions, including active-degree-of-freedom routing and
  reduced-index-map construction for truncated pitch grids, x-block active-DOF
  opt-ins, and PAS constraint-projection solves. The same owner holds the
  reusable JAX primitives for full-to-reduced gathers, reduced-to-full
  one-based scatters, PAS ``l=0`` flux-surface-average projection, and final
  RHSMode=1 cleanup. These primitives are shared by RHSMode=1 sparse-PC,
  x-block active-DOF, PAS-projected reduced residual paths, and final linear
  solve normalization.
- ``sfincs_jax/problems/profile_response/residual.py``
  (historical location: ``sfincs_jax/rhs1_residual.py``):
  small residual target, ratio, convergence, and host-scalar norm helpers used
  by RHSMode=1 sparse-PC and x-block diagnostics, plus the physics-aware
  x-block post-coarse direction builder and the bounded host/device subspace
  residual-equation correction kernels used after x-block solves. The module
  also owns residual-correction preconditioner composition, safe
  non-finite/clipped preconditioner wrapping, and scalar preconditioned-minres
  polish. This keeps fail-closed residual-polish algebra testable without
  entering the production driver.
- ``sfincs_jax/operators/profile_response/device_sparse.py``
  (historical location: ``sfincs_jax/rhs1_device_operator.py``):
  bounded JAX-device CSR materialization, active-index slicing, sparse matvec
  closures, and host-vs-device validation utilities for opt-in RHSMode=1
  device-QI and operator-reuse experiments.
- ``sfincs_jax/solvers/preconditioners/qi/basis.py``:
  deterministic QI basis, coarse-space, phase-space, residual-region,
  active-pattern, global-moment, and Galerkin/action coarse utilities. This
  owner replaces the historical ``rhs1_qi_coarse.py``,
  ``rhs1_qi_phase_space_coarse.py``,
  ``rhs1_qi_residual_region_coarse.py``,
  ``rhs1_qi_active_pattern_coarse.py``, and
  ``rhs1_qi_global_moment_closure.py`` shards.
- ``sfincs_jax/solvers/preconditioners/qi/corrections.py``:
  reusable QI correction primitives: local-plus-coarse two-level actions,
  block-Schur/angular/radial corrections, residual-deflated corrections,
  multilevel residual equations, residual-derived Galerkin selection, and the
  coupled residual equation. This owner replaces the historical
  ``rhs1_qi_two_level.py``, ``rhs1_qi_block_schur.py``,
  ``rhs1_qi_deflation.py``, ``rhs1_qi_multilevel_coarse.py``,
  ``rhs1_qi_residual_galerkin.py``, and ``rhs1_qi_coupled_residual.py``
  shards.
- ``sfincs_jax/solvers/preconditioners/qi/device.py``:
  device-local QI preconditioner and smoother primitives, including
  CSR-backed Jacobi, matrix-free residual-minimizing steps, fail-closed seed
  probes, and the production-shaped device-QI local-plus-coarse state.
- ``sfincs_jax/solvers/preconditioners/qi/policy.py``:
  pure promotion gates for QI hard-seed and production-ladder evidence. It
  requires complete seed/backend coverage, convergence, output and trace
  provenance, residual/observable bounds, and no host fallback before a true
  device-QI claim can be promoted.
- ``sfincs_jax/problems/profile_response/setup.py``
  (historical location: ``sfincs_jax/rhs1_preconditioner_dispatch.py``):
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/problems/profile_response/policies.py``
  (historical location: ``sfincs_jax/rhs1_preconditioner_auto_policy.py``):
  RHSMode=1 preconditioner environment alias normalization plus bounded
  automatic preconditioner policy predicates for PAS, DKES, tokamak, GPU sparse
  fallback, weak-default PAS promotion, PAS-family refinement, FP/DKES routing,
  large-FP near-zero-Er overrides, and sharded line-overrides.
- ``sfincs_jax/solvers/preconditioners/schur/profile_response.py``
  (historical location: ``sfincs_jax/rhs1_schur_policy.py``):
  RHSMode=1 Schur base-preconditioner alias normalization and automatic
  geometry/PAS/DKES routing policy.
- ``sfincs_jax/problems/profile_response/policies.py``
  (historical locations: ``sfincs_jax/rhs1_acceptance_policy.py``,
  ``sfincs_jax/rhs1_constraint0_policy.py``,
  ``sfincs_jax/rhs1_post_xblock_policy.py``,
  ``sfincs_jax/rhs1_sparse_exact_policy.py``,
  ``sfincs_jax/rhs1_sparse_rescue_policy.py``,
  ``sfincs_jax/rhs1_sparse_polish_policy.py``, and
  ``sfincs_jax/rhs1_stage2_policy.py``):
  RHSMode=1 profile-response solve-routing gates, including stage-2 triggers,
  sparse exact-LU admission, sparse-rescue ordering, sparse-polish budgets,
  post-x-block polish, large-PAS fast acceptance, host factor probes, and
  constraint-scheme-0 sparse/dense routing. It also owns current-backend
  wrappers used by the solve owner and small x-block/QI
  control helpers that used to live in ``v3_driver.py``: guarded PAS-TZ
  structured-level parsing, QI device extra-coarse environment controls,
  QI probe minres-step selection, and safe x-block fallback initial-guess
  admission.
- ``sfincs_jax/problems/profile_response/preconditioner_build.py``
  (historical locations: ``sfincs_jax/rhs1_strong_policy.py``,
  ``sfincs_jax/rhs1_strong_control.py``, and
  ``sfincs_jax/rhs1_strong_auto_kind.py``):
  strong-preconditioner request mapping, enable/disable control, automatic
  strong-kind selection, and post-selection adjustment policy.
- ``sfincs_jax/problems/profile_response/preconditioner_build.py``
  (historical location: ``sfincs_jax/rhs1_strong_fallback.py``):
  compatibility facade for historical RHSMode=1 strong-preconditioner fallback
  imports. The implementation owner is now
  ``sfincs_jax/problems/profile_response/preconditioner_build.py``.
- ``sfincs_jax/problems/profile_response/solver_diagnostics.py``
  (absorbed owner for former ``profile_response/handoff.py``):
  accepted-candidate replay, Krylov replay-state updates, final RHSMode=1
  solver diagnostics, and final linear-solve metadata. This is the
  source-mapped seam for the repeated RHSMode=1 driver pattern: compare a
  rescue/refinement candidate against the incumbent residual, apply optional
  measured solver-candidate gates, preserve the accepted residual vector, and
  update the KSP replay metadata only after a strict finite residual
  improvement.
- ``sfincs_jax/operators/profile_response/sources.py``
  (historical location: ``sfincs_jax/rhs1_constraint_sources.py``):
  JAX kernels that convert between kinetic ``f`` blocks and constraint-source
  amplitudes for constraint schemes 1 and 2, including flux-surface averages,
  density/pressure moments, source-basis injection with ``pointAtX0`` handling,
  and the constraintScheme=1 moment-Schur wrapper used by x-block
  preconditioners.
- ``sfincs_jax/problems/profile_response/policies.py``:
  RHSMode=1 host dense fallback, host sparse-direct, sparse-preconditioned
  GMRES rescue, factor-dtype, explicit sparse-helper policy, and automatic
  solver/fallback admission.
- ``sfincs_jax/host_refinement.py``:
  host direct-solve refinement and sparse-direct GMRES polish helpers. The
  monotone refinement loops are NumPy-only, while the polish helper accepts the
  JAX matvec and a host sparse factor so residual-polish behavior can be tested
  without importing the full driver.
- ``sfincs_jax/problems/profile_response/policies.py``
  (historical location: ``sfincs_jax/rhs1_large_cpu_policy.py``):
  large explicit full-FP CPU sparse rescue, x-block seed, exact-LU promotion,
  host x-block assembly, and species-x-block rescue policy.
- ``sfincs_jax/solvers/preconditioners/xblock/policy.py``
  (historical location: ``sfincs_jax/rhs1_xblock_policy.py``):
  pure x-block sparse-PC routing, Krylov-side selection, local factorization
  tuning, lower-fill acceptance gates, and non-autodiff device-host fallback
  metadata for large RHSMode=1 QI/full-FP solves.
- ``sfincs_jax/solvers/preconditioners/xblock/policy.py``
  (historical location: ``sfincs_jax/rhs1_xblock_sparse_host_policy.py``):
  host sparse x-block rescue policy and metadata normalization for the
  non-autodiff large-system fallback path.
- ``sfincs_jax/problems/profile_response/policies.py``:
  RHSMode=1 profile-response admission, post-solve correction, solver-path,
  and implicit/differentiable solve-mode policy.
- ``sfincs_jax/solvers/path_policy.py``:
  pure solver/preconditioner path policy for JIT admission, RHSMode=1 rescue
  slack, DKES GMRES budget defaults, sparse-PC defaults, preconditioner dtype,
  and backend resource-exhaustion classification.
- ``sfincs_jax/solvers/selection_policy.py``:
  measured candidate acceptance gates used by automatic solver/preconditioner
  promotions, including residual/parity checks and paired runtime/memory
  comparisons against an incumbent path.
- ``sfincs_jax/problems/transport_matrix/policies.py``
  (historical location: ``sfincs_jax/transport_policy.py``):
  pure transport backend, sparse-direct, host-GMRES, dtype, recycle, polish,
  residual-abort threshold parsing and failure-message formatting,
  RHSMode=2/3 initial solve, active-DOF, dense fallback, low-memory output,
  streamed-diagnostic, state-vector retention, GMRES restart, per-``whichRHS``
  loop, preconditioner-kind normalization, auto-selection, DD/sparse-JAX env,
  and reduced/full preconditioner builder-dispatch policy. ``TransportRuntimePolicy``
  binds backend-sensitive decisions to the active JAX backend and host
  sparse-factor dtype provider. The former ``handoff_policy.py``,
  ``residual_quality.py``, ``preconditioner_dispatch.py``, and
  ``solve_policy.py`` relays have been deleted; tests import this owner
  directly.
- ``sfincs_jax/problems/transport_matrix/setup.py``
  (historical location: ``sfincs_jax/transport_solve_setup.py``):
  side-effect-light RHSMode=2/3 setup resolution for transport max-iteration
  overrides, optional Krylov state-file loading/merging, ``whichRHS`` subset
  normalization, and CPU/GPU process-parallel worker requests. The driver emits
  the returned notes and keeps solve orchestration, while these pure setup rules
  are covered by direct unit tests.
- ``sfincs_jax/outputs/transport.py``
  (historical location: ``sfincs_jax/transport_streaming_outputs.py``):
  RHSMode=2/3 transport output-schema helpers, host-side streaming
  accumulators, and streaming HDF5 writes. It owns the per-``whichRHS`` NumPy
  buffers, NTV/source handling, final output-field dictionary assembly, solver
  diagnostic arrays, derivative conversion factors, and low-memory HDF5 output
  path.
- ``sfincs_jax/problems/transport_matrix/finalize.py``
  (historical locations: ``sfincs_jax/transport_finalization.py`` and
  ``sfincs_jax/transport_postsolve_diagnostics.py``):
  final RHSMode=2/3 transport solve bookkeeping and post-solve diagnostics. It
  recovers accepted full-space states, applies optional constraint projection,
  stores residual/KSP diagnostics, chooses streamed versus batched diagnostics,
  applies rematerialization/precompute/chunking policy, assembles
  species-by-``whichRHS`` flux arrays, and returns the transport matrix plus
  optional output fields.
- ``sfincs_jax/problems/transport_matrix/diagnostics.py``
  (historical location: ``sfincs_jax/transport_matrix.py``):
  JAX formulas for RHSMode=1 output moments, RHSMode=2/3 transport diagnostics,
  transport-matrix assembly, strict Fortran-order reductions, and cached
  geometry/species diagnostic precomputes.
- ``sfincs_jax/problems/transport_matrix/solve.py``:
  public RHSMode=2/3 transport solve orchestration, transport-specific Krylov
  dispatch, dense-LU solver/preconditioner construction for bounded fallback
  paths, host SciPy GMRES first-attempt/rescue solves, and optional small-system
  SciPy Krylov-history diagnostics. It also owns all-RHS dense-batch solves,
  loop-local full/reduced matvec caches, bounded Krylov recycle bases,
  sequential residual-gate/ETA progress bookkeeping, and RHSMode=2/3
  sparse-direct rescue implementation, including sparse-pattern admission,
  direct active FP operator factors, explicit sparse helper materialization,
  fallback sparse-ILU setup, host iterative refinement, float32 polish, and
  float64 retry. The former ``dense_lu.py``, ``dense_batch.py``,
  ``host_gmres.py``, ``iteration_stats.py``, ``loop.py``,
  ``sparse_direct_solve.py``, and stale ``linear_solve.py`` entries have been
  absorbed here; focused tests import this owner directly.
- ``sfincs_jax/problems/transport_matrix/finalize.py``
  (historical location: ``sfincs_jax/transport_solve_finalization.py``):
  sequential RHSMode=2/3 per-``whichRHS`` finalization after a solver branch has
  accepted a candidate. It owns reduced/full state bookkeeping, optional
  constraint projection, true-residual recomputation, streamed-output
  collection, recycle-basis updates, and optional KSP iteration-stat dispatch.
  Dense fallback accepted-state overrides are explicit so the refactor preserves
  the established active-DOF branch behavior.
- ``sfincs_jax/problems/transport_matrix/parallel/runtime.py``
  (historical location: ``sfincs_jax/transport_parallel_runtime.py``):
  transport parallel backend policy, benchmark scaling audits, worker-count
  validation, XLA worker flag rewriting, RHS partitioning, injected-dependency
  payload normalization, child-worker guard setup, merge-ready result packing,
  GPU worker NPZ conversion, persistent process-pool caching, worker-environment
  setup, backend-specific execution/retry/fallback, GPU subprocess launch,
  parent-side merge of per-worker state/residual/elapsed-time results, and
  single-case sharded-solve planning metadata. This module absorbed the old
  policy, sharding, payload, pool, execution, solve, and validation micro-files
  so transport parallelism has one canonical runtime owner.
- ``sfincs_jax/problems/transport_matrix/parallel/worker.py``
  (historical executable wrapper: ``sfincs_jax/transport_parallel_worker.py``):
  command-line worker entry point used by GPU transport subprocesses. The old
  ``python -m sfincs_jax.problems.transport_matrix.parallel.worker`` path remains supported and
  delegates to this implementation.
- ``sfincs_jax/validation/artifacts.py``:
  lightweight loaders and physics metrics for checked-in publication artifacts. This
  module is independent of the heavy solver path, so documentation and CI can verify
  collisionality, high-collisionality trend, trajectory-sweep, and dashboard artifacts
  plus frozen CPU/GPU Fortran-suite benchmark summaries without rerunning large scans
  or example-suite audits. It also owns the fail-closed schema validator for the
  Fortran-v3 vs SFINCS-JAX runtime/memory benchmark summary consumed by README/docs
  plots.
- ``sfincs_jax/phi1_newton_policy.py``:
  bounded nonlinear/Newton policy for Phi1 solves, including active-DOF mode
  selection, restart sizing, frozen-Jacobian cache policy, and line-search policy.
- ``sfincs_jax/phi1_newton_linear.py``:
  bounded nonlinear linear-step orchestration for Phi1 solves, including reduced/full
  routing, sparse-direct entry, KSP-history emission, and retry-without-preconditioner.
- ``sfincs_jax/problems/profile_response/phi1_newton.py``:
  accepted-iterate update logic and solve orchestration for the Newton path,
  including PETSc-like backtracking, fixed-candidate ``best`` search, and
  finite-state fallback handling.
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
- ``sfincs_jax/validation/benchmark_artifacts.py``:
  fast schema, provenance, and release-blocking classification policy for checked-in
  benchmark JSON artifacts.
- ``sfincs_jax/solvers/memory_model.py``:
  conservative dense/CSR/Krylov/preconditioner memory estimates used by solver
  restart caps, benchmark manifests, and measured solver-candidate gates. This is
  the preflight layer that keeps future memory-saving defaults testable before
  expensive operators or preconditioners are materialized.
- ``sfincs_jax/problems/profile_response/policies.py``:
  tested admission gates for RHSMode=1 host dense, sparse-host, constrained-PAS
  sparse-PC, CPU 3D full-FP sparse-PC, and GPU tokamak full-FP no-Er/Er
  sparse-PC auto lanes. These helpers keep solver path promotion rules explicit
  and unit-testable without assembling a kinetic operator.

``sfincs_jax/solver.py`` and ``sfincs_jax/solvers/implicit.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear-algebra infrastructure:

- Krylov wrappers,
- host-direct and sparse rescues,
- differentiable linear solves,
- traced-safe implicit-solve method resolution that maps host-only SciPy Krylov
  requests to JAX-native incremental GMRES inside autodiff paths,
- JAX-native linear solve utilities,
- augmented FGMRES hooks that reuse a checked coarse basis and stored operator
  action ``(U, A U)`` without assembling a dense global operator.

``sfincs_jax/problems/transport_matrix/diagnostics.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=2/3 postprocessing and transport-matrix assembly.
The former flat ``sfincs_jax/transport_matrix.py`` file has been deleted; new
code should import this owner or use the public API/CLI.

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
  ``sfincs_jax/problems/profile_response/solve.py``,
  ``sfincs_jax/problems/transport_matrix/solve.py``,
  ``sfincs_jax/solvers``, and ``sfincs_jax/solvers/preconditioners``
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
