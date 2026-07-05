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
``solver.py``, ``ambipolar.py``, ``sensitivity.py``, ``plotting.py``,
``compare.py``, ``io.py``, ``namelist.py``, ``input_compat.py``, ``grids.py``,
``diagnostics.py``, ``paths.py``, and ``profiling.py``. ``v3_driver.py`` is a
small compatibility facade for historical imports; it must not own physics,
operator, or solver implementation.

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
     - package root public comparison API
     - keep at root; it owns user-facing SFINCS comparison helpers and strict HDF5 output parity
   * - ``input_compat.py``
     - input compatibility owner
     - keep root public compatibility shim until input package exports cover callers
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
     - solvers public contracts owner
     - keep root shim until solvers exports cover public contracts
   * - ``v3_driver.py``
     - compatibility shim to problem owners
     - keep tiny shim until the compatibility deprecation window closes; public examples and scripts should not import it

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
- keeps legacy compatibility aliases for flat output-format helpers owned by
  ``sfincs_jax.outputs.formats``.

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
checks. Fixed-artifact QI device campaign ingestion belongs to
``sfincs_jax/validation/qi_device.py`` so workflow code stays focused on
reusable execution and evidence-generation tasks. Historical ``optimization_*``
workflow modules resolve through
package-level compatibility aliases in ``sfincs_jax.workflows`` instead of
separate implementation files.

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

``sfincs_jax/validation/qi_device.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QI device evidence owner:

- checks route-level QI device artifacts for fail-closed metadata,
- rejects GPU evidence that lacks backend/provenance fields,
- gates bounded QI ``15x`` GPU campaign JSON against residual and
  CPU/Fortran root-agreement criteria,
- keeps campaign-specific promotion policy out of reusable workflow modules.

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
- system metadata used by the driver and diagnostics.

``sfincs_jax/operators/profile_linear_systems.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Residual and source-term helpers. This is where the thermodynamic drives and other RHS
pieces are assembled before being fed to the solve stack.

``sfincs_jax/v3_driver.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compatibility shim for the former monolithic driver. It imports
``sfincs_jax.problems.profile_solve`` and
``sfincs_jax.problems.transport_solve``, exposes moved public and
legacy-private names for existing users, and aliases ``import
sfincs_jax.v3_driver`` to the profile-response solve owner so old monkeypatch
tests still mutate the globals used by moved functions. It intentionally
contains no physics equations, solver algorithms, output assembly, or
preconditioner setup logic.

When a solve behaves differently on CPU and GPU, inspect the problem owner
first: ``sfincs_jax.problems.profile_solve`` for RHSMode 1 and
``sfincs_jax.problems.transport_solve`` for RHSMode 2/3. The domain owners
replacing the old monolith are:

- ``sfincs_jax/problems/profile_solver_diagnostics.py``:
  RHSMode=1 linear-solve and Newton-Krylov result dataclasses, final
  profile-response result wrapping, output-visible solver metadata, and bounded
  PETSc-style KSP residual-history replay.
- ``sfincs_jax/problems/transport_finalize.py``:
  RHSMode=2/3 transport-matrix result dataclass plus per-RHS finalization,
  constraint projection, residual bookkeeping, and KSP replay request contracts.
- ``sfincs_jax/solver.py``:
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
  PAS-family variants. Legacy access through the historical driver namespace is
  provided by the tiny ``sfincs_jax.v3_driver`` shim, while this package remains
  the numerical owner.
- ``sfincs_jax/solvers/preconditioner_pas_angular.py``:
  PAS-only RHSMode=1 angular block-tridiagonal factors, including the
  tokamak-like theta/:math:`L` builder and the geometry-rich theta-zeta/:math:`L`
  builder. This module owns the block-Thomas factor setup, low-mode combined
  ``L=0,1`` singularity handling, optional structured velocity tail factors,
  PAS-TZ memory fallback invocation, cache population, and reduced/full apply
  wrappers. Fallback-builder wiring lives in the profile-response solve
  owner; ``v3_driver.py`` only exposes the compatibility import path.
- ``sfincs_jax/solvers/preconditioner_pas_xblock_ilu.py``
  (historical location: ``sfincs_jax/rhs1_pas_xblock_ilu.py``):
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
  reduced/full apply wrappers. Historical driver access is a compatibility
  alias, not an implementation owner.
- ``sfincs_jax/solvers/preconditioner_full_fp_structured.py``:
  structured full-Fokker-Planck RHSMode=1 f-block preconditioners. The module
  owns block-Jacobi, angular-line, pitch-angular, FP-radial grouped factors,
  and low-mode/moment/tail Schur correction builders over the structured
  f-block operator. Same-shape cache keys, metadata emission, memory guards,
  and matrix-free residual-correction composition live here; historical driver
  access is a compatibility alias, not an implementation owner.
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
  Historical driver access is a compatibility alias, not an implementation
  owner.
- ``sfincs_jax/solvers/preconditioner_xblock_tz_sparse.py``
  (historical location: ``sfincs_jax/rhs1_xblock_tz_sparse.py``):
  sparse per-``x`` RHSMode=1 full-FP preconditioner setup. This module owns the
  host/JAX x-block LU/ILU policy, compact CSR/padded triangular-factor apply,
  selected theta/zeta upwind sparse-stencil assembly, explicit FP assembled-host
  cache, host-assembly admission policy, per-block sparse matrix/diagonal
  assembly, sparse per-:math:`L` species/``x`` host rescue factors, one-shot
  sparse species/``x`` seed construction, skipped-block diagonal fallback,
  host-factor probe/cache-key policy, shared chunked unsharded matrix probing,
  and extra-variable Schur solve. Historical driver access is a compatibility
  alias, not an implementation owner.
- ``sfincs_jax/solvers/preconditioner_xblock_low_l_schur.py``:
  low-pitch x-block Schur preconditioners for exact RHSMode=1 full-CSR systems.
  This module owns the opt-in native ``x_ell`` kinetic factor, native
  ``x_ell`` plus dense-tail Schur factor, sparse low-``ell`` ``(theta,zeta)``
  x-block factor, physics low-mode coarse residual correction, and the shared
  low-``ell`` x-block index helper. Dispatch/admission wiring lives in the
  profile-response sparse owners and the public solve owner; the old
  ``rhs1_full_assembly.py`` module no longer exists.
- ``sfincs_jax/solvers/preconditioner_xblock_active.py``:
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
  Dispatch/admission wiring lives in the profile-response sparse owners and the
  public solve owner; the old ``rhs1_full_assembly.py`` module no longer
  exists.
- ``sfincs_jax/solvers/preconditioner_domain_decomposition.py``:
  angular line-block and restricted-additive-Schwarz preconditioners for
  RHSMode=1 domain-decomposition and strong fallback paths. It owns the
  theta-line, zeta-line, theta-domain, zeta-domain, theta-Schwarz,
  zeta-Schwarz, theta-line-with-``x``-diagonal, and full theta-zeta
  angular-block setup/apply kernels used by automatic line selection,
  Schur-base construction, and explicit strong-preconditioner requests. Shared
  axis-line index maps, cache keys, regularization policy, extra-variable tail
  solves, and multi-level residual correction hooks live here. Historical
  driver access is a compatibility alias, not an implementation owner.
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
- ``sfincs_jax/solvers/explicit_sparse.py``:
  explicit-sparse host-factor environment parsing, canonical factor-kind alias
  resolution, monolithic LU/ILU guard sizing, and the typed
  ``ExplicitSparseFactorSettings`` bundle, dense/CSR storage decisions,
  pattern-color probing, symbolic Schur/frontal/ND/BLR settings, SuperLU
  pivot/permutation options, ILU options, host explicit-sparse operator
  assembly, padded and compact-CSR JAX triangular-factor apply kernels,
  permutation inversion, logging, monolithic preflight guard, and factorization
  orchestration. Profile-response and transport solve owners pass the concrete
  operator/factor callbacks; ``v3_driver.py`` only exposes the compatibility
  import path. The old explicit-sparse policy, builder support, and triangular
  solve helper files were absorbed here so the explicit sparse host-factor lane
  has one review surface.
- ``sfincs_jax/solvers/preconditioner_symbolic_host.py``:
  RHSMode=1 host sparse ILU/LU factor setup used by non-differentiable
  CLI-oriented rescue paths. The module owns matrix-free column assembly,
  structural-threshold application, SuperLU retry/regularization policy, cached
  dense/JAX triangular-factor materialization, and the matrix-free full-system
  adapter used by coarse/Galerkin corrections. Historical private helper names
  are exposed through the profile-response solve owner and the tiny driver
  compatibility shim.
- ``sfincs_jax/solvers/preconditioner_symbolic_profile.py``:
  Fortran-v3-style reduced active sparse factors for RHSMode=1. The module owns
  reduced active matrix construction, support-mode parsing and preflight,
  symbolic-plan permutation, sparse equilibration, LU/ILU memory admission, and
  SuperLU/RCM factor setup for the non-differentiable host CSR lane. Historical
  private names are exposed through the profile-response solve owner; direct
  ``Pmat`` emission and active-preconditioner dispatch live in profile-response
  sparse/operator owners.
- ``sfincs_jax/solvers/preconditioner_symbolic_active.py``:
  active-projected RHSMode=1 sparse-factor preconditioners. The module owns the
  global active sparse factor, row/column-equilibrated active factor, and
  physics-filtered active sparse factor that retains selected off-diagonal
  kinetic couplings. These are host-side, non-differentiable preconditioner
  setup routines for explicit CSR solves; candidate dispatch lives in the
  profile-response sparse/solve owners.
- ``sfincs_jax/problems/profile_policies.py``
  (historical location: ``sfincs_jax/rhs1_direct_tail_policy.py``):
  RHSMode=1 direct-tail structured-preconditioner adapter, direct reduced-Pmat
  aliases, stable cache-key hashing, cache-hit metadata tagging, and adaptive
  direct-tail memory-cap policy. Historical compatibility names are exposed
  through the profile-response solve owner and tiny driver shim so existing
  debug scripts can still clear the direct-tail cache or inspect the policy
  through the historical driver namespace.
- ``sfincs_jax/operators/profile_reduced_tail.py``
  (historical location: ``sfincs_jax/rhs1_fortran_reduced_direct_tail.py``):
  RHSMode=1 Fortran-reduced constraintScheme=1 direct-tail sparse-operator
  materialization. The module emits source/tail columns and moment rows from the
  same formulas used by the matrix-free v3 operator, while structured full-CSR
  builder callbacks are supplied by the profile-response solve/operator owners.
- ``sfincs_jax/problems/profile_policies.py``
  (historical location: ``sfincs_jax/rhs1_active_preconditioner_policy.py``):
  active-projected RHSMode=1 full-CSR preconditioner auto-policy. The module
  owns environment parsing for the candidate ladder, large-system fallback
  guard, skipped-fallback metadata, and progress logging default. Candidate
  dispatch and setup timing live in the profile-response sparse/solve owners.
- ``sfincs_jax/solvers/preconditioner_symbolic_policy.py``
  (historical location: ``sfincs_jax/rhs1_fortran_reduced_factor_policy.py``):
  Fortran-v3-reduced RHSMode=1 active-Pmat factorization policy. The module
  owns factor-kind normalization, large-matrix ILU guards, LU prefill safety
  defaults, SuperLU/RCM ordering candidates, equilibration norm selection, and
  progress logging defaults; the symbolic-sparse RHSMode=1 Fortran-reduced
  module consumes this policy and performs the numerical sparse factor setup.
- ``sfincs_jax/solvers/preconditioner_symbolic_policy.py``
  (historical location: ``sfincs_jax/rhs1_symbolic_frontal_policy.py``):
  symbolic frontal/Schur RHSMode=1 active-preconditioner policy. The module
  owns frontal versus nested-dissection routing, separator/block limits, dense
  Schur update budgets, admission probe thresholds, and ND residual-polish
  controls; sparse symbolic analysis, factorization, and true-residual
  admission live in the symbolic-sparse and profile-response sparse owners.
- ``sfincs_jax/solvers/preconditioner_symbolic_policy.py``
  (historical location: ``sfincs_jax/rhs1_symbolic_sparse_policy.py``):
  symbolic superblock and separator-Schur RHSMode=1 active-preconditioner
  policy. The module owns grouped-block and block-Schur size gates, ordering
  defaults, separator/coarse limits, retained-cross-fraction gates, prefill
  safety factors, and admission probe thresholds; symbolic sparse analysis,
  host factorization, and true-residual admission live in the symbolic-sparse
  and profile-response sparse owners.
- ``sfincs_jax/operators/profile_full_system.py``
  (historical location: ``sfincs_jax/rhs1_structured_full_csr.py``):
  analytic RHSMode=1 full-CSR assembly plus the runtime/non-autodiff
  ``SparseOperatorBundle`` adapter used by sparse-PC solver paths. Unsupported
  or over-budget cases return ``None`` so callers can fall back to the
  established matrix-free or pattern-probed path.
- ``sfincs_jax/operators/profile_true_operator_rescue.py``
  (historical location: ``sfincs_jax/rhs1_true_operator_rescue.py``):
  support bundles and low-level helpers for RHSMode=1 true-operator
  residual-window, active-submatrix, coupled-coarse, and residual-coarse rescue
  preconditioners. The module owns the reusable true-action column cache,
  sparse-factor storage estimator, additive-rescue budget accounting, graph
  expansion, residual-window target parsing, and residual-driven window
  selection. It also owns the residual sparse-window/coarse builders, true-
  operator residual-window LSQ, active-block LSQ, active-residual-block LSQ,
  active-submatrix, coupled-coarse builders, and active residual diagnostic
  summaries. Historical private names are exposed through the profile-response
  solve owner and tiny driver compatibility shim.
- ``sfincs_jax/solvers/krylov_dispatch.py``:
  concrete Krylov solver routing for host-only SciPy methods, JIT/non-JIT JAX
  GMRES, distributed GMRES, diagnostic solver labels, and
  ``SFINCS_JAX_GMRES_DISTRIBUTED`` axis selection. Problem solve owners pass the
  selected solver callbacks through typed contexts; the driver shim only
  preserves historical import access.
- ``sfincs_jax/solvers/preconditioner_transport_matrix.py``:
  numerical builder implementations for the common RHSMode=2/3 transport
  preconditioners: collision diagonal, species/speed block, x-grid coarse
  correction, angular FFT/tridiagonal solve, point-block transport
  preconditioners, and the FP transport family: dense Fourier FP,
  block-Thomas Fourier line factors, Schur overlays, local-geometry line
  factors, x-block angular sparse LU, x-block Schur correction, and structured
  f-block LU. Historical private wrapper names are exposed through the
  transport/profile-response solve owners and the driver compatibility shim.
- ``sfincs_jax/solvers/preconditioner_pas_policy.py``
  (historical location: ``sfincs_jax/rhs1_pas_policy.py``):
  PAS applicability, PAS-TZ memory safety, PAS fallback routing, and PAS
  adaptive-smoother eligibility.
- ``sfincs_jax/solvers/preconditioner_pas_matrix_free.py``
  (historical location: ``sfincs_jax/rhs1_pas_matrixfree.py``):
  bounded matrix-free PAS correction probes, streaming L2 norms, candidate
  byte-budget preflights, and ``PasRuntimeChunkPlan`` metadata for keeping
  PAS-heavy residual/correction reductions inside configured memory budgets
  before a matvec is launched.
- ``sfincs_jax/problems/profile_policies.py``:
  typed RHSMode=1 solve-routing policy parsing for x-block probe-coarse,
  post-minres, post-coarse, post-residual-equation, bounded sparse-polish,
  host x-block factorization, current-backend dense/sparse admission wrappers,
  override semantics, and fail-closed high-resolution behavior. This keeps
  environment parsing and correction-policy defaults out of ``v3_driver.py``
  and out of the solve entry point.
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
  public names from this module remain reachable through the driver
  compatibility shim.
- ``sfincs_jax/problems/profile_preconditioner_build.py``:
  RHSMode=1/profile-response full and reduced preconditioner build
  orchestration. The driver passes solve-local builders and projection
  functions through typed contexts, and the helper returns explicit state for
  PAS-TZ guard metadata, collision fallback admission, and optional BiCGStab
  preconditioner reuse. It also owns RHSMode=1 strong-preconditioner family
  mapping, full/reduced strong fallback builders, PAS-Schur to PAS-hybrid build
  adjustment, ADI sweep parsing, and x-block TZ low-``l`` controls. It is also
  the canonical owner of the RHSMode=1 preconditioner registry and
  legacy binding layer for dispatch, PAS-family builders, Schur binding,
  transport ``tzfft`` reuse, x-block builders, and strong fallback binding; the
  solve owner imports these names only as compatibility seams.
- ``sfincs_jax/problems/profile_sparse_solve.py``:
  RHSMode=1/profile-response sparse-PC solve orchestration layer. It owns the
  driver-facing sparse-PC attempt orchestration that depends on solve-local
  cache/replay/residual routing, generic sparse-PC retry execution, direct-tail
  correction admission, finalization, and x-block sparse branch orchestration.
  The module provides a stable compatibility import surface for the profile
  solve owner while keeping sparse branch behavior covered by owner-level
  tests.
- ``sfincs_jax/problems/profile_sparse_policy.py``:
  generic sparse-PC policy and admission helpers: active-DOF map construction,
  entry classification, sparse factor policy, conservative-pattern setup,
  memory-budget preflight, factor residual-preflight gates, rescue-candidate
  acceptance, auto-retry selection, GMRES stagnation/post-MinRes controls, and
  the shared sparse env-token parser family used by direct, x-block, QI, and
  Fortran-reduced sparse owners. This module is intentionally independent of
  x-block assembled-operator and QI-device setup so it can stay reusable and
  easy to test.
- ``sfincs_jax/operators/profile_sparse_pattern.py``:
  conservative and Fortran-reduced sparse structural patterns for
  profile-response full-system operators, including active-index restricted
  patterns, sparse-pattern summaries, and memory-preflight estimates. This
  replaces the historical root ``sfincs_jax/v3_sparse_pattern.py`` owner.
- ``sfincs_jax/operators/profile_fblock.py``:
  matrix-free kinetic f-block operator builder and matvec for RHSMode-1
  profile-response solves, including collisionless streaming, ExB, magnetic
  drift, Er, PAS, and Fokker-Planck terms. This replaces the historical root
  ``sfincs_jax/v3_fblock.py`` owner.
- ``sfincs_jax/problems/profile_sparse_finalization.py``:
  sparse-PC GMRES result contracts, post-MinRes polish metadata, dtype-retry
  result assembly, completion messages, and final payload construction.
- ``sfincs_jax/problems/profile_sparse_direct.py``:
  explicit sparse operator admission, minimum-norm/direct host shortcuts,
  sparse-factor cache keys, host-memory probing, sparse-JAX preconditioner
  materialization, conservative full-pattern probing, ILU/direct-tail policy
  parsing through the shared sparse policy parser, structured direct-tail
  materialization, and final direct-tail metadata assembly.
- ``sfincs_jax/problems/profile_sparse_xblock.py``:
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
- ``sfincs_jax/problems/profile_sparse_fortran_reduced.py``:
  Fortran-reduced x-block backend policy, factor-build, Krylov setup/solve,
  optional moment/global coarse stages, and final payload construction. The
  optional global-coupling stage uses the canonical QI host builder by default
  when no test builder is injected.
- ``sfincs_jax/problems/profile_sparse_qi.py``:
  QI-specific x-block device/operator-reuse policy, coarse-seed, Galerkin,
  two-level, QI-device admission/build/probe/install, and residual-deflated
  stages. It also owns ``run_xblock_qi_preconditioner_pipeline()``, the
  aggregate runner that owns QI stage ordering, setup-time accounting,
  fail-closed reasons, and seed/device/deflated diagnostic scope. These helpers
  are separated from generic sparse-PC logic because they encode QI-specific
  coarse-basis and residual-space choices;
  ``build_xblock_qi_stage_pipeline_context()`` owns production default-builder
  wiring, while the profile-response solve owner supplies solve-local arrays,
  operators, timing, and active-DOF maps.
- ``sfincs_jax/problems/profile_dense.py``:
  RHSMode=1/profile-response dense and linear-solve helpers. This module owns
  Krylov routing for implicit, JIT, distributed, GMRES, and BiCGStab solve
  attempts; dense-KSP full/reduced solves; constraintScheme=0 PETSc-compatible
  sparse-ILU; host SciPy rescue; the reduced row-scaled LU path; and the
  full/reduced least-squares dense fallback used by non-differentiable host
  shortcut paths.
- ``sfincs_jax/problems/profile_sparse_qi.py``:
  matrix-free QI device seed correction for RHSMode=1 active-DOF solves. The
  driver passes solve-local state through a typed setup/context while the
  module owns env-gate resolution for early and pre-sparse seed hooks, QI
  coarse-basis setup, residual-improvement admission, metadata updates, and
  fail-closed diagnostics. The returned attempt object reports whether a hook
  was attempted and whether it improved the residual so the driver only updates
  replay state when the domain helper accepts a better seed.
- ``sfincs_jax/problems/profile_solver_diagnostics.py``
  (historical location: ``sfincs_jax/rhs1_solver_diagnostics.py``):
  typed RHSMode=1 x-block correction diagnostic records, historical solver
  metadata key assembly, and KSP replay diagnostic context forwarding. This
  keeps output-visible trace fields independently testable outside the driver
  compatibility shim.
- ``sfincs_jax/problems/profile_solver_diagnostics.py``:
  final RHSMode=1/profile-response linear-solve diagnostics, output-visible solver
  metadata, bounded PETSc-style KSP residual-history replay, and iteration-count
  diagnostics. It applies cleanup projection, emits optional replay diagnostics,
  writes final residual and elapsed-time progress lines, applies post-xblock
  acceptance-floor metadata, wraps the result in ``V3LinearSolveResult``, and
  owns the bounded PETSc-style GMRES history replay for the optional
  Phi1/Newton-Krylov full-system path.
- ``sfincs_jax/solvers/preconditioner_xblock_coarse.py``
  (historical location: ``sfincs_jax/rhs1_lowmode_coarse.py``):
  low-mode angular, moment, coupled f/tail, and tail-only feature construction
  plus matrix-free Galerkin/least-squares residual-correction builders for
  structured RHSMode=1 f-block preconditioners. The module keeps coarse-space
  algebra independently testable without materializing dense operator bases in
  the driver.
- ``sfincs_jax/solvers/preconditioner_domain_decomposition.py``
  (historical location: ``sfincs_jax/rhs1_domain_decomposition.py``):
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
- ``sfincs_jax/problems/profile_residual.py``
  (historical location: ``sfincs_jax/rhs1_residual.py``):
  small residual target, ratio, convergence, and host-scalar norm helpers used
  by RHSMode=1 sparse-PC and x-block diagnostics, plus the physics-aware
  x-block post-coarse direction builder and the bounded host/device subspace
  residual-equation correction kernels used after x-block solves. The module
  also owns residual-correction preconditioner composition, safe
  non-finite/clipped preconditioner wrapping, and scalar preconditioned-minres
  polish. This keeps fail-closed residual-polish algebra testable without
  entering the production driver.
- ``sfincs_jax/operators/profile_device_sparse.py``
  (historical location: ``sfincs_jax/rhs1_device_operator.py``):
  bounded JAX-device CSR materialization, active-index slicing, sparse matvec
  closures, and host-vs-device validation utilities for opt-in RHSMode=1
  device-QI and operator-reuse experiments.
- ``sfincs_jax/solvers/preconditioner_qi_basis.py``:
  deterministic QI basis, coarse-space, phase-space, residual-region,
  active-pattern, global-moment, and Galerkin/action coarse utilities. This
  owner replaces the historical ``rhs1_qi_coarse.py``,
  ``rhs1_qi_phase_space_coarse.py``,
  ``rhs1_qi_residual_region_coarse.py``,
  ``rhs1_qi_active_pattern_coarse.py``, and
  ``rhs1_qi_global_moment_closure.py`` shards.
- ``sfincs_jax/solvers/preconditioner_qi_corrections.py``:
  reusable QI correction primitives: local-plus-coarse two-level actions,
  block-Schur/angular/radial corrections, residual-deflated corrections,
  multilevel residual equations, residual-derived Galerkin selection, and the
  coupled residual equation. This owner replaces the historical
  ``rhs1_qi_two_level.py``, ``rhs1_qi_block_schur.py``,
  ``rhs1_qi_deflation.py``, ``rhs1_qi_multilevel_coarse.py``,
  ``rhs1_qi_residual_galerkin.py``, and ``rhs1_qi_coupled_residual.py``
  shards.
- ``sfincs_jax/solvers/preconditioner_qi_device.py``:
  device-local QI preconditioner and smoother primitives, including
  CSR-backed Jacobi, matrix-free residual-minimizing steps, fail-closed seed
  probes, and the production-shaped device-QI local-plus-coarse state.
- ``sfincs_jax/validation/qi_device.py``:
  QI device and production-ladder evidence policy. It checks fail-closed
  QI-device artifacts, bounded GPU-campaign provenance, and production
  promotion evidence requiring complete seed/backend coverage, convergence,
  output and trace provenance, residual/observable bounds, and no host fallback
  before a true device-QI claim can be promoted.
- ``sfincs_jax/problems/profile_setup.py``
  (historical location: ``sfincs_jax/rhs1_preconditioner_dispatch.py``):
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/problems/profile_policies.py``
  (historical location: ``sfincs_jax/rhs1_preconditioner_auto_policy.py``):
  RHSMode=1 preconditioner environment alias normalization plus bounded
  automatic preconditioner policy predicates for PAS, DKES, tokamak, GPU sparse
  fallback, weak-default PAS promotion, PAS-family refinement, FP/DKES routing,
  large-FP near-zero-Er overrides, and sharded line-overrides.
- ``sfincs_jax/solvers/preconditioner_schur_profile.py``
  (historical location: ``sfincs_jax/rhs1_schur_policy.py``):
  RHSMode=1 Schur base-preconditioner alias normalization and automatic
  geometry/PAS/DKES routing policy.
- ``sfincs_jax/problems/profile_policies.py``
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
- ``sfincs_jax/problems/profile_preconditioner_build.py``
  (historical locations: ``sfincs_jax/rhs1_strong_policy.py``,
  ``sfincs_jax/rhs1_strong_control.py``, and
  ``sfincs_jax/rhs1_strong_auto_kind.py``):
  strong-preconditioner request mapping, enable/disable control, automatic
  strong-kind selection, and post-selection adjustment policy.
- ``sfincs_jax/problems/profile_preconditioner_build.py``
  (historical location: ``sfincs_jax/rhs1_strong_fallback.py``):
  compatibility facade for historical RHSMode=1 strong-preconditioner fallback
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
- ``sfincs_jax/operators/profile_sources.py``
  (historical location: ``sfincs_jax/rhs1_constraint_sources.py``):
  JAX kernels that convert between kinetic ``f`` blocks and constraint-source
  amplitudes for constraint schemes 1 and 2, including flux-surface averages,
  density/pressure moments, source-basis injection with ``pointAtX0`` handling,
  and the constraintScheme=1 moment-Schur wrapper used by x-block
  preconditioners.
- ``sfincs_jax/problems/profile_policies.py``:
  RHSMode=1 host dense fallback, host sparse-direct, sparse-preconditioned
  GMRES rescue, factor-dtype, explicit sparse-helper policy, and automatic
  solver/fallback admission.
- ``sfincs_jax/solvers/explicit_sparse.py``:
  explicit host-sparse operator assembly/factorization policy plus host
  direct-solve refinement and sparse-direct GMRES polish helpers. The monotone
  refinement loops are NumPy-only, while the polish helper accepts the JAX
  matvec and a host sparse factor so residual-polish behavior can be tested
  without importing the full driver.
- ``sfincs_jax/problems/profile_policies.py``
  (historical location: ``sfincs_jax/rhs1_large_cpu_policy.py``):
  large explicit full-FP CPU sparse rescue, x-block seed, exact-LU promotion,
  host x-block assembly, and species-x-block rescue policy.
- ``sfincs_jax/solvers/preconditioner_xblock_policy.py``
  (historical location: ``sfincs_jax/rhs1_xblock_policy.py``):
  pure x-block sparse-PC routing, Krylov-side selection, local factorization
  tuning, lower-fill acceptance gates, and non-autodiff device-host fallback
  metadata for large RHSMode=1 QI/full-FP solves.
- ``sfincs_jax/solvers/preconditioner_xblock_policy.py``
  (historical location: ``sfincs_jax/rhs1_xblock_sparse_host_policy.py``):
  host sparse x-block rescue policy and metadata normalization for the
  non-autodiff large-system fallback path.
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
- ``sfincs_jax/problems/transport_policies.py``
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
- ``sfincs_jax/problems/transport_setup.py``
  (historical location: ``sfincs_jax/transport_solve_setup.py``):
  side-effect-light RHSMode=2/3 setup resolution for transport max-iteration
  overrides, optional Krylov state-file loading/merging, ``whichRHS`` subset
  normalization, CPU/GPU process-parallel worker requests, loop-local
  full/reduced matvec caches, bounded Krylov recycle bases, recycle-size
  admission, and sequential residual-gate/ETA progress bookkeeping. The driver
  emits the returned notes and keeps solve orchestration, while these setup and
  loop-state rules are covered by direct unit tests.
- ``sfincs_jax/outputs/transport.py``
  (historical location: ``sfincs_jax/transport_streaming_outputs.py``):
  RHSMode=2/3 transport output-schema helpers, host-side streaming
  accumulators, and streaming HDF5 writes. It owns the per-``whichRHS`` NumPy
  buffers, NTV/source handling, final output-field dictionary assembly, solver
  diagnostic arrays, derivative conversion factors, and low-memory HDF5 output
  path.
- ``sfincs_jax/problems/transport_finalize.py``
  (historical locations: ``sfincs_jax/transport_finalization.py`` and
  ``sfincs_jax/transport_postsolve_diagnostics.py``):
  final RHSMode=2/3 transport solve bookkeeping and post-solve diagnostics. It
  recovers accepted full-space states, applies optional constraint projection,
  stores residual/KSP diagnostics, chooses streamed versus batched diagnostics,
  applies rematerialization/precompute/chunking policy, assembles
  species-by-``whichRHS`` flux arrays, and returns the transport matrix plus
  optional output fields.
- ``sfincs_jax/problems/transport_diagnostics.py``
  (historical location: ``sfincs_jax/transport_matrix.py``):
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
- ``sfincs_jax/problems/transport_finalize.py``
  (historical location: ``sfincs_jax/transport_solve_finalization.py``):
  sequential RHSMode=2/3 per-``whichRHS`` finalization after a solver branch has
  accepted a candidate. It owns reduced/full state bookkeeping, optional
  constraint projection, true-residual recomputation, streamed-output
  collection, recycle-basis updates, and optional KSP iteration-stat dispatch.
  Dense fallback accepted-state overrides are explicit so the refactor preserves
  the established active-DOF branch behavior.
- ``sfincs_jax/problems/transport_parallel_runtime.py``
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
- ``sfincs_jax/problems/transport_parallel_worker.py``
  (historical executable wrapper: ``sfincs_jax/transport_parallel_worker.py``):
  command-line worker entry point used by GPU transport subprocesses. The old
  ``python -m sfincs_jax.problems.transport_parallel_worker`` path remains supported and
  delegates to this implementation.
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

``sfincs_jax/problems/transport_diagnostics.py``
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
