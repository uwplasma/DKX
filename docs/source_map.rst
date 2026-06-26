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

``sfincs_jax/geometry.py`` and ``sfincs_jax/io.py`` intentionally remain modules
for now because those import paths are already part of the active code. The
``geometry`` and ``io`` package names are reserved for a later migration step
that moves the implementation without shadowing or breaking existing imports.

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

``sfincs_jax/input_compat.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compatibility and search-order logic for equilibrium files, input normalization, and
user overrides. This is the module to inspect first when a case fails to find a VMEC or
Boozer file.

``sfincs_jax/jax_geometry_adapters.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional structural adapters for JAX-native geometry producers such as ``vmec_jax``
and ``booz_xform_jax``. These helpers do not make either package a required
dependency; they normalize in-memory VMEC-like ``wout`` objects to the internal
``VmecWout`` layout used by ``geometryScheme=5``. The module also owns the
machine-readable VMEC/Boozer proxy workflow contract and the shared no-solve
provenance gate used by the workflow status and autodiff examples.

``sfincs_jax/grids.py`` and ``sfincs_jax/xgrid.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Velocity-space discretization:

- collocation points in :math:`x`,
- quadrature weights,
- modal transforms used by the collision operator,
- special handling for monoenergetic ``RHSMode=3``.

``sfincs_jax/geometry.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry loading and normalized coefficient generation:

- analytic model fields,
- VMEC-derived coefficients,
- Boozer ``.bc`` evaluation,
- surface metrics and scalar geometry diagnostics.

``sfincs_jax/vmec_geometry.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC ``geometryScheme=5`` Fourier-sum evaluator. The public file path
``vmec_geometry_from_wout_file(...)`` reads ``wout`` data and delegates to
``vmec_geometry_from_wout(...)`` so optional in-memory producers can exercise the
same formulas and parity tests without hidden file I/O.

``sfincs_jax/collisionless.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streaming and mirror-force contributions in the Legendre basis.

``sfincs_jax/collisionless_exb.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`E\times B` terms in the kinetic operator, including angular advection and
the radial-electric-field contributions to :math:`\dot \xi` and :math:`\dot x` where
supported.

``sfincs_jax/magnetic_drifts.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Magnetic-drift coefficient construction, angular advection terms, upwinding masks, and
associated :math:`\partial_\xi` couplings.

``sfincs_jax/collisions.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collision models:

- PAS,
- full linearized Fokker-Planck,
- field-particle terms,
- Phi1-modified collision coefficients.

``sfincs_jax/v3_system.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

System construction:

- state-vector ordering,
- operator block composition,
- transport-RHS rewrites,
- cached operator application,
- system metadata used by the driver and diagnostics.

``sfincs_jax/residual.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

- ``sfincs_jax/v3_results.py``:
  typed solve-result dataclasses for linear, Newton-Krylov, and transport-matrix
  v3-compatible workflows. Moving these data models out of the driver makes the
  user-facing result contract explicit and easier to document.
- ``sfincs_jax/solver.py``:
  Krylov solve results, result finite-state checks, and XLA synchronization
  helpers around solver timing/profiling, plus small differentiable JAX-native
  linear algebra kernels such as the regularized tiny least-squares solve and
  recycled Krylov initial-guess builder used by RHSMode=1 and transport solves.
- ``sfincs_jax/solvers/preconditioner_operators.py``:
  diagonal and block-diagonal matrix reductions plus simplified
  preconditioner-operator builders. These are numerical building blocks with
  direct local-coupling tests.
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
- ``sfincs_jax/solvers/preconditioner_context.py``:
  mutable solve-context hints for preconditioner auto-selection, including
  cached operator size, geometry/collision metadata, sparse structural
  tolerance, factor dtype, and solver-JIT admission. The numerical policy lives
  in ``path_policy.py``; this module owns the runtime state bridge.
- ``sfincs_jax/solvers/preconditioner_operators.py``:
  pure dataclass/JAX transformations that build simplified ``V3FullSystemOperator``
  variants used as point, line, domain-decomposition, and Fortran-reduced
  preconditioner matrices. These helpers encode the PETSc/Fortran-v3-style
  ``Pmat`` shaping rules without carrying solve state, caches, or sparse
  factorization logic.
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
  (legacy alias: ``sfincs_jax/rhs1_pas_xblock_ilu.py``):
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
  (legacy alias: ``sfincs_jax/rhs1_xblock_tz_sparse.py``):
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
- ``sfincs_jax/solvers/preconditioners/domain_decomposition/line_blocks.py``:
  angular line-block and restricted-additive-Schwarz preconditioners for
  RHSMode=1 domain-decomposition and strong fallback paths. It owns the
  theta-line, zeta-line, theta-domain, zeta-domain, theta-Schwarz,
  zeta-Schwarz, theta-line-with-``x``-diagonal, and full theta-zeta
  angular-block setup/apply kernels used by automatic line selection,
  Schur-base construction, and explicit strong-preconditioner requests. Shared
  axis-line index maps, cache keys, regularization policy, extra-variable tail
  solves, and multi-level residual correction hooks live here; ``v3_driver.py``
  keeps compatibility wrappers only.
- ``sfincs_jax/solvers/preconditioners/schur/rhs1.py``:
  RHSMode=1 constraintScheme=2 constraint-source Schur preconditioner. It owns
  Schur base-preconditioner selection, diagonal/full/x-coupled Schur inverse
  setup, PAS-ILU Schur shortcuts, constraint-source injection/projection, cache
  population, and reduced/full apply wrappers. ``v3_driver.py`` injects the
  current base builders and keeps a compatibility wrapper only.
- ``sfincs_jax/solvers/preconditioners/schur/rhs1_coarse_policy.py``:
  environment-normalized policy for RHSMode=1 active native-stack and
  sparse-coarse residual preconditioners. It owns side-effect-free parsing for
  memory budgets, Schwarz admission, coarse-equation solver choice,
  field-split output/base-kind routing, and coupled-kinetic admission probes;
  ``rhs1_full_assembly.py`` keeps sparse basis assembly, factorization, cache
  ownership, and true residual admission.
- ``sfincs_jax/solvers/preconditioners/schur/rhs1_coarse_basis.py``:
  reusable RHSMode=1 active coarse residual basis construction for the Schur
  preconditioner family. It owns low-``L``/low-angular-mode config, normalized
  surface modes, sparse basis materialization, targeted ``(species,x,L)``
  window columns, and storage estimates; ``rhs1_full_assembly.py`` keeps only
  aliases at the existing internal call sites plus the matrix/operator-dependent
  adaptive residual-basis stages.
- ``sfincs_jax/solvers/preconditioners/schur/rhs1_full_csr.py``:
  complete full-CSR RHSMode=1 Schur preconditioner family for the explicit
  host solve lane. It owns the shared structured full-CSR preconditioner result
  type, Jacobi fallback, diagonal tail-Schur, zeta-line Schur, pitch-line Schur,
  radial-pitch Schur builders, block memory estimates, and regularized diagonal
  inversion. ``rhs1_full_assembly.py`` now imports the historical private names
  from this module so existing call sites and debug scripts keep working while
  the implementation has direct tests.
- ``sfincs_jax/problems/transport_matrix/direct_pmat.py``
  (legacy alias: ``sfincs_jax/transport_direct_pmat.py``):
  direct term-level RHSMode=2/3 reduced ``Pmat`` and exact active-operator
  emission for full-FP transport preconditioners, plus the physics/source
  coarse-basis columns used by symbolic Schur corrections. The module owns the
  sparse matrix assembly; ``v3_driver.py`` still owns admission, factor choice,
  fallback ordering, and solver orchestration.
- ``sfincs_jax/problems/transport_matrix/direct_block_schur.py``
  (legacy alias: ``sfincs_jax/transport_direct_block_schur.py``):
  bounded-memory direct active block-Schur preconditioner setup for RHSMode=2/3
  full-FP transport. It owns environment parsing, setup-time true-residual
  admission, residual-coarse rescue, cache storage, and host callback
  application, while the driver injects the current fallback preconditioner and
  cache-key policy.
- ``sfincs_jax/problems/transport_matrix/fortran_reduced_lu.py``
  (legacy alias: ``sfincs_jax/transport_fortran_reduced_lu.py``):
  global RHSMode=2/3 full-FP Fortran-reduced sparse-factor preconditioner. This
  module owns the PETSc/Fortran-v3-style reduced ``Pmat`` sparse setup,
  symbolic/BLR/ND/native factor policy, direct-``Pmat`` admission, exact-LU
  rescue, physics coarse correction, and host sparse-factor callback apply.
  ``v3_driver.py`` injects only the current fallback builder, cache-key policy,
  explicit sparse builder seam, and host-memory callback.
- ``sfincs_jax/solvers/preconditioner_setup.py``:
  shared setup utilities for preconditioner construction: memory-bounded
  basis-column chunking, selected-row/selected-column matrix-free submatrix
  probing, unsharded V3-operator probing for setup-time host factors, stable
  array hashes, and RHSMode=1/transport preconditioner cache-key construction.
  The keys intentionally omit RHS-only gradients so fixed-operator scan points
  can reuse factors.
- ``sfincs_jax/solvers/explicit_sparse_factor_policy.py``:
  explicit-sparse host-factor environment parsing, canonical factor-kind alias
  resolution, monolithic LU/ILU guard sizing, and the typed
  ``ExplicitSparseFactorSettings`` bundle consumed by the host sparse builder.
  The sparse builder still owns monkeypatch-sensitive operator/factor
  construction, but the dense/CSR budgets, pattern-color probing, symbolic
  Schur/frontal/ND/BLR settings, SuperLU pivot/permutation options, and ILU
  options are parsed in this directly tested policy layer.
- ``sfincs_jax/solvers/explicit_sparse_factor_builder.py``:
  host explicit-sparse operator assembly, logging, monolithic preflight guard,
  and factorization orchestration. ``v3_driver.py`` injects its current
  ``build_operator_from_matvec``, ``build_operator_from_pattern``,
  ``factorize_host_sparse_operator``, and backend callbacks so existing
  monkeypatch/debug workflows keep exercising the same runtime seam while the
  implementation is directly testable outside the monolith.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/host_factor.py``:
  RHSMode=1 host sparse ILU/LU factor setup used by non-differentiable
  CLI-oriented rescue paths. The module owns matrix-free column assembly,
  structural-threshold application, SuperLU retry/regularization policy, cached
  dense/JAX triangular-factor materialization, and the matrix-free full-system
  adapter used by coarse/Galerkin corrections. ``v3_driver.py`` imports the
  historical private helper names as aliases so existing debugging and
  monkeypatch tests keep using the same seam.
- ``sfincs_jax/solvers/preconditioners/symbolic_sparse/rhs1_fortran_reduced.py``:
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
- ``sfincs_jax/rhs1_direct_tail_policy.py``:
  RHSMode=1 direct-tail structured-preconditioner adapter, direct reduced-Pmat
  aliases, stable cache-key hashing, cache-hit metadata tagging, and adaptive
  direct-tail memory-cap policy. ``v3_driver.py`` imports the same private
  compatibility names so existing debug scripts can still clear the direct-tail
  cache or inspect the policy through the historical driver namespace.
- ``sfincs_jax/rhs1_fortran_reduced_direct_tail.py``:
  RHSMode=1 Fortran-reduced constraintScheme=1 direct-tail sparse-operator
  materialization. The module emits source/tail columns and moment rows from the
  same formulas used by the matrix-free v3 operator, while ``v3_driver.py``
  injects the structured full-CSR builder callback to preserve the existing
  monkeypatch/debug seam and avoid circular imports.
- ``sfincs_jax/rhs1_active_preconditioner_policy.py``:
  active-projected RHSMode=1 full-CSR preconditioner auto-policy. The module
  owns environment parsing for the candidate ladder, large-system fallback
  guard, skipped-fallback metadata, and progress logging default, leaving
  ``rhs1_full_assembly.py`` to dispatch candidates and measure setup results.
- ``sfincs_jax/rhs1_fortran_reduced_factor_policy.py``:
  Fortran-v3-reduced RHSMode=1 active-Pmat factorization policy. The module
  owns factor-kind normalization, large-matrix ILU guards, LU prefill safety
  defaults, SuperLU/RCM ordering candidates, equilibration norm selection, and
  progress logging defaults; the symbolic-sparse RHSMode=1 Fortran-reduced
  module consumes this policy and performs the numerical sparse factor setup.
- ``sfincs_jax/rhs1_symbolic_frontal_policy.py``:
  symbolic frontal/Schur RHSMode=1 active-preconditioner policy. The module
  owns frontal versus nested-dissection routing, separator/block limits, dense
  Schur update budgets, admission probe thresholds, and ND residual-polish
  controls; ``rhs1_full_assembly.py`` keeps sparse symbolic analysis,
  factorization, and true residual admission.
- ``sfincs_jax/rhs1_symbolic_sparse_policy.py``:
  symbolic superblock and separator-Schur RHSMode=1 active-preconditioner
  policy. The module owns grouped-block and block-Schur size gates, ordering
  defaults, separator/coarse limits, retained-cross-fraction gates, prefill
  safety factors, and admission probe thresholds; ``rhs1_full_assembly.py``
  keeps symbolic sparse analysis, host factorization, and true residual
  admission.
- ``sfincs_jax/rhs1_structured_full_csr.py``:
  runtime/non-autodiff wrapper that adapts analytic RHSMode=1 full-CSR assembly
  from ``rhs1_full_assembly.py`` into the ``SparseOperatorBundle`` contract used
  by sparse-PC solver paths. Unsupported or over-budget cases return ``None`` so
  callers can fall back to the established matrix-free or pattern-probed path.
- ``sfincs_jax/rhs1_true_operator_rescue.py``:
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
- ``sfincs_jax/solvers/preconditioner_caches.py``:
  passive dataclasses and global cache registries for RHSMode=1 and
  RHSMode=2/3 preconditioners.  The numerical setup/apply routines still live
  in the driver during this stage, but cache containers are now directly
  importable and tested.  ``v3_driver.py`` re-exports the same registry objects
  under the old private names so existing debugging scripts and tests keep
  clearing the real caches.
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
- ``sfincs_jax/rhs1_pas_policy.py``:
  PAS applicability, PAS-TZ memory safety, PAS fallback routing, and PAS
  adaptive-smoother eligibility.
- ``sfincs_jax/rhs1_pas_matrixfree.py``:
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
  cache/replay/residual routing, generic sparse-PC Krylov execution, final
  generic sparse-PC bundle assembly, and the compatibility import surface used
  by the monolithic solve owner while Tranche B continues. X-block final
  payload builders now live in ``sparse/xblock.py``; ``handoff.py`` only
  re-exports them for compatibility. Optional two-level and global-coupling
  stage contexts accept injected builders for tests, but resolve the canonical
  QI builders themselves in production so ``v3_driver.py`` no longer
  re-exports private QI builder aliases.
- ``sfincs_jax/problems/profile_response/sparse/policy.py``:
  generic sparse-PC policy and admission helpers: active-DOF map construction,
  entry classification, sparse factor policy, conservative-pattern setup,
  memory-budget preflight, factor residual-preflight gates, rescue-candidate
  acceptance, auto-retry selection, GMRES stagnation/post-MinRes controls, and
  the shared sparse env-token parser family used by direct, x-block, QI, and
  Fortran-reduced sparse owners. This module is intentionally independent of
  x-block assembled-operator and QI-device setup so it can stay reusable and
  easy to test.
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
  and initial-guess policy dataclasses, precondition-side/probe-coarse gates,
  device/host Krylov control, optional augmented Krylov setup, GMRES fallback
  routing, work estimates, progress messages, physical-residual measurement,
  post-Krylov post-solve correction/completion orchestration, and final
  x-block sparse-PC diagnostic metadata assembly. The driver-facing handoff
  accepts the solve-local scope, filters it into typed finalization state, and
  returns the final sparse-PC payload used by ``V3LinearSolveResult``. This
  module owns generic x-block stage mechanics; QI-specific coarse-basis choices
  remain in ``sparse/qi.py``.
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
  (legacy alias: ``sfincs_jax/rhs1_solver_diagnostics.py``):
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
- ``sfincs_jax/rhs1_lowmode_coarse.py``:
  low-mode angular, moment, coupled f/tail, and tail-only feature construction
  plus matrix-free Galerkin/least-squares residual-correction builders for
  structured RHSMode=1 f-block preconditioners. The module keeps coarse-space
  algebra independently testable without materializing dense operator bases in
  the driver.
- ``sfincs_jax/rhs1_domain_decomposition.py``:
  deterministic angular domain-decomposition patch ranges, shard-aware block
  sizing, and two-level Schwarz coarse-block heuristics. These rules are kept
  independent of the full operator so multi-device preconditioner policy can be
  tested without launching a solve.
- ``sfincs_jax/problems/profile_response/active_dof.py``
  (legacy aliases: ``sfincs_jax/rhs1_active_dof.py`` and
  ``sfincs_jax/rhs1_active_projection.py``):
  RHSMode=1 active-degree-of-freedom routing and reduced-index-map
  construction for truncated pitch grids, x-block active-DOF opt-ins, and PAS
  constraint-projection solves. The same owner now holds the reusable JAX
  primitives for full-to-reduced gathers, reduced-to-full one-based scatters,
  and PAS ``l=0`` flux-surface-average projection. These primitives are shared
  by RHSMode=1 sparse-PC, x-block active-DOF, and PAS-projected reduced
  residual paths.
- ``sfincs_jax/problems/profile_response/residual.py``
  (legacy alias: ``sfincs_jax/rhs1_residual.py``):
  small residual target, ratio, convergence, and host-scalar norm helpers used
  by RHSMode=1 sparse-PC and x-block diagnostics, plus the physics-aware
  x-block post-coarse direction builder and the bounded host/device subspace
  residual-equation correction kernels used after x-block solves. The module
  also owns residual-correction preconditioner composition, safe
  non-finite/clipped preconditioner wrapping, and scalar preconditioned-minres
  polish. This keeps fail-closed residual-polish algebra testable without
  entering the production driver.
- ``sfincs_jax/rhs1_device_operator.py``:
  bounded JAX-device CSR materialization, active-index slicing, sparse matvec
  closures, and host-vs-device validation utilities for opt-in RHSMode=1
  device-QI and operator-reuse experiments.
- ``sfincs_jax/rhs1_qi_coarse.py``:
  deterministic QI coarse-basis construction, rank/conditioning diagnostics,
  Galerkin/action coarse solves, and synthetic residual-reduction probes used
  by the true device-QI research lane. It also owns the operator-derived
  x-block QI coarse-basis padding, block-geometry metadata, global coarse-load
  direction builders, and smoothed-load QI basis construction that the
  production driver used to pass directly into the QI pipeline.
- ``sfincs_jax/rhs1_qi_galerkin_policy.py``:
  fail-closed Galerkin candidate parsing and true-residual selection. The
  production driver only keeps an experimental QI coarse candidate when this
  policy records a finite material residual reduction.
- ``sfincs_jax/rhs1_qi_two_level.py``:
  reusable local-smoother plus coarse-correction primitive,
  ``S_local^{-1} r + Q A_c^{-1} Q^T (r - A S_local^{-1} r)``, used as a
  directly tested architecture prototype before any hard-seed promotion. It
  also owns the x-block fixed two-level, host smoothed global-coupling, and
  device global-coupling preconditioner wrappers that build the actual
  coarse-action callables used by the sparse/profile-response stage helpers.
  ``v3_driver.py`` no longer imports or re-exports these private builders.
- ``sfincs_jax/rhs1_qi_device_smoother.py``:
  device-local QI smoother primitives, including CSR-backed Jacobi,
  matrix-free residual-minimizing steps, and fail-closed seed probes for the
  differentiable/device QI lane.
- ``sfincs_jax/rhs1_qi_block_schur.py``:
  standalone JAX-compatible QI block-Schur/angular/radial coarse-preconditioner
  primitive. It builds deterministic global, radial, angular, and block-Schur
  basis directions, applies a local-plus-coarse action, and exposes a
  fail-closed true-residual probe for future device-QI expansion.
- ``sfincs_jax/rhs1_qi_deflation.py``:
  residual-deflated, device-compatible QI preconditioner primitive. It builds a
  bounded preconditioned-residual Krylov basis, optionally merges
  physics-informed block-Schur directions, applies a local-plus-deflated
  least-squares action, and fail-closes on true-residual probes. It also
  provides the seed-only cycle-minres helper used by QI hard-seed evidence:
  repeated fixed-basis residual corrections are combined by a small
  ``min ||A Z c - r||`` solve before Krylov starts. The production driver
  exposes it through the opt-in
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER`` hook.
- ``sfincs_jax/rhs1_qi_device_preconditioner.py``:
  production-shaped device-QI field-split state. It builds a device Jacobi local
  smoother when device CSR is available, or a matrix-free coarse-only
  seed-correction path when full CSR is rejected. It can enrich the matrix-free
  basis with residual-generated Krylov vectors ``orth([Q, r, A r, ...])``,
  rank-gates the result, assembles ``A_c`` and ``A Q`` by JAX matvec probes,
  applies a pure-JAX correction, and exposes fail-closed true-residual probe
  metadata. The probe can optionally run a small bounded sequence of corrections
  controlled by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES``; every cycle
  recomputes the true residual and stops on non-finite or non-improving
  candidates. With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=matrix_free_minres``,
  the matrix-free path also gets a bounded residual-polynomial local smoother.
  It applies a fixed number of pure-JAX sweeps selected by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS``
  and scales each residual direction by a small
  ``min ||r - alpha A r||`` step before the coarse correction sees the remaining
  residual. This keeps the action device-compatible and avoids full CSR
  materialization.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=matrix_free_block_minres``,
  the matrix-free path builds residual pieces on x/species blocks from the QI
  block layout and solves a small projected problem
  ``min_c ||r - A D c||_2``. This is the first block/angular/radial local action:
  each block direction keeps all angular content inside the block while the
  small least-squares solve chooses the coupled correction coefficients. The
  block-projection implementation is covered by a transpose-safety regression,
  so this differentiable probe path has a finite ``vjp`` rather than relying on
  a forward-only JAX action. That closes the transpose-safety infrastructure
  blocker for this projected path; it does not close the residual/output or
  runtime-performance blockers for true device-QI.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING=block_x_species``,
  that projected space is augmented with radial-x and species aggregate
  residual directions, which gives the local action a bounded global-coupling
  handle without materializing the full sparse operator.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY=residual_minimizing``,
  the probe line-searches each correction direction by minimizing
  ``||r - alpha A d||_2`` before applying the same fail-closed residual gate.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT``,
  setup appends residuals left by the current coarse correction as additional
  rank-gated GCRO-style seed vectors.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT=1``,
  setup builds a bounded Arnoldi-like residual Krylov coarse space
  ``orth([Q, r, A r, A^2 r, ...])`` from the current true residual and reuses
  the final ``A Q`` action in the small coarse least-squares solve. The depth is
  controlled by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH``.
  This is a stronger operator-reuse coarse construction, not additional
  smoother tuning. The 2026-05-20 installed depth-64 plus multilevel hard-seed
  artifact is the current best checked GPU evidence for this route, but it
  remains below the production gate and is not a closed true device-QI claim.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV=1``
  passes the reusable ``(U, A U)`` QI coarse basis directly into JAX FGMRES so
  each cycle can either project the current residual over the stored operator
  action before Arnoldi starts or, in the default ``combined`` mode, solve the
  restart least-squares problem over ``[A U, A Z]`` and update through
  ``[U, Z]``. This is the current real operator-reuse hook for the device-QI
  lane: it avoids dense global assembly and differs from seed-only correction
  because the basis is active inside the Krylov residual equation. The runner
  preset ``recycled-augmented-device-qi`` applies the same hook with a larger
  fixed device-cycle budget and ``outer_k=32``. The checked GPU0 artifact
  improves the hard-seed residual to ``7.336295e-6`` in ``158.6 s`` but remains
  fail-closed because it does not satisfy the production write tolerance.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION=1``
  enables the deeper multilevel coarse-residual path. Setup builds separate
  per-level bases from the same QI block layout, caches each ``A Q_l`` action,
  and applies a staged residual equation ``min ||r_l - A Q_l c_l||`` before the
  optional global coarse polish. The level rank, order
  (``coarse_to_fine`` or ``fine_to_coarse``), and global-polish toggle are
  controlled by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK``,
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER``,
  and
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL``.
  The runner preset ``coarse-residual-device-qi`` records these controls for
  bounded hard-seed evidence. This is a coarse-grid residual-equation attempt,
  not a smoother/restart/projection tuning path.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT=1``
  adds the current hard-seed residual itself to the reusable coarse equation by
  restricting it to QI blocks and aggregates. With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT=1``,
  setup also adds adjoint-normal block snapshots ``A^T r_block`` through JAX
  VJP. These directions are setup-time only: the installed preconditioner still
  caches ``A Q`` and applies a forward ``min ||r - A Q c||`` solve. The runner
  preset ``residual-snapshot-device-qi`` is the checked evidence path; its first
  CPU hard-seed artifact improves the final residual to ``2.103015e-5`` but is
  still nonconverged, so it is not a production default.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION=1``
  promotes those residual snapshots into a staged residual-equation cascade.
  Setup solves per-stage ``min ||r_l - A Q_l c||`` problems, caches the
  accepted ``A Q_l`` actions, and the installed preconditioner remains pure JAX.
  The runner preset ``residual-snapshot-equation-device-qi`` records this path.
  Its first scale-0.60 CPU hard-seed artifact accepts
  ``3.021487e-5 -> 2.819970e-5`` but ends at ``2.320763e-5`` in ``260 s``,
  worse than the plain residual-snapshot path, so no production default or GPU
  claim is made from it.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION=1``
  enables a deeper staged block-Schur residual equation. Setup builds
  QI-block and aggregate source probes, solves small setup-time
  ``min ||r_l - A D_g c||`` problems, rank-gates each accepted correction, and
  then reuses cached ``A Q_l`` actions in the installed preconditioner. It now
  also tests a coupled block/aggregate source space and keeps the lower
  measured setup residual between the coupled and sequential constructions. The
  runner preset ``block-schur-device-qi`` records this fail-closed path. Its
  first scale-0.60 CPU hard-seed artifact is negative evidence: it accepts
  ``3.021487e-5 -> 2.840342e-5`` and ends at ``2.275188e-5`` in ``267 s``,
  worse than the residual-snapshot path, so no production default or GPU claim
  is made from it.
  The GPU0 best-of artifact improves the final residual to ``1.992464e-5`` in
  ``292 s`` but still fails the production write gate. The adaptive local
  smoother token ``adaptive_residual_equation`` maps to a multilevel
  ``block_hierarchy`` grouping, preserving global, aggregate, and block residual
  source spaces in the matrix-free projected residual smoother. Its first GPU0
  artifact ends at ``2.307995e-5`` in ``288 s``, so it is retained as an
  opt-in negative-evidence path rather than a default. The composite
  ``composite-closure-device-qi`` runner preset combines residual snapshots,
  residual-Galerkin/operator-image stages, and block-Schur residual equations;
  its GPU1 artifact accepts ``3.021487e-5 -> 2.575099e-5`` but ends worse at
  ``2.305955e-5``, so it remains negative evidence.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION=1``
  enables a global moment closure over profile, current, and reduced-tail
  constraint moments. Setup builds a rank-gated Galerkin Schur closure and
  installs only cached ``A Q`` actions. The checked scale-0.60 CPU artifact
  accepts ``3.021487e-5 -> 2.840364e-5`` and ends at ``2.420524e-5`` in
  ``256 s`` before refusing nonconverged output, so it remains fail-closed
  research evidence.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION=1``
  enables a residual-derived Galerkin coarse equation. Its coarse variables are
  built from the actual remaining residual and block residuals, then cached as
  ``A Q`` for device-compatible apply. The checked scale-0.60 CPU artifact
  accepts ``3.021487e-5 -> 2.766710e-5`` with rank ``16`` from ``21``
  candidates and ends at ``2.632208e-5`` in ``244 s`` before refusing
  nonconverged output. This is stronger than static moments but weaker than the
  residual-snapshot path, so it is not promoted.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT=1``
  adds a distinct adjoint-normal coarse space ``orth([A^T r, (A^T A)A^T r,
  ...])``. This targets non-normal left-error modes that residual-Krylov can
  miss. The transpose is setup-only through JAX VJP, controlled by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH``
  and
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE``;
  apply-time still reuses cached ``A Q`` and remains forward-operator-only.
  The scale-0.60 CPU hard-seed depth-2 artifact worsened the final residual, so
  this remains an explicit diagnostic/negative-evidence path rather than a
  recommended production path.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS=1``
  prepends bootstrap-current-like pitch moments, species/radial current
  moments, and reduced-tail constraint moments to the multilevel coarse basis.
  This is a structural coarse-space probe for flow/current/nullspace error, not
  a smoother knob. The 2026-05-20 GPU hard-seed artifact increased the rank from
  ``13`` to ``15`` but worsened the final residual and runtime, so it remains an
  opt-in negative-evidence path rather than the recommended preset.
  With
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT=1``,
  setup can also enrich the correction space with rank-gated operator images
  ``{Q, A Q, A^2 Q, ...}``, controlled by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH``.
  That path is retained as an opt-in diagnostic because it is CUDA-safe but did
  not improve the scale-0.60 GPU hard seed.
  The driver exposes it behind the explicit
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER`` opt-in and records
  accepted/rejected metadata without changing public defaults; the matrix-free
  fallback is separately gated by
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE``.
  Seed-only use is allowed when ``precondition_side=none``; Krylov installation
  is blocked in that mode. When matrix-free QI-device Krylov use is explicitly
  requested, the driver disables the automatic non-autodiff host fallback so the
  true device path can be tested. Explicit user-forced host fallback still wins.
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_WITH_BASE``
  is an opt-in diagnostic that applies the pre-existing x-block/device
  preconditioner first and then applies QI to the residual left by that base
  step. The scale-0.60 GPU hard seed showed this composition is slower and less
  accurate than the installed multilevel route, so it is intentionally not part
  of the recommended preset.
  The current installed hard-seed summaries are still failed/nonconverged
  blocker evidence with no HDF5 or solver trace output, so this routing change
  is not a validated true device-QI claim.
  Closure-state summary: the implemented QI lanes now include
  residual-deflated seed correction, device coarse reuse, augmented FGMRES
  operator reuse, augmented-seed Krylov coarse-space recycling, multilevel
  residual equations, block-Schur residual equations, global moment closure,
  residual-Galerkin closure, phase-space coarse reuse, and
  residual-region/bounce-region coarse reuse. They are opt-in research/evidence
  lanes, not public true-device-QI defaults. The practical non-autodiff
  host/x-block route is a separate large-QI fallback and must not be described
  as differentiable/device-QI closure. The negative evidence set includes the
  checked smoother/restart variants, assembled CSR reuse, phase-space and
  residual-bounce coarse probes, composite closure, global moment,
  residual-Galerkin, block-Schur, current/nullspace moment enrichments, and
  current augmented-seed and active-pattern hard-seed probe plumbing; each
  either failed to improve the best hard-seed result or remained far above the
  write gate. The aggregate
  manifest records failed/nonconverged artifacts as requested-only classes for
  promotion gating and preserves observed fail-closed machinery in separate
  metadata fields. Promotion still requires a hard-seed artifact with converged
  HDF5 output, solver trace metadata, accepted-converged status,
  residual/write-gate satisfaction, no host fallback, CPU/GPU consistency,
  promotion-eligible manifest classification, and then wider
  production-resolution seed/backend coverage. The active-pattern chunked
  coarse primitive is now wired through the device preconditioner, driver,
  runner, manifest, and tests, but the office GPU hard-seed artifact is
  promotion-negative. The next promotion attempt therefore needs a deeper
  coupled Schur/residual equation over accepted bounce/residual regions rather
  than another local smoother, restart, or basis-only knob.
- ``sfincs_jax/rhs1_qi_multilevel_coarse.py``:
  standalone multilevel angular-radial-pitch-current coarse prototype for the next true
  device-QI architecture. It constructs deterministic radial aggregate levels,
  angular harmonic directions, radial polynomial directions, pitch/xi moment
  directions, current/flow pitch moments that tolerate variable active xi
  counts, reduced-tail constraint moments, and radial-angular/radial-pitch product modes from
  ``RHS1QICoarseBlockLayout`` metadata,
  rank-gates the combined prolongation space, assembles ``A Q`` by JAX matvec
  probes, and applies a pure-JAX local-plus-coarse action-least-squares
  correction. The module is intentionally independent of ``v3_driver.py`` so it
  can be tested as architecture before promotion. Current tests cover
  deterministic hierarchy metadata, synthetic low-frequency angular-radial
  residual reduction, synthetic pitch-coupled residual reduction, synthetic
  current/tail residual reduction, fail-closed rejection without the needed
  coarse family, nested per-level residual-equation recovery of modes discarded
  by a flat coarse-rank gate, and JIT/gradient compatibility.
  It is not a production default until real scale-0.60 CPU/GPU hard-seed
  artifacts pass the promotion gates.
- ``sfincs_jax/rhs1_qi_global_moment_closure.py``:
  standalone global-moment closure primitive used to test current/profile/tail
  moment Schur closures independently from the production driver. It builds a
  compact moment basis, caches ``A Q`` and ``Q^T A Q``, and fails closed unless
  the measured setup residual improves.
- ``sfincs_jax/rhs1_qi_residual_galerkin.py``:
  standalone residual-derived Galerkin primitive. It builds staged coarse
  variables from the current operator residual and block residuals, caches
  ``A Q``, supports action least-squares or Galerkin solves, and fails closed on
  non-improving setup residuals.
- ``sfincs_jax/rhs1_qi_phase_space_coarse.py``:
  standalone deterministic phase-space coarse-space builder for the true
  device-QI research lane. It derives trapped/passing-like pitch bands,
  boundary bands, even/odd pitch-parity directions, and optional radial/species
  aggregates from ``RHS1QICoarseBlockLayout`` metadata, rank-gates them, and
  plugs into the existing setup-time residual-equation path. The controls
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION*``
  are opt-in and fail-closed; they are used only by explicit research probes
  such as ``phase-space-coarse-reuse-device-qi`` until scale-0.60 GPU hard-seed
  artifacts write HDF5 output, solver traces, and accepted-converged residual
  metadata.
- ``sfincs_jax/rhs1_qi_residual_region_coarse.py``:
  standalone residual-region / bounce-region coarse-space builder for hard
  RHSMode=1 QI seeds. It uses the setup residual to select energetic block,
  trapped/boundary/passing, radial, and species regions from
  ``RHS1QICoarseBlockLayout`` metadata, rank-gates residual-restricted columns,
  and plugs into the existing cached ``Q`` / ``A Q`` residual-equation path.
  The runner preset ``residual-bounce-region-device-qi`` emits explicit
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE*``
  controls, records them in manifests, and classifies solver-trace or
  failure-progress metadata. It remains opt-in and fail-closed, not a
  production claim, until scale-0.60 CPU/GPU hard-seed artifacts write
  converged HDF5 output and solver traces under the same promotion gates.
- ``sfincs_jax/rhs1_qi_active_pattern_coarse.py``:
  standalone residual active-pattern coarse-space builder for hard RHSMode=1
  QI seeds. It selects high-energy pitch, angular, radial, and species residual
  chunks from ``RHS1QICoarseBlockLayout`` metadata, rank-gates them, and now
  plugs into the same device-compatible cached ``Q`` / ``A Q``
  residual-equation path as the other true device-QI research probes. The
  runner preset ``active-pattern-device-qi`` emits
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE*``
  controls and keeps the lane fail-closed until bounded CPU/GPU hard-seed
  artifacts write converged output and solver traces.
- ``sfincs_jax/rhs1_qi_coupled_residual.py``:
  standalone coupled residual-equation primitive for the next true device-QI
  architecture. It takes accepted coarse bases from the existing
  block-Schur/multilevel/moment/residual families, re-orthonormalizes them into
  one joint coarse space, probes ``A Q`` once, and solves one action
  least-squares or Galerkin residual equation. This is intentionally different
  from smoother/restart tuning and from the previous staged residual cascade:
  the joint solve can update earlier coarse coefficients after later Schur or
  multilevel variables are included, matching the field-split/Schur and
  Petrov-Galerkin ideas used in PETSc-style block preconditioners. The primitive
  is wired into ``RHS1QIDevicePreconditionerConfig`` as
  ``coupled_residual_equation`` and is fail-closed unless setup residual
  decreases.
- ``sfincs_jax/problems/profile_response/sparse/qi.py``,
  ``sfincs_jax/v3_driver.py``, and ``scripts/run_qi_seed_robustness.py``:
  expose the coupled residual equation through
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION*``
  controls, progress logs, solver-trace keys, and the
  ``coupled-residual-device-qi`` hard-seed probe preset. The runner classifies
  this as a joint coupled-equation route rather than as another staged
  block-Schur or active-pattern variant. The optional
  ``*_INSTALL_IN_KRYLOV_ON_REJECT`` control installs a validated coupled stage
  as a Krylov preconditioner without changing ``x0`` when the one-shot seed
  probe is rejected, matching field-split preconditioner semantics. The runner
  also keeps coupled residual-equation and install-in-Krylov progress lines as
  sticky compact-log events, so failed GPU artifacts still preserve the actual
  preconditioner path while remaining fail-closed.
- ``sfincs_jax/rhs1_qi_promotion.py``:
  pure promotion gates for QI hard-seed and production-ladder evidence. It
  requires complete seed/backend coverage, convergence, output and trace
  provenance, residual/observable bounds, and no host fallback before a true
  device-QI claim can be promoted.
- ``sfincs_jax/rhs1_preconditioner_dispatch.py``:
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/rhs1_preconditioner_auto_policy.py``:
  RHSMode=1 preconditioner environment alias normalization plus bounded
  automatic preconditioner policy predicates for PAS, DKES, tokamak, GPU sparse
  fallback, weak-default PAS promotion, PAS-family refinement, FP/DKES routing,
  large-FP near-zero-Er overrides, and sharded line-overrides.
- ``sfincs_jax/rhs1_schur_policy.py``:
  RHSMode=1 Schur base-preconditioner alias normalization and automatic
  geometry/PAS/DKES routing policy.
- ``sfincs_jax/problems/profile_response/policies.py``
  (legacy aliases: ``sfincs_jax/rhs1_acceptance_policy.py``,
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
  (legacy aliases: ``sfincs_jax/rhs1_strong_policy.py``,
  ``sfincs_jax/rhs1_strong_control.py``, and
  ``sfincs_jax/rhs1_strong_auto_kind.py``):
  strong-preconditioner request mapping, enable/disable control, automatic
  strong-kind selection, and post-selection adjustment policy.
- ``sfincs_jax/rhs1_strong_fallback.py``:
  compatibility facade for historical RHSMode=1 strong-preconditioner fallback
  imports. The implementation owner is now
  ``sfincs_jax/problems/profile_response/preconditioner_build.py``.
- ``sfincs_jax/problems/profile_response/handoff.py``
  (legacy alias: ``sfincs_jax/rhs1_handoff.py``):
  accepted-candidate handoff and Krylov replay-state updates. This is the
  source-mapped seam for the repeated RHSMode=1 driver pattern: compare a
  rescue/refinement candidate against the incumbent residual, apply optional
  measured solver-candidate gates, preserve the accepted residual vector, and
  update the KSP replay metadata only after a strict finite residual
  improvement.
- ``sfincs_jax/rhs1_constraint_sources.py``:
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
- ``sfincs_jax/rhs1_large_cpu_policy.py``:
  large explicit full-FP CPU sparse rescue, x-block seed, exact-LU promotion,
  host x-block assembly, and species-x-block rescue policy.
- ``sfincs_jax/rhs1_xblock_policy.py``:
  pure x-block sparse-PC routing, Krylov-side selection, local factorization
  tuning, lower-fill acceptance gates, and non-autodiff device-host fallback
  metadata for large RHSMode=1 QI/full-FP solves.
- ``sfincs_jax/rhs1_xblock_sparse_host_policy.py``:
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
  (legacy alias: ``sfincs_jax/transport_policy.py``):
  pure transport backend, sparse-direct, host-GMRES, dtype, and recycle policy.
  ``TransportRuntimePolicy`` binds those pure decisions to the active JAX
  backend and host sparse-factor dtype provider, so ``v3_driver.py`` no longer
  carries private transport backend-injection wrapper functions.
- ``sfincs_jax/problems/transport_matrix/preconditioner_dispatch.py``
  (legacy alias: ``sfincs_jax/transport_preconditioner_dispatch.py``):
  shared transport preconditioner-kind normalization, auto-selection, DD/sparse-JAX
  env parsing, and reduced/full preconditioner builder dispatch.
- ``sfincs_jax/problems/transport_matrix/solve_policy.py``
  (legacy alias: ``sfincs_jax/transport_solve_policy.py``):
  shared RHSMode=2/3 initial solve policy, active-DOF transport policy,
  active-index map construction, dense fallback, dense-preconditioner, low-memory
  output, streamed-diagnostic, state-vector retention, and GMRES restart policy
  used before the transport preconditioner and solve handoff layers. It also owns
  the per-``whichRHS`` loop policy for E_parallel loose/Krylov routing,
  constraint-nullspace projection admission, KSP iteration-stat settings, and
  dense-batch fallback admission.
- ``sfincs_jax/problems/transport_matrix/setup.py``
  (legacy alias: ``sfincs_jax/transport_solve_setup.py``):
  side-effect-light RHSMode=2/3 setup resolution for transport max-iteration
  overrides, optional Krylov state-file loading/merging, ``whichRHS`` subset
  normalization, and CPU/GPU process-parallel worker requests. The driver emits
  the returned notes and keeps solve orchestration, while these pure setup rules
  are covered by direct unit tests.
- ``sfincs_jax/problems/transport_matrix/active_dense.py``
  (legacy alias: ``sfincs_jax/transport_active_dense_setup.py``):
  combined RHSMode=2/3 active-DOF and dense-path setup. It resolves the initial
  output/restart policy, active-index compaction state, dense fallback and dense
  preconditioner admission, and ordered user-facing notes before the transport
  loop builds matvecs or preconditioners. It also owns the active full-vector
  DOF index helper used when the driver needs to mirror Fortran's reduced
  active transport unknown layout.
- ``sfincs_jax/problems/transport_matrix/sparse_direct_solve.py``
  (legacy alias: ``sfincs_jax/transport_sparse_direct_solve.py``):
  RHSMode=2/3 sparse-direct rescue implementation. It owns sparse-pattern
  admission/caching, direct active FP operator factor reuse, explicit sparse
  helper materialization, fallback sparse-ILU setup, host iterative refinement,
  float32 polish, and float64 retry while receiving driver-local builders as
  explicit callbacks.
- ``sfincs_jax/problems/transport_matrix/streaming_outputs.py``
  (legacy alias: ``sfincs_jax/transport_streaming_outputs.py``):
  host-side streaming accumulator for RHSMode=2/3 transport diagnostics. It owns the
  per-``whichRHS`` NumPy buffers, NTV/source handling, and final output-field
  dictionary assembly used by low-memory transport solves, keeping HDF5-layout details
  out of the solver loop.
- ``sfincs_jax/problems/transport_matrix/postsolve_diagnostics.py``
  (legacy alias: ``sfincs_jax/transport_postsolve_diagnostics.py``):
  post-solve RHSMode=2/3 transport diagnostics orchestration. It chooses streamed
  versus batched diagnostics, applies the rematerialization/precompute/chunking
  environment policy, assembles species-by-``whichRHS`` flux arrays, and returns
  the transport matrix plus optional output fields. This keeps diagnostic memory
  policy out of the main Krylov solve loop.
- ``sfincs_jax/problems/transport_matrix/diagnostics.py``
  (legacy alias: ``sfincs_jax/transport_matrix.py``):
  JAX formulas for RHSMode=1 output moments, RHSMode=2/3 transport diagnostics,
  transport-matrix assembly, strict Fortran-order reductions, and cached
  geometry/species diagnostic precomputes.
- ``sfincs_jax/problems/transport_matrix/handoff_policy.py``
  (legacy alias: ``sfincs_jax/transport_handoff_policy.py``):
  shared transport retry residual metrics, better-candidate comparisons, and RHSMode=3
  polish threshold/restart/maxiter policy used by the reduced and full transport solve
  branches.
- ``sfincs_jax/problems/transport_matrix/residual_quality.py``
  (legacy alias: ``sfincs_jax/transport_residual_quality.py``):
  fast transport worker residual-abort threshold parsing and failure-message
  formatting for absolute and RHS-normalized diagnostics.
- ``sfincs_jax/problems/transport_matrix/dense_lu.py``
  (legacy alias: ``sfincs_jax/transport_dense_lu.py``):
  cached dense-LU solver and preconditioner construction used by bounded transport
  dense fallback and dense-preconditioner paths.
- ``sfincs_jax/problems/transport_matrix/dense_batch.py``
  (legacy alias: ``sfincs_jax/transport_dense_batch.py``):
  batched dense RHSMode=2/3 transport solve helper. It owns all-RHS dense matrix
  assembly, active-DOF reduction/expansion, optional streamed diagnostic
  collection, residual bookkeeping, and per-``whichRHS`` progress emission for
  the bounded dense branch.
- ``sfincs_jax/problems/transport_matrix/active_factor.py``
  (legacy alias: ``sfincs_jax/transport_active_factor.py``):
  active-operator block-Schur and residual-coarse factors for RHSMode=2/3
  transport. It owns symbolic active block ordering, bounded numerical block
  inverses, source/constraint Schur closure, and setup-time true-residual
  admission before any factor is eligible for production use.
- ``sfincs_jax/problems/transport_matrix/host_gmres.py``
  (legacy alias: ``sfincs_jax/transport_host_gmres.py``):
  host SciPy GMRES first-attempt/rescue solve helper for explicit transport paths,
  including PETSc-like preconditioned-residual acceptance for the relevant
  near-singular transport systems.
- ``sfincs_jax/problems/transport_matrix/iteration_stats.py``
  (legacy alias: ``sfincs_jax/transport_iteration_stats.py``):
  optional small-system SciPy Krylov history reruns used only for transport
  ``ksp_iterations`` progress diagnostics. Diagnostic failures are reported but
  never change the production solve result.
- ``sfincs_jax/problems/transport_matrix/linear_solve.py``
  (legacy alias: ``sfincs_jax/transport_linear_solve.py``):
  transport RHSMode=2/3 Krylov dispatch, including the transport-specific
  ``auto``/``default`` BiCGStab preference, implicit custom-solve routing,
  JIT/non-JIT solver selection, restart-budget policy, and distributed-axis
  residual-solve routing.
- ``sfincs_jax/problems/transport_matrix/loop.py``
  (legacy alias: ``sfincs_jax/transport_loop_support.py``):
  loop-local RHSMode=2/3 support for cached full/reduced transport matvec
  closures and bounded Krylov recycle bases. It owns recycle-size admission,
  stored-state seeding, reduced/full recycle-vector trimming, and small recycled
  initial guesses plus sequential residual-gate and ETA progress bookkeeping, so
  the main solve loop no longer carries this mutable cache/progress state.
- ``sfincs_jax/problems/transport_matrix/finalize.py``
  (legacy alias: ``sfincs_jax/transport_solve_finalization.py``):
  sequential RHSMode=2/3 per-``whichRHS`` finalization after a solver branch has
  accepted a candidate. It owns reduced/full state bookkeeping, optional
  constraint projection, true-residual recomputation, streamed-output
  collection, recycle-basis updates, and optional KSP iteration-stat dispatch.
  Dense fallback accepted-state overrides are explicit so the refactor preserves
  the established active-DOF branch behavior.
- ``sfincs_jax/problems/transport_matrix/parallel/policy.py``
  (legacy alias: ``sfincs_jax/transport_parallel_policy.py``):
  pure transport process-parallel backend selection, worker-count validation,
  benchmark scaling audits, process-pool cache keys, GPU-worker environment
  isolation, XLA worker flag rewriting, and multiprocessing fallback policy.
- ``sfincs_jax/problems/transport_matrix/parallel/runtime.py``
  (legacy alias: ``sfincs_jax/transport_parallel_runtime.py``):
  transport parallel RHS partitioning, injected-dependency payload
  normalization, child-worker guard setup, merge-ready result packing, GPU
  worker NPZ conversion, persistent process-pool caching, worker-environment
  setup, backend-specific execution/retry/fallback, GPU subprocess launch, and
  parent-side merge of per-worker state/residual/elapsed-time results. This
  module absorbed the old payload, pool, execution, solve, and validation
  micro-files.
- ``sfincs_jax/problems/transport_matrix/parallel/sharding.py``
  (legacy alias: ``sfincs_jax/transport_parallel_sharding.py``):
  pure single-case sharded-solve planning metadata. It caps requested device
  counts, records per-device workload balance, estimates whether setup and
  Krylov communication can be amortized, marks single-case sharding as
  experimental/non-release by default, and prevents malformed sharded payloads
  from becoming release scaling claims.
- ``sfincs_jax/problems/transport_matrix/parallel/worker.py``
  (legacy executable wrapper: ``sfincs_jax/transport_parallel_worker.py``):
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
- ``sfincs_jax/solvers/progress.py``:
  user-facing duration formatting, coarse runtime hints, one-shot large RHSMode=1
  progress messages, and transport whichRHS ETA text. This module is intentionally
  lightweight so CLI progress can stay informative without importing heavy solver
  dependencies. It is solver-neutral: it improves observability without affecting
  numerical decisions.
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
The legacy ``sfincs_jax/transport_matrix.py`` path remains a compatibility
alias for existing user scripts.

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
  ``sfincs_jax/v3_driver.py`` + ``sfincs_jax/solver.py``
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
