# SFINCS_JAX Final Research-Grade Implementation Plan

Last updated: 2026-06-26 (coverage, source-layout, and README guard pass)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review surface: PR #8, `refactor/v3-driver-architecture`, review-ready after
the consolidation pass below is completed and the remaining non-blocking
production-performance lanes are explicitly documented.

Status: this file is the controlling completion plan. `plan.md` remains the
execution log and historical record. The package/examples consolidation plan
below supersedes older iteration/tranche notes. Avoid one-helper refactor
commits, new implementation shards, and new root modules unless the same commit
deletes or merges a larger owner.

## One-Sentence Goal

Finish `sfincs_jax` as a compact, domain-organized, production-grade
neoclassical transport code: users provide a geometry and input file and get
accurate CPU/GPU results with automatic robust solver selection, while Python
users can opt into end-to-end differentiable residual, flux, ambipolar, and
optimization workflows with parity against SFINCS Fortran v3 wherever the
physics models overlap.

## Authoritative Consolidation Pass

This is the only active refactor plan for making PR #8 reviewable. The prior
`v3_driver.py` split is functionally complete enough to proceed: the driver is
a thin compatibility entry point, nested source packages have been flattened,
and the examples tree has a tutorial-led learning surface. The remaining
review blockers are coverage, final root-helper consolidation, benchmark
regeneration, and a final docs/README consistency pass.

### Current Audit Snapshot

Checked on 2026-06-26 from
`refactor/rhs1-full-assembly-preconditioners` after the profile-response,
transport-matrix, solver-preconditioner, and tutorial/examples tranches.

- `sfincs_jax/` contains `143` Python files after adding one-file
  compatibility shims for `operators/profile_response.py` and
  `problems/profile_response.py` / `problems/transport_matrix.py`, adding a
  one-file compatibility index at `solvers/preconditioners.py`, flattening the
  former nested operator modules into `operators/profile_*.py`, flattening the
  RHSMode=1 problem modules into `problems/profile_*.py`, flattening RHSMode=2/3
  transport modules into `problems/transport_*.py`, and flattening all
  preconditioner families into `solvers/preconditioner_*.py`.
- Largest source domains by line count are `problems` (`69k` lines), `solvers`
  (`47k`), and `operators` (`21k`).
- Empty root packages `sfincs_jax/benchmarks`, `sfincs_jax/compat`,
  `sfincs_jax/input`, and `sfincs_jax/parallel` were removed in the first
  consolidation tranche. The former nested
  `sfincs_jax/solvers/preconditioners/*` tree was removed from the working
  tree.
- No nested source package directories remain below `sfincs_jax/`; the
  source-tree guard keeps `temporary_nested_packages` empty.
- `v3_driver.py` is a 47-line compatibility shim that imports the domain-owned
  profile and transport solve modules, exposes former legacy names, and contains
  no physics or solver implementation.
- `sfincs_jax/README.md` documents the one-level source layout, stable public
  root modules, temporary compatibility roots, release-data manifest policy,
  and contributor move rules. The tiny release-data manifest lives under
  `validation/`, so there is no separate `data/` source folder.
  `tests/test_source_tree_consolidation.py`
  verifies that this README mentions every allowed root package and module.
- `examples/tutorials/` is the first learning surface. It contains three
  output-free workflow notebooks, a `00_start_here.ipynb` chooser notebook, and
  a fast script that writes HDF5/NetCDF/NPZ output and a diagnostics PDF.
  Existing topic folders remain as canonical runnable workflows to avoid
  unnecessary file churn.
- `examples/README.md` contains a task-based chooser, a learning path, and a
  complete folder map. `tests/test_examples_tree_contract.py` enforces the
  top-level folder set, README coverage, first-task entry points, generated-file
  exclusions, and the no-large-example-file policy.
- The root README no longer matches the explicit progress-language patterns
  flagged in the review prompt. Remaining historical/status language should be
  handled in docs pages and release notes rather than the first-page README.
- A repository-size audit found no tracked files larger than 2 MB. Local ignored
  caches (`docs/_build`, `.pytest_cache`, `.ruff_cache`, and `__pycache__`
  folders) were removed; ignored benchmark input caches remain local-only and
  are not part of a user clone or wheel.
- A fresh xdist coverage audit on this branch measured `83.053%`
  (`57,435 / 69,155` covered lines). The most recent local audit wrote
  `.coverage` and `coverage.xml`, but local focused JAX coverage invocations
  still sometimes abort with exit code `134` after the tests pass; use the
  full-suite report as the authoritative local number for this tranche, so the
  CI floor remains `80%` until the sharded GitHub Actions runtime is
  verified and the measured coverage has enough margin above the next floor.
  The target remains `95%` meaningful package coverage with GitHub Actions
  under `10 min`; this requires targeted unit, numerical, and frozen-reference
  tests, not slow full-solve CI jobs.
- A post-twentieth-tranche local coverage attempt completed `959` tests in
  `6:21` before hanging in JAX array teardown and was interrupted; a file-scoped
  pytest-cov run for `outputs/writer.py` also aborted with exit code `134`.
  Until the local coverage runner is stable, use normal fast pytest suites plus
  GitHub Actions coverage as the acceptance signal and keep adding targeted
  high-value tests from the static gap list.
- The largest coverage gaps by missing executable lines are
  `problems/profile_solve.py`, `problems/transport_solve.py`,
  `operators/profile_system.py`, `outputs/writer.py`,
  `solvers/preconditioner_transport_matrix.py`, `solvers/explicit_sparse.py`,
  `operators/profile_sparse_pattern.py`, `solvers/preconditioner_schur_profile.py`,
  and `operators/profile_true_operator_rescue.py`.
- The first post-audit coverage tranche added bounded PAS x-block ILU tests for
  inapplicable-model fallback, tiny-fixture padded factor construction,
  environment normalization, cache reuse, and reduced-vector application. That
  module is no longer a top coverage blocker.
- The second post-audit coverage tranche added synthetic full-FP kinetic
  preconditioner tests for diagonal, x-block, species/x-block, low-rank,
  environment-normalization, cache-reuse, and reduced-vector application paths.
  These tests exercise the same collision-matrix algebra used by the production
  solver without adding slow geometry solves to CI.
- The third post-audit coverage tranche added synthetic PAS angular
  preconditioner tests for tokamak theta-line and 3D theta-zeta line
  block-Thomas paths. The tests cover inactive pitch masking, structured-tail
  factors, memory/fallback admission, cache reuse, and reduced/full projection
  consistency while keeping CI execution below a few seconds.
- The fourth post-audit coverage tranche added synthetic RHSMode=2/3 transport
  preconditioner tests for collision diagonal, species/x block, low-rank,
  x-grid coarse correction, Fourier angular, and FP Fourier paths. These tests
  validate the numerical preconditioner algebra and reduced/full projection
  behavior without running a full transport solve.
- The fifth post-audit coverage tranche added output-streaming schema tests,
  periodic-stencil guard tests, release-data fixture safety tests, PETSc binary
  reference-reader tests, upstream-postprocessing wrapper tests, and active
  RHSMode=1 full-FP kinetic block-preconditioner tests. It also fixed short-file
  PETSc reader errors so malformed references fail with deterministic messages.
- The sixth post-audit coverage tranche added explicit-sparse admission tests
  for CSR matvec validation, forced storage paths, SuperLU storage accounting,
  deterministic probe edge cases, and sparse-factor setup rejection modes. It
  also added profile-system support tests for sharding policy, JAX-transform
  cacheability, full-vector padding/unpadding, analytic source moments, and
  RHSMode 2/3 transport-drive updates.
- The seventh post-audit coverage tranche added profile-solve refactor-boundary
  tests for Krylov dispatch hooks, distributed-axis policy injection, cache-key
  dtype injection, Schur builder wiring, transport FP preconditioner wiring, and
  dense fallback delegation. These protect the extracted owner modules without
  running slow RHSMode=1 production solves.
- The eighth post-audit coverage tranche expanded public grid tests for SFINCS
  v3 `uniformDiffMatrices` branches: include-xmax periodic schemes, aperiodic
  one-sided schemes, grid-placement/quadrature conventions, and finite-size
  guards. These are fast numerical tests anchored in the v3 finite-difference
  stencil definitions and do not add solve time to CI.
- The ninth post-audit coverage tranche added output-writer policy and geometry
  helper tests for eager-precompile opt-in handling, RHSMode=1 solver-method
  overrides, Phi1 frozen-linearization/history alignment, Fortran logical
  encoding, geometryScheme=4 radial-coordinate conversion, and Boozer
  Fourier-derivative evaluation. These tests pin output semantics and
  geometry algebra without launching full solves.
- The tenth post-audit coverage tranche added Schur/coarse policy tests for
  active-native stack memory budgeting, field-split sparse-coarse family
  classification, coupled-kinetic admission guards, and sparse residual-coarse
  base selection. These are fast numerical-policy gates for the production
  RHSMode=1 preconditioner defaults.
- The eleventh post-audit coverage tranche added true-operator rescue tests for
  residual-window additive/least-squares corrections, reusable one-hot action
  cache bypass behavior, active-submatrix residual damping, and coupled coarse
  LSQ correction algebra on identity systems. These tests directly exercise
  residual-polish math without constructing a full kinetic operator.
- The twelfth post-audit coverage tranche added RHSMode=2/3 transport
  preconditioner admission tests for memory-cap fallbacks, bounded FP line
  factors, Schur disable paths, reduced-view wrappers, and structured f-block
  factor metadata. These tests protect production solver selection and memory
  admission without materializing a production operator.
- The thirteenth post-audit coverage tranche added explicit-sparse solver tests
  for coarse-correction no-op guards, monotone direct refinement, sparse
  refinement, and GMRES-polish callback wiring. These protect the native sparse
  factor infrastructure used by production preconditioners without running a
  full production solve.
- The fourteenth post-audit coverage tranche added transport linear-solve
  routing tests for implicit residual solves, plain/JIT GMRES residual routes,
  distributed-axis dispatch, and callback binding. These keep RHSMode=2/3
  solver orchestration covered without launching expensive transport examples.
- The fifteenth post-audit coverage tranche added RHSMode=1 radial x-grid
  preconditioner tests for FP x-multigrid factors, PAS+Er x-upwind dispatch,
  reduced-vector application, cache reuse, finite factor values, and tail
  preservation. These tests exercise the radial preconditioner algebra used by
  production bootstrap-current solves without adding production solves to CI.
- The sixteenth post-audit coverage tranche added research-lane manifest policy
  tests for malformed manifests, bad lane schemas, duplicate IDs, closed/deferred
  lane gates, evidence path/claim checks, invalid JSON files, missing files, and
  structured exception wrappers. These keep release/readiness claims
  evidence-backed without launching any simulations.
- The seventeenth post-audit coverage tranche added input-compatibility tests
  for top-level case-insensitive lookup, empty/scalar namelist value handling,
  legacy Boozer equilibrium-file aliases, `wout_path`/`equilibriumFile`
  override rendering, radial-coordinate conversion errors, `psiAHat`
  precedence, and explicit gradient-coordinate overrides. These protect the
  CLI/Python input contract and Fortran-v3 namelist parity without launching
  solves.
- The eighteenth post-audit coverage tranche added namelist-parser tests for
  Fortran booleans, integer/`D`-exponent scalars, quoted comment markers,
  vector values, one- and two-dimensional indexed assignments, missing-group
  lookup, nested-group rejection, unterminated-group rejection, and indexed
  multi-value rejection. These protect the parser used by every example and
  CLI run without touching production solve budgets.
- The nineteenth post-audit coverage tranche added direct RHSMode=1
  PAS/composite and preconditioner-dispatch tests for composition order,
  environment-gated theta-zeta admission, zeta-dominated line selection,
  x-block fallback selection, hybrid fallback behavior, Schwarz progress
  messages, ADI sweep composition, named family routing, and default block
  dimension forwarding. These protect automatic solver-policy behavior in the
  extracted one-level solver modules without launching production solves.
- The twentieth post-audit coverage tranche added output/export policy tests
  for native-grid export, nearest-point export, invalid zeta/x/xi export
  guards, nontrivial distribution-map contractions, no-equilibrium localization,
  and Boozer-alias equilibrium localization/namelist patching. These are fast
  Fortran-v3 output-contract tests for `outputs/writer.py` that keep generated
  data out of the repository.
- The twenty-first post-audit coverage tranche added transport-loop support
  tests for inactive full-space matvec cache fallback, disabled recycle-basis
  no-op behavior, invalid/negative recycle environment values, zero-RHS
  relative-residual reporting, dense mixed-precision dtype selection, dense
  batch state/metadata storage, and streaming-diagnostics collector guards.
  These protect production RHSMode=2/3 loop bookkeeping without running a full
  transport solve.
- The twenty-second post-audit coverage tranche added profile-system helper
  tests for Fortran-style scalar option lookup, `pointAtX0` index conventions,
  `V3FullSystemOperator` PyTree roundtrips, public size properties, supported
  constraint-scheme sizes, and unsupported constraint-scheme rejection. These
  protect matrix-free operator metadata and JAX-transform compatibility without
  constructing production matrices.
- The twenty-third post-audit coverage tranche added Phi1 output-policy tests
  for sparse-direct rejection, incremental fallback, dense-auto admission,
  dense-auto skip messages, sparse-direct environment overrides, and frozen
  nonlinear-history trimming. These protect CLI/output behavior for nonlinear
  Phi1 runs without launching Newton solves.
- The twenty-fourth post-audit examples tranche added
  `examples/tutorials/00_start_here.ipynb`, linked it from the examples
  navigation README files, and added notebook contract tests for pedagogy,
  required physics topics, output-free cells, and checked first-path assets.
  This improves the user-facing learning path without adding generated outputs
  or changing production examples.
- The twenty-fifth post-audit coverage tranche added output-geometry metric
  tests for Boozer `.bc` and VMEC `wout` `gpsiHatpsiHat` assembly. The tests use
  tiny analytic surfaces with monkeypatched readers, so they exercise the same
  Fortran-v3 metric formulas, mode filtering, and radial interpolation branches
  used in production HDF5 output without adding large equilibrium fixtures or
  slow solves to CI.
- The twenty-sixth post-audit documentation tranche tightened release-facing
  docs wording in the validation matrix, performance page, paper-figure page,
  and release checklist. The front-page README already has no matches for the
  review-blocked progress phrases; remaining chronological/status wording is
  kept in release notes, roadmaps, research-lane pages, or implementation-detail
  docs where chronology and claim boundaries are the point.
- The twenty-seventh post-audit coverage tranche added enabled RHSMode=2/3 FP
  Schur-coarse tests for the Fourier-line and x-block angular LU transport
  preconditioners. The tests monkeypatch the true operator to a deterministic
  identity action, then verify bounded coarse-space construction, restriction
  mode selection, kinetic residual-error columns, cache behavior, and finite
  application without launching full transport solves.
- The twenty-eighth post-audit examples tranche added
  `examples/tutorials/04_geometry_validation_and_performance.ipynb` and linked it
  from both examples README files. The notebook gives a no-output, classroom-safe
  guide to analytic/Boozer/VMEC geometry choices, `wout_path`, validation against
  frozen SFINCS Fortran v3 evidence, solver metadata checks, CPU/GPU performance
  scripts, and parallelism entry points. The examples contract now requires this
  notebook and checks that it stays pedagogic, topic-complete, and output-free.
- The twenty-ninth post-audit coverage tranche added sparse-pattern helper tests
  for Fortran-style active index conventions, derivative-matrix support
  extraction, reduced x/ell support policies, and summary metadata
  serialization. These protect the native reduced-Pmat and sparse-pattern
  infrastructure without materializing production operators or adding slow
  solves to CI.
- The thirtieth post-audit consolidation tranche removed the last internal
  source import from `sfincs_jax.v3_driver`: the HDF5 writer now imports
  RHSMode-1 solve helpers directly from `sfincs_jax.problems.profile_solve`.
  `tests/test_source_tree_consolidation.py` now parses package imports and
  fails if implementation modules reintroduce a `v3_driver` dependency, keeping
  `v3_driver.py` as an external compatibility shim rather than an internal
  owner.
- The thirty-first post-audit documentation tranche updated user-facing
  numerics, physics-reference, API, performance-technique, and QI research-lane
  pages to point at the canonical profile/transport solve, preconditioner, and
  device-sparse owners instead of stale `v3_driver.py` or former `rhs1_*` module
  names. Historical source-map and roadmap references remain only where they
  document compatibility history. A follow-up validation-matrix wording pass now
  describes the remaining `95%` coverage work as owner-module coverage for
  profile solves, transport solves, operator assembly, and output writing rather
  than further `v3_driver.py` decomposition.
- The thirty-second post-audit coverage tranche added explicit-sparse settings
  tests for factor-kind aliases, monolithic guard parsing, LU/ILU size caps,
  bounded integer/float environment parsing, symbolic nested-dissection/BLR
  policy knobs, permutation fallbacks, and ILU option bounds. This protects the
  production sparse-factor memory/runtime admission layer with deterministic
  unit tests instead of full production solves.
- The thirty-third post-audit coverage tranche added profile-system algebra
  tests for nonlinear temperature/Phi1 L-coupling helpers and primitive
  theta/zeta/x padding utilities. These cover Fortran-style kinetic coupling
  and sharded-padding invariants directly without building full input-deck
  operators.
- The thirty-fourth post-audit coverage tranche added RHSMode=2/3
  block-preconditioner tests for local active-index block assembly, inactive
  pitch masking, extra source/constraint block inversion, submatrix chunk
  plumbing, and cache reuse. The test injects deterministic local submatrices,
  so it protects the transport preconditioner owner without materializing a
  production operator or adding slow transport solves to CI.
- The thirty-fifth post-audit examples/documentation tranche simplified the
  top-level examples landing page by removing long release-audit command
  runbooks from first-pass navigation, replacing them with concise pointers to
  parity, performance, publication-figure, and scaled-suite tooling. It also
  locked all five tutorial notebooks in the examples contract and removed
  branch-status phrasing from the Fortran-example status page so the docs read
  as standalone software documentation rather than a development log.
- The thirty-sixth post-audit coverage tranche added RHSMode=1 output-safety
  tests for PAS-projected active-size accounting, solver metadata integer/float
  parsing, bounded JSON compaction, and metadata-copy semantics. These tests
  protect production HDF5/NetCDF/NPZ diagnostic fields and nonconverged-output
  sidecar traces without running a profile solve.
- The thirty-seventh post-audit review-readiness tranche made the full
  ``sfincs_jax`` source tree pass Ruff by replacing assigned lambdas in the
  ambipolar scan interpolator, removing a duplicate comparison-tolerance key,
  documenting the intentional JAX x64 import-order exceptions, and renaming
  ambiguous Legendre-index locals in collision kernels. Focused physics,
  comparison, ambipolar, output, examples, and transport-preconditioner tests
  pass after the cleanup.
- The thirty-eighth post-audit coverage/runtime tranche added explicit-sparse
  metadata and admission tests for symbolic permutation serialization,
  operator/factor bundle dtype behavior, and non-finite factor rejection. It
  also fixed the host Jacobi sparse preconditioner to preserve floating matrix
  dtype instead of silently promoting float32 operators to float64, which keeps
  this fallback aligned with the memory-reduction goal while preserving the
  strict residual-admission path.
- The thirty-ninth post-audit coverage tranche added profile-system branch
  tests for nonlinear residual Phi1-state substitution, precompile kernel
  routing, and Jacobian shape guards. These tests protect differentiable
  nonlinear solve behavior and ahead-of-time compile policy with fake
  lower/compile hooks, so CI gains coverage without spending time compiling
  production operators.
- The CI coverage floor is `80%`. The next planned gate is `85%`, once the
  branch has a stable margin above `85%` and the sharded CI wall time remains
  below ten minutes.

### Target Package Shape

The source package should have a small set of root modules for public entry
points and a single level of domain folders. No new folder should be nested
inside another folder under `sfincs_jax/` unless a short justification is added
to `sfincs_jax/README.md` and the import surface is tested.

Root modules to keep:

- `api.py`: stable Python API.
- `cli.py` and `__main__.py`: executable entry points.
- `solver.py`: high-level solve orchestration and public solver result types.
- `ambipolar.py`: public ambipolar convenience API.
- `sensitivity.py`: public JVP/VJP/implicit sensitivity API.
- `plotting.py`, `compare.py`, `io.py`, `namelist.py`, `input_compat.py`, and
  `paths.py`: user-facing I/O, input compatibility, comparison, plotting, and
  path utilities.
- `diagnostics.py`, `grids.py`, and `profiling.py`: stable scientific/support
  APIs used directly by examples, docs, tests, and benchmark tooling.
- `__init__.py`: exports only stable public contracts and compatibility aliases.

Domain folders to keep:

- `discretization/`: grids, stencils, active indexing, coordinate maps.
- `geometry/`: analytic, VMEC, Boozer, and JAX geometry adapters.
- `operators/`: drift-kinetic operator terms and assembled/matrix-free actions.
- `problems/`: RHSMode-specific problem owners, ambipolar roots, transport
  matrix solves, and profile-response solves.
- `solvers/`: Krylov dispatch, linear-solve policy, sparse/native factors, and
  preconditioner implementations.
- `outputs/`: HDF5/NetCDF/NPZ writer, output schemas, and post-solve diagnostics.
- `physics/`: collision, classical, bootstrap, and normalization formulas.
- `validation/`: frozen references, parity summaries, the release-data
  manifest/fetcher, and figure-generation helpers.
- `workflows/`: optional research workflows that call the public API and are
  not required by the CLI.

Folders to remove or absorb:

- `benchmarks/`, `compat/`, `input/`, and `parallel/` empty package stubs have
  been removed; use `validation/`, `namelist.py`/`input_compat.py`, and the
  flat `problems/transport_parallel_runtime.py` / `transport_parallel_worker.py`
  owners for transport parallel code.
- Remove empty `solvers/preconditioners/coarse_space`. Complete.
- Flatten `operators/profile_response/*` into `operators/profile_*.py` files.
  This is complete; `operators/profile_response.py` is a compatibility shim.
- Flatten `problems/profile_response/*` and
  `problems/profile_response/sparse/*` into `problems/profile_*.py` files.
  This is complete; `problems/profile_response.py` is a compatibility shim.
- Flatten `problems/transport_matrix/*` and its `parallel/` subfolder into
  `problems/transport_*.py` files. This is complete; `transport_matrix.py` is
  a compatibility shim.
- Flatten `solvers/preconditioners/*` into one-level solver files named
  `solvers/preconditioner_*.py`. Complete; `solvers/preconditioners.py`
  preserves former nested import paths as a compatibility index.

Compatibility policy:

- Public old imports keep working through one release cycle using explicit
  `sys.modules` alias registration and direct import-contract tests.
- Internal imports must move to the new one-level domain modules in the same
  commit that introduces the alias.
- The docs/API pages should point to the new canonical modules. A separate
  compatibility appendix can list old paths.

### Source README Deliverable

`sfincs_jax/README.md` is present and guarded by source-tree tests. It must
continue to explain:

- what each root module is for;
- what each retained domain folder owns;
- which APIs are stable for users and which modules are internal;
- how compatibility aliases work during this refactor;
- where to look for CLI usage, Python solves, geometry loading, outputs,
  solver preconditioners, autodiff, validation references, and examples;
- the rule that package depth is normally one folder below `sfincs_jax/`;
- where large reference data live and why they are not stored in the clone.

### Finite Refactor Tranches

Tranche 0: lock the inventory and import map.

- Status: complete. `tests/fixtures/source_tree_expected.json` records allowed
  root modules, allowed domain folders, target root modules, and compatibility
  debt.
- Keep a generated-but-checked text summary under `docs/source_map.rst` or a
  compact JSON under `tests/fixtures/source_tree_expected.json` that records
  allowed root modules, allowed domain folders, and allowed compatibility
  aliases.
- Add a test that fails on new nested folders, new root modules, or empty
  `__init__.py`-only packages unless the allow-list is updated deliberately.
- Acceptance: no source moves yet, but the future target is enforced.

Tranche 1: README and docs cleanup.

- Status: mostly complete. `sfincs_jax/README.md` exists, the root README
  explicit progress-language patterns are cleared, and strict docs builds pass.
- Keep the root README opening self-contained and free of branch
  history. Move detailed production-gate caveats to docs pages.
- Replace user-facing "now", "previous", "new version", and "current main"
  language outside `docs/release_notes.rst` and `docs/development_roadmap.rst`.
- Acceptance: strict docs build passes; root README has no progress-language
  matches except links to release notes.

Tranche 2: remove empty packages and root shims.

- Delete or absorb `benchmarks`, `compat`, `input`, `parallel`, and empty
  preconditioner subpackages.
- Move root helpers only when the move deletes a compatibility surface rather
  than adding another shim. `diagnostics.py`, `grids.py`, `input_compat.py`, and
  `profiling.py` are stable root support APIs for this PR; keep them documented
  and covered by source-tree tests.
- Keep root shim aliases only where documented as stable public API. The only
  remaining root compatibility debt is `v3_driver.py`.
- Keep the small one-file compatibility indexes
  `operators/profile_response.py`, `problems/profile_response.py`,
  `problems/transport_matrix.py`, and `solvers/preconditioners.py` through this
  release cycle. A package-initializer alias experiment removed those files but
  introduced circular imports during import-contract tests, especially through
  solver preconditioner imports. Removing these shims is a later cleanup after
  legacy imports are either dropped or implemented with a real lazy import hook.
- Acceptance: source-tree test passes; import-contract tests verify old public
  imports; package root has at most the public modules listed above.

Tranche 3: flatten operators and problems.

- Move `operators/profile_response/*` into one-level `operators/profile_*.py`
  modules. Complete.
- Move `problems/profile_response/*` and
  `problems/profile_response/sparse/*` into one-level `problems/profile_*.py`
  modules. Complete.
- Move `problems/transport_matrix/*` into one-level `problems/transport_*.py`
  modules. Complete.
- Rename only when the new name is more domain-descriptive; avoid broad
  `rhs1_*` names except inside compatibility aliases.
- Acceptance: `v3_driver.py` stays below `75` lines; API docs import canonical
  modules; focused RHSMode 1/2/3 parity and policy tests pass.

Tranche 4: flatten preconditioners without losing domain ownership.

- Move `solvers/preconditioners/pas/*`, `xblock/*`, `qi/*`,
  `symbolic_sparse/*`, `full_fp/*`, and `schur/*` into one-level
  `solvers/preconditioner_*.py` modules. Complete.
- Preserve solver-policy discoverability with docstrings and a single
  `solvers/preconditioners.py` compatibility index. Complete.
- Acceptance: policy docstring tests pass; CPU/GPU solver-selection fixtures
  still select the same defaults; no new smoother-tuning-only code is added.

Tranche 5: examples redesign.

- Status: first learning layer complete. `examples/README.md`,
  `docs/examples.rst`, `examples/tutorials/README.md`, three tutorial
  notebooks, and `examples/tutorials/run_quick_output_and_plot.py` are
  committed and tested. Example README tests also reject stale progress
  language and broken Python-script references in documented task paths, and
  the fast tutorial output/plot script runs in the bounded examples smoke suite.
- Keep the current topic folders (`getting_started`, `transport`, `autodiff`,
  `vmec_jax_finite_beta`, `optimization`, `performance`, and `parity`) because
  they are stable user-task names. Do not rename them into numbered folders
  unless the same commit deletes real duplication and updates docs/tests.
- Each major user task should have one clear first script and, where plots or
  derivations matter, one notebook. The tutorial layer points users to those
  canonical scripts instead of duplicating heavy workflows.
- Move raw upstream SFINCS decks and benchmark-output JSON out of first-pass
  learning paths only when they are unused by tests/docs. Keep small input decks
  needed by examples.
- Remaining work: add notebook text references for any new public capability
  added later.
- Acceptance: example index tells users which file to run for each application;
  smoke execution of one script per major folder stays below the CI example
  budget.

Tranche 6: coverage ramp to 95%.

- Status: exact package coverage is `85.003%` after the latest local audit.
  The next increment should target large user-risk modules rather than helper
  edges: `problems/profile_solve.py`, `problems/transport_solve.py`,
  `operators/profile_system.py`, `outputs/writer.py`,
  `solvers/preconditioner_transport_matrix.py`, `solvers/explicit_sparse.py`,
  `operators/profile_sparse_pattern.py`,
  `solvers/preconditioner_schur_profile.py`, and
  `operators/profile_true_operator_rescue.py`.
- Add fast, literature-anchored tests before adding broad coverage-only tests:
  Onsager symmetry and positivity for transport matrices, pitch-angle collision
  conservation/nullspace checks, finite-difference stencil exactness on
  low-order polynomials, Simakov-Helander high-collisionality trend checks,
  Redl/bootstrap-current normalization fixtures, ambipolar root replay for
  options 1/2/3, adjoint dot-product/JVP/VJP consistency, and CPU/GPU numerical
  equivalence on bounded fixtures.
- Use frozen SFINCS Fortran v3 HDF5/JSON references in CI instead of running
  Fortran. Keep small references in `tests/fixtures`; fetch larger references
  from GitHub releases through `sfincs_jax.validation.data_fetch`.
- Add production-shape fast tests that exercise large-resolution sizing,
  policy admission, memory estimates, output schemas, and residual gates without
  solving the full production system.
- Raise coverage in gates: `85%`, `90%`, then `95%`. Do not raise a gate until
  Linux CI remains below `10 min` and the exact coverage has margin above that
  floor.
- Acceptance: coverage floor reaches `95%`, no multi-megabyte fixture bloat,
  and CI stays below the time budget.

Tranche 7: benchmark and validation regeneration.

- After source moves and coverage gates pass, rerun all fast examples and the
  production benchmark manifest.
- Rerun CPU and office-GPU runtime/memory gates and regenerate README/docs
  plots, parity tables, bootstrap-current QA/QH comparisons, and runtime/memory
  comparisons with SFINCS Fortran v3.
- If a production run is still not promoted, describe it once in docs and keep
  the README focused on supported behavior.
- Acceptance: generated figures/tables come from the current branch, use
  same-resolution CPU/GPU/Fortran comparisons where claimed, and the release
  checker passes.

### Completion Gates

The consolidation pass is complete only when all of the following are true:

- package depth is one level below `sfincs_jax/` except explicitly allowed
  compatibility aliases;
- root modules are only the stable public entry points listed above;
- `sfincs_jax/README.md`, root README, examples README, docs API pages, and
  testing docs all describe the same canonical structure;
- examples are organized by user task and include both scripts and notebooks
  for the major workflows;
- CI is under `10 min` and enforces the current coverage floor;
- coverage reaches `95%` through meaningful unit, numerical, physics, parity,
  and regression tests;
- benchmark/runtime/memory/parity figures are regenerated after the final
  source move;
- the PR contains one coherent refactor story and no generated cache/output
  clutter.

## Evidence Reviewed

### SFINCS Fortran v3 Source And Documentation

Local reference: `/Users/rogeriojorge/local/sfincs/fortran/version3`

Manual and paper references:

- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/equations.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/runs.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/inputParameters.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/outputParameters.tex`
- `/Users/rogeriojorge/local/sfincs/doc/sfincsPaper/sfincsPaper.tex`
- Public repository: <https://github.com/landreman/sfincs>

Reference decks and probe summaries for the ambipolar functionality live in
`benchmarks/fortran_v3_ambipolar_reference/`. These decks pin the source-code
behavior of `ambipolarSolveOption=1`, `2`, and `3` before the sfincs_jax
implementation begins.

Implementation progress on 2026-06-23:

- `sfincs_jax.problems.ambipolar` now owns the first-class ambipolar problem,
  iteration, and result contracts.
- The Fortran-compatible Brent root path is implemented and tested against the
  checked-in small and production Fortran v3 ambipolar summaries.
- The Fortran v3 ambipolar reference matrix now covers geometry-1 helical and
  geometry-4 W7-X-like decks at small and production tiers for options 1, 2,
  and 3, with PETSc/KSP/MUMPS profiling markers and `/usr/bin/time -lp` RSS
  summaries.
- The source-code validator for Fortran-compatible ambipolar restrictions is
  implemented for the reference decks and derivative-assisted option guards.
- The Brent owner can now evaluate real RHSMode 1 radial currents through
  in-process `write_sfincs_jax_output_h5` calls, records per-evaluation
  artifacts, and is exposed through `sfincs_jax ambipolar`.
- Each real ambipolar evaluation now records solver-trace provenance
  (selected path, residual, target, setup/solve/elapsed time, active size),
  uses an ambipolar-local geometry/output cache across `E_r` evaluations, and
  carries a shape-checked Krylov state file for warm starts/recycled basis
  reuse between nearby electric-field solves.
- RHSMode 1 active field-split symbolic orderings now use a package-level
  semantic fixed-shape key, so structured CSR and true-operator rescue paths
  can reuse active/full index maps across same-shape solves without reusing
  stale `E_r`-dependent numerical matrices or factors.
- A finite-difference `dJr/dEr` helper now provides the numerical derivative
  gate that implicit/adjoint derivatives must match.
- `sfincs_jax.sensitivity` now owns a reusable implicit linear-observable
  derivative certificate. It solves both the tangent equation and the adjoint
  equation, reports primal/tangent/adjoint residuals, and can compare the
  derivative against centered finite differences.
- The same module now exposes a fixed-shape `LinearObservableSystem` builder
  bridge. Concrete RHSMode 1/4/5 owners can provide the true operator, RHS,
  observable, and their scalar-parameter derivatives without coupling the
  ambipolar root solvers back to `v3_driver.py`.
- A bounded `probe_linear_observable_vector` helper can recover `c` and `J0`
  from an existing linear diagnostic `J(x) = c^T x + J0` on small validation
  decks. This is a validation bridge for pinning radial-current weights before
  replacing it with analytic production weights.
- `sfincs_jax.problems.transport_diagnostics` now exposes
  `radial_current_vm_psi_hat_from_state` and a chunked observable-vector helper
  for the magnetic-drift radial-current contribution, plus explicit
  `psiHat`/`rHat`/`rN` coordinate wrappers. A tiny RHSMode-1 deck checks the
  recovered vector against the existing diagnostic.
- `sfincs_jax.problems.ambipolar` now includes a size-limited dense RHSMode-1
  linear-observable builder that assembles the true small-deck operator/RHS,
  finite-differences their scalar-parameter derivatives, and feeds the result
  to the tangent/adjoint certificate. This pins the wiring contract before a
  production sparse/matrix-free implementation.
- `sfincs_jax.sensitivity` now also exposes a matrix-free
  `MatrixFreeLinearObservableSystem` certificate. Production owners can pass
  operator actions, transpose actions, derivative actions, and selected
  solve/transpose-solve closures without dense assembly. Ambipolar adapters
  route this certificate to the same `dJr/dEr` contract used by option-1/3
  Newton solvers.
- `sfincs_jax.problems.ambipolar` now exposes the first concrete RHSMode-1
  matrix-free radial-current builder. It uses the real full-system matrix-free
  operator action, caller-supplied transpose and solve closures, finite-
  difference operator/RHS derivative actions, and existing radial-current
  observable weights; this removes dense matrix assembly from the builder
  itself while keeping a finite-difference promotion gate.
- The same builder now accepts caller-supplied derivative actions or JAX
  operator tangents. `operator_tangent_from_centered_difference` constructs
  valid pytrees with `float0` tangents for integer/bool leaves, and a real
  electric-field `xDot` operator test verifies the JVP action against centered
  differences.
- `matrix_free_radial_current_derivative_provider` now bridges matrix-free
  implicit certificates into safeguarded Newton/bisection and pure Newton
  ambipolar root solvers. Fast option-1/3-style tests verify root convergence,
  derivative metadata, tangent/adjoint consistency, and finite-difference
  agreement through the public solver API.
- No-Phi1 existing-branch `Er` operator tangents now use the v3 radial
  conversion analytically. The helper updates stored `dphi_hat_dpsi_hat` leaves
  in the full operator and f-block suboperators, and a real electric-field
  `xDot` fixture verifies the JVP action against centered operator differences.
- `keep_zero_er_terms` now lets derivative gates retain zero-valued ExB and
  `Er` suboperators at `Er=0` without changing normal solve defaults. A real
  `xDot` fixture verifies that the opt-in zero-`Er` operator is numerically
  identical to the default operator, while its analytic JVP tangent matches
  nearby nonzero centered differences.
- `rhsmode1_radial_current_response_from_namelist` now provides the first
  namelist-backed RHSMode-1 radial-current response and derivative provider.
  A bounded real-deck gate uses fixed-shape zero-`Er` branches, the analytic
  JVP operator tangent, and dense validation closures to compare the implicit
  derivative against centered finite differences without exposing users to
  manual plus/minus operator assembly.
- The namelist-backed RHSMode-1 response now validates on Fortran-style active
  pitch-mode DOFs instead of the rectangular inactive-mode storage. It defaults
  to the Fortran ambipolar `particleFlux_vm_rN` current convention, infers the
  radial conversion from the namelist, and replays the checked
  `geometry1_helical_small_option1` Fortran option-1 current and Newton slope
  within `2e-5` relative tolerance.
- The same active provider also replays the checked small option-3 physical
  currents for `geometry1_helical_small_option3` and
  `geometry4_w7x_like_small_option3` within `2e-5` relative tolerance.
- `solve_rhsmode1_ambipolar_from_namelist` now wires that active provider into
  the real option-1/2/3 ambipolar root policies. Bounded small-deck helical
  option-1 and option-3 roots replay the checked Fortran v3 roots using the
  active `particleFlux_vm_rN` response rather than a synthetic table.
- `sfincs_jax.sensitivity` now exposes `jvp_flux`, `vjp_flux`, and
  `adjoint_dot_product_check`. The focused tests apply the dot-product identity
  to real RHSMode-1 particle-flux, heat-flux, flow, radial-current, and
  bootstrap-current diagnostics, which is the same consistency gate required
  before promoting RHSMode 4/5 adjoint outputs.
- `sfincs_jax.sensitivity` also owns the first RHSMode-4/5 Fortran source
  contract helpers. `validate_fortran_v3_adjoint_sensitivity_constraints`
  mirrors the `validateInput.F90` adjoint restrictions, and
  `fortran_v3_adjoint_sensitivity_output_fields` pins the HDF5 sensitivity
  fields written by `writeHDF5Output.F90`, including the source-code
  `dParallelFlowdLambda` gate.
- `benchmarks/fortran_v3_sensitivity_reference` now contains compact numerical
  RHSMode-4/5 Fortran summaries for tiny W7-X-like analytic radial-current,
  heat-flux, parallel-flow, bootstrap, and debug finite-difference sensitivity
  decks. Checked tests pin the HDF5 field names, tensor ranks, wall/RSS budgets,
  `dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`,
  `dTotalHeatFluxdLambda = sum_s dHeatFlux_s/dLambda`,
  `dBootstrapdLambda = sum_s Z_s dParallelFlow_s/dLambda`, the RHSMode-5
  `dPhidPsidLambda` constant-current output, and selected debug-adjoint
  finite-difference outputs without committing generated HDF5 files.
- Derivative-assisted safeguarded Newton/bisection and strict pure-Newton root
  solvers are implemented behind the same ambipolar owner. They accept a
  derivative provider, so finite-difference gates, direct implicit
  certificates, and builder-backed implicit certificates can be wired in
  without changing the root-solve contract.
- `docs/feature_matrix.rst` now records the audited Fortran-v3 feature owners,
  the matching `sfincs_jax` implementation owners, and the promotion gates for
  ambipolar option 1/3, RHSMode 4/5 sensitivities, solver backends, geometry,
  Phi1, outputs, and parallelism.
- Remaining Lane 3 work is deeper fixed-shape numerical operator/factor and
  preconditioner setup reuse behind that evaluator, plus larger-deck physical
  replay gates that run the namelist-backed derivative provider through real
  ambipolar root solves.

Important Fortran v3 implementation modules:

| Module | Functionality to mirror or compare |
| --- | --- |
| `sfincs_main.F90`, `sfincs.F90` | Program initialization, MPI/PETSc lifecycle, execution mode selection. |
| `readInput.F90`, `validateInput.F90` | Namelist schema, defaults, compatibility guards, automatic coercions. |
| `createGrids.F90`, `xGrid.F90`, `uniformDiffMatrices.F90`, `polynomialDiffMatrices.F90` | Velocity and angle grids, interpolation, differentiation matrices, monoenergetic overrides. |
| `geometry.F90`, `updateBoozerGeometry.F90`, `radialCoordinates.F90` | Geometry schemes 1, 2, 3, 4, 5, 11, 12, 13 and radial-coordinate conversions. |
| `populateMatrix.F90`, `evaluateResidual.F90`, `evaluateJacobian.F90`, `preallocateMatrix.F90`, `sparsify.F90` | Exact residual/Jacobian/Pmat construction, sparse preallocation, simplified preconditioner matrix. |
| `solver.F90` | SNES/KSP orchestration, direct/iterative choices, MUMPS/SuperLU_DIST/PETSc factor controls, adjoint solves. |
| `ambipolarSolver.F90` | In-solver ambipolar root solve using Brent, safeguarded Newton/bisection, or pure Newton. |
| `populateAdjointRHS.F90`, `populatedMatrixdLambda.F90`, `populatedRHSdLambda.F90`, `adjointDiagnostics.F90` | RHSMode 4/5 adjoint sensitivity system, `dL/dlambda f - dS/dlambda`, `dRadialCurrentdEr`. |
| `diagnostics.F90`, `writeHDF5Output.F90`, `export_f.F90` | Fluxes, flows, bootstrap current, Phi1 diagnostics, KSP/SNES status, HDF5 outputs, exported distribution functions. |
| `classicalTransport.F90` | Classical transport diagnostics. |
| `testingAdjointDiagnostics.F90` | Finite-difference checks for adjoint sensitivity implementation. |

Key Fortran v3 algorithm facts to preserve:

- The most general system solves for
  `{f_s1(theta,zeta,x,xi), Phi1(theta,zeta), S_s1, S_s2, lambda}`. Without
  self-consistent `Phi1`, the unknowns reduce to `{f_s1, S_s1, S_s2}`.
- RHSMode 1 solves one physical right-hand side. RHSMode 2 solves three
  right-hand sides for the energy-integrated transport matrix. RHSMode 3 solves
  two monoenergetic right-hand sides with its special `Nx=1` grid convention.
- RHSMode 4 computes adjoint sensitivities at fixed `E_r`. RHSMode 5 computes
  adjoint sensitivities at ambipolar `E_r`.
- `ambipolarSolveOption=1` uses safeguarded Newton/bisection,
  `ambipolarSolveOption=2` uses Brent, and `ambipolarSolveOption=3` uses pure
  Newton.
- Newton-based ambipolar solve options compute `dRadialCurrentdEr` with an
  adjoint solve. The derivative is based on
  `evaluateAdjointInnerProductFactor`, which forms
  `dL/dEr * f - dS/dEr`, followed by the free-energy inner product with the
  adjoint radial-current solution.
- Fortran v3 has a source/manual discrepancy for derivative-assisted
  ambipolar solves: the manual says options 1 and 3 require
  `magneticDriftScheme > 0`, but `validateInput.F90` rejects tangential
  magnetic drifts for `RHSMode>3` and for `ambipolarSolve=.true.` with
  `ambipolarSolveOption != 2`. The implementation plan follows the source:
  options 1 and 3 are validated first with `magneticDriftScheme == 0`, then
  any later tangential-drift extension must be an explicit new sfincs_jax
  capability with its own tests.
- The linear/nonlinear solver stack is PETSc SNES/KSP. Linear cases use one
  SNES step. `includePhi1=true` with self-consistent `Phi1` uses Newton line
  search and nested KSP solves.
- Iterative solves use GMRES with a factorized `PCLU` preconditioner. Direct
  solves use `KSPPREONLY + PCLU`.
- When parallel direct solvers are available, the factorization backend is
  MUMPS or SuperLU_DIST. Serial PETSc direct solves use a sparse ordering,
  typically RCM in v3, with diagonal-pivot safeguards.
- MUMPS robustness is improved by pivot threshold controls and by retrying with
  larger `ICNTL(14)` workspace when factorization fails. This is a solver
  policy and residual-admission pattern that `sfincs_jax` should mimic
  natively, not by depending on PETSc.
- For RHSMode 2/3, Fortran v3 builds the operator once, reuses the factorized
  operator/preconditioner across the right-hand sides, and only changes the
  drive terms.
- For adjoints, Fortran v3 reuses the transpose operator path through either a
  separately built adjoint matrix or `KSPSolveTranspose`.

### Ambipolar Probe Results From Fortran v3

Completed local probes on 2026-06-22 and 2026-06-23 used
`/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs` and the small decks
now checked into `benchmarks/fortran_v3_ambipolar_reference/namelists`. The
new reproducible summaries are
`small_profile_summary_2026-06-23.json` and
`production_profile_summary_2026-06-23.json`; raw PETSc/MUMPS logs stay in the
scratch paths recorded in those JSON files.

Observed facts to feed directly into implementation:

- `geometry4_w7x_like_small_option1`: safeguarded Newton/bisection completed
  with Er evaluations `[-20, 20, 0]`, radial currents about
  `[-1.09e-6, 1.43e-6, 2.51e-8]`, internal Fortran ambipolar time
  `0.143 s`, wall time `75.31 s`, and peak RSS about `135 MB`.
- `geometry4_w7x_like_small_option2`: Brent completed with the same bracket and
  initial point, internal Fortran ambipolar time `0.147 s`, wall time
  `75.66 s`, and peak RSS about `135 MB`.
- `geometry4_w7x_like_small_option3`: pure Newton completed from the initial
  guess, internal Fortran ambipolar time `0.063 s`, wall time `75.19 s`, and
  peak RSS about `120 MB`.
- `geometry1_helical_small_option2`: Brent completed on a distinct analytic
  helical geometry with Er values `[-20, 20, 0, -1.7273, -2.0106]` and final
  radial current about `1.7e-9`; internal Fortran ambipolar time was `0.373 s`,
  wall time was `75.54 s`, peak RSS was about `156 MB`, and the run reported an
  MPI finalization error after writing useful diagnostics.
- `geometry1_helical_small_option1` and `geometry1_helical_small_option3` now
  pin the same analytic helical problem for derivative-assisted Newton paths.
  Option 1 used four forward solves and four adjoint solves; option 3 used two
  forward solves and two adjoint solves. Both reached `|J_r| ~ 1.1e-9`.
- All small probes used MUMPS through PETSc. The logs show repeated full
  `whichMatrix=0` and `whichMatrix=1` setup per Er evaluation, followed by
  residual matrix assembly and diagnostics.
- For these tiny systems, Jacobian sizes alternated between about `19,566` and
  `26,750` nonzeros depending on the Er branch/active terms, while the
  preconditioner had about `15,436`, `19,356`, or `26,750` nonzeros depending
  on deck and preconditioner settings.
- `geometry4_w7x_like_production_option{1,2,3}` completed the larger reference
  deck with `Ntheta=13`, `Nzeta=19`, `Nxi=48`, and `Nx=5`. All three options
  converged to the same root near `Er=-3.57735`. Option 1 used six forward and
  six adjoint solves, option 2 used six forward solves, and option 3 used four
  forward and four adjoint solves. The profiled runs reported internal
  ambipolar times of about `18.7 s`, `13.8 s`, and `12.1 s`, respectively,
  and peak RSS around `1.35-1.39 GB`.
- `geometry1_helical_production_option1` and
  `geometry1_helical_production_option3` converged to `Er=-3.26189` with final
  `|J_r| ~ 6.5e-11`. The same production grid is substantially harder than the
  W7-X-like deck: peak RSS was `5.7-5.8 GB`, option 1 used ten forward plus ten
  adjoint solves, and option 3 used eight forward plus eight adjoint solves.
- `geometry1_helical_production_option2` reached `|J_r| ~ 8.1e-12` but did not
  print Fortran's Brent success marker before exhausting its 12 evaluation
  budget. This is an important policy gate: sfincs_jax should report both the
  best residual found and whether the declared convergence criterion was met,
  rather than treating a small final residual and a success marker as the same
  concept.
- The production decks used MUMPS and assembled matrices with about
  `1.57e6` to `2.28e6` Jacobian nonzeros and about `1.27e6` to `1.54e6`
  preconditioner nonzeros for W7-X-like geometry. The helical production decks
  used the same maximum Jacobian size but an exact preconditioner with up to
  `2.28e6` nonzeros, explaining their higher RSS.
- The large gap between Fortran's internal ambipolar time and process wall time
  comes from process/logging/finalization overhead and verbose MUMPS/PETSc
  diagnostics. sfincs_jax should not emulate this shell-style scan cost; it
  should keep fixed-shape geometry/operator/factor metadata alive across Er
  evaluations.

### Current sfincs_jax State

Current source size snapshot after the 2026-06-26 final consolidation audit,
completed owner moves, and private-root deletion pass:

- Whole package: 154 Python files after the completed profile-response
  solve-sequencer/handoff compression, output-writer move, transport/output
  payback, solver/preconditioner family compression, Batch A gate repair, Batch
  B transport linear-system consolidation, Batch C transport-parallel runtime
  consolidation, Batch D solver-core consolidation, workflow/validation
  consolidation, and mapped-x-grid consolidation. The historical
  symbolic-sparse `rhs1_*` filename and all top-level `rhs1_*` /
  `transport_*` implementation files have been removed, QI has durable owner
  modules, and the preconditioner file-count gate is met. Package source lines
  are 165,532. This is above the original line-count checkpoint but is
  justified by replacing many implementation shards with durable owner modules
  while preserving production behavior.
- Package root: 17 Python files. No top-level `rhs1_*` or `transport_*`
  implementation files remain.
- `sfincs_jax/v3_driver.py`: 47-line compatibility shim. It must not regain
  implementation logic.
- Historical roots deleted and routed to owners:
  `v3_results.py`, `v3_sparse_pattern.py`, `v3_fblock.py`, `v3_system.py`,
  `v3.py`, `constrained_pas_branch.py`, `constraint_projection.py`,
  `host_refinement.py`, `pas_smoother.py`, `phi1_newton_linear.py`, and
  `phi1_newton_policy.py`.
- `sfincs_jax/problems/profile_*.py`: 17 flat implementation files plus the
  one-file `problems/profile_response.py` compatibility shim, about 52.9k
  lines. The largest files are `profile_sparse_xblock.py` 7,689 lines,
  `profile_policies.py` 7,369 lines, `profile_solve.py` 5,420 lines,
  `profile_sparse_handoff.py` 5,500 lines, `profile_sparse_qi.py` 4,873
  lines, `profile_sparse_direct.py` 3,567 lines, `profile_dense.py` 3,287
  lines, and
  `preconditioner_build.py` 2,683 lines. The `solve.py <=5,500` and
  `handoff.py <=5,500` review gates are restored.
- `sfincs_jax/problems/transport_*.py`: 8 flat implementation files plus the
  one-file `transport_matrix.py` compatibility shim. The file-count gate is
  met. `postsolve_diagnostics.py` was merged into `transport_finalize.py`,
  `streaming_outputs.py` was merged into `outputs/transport.py`, active dense
  setup, active factors, direct reduced-``Pmat``, direct block-Schur setup, and
  Fortran-reduced LU setup were consolidated into `transport_linear_system.py`,
  and internal parallel policy/sharding helpers were consolidated into
  `transport_parallel_runtime.py`.
- `sfincs_jax/solvers`: 11 root files. Explicit sparse factor policy/building
  now lives in `explicit_sparse.py`; shared preconditioner state/setup/operator
  shaping now lives in `preconditioning.py`; progress, Krylov state,
  solver-trace records, and compact Fortran/JAX solver-profile comparisons now
  live in `diagnostics.py`.
- `sfincs_jax/solvers/preconditioner_*.py`: 28 flat files, about 38k lines.
  QI now has four durable owner files:
  `preconditioner_qi_basis.py`, `preconditioner_qi_corrections.py`,
  `preconditioner_qi_device.py`, and `preconditioner_qi_policy.py`.
  PAS, x-block, full-FP, Schur, symbolic-sparse, domain-decomposition, and
  transport-matrix preconditioners use the same flat naming scheme.
- `sfincs_jax/operators/profile_*.py`: 19 flattened profile-response operator
  files, about 20.6k lines, plus a one-file `profile_response.py`
  compatibility shim. `profile_full_system.py` is the largest owner. This
  domain remains large but no longer contributes a nested package.
- `sfincs_jax/io.py`: 64-line compatibility facade. The concrete writer now
  lives in `sfincs_jax/outputs/writer.py` at 4,268 lines and is exported from
  `sfincs_jax.outputs`. `outputs/transport.py` now owns both streaming
  transport-output accumulation and streaming HDF5 writes.

The remaining consolidation pass must not add more one-off helper files or turn
the current 154-file tree into a larger but more fragmented tree. The final
action is a bounded retained-boundary audit plus review locking: only delete or
merge code if the same commit removes a repeated internal section or multiple
files and keeps names clearer. Otherwise, record the retained boundary and stop
the refactor.

Useful existing assets:

- JAX-native residual/operator code already exists for large portions of the
  v3 model.
- `sfincs_jax/solvers/implicit.py` already wraps `jax.lax.custom_linear_solve`
  for implicit differentiation through linear solves.
- `sfincs_jax/problems/profile_phi1_newton.py` already uses
  `jax.linearize` to build Newton-Krylov JVPs.
- `sfincs_jax.problems.ambipolar` provides bounded/reference Fortran-v3-style
  ambipolar option 1/2/3 root solvers, derivative-provider hooks, setup-reuse
  metadata, and checked Fortran replay gates.
- Several profile-response and transport-matrix preconditioners exist, but the
  implementation surface is still too fragmented and too environment-variable
  driven.
- Docs and examples now include autodiff and optimization demonstrations, but
  they should be recast around a small public API and explicit validation
  contracts.

Important current gaps:

- RHSMode 4/5 bounded/reference output contracts are implemented and tested
  against compact Fortran-v3 fixtures. Production-grid parity remains an
  external release-refresh benchmark.
- Ambipolar option 1/2/3 bounded/reference solvers are implemented and tested;
  production-grid replay artifacts remain outside normal CI.
- The largest active refactor gap is now review locking after the driver move:
  `profile_solve.py` is a stable problem owner, `v3_driver.py` is a
  tiny compatibility shim, public examples/scripts no longer need private
  driver imports, and remaining historical names should be treated as
  compatibility aliases or documentation/source-map entries rather than active
  implementation modules.
- No complete derivative API for `dGamma/dEr`, `dQ/dEr`, `d<J.B>/dEr`,
  `dJr/dEr`, profile sensitivities, and geometry harmonic sensitivities across
  all supported solve lanes.
- Too many solver choices are exposed as low-level environment variables rather
  than automatic, tested policy decisions.
- Code ownership is close to review-ready but still needs root/public-surface
  classification and internal line-paydown gates. The target remains fewer
  domain packages with larger, coherent ownership boundaries, not a growing set
  of one-off files.

### External References For Differentiation And Workflows

Sources reviewed:

- Fast automated adjoints for spectral PDE solvers:
  <https://arxiv.org/abs/2506.14792>
- JAX `custom_linear_solve` documentation:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- JAX custom JVP/VJP documentation:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- Lineax documentation:
  <https://docs.kidger.site/lineax/>
- JAXopt implicit differentiation:
  <https://jaxopt.github.io/stable/implicit_diff.html>
- T3D documentation:
  <https://t3d.readthedocs.io/>
- NEOPAX repository and local checkout:
  <https://github.com/uwplasma/NEOPAX>

Derivative design implications:

- The right default is not to backpropagate through every Krylov iteration or
  every adaptive solver branch. The robust design is an explicit residual graph
  plus implicit differentiation at the converged solution.
- The primal equation is
  `R(u, p) = 0`, with solution `u(p)`.
- Forward sensitivity should use
  `R_u du/dp = -R_p`.
- Reverse sensitivity for scalar objective `J(u, p)` should solve
  `R_u^T lambda = J_u^T`, then use
  `dJ/dp = J_p - lambda^T R_p`.
- This matches the Fortran v3 adjoint structure and the spectral-PDE adjoint
  paper's message: build adjoints from the symbolic residual/operator graph and
  reuse efficient sparse/direct/iterative solves.
- `jax.lax.custom_linear_solve` is the base primitive for differentiating
  linear solve calls without tracing all iterations.
- `custom_jvp` and `custom_vjp` should wrap only stable public solve APIs and
  only when the default JAX transform would trace control flow or host-only
  operations.
- Lineax can be evaluated as an optional operator abstraction or least-squares
  backend, but it should be adopted only if benchmarks show lower compile time,
  lower runtime, lower memory, or cleaner derivative code for a concrete lane.
- T3D and NEOPAX need a profile/flux closure interface, not a file-only
  executable. `sfincs_jax` should expose shape-stable JAX-friendly functions for
  radial batches, Er scans, transport coefficients, ambipolar roots, and
  Jacobian/vector-Jacobian products.

## Product Requirements

### Default User Experience

- CLI users provide `input.namelist` and a geometry path. The default solver
  policy selects a robust and efficient method automatically.
- The terminal output reports phase timings, selected solver path, residual
  target, residual history summary, memory estimate, output file paths, and
  enough progress to judge whether a run will take seconds, minutes, or longer.
- Advanced users can override solver family, preconditioner, differentiability
  mode, output format, and profiling options, but routine runs should not
  require environment variables.

### Python And Optimization Experience

- Python users can call a compact public API:
  `solve_flux_surface`, `solve_transport_matrix`, `solve_ambipolar`,
  `make_flux_closure`, and `make_objective`.
- Differentiable mode is explicit and documented. It returns pytrees with
  solution, diagnostics, residual certificates, and derivative capabilities.
- Non-differentiable mode can use faster host/direct paths when called from the
  CLI or from Python with `differentiable=False`.
- The API should support optimization workflows such as:
  bootstrap-current minimization, ambipolar electron-root targeting, particle
  and heat flux minimization, impurity flux targeting, sensitivity analysis,
  inverse design, uncertainty quantification, and profile evolution coupling.

### Research-Grade Claim Boundary

Public claims require all of the following:

- Numerical residual passes the true operator residual gate.
- Shared outputs match SFINCS Fortran v3 within documented tolerances.
- CPU and GPU either match within tolerance or the difference is documented and
  gated.
- Runtime and peak memory are measured from fresh cold and warm runs.
- The exact resolution, device, commit, Python/JAX versions, and solver path
  are recorded.
- For derivative claims, JVP/VJP/adjoint results pass finite-difference checks
  on stable windows and transpose/adjoint residual checks.
- For optimization examples, the plotted improvement must come from a
  reproducible script, not a hand-edited artifact.

## Final Architecture Target

The final codebase should be organized around a small number of domain owners,
not around historical implementation names. This section supersedes all older
Lane 1 refactor notes. `plan.md` is only an execution log; the consolidation
steps below are the single authoritative plan for PR #8.

Target package boundaries:

| Package | Ownership |
| --- | --- |
| `sfincs_jax.api` | Stable Python entry points, return objects, differentiable closures, and documented high-level helpers. |
| `sfincs_jax.cli` | CLI parsing, progress reporting, profiling switches, plotting dispatch, and non-autodiff fast-path controls. |
| `sfincs_jax.input` plus thin `namelist.py` / `input_compat.py` shims | Namelist/TOML parsing, defaults, validation, Fortran-compatible coercions, and fixture-localization helpers. |
| `sfincs_jax.geometry` / `sfincs_jax.discretization` | VMEC/Boozer/Miller/analytic geometry adapters, radial coordinates, grids, quadrature, interpolation, and finite-difference stencils. |
| `sfincs_jax.operators` | Drift-kinetic residuals, Jacobians, RHS/source builders, collision operators, Phi1 terms, and reusable operator layouts. |
| `sfincs_jax.problems` | Problem orchestration only: profile response, transport matrices, ambipolar roots, monoenergetic cases, and solve-result certificates. |
| `sfincs_jax.solvers` | Krylov/direct/native factors, solver policies, reusable preconditioners, progress/trace state, implicit linear solves, and non-autodiff fast paths. |
| `sfincs_jax.outputs` | HDF5/netCDF/NPZ formats, output-cache state, diagnostics schema, streaming transport output, and plotting payloads. |
| `sfincs_jax.sensitivity` | JVP/VJP, implicit differentiation, adjoint checks, finite-difference certificates, and optimization derivative utilities. |
| `sfincs_jax.workflows` | Radial scans, optimization objectives, T3D/NEOPAX closures, examples that compose public APIs, and publication-figure scripts. |
| `sfincs_jax.validation` | Parity reports, benchmark harnesses, physics gates, artifact policies, and release-quality comparison plots. |

Structural rules:

- Prefer 5-10 durable modules in a domain package over dozens of algorithm-name
  shards. A file should exist because it has a domain owner, not because one
  helper was extracted from a monolith.
- Keep files below about 1500 lines when practical. Exceptions are allowed only
  for dense mathematical kernels or compatibility shims with an owner comment
  that explains why the file is intentionally large.
- Do not introduce new root-level implementation modules. Root modules are
  public API, compatibility shims, or long-standing physics kernels that have a
  documented deletion/migration reason.
- Avoid names that encode history instead of ownership: no new `v3_*`,
  `rhs1_*`, `transport_*`, `*_handoff`, `*_promotion`, or campaign-specific
  implementation files unless the file is a temporary compatibility shim with a
  deletion condition.
- Each consolidation commit must reduce net files or remove a large internal
  section. Moving code into a new file without deleting, merging, or shrinking
  another owner does not count as progress.
- Every moved owner boundary gets at least one canonical import test and at
  least one behavior test through the public API or problem-level entry point.
- The PR must stay light: no generated HDF5/NPZ/profiler outputs, no temporary
  plots, no local absolute-path artifacts, and no broad new lint ignores.

## Lane 1 - Final Consolidation Pass

Goal: finish PR #8 with a smaller, clearer source tree without changing
physics, outputs, tolerances, benchmark claims, public behavior,
differentiable Python paths, non-autodiff CLI fast paths, or current CPU/GPU
parity gates. This is the only active refactor plan. Older tranche, sweep,
helper-extraction, and experiment-lane notes are historical and must not be
followed as parallel plans.

### Current Audit Baseline

Audit commands for every batch:

```text
find sfincs_jax -name '*.py' -not -path '*/__pycache__/*' | wc -l
find sfincs_jax -maxdepth 1 -name '*.py' | wc -l
find sfincs_jax -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 wc -l | sort -nr | head -40
find sfincs_jax -name '*.py' -type f | sed 's#/[^/]*$##' | sort | uniq -c | sort -nr | head -50
find sfincs_jax/solvers -maxdepth 1 -type f -name 'preconditioner_*.py' -print0 | xargs -0 wc -l | sort -nr
rg -n "sfincs_jax\.(v3_driver|problems\.transport_matrix\.postsolve_diagnostics|solvers\.preconditioners\.symbolic_sparse\.rhs1_fortran_reduced)" docs README.md examples tests sfincs_jax
```

Current source inventory from the 2026-06-26 consolidation audit:

| Area | Current state | Review-ready target |
| --- | --- | --- |
| Whole package | 143 Python files, about 165k package lines | Keep `<=143` files unless a new durable owner deletes at least two old files in the same commit. Review target is no package-file increase and no line-count increase unless the change deletes files and simplifies ownership. |
| Package root | 17 Python files | Root gate is met. Root files now remain only when they are public API, CLI, compatibility, plotting/profiling helpers, or stable user-facing facades. |
| `v3_driver.py` / `io.py` | 47-line and 64-line compatibility shims | Keep below 80 lines. Do not put implementation logic back into either file. Delete only after public docs, examples, scripts, and compatibility tests no longer need the shim. |
| `problems/profile_*.py` | 17 flat implementation files plus `profile_response.py` compatibility shim; largest owners are `profile_sparse_xblock.py`, `profile_policies.py`, `profile_sparse_handoff.py`, `profile_solve.py`, `profile_sparse_qi.py`, and `profile_sparse_direct.py`. | Do not add profile-response files. Reduce complexity only by deleting duplicate policy branches or moving historical names into existing owners without creating a new monolith. |
| `problems/transport_*.py` | 8 flat implementation files plus `transport_matrix.py` compatibility shim; direct reduced-`Pmat`, active factors, and block-Schur setup live in `transport_linear_system.py`; parallel policy/sharding lives in `transport_parallel_runtime.py`. | Keep `transport_parallel_worker.py` only as the subprocess entry point. Do not grow `transport_solve.py` into another monolith. |
| `solvers/preconditioner_*.py` | 28 flat files; QI, PAS, x-block, full-FP, Schur, symbolic-sparse, domain-decomposition, and transport-matrix owners are explicit. | Keep mathematical family names. Merge only if one commit deletes at least three files and keeps ownership clearer. |
| `workflows` / `validation` | 16 files total. `validation.benchmark_artifacts` was merged into `validation.artifacts`; `validation.fortran_profile` was merged into `validation.fortran`; six `optimization_*` workflow implementation files were merged into `workflows.optimization`; and two mapped-x-grid files were merged into `workflows.mapped_xgrid`. Old workflow module imports are package-level compatibility aliases. | Further consolidation only when import graphs show this can delete files without file-level shims and without obscuring public workflow concepts. |
| Docs/tests/examples/scripts | Public examples/scripts are being migrated off `sfincs_jax.v3_driver`; tests still intentionally cover compatibility imports. | Public-facing material uses `api`, `cli`, `outputs`, `validation`, `workflows`, or problem owners. Owner tests may keep private imports. |

Locked checkpoints:

- Top-level `rhs1_*` and `transport_*` implementation files are gone.
- Historical root implementation shims `v3_results.py`, `v3_sparse_pattern.py`,
  `v3_fblock.py`, `v3_system.py`, and `v3.py` are gone or routed to owners.
- `sfincs_jax.v3_driver` is only a compatibility shim and must not regain
  implementation logic.
- `profile_solve.py <=5,500`,
  `profile_sparse_handoff.py <=5,500`, `v3_driver.py <=80`, and
  `io.py <=800` are met. Keep them locked while later batches move owners.
- Package root is solved for this PR. Do not spend more time moving root files
  unless a public facade is already demonstrably dead.
- `sparse/handoff.py` is retained for now: it is large, but the audit found
  real sparse-PC stage orchestration there, not just re-exports. Deleting it
  now would move code into another monolith or create import-cycle shims.

### Non-Negotiable Consolidation Rules

- Do not create another plan. This file is the plan; `plan.md` is only the
  execution log.
- Do not add helper-only files. A refactor commit must delete files, merge
  files, remove a large internal section, or replace several historical names
  with one durable domain owner.
- Prefer fewer durable files over many thin files. File names must describe
  domain ownership, not extraction history.
- Avoid new historical or campaign names: no new `v3_*`, `rhs1_*`,
  `transport_*`, `*_handoff`, `*_promotion`, or experiment-specific
  implementation modules.
- A new file is allowed only when the same commit deletes at least two smaller
  files or replaces a worse historical owner name with a durable domain owner.
- Keep compatibility shims only when they protect users or tests. Each shim
  must have an owner and deletion condition in docs/source maps or tests.
- Preserve differentiability boundaries. JAX-native residual/operator/
  implicit-derivative paths remain available through Python APIs; host-only
  fast paths stay explicit policy choices for CLI/default solves.
- Do not restart broad solver research inside this consolidation pass. Solver
  performance work resumes only if a correctness regression blocks review
  readiness.
- Every consolidation batch gets import-contract coverage plus at least one
  behavior test through a public API or problem-level entry point before
  commit.
- Do not split a large owner into several new smaller files just to lower line
  count. A new owner is allowed only if the same commit deletes at least two
  old files and the new name is a durable domain name.
- Prefer registry/data-structure simplification inside existing owners for
  oversized files such as `profile_solve.py`, `policies.py`,
  `sparse/xblock.py`, and `sparse/handoff.py`.
- Each remaining code tranche must be large enough to matter: delete at least
  two files, remove a repeated internal section of at least about 300 lines, or
  finish public docs/import migration. Otherwise skip it and move to review
  locking.

### File Disposition Matrix

| Area | Keep | Merge or move | Delete or shim condition |
| --- | --- | --- | --- |
| Root public surface | `api.py`, `cli.py`, `namelist.py`, `input_compat.py`, `grids.py`, `sensitivity.py`, `plotting.py`, stable physics kernels | Workflow/data and geometry support roots have moved to domain owners. Continue moving only if public imports can migrate without adding shims. | Delete no-op root shims after public imports migrate. |
| Compatibility roots | `v3_driver.py` only as temporary import shim | Result tests/docs import problem-owned result contracts directly | Delete `v3_driver.py` only if all public imports and tests move to domain APIs. |
| Profile response | Existing owners only: `setup.py`, `solve.py`, `policies.py`, `preconditioner_build.py`, `dense.py`, `residual.py`, `diagnostics.py`, `solver_diagnostics.py`, `phi1_newton.py`, `sparse/` | Restore `solve.py <=5,500` without creating files. Keep the `handoff.py` waiver only if documented as a compatibility re-export facade. Reduce oversized owners by deleting duplicate branch patterns, not by creating shards. | Do not create another profile-response file. |
| Transport matrix | `transport_solve.py`, `transport_setup.py`, `transport_diagnostics.py`, `transport_finalize.py`, `transport_policies.py`, `transport_linear_system.py`, `transport_parallel_runtime.py`, and `transport_parallel_worker.py` | Batch B merged `active_dense.py`, `active_factor.py`, `direct_block_schur.py`, `direct_pmat.py`, and `fortran_reduced_lu.py` into `transport_linear_system.py`. Batch C merged internal parallel policy/sharding into `transport_parallel_runtime.py`. | Keep `transport_parallel_worker.py` only as the documented `python -m sfincs_jax.problems.transport_parallel_worker` subprocess entry point. Delete tiny relay files after tests import the owner. Do not grow `transport_solve.py` into another monolith. |
| Outputs | `outputs/formats.py`, `outputs/cache.py`, `outputs/rhsmode1.py`, `outputs/transport.py`, `outputs/writer.py` | Continue moving schema/output-policy pieces into output owners only if total package complexity drops. | `io.py` remains a `<=800` line compatibility facade until public imports no longer need it. |
| Solver core | `explicit_sparse.py`, `preconditioning.py`, `diagnostics.py`, `implicit.py`, `krylov_dispatch.py`, `path_policy.py`, `selection_policy.py`, `memory_model.py`, and native factor kernels | Batch D merged `explicit_sparse_factor_builder.py` and `explicit_sparse_factor_policy.py` into `explicit_sparse.py`; merged `preconditioner_caches.py`, `preconditioner_context.py`, `preconditioner_operators.py`, and `preconditioner_setup.py` into `preconditioning.py`; and merged `progress.py`, `state.py`, `trace.py`, and `profile_compare.py` into `diagnostics.py`. | Keep this owner set stable unless a future commit deletes a larger real boundary and passes solver-dispatch/import tests. |
| QI preconditioners | Durable owners are fixed: `qi/basis.py`, `qi/corrections.py`, `qi/device.py`, `qi/policy.py`, plus `qi/__init__.py` | No more QI file movement unless a correctness bug appears. Simplify internally by deleting dead `qi_*` compatibility symbols or duplicated basis/correction code. | Keep compatibility aliases only through `qi/__init__.py` or owner tests, not as files. |
| Symbolic sparse | `symbolic_sparse/active_factors.py`, `symbolic_sparse/host_factor.py`, `symbolic_sparse/policy.py`, `symbolic_sparse/profile_response.py` | Merge only if it removes a real boundary and keeps names clearer. | No symbolic-sparse implementation file may use an `rhs1_*` filename. |
| X-block/PAS/full-FP | Role-based family owners only | Merge policy/detail shards into family owners when they are not independent mathematical kernels. | No new smoother or experiment files. |
| Operators | Current owners are acceptable for this PR | Only merge very small term files if needed for file-count gates and if docs remain clearer. | Do not split `full_system.py` during this consolidation. |
| Workflows | `scans.py`, `postprocess_upstream.py`, and any workflow whose name is already a stable user concept | Replace `optimization_*` files with one durable optimization owner only if the commit deletes the old files and updates docs/tests/examples. Replace `mapped_xgrid_*` with one mapped-x-grid owner only if imports are internal or documented. | No new campaign/promotion filenames. If an old workflow name is public, keep a package-level alias rather than a file shim. |
| Validation | `figures.py`, `h5_parity.py`, `research_lanes.py`, `qi_device.py`, and small numerical/reference owners | Merge `benchmark_artifacts.py` into `artifacts.py` if API names remain stable. Merge `fortran_profile.py` into `fortran.py` if tests/docs stay clear. | Do not merge unrelated physics-gate files just to reduce count. Validation names should describe evidence type, not a campaign. |

### Final Consolidation Execution Plan

This is the single authoritative consolidation plan for PR #8. The completed
Batch A-G and Closure Phase records below are historical logs only; do not use
them as an active work queue. The active work must land as a few owner-level
passes that delete files, merge historical names into domain owners, or simplify
large repeated policy blocks without changing physics, outputs, default solver
behavior, differentiable Python paths, non-autodiff CLI fast paths, or current
CPU/GPU parity gates.

Active consolidation passes:

1. **Public-surface lock and compatibility boundary: complete, verify only.**
   Public examples, scripts, CLI, output paths, workflow examples, and docs must
   import public APIs, problem owners, output owners, workflow owners, or
   validation owners rather than `sfincs_jax.v3_driver`. Keep compatibility
   tests that import `v3_driver`; they prove the 47-line shim remains safe for
   existing users. Exit target: stale-import scans find no public example/script
   driver imports, `v3_driver.py` stays below 80 lines, and CLI/output/import
   tests pass.
2. **Workflow/validation package consolidation: complete, verify only.** The
   completed passes merged validation benchmark/profiling artifacts into
   `validation.artifacts` and `validation.fortran`, merged six optimization
   implementation files into `workflows.optimization`, and merged mapped-x-grid
   evidence/objective files into `workflows.mapped_xgrid`. Old workflow module
   imports are package-level compatibility aliases, not file shims. Further
   workflow or validation consolidation is forbidden unless one commit deletes
   at least two files, avoids package-line growth, and passes workflow,
   validation, docs, examples, and import-contract tests.
3. **Large-owner retained-boundary audit: one bounded pass, then stop.** The
   only large-owner candidates are
   `problems/profile_policies.py`,
   `problems/profile_sparse_xblock.py`,
   `problems/profile_sparse_handoff.py`, and
   `solvers/preconditioner_qi_device.py`. The audit found real domain
   ownership in all four: policy/env resolution, x-block stage setup, sparse-PC
   orchestration, and device-compatible QI preconditioning. Edit them only if a
   single patch removes a repeated internal section of roughly 300 lines or more
   without adding files or changing public behavior. Otherwise record the
   retained boundary and do not churn.
4. **Preconditioner-family compression gate: retained unless a major deletion
   appears.** PAS, x-block, full-FP, symbolic-sparse, Schur, QI, and
   transport-matrix preconditioners now use mathematical family names. Merge
   family files only if a single commit deletes at least three implementation
   files and the resulting owner is clearer than the current family layout. QI
   filenames are locked as `basis.py`, `corrections.py`, `device.py`, and
   `policy.py` unless a correctness bug requires movement.
5. **Review-lock pass: next executable step.** Update `docs/source_map.rst`,
   API docs, developer docs, release notes, and tests only where stale paths are
   still present. Run stale-import scans for deleted modules, the focused
   review suite, Sphinx with warnings as errors, `ruff` on touched files,
   `git diff --check`, repository-size hygiene, and lightweight CLI/API
   behavior gates. If these pass, stop refactoring and make PR #8 review-ready.
   Do not start another solver research, performance tuning, or file-movement
   lane inside this PR.

Stop conditions:

- If a proposed move needs a compatibility shim, a new helper-only file, or a
  broad import cycle workaround, do not do the move; record the reason here and
  proceed to the next pass.
- If a pass cannot delete files or remove a repeated internal section, skip it.
  Cosmetic line movement does not count as progress.
- If a correctness, parity, differentiability, CPU/GPU, or output-regression
  test fails, fix that before further consolidation.
- Once the review-lock pass is green, merge preparation replaces refactoring as
  the active task.

Historical records below explain previous decisions and validation evidence.
They are not an active execution plan.

### Historical Consolidation Closure Log

#### Historical Closure Phase 1 - Import Graph And Deletion Manifest

Purpose: prevent churn by proving which files are public, private, dead, or
compatibility-only before moving code.

Actions:

1. Generate an import graph for `sfincs_jax`, `tests`, `examples`, `scripts`,
   and docs snippets, excluding generated docs/static artifacts.
2. Classify each package-root module as one of: public API, stable physics
   kernel, internal implementation, workflow/validation utility, or temporary
   compatibility shim.
3. Produce a move/delete manifest in `plan_final.md` with one line per file:
   owner after consolidation, compatibility risk, tests that protect the move,
   and deletion condition.
4. Mark files that should not move in this PR because moving them would add
   shims or obscure mathematical ownership.

Exit gates:

- No code movement yet except removal of verified dead imports.
- `docs/source_map.rst` matches the manifest.
- `tests/test_domain_package_import_contracts.py` fails closed for any new root
  implementation module without a manifest entry.
- Focused import/API/docs gates pass.

Phase 1 root-module move/delete manifest:

| Root file | Target owner | Disposition or deletion condition |
| --- | --- | --- |
| `__init__.py` | package root public facade | keep at root |
| `__main__.py` | package root CLI entry point | keep at root |
| `api.py` | package root public API | keep at root |
| `cli.py` | package root CLI entry point | keep at root |
| `ambipolar.py` | problems.ambipolar via public API facade | keep root shim until public docs/examples migrate |
| `compare.py` | validation comparison API | move only after examples/scripts use validation owner |
| `input_compat.py` | input compatibility owner | keep root public compatibility shim until input package exports cover callers |
| `io.py` | outputs writer/formats/cache owners | keep tiny root facade until public imports migrate |
| `namelist.py` | input namelist owner | keep root public parser until input package exports are documented |
| `plotting.py` | outputs/plotting public helper | keep root public helper unless API replacement is documented |
| `sensitivity.py` | package root differentiation API | keep at root |
| `constrained_pas_branch.py` | `solvers/preconditioner_pas_policy.py` owner | move in solver-policy group if no public shim is needed |
| `constraint_projection.py` | solvers constraint-projection owner | move only after transport/profile imports use solver owner |
| `diagnostics.py` | physics/output diagnostics owner | defer until diagnostics API split is explicit |
| `grids.py` | discretization public grid owner | keep root public helper until discretization package exports are documented |
| `host_refinement.py` | solvers refinement policy owner | move in solver-policy group if profile-response imports migrate |
| `pas_smoother.py` | `solvers/preconditioner_pas_policy.py` owner | move in solver-preconditioner group |
| `paths.py` | package root path support utility | keep at root unless a support package is introduced with broad import rewrite |
| `phi1_newton_linear.py` | `problems.profile_phi1_newton` owner | move if it deletes root file without adding shim |
| `phi1_newton_policy.py` | `problems.profile_phi1_newton` owner | move if it deletes root file without adding shim |
| `profiling.py` | solvers/validation profiling support | defer until profiling API boundary is explicit |
| `solver.py` | solvers public contracts owner | keep root shim until solvers exports cover public contracts |
| `v3_driver.py` | compatibility shim to problem owners | keep below 80 lines until the compatibility deprecation window closes; public examples/scripts should not import it |

#### Historical Closure Phase 2 - Root-To-Domain Package Move

Purpose: reduce root-level clutter by moving coherent domains, not helpers.

Planned moves, subject to the Phase 1 manifest:

1. Convert geometry ownership into a package boundary. Move VMEC/Boozer/JAX
   geometry support from root-level files such as `vmec_wout.py`,
   `vmec_geometry.py`, `boozer_bc.py`, and `jax_geometry_adapters.py` behind a
   documented geometry owner while preserving public imports through
   `sfincs_jax.api` and `sfincs_jax.geometry`.
2. Move workflow-like root modules such as `scans.py`, `data_fetch.py`, and
   `postprocess_upstream.py` into `workflows` or `validation` only if no public
   shim is required. If they are public user surfaces, keep them at root and
   document that explicitly.
3. Move internal kinetic/discretization support from root into existing
   `operators`, `physics`, or `discretization` owners only when the commit
   deletes at least three root files and keeps public examples unchanged.
4. Delete root shims only after import-contract tests and docs confirm the
   public API no longer advertises them.

Exit gates:

- Package root drops to `<=36` files, or the manifest explains every retained
  root file.
- No new root implementation file is created.
- Public `api`, CLI, examples, geometry loading, output writing, plotting, and
  docs tests pass.

Status on 2026-06-26:

- First Phase 2 move complete. The former root workflow/data implementations
  `scans.py`, `postprocess_upstream.py`, and `data_fetch.py` moved to durable
  owners `workflows/scans.py`, `workflows/postprocess_upstream.py`, and
  `validation/data_fetch.py` with no root compatibility shims.
- Second Phase 2 move complete. The former root geometry implementations
  `geometry.py`, `boozer_bc.py`, `vmec_wout.py`, `vmec_geometry.py`, and
  `jax_geometry_adapters.py` moved to the `sfincs_jax.geometry` package with
  submodule owners `boozer.py`, `vmec_wout.py`, `vmec.py`, and
  `jax_adapters.py`. Public `sfincs_jax.geometry` imports remain available as a
  package API, not a root implementation module.
- Third Phase 2 move complete. The former root discretization kernels
  `adaptive_maps.py`, `indices.py`, `periodic_stencil.py`,
  `structured_velocity.py`, and `xgrid.py` moved to
  `sfincs_jax.discretization` submodule owners without root compatibility
  shims. Tests, examples, scripts, docs, and import-contract gates import the
  package owners directly.
- Fourth Phase 2 move complete. The former root profile-response operator
  kernels `collisionless.py`, `collisionless_er.py`, `collisionless_exb.py`,
  `magnetic_drifts.py`, and `residual.py` moved into flat
  `sfincs_jax.operators.profile_*` owners. A one-file
  `sfincs_jax.operators.profile_response` shim preserves old imports without a
  nested source folder. Operator parity, sparse derivative, residual-JVP, docs,
  examples, and import-contract gates import the canonical flat owners directly.
- Fifth Phase 2 move complete. The former root physics kernels
  `collisions.py` and `classical_transport.py` moved to `sfincs_jax.physics`
  owners with collision/classical physics gates importing the package owners
  directly.
- Sixth Phase 2 move complete. The private root solver helpers
  `constrained_pas_branch.py`, `constraint_projection.py`,
  `host_refinement.py`, `pas_smoother.py`, `phi1_newton_linear.py`, and
  `phi1_newton_policy.py` moved into existing owners
  `sfincs_jax.solvers.preconditioner_pas_policy`,
  `sfincs_jax.solvers.preconditioning`, `sfincs_jax.solvers.explicit_sparse`,
  and `sfincs_jax.problems.profile_phi1_newton`. No compatibility
  shims or helper-only files were added.
- Package-root Python files drop from 43 to 17 after the workflow/data,
  geometry, discretization-kernel, operator-kernel, physics-kernel, and
  private-root moves. Whole-package file count is now 162.
- The remaining root files are public API/CLI/input/output/diagnostic/support
  facades or compatibility shims. They should stay unless their public API
  replacement is documented and tests prove deletion is safe.

#### Historical Closure Phase 3 - Problem/Solver/Output Owner Compression

Purpose: remove the remaining internal implementation fragmentation without
creating another monolith.

Actions:

1. Audit overlap between `problems/profile_response/sparse/*` and
   flat `solvers/preconditioner_{xblock,qi,pas,symbolic,...}.py` modules. Move the
   ownership boundary only if it deletes duplicate policy/residual/candidate
   code and does not introduce import cycles.
2. Collapse profile-response sparse owners into fewer durable owner files only
   when the resulting names are domain names, not algorithm-history names. Good
   target owners are `active_system`, `kinetic_blocks`, `residual_corrections`,
   and `qi_device`; bad target names are new `rhs1_*`, `*_handoff`, or
   campaign-specific files.
3. Keep `solve.py` as orchestration, not as a dumping ground. Any reduction in
   `solve.py`, `policies.py`, `sparse/xblock.py`, or `sparse/handoff.py` must
   come from deleting duplicated branches or table-driving repeated policy
   payloads, not by extracting one helper at a time.
4. Compress `outputs/writer.py` internally by moving repeated output-field
   definitions into schema tables inside existing `outputs` files. Do not add
   output files unless at least two old files are deleted.

Exit gates:

- `profile_solve.py <=5,500` and
  `profile_sparse_handoff.py <=5,500` remain locked.
- `solvers/preconditioner_*.py <=28` remains locked unless a later change
  deletes more code than it adds; otherwise
  each retained family file has an owner note in the source map.
- No `rhs1_*`, `transport_*`, `v3_*`, `*_handoff`, or campaign-specific
  implementation filename is introduced.
- RHSMode 1, transport matrix RHSMode 2/3, sparse-PC, QI, PAS, x-block,
  Phi1, output-format, ambipolar, and sensitivity gates pass.

#### Historical Closure Phase 4 - Docs, Examples, Tests, And Review Lock

Purpose: make the PR reviewable and stop the refactor.

Actions:

1. Rewrite `docs/source_map.rst`, API docs, developer docs, README developer
   notes, and release notes so they describe the final domain owners and do not
   mention deleted implementation names except in historical release notes.
2. Keep only canonical user examples in the public examples tree. Benchmark
   regeneration, parity-table refresh, and publication-figure scripts should
   call `validation` or `workflows` owners instead of importing private solver
   internals.
3. Consolidate duplicated test scaffolds into owner tests while preserving
   physics gates, regression tests, numerical identities, autodiff checks, CLI
   tests, output-format tests, and representative benchmark validations.
4. Run the review-ready validation bundle locally, then check CI after the push
   without blocking on repeated polling.

Exit gates:

- Package counts meet the closure targets or documented compatibility
  exceptions are present in this file.
- Local validation passes: focused owner tests, import/API contracts, CLI,
  outputs, ambipolar, sensitivity, representative physics gates, Sphinx `-W`,
  scoped Ruff, `py_compile`, and `git diff --check`.
- Remote CI passes for PR #8.
- The PR body names the final module layout, known compatibility shims,
  validation run, and explicit deferred performance/research lanes.
- After these gates, stop refactoring and move the PR to review/merge. New
  performance research belongs in a separate plan or branch.

### Historical Completed Batch Record

The following Batch A-G entries document how the current branch reached the
closure baseline. They are not the next implementation queue.

#### Batch A - Gate Repair And Compatibility Freeze

Purpose: repair the currently reopened review gate and freeze the compatibility
surface before further movement.

Actions:

1. Restore `profile_solve.py <=5,500` without creating files. Prefer
   compacting compatibility imports/re-export plumbing or deleting duplicated
   branch payload code; do not move one helper into a new module.
2. Audit `profile_sparse_handoff.py` without its local
   `F401,F811` waiver. If Ruff reports only intentional dynamic re-export and
   driver-scope shadowing errors, keep the waiver but document it in the file,
   `docs/source_map.rst`, and import-contract tests with a deletion condition.
3. Verify `profile_sparse_handoff.py <=5,500`, `v3_driver.py <=80`,
   and `io.py <=800`.

Exit gates:

- `profile_solve.py <=5,500`.
- `profile_sparse_handoff.py <=5,500`.
- `v3_driver.py <=80` and remains implementation-free.
- `io.py <=800` and remains a compatibility facade.
- Focused profile-response sparse-PC, active projection, QI admission, direct
  tail, dense fallback, output-diagnostic, import-contract, py_compile, scoped
  Ruff, and `git diff --check` pass.

#### Batch B - Transport Linear-System Owner Consolidation

Purpose: delete several small transport numerical shards in one owner-level
move without making `transport_matrix/solve.py` a second monolith.

Actions:

1. Create or reuse one durable owner named by the domain, preferably
   `sfincs_jax/problems/transport_linear_system.py`.
2. Move the coherent active-system/factor/direct-Pmat functionality from
   `active_dense.py`, `active_factor.py`, `direct_block_schur.py`,
   `direct_pmat.py`, and `fortran_reduced_lu.py` into that owner.
3. Delete those five old files in the same commit and update imports from
   `profile_solve.py`, transport tests, docs, and source maps.
4. Keep tests named for behavior, not old filenames. Owner tests should check
   active dense setup, active factorization, direct block-Schur/direct-Pmat
   paths, and Fortran-reduced LU behavior through the new owner.

Exit gates:

- Net package file count decreases by at least four.
- Transport problem implementation remains in the flat `problems/transport_*.py`
  owner set plus the `transport_matrix.py` compatibility shim.
- No top-level package-root `transport_*` implementation file or old
  algorithm-history filename is introduced.
- Transport RHSMode 2/3, monoenergetic, sparse/direct fallback, active-system,
  streaming-output, and import-contract tests pass.

#### Batch C - Transport Parallel Runtime Consolidation

Purpose: keep parallelization simple for users and reviewers by reducing the
parallel package to runtime ownership unless a file is a real independent
kernel.

Actions:

1. Audit `parallel/policy.py`, `parallel/sharding.py`, and `parallel/worker.py`
   imports and tests.
2. Merge purely internal policy and sharding helpers into
   `parallel/runtime.py`; delete the old files in the same commit if imports
   are internal.
3. Keep `parallel/worker.py` because it is a documented subprocess executable
   path, not a tiny implementation shard.
4. If any other public import must remain, expose it from a documented API
   owner, not by keeping a tiny implementation file.

Exit gates:

- Transport parallel implementation has two flat files:
  `transport_parallel_runtime.py` and `transport_parallel_worker.py`.
- Parallel runtime, sharding, worker-payload, CPU/GPU admission, and import
  tests pass.

Status on 2026-06-26:

- Complete and flattened. `parallel/policy.py` and `parallel/sharding.py` were
  absorbed into `transport_parallel_runtime.py`;
  `transport_parallel_worker.py` is the public executable wrapper. Live docs,
  examples, tests, and source import the flat transport owners, with old nested
  imports covered only by compatibility tests.
- Current metrics after the transport flattening tranche: `149` package Python
  files, `17` package-root files, about `165,650` package source lines, and
  `problems/transport_*.py` at 8 implementation files plus the
  `transport_matrix.py` compatibility shim.
- Validation: scoped py_compile and Ruff passed; the focused transport-parallel
  suite passed with `101 passed`; the full `tests/test_transport_*.py` pattern
  passed with `273 passed`; Sphinx `-W` passed; `git diff --check` passed; and
  the live stale-reference audit found no references to deleted
  `parallel.policy` or `parallel.sharding` modules outside frozen static
  evidence artifacts.

#### Batch D - Solver Core And Preconditioner Surface Consolidation

Purpose: reduce solver-family fragmentation while preserving automatic solver
selection, differentiable JAX paths, and non-autodiff CLI fast paths.

Actions:

1. Merge `explicit_sparse_factor_builder.py` and
   `explicit_sparse_factor_policy.py` into `explicit_sparse.py` if imports are
   acyclic; otherwise document the exact cycle and keep them as the only
   explicit-sparse support files.
2. Merge `preconditioner_caches.py`, `preconditioner_context.py`,
   `preconditioner_operators.py`, and `preconditioner_setup.py` into one
   preconditioning-state owner if doing so deletes at least three files and
   keeps setup/application semantics clearer.
3. Merge `progress.py`, `state.py`, `trace.py`, and `profile_compare.py` into
   one diagnostics/progress owner if they are internal implementation support.
4. Keep QI frozen at its five durable owners. Only delete dead compatibility
   symbols or duplicate implementation blocks; do not create more QI files.

Exit gates:

- `sfincs_jax/solvers` root file count decreases or the plan documents the
  import-cycle blocker.
- `solvers/preconditioner_*.py <=28` remains true, with stretch `<=24` only if the
  names stay clearer.
- Solver-selection, explicit sparse, native factor, preconditioner setup,
  progress/trace, QI, PAS, x-block, Schur, symbolic-sparse, full-FP, and
  differentiable implicit-solve tests pass.

Status on 2026-06-26:

- Explicit-sparse consolidation is complete. `explicit_sparse_factor_policy.py`
  and `explicit_sparse_factor_builder.py` were absorbed into
  `sfincs_jax/solvers/explicit_sparse.py`; the old files were deleted and live
  source imports now target the consolidated owner.
- Current metrics after this substep: `174` package Python files, `43`
  package-root files, `165,929` package source lines, `17` solver-root files,
  and `35` preconditioner files. `profile_solve.py` remains `5,420`
  lines, `profile_sparse_handoff.py` remains exactly `5,500` lines,
  `v3_driver.py` remains `47` lines, and `io.py` remains `64` lines.
- Validation: scoped py_compile and Ruff passed; explicit-sparse, sparse-helper,
  and matrix-reduction tests passed with `29 passed`; preconditioner policy and
  context tests passed with `125 passed`; the broader sparse/profile-response
  gate passed with `618 passed`; import/API/docstring contracts passed with
  `20 passed`; Sphinx `-W` passed; and `git diff --check` passed.
- A sparse-PC branch regression was fixed during validation: requested
  sparse-PC solve setup now initializes the current residual vector even when
  optional factor preflight is disabled, and the auto-preflight retry stage now
  reuses the already-built structured layout instead of eagerly deriving a
  layout from lightweight mocked operators.
- Preconditioning-state consolidation is complete. `preconditioner_caches.py`,
  `preconditioner_context.py`, `preconditioner_operators.py`, and
  `preconditioner_setup.py` were absorbed into
  `sfincs_jax/solvers/preconditioning.py`; the old files were deleted and live
  imports now target the consolidated owner.
- Current metrics after this substep: `171` package Python files, `43`
  package-root files, `165,865` package source lines, `14` solver-root files,
  `35` preconditioner files, and `preconditioning.py` at `1,173` lines.
  `profile_solve.py` remains `5,420` lines,
  `profile_sparse_handoff.py` remains exactly `5,500` lines,
  `v3_driver.py` remains `47` lines, and `io.py` remains `64` lines.
- Additional validation: scoped py_compile and Ruff passed; preconditioning
  setup/cache/context/matrix-reduction/Fortran-reduced/driver-dispatch tests
  passed with `87 passed`; broader profile-response/preconditioner-family
  gates passed with `490 passed`; import/API/docstring contracts passed with
  `20 passed`; full `tests/test_transport_*.py` passed with `273 passed`;
  Sphinx `-W` passed; stale live-import audit found no references to deleted
  preconditioner-state modules; and `git diff --check` passed.
- Diagnostics/progress consolidation is complete. `progress.py`, `state.py`,
  `trace.py`, and `profile_compare.py` were absorbed into
  `sfincs_jax/solvers/diagnostics.py`; the old files were deleted and live
  source imports now target the consolidated owner. This owner intentionally
  groups solver-neutral observability: CLI/runtime progress text, Krylov
  fixed-shape state files, portable solver traces, and compact Fortran-v3/JAX
  solver-profile comparisons.
- Current metrics after completed Batch D: `168` package Python files, `43`
  package-root files, `165,862` package source lines, `11` solver-root files,
  `35` preconditioner files, and `diagnostics.py` at `609` lines.
  `profile_solve.py` remains `5,420` lines,
  `profile_sparse_handoff.py` remains exactly `5,500` lines,
  `v3_driver.py` remains `47` lines, and `io.py` remains `64` lines.
- Validation: scoped py_compile and Ruff passed for `diagnostics.py`, output
  writers/formats, ambipolar trace readers, transport/profile-response import
  callers, and comparison scripts; focused solver diagnostics/output tests
  passed with `50 passed`; import/API/docstring contracts passed with
  `20 passed`; full `tests/test_transport_*.py` passed with `273 passed`;
  preconditioning/dispatch gates passed with `87 passed`; Sphinx `-W` passed;
  the stale live-import audit found no references to deleted diagnostics
  modules; and `git diff --check` passed.

#### Historical Batch E - Root/Public Surface And Workflow Classification

Purpose: reduce root-package ambiguity without breaking documented user imports.

Actions:

1. Classify every root module as public API, stable physics kernel,
   compatibility shim, or internal workflow. Record the classification in
   `docs/source_map.rst`.
2. For `scans.py`, `postprocess_upstream.py`, and `data_fetch.py`, move only if
   the public callers can migrate to `api`, `cli`, `workflows`, or
   `validation` without adding root shims. If public compatibility is needed,
   keep the root file and document it.
3. Ensure examples and docs import `sfincs_jax.api`, `sfincs_jax.cli`,
   `sfincs_jax.outputs`, or documented workflow modules rather than private
   sparse/preconditioner internals.

Exit gates:

- Package root is `<=40` files, or remains `<=44` with every remaining root
  file explicitly classified.
- No public example imports private sparse/preconditioner internals.
- CLI, API, scans, upstream postprocess, data-fetch, output-format, docs, and
  import-contract tests pass.

Status on 2026-06-26:

- Complete. The package root remains at `43` Python files, which is within the
  allowed `<=44` gate, and every remaining root file is explicitly classified
  in `docs/source_map.rst` as a public API/entry point, stable kernel/support
  utility, public workflow/support surface, or compatibility facade/shim.
- `scans.py`, `postprocess_upstream.py`, and `data_fetch.py` remain at the
  package root because they are documented or script/example-facing public
  support workflows; moving them without shims would break public imports.
- `tests/test_domain_package_import_contracts.py` now fails closed if a new
  package-root module appears without a classification, or if the source map
  advertises deleted flat `rhs1_*`/`transport_*` files as live owners or
  legacy aliases.
- Validation: stale source-map alias scan returned no matches; focused
  CLI/API/scans/upstream/data-fetch/import-contract tests passed with
  `39 passed`; scoped py_compile and Ruff passed for touched workflow/root
  classification files; Sphinx `-W` passed; Batch E metrics were `168`
  package Python files, `43` package-root files, `165,862` package source
  lines, `profile_solve.py` at `5,420` lines,
  `profile_sparse_handoff.py` at `5,500` lines, `v3_driver.py` at
  `47` lines, and `io.py` at `64` lines.

#### Historical Batch F - Profile-Response Internal Line Paydown

Purpose: lower complexity in the largest remaining profile-response owners
without creating more files.

Progress:

- First Batch F substep complete on 2026-06-26. `policies.py` now uses grouped
  requested-control metadata tables and a table-driven QI-device progress
  formatter instead of two large repeated metadata dicts and a long repeated
  progress branch. The public helper names and keyword-call behavior are
  preserved. `policies.py` decreased from 7,425 to 7,369 lines, and package
  source lines decreased from 165,862 to 165,806. No files were added.
- Validation passed for this substep:
  `python -m pytest tests/test_rhs1_xblock_fallback_initial_guess.py -q
  --tb=short` with 36 passed,
  `python -m pytest tests/test_profile_response_diagnostics.py
  tests/test_domain_package_import_contracts.py
  tests/test_policy_module_docstrings.py -q --tb=short` with 30 passed,
  targeted QI sparse-pattern metadata tests with 3 passed,
  `python -m pytest tests/test_rhs1_qi_*.py -q --tb=short` with 123 passed,
  `python -m pytest tests/test_profile_response_sparse_pc.py -q --tb=short`
  with 329 passed, and `python -m pytest tests/test_rhs1_device_operator_unit.py
  tests/test_rhs1_xblock_fallback_initial_guess.py -q --tb=short` with
  41 passed. Scoped py_compile, Ruff, and `git diff --check` passed.
- Second Batch F substep complete on 2026-06-26. `sparse/xblock.py` now derives
  post-solve correction driver-state metadata from the dataclass fields instead
  of a manual mirror, and `sparse/qi.py` now uses grouped QI-device enrichment
  and multilevel metadata specs. `sparse/xblock.py` decreased from 7,725 to
  7,689 lines, `sparse/qi.py` decreased from 4,885 to 4,873 lines, and package
  source lines decreased from 165,806 to 165,758. No files were added.
- Validation passed for this substep:
  `python -m pytest tests/test_profile_response_sparse_pc.py
  tests/test_rhs1_qi_*.py -q --tb=short` with 452 passed,
  `python -m pytest tests/test_rhs1_device_operator_unit.py
  tests/test_rhs1_xblock_fallback_initial_guess.py
  tests/test_profile_response_diagnostics.py -q --tb=short` with 57 passed,
  targeted QI sparse-pattern metadata tests with 3 passed, plus scoped
  py_compile, Ruff, and `git diff --check`.

Actions:

1. In `policies.py`, replace duplicated environment/namelist/default-selection
   branches with small policy tables or dataclasses where behavior is identical.
2. In `sparse/xblock.py` and `sparse/qi.py`, delete dead compatibility paths
   and consolidate repeated residual/admission/candidate-selection payloads.
3. In `sparse/handoff.py`, keep only compatibility re-export and orchestration
   glue; move no new code into it unless another file is deleted in the same
   commit.

Exit gates:

- `profile_solve.py <=5,500` remains true.
- `profile_sparse_handoff.py <=5,500` remains true.
- No new profile-response files are created.
- Package source lines trend downward from 166,045, or any remaining increase
  is justified by deleted files and clearer owner boundaries.
- Focused RHSMode 1, QI, x-block, sparse-PC, Phi1, ambipolar, sensitivity, and
  output-diagnostic tests pass.

#### Historical Batch G - Docs, Tests, Source Map, And Review Gate

Purpose: stop moving names, make the PR reviewable, and prove that the refactor
did not change physics or user behavior.

Actions:

1. Refresh `docs/source_map.rst`, `docs/api.rst`, README developer notes,
   examples, benchmark docs, testing docs, and release notes to the final
   module layout.
2. Consolidate duplicate import-contract tests into owner tests while keeping
   physics, regression, numerical, output-format, CLI, autodiff, docs, and
   representative benchmark gates.
3. Run repository-size hygiene: no generated HDF5/NPZ/profiler outputs, no
   temporary plots, no local absolute-path artifacts, and no accidental large
   files.
4. Run the review-ready validation set.

Review-ready acceptance gates:

- Package source file count is `<=190`; stretch target `<=175` only if clarity
  improves.
- Package source lines are below 165,398, or the remaining increase is
  explicitly justified by deleted files plus clearer ownership.
- Package root has `<=40` files preferred, `<=44` allowed only with explicit
  public/shim labels.
- `v3_driver.py` is deleted or below 80 lines.
- `profile_solve.py <=5,500`.
- `profile_sparse_handoff.py <=5,500`.
- `io.py <=800` or deleted.
- `problems/profile_*.py` has 17 implementation files plus the
  `profile_response.py` compatibility shim.
- `problems/transport_*.py` has 8 implementation files plus the
  `transport_matrix.py` compatibility shim.
- `solvers/preconditioner_*.py` has `<=28` files.
- No top-level `rhs1_*` or `transport_*` implementation files exist.
- No broad package-level lint ignores exist; the only allowed exception is a
  documented, file-local compatibility re-export waiver in
  `profile_sparse_handoff.py`.
- Focused tests pass for profile response RHSMode 1, transport matrix RHSMode
  2/3, ambipolar options 1/2/3, RHSMode 4/5 sensitivity contracts, sparse-PC,
  QI admission, output formats, CLI, public API imports, docs, representative
  physics gates, and differentiable linear-solve contracts.
- Sphinx builds with `-W`, `git diff --check` passes, and no temporary outputs
  are tracked.

If these gates pass, PR #8 can move from draft to review-ready. Any remaining
production-runtime optimization, true device-QI promotion, lower-memory native
factor work, or production benchmark refresh belongs in the research/release
lanes below unless it blocks correctness.

Status on 2026-06-26:

- Batch G stale-owner repairs are complete. The remaining stale imports found
  by fail-fast validation were corrected to their current owners:
  `sfincs_jax.validation.fortran`, `sfincs_jax.solvers.diagnostics`,
  `sfincs_jax.problems.profile_policies`, and
  `sfincs_jax.problems.profile_preconditioner_build`.
- Deterministic benchmark-summary metadata now uses repository-relative source
  report paths from `sfincs_jax.validation.artifacts`, avoiding local absolute
  paths in regenerated release artifacts.
- The research-lane manifest now points at the consolidated QI, transport
  parallel, and sparse handoff owners:
  `solvers/preconditioner_qi_{basis,corrections,device}.py`,
  `problems/transport_parallel_runtime.py`, and
  `problems/profile_sparse_handoff.py`.
- Validation passed with the review-ready focused bundle:
  `191 passed in 32.86s`; targeted benchmark-summary and research-lane gates
  passed; scoped Ruff and py_compile passed; Sphinx `-W` passed; and
  `git diff --check` passed.
- Fail-fast full-suite validation was attempted twice. The first run reached
  `1504 passed` before exposing the stale research-lane manifest paths fixed
  above. The second run repeated already-clean sections and was stopped after
  `486 passed`; no new failure was observed before interruption.
- Current structural counts satisfy the Batch G hard gates:
  154 package Python files, 17 package-root Python files,
  165,532 package Python lines, `v3_driver.py` at 47 lines, `io.py` at
  64 lines, `profile_solve.py` at 5,420 lines,
  `profile_sparse_handoff.py` at 5,500 lines, and no top-level
  `rhs1_*` or `transport_*` implementation files.

## Lane 2 - Full Fortran v3 Functionality Matrix

Goal: make the Fortran v3 feature surface explicit and either implemented,
tested, or documented as intentionally unsupported.

Feature matrix:

| Feature | Current target |
| --- | --- |
| RHSMode 1 profile response | Keep parity, improve automatic solver selection, finish production-grid gates. |
| RHSMode 2 energy-integrated transport matrix | Keep parity, reuse operator/factors across RHS, production-floor CPU/GPU reports. |
| RHSMode 3 monoenergetic transport matrix | Keep parity, preserve special grid normalization, compare with DKES-style literature gates. |
| RHSMode 4 adjoint sensitivities at fixed Er | Implement first-class API and HDF5/netCDF outputs. |
| RHSMode 5 adjoint sensitivities at ambipolar Er | Implement after Lane 3 ambipolar and Lane 6 adjoint transpose solve are stable. |
| `ambipolarSolveOption=1` | Implement safeguarded Newton/bisection with `dJr/dEr` from implicit/adjoint solve. |
| `ambipolarSolveOption=2` | Implement Brent root solve using completed physical solves and scan reuse. |
| `ambipolarSolveOption=3` | Implement pure Newton with robust failure certificate and fallback suggestion. |
| `includePhi1` self-consistent | Keep nonlinear Newton-Krylov parity and output all SNES-like diagnostics. |
| `readExternalPhi1` | Add or audit file input parity and cheaper fixed-Phi1 solves. |
| `includePhi1InCollisionOperator` | Keep explicit caveat until parity and memory gates pass. |
| `geometryScheme=1/2/3/4` | Keep analytic geometry parity and physics gates. |
| `geometryScheme=5` | VMEC wout parity, `vmec_jax` differentiable adapter lane, finite-beta QA/QH gates. |
| `geometryScheme=11/12` | Boozer `.bc` parity and memory/runtime reports. |
| `geometryScheme=13` | Decide implementation scope for direct Boozer-spectrum optimization input. |
| `export_f` | Implement distribution-function export on requested grids with bounded output sizes. |
| Matrix/vector debug dumps | Provide NPZ/netCDF debug dumps for operators/RHS/solution/preconditioner metadata. |
| HDF5 output parity | Continue adding all Fortran v3 outputs plus sfincs_jax diagnostics. |
| Classical transport | Audit term-by-term with Fortran v3 and add physics tests. |
| Input validation | Mirror Fortran v3 compatibility errors and warnings, but phrase them for Python/CLI users. |

Acceptance gates:

- A generated feature matrix in docs marks each item as implemented, tested,
  intentionally unsupported, or deferred.
- Every implemented item has at least one unit test, one behavior/regression
  test, and one docs example or reference page.
- Every unsupported/deferred item has a precise physics or engineering reason,
  not a vague "not yet".

## Lane 3 - Ambipolar Solver And Er Derivatives

Goal: make ambipolar calculations first-class, fast, and differentiable when
requested.

Implementation steps:

1. Create `sfincs_jax.problems.ambipolar` as the canonical owner.
2. Define `AmbipolarProblem`, `AmbipolarResult`, and `AmbipolarIteration`
   pytrees with solver path, residual, root type, derivative, and output fields.
3. Implement an in-process fixed-shape ambipolar driver. It must reuse parsed
   input, geometry, grids, operator metadata, factor/preconditioner setup, and
   diagnostic allocation across Er evaluations whenever the shape is unchanged.
4. Reuse the existing scan postprocessor only as a compatibility backend for
   reading completed scan directories; do not use scan-style repeated process
   launches as the primary sfincs_jax algorithm.
5. Implement Fortran-compatible radial-current conventions:
   `J_r = sum_s Z_s Gamma_s`, with the correct flux variant for `Phi1`.
6. Implement Brent with bracket provenance, monotonic interpolation diagnostics,
   and no derivative requirement.
7. Implement safeguarded Newton/bisection using `dJr/dEr`.
8. Implement pure Newton with strict derivative and trust-region guards.
9. Mirror the Fortran v3 source-code validator first:
   derivative-assisted options 1 and 3 require no `Phi1`, no inductive
   electric field, FP collisions, constraint scheme `-1` or `1`, full or DKES
   trajectory compatibility, and no tangential magnetic drifts.
10. Add a separately gated future extension for tangential-drift
    derivative-assisted ambipolar solves if the physics/operator derivatives
    are implemented.
11. Add automatic policy:
   - use derivative-assisted safeguarded Newton when an adjoint/implicit
     derivative is available and bracket is valid;
   - use Brent for robust CLI default when derivative setup is expensive;
   - fail with partial scan artifact if no bracket is found.
12. Add radial-batch ambipolar solve for profile workflows.
13. Add solve-complete/finalization separation: if diagnostics are written but
    profiling or backend finalization fails, record a warning status rather
    than marking the physical solve as failed.
14. Add CLI:
    `sfincs_jax ambipolar input.namelist --er-min ... --er-max ...`.

Derivative formula:

Given `R(u, Er, p) = 0` and `Jr(u, Er, p)`, compute

```text
du/dEr = -R_u^{-1} R_Er
dJr/dEr = partial_Er Jr + Jr_u du/dEr
```

For reverse mode, solve

```text
R_u^T lambda = Jr_u^T
dJr/dEr = partial_Er Jr - lambda^T R_Er
```

Acceptance gates:

- Reproduce Fortran v3 `ambipolarSolveOption=1/2/3` on at least one tokamak,
  one W7-X-like analytic case, one VMEC QA case, and one QH case.
- First reference reproduction target is the checked-in small probe set:
  `geometry4_w7x_like_small_option{1,2,3}` and
  `geometry1_helical_small_option2`. The Brent option-2 sequence is now covered
  by `tests/test_ambipolar_problem.py`.
- The checked `geometry1_helical_small_option1` derivative-assisted Newton
  point is now covered by `tests/test_sensitivity.py` using the active
  Fortran-style operator, `particleFlux_vm_rN`, and the implicit
  tangent/adjoint derivative certificate. The same checked small deck is also
  covered by `tests/test_ambipolar_problem.py` through the real option-1
  safeguarded Newton root solve.
- The checked small option-3 current points for helical and W7-X-like analytic
  decks are now covered by `tests/test_sensitivity.py` using the same active
  namelist-backed provider, and the helical small deck is covered by the real
  pure-Newton option-3 root solve.
- Production reference target is the checked-in production decks under
  `benchmarks/fortran_v3_ambipolar_reference/namelists`, which must be run
  before public benchmark claims are regenerated. The checked-in production
  Brent summary is now covered by `tests/test_ambipolar_problem.py`.
- `dJr/dEr` matches finite differences on a stable step window and matches the
  checked small Fortran option-1 Newton slope.
- Brent and Newton return the same root within tolerance when both are valid.
- CPU/GPU roots and root types match within tolerance for bounded cases.
- Failed brackets write a useful partial artifact and do not claim success.
- The sfincs_jax in-process ambipolar driver reports per-evaluation solver
  trace provenance and currently reuses geometry/output setup through a scoped
  cache, shape-checked Krylov state through a private state file, and symbolic
  active field-split orderings through a fixed-shape key. The remaining
  implementation gate is to avoid repeated numerical operator and
  factor/preconditioner setup when Er updates do not change the problem shape.

## Lane 4 - Adjoint Sensitivities And Differentiable Solves

Goal: replace ad hoc gradient examples with a tested derivative system aligned
with Fortran RHSMode 4/5 and modern JAX implicit differentiation.

Public APIs:

- `linearize_solve(problem, params)`
- `jvp_flux(problem, tangent_params)`
- `vjp_flux(problem, cotangent_outputs)`
- `adjoint_sensitivity(problem, objective)`
- `differentiate_ambipolar_root(problem, objective)`
- `finite_difference_check(problem, parameter, step_window)`

Implementation steps:

1. Define residual graph objects for each problem: residual, matvec, transpose
   matvec, RHS, diagnostics, and parameter leaves.
2. Ensure each residual graph has shape-stable pytrees and no hidden global
   state.
3. Use `jax.lax.custom_linear_solve` for linear solve differentiation.
4. Use `jax.linearize` for Newton/Phi1 JVPs where the nonlinear residual is
   solved in JAX.
5. Use `custom_vjp` only around public solve functions that contain adaptive
   solver choices or host-only fast paths.
6. For CLI/non-differentiable paths, return derivative-unavailable metadata
   rather than silently tracing host operations.
7. Implement Fortran-v3-style adjoint RHS builders for particle flux, heat flux,
   parallel flow, total heat flux, radial current, and bootstrap current.
8. Implement `dL/dlambda f - dS/dlambda` for:
   - `Er`,
   - Boozer `B`,
   - contravariant/covariant Boozer components,
   - `iota`,
   - radial derivative/metric terms after the exact Fortran mapping is audited.
9. Audit the exact Fortran `whichLambda` mapping before public docs, because
   comments and case labels must be reconciled term-by-term.
10. Keep RHSMode 4 fixed-Er output-field contracts synchronized with
    Fortran-v3 fixture summaries.
11. Keep RHSMode 5 ambipolar-Er `dPhi/dPsi` contracts synchronized with
    Fortran-v3 fixture summaries.

Acceptance gates:

- Fortran-v3 RHSMode 4/5 input restrictions and sensitivity HDF5 field names
  are pinned against the source-code behavior.
- Small RHSMode-4 Fortran radial-current, heat-flux, parallel-flow, bootstrap,
  and debug finite-difference sensitivity summaries plus one RHSMode-5
  constant-current heat-flux summary are checked in and tested. Intermediate
  and production-grid parity are release-refresh benchmarks.
- `A^T lambda - J_u^T` adjoint residual passes for every derivative gate.
- JVP and VJP agree through dot-product tests:
  `<JVP(dp), y> = <dp, VJP(y)>`.
- Finite-difference checks pass on documented stable windows.
- Fortran v3 RHSMode 4/5 output contracts match on checked small grids.
- Derivative examples run under CI without full production solves.

## Lane 5 - Native Solver And Preconditioner Architecture

Goal: provide PETSc-like robustness natively in Python/JAX without depending on
PETSc, MUMPS, or SuperLU_DIST at runtime.

Fortran behavior to emulate:

- Exact operator and simplified Pmat are separate.
- Factorization is reused across RHS and adjoint solves when possible.
- Ordering and pivot safeguards are part of solver policy.
- Direct solve failure is diagnosed and retried with safer factor controls.
- Adjoints reuse transpose solves.

Implementation steps:

1. Standardize operator objects with:
   `matvec`, `transpose_matvec`, optional explicit sparse emission,
   block metadata, true residual, and parameter leaves.
2. Standardize preconditioner objects with:
   `apply`, optional `apply_transpose`, setup diagnostics, residual admission,
   memory estimate, and reuse key.
3. Keep two solver stacks:
   - differentiable stack: pure JAX matvecs, Krylov, custom linear solve,
     optional Lineax only if it improves a measured lane;
   - production stack: native Python/JAX sparse/direct/block factors and host
     rescue paths when `differentiable=False`.
4. Implement reusable symbolic ordering metadata for active-only operators.
5. Implement native sparse/block factor families:
   - line factors in `x`, `ell`, `theta`, and `zeta`;
   - block triangular/angular streaming factors;
   - active-only coupled kinetic block factors;
   - moment/source/constraint Schur complement;
   - additive Schwarz patches;
   - nested-dissection/multifrontal-inspired separator updates where beneficial.
6. Add setup-time true-residual admission for every approximate factor:
   no auto promotion without measuring residual reduction on the actual
   operator.
7. Add factor reuse across:
   - RHSMode 2/3 multiple RHS;
   - Er scans at fixed shape;
   - radial batches where geometry shape is fixed;
   - adjoint transpose solves.
8. Add fail-fast memory guards before expensive factor setup.
9. Add structured progress logging for setup and solve phases.
10. Keep lower-memory replacement as research/deferred if it cannot beat the
    existing robust path within the documented budget.

Acceptance gates:

- Auto policy selects the best passing method without environment variables.
- Strict true-residual gate passes before outputs are promoted as converged.
- Production-floor QA/QH RHSMode 1, geometry-rich RHSMode 2/3, and Phi1 cases
  either pass or are documented as deferred with exact residual/runtime/RSS.
- Runtime is no worse than 20x SFINCS Fortran v3 for public production
  comparison claims, unless a case is explicitly marked as research/deferred.
- Peak memory fits documented CPU/GPU budgets and records device memory where
  applicable.

## Lane 6 - T3D, NEOPAX, And Optimization Integration

Goal: make `sfincs_jax` useful as a transport closure in profile solvers and
stellarator optimization, not only as a standalone file-based code.

Public closures:

- `make_flux_surface_closure(geometry, species, resolution, mode)`
- `make_transport_matrix_closure(geometry, species, resolution)`
- `make_ambipolar_closure(geometry, species, resolution, er_policy)`
- `make_radial_profile_closure(geometry_provider, profile_provider, radii)`

Closure inputs:

- radius or radial grid,
- density and temperature profiles,
- density and temperature gradients,
- electric field or ambipolar policy,
- species charges/masses,
- geometry object or geometry file,
- solver/differentiability mode.

Closure outputs:

- particle fluxes,
- heat fluxes,
- parallel flows,
- bootstrap current,
- radial current,
- ambipolar roots and root type,
- transport matrices,
- derivatives/JVP/VJP when requested,
- solver certificates.

Optimization examples:

- QA nfp=2 bootstrap-current minimization.
- QA or QI electron-root targeting from ambipolarity.
- Heat and particle flux minimization with impurity-flux target.
- VMEC-JAX to Booz-Xform-JAX to SFINCS-JAX differentiable pipeline.
- T3D/NEOPAX-style radial closure using a fixed geometry and evolving profiles.

Acceptance gates:

- Examples run from a clean install without private local paths.
- Differentiable examples have finite-difference or adjoint validation.
- Non-differentiable production examples report solver path and residual.
- Docs explain when gradients are exact implicit derivatives, proxy gradients,
  or unavailable.

## Lane 7 - Validation, Physics Gates, And Coverage

Goal: reach high meaningful coverage through real physics, numerical, and
regression tests without making CI expensive.

Test tiers:

| Tier | Runtime budget | Purpose |
| --- | --- | --- |
| Unit | seconds | Grids, coordinates, geometry, operators, collision terms, output schema. |
| Numerical | seconds to 2 min | Residual identities, transpose tests, conservation, quadrature convergence. |
| Physics gates | 2 to 8 min | Literature and Fortran v3 anchored transport/ambipolar/bootstrap checks. |
| CI regression | 5 to 10 min total | Public API, CLI, small parity, derivative gates, docs build. |
| Nightly/optional | 30 to 120 min | Production-floor CPU, GPU, Fortran v3 parity, memory/runtime. |
| Release | manual or scheduled | Full public benchmark regeneration, figures, parity matrix, artifacts. |

Physics gates to keep or add:

- Axisymmetric tokamak banana/plateau/Pfirsch-Schlueter trend checks.
- Monoenergetic DKES-style transport coefficients for W7-X/LHD/HSX analytic
  cases.
- Simakov-Helander high-collisionality limit with a pinned high-nu scan.
- QA/QH bootstrap current comparison against SFINCS Fortran v3 and Redl formula
  at identical resolution, with convergence error bars.
- W7-X ambipolarity with checked equilibrium/profile provenance.
- Finite-beta QA profile-current lane once residual convergence is clean.
- Phi1 sanity checks: flux-surface averaged lowest-order radial particle flux
  cancellation and nonlinear residual convergence.
- Adjoint sensitivity checks against finite differences and Fortran RHSMode 4/5.
- CPU/GPU reproducibility checks for representative PAS, FP, Phi1, QA, QH, and
  QI cases.

Coverage strategy:

- Aim for 95 percent meaningful package coverage after refactor, not by adding
  slow full-solve tests.
- Prefer tests on extracted pure functions and operator blocks.
- Add synthetic manufactured operators for solver/preconditioner unit tests,
  but keep every manufactured test tied to an actual operator identity.
- Keep large equilibria out of git history; fetch release fixtures only in
  optional/nightly jobs.
- Coverage gaps in host-only profiling paths can be exempted only with explicit
  comments and separate integration tests.

## Lane 8 - Benchmarks, Figures, Docs, And Release Artifacts

Goal: regenerate public claims from reproducible scripts after implementation
lanes pass.

Benchmark matrix:

- Devices/geometries: tokamak, HSX, W7-X, LHD, QA, QH, QI nfp=1/2/3/4 when
  available.
- Geometry modes: analytic, VMEC, Boozer `.bc`, direct Boozer spectrum if
  implemented.
- Physics modes: PAS, full FP, magnetic drifts on/off, Er on/off, Phi1 on/off,
  RHSMode 1/2/3/4/5 where supported.
- Resolution tiers:
  - CI tier: small but physically meaningful;
  - public benchmark tier: production-floor resolution;
  - stress tier: collaborator/NTX-like large cases.
- Devices: CPU cold/warm, GPU cold/warm, Fortran v3 reference.

Metrics:

- wall time,
- compile/setup/solve/diagnostics/output phase times,
- peak RSS,
- device memory,
- residual norm and target,
- Krylov iterations and factor setup status,
- selected solver path,
- all shared physical outputs,
- derivative residuals and finite-difference errors when applicable.

Required public figures:

- Runtime and memory comparison plot: Fortran v3, sfincs_jax CPU cold/warm,
  sfincs_jax GPU cold/warm.
- QA/QH bootstrap current: sfincs_jax, Fortran v3, Redl formula, same
  resolution, convergence error bars.
- Ambipolar Er educational/validation plot.
- Autodiff sensitivity validation plot.
- Solver phase timing plot for at least one production-floor case.
- Optional transport profile closure plot for T3D/NEOPAX-style use.

Docs requirements:

- Landing page explains the drift-kinetic equation before the discretized
  system.
- Usage page shows one CLI command, one Python solve, one plot command, and one
  differentiable objective.
- Method pages explain grids, collisions, geometry, Phi1, solvers,
  preconditioners, ambipolar solve, and adjoints.
- API pages expose stable entry points only, not internal helper churn.
- Validation pages explain every physics gate and benchmark provenance.
- Research-lane page lists only honest deferred items with exact status.

## Lane 9 - Release And Branch Hygiene

Goal: ship from a clean, lightweight repository with one review PR.

Steps:

1. Keep the active refactor work in PR #8 until this plan is complete.
2. Do not open additional PRs for sub-lanes.
3. Commit and push coherent batches frequently.
4. Keep large outputs and generated benchmark data out of git unless they are
   compressed public figures under the artifact-size policy.
5. Before review-ready:
   - full local focused tests pass;
   - docs build passes;
   - CI passes;
   - release benchmark scripts complete or deferred statuses are documented;
   - README figures and docs figures are regenerated from current reports;
   - `git diff --check` passes;
   - no temporary outputs or local paths remain.
6. After review-ready and merge:
   - tag a new version;
   - publish release notes;
   - confirm PyPI workflow;
   - archive release benchmark artifacts.

## Milestones

### M0 - Final Plan And Feature Matrix

Deliverables:

- `plan_final.md` committed.
- Generated Fortran-v3 feature matrix in docs.
- Current sfincs_jax gap table linked from docs.

Exit gate:

- Team agrees that `plan_final.md` is the single controlling plan.

### M1 - Refactor Skeleton

Deliverables:

- Public API and domain package boundaries finalized.
- `v3_driver.py` reduced to orchestration shim or eliminated.
- Compatibility shims marked and scheduled for deletion.

Exit gate:

- Focused refactor tests and docs build pass.

### M2 - Operator And Solver Ownership

Deliverables:

- Residual/operator objects standardized.
- Solver/preconditioner objects standardized.
- Auto policy no longer requires routine environment variables.

Exit gate:

- Representative RHSMode 1/2/3 cases pass CPU parity with automatic defaults.

### M3 - Ambipolar First-Class Solver

Deliverables:

- Brent, safeguarded Newton/bisection, and pure Newton implemented.
- `dJr/dEr` available via implicit/adjoint derivative.
- CLI and Python API documented.

Exit gate:

- Fortran-compatible roots and derivative checks pass on bounded cases.

### M4 - RHSMode 4/5 And General Sensitivities

Deliverables:

- Fixed-Er adjoint sensitivities implemented.
- Ambipolar-Er adjoint sensitivities implemented.
- Derivative outputs written to HDF5/netCDF.

Exit gate:

- Dot-product, finite-difference, and Fortran v3 adjoint parity gates pass.

### M5 - Optimization And Profile Closure

Deliverables:

- Stable closures for T3D/NEOPAX-style integration.
- QA bootstrap-current and electron-root optimization examples.
- VMEC-JAX/Booz-Xform-JAX/SFINCS-JAX workflow example.

Exit gate:

- Examples run and derivative/provenance gates pass.

### M6 - Production Benchmark Regeneration

Deliverables:

- Fresh CPU/GPU/Fortran benchmark reports.
- Runtime/memory plots regenerated.
- QA/QH bootstrap current plot regenerated.
- Parity matrix regenerated.

Exit gate:

- README and docs contain only current data.

### M7 - Review-Ready PR

Deliverables:

- CI green.
- Docs green.
- Coverage target met or remaining uncovered lines are justified.
- No temporary files or local-path artifacts.
- PR #8 ready for review.

Exit gate:

- The branch is merge-ready.

Status:

- In progress. PR #8 is structurally close to review-ready; the final
  review-ready state waits on the retained-boundary audit, stale-import/layout
  scans, the focused review-lock validation bundle, and PR-body documentation
  of retained large owners and deferred performance/research lanes.

### M8 - Release

Deliverables:

- Merge to main.
- Tag and release.
- PyPI workflow verified.
- Release notes include implemented features, limitations, and benchmark
  provenance.

## Immediate Next Steps

Current completion status:

- Lane 1 structural consolidation: about 97 percent. The original driver split,
  root cleanup, workflow/validation consolidation, mapped-x-grid consolidation,
  public import migration, and compatibility-shim locking are complete. The
  only remaining refactor work is the retained-boundary audit for the four large
  owners and the review-lock validation pass.
- Coverage and future-proof validation: about 85 percent by measured package
  coverage. The newest fast tests cover output streaming, periodic stencils,
  release-data fetching, PETSc reference readers, upstream wrapper behavior,
  active full-FP kinetic block preconditioners, sparse-pattern helpers,
  explicit-sparse settings, profile-system algebra helpers, and RHSMode=2/3
  transport block-preconditioner assembly. Output-safety helpers now also cover
  PAS-projection active-size accounting and solver-trace metadata sanitization.
  The next coverage work must focus on profile/transport solve owners, explicit
  sparse assembly, Schur/profile preconditioners, and profile true-operator
  rescue paths while keeping CI below ten minutes.
- Ambipolar bounded/reference functionality: about 85 percent. Small and
  bounded Fortran-compatible roots and derivatives are implemented; production
  refresh benchmarks remain outside normal CI.
- RHSMode 4/5 sensitivity contracts: about 75 percent. Small fixture contracts
  and derivative identities are implemented; production-grid parity refresh
  remains a release benchmark.
- Public docs/API stabilization for the refactor PR: about 98 percent. Source
  maps, import contracts, README/docs, artifact metadata, Sphinx, source Ruff,
  and CI pass for the consolidated layout in focused checks; the examples
  landing page and Fortran-example status page now keep first-pass navigation
  separate from release-audit runbooks. Rerun the review-lock bundle after this
  plan refresh.

Next ordered implementation steps:

1. Continue the coverage ramp with bounded tests for the highest-risk remaining
   modules: profile solve/system assembly, transport solve/output writer,
   explicit sparse assembly, Schur/profile preconditioners, and true-operator
   rescue. Prefer frozen references, algebraic invariants, and production-shape
   admission tests over slow full solves.
2. Run the stale-import and layout scans: no top-level `rhs1_*` or
   `transport_*` files, no public example/script imports of `v3_driver`, no
   imports of deleted validation/workflow modules except compatibility aliases,
   and no tracked generated outputs or large temporary artifacts.
3. Add examples/tutorial smoke coverage for the fast tutorial script and one
   canonical script per major workflow folder, keeping the CI example budget
   below ten minutes.
4. Run the review-lock validation bundle: import-contract tests, CLI/output
   tests, workflow/validation behavior tests, representative RHSMode 1/2/3,
   ambipolar, RHSMode 4/5 sensitivity, sparse-PC, QI, and differentiable
   linear-solve tests, plus Sphinx `-W`, Ruff, py_compile, and `git diff --check`.
5. Regenerate release-facing benchmark, parity, runtime/memory, and
   bootstrap-current figures only after the source/test structure is stable.
6. Fix only real regressions found by these gates. Do not start another
   performance or file-movement lane in this PR unless it blocks correctness.
7. Commit and push the final validation evidence to the PR branch, update the
   PR body with retained boundaries and validation status, then mark PR #8
   ready for review.

Completion definition for this plan:

- `plan_final.md` remains the only authoritative refactor plan.
- Package root is reduced or every retained root module is justified by the
  manifest.
- No deleted implementation name is advertised as live API.
- Current physics, parity, autodiff, CLI, output, and benchmark gates still
  pass.
- PR #8 is review-ready with no hidden generated files or local artifacts.

## Known Risks And Explicit Deferred Items

- A fully native MUMPS/SuperLU_DIST-equivalent factorization stack is a large
  numerical project. The near-term target is robust native block/Schur/factor
  infrastructure with honest gates, not an overclaim that it matches mature
  sparse direct solvers on every production case.
- Full production-grid QA/QH RHSMode 1 can remain slower or higher-memory than
  Fortran v3 if residuals and outputs are correct and the limitation is
  documented. It cannot be used for favorable public performance claims unless
  the benchmark gate passes.
- True device-QI and single-case multi-GPU strong scaling remain research lanes
  until strict residual, runtime, and reproducibility gates pass.
- GeometryScheme 13 and full external-Phi1 parity require a scoped audit before
  implementation. They should not block core RHSMode 1/2/3/ambipolar/adjoint
  release milestones unless users require them.
- Lineax, JAXopt, Equinox, and other ecosystem libraries should be adopted only
  after a measured benefit. Avoid adding dependencies for conceptual elegance
  without runtime, memory, compile-time, or derivative-maintenance wins.
