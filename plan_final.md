# SFINCS_JAX Final Review Plan

Last updated: 2026-07-06

Active branch: `refactor/v3-driver-architecture`

Review branch / PR: `refactor/v3-driver-architecture` / PR #8

This file is the single active plan for making the refactor branch review-ready.
`plan.md` is the historical execution log. Do not create another competing plan.

## One-Sentence Goal

Finish `sfincs_jax` as a compact, domain-organized, research-grade
neoclassical transport package: users provide a geometry and input file and get
accurate CPU/GPU results with automatic robust solver selection, while Python
users can opt into differentiable residual, flux, ambipolar, sensitivity, and
optimization workflows with parity against SFINCS Fortran v3 where the physics
models overlap.

## Current State

The main structural refactor is functionally complete:

- The package root contains only public entry points, stable support APIs, and
  documented compatibility facades.
- The implementation tree has one level of domain folders below `sfincs_jax/`.
  There are no nested source package directories and no `__init__.py`-only
  package stubs.
- The active domain folders are `discretization/`, `geometry/`, `operators/`,
  `outputs/`, `physics/`, `problems/`, `solvers/`, `validation/`, and
  `workflows/`.
- The expected package shape is guarded by
  `tests/fixtures/source_tree_expected.json` and
  `tests/test_source_tree_consolidation.py`.
- `sfincs_jax/README.md` documents the source layout, user-facing root modules,
  domain owners, compatibility policy, large-data policy, and contributor move
  rules. It also carries a guarded implementation-owner map for the main
  operator, problem, solver, output, and validation files so contributors can
  find canonical owners without following historical helper modules.
- `examples/README.md` and `docs/examples.rst` provide task-oriented example
  navigation, including tutorial notebooks and runnable scripts.
- `examples/README.md` includes an application-recipe map for the most common
  user goals: CLI output and plotting, analytic tokamak inputs, VMEC
  `wout_path`, RHSMode=2/3 transport matrices, bootstrap-current/Redl
  comparisons, ambipolar electric-field scans, autodiff, VMEC/Boozer/JAX
  handoff, QA/QI optimization, CPU/GPU timing, and frozen Fortran-v3 parity.
  The map is guarded by `tests/test_examples_tree_contract.py`, which checks
  both labels and every referenced script or input file.
- The 2026-07-05 source/navigation focused guard passed:
  `tests/test_source_tree_consolidation.py`,
  `tests/test_domain_package_import_contracts.py`, and
  `tests/test_examples_tree_contract.py` as `37 passed in 3.42 s`. The package
  tree has `16` allowed root modules, `9` one-level domain folders, no nested
  packages, and no `__init__.py`-only package stubs. The target and allowed
  root-module lists now match.
- The 2026-07-05 example/docs wording guard passed:
  `tests/test_examples_tree_contract.py` and
  `tests/test_benchmark_doc_claims.py` as `13 passed in 0.15 s`.
- The testing documentation coverage threshold is synchronized with the CI
  workflow. `tests/test_benchmark_doc_claims.py` parses
  `.github/workflows/ci.yml` and verifies that `docs/testing.rst` reports the
  same fail-under gate and staged path to the `95%` target.
- The policy-owner cleanup pass removed the dead private
  `profile_policies._qi_device_solver_env` wrapper. The real QI device solver
  environment normalizer remains owned by `solvers/preconditioner_qi_device.py`;
  policy and source-tree guards passed after the deletion.
- The runtime/memory benchmark figure generator now writes deterministic PDF
  metadata. The tracked benchmark summary regenerates cleanly from the verified
  CPU/GPU reports, and `tests/test_generate_fortran_suite_benchmark_summary.py`
  checks the fixed creation date plus repeatable PDF bytes.
- The 2026-07-05 user-facing documentation wording pass removed active prose
  labels such as "latest snapshot", "current release", "previous best", and
  progress-log phrasing from README-facing docs pages. The active-doc scan only
  retains false positives for "concurrently"; Sphinx `-W` and docs contract
  tests passed after the edit.
- The 2026-07-05 architecture-wording pass removed public "handoff" and
  deleted-driver phrasing from active README/docs/examples prose. The optional
  VMEC/Boozer path is now described as a workflow/interface, and the
  finite-beta example records the VMEC file boundary explicitly in its
  differentiability contract.
- `docs/examples.rst` carries the same application-recipe map as the examples
  README, with `tests/test_examples_tree_contract.py` checking that the README
  and ReadTheDocs entry points stay synchronized. The docs-sync focused guard
  passed as `14 passed in 0.12 s`.
- Transport worker payload tests cover scalar elapsed-time arrays,
  short elapsed-time arrays, missing RHS norms, and the explicit
  `set_child_environment=False` lane used by injected worker runtimes. Focused
  validation passed:
  `tests/test_transport_parallel_payload.py`,
  `tests/test_transport_parallel_runtime.py`, and
  `tests/test_transport_parallel_validation.py` as `37 passed in 0.44 s`.
- Transport parallel-runtime coverage now also covers XLA flag rewriting,
  release-quality transport-worker scaling audits, parallel-claim scope
  classification, multi-GPU throughput audits, sharded-solve audit gates,
  sharding/amortization/operator-reuse/deterministic-output plans, worker
  environment restoration, payload construction, and GPU-dispatch injection.
  These are bounded policy tests and launch no production solves. Focused
  validation passed as `59 passed in 0.64 s`; source-tree/import validation
  passed with the bundle as `112 passed in 5.43 s`.
- Profile sparse-PC diagnostics now cover malformed structured f-block
  metadata, the non-Fortran/global sparse-PC static metadata branch,
  precomputed metadata-section injection, zero-target residual ratios, and
  fail-fast type guards for Fortran-reduced xblock metadata sections. These
  are bounded report-schema and solver-policy tests for production diagnostics
  and add no solve time to CI. Focused validation passed:
  `tests/test_profile_response_diagnostics.py` as `25 passed in 0.57 s`; the
  broader diagnostics/finalization bundle passed as `51 passed in 0.67 s`;
  source-tree and import-contract guards passed as `35 passed in 2.83 s`.
- RHSMode=1 preconditioner policy coverage now includes the active-backend GPU
  sparse-fallback skip wrapper, including GPU acceptance, CPU rejection, Phi1
  rejection, and missing-PAS rejection. A source/test audit for
  `problems/profile_policies.py` reports zero production-used public policy
  helpers without direct tests. Focused validation passed:
  `tests/test_rhs1_preconditioner_auto_policy.py` as `30 passed in 0.20 s`.
- RHSMode=1 preconditioner-build coverage now includes direct tests for
  automatic threshold environment readers and direct strong-preconditioner
  reduced/full builder functions. A source/test audit for
  `problems/profile_preconditioner_build.py` reports zero production-used
  public helpers without direct tests. Focused validation passed:
  `tests/test_profile_response_preconditioner_build.py` as
  `20 passed in 0.44 s`.
- RHSMode=2/3 transport linear-system coverage now includes direct module
  boundary tests for active-DOF selection and both FP transport preconditioner
  builder fallback contracts. A source/test audit for
  `problems/transport_linear_system.py` reports zero production-used public
  helpers without direct tests. Focused validation passed:
  `tests/test_transport_active_factor.py` as `30 passed in 0.55 s`.
- RHSMode=1 dense profile-solve coverage now includes a direct public
  `solve_profile_linear` identity-system test beside the existing residual and
  dispatch tests. A source/test audit for `problems/profile_dense.py` reports
  zero production-used public helpers without direct tests. Focused validation
  passed: `tests/test_profile_response_linear_solve.py` as
  `13 passed in 2.26 s`.
- RHSMode=1 sparse x-block coverage now includes a direct disabled-branch
  context-schema test for the public `run_xblock_sparse_pc_branch` entry
  point. A source/test audit for `problems/profile_sparse_xblock.py` reports
  zero production-used public helpers without direct tests. The broader
  sparse-PC/source-guard bundle passed as `365 passed in 5.86 s`.
- RHSMode=1 sparse-solve orchestration coverage now includes direct bounded
  tests for requested sparse-PC GMRES dispatch, factor preflight residual/seed
  bookkeeping and diagnostics, auto-preflight retry no-op and accepted-candidate
  routing, residual candidate acceptance/rejection, residual-correction no-op
  routing plus true-active, true-window, residual-coarse, residual-window, and
  column-cache rescue branches, true-coupled coarse no-op and accepted routing,
  generic x-block backend deferral, active global-pattern setup, direct-tail
  host-factor fallback, direct-tail structured selection, support-mode preflight
  promotion, direct-tail rescue-policy defaults, and the Fortran-reduced x-block
  capability guard. A source/test audit for `problems/profile_sparse_solve.py`
  reports zero production-used public helpers without direct tests. The owner
  suite reports `profile_sparse_solve.py` at `74%` and
  `profile_sparse_direct.py` at `95%` with `362 passed in 7.00 s`; the
  sparse-PC/source-guard validation bundle passed as `415 passed in 9.24 s`.
- Speed-grid and mapped-grid numerics coverage now includes direct tests for
  SFINCS-v3 x-grid weights/derivative-ratio formulas, mapped barycentric
  differentiation, mapped-grid regularization diagnostics, vector Maxwellian
  moment helpers, relative moment errors, and dtype byte accounting. Source/test
  audits for `discretization/adaptive_maps.py`, `discretization/xgrid.py`,
  `workflows/mapped_xgrid.py`, and `solvers/memory_model.py` report zero
  production-used public helpers without direct tests. Focused validation
  passed as `48 passed in 14.06 s`.
- Geometry and output-format coverage now includes direct tests for Boozer
  `.bc` header/bracketing/effective-radius readers, geometry output cache-key
  construction, and direct NPZ/NetCDF writer payload/overwrite behavior.
  Source/test audits for `geometry/boozer.py` and `outputs/formats.py` report
  zero production-used public helpers without direct tests. Focused validation
  passed as `81 passed in 2.18 s`.
- Output-writer orchestration coverage now includes bounded owner tests for the
  geometry-only writer path with `wout_path`, extension-selected output format,
  solver-trace sidecar metadata, and `return_results`, plus the RHSMode=2/3
  streaming-HDF5 path with temporary environment restoration. These tests
  monkeypatch geometry construction and transport solves, so they cover the
  public writer routing without adding production solve time to CI. Focused
  validation passed as `83 passed in 2.40 s`; the broader output/trace bundle
  passed as `108 passed in 8.13 s`. The exact package-coverage delta is left to
  the next CI coverage shards because local pytest-cov still aborts before
  startup on this machine.
- Output-writer schema coverage now also includes bounded tests for
  geometryScheme 1 helical/tokamak metadata, geometryScheme 4 radial constants,
  VMEC geometryScheme 5 radial interpolation and RHSMode=3 monoenergetic
  `nu_n`/`EStar` overwrites, Boozer geometryScheme 11 header/bracketing
  metadata, export-f grid metadata, cache-miss geometry/classical payload
  retention, cached `uHat`, and verbose VMEC/Boozer progress messages. These
  tests avoid production solves and real equilibrium fixtures by using tiny
  algebraic geometry objects and monkeypatched readers. Focused writer/export
  validation passed as `108 passed in 3.08 s`; source-tree/import validation
  passed as `161 passed in 7.68 s`. Direct owner coverage for
  `outputs/writer.py` improved from `36%` to `52%`; the remaining missing
  region is mostly RHSMode=1/2/3 solve orchestration that should be split or
  covered through targeted monkeypatched solve-owner paths.
- Transport runtime, profile setup, sparse-direct, and diagnostics coverage now
  includes direct tests for worker-count validation, GPU subprocess policy
  injection, persistent-pool helpers, solve-method normalization, explicit
  sparse-pattern controls/cache keys, sparse-PC static metadata wrappers, and
  x-block sparse-PC result diagnostics wrappers. Source/test audits for
  `problems/transport_parallel_runtime.py`, `problems/profile_setup.py`,
  `problems/profile_sparse_direct.py`, and `problems/profile_diagnostics.py`
  report zero production-used public helpers without direct tests. Focused
  validation passed as `82 passed in 34.47 s`.
- Transport parallel-runtime coverage now also includes direct fail-closed tests
  for GPU worker-array validation, the standard GPU subprocess policy wrapper,
  one-shot process-pool fallback, worker-payload packing, missing RHS-norm
  packing, scaling-threshold validation, malformed payload provenance,
  malformed compile-amortization gates, malformed sharded deterministic gates,
  and malformed multi-GPU throughput artifacts. Focused validation passed as
  `82 passed in 0.51 s`; Ruff and `git diff --check` passed. A local
  pytest-cov/module-coverage retry aborted before pytest startup due to the
  user-site `coverage`/`numpy` importer issue
  (`ImportError: cannot load module more than once per process`), so the
  package-wide coverage percentage remains the last successful CI-mode
  measurement below until the next full coverage run.
- Transport parallel-runtime coverage now additionally exercises periodic
  GPU-worker progress logging, non-residual GPU subprocess failure reporting,
  malformed release-scaling boolean provenance, malformed payload coverage, and
  malformed compile-amortization notes. Focused validation passed as
  `84 passed in 0.46 s` with user-site packages disabled and
  `30 passed in 0.36 s` for the touched runtime owner under the standard local
  pytest invocation. Exact coverage deltas remain delegated to CI because
  local `coverage run` still aborts after NumPy reload on this machine.
- CI coverage recovery now keeps the fast RHSMode=2/3 transport preconditioner
  unit tests active in CI while continuing to skip only the slower
  transport-matrix parity and write-output integration files. The previous
  broad `test_transport_matrix_` CI skip hid direct owner tests for
  `solvers/preconditioner_transport_matrix.py`; the narrowed skip keeps
  `tests/test_transport_matrix_preconditioners.py` active. Focused validation
  passed under CI mode as `16 passed, 26 skipped in 5.24 s`; standalone
  validation of the recovered test file passed as `16 passed in 5.25 s`.
- CI coverage recovery also keeps the fast output-HDF5 scheme parity tests
  active in CI. Individual timing showed scheme1, scheme2, scheme4,
  scheme4-quick2species, and scheme5 each complete in about `1.3-1.5 s`;
  scheme11 remains skipped because it exceeded the bounded probe. The narrowed
  CI selection passed as `8 passed, 1 skipped in 1.85 s`, restoring fast
  writer/output parity coverage without admitting the slow scheme11 path.
- CI coverage recovery now also keeps bounded RHSMode=1 write-output,
  RHSMode=1 Phi1 write-output, state-recycle, ambipolar scan, and upstream
  scanplot tests active in CI. These tests exercise real writer/physics
  contracts while staying below production-run budgets; the recovered CI-mode
  bundle passed as `14 passed in 26.74 s`.
- Structured velocity, transport policy, sensitivity, and validation coverage
  now includes direct tests for block-tridiagonal solves, structured tz-FFT
  first-attempt policy/budget/environment thresholds, JVP/VJP flux wrappers,
  release-hosted data cache paths, and QI campaign JSON/mapping APIs.
  Source/test audits for `discretization/structured_velocity.py`,
  `problems/transport_policies.py`, `sensitivity.py`,
  `validation/data_fetch.py`, and `validation/qi_device.py` report zero
  production-used public helpers without direct tests. Focused validation
  passed as `110 passed in 62.60 s`.
- PAS policy, symbolic sparse-profile, and x-block policy coverage now
  includes direct owner tests for PAS applicability and memory ceilings,
  Fortran-v3 support-mode parsing, sparse ILU/LU byte estimates, sparse row and
  column equilibration, x-block local-factor tuning, and side-probe active-size
  floors. Source/test audits for `solvers/preconditioner_pas_policy.py`,
  `solvers/preconditioner_symbolic_profile.py`, and
  `solvers/preconditioner_xblock_policy.py` report zero production-used public
  helpers without direct tests. Focused validation passed as
  `17 passed in 0.87 s`, with the remaining LU-size helper test passing as
  `1 passed in 0.81 s`.
- Public Krylov wrapper coverage now includes direct tests for BiCGStab,
  TFQMR, and distributed-GMRES host fallback entry points on tiny linear
  systems. A source/test audit for `solver.py` reports zero production-used
  public helpers without direct tests. Focused validation passed as
  `3 passed in 1.50 s`.
- Public distributed-Krylov wrapper coverage now also exercises mesh-present
  fallback when no sharded callable is available, plus fake-sharded pad/trim
  behavior for non-device-divisible vectors with a preconditioner. The focused
  distributed subset passed as `8 passed, 51 deselected in 1.31 s`; the full
  solver GMRES owner file passed as `59 passed in 23.63 s`.
- Public ambipolar scan-postprocessing coverage now exercises non-`Er` scan
  variables, normalized-field-to-`Er` conversion, no-root scan output, malformed
  scan-directory failures, Fortran-style boolean fallback parsing, single-
  species Phi1 scanplot output rows, and invalid-rank fail-fast behavior. This
  keeps the CLI/postprocessing path future-proof without running new solves.
  Focused validation passed:
  `tests/test_er_scan_and_ambipolar.py tests/test_helper_module_coverage.py` as
  `15 passed in 7.89 s`.
- Native block-factor, QI basis, sparse cache-key, and host sparse-factor
  coverage now includes direct tests for native x-ell factor construction and
  application, QI global moment-basis rank gating, RHSMode-1 PAS/x-block sparse
  cache-key wrappers, host sparse factor cache reuse, and the bounded
  x-block-TZ host skip path. Source/test audits for
  `solvers/native_block_factor.py`, `solvers/preconditioner_qi_basis.py`,
  `solvers/preconditioner_pas_xblock_ilu.py`,
  `solvers/preconditioner_xblock_tz_sparse.py`, and
  `solvers/preconditioner_symbolic_host.py` report zero production-used public
  helpers without direct tests. Focused validation passed as
  `5 passed in 1.17 s`.
- Profile f-block operator-construction coverage now includes direct reduced
  fixture tests for collisionless, PAS, FP, and Phi1 FP from-namelist builders,
  plus the RHSMode-1 f-block layout adapter. Source/test audits for
  `operators/profile_fblock.py` and `operators/profile_kinetic.py` report zero
  production-used public helpers without direct tests. Focused validation
  passed as `1 passed in 1.72 s`.
- Optimization and preconditioning wrapper coverage now includes direct tests
  for QA neoclassical proxy components, promotion-pair comparison metrics, and
  unsharded V3 submatrix probing through the public preconditioning helper.
  Source/test audits for `workflows/optimization.py` and
  `solvers/preconditioning.py` report zero production-used public helpers
  without direct tests. Focused validation passed as `2 passed in 1.89 s`.
- Ambipolar and sparse-finalization wrapper coverage now includes direct tests
  for matrix-free radial-current derivative adaptation, the SFINCS_JAX-backed
  Brent wrapper's evaluator/root-solver wiring, and sparse-PC factor-dtype
  retry from finalization context. Source/test audits for
  `problems/ambipolar.py` and `problems/profile_sparse_finalization.py` report
  zero production-used public helpers without direct tests. Focused validation
  passed as `3 passed in 1.76 s`.
- Structured full-FP f-block preconditioner coverage now exercises all eight
  public builders through the canonical `solvers/preconditioner_full_fp_structured.py`
  owner rather than compatibility aliases. A source/test audit for that module
  reports zero production-used public helpers without direct tests. Focused
  validation passed as `8 passed in 15.08 s`.
- Active x-block preconditioner coverage now includes direct fail-closed
  admission tests for the global field-split Schur, multiline field split,
  bounded native stack, Fortran-v3 reduced native stack, diagonal Schur, x-ell
  kinetic line, and angular-line builders. It also checks identity-system
  application for the native x-ell and angular line builders. A direct-reference
  audit for `solvers/preconditioner_xblock_active.py` reports zero remaining
  untested public builder entry points. Focused validation passed as
  `14 passed in 0.87 s`.
- QI sparse-pipeline coverage now includes a direct default-disabled
  orchestration test for `run_xblock_qi_preconditioner_pipeline`, verifying
  that the complete optional QI lane preserves the base preconditioner and
  diagnostics without constructing coarse/device stages. A direct-reference
  audit for `problems/profile_sparse_qi.py` reports no remaining untested public
  pipeline entry point from the current audit slice. Focused validation passed
  with the active x-block tranche as `16 passed in 1.78 s`.
- Sparse/QI result-contract coverage now directly exercises explicit sparse
  factor metadata, QI global-moment basis/closure state, sparse x-block result
  containers, QI pipeline context/result containers, matrix-free QI seed
  setup/attempt containers, device-QI smoothers/preconditioner states/probes,
  QI block-Schur/coupled-residual/deflation/multilevel/residual-Galerkin/
  Galerkin-selection/two-level contracts, and transport active-dense/
  active-block admission containers. A direct-reference audit for
  `problems/profile_sparse_xblock.py`, `problems/profile_sparse_qi.py`,
  `solvers/explicit_sparse.py`, `solvers/preconditioner_qi_basis.py`,
  `solvers/preconditioner_qi_device.py`,
  `solvers/preconditioner_qi_corrections.py`, and
  `problems/transport_linear_system.py` reports zero untested public
  definitions in this audited slice. Focused validation passed as
  `137 passed in 24.21 s`; Ruff passed on all touched files.
- RHSMode=1 policy/sparse-result contract coverage now directly exercises the
  remaining public policy dataclasses in `problems/profile_policies.py`, the
  sparse-PC stage result containers in `problems/profile_sparse_solve.py`, and
  the structured full-CSR selection/solve-result serialization contract in
  `operators/profile_full_system.py`. The direct-reference audit for those
  three large owners reports zero untested public definitions. Focused
  validation passed as `4 passed in 0.55 s`; Ruff passed for the new test file.
- Solver/setup policy contract coverage now directly exercises the remaining
  public setup and result containers in `problems/profile_setup.py`,
  `problems/profile_dense.py`, `problems/transport_parallel_runtime.py`,
  `problems/transport_policies.py`, `problems/transport_setup.py`,
  `solvers/preconditioner_xblock_policy.py`, `solvers/native_block_factor.py`,
  and `solvers/preconditioner_symbolic_policy.py`. The direct-reference audit
  for those eight owners reports zero untested public definitions. Focused
  validation passed as `4 passed in 0.72 s`; Ruff passed for the new test file.
- The RHSMode=1 structured full-CSR Schur tests now cover singular local
  zeta-line, pitch-line, and x-pitch kinetic block fallback through
  pseudo-inverses, plus the corresponding regularized inverse paths. This
  protects the bounded local-factor path used when production grids contain
  rank-deficient active kinetic blocks. Focused Schur/preconditioner validation
  passed:
  `tests/test_rhs1_full_csr_schur_preconditioners.py`,
  `tests/test_rhs1_coarse_basis.py`, `tests/test_rhs1_coarse_policy.py`, and
  `tests/test_rhs1_schur_policy.py` as `55 passed in 0.86 s`.
- The RHSMode=1 active x-block projected preconditioner tests now cover
  empty/no-match active-index projection, SciPy CSR byte accounting, and
  fail-closed integer/float/boolean environment parser behavior. These are
  bounded policy/numerics checks for the active-DOF preconditioner admission
  path and do not add solve time to CI. Focused validation passed:
  `tests/test_rhs1_active_projected_xblock.py` as `8 passed in 0.32 s`.
- The same active x-block test owner now also covers the JAX-native indexed
  Schwarz active preconditioner: exact residual parity on a diagonal active
  system, fail-closed nonsquare/size-mismatch/empty/out-of-range admission
  branches, empty block-family rejection, and local-base dispatch to the
  canonical native owner. Focused validation passed:
  `tests/test_rhs1_active_projected_xblock.py` as `11 passed in 0.86 s`.
- The profile full-system operator tests now cover matrix-free matvec
  sharding-policy decisions, fail-closed invalid shard-axis handling,
  default-on shard padding, and full-vector pad/unpad round trips along theta,
  zeta, and speed-grid axes on the tiny Fortran-v3 fixture. These checks
  protect the CPU/GPU matvec infrastructure without adding production solves.
  Focused validation passed:
  `tests/test_full_system_operator_jit.py` as `5 passed in 6.93 s`.
- The public solver-kernel helper tests now cover restart memory-policy
  fail-closed branches, distributed-input host materialization, and
  right-preconditioned BiCGStab physical-initial-guess semantics. These checks
  protect user-facing CLI/Python solver behavior without adding slow transport
  solves. Focused validation passed:
  `tests/test_solver_heavy_helper_coverage.py` as `11 passed in 0.50 s`.
- The latest local xdist coverage audit measured `88%` package coverage:
  `4031 passed in 283.48 s` with `8089` missing executable lines. The latest
  explicit-sparse and true-operator rescue tranches reduced missing executable
  lines by `109` while keeping the full audit below five minutes.
- A serial CI-mode coverage audit after deleting the driver shim measured
  `86%` package coverage and wrote a complete term-missing table before manual
  interrupt during JAX teardown: `3938 passed, 195 skipped, 5 failed in
  1042.11 s`. The failures were all stale release/research manifest paths that
  still referenced `sfincs_jax/v3_driver.py`; follow-up release/research-lane
  validation passed as `21 passed in 0.22 s` after updating those paths to
  canonical owners.
- The latest bounded coverage tranches added RHSMode-1 Schur/coarse fallback
  tests, output-gradient coordinate contract tests, default
  preconditioner-selection tests, radial-preconditioner guard tests,
  differentiability-boundary tests, and RHSMode-1 output-trace contract tests.
  A full non-coverage regression check after the first bundle passed:
  `3925 passed in 247.02 s`.
- A full non-coverage regression check after the current coverage/docs/import
  bundle passed: `3940 passed in 241.53 s`.
- The next fast coverage tranche added 25 `discretization.v3` contract tests
  for Boozer `.bc` header parsing, VMEC `.txt -> .nc` path fallback, content
  identities, persistent geometry-cache round trips, stale-cache rejection,
  Fortran-style odd grid sizing, RHSMode=3 monoenergetic x-grid overrides,
  `Nxi_for_x` policies, and fail-closed unsupported knobs. Focused validation:
  `tests/test_discretization_v3_contracts.py` passed in `0.90 s`; adjacent
  mapped-grid tests plus the new contracts passed as `29 passed in 1.58 s`;
  source-tree and import-contract guards passed as `26 passed in 2.78 s`.
- The full local regression after the discretization tranche passed:
  `3965 passed in 595.09 s`. This is within the 10-minute target but leaves
  little room for additional slow solve tests; the 95% coverage push should use
  bounded unit, policy, schema, and frozen-fixture tests or delete obsolete
  code rather than adding expensive production solves to default CI.
- The active reduced-system setup used by RHSMode=1 active-DOF and PAS
  projection solves is now owned by `problems/profile_setup.py` as
  `build_rhs1_active_reduced_system_setup`. Focused validation passed:
  `tests/test_profile_response_setup.py` as `20 passed in 1.26 s`, the
  source-tree/import-contract bundle as `46 passed in 3.35 s`, the driver-level
  profile/recycle bundle as `27 passed in 5.38 s`, and the combined focused
  tranche as `73 passed in 8.38 s`. `ruff`, `compileall`, and `git diff
  --check` also passed for the changed files.
- The RHSMode=1 preconditioner route setup used by the driver is now owned by
  `problems/profile_policies.py` as
  `resolve_rhs1_preconditioner_route_setup`. Focused validation passed:
  `tests/test_rhs1_preconditioner_auto_policy.py` as `29 passed in 0.16 s`,
  the profile/recycle driver bundle as `27 passed in 6.09 s`, and the
  source-tree/setup/import-contract bundle as `46 passed in 3.73 s`. This
  tranche reduced `problems/profile_solve.py` to `4821` lines and
  `solve_v3_full_system_linear_gmres` to `3912` lines without adding files.
- The profile linear-solve dispatch setup is now owned by
  `problems/profile_dense.py` as `ProfileLinearSolveDispatch`, and structured
  f-block preconditioner metadata recording is now owned by
  `problems/profile_diagnostics.py`. Focused validation passed:
  profile diagnostics/linear tests as `30 passed in 1.89 s`, and the broader
  profile/refactor bundle passed as `132 passed in 10.01 s`. This reduced
  `problems/profile_solve.py` to `4745` lines and
  `solve_v3_full_system_linear_gmres` to `3836` lines without adding files.
- The RHSMode=1 output solve-method selector used by
  `write_sfincs_jax_output_h5` is now owned by `outputs/rhsmode1.py` as
  `select_rhsmode1_solve_method`. Focused validation passed:
  output/CLI policy tests as `103 passed in 1.32 s`, and HDF5/transport output
  checks as `8 passed in 5.83 s`. This reduced `outputs/writer.py` to `3971`
  lines and `write_sfincs_jax_output_h5` to `2323` lines without adding files.
- RHSMode=1 output correction helpers for constraintScheme=0 gauge alignment,
  PAS no-Phi1 output scaling, and large PAS no-Phi1 flow/current
  Fortran-reference alignment are now owned by `outputs/rhsmode1.py`. Focused
  validation passed: output/CLI/HDF5 policy checks as `114 passed in 6.86 s`.
  This reduced `outputs/writer.py` to `3734` lines and
  `write_sfincs_jax_output_h5` to `2083` lines without adding files.
- RHSMode=1 core diagnostic, Phi1 scalar, and electric-drift output schema
  writes are now owned by `outputs/rhsmode1.py`, including coordinate-converted
  flux variants, quasineutrality debug arrays, `vd`/`vd1` totals, and
  `heatFlux_withoutPhi1` fields. Focused validation passed:
  `tests/test_io_output_policy_coverage.py` as `61 passed in 0.90 s`, and the
  broader output/profile/source-tree bundle passed as `220 passed in 13.92 s`.
  This reduced `outputs/writer.py` to `3639` lines and
  `write_sfincs_jax_output_h5` to `1985` lines without adding files.
- RHSMode=1 per-iteration classical flux output is now owned by
  `outputs/rhsmode1.py`, including Phi1-history evaluation and coordinate
  variants for classical particle and heat fluxes. Focused validation passed:
  `tests/test_io_output_policy_coverage.py` as `62 passed in 1.60 s`, and the
  broader output/profile/source-tree bundle passed as `221 passed in 14.43 s`.
  This reduced `outputs/writer.py` to `3570` lines and
  `write_sfincs_jax_output_h5` to `1915` lines without adding files.
- RHSMode=1 NTV diagnostic recomputation is now owned by `outputs/rhsmode1.py`,
  including the geometryScheme=5 zero path and the non-axisymmetric L=2
  recomputation path. Focused validation passed:
  `tests/test_io_output_policy_coverage.py` as `64 passed in 1.85 s`, and the
  broader output/profile/source-tree bundle passed as `224 passed in 17.89 s`.
  This reduced `outputs/writer.py` to `3508` lines and
  `write_sfincs_jax_output_h5` to `1852` lines without adding files.
- RHSMode=1 output solve-method selector branch coverage now includes explicit
  environment overrides, tokamak PAS+Er sparse-PC, tokamak Er dense, 3D
  full-FP x-block, and constrained-PAS sparse-PC policy branches. Focused
  validation passed: `tests/test_io_output_policy_coverage.py` as
  `69 passed in 1.88 s`; the broader output bundle passed as
  `152 passed in 8.20 s`.
- The Fortran-compatible `export_f` output-grid mapping and distribution
  projection machinery is now owned by `outputs/formats.py` as
  `ExportFConfig`, `_export_f_config`, and `_apply_export_f_maps`; `writer.py`
  keeps compatibility aliases only. Focused validation passed:
  `tests/test_io_export_and_h5_coverage.py tests/test_io_output_policy_coverage.py`
  as `84 passed in 1.78 s`. This reduced `outputs/writer.py` to `3275`
  lines without adding source files.
- The duplicated RHSMode=1 and transport `export_f` state-vector HDF5 dataset
  writes now share `outputs/formats.py::write_export_f_state_vectors_to_data`.
  Focused validation passed as `86 passed in 1.87 s`, and the broader
  output/source guard passed as `180 passed in 9.86 s`. This reduced
  `outputs/writer.py` to `3250` lines.
- The RHSMode=1 current-backend preconditioner-policy wrapper layer now has
  bounded unit coverage through `tests/test_rhs1_host_policy.py`, validating
  CPU/GPU-sensitive wrapper delegation without running a solve. Focused policy
  validation passed as `122 passed in 0.43 s`.
- The RHSMode=1 policy environment readers and post-solve correction policies
  now have direct bounded coverage for Fortran-style boolean tokens,
  lower-bound clamps, disabled-step behavior, and namespaced coarse/residual
  correction controls. This also removed duplicate private env-reader
  definitions from `problems/profile_policies.py`; focused policy validation
  passed as `158 passed in 0.71 s`.
- The RHSMode=1 full-FP post-Krylov residual-polish ladder is now owned by
  `problems/profile_residual.py` as `run_rhs1_fp_post_solve_polish`, with the
  profile driver passing solver/preconditioner builders as explicit
  dependencies. Focused validation covered no-op gates, damped residual
  improvement, low-L block admission, projected L=1 correction, global low-L
  projected correction, and BiCGStab polish admission as
  `tests/test_rhs1_residual.py` (`23 passed in 0.69 s`) and
  `tests/test_profile_solve_module_wrappers.py` (`13 passed in 0.63 s`); the
  broader source/profile/policy bundle passed as `143 passed in 3.91 s`. This
  reduced `problems/profile_solve.py` to `4402` lines and
  `solve_v3_full_system_linear_gmres` to `3494` lines without adding files.
- Dead private profile-solve wrappers for GMRES dispatch and structured
  f-block cache-key injection were removed after confirming production code no
  longer used them; canonical coverage remains in `solvers/krylov_dispatch.py`
  and `solvers/preconditioning.py`. A duplicate unused xblock elapsed-time
  helper was also deleted. Focused validation passed:
  `tests/test_krylov_dispatch.py`, `tests/test_profile_solve_module_wrappers.py`,
  `tests/test_profile_solve_policy_coverage.py`, and
  `tests/test_profile_rhs1_dispatch_coverage.py` as `69 passed in 24.87 s`;
  the touched Schur heuristic test passed as `2 passed in 2.29 s`; source-tree
  and import-contract guards passed as `35 passed in 2.85 s`. This reduced
  `problems/profile_solve.py` to `4351` lines and
  `problems/profile_sparse_xblock.py` to `7681` lines without adding files.
- A private-symbol audit removed unreferenced helpers from
  `operators/profile_drifts.py`, `physics/collisions.py`, and
  `problems/profile_diagnostics.py`. The remaining audit hits are intentional
  private reference helpers that are tested or used by live progress/labeling
  code. Focused validation passed: collision/diagnostics tests as
  `46 passed in 5.34 s`, f-block stencil/parity tests as
  `21 passed in 8.11 s`, and source-tree/import guards as
  `35 passed in 2.84 s`.
- The RHSMode=2/3 transport linear-solve dispatch layer is now owned by
  `problems/transport_linear_system.py`, including
  `TransportLinearSolveContext`, `TransportLinearSolveCallbacks`,
  `solve_transport_linear`, `solve_transport_linear_with_residual`,
  `transport_solver_kind`, and `transport_restart_for_method`.
  `problems/transport_solve.py` keeps compatibility re-exports and now focuses
  on whichRHS orchestration. Focused validation passed:
  `tests/test_transport_linear_solve.py` as `20 passed in 2.16 s`, source-tree
  and import-contract guards as `26 passed in 2.93 s`, plus Ruff and compile
  checks. This reduced `problems/transport_solve.py` to `2949` lines without
  adding files.
- The RHSMode=2/3 dense-LU fallback and host SciPy GMRES support are also now
  owned by `problems/transport_linear_system.py`. `transport_solve.py` keeps
  public re-exports and private compatibility aliases, while the canonical
  tests patch the linear-system owner. Focused validation passed:
  dense/host/linear/loop transport tests as `50 passed in 2.42 s`, plus Ruff on
  the touched modules. This reduced `problems/transport_solve.py` to `2811`
  lines without adding files.
- The RHSMode=2/3 dense batched whichRHS fallback is now owned by
  `problems/transport_linear_system.py` with compatibility imports preserved
  from `transport_solve.py`. Focused validation passed:
  dense/host/linear/loop tests as `50 passed in 2.40 s`, the broader
  dense/host/sparse transport support bundle as `117 passed in 12.07 s`, and
  source/import guards as `26 passed in 2.59 s`. This reduced
  `problems/transport_solve.py` to `2589` lines without adding files.
- The RHSMode=2/3 loop-local matvec cache, recycled Krylov state, progress
  gate, and recycle-size resolver are now owned by
  `problems/transport_setup.py`, matching the existing max-iteration, state,
  RHS-selection, and parallel-request setup helpers. Compatibility imports from
  `transport_solve.py` are preserved. Focused validation passed:
  loop/linear/dense tests as `45 passed in 2.76 s`, the broader
  dense/host/sparse transport support bundle as `117 passed in 11.97 s`,
  source/import guards as `26 passed in 2.56 s`, plus Ruff, compile, and diff
  checks. This reduced `problems/transport_solve.py` to `2326` lines without
  adding files.
- A bounded active-DOF transport Krylov regression now compares the active and
  full transport solve paths on the tiny SFINCS Fortran v3 reference fixture,
  forcing non-dense Krylov behavior and checking residual-clean state-vector
  parity. Focused validation passed: `tests/test_transport_linear_solve.py` as
  `21 passed in 7.35 s`; the broader dense/host/sparse transport support bundle
  passed as `118 passed in 13.31 s`. This is a real numerical branch gate, not
  a smoke test.
- The native explicit-sparse nested-dissection factor now supports batched
  right-hand sides at the top-level solve wrapper, matching its local
  multi-RHS implementation and avoiding an identity fallback during residual
  polish probes. A bounded residual-polish gate in `tests/test_explicit_sparse.py`
  checks vector and batched true-residual parity against a dense reference;
  focused validation passed as `105 passed in 3.82 s`, with the adjacent
  explicit-sparse coverage slice improving from `81%` to `82%`.
- Transport linear-system coverage now includes the non-transport default
  solver route, dense direct-solver cache reuse, left-preconditioned host GMRES
  rejection of preconditioned-only convergence, and non-finite reported
  residual handling. Focused validation passed:
  `tests/test_transport_linear_solve.py` as `23 passed in 7.21 s`, and the
  broader transport-policy/source bundle passed as `78 passed in 12.01 s`.
- Transport preconditioner coverage now includes reduced-view equivalence for
  collision, species-x, angular FFT, and FP line factors; singular collision
  and x-grid coarse pseudo-inverse fallbacks; and ExB/DKES angular branches for
  both PAS and full-FP Fourier preconditioners. The current pass additionally
  covers FP-builder no-FP fallback to the collisionless angular factor, Phi1
  reduced-view Schur bypass, x-block host-factor apply failure fallback, and
  rejected structured-fblock selection fallback. Focused validation passed:
  `tests/test_transport_matrix_preconditioners.py` as `19 passed in 5.60 s`,
  and the broader transport preconditioner/policy bundle passed as
  `59 passed in 4.72 s` before this latest branch tranche.
- RHSMode=1 output policy coverage now includes the remaining automatic
  solve-method branches for tokamak full-FP no-Er sparse-PC, 3D full-FP
  sparse-PC, host dense shortcut reporting, dense-auto skip reporting,
  E_parallel BiCGStab, small PAS incremental, and includePhi1 linear
  dense/incremental selection. Focused validation passed:
  `tests/test_io_output_policy_coverage.py` as `76 passed in 1.85 s`; the
  broader output bundle passed as `95 passed in 7.28 s`.
- RHSMode=1 output reference-normalization coverage now includes malformed
  Fortran-HDF5 fail-closed diagnostics and a bounded constraintScheme=0
  gauge-adjustment path that changes only the kinetic distribution and
  preserves the state-vector tail. Focused validation passed:
  `tests/test_io_output_policy_coverage.py` as `78 passed in 2.04 s`.
- Validation artifact coverage now includes suite-report wrapper parsing,
  no-runtime-floor filtering, non-object autodiff-summary rejection, malformed
  gradient-check rejection, and mixed valid/invalid suite-row filtering. Focused
  validation passed: `tests/test_validation_policy_coverage.py` as
  `8 passed in 0.04 s`, and the adjacent validation artifact bundle passed as
  `42 passed in 1.38 s`.
- Explicit sparse-factor coverage now includes materialization and unknown-kind
  fail-closed checks, non-finite Jacobi diagonals, finite Jacobi fallback inside
  singular symbolic blocks, empty symbolic block-coarse factors, solver
  exception rejection in true-residual admission, dense-inverse coarse fallback
  when a coarse LU fails, and non-finite refinement-stop behavior. Focused
  validation passed: `tests/test_explicit_sparse.py` as `78 passed in 0.63 s`,
  and the adjacent explicit-sparse/transport sparse-direct bundle passed as
  `158 passed in 21.71 s`.
- True-operator RHSMode=1 rescue coverage now includes one-hot cache
  fail-closed fallback for malformed batched true actions, residual-window
  least-squares pseudo-inverse fallback, degenerate damping behavior for
  window/active/coupled corrections, residual-window shape and tail-only
  rejection, and invalid or fully dropped true-action columns. Focused
  validation passed: `tests/test_rhs1_true_operator_rescue.py` as
  `18 passed in 0.27 s`, and the adjacent RHSMode=1 sparse-rescue bundle
  passed as `368 passed in 3.41 s`.
- RHSMode=1 x-block sparse coverage now includes structural-threshold dropping
  for selected theta and zeta sparse operators, empty sparse operator assembly,
  non-finite and empty diagonal rejection, active pitch-index flattening across
  species and speed-grid points, species-decoupling policy wrappers, and FP/PAS
  sparse-LU cap defaults. Focused validation passed:
  `tests/test_sparse_assembly.py tests/test_rhs1_sxblock_tz_sparse_host.py` as
  `15 passed in 0.71 s`, and the adjacent xblock/host-policy bundle passed as
  `45 passed in 0.79 s`.
- The padded-row and compact-CSR JAX triangular-factor apply kernels are now
  owned by `solvers/explicit_sparse.py`; the helper-only
  `solvers/sparse_triangular.py` file was deleted and internal imports were
  moved to the explicit sparse owner. Focused validation passed:
  `tests/test_sparse_triangular.py tests/test_explicit_sparse.py
  tests/test_v3_sparse_pattern.py tests/test_domain_package_import_contracts.py
  tests/test_source_tree_consolidation.py` as `253 passed in 133.66 s`; the
  direct kernel test passed after lint cleanup as `4 passed in 0.44 s`.
- The RHSMode=1 structured full-CSR sparse-bundle adapter is now owned by
  `operators/profile_full_system.py`, next to the analytic full-CSR selector it
  wraps; the helper-only `operators/profile_structured_csr.py` file was
  deleted. Focused validation passed:
  `tests/test_rhs1_full_assembly.py tests/test_v3_sparse_pattern.py
  tests/test_structured_csr_docs.py tests/test_source_tree_consolidation.py` as
  `284 passed in 164.86 s`.
- Transport parallel-scaling coverage now includes pure audit gates for
  plan-only artifacts, single-case sharded-solve non-release scope,
  multi-GPU independent-case throughput, cold/release fail-closed behavior, and
  deterministic-output digest requirements. Focused validation passed:
  `tests/test_transport_policy_coverage.py` as `18 passed in 0.45 s`; the
  adjacent parallel runtime/validation bundle passed as `49 passed in 0.42 s`.
- Collisionality scan math for Simakov-Helander and FP/PAS validation artifacts
  is now owned directly by `validation/artifacts.py`; the helper-only
  `validation/math.py` file was deleted and tests now import the canonical
  validation artifact owner. Focused validation passed:
  `tests/test_validation_math.py tests/test_validation_artifacts.py
  tests/test_validation_policy_coverage.py tests/test_research_lane_policy.py`
  as `42 passed in 0.23 s`, and the source/import structure guard passed as
  `30 passed in 2.87 s`.
- Release-hosted equilibrium fixture tests now cover cache-miss fetch by known
  basename, corrupt-cache rejection without network access, failed-fetch
  no-file behavior, missing/size/hash verifier failures, successful local
  archive download with checksum admission, and offline cache-missing errors.
  This keeps large VMEC/Boozer fixtures out of the repository while testing the
  data path used by docs, examples, and CI. Focused validation passed:
  `tests/test_data_fetch.py` as `13 passed in 0.06 s`, with the
  source/import structure guard passing as `30 passed in 3.20 s`.
- Fortran-v3 active indexing is now owned by `discretization/v3.py::V3Indexing`
  alongside grids and geometry loading; the helper-only
  `discretization/indices.py` file was deleted, and parity examples/tests now
  import the canonical SFINCS-v3 discretization owner. Focused validation
  passed: operator/indexing parity plus source/import guards as
  `49 passed in 9.12 s`.
- The examples tree has been re-audited for navigation and repository size:
  every top-level task folder has a README, examples contract tests passed as
  `26 passed in 20.66 s`, and the nested `output/`, `artifacts/`,
  `provenance/`, and `reference/` folders contain only small checked JSON
  summaries used by tests/docs. The performance and publication-figure READMEs
  describe these as checked summary data and stable workflows, not branch
  history.
- Phase B example cleanup removed the vague tracked
  `examples/additional_examples/` learning-surface folder. Its only small
  namelist moved to `examples/data/qi_nfp2_reference.input.namelist`; benchmark
  and QI robustness tooling still labels the case `additional_examples` for
  report continuity. Focused examples and benchmark-default validation passed
  as `43 passed in 0.45 s`, and a direct scaled-suite probe confirmed the moved
  file maps to the historical case label.
- Phase B compatibility cleanup deleted the non-root one-file facades
  `operators/profile_response.py`, `problems/profile_response.py`,
  `problems/transport_matrix.py`, and `solvers/preconditioners.py`. Canonical
  flat owners are now the only supported implementation imports for those
  families. Focused source/import validation passed as `24 passed in 3.02 s`.
- Phase C workflow cleanup moved bounded QI `15x` GPU campaign-gating policy
  from the campaign-specific `workflows/qi_res15_gpu_campaign.py` module into
  the durable `validation/qi_device.py` owner, then deleted the workflow module.
  The public example script remains available, but fixed-artifact claim gates
  now live under validation rather than workflow implementation files.
- Phase C validation cleanup moved PETSc binary fixture readers from the tiny
  `validation/petsc_binary.py` module into `validation/fortran.py`, then
  deleted the helper file. Fortran-v3 execution, profiling, and frozen PETSc
  reference readers now have one canonical validation owner.
- Phase C sparse-owner cleanup consolidated the internal RHSMode-1 sparse-PC
  orchestration owner into `problems/profile_sparse_solve.py`. Deleted sparse
  replay filenames are guarded absent by source-tree tests, and docs/tests now
  point at the canonical solve owner.
- The upstream postprocessing workflow helper has been consolidated into
  `workflows/scans.py`, which now owns Er-scan execution and upstream
  `utils/` postprocessing wrappers used by the CLI and examples. The separate
  `workflows/postprocess_upstream.py` implementation file was deleted. Focused
  validation passed:
  `tests/test_validation_petsc_and_upstream_helpers.py`,
  `tests/test_upstream_scanplot2_smoke.py`,
  `tests/test_domain_package_import_contracts.py`, and
  `tests/test_source_tree_consolidation.py` as `39 passed in 9.60 s`.
- The root README runtime/memory summary no longer carries branch-history or
  benchmark-process phrasing; detailed audit and regeneration procedures belong
  in the performance, parity, and Fortran-example docs.
- The public stale-wording scan is clean for README, source-layout README,
  examples README, non-historical docs, and user-facing example prose. The
  archived NTX handoff page uses standalone solver-policy wording rather than
  progress-log phrasing; focused examples/benchmark docs validation passed as
  `13 passed in 0.11 s`.
- The CI coverage floor remains lower than the final target until measured
  margin is available; the review target is `95%` meaningful package coverage
  while keeping GitHub Actions under 10 minutes.

The largest coverage blockers from the fresh audit are:

- `problems/transport_parallel_runtime.py`: `56%`, 797 missing lines.
- `solvers/preconditioner_transport_matrix.py`: `57%`, 742 missing lines.
- `outputs/writer.py`: `74%`, 491 missing lines.
- `problems/profile_solve.py`: `68%`, 318 missing lines.
- `solvers/explicit_sparse.py`: `88%`, 288 missing lines.
- `operators/profile_full_system.py`: `85%`, 286 missing lines.
- `problems/profile_policies.py`: `89%`, 280 missing lines.
- `problems/transport_solve.py`: `68%`, 271 missing lines. The remaining
  uncovered code is concentrated in retry/rescue branches that should be
  covered through bounded branch-specific tests or kept as honest
  production-only paths.
- `operators/profile_system.py`: `78%`, 255 missing lines.
- `solvers/preconditioner_qi_corrections.py`: `88%`, 247 missing lines.
- `operators/profile_true_operator_rescue.py`: `82%`, 247 missing lines.
- `solvers/preconditioner_xblock_tz_sparse.py`: `77%`, 238 missing lines.
- `solvers/preconditioner_qi_device.py`: `89%`, 235 missing lines.
- `problems/profile_sparse_solve.py`: `86%`, 226 missing lines.
- `problems/transport_linear_system.py`: `84%`, 225 missing lines.
- `solvers/preconditioner_qi_basis.py`: `89%`, 190 missing lines.
- `solvers/preconditioner_schur_profile.py`: `84%`, 176 missing lines.
- `problems/profile_dense.py`: `87%`, 162 missing lines.
- `solver.py`: `88%`, 156 missing lines.
- `solvers/preconditioner_xblock_active.py`: `83%`, 154 missing lines.
- `problems/profile_sparse_qi.py`: `89%`, 153 missing lines.

## Source Structure Rules

Keep the codebase small and navigable:

- No nested implementation packages below `sfincs_jax/`.
- No new package-root modules unless they are public APIs and are documented in
  `sfincs_jax/README.md`.
- No helper-only implementation files. A new source file is allowed only if the
  same commit deletes at least two worse files or replaces a historical owner
  with a durable domain owner.
- Prefer fewer durable files over many small extraction files. File names must
  describe domain ownership, not refactor history.
- Do not create new `v3_*`, `rhs1_*`, `*_handoff`, `*_promotion`, or
  campaign-specific implementation modules.
- Do not move code if the move requires a broad import-cycle workaround, a file
  shim that only forwards imports, or a compatibility layer larger than the code
  being moved.
- Keep internal imports on canonical domain owners. Compatibility imports are
  tested separately and should not be used by implementation code.
- Keep generated outputs, traces, caches, large equilibria, and benchmark dumps
  out of the repository. Large validation data belong in release assets fetched
  by `sfincs_jax.validation.data_fetch`.

## Final Consolidation Target

This section is the authoritative remaining refactor plan. It supersedes
open-ended "move one helper at a time" tranches. The package already satisfies
the folder-depth goal; the remaining work is to reduce file count, duplicated
owners, compatibility shims, and example-tree ambiguity without changing
physics, solver defaults, outputs, or differentiability contracts.

Current source inventory:

- Root package: `16` Python files and `8585` lines. This is acceptable because
  the files are public entry points, stable helper APIs, or documented
  compatibility facades such as `io.py`.
- Domain folders: `discretization/` (`6` files), `geometry/` (`5`),
  `operators/` (`16`), `outputs/` (`5`), `physics/` (`3`),
  `problems/` (`25`), `solvers/` (`34`), `validation/` (`6`), and
  `workflows/` (`4`).
- The complexity hotspots are not nested folders; they are many
  same-family files in `problems/`, `solvers/`, and `operators/`.
  `problems/` and `solvers/` together own more than two thirds of the package
  lines and most of the remaining coverage gaps.

Target source shape for the review PR:

- Keep the existing one-level domain folders. Do not add deeper packages.
- Keep all root public modules listed in
  `tests/fixtures/source_tree_expected.json`; delete or hide compatibility
  facades only when docs, examples, scripts, and compatibility tests have moved
  to canonical imports.
- Reduce `problems/` from the current `25` files toward `16-20` durable files by merging
  setup, policy, diagnostics, sparse replay, and finalization helpers into
  coherent problem owners. The final names should describe physics/problem
  ownership: `profile_*`, `transport_*`, and `ambipolar.py` are acceptable;
  campaign, handoff, promotion, and temporary solver-path names are not.
- Reduce `solvers/` from the current `34` files toward `24-30` durable files by merging
  same-family preconditioners. Use physics/numerics families as the naming
  boundary: PAS, full-FP, x-block, QI, symbolic/native sparse factors,
  transport-matrix, Krylov dispatch, memory model, and path selection.
- Reduce `operators/` from `21` files toward `14-17` files by merging
  profile-response term files only when the move keeps clear term-level test
  seams. Operator files should represent drift-kinetic terms, sparse layouts,
  full-system assembly, and true-operator actions, not historical extraction
  phases.
- Keep `outputs/` small. `writer.py` should keep shrinking, but the durable
  owners are `writer.py`, `formats.py`, `rhsmode1.py`, `transport.py`, and
  output-schema helpers. Do not create one output file per diagnostic.
- Move or delete campaign-specific workflow modules once their behavior is
  represented by public examples, docs, tests, or validation owners.
  `workflows/` should contain reusable workflows, not historical validation
  campaigns. The bounded QI `15x` GPU campaign gate has been moved to
  `validation/qi_device.py`.

Files and families to review in order:

1. Compatibility facades:
   `operators/profile_response.py`, `problems/profile_response.py`,
   `problems/transport_matrix.py`, `solvers/preconditioners.py`, and
   `v3_driver.py` have been deleted. Keep import-contract tests in place so
   these aliases are not reintroduced.
2. RHSMode-1 profile problem helpers:
   `profile_sparse_solve.py`, `profile_sparse_direct.py`,
   `profile_sparse_fortran_reduced.py`, `profile_sparse_finalization.py`,
   `profile_sparse_policy.py`, `profile_sparse_qi.py`, and
   `profile_sparse_xblock.py`. Consolidate by role into sparse setup,
   sparse solve/rescue, and x-block/QI production owners. Do not keep
   historical transfer-state names in final implementation files.
3. Solver preconditioner families:
   merge `preconditioner_pas_*`, `preconditioner_full_fp_*`,
   `preconditioner_xblock_*`, `preconditioner_qi_*`, and
   `preconditioner_symbolic_*` only at family boundaries. A merge is accepted
   only if imports become simpler, tests move to the canonical owner, and file
   count drops without hiding unrelated algorithms in a grab-bag file.
4. Transport problem helpers:
   keep `transport_solve.py` as orchestration, `transport_linear_system.py` as
   solve dispatch, `transport_setup.py` as setup/cache/recycle policy, and
   `transport_finalize.py`/`transport_diagnostics.py` as output diagnostics.
   Delete compatibility wrappers after updating tests to canonical owners.
5. Operator term files:
   merge only if the resulting file still maps cleanly to the drift-kinetic
   equation terms documented in the physics docs. Do not merge merely to lower
   file count if it makes equations harder to find.

Consolidation gates:

- Each consolidation commit must delete at least one implementation file or
  remove at least `150` net source lines, unless it is a test-only
  compatibility-removal step.
- No commit may add a new implementation file unless it deletes at least two
  older implementation files in the same family.
- Internal imports must point to canonical owners. Compatibility imports may
  exist only in documented facades and must be covered by explicit tests.
- Every moved public or semi-public function needs either a docstring in the
  canonical owner or a nearby comment explaining the physics/numerical role.
- The source-tree fixture and `sfincs_jax/README.md` must be updated in the
  same commit as any source-shape change.
- Use full-regression tests only after several focused tranches. Avoid waiting
  on CI after every small edit; check GitHub Actions after meaningful bundles
  or before review.

Target examples shape for the review PR:

- Keep `examples/tutorials/` as the first stop: five notebooks plus the fast
  script are the classroom/user learning path.
- Keep task folders: `getting_started/`, `transport/`, `autodiff/`,
  `optimization/`, `vmec_jax_finite_beta/`, `parity/`, `performance/`,
  `publication_figures/`, `data/`, and `utils/`.
- Treat `examples/sfincs_examples/` and `examples/upstream/` as archived
  reference fixtures, not first-run examples. Their READMEs must say this
  clearly, and tutorial/example navigation must not send new users there
  first.
- The former `additional_examples/` input belongs in `data/` as
  `qi_nfp2_reference.input.namelist`; keep benchmark labels in reports, not as
  a top-level examples folder.
- For each public task folder, keep one "start here" script and one README
  table that maps user goals to scripts. Move specialized promotion or audit
  scripts to `publication_figures/`, `performance/`, or `scripts/` if they are
  not learning examples.
- Notebook cells should be pedagogical: short text, a runnable code cell, a
  plotted or printed diagnostic, and a note explaining what physics or
  numerical contract was checked.

Root modules retained for this PR:

- `api.py`, `cli.py`, `__main__.py`, and `__init__.py` for user entry points.
- `solver.py`, `ambipolar.py`, and `sensitivity.py` for public solve,
  ambipolar, and differentiability APIs.
- `plotting.py`, `compare.py`, `io.py`, `namelist.py`, `input_compat.py`, and
  `paths.py` for user-facing plotting, comparison, I/O, input, and path
  utilities.
- `diagnostics.py`, `grids.py`, and `profiling.py` for stable scientific and
  support APIs used by examples, docs, tests, and benchmark tooling.

Compatibility facades for the former monolithic driver, non-root
profile-response imports, transport-matrix helpers, and preconditioner families
have been deleted; public docs, examples, scripts, and tests should import
canonical flat owners directly.

## Open Lanes And Status

### Lane 1 - Review-Ready Refactor

Status: 99% for final review readiness.

Goal: finish the PR with a smaller, clearer source tree without changing
physics, outputs, tolerances, solver defaults, differentiable Python paths,
non-autodiff CLI fast paths, CPU/GPU behavior, or parity gates.

Latest AST audit:

- Folder depth is no longer the blocker: the package has one-level domain
  folders only and no `__init__.py`-only source packages.
- The source tree has 120 Python files, 16 package-root modules, and one-level
  domain folders only. The remaining structural blockers are file-family sprawl
  and owner size.
  The largest retained owners are `problems/profile_policies.py` (`7916`
  lines), `problems/profile_sparse_xblock.py` (`7727` lines),
  `operators/profile_full_system.py` (`6133` lines),
  `problems/profile_sparse_solve.py` (`5168` lines),
  `solvers/preconditioner_qi_device.py` (`5433` lines),
  `solvers/explicit_sparse.py` (`5198` lines),
  `problems/profile_sparse_qi.py` (`4873` lines),
  `problems/profile_solve.py` (`4351` lines), and
  `outputs/writer.py` (`2471` lines).
- The final consolidation pass should reduce file count and improve ownership
  before further line-by-line extraction. A patch that only moves a few
  functions but leaves the same number of files is not sufficient unless it
  closes a coverage or compatibility gap.
- The active-DOF/PAS-projection reduced-system setup has been extracted from
  the driver into `problems/profile_setup.py`. This is a safe first reduction
  and gives the solver driver a tested setup seam, but it is not enough by
  itself to make the driver review-sized.
- The RHSMode=1 route/preconditioner-selection setup has been extracted from
  the driver into `problems/profile_policies.py`. This is the second safe
  reduction and gives solver auto-selection a direct unit-test seam.
- Profile linear-solve dispatch setup has been extracted from the driver into
  `problems/profile_dense.py`, and structured f-block preconditioner metadata
  recording has been extracted into `problems/profile_diagnostics.py`.
- The RHSMode=1 output solve-method selection has been extracted from the HDF5
  writer into `outputs/rhsmode1.py`. This removes policy branching from the
  writer while preserving the legacy private `sfincs_jax.io` selector alias.
- RHSMode=1 output correction helpers have been extracted from the HDF5 writer
  into `outputs/rhsmode1.py`, leaving those optional parity/debug corrections
  unit-testable without running an output solve.
- RHSMode=1 output schema writes for vm-only diagnostics, Phi1 scalar fields,
  electric-drift fluxes, and derived flux totals have been extracted from the
  HDF5 writer into `outputs/rhsmode1.py`, giving the HDF5 schema a direct,
  fast regression seam.
- RHSMode=1 per-iteration classical flux output has been extracted from the
  HDF5 writer into `outputs/rhsmode1.py`, keeping the Fortran-style progress
  printout arrays available while giving the classical Phi1/no-Phi1 branches
  direct tests.
- RHSMode=1 NTV diagnostic recomputation has been extracted from the HDF5
  writer into `outputs/rhsmode1.py`, keeping the Fortran v3 `NTVKernel`
  convention and adding direct zero/non-axisymmetric branch tests.
- Final solver-trace sidecar assembly has been extracted from the HDF5 writer
  into `outputs/rhsmode1.py`, keeping solver provenance, residual targets,
  per-RHS transport diagnostics, and memory estimates on the RHSMode-aware
  output owner while leaving `outputs/writer.py` as orchestration.
- Scalar namelist conversion and Fortran logical encoding for output datasets
  are owned by `outputs/formats.py`; `outputs/writer.py` and
  `outputs/rhsmode1.py` keep compatibility aliases but no duplicate
  implementations.
- Strict numeric HDF5 parity has been folded into the root public
  `compare.py` API and the `validation/h5_parity.py` helper has been deleted.
  Scripts and tests now use one public comparison owner instead of a
  validation-only facade.
- Measured solver-candidate admission gates have been folded into
  `solvers/path_policy.py` and the separate `solvers/selection_policy.py`
  module has been deleted. Solver route policy, rescue/JIT policy, and
  measured candidate promotion rules now share one policy owner.
- QI production-ladder promotion gates have moved from the solver package into
  `validation/qi_device.py`, and `solvers/preconditioner_qi_policy.py` has
  been deleted. QI evidence policy now lives with QI artifact validation rather
  than numerical preconditioner implementations.
- The next consolidation pass must reduce those owner sizes using existing
  domain files. Do not add more package folders or helper-only files.

Completed work:

- Tranche 1: active reduced-system setup extraction into
  `problems/profile_setup.py`.
- Tranche 2: route/preconditioner-selection setup extraction into
  `problems/profile_policies.py`.
- Tranche 3: profile-solve consolidation of active reduced-system
  setup, preconditioner-route setup, linear-solve dispatch setup, and
  structured f-block metadata recording into existing owners. Continue this
  lane only when a complete solve/setup phase can move to an existing owner
  without adding helper-only files.
- Tranche 4: RHSMode=1 output solve-method selection extraction into
  `outputs/rhsmode1.py`.
- Tranche 5: RHSMode=1 output correction helper extraction into
  `outputs/rhsmode1.py`.
- Tranche 6: RHSMode=1 core diagnostic, Phi1 scalar, and
  electric-drift output schema extraction into `outputs/rhsmode1.py`. This
  phase dropped `write_sfincs_jax_output_h5` below 2000 lines and added direct
  HDF5-schema tests, but it was intentionally smaller than the earlier 200-line
  target because the remaining electric-drift computation is still entangled
  with operator internals.
- Tranche 7: RHSMode=1 classical flux output extraction into
  `outputs/rhsmode1.py`, covering both no-Phi1 and Phi1-history branches with
  tiny real-formula tests.
- Tranche 8: RHSMode=1 NTV diagnostic recomputation extraction into
  `outputs/rhsmode1.py`, covering geometryScheme=5 and non-axisymmetric L=2
  paths with tiny numerical tests.
- Tranche 9: moved the complete `export_f` output-grid mapping and
  distribution projection phase into existing `outputs/formats.py`, then
  centralized the RHSMode=1 and transport state-vector `delta_f`/`full_f`
  writes in the same owner. `outputs/writer.py` is down to `3250` lines, the
  compatibility aliases used by `sfincs_jax.io` are preserved, and the
  export/HDF5/output-policy tests pass.
- Tranche 10: moved the complete RHSMode=1 full-FP post-Krylov
  residual-polish ladder from `problems/profile_solve.py` into existing
  `problems/profile_residual.py`, keeping policy parsing in the driver and
  preserving all stage gates through explicit callback injection. This lowered
  the profile driver to `4402` lines and added bounded residual-stage tests.
- Tranche 11: moved the RHSMode=2/3 transport linear solve context,
  callbacks, solve-kind policy, and direct/JIT/implicit residual-returning
  dispatch helpers into existing `problems/transport_linear_system.py`. This
  lowered `problems/transport_solve.py` to `2949` lines while keeping
  compatibility imports from the transport solve module.
- Tranche 12: moved RHSMode=2/3 dense-LU fallback construction and
  host SciPy GMRES first-attempt support into existing
  `problems/transport_linear_system.py`. This lowered
  `problems/transport_solve.py` to `2811` lines while preserving public
  compatibility imports and existing dense/host transport tests.
- Tranche 13: moved RHSMode=2/3 dense batched whichRHS solve support
  into existing `problems/transport_linear_system.py`. This lowered
  `problems/transport_solve.py` to `2589` lines while preserving the public
  dense-batch API from `transport_solve.py` and updating tests to patch the
  canonical owner.
- Tranche 14: moved RHSMode=2/3 loop-local matvec/recycle/progress
  setup helpers into existing `problems/transport_setup.py`. This lowered
  `problems/transport_solve.py` to `2326` lines while preserving public
  compatibility imports and existing loop-support tests.
- Tranche 15: consolidated the RHSMode=1 sparse-PC orchestration owner into
  `profile_sparse_solve.py`, updated internal imports, docs, API references,
  and import contracts, and added a source-tree guard so deleted sparse replay
  filenames are not reintroduced.
- Tranche 16: moved strict numeric HDF5 parity from
  `validation/h5_parity.py` into the root public `compare.py` API, deleted the
  validation helper, and updated docs/scripts/tests to use one comparison
  owner.
- Tranche 17: merged measured solver-candidate admission gates from
  `solvers/selection_policy.py` into `solvers/path_policy.py`, deleted the
  separate selection module, and kept residual/runtime/memory promotion tests
  on the canonical policy owner.
- Tranche 18: moved QI production-ladder promotion gates from
  `solvers/preconditioner_qi_policy.py` into `validation/qi_device.py`, deleted
  the solver-policy file, and documented that QI evidence gates are validation
  policy rather than numerical preconditioners.
- Tranche 19: moved geometry-output cache helpers into existing
  `outputs/formats.py` and deleted helper-only `outputs/cache.py`. This keeps
  HDF5/NetCDF/NPZ schemas, output-file dispatch, and output-cache persistence
  in one durable output-format owner, reduces `outputs/` to `5` source files,
  and preserves the writer compatibility aliases used by `sfincs_jax.io`.
- Tranche 20: moved the RHSMode=1 preconditioner dispatch builder bundle and
  kind-to-builder ladder into existing `solvers/preconditioning.py`, deleted
  helper-only `solvers/preconditioner_dispatch.py`, and updated API/docs/tests
  to the canonical preconditioning owner. This reduces `solvers/` to `35`
  source files without changing solver defaults or public CLI behavior.
- Tranche 21: moved CI-fast research-lane manifest validation into existing
  `validation/artifacts.py`, deleted helper-only `validation/research_lanes.py`,
  and updated the checker script/tests to the canonical validation-artifact
  owner. This reduces `validation/` to `6` source files while preserving the
  release-lane evidence gate.
- Tranche 22: moved RHSMode=1 full-FP species-block and species-by-``(x,L)``
  preconditioners into existing `solvers/preconditioner_full_fp_kinetic.py`,
  deleted helper-only `solvers/preconditioner_full_fp_species.py`, and updated
  profile solve/setup imports, focused species-block tests, and the source map
  to the canonical full-FP kinetic owner. This reduces `solvers/` to `34`
  source files while preserving the same block inverse algorithms and caches.
- Tranche 23: moved matrix-free residual/JVP wrappers from
  `operators/profile_linear_systems.py` into existing
  `operators/profile_system.py`, deleted the standalone residual wrapper
  module, and updated the autodiff example, residual tests, API docs, numerics
  docs, physics-reference links, source map, and source-tree guard. This
  reduces `operators/` to `18` source files while keeping the differentiable
  residual API under the profile-system operator owner.
- Tranche 24: moved constraint-source moment kernels and the
  constraintScheme=1 x-block moment-Schur wrapper from
  `operators/profile_sources.py` into existing `operators/profile_system.py`,
  deleted the standalone source-helper module, and updated profile solve,
  reduced-tail assembly, true-operator rescue, Schur preconditioner imports,
  constraint-source tests, API docs, release notes, source map, and
  source-tree guard. This reduces `operators/` to `17` source files while
  keeping source/constraint equations beside the profile-system layout they
  act on.
- Tranche 25: moved Fortran-style compressed RHSMode=1 pitch-layout metadata
  from `operators/profile_compressed_layout.py` into existing
  `operators/profile_layout.py`, deleted the standalone compressed-layout
  module, and updated transport, Phi1 Newton, ambipolar, full-system,
  symbolic-preconditioner imports, compressed-layout tests, reduced-Pmat plan
  tests, source map, and source-tree guard. This reduces `operators/` to `16`
  source files while keeping full, active, field-split, and compressed layout
  concepts in one owner.
- Tranche 26: expanded `sfincs_jax/README.md` with a guarded implementation
  owner map for the consolidated operator, problem, solver, output, and
  validation files. The source-tree guard now checks that the README names the
  canonical owners for profile-system residual/source kernels, profile layouts,
  preconditioning dispatch, output formats, and validation artifacts, and that
  helper-only modules are not treated as valid navigation targets.
- Tranche 27: folded the GPU transport subprocess worker CLI into
  `problems/transport_parallel_runtime.py`, deleted
  `problems/transport_parallel_worker.py`, updated the internal subprocess
  launch path to `python -m sfincs_jax.problems.transport_parallel_runtime`,
  and added a source-tree guard so transport parallelism has one runtime owner.
  This reduces `problems/` to `25` source files without changing the worker
  payload schema, merge-ready NPZ output, or parent-side transport solve API.
- Tranche 28: removed dead private wrappers from `problems/profile_solve.py`
  that forwarded to canonical Krylov dispatch and structured f-block cache-key
  owners but were no longer called by production code. Tests now exercise those
  behaviors through `solvers/krylov_dispatch.py` and
  `solvers/preconditioning.py`, while `profile_solve.py` keeps only live
  orchestration wrappers. This also deleted an unused duplicate elapsed-time
  helper in `problems/profile_sparse_xblock.py`.
- Tranche 29: ran a private-symbol audit and removed unreferenced helpers from
  `operators/profile_drifts.py`, `physics/collisions.py`, and
  `problems/profile_diagnostics.py`. The retained single-reference private
  helpers are either directly tested reference implementations or used by live
  progress/label formatting paths.
- Tranche 30: added direct bounded coverage for the
  `rhs1_gpu_sparse_fallback_skip_allowed_current_backend` policy wrapper and
  reran a source/test audit showing no production-used public helpers in
  `problems/profile_policies.py` without direct tests.
- Tranche 31: added direct bounded coverage for
  `problems/profile_preconditioner_build.py` threshold readers and strong
  preconditioner builder functions, then reran the source/test audit showing no
  production-used public helpers in that module without direct tests.
- Tranche 32: added direct bounded coverage for
  `problems/transport_linear_system.py` active-DOF selection and FP transport
  preconditioner builder fallback contracts, then reran the source/test audit
  showing no production-used public helpers in that module without direct
  tests.
- Tranche 33: added a direct bounded identity-system test for the public
  `problems/profile_dense.py::solve_profile_linear` entry point, then reran the
  source/test audit showing no production-used public helpers in that module
  without direct tests.
- Tranche 34: added a direct bounded disabled-branch/context-schema test for
  `problems/profile_sparse_xblock.py::run_xblock_sparse_pc_branch`, then reran
  the source/test audit showing no production-used public helpers in that
  module without direct tests.
- Tranche 35: added direct bounded sparse-solve orchestration tests for
  `problems/profile_sparse_solve.py`, covering dispatch no-op, preflight
  residual math, residual candidate acceptance, no-op rescue stages, x-block
  backend deferral, direct-tail fallback/policy setup, and the x-block backend
  capability guard. The source/test audit now shows no production-used public
  helpers in that module without direct tests.
- Tranche 36: added direct numerics tests for mapped-grid differentiation,
  SFINCS-v3 x-grid weight formulas, Maxwellian speed-moment helpers, and memory
  dtype byte accounting, then reran source/test audits showing no
  production-used public helpers without direct tests in the targeted
  discretization/workflow/memory modules.
- Tranche 37: added direct geometry/output tests for Boozer `.bc`
  header/bracketing/effective-radius selection, output cache-key construction,
  and direct NPZ/NetCDF writer behavior, then reran source/test audits showing
  no production-used public helpers without direct tests in the targeted
  geometry/output modules.
- Tranche 38: added direct transport-runtime, profile-setup, sparse-direct,
  and sparse diagnostics wrapper tests, then reran source/test audits showing
  no production-used public helpers without direct tests in those targeted
  modules.
- Tranche 39: added direct structured-velocity, transport-policy, sensitivity,
  and validation API tests, then reran source/test audits showing no
  production-used public helpers without direct tests in those targeted
  modules.
- Tranche 40: added direct PAS-policy, symbolic sparse-profile, and x-block
  policy parser/memory tests, then reran source/test audits showing no
  production-used public helpers without direct tests in those targeted
  modules.
- Tranche 41: added direct public Krylov wrapper tests for BiCGStab, TFQMR,
  and distributed-GMRES fallback APIs, then reran the source/test audit showing
  no production-used public helpers in `solver.py` without direct tests.
- Tranche 41b: added bounded distributed-GMRES mesh-present fallback and
  fake-sharded pad/trim tests so distributed wrapper behavior is checked without
  requiring multi-device hardware in normal CI.
- Tranche 42: added direct native-factor, QI global-basis, sparse cache-key,
  host sparse-factor, and bounded x-block-TZ builder tests, then reran
  source/test audits showing no production-used public helpers without direct
  tests in those targeted modules.
- Tranche 43: added direct profile f-block from-namelist constructor and
  f-block layout-adapter tests, then reran source/test audits showing no
  production-used public helpers without direct tests in those targeted
  operator modules.
- Tranche 44: added direct optimization proxy/comparison and unsharded
  submatrix-probe wrapper tests, then reran source/test audits showing no
  production-used public helpers without direct tests in those targeted
  workflow/preconditioning modules.
- Tranche 45: added direct ambipolar derivative/Brent-wrapper and sparse-PC
  finalization retry-wrapper tests, then reran source/test audits showing no
  production-used public helpers without direct tests in those targeted problem
  modules.
- Tranche 46: moved representative structured full-FP f-block preconditioner
  tests to direct canonical builder calls, then reran the source/test audit
  showing no production-used public helpers in that solver module without
  direct tests.
- Tranche 47: added direct active x-block builder tests for global Schur,
  multiline field split, bounded native stack, Fortran-v3 reduced native stack,
  diagonal Schur, x-ell kinetic-line, and angular-line entry points, then reran
  the direct-reference audit showing no remaining missing functions in that
  audit slice.
- Tranche 48: added a direct default-disabled QI sparse-pipeline orchestration
  test for `run_xblock_qi_preconditioner_pipeline`, then reran the
  direct-reference audit showing the QI pipeline wrapper is covered without
  adding a slow solve.
- Tranche 49: added bounded public-contract tests for optimization evidence
  plans, Fortran-scan orchestration, benchmark-artifact validation, solver
  profile parsers, sparse/device metadata, mapped-grid evidence records, VMEC
  interpolation records, PAS/QI policy containers, and RHSMode-1 symbolic
  sparse metadata. The direct-reference audit now reports
  `modules_with_missing 0`; the measured coverage floor still needs the next
  branch/line coverage run before it can be raised.
- Tranche 50: migrated behavior tests away from the `sfincs_jax.v3_driver`
  compatibility shim and removed compatibility-only private-alias assertions.
  The source-tree guard now expects zero test-suite imports of `v3_driver`;
  the explicit domain import-contract test remains the single shim guard.
  Focused validation passed as `100 passed in 14.08 s`; Ruff and
  `git diff --check` passed.
- Tranche 51: deleted the final root `sfincs_jax/v3_driver.py` shim and updated
  the source-tree fixture, source-tree guard, import-contract guard, package
  README, source-map docs, and this plan so `sfincs_jax.v3_driver` is treated
  as a deleted root alias. Canonical profile/transport problem owners are now
  the only implementation import paths for those solve APIs.
- Tranche 52: narrowed `problems/profile_sparse_solve.py` from a dynamic
  cross-owner sparse namespace to an owned orchestration export surface. Its
  `__all__` now advertises only local sparse-solve orchestration and diagnostic
  symbols. Focused validation passed as
  `23 passed in 0.70 s`; Ruff, `git diff --check`, and an export sanity probe
  passed.
- Tranche 53: removed the sparse-solve transitional import waiver and migrated
  the large sparse-PC behavior test plus `profile_solve.py` to direct canonical
  sparse owners (`profile_sparse_direct`, `profile_sparse_finalization`,
  `profile_sparse_fortran_reduced`, `profile_sparse_policy`,
  `profile_sparse_qi`, and `profile_sparse_xblock`). The x-block public export
  list now includes the branch context/runner it owns, and import-contract
  tests now guard that the broad sparse-solve compatibility namespace is not
  required. Focused validation passed as `361 passed in 2.97 s`; Ruff and
  `git diff --check` passed.
- Tranche 54: synchronized active roadmap and feature-matrix docs with the
  deleted `v3_driver.py` monolith. Current docs now describe flat canonical
  profile/transport owners and the review-gated architecture state instead of
  active legacy driver shims. Lightweight docs/source guards passed as
  `49 passed in 3.03 s`; `git diff --check` passed.
- Tranche 55: fixed stale validation/research manifest source paths left by
  deleting `v3_driver.py`, updated active testing/research-lane prose, and
  cleaned source docstrings that described the retired file as an active
  integration point. Focused release/research/docs/source guards passed as
  `70 passed in 3.20 s`; Ruff and `git diff --check` passed.
- Tranche 56: narrowed the CI slow-test skip from every
  `test_transport_parallel*` node to only the genuinely slow
  `tests/test_transport_parallel.py` integration file. The fast transport
  parallel runtime, payload, sharding, solve, validation, worker-CLI, and
  artifact tests now run under `SFINCS_JAX_CI=1`, recovering coverage for the
  largest current blocker without duplicating tests. CI-mode validation for
  the transport-parallel group passed as `85 passed, 18 skipped in 0.78 s`;
  Ruff and `git diff --check` passed.
- Tranche 57: moved Phi1 Newton-step solve-policy helpers and Phi1 output
  history alignment from `outputs/writer.py` into existing
  `outputs/rhsmode1.py`. The writer now keeps compatibility aliases for
  legacy private imports but no longer owns that policy code. Focused
  validation passed:
  `tests/test_phi1_history_alignment.py tests/test_cli_solve_mode.py tests/test_io_output_policy_coverage.py`
  as `136 passed in 2.53 s`; Ruff passed. This reduced
  `outputs/writer.py` from `3250` to `3129` lines without adding source files.
- Tranche 58: moved SFINCS-v3 equilibrium path resolution and staged-run
  equilibrium localization from `outputs/writer.py` into existing
  `input_compat.py`, updated `io.py` and `outputs.__init__` compatibility
  exports, and converted input-compat tests to import the moved helpers from
  their canonical owner. The `io.py` facade was simplified to a loop-based
  legacy owner delegation and remains implementation-free at `74` lines.
  Focused validation passed:
  `tests/test_input_compat.py tests/test_io_output_policy_coverage.py tests/test_api_contracts.py tests/test_domain_package_import_contracts.py tests/test_source_tree_consolidation.py`
  as `164 passed in 7.54 s`; Ruff and `git diff --check` passed. This reduced
  `outputs/writer.py` from `3129` to `3040` lines without adding source files.
- Tranche 59: moved geometryScheme=4 radial normalization and
  `setInputRadialCoordinateWish` compatibility formulas from
  `outputs/writer.py` into existing `input_compat.py`, keeping legacy private
  aliases reachable through `sfincs_jax.io` while making tests import the
  public canonical input-compat names. This keeps radial-coordinate semantics
  with SFINCS-v3 input compatibility rather than output orchestration and
  reduces `outputs/writer.py` from `3040` to `2993` lines without adding source
  files.
- Tranche 60: moved Boozer and VMEC `gpsiHatpsiHat` metric reconstruction from
  `outputs/writer.py` into existing `geometry/boozer.py` and
  `geometry/vmec_wout.py`, leaving only thin writer compatibility wrappers that
  resolve namelist paths/options before calling the geometry owners. The
  geometry import-contract tests now advertise the metric helpers, and output
  policy tests patch the canonical geometry modules. Focused validation passed:
  `tests/test_io_output_policy_coverage.py tests/test_geometry_grid_helper_coverage.py tests/test_vmec_wout_conventions.py tests/test_domain_package_import_contracts.py tests/test_source_tree_consolidation.py`
  as `142 passed in 6.07 s`; Ruff passed. This reduced `outputs/writer.py`
  from `2993` to `2675` lines without adding source files.
- Tranche 61: added direct VMEC metric reconstruction tests for finite
  `gpsiHatpsiHat` output plus fail-closed gates for invalid
  `VMEC_Nyquist_option`, zero `bmnc(0,0)`, and an over-aggressive mode filter.
  The `sfincs_jax.io` facade test now also verifies that the moved Boozer
  evaluator still resolves through the legacy private facade. Focused
  validation passed:
  `tests/test_vmec_wout_conventions.py tests/test_output_formats.py tests/test_io_output_policy_coverage.py tests/test_geometry_grid_helper_coverage.py tests/test_domain_package_import_contracts.py`
  as `129 passed in 3.82 s`; Ruff and `git diff --check` passed.
- Tranche 62: removed the last package-source prose references to the deleted
  `v3_driver` architecture and added a source-tree guard that prevents package
  docstrings/comments from reintroducing it. The same tranche added bounded
  numerical tests for the symbolic host sparse-factor owner: row-nnz cap
  parsing, regularization retry defaults/fail-closed behavior, and equivalent
  dense/CSR drop-plus-diagonal-regularization paths. Focused validation passed:
  `tests/test_sparse_assembly.py tests/test_source_tree_consolidation.py`
  as `34 passed in 3.23 s`; Ruff, compileall, and `git diff --check` passed.
- Tranche 63: renamed the remaining stale test module
  `test_v3_driver_strong_fallback_coverage.py` to
  `test_rhs1_strong_preconditioner_fallback.py` and added a source-tree guard
  against reintroducing deleted-driver terminology in test filenames. The tests
  still protect the same RHSMode-1 strong-preconditioner fallback policy
  contracts; no source files or runtime paths changed.
- Tranche 64: hardened the Fortran/PETSc validation owner by making PETSc
  vector and AIJ-matrix readers reject negative dimensions and truncated binary
  payloads before NumPy creates views. This protects frozen Fortran-v3 parity
  fixtures without requiring Fortran to run in CI. Focused validation passed:
  `tests/test_validation_petsc_and_upstream_helpers.py` as `8 passed in
  0.61 s`; Ruff, compileall, and `git diff --check` passed.
- Tranche 65: extended Fortran-v3 profile-log coverage for parser fields used
  by solver and benchmark audits: D-exponent tolerances/timings, matrix and
  preconditioner nonzero counts, residual-f1 matrix counts, repeated solve
  driver timings, KSP residual history, MUMPS `INFOG` memory/factor metadata,
  and empty-log tolerance. Focused validation passed:
  `tests/test_validation_petsc_and_upstream_helpers.py` as `10 passed in
  0.62 s`; Ruff, compileall, and `git diff --check` passed.
- Tranche 66: extended validation-figure policy coverage for W7-X
  ambipolar-root publication claims. The tests now prove that the panel becomes
  literature-ready only when numerical gates, complete provenance, a matching
  JSON payload, and a Git-tracked source artifact all pass; tracked wrong-name
  and tracked payload-mismatch artifacts fail closed with explicit statuses.
  Focused validation passed: `tests/test_validation_figures.py` as `9 passed in
  0.08 s`; Ruff passed.
- Tranche 67: added strict HDF5 output-comparison contract tests for missing
  candidate datasets, missing reference datasets selected by key, shape
  mismatches, value mismatches, extra candidate datasets, nonnumeric metadata
  skips, ignore lists, and per-key tolerance overrides. These tests protect the
  CLI/parity audit path without shipping larger frozen outputs. Focused
  validation passed: `tests/test_compare_reference_corruption.py` as `20 passed
  in 0.33 s`; Ruff passed.
- Tranche 68: fixed a VMEC full-trajectory FP parity-tolerance bug in
  `compare.py` where a `for/else` indentation skipped the `totalDensity`
  tolerance unless the key already existed. Added a regression test that gates
  `totalDensity` and `particleFluxBeforeSurfaceIntegral_vm` for
  geometryScheme=5, RHSMode=1, full-FP, full-trajectory comparisons. Focused
  validation passed: `tests/test_compare_reference_corruption.py` as `21 passed
  in 0.48 s`; Ruff passed.
- Tranche 69: tightened source-tree architecture guardrails by adding an
  explicit per-domain module inventory to `source_tree_expected.json` and a
  test that fails if a domain package gains unplanned helper files. This keeps
  the refactor moving toward a small, intentional source tree instead of
  drifting back into file proliferation. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  as `38 passed in 3.27 s`; Ruff passed and the fixture remained valid JSON.
- Tranche 70: ran the documentation review gate after the refactor guard
  updates. `sphinx-build -W -b html docs docs/_build/html` completed
  successfully over all 44 documentation sources, including source-map,
  testing, examples, performance, parity, API, and release-checklist pages.
- Tranche 71: made the generated README example-suite audit block
  release-neutral. The generator now emits `EXAMPLE_SUITE_AUDIT` markers while
  accepting the older markers for one regeneration cycle, removes branch
  history wording from the public audit text, normalizes old same-resolution
  drift reasons from archived reports, keeps custom drift reasons actionable,
  and aligns the performance documentation terminology with the generated
  runtime-drift gate language. Focused validation passed:
  `tests/test_generate_readme_fast_branch_audit.py tests/test_benchmark_doc_claims.py`
  as `10 passed in 0.10 s`; `sphinx-build -W -b html docs docs/_build/html`
  passed; Ruff passed.
- Tranche 72: reduced the RHSMode-1 solve-orchestration compatibility surface
  by removing low-level domain-decomposition and diagonal-reduction helper
  aliases from `problems/profile_solve.py`. The affected tests now import those
  helpers from their canonical owners:
  `solvers/preconditioner_domain_decomposition.py`,
  `solvers/preconditioning.py`, `problems/profile_residual.py`, and
  `solver.py`. A source-tree guard prevents those accidental helper aliases
  from returning to `profile_solve.py`. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_profile_dd_reduction_coverage.py tests/test_rhs1_schwarz_heuristic.py tests/test_rhs1_domain_decomposition.py tests/test_pas_preconditioner_policy.py tests/test_profile_solve_policy_helpers.py`
  as `100 passed in 12.39 s`; Ruff passed.
- Tranche 73: moved policy-only tests off `problems/profile_solve.py` and onto
  their canonical owners. JIT/dtype tests now use
  `solvers/preconditioning.py`, RHSMode-1 route tests use
  `problems/profile_policies.py`, generic solver-path tests use
  `solvers/path_policy.py`, and transport policy tests use
  `problems/transport_policies.py`. A source-tree guard keeps those
  policy-only tests from re-widening the solve-orchestration API. Focused
  validation passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_solve_policy_helpers.py tests/test_profile_solve_policy_coverage.py`
  as `55 passed in 3.32 s`; Ruff passed.
- Tranche 74: moved sparse-helper coverage off private `profile_solve._*`
  aliases and onto canonical sparse/policy owners. The helper tests now use
  `problems/profile_sparse_direct.py`, `problems/profile_policies.py`,
  `solvers/explicit_sparse.py`, `solvers/path_policy.py`, and
  `solvers/preconditioner_xblock_tz_sparse.py` directly, while keeping the
  final solve integration test on `profile_solve.py`. A source-tree guard
  allows that high-level solve import but forbids private
  `profile_solve._*` helper assertions. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_sparse_helper_coverage.py`
  as `43 passed in 3.49 s`; Ruff passed.
- Tranche 75: moved Schur/PAS heuristic unit assertions off private
  `profile_solve._*` aliases and onto canonical policy/preconditioner owners.
  The heuristic tests now use `problems/profile_preconditioner_build.py` for
  Schur and PAS builder seams, `solvers/preconditioning.py` for shared caches
  and dtype/policy hints, `problems/profile_policies.py` for solver-route
  predicates, and `solvers/path_policy.py` for resource-exhaustion detection.
  High-level solve-driver integration remains on `profile_solve.py`, but a
  source-tree guard prevents unit assertions from drifting back to dotted
  private solve aliases. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_schur_precond_heuristic.py`
  as `51 passed in 27.59 s`; the broader refactor/owner bundle passed as
  `192 passed in 37.63 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 76: moved PAS applicability and memory-policy tests entirely onto
  `solvers/preconditioner_pas_policy.py`, removing duplicate assertions
  against private `profile_solve._*` policy aliases. The PAS builder tests
  still exercise `problems/profile_preconditioner_build.py`, but the pure
  policy tests now import no solve-orchestration module. A source-tree guard
  keeps `tests/test_pas_preconditioner_policy.py` on the PAS policy owner.
  Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_pas_preconditioner_policy.py`
  as `44 passed in 3.18 s`; the broader refactor/owner bundle passed as
  `193 passed in 37.62 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 77: moved standalone distributed-GMRES axis resolver tests from the
  profile-solve wrapper to the canonical Krylov dispatch owner,
  `solvers/krylov_dispatch.py`. Wrapper injection coverage remains in
  `tests/test_profile_solve_module_wrappers.py`, while
  `tests/test_distributed_gmres_axis.py` now directly validates the public
  resolver policy with explicit shard-axis injection. A source-tree guard keeps
  the resolver tests off `profile_solve.py`. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_distributed_gmres_axis.py`
  as `33 passed in 3.59 s`; the compact wrapper/owner bundle passed as
  `131 passed in 30.43 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 78: removed the final redundant `profile_solve.py` sparse-assembly
  alias assertions from `tests/test_sparse_assembly.py`. The tests now cover
  the RHSMode-1 FP x-block sparse assembly, host cache, and diagonal helpers
  directly through `solvers/preconditioner_xblock_tz_sparse.py`; a source-tree
  guard keeps sparse assembly tests from reintroducing solve-orchestration
  alias checks. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_sparse_assembly.py` as
  `43 passed in 3.70 s`; the compact owner bundle passed as
  `133 passed in 28.70 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 79: moved RHSMode-1 device-operator helper dependencies off
  `profile_solve.py` aliases where they are not solve-driver integration
  seams. Active-DOF selection now imports from
  `problems/transport_linear_system.py`, and the side-probe policy monkeypatch
  uses `solvers/preconditioner_xblock_policy.py`. The test still runs the
  high-level solve through `profile_solve.py`, and a source-tree guard prevents
  these two helper aliases from returning. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_rhs1_device_operator.py`
  as `35 passed in 6.18 s`; the compact owner bundle passed as
  `137 passed in 31.34 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 80: moved structured full-CSR documentation tests off the
  `profile_solve.py` compatibility alias for method names. The docs test now
  imports `STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS` from
  `problems/profile_setup.py`, where the setup policy is defined, and a
  source-tree guard keeps that docs test off solve orchestration. Focused
  validation passed:
  `tests/test_source_tree_consolidation.py tests/test_structured_csr_docs.py`
  as `38 passed in 3.27 s`; the compact owner bundle passed as
  `143 passed in 31.30 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 81: moved the remaining RHSMode-1 dispatch-coverage helper checks
  off private `profile_solve._*` aliases. DKES GMRES budget tests now use
  `solvers/path_policy.py`, and PAS-TZ guarded structured-level parsing tests
  now use `problems/profile_policies.py`. The file still imports
  `profile_solve.py` for high-level solve integration, but a source-tree guard
  prevents private solve-helper assertions from returning. Focused validation
  passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_rhs1_dispatch_coverage.py`
  as `71 passed in 42.48 s`; the broader owner/wrapper bundle passed as
  `192 passed in 70.54 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 82: added a global test-suite consolidation guard that allows
  `profile_solve._*` references only in
  `tests/test_profile_solve_module_wrappers.py` and the guard file itself.
  This locks the remaining private solve-orchestration usage to explicit
  wrapper-contract tests and prevents future unit tests from widening the
  solve module's private helper surface. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_solve_module_wrappers.py`
  as `46 passed in 3.92 s`; the broader owner/wrapper bundle passed as
  `193 passed in 76.71 s`; Ruff, `compileall`, and `git diff --check` passed.
- Tranche 83: removed duplicated top-level environment helper definitions from
  `problems/profile_policies.py`, `problems/profile_dense.py`, and
  `solvers/preconditioner_symbolic_policy.py`. This cut stale duplicate helper
  blocks from large owner files without changing policy behavior. A source-tree
  guard now scans every package module for repeated top-level function or class
  definitions so duplicate helper blocks cannot silently return. Focused
  validation passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_solve_policy_helpers.py tests/test_profile_response_linear_solve.py tests/test_profile_response_sparse_pc.py tests/test_fortran_reduced_preconditioner.py`
  as `435 passed in 24.48 s`; the broader sparse/preconditioner policy bundle
  passed as `450 passed in 20.33 s`; the broader source/policy/docs-adjacent
  bundle passed as `581 passed in 25.07 s`; Ruff, `compileall`, and
  `git diff --check` passed.
- Tranche 84: renamed the active RHSMode-1 accepted-solver-state API from
  historical transfer-state terminology to solver replay terminology:
  `RHS1KSPAcceptedCandidateState`,
  `rhs1_apply_candidate_to_replay_state`, and
  `tests/test_rhs1_solver_replay.py`. The implementation behavior is
  unchanged; the tranche removes historical naming from active solver
  diagnostics, profile dense replay prose, sparse x-block metadata prose,
  sensitivity integration prose, and source-map/performance docs. Focused
  validation passed:
  `tests/test_rhs1_solver_replay.py tests/test_profile_solve_module_wrappers.py tests/test_source_tree_consolidation.py`
  as `112 passed in 4.81 s`; the broader solver-replay/sparse-dispatch bundle
  passed as `472 passed in 48.55 s`; Ruff, `compileall`, and
  `git diff --check` passed.
- Tranche 85: removed active-source "legacy driver" / "historical driver"
  wording from profile diagnostics, solver replay diagnostics, sparse-PC
  finalization, sparse x-block metadata, RHSMode-1 policy parsing, PAS
  composite builders, shared preconditioning helpers, structured f-block docs,
  and the source-map page. The local x-block post-correction metadata callback
  is now `metadata_state()` instead of `driver_state()`, with tests updated to
  the new contract. This keeps current solver behavior unchanged while making
  the review surface describe the present architecture. Focused validation
  passed:
  `tests/test_profile_response_sparse_pc.py tests/test_profile_response_diagnostics.py tests/test_profile_solve_policy_helpers.py tests/test_profile_solve_policy_coverage.py tests/test_pas_preconditioner_policy.py tests/test_preconditioner_setup.py`
  as `421 passed in 4.22 s`; Ruff, `compileall`, and `git diff --check`
  passed.
- Tranche 86: added bounded transport-parallel policy coverage for release
  scope admission and fail-closed scaling claims. The new tests exercise
  measured independent-transport throughput scope, legacy deterministic-output
  admission for experimental sharded-solve snapshots, and warm-cache timing
  inference for multi-GPU case-throughput artifacts. This improves coverage of
  production-shape metadata gates without launching solves or adding fixtures.
  Focused validation passed:
  `tests/test_transport_parallel_execution.py tests/test_transport_policy_coverage.py tests/test_transport_parallel_sharding.py`
  as `58 passed in 0.48 s`; Ruff passed for the touched tests.
- Tranche 87: moved final solver-trace sidecar assembly from
  `outputs/writer.py` into existing `outputs/rhsmode1.py`, so the writer now
  calls `write_output_solver_trace_json` instead of owning duplicated
  provenance/residual/memory-estimate assembly. This reduced
  `outputs/writer.py` from `2675` to `2490` lines without adding source files.
  A direct trace test now covers transport-matrix selected path, metadata
  solver-method fallback, per-RHS residual/rhs norms, convergence, solver
  kind/method maps, and memory estimates. Focused validation passed:
  `tests/test_solver_trace_output_formats.py` as `12 passed in 0.35 s`; the
  broader output/CLI bundle
  `tests/test_solver_trace_output_formats.py tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py tests/test_cli_solve_mode.py`
  as `161 passed in 2.91 s`; Ruff, py_compile, and `git diff --check` passed.
- Tranche 88: centralized scalar namelist conversion and SFINCS-v3 logical
  output encoding in `outputs/formats.py`, replacing duplicate implementations
  in `outputs/writer.py` and `outputs/rhsmode1.py` with compatibility aliases.
  This reduced `outputs/writer.py` from `2490` to `2471` lines and
  `outputs/rhsmode1.py` from `2303` to `2298` lines without adding files.
  Focused validation passed:
  `tests/test_output_formats.py tests/test_io_output_policy_coverage.py tests/test_geometry_grid_helper_coverage.py tests/test_solver_trace_output_formats.py`
  as `114 passed in 4.60 s`; Ruff and py_compile passed.
- Tranche 89: removed public "handoff" wording from README/docs/example
  navigation and changed the finite-beta VMEC-JAX example contract key from a
  file-handoff label to a file-boundary label. This keeps public prose focused
  on the present workflow/interface model while leaving release-note and NTX
  archival pages untouched. The active stale-wording scan over README, docs,
  examples, and the example-contract test exited with no matches; the
  review-lock bundle
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `64 passed in 3.81 s`; the focused source/docs bundle
  `tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py tests/test_source_tree_consolidation.py`
  passed as `50 passed in 3.68 s`; Ruff, py_compile, notebook JSON parsing,
  and `git diff --check` passed.
- Tranche 90: renamed active sparse-PC/x-block finalization and diagnostic
  helpers from `driver_state` / `driver_scope` terminology to
  `solve_state` / `solve_scope` terminology. This was a behavior-preserving
  source/test API cleanup over RHSMode-1 sparse finalization owners, with no
  file moves or line-count growth. A stale-term audit over active problem
  owners and targeted tests found no remaining `driver_state`, `driver_scope`,
  `from_driver`, `driver-state`, or `driver-scope` matches. Focused validation
  passed:
  `tests/test_profile_response_sparse_pc.py tests/test_profile_response_diagnostics.py tests/test_rhs1_solver_diagnostics.py tests/test_profile_solve_module_wrappers.py`
  as `389 passed in 4.06 s`; source-tree/import guards
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `50 passed in 3.60 s`; Ruff, py_compile, and `git diff --check`
  passed.
- Tranche 91: added a source-tree regression guard that forbids
  `driver_state`, `driver_scope`, `driver-state`, `driver-scope`,
  `from_driver_state`, and `from_driver_scope` in active package sources. This
  locks the Tranche-90 cleanup to the current solve-state vocabulary and keeps
  future sparse-solve helpers from drifting back toward monolith-era naming.
  Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_profile_response_sparse_pc.py tests/test_profile_response_diagnostics.py tests/test_rhs1_solver_diagnostics.py`
  as `415 passed in 6.71 s`; Ruff, py_compile, and `git diff --check` passed.
- Tranche 92: removed the dynamic `profile_solve` global-copy dependency from
  `problems/transport_solve.py`. RHSMode=2/3 transport orchestration now imports
  canonical transport, preconditioner, sparse-direct, profiling, and policy
  owners explicitly and carries local transport wrappers only where fallback
  builder/cache injection is required. A source-tree guard prevents
  `_PROFILE_SOLVE`, dynamic `import_module("sfincs_jax.problems.profile_solve")`,
  `vars(_PROFILE_SOLVE)`, and `globals()[_name]` from returning. This tranche
  intentionally trades a larger explicit import block for a simpler dependency
  graph and removes a hidden broad namespace coupling between RHSMode=1 and
  RHSMode=2/3 solvers. Focused validation passed:
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_transport_sparse_direct.py tests/test_transport_matrix_preconditioners.py tests/test_transport_parallel_runtime.py tests/test_transport_parallel_payload.py`
  as `159 passed in 23.53 s`; the wrapper/transport bundle
  `tests/test_profile_solve_module_wrappers.py tests/test_transport_sparse_direct.py tests/test_transport_matrix_preconditioners.py tests/test_transport_parallel_runtime.py tests/test_transport_parallel_payload.py`
  passed as `118 passed in 19.94 s`; direct import of
  `sfincs_jax.problems.transport_solve` succeeded; Ruff, py_compile, and
  `git diff --check` passed.
- Tranche 93: moved transport active-DOF and RHSMode=2/3 reduced-Pmat test
  checks off `profile_solve` private aliases and onto `transport_solve` /
  `transport_linear_system` owners. RHSMode=1 compatibility checks remain on
  `profile_solve` where they intentionally test the solve-orchestration facade.
  Focused validation passed:
  `tests/test_fortran_reduced_preconditioner.py tests/test_rhs1_full_assembly.py tests/test_v3_sparse_pattern.py`
  as `293 passed in 205.62 s`; source-tree guard
  `tests/test_source_tree_consolidation.py` passed as `38 passed in 3.75 s`;
  Ruff, py_compile, and `git diff --check` passed.
- Tranche 94: added a source-tree regression guard that prevents RHSMode=2/3
  helper tests from importing transport active-DOF or reduced-Pmat builders
  through `profile_solve` aliases. The guard deliberately exempts
  `test_profile_solve_module_wrappers.py`, which is the compatibility-facade
  test by design, while requiring all other behavior tests to use the transport
  owners directly. Focused validation passed:
  `tests/test_source_tree_consolidation.py` as `39 passed in 4.61 s`; Ruff,
  py_compile, and `git diff --check` passed.
- Tranche 95: moved true-operator rescue tests in
  `tests/test_v3_sparse_pattern.py` from `profile_solve` private aliases to the
  canonical `operators/profile_true_operator_rescue.py` owner. This preserves
  the RHSMode-1 solve facade for compatibility while making the numerical
  rescue tests exercise the module that owns residual-window, active-block,
  active-residual, active-submatrix, and reusable true-action-column logic.
  Focused validation passed: `tests/test_v3_sparse_pattern.py` as
  `141 passed in 138.06 s`; `tests/test_source_tree_consolidation.py` as
  `39 passed in 4.50 s`; Ruff and py_compile passed.
- Tranche 96: moved structured RHSMode-1 full-CSR bundle and direct-tail
  structured preconditioner cache-key tests off `profile_solve` private aliases
  and onto `operators/profile_full_system.py` and `problems/profile_policies.py`.
  The remaining `profile_solve` usages in `tests/test_rhs1_full_assembly.py`
  are solve-entry checks rather than helper-owner tests. Focused validation
  passed: `tests/test_rhs1_full_assembly.py` as `123 passed in 50.27 s`;
  `tests/test_source_tree_consolidation.py` as `39 passed in 4.92 s`; Ruff and
  py_compile passed.
- Tranche 97: finished the helper-owner cleanup in
  `tests/test_v3_sparse_pattern.py` by moving sparse ILU cache/factorization
  checks to solver owners, RHSMode-1 preconditioner-operator checks to
  `solvers/preconditioning.py`, structured full-CSR checks to
  `operators/profile_full_system.py`, direct-tail policy/cache checks to
  `problems/profile_policies.py`, residual helper checks to
  `problems/profile_residual.py`, and x-block monkeypatching to the x-block
  policy module. The remaining `profile_solve_module` uses in that file are
  solve-orchestration monkeypatches or compatibility-path module arguments, not
  private helper-owner assertions. Focused validation passed:
  `tests/test_v3_sparse_pattern.py` as `141 passed in 138.04 s`;
  `tests/test_source_tree_consolidation.py` as `39 passed in 4.49 s`; Ruff and
  py_compile passed.
- Tranche 98: removed the remaining `profile_solve` private-helper dependency
  from `tests/test_fortran_reduced_preconditioner.py`. The test now imports
  RHSMode-1/RHSMode-2/3 preconditioner-operator builders from
  `solvers/preconditioning.py`, direct sparse factor construction from
  `problems/profile_sparse_direct.py`, and sparse-host solve-method
  classifications from `problems/profile_setup.py`. Focused validation passed:
  `tests/test_fortran_reduced_preconditioner.py` as `29 passed in 16.89 s`;
  `tests/test_source_tree_consolidation.py` as `39 passed in 4.61 s`; Ruff and
  py_compile passed.
- Tranche 99: ran the first post-owner-cleanup review-lock validation sweep.
  Source-layout, domain-import, examples-tree, and benchmark-claim guards passed
  as `67 passed in 4.87 s`; Sphinx `-W` passed; the public stale-wording scan
  passed after excluding generated `docs/_build` artifacts. The documented scan
  command now excludes `docs/_build/**` so generated HTML/download text from
  upstream source PDFs does not mask source-documentation regressions.
- Tranche 100: ran the bounded public-interface validation sweep after the
  owner cleanup. CLI mode/plotting/output-format/API tests passed as
  `97 passed in 2.67 s`; I/O cache/export/output-policy/precompile and
  write-output return-result tests passed as `111 passed in 3.16 s`. Ruff
  passed on the touched Python owners and test files. `sfincs_jax/io.py`
  remains a small compatibility wrapper at `73` lines, below the review-lock
  `80`-line cap.
- Tranche 101: ran the non-destructive benchmark-regeneration readiness pass.
  `materialize_production_stress_manifest.py --out-root /tmp/... --json`
  produced a 39-case manifest with 15 short-Fortran reference rows, 16 CPU/GPU
  benchmark-floor gap rows, and QI evidence covering `nfp=1,2,3,4`.
  `create_production_benchmark_inputs.py --out-root /tmp/... --clean` produced
  a temporary 39-case production input tree with large-grid cases. The
  production-manifest/doc-claim test bundle passed as `16 passed in 0.20 s`.
  `check_benchmark_artifacts.py` validated the current README-facing benchmark
  summary JSON at
  `examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`.
  The next Lane-4 step is fresh CPU/GPU/Fortran evidence generation, not script
  repair.
- Tranche 102: ran a bounded Zenodo QS bootstrap-current reference audit with
  `run_zenodo_vmec_parity_campaign.py --mode reference-only --max-cases 2`
  against the local QS-paper Zenodo tree. The report completed with
  `reference_ok=2` and read full Fortran reference HDF5 files, including
  `FSABjHat`, `FSABjHatOverRootFSAB2`, particle-flux, heat-flux, and
  `NIterations` datasets. This verifies the QA/QH parity runner inputs before
  launching fresh candidate solves.
- Tranche 103: ran a one-case Zenodo QS candidate-solve dry run. The generated
  command uses `sfincs_jax write-output --compute-solution --solve-method auto`
  and includes `--wout-path` pointing at the QA VMEC file
  `wout_new_QA_aScaling.nc`. The machine-readable report records this path as
  `solve.equilibrium_path`; the next step for this lane is an actual bounded
  candidate solve when compute budget is allocated.
- Tranche 104: tightened the examples learning surface without moving runnable
  files. `examples/README.md` and `docs/examples.rst` now include a
  one-command start table and explicit run-budget/output guidance for CLI
  output, file formats, VMEC `wout_path`, RHSMode=2/3 transport, autodiff,
  bootstrap-current/Redl, output-format timing, and frozen Fortran-v3 fixture
  checks. `tests/test_examples_tree_contract.py` now guards these entry points
  and docs labels, and `examples/tutorials/README.md` avoids release-history
  wording. Validation: `pytest tests/test_examples_tree_contract.py
  tests/test_benchmark_doc_claims.py` passed `14/14`, Ruff passed,
  `compileall` passed, `git diff --check` passed, and Sphinx `-W` passed.
- Tranche 105: added deterministic profiling coverage for the user-facing
  progress/timing layer. `tests/test_profiling.py` now verifies structured
  emission, quiet/verbosity behavior, environment-gated profiler activation,
  device-memory opt-in policy, and phase accounting for elapsed time, RSS,
  peak RSS, and device-memory metadata. Validation: `pytest
  tests/test_profiling.py` passed `5/5`, Ruff passed, `compileall` passed, and
  `git diff --check` passed. A focused pytest-cov invocation for this module
  still exited with code `134` before producing a report, so the measured
  package coverage percentage remains pinned to the latest successful full
  coverage audit until CI or a stable local coverage run updates it.
- Tranche 106: ran bounded executable examples for the review-lock examples
  lane. `examples/tutorials/run_quick_output_and_plot.py --out-dir /tmp/...`
  wrote HDF5, NetCDF, NPZ, and a PDF diagnostics panel with `125` output fields
  per data file. `examples/getting_started/build_grids_and_geometry.py` ran the
  analytic geometry path and reported the expected `5 x 7` angular grid.
  `examples/autodiff/autodiff_gradient_nu_n_residual.py` ran the residual
  gradient path and returned the frozen-fixture near-zero loss. The supporting
  example contract suite `tests/test_getting_started_examples.py
  tests/test_examples_tutorials.py tests/test_examples_tree_contract.py`
  passed `20/20` in `20.49 s`. This keeps Phase E executable evidence fresh
  without writing generated artifacts into the repository.
- Tranche 107: removed release-history wording from the getting-started
  geometry example by describing the "supported" Python API surface instead of
  the "current" surface. The file already uses a repository-root bootstrap for
  standalone execution, so it now carries a file-local `E402` Ruff waiver for
  that intentional example-script pattern. Validation: the script ran, Ruff
  passed, `compileall` passed, `git diff --check` passed, and the affected
  examples tests passed `17/17` before the waiver-only patch.
- Tranche 108: ran the Phase-G focused validation bundle after the examples and
  profiling tranches. Passed shards: source/import/examples/benchmark-doc
  guards `67/67`, CLI/output/API `97/97`, ambipolar/sensitivity/implicit-grad
  `41/41`, representative RHSMode/sparse/preconditioner policy `82/82`,
  I/O/output/artifact policy `158/158`, validation manifest/math/figure/policy
  `47/47`, generated validation/solver-path artifact tests `33/33`, and QI
  artifact-policy tests `47/47`. One I/O shard was rerun after a typo in the
  requested test filename; the corrected shard passed. This leaves the review
  bundle green except for the separate full coverage audit and fresh
  CPU/GPU/Fortran benchmark evidence generation.
- Tranche 109: expanded `tests/test_profiling.py` to cover `Timer`, resource
  fallback RSS, peak-RSS failure, and JAX device-memory parsing without solver
  imports. Focused validation passed `8/8`, Ruff passed, `compileall` passed,
  and `git diff --check` passed. Direct `coverage.py` reporting for
  `sfincs_jax/profiling.py` improved from `73%` to `98%` using
  `coverage run -m pytest tests/test_profiling.py` followed by
  `coverage report --include='sfincs_jax/profiling.py'`; the generated
  `.coverage` file was removed. Package-wide coverage remains pinned to the
  latest successful full audit until the full suite can run stably.
- Tranche 110: used PR #8 CI as the authoritative full coverage audit after the
  local serial coverage run was interrupted at `1111 passed, 2 skipped` in
  `7:37` inside a long JAX-heavy section. The CI run on commit `3cce604d`
  passed all four xdist coverage shards, examples smoke, external-data smoke,
  optional ecosystem gates, Docs, the coverage-report job, and the final
  required-job aggregator. The combined coverage report measured
  `TOTAL 69081 stmts, 7630 miss, 89%`, with the largest remaining gaps in
  `profile_solve.py`, `transport_solve.py`, `profile_system.py`, selected
  sparse/preconditioner owners, and validation/figure helpers.
- Tranche 111: expanded scan-workflow tests around user-facing Er scan behavior:
  endpoint scan-grid construction, upstream utility discovery, noninteractive
  postprocessing, missing utility/case errors, `skip_existing` reuse, and
  parallel scan orchestration with recycle-state clearing through a fake
  in-process executor. Validation: `tests/test_scans_progress_and_recycle.py`
  passed `18/18` in `0.44 s`, Ruff passed, `compileall` passed,
  `git diff --check` passed, and direct coverage for
  `sfincs_jax/workflows/scans.py` is `98%`.
- Tranche 112: expanded validation-figure tests around artifact-backed physics
  gates without adding large data: W7-X malformed scans and ambipolar roots,
  degenerate zero-current brackets, provenance/metadata fallback behavior,
  Simakov-Helander malformed high-nu scans, invalid tail-fit configuration,
  checked/tracked/mismatched/missing artifact statuses, git-tracking failure
  modes, and tail-asymptotic fail-closed metadata. Validation:
  `tests/test_validation_figures.py` passed `32/32` in `0.16 s`; source-tree
  plus import-contract focused validation passed `85/85`; Ruff, `compileall`,
  and `git diff --check` passed. Direct coverage for
  `sfincs_jax/validation/figures.py` is `100%`.
- Tranche 113: expanded QI device-validation coverage for research-lane
  evidence gates without running solves: QI ladder mapping normalization,
  host-fallback warnings, malformed resolution gates, observable mismatch
  rejection, fail-closed artifact metadata, malformed JSON/scalar artifacts,
  missing files, operator-reuse/legacy-GPU helper recognition, missing GPU
  lanes, missing promotion JSON, bad selected roots, missing CPU/Fortran
  references, and malformed residual summaries. Validation:
  `tests/test_qi_device_artifact_policy.py tests/test_qi_res15_gpu_campaign.py tests/test_rhs1_qi_promotion.py`
  passed `21/21` in `0.61 s`; source-tree plus import-contract focused
  validation passed `74/74`; Ruff, `compileall`, and `git diff --check`
  passed. Direct coverage for `sfincs_jax/validation/qi_device.py` is `100%`.
- Tranche 114: expanded release-artifact and benchmark-policy coverage without
  generated outputs or solves: high-collisionality helper edge cases,
  phase-timer error recording, suite/autodiff loader fail-closed paths,
  resolution-floor and Appendix-B geometry validation errors, PAS benchmark
  artifact policy/classification, Fortran-suite summary release gates,
  research-lane malformed entries, and collisionality slope error gates.
  Validation:
  `tests/test_validation_artifacts.py tests/test_benchmark_artifact_policy.py tests/test_validation_policy_coverage.py tests/test_research_lane_policy.py`
  passed `75/75` in `0.36 s`; source-tree plus import-contract focused
  validation passed `128/128`; Ruff, `compileall`, and `git diff --check`
  passed. Direct coverage for `sfincs_jax/validation/artifacts.py` improved
  from `86%` to `95%`.
- Tranche 115: completed external-data fetch coverage for release-hosted
  equilibrium fixtures without downloading real large artifacts. Added local
  tarball tests for user-facing download progress and post-extraction
  verification failure, preserving the small-repository policy while covering
  the CI data-cache edge cases. Validation: `tests/test_data_fetch.py` passed
  `15/15` in `0.13 s`; source-tree plus import-contract focused validation
  passed `68/68`; Ruff, `compileall`, and `git diff --check` passed. Direct
  coverage for `sfincs_jax/validation/data_fetch.py` is `100%`.
- Tranche 116: expanded Fortran/PETSc validation-wrapper coverage without
  requiring a local SFINCS Fortran v3 build in CI. Added fake-executable tests
  for `SFINCS_FORTRAN_EXE` discovery, missing-input and missing-executable
  errors, successful output creation, tolerated MPI-finalization failure after
  HDF5 diagnostics are written, hard command failures, missing outputs,
  automatic temporary work directories, caller environment merging, and the
  equilibrium-localization hook. Validation:
  `tests/test_fortran_profile.py tests/test_validation_petsc_and_upstream_helpers.py`
  passed `18/18` in `1.74 s`; source-tree plus import-contract focused
  validation passed `71/71`; Ruff, `compileall`, and `git diff --check`
  passed. Direct coverage for `sfincs_jax/validation/fortran.py` is `99%`.
- Tranche 117: expanded RHSMode=2/3 transport-policy coverage around the
  production solver-selection decisions without adding new source modules or
  slow solves. Added deterministic tests for low-memory initial solve policy,
  active-DOF compaction, dense fallback/preconditioner memory admission,
  per-RHS loop flags, GMRES polish thresholds, residual-gate arrays,
  preconditioner alias normalization, domain-decomposition and sparse-JAX
  environment parsing, FP auto-preconditioner branch priority, preconditioner
  dispatch fallbacks, sparse-JAX memory guards, and strong-preconditioner cache
  reuse. Validation: `tests/test_transport_policy_coverage.py` passed `25/25`
  in `0.60 s`; adjacent source-tree/import/transport-policy validation passed
  `121/121`; Ruff, `compileall`, and `git diff --check` passed. Direct
  coverage for `sfincs_jax/problems/transport_policies.py` improved from
  `70%` to `86%`.
- Tranche 118: expanded RHSMode=2/3 transport linear-system and active-block
  preconditioner coverage with tiny algebraic fixtures instead of full
  production solves. Added tests for active symbolic block orderings, exact
  block-Schur application on a sparse system with tail closure, deterministic
  admission probes, residual-derived coarse correction construction, memory
  and malformed-probe guards, and direct-Pmat physics coarse-basis columns for
  constraint schemes 1 and 2 plus tail Schur response columns. Validation:
  the combined transport-linear focused bundle passed `138/138` in `21.55 s`;
  adjacent source-tree/import guards passed `79/79`; Ruff, `compileall`, and
  `git diff --check` passed. Focused combined coverage for
  `sfincs_jax/problems/transport_linear_system.py` improved from `47%` to
  `71%`.
- Tranche 119: expanded explicit sparse solver fail-soft coverage around the
  native sparse-factor wrappers used by bounded-memory preconditioner paths.
  Added tiny algebraic tests for symbolic block-factor local failure fallback,
  non-finite cleanup, residual-polish fallback and refinement behavior,
  coarse-correction wrapping with valid and empty bases, coarse-solve failure
  fallback, and regularized SuperLU failure fallback to a Jacobi factor.
  Validation: the focused sparse bundle passed `99/99` in `6.95 s`; adjacent
  source-tree/import guards passed `134/134`; Ruff, `compileall`, and
  `git diff --check` passed. Focused coverage for
  `sfincs_jax/solvers/explicit_sparse.py` improved from `87%` to `88%`.
- Tranche 120: expanded RHSMode=1 default preconditioner-selector coverage
  for production solver-path decisions without running solves. Added bounded
  tests for explicit controls, non-RHS1/Phi1 disablement, FP-DKES xblock
  environment propagation, constrained tokamak/PAS/FP full-preconditioner
  branches, GPU PAS callback routing, Schur-auto sharded/DKES routing,
  FP/PAS fallback priority, point/collision/point-xdiag fallbacks, and
  conservative invalid-axis handling. Validation:
  `tests/test_rhs1_preconditioner_auto_policy.py` passed `41/41` in `0.32 s`;
  the broader RHSMode-1 policy bundle passed `540/540` in `123.00 s`;
  adjacent focused guards passed `112/112`; Ruff, `compileall`, and
  `git diff --check` passed. In the broad policy bundle,
  `sfincs_jax/problems/profile_policies.py` improved from `88%` to `91%`.
- Tranche 121: expanded RHSMode=1 preconditioner-build control coverage and
  removed unreachable PAS auto-selection branches shadowed by earlier `has_pas`
  returns. Added bounded tests for strong-preconditioner aliases, residual and
  retry environment parsing, weak-PAS/guarded-MINRES controls, strong-control
  skip messages, auto-selection and adjustment helpers, reduced/full selection
  skip gates, post-primary guarded and weak MinRes correction acceptance, and
  guarded PAS-TZ overlay polynomial/structured branches. Validation:
  `tests/test_profile_response_preconditioner_build.py` passed `31/31` in
  `0.65 s`; adjacent preconditioner/source guards passed `109/109`;
  Ruff, `compileall`, and `git diff --check` passed. Direct coverage for
  `sfincs_jax/problems/profile_preconditioner_build.py` improved from `58%`
  to `97%`, while source statements dropped from `977` to `957`.
- Tranche 122: expanded sparse-direct coverage without production solves. Added
  tiny LSQR minimum-norm and host-direct residual-fallback tests, polish-disabled
  and factor-guard tests, forced host-direct factor-control tests, sparse-direct
  wrapper injection tests, explicit sparse-pattern probes, sparse cache-key
  extension tests, and sparse-JAX cache/reuse tests. Validation:
  `tests/test_profile_response_sparse_pc.py` passed `347/347` in `5.57 s`;
  sparse-PC plus source/import guards passed `400/400`; Ruff, `compileall`,
  and `git diff --check` passed. Direct coverage for
  `sfincs_jax/problems/profile_sparse_direct.py` improved from `90%` to `95%`.
- Tranche 123: expanded sparse-solve orchestration coverage without production
  solves. Added requested sparse-PC dispatch, preflight residual diagnostics,
  auto-retry acceptance, residual candidate accept/reject, true-active/window
  residual-correction, true-column-cache, true-coupled coarse, global-pattern,
  direct-tail, structured-tail, and support-mode promotion tests. Validation:
  `tests/test_profile_response_sparse_pc.py` passed `362/362` in `7.00 s`;
  sparse-PC plus source/import guards passed `415/415`; Ruff, `compileall`,
  and `git diff --check` passed. Direct coverage for
  `sfincs_jax/problems/profile_sparse_solve.py` reached `74%`.
- Tranche 124: expanded transport parallel-runtime coverage with bounded
  policy and artifact tests. Added XLA flag rewriting, release-quality scaling
  audits, claim-scope classification, multi-GPU case-throughput summaries,
  sharded solve summaries/plans, environment restoration, payload construction,
  GPU dispatch injection, and fail-closed sharding/planning gates. Validation:
  focused transport-parallel coverage passed `59/59` in `0.64 s`; transport
  parallel plus source/import guards passed `112/112`; Ruff, `compileall`, and
  `git diff --check` passed. Direct coverage for
  `sfincs_jax/problems/transport_parallel_runtime.py` reached `79%`.
- Tranche 125: expanded output-writer schema coverage without production solves.
  Added geometryScheme 1/4/5/11 output-dictionary branch tests, VMEC
  monoenergetic overwrite tests, export-f metadata tests, cache-miss and cached
  `uHat` tests, and verbose VMEC/Boozer writer-progress tests. Validation:
  `tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py`
  passed `108/108` in `3.08 s`; output plus source/import guards passed
  `161/161` in `7.68 s`; Ruff, `compileall`, and `git diff --check` passed.
  Direct coverage for `sfincs_jax/outputs/writer.py` improved from `36%` to
  `52%`.
- Tranche 126: consolidated QI-device control parsing into the existing QI
  device preconditioner owner. Moved the QI extra-coarse and residual-correction
  environment readers out of the generic profile-policy owner, kept thin
  compatibility wrappers for public imports, switched the QI sparse pipeline to
  the canonical solver import, and removed the large generic-policy
  implementation block. Validation:
  `tests/test_rhs1_xblock_fallback_initial_guess.py tests/test_rhs1_qi_residual_galerkin.py tests/test_rhs1_xblock_policy.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed `155/155` in `7.52 s`; Ruff and `compileall` passed. This reduced
  `problems/profile_policies.py` from `7916` to `7597` lines while keeping QI
  behavior unchanged.
- Tranche 127: compressed the moved QI-device control readers into typed
  control specs inside the QI device preconditioner owner. This preserves every
  QI control key, default, and environment suffix while eliminating repetitive
  dictionary construction. Validation repeated the QI policy/source bundle as
  `155/155` in `7.52 s`; Ruff and `compileall` passed. The touched owner group
  now has `18064` lines, `158` fewer than the pre-consolidation baseline.
- Tranche 128: tightened the source-package navigation and review contract.
  Expanded `sfincs_jax/README.md` with root-module and domain-package tables,
  stability/compatibility guidance, and generated-file policy. Added
  source-tree tests that require those review sections and reject tracked
  caches, binary solve outputs, traces, or large runtime artifacts inside the
  importable package. Validation:
  `tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed `69/69` in `5.28 s`; `compileall` and `git diff --check` passed.
- Tranche 129: expanded top-level output writer orchestration coverage without
  production solves. Added a deterministic RHSMode=3 transport-output test that
  exercises the non-streaming NPZ branch, transport matrix writeout, elapsed
  timings, coordinate-converted fluxes, classical flux per-RHS dispatch,
  optional solver diagnostics, and versioned solver trace metadata while
  monkeypatching only the expensive numerical solve. Validation:
  `tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py tests/test_transport_output_schema.py tests/test_output_formats.py`
  passed `118/118` in `2.56 s`.
- Tranche 130: expanded RHSMode-1 sparse-PC orchestration coverage around
  production-safety branches. Added tests for required direct-tail structured
  preconditioner fail-fast behavior, ensuring explicitly requested low-memory
  paths do not silently fall back to expensive host factors, and for failed
  support-mode preflight plus invalid true-window spec reporting. Validation:
  `tests/test_profile_response_sparse_pc.py` passed `364/364` in `3.95 s`.
- Tranche 131: extracted RHSMode-1 writer dense-cutoff parsing and physics-input
  normalization out of the monolithic output writer body into pure module-level
  helpers. Added bounded tests for dense cutoff defaults, invalid environment
  fallback, nonnegative FP cutoff clamping, SFINCS logical aliases, DKES ExB
  aliases, and radial-electric-drive magnitude selection. Validation:
  `tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py tests/test_output_formats.py tests/test_write_output_return_results.py`
  passed `124/124` in `3.18 s`; source-tree/import plus writer validation
  passed `149/149` in `8.06 s`; Ruff, `compileall`, and `git diff --check`
  passed.

Remaining consolidation steps:

1. Compatibility cleanup: keep deleted-facade tests and stale-import scans in
   place. Behavior tests now import canonical domain owners; no
   `sfincs_jax.v3_driver` imports are allowed, and private `profile_solve`
   helper-owner checks are confined to the explicit wrapper test.
2. Problem-family consolidation: remove historical transfer-state and
   production-campaign names from implementation files by merging RHSMode-1
   sparse setup/rescue owners into canonical profile sparse owners and moving
   domain-specific knobs to their domain owners. The sparse-solve compatibility
   namespace and Ruff waiver are closed, and QI-device control parsing now lives
   with the QI device preconditioner. Any remaining cleanup should reduce owner
   size or clarify orchestration boundaries without reintroducing broad reexport
   surfaces. Success is fewer broad namespace surfaces, simpler canonical
   imports, and unchanged RHSMode-1 policy/output tests.
3. Solver-family consolidation: merge same-family preconditioner modules only
   at durable physics/numerics boundaries. Success is fewer solver files and no
   loss of targeted preconditioner tests, not a single oversized grab-bag file.
4. Operator cleanup: merge small profile-response term helpers only when
   equation-to-file mapping remains clear in docs and tests. Otherwise retain
   them as pedagogical owners.
5. Review lock: keep `io.py` below `80` lines and implementation-free, keep
   `sfincs_jax/README.md` aligned with the source tree, and run source-layout,
   import-contract, docs, examples, CLI/output, and focused behavior guards.

### Lane 2 - Coverage And Future-Proof Tests

Status: 89% CI-measured package coverage on PR #8 commit `3cce604d`
(`TOTAL 69081 stmts, 7630 miss, 89%`). The direct public contract audit remains
closed at `modules_with_missing 0`; focused owner coverage is `98%` for
`workflows/scans.py` after Tranche 111 and `100%` for
`validation/figures.py` after Tranche 112 and `validation/qi_device.py` after
Tranche 113; `validation/artifacts.py` is `95%` after Tranche 114,
`validation/data_fetch.py` is `100%` after Tranche 115, and
`validation/fortran.py` is `99%` after Tranche 116; `transport_policies.py`
is `86%` after Tranche 117; `transport_linear_system.py` reaches `71%` in the
combined focused transport bundle after Tranche 118; `explicit_sparse.py` is
`88%` after Tranche 119; `profile_policies.py` reaches `91%` in the broad
RHSMode-1 policy bundle after Tranche 120; `profile_preconditioner_build.py`
is `97%` after Tranche 121; `profile_sparse_direct.py` is `95%` after
Tranche 122; `profile_sparse_solve.py` reaches `74%` in the owner sparse-PC
suite after Tranche 123; `transport_parallel_runtime.py` reaches `79%` in the
focused transport-parallel bundle after Tranche 124; `outputs/writer.py`
reaches `52%` in the focused writer/export bundle after Tranche 125 and has
additional RHSMode-1 writer-policy helper coverage after Tranche 131. The final
target is still 95%; the next coverage tranche should prioritize remaining
high-missing owners with bounded behavior tests or split large orchestration
regions before adding slow full solves.

Goal: reach 95% meaningful package coverage without slow CI or fixture bloat.

Required test types:

- Unit tests for pure kernels, policies, data models, and fail-closed guards.
- Numerical tests for finite-difference stencils, interpolation, matrix-free
  operator actions, sparse projections, preconditioner residual reductions, and
  implicit-solve algebra.
- Physics gates anchored in the neoclassical literature: Onsager symmetry and
  positivity, pitch-angle-scattering nullspaces and conservation, collisional
  high-`nu` trends, Redl/bootstrap-current normalization, ambipolar-root
  replay, and CPU/GPU equivalence on bounded fixtures.
- Frozen-reference parity tests against SFINCS Fortran v3 outputs stored as
  small JSON/HDF5 fixtures or fetched from release assets when large.
- Production-shape tests that check sizing, memory estimates, solver admission,
  output schemas, residual gates, and benchmark-artifact provenance without
  running full production solves in CI.

Coverage ramp:

1. Raise the enforced CI floor only after the measured branch coverage has
   margin above the proposed floor.
2. Move from the current floor to 85%, then 90%, then 95%.
3. Keep the full local audit command below the current five-to-six-minute
   range and keep GitHub Actions below 10 minutes by using bounded shards.
4. Prefer deleting obsolete code and adding owner-level tests over covering
   one-line branches in unused paths.

### Lane 3 - Documentation And Examples

Status: 94%.

Goal: make the public docs, README files, examples, and notebooks self-contained
and understandable for new users.

Remaining work:

- Keep the root README free of branch-history and progress-log language.
- Move caveats, benchmark provenance, and deferred research details to docs
  pages where claim boundaries are appropriate.
- Keep `sfincs_jax/README.md` synchronized with the source tree.
- Keep every top-level examples folder self-documenting with a README.
- Keep the tutorial layer focused on first-pass learning, with scripts and
  notebooks for CLI runs, output plotting, geometry loading, transport,
  bootstrap current, ambipolar roots, sensitivity, and optimization.
- Add notebook text for any new public capability added after this refactor.

### Lane 4 - Benchmark, Parity, Runtime, And Memory Regeneration

Status: ready for fresh evidence generation from the current refactor branch.
The source-layout, docs, examples, CLI/output, and helper-owner guards have
passed after the latest cleanup; benchmark regeneration should use reports
created from this branch state, not older pre-refactor artifacts.

Goal: regenerate release-facing figures and tables from the final branch state.

Required regeneration:

- CPU and office-GPU runtime/memory comparisons.
- Same-resolution SFINCS_JAX CPU/GPU versus SFINCS Fortran v3 parity tables.
- Bootstrap-current QA/QH comparisons with SFINCS_JAX, SFINCS Fortran v3, and
  Redl where the reference data are available.
- Production benchmark manifests and promotion summaries.
- README/docs plots, including only supported public claims.

Rules:

- Do not regenerate public figures from stale reports.
- Do not show favorable runtime or memory claims unless the benchmark manifest
  records same-resolution inputs, residual/output gates, runtime, memory, and
  solver provenance.
- If a production case remains deferred, document it once in docs and avoid
  presenting it as a public performance row.

### Lane 5 - Solver And Production-Performance Boundaries

Status: correctness-focused; further performance optimization is deferred
unless it blocks review.

Goal: keep automatic solver selection robust and honest while preserving
differentiable Python paths and fast non-autodiff CLI paths.

Rules:

- Do not restart broad solver research in this refactor PR.
- Preserve strict residual and output gates for auto-promotion.
- Keep host-only fast paths explicit and documented as non-autodiff.
- Keep JAX-native and implicit-differentiation paths available for Python
  workflows.
- If a production grid is correct but slower or higher-memory than Fortran v3,
  document that limitation rather than hiding it.

## Ordered Execution Plan

## Review Stop Pass - Branch Consolidation And Evidence Refresh

2026-07-05 review-stop pass:

- Confirmed `main` is an ancestor of `refactor/v3-driver-architecture`; the
  refactor PR branch is the single authoritative branch for review.
- Regenerated artifact-backed publication figures and summaries:
  Fortran-suite runtime/memory, publication validation dashboard,
  W7-X high-`nu'` performance, high-collisionality trend proxy,
  Simakov-Helander limit audit, Simakov-Helander high-`nu'` run plan, and
  autodiff/sensitivity validation.
- The full collisionality `generate_sfincs_paper_figs.py --plot-only` command
  was not rerun because local scan work directories are intentionally absent;
  the checked full-resolution JSON summaries remain the source for the
  retained figures. Rebuilding those scan work directories is a separate
  long-run campaign, not a review-stop blocker.
- Updated the high-`nu'` run-plan JSON to mark it explicitly as a deferred run
  plan, not a completed converged validation artifact.
- Validation completed:
  `tests/test_generate_fortran_suite_benchmark_summary.py`,
  `tests/test_benchmark_doc_claims.py`, `tests/test_validation_artifacts.py`,
  and `tests/test_benchmark_artifact_policy.py` as `68 passed`;
  benchmark artifact index release gate as `release-blocking=0`;
  source/example/public-doc guards as `69 passed`;
  Sphinx `-W` build passed; Python-file Ruff passed; `git diff --check`
  passed; full local suite passed as `4319 passed, 3 skipped, 2 warnings`
  in `16:43`.

### Phase A - Lock The Structure And Baseline

1. Run the source inventory commands and compare with
   `tests/fixtures/source_tree_expected.json`.
2. Run the source-tree and import-contract tests.
3. Run stale public import scans for `sfincs_jax.v3_driver` and deleted module
   names in docs, examples, scripts, tests, and source.
4. Record the current file-count and line-count baseline for root modules and
   each domain folder.
5. Update `sfincs_jax/README.md`, `docs/source_map.rst`, and tests only if the
   actual tree changes.

Acceptance:

- No nested packages.
- No unexpected root files.
- No public examples/scripts importing `v3_driver`.
- Deleted root and non-root compatibility facades stay absent.
- The baseline identifies which files will be deleted, merged, or retained.

### Phase B - Compatibility And Example Surface Cleanup

1. Audit compatibility facades and public imports:
   `v3_driver.py`, `operators/profile_response.py`,
   `problems/profile_response.py`, `problems/transport_matrix.py`, and
   `solvers/preconditioners.py`.
2. Keep the deleted-facade import tests and docs synchronized with canonical
   profile, transport, and solver owners.
3. Keep the former `examples/additional_examples/` namelist in `examples/data/`
   and preserve its benchmark label in scripts. Review `examples/sfincs_examples/`
   and `examples/upstream/` so archival navigation stays out of the first-run
   path and README wording remains explicit.
4. Keep the tutorial notebooks and task folders as the public learning surface.
   Add or update notebook/script pointers only if a move changes paths.
5. Run source-layout, import-contract, example-tree, benchmark-doc, and stale
   wording checks.

Acceptance:

- Public examples route new users to tutorials and task folders, not archival
  upstream fixtures.
- Compatibility modules are either deleted or explicitly tested and documented.
- No branch-history, "current main", "new version", or progress-log wording is
  present in public README/docs/example prose.

### Phase C - Source Consolidation In Three Bounded Passes

Pass 1 - Problem owners:

1. Consolidate RHSMode-1 sparse rescue/finalization code into canonical
   profile sparse owners. Deleted sparse replay filenames stay guarded absent;
   remaining work is reducing file count and line count inside the
   sparse-profile family.
2. Keep `profile_solve.py` as orchestration. Move complete phases, not helper
   fragments, into existing setup, policy, residual, sparse, or diagnostics
   owners.
3. Keep transport orchestration split between setup, linear system, solve,
   diagnostics, and finalization. Delete wrappers only after imports are moved.

Pass 2 - Solver families:

1. Merge same-family preconditioner modules at durable boundaries: PAS,
   full-FP, x-block, QI, symbolic/native sparse, transport-matrix, and Krylov.
2. Retain `explicit_sparse.py` as a single owner unless a complete symbolic or
   numeric factor family can move into an existing sparse-factor owner while
   reducing file count and preserving tests.
3. Do not create a generic preconditioner grab-bag. Names must tell users which
   physics/numerics family they are looking at.

Pass 3 - Operator and output owners:

1. Merge profile-response operator term helpers only where equation ownership
   stays obvious and testable.
2. Continue reducing `outputs/writer.py` only by moving whole schema phases to
   existing output owners. Avoid one-file-per-diagnostic growth.
3. Update docs source-map/API pages and `sfincs_jax/README.md` after each
   source-shape change.

Acceptance:

- Each pass deletes files or materially reduces lines according to the
  consolidation gates.
- No new nested packages and no helper-only files.
- Focused tests for moved owners pass before the next pass.
- Internal imports use canonical owners.

### Phase D - Coverage Ramp

1. Target the highest-missing modules with bounded owner-level tests.
2. Add literature-anchored physics gates where they cover real scientific
   behavior.
3. Add frozen-reference parity fixtures only if small; otherwise put the large
   artifact in a release and fetch it in tests.
4. Run focused suites after each tranche and a full coverage audit after a
   meaningful bundle.
5. Raise CI coverage floors only when measured margin exists.

Acceptance:

- Coverage reaches 95%.
- CI remains below 10 minutes.
- Tests are not smoke-only; each added test protects a numerical, physics,
  regression, API, output, or solver-policy invariant.

### Phase E - Review-Lock Docs And Examples

1. Run README/docs stale wording scans.
2. Run Sphinx with warnings as errors.
3. Run examples navigation and tutorial contract tests.
4. Run one bounded executable example per major workflow folder.
5. Verify no generated output or large artifact is tracked.

Acceptance:

- README and docs describe stable supported behavior, not branch history.
- Examples are organized by user task and have clear first-run scripts or
  notebooks.
- Public docs and examples use public APIs rather than compatibility facades.

### Phase F - Regenerate Release Evidence

1. Rerun production benchmark manifests after source/test structure stabilizes.
2. Rerun CPU and office-GPU gates.
3. Rerun SFINCS Fortran v3 references where required, or use frozen verified
   references when the runtime would exceed the benchmark budget.
4. Regenerate runtime, memory, parity, and bootstrap-current plots and tables.
5. Update README/docs figures only from the fresh reports.

Acceptance:

- Same-resolution comparisons are used wherever a comparison is claimed.
- Residual, output, runtime, memory, and solver-provenance gates are recorded.
- Deferred cases are documented honestly outside the README headline path.

### Phase G - PR Review Readiness

1. Run the review-lock validation bundle:
   source layout, domain imports, CLI/output, examples/tutorials, ambipolar,
   sensitivity, representative RHSMode 1/2/3, sparse/preconditioner policy,
   output writer, validation artifacts, Sphinx `-W`, Ruff, compile checks, and
   `git diff --check`.
2. Run a full local coverage audit.
3. Clean local caches and generated artifacts.
4. Commit and push to both the active branch and PR branch.
5. Update the PR body with the final source structure, coverage, benchmark
   evidence, retained compatibility shims, deferred production-performance
   boundaries, and validation commands.

Acceptance:

- PR #8 has one coherent refactor story.
- No hidden generated files or large artifacts are left in git.
- The branch is ready for review without further broad refactor work.

## 2026-07-05 Final-Review Consolidation Pass

Branch consolidation:

- `refactor/v3-driver-architecture` is the single review branch for PR #8.
- `main` is the merge base for this branch (`0 968` ahead/behind from
  `main...HEAD`), and `git merge main` reported already up to date.
- The stale detached worktree at
  `/Users/rogeriojorge/local/tests/sfincs_jax_main_clean` contains old
  uncommitted large equilibrium data and is intentionally excluded from this
  PR branch. It should not be merged into the review branch.

Artifact and documentation regeneration:

- Regenerated the artifact-backed Fortran-suite benchmark summary, validation
  dashboard, W7-X high-nu performance figure, autodiff sensitivity figures,
  high-collisionality proxy/audit figures, Simakov-Helander run-plan artifact,
  QA optimization lane, QI electron-root screen, and QA bootstrap-current
  comparison artifact.
- Validation passed for the regenerated artifact bundle:
  `tests/test_generate_fortran_suite_benchmark_summary.py`,
  `tests/test_benchmark_doc_claims.py`, `tests/test_validation_artifacts.py`,
  and `tests/test_benchmark_artifact_policy.py` as `68 passed in 1.38 s`.
- Source-tree, import-contract, and examples-tree validation passed as
  `63 passed in 4.92 s`.
- The benchmark artifact index release gate passed with `total=210` and
  `release-blocking=0`.
- Sphinx documentation built with warnings as errors.

Fresh production-suite evidence from the final branch:

- A direct local production-suite attempt was stopped because it is not a
  publishable replacement for the frozen benchmark reports, but it exposed
  real final-review issues that were triaged.
- Local SFINCS Fortran v3 segfaulted on
  `tokamak_1species_FPCollisions_noEr`; this is a local direct-Fortran
  blocker for replacing frozen references, not a JAX output mismatch.
- `tokamak_1species_FPCollisions_noEr_withQN` produced a strict comparison
  mismatch driven by solver-iteration metadata (`NIterations`), while physics
  flux/current quantities agreed at the checked tolerance.
- Monoenergetic production cases hit a JAX sharding bug:
  `NameError: Found an unbound axis name: zeta`. The full-system operator now
  keeps pjit placement constraints on the roll-stencil path and reserves named
  halo exchange for future `shard_map`/`pmap` call sites. Focused regression
  coverage passed in `tests/test_v3_system_cached_matvec.py`.
- A bounded production `monoenergetic_geometryScheme1` rerun no longer raised
  the unbound-axis error, but it exceeded the local 10-minute cap. This fixes
  the correctness regression and leaves production mono runtime as a benchmark
  item, not as a new README performance row.
- Large RHSMode=1 PAS no-Er cases previously entered monolithic LU/ILU and
  failed the explicit-sparse guard at `n=469321`. RHSMode=1 sparse-PC policy
  now mirrors the transport solver policy by auto-switching large un-overridden
  monolithic factors to `symbolic_block_lu_coarse`; explicit user overrides
  still win.
- A bounded production `tokamak_1species_PASCollisions_noEr` probe selected
  `symbolic_block_lu_coarse`, built the factor in about `10.2 s` with about
  `1.04 GB` factor estimate, and entered GMRES instead of failing the guard.
  The residual plateaued near `6.07e3` before the 4-minute cap, so the next
  technical target is Krylov/preconditioner quality rather than path-selection
  failure.
- Focused validation for the final solver-policy changes passed:
  `tests/test_profile_sparse_helper_coverage.py`,
  `tests/test_explicit_sparse_factor_policy.py`,
  `tests/test_explicit_sparse_factor_builder.py`, and
  `tests/test_v3_system_cached_matvec.py` as `36 passed in 1.08 s`; Ruff
  passed on the touched source and tests.
- The post-consolidation full local suite passed as `4322 passed, 3 skipped,
  2 warnings in 993.79 s`. The two warnings are the existing nonfinite-tail
  sanitization checks in `tests/test_transport_active_factor.py`.

Review boundary:

- Do not replace README runtime/memory figures or parity tables with the
  interrupted direct local suite. The checked-in figure regeneration remains
  artifact-backed by the verified reports until a fresh CPU/GPU/Fortran suite
  completes cleanly.
- The PR can be prepared for review with these fixes and with the remaining
  production-runtime issues documented as deferred performance/preconditioner
  work, unless the reviewer requires a fresh full production suite before
  merge.

### Tranche 132: standalone validation-matrix wording guard

Scope:

- Remove branch-history/progress-log phrasing from
  `docs/validation_matrix.rst` so the page reads as standalone validation
  documentation rather than a running development log.
- Add `docs/validation_matrix.rst` to the public standalone-doc guard in
  `tests/test_benchmark_doc_claims.py`. The guard rejects stale terms such as
  `now`, `previous`, `currently`, `new version`, and the previously rejected
  benchmark-manifest fragments.

Validation:

- `python -m pytest tests/test_benchmark_doc_claims.py -q`
  passed as `7 passed in 0.06 s`.
- The focused stale-wording scan over `docs/validation_matrix.rst` returned no
  matches for the rejected branch-history terms.

### Tranche 133: single-branch review regeneration pass

Scope:

- Confirmed `refactor/v3-driver-architecture` is the single review branch and
  already contains every commit from `origin/main`; no second feature branch is
  needed for the PR review pass.
- Regenerated the README-facing SFINCS Fortran v3 / SFINCS_JAX CPU/GPU
  runtime-memory summary from the checked CPU/GPU suite reports.
- Added a `--from-summary-json` mode to
  `examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py` so the
  checked QA/QH bootstrap-current figure bundles can be re-rendered from their
  committed summary JSONs without rerunning expensive kinetic solves or
  requiring ignored HDF5 sidecar directories.
- Re-rendered the checked QA/QH same-resolution and whole-radius
  bootstrap-current PDFs from the committed summary JSONs.
- Documented the summary-JSON figure regeneration command in the root README
  and `docs/examples.rst`.

Validation:

- GitHub PR #8 check rollup before this tranche was green and merge state was
  `CLEAN`.
- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py --min-fortran-runtime-s 10`
  completed and rewrote the benchmark summary figure bundle.
- `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas --min-fortran-runtime-s 10`
  completed and refreshed the README audit block.
- The QA/QH bootstrap-current figures were re-rendered from summary JSON with
  `--from-summary-json` for:
  `qs_paper_qa_same_resolution_11surface`,
  `qs_paper_qh_same_resolution_11surface`,
  `qs_paper_sfincs_jax_redl_comparison`, and
  `qs_paper_qh_sfincs_jax_redl_comparison`.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `pytest -q tests/test_benchmark_doc_claims.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_finite_beta_vmec_example.py`
  passed as `39 passed in 2.92 s`.
- `ruff check examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py tests/test_finite_beta_vmec_example.py`
  passed.
- `python -m compileall -q examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py tests/test_finite_beta_vmec_example.py`
  passed.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.
- A bounded fresh SFINCS Fortran v3 comparison passed on
  `quick_2species_FPCollisions_noEr` using
  `/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs` with
  `rtol=atol=1e-8`: `ok_fortran=1`, `ok_compare_common=1`, 209 common HDF5
  keys, and no missing JAX keys.
- The full local suite passed as `4329 passed in 659.65 s`.

### Tranche 134: examples workflow catalog

Scope:

- Added `examples/workflow_catalog.json` as a machine-readable navigation map
  for the example tree. The catalog records supported topic folders, first-pass
  entry points, typical commands, runtime budgets, and whether a workflow needs
  a local SFINCS Fortran v3 executable.
- Linked the catalog from `examples/README.md` and `docs/examples.rst` so the
  human-readable learning path and the checked catalog stay aligned.
- Extended `tests/test_examples_tree_contract.py` to validate catalog schema,
  approved folder coverage, entrypoint existence, and the first-run
  no-local-Fortran contract.

Validation:

- `python -m json.tool examples/workflow_catalog.json` passed.
- `pytest -q tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `16 passed in 0.16 s`.
- `ruff check tests/test_examples_tree_contract.py` passed.
- `python -m compileall -q tests/test_examples_tree_contract.py` passed.
- `sphinx-build -W -b html docs docs/_build/html` passed.

### Tranche 135: final consolidated review checkpoint

Scope:

- Verified that `refactor/v3-driver-architecture` is the only active PR branch
  needed for review and that it contains every commit from `origin/main`.
- Regenerated the README/docs runtime-memory benchmark summary from the checked
  CPU/GPU suite reports; the generator produced no content diff for the
  checked JSON, PNG, or README table.
- Re-rendered the QA/QH bootstrap-current comparison figures from committed
  summary JSONs; the checked PNGs were stable and the PDFs were refreshed.
- Rechecked the public benchmark artifact, docs build, examples catalog, and a
  fresh bounded SFINCS Fortran v3 parity comparison before handing the PR back
  for human review.

Validation:

- `git fetch origin --prune` followed by
  `git rev-list --left-right --count origin/main...HEAD` returned `0 977`
  before the final PDF-regeneration commit, confirming that the PR branch
  included all of `main`.
- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py --min-fortran-runtime-s 10`
  passed.
- `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas --min-fortran-runtime-s 10`
  passed and produced no README diff.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- The four QA/QH bootstrap-current comparison figures were re-rendered with
  `--from-summary-json` for the same-resolution and whole-radius QA/QH stems.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `python scripts/compare_v3_example_suite.py --pattern quick_2species --limit 1 --fortran-exe /Users/rogeriojorge/local/sfincs/fortran/version3/sfincs --fortran-timeout-s 90 --compute-solution --rtol 1e-8 --atol 1e-8 --out-root /tmp/sfincs_jax_final_fortran_compare_review_rtol1e8 -v`
  passed with `ok_fortran=1`, `ok_compare_common=1`, and 209 common HDF5 keys.
- `git diff --check` passed.
- `python -m compileall -q sfincs_jax tests` passed.
- `ruff check sfincs_jax tests/test_examples_tree_contract.py tests/test_finite_beta_vmec_example.py tests/test_benchmark_doc_claims.py`
  passed.
- The full local suite passed as `4330 passed in 669.00 s`.

### Tranche 136: RHSMode-1 preconditioner coverage checkpoint

Scope:

- Ran a fresh package-wide coverage audit on the refactor branch. The local
  measurement is `90%` package coverage with `4330 passed in 831.95 s`; the
  `95%` review target remains open and should be closed by behavior-owner
  tests rather than slow full-solve tests.
- Added a finite FP xMG radial-preconditioner test that exercises valid
  electric-field xDot metadata, invalid numeric environment fallbacks, cached
  coarse factors, and tail passthrough on a bounded operator fixture.
- Added direct reduced-tail operator tests covering fail-closed layout
  admission and the structured-CSR direct-tail callback path, including the
  shifted matrix, matvec, metadata, and progress-message contract.
- Rechecked source-tree, domain-package, examples-tree, and benchmark-doc
  guards so this coverage tranche does not disturb the simplified package
  layout or public wording contracts.

Validation:

- `python -m pytest -q --cov=sfincs_jax --cov-report=term-missing:skip-covered --cov-report=json:coverage.json`
  passed as `4330 passed in 831.95 s` with `90%` package coverage.
- `pytest -q tests/test_profile_reduced_tail_operator.py tests/test_rhs1_xblock_radial.py`
  passed as `14 passed in 1.30 s`.
- `pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 4.82 s`.
- `ruff check tests/test_rhs1_xblock_radial.py tests/test_profile_reduced_tail_operator.py`
  passed.
- `python -m compileall -q tests/test_rhs1_xblock_radial.py tests/test_profile_reduced_tail_operator.py`
  passed.
- The public README/docs/examples wording scan returned no active prose
  matches.

### Tranche 137: reduced-tail fallback assembly guard

Scope:

- Added a bounded direct-tail fallback-materialization test for
  `operators/profile_reduced_tail.py`.
- The test disables structured CSR admission, forces the pattern-probe fallback
  path, monkeypatches the true full-system action on a tiny constraintScheme=1
  operator, and checks kinetic diagonal shifts plus source/moment tail blocks.
- This closes another behavior branch in the RHSMode-1 lower-memory
  preconditioner stack without adding any production solve or benchmark runtime
  to CI.

Validation:

- `pytest -q tests/test_profile_reduced_tail_operator.py tests/test_rhs1_xblock_radial.py`
  passed as `15 passed in 1.94 s`.
- `ruff check tests/test_profile_reduced_tail_operator.py` passed.
- `python -m compileall -q tests/test_profile_reduced_tail_operator.py`
  passed.
- `git diff --check` passed.

### Tranche 138: transport writer diagnostic-completion guard

Scope:

- Added a bounded output-writer orchestration test for RHSMode=2 transport
  output when the solver returns only sparse/minimal transport fields.
- The test monkeypatches the transport solve, VM-only diagnostic fill, and
  classical-flux callbacks on a tiny operator fixture, then verifies that the
  writer completes production diagnostic datasets, coordinate variants,
  transport matrix layout, classical fluxes, and `NIterations` without running
  a physical solve.
- This targets one of the remaining large orchestrator coverage gaps while
  preserving CI runtime and the simplified source-tree contract.

Validation:

- `pytest -q tests/test_io_output_policy_coverage.py -k 'transport_npz'`
  passed as `2 passed, 93 deselected in 0.90 s`.
- `pytest -q tests/test_io_output_policy_coverage.py` passed as
  `95 passed in 2.40 s`.
- `pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 4.65 s`.
- `ruff check tests/test_io_output_policy_coverage.py` passed.
- `python -m compileall -q tests/test_io_output_policy_coverage.py` passed.
- `git diff --check` passed.

### Tranche 139: transport geometry-only writer guard

Scope:

- Added a bounded output-writer test for RHSMode=2 geometry-only output.
- The test verifies that `compute_transport_matrix=False` forces
  `NIterations=0` even if the base output dictionary contains a stale value,
  writes the selected output format through the common writer route, and records
  a solver trace with `selected_path="geometry_only"`.
- This protects a user-facing CLI/API branch and adds coverage to the large
  writer orchestrator without running a transport solve.

Validation:

- `pytest -q tests/test_io_output_policy_coverage.py -k 'geometry_only or transport_npz'`
  passed as `3 passed, 93 deselected in 0.82 s`.
- `pytest -q tests/test_io_output_policy_coverage.py` passed as
  `96 passed in 2.23 s`.
- `pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 4.37 s`.
- `ruff check tests/test_io_output_policy_coverage.py` passed.
- `python -m compileall -q tests/test_io_output_policy_coverage.py` passed.
- `git diff --check` passed.

### Tranche 140: structural review audit checkpoint

Scope:

- Re-audited the active source tree against the explicit review goals: the
  importable package has only root modules plus one level of domain folders,
  with no nested source packages beyond local `__pycache__` directories.
- Re-ran the public README/docs/examples stale-wording scan for branch-history
  phrasing, including the previously rejected README fragments.
- Rechecked tracked README/docs/examples/source files for large blobs over
  `2 MiB`; no tracked files in those public surfaces exceeded the limit.
- Confirmed these audits are already enforced by
  `tests/test_source_tree_consolidation.py`,
  `tests/test_examples_tree_contract.py`, and
  `tests/test_benchmark_doc_claims.py`, so no duplicate guard was added.

Validation:

- `find sfincs_jax -type d -name __pycache__ -prune -o -type d -print | awk -F/ 'NF>3 {print}'`
  returned no nested source package directories.
- The public wording `rg` scan returned no active prose matches.
- The tracked-file size scan over `README.md`, `docs`, `examples`, and
  `sfincs_jax` reported `large_count 0`.
- `pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 4.37 s` in the previous tranche.

### Tranche 141: final PR-review validation and artifact refresh

Scope:

- Confirmed the branch topology for review: `refactor/v3-driver-architecture`
  is the single active PR branch, is `0` commits behind `origin/main`, and is
  exactly synchronized with `origin/refactor/v3-driver-architecture` before
  this final checkpoint.
- Regenerated the public SFINCS Fortran v3 vs `sfincs_jax` CPU/GPU
  runtime-memory summary from the canonical checked CPU/GPU suite reports.
  The regenerated JSON/PNG/PDF and README audit block were deterministic and
  produced no tracked diff.
- Rechecked the canonical benchmark JSON with the artifact policy gate. The
  README-facing benchmark set contains `24` CPU rows and `24` GPU rows after
  excluding the documented low-Fortran-runtime smoke rows; every included row
  is `parity_ok` with zero practical and strict common-output mismatches.
- Regenerated the QA/QH finite-beta bootstrap-current comparison PDFs from the
  checked summary JSON files, including the same-resolution QA/QH panels and
  the broader QA/QH SFINCS-JAX/SFINCS Fortran v3/Redl panels.
- Fixed the only actionable focused-ruff issue found in the validation bundle:
  `tests/test_jax_ecosystem_backend_probes.py` now explicitly documents the
  intentional import ordering needed to enable JAX X64 before importing JAX
  array modules.
- Re-ran a fresh bounded SFINCS Fortran v3 parity check on
  `quick_2species_FPCollisions_noEr`. Fortran v3 detected MUMPS, JAX wrote the
  output, and the common HDF5 output comparison passed with zero mismatches.
- Rebuilt the Sphinx documentation after plot regeneration.

Validation:

- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py`
  completed and wrote the public benchmark JSON and plot.
- `python scripts/generate_readme_fast_branch_audit.py` completed and produced
  no tracked README diff.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `python scripts/check_repo_size.py` passed with `0` reviewed files above
  `2 MiB`.
- `python scripts/check_release_gates.py` passed.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --from-summary-json ...`
  regenerated the QA/QH comparison figures from checked summaries.
- `python -m pytest -q tests/test_generate_fortran_suite_benchmark_summary.py tests/test_benchmark_doc_claims.py`
  passed as `14 passed in 1.50 s`.
- `python -m pytest -q tests/test_finite_beta_vmec_example.py tests/test_examples_tree_contract.py`
  passed as `34 passed in 1.50 s`.
- `python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 4.47 s`.
- `python -m pytest -q tests/test_generate_fortran_suite_benchmark_summary.py tests/test_benchmark_doc_claims.py tests/test_finite_beta_vmec_example.py tests/test_examples_tree_contract.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `103 passed in 8.88 s`.
- `python -m pytest -q tests/test_jax_ecosystem_backend_probes.py` passed as
  `4 passed in 3.67 s`.
- `python -m ruff check sfincs_jax tests scripts/generate_readme_fast_branch_audit.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py`
  passed.
- `python -m compileall -q sfincs_jax tests/test_jax_ecosystem_backend_probes.py scripts/generate_readme_fast_branch_audit.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py`
  passed.
- `git diff --check` passed.
- `python -m sphinx -b html docs docs/_build/html` passed.
- `python scripts/compare_v3_example_suite.py --pattern quick_2species --limit 1 --fortran-exe /Users/rogeriojorge/local/sfincs/fortran/version3/sfincs --fortran-timeout-s 90 --compute-solution --rtol 1e-8 --atol 1e-8 --out-root /tmp/sfincs_jax_final_fortran_compare_review_rtol1e8 -v`
  passed with `ok_write_output=True`, `ok_fortran=True`,
  `ok_compare_common=True`, and zero common-output mismatches.
- `python -m pytest -q` passed as `4336 passed in 653.68 s`.

Review note:

- A broad `ruff check .` and `compileall ... examples` intentionally remain
  outside the release validation contract because they include archived
  upstream SFINCS example utilities and path-bootstrapped scripts that are not
  formatted as importable Python package code.
- A fresh live all-example SFINCS Fortran/JAX rerun is not claimed in this
  checkpoint; the public runtime/memory/parity figure is regenerated from the
  checked CPU/GPU/Fortran suite reports, and the final live Fortran validation
  is the bounded `quick_2species` spot comparison above.

### Tranche 142: coverage-directed direct-tail operator gates

Scope:

- Downloaded and remapped the completed CI coverage shards from run
  `28771095543` to get a valid per-file coverage table without re-running the
  full coverage suite locally.
- Confirmed the current package coverage baseline is `90%`
  (`69,098` statements, `6,968` missing). The 95% objective remains open and
  requires broad coverage over the largest solver/orchestrator modules rather
  than a small final patch.
- Ranked the largest gaps: `problems/profile_solve.py` at `68%`,
  `problems/transport_solve.py` at `70%`,
  `solvers/preconditioner_xblock_tz_sparse.py` at `77%`,
  `operators/profile_system.py` at `79%`, and
  `operators/profile_true_operator_rescue.py` at `82%`.
- Added bounded unit/regression tests for
  `operators/profile_reduced_tail.py`, which is a smaller but important
  solver-admission module used by the lower-memory RHSMode=1 path.
- The new tests cover three direct-tail active term-level cases:
  incomplete active f-block blocks fall back safely, projected CSR budget
  rejection falls back safely, and successful whichMatrix=0 active f-block
  projection assembles the kinetic, source, moment, and shifted tail blocks
  without pattern-probing the f-block.

Validation:

- `python -m pytest -q tests/test_profile_reduced_tail_operator.py` passed as
  `6 passed in 0.67 s`.
- `python -m pytest -q tests/test_profile_reduced_tail_operator.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `61 passed in 4.79 s`.
- `python -m ruff check tests/test_profile_reduced_tail_operator.py` passed.
- `python -m compileall -q tests/test_profile_reduced_tail_operator.py`
  passed.
- `git diff --check` passed.

Next coverage target:

- Continue with bounded tests for the largest low-coverage modules in this
  order unless CI coverage shows a different ranking:
  `profile_solve.py`, `transport_solve.py`,
  `preconditioner_xblock_tz_sparse.py`, `profile_system.py`, and
  `profile_true_operator_rescue.py`.

## Tranche 143: PR Integration And Final Review Pass

Status:

- The active review branch is `refactor/v3-driver-architecture`; `origin/main`
  is an ancestor of the branch, so there is no separate main-line work to merge.
  The only additional checkout observed locally is the detached
  `sfincs_jax_main_clean` worktree used for clean-run comparisons.
- Regenerated the runtime/memory benchmark summary and README audit block from
  the checked CPU/GPU suite reports. The public benchmark still has 39 audited
  CPU/GPU parity rows and 24 plotted rows after applying the 10 s Fortran-v3
  runtime floor.
- Regenerated the QA/QH bootstrap-current comparison figure bundles from their
  checked summary JSONs. The re-render touched only the small PDF artifacts; the
  scientific JSON/PNG data stayed stable.
- Ran a live bounded SFINCS Fortran v3 parity check for
  `quick_2species_FPCollisions_noEr` with `--compute-solution` and `1e-8`
  tolerances. Fortran v3 detected MUMPS, the JAX HDF5 output was written, and
  common-output comparison passed with zero mismatches.

Validation:

- `python scripts/check_repo_size.py` passed.
- `python scripts/check_release_gates.py` passed.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `python -m sphinx -b html docs docs/_build/html` passed.
- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py`
  passed.
- `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas --min-fortran-runtime-s 10`
  passed.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --from-summary-json ...`
  passed for the QA/QH same-resolution and README comparison summaries.
- `python -m ruff check sfincs_jax tests scripts/generate_readme_fast_branch_audit.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py`
  passed.
- `python -m compileall -q sfincs_jax tests/test_jax_ecosystem_backend_probes.py scripts/generate_readme_fast_branch_audit.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py`
  passed.
- The public stale-wording `rg` scan over README/docs/examples passed with no
  matches.
- `python -m pytest -q` passed as `4339 passed in 596.09 s`.

## Tranche 144: Sparse X-Block Route Coverage

Scope:

- Added bounded tests for the RHSMode=1 sparse x-block preconditioner routes in
  `tests/test_rhs1_sxblock_tz_sparse_host.py`.
- The tests use synthetic diagonal block factors and a diagonal extra/source
  probe, so they exercise the production route logic without assembling a full
  SFINCS operator or running a Krylov solve.
- Covered three route families that matter for large-run robustness:
  host-side sparse x-block factors, padded JAX/device factors, and compact CSR
  JAX/device factors.
- Verified that inactive high-`L` padding remains unchanged, active x-blocks
  receive the expected local inverse, and the extra/source block is inverted
  through the same tail solve used by real preconditioner setup.

Validation:

- `python -m pytest -q tests/test_rhs1_sxblock_tz_sparse_host.py` passed as
  `11 passed in 0.81 s`.
- `python -m pytest -q tests/test_rhs1_sxblock_tz_sparse_host.py tests/test_sparse_assembly.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `78 passed in 5.57 s`.
- `python -m ruff check tests/test_rhs1_sxblock_tz_sparse_host.py` passed.
- `python -m compileall -q tests/test_rhs1_sxblock_tz_sparse_host.py`
  passed.

Coverage note:

- A targeted local `pytest --cov` run still aborts with exit code `134`, which
  is the same local pytest-cov limitation recorded earlier. Treat the GitHub
  Actions coverage shards as the authoritative package-coverage measurement.

## Tranche 145: Examples Decision-Map Navigation

Scope:

- Added a compact decision map to `examples/README.md` so users can choose a
  workflow by task before reading the longer tables.
- Mirrored the same decision map in `docs/examples.rst` so the ReadTheDocs page
  remains consistent with the repository examples index.
- Extended `tests/test_examples_tree_contract.py` to require the decision-map
  section, labels, and target folders/scripts. This prevents the examples tree
  from drifting back into an unstructured list of scripts while preserving the
  existing topic-folder layout.

Validation:

- `python -m pytest -q tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `16 passed in 0.15 s`.
- `python -m sphinx -b html docs docs/_build/html` passed.
- `python -m ruff check tests/test_examples_tree_contract.py` passed.
- `python -m compileall -q tests/test_examples_tree_contract.py` passed.

## Tranche 146: Single-Branch Review Validation and Figure Refresh

Scope:

- Verified that the only active review branch is
  `refactor/v3-driver-architecture` and that `origin/main` is already an
  ancestor of it. The extra checkout at
  `/Users/rogeriojorge/local/tests/sfincs_jax_main_clean` is detached and holds
  uncommitted large equilibrium blobs; those files are intentionally excluded
  from this PR and from the release-data policy.
- Regenerated the README-facing SFINCS Fortran v3 / SFINCS_JAX CPU/GPU runtime
  and memory summary from the checked CPU/GPU suite reports.
- Regenerated the QA/QH bootstrap-current figure bundles from their checked
  summary JSON files.
- Made the QS-paper bootstrap-current PDF writer deterministic by pinning PDF
  metadata. Future plot refreshes should no longer create timestamp-only PDF
  diffs.

Validation:

- GitHub PR #8 checks are green for head `cc05cc79`: coverage shards,
  coverage-report, examples smoke, external-data smoke, optional ecosystem
  gates, tests, and docs all passed.
- The combined CI coverage artifacts report `TOTAL 69098 stmts, 6921 miss,
  90%`. The 95% coverage gate is still open and should stay explicit in this
  plan.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `python scripts/benchmark_artifact_index.py examples/publication_figures/artifacts docs/_static/figures/vmec_jax_finite_beta`
  reported `release-blocking=0`.
- `python -m pytest -q tests/test_benchmark_doc_claims.py tests/test_validation_artifacts.py tests/test_benchmark_artifact_policy.py tests/test_examples_tree_contract.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `126 passed in 4.69 s`.
- The public stale-wording `rg` scan over README/docs/examples passed with no
  matches.
- The targeted frozen Fortran/operator/HDF5 parity bundle passed as
  `220 passed in 217.44 s`.
- The targeted physics/validation/benchmark bundle passed as
  `1598 passed in 99.27 s`.
- `python -m pytest -q` passed as `4342 passed in 613.35 s`.
- `python -m sphinx -b html docs docs/_build/html` passed.
- `python -m ruff check sfincs_jax tests` passed.
- `python -m pytest -q tests/test_finite_beta_vmec_example.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `41 passed in 1.64 s` after the deterministic PDF metadata change.
- `python -m ruff check examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py tests/test_finite_beta_vmec_example.py`
  passed.
- `python -m compileall -q examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py tests/test_finite_beta_vmec_example.py`
  passed.
- `git diff --check` passed.

Live Fortran note:

- A bounded local one-case Fortran-v3/JAX runner probe on
  `quick_2species_FPCollisions_noEr` completed but reported two tiny
  solver-bucket mismatches (`full_f`, `totalDensity`) at maximum absolute
  difference `5.39e-10`. The runner classified the blocker as
  `reference solver quality` because the Fortran final residual was
  `5.524e-09`, more than `10x` the estimated target. This was not promoted into
  tracked generated docs or fixtures; the production README figures remain
  backed by the checked suite reports.

## Tranche 147: Optional-Physics Sharded-Grid Padding

Scope:

- Fixed x-padding for optional profile-system operators that carry
  `n_xi_for_x` metadata. ExB theta/zeta, Er x-dot, and magnetic-drift
  theta/zeta now expand their active-L metadata consistently with the padded
  f-block shape.
- Added a fast regression over real tiny v3 fixtures covering Er x-dot, Er
  xi-dot, magnetic drift, PAS, and FP-with-Phi1 collision branches. The test
  pads theta, zeta, and x, then verifies full-vector roundtrip and optional
  operator metadata shapes without launching a solve.
- This closes a device/sharded-matvec correctness gap found while increasing
  meaningful coverage in `profile_system.py`.

Validation:

- `python -m pytest -q tests/test_profile_system_support.py` passed as
  `41 passed in 11.09 s`.
- `python -m ruff check sfincs_jax/operators/profile_system.py tests/test_profile_system_support.py`
  passed.
- `python -m compileall -q sfincs_jax/operators/profile_system.py tests/test_profile_system_support.py`
  passed.
- `python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 5.52 s`.
- `python -m pytest -q tests/test_profile_system_support.py tests/test_full_system_operator_jit.py tests/test_collisionless_operator_parity.py tests/test_exb_theta_parity.py tests/test_magnetic_drifts_parity.py`
  passed as `51 passed in 18.94 s`.
- `python -m pytest -q tests/test_pas_collision_operator_parity.py tests/test_fblock_pas_matvec_parity.py tests/test_fblock_fokker_planck_matvec_parity.py tests/test_fblock_fused_matvec.py tests/test_v3_fblock_smoke.py`
  passed as `6 passed in 7.16 s`.
- `python -m ruff check sfincs_jax tests` passed.
- `git diff --check` passed.

## Tranche 148: Padded Operator Matvec Preservation

Scope:

- Extended x-padding to the always-on collisionless operator `x` grid and to
  the Er x-dot derivative matrices. The optional x-dependent magnetic-drift
  terms now also carry padded `x` metadata, while ExB and Er xi-dot retain only
  their active-L metadata padding.
- Strengthened the optional-physics padding regression so it applies the padded
  full-system operator, unpads the result, and checks equality with the
  unpadded operator to `1e-10`. This verifies that sharded/device padding is not
  only shape-consistent but also mathematically transparent on the original
  degrees of freedom.
- The regression covers Er x-dot, Er xi-dot, magnetic drift, PAS, and FP-with-
  Phi1 branches using real tiny v3 namelists and no linear solves.

Validation:

- `python -m pytest -q tests/test_profile_system_support.py` passed as
  `41 passed in 15.54 s`.
- `python -m ruff check sfincs_jax/operators/profile_system.py tests/test_profile_system_support.py`
  passed.
- `python -m compileall -q sfincs_jax/operators/profile_system.py tests/test_profile_system_support.py`
  passed.
- `python -m pytest -q tests/test_profile_system_support.py tests/test_full_system_operator_jit.py tests/test_collisionless_operator_parity.py tests/test_exb_theta_parity.py tests/test_magnetic_drifts_parity.py`
  passed as `51 passed in 21.53 s`.
- `python -m pytest -q tests/test_pas_collision_operator_parity.py tests/test_fblock_pas_matvec_parity.py tests/test_fblock_fokker_planck_matvec_parity.py tests/test_fblock_fused_matvec.py tests/test_v3_fblock_smoke.py`
  passed as `6 passed in 5.13 s`.
- `git diff --check` passed.

## Tranche 149: Profile-System Physics and Differentiability Branch Gates

Scope:

- Added non-solve profile-system tests for branches that matter for production
  physics and differentiable workflows:
  - invalid full-vector shape admission;
  - quasineutrality option 1 with nonlinear Phi1 diagonal scaling and invalid
    environment-value fallback;
  - Phi1-in-kinetic Jacobian dependence on the current distribution state;
  - cached operator fallback inside a JAX transform, ensuring transformed calls
    stay on the local JIT path instead of entering the top-level sharded mesh.
- These tests exercise real tiny v3 namelists and physics operators without
  adding slow Krylov solves to CI.

Validation:

- `python -m pytest -q tests/test_profile_system_support.py` passed as
  `45 passed in 17.99 s`.
- `python -m ruff check tests/test_profile_system_support.py sfincs_jax/operators/profile_system.py`
  passed.
- `python -m compileall -q tests/test_profile_system_support.py sfincs_jax/operators/profile_system.py`
  passed.
- `python -m pytest -q tests/test_profile_system_support.py tests/test_full_system_operator_jit.py tests/test_collisionless_operator_parity.py tests/test_exb_theta_parity.py tests/test_magnetic_drifts_parity.py`
  passed as `55 passed in 26.54 s`.
- `python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 5.68 s`.
- `git diff --check` passed.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_tranche149.json`
  passed as `4361 passed in 303.15 s`. Coverage is
  `62340 / 69108` statements, `6768` missing lines, `90.21%` total. The 95%
  meaningful-coverage gate remains open and requires larger structural coverage
  work in the orchestration and preconditioner modules.

## Tranche 150: Transport-Solve Orchestration Wrapper Gates

Scope:

- Added `tests/test_transport_solve_module_wrappers.py` to mirror the
  profile-solve wrapper coverage for RHSMode-2/3 transport orchestration.
- Covered the transport-specific distributed-axis wrapper, preconditioner cache
  key dtype injection, FP direct/reduced preconditioner wrapper hooks,
  parallel-worker payload delegation, and the top-level early return through
  the parallel runtime.
- The tests are no-solve dependency-injection checks that guard the refactor
  seams keeping `transport_solve.py` as orchestration rather than a sink for
  helper implementations.

Validation:

- `python -m pytest -q tests/test_transport_solve_module_wrappers.py` passed
  as `5 passed in 0.85 s`.
- `python -m ruff check tests/test_transport_solve_module_wrappers.py` passed.
- `python -m compileall -q tests/test_transport_solve_module_wrappers.py`
  passed.
- `python -m pytest -q tests/test_transport_solve_module_wrappers.py tests/test_transport_loop_support.py tests/test_transport_sparse_direct.py tests/test_transport_parallel.py`
  passed as `97 passed in 37.93 s`.
- `python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `71 passed in 5.90 s`.
- `git diff --check` passed.

## Tranche 151: Consolidated PR Review Checkpoint

Scope:

- Verified branch hygiene for the review pass:
  - `refactor/v3-driver-architecture` is the only active PR branch.
  - `origin/main` is already an ancestor of the PR branch.
  - The branch is `995` commits ahead of `origin/main` and `0` commits behind,
    so no additional merge from `main` is required before review.
- Regenerated the release-facing runtime/memory comparison artifacts from the
  canonical CPU/GPU suite reports:
  - CPU report:
    `tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak/suite_report.json`
  - GPU report:
    `tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas/suite_report.json`
  - summary JSON:
    `examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  - figure:
    `docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png`
- Regenerated the README example-suite audit block from the same CPU/GPU suite
  reports. The README audit remains `39/39` CPU parity and `39/39` GPU parity,
  with no practical or strict mismatches.
- Re-rendered the QA and QH SFINCS-JAX / SFINCS Fortran v3 / Redl bootstrap
  current comparison figures from their checked summary JSON files so the PNG,
  PDF, JSON, and README text remain synchronized.
- Confirmed the public runtime/memory plot is still a reference-runtime-window
  comparison filtered to SFINCS Fortran v3 rows with runtime at least `10 s`.
  The summary JSON still records production-resolution floor violations for
  lower-resolution historical/smoke rows instead of silently presenting them as
  production-resolution performance claims.

Validation:

- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py --cpu-report tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak/suite_report.json --gpu-report tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas/suite_report.json --out-dir docs/_static/figures/paper --summary-json examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json --min-fortran-runtime-s 10`
  passed and rewrote the benchmark summary artifact/figure.
- `python scripts/generate_readme_fast_branch_audit.py --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas --min-fortran-runtime-s 10`
  passed and refreshed the README audit block.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --from-summary-json docs/_static/figures/vmec_jax_finite_beta/qs_paper_sfincs_jax_redl_comparison.json --fig-dir docs/_static/figures/vmec_jax_finite_beta --stem qs_paper_sfincs_jax_redl_comparison`
  passed.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --from-summary-json docs/_static/figures/vmec_jax_finite_beta/qs_paper_qh_sfincs_jax_redl_comparison.json --fig-dir docs/_static/figures/vmec_jax_finite_beta --stem qs_paper_qh_sfincs_jax_redl_comparison`
  passed.
- `python scripts/check_benchmark_artifacts.py examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `python -m pytest -q tests/test_validation_figures.py tests/test_validation_artifacts.py tests/test_benchmark_doc_claims.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py`
  passed as `124 passed in 4.78 s`.
- `python -m ruff check sfincs_jax tests scripts/generate_readme_fast_branch_audit.py scripts/check_benchmark_artifacts.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py`
  passed.
- `python -m compileall -q sfincs_jax tests scripts/generate_readme_fast_branch_audit.py scripts/check_benchmark_artifacts.py examples/publication_figures/generate_fortran_suite_benchmark_summary.py examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py`
  passed.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_final_review.json`
  passed as `4366 passed in 320.30 s`. Coverage is
  `62340 / 69108` statements, `6768` missing lines, `90.21%` total.
- PR #8 GitHub checks passed: build, four coverage shards, coverage report,
  examples smoke, external-data smoke, optional ecosystem gates, and tests.

Review status:

- This checkpoint is ready for PR review.
- It does not close the 95% coverage gate.
- It does not replace the tracked CPU/GPU suite reports with a fresh
  production-resolution SFINCS Fortran v3 rerun; that remains a separate
  explicitly budgeted compute campaign.

## Tranche 152: RHSMode=1 Solver-Trace Robustness Gates

Scope:

- Added fail-closed RHSMode=1 output trace tests for production runs that fail
  or return partial solver metadata:
  - nonconverged trace generation without an operator on the solve result;
  - `op0` fallback when the result has malformed size/collision metadata;
  - final solver-trace generation with malformed residual/rhs/elapsed fields
    but valid per-RHS residual summaries and profiler entries.
- These tests protect the user-facing progress/profiling lane for large
  RHSMode=1 runs without adding slow solves to CI.
- The branch is intentionally diagnostic/output-policy focused: it hardens
  trace sidecars that explain residuals, memory estimates, selected solve path,
  and partial metadata when a production run is refused as nonconverged.

Validation:

- `python -m pytest -q tests/test_solver_trace_output_formats.py` passed as
  `15 passed in 0.49 s`.
- `python -m ruff check tests/test_solver_trace_output_formats.py` passed.
- `python -m compileall -q tests/test_solver_trace_output_formats.py` passed.
- `python -m pytest -q tests/test_solver_trace_output_formats.py tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py tests/test_rhsmode1_current_closure.py`
  passed as `132 passed in 2.45 s`.
- `python -m pytest -q tests/test_solver_trace_output_formats.py tests/test_io_output_policy_coverage.py tests/test_io_export_and_h5_coverage.py tests/test_rhsmode1_current_closure.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py`
  passed as `203 passed in 6.83 s`.
- `git diff --check` passed.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_trace_tranche.json`
  passed as `4369 passed in 321.20 s`. Coverage is
  `62366 / 69108` statements, `6742` missing lines, `90.24%` total.
- `sfincs_jax/outputs/rhsmode1.py` improved from `108` missing lines to `83`
  missing lines, raising that module from `88.96%` to `91.51%`.
- A module-only `pytest --cov=sfincs_jax.outputs.rhsmode1` probe aborted
  locally with exit code `134` and no traceback, but the standard full-suite
  package coverage gate above passed and is the authoritative coverage audit
  for this tranche.

Status:

- The 95% meaningful package coverage gate remains open.
- Highest remaining coverage debt remains concentrated in
  `problems/profile_solve.py`, `operators/profile_full_system.py`,
  `solvers/explicit_sparse.py`, `problems/transport_solve.py`, and the
  QI/x-block preconditioner modules.

## Tranche 153: Native Sparse And Structured Full-CSR Coverage Gates

Scope:

- Confirmed branch consolidation status for the review PR: `main` and
  `origin/main` are ancestors of `refactor/v3-driver-architecture`; there is no
  divergent main-side work to merge. The refactor branch is the single review
  branch.
- Fixed a fail-closed sparse coarse-correction bug:
  `_SparseCoarseCorrectionFactor.solve()` now replaces non-finite base-factor
  output entries with the incoming residual before applying the coarse
  correction.
- Added native sparse symbolic/factor tests for:
  - symbolic analysis and small block factor solves;
  - non-finite solve sanitization;
  - sparse coarse and residual-polish fail-closed paths;
  - BLR Schur Woodbury and GMRES fallback routes;
  - symbolic Schur, superblock, and nested-dissection factor wrappers;
  - deterministic probe and admission-gate edge cases.
- Added structured RHSMode=1 full-CSR tests for:
  - full-system CSR cache reuse and cache-time memory-budget rejection;
  - preflight CSR-budget rejection before f-block assembly;
  - active projection, post-build budget rejection, and sparse-bundle matvecs;
  - direct and LGMRES solve wrapper validation/error gates.
- Regenerated the public Fortran-v3 runtime/memory summary and QA/QH
  bootstrap-current figures from the tracked JSON artifacts. The regenerated
  outputs were byte-stable relative to the repository.

Validation:

- `python -m pytest -q tests/test_profile_full_system_structured_selection.py`
  passed as `4 passed in 0.30 s`.
- `python -m pytest -q tests/test_explicit_sparse_symbolic_native.py` passed
  as `7 passed in 0.14 s`.
- `python -m pytest -q tests/test_explicit_sparse_symbolic_native.py tests/test_explicit_sparse_factor_builder.py tests/test_profile_sparse_helper_coverage.py tests/test_profile_response_sparse_pc.py tests/test_rhs1_full_assembly.py tests/test_jax_ecosystem_backend_probes.py tests/test_v3_sparse_pattern.py`
  passed as `660 passed in 149.20 s`.
- `python -m ruff check sfincs_jax/solvers/explicit_sparse.py tests/test_explicit_sparse_symbolic_native.py tests/test_profile_full_system_structured_selection.py`
  passed.
- `python -m compileall -q sfincs_jax/solvers/explicit_sparse.py tests/test_explicit_sparse_symbolic_native.py tests/test_profile_full_system_structured_selection.py`
  passed.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_sparse_structured_tranche.json`
  passed as `4380 passed in 285.20 s`. Coverage is `90.3011%`
  (`6703` missing lines).
- `sfincs_jax/solvers/explicit_sparse.py` remains at `90.26%` with `243`
  missing lines; `sfincs_jax/operators/profile_full_system.py` remains at
  `85.27%` with `272` missing lines.
- `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py`
  regenerated the README runtime/memory figure and summary JSON from the
  tracked CPU/GPU suite reports.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --case QA --from-summary-json docs/_static/figures/vmec_jax_finite_beta/qs_paper_sfincs_jax_redl_comparison.json --fig-dir docs/_static/figures/vmec_jax_finite_beta --stem qs_paper_sfincs_jax_redl_comparison`
  regenerated the QA bootstrap-current plot from tracked summary data.
- `python examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py --case QH --from-summary-json docs/_static/figures/vmec_jax_finite_beta/qs_paper_qh_sfincs_jax_redl_comparison.json --fig-dir docs/_static/figures/vmec_jax_finite_beta --stem qs_paper_qh_sfincs_jax_redl_comparison`
  regenerated the QH bootstrap-current plot from tracked summary data.
- `python scripts/compare_v3_example_suite.py --fortran-exe /Users/rogeriojorge/local/sfincs/fortran/version3/sfincs --out-root /tmp/sfincs_jax_final_parity_sanity --limit 3 --fortran-timeout-s 180 -v`
  wrote JAX outputs for the first three upstream HSX examples and completed
  two Fortran comparisons with `0` common-key mismatches. The third Fortran
  solve hit the 180 s cap.
- `python scripts/compare_v3_example_suite.py --fortran-exe /Users/rogeriojorge/local/sfincs/fortran/version3/sfincs --out-root /tmp/sfincs_jax_final_parity_sanity_fulltraj --pattern '^.*/HSX_FPCollisions_fullTrajectories/input\\.namelist$|HSX_FPCollisions_fullTrajectories' --limit 1 --fortran-timeout-s 360 -v`
  reran the timed-out HSX full-trajectory case and completed with
  `ok_fortran=1`, `ok_compare_common=1`.

Status:

- PR branch consolidation is complete; continue on
  `refactor/v3-driver-architecture` only.
- The 95% meaningful package coverage gate remains open.
- The public figures are regenerated from tracked artifacts, but a fresh
  production-resolution CPU/GPU/Fortran benchmark campaign remains a separate
  compute gate before changing performance claims.
- The local SFINCS Fortran v3 executable for bounded reruns is
  `/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs`.

## Tranche 154: RHSMode=1 Progress/Profiler Wrapper Gate

Scope:

- Extended the allowed `profile_solve` wrapper-contract test to cover the
  progress/profiler branch before expensive RHSMode=1 sparse-host-safe exits.
- The new test verifies that:
  - materialization marks are forwarded to the profiler hook;
  - active-DOF PAS projection progress messages are emitted;
  - post-active policy messages are emitted;
  - GMRES tolerance/restart/maxiter and matrix-free Jacobian progress messages
    are visible before the sparse-host-safe no-solve branch returns.
- This keeps coverage on the canonical orchestration owner while preserving the
  source-tree rule that ordinary tests should import the smaller policy/helper
  owners instead of private `profile_solve` aliases.

Validation:

- `python -m pytest -q tests/test_profile_solve_module_wrappers.py` passed as
  `12 passed in 1.01 s`.
- `python -m ruff check tests/test_profile_solve_module_wrappers.py` passed.
- `python -m compileall -q tests/test_profile_solve_module_wrappers.py` passed.
- `python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py tests/test_profile_solve_module_wrappers.py`
  passed as `67 passed in 4.62 s`.
- `python -m pytest -q tests/test_profile_solve_module_wrappers.py --cov=sfincs_jax.problems.profile_solve --cov-report=term --cov-report=json:/tmp/profile_solve_wrapper_cov.json`
  aborted locally with exit code `134`, matching the known module-only coverage
  anomaly. Treat full package coverage as authoritative.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_profile_solve_progress.json`
  passed as `4381 passed in 296.17 s`. Total coverage was `90.2968%`
  (`6706` missing lines). `sfincs_jax/problems/profile_solve.py` improved from
  `318` to `315` missing lines, while `sfincs_jax/__init__.py` varied from
  `12` to `18` missing lines in the full-run coverage report.

Status:

- The new test improves direct evidence for RHSMode=1 user progress reporting
  and fail-fast sparse-host-safe routing.
- The global 95% gate remains open; coverage work must increasingly target
  larger uncovered blocks in `profile_solve.py`, `transport_solve.py`,
  `profile_full_system.py`, and the QI/preconditioner modules or remove dead
  code rather than adding narrow wrapper tests.

## Tranche 155: Review-Prep Branch Consolidation and Public Artifact Check

Scope:

- Verified branch topology before review:
  `main` and `origin/main` are ancestors of
  `refactor/v3-driver-architecture`, so there is no second divergent local
  feature branch to merge. The draft PR branch is the single branch carrying
  the refactor work.
- Rewrote remaining public documentation phrases that described the project as
  a sequence of temporary branch states instead of a standalone code release.
  The pass covered `docs/testing.rst`, `docs/performance_techniques.rst`,
  `docs/research_lanes.rst`, `docs/parallelism.rst`, and
  `docs/validation_matrix.rst`.
- Regenerated the README runtime/memory figure and canonical summary JSON from
  the tracked CPU/GPU suite reports:
  `examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  and `docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.{png,pdf}`.
- Regenerated the QA and QH bootstrap-current comparison figures from tracked
  summary JSON, without rerunning kinetic solves:
  `docs/_static/figures/vmec_jax_finite_beta/qs_paper_sfincs_jax_redl_comparison.{json,png,pdf}`
  and
  `docs/_static/figures/vmec_jax_finite_beta/qs_paper_qh_sfincs_jax_redl_comparison.{json,png,pdf}`.

Evidence:

- Branch checks:
  `git merge-base --is-ancestor main HEAD` and
  `git merge-base --is-ancestor origin/main HEAD` both returned success.
  `git log --oneline main..HEAD | wc -l` reported `1000`, and
  `git log --oneline HEAD..main | wc -l` reported `0`.
- Public benchmark summary:
  the tracked source reports contain `39` CPU and `39` GPU source cases.
  The README-facing `min_fortran_runtime_s=10` filter reports `24` CPU and
  `24` GPU rows, with `strict_mismatch_total=0` for both backends.
- Strict production-floor audit:
  `python examples/publication_figures/generate_fortran_suite_benchmark_summary.py --enforce-public-resolution-floor`
  fails closed because the tracked public reports still contain `15` below-floor
  historical rows per backend. This is an honest remaining compute gate before
  replacing public performance claims with a fully production-floor suite.
- QA/QH bootstrap-current artifacts:
  the QA tracked profile has `39/39` completed SFINCS_JAX points, maximum
  JAX-vs-Fortran relative difference `6.94%`, maximum JAX-vs-Redl relative
  difference `23.95%`, SFINCS_JAX elapsed sum `232.55 s`, and archived
  Fortran-v3 elapsed sum `696.07 s` on the same surfaces. The QH tracked profile
  has `39/39` completed SFINCS_JAX points, maximum JAX-vs-Fortran relative
  difference `18.77%`, maximum JAX-vs-Redl relative difference `15.31%`,
  SFINCS_JAX elapsed sum `261.08 s`, and archived Fortran-v3 elapsed sum
  `655.48 s`.

Validation:

- `python -m pytest -q -n auto --dist=loadscope` passed as
  `4381 passed in 262.99 s`.
- `python -m pytest -q tests/test_benchmark_doc_claims.py tests/test_generate_fortran_suite_benchmark_summary.py tests/test_validation_artifacts.py tests/test_finite_beta_vmec_example.py`
  passed as `60 passed in 2.96 s`.
- `python -m pytest -q tests/test_benchmark_doc_claims.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `62 passed in 4.60 s`.
- `python -m ruff check tests/test_profile_solve_module_wrappers.py` passed.
- `python -m compileall -q sfincs_jax tests scripts examples/getting_started examples/optimization examples/parity examples/performance examples/publication_figures examples/tutorials examples/transport examples/vmec_jax_finite_beta`
  passed. A broader `compileall examples` intentionally hits archived upstream
  Python-2 utilities under `examples/sfincs_examples`, which are excluded from
  maintained-code validation.

Status:

- The draft PR branch is ready for user review from a branch-consolidation and
  local-test perspective once this tranche is committed and pushed.
- The 95% coverage gate remains open at the previously measured `90.2968%`.
- Fresh production-floor CPU/GPU/Fortran benchmark reports remain open if the
  README performance figure must be based only on rows satisfying
  `Ntheta>=25`, `Nzeta>=51`, `Nxi>=100`, and `Nx>=4`.
- QH bootstrap current is usable as checked evidence but remains the less tight
  finite-beta comparison: it is below `20%` JAX-vs-Fortran maximum relative
  difference in the tracked profile, not a sub-10% agreement claim.

## Tranche 156: Searchable Examples Workflow Browser

Scope:

- Added `examples/list_workflows.py`, a zero-dependency terminal browser over
  `examples/workflow_catalog.json`.
- Added searchable workflow keywords for the public example catalog, covering
  CLI output, file formats, VMEC geometry, transport matrices, autodiff,
  bootstrap-current/Redl comparisons, performance, parity, optimization,
  VMEC-JAX/Boozer geometry, and publication-figure regeneration.
- Updated `examples/README.md` and `docs/examples.rst` with the browser
  commands:
  `python examples/list_workflows.py --list-topics`,
  `python examples/list_workflows.py --topic bootstrap --long`, and
  `python examples/list_workflows.py --search "VMEC geometry"`.
- Added tests that keep the workflow catalog searchable, validate the terminal
  browser in text and JSON modes, and ensure first-run workflows remain
  discoverable from the examples tree.

Validation:

- `python -m pytest -q tests/test_examples_workflow_browser.py tests/test_examples_tree_contract.py tests/test_examples_tutorials.py`
  passed as `16 passed in 0.21 s`.
- `python examples/list_workflows.py --topic bootstrap --long` printed both
  the Redl/bootstrap comparison and QA/QI objective workflows with first-run
  commands and local-Fortran requirements.
- `python examples/list_workflows.py --topic redl --json` returned a
  machine-readable payload with the `bootstrap_redl` workflow.
- `python -m json.tool examples/workflow_catalog.json` passed.
- `python -m compileall -q examples/list_workflows.py tests/test_examples_workflow_browser.py tests/test_examples_tree_contract.py`
  passed.
- `python -m ruff check examples/list_workflows.py tests/test_examples_workflow_browser.py tests/test_examples_tree_contract.py`
  passed.
- `python -m pytest -q tests/test_benchmark_doc_claims.py tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py`
  passed as `62 passed in 4.57 s`.
- `python -m sphinx -b html docs docs/_build/html` passed.
- `python -m pytest -q -n auto --dist=loadscope` passed as
  `4385 passed in 252.11 s`.
- The public wording scan remains clean except for a historical
  `examples/publication_figures/validation_manifest.json` provenance string.

Status:

- The examples tree remains folder-stable, but discoverability no longer
  depends only on reading long README tables.
- This tranche moves the examples-refactor lane forward without adding package
  API complexity or generated artifacts.
- The 95% package coverage gate is unchanged; the next code-focused tranche
  should target larger uncovered solver/operator blocks or delete dead code.

## Tranche 157: RHSMode=1 Domain-Decomposition Numerical Coverage

Scope:

- Extended `tests/test_rhs1_domain_decomposition.py` from sizing-only checks
  into numerical coverage for the RHSMode=1 domain-decomposition
  preconditioner builders.
- Added a tiny diagonal full-system operator scaffold that exercises:
  theta-line and zeta-line preconditioners, reduced active-vector
  expand/reduce contracts, theta and zeta restricted additive Schwarz,
  theta-line-xdiag blocks, full angular theta-zeta blocks, cache reuse, extra
  variable inversion, line-index maps, and pseudo-inverse fail-closed behavior
  for singular local blocks.
- The test is a numerical solver-policy gate, not a smoke test: for a diagonal
  operator with diagonal value 2, all local preconditioners must apply the
  regularized inverse and return a finite vector; for a singular local block the
  pseudo-inverse branch must return finite zeros instead of NaNs.

Validation:

- `python -m pytest -q tests/test_rhs1_domain_decomposition.py` passed as
  `14 passed in 0.96 s`.
- `python -m pytest -q tests/test_rhs1_domain_decomposition.py tests/test_domain_package_import_contracts.py tests/test_source_tree_consolidation.py`
  passed as `69 passed in 5.83 s`.
- `python -m ruff check tests/test_rhs1_domain_decomposition.py examples/list_workflows.py tests/test_examples_workflow_browser.py tests/test_examples_tree_contract.py`
  passed.
- `python -m compileall -q tests/test_rhs1_domain_decomposition.py sfincs_jax/solvers/preconditioner_domain_decomposition.py examples/list_workflows.py tests/test_examples_workflow_browser.py`
  passed.
- `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_examples_dd_tranche.json`
  passed as `4392 passed in 307.77 s`. Total package coverage is `90.3315%`
  with `6682` missing lines.

Coverage movement:

- `sfincs_jax/solvers/preconditioner_domain_decomposition.py` improved from
  `85.95%` with `85` missing lines to `89.92%` with `61` missing lines.
- Total package missing lines improved from `6706` to `6682`.
- The 95% gate remains open. The largest remaining coverage blockers are still
  `profile_solve.py` (`315` missing lines), `profile_full_system.py` (`272`),
  `transport_solve.py` (`250`), `profile_true_operator_rescue.py` (`247`), and
  `explicit_sparse.py` (`243`).

## Tranche 158: Single-Branch Evidence Refresh And PAS Production Parity Fix

Scope:

- Confirmed `origin/main` is an ancestor of
  `origin/refactor/v3-driver-architecture`; the refactor PR branch remains the
  single review branch and no second feature branch needs to be merged.
- Re-rendered the checked Fortran-suite runtime/memory figure, README audit
  block, and QA/QH bootstrap-current PNG/PDF/JSON bundles from committed
  summary artifacts. The artifacts are deterministic and produced no tracked
  figure diffs.
- Materialized a temporary 39-case production-resolution input tree and reran
  the bounded CPU production subset against local SFINCS Fortran v3 at the
  raised production floor.
- The first bounded CPU pass exposed two strict solver-derived output
  mismatches in `tokamak_1species_PASCollisions_withEr_fullTrajectories` and
  `tokamak_2species_PASCollisions_withEr_fullTrajectories`. The mismatches were
  only in flow/current diagnostics at about `1e-7` relative error; geometry and
  physics-flux buckets were clean.
- Tightened the default RHSMode=1 PAS tolerance floor for large Phi1-free PAS
  solves to `1e-8`, with `SFINCS_JAX_RHSMODE1_PAS_TOL` and
  `SFINCS_JAX_RHSMODE1_PAS_TOL_MIN_SIZE` retaining explicit user override and
  disable control. Small PAS solves keep the namelist tolerance.

Validation:

- Focused setup validation passed:
  `tests/test_profile_response_setup.py tests/test_profile_solve_module_wrappers.py`
  as `33 passed in 1.57 s`; Ruff and compile checks passed for touched files.
- Artifact/docs validation passed:
  `tests/test_profile_response_setup.py`,
  `tests/test_profile_solve_module_wrappers.py`,
  `tests/test_benchmark_doc_claims.py`,
  `tests/test_generate_fortran_suite_benchmark_summary.py`,
  `tests/test_validation_artifacts.py`,
  `tests/test_benchmark_artifact_policy.py`, and
  `tests/test_finite_beta_vmec_example.py` as `127 passed in 4.00 s`.
- The benchmark artifact reproducibility check passed for
  `examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`.
- The benchmark artifact index release gate over the tracked artifact roots
  reported `release_blocking=0`.
- The post-fix bounded CPU production rerun passed both launched cases with
  `parity_ok`, `0/195` practical mismatches, `0/195` strict mismatches, no
  missing output keys, and `8/9` print-parity signals. Runtime stayed bounded:
  about `8.0 s` JAX versus `1.7 s` Fortran for the one-species case and
  `14.5 s` JAX versus `7.0 s` Fortran for the two-species case. Peak RSS was
  about `3.1 GB` and `5.7 GB`, respectively.
- `ssh office` timed out twice during this pass, so no fresh GPU rerun was
  produced. The checked GPU report remains the current GPU evidence until a
  reachable office/cluster lane regenerates it from the same branch.

Status:

- The refactor branch is branch-consolidated and has a fresh local
  production-floor CPU parity fix for the only locally bounded production rows.
- The README-facing runtime/memory plot is still generated from the verified
  checked CPU/GPU reports; it should not be replaced by the bounded temporary
  CPU probe because the full production suite did not run.
- The full fresh 39-case CPU/GPU/Fortran production benchmark campaign remains
  the main evidence-refresh item before final merge if reviewers require all
  public performance claims to be regenerated from this branch state.

## Tranche 159: Transport Solve-Loop Branch Coverage

Scope:

- Added a bounded diagonal-operator harness to
  `tests/test_transport_solve_module_wrappers.py` so the top-level RHSMode=2/3
  transport orchestration can be tested without launching production-scale
  geometry or matrix assembly.
- Covered three production-relevant solver-policy branches in
  `solve_v3_transport_matrix_linear_gmres`:
  host SciPy GMRES first-attempt failure followed by the normal matrix-free
  fallback, host sparse-LU first attempt with safe containment of state-write
  failures, and dense true-residual fallback acceptance when the Krylov
  candidate misses the requested residual gate.
- Kept the tranche inside an existing test file and avoided new package files,
  so it supports the simplification/refactor goal rather than expanding the
  public source tree.

Validation:

- `python -m pytest -q tests/test_transport_solve_module_wrappers.py` passed
  as `8 passed in 0.54 s`.
- `python -m ruff check tests/test_transport_solve_module_wrappers.py` passed.
- `python -m pytest -q tests/test_transport_solve_module_wrappers.py
  tests/test_transport_linear_solve.py tests/test_transport_solve_finalization.py
  tests/test_transport_loop_support.py tests/test_transport_postsolve_diagnostics.py`
  passed as `62 passed in 7.40 s`.
- `python -m pytest -q tests/test_transport_solve_module_wrappers.py
  tests/test_transport_linear_solve.py tests/test_transport_solve_finalization.py
  tests/test_transport_loop_support.py tests/test_transport_postsolve_diagnostics.py
  tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py
  tests/test_benchmark_doc_claims.py` passed as `124 passed in 12.36 s`.
- `python -m compileall -q tests/test_transport_solve_module_wrappers.py
  sfincs_jax/problems/transport_solve.py sfincs_jax/problems/transport_finalize.py
  sfincs_jax/problems/transport_linear_system.py` passed.
- A package-level coverage probe for just this wrapper file reported
  `sfincs_jax/problems/transport_solve.py` at `41.81%` within that isolated
  run; the project-wide 95% gate still requires the full coverage campaign.

Status:

- This tranche closes several untested solver fallback branches while keeping
  the source layout stable.
- The next coverage/refactor tranche should target either
  `profile_full_system.py` branch-heavy operator assembly paths or the
  explicit sparse/native-factor modules, since these remain among the largest
  meaningful coverage blockers.

## Tranche 160: Native Sparse Factor Edge-Case Coverage

Scope:

- Added a compact symbolic-native sparse test covering finite fallback behavior
  in the bounded sparse factor stack:
  no-separator block-Schur solves, empty BLR Schur factors, invalid Woodbury
  metadata falling through to Krylov, and GMRES exceptions falling back to the
  base sparse factor.
- Kept the test inside `tests/test_explicit_sparse_symbolic_native.py`; no
  source files or package structure were changed.
- The numerical gate is intentionally bounded but meaningful for the
  production-memory lane: native sparse factors must fail soft and return finite
  vectors because they are preconditioners admitted by true residual checks,
  not the final correctness gate.

Validation:

- `python -m pytest -q tests/test_explicit_sparse_symbolic_native.py
  tests/test_explicit_sparse.py` passed as `89 passed in 0.62 s`.
- `python -m ruff check tests/test_explicit_sparse_symbolic_native.py` passed.
- `python -m compileall -q tests/test_explicit_sparse_symbolic_native.py
  sfincs_jax/solvers/explicit_sparse.py` passed.
- `python -m pytest -q tests/test_explicit_sparse_symbolic_native.py
  tests/test_explicit_sparse.py tests/test_explicit_sparse_factor_builder.py
  tests/test_explicit_sparse_symbolic_native.py tests/test_explicit_sparse_factor_policy.py
  tests/test_transport_solve_module_wrappers.py tests/test_source_tree_consolidation.py
  tests/test_domain_package_import_contracts.py` passed as `164 passed in
  5.70 s`.

Status:

- This closes additional native sparse fail-soft branches with negligible CI
  cost.
- The 95% coverage gate remains open; a fresh full package coverage audit is
  required after this tranche is committed to measure exact movement.

## Tranche 161: Current-Head Bounded Production Evidence

Scope:

- Re-ran the locally bounded production-resolution CPU/Fortran comparison on
  the current PR head after the sparse and transport coverage tranches.
- The rerun used the generated production input manifest under `/tmp` and did
  not replace the public README plot, since the README figure is generated from
  the checked full-suite CPU/GPU reports rather than this two-case local probe.
- The branch remains consolidated: `origin/main` is an ancestor of
  `refactor/v3-driver-architecture`, and no second active development branch is
  required for review.

Validation:

- `tokamak_1species_PASCollisions_withEr_fullTrajectories` passed with
  `parity_ok`: JAX `9.19 s`, SFINCS Fortran v3 `1.93 s`, JAX peak RSS
  `3.09 GB`, Fortran peak RSS `0.71 GB`.
- `tokamak_2species_PASCollisions_withEr_fullTrajectories` passed with
  `parity_ok`: JAX `15.56 s`, SFINCS Fortran v3 `3.45 s`, JAX peak RSS
  `5.69 GB`, Fortran peak RSS `1.34 GB`.
- The focused release-review test bundle passed as `213 passed in 13.83 s`.
- `python -m ruff check sfincs_jax tests` passed.
- `python -m compileall -q sfincs_jax tests` passed.
- `python -m sphinx -W -b html docs docs/_build/html` passed.

Status:

- Local CPU correctness/parity evidence for the bounded slow PAS cases is clean
  at the current PR head.
- A fresh GPU rerun was not produced in this tranche because `ssh office`
  timed out; the checked GPU reports remain the public GPU evidence until a GPU
  host is reachable.
- The remaining review gates are CI completion, optional fresh GPU reruns, and
  the longer-term 95% coverage target.

## Tranche 162: Active Full-System Preconditioner Dispatch Coverage

Scope:

- Extended `tests/test_profile_full_system_structured_selection.py` to cover
  the active RHSMode=1 full-CSR preconditioner dispatcher without adding any
  production-scale solves.
- Added alias and fail-closed checks for the active diagonal Schur, sparse
  coarse, Fortran-v3 reduced, low-l, ell-band, xell-window, coupled kinetic,
  filtered sparse-factor, symbolic frontal/superblock/block/coupled Schur,
  native-stack, native xell/angular/multiline sparse-coarse, global Schur,
  Schwarz, angular-line, native-indexed-Schwarz, xblock, and ILU-coarse
  families.
- Added builder-routing checks that monkeypatch the corresponding builder
  hooks and verify the normalized requested kind, memory cap, and active-index
  payloads. This guards solver-policy refactors without coupling the tests to
  expensive factorization setup.
- Added explicit ILU budget and unsupported-kind checks so active sparse
  preconditioners fail closed when memory admission rejects a candidate.

Validation:

- `python -m pytest -q tests/test_profile_full_system_structured_selection.py`
  passed as `53 passed in 0.38 s`.
- `python -m pytest -q tests/test_profile_full_system_structured_selection.py
  tests/test_rhs1_full_assembly.py tests/test_v3_sparse_pattern.py
  tests/test_profile_response_sparse_pc.py::test_direct_tail_structured_build_uses_direct_reduced_pmat_builder
  tests/test_profile_response_sparse_pc.py::test_direct_tail_structured_admission_allows_direct_reduced_pmat_without_bundle`
  passed as `319 passed in 141.86 s`.
- `python -m ruff check sfincs_jax tests`, `python -m compileall -q
  sfincs_jax tests`, and `git diff --check` passed.
- Full package coverage validation passed:
  `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax
  --cov-report=term --cov-report=json:/tmp/sfincs_jax_coverage_after_active_pc_dispatch.json`
  as `4445 passed in 279.75 s`.

Coverage movement:

- `sfincs_jax/operators/profile_full_system.py` improved from `272` missing
  lines to `259` missing lines.
- Total package missing lines improved from `6632` to `6619`.
- Total package coverage remains about `90%`; the 95% target remains open.

Status:

- This tranche strengthens the active solver-policy contract and measurably
  advances the coverage target without increasing package source complexity.
- The next high-impact coverage targets remain `profile_solve.py`,
  `profile_true_operator_rescue.py`, `transport_solve.py`,
  `transport_linear_system.py`, and the larger preconditioner owners.

## Latest Execution Log: True-Operator Residual Rescue Edges

What changed:

- Added direct tests for the RHSMode=1 true-operator residual-window rescue
  helpers: empty-window fallback, malformed one-hot cache inputs, sparse-graph
  edge cases, duplicate/empty explicit window specs, and setup-time size/budget
  rejection messages.
- Revalidated the README-facing Fortran/JAX runtime and memory plot pipeline
  from the tracked CPU/GPU suite reports. The benchmark JSON passed the release
  artifact policy, the figure regenerated cleanly, and the README audit block
  regenerated with no tracked artifact churn.

Validation:

- `python -m pytest -q tests/test_rhs1_true_operator_rescue.py
  tests/test_rhs1_sparse_rescue_policy.py
  tests/test_profile_response_sparse_pc.py::test_direct_tail_structured_build_uses_direct_reduced_pmat_builder
  tests/test_profile_response_sparse_pc.py::test_direct_tail_structured_admission_allows_direct_reduced_pmat_without_bundle
  tests/test_profile_response_diagnostics.py::test_sparse_pc_direct_tail_result_metadata_preserves_driver_conversions`
  passed as `47 passed in 0.48 s`.
- `python -m pytest -q tests/test_generate_fortran_suite_benchmark_summary.py
  tests/test_benchmark_doc_claims.py tests/test_source_tree_consolidation.py
  tests/test_examples_tree_contract.py` passed as `64 passed in 6.15 s`.
- `python scripts/check_benchmark_artifacts.py
  examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json`
  passed.
- `python -m ruff check sfincs_jax tests`, `python -m compileall -q
  sfincs_jax tests`, and `git diff --check` passed.
- The full local coverage audit completed and wrote
  `/tmp/sfincs_jax_coverage_after_true_rescue_edges.json`; pytest's
  `lastfailed` cache was empty after the run.

Coverage movement:

- `sfincs_jax/operators/profile_true_operator_rescue.py` improved from `247`
  missing lines to `228` missing lines.
- Total package missing lines improved from `6619` to `6601`.
- Total package coverage improved from `90.422%` to `90.448%`.

Benchmark/parity status from the regenerated public artifact:

- Source suite rows remain `39` CPU and `39` GPU.
- README-facing rows remain `24` CPU and `24` GPU after the `10 s` minimum
  SFINCS Fortran v3 reference-runtime filter.
- Public rows are all `parity_ok`, with zero practical and zero strict common
  output mismatches.

Status:

- The two-branch concern is resolved: `origin/main` is an ancestor of
  `refactor/v3-driver-architecture`, and PR #8 is the single review branch.
- The local plot/table regeneration path is reproducible from the tracked
  suite reports, but a genuinely fresh CPU/GPU/Fortran production rerun remains
  an expensive compute gate before replacing the frozen reports.
- The refactor branch remains not complete: the 95% meaningful coverage target,
  fresh production CPU/GPU benchmark regeneration, and final PR review cleanup
  are still open.

## Latest Execution Log: Transport Dense Batch and Block-Schur Guards

What changed:

- Added direct transport dense-batch tests for disabled-backend admission,
  RHS3-special-Krylov admission rejection, active-DOF dense solves, streaming
  output collection, residual reporting, and mixed-precision dense refinement.
- Expanded active block-Schur numerical guard tests to cover invalid angular
  block orderings, singular block pseudo-inverse fallback, and fail-closed
  zero-residual coarse-correction setup.

Validation:

- `python -m pytest -q tests/test_transport_linear_solve.py` passed as
  `28 passed in 7.40 s` after the final guard additions.
- `python -m ruff check tests/test_transport_linear_solve.py`,
  `python -m compileall -q tests/test_transport_linear_solve.py
  sfincs_jax/problems/transport_linear_system.py`, and `git diff --check`
  passed.
- Full package coverage validation after the dense-batch additions passed:
  `python -m pytest -q -n auto --dist=loadscope --cov=sfincs_jax
  --cov-report=term
  --cov-report=json:/tmp/sfincs_jax_coverage_after_transport_dense_mixed.json`
  as `4452 passed in 281.47 s`.

Coverage movement from the full audit:

- `sfincs_jax/problems/transport_linear_system.py` improved from `210`
  missing lines to `205` missing lines after the dense-batch additions.
- Total package missing lines improved from `6601` to `6595`.
- Total package coverage improved from `90.448%` to `90.457%`.

Status:

- The pushed PR head before this tranche (`ebc0b06a`) had all GitHub checks
  green and merge state `CLEAN`.
- This tranche strengthens real transport-matrix dense and block-Schur
  numerical behavior without changing production source code.
- The 95% coverage target remains open; the next higher-ROI coverage targets
  remain `profile_solve.py`, `transport_solve.py`,
  `preconditioner_xblock_tz_sparse.py`, and the larger QI/PAS preconditioner
  owners.

## Latest Execution Log: Single-Branch Finalization and Artifact Refresh

What was checked:

- Confirmed the branch topology is a single PR line: `main` is an ancestor of
  `refactor/v3-driver-architecture`, and PR #8 is the only open review branch.
- Regenerated the README/docs SFINCS Fortran v3 versus `sfincs_jax` CPU/GPU
  runtime-memory plot and summary JSON from the checked CPU/GPU suite reports.
- Regenerated the artifact-backed publication figures for the validation
  dashboard, high-collisionality trend proxy, Simakov-Helander audit, W7-X
  high-`nu` performance, autodiff validation, and QA/QH bootstrap-current
  comparisons.
- Rebuilt the README audit block from the checked CPU/GPU suite reports.

Benchmark and parity evidence:

- Checked suite reports remain `39/39` CPU rows and `39/39` GPU rows with
  zero practical and zero strict common-output mismatches.
- README-facing runtime/memory rows remain `24` CPU and `24` GPU after the
  documented `10 s` minimum SFINCS Fortran v3 runtime filter.
- The public benchmark summary records median runtime ratios of `0.0213x`
  CPU cold, `0.0152x` CPU warm/logged, `0.0367x` GPU cold, and `0.0303x`
  GPU warm/logged versus SFINCS Fortran v3 on the filtered public rows.
- Live bounded Fortran sanity with
  `/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs` passed:
  `quick_2species_FPCollisions_noEr`,
  `HSX_FPCollisions_DKESTrajectories`,
  `HSX_FPCollisions_fullTrajectories` with a `360 s` Fortran budget, and
  `HSX_PASCollisions_DKESTrajectories` all wrote JAX output, ran Fortran, and
  compared cleanly.
- A live `transportMatrix_geometryScheme2` RHSMode=2 probe passed the
  production-suite comparator (`rtol=5e-4`, `atol=1e-9`). The same run shows
  expected `1e-5`-level flow/current differences under the stricter generic
  `1e-10` comparator, so it is not a release parity blocker.

Validation:

- `python -m pytest -q tests/test_generate_fortran_suite_benchmark_summary.py
  tests/test_benchmark_doc_claims.py tests/test_benchmark_artifact_policy.py
  tests/test_audit_suite_runtime_drift.py` passed as `51 passed`.
- `python -m pytest -q tests/test_validation_figures.py
  tests/test_generate_w7x_high_nu_performance.py
  tests/test_generate_fortran_suite_benchmark_summary.py
  tests/test_benchmark_doc_claims.py tests/test_finite_beta_vmec_example.py`
  passed as `73 passed`.
- `python -m pytest -q tests/test_source_tree_consolidation.py
  tests/test_domain_package_import_contracts.py tests/test_examples_tutorials.py
  tests/test_validation_artifacts.py tests/test_solver_path_artifacts.py`
  passed as `99 passed`.
- `python -m pytest -q -n auto --dist=loadscope` passed as
  `4452 passed in 264.42 s`.
- `python -m pytest -q` for the CI external-data and optional ecosystem gate
  subset passed as `20 passed`.
- `python -m sphinx -b html docs /tmp/sfincs_jax_docs_html` succeeded.
- `python -m ruff check sfincs_jax tests`, `python -m compileall -q
  sfincs_jax tests`, and `git diff --check` passed.
- GitHub PR #8 checks are green and the merge state is `CLEAN`.

Status:

- A fresh remote GPU rerun was not launched because `ssh office` timed out
  while connecting to `plasmaworkstation.physics.wisc.edu:3281`.
- The final PR review state is technically consistent with the checked
  CPU/GPU reports and bounded live Fortran sanity. A completely fresh 39-case
  CPU/GPU/Fortran production rerun remains an expensive replacement-report
  activity, not a prerequisite for reviewing the refactor diff.

## Latest Execution Log: GPU RHSMode=1 Dense-Route Efficiency Tranche

What was checked:

- Confirmed PR #8 remained on a clean branch head with green GitHub checks.
- Retried `ssh office`; the host still timed out while connecting to
  `plasmaworkstation.physics.wisc.edu:3281`, so no live GPU run was launched
  in this tranche.
- Re-read the checked GPU solver-path artifacts for bounded full-FP
  RHSMode=1 cases. They show the same qualitative bottleneck reported by
  collaborators: when a moderate accelerator run falls into Krylov/probe work
  before dense fallback, runtime and RSS can be worse than the direct
  dense-host path.

Source change:

- `rhs1_host_dense_shortcut_allowed` now lets bounded accelerator full-FP
  RHSMode=1 systems use the same dense fallback budget as the default host
  dense shortcut cap instead of the old hard-coded `900` unknown cap.
- The change keeps the existing safety guards: no PAS, no `Phi1`, no implicit
  differentiable lane, no CPU behavior change, no explicit `dense` override
  rewrite, and explicit environment variables can still lower or disable the
  shortcut.
- CLI progress text now reports `accelerator FP bounded system -> using host
  dense shortcut`, which better matches the widened but still bounded policy.
- The output-writer solver selector uses the same `bounded system` wording, and
  `scripts/summarize_solver_paths.py` recognizes both old and new dense-auto
  log messages so existing artifacts remain parseable.

Validation:

- `python -m pytest -q tests/test_rhs1_host_policy.py
  tests/test_rhs1_sparse_first_heuristic.py tests/test_profile_response_dense.py`
  passed as `140 passed`.
- `python -m pytest -q tests/test_rhs1_host_policy.py
  tests/test_rhs1_sparse_first_heuristic.py tests/test_profile_response_dense.py
  tests/test_io_output_policy_coverage.py` passed as `236 passed`.
- After the output-selector/parser cleanup, `python -m pytest -q
  tests/test_io_output_policy_coverage.py tests/test_profile_response_dense.py
  tests/test_rhs1_host_policy.py tests/test_rhs1_sparse_first_heuristic.py
  tests/test_summarize_solver_paths.py` passed as `238 passed`.
- `python -m ruff check sfincs_jax/problems/profile_policies.py
  sfincs_jax/problems/profile_dense.py tests/test_rhs1_host_policy.py
  tests/test_rhs1_sparse_first_heuristic.py tests/test_profile_response_dense.py`
  passed.
- `python -m compileall -q sfincs_jax/problems/profile_policies.py
  sfincs_jax/problems/profile_dense.py tests/test_rhs1_host_policy.py
  tests/test_rhs1_sparse_first_heuristic.py tests/test_profile_response_dense.py`
  passed.
- `git diff --check` passed.

Status:

- This closes the local policy implementation portion of the GPU-efficiency
  target for bounded full-FP RHSMode=1 systems.
- A live office GPU rerun remains the next evidence step once SSH is reachable:
  rerun the neighboring `Nxi=20/40` full-FP cases and confirm the default route
  uses the host-dense shortcut with lower wall time and no parity drift.

## Latest Execution Log: Examples Navigation Category Tranche

What was checked:

- Rechecked the package source tree: it remains one level deep under
  `sfincs_jax/` except ignored `__pycache__` folders, and
  `sfincs_jax/README.md` documents the current package ownership map.
- Re-ran the public stale-wording scan for README/docs/examples wording such as
  `On the current main branch`, `new version`, and `currently`; the configured
  public paths returned no matches.
- Audited `examples/`: the tree already has a workflow catalog and topic
  READMEs, but the top-level folders still mixed first-pass learning folders
  with reference/support folders in a way that could confuse new users.

Source/docs change:

- Added a `category` field to every top-level folder in
  `examples/workflow_catalog.json`: `learning`, `capability`, `validation`,
  `reference`, or `support`.
- Updated `examples/list_workflows.py --list-topics` to print those categories
  so users can distinguish tutorials and capability workflows from vendored
  upstream/reference/support folders at the terminal.
- Added a `Top-Level Folder Categories` section to `examples/README.md` and
  the matching `Top-level folder categories` section to `docs/examples.rst`.
- Strengthened `tests/test_examples_tree_contract.py` so future example folders
  must have an intentional category and the README/docs must describe it.

Validation:

- `python -m pytest -q tests/test_examples_tree_contract.py
  tests/test_examples_workflow_browser.py tests/test_examples_tutorials.py
  tests/test_getting_started_examples.py` passed as `25 passed`.
- `python -m ruff check examples/list_workflows.py
  tests/test_examples_tree_contract.py tests/test_examples_workflow_browser.py`
  passed.
- `python -m compileall -q examples/list_workflows.py
  tests/test_examples_tree_contract.py tests/test_examples_workflow_browser.py`
  passed.
- `git diff --check` passed.

Status:

- This moves the examples-refactor lane toward review readiness by making the
  current folder structure explicit and enforceable instead of relying on prose.
- The remaining examples work is not a broad folder move; it is periodic
  pruning of stale scripts or generated artifacts if the strengthened tests
  expose them.

## Latest Execution Log: Public Facade And Path-Resolver Coverage Tranche

What was checked:

- Reviewed the remaining root compatibility facades and the public path
  resolver because these are critical during refactoring: source modules can
  move only if old imports and equilibrium path lookup remain stable.
- Confirmed existing tests covered many direct I/O paths but did not explicitly
  test `sfincs_jax.io` private-compatibility delegation or release-data fallback
  resolution from `resolve_existing_path`.

Source/test change:

- Added `tests/test_public_facades_and_paths.py`.
- New tests cover:
  - relative and stale-absolute equilibrium resolution through
    `extra_search_dirs`,
  - release-hosted equilibrium resolution through the external data resolver,
  - failure diagnostics that preserve attempted paths,
  - `sfincs_jax.io` compatibility `__getattr__` and `__setattr__` delegation to
    domain-owned output modules,
  - fail-closed `AttributeError` behavior for unknown compatibility names.

Validation:

- `python -m pytest -q tests/test_public_facades_and_paths.py
  tests/test_helper_module_coverage.py tests/test_output_formats.py
  tests/test_api_contracts.py` passed as `38 passed`.
- `python -m ruff check tests/test_public_facades_and_paths.py` passed.
- `python -m compileall -q tests/test_public_facades_and_paths.py` passed.
- `git diff --check` passed.

Status:

- This improves meaningful coverage around low-cost public contracts that guard
  the refactor, without adding solve-heavy CI time.
- The 95% coverage lane remains open; next high-value coverage tranches should
  continue to target public contracts and compact policy/helper modules before
  attempting any solve-heavy coverage expansion.

## Latest Execution Log: Bounded GPU Dense-Route Performance Tranche

What was checked:

- Reviewed the RHSMode=1 full-FP GPU auto-selection path in
  `outputs/rhsmode1.py`, `problems/profile_solve.py`, and
  `problems/profile_dense.py`.
- Confirmed the execution layer already has real reduced/full host-dense
  shortcut stages that skip preconditioner construction and Krylov probes for
  bounded accelerator full-FP systems.
- Found that the policy helper and output selector could still promote moderate
  accelerator systems directly to `solve_method="dense"` before those shortcut
  stages, which obscures solver provenance and can route through accelerator
  dense/Krylov machinery unless accelerator dense linear algebra is explicitly
  intended.

Source/test change:

- Changed `rhs1_dense_auto_fp_allowed` so default accelerator runs no longer
  auto-promote to backend dense LU unless
  `SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR=1` is set.
- Changed `select_rhsmode1_solve_method` so the accelerator full-FP bounded
  window falls through to the explicit host-dense shortcut branch instead of
  returning `dense`.
- Updated `scripts/summarize_solver_paths.py` to report host-dense shortcut
  usage as a distinct solver-path provenance field and markdown column, so
  future GPU benchmark summaries do not hide this route as generic auto/Krylov.
- Added stable host-dense shortcut metadata (`solver_path=host_dense_shortcut`,
  `solver_kind=host_dense_lu`, shortcut backend/size/system) at the reduced and
  full shortcut execution sites so solver traces and HDF5 metadata can be
  audited without parsing logs.
- Updated policy/selector tests to require CPU dense auto, default GPU
  host-shortcut routing, opt-in accelerator dense behavior, and host-shortcut
  solver-path summarization.

Validation:

- `python -m pytest -q tests/test_rhs1_host_policy.py
  tests/test_io_output_policy_coverage.py tests/test_profile_response_dense.py
  tests/test_rhs1_sparse_first_heuristic.py` passed as `236 passed in 3.48 s`.
- `python -m pytest -q tests/test_summarize_solver_paths.py` passed as
  `3 passed`.
- `python -m pytest -q tests/test_profile_response_dense.py
  tests/test_profile_response_finalization.py tests/test_summarize_solver_paths.py`
  passed as `51 passed`.
- `python -m pytest -q tests/test_gpu_solver_path_artifacts.py
  tests/test_solver_path_artifacts.py tests/test_benchmark_case_variants.py
  tests/test_benchmark_pas_tz_memory_fallback.py` passed as `52 passed in
  10.72 s`.
- A minimal `ssh office` GPU probe was attempted but timed out at the configured
  SSH host/port before any remote command produced output. Remote GPU evidence
  still needs a reachable session.

Status:

- This tightens the default GPU path for bounded full-FP RHSMode=1 cases without
  changing CPU behavior, differentiable implicit paths, explicit user solver
  overrides, or the strict residual gates.
- The next GPU step is one bounded office-GPU provenance run once SSH connects,
  checking that the solver log records the host-dense shortcut and that the
  output residual/parity gate remains clean.

## Latest Execution Log: Public Wording Contract Tranche

What was checked:

- Re-ran the existing README/docs/examples/source guards after the GPU route
  changes.
- Confirmed the manual public wording scan found no current README, docs, or
  examples matches for rejected branch-history phrases such as
  `On the current main branch`, `new version`, `new benchmarks`,
  `currently`, and the production-manifest progress-log wording.
- Identified that this broad scan was only documented as a manual review
  command, while the existing docs tests covered a narrower curated page list
  and docs-tree phrase list.

Source/test change:

- Added `tests/test_public_docs_wording_contract.py`.
- The test scans tracked public text under `README.md`, `sfincs_jax/README.md`,
  `docs/`, and `examples/` for the rejected branch-history/progress-log
  fragments.
- It intentionally excludes release notes, upstream/vendor examples, static
  artifacts, generated run outputs, and provenance/output folders so public
  prose is guarded without failing on archived evidence or binary artifacts.

Validation:

- `python -m pytest -q tests/test_public_docs_wording_contract.py
  tests/test_benchmark_doc_claims.py tests/test_examples_tree_contract.py
  tests/test_source_tree_consolidation.py` passed as `58 passed in 4.73 s`.
- `python -m ruff check tests/test_public_docs_wording_contract.py` passed.
- `python -m compileall -q tests/test_public_docs_wording_contract.py` passed.
- `git diff --check` passed.

Status:

- This closes a review-readiness gap for the README/docs/examples wording lane:
  the public text is now guarded as self-contained software documentation rather
  than relying on an untracked manual scan.
- The documentation/examples lane still needs a final rendered-docs pass after
  benchmark evidence regeneration, but stale public wording is now CI-visible.

## Latest Execution Log: GPU Host-Dense Memory Admission Tranche

What was checked:

- Revisited the bounded accelerator RHSMode=1 full-FP path after the
  host-dense shortcut promotion.
- Confirmed the previous tranche avoided accidental GPU dense LU by routing
  moderate accelerator FP systems through an explicit host-dense shortcut, but
  the shortcut was still admitted only by active-size count.
- Retried `ssh office` twice for live GPU validation; both attempts timed out
  on the configured SSH host/port before any remote command could run.

Source/test change:

- Added a byte-level admission guard for `rhs1_host_dense_shortcut_allowed`.
  The default ceiling is `1.5e9` bytes and the estimate includes dense matrix,
  factor-overhead, and vector work storage. Experts can set
  `SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX_BYTES=0` for a diagnostic run,
  but the default route now fails closed if a raised active-size cap would imply
  excessive dense host storage.
- Added stable shortcut memory provenance fields to
  `rhs1_host_dense_shortcut_metadata`:
  `host_dense_shortcut_estimated_nbytes` and
  `host_dense_shortcut_max_nbytes`.
- Added focused tests for the memory cap and metadata schema.

Validation:

- `python -m pytest -q tests/test_rhs1_host_policy.py
  tests/test_profile_response_dense.py tests/test_io_output_policy_coverage.py`
  passed as `167 passed in 3.26 s`.
- `python -m ruff check sfincs_jax/problems/profile_policies.py
  sfincs_jax/problems/profile_dense.py tests/test_rhs1_host_policy.py
  tests/test_profile_response_dense.py` passed.
- `python -m compileall -q sfincs_jax/problems/profile_policies.py
  sfincs_jax/problems/profile_dense.py tests/test_rhs1_host_policy.py
  tests/test_profile_response_dense.py` passed.

Status:

- This is a concrete GPU-efficiency safeguard: bounded accelerator full-FP
  systems can still skip GPU dense scratch and Krylov/probe setup, but the host
  shortcut cannot silently become a high-memory route when active-size caps are
  raised.
- The next GPU step remains a live office-GPU run once SSH is reachable, using
  solver traces to verify shortcut provenance, residual cleanliness, and memory
  behavior on the target hardware.

## Standard Validation Commands

Use focused checks after each tranche:

```bash
python -m pytest tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py -q --tb=short
python -m pytest tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py -q --tb=short
python -m ruff check <touched files>
python -m compileall -q sfincs_jax <touched tests>
git diff --check
```

Use the full coverage audit only after a meaningful coverage bundle:

```bash
python -m pytest -q -n auto --dist=loadscope \
  --cov=sfincs_jax --cov-report=term-missing --cov-report=json:coverage.json
```

Use the public wording scan before review:

```bash
rg -n "On the current main branch|not replacements for the production-resolution gates|The production benchmark manifest|not a public performance row|current main|new benchmarks|At the moment|new version|previous version|\\bpreviously\\b|now supports|now has|now includes|\\bcurrently\\b" \
  README.md sfincs_jax/README.md examples docs \
  --glob '!docs/_build/**' \
  --glob '!docs/release_notes.rst' --glob '!docs/upstream/**' \
  --glob '!docs/_static/**' --glob '!docs/ntx_*.rst' \
  --glob '!examples/sfincs_examples/**' --glob '!examples/**/output/**' \
  --glob '!examples/**/artifacts/**' --glob '!examples/**/provenance/**'
```

## Completion Gates

The plan is complete only when all gates pass:

- The source tree remains one-level deep under `sfincs_jax/`.
- Root modules are documented stable public entry points or explicit
  compatibility facades.
- `sfincs_jax/README.md`, root README, examples README, docs API pages, and
  testing docs describe the same structure.
- Examples are task-oriented, pedagogical, and runnable within their documented
  budgets.
- Meaningful package coverage reaches 95%.
- CI stays under 10 minutes.
- CPU/GPU/Fortran parity, runtime, memory, and bootstrap-current evidence are
  regenerated from the final branch state.
- Public performance claims use fresh reports with solver provenance.
- PR #8 is review-ready and contains no generated clutter.

## Explicit Deferred Items

These are not blockers for the refactor PR unless a regression is found:

- A fully native MUMPS/SuperLU_DIST-equivalent sparse direct solver stack.
- Further lower-memory production preconditioner research after current
  correctness gates are stable.
- Full production-grid QA/QH performance parity with SFINCS Fortran v3 when
  residuals and outputs are correct but runtime or memory remain worse.
- Additional device-QI research beyond the documented residual floor.
