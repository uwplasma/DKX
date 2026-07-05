# SFINCS_JAX Final Review Plan

Last updated: 2026-07-05

Active branch: `refactor/rhs1-full-assembly-preconditioners`

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
- The 2026-07-05 user-facing documentation wording pass removed active prose
  labels such as "latest snapshot", "current release", "previous best", and
  progress-log phrasing from README-facing docs pages. The active-doc scan only
  retains false positives for "concurrently"; Sphinx `-W` and docs contract
  tests passed after the edit.
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
  bookkeeping, auto-preflight retry no-op state preservation, residual
  candidate acceptance, residual-correction no-op routing, true-coupled coarse
  no-op routing, generic x-block backend deferral, direct-tail host-factor
  fallback, direct-tail rescue-policy defaults, and the Fortran-reduced x-block
  capability guard. A source/test audit for `problems/profile_sparse_solve.py`
  reports zero production-used public helpers without direct tests. The broader
  sparse-PC/source-guard bundle passed as `375 passed in 5.80 s`.
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
- Phase C sparse-owner cleanup renamed the internal RHSMode-1 sparse-PC
  orchestration owner from `problems/profile_sparse_handoff.py` to
  `problems/profile_sparse_solve.py`. The old file path is guarded absent by
  source-tree tests, and docs/tests now point at the canonical solve owner.
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
  setup, policy, diagnostics, sparse-handoff, and finalization helpers into
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
   "handoff" names in final implementation files.
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

Status: 98% for final review readiness.

Goal: finish the PR with a smaller, clearer source tree without changing
physics, outputs, tolerances, solver defaults, differentiable Python paths,
non-autodiff CLI fast paths, CPU/GPU behavior, or parity gates.

Latest AST audit:

- Folder depth is no longer the blocker: the package has one-level domain
  folders only and no `__init__.py`-only source packages.
- The source tree has 120 Python files, 16 package-root modules, and one-level
  domain folders only. The remaining structural blockers are file-family sprawl
  and owner size.
  The largest retained owners are `problems/profile_policies.py` (`7936`
  lines), `problems/profile_sparse_xblock.py` (`7681` lines),
  `operators/profile_full_system.py` (`6130` lines),
  `problems/profile_sparse_solve.py` (`5168` lines),
  `solvers/preconditioner_qi_device.py` (`5433` lines),
  `solvers/explicit_sparse.py` (`5198` lines),
  `problems/profile_sparse_qi.py` (`4873` lines),
  `problems/profile_solve.py` (`4351` lines), and
  `outputs/writer.py` (`2675` lines).
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
- Tranche 15: renamed the RHSMode=1 sparse-PC orchestration owner from
  `profile_sparse_handoff.py` to `profile_sparse_solve.py`, updated internal
  imports, docs, API references, and import contracts, and added a source-tree
  guard so the historical filename is not reintroduced.
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

Remaining consolidation steps:

1. Compatibility cleanup: keep deleted-facade tests and stale-import scans in
   place. Behavior tests should import canonical domain owners; no
   `sfincs_jax.v3_driver` imports are allowed.
2. Problem-family consolidation: remove "handoff" and production-campaign
   names from implementation files by merging RHSMode-1 sparse setup/rescue
   owners into canonical profile sparse owners. The sparse-solve compatibility
   namespace and Ruff waiver are closed; any remaining cleanup should reduce
   owner size or clarify orchestration boundaries without reintroducing broad
   reexport surfaces. Success is fewer broad namespace surfaces, simpler
   canonical imports, and unchanged RHSMode-1 policy/output tests.
3. Solver-family consolidation: merge same-family preconditioner modules only
   at durable physics/numerics boundaries. Success is fewer solver files and no
   loss of targeted preconditioner tests, not a single oversized grab-bag file.
4. Operator cleanup: merge small profile-response term helpers only when
   equation-to-file mapping remains clear in docs and tests. Otherwise retain
   them as pedagogical owners.
5. Review lock: keep `io.py` below `80` lines and implementation-free, update
   `sfincs_jax/README.md`, and run source-layout, import-contract, docs,
   examples, CLI/output, and focused behavior guards.

### Lane 2 - Coverage And Future-Proof Tests

Status: 86% local CI-mode measured package coverage
(`3938 passed, 195 skipped, 5 stale-manifest failures before the manifest-path
fix, in 17:22` on 2026-07-05) with the direct public contract audit closed at
`modules_with_missing 0`.

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

Status: 90%.

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

Status: blocked until the source/test structure is stable.

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
   profile sparse owners. The historical sparse handoff filename has been
   replaced by `profile_sparse_solve.py`; remaining work is reducing file count
   and line count inside the sparse-profile family.
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
