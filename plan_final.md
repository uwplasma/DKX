# SFINCS_JAX Final Review Plan

Last updated: 2026-06-27

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

- `sfincs_jax/v3_driver.py` is a 47-line compatibility facade with no physics
  or solver implementation.
- The package root contains only public entry points, stable support APIs, and
  compatibility facades.
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
  rules.
- `examples/README.md` and `docs/examples.rst` provide task-oriented example
  navigation, including tutorial notebooks and runnable scripts.
- The latest local xdist coverage audit measured `88%` package coverage:
  `3995 passed in 293.58 s` with `8412` missing executable lines.
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
- The root README runtime/memory summary no longer carries branch-history or
  benchmark-process phrasing; detailed audit and regeneration procedures belong
  in the performance, parity, and Fortran-example docs.
- The public stale-wording scan is clean for README, source-layout README,
  examples README, and non-historical docs. The archived NTX handoff page uses
  standalone solver-policy wording rather than progress-log phrasing; focused
  examples/benchmark docs validation passed as `13 passed in 0.14 s`.
- The CI coverage floor remains lower than the final target until measured
  margin is available; the review target is `95%` meaningful package coverage
  while keeping GitHub Actions under 10 minutes.

The largest coverage blockers from the fresh audit are:

- `problems/profile_solve.py`: `62%`, 419 missing lines.
- `problems/transport_solve.py`: `73%`, 326 missing lines.
- `solvers/explicit_sparse.py`: `87%`, 310 missing lines.
- `problems/profile_policies.py`: `89%`, 276 missing lines.
- `solvers/preconditioner_transport_matrix.py`: `83%`, 296 missing lines.
- `operators/profile_full_system.py`: `84%`, 279 missing lines.
- `operators/profile_true_operator_rescue.py`: `81%`, 259 missing lines.
- `problems/profile_sparse_handoff.py`: `85%`, 258 missing lines.
- `solvers/preconditioner_xblock_tz_sparse.py`: `76%`, 251 missing lines.
- `problems/transport_parallel_runtime.py`: `86%`, 250 missing lines.
- `solvers/preconditioner_qi_corrections.py`: `88%`, 247 missing lines.
- `solvers/preconditioner_qi_device.py`: `89%`, 235 missing lines.
- `operators/profile_system.py`: `77%`, 234 missing lines.
- `problems/transport_linear_system.py`: `81%`, 224 missing lines.
- `solvers/preconditioner_qi_basis.py`: `89%`, 194 missing lines.
- `solvers/preconditioner_schur_profile.py`: `84%`, 185 missing lines.
- `solver.py`: `86%`, 183 missing lines.
- `outputs/rhsmode1.py`: `79%`, 164 missing lines.
- `outputs/writer.py`: `91%`, 162 missing lines.
- `problems/profile_dense.py`: `87%`, 162 missing lines.

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

Root modules retained for this PR:

- `api.py`, `cli.py`, `__main__.py`, and `__init__.py` for user entry points.
- `solver.py`, `ambipolar.py`, and `sensitivity.py` for public solve,
  ambipolar, and differentiability APIs.
- `plotting.py`, `compare.py`, `io.py`, `namelist.py`, `input_compat.py`, and
  `paths.py` for user-facing plotting, comparison, I/O, input, and path
  utilities.
- `diagnostics.py`, `grids.py`, and `profiling.py` for stable scientific and
  support APIs used by examples, docs, tests, and benchmark tooling.
- `v3_driver.py` as the only root compatibility facade.

Compatibility shims retained for one release cycle:

- `sfincs_jax.v3_driver`.
- `sfincs_jax.operators.profile_response`.
- `sfincs_jax.problems.profile_response`.
- `sfincs_jax.problems.transport_matrix`.
- `sfincs_jax.solvers.preconditioners`.

These shims are intentionally small and tested. Delete them only after public
docs, examples, scripts, and compatibility tests no longer require the legacy
paths, or after a real lazy alias mechanism avoids circular imports without
adding implementation complexity.

## Open Lanes And Status

### Lane 1 - Review-Ready Refactor

Status: 99%.

Goal: finish the PR with a smaller, clearer source tree without changing
physics, outputs, tolerances, solver defaults, differentiable Python paths,
non-autodiff CLI fast paths, CPU/GPU behavior, or parity gates.

Latest AST audit:

- Folder depth is no longer the blocker: the package has one-level domain
  folders only and no `__init__.py`-only source packages.
- The remaining structural blocker is owner size. The largest retained owners
  are `problems/profile_solve.py` (`4745` lines, with
  `solve_v3_full_system_linear_gmres` spanning `3836` lines),
  `outputs/writer.py` (`3250` lines, with `write_sfincs_jax_output_h5`
  spanning roughly `1852` lines), `solvers/explicit_sparse.py` (`5056`
  lines), and `problems/transport_solve.py` (`3191` lines).
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
- The next consolidation pass must reduce those owner sizes using existing
  domain files. Do not add more package folders or helper-only files.

Remaining work:

- Completed Tranche 1: active reduced-system setup extraction into
  `problems/profile_setup.py`.
- Completed Tranche 2: route/preconditioner-selection setup extraction into
  `problems/profile_policies.py`.
- Completed Tranche 3: profile-solve consolidation of active reduced-system
  setup, preconditioner-route setup, linear-solve dispatch setup, and
  structured f-block metadata recording into existing owners. Continue this
  lane only when a complete solve/setup phase can move to an existing owner
  without adding helper-only files.
- Completed Tranche 4: RHSMode=1 output solve-method selection extraction into
  `outputs/rhsmode1.py`.
- Completed Tranche 5: RHSMode=1 output correction helper extraction into
  `outputs/rhsmode1.py`.
- Completed Tranche 6: RHSMode=1 core diagnostic, Phi1 scalar, and
  electric-drift output schema extraction into `outputs/rhsmode1.py`. This
  phase dropped `write_sfincs_jax_output_h5` below 2000 lines and added direct
  HDF5-schema tests, but it was intentionally smaller than the earlier 200-line
  target because the remaining electric-drift computation is still entangled
  with operator internals.
- Completed Tranche 7: RHSMode=1 classical flux output extraction into
  `outputs/rhsmode1.py`, covering both no-Phi1 and Phi1-history branches with
  tiny real-formula tests.
- Completed Tranche 8: RHSMode=1 NTV diagnostic recomputation extraction into
  `outputs/rhsmode1.py`, covering geometryScheme=5 and non-axisymmetric L=2
  paths with tiny numerical tests.
- Completed Tranche 9: moved the complete `export_f` output-grid mapping and
  distribution projection phase into existing `outputs/formats.py`, then
  centralized the RHSMode=1 and transport state-vector `delta_f`/`full_f`
  writes in the same owner. `outputs/writer.py` is down to `3250` lines, the
  compatibility aliases used by `sfincs_jax.io` are preserved, and the
  export/HDF5/output-policy tests pass.
- Tranche 10: retain `explicit_sparse.py` as one owner unless a patch can move a
  complete symbolic-factor family into an existing solver owner while deleting
  more code than it adds. Do not fragment sparse factor code into many small
  files.
- Keep `v3_driver.py` and `io.py` below 80 lines and implementation-free.
- Run source-layout, import-contract, docs, examples, and CLI/output guards.

### Lane 2 - Coverage And Future-Proof Tests

Status: 88% measured package coverage.

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

### Phase A - Lock The Structure

1. Run the source inventory commands and compare with
   `tests/fixtures/source_tree_expected.json`.
2. Run the source-tree and import-contract tests.
3. Run stale public import scans for `sfincs_jax.v3_driver` and deleted module
   names in docs, examples, scripts, tests, and source.
4. Update `sfincs_jax/README.md`, `docs/source_map.rst`, and tests only if the
   actual tree changes.

Acceptance:

- No nested packages.
- No unexpected root files.
- No public examples/scripts importing `v3_driver`.
- Compatibility imports still resolve to canonical owners.

### Phase B - Do One High-Impact Refactor Or Stop

1. Inspect the large owners listed in Lane 1.
2. Choose one owner-level edit only if it removes duplicate code, deletes files,
   or materially simplifies API ownership.
3. If no owner-level edit meets the bar, record the retained boundary and stop
   refactor churn.
4. Run focused behavior tests for any changed owner.

Acceptance:

- No new implementation files unless the same commit deletes at least two old
  files.
- No package-line increase unless the ownership boundary is simpler and tested.
- Physics, output, solver-policy, and differentiability behavior are unchanged.

### Phase C - Coverage Ramp

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

### Phase D - Review-Lock Docs And Examples

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

### Phase E - Regenerate Release Evidence

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

### Phase F - PR Review Readiness

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
rg -n "On the current main branch|not replacements for the production-resolution gates|The production benchmark manifest|not a public performance row|current main|new benchmarks|At the moment" \
  README.md sfincs_jax/README.md examples/README.md docs \
  --glob '!docs/release_notes.rst' --glob '!docs/upstream/**'
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
