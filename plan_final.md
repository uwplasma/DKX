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
- The latest local xdist coverage audit measured `87%` package coverage:
  `3925 passed in 272.20 s` with `8701` missing executable lines.
- The latest bounded coverage tranches added RHSMode-1 Schur/coarse fallback
  tests, output-gradient coordinate contract tests, and default
  preconditioner-selection tests. A full non-coverage regression check after
  these tranches passed: `3925 passed in 247.02 s`.
- The CI coverage floor remains lower than the final target until measured
  margin is available; the review target is `95%` meaningful package coverage
  while keeping GitHub Actions under 10 minutes.

The largest coverage blockers from the fresh audit are:

- `problems/profile_solve.py`: `58%`, 568 missing lines.
- `outputs/writer.py`: `85%`, 384 missing lines.
- `problems/transport_solve.py`: `73%`, 326 missing lines.
- `solvers/explicit_sparse.py`: `87%`, 310 missing lines.
- `solvers/preconditioner_transport_matrix.py`: `83%`, 296 missing lines.
- `operators/profile_full_system.py`: `84%`, 279 missing lines.
- `operators/profile_true_operator_rescue.py`: `81%`, 259 missing lines.
- `problems/profile_sparse_handoff.py`: `85%`, 258 missing lines.
- `solvers/preconditioner_xblock_tz_sparse.py`: `76%`, 251 missing lines.
- `problems/transport_parallel_runtime.py`: `86%`, 250 missing lines.
- `solvers/preconditioner_qi_corrections.py`: `88%`, 247 missing lines.
- `problems/profile_policies.py`: `90%`, 243 missing lines.
- `solvers/preconditioner_qi_device.py`: `89%`, 235 missing lines.
- `operators/profile_system.py`: `77%`, 234 missing lines.
- `problems/transport_linear_system.py`: `81%`, 224 missing lines.
- `solvers/preconditioner_qi_basis.py`: `89%`, 194 missing lines.
- `solvers/preconditioner_schur_profile.py`: `84%`, 185 missing lines.
- `solver.py`: `86%`, 183 missing lines.

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

Status: 98%.

Goal: finish the PR with a smaller, clearer source tree without changing
physics, outputs, tolerances, solver defaults, differentiable Python paths,
non-autodiff CLI fast paths, CPU/GPU behavior, or parity gates.

Remaining work:

- Run one retained-boundary audit over the large owners:
  `problems/profile_solve.py`, `problems/profile_policies.py`,
  `problems/profile_sparse_xblock.py`,
  `problems/profile_sparse_handoff.py`,
  `problems/transport_solve.py`, `solvers/preconditioner_qi_device.py`, and
  `outputs/writer.py`.
- Edit a large owner only if a patch removes a repeated internal section of at
  least about 300 lines, deletes files, or clearly simplifies a public boundary.
  Otherwise document the retained boundary and stop refactor churn.
- Keep `v3_driver.py` and `io.py` below 80 lines and implementation-free.
- Run source-layout, import-contract, docs, examples, and CLI/output guards.

### Lane 2 - Coverage And Future-Proof Tests

Status: 87% measured package coverage.

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
