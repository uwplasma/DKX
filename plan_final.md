# SFINCS_JAX Review-Ready Refactor Plan

Last updated: 2026-07-06

Active branch: `refactor/v3-driver-architecture`

Review branch / PR: `refactor/v3-driver-architecture` / PR #8

This file is the single active plan for making the refactor branch review-ready.
`plan.md` is the historical execution log. Do not create another competing plan.

## One-Sentence Goal

Ship `sfincs_jax` as a compact, domain-organized, research-grade
neoclassical-transport package: users provide a geometry and input file and get
accurate CPU/GPU results with automatic robust solver selection, while Python
users can opt into differentiable residual, flux, ambipolar, sensitivity, and
optimization workflows with parity against SFINCS Fortran v3 where the physics
models overlap.

## Current Review State

- Package layout is one level deep below `sfincs_jax/` with no nested source
  packages and no `__init__.py`-only package stubs.
- Public root modules are stable user-facing entry points or compatibility
  facades; implementation code lives in domain folders:
  `discretization/`, `geometry/`, `operators/`, `outputs/`, `physics/`,
  `problems/`, `solvers/`, `validation/`, and `workflows/`.
- `sfincs_jax/README.md`, `examples/README.md`, `docs/examples.rst`, and
  `docs/source_map.rst` document the current source and example structure.
- CPU-local validation after the latest refactor baseline passed:
  `4584 passed, 4 skipped in 933.46 s`, Sphinx `-W`, the quick tutorial
  HDF5/NetCDF/NPZ/PDF output path, Ruff, compile checks, `git diff --check`,
  and the large-file audit.
- The July 6 coverage tranches added bounded comparison/plotting,
  drift-operator, and discretization tests:
  `9c48d211`, `653d6ac5`, and `14e0dbd5`. Focused local coverage reports
  `compare.py` at 94%, `plotting.py` at 100%, `profile_exb.py` at 97%,
  `profile_magnetic_drifts.py` at 94%, `adaptive_maps.py` at 97%,
  `periodic_stencil.py` at 98%, and `structured_velocity.py` at 99%.
- The latest bounded local review bundle passed: source-tree guards,
  docs/example guards, and comparison/plotting checks ran `123 passed in
  9.70 s`; Sphinx `-W` passed in `17.56 s`.
- The transport cache and RHSMode=1 routing tranches fixed accepted
  residual-coarse factor reuse, moved BiCGStab, dense-auto, and initial
  sparse-shortcut route decisions out of `profile_solve.py`, aliased duplicate
  profile transport wrappers to the canonical `transport_solve.py` owner, and covered
  true-residual admission, fail-closed fallback, FP/PAS/DKES/tokamak
  boundaries, CPU/GPU dense admission, constraintScheme=0 sparse/PETSc routes,
  implicit rejection, and explicit-method preservation. Guards: `35 passed`,
  `217 passed`, `49 passed`, `44 passed`, `102 passed`, and source/domain
  contracts `58 passed`.
- The last checked PR CI state after `14e0dbd5` had build and
  external-data smoke passing; coverage shards, examples-smoke, optional
  ecosystem checks, and coverage-report were still pending. Recheck CI after
  the next coherent push rather than waiting idle on every job.
- Fresh GPU validation is deferred until the office GPU host is reachable.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Review-ready refactor | Active | 98% | Source tree remains shallow, root modules are public, large owners have review-size guards, no generated clutter is tracked, and PR #8 has clear evidence. |
| Coverage and future-proof tests | Active | 94% | Meaningful package coverage reaches 95% through bounded physics, numerical, unit, and regression tests without pushing CI above 10 minutes. |
| Documentation and examples | Active | 97% | README, package README, docs, tutorial notebooks, and examples describe the same workflows without branch-history or progress-log phrasing. |
| Benchmark/parity/runtime regeneration | Active | 70% | CPU/GPU/Fortran runtime, memory, parity, and bootstrap-current figures are regenerated from the final branch state with solver provenance. |
| Solver/performance boundaries | Active | 90% | Automatic defaults remain residual-clean and documented; expensive research candidates stay opt-in unless they pass strict residual/runtime/RSS gates. |

Overall review readiness: about 92%. The gap is dominated by final coverage
evidence, fresh benchmark regeneration, and GPU validation.

## Source Structure Rules

- Keep only public APIs, CLI entry points, and documented compatibility facades
  in the package root.
- Keep implementation modules inside the existing domain folders; do not add
  deeper packages unless this plan and the source-tree tests are updated first.
- Consolidate by ownership, not by helper names. Prefer moving complete phases
  into existing owners over adding new `rhs1_*`, `transport_*`, or
  `preconditioner_*` helper files.
- Keep domain names descriptive and stable. New implementation files must have
  an obvious physics, geometry, solver, output, validation, or workflow owner.
- Preserve public imports through compatibility aliases only when documented
  user workflows need them. Internal imports should use canonical owners.
- Do not commit run outputs, profiler traces, large equilibrium files, caches,
  or generated docs builds.

## Ordered Finish Plan

### Phase A - Lock CI And Local Cleanliness

1. Check PR #8 CI before each coherent push; fix failing checks first.
2. Keep the worktree free of generated files and files larger than 2 MB.
3. Run the public stale-wording scan before review.

Acceptance:

- Required CI checks are green or have a concrete local fix in the next commit.
- `git status --short` contains only intentional source/docs/test changes.
- The large-file audit reports no worktree files larger than 2 MB outside
  `.git`.

### Phase B - Finish Source Consolidation

1. Keep `outputs/writer.py` and `write_sfincs_jax_output_h5` below their guarded
   review-size budgets.
2. Continue reducing large owners only when a complete phase can move to an
   existing canonical module without creating another helper-only file.
3. Add source-tree guards when a new budget or ownership boundary matters.

Acceptance:

- `tests/test_source_tree_consolidation.py` passes.
- No nested packages, `__init__.py`-only packages, or unplanned modules appear.
- The source map and package README name the canonical owner for each moved
  phase.

### Phase C - Coverage Ramp To 95%

1. Use bounded tests that exercise real physics/numerics contracts: drift
   kinetic operator identities, conservation and moment constraints, ambipolar
   derivative checks, bootstrap-current normalization, geometry interpolation,
   output schema parity, and solver residual gates.
2. Treat the next large coverage tranche as solver-orchestration coverage, not
   isolated helper coverage: `profile_solve.py`,
   `transport_linear_system.py`, `transport_solve.py`,
   `profile_true_operator_rescue.py`, and Schur/profile preconditioner owners
   contain the remaining high-value uncovered branches.
3. Prefer deleting obsolete code over adding tests for unused branches.
4. Keep slow production solves out of default CI; use frozen fixtures,
   monkeypatched owners, tiny analytic geometries, and release-fetched data.

Acceptance:

- CI coverage report reaches the staged gate and records a path to 95%.
- Local focused coverage tranches do not add production solve time.
- Tests remain scientific checks, not smoke-only scaffolds.

### Phase D - Documentation And Examples Review Lock

1. Keep README self-contained: installation, one-command usage, plotting,
   physics summary, validated capabilities, and honest benchmark scope.
2. Keep detailed equations, knobs, failure modes, and research-lane boundaries
   in docs rather than overloading the README.
3. Keep examples task-oriented: getting started, transport, autodiff,
   optimization, VMEC/Redl/bootstrap, parity, performance, publication figures.
4. Keep notebook and script entry points synchronized with
   `examples/workflow_catalog.json`.

Acceptance:

- `tests/test_examples_tree_contract.py` and
  `tests/test_benchmark_doc_claims.py` pass.
- Sphinx builds with `-W`.
- Public docs avoid branch-history wording such as “current main branch”,
  “new version”, and “previous version”.

### Phase E - Evidence Regeneration

1. Regenerate CPU benchmark/parity/runtime/memory plots locally after the final
   CPU-source state is locked.
2. Regenerate GPU evidence only when the GPU host is reachable; do not claim
   fresh GPU performance from CPU-only runs.
3. Regenerate QA/QH bootstrap-current comparison figures from same-resolution
   SFINCS_JAX, SFINCS Fortran v3, and Redl inputs where local evidence exists.
4. Keep generated heavyweight outputs out of git; commit only compressed,
   release-facing figures and small summaries.

Acceptance:

- README and docs plots/tables match checked summary JSON.
- Runtime and memory tables use solver provenance and clearly state benchmark
  scope.
- Any missing GPU evidence is labeled deferred rather than implied.

### Phase F - PR Review Handoff

1. Run the focused source/docs/example/test guard bundle.
2. Run the full no-coverage test suite locally when source changes are complete.
3. Update PR #8 body with the final validation matrix, known deferred items, and
   branch head.
4. Stop before merging; PR #8 remains the single draft review surface until the
   user approves review/merge.

Acceptance:

- PR #8 is green or has only explicitly deferred GPU checks.
- No generated clutter is tracked.
- The final response lists completed work, remaining deferred lanes, and exact
  validation commands/results.

## Standard Validation Commands

Use focused checks after each tranche:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q \
  tests/test_source_tree_consolidation.py \
  tests/test_domain_package_import_contracts.py
PYTHONNOUSERSITE=1 python -m pytest -q \
  tests/test_examples_tree_contract.py \
  tests/test_benchmark_doc_claims.py
PYTHONNOUSERSITE=1 python -m ruff check <touched files>
PYTHONNOUSERSITE=1 python -m compileall -q sfincs_jax <touched tests>
git diff --check
```

Use the full CPU review baseline once the local tranche is complete:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q
PYTHONNOUSERSITE=1 python -m sphinx -W -b html docs docs/_build/html
python examples/tutorials/run_quick_output_and_plot.py --out-dir /tmp/sfincs_jax_quick_output_review
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

- Source tree remains one-level deep under `sfincs_jax/`.
- Root modules are documented stable public entry points or explicit
  compatibility facades.
- Package README, root README, examples README, docs API/source pages, and
  testing docs describe the same structure.
- Examples are task-oriented, pedagogical, and runnable within documented
  budgets.
- Meaningful package coverage reaches 95%.
- CI stays under 10 minutes.
- CPU/GPU/Fortran parity, runtime, memory, and bootstrap-current evidence are
  regenerated from the final branch state, or unavailable GPU evidence is
  explicitly deferred.
- Public performance claims use fresh reports with solver provenance.
- PR #8 is review-ready and contains no generated clutter.

## Explicit Deferred Items

These are not blockers for the refactor PR unless a regression is found:

- A fully native MUMPS/SuperLU_DIST-equivalent sparse direct solver stack.
- Further lower-memory production preconditioner research after correctness
  gates are stable.
- Full production-grid QA/QH performance parity with SFINCS Fortran v3 when
  residuals and outputs are correct but runtime or memory remain worse.
- Additional device-QI research beyond the documented residual floor.
- Fresh GPU benchmarking while the GPU host is unavailable.
