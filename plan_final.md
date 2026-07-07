# SFINCS_JAX Core-Slim Review Plan

Last updated: 2026-07-07

Active branch / PR: `refactor/v3-driver-architecture` / PR #8

This file is the single active plan for making the refactor branch review-ready.
`plan.md` is the historical execution log. Do not create another competing plan.

## One-Sentence Goal

Ship `sfincs_jax` as a small, understandable, research-grade core package:
users provide a geometry and input file and get accurate CPU/GPU results with
automatic robust defaults, parity with SFINCS Fortran v3 where models overlap,
competitive runtime/memory for supported examples, and differentiable Python
workflows for validated sensitivities, ambipolar solves, bootstrap current,
transport coefficients, plotting, and optimization.

## Current Review State

- Latest pushed branch head: `5be6ad72`.
- Last green PR CI evidence: `96fb2677`; coverage shards/report, examples
  smoke, external-data smoke, optional ecosystem gates, build, tests, and
  Codecov patch passed. Latest exact aggregate coverage was 91.753% (`92%`).
- Package structure is one-level deep under `sfincs_jax/`, but the branch is
  still too large: 120 Python source files and about 166k package source lines.
  Most complexity is concentrated in `problems/` and `solvers/`.
- Tracked non-package volume is also too high: `tests/` has 726 files,
  `examples/` has 497 files, `benchmarks/` has 268 files, and `scripts/` has
  47 files.
- No tracked file larger than 2 MB was found in the latest audit.
- There is no tracked `sfincs_jax/data/` package directory. Release-data
  plumbing lives under `sfincs_jax/validation/data_fetch.py`; any future data
  package must be justified as validation/release-asset infrastructure.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Core-main slimming | Active | 35% | Main keeps only stable, parity-clean, runtime-acceptable solvers and public APIs; experimental code is moved to research PR branches or deleted. |
| Source simplification | Active | 45% | Package is reduced toward <=50 source files and <=50k lines, with a stretch target near 10x smaller if functionality permits. |
| Example consolidation | Active | 40% | Examples contain original SFINCS Fortran-v3 reference inputs plus at most 10 curated user-facing workflows. |
| Test consolidation and 95% coverage | Active | 45% | Tests are organized into a small folder set, avoid historical script bloat, and reach 95% by deleting unused code plus targeted physics/numerics gates. |
| Benchmark/script cleanup | Active | 30% | `benchmarks/` is removed from the release tree; reusable benchmark logic is one tested module/fixture path, and scripts are promoted to examples/tests or removed. |
| Docs/readme regeneration | Active | 80% | README/docs describe the slim core, extracted research PRs, curated examples, and fresh parity/runtime/memory evidence. |
| Final parity/performance evidence | Active | 70% | Supported core examples rerun against SFINCS Fortran v3 on CPU and GPU when available; unsupported research lanes are not marketed as core. |

Overall review readiness: about 60% under the new core-slim goal. The previous
refactor structure is useful, but the product target changed from "organized
large branch" to "small stable main plus separate research PRs."

## Source Structure Rules

- Main branch contains only production-supported code paths with demonstrated
  accuracy, parity, and bounded runtime/memory for documented examples.
- Experimental QI, native sparse-direct replacements, unfinished GPU/device QI,
  unpromoted preconditioners, profiling-only routes, and solver-policy probes
  move to separate branches/PRs before deletion from the core branch if they
  have research value. Obsolete dead code is deleted without a preservation PR.
- Root modules stay limited to public API, CLI, I/O, plotting, comparison,
  sensitivity, ambipolar, grids, namelist, paths, profiling, and solver facade
  surfaces. Implementation stays in one-level domain folders only.
- Domain folders must be physics/numerics oriented: `discretization/`,
  `geometry/`, `operators/`, `outputs/`, `physics/`, `problems/`, `solvers/`,
  `validation/`, and `workflows/`. Do not add helper-only folders.
- Prefer deleting or extracting whole unused families over moving small helper
  functions around. A refactor tranche must reduce net files or net lines unless
  it adds a required public feature or test gate.
- No tracked generated outputs, profiler traces, caches, large equilibria, or
  release artifacts in the source tree.

## Ordered Finish Plan

### Phase A - Freeze And Classify

1. Record the current branch head and CI state, then stop adding new features to
   PR #8 until the slimming inventory is complete.
2. Classify every large package module as one of: core, compatibility facade,
   test-only, extract-to-research-PR, or delete.
3. Classify every example, script, benchmark file, and test fixture using the
   same categories.
4. Create a preservation list for research-value code before removing it from
   the core branch.

- A checked inventory maps each large source/example/test/script/benchmark
  owner to core, extract, or delete.
- No code is deleted before research-value extraction is identified.

### Phase B - Extract Or Delete Experimental Solvers

1. Move QI experimental solvers, unfinished device-QI paths, unpromoted
   native-MUMPS-like sparse-direct work, broad preconditioner experiments, and
   profiling/probe-only solver branches out of the core branch.
2. Keep only automatic default solver paths that pass strict residual/parity
   gates on the supported example set.
3. Collapse compatibility aliases after users and docs no longer import the
   extracted modules.
4. Recompute source-tree budgets after each deletion/extraction tranche.

- Supported examples still run with no environment-variable solver selection.
- Package target is <=50 source files and <=50k lines, or a documented blocker
  explains why more core code is required.
- `tests/test_source_tree_consolidation.py` enforces the new slim source list.

### Phase C - Slim Examples To Teaching And Parity

1. Keep original SFINCS Fortran-v3 reference input examples needed for parity
   and migration.
2. Keep at most 10 curated workflows:
   CLI run and plot; Python run and output; VMEC/wout geometry; RHSMode=2/3
   transport coefficients; ambipolar electric-field solve; autodiff
   sensitivity; bootstrap current and Redl comparison; optimization objective;
   output-format/plotting; parity check against frozen Fortran-v3 fixture.
3. Move historical upstream examples, large publication figure scripts,
   performance campaigns, and one-off audit examples to docs/release assets or
   research PR branches.
4. Keep `examples/README.md` short and task-based; detailed background belongs
   in docs.

- `examples/` has no more than the original Fortran-v3 reference set plus 10
  curated workflows.
- No example folder exists for a single file unless it is one of the curated
  workflows.
- `examples/workflow_catalog.json` and docs point only to retained workflows.

### Phase D - Replace Benchmark Folder With Tests

1. Remove the release `benchmarks/` folder from the core tree.
2. Preserve only a small checked summary fixture and one test module that
   verifies benchmark JSON, plotted numbers, and README/docs consistency.
3. Move long-run benchmark campaigns to release assets or a separate benchmark
   PR branch.

- No top-level `benchmarks/` directory is required for install, tests, or docs.
- Benchmark claims are checked by one bounded test file and small fixtures.

### Phase E - Reorganize Tests Without Losing Coverage

1. Replace hundreds of scattered tests with a few folders:
   `tests/unit/`, `tests/physics/`, `tests/regression/`, `tests/cli_io/`,
   `tests/fixtures/`, and `tests/integration_optional/`.
2. Delete historical scripts and duplicated fixtures after the corresponding
   code paths are extracted or removed.
3. Reach 95% coverage primarily by deleting unused branches, then by targeted
   literature-anchored physics/numerics tests for retained core functions.
4. Keep default CI under 10 minutes; production and GPU gates remain optional
   or release-asset backed.

- Coverage is >=95% for the slim core.
- Test file count and line count are substantially lower than today; no
  historical output directories are tracked under `tests/`.
- Tests check real equations, conservation/moment constraints, residual gates,
  output schema parity, Redl/bootstrap normalization, geometry interpolation,
  ambipolar derivatives, CLI behavior, and Fortran-v3 fixture parity.

### Phase F - Promote Or Remove Scripts

1. Promote user-facing scripts to curated examples.
2. Promote validation/report generators to tests or docs build helpers.
3. Remove one-off profiling, audit, and historical campaign scripts from the
   core branch after extracting research-value copies.

- `scripts/` is empty or contains only documented release-maintenance commands.
- Every remaining script has an owner, a test, and README/docs mention.

### Phase G - Refresh Docs, README, And Evidence

1. Rewrite README around the slim core: install, one command, plotting,
   supported physics, curated examples, parity/performance summary, and honest
   exclusions.
2. Update docs/source map, examples docs, testing docs, performance docs, and
   research-lane docs to describe extracted PR lanes instead of shipping them
   as core.
3. Rerun supported CPU parity/runtime/memory/bootstrap-current evidence against
   SFINCS Fortran v3. Rerun GPU evidence only when available.

- Public docs contain no branch-history or progress-log phrasing.
- Figures/tables match checked summary fixtures.
- Research lanes are clearly outside the stable core.

### Phase H - Review Handoff

1. Run focused source/docs/example/test guards after each tranche.
2. Run full default CI locally before review.
3. Update PR #8 with final source/test/example counts, coverage, benchmark
   scope, extracted PR branches, and deferred research lanes.
4. Stop before merge; PR #8 remains the review surface until approval.

- PR #8 is green, small enough to review, and contains no generated clutter.
- Main can be merged as a stable core without experimental baggage.

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

Use the slim-core baseline before review:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q
PYTHONNOUSERSITE=1 python -m coverage run -m pytest -q
PYTHONNOUSERSITE=1 python -m coverage report --fail-under=95
PYTHONNOUSERSITE=1 python -m sphinx -W -b html docs docs/_build/html
python examples/run_cli_and_plot.py --out-dir /tmp/sfincs_jax_quick_review
```

Use the public wording and size scans before review:

```bash
rg -n "current main|new benchmarks|At the moment|new version|previous version|\\bcurrently\\b" \
  README.md sfincs_jax/README.md examples docs \
  --glob '!docs/_build/**' --glob '!docs/release_notes.rst' \
  --glob '!docs/upstream/**' --glob '!docs/_static/**'
find . -path ./.git -prune -o -type f -size +2M -print
```

## Completion Gates

The plan is complete only when all gates pass:

- Core source is <=50 Python files and <=50k lines, or the blocker is justified
  by retained stable functionality.
- Experimental solver/preconditioner/QI/profiling lanes are in separate
  branches/PRs or deleted.
- `examples/` contains only original Fortran-v3 reference inputs plus at most
  10 curated workflows.
- Top-level `benchmarks/` is gone; benchmark claims are tested through one
  bounded fixture/test path.
- `scripts/` is empty or limited to documented release-maintenance commands.
- Tests are reorganized, much smaller, scientifically meaningful, and >=95%
  coverage on the slim core.
- CI stays under 10 minutes.
- README/docs/plots/tables match fresh supported parity/runtime/memory evidence.
- PR #8 is review-ready and contains no generated clutter.

## Explicit Deferred Items

These move out of the stable core unless they satisfy all production gates:

- Experimental QI and true device-QI solver/preconditioner research.
- Native MUMPS/SuperLU_DIST-equivalent sparse-direct replacement research.
- Lower-memory production preconditioner experiments not promoted by strict
  residual, parity, runtime, and RSS gates.
- Long-run GPU and multi-GPU performance campaigns while GPU evidence is absent.
- Historical benchmark campaigns, profiler traces, and publication audit scripts.
