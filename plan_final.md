# SFINCS_JAX Core-Slim Review Plan

Last updated: 2026-07-07

Active branch / PR: `refactor/v3-driver-architecture` / PR #8

This file is the single active plan for making the refactor branch review-ready.
`plan.md` is the historical execution log. Do not create another competing plan.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core: one input
file plus geometry should run accurate CPU/GPU neoclassical calculations with
automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime/memory, clear physics/numerics in the code, and validated
differentiable Python workflows for sensitivities, ambipolar solves, bootstrap
current, transport coefficients, plotting, and optimization.

## Current Review State

- Latest audited branch head: `cac7fb3a`; last green CI evidence: `96fb2677`.
  Exact aggregate coverage from that CI was 91.753% (`92%`).
- Package layout is shallow, but the branch is too large: 114 Python source
  files and about 144k source lines. Largest complexity owners remain
  `problems/`, `solvers/`, RHSMode-1/QI/preconditioner infrastructure, and
  compatibility layers around those paths.
- Non-package Python volume is still too high: `tests/` has 331 Python files,
  `examples/` has 122 Python files, and `scripts/` has 42 Python files.
  Top-level `benchmarks/` is removed from the active tracked tree; compact
  Fortran-v3 references live in `tests/fixtures/fortran_v3_reference_fixture.json`.
- No tracked file larger than 2 MB was found. There is no tracked
  `sfincs_jax/data/`; release-data logic is `validation/data_fetch.py`.
- The product target changed: PR #8 should become a small stable core, while
  experimental or not-yet-competitive work moves to separate research PRs.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 20% | Every retained file/function/line has a core reason, a caller, and a test/doc owner; everything else is extracted or deleted. |
| Core-main slimming | Active | 40% | Main keeps only stable, parity-clean, runtime-acceptable solvers and public APIs; research code is outside core. |
| Source simplification | Active | 45% | Package moves toward <=50 source files and <=50k lines, with a 10x reduction as stretch if functionality permits. |
| Examples/tests/scripts cleanup | Active | 45% | Examples are <=10 curated workflows plus Fortran-v3 references; tests are smaller, organized, and >=95% coverage; scripts are removed or promoted; benchmarks are gone. |
| Parity/performance evidence | Active | 70% | Supported examples rerun against SFINCS Fortran v3 with runtime/RSS/bootstrap evidence; unsupported research lanes are not marketed. |
| Docs/readme regeneration | Active | 80% | README/docs describe only the stable core plus explicit external research PR lanes. |

Overall readiness under this stricter core-slim goal is about 60-65%.

## Source Structure Rules

- A retained line must satisfy at least one reason: public API, physics equation,
  numerical method, validated default solver path, I/O schema, plotting path,
  differentiability path, compatibility facade, or test/validation support.
- A retained function must have a caller or documented public import, a bounded
  test, and a clear physics/numerics/output role. Otherwise extract or delete.
- Core defaults must be automatic: users should not need environment variables
  or solver-path knowledge for supported examples.
- Experimental QI, unfinished device-QI, native sparse-direct replacements,
  unpromoted preconditioners, profiling/probe-only routes, long benchmark
  campaigns, and historical audits move to research PR branches before removal
  if they have research value. Dead code is deleted directly.
- Root modules remain public API/CLI/I/O/plotting/comparison/sensitivity/
  ambipolar/grids/namelist/paths/profiling/solver facades. Implementation stays
  in one-level domain folders: `discretization/`, `geometry/`, `operators/`,
  `outputs/`, `physics/`, `problems/`, `solvers/`, `validation/`, `workflows/`.
- Each tranche must reduce net files or net lines unless it adds a required
  public feature, accuracy gate, or performance gate.

## Line Audit Protocol

For every touched file, classify code in this order before editing:

1. Public surface: keep only imports/classes/functions documented in README,
   API docs, examples, or compatibility tests.
2. Physics/numerics: keep equations, operator assembly, quadrature, collision,
   drift, solve, and diagnostic code only when tied to a retained model and a
   bounded accuracy/residual/parity test.
3. Runtime-critical implementation: keep only paths used by automatic defaults
   or a documented advanced option that passes runtime/RSS gates.
4. Differentiability: keep JAX-native code needed for supported JVP/VJP,
   implicit sensitivity, optimization, and ambipolar derivative workflows.
5. Compatibility: keep thin aliases only if migration tests require them; no
   compatibility file may own new logic.
6. Experimental/research/profiling/history: extract to a preservation branch or
   delete from core; do not leave dormant imports, env-only routes, or unused
   policy branches in stable modules.

## Concrete Code-Audit Rules

The refactor is not a file-shuffle exercise. Each tranche must prove that the
remaining code is necessary, simpler, and still accurate. Apply these rules to
every source, test, example, script, and documentation file.

### Required Inventory Columns

Every retained or removed file must have one row in the checked inventory with:

- `path`: current repository path.
- `kind`: `source`, `test`, `example`, `script`, `docs`, `fixture`, or
  `generated-artifact`.
- `line_count`: current physical lines.
- `public_surface`: documented import, CLI command, example entry point, or
  `none`.
- `callers`: internal callers found by import/call search, or `none`.
- `physics_or_numerics`: the equation, model, discretization, solver,
  diagnostic, or output contract owned by the file, or `none`.
- `tests`: bounded tests that fail if the file is wrong, or `missing`.
- `docs_examples`: README/docs/example references, or `none`.
- `runtime_role`: `default`, `advanced`, `validation-only`, `research-only`,
  or `none`.
- `autodiff_role`: `primal`, `jvp`, `vjp`, `implicit`, `custom-rule`, or
  `none`.
- `action`: `keep`, `merge`, `thin-compat`, `extract-pr`, or `delete`.
- `justification`: one sentence explaining why this exact action is the
  simplest safe choice.

Rows with `tests=missing`, `callers=none`, `public_surface=none`, and
`runtime_role=none` default to `delete` unless they are small fixtures required
by another retained test.

### Per-Line Keep/Delete Questions

When touching a file, each import, constant, helper, branch, dataclass field,
environment variable, diagnostic key, and compatibility alias must answer:

1. What retained user workflow, physics equation, numerical method, or output
   schema needs this line?
2. What test would fail if this line were removed or simplified?
3. Is the same behavior already implemented elsewhere under a clearer name?
4. Is this an automatic default path, or only an experimental/manual knob?
5. Can this be expressed as data/configuration instead of another branch?
6. Can this line be moved to a research branch without reducing stable parity?
7. Does it help runtime, memory, differentiability, or clarity enough to pay for
   its complexity?

If the answer to questions 1, 2, and 7 is weak, remove the line, merge it into a
clearer retained helper, or extract it to a research PR.

### Function-Level Rules

- Functions over 80 lines require a stated stage boundary and should be split
  only when the split creates a physics/numerics stage with a direct test.
- Functions under 10 lines that are used once should usually be inlined unless
  they name a physics equation or public API concept.
- Two functions with the same inputs/outputs and different policy names should
  collapse into one implementation plus explicit data/options.
- Environment-variable-only functions are not stable core features. Promote
  them to documented advanced options with tests, or extract/delete them.
- Fail-closed/probe functions are allowed only if they prevent expensive or
  inaccurate default solves and have direct admission/rejection tests.
- Compatibility functions must delegate without owning logic and must have a
  removal note or migration test.

### File-Structure Rules

- Prefer fewer, domain-named modules over many narrow policy files. A new file
  is allowed only when it owns a stable domain concept: geometry loading,
  discretization, operator assembly, physics diagnostics, solver policy,
  output, plotting, validation, or workflow orchestration.
- Do not keep folders that only contain `__init__.py` plus one nested folder.
  Flatten them into the nearest domain package or remove the package.
- Avoid names based on historical implementation details (`v3_`, `rhs1_`,
  `transport_` everywhere, `probe`, `campaign`, `qi_*`) in stable core unless
  the name is a public migration facade.
- Stable source should be readable in this order: public API/CLI, physics
  models, discretization/operators, solver defaults, outputs/plotting,
  validation.

### Extraction And Deletion Gates

- `extract-pr`: code has research value but lacks production parity,
  acceptable runtime/RSS, automatic defaults, or stable docs. It must be
  preserved on a named branch before deletion from the stable core.
- `delete`: code is obsolete, duplicated, uncalled, a generated artifact, a
  stale audit/profiling script, or a test for deleted behavior.
- `thin-compat`: public import names that users may still import. These files
  must contain no solver logic and should only delegate to the new stable
  module.
- No deleted/extracted module may remain in API docs, public import contracts,
  source-tree fixtures, examples catalogues, README figures, or default solver
  policy imports.

### Complexity Budgets

The final PR should meet these budgets or document each exception:

- `sfincs_jax/`: <=50 Python files and <=50k lines; stretch target is a 10x
  reduction from the pre-slim source-line count.
- `tests/`: <=120 Python files with grouped domain ownership and >=95%
  coverage of the slim core.
- `examples/`: original Fortran-v3 reference inputs plus <=10 curated workflows.
- `scripts/`: zero or only documented release-maintenance commands.
- `benchmarks/`: removed from core; benchmark evidence lives in bounded tests,
  fixtures, docs, or release assets.
- Tracked artifacts over 2 MB are forbidden unless explicitly fetched from
  release data by tests/docs.

### Tranche Acceptance Checklist

Each commit-level tranche must report:

- Net source/test/example/script/docs files added and removed.
- Net line-count change.
- Modules extracted, deleted, merged, or retained with justification.
- Focused tests run and their results.
- Ruff/compile/diff/large-file checks.
- Any parity/performance claim changed by the tranche.
- Remaining open lanes and completion estimate.

### Repository-Wide Audit Tranches

Finish the simplification by auditing in this fixed order. Each tranche must
edit the inventory first, then remove or merge code, then run focused guards.

1. Public surface: `api.py`, `cli.py`, `namelist.py`, `io.py`, plotting,
   output formats, examples used in README, and documented imports. Delete
   undocumented aliases unless migration tests require a thin facade.
2. Physics models: collisions, drifts, classical transport, bootstrap/Redl,
   ambipolarity, and geometry loaders. Keep equations with tests and citations;
   merge duplicate normalization helpers.
3. Discretization/operators: velocity grids, periodic stencils, RHSMode-1
   full-system operators, transport operators, and sparse patterns. Keep only
   operators used by supported automatic defaults or frozen parity tests.
4. Solver defaults: dense, structured, sparse, PAS, FP, x-block, and
   transport-matrix paths. Delete env-only experiments and retain only routes
   with residual, parity, runtime, and RSS gates.
5. Differentiability: primal/JVP/VJP/implicit sensitivity paths used by
   examples and tests. Remove branches that break JAX transforms or lack a
   documented non-differentiable CLI-only boundary.
6. Validation and tests: collapse duplicated smoke tests into physics,
   regression, unit, CLI/I/O, and optional-integration groups; keep only small
   fixtures or release-fetched artifacts.
7. Examples, scripts, and benchmarks: keep <=10 curated workflows plus
   Fortran-v3 reference inputs; move campaigns/profilers/publication one-offs
   to research branches or delete generated outputs.
8. Documentation: rewrite docs after code cuts, not before. Docs must describe
   the stable core as standalone software and list extracted research lanes
   only in the research-lanes page.

## Ordered Finish Plan

### Phase A - Build The Auditable Inventory

1. Generate a checked inventory with columns: path, lines, public imports,
   internal callers, tests, docs/examples, category, action, and justification.
2. Classify every file as `core`, `compat`, `test-fixture`, `extract-pr`, or
   `delete`. Classify high-line functions inside the largest files the same way.
3. Start with the top owners: `profile_policies.py`, `profile_sparse_xblock.py`,
   `profile_full_system.py`, `preconditioner_qi_device.py`, `explicit_sparse.py`,
   `profile_sparse_solve.py`, `profile_sparse_qi.py`,
   `preconditioner_qi_corrections.py`, `profile_solve.py`,
   `transport_linear_system.py`, `preconditioner_qi_basis.py`,
   `profile_sparse_direct.py`, `transport_parallel_runtime.py`,
   `profile_dense.py`, `solver.py`, `outputs/rhsmode1.py`, and
   `validation/artifacts.py`.
4. Use `rg`, import graphs, coverage JSON, tests, README/docs references, and
   example calls to decide each action. No broad deletion without this record.

- Inventory exists and maps the large source/test/example/script/benchmark
  owners to core, extract, or delete.
- Each high-line retained module has a specific reduction target and owner.

### Phase B - Extract Research Families First

1. Preservation branches already exist. Remove core imports in this order:
   QI/device-QI, native sparse-direct/reduced-Pmat rescue, parallel campaigns,
   then publication audits.
2. After each import cut, delete or move the associated source/tests/examples/
   scripts and update the inventory so missing extracted paths are intentional.
3. Keep only small summary fixtures/figures needed by stable README/docs; all
   large or regenerable artifacts stay in releases or research branches.

- Core imports no longer reach extracted modules.
- Extracted research PRs preserve useful work but are not required by main.
- Core source count and line count drop materially after each extraction.

### Phase C - Reduce Core Solvers And Operators

1. Keep only solver paths passing strict residual/parity gates and acceptable
   runtime/RSS on supported examples.
2. Delete duplicated routing layers, fail-closed probes, env-only policies, and
   unpromoted experimental paths.
3. Collapse RHSMode-1 and RHSMode-2/3 orchestration into visible stages:
   setup, operator/RHS assembly, solve, diagnostics, output.
4. Re-profile retained examples after each solver cut; fix JIT boundaries,
   operator reuse, sparse/dense selection, and output I/O before adding code.

- Supported examples run without solver environment variables.
- Retained defaults are faster or no worse within documented tolerance.
- Source-tree guards enforce the reduced allowed module list.

### Phase D - Slim Examples To Teaching And Parity

1. Keep original SFINCS Fortran-v3 reference input examples needed for migration
   and parity.
2. Keep at most 10 curated workflows: CLI run/plot, Python run/output,
   VMEC/wout geometry, transport coefficients, ambipolar solve, autodiff,
   bootstrap current/Redl, optimization objective, output formats/plotting, and
   frozen Fortran-v3 parity.
3. Move historical upstream examples, performance campaigns, publication figure
   generators, QI examples, one-off audits, and single-file folders to research
   PRs, docs release assets, or deletion.

- `examples/README.md`, `workflow_catalog.json`, docs, and CI point only to the
  retained workflows.
- Examples are small enough for a new user to scan in minutes.

### Phase E - Remove Benchmarks And Triage Scripts

1. Top-level `benchmarks/` is removed from core. Keep only small summary
   fixtures and bounded test paths for README/docs number consistency.
2. For `scripts/`, promote user commands to examples, validation generators to
   tests/docs helpers, release-maintenance commands to documented scripts, and
   one-off audit/profiling/benchmark scripts to research PRs or deletion.
3. Next script tranche: delete or move standalone audit/profiler/campaign
   scripts that are not documented CLI workflows, not test helpers, and not
   needed for release maintenance.

- No install/test/docs path depends on `benchmarks/`.
- Every remaining script has a public purpose, a test, and docs mention.

### Phase F - Rebuild Tests Around The Slim Core

1. Replace scattered tests with `tests/unit/`, `tests/physics/`,
   `tests/regression/`, `tests/cli_io/`, `tests/fixtures/`, and
   `tests/integration_optional/`.
2. Delete historical output directories and duplicated fixtures after their code
   is extracted or removed.
3. Reach 95% coverage mostly by deleting unused code, then by targeted tests for
   retained physics/numerics: drift-kinetic identities, conservation/moments,
   residual gates, geometry interpolation, Redl/bootstrap normalization,
   ambipolar derivatives, output schema parity, CLI behavior, and Fortran-v3
   frozen fixtures.

- Coverage is >=95% for the slim core.
- Default CI stays under 10 minutes.
- Tests validate science and numerics, not just smoke imports.

### Phase G - Refresh Documentation And Evidence

1. Rewrite README around the slim core: install, one command, plotting, physics,
   curated examples, supported solvers, parity/performance, and exclusions.
2. Update package README, source map, examples docs, testing docs, performance
   docs, and research-lane docs to match the extracted core/research split.
3. Rerun supported CPU parity/runtime/RSS/bootstrap evidence against SFINCS
   Fortran v3; rerun GPU evidence only when available.

- Public docs contain no branch-history or progress-log phrasing.
- Figures/tables match checked summary fixtures and retained examples.

### Phase H - Review Handoff

1. Run focused source/docs/example/test guards after each tranche.
2. Run full default CI and coverage locally before review.
3. Update PR #8 with final source/test/example counts, coverage, benchmark
   scope, extracted PR branches, and deferred research lanes.
4. Stop before merge; PR #8 remains the review surface until approval.

- PR #8 is green, small enough to review, and contains no generated clutter.

## Standard Validation Commands

Use after each tranche:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py
PYTHONNOUSERSITE=1 python -m pytest -q tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py
PYTHONNOUSERSITE=1 python -m ruff check <touched files>
PYTHONNOUSERSITE=1 python -m compileall -q sfincs_jax <touched tests>
git diff --check
find . -path ./.git -prune -o -type f -size +2M -print
```

Use before review:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q
PYTHONNOUSERSITE=1 python -m coverage run -m pytest -q
PYTHONNOUSERSITE=1 python -m coverage report --fail-under=95
PYTHONNOUSERSITE=1 python -m sphinx -W -b html docs docs/_build/html
python examples/run_cli_and_plot.py --out-dir /tmp/sfincs_jax_quick_review
```

## Completion Gates

- Core source is <=50 Python files and <=50k lines, or every excess file/line
  has a documented core-retention justification.
- Experimental solver/preconditioner/QI/profiling lanes are in separate PRs or
  deleted.
- `examples/` contains only original Fortran-v3 reference inputs plus at most
  10 curated workflows.
- Top-level `benchmarks/` is gone; benchmark claims are tested through one
  bounded fixture/test path.
- `scripts/` is empty or limited to documented release-maintenance commands.
- Tests are organized, much smaller, scientifically meaningful, and >=95%
  coverage on the slim core.
- Supported examples have fresh Fortran-v3 parity, runtime, memory, and
  bootstrap-current evidence.
- CI stays under 10 minutes, README/docs match the slim core, and PR #8 is
  review-ready with no generated clutter.

## Explicit Deferred Items

These move out of the stable core unless they satisfy all production gates:
experimental QI/device-QI, native MUMPS/SuperLU_DIST-equivalent sparse-direct
research, unpromoted lower-memory preconditioners, long GPU/multi-GPU
campaigns without fresh evidence, and historical profiler/publication audits.
