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

- Latest audited code-change head: `141be05b`; last green CI evidence: `96fb2677`.
  Exact aggregate coverage from that CI was 91.753% (`92%`).
- Package layout is shallow, but the branch is too large: 114 tracked Python
  source files and about 141.5k tracked source lines. Largest complexity owners remain
  `problems/`, `solvers/`, RHSMode-1/QI/preconditioner infrastructure, and
  compatibility layers around those paths.
- Non-package Python volume is still too high: `tests/` has 313 tracked Python
  files and about 127.1k lines, `examples/` has 122 tracked Python files, and
  `scripts/` has 12 tracked Python files.
  Top-level `benchmarks/` is removed from the active tracked tree; compact
  Fortran-v3 references live in `tests/fixtures/fortran_v3_reference_fixture.json`.
- No tracked file larger than 2 MB was found. There is no tracked
  `sfincs_jax/data/`; release-data logic is `validation/data_fetch.py`.
  Ignored local outputs such as `benchmarks/` and `__pycache__/` are not counted
  in review budgets and must not be committed.
- The product target changed: PR #8 should become a small stable core, while
  experimental or not-yet-competitive work moves to separate research PRs.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 36% | Every retained file/function/line has a core reason, a caller, and a test/doc owner; everything else is extracted or deleted. |
| Core-main slimming | Active | 51% | Main keeps only stable, parity-clean, runtime-acceptable solvers and public APIs; research code is outside core. |
| Source simplification | Active | 51% | Package moves toward <=50 source files and <=50k lines, with a 10x reduction as stretch if functionality permits. |
| Examples/tests/scripts cleanup | Active | 77% | Examples are <=10 curated workflows plus Fortran-v3 references; tests are smaller, organized, and >=95% coverage; scripts are removed or promoted; benchmarks are gone. |
| Parity/performance evidence | Active | 70% | Supported examples rerun against SFINCS Fortran v3 with runtime/RSS/bootstrap evidence; unsupported research lanes are not marketed. |
| Docs/readme regeneration | Active | 80% | README/docs describe only the stable core plus explicit external research PR lanes. |

Overall readiness under this stricter core-slim goal is about 81-84%.

## Concrete Code-Audit Rules

Default action for every line is removal. Retain a line only when it protects a
stable workflow, physics equation, numerical method, output schema, parity gate,
runtime/RSS gate, or differentiability workflow. Each retained file needs one
inventory row with `path`, `kind`, `line_count`, public surface, callers,
physics/numerics role, tests, docs/examples, runtime role, autodiff role,
`action`, and a one-sentence justification. Rows missing callers, tests, public
surface, and runtime role default to `delete`.

For every touched import, constant, helper, branch, dataclass field, env token,
diagnostic key, test, example, and doc paragraph, answer:

1. Which retained workflow/equation/method/schema needs this?
2. Which bounded test fails if it is removed?
3. Is this duplicated elsewhere under a clearer name?
4. Is it an automatic default or only an experimental/manual knob?
5. Can it be data/configuration instead of another branch?
6. Can it move to a research PR without reducing stable parity?
7. Does it improve runtime, memory, differentiability, or clarity enough to pay
   for its complexity?

Weak answers to 1, 2, or 7 mean `delete`, `merge`, or `extract-pr`. Lines for an
extracted feature move as a family: imports, constants, env vars, tests, docs,
fixtures, examples, and source hooks in the same tranche.

### Mandatory Line-Review Ledger

The review is executed through `tests/fixtures/core_slim_inventory.json`, not
ad-hoc notes. It must cover every tracked Python file and high-line non-Python
file affecting docs, examples, CI, packaging, or release artifacts.

Each row records `path`, `owner_domain`, `public_surface`, `callers`, `proof`,
`runtime_role`, `autodiff_role`, `decision`, `target`, and one-sentence
`justification`. `owner_domain` is one of public_api, cli_io, namelist,
geometry, discretization, physics, operators, rhs1_profile,
transport_coefficients, ambipolar, outputs, plotting, solver_core,
solver_research, validation, examples, tests, docs, release_tooling, or delete.

Rows with no public surface, no runtime role, and no focused proof are deleted
or extracted unless they are private helpers inside one retained equation or
method block. Env-only routes are extracted unless promoted to documented
advanced API options with tests and unchanged automatic defaults.

### Per-Line Decision Tags

During each tranche, classify touched lines as `PHYSICS`, `NUMERICS`, `API`,
`EVIDENCE`, `PERF`, `AUTODIFF`, `COMPAT`, `RESEARCH`, or `OBSOLETE`.
`RESEARCH` means promising but not stable/parity-clean/runtime-clean; extract.
`OBSOLETE` means duplicate, unreachable, historical, or no caller/proof; delete.

The retained source should read as physics and numerics, not as a history of
failed experiments. `RESEARCH` and `OBSOLETE` tags must not remain in core code
at review handoff.

### Complexity-Reduction Heuristics

For every retained function, prefer these before adding logic: inline one-call
wrappers unless they name an equation/API boundary; replace Boolean clusters
with one enum/config; delete non-user-visible branches; replace duplicated
diagnostic dictionaries with one typed builder; keep equations near assembly;
avoid generic `probe`, `candidate`, `rescue`, and `legacy` modules; collapse
route selection to one policy table per problem family; use compact data
fixtures instead of campaign scripts; separate fast CLI-only paths from
differentiable Python paths when that removes complexity; delete compatibility
after migration tests prove no retained caller uses it.

## Source Structure Rules

Package root remains public API/CLI/I/O/plotting/namelist/paths/solver facades
plus `README.md`. Implementation stays in one-level domain packages:
`discretization`, `geometry`, `operators`, `outputs`, `physics`, `problems`,
`solvers`, `validation`, and `workflows`. Avoid stable names based on historical
details (`v3_`, broad `rhs1_`, `probe`, `campaign`, `qi_*`) unless they are thin
compatibility facades with no logic.

Non-package targets:

- `examples/`: original Fortran-v3 references plus <=10 curated workflows: CLI
  run/plot, Python solve, geometry loading, transport coefficients, ambipolar
  solve, autodiff, bootstrap/Redl, optimization, output/plotting, frozen parity.
- `tests/`: grouped into unit, physics, regression, cli_io, optional
  integration, and fixtures; fewer files, less scaffolding, >=95% coverage.
- `scripts/`: zero by default; user commands become examples, validators become
  tests, and release-maintenance commands need docs and tests.
- `benchmarks/`: absent; benchmark claims use compact fixtures, bounded tests,
  docs, or release assets.

## Target Stable Source Shape

The final stable tree should have these modules and no deeper nesting. File
counts are budgets, not aspirations; exceeding a budget requires an inventory
exception with a test and retained workflow.

| Domain | Budget | Retain | Delete/extract |
| --- | ---: | --- | --- |
| Package root | <=14 files | `api`, `cli`, `io`, `namelist`, `plotting`, `solver` facade, `paths`, `sensitivity`, `ambipolar`, compatibility imports | historical wrappers, duplicate compare/profiling helpers without public docs |
| `geometry/` | <=5 files | VMEC/wout/Boozer/adapters with tests | generated campaigns and one-off validation loaders |
| `discretization/` | <=5 files | grids, velocity maps, periodic stencils | historical v3 wrappers with no direct caller |
| `physics/` | <=4 files | collision/transport formulas and normalizations | duplicated formula copies in problem modules |
| `operators/` | <=8 files | profile system assembly, collision/drift/electric/field terms, sparse pattern | true-operator rescue and experimental operator probes |
| `problems/` | <=10 files | profile solve, transport solve, ambipolar solve, setup, diagnostics, policies | QI/device-QI, native sparse-direct research, duplicated route orchestration |
| `solvers/` | <=10 files | Krylov dispatch, explicit sparse primitives, path policy, stable preconditioners, implicit differentiation | unpromoted direct-factor/ND/HSS/multifrontal and hard-seed experiments |
| `outputs/` | <=4 files | writer, formats, profile/transport output schemas | solver-experiment diagnostics keys |
| `validation/` | <=4 files | release-data fetch, compact artifacts, Fortran fixture helpers | publication campaign generators and QI device validators |
| `workflows/` | <=4 files | curated high-level Python workflows | campaign wrappers and research optimization experiments |

The target package is therefore roughly <=68 files by first review, then <=50
after compatibility deletion. The preferred end state is not many tiny files; it
is a small number of domain files whose names match the physics/numerics.

## Extraction Map And File Budgets

| Family | Stable action | Destination / gate |
| --- | --- | --- |
| QI/device-QI hard-seed machinery | Remove policy helpers, env tokens, examples, tests, docs from core; keep only research pointer. | `research/qi-device-hard-seed`; returns after CPU/GPU production residual/runtime/RSS/differentiability gates. |
| Native sparse-direct, reduced-Pmat, nested-dissection, multifrontal, HSS | Extract source/tests/docs and default hooks; keep only sparse primitives used by stable defaults. | `research/native-sparse-direct`; returns after true-residual admission and production-floor runtime/RSS gates. |
| Parallel/GPU/multi-device campaigns | Keep only public helpers with docs/tests; move runners, traces, raw regenerators out. | `research/parallel-performance` or release assets. |
| Publication/audit generators | Keep final compressed figures only when referenced; move generators/raw outputs out. | `research/publication-audits`. |
| Legacy wrappers | Thin delegates if public; otherwise delete. | Stable only with migration/import tests. |
| Env-only solver branches | Promote to documented advanced options with gates or extract/delete. | Stable only if automatic defaults stay simple. |
| Duplicated test scaffolds | Merge into physics/numerics/regression tests; delete extracted-path tests. | Stable CI under 10 minutes. |
| Single-file example/campaign folders | Merge into curated examples or extract/delete. | Stable examples readable in minutes. |

Review budgets requiring inventory exceptions if exceeded:

- `profile_policies.py`: <=1,500 lines after QI/device-QI residue deletion.
- `profile_sparse_xblock.py`: <=2,500 lines after experimental rescue extraction.
- `profile_full_system.py`: <=3,000 lines with physics assembly preserved.
- `explicit_sparse.py`: <=2,000 lines; keep primitives, extract research factors.
- `profile_sparse_solve.py` + `profile_solve.py`: <=4,000 combined lines.
- Transport solve/policy/linear-system family: <=4,000 combined lines.
- `solver.py`: <=800 lines as a facade.
- `outputs/*`: <=4,000 combined lines.
- `validation/*`: <=2,500 combined lines.
- `tests/test_profile_response_sparse_pc.py` and related solver tests:
  <=4,000 combined lines after extracted-path tests are removed.

## Ordered Finish Plan

These tranches replace open-ended "continue refactoring" work. Each tranche is
large enough to change reviewability, but bounded enough to validate locally.
Do not start a later tranche if an earlier one leaves stale imports, stale
tests, or stale README/docs claims.

### Tranche 1 - Finish Research Extraction From Core

Scope: QI/device-QI, native sparse-direct/reduced-Pmat rescue, parallel
campaigns, and publication/audit generators.

Required actions:

- Remove any remaining imports, env tokens, diagnostics, examples, tests, and
  docs that expose extracted QI/device-QI as a stable feature.
- Extract or delete unpromoted direct-factor/nested-dissection/HSS/multifrontal
  hooks unless they are part of the retained default sparse primitive layer.
- Remove long-run profiler/campaign scripts from stable docs and tests.
- Update `research_lanes.rst` with one short pointer per extracted branch.

Acceptance gate:

- `rg "qi_device|device-QI|true_operator_rescue|native_sparse_direct|nested_dissection|multifrontal|HSS|campaign" sfincs_jax tests examples scripts README.md docs`
  returns only explicit research-lane text or retained generic sparse terms.
- Net source/test/example/script lines decrease materially.
- Focused solver, docs-claim, and source-tree tests pass.

### Tranche 2 - Collapse RHSMode-1 Stable Solve Path

Scope: `profile_policies.py`, `profile_sparse_xblock.py`,
`profile_sparse_solve.py`, `profile_solve.py`, `profile_dense.py`,
`profile_preconditioner_build.py`, and related solver tests.

Required actions:

- Reduce route selection to one default policy table and one advanced-options
  parser; delete duplicated candidate/probe/rescue branches.
- Keep a small dense path, a strict sparse/x-block path, and explicit residual
  admission; extract everything else.
- Move repeated diagnostic metadata construction into one builder.
- Make the solve code read in this order: setup, operator/RHS assembly,
  preconditioner selection, Krylov/direct solve, residual validation,
  diagnostics/output.

Acceptance gate:

- `profile_policies.py <= 2500` lines, `profile_sparse_xblock.py <= 2800`,
  `profile_sparse_solve.py <= 2300`, `profile_solve.py <= 1800`, or each excess
  line has an inventory exception.
- Supported RHSMode-1 smoke/parity fixtures pass with no solver env vars.
- Solver policy tests prove automatic defaults choose a stable path.

### Tranche 3 - Collapse Transport/RHSMode-2/3 Path

Scope: `transport_linear_system.py`, `transport_solve.py`,
`transport_policies.py`, `preconditioner_transport_matrix.py`, transport
diagnostics/finalization, and transport tests.

Required actions:

- Keep one transport linear-system builder, one policy table, and stable
  preconditioners that satisfy residual gates.
- Extract parallel worker/campaign code unless it is a documented, tested public
  helper.
- Remove duplicated RHS assembly/diagnostic logic already present in operators
  or outputs.

Acceptance gate:

- Transport solve/policy/linear-system family <=4k combined lines or inventory
  exceptions.
- GeometryScheme 2/11 compact production-floor fixtures pass strict residual
  gates.
- README/docs performance claims use checked fixtures, not generated campaign
  scripts.

### Tranche 4 - Move Physics Equations Into Clear Domain Modules

Scope: `operators/profile_full_system.py`, `operators/profile_system.py`,
collision/drift/electric/ExB/magnetic terms, `physics/*`, and normalization
helpers.

Required actions:

- Keep formulas where the physics is visible: term name, equation reference,
  units/normalization, and shape contract.
- Delete duplicated normalization or drift/collision formula copies.
- Replace long internal helper chains with small equation blocks and typed
  assembly records.

Acceptance gate:

- `profile_full_system.py <= 3000` lines or inventory exception.
- Unit tests check term-level identities, conservation/moment relations, and
  finite-beta/electric-field switches.
- Method docs link each retained equation to its source module.

### Tranche 5 - Shrink Outputs, Validation, Examples, Scripts

Scope: `outputs/*`, `validation/*`, `examples/`, `scripts/`, docs/static
claim data, and tests that refer to these paths.

Required actions:

- Keep only output schema fields that are public, parity-required, or plotted.
- Move publication/raw benchmark generators out; keep compact JSON/HDF5 fixtures
  only when they are small and directly tested.
- Reduce examples to original Fortran-v3 references plus <=10 curated workflows.
- Empty `scripts/` unless a script is documented release tooling with tests.

Acceptance gate:

- `examples/README.md` is a curated workflow guide, not a file dump.
- `scripts/` has zero undocumented commands.
- No tracked file exceeds 2 MB; no ignored output is staged.

### Tranche 6 - Test Consolidation And Coverage To 95%

Scope: all `tests/`.

Required actions:

- Reorganize tests into unit, physics, regression, cli_io, fixtures, and
  optional integration without increasing total test lines.
- Delete tests for extracted research paths.
- Prefer fewer parametrized tests over many one-off files.
- Raise coverage primarily by deleting dead code, then by targeted physics and
  numerical tests.

Acceptance gate:

- Default CI under 10 minutes.
- Coverage >=95% on slim core.
- Tests include drift-kinetic identities, residual gates, geometry
  interpolation, Redl/bootstrap normalization, ambipolar derivatives,
  Fortran-v3 frozen fixtures, output schemas, CLI, plotting, and autodiff.

### Tranche 7 - Evidence Regeneration And Documentation

Scope: README, docs, package README, checked figures/tables, parity/runtime/RSS
fixtures.

Required actions:

- Rewrite public docs as standalone software documentation, not branch history.
- Regenerate only supported CPU evidence locally; GPU evidence is refreshed when
  an available GPU is confirmed.
- Remove public claims for extracted research lanes from README.

Acceptance gate:

- README has install, one command, Python API, physics summary, curated
  examples, parity/performance figure, and clear unsupported/deferred research
  language.
- Docs build with warnings as errors.
- Checked tables/figures match fixtures.

### Tranche 8 - Review Handoff

Scope: final PR #8 state.

Required actions:

- Run full tests, coverage, docs build, examples smoke, size guard, and diff
  hygiene.
- Commit a final inventory with before/after counts.
- Push PR #8 and leave it unmerged for review.

Acceptance gate:

- Branch is clean, PR is green, no generated clutter is tracked, and every
  remaining exception to file/line budgets is explicitly justified.

## Audit Order And Tranche Gates

Audit in this order: public surface; physics; discretization/operators; solver
defaults; differentiability; validation/tests; examples/scripts/benchmarks;
docs. Each tranche edits the inventory first, removes/merges/extracts code, runs
focused guards, and reports net file/line change, tests, static checks,
parity/performance claim changes, and lane completion. No tranche is accepted if
it only moves complexity without reducing retained lines or user-facing concepts.

Final budgets: <=50 package Python files, <=50k source lines unless justified,
<=120 test files, <=10 curated workflows, zero benchmark tree, zero undocumented
scripts, no tracked artifact over 2 MB, >=95% slim-core coverage, and fresh
parity/runtime/RSS/bootstrap evidence for supported examples.

## Per-Tranche Operating Loop

Every tranche follows the same loop: update inventory, remove/merge/extract/
replace code, remove matching tests/docs/fixtures/examples/env vars/diagnostics,
run focused checks, record net tracked file/line changes and lane percentages,
then commit and push before the next tranche.

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
- Core source is <=50 Python files and <=50k lines, or each exception is
  justified; experimental solver/preconditioner/QI/profiling lanes are separate
  PRs or deleted.
- `examples/` keeps original Fortran-v3 references plus <=10 curated workflows;
  `benchmarks/` is absent; `scripts/` is empty or documented release tooling.
- Tests are organized, smaller, scientifically meaningful, >=95% coverage, and
  CI stays under 10 minutes.
- Supported examples have fresh Fortran-v3 parity, runtime, memory, and
  bootstrap-current evidence; README/docs match the slim core; PR #8 is clean.
## Explicit Deferred Items
Deferred unless production-gated: experimental QI/device-QI, native sparse-direct
research, lower-memory preconditioners, GPU/multi-GPU campaigns, and audits.
