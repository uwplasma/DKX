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

- Latest audited code-change head: `8fd06b85`; last green CI evidence: `96fb2677`.
  Exact aggregate coverage from that CI was 91.753% (`92%`).
- Package layout is shallow, but the branch is too large: 114 Python source
  files and about 142.4k source lines. Largest complexity owners remain
  `problems/`, `solvers/`, RHSMode-1/QI/preconditioner infrastructure, and
  compatibility layers around those paths.
- Non-package Python volume is still too high: `tests/` has 313 Python files
  and about 127.3k lines, `examples/` has 122 Python files, and `scripts/` has 12 Python files.
  Top-level `benchmarks/` is removed from the active tracked tree; compact
  Fortran-v3 references live in `tests/fixtures/fortran_v3_reference_fixture.json`.
- No tracked file larger than 2 MB was found. There is no tracked
  `sfincs_jax/data/`; release-data logic is `validation/data_fetch.py`.
- The product target changed: PR #8 should become a small stable core, while
  experimental or not-yet-competitive work moves to separate research PRs.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 28% | Every retained file/function/line has a core reason, a caller, and a test/doc owner; everything else is extracted or deleted. |
| Core-main slimming | Active | 44% | Main keeps only stable, parity-clean, runtime-acceptable solvers and public APIs; research code is outside core. |
| Source simplification | Active | 48% | Package moves toward <=50 source files and <=50k lines, with a 10x reduction as stretch if functionality permits. |
| Examples/tests/scripts cleanup | Active | 77% | Examples are <=10 curated workflows plus Fortran-v3 references; tests are smaller, organized, and >=95% coverage; scripts are removed or promoted; benchmarks are gone. |
| Parity/performance evidence | Active | 70% | Supported examples rerun against SFINCS Fortran v3 with runtime/RSS/bootstrap evidence; unsupported research lanes are not marketed. |
| Docs/readme regeneration | Active | 80% | README/docs describe only the stable core plus explicit external research PR lanes. |

Overall readiness under this stricter core-slim goal is about 80-84%.

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

## Audit Order And Tranche Gates

Audit in this fixed order: public surface; physics models; discretization and
operators; solver defaults; differentiability; validation/tests; examples,
scripts, and benchmarks; docs. Each tranche must edit the inventory first,
remove/merge/extract code, run focused guards, and report net file/line change,
modules changed, tests run, static checks, parity/performance claim changes, and
remaining lane completion. No tranche is accepted if it only creates more files
or moves complexity without reducing user-visible complexity or total lines.

Final budgets: <=50 package Python files and <=50k source lines unless every
exception is justified, <=120 test files, <=10 curated workflows, zero benchmark
tree, zero undocumented scripts, no tracked artifact over 2 MB, >=95% slim-core
coverage, and fresh parity/runtime/RSS/bootstrap evidence for supported
examples.

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
