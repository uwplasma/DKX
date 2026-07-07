# SFINCS_JAX Core-Slim Review Plan

Last updated: 2026-07-07

Active branch / PR: `refactor/v3-driver-architecture` / PR #8

This is the single active plan for the refactor branch. `plan.md` is the historical execution log. Do not create another competing plan.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core where one
input file plus geometry runs accurate CPU/GPU neoclassical calculations with
automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime and memory, evident physics/numerics in the source, and
validated differentiable Python workflows for sensitivities, ambipolar solves,
bootstrap current, transport coefficients, plotting, and optimization.

## Current Review State

- Latest audited code-change head: `69de75cb+true-operator-stage-removal`; last green CI evidence:
  `96fb2677`. Exact aggregate coverage from that CI was 91.753% (`92%`).
- Current tracked Python volume is the problem to solve before review:
  114 package files / 139.3k source lines, 313 test files / 126.0k test lines,
  122 example Python files, and 12 tracked scripts.
- Largest source owners are the first audit targets:
  `problems/profile_sparse_xblock.py`, `problems/profile_policies.py`,
  `operators/profile_full_system.py`, `solvers/explicit_sparse.py`,
  `problems/profile_sparse_solve.py`, `problems/profile_solve.py`,
  `problems/transport_linear_system.py`, `problems/profile_sparse_direct.py`,
  `problems/transport_parallel_runtime.py`, `problems/profile_dense.py`,
  `solver.py`, `outputs/rhsmode1.py`, and
  `solvers/preconditioner_transport_matrix.py`.
- Largest test owners are the second audit targets:
  `tests/test_profile_response_sparse_pc.py`, `tests/test_rhs1_full_assembly.py`,
  `tests/test_io_output_policy_coverage.py`, `tests/test_v3_sparse_pattern.py`,
  `tests/test_explicit_sparse.py`, `tests/test_rhs1_solver_replay.py`, dense
  profile tests, transport policy tests, and all extracted-path tests.
- The stable branch must stop carrying the history of solver experiments.
  Experimental QI/device-QI, native sparse-direct, nested-dissection,
  multifrontal, HSS/BLR, long profiling campaigns, and publication generators
  are preserved only in research PRs unless they satisfy the stable admission
  gates below.
- No tracked file larger than 2 MB was found. Ignored local outputs,
  `__pycache__`, and generated traces must not be committed.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 43% | Every retained file/function/line has a caller, a physics/numerics/API reason, and a test/doc/perf owner. |
| Core-main slimming | Active | 57% | Main keeps only stable, parity-clean, runtime-acceptable solvers/APIs; research code is outside core. |
| Source simplification | Active | 57% | Package moves first to <=68 files, then <=50 files and <=50k lines unless exceptions are justified. |
| Tests/examples/scripts cleanup | Active | 77% | Examples are curated, tests are smaller but reach >=95% coverage, scripts are removed or documented release tooling. |
| Parity/performance evidence | Active | 70% | Supported examples have checked SFINCS Fortran v3 parity/runtime/RSS/bootstrap evidence. |
| Docs/readme | Active | 80% | Public docs describe stable software, not branch history or unpromoted campaigns. |

Overall readiness under this stricter core-slim goal is about 84-87%.

## Concrete Code-Audit Rules

Default action for every line is removal. A retained line needs one of these
proofs:

- `PHYSICS`: it implements a documented drift-kinetic, collision, geometry,
  normalization, flux, bootstrap, ambipolar, or transport equation.
- `NUMERICS`: it implements a retained discretization, residual, linear solve,
  preconditioner, error estimate, output schema, or differentiability method.
- `API`: it is part of a documented public Python/CLI/namelist/output contract.
- `EVIDENCE`: it is required by a compact parity, physics, regression,
  coverage, benchmark, or documentation gate.
- `PERF`: it measurably reduces runtime or memory for a supported default path.
- `AUTODIFF`: it is needed for differentiable solves, sensitivities, or
  optimization through a supported stable workflow.

Everything else is one of:

- `COMPAT`: keep only as a thin import/argument shim with a migration test and
  deletion date.
- `RESEARCH`: move to a research PR, remove stable imports/env vars/tests/docs,
  and leave at most one docs pointer.
- `OBSOLETE`: delete together with imports, tests, docs, fixtures, examples,
  env vars, diagnostics, and generated artifacts.

For each file, use this exact loop:

1. List public symbols and callers with `rg`.
2. Mark each top-level object `PHYSICS`, `NUMERICS`, `API`, `EVIDENCE`, `PERF`,
   `AUTODIFF`, `COMPAT`, `RESEARCH`, or `OBSOLETE`.
3. Delete unreachable, one-call, historical, duplicated, and env-only objects.
4. Merge wrappers that do not name a real equation/API boundary.
5. Move formulas toward `physics/`, `operators/`, or `discretization/`; move
   route orchestration toward one policy table per problem family.
6. Remove matching tests/docs/examples/fixtures for deleted or extracted paths.
7. Run the smallest tests that would fail if the retained line mattered.
8. Record line/file deltas in `tests/fixtures/core_slim_inventory.json`.

No touched function may remain without an obvious caller and a testable reason.
No public docs sentence may describe a non-default research path as supported.
No solver branch may be stable only because an environment variable can reach it.

## Stable-Core Admission Gates

A solver, preconditioner, workflow, example, or documentation claim stays on
this branch only if it satisfies all relevant gates: strict residual or frozen
Fortran-v3 parity; useful runtime; bounded local CPU memory; device-safe GPU
claim or explicit GPU exclusion; JAX-compatible differentiable Python path or
isolated non-differentiable CLI path; no required internal solver/env knowledge;
compact tests under CI budget; and docs that state the exact supported scope.

Any path failing one gate is `RESEARCH`, not stable core.

## Mandatory Line-Review Ledger

`tests/fixtures/core_slim_inventory.json` is the line-review ledger. It records
each retained, extracted, or deleted large family with `path`, `owner_domain`,
`public_surface`, `callers`, `proof`, `runtime_role`, `autodiff_role`,
`decision`, `target`, and `justification`.

Allowed owner domains are public_api, cli_io, namelist, geometry,
discretization, physics, operators, rhs1_profile, transport_coefficients,
ambipolar, outputs, plotting, solver_core, solver_research, validation,
examples, tests, docs, release_tooling, and delete.

Rows with no public surface, no runtime role, and no focused proof are deleted
or extracted unless they are private helpers inside one retained equation block.

## Source Structure Rules

The package root is for public API/CLI/I/O/plotting/namelist/paths/solver
facades plus `README.md`. Implementation stays in one-level domain packages:
`discretization`, `geometry`, `operators`, `outputs`, `physics`, `problems`,
`solvers`, `validation`, and `workflows`.

Stable module names should describe physics or numerics, not history. Avoid
new stable names based on `v3_`, broad `rhs1_`, `probe`, `campaign`, `rescue`,
`candidate`, `legacy`, or `qi_*`. Existing names with those terms are debt and
must be deleted, renamed, or justified as temporary compatibility.

Non-package targets: `examples/` keeps original SFINCS Fortran v3 references
plus <=10 curated workflows; `tests/` is grouped into unit, physics,
regression, cli_io, optional integration, and fixtures with >=95% coverage;
`scripts/` is zero by default except documented release tooling; `benchmarks/`
is absent and benchmark claims use compact fixtures, docs, or release assets.

## Target Stable Source Shape

| Domain | Budget | Retain | Delete/extract |
| --- | ---: | --- | --- |
| Package root | <=14 files | public facades, CLI, I/O, namelist, plotting, paths, sensitivity, ambipolar | duplicate compare/profiling helpers, hidden release scripts |
| `geometry/` | <=5 files | VMEC/wout/Boozer/JAX adapters with tests | generated campaigns, one-off loaders |
| `discretization/` | <=5 files | grids, velocity maps, periodic stencils | historical wrappers with no direct caller |
| `physics/` | <=4 files | collisions, classical transport, Redl/bootstrap normalizations | formula copies in problem modules |
| `operators/` | <=8 files | collision/drift/electric/field/system assembly | true-operator rescue/probe experiments |
| `problems/` | <=10 files | profile solve, transport solve, ambipolar solve, setup, diagnostics, policies | QI/device-QI, native sparse-direct, duplicated route orchestration |
| `solvers/` | <=10 files | Krylov dispatch, sparse primitives, path policy, stable preconditioners, implicit diff | ND/HSS/multifrontal/hard-seed experiments |
| `outputs/` | <=4 files | writer, formats, profile and transport schemas | experiment-only diagnostics |
| `validation/` | <=4 files | release-data fetch, compact artifacts, Fortran fixture helpers | publication campaign generators |
| `workflows/` | <=4 files | curated high-level workflows | campaign wrappers and research optimization experiments |

First review budget: <=68 package Python files. Final budget: <=50 package
Python files and <=50k package lines unless each exception is justified in the
ledger.

## Exact Source Backlog

### Delete or Extract First

- `profile_sparse_solve.py`: disconnected true-operator residual rescue stage
  APIs/tests are removed; next collapse generic branch setup and direct-tail
  policy wiring that still belongs to extracted native sparse-direct research.
- `profile_sparse_direct.py`: keep only stable sparse direct policy that passes
  admission gates; move native direct-factor research to
  `research/native-sparse-direct`.
- `solvers/native_block_factor.py` and symbolic preconditioner files: retain
  only primitives used by stable defaults; extract ND/HSS/multifrontal research.
- `transport_parallel_runtime.py`: keep public serial/default helper if needed;
  extract multi-worker/GPU campaign runners and trace plumbing.
- Publication/performance campaign examples: move to research/release assets
  unless they are one of the curated workflows.
- `scripts/`: delete, convert to tests, or move to examples/release tooling.

### Collapse Next

- `profile_policies.py`: one default policy table, one advanced-options parser,
  no duplicated probe/candidate/rescue logic.
- `profile_sparse_xblock.py`: stable x-block/preconditioner path only; remove
  research hooks and duplicate diagnostics.
- `profile_solve.py` and `profile_sparse_solve.py`: read as setup, assembly,
  policy selection, solve, residual validation, diagnostics/output.
- `solver.py`: shrink to facade plus public contracts; move implementation into
  domain modules.
- `outputs/rhsmode1.py` and `outputs/writer.py`: one schema builder and one
  writer path per format; remove experiment-only output keys.

### Move Equations to Domain Modules

- `operators/profile_full_system.py`: retain equation blocks, shape contracts,
  and sparse/JAX assembly; move duplicated normalizations to `physics/`.
- Drift/electric/collision modules: consolidate repeated term assembly and keep
  tests close to equation identities.
- `physics/collisions.py` and `physics/classical_transport.py`: own collision,
  Redl/bootstrap, and transport normalization formulas.

### Transport Cleanup

- `transport_linear_system.py`, `transport_solve.py`, `transport_policies.py`,
  `preconditioner_transport_matrix.py`: one linear-system builder, one policy
  table, stable preconditioners only, shared diagnostics/output builders.

## Extraction Map

| Family | Stable action | Destination / return gate |
| --- | --- | --- |
| QI/device-QI hard-seed machinery | Remove policy helpers, env tokens, examples, tests, docs; leave research pointer only. | `research/qi-device-hard-seed`; returns after CPU/GPU production residual/runtime/RSS/autodiff gates. |
| Native sparse-direct, reduced-Pmat, ND, multifrontal, HSS | Extract source/tests/docs/default hooks; keep stable sparse primitives only. | `research/native-sparse-direct`; returns after true-residual admission and production-floor runtime/RSS gates. |
| Parallel/GPU/multi-device campaigns | Keep only documented public helpers with tests; move runners/traces/regenerators out. | `research/parallel-performance` or release assets. |
| Publication/audit generators | Keep final compressed figures only when referenced; move generators/raw outputs out. | `research/publication-audits`. |
| Legacy wrappers | Thin delegates only if public; otherwise delete. | Stable only with migration/import tests. |
| Env-only solver branches | Promote to documented advanced API or extract/delete. | Stable only if automatic defaults remain simple. |
| Duplicated test scaffolds | Merge into focused physics/numerics/regression tests. | Stable CI under 10 minutes. |

## Ordered Finish Plan

### Tranche 1 - Finish Research Extraction From Core

Scope: QI/device-QI residue, true-operator rescue residue, native sparse-direct
research residue, long profiler/campaign scripts, and publication generators.

Actions:

- Keep deleted true-operator rescue stage APIs/tests out of stable core.
- Remove stable imports/env vars/docs for QI/device-QI and native direct-factor
  research.
- Move or delete long-run profiler/campaign scripts and generated example
  outputs.
- Update `docs/research_lanes.rst` with short research pointers only.

Gate:

```bash
rg "qi_device|device-QI|true_operator_rescue|native_sparse_direct|nested_dissection|multifrontal|HSS|campaign" sfincs_jax tests examples scripts README.md docs
```

returns only research-lane text, generic sparse terminology, or retained stable
features with ledger entries. Net source/test/example/script lines decrease.

### Tranche 2 - Collapse RHSMode-1 Stable Solve Path

Scope: profile policies, sparse/x-block solve, dense solve, setup,
preconditioner build, diagnostics, and related tests.

Actions:

- Replace duplicated candidate/probe/rescue branches with one default policy
  table and one advanced-options parser.
- Keep dense, sparse/x-block, residual validation, and output diagnostics.
- Merge repeated diagnostics dictionaries into one typed builder.
- Delete tests for extracted paths; keep parity/residual/policy tests for
  stable defaults.

Gate:

- `profile_policies.py <= 2500` lines.
- `profile_sparse_xblock.py <= 2800` lines.
- `profile_sparse_solve.py <= 2300` lines.
- `profile_solve.py <= 1800` lines.
- Supported RHSMode-1 fixtures pass without solver env vars.

### Tranche 3 - Collapse Transport/RHSMode-2/3 Path

Scope: transport linear system, transport solve, transport policy, transport
preconditioner, transport diagnostics/finalization, and tests.

Actions:

- Keep one transport linear-system builder and one policy table.
- Extract multi-worker/parallel campaign logic unless it is a documented public
  helper.
- Reuse output and residual diagnostics from shared builders.

Gate:

- Transport solve/policy/linear-system family <=4k combined lines or ledger
  exceptions.
- GeometryScheme 2/11 compact production-floor fixtures pass strict residual
  gates.

### Tranche 4 - Make Physics And Numerics Evident

Scope: operators, physics, discretization, normalizations, and term-level tests.

Actions:

- Keep each equation in a named block with units/normalization and shape
  contract.
- Delete duplicated formula copies.
- Add/retain focused tests for collision symmetry, conservation moments, drift
  switches, electric-field terms, Redl/bootstrap normalization, geometry
  interpolation, residual JVP, and ambipolar derivatives.

Gate:

- `profile_full_system.py <= 3000` lines or ledger exception.
- Docs link retained equations to source modules.

### Tranche 5 - Shrink Outputs, Validation, Examples, Scripts

Scope: outputs, validation, examples, scripts, docs/static claim data, and tests.

Actions:

- Keep only output fields that are public, parity-required, or plotted.
- Reduce examples to original Fortran-v3 references plus <=10 curated workflows.
- Convert scripts to examples/tests/release tooling or delete.
- Move raw benchmark/publication generators out of stable core.

Gate:

- `examples/README.md` is a curated workflow map.
- `scripts/` contains zero undocumented commands.
- No tracked file exceeds 2 MB.

### Tranche 6 - Test Consolidation And Coverage To 95%

Scope: all tests.

Actions:

- Merge one-off solver tests into parametrized unit/physics/regression suites.
- Delete tests for extracted research paths.
- Raise coverage first by deleting dead code, then by targeted meaningful tests.
- Keep default CI under 10 minutes.

Gate:

- Coverage >=95% on the slim core.
- Test files <=120 unless each exception is justified.
- Tests include physics gates, numerical identities, frozen Fortran-v3 fixtures,
  CLI/output/plotting, and autodiff.

### Tranche 7 - Evidence Regeneration And Documentation

Scope: README, docs, package README, checked figures/tables, parity/runtime/RSS
fixtures.

Actions:

- Rewrite public docs as standalone software documentation, not branch history.
- Regenerate supported CPU evidence locally; refresh GPU evidence only when a
  GPU is available.
- Remove public claims for extracted research lanes from README.

Gate:

- README has install, one command, Python API, physics summary, curated
  examples, parity/performance figure, and explicit research-lane boundaries.
- Docs build with warnings as errors.
- Checked tables/figures match fixtures.

### Tranche 8 - Review Handoff

Scope: final PR #8 state.

Actions:

- Run full tests, coverage, docs build, examples smoke, size guard, and diff
  hygiene.
- Commit final inventory with before/after counts.
- Push PR #8 and leave it unmerged for review.

Gate:

- Branch is clean, PR is green, no generated clutter is tracked, and every
  remaining exception to file/line budgets is justified.

## Per-Tranche Operating Loop

Every tranche: update inventory decisions; remove/merge/extract code plus
matching tests/docs/fixtures/examples/env vars/diagnostics; run focused/static
checks; record net file/line changes and lane percentages; commit and push. A
tranche is rejected if it only moves code without reducing retained lines,
files, public knobs, or duplicated concepts.

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

Before review, also run the full test suite, coverage with `--fail-under=95`,
docs with `sphinx -W`, and the quick CLI/plot example.

## Completion Gates

- Core source is <=50 Python files and <=50k lines, or every exception is justified in the ledger.
- Experimental solver/preconditioner/QI/profiling lanes are separate PRs or deleted from stable.
- `examples/` keeps original Fortran-v3 references plus <=10 curated workflows.
- `benchmarks/` is absent; `scripts/` is empty or documented release tooling.
- Tests are smaller, meaningful, >=95% coverage, and default CI stays under 10 minutes.
- Supported examples have fresh Fortran-v3 parity/runtime/memory/bootstrap evidence.
- README/docs match the slim core and do not market research paths as stable.
- PR #8 is clean and ready for review.

## Explicit Deferred Items

Deferred unless production-gated: experimental QI/device-QI, native sparse-direct research, nested-dissection/multifrontal/HSS replacements, lower-memory preconditioner research, GPU/multi-GPU campaigns, and publication audits. They may be referenced in `docs/research_lanes.rst` only; they should not remain as stable source, examples, tests, README claims, or default solver branches.
