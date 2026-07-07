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

- Latest audited code-change head: `185b819a7098`
  (`Remove direct-tail support-mode experiment`);
  last green CI evidence: `96fb2677`. Exact aggregate coverage from that CI was
  91.753% (`92%`); this tranche has focused local evidence listed below.
- Current tracked Python volume is the problem to solve before review:
  115 package files / 137.6k source lines, 313 test files / 125.3k test lines,
  122 example Python files, and 7 tracked Python scripts after consolidating
  lightweight release/data/audit scripts into package validation modules.
- Largest source owners are the first audit targets: `profile_sparse_xblock.py`, `profile_policies.py`, `profile_full_system.py`, `explicit_sparse.py`, `profile_sparse_solve.py`, `profile_solve.py`, `transport_linear_system.py`, `profile_sparse_direct.py`, `transport_parallel_runtime.py`, `profile_dense.py`, `solver.py`, `outputs/rhsmode1.py`, and `preconditioner_transport_matrix.py`.
- Largest test owners are the second audit targets: `test_profile_response_sparse_pc.py`, `test_rhs1_full_assembly.py`, `test_io_output_policy_coverage.py`, `test_v3_sparse_pattern.py`, `test_explicit_sparse.py`, `test_rhs1_solver_replay.py`, dense profile tests, transport policy tests, and all extracted-path tests.
- The stable branch must stop carrying the history of solver experiments.
  Experimental QI/device-QI, native sparse-direct, nested-dissection,
  multifrontal, HSS/BLR, long profiling campaigns, and publication generators
  are preserved only in research PRs unless they satisfy the stable admission
  gates below.
- No tracked file larger than 2 MB was found. Ignored outputs, `__pycache__`, and generated traces must not be committed.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 45% | Every retained file/function/line has a caller, a physics/numerics/API reason, and a test/doc/perf owner. |
| Core-main slimming | Active | 59% | Main keeps only stable, parity-clean, runtime-acceptable solvers/APIs; research code is outside core. |
| Source simplification | Active | 59% | Package moves first to <=68 files, then <=50 files and <=50k lines unless exceptions are justified. |
| Tests/examples/scripts cleanup | Active | 77% | Examples are curated, tests are smaller but reach >=95% coverage, scripts are removed or documented release tooling. |
| Parity/performance evidence | Active | 70% | Supported examples have checked SFINCS Fortran v3 parity/runtime/RSS/bootstrap evidence. |
| Docs/readme | Active | 80% | Public docs describe stable software, not branch history or unpromoted campaigns. |

- Focused local evidence for this tranche:
  `tests/test_v3_sparse_pattern.py` passed `93` tests in 93 s; the touched
  sparse-policy/output subset passed `554` tests in 136 s; the refactor/tree
  contract subset passed `76` tests; the release/data/audit subset passed `59`
  tests; package CLIs and Ruff/compileall/json/diff/size hygiene passed.

Overall readiness under this stricter core-slim goal is about 87-89%.

## Concrete Code-Audit Rules

The default action for every line is deletion. A line may remain only when it
has one of these owners and a matching proof:

| Owner | Retain only if the line... | Proof required |
| --- | --- | --- |
| `PHYSICS` | states or evaluates a DKE, collision, drift, geometry, flux, current, Redl, ambipolar, or normalization equation | equation test, parity fixture, or documented derivation |
| `NUMERICS` | builds a grid, stencil, operator, residual, preconditioner, Krylov solve, interpolation, quadrature, or convergence diagnostic | residual/identity test and runtime/RSS gate |
| `API` | is part of the stable Python API, CLI, namelist compatibility, plotting, or output schema | public test and docs reference |
| `AUTODIFF` | keeps a differentiable path, JVP/VJP/implicit derivative, or branch-safe differentiable wrapper | gradient test against finite difference or analytic identity |
| `PERF` | measurably reduces runtime, memory, JIT overhead, or output cost on supported examples | benchmark fixture or policy test |
| `EVIDENCE` | is a compact validation fixture, claim checker, or release evidence reader | schema test and docs claim |

Everything else is `RESEARCH`, `COMPAT`, `DUPLICATE`, `GENERATED`, or
`OBSOLETE`. `RESEARCH` moves to an existing preservation branch/PR or a new
research branch before deletion from this PR. `COMPAT` stays only if it is a
documented SFINCS Fortran v3 input/output compatibility surface. `DUPLICATE`,
`GENERATED`, and `OBSOLETE` are deleted with their imports, tests, docs, output
keys, env vars, fixtures, and examples.

Per file, complete this exact loop before any code movement:

1. Record file path, line count, public symbols, imports, callers, env vars,
   output keys, tests, docs, and examples in `tests/fixtures/core_slim_inventory.json`.
2. Mark the file as `keep`, `merge`, `delete`, or `extract-pr`.
3. For every retained public symbol, write the owner tag, stable caller, proof
   test, and docs owner. Symbols missing any item are removed or extracted.
4. Remove one-call wrappers unless the wrapper names a real physics/numerics/API
   boundary that makes the code easier to read.
5. Collapse duplicate diagnostics dictionaries, policy branches, shape helpers,
   namelist aliases, and solver option parsing into shared typed builders.
6. Delete path-specific tests when the path is extracted; keep compact absence
   tests so the stable core cannot accidentally re-import the research path.
7. Run the smallest proof tests plus `ruff`, `compileall`, `git diff --check`,
   and the size guard.
8. Commit only if retained file count, line count, public knobs, generated
   files, or test burden decreases. A pure move is not progress.

Budgets are hard unless the ledger explains the exception: 114 package Python
files and 137.0k source lines now; <=68 files and <=80k lines before review;
<=50 files and <=50k lines final. Tests move from 313 files and 125.3k Python
lines to <=180 files before review, then <=120 files with >=95% coverage.
Examples move from 122 Python scripts to original SFINCS Fortran v3 references
plus <=10 curated workflows. `scripts/` becomes empty by default; exceptions
must be documented release tooling. The largest five source files must fall
below 18k combined lines before review.

Stable owners are limited to public API/CLI/I/O/namelist/plotting/path facades,
geometry adapters, grid/stencil modules, physics formulas, operator assembly,
profile/transport/ambipolar problem modules, admitted solver/preconditioner
modules, output schemas, compact validation helpers, and curated workflows.
Stable file names must describe the domain boundary. New or retained stable
names based on `v3_`, broad `rhs1_`, `probe`, `campaign`, `rescue`,
`candidate`, `legacy`, `hard_seed`, `native`, `symbolic`, or `qi_*` need an
explicit ledger exception or are extracted.

Extraction is complete only when `rg` finds no stable import, public knob,
environment variable, output key, documentation claim, or example path for the
extracted family. README/docs may mention extracted work only in
`docs/research_lanes.rst` as unsupported research history.

## File Disposition Map

The next commits must follow this order. Each row is a deletion target, not a
move target, unless the destination is explicitly listed.

| Area | Current files/lines | Stable destination | Delete/extract | Proof before commit |
| --- | ---: | --- | --- | --- |
| Package root | 16 files | `api.py`, `cli.py`, `io.py`, `namelist.py`, `plotting.py`, `solver.py`, `paths.py`, `README.md` | merge `compare.py`, `diagnostics.py`, broad facades, and duplicate path helpers into root API or validation/output modules | API, CLI, plotting, namelist tests |
| `problems/` profile path | 24 files, largest 6.9k/6.4k/4.2k/2.9k lines | `profile.py`, `profile_policy.py`, `profile_preconditioners.py`, `profile_diagnostics.py` or existing equivalent names if fewer files | extract sparse-direct/native/symbolic rescue branches; delete duplicated branch setup, direct-tail naming, probe/candidate code | RHSMode-1 parity/residual/policy/output tests |
| `problems/` transport path | 7 transport files, 16k+ lines | `transport.py`, `transport_policy.py`, `transport_diagnostics.py` | extract multi-worker campaign runtime and duplicate policy/preconditioner probes | RHSMode-2/3 geometryScheme 2/11 gates |
| `operators/` | 15 files, largest 6.1k lines | equation-oriented blocks: layout, collisions, drifts, electric field, full system | merge duplicate reduced-tail/fblock/device-sparse helpers if they do not name equations | operator identity, conservation, residual/JVP tests |
| `solvers/` | 31 files, largest 5.2k lines | `krylov.py`, `preconditioners.py`, `sparse.py`, `memory.py`, `diagnostics.py` or equivalent compact modules | extract native/symbolic/ND/BLR/HSS/multifrontal research; delete unpromoted smoother/probe variants | solver residual, memory model, policy tests |
| `outputs/` | 5 files, 7.6k lines | one schema builder plus one writer/format module | delete experiment-only keys and duplicate HDF5/NPZ/NetCDF plumbing | output schema, HDF5/NPZ/NetCDF, plot tests |
| `validation/` | 5 files, 2.4k artifact reader | compact fixtures plus claim validators | move publication campaign readers to release assets unless used by docs/tests | artifact schema and docs claim tests |
| `examples/` | 122 Python scripts | <=10 curated user workflows plus original Fortran-v3 examples | move performance/publication/long-run optimization campaigns to research branches or release assets | examples tree contract and smoke CLI/Python examples |
| `scripts/` | 7 Python files, 6.1k lines | zero by default, or `scripts/release_*` only | convert remaining suite/profiling/release tooling to tests/examples/package CLIs or delete | script contract test |
| `tests/` | 313 Python files, 125.3k lines | unit, physics, regression, cli_io, integration, fixtures | consolidate one-off extracted-path tests and generated historical tests | coverage >=95%, CI <10 min |

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

## Exact Source Backlog

### Delete or Extract First

- `profile_sparse_solve.py`: disconnected true-operator residual rescue stage
  APIs/tests and direct-tail residual/true-active/coupled-coarse rescue policy
  wiring are removed; next collapse generic branch setup/direct-tail naming
  that still belongs to extracted native sparse-direct research.
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

## Ordered Finish Plan

### Tranche 1 - Finish Research Extraction From Core

Scope: QI/device-QI residue, true-operator rescue residue, native sparse-direct
research residue, long profiler/campaign scripts, and publication generators.

Actions:

- Keep deleted true-operator rescue stage APIs/tests out of stable core.
- Keep deleted direct-tail residual/true-active/coupled-coarse rescue policy
  wiring and direct-tail support-mode preflight out of stable core.
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
