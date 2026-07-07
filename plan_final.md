# SFINCS_JAX Core-Slim Final Plan

Last updated: 2026-07-07

Active branch / PR: `refactor/v3-driver-architecture` / PR #8

This is the single active plan for the refactor branch. `plan.md` is the historical execution log. Do not create another competing plan. If any README,
docs page, issue, old branch note, or historical checklist conflicts with this
file, follow this file.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core where one
input file plus one geometry runs accurate CPU/GPU neoclassical calculations
with automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime and memory, evident physics/numerics in the source, and
validated differentiable Python workflows for sensitivities, ambipolar roots,
bootstrap current, transport coefficients, plotting, and optimization.

## Current Review State

- Current branch head includes the release/data/hygiene script consolidation,
  generated-output removal, and direct-tail experiment removal.
- Current tracked Python volume is still too large for review:
  115 package Python files / 137.6k source lines, 313 test files / 124.9k test
  lines, 122 example Python files, 5 tracked Python scripts, and one shell
  wrapper.
- Largest source owners to audit first:
  `profile_sparse_xblock.py`, `profile_policies.py`,
  `profile_full_system.py`, `explicit_sparse.py`, `profile_solve.py`,
  `transport_linear_system.py`, `transport_parallel_runtime.py`,
  `profile_dense.py`, `solver.py`, `profile_sparse_solve.py`,
  `outputs/rhsmode1.py`, and `preconditioner_transport_matrix.py`.
- Largest test owners to consolidate first:
  `test_profile_response_sparse_pc.py`, `test_rhs1_full_assembly.py`,
  `test_io_output_policy_coverage.py`, `test_v3_sparse_pattern.py`,
  `test_explicit_sparse.py`, `test_rhs1_solver_replay.py`, dense profile
  tests, transport policy tests, and extracted-path tests.
- Experimental QI/device-QI, native sparse-direct, nested-dissection,
  multifrontal, HSS/BLR, long profiling campaigns, publication generators, and
  solver tuning variants are preservation-branch material unless they pass the
  stable admission gates below.
- Overall PR readiness under this stricter small-core goal is about 88-90%.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 50% | Every retained file, function, public knob, and nontrivial branch has an owner, caller, proof test, and docs/perf reason. |
| Core-main slimming | Active | 63% | Stable branch keeps only parity-clean, runtime-acceptable, documented defaults; research code is deleted or moved to research PRs. |
| Source simplification | Active | 61% | Package falls first to <=68 Python files / <=80k lines, then <=50 files / <=50k lines unless ledger exceptions are justified. |
| Tests/examples/scripts cleanup | Active | 82% | Examples are curated, scripts are gone or package CLIs, tests are smaller but reach >=95% meaningful coverage. |
| Parity/performance evidence | Active | 70% | Supported cases have checked Fortran-v3 parity/runtime/RSS/bootstrap evidence at documented grids. |
| Docs/readme | Active | 82% | Public docs describe stable standalone software, not branch history, unpromoted campaigns, or old benchmark caveats. |

## Source Structure Rules

The stable package keeps one-level domain packages only:

| Area | Purpose | Target files |
| --- | --- | ---: |
| package root | public API, CLI, I/O, namelist, plotting, solver facade, paths, README | <=9 |
| `physics/` | normalizations, collisions, Redl/bootstrap, classical transport formulas | <=4 |
| `discretization/` | grids, indexing, stencils, velocity/radial maps | <=5 |
| `geometry/` | Boozer/VMEC adapters and geometry loading | <=5 |
| `operators/` | DKE term assembly with equation-oriented names | <=9 |
| `problems/` | profile/RHSMode-1, transport/RHSMode-2/3, ambipolar problem orchestration | <=9 |
| `solvers/` | Krylov, sparse utilities, admitted preconditioners, memory diagnostics | <=8 |
| `outputs/` | output schema plus HDF5/NetCDF/NPZ writers/readers | <=3 |
| `validation/` | compact fixture readers, release gates, data fetch, figure evidence | <=5 |
| `workflows/` | curated differentiable workflows, scans, optimization helpers | <=3 |

Stable module names must describe a physics or numerical boundary. New stable
names based on `v3_`, broad `rhs1_`, `probe`, `campaign`, `rescue`,
`candidate`, `legacy`, `hard_seed`, `native`, `symbolic`, or `qi_*` are not
allowed without a ledger exception and a stable proof. Existing names with
those terms are debt to delete, rename, or extract.

The package root may not become a dumping ground. It should contain only
stable facades and `README.md`; implementation belongs in the domain packages.
No nested package directories are allowed under `sfincs_jax/`.

## Concrete Code-Audit Rules

The default action for every line is deletion. A line remains only if it has
one owner and proof:

| Owner | Retain only if the line... | Proof |
| --- | --- | --- |
| `PHYSICS` | states/evaluates a DKE, collision, drift, geometry, flux, current, Redl, ambipolar, or normalization equation | equation, parity, or literature-anchored test |
| `NUMERICS` | builds a grid, stencil, operator, residual, preconditioner, Krylov solve, interpolation, quadrature, convergence diagnostic | identity/residual test and runtime/RSS gate |
| `API` | is stable Python API, CLI, namelist compatibility, plotting, or output schema | public test and docs reference |
| `AUTODIFF` | preserves differentiable solve, JVP/VJP, implicit derivative, or branch-safe wrapper | gradient test against finite difference or analytic identity |
| `PERF` | measurably reduces runtime, memory, JIT overhead, or output cost | benchmark fixture or policy gate |
| `EVIDENCE` | reads compact validation fixture, release evidence, or docs claim data | schema/docs-claim test |

Everything else is `RESEARCH`, `COMPAT`, `DUPLICATE`, `GENERATED`, or
`OBSOLETE`. `RESEARCH` moves to a preservation branch/PR before deletion from
this PR. `COMPAT` stays only if it is a documented SFINCS Fortran v3
input/output compatibility surface. `DUPLICATE`, `GENERATED`, and `OBSOLETE`
are deleted with imports, tests, docs, output keys, env vars, fixtures, and
examples.

Per file, perform this exact loop:

1. Record path, line count, public symbols, imports, callers, env vars, output
   keys, tests, docs, and examples in `tests/fixtures/core_slim_inventory.json`.
2. Decide `keep`, `merge`, `delete`, or `extract-pr`.
3. For every public symbol, record owner tag, stable caller, proof test, docs
   owner, and whether it is autodiff-safe.
4. Remove one-call wrappers unless they name a real physics/numerics/API
   boundary that makes the code easier to read.
5. Collapse duplicate diagnostics dictionaries, policy branches, shape helpers,
   namelist aliases, solver option parsing, and output-key builders.
6. Delete tests for extracted paths; keep compact absence tests so stable core
   cannot silently re-import research paths.
7. Run focused tests plus Ruff, compileall, diff hygiene, size guard, and
   package import checks.
8. Commit only when retained files, lines, knobs, duplicated concepts, generated
   artifacts, or test burden decrease. A pure move is not progress.

## Repository-Wide Line Sweep

Every tracked source, test, example, script, docs, and fixture file must be
classified in `core_slim_inventory.json` before the PR leaves draft. The line
sweep is done in this order so the largest complexity owners are removed before
fine polishing:

1. Stable source files over 1500 lines, largest first.
2. Solver/preconditioner files containing research words:
   `qi`, `native`, `symbolic`, `nested`, `multifrontal`, `hss`, `blr`,
   `candidate`, `probe`, `rescue`, `campaign`, or `hard_seed`.
3. Problem orchestration files in `problems/`, because they decide runtime,
   memory, solver policy, branch selection, output keys, and differentiability.
4. Tests over 1200 lines, especially tests coupled to extracted research paths.
5. Examples and scripts, keeping only curated user workflows and release tools.
6. README/docs text, removing branch-history language after source decisions are
   final so public claims only describe retained stable software.

For every file, record these fields in the inventory: `decision`, `owner_tags`,
`stable_callers`, `public_symbols`, `test_proofs`, `docs_owner`,
`autodiff_scope`, `runtime_memory_scope`, `delete_candidates`,
`extract_candidates`, and `line_target`. A file cannot be marked `core` unless
it has at least one stable caller, one proof test, and a docs/API reason.

Line-level deletion rules are mandatory:

- Delete wrappers that only rename another function unless the wrapper is the
  public API or names a real equation/numerical boundary.
- Delete env-var-only solver branches unless the same route is admitted by the
  automatic policy and has strict residual/runtime/RSS proof.
- Delete duplicate diagnostics/output dictionaries; one result schema per
  problem family owns all public keys.
- Delete historical comments, old benchmark caveats, dead compatibility aliases,
  and branch-history prose from public docs.
- Delete generated artifacts, run outputs, local profiles, and uncompressed
  figures unless they are compact checked evidence with a claim test.
- Extract, rather than keep, any code whose only justification is future
  research, partial accuracy, partial parity, GPU-only experiments, or long
  campaign generation.

The refactor must reduce complexity, not redistribute it. A tranche is accepted
only if it lowers at least one of these without failing tests: source lines,
test lines, public knobs, solver route count, env-var branches, duplicated
schema builders, examples, scripts, or docs pages carrying unstable claims.

## Stable-Core Triage Matrix

| Area | First action | Keep only | Move/delete |
| --- | --- | --- | --- |
| root package | keep facades thin | API/CLI/I/O/plot/solver entry points | implementation helpers and duplicate facades |
| `problems/` | collapse orchestration | one profile path, one transport path, one ambipolar path | probe/candidate/rescue/research policy branches |
| `solvers/` | admit by residual gate | Krylov dispatch, memory model, default sparse/x-block routes | symbolic/native/ND/HSS/QI paths without stable proof |
| `operators/` | expose equations | DKE term blocks with shape contracts | duplicate reduced/device helpers not tied to equations |
| `outputs/` | unify schema | one schema and writers for HDF5/NetCDF/NPZ | experiment-only fields and duplicated writers |
| `validation/` | keep evidence readers | compact fixtures, release gates, figure-claim checks | long campaign runners and raw trace analyzers |
| examples | curate workflows | <=10 user-facing scripts/notebooks plus Fortran-v3 references | performance/publication/optimization campaigns |
| tests | parametrize by domain | unit, physics, numerical, regression, CLI/I/O, compact parity | one-off scaffolds for extracted paths |
| scripts | eliminate by default | documented release-only exceptions | anything better represented as CLI, test, or example |

## Extraction PR Plan

Before removing a research family from this branch, ensure it exists on a
preservation branch and open/refresh a draft research PR if the code is still
worth preserving.

| Research branch/PR | Move out of stable branch | Stable replacement |
| --- | --- | --- |
| `research/qi-device-hard-seed` | QI/device-QI hard seed, special QI preconditioners, QI promotion campaigns | documented research-lane pointer only |
| `research/native-sparse-direct` | native direct factors, symbolic active factors, nested-dissection, multifrontal, HSS/BLR, true-operator rescue variants | admitted sparse/x-block/default policy only |
| `research/parallel-performance` | sharded/multi-worker/multi-GPU campaigns and long trace generators | serial/default CPU/GPU solve APIs plus documented future lane |
| `research/publication-audits` | long benchmark/publication figure generators, Zenodo sweeps, raw profiling scripts | compact fixture readers and checked docs figures |
| `research/optimization-experiments` | unpromoted stellarator optimization objectives and long campaigns | <=1 curated optimization workflow |

Extraction is complete only when `rg` finds no stable import, default env var,
public knob, README claim, example path, or required test for the extracted
family. README/docs may mention extracted work only in `docs/research_lanes.rst`
as unsupported research history.

## File-Family Disposition

| Family | Keep in stable core | Delete/extract | Gate |
| --- | --- | --- | --- |
| Root modules | `api.py`, `cli.py`, `io.py`, `namelist.py`, `plotting.py`, `solver.py`, `paths.py`, `README.md` | merge `compare.py`, `diagnostics.py`, `profiling.py`, broad helper facades if not public | root <=9 files; API/CLI/plotting tests pass |
| Profile/RHSMode-1 | one profile problem, one policy table, one setup/solve path, one diagnostics module | probe/candidate/rescue/native/direct-tail/QI branches and duplicate backend wrappers | supported RHSMode-1 fixtures pass with no env vars |
| Transport/RHSMode-2/3 | one linear-system builder, one policy table, one solve/finalize path | multi-worker campaigns, duplicate preconditioner probes, unpromoted active-factor research | geometryScheme 2/11 gates pass |
| Operators | named equation blocks for collisions, drifts, electric field, ExB, layout, full system | duplicate reduced-tail/fblock/device-sparse helpers not tied to equations | operator identity/conservation/JVP tests pass |
| Solvers | Krylov dispatch, memory model, admitted preconditioners, compact sparse utilities | symbolic/native/ND/HSS/multifrontal/smoother-tuning variants not admitted | residual/runtime/RSS gates pass |
| Outputs | one schema builder and one writer/reader path per format | experiment-only output keys and duplicate HDF5/NPZ/NetCDF plumbing | output schema and plot tests pass |
| Validation | compact release gates, data fetch, artifact readers, figure claim checks | publication campaign runners and raw trace readers | docs-claim tests pass |
| Examples | original Fortran-v3 references plus <=10 curated workflows | performance/publication/long optimization campaign scripts | examples tree contract and smoke examples pass |
| Tests | unit, physics, regression, cli_io, integration, fixtures | one-off extracted-path tests and historical scaffolds | >=95% coverage, CI <10 min |
| Scripts | none by default | all commands become package CLIs, examples, tests, or release-only docs | `scripts/` empty or documented release exceptions |

## Stable-Core Admission Gates

A solver, preconditioner, example, workflow, output field, or docs claim stays
on this branch only if it satisfies all relevant gates:

- Accuracy: strict residual or frozen Fortran-v3 parity for overlapping models.
- Runtime: useful on supported examples without hidden env knobs.
- Memory: bounded local CPU RSS and explicit GPU memory claim/exclusion.
- Autodiff: differentiable Python path is tested, or non-differentiable CLI path
  is isolated and documented.
- Usability: a user can provide an input file and geometry without knowing
  internal solver families.
- Tests: compact unit/physics/regression coverage under CI budget.
- Docs: scope, equations, grids, outputs, limitations, and examples are stated
  without historical "now/previous/new version" language.

Any path failing one gate is `RESEARCH`, not stable core.

## Ordered Finish Plan

### Phase 1 - Plan And Ledger Tightening

- Update `core_slim_inventory.json` to current counts and add one row for every
  file family listed above.
- Add a `decision` and `target` for every large file over 1500 lines.
- Add one `owner` entry for every public function family; test-only or
  docs-only symbols are delete/extract candidates.

Exit gate: inventory validates, source-tree tests pass, and no plan line
mentions a second active plan.

### Phase 2 - Extract Research From Stable Core

- Remove stable imports, env vars, docs claims, examples, and tests for QI,
  device-QI, native sparse-direct, ND/multifrontal/HSS, long profiler campaigns,
  and publication generators.
- Preserve useful work on the research branches above before deletion.
- Keep only compact research-lane notes in docs.

Exit gate:

```bash
rg "qi_device|device-QI|true_operator_rescue|native_sparse_direct|nested_dissection|multifrontal|HSS|campaign" sfincs_jax tests examples scripts README.md docs
```

returns only research-lane text or stable sparse terminology with ledger proof.

### Phase 3 - Collapse Profile/RHSMode-1 Core

- Split retained logic conceptually into setup, policy, operator/preconditioner,
  solve, residual validation, diagnostics, and output.
- Replace duplicated candidate/probe/rescue policy branches with one typed
  default policy table and one advanced-options parser.
- Merge duplicate diagnostics dictionaries into a single typed result builder.
- Delete backend wrappers that have no production caller; keep only wrappers
  that isolate JAX backend behavior or public compatibility.

Exit gate: profile family <=14k lines combined; key files below these targets:
`profile_policies.py <=2500`, `profile_sparse_xblock.py <=2800`,
`profile_sparse_solve.py <=2000`, `profile_solve.py <=1800`,
`profile_dense.py <=1600`; RHSMode-1 fixtures pass without env vars.

### Phase 4 - Collapse Transport/RHSMode-2/3 Core

- Keep one transport linear-system builder, one transport policy table, and one
  solve/finalize path.
- Reuse shared diagnostics/output builders.
- Extract parallel campaign runtime unless it is a public supported helper.

Exit gate: transport family <=4k lines combined or ledger exceptions;
geometryScheme 2/11 compact production-floor gates pass.

### Phase 5 - Make Equations And Numerics Evident

- Move repeated normalizations into `physics/`.
- Keep DKE terms in named operator blocks with shape contracts and comments
  that state the equation role.
- Remove duplicated formula copies and implicit shape magic.
- Preserve differentiability with tests for residual JVPs, implicit
  derivatives, ambipolar derivative signs, and finite-difference agreement.

Exit gate: `profile_full_system.py <=3000` or exception; operator, collision,
geometry interpolation, Redl/bootstrap, drift-switch, and electric-field tests
pass.

### Phase 6 - Shrink Outputs, Examples, Scripts

- Collapse output schema/writer code and delete experiment-only fields.
- Keep original Fortran-v3 example references plus at most ten curated
  workflows: quick CLI run/plot, Python solve, transport coefficients,
  bootstrap current vs Redl, ambipolar Er, VMEC geometry loading, autodiff
  sensitivity, QA optimization objective, output plotting, and advanced solver
  options.
- Convert remaining scripts into package CLIs/tests/examples or delete them.

Exit gate: examples Python files <=30 before review and <=10 curated workflow
scripts final, `scripts/` empty or documented release-only, no tracked file
above 2 MB.

### Phase 7 - Consolidate Tests And Reach 95% Coverage

- Delete tests for extracted research paths.
- Merge one-off solver tests into parametrized suites by domain:
  unit, physics, numerical identities, regression, cli_io, optional integration,
  fixtures.
- Raise coverage by deleting dead code first, then adding meaningful tests for
  stable physics/numerics/API gaps.
- Keep large fixtures in release assets; the repo keeps only compact fixtures.

Exit gate: coverage >=95%, default CI <=10 minutes, tests <=180 files before
review and <=120 final unless exceptions are justified.

### Phase 8 - Regenerate Evidence And Docs

- Rerun supported CPU parity/runtime/RSS/bootstrap evidence locally.
- Refresh GPU evidence only when a GPU is available; otherwise docs must state
  the last checked GPU scope without blocking CPU review.
- Regenerate README/docs tables and figures from checked fixtures.
- Rewrite README/docs text to be standalone and remove "current branch",
  "now", "previous", "new version", and unpromoted campaign language.

Exit gate: docs build with warnings as errors; README and docs claims match
fixtures; package README maps the slim structure.

### Phase 9 - Review Handoff

- Run focused suites, then full tests, coverage, examples smoke, docs, Ruff,
  compileall, size guard, and diff hygiene.
- Commit final before/after inventory.
- Push PR #8 and leave it unmerged for review.

Exit gate: clean branch, green checks, no generated clutter, every file/line
budget exception justified in the ledger.

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

Before review, also run full tests, coverage with `--fail-under=95`, docs with
`sphinx -W`, the quick CLI solve/plot example, and release gates.

## Completion Gates

- Package source is <=50 Python files and <=50k lines, or every exception is
  justified by the ledger and review notes.
- Experimental solver/preconditioner/QI/profiling/publication lanes are separate
  PRs or deleted from stable.
- `examples/` keeps original Fortran-v3 references plus <=10 curated workflows.
- `benchmarks/` is absent; `scripts/` is empty or documented release tooling.
- Tests are smaller, meaningful, >=95% coverage, and default CI stays under
  10 minutes.
- Supported examples have fresh Fortran-v3 parity/runtime/memory/bootstrap
  evidence at documented grids.
- README/docs match the slim core and do not market research paths as stable.
- PR #8 is clean, pushed, and ready for review.

## Explicit Deferred Items

Deferred unless production-gated: experimental QI/device-QI, native
sparse-direct research, nested-dissection/multifrontal/HSS replacements,
lower-memory preconditioner research, GPU/multi-GPU campaigns, publication
audits, and long stellarator optimization campaigns. They may be referenced in
`docs/research_lanes.rst` only; they must not remain as stable source,
examples, tests, README claims, or default solver branches.
