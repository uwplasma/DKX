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
  generated-output removal, direct-tail experiment removal, unsupported
  sharding-campaign extraction, and high-nu publication-audit generator
  extraction.
- Current tracked Python volume is still too large for review:
  115 package Python files / 137.6k source lines, 304 test files / 123.1k test
  lines, 109 example Python files / 18.0k lines, 5 tracked Python scripts /
  5.9k lines, and one shell wrapper.
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
- Root cleanup has started by deleting duplicate private `sfincs_jax.compare.main`;
  the supported user entry point remains `sfincs_jax compare-h5`.
- Overall PR readiness under this stricter small-core goal is about 88-90%.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 50% | Every retained file, function, public knob, and nontrivial branch has an owner, caller, proof test, and docs/perf reason. |
| Core-main slimming | Active | 65% | Stable branch keeps only parity-clean, runtime-acceptable, documented defaults; research code is deleted or moved to research PRs. |
| Source simplification | Active | 61% | Package falls first to <=68 Python files / <=80k lines, then <=50 files / <=50k lines unless ledger exceptions are justified. |
| Tests/examples/scripts cleanup | Active | 89% | Examples are curated, scripts are gone or package CLIs, tests are smaller but reach >=95% meaningful coverage. |
| Parity/performance evidence | Active | 70% | Supported cases have checked Fortran-v3 parity/runtime/RSS/bootstrap evidence at documented grids. |
| Docs/readme | Active | 82% | Public docs describe stable standalone software, not branch history, unpromoted campaigns, or old benchmark caveats. |

## Non-Negotiable Refactor Principle

This branch is not allowed to trade one kind of complexity for another. A
change counts as progress only when it removes or simplifies at least one of:
stable files, stable lines, public knobs, solver routes, env-var-only branches,
duplicated data schemas, generated artifacts, long examples, or one-off tests.
Pure moves, broader facades, new compatibility wrappers, and new file families
do not count unless they make a later deletion possible in the same commit or
the immediately following commit.

Every retained line must answer one of these questions:

- What equation, numerical method, user API, output contract, differentiable
  path, performance gate, or validation claim does this line support?
- Which stable caller uses it without hidden environment variables?
- Which test fails if the line is removed or made wrong?
- Which documentation page or public CLI/API behavior depends on it?
- Is it stable-core code, or should it live on a research PR?

If those answers are missing, the line is deleted or extracted. If the answer is
"future optimization", "possible QI lane", "historical benchmark",
"publication audit", "debugging", "manual solver tuning", or "works only with
special env vars", the line belongs on a research branch, not in this PR.

## Source Structure Rules

The stable package keeps one-level domain packages:
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
4. For every private helper over 20 lines, either inline it, move it beside the
   equation/numerical method it supports, or record why the abstraction reduces
   cognitive load.
5. Remove one-call wrappers unless they name a real physics/numerics/API
   boundary that makes the code easier to read.
6. Collapse duplicate diagnostics dictionaries, policy branches, shape helpers,
   namelist aliases, solver option parsing, and output-key builders.
7. Delete tests for extracted paths; keep compact absence tests so stable core
   cannot silently re-import research paths.
8. Run focused tests plus Ruff, compileall, diff hygiene, size guard, and
   package import checks.
9. Commit only when retained files, lines, knobs, duplicated concepts, generated
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

## File-By-File Review Worksheet

Use this worksheet for each retained or touched file. Add the result to
`core_slim_inventory.json`; do not leave it only in a commit message.

| Field | Required content |
| --- | --- |
| `decision` | `keep`, `merge`, `delete`, or `extract-pr` |
| `owner_tags` | one or more of `PHYSICS`, `NUMERICS`, `API`, `AUTODIFF`, `PERF`, `EVIDENCE`, `COMPAT` |
| `stable_callers` | import/call path exercised by default CLI or documented Python API |
| `public_symbols` | exported functions/classes/dataclasses and their proof owner |
| `test_proofs` | compact tests that fail on physics/numerics/API regression |
| `docs_owner` | README/docs page that explains the behavior, or `none` for internal code |
| `autodiff_scope` | `differentiable`, `non_differentiable_cli`, or `not_applicable` |
| `runtime_memory_scope` | claimed grid/device scope and evidence fixture, or `unclaimed` |
| `delete_candidates` | lines/functions/options to delete in the next tranche |
| `extract_candidates` | research-only lines/functions/options and target branch |
| `line_target` | target retained line count after this PR |

A file with `decision=keep` must have at least one stable caller, one proof
test, one owner tag, and a line target. A file with only docs/tests/examples as
callers is not stable source; it is a fixture or should be extracted.

## Specific Consolidation Tranches

Run these tranches in order. Each tranche should be a reviewable commit or a
small group of commits with focused tests.

| Tranche | Main files | Action | Target reduction | Required proof |
| --- | --- | --- | ---: | --- |
| A. Finish research extraction | `examples/publication_figures/*`, `examples/performance/*`, `examples/optimization/*`, related tests | Keep compact checked figures/artifacts and <=10 curated workflows; move long generators/campaigns to research PRs | -25 to -45 example/test files, -8k to -15k lines | examples contract, docs claims, validation manifest tests |
| B. Root package cleanup | `compare.py`, `diagnostics.py`, `profiling.py`, `solver.py`, `api.py`, `cli.py`, `io.py` | Keep root as API/CLI/I/O/plot/solver facade only; move implementation to existing domains or delete | root <=9 Python files, -2k to -4k lines | API, CLI, output, plotting tests |
| C. RHSMode-1 policy collapse | `profile_policies.py`, `profile_sparse_policy.py`, `profile_preconditioner_build.py`, `profile_sparse_solve.py`, `profile_sparse_xblock.py`, `profile_solve.py`, `profile_dense.py` | Replace candidate/probe/rescue branches with one default policy table plus advanced options; delete env-var-only paths | profile family <=14k lines, -14k to -22k lines | RHSMode-1 parity, residual, bootstrap, autodiff tests |
| D. Solver research extraction | `native_block_factor.py`, `preconditioner_symbolic_*`, `preconditioner_xblock_active.py`, `profile_sparse_direct.py` | Keep only automatically admitted, residual-clean defaults; extract ND/multifrontal/HSS/native research | -8 to -15 files or ledger exceptions | strict residual gates and absence tests |
| E. Operator equation cleanup | `profile_full_system.py`, `profile_system.py`, `profile_layout.py`, `profile_reduced_tail.py`, `profile_device_sparse.py`, `profile_fblock.py` | Keep equation-oriented DKE blocks; merge duplicate shape/layout helpers and delete unused reduced/device helpers | operator family <=12k lines | conservation, symmetry, drift-switch, collision, JVP tests |
| F. Transport collapse | `transport_linear_system.py`, `transport_solve.py`, `transport_policies.py`, `transport_diagnostics.py`, `transport_finalize.py`, `transport_parallel_runtime.py` | Keep one RHSMode-2/3 assembly/solve/finalize path; extract campaign/scaling code | transport family <=8k lines | geometryScheme 2/11 parity/runtime/RSS gates |
| G. Output/schema collapse | `outputs/rhsmode1.py`, `outputs/transport.py`, `outputs/writer.py`, `outputs/formats.py` | One typed result schema per problem family and one writer path for HDF5/NetCDF/NPZ | outputs <=4 files / <=3k lines | output schema, plot, restart/readback tests |
| H. Test consolidation | largest tests in `tests/` | Parametrize by domain; delete one-off extracted-path tests; add coverage only for retained stable code | <=180 test files first, <=120 final, coverage >=95% | default CI <=10 min and coverage gate |
| I. Docs/readme/evidence | `README.md`, `docs/`, checked figures/tables | Regenerate claims from retained fixtures; remove branch-history language and unsupported performance rows | no stale "now/previous/current branch" public text | docs build, docs-claim tests |

Do not start a new source family while a previous tranche still has stale
imports, tests, docs claims, or inventory entries. If a tranche exposes a bug,
fix the minimal stable path first, then resume deletion/extraction.

## Commit-Sized Execution Queue

The remaining refactor must proceed as a finite queue, not open-ended audit work.
A commit is acceptable only if it deletes/extracts code, collapses duplicate
routes, or proves a retained file; except for bug fixes, it should remove at
least 500 stable-core lines or 3 stable-core files, or explain why a smaller cut
unlocks the next deletion.

1. **Finish root duplicate cleanup.**
   Remove private duplicate entry points and tests. Keep public `api.py`,
   `cli.py`, `io.py`, `namelist.py`, `paths.py`, `plotting.py`, and
   `compare.py`; move/delete implementation files
   (`ambipolar.py`, `diagnostics.py`, `grids.py`, `input_compat.py`,
   `profiling.py`, `sensitivity.py`, `solver.py`) behind domain facades. Root
   target: <=9 Python files and <=4.5k lines.
2. **Extract research solver families before simplifying defaults.**
   Preserve any still-useful QI/device-QI, native sparse-direct,
   nested-dissection, multifrontal, BLR/HSS, true-operator rescue, and
   multi-GPU campaign code on the matching `research/*` branches, then delete
   stable imports, env vars, docs claims, tests, and examples. Stable default
   solvers may only reference residual-clean automatic policies.
3. **Collapse RHSMode-1 policy and solve orchestration.**
   Replace scattered candidate/probe/rescue functions in
   `profile_solve.py`, `profile_policies.py`, `profile_sparse_policy.py`,
   `profile_preconditioner_build.py`, `profile_sparse_solve.py`,
   `profile_sparse_xblock.py`, and `profile_dense.py` with one small policy
   table and one solve pipeline:
   `prepare_problem -> choose_policy -> solve -> strict_residual_check ->
   finalize_outputs`. Keep advanced user knobs only when they are documented
   and tested. Target: RHSMode-1 problem family <=14k lines.
4. **Collapse operator files around equations, not implementation history.**
   Keep terms named by physics:
   streaming, electric-field, magnetic-drift, ExB, collisions, constraints,
   source moments, and Phi1 coupling. Delete or merge files whose primary
   boundary is historical (`profile_device_sparse`, `profile_reduced_tail`,
   duplicated layout/index helpers). Every retained operator must have a
   local numerical identity or parity test.
5. **Collapse transport RHSMode-2/3.**
   Keep one assembly, one policy, one solve, one finalize path. Move
   parallel-scaling campaigns and long benchmark orchestration out of stable.
   Target: transport family <=8k lines with geometryScheme 2/11 parity and
   runtime/RSS evidence.
6. **Collapse output schemas and writers.**
   Keep one typed result schema for profile solves, one for transport solves,
   and one writer/reader path that dispatches HDF5/NetCDF/NPZ by file suffix.
   Delete duplicated output-key builders and tests that assert implementation
   history rather than public output contracts.
7. **Curate examples and scripts.**
   Keep original SFINCS-v3 examples plus at most 10 teaching workflows:
   CLI solve/plot, Python solve, output formats, transport coefficients,
   bootstrap-current/Redl comparison, ambipolar root, autodiff/JVP,
   VMEC/Boozer geometry loading, optimization objective, and validation
   fixture comparison. Convert useful scripts into CLIs/tests/examples; delete
   unclear scripts and long campaign launchers from stable.
8. **Consolidate tests while increasing meaningful coverage.**
   Replace one-off path coverage with parametrized physics/numerics/API suites.
   Keep fast analytic, fixture, and parity gates; move long production
   campaigns to optional release workflows. Target first <=180 test files, then
   <=120 files, with >=95% meaningful coverage and default CI under 10 min.
9. **Regenerate public evidence and scrub public text.**
   Regenerate parity/runtime/memory/bootstrap figures only from retained
   stable workflows. README and docs must describe standalone software, not
   branch history, "current main", "new version", unpromoted candidates, or
   partial research campaigns.

This queue supersedes older tranche notes. Work in order unless a failing test
blocks the current step; then fix only the minimal stable path.

## File Disposition Targets

These targets make the line-by-line audit concrete. Any exception must be
recorded in `core_slim_inventory.json` with owner, caller, proof, docs, and
line target.

| Current area | Keep in stable | Merge/delete/extract |
| --- | --- | --- |
| package root | `__init__.py`, `__main__.py`, `api.py`, `cli.py`, `compare.py`, `io.py`, `namelist.py`, `paths.py`, `plotting.py`, `README.md` | Move `solver.py`, `sensitivity.py`, `grids.py`, `diagnostics.py`, `profiling.py`, `input_compat.py`, and `ambipolar.py` behind domain facades or delete duplicates. |
| `problems/profile_*` | one RHSMode-1 setup/solve/finalize pipeline plus diagnostics | candidate/probe/rescue/history variants, env-var-only paths, duplicated policy readers, and QI-only hard-seed helpers. |
| `operators/profile_*` | equation-oriented DKE term builders and shared layout/index code | device/reduced-tail variants not used by default policies, duplicated geometry/shape helpers, historical sparse pattern probes. |
| `solvers/` | Krylov dispatch, admitted preconditioners, sparse utilities, memory model | unpromoted native-symbolic/ND/multifrontal/HSS/BLR/true-operator rescue families and duplicate wrappers. |
| `outputs/` | profile schema, transport schema, writer/format dispatch | duplicate result dictionaries, HDF5-only ad hoc key builders, internal history fields not used by public outputs. |
| `validation/` | release artifact readers, compact fixture fetch, claim figure helpers | large publication generators, raw profiling traces, stale manifest rows. |
| `workflows/` | curated autodiff/scans/optimization helpers used by docs/examples | long campaigns, promotion experiments, one-off optimization evidence scripts. |
| `examples/` | original v3 examples plus <=10 curated workflows | most campaign/performance/publication folders and single-file folders without a teaching purpose. |
| `tests/` | compact unit, physics, numerical, parity, API, docs-claim tests | tests for extracted code, tests that only pin implementation history, duplicate coverage-only files. |
| `scripts/` | release/data fetch tooling only if documented | anything better represented as a CLI, test, or curated example. |

## No-Microtranche Rule

Do not spend a commit on a single private helper unless it immediately unlocks
a larger deletion. Each normal work block starts from one largest owner table
(root, RHSMode-1, transport, outputs, examples, tests, or docs), deletes or
extracts before moving code, keeps file count flat or lower, prefers one clear
domain file over many attempt-named files, adds absence tests for extracted
paths instead of new tests for soon-deleted code, and runs focused tests plus
one import/compile guard before committing. Full coverage runs happen only at
queue milestones.

## Stable vs Research Decision Gates

Stable code stays in this PR only if all relevant rows pass:

| Question | Stable answer | Research answer |
| --- | --- | --- |
| Does default `sfincs_jax input.namelist` use it? | yes, without hidden env vars | no, only manual opt-in |
| Does it have Fortran-v3 parity or analytic/literature gate? | yes, compact fixture/test | no or proxy-only |
| Does it reduce runtime/RSS/JIT for supported grids? | measured or neutral | unmeasured, slower, or memory risky |
| Is Python use differentiable where promised? | JVP/VJP/implicit/finite-difference test | branch/host solver breaks autodiff |
| Is the method explained in docs? | equations, grids, knobs, outputs | paper idea, campaign note, or TODO |
| Can it run in CI budget? | unit/physics/regression coverage | long campaign or manual GPU run |

Research answers are not failures; they mean the code moves to the matching
research PR and stable docs keep only a short pointer in `docs/research_lanes.rst`.

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

## Ordered Finish Plan

Execute the specific consolidation tranches A-I above. The final handoff gate
is: inventory current, no stale research imports/claims, source/tests/examples
below their target budgets or justified in the ledger, supported parity and
performance evidence regenerated, docs warning-clean, full tests and coverage
green, no generated clutter, branch pushed, and PR #8 left unmerged for review.

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
