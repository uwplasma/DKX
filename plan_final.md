# SFINCS_JAX Core-Slim Final Plan

Last updated: 2026-07-07. Active branch / PR: `refactor/v3-driver-architecture` / PR #8

This is the single active plan for the refactor branch. `plan.md` is the historical execution log. Do not create another competing plan. If any README,
docs page, old branch note, issue, benchmark artifact, or checklist conflicts
with this file, follow this file.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core where one
input file plus one geometry runs accurate CPU/GPU neoclassical calculations
with automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime and memory, evident physics/numerics in the source, and
validated differentiable Python workflows for sensitivities, ambipolar roots,
bootstrap current, transport coefficients, plotting, and optimization.

## Current Review State

- Branch head is PR #8 on `refactor/v3-driver-architecture`. Completed work
  includes script promotion, generated-output removal, direct-tail experiment
  removal, sharding/high-nu audit extraction, root Krylov cleanup, BLR/HSS and
  nested-dissection frontal route removal, and first examples cleanup.
- Current burden after the latest source cleanup is 115 package Python files /
  139,678 package lines, 300 test files / 120,448 test lines, 99 example Python
  files / 16,027 example lines, and 450 tracked example files. These numbers must
  decrease by deletion, merging, or research extraction; moving lines into more
  files is a failed tranche.
- `core_slim_inventory.json` is file-complete only at broad path-rule
  granularity; before edits, add section/symbol cards for every source/test
  file over 500 lines and every example folder.
- Root files still hiding implementation are `ambipolar.py`, `diagnostics.py`,
  `grids.py`, `input_compat.py`, `profiling.py`, and `sensitivity.py`. Each
  must become a tiny documented facade, move into a domain owner, or disappear.
- Latest pushed cleanup tranches removed RHSMode-1 active-symbolic wrapper
  routes, renamed the remaining sparse-preconditioner owner modules to domain
  names, and folded the Fortran-reduced factor policy into
  `solvers/preconditioner_reduced_pmat.py`. The next cleanup order is examples
  reference pruning, large-test consolidation, then RHSMode-1 orchestration
  collapse.
- Largest package owners, in order: `profile_sparse_xblock.py`,
  `profile_policies.py`, `profile_full_system.py`, `validation/suite.py`,
  `profile_solve.py`, `explicit_sparse.py`, `transport_parallel_runtime.py`,
  `transport_linear_system.py`, `profile_dense.py`, `validation/artifacts.py`,
  `krylov.py`, `profile_sparse_solve.py`, `outputs/rhsmode1.py`, and
  `preconditioner_transport_matrix.py`.
- Largest test/example clutter owners are `test_profile_response_sparse_pc.py`,
  `test_rhs1_full_assembly.py`, `test_io_output_policy_coverage.py`,
  `test_v3_sparse_pattern.py`, `test_rhs1_solver_replay.py`,
  `test_profile_response_dense.py`, `test_explicit_sparse.py`,
  `examples/publication_figures`, `examples/performance`,
  `examples/optimization`, and `benchmarks/`.
- Overall PR readiness is about 88-90%; the blocker is reviewability and stable
  surface discipline, not more experimental solver code.

## Open Lanes

| Lane | Status | Done when |
| --- | --- | --- |
| Whole-repo line audit | Active | `core_slim_inventory.json` classifies every tracked source, test, example, script, benchmark, docs, and fixture file with owner, caller, proof, docs, line target, keep/merge/delete/extract decision, and first public-symbol/env-var/output-key disposition. |
| Core-main slimming | Active | Stable branch keeps only parity-clean, runtime-acceptable, documented defaults; all research-only QI/device-QI/native sparse/long-campaign code is deleted or preserved on research PRs. |
| Source simplification | Active | Package falls first to <=68 Python files / <=80k lines, then <=50 files / <=50k lines unless ledger exceptions are justified by stable callers and proof tests. |
| Examples/tests/scripts cleanup | Active | Examples are original SFINCS-v3 references plus <=10 curated workflows; tests are smaller, meaningful, >=95% coverage, and default CI is under 10 minutes. |
| Parity/performance evidence | Active | Supported cases have checked Fortran-v3 parity/runtime/RSS/bootstrap evidence at documented grids; unpromoted cases are not marketed as stable. |
| Docs/readme | Active | Public docs describe standalone stable software, not branch history, old benchmark caveats, partial research campaigns, or unsupported solver routes. |

## Source Structure Rules

The stable package keeps one-level domain packages only.

| Area | Stable purpose | Target |
| --- | --- | ---: |
| package root | public API, CLI, compare command, I/O, namelist compatibility, plotting, paths, README, solver compatibility facade | <=9 Python files |
| `physics/` | normalizations, collisions, Redl/bootstrap, classical formulas | <=4 files |
| `discretization/` | v3 grids, indexing, stencils, velocity/radial maps | <=5 files |
| `geometry/` | Boozer/VMEC adapters and geometry loading | <=5 files |
| `operators/` | equation-oriented DKE term assembly | <=9 files |
| `problems/` | profile/RHSMode-1, transport/RHSMode-2/3, ambipolar orchestration | <=9 files |
| `solvers/` | Krylov, sparse utilities, admitted preconditioners, memory diagnostics | <=8 files |
| `outputs/` | output schemas plus HDF5/NetCDF/NPZ writers/readers | <=3 files |
| `validation/` | compact fixture readers, release gates, data fetch, figure evidence | <=5 files |
| `workflows/` | curated differentiable scans and optimization helpers | <=3 files |

Stable file names must describe physics or numerics. Names based on `v3_`,
broad `rhs1_`, `probe`, `campaign`, `rescue`, `candidate`, `legacy`,
`hard_seed`, `native`, `symbolic`, `multifrontal`, `hss`, `blr`, or `qi_*` are
debt unless the inventory records a stable proof and a reviewer-facing reason.
No nested package directories are allowed under `sfincs_jax/`.

## Concrete Code-Audit Rules

The audit is reductive and file-card driven. Before editing any tracked file,
record path, lines, imports, callers, public symbols, env vars, CLI flags,
namelist aliases, output/diagnostic keys, examples, tests, docs, owner tags,
line target, decision, and extraction branch. A file with no card is not edited.
Allowed owner tags and proof are:

| Owner | Retain only if it supports | Required proof |
| --- | --- | --- |
| `PHYSICS` | DKE terms, collisions, drifts, geometry, fluxes, current, Redl, ambipolarity, normalizations | equation/literature or Fortran-v3 parity test |
| `NUMERICS` | grids, stencils, operators, residuals, preconditioners, Krylov, interpolation, quadrature | identity/residual test plus runtime/RSS gate when claimed |
| `API` | public Python API, CLI, namelist compatibility, plotting, output schema | public contract test and docs reference |
| `AUTODIFF` | JVP/VJP, implicit derivatives, branch-safe differentiable solves | analytic or finite-difference gradient test |
| `PERF` | runtime, memory, JIT, setup, output-write reduction | benchmark fixture or policy gate |
| `EVIDENCE` | compact validation fixture, release evidence, docs claim data | schema/docs-claim test |
| `COMPAT` | documented SFINCS Fortran v3 input/output compatibility | compatibility fixture or parser/output test |
Every retained line gets one disposition during the section review. The default
disposition is `delete-core`; a line is promoted from that default only when
the file card names its owner tag, caller, proof, public/domain purpose, and
rejected simpler owner. Review line ranges as imports/constants, public API,
helpers over 20 lines, policy/solver branches, output/diagnostic keys,
comments/docstrings, tests, or examples. Lines kept only by habit, coverage,
history, manual env-var tuning, or unpromoted campaigns are deleted/extracted.

| Disposition | Keep/move rule | Required result |
| --- | --- | --- |
| `keep-core` | shortest tested expression of a stable owner tag | caller, proof test, docs/API owner, and no simpler local implementation |
| `merge-core` | correct code in the wrong owner or duplicated elsewhere | moved to canonical domain file; old import, alias, and test history deleted |
| `delete-core` | dead, duplicate, generated, obsolete, coverage-only, historical, or branch-prose code | removed with imports, tests, docs, env vars, fixtures, and examples |
| `extract-research` | useful but not stable: QI/device-QI, native direct factors, long campaigns, special GPU work, publication experiments, unsupported optimization studies | preserved on a research branch/PR, then removed from stable imports and README claims |

The required file card is: path, current/target lines, imports, callers, public
symbols, helpers over 20 lines, env vars, CLI flags, namelist aliases,
output/diagnostic keys, docs/examples, tests, section ranges/dispositions,
rejected simpler owner, extraction branch, and reviewer note. Procedure:

1. Build the file card from `git ls-files`, AST public symbols, `rg` callers,
   tests, docs, examples, env vars, and output keys.
2. Divide files over 500 lines into named sections before editing; classify
   each section as stable, duplicate, research, generated, or obsolete.
3. Classify every public symbol, env var, CLI flag, namelist alias, output key,
   diagnostics key, and private helper over 20 lines.
4. Delete one-call wrappers unless they are the public API or the clearest name
   for a physics/numerics boundary.
5. Collapse duplicate diagnostics dictionaries, policy predicates, shape
   helpers, parser aliases, metadata builders, profiler wrappers, output-key
   builders, and one-call compatibility wrappers into one owner per problem
   family.
6. Delete tests for extracted code; keep only compact absence tests and stable
   behavior tests.
7. Run focused tests, Ruff, compileall, JSON validation, diff hygiene, import
   checks, and size guards.
8. Commit only when stable files, lines, public knobs, solver routes, schemas,
   examples, scripts, generated artifacts, or test burden decrease. A commit
   that keeps the same complexity must name the larger deletion it unlocks.

A commit that only moves code from one large file into many attempt-named files
without decreasing stable lines, file count, or duplicated branches fails this
plan.

## Repository-Wide Line Sweep

The line sweep is mandatory and file-complete. The first pass classifies every
tracked path exactly once. The second pass adds section cards for large files
and folder cards for examples/tests. The third pass deletes, merges, or
extracts code. Do not edit a large owner until its card lists line ranges,
target line count, and each range as stable physics/numerics, public API,
compatibility, test evidence, obsolete, duplicate, or research. The sweep order
is:

1. Package source files over 500 lines, largest first.
2. Solver/preconditioner files containing research words: `qi`, `native`,
   `symbolic`, `nested`, `multifrontal`, `hss`, `blr`, `candidate`, `probe`,
   `rescue`, `campaign`, or `hard_seed`.
3. `problems/`, because these files choose policies, branches, outputs,
   differentiability, runtime, and memory.
4. Tests over 1200 lines and tests coupled to extracted research paths.
5. Examples, scripts, benchmark inputs, and fixture directories.
6. README/docs text and figures, after source decisions are stable.

The first concrete audit batches are:
| Batch | Files/folders | Required simplification |
| --- | --- | --- |
| 1 | `profile_sparse_xblock.py`, `profile_policies.py`, `profile_full_system.py`, `explicit_sparse.py`, `transport_linear_system.py`, `validation/suite.py` | keep one stable route/pipeline per physics problem; extract native-symbolic, rescue, probe, campaign, and hidden-env branches |
| 2 | `test_profile_response_sparse_pc.py`, `test_rhs1_full_assembly.py`, `test_io_output_policy_coverage.py`, `test_v3_sparse_pattern.py`, `test_rhs1_solver_replay.py`, `test_solver_gmres.py`, `test_explicit_sparse.py`, `test_transport_policy_coverage.py` | merge into behavior suites: physics, numerical identities, API/output, autodiff, parity fixtures, docs claims |
| 3 | `tests/solver_policy_trace_gate_*`, `tests/scaled_example_suite_*`, `tests/reference_solver_path_artifacts/`, `examples/publication_figures/`, `examples/performance/` | move long reports/generators to research PRs or release assets; keep compact checked evidence only |
| 4 | `examples/getting_started/`, `examples/parity/`, `examples/optimization/`, `examples/transport/`, `examples/vmec_jax_finite_beta/`, `examples/autodiff/` | reduce to <=10 workflows with one README-backed teaching purpose and one runnable command each |

The sweep is complete only when these repo-level budgets are met or each
exception has a ledger entry with proof:

| Area | Current pressure point | Hard target for this PR |
| --- | --- | --- |
| package files | 115 Python files in the working tree | <=68 first, <=50 final or justified exceptions |
| package lines | 139,678 source lines in the working tree | <=80k first, <=50k final or justified exceptions |
| tests | 300 Python files / 120,448 lines | <=120 files / <=70k lines while keeping >=95% coverage |
| examples | 99 Python files / 16,027 lines and 450 tracked files | original v3 examples plus <=10 curated workflows |
| scripts | no Python scripts after promotion | only documented shell/release tooling, otherwise empty |
| validation package | 5 implementation modules plus `__init__.py` after command consolidation | target met; next reduce lines |

Inventory entries must include: `decision`, `owner_tags`, `stable_callers`,
`public_symbols`, `test_proofs`, `docs_owner`, `autodiff_scope`,
`runtime_memory_scope`, `delete_candidates`, `extract_candidates`,
`line_target`, and `review_status`. A file cannot be marked `core` unless it
has at least one stable caller, one proof test, and one docs/API reason.

Line-level deletion rules are mandatory:

- Delete env-var-only solver branches unless the same route is admitted by
  automatic policy and has strict residual/runtime/RSS proof.
- Delete duplicate result schemas and diagnostics builders; one schema per
  problem family owns public keys.
- Delete generated artifacts, local profiles, run outputs, caches, and
  uncompressed figures unless they are compact checked evidence.
- Delete historical comments and public prose about previous branches,
  "current main", "new version", "new benchmarks", partial campaigns, or old
  caveats; stable docs describe the software as it is.
- Extract any code justified only by future research, partial parity, special
  env vars, GPU-only experiments, or long campaign generation.

## File-Level Execution Queues

The next refactor passes use these queues. Do not start from arbitrary helpers.
For every row, first classify sections in the inventory, then remove/extract
whole sections before moving retained lines.

| Queue | Files | Main decision | First stop condition |
| --- | --- | --- | --- |
| Solver research | `solvers/explicit_sparse.py`, `native_block_factor.py`, `preconditioner_symbolic_*`, `preconditioner_xblock_active.py` | keep CSR/Krylov/admitted defaults; BLR/HSS and nested-dissection frontal routes are removed from stable; extract multifrontal, true-operator rescue, QI hard-seed, env-var-only native factors, and manual rescue routes unless `auto` selects them with strict true-residual/runtime/RSS proof | no stable import/default policy names `multifrontal`, `true_operator_rescue`, QI hard-seed, manual rescue, `blr`, `hss`, or `symbolic_nd` outside this plan/inventory and absence tests |
| RHSMode-1 orchestration | `profile_policies.py`, `profile_sparse_xblock.py`, `profile_solve.py`, `profile_sparse_solve.py`, `profile_dense.py` | one setup -> policy -> solve -> residual -> output pipeline; one advanced policy dataclass; one diagnostics schema | duplicate route predicates, dense/sparse split wrappers, QI metadata, and one-off policy probes removed |
| RHSMode-1 operators | `profile_full_system.py`, `profile_system.py`, `profile_layout.py`, term files | equation-owned DKE terms only; common layout/shape helpers have one owner | device/reduced-tail/sparse-pattern experiments not used by defaults are extracted or deleted |
| RHSMode-2/3 transport | `transport_*`, `preconditioner_transport_matrix.py` | one setup/linear-system/solve/finalize path and one transport policy table | examples pass parity with no hidden tuning env vars; scaling campaigns are not stable source |
| Validation/evidence | `validation/artifacts.py`, `validation/suite.py`, `validation/release.py`, docs figures | compact claim/evidence readers plus data fetch/release gates | long publication/campaign generators and raw QI/device-QI artifacts are extracted |
| Examples/tests | `examples/`, `tests/`, `benchmarks/`, `scripts/` | original v3 examples plus <=10 workflows; tests grouped by behavior, not history | default CI under 10 minutes with >=95% meaningful coverage target |
| Root/API | root modules and `sfincs_jax/README.md` | root stays public API/CLI/I/O/plot/namelist/paths/compare plus tiny facades | root has <=9 implementation-light modules or ledger exceptions |

Exact removals: done BLR/HSS and nested-dissection frontal stable routes;
in-progress QI/device-QI generated artifacts, promotion tests, reference inputs,
and README/docs promotion prose; next QI source coupling, unadmitted
multifrontal/native factors, historical generators, benchmark-only examples,
tests for extracted research, and README/docs deferred-path marketing.

## File Disposition Targets

| Area | Keep in stable | Merge/delete/extract |
| --- | --- | --- |
| package root | `__init__.py`, `__main__.py`, `api.py`, `cli.py`, `compare.py`, `io.py`, `namelist.py`, `paths.py`, `plotting.py`, `README.md`, and tiny compatibility facades only when documented | Move `ambipolar.py`, `diagnostics.py`, `grids.py`, `input_compat.py`, `profiling.py`, `sensitivity.py`, and implementation-heavy facades to domain owners or delete. |
| `problems/profile_*` | one RHSMode-1 setup/solve/finalize pipeline plus diagnostics | candidate/probe/rescue/history variants, env-var-only paths, duplicate policy readers, QI-only hard-seed helpers. |
| `operators/profile_*` | streaming, electric-field, magnetic-drift, ExB, collisions, constraints, source moments, Phi1 coupling, shared layout | device/reduced-tail variants not used by defaults, duplicate geometry/shape helpers, historical sparse-pattern probes. |
| `solvers/` | Krylov dispatch, admitted preconditioners, sparse utilities, memory model | unpromoted native-symbolic/ND/multifrontal/true-operator rescue families and duplicate wrappers; BLR/HSS stable routes are already removed and must stay absent. |
| `outputs/` | profile schema, transport schema, writer/format dispatch | duplicate result dictionaries, HDF5-only ad hoc key builders, internal history fields not in public outputs. |
| `validation/` | compact release artifact readers, fixture fetch, docs-claim figures | large publication generators, raw profiling traces, stale manifest rows. |
| `workflows/` | curated autodiff/scans/optimization helpers used by docs/examples | long campaigns, promotion experiments, one-off optimization evidence scripts. |
| `examples/` | original SFINCS-v3 examples plus <=10 curated workflows: CLI solve/plot, Python solve, output formats, transport, bootstrap/Redl, ambipolarity, autodiff, VMEC/Boozer, optimization, validation comparison | most campaign/performance/publication folders and single-file folders without a teaching purpose. |
| `tests/` | compact unit, physics, numerical, parity, API, docs-claim tests, parameterized by behavior | tests for extracted code, implementation-history pins, duplicate coverage-only private-helper files. |
| `benchmarks/` | nothing as a directory | move small benchmark contract into tests; move large inputs/artifacts to releases. |
| `scripts/` | documented release/data-fetch tooling only | convert to CLIs/tests/examples or delete. |
| `utils/` | nothing unless a user-facing utility is documented | delete obsolete upstream helpers or move to examples/tests. |

## Specific Consolidation Tranches

Each tranche must delete/extract/merge before adding abstractions. A normal
commit should remove at least 500 stable-core lines or 3 stable-core files, or
explain why a smaller change unlocks the next deletion.

| Tranche | Main files | Action | Required proof |
| --- | --- | --- | --- |
| A. Inventory expansion | every tracked file | expand `core_slim_inventory.json` from large-owner sample to full-repo ledger | source-tree and inventory tests |
| B. Research extraction | QI/device-QI/native sparse/multifrontal/performance/publication code | preserve on research PRs, then delete stable imports, knobs, tests, docs, examples | absence tests and focused policy tests |
| C. Root cleanup | root implementation files | root becomes API/CLI/I/O/plot/namelist/paths/compare plus tiny facades | API, CLI, output, plotting tests |
| D. RHSMode-1 collapse | profile policy/solve/sparse/dense files | one policy table and one pipeline: prepare, choose, solve, residual-check, finalize | RHSMode-1 parity, residual, bootstrap, autodiff tests |
| E. Operator cleanup | profile operator files | equation-oriented DKE blocks; delete historical sparse/device/reduced helpers | conservation, symmetry, drift-switch, collision, JVP tests |
| F. Transport collapse | RHSMode-2/3 files | one assembly, one policy, one solve, one finalize path | geometryScheme 2/11 parity/runtime/RSS gates |
| G. Solver cleanup | solver/preconditioner files | keep admitted default methods; extract experimental factors and rescues | strict true-residual gates |
| H. Output/schema collapse | output files | one typed schema per problem family and one suffix-based writer/reader | output schema, plot, readback tests |
| I. Tests/examples/scripts | tests, examples, scripts, benchmarks, utils | curate examples, parametrize tests, delete clutter, move data to releases | coverage >=95%, CI <=10 min |
| J. Docs/evidence | README, docs, figures, tables | regenerate claims from retained evidence and scrub branch-history text | docs build and docs-claim tests |

## Review Granularity For This PR
Use these unit sizes so the refactor finishes in a few large passes:
- File over 1500 lines: classify sections and delete/extract at least one section; reject helper-only renames or attempt-named files.
- Policy/solver route: keep exactly one route name, parser, metadata schema, docs row, and test owner; reject dense/sparse/native/rescue aliases for the same route.
- Output/diagnostics key: one schema owner and one writer/readback test; reject duplicated dictionaries and campaign-only keys.
- Example folder: one README-backed teaching purpose and one runnable command; reject artifact, promotion-scan, or historical-output folders.
- Test file: prove physics, numerical identity, API/CLI/output, autodiff, parity fixture, or docs claim; reject private-history or extracted-research tests.

## Extraction Protocol

Research code is preserved before it is removed from this PR. Use only these
branches unless the plan is edited: QI/device-QI to
`research/qi-device-hard-seed`; native sparse, multifrontal, true-operator
rescue, and manual direct factors to `research/native-sparse-direct`; scaling,
sharding, GPU campaigns, and profiling tools to `research/parallel-performance`;
publication audits, long optimization campaigns, and figure-generation studies
to `research/publication-audits`. For each extracted family, verify the branch
contains the source/tests/examples/docs artifact, delete stable imports, hidden
env vars, settings fields, metadata keys, docs prose, examples, and tests, add
an absence test, and record removed paths in `core_slim_inventory.json`. Do not
leave a compatibility alias unless a documented stable API uses it.

## Immediate Execution Sequence

Follow this sequence without inserting unrelated micro-tranches.

| Gate | Files | Keep only if | Same-commit deletion |
| --- | --- | --- | --- |
| Solver family removal | `solvers/explicit_sparse.py`, `problems/transport_linear_system.py`, `problems/profile_sparse_policy.py`, solver tests | `auto` can select the route without hidden env vars and strict true-residual tests prove it on a representative parity fixture | aliases, settings fields, env vars, metadata keys, tests, docs, README prose |
| QI/device-QI extraction | all `qi_*`, `hard_seed`, `device_qi`, `production_qi` symbols | helper is general physics/numerics used by non-QI workflows | stable imports plus absence/import tests after preservation on `research/qi-device-hard-seed` |
| RHSMode-1 policy collapse | `profile_*` problem/solver files | branch is clearly `geometry`, `operator`, `preconditioner`, `linear solve`, `residual gate`, or `output` | duplicate route predicates and one-off diagnostics schemas |
| Transport collapse | RHSMode-2/3 transport files | documented automatic policy used by examples/parity tests | campaign/profiling knobs and duplicate setup/solve/finalize code |
| Examples/tests cleanup | `examples/`, `tests/`, `benchmarks/`, `scripts/` | proves physics, numerical identity, API/output, parity fixture, autodiff, or docs claim | historical examples and coverage-only private-helper tests |
| Docs/evidence | README, docs, figures, tables | backed by retained compact evidence | branch-history prose and rejected solver-campaign claims |

Each gate has the same completion checklist: the file cards are updated first,
`rg` proves removed route names are absent from source/tests/public docs,
focused tests pass, package imports compile, the inventory line target improves,
and the commit deletes more stable code than it adds unless the added code
replaces multiple deleted branches with one clearer domain owner.

The next implementation passes must be coarse-grained:

1. Finish the validation-script promotion, then immediately shrink
   `validation/` by merging release/report/figure helpers into at most five
   files and deleting docs-only one-off code.
2. Extract or delete unpromoted solver research in `solvers/` and
   `problems/profile_*`: QI-only hard-seed routes, native-symbolic direct
   experiments, multifrontal sketches, and env-var-only rescue branches.
3. Collapse RHSMode-1 into one visible pipeline and one advanced policy table;
   remove all parallel private policy readers that cannot be explained to a
   user as an equation, discretization, solver, or output stage.
4. Collapse RHSMode-2/3 transport into one setup/solve/finalize path; move
   scaling campaigns and production-floor generation out of stable source.
5. Consolidate tests by behavior, not by historical file: physics gates,
   numerical identities, API/CLI/output contracts, autodiff checks, parity
   fixtures, and docs/evidence claims.
6. Curate examples last, after stable APIs stop moving, so examples teach the
   final code instead of preserving temporary workflows.

## Stable vs Research Decision Gates

| Question | Stable answer | Research answer |
| --- | --- | --- |
| Does default `sfincs_jax input.namelist` use it? | yes, without hidden env vars | no, manual opt-in only |
| Does it have Fortran-v3 parity or analytic/literature proof? | compact fixture/test | no or proxy-only |
| Does it reduce or preserve runtime/RSS/JIT for supported grids? | measured or neutral | unmeasured, slower, or memory risky |
| Is promised Python use differentiable? | JVP/VJP/implicit/finite-difference proof | host branch or adaptive path breaks autodiff |
| Is the method explained in docs? | equations, grids, knobs, outputs | idea note, campaign note, or TODO |
| Can CI test it cheaply? | fast unit/physics/regression gate | long campaign/manual GPU run |

Research answers are not failures; they mean the code moves to a research PR
and stable docs keep only a short pointer in `docs/research_lanes.rst`.

## Ordered Finish Plan

1. Finish inventory section cards for every source file over 500 lines, test
   file over 1200 lines, and example folder; cards must name keep/merge/delete/
   extract line ranges before code edits.
2. Extract research families to preservation branches, then remove stable
   imports, hidden env vars, settings fields, metadata keys, examples, tests,
   and docs for QI/device-QI, native sparse-direct, profiling campaigns,
   publication audits, and long optimization campaigns.
3. Make the root package a facade: public API/CLI/I/O/plot/namelist/paths/
   compare plus compatibility aliases only; move/delete implementation-heavy
   `grids`, `input_compat`, `profiling`, `diagnostics`, `ambipolar`, and
   `sensitivity`.
4. Collapse physics/numerics owners: one RHSMode-1 pipeline, one RHSMode-2/3
   pipeline, equation-owned DKE operator blocks, residual-clean solver defaults,
   and one output schema per problem family.
5. Curate examples to original v3 references plus <=10 workflows: CLI solve/
   plot, Python solve, output formats, transport coefficients, bootstrap/Redl,
   ambipolar root, autodiff/JVP, VMEC/Boozer loading, optimization objective,
   and validation fixture comparison.
6. Consolidate tests into behavior suites: physics gates, numerical identities,
   API/CLI/output contracts, autodiff checks, parity fixtures, docs/evidence
   claims, and absence tests. Target >=95% meaningful coverage with default CI
   under 10 minutes.
7. Regenerate README/docs parity/runtime/memory/bootstrap figures and tables
   from retained workflows only; remove branch-history prose and unsupported
   performance or research claims.
8. Run final checks, push PR #8 as the single draft/review PR, and do not merge
   until review is complete.

## No-Microtranche Rule

Do not spend a commit on one private helper unless it immediately unlocks a
larger deletion. Start every work block from one largest owner table, delete or
extract before moving code, keep file count flat or lower, prefer one clear
domain file over many attempt-named files, add absence tests for extracted
paths, and run focused tests plus one import/compile guard before committing.
Full coverage and production benchmark runs happen only at tranche milestones.

## Standard Validation Commands

Use after each tranche:

```bash
PYTHONNOUSERSITE=1 python -m pytest -q tests/test_source_tree_consolidation.py tests/test_domain_package_import_contracts.py
PYTHONNOUSERSITE=1 python -m pytest -q tests/test_examples_tree_contract.py tests/test_benchmark_doc_claims.py
PYTHONNOUSERSITE=1 python -m ruff check <touched files>
PYTHONNOUSERSITE=1 python -m compileall -q sfincs_jax <touched tests>
python -m json.tool tests/fixtures/source_tree_expected.json >/dev/null
python -m json.tool tests/fixtures/core_slim_inventory.json >/dev/null
git diff --check
find . -path ./.git -prune -o -type f -size +2M -print
```

Before review, also run full tests, coverage with `--fail-under=95`, docs with
`sphinx -W`, the quick CLI solve/plot example, release gates, and the supported
Fortran-v3 parity/runtime/RSS/bootstrap comparison suite.

## Completion Gates

- Package source is <=50 Python files and <=50k lines, or every exception is
  justified by the ledger with caller, proof, docs, and line target.
- Experimental solver/preconditioner/QI/profiling/publication lanes are separate
  PRs or deleted from stable.
- Root package has <=9 implementation-light Python files plus README.
- `examples/` keeps original Fortran-v3 references plus <=10 curated workflows.
- `benchmarks/` is absent; `scripts/` is empty or documented release tooling.
- Tests are smaller, meaningful, >=95% coverage, and default CI stays under
  10 minutes.
- Supported examples have fresh Fortran-v3 parity/runtime/memory/bootstrap
  evidence at documented grids.
- README/docs match the slim core and do not market research paths as stable.
- No generated clutter, caches, local profiles, large binary artifacts, or stale
  release outputs remain in the tracked tree.
- PR #8 is clean, pushed, and ready for review.

## Explicit Deferred Items

Deferred unless production-gated: experimental QI/device-QI, native
sparse-direct research, multifrontal replacements, lower-memory preconditioner
research, GPU/multi-GPU campaigns, publication audits, and long stellarator
optimization campaigns. They may be referenced in `docs/research_lanes.rst`
only; they must not remain as stable source, examples, tests, README claims, or
default solver branches.
