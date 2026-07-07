# SFINCS_JAX Core-Slim Final Plan

Last updated: 2026-07-07

Active branch / PR: `refactor/v3-driver-architecture` / PR #8

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

- Branch head has release/data/hygiene script consolidation, generated-output
  removal, direct-tail experiment removal, unsupported sharding-campaign
  extraction, high-nu publication-audit extraction, Krylov implementation moved
  from the package root to `sfincs_jax.solvers.krylov`, and the retained Python
  release/profiling scripts promoted to `sfincs_jax.validation`.
- Tracked code is still too large for review: 116 package Python files /
  137.6k source lines, 304 test Python files / 123.1k test lines, 109 example
  Python files / 18.0k lines, 5 tracked Python scripts / 5.9k lines, and one
  shell wrapper at the last committed audit. The current working tree is about
  119 package Python files / 143.6k source lines, 303 test Python files /
  123.4k lines, 110 example Python files / 18.3k lines, and no tracked Python
  scripts; this count must drop, not merely move between folders.
- The package root is still too broad: `ambipolar.py`, `diagnostics.py`,
  `grids.py`, `input_compat.py`, `profiling.py`, and `sensitivity.py` are
  implementation owners or mixed facades that must move behind domain modules
  or be deleted.
- Largest source audit owners are, in order:
  `profile_sparse_xblock.py`, `profile_policies.py`,
  `profile_full_system.py`, `explicit_sparse.py`, `profile_solve.py`,
  `transport_linear_system.py`, `transport_parallel_runtime.py`,
  `profile_dense.py`, `krylov.py`, `profile_sparse_solve.py`,
  `outputs/rhsmode1.py`, and `preconditioner_transport_matrix.py`.
- Largest test/example clutter owners are:
  `test_profile_response_sparse_pc.py`, `test_rhs1_full_assembly.py`,
  `test_io_output_policy_coverage.py`, `test_v3_sparse_pattern.py`,
  `test_explicit_sparse.py`, `test_rhs1_solver_replay.py`, generated
  suite-report directories, `examples/publication_figures`,
  `examples/performance`, `examples/optimization`, and `benchmarks/`.
- Overall PR readiness under the strict small-core goal is about 88-90%.

## Open Lanes

| Lane | Status | Done when |
| --- | --- | --- |
| Whole-repo line audit | Active | `core_slim_inventory.json` classifies every tracked source, test, example, script, benchmark, docs, and fixture file with owner, caller, proof, docs, line target, and keep/merge/delete/extract decision. |
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

The default action for every line is deletion. A line remains only if it has one
stable owner and proof:

| Owner | Retain only if it supports | Required proof |
| --- | --- | --- |
| `PHYSICS` | DKE terms, collisions, drifts, geometry, fluxes, current, Redl, ambipolarity, normalizations | equation/literature or Fortran-v3 parity test |
| `NUMERICS` | grids, stencils, operators, residuals, preconditioners, Krylov, interpolation, quadrature | identity/residual test plus runtime/RSS gate when claimed |
| `API` | public Python API, CLI, namelist compatibility, plotting, output schema | public contract test and docs reference |
| `AUTODIFF` | JVP/VJP, implicit derivatives, branch-safe differentiable solves | analytic or finite-difference gradient test |
| `PERF` | runtime, memory, JIT, setup, output-write reduction | benchmark fixture or policy gate |
| `EVIDENCE` | compact validation fixture, release evidence, docs claim data | schema/docs-claim test |
| `COMPAT` | documented SFINCS Fortran v3 input/output compatibility | compatibility fixture or parser/output test |

Everything else is `RESEARCH`, `DUPLICATE`, `GENERATED`, or `OBSOLETE`.
`RESEARCH` moves to a preservation branch/PR before deletion from this PR.
`DUPLICATE`, `GENERATED`, and `OBSOLETE` are deleted with their imports, tests,
docs, env vars, fixtures, and examples.

For each file, perform this loop. This is the required "every line" review
mechanism; no file is exempt because it is old, covered, or difficult:

1. Record path, line count, public symbols, imports, callers, env vars, output
   keys, tests, docs, examples, and current line target in
   `tests/fixtures/core_slim_inventory.json`.
2. Decide exactly one action: `keep`, `merge`, `delete`, or `extract-pr`.
3. For every public symbol, record owner tag, stable caller, proof test, docs
   owner, autodiff scope, and runtime/memory scope.
4. For every top-level constant and env var, keep it only if it is part of the
   public API, namelist compatibility, an output schema, or a tested automatic
   policy. Hidden tuning knobs and one-off campaign flags are research debt.
5. For every private helper over 20 lines, inline it, move it beside the
   equation/numerical method it supports, or record why the abstraction reduces
   cognitive load.
6. Delete one-call wrappers unless they are the public API or name a real
   physics/numerics boundary.
7. Collapse duplicate diagnostics dictionaries, policy branches, shape helpers,
   namelist aliases, solver option parsing, and output-key builders.
8. Delete tests for extracted code; keep compact absence tests that prevent
   stable imports from silently returning.
9. Run focused tests, Ruff, compileall, JSON validation, diff hygiene, package
   import checks, and size guard.
10. Commit only when files, lines, public knobs, solver routes, duplicated
   schemas, examples, scripts, generated artifacts, or test burden decrease.

Line-level dispositions must be explicit in review notes or inventory:

| Disposition | Meaning | Required result |
| --- | --- | --- |
| `keep-core` | Stable physics, numerics, API, output, or validation evidence. | Has caller, proof test, docs/API owner, and no simpler local expression. |
| `merge-core` | Correct code in the wrong place or duplicated under another name. | Moved into the canonical domain owner and old import deleted. |
| `delete-core` | Dead, duplicate, generated, obsolete, coverage-only, or historical code. | Removed with imports, docs, tests, env vars, fixtures, and examples. |
| `extract-research` | Useful but not stable: QI/device-QI, native direct factors, long campaigns, special GPU work, publication experiments, or unsupported optimization studies. | Preserved on a research branch/PR, then deleted from stable imports and README claims. |

Large files are reviewed by section, not by helper churn. For each file over
1500 lines, first split an outline into stable sections, duplicate sections,
research sections, and delete sections. Only then move or delete code. A commit
that moves code from one large file into many small attempt-named files without
reducing total stable lines fails this plan.

## Repository-Wide Line Sweep

The line sweep is mandatory and file-complete. The first pass may classify a
file at module granularity; the second pass must classify every public symbol
and every private helper over 20 lines. The sweep order is:

1. Package source files over 1500 lines, largest first.
2. Solver/preconditioner files containing research words: `qi`, `native`,
   `symbolic`, `nested`, `multifrontal`, `hss`, `blr`, `candidate`, `probe`,
   `rescue`, `campaign`, or `hard_seed`.
3. `problems/`, because these files choose policies, branches, outputs,
   differentiability, runtime, and memory.
4. Tests over 1200 lines and tests coupled to extracted research paths.
5. Examples, scripts, benchmark inputs, and fixture directories.
6. README/docs text and figures, after source decisions are stable.

The sweep is complete only when these repo-level budgets are met or each
exception has a ledger entry with proof:

| Area | Current pressure point | Hard target for this PR |
| --- | --- | --- |
| package files | 119 Python files in the working tree | <=68 first, <=50 final or justified exceptions |
| package lines | 143.6k source lines in the working tree | <=80k first, <=50k final or justified exceptions |
| tests | 303 Python files / 123.4k lines | <=120 files / <=70k lines while keeping >=95% coverage |
| examples | 110 Python files / 18.3k lines | original v3 examples plus <=10 curated workflows |
| scripts | no Python scripts after promotion | only documented shell/release tooling, otherwise empty |
| validation package | 9 Python files after figure and suite-runner merge | <=5 compact evidence/fetch/release modules |

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

## File Disposition Targets

| Area | Keep in stable | Merge/delete/extract |
| --- | --- | --- |
| package root | `__init__.py`, `__main__.py`, `api.py`, `cli.py`, `compare.py`, `io.py`, `namelist.py`, `paths.py`, `plotting.py`, `README.md`, and tiny compatibility facades only when documented | Move `ambipolar.py`, `diagnostics.py`, `grids.py`, `input_compat.py`, `profiling.py`, `sensitivity.py`, and implementation-heavy facades to domain owners or delete. |
| `problems/profile_*` | one RHSMode-1 setup/solve/finalize pipeline plus diagnostics | candidate/probe/rescue/history variants, env-var-only paths, duplicate policy readers, QI-only hard-seed helpers. |
| `operators/profile_*` | streaming, electric-field, magnetic-drift, ExB, collisions, constraints, source moments, Phi1 coupling, shared layout | device/reduced-tail variants not used by defaults, duplicate geometry/shape helpers, historical sparse-pattern probes. |
| `solvers/` | Krylov dispatch, admitted preconditioners, sparse utilities, memory model | unpromoted native-symbolic/ND/multifrontal/HSS/BLR/true-operator rescue families and duplicate wrappers. |
| `outputs/` | profile schema, transport schema, writer/format dispatch | duplicate result dictionaries, HDF5-only ad hoc key builders, internal history fields not in public outputs. |
| `validation/` | compact release artifact readers, fixture fetch, docs-claim figures | large publication generators, raw profiling traces, stale manifest rows. |
| `workflows/` | curated autodiff/scans/optimization helpers used by docs/examples | long campaigns, promotion experiments, one-off optimization evidence scripts. |
| `examples/` | original SFINCS-v3 examples plus <=10 curated workflows | most campaign/performance/publication folders and single-file folders without a teaching purpose. |
| `tests/` | compact unit, physics, numerical, parity, API, docs-claim tests | tests for extracted code, implementation-history pins, duplicate coverage-only files. |
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
| B. Research extraction | QI/device-QI/native sparse/multifrontal/HSS/performance/publication code | preserve on research PRs, then delete stable imports, knobs, tests, docs, examples | absence tests and focused policy tests |
| C. Root cleanup | root implementation files | root becomes API/CLI/I/O/plot/namelist/paths/compare plus tiny facades | API, CLI, output, plotting tests |
| D. RHSMode-1 collapse | profile policy/solve/sparse/dense files | one policy table and one pipeline: prepare, choose, solve, residual-check, finalize | RHSMode-1 parity, residual, bootstrap, autodiff tests |
| E. Operator cleanup | profile operator files | equation-oriented DKE blocks; delete historical sparse/device/reduced helpers | conservation, symmetry, drift-switch, collision, JVP tests |
| F. Transport collapse | RHSMode-2/3 files | one assembly, one policy, one solve, one finalize path | geometryScheme 2/11 parity/runtime/RSS gates |
| G. Solver cleanup | solver/preconditioner files | keep admitted default methods; extract experimental factors and rescues | strict true-residual gates |
| H. Output/schema collapse | output files | one typed schema per problem family and one suffix-based writer/reader | output schema, plot, readback tests |
| I. Tests/examples/scripts | tests, examples, scripts, benchmarks, utils | curate examples, parametrize tests, delete clutter, move data to releases | coverage >=95%, CI <=10 min |
| J. Docs/evidence | README, docs, figures, tables | regenerate claims from retained evidence and scrub branch-history text | docs build and docs-claim tests |

The next implementation passes must be coarse-grained:

1. Finish the validation-script promotion, then immediately shrink
   `validation/` by merging release/report/figure helpers into at most five
   files and deleting docs-only one-off code.
2. Extract or delete unpromoted solver research in `solvers/` and
   `problems/profile_*`: QI-only hard-seed routes, native-symbolic direct
   experiments, nested-dissection/multifrontal/HSS sketches, and env-var-only
   rescue branches.
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

1. Expand `core_slim_inventory.json` to cover every tracked file and every
   public symbol in the package; add tests that fail on missing inventory rows.
2. Create/refresh preservation branches for QI/device-QI, native sparse-direct,
   parallel performance, publication audits, and optimization experiments; then
   remove those paths from stable.
3. Finish root cleanup by moving `grids`, `input_compat`, `profiling`,
   `diagnostics`, `ambipolar`, and `sensitivity` implementation into existing
   domain modules or deleting them; root target is <=9 implementation-light
   Python files.
4. Collapse RHSMode-1 orchestration into one readable policy/solve/finalize
   path and one public advanced-options surface; delete all hidden
   env-var-only branches and duplicate rescue labels.
5. Collapse operator files around DKE terms and numerical identities; delete
   reduced/device/sparse-pattern helpers that no default policy uses.
6. Collapse RHSMode-2/3 transport into one assembly/policy/solve/finalize path;
   move scaling campaigns out of stable.
7. Collapse solver/preconditioner families to admitted defaults with strict
   residual gates; extract experimental factorization research.
8. Collapse output schemas and writer/reader dispatch; keep all public output
   keys tested against compact fixtures.
9. Curate examples to original v3 examples plus <=10 workflows: CLI solve/plot,
   Python solve, output formats, transport coefficients, bootstrap/Redl,
   ambipolar root, autodiff/JVP, VMEC/Boozer loading, optimization objective,
   and validation fixture comparison.
10. Consolidate tests to domain-parametrized physics/numerics/API suites; reach
    >=95% meaningful coverage without default CI exceeding 10 minutes.
11. Regenerate README/docs parity/runtime/memory/bootstrap figures and tables
    from retained workflows only; remove unsupported claims.
12. Run final checks, push PR #8 as the single draft/review PR, and do not merge
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
sparse-direct research, nested-dissection/multifrontal/HSS replacements,
lower-memory preconditioner research, GPU/multi-GPU campaigns, publication
audits, and long stellarator optimization campaigns. They may be referenced in
`docs/research_lanes.rst` only; they must not remain as stable source,
examples, tests, README claims, or default solver branches.
