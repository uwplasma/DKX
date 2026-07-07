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

- Latest audited branch head: `9bfa7dc4`; last green CI evidence: `96fb2677`.
  Exact aggregate coverage from that CI was 91.753% (`92%`).
- Package layout is shallow, but the branch is too large: 120 Python source
  files and about 166k source lines. Largest complexity owners are
  `problems/`, `solvers/`, and RHSMode-1/QI/preconditioner infrastructure.
- Non-package volume is also too high: `tests/` has 726 files, `examples/` has
  497 files, `benchmarks/` has 268 files, and `scripts/` has 47 files.
- No tracked file larger than 2 MB was found. There is no tracked
  `sfincs_jax/data/`; release-data logic is `validation/data_fetch.py`.
- The product target changed: PR #8 should become a small stable core, while
  experimental or not-yet-competitive work moves to separate research PRs.

## Open Lanes

| Lane | Status | Completion | Definition of done |
| --- | --- | ---: | --- |
| Line-by-line audit | Active | 10% | Every retained file/function/line has a core reason, a caller, and a test/doc owner; everything else is extracted or deleted. |
| Core-main slimming | Active | 35% | Main keeps only stable, parity-clean, runtime-acceptable solvers and public APIs; research code is outside core. |
| Source simplification | Active | 45% | Package moves toward <=50 source files and <=50k lines, with a 10x reduction as stretch if functionality permits. |
| Examples/tests/scripts cleanup | Active | 35% | Examples are <=10 curated workflows plus Fortran-v3 references; tests are smaller, organized, and >=95% coverage; scripts/benchmarks are removed or promoted. |
| Parity/performance evidence | Active | 70% | Supported examples rerun against SFINCS Fortran v3 with runtime/RSS/bootstrap evidence; unsupported research lanes are not marketed. |
| Docs/readme regeneration | Active | 80% | README/docs describe only the stable core plus explicit external research PR lanes. |

Overall readiness under this stricter core-slim goal is about 55-60%.

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

1. Create preservation branches/PRs, then remove from core:
   `research/qi-device-hard-seed` for `preconditioner_qi_*`,
   `profile_sparse_qi.py`, QI validation, QI scripts, QI examples, and QI tests.
2. Create `research/native-sparse-direct` for symbolic/native sparse direct,
   true-operator rescue, reduced-Pmat, active-factor, multifrontal/Schur, and
   related audit/profile scripts and tests.
3. Create `research/parallel-performance` for multi-GPU/sharded/worker scaling
   campaigns, long profiling scripts, and performance-only docs/artifacts.
4. Create `research/publication-audits` for historical figure/collisionality/
   high-nu/Zenodo campaigns that are not needed for core installation or use.

- Core imports no longer reach extracted modules.
- Extracted research PRs preserve useful work but are not required by main.
- Core source count and line count drop materially after each extraction.

### Phase C - Reduce Core Solvers And Operators

1. Keep only solver paths that pass strict residual/parity gates and have
   competitive runtime/RSS on the supported examples.
2. Remove duplicated routing layers, fail-closed probes, and solver policies
   whose only purpose is to select unpromoted experimental paths.
3. Simplify RHSMode-1 and RHSMode-2/3 orchestration so physics assembly,
   linear solve, diagnostics, and output are visible in short modules.
4. Re-profile retained examples after each solver cut; if runtime regresses,
   fix JIT boundaries, operator reuse, sparse/dense selection, and output I/O
   before adding new algorithms.

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

1. Remove top-level `benchmarks/` from core. Keep only small summary fixtures
   and one bounded test path for README/docs number consistency.
2. For `scripts/`, promote user commands to examples, validation generators to
   tests/docs helpers, release-maintenance commands to documented scripts, and
   one-off audit/profiling/benchmark scripts to research PRs or deletion.

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

- Experimental QI and true device-QI solver/preconditioner research.
- Native MUMPS/SuperLU_DIST-equivalent sparse-direct replacement research.
- Lower-memory production preconditioner experiments not promoted by strict
  residual, parity, runtime, and RSS gates.
- Long-run GPU/multi-GPU performance campaigns while GPU evidence is absent.
- Historical benchmark campaigns, profiler traces, and publication audits.
