# SFINCS_JAX Active Execution Plan

Last updated: 2026-06-19 (America/Chicago)
Branch: `refactor/v3-driver-architecture`
PR state: one draft refactor PR; do not merge until this plan is complete.

This file is intentionally compact. The previous 35k-line checkpoint log made
`plan.md` exceed the repository size policy. Historical detail remains
available in git history; this file is the current authoritative plan.

## Objective

Make `sfincs_jax` research-grade while preserving the public user contract:

- `sfincs_jax input.namelist` should pick accurate, fast, memory-aware defaults
  without requiring users to know internal solver machinery.
- The Python API should keep end-to-end differentiable lanes for sensitivity,
  inverse design, uncertainty quantification, and optimization.
- CLI/non-autodiff production lanes may use faster host/native paths when that
  improves runtime or memory.
- Parity with SFINCS Fortran v3 remains the trust gate where comparable runs
  exist.
- Runtime and memory should be competitive and should not regress silently.
- The codebase should be maintainable: fewer monolithic functions, cohesive
  domain modules, explicit data contracts, focused tests, and useful docstrings.

## Non-Negotiable Constraints

- Preserve numerical behavior and output schemas at every refactor step.
- Keep public examples, README figures, benchmark tables, and docs honest:
  regenerate them only from complete checked reports.
- Do not add large tracked artifacts. Equilibria and large benchmark outputs
  belong in releases or generated caches, not in git.
- Do not grow coverage with smoke-only scaffolds. Tests should protect physics,
  numerical, API, I/O, policy, or regression behavior.
- Keep JAX-facing kernels pure and transformation-friendly. Host-only caches and
  sparse factorization stay outside differentiable code paths.
- Keep the repo light. No tracked file should exceed the size-policy threshold
  without an explicit reviewed exception.

## Current Refactor State

Recent checkpoints:

- Sparse-PC factor-preflight evaluation extraction (current checkpoint).
- Direct-tail coupled-coarse rescue policy extraction.
- Direct-tail true-active rescue policy extraction.
- Direct-tail residual-rescue policy extraction.
- Sparse-PC factor-preflight policy extraction.
- Direct-tail support-mode preflight extraction.
- Direct-tail structured preconditioner construction/cache extraction.
- Direct-tail structured preconditioner admission extraction.
- Direct-tail materialization extraction.
- `0e9b5fb` Compact active plan.
- `a1721b8` Extract sparse memory preflight.
- `cb295ce` Extract sparse pattern setup.
- `4b6a5b4` Extract sparse factor policy.

Current source-size snapshot after the residual-candidate acceptance extraction:

- `sfincs_jax/v3_driver.py`: `19457` lines.
- `solve_v3_full_system_linear_gmres`: `14128` lines.

Recent local validation:

- Focused sparse-PC/direct-tail/residual-acceptance shard:
  `134 passed in 1.49 s`.
- Latest broad profile-response/x-block/sparse-pattern shard:
  `412 passed in 117.63 s`.

Known CI issue fixed by this rewrite:

- `tests/test_repo_size_policy.py::test_tracked_large_files_are_reviewed`
  failed because `plan.md` reached `2.01 MiB`.

## Active Work Lanes

### 1. `v3_driver.py` Architecture Refactor

Completion estimate: 77%.

Goal:

- Retire the monolithic driver incrementally without behavior drift.
- Keep compatibility seams until downstream tests, docs, and examples migrate.

Completed recent boundaries:

- X-block and Fortran-reduced sparse-PC policy/setup/result extraction.
- Generic sparse-PC active-DOF setup.
- Generic sparse-PC direct-tail result metadata.
- Generic sparse-PC factor policy.
- Generic sparse-PC pattern setup.
- Generic sparse-PC memory-budget preflight.
- Direct-tail materialization setup.
- Structured direct-tail preconditioner admission policy.
- Structured direct-tail preconditioner construction and cache setup.
- Structured direct-tail support-mode preflight setup.
- Sparse-PC factor-preflight policy parsing.
- Direct-tail residual-rescue policy parsing.
- Direct-tail true-active rescue policy parsing.
- Direct-tail coupled-coarse rescue policy parsing.
- Sparse-PC factor-preflight residual evaluation.
- Sparse-PC residual-candidate acceptance/update bookkeeping.

Next steps:

- Extract remaining sparse-PC retry candidate admission/update bookkeeping into
  separately tested helpers.
- Move remaining generic sparse-PC solve/result assembly into cohesive
  `profile_response` helpers.
- Continue shrinking `solve_v3_full_system_linear_gmres` in behavior-preserving
  tranches.

### 2. Differentiability And Solver-Lane Separation

Completion estimate: 70%.

Goal:

- Keep differentiable Python/API solves JAX-native and transformation-safe.
- Keep CLI/non-autodiff production solves fast and memory-aware.

Next steps:

- Keep host sparse factors out of autodiff paths.
- Document the lane split in API/docs once the driver refactor stabilizes.
- Add focused tests for implicit differentiation and branch-stable solver
  selection where practical.

### 3. Performance And Memory

Completion estimate: 76%.

Goal:

- Keep solver selection automatic while avoiding aggressive/incorrect path
  switches.
- Preserve strict residual gates and production memory guards.

Next steps:

- Preserve existing CPU/GPU parity/performance gates while refactoring.
- Re-run production-resolution CPU/GPU and Fortran comparisons after the
  current source split stabilizes.
- Use `ssh office` only for GPU or production-scale validation that cannot be
  done locally.

### 4. Validation, Coverage, And Documentation

Completion estimate: 62%.

Goal:

- Maintain strong physics, numerical, regression, and API tests without making
  CI too slow.
- Keep docs/README aligned with actual capabilities and validated benchmark
  reports.

Next steps:

- Add tests at each extraction boundary.
- Re-run full local/CI suites after larger structural milestones, not after
  every small helper.
- Update docs after behavior-facing APIs or solver controls change.

## Immediate Next Steps

1. Commit and push the residual-candidate acceptance extraction after final
   cleanup.
2. Continue with sparse-PC auto-preflight retry admission/update extraction.
3. Run focused sparse-PC tests and the broad profile-response/x-block/sparse
   shard after each extraction.
4. Snapshot CI but do not wait on queued runs unless a completed failure appears.

## Completion Criteria

This plan is complete only when current evidence shows:

- The refactor branch has no dirty or untracked generated artifacts.
- CI required jobs pass.
- `v3_driver.py` no longer owns large independent solver subsystems that belong
  in domain modules.
- Public CLI and Python APIs still work.
- Differentiable and non-differentiable lanes are explicit and tested.
- Benchmarks/parity/docs are regenerated from checked reports where behavior or
  performance claims changed.
- Repository size gates pass without allow-listing avoidable planning/output
  blobs.
