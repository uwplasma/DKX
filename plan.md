# SFINCS_JAX Active Execution Plan

Last updated: 2026-06-21 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review PR: #8, `refactor/v3-driver-architecture`, is open, non-draft,
merge-clean, and green at commit `43edc4f`. Keep that PR stable unless CI or
review exposes a real defect. This branch is the follow-up refactor track.

Historical checkpoint detail is intentionally not kept in this file. It remains
available in git history. This file is the current authoritative plan.

## Objective

Make `sfincs_jax` research-grade while preserving the public user contract:

- `sfincs_jax input.namelist` should choose accurate, fast, memory-aware
  defaults without requiring users to understand solver internals.
- The Python API should keep end-to-end differentiable lanes for sensitivity
  analysis, inverse design, uncertainty quantification, and optimization.
- CLI and non-autodiff production lanes may use faster host/native paths when
  that improves runtime or memory.
- Parity with SFINCS Fortran v3 remains the trust gate where comparable runs
  exist.
- Runtime and memory should be competitive and should not regress silently.
- The codebase should be maintainable: cohesive domain modules, explicit data
  contracts, focused tests, useful docstrings, and fewer monolithic functions.

## Non-Negotiable Constraints

- Preserve numerical behavior and output schemas at every refactor step.
- Keep public examples, README figures, benchmark tables, and docs honest:
  regenerate them only from complete checked reports.
- Do not add large tracked artifacts. Equilibria and benchmark outputs belong
  in releases or generated caches, not in git.
- Do not grow coverage with smoke-only scaffolds. Tests should protect physics,
  numerical, API, I/O, policy, or regression behavior.
- Keep JAX-facing kernels pure and transformation-friendly. Host-only caches and
  sparse factorization stay outside differentiable code paths.
- Do not create broad new top-level module families. New code should live under
  domain packages such as `problems/`, `solvers/`, `physics/`, `input/`,
  `validation/`, or `workflows/` unless it is a compatibility shim.

## Current Evidence

- Worktree is clean on `refactor/rhs1-full-assembly-preconditioners`.
- Latest checked completed follow-up branch Docs workflow passed. Avoid repeated
  CI polling; inspect only completed failures or final review gates.
- PR #8 CI/docs/examples/external-data/optional gates/coverage checks passed.
- Repository size audit has passed after the plan compaction work.
- No README benchmark or parity figure regeneration is required for current
  behavior-preserving refactor work.

Current largest source files:

- `sfincs_jax/v3_driver.py`: about 14.4k lines.
- `sfincs_jax/rhs1_full_assembly.py`: about 11.4k lines.
- `sfincs_jax/io.py`: about 5.8k lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: about 3.6k lines.

Current structural issue:

- The refactor successfully introduced domain packages, but there are still many
  compatibility shims and top-level `rhs1_*` / `transport_*` modules. Keep the
  shims for user/test compatibility, but do not add more generic top-level
  modules. Move new implementation into clear domain packages.

Latest local validation for this branch:

- Native-stack/sparse-coarse policy extraction: targeted `py_compile` passed.
- Targeted `ruff` passed for the new Schur policy module, package facade,
  `rhs1_full_assembly.py`, and touched tests.
- `tests/test_rhs1_coarse_policy.py`, domain package import contracts, and
  policy docstring/source-map tests passed (`15 passed`).
- `tests/test_rhs1_full_assembly.py` passed (`121 passed`).
- `tests/test_v3_sparse_pattern.py` passed (`132 passed`).
- Final hygiene for the native-stack/sparse-coarse policy extraction passed:
  targeted `py_compile`, targeted `ruff`, `git diff --check`, repository-size
  audit, and Sphinx HTML docs build.

## Open Lanes

### 1. PR #8 Review Readiness

Status: 100% for current evidence.

Actions:

- Keep PR #8 stable.
- Do not merge PR #8 until review is complete.
- Check CI only after completed runs or when a real failure is reported.
- If review requests changes, make the smallest behavior-preserving patch.

Acceptance:

- PR remains merge-clean.
- Required CI/docs checks pass.
- No generated outputs or large files are committed.

### 2. Follow-Up RHSMode=1 Full-Assembly Refactor

Status: about 48%.

Completed on this branch:

- Active projected auto-policy split.
- Fortran-v3-reduced factor-policy split.
- Symbolic frontal policy split.
- Symbolic sparse policy split.
- Native-stack and sparse-coarse residual policy split into
  `sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_policy`.

Open work:

- Move one complete RHSMode=1 preconditioner family implementation into the
  `solvers/preconditioners/` domain tree once policy boundaries are stable.
- Keep compatibility aliases only where tests/debug scripts depend on private
  names.

Acceptance:

- Each extraction reduces the old call site or makes it explicitly typed.
- Policy modules are side-effect-free and directly tested.
- Numerical builders, factorization, residual admission, cache ownership, and
  replay state remain behavior-compatible.
- Local focused tests pass before each commit.

### 3. Driver Reviewability

Status: about 88%.

Open work:

- Continue shrinking `solve_v3_full_system_linear_gmres(...)` only at cohesive
  stage boundaries.
- Keep cache keys, callback construction, replay mutation, and residual-vector
  routing explicit in the driver until the replacement boundary is smaller and
  typed.
- Avoid moving code just to decrease line count if it creates a harder-to-read
  call graph.

Acceptance:

- `v3_driver.py` is primarily orchestration plus compatibility seams.
- Any remaining large driver block is documented as driver-owned state
  management, not hidden solver policy.

### 4. Domain Package Simplification

Status: about 70%.

Open work:

- Prefer these domains for new implementation:
  `problems/profile_response/`, `problems/transport_matrix/`,
  `solvers/preconditioners/`, `physics/`, `input/`, `validation/`,
  and `workflows/`.
- Gradually convert top-level `transport_*` modules into compatibility aliases
  over `problems/transport_matrix/*`.
- Gradually convert top-level implementation-heavy `rhs1_*` modules into
  `problems/profile_response/*` or `solvers/preconditioners/*` modules.
- Do not create many thin files with vague names. File names should describe the
  physical or numerical responsibility, not just the old branch name.

Acceptance:

- Public imports remain compatible.
- New code has one obvious home.
- Source map and API docs explain compatibility shims versus implementation
  modules.

### 5. Differentiability And Solver-Lane Separation

Status: about 78%.

Open work:

- Keep autodiff-facing Python lanes JAX-native and transformation-safe.
- Keep host sparse factors and host-only production shortcuts outside autodiff
  paths.
- Add targeted gradient/implicit-differentiation tests when solver-selection,
  residual-equation, or public API boundaries are touched.
- Document the difference between differentiable Python workflows and
  non-autodiff CLI production workflows.

Acceptance:

- Differentiable APIs remain explicit and tested.
- Adaptive/default solver selection does not silently put autodiff users onto a
  host-only path.

### 6. Validation, Coverage, And Documentation

Status: about 96% for current refactor validation, lower for the long-term 95%
coverage target.

Open work:

- Keep adding focused unit/regression tests at every extraction boundary.
- Add physics/numerical tests only when they protect a real equation,
  normalization, residual, conservation property, or literature-anchored gate.
- Do not run slow production benchmark campaigns for behavior-preserving
  refactors.
- Update docs/source maps after module moves and API-facing changes.

Acceptance:

- Targeted local tests pass for each tranche.
- Broader shards pass after larger structural milestones.
- Docs build succeeds after source-map changes.

### 7. Performance, Memory, And Production Benchmarks

Status: deferred for this follow-up refactor unless behavior changes.

Open work:

- Re-run CPU/GPU/Fortran production-resolution benchmark and parity campaigns
  only after the current source split stabilizes or solver behavior changes.
- Use `ssh office` only for GPU or production-scale validation that cannot be
  done locally.
- Keep benchmark/README figures unchanged during behavior-preserving refactors.

Acceptance:

- Any regenerated public plot/table is based on complete checked reports.
- Runtime and memory regressions are captured by documented benchmark gates, not
  by partial exploratory runs.

## Finite Execution Sequence

1. Keep PR #8 stable and review-ready.
2. On this branch, move one complete RHSMode=1 preconditioner family
   implementation into the `solvers/preconditioners/` domain tree.
3. Validate with targeted `py_compile`, `ruff`, `tests/test_rhs1_full_assembly.py`,
   `tests/test_v3_sparse_pattern.py`, `git diff --check`, and repository-size
   audit.
4. Update `docs/source_map.rst` and this plan after each successful structural
   tranche.
5. Commit and push each complete tranche.
6. Move one complete RHSMode=1 preconditioner family implementation into the
   `solvers/preconditioners/` domain tree.
7. Run a broader RHSMode=1/profile-response shard after the first implementation
   family move.
8. Split `io.py` only after RHSMode=1 boundaries are stable, starting with
   output schema construction versus file writers.
9. Reassess whether this follow-up branch should become a second PR or whether
   only PR #8 should proceed first.

## Completion Criteria

This plan is complete only when current evidence shows:

- PR #8 is either merged after review or explicitly kept as the only open
  review target.
- This follow-up branch has no dirty or untracked generated artifacts.
- Required local validation for every tranche passes.
- CI required jobs pass for review branches.
- Public CLI and Python APIs still work.
- Differentiable and non-differentiable lanes are explicit and tested.
- Any behavior or performance claims changed in docs/README are regenerated
  from checked reports.
- Repository size gates pass without allow-listing avoidable planning/output
  blobs.
