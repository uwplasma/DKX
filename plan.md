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

- RHSMode=1 strict linear retry handoff extraction for forced full,
  collision-preconditioner, and PAS Schur rescue branches
  (current checkpoint).
- RHSMode=1 strong-retry reuse of measured linear handoff
  (`0d35c2b`).
- RHSMode=1 measured linear retry handoff extraction for reduced/full stage2
  (`2c4d7e0`).
- RHSMode=1 CPU SciPy rescue execution helper extraction
  (`c6e31bc`).
- Projected residual-polish helper extraction for FP L1/global low-L paths
  (`62cc9ab`).
- FP low-L/L1 active-index helper extraction
  (`00f7b9e`).
- Damped preconditioned residual-polish helper extraction
  (`a8197f4`).
- RHSMode=1 reduced dense fallback candidate extraction
  (`fd46465`).
- RHSMode=1 fast post-xblock polish handoff extraction
  (`029e1c5`).
- RHSMode=1 true-residual `GMRESSolveResult` helper extraction
  (`62948d0`).
- Payload-to-`V3LinearSolveResult` result-layer wrapper extraction
  (`7652c24`).
- Explicit sparse-host direct factor/solve orchestration extraction
  (`1d43c2d`).
- Explicit sparse minimum-norm materialization/solve orchestration extraction
  (`d802be4`).
- RHSMode=1 sparse-host direct fallback progress emission centralization
  (`8fd4866`).
- RHSMode=1 sparse-host direct fallback solve/polish/residual orchestration
  extraction (`3fa677a`).
- RHSMode=1 dense-shortcut true-residual scalar helper extraction
  (`57153f9`).
- RHSMode=1 left-preconditioned replay residual norm helper extraction
  (`9ce5c27`).
- RHSMode=1 true-residual recomputation helper extraction
  (`6fbd66a`).
- RHSMode=1 candidate accept-and-replay handoff consolidation
  (`7d31161`).
- Implicit-solve host-only Krylov downgrade contract and gradient tests
  (`04555a5`).
- Generic sparse-PC dtype-retry/finalization handoff extraction
  (`f3af854`).
- X-block sparse-PC final-payload extraction.
- Explicit-sparse factor-builder compatibility wrapper simplification.
- Fortran-reduced x-block final-payload extraction and duplicate sparse-rescue
  metadata cleanup.
- X-block sparse-PC metadata helper extraction.
- Sparse-PC finalization helper extraction.
- KSP replay-state contract extraction.
- Measured candidate handoff consolidation.
- Sparse fallback measured-handoff extraction.
- Sparse-PC factor-preflight evaluation extraction.
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

Current source-size snapshot after RHSMode=1 strong-retry measured handoff
reuse:

- `sfincs_jax/v3_driver.py`: `18450` lines.
- `solve_v3_full_system_linear_gmres`: `13145` lines.
- `sfincs_jax/v3_results.py`: `119` lines.
- `sfincs_jax/problems/profile_response/residual.py`: `981` lines.
- `sfincs_jax/problems/profile_response/handoff.py`: `417` lines.
- `sfincs_jax/problems/profile_response/dense.py`: `407` lines.
- `sfincs_jax/problems/profile_response/linear_solve.py`: `327` lines.
- `sfincs_jax/problems/profile_response/active_projection.py`: `116` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: `8034` lines.

Recent local validation:

- RHSMode=1 handoff helper shard:
  `27 passed in 0.33 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 127 deselected in 32.24 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `1008 passed in 48.82 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 handoff helper shard:
  `27 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 127 deselected in 32.34 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `1008 passed in 49.18 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 linear-solve helper shard:
  `5 passed in 1.05 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 33.01 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `1005 passed in 48.49 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 residual/active-projection helper shard:
  `21 passed in 1.36 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 33.41 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `1003 passed in 48.85 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 active-projection helper shard:
  `4 passed in 0.71 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 34.08 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `1001 passed in 48.36 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 residual helper shard:
  `15 passed in 0.79 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 32.34 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `999 passed in 47.06 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 reduced dense fallback helper shard:
  `6 passed in 1.09 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 33.31 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `997 passed in 46.75 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after RHSMode=1 fast post-xblock polish extraction:
  `995 passed in 47.83 s`.
- RHSMode=1 handoff helper unit shard:
  `24 passed in 0.31 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 124 deselected in 31.16 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 residual helper unit shard:
  `13 passed in 0.74 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 113 deselected in 35.30 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Result-wrapper/profile-response sparse-PC focused shard:
  `177 passed in 1.77 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.10 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Explicit sparse-host direct focused sparse-PC shard:
  `173 passed in 1.68 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 33.63 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after explicit sparse host extractions:
  `992 passed in 42.63 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Explicit sparse minimum-norm focused sparse-PC shard:
  `172 passed in 1.63 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 33.70 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse-host fallback orchestration focused shard:
  `170 passed in 1.61 s`.
- RHSMode=1 residual/handoff/sparse-PC/diagnostics shard:
  `215 passed in 1.40 s`.
- Explicit sparse-host/direct-tail driver shard:
  `32 passed, 100 deselected in 32.57 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep:
  `989 passed in 43.41 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 dense-shortcut residual/refactor focused shard:
  `37 passed in 8.61 s`.
- Profile-response dense-shortcut residual/handoff/sparse-PC shard:
  `214 passed in 1.69 s`.
- RHSMode=1 replay residual/refactor focused shard:
  `35 passed in 9.78 s`.
- Profile-response replay residual/handoff/sparse-PC shard:
  `212 passed in 1.79 s`.
- RHSMode=1 residual/refactor focused shard:
  `31 passed in 9.48 s`.
- Profile-response residual/handoff/sparse-PC shard:
  `208 passed in 1.84 s`.
- RHSMode=1 accept-and-replay handoff shard:
  `70 passed in 14.17 s`.
- Broad profile-response/x-block/sparse-pattern handoff shard:
  `394 passed in 112.33 s`.
- Implicit/autodiff host-only solve-method downgrade shard:
  `15 passed in 4.91 s`.
- Sparse-PC helper shard:
  `169 passed in 1.74 s`.
- Profile-response diagnostics plus explicit sparse/direct-tail driver checks:
  `16 passed in 15.27 s`.
- Sparse helper coverage plus explicit sparse factor-builder tests:
  `18 passed in 1.22 s`.
- End-to-end explicit sparse/direct-tail driver checks:
  `5 passed in 14.22 s`.
- Latest broad profile-response/x-block/sparse-pattern shard:
  `448 passed in 114.37 s`.
- Handoff/replay and profile-response diagnostics shard:
  `35 passed in 0.76 s`.
- Handoff helper unit shard:
  `20 passed in 0.34 s`.
- Focused sparse-PC shard:
  `164 passed in 1.68 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Latest pushed CI/Docs before this tranche (`eff0ba4`) are green.
- Latest pushed Docs for sparse fallback measured-handoff (`c3802d5`) are green;
  CI is in progress and not yet waited on.
- Latest pushed Docs for measured candidate consolidation (`6ec04b2`) are green;
  CI is in progress and not yet waited on.
- Latest pushed CI/Docs for KSP replay-state contract extraction (`75ae32d`)
  are green.
- Latest pushed Docs for sparse-PC finalization helper extraction (`e793d77`)
  are green; CI is in progress and not yet waited on.
- Latest pushed Docs for x-block sparse-PC metadata helper extraction
  (`c7cddc3`) are green; CI is queued and not waited on.

Known CI issue fixed by this rewrite:

- `tests/test_repo_size_policy.py::test_tracked_large_files_are_reviewed`
  failed because `plan.md` reached `2.01 MiB`.

## Active Work Lanes

### 1. `v3_driver.py` Architecture Refactor

Completion estimate: 98%.

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
- Sparse-PC auto-preflight retry candidate selection and scalar policy
  evaluation.
- Sparse-PC GMRES stagnation and post-minres control policy parsing.
- Sparse-PC GMRES result metadata schema extraction.
- Sparse-PC factor-dtype retry decision, seed selection, and driver-state
  orchestration extraction.
- Sparse-PC post-minres default/update bookkeeping and driver-state
  orchestration extraction.
- Sparse-PC GMRES completion-message formatting and emit orchestration
  extraction.
- Sparse-PC final payload, convergence acceptance, and factor-quality metadata
  bookkeeping extraction.
- Sparse minimum-norm (`sparse_lsmr`/`sparse_lsqr`/PETSc-compatible)
  policy parsing, LSQR/LSMR execution, residual gating, progress-message, and
  result-metadata extraction.
- Explicit sparse-host direct-solve refinement, true-residual recomputation,
  completion-message, and result-metadata extraction.
- Shared explicit sparse conservative-pattern progress, CSR/drop policy
  parsing, and minimum-norm operator-materialization extraction.
- Explicit host sparse non-autodiff request validation extraction.
- Host sparse direct factor-dispatch extraction for explicit-factor versus
  ILU/CSR refinement fallbacks.
- Host sparse direct float32 GMRES-polish gating, progress, and acceptance
  extraction.
- RHSMode=1 measured solver-candidate metric construction and sparse host/JAX
  fallback handoff extraction.
- RHSMode=1 stage-2, strong-retry, dense fallback, and sparse fallback measured
  candidate handoff consolidation; driver-local metrics builder removed.
- RHSMode=1 KSP replay diagnostics state extracted into a profile-response
  handoff contract with unit tests.
- Sparse-PC final post-minres, completion emission, and final payload assembly
  consolidated into a single profile-response helper.
- X-block sparse-PC result and correction metadata handoffs consolidated into
  one profile-response helper.
- Fortran-reduced x-block Krylov result expansion, true-residual acceptance,
  factor-quality gate, and final metadata payload assembly consolidated into a
  profile-response helper.
- Duplicate intermediate sparse-rescue metadata updates removed; the final
  combined sparse-rescue tail update remains the single authoritative path.
- Explicit-sparse factor-builder wrapper in `v3_driver.py` simplified to a
  patchable compatibility shim so the builder schema is defined only in
  `explicit_sparse_factor_builder.py`.
- X-block sparse-PC final vector expansion, residual payload, and convergence
  acceptance metadata consolidated into a profile-response helper.
- Generic sparse-PC dtype retry, state handoff, post-minres finalization, and
  final payload construction consolidated into one profile-response helper.
- RHSMode=1 rescue/refinement candidate acceptance and KSP replay-state updates
  consolidated into profile-response handoff helpers.
- RHSMode=1 true-residual recomputation before fallback decisions consolidated
  into a tested profile-response residual helper.
- RHSMode=1 left-preconditioned replay residual norm measurement for
  dense-fallback gates consolidated into a tested residual helper.
- RHSMode=1 dense-shortcut true-residual scalar measurement consolidated into
  a tested residual helper.
- RHSMode=1 host sparse direct fallback solve, optional float32 polish, and
  true-residual-vector recomputation consolidated into a tested
  profile-response helper used by both reduced active-DOF and full-system
  fallback branches.
- RHSMode=1 host sparse direct fallback progress emission consolidated into
  the same helper, removing duplicated driver-side progress-line blocks.
- Explicit sparse minimum-norm conservative-pattern materialization, progress
  emission, policy parsing, LSQR/LSMR execution, completion emission, and
  matrix-required gate consolidated into a tested profile-response helper.
- Explicit sparse-host direct conservative-pattern progress, host factor build,
  direct refinement solve, true-residual payload, and completion emission
  consolidated into a tested profile-response helper.
- Payload-to-`V3LinearSolveResult` conversion consolidated in the result layer,
  removing repeated driver-side `GMRESSolveResult` wrapping for sparse-PC,
  x-block, and explicit host sparse payloads.
- RHSMode=1 true-residual result/vector construction consolidated in the
  profile-response residual module and reused by dense and SciPy rescue paths.
- RHSMode=1 fast post-xblock polish execution, progress emission, and
  strict residual-improvement acceptance consolidated into a tested
  profile-response handoff helper.
- RHSMode=1 reduced dense fallback candidate execution consolidated into
  `profile_response.dense` while driver-side measured acceptance remains
  unchanged.
- RHSMode=1 damped preconditioned residual polish consolidated into a tested
  residual helper.
- FP low-L/L1 active mode index construction consolidated into active
  projection helpers, removing duplicated flattened-index loops.
- FP L1 and global low-L projected residual polish consolidated into a tested
  residual-equation helper with explicit projected/full residual gates.
- RHSMode=1 CPU SciPy rescue GMRES/BiCGStab execution consolidated into
  `profile_response.linear_solve`; driver-side thresholds, size caps, metadata,
  and true-residual acceptance remain unchanged.
- Reduced active-DOF and full-system stage2 retry execution/measured
  acceptance consolidated into a replay-aware handoff helper.
- Reduced active-DOF and full-system strong-preconditioner retry branches now
  reuse the same measured handoff helper.
- RHSMode=1 forced full-preconditioner, reduced/full collision-preconditioner,
  and PAS Schur rescue linear retries now reuse a strict-improvement
  replay-aware handoff helper, preserving their non-measured acceptance
  contract.

Next steps:

- Move remaining generic sparse-PC solve/result assembly into cohesive
  `profile_response` helpers.
- Extract the remaining adaptive smoother handoff shape only if it can preserve
  its explicit residual-vector construction without hiding solver policy.
- Extract remaining explicit sparse-host direct factor-setup policy seams only
  where this can be done without pulling driver-specific caches into domain
  modules.
- Continue consolidating duplicated host sparse fallback acceptance/metadata
  seams behind tested profile-response helpers.
- Continue shrinking `solve_v3_full_system_linear_gmres` in behavior-preserving
  tranches.

### 2. Differentiability And Solver-Lane Separation

Completion estimate: 73%.

Goal:

- Keep differentiable Python/API solves JAX-native and transformation-safe.
- Keep CLI/non-autodiff production solves fast and memory-aware.

Next steps:

- Keep host sparse factors out of autodiff paths.
- Keep documenting the CLI/non-autodiff versus JAX/autodiff lane split as
  behavior-facing solver controls change.
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

Completion estimate: 64%.

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

1. Continue with remaining generic sparse-PC solve/result assembly extraction
   where behavior and cache boundaries remain clean.
2. Extract remaining full-system RHSMode=1 collision/PAS-Schur rescue
   orchestration only where replay-state and metadata contracts can stay
   explicit.
3. Run focused implicit/sparse-PC/profile-response shards after each extraction,
   and broad shards only after behavior-facing seams move.
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
