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

- RHSMode=1 reduced sparse-operator matvec admission now uses a tested
  side-effect-free policy helper, including implicit-solve and size rejection
  messages while driver-local operator materialization remains unchanged
  (current checkpoint).
- RHSMode=1 sparse-preconditioner env/default parsing now uses a tested
  profile-response policy config object, including sparse backend aliases,
  non-diff/matvec/operator switches, PAS/DKES size defaults, drop controls,
  and dense-cache limits (`8d90924`).
- RHSMode=1 sparse-JAX retry env/config parsing now uses a tested
  profile-response policy config object instead of driver-local parsing
  (`908fb81`).
- Sparse rescue trace-message formatting for reduced and full branches now uses
  tested side-effect-free policy helpers while preserving driver emission order
  (`049d39e`).
- Sparse rescue enable/kind/order setup and sparse-JAX memory-admission
  messaging for reduced and full branches now uses one tested policy helper;
  solve execution and cache policy remain driver-owned (`bb99065`).
- Sparse-JAX retry preconditioner build/progress emission for reduced and full
  branches now uses a tested sparse-PC helper while cache-key policy remains
  driver-owned (`60b18f6`).
- Sparse-JAX Jacobi retry branches now reuse the existing measured
  linear-candidate handoff helper instead of manual timer/solve/accept blocks
  (`ac48e62`).
- Host SciPy GMRES execution/result wrapping for reduced, full, and
  sparse-operator-preconditioned sparse fallback branches now uses a tested
  sparse-PC helper; the driver still owns solver controls and admission gates
  (`bc3db1d`).
- Compact active-plan validation log restored (`f4435ad`).
- Sparse JAX and host sparse retry measured-acceptance/replay updates now use
  a tested handoff helper with consistent candidate/baseline naming and
  candidate-seed replay state (`7266d37`).
- Reduced active-DOF, full-system, and sparse-operator-preconditioned host
  SciPy fallback paths now share a tested sparse-PC callback builder for the
  host factor apply and optional explicit-matrix matvec (`19abab1`).
- Reduced active-DOF and full-system implicit sparse-ILU preconditioner
  construction now uses a tested profile-response helper with explicit
  dense-vs-padded triangular factor modes; legacy private triangular helper
  exports in `v3_driver.py` remain available for tests/debug scripts
  (`da1cf20`).
- Reduced active-DOF and full-system sparse host/direct-vs-ILU factor setup now
  uses a tested profile-response helper; cache keys, matvecs, explicit sparse
  patterns, and host-only callbacks remain driver-owned (`c5cccef`).
- RHSMode=1 PAS adaptive smoother handoff now uses a tested replay-aware
  helper while preserving explicit reduced/full residual-vector routing
  (`33aba27`).
- X-block sparse-PC post-residual-equation correction now shares the same
  subspace-correction helper, preserving cached-QI progress diagnostics
  (`d45d9f8`).
- X-block sparse-PC post-coarse correction now uses a tested subspace
  correction helper while keeping direction construction driver-local
  (`e99286e`).
- X-block sparse-PC post-minres now reuses the generic sparse-PC post-minres
  helper with a solver-label parameter (`892beaf`).
- RHSMode=1 strict linear retry handoff extraction for forced full,
  collision-preconditioner, and PAS Schur rescue branches
  (`353851c`).
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

Current source-size snapshot after sparse-operator admission extraction:

- `sfincs_jax/v3_driver.py`: `18064` lines.
- `solve_v3_full_system_linear_gmres`: `12739` lines.
- `sfincs_jax/v3_results.py`: `119` lines.
- `sfincs_jax/problems/profile_response/residual.py`: `981` lines.
- `sfincs_jax/problems/profile_response/handoff.py`: `598` lines.
- `sfincs_jax/problems/profile_response/policies.py`: `2937` lines.
- `sfincs_jax/problems/profile_response/dense.py`: `407` lines.
- `sfincs_jax/problems/profile_response/linear_solve.py`: `327` lines.
- `sfincs_jax/problems/profile_response/active_projection.py`: `116` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: `8543` lines.

Recent local validation:

- Sparse rescue policy/docstring shard after sparse-operator admission
  extraction:
  `22 passed in 0.65 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.14 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse-operator admission extraction:
  `1042 passed in 49.47 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse rescue policy/docstring shard after sparse preconditioner config
  extraction:
  `19 passed in 0.63 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.51 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse preconditioner config extraction:
  `1039 passed in 49.77 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse rescue policy/docstring shard after sparse-JAX config extraction:
  `16 passed in 0.64 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.85 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse-JAX config extraction:
  `1036 passed in 50.04 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse rescue policy shard after sparse rescue message extraction:
  `10 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.83 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse rescue message extraction:
  `1033 passed in 49.43 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse rescue policy shard after sparse rescue policy setup extraction:
  `8 passed in 0.35 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.81 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse rescue policy setup extraction:
  `1031 passed in 49.48 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse-PC helper shard after sparse-JAX build/progress extraction:
  `187 passed in 1.96 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.88 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse-JAX build/progress extraction:
  `1029 passed in 49.84 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 handoff helper shard after sparse-JAX measured retry reuse:
  `34 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.44 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse-JAX measured retry reuse:
  `1028 passed in 50.38 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse-PC helper shard after host SciPy sparse GMRES extraction:
  `186 passed in 1.98 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.87 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after host SciPy sparse GMRES extraction:
  `1028 passed in 49.73 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- RHSMode=1 handoff helper shard after sparse retry handoff extraction:
  `34 passed in 0.37 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.20 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse retry handoff extraction:
  `1026 passed in 49.57 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse-PC helper shard after host SciPy sparse callback extraction:
  `184 passed in 1.98 s`.
- Sparse-PC helper shard after implicit ILU preconditioner extraction:
  `181 passed in 2.00 s`.
- Sparse host/direct-vs-ILU factor setup helper shard:
  `178 passed in 1.47 s`.
- Older focused and broad validation checkpoints are intentionally omitted from
  this active plan; they remain available in git history.

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
- X-block sparse-PC post-minres residual-polish orchestration now uses the
  same tested helper as generic sparse-PC GMRES, with stable x-block progress
  labels and unchanged metadata variables.
- X-block sparse-PC post-coarse residual-polish orchestration now uses a
  reusable subspace-correction helper with explicit driver-provided direction
  builders, stable progress labels, and unchanged metadata variables.
- X-block sparse-PC post-residual-equation orchestration now uses the same
  subspace-correction helper with explicit driver-provided direction builders
  and cached-QI kwargs/suffix diagnostics preserved.
- RHSMode=1 reduced active-DOF and full-system PAS adaptive smoother
  candidate/replay handoff now uses a tested helper; the reduced branch keeps
  the current residual vector and the full branch still explicitly recomputes
  `rhs - A x` at the call site.
- Reduced active-DOF and full-system sparse host/direct-vs-ILU factor setup now
  uses a tested helper; explicit sparse patterns, cache keys, and matrix-free
  callbacks stay in the driver while the generic factor-selection result
  schema lives in `profile_response.sparse_pc`.
- Reduced active-DOF and full-system implicit sparse-ILU preconditioner
  construction now uses a tested helper; dense triangular and padded sparse
  triangular modes are tested, and branch-specific lower-diagonal admission is
  explicit.
- Reduced active-DOF, full-system, and sparse-operator-preconditioned host
  SciPy fallback branches now use one tested callback builder for the host
  factor apply and optional explicit sparse matvec.
- Sparse JAX and host sparse retry candidate acceptance now uses a tested
  measured-handoff helper with driver-provided matvec/preconditioner/rhs
  routing preserved.
- Host SciPy GMRES execution/result wrapping for reduced, full, and
  sparse-operator-preconditioned sparse fallback branches now uses a tested
  helper; optional true residual-vector construction remains explicit by
  call-site through a provided residual matvec.
- Sparse-JAX Jacobi retry branches now use the shared measured linear-candidate
  handoff helper, preserving reduced/full residual-vector routing.
- Sparse-JAX retry preconditioner build/progress emission now uses a tested
  helper with driver-provided cache keys and builder callback.
- Sparse rescue enable/kind/order setup and sparse-JAX memory-admission
  messaging now uses a tested policy helper shared by reduced active-DOF and
  full-system branches.
- Sparse rescue initial and tail skip trace-message formatting now uses tested
  policy helpers while preserving driver emission order.
- RHSMode=1 sparse-JAX retry max-memory, sweep count, damping, and
  regularization env parsing now uses a tested profile-response config object.
- RHSMode=1 sparse-preconditioner backend, non-diff/matvec/operator toggles,
  PAS/DKES sizing, drop controls, dense-factor cap, and cache cap parsing now
  uses a tested profile-response config object.
- RHSMode=1 reduced sparse-operator matvec admission now uses a tested
  profile-response policy helper while preserving driver-local operator
  construction and cache-key ownership.

Next steps:

- Move remaining generic sparse-PC solve/result assembly into cohesive
  `profile_response` helpers.
- Continue extracting sparse-PC state/metadata seams after the source split
  stabilizes; avoid moving driver-specific direction builders or caches into
  generic helpers.
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
