# SFINCS_JAX Active Execution Plan

Last updated: 2026-06-20 (America/Chicago)
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

- RHSMode=1 reduced strong-preconditioner kind selection now uses a tested
  `resolve_rhs1_reduced_strong_preconditioner_selection(...)` helper. The
  driver still owns user progress messages, builder/cache setup, residual
  admission, and Krylov retry execution, but auto-kind selection, xblock-L
  truncation, PAS weak-skip, guarded PAS-TZ skip, and QI-device skip gates are
  centralized in the strong-preconditioning policy layer.
- RHSMode=1 PAS near-zero-Er small-system default routing now uses a tested
  `rhs1_pas_small_near_zero_er_kind(...)` policy helper. This removes three
  duplicated PAS-lite/PAS-hybrid/xmg threshold blocks from `v3_driver.py` while
  preserving the existing `SFINCS_JAX_PAS_LITE_TZ_MAX` and
  `SFINCS_JAX_PAS_LITE_MIN` behavior.
- RHSMode=1 x-block sparse-PC post-Krylov correction and completion emission
  now use a tested `complete_xblock_post_krylov_stage(...)` helper. The helper
  composes the existing post-solve correction runner with the completion
  progress emitter, while preserving the full correction diagnostics object
  consumed by final metadata.
- RHSMode=1 x-block sparse-PC first-attempt plus optional GMRES fallback solve
  execution now uses a tested `run_xblock_krylov_solve_stage(...)` helper.
  The helper owns first-attempt dispatch, physical true-residual measurement,
  fallback admission/execution, and candidate/final state separation; the
  driver keeps only explicit metadata scalar handoff and post-solve correction
  orchestration.
- RHSMode=1 x-block sparse-PC host/device Krylov progress callbacks now use a
  tested `profile_response.sparse_pc` callback builder. The driver passes an
  explicit timing/emitter context instead of owning local progress closures,
  and fallback GMRES uses the same host-progress callback as the first Krylov
  attempt.
- X-block sparse-PC final metadata now has typed grouped state contexts for
  core solve counters, device/QI state, preflight/probe state, and nested
  assembled-operator/coarse/QI/side-probe metadata. The driver builds
  `XBlockSparsePCFinalMetadataStateContext` directly, so the production
  x-block finalization path no longer consumes the driver frame through
  `locals()`; the legacy wrapper remains only as a compatibility and
  missing-key audit path.
- Generic sparse-PC finalization now builds direct-tail, factor-preflight,
  sparse-pattern, and static solver metadata from typed contexts before
  finalization. The RHSMode=1 generic finalizer receives those five dynamic
  convergence/reporting scalars plus compact metadata dictionaries through
  `SparsePCGMRESFinalizationStateContext`; it no longer consumes a driver
  frame or ad hoc mapping at the production call site.
- X-block sparse-PC physical residual recomputation and reported Krylov
  iteration/matvec counters now use tested `profile_response.sparse_pc`
  helpers, removing duplicated reporting code from the driver while preserving
  fallback semantics.
- RHSMode=1 full-system strong fallback ADI sweep parsing now lives in
  `rhs1_strong_fallback`, matching the reduced strong fallback helper and
  removing the final full-branch inline ADI env parse (`c20e4b1`).
- RHSMode=1 reduced active-DOF strong fallback preconditioner construction now
  uses the shared dispatch helper through `rhs1_strong_fallback`, preserving
  xblock-lmax env parsing, ADI fallback semantics, and driver monkeypatch
  compatibility while removing the manual reduced-kind switch from the driver
  (`d8d1043`).
- RHSMode=1 reduced PAS-Schur strong-retry size downgrade now uses a tested
  profile-response strong-preconditioning helper while strong preconditioner
  construction and solve execution remain driver-owned (`49970d5`).
- RHSMode=1 PAS force-full routing after weak collision-preconditioned solves
  now uses a tested `rhs1_pas_policy` decision helper while fallback
  preconditioner construction and replay remain driver-owned
  (`862451f`).
- RHSMode=1 guarded PAS-TZ and weak PAS MINRES correction controls now use
  tested profile-response strong-preconditioning policy helpers while
  correction application and residual acceptance remain driver-owned
  (`a9bf4f0`).
- RHSMode=1 FP-only strong-preconditioner size guard now uses a tested
  profile-response strong-preconditioning policy helper while strong
  preconditioner construction and solve execution remain driver-owned
  (`ad43877`).
- RHSMode=1 collision-preconditioner retry admission and guarded PAS-TZ
  strong-retry opt-in parsing now use tested policy helpers while collision
  preconditioner construction, strong builder execution, measured acceptance,
  and KSP replay remain driver-owned (`c0e0404`).
- RHSMode=1 full-system PAS Schur rescue admission and retry controls now use
  a tested `rhs1_pas_policy` helper while Schur preconditioner construction,
  rescue solve execution, and KSP replay remain driver-owned
  (`2b70d57`).
- RHSMode=1 PAS adaptive smoother execution controls now use a tested
  `rhs1_pas_policy` helper for sweeps and damping while smoother construction,
  measured acceptance, and KSP replay remain driver-owned (`775d008`).
- RHSMode=1 strong-preconditioner request env normalization and PAS
  force-strong ratio parsing now live in the profile-response
  strong-preconditioning helper module while branch-specific messaging,
  auto-kind selection, build, solve, and replay remain driver-owned
  (`a51497f`).
- RHSMode=1 strong-preconditioner retry restart/maxiter controls now use a
  tested profile-response strong-preconditioning helper shared by reduced and
  full-system strong fallback solves while build, solve, measured acceptance,
  and replay remain driver-owned (`5fc3c55`).
- RHSMode=1 strong-preconditioner residual-trigger controls now use a tested
  profile-response strong-preconditioning helper for residual ratio parsing,
  reduced PAS delayed fallback defaults, tokamak PAS delay, and FP absolute
  force thresholds while auto kind selection, build, solve, and replay remain
  driver-owned (`18603cb`).
- RHSMode=1 small-system GMRES cutoff parsing now lives next to profile
  linear-solve routing in `profile_response.linear_solve` while profile
  context construction and solve execution remain driver-owned
  (`1566f2f`).
- RHSMode=1 Stage-2 admission and elapsed-time budget controls now use a
  tested profile-response policy helper while solver-kind classification,
  fallback retry execution, and replay updates remain driver-owned
  (`a6d5def`).
- RHSMode=1 Stage-2 retry restart/maxiter/method controls now use a tested
  profile-response policy helper shared by reduced and full-system fallback
  branches while retry admission, preconditioner construction, measured
  acceptance, and replay updates remain driver-owned (`ac4ac77`).
- Shared KSP diagnostics env parsing for RHSMode=1 and Phi1 Newton-Krylov now
  lives in `rhs1_ksp_diagnostics`, covering Fortran-style stdout, bounded KSP
  history replay, and optional iteration statistics while diagnostic replay
  execution remains in the existing diagnostics helpers (`3bc6461`).
- RHSMode=1 Krylov routing controls now use tested profile-response policy
  helpers for GMRES precondition-side validation and distributed Krylov solver
  normalization while sharded matvec selection and solve execution remain
  driver-owned (`ccd4398`).
- RHSMode=1 BiCGStab-to-GMRES fallback controls now use tested
  profile-response policy helpers for strict-mode parsing and the distributed
  PAS absolute-floor target while fallback solve execution and KSP replay
  updates remain driver-owned (`832f6c5`).
- RHSMode=1 x-block sparse-PC side-probe controls now use a tested
  `rhs1_xblock_policy` resolver for probe enablement, probe Krylov limits,
  switch threshold, LGMRES rescue backend/method caps, global-coupling
  keep-left threshold, and fallback-to-GMRES defaulting while the side-probe
  solve, seed preservation, side/method mutation, and candidate solve remain
  driver-owned (`90ef4ec`).
- RHSMode=1 PAS source-zero cleanup tolerance now uses one tested
  profile-response policy helper shared by active-DOF and full-system result
  finalization while source cleanup application remains driver-owned
  (`bf85368`).
- RHSMode=1 CPU SciPy rescue controls now use a tested profile-response policy
  helper for enablement, residual-ratio threshold, restart/maxiter bounds,
  strong-preconditioner preference, and method selection while threshold
  application, active-size skip metadata, host solve execution, and result
  acceptance remain driver-owned (`2b1d31d`).
- RHSMode=1 FP BiCGStab polish controls now use a tested profile-response
  policy helper for opt-in, min-size admission, maxiter bounds, and tolerance
  parsing while preconditioner choice, solve execution, and residual acceptance
  remain driver-owned (`e40c532`).
- RHSMode=1 FP L1 and global low-L projected-polish controls now use tested
  profile-response policy helpers for enablement, Krylov bounds, residual
  thresholds, lmax/size caps, and acceptance ratios while active-index
  construction, projected solves, and result acceptance remain driver-owned
  (`8192d0f`).
- RHSMode=1 FP low-L polish controls now use a tested profile-response policy
  helper for lmax, small-angular-grid default bump, block cap, restart, and
  maxiter while low-L preconditioner construction and solve execution remain
  driver-owned (`8a85885`).
- RHSMode=1 FP damped residual-polish controls now use a tested
  profile-response policy helper for min size, step count, hybrid enable,
  damping, and backtracking bounds while hybrid preconditioner construction
  and polish execution remain driver-owned (`6cbad22`).
- RHSMode=1 fast post-xblock polish restart/maxiter/tolerance controls now use
  a tested profile-response policy helper while the polish execution and
  acceptance remain in the existing handoff helper (`8746c03`).
- RHSMode=1 dense-fallback residual-ratio threshold parsing now uses a tested
  dense profile-response helper for default/invalid env handling, optional
  huge-residual limits, and trigger calculation across early, reduced, and
  full-system dense fallback gates (`7c39de7`).
- Reduced active-DOF and full-system sparse host/direct-vs-ILU factor-control
  setup now share a tested sparse-PC helper for direct intent, factor dtype,
  cache-key stamping, dense/JAX factor flags, dense cache admission, and
  explicit sparse admission while preserving driver-local matvec, pattern, and
  operator-preconditioned rescue routing (`f9c0dd7`).
- RHSMode=1 PAS preconditioner probe/default routing now uses tested
  PAS-policy helpers for env parsing, tokamak-like Schur defaulting, heavy-path
  admission, large-system collision skip, and residual-threshold decisions
  while keeping probe execution and cache mutation driver-owned (`8e9f556`).
- RHSMode=1 FP dense-fallback preconditioner probe selection now uses a tested
  dense profile-response helper, including DKES, size, explicit-env,
  dense-solve, and heavy-preconditioner guards (`4be8747`).
- RHSMode=1 dense shortcut/fallback setup now uses a tested dense
  profile-response helper for shortcut ratio parsing, PAS dense gating, backend
  fallback caps, and backend-disabled progress messages (`6c104bb`).
- RHSMode=1 reduced dense-probe enable/admission and shortcut/skip decision
  logic now uses tested dense profile-response helpers while probe execution
  and KSP replay updates remain driver-owned (`35ed3af`).
- RHSMode=1 constraintScheme=0 PETSc-compatible sparse-ILU controls and
  diagonal regularization parsing now use tested policy helpers while matrix
  assembly, ordering, factorization, and solve execution remain driver-owned
  (`29d0ca6`).
- RHSMode=1 reduced sparse-operator matvec admission now uses a tested
  side-effect-free policy helper, including implicit-solve and size rejection
  messages while driver-local operator materialization remains unchanged
  (`e900e35`).
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

Current source-size snapshot after reduced strong-kind selection extraction:

- `sfincs_jax/v3_driver.py`: `15837` lines.
- `solve_v3_full_system_linear_gmres`: `11083` lines.
- `sfincs_jax/v3_results.py`: `119` lines.
- `sfincs_jax/rhs1_ksp_diagnostics.py`: `306` lines.
- `sfincs_jax/rhs1_pas_policy.py`: `889` lines.
- `sfincs_jax/rhs1_strong_fallback.py`: `147` lines.
- `sfincs_jax/problems/profile_response/strong_preconditioning.py`: `803` lines.
- `sfincs_jax/problems/profile_response/residual.py`: `981` lines.
- `sfincs_jax/problems/profile_response/handoff.py`: `598` lines.
- `sfincs_jax/problems/profile_response/policies.py`: `3463` lines.
- `sfincs_jax/problems/profile_response/dense.py`: `701` lines.
- `sfincs_jax/problems/profile_response/linear_solve.py`: `339` lines.
- `sfincs_jax/problems/profile_response/active_projection.py`: `116` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: `14949` lines.
- `sfincs_jax/rhs1_xblock_policy.py`: `1215` lines.

Recent local validation:

- PAS policy shard after near-zero-Er PAS routing extraction:
  `36 passed in 0.38 s`.
- Solver-selection policy shard after near-zero-Er PAS routing extraction:
  `162 passed in 0.90 s`.
- RHSMode=1/profile-response shard after near-zero-Er PAS routing extraction:
  `1254 passed in 47.25 s`.
- Hygiene after near-zero-Er PAS routing extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block post-Krylov completion extraction:
  `287 passed in 2.52 s`.
- RHSMode=1/profile-response shard after x-block post-Krylov completion
  extraction: `1251 passed in 48.32 s`.
- Hygiene after x-block post-Krylov completion extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block Krylov solve-stage extraction:
  `286 passed in 2.56 s`.
- RHSMode=1/profile-response shard after x-block Krylov solve-stage
  extraction: `1250 passed in 47.34 s`.
- Hygiene after x-block Krylov solve-stage extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block Krylov-progress callback extraction:
  `284 passed in 2.46 s`.
- RHSMode=1/profile-response shard after x-block Krylov-progress callback
  extraction: `1248 passed in 47.56 s`.
- Hygiene after x-block Krylov-progress callback extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block augmented-Krylov stage extraction:
  `281 passed in 2.48 s`.
- RHSMode=1/profile-response shard after x-block augmented-Krylov stage
  extraction: `1245 passed in 47.64 s`.
- Hygiene after x-block augmented-Krylov stage extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block Krylov-control setup extraction:
  `278 passed in 2.54 s`.
- RHSMode=1/profile-response shard after x-block Krylov-control setup
  extraction: `1242 passed in 47.64 s`.
- Hygiene after x-block Krylov-control setup extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block preflight-gate extraction:
  `275 passed in 2.48 s`.
- RHSMode=1/profile-response shard after x-block preflight-gate extraction:
  `1239 passed in 47.08 s`.
- Hygiene after x-block preflight-gate extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block probe-coarse stage extraction:
  `269 passed in 2.48 s`.
- RHSMode=1/profile-response shard after x-block probe-coarse stage
  extraction: `1233 passed in 46.75 s`.
- Hygiene after x-block probe-coarse stage extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block side-probe stage extraction:
  `265 passed in 2.53 s`.
- RHSMode=1/profile-response shard after x-block side-probe stage extraction:
  `1229 passed in 47.34 s`.
- Hygiene after x-block side-probe stage extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after x-block QI residual-deflated stage extraction:
  `261 passed in 2.47 s`.
- RHSMode=1/profile-response shard after x-block QI residual-deflated stage
  extraction: `1225 passed in 46.78 s`.
- Hygiene after x-block QI residual-deflated stage extraction:
  `ruff check`, `py_compile`, `compileall`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed. Repo-size audit reported no
  reviewed files above 2 MiB.
- Sparse-PC helper shard after xblock reporting helper extraction:
  `193 passed in 1.90 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.96 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock reporting helper extraction:
  `1147 passed in 48.76 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- Strong fallback/preconditioner-build shard after full strong fallback
  ADI-control extraction: `11 passed in 0.75 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.61 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after full strong fallback ADI-control extraction:
  `1143 passed in 48.07 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- Strong fallback/preconditioner-build shard after reduced strong fallback
  dispatch extraction: `10 passed in 0.78 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.60 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after reduced strong fallback dispatch extraction:
  `1143 passed in 49.36 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- Strong policy shard after reduced PAS-Schur strong-size downgrade
  extraction: `7 passed in 0.32 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.56 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after reduced PAS-Schur strong-size downgrade extraction:
  `1143 passed in 48.09 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- PAS policy shard after force-full routing extraction:
  `33 passed in 0.32 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 32.83 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after force-full routing extraction:
  `1142 passed in 41.97 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- Strong policy/control shard after guarded/weak PAS MINRES-control
  extraction: `20 passed in 0.31 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 32.81 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after guarded/weak PAS MINRES-control extraction:
  `1140 passed in 42.05 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- Strong-control policy shard after FP-only strong-size guard extraction:
  `14 passed in 0.28 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 32.92 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP-only strong-size guard extraction:
  `1138 passed in 41.98 s`.
- Hygiene:
  `py_compile` and `ruff` passed before the broad shards.
- PAS/strong-control policy shard after collision retry and guarded PAS-TZ
  strong-retry policy extraction: `43 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 33.07 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after collision retry and guarded PAS-TZ strong-retry policy
  extraction: `1136 passed in 42.16 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- PAS policy/smoother shard after PAS Schur rescue-control extraction:
  `43 passed in 0.42 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 34.03 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after PAS Schur rescue-control extraction:
  `1134 passed in 43.30 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- PAS policy/smoother shard after adaptive smoother-control extraction:
  `40 passed in 0.44 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.60 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after PAS smoother-control extraction:
  `1131 passed in 44.49 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Strong-preconditioner policy shard after env-normalization extraction:
  `11 passed in 0.31 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 34.19 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after strong-env normalization extraction:
  `1117 passed in 44.32 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Strong-preconditioner policy shard after retry-control extraction:
  `10 passed in 0.31 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 33.99 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after strong-retry control extraction:
  `1116 passed in 44.49 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Strong-preconditioner policy shard after trigger-control extraction:
  `8 passed in 0.29 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 34.22 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after strong-trigger control extraction:
  `1114 passed in 43.97 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Profile-response linear-solve shard after small-GMRES cutoff extraction:
  `6 passed in 0.99 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.89 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after small-GMRES cutoff extraction:
  `1110 passed in 45.37 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Stage-2 policy shard after admission-control extraction:
  `15 passed in 0.33 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.85 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after Stage-2 admission-control extraction:
  `1109 passed in 45.53 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Stage-2 policy shard after retry-control extraction:
  `11 passed in 0.32 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 33.59 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after Stage-2 retry-control extraction:
  `1105 passed in 43.76 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- KSP diagnostics and Newton-Krylov diagnostics shard after shared diagnostics
  env parsing extraction:
  `19 passed in 0.58 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.49 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after shared diagnostics env parsing extraction:
  `1101 passed in 46.16 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after Krylov routing-control extraction:
  `32 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 34.54 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after Krylov routing-control extraction:
  `1093 passed in 44.50 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after BiCGStab fallback-control extraction:
  `29 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.94 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after BiCGStab fallback-control extraction:
  `1090 passed in 48.07 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- X-block policy shard after x-block side-probe/fallback control extraction:
  `64 passed in 0.41 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.27 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after x-block side-probe/fallback control extraction:
  `1087 passed in 48.97 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after PAS source-zero tolerance extraction:
  `26 passed in 0.35 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.76 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after PAS source-zero tolerance extraction:
  `1083 passed in 48.47 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after CPU SciPy rescue-control extraction:
  `25 passed in 0.35 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.86 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after CPU SciPy rescue-control extraction:
  `1082 passed in 48.54 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after FP BiCGStab polish-control extraction:
  `23 passed in 0.35 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.51 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP BiCGStab polish-control extraction:
  `1080 passed in 48.96 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after FP L1/global low-L polish-control extraction:
  `21 passed in 0.36 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 38.35 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP L1/global low-L polish-control extraction:
  `1078 passed in 48.83 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Post-xblock policy shard after FP low-L polish-control extraction:
  `17 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.71 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP low-L polish-control extraction:
  `1074 passed in 47.45 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Post-xblock policy shard after FP residual-polish control extraction:
  `15 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.13 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP residual-polish control extraction:
  `1072 passed in 48.77 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Post-xblock policy shard after fast polish-control extraction:
  `13 passed in 0.34 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.12 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after fast polish-control extraction:
  `1070 passed in 48.62 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Dense profile-response shard after dense fallback-threshold extraction:
  `19 passed in 1.10 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.83 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after dense fallback-threshold extraction:
  `1068 passed in 50.11 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Sparse-PC helper shard after sparse factor-control extraction:
  `189 passed in 2.01 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.02 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse factor-control extraction:
  `1065 passed in 49.69 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- PAS policy shard after PAS preconditioner probe extraction:
  `26 passed in 0.35 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.90 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after PAS preconditioner probe extraction:
  `1063 passed in 49.40 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Dense profile-response shard after FP preconditioner probe extraction:
  `16 passed in 1.11 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.51 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after FP preconditioner probe extraction:
  `1056 passed in 49.76 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Dense profile-response shard after dense shortcut setup extraction:
  `13 passed in 1.11 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 38.31 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after dense shortcut setup extraction:
  `1053 passed in 49.63 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- Dense profile-response shard after dense-probe policy extraction:
  `10 passed in 1.12 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.30 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after dense-probe policy extraction:
  `1050 passed in 49.59 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
- ConstraintScheme=0 policy/docstring shard after PETSc-compat config
  extraction:
  `13 passed in 0.63 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.08 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after PETSc-compat config extraction:
  `1046 passed in 49.75 s`.
- Hygiene:
  `ruff`, `compileall`, `git diff --check`, and `scripts/check_repo_size.py`
  passed.
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
- Sparse-PC helper shard after xblock sparse-PC reporting/fallback-admission
  extraction:
  `198 passed in 1.97 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 37.30 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock sparse-PC reporting/fallback-admission extraction:
  `1152 passed in 48.65 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Profile-response diagnostics shard after cached-QI correction-basis setup
  extraction:
  `14 passed in 0.72 s`.
- Combined sparse-PC/diagnostics shard:
  `212 passed in 1.74 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after cached-QI correction-basis setup extraction:
  `1155 passed in 47.76 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock sparse-PC work-estimate extraction:
  `200 passed in 1.91 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.13 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock sparse-PC work-estimate extraction:
  `1157 passed in 48.36 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock completion-emitter extraction:
  `204 passed in 2.15 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock completion-emitter extraction:
  `1161 passed in 47.50 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock post-solve correction orchestration
  extraction:
  `206 passed in 1.94 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.19 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock post-solve correction orchestration extraction:
  `1163 passed in 48.81 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock final payload state consolidation:
  `206 passed in 1.92 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.25 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock final payload state consolidation:
  `1163 passed in 48.15 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock GMRES fallback execution extraction:
  `208 passed in 1.95 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.92 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock GMRES fallback execution extraction:
  `1165 passed in 49.12 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock device-Krylov result unpacking
  extraction:
  `210 passed in 1.96 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.36 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock device-Krylov result unpacking extraction:
  `1167 passed in 48.59 s`.
- Hygiene:
  `py_compile`, `ruff`, `compileall`, `git diff --check`, and
  `scripts/check_repo_size.py` passed.
- Sparse-PC helper shard after xblock first-attempt Krylov dispatch
  extraction:
  `213 passed in 2.08 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.83 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock first-attempt Krylov dispatch extraction:
  `1170 passed in 48.75 s`.
- Sparse-PC helper shard after xblock Krylov solve-space/equilibration
  extraction:
  `215 passed in 2.17 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.64 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock Krylov solve-space/equilibration extraction:
  `1172 passed in 49.25 s`.
- Sparse-PC helper shard after xblock augmented-QI Krylov basis preparation
  extraction:
  `220 passed in 2.16 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.61 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock augmented-QI Krylov basis preparation extraction:
  `1177 passed in 48.58 s`.
- Sparse-PC helper shard after xblock sparse-PC progress-message formatting
  extraction:
  `222 passed in 2.23 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.17 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock sparse-PC progress-message formatting extraction:
  `1179 passed in 48.56 s`.
- Sparse-PC helper shard after xblock solve-state/fallback-state extraction:
  `224 passed in 2.24 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.40 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock solve-state/fallback-state extraction:
  `1181 passed in 48.58 s`.
- Sparse-PC helper shard after xblock explicit completion-emission extraction:
  `226 passed in 1.79 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.10 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock explicit completion-emission extraction:
  `1183 passed in 48.49 s`.
- Sparse-PC helper shard after xblock explicit final-payload context
  extraction:
  `227 passed in 2.22 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.44 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after xblock explicit final-payload context extraction:
  `1184 passed in 49.00 s`.
- Sparse-PC helper shard after fortran-reduced xblock explicit final-payload
  context extraction:
  `228 passed in 2.22 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 38.11 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after fortran-reduced xblock explicit final-payload context
  extraction:
  `1185 passed in 49.90 s`.
- Sparse-PC helper shard after generic sparse-PC explicit finalization-context
  extraction:
  `229 passed in 2.19 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.38 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC explicit finalization-context
  extraction:
  `1186 passed in 48.03 s`.
- Profile-response diagnostics shard after sparse-rescue explicit tail
  metadata context extraction:
  `14 passed in 0.71 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 35.98 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after sparse-rescue explicit tail metadata context extraction:
  `1186 passed in 47.90 s`.
- Sparse-PC helper shard after fortran-reduced xblock explicit metadata-state
  handoff:
  `229 passed in 1.79 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 34.49 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after fortran-reduced xblock explicit metadata-state handoff:
  `1186 passed in 47.37 s`.
- Profile-response diagnostics/sparse-PC shard after direct-tail metadata
  context extraction:
  `243 passed in 1.96 s`.
- Sparse-host/minimum-norm/direct-tail driver shard:
  `32 passed, 100 deselected in 36.88 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after direct-tail metadata context extraction:
  `1186 passed in 48.83 s`.
- Fortran-reduced xblock explicit metadata-state CI fix:
  `python -m py_compile sfincs_jax/v3_driver.py`,
  `ruff check sfincs_jax/v3_driver.py`, and the two xblock backend tests passed
  after reading solve time from `SparsePCGMRESResult` and restoring the
  required moment-Schur probe fields in the explicit final metadata state.
- Profile-response diagnostics/sparse-PC shard after that xblock metadata CI
  fix: `243 passed in 1.99 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after that xblock
  metadata CI fix: `36 passed, 96 deselected in 37.66 s`.
- Hygiene after that xblock metadata CI fix:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Sparse-PC whitelist finalization-state helper focused tests:
  `2 passed in 1.02 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  finalization whitelist extraction: `244 passed in 1.91 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC finalization whitelist extraction:
  `36 passed, 96 deselected in 36.23 s`.
- Hygiene after generic sparse-PC finalization whitelist extraction:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- X-block whitelist final metadata-state helper focused tests:
  `2 passed in 1.05 s`.
- Profile-response diagnostics/sparse-PC shard after x-block final metadata
  whitelist extraction: `245 passed in 1.97 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after x-block final
  metadata whitelist extraction:
  `36 passed, 96 deselected in 36.51 s`.
- Hygiene after x-block final metadata whitelist extraction:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- X-block final metadata whitelist completeness fix:
  `tests/test_rhs1_device_operator.py::test_xblock_side_probe_switch_keeps_physical_left_probe_seed_for_right_pc`
  passed after excluding payload/post-correction-computed fields from the
  driver-scope whitelist and adding the nested driver-owned coarse/QI/device
  diagnostics keys.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after x-block final metadata whitelist completeness fix:
  `1188 passed in 48.40 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after x-block final
  metadata whitelist completeness fix:
  `36 passed, 96 deselected in 38.24 s`.
- Hygiene after x-block final metadata whitelist completeness fix:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC finalization direct-tail metadata reduction focused tests:
  `2 passed in 1.02 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  finalization direct-tail metadata reduction: `245 passed in 1.93 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC finalization direct-tail metadata reduction:
  `36 passed, 96 deselected in 36.20 s`.
- Hygiene after generic sparse-PC finalization direct-tail metadata reduction:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC factor-preflight metadata context focused tests:
  `3 passed in 1.04 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  factor-preflight metadata context extraction: `246 passed in 1.93 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC factor-preflight metadata context extraction:
  `36 passed, 96 deselected in 36.39 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC factor-preflight metadata context
  extraction: `1189 passed in 47.94 s`.
- Hygiene after generic sparse-PC factor-preflight metadata context extraction:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC sparse-pattern metadata context focused tests:
  `4 passed in 1.07 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  sparse-pattern metadata context extraction: `247 passed in 1.97 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC sparse-pattern metadata context extraction:
  `36 passed, 96 deselected in 37.62 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC sparse-pattern metadata context
  extraction: `1190 passed in 48.57 s`.
- Hygiene after generic sparse-PC sparse-pattern metadata context extraction:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- X-block nested metadata precompute focused tests:
  `2 passed in 1.07 s`.
- Profile-response diagnostics/sparse-PC shard after x-block nested metadata
  precompute: `247 passed in 1.98 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after x-block
  nested metadata precompute: `36 passed, 96 deselected in 38.48 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after x-block nested metadata precompute:
  `1190 passed in 48.36 s`.
- Hygiene after x-block nested metadata precompute:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC post-MinRes explicit-finalization focused tests:
  `3 passed in 1.12 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  post-MinRes explicit finalization: `248 passed in 1.97 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC post-MinRes explicit finalization:
  `36 passed, 96 deselected in 38.32 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC post-MinRes explicit finalization:
  `1191 passed in 48.64 s`.
- Hygiene after generic sparse-PC post-MinRes explicit finalization:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC dtype-retry explicit-finalization focused tests:
  `3 passed in 1.09 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC
  dtype-retry explicit finalization: `248 passed in 1.98 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC dtype-retry explicit finalization:
  `36 passed, 96 deselected in 38.26 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC dtype-retry explicit finalization:
  `1191 passed in 48.84 s`.
- Hygiene after generic sparse-PC dtype-retry explicit finalization:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC explicit elapsed/completion focused tests:
  `3 passed in 1.07 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC explicit
  elapsed/completion cleanup: `248 passed in 1.97 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC explicit elapsed/completion cleanup:
  `36 passed, 96 deselected in 36.88 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC explicit elapsed/completion cleanup:
  `1191 passed in 48.44 s`.
- Hygiene after generic sparse-PC explicit elapsed/completion cleanup:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- Generic sparse-PC static metadata precompute focused tests:
  `3 passed in 1.09 s` plus two final-payload tests `2 passed in 0.68 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC static
  metadata precompute: `248 passed in 2.02 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC static metadata precompute:
  `36 passed, 96 deselected in 41.34 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC static metadata precompute:
  `1191 passed in 49.01 s`.
- Hygiene after generic sparse-PC static metadata precompute:
  `python -m compileall -q sfincs_jax`, `git diff --check`, and
  `python scripts/check_repo_size.py` passed.
- X-block typed nested/final metadata focused equivalence tests:
  `5 passed in 0.63 s`.
- Profile-response diagnostics/sparse-PC shard after x-block typed nested/final
  metadata: `250 passed in 2.01 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after x-block typed
  nested/final metadata: `36 passed, 96 deselected in 39.45 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after x-block typed nested/final metadata:
  `1193 passed in 48.35 s`.
- Generic sparse-PC typed finalization-state focused test:
  `1 passed in 1.01 s`.
- Profile-response diagnostics/sparse-PC shard after generic sparse-PC typed
  finalization-state handoff: `250 passed in 1.97 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after generic
  sparse-PC typed finalization-state handoff:
  `36 passed, 96 deselected in 37.04 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after generic sparse-PC typed finalization-state handoff:
  `1193 passed in 47.58 s`.
- RHSMode=1 PAS/Schwarz monkeypatch compatibility tests after preconditioner
  wrapper alias cleanup: `31 passed in 9.30 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after preconditioner wrapper alias cleanup:
  `1193 passed in 49.22 s`.
- Xblock/sparse-host/minimum-norm/direct-tail driver shard after preconditioner
  wrapper alias cleanup: `36 passed, 96 deselected in 40.57 s`.
- Transport/preconditioner dispatch shard after preconditioner wrapper alias
  cleanup: `554 passed in 29.70 s`.
- Policy, sparse-helper, transport, and distributed-GMRES focused tests after
  policy/refinement/parallel alias cleanup: `102 passed in 13.06 s`.
- Transport parallel policy/runtime shard after policy/refinement/parallel
  alias cleanup: `329 passed in 52.75 s`.
- RHSMode=1 PAS/Schwarz monkeypatch compatibility tests after policy alias
  cleanup: `31 passed in 10.18 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after policy/refinement/parallel alias cleanup:
  `1193 passed in 50.24 s`.
- Transport/preconditioner dispatch shard after policy/refinement/parallel
  alias cleanup: `554 passed in 29.90 s`.
- Profile-response setup unit tests after initial-route extraction:
  `13 passed in 0.32 s`.
- Structured CSR and auto-host routing shard after initial-route extraction:
  `109 passed in 23.10 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after initial-route extraction: `1195 passed in 48.86 s`.
- Profile-response setup and active-DOF unit tests after active-problem setup
  extraction: `19 passed in 0.35 s`.
- Structured CSR and auto-host routing shard after active-problem setup
  extraction: `109 passed in 23.22 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after active-problem setup extraction: `1198 passed in 48.29 s`.
- Profile-response sparse-PC policy tests after x-block branch setup
  extraction: `235 passed in 1.87 s`.
- Driver sparse-helper/solve-policy compatibility shard after x-block branch
  setup extraction: `34 passed in 0.94 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after x-block branch setup extraction: `1199 passed in 49.26 s`.
- Profile-response sparse-PC helper tests after local x-block preconditioner
  build extraction: `237 passed in 2.29 s`.
- Driver sparse-helper/solve-policy compatibility shard after local x-block
  preconditioner build extraction: `34 passed in 0.95 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after local x-block preconditioner build extraction:
  `1201 passed in 48.02 s`.
- Profile-response sparse-PC helper tests after assembled x-block operator
  orchestration extraction: `240 passed in 2.38 s`.
- Driver sparse-helper/solve-policy compatibility shard after assembled
  x-block operator orchestration extraction: `34 passed in 0.95 s`.
- Broad profile-response/RHSMode=1 policy, setup, diagnostics, solver, and
  helper sweep after assembled x-block operator orchestration extraction:
  `1204 passed in 49.19 s`.
- Older focused and broad validation checkpoints are intentionally omitted from
  this active plan; they remain available in git history.

Known CI issue fixed by this rewrite:

- `tests/test_repo_size_policy.py::test_tracked_large_files_are_reviewed`
  failed because `plan.md` reached `2.01 MiB`.

## Active Work Lanes

### 1. `v3_driver.py` Architecture Refactor

Completion estimate: 90%.

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
- Generic sparse-PC finalization-state handoff now uses
  `SparsePCGMRESFinalizationStateContext` directly from the driver instead of
  a raw mapping wrapper.
- X-block sparse-PC nested diagnostics and final metadata state now use typed
  contexts directly from the driver; production x-block finalization no longer
  depends on a `locals()` frame copy.
- Pure forwarding preconditioner compatibility wrappers in `v3_driver.py` are
  now patchable aliases to their domain implementations, removing about 575
  net driver lines while preserving existing monkeypatch-based tests.
- Pure forwarding policy, host-refinement, submatrix setup, and transport
  parallel compatibility wrappers are now patchable aliases to domain helpers,
  removing another driver boilerplate tranche without changing backend-injected
  or callback-adapting wrappers.
- RHSMode=1 initial solve-route setup now lives in the profile-response setup
  module with unit coverage for solve-method classification, structured-CSR
  auto admission, force-Krylov rejection, and multi-device sharding metadata.
- RHSMode=1 recycled-Krylov basis filtering, reduced-pitch-mode shape
  detection, DKES adjustment, active-DOF admission, preconditioner-option
  parsing, and active-map construction are consolidated in a typed
  active-problem setup helper with focused tests.
- RHSMode=1 x-block sparse-PC branch setup now composes local factor, side
  policy, and QI-device operator-reuse decisions in a single typed
  sparse-PC-domain helper while keeping driver monkeypatch compatibility
  aliases during migration.
- RHSMode=1 local x-block preconditioner construction and timing are now a
  sparse-PC-domain helper with tested identity-skip and delegated factor-build
  paths.
- RHSMode=1 assembled x-block operator materialization, validation,
  equilibration, optional device CSR setup, matvec replacement, and fail-closed
  metadata now live in a sparse-PC-domain helper with focused success,
  disabled, and rejection tests.
- RHSMode=1 x-block constraint moment-Schur, two-level, and global-coupling
  stage orchestration now lives in tested sparse-PC-domain helpers. The driver
  still resolves high-level policies and passes explicit operator callbacks,
  but build/probe/failure metadata and preconditioner replacement are no
  longer maintained inline.
- RHSMode=1 x-block QI coarse-seed basis construction, guarded residual
  correction, acceptance/rejection messages, failure reason, and diagnostics
  now use a tested sparse-PC-domain stage helper. The driver retains only the
  shared-basis object and scalar metadata needed by later QI Galerkin/two-level
  setup.
- RHSMode=1 x-block QI Galerkin preconditioner setup now uses a tested
  sparse-PC-domain stage helper for shared-basis reuse/build, Galerkin
  preconditioner construction, true-residual probe selection, preconditioner
  installation, stats, and failure metadata.
- RHSMode=1 x-block QI two-level preconditioner setup now uses a tested
  sparse-PC-domain stage helper for shared-basis reuse/build, smoothed-load
  basis construction, residual augmentation, true-residual damping selection,
  seed/preconditioner installation, stats, and failure metadata.
- RHSMode=1 x-block QI device preconditioner probe metadata now uses a typed
  sparse-PC-domain helper for probe histories, augmented-seed fields,
  enrichment/multilevel controls, local-smoother metadata, Krylov installation
  flags, and residual-correction metadata. This also restored the late-bound
  driver `_matvec_submatrix` compatibility shim required by CI monkeypatch
  tests after the submatrix helper extraction.
- RHSMode=1 x-block QI device preconditioner setup now builds its geometry
  metadata and `RHS1QIDevicePreconditionerConfig` through a tested
  sparse-PC-domain setup helper. The driver keeps compatibility aliases for
  legacy tests/debug scripts, but no longer owns the device config literal or
  QI tail-block geometry assembly at the production call site.
- RHSMode=1 x-block QI residual-deflated preconditioner controls now use a
  tested sparse-PC-domain policy resolver for Krylov/rank/damping/cycle
  controls, seed-solver normalization, composition, raw-residual admission,
  and extra global-load directions.
- RHSMode=1 x-block QI residual-deflated preconditioner build/probe/install
  now uses a tested sparse-PC-domain stage helper. The driver keeps only the
  policy resolution, explicit callback wiring, scalar diagnostics handoff, and
  timing aggregation; acceptance/rejection/failure behavior is covered by
  focused sparse-PC tests.
- RHSMode=1 x-block precondition-side probe now uses a tested
  sparse-PC-domain stage helper. The driver resolves the side-probe policy and
  keeps downstream diagnostic variable names, while probe execution, seed
  preservation, side switching, LGMRES rescue, failure handling, and emitted
  action messages are owned by the helper.
- RHSMode=1 x-block probe-coarse seed correction now uses a tested
  sparse-PC-domain stage helper. The driver still owns the physics-aware
  coarse-direction builder closure, while seed initialization, projected
  correction, accept/reject/failure diagnostics, and scalar metadata handoff
  are owned by the helper.
- RHSMode=1 x-block preflight residual gate now uses a tested
  sparse-PC-domain helper. The driver still resolves the environment controls,
  while seed residual evaluation, improvement/target acceptance, required-gate
  errors, and warning emission are owned by the helper.
- RHSMode=1 x-block Krylov runtime controls now use a tested
  sparse-PC-domain setup helper. The helper resolves cycle synchronization,
  TFQMR replacement, device-JIT controls, QI augmented-Krylov controls, and
  setup/user-facing emissions; the driver keeps only the resulting scalar
  handoff to the first Krylov attempt and final metadata.
- RHSMode=1 x-block QI augmented-Krylov solve setup now uses a tested
  sparse-PC-domain stage helper. Basis construction still uses the existing
  solve-space helper, while request gating, metadata updates, seed-used handoff,
  and acceptance/rejection emissions are owned by the stage helper.
- RHSMode=1 x-block Krylov host/device progress callbacks now use a tested
  sparse-PC-domain callback builder. The driver owns only the emitter, elapsed
  timer, and progress stride, while device-cycle and host-iteration progress
  formatting and no-op behavior are covered by focused tests.
- RHSMode=1 x-block Krylov first-attempt plus optional GMRES fallback solve
  execution now uses a tested sparse-PC-domain stage helper. Candidate
  true-residual state, optional fallback state, fallback progress/emission, and
  final reported Krylov counters are built outside the driver while preserving
  existing metadata scalar names.
- RHSMode=1 x-block post-Krylov correction and completion emission now use a
  tested sparse-PC-domain helper. Post-minres, post-coarse, and
  post-residual-equation diagnostics remain available through the same
  correction result object used by final metadata.
- RHSMode=1 PAS near-zero-Er small-system default routing now uses a tested
  PAS policy helper for PAS-lite/PAS-hybrid/xmg selection, eliminating three
  duplicate env-parsing branches from the driver.
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
- RHSMode=1 constraintScheme=0 PETSc-compatible sparse-ILU controls and
  diagonal regularization parsing now use tested profile-response policy
  helpers while preserving driver-local SciPy sparse execution.
- RHSMode=1 reduced dense-probe global enable, admission guards, shortcut
  acceptance, seed decision, and skip-message formatting now use tested dense
  profile-response helpers.
- RHSMode=1 dense shortcut ratio, PAS dense fallback gate, backend dense caps,
  and backend-disabled progress messaging now use a tested dense
  profile-response setup helper.
- RHSMode=1 FP dense-probe preconditioner-kind downgrades to collision now use
  a tested dense profile-response policy helper while preserving driver-local
  preconditioner construction.
- RHSMode=1 PAS Schur downgrade decisions now use a tested strong-fallback
  policy helper; the driver still owns the actual preconditioner construction
  and replay ordering.
- RHSMode=1 reduced/full strong fallback dispatch now uses one tested helper
  for ADI seed retries, ADI combo retries, and fallback replay metadata.
- RHSMode=1 full strong-ADI combo controls now use a tested policy helper
  instead of in-line environment parsing in the driver.
- X-block sparse-PC Krylov work reporting and physical-space true-residual
  measurement now use tested profile-response helpers.
- X-block sparse-PC GMRES fallback admission now uses a tested
  profile-response helper; the environment policy and retry execution remain
  behavior-compatible.
- X-block post-residual-equation cached QI-basis setup now uses a tested
  solver-diagnostics helper, keeping the driver focused on applying the
  correction rather than preparing diagnostic/cache payloads.
- X-block sparse-PC solver-kind labels and Krylov work-memory estimates now
  use a tested profile-response helper, and `v3_driver.py` no longer imports
  low-level Krylov memory estimators directly for xblock metadata.
- X-block sparse-PC completion progress emission now uses a tested
  state-based profile-response helper while preserving the user-facing message
  format.
- X-block sparse-PC post-residual-equation, post-minres, and post-coarse
  correction orchestration now uses one tested profile-response helper with a
  compatibility state map for the existing final metadata payload.
- X-block sparse-PC final payload assembly now owns work-estimate metadata and
  post-correction state merging, leaving the driver to pass the current solve
  state plus the correction result object.
- X-block sparse-PC non-GMRES to GMRES fallback execution now uses a tested
  profile-response helper with explicit initial-guess, GMRES, and
  physical-residual callbacks.
- X-block device Krylov result unpacking now uses a tested profile-response
  helper shared by `gmres_jax`, `fgmres_jax`, `bicgstab_jax`, and
  `tfqmr_jax` branches.
- X-block sparse-PC first-attempt Krylov dispatch now uses a tested
  profile-response helper for host SciPy, JAX FGMRES/GMRES, JAX BiCGStab,
  JAX TFQMR, GCROT, and fallback host GMRES/BiCGStab methods while the driver
  still owns operator preparation, true-residual recomputation, and metadata.
- X-block sparse-PC Krylov solve-space preparation now uses a tested
  profile-response helper for row/column equilibration, scaled RHS,
  scaled initial guesses, preconditioner scaling, and physical-solution
  recovery.
- X-block sparse-PC augmented-QI Krylov basis preparation now uses a tested
  profile-response helper for seed/state admission, row/column scaling,
  left-preconditioned operator-action scaling, rank reporting, and rejection
  reasons.
- X-block sparse-PC device-cycle and host-Krylov progress message formatting
  now uses tested profile-response helpers, preserving elapsed-time and
  residual formatting.
- X-block sparse-PC first-attempt physical solve-state construction and
  optional GMRES fallback-state reporting now use tested profile-response
  helpers while true-residual callbacks remain explicit at the driver call
  site.
- X-block sparse-PC completion emission now uses a tested explicit context
  instead of a driver `locals()` handoff; the legacy state-wrapper remains
  covered for compatibility.
- X-block sparse-PC final payload assembly now uses a tested explicit context
  instead of a driver `locals()` handoff; the legacy state-wrapper delegates
  to the explicit path for compatibility.
- Fortran-reduced x-block sparse-PC final payload assembly now uses a tested
  explicit context for convergence-gate inputs, with the broad diagnostics
  mapping named separately and the legacy state-wrapper kept for compatibility.
- Generic sparse-PC dtype-retry, post-minres, completion, and final payload
  assembly now uses a tested explicit context for current solve result and
  factor state, with the broad diagnostics mapping named separately and the
  legacy state-wrapper kept for compatibility.
- RHSMode=1 sparse-rescue tail metadata now uses tested explicit diagnostics
  contexts instead of a driver `locals()` handoff; the legacy mapping wrapper
  remains covered for compatibility.
- Fortran-reduced x-block final payload assembly now receives an explicit
  final metadata-state mapping from the driver instead of the full driver
  frame. The generic direct-tail metadata handoff is intentionally deferred
  until direct-tail diagnostics are typed; an inline all-key dictionary would
  make the driver less maintainable.
- Generic sparse-PC direct-tail metadata now has a tested explicit context
  path with named suffix groups. The legacy mapping wrapper remains compatible,
  and generic sparse-PC result metadata can now consume precomputed direct-tail
  metadata without carrying every raw direct-tail driver key.
- Generic sparse-PC finalization now receives a whitelisted driver-state copy
  from a tested `profile_response.sparse_pc` helper instead of handing the
  whole frame to `SparsePCGMRESFinalizationContext`. This is a transitional
  compatibility boundary: the helper makes the required keys explicit while the
  next step replaces key groups with typed direct-tail, preflight, post-MinRes,
  and pattern-summary contexts.
- Generic sparse-PC finalization now precomputes direct-tail metadata before
  the finalizer state is built, so raw direct-tail setup/rescue keys no longer
  propagate through `SparsePCGMRESFinalizationContext`.
- Generic sparse-PC factor-preflight metadata now uses a typed diagnostics
  context and is precomputed before finalization, so raw preflight probe fields
  no longer propagate through `SparsePCGMRESFinalizationContext`.
- Generic sparse-PC sparse-pattern metadata now uses a typed diagnostics
  context and is precomputed before finalization, so raw pattern summary,
  scope, and build-time fields no longer propagate through
  `SparsePCGMRESFinalizationContext`.
- X-block sparse-PC final metadata now receives typed grouped diagnostics
  contexts from the driver instead of a whitelisted local-scope copy. The
  compatibility wrapper remains covered for missing-key audits, but production
  finalization now avoids both full-frame and filtered-frame handoffs.
- X-block sparse-PC final metadata now precomputes assembled-operator,
  coarse-correction, QI seed/device/deflated, and side-probe metadata groups
  before final payload construction. The copied final metadata state is down
  from 219 raw driver keys to 75 copied scalar/source keys plus 6 compact
  metadata dictionaries; the 219-key raw scope inventory remains available for
  diagnostics derivation and missing-key audits.
- Generic sparse-PC finalization now passes post-MinRes dependencies through a
  typed `SparsePCPostMinresFinalizationContext` instead of the metadata map.
  The generic copied finalization state is down from 48 keys to 38 keys while
  dtype retry remains mapping-backed for the next explicit-context tranche.
- Generic sparse-PC finalization now passes factor-dtype retry dependencies
  through `SparsePCFactorDtypeRetryFinalizationContext`, removing factor
  matvec, pattern, RHS dtype, retry seed, and PAS/tokamak retry flags from the
  metadata map. The generic copied finalization state is down to 32 keys.
- Generic sparse-PC finalization now reports elapsed time and completion from
  explicit finalization contexts in the typed path, so `emit` and
  `sparse_timer` no longer propagate through the generic metadata map. The
  generic copied finalization state is down to 30 keys.
- Generic sparse-PC finalization now precomputes static result metadata
  before finalization. Backend labels, preconditioner/factorization labels,
  factor defaults, active-size metadata, Fortran-reduced settings, and full
  size no longer propagate as raw finalizer state. The generic copied
  finalization state is down to 5 dynamic convergence/reporting keys, with a
  30-key raw scope inventory kept only for static metadata derivation and
  missing-key audits.
- Generic sparse-PC finalization now builds direct-tail metadata from
  semantic policy/result contexts in `profile_response.sparse_pc`; the driver
  passes grouped policies and runtime outcomes instead of every historical
  `direct_tail_*` report key.
- The RHSMode=1 generic sparse-PC and x-block sparse-PC finalizers no longer
  have production driver `locals()` handoffs. X-block nested diagnostics are
  grouped into typed assembled-operator, coarse-correction, QI seed/device/
  deflated, and side-probe contexts before final payload construction.

Next steps:

- Continue moving remaining generic sparse-PC result/diagnostic seams into
  cohesive `profile_response` helpers only where the replacement context can
  stay explicit and tested.
- Continue reducing `v3_driver.py` surface area by moving cohesive solver
  policy/result seams into `profile_response` helpers, but only when the new
  boundary is explicit, typed, and smaller than the code it replaces.
- Continue replacing residual sparse-PC compatibility wrappers with explicit
  contexts only where the new boundary reduces driver complexity.
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
