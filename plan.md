# SFINCS_JAX Refactor And Release-Readiness Plan

Last updated: 2026-06-21 (America/Chicago)

Active implementation branch: `refactor/rhs1-full-assembly-preconditioners`

Intended review PR: #8, `refactor/v3-driver-architecture`

PR state: draft. The active implementation branch has been pushed to the PR
branch at the latest clean commit. Do not open additional refactor PRs; keep PR
#8 as the single review surface until this plan reaches the review-ready
boundary.

## One-Sentence Plan

Make `sfincs_jax` a small, domain-organized, research-grade neoclassical
transport code with parity against SFINCS Fortran v3 where models overlap,
simple input-file CLI defaults, explicit differentiable Python lanes, fast and
memory-bounded CPU/GPU execution, a manageable number of well-named files, and
README/docs/tests/benchmarks that clearly separate production claims from
deferred research lanes.

## Current Audit Snapshot

### Done

- Major RHSMode=1 preconditioner families now have domain owners:
  - full-CSR Schur preconditioners,
  - Fortran-reduced symbolic sparse factors,
  - low-`ell` x-block Schur preconditioners,
  - active-projected x-block / overlap-Schwarz preconditioners,
  - active sparse-factor preconditioners.
- Flat output file-format helpers moved to `sfincs_jax.outputs.formats`.
- Nonlinear Phi1 Newton-Krylov profile-response solve logic moved to
  `sfincs_jax.problems.profile_response.phi1_newton`.
- Transport-parallel runtime glue moved out of `v3_driver.py` into
  `sfincs_jax.problems.transport_matrix.parallel`, and the active transport
  DOF index helper moved into `sfincs_jax.problems.transport_matrix.active_dense`
  in commit `eeb2a85`.
- Final RHSMode=1/profile-response linear-solve handoff moved into
  `sfincs_jax.problems.profile_response.finalization` in commit `e1c6fa4`.
- Initial operator/RHS problem materialization moved into
  `sfincs_jax.problems.profile_response.setup` in commit `611e283`.
- QI-device admission/build/probe/install logic moved out of `v3_driver.py`
  into `sfincs_jax.problems.profile_response.sparse.qi`; `v3_driver.py` now
  injects solve-local operators and compatibility probe aliases for the tested
  fail-closed QI research lane.
- Stale private `v3_driver.py` QI policy aliases are being deleted when their
  canonical owner is now `sfincs_jax.problems.profile_response.policies`; tests
  now enforce that ownership boundary instead of preserving aliases only for
  historical convenience.
- Current next tranche is one more cohesive `v3_driver.py` stage extraction or,
  if the next driver seam would be wrapper-only, the first `io.py` output-schema
  split.
- The README and docs currently state the public claim boundary: the documented
  release suite is CPU/GPU parity-clean, while production-resolution QI, true
  device-QI, lower-memory native factor replacement, full-grid QA/QH RHSMode=1,
  and single-case multi-GPU scaling remain fail-closed research lanes.

### Local Validation From This Audit

- Focused transport/refactor tests pass:
  `78 passed in 15.10s`.
- Focused profile-response finalization tests pass:
  `21 passed in 1.05s`.
- Focused profile-response setup/finalization tests pass:
  `42 passed in 1.15s`.
- Focused QI baseline tests pass before extraction:
  `549 passed in 35.17s`.
- Extracted QI-device stage, QI sparse re-export, and compatibility-sensitive
  v3 sparse-pattern tests pass:
  `455 passed in 125.61s`.
- Full bounded QI campaign passes after extraction:
  `550 passed in 34.31s`.
- Stale QI driver-alias tests pass after canonical policy ownership cleanup:
  `4 passed in 1.94s`.
- The previously failing CI coverage shard 1 reproduces locally as passing
  after the QI alias cleanup and deterministic x64 setup for precision gates:
  `814 passed, 39 skipped in 55.97s`.
- Optional Equinox/JAXopt, optional Lineax implicit-solve, collision-stencil,
  collision-physics, and sparse-helper precision gates now set x64 before JAX
  array APIs are imported; the affected focused group passes:
  `43 passed in 6.73s`.
- `ruff` and `py_compile` pass on touched transport-parallel, finalization, and
  setup/QI files.
- PR #8 remains draft. Check CI after the next meaningful push rather than
  polling continuously during local refactor tranches.

### Current Code Shape

- `sfincs_jax/v3_driver.py`: about 13.4k lines, still the largest orchestration
  and compatibility surface.
- `sfincs_jax/rhs1_full_assembly.py`: about 7.9k lines, now mostly RHSMode=1
  exact/active CSR assembly, admission, dispatch, and compatibility.
- `sfincs_jax/io.py`: about 5.5k lines, still owns too much output schema and
  diagnostics materialization.
- Package size: about 289 Python files and 160k package lines.
- Largest remaining package clusters:
  `problems/transport_matrix`, `problems/profile_response`,
  `solvers/preconditioners`, plus many historical top-level compatibility
  modules.

### Current Documentation Shape

- `README.md` is user-facing and should stay focused on install, one-command
  usage, output/plotting, current benchmark figures, and short claim-scope
  notes.
- `docs/source_map.rst` is the equation-to-source map and must be updated after
  every ownership move.
- `docs/testing.rst` is the validation-tier map and must distinguish normal CI,
  release, manual GPU, and research tiers.
- `docs/research_lanes.rst` is the correct home for fail-closed algorithmic
  research evidence.
- `docs/development_roadmap.rst` is a public stable roadmap; this `plan.md` is
  the active branch checklist.

## Goals

1. **Small, understandable code**
   Keep a limited set of domain packages with names tied to physics and
   numerical responsibility. Reduce monoliths, delete redundant wrappers, and
   avoid adding more flat `rhs1_*`, `transport_*`, or `v3_*` implementation
   files.

2. **Simple user workflow**
   A typical user should run
   `sfincs_jax input.namelist --wout-path wout.nc --out sfincsOutput.h5` and
   get a robust solve, clear progress, phase timing, output metadata, and
   plots without knowing solver internals.

3. **Fortran v3 parity**
   Where the same equations and normalizations are solved, SFINCS Fortran v3
   remains the comparison anchor. Shared outputs, residual metadata, runtime,
   memory, and solver path must be compared from matched inputs and resolutions.

4. **Explicit differentiability**
   Python users must be able to select JAX-native differentiable solve lanes
   for sensitivity analysis, inverse design, UQ, and optimization. Host-only
   shortcuts are allowed only in CLI / `differentiable=False` lanes and must be
   recorded in metadata.

5. **Fast CPU/GPU execution**
   Defaults should minimize runtime and memory on CPU and GPU while failing
   closed through true-residual gates. Adaptive `auto` choices must record a
   branch certificate: selected method, rejected candidates, residual margins,
   backend, memory estimate, and warnings near branch boundaries.

6. **Research-grade tests and docs**
   Coverage must rise through extracted-module tests, numerical identities,
   physics gates, and regression artifacts, not slow full-solve smoke tests.
   Docs must clearly label production quality, reduced-grid evidence, and
   deferred research.

## Technical Open Lanes

These are not refactor PR blockers unless a code change touches their public
claim. They must stay documented, fail-closed, and gated.

1. **True device-QI and production-resolution QI**
   Current status: bounded CPU/GPU evidence exists, but the hard seed remains
   above production write tolerance. Keep non-autodiff host fallback explicit
   and do not hide it in differentiable paths.

2. **Full-grid QA/QH RHSMode=1 production convergence**
   Current status: reduced-grid bootstrap-current comparisons are useful and
   documented; full production convergence remains a validation lane.

3. **Lower-memory native sparse-factor replacement**
   Current status: direct `Pmat`, symbolic ordering, nested-dissection,
   BLR/HSS-style, and residual-admission infrastructure exists, but it is not
   promoted for hardest geometry-rich production cases.

4. **Geometry-rich RHSMode=2/3 production preconditioner**
   Current status: reduced geom2/geom11 gates pass; full production setup is
   still too slow for default promotion.

5. **Single-case multi-GPU strong scaling**
   Current status: independent-case/RHS parallelism is the practical public
   scaling story; single-case strong scaling remains experimental.

6. **95% meaningful coverage**
   Current status: target requires more ownership extraction and focused tests,
   not slow production solves in normal CI.

## Refactor Open Lanes

1. **Make `v3_driver.py` orchestration-only**
   Extract one more cohesive driver boundary if it removes real responsibility,
   not just wrapper lines. The next likely candidates are progress/timing
   reporting, a larger result/output handoff, or solve-result metadata assembly
   not already covered by finalization.

2. **Split output schema from `io.py`**
   Move solved-field schema, diagnostics, solver metadata, timing, memory, and
   provenance contracts behind a small output contract. Keep file-format
   writers in `outputs.formats`.

3. **Stabilize RHSMode=1 ownership**
   Stop broad RHSMode=1 churn unless a complete remaining family has a clear
   domain home. Keep `rhs1_full_assembly.py` as assembly/dispatch/admission
   owner until a full family can move cleanly.

4. **Consolidate package layout**
   Identify compatibility-only top-level modules, remove redundant aliases, and
   prefer fewer clearer domain modules over many small historical wrappers.

5. **Preserve differentiable/non-differentiable API separation**
   Make branch certificates and implicit-differentiation contracts explicit in
   solver result metadata and docs.

6. **Raise coverage through real tests**
   Every extraction gets direct tests. Add numerical and physics gates where
   they are cheap and meaningful; keep CPU/GPU/Fortran sweeps in release/manual
   tiers.

7. **Keep documentation synchronized**
   Update `README.md`, `docs/source_map.rst`, `docs/testing.rst`,
   `docs/development_roadmap.rst`, and `docs/research_lanes.rst` only when
   claims or ownership change.

## Prioritized Execution Plan

### P0. Keep Branch/PR Hygiene Green

Goal: keep one draft PR and no hidden branch divergence.

Actions:

1. Push coherent tranches to both the active implementation branch and PR #8
   branch.
2. Keep PR #8 in draft until the review-ready boundary is met.
3. Check CI after meaningful pushes, not after every local edit.

Acceptance:

- PR #8 remains the single draft PR.
- Local worktree is clean after each pushed tranche.

### P1. Extract One More Real Driver Stage

Goal: reduce `v3_driver.py` by moving a cohesive stage, not wrapper clutter.

Preferred choices:

1. progress/timing reporting,
2. solve-result metadata assembly that is not already covered by finalization,
3. a larger output/result handoff boundary.

Acceptance:

- The extracted module has direct tests.
- Driver keeps only orchestration and dependency injection.
- Public CLI/Python behavior is unchanged.

### P2. Split `io.py` Output Schema

Goal: make output behavior testable without solver internals.

Actions:

1. Define one file-format-independent output schema for solved fields,
   diagnostics, solver metadata, timing, memory, and provenance.
2. Keep HDF5/NetCDF/NPZ serialization in `outputs.formats`.
3. Add direct tests proving `.h5`, `.nc`, and `.npz` share the same core fields.

Acceptance:

- `io.py` becomes smaller orchestration/compatibility code.
- Output schema tests catch missing metadata and format drift.

### P3. Consolidate And Delete Compatibility Surfaces

Goal: reduce file count and cognitive load.

Actions:

1. Audit top-level historical modules for implementation, compatibility-only,
   or dead status.
2. Move implementation into existing domain packages only when it improves
   ownership.
3. Delete redundant aliases after import-contract tests prove they are unused.
4. Add short module docstrings to explain physics/numerical responsibility.

Acceptance:

- File count does not grow without a domain reason.
- Developers can infer code location from the equation, solver, or workflow.

### P4. Make Solver Contracts Explicit

Goal: keep adaptive performance and differentiability honest.

Actions:

1. Record branch certificates for `auto` decisions.
2. Keep host-only fallbacks out of differentiable lanes.
3. Use implicit linear-solve differentiation contracts for JAX-native solves.
4. Treat `lineax`, `jaxopt`, `equinox`, and `optax` as optional measured
   clarity/performance lanes, not required dependencies unless they prove value.

Acceptance:

- Differentiable examples remain JAX-transformable on documented fixtures.
- CLI remains fast, robust, and residual-clean.

### P5. Raise Coverage With Meaningful Tests

Goal: move toward 95% meaningful coverage while keeping CI practical.

Actions:

1. Target extracted modules first: policies, metadata, output schema,
   active-DOF layouts, sparse/preconditioner primitives, and result contracts.
2. Add synthetic-operator solver tests for residual gates and fail-closed
   behavior.
3. Add physics gates for conservation/null modes, radial normalization,
   collisionality trends, ambipolar sign/root behavior, and bootstrap-current
   normalization.
4. Keep expensive full CPU/GPU/Fortran runs outside normal CI.

Acceptance:

- Coverage increases because responsibilities are smaller and testable.
- Normal CI remains in the practical budget.

### P6. Documentation And README Pass

Goal: keep public claims clear and reviewer-proof.

Actions:

1. README stays short: install, one-command solve, plot command, public figures,
   and short scope notes.
2. Deep algorithms, equations, validation tiers, and deferred lanes stay in
   docs.
3. Source map and API docs match the refactored module ownership.

Acceptance:

- No README claim depends on incomplete production-resolution evidence.
- Docs show where equations, algorithms, tests, and claims live.

### P7. Benchmarks, Parity, And Figures

Goal: regenerate public artifacts only from complete evidence.

Actions:

1. Do not regenerate runtime/memory/parity plots for behavior-preserving
   refactors.
2. After solver behavior changes, run complete CPU reports locally and GPU
   reports on `ssh office` when needed.
3. Regenerate README/docs plots only from canonical complete JSON reports.

Acceptance:

- Public plots trace to checked complete reports.
- Fortran v3 comparisons use matched physics, resolution, and normalization.

### P8. Make PR #8 Review-Ready

Goal: one coherent refactor PR with no hidden release-claim drift.

Actions:

1. Ensure PR #8 points at the active refactor head and remains draft until all
   review gates pass.
2. Run focused tests for touched domains, Sphinx `-W`, `ruff`, `py_compile`,
   `git diff --check`, and repo-size checks.
3. Check CI after a meaningful push, not after every local edit.
4. Summarize what changed, what stayed behavior-preserving, and which research
   lanes remain deferred.

Acceptance:

- PR story is understandable.
- `v3_driver.py`, `rhs1_full_assembly.py`, and `io.py` have documented
  remaining responsibilities.
- README/docs/tests match the code.

## Review-Ready Boundary

The refactor PR is ready for review when:

- P0 through P3 are complete or explicitly deferred with rationale.
- `v3_driver.py` is primarily orchestration and dependency injection.
- `rhs1_full_assembly.py` no longer hides a major unowned preconditioner family.
- `io.py` has a documented output-schema split plan or an implemented schema
  extraction.
- Public CLI and Python APIs preserve existing output schemas.
- Differentiable and non-differentiable lanes are explicit and tested.
- Focused local validation, `ruff`, `py_compile`, `git diff --check`,
  repo-size audit, and Sphinx pass.
- README/docs distinguish production claims, reduced-grid evidence, and
  deferred research lanes.

## Explicitly Deferred Research Work

These lanes remain important but should not block the refactor PR unless a
change touches their public claims:

- true differentiable device-QI at production tolerance,
- production-resolution QI ladders,
- full-grid QA/QH RHSMode=1 production convergence beyond reduced-grid
  documentation evidence,
- single-case multi-GPU strong scaling as a public performance claim,
- lower-memory native sparse-factor replacement for the largest geometry-rich
  RHSMode=2/3 and full-grid QA/QH RHSMode=1 cases.

Deferred means fail-closed, documented, and test-gated where possible. Future
algorithm work should target stronger operator/coarse/factor architectures and
complete CPU/GPU/Fortran gates, not more smoother/restart tuning.
