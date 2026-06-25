# SFINCS_JAX Refactor And Release-Readiness Plan

Last updated: 2026-06-25 (America/Chicago)

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

- A source-audited Fortran v3 versus `sfincs_jax` feature matrix now lives in
  `docs/feature_matrix.rst` and is linked from the Sphinx index. It records the
  implementation owner and promotion gate for ambipolar options 1/2/3, RHSMode
  4/5 sensitivities, solver backends, geometry/Phi1 compatibility, outputs, and
  parallelism so the refactor PR has a single review anchor for parity and
  deferred functionality.
- Ambipolar Fortran-v3 profile-summary replay gates now cover option-1
  safeguarded Newton/bisection and option-3 pure Newton on the small helical
  sequence, and the production metadata gate checks solver-count proxies,
  adjoint-solve counts, RSS bounds, PETSc marker provenance, and physical
  residual/success-marker separation without running Fortran in CI.
- The public API contract now includes lazy workflow facades
  `sfincs_jax.write_output`, `sfincs_jax.read_output`, and
  `sfincs_jax.run_ambipolar_brent`, with fast routing tests. This gives users a
  stable Python entry point before more internals move out of legacy modules.
- The real in-process ambipolar evaluator now has an explicit fixed-shape setup
  reuse admission contract. Each evaluation records whether a prior state
  existed, whether it was used, the input/output shape signatures, the admission
  reason, and a cumulative same-shape reuse count; the CLI summary serializes
  the same fields.
- `sfincs_jax.sensitivity` now has a matrix-free implicit
  linear-observable derivative contract, and
  `sfincs_jax.problems.ambipolar` exposes matching radial-current adapters.
  Production owners can provide operator actions, transpose actions,
  parameter-derivative actions, and solve/transpose-solve closures without
  dense assembly; focused tests compare the certificate against the dense path
  and centered finite differences.
- `sfincs_jax.problems.ambipolar` now has a concrete
  `matrix_free_rhs1_vm_radial_current_linear_observable_system` builder for the
  RHSMode-1 radial-current path. The builder uses real matrix-free full-system
  operator actions and caller-supplied transpose/solve closures, while the
  current derivative actions remain finite-difference gates until analytic/JVP
  `Er` derivatives are wired.
- Major RHSMode=1 preconditioner families now have domain owners:
  - full-CSR Schur preconditioners,
  - Fortran-reduced symbolic sparse factors,
  - low-`ell` x-block Schur preconditioners,
  - active-projected x-block / overlap-Schwarz / diagonal-Schur / line
    preconditioners,
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
  passes solve-local arrays/operators/timing into the domain-owned QI pipeline
  context for the tested fail-closed research lane.
- Stale private `v3_driver.py` QI policy aliases are being deleted when their
  canonical owner is now `sfincs_jax.problems.profile_response.policies`; tests
  now enforce that ownership boundary instead of preserving aliases only for
  historical convenience.
- QI seed/Galerkin/two-level/device/deflated stage orchestration no longer
  lives inline in `v3_driver.py`; the driver now calls
  `run_xblock_qi_preconditioner_pipeline()` from
  `sfincs_jax.problems.profile_response.sparse.qi`, while QI builder wiring
  and metadata construction stay in the profile-response sparse package.
- Final stale `v3_driver.py` QI coarse-basis/block-metadata aliases were
  removed. The extracted QI pipeline now obtains canonical coarse-basis and
  block-metadata builders from its own default context factory, and tests assert
  that old private driver aliases stay gone rather than preserving them for
  historical convenience.
- RHSMode=1 PAS-family binding moved into
  `sfincs_jax.solvers.preconditioners.pas.RHS1PasFamilyBuilders`; `v3_driver.py`
  now keeps only thin private wrappers that inject current low-level builders
  for compatibility-sensitive tests and debug workflows.
- RHSMode=1 strong-preconditioner family build mapping moved from flat
  `rhs1_strong_fallback.py`/driver imports into
  `sfincs_jax.problems.profile_response.preconditioner_build.RHS1StrongPreconditionerFamilyBuilders`.
  `rhs1_strong_fallback.py` is now only a compatibility facade for historical
  imports.
- RHSMode=1 production-output refusal gates, residual/target extraction,
  solver-trace memory estimates, and nonconverged sidecar trace writing moved
  from `io.py` into `sfincs_jax.outputs.rhsmode1`.
- RHSMode=1 solver-diagnostics output schema, including residual convergence,
  sparse-PC timing/memory fields, direct-tail support-mode metadata, true
  coupled-coarse fields, and residual-target ratios, moved from `io.py` into
  `sfincs_jax.outputs.rhsmode1` without adding a new file.
- RHSMode=2/3 transport solver residual-output arrays moved from `io.py` into
  `sfincs_jax.outputs.transport`, with a direct output-schema test covering
  missing RHS diagnostics and max residual summaries.
- RHSMode=2/3 streaming HDF5 transport writer and radial derivative conversion
  factors moved from `io.py` into `sfincs_jax.outputs.transport`. `io.py` now
  orchestrates when streaming is selected, while the output-domain module owns
  slice-wise HDF5 layout, flux variants, classical flux arrays, transport
  matrix, elapsed time, and solver-diagnostic datasets.
- Output geometry-cache gates, stable cache paths, content-based equilibrium
  identity, cache-key construction, and disk load/save moved from `io.py` into
  `sfincs_jax.outputs.cache`. `io.py` now only injects its equilibrium resolver
  into a thin compatibility wrapper for the cache-key helper.
- RHSMode=1 x-block sparse-PC final metadata and payload assembly moved out of
  `v3_driver.py` into the profile-response sparse x-block handoff helpers. The
  driver now supplies solve-local scope plus the accepted physical solution,
  while `sfincs_jax.problems.profile_response.sparse_pc` owns the final
  diagnostic payload contract used by `V3LinearSolveResult`.
- QI-specific code left in `v3_driver.py` was audited after the extraction.
  The remaining references are live solve-local handoff for the tested QI
  pipeline and augmented-Krylov path; metadata-only relay locals made obsolete
  by the sparse-PC finalization handoff were deleted.
- QI default-builder wiring moved into
  `sfincs_jax.problems.profile_response.sparse.qi.build_xblock_qi_stage_pipeline_context()`.
  `v3_driver.py` now passes only solve-local arrays/operators/timing into the
  QI pipeline instead of importing every coarse/Galerkin/two-level/device/
  deflated helper directly. Tests patch the canonical QI device-preconditioner
  module rather than stale driver aliases.
- Matrix-free QI seed context construction, env-gate resolution, and optional
  early/pre-sparse attempt bookkeeping moved into
  `sfincs_jax.problems.profile_response.qi_device_seed`. `v3_driver.py` now
  keeps only the reduced-solve hook placement, the strong-solver skip flag, and
  replay-state update when the domain helper reports an improved seed.
- Final direct QI two-level/global-coupling builder aliases were removed from
  `v3_driver.py`. The sparse-PC stage helpers now resolve canonical QI default
  builders internally when the driver does not inject a test builder, and tests
  assert that the old private driver aliases stay absent.
- Transport runtime backend/dtype policy binding moved into
  `sfincs_jax.problems.transport_matrix.policies.TransportRuntimePolicy`.
  `v3_driver.py` now keeps bound method aliases for compatibility and no longer
  owns private transport backend-injection wrapper functions.
- Active-projected diagonal-Schur, x-ell kinetic-line, angular-line, and native
  indexed Schwarz
  preconditioners moved from `rhs1_full_assembly.py` into
  `sfincs_jax.solvers.preconditioners.xblock.active_projected`. The assembly
  module now keeps compatibility aliases plus dispatch/admission logic for
  these builders.
- Active-projected global field-split, multiline field-split, bounded native
  stack, and Fortran-v3-reduced native-stack preconditioners moved from
  `rhs1_full_assembly.py` into
  `sfincs_jax.solvers.preconditioners.xblock.active_projected`. The extracted
  module owns local base dispatch for x-block/angular/native-indexed bases, and
  `rhs1_full_assembly.py` injects its dispatcher only where a still-local base
  family is needed.
- The previous next tranche was a larger result/output handoff or `io.py`
  solved-field schema split. The revised next tranche is now either the
  ambipolar/sensitivity owner or a full profile-response orchestration split,
  because both reduce churn more than wrapper-level moves. The QI audit is
  complete for this refactor stage:
  no standalone `qi_*` functions/classes remain in `v3_driver.py`, no direct QI
  builder aliases remain there, and the remaining QI references are live
  solve-local hook placements into tested domain modules.
- The README and docs currently state the public claim boundary: the documented
  release suite is CPU/GPU parity-clean, while production-resolution QI, true
  device-QI, lower-memory native factor replacement, full-grid QA/QH RHSMode=1,
  and single-case multi-GPU scaling remain fail-closed research lanes.
- The latest SFINCS Fortran v3 source audit found functionality that should be
  tracked separately from solver-performance refactors:
  - `ambipolarSolve=.true.` with `ambipolarSolveOption=1` uses safeguarded
    Newton/bisection, option 2 uses Brent, and option 3 uses pure Newton.
  - Options 1 and 3 compute `dRadialCurrentdEr` through an adjoint solve:
    `populateAdjointRHS(..., particle flux, species-summed)` followed by
    `computedRadialCurrentdEr()`, which forms
    `dL/dEr * f - dS/dEr` via `populatedMatrixdLambda(..., whichLambda=0)`
    and `populatedRHSdLambda(..., whichLambda=0)`.
  - `RHSMode=4/5` expose adjoint sensitivity outputs for particle flux, heat
    flux, parallel flow, total heat flux, radial current, bootstrap current,
    and, for `RHSMode=5`, `dPhi/dPsi`. Debug mode also writes finite-difference
    sensitivity checks and percent errors.
  - `sfincs_jax` currently has Er-scan postprocessing compatible with upstream
    scan scripts and several autodiff validation examples, but not yet a
    first-class in-solver ambipolar Newton / adjoint-sensitivity API matching
    these Fortran v3 modes.

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
- QI metadata simplification preserves sparse/QI driver behavior:
  `504 passed in 125.76s`.
- QI pipeline extraction preserves profile-response sparse-PC, QI driver
  sparse-pattern metadata, QI unit, QI artifact/helper, and import-contract
  behavior:
  `323 passed in 2.14s`, `57 passed in 26.34s`, `178 passed in 28.85s`,
  and `7 passed in 0.53s`.
- QI alias cleanup preserves live QI coarse/device/two-level/deflated behavior
  and sparse-driver handoff behavior:
  `81 passed in 18.53s` and `552 passed in 112.36s`.
- QI default-builder ownership cleanup preserves sparse-PC behavior while
  removing direct QI helper imports from `v3_driver.py`:
  `324 passed in 2.14s`.
- Active-projected diagonal-Schur/line extraction preserves full-assembly,
  sparse-pattern, package-import, active-projected, and Schur focused behavior:
  `121 passed in 36.71s`, `132 passed in 116.36s`, and `18 passed in 0.64s`.
- QI domain-ownership audit plus native indexed Schwarz extraction preserve the
  focused QI and active-CSR gates:
  `400 passed in 18.96s`, `344 passed in 7.07s`, `121 passed in 38.05s`,
  `132 passed in 129.62s`, and `9 passed in 1.07s`.
- Transport-streaming output extraction preserves output-format, streaming,
  coordinate-conversion, import-contract, and write-output smoke/parity paths:
  `40 passed in 9.46s`, `8 passed in 0.76s`, and `8 passed in 185.68s`.
- QI code is still live and tested: after the latest audit, the QI unit suite
  and artifact/research-lane checks pass while old private driver aliases stay
  absent:
  `127 passed in 32.65s`, `102 passed in 1.05s`, and the focused QI seed/coarse
  tests pass with the new domain-owned seed setup (`17 passed in 5.56s`).
- Final QI driver-surface cleanup preserves canonical default-builder behavior
  and confirms that the remaining QI code works in the documented fail-closed
  scope:
  `3 passed in 1.98s`, `334 passed in 5.15s`, and `229 passed in 30.40s`.
- The combined focused QI/sparse validation after this cleanup passes:
  `556 passed in 32.10s`.
- Transport runtime-policy binding preserves driver compatibility and canonical
  transport policy behavior:
  `83 passed in 11.20s`; broader transport tests pass with
  `287 passed in 54.12s`.
- Active global-field-split/native-stack extraction preserves all
  RHSMode=1 full-assembly behavior:
  `121 passed in 39.49s`.
- Package facade updates for the x-block preconditioner domain preserve import
  contracts:
  `7 passed in 0.70s`.
- RHSMode=1 output-gate extraction preserves IO helper coverage and solver-trace
  output-format behavior:
  `18 passed in 0.33s`; after switching moved-helper tests to the new owner
  module and combining with import contracts, `25 passed in 0.62s`.
- Broader write-output and CLI forwarding paths remain intact:
  `58 passed in 20.29s`.
- RHSMode=1 PAS-family extraction preserves PAS composite, driver policy,
  Schur heuristic, import-contract, preconditioner context, and profile-response
  preconditioner build behavior:
  `48 passed in 26.32s` and `18 passed in 0.38s`.
- RHSMode=1 strong-family extraction preserves direct profile-response builder
  behavior, legacy driver compatibility, and strong-control/policy/auto-kind
  gates:
  `63 passed in 0.69s`.
- RHSMode=1 solver-diagnostics output-schema extraction preserves HDF5/export,
  RHSMode=1 write-output, Phi1 write-output, and state-recycle parity fields:
  `23 passed in 20.43s`.
- RHSMode=1 x-block sparse-PC finalization handoff preserves the sparse-PC
  result-contract tests and driver import boundaries:
  `323 passed in 2.30s`; `ruff` passes on the touched driver and sparse-PC
  tests.
- Transport output-schema split adds a direct test for residual diagnostics
  and preserves output/import contracts:
  `15 passed in 1.30s`.
- Output geometry-cache extraction preserves cache gating, content identity,
  cache roundtrip, VMEC/equilibrium cache-key behavior, legacy writer cache
  aliases, and output-policy helpers:
  `22 passed in 0.98s`.
- `ruff` and `py_compile` pass on touched transport-parallel, finalization, and
  setup/QI files.
- PR #8 remains draft. Check CI after the next meaningful push rather than
  polling continuously during local refactor tranches.

### Current Code Shape

- `sfincs_jax/v3_driver.py`: about 12.2k lines, still the largest orchestration
  and compatibility surface.
- `sfincs_jax/rhs1_full_assembly.py`: about 6.0k lines, now mostly RHSMode=1
  exact/active CSR assembly, admission, dispatch, and compatibility.
- `sfincs_jax/io.py`: about 4.3k lines, still owns too much solved-field physics
  schema and provenance materialization; RHSMode=1 output safety and solver
  diagnostics now live in `sfincs_jax.outputs.rhsmode1`, transport-output
  helpers live in `sfincs_jax.outputs.transport`, and geometry-cache helpers
  live in `sfincs_jax.outputs.cache`.
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

7. **Fortran v3 ambipolar and adjoint-sensitivity functionality**
   Current status: scan-based ambipolar postprocessing exists, but Fortran-v3
   in-solver ambipolar Newton options and `RHSMode=4/5` adjoint sensitivity
   outputs are not yet full feature-parity claims. The missing public
   capabilities include:
   - direct solves for `ambipolarSolve=1` with options 1, 2, and 3,
   - `d(radial current)/dEr` from a residual/Jacobian derivative path,
   - derivatives of particle flux, heat flux, parallel flow, total heat flux,
     bootstrap current, and `dPhi/dPsi` with respect to `Er` and Boozer/geometry
     harmonic parameters where the model is shared,
   - output-schema fields equivalent to Fortran v3
     `dParticleFluxdLambda`, `dHeatFluxdLambda`, `dParallelFlowdLambda`,
     `dTotalHeatFluxdLambda`, `dRadialCurrentdLambda`,
     `dBootstrapdLambda`, and `dPhidPsidLambda`,
   - finite-difference or complex-step debug gates for small cases.

   Preferred implementation: use JAX implicit differentiation of the linear
   residual as the primary differentiable lane, and use the same derivative
   kernels in a fast non-differentiable CLI lane for Newton/Brent ambipolar root
   solves. Host-only finite differences are allowed only as debug/reference
   checks and must be recorded in metadata.

## Refactor Open Lanes

1. **Make `v3_driver.py` orchestration-only**
   Extract complete owner boundaries, not aliases. The next acceptable driver
   tranches are:
   - an ambipolar/root-solve owner that handles direct Er root solves and
     derivative certificates,
   - a profile-response solve orchestration owner that removes the remaining
     large solve-local decision tree,
   - a result/output handoff owner that owns solved-field metadata assembly,
   - progress/timing/profiling reporting as one cohesive runtime-observability
     module.

   Do not spend more tranches moving one-off wrappers unless the same commit
   deletes the compatibility path or removes a full branch of the driver.

2. **Split output schema from `io.py`**
   Move solved-field schema, diagnostics, timing, memory, and provenance
   contracts behind a small output contract. Keep file-format writers in
   `outputs.formats` and RHSMode=1 output-safety gates in `outputs.rhsmode1`.
   The geometry-output cache family is now complete in `outputs.cache`; the
   next output split should target solved-field/provenance schema assembly or
   export-f mapping, not another cache helper.

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

8. **Close Fortran-v3 feature gaps deliberately**
   Treat missing Fortran-v3 functionality as product scope, not incidental
   cleanup. Ambipolar Newton and adjoint sensitivities get their own design,
   tests, docs, and output schema rather than being hidden in scan examples.

## Tranche Efficiency Rules

The previous pattern created many small files while only slowly reducing the
monolith. Starting from this point, a refactor tranche is considered worthwhile
only if it satisfies at least one of these gates:

1. It removes a complete owner boundary of at least one real algorithm family
   from `v3_driver.py`, `io.py`, or `rhs1_full_assembly.py`.
2. It deletes stale compatibility code or reduces a monolith by at least about
   300 lines without adding an equivalent amount of wrapper code elsewhere.
3. It turns a missing functionality lane into a tested, documented feature with
   clear public API and output schema.
4. It consolidates multiple historical top-level files into an existing domain
   package and reduces file-count or import-path ambiguity.

Every tranche must start with a short design note in the commit message or PR
summary covering owner module, code to delete, public compatibility, tests, and
docs touched. Every tranche must end with:

- direct tests for the new owner,
- one integration or import-contract test for compatibility,
- `ruff`/`py_compile` on touched files,
- `git diff --check`,
- `docs/source_map.rst` update when equation ownership changes.

The default next tranche is therefore not another alias move. It is either:

1. **Ambipolar/sensitivity owner extraction and feature design** if we prioritize
   Fortran-v3 functionality parity, or
2. **Profile-response solve orchestration extraction** if we prioritize
   monolith reduction.

Both are large enough to reduce churn. The recommended order is ambipolar /
sensitivity first because it adds missing product functionality and gives the
refactor a clear differentiable/non-differentiable API boundary.

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

1. ambipolar root-solve and derivative-certificate orchestration,
2. profile-response solve orchestration after RHSMode=1 setup/finalization,
3. progress/timing/profiling reporting,
4. solve-result metadata assembly that is not already covered by finalization,
5. a larger output/result handoff boundary.

Acceptance:

- The extracted module has direct tests.
- Driver keeps only orchestration and dependency injection.
- Public CLI/Python behavior is unchanged.
- The tranche deletes a full owner boundary or records a concrete reason why a
  compatibility shim remains.

### P1A. Add Fortran-V3 Ambipolar And Sensitivity Parity Design

Goal: make missing ambipolar Newton and adjoint sensitivity capabilities an
explicit implementation lane.

Actions:

1. Add a domain owner, tentatively `sfincs_jax.problems.ambipolar` or
   `sfincs_jax.problems.profile_response.sensitivity`, with:
   - Brent root solve matching `ambipolarSolveOption=2`,
   - safeguarded Newton/bisection matching `ambipolarSolveOption=1`,
   - pure Newton matching `ambipolarSolveOption=3`,
   - derivative certificates for `d(radial current)/dEr`.
2. Implement derivative kernels using the linear residual contract:
   `dF/dEr = dDiagnostic/dEr - dDiagnostic/df * A^{-1} * dResidual/dEr`
   or an equivalent JAX `custom_linear_solve` / implicit-differentiation form.
3. Extend the Python API with explicit differentiable and non-differentiable
   paths:
   - differentiable: JAX implicit differentiation and transformable result
     functions,
   - CLI/non-differentiable: robust root solve using derivative-backed Newton
     when available and Brent fallback otherwise.
4. Add output metadata and optional output datasets for:
   `ambipolarEr`, root history, radial-current history,
   `dRadialCurrentdEr`, selected option, residual/root tolerances, and branch
   certificate.
5. Add tests:
   - analytic scalar root fixture for all three root options,
   - finite-difference vs JAX derivative of radial current on a tiny linear
     fixture,
   - Fortran-v3 small-input parity for Brent root and Newton derivative sign,
   - bootstrap/current derivative API shape and metadata tests,
   - docs example that differentiates current or ambipolar residual with
     respect to `Er`.

Acceptance:

- `sfincs_jax` can answer "what is d current / d Er?" from a public API.
- Fortran-v3 ambipolar option semantics are documented and tested where shared.
- Host-only finite differences are labeled debug/reference, not the primary
  differentiable claim.
- Full production-resolution Er-root campaigns remain release/manual-tier tests
  unless they fit the CI budget.

### P2. Split `io.py` Output Schema

Goal: make output behavior testable without solver internals.

Actions:

1. Define one file-format-independent output schema for solved fields,
   diagnostics, timing, memory, and provenance.
2. Keep HDF5/NetCDF/NPZ serialization in `outputs.formats` and convergence
   refusal/trace policy in `outputs.rhsmode1`.
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
- Historical top-level modules that only re-export one domain owner are deleted
  once import-contract tests and downstream examples no longer need them.

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
- P1A has either an implemented first tranche or a checked design issue with
  explicit API/tests/docs acceptance criteria.
- `v3_driver.py` is primarily orchestration and dependency injection.
- `rhs1_full_assembly.py` no longer hides a major unowned preconditioner family.
- `io.py` has a documented output-schema split plan or an implemented schema
  extraction.
- Public CLI and Python APIs preserve existing output schemas.
- Differentiable and non-differentiable lanes are explicit and tested.
- Fortran-v3 ambipolar/adjoint-sensitivity functionality is no longer confused
  with scan-only postprocessing in docs or README claims.
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

The Fortran-v3 ambipolar Newton and `RHSMode=4/5` adjoint-sensitivity feature
gap is not a performance research lane. It is a functionality parity lane and
should be designed and implemented in bounded phases, with production-resolution
campaigns remaining release/manual-tier validation.

Deferred means fail-closed, documented, and test-gated where possible. Future
algorithm work should target stronger operator/coarse/factor architectures and
complete CPU/GPU/Fortran gates, not more smoother/restart tuning.

## 2026-06-23 Ambipolar Fortran v3 Reference Refresh

Steps taken:

1. Added missing geometry-1 helical ambipolar option-1 and option-3 namelists
   at small and production tiers.
2. Extended `benchmarks/fortran_v3_ambipolar_reference/run_fortran_v3_ambipolar.py`
   with PETSc profiling flags, log-path provenance, RSS parsing, KSP/PC profile
   markers, and convergence-marker separation.
3. Ran the full small-tier Fortran v3 ambipolar matrix:
   geometry-1 helical and geometry-4 W7-X-like, options 1/2/3.
4. Ran the full production-tier matrix at `13 x 19 x 48 x 5` for the same
   geometries and options.
5. Updated `plan_final.md` so the new 2026-06-23 summaries are the controlling
   ambipolar implementation gates.
6. Added CI-light tests that replay a derivative-assisted Newton sequence and
   check that production summaries preserve solver counts, RSS, and the
   distinction between small residual and declared Fortran success marker.

Results:

- Small tier: all six decks completed useful physical roots through PETSc/MUMPS.
- Production W7-X-like: options 1/2/3 converged to `Er ~ -3.57735`, with peak
  RSS around `1.35-1.39 GB`.
- Production helical: options 1 and 3 converged to `Er ~ -3.26189`, with peak
  RSS around `5.7-5.8 GB`.
- Production helical option 2 reached `|J_r| ~ 8.1e-12` but did not print
  Fortran's Brent success marker before exhausting its evaluation budget. This
  must remain a marker/residual split in sfincs_jax, not an implicit success.
- Focused validation passed: `python -m pytest tests/test_ambipolar_problem.py
  -q --tb=short`, JSON validation, runner `py_compile`, namelist sanity check,
  and `git diff --check`.

Next best steps:

1. Implement the exact implicit/adjoint `dJr/dEr` helper against a small RHSMode
   1 residual graph and finite-difference gate.
2. Wire that derivative into the existing safeguarded Newton and pure Newton
   ambipolar solvers.
3. Add sfincs_jax-vs-Fortran replay gates for the new production profiles
   without running Fortran in CI.

## 2026-06-25 Implicit Linear Sensitivity Certificate

Steps taken:

1. Added `sfincs_jax.sensitivity` as the domain owner for reusable
   implicit/adjoint derivative certificates.
2. Implemented `implicit_linear_observable_derivative` for
   `A(p) x(p) = b(p)` and `J(p) = c(p)^T x(p) + J0(p)`.
3. The helper solves both the tangent equation
   `A x_p = b_p - A_p x` and the adjoint equation `A^T lambda = c`, reports
   primal/tangent/adjoint residuals, and optionally compares to centered finite
   differences.
4. Added `implicit_linear_radial_current_derivative` in
   `sfincs_jax.problems.ambipolar`, returning the existing
   `RadialCurrentDerivativeResult` contract expected by the safeguarded Newton
   and pure Newton ambipolar solvers.
5. Added API/docs entries and import-contract coverage for the new public
   module and ambipolar adapter.

Results:

- Focused tests passed:
  `python -m pytest tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `24 passed`.
- Sphinx docs build passed:
  `python -m sphinx -b html docs docs/_build/html -q`.
- Static checks passed:
  `python -m py_compile sfincs_jax/sensitivity.py
  sfincs_jax/problems/ambipolar.py sfincs_jax/problems/__init__.py` and
  `git diff --check`.

Next best steps:

1. Wire the certificate to the concrete RHSMode 1 radial-current residual graph
   so `dJr/dEr` uses the actual SFINCS_JAX operator/RHS derivatives.
2. Compare the concrete derivative against centered finite differences on a
   tiny RHSMode 1 case.
3. Use that derivative provider in safeguarded Newton and pure Newton
   ambipolar solves for Fortran option-1/3 parity gates.

## 2026-06-25 Linear Observable Builder Bridge

Steps taken:

1. Added `LinearObservableSystem` and
   `implicit_linear_observable_derivative_from_builder` to
   `sfincs_jax.sensitivity`.
2. The builder API gives concrete RHSMode 1/4/5 owners one fixed-shape handoff:
   emit `A(p)`, `b(p)`, the scalar observable vector, and their parameter
   derivatives from the true operator graph.
3. Added `evaluate_linear_observable` so centered finite-difference gates can
   be generated from the same builder used by the implicit derivative.
4. Added `implicit_linear_radial_current_derivative_from_builder` to the
   ambipolar problem owner, preserving the existing Newton derivative-provider
   contract while allowing custom forward and transpose solvers.

Results:

- Focused tests passed:
  `python -m pytest tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `26 passed`.
- Static compile passed:
  `python -m py_compile sfincs_jax/sensitivity.py
  sfincs_jax/problems/ambipolar.py sfincs_jax/problems/__init__.py`.

Next best steps:

1. Implement the concrete RHSMode 1 radial-current builder from the active
   operator/RHS/observable path.
2. Compare the builder derivative against finite differences on a tiny real
   RHSMode 1 deck before using it in Newton ambipolar solves.
3. Generalize the builder shape to RHSMode 4/5 adjoint outputs once the RHSMode
   1 radial-current gate is stable.

## 2026-06-25 Chunked Observable-Vector Probe

Steps taken:

1. Added `probe_linear_observable_vector` to `sfincs_jax.sensitivity`.
2. The helper recovers `J(x) = c^T x + J0` from an existing scalar diagnostic
   by probing basis vectors in bounded chunks.
3. Added tests for nonzero-offset observables and chunk sizes that do not
   divide the vector length.

Results:

- Focused tests passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `14 passed`.
- Static checks passed:
  `python -m py_compile sfincs_jax/sensitivity.py` and `git diff --check`.

Next best steps:

1. Use the probe on a tiny real RHSMode 1 deck to pin the radial-current
   observable vector against the existing output diagnostics.
2. Replace the probe with analytic radial-current weights after the diagnostic
   formula and coordinate conversion are pinned.

## 2026-06-25 RHSMode 1 Radial-Current Observable Hook

Steps taken:

1. Added `radial_current_vm_psi_hat_from_state` to
   `sfincs_jax.problems.transport_matrix.diagnostics`.
2. Added `radial_current_vm_psi_hat_observable_vector`, which recovers the
   scalar radial-current observable vector from the existing diagnostic in
   bounded chunks.
3. Added coordinate-aware `radial_current_vm_from_state` and
   `radial_current_vm_observable_vector` helpers for `psiHat`, `rHat`, and
   `rN`.
4. Added a tiny RHSMode 1 fixture test showing that `c^T x + J0` matches the
   existing diagnostic for a random state.

Results:

- Tiny real-observable tests passed:
  `python -m pytest tests/test_sensitivity.py -q --tb=short` with
  `7 passed`.

Next best steps:

1. Build the corresponding RHSMode 1 `LinearObservableSystem` by pairing this
   observable with the active operator/RHS and their `Er` derivatives.
2. Add Phi1 drift-current support after the no-Phi1 magnetic-drift gate passes.

## 2026-06-25 Dense RHSMode 1 Linear-Observable Derivative Gate

Steps taken:

1. Added `dense_rhs1_vm_radial_current_linear_observable_system` to
   `sfincs_jax.problems.ambipolar`.
2. The helper assembles the true small-deck RHSMode 1 operator, RHS, radial
   current observable, and centered finite-difference derivatives from an
   operator triplet.
3. Added a tiny real RHSMode 1 gate that perturbs the density-gradient source,
   solves tangent/adjoint systems, and compares the resulting derivative against
   a centered finite-difference solve.

Results:

- Dense real RHSMode 1 derivative gate passed:
  `python -m pytest
  tests/test_sensitivity.py::test_dense_rhs1_radial_current_linear_observable_system_matches_finite_difference
  -q --tb=short` with `1 passed` in about 9 seconds.

Next best steps:

1. Replace dense validation assembly with a sparse/matrix-free builder for
   production RHSMode 1 surfaces.
2. Provide true `Er` derivatives of the operator/RHS, either analytically from
   the collisionless electric-field blocks or by a checked JVP path.
3. Wire the resulting derivative provider into safeguarded Newton and pure
   Newton ambipolar solves for Fortran option-1/3 parity gates.

## 2026-06-25 JVP/VJP Adjoint Dot-Product Gate

Steps taken:

1. Added `jvp_flux`, `vjp_flux`, and `adjoint_dot_product_check` to
   `sfincs_jax.sensitivity`.
2. Added a real RHSMode 1 radial-current diagnostic test for
   `<JVP(dp), y> = <dp, VJP(y)>`.
3. Updated docs to describe the dot-product gate as the first reusable
   consistency check for RHSMode 4/5 adjoint sensitivity outputs.

Results:

- Focused dot-product and import-contract tests passed:
  `python -m pytest
  tests/test_sensitivity.py::test_rhs1_radial_current_jvp_vjp_dot_product_gate
  tests/test_domain_package_import_contracts.py::test_active_modules_are_importable_with_expected_exports
  -q --tb=short` with `2 passed`.

Next best steps:

1. Extend the dot-product gate to Phi1 drift-current, total heat-flux, and
   intermediate-grid diagnostics.
2. Add Fortran RHSMode 4/5 small-output fixtures once the diagnostic adjoint
   gates are stable.

## 2026-06-25 Expanded Transport-Diagnostic Adjoint Gates

Steps taken:

1. Extended the real RHSMode 1 diagnostic JVP/VJP dot-product tests to
   particle flux, heat flux, FSAB flow, and `FSABjHat = sum_s Z_s FSABFlow_s`.
2. Kept the gate on the tiny PAS deck so the coverage remains fast enough for
   normal CI while exercising the production diagnostic code path.

Results:

- Expanded diagnostic gate passed:
  `python -m pytest
  tests/test_sensitivity.py::test_rhs1_transport_diagnostic_jvp_vjp_dot_product_gates
  -q --tb=short` with `1 passed`.

Next best steps:

1. Add Phi1 drift-current and total heat-flux diagnostic adjoint gates.
2. Add small Fortran RHSMode 4/5 fixture parity once the no-Phi1 diagnostic
   adjoint suite is stable.

## 2026-06-25 Matrix-Free Implicit Derivative Contract

Steps taken:

1. Added `MatrixFreeLinearObservableSystem` to `sfincs_jax.sensitivity`.
2. Added matrix-free tangent/adjoint derivative certificates that use supplied
   operator actions, transpose actions, parameter-derivative actions, and
   solve/transpose-solve closures instead of dense matrices.
3. Added ambipolar radial-current adapters so this production-facing
   certificate returns the same `RadialCurrentDerivativeResult` contract used
   by finite-difference and dense implicit derivative providers.
4. Added import-contract and numerical tests comparing the matrix-free path
   against the dense certificate and centered finite differences.

Results:

- Focused sensitivity/import gates passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `20 passed`.

Next best steps:

1. Wire concrete RHSMode 1 `Er` operator/RHS derivative actions into the
   matrix-free builder, starting with the no-Phi1 magnetic-drift radial-current
   path.
2. Add Phi1 drift-current branches after the no-Phi1 derivative action passes
   finite-difference and tangent/adjoint residual gates.
3. Feed the exact derivative provider into safeguarded Newton/bisection and
   pure Newton ambipolar option-1/3 parity gates.

## 2026-06-25 RHSMode 1 Matrix-Free Radial-Current Builder

Steps taken:

1. Added `matrix_free_rhs1_vm_radial_current_linear_observable_system` to the
   ambipolar problem owner.
2. The builder uses the real RHSMode-1 full-system matrix-free operator action,
   caller-supplied transpose and solve closures, finite-difference operator/RHS
   derivative actions, and existing radial-current observable weights.
3. Added a tiny real RHSMode-1 gate comparing the new matrix-free builder
   against the dense certificate without assembling the matrix inside the
   builder.

Results:

- Focused sensitivity/import gates passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `21 passed`.

Next best steps:

1. Replace finite-difference `Er` operator/RHS derivative actions with analytic
   terms or checked JVP actions for the no-Phi1 radial-current path.
2. Add a production-size matrix-free gate that uses the actual selected
   RHSMode-1 solve and transpose-solve routes instead of dense small-deck
   closures.
3. Add Phi1 drift-current support after the no-Phi1 action passes residual and
   finite-difference gates.
