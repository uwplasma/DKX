# SFINCS_JAX Refactor And Release-Readiness Execution Log

Last updated: 2026-06-26 (America/Chicago)

Active implementation branch: `refactor/rhs1-full-assembly-preconditioners`

Intended review PR: #8, `refactor/v3-driver-architecture`

PR state: draft. The active implementation branch has been pushed to the PR
branch at the latest clean commit. Do not open additional refactor PRs; keep PR
#8 as the single review surface until the authoritative `plan_final.md` gates
reach the review-ready boundary.

Authoritative plan: `plan_final.md`. This file is the execution log and
historical record only; if this file conflicts with `plan_final.md`, follow
`plan_final.md`.

Latest controlling update: `plan_final.md` now defines Lane 1 as locked
completed checkpoints plus five active consolidation batches only: Batch 1
transport/output/root payback, Batch 2 solver/preconditioner family
compression, Batch 3 compatibility-waiver/profile-response freeze, Batch 4
public API/docs/tests/source-map cleanup, and Batch 5 review-ready validation.
Older phase, sweep, tranche, iteration, and pass labels in this execution log
are historical context, not instructions to follow.

Latest execution checkpoint:

- The profile-response solve sequencer and sparse handoff export layer are
  compressed enough for the Lane 1 review gates:
  `profile_response/solve.py` is 5,358 lines and
  `profile_response/sparse/handoff.py` is 5,498 lines.
- The concrete output writer now lives in `sfincs_jax.outputs.writer`, while
  root `sfincs_jax.io` is a 49-line compatibility facade.
- The authoritative plan was refreshed so remaining work is not another series
  of one-helper moves: it must pay down files in transport/output/root owners,
  compress solver/preconditioner families, freeze compatibility shims, refresh
  docs/tests/source maps, and pass the review-ready gate.
- Current counts after this checkpoint: 196 package Python files, 43 package
  root files, 18 `problems/profile_response` files including `sparse`, 18
  `problems/transport_matrix` files including `parallel`, 47
  `solvers/preconditioners` files, `io.py` at 49 lines,
  `outputs/writer.py` at 4,264 lines, and 165,430 package Python lines.

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
- That matrix-free builder now accepts caller-supplied derivative actions and
  JAX operator tangents through `operator_tangent_from_centered_difference`.
  The helper constructs valid pytrees with `float0` tangents for integer/bool
  leaves, and a real electric-field `xDot` operator test verifies the JVP
  derivative action against centered differences.
- Matrix-free implicit radial-current derivative providers now plug directly
  into safeguarded Newton/bisection and pure Newton ambipolar root solvers.
  Fast option-1/3-style tests verify root convergence and derivative
  certificate metadata through the public root-solver API.
- No-Phi1 existing-branch `Er` operator tangents now use the v3 radial
  conversion analytically. The helper updates stored `dphi_hat_dpsi_hat` leaves
  in the full operator and f-block suboperators, and a real `xDot` fixture
  verifies the JVP action against centered operator differences.
- `keep_zero_er_terms` is now an explicit opt-in on the f-block and full-system
  operator builders. Normal solves keep previous branch behavior, while
  derivative gates can retain zero-valued ExB and `Er` suboperators at `Er=0`
  and compare analytic JVP tangents against nearby nonzero centered differences.
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

## 2026-06-25 RHSMode 1 JVP Operator-Tangent Derivative Actions

Steps taken:

1. Added `operator_tangent_from_centered_difference` to build valid JAX pytrees
   for operator tangents, including `float0` tangents for integer/bool leaves.
2. Extended the matrix-free RHSMode-1 radial-current builder so callers can
   supply explicit derivative actions, explicit RHS/observable derivatives, or
   an operator tangent used through `jax.jvp`.
3. Added a real electric-field `xDot` operator test that checks the JVP action
   against centered differences on the matrix-free full-system operator.

Results:

- Focused sensitivity/import gates passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `22 passed`.

Next best steps:

1. Replace the centered-difference operator-tangent construction with analytic
   `Er` tangents from namelist radial-coordinate conversion where possible.
2. Add a real derivative-assisted ambipolar option-1/3 gate using the
   matrix-free/JVP derivative provider.
3. Add Phi1 drift-current derivative actions and dot-product gates.

## 2026-06-25 Matrix-Free Provider For Ambipolar Option 1/3 Gates

Steps taken:

1. Added `matrix_free_radial_current_derivative_provider` as a reusable bridge
   from a matrix-free linear-observable builder to the ambipolar derivative
   provider contract.
2. Added a fast nonlinear linear-observable current test where both
   safeguarded Newton/bisection and pure Newton consume that provider through
   the public root-solver API.
3. Updated the feature matrix to distinguish this closed provider gate from the
   still-open physical Fortran option-1/3 RHSMode-1 replay gates.

Results:

- Combined ambipolar/sensitivity/import gates passed:
  `python -m pytest tests/test_ambipolar_problem.py tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `37 passed`.

Next best steps:

1. Build analytic `Er` operator tangents from namelist radial-coordinate
   conversion and compare them to the current centered-difference operator
   tangent on no-Phi1 decks.
2. Use the analytic/JVP provider in one real RHSMode-1 option-1/3 physical
   parity gate against checked Fortran v3 helical summaries.
3. Extend the same provider path to Phi1 drift-current branches.

## 2026-06-25 Analytic Existing-Branch Er Operator Tangents

Steps taken:

1. Added `dphi_hat_dpsi_hat_er_derivative_from_namelist`, which reuses the v3
   radial-coordinate conversion to compute `d(dPhiHat/dpsiHat)/dEr`.
2. Added `er_operator_tangent_from_dphi_hat_dpsi_hat_derivative`, which builds
   a fixed-shape operator tangent by updating existing `dphi_hat_dpsi_hat`
   leaves in the full-system operator and f-block suboperators.
3. Added a real no-Phi1 electric-field `xDot` fixture gate comparing the
   analytic tangent JVP action against centered operator differences.

Results:

- Focused sensitivity/import gates passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `23 passed`.

Next best steps:

1. Promote a fixed-shape zero-`Er` branch policy so differentiating at
   `Er=0` does not drop electric-field suboperators.
2. Use the analytic/JVP provider in one real RHSMode-1 option-1/3 physical
   parity gate against checked Fortran v3 summaries.
3. Add Phi1 drift-current tangents after the no-Phi1 physical gate is stable.

## 2026-06-25 Opt-In Fixed-Shape Zero-Er Branch Retention

Steps taken:

1. Added `keep_zero_er_terms` to `fblock_operator_from_namelist` and
   `full_system_operator_from_namelist`.
2. Preserved default solve behavior while allowing derivative gates to retain
   zero-valued ExB and `Er` suboperators at `Er=0`.
3. Added a real `xDot` fixture gate showing the opt-in zero-`Er` operator is
   numerically identical to the default zero-`Er` operator, while its analytic
   JVP tangent matches centered nonzero-`Er` operator differences.

Results:

- Focused sensitivity/import gates passed:
  `python -m pytest tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `24 passed`.

Next best steps:

1. Use `keep_zero_er_terms=True` plus the analytic/JVP provider in one real
   RHSMode-1 option-1/3 physical parity gate against checked Fortran summaries.
2. Add an API-level helper that builds the derivative provider from a namelist
   without requiring users to manually assemble plus/minus operators.
3. Extend the same fixed-shape approach to Phi1 drift-current derivatives.

## 2026-06-25 Namelist-Backed RHSMode 1 Er Derivative Response

Steps taken:

1. Added `RHSMode1RadialCurrentResponse` and
   `rhsmode1_radial_current_response_from_namelist` to the ambipolar problem
   owner and package exports.
2. The helper builds fixed-shape RHSMode-1 operators directly from a namelist,
   preserves zero-valued `Er` branches for derivative gates, supplies the
   analytic/JVP `Er` operator tangent, and accepts caller-supplied production
   solve/transpose-solve closures while defaulting to bounded dense validation
   on small decks.
3. Added a real small-deck gate showing the implicit derivative agrees with a
   centered radial-current finite difference through the new public helper.
4. Updated the feature matrix, release notes, and final plan so the remaining
   Lane 3 item is now the Fortran option-1/3 physical replay, not manual
   provider assembly.

Results:

- Focused helper validation passed:
  `python -m pytest
  tests/test_sensitivity.py::test_rhsmode1_namelist_response_uses_fixed_shape_er_derivative_provider
  -q --tb=short` with `1 passed`.

Next best steps:

1. Use the namelist-backed provider in one real derivative-assisted
   ambipolar option-1/3 physical replay gate against the checked Fortran
   summaries.
2. Extend JVP/VJP dot-product gates to Phi1 drift-current, total heat-flux, and
   intermediate-grid diagnostics.
3. Add small Fortran RHSMode 4/5 output fixtures once the no-Phi1 physical
   derivative replay is stable.

## 2026-06-25 Active Fortran-Style Option-1 Ambipolar Derivative Replay

Steps taken:

1. Audited Fortran v3 `ambipolarSolver.F90`, `diagnostics.F90`, and
   `adjointDiagnostics.F90` for the derivative-assisted option-1 current
   contract.
2. Found that Fortran's root scalar is
   `sum_s Z_s particleFlux_vm_rN_s`, and that small collisionless
   `constraintScheme=1` RHSMode-1 decks must be solved in the active
   compressed pitch-mode ordering rather than the rectangular storage layout.
3. Updated the bounded namelist-backed validation backend to assemble and solve
   the active reduced operator, scatter solutions back to full diagnostic
   ordering, infer radial-current conversion factors from the namelist, and
   default the ambipolar helper to the Fortran `rN` current convention.
4. Added a checked `geometry1_helical_small_option1` regression that replays
   the Fortran active current and Newton slope.

Results:

- Small Fortran option-1 replay at `Er = [-20, 20, 0, -2.01579684708909]`
  now matches radial currents with relative errors `2.8e-10`, `4.9e-9`,
  `6.1e-7`, and `1.9e-6`.
- The active implicit derivative at `Er=0` is `5.3516725e-7`, matching the
  Fortran implied Newton slope `5.3516663e-7` within the new `2e-5` gate.
- Focused gates passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_sensitivity.py
  tests/test_ambipolar_problem.py tests/test_domain_package_import_contracts.py
  -q --tb=short` with `41 passed in 45.93 s`.

Next best steps:

1. Promote the provider into the in-process ambipolar option-1/3 root driver
   for bounded small decks, keeping Brent as the robust CLI fallback.
2. Run the production option-1/3 decks with bounded setup reuse and add replay
   artifacts outside normal CI.
3. Start the RHSMode 4 fixed-Er output fixture lane using the now-validated
   active operator/transpose contract.

## 2026-06-25 Active Fortran-Style Option-3 Ambipolar Current Replay

Steps taken:

1. Replayed the checked `geometry1_helical_small_option3` and
   `geometry4_w7x_like_small_option3` Fortran v3 current points with the active
   namelist-backed RHSMode-1 response.
2. Added a lightweight regression that checks those option-3 physical currents
   without adding another expensive derivative solve.

Results:

- Helical option-3 current replay matches Fortran at `Er=0` and the Newton
  update point with relative errors `6.1e-7` and `1.9e-6`.
- W7-X-like option-3 `Er=0` current replay matches Fortran with relative error
  `5.7e-8`.
- Focused sensitivity gate passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_sensitivity.py -q
  --tb=short` with `20 passed in 60.98 s`.

Next best steps:

1. Promote the active provider into the in-process ambipolar option-1/3 root
   driver for bounded small decks.
2. Run production option-1/3 replay outside normal CI and record setup reuse,
   residual, runtime, and RSS.
3. Start RHSMode 4 fixed-Er fixture design using the active operator/transpose
   contract now validated by option-1/3.

## 2026-06-25 Bounded Option-1/3 Ambipolar Root Replay

Steps taken:

1. Added `solve_rhsmode1_ambipolar_from_namelist` as the bounded
   namelist-backed RHSMode-1 ambipolar driver for options 1, 2, and 3.
2. Wired the active `particleFlux_vm_rN` response into the real safeguarded
   Newton and pure-Newton root solvers, including current-evaluation caching
   and Fortran-compatibility validation.
3. Exposed the helper through `sfincs_jax.problems` and added import-contract
   coverage.
4. Added physical small-deck replay tests for the checked helical option-1 and
   option-3 Fortran roots.

Results:

- Option 1 converges through the real safeguarded Newton path with one
  derivative evaluation and no bisection fallback.
- Option 3 converges through the real pure-Newton path using the same active
  namelist-backed derivative provider.
- Root electric fields replay the checked Fortran value
  `Er=-2.01579684708909` within the `2e-5` absolute gate, and final currents
  agree below `1e-12` absolute residual.
- Focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_ambipolar_problem.py
  tests/test_sensitivity.py tests/test_domain_package_import_contracts.py -q
  --tb=short` with `45 passed in 81.34 s`.

Current lane status:

- Ambipolar solver lane: 99% for bounded small decks; production option-1/3
  replay with setup reuse remains outside normal CI.
- RHSMode 4/5 sensitivity lane: 72%; the active operator/transpose contract is
  now stable enough to start fixed-Er fixture output work.
- Refactor/review-ready PR lane: 85%; the problem-domain API surface is cleaner
  and the helper no longer requires direct `v3_driver.py` coupling.
- Overall completion: about 87%.

Next best steps:

1. Start the RHSMode 4 fixed-Er fixture lane with small Fortran outputs and
   compare exported sensitivity diagnostics.
2. Run production option-1/3 replay outside CI with setup-reuse timing/RSS
   summaries.
3. Continue the refactor/review tranche by moving sensitivity fixture support
   behind the domain API rather than adding new `v3_driver.py` entry points.

## 2026-06-25 RHSMode 4/5 Source-Contract Gate

Steps taken:

1. Audited Fortran v3 `validateInput.F90`, `solver.F90`,
   `adjointDiagnostics.F90`, `testingAdjointDiagnostics.F90`, and
   `writeHDF5Output.F90` for RHSMode 4/5 adjoint restrictions and output
   fields.
2. Added `validate_fortran_v3_adjoint_sensitivity_constraints` to
   `sfincs_jax.sensitivity`, mirroring the Fortran source restrictions for
   no Phi1, no inductive field, no tangential magnetic drifts,
   `constraintScheme=-1/1`, `collisionOperator=0`, Boozer-coordinate geometry
   restrictions, and RHSMode-5 radial-current sensitivity rejection.
3. Added `fortran_v3_adjoint_sensitivity_output_fields` to pin the HDF5
   sensitivity field names written by `writeHDF5Output.F90`, including the
   source-code quirk that `dParallelFlowdLambda` is gated by
   `adjointParticleFluxOption` or `debugAdjoint`.
4. Added focused tests for valid RHSMode-4 contracts, invalid source-level
   combinations, RHSMode-5 debug output fields, and import-contract coverage.
5. Updated release notes, validation docs, performance docs, feature matrix,
   and `plan_final.md`.

Results:

- Focused contract/import tests passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_constraints_and_output_fields
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_constraints_reject_invalid_source_combinations
  tests/test_sensitivity.py::test_fortran_v3_adjoint_output_fields_preserve_parallel_flow_source_gate
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `11 passed in 0.60 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_sensitivity.py
  tests/test_ambipolar_problem.py tests/test_domain_package_import_contracts.py
  -q --tb=short` with `48 passed in 63.27 s`.
- `python -m ruff check sfincs_jax/sensitivity.py tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py`, `git diff --check`, and
  `python -m sphinx -b html docs docs/_build/html -q` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; production replay remains outside CI.
- RHSMode 4/5 sensitivity lane: 78%; source-contract and generic derivative
  spine are now tested, numerical Fortran output replay remains.
- Refactor/review-ready PR lane: 86%; the sensitivity contract lives in the
  domain-neutral sensitivity module without adding another file.
- Overall completion: about 88%.

Next best steps:

1. Create small RHSMode-4 Fortran fixture namelists and output summaries for
   one or two sensitivity targets.
2. Add sfincs_jax numerical output-field scaffolding only after those fixture
   values are pinned.
3. Continue the review-ready refactor tranche by moving fixture generation and
   comparison helpers behind compact domain APIs.

## 2026-06-25 Shared Input-Contract Refactor

Steps taken:

1. Moved duplicated case-insensitive namelist/nested-mapping lookup helpers
   from `sfincs_jax.problems.ambipolar` and `sfincs_jax.sensitivity` into
   `sfincs_jax.input_compat`.
2. Rewired the ambipolar and RHSMode 4/5 sensitivity validators to use the
   shared helpers.
3. Added direct `input_compat` coverage for scalar defaults, vector boolean
   coercion, and nested mapping/Namelist lookup.

Results:

- The refactor removes duplicate local helper implementations while preserving
  the validated ambipolar and RHSMode 4/5 source-contract behavior.
- Focused refactor tests passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `71 passed in 67.78 s`.
- `python -m ruff check sfincs_jax/input_compat.py sfincs_jax/sensitivity.py
  sfincs_jax/problems/ambipolar.py tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py`, `git diff --check`, and
  `python -m sphinx -b html docs docs/_build/html -q` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; production replay remains outside CI.
- RHSMode 4/5 sensitivity lane: 78%; unchanged technically, but the input
  contract is now shared instead of duplicated.
- Refactor/review-ready PR lane: 87%; common Fortran-style option lookup now
  has one owner.
- Overall completion: about 89%.

Next best steps:

1. Create small RHSMode-4 Fortran fixture namelists/output summaries.
2. Add numerical output-field comparison scaffolding after those fixtures are
   pinned.
3. Continue extracting duplicated policy/config glue into existing domain
   modules only when it reduces driver/module complexity.

## 2026-06-25 First RHSMode-4 Fortran Sensitivity Fixture

Steps taken:

1. Built scratch RHSMode-4 decks from the checked small ambipolar references.
2. Rejected the helical geometry-1 deck as a fixture because SFINCS Fortran v3
   reports `Adjoint not compatible with stellarator asymmety`.
3. Reran the symmetric `geometryScheme=4` W7-X-like analytic deck with
   `adjointRadialCurrentOption=.true.` and particle-flux adjoint options.
4. Checked in only a compact namelist and JSON summary under
   `benchmarks/fortran_v3_sensitivity_reference`, not the generated HDF5 file.
5. Added a regression that validates the namelist contract, expected HDF5 field
   names, wall/RSS budget, and
   `dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`.

Results:

- Fortran v3 W7-X-like RHSMode-4 fixture completed with wall time `0.11 s`,
  main solve time `0.0355 s`, two adjoint solves of about `0.0019 s` and
  `0.0012 s`, and peak RSS `122,994,688` bytes.
- The compact summary pins `dParticleFluxdLambda`, `dParallelFlowdLambda`, and
  `dRadialCurrentdLambda` shapes and values.
- Focused fixture gate passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_constraints_and_output_fields
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity
  -q --tb=short` with `2 passed in 0.46 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `72 passed in 65.22 s`.
- `python -m sphinx -b html docs docs/_build/html -q`, ruff, and
  `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; production replay remains outside CI.
- RHSMode 4/5 sensitivity lane: 82%; first numerical Fortran RHSMode-4 summary
  is checked in, with more observables/RHSMode-5 still open.
- Refactor/review-ready PR lane: 87%; unchanged by this fixture except for
  cleaner reference organization.
- Overall completion: about 90%.

Next best steps:

1. Add one more RHSMode-4 fixture targeting heat flux or bootstrap current.
2. Add the first sfincs_jax output-surface scaffolding for RHSMode-4 fields
   once at least two Fortran summaries are pinned.
3. Keep production option-1/3 replay and RHSMode-5 constant-current fixtures
   outside normal CI until the small fixture layer is stable.

## 2026-06-25 Second RHSMode-4 Fixture And Output-Surface Contract

Steps taken:

1. Reused the checked W7-X-like analytic RHSMode=4 deck and reran SFINCS
   Fortran v3 with `adjointHeatFluxOption=.true. .true.` and
   `adjointTotalHeatFluxOption=.true.`.
2. Checked in only the compact heat-flux namelist and JSON summary under
   `benchmarks/fortran_v3_sensitivity_reference`; the generated HDF5 file
   remains a scratch artifact.
3. Added `fortran_v3_adjoint_sensitivity_output_ranks` and
   `validate_fortran_v3_adjoint_sensitivity_output_surface` to
   `sfincs_jax.sensitivity`, so RHSMode=4/5 output field names and tensor
   ranks can be validated against either HDF5-like arrays or lightweight
   summaries.
4. Added tests pinning both compact Fortran RHSMode=4 summaries and the
   heat-flux identity `dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`.
5. Updated release notes, validation docs, feature matrix, and
   `plan_final.md`.

Results:

- Fortran v3 W7-X-like heat-flux RHSMode=4 fixture completed with wall time
  `0.08 s`, main solve time `0.028633 s`, two adjoint solves of about
  `0.0018 s` and `0.0011 s`, and peak RSS `125,288,448` bytes.
- The compact summary pins `dHeatFluxdLambda` and `dTotalHeatFluxdLambda`
  shapes and values.
- Narrow fixture gate passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_heat_flux_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_output_surface_reports_missing_or_misranked_fields
  -q --tb=short` with `3 passed in 0.29 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `74 passed in 65.69 s`.
- `python -m ruff check sfincs_jax/sensitivity.py tests/test_sensitivity.py
  tests/test_domain_package_import_contracts.py`, `git diff --check`,
  `python -m json.tool
  benchmarks/fortran_v3_sensitivity_reference/small_rhsmode45_summary_2026-06-25.json`,
  and `python -m sphinx -b html docs docs/_build/html -q` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; production replay remains outside CI.
- RHSMode 4/5 sensitivity lane: 86%; two RHSMode=4 Fortran numerical output
  families are pinned and the reusable output-surface contract exists.
- Refactor/review-ready PR lane: 88%; the sensitivity surface stayed in the
  existing domain module and did not add a new file.
- Overall completion: about 91%.

Next best steps:

1. Run the broader ambipolar/sensitivity/import/docs validation slice.
2. Add a compact RHSMode=5 constant-current fixture once the RHSMode=4 surface
   remains stable.
3. Return to the refactor/review-ready PR lane by moving the next duplicated
   compatibility glue into existing domain modules only when it reduces total
   complexity.

## 2026-06-25 RHSMode-5 Constant-Current Sensitivity Fixture

Steps taken:

1. Reviewed SFINCS Fortran v3 `ambipolarSolver.F90`, `solver.F90`,
   `adjointDiagnostics.F90`, `validateInput.F90`, and `writeHDF5Output.F90`
   for the RHSMode=5 path.
2. Built a small RHSMode=5 W7-X-like heat-flux deck from the checked
   geometry-4 Brent ambipolar fixture, with `ambipolarSolve=.true.`,
   `ambipolarSolveOption=2`, `adjointHeatFluxOption=.true. .true.`, and
   `adjointTotalHeatFluxOption=.true.`.
3. Confirmed Fortran v3 first finds the ambipolar `E_r` using RHSMode=1, then
   re-enters RHSMode=5 at that root and writes the constant-current
   `dPhidPsidLambda` sensitivity output.
4. Renamed the compact summary file to
   `benchmarks/fortran_v3_sensitivity_reference/small_rhsmode45_summary_2026-06-25.json`
   because it now contains both RHSMode=4 and RHSMode=5 fixtures.
5. Added a regression test that validates the RHSMode=5 namelist/source
   contract, required output fields/ranks, finite nonzero `dPhidPsidLambda`,
   wall/RSS budgets, and
   `dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`.

Results:

- Fortran v3 W7-X-like RHSMode=5 heat-flux fixture completed with wall time
  `0.18 s`, ambipolar solve time `0.087104 s`, main solve times around
  `0.026-0.028 s`, three adjoint solves between `0.0012 s` and `0.0019 s`,
  and peak RSS `138,674,176` bytes.
- The HDF5 output contained `dHeatFluxdLambda`, `dTotalHeatFluxdLambda`, and
  `dPhidPsidLambda`; `dPhidPsidLambda` had shape `[1, 4, 1]` and maximum
  magnitude about `3.77`.
- Narrow fixture gate passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_heat_flux_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs5_reference_summary_pins_constant_current_heat_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_output_surface_reports_missing_or_misranked_fields
  -q --tb=short` with `4 passed in 0.38 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `75 passed in 93.49 s`.
- `python -m ruff check tests/test_sensitivity.py`, `git diff --check`,
  `python -m json.tool
  benchmarks/fortran_v3_sensitivity_reference/small_rhsmode45_summary_2026-06-25.json`,
  and `python -m sphinx -b html docs docs/_build/html -q` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; production replay remains outside CI.
- RHSMode 4/5 sensitivity lane: 90%; RHSMode=4 radial-current/heat and
  RHSMode=5 constant-current heat output families are pinned. Remaining work is
  bootstrap/flow/debug finite-difference fixtures and production-grid parity.
- Refactor/review-ready PR lane: 88%; unchanged except for the compact
  benchmark fixture rename and tests.
- Overall completion: about 92%.

Next best steps:

1. Run focused sensitivity/import/docs validation and commit/push the RHSMode=5
   fixture tranche.
2. Start the refactor/review-ready PR lane by auditing the next large
   `v3_driver.py` compatibility cluster for removal or domain-package
   consolidation.
3. Keep full production RHSMode=4/5 solve parity outside normal CI until the
   compact source/output fixture layer is complete.

## 2026-06-25 Refactor Tranche: Preconditioner Compatibility Wrappers

Steps taken:

1. Audited the remaining `v3_driver.py` helper surface after the RHSMode=4/5
   fixture work. The file was still `12,242` lines and dominated by two large
   solve functions plus compatibility glue.
2. Removed four RHSMode=2/3 transport domain-decomposition wrapper bodies and
   replaced them with aliases to the equivalent RHSMode=1 line/block builder
   kernels. This preserves the legacy monkeypatch/debug names while eliminating
   duplicate code.
3. Replaced six repeated PAS-family compatibility wrappers with one small
   `_rhs1_pas_family_compat_builder` factory and named compatibility exports.
   No new files were added.

Results:

- `v3_driver.py` decreased from `12,242` to `12,133` lines in this tranche.
- The diff is a net simplification of one file:
  `sfincs_jax/v3_driver.py | 207 ++++++++++++------------------------------------`.
- `python -m py_compile sfincs_jax/v3_driver.py` passed.
- Focused dispatch/PAS validation passed:
  `python -m pytest tests/test_transport_preconditioner_dispatch.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py -q --tb=short`
  with `80 passed in 39.89 s`.
- Transport alias parity validation passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_transport_parallel.py::test_transport_theta_dd_preconditioner_matches_default
  tests/test_transport_parallel.py::test_transport_theta_schwarz_preconditioner_matches_default
  -q --tb=short` with `2 passed in 18.26 s`.
- `python -m ruff check sfincs_jax/v3_driver.py` and `git diff --check`
  passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 90%; no change after the RHSMode=5 fixture
  commit.
- Refactor/review-ready PR lane: 89%; one more compatibility cluster was
  simplified without increasing file count.
- Overall completion: about 92%.

Next best steps:

1. Commit and push this refactor tranche.
2. Continue the refactor/review lane with a larger audit of the two monolithic
   solve functions, identifying seams that can be moved into existing
   `profile_response` or `transport_matrix` domain modules without adding new
   top-level files.
3. Run a broader import/driver slice after the next tranche, not after every
   small wrapper cleanup.

## 2026-06-25 RHSMode-4 Parallel-Flow And Bootstrap Fixture

Steps taken:

1. Built a compact W7-X-like RHSMode=4 fixture with
   `adjointParallelFlowOption=.true. .true.` and
   `adjointBootstrapOption=.true.`.
2. Also enabled `adjointParticleFluxOption=.true. .true.` because SFINCS
   Fortran v3 writes `dParallelFlowdLambda` only when particle-flux adjoints or
   debug adjoints are active.
3. Reran SFINCS Fortran v3 locally and checked in only the compact namelist and
   JSON summary, not the generated HDF5 file.
4. Added a regression that validates the namelist/source contract, field names,
   tensor ranks, wall/RSS budgets, and
   `dBootstrapdLambda = sum_s Z_s dParallelFlow_s/dLambda`.
5. Updated the feature matrix, release notes, validation matrix, benchmark
   README, and `plan_final.md`.

Results:

- Fortran v3 W7-X-like RHSMode=4 parallel/bootstrap fixture completed with
  wall time `0.27 s`, main solve time `0.046349 s`, four adjoint solves between
  `0.0018 s` and `0.0046 s`, and peak RSS `125,599,744` bytes.
- The HDF5 output contained `dParticleFluxdLambda`, `dParallelFlowdLambda`,
  and `dBootstrapdLambda`.
- The bootstrap charge-sum identity matched to `3.7e-18` in the scratch HDF5
  extraction.
- Narrow fixture gate passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_heat_flux_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs4_reference_summary_pins_parallel_flow_and_bootstrap_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_rhs5_reference_summary_pins_constant_current_heat_sensitivity
  tests/test_sensitivity.py::test_fortran_v3_adjoint_sensitivity_output_surface_reports_missing_or_misranked_fields
  -q --tb=short` with `5 passed in 0.47 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `76 passed in 93.94 s`.
- `python -m ruff check tests/test_sensitivity.py`, `python -m json.tool
  benchmarks/fortran_v3_sensitivity_reference/small_rhsmode45_summary_2026-06-25.json`,
  `git diff --check`, and `python -m sphinx -b html docs docs/_build/html -q`
  passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 93%; compact RHSMode=4 radial-current,
  heat-flux, parallel-flow, bootstrap, and RHSMode=5 constant-current heat
  output families are pinned. Remaining work is debug finite-difference and
  production-grid parity.
- Refactor/review-ready PR lane: 89%; no change in this fixture tranche.
- Overall completion: about 93%.

Next best steps:

1. Commit and push this fixture tranche.
2. Add a compact debug-adjoint finite-difference fixture only if it remains
   bounded; otherwise document it as a nightly/prolonged gate.
3. Continue the review-ready refactor lane by extracting one larger solve-loop
   seam into existing domain modules.

## 2026-06-25 RHSMode-4 Debug Finite-Difference Fixture

Steps taken:

1. Probed a smaller debug-adjoint deck and rejected it because SFINCS Fortran
   v3 requires `Ntheta >= 5`.
2. Reran the checked small W7-X-like RHSMode=4 radial-current fixture with
   `debugAdjoint=.true.`.
3. Checked in a compact debug namelist and a separate lightweight JSON summary
   with all debug output field shapes plus selected analytic, finite-
   difference, and percent-error values.
4. Extended `fortran_v3_adjoint_sensitivity_output_ranks` so debug percent-
   error fields have checked tensor ranks.
5. Added a regression that validates the full debug output surface, selected
   finite-difference values, finite percent-error bounds, and the Fortran NaN
   mask for unfilled lambda/mode entries.
6. Updated release notes, validation docs, feature matrix, benchmark README,
   and `plan_final.md`.

Results:

- Fortran v3 W7-X-like RHSMode=4 debug fixture completed with wall time
  `0.25 s`, finite-difference time `0.110304 s`, five main solve times around
  `0.026-0.036 s`, and peak RSS `144,588,800` bytes.
- The debug HDF5 output included all analytic sensitivity fields, all
  `_finitediff` fields, and all percent-error fields.
- Fortran leaves some finite-difference lambda/mode entries as NaN; the JSON
  summary records those as `null`, and the regression pins that mask.
- Full sensitivity module passed:
  `JAX_ENABLE_X64=True python -m pytest
  tests/test_sensitivity.py::test_fortran_v3_rhs4_debug_reference_summary_pins_finite_difference_outputs
  tests/test_sensitivity.py -q --tb=short` with `29 passed in 44.36 s`.
- Broader focused validation passed:
  `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_sensitivity.py tests/test_ambipolar_problem.py
  tests/test_domain_package_import_contracts.py -q --tb=short` with
  `77 passed in 66.48 s`.
- `python -m ruff check sfincs_jax/sensitivity.py tests/test_sensitivity.py`,
  JSON validation for both RHSMode 4/5 summaries, `git diff --check`, and
  `python -m sphinx -b html docs docs/_build/html -q` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 95%; compact output families and debug
  finite-difference gates are pinned. Remaining work is intermediate and
  production-grid parity plus first-class public RHSMode 4/5 solve output
  generation.
- Refactor/review-ready PR lane: 89%; no change in this fixture tranche.
- Overall completion: about 94%.

Next best steps:

1. Commit and push this debug fixture tranche.
2. Return to the refactor/review-ready PR lane and extract one larger
   solve-loop seam into existing domain modules.
3. Add intermediate RHSMode=4/5 parity gates only after the API surface for
   writing those outputs is stable.

## 2026-06-25 Transport Strong-Preconditioner Cache Refactor

Steps taken:

1. Reviewed the RHSMode=2/3 transport solve-loop seam in `v3_driver.py` and
   selected the strong-preconditioner lazy-cache closure because it belongs to
   transport preconditioner dispatch rather than driver orchestration.
2. Added `TransportStrongPreconditionerCache` to
   `sfincs_jax/problems/transport_matrix/preconditioner_dispatch.py`.
3. Replaced the nested full/reduced cache closure in
   `solve_v3_transport_matrix_linear_gmres` with the new domain helper while
   preserving lazy construction, reduced/full separation, and primary
   preconditioner reuse.
4. Added focused tests that verify each full/reduced strong preconditioner is
   built once and that same-kind strong preconditioners reuse the already-built
   primary preconditioner without dispatching another builder.

Results:

- `v3_driver.py` dropped from `12,133` to `12,112` lines in this tranche.
- `python -m pytest tests/test_transport_preconditioner_dispatch.py -q --tb=short`
  passed with `30 passed in 0.07 s`.
- `python -m pytest tests/test_transport_preconditioner_dispatch.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py -q --tb=short` passed
  with `82 passed in 24.58 s`.
- `JAX_ENABLE_X64=True python -m pytest
  tests/test_transport_parallel.py::test_transport_theta_dd_preconditioner_matches_default
  tests/test_transport_parallel.py::test_transport_theta_schwarz_preconditioner_matches_default
  -q --tb=short` passed with `2 passed in 12.55 s`.
- `python -m ruff check
  sfincs_jax/problems/transport_matrix/preconditioner_dispatch.py
  sfincs_jax/v3_driver.py tests/test_transport_preconditioner_dispatch.py`
  and `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 95%; no change in this tranche.
- Refactor/review-ready PR lane: 90%; the transport strong-preconditioner
  cache is now in a tested domain module, but the main solve loops still need
  larger protocol-level extractions before the PR is review-ready.
- Overall completion: about 94%.

Next best steps:

1. Commit and push this refactor tranche to both the active branch and the
   draft PR branch.
2. Extract the next RHSMode=2/3 seam with real ownership, preferably sparse
   direct setup or per-RHS linear-solve orchestration, into existing transport
   modules.
3. Keep reducing driver-level policy and cache code without adding new narrow
   files unless a domain boundary requires it.

## 2026-06-25 Transport Sparse-Direct Context Refactor

Steps taken:

1. Reviewed the RHSMode=2/3 sparse-direct rescue setup in `v3_driver.py` and
   confirmed that environment parsing, factor-cache creation, and pattern-cache
   creation belong with the sparse-direct transport module.
2. Added `transport_sparse_direct_context_from_env` to
   `sfincs_jax/problems/transport_matrix/sparse_direct_solve.py`.
3. Replaced the inline `v3_driver.py` sparse-drop parsing and context
   construction with that helper, preserving the same per-solve fresh cache
   semantics and driver-provided callback hooks.
4. Added a focused sparse-direct test that checks invalid-env fallback,
   numeric env parsing, and independent per-context caches.

Results:

- `v3_driver.py` dropped from `12,112` to `12,097` lines in this tranche.
- `python -m pytest tests/test_transport_sparse_direct_solve.py -q --tb=short`
  passed with `3 passed in 0.21 s`.
- `python -m pytest tests/test_transport_sparse_direct_solve.py
  tests/test_transport_preconditioner_dispatch.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py -q --tb=short` passed
  with `85 passed in 24.21 s`.
- `JAX_ENABLE_X64=True python -m pytest
  tests/test_transport_parallel.py::test_transport_theta_dd_preconditioner_matches_default
  tests/test_transport_parallel.py::test_transport_theta_schwarz_preconditioner_matches_default
  -q --tb=short` passed with `2 passed in 12.35 s`.
- `python -m ruff check
  sfincs_jax/problems/transport_matrix/sparse_direct_solve.py
  sfincs_jax/v3_driver.py tests/test_transport_sparse_direct_solve.py` and
  `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 95%; no change in this tranche.
- Refactor/review-ready PR lane: 91%; transport preconditioner cache and
  sparse-direct context setup are now owned by tested transport modules.
- Overall completion: about 95%.

Next best steps:

1. Commit and push this sparse-direct refactor tranche.
2. Extract per-RHS transport linear-solve orchestration into the existing
   `transport_linear_solve` or `transport_solve_finalization` domain modules.
3. Continue avoiding new micro-files; consolidate around owner modules already
   present in the transport package.

## 2026-06-25 Transport Linear-Solve Callback Refactor

Steps taken:

1. Reviewed the per-RHS RHSMode=2/3 transport loop and identified the local
   `_solve_linear` and `_solve_linear_with_residual` wrappers as driver glue
   over `TransportLinearSolveContext`.
2. Added `TransportLinearSolveCallbacks` to
   `sfincs_jax/problems/transport_matrix/linear_solve.py`, binding the context
   once and exposing `solve` and `solve_with_residual` callbacks.
3. Replaced the two local wrapper functions in `v3_driver.py` with the bound
   callback methods.
4. Added a focused unit test proving the callback object preserves context
   routing and preconditioner-side forwarding.

Results:

- `v3_driver.py` dropped from `12,097` to `12,046` lines in this tranche.
- `python -m pytest tests/test_transport_linear_solve.py -q --tb=short`
  passed with `5 passed in 0.63 s`.
- `python -m pytest tests/test_transport_linear_solve.py
  tests/test_transport_sparse_direct_solve.py
  tests/test_transport_preconditioner_dispatch.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py -q --tb=short` passed
  with `90 passed in 28.34 s`.
- `JAX_ENABLE_X64=True python -m pytest
  tests/test_transport_parallel.py::test_transport_theta_dd_preconditioner_matches_default
  tests/test_transport_parallel.py::test_transport_theta_schwarz_preconditioner_matches_default
  -q --tb=short` passed with `2 passed in 12.32 s`.
- `python -m ruff check
  sfincs_jax/problems/transport_matrix/linear_solve.py sfincs_jax/v3_driver.py
  tests/test_transport_linear_solve.py` and `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 99% bounded; no change in this tranche.
- RHSMode 4/5 sensitivity lane: 95%; no change in this tranche.
- Refactor/review-ready PR lane: 92%; transport preconditioner cache,
  sparse-direct setup, and linear-solve callbacks are now domain-owned and
  tested.
- Overall completion: about 95%.

Next best steps:

1. Commit and push this linear-solve refactor tranche.
2. Extract dense-batch setup or transport RHS finalization setup next, using
   existing modules rather than adding new namespaced files.
3. Run a broader smoke slice after the next tranche because the refactor will
   start touching solve-loop finalization and diagnostic output ownership.

## 2026-06-25 Ambipolar Production Metadata Closure Gate

Steps taken:

1. Re-reviewed the checked Fortran-v3 ambipolar small and production summaries
   against the remaining `plan_final.md` ambipolar lane.
2. Confirmed that sfincs_jax already has in-process option-1/2/3 root solvers,
   small real-deck option-1/3 gates, Brent replay gates, fixed-shape setup
   reuse metadata, and CLI serialization tests.
3. Added a production option-1/3 metadata gate that pins the derivative-solve
   coverage from the Fortran-v3 production references: one adjoint derivative
   solve per physical solve, Newton success markers, MUMPS package provenance,
   PETSc profile markers, nonzero Jacobian/preconditioner sizes, RSS/timing
   provenance, and final radial-current bounds.

Results:

- `python -m pytest tests/test_ambipolar_problem.py -q --tb=short` passed with
  `18 passed in 25.61 s`.
- `JAX_ENABLE_X64=True python -m pytest tests/test_ambipolar_problem.py
  tests/test_sensitivity.py tests/test_domain_package_import_contracts.py
  -q --tb=short` passed with `55 passed in 63.44 s`.
- `python -m ruff check tests/test_ambipolar_problem.py` and
  `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 100% closed for the bounded/reference PR scope.
  Heavy production reruns remain release-refresh artifacts, not CI gates.
- RHSMode 4/5 sensitivity lane: 95%; unchanged in this tranche.
- Refactor/review-ready PR lane: 92%; unchanged in this tranche.
- Overall completion: about 96%.

Next best steps:

1. Commit and push this ambipolar closure gate.
2. Move to RHSMode 4/5 sensitivity lane closure by adding one more bounded
   source/output-surface gate or documenting the production-grid parity gate as
   an external release-refresh benchmark.
3. Resume refactor with another high-value owner-boundary extraction after the
   RHSMode 4/5 lane is updated.

## 2026-06-25 RHSMode 4/5 Bounded Fixture Coverage Closure

Steps taken:

1. Re-reviewed the checked RHSMode 4/5 Fortran-v3 sensitivity fixtures and
   confirmed they cover radial current, heat flux, total heat flux, particle
   flux, parallel flow, bootstrap current, constant-current `dPhidPsidLambda`,
   and debug finite-difference percent-error outputs.
2. Added an aggregate fixture coverage gate so future changes cannot drop a
   release-facing sensitivity output family while still passing individual
   per-case tests.
3. Updated the feature matrix and release notes to state the current status
   precisely: bounded/reference RHSMode 4/5 coverage is implemented and tested;
   production-grid parity is an external release-refresh benchmark, not a
   normal CI gate.

Results:

- `JAX_ENABLE_X64=True python -m pytest tests/test_sensitivity.py -q
  --tb=short` passed with `30 passed in 45.51 s`.
- `JAX_ENABLE_X64=True python -m pytest tests/test_sensitivity.py
  tests/test_ambipolar_problem.py tests/test_domain_package_import_contracts.py
  -q --tb=short` passed with `56 passed in 64.75 s`.
- `python -m sphinx -b html docs docs/_build/html -q`,
  `python -m ruff check tests/test_sensitivity.py sfincs_jax/sensitivity.py`,
  and `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 100% closed for the bounded/reference PR scope.
- RHSMode 4/5 sensitivity lane: 100% closed for the bounded/reference PR scope.
  Heavy production-grid parity remains a release-refresh benchmark.
- Refactor/review-ready PR lane: 92%; next active lane.
- Overall completion: about 97%.

Next best steps:

1. Commit and push this RHSMode 4/5 closure tranche.
2. Return to the refactor/review-ready PR lane and continue reducing
   `v3_driver.py` through high-value owner-boundary extractions only.
3. Run a broader fast test slice after the next refactor tranche and then
   prepare the draft PR for final review-readiness checks.

## 2026-06-25 Transport Constraint-Projection Owner Refactor

Steps taken:

1. Re-inventoried the remaining nested helpers in `v3_driver.py` after the
   ambipolar and RHSMode 4/5 closure work.
2. Moved the RHSMode=2/3 constraint-nullspace projection adapter into
   `sfincs_jax/problems/transport_matrix/finalize.py` as
   `TransportConstraintNullspaceProjector`.
3. Rewired `v3_driver.py` so dense-batch and RHS finalization paths receive
   `constraint_projector.project` instead of a local projection closure.
4. Updated projection tests to target the transport finalization owner instead
   of a private `v3_driver` alias.

Results:

- `v3_driver.py` dropped from `12,046` to `12,030` lines in this tranche.
- `python -m pytest tests/test_constraint_projection.py
  tests/test_transport_solve_finalization.py -q --tb=short` passed with
  `9 passed in 1.18 s`.
- `python -m pytest tests/test_transport_dense_batch.py
  tests/test_transport_linear_solve.py tests/test_transport_sparse_direct_solve.py
  tests/test_transport_preconditioner_dispatch.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py -q --tb=short` passed
  with `93 passed in 24.97 s`.
- `JAX_ENABLE_X64=True python -m pytest
  tests/test_transport_parallel.py::test_transport_theta_dd_preconditioner_matches_default
  tests/test_transport_parallel.py::test_transport_theta_schwarz_preconditioner_matches_default
  -q --tb=short` passed with `2 passed in 12.16 s`.
- `python -m ruff check
  sfincs_jax/problems/transport_matrix/finalize.py sfincs_jax/v3_driver.py
  tests/test_constraint_projection.py`, `python -m py_compile
  sfincs_jax/v3_driver.py`, and `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 100% closed for the bounded/reference PR scope.
- RHSMode 4/5 sensitivity lane: 100% closed for the bounded/reference PR scope.
- Refactor/review-ready PR lane: 93%; transport preconditioner caching,
  sparse-direct setup, linear-solve callbacks, and constraint projection now
  have tested domain owners.
- Overall completion: about 97%.

Next best steps:

1. Commit and push this projection refactor tranche.
2. Run one broader fast validation slice across ambipolar, sensitivity,
   transport, CLI import contracts, and docs.
3. Prepare a review-readiness checklist for the draft PR: clean worktree,
   no large generated artifacts, current branch pushed, plan statuses aligned,
   and remaining production refresh benchmarks explicitly documented.

## 2026-06-25 Broad Fast Validation After Lane Closures

Steps taken:

1. Ran the broader fast validation slice after closing bounded ambipolar,
   bounded RHSMode 4/5, and the latest transport projection refactor.
2. Built the Sphinx docs and checked whitespace.
3. Probed whole-repo ruff once. It still reports pre-existing broad lint debt
   outside the touched files, mostly JAX-config import-order patterns,
   ambiguous `l` loop/index names in older numerical modules, and a repeated
   dictionary key in `compare.py`. The scoped ruff checks for touched files
   passed in their tranches, so the broad lint debt is tracked as a separate
   cleanup item rather than mixed into this refactor/feature closure pass.

Results:

- `JAX_ENABLE_X64=True python -m pytest tests/test_input_compat.py
  tests/test_ambipolar_problem.py tests/test_sensitivity.py
  tests/test_constraint_projection.py tests/test_transport_solve_finalization.py
  tests/test_transport_dense_batch.py tests/test_transport_linear_solve.py
  tests/test_transport_sparse_direct_solve.py
  tests/test_transport_preconditioner_dispatch.py
  tests/test_domain_package_import_contracts.py
  tests/test_cli_validation_io_fast_coverage.py -q --tb=short` passed with
  `141 passed in 68.60 s`.
- `python -m sphinx -b html docs docs/_build/html -q` and `git diff --check`
  passed.
- Worktree was clean after the pushed commits.

Current lane status:

- Ambipolar solver lane: 100% closed for the bounded/reference PR scope.
- RHSMode 4/5 sensitivity lane: 100% closed for the bounded/reference PR scope.
- Refactor/review-ready PR lane: 93%; remaining work is review-readiness
  cleanup, not another new physics feature.
- Overall completion: about 97%.

Next best steps:

1. Commit and push this validation log.
2. Do the review-readiness checklist: verify no large generated files, summarize
   remaining deferred release-refresh benchmarks, and identify any final
   high-value `v3_driver.py` extraction that can be completed without new
   architectural churn.
3. Leave whole-repo lint cleanup as a scoped future lane unless the project
   wants to normalize all JAX-config import-order patterns now.

## 2026-06-25 Transport Sparse-Direct Owner Boundary

Steps taken:

1. Moved the RHSMode=2/3 sparse-direct sparse-pattern and solve adapters onto
   `TransportSparseDirectContext`.
2. Rewired `v3_driver.py` to call `transport_sparse_direct_context.solve(...)`
   directly instead of keeping local wrapper closures in the monolithic solve
   loop.
3. Added method-level tests so the sparse-direct owner boundary is pinned
   without relying only on the legacy driver path.

Results:

- `v3_driver.py` dropped from `12,030` to `11,992` lines in this tranche.
- `python -m pytest tests/test_transport_sparse_direct_solve.py -q
  --tb=short` passed with `5 passed in 0.21 s`.
- `python -m pytest tests/test_transport_sparse_direct_solve.py
  tests/test_transport_linear_solve.py tests/test_transport_dense_batch.py
  tests/test_transport_preconditioner_dispatch.py
  tests/test_transport_solve_finalization.py tests/test_constraint_projection.py
  -q --tb=short` passed with `52 passed in 1.82 s`.
- `python -m ruff check
  sfincs_jax/problems/transport_matrix/sparse_direct_solve.py
  sfincs_jax/v3_driver.py tests/test_transport_sparse_direct_solve.py`,
  `python -m py_compile sfincs_jax/v3_driver.py
  sfincs_jax/problems/transport_matrix/sparse_direct_solve.py`, and
  `git diff --check` passed.

Current lane status:

- Ambipolar solver lane: 100% closed for the bounded/reference PR scope.
- RHSMode 4/5 sensitivity lane: 100% closed for the bounded/reference PR scope.
- Refactor/review-ready PR lane: 94%; the remaining blocker is final PR #8
  CI/coverage status plus any concrete failures it exposes.
- Overall completion: about 98%.

Next best steps:

1. Commit and push this sparse-direct owner-boundary tranche.
2. Re-check PR #8 CI after the coverage shards finish.
3. If CI passes, finish the review-readiness audit; if CI fails, fix the
   failing job directly before doing more refactor work.

## 2026-06-25 Lane 1 Iteration 1 Compatibility Alias Deletion

Steps taken:

1. Rewrote internal, test, script, example, and docs imports from deleted
   top-level compatibility aliases to their canonical domain owners.
2. Deleted all 31 top-level `transport_*` compatibility modules.
3. Deleted all 17 top-level `rhs1_*` compatibility modules that already had
   canonical owners in `problems.profile_response` or `solvers.preconditioners`.
4. Updated API/testing/release documentation so it no longer claims the deleted
   aliases remain available.
5. Updated `plan_final.md` so Lane 1 Iteration 1 is marked complete and
   Iteration 2 is the next active consolidation batch.

Results:

- Python source files decreased from `295` to `247`.
- Top-level `transport_*` files decreased from `31` to `0`.
- Top-level `rhs1_*` files decreased from `64` to `47`; the remaining files
  are real RHSMode-1 implementation and are the target of Iteration 2.
- `v3_driver.py` remains `11,992` lines; the driver-size target is addressed in
  Lane 1 Iteration 4 after the real RHSMode-1 modules have canonical homes.

Validation:

- `python -m py_compile sfincs_jax/v3_driver.py
  sfincs_jax/problems/transport_matrix/parallel/runtime.py
  sfincs_jax/problems/transport_matrix/parallel/worker.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_policy_module_docstrings.py tests/test_helper_module_coverage.py
  -q --tb=short` passed with `20 passed in 1.99 s`.
- `python -m pytest` over the modified RHSMode-1 policy and transport module
  tests passed with `420 passed in 13.62 s`.
- Transport parity smoke and benchmark policy tests passed with
  `62 passed in 10.98 s`.
- `python -m sphinx -b html docs docs/_build/html -q`,
  scoped `ruff`, `git diff --check`, and the deleted-alias import scan passed.

Current lane status:

- Lane 1 Iteration 1: 100%.
- Lane 1 overall: about 20%; Iterations 2-5 remain.
- Overall consolidation goal: not complete.

Next best steps:

1. Commit and push Iteration 1.
2. Start Lane 1 Iteration 2 by moving real top-level RHSMode-1 modules into
   domain packages in one batch, preserving behavior and public API paths.

## 2026-06-25 Lane 1 Concrete Consolidation Map

Steps taken:

1. Reviewed all 47 remaining real top-level `rhs1_*` files after deleting the
   compatibility aliases.
2. Grouped the remaining files into five actual ownership domains instead of
   treating them as independent modules: profile-response operators, profile
   policy/diagnostics, PAS/x-block/sparse solver policy, QI preconditioners,
   and removable facades.
3. Updated `plan_final.md` with an explicit Iteration 2 and Iteration 3 move
   map so the next implementation batches can be mechanical and bounded.

Results:

- Iteration 2 will move operator/layout/source/CSR/KSP/profile policy files
  into `operators.profile_response` and `problems.profile_response`.
- Iteration 3 will move PAS, x-block, symbolic sparse, Schur, dispatch, and QI
  solver families into `solvers.preconditioners`.
- `rhs1_strong_fallback.py` is identified as a delete candidate once imports
  point at `problems.profile_response.preconditioner_build`.

Current lane status:

- Lane 1 Iteration 1: committed and pushed as `d432600`.
- Lane 1 Iteration 2: planned with concrete file destinations; implementation
  is the next active step.
- Lane 1 overall: about 25%.

Next best steps:

1. Execute Iteration 2 as one bounded batch: move operator/layout/source/CSR
   modules and profile-response diagnostics/policy owners, then rewrite imports.
2. Run focused RHSMode-1 operator/profile-response tests, import-contract
   tests, docs build, and whitespace checks.
3. Commit and push the Iteration 2 checkpoint before starting the QI/PAS/x-block
   solver-family consolidation in Iteration 3.

## 2026-06-25 Lane 1 Iteration 2 RHSMode-1 Domain Move

Steps taken:

1. Moved RHSMode-1 operator/layout/source modules from top-level `rhs1_*`
   names into `sfincs_jax.operators.profile_response`.
2. Moved full-system CSR assembly, structured CSR bundle construction,
   true-operator rescue, Fortran-reduced direct-tail materialization, and
   device-CSR runtime helpers into the same operator domain package.
3. Moved KSP diagnostics, host/large-CPU/direct-tail policy, solver-policy
   parsing, active-preconditioner auto policy, and general RHSMode-1
   preconditioner auto policy into `sfincs_jax.problems.profile_response`.
4. Deleted the obsolete `rhs1_strong_fallback.py` facade after imports were
   pointed at `problems.profile_response.preconditioner_build`.
5. Rewrote source, tests, scripts, and API docs to import the new canonical
   module owners.
6. Added explicit owner comments to intentionally large moved files:
   `operators.profile_response.layout`,
   `operators.profile_response.full_system`, and
   `operators.profile_response.true_operator_rescue`.

Results:

- Top-level `rhs1_*` files decreased from `47` to `28`.
- The remaining `28` top-level `rhs1_*` files are solver-family modules queued
  for Lane 1 Iteration 3: QI, PAS, x-block, symbolic sparse, Schur, dispatch,
  and related production preconditioner policies.
- Python source file count remains `247` because this batch moved real
  implementation files and replaced the deleted facade with the new
  `operators.profile_response` package initializer.
- `v3_driver.py` remains `11,992` lines; its large solve entry points are still
  the Lane 1 Iteration 4 target.

Validation:

- `python -m py_compile sfincs_jax/operators/profile_response/*.py
  sfincs_jax/problems/profile_response/*.py sfincs_jax/v3_driver.py
  sfincs_jax/io.py` passed.
- Focused moved RHSMode-1 module tests passed with
  `302 passed in 77.53 s`.
- Broad RHSMode-1/profile-response sweep passed with
  `1140 passed in 234.43 s`.
- Import-contract/docstring/helper tests passed with `20 passed in 1.94 s`.
- `python -m sphinx -b html docs docs/_build/html -q`, scoped `ruff`,
  `git diff --check`, and moved-top-level import scans passed.

Current lane status:

- Lane 1 Iteration 1: 100%.
- Lane 1 Iteration 2: 100%.
- Lane 1 overall: about 45%.
- Overall consolidation goal: not complete.

Next best steps:

1. Commit and push Iteration 2.
2. Execute Iteration 3 as one bounded solver-family consolidation: move the
   remaining 28 top-level `rhs1_*` files under `solvers.preconditioners`, merge
   tiny policy-only fragments, and delete stale private aliases.
3. Re-run focused RHSMode-1 solver/preconditioner tests and import-contract
   checks before moving to the `v3_driver.py` solve-entry extraction.

## 2026-06-25 Lane 1 Iteration 3 Solver-Family Ownership Move

Steps taken:

1. Moved all remaining top-level RHSMode-1 solver/preconditioner files into
   `sfincs_jax.solvers.preconditioners`.
2. Routed PAS policy/runtime files into `solvers.preconditioners.pas`.
3. Routed x-block, low-mode coarse, and x-block sparse-host policy files into
   `solvers.preconditioners.xblock`.
4. Routed QI coarse/device/deflation/Galerkin/multilevel/residual-region
   modules into `solvers.preconditioners.qi`.
5. Routed symbolic reduced-Pmat/frontal/sparse-factor policies into
   `solvers.preconditioners.symbolic_sparse`.
6. Routed Schur policy, full-FP kinetic CSR preconditioner, domain-decomposition
   policy, and the preconditioner dispatch module into their solver-family
   packages.
7. Rewrote source, tests, scripts, and API docs to use the canonical solver
   package paths.

Results:

- Top-level `rhs1_*` files decreased from `28` to `0`.
- Top-level `transport_*` remains `0`.
- Python source file count remains `247`; this means the top-level ownership
  target is met, but the below-240 source-count target still requires at least
  eight merges/deletions in the next consolidation subtask.
- `v3_driver.py` remains `11,992` lines; its solve-entry extraction remains
  Lane 1 Iteration 4.

Validation:

- `python -m py_compile sfincs_jax/solvers/preconditioners/**/*.py
  sfincs_jax/v3_driver.py` passed.
- Focused solver-family tests passed with `619 passed in 46.01 s`.
- Broad RHSMode-1/profile-response/v3 sparse sweep passed with
  `1472 passed in 334.53 s`.
- Import-contract/docstring/helper tests passed with `20 passed in 1.91 s`.
- `python -m sphinx -b html docs docs/_build/html -q`, scoped `ruff`, and
  `git diff --check` passed.

Current lane status:

- Lane 1 Iteration 1: 100%.
- Lane 1 Iteration 2: 100%.
- Lane 1 Iteration 3 ownership move: 100%.
- Lane 1 Iteration 3 count consolidation: incomplete; source files must drop
  from `247` to below `240`.
- Lane 1 overall: about 60%.
- Overall consolidation goal: not complete.

Next best steps:

1. Commit and push the solver-family ownership move.
2. Merge or delete at least eight package files without changing solver
   behavior. Preferred low-risk targets are tiny policy/helper files inside
   `domain_decomposition`, `xblock`, `schur`, `qi`, and
   `symbolic_sparse`.
3. Re-run focused solver-family tests and count checks; then proceed to
   Iteration 4, extracting the two large solve entry points from
   `v3_driver.py`.

## 2026-06-25 Lane 1 Iteration 3 Count Consolidation

Steps taken:

1. Merged x-block sparse-host policy helpers into
   `sfincs_jax.solvers.preconditioners.xblock.policy`.
2. Merged QI Galerkin probe-selection policy helpers into
   `sfincs_jax.solvers.preconditioners.qi.residual_galerkin`.
3. Merged domain-decomposition patch/block-size policy helpers into
   `sfincs_jax.solvers.preconditioners.domain_decomposition.line_blocks`.
4. Merged RHSMode-1 Schur policy resolution into
   `sfincs_jax.solvers.preconditioners.schur.rhs1`.
5. Merged symbolic-sparse frontal/reduced-factor policy helpers into
   `sfincs_jax.solvers.preconditioners.symbolic_sparse.policy`.
6. Merged reduced-Pmat elimination-plan helpers into
   `sfincs_jax.solvers.preconditioners.symbolic_sparse.rhs1_fortran_reduced`.
7. Deleted the empty profile-response operator namespace initializer and
   removed duplicate API-doc entries created by the redirected modules.

Results:

- Python source files decreased from `247` to `239`, meeting the Lane 1
  below-240 file-count gate.
- Top-level `rhs1_*` remains `0`.
- Top-level `transport_*` remains `0`.
- `v3_driver.py` remains `11,992` lines; this is now the dominant Lane 1 gap.
- `plan_final.md` now treats Lane 1 Iteration 3 as complete and makes
  Iteration 4, the solve-entry extraction, the next mandatory coarse tranche.

Validation:

- `python -m py_compile sfincs_jax/solvers/preconditioners/**/*.py
  sfincs_jax/v3_driver.py sfincs_jax/operators/profile_response/*.py
  sfincs_jax/problems/profile_response/*.py
  sfincs_jax/problems/profile_response/sparse/*.py` passed.
- Focused merged-policy/profile-response tests passed with
  `669 passed in 152.15 s`.
- Import-contract/docstring/helper tests passed with `20 passed in 1.26 s`.
- `python -m sphinx -b html docs docs/_build/html -q`, scoped `ruff`, and
  `git diff --check` passed.

Current lane status:

- Lane 1 Iteration 1: 100%.
- Lane 1 Iteration 2: 100%.
- Lane 1 Iteration 3 ownership and count consolidation: 100%.
- Lane 1 Iteration 4 solve-entry extraction: 0%.
- Lane 1 Iteration 5 dead-code/docs/API hardening: 0%.
- Lane 1 overall: about 70%.
- Overall consolidation goal: not complete.

Next best steps:

1. Commit and push the Iteration 3 count-consolidation batch.
2. Execute Lane 1 Iteration 4 in one coarse move: relocate
   `solve_v3_full_system_linear_gmres` to
   `sfincs_jax.problems.profile_response.solve` and
   `solve_v3_transport_matrix_linear_gmres` to
   `sfincs_jax.problems.transport_matrix.solve`, leaving `v3_driver.py` as a
   thin compatibility shim.
3. After Iteration 4, execute Lane 1 Iteration 5 once: dead-code pruning,
   docs/API cleanup, final count checks, focused physics/refactor tests, docs
   build, and one CI-equivalent local validation pass.

## 2026-06-25 Lane 1 Tranche A Planning And Solve-Entry Extraction Checkpoint

Steps taken:

1. Re-inventoried the package after the driver extraction and confirmed the
   current structural debt has moved from `v3_driver.py` into larger domain
   owners rather than disappearing.
2. Moved the legacy profile-response solve entry point into
   `sfincs_jax.problems.profile_response.solve`.
3. Moved the legacy RHSMode 2/3 transport solve entry point into
   `sfincs_jax.problems.transport_matrix.solve`.
4. Absorbed the old low-level profile-response and transport
   `linear_solve.py` implementations into their solve owners and queued those
   two files for deletion.
5. Replaced `sfincs_jax.v3_driver` with a small compatibility shim that aliases
   the profile-response solve owner and injects the transport solve entry
   points for legacy imports.
6. Rewrote `plan_final.md` so it is the single authoritative plan. Lane 1 now
   has four finite consolidation tranches: solve-entry boundary, profile-
   response collapse, transport/output collapse, and solver/preconditioner
   review hardening.

Current inventory:

- Package Python files: `239`.
- Top-level `rhs1_*` files: `0`.
- Top-level `transport_*` files: `0`.
- `sfincs_jax/v3_driver.py`: `47` lines.
- `sfincs_jax/problems/profile_response/solve.py`: `11,279` lines.
- `sfincs_jax/problems/transport_matrix/solve.py`: `1,763` lines.
- `problems/profile_response`: `33` Python files and about `50k` lines.
- `problems/transport_matrix`: `33` Python files and about `15k` lines.
- `solvers/preconditioners`: `50` Python files and about `37k` lines.

Validation so far:

- `python -m py_compile sfincs_jax/v3_driver.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/transport_matrix/solve.py
  sfincs_jax/problems/profile_response/*.py
  sfincs_jax/problems/transport_matrix/*.py` passed.
- Import smoke confirmed `sfincs_jax.v3_driver` resolves to the
  profile-response solve owner and that the profile-response and transport
  solve entry points have domain-owned `__module__` values.
- Focused CLI/import/profile/transport tests passed with
  `86 passed in 35.14 s`.
- Focused transport/profile solve tests passed with
  `71 passed in 22.12 s`.
- Legacy helper compatibility tests passed with `101 passed in 25.14 s`.
- Broad RHSMode-1 sparse-pattern sweep passed with
  `242 passed in 164.52 s`.
- Sphinx `-W` build passed after documenting `sfincs_jax.v3_driver` as a
  compatibility shim.
- `git diff --check` passed.
- Broad `python -m ruff check sfincs_jax tests` still fails on existing
  package-wide lint debt outside this extraction; Lane 1 Tranche B/D now
  explicitly require narrowing or removing temporary broad ignores on extracted
  modules.

Current lane status:

- Lane 1 Tranche A: about `85%`; implementation and focused validation are
  done, but final plan/log validation and commit/push remain.
- Lane 1 Tranche B: `0%`.
- Lane 1 Tranche C: `0%`.
- Lane 1 Tranche D: `0%`.
- Lane 1 overall: about `45%` of the new consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Finish Tranche A by committing the solve-entry extraction and the
   authoritative `plan_final.md` update.
2. Execute Tranche B in one large profile-response consolidation: shrink
   `profile_response/solve.py`, merge policy shards into `policies.py`, merge
   sparse handoff shards into role owners, and remove temporary broad lint
   ignores.
3. Execute Tranche C in one transport/output consolidation: collapse
   transport-matrix helper shards, compact parallel worker files, and move
   `io.py` orchestration into `outputs`.
4. Execute Tranche D in one solver/preconditioner review pass: collapse QI by
   role, remove or rename remaining `rhs1`-named solver internals, refresh docs
   and source maps, and bring package file count below `220`.

## 2026-06-25 Lane 1 Tranche B Policy/Sparse Checkpoint

Steps taken:

1. Consolidated six profile-response policy shards into
   `sfincs_jax.problems.profile_response.policies`:
   `active_preconditioner_policy.py`, `direct_tail_policy.py`,
   `host_policy.py`, `large_cpu_policy.py`,
   `preconditioner_auto_policy.py`, and `solver_policy.py` were deleted.
2. Consolidated top-level profile-response finalization and KSP replay
   diagnostics into `sfincs_jax.problems.profile_response.solver_diagnostics`;
   `finalization.py` and `ksp_diagnostics.py` were deleted.
3. Moved sparse-PC Krylov execution helpers from
   `sfincs_jax.problems.profile_response.sparse.krylov` into
   `sfincs_jax.problems.profile_response.sparse_pc`; `sparse/krylov.py` was
   deleted. Shared sparse-PC finalization remains in
   `profile_response/sparse/finalization.py` because direct/x-block/
   Fortran-reduced sparse paths share the same payload contracts.
4. Rewrote internal imports and tests to use canonical owners, and removed
   deleted modules from `docs/api.rst` and `docs/testing.rst`.
5. Repaired a transient duplicate sparse-PC finalization merge by restoring
   the shared sparse finalization owner and keeping only Krylov execution in
   `sparse_pc.py`.
6. Updated `plan_final.md` as the single authoritative plan for the remaining
   consolidation pass. The plan now names the remaining large tranches and the
   concrete delete/merge targets instead of allowing helper-by-helper churn.

Current inventory:

- Package Python files: `230`.
- `problems/profile_response`: `24` Python files and about `50k` lines.
- `sfincs_jax/problems/profile_response/solve.py`: `11,279` lines.
- `sfincs_jax/problems/profile_response/policies.py`: `6,380` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: `3,761` lines.
- `sfincs_jax/problems/profile_response/solver_diagnostics.py`: `812` lines.
- `sfincs_jax/problems/transport_matrix`: `33` Python files.
- `sfincs_jax/solvers/preconditioners`: `50` Python files.

Validation:

- Focused policy/finalization/sparse-PC tests passed:
  `526 passed in 39.20s`.
- Scoped ruff passed for the consolidated profile-response modules and sparse
  tests.
- `python -m py_compile sfincs_jax/problems/profile_response/*.py
  sfincs_jax/problems/profile_response/sparse/*.py` passed.
- `python -m sphinx -W -b html docs docs/_build/html` passed.

Current lane status:

- Lane 1 Tranche A: `100%` implementation/validation complete pending final
  checkpoint commit.
- Lane 1 Tranche B: about `35%`; policy/diagnostics/krylov shards are
  consolidated and the old 24-file intermediate count is reached, but the
  final 22-file target remains open, `profile_response/solve.py` still must be
  reduced below `3.5k` lines, and `sparse_pc.py` must be deleted or moved under
  the sparse package.
- Lane 1 Tranche C: `0%`.
- Lane 1 Tranche D: `0%`.
- Lane 1 overall: about `55%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Commit and push this Tranche B checkpoint after `git diff --check` passes.
2. Finish Tranche B in one large solve-phase move: move setup/materialization,
   residual/admission, sparse branch orchestration, final handoff, and
   diagnostics from `profile_response/solve.py` into existing profile-response
   owners, then delete or fold `sparse_pc.py`.
3. Run the focused RHSMode-1, sparse-PC, QI admission, ambipolar, sensitivity,
   docs, and import-contract gates before starting Tranche C.

## 2026-06-25 Lane 1 Tranche B Solve-Phase Owner Checkpoint

Steps taken:

1. Moved the profile linear-solve routing contracts from
   `sfincs_jax.problems.profile_response.solve` into the existing dense owner:
   `ProfileLinearSolveContext`, `solve_profile_linear`,
   `solve_profile_linear_with_residual`, dense-KSP full/reduced solve
   contexts, constraintScheme=0 PETSc-compatible sparse-ILU, and host SciPy
   rescue now live in `sfincs_jax.problems.profile_response.dense`.
2. Updated `tests/test_profile_response_linear_solve.py` to import from the
   canonical dense owner instead of the monolithic solve owner.
3. Moved the explicit host structured-CSR RHSMode-1 solve entry point into
   `sfincs_jax.problems.profile_response.dense`, beside the auto-routing
   code that invokes it. `profile_response.solve` imports it only for
   compatibility and internal dispatch.
4. Preserved behavior and legacy import compatibility while removing another
   solve-phase block from `profile_response.solve`.

Current inventory:

- Package Python files: `230`.
- `problems/profile_response`: `24` Python files.
- `sfincs_jax/problems/profile_response/solve.py`: `10,400` lines.
- `sfincs_jax/problems/profile_response/dense.py`: `2,487` lines.
- `sfincs_jax/problems/profile_response/auto_solve.py`: `550` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: `3,761` lines.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- Focused profile-response dense/structured/policy/sparse tests passed:
  `522 passed in 43.29s`.
- Scoped ruff passed for `profile_response.dense`, `profile_response.auto_solve`,
  and the moved linear-solve test.
- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/auto_solve.py
  sfincs_jax/problems/profile_response/dense.py` passed.
- `python -m sphinx -W -b html docs docs/_build/html` passed.

Current lane status:

- Lane 1 Tranche A: `100%`.
- Lane 1 Tranche B: about `45%`; policy/diagnostics/krylov and dense/
  structured solve-phase ownership moves are done. Remaining hard gates are
  `profile_response/solve.py < 3.5k`, `problems/profile_response <= 22` files,
  and `sparse_pc.py` deleted or moved under the sparse package.
- Lane 1 Tranche C: `0%`.
- Lane 1 Tranche D: `0%`.
- Lane 1 overall: about `60%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Finish Tranche B with the largest remaining coherent move: move sparse-PC
   branch orchestration and final sparse handoff out of `solve.py` into
   `profile_response.sparse` owners, then delete or rename `sparse_pc.py`.
2. Move residual/admission and setup/materialization blocks from `solve.py`
   into `residual.py`, `setup.py`, `handoff.py`, and `solver_diagnostics.py`.
3. Run the same focused RHSMode-1, sparse-PC, QI admission, ambipolar,
   sensitivity, docs, and import-contract gates before starting Tranche C.

## 2026-06-25 Lane 1 Tranche B Sparse Handoff Checkpoint

Steps taken:

1. Moved the former top-level
   `sfincs_jax/problems/profile_response/sparse_pc.py` module into
   `sfincs_jax/problems/profile_response/sparse/handoff.py`.
2. Rewrote source and focused tests to import the canonical sparse-package
   owner.
3. Exported the handoff owner through
   `sfincs_jax.problems.profile_response.sparse`.
4. Updated `docs/api.rst` and `docs/source_map.rst` so docs no longer point at
   deleted `profile_response/sparse_pc.py`, `profile_response/linear_solve.py`,
   `profile_response/finalization.py`, or `sparse/krylov.py` owners.

Current inventory:

- Package Python files: `230`.
- `problems/profile_response`: `24` Python files.
- `sfincs_jax/problems/profile_response/solve.py`: `10,400` lines.
- `sfincs_jax/problems/profile_response/sparse/handoff.py`: `3,761` lines.
- `sfincs_jax/problems/profile_response/sparse_pc.py`: deleted.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- Focused sparse/RHSMode-1 tests passed: `498 passed in 42.21s`.
- Scoped ruff passed for `profile_response.sparse.handoff`, sparse exports,
  `profile_response.solve`, and `tests/test_profile_response_sparse_pc.py`.
- `python -m py_compile sfincs_jax/problems/profile_response/*.py
  sfincs_jax/problems/profile_response/sparse/*.py` passed.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Current lane status:

- Lane 1 Tranche A: `100%`.
- Lane 1 Tranche B: about `50%`; sparse top-level handoff is now under the
  sparse package. Remaining hard gates are `profile_response/solve.py < 3.5k`,
  `problems/profile_response <= 22` files, and policy/env parser duplication
  cleanup.
- Lane 1 Tranche C: `0%`.
- Lane 1 Tranche D: `0%`.
- Lane 1 overall: about `62%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Continue Tranche B by moving residual/admission and setup/materialization
   blocks from `solve.py` into `residual.py`, `setup.py`, `handoff.py`, and
   `solver_diagnostics.py`.
2. Fold `profile_response/sparse/handoff.py` further into direct/x-block/QI
   sparse owners only if it reduces file count without making those owners too
   large to review.
3. After `solve.py` is below `3.5k` and profile-response is at or below `22`
   files, run the focused RHSMode-1, sparse-PC, QI admission, ambipolar,
   sensitivity, docs, and import-contract gates before starting Tranche C.

## 2026-06-25 Lane 1 Phase 1 Ownership Batch

Steps taken:

1. Merged `sfincs_jax/problems/profile_response/active_projection.py` into
   `sfincs_jax/problems/profile_response/active_dof.py`.
2. Merged `sfincs_jax/problems/profile_response/qi_device_seed.py` into
   `sfincs_jax/problems/profile_response/sparse/qi.py`.
3. Merged `sfincs_jax/problems/profile_response/strong_preconditioning.py`
   into `sfincs_jax/problems/profile_response/preconditioner_build.py`.
4. Rewrote source, tests, and docs to import the canonical owners.
5. Updated `docs/source_map.rst` and `docs/api.rst` so the merged owners are
   documented once.

Current inventory:

- Package Python files: `227`.
- `problems/profile_response`: `21` Python files.
- `sfincs_jax/problems/profile_response/solve.py`: `10,400` lines.
- `sfincs_jax/problems/profile_response/active_projection.py`: deleted.
- `sfincs_jax/problems/profile_response/qi_device_seed.py`: deleted.
- `sfincs_jax/problems/profile_response/strong_preconditioning.py`: deleted.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- Focused owner tests passed:
  `63 passed in 1.68s`.
- Broader sparse/RHSMode-1 tests passed:
  `498 passed in 42.12s`.
- Scoped ruff passed for touched profile-response source and tests.
- `python -m py_compile sfincs_jax/problems/profile_response/*.py
  sfincs_jax/problems/profile_response/sparse/*.py` passed.
- `git diff --check` passed.

Current lane status:

- Lane 1 Phase 0: `100%`.
- Lane 1 Phase 1: about `35%`; three small owners are merged, but the hard
  gates remain `profile_response/solve.py < 3.5k` and
  `problems/profile_response <= 20` files.
- Lane 1 Phase 2: `0%`.
- Lane 1 Phase 3: `0%`.
- Lane 1 Phase 4: `0%`.
- Lane 1 Phase 5: `0%`.
- Lane 1 overall: about `42%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Continue Phase 1 by moving sparse pattern/materialization and host sparse
   direct setup helpers out of `profile_response/solve.py` into
   `profile_response/sparse/direct.py` and `profile_response/sparse/policy.py`.
2. Move final payload/progress replay code from `solve.py` into
   `handoff.py` and `solver_diagnostics.py`.
3. Re-evaluate whether `sparse/finalization.py` can be merged after sparse
   handoff cycles are reduced; it is currently shared by `handoff.py`,
   `xblock.py`, and `fortran_reduced.py`, so it was not merged in this batch.

## 2026-06-25 Lane 1 Tranche 1 Sparse-Direct Ownership Move

Steps taken:

1. Moved sparse-factor cache-key construction, host memory probing,
   RHSMode-1 explicit sparse-pattern probing, sparse-JAX preconditioner
   materialization, host sparse direct factor-builder callback injection, host
   sparse direct polish, and unsharded submatrix probing from
   `sfincs_jax/problems/profile_response/solve.py` into
   `sfincs_jax/problems/profile_response/sparse/direct.py`.
2. Kept the historical private names visible through `solve.py` imports so
   existing solve-path calls and `sfincs_jax.v3_driver` compatibility imports
   still resolve.
3. Updated helper-internal tests to patch the canonical sparse-direct owner
   instead of patching old `v3_driver` globals for moved implementation
   details.
4. Updated `docs/source_map.rst` and `plan_final.md` with the new
   sparse-direct ownership boundary.

Current inventory:

- Package Python files: `227`.
- `problems/profile_response`: `21` Python files.
- `sfincs_jax/problems/profile_response/solve.py`: `10,175` lines.
- `sfincs_jax/problems/profile_response/sparse/direct.py`: `3,616` lines.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/direct.py
  sfincs_jax/problems/transport_matrix/solve.py` passed.
- Scoped ruff passed for touched source and tests.
- Sparse-helper canonical-owner coverage passed:
  `15 passed in 0.74s`.
- Broader sparse/preconditioner coverage passed:
  `124 passed in 49.31s`.

Current lane status:

- Lane 1 Tranche 0: `100%`.
- Lane 1 Tranche 1: about `40%`; sparse-direct materialization ownership is
  moved, but the hard gates remain `profile_response/solve.py < 3.5k` and
  `problems/profile_response <= 18` files.
- Lane 1 Tranche 2: `0%`.
- Lane 1 Tranche 3: `0%`.
- Lane 1 Tranche 4: `0%`.
- Lane 1 Tranche 5: `0%`.
- Lane 1 overall: about `48%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Move sparse-PC branch orchestration and final sparse-PC payload assembly
   from `solve.py` into the existing sparse handoff/x-block/QI/fortran-reduced
   owners.
2. Move remaining backend-policy wrappers from `solve.py` into `policies.py`
   and `sparse/policy.py`, then delete duplicated env parsers where imports no
   longer create cycles.
3. Run the full focused RHSMode-1 sparse/QI/ambipolar/sensitivity/docs gate
   before starting Tranche 2.

## 2026-06-25 Lane 1 Tranche 1 Current-Backend Policy Wrapper Move

Steps taken:

1. Moved the remaining current-backend RHSMode-1 dense/sparse/PAS/x-block
   admission wrappers from `sfincs_jax/problems/profile_response/solve.py` into
   `sfincs_jax/problems/profile_response/policies.py`.
2. Kept the historical private names visible through `solve.py` imports so
   existing solve-path calls and `sfincs_jax.v3_driver` compatibility imports
   still resolve.
3. Updated `docs/source_map.rst` to remove stale top-level RHSMode-1 policy
   shard references and document `policies.py` as the owner of current-backend
   wrapper glue.
4. Updated `plan_final.md` with the new counts and next Tranche 1 target.

Current inventory:

- Package Python files: `227`.
- `problems/profile_response`: `21` Python files.
- `sfincs_jax/problems/profile_response/solve.py`: `9,730` lines.
- `sfincs_jax/problems/profile_response/policies.py`: `6,876` lines.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/policies.py` passed.
- Scoped ruff passed for touched source.
- Policy and heuristic coverage passed:
  `147 passed in 25.46s`.
- Sparse-PC owner coverage passed:
  `327 passed in 3.17s`.

Current lane status:

- Lane 1 Tranche 0: `100%`.
- Lane 1 Tranche 1: about `45%`; sparse-direct materialization and
  current-backend policy ownership have moved, but hard gates remain
  `profile_response/solve.py < 3.5k` and `problems/profile_response <= 18`
  files.
- Lane 1 Tranche 2: `0%`.
- Lane 1 Tranche 3: `0%`.
- Lane 1 Tranche 4: `0%`.
- Lane 1 Tranche 5: `0%`.
- Lane 1 overall: about `50%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Move sparse-PC branch orchestration and final sparse-PC payload assembly
   from `solve.py` into `sparse/handoff.py`, `sparse/xblock.py`,
   `sparse/qi.py`, and `sparse/fortran_reduced.py`.
2. Move final solve result normalization and progress replay from `solve.py`
   into `handoff.py` and `solver_diagnostics.py`.
3. Re-check whether `sparse/finalization.py` can be merged or whether it
   remains a true shared owner after the sparse-PC handoff code moves.

## 2026-06-25 Lane 1 Tranche 1 RHSMode-1 Preconditioner Registry Move

Steps taken:

1. Moved the current RHSMode-1 preconditioner registry and binding layer from
   `sfincs_jax/problems/profile_response/solve.py` into
   `sfincs_jax/problems/profile_response/preconditioner_build.py`.
2. The canonical owner now exposes the dispatch binding, PAS-family
   compatibility builders, Schur binding, x-block builder aliases,
   transport `tzfft` reuse, and strong fallback binding.
3. Updated RHSMode-1 dispatch, strong-fallback, PAS-policy, and Schwarz
   heuristic tests to patch/call `preconditioner_build.py` rather than the
   compatibility `v3_driver.py` surface for these internals.
4. Updated `docs/source_map.rst` and `plan_final.md` so the source map and
   single authoritative plan match the current owner boundaries.

Current inventory:

- Package Python files: `227`.
- Package source lines: `163,622`.
- `problems/profile_response`: `21` Python files and `50,187` lines.
- `solvers/preconditioners`: `50` Python files and `37,043` lines.
- `sfincs_jax/problems/profile_response/solve.py`: `9,410` lines.
- `sfincs_jax/problems/profile_response/preconditioner_build.py`: `2,683`
  lines.
- `sfincs_jax/v3_driver.py`: `47` lines.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/preconditioner_build.py` passed.
- `python -m ruff check
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/preconditioner_build.py` passed.
- `python -m ruff check
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_strong_fallback_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py
  tests/test_rhs1_schwarz_heuristic.py` passed.
- RHSMode-1 dispatch/policy/Schwarz tests passed:
  `75 passed in 32.34s`.
- Preconditioner-build and sparse-PC owner tests passed:
  `86 passed in 0.50s` and `327 passed in 2.86s`.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Current lane status:

- Lane 1 Tranche 0: `100%`.
- Lane 1 Tranche 1: about `52%`; sparse-direct setup, current-backend policy
  wrappers, and RHSMode-1 preconditioner registry ownership have moved, but
  hard gates remain `profile_response/solve.py < 3.5k` and
  `problems/profile_response <= 18` files.
- Lane 1 Tranche 2: `0%`.
- Lane 1 Tranche 3: `0%`.
- Lane 1 Tranche 4: `0%`.
- Lane 1 Tranche 5: `0%`.
- Lane 1 overall: about `52%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Move sparse-PC branch orchestration and final sparse-PC payload assembly
   from `solve.py` into `sparse/handoff.py`, `sparse/xblock.py`,
   `sparse/qi.py`, and `sparse/fortran_reduced.py`.
2. Move final solve result normalization and progress replay from `solve.py`
   into `handoff.py` and `solver_diagnostics.py`.
3. Collapse duplicated env-token parsing into one parser family in
   `policies.py`, then delete local parser duplicates after circular-import
   checks.

## 2026-06-25 Lane 1 Tranche 1 Sparse Parser And X-Block Payload Consolidation

Steps taken:

1. Moved x-block sparse-PC final metadata/payload builders from
   `sfincs_jax/problems/profile_response/sparse/handoff.py` into the canonical
   x-block owner, `sfincs_jax/problems/profile_response/sparse/xblock.py`.
   `handoff.py` now re-exports those names only for compatibility.
2. Updated `solve.py` to import the moved x-block final payload helpers from
   `sparse.xblock` rather than from the handoff compatibility surface.
3. Collapsed duplicate sparse env-token parser families from
   `sparse/direct.py`, `sparse/xblock.py`, `sparse/qi.py`, and
   `sparse/fortran_reduced.py` into the shared parser implementation in
   `sparse/policy.py`.
4. Updated the sparse-PC import-contract tests so x-block payload internals are
   monkeypatched on the canonical `sparse.xblock` module.
5. Updated `docs/source_map.rst` and `plan_final.md` with the new ownership
   boundaries and current counts.

Current inventory:

- Package Python files: `227`.
- Package source lines: `163,474`.
- `sfincs_jax/problems/profile_response/solve.py`: `9,412` lines.
- `sfincs_jax/problems/profile_response/sparse/direct.py`: `3,569` lines.
- `sfincs_jax/problems/profile_response/sparse/handoff.py`: `3,649` lines.
- `sfincs_jax/problems/profile_response/sparse/xblock.py`: `4,572` lines.
- `sfincs_jax/problems/profile_response/sparse/qi.py`: `4,885` lines.
- `sfincs_jax/problems/profile_response/sparse/fortran_reduced.py`: `1,335`
  lines.
- `sfincs_jax/problems/profile_response/sparse/policy.py`: `1,133` lines.

Validation:

- `python -m ruff check sfincs_jax/problems/profile_response/sparse` passed.
- `python -m py_compile
  sfincs_jax/problems/profile_response/sparse/*.py
  sfincs_jax/problems/profile_response/solve.py` passed.
- Sparse parser audit shows only `sparse/policy.py` owns the bool/int/float
  parser family under `problems/profile_response/sparse`; the remaining
  `fortran_reduced._env_float_first` is a different multi-key helper.
- Sparse-PC owner tests passed: `327 passed in 3.62s`.
- RHSMode-1 dispatch/policy tests passed: `75 passed in 38.13s`.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Current lane status:

- Lane 1 Tranche 0: `100%`.
- Lane 1 Tranche 1: about `55%`; sparse parser duplication is closed and
  x-block final payload ownership is moved, but hard gates remain
  `profile_response/solve.py < 3.5k` and `problems/profile_response <= 18`
  files.
- Lane 1 overall: about `55%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Move generic sparse-PC GMRES finalization bundle/state builders from
   `sparse/handoff.py` into `sparse/finalization.py`, then keep handoff as a
   compatibility re-export only.
2. Move sparse-PC branch orchestration from `solve.py` into the existing sparse
   owners, starting with the generic sparse-PC GMRES branch because it has a
   clear return boundary.
3. Move final progress replay/result normalization from `solve.py` into
   `solver_diagnostics.py` and sparse finalization owners.

## 2026-06-25 Lane 1 Batch 0 Root Helper Cleanup

Steps taken:

1. Deleted five obsolete top-level helper modules instead of keeping
   compatibility shims:
   `sfincs_jax/solver_runtime.py`, `sfincs_jax/matrix_reductions.py`,
   `sfincs_jax/solve_mode_policy.py`,
   `sfincs_jax/solver_progress_policy.py`, and
   `sfincs_jax/phase_timing.py`.
2. Moved GMRES result finite-state and XLA readiness helpers into the canonical
   Krylov owner, `sfincs_jax/solver.py`.
3. Moved diagonal and block-diagonal matrix-reduction helpers into
   `sfincs_jax/solvers/preconditioner_operators.py`, where simplified
   preconditioner-operator shaping already lives.
4. Moved implicit/differentiable solve-mode selection into
   `sfincs_jax/problems/profile_response/policies.py`.
5. Merged pure duration/runtime/progress-threshold helpers into
   `sfincs_jax/solvers/progress.py`.
6. Moved benchmark/audit phase timing into
   `sfincs_jax/validation_artifacts.py`.
7. Rewrote source, script, docs, and focused-test imports to use canonical
   owners and removed stale API/source-map entries for the deleted modules.

Current inventory:

- Package Python files: `222` (down from `227`).
- Package-root Python files: `90` (down from `95`).
- Package source lines: `163,423` (down from `163,474`).
- `sfincs_jax/v3_driver.py`: `47` lines.
- `sfincs_jax/problems/profile_response/solve.py`: `9,412` lines.

Validation:

- `python -m ruff check
  sfincs_jax/solver.py
  sfincs_jax/solvers/preconditioner_operators.py
  sfincs_jax/problems/profile_response/policies.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/phi1_newton.py
  sfincs_jax/problems/transport_matrix/solve.py
  sfincs_jax/solvers/progress.py
  sfincs_jax/validation_artifacts.py
  tests/test_solver_runtime.py
  tests/test_matrix_reductions.py
  tests/test_phase_timing.py
  tests/test_solve_mode_policy.py
  tests/test_solver_progress_policy.py
  scripts/run_zenodo_vmec_parity_campaign.py` passed.
- `python -m py_compile
  sfincs_jax/solver.py
  sfincs_jax/solvers/preconditioner_operators.py
  sfincs_jax/problems/profile_response/policies.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/phi1_newton.py
  sfincs_jax/problems/transport_matrix/solve.py
  sfincs_jax/solvers/progress.py
  sfincs_jax/validation_artifacts.py` passed.
- Focused helper tests passed:
  `python -m pytest
  tests/test_solver_runtime.py
  tests/test_matrix_reductions.py
  tests/test_phase_timing.py
  tests/test_solve_mode_policy.py
  tests/test_solver_progress_policy.py -q --tb=short`
  with `20 passed in 0.66 s`.
- Focused RHSMode-1 dispatch and fallback coverage passed:
  `python -m pytest
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_v3_driver_strong_fallback_coverage.py
  tests/test_v3_driver_pas_precond_policy_coverage.py
  tests/test_rhs1_schwarz_heuristic.py
  tests/test_v3_driver_dd_reduction_coverage.py -q --tb=short`
  with `84 passed in 37.80 s`.
- Transport linear-solve and parallel regression coverage passed:
  `python -m pytest
  tests/test_small_regularized_lstsq.py
  tests/test_transport_solve_policy.py
  tests/test_transport_parallel.py -q --tb=short`
  with `34 passed in 40.46 s`. This caught one missed transport
  domain-decomposition builder import, which was fixed by importing the
  canonical RHSMode-2/3 aliases from `profile_response/preconditioner_build.py`.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- Stale deleted-module import audit found no source imports; the only hit is a
  retained test filename mention in `docs/testing.rst`.
- `git diff --check` passed.

Current lane status:

- Lane 1 Batch 0: helper cleanup checkpoint complete; broader compatibility
  import audits for `v3_*`, `io.py`, and root-level solver/preconditioner
  modules remain.
- Lane 1 Batch 1: not complete; `profile_response/solve.py` remains the main
  structural blocker at `9,412` lines.
- Lane 1 overall: about `58%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Run Sphinx `-W` and focused RHSMode-1 import/dispatch tests for the root
   helper cleanup.
2. Continue Batch 1 by moving sparse-PC branch orchestration and final payload
   assembly from `profile_response/solve.py` into the existing sparse owners.
3. Avoid any new helper-only files; the next implementation batch should either
   delete more files or reduce `profile_response/solve.py` materially.

## 2026-06-25 Lane 1 Batch 0 Small Root Owner Cleanup

Steps taken:

1. Deleted five more obsolete package-root implementation modules:
   `sfincs_jax/linear_algebra.py`,
   `sfincs_jax/newton_krylov_diagnostics.py`,
   `sfincs_jax/phi1_line_search.py`, `sfincs_jax/sparse.py`, and
   `sfincs_jax/verbose.py`.
2. Moved the differentiable tiny least-squares kernel and recycled Krylov
   initial-guess builder into `sfincs_jax/solver.py`.
3. Moved the JAX-native CSR matvec into `sfincs_jax/solvers/explicit_sparse.py`, where
   sparse operator/factor infrastructure already lives.
4. Moved deterministic `make_emit` and `Timer` utilities into
   `sfincs_jax/profiling.py`.
5. Moved Phi1 accepted-iterate line-search/update logic into
   `sfincs_jax/problems/profile_response/phi1_newton.py`.
6. Moved optional Newton-Krylov PETSc-style KSP history replay into
   `sfincs_jax/problems/profile_response/solver_diagnostics.py`.
7. Rewrote source, example, test, and API docs imports to use the canonical
   owners; no compatibility shims were kept for the deleted modules.

Current inventory:

- Package Python files: `217` (down from `222`).
- Package-root Python files: `85` (down from `90`).
- Package source lines: `163,358` (down from `163,423`).
- `sfincs_jax/v3_driver.py`: `47` lines.
- `sfincs_jax/problems/profile_response/solve.py`: `9,412` lines.

Validation:

- `python -m ruff check` passed for all touched source/test modules.
- `python -m py_compile` passed for all touched source modules.
- Focused moved-helper tests passed:
  `python -m pytest
  tests/test_small_regularized_lstsq.py
  tests/test_sparse_csr.py
  tests/test_phi1_line_search.py
  tests/test_newton_krylov_diagnostics.py
  tests/test_helper_module_coverage.py -q --tb=short`
  with `25 passed in 3.46 s`.
- Focused RHSMode-1 dispatch and fallback coverage passed:
  `84 passed in 44.67 s`.
- Transport solve/parallel regression coverage passed:
  `34 passed in 44.15 s`.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- Deleted-module import audit found no remaining source imports. Remaining
  broad `sparse.py` text hits are unrelated filenames such as `tz_sparse.py`
  and test names.

Current lane status:

- Lane 1 Batch 0: second helper cleanup checkpoint complete; package-root
  count is moving in the right direction but broader `v3_*`, `io.py`, solver,
  preconditioner, and transport compatibility audits remain.
- Lane 1 overall: about `60%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Start the transport-parallel consolidation: merge `execution.py`,
   `payload.py`, `pool.py`, `solve.py`, and `validation.py` into
   `problems/transport_matrix/parallel/runtime.py`, leaving only
   `runtime.py`, `policy.py`, `sharding.py`, and `worker.py` in the subpackage
   before the final policy merge.
2. Continue Batch 1 with a real profile-response solve cut, targeting sparse
   branch orchestration/finalization rather than moving one-off helpers.
3. Keep avoiding deleted-module shims; update tests/docs to canonical owners
   instead.

## 2026-06-25 Lane 1 Batch 2 Transport-Parallel Runtime Consolidation

Steps taken:

1. Merged parent-side execution, payload packing, process-pool cache
   management, parent solve orchestration, and worker-result validation into
   `sfincs_jax/problems/transport_matrix/parallel/runtime.py`.
2. Deleted five transport-parallel micro-files:
   `execution.py`, `payload.py`, `pool.py`, `solve.py`, and `validation.py`.
3. Updated source, tests, examples, and docs so payload, execution, pool,
   solve, and validation symbols are imported from the canonical runtime owner.
4. Preserved `parallel/policy.py` for now because it contains the large
   parallel scaling/audit policy surface and has direct docs/tests. It remains
   a planned follow-up before final review if we want exactly three parallel
   implementation files.
5. Added public docstrings for the runtime symbols moved into the canonical
   owner so source-map/docstring gates still enforce maintainability.

Current inventory:

- Package Python files: `212` (down from `217`).
- Package-root Python files: `85` (unchanged).
- Package source lines: `163,301` (down from `163,358`).
- `sfincs_jax/problems/transport_matrix`: `28` Python files.
- `sfincs_jax/problems/transport_matrix/parallel`: `5` Python files including
  `__init__.py`; implementation owners are `runtime.py`, `policy.py`,
  `sharding.py`, and `worker.py`.
- `sfincs_jax/problems/profile_response/solve.py`: `9,412` lines.

Validation:

- `python -m ruff check` passed for touched runtime/source/test modules.
- `python -m py_compile` passed for
  `parallel/runtime.py`, `parallel/worker.py`, and
  `profile_response/solve.py`.
- Transport-parallel test battery passed:
  `python -m pytest
  tests/test_transport_parallel_execution.py
  tests/test_transport_parallel_payload.py
  tests/test_transport_parallel_validation.py
  tests/test_transport_parallel_solve.py
  tests/test_transport_parallel_runtime.py
  tests/test_transport_parallel.py -q --tb=short`
  with `66 passed in 31.71 s`.
- Import-contract, docstring, policy, and benchmark-audit coverage passed:
  `57 passed in 1.07 s`.
- Transport solve setup/policy coverage passed:
  `25 passed in 2.31 s`.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- Stale deleted-module import audit found no remaining references to
  `parallel.execution`, `parallel.payload`, `parallel.pool`,
  `parallel.solve`, or `parallel.validation`.

Current lane status:

- Lane 1 Batch 2 transport-parallel substep: complete for execution/payload/
  pool/solve/validation. Remaining transport/output work includes merging or
  justifying `parallel/policy.py`, collapsing the other transport micro-files,
  and moving output ownership out of `io.py`.
- Lane 1 overall: about `63%` of the authoritative consolidation plan.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Either merge `parallel/policy.py` into `parallel/runtime.py` or document why
   it remains a durable policy owner, then continue transport consolidation
   with `postsolve_diagnostics.py -> finalize.py` and active/dense policy
   owner merges.
2. Start the large Batch 1 profile-response solve cut once the transport
   micro-file reductions are stable.
3. Keep the no-shim rule: update imports to canonical owners and delete
   obsolete files rather than keeping compatibility files alive.

## 2026-06-25 Lane 1 Tranche A Validation-Domain Root Disposition

Steps taken:

1. Moved six root validation/artifact policy modules into the canonical
   `sfincs_jax.validation` package:
   `validation/artifacts.py`, `validation/figures.py`, `validation/math.py`,
   `validation/benchmark_artifacts.py`, `validation/research_lanes.py`, and
   `validation/qi_device.py`.
2. Updated imports in source, tests, examples, scripts, and docs so the deleted
   root modules are not kept as compatibility shims.
3. Refreshed `docs/api.rst`, `docs/source_map.rst`, `docs/testing.rst`,
   `examples/publication_figures/validation_manifest.json`, and
   `docs/_static/research_lane_completion_2026_05_12.json` so checked
   documentation and release/research manifests point at existing canonical
   owners.
4. Removed stale deleted-parallel-file section comments from
   `problems/transport_matrix/parallel/runtime.py`.

Current inventory:

- Package Python files: `212` (unchanged because files moved, not deleted).
- Package-root Python files: `79` (down from `85`).
- `sfincs_jax.validation`: `7` Python files including `__init__.py`.

Validation:

- Scoped `ruff` passed for moved validation modules, touched scripts, touched
  tests, and `transport_matrix/parallel/runtime.py`.
- `python -m py_compile` passed for `sfincs_jax/validation/*.py` and
  `transport_matrix/parallel/runtime.py`.
- Focused validation/release tests passed:
  `python -m pytest tests/test_research_lane_policy.py
  tests/test_release_gate_metadata.py tests/test_validation_artifacts.py
  tests/test_validation_figures.py tests/test_validation_math.py
  tests/test_benchmark_artifact_policy.py
  tests/test_qi_device_artifact_policy.py
  tests/test_validation_policy_coverage.py tests/test_phase_timing.py
  tests/test_cli_validation_io_fast_coverage.py -q --tb=short`
  with `93 passed in 1.41 s`.
- `python scripts/check_qi_device_artifacts.py docs/_static --min-relevant 1`
  passed with `failed=0`.
- `python scripts/check_release_gates.py` passed.
- `python -m sphinx -W -b html docs docs/_build/html` passed.
- Stale import audit found no remaining imports of the deleted root validation
  modules.

Current lane status:

- Lane 1 Tranche A: partially complete. The validation/artifact-policy
  disposition checkpoint is done. Remaining Tranche A work is the full
  root-disposition table and moving or deleting the next root module family.
- Lane 1 overall: about `65%`.

Next best steps:

1. Continue Tranche A with either root workflow/evidence modules
   (`optimization_*`, `mapped_xgrid_*`, campaign policies) or root solver
   utility modules (`solver_*`, `explicit_sparse*`, `preconditioner_*`) as one
   coherent family move.
2. After Tranche A root disposition is complete, execute Tranche B to cut
   `profile_response/solve.py` below the target instead of doing more helper
   churn.

## 2026-06-25 Lane 1 Tranche A Workflow-Domain Root Disposition

Steps taken:

1. Moved nine workflow/evidence root modules into `sfincs_jax.workflows`
   without compatibility shims:
   `optimization_comparison.py`, `optimization_evidence.py`,
   `optimization_ladder.py`, `optimization_objectives.py`,
   `optimization_promotion.py`, `optimization_workflow.py`,
   `mapped_xgrid_objectives.py`, `mapped_xgrid_transport_evidence.py`, and
   `qi_res15_gpu_campaign.py`.
2. Updated imports in tests, examples, scripts, and docs to canonical
   `sfincs_jax.workflows.*` module paths.
3. Updated workflow module relative imports after moving them below the
   package root.
4. Updated `docs/validation_matrix.rst` so the mapped-xgrid source anchors
   point at the new workflow package paths.

Current inventory:

- Package Python files: `212` (unchanged because files moved, not merged).
- Package-root Python files: `70` (down from `79`).
- `sfincs_jax.workflows`: `10` Python files including `__init__.py`.

Validation:

- `python -m py_compile sfincs_jax/workflows/*.py` passed.
- Scoped `ruff` passed for moved workflow modules, touched tests, touched
  scripts, and touched optimization examples.
- Focused workflow tests passed:
  `python -m pytest tests/test_optimization_comparison.py
  tests/test_optimization_evidence.py tests/test_optimization_ladder.py
  tests/test_optimization_neoclassical_objectives.py
  tests/test_optimization_promotion.py tests/test_optimization_real_artifacts.py
  tests/test_optimization_workflow.py tests/test_mapped_xgrid_objectives.py
  tests/test_mapped_xgrid_transport_evidence.py
  tests/test_run_mapped_xgrid_transport_evidence.py
  tests/test_qi_res15_gpu_campaign.py tests/test_optimization_public_scripts_cli.py
  -q --tb=short`
  with `81 passed in 42.07 s`.
- Stale import audit found no remaining imports of the deleted root workflow
  modules.

Current lane status:

- Lane 1 Tranche A: materially advanced. Root files are now `70`, down from
  `85` at the start of this tranche. Remaining root disposition should target
  solver/preconditioner utilities or output ownership rather than more
  workflow movement.
- Lane 1 overall: about `67%`.

Next best steps:

1. Continue Tranche A with a root solver utility family move into `solvers`
   (`solver_*`, `explicit_sparse*`, `preconditioner_*`,
   `native_block_factor.py`, `sparse_triangular.py`) or the output ownership
   move from `io.py` into `outputs`.
2. After root count is near the `<=55` target, execute Tranche B as the large
   `profile_response/solve.py` cut.

## 2026-06-25 Lane 1 Tranche A Solver-Utility Root Disposition

Steps taken:

1. Moved ten root solver utility modules into `sfincs_jax.solvers` without
   compatibility shims:
   `path_policy.py`, `profile_compare.py`, `progress.py`,
   `selection_policy.py`, `state.py`, `trace.py`, `krylov_dispatch.py`,
   `implicit.py`, `memory_model.py`, and `sparse_triangular.py`.
2. Updated source, tests, examples, scripts, docs, and release/validation
   manifests to canonical `sfincs_jax.solvers.*` imports or source paths.
3. Patched internal relative imports in IO/output/profile-response/PAS/x-block
   owners that still referenced the old root modules.

Current inventory:

- Package Python files: `212` (unchanged because files moved, not merged).
- Package-root Python files: `60` (down from `70`).
- `sfincs_jax.solvers`: `11` top-level Python files including `__init__.py`,
  plus preconditioner subpackages.

Validation:

- Scoped `ruff` passed for moved solver modules and touched source owners.
- `python -m py_compile` passed for moved solver modules and touched source
  owners.
- Focused solver/preconditioner tests passed:
  `python -m pytest tests/test_implicit_linear_solve_grad.py
  tests/test_sparse_triangular.py tests/test_solver_trace.py
  tests/test_solver_trace_output_formats.py tests/test_solver_state_history.py
  tests/test_solver_progress.py tests/test_solver_progress_policy.py
  tests/test_solver_path_policy.py tests/test_solver_selection_policy.py
  tests/test_solver_profile_compare.py tests/test_memory_model.py
  tests/test_krylov_dispatch.py tests/test_runtime_helper_coverage.py
  tests/test_solver_heavy_helper_coverage.py
  tests/test_rhs1_preconditioner_auto_policy.py tests/test_rhs1_handoff.py
  tests/test_io_export_and_h5_coverage.py tests/test_profile_response_sparse_pc.py
  -q --tb=short`
  with `515 passed in 8.03 s`.
- Stale import audit found no remaining imports of the deleted root solver
  utility modules.

Current lane status:

- Lane 1 Tranche A: root module count is now `60`, five above the `<=55`
  final target. Remaining root-disposition candidates are output ownership
  (`io.py` into `outputs`) and heavier solver/preconditioner implementation
  modules (`explicit_sparse*`, `preconditioner_*`, `native_block_factor.py`).
- Lane 1 overall: about `69%`.

Next best steps:

1. Finish the root-count target with one more coherent move: either start
   `io.py -> outputs` ownership or move the remaining solver implementation
   family into `solvers`.
2. Once root count reaches `<=55`, start Tranche B and cut
   `profile_response/solve.py`.

## 2026-06-25 Lane 1 Tranche A Solver/Preconditioner Implementation Root Disposition

Steps taken:

1. Moved the remaining reusable root solver/preconditioner implementation
   family into `sfincs_jax.solvers` without compatibility shims:
   `explicit_sparse.py`, `explicit_sparse_factor_builder.py`,
   `explicit_sparse_factor_policy.py`, `native_block_factor.py`,
   `preconditioner_caches.py`, `preconditioner_context.py`,
   `preconditioner_operators.py`, and `preconditioner_setup.py`.
2. Rewrote source, tests, examples, scripts, and docs to use canonical
   `sfincs_jax.solvers.*` imports and source paths.
3. Fixed moved-module relative imports, including the lazy V3 full-system
   operator import used by x-block sparse preconditioner setup.
4. Updated `plan_final.md` so it remains the single authoritative plan:
   package-root files are now below the review gate, and the next work starts
   with Tranche B/E blockers rather than more root solver moves.

Current inventory:

- Package Python files: `212` (unchanged because files moved, not merged).
- Package-root Python files: `52` (down from `60`; root-count gate met).
- `sfincs_jax.solvers`: `19` top-level Python files including `__init__.py`,
  plus preconditioner subpackages.
- `problems/profile_response`: `13` package files, with `solve.py` still the
  main blocker at `9,411` lines.
- `problems/transport_matrix`: `23` package files.
- `solvers/preconditioners`: `50` package files.

Validation:

- Scoped `ruff` passed for moved solver modules, profile-response owners,
  transport-matrix owners, and focused tests.
- `python -m py_compile` passed for moved solver modules, selected
  preconditioner family modules, profile-response owners, and
  transport-matrix owners.
- Focused solver/preconditioner regression passed:
  `python -m pytest tests/test_explicit_sparse.py
  tests/test_explicit_sparse_factor_builder.py
  tests/test_explicit_sparse_factor_policy.py
  tests/test_preconditioner_context.py tests/test_preconditioner_setup.py
  tests/test_preconditioner_caches.py tests/test_matrix_reductions.py
  tests/test_native_block_factor.py tests/test_sparse_csr.py
  tests/test_v3_sparse_pattern.py tests/test_rhs1_device_operator.py
  tests/test_rhs1_device_operator_unit.py tests/test_rhs1_full_csr_kinetic_pc.py
  tests/test_rhs1_true_operator_rescue.py
  tests/test_fortran_reduced_preconditioner.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py
  tests/test_profile_response_sparse_pc.py tests/test_transport_solve_setup.py
  tests/test_transport_solve_policy.py
  tests/test_transport_preconditioner_dispatch.py -q --tb=short`
  with `695 passed in 164.02 s`.
- Import-contract and policy-docstring tests passed with `11 passed in 0.93 s`.
- Sphinx built with `-W`.
- `git diff --check` passed.

Current lane status:

- Lane 1 Tranche A: complete for root solver/workflow/validation disposition;
  only stale-import/source-map audit maintenance remains.
- Lane 1 overall: about `71%`.
- Root-count review gate: complete at `52 <= 55`.

Next best steps:

1. Start Tranche B as the next large refactor: move sparse branch
   orchestration, sparse finalization, result payload assembly, progress replay,
   and diagnostic normalization out of `profile_response/solve.py`.
2. Do not add helper-only files; each Tranche B checkpoint should materially
   reduce `profile_response/solve.py` or delete re-export-only owners.
3. After Tranche B, execute transport/output consolidation and solver-domain
   collapse, then run the final docs/API/tests review gate.

## 2026-06-25 Lane 1 Tranche B X-Block Sparse-PC Branch Extraction

Steps taken:

1. Moved the driver-facing RHSMode=1 x-block sparse-PC GMRES branch from
   `sfincs_jax/problems/profile_response/solve.py` into the existing
   `sfincs_jax/problems/profile_response/sparse/handoff.py` owner.
2. Added `XBlockSparsePCBranchContext` and `run_xblock_sparse_pc_branch()` so
   the solve entry point passes solve-local state and callbacks explicitly.
   The numerical x-block stage kernels and final payload builders remain in
   `sparse/xblock.py`; `handoff.py` now owns the branch orchestration that
   decides and wires those stages.
3. Kept `sparse/finalization.py` in place after audit because both
   `sparse/xblock.py` and `sparse/fortran_reduced.py` still depend on its
   shared payload/result contracts. Deleting it now would create a circular
   dependency or force generic result types into the x-block owner.
4. Updated `docs/source_map.rst` and `plan_final.md` with the new owner
   boundary and current counts.

Current inventory:

- Package Python files: `212`.
- Package-root Python files: `52`.
- Package source lines: `163,535`.
- `sfincs_jax/problems/profile_response/solve.py`: `8,671` lines, down from
  `9,411` in the previous checkpoint.
- `sfincs_jax/problems/profile_response/sparse/handoff.py`: `4,623` lines
  after taking the x-block sparse-PC branch orchestration.
- `problems/profile_response`: `13` package files.

Validation:

- Scoped `ruff` passed for `profile_response/solve.py`,
  `profile_response/sparse/handoff.py`, and the focused sparse/x-block tests.
- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- Targeted x-block/export tests passed with `5 passed in 11.14 s`.
- Broader sparse/profile-response regression passed:
  `python -m pytest tests/test_profile_response_sparse_pc.py
  tests/test_v3_sparse_pattern.py tests/test_rhs1_device_operator.py
  tests/test_rhs1_device_operator_unit.py tests/test_rhs1_full_csr_kinetic_pc.py
  tests/test_rhs1_true_operator_rescue.py
  tests/test_fortran_reduced_preconditioner.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short`
  with `545 passed in 157.67 s`.
- Import-contract and policy-docstring tests passed with `11 passed in 0.57 s`.
- `git diff --check` passed.

Current lane status:

- Lane 1 Tranche A: complete except ongoing stale-reference maintenance.
- Lane 1 Tranche B: about `18%`; the first sparse-PC branch is extracted, but
  the generic sparse-PC/factor-preflight branch and final result/progress
  normalization remain in `solve.py`.
- Lane 1 overall: about `74%`.

Next best steps:

1. Continue Tranche B by extracting the generic sparse-PC/factor-preflight
   branch from `solve.py` into existing sparse owners.
2. Then move final progress replay/result normalization into
   `solver_diagnostics.py`, `diagnostics.py`, and sparse owners.
3. Only after `solve.py` is much smaller should Tranche C transport/output
   consolidation begin.

## 2026-06-25 Lane 1 Pass 3 Schur-Family Consolidation

Steps taken:

1. Consolidated the Schur RHSMode-1 implementation family into
   `sfincs_jax.solvers.preconditioners.schur.profile_response`.
2. Deleted the historical implementation files:
   `schur/rhs1.py`, `schur/rhs1_coarse_basis.py`,
   `schur/rhs1_coarse_policy.py`, and `schur/rhs1_full_csr.py`.
3. Updated package exports, tests, source-map documentation, and dependent
   x-block/symbolic-sparse imports to use the canonical Schur owner.
4. Preserved the `sfincs_jax.v3_driver` compatibility behavior by keeping a
   local Schur wrapper in `profile_response.solve` that binds builder globals
   from that module, so existing monkeypatch/debug scripts still work without
   recreating old implementation files.

Results:

- Package Python files decreased from `212` to `209`.
- `solvers/preconditioners` Python files decreased from `50` to `47`.
- `solvers/preconditioners/schur` now has two files:
  `__init__.py` and `profile_response.py`.
- No deleted Schur module paths remain in source, tests, docs, examples, or
  scripts outside generated docs build artifacts.
- `profile_response/solve.py` is `8,729` lines after the compatibility wrapper;
  the next large reduction still needs the generic sparse-PC/factor-preflight
  branch extraction from Lane 1 Pass 1.

Validation:

- `python -m ruff check` passed for the touched Schur, x-block,
  symbolic-sparse, profile-response, and focused test files.
- `python -m py_compile` passed for the touched Schur, profile-response,
  x-block, symbolic-sparse, and operator modules.
- `python -m pytest tests/test_rhs1_coarse_basis.py
  tests/test_rhs1_coarse_policy.py
  tests/test_rhs1_full_csr_schur_preconditioners.py
  tests/test_rhs1_schur_policy.py -q --tb=short` passed with `19 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_schur_precond_heuristic.py -q --tb=short` passed with
  `31 passed`.
- Broader Schur/dependency gate passed with `140 passed`.
- Broader sparse/profile-response regression gate passed with `474 passed`.

Completion:

- Lane 1 Pass 3: about `20%`; Schur is consolidated, but QI, symbolic sparse,
  x-block, PAS, and full-FP still need owner collapse.
- Lane 1 overall: about `76%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Return to Lane 1 Pass 1 and extract the generic sparse-PC/factor-preflight
   branch or result/progress normalization from `profile_response/solve.py`.
2. Continue Pass 3 afterward by collapsing symbolic-sparse RHSMode-1 naming and
   QI experiment-history files into role-based owners.
3. Keep running focused owner tests plus import-contract and Sphinx checks after
   each large consolidation batch.

## 2026-06-25 Lane 1 Pass 1 Full-Space Sparse Retry Extraction

Steps taken:

1. Moved the full-space sparse retry stage from
   `sfincs_jax/problems/profile_response/solve.py` into the existing sparse
   handoff owner, `sfincs_jax/problems/profile_response/sparse/handoff.py`.
2. Added `RHS1FullSparseRetryStageContext`,
   `RHS1FullSparseRetryStageResult`, and
   `run_rhs1_full_sparse_retry_stage` so the solve entry point now delegates
   sparse-JAX retry, host LU/ILU retry, direct-host polish, replay acceptance,
   and failure reporting as one coherent phase.
3. Added a focused unit test proving the extracted sparse-JAX full retry path
   still uses the measured-candidate replay gate with the expected
   full-scope labels.
4. Updated `plan_final.md` with current counts and the revised Lane 1 status.

Results:

- `profile_response/solve.py` decreased from `8,729` lines to `8,558` lines.
- `profile_response/sparse/handoff.py` increased from `4,623` lines to
  `4,976` lines because it now owns the explicit sparse retry context and
  execution phase.
- Package file count stayed at `209`; package-root file count stayed at `52`.
- Package lines are now `163,724`; this checkpoint improves behavior
  ownership but does not yet satisfy the final total-line reduction gate.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_profile_response_sparse_pc.py` passed.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_rhs1_full_sparse_retry_stage_uses_measured_sparse_jax_path
  -q --tb=short` passed with `1 passed`.
- `python -m pytest tests/test_profile_response_sparse_pc.py
  tests/test_rhs1_handoff.py -q --tb=short` passed with `393 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `45 passed`.
- `git diff --check` passed.

Completion:

- Lane 1 Pass 1: about `25%`; the full-space sparse retry stage is extracted,
  but the larger generic sparse-PC/factor-preflight branch and result/progress
  normalization still live in `solve.py`.
- Lane 1 overall: about `77%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Extract the reduced sparse retry branch using the same sparse handoff stage
   pattern, then collapse duplicated full/reduced sparse retry context fields
   if doing so reduces code rather than adding abstraction.
2. Move result payload assembly, progress replay, and sparse fallback summaries
   from `profile_response/solve.py` into `diagnostics.py`,
   `solver_diagnostics.py`, or existing sparse owners.
3. Continue Pass 3 after the next Pass 1 checkpoint by collapsing
   symbolic-sparse and QI experiment-history files into role-based owners.

## 2026-06-25 Lane 1 Pass 1 Reduced Sparse Retry Extraction

Steps taken:

1. Generalized `run_rhs1_full_sparse_retry_stage` so it covers both full-space
   and reduced active-DOF sparse retry paths with explicit scope, size,
   residual-vector, and operator-PC controls.
2. Replaced the reduced sparse-JAX and host sparse LU/ILU retry implementation
   in `profile_response/solve.py` with a call to the sparse handoff owner.
3. Converted the sparse retry unit test into a parametrized full/reduced
   contract, proving the measured-candidate gate receives the correct
   `sparse_jax_full` and `sparse_jax_reduced` labels and residual-vector
   behavior.
4. Updated `plan_final.md` with current counts.

Results:

- `profile_response/solve.py` decreased from `8,558` lines to `8,453` lines.
- `profile_response/sparse/handoff.py` increased from `4,976` lines to
  `5,007` lines because one stage now owns both full and reduced sparse retry
  routing.
- Package file count stayed at `209`; package-root file count stayed at `52`.
- Package lines decreased from `163,724` to `163,650` for this checkpoint.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_profile_response_sparse_pc.py` passed.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_rhs1_sparse_retry_stage_uses_measured_sparse_jax_path
  -q --tb=short` passed with `2 passed`.
- `python -m pytest tests/test_profile_response_sparse_pc.py
  tests/test_rhs1_handoff.py -q --tb=short` passed with `394 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `45 passed`.
- `git diff --check` passed.

Completion:

- Lane 1 Pass 1: about `30%`; sparse retry routing is now behavior-owned, but
  generic sparse-PC/factor-preflight and final result/progress normalization
  still need extraction.
- Lane 1 overall: about `77%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Extract sparse result/progress metadata normalization from
   `profile_response/solve.py` into `diagnostics.py` and
   `solver_diagnostics.py`, which should reduce solve size without growing
   sparse handoff further.
2. Then move the remaining generic sparse-PC/factor-preflight branch into
   existing sparse owners as one larger checkpoint.
3. Continue Pass 3 solver-family consolidation only after the next Pass 1
   diagnostic/metadata checkpoint is committed.

## 2026-06-25 Lane 1 Pass 1 SciPy Rescue Stage Extraction

Steps taken:

1. Added `RHS1ScipyRescueStageContext`,
   `RHS1ScipyRescueStageResult`, and `run_rhs1_scipy_rescue_stage` to
   `sfincs_jax/problems/profile_response/dense.py`.
2. Moved the CPU-only SciPy rescue admission, active-size-cap metadata,
   x-block skip message, rescue execution, improvement gate, and failure
   metadata out of `profile_response/solve.py`.
3. Replaced the old inlined rescue block with one dense-stage call and removed
   stale SciPy rescue policy imports from `solve.py`.
4. Added direct stage tests for a real improving SciPy rescue and for the
   active-size-cap skip metadata contract.
5. Updated `plan_final.md` with current counts and this checkpoint status.

Results:

- `profile_response/solve.py` decreased from `8,453` lines to `8,328` lines.
- `profile_response/dense.py` increased from `2,487` lines to `2,751` lines
  because it now owns the dense SciPy rescue stage as well as the low-level
  SciPy rescue solve.
- Package file count stayed at `209`; package-root file count stayed at `52`.
- Package lines increased from `163,650` to `163,789`, so the final total-line
  reduction gate remains open.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/dense.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/dense.py
  tests/test_profile_response_linear_solve.py` passed.
- `python -m pytest
  tests/test_profile_response_linear_solve.py::test_run_rhs1_scipy_rescue_stage_accepts_improving_cpu_rescue
  tests/test_profile_response_linear_solve.py::test_run_rhs1_scipy_rescue_stage_records_active_size_cap_skip
  -q --tb=short` passed with `2 passed`.
- `python -m pytest tests/test_profile_response_linear_solve.py
  tests/test_profile_response_sparse_pc.py tests/test_rhs1_handoff.py
  -q --tb=short` passed with `405 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `45 passed`.
- `git diff --check` passed.

Completion:

- Lane 1 Pass 1: about `35%`; sparse retry and SciPy rescue routing are now
  behavior-owned, but the generic sparse-PC/factor-preflight and final
  result/progress normalization paths still need extraction.
- Lane 1 overall: about `78%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Extract final result/progress metadata normalization into
   `solver_diagnostics.py` or `diagnostics.py` if it can reduce
   `solve.py` without duplicating finalization logic.
2. Move the remaining generic sparse-PC/factor-preflight branch from
   `profile_response/solve.py` into existing sparse owners as the next large
   Pass 1 checkpoint.
3. Then resume Pass 3 solver-family consolidation for symbolic sparse and QI.

## 2026-06-25 Lane 1 Batch A Validation Helper Root Cleanup

Steps taken:

1. Audited the current refactor branch against `plan_final.md` Batch A.
   Current hard counts before this checkpoint were: 209 package Python files,
   52 package-root Python files, `v3_driver.py` at 47 lines,
   `profile_response/solve.py` at 8,328 lines, and `io.py` at 4,263 lines.
2. Confirmed that there are no top-level `rhs1_*.py` or `transport_*.py`
   implementation/alias modules left in `sfincs_jax`.
3. Built an AST-backed root import inventory. The safe root cleanup target was
   validation/benchmark tooling, not core physics or public API modules.
4. Moved these root validation helpers into canonical validation owners:
   `sfincs_jax/fortran.py` -> `sfincs_jax/validation/fortran.py`,
   `sfincs_jax/fortran_profile.py` ->
   `sfincs_jax/validation/fortran_profile.py`, and
   `sfincs_jax/h5_parity.py` -> `sfincs_jax/validation/h5_parity.py`.
5. Rewrote CLI, scripts, examples, and tests to import the new validation
   owners directly. No compatibility shim was kept because all in-repository
   callers were updated and these helpers are validation tooling, not the
   primary public API.

Root-disposition table for the Batch A audit:

| Root file | Disposition |
| --- | --- |
| `__init__.py` | Public package import and JAX runtime bootstrap; keep. |
| `__main__.py` | CLI entry-point shim; keep. |
| `adaptive_maps.py` | Stable differentiable x-grid/mapping kernel; keep pending later discretization review. |
| `ambipolar.py` | Public scan/ambipolar postprocess helper; keep pending API sweep. |
| `api.py` | Public Python API contracts; keep. |
| `boozer_bc.py` | Stable geometry file reader; keep pending geometry package consolidation. |
| `classical_transport.py` | Physics kernel; keep. |
| `cli.py` | Public CLI owner; keep. |
| `collisionless.py` | Physics operator kernel; keep. |
| `collisionless_er.py` | Physics operator kernel; keep. |
| `collisionless_exb.py` | Physics operator kernel; keep. |
| `collisions.py` | Physics operator kernel; keep. |
| `compare.py` | User-facing parity/comparison helper; keep pending validation/API sweep. |
| `constrained_pas_branch.py` | Physics/numerics helper with direct tests; keep pending PAS consolidation. |
| `constraint_projection.py` | Shared constraint projection helper; keep pending problem-owner consolidation. |
| `data_fetch.py` | Equilibrium fixture fetch helper; later candidate for `validation` or `input`. |
| `diagnostics.py` | Shared physical diagnostics; keep pending output/diagnostics consolidation. |
| `fortran.py` | Moved to `validation/fortran.py`; root file deleted. |
| `fortran_profile.py` | Moved to `validation/fortran_profile.py`; root file deleted. |
| `geometry.py` | Core geometry dataclasses/kernels; keep. |
| `grids.py` | Core grid construction; keep pending discretization package review. |
| `h5_parity.py` | Moved to `validation/h5_parity.py`; root file deleted. |
| `host_refinement.py` | Host-only refinement helper; later candidate for solver/validation owner. |
| `indices.py` | Small shared indexing helper; keep pending operator-layout review. |
| `input_compat.py` | Fortran-compatible input coercions; keep as compatibility shim until `input` package is complete. |
| `io.py` | Large compatibility/output owner; Batch C target to shrink below 800 lines or delete. |
| `jax_geometry_adapters.py` | Geometry integration adapter; keep pending geometry package consolidation. |
| `magnetic_drifts.py` | Physics operator kernel; keep. |
| `namelist.py` | Public namelist parser; keep. |
| `pas_smoother.py` | PAS numerical helper; later candidate for PAS preconditioner consolidation. |
| `paths.py` | Shared path/equilibrium resolution helper; keep. |
| `periodic_stencil.py` | Shared stencil helper; keep pending discretization package review. |
| `petsc_binary.py` | Moved to `validation/petsc_binary.py`; root file deleted in the next Batch A checkpoint. |
| `phi1_newton_linear.py` | Phi1 linear helper; later candidate for `problems.profile_response.phi1_newton`. |
| `phi1_newton_policy.py` | Phi1 policy helper; later candidate for `problems.profile_response.phi1_newton`. |
| `plotting.py` | Public plotting helper; keep. |
| `postprocess_upstream.py` | CLI/example postprocess helper; later candidate for workflows/output. |
| `profiling.py` | Shared profiling helper; keep pending validation/solver split. |
| `residual.py` | Shared residual helpers; keep pending problem residual consolidation. |
| `scans.py` | User-facing scan helper; later candidate for workflows. |
| `sensitivity.py` | Public sensitivity/autodiff owner; keep. |
| `solver.py` | Core linear solver owner; keep pending solver package consolidation. |
| `structured_velocity.py` | Velocity-grid helper; keep pending discretization package review. |
| `v3.py` | Fortran-v3-compatible grid/geometry setup owner; keep pending naming/API review. |
| `v3_driver.py` | 47-line compatibility shim; keep until final import sweep or delete if all legacy imports migrate. |
| `v3_fblock.py` | Core v3 f-block operator; keep pending operator package consolidation. |
| `v3_results.py` | Typed solve-result contracts; keep pending result/API consolidation. |
| `v3_sparse_pattern.py` | Sparse pattern utility; keep pending operator/sparse consolidation. |
| `v3_system.py` | Core v3 full-system operator; keep pending operator package consolidation. |
| `vmec_geometry.py` | VMEC geometry evaluator; keep pending geometry package consolidation. |
| `vmec_wout.py` | VMEC wout reader; keep pending geometry/input package consolidation. |
| `xgrid.py` | Speed-grid construction; keep pending discretization package review. |

Results:

- Package-root files decreased from `52` to `49`.
- Package Python files stayed at `209` because this was a domain move, not a
  file deletion.
- No root `rhs1_*` or `transport_*` files exist.
- `v3_driver.py` remains a 47-line compatibility shim.

Validation:

- `python -m py_compile sfincs_jax/validation/fortran.py
  sfincs_jax/validation/fortran_profile.py
  sfincs_jax/validation/h5_parity.py sfincs_jax/cli.py
  scripts/summarize_fortran_v3_profile.py
  scripts/run_zenodo_vmec_parity_campaign.py
  scripts/compare_v3_example_suite.py
  examples/parity/geometry_scheme4_parity.py
  examples/sfincs_examples/run_sfincs_jax.py
  tests/test_fortran_profile.py tests/test_h5_parity.py
  tests/test_helper_module_coverage.py` passed.
- `python -m ruff check` over the same touched source/script/example/test set
  passed.
- `python -m pytest tests/test_fortran_profile.py tests/test_h5_parity.py
  tests/test_helper_module_coverage.py
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `23 passed`.
- `git diff --check` passed.

Completion:

- Lane 1 Batch A: about `55%`; the root-disposition table exists and one safe
  root cleanup batch landed, but final root cleanup and source-map refresh are
  still pending.
- Lane 1 structural consolidation: about `79%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Validate and commit this Batch A root cleanup.
2. Continue Batch A with `postprocess_upstream.py`, `scans.py`, and
   `data_fetch.py` only if their imports can be rewritten as a single
   ownership batch without adding compatibility shims.
3. Resume Batch B by moving final result/progress metadata normalization and
   then the generic sparse-PC/factor-preflight branch out of
   `profile_response/solve.py`.

## 2026-06-25 Lane 1 Batch A PETSc Binary Reader Root Cleanup

Steps taken:

1. Audited `petsc_binary.py` imports. It is used by parity tests, debug
   scripts, and educational examples to read PETSc Vec/AIJ binary artifacts,
   not by production solve orchestration.
2. Moved `sfincs_jax/petsc_binary.py` to
   `sfincs_jax/validation/petsc_binary.py`.
3. Rewrote all in-repository imports and `docs/api.rst` to reference
   `sfincs_jax.validation.petsc_binary` directly. No root compatibility shim
   was kept because this is validation/debug tooling and the PR is explicitly
   migrating internal imports to canonical owners.

Results:

- Package-root files decreased from `49` to `48`, satisfying the Batch A root
  target from `plan_final.md`.
- Package Python files stayed at `209`.
- No top-level `rhs1_*`, `transport_*`, `fortran*`, `h5_parity.py`, or
  `petsc_binary.py` files remain in the package root.

Validation:

- `git diff --name-only --diff-filter=ACMR -- '*.py' | xargs python -m
  py_compile` passed for the Python files touched by the PETSc import rewrite.
- `python -m pytest tests/test_v3_sparse_pattern.py
  tests/test_full_system_matvec_parity.py tests/test_full_system_rhs_parity.py
  tests/test_transport_matrix_rhsmode2_parity.py
  tests/test_transport_matrix_rhsmode3_parity.py
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `176 passed`.
- `python -m ruff check sfincs_jax/validation/petsc_binary.py
  sfincs_jax/validation/fortran.py
  sfincs_jax/validation/fortran_profile.py
  sfincs_jax/validation/h5_parity.py` passed.
- Direct imports from `sfincs_jax.validation.petsc_binary`,
  `sfincs_jax.validation.fortran`, `sfincs_jax.validation.fortran_profile`,
  and `sfincs_jax.validation.h5_parity` passed.
- `git diff --check` passed.
- A full ruff pass over every import-rewritten parity/example file was not used
  as a gate for this checkpoint because it exposed pre-existing `E402`/`E741`
  lint debt in those parity scripts and tests that is unrelated to the module
  move. That debt remains for Batch E/test cleanup rather than this root-owner
  move.

Completion:

- Lane 1 Batch A: about `70%`; the root-disposition table exists and the
  first root target (`<=48` files) is reached. Remaining Batch A work is
  limited to source-map/API cleanup and deciding whether `data_fetch.py`,
  `postprocess_upstream.py`, or `scans.py` can move without public API churn.
- Lane 1 structural consolidation: about `80%`.
- Overall refactor/review-ready PR goal: not complete.

Next best steps:

1. Validate and commit this PETSc binary reader move.
2. If validation is clean, stop Batch A root moves unless another safe
   validation/workflow helper can move as a batch; otherwise proceed to
   Batch B profile-response result/progress normalization.

## 2026-06-26 Final Lane 1 Consolidation Plan Refresh

Steps taken:

1. Re-audited the active refactor branch after the validation-helper and PETSc
   binary-reader root cleanups.
2. Confirmed the current structural inventory:
   package source files: `209`;
   package-root Python files: `48`;
   `problems/profile_response` including `sparse`: `21`;
   `problems/transport_matrix` including `parallel`: `28`;
   `solvers/preconditioners`: `47`.
3. Identified the current large-file pressure points:
   `profile_response/solve.py` at `8,328` lines,
   `profile_response/policies.py` at `6,885` lines,
   `io.py` at `4,263` lines,
   sparse profile-response owners between about `1.1k` and `5.0k` lines,
   and fragmented transport/QI/symbolic/x-block owners.
4. Updated `plan_final.md` so it is the single authoritative consolidation
   plan and so the remaining work is only four implementation/review sweeps:
   Batch B profile-response sparse/finalization collapse,
   Batch C transport/output/root ownership collapse,
   Batch D solver/preconditioner family collapse,
   and Batch E docs/API/tests/review readiness.
5. Explicitly recorded that more root-file churn is not the priority because
   the root count is already at the `<=48` gate. The next implementation
   checkpoint should materially shrink `profile_response/solve.py` without
   adding a new helper-only file.

Results:

- `plan_final.md` now forbids new micro-tranche plans, helper-only refactor
  commits, and new implementation shards unless the same commit deletes or
  merges a larger owner.
- Batch A is effectively closed except for source-map/API cleanup.
- The next concrete target is the generic sparse-PC/factor-preflight branch in
  `profile_response/solve.py`, using existing sparse/diagnostics owners.

Progress:

- Lane 1 structural consolidation: about `80%`.
- Batch A root/boundary sweep: about `90%`.
- Batch B profile-response owner collapse: about `25%`.
- Batch C transport/output/root collapse: about `20%`.
- Batch D solver/preconditioner family collapse: about `25%`.
- Batch E docs/API/tests/review gate: about `20%`.

Next best steps:

1. Implement Batch B as one coherent code move: generic sparse-PC active setup,
   operator/backend selection, factor/preflight admission, retry metadata,
   progress replay, and final sparse payload normalization should leave
   `solve.py` and move into existing owners.
2. Run focused RHSMode-1 sparse/profile-response tests, scoped ruff,
   py_compile, and `git diff --check`.
3. Commit and push the Batch B checkpoint only if the source tree shrinks
   materially and no new implementation file is introduced.

## 2026-06-26 Lane 1 Batch B Fortran-Reduced X-Block Backend Extraction

Steps taken:

1. Moved the fortran-reduced x-block sparse-PC backend implementation out of
   `sfincs_jax/problems/profile_response/solve.py` and into the existing
   sparse owner `sfincs_jax/problems/profile_response/sparse/handoff.py`.
2. Added `FortranReducedXBlockBackendContext` and
   `solve_fortran_reduced_xblock_backend` so the driver passes a typed context
   and receives the same linear-solve payload as before.
3. Removed the moved fortran-reduced x-block builder/policy/final-payload
   imports from `solve.py`.
4. Added the new sparse-owner context/helper to `handoff.py.__all__` and to
   the profile-response import-contract test.
5. Updated `plan_final.md` with the new `solve.py` count and Batch B status.

Results:

- `profile_response/solve.py` decreased from `8,328` lines to `8,104` lines.
- No new implementation file was created.
- `profile_response/sparse/handoff.py` is larger, but it is the existing owner
  for sparse branch handoff and backend orchestration.
- Package source-file count remains `209`.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m pytest tests/test_profile_response_sparse_pc.py
  tests/test_rhs1_handoff.py tests/test_domain_package_import_contracts.py -q
  --tb=short` passed with `402 passed`.
- `python -m pytest tests/test_profile_response_diagnostics.py
  tests/test_rhs1_fortran_reduced_symbolic_sparse.py
  tests/test_rhs1_xblock_policy.py tests/test_rhs1_xblock_sparse_host_policy.py
  tests/test_v3_driver_sparse_helper_coverage.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `140 passed`.
- `python -m pytest tests/test_sparse_assembly.py
  tests/test_rhs1_xblock_block_jacobi.py
  tests/test_rhs1_sxblock_tz_sparse_host.py
  tests/test_rhs1_active_projected_xblock.py -q --tb=short` passed with
  `17 passed`.
- `git diff --check` passed before the plan-log update.

Progress:

- Lane 1 structural consolidation: about `81%`.
- Batch A root/boundary sweep: about `90%`.
- Batch B profile-response owner collapse: about `30%`.
- Batch C transport/output/root collapse: about `20%`.
- Batch D solver/preconditioner family collapse: about `25%`.
- Batch E docs/API/tests/review gate: about `20%`.

Next best steps:

1. Continue Batch B with the generic sparse-PC setup/factor-preflight branch:
   active setup, preconditioner-operator selection, backend selection, pattern
   setup, factor policy, memory-budget admission, and preflight/rescue metadata
   should move into existing sparse owners.
2. After the generic branch is extracted, move sparse final payload/progress
   normalization to `sparse/finalization.py`, `solver_diagnostics.py`, or
   `diagnostics.py` without adding a new file.
3. Only then proceed to Batch C transport/output collapse.

## 2026-06-26 Lane 1 Batch B Generic Sparse-PC Setup Extraction

Steps taken:

1. Added `SparsePCGenericBranchSetupContext`,
   `SparsePCGenericBranchSetupResult`, and
   `build_sparse_pc_generic_branch_setup` to the existing sparse handoff owner.
2. Moved generic sparse-PC active-DOF setup, preconditioner-operator
   selection, fortran-reduced backend selection, sparse-pattern setup,
   factor-policy resolution, and memory-budget admission out of
   `profile_response/solve.py`.
3. Updated `solve.py` so the driver calls the typed setup stage, then routes to
   the x-block backend or continues to factor/rescue phases.
4. Exported the new setup context/helper from `sparse/handoff.py` and pinned
   them in the profile-response import-contract test.
5. Updated `plan_final.md` with the new `solve.py` count.

Results:

- `profile_response/solve.py` decreased from `8,104` lines to `8,057` lines.
- The combined 2026-06-26 Batch B sparse-owner work reduced `solve.py` from
  `8,328` to `8,057` lines.
- No new implementation file was created.
- Package source-file count remains `209`.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_profile_response_sparse_pc.py tests/test_rhs1_handoff.py -q
  --tb=short` passed with `402 passed`.
- `python -m pytest tests/test_profile_response_diagnostics.py
  tests/test_rhs1_fortran_reduced_symbolic_sparse.py
  tests/test_rhs1_xblock_policy.py tests/test_rhs1_xblock_sparse_host_policy.py
  tests/test_v3_driver_sparse_helper_coverage.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `140 passed`.
- `python -m pytest tests/test_sparse_assembly.py
  tests/test_rhs1_xblock_block_jacobi.py
  tests/test_rhs1_sxblock_tz_sparse_host.py
  tests/test_rhs1_active_projected_xblock.py -q --tb=short` passed with
  `17 passed`.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `82%`.
- Batch A root/boundary sweep: about `90%`.
- Batch B profile-response owner collapse: about `34%`.
- Batch C transport/output/root collapse: about `20%`.
- Batch D solver/preconditioner family collapse: about `25%`.
- Batch E docs/API/tests/review gate: about `20%`.

Next best steps:

1. Continue Batch B by moving the direct-tail structured setup and
   factor-preflight/rescue policy-variable initialization out of `solve.py`
   into existing sparse owners.
2. Then move sparse final payload/progress normalization out of `solve.py`.
3. Once profile-response `solve.py` is substantially smaller, start Batch C
   transport/output/root consolidation.

## 2026-06-26 Lane 1 Batch B Direct-Tail Factor Setup Extraction And Final Plan Refresh

Steps taken:

1. Added `SparsePCDirectTailFactorSetupContext`,
   `SparsePCDirectTailFactorSetupResult`, and
   `build_sparse_pc_direct_tail_factor_setup` to the existing sparse handoff
   owner.
2. Moved direct-tail materialization, structured-PC admission/build, direct
   reduced-Pmat setup metadata, structured-cache handoff, host-factor fallback,
   and direct-tail setup timing out of `profile_response/solve.py`.
3. Updated `solve.py` so the driver calls the typed direct-tail setup stage and
   receives the same variables needed by later sparse-PC finalization.
4. Exported the new setup context/helper from `sparse/handoff.py` and pinned
   them in the profile-response import-contract test.
5. Refreshed `plan_final.md` as the single authoritative consolidation plan:
   the current inventory is now explicit, the stale direct-tail/generic setup
   blockers are removed, and the remaining work is reduced to four broad
   sweeps: profile-response finalization, transport/output/root collapse,
   solver/preconditioner family collapse, and review-readiness.

Results:

- `profile_response/solve.py` decreased from `8,057` lines to `7,930` lines.
- The combined 2026-06-26 Batch B sparse-owner work reduced `solve.py` from
  `8,328` to `7,930` lines.
- No new implementation file was created.
- Package source-file count remains `209`; package-root files remain `48`.
- `plan_final.md` is still the only active refactor plan; `plan.md` remains
  the historical execution log.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_profile_response_sparse_pc.py tests/test_rhs1_handoff.py -q
  --tb=short` passed with `402 passed`.
- `python -m pytest tests/test_profile_response_diagnostics.py
  tests/test_rhs1_fortran_reduced_symbolic_sparse.py
  tests/test_rhs1_xblock_policy.py tests/test_rhs1_xblock_sparse_host_policy.py
  tests/test_v3_driver_sparse_helper_coverage.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `140 passed`.
- `python -m pytest tests/test_sparse_assembly.py
  tests/test_rhs1_xblock_block_jacobi.py
  tests/test_rhs1_sxblock_tz_sparse_host.py
  tests/test_rhs1_active_projected_xblock.py -q --tb=short` passed with
  `17 passed`.

Progress:

- Lane 1 structural consolidation: about `83%`.
- Batch A root/boundary sweep: about `90%`.
- Batch B profile-response owner collapse: about `40%`.
- Batch C transport/output/root collapse: about `20%`.
- Batch D solver/preconditioner family collapse: about `25%`.
- Batch E docs/API/tests/review gate: about `20%`.

Next best steps:

1. Finish Batch B by moving sparse finalization/progress replay, retry
   bookkeeping, final sparse payload normalization, and remaining
   factor-preflight/rescue-policy state out of `solve.py` into existing sparse,
   diagnostics, or solver-diagnostics owners.
2. Do not create new sparse helper files; use `sparse/handoff.py`,
   `sparse/policy.py`, `sparse/direct.py`, `sparse/finalization.py`,
   `diagnostics.py`, and `solver_diagnostics.py`.
3. After Batch B reaches its gate, execute Batch C as one transport/output/root
   ownership sweep.

## 2026-06-26 Lane 1 Batch B Direct-Tail Rescue Policy Setup Extraction

Steps taken:

1. Added `SparsePCDirectTailRescuePolicySetupContext`,
   `SparsePCDirectTailRescuePolicySetupResult`, and
   `build_sparse_pc_direct_tail_rescue_policy_setup` to the existing sparse
   handoff owner.
2. Moved direct-tail support-mode preflight, factor-preflight policy setup, and
   residual/window/active/coupled-coarse rescue-policy state construction out
   of `profile_response/solve.py`.
3. Kept the later residual-correction code behavior-preserving by exposing an
   ordered driver-state tuple from the sparse owner; this avoids unsafe dynamic
   `locals()` mutation while keeping the current driver variables intact.
4. Exported the new setup context/helper from `sparse/handoff.py` and pinned
   them in the profile-response import-contract test.
5. Updated `plan_final.md` so the remaining Batch B work is now focused on
   factor-preflight execution, residual-correction execution, sparse retry
   bookkeeping, progress replay, and final sparse payload normalization.

Results:

- `profile_response/solve.py` decreased from `7,930` lines to `7,834` lines.
- The combined 2026-06-26 Batch B sparse-owner work reduced `solve.py` from
  `8,328` to `7,834` lines.
- No new implementation file was created.
- Package source-file count remains `209`; package-root files remain `48`.
- Package source lines are temporarily higher because the sparse owner now
  carries explicit policy mapping; Batch B must still collapse execution and
  finalization boilerplate before the final package-line gate can pass.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_profile_response_sparse_pc.py tests/test_rhs1_handoff.py -q
  --tb=short` passed with `402 passed`.
- `python -m pytest tests/test_profile_response_diagnostics.py
  tests/test_rhs1_fortran_reduced_symbolic_sparse.py
  tests/test_rhs1_xblock_policy.py tests/test_rhs1_xblock_sparse_host_policy.py
  tests/test_v3_driver_sparse_helper_coverage.py
  tests/test_v3_driver_rhs1_dispatch_coverage.py -q --tb=short` passed with
  `140 passed`.
- `python -m pytest tests/test_sparse_assembly.py
  tests/test_rhs1_xblock_block_jacobi.py
  tests/test_rhs1_sxblock_tz_sparse_host.py
  tests/test_rhs1_active_projected_xblock.py -q --tb=short` passed with
  `17 passed`.

Progress:

- Lane 1 structural consolidation: about `84%`.
- Batch A root/boundary sweep: about `90%`.
- Batch B profile-response owner collapse: about `44%`.
- Batch C transport/output/root collapse: about `20%`.
- Batch D solver/preconditioner family collapse: about `25%`.
- Batch E docs/API/tests/review gate: about `20%`.

Next best steps:

1. Continue Batch B by moving factor-preflight execution and the residual
   correction execution stages out of `solve.py` into existing sparse owners.
2. Then move sparse final payload/progress normalization into existing sparse
   finalization, diagnostics, or solver-diagnostics owners.
3. After the remaining RHSMode-1 sparse execution/finalization code is owned
   outside `solve.py`, start Batch C transport/output/root consolidation.

## 2026-06-26 Superseded Consolidation Sweep Plan Refresh

Steps taken:

1. Re-audited the current source tree before any further code movement:
   `209` Python package files, `164,865` package lines, `48` package-root
   Python files, `profile_response/solve.py` at `7,834` lines,
   `profile_response` plus `sparse` at `21` files and `51,776` lines,
   `transport_matrix` plus `parallel` at `28` files and `15,026` lines, and
   `solvers/preconditioners` at `47` files and `36,992` lines.
2. Replaced the then-active Lane 1 refactor language in `plan_final.md` with a
   Sweep 0-4 consolidation plan. This checkpoint is now superseded by the
   current `plan_final.md` Batch A-E plan and remains only as historical
   execution-log provenance.
3. Added concrete file-level consolidation targets for historical root kernels,
   `profile_response/solve.py`, `profile_response/sparse`, transport
   micro-files, `io.py`, QI preconditioner files, symbolic sparse
   `rhs1_fortran_reduced.py`, and public API/docs/tests cleanup.
4. At that time, tightened review gates to emphasize fewer durable domain
   owners:
   package files `<=185` with stretch `<=175`, root files `<=44` or explicit
   shim labels, `profile_response` plus `sparse <=15` files,
   `transport_matrix` plus `parallel <=14` files,
   `solvers/preconditioners <=30` files, `solve.py <=3,500` lines, and
   `io.py <=800` lines or deleted.

Results:

- No source behavior changed; this was a planning/documentation checkpoint.
- At that checkpoint, `plan_final.md` was again the single authoritative plan
  for PR #8. This result is superseded by the current Batch A-E plan.
- `plan.md` remains an execution log; follow the current `plan_final.md`
  Batch A-E sequence, not this historical Sweep 0-4 sequence.

Progress:

- Lane 1 structural consolidation remains about `84%`; the percent does not
  increase from a plan-only refresh.
- Sweep 0 freeze/delete/route: `0%` against the new gate.
- Sweep 1 profile-response collapse: inherited work is about `44%`, but the
  new gate requires the remaining execution/finalization collapse and file-count
  cleanup.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `20%`.

Next best steps:

1. Execute Sweep 0: generate the import/use map, classify all large/root files,
   delete re-export-only aliases, route historical root kernels, and refresh
   `docs/source_map.rst`, `docs/api.rst`, and import-contract tests.
2. Execute Sweep 1: move factor-preflight execution, residual-correction
   execution, sparse retry bookkeeping, progress replay, and final sparse
   payload normalization out of `profile_response/solve.py` without adding a
   helper-only file.
3. Run focused compile, ruff, RHSMode-1 sparse/dense/Phi1/ambipolar/sensitivity
   tests, and `git diff --check` after each coherent sweep checkpoint.

## 2026-06-26 Sweep 0 Result-Contract Root Routing

Steps taken:

1. Moved RHSMode=1 result contracts (`V3LinearSolveResult`,
   `V3NewtonKrylovResult`, and `v3_linear_solve_result_from_payload`) into the
   existing `sfincs_jax.problems.profile_response.solver_diagnostics` owner.
2. Moved `V3TransportMatrixSolveResult` into the existing
   `sfincs_jax.problems.transport_matrix.finalize` owner.
3. Replaced `sfincs_jax/v3_results.py` with a 13-line compatibility facade for
   historical imports.
4. Rewrote package-internal imports in profile-response, transport, parallel,
   and workflow modules to use the new domain owners directly.
5. Updated `docs/source_map.rst`, `docs/api.rst`,
   `tests/test_domain_package_import_contracts.py`, and
   `tests/test_v3_results.py` so the result-contract ownership is explicit and
   the facade is tested as a facade.

Results:

- Package-internal imports from `sfincs_jax.v3_results`: `0`.
- `sfincs_jax/v3_results.py` decreased from `119` lines to `13` lines.
- Package file count remains `209`; package-root file count remains `48`.
- Package source lines are `164,861`, below the `164,865` consolidation
  baseline.
- No source behavior changed; this checkpoint only moved result-contract
  ownership and preserved compatibility imports.

Validation:

- `python -m py_compile` on all touched Python modules and tests passed.
- `python -m ruff check` on all touched Python modules and tests passed.
- `python -m pytest tests/test_v3_results.py
  tests/test_profile_response_finalization.py
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `16 passed`.
- `python -m pytest tests/test_full_system_newton_krylov.py
  tests/test_full_system_operator_jit.py tests/test_write_output_return_results.py
  -q --tb=short` passed with `10 passed`.
- `python -m pytest tests/test_transport_parallel.py
  tests/test_transport_streaming_outputs.py
  tests/test_transport_matrix_rhsmode2_parity.py
  tests/test_transport_matrix_rhsmode3_parity.py -q --tb=short` passed with
  `36 passed`.
- A direct compatibility import check confirmed `sfincs_jax.v3_results`
  re-exports the domain-owned classes and `sfincs_jax.v3_driver` still imports
  as the profile-response solve module.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `85%`.
- Sweep 0 freeze/delete/route: about `15%`; result-contract routing is done,
  but the heavier root kernels still need routing.
- Sweep 1 profile-response collapse: about `44%`.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `22%`.

Next best steps:

1. Continue Sweep 0 with `v3_sparse_pattern.py`, because it has a clearer
   domain owner (`operators/profile_response` or sparse layout) than
   `v3_system.py` and is heavily used by sparse/preconditioner tests.
2. Then route `v3_fblock.py` and `v3_system.py` behind operator-domain owners
   while keeping compatibility facades until examples and external imports are
   migrated.
3. After remaining root kernels are routed, resume Sweep 1 and collapse the
   remaining factor-preflight/residual-correction/final sparse payload code out
   of `profile_response/solve.py`.

## 2026-06-26 Sweep 0 Sparse-Pattern Root Routing

Steps taken:

1. Moved conservative and Fortran-reduced sparse structural patterns from the
   historical root module `sfincs_jax/v3_sparse_pattern.py` into the
   operator-domain owner
   `sfincs_jax/operators/profile_response/sparse_pattern.py`.
2. Removed the historical root file instead of keeping a compatibility facade,
   because package-internal imports, docs, examples, scripts, and tests no
   longer import `sfincs_jax.v3_sparse_pattern`.
3. Rewired sparse-pattern imports in profile-response full-system operators,
   sparse direct/preconditioner paths, transport-matrix sparse/direct helpers,
   and focused tests to use the operator-domain owner.
4. Updated `docs/source_map.rst`, `docs/api.rst`,
   `docs/performance_techniques.rst`, and
   `tests/test_domain_package_import_contracts.py` so the sparse-pattern
   ownership is explicit and import-contract tested.
5. Fixed a real setup inefficiency exposed by the x-block backend regression:
   `fortran_reduced_pc_gmres` with backend `xblock` now returns before
   building the monolithic Fortran-reduced sparse pattern. The generic branch
   setup records a deferred x-block pattern scope with zero monolithic pattern
   build time, and `solve.py` continues to dispatch directly to the x-block
   backend.

Results:

- Package-internal imports from `sfincs_jax.v3_sparse_pattern`: `0`.
- `sfincs_jax/v3_sparse_pattern.py` was removed from the package root.
- Package file count remains `209`; package-root file count decreases from
  `48` to `47`.
- Package source lines are `164,891`. This is temporarily above the
  `164,865` consolidation baseline because the x-block guard added explicit
  admission metadata; the review-ready target remains net source-line
  reduction below the baseline after later deletion/merge sweeps.
- `operators/profile_response` now owns `12` files and `15,268` lines,
  including the sparse-pattern owner.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/sparse/handoff.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/operators/profile_response/sparse_pattern.py
  tests/test_v3_sparse_pattern.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/sparse/handoff.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/operators/profile_response/sparse_pattern.py
  tests/test_v3_sparse_pattern.py` passed.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system
  -q --tb=short` passed with `1 passed`.
- `python -m pytest tests/test_transport_sparse_direct_solve.py
  tests/test_v3_driver_sparse_helper_coverage.py
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `28 passed`.
- `python -m pytest tests/test_v3_sparse_pattern.py
  tests/test_rhs1_device_operator.py -q --tb=short` passed with `135 passed`.

Progress:

- Lane 1 structural consolidation: about `86%`.
- Sweep 0 freeze/delete/route: about `30%`; result contracts and sparse
  structural patterns are routed, while `v3_fblock.py`, `v3_system.py`, and
  possible `v3.py` compatibility disposition remain.
- Sweep 1 profile-response collapse: about `44%`.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `23%`.

Next best steps:

1. Continue Sweep 0 with `v3_fblock.py`, because it should become a
   profile-response operator-domain owner and is the next historical root
   implementation module after sparse patterns.
2. Then route `v3_system.py` behind an operator/problem-domain owner and decide
   whether `v3.py` remains a documented public compatibility facade.
3. Resume Sweep 1 only after the remaining root-kernel routing is finished:
   move sparse finalization/progress normalization, factor-preflight execution,
   and residual-correction execution out of `profile_response/solve.py`
   without adding helper-only files.

## 2026-06-26 Sweep 0 F-Block Root Routing

Steps taken:

1. Moved the RHSMode-1 matrix-free kinetic f-block implementation from the
   historical root module `sfincs_jax/v3_fblock.py` into the operator-domain
   owner `sfincs_jax/operators/profile_response/fblock.py`.
2. Removed the historical root file instead of keeping a compatibility facade,
   because package-internal imports, docs, examples, scripts, and focused
   tests now import `sfincs_jax.operators.profile_response.fblock`.
3. Rewired direct imports in `v3_system.py`, `residual.py`, ambipolar helpers,
   examples, scripts, f-block tests, and residual/JVP tests to the new owner.
4. Updated `docs/source_map.rst`, `docs/api.rst`,
   `docs/physics_reference.rst`, `docs/performance_techniques.rst`, and
   `tests/test_domain_package_import_contracts.py` so the f-block ownership is
   explicit and import-contract tested.
5. Adjusted repository-root path resolution in the moved module so
   geometryScheme 5/11/12 equilibrium lookup still searches the checked
   fixture and package-data locations after the file move.

Results:

- Package-internal imports from `sfincs_jax.v3_fblock`: `0`.
- `sfincs_jax/v3_fblock.py` was removed from the package root.
- Package file count remains `209`; package-root file count decreases from
  `47` to `46`.
- Package source lines are `164,903`. This is still temporarily above the
  `164,865` consolidation baseline because the owner-map, docs, and import
  contracts added lines; the review-ready target remains net source-line
  reduction below the baseline after deletion/merge sweeps.
- `operators/profile_response` now owns `13` files and `16,253` lines,
  including the sparse-pattern and f-block owners.

Validation:

- `python -m py_compile sfincs_jax/operators/profile_response/fblock.py
  sfincs_jax/v3_system.py sfincs_jax/residual.py
  sfincs_jax/problems/ambipolar.py tests/test_v3_fblock_smoke.py
  tests/test_rhs1_fblock_assembly.py tests/test_domain_package_import_contracts.py`
  passed.
- `python -m ruff check sfincs_jax/operators/profile_response/fblock.py
  sfincs_jax/v3_system.py sfincs_jax/residual.py
  sfincs_jax/problems/ambipolar.py tests/test_v3_fblock_smoke.py
  tests/test_rhs1_fblock_assembly.py tests/test_domain_package_import_contracts.py`
  passed.
- `python -m pytest tests/test_v3_fblock_smoke.py tests/test_residual_jvp.py
  tests/test_input_compat.py tests/test_domain_package_import_contracts.py
  -q --tb=short` passed with `33 passed`.
- `python -m pytest tests/test_rhs1_collisionless_stencils.py
  tests/test_rhs1_collision_stencils.py tests/test_fblock_pas_matvec_parity.py
  tests/test_fblock_fokker_planck_matvec_parity.py
  tests/test_fblock_fused_matvec.py tests/test_rhs1_fblock_assembly.py
  -q --tb=short` passed with `45 passed`.
- A direct import check confirmed
  `sfincs_jax.operators.profile_response.fblock.V3FBlockOperator.__module__`
  is the new owner and `sfincs_jax.v3_fblock` is no longer importable.

Progress:

- Lane 1 structural consolidation: about `87%`.
- Sweep 0 freeze/delete/route: about `45%`; result contracts, sparse
  structural patterns, and the f-block are routed, while `v3_system.py` and
  possible `v3.py` compatibility disposition remain.
- Sweep 1 profile-response collapse: about `44%`.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `24%`.

Next best steps:

1. Continue Sweep 0 with `v3_system.py`, routing the full-system operator into
   an operator/problem-domain owner and updating imports/docs/tests in the same
   checkpoint.
2. Decide whether `v3.py` remains a public compatibility facade or can be
   split into geometry/grid owners without breaking documented public usage.
3. Resume Sweep 1 only after root-kernel routing is complete: move sparse
   finalization/progress normalization, factor-preflight execution, and
   residual-correction execution out of `profile_response/solve.py` without
   adding helper-only files.

## 2026-06-26 Sweep 0 Full-System Root Routing

Steps taken:

1. Moved the matrix-free full-system profile-response operator from the
   historical root module `sfincs_jax/v3_system.py` into the operator-domain
   owner `sfincs_jax/operators/profile_response/system.py`.
2. Removed the historical root file instead of keeping a compatibility facade,
   because package-internal imports, docs, examples, scripts, and focused
   tests now import `sfincs_jax.operators.profile_response.system`.
3. Rewired imports across profile-response, transport-matrix, preconditioner,
   residual, output, examples, scripts, and tests to the new owner.
4. Updated docs and import contracts so `operators.profile_response.system`
   owns `V3FullSystemOperator`, `full_system_operator_from_namelist`, and
   `apply_v3_full_system_operator_cached`.
5. Adjusted repository-root path resolution in the moved module so
   geometryScheme 5/11/12 equilibrium lookup still searches checked fixtures
   and package-data locations after the file move.

Results:

- Package-internal imports from `sfincs_jax.v3_system`: `0`.
- `sfincs_jax/v3_system.py` was removed from the package root.
- Package file count remains `209`; package-root file count decreases from
  `46` to `45`.
- Package source lines are `164,907`; still above the `164,865`
  consolidation baseline until later deletion/merge sweeps remove more
  compatibility and implementation surface.
- `operators/profile_response` now owns `14` files and `18,368` lines.

Validation:

- `python -m py_compile sfincs_jax/operators/profile_response/system.py
  sfincs_jax/residual.py sfincs_jax/constraint_projection.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/transport_matrix/solve.py
  tests/test_full_system_operator_jit.py tests/test_v3_system_cached_matvec.py`
  passed.
- `python -m ruff check sfincs_jax/operators/profile_response/system.py
  sfincs_jax/residual.py sfincs_jax/constraint_projection.py
  sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/transport_matrix/solve.py
  tests/test_full_system_operator_jit.py tests/test_v3_system_cached_matvec.py`
  passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_full_system_operator_jit.py tests/test_v3_system_cached_matvec.py
  tests/test_full_system_rhs_parity.py tests/test_full_system_residual_parity.py
  tests/test_full_system_residual_jvp.py tests/test_full_system_matvec_parity.py
  -q --tb=short` passed with `45 passed`.
- `python -m pytest tests/test_transport_matrix_rhsmode2_parity.py
  tests/test_transport_matrix_rhsmode3_parity.py
  tests/test_transport_streaming_outputs.py -q --tb=short` passed with
  `21 passed`.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_conservative_sparse_pattern_covers_pas_fortran_matrix
  tests/test_v3_sparse_pattern.py::test_conservative_sparse_pattern_covers_fp_fortran_matrix
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_pc_operator_preserves_angular_coupling
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system
  tests/test_rhs1_device_operator.py -q --tb=short` passed with `7 passed`.

Progress:

- Lane 1 structural consolidation: about `88%`.
- Sweep 0 freeze/delete/route: about `60%`; result contracts, sparse
  structural patterns, f-block, and full-system operator are routed. The
  remaining Sweep 0 decision is whether `v3.py` stays as a compatibility
  facade or is split into geometry/grid domain owners.
- Sweep 1 profile-response collapse: about `44%`.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `25%`.

Next best steps:

1. Finish Sweep 0 by auditing `v3.py` external/public usage and deciding
   whether to keep it as a compatibility facade or move grids/geometry builders
   to durable domain owners.
2. If `v3.py` cannot be deleted safely, document it explicitly as a public
   compatibility facade with deletion conditions and import-contract coverage.
3. Resume Sweep 1 after that: move sparse finalization/progress normalization,
   factor-preflight execution, and residual-correction execution out of
   `profile_response/solve.py` without adding helper-only files.

## 2026-06-26 Sweep 0 V3 Grid/Geometry Root Deletion

Steps taken:

1. Moved the SFINCS-v3-compatible grid/geometry implementation from the
   historical root module `sfincs_jax/v3.py` into
   `sfincs_jax/discretization/v3.py`.
2. Rewrote package-internal imports to the domain owner, including diagnostics,
   I/O, output-cache helpers, profile-response f-block/full-system operators,
   transport parallel runtime, and profile-response solve orchestration.
3. Migrated docs, examples, scripts, and focused tests away from
   `sfincs_jax.v3` to `sfincs_jax.discretization.v3`, so the historical root
   facade could be deleted instead of preserved.
4. Updated `docs/api.rst`, `docs/source_map.rst`,
   `docs/performance_techniques.rst`, `docs/inputs.rst`,
   `docs/validation_matrix.rst`, `docs/adaptive_speed_grid.rst`,
   `docs/usage.rst`, and `tests/test_domain_package_import_contracts.py` so
   `discretization.v3` is documented and tested as the implementation owner.
5. Adjusted repository-root path resolution in the moved implementation so
   geometryScheme 5/11/12 equilibrium lookup still searches checked fixtures
   and package-data locations after the move.

Results:

- Package-internal imports from `sfincs_jax.v3`: `0`.
- Repo references to `sfincs_jax.v3` or `sfincs_jax/v3.py`: `0`, except the
  intentional historical-owner note in `docs/source_map.rst`.
- `sfincs_jax/v3.py` was deleted.
- Package file count remains `209`; package-root file count decreases from
  `45` to `44`, meeting the Sweep 0 package-root target.
- Package source lines are `164,911`; still above the `164,865`
  consolidation baseline until later deletion/merge sweeps remove more
  implementation and compatibility surface.

Validation:

- `python -m py_compile sfincs_jax/discretization/v3.py
  sfincs_jax/diagnostics.py sfincs_jax/io.py
  sfincs_jax/operators/profile_response/fblock.py
  sfincs_jax/operators/profile_response/system.py
  sfincs_jax/problems/profile_response/solve.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/discretization/v3.py
  sfincs_jax/diagnostics.py sfincs_jax/io.py
  sfincs_jax/operators/profile_response/fblock.py
  sfincs_jax/operators/profile_response/system.py
  sfincs_jax/problems/profile_response/solve.py
  tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py
  tests/test_mapped_xgrid_v3.py tests/test_v3_geometry_scheme4.py
  tests/test_geometry_scheme11_parity.py tests/test_write_output_return_results.py
  tests/test_full_system_operator_jit.py tests/test_v3_system_cached_matvec.py
  tests/test_v3_fblock_smoke.py tests/test_transport_matrix_rhsmode2_parity.py
  tests/test_transport_matrix_rhsmode3_parity.py -q --tb=short` passed with
  `46 passed`.

Progress:

- Lane 1 structural consolidation: about `89%`.
- Sweep 0 freeze/delete/route: `100%` for historical root implementation
  routing. Remaining package-root work is now covered by later Sweep 2/4
  compatibility and API cleanup, not Sweep 0.
- Sweep 1 profile-response collapse: about `44%`.
- Sweep 2 transport/output/root cleanup: about `20%`.
- Sweep 3 solver/preconditioner family consolidation: about `25%`.
- Sweep 4 public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Start Sweep 1: move sparse finalization/progress normalization,
   factor-preflight execution, and residual-correction execution out of
   `profile_response/solve.py` into existing sparse, diagnostics, and
   solver-diagnostics owners.
2. Do not add helper-only files. Each Sweep 1 checkpoint should remove a
   coherent solve-phase section from `solve.py` or merge/delete existing sparse
   wrappers.
3. After Sweep 1 reduces `solve.py`, continue Sweep 2 transport/output cleanup
   and Sweep 3 solver/preconditioner file-family consolidation.

## 2026-06-26 Final Batch-Based Consolidation Plan Refresh

Steps taken:

1. Re-audited the active refactor branch file inventory after the historical
   root routing checkpoints.
2. Replaced the active Lane 1 instructions in `plan_final.md` with a single
   batch-based consolidation plan, so future work no longer follows older
   sweep/tranche/pass vocabulary.
3. Added a file-disposition matrix covering package-root files,
   profile-response orchestration, transport matrix files, output ownership,
   QI preconditioners, symbolic sparse owners, x-block/PAS/full-FP families,
   and profile-response operator files.
4. Tightened the rules for creating new files: new implementation files are
   allowed only when the same commit deletes at least two smaller files or
   moves a large implementation into a durable owner with behavior tests.

Results:

- Current package inventory: `209` Python files, `44` package-root Python
  files, and `164,911` package source lines.
- Largest remaining owners:
  `profile_response/solve.py` (`7,836` lines),
  `profile_response/policies.py` (`6,885` lines),
  `profile_response/sparse/handoff.py` (`6,605` lines),
  `operators/profile_response/full_system.py` (`5,978` lines), and
  the QI/x-block/symbolic sparse preconditioner families.
- Active Lane 1 plan is now:
  Batch A profile-response collapse,
  Batch B transport/output/root cleanup,
  Batch C solver/preconditioner family consolidation,
  Batch D public API/docs/tests/review gate.

Progress:

- Lane 1 structural consolidation remains about `89%` overall, but the
  remaining work is now grouped into four larger batches rather than small
  helper extractions.
- Batch A profile-response collapse: inherited work is about `45%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Start Batch A with factor-preflight execution and progress reporting, moving
   it from `profile_response/solve.py` into the existing sparse owner without
   adding a helper-only file.
2. Continue Batch A by moving residual-correction execution, retry
   bookkeeping, final sparse payload normalization, and solver-trace result
   normalization into existing sparse/diagnostic owners.
3. Once Batch A reaches its line/file gates, execute Batch B as one transport
   and output consolidation push, then Batch C as one solver-family
   consolidation push.

## 2026-06-26 Batch A Sparse Factor-Preflight Execution Extraction

Steps taken:

1. Added `SparsePCFactorPreflightRunContext`,
   `SparsePCFactorPreflightRunResult`, and `run_sparse_pc_factor_preflight()`
   to `sfincs_jax.problems.profile_response.sparse.handoff`, keeping the
   factor-preflight evaluator and progress messages inside the existing sparse
   owner instead of adding another helper file.
2. Replaced the inline factor-preflight execution block in
   `profile_response/solve.py` with a single sparse-owner call and local state
   assignment.
3. Removed the direct `evaluate_sparse_pc_factor_preflight`/
   `SparsePCFactorPreflightEvaluationContext` dependency from `solve.py`.
4. Updated `plan_final.md` current counts and next steps so factor-preflight
   execution is no longer listed as an open Batch A blocker.

Results:

- `profile_response/solve.py` decreased from `7,836` to `7,781` lines.
- `profile_response/sparse/handoff.py` increased from `6,605` to `6,720`
  lines because it now owns the execution/progress wrapper.
- Package file count stayed at `209`; package-root file count stayed at `44`.
- Package source lines are now about `164,971`; later Batch B/C file merges
  still need to remove enough compatibility and micro-file surface to pass the
  final source-line gate.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_structured_pc_preflight_can_fail_fast
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_explicit_structured_pc_rejection_is_fast
  tests/test_rhs1_device_operator.py -q --tb=short` passed with `6 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py -q
  --tb=short` passed with `8 passed`.
- `python -m pytest tests/test_v3_sparse_pattern.py -q --tb=short` passed
  with `132 passed`.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `90%`.
- Batch A profile-response collapse: about `48%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Continue Batch A by moving residual-correction execution and retry
   bookkeeping out of `profile_response/solve.py` into existing sparse owners.
2. Move final sparse payload normalization and solver-trace result
   normalization into `solver_diagnostics.py`, `diagnostics.py`, or
   `sparse/finalization.py`.
3. After those stages are outside `solve.py`, decide whether
   `profile_response/handoff.py` or `sparse/finalization.py` can be merged or
   deleted without creating import cycles.

## 2026-06-26 Batch A Direct-Tail Auto Preflight Retry Extraction

Steps taken:

1. Added `SparsePCAutoPreflightRetryStageContext`,
   `SparsePCAutoPreflightRetryStageResult`, and
   `run_sparse_pc_auto_preflight_retry_stage()` to the existing
   `profile_response.sparse.handoff` owner.
2. Moved the direct-tail structured-PC auto preflight retry candidate
   selection, retry preconditioner build, residual check, acceptance policy,
   progress messages, and retry metadata update out of
   `profile_response/solve.py`.
3. Kept builder injection explicit: `solve.py` still passes the current
   active-projected structured preconditioner builder and structured factor
   bundle adapter into the sparse stage, preserving monkeypatch/debug behavior
   while moving the retry logic out of the driver.
4. Fixed the extraction bug found by the full sparse-pattern shard: sparse
   `_env_value()` returns an empty string for missing keys and does not accept a
   `default=` keyword, so the stage now uses `(_env_value(...) or default)`.

Results:

- `profile_response/solve.py` decreased from `7,781` to `7,657` lines.
- `profile_response/sparse/handoff.py` increased from `6,720` to `7,029`
  lines because it now owns direct-tail auto retry execution.
- Package file count stayed at `209`; package-root file count stayed at `44`.
- Package source lines are now about `165,156`; source-line reduction remains a
  downstream Batch B/C gate after micro-file merges and compatibility cleanup.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_structured_pc_preflight_can_fail_fast
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_explicit_structured_pc_rejection_is_fast
  tests/test_rhs1_device_operator.py -q --tb=short` passed with `6 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py -q
  --tb=short` passed with `8 passed`.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_auto_retries_active_lu_after_native_preflight_failure
  -q --tb=short` passed after the `_env_value` fix.
- `python -m pytest tests/test_v3_sparse_pattern.py -q --tb=short` passed
  with `132 passed`.

Progress:

- Lane 1 structural consolidation: about `90%`.
- Batch A profile-response collapse: about `51%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Continue Batch A with the remaining residual-correction stages:
   true active submatrix/block/residual-block, true residual window, residual
   coarse, and residual window.
2. Prefer extracting a shared "candidate residual acceptance/update" helper
   inside the existing sparse owner before moving each candidate family, so the
   next move removes repeated accept/update boilerplate rather than only one
   narrow block.
3. After the residual-correction stages move, collapse final sparse payload
   normalization and then revisit whether `sparse/finalization.py` can merge
   into a lower shared sparse owner.

## 2026-06-26 Batch A True-Coupled Coarse Stage Extraction

Steps taken:

1. Added `SparsePCTrueCoupledCoarseStageContext`,
   `SparsePCTrueCoupledCoarseStageResult`, and
   `run_sparse_pc_true_coupled_coarse_stage()` to the existing
   `profile_response.sparse.handoff` owner.
2. Moved the true-operator coupled coarse auto-selection, builder invocation,
   residual recomputation, acceptance policy, progress messages, and state
   updates out of `profile_response/solve.py`.
3. Kept the actual true-operator builder and additive-memory estimator injected
   from `solve.py`, so this move does not introduce a new import cycle and
   still preserves monkeypatch/debug behavior for the current tests.
4. Updated `plan_final.md` so the controlling Batch A plan marks
   factor-preflight execution, direct-tail auto preflight retry, and the
   true-coupled coarse stage as complete.

Results:

- `profile_response/solve.py` decreased from `7,657` to `7,567` lines.
- `profile_response/sparse/handoff.py` increased from `7,029` to `7,309`
  lines because it now owns true-coupled coarse execution.
- Package file count stayed at `209`; package-root file count stayed at `44`.
- Package source lines are now about `165,346`; `profile_response` source
  lines are about `52,317`. Source-line reduction remains a downstream Batch
  B/C gate after compatibility and micro-file merges.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m pytest
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_true_coupled_coarse_records_bounded_metadata
  tests/test_v3_sparse_pattern.py::test_fortran_reduced_direct_tail_true_coupled_coarse_auto_promotes_active_lu
  -q --tb=short` passed with `2 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py -q
  --tb=short` passed with `8 passed`.
- `python -m pytest tests/test_v3_sparse_pattern.py -q --tb=short` passed
  with `132 passed`.

Progress:

- Lane 1 structural consolidation: about `90%`.
- Batch A profile-response collapse: about `54%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Continue Batch A with the remaining true-active and residual-window
   families: true active submatrix/block/residual-block, true residual window,
   residual coarse, and residual window.
2. Extract a shared sparse-owner candidate residual acceptance/update helper
   before moving those remaining families, so the next tranche removes repeated
   state mutation code rather than only another narrow block.
3. Move final sparse payload normalization and progress replay after the
   residual-correction stages are outside `solve.py`.

## 2026-06-26 Batch A Shared Residual-Candidate Update Extraction

Steps taken:

1. Added `SparsePCResidualCandidateUpdateContext`,
   `SparsePCResidualCandidateUpdateResult`, and
   `apply_sparse_pc_residual_candidate_update()` to the existing
   `profile_response.sparse.handoff` owner.
2. Replaced repeated residual-candidate admission and state mutation in the
   true-active submatrix, true-active block, true-active residual-block, true
   residual-window, residual-coarse, and residual-window paths with the shared
   sparse-owner updater.
3. Preserved the candidate builders and true residual recomputation at the
   current call site, so this is a behavior-preserving consolidation step that
   prepares the next larger stage extraction.
4. Removed direct imports of `SparsePCResidualCandidateAcceptanceContext` and
   `evaluate_sparse_pc_residual_candidate_acceptance` from
   `profile_response/solve.py`; residual admission is now consumed through the
   sparse owner.
5. Updated `plan_final.md` so the controlling Batch A plan marks the shared
   residual-candidate accept/update extraction as complete.

Results:

- `profile_response/solve.py` decreased from `7,567` to `7,539` lines.
- `profile_response/sparse/handoff.py` increased from `7,309` to `7,479`
  lines because it now owns shared residual-candidate state updates.
- Package file count stayed at `209`; package-root file count stayed at `44`.
- Package source lines are now about `165,488`; this helper extraction is a
  staging step for moving the full residual-correction family out of
  `solve.py`.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_sparse_pc_residual_candidate_acceptance_base_improvement_override_sets_passed
  tests/test_v3_sparse_pattern.py::test_true_operator_residual_window_lsq_reduces_global_residual
  tests/test_v3_sparse_pattern.py::test_residual_window_host_preconditioner_solves_targeted_kinetic_window
  tests/test_v3_sparse_pattern.py::test_residual_coarse_host_preconditioner_solves_adaptive_identity_residual
  -q --tb=short` passed with `4 passed`.
- `python -m pytest tests/test_domain_package_import_contracts.py -q
  --tb=short` passed with `8 passed`.
- `python -m pytest tests/test_v3_sparse_pattern.py -q --tb=short` passed
  with `132 passed`.

Progress:

- Lane 1 structural consolidation: about `90%`.
- Batch A profile-response collapse: about `57%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Move the full true-active/residual-window correction family into a single
   sparse-owner stage now that all candidates share the same updater.
2. Then move final sparse payload normalization and progress replay out of
   `solve.py`.
3. After Batch A reaches the line/file gates, switch to Batch B transport and
   output consolidation.

## 2026-06-26 Iteration 1 v3 Results Root Facade Deletion

Steps taken:

1. Confirmed `sfincs_jax/v3_results.py` had no package-internal imports; only
   docs and tests still used the 13-line compatibility facade.
2. Rewrote `tests/test_v3_results.py` and
   `tests/test_profile_response_finalization.py` to import result contracts
   from their canonical problem owners:
   `problems.profile_response.solver_diagnostics` and
   `problems.transport_matrix.finalize`.
3. Removed the `sfincs_jax.v3_results` API page entry and source-map entry.
4. Deleted `sfincs_jax/v3_results.py`.
5. Updated `plan_final.md` so `v3_results.py` is listed with the deleted
   historical roots and `v3_driver.py` is the only remaining `v3_*` root shim.

Results:

- Package Python files decreased from `209` to `208`.
- Package-root Python files decreased from `44` to `43`.
- Only `sfincs_jax/v3_driver.py` remains under the package root matching
  `v3_*.py`.
- Top-level `rhs1_*` and `transport_*` aliases remain deleted.

Validation to run next:

1. `python -m pytest tests/test_v3_results.py
   tests/test_profile_response_finalization.py -q --tb=short`
2. `python -m pytest tests/test_domain_package_import_contracts.py -q
   --tb=short`
3. `python -m py_compile` and `ruff` on the touched docs/source code where
   applicable.

## 2026-06-26 Batch A Residual-Correction Stage Extraction And Final Plan Refresh

Steps taken:

1. Reviewed the dirty local worktree from the interrupted consolidation pass.
   The branch had a newly added sparse-owned residual-correction stage in
   `problems/profile_response/sparse/handoff.py`, but
   `problems/profile_response/solve.py` still carried the old inline
   true-active/true-window/residual-window candidate branches.
2. Replaced the remaining inline residual-correction family in `solve.py` with
   one call to `run_sparse_pc_residual_correction_stage(...)`.
3. Removed obsolete direct imports of the residual-candidate update context and
   updater from `solve.py`; sparse candidate admission now stays inside the
   sparse owner.
4. Audited the current package topology and large files before revising the
   authoritative plan:
   `208` package Python files, `43` package-root Python files,
   `v3_driver.py` `47` lines, `profile_response/solve.py` `7,014` lines,
   `profile_response/sparse/handoff.py` `8,231` lines, `io.py` `4,264` lines,
   `problems/transport_matrix` `28` files, and
   `solvers/preconditioners` `47` files.
5. Updated `plan_final.md` so it is the single authoritative consolidation
   plan. The plan now explicitly prevents `sparse/handoff.py` from becoming a
   new monolith and defines the remaining work as four coarse batches:
   profile-response collapse, transport/output/root cleanup,
   solver/preconditioner family consolidation, and public API/docs/tests review
   gate.

Results:

- `profile_response/solve.py` decreased from `7,539` to `7,014` lines.
- `profile_response/sparse/handoff.py` increased from `7,479` to `8,231`
  lines, so Batch A now includes an explicit handoff-owner reduction gate.
- Package file count stayed at `208`; package-root file count stayed at `43`.
- The true-active, true-window, residual-coarse, and residual-window
  correction family now shares one sparse-owned execution and admission path.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py
  sfincs_jax/problems/profile_response/sparse/handoff.py` passed.
- `git diff --check` passed.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_sparse_pc_residual_candidate_acceptance_base_improvement_override_sets_passed
  tests/test_v3_sparse_pattern.py::test_true_operator_residual_window_lsq_reduces_global_residual
  tests/test_v3_sparse_pattern.py::test_residual_window_host_preconditioner_solves_targeted_kinetic_window
  tests/test_v3_sparse_pattern.py::test_residual_coarse_host_preconditioner_solves_adaptive_identity_residual
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `12 passed`.
- `python -m pytest tests/test_v3_sparse_pattern.py -q --tb=short` passed
  with `132 passed in 114.32s`.

Progress:

- Lane 1 structural consolidation: about `91%`.
- Batch A profile-response collapse: about `62%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `26%`.

Next best steps:

1. Finish Batch A by moving sparse finalization, progress replay, remaining
   retry bookkeeping, final sparse payload normalization, and fallback summaries
   out of `solve.py`.
2. Reduce `sparse/handoff.py` by moving stable groups into existing sparse
   owners (`direct.py`, `xblock.py`, `qi.py`, `fortran_reduced.py`,
   `policy.py`, or `finalization.py`) rather than creating another helper file.
3. Execute Batch B as one transport/output/root cleanup push, then Batch C as
   one solver/preconditioner family cleanup push.

## 2026-06-26 Batch A Sparse GMRES Finalization Owner Move

Steps taken:

1. Moved sparse-PC GMRES attempt helpers, stagnation/progress callback logic,
   typed finalization state builders, finalization bundle construction, and
   final GMRES payload construction from
   `sfincs_jax/problems/profile_response/sparse/handoff.py` to the existing
   `sfincs_jax/problems/profile_response/sparse/finalization.py` owner.
2. Kept the legacy sparse handoff import surface stable by importing and
   re-exporting the moved names from the compatibility layer.
3. Updated sparse-PC import-contract tests so the canonical owner is now
   `sparse.finalization`, including the monkeypatch seam for
   `finalize_sparse_pc_gmres_with_dtype_retry`.
4. Re-audited the package topology and refreshed `plan_final.md` as the single
   controlling consolidation plan with current counts and the final four-batch
   sequence.

Results:

- `profile_response/sparse/handoff.py` decreased from `8,231` to `7,577`
  lines.
- `profile_response/sparse/finalization.py` increased to `1,551` lines and is
  now the documented sparse-PC final-payload owner.
- `profile_response/solve.py` remains `7,014` lines.
- Package file count remains `208`; package-root file count remains `43`;
  package source lines are `165,674`.
- Batch A now has one less blocker: sparse GMRES/finalization ownership is
  complete, while progress replay, fallback summaries, remaining retry
  bookkeeping, and `sparse/handoff.py` owner reduction remain open.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/sparse/finalization.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/sparse/finalization.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py tests/test_profile_response_sparse_pc.py` passed.
- `python -m pytest tests/test_profile_response_sparse_pc.py -q --tb=short`
  passed with `329 passed in 3.09s`.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_sparse_finalization_module_reexports_match_compat_layer
  tests/test_profile_response_sparse_pc.py::test_sparse_krylov_helpers_live_on_sparse_pc_owner
  tests/test_profile_response_sparse_pc.py::test_sparse_pc_gmres_finalization_state_from_driver_scope_filters_scope
  tests/test_profile_response_sparse_pc.py::test_sparse_pc_gmres_finalization_bundle_from_driver_scope_groups_locals
  tests/test_profile_response_sparse_pc.py::test_finalize_sparse_pc_gmres_bundle_builds_typed_state
  tests/test_profile_response_sparse_pc.py::test_finalize_sparse_pc_gmres_with_dtype_retry_uses_explicit_finalization_contexts
  tests/test_domain_package_import_contracts.py -q --tb=short` passed with
  `14 passed`.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `92%`.
- Batch A profile-response collapse: about `66%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `27%`.

Next best steps:

1. Finish Batch A as one larger owner move: progress replay, remaining retry
   bookkeeping, final fallback summaries, and solver-trace result normalization
   go to existing sparse/dense/diagnostic owners.
2. Reduce `sparse/handoff.py` by moving direct-tail, x-block, QI, policy, and
   result-relay groups into existing sparse owners; do not create new helper
   files.
3. Then execute Batch B transport/output/root cleanup and Batch C
   solver/preconditioner family consolidation before the Batch D docs/API/test
   review gate.

## 2026-06-26 Batch A X-Block Owner Consolidation

Steps taken:

1. Moved the x-block branch/setup/stage cluster from
   `sfincs_jax/problems/profile_response/sparse/handoff.py` to the existing
   `sfincs_jax/problems/profile_response/sparse/xblock.py` owner. This moved
   x-block sparse-PC setup policy, side-policy setup, assembled-operator
   setup/preflight/device/matvec helpers, local preconditioner setup,
   moment-Schur/two-level/global-coupling stage helpers, seed policy setup,
   and the x-block sparse-PC branch orchestration.
2. Kept the legacy `sparse.handoff` import surface stable through explicit
   re-export aliases for the moved x-block names.
3. Expanded the sparse x-block import-contract test so the newly moved setup
   and stage names are proven to come from `sparse.xblock`.
4. Updated `plan_final.md` so the handoff-size gate is marked as passed and
   the remaining Batch A work is narrowed to progress/fallback/retry/result
   normalization plus direct-tail/generic sparse cleanup.
5. Updated `docs/source_map.rst` so it no longer describes x-block branch
   orchestration as owned by `sparse/handoff.py`.

Results:

- `profile_response/sparse/handoff.py` decreased from `7,577` to `4,438`
  lines, passing the `<=5,500` Batch A handoff gate.
- `profile_response/sparse/xblock.py` increased to `7,725` lines and is now
  the single x-block setup/orchestration owner for this PR.
- `profile_response/solve.py` remains `7,014` lines.
- `problems/profile_response` remains `21` files and is now `52,672` lines.
- At that checkpoint, package file count remained `208`, package-root file
  count remained `43`, and package source lines were `165,688`.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/sparse/xblock.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/sparse/xblock.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py tests/test_profile_response_sparse_pc.py` passed.
- `python -m pytest tests/test_profile_response_sparse_pc.py -q --tb=short`
  passed with `329 passed in 2.66s`.
- `python -m pytest tests/test_domain_package_import_contracts.py tests/test_profile_response_sparse_pc.py::test_sparse_xblock_module_reexports_match_compat_layer -q --tb=short`
  passed with `9 passed`.
- `python -m pytest
  tests/test_profile_response_sparse_pc.py::test_xblock_assembled_equilibration_setup_builds_row_scales
  tests/test_profile_response_sparse_pc.py::test_xblock_assembled_equilibration_setup_builds_row_and_column_scales
  tests/test_profile_response_sparse_pc.py::test_xblock_assembled_device_setup_builds_and_validates_operator
  tests/test_profile_response_sparse_pc.py::test_xblock_assembled_matvec_setup_host_counts_progress
  tests/test_profile_response_sparse_pc.py::test_xblock_assembled_matvec_setup_device_counts_progress
  -q --tb=short` passed with `5 passed`.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `93%`.
- Batch A profile-response collapse: about `73%`.
- Batch B transport/output/root cleanup: about `20%`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `28%`.

Next best steps:

1. Finish the remaining Batch A owner move: progress replay, fallback summaries,
   remaining retry bookkeeping, and final trace/result normalization should move
   to `solver_diagnostics.py`, `diagnostics.py`, or `sparse/finalization.py`.
2. Move the remaining direct-tail and generic sparse setup pieces out of
   `sparse/handoff.py` only where they land in existing durable owners.
3. Then switch to Batch B transport/output/root cleanup.

## 2026-06-26 Final Consolidation Plan Refresh

Steps taken:

1. Re-audited the current source layout after the latest Batch A commits:
   `206` package Python files, `43` package-root files, `19`
   profile-response files including `sparse/`, `28` transport-matrix files
   including `parallel/`, and `165,631` package source lines.
2. Reviewed the remaining largest owners: `sparse/xblock.py`, `solve.py`,
   `policies.py`, `operators/profile_response/full_system.py`, `sparse/qi.py`,
   `sparse/handoff.py`, `io.py`, and the transport/preconditioner micro-file
   clusters.
3. Replaced the old phase/tranche wording in `plan_final.md` with one
   authoritative Batch A-E consolidation plan. The new plan separates
   review-ready gates from stretch cleanup targets so the PR can become
   reviewable without unsafe line-count churn.
4. Recorded explicit cut/defer decisions:
   `active_dof.py` should merge into `profile_response/setup.py`;
   `sparse/policy.py` should not be quick-merged because its `_env_*` helper
   signatures conflict with `profile_response/policies.py`;
   `sparse/finalization.py` stays until import-cycle risk is eliminated; and
   no new helper/handoff/campaign files should be created.

Results:

- `plan_final.md` is now the single authoritative refactor plan.
- The required review-ready gates are: package files `<=190`, root files
  `<=44`, `profile_response/solve.py <=5,500` lines,
  `profile_response` plus `sparse <=18` files, transport matrix plus parallel
  `<=18` files, preconditioners `<=35` files, and `io.py <=1,200` lines.
- The stricter targets remain as stretch goals: package files `<=175`,
  `profile_response/solve.py <=3,500` lines, profile-response `<=15` files,
  transport matrix `<=14` files, preconditioners `<=30` files, and
  `io.py <=800` lines or deletion.

Next best steps:

1. Continue Batch A without more one-helper churn: move remaining
   result/retry/progress relays from `solve.py` into existing
   dense/sparse/diagnostic owners.
2. Execute Batch B as a transport/output/root compression pass, deleting
   transport micro-files and moving implementation from `io.py` to
   `outputs/`.
3. Execute Batch C as a solver/preconditioner family compression pass, with QI
   and symbolic-sparse filenames as the first targets.

## 2026-06-26 Batch A Active-DOF Owner Merge

Steps taken:

1. Merged `sfincs_jax/problems/profile_response/active_dof.py` into the
   existing `sfincs_jax/problems/profile_response/setup.py` owner. Active-DOF
   decisions, active index maps, full/reduced JAX gather/scatter primitives,
   PAS constraint projection, FP pitch-mode active-index selection, and final
   RHSMode-1 cleanup now live with the setup/finalization contracts that use
   them.
2. Rewired `profile_response/solve.py`, `solver_diagnostics.py`, focused
   tests, import-contract tests, API docs, and the source map to import from
   `profile_response.setup`.
3. Deleted the standalone `active_dof.py` file instead of leaving another
   compatibility shim.
4. Updated `plan_final.md` metrics and Batch A status. The
   profile-response file-count review-ready gate is now met.

Results:

- Package Python files decreased from `206` to `205`.
- `problems/profile_response` files including `sparse/` decreased from `19`
  to `18`, meeting the Batch A review-ready file-count gate.
- Package source lines decreased from `165,631` to `165,586`.
- `profile_response/solve.py` is now `7,008` lines; the remaining Batch A
  blocker is the `<=5,500` review-ready solve-sequencer gate.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/setup.py sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/solver_diagnostics.py tests/test_rhs1_active_dof.py tests/test_rhs1_active_projection.py tests/test_profile_response_active_projection.py tests/test_profile_response_sparse_pc.py tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/setup.py sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/solver_diagnostics.py tests/test_rhs1_active_dof.py tests/test_rhs1_active_projection.py tests/test_profile_response_active_projection.py tests/test_profile_response_sparse_pc.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_rhs1_active_dof.py tests/test_rhs1_active_projection.py tests/test_profile_response_active_projection.py tests/test_profile_response_setup.py tests/test_domain_package_import_contracts.py -q --tb=short` passed with `39 passed`.
- `python -m pytest tests/test_profile_response_sparse_pc.py -q --tb=short`
  passed with `329 passed`.
- `python -m pytest tests/test_profile_response_finalization.py tests/test_profile_response_auto_solve.py tests/test_profile_response_linear_solve.py -q --tb=short`
  passed with `21 passed`.

Next best steps:

1. Continue Batch A by moving final sparse-result normalization, progress
   replay, retry bookkeeping, and fallback summaries out of
   `profile_response/solve.py` into existing `solver_diagnostics.py`,
   `sparse/finalization.py`, `sparse/direct.py`, `sparse/xblock.py`,
   `sparse/qi.py`, or `dense.py`.
2. Once `solve.py <=5,500` lines, switch to Batch B transport/output/root
   compression.

## 2026-06-26 Batch B Transport Handoff Policy Merge

Steps taken:

1. Merged the transport retry/residual polish relay
   `sfincs_jax/problems/transport_matrix/handoff_policy.py` into the durable
   transport policy owner `sfincs_jax/problems/transport_matrix/policies.py`.
2. Rewired `profile_response/solve.py`, focused handoff-policy tests, import
   contract tests, API docs, and the source map to import the policy owner
   directly.
3. Deleted `handoff_policy.py` instead of leaving a compatibility shim.
4. Updated `plan_final.md` so the single authoritative consolidation plan now
   marks `handoff_policy.py` as done and keeps the remaining Batch B policy
   targets focused on `solve_policy.py` and `preconditioner_dispatch.py`.

Results:

- Package Python files decreased from `205` to `204`.
- `problems/transport_matrix` files including `parallel/` decreased from `28`
  to `27`.
- `sfincs_jax/problems/transport_matrix/policies.py` is now `747` lines and
  owns dense/sparse/direct/tzfft runtime admission plus RHSMode-3 polish
  retry policy.
- The Batch B review-ready gate remains open: transport matrix plus
  `parallel/` must still decrease to `<=18` files.

Validation:

- `python -m py_compile sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_handoff_policy.py tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_handoff_policy.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_transport_handoff_policy.py tests/test_domain_package_import_contracts.py tests/test_transport_solve_policy.py -q --tb=short` passed with `26 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.

Next best steps:

1. Continue Batch B with a larger policy/dispatch merge:
   `solve_policy.py` and `preconditioner_dispatch.py` should move into
   `transport_matrix/policies.py` or a single active-system owner, with tests
   importing the owner directly.
2. Then merge transport solver shards (`dense_lu.py`, `dense_batch.py`,
   `host_gmres.py`, `loop.py`, `iteration_stats.py`, `residual_quality.py`,
   and `sparse_direct_solve.py`) into `solve.py` or one active-system owner.
3. After Batch B reaches the transport file-count gate, return to Batch A's
   remaining `profile_response/solve.py <=5,500` review-ready gate.

## 2026-06-26 Batch B Transport Policy/Dispatch Owner Merge

Steps taken:

1. Merged `sfincs_jax/problems/transport_matrix/solve_policy.py` into the
   durable owner `sfincs_jax/problems/transport_matrix/policies.py`.
   Geometry-scheme parsing, low-memory output policy, active-DOF admission,
   dense fallback/preconditioner admission, state-vector retention, GMRES
   budgets, and per-`whichRHS` loop policy now live in the policy owner.
2. Merged `sfincs_jax/problems/transport_matrix/preconditioner_dispatch.py`
   into the same policy owner. Preconditioner-kind normalization, automatic
   transport preconditioner selection, DD/sparse-JAX environment parsing,
   reduced/full builder dispatch, and strong-preconditioner caching now live
   with the transport policy they serve.
3. Rewired `profile_response/solve.py`, `transport_matrix/active_dense.py`,
   focused transport policy/preconditioner tests, import-contract tests, API
   docs, testing docs, release notes, and the source map to import
   `transport_matrix.policies` directly.
4. Deleted both relay files instead of leaving compatibility shims.
5. Updated `plan_final.md` metrics and Batch B status so the next transport
   work is solver-shard consolidation, not another policy merge.

Results:

- Package Python files decreased from `204` to `202`.
- Package source lines decreased from `165,574` to `165,517`.
- `problems/transport_matrix` files including `parallel/` decreased from `27`
  to `25`.
- `sfincs_jax/problems/transport_matrix/policies.py` is now `2,123` lines and
  owns the transport runtime, solve, retry, active/dense, and preconditioner
  dispatch policy family.
- The Batch B transport file-count gate remains open: `25` files must decrease
  to `<=18`.

Validation:

- `python -m py_compile sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/active_dense.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_solve_policy.py tests/test_transport_preconditioner_dispatch.py tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/active_dense.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_solve_policy.py tests/test_transport_preconditioner_dispatch.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_transport_solve_policy.py tests/test_transport_preconditioner_dispatch.py tests/test_transport_active_dense_setup.py tests/test_domain_package_import_contracts.py -q --tb=short` passed with `54 passed`.
- `python -m pytest tests/test_transport_*.py -q --tb=short` passed with
  `273 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Next best steps:

1. Continue Batch B by merging transport solver shards with direct behavioral
   coverage: `dense_lu.py`, `dense_batch.py`, `host_gmres.py`, `loop.py`,
   `iteration_stats.py`, `residual_quality.py`, and `sparse_direct_solve.py`
   should move into `transport_matrix/solve.py` or one active-system owner.
2. Then merge `postsolve_diagnostics.py` into `finalize.py` and
   `streaming_outputs.py` into `outputs/transport.py` to reduce output
   ownership complexity.
3. After transport reaches `<=18` files, return to Batch A's
   `profile_response/solve.py <=5,500` line gate and then Batch C
   preconditioner-family compression.

## 2026-06-26 Batch B Transport Solve-Helper Owner Merge

Steps taken:

1. Merged `sfincs_jax/problems/transport_matrix/dense_lu.py` into
   `sfincs_jax/problems/transport_matrix/solve.py`. Dense-LU solver and
   preconditioner construction now live with the transport solve branches that
   call them.
2. Merged `sfincs_jax/problems/transport_matrix/host_gmres.py` into
   `transport_matrix/solve.py`. The explicit host SciPy GMRES fallback/rescue
   helper now lives with solve orchestration while keeping the same residual
   acceptance policy from `transport_matrix/policies.py`.
3. Merged `sfincs_jax/problems/transport_matrix/iteration_stats.py` into
   `transport_matrix/solve.py`. Optional host-side Krylov history diagnostics
   now live with the solve owner and remain non-fatal.
4. Merged `sfincs_jax/problems/transport_matrix/residual_quality.py` into
   `transport_matrix/policies.py`. Transport worker residual-abort threshold
   parsing and diagnostic formatting now live with policy, and both sequential
   and parallel transport paths import that owner.
5. Rewired focused tests, import-contract tests, source-map docs, API docs,
   testing docs, release notes, `profile_response/solve.py`, `loop.py`, and
   `parallel/runtime.py` to the new owners.
6. Deleted the four absorbed modules.

Results:

- Package Python files decreased from `202` to `198`.
- Package source lines decreased from `165,517` to `165,472`.
- `problems/transport_matrix` files including `parallel/` decreased from `25`
  to `21`.
- `transport_matrix/solve.py` is now `2,024` lines and owns transport Krylov
  dispatch plus dense-LU, host-GMRES, and KSP iteration diagnostics.
- `transport_matrix/policies.py` is now `2,200` lines and owns residual gates
  plus transport runtime/solve/retry/active/dense/preconditioner policy.
- The Batch B transport file-count gate remains open but is close: `21` files
  must decrease to `<=18`.

Validation:

- `python -m py_compile sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/loop.py sfincs_jax/problems/transport_matrix/parallel/runtime.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_dense_lu.py tests/test_transport_host_gmres.py tests/test_transport_iteration_stats.py tests/test_transport_residual_quality.py tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/transport_matrix/loop.py sfincs_jax/problems/transport_matrix/parallel/runtime.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_dense_lu.py tests/test_transport_host_gmres.py tests/test_transport_iteration_stats.py tests/test_transport_residual_quality.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_transport_dense_lu.py tests/test_transport_host_gmres.py tests/test_transport_iteration_stats.py tests/test_transport_residual_quality.py tests/test_transport_loop_support.py tests/test_transport_parallel_runtime.py tests/test_domain_package_import_contracts.py -q --tb=short` passed with `44 passed`.
- `python -m pytest tests/test_transport_*.py -q --tb=short` passed with
  `273 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Next best steps:

1. Finish the Batch B transport file-count gate by merging at least three of
   the remaining transport shards: `dense_batch.py`, `loop.py`,
   `sparse_direct_solve.py`, `postsolve_diagnostics.py`, `streaming_outputs.py`,
   `active_dense.py`, `active_factor.py`, or `parallel/policy.py`.
2. Prefer a coherent next owner merge over one-helper churn:
   `postsolve_diagnostics.py` into `finalize.py` plus
   `streaming_outputs.py` into `outputs/transport.py`, or
   `dense_batch.py` and `loop.py` into `transport_matrix/solve.py`.
3. Once `transport_matrix` reaches `<=18` files, return to Batch A's
   `profile_response/solve.py <=5,500` line gate.

## 2026-06-26 Batch B Transport Solve Owner Gate Merge

Steps taken:

1. Merged `sfincs_jax/problems/transport_matrix/dense_batch.py` into
   `sfincs_jax/problems/transport_matrix/solve.py`. All-RHS dense transport
   matrix assembly, active-DOF projection, streamed diagnostic collection,
   residual bookkeeping, and per-`whichRHS` dense progress emission now live
   with the solve owner.
2. Merged `sfincs_jax/problems/transport_matrix/loop.py` into
   `transport_matrix/solve.py`. Full/reduced matvec caching, bounded recycle
   bases, stored-state recycle seeding, recycled initial guesses, residual
   gates, and ETA progress bookkeeping now live with solve orchestration.
3. Merged `sfincs_jax/problems/transport_matrix/sparse_direct_solve.py` into
   `transport_matrix/solve.py`. Sparse-pattern admission/caching, direct
   active FP operator factor reuse, explicit sparse helper materialization,
   fallback sparse-ILU setup, host iterative refinement, float32 polish, and
   float64 retry now live with the solve branches that invoke them.
4. Rewired `profile_response/solve.py`, focused dense-batch/loop/sparse-direct
   tests, import-contract tests, API docs, source map, testing docs, and release
   notes to the `transport_matrix.solve` owner.
5. Deleted the three absorbed modules.

Results:

- Package Python files decreased from `198` to `195`.
- Package source lines decreased from `165,472` to `165,398`.
- `problems/transport_matrix` files including `parallel/` decreased from `21`
  to `18`, meeting the Batch B review-ready file-count gate.
- `transport_matrix/solve.py` is now the owner for transport Krylov dispatch,
  dense-LU fallback, dense-batch fallback, host GMRES fallback, iteration
  diagnostics, loop-local recycle/progress state, and sparse-direct rescue.

Validation:

- `python -m py_compile sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_dense_batch.py tests/test_transport_loop_support.py tests/test_transport_sparse_direct_solve.py tests/test_domain_package_import_contracts.py` passed.
- `python -m ruff check sfincs_jax/problems/transport_matrix/solve.py sfincs_jax/problems/transport_matrix/policies.py sfincs_jax/problems/profile_response/solve.py tests/test_transport_dense_batch.py tests/test_transport_loop_support.py tests/test_transport_sparse_direct_solve.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_transport_dense_batch.py tests/test_transport_loop_support.py tests/test_transport_sparse_direct_solve.py tests/test_transport_dense_lu.py tests/test_transport_host_gmres.py tests/test_transport_iteration_stats.py tests/test_transport_residual_quality.py tests/test_domain_package_import_contracts.py -q --tb=short` passed with `36 passed`.
- `python -m pytest tests/test_transport_*.py -q --tb=short` passed with
  `273 passed`.
- `sphinx-build -W -b html docs docs/_build/html` passed.
- `git diff --check` passed.

Next best steps:

1. Batch B review-ready transport file-count gate is met. Remaining Batch B
   work is optional stretch cleanup: `postsolve_diagnostics.py` into
   `finalize.py`, `streaming_outputs.py` into `outputs/transport.py`, and
   possibly `parallel/policy.py` into `parallel/runtime.py` if review still
   favors fewer files.
2. Return to Batch A next: reduce `profile_response/solve.py` from `7,008`
   lines to `<=5,500` by moving progress replay, retry bookkeeping, final
   sparse-result normalization, and fallback summaries into existing sparse,
   dense, or diagnostic owners.
3. After the Batch A solve-line gate is met, proceed to Batch C solver and
   preconditioner-family compression.

## 2026-06-26 Batch A Solve-Sequencer Gate Compression

Steps taken:

1. Moved the full explicitly requested `sparse_pc_gmres` /
   `xblock_sparse_pc_gmres` branch out of
   `sfincs_jax/problems/profile_response/solve.py` into the existing
   `sfincs_jax/problems/profile_response/sparse/handoff.py` sparse owner.
   The solve entry point now delegates through
   `try_run_requested_sparse_pc_gmres_branch(...)`.
2. Moved the large default RHSMode-1 preconditioner selection policy block out
   of `profile_response/solve.py` into
   `sfincs_jax/problems/profile_response/policies.py` as
   `resolve_rhs1_default_preconditioner_selection(...)`.
3. Kept behavior stable by passing the existing driver scope into the moved
   owner functions and by updating only the values that the moved policy branch
   actually assigns.
4. Updated sparse and policy owner exports for the new owner-level entry
   points.
5. Updated `plan_final.md` with current metrics and the remaining handoff
   paydown blocker.

Results:

- `profile_response/solve.py` decreased from `6,985` to `5,358` lines,
  meeting the `<=5,500` Batch A review-ready solve-sequencer gate.
- Package Python files remain `195`.
- Package source lines increased from `165,398` to `165,653` because the
  mechanical context wrappers added overhead. This must be paid down before
  the final review gate.
- `profile_response/sparse/handoff.py` increased from `4,438` to `5,780`
  lines. The handoff review target is therefore open again by about `280`
  lines, and the next owner move should relocate direct-tail/generic setup
  support into existing sparse policy/direct owners.
- `profile_response/policies.py` increased to `7,425` lines and now owns the
  default RHSMode-1 preconditioner policy block.

Validation:

- `python -m py_compile sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/policies.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/policies.py` passed.
- `python -m pytest tests/test_profile_response_sparse_pc.py tests/test_profile_response_dense.py tests/test_rhs1_solver_policy.py tests/test_v3_sparse_pattern.py -q --tb=short` passed with `510 passed in 103.18s`.
- `python -m pytest tests/test_domain_package_import_contracts.py tests/test_profile_response_finalization.py tests/test_profile_response_linear_solve.py tests/test_profile_response_auto_solve.py -q --tb=short` passed with `29 passed in 1.68s`.
- `git diff --check` passed.

Progress:

- Lane 1 structural consolidation: about `94%`.
- Batch A profile-response collapse: about `86%`. The solve-sequencer gate is
  met; the remaining blocker is `sparse/handoff.py <=5,500` plus line-count
  paydown.
- Batch B transport/output/root cleanup: about `45%`; transport file-count is
  met, but output/root compression remains.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `30%`.

Next best steps:

1. Finish Batch A paydown by moving
   `SparsePCGenericBranchSetup*` and/or direct-tail setup support from
   `sparse/handoff.py` into existing sparse policy/direct owners without
   adding a new file.
2. Keep `profile_response/solve.py` below `5,500` while reducing package lines
   back below the pre-Batch-A `165,398` checkpoint before review.
3. Then proceed to Batch B output/root compression and Batch C
   preconditioner-family compression.

## 2026-06-26 Batch A Handoff Export Paydown

Steps taken:

1. Kept the large sparse-PC branch in the existing
   `sfincs_jax/problems/profile_response/sparse/handoff.py` owner, but
   replaced its duplicated 364-entry literal re-export list with owner-module
   export composition plus explicit local and diagnostics compatibility
   exports.
2. Verified the new `__all__` has the exact same exported symbol set as the
   pre-change `HEAD` handoff module: `364` old symbols, `364` new symbols,
   no missing names, and no extra names.
3. Added a documented file-local `F401,F811` ruff waiver for the handoff
   compatibility facade. This is limited to the re-export and driver-scope
   shadowing behavior in that file and is tracked in `plan_final.md` for
   Batch E review.
4. Refreshed `plan_final.md` so Batch A now records both review-ready gates:
   `profile_response/solve.py <=5,500` and
   `profile_response/sparse/handoff.py <=5,500`.

Results:

- `profile_response/sparse/handoff.py` decreased from `5,780` lines after the
  previous solve-sequencer move to `5,498` lines, meeting the Batch A handoff
  review gate.
- `profile_response/solve.py` remains `5,358` lines, still under the Batch A
  solve-sequencer gate.
- Package Python files remain `195`.
- Package source lines decreased from `165,653` to `165,371`, back below the
  pre-Batch-A `165,398` checkpoint.
- No top-level `rhs1_*` or `transport_*` modules exist.

Validation:

- Exact handoff export-set comparison against `HEAD` passed with no missing or
  extra symbols.
- `python -m py_compile sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/policies.py` passed.
- `python -m ruff check sfincs_jax/problems/profile_response/sparse/handoff.py sfincs_jax/problems/profile_response/solve.py sfincs_jax/problems/profile_response/policies.py` passed.
- `python -m pytest tests/test_profile_response_sparse_pc.py tests/test_profile_response_dense.py tests/test_rhs1_solver_policy.py tests/test_v3_sparse_pattern.py -q --tb=short` passed with `510 passed in 103.81s`.
- `python -m pytest tests/test_domain_package_import_contracts.py tests/test_profile_response_finalization.py tests/test_profile_response_linear_solve.py tests/test_profile_response_auto_solve.py -q --tb=short` passed with `29 passed in 1.98s`.

Progress:

- Lane 1 structural consolidation: about `95%`.
- Batch A profile-response collapse: about `92%`. The formal review-ready
  size gates are now met; only stretch cleanup remains unless new failures
  appear.
- Batch B transport/output/root cleanup: about `45%`; next required work is
  output/root compression, especially `io.py`.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `30%`.

Next best steps:

1. Move to Batch B instead of continuing handoff churn: reduce `io.py` from
   `4,264` lines to `<=1,200` by moving concrete output implementation into
   `sfincs_jax/outputs`.
2. Merge only obvious transport/output relay files where the owner is clearer;
   do not add new helper shards.
3. After Batch B, execute Batch C solver/preconditioner-family compression,
   targeting `solvers/preconditioners <=35` files and removal of the remaining
   historical `symbolic_sparse/rhs1_fortran_reduced.py` filename.

## 2026-06-26 Batch B Output Writer Ownership Move

Steps taken:

1. Moved the concrete output writer implementation from `sfincs_jax/io.py` to
   the output-domain owner `sfincs_jax/outputs/writer.py`.
2. Recreated `sfincs_jax/io.py` as a small compatibility facade that re-exports
   public output readers/writers and delegates legacy private names to
   `outputs.writer` through module-level `__getattr__`.
3. Rewrote moved-module imports from root-relative local imports to parent
   package imports and output-sibling imports.
4. Added the moved writer API to `sfincs_jax.outputs.__all__` and refreshed the
   domain import-contract test.
5. Updated `plan_final.md` to mark the `io.py <=1,200` gate as met and to
   record the temporary file-count/source-line debt from adding
   `outputs/writer.py`.

Results:

- `sfincs_jax/io.py` decreased from `4,264` lines to `49` lines.
- `sfincs_jax/outputs/writer.py` now owns the concrete `4,264`-line writer.
- Package Python files increased from `195` to `196`; this must be paid back
  before the final review gate.
- Package source lines increased from `165,371` to `165,430`; this must be
  paid back below the previous `165,398` checkpoint before review.
- Public import parity is preserved:
  `sfincs_jax.io.write_sfincs_jax_output_h5` and
  `sfincs_jax.outputs.write_sfincs_jax_output_h5` resolve to the same function.
- Legacy private imports used by focused tests still resolve through the
  `io.py` facade.

Validation:

- `python -m py_compile sfincs_jax/io.py sfincs_jax/outputs/__init__.py sfincs_jax/outputs/writer.py` passed.
- `python -m ruff check sfincs_jax/io.py sfincs_jax/outputs/__init__.py sfincs_jax/outputs/writer.py tests/test_domain_package_import_contracts.py` passed.
- `python -m pytest tests/test_domain_package_import_contracts.py tests/test_output_h5_scheme5_parity.py tests/test_input_compat.py tests/test_transport_matrix_rhsmode3_parity.py tests/test_rhsmode1_current_closure.py -q --tb=short` passed with `42 passed`.
- `python -m pytest tests/test_api_contracts.py tests/test_cli_solve_mode.py tests/test_cli_validation_io_fast_coverage.py tests/test_io_output_policy_coverage.py tests/test_output_formats.py tests/test_solver_trace_output_formats.py tests/test_transport_output_schema.py tests/test_transport_streaming_outputs.py tests/test_write_output_return_results.py -q --tb=short` passed with `101 passed`.
- Explicit legacy-private import probe for `_OUTPUT_GEOM_CACHE`,
  `_select_rhsmode1_linear_solve_method`, and `_apply_export_f_maps` passed.

Progress:

- Lane 1 structural consolidation: about `95%`.
- Batch A profile-response collapse: about `92%`.
- Batch B transport/output/root cleanup: about `60%`. The major `io.py` facade
  gate is met; the remaining blocker is payback of the added writer owner by
  deleting/merging relay modules.
- Batch C solver/preconditioner family consolidation: about `25%`.
- Batch D public API/docs/tests/review gate: about `35%`.

Next best steps:

1. Continue Batch B payback by merging obvious transport/output relay modules
   whose owners are already clear, targeting package files back to `<=195`
   immediately and ultimately `<=190`.
2. Decide whether `data_fetch.py`, `postprocess_upstream.py`, and `scans.py`
   should stay at root as public workflows or move under domain packages.
3. Then execute Batch C solver/preconditioner-family compression.
