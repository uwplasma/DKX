# SFINCS_JAX Final Refactor Plan

Last updated: 2026-06-21 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review PR: #8, `refactor/v3-driver-architecture`, remains the single draft
review PR for this architecture/refactor campaign. Do not open more PRs and do
not merge this PR until the review boundary below is complete.

## One-Sentence Plan

Make `sfincs_jax` a smaller, domain-organized, research-grade neoclassical
transport code that preserves SFINCS Fortran v3 parity where the models overlap,
keeps CPU/GPU defaults fast and memory-aware, exposes explicit differentiable
Python solve lanes, and backs every public claim with focused tests,
documentation, README figures, and complete benchmark reports.

## Current State

### What Is Done

- Public release artifacts document a CPU/GPU-clean audited 39-case suite with
  no `jax_error`, no `max_attempts`, and no strict parity mismatches in the
  release-facing scope.
- README and docs distinguish production-quality claims from reduced-grid,
  fail-closed, and deferred research evidence.
- Large equilibrium fixtures were moved out of git into release-hosted data with
  checksum/fetch tests.
- The refactor branch introduced domain packages for input, physics,
  discretization, operators, problems, solvers, parallel workflows, validation,
  benchmarks, and compatibility.
- Several RHSMode=1 and transport policy/coarse/preconditioner slices already
  have focused modules and tests.
- `sfincs_jax.solvers.preconditioners.schur.rhs1_full_csr` now owns the
  structured full-CSR RHSMode=1 Schur/Jacobi preconditioner family, with direct
  tests and source-map documentation.
- `sfincs_jax.solvers.preconditioners.symbolic_sparse.rhs1_fortran_reduced`
  now owns the Fortran-v3-style RHSMode=1 reduced active sparse-factor family,
  including reduced-support construction, support-mode preflight, symbolic-plan
  permutation, equilibration, LU/ILU setup, and memory admission.
- `sfincs_jax.solvers.preconditioners.xblock.low_l_schur` now owns the
  low-pitch x-block Schur family for exact RHSMode=1 full-CSR systems,
  including native `x_ell` kinetic factors, native `x_ell` plus tail Schur,
  sparse low-`ell` x-block Schur, physics coarse residual correction, and the
  shared low-`ell` x-block index helper.
- `sfincs_jax.outputs.formats` now owns flat HDF5/NetCDF/NPZ output
  readers/writers, output suffix dispatch, SFINCS Fortran-compatible HDF5
  layout conversion, NetCDF-safe names, and solver-trace attachment.  `io.py`
  keeps compatibility aliases for existing imports.

### What The Latest Review Found

- The branch has committed the symbolic-sparse RHSMode=1 Fortran-reduced
  extraction (`b7a0bf2`) and the flat output-format split (`11abd75`). The
  current local tranche extracts the RHSMode=1 x-block low-`ell` Schur family
  and has passed focused full-assembly, sparse-pattern, import, and direct
  x-block tests locally.
- Current largest source files:
  - `sfincs_jax/v3_driver.py`: about 14.4k lines, 79 functions.
  - `sfincs_jax/rhs1_full_assembly.py`: about 9.2k lines after this tranche.
  - `sfincs_jax/io.py`: about 5.5k lines after the output-format split.
  - `sfincs_jax/problems/profile_response/sparse/xblock.py`: about 4.5k lines.
  - `sfincs_jax/rhs1_qi_device_preconditioner.py`: about 4.4k lines.
- The repository has about 286 Python source files and about 159k source lines.
  The file count is high enough that new modules must now consolidate domain
  ownership, not add more flat historical `rhs1_*` or `transport_*` helpers.
- Docs are extensive and mostly accurate, but `docs/source_map.rst`,
  `docs/testing.rst`, `docs/development_roadmap.rst`, and README must be updated
  whenever ownership or public claims change.

## North-Star Goals

1. User simplicity
   A normal user should provide one input file and optional geometry path:
   `sfincs_jax input.namelist --wout-path wout.nc --out sfincsOutput.h5`. The
   solver should choose a safe default without requiring environment variables.

2. SFINCS Fortran v3 parity
   Where the same model is solved, parity with SFINCS Fortran v3 remains the
   release trust gate. Comparisons must include output quantities, residual
   metadata, runtime, memory, and solver path.

3. Differentiable research API
   Python users must be able to choose JAX-native differentiable lanes for
   sensitivity analysis, inverse design, uncertainty quantification, and
   stellarator optimization. Host-only shortcuts must not be hidden inside
   differentiable paths.

4. Fast CPU/GPU production defaults
   CLI and `differentiable=False` Python calls may use faster non-autodiff
   sparse factors, caches, and host/device-specific policies, but accepted
   results must pass true-residual gates and record solver metadata.

5. Small, understandable code structure
   Reduce monolith responsibility and avoid file explosion. New modules should
   be named by domain concept: physics, discretization, operators, profile
   response, transport matrix, solver, preconditioner, output, validation, or
   workflow.

6. Research-grade testing and documentation
   Coverage should increase through focused tests on extracted modules,
   physics/numerical gates, and regression artifacts, not by adding slow smoke
   solves. Docs and README must state what is production quality, what is
   reduced-grid evidence, and what remains deferred research.

## Non-Negotiable Constraints

- Preserve numerical behavior and output schemas during refactors.
- Keep compatibility aliases only where tests, docs, public imports, or known
  downstream debug scripts need them.
- Do not replace monoliths with many vague thin files.
- Do not commit generated profiler dumps, large raw benchmark trees, or large
  equilibrium artifacts.
- Keep normal CI practical; expensive CPU/GPU/Fortran sweeps belong in manual
  release or benchmark tiers.
- Use complete checked benchmark JSON before changing public runtime/memory
  figures or parity tables.

## Priority Plan

### P0. Branch Hygiene And Tranche Gate

Status: active guardrail for every refactor tranche.

Actions:

1. Keep each tranche small enough to validate and review, but large enough to
   move a complete domain responsibility.
2. Run focused tests for the touched domain plus `ruff`, `py_compile`, Sphinx,
   repo-size, and `git diff --check`.
3. Commit and push only clean tranches; do not accumulate generated artifacts or
   temporary simulation outputs.

Acceptance:

- `git status` has no accidental untracked implementation files after each
  commit.
- New module tests and legacy compatibility tests pass for the touched domain.
- `ruff`, `py_compile`, `git diff --check`, repo-size audit, and Sphinx pass.

### P1. Finish RHSMode=1 Full-Assembly Ownership Split

Status: about 88% after the full-CSR Schur, symbolic-sparse Fortran-reduced,
and x-block low-`ell` Schur extractions.  Continue only for cohesive
implementation families, not wrapper churn.

Actions:

1. Move one complete preconditioner/solver behavior slice at a time out of
   `rhs1_full_assembly.py`.
2. Valid homes are:
   - `solvers/preconditioners/schur/` for Schur and moment/coarse closures.
   - `solvers/preconditioners/symbolic_sparse/` for host sparse factors,
     symbolic ordering, reduced `Pmat`, and factor admission.
   - `solvers/preconditioners/xblock/` for x-block and radial/pitch block
     structure.
   - `problems/profile_response/sparse/` for RHSMode=1 problem orchestration
     that is not reusable solver machinery.
3. Keep `rhs1_full_assembly.py` as orchestration plus compatibility aliases.
4. Stop the split when the remaining code is mostly assembly orchestration,
   compatibility, or stateful integration with `v3_driver.py`; do not keep
   extracting low-value wrappers.

Acceptance:

- `rhs1_full_assembly.py` is materially smaller and easier to scan.
- Every moved implementation has direct tests.
- Existing full-assembly tests remain green.
- Source-map docs identify the new owner of every moved family.

### P2. Reduce `v3_driver.py` To Orchestration

Status: about 88%; still too large, but less urgent than completing the current
RHSMode=1 tranche cleanly.

Actions:

1. Extract only cohesive stage boundaries: solver dispatch, preconditioner
   setup, residual correction, progress reporting, output handoff, and result
   contracts.
2. Do not extract driver-local mutable state until it has a typed owner.
3. Preserve debug seams only when tests or downstream workflows actually use
   them.
4. Prefer deleting redundant compatibility wrappers over adding more files.

Acceptance:

- `v3_driver.py` primarily coordinates input -> problem setup -> solver -> output.
- New tests target extracted modules directly.
- No behavior or output-schema regressions.

### P3. Split I/O By Schema And Format

Status: started.  Flat format readers/writers are extracted to
`sfincs_jax.outputs.formats`; schema construction and high-level output
orchestration still live in `sfincs_jax/io.py`.

Actions:

1. Introduce one output schema contract for solved fields, diagnostics, solver
   metadata, timing, memory, and provenance.
2. Keep HDF5, NetCDF, and NPZ writing behind the format-specific owner already
   extracted in `sfincs_jax.outputs.formats`.
3. Keep CLI output behavior, `--plot`, and current dataset names unchanged.
4. Keep plotting and diagnostics outside solver internals.

Acceptance:

- `io.py` becomes a small orchestration/compatibility surface.
- HDF5/NetCDF/NPZ tests prove shared fields are identical.
- CLI and Python output tests pass.

### P4. Consolidate The Package Layout

Status: about 70%; skeleton exists, but too many flat compatibility modules
remain.

Actions:

1. Keep these as the preferred domain homes:
   `input/`, `physics/`, `discretization/`, `operators/`,
   `problems/profile_response/`, `problems/transport_matrix/`, `solvers/`,
   `parallel/`, `workflows/`, `validation/`, `benchmarks/`, and `compat/`.
2. Convert top-level `rhs1_*` and `transport_*` implementation files gradually
   into those homes.
3. Keep top-level files as compatibility shims only when public imports,
   existing tests, or documented workflows require them.
4. Remove duplicate historical helpers once their new owners are tested.

Acceptance:

- A developer can infer module location from physics or numerical responsibility.
- Source files have descriptive names and short module docstrings.
- The number of active implementation files stops growing; compatibility shims
  are marked as such in docs.

### P5. Preserve Differentiability And Fast Non-Autodiff Lanes

Status: about 80%.

Actions:

1. Make public solve entries explicit about `differentiable=True` versus
   `differentiable=False`.
2. Keep host sparse factors and SciPy/SuperLU-style setup out of differentiable
   lanes unless the API explicitly says the branch is non-differentiable.
3. Add or maintain branch certificates for `auto`: selected method, rejected
   methods, residual margins, backend, memory estimates, and warnings near
   branch boundaries.
4. Use implicit differentiation/custom-linear-solve style contracts for
   differentiable linear solves rather than differentiating through every Krylov
   or setup iteration.

Acceptance:

- Differentiable workflows remain JAX-transformable on documented reduced
  fixtures.
- CLI defaults stay fast and residual-clean.
- No user is silently routed through a host-only fallback when gradients are
  requested.

### P6. Raise Meaningful Coverage

Status: long-term target is 95% meaningful coverage; current work should improve
coverage through extraction, not slow full-solve tests.

Actions:

1. Every extracted module gets direct unit/regression tests.
2. Physics/numerical gates should cover conservation/null modes, finite-
   difference order, symmetry limits, residual gates, output normalizations,
   and known bootstrap/transport trends.
3. Synthetic sparse/operator tests should cover solver primitives without
   expensive full solves.
4. Keep normal CI near the intended practical runtime; use manual/release tiers
   for expensive CPU/GPU/Fortran comparisons.

Acceptance:

- Coverage rises because monolith responsibilities become testable.
- Testing docs classify tests as CI, release, manual GPU, or research tier.
- No coverage increase depends mainly on slow smoke-only full solves.

### P7. Documentation And README Final Pass

Status: mostly accurate, but must follow each ownership and claim change.

Actions:

1. Keep `docs/source_map.rst` current with the real source ownership.
2. Keep `docs/testing.rst` current with the validation tiers and coverage plan.
3. Keep README focused on install, quick CLI usage, plotting, current public
   runtime/memory figures, and honest scope notes.
4. Move deep research-lane details to docs, not the README.

Acceptance:

- README is user-friendly and not overloaded with internal refactor details.
- Docs are detailed enough for developers and reviewers to trace equations,
  algorithms, tests, and claims to source files.
- No docs claim production status for deferred research lanes.

### P8. Benchmarks, Parity, And Figures

Status: deferred for behavior-preserving refactors.

Actions:

1. Do not regenerate public runtime/memory figures for pure refactors.
2. After behavior-changing solver work, rerun complete CPU reports locally and
   GPU reports on `ssh office` when needed.
3. Regenerate README/docs plots and tables only from canonical complete JSON.
4. Keep reduced-grid QA/QH/QI figures labeled as reduced-grid until production
   gates pass.

Acceptance:

- Public plots trace to complete checked reports.
- Runtime/memory regressions are fixed or documented before promotion.
- Fortran v3 comparisons use matching model/resolution contracts.

### P9. Review-Ready Gate For PR #8

Status: not ready yet.

Actions:

1. Complete P0-P3 or explicitly document any remaining split as deferred.
2. Run focused local validation for all touched domains.
3. Build docs with warnings as errors.
4. Run repository-size checks.
5. Push all changes and inspect CI only after enough work has landed to make the
   wait useful.

Acceptance:

- PR has one coherent story: monolith responsibilities moved into domain
  packages, behavior preserved, tests/docs updated.
- Branch has no generated artifacts or accidental large files.
- Reviewers can understand module ownership from docs and source names.

## Deferred Technical Research Lanes

These remain important but are not blockers for the current refactor PR unless
the refactor directly changes their public claims.

- True differentiable device-QI at production tolerance.
- Production-resolution QI ladders.
- Single-case multi-GPU strong scaling as a public performance claim.
- Lower-memory native sparse-factor replacement for the largest geometry-rich
  RHSMode=2/3 and full-grid QA/QH RHSMode=1 cases.
- Full-grid QA/QH RHSMode=1 production convergence beyond reduced-grid
  documentation evidence.

Deferred means fail-closed and documented, not forgotten. Future work should
target genuinely stronger operator/coarse/factor architectures and complete
CPU/GPU/Fortran gates, not more smoother/restart tuning.

## Immediate Ordered Next Steps

1. Land the current `outputs.formats` tranche, which is the only active local
   split at this checkpoint.
2. Extract one more high-value `io.py` schema/diagnostics contract only if it
   reduces responsibility without adding file sprawl; otherwise stop P3 after
   the format split and document the remaining boundary.
3. Reassess whether P1 should stop after the x-block low-`ell` Schur extraction
   or extract one final cohesive family. Do not continue if the remaining work
   would only create wrapper churn or vague files.
4. Reduce `v3_driver.py` only at stage boundaries that already have stable
   extracted types: solver dispatch, progress reporting, output handoff, or
   result contracts.  Do not split driver-local mutable state into vague helper
   files.
5. Run the focused local validation matrix for touched domains, Sphinx with
   warnings as errors, repo-size checks, and then inspect CI once enough work has
   landed to make the wait useful.
6. Mark PR #8 ready for review only after the branch is clean, docs are current,
   public behavior is unchanged, and deferred technical research lanes are
   explicitly fail-closed rather than mixed into the refactor PR.

## Done Definition

This plan is complete when:

- PR #8 is a single coherent, reviewable refactor PR.
- `v3_driver.py`, `rhs1_full_assembly.py`, and `io.py` no longer hide major
  solver, preconditioner, or output contracts inside untested monolithic bodies.
- Public CLI and Python APIs still work with the same output schemas.
- Differentiable and non-differentiable solver lanes are explicit and tested.
- SFINCS Fortran v3 parity gates remain clean or are regenerated from complete
  checked reports after behavior changes.
- README/docs accurately describe install, usage, equations, solver lanes,
  testing, benchmarks, and deferred research scope.
- Normal CI remains practical, and release/manual tiers cover expensive CPU/GPU
  and Fortran comparison evidence.
