# SFINCS_JAX Active Execution Plan

Last updated: 2026-06-21 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review PR: #8, `refactor/v3-driver-architecture`, is the single review PR for
the architecture work. Keep it stable until the plan below reaches a coherent
review boundary; do not open more PRs or merge partial refactor branches before
the plan is complete.

## One-Sentence Plan

Build a smaller, domain-organized `sfincs_jax` that keeps SFINCS Fortran v3
parity where the models overlap, exposes explicit differentiable Python solve
lanes, keeps fast low-memory CPU/GPU production defaults for CLI/non-autodiff
users, and supports that contract with focused tests, honest README/docs, and
regenerated benchmarks only from complete checked reports.

## Current Snapshot

What is already done:

- The public release story is documented: the 39-case suite is parity-clean on
  CPU and GPU, with no `jax_error`, no `max_attempts`, and no strict mismatches
  in the release-facing artifacts.
- Large public data files were moved out of git and into fetchable release
  assets; repository-size checks are part of the workflow.
- README, docs, release checklist, validation matrix, and research-lane pages
  distinguish public claims from deferred production-resolution/QI/multi-GPU
  research lanes.
- The architecture branch introduced domain packages and many directly tested
  helper modules without changing validated numerical behavior.
- The follow-up branch has already split several RHSMode=1 policy and coarse
  basis layers out of the monolith:
  `rhs1_coarse_policy.py` and `rhs1_coarse_basis.py` now live under
  `sfincs_jax.solvers.preconditioners.schur`.
- The first complete RHSMode=1 preconditioner implementation family has moved:
  `sfincs_jax.solvers.preconditioners.schur.rhs1_full_csr` now owns the
  structured full-CSR preconditioner result type plus Jacobi, diagonal
  tail-Schur, zeta-line Schur, pitch-line Schur, and radial-pitch Schur
  builders.

Current source-size pressure points:

- `sfincs_jax/v3_driver.py`: about 14.4k lines.
- `sfincs_jax/rhs1_full_assembly.py`: about 10.6k lines.
- `sfincs_jax/io.py`: about 5.8k lines.
- `sfincs_jax/problems/profile_response/sparse/xblock.py`: about 4.5k lines.
- `sfincs_jax/rhs1_qi_device_preconditioner.py`: about 4.4k lines.
- Total Python source is about 159k lines.

Current docs-size pressure points:

- `docs/performance_techniques.rst`, `docs/testing.rst`, `docs/usage.rst`,
  `docs/parallelism.rst`, `docs/source_map.rst`, and `README.md` are long but
  currently useful as claim-boundary documents. Do not delete technical detail
  until the corresponding source map or API docs have a better home for it.

Latest validation evidence for this follow-up branch:

- Docs workflow passed on the latest pushed commits.
- Focused RHSMode=1 full-assembly tests passed after the recent extractions.
- `tests/test_v3_sparse_pattern.py` passed after the recent extractions.
- `tests/test_rhs1_full_csr_schur_preconditioners.py` directly validates the
  extracted full-CSR Schur family on exact small sparse systems.
- Targeted `ruff`, `git diff --check`, repository-size audit, and Sphinx build
  passed after the recent extractions.

## North-Star Goals

1. User simplicity
   A normal user should run `sfincs_jax input.namelist --wout-path wout.nc` and
   get a residual-clean result without knowing solver internals.

2. Fortran-v3 parity
   Where SFINCS Fortran v3 and `sfincs_jax` solve the same model, parity and
   output-key coverage remain the release trust gate.

3. Differentiable research API
   Python users can explicitly request differentiable JAX-native lanes for
   sensitivity analysis, inverse design, uncertainty quantification, and
   stellarator optimization.

4. Performance and memory
   CLI/non-autodiff paths may use faster host-native sparse factors, caches, or
   production shortcuts. Differentiable paths stay JAX-native. Both CPU and GPU
   paths must report enough progress, residual, runtime, and memory metadata to
   debug slow or stalled runs.

5. Maintainable code
   The code should move from historical monoliths into a small number of domain
   packages with clear names, typed contracts, docstrings, and direct tests. A
   smaller line count is good, but a simpler ownership model is more important.

6. Honest public artifacts
   README/docs plots and tables are regenerated only from complete checked
   CPU/GPU/Fortran reports. Reduced-grid plots stay labeled as reduced-grid
   diagnostics, not production-resolution claims.

## Non-Negotiable Constraints

- Preserve numerical behavior and output schemas at every refactor step.
- Keep legacy imports and private compatibility aliases when tests, docs, or
  downstream debug scripts still rely on them.
- Do not replace one monolith with many vague thin files. New files must own a
  physics, numerics, I/O, validation, or workflow concept.
- Do not add large generated outputs, profiler dumps, equilibrium files, or raw
  benchmark trees to git.
- Do not grow coverage with smoke-only scaffolds. Tests must guard physics,
  numerical identities, solver-policy behavior, I/O contracts, API behavior, or
  regression artifacts.
- Keep host-only solver shortcuts outside differentiable code paths.
- Avoid CI polling as a work loop. Inspect completed failures or final review
  gates, then keep refactoring.

## Open Lanes And Priority

### P0. Stabilize The Review Boundary

Status: effectively complete, maintain only.

Actions:

- Keep PR #8 as the only review PR for architecture work.
- Do not merge PR #8 until this plan reaches a coherent review boundary.
- Do not start a second PR unless the current PR is merged or explicitly
  abandoned.

Acceptance:

- Branch remains clean and pushable.
- Required checks pass when inspected.
- No generated or large artifacts are committed.

### P1. Finish The RHSMode=1 Full-Assembly Split

Status: about 65%.

This is the next highest-value refactor because it attacks the largest active
solver monolith after `v3_driver.py` and unlocks better tests for the production
RHSMode=1 solver lanes.

Actions, in order:

1. Complete the next cohesive RHSMode=1 preconditioner family move. The first
   completed family is the full-CSR Schur/Jacobi block family; the next best
   candidates are active native-stack/sparse-coarse apply builders, direct-tail
   factor helpers, or active sparse-coarse residual admission.
2. If the move is blocked by circular imports or result contracts, first extract
   the shared typed contracts into a neutral solver/preconditioner module.
3. Move the family into `sfincs_jax/solvers/preconditioners/schur/` or a more
   precise subpackage name if the numerical structure is not Schur-like.
4. Keep `rhs1_full_assembly.py` importing historical private names only as
   compatibility aliases.
5. Add direct unit/regression tests for the moved implementation and keep the
   existing full-assembly tests green.

Acceptance:

- The extracted module owns a complete behavior slice: build/setup, metadata,
  residual admission, and apply/replay behavior where applicable.
- `rhs1_full_assembly.py` gets smaller for a real reason, not only by adding
  wrappers.
- Focused tests pass:
  `tests/test_rhs1_full_assembly.py`,
  `tests/test_v3_sparse_pattern.py`, and the new module tests.
- `docs/source_map.rst`, `docs/testing.rst` if needed, and this plan mention
  the new ownership boundary.

### P2. Reduce `v3_driver.py` To Orchestration

Status: about 88%.

Actions:

1. Keep moving only cohesive stage boundaries out of the driver:
   result contracts, solver dispatch, policy selection, preconditioner setup,
   residual correction, progress reporting, and output handoff.
2. Leave driver-local mutable state in the driver until it has a typed owner.
3. Preserve monkeypatch/debug seams with compatibility aliases only where tests
   or downstream workflows need them.

Acceptance:

- `v3_driver.py` primarily coordinates input -> problem -> solver -> output.
- Any remaining large block has an explicit reason: callback wiring, cache
  mutation, replay state, or legacy compatibility.
- New tests target the extracted module directly instead of increasing
  driver-wrapper tests.

### P3. Split `io.py` After RHSMode=1 Boundaries Stabilize

Status: not started on this follow-up branch.

Actions:

1. Split output schema construction from file-format writers.
2. Move HDF5, NetCDF, and NPZ serialization behind one schema contract.
3. Keep CLI output behavior unchanged.
4. Keep plotting hooks and diagnostic-panel generation outside solver internals.

Acceptance:

- `io.py` becomes orchestration plus compatibility imports.
- Output-schema tests prove shared fields are identical across HDF5/NetCDF/NPZ.
- Existing CLI and Python output tests pass.

### P4. Simplify Domain Packages Without File Explosion

Status: about 70%.

Actions:

- Use these homes for new implementation:
  `input/`, `physics/`, `discretization/`, `operators/`,
  `problems/profile_response/`, `problems/transport_matrix/`, `solvers/`,
  `parallel/`, `workflows/`, `validation/`, `benchmarks/`, and `compat/`.
- Convert implementation-heavy top-level `rhs1_*` and `transport_*` modules
  gradually into domain modules.
- Keep top-level files as compatibility shims only when public imports or tests
  require that.

Acceptance:

- A developer can infer the module location from the physics or numerical
  responsibility.
- New files have descriptive names and docstrings.
- Source-map docs explain which modules are compatibility shims and which own
  implementation.

### P5. Preserve Differentiability While Keeping Fast CLI Paths

Status: about 78%.

Actions:

1. Make every public solve entry point explicit about differentiable versus
   non-differentiable behavior.
2. Keep adaptive `auto` decisions auditable: selected branch, rejected branches,
   residual margins, backend, memory estimate, and warnings near branch
   boundaries.
3. Add gradient/implicit-differentiation checks when public solver boundaries
   or branch certificates change.
4. Use host sparse factors only when `differentiable=False` or when the public
   API makes the non-differentiable path explicit.

Acceptance:

- Differentiable Python workflows remain JAX-transformable.
- CLI defaults stay fast and memory-aware.
- No autodiff user is silently routed through a host-only fallback.

### P6. Raise Meaningful Coverage Without Slow CI

Status: release branch around 74% package coverage; long-term target is 95%
meaningful coverage.

Actions:

1. Test extracted modules directly at high coverage as they leave the monoliths.
2. Add physics gates for invariants already used by the code:
   conservation/null modes, finite-difference order, symmetry limits, residual
   gates, output normalizations, and known bootstrap/transport trends.
3. Keep full production solves in release/manual benchmark tiers, not normal CI.
4. Keep normal CI in the intended 5-10 minute range where possible.

Acceptance:

- Every extraction has focused tests.
- Coverage increases because monolith responsibilities shrink and become
  testable, not because slow smoke solves are duplicated.
- Testing docs state which gates are CI, release, manual GPU, or research tier.

### P7. Refresh Benchmarks And Public Figures Only After Behavior Changes

Status: deferred for behavior-preserving refactors.

Actions:

1. Do not regenerate README runtime/memory plots for pure refactors.
2. After solver behavior changes, rerun CPU suite locally and GPU suite on
   `ssh office` if needed.
3. Compare against SFINCS Fortran v3 only from complete reports.
4. Regenerate README/docs plots and tables from canonical JSON, not ad hoc
   partial runs.

Acceptance:

- Public plots and parity tables trace to complete checked reports.
- Runtime/memory regressions are documented or fixed before promotion.
- Reduced-grid QA/QH/QI plots stay labeled as reduced-grid until production
  gates pass.

## Deferred Technical Research Lanes

These are important, but they are not blockers for the current refactor PR.
Keep them documented and fail-closed rather than mixing them into the
maintainability critical path.

- True differentiable device-QI: current best artifacts are useful but still
  miss production write/residual gates.
- Production-resolution QI ladders: bounded CPU/GPU/Fortran rungs exist, but
  full production floors remain open.
- Single-case multi-GPU strong scaling: correct infrastructure exists, but not
  a public performance claim.
- Lower-memory native sparse-factor replacement for the largest geometry-rich
  RHSMode=2/3 and full-grid QA/QH RHSMode=1 cases: opt-in infrastructure exists,
  but current defaults should continue to prioritize residual-clean parity.
- Full-grid QA/QH RHSMode=1 production convergence: tracked as a validation and
  production-solver lane, not a refactor blocker.

## Finite Execution Sequence From Here

1. Keep PR #8 stable and stop broadening the review surface.
2. Complete P1 by moving one real RHSMode=1 preconditioner implementation
   family into the solver/preconditioner domain tree.
3. Run focused tests and hygiene after that tranche.
4. Update source-map/testing docs for the new ownership boundary.
5. Commit and push the tranche.
6. Repeat P1 only if the next coherent family is obvious and low-risk;
   otherwise move to P3 and split `io.py`.
7. After RHSMode=1 and I/O boundaries are stable, run broader local shards and
   assess whether PR #8 can be marked ready for review.
8. Only after the refactor branch is review-ready, decide whether a separate
   performance/production benchmark campaign is warranted.

## Completion Criteria

This plan is complete when:

- PR #8 has one coherent review surface and no generated artifacts.
- `v3_driver.py`, `rhs1_full_assembly.py`, and `io.py` no longer hide the main
  solver, preconditioner, or output contracts in untested monolithic bodies.
- Public CLI and Python APIs still work with the same output schemas.
- Differentiable and non-differentiable lanes are explicit and tested.
- SFINCS Fortran v3 parity gates remain unchanged or are regenerated from
  complete checked reports.
- README/docs accurately state what is production quality, what is reduced-grid
  evidence, and what remains deferred research.
- Normal CI remains practical, and release/manual tiers cover expensive CPU/GPU
  and Fortran comparison evidence.
