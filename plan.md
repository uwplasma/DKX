# SFINCS_JAX Refactor And Release-Readiness Plan

Last updated: 2026-06-21 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Intended review PR: #8, `refactor/v3-driver-architecture`

Important hygiene note: PR #8 currently points at
`refactor/v3-driver-architecture`, while the active work in this checkout is on
`refactor/rhs1-full-assembly-preconditioners`. Do not open another PR. Before
review, reconcile this mismatch by moving the active branch work onto the PR
branch or otherwise updating PR #8 to the active refactor head. Keep the PR in
draft/review-prep state until this plan's review boundary is complete.

## One-Sentence Plan

Make `sfincs_jax` a small, domain-organized, research-grade neoclassical
transport code that preserves SFINCS Fortran v3 parity where the models
overlap, gives users a simple input-file CLI with fast CPU/GPU defaults and low
memory use, exposes explicit differentiable Python solve lanes for optimization
and sensitivity work, keeps the implementation in a manageable number of
well-named files, and backs README/docs claims with focused tests and complete
benchmark artifacts.

## Current Results

- Latest local branch state is clean and pushed to
  `origin/refactor/rhs1-full-assembly-preconditioners`.
- Latest commits on this branch extracted:
  - RHSMode=1 full-CSR Schur preconditioners.
  - RHSMode=1 Fortran-reduced symbolic sparse factors.
  - RHSMode=1 low-`ell` x-block Schur preconditioners.
  - RHSMode=1 active-projected x-block / overlap-Schwarz preconditioners.
  - Flat HDF5/NetCDF/NPZ output-format helpers.
  - Nonlinear Phi1 Newton-Krylov profile-response solve stage.
- GitHub Docs CI is green for the latest pushed commits on the active branch.
- Current largest package files from the fresh source audit:
  - `sfincs_jax/v3_driver.py`: about 13.9k lines.
  - `sfincs_jax/rhs1_full_assembly.py`: about 8.6k lines.
  - `sfincs_jax/io.py`: about 5.5k lines.
  - `sfincs_jax/problems/profile_response/sparse/xblock.py`: about 4.5k lines.
  - `sfincs_jax/rhs1_qi_device_preconditioner.py`: about 4.4k lines.
- Package size is still high: about 288 Python files and 160k package lines.
  The next work must consolidate ownership and delete duplicate compatibility
  surfaces where possible, not create another layer of flat wrapper modules.
- README/docs are broadly accurate about current public claims: the audited
  release suite is CPU/GPU parity-clean, production-resolution QI and true
  device-QI remain deferred, and lower-memory native factor/preconditioner
  work remains opt-in research infrastructure.

## Goals

1. User simplicity
   A normal user should run `sfincs_jax input.namelist --wout-path wout.nc --out
   sfincsOutput.h5`, get progress and phase timing, and not need solver
   environment variables.

2. Fortran v3 parity
   Where the same physics model is solved, SFINCS Fortran v3 remains the
   trust anchor. Comparisons must include shared output quantities, residual
   metadata, runtime, memory, and solver path.

3. Differentiable research API
   Python users must be able to choose JAX-native differentiable solve lanes for
   sensitivity analysis, inverse design, uncertainty quantification, and
   stellarator optimization. Host-only shortcuts must be explicit and disabled
   when gradients are requested.

4. Fast CPU/GPU defaults
   CLI and `differentiable=False` Python calls may use faster host sparse
   factors, caches, and backend-specific policies. Accepted results must pass
   true-residual gates and write method/timing/memory metadata.

5. Small and manageable code structure
   The active implementation should move toward a small set of domain packages:
   `input`, `physics`, `discretization`, `operators`,
   `problems/profile_response`, `problems/transport_matrix`, `solvers`,
   `parallel`, `outputs`, `workflows`, `validation`, `benchmarks`, and `compat`.
   Avoid new historical `rhs1_*`, `transport_*`, or `v3_*` implementation files.

6. Research-grade tests and docs
   Coverage should rise through extracted-module tests, numerical identities,
   physics gates, and regression artifacts, not by adding slow smoke-only full
   solves to normal CI. Docs must say what is production quality, reduced-grid
   evidence, or deferred research.

## Technical Open Lanes

These are not all blockers for the refactor PR, but they must remain documented
and fail-closed.

1. Production-resolution QI and true device-QI
   Current status: bounded evidence exists and the hard seed gets below
   `3e-5`, but production tolerance and full write gates remain open.
   Refactor implication: preserve explicit differentiable/non-differentiable
   lane separation and branch certificates; do not bury host fallbacks inside
   gradient paths.

2. Full-grid QA/QH RHSMode=1 production convergence
   Current status: reduced-grid QA/QH bootstrap-current comparisons are useful
   and documented; full `25 x 39 x 60 x 7` production convergence remains a
   separate validation lane.
   Refactor implication: keep RHSMode=1 profile-response modules testable and
   solver metadata complete.

3. Lower-memory native sparse-factor replacement
   Current status: direct `Pmat`, symbolic ordering, nested-dissection,
   BLR/HSS-style, and residual-admission infrastructure exists, but it is not a
   promoted default for the hardest geometry-rich cases.
   Refactor implication: keep this as opt-in research infrastructure under
   `solvers/preconditioners`, not as driver-local policy sprawl.

4. Geometry-rich RHSMode=2/3 production preconditioner
   Current status: reduced geom2/geom11 gates pass, full production setup is
   still too slow for promotion.
   Refactor implication: extract reusable transport-matrix stage boundaries
   only if they simplify the problem package and preserve tests.

5. Single-case multi-GPU strong scaling
   Current status: available as experimental benchmarking, but the public
   production recommendation remains one GPU per independent case/scan point.
   Refactor implication: keep parallel runtime code separate from solver
   correctness and avoid making sharded single-case paths the default.

6. 95% meaningful coverage
   Current status: docs describe roughly mid-50% package coverage; the target
   requires splitting monoliths into testable modules and adding real numerical
   and physics gates.
   Refactor implication: every extraction needs direct tests; normal CI should
   remain practical.

## Refactor Open Lanes

1. Branch/PR reconciliation
   PR #8 is open on `refactor/v3-driver-architecture`, but active commits are on
   `refactor/rhs1-full-assembly-preconditioners`. This is the first review
   blocker.

2. `v3_driver.py` orchestration boundary
   Status: improved, but still the largest file. Phi1 Newton-Krylov is now in
   `problems/profile_response/phi1_newton.py`.
   Next extract only one cohesive stage boundary: result/output handoff,
   progress reporting, or transport parallel execution. Prefer deletion of
   redundant wrappers over adding a new file.

3. `rhs1_full_assembly.py` ownership split
   Status: most major preconditioner families have owners. Stop unless a large
   cohesive family remains. This file should now trend toward assembly
   orchestration, dispatch/admission, and compatibility.

4. `io.py` schema split
   Status: flat file formats are extracted to `outputs.formats`; output schema,
   solved-field assembly, diagnostics, timing, memory, and provenance contracts
   still live in `io.py`.
   Next split only after the next driver seam is stable.

5. Package consolidation
   Status: domain package skeleton exists, but top-level compatibility and
   historical modules remain numerous.
   Next move should reduce cognitive load: consolidate related files, mark
   compatibility shims, and delete redundant aliases when tests prove they are
   unused.

6. Documentation consistency
   Status: `docs/source_map.rst`, `docs/testing.rst`, `docs/research_lanes.rst`,
   and README already describe most current claims, but they must be updated
   after each ownership move.

## Prioritized Next Steps

### P0. Reconcile Branch And PR Metadata

Goal: one PR, one active refactor head, no hidden branch divergence.

Actions:

1. Decide whether to fast-forward/update `refactor/v3-driver-architecture` with
   the active branch commits or retarget PR #8 to
   `refactor/rhs1-full-assembly-preconditioners`.
2. Keep PR #8 as the single review PR.
3. Make the PR draft/review-prep until the review boundary below is complete.

Acceptance:

- `gh pr list` shows one open PR for the active refactor head.
- The plan and PR metadata agree on branch name and readiness state.

### P1. Finish One More `v3_driver.py` Boundary

Goal: reduce the driver to orchestration without creating wrapper clutter.

Actions:

1. Inspect the remaining driver-local stage seams.
2. Choose exactly one cohesive boundary, preferably one of:
   result/output handoff, progress/timing reporting, or transport parallel
   execution.
3. Move implementation into an existing domain package where possible.
4. Keep driver compatibility aliases only for public imports, monkeypatch seams,
   or active tests.

Acceptance:

- `v3_driver.py` loses real responsibility, not just a few wrapper lines.
- Extracted module has direct tests.
- Existing CLI/output/transport/profile-response tests still pass.

### P2. Stabilize RHSMode=1 Assembly Ownership

Goal: stop broad RHSMode=1 extraction churn unless a complete family remains.

Actions:

1. Audit remaining large sections in `rhs1_full_assembly.py`.
2. Move only a complete remaining preconditioner/solver family if it has a
   clear home under `solvers/preconditioners` or `problems/profile_response`.
3. Otherwise mark the file as current orchestration/compatibility owner and move
   to I/O/refactor consolidation.

Acceptance:

- No vague new `rhs1_*` module is added.
- Source map explains the owner of every major RHSMode=1 family.

### P3. Split Output Schema From `io.py`

Goal: make output behavior easy to test without touching solver internals.

Actions:

1. Define one output schema contract for solved fields, diagnostics, solver
   metadata, timing, memory, and provenance.
2. Keep HDF5/NetCDF/NPZ writer functions in `outputs.formats`.
3. Preserve CLI output suffix behavior and `--plot`.
4. Add direct tests proving `.h5`, `.nc`, and `.npz` share the same core fields.

Acceptance:

- `io.py` becomes a smaller orchestration/compatibility surface.
- Output tests cover schema and format equivalence directly.

### P4. Consolidate Layout And Delete Redundancy

Goal: simplify the codebase, not just move lines.

Actions:

1. Identify top-level historical files that are now compatibility-only.
2. Move implementation into domain packages or delete redundant wrappers.
3. Keep module names descriptive and pedagogical; avoid generic
   `transport_*`, `rhs1_*`, and `v3_*` names for new implementation.
4. Add short module docstrings explaining the equation, algorithm, or workflow
   responsibility.

Acceptance:

- File count does not grow without a real domain reason.
- Developers can infer where code lives from physics/numerical responsibility.

### P5. Preserve Differentiable And Fast Non-Differentiable Contracts

Goal: make adaptive solver choices auditable without pretending they are smooth.

Actions:

1. Keep public solve entries explicit about `differentiable=True` versus
   `differentiable=False`.
2. Record branch certificates for `auto`: selected method, rejected candidates,
   residual margins, backend, memory estimates, and warnings near branch
   boundaries.
3. Use implicit differentiation/custom-linear-solve contracts for
   differentiable linear solves rather than differentiating through every Krylov
   or setup iteration.
4. Keep optional `lineax`, `jaxopt`, `equinox`, and `optax` as measured optional
   lanes unless they clearly improve runtime, memory, accuracy, or clarity.

Acceptance:

- Differentiable workflows remain JAX-transformable on documented reduced
   fixtures.
- CLI defaults remain fast and residual-clean.
- No host-only fallback is silently used in a gradient path.

### P6. Raise Coverage With Real Tests

Goal: move toward 95% meaningful coverage without slowing CI.

Actions:

1. Every extracted module gets direct unit/regression tests.
2. Add physics/numerical gates for conservation/null modes, finite-difference
   order, symmetry limits, residual gates, output normalizations, and known
   bootstrap/transport trends.
3. Test solver primitives with synthetic sparse/operator systems instead of
   expensive production solves.
4. Keep CPU/GPU/Fortran sweeps in manual or release tiers.

Acceptance:

- Coverage increases because monolith responsibilities become testable.
- `docs/testing.rst` classifies tests as CI, release, manual GPU, or research
   tier.
- Normal CI remains practical.

### P7. Documentation And README Final Pass

Goal: keep public claims honest and usable.

Actions:

1. Update `docs/source_map.rst` after every ownership move.
2. Update `docs/testing.rst` when tests or validation tiers change.
3. Keep README focused on install, quick CLI usage, plotting, current
   runtime/memory figures, and short scope notes.
4. Keep deep research-lane details in docs, not the README.

Acceptance:

- README remains user-friendly.
- Docs are detailed enough for reviewers to trace equations, algorithms,
   source files, tests, and claims.
- Deferred research lanes stay explicitly labeled.

### P8. Benchmarks, Parity, And Figures

Goal: update public plots only from complete evidence.

Actions:

1. Do not regenerate runtime/memory/parity figures for pure refactors.
2. After behavior-changing solver work, rerun complete CPU reports locally and
   GPU reports on `ssh office` when needed.
3. Regenerate README/docs plots only from canonical complete JSON reports.
4. Keep reduced-grid QA/QH/QI figures labeled as reduced-grid until production
   gates pass.

Acceptance:

- Public plots trace to complete checked reports.
- Fortran v3 comparisons use matching model and resolution contracts.

## Review-Ready Boundary

The refactor PR is ready for review when:

- PR #8 points at the active branch and has one coherent story.
- `v3_driver.py`, `rhs1_full_assembly.py`, and `io.py` no longer hide major
  solver/preconditioner/output contracts in untested monolithic bodies, or any
  remaining responsibility is explicitly documented as deferred.
- Public CLI and Python APIs keep the same output schemas.
- Differentiable and non-differentiable solve lanes are explicit and tested.
- Focused local validation, `ruff`, `py_compile`, `git diff --check`, repo-size
  audit, and Sphinx pass.
- CI is checked after a meaningful push, not after every small local edit.
- README/docs accurately distinguish production claims, reduced-grid evidence,
  and deferred research lanes.

## Work That Is Explicitly Deferred

These lanes are important but should not block the refactor PR unless a code
change touches their public claims:

- True differentiable device-QI at production tolerance.
- Production-resolution QI ladders.
- Full-grid QA/QH RHSMode=1 production convergence beyond reduced-grid
  documentation evidence.
- Single-case multi-GPU strong scaling as a public performance claim.
- Lower-memory native sparse-factor replacement for the largest geometry-rich
  RHSMode=2/3 and full-grid QA/QH RHSMode=1 cases.

Deferred means fail-closed, documented, and test-gated where possible. Future
algorithm work should target genuinely stronger operator/coarse/factor
architectures and complete CPU/GPU/Fortran gates, not more smoother/restart
tuning.
