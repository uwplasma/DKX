# SFINCS_JAX Core-Slim Final Plan

Last updated: 2026-07-11. Active branch: `main` (PR #8 merged; single-branch repo)

This is the single active plan for the refactor branch. `plan.md` is the historical execution log. Do not create another competing plan. If any README, docs page, old
branch note, issue, benchmark artifact, or checklist conflicts with this file, follow
this file. This revision folds the previously separate architecture plan (see git
history of `plan_claude.md`) into this document; its research notes, literature
citations, and phase details remain available in history and in `docs/`.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core where one
input file plus one geometry runs accurate CPU/GPU neoclassical calculations
with automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime and memory, evident physics/numerics in the source, and
validated differentiable Python workflows for sensitivities, ambipolar roots,
bootstrap current, transport coefficients, plotting, and optimization.

## Current Review State

- Branch head follows the canonical-stack series: a parity-pinned replacement
  architecture now exists as flat root modules — `constants.py`, `species.py`,
  `phase_space.py` (grids/discretization), `magnetic_geometry.py`,
  `collisions.py`, `drift_kinetic.py` (the `KineticOperator`), `solve.py`
  (three solver tiers on the external `solvax` library), `moments.py`
  (diagnostics keyed by sfincsOutput.h5 names), `inputs.py` (typed namelist
  with Fortran-cited defaults), and `console.py` (byte-parity Fortran stdout).
  Each was admitted with equivalence tests against the old implementation
  (1e-13..1e-15) and, where applicable, Fortran golden data
  (`reference-data-v2` release) and tiny-grid PETSc matrix dumps.
- The canonical stack currently covers PAS + full-FP collisions, DKES and full
  trajectories, RHSMode 1/2/3, constraint schemes 0/1/2, geometry schemes
  1/2/3/4/5/11/12/13. It does not yet cover Phi1/quasineutrality, tangential
  magnetic drifts, constraint schemes 3/4, mapped speed grids, or export_f —
  the old stack remains the owner for those until each vertical slice lands.
- The external solver library `solvax` (github.com/uwplasma/SOLVAX) owns the
  reusable numerics: block-tridiagonal Schur elimination (full, factor/solve
  split, truncated-storage, transposed factor reuse, callback block assembly),
  banded/periodic LU, recycled FGMRES/GCROT, coarse-operator and multigrid
  preconditioners, implicit differentiation, mixed-precision refinement, and a
  host SuperLU bridge (98% coverage, docs on Read the Docs). `sfincs_jax`
  treats it as an optional dependency until it is published on PyPI; imports
  are lazy/guarded and CI installs it from git.
- Package size must still fall: the canonical stack is additive until the old
  owners are deleted slice by slice. Moving lines into more files without a
  same-series deletion is a failed tranche. The measured Fortran baselines
  (strong-scaling table, production-size memory limits, the ill-conditioned
  scheme-1 monoenergetic off-diagonal) live in `docs/dev/failure_analysis.md`
  pending promotion into `docs/` performance pages.

## Open Lanes

| Lane | Status | Done when |
| --- | --- | --- |
| Canonical-stack replacement | Active | Public API/CLI route every supported case through `inputs -> drift_kinetic -> solve -> moments -> writer/console`; each slice deletes its old owners in the same series. |
| Physics completion | Active | Phi1/quasineutrality, tangential magnetic drifts, constraint schemes 3/4, export_f, and non-stellarator-symmetric VMEC land in the canonical stack with parity gates, or are explicitly deferred with the old owner retained. |
| Performance and memory | Active | Supported cases meet the production admission gates below; the truncated block elimination demonstrates production-resolution runs that dense/MUMPS approaches cannot fit locally. |
| Strong-scaling parallelization | Active | On the local dev machine, sfincs_jax at N cores is competitive with `mpiexec -n N` Fortran (measured floor for the 744k-unknown HSX PAS case: 229.5 s at 2 ranks, degrading beyond); sharding is declared once at the grid layer (`shard_map`, virtual CPU devices), batch axes first, operator rows second. |
| Examples/tests/scripts cleanup | Active | Examples are original SFINCS-v3 references plus <=10 curated workflows; tests are smaller, meaningful, >=95% coverage, and default CI is under 10 minutes. |
| Parity/performance evidence | Active | Supported cases have checked Fortran-v3 parity/runtime/RSS/bootstrap evidence at documented grids; unpromoted cases are not marketed as stable. |
| Docs/readme | Active | Public docs describe standalone stable software with equations, algorithm derivations, and verified citations; README <=~250 lines with badges and honest benchmarks. |

## Source Structure Rules

The stable package converges to flat, physics-named root modules; one-level
domain packages only where a package earns its keep.

| Canonical owner | Stable purpose |
| --- | --- |
| `constants.py`, `species.py` | normalizations, radial-coordinate Jacobians, species pytrees, collisionality |
| `phase_space.py` | theta/zeta grids and derivative matrices, Legendre pitch machinery, speed grid, Nxi-for-x ramps |
| `magnetic_geometry.py` | all geometry schemes, VMEC/Boozer readers, differentiable Fourier path |
| `collisions.py` | pitch-angle scattering and full Fokker-Planck with Rosenbluth terms |
| `drift_kinetic.py` | the KineticOperator: term assembly, matrix-free apply, analytic Legendre blocks, RHS drives, bordered constraints |
| `solve.py` | three-tier policy: structured block elimination, preconditioned recycled Krylov, host direct referee; implicit differentiation |
| `moments.py` | velocity-space moments, flux families, transport matrices, NTV, classical transport |
| `inputs.py`, `console.py` | typed namelist (Fortran-cited defaults/validation), byte-parity stdout blocks |
| `api.py`, `cli.py` | thin public surface over the canonical modules |
| future: `er.py`, `phi1.py`, `writer.py` | ambipolar root solve, quasineutrality slice, output files |

Old owners (`problems/`, `operators/`, `solvers/`, `outputs/`, `discretization/`,
`geometry/`, `physics/`, `validation/`, `workflows/`, and implementation-heavy
root modules) shrink to zero through vertical-slice replacement; every slice
that routes a case family through the canonical stack deletes the superseded
old files in the same series. Stable file names must describe physics or
numerics — no `v3_`, version suffixes (`_v2`), `probe`, `campaign`, `rescue`,
`candidate`, `legacy`, `hard_seed`, `native`, `symbolic`, `multifrontal`,
`hss`, `blr`, or `qi_*` names. No nested package directories under
`sfincs_jax/`.

## Concrete Code-Audit Rules

- Every stable file starts with a module docstring naming its physics/numerics
  purpose and the SFINCS Fortran counterpart it mirrors; public functions
  document units and normalization conventions.
- Equivalence is the admission referee: a canonical module lands only with
  tests pinning it to the old implementation (or Fortran golden data) at
  documented tolerances; behavior changes require an explicit, tested reason.
- Solver or preconditioner promotion into `auto` requires the production
  admission gates: strict true residual, field-by-field Fortran parity on the
  supported matrix, cold and warm runtime, process peak memory, CPU/GPU
  agreement where available, and a finite-difference-checked gradient when the
  path claims differentiability.
- No env-var-only solver routes in stable code. Opt-in switches are namelist
  or API arguments with documented semantics; experiments live on research
  branches.
- The `solvax` boundary: anything with no neoclassical physics in it (linear
  algebra, Krylov, preconditioners, implicit diff) belongs in solvax, not in
  this package. sfincs_jax must import solvax lazily until it is on PyPI, and
  every module must remain importable without it.
- Attribution: all commits authored as Rogerio Jorge; no AI co-author
  trailers. Third-party research codes other than SFINCS are never named in
  tracked files; adopted numerical ideas cite the primary literature.

## Repository-Wide Line Sweep

Inventory-driven deletion continues, but deletions are now organized by
vertical slice rather than by file family: when a case family (for example
RHSMode 3 PAS/DKES) is fully served by the canonical stack — CLI, API, output
files, prints, parity evidence — the old pipeline files for that family are
deleted in the same series, their tests replaced by the canonical tests, and
`tests/fixtures/source_tree_expected.json` plus `sfincs_jax/README.md` updated
in the same commit. The package line count must decrease at every slice
milestone; the target remains <=50 files / <=50k lines first, then the
canonical end-state of roughly 15-20 root modules plus small `io`-adjacent
helpers (<=30k lines including docstrings).

## File-Level Execution Queues

1. RHSMode 3 slice (PAS/DKES; exact block structure): route CLI + API through
   the canonical stack end to end; delete the old transport-matrix pipeline
   for the supported subset.
2. RHSMode 2 slice (same operators, three drives).
3. RHSMode 1 PAS slice, then RHSMode 1 FP (tier-2 coarse-preconditioned
   Krylov), each with writer/console parity and old-owner deletion.
4. Ambipolar `er.py` on the canonical stack (Brent parity + differentiable
   root), replacing `ambipolar.py`.
5. Phi1/quasineutrality slice in `drift_kinetic.py` + `phi1.py` (Newton over
   the linear kernel with preconditioner reuse), unlocking the withPhi1
   examples; then tangential magnetic drifts.
6. Writer consolidation (`writer.py` from `outputs/` and `moments.py`), then
   export_f or its explicit deferral.

## Ordered Finish Plan

1. Keep CI green: the source-tree manifest, package README, and this plan are
   updated in the same commit as any module addition, rename, or deletion.
2. Execute the vertical slices in the order above; each slice ends with: the
   canonical path as default for its cases, old owners deleted, parity/runtime
   evidence recorded, and package lines lower than before the slice.
3. Land the strong-scaling lane: joint XLA/BLAS thread budgeting exposed as
   one `--cores`/API knob; shard the tier-1 batch axes and preconditioner line
   batches via one mesh declaration; add `tools/benchmarks/strong_scaling.py`
   and publish speedup-vs-cores curves against the measured Fortran table.
4. Promote the benchmark/evidence tooling: the head-to-head and
   reference-data generators live in `tools/`, produce the README table, and
   the release gates consume `reference-data-v2` (retiring hard-coded local
   paths and the v1-only manifest).
5. Curate examples to original v3 references plus <=10 workflows (CLI solve/
   plot, Python solve, output formats, transport coefficients, bootstrap/Redl,
   ambipolar root, autodiff/JVP, VMEC/Boozer loading, optimization objective,
   validation comparison), each a single readable script on the public API.
   Example style contract (simsopt-like): no main() functions; input
   parameters at the top; the user writes the objective function in the
   script; gradients come from jax.grad/value_and_grad or library functions
   that ship with gradients; scripts show how to read/create input files,
   write outputs, plot, and print initial conditions, run progress, and final
   results; no auxiliary functions that belong in the library. Flagship
   optimization: QA stellarator with low bootstrap current following the
   vmec_jax QA workflow with <j.B> from the kinetic solve (warm-started
   solves + recycling across iterations, autodiff accuracy tested vs finite
   differences, fast on CPU and GPU); alternative objectives (D11, L1, ...)
   present as commented lines that are themselves CI-tested so uncommenting
   them just works.
6. Consolidate tests into behavior suites; coverage >=95% comes primarily from
   deleting unreachable/experimental code, not from new test mass; default CI
   under 10 minutes.
7. Rebuild README/docs from the canonical stack: equations, discretization and
   solver-tier derivations with literature citations, complete namelist and
   output references, honest benchmark tables from `tools/benchmarks/`.
8. Only after the replacement is complete, CI is green, and the PR reviewed:
   repository hygiene (large-blob history rewrite, mailmap email unification,
   release assets policy) and the PyPI/docs release train for `sfincs_jax` and
   `solvax`. History rewriting is not part of ordinary refactor execution.

## No-Microtranche Rule

Do not spend a commit on one private helper unless it immediately unlocks a
larger deletion. Start every work block from the slice queue above, delete or
extract before moving code, keep file count flat or lower, prefer one clear
domain file over many attempt-named files, add absence tests for extracted
paths, and run focused tests plus one import/compile guard before committing.
Full coverage and production benchmark runs happen at slice milestones.

## Standard Validation Commands

- `python -m pytest tests/test_source_tree_consolidation.py -q` (structure,
  plan governance, README contract)
- `python -m pytest <touched-test-files> -q` per change; full
  `python -m pytest -q` at slice milestones
- `ruff check sfincs_jax tests tools`
- `python -m compileall sfincs_jax -q`
- `python -m pytest tests/test_docs_claims.py -q` when docs/README change
- Size guard: no tracked file >2 MB; large artifacts go to GitHub releases

## Completion Gates

- Every supported case family runs through the canonical stack by default;
  the old `problems/operators/solvers/outputs` owners for that family are
  deleted; package lines decrease at every slice milestone toward <=50k and
  the canonical end-state.
- Solver promotions pass the production admission gates (residual, parity,
  cold/warm runtime, peak memory, CPU/GPU, gradient).
- `solvax` is a declared dependency once published on PyPI; until then it is
  optional, lazily imported, and CI-installed from git.
- Examples <=10 curated workflows plus v3 references; `benchmarks/` absent;
  `scripts/` empty or documented release tooling.
- Tests are smaller and meaningful, >=95% coverage, default CI under 10
  minutes; parity gates consume the `reference-data-v2` release assets.
- README/docs match the slim core, include the measured baselines (Fortran
  strong-scaling table, memory limits), and do not market research paths.
- No generated clutter, caches, local profiles, large binary artifacts, or
  machine-specific paths in the tracked tree.
- PR #8 is clean, pushed, and ready for review; history rewrite and releases
  happen only after review.

## Explicit Deferred Items

Deferred unless production-gated: experimental QI/device-QI, native
sparse-direct research, multifrontal replacements, lower-memory preconditioner
research, GPU/multi-GPU campaigns, publication audits, and long stellarator
optimization campaigns. They may be referenced in `docs/research_lanes.rst`
only; they must not remain as stable source, examples, tests, README claims, or
default solver branches. Now canonical (Fortran-golden gated; their legacy
owners are the next deletions): Phi1/quasineutrality (kinetic and collision
coupling plus readExternalPhi1), tangential magnetic drifts (scheme 1),
constraint schemes 3/4, export_f plus `.npz` output and solver traces,
geometryScheme 13 (namelist Boozer spectrum), and non-stellarator-symmetric
VMEC (lasym). Still deferred with the old stack as interim owner — the last
gate before the legacy-stack deletion (decision: implement all, then trim):
xGridScheme 3/4/7/8 with `xDotDerivativeScheme != 0`, and magneticDriftScheme
2-9. Invalid-in-Fortran namelist values (quasineutralityOption > 2,
collisionOperator > 1, constraintScheme > 4) become validation errors, not
legacy fallbacks. The repository history rewrite and the PyPI release train are
deferred to after review per the Ordered Finish Plan.
