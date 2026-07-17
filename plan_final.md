# SFINCS_JAX Active Plan

Last updated: 2026-07-15. Active branch: `main` (single-branch repo; v1.2.0
released on PyPI and GitHub).

This is the single active plan. `plan.md` is the historical execution log.
Do not create another competing plan. If any README, docs page, old branch
note, issue, benchmark artifact, or checklist conflicts with this file,
follow this file.

## One-Sentence Goal

Ship a small, understandable, fast, research-grade `sfincs_jax` core where one
input file plus one geometry runs accurate CPU/GPU neoclassical calculations
with automatic robust defaults, SFINCS Fortran v3 parity where models overlap,
competitive runtime and memory, evident physics/numerics in the source, and
validated differentiable Python workflows for sensitivities, ambipolar roots,
bootstrap current, transport coefficients, plotting, and optimization.

## Current Review State

Shipped state as of v1.2.0 (2026-07-13):

- The canonical flat-module stack covers every SFINCS v3 physics family: PAS,
  full linearized Fokker-Planck, and the improved Sugama model operator
  (`collisionOperator = 0/1/3`); DKES and full trajectories; RHSMode 1/2/3;
  constraint schemes -1..4; geometry schemes 1/2/3/4/5/11/12/13 with lasym;
  Phi1/quasineutrality (kinetic, collision, readExternalPhi1); tangential
  magnetic drifts 0-9; xGridScheme 1-8 with xDotDerivativeScheme -2..11;
  export_f/.npz/solver traces. The legacy stack is deleted (98k -> ~34k lines)
  and must not return.
- Research capabilities beyond Fortran v3: Sugama-Nishimura momentum
  correction, monoenergetic (nuPrime, EStar) database mode with energy
  convolution, batched multi-`Er`/multi-surface `jax.vmap` scans, variational
  D11 upper/lower bounds, Shaing-Callen collisionless limit, and a
  differentiable bounce-averaged 1/nu effective-ripple surrogate with
  differentiable bounce points.
- Solver tiers live on `solvax` (PyPI, a **core dependency**): tier-1
  structured block-Thomas direct (full/truncated/ramp-aware), tier-2 recycled
  GCROT with an exact bordered-Schur coarse preconditioner, tier-3 host
  SuperLU referee. The bordered-Schur primitive with a nonzero border block
  `D` is upstreamed (`solvax.operators.schur_projected_precond(d_block=...)`);
  `solve.py` dispatches to it when the installed solvax exposes it. The Phi1
  Newton inner solve is preconditioned through the same machinery (production
  PAS Phi1: 9198 unpreconditioned inner iterations / ~398 s -> 5 / ~13.5 s,
  answers identical to machine precision). The improved-Sugama operator
  solves on the differentiable tier-2 (verified tier2 == tier3 to 5.7e-12,
  AD-vs-FD gradient 8.8e-12).
- Repository: 15 MB tracked (goldens lzma-compressed and materialised on
  demand; docs 4.7 MB with primary-literature citations only); docs build
  clean under `sphinx -W`; CI green (six coverage shards, <10 min each,
  heavy multi-compilation integration tests marked `slow`); 1000+ tests,
  Fortran-golden parity referees throughout.
- Flagship example: precise-QA vs precise-QA-held + bootstrap-reduction
  through the differentiable vmec_jax -> booz_xform_jax -> sfincs_jax chain
  (exact implicit/adjoint gradients end to end).

## Concrete Code-Audit Rules

- Equivalence is the admission referee: canonical changes land with tests
  pinning them to Fortran golden data at documented tolerances.
- Solver/preconditioner promotion into `auto` requires the production gates:
  strict true residual, field-by-field Fortran parity, cold/warm runtime,
  peak memory, CPU/GPU agreement where available, FD-checked gradients when
  the path claims differentiability.
- No env-var-only solver routes in stable code; opt-ins are namelist or API
  arguments. (Applies to external PRs too — request an API knob when a
  contribution ships env-var-only routing.)
- The `solvax` boundary: physics-free numerics belong in solvax. Contribute
  via PR branches off `origin/main` (solvax is actively PR-developed); gate
  downstream adoption on the released PyPI version with signature guards.
- Attribution: all commits authored as Rogerio Jorge; no AI co-author
  trailers. Third-party research codes other than SFINCS are never named in
  tracked files; adopted numerical ideas cite the primary literature.
- Repo hygiene: no tracked file > 2 MB; heavy goldens committed only as
  `tests/ref/*.xz`; README <= ~250 lines; docs cite, never vendor.

## Open Lanes

| Lane | Status | Done when |
| --- | --- | --- |
| Strong-scaling / time-to-solution evidence | Done (2026-07-16) | Published: docs/performance.rst cross-machine table + README bullet; harness tools/benchmarks/time_to_solution.py. |
| Research roadmap items 1-6 | Queued | Each lands per the roadmap section with parity/gradient gates. |

### Strong-scaling / time-to-solution lane

The community norm (set by the recent monoenergetic-solver literature) is
time-to-solution tables plus resolution-convergence evidence, not classic
speedup curves. Deliverable:

1. Measured table on the 744k-unknown HSX PAS production deck: Fortran v3
   MPI at n = 1/2/4/8 (laptop, 10 cores) and n = 1..32 (36-core
   workstation), against `sfincs_jax` single-process (all cores) cold and
   warm, and single-GPU. Report solve-driver seconds and end-to-end seconds.
2. Batched-throughput framing: monoenergetic coefficients/second and
   surfaces/second from the vmap scan API on CPU and GPU (the axis where a
   single JAX process replaces an MPI allocation).
3. Publish: `tools/benchmarks/` harness + a docs/performance section with the
   table and reproduction commands. Local Fortran baselines recorded at
   `fortran_scaling_baseline/`; workstation runs under `~/sfincs_scaling`.

## Research Roadmap (2026-07 literature review)

Priorities informed by the current landscape: a direct JAX competitor exists
without a published methods paper; the recognized benchmark template is the
monoenergetic-database comparison; kinetic-solver-in-the-loop optimization is
the community-stated need; gradient-based optimization through a full local
DKE solve has not been published by anyone.

1. **Methods + benchmark paper (top priority, medium scope).** Assemble: the
   ICNTS-style monoenergetic benchmark (D11*, D31* vs collisionality and
   Er/v for W7-X standard, LHD, TJ-II) against DKES/Fortran references;
   bootstrap convergence toward the Shaing-Callen limit at low collisionality
   (including the documented sub-asymptote dip); one W7-X ambipolar-Er
   experimental case; a reactor-profile full-transport case; wall-clock
   time-to-solution and coefficients/second tables; AD-vs-FD gradient
   verification tables. Most pieces exist as modules/tests — the work is
   assembly, convergence studies, and writing.
2. **Kinetic-in-loop bootstrap-consistent optimization demo (medium).**
   Replace analytic bootstrap proxies with the actual kinetic solve in a
   precise-QA/QH optimization (gradients are free here), plus one QI
   bootstrap-minimization case on a public benchmark boundary.
3. **Differentiable ambipolar/electron-root optimization workflow
   (small-medium).** Productize d(ambipolar Er)/d(shape): multi-species
   reactor profiles, branch handling near root transitions, one
   ion-to-electron transport-ratio optimization example.
4. **Impurity package (small-medium).** Classical fluxes (nearly free,
   algebraic), mixed-collisionality high-Z benchmark vs Fortran Phi1
   results, vmap over charge states, temperature-screening-aware gradients.
5. **Low-collisionality validity extension (large).** sqrt(nu) and
   superbanana-plateau regimes validated against bounce-averaged references
   and the Shaing-Callen convergence study; emit local-validity diagnostics
   (orbit-width/Er-layer parameters).
6. **Solver kernels (medium, opportunistic).** Adopt new solvax capabilities
   as released: `d_block` Schur (done upstream), iterative refinement /
   mixed-precision block-Thomas (replace the hand-rolled refinement step in
   `solve.py`), `chunk_map` (replace hand-rolled batch chunking), deflated /
   recycled Krylov across scan and optimizer continuation points.
7. Nice-to-have: differentiable ML surrogate distilled from the
   monoenergetic database; parallel-flow validation module against the
   published W7-X flow database; low-rank/tensor-train exploration stays a
   research branch.

## Ordered Finish Plan

1. Strong-scaling/time-to-solution lane: DONE (docs/performance.rst table,
   README bullet, tools/benchmarks/time_to_solution.py).
2. Stale-docs sweep: DONE (merged 32589968).
3. External PR #9 (Rosenbluth quadrature stabilization): reviewed, tests
   pass locally; API-knob request posted on the PR — merge once the
   contributor lands it.
4. solvax: next release (>=0.8.5) carries `d_block`; bump the sfincs_jax
   minimum then and swap the hand-rolled refinement/chunking for the solvax
   primitives (roadmap item 6).
5. Roadmap item 1 in progress: the ICNTS-style W7-X monoenergetic
   benchmark landed (examples/paper_benchmarks/, Fortran cross-checks at
   solver precision, orientation-robust Beidler normalization). Next
   pieces: TJ-II/HSX configurations, Shaing-Callen low-nu convergence
   study, gradient-verification table, W7-X ambipolar-Er case.

## Source Structure Rules

Flat, physics-named root modules; one-level domain packages only where a
package earns its keep (`validation/`, `workflows/`). Stable file names
describe physics or numerics — no version suffixes or attempt-names; no
nested packages under `sfincs_jax/`. The module table in
`sfincs_jax/README.md` and `tests/fixtures/source_tree_expected.json` are the
enforced inventory; both are updated in the same commit as any module
addition, rename, or deletion.

## Standard Validation Commands

- `python -m pytest tests/test_source_tree_consolidation.py -q` (structure,
  plan governance, README contract)
- `python -m pytest <touched-test-files> -q` per change; full
  `python -m pytest -q -n auto -m "not slow"` at milestones
- `ruff check sfincs_jax tests tools` and `python -m compileall sfincs_jax -q`
- `micromamba run -n sfincs-jax python -m sphinx -b html docs docs/_build/html -q -W --keep-going`
  when docs/README change
- Size guard: no tracked file > 2 MB; heavy goldens only as `tests/ref/*.xz`

## Completion Gates

- CI green on main (six coverage shards under 10 minutes each); docs `-W`
  clean; coverage >= the enforced fail-under.
- Solver promotions pass the production admission gates (residual, parity,
  cold/warm runtime, peak memory, CPU/GPU, gradient).
- README/docs match the shipped core with measured, reproducible benchmarks;
  research paths are not marketed as stable.
- Releases: version bumped in `pyproject.toml` + `__init__.py` together; a
  `v*` tag push publishes to PyPI (irreversible — explicit sign-off only).

## Explicit Deferred Items

Deferred unless production-gated: radially-global effects, MPI multi-node
execution (single-node multicore + GPU is the supported envelope),
experimental QI/device-QI campaigns, native sparse-direct research,
low-rank/tensor-train kinetic solvers, publication audits, and long
stellarator optimization campaigns. Invalid-in-Fortran namelist values
(quasineutralityOption > 2, collisionOperator not in {0, 1, 3},
constraintScheme > 4) are validation errors. History rewrites happen only as
explicit, reviewed operations.
