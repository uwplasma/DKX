# Changelog

## Unreleased

### Execution

- Apply the exact speed triangle (`preconditioner="coarse_triangle"`) by
  back-substitution over `x`, rather than reaching the same inverse as a
  nilpotent series of `n_x` sweeps of the whole band. The map, the factors and
  the memory are unchanged, and one application at NCSX `(25, 37, 61, 8)` costs
  3.7x less: 2.3x a `coarse` application against 8.5x.
- Escalate a stalled recycled-Krylov solve to that triangle before `sparse` and
  `multigrid`. Those three are inverses of one simplified operator, so none of
  them answers a stall caused by the Fokker-Planck speed coupling that operator
  drops; retaining its upper triangle changes the operator being inverted, cuts
  NCSX iterations 2.3-4.1x, and reuses the factors the coarse route already
  built. A deck with no dense collision operator skips the rung, because there
  the triangle is the stalled preconditioner under another name.

### Research records

- The speed-triangle back-substitution measurement, and the batched LU dispatch
  measurement on CPU and GPU, in `docs/experiments/`.

## v2.4.0 — 2026-09-15

The integrated #169–#191 stack, the reconciled #190 plan and the work from the
2026-09-13 independent review (#229–#246).

### Correctness and differentiation

- Report stalled solves without attributing an unmeasured physical cause or
  recommending changes to the requested electric field and accuracy.
- Add the opt-in `SfincsMatrixThreshold` parity switch, which reproduces SFINCS
  v3's `1d-12` matrix sparsification on the Fokker-Planck operator. It accounts
  for a 12-19% bootstrap-current gap on a hot-electron HSX-like deck; the
  default keeps every entry.
- Add `dkx.validity.normalized_radial_electric_field` and its operator-build
  form: the per-species `E_*` of Landreman et al. (2014), whose magnitude above
  about 1/3 marks the E x B resonance regime where trajectory models separate and
  resolution must follow `E_r`.

- Add opt-in bounded GMRES reuse to the host ambipolar root, with one cold retry
  after failed admission and independent final acceptance. Keep retry policy out
  of root differentiation.
- Correct the ambipolar teaching case to quasineutral analytic W7-X with full-FP
  collisions. JIT the geometry-optimization value/gradient, check its original
  kinetic residual and retain a three-step finite-difference window.
- Report monotone refinement ladders that converge faster than `max_order`, as
  spectral pitch and angle directions do, with three times the last difference
  as the grid uncertainty instead of refusing them. Diverging and oscillating
  ladders are still refused.

- Keep discrete pitch layouts static under JIT; refresh full-FP density kernels
  and opt-in temperature-dependent coefficients for prepared profile derivatives
  (#173, #174, #178, #186).
- Prepare immutable native profile/field scans with per-input original-equation
  status and complete kinetic-state recovery. Differentiate supported regular
  ambipolar roots with original primal/transpose admission (#182, #184, #187, #188).
- Reject invalid kinetic states, roots and adjoint references; qualify physical
  profile sensitivities with independent cold solves and Taylor checks
  (#180, #184, #188).

### Execution

- Bound the first automatic Krylov attempt and reuse its factors with a wider
  restart window when the basis-size guard permits. Preserve explicit solver
  settings and the differentiated path.

- Propagate nested legacy scan failures, distinguish failed progress from success,
  and retry incomplete Er points while preserving previous attempt files.
- Default the host BLAS pools to one thread. XLA's threadpool already runs
  batched CPU LAPACK in parallel; four BLAS threads per call made the NCSX
  `(21,37,61,8)` coarse factorization about 3x slower. Explicit settings win.

- Build the transposed coarse preconditioner on its first application, and
  synchronize coarse builds on their factors instead of a discarded zero-vector
  application. Non-differentiable solves no longer pay two transposed coarse
  applications per build; the preconditioner maps are unchanged.
- Factor the dense coarse preconditioner straight from its pinned row generator
  inside one compiled computation, so no band is materialized before
  elimination, and store only the rows each subsystem's `Nxi_for_x` keeps. The
  elimination and both substitution sweeps run one Legendre row at a time,
  batched over the subsystems active at that row, so the kernels launched per
  application scale with the longest chain. The truncated rows are an uncoupled
  `(1 + floor) I` and are applied as such. The map is unchanged: the f-block
  inverse agrees with the reusable Schur-LU route to rounding, GCROT iteration
  counts are identical, and the band guard is left as it was, conservative for
  ramped decks (`tests/test_coarse_ragged_chains.py`).

- Preserve independent-device batch sharding through JIT and gradients, including
  uneven batches (#179). Each complete system still resides on one device.
- Reuse generated SOLVAX Schur factors within supported PAS solve executions for
  multiple right-hand sides, transpose solves and refinement, subject to memory
  policy (#188). Persistent reuse across changed operators is still future work.
- Require SOLVAX 0.21.0, whose bordered Krylov preconditioner applies the coarse
  inverse once per iteration instead of twice, matching the release used for
  recent DKX timings.

### Benchmarking and development

- Supervise benchmark process groups and cancellation; reject invalid reference
  outputs and stale resumed results; retain original-equation checks and evidence
  (#170, #183, #185).
- Preserve requested PETSc options and observed backends; make source inventories
  reproducible; correct sparse reference matrix interpretation (#169, #176, #177).
- Preserve the tested package tree when GitHub merge refs move (#181), and balance
  thirteen coverage shards using measured test durations (#191).
- Adopt one authoritative figure-first plan, a concise workflow-oriented README,
  grouped documentation and citation metadata (#189, #190). Establish decision
  records and an experiment template in Phase 0.
- Revise the plan from the 2026-09-13 independent review (#230). Take
  finite-difference gradient references at `1e-12`, bound the sparse-map `pas`
  check at `1e-7`, and regenerate `.test_durations` from one complete run on one
  host so the thirteen coverage shards fit their time cap (#238, #241).

### Research records

- NCSX `(21,37,61,8)` refinement ladder with baseline adjoint estimates, its
  `Nx = 11, 12` speed rungs, and the cause of GCROT iteration growth with
  `Ntheta`: the dropped Fokker-Planck speed coupling (#237, #240, #242).
- SFINCS v3's `1e-12` sparsification threshold explains the 12-19% current gap
  on the HSX-like deck, confirmed from the SFINCS side and reported upstream as
  landreman/sfincs#27 (#243).
- SOLVAX operator couplings in the coarse preconditioner halve its memory but
  run 3x slower, so the dense route stays; explicit-inverse storage is rejected
  as not backward stable (#244).
- The HSX-like deck at resonant `E_*` is resolved in pitch by `Nxi = 120` but
  not in speed for the ion channel; no bootstrap current or ion flux at its two
  points is admitted (#245, #246).

These changes do not certify converged production performance, general persistent
restart reuse, full native Phi1 support or complete equilibrium-boundary optimization.
Historical measurements and their limits remain in the performance documentation.
