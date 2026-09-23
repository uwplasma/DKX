# Changelog

## v2.7.0 — unreleased

### Optimization

- `dkx.bootstrap.KineticBootstrapMismatch`: the drift-kinetic bootstrap
  current as a traced VMEX objective term, with the interface and normalized
  residual of VMEX's `RedlBootstrapMismatch`. On each kinetic surface the chain
  VMEX state -> `boozer_input_tables` -> `booz_xform_jax` -> DKX structured
  solve -> `<j.B>` is traced, so VMEX's implicit Jacobian differentiates it;
  `mismatch=False` gives a pure `<j.B>` row. Pitch-angle scattering is the
  default collision operator (cheap, no momentum restoration).
- `examples/optimization/QA_optimization_bootstrap_dkx.py` is VMEX's
  `QA_optimization_bootstrap.py` with the DKX row in place of Redl's
  (`BOOTSTRAP_MODEL = "dkx" | "redl" | "both"`), replacing the
  finite-difference version; the seed deck ships in `examples/data`.

### Correctness

- Convert the handedness of the Boozer route in one place
  (`boozer_route_psi_a_hat`, and its traced form
  `convert_boozer_route_handedness`): `psiAHat = |phi_edge|/(2 pi) signgs
  sign(G + iota I)`. A VMEX equilibrium has `signgs = -1` and `booz_xform`
  returns `G > 0`, so taking `psiAHat = +|phi_edge|/(2 pi)` flipped every flux
  and `<j.B>` on that route against the VMEC-file route and Redl.
  `optimize_QA_bootstrap.py`, `optimize_QH_bootstrap.py` and the gradient hook
  of `bootstrap_consistency_kinetic_loop.py` printed the flipped sign; their
  objectives were squares or magnitudes, so the optimizations were unaffected.

## v2.6.0 — 2026-09-21

Acceptance tightened where it could pass a wrong answer, MUMPS as an explicit
direct backend, and convergence and optimization reports that no longer claim
more than their evidence supports.

### Correctness

- Admit each right-hand side against its own norm (#265). A multi-column solve
  admitted every column against the *largest* right-hand side's norm, so with
  norms `[1, 1e6]`, residuals `[1e-5, 0]` and a tolerance of `1e-10` the small
  column passed while missing its own target by a factor of 100,000. Each
  column now meets `max(atol, tol * ||b_j||)`, reused factors included; zero
  columns use the requested absolute tolerance, and nonfinite inputs, negative
  residual norms and invalid tolerances are refused.
- Include the magnetic drift's upwind support in the sparse assembly (#268).
  Streaming uses centred derivatives and the tangential magnetic drift separate
  upwind ones, whose toroidal radius is three points rather than two on the
  public reduced W7-X case; the pattern and the column groups now use the union
  of the active supports. The assembly's own `1e-10` check against the operator
  refused the old pattern, so no wrong matrix was ever factored; decks without
  magnetic drifts assemble exactly as before.
- Reject invalid optimization evidence and require a genuine refinement before
  promotion (#264): nonfinite or negative residuals, a single baseline grid
  labelled as a ladder, and a refinement that changes radius or species count
  are refused, and an absent backend comparison is reported as `untested`.

### Execution

- Select MUMPS explicitly as the sparse direct backend,
  `SolverOptions(method="direct", direct_backend="mumps", memory_budget_gb=...)`
  (#267). SuperLU stays the default and the default path is unchanged. The
  budget is checked after assembly and before factorization, against measured
  process memory, the COO storage, the right-hand-side buffers and the host's
  currently available memory, and again before a retained factorization is
  applied to a wider right-hand side. An explicit MUMPS request that cannot be
  met is refused and never silently falls back to SuperLU. It needs the MUMPS
  adapter of SOLVAX 0.25.0 and PyMUMPS; with SOLVAX 0.24 it raises an
  `ImportError` naming what is missing.

### Reporting

- Report grid convergence and original-equation acceptance as separate
  verdicts (#269). `dkx converge` keeps `converged` for the observable changes
  and adds `original_equations_accepted`, which requires complete, finite,
  typed residual evidence within the requested tolerance on every rung; the
  exit status is zero only when both pass. Native runs reapply the operator to
  the returned state and reject a false solver success.
- Compare bootstrap current in `dkx converge`, keep earlier rungs when a later
  one fails, and record failed and refused refinements without claiming
  convergence (#266). The NCSX record gains an executable reconstruction from
  pinned public sources with four SHA-256 checks.
- Report memory in one unit (#271). Traces mixed decimal MB with MiB and could
  present device capacity or a historical peak as current usage; profiling now
  uses MiB throughout and keeps current RSS, peak RSS and current device usage
  distinct.

### Development

- Let a cancelled CI run stop before the final gate (#270).

### Known limitation

- The sparse direct route's transposed solve can stall above its tolerance for
  a general cotangent. On the operator of
  `tests/test_operator_assembly.py::DECK` (1,204 unknowns) with the cotangent
  `numpy.random.default_rng(0).standard_normal(1204)`, it reaches a relative
  residual of `3.18e-8` against the forward solve's `1e-10` on the same
  factors, in 2.5.0 and here alike; further refinement sweeps do not move it.
  The route reports `converged=False` rather than a value, so this is an
  accuracy limit that is reported, not a wrong answer. The 1,962-unknown
  full-FP grid of #265 reaches `2e-13`, so the limit depends on the operator.

## v2.5.0 — 2026-09-20

The production solver program of #253: the operator assembled from products
with it, the sparse direct route rebuilt on that assembly and made reusable,
and the gap deck's iteration growth traced to the speed resolution with six
candidate causes measured and excluded.

### Execution

- Return the factorization the two direct routes build, in
  `SolveResult.factors`, and take it back through `solve(..., factors=...)`;
  `solve(..., transpose=True)` solves `A^T x = b` from those same factors. A
  transport matrix delivered as three separate calls cost three factorizations
  and now costs none: 0.96 s against 0.28 s on a 16,230-unknown structured
  deck, 1.19 s against 0.28 s on a 1,962-unknown sparse one. The sparse direct
  route had no adjoint at all; a transposed solve is now 0.15 of a primal,
  reached by substituting through the stored `L` and `U` in the other order.
  Reuse across a *neighbouring* operator is permitted and checked rather than
  assumed: the defect is measured against the operator actually passed in, and
  a solve that misses its tolerance refactorizes once, so recovery is bounded
  by a factorization instead of an unbounded Krylov wait.
- Take the operator's transposition once per solve rather than once per
  application. `_transposed_apply` called `jax.linear_transpose` inside the
  callable it returned, so every adjoint application re-traced the whole
  operator. The focused solve suites run in 247 s where they took 406 s.
- Assemble the sparse direct route's matrix in fewer products by dropping the
  requirement that the angular separation divide the grid. The grouping was
  already optimal for its own conflict structure -- every group held exactly 72
  columns and `633,600 / 72 = 8,800` -- but at `Ntheta = 11`, a prime, the only
  divisor above twice the stencil radius is 11 itself, so every theta became
  its own class. Checking the wrap-around gap directly instead takes the
  633,604-unknown deck from 8,800 products to 4,800, each product being one
  operator application; the recovered matrix is unchanged entrywise.

- Build the sparse direct route's matrix from products with the operator rather
  than one column at a time, scale it before factoring, and correct the defect
  once. Sampling costs one operator application per column, which is what
  `max_dense_size` bounds and why the route refused every production deck; the
  assembly costs one per group of columns that share no row, 5,508 for the
  66,004-unknown collaborator grid. That deck now solves in 846 s to a relative
  residual of `1.3e-14`, where the route previously refused it.
- Apply the exact speed triangle (`preconditioner="coarse_triangle"`) by
  back-substitution over `x`, rather than reaching the same inverse as a
  nilpotent series of `n_x` sweeps of the whole band. The map, the factors and
  the memory are unchanged, and one application at NCSX `(25, 37, 61, 8)` costs
  3.7x less: 2.3x a `coarse` application against 8.5x.
- Continue every escalation rung from the stalled iterate, rather than from the
  original guess, when that iterate is finite and improved on the right-hand
  side it started from. On a forced NCSX stall the rung that converges takes 18
  iterations instead of 27, for the same answer. A diverged iterate is refused,
  so a rung never starts further away than the original guess.
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
- Why the gap deck is slow, in four records. The dropped collision coupling is
  not the cause although it is 0.9999997 of `A - M`; retaining it exactly costs
  17% *more* iterations. The cost is carried by `Nx` and not `Nxi`, and is
  independent of the tolerance. The preconditioned operator is strongly
  non-normal and far more so with `Nx`, and a diagonal similarity that removes
  1.5e10 of that eigenvector conditioning moves the iteration count by at most
  1.05x, so the balanced solve is killed on its own criterion. Six candidate
  causes are now measured and excluded.
- What one factorization serving many solves does and does not buy: the
  transport matrix admitted on both direct routes, the gradient admitted on the
  sparse one, and `jax.grad` measured to miss its gate for a reason factor
  reuse cannot address, since reverse mode re-executes the forward pass and the
  adjoint already runs on the primal's factors.

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
