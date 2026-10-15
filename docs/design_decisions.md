# Design decisions

This page records the main design choices in DKX, why each was made, and what
each costs. Every decision is stated as context, choice and consequence, with
the measurement that supports it and where that measurement comes from. The
numerical detail behind each entry is on the linked page.

## JAX as the implementation language

**Context.** Neoclassical transport enters stellarator optimization as an
objective, and an objective needs derivatives with respect to hundreds of
boundary or profile parameters. A Fortran/PETSc code gives one solve per call
and derivatives only by finite differences, at $2k$ solves for $k$ parameters.
The same code should also run on laptops, workstations and GPUs.

**Choice.** Write the operator, the right-hand side, the moments and the solver
glue as pure JAX functions, compiled with `jax.jit`, differentiated with
`jax.grad`, batched with `jax.vmap`, on CPU or GPU from one source.

**Consequence.**

- Gradients cost about one extra solve. Compiled, `jax.grad` of `FSABjHat` on a
  16,230-unknown structured deck costs 1.00 (median) to 1.11 (minimum) of its
  primal, while central differences cost 3.1 s per parameter on the two-species
  upstream decks (`tools/benchmarks/derivative_cost.py`,
  `tools/paper_benchmarks/gradient_cost_scaling.py`; {doc}`numerics/differentiation`).
- One source serves CPU and GPU. On the 744,610-unknown HSX case a development
  MacBook CPU and an RTX A4000 warm-solve in 25.0 s and 26.2 s
  ({doc}`benchmarks/performance`).
- Compilation is a real cost. On tiny monoenergetic decks compilation takes 1.2
  to 1.4 s against a 0.05 to 0.06 s warm solve
  (`docs/_static/figures/transport_compile_runtime_cache_2x2.json`), so single
  small runs are compile-dominated and benchmarks must be reported warm.
- Uncompiled callers pay per-primitive dispatch: an eager gradient was 1.76 to
  1.79 times its compiled time before the structured route compiled itself
  ({doc}`numerics/compilation_and_parallelism`).
- XLA:CPU runs batched LAPACK calls serially per batch element and its thread
  pool inverts past about 8 threads (4.87 s at 8 threads against 56.6 s at 32 on
  a 36-core host), so DKX clamps threads by default and does not gain from wide
  CPU batches.
- There is no MPI decomposition of one solve. DKX targets one node and gets
  throughput from batching independent solves instead.

## SFINCS version 3 compatibility

**Context.** SFINCS v3 is the reference radially local drift-kinetic code, with
a decade of published results, benchmark decks and users. A new solver is only
credible if it can be compared against it entry by entry, and users should be
able to bring their decks unchanged.

**Choice.** Reproduce the v3 discretization, normalization and output format
exactly: the same namelist with the same defaults, the same grids, the same
`sfincsOutput.h5` dataset names, the same stdout blocks. Parity is tested
against frozen PETSc matrices and Fortran output goldens, and DKX-only options
(improved Sugama collisions, widened upwind stencils numbered in a 100 block
upstream cannot collide with) are opt-in and never defaults.

**Consequence.**

- The discrete operators can be compared directly, which is how discrepancies
  are attributed. On a hot-electron HSX-like deck the two codes disagreed on the
  bootstrap current by 12 to 19 % while right-hand sides agreed to 6e-15; the
  cause was SFINCS's hard-coded sparsification, which drops matrix entries with
  $|a_{ij}| \le 10^{-12}$ and with them part of the ion–electron collision block.
  DKX keeps every entry and offers `SfincsMatrixThreshold` as an opt-in parity
  switch only ({doc}`benchmarks/sfincs`).
- DKX inherits the Legendre-modal pitch basis, in which streaming and the mirror
  force have no diagonal. That rules out multigrid smoothing and fixes the
  preconditioner family ({doc}`numerics/krylov_and_preconditioners`). A pitch
  grid would allow it but would change answers at fixed resolution and break
  matrix parity, so it lives in the experimental `dkx.collocation`.
- The rectangular state layout keeps inactive $N_\xi(x)$ entries as pinned zero
  rows instead of the packed Fortran indexing; solutions and adjoints match the
  packed system ({doc}`numerics/discretization`).

## Several solver routes instead of one

**Context.** The operator family is wide. Pitch-angle scattering with DKES
trajectories is block tridiagonal in $L$ with uncoupled speed and species; full
Fokker–Planck and Sugama collisions keep that structure but couple every species
and speed inside each block; $L\pm2$ $E_r$ terms, tangential magnetic drifts and
$\Phi_1$ break it. No single method is both exact and affordable across that family.

**Choice.** Three routes behind one `solve` call: an exact structured direct
elimination for the block-tridiagonal family, a preconditioned recycled Krylov
method for the rest, and a sparse direct referee that shares no algorithm with
the other two. `method="auto"` picks by operator structure and memory estimate
and prints why.

**Consequence.**

- Exact answers where the structure allows. The structured route needs no
  iteration count, tolerance tuning or preconditioner.
- Memory is traded for speed at every level, explicitly. The truncated
  structured route solves the 744,610-unknown HSX case in about 0.3 GB where a
  full-band factorization needs about 91 GB. The Krylov restart widens to 1,000
  only when its basis ($2 \times \texttt{restart} \times N \times 8$ bytes) fits a
  quarter of available memory, which cut the `Nx = 16` HSX-like solve from 2,788
  to 357 iterations for 1.6 GiB of basis. The coarse preconditioner switches
  from dense bands to regenerated Schur factors to checkpointed elimination as
  memory runs out ({doc}`numerics/krylov_and_preconditioners`).
- The referee is expensive by design: its cost grows as the 2.8 power of the
  unknowns in time and 1.6 in memory, so it reaches a few hundred thousand
  unknowns on a 62 GiB host ({doc}`numerics/solver_routes`).
- Three routes are three things to test. Each route recomputes the original
  residual itself, and the routes cross-check each other on shared decks.

## Structured block-tridiagonal elimination in the Legendre index

**Context.** In the Legendre basis, streaming and mirror couple $L\pm1$, while
$E\times B$ and pitch-angle scattering are diagonal in $L$. For that family the
species and speed indices decouple, leaving $N_\mathrm{species}N_x$ independent
chains of dense $N_\theta N_\zeta$ blocks. A general sparse factorization does
not see this structure and fills in.

**Choice.** Eliminate each chain with block-Thomas (following Escoto,
arXiv:2510.27513, and Hirshman et al., *Phys. Fluids* **29**, 2951 (1986)), and
keep only the lowest `tier1_keep_lowest = 3` blocks of the solution, where every
RHSMode 1, 2 and 3 drive and output moment lives.

**Consequence.**

- Peak memory is $\mathcal{O}(K m^2)$ with $m = N_\theta N_\zeta$, independent of
  $N_\xi$ and $N_x$. On one RTX A4000 a 2,525,010-unknown solve peaks at 2.21 GB
  where the full-band charge is about 208 GB
  (`tools/benchmarks/gpu_anatomy_figure.py`).
- On the 744,610-unknown HSX case DKX finishes end to end in 41.4 s on a
  laptop CPU where SFINCS v3 with MUMPS had not finished after 2.6 h
  ({doc}`benchmarks/performance`).
- The truncated tail is zero-filled: moments are exact, but the state is not a
  full-equation solution. Residual audits and restarts must request every block.
- The same elimination is the preconditioner of the Krylov route, applied to a
  simplified operator. It cannot be shortened there: cutting one link of a
  48-block coarse chain raised GCROT iterations from 26 to 133, cutting four
  prevented convergence (`tools/benchmarks/tier2_coarse_truncation.py`).

## Float64 throughout

**Context.** Drift-kinetic operators at low collisionality are badly
conditioned: pitch-angle decks have been measured near $\kappa = 3\times10^{12}$.
Single precision halves memory and doubles GPU throughput on consumer cards.

**Choice.** Every solve runs in float64. `dkx.runtime.configure()` enables x64
and `dkx.require_float64()` refuses to solve without it. Reduced precision is
allowed only inside the Krylov preconditioner, as the opt-in
`DKX_COARSE_FACTOR_DTYPE=float32`, where it can change iteration counts but not
the answer.

**Consequence.**

- With one float64 refinement sweep the structured route's forward error stays
  at 2.5e-14 at $\kappa = 3.3\times10^6$, against 1.4e-7 without the sweep,
  measured against a 50-digit reference on analytic decks
  ({doc}`numerics/solver_routes`). No extended-precision residual is needed on
  the decks where a reference exists.
- Float32 preconditioner factors are free on some decks and expensive on
  others: 19 to 24 GCROT iterations on a tokamak deck, but 3 to 228 on a
  pitch-angle deck where the preconditioner is nearly exact (40 to 76 times the
  work). The default stays float64.
- FP64 rate on consumer GPUs is low (1/32 on the RTX A4000), and the structured
  route is FP64-compute-bound there.

## Factor reuse

**Context.** A transport matrix is three right-hand sides of one operator, a
gradient is one transposed solve, and a workflow often delivers those one call
at a time.

**Choice.** The direct routes return their factorization in
`SolveResult.factors` and accept it back through `factors=`, with
`transpose=True` for the adjoint. Factors of a neighbouring operator are
admitted only after the original residual is recomputed; a miss refactorizes
once.

**Consequence.**

- Three right-hand sides in three calls drop from three factorizations and
  0.96 s to none and 0.28 s on a 16,230-unknown structured deck; a sparse
  transposed solve costs 0.15 of a primal ({doc}`numerics/factor_reuse`).
- Reuse across neighbours is narrow at production tolerance: the structured
  route tolerates a $10^{-6}$ relative change in `THat`, the sparse route
  $10^{-3}$. Beyond that, the bounded recovery is a factorization.
- Reuse is explicit. Nothing is cached behind the caller's back, so no stale
  factor can silently serve a changed operator.

## Implicit differentiation instead of unrolled solves

**Context.** Differentiating through Krylov iterations or an unrolled
elimination tapes every step: memory grows with iteration count or $N_\xi$, and
the derivative inherits the iteration's noise.

**Choice.** Wrap the linear solve in `jax.lax.custom_linear_solve` and the
outer root problems ($E_r$, $\Phi_1$) in `jax.lax.custom_root`, through SOLVAX.
The adjoint is one transposed solve at the converged state, reusing the forward
factors or preconditioner. The transposed residual is recomputed from the
operator and a miss raises.

**Consequence.**

- The gradient's cost does not depend on how many iterations the forward solve
  took, and a compiled gradient costs 1.00 to 1.11 of its primal.
- Where the implicit wrapper cannot be used, the difference is measured: taping
  the truncated structured kernel grows the gradient's working set from 0.161
  to 3.949 MiB as $N_\xi$ goes from 8 to 256, while the windowed custom VJP stays
  at 0.068 MiB ({doc}`numerics/differentiation`).
- A stalled adjoint cannot hide. On a singular full Fokker–Planck system the
  forward solve converges while the transposed solve fails; the check turns
  that into an error instead of a wrong gradient.
- The sparse direct route is host code and is not differentiable under
  `jax.grad`; it offers an explicit transposed solve instead.

## A native case format beside the namelist route

**Context.** A SFINCS namelist is in normalized units, mixes physics and
numerics, and depends on external files named by path. That is right for parity
and migration and awkward for scans, optimization and reproducible records.

**Choice.** Two front doors to the same kernels. The native `dkx.Case` is an
immutable model in physical units with a deterministic case ID,
executed without a namelist round-trip. The namelist route reads SFINCS decks
exactly as the Fortran code would and writes `sfincsOutput` files.

**Consequence.**

- Case files are portable, validated on load, and identified by content; the
  tutorial scripts assert that each case file and its Python script share one
  case ID.
- The native executor covers a subset of the namelist physics. The
  monoenergetic and transport-matrix workflows and the $\Phi_1$ options
  beyond `phi1 = "kinetic"` run only through the namelist or the operator API ({doc}`user_guide/inputs`, {doc}`user_guide/sfincs_namelist`).
- Two input paths must stay equivalent where they overlap, which is a maintenance cost.

## Convergence certificates

**Context.** A converged linear solve says nothing about grid error, and a small
Krylov residual estimate says nothing about the true residual. Published
transport numbers need both kinds of evidence, stated separately.

**Choice.** Every result carries its own evidence. The residual used for
acceptance is recomputed from the original operator, per right-hand-side
column. `Result.certificate()` records the route and why it was chosen, the
residual evidence, iterations, versions, precision, device, geometry checksum,
timings and peak memory. `dkx converge` reports `converged` (grid convergence of
the observables) and `original_equations_accepted` (every rung's full-state
residual) separately and exits zero only when both pass. Richardson error bars
are refused, not reported, when the ladder is not monotone.

**Consequence.**

- An oscillating ladder is caught: the measured `Er = 15` pitch ladder, whose
  bootstrap current changed sign between `Nxi = 40` and `60`, is refused by the
  convergence-ratio test, and a regression pins that.
- A spectrally converging ladder is reported with a conservative bar: NCSX
  pitch rungs at `Nxi = 81, 101, 121` show apparent orders of 17 to 25, above
  `max_order = 12`, and receive Roache's factor-3 bar instead of an
  extrapolation ({doc}`user_guide/convergence`).
- Certificates cost an extra operator application per accepted state, and a
  memory-bounded structured solve must recover every Legendre block to be
  auditable.

## Evidence-pinned documentation

**Context.** A solver's documentation drifts from its code: defaults change,
records of one-off measurements are superseded, and numbers lose their origin.

**Choice.** Every quantitative claim in these pages names its source: a
checked-in test, a script under `tools/` or `examples/`, a data file beside the
figure, or the deck, host and settings it was measured on. Each figure is listed with its generator in `docs/figure_provenance.json`, which
`tests/test_figure_provenance.py` enforces. Measurements from
bounded experiments are carried into the reference pages with their provenance,
so the pages stand on their own. Tests pin key phrases and the package layout
statements. Work is scheduled against a named figure or table or a correctness
defect (ADR 0001 in `docs/adr/`).

**Consequence.**

- A reader can rerun the script behind a number, and a reviewer can tell a
  measured number from an estimate.
- Wall times on shared hosts are labelled as such; iteration counts and
  factorization counts, which contention cannot move, are preferred as
  evidence.
- Scope statements are part of the evidence: what is not supported, not
  converged or not measured is stated next to what is.
