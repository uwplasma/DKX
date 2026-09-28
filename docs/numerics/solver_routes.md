# Solver routes

Every linear solve in DKX goes through one function, `dkx.solve.solve`, over a
`KineticOperator`. With `method="auto"` (the default) it picks the cheapest
route that can answer the operator in front of it. There are three routes. The
factorizations, the Krylov method and the implicit-differentiation wrappers
underneath them come from the SOLVAX library, a required dependency
(`solvax>=0.24.0` in `pyproject.toml`).

```{list-table}
:header-rows: 1

* - Route
  - Case-file `[solver] method`
  - `solve(method=...)`
  - What it does
* - Structured direct
  - `structured_direct`
  - `"block_tridiagonal"`, `"block_tridiagonal_truncated"`
  - Exact block elimination along the Legendre index, one chain per (species, speed) pair
* - Recycled Krylov
  - `recycled_krylov`
  - `"iterative"` (alias `"gmres"`)
  - Matrix-free GCROT-recycled flexible GMRES, right-preconditioned by an exact solve of a simplified operator
* - Sparse direct referee
  - `sparse_direct_referee`
  - `"direct"`
  - Host factorization of the assembled, equilibrated matrix
```

Some code identifiers keep a numbering from the package's history: `tier1`
names belong to the structured direct route (`tier1_keep_lowest`,
`DKX_TIER1_MEMORY_BUDGET_GB`), `tier2` to recycled Krylov
(`DKX_TIER2_MEMORY_GUARD`) and `tier3` to sparse direct. Prose on these pages
uses the route names.

## The `auto` policy

```{list-table}
:header-rows: 1

* - Operator
  - Route taken
  - Reason
* - DKES trajectories or $E_r = 0$, pitch-angle scattering, `constraintScheme` 0 or 2, uniform `Nxi_for_x`, full-band estimate within budget
  - structured direct, full factorization
  - block tridiagonal in $L$; $(s, x)$ chains uncoupled
* - Same family, ramped `Nxi_for_x` or full-band estimate over budget, RHSMode 1, 2 or 3
  - structured direct, truncated
  - drives and output moments live on $L \le 2$
* - Full Fokker–Planck or improved Sugama collisions
  - recycled Krylov
  - dense speed and species coupling
* - Full-trajectory $E_r$ terms, magnetic drifts
  - recycled Krylov
  - $L \pm 2$ couplings
* - Recycled Krylov stalled after its escalation ladder, and the system is small (or MUMPS was selected with a budget)
  - sparse direct
  - last resort, printed as a one-line notice
```

The structured route is admitted by `dkx.solve.tier1_available`, which asks the
operator's own block extraction (`KineticOperator.legendre_blocks` refuses
Fokker–Planck and $L\pm2$ terms) and requires the constraint border to be
diagonal over $(s, x)$: `constraintScheme` 0 or 2, and not a speed grid with a
node at $x = 0$. The choice between full and truncated elimination is a memory
decision. The full factorization is charged 2.5 times its band storage
(`tier1_peak_memory_bytes`) and is taken when that fits
`tier1_memory_budget_gb`, default 8 GB (`DKX_TIER1_MEMORY_BUDGET_GB`); the default
matches the validated HSX head-to-head (`tools/benchmarks/tier1_hsx_head_to_head.py`).

Each decision prints one line, for example
`[dkx.solve] memory-bounded structured direct route: truncated block-Thomas (keep_lowest=3); ...`,
so a log shows which route ran and why. An explicit `method` that cannot handle
the operator raises instead of falling back.

## Structured direct: block elimination along $L$

When the operator is block tridiagonal in $L$ and the $(s,x)$ axes are
uncoupled, the system splits into $N_\mathrm{species} N_x$ independent chains of
$N_\xi$ dense $m\times m$ blocks, $m = N_\theta N_\zeta$, each with a rank-one
constraint border that is absorbed exactly. For one chain with diagonal blocks
$D_k$ and couplings $L_k, U_k$,

$$
A =
\begin{bmatrix}
D_0 & U_0 \\
L_0 & D_1 & U_1 \\
& \ddots & \ddots & \ddots \\
& & L_{n-2} & D_{n-1}
\end{bmatrix},
$$

the block-Thomas recursion factors

$$
S_0 = D_0,\qquad C_0 = S_0^{-1}U_0,\qquad
S_k = D_k - L_{k-1}C_{k-1},\qquad C_k = S_k^{-1}U_k,
$$

and every later right-hand side costs one forward and one backward sweep,

$$
y_0 = S_0^{-1}b_0,\qquad y_k = S_k^{-1}\left(b_k - L_{k-1}y_{k-1}\right),\qquad
u_k = y_k - C_k u_{k+1}.
$$

The chains are eliminated with a `vmap`-ed block-Thomas kernel from
`solvax.direct`, and multiple right-hand sides share one elimination. The
tridiagonal Legendre structure and its elimination follow Escoto's thesis
(arXiv:2510.27513) and the block-tridiagonal treatment of the monoenergetic
equation by Hirshman, Shaing, van Rij, Beasley & Crume, *Phys. Fluids* **29**,
2951 (1986).

### Truncated storage

The drives of RHSMode 1, 2 and 3 (radial gradients on $L = 0, 2$, the inductive
field on $L = 1$) and every transport moment live on the lowest three Legendre
modes. The truncated kernel therefore runs the forward elimination over all
$N_\xi$ blocks but keeps only the lowest `tier1_keep_lowest` blocks of the
solution (default 3). Its peak memory is

$$
\mathcal{O}\!\left(K\,m^2\right),\qquad K = \texttt{tier1\_keep\_lowest},
$$

independent of $N_\xi$ and $N_x$. At the 744,610-unknown HSX resolution one
$2875^2$ block is about 66 MB, and the truncated route needs about 0.3 GB where a
full-band factorization of the same operator would need about 91 GB. On an RTX
A4000 a 2,525,010-unknown solve peaks at 2.21 GB of device memory where the
full-band charge is about 208 GB (`tools/benchmarks/gpu_anatomy_figure.py`; see
{doc}`compilation_and_parallelism`).

The truncated tail is zero-filled. Moments computed from it are exact, but the
state is not a solution of the full equation, so workflows that audit the
original residual request every block with `SolverOptions(keep_lowest=Nxi)` or
`retain_full_state=True` ({doc}`../user_guide/convergence`). Each $(s,x)$ chain
is eliminated with its own length `Nxi_for_x[ix]`, which is why ramped decks run
on the truncated kernel only: the full-band factorization needs uniform
`Nxi_for_x`.

`subsystem_batch` sets how many of the $N_\mathrm{species} N_x$ chains are
eliminated at once. Any width gives identical per-chain arithmetic;
`"auto"` picks width 1 on CPU and the widest width that fits the memory budget
on accelerators ({doc}`compilation_and_parallelism`).

### One refinement sweep

After the substitution the route performs one defect-correction sweep against
the original operator. A small residual does not by itself bound the forward
error, which is bounded by $\kappa u$; the sweep is what keeps the forward
error near machine precision as conditioning grows. Measured on analytic
one-species decks against a reference refined with a 50-digit residual (dense
operators, conditioning raised by lowering $\nu_n$):

```{list-table}
:header-rows: 1

* - $\mathrm{cond}(A)$
  - sweeps
  - relative residual
  - forward error
* - $3.3\times10^4$
  - 0
  - 6.9e-11
  - 1.7e-11
* - $3.3\times10^4$
  - 1 (default)
  - 7.7e-13
  - 5.4e-15
* - $3.3\times10^6$
  - 0
  - 3.7e-07
  - 1.4e-07
* - $3.3\times10^6$
  - 1 (default)
  - 7.6e-11
  - 2.5e-14
* - $3.3\times10^6$
  - 3
  - 5.2e-11
  - 1.4e-15
```

On a 16,230-unknown deck each sweep costs about 15 % of the solve (0.147 s with
none, 0.170 s with one, median of five warm solves), and the residual saturates
after the first. Ordinary float64 refinement already reaches the 50-digit
reference, so an extended-precision residual has nothing to recover on these
decks. The decks reach $\kappa = 3.3\times10^6$; production pitch-angle decks
have been measured near $\kappa = 3\times10^{12}$, where no dense reference
exists, so the forward error there is not measured.

### Error bounds from the solution

For the monoenergetic transport coefficient the same discrete operator gives a
two-sided bound computed from the solution alone: the variational
entropy-production functionals of `dkx.variational` bracket $D_{11}$ from above
and below ({doc}`../benchmarks/analytic_limits`).

## Recycled Krylov

When the operator is not block tridiagonal in $L$ (full Fokker–Planck or
improved Sugama collisions, full-trajectory $E_r$ terms, magnetic drifts,
$\Phi_1$), the route runs matrix-free GCROT-recycled flexible GMRES on
`KineticOperator.apply`, right-preconditioned by an exact structured solve of a
SFINCS-simplified coarse operator (self-species, speed-diagonal collisions, no
$L\pm2$ terms). The recycle pair $(C, U)$ is returned in the result and can be
passed to the next solve of a neighbouring operator. Restart policy,
preconditioner choices and their measured behaviour are in
{doc}`krylov_and_preconditioners`.

### Stall escalation

Under `method="auto"`, a recycled Krylov solve that misses its tolerance is not
returned as an answer. It escalates at fixed physics, starting each rung from
the best finite iterate reached so far:

1. Other preconditioners of the same simplified operator: first
   `coarse_triangle` (retains the collision operator's upper speed triangle,
   Fortran `preconditioner_x = 2`) when there is a Fokker–Planck or Sugama
   operator, then `sparse`, then `multigrid`.
2. Four times the outer-cycle budget, at the wide restart of the memory-aware
   policy.
3. The sparse direct route, if the system has at most `max_dense_size`
   unknowns (default 8,192) or MUMPS was selected with a memory budget.

If every rung fails, the solve raises with the attempts, the best residual, and
the reminder that non-convergence alone does not identify under-resolution, ill
conditioning or a preconditioner defect.

## Sparse direct referee

This route factors the whole operator. It answers decks the structured route
refuses, and because it shares no algorithm with the other two routes it is the
independent cross-check on them. It runs on `method="direct"` or as the last
escalation rung.

1. **Assembly from operator products** (`dkx.assembly`). The operator's
   couplings are known: per row, the angular stencil and $|L'-L| \le 2$ at one
   $(s, x)$, and every $(s', x')$ at one angular point. Columns that share no row
   are grouped, and one application of the matrix-free operator recovers a
   whole group (`solvax.compression`). The 66,004-unknown collaborator grid
   assembles from 5,508 products and the 633,604-unknown HSX-like gap deck
   from 4,800, against one product per column when sampled. The dense
   constraint border is probed separately. The result is checked against the
   operator on random vectors to `1e-10` before it is returned.
2. **Ruiz equilibration** (`solvax.equilibration`). Rows and columns are scaled
   before factorization, so pivots are not chosen from rows mixing streaming,
   collision and constraint scales. On the collaborator grid, PARDISO without the
   scaling perturbed nine pivots and returned a relative residual of `9.4e-2`;
   with it, `1.2e-10` after refinement.
3. **Factorization and one defect correction.** SuperLU is the default. The
   solution is corrected once against the original operator, and each
   right-hand-side column is accepted against its own norm,
   `max(atol, tol * ||b_j||)`.

The collaborator grid solves to a relative residual of `1.3e-14` in 846 s. Cost
grows as the 2.8 power of the unknowns in time and the 1.6 power in memory, so
the route reaches a few hundred thousand unknowns on a 62 GiB host.

**MUMPS backend.** `SolverOptions(method="direct", direct_backend="mumps", memory_budget_gb=...)`
selects MUMPS. It needs SOLVAX 0.25.0 or later and PyMUMPS; with SOLVAX 0.24
it raises an `ImportError` naming what is missing. The budget is checked after
assembly and before factorization against measured process memory, the
assembled storage, the right-hand-side buffers and the host's available memory,
and again before stored factors serve a wider right-hand side. The admission
reserves seven full right-hand-side buffers; it is a conservative estimate, not
a hard RSS bound. A MUMPS request the budget refuses fails loudly and never
falls back to SuperLU.

**Differentiation.** The route runs on the host and refuses `jax.grad` and
tracing. It has an explicit adjoint instead: `solve(..., factors=..., transpose=True)`
solves $A^\mathsf{T}u = b$ from the stored factors ({doc}`factor_reuse`).

## Precision

Every route runs in float64. `dkx.runtime.configure()` enables JAX x64 before
the first array exists, and `dkx.require_float64()` refuses to solve if a caller
disabled it. The one reduced-precision option is inside the Krylov
preconditioner (`DKX_COARSE_FACTOR_DTYPE=float32`), where it changes iteration
counts and never the answer; see {doc}`krylov_and_preconditioners` and
{doc}`../design_decisions`.

## Device placement

`solve(device=...)` and `DKX_SOLVE_DEVICE` can place a solve on the host CPU on
an accelerator host. The size thresholds that route small systems to the CPU,
`DKX_SOLVE_CPU_MAX_SIZE_TIER1` and `DKX_SOLVE_CPU_MAX_SIZE_TIER2`, default to 0
(off): on a 36-core workstation with an RTX A4000 the GPU won every structured
direct warm solve measured down to 6.5k unknowns and every preconditioned
recycled Krylov warm solve down to 2.8k unknowns (comment above
`_SOLVE_CPU_MAX_TIER1_DEFAULT` in `src/dkx/solve.py`). Small $\Phi_1$ Newton
workloads are better run whole-process on CPU with `JAX_PLATFORMS=cpu`.
