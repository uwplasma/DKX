# Krylov solver and preconditioners

Every physics option that makes DKX more than a DKES-type solver (full
Fokker–Planck and improved Sugama collisions, $\Phi_1$ and quasineutrality,
magnetic drifts, the $E_r$ terms in $\dot x$ and $\dot\xi$) breaks the
block-tridiagonal-in-$L$ structure and runs on the recycled Krylov route. This
page describes that route: the Krylov method, its restart policy, the
preconditioner family, and the measurements that decided each default.

Iteration counts quoted here are deterministic for a given deck, commit and
machine; wall times on shared hosts are indicative and are labelled with their
host. Counts are not comparable across machines: the same deck and settings
gave 1,041 iterations on a laptop and 1,260 on a workstation at different
commits.

## The Krylov method

The route is flexible GMRES with GCROT subspace recycling (`solvax`), applied
matrix-free to `KineticOperator.apply` and right-preconditioned. Its knobs on
`dkx.solve.solve`:

```{list-table}
:header-rows: 1

* - Argument
  - Default
  - Meaning
* - `restart`
  - `None`
  - FGMRES cycle size; `None` selects the memory-aware policy below
* - `recycle_dim`
  - 8
  - recycled directions $k$ carried between cycles and returned for warm starts
* - `max_restarts`
  - 200
  - outer-cycle cap; exceeding it triggers the escalation ladder of {doc}`solver_routes`
* - `preconditioner`
  - `None` (`"coarse"`)
  - `"coarse"`, `"coarse_triangle"`, `"sparse"`, `"multigrid"`, `"none"`
* - `krylov_memory_budget_gb`
  - `None`
  - memory the widened basis may take; else `DKX_KRYLOV_MEMORY_BUDGET_GB`, else a quarter of available memory
* - `recycle`, `x0`
  - `None`
  - recycle pair and initial iterate from another solve
```

Convergence is judged on the true residual $\|b - Au\|$ recomputed from the
operator, per right-hand-side column, against `max(atol, tol * ||b_j||)` with
`tol = 1e-10` by default. The Krylov method's internal estimate is never the
acceptance test.

## Restart length

The restart length decides more of the iteration count than the speed grid
does. On the HSX-like deck (`Nxi = 20`, `coarse` preconditioner, `tol = 1e-10`,
8 recycled directions; Xeon W-2295 host, one core per point, CPU float64;
`tools/benchmarks/restart_and_direct_reach/`), iterations to tolerance were:

```{list-table}
:header-rows: 1

* - `Nx`
  - unknowns
  - restart 200
  - restart 1,000
  - restart 2,000
* - 10
  - 66,004
  - 174
  - 174
  - 174
* - 11
  - 72,604
  - 187
  - 187
  - 187
* - 13
  - 85,804
  - 969
  - 259
  - 259
* - 14
  - 92,404
  - 1,186
  - 384
  - 384
* - 15
  - 99,004
  - 1,970
  - 323
  - 323
* - 16
  - 105,604
  - 2,788
  - 357
  - 357
```

At restart 200 the iteration count grows 16-fold from `Nx = 10` to 16; without
restarting it grows 2.05-fold. The growth is restart stagnation, not a property
of the speed discretization. At twice the pitch resolution,
`(Nxi, Nx) = (40, 16)` and 211,204 unknowns, restart 1,000 converges in 390
iterations against 7,167 at restart 200. Every converged point reached a true
relative residual between 7.6e-11 and 1.0e-10.

`Nx = 12` does not converge at any restart: its true residual sits at 1.07e-10
to 1.14e-10, just above the requested `1e-10`, and each cycle ends when the
GMRES estimate crosses the tolerance while the recomputed residual does not.
Asked for `2e-10` at restart 2,000 it converges in 427 iterations. A tolerance
at a grid's attainable floor makes the solve wander; the escalation message
says so.

A longer restart has two costs, measured at `Nx = 16`:

```{list-table}
:header-rows: 1

* - restart
  - iterations
  - Krylov time
  - peak RSS
  - FGMRES basis
* - 200
  - 2,788
  - 733 s
  - 2.06 GiB
  - 0.33 GiB
* - 1,000
  - 357
  - 216 s
  - 3.11 GiB
  - 1.59 GiB
* - 2,000
  - 357
  - 295 s
  - 4.93 GiB
  - 3.16 GiB
```

Flexible GMRES stores both the Arnoldi basis $V$ and the preconditioned basis
$Z$, so the basis takes $2 \times \texttt{restart} \times N \times 8$ bytes. And
SOLVAX orthogonalizes each step against the whole allocated basis, so an unused
restart still costs time: the same 174 iterations at `Nx = 10` took 42 s at
restart 200, 84 s at 1,000 and 111 s at 2,000.

### The memory-aware default

`restart=None` follows from both measurements. A host-controlled `method="auto"`
solve runs five cycles of 30, then two cycles of 100. A deck that converges
within those 350 steps never allocates a wide basis. A deck that has not
converged continues from its iterate and recycled subspace at the longest
restart, at most 1,000 and at most $N$, whose basis fits
`krylov_memory_budget_gb`. The total stays within `30 * max_restarts` inner
steps, and the escalation ladder's larger-budget rung uses the same wide
restart. An integer `restart` fixes the cycle size; explicit methods,
differentiable and traced solves, and caller-supplied preconditioners always
use a fixed size.

Two public reduced decks from `tests/reduced_inputs/` at refined grids, with
the fixed policy (five cycles of 30, then cycles of 100) against the
memory-aware one, on a 14-core laptop at load average 74 to 157 with one BLAS
and XLA thread per run (iteration counts are unaffected by load; wall times are
indicative):

```{list-table}
:header-rows: 1

* - deck
  - unknowns
  - fixed cycles of 100
  - memory-aware
* - HSX FP full trajectories, `Nx = 16`, `Nxi = 20`
  - 40,324
  - 222 iterations, Krylov 21 s, 0.84 GiB
  - 222 iterations, Krylov 20 s, 0.57 GiB
* - tokamak FP with `Er`, `Nx = 16`, `Nxi = 40`
  - 58,242
  - 545 iterations, Krylov 72 s, 0.62 GiB
  - 460 iterations, Krylov 72 s, 1.37 GiB
* - tokamak FP with `Er`, `Nx = 24`, `Nxi = 40`
  - 87,362
  - no answer: 5,950 iterations, every escalation rung failed, best residual 5.8e-3, 1,459 s, 2.80 GiB
  - 1,736 iterations, Krylov 276 s, 1.85 GiB
```

The first deck converges inside the short cycles either way. The second pays
0.75 GiB for a basis it fills only in part. The third is the case the policy
exists for.

## The coarse preconditioner

The default preconditioner (`dkx.coarse_precond.build_coarse_preconditioner`)
inverts a SFINCS-simplified operator exactly. The simplification is the Fortran
`preconditionerOptions` idiom: `preconditioner_species = 1` (self-collisions
only) and `preconditioner_x = 1` (speed-diagonal collisions) reduce the
collision operator to an $L$-diagonal coefficient, and the $E_r$ $L\pm2$ terms
are dropped. What remains is block tridiagonal in $L$, so the preconditioner is
itself a structured direct solve, batched over $(s, x)$ with dense
$N_\theta N_\zeta$ blocks. The bordered constraint and $\Phi_1$ rows are
eliminated exactly by a bordered Schur complement. On the production
pitch-angle-scattering $\Phi_1$ case this took the Newton inner solve from 9,198
unpreconditioned iterations (about 398 s) to 5 (about 13.5 s), with answers
identical to machine precision.

The magnetic drifts' $L$-diagonal half is carried in the coarse operator, as
Fortran does with `preconditioner_magnetic_drifts_max_L`. On the two W7-X
magnetic-drift decks (`filteredW7XNetCDF_2species_magneticDrifts_noEr` and
`_withEr`, 36-core 62 GB host, float32 reusable factors) this cut iterations
from 1,260 to 123 and from 3,384 to 120, and wall time from 10 h 08 min to
1 h 56 min and from 26 h 07 min to 2 h 17 min, at unchanged residuals and
0.3–0.5 GB of extra storage. The gain grows with pitch resolution: 1.7× on a
tiny fixture at `Nxi = 6`, 2.7× at `Nxi = 40`, 10–28× at `Nxi = 100`.

### The $L = 0$ pin

The simplified $L = 0$ block annihilates a distribution constant over the flux
surface, so its null vector is removed by a rank-one pin. Sizing the pin by the
mean diagonal over all $L$, which the $\nu L(L+1)/2$ collision diagonal makes
about $10^3$ times larger than the $L = 0$ block, lets the pin dominate the block
it regularizes. The pin is therefore adaptive
(`dkx.coarse_precond._l0_pin_gamma`): sized by the `1e-8` invertibility floor
and applied only where the block's own $L = 0$ diagonal does not clear it. On an
NCSX geometry at `11 x 21 x 41 x 5` (`tools/benchmarks/tier2_multigrid_ladder.py`,
Apple M4):

```{list-table}
:header-rows: 1

* - deck
  - adaptive pin
  - unconditional pin
* - full Fokker–Planck with `Er`
  - 21
  - 87
* - improved Sugama with `Er`
  - 20
  - 84
* - pitch-angle scattering
  - 7
  - 7
* - `Er = 0`
  - 18
  - 18
```

A uniform diagonal shift is worse: 60 iterations at `1e-2` of the mean collision
diagonal and no convergence at `1.0`. The pin cannot be dropped: a
collisionless, drift-free $f$-block has an exactly zero $L = 0$ diagonal.

### Storage policies

The dense coarse preconditioner stores three $m\times m$ blocks per
$(s, x, L)$, $m = N_\theta N_\zeta$: $\mathcal{O}(N_\xi N_s N_x m^2)$ memory and
$\mathcal{O}(N_\xi N_s N_x m^3)$ work. At `21 x 41` angles the bands alone are
about 10 GB. The route therefore chooses between three storage policies by
measured size, and warns with both sizes at each transition:

```{list-table}
:header-rows: 1

* - Policy
  - Stored per subsystem
  - Cost model
  - Selected when
* - Dense bands
  - Schur LU plus both off-diagonal bands
  - factor once, cheapest application
  - bands fit physical RAM (`_coarse_bands_fit`)
* - Reusable Schur LU (`store_offdiagonals=False`)
  - Schur LU only; off-diagonal blocks regenerated during each substitution
  - factor once; $1/3 + 1/(6m)$ of the bands, a sixth with float32 factors
  - the factors fit (`_coarse_factors_fit`)
* - Checkpointed (`block_thomas_checkpointed_fn`)
  - one Schur checkpoint per $\lceil\sqrt{N_\xi}\rceil$ rows
  - repeats the whole elimination on every application
  - nothing else fits
```

The off-diagonal bands can be dropped because each is one shared streaming
matrix scaled by a Legendre coefficient plus a diagonal, which a generator
rebuilds exactly. On `geometryScheme4_2species_withEr_fullTrajectories`
(`Nxi = 48`, $m = 247$, 10-core Apple M4, `tools/benchmarks/tier2_generated_coarse.py`)
the dense route costs 3.3 s to build and apply once and 0.047 s per warm
application; the checkpointed route costs 32.9 s and 1.46 s. The checkpointed
route is a memory fallback, not a speedup. The routes agree to `5e-14` forward
and `9e-14` transposed.

`DKX_TIER2_MEMORY_GUARD=off` forces the dense route; `DKX_COARSE_FACTOR_DTYPE=float32`
halves the Schur LU.

### Float32 factors

Taking the Schur factors to float32 changes the preconditioner, never the
answer, and its cost depends on how exact the preconditioner already was. Where
the simplified operator is far from the full one, it is nearly free. GCROT
iterations to `1e-10` (`tools/benchmarks/tier2_coarse_truncation.py`):

```{list-table}
:header-rows: 1

* - deck
  - float64
  - float32
* - `geometryScheme4_2species_noEr`
  - 26
  - 26
* - `geometryScheme4_2species_withEr_fullTrajectories`
  - 29
  - 29
* - `filteredW7XNetCDF_2species_noEr`
  - 20
  - 22
* - `tokamak_2species_PASCollisions_withEr_fullTrajectories`
  - 19
  - 24
```

Where the preconditioner is nearly exact, the factor error becomes the dominant
error and the switch is expensive. On analytic decks at 9x9 angles, `Nxi = 16`,
`Nx = 5`, dense bands: pitch-angle scattering ($\|A-M\|/\|A\| = 0$) goes from 3 to
228 iterations, full Fokker–Planck ($\|A-M\|/\|A\| = 0.003$) from 12 to 474. Both
still reach `1e-10`. Float64 stays the default.

### Why the coarse chain is not truncated

The structured direct route saves memory by keeping only the lowest
Legendre blocks of its solution. Doing the same to the coarse factorization
would store $\mathcal{O}(K m^2)$ and be reused on every application. It does not
work, because the two truncations are different operations: the structured
route sweeps every block and truncates the retained solution, while factoring
only the leading $K$ blocks severs the $L\pm1$ streaming coupling at $L = K$, and
in the coarse operator that coupling is the leading term. On
`geometryScheme4_2species_noEr` (`Nxi = 48`), with every block above $K$ inverted
exactly:

```{list-table}
:header-rows: 1

* - blocks kept
  - GCROT iterations to `1e-10`
  - residual reached
* - 48 (whole chain)
  - 26
  - 4.9e-11
* - 47
  - 133
  - 7.9e-11
* - 44
  - no convergence in 300
  - 1.7e-06
* - 24
  - no convergence in 300
  - 4.0e-01
* - 3
  - no convergence in 300
  - 9.8e-01
```

Cutting one link of 48 costs five times the iterations; cutting four ends
convergence. Dropping the $L\pm1$ coupling everywhere
(`drop_l_coupling_in_precond=True`) is the limit of the same family and does not
converge either. This option is not Fortran's `preconditioner_xi`, which drops
the $L\pm2$ terms (the default coarse operator already omits those). The coarse
operator is cheap to simplify and expensive to shorten.

## The speed triangle: `coarse_triangle`

`preconditioner_x = 1` discards the collision operator's speed coupling. The
Fokker–Planck collision matrix is upper triangular in speed apart from its
diagonal, so the coupling can be retained exactly (Fortran
`preconditioner_x = 2`) by back-substitution over $x$, reusing the `coarse`
factors with no new factorization. On NCSX `(25, 37, 61, 8)` (Xeon W-2295, four
pinned cores) this route cuts GCROT iterations 2.3× (64 to 28), but each
application costs 2.3× a `coarse` application and the solve peaks at 17.4 GiB
against 10.9 GiB, so it loses end to end (64.8 s against 45.7 s).

It pays where the default route's iterations run away. On the HSX-like deck at
`Nxi = 120`, restart 100:

```{list-table}
:header-rows: 1

* - `Nx`
  - route
  - iterations
  - wall
  - peak RSS
* - 10
  - `coarse`
  - 150
  - 53.7 s
  - 4.32 GiB
* - 10
  - `coarse_triangle`
  - 85
  - 146.3 s
  - 7.06 GiB
* - 16
  - `coarse`
  - 6,697
  - 2,519 s
  - 6.0 GiB
* - 16
  - `coarse_triangle`
  - 2,599
  - 1,358 s
  - 11.09 GiB
```

Both routes give the same `FSABjHat` to 1.6e-10. `coarse` stays the default,
and `coarse_triangle` is the first escalation rung after a stall on decks with a
Fokker–Planck or Sugama operator. Choosing it from grid parameters alone is not
supported: what crosses is the default route's iteration count.

## A fill-reducing elimination: `sparse`

The blocks the coarse route factors are sparse in the operator: on a
`19 x 59` surface about 9 of 1,121 entries per row are nonzero. Eliminating $L$
first fills them in, since the Schur complement
$D_L - L_L D_{L-1}^{-1} U_{L-1}$ is dense. Fortran assembles the same simplified
operator as one sparse PETSc matrix and lets MUMPS choose the ordering.
`preconditioner="sparse"` (`dkx.sparse_precond`) does the same: CSR assembly from
the `legendre_blocks` coefficients, host SuperLU per $(s,x)$ subsystem, and a
Sherman–Morrison correction that keeps the dense rank-one $L = 0$ pin out of the
sparsity pattern. It is the same linear map as `coarse` to factorization
round-off (`tests/test_sparse_precond.py`). Stored size
(`tools/benchmarks/tier2_sparse_fill.py`, structural, machine-independent):

```{list-table}
:header-rows: 1

* - deck
  - $N_\theta N_\zeta$
  - classical bands
  - SuperLU factors
* - `tokamak_2species_PAS_withEr_fullTrajectories`
  - 21
  - 0.01 GB
  - 0.003 GB
* - `geometryScheme4_2species_withEr_fullTrajectories`
  - 247
  - 0.65 GB
  - 0.09 GB
* - `sfincsPaperFigure3_geometryScheme11_PAS_2Species_fullTrajectories`
  - 1,121
  - 16.85 GB
  - 1.57 GB
* - `filteredW7XNetCDF_2species_magneticDrifts_withEr`
  - 1,265
  - 42.92 GB
  - 5.97 GB
```

Wall time is not measured for this route, and each Krylov iteration pays a host
callback, so it is opt-in. It cannot run with traced operator leaves. A
preconditioner is never differentiated (the implicit adjoint differentiates the
solution), which is why a host callback is admissible here and not on the solve
path.

## Why multigrid does not replace the factorization

`preconditioner="multigrid"` (`dkx.multigrid`) keeps the same simplified operator
and bordered elimination and replaces the inner inverse with a semicoarsened
geometric multigrid V-cycle over $(\theta, \zeta[, \xi])$, with the block-Thomas
solve on the coarsest grid. It is linear in grid size. Measured on NCSX,
`collisionOperator = 0` with the $\dot x$ and $\dot\xi$ terms on,
`solverTolerance = 1e-8`, GCROT `m = 30, k = 8` capped at 600 iterations (Apple
M4, `tools/benchmarks/tier2_multigrid_ladder.py`):

```{list-table}
:header-rows: 1

* - grid ($N_\theta\times N_\zeta\times N_\xi\times N_x$)
  - unknowns
  - route
  - iterations
  - build + solve
  - peak RSS
  - final residual
* - 11 x 21 x 41 x 5
  - 47,357
  - coarse
  - 21
  - 4.3 s
  - 2.7 GB
  - 1.7e-11
* - 11 x 21 x 41 x 5
  - 47,357
  - multigrid
  - 600 (cap)
  - 23.0 s
  - 1.6 GB
  - 2.2e-03
* - 21 x 41 x 81 x 7
  - 488,194
  - coarse
  - killed after 40 min
  - —
  - 11.6 GB
  - —
* - 21 x 41 x 81 x 7
  - 488,194
  - multigrid
  - 600 (cap)
  - 186 s
  - 5.9 GB
  - 4.8e-03
```

Multigrid makes the large case runnable and does not reach tolerance. The cause
is structural and pinned by `tests/test_multigrid.py`: **parallel streaming and
the mirror force are strictly off-diagonal in the Legendre index.** A $\theta$-
or $\zeta$-line relaxation at fixed $L$ never sees the dominant term, and the
$L$-line contains the mirror force, which is nearly skew-symmetric with a
diagonal ($\nu_D L(L+1)/2$) that vanishes with collisionality. On one
$(s, x)$ block of the W7-X standard-configuration simplified operator
(`9 x 11 x 13`, alternating exact line solves, $\omega = 1$,
`tools/benchmarks/tier2_pitch_basis_study.py`):

```{list-table}
:header-rows: 1

* - discretization
  - $\rho(S)$ at $\nu_n = 8.3\times10^{-3}$
  - two-grid factor at $\nu_n = 10^{-1}$ / $8.3\times10^{-3}$ / $10^{-4}$
* - Legendre modes, coarsen $(\theta,\zeta)$
  - 5.9e6
  - 1.5e7 / 4.0e13 / 3.3e24
* - pitch grid, first-order upwind
  - 0.97
  - 0.39 / 0.24 / 0.74
* - pitch grid, centered
  - 2.2e2
  - 1.3e2 / 4.7e4 / 5.1e11
```

$\rho(S) > 1$ means the relaxation diverges, and no coarse-grid correction
rescues it. Two further observations close the other doors: the operator's
near-null directions are constant along field lines, which are neither $\theta$
nor $\zeta$ modes, so coarsening either angle changes the discrete near-null
space; and coarsening $N_\xi$ as well makes the two-grid correction worse than
none (error recovered to 1.023), so `coarsen_xi` is `False` by default. A
low-order pitch-grid surrogate inside the preconditioner cannot be both accurate
(18 GMRES iterations, centered, but divergent smoothing) and smoothable (0.24
two-grid factor, but 201 iterations).

What would work is a different discretization, pitch on a grid, which changes
answers at fixed resolution, breaks Fortran matrix parity, and needs the
collision operators re-derived where they are dense. `dkx.collocation` is that
experiment ({doc}`discretization`). The multigrid preconditioner stays opt-in,
for grids where the exact factorization does not fit.

## Decks the dense bands do not fit

With the reusable Schur-LU policy and the magnetic-drift diagonal, the upstream
decks whose coarse bands exceed a 24 GB machine complete on a 36-core, 62 GB
host with float32 reusable factors:

```{list-table}
:header-rows: 1

* - deck
  - iterations
  - wall
  - peak RSS
* - `HSX_PASCollisions_fullTrajectories`
  - 11
  - 50 min
  - 13.57 GB
* - `HSX_FPCollisions_DKESTrajectories`
  - 46
  - 69 min
  - 13.06 GB
* - `HSX_FPCollisions_fullTrajectories`
  - 49
  - 71 min
  - 13.22 GB
* - `filteredW7XNetCDF_2species_magneticDrifts_noEr`
  - 123
  - 1 h 56 min
  - 11.66 GB
* - `filteredW7XNetCDF_2species_magneticDrifts_withEr`
  - 120
  - 2 h 17 min
  - 11.72 GB
```

For comparison, on the `sfincsPaperFigure3` two-species full-trajectory deck the
Fortran reference's main solve takes 47 s against 312 s for recycled Krylov
with the dense coarse preconditioner. Closing that gap is open work; head-to-head
timings are in {doc}`../benchmarks/performance`.

## Warm starts and recycling

The recycle pair and the final iterate of one solve can seed the next through
`recycle=` and `x0=`, and a built preconditioner can be passed with `precond=`.
Neighbouring points of a scan or Newton iteration converge in fewer iterations.
A preconditioner frozen too far from the operator does not: one built at
`Er = 0` left a solve at `Er = 80` unconverged after 6,000 iterations at a
residual of 0.35. Reuse rules and measured limits are in {doc}`factor_reuse`.
