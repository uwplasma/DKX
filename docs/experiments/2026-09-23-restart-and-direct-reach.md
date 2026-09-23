# The `Nx` wall is a restart artifact; the direct route's reach is still unmeasured

Status: **paused by the owner** partway through. Step 5 (the restart test) has
its admission result on the `Nxi = 20` ladder and at `(Nxi, Nx) = (40, 16)`.
The full gap deck and step 6a (MUMPS analysis only) were not run. What remains
is listed under Follow-up.

## Hypothesis

`plan.md` section 13, block B, step 5, from finding 1 of
`2026-09-22-literature-review.md`. On the HSX-like gap deck, GCROT iterations
grow with the speed resolution `Nx` and not with `Nxi`: 174 at `Nx = 10`
against 2,788 at `Nx = 16`, `Nxi = 20`
(`2026-09-19-nx-drives-the-iteration-growth.md`). DKX restarts GCROT every 200
inner steps with 8 recycled directions, and the ladder is cheap exactly while a
solve fits in one cycle. The hypothesis: the growth is restart stagnation, not
a property of the speed discretization, and a longer restart removes it.

## Admission test

- The ladder of the 2026-09-19 record: `Nxi = 20`, `Nx` from 10 to 16,
  `preconditioner="coarse"`, `tol = 1e-10`, `atol = 0`, `recycle_dim = 8`, at
  GCROT restart 200, 1,000 and 2,000, with `max_restarts` set so the budget is
  at least 20,000 inner steps (100, 20 and 10 cycles). The restart-200 row is
  first run at the recorded 60 cycles, to check that it reproduces.
- `dkx.solve.solve(..., method="iterative")` is called directly, so a rejected
  solve still reports its iterations. The relative residual `||b - A x||/||b||`
  is recomputed from the pinned operator after the solve.
- **Admission:** at restart >= 1,000 the `Nx = 16` point converges in a small
  multiple of the `Nx = 10` count. **Kill:** iterations still grow with `Nx`
  at restart 2,000; then plan step 6.
- Office host (Xeon W-2295, 62 GiB), CPU only, float64, DKX `73436c1e`, JAX
  0.9.2, SOLVAX 0.25.0. Each point is its own process on cores 12-15 with one
  BLAS and XLA thread, an RSS guard and a wall-clock cap; up to three points ran
  at once. The host was shared and heavily loaded throughout (load average 42
  to 103 on 36 threads, swap near full at times), so wall times carry
  contention and iteration counts do not. The load at the start and end of
  each point is in `tools/benchmarks/restart_and_direct_reach/restart_ladder_results.json`.

## Result

**The restart-200 row reproduces.** Every converged point matches the
2026-09-19 record to the iteration: 174, 187, 969, 1,186, 1,970 and 2,788 for
`Nx` = 10, 11, 13, 14, 15 and 16, with the same residuals to three digits.
`Nx = 12` does not converge, as recorded, but its count moves: 3,810 in the
record, 3,499 on the laptop at `73436c1e`, 3,332 on the office host, and 4,753
when given 100 cycles instead of 60. That count is set by the cycle cap, not by
convergence; the section on `Nx = 12` below explains it.

**Iterations to `1e-10`, `Nxi = 20`:**

| `Nx` | unknowns | restart 200 | restart 1,000 | restart 2,000 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 66,004 | 174 | 174 | 174 |
| 11 | 72,604 | 187 | 187 | 187 |
| 12 | 79,204 | 3,332, **not converged** | 1,444, **not converged** | 904, **not converged** |
| 13 | 85,804 | 969 | 259 | 259 |
| 14 | 92,404 | 1,186 | 384 | 384 |
| 15 | 99,004 | 1,970 | 323 | 323 |
| 16 | 105,604 | 2,788 | **357** | **357** |

**Admitted.** At restart 1,000 or 2,000, `Nx = 16` converges in 357
iterations, 2.05 times the `Nx = 10` count, against 16.0 times at restart 200.
Every point from `Nx = 13` up converges inside the first cycle, which is why
the two longer restarts give identical counts: neither restarts at all. The
residual each reached, recomputed from the operator, is 7.6e-11 to 1.0e-10
against the `1e-10` asked for.

**It holds at twice the Legendre resolution.** At `(Nxi, Nx) = (40, 16)`,
211,204 unknowns, restart 1,000 converged in **390** iterations to `9.25e-11`,
against 7,167 at restart 200 in the 2026-09-19 record: 18 times fewer.
`Nxi` again costs almost nothing once the solve is not restarted (357 at
`Nxi = 20`, 390 at `Nxi = 40`).

**Time and memory, `Nx = 16`, `Nxi = 20`** (office, one core per point,
shared host):

| restart | iterations | Krylov time | wall | peak RSS | FGMRES basis | load start/end |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 2,788 | 733 s | 828 s | 2.06 GiB | 0.33 GiB | 95 / 76 |
| 1,000 | 357 | 216 s | 291 s | 3.11 GiB | 1.59 GiB | 67 / 61 |
| 2,000 | 357 | 295 s | 401 s | 4.93 GiB | 3.16 GiB | 100 / 89 |

- **The memory the restart costs is twice the brief's estimate.** SOLVAX's
  flexible GMRES stores the Arnoldi basis `V`, `(m + 1) x n`, and the
  preconditioned basis `Z`, `m x n`, so the basis costs about
  `2 x restart x n x 8` bytes, not `restart x n x 8`. The measured peak RSS
  tracks that column.
- **A longer restart costs time per iteration even when it is not used.**
  SOLVAX's cycle orthogonalizes against the whole zero-padded `(m + 1)`-row
  basis at every inner step, so an iteration costs in proportion to `m`, not
  to the step index. The same 174 iterations at `Nx = 10` took 42 s at restart
  200, 84 s at 1,000 and 111 s at 2,000. Restart 1,000 is therefore faster than
  2,000 wherever both converge in one cycle, and the saving from restarting
  less still dominates at `Nx = 16`: 216 s against 733 s.

**`Nx = 12` is the attainable-accuracy floor, not restarting.** It fails at
every restart, and its count grows with whatever cycle budget it is given
(3,332 at 60 cycles, 4,753 at 100). Its true residual sits at 1.07e-10 to
1.14e-10, just above the `1e-10` asked for. Each cycle ends when GMRES's
residual estimate crosses the tolerance, the recomputed true residual is above
it, and the next cycle starts from there and ends again within a few steps, so
the cap and not convergence decides when it stops. Asked for `2e-10` at restart
2,000, the same point converges in 427 iterations to `1.87e-10`. That fits
the literature review's warning that a tolerance at the attainable floor makes
the solve wander. The floor is particular to this grid: every other `Nx`
reaches below `1e-10`.

## Decision

**Step 5 is admitted: the `Nx` wall is a GCROT restart artifact.** At restart
200, iterations grow 16-fold from `Nx = 10` to 16. Unrestarted, they grow
2.05-fold, and at `Nxi = 40` the 7,167-iteration point takes 390. Plan step 6
(the field of values and outlier count of `A M^-1`) is not needed for the
`Nx` question. The earlier readings of this growth as a speed-coupling,
conditioning or factorization effect are superseded for the iteration count.
The measurements they rest on stand.

**Recommendation, not made here:** a memory-aware restart default for the
recycled Krylov route. Take the largest restart whose FGMRES basis,
`2 x restart x total_size x 8` bytes, fits a stated fraction of the memory
budget, capped near 1,000, since beyond one cycle's need a longer restart only
adds orthogonalization cost. At `Nxi = 20`, `Nx = 16` that is 1.6 GiB for a
3.4x faster solve. This changes `src/dkx/solve.py`, which another workstream
is editing, so it is left to that owner. Two SOLVAX changes would make the
trade cheaper: orthogonalizing only against the filled rows of the basis, so
an unused restart costs nothing, and dropping `Z` when the preconditioner is
fixed, since `coarse` does not vary between steps, which halves the memory.

## Follow-up

Not run when the owner paused the work:

- **The full gap deck, `(Nxi, Nx) = (120, 16)`, 633,604 unknowns.** Not
  attempted. At restart 1,000 its FGMRES basis alone is 9.5 GiB. Extrapolating
  the measured peak RSS (3.1 GiB at 105,604 unknowns, 5.6 GiB at 211,204, both
  at restart 1,000) gives about 16 GiB, at the guard. Restart 800 or a larger
  guard, with the host's free memory checked first, is the next run: if it
  converges, it is the first deck where DKX converges and SFINCS fails by every
  route (`2026-09-19-sfincs-on-the-gap-deck.md`). The point `(40, 16)` at
  restart 2,000 was queued and not started.
- **Step 6a, MUMPS analysis only (`JOB=1`) on the gap deck.** Not run. The
  environment is ready and the script is written
  (`tools/benchmarks/restart_and_direct_reach/mumps_analysis.py`): PyMUMPS
  0.4.0 with MUMPS 5.8.2, METIS and SCOTCH from conda-forge, in a private
  environment on the office host. SOLVAX 0.25.0's MUMPS adapter runs `JOB=1`
  and reads INFOG(16) before it admits a factorization, but it cannot take an
  ordering or stop after analysis, so the script calls it once with a 1 MB
  budget as a cross-check (it must refuse after analysis) and drives PyMUMPS
  directly for the METIS (`ICNTL(7) = 5`), Legendre-pair-major user
  (`ICNTL(7) = 1`, pairs eliminated from the highest `l` down) and AMD
  orderings. INFOG(29) is a factorization output and is not available from
  analysis; the script reports INFOG(3) and INFOG(20) as the factor-size
  estimates instead. It never runs the numerical factorization.
- **The A4000 float64 against float32 dense LU of step 6a is deferred**: both
  GPUs were fully occupied by another job.
- A restart default and the two SOLVAX changes above, by their owners.
