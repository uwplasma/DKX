# The speed triangle by back-substitution over `x`

## Hypothesis

Follow-up to `2026-09-14-ntheta-iteration-growth.md`, queued in the #230 handoff
behind Q5. That record found the cause of GCROT iteration growth with `Ntheta`
and a route that removes it: retaining the collision operator's upper speed
triangle (`preconditioner="coarse_triangle"`) cuts NCSX iterations 4.1× at
`Ntheta = 25`. It failed its adoption bar on cost alone, at about 11× a `coarse`
apply per iteration.

The cost has a mechanical cause. `M = D + U` is speed-diagonal plus a strictly
upper speed coupling, and `D^-1 U` is nilpotent of index `n_x`, so the shipped
apply reached `M^-1` as a series of `n_x` sweeps of the whole band. Each sweep
reads every stored factor. Back-substitution over `x` reaches the same inverse
by solving one speed at a time, reading each speed's factors once, which is one
sweep's worth of work in total.

Owner: independent review. Budget: one office afternoon. Expected: an apply at
most twice a `coarse` apply, and a solve that beats `coarse` end to end.

## Admission test

**Setup.** DKX `e9cce648`, SOLVAX 0.22.0, JAX 0.10.2, x64, CPU. Office Xeon
W-2295, four pinned cores (`DKX_CORES=4`), one BLAS thread, memory and time
guards. NCSX `(Ntheta, 37, 61, 8)`, one species, at `Ntheta = 13` and `25`.

**Routes.** `coarse` is the default preconditioner. `series` is the shipped
triangle, reachable as `retain_speed_triangle = n_x - 1`. `back` is
`retain_speed_triangle=True`, the back-substitution.

**Measurement.** One process per route, so a peak RSS belongs to that route
alone. Median of five applications after a warm-up that pays factorization and
compilation. The bordered operator applies the coarse inverse more than once per
preconditioner application; that is common to all three routes.

**Bars, set in the `Ntheta` record before these runs.**
- The triangle apply costs at most 2× a `coarse` apply at `(25, 37, 61, 8)`.
- The two triangle routes invert the same operator, so their maps must agree.
- The solve must beat `coarse` end to end, which is what the apply bar was a
  proxy for.

## Result

**One application** (median of five, seconds):

| `Ntheta` | Route | Apply | vs `coarse` | Transposed apply | Peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 13 | `coarse` | 0.339 | 1.00 | 0.290 | 3.85 GiB |
| 13 | `back` | 0.791 | 2.33 | 0.673 | 4.57 GiB |
| 13 | `series` | 2.675 | 7.90 | 2.244 | 3.86 GiB |
| 25 | `coarse` | 1.022 | 1.00 | 0.928 | 9.83 GiB |
| 25 | `back` | 2.365 | 2.31 | 2.194 | 9.84 GiB |
| 25 | `series` | 8.735 | 8.54 | 7.938 | 9.91 GiB |

- **The triangle apply costs 3.7× less.** The series reads every factor once per
  sweep; back-substitution reads each speed's factors once.
- **It misses the 2× bar by 15%.** The remaining gap is parallel efficiency, not
  work: XLA runs a batch across its thread pool, and one speed's chains are
  `n_species` of the `n_species * n_x` the default applies together. On this
  single-species deck each per-speed solve is one chain. The `Ntheta` record
  expected at most the core count, 4×, for running the speeds in sequence.
- **Memory is unchanged.** Both triangle routes reuse the `coarse` factors and
  add no factorization; the extra 0.7 GiB at `Ntheta = 13` is the accumulated
  coupling and the assembled columns, which is the same absolute size at 25.

**One NCSX solve** at `(25, 37, 61, 8)`, GMRES `tol = 1e-10`, restart 100:

| Route | Iterations | Setup | Solve | Per iteration | Peak RSS | `FSABjHat` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse` | 64 | 23.3 s | 45.7 s | 0.71 s | 10.86 GiB | −6.8586813823e-02 |
| `coarse_triangle` | 28 | 24.8 s | 64.8 s | 2.31 s | 17.37 GiB | −6.8586813820e-02 |

- **The triangle still loses end to end.** It cuts iterations 2.3×, and each one
  costs 3.3× more, so the solve takes 64.8 s against 45.7 s.
- **Host spread.** A repeat pair on the same host and settings gave 37.5 s for
  `coarse` (0.59 s per iteration) and 67.9 s for the triangle (2.42 s), so these
  are 20% apart run to run. The ordering does not change.
- **Memory is the other cost.** The triangle solve peaks 60% above `coarse`, at
  17.4 GiB against 10.9 GiB, where the two applications alone differ by 0.7 GiB.
- The two routes agree on the current to ten digits.

**Ablation: one compiled program for all speeds.** Tracing every speed into a
single executable, instead of dispatching one program per speed, changed the
`Ntheta = 25` apply by 2% (2.374 s against 2.365 s) and the solve by 5%, while
raising the build-and-first-apply from 46.6 s to 60.5 s. Dispatch is therefore
not what the remaining gap is made of, and the change was dropped.

**The two triangle routes are the same map** at `Ntheta = 13`, built on one
operator from one set of factors: relative difference 1.0e-15 forward and
7.0e-15 transposed. The adjoint identity `<M^-1 u, w> = <u, M^-T w>` holds to
6.9e-15 on the deck itself, and the unit tests pin it on a small grid.

Scripts, records and logs are kept outside Git in
`dkx-review-evidence-20260913/q10_triangle/`: `triangle_apply_bench.py`,
`triangle_solve_bench.py`, `triangle_agreement.py`, `run_bench.sh`,
`run_solves.sh`, `out/*.json`, `runs/*/record.json`, `bench.log`, `solves.log`.

## Decision

**Land the back-substitution. Keep `coarse` as the default; the triangle stays
opt-in.**

- **The apply is what was fixed.** 3.7× less at both sizes, at the same map, the
  same factors and the same memory. The module already documented the apply as a
  back-substitution over `x`; the series was the implementation that did not
  match it.
- **Both adoption bars still fail.** The apply is 2.3× a `coarse` apply against a
  bar of 2×, and the solve is 1.4× slower rather than faster. The bar's purpose
  was a triangle that wins end to end on this deck, and it does not.
- **What the remaining gap is.** Parallel efficiency, not work. XLA runs a batch
  across its thread pool; one speed's slice holds `n_species` chains where the
  default holds `n_species * n_x`, and this deck has one species. The `Ntheta`
  record expected at most the core count, 4×, for running the speeds in
  sequence; the measurement is 2.3×. The ablation rules out the per-speed
  program count as the cause.
- **Where the change should pay.** The exact triangle's cost stops growing with
  `Nx`: one pass over the speeds, rather than `n_x` sweeps of the whole band. At
  `Nx = 16` the series costs sixteen sweeps, which is why
  `2026-09-14-hsx-like-resolution-ladder.md` records no result for the
  speed-triangle retry at point A. That deck, not NCSX, is the next test of this
  route.
- A route that keeps exactness while batching several speeds per solve would
  need block back-substitution over speed groups. It is queued behind the
  `Nx = 16` test, which decides whether the triangle is worth more work at all.
