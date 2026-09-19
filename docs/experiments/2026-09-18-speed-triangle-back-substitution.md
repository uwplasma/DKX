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

**SOLVE_TABLE_PLACEHOLDER**

**AGREEMENT_PLACEHOLDER**

Scripts, records and logs are kept outside Git in
`dkx-review-evidence-20260913/q10_triangle/`: `triangle_apply_bench.py`,
`triangle_solve_bench.py`, `triangle_agreement.py`, `run_bench.sh`,
`run_solves.sh`, `out/*.json`, `runs/*/record.json`, `bench.log`, `solves.log`.

## Decision

**DECISION_PLACEHOLDER**
