# Keeping the collision triangle in the preconditioner

## Hypothesis

Plan phase 2, step 2; follows `2026-09-07-collision-speed-structure.md`.

The coarse preconditioner keeps only the collision operator's speed diagonal,
and the dropped part is essentially all of `A - M`. Because the operator is
upper triangular in speed, retaining its upper triangle (Fortran
`preconditioner_x = 2`) is very nearly the full coupling and needs no new
factorization: `D^-1 U` is nilpotent of index `Nx`, so `Nx - 1` correction
sweeps over the existing block-Thomas factors give the exact inverse of
`D + U`. If the resulting preconditioner cuts iterations by more than it costs
to apply, it should become the default for full-Fokker-Planck decks.

## Admission test

GCROT iterations and wall time against the current preconditioner on
full-Fokker-Planck decks, original-equation residual as the acceptance. Kill on
less than a 1.5x iteration reduction, or on a net wall-time loss once the apply
is charged.

## Result

Implemented as `build_coarse_preconditioner(..., retain_speed_triangle=...)`
and the `coarse_triangle` Krylov kind, off by default. Correctness is pinned in
`tests/test_speed_triangle_precond.py`: it inverts `D + U` exactly, the
retained block is strictly upper, and it is inert for pitch-angle scattering.
The approximation improves as predicted, `||A - M||/||A||` from 0.2981 to
0.0007 on a 752-unknown deck.

Iterations and wall time, one species, `Ntheta = Nzeta = 7`, `Nxi = 8`,
CPU, compilation warmed before timing:

| `Nx` | unknowns | iterations, diagonal | iterations, exact triangle | iteration ratio | wall-time ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1,178 | 14 | 8 | 1.75 | 0.50 |
| 5 | 1,962 | 23 | 12 | 1.92 | 0.51 |
| 8 | 3,138 | 45 | 18 | 2.50 | 0.31 |
| 10 | 3,922 | 54 | 25 | 2.16 | 0.25 |

**The iteration half passes and the wall-time half fails.** Iterations fall by
1.75 to 2.5 times, comfortably past the threshold, and the exact triangle is
two to four times slower overall, worsening with `Nx`.

The reason is arithmetic, not implementation. An exact triangular solve needs
`Nx` sequential steps however it is arranged. Back-substitution does the same
total subsystem work as the default but serializes it into `Nx` batches of
`n_species` instead of one batch of `n_species * Nx`; the nilpotent series keeps
every step batched but performs `Nx` times the subsystem solves. Both were
implemented and measured within a few per cent of each other, which is what
that equivalence predicts.

Truncating the series is the only variant that is not clearly a loss:

| `Nx` | sweeps | iterations | wall-time ratio |
| ---: | ---: | ---: | ---: |
| 5 | 0 (current) | 23 | 1.00 |
| 5 | 1 | 19 | 0.92 |
| 5 | 2 | 17 | 1.08 |
| 5 | 4 (exact) | 12 | 0.82 |
| 10 | 0 (current) | 54 | 1.00 |
| 10 | 1 | 49 | 1.34 |
| 10 | 2 | 40 | 1.10 |
| 10 | 9 (exact) | 25 | 0.54 |

## Decision

**Stop the exact triangle. It is killed by its own admission test on CPU.**

The truncated variant is a lead, not a result. Those wall times are 0.3 to 0.7
seconds, single-shot, on a laptop under other load, which is not a measurement
this plan accepts: the benchmark rule asks for a median of at least five
repetitions on an idle machine, alternating, with hardware and versions
reported. A 1.34 that comes from one pair of sub-second runs is inside the
noise it would take to measure properly.

The code stays, off by default, because it is what the remaining measurement
needs and it is tested and correct. It is not adopted, and nothing selects it
automatically.

What would reopen the exact form: a device where the sequential depth is cheap
relative to the solve, which is the GPU question this could not answer because
the accelerator was unreachable. The batching penalty measured here is a CPU
penalty with one species, where each sequential step carries a batch of one.
Decks with several species, or a device with more parallelism per step, change
that arithmetic and are worth one bounded retest.
