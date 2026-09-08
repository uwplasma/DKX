# What actually pays across a field sweep

## Hypothesis

Plan phase 2, step 3.

Recycled harmonic Ritz vectors approximate invariant subspaces of the
*preconditioned* operator, so rebuilding the preconditioner at every point of a
sweep should invalidate the recycle space and waste the orthogonalizations that
maintain it. If so, holding the preconditioner fixed over a window and
recycling against it should cut iterations materially, and that would explain
why the recycled Krylov route wins only 7 of the 23 decks it owns.

## Admission test

A ten-point `E_r` sweep of one full-Fokker-Planck deck under four ablations:
preconditioner rebuilt or held fixed, crossed with recycling on or off. Total
Krylov iterations and wall time, with the original-equation residual checked at
every point. Kill on a fixed-preconditioner recycling gain below 1.5x in
iterations.

## Result

One species, `Ntheta = Nzeta = 7`, `Nxi = 8`, `Nx = 5`, 1,962 unknowns,
`E_r` from 0 to 9 in ten steps, CPU, compilation warmed. Every ablation met the
1e-10 original-equation tolerance at every point.

| Preconditioner | Recycle | Total iterations | Wall time (s) | vs baseline |
| --- | --- | ---: | ---: | ---: |
| rebuilt per point | off | 206 | 4.33 | 1.00x |
| rebuilt per point | on | 161 | 4.48 | 1.28x fewer iterations |
| held fixed | off | 225 | 1.80 | 0.92x |
| held fixed | on | 174 | 2.16 | 1.18x fewer iterations |

**Recycling is not the lever.** It reduces iterations by 1.28x when the
preconditioner is rebuilt and 1.18x when it is fixed, both under the 1.5x
threshold, and it does not reduce wall time in either case: 4.33 to 4.48
seconds, and 1.80 to 2.16. The orthogonalizations cost more than the iterations
they save at this size.

**Holding the preconditioner fixed is the lever, and it was not the
hypothesis.** It costs 9 per cent more iterations, 206 to 225, and takes 2.4
times less wall time, 4.33 to 1.80 seconds, because rebuilding the
preconditioner at every point costs more than the extra iterations do.

The iteration counts are deterministic and are the reliable half of this table.
The wall times are single-shot on a laptop under other load; the 2.4x is far
outside plausible noise, the 4.48-against-4.33 comparison is not, and neither is
a measurement under the plan's benchmark protocol.

### How long a fixed preconditioner stays good

Built once at `E_r = 0` and reused across a much wider range than a sweep would
normally span:

| `E_r` | Iterations, rebuilt | Iterations, fixed at 0 | Ratio | Original residual |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 17 | 1.00 | 1.5e-11 |
| 3 | 21 | 22 | 1.05 | 1.6e-11 |
| 9 | 22 | 26 | 1.18 | 8.6e-11 |
| 20 | 24 | 55 | 2.29 | 8.7e-11 |
| 40 | 29 | 420 | 14.5 | 9.9e-11 |
| 80 | 85 | 6000 | 70.6 | **3.5e-01, did not converge** |

Reuse is nearly free out to `E_r` of about 9, degrades steeply past 20, and
fails outright at 80.

**The failure is reported, not silent.** At `E_r = 80` the solve returns
`converged = False` after exhausting its restarts, and the original-equation
residual is 0.35 against a requested 1e-10. A stale preconditioner therefore
costs time and is caught, rather than returning a plausible wrong answer.

## Decision

**Stop the recycling-discipline experiment; it is killed by its own criterion.**
Nothing here says the recycle space is harmful, only that at these sizes it does
not earn its orthogonalizations, and it is not why the Krylov route loses.

**Open preconditioner reuse instead, as a refresh policy rather than a
constant.** The ablation found a 2.4x wall-time factor that the hypothesis did
not predict, and the range scan gives the shape of its validity: cheap while the
operator is nearby, sharply worse once it is not. That is exactly the economics
the plan's reuse contract already specifies, comparing rebuild cost against the
extra iterations over the remaining horizon, and these numbers are the first
calibration of it. A fixed distance threshold in `E_r` would be the wrong
mechanism, because the useful window depends on the deck; the diagnostic to
track is the iteration count itself, refreshing when it climbs.

What this does not establish: a single deck, one sweep direction, CPU only,
and no multi-species case, where the preconditioner is larger and rebuilding it
costs more, which should shift the balance further toward reuse.
