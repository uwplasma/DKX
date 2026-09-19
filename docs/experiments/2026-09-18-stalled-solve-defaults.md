# What a stalled solve does by default

## Hypothesis

`method="auto"` escalates when the recycled-Krylov solve breaches its iteration
cap, so what the ladder tries is what a user gets without passing anything. Two
defaults looked wrong against what this campaign measured.

- **The rungs could not answer the diagnosed stall.** `coarse`, `sparse` and
  `multigrid` are three inverses of one simplified operator. They differ in fill
  and cost, not in what they approximate, so a stall caused by the
  Fokker-Planck speed coupling that operator drops
  (`2026-09-14-ntheta-iteration-growth.md`) cannot be answered by swapping among
  them. Retaining that coupling's upper triangle is the only rung that changes
  the operator being inverted, and
  `2026-09-18-speed-triangle-back-substitution.md` made its application 3.7×
  cheaper.
- **Each rung restarted from the original guess.** The stalled solve's iterate
  was discarded, so a deck that stalls pays for that work twice.

Owner: independent review. Budget: one office hour.

## Admission test

NCSX `(13, 37, 61, 8)`, office, four pinned cores, one BLAS thread, JAX 0.10.2,
x64, DKX `55584961` against `ffe93dab`. The Krylov route is forced with a 1 GB
budget and the cap set below what the default preconditioner needs
(`restart = 10`, `max_restarts = 3`, so 30 iterations against the 53 it takes),
which is the stall this ladder exists for. Arms alternate, three runs each, on
the same host and settings.

**Bars.** The rung order is admitted if the ladder converges; the warm start is
admitted only if it reduces the iterations of the rung that converges and
returns the same answer. A changed answer rejects it outright.

## Result

| Arm | Iterations in the rung that converged | Wall | `FSABjHat` |
| --- | ---: | ---: | ---: |
| Discarding the stalled iterate | 27, 27, 27 | 70.0 s, 43.7 s, 43.8 s | −6.878756983745e-2 |
| Continuing from it | 18, 18, 18 | 38.7 s, 38.3 s, 38.7 s | −6.878756979405e-2 |

- **A third fewer iterations**, identical across repetitions.
- **Wall time falls 12%**, from 43.8 s to 38.7 s at the steady state. Setup and
  compilation are about 25 s of each run, so the solve itself falls by more. The
  first cold run took 70.0 s on a contended host and is reported rather than
  dropped.
- **The answer is unchanged** to 6.3e-10 relative, with both arms converging to a
  6e-13 residual against a `1e-10` tolerance.
- Both arms converge at the first rung, so this deck exercises the rung order
  but does not separate the rungs from each other.

Scripts and records are kept outside Git in
`dkx-review-evidence-20260913/q10_triangle/`: `escalation_bench.py`,
`run_escalation_ab.sh`, `esc/*/record.json`, `escalation_ab*.log`.

## Decision

**Adopt both.** They change what happens after a stall, not what a converged
solve computes.

- The ladder tries the retained speed triangle first on decks that have a dense
  collision operator, and skips it where there is none to retain, since there it
  is the stalled preconditioner under another name.
- Every rung continues from the stalled iterate when it is finite and its
  residual improved on the right-hand side. A diverged iterate is refused, so a
  rung never starts further away than the original guess.
- Neither is a new knob. A user reaches both through `method="auto"`, the
  default.
- Not measured here: how the rungs compare on a deck where the first one fails.
  The HSX-like deck at `Nx = 16` is the case for that, and it is queued.
