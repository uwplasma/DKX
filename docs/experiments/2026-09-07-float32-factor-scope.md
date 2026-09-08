# Where float32 factors are free, and where they are not

## Hypothesis

Plan phase 2, step 5.

Step 1 confirmed memory as the binding loss: five of the six decks that did not
complete were exhausted in the coarse preconditioner's dense angular bands. The
step proposed halving the factors with `float32`, admissible only after Ruiz
equilibration and only for scaled conditioning well below 1e10.

## What was already true before measuring

Most of step 5 is already implemented and documented, and re-running it would
have been waste. The route choice is automatic: `_coarse_bands_fit` and
`_coarse_factors_fit` send a deck to the dense bands, the reusable Schur LU or
the checkpointed generator by size. `DKX_COARSE_FACTOR_DTYPE=float32` exists and
is opt-in. `docs/performance.rst` records a completed production campaign in
which every deck the reusable route was built for completes, and in which
`float32` factors cost no extra iterations, reached the same residual, and used
less memory and less wall time.

Two parts of the step's premise do not survive contact with the code. There is
no Ruiz equilibration in the solver path at all; it exists only inside
`tools/benchmarks/operator_conditioning.py`, so "fp32 after Ruiz" cannot be done
as written, and the campaign shows `float32` working without it.

## Result

The documented `float32` result does not generalize, and the direction of the
error is the dangerous one: it invites enabling the switch to save memory.
Analytic decks at 9x9 angles, `Nxi = 16`, `Nx = 5`, dense-band route, same
commit and machine:

| Deck | `||A-M||/||A||` | Iterations, float64 | Iterations, float32 | Cost |
| --- | ---: | ---: | ---: | ---: |
| pitch-angle scattering | 0.000 | 3 | 228 | 76x |
| full Fokker-Planck | 0.003 | 12 | 474 | 40x |

A larger low-collisionality deck of 12,105 unknowns showed the same shape: 3
iterations and 0.44 s at float64 against 2,510 iterations and 8.7 s at float32.

Both still reach 1e-10, so the switch is safe rather than wrong. It is simply
not free away from the decks the campaign measured.

The mechanism is that a factor error only disappears into an approximation that
is already larger. Where the coarse preconditioner is nearly exact, the float32
factor error becomes the dominant error in the preconditioner and the Krylov
method pays for it directly. The production decks in the campaign are the
opposite case: the preconditioner is a poor approximation there, iterations are
already in the hundreds or thousands, and a float32 factor adds nothing
measurable on top.

## Decision

**Keep `float32` opt-in and keep float64 the default.** The measurement supports
the existing choice rather than changing it, and the docstring's reason for that
default rests on a measured counterexample rather than on caution alone.

**Scope the claim in the performance page.** "Better on every axis" is true of
the deck it was measured on and false by a factor of 40 to 76 on decks where the
preconditioner fits and is nearly exact. The page carries both tables.

**Drop the Ruiz precondition from the step.** There is no Ruiz equilibration in
the solver, and the campaign shows `float32` working without it. Adding
equilibration is a separate piece of work that should be justified on its own
evidence, not smuggled in as a precondition for a switch that already works.

What remains genuinely open in step 5 is the one thing needing hardware: the
campaign that showed the six decks completing ran on a 36-core, 62 GB machine,
and the step's admission asks for them within the office GPU's 16 GB. That
measurement is unchanged by anything here and still needs the accelerator.
