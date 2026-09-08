# Does the exact route need an extended-precision residual?

## Hypothesis

Plan phase 2, step 4.

A small residual does not bound the answer: the forward error of a linear solve
is bounded by `kappa * u`, and the pitch-angle decks were measured at `kappa`
near 3e12, which would put that bound at 3e-4. If the shipped solutions sit
anywhere near it, the 1e-10-level agreement claimed against SFINCS and DKES is
luck rather than accuracy, and refinement with a compensated double-double
residual would be needed to recover full double precision.

## Admission test

Measure the actual forward error of the structured direct route against a
reference obtained by refining with a residual computed in 50 decimal digits,
across a range of conditioning, and see whether it tracks `kappa * u`. Then
measure what the refinement already in the route is worth, and what more of it
costs.

## Result

Analytic decks, one species, dense operators small enough to reference exactly.
Conditioning is raised by lowering the normalized collisionality.

**The forward error does not track the bound.** It stays near machine precision
while `kappa` grows a thousandfold:

| `nu_n` | `cond(A)` | residual | forward error | `kappa * u` |
| --- | ---: | ---: | ---: | ---: |
| 8.5e-3 | 3.3e3 | 6.2e-14 | 3.6e-15 | 7.3e-13 |
| 8.5e-4 | 3.3e4 | 7.7e-13 | 5.4e-15 | 7.3e-12 |
| 8.5e-5 | 3.3e5 | 6.4e-12 | 3.0e-15 | 7.3e-11 |
| 8.5e-6 | 3.3e6 | 7.6e-11 | 2.5e-14 | 7.3e-10 |

**The reason is the refinement sweep the route already performs**, and its value
is large:

| `cond(A)` | sweeps | residual | forward error |
| ---: | ---: | ---: | ---: |
| 3.3e4 | 0 | 6.9e-11 | 1.7e-11 |
| 3.3e4 | 1 (current) | 7.7e-13 | 5.4e-15 |
| 3.3e4 | 3 | 7.2e-13 | 4.2e-15 |
| 3.3e6 | 0 | 3.7e-07 | **1.4e-07** |
| 3.3e6 | 1 (current) | 7.6e-11 | 2.5e-14 |
| 3.3e6 | 3 | 5.2e-11 | 1.4e-15 |

Without it the forward error at `kappa` of 3.3e6 is 1.4e-7, which would indeed
make a 1e-10 claim meaningless. With one sweep it is 2.5e-14.

**Extended precision is not what does the work.** Those sweeps are ordinary
float64, and the reference they are compared against was refined with a
50-digit residual. Float64 refinement reaches that reference to 2.5e-14 after
one sweep and 1.4e-15 after three, so there is no headroom for a double-double
residual to recover.

Cost, on a 16,230-unknown deck, median of five warm solves:

| sweeps | wall time (s) | residual |
| ---: | ---: | ---: |
| 0 | 0.1472 | 6.0e-11 |
| 1 (current) | 0.1698 | 8.0e-15 |
| 2 | 0.1966 | 7.8e-15 |
| 3 | 0.2273 | 7.7e-15 |

About 15 per cent of the solve per sweep, and on this deck the residual
saturates after the first.

## Decision

**Drop the double-double proposal.** The premise behind it is sound and the
measurement confirms the danger it names, but the mechanism is already present
and ordinary float64 refinement closes the gap to an extended-precision
reference.

**Keep one sweep as the default.** It is worth four to six orders of magnitude
of forward error for 15 per cent of the solve, and the second and third sweeps
buy little on the decks reachable here.

What is not settled, and is the one thing worth revisiting: every deck measured
here reached `kappa` of 3.3e6, three orders below the 3e12 recorded for
production pitch-angle decks, because the reference needs a dense operator. The
trend across the sweeps table is that a single sweep leaves roughly `1e-20 *
kappa` of forward error; extrapolated to 3e12 that is about 1e-8, which would
matter for a 1e-10 claim. That extrapolation is not evidence. Measuring it needs
a reference at production size, which means an iterative extended-precision
residual against the matrix-free operator rather than a dense factorization, and
is the shape of the follow-up if the observable error bars from phase 1 ever
disagree with the residual.
