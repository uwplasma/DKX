# Balancing the preconditioned matrix does not balance the solve

## Hypothesis

Phase 2b step 2 of `plan.md`. `2026-09-19-nx-drives-the-iteration-growth.md`
placed the `Nx` growth in the spectrum of `A M⁻¹` and then narrowed it
further: the eigenvalues barely move between `Nx = 10` and `Nx = 16` while the
conditioning of the eigenvector basis goes 1.95e18 to 1.11e22, and LAPACK
balancing — a diagonal similarity — takes those two numbers to 9.82e10 and
1.57e11, flattening the `Nx` dependence from 5,700x to 1.6x.

That measurement is of a matrix. The hypothesis under test is the step it does
not establish: that solving the *balanced system* costs fewer GCROT
iterations. Right-preconditioned GCROT minimizes its residual in the unscaled
norm, so balancing is a change to the solve and not only to the matrix. With
`D = diag(d)` the balancing diagonal, the balanced solve is

    (D⁻¹ A M⁻¹ D) z = D⁻¹ b,   x = M⁻¹ D z

which in a right-preconditioned Krylov method is the operator `v -> D⁻¹ A v`,
the preconditioner `w -> M⁻¹ D w` and the right-hand side `D⁻¹ b`. The vector
that method returns is `x` itself, in the original variables; what changes is
the norm the residual is minimized in.

## Admission test

The cheapest test first, on the small assembled grid `Ntheta = 5`,
`Nzeta = 5`, `Nxi = 8` of the HSX gap deck, where the growth survives at 113
GCROT iterations at `Nx = 10` against 340 at `Nx = 16` and where the exact
balancing diagonal is available from `scipy.linalg.matrix_balance` on the
column-by-column assembly of `A M⁻¹`. Two species, `collisionOperator = 0`,
`Er = 15`, float64, `coarse` preconditioner, `restart = 200`,
`recycle_dim = 8`, `max_restarts = 60`, `tol = 1e-10`, `atol = 0`.

Acceptance is on the **original** equation: the recovered `x` must satisfy
`||b − A x|| <= 1e-10 ||b||`. A balanced solve that stops early in its own
scaled norm is not a result, so the comparison is made at matched unscaled
acceptance as well as at the balanced arm's own.

Kill, from `plan.md`: the eigenvector conditioning falls as measured but the
iteration count does not move by 1.5x. Then the gap is between the balanced
matrix and the balanced solve, and the next measurement is the GCROT residual
curve in both norms rather than another preconditioner. A negative here stops
the step before any matrix-free estimator of `d` is built.

## Result

**The kill fires.** Balancing removes 1.5e10 of eigenvector conditioning and
changes the iteration count to a matched unscaled acceptance by at most 1.05x.

**The setup reproduces the record it builds on.** GCROT with the `coarse`
preconditioner returns 113 iterations at `Nx = 10` and 340 at `Nx = 16`, at
true relative residuals 6.27e-11 and 7.28e-11. The assembled `A M⁻¹` has
eigenvector conditioning 1.95e18 and 1.11e22, and balancing with the
`permute=True` default of `matrix_balance` returns 9.82e10 and 1.57e11 —
the record's numbers to three figures.

**A solve can only use the diagonal part.** A balancing permutation reorders
rows, which is not available matrix-free and not what the step proposes. With
`permute=False` the pure diagonal returns 2.07e11 and 7.49e11: a flattening
of the `Nx` dependence from 5,673x to 3.6x rather than to 1.6x, which is the
price of dropping the permutation and still seven to eleven orders of
conditioning removed. Every solve below uses that diagonal.

**The algebra is verified before the result is read.** Applying the balanced
operator and preconditioner in sequence reproduces `D⁻¹ (A M⁻¹) D` to 1.40e-10
relative at `Nx = 10` and 6.99e-10 at `Nx = 16`, against the unbalanced arm's
own 7.92e-13 and 2.75e-12 agreement with the assembled `A M⁻¹` — the
difference is the scaling's dynamic range acting on the same floor. And the
recovered `x` does satisfy the original equation whenever the scaled tolerance
is tight enough to ask it to: 4.16e-11 and 9.90e-11 relative. The recovery is
correct, and the negative result is not an algebra error.

**The balanced arm stops in the wrong norm.** GCROT, `restart = 200`:

| `Nx` | arm | stop criterion | iterations | converged | `\|\|b − A x\|\|/\|\|b\|\|` |
| ---: | --- | --- | ---: | :---: | ---: |
| 10 | base | unscaled 1e-10 | **113** | yes | 6.27e-11 |
| 10 | balanced | scaled 1e-10 | 206 | yes | 2.74e-9 |
| 10 | balanced | scaled 1e-11 | 290 | yes | 6.86e-10 |
| 10 | balanced | scaled 3e-12 | 2,780 | no | 4.16e-11 |
| 16 | base | unscaled 1e-10 | **340** | yes | 7.28e-11 |
| 16 | balanced | scaled 1e-10 | 544 | yes | 5.08e-9 |
| 16 | balanced | scaled 3e-11 | 397 | yes | 1.67e-9 |
| 16 | balanced | scaled 3e-12 | 5,787 | no | 9.90e-11 |

At its own tolerance the balanced arm costs 1.8x and 1.6x the iterations and
lands 27x and 51x above the bar it is being compared against. Asked for the
bar itself, it costs 25x and 17x. The iteration count is not monotone in the
requested tolerance — `Nx = 10` takes 571 iterations at a scaled 3e-11 and 290
at 1e-11 — which is restarted GCROT with recycling and not a property of the
scaling.

**Neither is the restart the explanation.** Run as one cycle with no restart,
so that both arms use the best iterate their budget allows, and take the first
budget whose recovered `x` clears `1e-10 ||b||` on the original equation:

| `Nx` | base | balanced |
| ---: | ---: | ---: |
| 10 | 115 | 120 |
| 16 | stalls at 1.9e-10 through 260 | 220 |

At `Nx = 10` balancing costs 4% and buys nothing. At `Nx = 16` a single cycle
of the base arm stagnates just above the bar, at 1.9e-10 from budget 205 to
260, where the balanced arm reaches 1.05e-10 at 220 — a lower attainable floor
in one cycle, worth about 2x in residual and nothing in rate, and the restarted
base arm clears the same bar at 340 iterations regardless. No configuration
produces the 1.5x the kill criterion names.

**The residual curves say what balancing actually does.** One cycle at
`Nx = 16`, both norms, relative to each arm's own right-hand side:

| iterations | base, unscaled | base, scaled | balanced, unscaled | balanced, scaled |
| ---: | ---: | ---: | ---: | ---: |
| 113 | 3.28e-2 | 38.4 | 3.30 | 6.01e-2 |
| 160 | 1.31e-4 | 1.84e-1 | 3.91e-3 | 1.54e-5 |
| 240 | 2.23e-10 | 7.64e-7 | 6.32e-11 | 2.12e-6 |

Each arm leads by two to three orders in its own norm and trails by two to
three in the other, and by 240 iterations the two have converged to the same
place. Balancing is doing exactly what it claims — it makes the scaled
residual fall fast — and the scaled residual is not the quantity the solve is
accepted on.

**So the eigenvector conditioning of `A M⁻¹` is not what governs this solve.**
That is the substantive finding. The diagnosis in
`2026-09-19-nx-drives-the-iteration-growth.md` is correct about the matrix and
does not transfer to the solve: 1.5e10 of eigenvector conditioning can be
removed by a diagonal similarity that the solve can use, and the iteration
count to a fixed accuracy in the original equation does not move. The `Nx`
growth survives balancing — 120 to 220 iterations on the balanced arm against
115 to beyond 260 on the base arm, over the same interval where the balanced
conditioning ratio is 3.6x.

**And the balancing diagonal names the wrong rows.** The standing candidate in
`plan.md` and in the record above is the high-speed rows of the light species.
The exact `d` is the opposite on both counts. It is a clean function of the
speed index alone — LAPACK's radix-2 powers, `2⁻¹³` to `2³¹` at `Nx = 16` —
and it decreases monotonically with speed:

| species | `log₂ d` at the lowest speed | at the highest |
| --- | ---: | ---: |
| ions, `Nx = 10` | 23 | −2 |
| electrons, `Nx = 10` | 9 | −6 |
| ions, `Nx = 16` | 31 | −5 |
| electrons, `Nx = 16` | 24 | −8 |

Within one `(species, speed, Legendre)` block, across the 25 angular points,
`log₂ d` spreads 1.48 powers of two on average at `Nx = 10` and 2.27 at
`Nx = 16`, at most 5 and 7; across Legendre index at fixed speed it moves by
two to three. The `Nxi_for_x`-truncated degrees of freedom and the four
constraint rows take `d = 1` exactly, which is what identity rows should take.
A closed form `log₂ d = a + b x² + c log x` fits to within 1.29 powers of two
for the ions and 2.06 for the electrons at `Nx = 16`, with `b` −0.86 and −0.60
and `c` about −1.7 — that is, `d ≈ x^-1.7 exp(−0.6 x²)`, a Maxwellian-like
weight rather than per-row noise. A structural surrogate was therefore
available and is not worth building.

**What `d` corrects is a row/column asymmetry at the bottom of the speed
grid.** The row and column norms of `A M⁻¹` run in opposite directions across
the speed grid, at `Nx = 16` and `l = 0`:

| | lowest speed | highest speed |
| --- | ---: | ---: |
| ion row norm | 10^10.7 | 10^1.6 |
| ion column norm | 10^0.0 | 10^9.7 |
| electron row norm | 10^10.3 | 10^1.8 |
| electron column norm | 10^1.2 | 10^11.0 |

Ten orders each way, and `d` tracks the row norm (the correlation of `log₂ d`
against `½ log₂` of the column-to-row ratio is −0.92). The imbalance is at the
low-speed end, where the collision frequency diverges, and it is larger for
the ions than for the electrons at every speed but the top.

## Decision

**Stop. The step is killed on its own criterion and no matrix-free estimator
of the balancing diagonal is built.** The conditioning falls as measured and
the iterations do not move by 1.5x, which is the kill as written. The gap is
between the balanced matrix and the balanced solve, and the residual curve in
both norms — the measurement the kill criterion asks for next — is taken above
rather than deferred: the two arms each converge fast in their own norm and
meet at the same iterate, so there is nothing for a better estimate of `d` to
recover.

**Two attributions are withdrawn.** That the eigenvector conditioning of
`A M⁻¹` is the quantity the `Nx` growth acts through: it is a true statement
about the matrix and a false one about the solve. And that the offending
directions are the high-speed rows of the light species: the balancing
diagonal corrects the low-speed rows, and the heavy species harder than the
light one.

**What is kept.** The structure of `d` is a physical result and stands on its
own: the preconditioned operator's row norms fall ten orders across the speed
grid while its column norms rise ten orders, the asymmetry is a smooth
Maxwellian-like function of speed with no angular or Legendre structure, and
it is worst where the collision frequency diverges. Any future scaling of this
operator has its diagonal already measured.

## Follow-up

- The `Nx` growth is unexplained again. The dropped collision coupling, the
  tolerance, chain equilibration, the factorization's accuracy, a second exact
  elimination, and now the eigenvector conditioning of the preconditioned
  operator have each been measured and each ruled out. The next candidate
  worth a probe is the transient rather than the asymptotic behaviour — the
  field of values or a pseudospectral radius of `A M⁻¹`, which bound GMRES
  where eigenvalues and eigenvector conditioning do not.
- `Nx = 12` on the production grid remains open from the record above: 3,810
  iterations at `1e-10` against 564 at `1e-9`, where its neighbours pay almost
  nothing for that decade.
- The gap deck's point A is untouched by this step and returns to the queue
  for the routes that do not depend on a scaling.
