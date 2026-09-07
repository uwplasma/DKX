# Why banding the speed coupling cannot help

## Hypothesis

Plan phase 2, step 2; follows
`2026-09-07-dropped-coupling-attribution.md`, which measured the collision
operator's dropped speed and species coupling as essentially all of `A - M`.

SFINCS's namelist offers weaker simplifications on that axis than the diagonal
DKX keeps: `preconditioner_x = 3` retains the tridiagonal in speed and `= 4` the
diagonal and superdiagonal. If the dropped mass lies near the diagonal, taking
one of those intermediate points would recover much of it while keeping each
`(species, x)` subsystem's block-Thomas structure in `L` intact and cheap.

## Admission test

Measure `||A - M||/||A||` with the collision coupling retained at bandwidths 0
(the diagonal in use now), 1 and 2, and with the full speed coupling, on the
full-Fokker-Planck decks. Then look at where the collision matrix's mass
actually sits as a function of `|x - x'|`.

Kill: if bandwidth 1 or 2 does not materially reduce the residual operator, the
banded route is not worth implementing.

## Result

Banding recovers nothing. Two probes each:

| Deck | diagonal (now) | tridiagonal | pentadiagonal | full speed coupling |
| --- | ---: | ---: | ---: | ---: |
| `tokamak_full_fp_finite_er_high` | 0.9998 | 0.9998 | 0.9999 | 0.0176 |
| `w7x_sc1_full_fp_high` | 0.9969 | 0.9963 | 0.9962 | 0.0000 |

The reason is in the matrix. The linearized Fokker-Planck collision operator is
an integral operator in speed, so in this basis it is **essentially upper
triangular with its mass at the far corner**, not banded:

| Deck | `Nx` | strict lower triangle, Frobenius share | `\|x-x'\|=0` | `\|x-x'\|=1` | corner `\|x-x'\|=Nx-1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w7x_sc1_full_fp_high` | 7 | 8.1e-05 | 0.978 | 0.004 | 0.206 |
| `tokamak_full_fp_finite_er_high` | 10 | 1.5e-05 | 0.389 | 0.001 | 0.918 |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 3 | 3.1e-03 | 0.912 | 0.195 | 0.360 |

The first superdiagonal carries a tenth of a per cent. The corner carries up to
92 per cent. A band around the diagonal is the wrong shape for this operator.

## Decision

**Drop the banded proposal and take the triangle instead.** The strict lower
triangle is four to five orders of magnitude below the whole matrix, so
SFINCS's `preconditioner_x = 2`, which keeps the upper-triangular part in speed,
is for this operator very nearly the full collision coupling: the measurement
above says retaining it takes `||A - M||/||A||` from about 1.0 to 0.018 or below.

Its cost is a back-substitution, not a dense solve. Writing `M = D + U` with `D`
the speed-diagonal blocks already factored and `U` strictly upper in `x`, the
apply solves for `x = Nx-1` down to `0` with `y_x = D_x^{-1}(r_x - sum_{x'>x} U_{x,x'} y_{x'})`,
so each step reuses the existing block-Thomas factors unchanged and no new
factorization is needed.

The cost that must be measured before adopting it: the apply becomes sequential
over `Nx` instead of a single vmapped batch over `(species, x)`, which trades a
much better preconditioner against `Nx` dependent steps and `Nx^2/2` block
matrix-vector products. On GPU that sequencing may cost more than the iterations
it saves. That is the experiment, and it is worth running because nothing else
measured so far addresses more than two per cent of the dropped mass.
