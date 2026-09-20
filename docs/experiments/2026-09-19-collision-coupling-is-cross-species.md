# The retained speed triangle kept the wrong block

## Hypothesis

Phase 2b step 2, from the [#253](https://github.com/uwplasma/DKX/pull/253)
queue. That step proposed keeping the whole collision coupling as a second
factor, `M⁻¹ = (D_c + C_off)⁻¹ D_c (S + D_c)⁻¹`, so that both speed triangles
and the species coupling are kept while the first factor stays one batched
apply. The collision array carries no angular index, so the second factor is
one `n_species · n_x` square block per Legendre row: 32 wide on the HSX deck,
and free beside a chain solve.

Before building it, two things were measured that the step had assumed: where
the collision operator's mass sits on a two-species deck, and what the factored
splitting costs.

## Admission test

`||A − M||/||A||` on random probes of the f-block, per retention, and GCROT
iterations on the HSX deck of `2026-09-19-sfincs-on-the-gap-deck.md` at
`Nxi = 20, 40, 60` and `Nx = 10, 16`. The retention arms are the shipped
`coarse` (the collision diagonal `mat[s, s, l, x, x]`, Fortran
`preconditioner_x = 1`), the shipped `coarse_triangle` (that plus the strict
upper speed triangle of each species' own block, `preconditioner_x = 2`), and
the same triangle taken over every species pair.

## Result

**The mass is cross-species.** Per block of `mat`, as a share of the whole
operator's Frobenius norm, on the HSX deck at `(Nxi, Nx) = (20, 10)`:

| block | diagonal in `x` | strict upper | strict lower |
| --- | ---: | ---: | ---: |
| ion row, ion column | 2.7e-3 | 2.5e-2 | 4.1e-7 |
| ion row, electron column | 2.0e-3 | **9.990e-1** | 2.0e-3 |
| electron row, ion column | 1.1e-18 | 2.7e-13 | 2.1e-19 |
| electron row, electron column | 3.6e-2 | 6.3e-4 | 1.6e-7 |

The ion-electron block holds 99.9% of the norm, and the electron-ion block is
at round-off: the block is triangular in species as well as in speed. What each
retention drops, as a share of that norm: the collision diagonal 0.9993, the
self-species triangle 0.9990, every species pair's triangle 0.0020.

**So `preconditioner_x = 2` recovers nothing here.** `2026-09-07-collision-speed-structure.md`
measured the full speed coupling of the *matrix* and read the self-species
triangle off the strict lower triangle's share, which is sound on the
one-species decks it used. `_strict_upper_speed_coupling` then kept
`mat[s, s]`, so on a multi-species deck the option retains one species' own
block and leaves the coupling that carries the mass in the dropped remainder.
Measured as `||A − M||/||A||` over three probes:

| deck point | `coarse` | self-species triangle | every species pair |
| --- | ---: | ---: | ---: |
| `Nxi = 20, Nx = 10` | 0.99999 | 1.00003 | 4.0e-3 |
| `Nxi = 40, Nx = 16` | 1.00000 | 0.98701 | 4.3e-4 |

`dropped_couplings` agrees on the cause: at `(40, 16)` the Fokker-Planck share
of `A − M` is 0.9999997, against 4.3e-4 for the `E_r` xDot term and 1.2e-9 for
xiDot, so on this deck the dropped `E_r` and drift terms are not what the
preconditioner is missing.

**Taking every species pair costs nothing.** The retained coupling is strictly
upper in speed whichever species pair it comes from, so `D + U` is unchanged in
structure and the existing back-substitution over `x` inverts it exactly. The
apply reads one more index in its contraction and factors nothing new.

**The factored splitting is a negative result.** Built and measured before the
structure above was known: on a one-species test deck its
`||M⁻¹ A v − v||/||v||` is 5.1e5 against 1.5 for `coarse`, and it stays four to
six orders of magnitude worse across `nu_n` from 8.5e-3 to 1e2. The reason is
the splitting error `S D_c⁻¹ C_off`: `D_c⁻¹ C_off` is not small, because the
collision block's off-diagonal exceeds its diagonal by up to 7.9e4 on this
deck, so putting `C_off` in a separate factor amplifies rather than corrects.
An exact substitution keeps the same coupling with no such term. The code was
removed.

**And a closer `M` converges worse.** GCROT on the gap deck at
`(Nxi, Nx) = (40, 16)`, 192,000 unknowns, `tol = 1e-10`, `restart = 200`,
identical operator, one arm per tree:

| preconditioner | `||A − M||/||A||` | iterations | wall |
| --- | ---: | ---: | ---: |
| `coarse` | 1.00000 | 7,167 | 316 s |
| self-species triangle | 0.98701 | 5,799 | 472 s |
| every species pair | 4.3e-4 | 6,786 | 511 s |

Retaining the block that carries the mass makes `M` 2,300 times closer to `A`
and costs 17% *more* iterations than retaining the block that carries almost
none of it. The bootstrap current agrees across all three arms to 1.6e-9
relative, so this is a conditioning result and not a wrong solve.

The reason is that `D + U` is inverted exactly, and that inverse is
ill-conditioned: the collision block's off-diagonal exceeds its diagonal by up
to 7.9e4, so back-substitution over `x` amplifies by about 3.1e3 in the norm of
the applied map. A preconditioner is judged by the spectrum of `M⁻¹A`, not by
`||A − M||`, and on this deck the two point opposite ways. `dropped_couplings`
warns that its attribution is not a convergence prediction; this measures how
far apart the two can be.

Wall time makes it worse: both triangles lose to `coarse` on this deck, because
the apply is sequential over `Nx` where `coarse` batches
(`2026-09-18-speed-triangle-back-substitution.md`).

## Decision

**Keep the self-species triangle and record why the larger block is refused.**
The cross-species generalization is one contraction index and no new
factorization, so it was cheap to build and cheap to reject; it is rejected on
its iteration count, which is the plan's stated criterion, not on its cost. The
code is reverted and `tests/test_speed_triangle_precond.py` pins the choice
with a pointer here, so the block is not retained again by someone reading only
the norm.

**The factored splitting of the #253 queue's step 2 is dropped**, and with it
the assumption that the collision coupling is what the Krylov route is missing.
`2026-09-19-nx-drives-the-iteration-growth.md` measures what it is instead.

## Follow-up

- A retention that is *not* inverted exactly stays open: damping `U` toward
  `D`, or one truncated sweep rather than the full back-substitution, trades
  exactness in `M` for a better conditioned map. The series form already exists
  behind an integer `retain_speed_triangle`.
- The claim in `2026-09-07-collision-speed-structure.md` that
  `preconditioner_x = 2` is "very nearly the full collision coupling" holds for
  the one-species decks it measured and not for multi-species decks.
