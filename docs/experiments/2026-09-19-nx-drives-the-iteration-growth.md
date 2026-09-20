# The speed resolution drives the iteration growth, not the Legendre one

## Hypothesis

Phase 2b, from the [#253](https://github.com/uwplasma/DKX/pull/253) queue. The
HSX gap deck of `2026-09-19-sfincs-on-the-gap-deck.md` is named by its point A,
`Nxi = 120, Nx = 16`, where both DKX's recycled Krylov route and every SFINCS
configuration fail on a 36 GiB laptop. The queue treated that as one wall and
attributed it to the collision coupling the coarse preconditioner drops, which
is 0.9999997 of `A - M` on this deck by `dropped_couplings`.

Retaining that coupling did not pay (`2026-09-19-collision-coupling-is-cross-species.md`),
so the attribution was tested directly: which of the two resolutions the point
raises is responsible.

## Admission test

GCROT iterations and wall time with the default `coarse` preconditioner, on the
gap deck at the corners of `Nxi` in `{20, 40}` and `Nx` in `{10, 16}`, then a
ladder in `Nx` at fixed `Nxi = 20` to locate the break. Laptop, float64,
`tol = 1e-10`, `restart = 200`. The machine was shared, so wall times carry
contention and iteration counts do not.

## Result

**`Nxi` is free; `Nx` is not.** GCROT iterations:

| | `Nx = 10` | `Nx = 16` |
| --- | ---: | ---: |
| `Nxi = 20` | 174 | 2,788 |
| `Nxi = 40` | 157 | 7,167 |

Doubling the Legendre resolution at `Nx = 10` costs nothing at all -- 174
iterations become 157 -- while raising the speed resolution from 10 to 16 costs
16x at `Nxi = 20`. `Nxi` only bites once `Nx` is already large, where it adds a
further 2.6x. The point A wall is therefore an `Nx` wall.

**A ladder in `Nx` at `Nxi = 20`**, with the relative residual `||r||/||b||`
each solve reached against the `1e-10` it was asked for:

| `Nx` | iterations | relative residual | converged | wall |
| ---: | ---: | ---: | :---: | ---: |
| 10 | 174 | 7.56e-11 | yes | 7.0 s |
| 11 | 187 | 7.57e-11 | yes | 15.7 s |
| 12 | 3,810 | 1.05e-10 | **no** | 80.1 s |
| 13 | 969 | 9.87e-11 | yes | 25.1 s |
| 14 | 1,186 | 9.78e-11 | yes | 36.1 s |
| 15 | 1,970 | 9.74e-11 | yes | 65.6 s |
| 16 | 2,788 | 9.03e-11 | yes | 82.1 s |

`Nx = 12` is the one failure, and it misses by 5% where `Nx = 13` passes by
1.3%.

**It is not the tolerance.** Every solve stops just under the threshold it is
given, which is what a Krylov method does, so the near-tolerance residuals
above say nothing on their own. Asked for `1e-9` instead, the same points give
169 iterations at `Nx = 10`, 564 at `Nx = 12`, 959 at `Nx = 13` and 2,589 at
`Nx = 16`. A whole decade of tolerance buys 7% at `Nx = 16` and nothing at
`Nx = 13`: the growth is a convergence *rate*, not a floor the solves are
grinding into. Only `Nx = 12` changes, from 3,810 iterations to 564, so its
convergence curve flattens across `1e-10` and its failure is particular to that
grid rather than a property of the ladder.

**The speed discretization is what degrades.** Conditioning at `Nxi = 20`:

| `Nx` | `x_max` | `cond(ddx)` | worst collision block `cond` | collision diagonal range |
| ---: | ---: | ---: | ---: | --- |
| 10 | 4.26 | 2.5e6 | 7.3e24 | 2.9e-3 to 6.7e4 |
| 12 | 4.77 | 2.2e8 | 6.1e25 | 2.5e-3 to 1.4e5 |
| 13 | 5.01 | 2.2e9 | 1.1e25 | 2.3e-3 to 2.0e5 |
| 14 | 5.24 | 2.3e10 | 6.0e25 | 2.2e-3 to 2.8e5 |
| 16 | 5.68 | 2.5e12 | 1.5e28 | 2.0e-3 to 4.9e5 |

`cond(ddx)` gains about a decade per grid point: the speed collocation
derivative loses six digits at `Nx = 10` and twelve at `Nx = 16`. This is a
property of the discretization, not of the solver or the preconditioner, and it
is why retaining more of the collision coupling exactly did not help: the
coupling is not what the Krylov route is struggling with.

**And the conditioning is a scaling artifact.** Ruiz equilibration of the same
matrices, row and column scaling alone:

| `Nx` | `cond(ddx)` | equilibrated | `cond(x d/dx)` | equilibrated |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 2.47e6 | 13.9 | 9.50e4 | 12.2 |
| 12 | 2.24e8 | 16.4 | 6.47e6 | 14.7 |
| 13 | 2.23e9 | 17.7 | 5.71e7 | 16.0 |
| 16 | 2.51e12 | 21.7 | 4.69e10 | 20.0 |

A diagonal scaling removes eleven orders of magnitude at `Nx = 16` and leaves a
condition number of order ten that barely moves with `Nx`. The speed
discretization is not intrinsically ill-conditioned; it is badly scaled. That
is a fact about `ddx`, and the section below measures that it does *not* carry
over to the chains the preconditioner factors, where scaling changes nothing.

## Decision

**Point A is an `Nx` problem, and the queue's attribution of it to the
collision coupling is withdrawn.** `dropped_couplings` put the Fokker-Planck
share of `A - M` at 0.9999997 on this deck, and retaining that coupling made
convergence worse rather than better
(`2026-09-19-collision-coupling-is-cross-species.md`). The resolution that
costs is the one the operator-norm attribution says nothing about.

**And the factors do lose the chain, one speed at a time.** Comparing the
block-Thomas factors against the pinned chain its own generators produce --
`||K_b y - v||/||v||` per subsystem `b = (species, speed)`, float64 factors --
places the loss exactly where the speed grid grows. At `Nx = 10`:

| subsystem | ions | electrons |
| --- | ---: | ---: |
| lowest speed | 2.1e-16 | 4.7e-16 |
| highest speed | 8.7e-14 | 7.9e-8 |

The error rises monotonically with the speed index and is some six orders worse
for the light species, whose streaming coefficients carry the `sqrt(T/m)` that
the collision diagonal does not. At `Nx = 16` the top seven electron chains run
2.2e-10, 8.9e-10, 2.9e-9, 6.3e-9, 2.8e-8, 1.0e-7 and 3.2e-7: each speed point
added multiplies the worst chain's backward error by about three, and raising
`Nx` adds exactly such points at the top of the grid.

**It is the elimination, not the scaling.** Assembling those chains outright
separates the two. Against a dense LU of the *same* chain with the *same*
right-hand side, float64 throughout, at `Nx = 16`:

| subsystem | chain `cond` | block-Thomas | dense LU | ratio |
| --- | ---: | ---: | ---: | ---: |
| electrons, `x = 15` | 1.79e6 | 2.62e-7 | 1.36e-11 | 1.9e4 |
| electrons, `x = 14` | 1.18e6 | 5.75e-8 | 9.06e-12 | 6.4e3 |
| ions, `x = 15` | 6.36e3 | 7.88e-13 | 9.78e-14 | 8.1 |
| ions, `x = 0` | 6.5e2 | 1.85e-16 | 3.03e-16 | 0.6 |

A stable factorization of the worst chain returns `1.4e-11`, which is what its
condition number allows. The block-Thomas elimination returns `2.6e-7` from the
same matrix: a growth factor of about `3e9`, and it is the elimination that
loses it. The ratio is 0.6 where the chain is well conditioned and 1.9e4 where
it is not, which is the signature of an elimination that does not pivot.

**Equilibration is not the fix.** Ruiz scaling of the worst chain before a
dense LU returns `1.2e-11` against the unscaled `1.4e-11` -- nothing. The entry
spread of that chain is 9.9e6 and scaling removes it, and the backward error
does not move, so the loss is not the scaling the speed derivative's
conditioning suggested. The chains are genuinely ill-conditioned at the top of
the speed grid, `cond` 6.5e2 on the first and 1.79e6 on the worst, and they get
worse as `Nx` adds points there.

**And that accuracy is not what costs the iterations.** The sensitivity can be
measured rather than argued, by making the factors far worse on purpose.
`DKX_COARSE_FACTOR_DTYPE=float32` takes the worst chain's backward error from
`2.6e-7` to `793`, 9.4 decades, and takes GCROT at `(Nxi, Nx) = (20, 16)` from
2,788 iterations to 10,391 -- a factor of 3.7. The iteration count is that
weakly sensitive to how well the chains are factored.

Now compare what `Nx` actually does to those factors. From `Nx = 10` to
`Nx = 16` the worst chain's float64 backward error moves from `7.9e-8` to
`3.2e-7`: 0.6 of a decade, against the 9.4 decades that bought 3.7x. That
cannot produce the 16x in iterations over the same interval, and recovering it
entirely -- down to the dense LU's `1.4e-11`, 4.3 decades the other way --
would by the same measured sensitivity buy well under 2x.

A second elimination agrees. The `sparse` route eliminates the same operator in
a fill-reducing order rather than `L`-first, and at `(20, 16)` it returns
*exactly* 2,788 iterations and the same `9.03e-11` residual, with the two
preconditioner maps differing by `5.2e-9`.

**So the factorization is exonerated, and the growth is in the spectrum.** What
remains is the eigenvalue and eigenvector structure of `A M^-1` as `Nx` grows,
which none of the measurements here touch. That is the next thing to look at,
and it is a different kind of work from building another preconditioner.

**What is still not shown.** Why the spectrum degrades with `Nx` and not with
`Nxi`, and whether anything short of coupling the speeds in `M` addresses it --
noting that coupling them exactly is separately measured to make convergence
worse (`2026-09-19-collision-coupling-is-cross-species.md`). The weaker check,
whether the preconditioner inverts `_coarse_operator` as `Nx` grows, is
inconclusive and was abandoned: that operator omits the floor, the `l = 0` pin
and the drift diagonal the chains carry, leaving a constant 0.55 to 0.58
relative error at every `Nx`.

## Follow-up

- Measure the spectrum of `A M^-1` against `Nx` on a deck small enough to
  assemble both. Everything cheaper has been tried here: retaining more
  coupling, loosening the tolerance, scaling, and a second elimination, and
  none of them is where the growth lives.
- `Nx = 12` deserves its own look: 3,810 iterations at `1e-10` against 564 at
  `1e-9`, where its neighbours pay almost nothing for that decade.
