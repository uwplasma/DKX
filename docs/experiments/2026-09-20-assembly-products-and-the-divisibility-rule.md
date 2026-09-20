# The assembly's product count, and the rule that was costing half of it

## Hypothesis

P8 of the [#253](https://github.com/uwplasma/DKX/pull/253) queue asks for at
most 1,000 products per assembly against the 8,800 in use on the gap deck,
quoting "a bound of 198". That 198 is the densest row of the pattern, which is
the Curtis-Powell-Reid lower bound on the number of colours. A lower bound is
not a target: it is only reachable if the conflict graph admits groups that
large.

## Admission test

Measure, on the gap deck at `(Nxi, Nx) = (120, 16)` with `Ntheta = 11` and
`Nzeta = 15`: the angular stencil radii, the separations in use, the number of
groups, their sizes, and the largest group the conflict structure permits. Then
reduce the count if there is slack, with the assembled matrix unchanged
entrywise.

## Result

**The scheme was already at its own optimum.** Every one of the 8,800 groups
held exactly 72 columns, and `f_size / 72 = 633,600 / 72 = 8,800`. No colouring
of that structure can do better, so the 198 was never reachable and the gap
between it and 8,800 is not slack.

The size 72 is `3 x 24`: three angular points per group and 24 Legendre slots
at a separation of 5. Two columns at distinct angular points of the lattice
never share a row whatever their species, speed or Legendre index -- the
collision operator reaches every `(species, speed)` but only at its own angular
point, and the streaming stencil reaches other angular points but only within
one `(species, speed)`. So the angular lattice is what sets the group size, and
the species-speed index rides along for free.

**The slack was in a rule, not in the colouring.** The angular separation was
required to *divide* the grid, which makes the residue classes uniform and the
periodic wrap automatic. With `Ntheta = 11`, a prime, and a stencil radius of
2, the only divisor above `2 x 2` is 11 itself: every theta became its own
class. Nothing about the geometry requires that. `{0, 5}` has cyclic gaps 5 and
6, both above 4, so it is a legal class, and the eleven points partition as
`{0,5} {1,6} {2,7} {3,8} {4,9} {10}` -- six classes where divisibility gives
eleven.

Dropping the divisibility and checking the wrap-around gap directly:

| | before | after |
| --- | ---: | ---: |
| groups on the gap deck | 8,800 | **4,800** |
| group sizes | 72 | 72 to 144 |
| theta classes (`Ntheta = 11`) | 11 | 6 |
| zeta classes (`Nzeta = 15`) | 5 | 5 |

Each product is one operator application whatever the group holds, so this is
1.83x less assembly work on that deck. Where a dividing separation is already
tight -- `Nzeta = 15` here, or any grid short enough that the radius spans it --
the classes reproduce it exactly, so no deck gets a worse count.

**The matrix is unchanged.** On a deck built to exercise the new classes,
`Ntheta = 11` and `Nzeta = 15` at `(Nxi, Nx) = (6, 4)`, the assembly recovers
1,208 products against 7,924 columns with a verification error of 2.2e-16, and
every entry agrees with full column-by-column sampling to 0.0 exactly.

## Decision

**Take the 1.83x and restate P8's acceptance, which was unreachable.** The
target should be the structural bound `f_size / (angular lattice points x
Legendre slots)`, not the densest row. On the gap deck that is now 4,800, and
the remaining lever is the angular lattice itself: with radius 2 on 11 points,
two is the most that can be pairwise separated, because three arcs of 5 do not
fit in a circle of 11.

**A resolution note worth having.** The product count depends on the arithmetic
of `Ntheta` and `Nzeta` and not only on their size. At radius 2, `Ntheta = 12`
admits classes of 2 with no leftover where 11 admits five pairs and a
singleton, and a grid whose length is a multiple of `2 x radius + 1` is the
happy case. A deck chosen one point larger can assemble faster.

## Follow-up

- The Legendre slot count, 24 at `Nxi = 120`, comes from a separation of
  `2 x l_bandwidth + 1 = 5`. Two columns at one angular point with different
  `(species, speed)` need only `|dl| > 2`; it is the same-`(species, speed)`
  pair that needs `|dl| > 4`. A grouping that mixes the species-speed index
  within an angular point could use a separation of 3 and take the slots from
  24 to 40, which is the next factor available.
