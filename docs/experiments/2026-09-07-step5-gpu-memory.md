# Do the decks the bands cannot hold fit on a 16 GB accelerator?

## Hypothesis

Plan phase 2, step 5, the admission that needed hardware.

Five of the six decks that did not complete were exhausted while the coarse
preconditioner allocated its dense angular bands. The reusable Schur LU route
fixed that on a 36-core, 62 GB machine, where every deck the route was built for
completes. The step asks for the same on the office GPU's 16 GB, with the
original-equation residual at or below 1e-10, and the `float32` factor switch
exists precisely to halve the factors where that decides whether a deck fits.

## Admission test

Build the operator and the coarse preconditioner for each of the five decks on
an RTX A4000 and report peak device memory, before committing to solves that
take between fifty minutes and two hours each on CPU.

## Result

All five route correctly: `_coarse_bands_fit` is false and `_coarse_factors_fit`
is true, so each takes the reusable Schur LU rather than the dense bands.

**At float64 none of them fit.** Every deck fails, and the failure is real
rather than an artefact of tuning: with XLA autotuning left on, the error is
`Autotuning failed ... Out of memory while trying to allocate 1.21GiB` for an
`f64[1265,100,1265]` transpose; with autotuning disabled the same deck fails
with a plain `RESOURCE_EXHAUSTED` from the computation itself.

**At float32 all five fit**, on a 16,376 MiB device:

| Deck | unknowns | peak device memory | build |
| --- | ---: | ---: | ---: |
| `HSX_PASCollisions_fullTrajectories` | 1,884,860 | 11.28 GB | 61 s |
| `HSX_FPCollisions_DKESTrajectories` | 1,884,854 | 10.77 GB | 54 s |
| `HSX_FPCollisions_fullTrajectories` | 1,884,854 | 10.77 GB | 48 s |
| `filteredW7XNetCDF_2species_magneticDrifts_noEr` | 1,518,004 | 8.98 GB | 40 s |
| `filteredW7XNetCDF_2species_magneticDrifts_withEr` | 1,518,004 | 8.94 GB | 38 s |

Disabling autotuning is not required: the same deck builds at float32 with
autotuning on, using 9.99 GB against 8.98 GB. It is worth about a gigabyte of
headroom and nothing else.

## Decision

**`float32` factors are what make these decks fit on this device, and that is
the use the switch was added for.** The measurement supports keeping it opt-in
and float64 the default, while documenting that a 16 GB accelerator running
these decks is the case where it should be switched on.

This also places the earlier float32 result precisely. On small decks where the
coarse preconditioner is nearly exact, float32 costs 40 to 76 times the
iterations, because the factor error becomes the dominant error in the
preconditioner. These decks are the opposite regime: the preconditioner is a
poor approximation, iterations are already high, and the factor error disappears
into an approximation that is already much larger. Both statements are true, and
which applies is decided by how good the preconditioner already is rather than
by preference.

**Fitting is not completing, and the admission fails on the second half.** With
the preconditioner built at float32 and holding about 9 GB, the GMRES loop then
asks for 7.96 GB more and the solve dies with `RESOURCE_EXHAUSTED` inside
`jit_while`. XLA's own rematerialization pass reports it cannot get the loop
body below 8.15 GB, so the total is about 17 GB against a 16 GB device.

That 8 GB is the loop's working set rather than its Krylov vectors: at default
`restart = 30` and `recycle_dim = 8` those hold 38 vectors of 1.5M float64,
about 456 MB, so shrinking them cannot cover the gap.

The step's most memory-lean lever does not close it either. Forcing the
checkpointed coarse route, which regenerates rows on every application instead
of storing the Schur LU, and cutting the Krylov workspace to `restart = 12` and
`recycle_dim = 2`, still fails after eight minutes, with CUDA unable to load a
compiled kernel under the remaining pressure.

The admission as written is therefore **not met on this hardware** at any
setting tried. What is established is narrower and still useful: the preconditioner,
which is what exhausted memory on CPU and motivated the whole step, now fits
comfortably, and the remaining obstacle has moved to the solve's working set.

What would settle it, and is not guesswork from here: these are 1.5 to 1.9
million unknowns on a 16 GB card, and the same decks complete on a 62 GB CPU
machine. Either a larger accelerator, or distributing one system across the two
A4000s rather than replicating it, is the shape of the answer, and single-device
state decomposition is listed under deferred work for exactly this reason.
