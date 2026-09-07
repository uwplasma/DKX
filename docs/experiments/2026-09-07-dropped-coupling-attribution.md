# Which coupling the coarse preconditioner drops

## Hypothesis

Plan phase 2, step 1; feeds the solver figure of the methods paper.

Every deck that leaves the pitch-angle-scattering family is preconditioned by
the SFINCS-simplified operator `M`, which the coarse preconditioner inverts
exactly. The recycled Krylov route nonetheless wins only 7 of 23 such decks. If
the shortfall is the preconditioner's approximation rather than the Krylov
method, then one of the couplings `M` drops should dominate `A - M`, and which
one it is should decide whether to strengthen `M` or to extend an exact route.

Owner: research. Time budget: one day for the attribution; iteration counts and
memory are a separate measurement.

## Admission test

For each available deck, split `A - M` by mechanism, matrix-free, on random
probes of the f-block, and report each mechanism's share of `||(A - M) f||`
together with `||(A - M) f|| / ||A f||`. The split must be an identity, not an
estimate, and must be insensitive to the probe.

Kill criterion: if no mechanism exceeds half the dropped mass on any deck, or
if the attribution changes with the probe, the diagnosis is uninformative and
Phase 2 proceeds to direct iteration-count measurement instead.

## Result

`dropped_couplings` in `src/dkx/coarse_precond.py`, tested in
`tests/test_dropped_couplings.py`. The decomposition is exact: the operator is
a sum of terms, so the restored mechanisms reproduce `(A - M) f` to round-off,
which is pinned by a regression. The baseline is the operator the
preconditioner actually inverts, including the collision diagonal it retains
under `preconditioner_species=1` and `preconditioner_x=1`; an earlier draft
compared against a collisionless operator and overstated the collision share
(0.86 against 0.38 on the Phi1 deck).

Decks in `validation/inputs` and `tests/reduced_upstream_examples`, two probes:

| Deck | `||A-M||/||A||` | Dominant | Shares |
| --- | ---: | --- | --- |
| `w7x_eim_mdke`, five `w7x_standard_pas_dkes_*` | 0.0000 | none, `M` is exact | pitch-angle scattering is already diagonal in speed and species |
| `tokamak_1species_FPCollisions_noEr_withPhi1InDKE` | 0.3764 | `fokker_planck` | 1.000 |
| `w7x_sc1_full_fp_high` | 0.9953 | `fokker_planck` | 1.000 |
| `tokamak_full_fp_high` | 0.9997 | `fokker_planck` | 1.000 |
| `tokamak_full_fp_finite_er_high` | 0.9997 | `fokker_planck` | 1.000, `er_xdot` 0.017 |
| `tokamak_full_fp_finite_er_ultra` | 0.9999 | `fokker_planck` | 1.000, `er_xdot` 0.010 |
| `w7x_sc1_full_fp_ultra` | 0.9994 | `fokker_planck` | 1.000 |
| `transportMatrix_geometryScheme2` | 1.0020 | `fokker_planck` | 1.000 |

The attribution is robust to the probe. On `tokamak_full_fp_finite_er_high`,
white noise, a smooth field with only `L <= 1` populated, and a smooth field
decaying in `L` all give `fokker_planck` 1.000 with `er_xdot` between 0.007 and
0.018.

## Decision

**The dropped speed and species coupling of the collision operator is the only
lever that matters on every deck measured.** The `E_r` terms carry one to two
per cent of the dropped mass; the magnitude `||A-M||/||A||` ranges from 0.38 to
1.00, so on the collisional decks the preconditioner does not represent most of
the operator's action.

This reorders the phase's remaining experiments. Making the `E_r` and
tangential-drift terms exact through a block-pentadiagonal solve, proposed as
the phase's second experiment, addresses at most two per cent of the dropped
mass on these decks and is deferred until a deck is measured in which those
terms dominate. Work on the collision coupling instead: retain more of it in
`M`, by keeping a banded rather than diagonal speed coupling, or by a coarse
correction that restores the cross-species blocks.

Two limits of this record, both material. No deck exercising tangential
magnetic drifts is in the available set, so `magnetic_drifts` is untested here;
the two W7-X drift decks are the ones that exhausted memory in the preconditioner
bands. And this is an operator-norm attribution, not a convergence prediction:
Krylov convergence depends on the spectrum of `M^-1 A`, so the ranking says
what to address first, while the iteration counts and memory that decide
whether it pays remain to be measured.
