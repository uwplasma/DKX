# The HSX bootstrap-current discrepancy is SFINCS's sparsification threshold

## Hypothesis

Phase 1 (collaborator replay, cross-code parity for the positioning figure).
On a collaborator's quasi-helically symmetric VMEC equilibrium (nfp = 4,
minor radius 0.126 m, `T_i = 59 eV`, `T_e = 1.37 keV`, two species, full
Fokker–Planck, full-trajectory `E_r` terms, `(Ntheta, Nzeta, Nxi, Nx) =
(11, 15, 20, 10)`, 66,004 rectangular / 43,894 active unknowns), tight DKX
and refined SFINCS v3 solves disagree on the bootstrap current by 19% (point
A, `rN = 0.187`, `Er = 15`) and 12% (point B, `rN = 0.367`, `Er = 14.39`),
while the right-hand sides agree to 6e-15 and a random operator action to
2.4e-12 (#229). The hypothesis tested here is that the two discrete
operators differ in an identifiable block, and that the difference is
amplified by a near-null parallel-flow mode rather than being a physics or
resolution difference. Owner: independent review. Budget: one day on the
laptop CPU, no code change.

## Admission test

Same deck, equilibrium (SHA-256 `db6197f0…79e0d9a`) and physics as #229;
DKX `cf0038f4`, SOLVAX 0.20.0, JAX 0.11.1, x64, CPU. Inputs: the refined
SFINCS state (`sfincsBinary_iteration_000_stateVector`, original residual
8.0e-14 against the SFINCS matrix `whichMatrix_3`), the SFINCS matrix itself,
and the tight DKX state (GCROT restart 100, `rtol = 1e-11`, original residual
8.65e-12).

1. Decompose `A_DKX x_SFINCS − b` and `x_DKX − x_SFINCS` by species, Legendre
   index and speed index. Pass: the difference localizes to a named block.
2. Compare single columns of `A_DKX` (one operator application on a unit
   vector) with the same column of the SFINCS matrix, split by row species.
3. Recompute the DKX collision operator with all three Rosenbluth-potential
   routes (`quadpack`, `analytic`, `hybrid`) and re-solve; pass if they agree
   with each other to 1e-10 in the current.
4. Kill criterion for the "operator difference" hypothesis: if no block
   differs by more than 1e-9 relative, attribute the gap to conditioning and
   resolution instead.

## Result

Amplification. The DKX operator applied to the SFINCS state has relative
residual 1.14e-8; the two states differ by 1.0% in norm: a state-over-residual
amplification of 9e5. The state is 98% Legendre `L = 1` (ion parallel flow).
The ion block of the two operators agrees to 1e-10 on every column tested;
the 1% ion-flow difference is roundoff-level operator differences at the
lowest speed point (`x = 0.039`, diagonal entries 37.0487682723 versus
37.0487682712) amplified by the near-null flow mode.

Localization. The electron block of the SFINCS state has norm 1.9e-9 of the
ion block (mass-ratio normalization). Relative to the electron right-hand
side, the DKX operator's residual on the SFINCS state is 1.4e-4 (ion rows:
1.1e-8), concentrated in electron `L = 1` rows at `x_e` indices 3–5. Column
comparison: electron columns agree to 1e-14 in every row; ion columns agree
to 1e-10 in ion rows; **ion columns differ by 2.8e-5 to 6.8e-3 in electron
rows**, and SFINCS holds exact zeros in electron rows wherever the DKX entry
of the ion→electron field-particle collision block is below about 1e-12 in
magnitude (for example ion column `x_2, L = 1`: DKX rows `−1.94e-12,
−4.0e-13, −7.5e-14, …`; SFINCS keeps the first and stores nothing for the
rest).

Mechanism. SFINCS inserts matrix entries through `MatSetValueSparse`
(`fortran/version3/sparsify.F90`), which skips any value with
`abs(value) <= threshholdForInclusion = 1d-12`. The ion→electron
field-particle block on this deck (`sqrt(T_e m_i / (T_i m_e)) = 207`) has
entries between 1e-26 and 8e-7. Applying the same absolute threshold to
DKX's collision tensor removes 442 of 8,000 entries at point A (all in the
electron←ion block) and 325 at point B, and then:

| Point | DKX | DKX with the 1e-12 threshold applied | SFINCS refined |
| --- | ---: | ---: | ---: |
| A, `FSABjHat` | 0.0110337063 | 0.0136166376026 | 0.0136166376004 |
| A, electron `FSABFlow` | 0.00534222 | 0.00259311 | 0.00259311 |
| B, `FSABjHat` | −0.0314056428 | −0.0356737206647 | −0.0356737206663 |

The thresholded DKX column matches the SFINCS column to 5.7e-15 (ion column
`x_4, L = 1`, electron rows), the thresholded DKX residual on the SFINCS
state drops from 1.4e-4 to 4.3e-11 relative in the electron rows, and the
currents agree to 2e-10 (A) and 4e-11 (B). GCROT iteration counts are
unchanged (895/897 at A, 382/382 at B).

Independence of DKX's value. The `quadpack`, `analytic` and `hybrid`
Rosenbluth routes agree with one another to 1e-12 on the electron←ion
columns and give the same current to 7e-12. SFINCS's own `dqage`
(`quadpack.f`, key 6, tolerances 1e-13) integrates the relevant Maxwellian
moments on `[0, x_b]` to 1e-15 for `x_b` up to 877, so the quadrature is not
the cause; the sparsification threshold is.

Context from the SFINCS paper and manual (Landreman, Smith, Mollén &
Helander 2014; `doc/manual/version3/resolution.tex`): the deck sits above the
`E_r` resonance band (`E_* ≈ 1–2` against the `E_* < 1/3` range where the
trajectory models agree), and HSX with very small `T_i/T_e` is the one case
the manual names where resolution must follow `E_r`. That explains the 9e5
amplification and the "patternless" Krylov stagnation on a `(radius, Er)`
grid (lines of constant `E_*` are rays), not the cross-code gap.

Scripts and JSON outputs are retained outside Git in
`dkx-review-evidence-20260913/` (`localize_discrepancy.py`,
`species_residuals.py`, `column_blocks.py`, `rosenbluth_arbiter.py`,
`threshold_test.py`, `threshold_b.py`, `dqage_test.f90`).

## Decision

The 12–19% bootstrap-current gap is a reference artifact, not a DKX defect:
SFINCS's hard-coded 1e-12 absolute sparsification threshold removes the
ion→electron field-particle coupling on hot-electron, cold-ion decks, and
the electron parallel flow on a quasisymmetric surface is sensitive to it
because electron viscosity is weak. DKX keeps the entries and is the more
faithful discretization of the stated model.

Consequences for the plan: SFINCS references for parity on decks with
`sqrt(T_e m_i / (T_i m_e)) ≳ 30` are admitted only with the threshold
disabled (rebuild with `threshholdForInclusion` lowered or the
`valueToSet /= 0` variant) or compared against DKX with an explicit, opt-in
`SFINCS`-threshold compatibility switch used for parity tests only; report
the finding upstream; add an `E_*` diagnostic to results so scans can be
plotted against the resonance parameter; and treat resolution ladders near
`E_* ≳ 1/3` as mandatory in `Nx` and `Nxi` for both fluxes and current.
