# NCSX resolution ladder and baseline algebraic error estimates on the new route

## Hypothesis

Phase 1 step 2 and item Q9 of the #230 handoff. Two NCSX resolution checks were blocked on memory: theta25 was gated at "56.1 GiB available", and the baseline adjoint campaign at 42 GiB. With #231, #233 and single-threaded BLAS, those grids should fit in 20 GiB. Three things are tested:

- The refinement ladder and the four baseline adjoints run on one shared office CPU host, with unchanged physics and tolerances.
- The code changes reproduce the archived moments to rounding.
- The ladder yields per-axis error bars, or states which axes are not yet in the asymptotic range.

Owner: independent review. Budget: one afternoon of office CPU.

## Admission test

**Setup.**
- Deck: NCSX single-species, full Fokker–Planck with full-trajectory `E_r` (the Phase 1 baseline; `ncsx_baseline.namelist`). Only the resolution is changed.
- Code: DKX `main` `2b7641d`, SOLVAX 0.21.0 (PyPI), JAX 0.10.2, x64.
- Host: office Xeon W-2295, cores 8–11. Environment `DKX_CORES=4` with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` all set to 1. One fresh process per point.
- Watchdog: stops a point above 30 GiB RSS, below 8 GiB host-available, or at the time cap.

**Acceptance per point.** Original relative residual ≤ 1e-10, with the complete state returned by the Krylov route.

**Reproduction.** Points that also exist in the archived campaign (`dkx-phase1-completion/qualification-summary.json`) must agree with it to ≤ 1e-12 relative.

**Error bars.**
- Discretization: `dkx.workflows.converge.richardson_uncertainty` (fine-grid GCI; it refuses non-monotone ladders).
- Algebraic: `dkx.sensitivity.linear_observable_algebraic_error`, using a transposed GCROT with the solve's own transposed preconditioner at 1e-10 and 1e-12, to check that the estimate is stable.

## Result

**Ladder, first pass** (relative differences against the baseline):

| Point | Grid (θ, ζ, ξ, x) | Iterations | Build + solve | Peak RSS | Δ flow | Δ particle flux | Δ heat flux |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | (21, 37, 61, 8) | 53 | 43 s | 7.7 GiB | — | — | — |
| theta25 | (25, 37, 61, 8) | 114 | 98 s | 10.4 GiB | +3.06e-4 | −1.44e-4 | −3.33e-4 |
| theta29 | (29, 37, 61, 8) | 111 | 146 s | 13.3 GiB | +4.16e-4 | −3.35e-4 | −6.26e-4 |
| zeta43 | (21, 43, 61, 8) | 53 | 76 s | 10.0 GiB | −2.26e-4 | +2.73e-3 | +2.82e-3 |
| pitch60 | (21, 37, 60, 8) | 54 | 53 s | 7.8 GiB | +2.03e-4 | −1.49e-2 | −1.55e-2 |
| pitch81 | (21, 37, 81, 8) | 53 | 86 s | 10.1 GiB | +8.43e-5 | −7.11e-3 | −7.33e-3 |
| speed9 | (21, 37, 61, 9) | 60 | 70 s | 8.8 GiB | −1.69e-3 | −1.43e-4 | +2.27e-5 |
| joint | (25, 43, 81, 9) | 137 | 258 s | 19.7 GiB | −1.46e-3 | −6.24e-3 | −6.29e-3 |

- **Acceptance.** Every point converged with original residual between 4.9e-11 and 9.8e-11.
- **Reproduction.** The baseline, speed9, zeta43 and pitch81 reproduce the archived moments to 1.6e-14, 2.0e-14, 2.2e-14 and 3.1e-14 relative. So #231, #233, SOLVAX 0.21.0 and the BLAS setting did not change an answer.
- **Cost.** The archived runs of the same three grids took 128–278 s at 15.7–16.2 GiB. Here they take 70–86 s at 8.8–10.1 GiB. theta25, previously refused for memory, completes at 10.4 GiB.

**Poloidal ladder** `Ntheta = 21 → 25 → 29` (GCI):

| Moment | Result |
| --- | --- |
| Flow and current | asymptotic, observed order 5.3, GCI 1.2e-4 |
| Heat flux | asymptotic but at order 0.23, GCI 1.1e-2; the low order makes the band wide |
| Particle flux | refused (differences change behaviour; not in the asymptotic range) |

**Pitch neighbours.** Relative to `Nxi = 61`, `Nxi = 60` moves the particle and heat fluxes by −1.5% and `Nxi = 81` by −0.7%. The pitch resolution is therefore the dominant discretization uncertainty in the fluxes, at the percent level, and an even/odd neighbour differs by twice the 61 → 81 change. The flow is insensitive to pitch at the 2e-4 level.

**Iteration counts** roughly double when `Ntheta` rises from 21 to 25 or 29 (53 → 111–114) and reach 137 on the joint grid. The coarse preconditioner loses quality with poloidal resolution on this deck; this is recorded, not yet explained.

**Ladder, second pass** (same host and settings, relative differences against the baseline):

| Point | Grid (θ, ζ, ξ, x) | Iterations | Build + solve | Peak RSS | Δ flow | Δ particle flux | Δ heat flux |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| theta33 | (33, 37, 61, 8) | 102 | 212 s | 16.8 GiB | +4.96e-4 | −3.82e-4 | −7.23e-4 |
| zeta49 | (21, 49, 61, 8) | 53 | 85 s | 12.4 GiB | −2.73e-4 | +3.67e-3 | +3.80e-3 |
| pitch101 | (21, 37, 101, 8) | 53 | 115 s | 12.2 GiB | +8.53e-5 | −7.45e-3 | −7.75e-3 |
| pitch121 | (21, 37, 121, 8) | 53 | 135 s | 14.3 GiB | +8.53e-5 | −7.47e-3 | −7.77e-3 |
| speed10 | (21, 37, 61, 10) | 77 | 93 s | 9.7 GiB | −8.60e-4 | −4.24e-4 | −1.96e-4 |

**Per axis**, from three-rung ladders. `R` is the ratio of successive differences; `richardson_uncertainty` supports an estimate only for monotone convergence (`0 < R < 1`) at an observed order ≤ 12.

| Axis (rungs) | Flow | Particle flux | Heat flux |
| --- | --- | --- | --- |
| θ (25, 29, 33) | asymptotic, GCI 5.6e-4 | asymptotic, GCI 2.7e-5 | asymptotic, GCI 8.5e-5 |
| ζ (37, 43, 49) | monotone (R = 0.21) but order ≈ 12.1 > 12, refused; last change −4.7e-5 | asymptotic, GCI 8.8e-4 | asymptotic, GCI 9.1e-4 |
| ξ (81, 101, 121) | monotone (R = 0.011), order ≈ 25, refused; last change 1.2e-8 | monotone (R = 0.042), order ≈ 18, refused; last change −1.4e-5 | monotone (R = 0.050), order ≈ 17, refused; last change −2.1e-5 |
| x (8, 9, 10) | oscillatory (R = −0.49), spread 1.7e-3 | divergent (R = 2.0), spread 4.2e-4 | oscillatory (R = −9.6), spread 2.2e-4 |

- **Pitch (ξ) is converged by `Nxi = 101`.** The Legendre expansion converges faster than any power, so the helper's power-law order check refuses it. The last difference (≤ 2e-5 in the fluxes) is the honest estimate.
- **Baseline pitch error.** Relative to the converged pitch value, the baseline grid `Nxi = 61` understates both fluxes by 0.75%. This is the largest single discretization error at the baseline.
- **Speed (x).** It is not in the asymptotic range at `Nx = 8–10`, and its spread dominates the flow uncertainty (1.7e-3).

**Baseline algebraic error estimates.** One primal solve at `(21,37,61,8)`: 53 iterations, residual 8.54e-11. For each moment, a transposed GCROT used the transposed preconditioner from that solve. The audit took 4 min in total, peaking at 9.7 GiB; the campaign had been gated at 42 GiB available.

| Moment | Adjoint iterations (1e-10 / 1e-12) | Adjoint residual (1e-10 / 1e-12) | Signed correction `λᵀr` | Relative to the moment |
| --- | ---: | ---: | ---: | ---: |
| `FSABFlow` = `FSABjHat` | 33 / 81 | 9.0e-11 / 6.4e-12 | −1.206e-11 | +1.8e-10 |
| particle flux | 30 / 37 | 7.6e-11 / 4.7e-13 | −8.407e-17 | −1.6e-10 |
| heat flux | 28 / 34 | 9.7e-11 / 8.9e-13 | −8.362e-17 | −4.5e-11 |

- **Acceptance.** All eight audits pass their original primal and transpose residual gates.
- **Stability.** Tightening the adjoint from 1e-10 to 1e-12 changes each correction by less than 3e-11 of itself. The flow adjoint at 1e-12 stopped at its restart cap with residual 6.4e-12; that residual passes the audit and gives the same estimate.
- **Interpretation.** These are estimates, not bounds. The algebraic error of every reported moment is about 1e-10 relative: eight orders of magnitude below the discretization differences above.

Scripts, records and logs are kept outside Git in `dkx-review-evidence-20260913/q9/` (`q9_ladder.py`, `q9_adjoints.py`, `run_ladder.sh`, `run_ladder2.sh`, `run_adjoints.sh`, `*.json`, `run.log`).

## Decision

Continue Phase 1 at a revised reporting grid, and record three follow-ups.

**Error budget.** Algebraic error (≈ 1e-10) is negligible. At the baseline `(21,37,61,8)` the fluxes carry a −0.75% pitch error, +0.37% from ζ, and ≲ 1e-3 from θ and x. The flow carries ≲ 2e-3, dominated by x.

**Reporting grid.** For the positioning figure use at least `Nxi = 101` and `Nzeta = 49` for the fluxes, and extend the speed ladder to `Nx = 11, 12` before quoting a flow error bar. Each of these rungs fits in ≤ 17 GiB and runs in ≤ 4 min on four cores.

**Follow-ups:**
1. `richardson_uncertainty` refuses monotone ladders that converge faster than order 12, which is the normal behaviour of the spectral pitch and ζ directions here. It should report the last difference as the estimate in that case, rather than "not in the asymptotic range".
2. GCROT iterations double from `Ntheta = 21` to 25–33 (53 → 102–114) and reach 137 on the joint grid. The coarse preconditioner's quality versus poloidal resolution needs a diagnosis; this is a performance item, not an accuracy item.
3. Repeat the ladder on a two-species deck and on the collaborator's HSX-like deck at its resonant `E_*`, where the SFINCS manual expects the speed resolution to matter most.
