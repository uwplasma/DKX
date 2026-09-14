# Resolution ladder on the HSX-like deck at resonant `E_*`

## Hypothesis

This is follow-up 3 of `2026-09-14-ncsx-refinement-ladder.md` and a consequence of `2026-09-13-sfincs-sparsify-threshold.md` (Q9 and Q8 in the #230 handoff). That record's decision holds that near `E_* ≳ 1/3`, ladders in `Nx` and `Nxi` are mandatory for both the fluxes and the current.

This record tests whether the collaborator's grid `(Ntheta, Nzeta, Nxi, Nx) = (11, 15, 20, 10)` resolves the bootstrap current and the fluxes at the two reconstructed points. It also identifies which velocity-space axis dominates. Owner: independent review. Budget: one afternoon of office CPU.

## Admission test

**Setup.**
- **Deck.** The collaborator's quasi-helically symmetric deck (`nfp = 4`, two species, full Fokker–Planck, full-trajectory `E_r` terms). The equilibrium is private, SHA-256 `db6197f0…79e0d9a`.
- **Points.** A (`rN = 0.187`, `Er = 15`) and B (`rN = 0.367`, `Er = 14.39`), both at `E_* ≈ 1–2`. Only `Nxi` and `Nx` change.
- **Code.** DKX `main` `e83307d`, SOLVAX 0.21.0, JAX 0.10.2, x64, CPU.
- **Solver.** `run_from_namelist` with GMRES `tol = 1e-10`, restart 100, up to 60 restarts.
- **Host.** Office Xeon W-2295, with points A and B on separate sets of four pinned cores and one BLAS thread. Each point has a 1 h timeout and a memory watchdog.

**Acceptance per point.** Converged, with final relative residual ≤ 1e-10. A point that does not converge is recorded as a failure, not as a value.

**Resolution rule.** This rule was set when writing this record, after the runs. An axis counts as resolved for a moment when its last refinement changes the moment by less than 1%.

## Result

Every converged point reached a residual between 2.1e-11 and 2.8e-11. The collaborator's own grid (`Nxi = 20`, `Nx = 10`), solved tight, gives `FSABjHat` +1.103e-2 at A and −3.141e-2 at B.

**Pitch, `Nx = 10`:**

| Point | `Nxi` | Iterations | `FSABjHat` | Ion flow | Ion particle flux | Ion heat flux | Electron heat flux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 60 | 361 | −1.6936e-2 | +6.778e-3 | −5.291e-10 | −5.593e-11 | −2.313e-8 |
| A | 120 | 366 | −1.5790e-2 | +6.767e-3 | −5.287e-10 | −5.589e-11 | −2.431e-8 |
| A | 180 | 365 | −1.5831e-2 | +6.767e-3 | −5.287e-10 | −5.590e-11 | −2.445e-8 |
| B | 60 | 153 | +6.8396e-3 | −7.943e-3 | +2.814e-9 | +6.379e-10 | −2.575e-8 |
| B | 120 | 150 | +6.3380e-3 | −7.945e-3 | +2.818e-9 | +6.386e-10 | −2.809e-8 |
| B | 180 | 150 | +6.3408e-3 | −7.945e-3 | +2.818e-9 | +6.386e-10 | −2.811e-8 |

- **Pitch is resolved by `Nxi = 120` at both points.** From `Nxi = 120` to `180` every moment changes by less than 0.6%. The current changes by 2.6e-3 at A and 4.4e-4 at B.
- **Richardson.** `richardson_uncertainty` refuses the current at both points, because the last difference changes sign. The electron heat flux is asymptotic at A (GCI 3.5e-3). At B it is faster than order 12 (bar 1.7e-3).
- **The collaborator grid is far from this.** Against the `Nxi ≥ 120` value at `Nx = 10`, its current has the opposite sign at both points, with 0.7× the magnitude at A and 5× at B.

**Speed, `Nxi = 120`:**

| Point | `Nx` | Iterations | `FSABjHat` | Ion flow | Ion particle flux | Ion heat flux | Electron heat flux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 10 | 366 | −1.5790e-2 | +6.767e-3 | −5.287e-10 | −5.589e-11 | −2.431e-8 |
| A | 13 | 1575 | −3.8430e-2 | −5.349e-2 | +9.644e-9 | +2.266e-9 | −2.432e-8 |
| B | 10 | 150 | +6.3380e-3 | −7.945e-3 | +2.818e-9 | +6.386e-10 | −2.809e-8 |
| B | 13 | 495 | +6.5609e-3 | −7.447e-3 | +2.728e-9 | +6.143e-11 | −2.810e-8 |

- **Speed is not resolved in the ion channel.** At A, `Nx` 10 → 13 multiplies the current by 2.4. It reverses the ion flow and multiplies its magnitude by 8, and it reverses the ion particle and heat fluxes. At B the current moves 3.5%, the ion flow 6%, and the ion heat flux drops by 90%.
- **The electron heat flux is resolved in speed.** It changes by 5e-4 at A and 4e-4 at B.
- **Iterations grow with `Nx`.** They rise from 366 to 1575 at A and from 150 to 495 at B.

**`Nx = 16`.**
- **Default coarse preconditioner.** Neither point converged within the 1 h cap. The relative residual was 0.056 after 3359 s at A and 5.0e-7 after 3309 s at B.
- **Speed-triangle retry.** `preconditioner = "coarse_triangle"`, up to 200 restarts, one point at a time on four cores. Point A did not finish. After 2 h 16 min in the solve, another job on the shared host took 26 GiB, and the memory guard (available memory below 8 GiB) stopped the run. This route prints no residual history, so no residual is recorded. The same guard stopped point B within a second of starting.

Scripts, records and logs are kept outside Git in `dkx-review-evidence-20260913/hsx_ladder/`: `hsx_ladder.py`, `hsx_ladder_v2.py`, `run_chain.sh`, `run_retry.sh`, `run_retry_sequential.sh`, `hsx_ladder_analysis.py`, `runs/*/record.json`, `chain_*.log` and `retry_*.log`.

## Decision

The collaborator grid does not resolve this deck at resonant `E_*`, and neither the current nor the ion fluxes from it should be quoted.

- **Pitch.** Use `Nxi ≥ 120`.
- **Speed.** The ion channel is the limiting axis. `Nx = 13` is not converged at A, so no bootstrap current or ion flux at these points is admitted until an `Nx = 16` rung converges and agrees with `Nx = 13`.
- **Electrons.** The electron heat flux is resolved at `Nxi = 120`, `Nx = 10`.
- **Solver.** The `Nx ≥ 16` rungs need a stronger preconditioner than the default coarse route, and the speed-triangle route does not finish point A in 2 h on four CPU cores. See the `Ntheta` record for why the dropped Fokker–Planck speed coupling matters. The operator-coupled apply does not make the triangle cheaper (`2026-09-14-coarse-operator-couplings.md`), so these rungs wait on the speed back-substitution item.
