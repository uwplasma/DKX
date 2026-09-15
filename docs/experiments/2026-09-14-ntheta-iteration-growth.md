# Why GCROT iterations grow with poloidal resolution on NCSX

## Hypothesis

Follow-up 2 of `2026-09-14-ncsx-refinement-ladder.md` (#230 handoff, Q9). On the NCSX baseline deck, GCROT iterations roughly double from `Ntheta = 21` to 25–33 (53 → 102–114) at otherwise fixed resolution. Three candidate causes are tested:

- The coupling the coarse preconditioner drops grows with `Ntheta`.
- The `E_r` terms the preconditioner drops (`xDot`, `xiDot`) are responsible.
- The Fokker–Planck speed coupling the preconditioner drops (`preconditioner_x = 1`) interacts with poloidal resolution.

Owner: independent review. Budget: one hour of office CPU.

## Admission test

**Setup.**
- Deck: `ncsx_baseline.namelist` from the ladder record (single species, full Fokker–Planck, `Er = -1`). Only the resolution, `Er`, or the preconditioner kind changes.
- Code: DKX `main` `2b7641d`, SOLVAX 0.21.0, JAX 0.10.2, x64.
- Host: office Xeon W-2295, four pinned cores per run, `DKX_CORES=4`, BLAS pools at one thread. One fresh process per point, with a watchdog on resident memory, host availability and time.
- Reduced grid: `(Ntheta, 25, 41, 6)`, about 30 s per point. Full grid: `(Ntheta, 37, 61, 8)`.

**Acceptance per point.** Converged with relative residual ≤ 1e-10 (every point below passed, between 5.3e-11 and 9.8e-11).

**Criteria.**
- **Adoption bar for the triangle, set before the full-grid runs.** `coarse_triangle` becomes a candidate default only if its full-grid solve at `Ntheta = 25` is faster than `coarse` at the same residual gate.
- **Attribution rule, written after the reduced sweep and before the controls.** A candidate cause is kept when removing or restoring it moves the `Ntheta` 21 → 25 iteration ratio toward 1 by at least 20% of the `coarse` ratio's excess over 1. It is rejected when the ratio stays or grows.

**Instruments.**
- `dkx.coarse_precond.dropped_couplings` (four random probes) for the size and composition of `A − M`.
- `solve(..., preconditioner="coarse_triangle")`, which also retains the collision operator's upper speed triangle (Fortran `preconditioner_x = 2`), for the speed-coupling test.

## Result

**The growth reproduces on the reduced grid**:

| `Ntheta` | 15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iterations | 41 | 36 | 40 | 44 | 56 | 69 | 60 | 80 |
| Peak RSS (GiB) | 2.2 | 2.5 | 2.8 | 3.0 | 3.3 | 3.6 | 3.9 | 4.2 |

**The dropped coupling does not grow.** `‖(A − M) f‖ / ‖A f‖` stays between 0.531 and 0.540 for every `Ntheta` above. The Fokker–Planck remainder carries all of it (share 1.000). The `xDot` and `xiDot` terms carry 1.9e-3 and 2.4e-4. The growth is therefore spectral: `M⁻¹A` changes with resolution, while the size of `A − M` does not.

**Controls at `Ntheta = 21` and 25** (reduced grid):

| Run | Iterations at 21 | Iterations at 25 | Ratio |
| --- | ---: | ---: | ---: |
| `coarse` (default) | 44 | 69 | 1.57 |
| `coarse`, `Er = 0` | 34 | 100 | 2.94 |
| `coarse_triangle` | 16 | 22 | 1.38 |

- Removing `E_r` makes the growth worse, so the `E_r` terms are not the cause.
- Retaining the speed triangle cuts iterations by 2.8–3.1× and removes most of the growth. The cause is the Fokker–Planck speed coupling, which in this basis is essentially upper triangular (`2026-09-07-collision-speed-structure.md`).

**Full grid** `(Ntheta, 37, 61, 8)`; the `coarse` rows are the ladder record's runs on the same host and settings:

| `Ntheta` | Kind | Iterations | Build | Solve | Solve per iteration | Peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 21 | `coarse` | 53 | 17.0 s | 25.6 s | 0.48 s | 7.7 GiB |
| 21 | `coarse_triangle` | 20 | 21.7 s | 107.1 s | 5.35 s | 8.3 GiB |
| 25 | `coarse` | 114 | 34.9 s | 63.3 s | 0.55 s | 10.4 GiB |
| 25 | `coarse_triangle` | 28 | 27.2 s | 180.7 s | 6.45 s | 10.8 GiB |

- On the full grid the triangle cuts iterations 2.7× at `Ntheta = 21` and 4.1× at 25.
- Each iteration costs about 11× a `coarse` iteration, so the solve is 2.9–4.2× slower.
- The triangle runs shared the host with other jobs (load 8–13), against 2–5 for the ladder runs. That contention does not account for an 11× per-iteration gap.

Scripts, records and logs are kept outside Git in `dkx-review-evidence-20260913/q9/ntheta_sweep/` (`run.sh`, `dropped_probe.py`, `triangle_control.py`, `triangle_control.sh`, `er0_control.sh`, `triangle_full.sh`, `*.json`, `*.log`).

## Decision

**Keep `coarse` as the default. Record the cause, and queue the triangle apply cost behind Q5.**

- **Attribution.** The `coarse` ratio 1.57 sets the cut-off at 1.46. Restoring the speed triangle gives 1.38, so that cause is kept. Removing `E_r` gives 2.94, so that cause is rejected.
- The poloidal growth in GCROT iterations comes from the Fokker–Planck speed coupling that `preconditioner_x = 1` drops. It does not come from `E_r`, and not from a larger dropped term.
- **Adoption bar fails.** At `Ntheta = 25` the triangle solve takes 180.7 s against 63.3 s for `coarse`.
- The triangle already fixes the iteration count. What fails is its cost per apply. Running the eight speed levels in sequence instead of as one batch should cost at most about the core count, 4× here. The measured 11× points at the apply implementation.
- **Follow-up.** Profile the `coarse_triangle` apply at NCSX `(25, 37, 61, 8)` after the operator-coupling work (Q5). The admission bar is a triangle apply at ≤ 2× a `coarse` apply. At that cost the `Ntheta = 25` solve would take about 28 × 1.1 s ≈ 31 s, against 63 s.
- For ladders and scans at `Ntheta ≥ 25` before that lands, expect roughly twice the `Ntheta = 21` iteration count on this deck. Neither the answer nor the admission changes.
