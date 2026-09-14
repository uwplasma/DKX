# Operator couplings in the coarse preconditioner

## Hypothesis

This is Phase 2 step 6.4, item Q5 of the #230 handoff.

The coarse preconditioner's `L ± 1` couplings are `x (c_L S + c'_L diag(mirror))`: one streaming stencil with about 9 nonzeros per row of 777 on NCSX, which the dense route stores and applies as `(Ntheta*Nzeta)^2` blocks. SOLVAX 0.22 adds `block_thomas_factor_ops`, which takes the couplings as actions and stores one Schur block per row, and its benchmark applies such a chain 6× faster on CPU with explicit inverses (`benchmarks/results/operator-couplings-cpu-x64-m777.json`).

The expected outcome was a cheaper apply at a third of the storage, with NCSX GCROT at `(21, 37, 61, 8)` within 30 s on 4 CPUs. Owner: independent review. Budget: one day.

## Admission test

- **Route.** An opt-in switch, `DKX_COARSE_OPERATOR_COUPLINGS=lu|inverse`, on the dense coarse route. `_stencil_couple` applies the couplings through the Kronecker factors of the streaming operator. The diagonal blocks, the `1e-8` floor, identity rows on truncated rows and the `l = 0` pin come from the unchanged `_coarse_subsystem_block_fn`. Chains of equal `Nxi_for_x` length are grouped.
- **Certification (plan).** Map equality ≤ 1e-12, forward and transposed, and identical GCROT iteration counts on PAS, Fokker–Planck, Sugama, `Phi1` and magnetic-drift decks. The item is killed on any iteration-count change.
- **Exit.** NCSX GCROT ≤ 30 s on 4 CPUs.
- **Hosts.** Laptop (JAX 0.11.1) for the decks. Office Xeon W-2295 for NCSX, on cores 4–7 with `DKX_CORES=4`, one BLAS thread, JAX 0.10.2, SOLVAX 0.22.0 and float64.

## Result

**Coupling action.** It reproduces the generated coupling blocks to ≤ 4e-16 on every deck, forward and transposed.

**Map agreement with the dense route**, forward / transposed, f-block:

| Deck | Worst row `cond₁` | `lu` | `inverse` |
| --- | ---: | ---: | ---: |
| Fokker–Planck | 9e3 | 6.0e-14 / 6.4e-14 | 1.4e-13 / 8.7e-14 |
| improved Sugama | ~9e3 | 2.3e-13 / 1.6e-13 | 1.0e-13 / 1.1e-13 |
| `Phi1` in collisions | 1.1e5 | 1.1e-12 / 9.6e-13 | 2.3e-12 / 1.9e-12 |
| PAS tiny | 5.2e9 | 3.3e-8 / 3.9e-8 | 2.2e-8 / 2.6e-8 |
| magnetic drift, `nu_n = 0` | 2.3e16 | 1.0e-3 / 9.1e-4 | 0.63 / 0.51 |
| NCSX `(21, 37, 61, 8)` | — | 2.8e-13 / 4.3e-13 | 3.4e-13 / 3.7e-13 |

- **Where 1e-12 is missed.** Backward errors on the assembled pinned chains decide the cause. `lu` matches the dense route on every deck: 2.6e-15 against 2.2e-15 on PAS, and 5.4e-10 against 3.9e-10 on the drift deck. So its map gaps are round-off amplified by conditioning.
- **`inverse` is not backward stable on the drift deck.** Its backward error is 1.1e-1 against 3.9e-10 for dense.

**GCROT iterations**, dense / `lu` / `inverse`:
- **JAX 0.11.1:** Fokker–Planck 90 / 90 / 90, Sugama 100 / 100 / 100, PAS 2 / 2 / 2, PAS ramps 3 / 3 / 3 and 2 / 2 / 2.
- **Magnetic drift:** 18 / 18 / no convergence (6000 iterations).
- **JAX 0.10.2:** Sugama 99 / 99 / 100.
- **NCSX:** 53 / 53 / 53, with moments within 3.1e-14 of the refinement-ladder baseline.

**NCSX `(21, 37, 61, 8)`, office.** Load 3–9, 33–54 GB available.

| Route | Build | Solve | Per iteration | Whole call | Peak RSS | Warm apply in `jit` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 16.9 s | 23.0 s | 0.43 s | 39.9 s | 7.69 GiB | 0.46 s |
| `lu` | 18.2 s | 70.9 s | 1.34 s | 89.1 s | 3.69 GiB | 1.03 s |
| `inverse` | 20.1 s | 20.0 s | 0.38 s | 40.1 s | 3.70 GiB | 0.26 s |

- **Timing noise.** Build times vary by about ±15% between runs on this shared host.
- **Padding.** Padding every chain to full length makes both storages slower, so the chain grouping is not what makes `lu` slow. The cause is not isolated.
- **A hang.** On JAX 0.10.2, the compiled `inverse` factorization of `pas_2species_ramp` never returns at `DKX_CORES=2`. It works at 1 and 4 threads.

The route is 380 lines in 5 files. It is kept unmerged as `dkx-review-evidence-20260913/dkx-perf-coarse-operator-couplings.bundle` (branch `perf/coarse-operator-couplings`, `efbf896`). Run records are in `dkx-review-evidence-20260913/q5_ops/`.

## Decision

**Stop Q5 as a CPU performance item, and keep the dense route.**

- **`store="inverse"` is killed.** It changes an iteration count (Sugama 99 → 100 on JAX 0.10.2) and does not converge on the magnetic-drift deck, where its backward error is 1e-1. It also hangs at one thread count. Its 13% faster NCSX solve does not justify that.
- **`store="lu"` certifies.** Its iteration counts are identical and its backward error equals the dense route's, at half the peak memory (7.69 → 3.69 GiB). But it makes the NCSX solve 3× slower, so it is not merged as a speed-up. It is the candidate if a memory-limited route is needed.
- **Why the SOLVAX gain does not carry over.** The SOLVAX benchmark compares against a stored-band Thomas sweep. DKX's dense route already eliminates row by row in batched solves (#233), which the operator-coupled `lu` apply does not match on CPU.
- **Exit target.** The NCSX solve phase is 23.0 s on the dense route; the whole call, including the build, is 39.9 s.
- **Follow-ups.**
  - Profile the `lu` apply against the row-batched dense sweep before any GPU decision (Q6).
  - The speed-triangle back-substitution item stands on its own.
  - The SOLVAX documentation should state that explicit inverses are not backward stable on nearly singular chains.
