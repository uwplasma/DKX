# Where the NCSX (21, 37, 61, 8) time and memory go, and what is exact to remove

## Hypothesis

Phase 1 (memory gate on the poloidal refinement ladder) and Phase 2 (why the
Krylov route loses). The merged profiles (#228, `2026-09-12-reuse-admission.md`)
put the NCSX baseline at 164–187 s and 27–28 GiB on four office CPUs, with
56–57 s in fused band assembly and factorization and ~70 s in 53 GCROT
iterations, and gate the next refinement `(25, 37, 61, 8)` on 56.1 GiB of
available memory. The hypothesis is that most of that peak and most of the
per-iteration cost are storage and lowering choices that can be removed
without changing the preconditioner's linear map, so the gate is a defect
rather than a resource requirement. Owner: independent review. Budget: one
day, kernel microbenchmarks only, no production code change.

## Admission test

Kernels are the production ones (`solvax.direct.block_thomas_factor` and
`block_thomas_solve`, the pinned SOLVAX `9357191` snapshot the profiles
used) at the production block size `m = Ntheta·Nzeta = 777`, timed under
`jax.jit` after a warm call, best of repeats, on four pinned office CPU cores
(Xeon W-2295, JAX 0.10.2, x64) and on an idle RTX A4000. A structural
prototype that keeps the same linear map is admitted only if it agrees with
the production solve to 1e-12 relative on random right-hand sides and its
residual against the assembled bands is at roundoff. SuperLU is timed on
the real assembled NCSX simplified subsystems (`dkx.sparse_precond.
assemble_simplified` on the frozen deck), factor time and fill against the
dense bands. Kill for the "avoidable" hypothesis: if the measured per-block
factor cost accounts for the 56 s and the apply cost is within 2× of the
BLAS-2 bound, the profile is intrinsic.

## Result

Stage ledger, from the merged runs' `stages.jsonl` and `resources.jsonl`
joined on the same clock (pre-#228 run; post-#228 differs by ~20 s in the
border and probe stages):

| Stage | Time | RSS during stage | Cause |
| --- | ---: | ---: | --- |
| Dense band extraction | 9.2 s | → 13.4 GiB | eager Python list of 61×3 dense blocks |
| Fused assembly + factorization | 57.5 s | **8.5 → 28.0 GiB (peak)** | 3 input bands + 3 output bands + ~6 intermediates of 2.2 GiB each; 178 of 488 blocks are `Nxi_for_x` identity padding |
| Border materialization + two border setups | 8.1 + 12.2 + 13.4 s | 14 → 20 GiB | 242 eager compiles; the transposed preconditioner is built although non-differentiable solves never use it |
| Zero probe | 8.8 s | +4.4 GiB transient | a discarded synchronizing apply with its own compile |
| GCROT, 53 iterations | 70.3 s (≈7.7 s compile) | steady +7 GiB | ≈1.17 s per iteration, ~90% preconditioner apply |
| Summed compile time | 25 s | | the unrolled 61-block substitution is compiled three times and inlined a fourth |

Kernel costs (per 777×777 block):

| Kernel | 4 CPU cores | A4000 |
| --- | ---: | ---: |
| `block_thomas_factor` step, vmapped over 2 subsystems | 94 ms (74 ms unbatched; SciPy 55 ms) | 14.4 ms |
| `block_thomas_solve` per block | 3.9–4.8 ms | 1.7–3.8 ms |
| of which `lu_solve` (two `trsv`) | 2.3 ms (SciPy 1.3 ms) | 0.15 ms |
| of which dense coupling `gemv` | 0.13 ms | 0.09 ms |

488 blocks × 94 ms = 46 s of the 57 s stage is genuine LAPACK work at ~11
GFlop/s; the rest is the elementwise assembly passes. 36.5% of the blocks
are identity padding (per-`x` chain lengths `[7, 13, 23, 35, 49, 61, 61,
61]`), so the exact work is ~30 s on CPU and ~5 s on the A4000.

Structure. Every coupling block is `x·c_L·S_s + x·c'_L·diag(mirror)` with one
sparse 5-point streaming stencil `S_s` per species (9 nonzeros per row of
777); only the Schur complements `Δ_L` fill in. A scratchpad prototype that
stores `Δ_L^{-1}` (or its LU) and applies the couplings through the stencil
reproduces the production solve to 2.7e-15 (CPU) / 1.6e-15 (GPU) with a
residual of 5e-15 against the dense bands, at one third of the storage
(309 MB versus 928 MB for 2×32 blocks), and applies **5.2× faster on CPU and
22× faster on the A4000**. Factorization through an explicit inverse is 1.2×
faster on the GPU and 2.0× slower on CPU, so the CPU route keeps LU.

Exact byte counts at the production grids (float64 LU of active blocks):
NCSX baseline 1.39 GiB (bands 6.6 GiB, measured peak 27–28 GiB); NCSX
`(25, 37, 61, 8)` about 2–3 GiB (bands 9.3 GiB; gated at 56.1 GiB); HSX
full-FP 1.88M unknowns 17.8 GiB float64 / 8.9 GiB float32 (bands 53 GiB);
yancc's NCSX grid `(43×65, Nxi 61 / 121)` 25 / 49 GiB.

SuperLU on the real NCSX simplified subsystems (`n = 47,397` per (species,
x)): COLAMD 92 s and 1.96 GiB of factors for the full-`L` subsystem (dense
block-Thomas ≈ 5.7 s and 0.82 GiB of bands), 47–65 s for a padded one;
minimum-degree did not finish inside 420 s. The fill-reducing route is not
competitive on `m ≈ 800` full-FP decks; its documented 7–11× fill
advantage came from other decks and was never a wall-time claim.

Releases. DKX requires `solvax >= 0.19.0`; the latest SOLVAX release is
0.20.0, so users and CI install a preconditioner that applies the principal
inverse twice per GCROT iteration (SOLVAX #101 is merged but unreleased),
while every recorded profile binds SOLVAX git `9357191`. The route guard
sizes coarse factors against host memory (`/proc/meminfo`), not device
memory, on accelerators.

Retained outside Git in `dkx-review-evidence-20260913/`: `kernel_bench.py`,
`apply_bench.py`, `structured_bench.py`, `sparse_bench.py`, `gpu_lu_batch.py`
and their JSON outputs.

## Decision

Continue with the exact, structure-preserving storage rewrite (plan §4,
Phase 2 step 6) before any further memory-gated refinement or reuse-policy
qualification: generate blocks inside the factorization scan, factor per
`Nxi_for_x` group, apply couplings as stencils, thread the right-hand side
through a scan instead of whole-band slices, build the transposed
preconditioner lazily, drop the zero probe, and size the guard against
device memory. Release SOLVAX 0.21.0 and raise DKX's floor so published
timings match installed code. Stop the SuperLU route for `m ≈ 800` decks
and keep multigrid closed (DKX's own smoother study). Certification of every
item is map equality to 1e-12 and identical GCROT iteration counts, not
wall-clock.
