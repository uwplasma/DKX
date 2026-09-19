# Batched LU dispatch on CPU and GPU

## Hypothesis

Plan Phase 2 step 7. Both the dense coarse route and the generated Schur route
factor many dense blocks of the same shape, so how that batch is dispatched sets
their cost. Two questions: is jaxlib's batched CPU LU a serial loop over the
batch, which would make sharding across host devices worth its complexity; and
which dispatch should an accelerator use. Owner: independent review. Budget: one
office day.

Measured on random matrices at the size the NCSX coarse route factors,
`n = 777`, not on operator rows: this prices dispatch, not the preconditioner.

## Admission test

- **CPU.** Eight pinned physical cores on the office Xeon W-2295. Every
  configuration runs in one process, interleaved in rotating rounds, with the
  BLAS thread count switched per sample and verified with `threadpoolctl`, so a
  comparison is not a comparison of process start times. Best and median over
  the samples. Admission threshold: 1.5× at equal cores against the best
  non-sharded configuration.
- **GPU.** One RTX A4000, admitted only with no foreign compute process and at
  most 10% utilisation across five polls. Best of five warm repetitions, with
  compilation excluded and a foreign-process monitor running throughout.
- **Both.** Every variant must reproduce the reference factorization.
- The host is shared, so absolute rates are lower bounds and only same-process
  ratios decide. A first CPU attempt at 14:33 and a GPU attempt on 2026-09-13
  were discarded for foreign load, unmeasured rather than reported.

## Result

**CPU**, `n = 777`, milliseconds per matrix, best and median of 60 and 40
samples, against the best non-sharded configuration:

| Configuration | Best | Median | vs `vmap`, one BLAS thread |
| --- | ---: | ---: | ---: |
| `lu_factor` B=64, `vmap`, 8 BLAS threads | 621.2 | — | 0.01× |
| `lu_factor` B=64, `vmap`, 1 BLAS thread | 3.45 | 4.53 | 1.00× |
| `lu_factor` B=64, `shard_map` k=8, 1 BLAS thread | 2.89 | 3.88 | 1.19× |
| `lu_factor` B=32, `vmap`, 1 BLAS thread | 4.57 | 5.57 | 1.00× |
| `lu_factor` B=32, `shard_map` k=8, 1 BLAS thread | 2.36 | 3.20 | 1.94× |
| Schur step B=8, `vmap`, 1 BLAS thread | 17.1 | 19.8 | 1.00× |
| Schur step B=8, `shard_map` k=8, 1 BLAS thread | 10.3 | 12.2 | 1.65× |
| `lu_factor` B=64, SciPy `ThreadPool(8)`, 1 BLAS thread | 1.66 | 2.16 | 2.08× |

- **The batch is not a serial loop.** At one BLAS thread the batched LU occupies
  3.8–4.9 cores and beats a serial SciPy loop by 2×, while staying 2× short of
  an eight-thread pool.
- **Multithreaded BLAS inside the batch collapses it**, to 513–621 ms per matrix
  against 3.4–4.6, about 100×. It reproduced in every attempt. The mechanism was
  not isolated; a spin-waiting BLAS pool against XLA's intra-op pool fits.
- JAX factorizations match SciPy to 8e-12, and the Schur step to 8e-12.

**GPU**, one A4000, `n = 777`, milliseconds per matrix, best of five:

| Operation | `vmap` | `lax.map` batch 6 | `lax.map` batch 1 |
| --- | ---: | ---: | ---: |
| `lu_factor`, B = 8 | 8.24 | 5.74 | 5.81 |
| `lu_factor`, B = 32 | 2.72 | 5.81 | 5.80 |
| `lu_factor`, B = 64 | 2.58 | 5.74 | 5.73 |
| `lu_solve` 777 RHS, B = 8 | 4.33 | 4.77 | 7.74 |
| `lu_solve` 777 RHS, B = 32 | 3.95 | 4.69 | 8.01 |
| `lu_solve` 777 RHS, B = 64 | 3.84 | 4.59 | 7.92 |

- **The dispatch crossover is the batch size.** From 32 matrices up, `vmap`
  factors 2.1–2.2× faster than chunked `lax.map`; at 8 it is 1.4× slower.
- At B = 64 the A4000 matches the best eight-core CPU configuration measured
  here, 2.58 ms against 2.66 ms per matrix.
- The variants agree to 2.2e-11.

Scripts, raw JSON and logs are kept outside Git in
`dkx-review-evidence-20260913/step7/`: `interleaved.py`, `gpu_lu.py`,
`gpu_poll.py`, `cpu_json_interleaved/`, `gpu_lu_gpu0.json`, `RESULTS.md`.

## Decision

- **One BLAS thread around batched CPU LU: adopted**, as the host default in
  `dkx.runtime` (#234). It is a precondition for every other comparison here.
- **Host-device sharding for the Schur recursion: conditional.** `shard_map`
  k=8 at one BLAS thread clears the 1.5× bar on the dominant per-block cost
  (1.61–1.73×) but not on plain `lu_factor` at B=64 (0.87–1.19×). The margin was
  measured on a contended host, so it is queued for re-confirmation on a quiet
  one rather than adopted.
- **Accelerator dispatch: `vmap` from 32 matrices up, chunked `lax.map` below.**
- **What that implies for the coarse route, and why it is not a change here.**
  The dense coarse route factors `n_species * n_x` blocks per Legendre row, 16
  on NCSX, which is below the crossover. On an accelerator the table predicts a
  chunked dispatch factors about 1.4× faster there. That is a prediction from
  random matrices on one device; it is queued behind a re-measurement on the
  operator's own rows with an idle GPU, and the CPU default is unaffected.
- **Side finding, not an admission item.** SciPy's `ThreadPool(8)` at one BLAS
  thread is 2.1× the best JAX CPU configuration measured. A host-callback path
  is a separate question.
