# The GPU questions phase 2 left open

## Hypothesis

Two steps were decided on CPU with an explicit caveat that a device with more
parallelism per step might change the arithmetic, and both caveats are now
testable: office was unreachable when they were written.

Step 2 killed the exact upper speed triangle because its apply is sequential
over `Nx` where the default is one batched solve, and each sequential step
carried a batch of one species. Step 3 found that holding the preconditioner
fixed across a field sweep is the lever, measured on one species on CPU.

## Admission test

Repeat both on an RTX A4000, and extend the reuse ablation to two species,
where the preconditioner is larger and rebuilding it should cost more.

## Result

Analytic full-Fokker-Planck decks, `Ntheta = Nzeta = 7`, `Nxi = 8`, JAX 0.10.2,
CUDA, medians of three warm solves after `block_until_ready`.

### The triangle is not rescued by the accelerator

| `Nx` | unknowns | iterations, diagonal | iterations, triangle | iteration ratio | wall-time ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1,178 | 9 | 6 | 1.50 | 0.92 |
| 5 | 1,962 | 12 | 8 | 1.50 | 0.27 |
| 8 | 3,138 | 16 | 10 | 1.60 | 0.27 |
| 10 | 3,922 | 18 | 11 | 1.64 | 0.20 |

Three to five times slower, the same shape as CPU and worsening with `Nx`. The
iteration gain is smaller here than the 1.75 to 2.5 measured on CPU. **The kill
stands on both devices**, and the caveat that motivated retesting is closed
rather than left hanging.

### Preconditioner reuse holds, and is larger on GPU

Eight `E_r` points from 0 to 9, every point meeting 1e-10 on the original
equation:

| Species | Preconditioner | Recycle | Total iterations | Wall time (s) |
| ---: | --- | --- | ---: | ---: |
| 1 | rebuilt per point | off | 110 | 17.15 |
| 1 | rebuilt per point | on | 102 | 13.91 |
| 1 | **held fixed** | **off** | 401 | **6.21** |
| 1 | held fixed | on | 291 | 10.63 |
| 2 | rebuilt per point | off | 112 | 15.65 |
| 2 | rebuilt per point | on | 103 | 11.11 |
| 2 | **held fixed** | **off** | 397 | **6.12** |
| 2 | held fixed | on | 282 | 9.14 |

Holding the preconditioner fixed is **2.8 times faster** on one species and 2.6
on two, against 2.4 on CPU. The two-species case behaves like the one-species
case, so doubling the unknowns did not change the balance.

Two differences from CPU are worth recording rather than smoothing over. The
iteration price of a fixed preconditioner is much higher here, 110 to 401
against 206 to 225 on CPU, and it is still overwhelmingly worth paying because
rebuilding is what dominates. And recycling helps on GPU where it did not on
CPU, 17.15 to 13.91 seconds with the preconditioner rebuilt, but it is
*counterproductive* combined with a fixed preconditioner, 6.21 to 10.63. The
best configuration measured on either device is a fixed preconditioner with no
recycling.

## Decision

**Step 2 stays killed**, now on both devices, and the GPU caveat is discharged.

**Step 3's direction is confirmed and sharpened.** The refresh policy should
hold the preconditioner and *not* recycle against it, which is the opposite of
the pairing the step originally proposed. Recycling remains defensible only
while the preconditioner is being rebuilt anyway, which is the configuration the
policy is meant to replace.

These decks are small, 1,178 to 3,930 unknowns, and the sweep is one direction
in one parameter. The reuse economics still need a production-scale deck before
becoming a default, which is a separate measurement from this one.
