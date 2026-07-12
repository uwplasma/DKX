# Phase 1 failure analysis (working notes — folded into docs/numerics/performance.md later)

Measured on the development MacBook (Apple silicon, ~10 cores, 24 GB RAM), branch
`refactor/v3-driver-architecture` @ 01ccaf30, JAX CPU backend, x64. Machine was under
concurrent load (Fortran reference sweep) for the long runs; treat wall times as upper
bounds and re-measure key numbers on an idle machine before publishing.

## Confirmed defects / gaps

1. **Branch tip was broken (fixed).** `outputs/writer.py` imported
   `_rhsmode1_host_dense_shortcut_allowed` which commit `da621048` ("Collapse host dense
   shortcut wrapper") had removed — every RHSMode=1 CLI run crashed at the solve step.
   Fixed in `01ccaf30` by importing the policy functions directly. Lesson for Phase 3:
   the writer↔solver coupling has no import-contract test at the CLI level; add a CLI
   smoke test that runs a tiny namelist end-to-end.

2. **Memory, small case.** `tokamak_1species_FPCollisions_noEr_withQN` (tiny grid):
   warm solve+diagnostics 4.0–4.2 s, wall 5.7 s, **peak RSS 1.4–1.6 GB**. The Fortran
   binary solves the same case in O(100 MB). The gap is structural: dense/CSR
   materialization plus JAX allocator retention on the host path.

3. **Memory + runtime, production case.** `HSX_PASCollisions_DKESTrajectories` at the
   production manifest resolution (Ntheta=25, Nzeta=115, Nxi=149, Nx=5; 2.14 M phase
   points): observed 3.6–5.7 GB RSS, **killed after ~2.6 h wall without completing**
   (machine under concurrent load, but the point stands). This is the exact workload the
   plan's tier-1/tier-2 solvers target: the operator is block-tridiagonal in Legendre
   modes with dense (25·115)² = (2875)² blocks — one factor/solve sweep should be
   O(Nxi · 2875³) flops of batched GEMM, minutes not hours, with O((NθNζ)²) memory.

4. **Fortran baseline sanity.** The conda PETSc 3.23 + MUMPS 5.8.2 build passes the
   upstream example checks to ~4e-5 relative. Two local (never-upstreamed) patches were
   required for modern PETSc mpi_f08 typing: `MPIU_Comm` declarations in
   `globalVariables.F90` and `sfincs_main.F90` (version-guarded with
   `PETSC_VERSION_GE(3,19,0)`).

## Baseline timings collected so far (this machine)

| Case | Fortran (1 rank) | sfincs_jax warm | JAX RSS |
|---|---|---|---|
| quick_2species_FPCollisions_noEr | 0.06 s solve | 20 s cold / n.a. warm | ~1 GB |
| tokamak_1species_FPCollisions_noEr_withQN | (from reference sweep manifest) | 4.2 s | 1.6 GB |
| HSX_PAS production (25×115×149×5) | (pending mpiexec sweep) | >2 h (unfinished) | 3.6–5.7 GB |

Full Fortran per-case wall times land in `reference-data-v2/manifest.json` when the
sweep completes; Phase 5 adds the `mpiexec -n {1,2,4,8}` scaling columns.

## Root-cause reading (matches plan §0.2/§2.3 hypotheses)

- The production-resolution stall is consistent with weakly-preconditioned iterative
  solves on a convection-dominated operator: the auto policy's host paths (dense
  shortcut, CSR SuperLU) are size-capped, and the remaining GMRES path lacks a
  complete-factorization-of-simplified-operator preconditioner — precisely what the
  Fortran MUMPS Pmat provides.
- Memory: multiple simultaneous representations of the operator (matrix-free terms,
  CSR copies for host solves, dense fallbacks) are alive at once; the §2.3 design keeps
  one source of truth with three *lazily materialized* consumers.

## Golden-data caveat: monoenergetic_geometryScheme1 transportMatrix[0,1]

The conda PETSc-3.23/MUMPS-5.8 Fortran build fails upstream's own `tests.py` for
`monoenergetic_geometryScheme1` on the [0,1] transport-matrix element only
(+1.62 vs expected −1.08 at solverTolerance 1e-6; +26.3 at 1e-12 — tolerance-unstable,
so the element is ill-conditioned in this configuration/build). Its Onsager partner
[1,0] and [1,1] are fine, and `monoenergetic_geometryScheme11` / `_geometryScheme5`
pass all upstream checks. Parity tests must therefore pin the scheme1 [0,1] element to
upstream's expected value (−1.07986), not to the reference-data-v2 h5. The solvax
block-Thomas RHSMode=3 path reproduces upstream's expected values to 4.2e-6 — the
direct solve is immune to this instability by construction.

## Tier-1 result (RHSMode=3 block-Thomas POC)

Probing the existing matrix-free operator into Legendre bands (mod-3 phase probing,
3m+1 matvecs), null space fixed by exact rank-one absorbed bordering, both drives +
source column in one multi-RHS `solvax` block-Thomas solve; memory-lean variant on
`block_thomas_truncated_fn` (RHS support is exactly l ≤ 2 — padding higher l changes
the transport matrix by 0.0). Existing parity suites 21/21 with the switch ON and OFF
(agreement ~8e-13); warm benchmark: shipped scheme11 0.73 s vs 1.83 s auto (**2.5×**),
25×25×100 block-Thomas 6.6 s at 3.15 GB peak.

## Fortran baseline at production resolution: does not fit either

`HSX_PASCollisions_DKESTrajectories` at the production manifest resolution builds a
**2,512,760 × 2,512,760** system. Fortran + MUMPS at `mpiexec -n 1` on this 24 GB
machine never reached the solve: the factorization drove macOS swap to 46.5/47 GB
(the same swap-exhaustion pattern that crashed the machine in an earlier run) and was killed. Conclusion for
Phase 4/5: at production resolution on a laptop, *neither* code can rely on a global
sparse factorization; the Legendre block-elimination tier (memory O((NθNζ)²) ≈ 66 MB
per 2875² block, independent of Nξ) is the only locally-viable direct path, and the
Fortran strong-scaling baseline must use a case sized to fit (~≤8 GB MUMPS footprint,
e.g. Ntheta=25, Nzeta=51, Nxi=100 PAS, ~450 k unknowns).

## Phase 5.1 Fortran strong-scaling baseline (right-sized case)

`HSX_PASCollisions_DKESTrajectories` at Ntheta=25, Nzeta=51, Nxi=100, Nx=5
(744,610 unknowns), conda PETSc 3.23 + MUMPS, this MacBook (~10 cores, 24 GB):

| MPI ranks | solve time | speedup | parallel efficiency | peak RSS |
|---|---|---|---|---|
| 1 | 463.6 s | 1.00 | — | 3.98 GB |
| 2 | 229.5 s | 2.02 | 101% | 2.86 GB |
| 4 | 240.9 s | 1.92 | 48% | 2.88 GB |
| 8 | 270.5 s | 1.71 | 21% | 1.61 GB |

Fortran/MUMPS saturates at 2 ranks on this machine and *degrades* beyond
(performance/efficiency core asymmetry + MUMPS OpenMP contention). The practical
Fortran floor for this case is ~230 s. That is the concrete G7 target for the
sharded JAX solver, and it also means matching "Fortran at N ranks" gets *easier*,
not harder, as N grows on laptop-class hardware.

## Next measurements

- [ ] Final HSX JAX outcome (converged? wall? peak RSS?).
- [ ] Same case via the solvax block-Thomas tier-1 path (POC in progress).
- [ ] QH bootstrap profile at 25×51×100×4 (queued behind machine load).
- [ ] Fortran `mpiexec -n {1,2,4,8}` strong-scaling baselines (Phase 5.1).

## Phase-4 head-to-head: canonical tier-1 vs Fortran (744k unknowns, HSX PAS DKES RHSMode=1)

Truncated Legendre elimination (`block_thomas_truncated_fn`, blocks assembled on the
fly from the analytic operator coefficients, keep_lowest=3 — exact for every
RHSMode-1 output). Full-band tier-1 would need ~91 GB; the truncated route ~0.3 GB.

| | MacBook M4 CPU | office Xeon (36t) | RTX A4000 | Fortran 1 rank | Fortran 2 ranks |
|---|---|---|---|---|---|
| solve warm [s] | 44.3 (uniform) / **27.2 (ramp)** | 1591 | 45.0 | 463.6 | 229.5 |
| peak RSS [GB] | 0.93 / 1.16 | 1.68 | 1.88 (0.05 GB VRAM buffers) | 3.98 | 2.86 |

- Matched (ramp) discretization: **17x faster than 1-rank Fortran, 8.4x faster than
  its best parallel floor, at ~30% of the memory** — the G3/G7 gates are exceeded by
  nearly an order of magnitude on this case.
- Physics: the direct solve is *more* converged than the Fortran reference — Fortran's
  own electron FSABFlow scatters 51% across its 1/2/4/8-rank runs (KSP rtol=1e-6
  noise); JAX matches the closest Fortran run to 2e-10 and sits inside Fortran's own
  spread on every quantity. Ramp-vs-uniform Nxi differences are <=0.9% (electrons).
- Where time goes: ~100% in the L-scan elimination (42.3 ms/step at TZ=1275; block
  assembly is ~3%). GPU == M4 because the scan is serial in L and A4000 FP64 is 1/32
  rate: GPU upside requires batching over (species, x, surfaces/Er) or fp32 factors
  with fp64 refinement.
- office Xeon XLA-CPU pathology: 36 threads, 36x slower than the M4 on sequential
  1275^2 LU steps — Phase-5 thread budgeting must cap intra-op parallelism per step.

## Bug: tier-2 differentiable adjoint silently wrong on singular FP systems

Full Fokker-Planck with constraintScheme=1 on the flagship-optimization deck
yields a numerically singular system (~5 zero singular values, cond ~2e36); the
tier-2 GCROT adjoint stagnates and the implicit-diff VJP returns a wrong
gradient without any error (AD -1.7e-3 vs FD +2.8e-5 on the affected dof).
PAS+Er tier-2 gradients on the same chain are exact (2.9e-6 vs FD). Fix
direction: surface adjoint-solve convergence in SolveResult and raise/flag when
the adjoint residual misses tolerance; investigate the constraintScheme=1
bordering conditioning in drift_kinetic. A reproducer was filed as a spawned
task from the flagship-example work.

## Mixed precision on GPU: measured negative result (do not auto-engage)

solvax 0.2.0 ships `mixed_precision_block_thomas` (fp32 factor + fp64 refinement).
On RANDOM well-conditioned diagonally-dominant blocks it is 1.79x faster on an
RTX A4000 at 1e-15 accuracy (and 0.22-0.98x, up to 4.5x SLOWER, on CPU — fp32 gives
no CPU throughput gain). BUT on the ACTUAL tier-1 bordered kinetic operators the
result reverses: the constraint-bordered, low-collisionality streaming operators are
too ill-conditioned for float32 (kappa * u_fp32 >> 1), so refinement DIVERGES on every
realistic-sized case (m >= ~169 probed at low Nxi / high collisionality all diverge;
only m <= 49 converge, too small to benefit). The isolated fp32 factorization is ~1.5x
faster on the A4000, but the "large enough to help" and "well-conditioned enough for
fp32" regimes do not overlap for these operators. Conclusion: mixed precision is NOT
the GPU lever for sfincs_jax tier-1 and must not be auto-engaged (it would silently
return non-converged results or always fall back to fp64, a net regression). The tool
remains in solvax for well-conditioned consumers. The real GPU levers here are (a)
better operator conditioning / preconditioning so fp32 becomes viable, and (b)
accepting that the per-subsystem Schur L-scan is inherently serial — GPU parallelism is
across the (species, x, Er, surface) subsystems (already vmapped), not within the scan.

## Production profiling battery 2026-07-11 (tools/benchmarks/profile_production.py)

CPU = the development MacBook (idle), main @ 95e34659, per-case fresh subprocess.

| case | DOFs | method | solve cold / warm | peak RSS |
| --- | --- | --- | --- | --- |
| hsx_pas_dkes_mid (Nxi_for_x ramp) | 336,610 | gcrot (tier-2) | 32.5 s / 15.4 s | **10.8 GB** |
| w7x_fp_2species | 78,628 | gcrot, 115 iters | 17.5 s / 7.6 s | 4.1 GB |
| mono_rhs3_scheme1 | — | block_tridiagonal | 4.8 s e2e | 2.0 GB |
| phi1_newton (4.5k, at the 6000 cap) | 4,548 | unpreconditioned GCROT | **232 s cold** (12,360 inner iters, 3 Newton) / 0.04 s warm | 1.4 GB |
| grad_pas_scheme1 (value_and_grad) | 39,318 | tier-1 differentiable | 5.3 s / 2.1 s | 4.3 GB |
| ambipolar_er_2species | — | Brent root | 31.2 s | 3.9 GB |

Findings and actions:

1. **Ramped decks bypassed tier-1 (fixed same day).** The production-default
   `Nxi_for_x` ramp hit a blanket refusal in `tier1_available`, routing PAS
   production decks to tier-2 GCROT (10.8 GB / 15.4 s warm above). Promoting the
   per-(species,x) truncated kernel (`n_blocks = Nxi_for_x[ix]`, the
   tier1_hsx_head_to_head machinery) into `solve(method="auto")` gives, same
   case: **block_tridiagonal_truncated, peak 885 MB (12.2x less), solve
   4.1 s cold / 3.5 s warm (8.0x / 4.4x faster), e2e 34.9 -> 23.5 s.** AD-vs-FD
   gradient through the ramped route agrees at rtol 1e-6.
2. **Phi1 Newton is the top remaining runtime target.** The inner solve is
   unpreconditioned full-restart GCROT capped at total_size <= 6000; at 4.5k
   DOFs it takes 232 s (12,360 inner iterations for 3 Newton steps). Fix
   design: absorb the Phi1(T*Z)+lambda rows into the tier-1 bordered Schur
   (solvax.BorderedOperator) for PAS decks -> exact direct inner solve, cap
   removed; FP+Phi1 uses GCROT preconditioned by the Phi1-stripped coarse
   operator.
3. **e2e is now operator-build-bound** on the HSX case (7.8 s build vs 3.5 s
   warm solve): geometry + collision-matrix assembly is the next profile
   target after the Phi1 work.
4. **GPU battery invalidated by a stale install**: the office `sfincs-gpu` env
   had an old wheel shadowing the repo (no `sfincs_jax.phi1`/`er`), so all GPU
   numbers ran old code (fixed with an editable install; re-run pending). Under
   the OLD tier-2 route the ramped HSX cases also **OOM'd on the 16 GB A4000**
   (1.4 GiB / 12.1 GiB allocations failed) — the ramp-aware tier-1 (885 MB on
   CPU) is expected to make both HSX cases fit on GPU. One suggestive old-code
   number: the 39k-DOF gradient case ran slower on GPU (6.8 s warm) than CPU
   (2.1 s) — small cases do not amortize GPU dispatch; the serial L-scan
   ceiling note above stands.

## Post-fix GPU + 744k head-to-head (same day, main @ 33c88f7d)

GPU battery re-run (office A4000 16 GB, editable install fixed, ramp-aware
tier-1 active) and the full production HSX case on both backends:

| hsx_pas_dkes_prod (Ntheta=25 Nzeta=51 Nxi=100 Nx=5; 744,610 packed DOFs) | runtime | peak RSS |
| --- | --- | --- |
| Fortran v3 (PETSc+MUMPS, dev MacBook, earlier baseline) | > 2.6 h, unfinished | 3.6-5.7 GB |
| sfincs_jax CPU (dev MacBook) | 41.4 s e2e (25.0 s warm solve) | 1.35 GB |
| sfincs_jax GPU (A4000) | 59.6 s e2e (26.2 s warm solve) | 2.3 GB |

Both backends route block_tridiagonal_truncated; the case previously OOM'd the
16 GB GPU under tier-2. Mid-size HSX warm solve is at CPU/GPU parity (3.5 s vs
3.3 s). Every iterative/small path (FP gcrot 115 iters, Phi1 Newton,
value_and_grad, ambipolar Brent, monoenergetic one-shot) runs 2-5x SLOWER on
the GPU — dispatch-bound serial iterations, consistent with the serial L-scan
ceiling note. Conclusion: the direct truncated tier is the GPU-viable path;
GPU wins need batched work (multi-Er/multi-surface vmaps), not single solves.
