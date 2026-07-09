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

## Next measurements

- [ ] Final HSX JAX outcome (converged? wall? peak RSS?).
- [ ] Same case via the solvax block-Thomas tier-1 path (POC in progress).
- [ ] QH bootstrap profile at 25×51×100×4 (queued behind machine load).
- [ ] Fortran `mpiexec -n {1,2,4,8}` strong-scaling baselines (Phase 5.1).
