# DKX solver literature deep dive: preconditioners, direct solvers, adjoints, Le1/Li1

Date: 2026-09-22. Research only: no code, no PRs.

## Status at pause

The owner paused the work before it finished. What stands:

- **Q1, preconditioners and the `Nx` growth: answered, with one new hypothesis that has strong evidence.** DKX's own ladder is consistent with restart-induced stagnation of GCROT(200, 8) rather than an intrinsic `Nx` wall (§2.1). SFINCS, yancc, MONKES, NEO and NEO-2 were read at source level.
- **Q2, sparse direct at 633,604 unknowns: partly answered.** I read SFINCS's MUMPS settings from source. I derived a structured, MONKES-style dense-block direct estimate (§3). A delegated survey of MUMPS BLR/OOC, SuperLU_DIST, PARDISO, STRUMPACK and cuDSS was still running at the pause and is **not** in this report. Every per-solver number in §3.3 is marked as unverified.
- **Q3, adjoints: answered.** Paul et al. (2019) was read in full text. The IFT formulation and the explanation of the 2.0–2.6× are in §4.
- **Q4, `Le1`/`Li1`: found, high confidence (about 95%).** The paper is E. Lascas Neto, R. Jorge, C. D. Beidler, J. Lion, "Electron root optimisation for stellarator reactor designs", *J. Plasma Phys.* 91(1), E24 (2025), arXiv:2405.12058. In it Le1/Li1 are **L₁₁ᵉ and L₁₁ⁱ**: the SFINCS **RHSMode = 2** `transportMatrix(1,1)` per species. I verified the definitions against the arXiv text and the authors' code. There is one normalisation discrepancy between the paper and its code, flagged in §5.

---

## 1. Executive summary: ranked actions

| # | Action | Where | Expected gain | Memory | Confidence | Effort |
|---|---|---|---|---|---|---|
| 1 | **Stop restarting.** Run GCROT/GMRES with `m ≥ 600–1000` and `k ≈ 50–100` recycled vectors, and measure the unrestarted iteration count on the `Nx` ladder. | SOLVAX (Krylov); DKX (defaults) | At `(Nxi, Nx) = (20, 16)`: 2,788 → plausibly 300–600 iterations (5–9×). On the small deck the unrestarted run clears about 2e-10 at about 240 iterations against 340 restarted (measured by DKX). | `(m+2k)·n·8 B`: about 1 GB at n = 105,600 and about 6 GB at the gap deck (633,604). | Medium-high. The ladder jumps exactly where the count crosses `restart = 200` (§2.1). | Hours |
| 2 | **Diagnose with the right quantities.** Measure the true-residual floor of long unrestarted runs; ‖(A−M)M⁻¹‖₂ by power iteration; the numerical range of AM⁻¹ (Johnson's algorithm, Lanczos on the Hermitian part); and the outlier count against iterations on the 5×5×8 deck. Replace ‖A−M‖/‖A‖, which the record shows points the wrong way. | DKX (probe); SOLVAX (FOV/power-iteration primitives) | Decides between items 3 and 4 before either is built. | Negligible | High that these are the right diagnostics (Greenbaum–Pták–Strakoš; Embree; Campbell et al.) | 1–2 days |
| 3 | **Deflate or recycle the slow modes, NEO-2 style.** Find the few eigenvectors of the iteration operator with \|λ(I − AM⁻¹)\| > 0.5 by Arnoldi and project them out, or use GCRO-DR recycling with k equal to the outlier count, reused across `Er` and optimizer steps. | SOLVAX (GCRO-DR / deflation); DKX (policy) | Removes the outlier phase. The on-deck gain is unknown until item 2. | k·n·8 B | Medium. NEO-2 does exactly this for its collision integral part (§2.4). | Days |
| 4 | **Structured exact direct solve in Legendre with dense (species × speed × angle) blocks**, a generalisation of the MONKES recursion. Eliminate from L = Nxi−1 downward, carry the right-hand sides through, and discard factors. Only L ≤ 2–3 are needed for the fluxes. | SOLVAX (banded-block dense LU with RHS streaming and checkpointing); DKX (ordering, border) | Gap deck: about 2.2e14 flops, roughly 4–8 min at 0.5–1 TFLOP/s, with a working set of about 2 GiB. Compare the projected 14 h / 190 GiB for general sparse LU. | Primal: O(5 b²) ≈ 1.1 GiB. Full factor storage: about 134 GiB (fp64), so the adjoint needs √-checkpointing or recomputation. | Medium. The flop count is solid. Stability without inter-block pivoting is untested, and DKX's per-chain Thomas already shows growth factor 3e9 (§3.1). | 1–2 weeks |
| 5 | **Observable-level custom VJP (IFT).** Backward = one transposed solve on the stored factors + one VJP of `p ↦ A(p) f − b(p)` at fixed f, matrix-free, with stop_gradient on factors and preconditioners. Never differentiate the assembly or the coefficient construction beyond the geometry arrays. | SOLVAX (implicit-diff primitive); DKX (geometry-array interface) | grad/primal 2.0–2.6 → about 1.15–1.35. DKX already measured explicit primal+adjoint at 1.15 (sparse) and 1.31 (structured). | Stores f, λ and the factors. | High for direct routes. Krylov routes floor near 2× unless item 6. | Days |
| 6 | **Krylov-route adjoint at below-2× cost**: simultaneous primal–dual solves (Lu & Darmofal 2003 QMR) or recycling the primal subspace into the transposed solve. | SOLVAX | 2.06 → maybe 1.3–1.5 | Modest | Low-medium (never measured on a DKE) | 1–2 weeks |
| 7 | **Ambipolar root: predictor–corrector continuation.** Predict the root from the IFT sensitivity already computed for the gradient, then do 1–2 Newton corrections whose ∂Jr/∂Er comes from a tangent solve on the same factors. At the root, fold Paul's two adjoints into one combined right-hand side (§4.3). | DKX | Root inside an optimization loop: 9.72× → about 1.3–2× of one solve. A standalone cold root stays at about 3–5 evaluations. | — | Medium-high | Days |
| 8 | **Check whether `Nx = 16` is needed.** The SFINCS manual says `Nx = 5–8` is typical and `Nx` rises only near the Er resonance (E* > 1/3). yancc uses `nx = 5–10`. Lascas Neto used `Nx = 4`. | DKX (resolution ladder) | If `Nx ≤ 11` suffices for the observable, the wall disappears: 174–187 iterations. | — | Deck-dependent | Hours |

The spine of these recommendations: **the measured `Nx` wall is most likely a restart artifact sitting on top of a moderate outlier count.** Fix the Krylov memory first (item 1), measure the right quantities (item 2), and only then build preconditioners (items 3–4).

---

## 2. Q1: preconditioners and the speed coupling

### 2.1 The most likely reading of DKX's own data: restart stagnation

DKX's `Nx` ladder at `Nxi = 20` (DKX `docs/experiments/2026-09-19-nx-drives-the-iteration-growth.md`): GCROT with `restart = 200` and `recycle_dim = 8` took 174, 187, **3,810 (failed)**, 969, 1,186, 1,970 and 2,788 iterations for `Nx` = 10…16. At `Nxi = 40, Nx = 10` it took 157.

- **Every converged run below 200 iterations is cheap. Every run that needs more than one cycle explodes.** The break falls exactly between `Nx = 11` (187) and `Nx = 12` (3,810). That is the classical loss of superlinear convergence under restarting. Restarted GMRES discards the Ritz information that deflates the outlying eigenvalues and can stagnate indefinitely:
  - van der Vorst & Vuik, *J. Comput. Appl. Math.* 48 (1993) 327–341, on superlinear convergence;
  - Embree, "The tortoise and the hare restart GMRES", *SIAM Rev.* 45 (2003) 259–266;
  - Baker, Jessup & Manteuffel, *SIMAX* 26 (2005) 962 (LGMRES);
  - Morgan, *SISC* 24 (2002) 20–37 (GMRES-DR).
- **The small deck corroborates it.** DKX `2026-09-20-the-balanced-solve.md`, deck (5,5,8), `Nx = 16`:
  - restarted: 340 iterations;
  - one unrestarted cycle: residual 3.3e-2 at 113, 1.3e-4 at 160, 2.2e-10 at 240;
  - unrestarted growth from `Nx` 10 → 16 is therefore about 2.1× (113 → about 240), not the 16× of the restarted gap deck;
  - the residual curve (a slow phase, then fast convergence) is the "outliers first, then the cluster" shape of Campbell, Ipsen, Kelley & Meyer, "GMRES and the minimal polynomial", *BIT* 36 (1996) 664–675.
- **The outlier count matches the iteration count on the small deck.** DKX measured 96.7% → 94.3% of eigenvalues within 0.5 of 1. That leaves about 3.3% of 4,004 ≈ 132 outliers at `Nx = 10` (113 iterations) and about 5.7% of 6,404 ≈ 365 at `Nx = 16` (340 restarted, about 240 unrestarted). This is a correlation, not a proof. It is exactly what the Campbell et al. bound predicts (iterations ≈ number of outliers + log(tol)/log(cluster radius)).
- **SFINCS does not restart.** It calls `KSPGMRESSetRestart(KSPInstance, 2000)` (`sfincs/fortran/version3/solver.F90:151`), and yancc restarts only after 150 inner steps inside GCROT (yancc paper §4.5). DKX's 200/8 is the outlier.
- **Watch the attainable-accuracy floor.** The unrestarted base arm stalls at 1.9e-10 against `tol = 1e-10` at `Nx = 16` on the small deck (same record). Right-preconditioned GMRES with an ill-conditioned M (DKX measured chain conditions up to 1.8e6 and backward errors rising about 3× per speed point) has a floor that grows with `Nx`. A tolerance near that floor makes the solve wander, which may be the `Nx = 12` failure. Measure the floor, and either set `tol` above it or use an outer refinement loop that uses the true residual (GMRES-IR; Carson & Higham, *SISC* 40 (2018) A817–A847).

**Recommendation.** Before any new preconditioner, rerun the ladder with unrestarted FGMRES, or GCROT with m ≥ 1000 and k ≈ 50–100. Record the true residual at every step and the stagnation floor. Memory is not the constraint: 1,000 vectors × 633,604 × 8 B ≈ 5 GB.

### 2.2 Why "a closer M converged worse" is not paradoxical

DKX found that keeping the speed triangle over every species pair drove ‖A−M‖/‖A‖ from 1.0 to 4.3e-4 yet cost 17% more iterations (6,786 against 5,799).

- For right preconditioning the relevant operator is AM⁻¹ = I + (A−M)M⁻¹. A small ‖A−M‖ says nothing when ‖M⁻¹‖ is huge in some directions. Here it is: near-collisionless high-speed electron chains, where ν_D ∝ x⁻³.
- The quantity to measure is ‖(A−M)M⁻¹‖₂. If it is below 1, GMRES converges at least at that rate. Also measure the numerical range W(AM⁻¹):
  - Elman / Eisenstat–Elman–Schultz, *SINUM* 20 (1983) 345–357;
  - Starke, *Numer. Math.* 78 (1997) 103–117;
  - Beckermann–Goreinov–Tyrtyshnikov, *SIMAX* 27 (2006) 772–778;
  - Crouzeix & Palencia, *SIMAX* 38 (2017) 649–655: W(A) is a (1+√2)-spectral set, so if 0 ∉ W(AM⁻¹) the convergence rate is bounded by the geometry of W.
- Eigenvalues cannot settle it. Greenbaum, Pták & Strakoš, "Any nonincreasing convergence curve is possible for GMRES", *SIMAX* 17 (1996) 465–469, and Arioli, Pták & Strakoš, *BIT* 38 (1998) 636–643, show that any GMRES curve is compatible with any spectrum. This is why the eigenvector-conditioning probe was uninformative. Trefethen & Embree, *Spectra and Pseudospectra* (Princeton, 2005), ch. 26, and Embree, "How descriptive are GMRES convergence bounds?" (Oxford NA report 99/08, 1999), give the pseudospectral bounds.
  - Tooling: matrix-free pseudospectra via the projected Hessenberg matrix (Wright & Trefethen, *SISC* 23 (2001) 591); numerical range by Johnson's method (*SINUM* 15 (1978) 595–602), i.e. the largest eigenvalue of the Hermitian part of e^{iφ}AM⁻¹ over φ, by Lanczos.
- **A physics-natural inner product exists and was not the one balanced.**
  - Paul et al. 2019 §3.1, eq. (3.1): the inner product ⟨F,G⟩ = Σ_s⟨∫d³v f g / f_M⟩ + sources (the free-energy norm). In it the linearized FP operator is self-adjoint (equal temperatures). The DKES-trajectory streaming/drift operator is anti-self-adjoint, eq. (3.13). The full-trajectory model breaks this only through the non-divergence-free E×B term, eq. (3.14).
  - In that norm A = S + K, with S the symmetric, semidefinite collisions and K skew. This is the structure for which FOV bounds are sharp and weighted GMRES is justified (Essai, *Numer. Algorithms* 18 (1998) 277; Güttel & Pestana, *Numer. Algorithms* 67 (2014) 733; Pestana & Wathen, *JCAM* 249 (2013) 57).
  - DKX's "balanced solve" used a LAPACK balancing diagonal d ≈ x^-1.7 e^{-0.6x²}. That is close to, but not, this weight: the proper weight also carries the quadrature weights, the Legendre normalisation 2/(2L+1) and the flux-surface Jacobian.
  - Test only this weight, and only as a probe: measure whether 0 ∉ W(AM⁻¹) in it. DKX showed that a pure diagonal similarity does not move iterations. It changes the norm GMRES minimises, not the M that is dropped.

### 2.3 Where the `Nx`-dependent scaling comes from (explained, not a fix)

- Landreman & Ernst, *J. Comput. Phys.* 243 (2013) 130, arXiv:1210.5289 §2. The SFINCS/DKX collocation represents f on Maxwell-polynomial nodes, with the differentiation matrix built from interpolants times the weight e^{−x²}. So D = W(D_poly − 2X)W⁻¹ with W = diag(e^{−x_i²}).
- cond(ddx) should therefore scale like e^{x_max²}. DKX's data: x_max = 4.26 → 5.68 gives e^{5.68²−4.26²} = e^{14.1} ≈ 1.4e6. DKX measured cond(ddx) going from 2.5e6 to 2.5e12, a ratio of 1e6. The "decade per grid point" is x_max² growing about 2.36 per node.
- This confirms DKX's finding that the growth is a scaling artifact, removable by Ruiz. It also shows that **any preconditioner that is block-diagonal in nodal x is invariant to it**: dropping off-diagonal x entries commutes with diagonal x-scaling. Only a *non-diagonal* change of speed basis changes which coupling the preconditioner drops (§2.5).

### 2.4 How the other codes treat speed coupling

| Code | Speed treatment | Solver | Relevance |
|---|---|---|---|
| **SFINCS v3** | Maxwell-polynomial collocation (`xGridScheme = 5`); `includeXDotTerm = .true.` by default with a spectral d/dx (`xDotDerivativeScheme = 0`) (`globalVariables.F90:144,180,183`) | GMRES, restart 2000, right-preconditioned by a MUMPS LU of a simplified matrix. `preconditioner_x = 1` (default) keeps only the x-diagonal of the collision *and* xDot blocks (`createGrids.F90:950–990`, `populateMatrix.F90:2101–2145`). `preconditioner_x_min_L` keeps the full x coupling for L < min_L. `preconditioner_species = 1`, `preconditioner_xi = 1` (drops L±2). MUMPS `CNTL(1) = 1e-6`, `ICNTL(14) = 50`, doubled on failure up to 1024 (`solver.F90:58–82, 363–417, 452–464`). | The manual (`doc/manual/version3/inputParameters.tex:1610–1660`, `resolution.tex:40–70`) calls `preconditioner_x = 1` "strongly recommended, except perhaps at high collisionality", and gives typical `Nx = 5–8`, rising only near the Er resonance (E* > 1/3; HSX with small Ti/Te is named as the exception). **SFINCS has no published experience at `Nx = 16`.** DKX has not recorded a test of `preconditioner_x_min_L` = 1–3 (full dense speed coupling at L ≤ 2, where the field-particle terms matter). It is cheap: dense Nx×Nx blocks only at low L. |
| **yancc** (Conlin & Landreman, arXiv:2607.20861; source read at f0uriest/yancc@29b1f26, 2026-09-15) | Maxwell-polynomial collocation, `nx = 5–10` ("we find little benefit in coarsening x", §4.1). FD in α, θ, ζ with a widened, diagonally dominant upwind stencil (Table 1). | Right-preconditioned GCROT with inner GMRES of 150 (§4.5). Preconditioner: geometric multigrid V-cycle, **semi-coarsened (x never coarsened)**. Smoother: damped (ω = 0.6) block-Jacobi applied as 4 sweeps, each keeping full coupling along one of (x, α, θ, ζ) (§4.2; `smoothers.py`, `DKEJacobiSmoother` with axorder ending in `x`, and a separate `DKEJacobi2Smoother` "keeping coupling in s,x"). Coarse grid: dense LU at a few thousand DOF. | Speed coupling is kept **exactly, locally** inside the x-line smoother at every point, never dropped globally. **The paper reports no `Nx` scaling study**; benchmarks use `nx = 7`. Measured: about 5× faster than SFINCS (128 cores vs one A100), up to about 100× at high ν "where SFINCS slows down significantly", which they attribute to `preconditioner_x` dropping energy scattering (§5.2). Memory 6 GB against more than 50 GB. |
| **MONKES** (Escoto et al., *Nucl. Fusion* 64 (2024) 036009, arXiv:2312.12248; local `MONKES/doc/NF-paper/monkes-paper.tex` §3) | Monoenergetic, so no speed coupling. PAS only. | Exact block-tridiagonal Gaussian elimination in Legendre index with dense N_fs × N_fs Fourier blocks. Eliminates k = Nξ−1 → 0 (Schur complements Δ_k, eqs. 33–34). Stores only Δ₀…Δ₂, because the D_ij need only modes 0–2. Cost O(Nξ N_fs³), measured C_alg ≈ 7e-11 s·N_fs³ per mode on one Xeon core (§4.2). Invertibility of Δ_k is proven in Appendix C. | The template for action 4. It differs from DKX's problem: monoenergetic, tridiagonal, no speed or species coupling. |
| **NEO** (gacode, `neo/src`, read locally) | Spectral in energy: `LAGUERRE_METHOD = 1` default (L^{1/2} for l = 0, x·L^{3/2} for l ≥ 1); option 2 is the Sonine basis x^l L_k^{(l+1/2)}(x²) (`neo_energy_grid.f90:58–88`). Typical `N_ENERGY = 6`. | Sparse direct only: UMFPACK default, with SuperLU/PETSc/CUDA options (`neo_sparse_solve.F90`). | In the Sonine basis the x-multiplying streaming term maps x^l → x^{l±1}·poly(x²), so it is **banded in k**. The collision test-particle part is near-diagonal. There is no e^{x²} dynamic range. |
| **NEO-2** (itpplasma/NEO-2, `NEO-2-QL/ripple_solver_ArnoldiOrder2_test.f90:4519–4533`, `arnoldi_mod.f90:1–60, 81–82`) | Associated Laguerre basis by default (`collop_definitions.f90:23`) | Preconditioned Richardson: Lbar = (L_V − L_D)⁻¹ by a SuiteSparse sparse direct solve (Vlasov + *differential* collision part, including energy diffusion). The **integral (field-particle) part is iterated**. Arnoldi finds the iteration-operator eigenvalues with \|λ\| > `tol0 = 0.5` and applies P φ_m = φ_m/(1−λ_m) to those "bad modes". | **The direct precedent for action 3**: factor the local and differential speed coupling, and deflate the handful of slow modes from the nonlocal integral part. |
| **DKES / KNOSOS** | Monoenergetic (DKES: variational, block-tridiagonal Legendre-Fourier); bounce-averaged (KNOSOS, Velasco et al., *J. Comput. Phys.* 418 (2020) 109512). PAS only. | — | No speed coupling to learn from. KNOSOS solver details were not verified before the pause. |

**Is the `Nx` growth a known phenomenon?** Not as such in the neoclassical literature. SFINCS avoids it by staying at `Nx ≤ 8`, yancc does not test it, and MONKES and DKES have no speed variable. The generic phenomenon is well known: iteration growth with resolution when the preconditioner drops a coupling whose relative strength grows with that resolution, amplified by restarts (§2.1).

### 2.5 Preconditioner families that target speed coupling specifically (ranked)

1. **Deflation or recycling of the slow modes** (NEO-2's Arnoldi projection; GCRO-DR, Parks, de Sturler et al., *SISC* 28 (2006) 1651–1674; framework in Gaul, Gutknecht, Liesen & Nabben, *SIMAX* 34 (2013) 495–518). The subspace is reusable across `Er` points and optimizer steps. SOLVAX primitive.
2. **Keep the full dense speed/species coupling only at L < L_c** (SFINCS `preconditioner_x_min_L`). The field-particle (Rosenbluth) terms are largest at L = 0, 1, 2, where they carry the energy- and momentum-restoring structure. Cost: dense (Ns·Nx)-coupled Thomas only on L_c rows. DKX, cheap to test.
3. **An x-line block smoother inside a two-level or multigrid method** (yancc §4.2). It keeps speed exact locally and angle/pitch approximately. DKX already has `multigrid.py`.
4. **Change the speed basis for the preconditioner only.** Build M block-diagonal in the Sonine index k rather than the node index (NEO option 2). Apply it as T⁻¹ blockdiag_k(T A T⁻¹) T, with T the Nx×Nx nodal-to-modal transform. Drops then happen in a basis where collisions are near-diagonal and there is no e^{x²} range. The cost: streaming is no longer diagonal (it becomes banded in k for the Sonine basis). Speculative, and untested anywhere for a DKE preconditioner.
5. **Additive or multiplicative Schwarz over speed.** In a nodal basis this is the SFINCS family (`preconditioner_x` = 2–4), which DKX measured: banding is useless, and the triangle wins 1.85× only at `Nx = 16` (`2026-09-18-speed-triangle-back-substitution.md`). Deprioritise.

---

## 3. Q2: sparse direct at this scale

### 3.1 Structured direct in Legendre (MONKES generalisation): the estimate

- **Block size.** The gap deck is 633,604 = 2 species × 16 speeds × 120 L × 165 angle points + 4 border unknowns. The per-L block is b = Ns·Nx·NθNζ = 5,280.
- **Coupling in L.** With the `Er` xDot/ξDot and tangential-drift terms, L is coupled pentadiagonally (SFINCS `populateMatrix.F90:960–1060`; DKX `plan.md` §"block-pentadiagonal structured solve").
- **Flops.** A block-banded LU with block bandwidth 2 costs about 12.7 b³ per L ≈ 1.9e12 per L, or **≈ 2.2e14 in total**. Tridiagonal case (Er = 0, no tangential drifts): about 4.7 b³ per L ≈ 8e13.
- **Time.** About 4 min at 1 TFLOP/s of DGEMM-dominated work on a multicore CPU. On one A4000 at nominal FP64 (the GA104 FP64 rate is 1/32 of FP32, about 0.6 TFLOP/s; not independently verified before the pause), about 6–8 min.
- **Memory for the primal.** Carry all right-hand sides (3 for RHSMode 2, plus the 4 border columns eliminated by a small Schur complement) through the elimination from L = Nxi−1 downward, as MONKES does. Discard each block once used. Keep only L ≤ 2–3 for the moments. Working set is about 5 b² × 8 B ≈ 1.1 GiB.
- **Memory for the adjoint and gradient.** These need full f and λ over all L. Full factor storage is about 134 GiB (fp64), so use √N_ξ checkpointing (about 11 stored Schur states ≈ 10 GiB at roughly 2× flops) or spill to NVMe.
- **Risk 1, stability.** DKX's per-(s,x) block-Thomas already shows growth factors of about 3e9 on high-speed electron chains (`2026-09-19-nx-drives-the-iteration-growth.md`), the signature of block elimination without pivoting across blocks (Demmel, Higham & Schreiber, *NLAA* 2 (1995) 173–190). Mitigations:
  - eliminate in MONKES's direction, from high L (collision-dominated, diagonally dominant as ν L(L+1) grows) toward L = 0;
  - or use a band LU with partial pivoting within the block band (dgbtrf-like; similar flops, band storage about 1.5×).
  - **Open:** DKX's `coarse_precond.py` iterates `for ell in reversed(range(...))` (lines 862, 930, 983). I did not finish checking which sweep is the factorization, so whether DKX already eliminates high-to-low L is unconfirmed.
- **Risk 2, scaling with the angular grid.** This route scales as (NθNζ)³. It is cheap here only because NθNζ = 165. At W7-X production grids (NθNζ ≈ 1,000–2,000) b grows 6–12× and cost 200–1,700×. There, nested dissection in angle inside each block (MUMPS-like), or BLR compression of Δ_L (already a DKX research bet), is required.

### 3.2 Why MUMPS OOMs where this does not

The flops are similar. Nested dissection separates the L direction with separators of size about 2b, so fronts of about 10⁴. But MUMPS **stores the whole factor**: roughly n × (average front) × 8 B ≈ 50–130 GB. SFINCS's direct arm was killed at 22.5 min on 36 GiB, and `preconditioner_x = 2` at 30.6 min (DKX `2026-09-19-sfincs-on-the-gap-deck.md`). MUMPS out-of-core (ICNTL(22) = 1) writes factors to disk and should fit the 62 GiB host. SFINCS never sets ICNTL(22), ICNTL(35) (BLR) or an ordering: only CNTL(1) = 1e-6, ICNTL(4) = 2 and ICNTL(14) (`solver.F90:388–417`).

### 3.3 Solver-by-solver survey: **incomplete at pause**

The delegated survey was still running at the pause. Its questions: BLR complexity and measured gains (Amestoy, Buttari, L'Excellent & Mary, *SISC* 2017 and *ACM TOMS* 2019); MUMPS OOC; SuperLU_DIST 3D and GPU; PARDISO OOC and matching (iparm 11/13); STRUMPACK HSS/HODLR/BLR on GPU; cuDSS hybrid memory and multi-GPU; A4000 FP64. **No claim in this section beyond §3.1–3.2 is verified.** DKX's own measurements stand: PARDISO needs Ruiz equilibration (201 s at residual 9.4e-2 unequilibrated, 103 s at 1.2e-10 equilibrated), and MUMPS scales and matches by default.

---

## 4. Q3: adjoint and implicit derivatives

### 4.1 Paul, Abel, Landreman & Dorland, *J. Plasma Phys.* 85 (2019) 795850501, arXiv:1904.06430 (read in full)

- **Two adjoints.**
  - **Continuous** (§3.1): in the free-energy inner product, eq. (3.1), the adjoint operator is eq. (3.11). For DKES trajectories L₀† = −L₀ (anti-self-adjoint, eq. 3.13). For full trajectories L₀† = −L₀ + (q/T)(dΦ/dψ) v_m·∇ψ (eq. 3.14). It needs equal species temperatures, or the Appendix B modification.
  - **Discrete** (§3.2): the matrix transpose.
- **The choice is the discrete adjoint.** It is exact for the discretized problem, and it **reuses the forward preconditioner's LU factors** (Aᵀ = UᵀLᵀ). The continuous adjoint needs its own preconditioner factorization, because "we do not obtain convergence when the same preconditioner is used for both problems" (§3.2).
- **Accuracy against forward differences.** Error falls as ΔB₀₀ down to about 1e-4 relative step. The discrete adjoint reaches lower minimum error. The continuous adjoint is within ≤ 0.1% at (41 θ, 61 ζ, 85 ξ, 7 x) (§4, Fig. 1).
- **Cost** (measured, NERSC Edison, 48 cores).
  - One forward + one adjoint solve per objective, independent of N_Ω. But "when N_Ω is large, the cost … is dominated not by the linear solve but by constructing ∂S/∂Ω and ∂L/∂Ω and computing the inner product. Thus the cost still scales with N_Ω" (§3.1).
  - Speed-up ≈ 50× at large N_Ω for fixed Er (§4, Fig. 2b) and ≈ 200× at fixed ambipolarity (§5.3.2, Fig. 6a).
  - Parameters: Ω = {B^c_mn, I, G, ι} in Boozer coordinates. All geometry enters through B(θ,ζ), G, I, ι (§4, eqs. 4.1–4.4).
- **Ambipolar Er.**
  - (a) Newton root-finding with ∂Jr/∂Er from the adjoint: 14% wall-clock saving over Brent on W7-X (§5.3.1, Fig. 5).
  - (b) Derivatives at fixed Jr = 0 cost **one extra adjoint solve**, L†q^{Jr} = J̃r (eqs. 5.11–5.12). The final formula is App. E eq. (E7), the IFT expression dR/dΩ|_Jr = ∂R/∂Ω|_Er − (∂R/∂Er)(∂Jr/∂Ω)/(∂Jr/∂Er), with every partial taken from q^R and q^{Jr}.
  - The stated limitation: the root must persist. Fall back to fixed Er when \|∂Jr/∂Er\| is small (§5.3.2).
- **Follow-ups.** Shape gradients: Antonsen, Paul & Landreman, *J. Plasma Phys.* 85 (2019) 905850207; Paul, Antonsen, Landreman & Cooper, *J. Plasma Phys.* 86 (2020) 905860103. The neoclassical shape gradient needs one more MHD adjoint (§5.2.2 of the 2019 paper).
- **yancc's strategy** (source). `jax.lax.custom_linear_solve(A.mv, b, solve, transpose_solve)` around GCROT (`krylov.py:816`). The transposed solve uses the transposed multigrid preconditioner. `_freeze_preconditioner` applies stop_gradient to all preconditioner arrays, "freeing AD from building the (unused) factorization derivatives" (`solve.py:32–43`). The paper verifies no derivative numerically; DKX `plan.md` already flags this.

### 4.2 Why `jax.grad` costs 2.0–2.6× here, and the formulation that reaches about 1.2×

DKX's record (`2026-09-20-one-factorization-many-solves.md`) measured:

| Route | grad / primal | Explicit primal + adjoint |
|---|---|---|
| Structured direct | 2.57 | 1.31 (median; 1.11–1.54 across campaigns) |
| Recycled Krylov | 2.06 | — |
| Sparse direct | — | 1.15 |

It found exactly one factorization under grad. **The linear algebra already meets the 1.3 gate. The overhead is AD over everything else.**

1. **Reverse mode's forward is the primal plus residual storage.** `jax.grad` = linearize + transpose. It runs the forward once, saving residuals of every traced op that depends on the differentiated inputs. It is not literally a second forward unless something forces one: `jax.checkpoint`/remat, a `custom_vjp` whose fwd does not save f, or computing value and grad in separate calls. Use `jax.value_and_grad`.
2. **Tape volume.** The backward transposes the linearization of every coefficient-construction op that depends on p: collision matrices, Rosenbluth kernels, geometry-to-grid transforms, assembly by compression (8,800 operator products on the gap deck, about 1.2 min). Its cost is about 1–3× those ops (the cheap-gradient bound; Griewank & Walther, *Evaluating Derivatives*, 2nd ed., SIAM 2008, ch. 4). When the factorization does not dominate the primal, as on DKX's small benchmark decks, this is 1.0–1.6× the primal, exactly the measured residue.
3. **Formulation that removes it.** Use an observable-level `custom_vjp` (the IFT). The DKE operator is multilinear in a small set of geometry arrays g = {B, ∂θB, ∂ζB, B_θ, B_ζ, their derivatives, G, I, ι} on the (θ,ζ) grid (Paul §4), and in species parameters.
   - fwd: build the coefficients with stop_gradient everywhere except g; factor (stop_gradient); solve f; compute R(f, g). Save f and the factors.
   - bwd: λ = A⁻ᵀ ∂R/∂f (0.15 primal on the sparse route, measured). Then ḡ = ∂R/∂g − VJP_g[g ↦ A(g) f − b(g)](λ). That is **one** matrix-free operator-apply VJP at fixed f, **never** the assembly's VJP: 8,800 products versus 1.
   - Chain ḡ to Boozer or VMEC parameters outside DKX (booz_xform_jax / vmec_jax VJPs).
   - This also removes Paul's "cost still scales with N_Ω" term, which came from forming ∂L/∂Ω per parameter. Reverse mode gets λᵀ(∂A/∂g)f for all g in one pass.
   - Expected cost ≈ 1 + 0.15 + (a matvec-VJP ≈ 0.02–0.1) ≈ **1.2×** on direct routes. Use `jax.checkpoint` policies only for the geometry pre-processing (`save_anything_except_these_names` or `everything_saveable`: spend memory, not recompute).
4. **Krylov routes have a floor near 2×**, because the adjoint is a second, cold Krylov solve of equal difficulty (DKX: cold-started GCROT on the transpose). Options:
   - recycle the primal Krylov/deflation subspace into the transposed solve;
   - solve primal and dual simultaneously with a Lanczos-based method (Lu & Darmofal, "A quasi-minimal residual method for simultaneous primal-dual solutions and superconvergent functional estimates", *SISC* 24 (2003) 1693–1709);
   - or use a direct route for the gradient.

### 4.3 The ambipolar root

- **Gradient at the root.** DKX's `er.py:960–1060` uses `jax.lax.custom_root`, so the IFT term dEr/dp = −(∂Jr/∂Er)⁻¹ ∂Jr/∂p comes from autodiff of `radial_current`. Its VJP then runs through the full DKE solve again (point 2 above). Better:
  - Form Paul's (E7) with **one** adjoint solve. With c = (dR/dEr)/(dJr/dEr), solve Aᵀλ = ∂R/∂f − c ∂Jr/∂f. Take c from one *tangent* (forward-mode) solve in Er: A f_Er = ∂b/∂Er − (∂A/∂Er) f. The final acceptance JVP already computes that solve (`er.py`: `jax.jvp(f, (x_root,), ...)`). Store f_Er instead of discarding it.
  - Total extra cost at the root: one adjoint solve on stored factors plus one operator VJP.
- **The root itself** (9 evaluations = 9.72×; DKX's own analysis says only a 2–3 evaluation continuation can reach 1.5×).
  - Inside optimization, predict Er_new = Er_old + (dEr/dp)·Δp from the IFT sensitivity already computed for the gradient, then take 1–2 safeguarded Newton steps with tangent-solve slopes. Er enters A and b affinely, so each Newton step is one solve plus one tangent solve on the same factors.
  - For a standalone cold root, a Hermite-interpolatory reduced model of Jr(Er) = cᵀ(A₀ + Er A₁)⁻¹(b₀ + Er b₁) from 2 solves and their adjoints/tangents gives an accurate rational prediction (Baur, Beattie, Benner & Gugercin, *SISC* 33 (2011) 2489–2518). That is about 3–4 solves in total.

---

## 5. Q4: `Le1` / `Li1`, found

- **Citation.** E. Lascas Neto, R. Jorge, C. D. Beidler, J. Lion, "Electron root optimisation for stellarator reactor designs", *J. Plasma Phys.* 91(1), E24 (2025), doi:10.1017/S0022377824001466, arXiv:2405.12058. Code: github.com/eduardolneto/Electron-Root-Optimisation.
- **The paper never writes "Le1"/"Li1".** The quantities are **L₁₁ᵉ and L₁₁ⁱ**.
- **Definitions** (arXiv numbering; verified against the arXiv text).
  - Thermal coefficients, eqs. (27–29): L₁ₚᵃ = −⟨∫d³x π^{−3/2} f₁ₚᵃ **v**_d·∇ψ⟩.
  - Energy convolution of monoenergetic D_ij, eq. (31): L_ijᵃ = (2/√π)∫dx² D_ij(x) x e^{−x²} h_i h_j, with h₁ = h₃ = 1 and h₂ = x² (Beidler et al., *Nucl. Fusion* 51 (2011) 076001).
  - Radial coordinate ψ (toroidal flux) in the theory. Optimisation surfaces ρ = √s = r/a = 0.2, 0.29, 0.35.
- **Objective**, eq. (46): J₂ = Σ_j w_j [ L₁₁ₛⁱ/L₁₁ₛᵉ × (T_e² v_thⁱ)/(T_i² v_thᵉ) ]².
- **How it is actually computed** (verified in `Optimisation_Functions/neo_er_single.py`).
  - SFINCS v3 **RHSMode = 2**, **one single-species run per species**, reading `transportMatrix[0,0]` (`get_transport_matrix`, lines 427, 550). `mHats = THats = 1` (reference quantities = the species'). `collisionOperator = 1` (PAS, "Only Lorentz operator").
  - DKES trajectories by default: `useDKESExBDrift = true`, `includeXDotTerm = false`, `includeElectricFieldTermInXiDot = false`. The paper states Nθ = 25, Nφ = 31, Nξ = 34, Nx = 4 and eEr/T = 1.
  - Consequence: species and speeds decouple. L₁₁ is SFINCS's *internal* Nx-point Maxwell-quadrature energy convolution of PAS monoenergetic solves. It is not a separate RHSMode = 3 table. RHSMode 3 was used only to build the DKES-format validation database (App. A).
- **Mapping to DKX.**
  - Le1 = DKX's RHSMode-2 `transportMatrix[0,0]` for a single-species electron run (Z = −1, m = 5.445e-4 m_p). Li1 is the same for the ion.
  - SFINCS normalisation (`diagnostics.F90:770`): transportMatrix(1,1) = 4/Δ² · √(T̂/m̂) · Z² · (G + ιI) · particleFlux_vm_psiHat · B₀/(T̂² Ĝ²).
  - This is DKX's structured-direct PAS route (exact, cheap). For parity, match `Nx = 4`, PAS and the DKES-trajectory flags. For physics, converge `Nx` separately.
- **Discrepancy to settle before reuse.** The code's mode-0 objective is `J = L11_I/L11_e * sqrt(m_I/m_e)` (line 285). The printed eq. (46) applies (T_e² v_thⁱ)/(T_i² v_thᵉ), which is √(m_e/m_i) at equal T, **the inverse factor**. The subagent's derivation from the SFINCS normalisation supports the code, which would make eq. (46) a typo. I have not verified this independently, so DKX should derive it from `diagnostics.F90` itself.
  - Further, the electron-only helper uses `Er = 1e-5` (line 178), while the joint objective passes the same Er to both species. The sign is flipped for electrons only in the flux helper (`sfincs_fluxes(Lij_e, -Er, ...)`).
  - Choosing the ratio versus its inverse, or the mass factor, silently changes the objective. Pin it with a test against the paper's reported values.

---

## 6. Method table

| Method | Applies to DKX operator? | Expected gain | Memory | GPU? | Maturity | Dependency cost |
|---|---|---|---|---|---|---|
| Long-memory GMRES/GCROT (m ≥ 1000) | Yes, directly | 5–9× iterations at Nx = 16 (inferred) | +1–6 GB | Yes | Textbook | None (SOLVAX) |
| FOV / ‖EM⁻¹‖ / pseudospectra diagnostics | Yes | Diagnostic | Small | Yes | Mature (EigTool ideas) | None |
| Arnoldi deflation / GCRO-DR recycling | Yes | Removes outlier phase; reusable across Er | k·n | Yes | Mature; NEO-2 production precedent | None (SOLVAX) |
| SFINCS `preconditioner_x_min_L` = 1–3 | Yes | Unknown; cheap test | Dense Nx blocks at low L | Yes | In SFINCS since v3 | None |
| Structured dense-block Legendre direct (MONKES-generalized) | Yes on small angular grids | 14 h → minutes (estimate) | About 1 GiB primal; checkpointing for the adjoint | Yes (DGEMM) | MONKES-proven for the monoenergetic case only | SOLVAX kernel |
| yancc-style semi-coarsened MG + x-line smoother | Yes (FD angles needed) | yancc: flat in ν at nx = 7; untested at Nx = 16 | Low | Yes | Published 2026 | DKX `multigrid.py` exists |
| Sonine-basis preconditioner | Speculative | Unknown | Low | Yes | Untested for a DKE preconditioner | DKX |
| MUMPS OOC / BLR | Yes (via PETSc or an adapter) | Makes the gap deck fit (inferred); survey incomplete | Disk ≈ 50–130 GB | No (CPU) | Mature | MUMPS build |
| Observable-level custom VJP (IFT) | Yes | grad 2.0–2.6 → about 1.2× | f, λ, factors | Yes | Standard (Paul 2019; custom_linear_solve) | None |
| Primal–dual QMR | Krylov routes | Below 2× for the gradient | Modest | Yes | Published 2003, niche | SOLVAX |
| Newton continuation for the root | Yes | 9.72× → about 1.3–2× inside optimization | — | Yes | Paul 2019 (Newton) | DKX |

## 7. Open questions

1. The unrestarted iteration count on the gap-deck ladder (the decisive test of §2.1), and the attainable residual floor at each `Nx`.
2. Whether AM⁻¹'s outlier eigenvectors concentrate on the high-speed electron chains. Project them onto the (species, speed) chains on the 5×5×8 deck.
3. The elimination direction in DKX's block-Thomas (`coarse_precond.py:862/930/983`) and whether eliminating high L to low L removes the growth factor of about 3e9.
4. Whether `Nx = 16` is required on the HSX gap deck for each observable (E* > 1/3?).
5. The whole Q2 per-solver survey: BLR gains on kinetic operators, MUMPS OOC time, SuperLU_DIST 3D/GPU, STRUMPACK, cuDSS on the A4000, and verified A4000 FP64 throughput.
6. The L₁₁ normalisation discrepancy in Lascas Neto eq. (46) against the code.
7. KNOSOS's and DKES's linear-solver details (not verified).

## Addendum: the sparse-direct survey of question 2, completed the same day

The per-solver survey that section 3.3 marks incomplete finished after the pause. Its conclusions, with the one point checked against DKX's own measurements:

- **Memory, not time, is the wall for a full-rank factorization of the gap deck.** The graph is a one-dimensional chain of thick slabs in the Legendre index: one slab is `2 · 16 · 165 = 5,280` unknowns, a pentadiagonal pair about 10,560, and there are about 60 block steps. Block elimination along that chain costs about `3.3e14` flops, minutes to an hour at the flop rates the 66,004-unknown run implies; an explicit sparse-LU factor holds about 150 GiB in float64, and an implicit block LU keeping only the diagonal-block factors about 50 GiB in float64 or 25 GiB in float32. This is the same order as section 3.1's estimate, which assumed smaller blocks.
- **Only two routes fit 62 GiB.** MUMPS with block low-rank compression and out-of-core storage (`ICNTL(35)=2`, `CNTL(7)` swept over 1e-10 to 1e-8 on the scaled matrix, `ICNTL(22)=1`, `ICNTL(23)` about 56,000 MB, matching `ICNTL(6)=5` with centralized input), or a MONKES-style Legendre-pair block elimination eliminating from high `l` down, which needs only a few blocks of working memory when only the `l ≤ 2` moments are wanted. Out-of-core full rank fits too, but re-reads about 160 GB per solve, which suits a few right-hand sides and not a Krylov preconditioner.
- **The cheapest decisive step is a MUMPS analysis-only run** (`JOB=1`) on the 633,604-unknown matrix, once with METIS ordering and once with a Legendre-pair-major user ordering. It reports the factor entries, flops and in-core and out-of-core memory in minutes, and replaces every extrapolation here.
- **BLR as a preconditioner needs a small tolerance at this conditioning.** GMRES-based refinement requires roughly `(u_g + u_p κ)(κ² ε² + 1) ≪ 1`, so at a scaled `κ` near 3e10 only `ε ≲ 1e-8` is admissible; `ε` of 1e-4 to 1e-6 needs a scaled `κ` near 1e6 or below. In a plasma-physics precedent (JOREK), GMRES stopped converging at `ε = 1e-4` at the highest resolution.
- **Not viable here:** cuDSS, whose factors must fit in host memory; SuperLU_DIST, with no out-of-core or compression and mixed precision needing `κ` below about 1e5; JAX's `spsolve`, which keeps no factorization.
- **MKL PARDISO's residual of 9.4e-2** is consistent with its scaling and matching being off, which matches DKX's own finding that its defaults do not scale.
- **SFINCS does not factor the full operator by default**: it factors a simplified preconditioner and iterates, and its `solver.F90` sets `CNTL(1) = 1e-6` and grows `ICNTL(14)` from 50 on failure up to 1024.

**One figure is disputed and must be measured before it is used.** The survey takes the RTX A4000's float64 rate as 1/64 of float32, about 0.3 TFLOPS per card, from NVIDIA's GA102 whitepaper. An earlier measurement on these same office cards found true-float32 against float64 contraction speed-ups of 0.9× at 64³, 4.1× at 128³ and 12× at 192³, rising with size, and identified an earlier 1/64 figure as having come from the TF32 rating. The ratio at the size a Legendre-pair block factorization uses, `m` near 1e4, has not been measured; do that — a dense float64 against float32 LU on one card — before planning GPU factorization around either number.
