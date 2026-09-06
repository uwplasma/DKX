# Research B — Numerical methods survey for the DKX research plan

Date: 2026-09-06. Scope: numerical linear algebra, preconditioning, multigrid, Krylov recycling,
mixed precision, GPU direct solvers, JAX solver ecosystem, implicit differentiation, a posteriori
error estimation, and velocity-space discretization — all read against the DKX facts below.

## Given facts (from the DKX benchmark campaign, not re-verified here)

- DKX = JAX/GPU differentiable reimplementation of SFINCS v3 (drift-kinetic, stellarator neoclassics).
- Exact structured direct solver: block-tridiagonal in Legendre pitch modes L. Wins on all 9
  pitch-angle-scattering (PAS)/DKES decks: 744k unknowns in 27 s vs 464 s Fortran SFINCS+MUMPS.
- Structure breaks for: full Fokker-Planck (Rosenbluth potentials) speed coupling, tangential
  magnetic drifts, Er xDot/xiDot terms, Phi1 iteration. There DKX falls back to recycled Krylov:
  wins 7/23, lower memory on 3/32, fails 6/38 decks.
- Condition numbers: ~3e12 (PAS), ~1e18 saturated (Er-xDot). Ruiz equilibration improves PAS by 456x.
- SFINCS preconditions the full operator with a simplified one (drop L+-2 pitch couplings ->
  tridiagonal in L; drop x-coupling above a chosen L; drop inter-species blocks), factored by MUMPS.
- yancc (arXiv 2607.20861): finite differences + semi-coarsened multigrid preconditioning GCROT,
  1e6-1e7 unknowns on one GPU.
- SOLVAX (github.com/uwplasma/SOLVAX): sibling linear-algebra library (generated block factors,
  GCRO-DR-style recycling, implicit primitives).

Citation policy: every reference carries a URL that was fetched or seen verbatim in a search
result during this survey. Items that could not be confirmed are marked UNVERIFIED.

## Topic 1 — Physics-based / simplified-operator preconditioning for kinetic linear systems

### SFINCS (the direct template)
- 2014 paper (Landreman, Smith, Mollen, Helander, Phys. Plasmas 21, 042503; arXiv:1312.6058, PDF text
  extracted): "The resulting large sparse linear system is solved using the PETSc library. A
  preconditioned iterative Krylov solver is employed, either GMRES or BICGStab(l). An effective
  preconditioner is typically obtained by dropping all coupling in the x_a coordinate, either for all
  Legendre modes in xi, or for all but the first one or two Legendre modes. The preconditioner is
  LU-factorized directly using the SuperLU-dist package."
  Block structure (eq. 36): kinetic block M11, constraint rows M21/M31, source columns M12/M13; for
  several species "coupling between species only through the collision operators in the M11 blocks".
- v3 manual (github.com/landreman/sfincs, doc/manual/version3/runs.tex, raw file read): MUMPS is now the
  default LU for the preconditioner; "In side-to-side comparisons, we find MUMPS systematically uses
  substantially less memory and time than superludist for factorization"; out-of-core MUMPS available
  (-mat_mumps_icntl_22 1). For the nonlinear Phi1 iteration there is a `reusePreconditioner` option
  ("setting the preconditioner option reusePreconditioner = false can lead to a significant increase in
  memory requirements"; "If the nonlinear solver fails to converge with the default settings, a first
  useful thing to try is setting reusePreconditioner = false").
- The additional v3 preconditioner switches stated in the task (drop L+-2 pitch couplings -> tridiagonal
  in L; drop x-coupling above a chosen L; drop inter-species blocks) are consistent with the 2014 text
  but the namelist documentation itself could not be fetched (input.tex 404) — UNVERIFIED as named
  options; the x-coupling and species statements are verified from the 2014 paper.
- Why it works (analysis): the operator is A = streaming + mirror + Er terms + nu*(test-particle +
  field-particle) + species coupling. The stiff, resolution-dependent part (streaming/mirror at high L,
  PAS diagonal ~ nu L(L+1)) is kept exactly; what is dropped (x-coupling at L >= 2, field-particle
  integrals, inter-species blocks) is bounded by O(nu) and concentrated at low L (momentum L=1, energy
  L=0), so M^{-1}(A - M) is a compact perturbation and GMRES converges in few iterations. SFINCS's
  "keep x-coupling for the first one or two Legendre modes" is precisely keeping the L=0,1 field-particle
  physics inside M.

### yancc (arXiv:2607.20861)
- Multigrid-preconditioned GCROT; inside the alpha-coupled block smoother "we drop the off band terms
  corresponding to the integrals in the field part of the collision operator" — the field-particle
  (Rosenbluth) integrals are removed from the preconditioner and left to the Krylov outer loop. Same
  physics-based simplification as SFINCS, applied to a smoother instead of an LU.

### Other kinetic codes (what could be verified)
- COGENT: Dorf, Dorr, Ghosh, Umansky, Soukhanovskii, "Implicit full-F simulations of neoclassical ion
  transport", Phys. Plasmas 32(8) (2025) (osti.gov/biblio/2588989): "advanced preconditioning of
  individual physics operators is developed, and a global multi-physics preconditioner is constructed
  by adopting an operator splitting methodology"; "substantial speedup over the corresponding explicit
  approach"; applied to main ions and a lithium impurity. Block details not in the abstract.
- Gkeyll: implicit BGK collision operator in the DG full-f gyrokinetic solver (arXiv:2507.22821), which
  "can significantly increase the time step ... in highly collisional regimes", "generalized to handle
  cross-species collisions"; an iterative correction makes the discretized Maxwellian conservative.
- GS2: Barnes, Abel, Dorland et al., "Linearized model Fokker-Planck collision operators for gyrokinetic
  simulations. II. Numerical implementation and tests", Phys. Plasmas 16, 072107 (2009)
  (arXiv:0809.3945): implementation "is fully implicit and guarantees exact satisfaction of conservation
  properties" (abstract only).
- NEO (Belli & Candy): full linearized Fokker-Planck with Legendre in xi and a Laguerre spectral energy
  basis chosen "to ameliorate the rapid numerical precision loss" of standard Laguerre
  (iopscience 10.1088/0741-3335/54/1/015015, abstract). Solver/preconditioner details UNVERIFIED.
- GENE-X: conservative multi-species LBD collision operator implemented and verified (researchgate
  357053641); no preconditioner information verified. GENE (flux-tube) and XGC: nothing preconditioner-
  specific was verified in this survey — UNVERIFIED / not covered.
- The generic "physics-based preconditioning" framing (JFNK literature, Knoll & Keyes 2004) is standard
  but was not fetched — UNVERIFIED citation, omitted from the reference list.

### Exact PAS block-tridiagonal solve as preconditioner for full FP (analysis for DKX)
- DKX already owns a *better* simplified operator than SFINCS's: the exact block-tridiagonal-in-L
  solve keeps all x-coupling of the test-particle operator and all L+-1 physics, and it is exact, so
  M^{-1}(A - M) contains only: (i) field-particle integrals (dense in x, diagonal in L, decaying with L),
  (ii) L+-2 couplings from Er xiDot and tangential drifts, (iii) inter-species blocks, (iv) Phi1 terms.
  All are O(nu) or O(Er/v_th) relative to streaming — the same smallness SFINCS relies on.
- Cost model (dense-block hypothesis; check against SOLVAX's actual factor form): block-tridiagonal LU
  costs O(N_L n_b^3), back-substitution O(N_L n_b^2). With 744k unknowns and N_L ~ 40, n_b ~ 1.9e4 and
  the factorization is ~2.6e14 flops — consistent with the observed 27 s at ~10 TF/s — while one
  back-substitution is ~3e10 flops (milliseconds). Hence 20-50 preconditioned Krylov iterations add
  a few percent to the PAS solve: full-FP / drift / Er decks could run at essentially PAS cost if the
  iteration count is O(10-50). If the blocks are instead banded/sparse the ratio changes but the
  conclusion (solve << factor) survives.
- Block-Jacobi over species with the exact per-species PAS solve as the block is exactly SFINCS's
  species drop. Expect weak coupling for electron-ion (mass-ratio-small field terms on electrons) and
  O(1) coupling for ion-impurity at comparable Z^2 n_z / n_i, i.e. impurity decks need more iterations;
  a 2-species block (ion+impurity together, electrons separate) is the natural refinement.
- Reuse: hold M (the factorization) fixed across neighboring Er / nu points of a sweep (SFINCS's
  reusePreconditioner) so that (a) factorization cost is amortized and (b) Krylov recycling stays valid
  (Topic 3).

## Topic 2 — Multigrid for velocity-space / kinetic operators

### What yancc actually does (the only DKE-specific MG design in the literature)
Source: arXiv:2607.20861v1 (HTML full text fetched).
- Semi-coarsening: coarsens only in (alpha, theta, zeta), never in speed x: "Because even the finest
  grid generally requires very low resolution in x (n_x ~ 5-10) we find little benefit in coarsening
  the x coordinate." Factor 2 per direction per level, "roughly 8 [cost reduction] per grid level".
- Coarse operators by direct re-discretization, not Galerkin, "to keep memory usage low"; with
  semi-coarsening a Galerkin coarse operator "can have more nonzero elements than A_fine".
- Smoother: block-diagonal family of 4 smoothers (3 for monoenergetic), each keeping "full coupling
  along one coordinate" (line/plane smoothing), damping omega_s = 0.6 for the full 4D DKE. In the
  alpha-coupled smoother "we drop the off band terms corresponding to the integrals in the field part
  of the collision operator" — i.e. the Rosenbluth field-particle integrals are dropped inside the
  preconditioner, exactly the SFINCS philosophy transplanted into a smoother.
- Cycle: V-cycle for the 4D DKE, W-cycle (cycle index 3) for monoenergetic; coarsest level "a few
  thousand degrees of freedom" solved by dense LU.
- Defect correction: 4th-order operator, 2nd-order preconditioner.
- Stencil engineering for MG: wide upwind stencils (0,1,4) [2nd order] and (-2,0,1,3,4) [4th order]
  raise the diagonal-dominance ratio d = |a_ii| / sum_{j!=i} |a_ij| from 0.60 -> 0.88 (2nd order)
  and 0.24 -> 0.62 (4th order) "while maintaining similar leading error constant". Smoothing factor
  is markedly lower with the wide stencil "especially at low collisionality".
- Spectrum of I - MA: "clusters the vast majority of the eigenvalues near zero ... however there are
  several eigenvalues near the unit circle which are only weakly damped" — this is why GCROT (which
  keeps the important Krylov directions across restarts) is paired with the MG preconditioner.
- Reported outcome: 1e6-1e7 DOF on one A100; 6 GB vs >50 GB for SFINCS on 128 cores; "roughly 5x"
  faster across a collisionality scan, "up to nearly 2 orders of magnitude at higher collisionality";
  yancc runtime "relatively flat" in collisionality while "SFINCS slows down significantly" at high nu.
  Iteration counts and condition numbers are NOT reported.

### Angular multigrid for Fokker-Planck-type scattering (radiation transport analogue)
- arXiv:2010.04559 / J. Comput. Appl. Math. (sciencedirect S0377042718306174): an angular multigrid
  preconditioner for the Boltzmann transport equation with forward-peaked (Fokker-Planck) scatter,
  DG in space and angle; "iteration counts nearly independent of problem size even for highly
  non-isotropically refined angular meshes". The Legendre-scatter variant is the closest analogue
  to a pitch-angle-scattering operator in a Legendre basis (search-result abstract only; UNVERIFIED
  details).

### p-multigrid for hierarchical Legendre bases
- W-cycle p-multigrid for SIPDG with hierarchical Legendre polynomials (Springer, J. Sci. Comput.
  2025, doi 10.1007/s10915-025-03105-7; abstract seen in search only). For a modal Legendre
  discretization in xi there is no geometric hierarchy in pitch; the natural hierarchy is p-coarsening
  = truncating L. SFINCS's "drop x-coupling above a chosen L" preconditioner is a one-level version
  of exactly this idea.

### Reading for DKX (analysis, not citation)
- DKX's Legendre-modal xi grid cannot use yancc's geometric semi-coarsening in pitch. The available
  hierarchies are: (i) semi-coarsening in (theta, zeta) only, with DKX's exact block-tridiagonal-in-L
  solve acting as the "plane smoother" that keeps full coupling along L and x; (ii) p-coarsening in L.
- Because DKX's structured solve is exact on the PAS operator, a two-level scheme (exact PAS solve
  as coarse/smoother + Krylov for the L+-2, field-particle and species remainders) is the analogue
  of yancc's design and needs no new discretization. Multigrid proper (in theta,zeta) would only be
  needed if the exact-PAS preconditioner iteration count grows with resolution — a measurable
  admission test (see recommendations).
- yancc's stencil lesson transfers: any FD upwinding DKX uses in theta/zeta should be chosen for
  diagonal dominance if MG smoothing is ever used; for a direct/exact-PC design it is irrelevant.

## Topic 3 — Krylov recycling across parameter sweeps

### Sources (verified)
- Parks, de Sturler, Mackey, Johnson, Maiti, "Recycling Krylov subspaces for sequences of linear
  systems", SIAM J. Sci. Comput. 28(5), 1651-1674 (2006), doi:10.1137/040607277 (vtechworks.lib.vt.edu;
  osti.gov/biblio/900417): for "a long sequence of slowly changing linear systems", GCRO-DR
  "significantly reduce[s] the total number of matrix-vector products", for "the general case where
  both the matrix and right-hand side change with no assumptions regarding the change in the
  right-hand sides".
- de Sturler, "Truncation strategies for optimal Krylov subspace methods", SIAM J. Numer. Anal. 36(3),
  864-889 (1999), doi:10.1137/S0036142997315950 — GCROT (reference [39] in the survey).
- Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24(1), 20-37 (2002) — GMRES-DR
  (reference [104] in the survey).
- Soodhalter, de Sturler, Kilmer, "A survey of subspace recycling iterative methods", GAMM-Mitteilungen
  43(4), e202000016 (2020), doi:10.1002/gamm.202000016 (arXiv:2001.10347; PDF text extracted):
  * GCROT "extends GCRO by computing an optimal subspace to recycle for subsequent iterations. The
    optimality is based on considering the canonical angles between the subspaces generated by
    restarted GMRES"; recycling GCROT/GCRODR handle "a sequence of linear systems where the matrix
    changes slowly with right hand sides that may or may not be close".
  * GCRO-DR "builds on the same GCRO-type minimization but combined with the deflated-restarting
    strategies of Morgan, wherein the subspace retained between restart cycles is taken to be some
    harmonic Ritz vectors".
  * Validity across a sequence: "if two matrices A(i) and A(i+1) are 'close enough', then particular
    respective invariant subspaces may also be close. ... If the sequence of linear systems is induced
    by local changes in the matrix entries, then the changes to the invariant subspaces associated to
    the higher-frequency eigenvectors (with larger eigenvalues) dominate" — i.e. the recycled
    small-eigenvalue space is the robust one.
  * Nonnormal caveat: "Care must be taken, however, as it has been shown that residual convergence
    need not necessarily be connected to the spectral properties of the coefficient matrix".
  * Shifted systems A + gamma I (Sec. 7.1.1): "for general non-Hermitian coefficient matrices and
    augmentation subspaces, it is not possible to embed a shifted restarted GMRES within a subspace
    recycling framework"; the proposed alternative's "effectiveness decreases as the magnitude of the
    shift increases"; projected shift invariance K_j((I-Q)A, v) = K_j((I-Q)(A + gamma I), v) holds
    (Prop. 1).
  * Numerical caution: "Numerical difficulties are observed if Range(U) is very close to an invariant
    subspace of A" for one of the two augmented-projection formulations.
- Applications with quantified gains (URLs seen in search, not fetched): CFD (arXiv:1501.03358),
  aerostructural adjoints (arXiv:2309.09925), neural-operator data generation (arXiv:2401.09516),
  Krylov recycling + truncation in TOMS (dl.acm.org/doi/10.1145/3439746), mixed-precision GMRES-IR
  with recycling (arXiv:2201.09827).

### When recycling is invalid or wasted for DKX (analysis)
1. Preconditioner changes between systems. Recycled harmonic Ritz vectors approximate invariant
   subspaces of the *preconditioned* operator M^{-1}A (or A M^{-1}). If M is refactored per Er/nu point,
   the recycled space refers to a different operator and mostly wastes the k extra orthogonalizations.
   Remedy: hold M fixed over a window of the sweep (SFINCS reusePreconditioner); refresh M only when
   iterations exceed a threshold; restart the recycle space whenever M changes.
2. Not a shift. An Er or nu sweep changes A(Er) = S + Er*E + nu*C linearly in the parameter, not as
   A + gamma I, so shifted-Krylov shortcuts (shared Krylov space, collinear residuals) do not apply;
   only the "slowly changing matrix" regime is available, whose validity is empirical: monitor the
   first-cycle residual reduction with vs without the recycled space and drop it when it does not help.
3. Adjoint solves. A^T has the same eigenvalues but different (left) invariant subspaces; a recycle
   space built for A is not a recycle space for A^T when A is nonnormal (the DKE operator is strongly
   nonnormal: advection-dominated). Keep a second recycle space for the transposed sweep.
4. Grid or L_max changes (vector dimension changes) — no recycling; prolongating recycled vectors is
   possible in principle but unvalidated here (UNVERIFIED).
5. Nonnormality. Convergence may not follow the spectrum; deflating near-unit-circle eigenvalues of
   I - MA (as yancc observes) helps GCROT within a solve, but across a sweep the benefit must be
   measured, not assumed.
6. Right-hand sides. DKX solves a fixed small set of drives per point; block/multi-RHS recycling (all
   drives at once) is the cheap win the survey documents for "multiple right-hand sides, or both".
- Why DKX wins only 7/23 with recycled Krylov today (hypothesis, testable): items 1 and 3 — the
  preconditioner is rebuilt per point and the adjoint solves reuse primal spaces. Admission test:
  log iterations per solve with (a) fixed M + recycling, (b) fresh M + recycling, (c) no recycling,
  on a 10-point Er sweep of one full-FP deck.

## Topic 4 — Mixed precision / iterative refinement on GPU; admissibility at kappa = 1e12 and 1e18

### Sources (verified)
- Carson & Higham, "Accelerating the Solution of Linear Systems by Iterative Refinement in Three
  Precisions", SIAM J. Sci. Comput. (2018); MIMS EPrint 2017.24 (nhigham.com blog post fetched):
  two-precision IR (single/double) needs kappa_inf(A) <= 1e8; three-precision LU-IR (half/single/double)
  needs kappa_inf(A) <= 1e4; GMRES-IR (half/single/double) recovers kappa_inf(A) <= 1e8, all reaching
  errors O(1e-8), and GMRES-IR does "potentially half the work" of the traditional scheme.
- Carson & Higham 2017 SISC "A New Analysis of Iterative Refinement and Its Application to Accurate
  Solution of Ill-Conditioned Sparse Linear Systems" — referenced as [3]/[4] in the five-precision
  paper (title not independently fetched; UNVERIFIED title).
- Amestoy, Buttari, Higham, L'Excellent, Mary, Vieuble, "Five-precision GMRES-based iterative
  refinement", MIMS EPrint 2021.5, SIAM J. Matrix Anal. Appl. 45, 529-552 (2024)
  (eprints.maths.manchester.ac.uk/2852/1/paper.pdf, text extracted). Exact conditions:
  * LU-IR3 "is only guaranteed to converge when kappa(A) u_f << 1"; "kappa(A) << 2 x 10^3 with IEEE
    fp16 (half) precision, and kappa(A) << 3 x 10^2 with bfloat16".
  * GMRES-IR3 (preconditioner applied and residual computed at u^2): "guaranteed to converge as long as
    kappa(A)^2 u_f^2 u << 1"; Table 2.1: forward error converges if kappa(A) << u^{-1/2} u_f^{-1},
    backward if kappa(A) << u^{-1/2} u_f^{-1/2}; limiting forward error q u_r cond(A,x) + u, limiting
    backward error q u_r + u. Uses kappa(A~) <~ kappa(A)^2 u_f^2 for the LU-preconditioned matrix (2.5).
  * GMRES-IR5 with no extra precision (u_p = u_g = u): forward-error condition "kappa(A)^3 u u_f^2 << 1",
    backward "kappa(A)^2 u (1 + kappa(A) u_f) << 1".
  * Worked example (u_f = fp16, u = fp64): LU-IR3 kappa << 2e3; GMRES-IR5 with u_p = fp128: kappa << 2e11;
    u_p = fp64: kappa << 3e7; u_p = fp32: kappa << 4e4.
  * Table 3.1 unit roundoffs: bf16 3.91e-3, fp16 4.88e-4, fp32 5.96e-8, fp64 1.11e-16, fp128 9.63e-35.
- Higham & Mary, "Mixed precision algorithms in numerical linear algebra", Acta Numerica 31 (2022)
  (eprints.maths.manchester.ac.uk/2841/; research.manchester.ac.uk) — broad survey; not fetched beyond
  metadata. Survey of mixed-precision methods (Abdelfattah et al., arXiv:2007.06674) seen.
- Mixed-precision GMRES-IR with recycling (arXiv:2201.09827) — combines Topics 3 and 4; URL seen only.
- GPU direct solvers with mixed precision in the loop: cuDSS offers "Single, double, and double-double
  precision datatypes for values" (Topic 5).

### Admissibility for DKX (derived from the conditions above; constants f(n,k,rho_n) ignored — for
### n ~ 7e5 they can cost 2-3 orders of magnitude, so treat the numbers as upper bounds)
Working precision u = fp64 = 1.11e-16 throughout.
1. fp32 factorization of M (or of A) + fp64 GMRES-IR, preconditioner applied in fp64 (u_p = u, no
   extended precision — the realistic GPU case): forward condition kappa^3 u u_f^2 << 1 gives
   kappa << (1/(1.11e-16 * 3.55e-15))^{1/3} ~ 1.4e10; backward condition ~ kappa << 5e7.
   -> PAS decks: 3e12 unscaled FAILS; 3e12/456 ~ 7e9 after Ruiz equilibration is borderline-admissible
      (forward error) — Ruiz scaling is therefore a *prerequisite*, not an optimization, for any fp32
      factor. Er-xDot decks at 1e18: inadmissible.
   -> Payoff: factor storage halves (memory is where DKX loses on 29/32 decks); fp32 vs fp64 throughput
      gain is hardware dependent and not sourced here.
2. fp32 factorization used purely as a GMRES preconditioner (no IR semantics): kappa(M^{-1}A) <~
   kappa^2 u_f^2 (eq. 2.5 applied to A itself). Scaled PAS: (7e9)^2 * 3.55e-15 ~ 1.7e5 -> GMRES in fp64
   converges in modest iterations; unscaled 3e12: ~3e10 -> hopeless. Same conclusion as (1).
3. fp64 factorization + extended-precision residual (u_f = u = fp64, u_r = double-double ~ 1e-32):
   LU-IR3 condition kappa u << 1 -> kappa << ~1e14 (again minus constants): all PAS decks (3e12)
   admissible; limiting forward error q u_r cond(A,x) + u ~ u, i.e. full double accuracy recovered from
   the current kappa*u ~ 3e-4 (unscaled) / 7e-7 (scaled) forward-error bound. Cost: 1-3 extra
   back-substitutions plus a compensated (TwoSum/TwoProd) residual kernel. This is the cheapest route
   to defensible 1e-10-level agreement with DKES/SFINCS on PAS decks.
4. kappa ~ 1e18 "saturated": kappa u ~ 1e2 >> 1. No refinement scheme with an fp64 factorization
   converges; GMRES-IR with an fp64 factor as preconditioner has kappa(A~) <~ 1e36 * 1e-32 = 1e4 only if
   the factor were computed in double-double. The saturation at ~1/u means the matrix is numerically
   singular in double: the fix is formulation (null-space / constraint-row / source-column scaling,
   consistent with SFINCS's eq. 36 block structure) or a double-double factorization as a diagnostic
   (cuDSS supports double-double values; a small deck can be checked in mpmath). Preconditioning
   cannot repair it.
5. Half precision (fp16/bf16) anywhere in the factor: kappa << 2e3 / 3e2 for LU-IR; even GMRES-IR5 with
   fp64 PC application gives kappa << 3e7. Not admissible for DKE operators; exclude.

### Practical GPU notes (analysis)
- On the exact direct path, "iterative refinement" is just: r = b - A x in extended precision, solve
  M d = r with the existing factors, x += d. It reuses SOLVAX's factors and is a 20-line addition.
- On the Krylov path, GMRES-IR is structurally identical to right-preconditioned restarted GMRES with
  the residual recomputed explicitly at each restart — SOLVAX should recompute the true residual (not
  the Arnoldi estimate) at every restart anyway (Topic 6, jax.scipy gmres silent-failure issue).

## Topic 5 — GPU sparse direct solvers callable from Python/JAX

### NVIDIA cuDSS (docs.nvidia.com/cuda/cudss, v0.8.0 at fetch time; still labelled "Preview")
Verified feature list (quoted from the docs page):
- "Real/complex general/symmetric/positive-definite (incl. complex symmetric) sparse matrices"
- "Single, double, and double-double precision datatypes for values"  <- relevant to the kappa~1e18 decks
- "Hybrid host/device memory mode"  <- lets the factors exceed GPU memory
- "Multi-GPU multi-node (MGMN) execution with a user-definable communication layer"
- "Non-uniform batching (solving multiple different systems of different sizes)" and "Uniform batching
  (solving multiple systems with the same sparsity pattern)"  <- parameter sweeps / vmap
- Phases: "analysis", "numerical factorization and solving. Optionally, it includes refactorization and
  solve sub-phases"  <- reuse the symbolic analysis across an Er/nu sweep with fixed pattern
- "Optionally deterministic computations (bit-wise reproducibility ...)"; "Numerical pivoting controls";
  reorderings via SuiteSparse AMD, COLAMD and METIS.
- Python: nvmath-python exposes "a specialized sparse direct solver API based on the cuDSS library"
  (docs.nvidia.com/cuda/nvmath-python/0.5.0/host-apis/sparse/index.html; examples in
  github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver); wheels
  nvidia-cudss-cu13 on PyPI.
- JAX bindings that already exist: spineax (github.com/johnviljoen/spineax) "exposes most features of
  cuDSS to JAX with zero-copy arrays and full FFI jit/vmap integration including custom batching";
  cudss_jax MWE (github.com/stergiosba/cudss_jax); discussion github.com/jax-ml/jax/discussions/33205.
  Feasibility of an XLA-FFI custom call into cuDSS is therefore demonstrated, not hypothetical.
- Caveats (analysis): proprietary, NVIDIA-only (no ROCm path); differentiability must be added by hand
  (custom_vjp: the transpose solve reuses the same factors, solve with A^T); "Preview" API churn.

### STRUMPACK (LBNL, BSD)
- Ghysels & Synk, "High performance sparse multifrontal solvers on modern GPUs", Parallel Computing 110
  (2022) (osti.gov/pages/biblio/1960514): factorization + triangular solves on GPU via cuBLAS/cuSOLVER
  (NVIDIA) and rocBLAS/rocSOLVER (AMD); "runs ~10X faster when using all 24 V100 GPUs compared to when
  it only uses the 168 POWER9 cores"; "on average 5X (median 4X) faster" than SuperLU across 17 test
  matrices on a single V100; "48 V100 GPUs, the sparse solver reaches over 50TFlop/s".
- Claus, Ghysels, Boukaram, Li, "A graphics processing unit accelerated sparse direct solver and
  preconditioner with block low rank compression", Int. J. HPC Appl. (2025), doi
  10.1177/10943420241288567 (abstract seen via search; full text not fetched): BLR compression inside
  the multifrontal solver on GPU, vendor libraries for NVIDIA/AMD/Intel, "publicly available on github
  with a permissive BSD license"; positioned explicitly as a *preconditioner* (approximate LU).
- No Python/JAX binding known (UNVERIFIED); would need the same FFI work as cuDSS but is portable.

### SuperLU_DIST and MUMPS
- SFINCS 2014 (arXiv:1312.6058, PDF text extracted) used SuperLU-dist for the preconditioner LU.
- SFINCS v3 manual (doc/manual/version3/runs.tex, raw GitHub): MUMPS is the default, "In side-to-side
  comparisons, we find MUMPS systematically uses substantially less memory and time than superludist for
  factorization"; out-of-core MUMPS via -mat_mumps_icntl_22 1. MUMPS GPU offload status: UNVERIFIED
  (not fetched). SuperLU_DIST GPU status: UNVERIFIED (not fetched); the Li/Ghysels ATPESC 2022/2023
  slides (extremecomputingtraining.anl.gov) were seen in search results only.

### Other JAX-side sparse options seen (URLs verbatim from search)
- jax.experimental.sparse.linalg.spsolve: "only has CUDA GPU backend implemented with CPU fallback to
  scipy", no vmap (docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html).
- sparsax (github.com/knaaptime/sparsax): SuiteSparse CHOLMOD/KLU as XLA FFI custom calls (CPU).
- JAXMg (arXiv:2601.14466): cuSOLVERMg multi-GPU *dense* solver via XLA FFI C++ extension.

### Reading for DKX (analysis)
- A general sparse LU on GPU is not a replacement for DKX's structured solver on PAS decks (27 s for
  744k unknowns is already a direct solve); its role is the SFINCS-style *simplified-operator
  preconditioner* on the 23+ broken-structure decks, with (a) analysis reused across the sweep,
  (b) refactorization per parameter point, (c) hybrid memory when factors exceed device RAM,
  (d) double-double as a diagnostic for the kappa~1e18 decks.
- cuDSS is the lowest-cost path (bindings exist); STRUMPACK-BLR is the portable/approximate path.

## Topic 6 — JAX ecosystem solvers: Lineax, Optimistix, jaxopt, jax.scipy.sparse.linalg

### Lineax (Rader, Lyons, Kidger; arXiv:2311.17283; github.com/patrick-kidger/lineax; docs.kidger.site/lineax)
Verified from the paper text (PDF extracted locally):
- Unifies "linear solves and least-squares into a single, autodifferentiable API"; "solvers and operators
  [are] user-extensible without requiring the user to implement any custom derivative rules".
- Section 3.1 "Operator tags": "Tags are an optional argument to most linear operators, and indicate
  properties of the operator A" (e.g. positive semidefinite -> safe for lineax.CG); tags "are also used
  to select the appropriate" solver in AutoLinearSolver. Solvers named in the text: lineax.Tridiagonal,
  lineax.CG, lineax.QR, GMRES, BiCGStab (plus LU/Cholesky/SVD/Diagonal/Triangular per the docs — the
  docs page itself was not fetched; UNVERIFIED beyond the names above).
- Differentiation: "The transpose rule for the linear solve is implemented as a custom JAX primitive";
  JVP through well-posed and pseudoinverse solves is special-cased on row/column independence.
- Custom solvers: subclass "which requires the methods: init, compute, transpose, allow_dependent_rows,
  and allow_dependent_columns"; for two-stage solvers "init computes the factorisation, and compute
  performs the solve for the specific right hand b. transpose computes the transpose of the
  factorisation provided by init, which allows us to skip computing the factorisation of the transpose
  operator directly". This is exactly the hook SOLVAX's generated block factors need to be adjoint-
  aware for free (factor once, reuse for A^T in the VJP).
- "iterative solvers (CG, GMRES, ...) compile roughly twice as fast" than core JAX equivalents;
  PyTree-valued operators; operators may be matrix-free functions.
- Sparse (BCOO) operators: not mentioned in the extracted text — UNVERIFIED.

### Optimistix (arXiv:2402.09983; docs.kidger.site/optimistix/api/adjoints/ — fetched)
- ImplicitAdjoint: for a root f(y(theta), theta) = 0, dy/dtheta = -(df/dy)^{-1} df/dtheta; takes a
  `linear_solver` argument, default `lineax.AutoLinearSolver(well_posed=None)`, "Users may pass custom
  linear solvers"; "For most problems this is the preferred technique for backpropagating through a
  nonlinear solve."
- RecursiveCheckpointAdjoint: differentiates through solver iterates with binomial checkpointing;
  `checkpoints=None` scales "as log(max_steps)"; "The amount of memory used by the iterative solve will
  be roughly equal to the number of checkpoints multiplied by the size of y0"; "this cannot be
  forward-mode autodifferentiated".
- Relevance: the ambipolar Er root (sum_s Z_s Gamma_s(Er) = 0) is a scalar (or few-parameter) root of a
  function whose evaluation is a full DKE solve. optimistix.root_find with ImplicitAdjoint means the
  gradient w.r.t. geometry/profiles costs one extra linear solve of size 1 (the Jacobian dGamma/dEr is
  a scalar) plus DKX's own VJP through the DKE solve at the converged Er — no differentiation through
  the Er iterations. The Phi1 nonlinear iteration is the same pattern at the size of the Phi1 grid.

### jaxopt / Blondel et al., "Efficient and Modular Implicit Differentiation" (arXiv:2105.15183, NeurIPS 2022)
- Implicit differentiation "relates an optimization problem solution to its inputs using optimality
  conditions" and "doesn't require solver reimplementation"; the linear system in the VJP is solved
  with "GMRES or BiCGSTAB ... matrix-free algorithms requiring only matrix-vector products"; the
  alternative of unrolling has "memory complexity that scales linearly with the number of algorithm
  iterations".

### jax.scipy.sparse.linalg limits (URLs seen verbatim in search)
- github.com/jax-ml/jax/issues/15837 "GMRES Fails Silently and Frequently from Stagnation": fails to
  converge when restart < problem dimension "but will return without indicating failure"; "Roughly 90%
  of the time GMRES will fail silently on a problem of dimension 25 if restart=20"; the `info` output
  "is a placeholder for convergence information".
- JEP 18137 (docs.jax.dev/en/latest/jep/18137-numpy-scipy-scope.html) limits the scope of further
  jax.scipy wrappers; issue #11376 tracks jax.scipy.sparse.linalg development.
- No preconditioned flexible GMRES, no deflation/recycling, no residual history, no left/right PC
  choice for BiCGStab beyond M — hence the need for SOLVAX.

### Reading for DKX (analysis)
- Wrap the DKX/SOLVAX structured solver as a lineax.AbstractLinearSolver (init = block factorization,
  compute = block back-substitution, transpose = reuse factors). Then optimistix.root_find /
  fixed_point for Er and Phi1 get the exact adjoint through the linear solve with no extra code, and
  the DKE solve itself becomes composable with jax.grad / jax.vmap.
- Never surface jax.scipy.sparse.linalg.gmres convergence flags as "converged"; SOLVAX must return the
  true residual norm (not the Arnoldi estimate) and an explicit failure flag.

## Topic 7 — Implicit differentiation through roots and linear solves; adjoint of GMRES; checkpointing

### The adjoint of a linear solve is another linear solve with A^T
- For x = A^{-1} b with cotangent xbar: bbar = A^{-T} xbar, Abar = -bbar x^T (in whatever operator
  parameterization A carries). Lineax implements exactly this as the transpose rule of a custom
  primitive and reuses the factorization for the transpose (arXiv:2311.17283, Sec. on custom solvers:
  "transpose computes the transpose of the factorisation provided by init"). For DKX's block-
  tridiagonal-in-L factorization, the transposed factorization is the same block factors transposed
  and re-ordered — no second factorization. This is why the differentiable-direct path is cheap:
  gradient cost ~= one extra back-substitution per output functional.

### Why not differentiate through GMRES iterations
- Blondel et al. (arXiv:2105.15183): unrolled reverse mode has "memory complexity that scales linearly
  with the number of algorithm iterations"; implicit differentiation instead solves the adjoint linear
  system with matrix-free GMRES/BiCGSTAB needing "only JVPs or VJPs". The derivative of a truncated
  iteration is also not the derivative of the exact solution (it depends on the iteration path), so
  unrolling is both expensive and biased unless converged to roundoff.
- Optimistix defaults to ImplicitAdjoint for this reason and offers RecursiveCheckpointAdjoint
  (binomial checkpointing, memory ~ checkpoints x size(y0), default checkpoints ~ log(max_steps)) only
  as the fallback (docs.kidger.site/optimistix/api/adjoints/).
- Recent PDE-solver practice agrees: torch-sla, "Differentiable Sparse Linear Algebra with Adjoint
  Solvers" (arXiv:2601.13994) ships adjoint solvers rather than unrolled ones; "Differentiate the Solver,
  Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation" (arXiv:2608.08559) and
  "Automating Steady and Unsteady Adjoints: Efficiently Utilizing Implicit and Algorithmic
  Differentiation" (arXiv:2306.15243) — titles/URLs seen in search; contents UNVERIFIED.

### Adjoint solves with recycled Krylov (analysis, no citation)
- With a recycled-Krylov primal solve, the adjoint needs A^T y = xbar. Harmonic Ritz vectors recycled
  for A approximate right invariant subspaces of A; A^T needs the left ones. Recycle spaces are
  therefore NOT transferable primal->adjoint for nonnormal A, and DKX should keep a separate recycle
  space for the transposed operator across the sweep (the transposed operator changes as slowly as the
  primal one). Alternative: since DKX solves for a handful of RHS (drives) and a handful of outputs
  (transport-matrix moments), a block adjoint solve with all output cotangents as RHS amortizes the
  recycle space.
- Tolerance coupling: the gradient error is bounded by ||r_primal|| ||psi|| + ||r_adjoint|| ||f|| (DWR
  argument, Topic 8); the adjoint tolerance must match the primal one or the gradient is
  inconsistent with the function — a common cause of optimizer stalls.

### Checkpointing memory for reverse mode through iterative/nonlinear solvers
- Only needed for the Phi1 / Er outer iterations if ImplicitAdjoint is not used. Optimistix's
  RecursiveCheckpointAdjoint gives the log(max_steps) memory scaling. Classical revolve (Griewank &
  Walther) — UNVERIFIED URL, not fetched; jax.checkpoint (remat) is the primitive-level tool for the
  same trade-off inside a jitted solve (standard JAX API; not cited).
- For DKX the recommended structure is: exact/implicit adjoint through every linear solve (Lineax
  primitive), implicit-function-theorem adjoint through the Er root and the Phi1 fixed point
  (Optimistix), and no unrolling anywhere. Memory is then O(one factorization + a few vectors), not
  O(iterations).

## Topic 8 — A posteriori error estimation of integral outputs; verification methodology

### Adjoint-weighted residual (Giles & Pierce; Becker & Rannacher DWR)
- Pierce & Giles, "Adjoint recovery of superconvergent functionals from PDE approximations", SIAM Review
  42(2) 247-264 (2000) (search-result metadata; people.maths.ox.ac.uk/gilesm/old/error.html lists the
  series). Idea: use "the approximate solution of an appropriately defined adjoint problem to accurately
  estimate the error in the functional due to the residual error in approximating the original partial
  differential equation"; examples "obtain answers with twice the order of accuracy of the underlying
  numerical solution".
- Giles & Pierce, "Adjoint Error Correction for Integral Outputs" (Springer chapter,
  doi 10.1007/978-3-662-05189-4_2) and "Progress in adjoint error correction for integral functionals",
  Comput. Vis. Sci. (people.maths.ox.ac.uk/~gilesm/files/cvs04.pdf) — URLs seen; not fetched.
- Becker & Rannacher, "An optimal control approach to a posteriori error estimation in finite element
  methods", Acta Numerica 2001 (cambridge.org/core/journals/acta-numerica/... 5C67A03F528C6FA69F37A97DF5C3BE19):
  the DWR method "based on duality principles as used in optimal control"; error estimates "tailored to
  the particular goal of the computation". Linearization-error caveats for nonlinear problems:
  arXiv:2305.15285 (title seen).
- Formula (standard, both sources): for J(f) linear, J(f) - J(f_h) ~= -psi^T r_h(f_h) where r_h is the
  residual of the coarse solution in a finer/higher-order discretization and psi solves the adjoint
  A_fine^T psi = dJ/df. Each term is a matrix-vector product plus an adjoint solve.

### Richardson extrapolation / GCI, MMS, standards
- Roache's Grid Convergence Index: "derived from estimated fractional error obtained from the
  generalization of Richardson extrapolation"; Roy's mixed-order GCI analysis
  (aoe.vt.edu/.../grid-converg.submit-final.pdf) — seen in search.
- Salari & Knupp, "Code Verification by the Method of Manufactured Solutions", SAND2000-1444 (2000)
  (osti.gov/biblio/759450): MMS "provides a straightforward and general procedure for generating
  benchmark solutions" and "produces strong Code Verifications with a theorem-like quality and a
  clearly defined completion point". Roache, J. Fluids Eng. 124(1) 4 (2002) same title (ASME DC).
- ASME V&V 20-2009 (R2021), "Standard for Verification and Validation in Computational Fluid Dynamics
  and Heat Transfer" (webstore.ansi.org): quantifies "the degree of accuracy inferred from the
  comparison of solution and data" with uncertainties on both sides.
- Oberkampf & Roy, "Verification and Validation in Scientific Computing", CUP 2010, ISBN 9780521113601.

### Reading for DKX (analysis)
- DKX's outputs (transport matrix, bootstrap current, flows) are linear functionals J = c^T f1 of the
  solution, and DKX already computes adjoints for gradients. The Giles-Pierce/DWR estimate is therefore
  nearly free: (1) solve on the working grid; (2) inject f1 into a refined operator (more L modes, more
  x points, or the 4th-order stencil vs 2nd-order — yancc's defect-correction pair is exactly this);
  (3) one adjoint solve on the refined operator (or reuse the working-grid adjoint prolongated, at the
  cost of a lower-order estimate); (4) report J_corrected = J_h - psi^T r_h and |psi^T r_h| as the error
  bar. This turns "converged in L, Nx, Ntheta, Nzeta?" from a manual scan into a per-run number.
- Algebraic error must be included when the Krylov fallback is used: |J - J_h| <= |disc.| + |psi^T r_alg|
  with r_alg the final Krylov residual — the same adjoint vector gives it.
- With the exact direct solver, algebraic error is roundoff-only but NOT negligible at kappa = 3e12:
  forward-error bound kappa * u ~ 3e12 * 1.1e-16 ~ 3e-4 relative (unscaled), ~7e-7 after the 456x Ruiz
  equilibration. Report it; do not claim 1e-10 agreement with DKES on such decks without a
  double-double or iterative-refinement check.
- MMS is unusually cheap in JAX: pick a smooth manufactured f1_M(theta, zeta, xi, x), apply the DKX
  operator (a jitted function) to obtain the source to roundoff, solve, and measure the observed order
  in each coordinate. This should be the standing CI test for every new term (drifts, Er, field-
  particle), with Richardson/GCI on the transport-matrix entries as the integral-output check.

## Topic 9 — Pitch/speed discretization choices and their effect on conditioning and multigrid

### What the codes do (verified)
- SFINCS (arXiv:1312.6058, text extracted): "finite differences with a 5-point stencil in theta and zeta,
  using a truncated Legendre modal expansion in xi, and using a spectral collocation method in x_a";
  "The time-independent kinetic equation is solved directly (by solving a single sparse linear system),
  so the rate of convergence is not limited by the timescale of physical relaxation." Block structure
  (their eq. 36): kinetic block M11 plus constraint rows M21, M31 and source columns M12, M13 (particle
  and heat sources enforcing <int d3v f1> = 0 and <int d3v f1 v^2> = 0); multi-species coupling "only
  through the collision operators in the M11 blocks".
- Landreman & Ernst, "New velocity-space discretization for continuum kinetic calculations and
  Fokker-Planck collisions", J. Comput. Phys. 243, 130-150 (2013) (arXiv:1210.5289): speed collocation
  on polynomials orthogonal w.r.t. exp(-x^2) on [0, inf) (Maxwell polynomials); "performs far better than
  other discretization schemes at both integrating and differentiating functions relevant to kinetic
  theory"; procedures for the field term of the Fokker-Planck operator.
- yancc (arXiv:2607.20861): Maxwell-polynomial collocation with "5-10 points in the speed coordinate";
  uniform FD grids in (alpha, theta, zeta) with centered differences for collisions and the wide upwind
  stencils above for advection; diagonal-dominance figures d = 0.88 / 0.62 vs 0.60 / 0.24.
- MONKES (Escoto et al., arXiv:2312.12248; thesis arXiv:2510.27513, abstract fetched): "The Legendre
  representation of the monoenergetic drift-kinetic equation possesses a tridiagonal structure, which is
  exploited to solve the equation fast and accurately at low collisionality by employing the standard
  block tridiagonal algorithm"; monoenergetic coefficients in "approximately one minute on a single
  core". This is the same structure DKX's exact solver exploits, so DKX's PAS result is a GPU/multi-x
  generalization of MONKES's algorithm rather than a new idea — cite it as such.
- NEO (Belli & Candy; IOP 0741-3335/54/1/015015, abstract via search): "A Legendre series expansion in
  xi ... is combined with a novel Laguerre spectral method in energy to ameliorate the rapid numerical
  precision loss that occurs for traditional Laguerre spectral methods" — direct evidence that the
  speed basis choice is a conditioning issue in full-FP neoclassical solvers.
- GS2 collisions (Barnes, Abel, Dorland et al., Phys. Plasmas 16, 072107 (2009); arXiv:0809.3945):
  "fully implicit" model operator with exact conservation; pitch-angle part handled implicitly
  (tridiagonal in the FD pitch grid) — details UNVERIFIED beyond the abstract.
- Energy-diffusion spectral schemes with Maxwell polynomials: arXiv:1402.2971 (JCP 2015,
  S0021999115001941) and arXiv:1708.09031; singular Sturm-Liouville transform SIAM J. Appl. Math.
  doi 10.1137/130941948 — URLs seen in search, contents UNVERIFIED.
- Older: PPPL-4775 (Legendre in pitch, Fourier in theta, finite elements in v; "block tridiagonal system
  solved ... at each psi") — URL seen.

### Structure vs conditioning (analysis, derivable from the operator, not a citation)
- In a Legendre-modal xi basis: PAS collision term is diagonal in L with eigenvalue -nu L(L+1)/2;
  parallel streaming xi * d/dl and the mirror force (1-xi^2) d/dxi couple L+-1 -> block tridiagonal in L.
  Terms containing xi^2 (Er-driven xiDot ~ xi(1-xi^2) and tangential magnetic drift factors
  ~ (1+xi^2)) couple L+-2 -> block pentadiagonal in L. Terms containing xi * x-derivatives (Er xDot)
  keep L+-1 but add x-coupling; the linearized field-particle (Rosenbluth) operator for a Maxwellian
  background is diagonal in L but dense in x and couples species. So on paper: Er-xiDot and tangential
  drifts break tridiagonality only to pentadiagonality; the field-particle operator and species coupling
  densify the diagonal blocks but do not widen the L band. DKX should verify which of these actually
  defeats its structured solver (admission test A1 below) — the pentadiagonal case costs roughly 4x a
  tridiagonal block LU (bandwidth^2 scaling) and stays exact.
- Conditioning in the modal basis: the diagonal grows like nu L_max^2 while the smallest streaming
  eigenvalues shrink with collisionality and resolution -> kappa ~ nu L_max^2 / lambda_min, consistent
  with the 3e12 seen on PAS decks at low nu. Ruiz equilibration (456x on PAS) removes the row/column
  scale disparity (the L^2 growth), not the intrinsic near-null direction from the constraint/source
  structure; the 1e18 saturation on Er-xDot decks indicates an (almost) exactly singular direction in
  double, not just poor scaling (see Topic 4 for what is and is not recoverable).
- FD in pitch (yancc) trades exact L-structure for diagonal dominance and a geometric MG hierarchy;
  Legendre-modal (SFINCS/DKX/MONKES) trades MG for an exact banded solve and spectral accuracy at the
  trapped-passing boundary layer only when L_max is large. There is no free lunch; DKX's choice is the
  right one for the exact-solver strategy and should be kept, with p-coarsening in L (Topic 2) as the
  hierarchy if a multilevel PC is ever needed.
- Speed grid: Maxwell-polynomial collocation is the community standard (SFINCS, yancc, MONKES, NEO
  variants) and yancc shows 5-10 points suffice for full-FP fluxes; DKX's x-coupled blocks are therefore
  small (n_x^2 per (theta, zeta, L) point), which is what makes keeping x-coupling *inside* the
  preconditioner affordable, unlike SFINCS in 2014.

## Recommendations for DKX (ranked)

Ranking weighs expected payoff on the 23 broken-structure decks and 6 failures against cost and risk.
Quantities marked (lit.) come from fetched sources; (est.) are derived in the topic sections above.

| # | Change | Expected payoff | Implementation cost | Prerequisite admission test | Risk |
|---|--------|-----------------|---------------------|-----------------------------|------|
| 1 | Use the exact PAS block-tridiagonal factorization as the right preconditioner of FGMRES/GCRO-DR on every non-PAS deck (full FP, species-coupled, Er xDot, Phi1), i.e. SFINCS's simplified-operator PC but with all x-coupling and L+-1 physics kept exactly (Topic 1) | Full-FP/drift/Er decks at ~PAS cost if iterations are O(10-50): back-substitution is ~n_b times cheaper than factorization (est.: ms vs 27 s on the 744k deck). SFINCS 2014 and yancc both show the dropped remainder (field-particle integrals, species blocks) is Krylov-friendly (lit.). Targets the 16 losses and most of the 6 failures | Low-medium: SOLVAX already has the factors and GCRO-DR; add M^{-1} as a LinearOperator, flexible right preconditioning, true-residual restarts | A1: assemble A - M_PAS on three decks (full-FP, impurity, Er) and histogram nonzeros by |dL|; run preconditioned GMRES and require iteration count flat in (Ntheta, Nzeta, Nx) and <= 50 | High-nu full-FP decks: O(nu) field-particle terms at L=0,1 are not small -> keep x-coupling for L<=1 inside M (SFINCS practice); impurity decks may need a 2-species block |
| 2 | Generalize the structured solver from block-tridiagonal to block-pentadiagonal in L so Er xiDot and tangential-drift decks are solved exactly again (Topic 9) | Exactness on those decks at ~4x the tridiagonal factor cost (bandwidth^2 scaling, est.) — still ~5x faster than Fortran SFINCS+MUMPS on the reference deck if 27 s -> ~100 s; removes them from the Krylov fallback entirely | Medium: new generated block factors in SOLVAX (banded block LU, bandwidth 2) | A2: verify on the Er/drift decks that the only new couplings are |dL| = 2 (A1 histogram); confirm factor memory fits (2x storage of L,U bands) | If species or Phi1 coupling is also present the pentadiagonal solve is again only a preconditioner (then it feeds Rec. 1) |
| 3 | fp64 factorization + extended-precision (double-double) residual iterative refinement on the exact direct path (Topic 4) | Forward error from the current bound kappa*u ~ 3e-4 unscaled / 7e-7 Ruiz-scaled to ~1e-15 on PAS decks (lit. Table 2.1: LU-IR converges for kappa*u << 1, limiting error q u_r cond + u); makes 1e-10-level DKES/SFINCS comparisons defensible | Low: compensated residual kernel + 1-3 back-substitutions with existing factors | A3: measured kappa after Ruiz <= ~1e13 on the deck; forward error vs an mpmath reference on a small deck drops to ~1e-14 | Double-double matvec needs a custom TwoSum/TwoProd kernel in JAX; none for kappa~1e18 decks |
| 4 | Diagnose and reformulate the kappa ~ 1e18 (Er-xDot) decks instead of solving them harder: saturation at 1/u means numerical singularity, not scaling (Topic 4, item 4) | Turns the 6 failures into solvable systems; prerequisite for any Krylov or mixed-precision method to work on them — no preconditioner rescues kappa*u ~ 1e2 (lit. conditions) | Medium, analysis-heavy: double-double factorization (cuDSS supports it) or mpmath on a reduced deck to get the true kappa; then null-space projection / constraint-row and source-column scaling (SFINCS eq. 36 structure) | A4: true kappa in double-double; identify the near-null vector (moments of it tell whether it is a missing constraint, a Lagrange-multiplier scaling, or a genuine physical nullity) | May expose a formulation gap (e.g. constraint consistency with the Er terms) needing physics-level changes |
| 5 | Ruiz equilibration as a mandatory pre-step, then fp32 factorization of M with fp64 GMRES-IR (preconditioner applied in fp64) (Topic 4, items 1-2) | Factor memory halved — DKX loses on memory on 29/32 decks; admissible only when scaled kappa << ~1e10 (lit. condition kappa^3 u u_f^2 << 1); Ruiz-scaled PAS at ~7e9 is borderline, unscaled 3e12 fails | Low: dtype switch on the factor path + IR loop (same code as Rec. 3) | A5: per-deck scaled kappa estimate; fp32-factor solution refined to 1e-10 relative vs fp64 reference on 3 decks; iteration count <= 10 | Constants f(n,k,rho_n) for n ~ 7e5 can eat 2-3 orders of magnitude -> some decks stay fp64; fp32 flop gain is hardware dependent |
| 6 | Ambipolar Er root and Phi1 fixed point through optimistix.root_find / fixed_point with ImplicitAdjoint whose linear_solver is a Lineax wrapper of the SOLVAX factorization (Topics 6-7) | Gradients of ambipolar fluxes at the cost of one adjoint solve; no unrolled memory (lit.: unrolling scales linearly with iterations; ImplicitAdjoint is "the preferred technique"); factor reused for A^T via Lineax's transpose hook | Low-medium: implement lineax.AbstractLinearSolver (init/compute/transpose) around SOLVAX | A6: dGamma/dEr nonsingular at the converged root; gradient check vs finite differences on one deck; transposed-factor solve matches a fresh A^T factorization | Multiple ambipolar roots (ion/electron): IFT differentiates the branch found; root switching under optimization is a nondifferentiable event |
| 7 | Recycling discipline: hold M fixed over a window of the sweep (SFINCS reusePreconditioner), recycle harmonic Ritz vectors of the fixed preconditioned operator, keep a separate recycle space for A^T, block-solve all drives, drop the space when the first-cycle reduction does not beat no-recycling (Topic 3) | Recovers the "significantly reduce[d] matrix-vector products" regime of Parks 2006 (lit.); plausible explanation for recycling winning only 7/23 today (est.: M rebuilt per point invalidates the space) | Low: bookkeeping in SOLVAX's GCRO-DR | A7: 10-point Er sweep with (fixed M + recycle), (fresh M + recycle), (no recycle); require monotone iteration savings | Nonnormality: convergence need not follow the spectrum (lit.) -> benefit must be measured per deck class |
| 8 | Adjoint-weighted (Giles-Pierce / DWR) error estimates on transport-matrix entries, with MMS in CI for every new term (Topic 8) | Per-run discretization error bars and superconvergent corrected outputs ("twice the order of accuracy", lit.); verification story in ASME V&V 20 language; nearly free because the adjoint already exists | Low: one refined-operator residual + one adjoint solve; MMS source = DKX operator applied to a manufactured f1 | A8: on one PAS deck the corrected J matches the refined-grid J within the estimate; observed order matches design order in each coordinate | Krylov-fallback runs must add the algebraic term psi^T r_alg; nonlinear Phi1 adds linearization error |
| 9 | SFINCS-style simplified-operator LU on GPU via cuDSS through an XLA FFI custom call (spineax pattern) as the fallback when no structured M exists (Topic 5) | Parity with SFINCS+MUMPS design on GPU; analysis reused across a sweep (refactorization phase), hybrid host/device memory, double-double option (lit. feature list) | Medium-high: FFI extension, custom_vjp (transpose solve reuses factors), packaging; NVIDIA-only | A9: cuDSS factor time and memory on the 744k pattern vs DKX's 27 s / current memory; solve accuracy at kappa 3e12 | Proprietary "Preview" API; no ROCm path (STRUMPACK-BLR is the portable alternative, no Python binding known) |
| 10 | Semi-coarsened multigrid in (theta, zeta) with the exact block-tridiagonal-in-L solve as plane smoother; p-coarsening in L as the second hierarchy (Topic 2) | Resolution-independent iteration counts as in yancc (lit.: 1e6-1e7 DOF, ~5x over SFINCS on 128 cores, 6 GB vs >50 GB) — only needed if Rec. 1 iterations grow with resolution | High: new solver component; coarse-operator rediscretization; smoother tuning | A10: Rec. 1 iteration count vs (Ntheta, Nzeta) shows growth on the target decks | Legendre-modal xi has no geometric pitch hierarchy; yancc's stencil engineering does not transfer |

Not recommended: half precision anywhere in the factor (kappa << 2e3-3e7 admissibility, lit.);
differentiating through Krylov iterations (memory linear in iterations, biased derivative);
relying on jax.scipy.sparse.linalg.gmres convergence flags (silent stagnation, lit.).

## Reference list (URLs fetched or seen verbatim in search results during this survey)

Primary DKE / neoclassical solvers
- "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv:2607.20861 (2026). https://arxiv.org/abs/2607.20861 ; full text https://arxiv.org/html/2607.20861v1 (fetched). Author list not extracted — UNVERIFIED authors.
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014). https://arxiv.org/pdf/1312.6058 (fetched, text extracted); https://pubs.aip.org/aip/pop/article-abstract/21/4/042503/818401/
- SFINCS repository and v3 manual. https://github.com/landreman/sfincs ; https://raw.githubusercontent.com/landreman/sfincs/master/doc/manual/version3/runs.tex (read). input.tex not found (404) — namelist option names UNVERIFIED.
- Escoto et al., "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", arXiv:2312.12248. https://arxiv.org/pdf/2312.12248 (URL seen)
- Escoto Lopez, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas" (thesis), arXiv:2510.27513 (2025). https://arxiv.org/abs/2510.27513 (fetched)
- Belli & Candy, "Full linearized Fokker-Planck collisions in neoclassical transport simulations", PPCF 54, 015015 (2012). https://iopscience.iop.org/article/10.1088/0741-3335/54/1/015015
- Landreman & Ernst, "New velocity-space discretization for continuum kinetic calculations and Fokker-Planck collisions", J. Comput. Phys. 243, 130-150 (2013). https://arxiv.org/abs/1210.5289 ; https://www.sciencedirect.com/science/article/abs/pii/S0021999113001605
- Velasco et al., KNOSOS. https://arxiv.org/pdf/2106.01727 ; https://github.com/joseluisvelasco/KNOSOS (URLs seen)
- PPPL-4775, "Numerical Calculation of Neoclassical Distribution Functions ..." https://bp-pub.pppl.gov/pub_report/2012/PPPL-4775.pdf (URL seen)
- DKX and SOLVAX repositories (given by the task; not fetched). https://github.com/uwplasma/DKX ; https://github.com/uwplasma/SOLVAX

Kinetic-code preconditioning
- Dorf, Dorr, Ghosh, Umansky, Soukhanovskii, "Implicit full-F simulations of neoclassical ion transport", Phys. Plasmas 32(8) (2025). https://www.osti.gov/biblio/2588989 (fetched)
- "Axisymmetric Gyrokinetic Simulation of ASDEX-Upgrade Scrape-off Layer Using a Conservative Implicit BGK Collision Operator" (Gkeyll), arXiv:2507.22821. https://arxiv.org/abs/2507.22821
- Barnes, Abel, Dorland et al., "Linearized model Fokker-Planck collision operators for gyrokinetic simulations. II. Numerical implementation and tests", Phys. Plasmas 16, 072107 (2009). https://arxiv.org/abs/0809.3945
- GENE-X LBD collision operator (implementation/verification). https://www.researchgate.net/publication/357053643_Implementation_and_verification_of_a_conservative_multi-species_gyro-averaged_full-f_Lenard-Bernstein_Dougherty_collision_operator_in_the_gyrokinetic_code_GENE-X

Multigrid
- "An Angular Multigrid Preconditioner for the Radiation Transport Equation with Forward-Peaked Scatter", arXiv:2010.04559. https://arxiv.org/html/2010.04559 ; Fokker-Planck variant https://www.sciencedirect.com/science/article/pii/S0377042718306174
- "P-Multigrid Method for the Discontinuous Galerkin Discretization of Elliptic Problems", J. Sci. Comput. (2025). https://link.springer.com/article/10.1007/s10915-025-03105-7

Krylov recycling
- Parks, de Sturler, Mackey, Johnson, Maiti, "Recycling Krylov subspaces for sequences of linear systems", SIAM J. Sci. Comput. 28(5), 1651-1674 (2006), doi:10.1137/040607277. https://vtechworks.lib.vt.edu/items/590c07fe-a0c8-49b2-9494-be5061f5fbf7 ; https://www.osti.gov/biblio/900417
- Soodhalter, de Sturler, Kilmer, "A survey of subspace recycling iterative methods", GAMM-Mitt. 43(4), e202000016 (2020), doi:10.1002/gamm.202000016. https://arxiv.org/abs/2001.10347 ; https://arxiv.org/pdf/2001.10347 (fetched, text extracted); https://onlinelibrary.wiley.com/doi/10.1002/gamm.202000016
- de Sturler, "Truncation strategies for optimal Krylov subspace methods", SIAM J. Numer. Anal. 36(3), 864-889 (1999), doi:10.1137/S0036142997315950 (DOI taken from the survey's reference list; not fetched separately)
- Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24(1), 20-37 (2002) (from the survey's reference list; not fetched separately)
- Kilmer & de Sturler, "Recycling subspace information for diffuse optical tomography", SIAM J. Sci. Comput. (2006) (from the survey's reference list)
- "Recycling Krylov Subspaces and Truncating Deflation Subspaces for Solving Sequence of Linear Systems", ACM TOMS (2021). https://dl.acm.org/doi/10.1145/3439746
- Applications: https://arxiv.org/pdf/1501.03358 (CFD); https://arxiv.org/pdf/2309.09925 (aerostructural adjoints); https://arxiv.org/pdf/2401.09516 (neural-operator data generation)

Mixed precision
- Carson & Higham, "Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions", SIAM J. Sci. Comput. (2018); MIMS EPrint 2017.24. https://nhigham.com/2017/07/26/accelerating-the-solution-of-linear-systems-by-iterative-refinement-in-three-precisions/ (fetched)
- Amestoy, Buttari, Higham, L'Excellent, Mary, Vieuble, "Five-precision GMRES-based iterative refinement", SIAM J. Matrix Anal. Appl. 45, 529-552 (2024); MIMS EPrint 2021.5. https://eprints.maths.manchester.ac.uk/2852/1/paper.pdf (fetched, text extracted); https://eprints.maths.manchester.ac.uk/2807/
- Higham & Mary, "Mixed precision algorithms in numerical linear algebra", Acta Numerica 31 (2022). https://eprints.maths.manchester.ac.uk/2841/ ; https://research.manchester.ac.uk/en/publications/mixed-precision-algorithms-in-numerical-linear-algebra/
- Abdelfattah et al., "A Survey of Numerical Methods Utilizing Mixed Precision Arithmetic". https://arxiv.org/pdf/2007.06674
- "Mixed Precision GMRES-based Iterative Refinement with Recycling". https://arxiv.org/pdf/2201.09827

GPU sparse direct solvers and JAX bindings
- NVIDIA cuDSS documentation (v0.8.0, Preview). https://docs.nvidia.com/cuda/cudss/index.html (fetched); https://developer.nvidia.com/cudss
- nvmath-python sparse direct solver (cuDSS-backed). https://docs.nvidia.com/cuda/nvmath-python/0.5.0/host-apis/sparse/index.html ; https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver ; https://pypi.org/project/nvidia-cudss-cu13/
- spineax (cuDSS in JAX via FFI). https://github.com/johnviljoen/spineax ; cudss_jax MWE https://github.com/stergiosba/cudss_jax ; JAX discussion https://github.com/jax-ml/jax/discussions/33205
- sparsax (SuiteSparse CHOLMOD/KLU via XLA FFI). https://github.com/knaaptime/sparsax/blob/main/README.md
- JAXMg (cuSOLVERMg multi-GPU dense via FFI). https://arxiv.org/pdf/2601.14466
- jax.experimental.sparse.linalg.spsolve docs. https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
- Ghysels & Synk, "High performance sparse multifrontal solvers on modern GPUs", Parallel Computing 110 (2022). https://www.osti.gov/pages/biblio/1960514 (fetched)
- Claus, Ghysels, Boukaram, Li, "A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression", Int. J. HPC Appl. (2025), doi:10.1177/10943420241288567. https://journals.sagepub.com/doi/10.1177/10943420241288567 ; https://escholarship.org/uc/item/7tn9n67r (fetch returned empty)
- Li & Ghysels, ATPESC direct-solver lectures 2022/2023 (URLs seen). https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2023/08/ATPESC-2023-Track-5-Talk-3-Li-Ghysels-DirectSolvers.pdf

JAX ecosystem and implicit differentiation
- Rader, Lyons, Kidger, "Lineax: unified linear solves and linear least-squares in JAX and Equinox", arXiv:2311.17283 (NeurIPS 2023 AI4Science). https://arxiv.org/abs/2311.17283 ; https://arxiv.org/pdf/2311.17283 (fetched, text extracted); https://github.com/patrick-kidger/lineax ; https://docs.kidger.site/lineax/
- Rader et al., "Optimistix: modular optimisation in JAX and Equinox", arXiv:2402.09983. https://arxiv.org/pdf/2402.09983 ; adjoints doc https://docs.kidger.site/optimistix/api/adjoints/ (fetched); https://docs.kidger.site/optimistix/api/root_find/ ; https://github.com/patrick-kidger/optimistix
- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-Lopez, Pedregosa, Vert, "Efficient and Modular Implicit Differentiation", NeurIPS 2022, arXiv:2105.15183. https://arxiv.org/pdf/2105.15183 ; https://ar5iv.labs.arxiv.org/html/2105.15183
- JAX issue #15837 "GMRES Fails Silently and Frequently from Stagnation". https://github.com/jax-ml/jax/issues/15837 ; gmres docs https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html ; JEP 18137 https://docs.jax.dev/en/latest/jep/18137-numpy-scipy-scope.html ; issue #11376 https://github.com/jax-ml/jax/issues/11376
- torch-sla, "Differentiable Sparse Linear Algebra with Adjoint Solvers ...", arXiv:2601.13994. https://arxiv.org/pdf/2601.13994
- "Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation", arXiv:2608.08559. https://arxiv.org/html/2608.08559 (title only)
- "Automating Steady and Unsteady Adjoints: Efficiently Utilizing Implicit and Algorithmic Differentiation", arXiv:2306.15243. https://arxiv.org/html/2306.15243 (title only)

A posteriori error estimation and V&V
- Pierce & Giles, "Adjoint recovery of superconvergent functionals from PDE approximations", SIAM Review 42(2), 247-264 (2000) (metadata from search); Giles' error-analysis page https://people.maths.ox.ac.uk/gilesm/old/error.html
- Giles & Pierce, "Adjoint Error Correction for Integral Outputs", Springer (doi 10.1007/978-3-662-05189-4_2). https://link.springer.com/chapter/10.1007/978-3-662-05189-4_2
- Giles & Pierce, "Progress in adjoint error correction for integral functionals", Comput. Vis. Sci. https://people.maths.ox.ac.uk/~gilesm/files/cvs04.pdf ; https://link.springer.com/article/10.1007/s00791-003-0115-y
- Becker & Rannacher, "An optimal control approach to a posteriori error estimation in finite element methods", Acta Numerica (2001). https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation-in-finite-element-methods/5C67A03F528C6FA69F37A97DF5C3BE19
- "Linearization Errors in Discrete Goal-Oriented Error Estimation", arXiv:2305.15285. https://arxiv.org/pdf/2305.15285 (title only)
- Roache, Grid Convergence Index (secondary sources). https://cfd.university/blog/how-to-manage-uncertainty-in-cfd-the-grid-convergence-index/ ; Roy, "Grid Convergence Error Analysis for Mixed-Order Numerical Schemes" https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/grid-converg.submit-final.pdf
- Salari & Knupp, "Code Verification by the Method of Manufactured Solutions", SAND2000-1444 (2000). https://www.osti.gov/biblio/759450/
- Roache, "Code Verification by the Method of Manufactured Solutions", J. Fluids Eng. 124(1), 4 (2002). https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791/Code-Verification-by-the-Method-of-Manufactured
- ASME V&V 20-2009 (R2021), "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer". https://webstore.ansi.org/standards/asme/asme2020092021
- Oberkampf & Roy, "Verification and Validation in Scientific Computing", Cambridge University Press (2010), ISBN 9780521113601. https://books.google.com/books/about/Verification_and_Validation_in_Scientifi.html?id=7d26zLEJ1FUC

Velocity-space discretization (additional)
- "Accurate spectral numerical schemes for kinetic equations with energy diffusion", J. Comput. Phys. (2015). https://arxiv.org/pdf/1402.2971 ; https://www.sciencedirect.com/science/article/abs/pii/S0021999115001941
- "Pseudo spectral collocation with Maxwell polynomials for kinetic equations with energy diffusion". https://arxiv.org/pdf/1708.09031
- "A Spectral Transform Method for Singular Sturm-Liouville Problems with Applications to Energy Diffusion in Plasma Physics", SIAM J. Appl. Math. https://dx.doi.org/10.1137/130941948

UNVERIFIED (named in the text, no URL fetched or seen): SFINCS v3 preconditioner namelist option names;
Knoll & Keyes (2004) JFNK review; Griewank & Walther revolve; MUMPS and SuperLU_DIST GPU-offload status;
Lineax sparse (BCOO) operator support; exact title of Carson & Higham 2017 SISC; NEO and GS2 solver internals
beyond their abstracts; yancc author list.
