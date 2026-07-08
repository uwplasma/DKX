# plan_claude.md — sfincs_jax v2: solver-library extraction, consolidation, performance, and release

**Status:** authoritative execution plan, written 2026-07-08 after a full audit of the local
repositories and a verified literature survey. It is written to be executed end-to-end by an
autonomous agent (Cowork) with full local access to `/Users/rogerio/local/`. Read this file
completely before doing anything. Where this plan conflicts with older plan files
(`plan_final.md`, `plan.md`), **this file wins**; the older ones are historical context and are
deleted from the tree (and from git history) in Phase 7.

**Branch policy:** all work starts from `refactor/v3-driver-architecture` and only that branch.
Create working branches off it; merge back via PRs authored by `rogeriojorge`. `main` is the
"old world" and will be replaced when the refactor PR lands.

**Attribution policy (hard rule):** every commit made during this plan must be authored and
committed as Rogerio Jorge / `rogeriojorge` (set `git config user.name "Rogerio Jorge"` and
`git config user.email "rogerio.jorge@wisc.edu"` in every clone before the first commit; never
add `Co-Authored-By:` or `Generated with` trailers). The current history is already clean —
verified 2026-07-08: the only identities across all refs are `Rogerio Jorge
<rogerio.jorge@ist.utl.pt>`, `Rogerio Jorge <rogerio.jorge@wisc.edu>`, and GitHub's noreply
merge committer; zero AI-assistant identities or co-author trailers exist. Phase 7 still
verifies the live GitHub contributors API and unifies the two Rogerio emails via mailmap.

**Reference-code policy (hard rule):** besides SFINCS (Fortran v3), several other
neoclassical/kinetic solvers are checked out under `/Users/rogerio/local/` (JAX-based and
Fortran-based). Survey them, run them, profile them, and adopt the good *ideas* — algorithms,
discretizations, preconditioners, convergence tricks — re-implemented from scratch. Those
codes must **never be named** in any committed file: not in this plan's successors, not in
source, comments, README, docs, or commit messages. Describe every adopted idea purely in
terms of the underlying numerical method, with citations to the primary literature (papers,
textbooks, theses) listed in §1. This file itself follows that rule.

---

## 0. Context

### 0.1 What SFINCS (Fortran v3) is — the parity reference

SFINCS (`/Users/rogerio/local/sfincs`, `fortran/version3`, ~22.3 kLOC of F90 + bundled
mini_libstell) solves the radially-local, 4D (θ, ζ, ξ, x) drift-kinetic equation for the
non-Maxwellian perturbation f₁ₛ of each species s on one flux surface:

```
(v‖ b + v_E + v_m) · ∇f₁ₛ − C_s(f₁ₛ) = −v_ms · ∇ψ ∂f_Mₛ/∂ψ + (E‖ and Er drive terms)
```

with the linearized Fokker–Planck–Landau collision operator (field term via Rosenbluth
potentials), four trajectory models (full vs DKES-like E×B, with/without magnetic drifts),
and optional nonlinear Φ₁ via quasineutrality. Key implementation facts confirmed by audit:

- **Discretization.** θ: `Ntheta` uniform points, `thetaDerivativeScheme` 0/1/2 (spectral /
  2nd / 4th-order FD, default 2); ζ: same on [0, 2π/NPeriods]; ξ: **Legendre modal**, `Nxi`
  modes, collision coupling depth `NL` (default 4), with x-dependent mode count via
  `Nxi_for_x_option` (default 1: linear ramp 0.1·Nxi→Nxi over x∈[0,2]); x: `Nx` (default 5)
  polynomial spectral collocation with weight exp(−x²)xᵏ on [0,∞) (`xGridScheme` default 5,
  the Landreman–Ernst grid, nodes/weights from a Stieltjes/Jacobi eigenproblem in
  `xGrid.F90`). Matrix ordering documented at the top of `indices.F90`:
  `getIndex(iSpecies, ix, ixi, itheta, izeta, block)`, DKE_size = Σ_x Nxi_for_x·Ntheta·Nzeta
  per species, plus constraint rows (2·Nspecies for `constraintScheme` 1, Nx·Nspecies for 2)
  and, with Φ₁, Ntheta·Nzeta quasineutrality rows + one ⟨Φ₁⟩=0 row.
- **Solver stack (`solver.F90`).** SNES Newton (1 iteration for linear runs; true Newton for
  `includePhi1`); KSP GMRES, restart 2000, rtol `solverTolerance` (1e-6), preconditioned by a
  **complete LU factorization (MUMPS, else SuperLU_DIST) of a simplified "preconditioner
  matrix"** in which selected couplings are dropped. The knobs (all in the
  `preconditionerOptions` namelist, applied in `populateMatrix.F90`):
  `preconditioner_x` (0–4, default 1 = keep only diagonal-in-x collision coupling),
  `preconditioner_x_min_L`, `preconditioner_xi` (default 1 = drop L→L±1 streaming coupling),
  `preconditioner_species` (default 1 = self-collisions only), `preconditioner_theta`,
  `preconditioner_zeta` (default 0 = keep full), `preconditioner_theta_min_L`,
  `preconditioner_zeta_min_L`, `preconditioner_magnetic_drifts_max_L` (default 2),
  `reusePreconditioner` (default true). MUMPS is tuned (CNTL(1)=1e-6, ICNTL(14)=50) and on
  factorization failure ICNTL(14) is doubled and the whole solve retried, up to 1024.
- **Modes.** `RHSMode` 1 = physical-gradient solve; 2 = 3-RHS transport matrix; 3 =
  monoenergetic 2-RHS transport matrix (forces Nx=1, pitch-angle scattering, DKES E×B,
  nuPrime/EStar inputs); 4/5 = adjoint sensitivity modes. `whichMatrix` 0 = preconditioner,
  1 = Jacobian, 2/3 = residual pieces, 4/5 = adjoint (`discreteAdjointOption=.true.` default
  uses `KSPSolveTranspose` — the discretely-consistent adjoint).
- **Er / ambipolarity (`ambipolarSolver.F90`).** Radial field enters as `dPhiHatdpsiHat`;
  `ambipolarSolve` finds Er with Σ_a Z_a Γ_a = 0 by Brent (default, `ambipolarSolveOption=2`,
  bracket [Er_min, Er_max] expanded until sign change) or Newton using the adjoint dJr/dEr.
- **Geometry (`geometry.F90`, 3.6 kLOC).** `geometryScheme`: 1 = analytic 3-helicity Boozer
  model; 2/3 = LHD; 4 = W7-X model; 5 = VMEC `wout` file; 11/12 = Boozer `.bc` file;
  13 = namelist bmnc/bmns (STELLOPT adjoint path).
- **Outputs (`writeHDF5Output.F90`, 1.9 kLOC).** The full field list — grids and weights,
  geometry arrays, `particleFlux_{vm0,vm,vE0,vE,vd1,vd}_{psiHat,psiN,rHat,rN}` (likewise
  momentum/heat, plus `_vs_x` and `BeforeSurfaceIntegral` variants), classical fluxes,
  `FSABFlow`, `FSABjHat[OverB0|OverRootFSAB2]`, `NTV`, `Phi1Hat`, `transportMatrix`,
  `elapsed time (s)`, `finished` — is the netCDF/HDF5 parity contract for §4.
- **stdout.** Banner → per-namelist "Successfully read parameters from …" → physics-parameter
  block → grid summary (`Ntheta = …`, `The matrix is N x N elements.`) → solver banner
  ("mumps detected", "Beginning the main solve. This could take a while ...") → KSP residual
  monitor lines → per-species flux table (`Results for species i:`, `FSABFlow: …`,
  `FSABjHat (bootstrap current): …`) → `Goodbye!`. These blocks are the print-parity contract.
- **Docs.** `/Users/rogerio/local/sfincs/doc/` holds the user manual LaTeX + PDF and the
  definitive technical notes: `20150507-01 Technical documentation for version 3`,
  `20150402-01 Implementation of the Fokker-Planck operator`, Φ₁ implementation notes,
  classical-flux notes, DKES-comparison notes. Extract every equation we re-implement into
  our docs (§8) from these plus the papers in §1.
- **Tests.** `examples/` has ~40 cases (tokamak, HSX, W7-X, analytic schemes 1–5/11/12,
  1–3 species, PAS vs FP, Er on/off, Φ₁, transport-matrix); each `tests.py` asserts HDF5
  fields to ~0.3% relative tolerance via `shouldBe`.

### 0.2 What sfincs_jax is today (audited 2026-07-08, branch `refactor/v3-driver-architecture`)

Correcting the folklore: **`v3_driver.py` no longer exists.** It was a 2.5 MB monolith,
already split on this branch into `problems/` (54 kLOC), `solvers/` (34 kLOC), `operators/`
(17 kLOC), `outputs/` (7 kLOC), `validation/` (10 kLOC), `workflows/`, `geometry/`,
`discretization/`, `physics/`, plus 16 root modules (`cli.py` alone is 50 kB). The real
problems now:

1. **Size and legibility.** 115 library files / 137 kLOC; 300 test files / 118 kLOC; 90
   example files. Six-times larger than the 22 kLOC Fortran it ports. ~20
   `solvers/preconditioner_*` files, ~15 `SFINCS_JAX_RHS1_FULL_CSR_*` env-var-gated
   experimental routes, ~40 `NotImplementedError` stubs. `plan_final.md`'s own verdict:
   the blocker is reviewability, not physics.
2. **Performance profile (from the README's 39-case benchmark table).** sfincs_jax is
   already *faster* than Fortran on most parity cases (often 10–100×, e.g. W7-X 2-species
   128.5 s → 0.98 s warm). The genuine weak spots: (a) Φ₁/quasineutrality Fokker–Planck
   tokamak cases (0.30×–0.75× ratios — barely faster or break-even); (b) **memory**: 2–9×
   the Fortran active-solver RSS, up to ~15–20 GB peak on high-ν W7-X FP cases; (c) the
   **QH finite-beta bootstrap-current profile at production resolution does not have a
   closed convergence lane** (QA is close; QH shows up to ~24% refinement drift at reduced
   grids); (d) GPU memory fragmentation on 4–8 device sharding (experimental).
3. **Solver backends.** JAX-native Krylov + scipy SuperLU (`splu`) host path. **No
   petsc4py, no MUMPS, no SuperLU_DIST** (contrary to what the old plans imply — MUMPS
   appears only in comments and in parsing Fortran logs). This is fine: the fix is better
   structure, not more bindings (§2.3).
4. **Differentiability is real but shallow.** `jax.jvp`/`jax.vjp` paths,
   `sensitivity.py::vjp_flux`, a `differentiable=True` implicit solve path, one
   `value_and_grad` optimization workflow. No `custom_vjp`, no lineax at runtime (test-only
   optional), no implicit-function-theorem wrapper around the production solver.
5. **Output gaps.** `outputs/writer.py` covers the supported 39 cases completely, but
   RHSMode-2/3 transport diagnostics are vm-only: the `vE/vd/Phi1`-related transport-flux
   families are not emitted yet. Non-stellarator-symmetric VMEC (bmns/gmns) unsupported.
6. **Repo bloat.** Working tree 108 MB (docs/_static 14 MB of figures, docs/upstream 5.7 MB
   of third-party PDFs, tests/ref 12 MB of .h5/.petscbin, examples 2.5 MB). `.git` is 62 MB,
   and the top-30 largest historical blobs are ALL revisions of the deleted `v3_driver.py`
   (786 copies at ~2.5 MB). History rewrite (Phase 7) collapses this.
7. **Packaging is already close.** Deps unpinned (`jax, numpy, scipy, h5py, netCDF4,
   matplotlib`), PyPI publishing workflow exists, CI has 4-way sharded coverage but gates at
   80% (target: 95%), docs are sphinx-rtd with ~38 .rst pages but not pedagogic.
8. **Branches.** Four intentional `research/*` archive branches exist
   (qi-device-hard-seed, native-sparse-direct, parallel-performance, publication-audits) —
   keep them; they are the designated home for experimental code deleted from the core.

### 0.3 End-state goals (definition of done)

- **G1.** A standalone repo `github.com/uwplasma/SOLVAX` (§3) holding all reusable
  linear-algebra / solver / preconditioner / matrix-free / implicit-diff machinery,
  JAX-native, differentiable, published on PyPI as `solvax`, consumed by sfincs_jax and
  usable by NTX, vmec_jax, SPECTRAX/spectrax-gk, NEOPAX, ESSOS (§3.0 documents their
  audited needs).
- **G2.** sfincs_jax consolidated to ≤ 40 library files / ≤ 30 kLOC (stretch: 30/20k) with
  the §4 layout: physics/numerics/io/api separation, no experimental env-var routes, ≥95%
  coverage, Fortran-parity prints and output files, simsopt-style examples.
- **G3.** High-resolution QA **and QH** bootstrap-current profile runs converge on CPU and
  GPU with runtime ≤ 1.5× Fortran+MUMPS on every README benchmark case (most already beat
  it; the gate binds on Φ₁/QN FP cases) and peak RSS ≤ 2× Fortran (today 2–9×).
  Monoenergetic-mode cases must beat Fortran outright via the structured direct solver.
- **G4.** End-to-end differentiability of the library path (`jax.grad` through geometry →
  operator → solve → fluxes/⟨j‖B⟩ → ambipolar Er) via implicit differentiation, verified
  against finite differences and against published adjoint-DKE sensitivity results.
- **G5.** Rebuilt pedagogic docs (Read the Docs) with equations, derivations, algorithm
  pages, complete input/output reference, and verified clickable citations; README ≤ ~250
  lines.
- **G6.** Repo ≤ 10 MB after history rewrite; contributors page = rogeriojorge only
  (verified, expected already true); CI (tests, docs, coverage ≥95%, PyPI trusted
  publishing) green.

---

## 1. Literature and methods (verified 2026-07-08; every link fetched)

Fold the derivations into `docs/` (§8), not into this plan. Corrections found during
verification are baked in below — do not "fix" them back.

### 1.1 Physics

- **DKE & neoclassical theory.** Helander & Sigmar, *Collisional Transport in Magnetized
  Plasmas*, Cambridge UP (2002) — collision-operator conservation laws and Onsager symmetry
  are the properties our discrete tests assert. The SFINCS paper: Landreman, Smith, Mollén,
  Helander, *Phys. Plasmas* **21**, 042503 (2014), https://arxiv.org/abs/1312.6058 — defines
  the four trajectory models; keep the trajectory-model switch first-class (the DKES-like
  E×B model has known resonance artifacts at large Er that users probe deliberately).
  Speed grid: Landreman & Ernst, *J. Comput. Phys.* **243**, 130 (2013),
  https://arxiv.org/abs/1210.5289.
- **Monoenergetic coefficients.** Hirshman et al., *Phys. Fluids* **29**, 2951 (1986);
  van Rij & Hirshman, *Phys. Fluids B* **1**, 563 (1989) (DOI 10.1063/1.859116) — define the
  DKES D_ij conventions. Benchmark targets: Beidler et al., *Nucl. Fusion* **51**, 076001
  (2011), https://doi.org/10.1088/0029-5515/51/7/076001 (the ICNTS multi-code dataset).
- **Legendre block-tridiagonal direct solution of the monoenergetic DKE.** Escoto, PhD
  thesis (2025), https://arxiv.org/abs/2510.27513 — full derivation of the block-tridiagonal
  structure in Legendre index l, the forward–backward block elimination, adjoint properties,
  Onsager symmetry, and application to fast bootstrap-current convergence scans. Cite this
  thesis plus the block-LU literature (Golub & Van Loan §4.5; Demmel, Higham & Schreiber,
  *Numer. Linear Algebra Appl.* **2**, 173 (1995) — stability of block LU without pivoting
  requires block diagonal dominance, which weakens as ν→0: pivot *within* blocks and monitor
  conditioning). **Adopt as our tier-1 kernel (§2.3).**
- **Bootstrap current.** Redl et al., *Phys. Plasmas* **28**, 022502 (2021) — analytic
  cross-check. Landreman, Buller, Drevlak, *Phys. Plasmas* **29**, 082501 (2022),
  **https://arxiv.org/abs/2205.02914** (NOT 2205.04954 — that ID is an unrelated math paper)
  — defines the precise-QA/QH self-consistent-bootstrap use case and our headline example.
- **Ambipolar Er.** Turkin et al., *Phys. Plasmas* **18**, 022505 (2011) (NTSS) and
  Velasco et al., https://arxiv.org/abs/1908.11615 (KNOSOS, JCP 2020) — root structure
  (ion/electron/unstable) demands a bracketed solver over the physical Er range, not bare
  Newton; D_ij(Er) tabulation + interpolation is the standard fast pattern.
- **Impurities & temperature screening.** Mollén et al., *Phys. Plasmas* **22**, 112508
  (2015), https://arxiv.org/abs/1504.04810 — full inter-species FP coupling is
  non-negotiable for impurity work. Martin & Landreman, *J. Plasma Phys.* **86** (2020),
  https://arxiv.org/abs/2002.09731 (screening degrades away from quasisymmetry — this is
  the *Martin & Landreman* paper, previously misattributed in plan drafts). Φ₁ effects on
  impurity transport: García-Regaña et al., https://arxiv.org/abs/1501.03967; Calvo et al.,
  https://arxiv.org/abs/1804.11104.
- **Adjoint neoclassical optimization.** Paul, Abel, Landreman, Dorland, *J. Plasma Phys.*
  **85** (2019), https://arxiv.org/abs/1904.06430 — the G4 acceptance test: our
  implicit-diff gradients must reproduce their sensitivity benchmarks. Note SFINCS itself
  ships a discrete adjoint (RHSMode 4/5, `KSPSolveTranspose`) we can cross-check against.

### 1.2 Numerics (what actually fixes convergence and memory)

- **Why plain GMRES stalls.** At low ν the discrete DKE is convection-dominated and highly
  non-normal; restarts destroy superlinear convergence. Fortran survives because its
  preconditioner is a *complete LU of a physics-simplified operator*. Any pure-JAX plan
  that ignores this loses. Remedies, in the order we adopt them:
  - **Deflated/recycled Krylov.** Morgan, "GMRES with Deflated Restarting", *SISC* **24**,
    20 (2002), https://doi.org/10.1137/S1064827599364659; Parks, de Sturler et al.,
    "Recycling Krylov Subspaces for Sequences of Linear Systems", *SISC* **28**, 1651
    (2006), https://doi.org/10.1137/040607277 — recycle deflation spaces across Er-scan /
    Newton / optimization steps, exactly our workload.
  - **Kronecker-structured preconditioning.** Van Loan & Pitsianis, "Approximation with
    Kronecker Products" (1993), https://link.springer.com/chapter/10.1007/978-94-015-8196-7_17
    — the DKE operator is close to (streaming ⊗ I_x) + (I_θζξ ⊗ collision); nearest-KP
    factors inherit bandedness and are matrix-free-applicable and differentiable.
  - **p-multigrid for spectral operators.** Fischer et al., https://arxiv.org/abs/2110.07663;
    Thompson et al., https://arxiv.org/abs/2108.01751 — coarsen in resolution/order with
    Chebyshev or weighted-Jacobi smoothing; all levels are dense tensor-product ops, ideal
    for JAX.
  - **Mixed-precision iterative refinement.** Carson & Higham, *SISC* **40**, A817 (2018),
    https://doi.org/10.1137/17M1140819 — factor blocks in fp32 (fast on GPU), refine with
    fp64 residuals, GMRES-IR when ill-conditioned.
  - **Physics-based (moment/coarse-operator) preconditioning for kinetic Jacobians.**
    Chen & Chacón, https://arxiv.org/abs/1309.6243 — precondition the kinetic operator with
    a simplified/fluid operator; the neoclassical analog is preconditioning full
    trajectories with the DKES-like or coupling-dropped operator (exactly the SFINCS knobs).
- **Differentiable solves.** Never backprop through Krylov iterations. IFT/adjoint:
  Blondel et al., "Efficient and Modular Implicit Differentiation",
  https://arxiv.org/abs/2105.15183. Structured-adjoint precedent for spectral solvers:
  Skene & Burns, "Fast automated adjoints for spectral PDE solvers", *PNAS* (2026),
  **https://arxiv.org/abs/2506.14792** — the adjoint of a structured spectral solve should
  reuse the same structured machinery transposed, not a generic AD tape. Our design (adjoint
  solve with the transposed preconditioner, VJPs only through operator *application*)
  matches; cite both in docs.
- **JAX ecosystem facts that constrain design (verified mid-2026).**
  - `lineax` (https://github.com/patrick-kidger/lineax) is mature: `AbstractLinearOperator`,
    LU/QR/GMRES/BiCGStab, IFT differentiation, structure tags. **It does not provide**
    block-tridiagonal direct solves, Kronecker/coarse-operator preconditioners, or
    deflated/recycled GMRES — that is precisely SOLVAX's niche; build *on* lineax's operator
    abstraction rather than beside it.
  - `jax.experimental.sparse` remains experimental; `sparse.linalg.spsolve` is CSR-only,
    single-RHS, no vmap. **Do not build around JAX sparse direct solves**; build around
    dense-block structure + matrix-free applies, with host scipy SuperLU as the CPU direct
    fallback.
  - `jax.lax.linalg.lu` batches over leading dims on CPU (threaded LAPACK) and GPU (batched
    cuSOLVER) — the workhorse for vmapped block elimination. `shard_map` is stable; the
    persistent compilation cache is standard practice and matters for our large jaxprs.
  - `equinox`, `optimistix`, `optax` healthy; optimistix Newton/LM is the outer loop for Φ₁
    and ambipolar root-finding. (Known wild bug: persistent-cache + vmap miscompile
    affecting `optimistix.minimise`, jax#31733 — add a regression canary test.)
  - Fresh 2026 prior art for "differentiable solver via custom_vjp around native libraries"
    exists (petsc4py/DLPack bridges, differentiable AMG), but nothing occupies the
    structured-kinetic/block-tridiagonal/recycled-Krylov pure-JAX niche. SOLVAX has a clear
    gap to fill; say so in its README.

### 1.3 Dependency policy

`pyproject.toml` dependencies with **no version pins** ("jax", not "jax>=0.4"), both repos:
sfincs_jax: `jax, numpy, scipy, h5py, netCDF4, f90nml, equinox, optimistix, optax, solvax`
(matplotlib moves to an extra). SOLVAX: `jax, equinox, lineax`. Extras: `[native] = scipy`
(+ document petsc4py/pypardiso as opt-in, never hard deps), `[dev]`, `[docs]`.
`requires-python = ">=3.10"` is allowed (not a dep pin).

---

## 2. Architecture and design decisions

### 2.1 Two-package split

- **SOLVAX** (§3): everything with no neoclassical physics in it — operators, Krylov,
  structured direct solves, preconditioners, implicit diff, mixed precision, native bridges.
- **sfincs_jax** (§4): physics (geometry, species, collisions, DKE terms, moments, Er, Φ₁),
  I/O (namelist, VMEC/Boozer, HDF5/netCDF, Fortran-parity prints), API, CLI, examples,
  optimization glue. It consumes SOLVAX.

Placement rule: "would NTX / vmec_jax / spectrax-gk want this function unchanged?" → SOLVAX.
"Does it know what a Maxwellian or a flux surface is?" → sfincs_jax.

### 2.2 Operator representation — one source of truth, three consumers

A `KineticOperator` (equinox module) holds per-term matrix-free applies **and** per-term
structured-block extractors:

- **Directional applies as batched dense matmuls**: each 1-D derivative is applied by moving
  the target axis last and hitting it with the (small, dense or banded) differentiation
  matrix — XLA sees batched GEMMs, never a global sparse matrix. Keep SFINCS's default
  discretizations for parity (4th-order FD in θ/ζ, Legendre modal ξ, Landreman–Ernst x);
  the *preconditioner* may legally use different, more dissipative stencils (upwinded,
  lower order) — the operator/PC split makes that a free knob.
- **Streaming/mirror**: banded couplings in l (l→l±1 with factors l/(2l−1), (l+1)/(2l+3)),
  diagonal in x for DKES trajectories, banded in x for full trajectories.
- **Collisions**: pitch-angle diagonal in l (ν_D l(l+1)/2) plus dense-in-x blocks per
  (species-pair, l) from Rosenbluth potentials — identical math to `xGrid.F90`,
  tested element-wise against Fortran dumps.
- **Constraints/sources as a bordered system.** Assemble `[[A, B], [C, 0]]` where B holds
  the per-species particle/heat source shapes ((x²−5/2)F_M, (1−2x²/3)F_M) and C the
  flux-surface-average density/pressure constraints (mirroring `constraintScheme` 1/2).
  Scale pinned/constraint rows by mean(|streaming coefficient|)/h so no equation is orders
  of magnitude off.

The same module (a) applies matrix-free for Krylov, (b) emits Legendre-block-tridiagonal
dense blocks for the structured direct solver/preconditioner, (c) emits a host CSR for the
CPU sparse-direct path. Golden test: matrix-free apply ≡ materialized matrix·vector to 1e-13,
and per-term entries ≡ Fortran `populateMatrix` entries on a tiny grid (§5, Phase 3.2).

### 2.3 Solver strategy (the fix for high-resolution QA/QH and the memory bill)

Auto-policy in `sfincs_jax/solve.py`, all kernels living in SOLVAX:

1. **Tier 1 — structured direct (block Thomas over Legendre modes).** For the
   monoenergetic/pitch-angle-scattering/DKES-trajectory family the operator is exactly
   block-tridiagonal in l with dense (Nθ·Nζ) blocks. Forward Schur recursion from l = Nξ−1
   down: Δ_{Nξ−1} = D_{Nξ−1}; X_{l+1} = Δ_{l+1}⁻¹ L_{l+1} (one LU solve); Δ_l = D_l −
   U_l X_{l+1} (one GEMM). **Store only the l ≤ 2 blocks and their LU factors** — the
   sources have no Legendre content above l=2 and the flux/flow/bootstrap moments only touch
   f⁽⁰⁾, f⁽¹⁾, f⁽²⁾, so backward substitution stops at l=2. Memory O((NθNζ)²) *independent
   of Nξ*; time O(Nξ·(NθNζ)³) as batched GEMM/LU — GPU-optimal via `lax.scan` over l and
   `vmap` over (species, x, ν, Er). Null space (constant on the surface) fixed by a
   single-point pin: overwrite one row of D₀ with a scaled identity row, zero the matching
   U₀ row and source entry — square, differentiable, no augmented system. Solve both RHS
   (radial drive; parallel drive) through the same factorizations. Exploit Onsager symmetry
   (D₂ⱼ = D₁ⱼ, D_ᵢ₂ = D_ᵢ₁) so only D₁₁, D₁₃, D₃₁, D₃₃ are computed.
2. **Tier 2 — preconditioned, recycled Krylov on the full operator.** Flexible GMRES core
   (Givens least-squares, classical Gram–Schmidt applied twice — MGS buys nothing on GPU)
   inside GCROT(m,k)-style outer recycling; recycled subspaces (C, U) and the solution are
   returned as explicit warm-start state for Er/ν continuation, Newton steps, and
   optimization iterations. Eisenstat–Walker inner-tolerance forcing; a guaranteed-decrease
   line-search on the update so an imperfect preconditioner can never increase the residual;
   a "linear-preconditioner" memory mode that skips storing the preconditioned basis and
   reconstructs it by linearity. Right-preconditioners, composable:
   - **Coarse-operator LU (default, = what saves Fortran).** Apply tier-1 to the
     SFINCS-simplified operator: drop x-coupling in collisions above `preconditioner_x_min_L`
     (block-diagonal over speed nodes), drop inter-species coupling, optionally drop l→l±1
     streaming — mirroring the Fortran `preconditioner_*` namelist knobs with the same
     names and defaults, so namelists behave identically.
   - **p-multigrid** on (θ, ζ, ξ) (never coarsen x or species): ~2.5× coarsening per level
     down to ~10⁴ unknowns, exact LU at the coarsest level; smoother = alternating-direction
     line block-Jacobi that exactly inverts the block-diagonal along one coordinate at a
     time, rotating through axis orderings (cheap axis permutations, coupled axis last);
     per-line blocks solved by a **non-pivoted banded LU with static pivoting (clamp tiny
     pivots to √ε), row equilibration, and a Sherman–Morrison capacitance correction for
     periodic wrap-around** — robust for advection-dominated periodic lines without
     pivoting, which XLA hates; collisionality-dependent under-relaxation weights
     precomputed offline and interpolated in log₁₀(ν*).
   - **Mixed precision**: factor/apply PC blocks in fp32, keep Krylov residuals fp64,
     iterative refinement on top (Carson–Higham).
3. **Tier 3 — host sparse-direct fallback (CLI fast path, non-differentiable).** Emit CSR
   once, hand to scipy SuperLU (always available; optional petsc4py/MUMPS or pypardiso when
   importable, import-guarded, conda-documented, never a hard dep) outside jit, with a loud
   one-line notice matching the Fortran solver prints. This is the "always converges" net
   and often the fastest single-solve CPU path.

**Continuation policy** (structural cure for "struggling high-resolution runs"): for each
surface, order solves high-ν → target-ν and Er=0 → target-Er; reuse previous f as x0,
previous PC factorization (defect correction, mirroring `reusePreconditioner`), and recycled
Krylov subspaces. Nonlinear Φ₁ runs are optimistix Newton over the same linear kernel with
the same reuse.

**Memory rules** (attacks the 2–9× RSS gap): never materialize the full operator on device;
PC blocks in fp32; `donate_argnums` on the CLI path; padding to friendly shapes with one
compile per resolution; persistent compilation cache on by default.

### 2.4 Differentiability boundary

- Library path (`sfincs_jax.api`): fully differentiable — profiles, Er, B-Fourier
  coefficients → fluxes, flows, ⟨j‖B⟩ — via `custom_linear_solve`/IFT in SOLVAX; the adjoint
  solve reuses the transposed operator *and transposed preconditioner* (closed-form
  transposes on every SOLVAX operator, including multigrid = swap prolong/restrict); the
  ambipolar root is differentiated implicitly through Σ Z_a Γ_a(Er) = 0.
- CLI path: same functions, may select tier 3, no gradient guarantee, jits with donation for
  peak speed/memory.

### 2.5 Naming conventions

Physics-meaningful, PEP8: `n_theta, n_zeta, n_xi, n_x, n_species`, `nu_prime`,
`bootstrap_current_density`, `particle_flux_psi`, `heat_flux_psi`, `parallel_flow`,
`collisionality`. Fortran namelist/HDF5 names survive only in the compatibility layer
(`io/`), backed by a `docs/reference/sfincs_name_map.md` table mapping every Fortran name →
Python name (also used by the writer to emit Fortran-compatible variable names).

---

## 3. New shared repository: `github.com/uwplasma/SOLVAX`

**Name decision:** repo `SOLVAX`, PyPI dist `solvax`, import `solvax` — "structured operators
and linear SOLVers in jAX". Fits the uwplasma standalone-library family (ESSOS, SPECTRAX,
NEOPAX, NTX, MHX) while the `_jax` suffix family stays for ports (sfincs_jax, vmec_jax,
booz_xform_jax). Verified **free on PyPI 2026-07-08** (fallbacks, also all free: `krylax`,
`matfreejax`, `uwsolve`). Scope statement must distinguish it from NEOPAX (which owns the
"neoclassical transport in JAX" slot): SOLVAX is *generic* solver infrastructure. Tagline:
*"Differentiable structured linear solvers, preconditioners and matrix-free methods in JAX."*
License MIT, author "UW Plasma", `rogerio.jorge@wisc.edu`.

### 3.0 Audited consumers and what they need (drives the API)

- **sfincs_jax** — everything in §2.3.
- **NTX** — already ships a dense block-tridiagonal Schur solve with a hand-written
  `custom_vjp` adjoint (`src/ntx/_solver_*.py`) and its plan explicitly flags evaluating
  lineax. SOLVAX tier-1 + implicit diff replaces that bespoke stack outright. NTX is also
  the packaging template (hatchling, src-layout, inline ruff/mypy/pytest config, sharded CI).
- **vmec_jax** — uses lineax `FunctionLinearOperator` + BiCGStab with a hand-rolled
  IFT adjoint and CG fallback; wants the matrix-free `solve()` + fallback policy as a
  first-class config instead of per-repo boilerplate.
- **spectrax-gk** — GMRES, Arnoldi, shift-invert eigenpairs, pluggable preconditioners, and
  is already benchmarking lineax; SPECTRAX has a Newton-GMRES implicit integrator.
  → Newton–Krylov and (roadmap) shift-invert eigensolver interfaces.
- **ESSOS / JAX-in-Cell** — nonlinear root/least-squares with implicit diff; small dense
  solves. → thin `root_solve` / factorize-reuse APIs.

Common denominators: PyTree-in/PyTree-out; jit/vmap/grad-transparent; iteration/residual
stats without breaking tracing; factorize/solve split for RHS and scan reuse; graceful
fallback chains.

### 3.1 Layout

```
SOLVAX/
  src/solvax/
    __init__.py        # public API: solve, operators, direct, precond, implicit, native
    operators.py       # MatrixFreeOperator, SumOperator, KroneckerOperator,
                       # BlockTridiagonalOperator, BorderedOperator (constraint rows with
                       # analytic Schur projection I − B(CB)⁻¹C), closed-form transpose()
                       # for every class, materialize() -> dense / host CSR
    direct.py          # batched dense LU/QR (lax.linalg.lu); block_thomas: forward Schur
                       # recursion via lax.scan, truncated storage (keep-lowest-K blocks),
                       # multi-RHS through shared factors, factorize()/solve() split
    banded.py          # banded + periodic-banded non-pivoted LU: static pivoting,
                       # row equilibration, Sherman–Morrison periodic correction
    krylov.py          # fgmres core (cgs2 + Givens); gcrot(m,k) with recycled-subspace
                       # (C,U) state in/out; deflated restarts; Eisenstat–Walker forcing;
                       # guaranteed-decrease refinement; linear-PC memory mode;
                       # lineax AbstractLinearSolver adapters
    precond.py         # identity, jacobi, block_jacobi, line-smoother (alternating-
                       # direction block-Jacobi over axis orderings), coarse_operator
                       # (LU/block_thomas of a user-supplied simplified operator),
                       # p-multigrid (restrict/prolong per axis, Chebyshev/weighted-Jacobi
                       # or line smoothers, exact coarse LU), kronecker_nkp,
                       # mixed_precision wrapper
    refine.py          # iterative refinement / defect correction, fp32-factor+fp64-residual
    implicit.py        # custom_vjp linear solve (IFT; adjoint reuses transposed PC);
                       # differentiable root_solve (scalar & vector; for ambipolarity-type
                       # conditions); newton_krylov built on optimistix
    native.py          # import-guarded host bridges: scipy.sparse.linalg.splu (always),
                       # pypardiso, petsc4py KSP+PC (mumps/superlu_dist). Pure-JAX
                       # fallbacks always exist; loud fallback notices; never hard deps
    utils.py           # padding policies, dtype policies, Solution pytree (x, stats),
                       # timers, stats plumbing that survives jit
  tests/               # ≥95% coverage; property tests: block_thomas ≡ dense solve 1e-12;
                       # adjoint consistency ⟨Ax,y⟩=⟨x,Aᵀy⟩ for every operator; PC cuts
                       # manufactured advection-dominated GMRES iterations >500 → <30;
                       # grad(solve) ≡ finite differences 1e-6 at ~1 extra solve (assert
                       # via stats); banded-periodic LU vs scipy on random periodic systems;
                       # mixed-precision refinement recovers fp64 accuracy
  docs/                # RTD (furo + myst + mathjax + bibtex): one theory page per module
                       # with citations (Saad; Golub & Van Loan; Morgan 2002; Parks 2006;
                       # Demmel-Higham-Schreiber 1995; Van Loan & Pitsianis; Carson-Higham
                       # 2018; Blondel 2021; Skene & Burns 2026); examples: Poisson,
                       # advection–diffusion, kinetic-like block-tridiagonal demo
  examples/
  .github/workflows/   # tests.yml (ubuntu+macos matrix), docs.yml, publish.yml
                       # (trusted publishing: environment pypi, id-token: write,
                       # pypa/gh-action-pypi-publish@release/v1 — NOT inside a reusable
                       # workflow; attestations on by default)
  pyproject.toml       # hatchling; deps: jax, equinox, lineax (unpinned);
                       # extras [native]=[scipy], [dev], [docs]
  .readthedocs.yaml    # version: 2, ubuntu-24.04, python 3.12, sphinx conf docs/conf.py
  README.md, LICENSE (MIT), CITATION.cff, codecov.yml (target 95%)
```

Design invariants: everything jit/vmap-able except `native.py`; every solver returns a
`Solution` pytree (x, stats: iterations, residual history, dtype path, recycled state); all
public functions carry doctested examples; no global state; closed-form `transpose()`
everywhere so adjoints never differentiate through while_loops.

### 3.2 API sketch (what sfincs_jax calls)

```python
import solvax as sx

A  = sx.SumOperator([streaming, mirror, collisions])            # matrix-free physics op
K  = sx.BorderedOperator(A, sources, constraints)               # constraint rows
M  = sx.precond.coarse_operator(A_simplified.to_block_tridiagonal(),
                                factor=sx.direct.block_thomas)  # SFINCS-style PC
sol = sx.solve(K, b, solver=sx.krylov.gcrot(m=200, k=20, rtol=1e-6),
               preconditioner=M, x0=prev.x, recycle=prev.stats.recycle)  # differentiable
f   = sx.direct.block_thomas(L, D, U, rhs, keep_lowest=3)       # tier-1 exact kernel
x   = sx.native.splu_solve(K.to_csr_host(), b)                  # CLI fast path
```

### 3.3 Acceptance criteria for solvax v0.1.0

- block_thomas ≡ dense solve to 1e-12; linear scaling in N_blocks demonstrated; truncated
  storage verified (memory independent of N_blocks); CPU/GPU benchmark plot in docs
  (compressed < 150 kB).
- Coarse-operator-PC GMRES solves a manufactured advection-dominated problem in < 30
  iterations where unpreconditioned needs > 500 (regression test).
- `jax.grad` through `sx.solve` matches FD to 1e-6 and costs ~1 extra solve.
- Recycling demo: a 10-step parameter continuation with recycling uses ≤ 50% of the
  iterations of cold restarts (regression test).
- Published to PyPI via trusted publishing; RTD live; repo < 5 MB; coverage ≥ 95%.
- One PR each lands in NTX and spectrax-gk replacing a bespoke solve with solvax (validates
  the API against real consumers; can trail v0.1.0 but must precede sfincs_jax v2 release).

---

## 4. sfincs_jax: target package layout

Hard targets: ≤ 40 library files / ≤ 30 kLOC (from 115 / 137k); tests ≤ 100 files / ≤ 50 kLOC
at ≥95% coverage (from 300 / 118k); examples = §7's nine scripts + the Fortran-parity
namelist set; `scripts/` deleted; zero env-var-gated solver routes (auto-policy or delete —
experimental material goes to the existing `research/*` branches).

```
sfincs_jax/
  __init__.py        # re-export: run, Simulation, Results, Geometry, SpeciesSet, __version__
  api.py             # Simulation (equinox Module: geometry+species+grids+settings);
                     # run(namelist | Simulation) -> Results; results.to_netcdf/.to_hdf5
  cli.py             # `sfincs_jax input.namelist`: Fortran-parity prints, tier-3-eligible
                     # fast path; subcommands: run, scan-er, ambipolar, transport-matrix,
                     # plot, compare  (collapse today's 11 subcommands into these 6)
  constants.py       # normalizations (B̄, R̄, n̄, T̄, Delta, alpha, nu_n), physical constants
  geometry.py        # flux-surface geometry from all geometrySchemes (1–5, 11–13):
                     # B, dB/dθ, dB/dζ, jacobian, drifts; differentiable w.r.t. bmnc/bmns
  io/
    namelist.py      # f90nml read/validate; defaults identical to readInput.F90 +
                     # validateInput.F90 rules
    geometry_files.py# VMEC wout + Boozer .bc readers (add bmns/gmns non-symmetric support)
    output.py        # HDF5/netCDF/NPZ writer with the Fortran variable-name map; closes
                     # the vE/vd/Phi1 transport-field gaps; parity checker utility
    prints.py        # replicates the version3 stdout blocks; golden-tested vs Fortran logs
  species.py         # Species/SpeciesSet pytrees: Z, m, n, T, dn/dψ, dT/dψ; collisionality
  grids.py           # θ/ζ grids + FD/spectral D-matrices, Legendre ξ machinery,
                     # Landreman–Ernst x nodes/weights, Nxi_for_x ramps, quadratures
  collisions.py      # pitch-angle + full FP x-blocks (Rosenbluth potentials); tested
                     # element-wise vs Fortran xGrid dumps
  dke.py             # KineticOperator: all trajectory/drift/Er terms, RHS drives,
                     # sources/constraints — the physics core (from operators/ + problems/)
  moments.py         # fluxes, flows, ⟨j‖B⟩, FSAB2, density/pressure perturbations, NTV,
                     # monoenergetic D_ij, classical transport
  er.py              # ambipolarity: residual(Er), Brent bracket + Newton-with-adjoint,
                     # root classification (ion/electron/unstable), differentiable via
                     # solvax.implicit.root_solve
  phi1.py            # nonlinear Φ₁ quasineutrality Newton loop (optimistix over the same
                     # linear kernel, PC reuse)
  solve.py           # §2.3 auto-policy + continuation/warm-start orchestration on solvax
  optimize.py        # composable objectives (bootstrap, flux, screening metrics) + thin
                     # optax/scipy glue — one file, not a package
tests/               # unit + parity + gradient tests (§6); assets >200 kB live in the
                     # GitHub release `sfincs-jax-data-v1` (fetch+cache, offline-skip)
examples/            # §7
docs/                # §8
.github/workflows/   # ci.yml (coverage gate 95%), docs.yml, publish.yml (trusted publishing)
pyproject.toml, .readthedocs.yaml, README.md, LICENSE, CITATION.cff, codecov.yml
```

Migration map (audit-grounded):

| Current | Destination |
|---|---|
| `problems/profile_*` policy/solve/dense/sparse files (~25 kLOC) | `solve.py` + solvax (most policy tables die; auto-policy §2.3) |
| `problems/transport_*` | `solve.py` (RHSMode 2/3 = loop of tier-1/2 solves) + `moments.py` |
| `solvers/preconditioner_*` (~20 files) | `solvax.precond` (the ~4 that earn their keep) — rest deleted (ideas already preserved on `research/native-sparse-direct`) |
| `solvers/krylov*.py`, `native_block_factor.py`, `explicit_sparse*.py`, `memory_model.py` | `solvax.krylov/direct/native/utils` or delete |
| `operators/profile_*` (17 kLOC) | `dke.py` (+ `collisions.py`), term-by-term with parity tests |
| `outputs/` (writer/rhsmode1/transport/formats) | `io/output.py` |
| `validation/` (10 kLOC) | `tests/` + a small `tools/generate_reference_data.py`; release-asset fetch logic kept (it already works) |
| `workflows/optimization.py` | `optimize.py` + examples |
| root `ambipolar.py, sensitivity.py, diagnostics.py, grids.py, input_compat.py, compare.py, plotting.py, profiling.py, io.py, paths.py` | `er.py`, `moments.py`, `grids.py`, `io/`, `cli.py` (compare/plot subcommands); `sensitivity.py` dies — implicit diff subsumes it |
| `geometry/`, `discretization/`, `physics/` | `geometry.py`, `grids.py`, `species.py`/`constants.py` |

Every module: top docstring stating physics/numerics purpose, the Fortran file(s) it
corresponds to, and doc links. Every public function: docstring with units and SFINCS
normalization conventions.

---

## 5. Execution phases

Work in this order. Each phase ends with green CI and a dated entry appended to
`PROGRESS.md` (untracked; never committed). Time-box exploration; ship structure, iterate.

### Phase 0 — Environments, Fortran baseline, golden data

- 0.1 Isolated envs (mamba/conda preferred over brew for PETSc reproducibility on macOS;
  brew fine for cmake/gfortran):
  `mamba create -n sfincs-fortran compilers openmpi "petsc=*=*mumps*" mumps-mpi superlu_dist
  hdf5 netcdf-fortran pkg-config make python h5py` — verify `pkg-config --libs PETSc` shows
  mumps + superlu_dist. Separate envs: `sfincs-jax` (python 3.12, `pip install -e
  '.[dev,docs]'`), and one per reference repo actually run (never share envs).
- 0.2 Build Fortran version3 against that PETSc (new `makefiles/makefile.conda`); run the
  examples suite. Free to add temporary verbose prints / matrix dumps to the Fortran for
  understanding (never pushed upstream); profile with `-log_view` at high resolution and
  record where time/memory go (feeds docs/performance page and §2.3 tuning).
- 0.3 Golden data: `tools/generate_reference_data.py` runs the Fortran binary over the case
  matrix {tokamak, W7-X, HSX, precise-QA, precise-QH} × {low, med, high resolution} ×
  {1,2,3 species} × {Er=0, Er≠0} × {PAS, full FP} × {RHSMode 1,2,3}, capturing stdout logs +
  sfincsOutput.h5 + PETSc binary matrices for tiny grids. Upload tarball to a GitHub release
  (extend the existing `sfincs-jax-data-v1` pattern → `reference-data-v2`); tests
  download+cache; nothing large enters git. Migrate the current 12 MB `tests/ref/` into the
  same release and delete from the tree.

### Phase 1 — Audit-verification and failure reproduction

- 1.1 Re-verify the §0.2 numbers on the day work starts (`cloc`, file table with
  keep/move-to-solvax/rewrite/delete verdicts — seed it from the §4 migration map).
- 1.2 Reproduce the failures: QH bootstrap profile at production resolution (25×51×100×4
  floor), the Φ₁/QN FP tokamak cases, and a high-ν W7-X FP memory case; capture
  time/RSS/iteration counts with the existing profiling hooks; confirm or amend the §2.3
  root-cause analysis in a temporary `docs/dev/failure_analysis.md` (folded into the
  performance docs later).
- 1.3 Survey all sibling solvers under `/Users/rogerio/local/` (own envs, run examples,
  read solver/preconditioner/collision code, profile). Produce a private ideas list mapped
  to SOLVAX/sfincs_jax modules. Ideas in, names out.

### Phase 2 — Build SOLVAX (§3)

2.1 scaffold (pyproject, CI, RTD, README, trusted publishing) → 2.2 operators + direct
(block_thomas first — it is the tier-1 kernel and the PC engine) → 2.3 banded + krylov →
2.4 precond (coarse-operator, then multigrid) → 2.5 implicit → 2.6 native → 2.7 docs +
examples → 2.8 tag v0.1.0, PyPI. sfincs_jax installs solvax unpinned (git+https during
development, PyPI after).

### Phase 3 — Refactor sfincs_jax core (§4)

- 3.1 Create the skeleton; port geometry/grids/collisions with element-wise tests vs Fortran
  intermediates (temporary Fortran dumps of B(θ,ζ), collision x-matrices, grid nodes).
- 3.2 Port `dke.py` term-by-term: matrix-free ≡ materialized test per term, plus comparison
  against Fortran `populateMatrix` entries on a tiny grid (write a Fortran patch dumping the
  PETSc matrix in ASCII for ~4·5·4·3; keep under `tools/fortran_patches/` — the single
  highest-value parity tool).
- 3.3 `solve.py` + `er.py` + `phi1.py` + `moments.py`; end-to-end parity: every field in
  sfincsOutput.h5 vs golden data (fluxes 1e-8 relative at matched resolution; document any
  principled differences). Close the output gaps: vE/vd/Phi1 transport families, bmns/gmns.
- 3.4 `io/prints.py`: byte-similar stdout (timing lines may differ); golden-log tests.
- 3.5 **The purge commit(s):** delete `problems/`, `solvers/`, `operators/`, `outputs/`,
  `validation/`, `workflows/` and superseded roots once parity is green. Consolidate tests
  to ≤100 files while holding coverage ≥95% (raise the CI gate from 80 in this phase).

### Phase 4 — Performance & the high-resolution fix

- 4.1 §2.3 tiers + continuation wired end-to-end; gate: high-res QA **and QH** bootstrap
  profiles converge on CPU; then GPU (no Metal — use a CUDA box if available, else ship the
  GPU path with a script-form test and document).
- 4.2 Profile (jax profiler + `-log_view` comparison): kill recompiles (static shapes),
  cache collision matrices across surfaces, fuse moment reductions, fp32 PC blocks,
  donation on CLI. Memory gate: ≤ 2× Fortran RSS on the benchmark table.
- 4.3 Regenerate the README benchmark table with `tools/benchmarks/` (Fortran vs CLI vs
  library, CPU/GPU, wall time + peak RSS); compressed plots (< 150 kB, optipng/pngquant).
  Gate: no case slower than 1.5× Fortran; monoenergetic cases win outright.

### Phase 5 — Examples (§7).
### Phase 6 — Documentation overhaul (§8).
### Phase 7 — Repo hygiene, history rewrite, size gate (§9). Do LAST, after merge.
### Phase 8 — Release: sfincs_jax v2.0.0 + solvax v0.1.x to PyPI (trusted publishing), RTD
builds, GitHub release notes summarizing the refactor and the benchmark table.

---

## 6. Testing, CI, coverage, publishing

- **Unit:** grids (quadrature exactness; Landreman–Ernst nodes vs Fortran to 1e-13),
  collisions (vs Fortran x-matrices; momentum/energy conservation of the FP operator),
  geometry (vs Fortran B(θ,ζ) dumps; vs booz_xform), operators (matrix-free ≡ matrix;
  adjoint identity), solver policy (auto-selection table).
- **Parity:** HDF5/netCDF field-by-field vs release golden data (cached; skip-with-warning
  offline); stdout golden logs; the Fortran `shouldBe` 0.3% example assertions replicated.
- **Gradients:** `jax.grad` of ⟨j‖B⟩ and fluxes w.r.t. (dT/dψ, dn/dψ, Er, selected bmnc)
  vs central FD (rtol 1e-5); one test reproducing an adjoint-DKE sensitivity trend
  (Paul et al. 2019); cross-check against SFINCS RHSMode-4/5 adjoint output on one case;
  ambipolar-root implicit gradient vs FD.
- **Coverage:** pytest-cov, 4-way sharding (keep the existing pytest-split setup), combined
  gate ≥95% enforced via `codecov.yml` project target; sparing `# pragma: no cover` only on
  print formatting.
- **Workflows:** `ci.yml` (ubuntu + macos, latest CPython, pip cache, `-n auto`, examples
  smoke job at toy resolution via `SFINCS_JAX_CI=1`, optional-deps job, all ≤ 10 min);
  `docs.yml` (sphinx -W; non-gating link check); `publish.yml` on tag `v*`
  (`environment: pypi`, `permissions: id-token: write`,
  `pypa/gh-action-pypi-publish@release/v1`, not inside a reusable workflow).
- **Size guard:** CI script failing any PR adding a file > 200 kB or pushing repo size past
  10 MB.

## 7. Examples (each one file, ≤ ~120 lines, simsopt-style: params → objective → optimize →
plot; API-only, no hidden helper modules; header comment states expected laptop runtime and
the key citation; all run in CI at toy resolution)

1. `01_single_species_tokamak.py` — run + fluxes vs analytic banana-regime check.
2. `02_w7x_fluxes_and_flows.py` — multi-species FP, Er scan, Γ(Er) plot with roots marked.
3. `03_ambipolar_er_roots.py` — er.py root finding; ion vs electron root classification.
4. `04_bootstrap_profile_precise_QA_QH.py` — the previously-failing case: ⟨j‖B⟩ profile
   across s for precise-QA and precise-QH at production resolution vs Redl and vs the
   Fortran reference curve; doubles as the G3 regression proof.
5. `05_monoenergetic_database.py` — D11/D31/D33 vs collisionality for W7-X/HSX in the
   ICNTS-benchmark figure layout (Beidler NF 2011).
6. `06_optimize_min_bootstrap_QA.py` — gradient-based: minimize ⟨j‖B⟩² over boundary modes
   at fixed resolution (structure mirrors simsopt's QH_fixed_resolution.py: problem →
   jax.value_and_grad objective → scipy.optimize.minimize → report).
7. `07_optimize_electron_root.py` — objective on er.py root properties (target Er>0 root
   existence/depth), literature-guided.
8. `08_optimize_impurity_screening.py` — maximize the temperature-screening metric (impurity
   flux sign) for a trace C⁶⁺ impurity (Mollén 2015; Martin & Landreman 2020).
9. `09_gradients_and_sensitivities.py` — one-page tour of jax.grad/jacfwd through the whole
   pipeline, printing d⟨j‖B⟩/d(dT/dψ), dΓ/dEr, dD31/d(bmnc).

## 8. Documentation plan (docs/ + Read the Docs)

Template: sphinx + **furo**, myst-parser, MathJax, sphinx-copybutton, sphinxcontrib-bibtex
with `references.bib` (every §1 entry, links verified).

```
docs/
  index.md                 # what it is, 20-line quickstart, badges
  installation.md          # pip; optional native solvers (conda petsc4py note); GPU notes
  tutorials/               # executed notebooks-as-md (jupytext) mirroring examples 1–5
  physics/
    drift_kinetic_equation.md   # derivation & normalizations (from the SFINCS technical
                                # notes + Helander & Sigmar), trajectory models, Er terms
    collisions.md               # PAS + full FP, Rosenbluth potentials, speed grid
    moments_and_transport.md    # fluxes, flows, bootstrap, Onsager, monoenergetic D_ij,
                                # classical transport
    ambipolarity_and_phi1.md    # roots physics, screening, quasineutrality, references
  numerics/
    discretization.md           # FD/Fourier/Legendre/speed-collocation, quadratures,
                                # Nxi_for_x ramps
    solvers_and_preconditioners.md  # §2.3 in full: block-tridiagonal derivation +
                                    # truncated-storage proof, coarse-operator PC ≡ the
                                    # Fortran preconditioner_* knobs, multigrid/smoothers,
                                    # recycling, convergence plots, when-to-use-what table
    differentiability.md        # IFT adjoint, what is/isn't differentiable, costs
    performance.md              # profiling results (incl. Fortran -log_view comparison),
                                # GPU guidance, memory model, benchmark table
  reference/
    api.md (autodoc); namelist.md (EVERY input, default, and Fortran name — sourced from
    readInput.F90 + validateInput.F90); outputs.md (every output variable vs
    writeHDF5Output.F90 — the gap table must reach zero); sfincs_name_map.md
  citing.md, changelog.md
```

Every physics/numerics page: LaTeX equations, ≥3 clickable citations, and a "where in the
code" box linking to source. All images compressed; total docs assets < 2 MB (today: 14 MB
`_static` + 5.7 MB third-party PDFs in `docs/upstream` — delete the PDFs, link to
publishers/arXiv instead).

## 9. Repo hygiene, history rewrite, contributor verification (Phase 7 — LAST; coordinate
the force-push; keep an untouched backup clone until the release is verified)

- 9.1 Working-tree deletions: all `plan*.md` (including this file — its job is done),
  `PROGRESS.md`, `docs/upstream/*.pdf|eps|jpg`, uncompressed figures (regenerate < 150 kB),
  `tests/ref/` (→ release asset), stray scripts.
- 9.2 `git filter-repo` on a fresh `--mirror` clone:
  - `--strip-blobs-bigger-than 200K` plus explicit `--path v3_driver.py --invert-paths`
    (the 786 × 2.5 MB blobs are the whole 62 MB story) and the deleted plan/experiment
    files;
  - `--mailmap` unifying `Rogerio Jorge <rogerio.jorge@ist.utl.pt>` →
    `Rogerio Jorge <rogerio.jorge@wisc.edu>` (no AI identities exist — verified across all
    refs 2026-07-08 — but grep the enumeration again on the mirror and extend the mailmap
    if anything new appears);
  - `--message-callback` stripping any `Co-Authored-By:`/`Co-authored-by:` and
    `Generated with` lines (currently zero — cheap insurance, both capitalizations).
- 9.3 Re-add origin, `git push --mirror --force`; re-protect branches; verify
  `git count-objects -vH` < 10 MB and the GitHub UI; verify Actions green; confirm the
  GitHub contributors API lists only rogeriojorge (the page can take time to recompute);
  re-tag if tags were rewritten. The four `research/*` branches survive the rewrite (they
  are the designated archive for the deleted experimental code).
- 9.4 Policy forward: files > 200 kB → release assets; plots → compressed PNG/SVG < 150 kB;
  the CI size-guard from §6 enforces it.

## 10. Risks & mitigations

- **PETSc/MUMPS build pain on macOS** → conda-forge binaries only (Phase 0); Fortran is
  never a CI dependency (golden data decouples).
- **Full-FP + full-trajectory operator is not exactly block-tridiagonal in l** → tier-1 is
  then a *preconditioner*, not a solver; acceptance is iteration counts (≤ 30 GMRES
  iterations across the benchmark matrix), not exactness.
- **Block elimination without global pivoting can be unstable as ν→0**
  (Demmel–Higham–Schreiber) → pivoted LU within blocks, condition monitoring, iterative
  refinement, tier-2/3 fallback on detection.
- **fp32 PC accuracy on GPU** → fp64 residuals + refinement; per-dtype parity tolerances.
- **jit compile times at many resolutions** → static-shape policy, padding, persistent
  compilation cache (documented); one compile per resolution.
- **optimistix/vmap/persistent-cache miscompile (jax#31733)** → canary regression test;
  document the workaround.
- **History rewrite** → mirror clone + untouched backup; do last; verify before deleting
  the backup.
- **Scope creep from the reference-solver survey** → ideas enter only through §2.3's tiers
  with a benchmark gate: default-on only if neutral-or-better runtime/RSS *and*
  parity-clean, else `research/*` branch.
- **AMR / adaptive Legendre truncation / analytic high-l tails** → post-v2 roadmap issues,
  but keep every resolution a per-axis runtime parameter so p-refinement stays cheap to add.

## 11. Kickoff checklist for Cowork (chronological)

- [ ] Read this file fully; skim `plan_final.md`/`plan.md` for context only; this file wins.
- [ ] Configure git identity (rogeriojorge) in every clone; confirm push access.
- [ ] Phase 0: envs; Fortran+PETSc/MUMPS/SuperLU_DIST build; `-log_view` profiles; golden
      data release (`reference-data-v2`), including migrating `tests/ref/`.
- [ ] Phase 1: re-verify audit numbers; reproduce QH high-res, Φ₁/QN, and memory failures;
      file-by-file migration verdicts; reference-solver idea list (names never committed).
- [ ] Phase 2: create uwplasma/SOLVAX (PyPI name re-check first), implement §3, publish
      v0.1.0; open consumer PRs (NTX, spectrax-gk).
- [ ] Phase 3: refactor per §4 with per-term Fortran parity; close output/print gaps;
      purge commit; coverage gate → 95%.
- [ ] Phase 4: solver tiers + continuation; QA and QH production-resolution profiles
      converge CPU+GPU; ≤1.5× runtime and ≤2× RSS vs Fortran everywhere; new README
      benchmark table + compressed plots.
- [ ] Phase 5: nine examples, CI at toy resolution.
- [ ] Phase 6: docs rebuilt (furo, equations, bibtex); RTD green; namelist/outputs
      reference complete (gap table = zero).
- [ ] Phase 7: hygiene + filter-repo (strip v3_driver.py history, >200 kB blobs, mailmap
      email unification) + force-push; repo < 10 MB; contributors = rogeriojorge only.
- [ ] Phase 8: tag and publish sfincs_jax v2.0.0 + solvax; release notes; delete this file.

*End of plan.*
