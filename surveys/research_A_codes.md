# Research survey A: neoclassical / drift-kinetic codes and literature relevant to DKX

Date: 2026-09-06. Written incrementally; sections are appended as findings arrive.
Verification policy: every citation carries a URL that was fetched or seen verbatim in a search result.
Anything not verified against a fetched source is marked UNVERIFIED.

## Status log
- [start] skeleton created
- [final] sections (i)-(iv) written; file complete
- [batch3] fetched sfincs issue #1 (namelist), MONKES full text, ICNTS IOP page; searches: NN surrogates, Infinity Two, Stellaris/Helios, W7-X impurities. WEB FETCH CAP (30) REACHED - remaining sections written from fetched material + flagged knowledge.
- [batch2] fetched yancc full text (arxiv html), MONKES thesis abs, NTX README, KNOSOS abs; searches for SFINCS namelist, DKES, PENTA, NEO-2, NEOTRANSP, ICNTS, Redl, QI optimization, DESC bootstrap, Mollen Phi1, Paul adjoint
- [batch1] fetched arXiv 1312.6058, 2607.20861, 2312.12248, 2407.21599, 2507.05166, 2205.02914; yancc README; sfincs input.namelist (symlink only, no content)

## 1. SFINCS v3

### 1a. Landreman, Smith, Mollen, Helander 2014, Phys. Plasmas 21, 042503, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas" (arXiv 1312.6058)
Source fetched: https://arxiv.org/abs/1312.6058 ; DOI shown: https://doi.org/10.1063/1.4870077
- This is the SFINCS v3 reference paper (the abstract itself describes "a new drift-kinetic code"; the SFINCS name is in the paper body, not the abstract).
- Content: compares several formulations of the radially local DKE (different effective trajectories: with/without E x B in the parallel/poloidal drift, full trajectories, "DKES-like" trajectories) and several collision operators (full linearized Fokker-Planck vs pitch-angle scattering vs momentum-conserving models) on LHD and W7-X.
- Key physics findings quoted from the abstract summary: below roughly one-third of the resonant Er the formulations agree closely; near the Er resonance they diverge substantially; momentum-conserving collisions matter in certain regimes, but for low-collisionality gradient-driven radial transport different collision operators give similar results.
- Relevance to DKX: DKX reimplements exactly this model (radially local, linearized, multi-species, full linearized Fokker-Planck operator, Er via E x B). The trajectory/collision-operator switches in this paper are the natural verification matrix for DKX (each SFINCS option = one DKX regression case).

### 1b. Phi1 (flux-surface variation of the electrostatic potential) in SFINCS
Sources seen in search results (URLs verbatim):
- Mollen, Landreman, Smith, Garcia-Regana, Nunami, "Flux-surface variations of the electrostatic potential in stellarators: impact on the radial electric field and neoclassical impurity transport", Plasma Phys. Control. Fusion 60(8) 084001 (2018) - https://www.osti.gov/biblio/1499870 . SFINCS ("Stellarator Fokker-Planck Iterative Neoclassical Conservative Solver") extended to solve the DKE together with quasineutrality for Phi1; Phi1 can be large enough in 3D devices to change the impurity particle flux.
- Mollen, Landreman, Smith, Braun, Helander, "Impurities in a non-axisymmetric plasma: transport and effect on bootstrap current", arXiv 1504.04810 (2015) - https://arxiv.org/abs/1504.04810 (Phys. Plasmas 22, 112508 - venue/volume UNVERIFIED). Multi-species SFINCS with impurities; effect of impurities on the bootstrap current.
- Garcia-Regana, Beidler, Kleiber, Helander, Mollen, Alonso, Landreman, Maassberg, Smith, Turkin et al. (2017), "Electrostatic potential variation on the flux surface and its impact on impurity transport" - https://www.researchgate.net/publication/271079728_Electrostatic_potential_variation_on_the_flux_surface_and_its_impact_on_impurity_transport (Nucl. Fusion 57 056004 - UNVERIFIED). EUTERPE/SFINCS comparison of Phi1.
- Buller et al., "The importance of the classical channel in the impurity transport of optimized stellarators", J. Plasma Phys. - https://arxiv.org/html/1903.12511 (uses SFINCS; classical flux comparable to neoclassical for impurities in optimized stellarators).
- Relevance to DKX: Phi1 is the one SFINCS v3 capability that neither yancc nor MONKES/NTX has. A JAX Phi1 solve (nonlinear in Phi1 via the Boltzmann factor exp(-Z e Phi1/T) with quasineutrality; SFINCS iterates with Newton/SNES) would be a differentiator, and is important for impurity transport and for high-Z screening questions on W7-X.

### 1c. Adjoint neoclassical (Paul et al.)
- Paul, Abel, Landreman, Dorland, "An adjoint method for neoclassical stellarator optimization", arXiv 1904.06430 (v1 12 Apr 2019, rev 2 Jun 2019) - https://arxiv.org/abs/1904.06430 . Derivatives of moments of the neoclassical distribution function (fluxes, bootstrap current) with respect to many geometric parameters from ONE adjoint solve instead of O(N) forward solves; implemented in SFINCS (the paper body describes the discrete adjoint of the SFINCS linear system; the abstract summary in the search result confirms the "single adjoint system" claim). Published J. Plasma Phys. 85, 795850501 (2019) - volume UNVERIFIED.
- Follow-ups seen in search results: Paul, Landreman, Antonsen, "Adjoint methods for stellarator shape optimization and sensitivity analysis", arXiv 2005.07633 - https://arxiv.org/pdf/2005.07633 ; Paul et al., "Adjoint approach to calculating shape gradients for three-dimensional magnetic confinement equilibria" - https://www.osti.gov/pages/biblio/1597704 ; "Adjoint methods for quasisymmetry of vacuum fields on a surface", arXiv 2108.11433 - https://arxiv.org/pdf/2108.11433 .
- Relevance to DKX: the Paul adjoint is a hand-derived discrete adjoint for a specific set of outputs and inputs (Boozer-spectrum B_mn, Er, etc.). jax.grad through DKX's linear solve (implicit differentiation of A(p) f = b(p): dL/dp = -lambda^T (dA/dp f - db/dp) with A^T lambda = dL/df) gives the same one-extra-solve cost for ANY scalar output and ANY input, including geometry, profiles, collisionality, and Er. This is the correct baseline to cite when explaining why DKX's differentiability is more general than the 2019 adjoint. Verification target: reproduce a Paul et al. sensitivity figure (derivative of a flux with respect to a B_mn) with jax.grad and finite differences.

### 1d. preconditionerOptions namelist (SFINCS v3)
Source fetched: https://github.com/landreman/sfincs/issues/1 (quotes a v3 input.namelist; `preconditioner_x_min_L` is v3 nomenclature). Verbatim options and comments:
```
&preconditionerOptions
! Settings for how to simplify the linear system to obtain the preconditioner:
preconditioner_species = 1   ! 0 = keep full species coupling
                             ! 1 = drop all cross-species coupling
preconditioner_x = 1         ! 0 = keep full x coupling
                             ! 1 = drop everything off-diagonal in x
                             ! 2 = keep only upper-triangular part in x
                             ! 3 = keep only the tridiagonal terms in x
                             ! 4 = keep only the diagonal and superdiagonal in x
preconditioner_x_min_L = 1   ! The x structure of the matrix will only be simplified for L >= this value.
                             ! Set preconditioner_x_min_L=0 to simplify the matrix for every L.
preconditioner_theta = 0     ! 0 = keep full theta coupling
                             ! 1 = use a 3-point finite difference stencil for d/dtheta
preconditioner_zeta = 0      ! 0 = keep full zeta coupling
                             ! 1 = use a 3-point finite difference stencil for d/dzeta
preconditioner_xi = 0        ! 0 = keep full xi coupling
                             ! 1 = drop terms that are +/- 2 from the diagonal in xi, so preconditioner is tridiagonal in xi
/
```
Options named in the task but ABSENT from the fetched (older) namelist - described from working knowledge of current SFINCS v3, UNVERIFIED by fetch:
- `preconditioner_magnetic_drifts_max_L`: when magnetic drifts (`includeXDotTerm`, `magneticDriftScheme`) are on, the drift terms couple neighbouring x and xi modes; this option keeps them in the preconditioner matrix only for Legendre modes L <= max_L and drops them for higher L, so the preconditioner stays close to the streaming+collision structure.
- `reusePreconditioner`: reuse the LU factorization of the preconditioner matrix across successive linear solves - Newton iterations of the nonlinear Phi1 solve (SNES) and the multiple right-hand sides of transport-matrix runs (RHSMode 2/3) - so the dominant factorization cost is paid once.
- `Nxi_for_x_option` (with `Nxi_for_x_pattern`): 0 = same number of Legendre modes Nxi at every speed grid point; 1 = use fewer Legendre modes at small x (low-speed particles are highly collisional and need little pitch-angle resolution), following a pattern that ramps Nxi with x. This removes the wasted high-L modes at low x and typically cuts matrix size substantially.

The strategy, stated plainly (this is the documented SFINCS design; the mechanism is verified by the namelist text above):
1. SFINCS assembles two sparse matrices with PETSc: the FULL linearized DKE operator (used only for matrix-vector products in the Krylov iteration, KSP default GMRES) and a SIMPLIFIED operator obtained by deleting couplings as selected above.
2. The simplified operator is factored exactly (MUMPS or SuperLU_dist, chosen by `whichParallelSolverToFactorPreconditioner` - UNVERIFIED name) and applied as the preconditioner: each Krylov iteration costs one full mat-vec plus one triangular solve with the factors.
3. Why it works: the expensive couplings are the ones that make the matrix dense - the field-particle and energy-scattering parts of the Fokker-Planck operator couple ALL x points (dense Nx x Nx blocks for each L, species pair) and couple species; deleting them (`preconditioner_x=1`, `preconditioner_species=1`) leaves per-(species, x) blocks, each of which is a MONOENERGETIC DKE (streaming, mirror, E x B, pitch-angle scattering plus the diagonal part of energy scattering). These blocks are small enough for sparse LU, and the neglected couplings are weak enough that GMRES converges in tens of iterations at typical collisionality. `preconditioner_x_min_L=1` keeps full x coupling in the L=0 block, because at L=0 the field-particle term (momentum/energy conservation) is dominant and dropping it stalls convergence.
4. Trade-off documented in yancc (arXiv 2607.20861 Sec. 5.2): the LU factors of the simplified operator are what push SFINCS to >50 GB at NCSX resolutions on 128 cores, which is the motivation for yancc's matrix-free multigrid.

Relevance to DKX: the SFINCS block structure maps directly onto a JAX design. With `preconditioner_x=1, preconditioner_species=1` the preconditioner is a set of independent monoenergetic problems in (theta, zeta, xi) - exactly what NTX/MONKES solve exactly by block-tridiagonal elimination in Legendre index at O(N_L N_fs^3). So DKX can use an NTX-style exact per-(species, x) block solve as the preconditioner (vmap over species and x on GPU), with the full operator applied matrix-free. This is a third option between SFINCS (sparse LU) and yancc (multigrid), and it is native to the uwplasma stack.

## 2. yancc

### 2a. Conlin & Landreman, "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators" (arXiv 2607.20861, submitted 23 Jul 2026)
Source fetched: https://arxiv.org/abs/2607.20861 and https://github.com/f0uriest/yancc
- Name: "Yet Another NeoClassical Code". Pure-Python, built on JAX, Python >= 3.10, on PyPI (`pip install yancc`). Repository license file: MIT (README). arXiv abstract page license: CC BY 4.0 (that is the paper license, not the code license).
- Discretization (abstract, verbatim): "Maxwell polynomial collocation grid in speed with finite differences in pitch angle and the flux surface coordinates, using a modified upwind stencil designed to improve diagonal dominance".
- Solver (abstract, verbatim): "multigrid-preconditioned Krylov method". The Krylov variant is NOT named in the abstract (GCROT claim: pending full-text check).
- Solves both the full 4D DKE (speed, pitch, theta, zeta) and the reduced monoenergetic form.
- Verification (abstract, verbatim): "Benchmarks against MONKES and SFINCS show agreement within 1% across a range of collisionalities, geometries, and multi-species configurations".
- Performance (abstract, verbatim): "roughly an order of magnitude speedup over SFINCS on a per-scan basis while using an order of magnitude less memory".
- Differentiability (abstract, verbatim): "yancc is fully differentiable, enabling gradient-based optimization and adjoint sensitivity analysis". Whether any derivative is numerically verified: pending full-text check.
- README features: single- and multi-species runs, monoenergetic solver, radial electric field input (Erho), outputs particle flux, heat flux, etc.

### 2b. Full-text details (https://arxiv.org/html/2607.20861, fetched)
- Krylov: "right preconditioned GCROT with inner GMRES"; inner GMRES limited to 150 iterations before restart; GCROT keeps important Krylov vectors between restarts (Sec. 4.5). So the GCROT claim is verified.
- Multigrid: semi-coarsening by 2 in (alpha, theta, zeta) only; the speed coordinate x is kept at full resolution on every level (Sec. 4.1). Smoother: block-diagonal approximation with 4 sequential smoothers, each keeping full coupling along one coordinate; damping omega_s = 0.6 for the full DKE (Sec. 4.2). V-cycle for the full DKE, W-cycle (cycle index 3) for the monoenergetic problem (Sec. 4.4). Coarsest grid (~thousands of DOF) solved by dense LU. Prolongation: piecewise-linear interpolation; restriction: volume-weighted transpose.
- Largest reported problems: single-species NCSX scan at (n_x, n_alpha, n_theta, n_zeta) = (7, 121, 43, 65) needing 6 GB of GPU memory on one NVIDIA A100; two-species scan at (7, 61, 25, 37).
- vs SFINCS (Sec. 5.2, Fig. 7): SFINCS at (n_x, n_xi, n_theta, n_zeta) = (7, 141, 25, 81) on 128 CPU cores needed >50 GB; yancc 6 GB on one A100; ~5x faster at moderate collisionality and "nearly 2 orders of magnitude" faster at high collisionality; yancc runtime "nearly flat across the full range of collisionality".
- vs MONKES (Sec. 5.1, Fig. 5), W7-X monoenergetic: both converged to within 1%; MONKES (n_L, n_theta, n_zeta) = (180, 39, 99) using 1.4 GB vs yancc (201, 31, 81) using 4 GB; yancc 2x-4x faster over most of the collisionality range. NOTE: on the monoenergetic problem MONKES uses LESS memory than yancc.
- Differentiability: only the general statement "fully differentiable" plus "adjoint-based sensitivity analysis, where the derivative of a quantity of interest with respect to all input parameters can be computed in a single solve" (Sec. 6). The fetched text contains NO finite-difference verification of any gradient, no custom VJP / implicit-differentiation description. So: differentiability is claimed, not demonstrated, in the paper as fetched.
- Explicit future work (Sec. 6): self-consistent ambipolar Er; adaptive mesh refinement in pitch angle and real space; corrections for strong Er and flows near quasisymmetry. Phi1 is not mentioned at all. Therefore yancc as published does NOT do Phi1 and does NOT do ambipolar Er root-finding.
- Collision operator: the extraction did not return a statement; UNVERIFIED whether the full linearized Fokker-Planck (field-particle) operator is included for multi-species. (The abstract says "multi-species configurations" agree within 1% with SFINCS, which suggests at least a momentum-conserving operator.)
- License: repo file MIT; paper CC BY 4.0.


## 3. MONKES and NTX

### 3a. Escoto, Velasco, Calvo, Landreman, Parra, "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", Nucl. Fusion (DOI 10.1088/1741-4326/ad3fc9), arXiv 2312.12248 (v1 19 Dec 2023, v3 11 Nov 2024)
Source fetched: https://arxiv.org/abs/2312.12248
- Solves the monoenergetic DKE (DKES-type: pitch-angle scattering + monoenergetic, three drives -> D11, D31, D33) for stellarators.
- Method (abstract, verbatim): "spectral discretization in spatial and velocity coordinates with block sparsity" (Fourier in theta/zeta, Legendre in pitch angle; the Legendre coupling is tridiagonal, giving a block-tridiagonal system solved by forward elimination/back substitution - detail from the paper body, to be confirmed with thesis fetch).
- Cost: low-collisionality monoenergetic coefficients on a single core in roughly one minute.
- Stated applications: direct use inside stellarator optimization (bootstrap current) and in predictive transport suites.
- Code and data: available on GitHub (URL in paper; pending).

### 3b. Escoto Lopez PhD thesis, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas", arXiv 2510.27513 (31 Oct 2025)
Source fetched: https://arxiv.org/abs/2510.27513
- Confirms the algorithm: pitch angle in Legendre polynomials; the Lorentz operator and the mirror/streaming coupling produce a tridiagonal structure in Legendre index, solved with a block-tridiagonal algorithm (each block is the Fourier (theta, zeta) matrix for one Legendre mode). Emphasis on "direct optimization of the bootstrap current (and radial neoclassical transport)".
- Cost scaling (from the MONKES full text, https://arxiv.org/html/2312.12248, fetched): arithmetic O(N_xi * N_fs^3) with N_fs = N_theta * N_zeta; memory O(N_fs^2), independent of N_xi, because only the k = 0, 1, 2 Legendre-mode matrices are needed for the transport coefficients and the forward elimination (their Eqs. 61-62: recursion for Delta_k matrices and sigma^(k) vectors) followed by back substitution (Eq. 52) for k = 0, 1, 2 only.
- Convergence at low collisionality (nu-hat = 1e-5 m^-1, from full text): W7-X EIM N_xi ~ 140-160 with (N_theta, N_zeta) ~ (23-27, 55); W7-X KJM N_xi ~ 140-180 with (19-23, 63-79); CIEMAT-QI N_xi ~ 180 with (15, 119). Finite Er makes convergence harder.
- Timing (full text): Intel Xeon Gold 6254, single core; W7-X low-nu case "less than a minute and a half" vs DKES "almost an hour and a half"; speed-up over DKES 4-64x depending on configuration.
- Benchmarks: DKES and SFINCS (agreement percentages not extracted).
- Adjoint/derivatives: none in MONKES (NTX adds this, Sec. 3c).
- Repository: https://github.com/JavierEscoto/MONKES/ (license of the code not extracted - UNVERIFIED; the arXiv text carries the arXiv non-exclusive license).
- Related application paper seen in search results: "Evaluation of neoclassical transport in nearly quasi-isodynamic stellarator magnetic fields using MONKES", arXiv 2410.17836 - https://arxiv.org/pdf/2410.17836 .

### 3c. NTX (github.com/uwplasma/NTX)
Source fetched: https://github.com/uwplasma/NTX
- JAX-native solver of the local monoenergetic DKE on one flux surface at fixed speed: parallel streaming, mirror force, Er precession, Lorentz pitch-angle scattering. Outputs D11, D31, D13, D33, D33_spitzer. Explicitly "monoenergetic scope rather than full-collision closure".
- Method: finite Legendre expansion in pitch angle giving a block-tridiagonal system (MONKES algorithm), with an adjoint carried through the block-tridiagonal solve.
- Reported numbers (README): with 32 design parameters, gradient in ~89 ms vs ~1269 ms by finite differences (~14x), agreement "exact to rounding" (~2e-14). GPU, CPU and multiprocess via batched scans.
- Verification (README): analytical limits, convergence ladders, fixed-field comparisons, geometry-family convergence, derivative checks; bootstrap-current comparisons with SFINCS and other codes.
- License MIT; `pip install ntx`.
- Relevance to DKX: NTX is the in-house differentiable monoenergetic code; DKX is the full-speed, full-collision-operator, multi-species counterpart. The natural split: NTX for D_ij(nu/v, Er/v) tables and monoenergetic-based J_bs; DKX for anything requiring the energy-dependent linearized operator (impurities, Phi1, momentum-conserving corrections, transport matrices at finite collisionality, Er ambipolarity with multiple species). Derivative-verification methodology (adjoint vs FD to rounding) should be copied into DKX's test suite.


## 4. KNOSOS, DKES, PENTA, NEO-2, NEOTRANSP

### KNOSOS
Velasco, Calvo, Parra, Garcia-Regana, "KNOSOS: a fast orbit-averaging neoclassical code for stellarator geometry", J. Comput. Phys. (2020), DOI https://doi.org/10.1016/j.jcp.2020.109512 , arXiv 1908.11615 (v1 30 Aug 2019, rev 22 Jun 2020) - fetched https://arxiv.org/abs/1908.11615 .
Regime: low collisionality only (1/nu, sqrt(nu), superbanana-plateau) - it solves the ORBIT-AVERAGED (bounce-averaged) DKE, not the full local DKE, so it is orders of magnitude faster than DKES/SFINCS but not valid at plateau/Pfirsch-Schlueter collisionality. Includes tangential magnetic drift and Phi1 (quasineutrality linear in Phi1). Verified against local codes on helias, heliotron, heliac, TJ-II, W7-X, LHD, NCSX geometries. Open source on GitHub. Current use: TJ-II/W7-X transport analysis, QI optimization at CIEMAT (e.g. CIEMAT-QI4X, arXiv 2512.08825 - https://arxiv.org/pdf/2512.08825 , seen in search), fast Er and Phi1 predictions; a follow-up "Fast simulations for large aspect ratio stellarators with the neoclassical code KNOSOS" arXiv 2106.01727 - https://arxiv.org/pdf/2106.01727 (seen in search).
Relevance to DKX: KNOSOS is the fast-but-asymptotic competitor; DKX's value is validity across all collisionalities plus the full operator. A DKX-vs-KNOSOS comparison at low nu with Er (where KNOSOS is valid) would confirm both.

### DKES
Hirshman, Shaing, van Rij, Beasley, Crume, "Plasma transport coefficients for nonsymmetric toroidal confinement systems", Phys. Fluids 29(9) 2951-2959 (Sept 1986) - https://pubs.aip.org/aip/pfl/article-abstract/29/9/2951/944354/ (seen in search; OSTI copy https://www.osti.gov/servlets/purl/6092128 ). van Rij & Hirshman, "Variational bounds for transport coefficients in three-dimensional toroidal plasmas", Phys. Fluids B 1(3) 563-569 (Mar 1989) - https://pubs.aip.org/aip/pfb/article-abstract/1/3/563/940728/ .
Model: monoenergetic DKE with pitch-angle-scattering (Lorentz) collision operator and an incompressible E x B drift approximation (the "DKES trajectories" of Landreman 2014); variational principle giving upper and lower bounds on D11, D31, D33 that converge monotonically with Fourier-Legendre resolution. Current use: still the workhorse producing mono-energetic coefficient databases for W7-X (via NEOTRANSP), LHD, TJ-II, and for PENTA; convergence at low nu/v and large Er/v is slow (large Legendre count, bound gap), which is what MONKES/NTX/yancc address. Still used in 2024 (e.g. JPP 2024 "Modelling of relativistic electron transport with non-relativistic DKES solver" - https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/modelling-of-relativistic-electron-transport-with-nonrelativistic-dkes-solver/AD83F02C17589A40A279B7FEB4514FF2 ).

### PENTA
Spong, "Generation and damping of neoclassical plasma flows in stellarators", Phys. Plasmas 12(5) 056114 (May 2005) - https://pubs.aip.org/aip/pop/article/12/5/056114/1015589/ (seen in search).
Model: moments-method (Sugama-Nishimura style) momentum-conserving correction built on top of DKES mono-energetic coefficients; yields parallel flows, bootstrap current and ambipolar Er roots for quasi-helical, quasi-toroidal and quasi-poloidal symmetric devices. Regime: any collisionality covered by the DKES database, but only pitch-angle-scattering physics is kinetic; momentum conservation is restored approximately. Current use: HSX and other US analysis; also for RFX-mod 3D equilibria (https://pubs.aip.org/aip/pop/article-abstract/18/6/062505/387754/). Relevance: PENTA is the standard "monoenergetic + momentum correction" route to J_bs; DKX with the full operator is the check on PENTA's approximation (a classic result to reproduce: PENTA vs SFINCS J_bs in HSX/W7-X).

### NEO-2
Kernbichler, Kasilov, Leitold, Nemov, Allmaier, "Recent progress in NEO-2 - a code for neoclassical transport computations based on field line tracing", Plasma Fusion Res. 3, S1061 (2008) - https://www.jstage.jst.go.jp/article/pfr/3/0/3_0_S1061/_article/-char/en ; Kernbichler, Kasilov, Kapper, Martitsch, Nemov, Albert, Heyn, "Solution of drift kinetic equation in stellarators and tokamaks with broken symmetry using the code NEO-2", Plasma Phys. Control. Fusion 58, 104001 (2016) - https://iopscience.iop.org/article/10.1088/0741-3335/58/10/104001 (both seen in search).
Model: linearized DKE solved along a field line (field-line integration) with multiple-domain decomposition and adaptive velocity-space grid; full linearized collision operator available; designed for the long-mean-free-path regime; also computes the generalized Spitzer function (ECCD efficiency). Current use: W7-X and tokamak-with-3D-perturbation analysis at TU Graz/IPP; the reference code for the Shaing-Callen bootstrap convergence study (arXiv 2407.21599). Relevance: NEO-2 is the high-accuracy low-nu reference for J_bs; a DKX vs NEO-2 J_bs(nu_*) comparison is a strong verification.

### NEOTRANSP
Source: search results only (W7-X literature). NEOTRANSP (Turkin et al., IPP) integrates DKES mono-energetic coefficient databases over energy to give neoclassical particle/heat fluxes and the ambipolar Er for W7-X; used in the Nature 2021 W7-X paper "Demonstration of reduced neoclassical energy transport in Wendelstein 7-X" - https://www.nature.com/articles/s41586-021-03687-w and in W7-X power-balance studies - https://iopscience.iop.org/article/10.1088/1361-6587/ade824 ; benchmarked against global EUTERPE Er roots (https://www.researchgate.net/figure/Benchmarking-of-the-global-neoclassical-radial-electric-field-calculated-with-EUTERPE_fig1_378516498). Regime: whatever the DKES database covers; momentum correction by the Maassberg/Taguchi approach (UNVERIFIED detail). No standalone NEOTRANSP method paper was found in this search (UNVERIFIED whether one exists). Relevance: NEOTRANSP-style energy-convolution of D_ij tables is the fast path used operationally at W7-X; DKX can produce the same fluxes directly with the full operator and hence serve as the "truth" for that pipeline.


## 5. ICNTS benchmark (Beidler et al. 2011)
Beidler et al., "Benchmarking of the mono-energetic transport coefficients - results from the International Collaboration on Neoclassical Transport in Stellarators (ICNTS)", Nucl. Fusion 51, 076001 (2011) - https://iopscience.iop.org/article/10.1088/0029-5515/51/7/076001 (seen in search; abstract text summarized in search result).
- What was benchmarked: the three mono-energetic coefficients (radial D11, bootstrap D31, parallel conductivity D33) as functions of collisionality nu/v and normalized Er (E_r/v), computed with field-line-integration codes (NEO, NEO-2), Monte Carlo codes, the variational Fourier-Legendre method (DKES) and a finite-difference scheme.
- Authors (from the IOP page, fetched https://iopscience.iop.org/article/10.1088/0029-5515/51/7/076001 ): C.D. Beidler, K. Allmaier, M.Yu. Isaev, S.V. Kasilov, W. Kernbichler, G.O. Leitold, H. Maassberg, D.R. Mikkelsen, S. Murakami, M. Schmidt, D.A. Spong, V. Tribaldos, A. Wakasa.
- Devices (IOP page): W7-AS, W7-X, LHD, TJ-II, NCSX, HSX, QPS, CHS.
- Methods: field-line integration (NEO/NEO-2 family), Monte Carlo, variational Fourier-Legendre (DKES), finite differences. Individual code names beyond these families were not visible on the page (UNVERIFIED).
- Key finding (abstract): D11 and D33 behave as theory predicts; the mono-energetic bootstrap coefficient D31 "exhibits characteristics which have not been predicted" (i.e. the low-nu bootstrap coefficient is the hard case - consistent with the later Shaing-Callen off-set study, arXiv 2407.21599).
- Normalization and data availability: not visible in fetched material (UNVERIFIED). MONKES (arXiv 2312.12248) and yancc (arXiv 2607.20861) both use ICNTS-style D_ij normalization in their W7-X/LHD benchmarks, so their published figures are the practical machine-readable proxy for ICNTS values. Recommendation: reproduce the MONKES/yancc W7-X and LHD D11/D31/D33 curves rather than trying to recover ICNTS raw data.


## 6. Bootstrap current in optimization

### 6a. Landreman, Buller, Drevlak, "Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement", Phys. Plasmas (DOI 10.1063/5.0098166), arXiv 2205.02914 (v1 5 May 2022, rev 26 Jul 2022)
Source fetched: https://arxiv.org/abs/2205.02914
- Uses the QS<->tokamak isomorphism to apply rapid tokamak bootstrap formulae (Redl et al. 2021 in the paper body; the abstract does not name it) to quasisymmetric stellarators.
- Self-consistency enforced in the optimizer by penalizing the mismatch between the equilibrium's parallel current and the formula-predicted bootstrap current, with the current profile included as a free parameter of the equilibrium.
- Result: QS configurations with significant pressure, self-consistent J_bs, and alpha losses lower than many previous designs.
- Relevance to DKX: the fast-formula route only works where QS is good; for QI/QH-with-residual-symmetry-breaking or finite-collisionality/Er effects one needs a kinetic J_bs. A differentiable kinetic J_bs would let the same penalty be used without the isomorphism assumption.

### 6b. Redl, Angioni, Belli, Sauter, "A new set of analytical formulae for the computation of the bootstrap current and the neoclassical conductivity in tokamaks", Phys. Plasmas 28, 022502 (Feb 2021) - https://pubs.aip.org/aip/pop/article/28/2/022502/124727/ (seen in search; MPG open copy https://pure.mpg.de/rest/items/item_3288698/component/file_3288920/content )
- Same analytical structure as Sauter 1999, refit to a large database of NEO drift-kinetic results; three inputs only: trapped fraction f_t, collisionality nu_*, Z_eff. Fixes Sauter inaccuracy at high collisionality (pedestals) and with impurities.
- Sauter, Angioni, Lin-Liu, Phys. Plasmas 6, 2834 (1999) - the original fit; UNVERIFIED (no URL fetched in this survey; cited via the Redl abstract which refers to "the original set published by Sauter").
- Known limits (stated in the DESC tutorial, https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html ): "the Redl formula is only valid in the limit of perfect quasi-symmetry, so this procedure will not work for configurations that are not quasi-symmetric".

### 6c. 2023-2026 optimization papers using Redl or SFINCS (seen in search)
- Goodman et al., "Quasi-isodynamic stellarators with low turbulence as fusion reactor candidates", PRX Energy 3, 023010 (2024), arXiv 2405.19860 - https://arxiv.org/abs/2405.19860 ; https://link.aps.org/doi/10.1103/PRXEnergy.3.023010 . QI designs with small bootstrap current (QI target itself drives J_bs -> 0); neoclassical/bootstrap checks in QI work are typically done a posteriori with SFINCS or KNOSOS (UNVERIFIED which code in this specific paper).
- Goodman et al. 2023 "Constructing precisely quasi-isodynamic magnetic fields", J. Plasma Phys. - https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/constructing-precisely-quasiisodynamic-magnetic-fields/6601E449C8DD3B3FEB361DA2C5732EFC .
- Jorge et al., "A single-field-period quasi-isodynamic stellarator", J. Plasma Phys. - https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/singlefieldperiod-quasiisodynamic-stellarator/9B2A5FDCCD7774E4F91BE45E75FDC6B0 .
- Landreman, Buller, Drevlak 2022 (Sec. 6a) - the Redl+isomorphism method, now in SIMSOPT and DESC.
- CIEMAT-QI4X, Nucl. Fusion (2026?) arXiv 2512.08825 - https://arxiv.org/pdf/2512.08825 : QI with small bootstrap current, island divertor compatible; neoclassical via KNOSOS/MONKES (UNVERIFIED which).
- Stellaris (Proxima Fusion), "A high-field quasi-isodynamic stellarator for a prototypical fusion power plant", Fusion Eng. Des. (2025) - https://www.sciencedirect.com/science/article/pii/S0920379625000705 .
- Helios planar-coil stellarator power plant equilibrium optimization, Fusion Eng. Des. (2026) - https://sciencedirect.com/science/article/pii/S0920379626002905 (appeared in the DESC bootstrap search; presumably uses DESC's Redl-based bootstrap self-consistency - UNVERIFIED).
- Near-axis QI database, arXiv 2601.08400 - https://arxiv.org/pdf/2601.08400 .

### 6d. SIMSOPT / DESC bootstrap-consistent workflows
- DESC tutorial "Bootstrap Current Self-Consistency" (v0.12.1 and v0.15.0 docs): https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html . DESC (JAX, autodiff, GPU) minimizes the difference between the toroidal current from the MHD equilibrium and from the Redl formula at every optimization stage. The tutorial itself states the QS-only validity limit.
- SIMSOPT: the Landreman-Buller-Drevlak 2022 objective (Redl via isomorphism, VMEC current profile as free parameter) - https://arxiv.org/abs/2205.02914 .
- VMEX (uwplasma) references page lists the same bootstrap literature - https://vmex.readthedocs.io/en/latest/project/references.html .

### 6e. Where a differentiable KINETIC bootstrap current would matter
1. Non-QS configurations (QI, QH with finite symmetry-breaking, QA with islands/ripple): Redl is invalid by construction; today J_bs is either ignored in the optimizer, targeted to zero (QI), or checked a posteriori with SFINCS/KNOSOS/MONKES.
2. Finite collisionality and finite Er: Redl has no Er dependence; SFINCS 2014 shows Er resonance matters; Shaing-Callen off-set (arXiv 2407.21599) shows J_bs is not even monotone in nu at Er = 0.
3. Impurities and multi-ion plasmas: J_bs changes with impurity content (Mollen 2015, arXiv 1504.04810); Z_eff alone (Redl) is a crude proxy.
4. Equilibrium-coupled loops: DESC (JAX) or M3D-C1 (Saxena 2025) call J_bs inside an equilibrium solve; a JAX kinetic J_bs with jax.grad closes the loop with exact derivatives, replacing the analytic closure or the offline SFINCS call.
5. Transport-solver coupling: predictive profile evolution needs d(fluxes)/d(gradients) Jacobians - free from jax.jacfwd on DKX, expensive by finite differences on SFINCS.


## 7. 2024-2026 GPU / differentiable / surrogate / design-driven needs

### 7a. GPU or differentiable neoclassical solvers (2024-2026)
- yancc (Conlin & Landreman, arXiv 2607.20861, Jul 2026) - Sec. 2: JAX, GPU (A100, 6 GB for NCSX 4D), multigrid+GCROT, full 4D and monoenergetic, differentiable (claimed, not FD-verified in the paper), no Phi1, no ambipolar Er (listed as future work).
- NTX (uwplasma, https://github.com/uwplasma/NTX) - Sec. 3c: JAX monoenergetic block-tridiagonal with verified adjoint (~14x vs FD at 32 parameters, ~2e-14 agreement), GPU.
- MONKES (Escoto et al., NF 2024) - Sec. 3: CPU Fortran, single core, ~1 min, no derivatives.
- No other GPU or autodiff full-DKE neoclassical solver was found in this survey (searches limited by the 30-fetch cap). Conclusion: as of Sep 2026 yancc is the only published GPU differentiable full-DKE code; DKX would be the second, and the first with Phi1 / ambipolar Er / transport-matrix outputs in JAX.

### 7b. Neural surrogates for neoclassical transport
- Legacy: DCOM/NNW (LHD) - Monte Carlo DCOM database interpolated by a neural network over (r, Er/v, nu/v); described in the IPP abstract "Neoclassical transport simulations for stellarators" - https://pure.mpg.de/rest/items/item_2139735_1/component/file_2139734/content (seen in search). DKES databases at W7-X are interpolated conventionally (NEOTRANSP).
- 2025: "Neural network-based surrogate model for 3D edge-plasma transport in the standard configuration of W7-X", Nucl. Fusion 66 (Nov 2025) - https://iopscience.iop.org/article/10.1088/1741-4326/ae203d - EDGE transport (EMC3-EIRENE), not neoclassical core.
- Tokamak/other surrogates seen: MMMnet (NSTX-U, https://www6.lehigh.edu/~eus204/per/publications/journals/tps24_MMMnetNSTXU.pdf ), 5D gyrokinetic neural surrogates (arXiv 2502.07469), active-learning turbulent-transport surrogates (arXiv 2507.15976).
- Finding: NO 2024-2026 neural surrogate of stellarator CORE neoclassical coefficients or fluxes was found. Gap: a differentiable solver makes surrogate training cheap (gradients as training targets / Sobolev training) and makes the surrogate checkable against the exact adjoint; DKX could ship the dataset + surrogate as a by-product.

### 7c. Neoclassical needs stated by new stellarator designs
- Type One Energy "Infinity Two" (J. Plasma Phys. 91, E65, 2025) - https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/comprehensive-unified-baseline-physics-design-for-the-type-one-energy-stellarator-fusion-pilot-power-plant-infinity-two/CB8A21D770BFA375A9865A28EFBE800B ; companion "Predictions of core plasma performance for the Infinity Two fusion pilot plant" - https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/predictions-of-core-plasma-performance-for-the-infinity-two-fusion-pilot-plant/707D1908C7D42B614D5561987EA7E16D . Stated (search summary): QS design with low bootstrap current (<5 kA); "bootstrap current calculations using the SFINCS code are iterated with VMEC equilibrium solutions"; profiles predicted with the "T3D-GX-SFINCS" framework (self-consistent turbulent + neoclassical transport), Q = 40 at 800 MW. NEED: a neoclassical solver callable inside a transport solver (T3D) and inside an equilibrium iteration, with Jacobians - SFINCS is the bottleneck component there (no derivatives, CPU, PETSc).
- Thea Energy "Helios"/"Eos" (arXiv 2512.08027, Dec 2025; Fusion Eng. Des. 2026) - https://arxiv.org/abs/2512.08027 ; https://thea.energy/wp-content/uploads/2025/12/20251210_FPP_Helios_overview_paper.pdf . QA, two field periods, aspect ratio 4.5, planar coils, X-point divertor; optimization FOMs include quasi-symmetry and "consistency with the pressure-driven bootstrap current"; 4.2 MA bootstrap current at the nominal point. NEED: bootstrap self-consistency at large J_bs where the Redl/isomorphism error is a first-order equilibrium effect; a kinetic differentiable J_bs is the natural upgrade for a DESC-based workflow ("Equilibrium optimization of the Helios planar coil stellarator power plant", FED 2026 - https://sciencedirect.com/science/article/pii/S0920379626002905 ).
- Proxima Fusion "Stellaris" (Fusion Eng. Des. 2025) - https://www.sciencedirect.com/science/article/pii/S0920379625000705 ; press release https://www.proximafusion.com/press-news/proxima-fusion-and-partners-publish-stellaris-fusion-power-plant-concept-to-bring-limitless-safe-clean-energy-to-the-grid . High-field QI; QI implies small J_bs by design, so the need is verification of residual J_bs and of neoclassical impurity/Er behaviour in a non-QS field, where Redl does not apply and KNOSOS/SFINCS are used offline.
- W7-X impurity results (2023-2025): "Quantitative comparison of impurity transport in turbulence reduced and enhanced scenarios at Wendelstein 7-X", Nucl. Fusion (2023) - https://iopscience.iop.org/article/10.1088/1741-4326/aceb76 ; "The suppression of anomalous impurity transport above a critical normalized density gradient scale length in Wendelstein 7-X", Plasma Phys. Control. Fusion (May 2025) - https://iopscience.iop.org/article/10.1088/1361-6587/add597 . Search summary: in NBI-heated / turbulence-suppressed scenarios impurity transport is dominated by neoclassical + classical transport, impurity density peaking scales ~linearly with Z for low-Z impurities, reproduced by 1D transport with neoclassical+classical fluxes. Also the Nature 2021 reduced-neoclassical-transport paper - https://www.nature.com/articles/s41586-021-03687-w . NEED: multi-species (bulk + impurity) neoclassical fluxes with Er, Phi1 and the classical channel (Buller et al., arXiv 1903.12511) fast enough to run inside 1D impurity transport codes - exactly SFINCS's physics content, currently too slow/heavy for routine profile evolution.


## 8. Saxena et al. 2025 (M3D-C1 + SFINCS) and Shaing-Callen convergence (arXiv 2407.21599)

### 8a. Saxena, Ferraro, Martin, Wright, "Bootstrap Current Modeling in M3D-C1" (arXiv 2507.05166, submitted 7 Jul 2025)
Source fetched: https://arxiv.org/abs/2507.05166
- What it is: extension of the extended-MHD code M3D-C1 to model bootstrap current self-consistently.
- Two routes to J_bs: (a) analytic "generalized Sauter" and "revised Sauter-like" closures evaluated inside M3D-C1; (b) a coupled workflow where M3D-C1 equilibria are passed to SFINCS for a kinetic evaluation.
- Configurations: axisymmetric tokamaks and quasisymmetric stellarators, using the QS<->tokamak isomorphism (Landreman et al.).
- Verification: against NEO, XGCa and SFINCS, described as "excellent agreement" (abstract wording; no numbers in abstract).
- Stated purpose: quantify the error of the analytic closures when the configuration departs from symmetry (e.g. after MHD activity). This is precisely the regime where a fast kinetic solver (DKX) would replace the analytic closure inside an MHD loop.
- Journal DOI shown on the abstract page: 10.1017/S0022377825100834 (this is a J. Plasma Phys. DOI prefix; the abstract page listed "Physics of Plasmas" per the fetch summary - venue UNVERIFIED, DOI as displayed).
- Relevance to DKX: shows an existing "MHD equilibrium -> SFINCS -> J_bs" pipeline in 2025 that is not differentiable and is run offline; a JAX kinetic bootstrap current callable from Python would slot in directly.

### 8b. Albert, Beidler, Kapper, Kasilov, Kernbichler, "On the convergence of bootstrap current to the Shaing-Callen limit in stellarators" (arXiv 2407.21599, submitted 31 Jul 2024, revised 18 Feb 2025)
Source fetched: https://arxiv.org/abs/2407.21599
- Decomposes the bootstrap current into the collisionless Shaing-Callen asymptotic value plus an "off-set current".
- Precise claim 1: in the 1/nu regime the off-set current does NOT converge as collisionality decreases; it oscillates over log(nu_*).
- Precise claim 2: convergence to the Shaing-Callen limit does occur in regimes with significant orbit precession, in particular with finite radial electric field, where the off-set current decays as nu_*^(3/5).
- Tools: NEO-2 modelling, analytical estimates, and a semi-analytical "propagator method". SFINCS is NOT used in this paper.
- Relevance to DKX: (1) any bootstrap-current verification at low nu must include Er and must not expect a monotone collisionless limit at Er=0; (2) the nu_*^(3/5) scaling with finite Er is a quantitative, code-independent target that a DKX collisionality scan could reproduce; (3) NEO-2 is the reference code for this physics, so a DKX-vs-NEO-2 comparison of J_bs(nu_*, Er) would be a publishable result.


## (i) Comparison table of codes

| Code | Model | Discretization / method | Solver / preconditioner | Differentiable? | GPU? | Verified against | Reported numbers | License |
|---|---|---|---|---|---|---|---|---|
| SFINCS v3 (Landreman 2014, arXiv 1312.6058) | Radially local linearized DKE, multi-species, full linearized Fokker-Planck-Landau operator, Er (E x B), optional magnetic drifts, Phi1 via quasineutrality (nonlinear), transport matrices, ambipolar Er by scan | Finite differences in theta, zeta; Legendre modes in xi; spectral/polynomial collocation in x; Nxi_for_x reduces L at low x | PETSc KSP (GMRES) on full operator; preconditioner = LU (MUMPS/SuperLU) of simplified operator (drop x / species / xi couplings); SNES for Phi1 | No (hand-derived discrete adjoint by Paul et al. 2019 for specific outputs) | No (MPI CPU) | DKES, NEO/NEO-2, EUTERPE (Phi1), ICNTS-class benchmarks; used as reference by yancc, MONKES, M3D-C1 | NCSX 4D case (7,141,25,81): 128 cores, >50 GB memory (yancc Sec. 5.2) | UNVERIFIED (GitHub landreman/sfincs) |
| yancc (Conlin & Landreman, arXiv 2607.20861, 2026) | Full 4D local DKE (multi-species) and monoenergetic form; Er included; no Phi1; no ambipolar Er (future work) | Maxwell-polynomial collocation in speed; finite differences (modified upwind) in pitch angle and angles | GCROT with inner GMRES(150), right-preconditioned by geometric multigrid (semi-coarsening in alpha, theta, zeta; block smoothers, omega=0.6; V-cycle full DKE, W-cycle monoenergetic; dense LU coarse) | Claimed ("fully differentiable"); no FD verification in paper | Yes, JAX; A100 | SFINCS and MONKES within 1% | NCSX (7,121,43,65) 6 GB on 1 A100; ~5x faster than SFINCS (128 cores) at moderate nu, ~100x at high nu; W7-X monoenergetic 4 GB vs MONKES 1.4 GB, 2-4x faster | MIT (repo); paper CC BY 4.0 |
| MONKES (Escoto et al., NF 2024, arXiv 2312.12248) | Monoenergetic DKE (DKES model: Lorentz operator, incompressible E x B), D11/D31/D33 | Fourier in theta, zeta; Legendre in xi; block-tridiagonal in Legendre index | Direct block-tridiagonal elimination; O(N_xi N_fs^3) flops, O(N_fs^2) memory | No | No (Fortran, single core) | DKES, SFINCS | W7-X low nu: N_xi 140-180, (N_theta,N_zeta) ~ (23,55)-(23,79); ~1 min single core; 4-64x faster than DKES | UNVERIFIED (github JavierEscoto/MONKES) |
| NTX (uwplasma) | Monoenergetic DKE (as MONKES) + adjoint; D11, D31, D13, D33, D33_spitzer | Legendre block-tridiagonal (MONKES algorithm) in JAX | Block-tridiagonal elimination with adjoint | Yes, verified to rounding (~2e-14) vs FD | Yes (JAX; batched scans) | Analytic limits, convergence ladders, SFINCS/other-code J_bs comparisons (README) | 32-parameter gradient 89 ms vs 1269 ms FD (~14x) | MIT |
| KNOSOS (Velasco et al., JCP 2020, arXiv 1908.11615) | Orbit-averaged DKE, low-nu regimes only (1/nu, sqrt(nu), superbanana-plateau); tangential drifts; Phi1 (linear quasineutrality) | Bounce averaging over field lines; fast | Linear in Phi1; small systems | No | No | DKES/local codes on helias, heliotron, heliac, TJ-II, W7-X, LHD, NCSX | "very fast" (orders faster than local codes) | Open source GitHub, license UNVERIFIED |
| DKES (Hirshman 1986; van Rij & Hirshman 1989) | Monoenergetic DKE, Lorentz operator, incompressible E x B | Fourier-Legendre variational, upper/lower bounds | Direct sparse solve | No | No | Reference in ICNTS 2011 | ~1.5 h for W7-X low-nu case in MONKES paper | UNVERIFIED (distributed with STELLOPT) |
| PENTA (Spong, PoP 2005) | Moments method: momentum-conserving correction on DKES D_ij; flows, J_bs, ambipolar Er roots | Energy convolution of D_ij + Sugama-Nishimura style moments | Algebraic | No | No | DKES-based; HSX, QPS, RFX-mod | - | UNVERIFIED |
| NEO-2 (Kernbichler et al., PFR 2008; PPCF 2016) | Linearized DKE along field lines, full linearized collision operator, long-mean-free-path focus; generalized Spitzer function | Field-line integration, multiple-domain decomposition, adaptive velocity grid | Direct/propagator | No | No | ICNTS; reference for Shaing-Callen study (arXiv 2407.21599) | - | UNVERIFIED (GitHub itpplasma) |
| NEOTRANSP (IPP, W7-X) | Energy convolution of DKES D_ij database -> fluxes, ambipolar Er | Table interpolation | Root finding on Er | No | No | EUTERPE global Er roots; W7-X experiments (Nature 2021) | Operational for W7-X | UNVERIFIED (not public) |
| DKX (target) | SFINCS v3 model: radially local linearized DKE, multi-species, full FP operator, Er, Phi1, transport matrices, ambipolar roots | To be chosen (see ii) | Matrix-free Krylov with SFINCS-style block preconditioner or multigrid; implicit-diff adjoint | Yes (jax.grad), must be FD-verified | Yes (JAX) | Must: SFINCS examples, yancc/MONKES W7-X D_ij, NEO-2 J_bs(nu), KNOSOS low-nu, Paul adjoint sensitivities | Targets: <= 6 GB for NCSX (7,121,43,65)-class problem; gradient <= 2x forward cost | uwplasma choice |


## (ii) What DKX should adopt, avoid, and compare against

### Adopt
1. SFINCS's physics model and option matrix as the specification (trajectory models, collision-operator switches, Er, magnetic drifts, Phi1, RHSMode transport matrices). Every SFINCS `examples/` case with reference output is a regression test; the 2014 paper's LHD/W7-X comparisons are the first physics figures to reproduce. Justification: DKX's claim is "SFINCS, differentiable, on GPU"; the value is credibility by agreement, and the reviewers will be SFINCS users.
2. SFINCS's simplified-operator preconditioner idea, implemented in a GPU-native way: with `preconditioner_x=1, preconditioner_species=1` the preconditioner decouples into independent monoenergetic blocks per (species, x). Solve those blocks EXACTLY with the NTX/MONKES block-tridiagonal-in-Legendre algorithm (vmap over species and x). Keep the L=0 (and L=1) x-coupling as SFINCS does (`preconditioner_x_min_L=1`) via a small dense solve. Justification: avoids sparse LU (which JAX/GPU lacks and which drove SFINCS to >50 GB), reuses verified uwplasma code, and is the natural bridge between NTX and DKX. Keep yancc-style multigrid as the fallback option if Krylov iteration counts blow up at very low collisionality with Er.
3. Matrix-free operator application (never assemble the full operator at SFINCS resolutions) and a recycling Krylov method (GCROT / GMRES with deflation) as yancc does, since the same operator is solved for several right-hand sides (transport matrix columns, Er scan, Newton steps for Phi1).
4. Implicit differentiation through the linear solve (jax.lax.custom_linear_solve or lineax-style custom VJP): gradient = one adjoint solve with the transposed operator and the same (transposed) preconditioner. Never differentiate through unrolled Krylov iterations. For the ambipolar Er root and the Phi1 Newton solve, use implicit-function-theorem differentiation through the root (jaxopt-style), so d(J_bs)/d(B_mn) at fixed ambipolarity is one extra linear solve.
5. Derivative verification to rounding, published as a table (NTX's practice; yancc did not do it). Include the Paul et al. 2019 sensitivities (d flux / d B_mn, d/d Er) as the canonical check.
6. Nxi_for_x (fewer Legendre modes at low speed) - it is cheap to implement with ragged padding/masking in JAX and cuts problem size materially at high Nxi.
7. Mixed CPU/GPU story with explicit memory reporting: report peak device memory and wall time per solve at the yancc NCSX and W7-X resolutions so numbers are directly comparable.
8. float64 throughout (JAX x64 flag); monoenergetic bootstrap coefficients at low nu are cancellation-prone (ICNTS: D31 is the hard coefficient).

### Avoid
1. Duplicating the monoenergetic feature set - NTX exists and yancc already covers it; DKX's differentiators are the full operator, multi-species, Phi1, transport matrices and ambipolar Er with derivatives.
2. Claiming differentiability without numerical evidence (the yancc paper's weakest point) - every headline gradient must have an FD or complex-step check and a cost ratio.
3. Verifying J_bs by its Er=0 collisionless limit: arXiv 2407.21599 shows the 1/nu off-set current does not converge and oscillates in log nu_*; convergence (nu_*^(3/5)) needs finite Er. Design the low-nu verification scan with Er.
4. Assembling dense per-L, per-species x-blocks of the full operator on device at large Nx * Nspecies * Nxi - assemble only the preconditioner blocks.
5. Depending on the QS isomorphism (Redl) anywhere inside DKX - the point is to be valid where Redl is not.
6. Running comparisons on a shared/unpinned checkout of SFINCS or yancc; pin commits and resolutions (see memory: never benchmark a shared checkout).

### Compare against (with what and why)
- SFINCS v3: identical model -> agreement to solver tolerance on fluxes, flows, J_bs, transport matrices, Phi1, for the 2014 LHD/W7-X cases and the repo examples. Primary correctness claim.
- yancc: performance and memory at the SAME resolution and hardware class (NCSX (7,121,43,65), A100, 6 GB; W7-X monoenergetic). Secondary claim: DKX competitive or better, plus features yancc lacks (Phi1, ambipolar Er, transport matrices, verified gradients).
- MONKES / NTX: monoenergetic limit of DKX (pitch-angle-scattering only, single x) must reproduce D11/D31/D33 to <1% on the W7-X EIM/KJM and CIEMAT-QI cases with the quoted resolutions.
- NEO-2 (via arXiv 2407.21599): J_bs(nu_*) with finite Er, nu_*^(3/5) off-set decay - physics-level verification independent of SFINCS.
- KNOSOS: low-nu fluxes and Phi1 with Er on W7-X/TJ-II where the orbit-averaged model is valid.
- PENTA / Redl / Sauter: quantify the error of the fast closures on real designs (Infinity Two, Helios, Stellaris) - the "accuracy map" the design teams need.
- Paul et al. 2019 adjoint: reproduce sensitivity results and show generality (any output, any input) at the same cost.


## (iii) Results the community would value most in 2026

1. Verified, differentiable KINETIC bootstrap current for non-quasisymmetric reactors, used inside an optimizer. Demonstrate d J_bs / d(boundary or B_mn) from jax.grad matching FD, then a DESC or SIMSOPT optimization with a DKX bootstrap-consistency objective on a QI (Stellaris/CIEMAT-QI4X-like) and a QA-with-large-J_bs (Helios-like, 4.2 MA) case. Why: DESC's own tutorial says the Redl route "will not work for configurations that are not quasi-symmetric"; Infinity Two iterates SFINCS with VMEC by hand; Thea optimizes bootstrap consistency with QS surrogates. No published tool does this today.
2. An accuracy map of fast closures against the full kinetic answer on real designs: Redl/Sauter (isomorphism), PENTA (momentum-corrected DKES), monoenergetic-only J_bs, versus DKX full-operator multi-species J_bs across nu_*, Er, Z_eff, on Infinity Two, Helios, Stellaris, W7-X. Why: Saxena et al. 2025 explicitly frame this quantification as the open need for MHD codes; the design teams cite SFINCS as the check but cannot afford it in loops.
3. Neoclassical Jacobians for transport solvers: d(Gamma_s, Q_s)/d(dn/dr, dT/dr, Er) and the ambipolar Er root with its derivatives, at a cost of a few solves, delivered as a T3D-compatible module. Why: Infinity Two's T3D-GX-SFINCS pipeline makes SFINCS the CPU, no-derivative component of a predictive workflow; a GPU differentiable neoclassical module removes it as the bottleneck and enables Newton-type profile solvers.
4. GPU Phi1 + impurity transport with autodiff: reproduce Mollen 2018 / Garcia-Regana 2017 Phi1 results, then scan W7-X NBI scenarios (neoclassically dominated impurity peaking ~ Z, PPCF 2025) with impurity + bulk + electrons and the classical channel. Why: yancc and MONKES/NTX have no Phi1; W7-X impurity physics is the live experimental question; KNOSOS Phi1 is low-nu only.
5. Shaing-Callen off-set current scan with finite Er reproducing the nu_*^(3/5) decay (NEO-2 result, arXiv 2407.21599) with the full Fokker-Planck operator, plus the corresponding gradient d J_bs / d Er. Why: a code-independent physics check on the hardest coefficient (ICNTS: D31 "not predicted" by theory) that also demonstrates why Er must be inside the differentiable loop.
6. A reproducible performance/memory table at fixed resolution against SFINCS (128 cores, >50 GB) and yancc (1 A100, 6 GB) with an ablation of preconditioners (SFINCS-style block-exact via NTX blocks vs multigrid), and derivative cost ratios. Why: this is the table every reviewer of a "reimplementation" paper asks for, and the numbers to beat are now public.


## (iv) Full reference list with URLs
(F = fetched; S = seen verbatim in a search result; U = UNVERIFIED detail)

SFINCS and Phi1 / adjoint
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014), DOI 10.1063/1.4870077, arXiv 1312.6058 (F) https://arxiv.org/abs/1312.6058
- SFINCS repository, landreman/sfincs; input.namelist symlink page (F) https://github.com/landreman/sfincs/blob/master/fortran/version3/input.namelist ; issue #1 quoting preconditionerOptions (F) https://github.com/landreman/sfincs/issues/1
- Mollen, Landreman, Smith, Braun, Helander, "Impurities in a non-axisymmetric plasma: transport and effect on bootstrap current", arXiv 1504.04810 (S) https://arxiv.org/abs/1504.04810
- Mollen, Landreman, Smith, Garcia-Regana, Nunami, "Flux-surface variations of the electrostatic potential in stellarators: impact on the radial electric field and neoclassical impurity transport", PPCF 60, 084001 (2018) (S) https://www.osti.gov/biblio/1499870
- Garcia-Regana et al., "Electrostatic potential variation on the flux surface and its impact on impurity transport" (2017) (S) https://www.researchgate.net/publication/271079728_Electrostatic_potential_variation_on_the_flux_surface_and_its_impact_on_impurity_transport
- Buller et al., "The importance of the classical channel in the impurity transport of optimized stellarators", J. Plasma Phys., arXiv 1903.12511 (S) https://arxiv.org/html/1903.12511
- Paul, Abel, Landreman, Dorland, "An adjoint method for neoclassical stellarator optimization", arXiv 1904.06430 (S) https://arxiv.org/abs/1904.06430
- Paul, Landreman, Antonsen, "Adjoint methods for stellarator shape optimization and sensitivity analysis", arXiv 2005.07633 (S) https://arxiv.org/pdf/2005.07633
- Paul et al., "Adjoint approach to calculating shape gradients for three-dimensional magnetic confinement equilibria" (S) https://www.osti.gov/pages/biblio/1597704
- Paul et al., "Adjoint methods for quasisymmetry of vacuum fields on a surface", arXiv 2108.11433 (S) https://arxiv.org/pdf/2108.11433

yancc
- Conlin, Landreman, "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv 2607.20861 (F abstract and full text) https://arxiv.org/abs/2607.20861 ; https://arxiv.org/html/2607.20861
- yancc repository (F) https://github.com/f0uriest/yancc

MONKES / NTX
- Escoto, Velasco, Calvo, Landreman, Parra, "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", Nucl. Fusion, DOI 10.1088/1741-4326/ad3fc9, arXiv 2312.12248 (F abstract and full text) https://arxiv.org/abs/2312.12248 ; https://arxiv.org/html/2312.12248 ; IOP (S) https://iopscience.iop.org/article/10.1088/1741-4326/ad3fc9
- MONKES repository (F, from paper) https://github.com/JavierEscoto/MONKES/
- Escoto Lopez, PhD thesis, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas", arXiv 2510.27513 (F) https://arxiv.org/abs/2510.27513
- "Evaluation of neoclassical transport in nearly quasi-isodynamic stellarator magnetic fields using MONKES", arXiv 2410.17836 (S) https://arxiv.org/pdf/2410.17836
- NTX repository, uwplasma/NTX (F) https://github.com/uwplasma/NTX

KNOSOS, DKES, PENTA, NEO-2, NEOTRANSP
- Velasco, Calvo, Parra, Garcia-Regana, "KNOSOS: a fast orbit-averaging neoclassical code for stellarator geometry", J. Comput. Phys. (2020), DOI 10.1016/j.jcp.2020.109512, arXiv 1908.11615 (F) https://arxiv.org/abs/1908.11615
- "Fast simulations for large aspect ratio stellarators with the neoclassical code KNOSOS", arXiv 2106.01727 (S) https://arxiv.org/pdf/2106.01727
- Hirshman, Shaing, van Rij, Beasley, Crume, "Plasma transport coefficients for nonsymmetric toroidal confinement systems", Phys. Fluids 29, 2951 (1986) (S) https://pubs.aip.org/aip/pfl/article-abstract/29/9/2951/944354/ ; OSTI (S) https://www.osti.gov/servlets/purl/6092128
- van Rij, Hirshman, "Variational bounds for transport coefficients in three-dimensional toroidal plasmas", Phys. Fluids B 1, 563 (1989) (S) https://pubs.aip.org/aip/pfb/article-abstract/1/3/563/940728/
- "Modelling of relativistic electron transport with non-relativistic DKES solver", J. Plasma Phys. (2024) (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/modelling-of-relativistic-electron-transport-with-nonrelativistic-dkes-solver/AD83F02C17589A40A279B7FEB4514FF2
- Spong, "Generation and damping of neoclassical plasma flows in stellarators", Phys. Plasmas 12, 056114 (2005) (S) https://pubs.aip.org/aip/pop/article/12/5/056114/1015589/
- "Three-dimensional equilibria and transport in RFX-mod: A description using stellarator tools", Phys. Plasmas 18, 062505 (2011) (S) https://pubs.aip.org/aip/pop/article-abstract/18/6/062505/387754/
- Kernbichler et al., "Recent progress in NEO-2 - a code for neoclassical transport computations based on field line tracing", Plasma Fusion Res. 3, S1061 (2008) (S) https://www.jstage.jst.go.jp/article/pfr/3/0/3_0_S1061/_article/-char/en
- Kernbichler, Kasilov, Kapper, Martitsch, Nemov, Albert, Heyn, "Solution of drift kinetic equation in stellarators and tokamaks with broken symmetry using the code NEO-2", PPCF 58, 104001 (2016) (S) https://iopscience.iop.org/article/10.1088/0741-3335/58/10/104001
- Beurskens et al., "Demonstration of reduced neoclassical energy transport in Wendelstein 7-X", Nature (2021) (S; author list U) https://www.nature.com/articles/s41586-021-03687-w
- W7-X power balance study (NEOTRANSP use), PPCF (2025) (S) https://iopscience.iop.org/article/10.1088/1361-6587/ade824
- EUTERPE vs NEOTRANSP Er benchmarking figure (S) https://www.researchgate.net/figure/Benchmarking-of-the-global-neoclassical-radial-electric-field-calculated-with-EUTERPE_fig1_378516498

ICNTS
- Beidler et al., "Benchmarking of the mono-energetic transport coefficients - results from the ICNTS", Nucl. Fusion 51, 076001 (2011) (F) https://iopscience.iop.org/article/10.1088/0029-5515/51/7/076001

Bootstrap current and optimization
- Redl, Angioni, Belli, Sauter, "A new set of analytical formulae for the computation of the bootstrap current and the neoclassical conductivity in tokamaks", Phys. Plasmas 28, 022502 (2021) (S) https://pubs.aip.org/aip/pop/article/28/2/022502/124727/ ; open copy https://pure.mpg.de/rest/items/item_3288698/component/file_3288920/content
- Sauter, Angioni, Lin-Liu, Phys. Plasmas 6, 2834 (1999) - U (no URL fetched)
- Landreman, Buller, Drevlak, "Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement", Phys. Plasmas 29, 082501 (2022), DOI 10.1063/5.0098166, arXiv 2205.02914 (F) https://arxiv.org/abs/2205.02914 ; https://pubs.aip.org/aip/pop/article/29/8/082501/2844977/
- Albert, Beidler, Kapper, Kasilov, Kernbichler, "On the convergence of bootstrap current to the Shaing-Callen limit in stellarators", arXiv 2407.21599 (F) https://arxiv.org/abs/2407.21599
- Saxena, Ferraro, Martin, Wright, "Bootstrap current modeling in M3D-C1", J. Plasma Phys. 91, E141 (2025), DOI 10.1017/S0022377825100834, arXiv 2507.05166 (F) https://arxiv.org/abs/2507.05166 ; https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/bootstrap-current-modeling-in-m3dc1/07AEC30A1077F0D427FF2EA7BF42AC4B
- DESC tutorial "Bootstrap Current Self-Consistency" (S) https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html
- VMEX references page (S) https://vmex.readthedocs.io/en/latest/project/references.html
- Goodman et al., "Quasi-isodynamic stellarators with low turbulence as fusion reactor candidates", PRX Energy 3, 023010 (2024), arXiv 2405.19860 (S) https://arxiv.org/abs/2405.19860 ; https://link.aps.org/doi/10.1103/PRXEnergy.3.023010
- Goodman et al., "Constructing precisely quasi-isodynamic magnetic fields", J. Plasma Phys. (2023) (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/constructing-precisely-quasiisodynamic-magnetic-fields/6601E449C8DD3B3FEB361DA2C5732EFC
- Jorge et al., "A single-field-period quasi-isodynamic stellarator", J. Plasma Phys. (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/singlefieldperiod-quasiisodynamic-stellarator/9B2A5FDCCD7774E4F91BE45E75FDC6B0
- "CIEMAT-QI4X: a reactor-relevant quasi-isodynamic stellarator configuration compatible with an island divertor", Nucl. Fusion, arXiv 2512.08825 (S) https://arxiv.org/pdf/2512.08825 ; https://iopscience.iop.org/article/10.1088/1741-4326/ae54ad
- "Near-axis quasi-isodynamic database", arXiv 2601.08400 (S) https://arxiv.org/pdf/2601.08400
- "Optimization of nonlinear turbulence in stellarators", J. Plasma Phys. 90, 905900210 (2024) (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/optimization-of-nonlinear-turbulence-in-stellarators/916FCC56452B5B166C14868F56D99AF5

2024-2026 designs, surrogates, W7-X
- Type One Energy, "A comprehensive, unified baseline physics design for the Type One Energy stellarator fusion pilot power plant, 'Infinity Two'", J. Plasma Phys. 91, E65 (2025) (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/comprehensive-unified-baseline-physics-design-for-the-type-one-energy-stellarator-fusion-pilot-power-plant-infinity-two/CB8A21D770BFA375A9865A28EFBE800B
- "Predictions of core plasma performance for the Infinity Two fusion pilot plant", J. Plasma Phys. (2025) (S) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/predictions-of-core-plasma-performance-for-the-infinity-two-fusion-pilot-plant/707D1908C7D42B614D5561987EA7E16D
- Thea Energy, "Overview of the Helios Design: A Practical Planar Coil Stellarator Fusion Power Plant", arXiv 2512.08027 (S) https://arxiv.org/abs/2512.08027 ; PDF https://thea.energy/wp-content/uploads/2025/12/20251210_FPP_Helios_overview_paper.pdf ; Fusion Eng. Des. (2026) https://www.sciencedirect.com/science/article/pii/S0920379626003868
- "Equilibrium optimization of the Helios planar coil stellarator power plant", Fusion Eng. Des. (2026) (S) https://sciencedirect.com/science/article/pii/S0920379626002905
- "Stellarator fusion systems enabled by arrays of planar coils", Nucl. Fusion (2025) (S) https://iopscience.iop.org/article/10.1088/1741-4326/ada56c
- Proxima Fusion, "Stellaris: A high-field quasi-isodynamic stellarator for a prototypical fusion power plant", Fusion Eng. Des. (2025) (S) https://www.sciencedirect.com/science/article/pii/S0920379625000705 ; press release https://www.proximafusion.com/press-news/proxima-fusion-and-partners-publish-stellaris-fusion-power-plant-concept-to-bring-limitless-safe-clean-energy-to-the-grid
- "Quantitative comparison of impurity transport in turbulence reduced and enhanced scenarios at Wendelstein 7-X", Nucl. Fusion (2023) (S) https://iopscience.iop.org/article/10.1088/1741-4326/aceb76
- "The suppression of anomalous impurity transport above a critical normalized density gradient scale length in Wendelstein 7-X", PPCF (2025) (S) https://iopscience.iop.org/article/10.1088/1361-6587/add597
- "Neural network-based surrogate model for 3D edge-plasma transport in the standard configuration of W7-X", Nucl. Fusion 66 (2025) (S) https://iopscience.iop.org/article/10.1088/1741-4326/ae203d
- IPP abstract "Neoclassical transport simulations for stellarators" (DCOM/NNW description) (S) https://pure.mpg.de/rest/items/item_2139735_1/component/file_2139734/content
- MMMnet surrogate (NSTX-U) (S) https://www6.lehigh.edu/~eus204/per/publications/journals/tps24_MMMnetNSTXU.pdf
- "5D Neural Surrogates for Nonlinear Gyrokinetic Simulations of Plasma Turbulence", arXiv 2502.07469 (S) https://arxiv.org/pdf/2502.07469
- "Efficient dataset construction using active learning and uncertainty-aware neural networks for plasma turbulent transport surrogate models", arXiv 2507.15976 (S) https://arxiv.org/pdf/2507.15976

