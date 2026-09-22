# DKX research plan

**Authoritative plan, written 2026-09-06 by reconciling three predecessors: the R0–R11 work queue (`be6506fe`), [#189](https://github.com/uwplasma/DKX/pull/189) and [#190](https://github.com/uwplasma/DKX/pull/190). It is the only planning file in the repository, by governance test. Planning only; nothing here authorizes a release.**

## 0. Start here

This document is written so that an independent agent can pick it up cold and know what to do on Monday. Read section 0, then the phase you are assigned in section 4, then the working method in section 7. Everything else is reference.

**What DKX is.** A JAX reimplementation of the SFINCS Fortran v3 drift-kinetic neoclassical transport code: radially local, linearized drift-kinetic equation on a flux surface; fluxes, flows, bootstrap current, transport matrices and ambipolar radial electric fields; pitch-angle-scattering and full linearized Fokker–Planck collisions; Phi1; analytic, VMEC and Boozer geometry; SFINCS deck and HDF5 compatibility; differentiable end to end; CPU and GPU. Physics and solver policy live in `src/dkx`; reusable linear algebra, Krylov, factorization, recycling and implicit-differentiation primitives live in the sibling library [SOLVAX](https://github.com/uwplasma/SOLVAX). The closest external codes are SFINCS (the model), yancc (differentiable GPU full-DKE, no Phi1, no ambipolar root), MONKES and NTX (monoenergetic), and KNOSOS (bounce-averaged).

**How this file is maintained.** It is a logbook as much as a plan, and it is the handoff: an agent who has never seen this project must be able to read it alone and know what was tried, what was measured, what was killed and why. Revisions therefore *append and supersede in place* — a step whose premise a measurement destroys keeps its text and gains the measurement and the retraction beside it. Nothing is deleted because it turned out to be wrong; a wrong turn that is removed is a wrong turn the next agent repeats. Prose that is merely long is not a defect here. Slim the code, not this file.

**Where things are.** `src/dkx/solve.py` owns the three linear-solve routes and the `method: auto` policy; `coarse_precond.py`, `multigrid.py` and `sparse_precond.py` are three inverses of one SFINCS-simplified operator used as the Krylov preconditioner; `er.py` owns the ambipolar root, differentiated by the implicit function theorem with original-equation admission; `batch.py` owns sharded independent batches; `api.py` exposes `prepare_er_scan`, `batched_er_scan` and `ErProblem.with_profiles`; `tools/benchmarks/parity_performance_matrix.py` is the supervised, resumable, verifiable benchmark runner; `validation/` holds the sealed cross-code artifacts and `baseline.toml`; `docs/` is Sphinx built with `-W`; `examples/01`–`09` is the example ladder. CI is thirteen coverage shards balanced by `.test_durations`, a wheel-install job that measures the size contract, a docs job, and a `commit-trailers` gate: every commit is authored by the maintainer alone.

**How to run things.**

```bash
pip install -e .                                                  # from a checkout; Python >= 3.11
JAX_ENABLE_X64=True DKX_CI=1 pytest -q -n 4 -m "not slow"         # the PR gate, about 10 min on a laptop
python -m sphinx -b html -W docs _build                            # docs must build warning-free
dkx run examples/01_tokamak_profile/case.toml --out r.nc && dkx converge examples/01_tokamak_profile/case.toml
```

**How progress is recorded.** This file holds phases, figures and criteria and never status prose. Decisions go to `docs/adr/NNNN-*.md`, one page each, immutable. Time-boxed experiments go to `docs/experiments/` as one-page records. What shipped goes to `CHANGELOG.md` per release. Execution detail lives in PR descriptions. Review reports are dated files under `docs/reviews/`. Section 9 records how this plan came to be and where its predecessors are.

**What to do first.** Phase 0 in section 4 is a checklist with commands.

**What never to do.** Do not silently change collisions, trajectories, precision, resolution, root-search scope or requested parallel execution. Do not call an approximation a full model. Do not promote a smoke grid to a research result. Do not compare DKX solve-only time with SFINCS whole-process time. Do not report a warm number without its cold companion. Do not open a stack of PRs without the tooling in section 7. Do not add evidence or provenance infrastructure until a figure or a reviewer asks for it. Do not begin a second algorithm experiment before the first has a written decision.

## 1. Mission, retained decisions, thesis

**Mission.** Deliver a research-grade, differentiable neoclassical code for stellarators and tokamaks with the scientifically supported functionality of SFINCS Fortran v3, reliable CPU/GPU execution, and derivatives that are verified rather than claimed. The flagship calculation is: specified toroidal equilibrium and species profiles → resolved transport and bootstrap current → a retained stellarator ambipolar branch → checked derivatives → fast repeated evaluations → one actual equilibrium-boundary optimization. For tokamaks `E_r` is prescribed; an intrinsically ambipolar local model cannot determine it. Density and temperature stay explicit; pressure alone fixes neither.

| Retained decision | Contract |
| --- | --- |
| Native interface | Immutable `Case`, TOML first and equivalent JSON; `Result` with versioned NetCDF. Physical units are normalized once. Source-relative paths and deterministic semantic IDs. |
| Compatibility | Permanent SFINCS namelist/HDF5 adapters, with explicit unsupported controls; DKX 3 may change the DKX 2 Python API. |
| Expert interface | Typed geometry, grids, operator, moments, prepared solve, and sensitivity contracts for composition with other codes. File I/O and Python orchestration need not be differentiable. |
| Runtime | Python ≥3.11, `src/dkx`, explicit runtime configuration; CLI uses argparse/Rich. Core emits structured progress and diagnostics. |
| Documentation | Sphinx/MyST/Furo; tutorials, how-to guides, explanation, reference. One source for schemas and capability status. |
| Algorithms | SOLVAX owns reusable linear algebra, differentiation primitives, and generic parallel primitives. DKX owns physics, discretization, and physics-dependent solver policy. |
| Platforms | Named M3 Max laptop CPU baseline and office NVIDIA accelerator lane. Performance results identify exact hardware and toolchain. |
| Size | Fresh full clone, wheel, sdist, and installed DKX-owned files each target <20 MiB. Report each measurement separately; dependencies are excluded from owned-file size, not from installation instructions. Measured 2026-09-01 on a fresh clone: 14.49 MiB tracked tree (3.07 MiB media), 14.83 MiB object store, 29.32 MiB total. This leaves about 5.5 MiB for Git metadata/history under the existing target; it does not prove that a rewrite could or could not achieve it. Measure reachable packed objects before considering another rewrite. Keep the <20 MiB target; do not silently restate it to make a gate pass. |
| Quality | 95% line **and** branch coverage of stable reachable code is the release target. Maintainer-deferred coverage work must not block urgent correctness fixes. Assertions must test behavior or science. |
| Workflow | One coherent implementation slice per PR; commits authored and committed by `rogeriojorge`, without assistant coauthor trailers. Preserve uncommitted work. |

Do not silently change collisions, trajectories, precision, resolution, root-search scope, or requested parallel execution. Do not call an approximation a full model. New evidence may change a policy, but must identify the affected contract and replace the superseded claim.

**Thesis.** The unit of progress is a figure, not a gate: each phase in section 4 is defined by a publishable figure or table, and code, tooling and documentation are written when a figure needs them. The differentiator is not "differentiable GPU neoclassics", which yancc has published; it is the complete SFINCS-v3 model including Phi1 and the ambipolar root, on GPU, with an error bar on every reported observable and derivatives verified through the root and through geometry. The algorithmic work is a diagnosis first and two bounded, kill-gated experiments second, because DKX already implements SFINCS's simplified-operator preconditioner as its default and the reasons the Krylov route still loses are specific and measurable.

## 2. State of the code and the evidence, 2026-09-06

### 2.1 What landed with the implementation stack (#169–#188)

| Work to retain | Source / review | Remaining boundary |
| --- | --- | --- |
| Supervised, resumable benchmark attempts; original residual and complete-output checks; PETSc backend/provenance verification | #169–170, #176–177, #181, #183, #185; `tools/benchmarks/parity_performance_matrix.py` | Fresh installed production replay, valid references and observable/grid admission are still needed. |
| Static operator layouts, refreshed collision coefficients and native profile preparation | #173–174, #178, #182, #186–187; `drift_kinetic.py`, `collisions.py`, `execution.py`, `er.py` | Profile updates are opt-in and hold geometry, normalization, species, Coulomb log and discrete layout fixed. They do not make `Case.run` a JAX transformation. |
| Independent batch sharding through JIT and gradients, uneven-batch handling and per-input algebraic status | #179, #182, #187–188; `batch.py`, `api.py` | One whole system remains on one device. Native Case execution does not expose every expert batch option; memory budgets are estimates. |
| Original kinetic and transpose checks; qualified dense adjoint references; native profile-to-root Taylor tests | #180, #184, #188; `solve.py`, `er.py`, solver/root tests | Small fixed-grid evidence does not establish resolution, branch or geometry uncertainty. Partial state recovery has a different derivative contract. |
| Generated Schur-factor reuse across RHSs, forward/transpose solves and refinement | #188 `be6506fe`; existing SOLVAX generated-factor API | Factors belong to one solve execution. Persistent reuse across changed operators is not implemented by this change. Full-FP uses the Krylov route. |

SOLVAX `>=0.19.0` is sufficient for everything the stack uses; requalify 0.20.x explicitly rather than by branch. `examples/08_vmex_optimization` is an analytic Boozer-spectrum proxy that does not import VMEX; real VMEX calls exist only in optional scripts under `examples/optimization` and `examples/autodiff`.

### 2.2 Useful results, with their limits

The #188 report records 244 distinct CPU cases and 22 GPU cases for its factor
integration, and native root Taylor orders of 2.00–2.01. On one **7,850-unknown,
three-field PAS objective**, twelve alternating synchronized pairs reduced warm
value/gradient medians from **67.59 to 37.09 ms on CPU** and **221.74 to 195.56 ms
on A4000**; GPU LU calls fell from 288 to 144. CPU used M3 Max/JAX 0.9.2; GPU used
JAX 0.10.2. These compare two checked routes
on each host. They are not full-FP, fresh compilation, allocator peak, whole-root,
whole-optimization or cross-host scaling results. The underlying external traces
were reported in the prior review; the 2026-09-06 review did not relabel them as new runs.

Earlier README claims about a 744k HSX case and a broad upstream speed ranking
are historical, with differing runtime/memory definitions and invalidated
reference campaigns elsewhere in the record. Retain their evidence in the
performance documentation; do not use them as an unqualified headline. In
particular, small algebraic residuals and agreement with a failed Fortran run
cannot certify a transport observable.

The 2026-09-06 review evidence is outside Git in `dkx-plan-evidence-20260906`, with exact commands,
inputs, source identities and logs. Its role is to choose
work, not to declare a new production benchmark. The initial bounded local pilot
already reproduced a crucial distinction: a SFINCS full-FP solve reported success
but its original residual was **1.10e-9 at a requested 1e-10**; DKX's was 8.48e-11.
A separate direct reference also missed the gate. Neither failed-reference pair is admitted for a
performance comparison. An initial HSX copy omitted its equilibrium and was
rejected; its partial DKX state also failed the full residual check. Repair inputs
and explicitly choose complete-state versus moment-only evidence before rerunning.

A subsequent **right-preconditioned GMRES/MUMPS** run with an unpreconditioned
stopping norm and inner rtol=1e-13 passed the common original 1e-10 gate: SFINCS
1.20e-11 and DKX 8.48e-11. On this full-FP grid (5×5 angles, Nxi=6, Nx=3, two species, ramped
pitch; 654 rows in the constrained Fortran matrix), scaled differences
were 4.08e-6 for flow, 3.94e-6 for parallel current and below 8.64e-8 for fluxes.
The PETSc log records one symbolic factorization, one numeric factorization and
22 factor applications. This establishes a bounded algebraic comparison, not
grid convergence or a runtime ranking; concurrent review tests exclude performance
promotion. Solver/norm configuration must therefore be part of reference admission.

| Check on the reviewed source (2026-09-06) | Result / limit |
| --- | --- |
| DKX solver, Er, batch, native execution and planning suites | 248 passed on M3 Max CPU/JAX 0.9.2/SOLVAX 0.20.0; not a full coverage campaign. |
| Office GPU selection | Four targeted batch/AD/full-state cases passed. The unchanged two-device harness also passed PAS/full-FP states, original residuals, uneven batches, actual placement, JIT gradients and an FD check on two A4000s, JAX 0.10.2. |
| YANCC at `6f399a21` | 59 preconditioner/solve/collision tests passed locally, including its SFINCS/MONKES fixture comparisons, coordinate representations, warm state and derivative tests. This does not establish comparative runtime or general warm-reuse correctness. |
| README examples | Native Python and CLI run/inspect pass; Python including the advanced JIT gradient also passes from an isolated DKX install inheriting host dependencies. |
| Existing independent comparison artifact | All three coefficient rows and external YANCC input hashes pass the offline audit; this is an artifact audit, not three fresh kinetic benchmarks. |
| Documentation | Five planning/wording checks and standard Sphinx `-W` pass. Extra `-n` reference checking finds 219 warnings in both baseline and revised docs; resolve this existing API-link debt during consolidation. |

Facts established by the 2026-09-06 code survey that earlier plans got wrong or missed: the SFINCS-style simplified-operator preconditioner already exists and is DKX's default Krylov preconditioner (`coarse_precond.py:916`: self-species, x-diagonal collisions; L±2 Er and drift terms dropped; exact block-Thomas inverse over (species, x); Schur-eliminated border; Phi1-aware), and `multigrid.py` and `sparse_precond.py` are alternate inverses of the same operator that the escalation ladder already tries in order (`solve.py:1072–1248`). Of the six upstream decks that did not complete, five were killed by memory while the coarse preconditioner allocated its dense (N_θN_ζ)² bands (42.9 GB on `filteredW7XNetCDF_2species_magneticDrifts_noEr`) and one is a LIBSTELL `wout` reader case; memory-lean routes now exist and those decks run in 50 minutes to 2 hours 17 minutes. The ambipolar root is differentiated by the implicit function theorem (`solvax.implicit.root_solve` = `jax.lax.custom_root`, `er.py:909–1037`) with finite-slope and original-equation admission. Batched scans thread neither recycle spaces nor preconditioners across points; the root driver does when `warm_start=True`; the adjoint solve always cold-starts and has no recycle space of its own. There is no banana/plateau/Pfirsch–Schlüter flux-limit test in `tests/`.

### 2.3 Capability priorities

| Capability | State at the reviewed head | Decision |
| --- | --- | --- |
| Analytic, VMEC and Boozer native profiles; PAS and full-FP; prescribed Er and stellarator roots | Implemented in a restricted DKES/no-Phi1 native domain | First supported research envelope; qualify multispecies fluxes **and** current separately. |
| Prepared native profile/Er sensitivities and expert batching | Implemented with explicit fixed dependencies | Productize and measure, rather than introduce another public interface. |
| RHSMode 1/2/3, richer trajectories, Phi1, magnetic drifts and distribution export | Broader compatibility/expert support | Preserve regression coverage; a checkmark must distinguish equation, native interface, derivative and validation support. |
| Native single-surface profiles, transport matrices, explicit Case sharding, Phi1/full drifts | Rejected or incomplete in `execution.py` | Single-surface explicit drives and prepared-scan access are useful near-term interface work; Phi1/full drifts require physics gates, not just removing a rejection. |
| Real VMEX boundary optimization | Example 08 is an analytic harmonic geometry proxy | Keep it as a teaching example; do not cite it as the design deliverable. |

Extend the existing `validation/capabilities.toml` and SFINCS control inventory
semantically: collision operators and backgrounds; trajectories/Er terms;
Phi1/quasineutrality and gauge; geometries/asymmetry/radial conventions;
RHS modes, sources/constraints and moments; grids/potentials; exports and solver
controls. The recorded 145 declared namelist controls are an inventory, not 145
validated scientific features. Map unsupported combinations to explicit reasons,
references and tests. Do not promise parity with every experimental SFINCS branch.

### 2.4 Position relative to yancc, MONKES, NTX and KNOSOS

A research-grade plan states what the code is expected to demonstrate that its neighbours do not, and where it must not overclaim. Reported numbers below are the other codes' own; none has been reproduced here.

| Code | Model and method | Reported | What DKX must show against it |
| --- | --- | --- | --- |
| [yancc](https://arxiv.org/abs/2607.20861) | Full 4D DKE, linearized FP with field-particle terms and speed/pitch electric-field terms; tangential magnetic-drift support requires a separate audit; `E_r` is an input, no Phi1 self-consistency; finite differences with a diagonally dominant upwind stencil, Maxwell collocation in speed, semi-coarsened multigrid V-cycle preconditioning GCROT; JAX. | The paper reports close SFINCS/MONKES agreement and roughly an order-of-magnitude per-scan speed/memory improvement. Those are author-reported comparisons, not a DKX benchmark. Audit derivative verification independently of the differentiability claim. | Certified observable error (§3.1), matched Phi1/quasineutrality scope and retained ambipolar-root evidence; verified derivatives through roots and geometry (B10). Historical DKX two-GPU batch scaling was below 1; the R4 checkpoint now records a bounded sharded PAS improvement and its structured direct route is exact only on PAS/DKES; do not claim the 10⁷-unknown GPU envelope until B12 measures it. |
| [MONKES](https://arxiv.org/abs/2312.12248), [thesis](https://arxiv.org/abs/2510.27513) | Monoenergetic DKE, Lorentz collisions, Legendre in pitch and Fourier collocation on the surface, block-tridiagonal direct elimination, O(N_ξ N_fs³). | 4–64× over DKES; about one minute per case on one core at ν̂ = 1e-5; converged with ≤180 Legendre modes and about 2,000 surface points; Onsager symmetry checked. Momentum conservation requires external correction. | The same block-tridiagonal structure DKX exploits on PAS decks; B3 must document its convergence criterion and meet DKX's observable tolerances while checking the required `N_ξ` resolution, and the [Boozer-reader off-by-one](https://github.com/JavierEscoto/MONKES/issues/1) must be excluded from any reference build. |
| [NTX](https://github.com/uwplasma/NTX) | JAX-native monoenergetic solver implementing the MONKES Legendre formulation with an embedded adjoint. | The README reports a 14× adjoint advantage over finite differences at 32 parameters. Pin the benchmark and audit gradient accuracy, normalization and timing boundaries before adoption. | The monoenergetic derivative reference for B3/B10, and a candidate owner of the monoenergetic database workflow rather than a duplicate in DKX; decide ownership explicitly in R5. |
| [KNOSOS](https://doi.org/10.1016/j.jcp.2020.109512) | Bounce-averaged, low-collisionality, radially local; includes the tangential magnetic drift and surface variation of the potential. | Fast enough for optimization loops; valid in its asymptotic regime. | Comparison only inside that regime (B5); a disagreement outside it is not a defect of either code. |
| [PENTA](https://ui.adsabs.harvard.edu/abs/2010APS..DPPTP9124L/abstract), [NEO-2](https://www.semanticscholar.org/paper/ef15bce085ab33694ac20af4ea1ce6591f3dbb0b) | Momentum-corrected transport from DKES-type coefficients; field-line-tracing full linearized collisions. | Standard references for momentum correction and parallel flows. | A momentum-conserving full-FP model should not add a DKES momentum correction; verify discrete conservation independently; these are the references for parallel-flow and bootstrap comparisons in B2/B4 and for the database-to-thermal path in B9. |

The differentiating claims are in the last column of the first row: certified observable error, the complete SFINCS-v3 model with root evidence, and verified derivatives of that model. Speed against yancc on the full DKE is not a differentiator until measured on matched grids.

Numbers from the 2026-09-06 literature survey, each traceable to a fetched source in section 10: yancc's largest reported problem is single-species NCSX at (n_x, n_α, n_θ, n_ζ) = (7, 121, 43, 65) in 6 GB on one A100, against SFINCS at (7, 141, 25, 81) on 128 cores needing over 50 GB, about 5× faster at moderate collisionality; its full text contains no finite-difference or adjoint check of any derivative, no Phi1, and lists the ambipolar `E_r` as future work. MONKES converges low-collisionality W7-X coefficients at N_ξ ≈ 140–180 with (N_θ, N_ζ) ≈ (23, 55–79) in about a minute on one core, and used 1.4 GB where yancc used 4 GB on the same monoenergetic case. NTX reports its adjoint at 14× the cost advantage over finite differences at 32 parameters with agreement to about 2e-14. Albert et al. 2024 show the `E_r = 0` off-set bootstrap current does not converge in the 1/ν regime and decays as `ν*^(3/5)` with finite `E_r`. Saxena et al. 2025 frame "how wrong is the analytic bootstrap closure off-symmetry" as the open question; DESC's own tutorial states the Redl isomorphism does not apply to non-quasisymmetric fields; Infinity Two iterates SFINCS with VMEC by hand; Stellaris is quasi-isodynamic and needs off-Redl verification.

## 3. Scientific contract and evidence hierarchy

### 3.1 SFINCS-v3 functionality to close

The Fortran source declares 145 namelist members in nine groups, inventoried by `tools/parity/output_key_coverage_report.py` at pinned commit `8df5453`. That is an inventory, not 145 validated features. A family is closed only when the source audit, the acceptance test and the documentation agree.

| Family to close | Source/API audit and acceptance requirement |
| --- | --- |
| Species and drives | Charges, masses, kinetic/adiabatic roles, density/temperature gradients, unequal temperatures, inductive drive, reference scales, Coulomb logarithm and quasineutrality assumptions. Independent density and temperature profiles are mandatory; pressure is not a substitute. |
| Collisions | Lorentz/PAS, full linearized multispecies Fokker–Planck and retained Sugama variants. Test field-particle terms, Rosenbluth potentials, conservation, Maxwellian limits, high-Z disparity and unequal-temperature domain. State where an H-theorem/equilibrium nullspace applies. |
| Trajectories | Full and DKES-like choices, compressible/incompressible electric drift, speed and pitch derivatives, `includeXDotTerm`/`includeElectricFieldTermInXiDot`, default combinations. Compare each operator term before comparing moments. |
| Magnetic drifts | Every supported `magneticDriftScheme` (including the 0–9 choices where implemented), radial/tangential terms, curvature/grad-B and electric-field conventions. Native full magnetic-drift selection is still a gap. |
| Phi1 and external distributions | Linear/nonlinear quasineutrality, kinetic species response, density/gauge constraints, external Phi1/distribution inputs, NBI conditional build behavior and temperature-equilibration limitations. Verify coupled residual/Jacobian and admissible physics. Native Phi1 is still a gap. |
| Nullspaces and sources | All retained `constraintScheme` choices, density/energy constraints and source coefficients. Demonstrate constrained rank, uniqueness and normalization; compare physical moments across equivalent constraints. |
| Full-kinetic outputs | RHSMode 1 particle/heat flux, parallel flow/current, bootstrap, classical terms, NTV/momentum diagnostics and electrostatic corrections where supported. Document heat versus energy flux and every physical-unit conversion. |
| Transport/database modes | RHSMode 2/3 matrices, monoenergetic coefficients, field/collisionality/speed scans, thermal convolution, interpolation validity and sign conventions. Compatibility exists; native workflow/Result admission remains incomplete. |
| Ambipolarity | Charge-weighted total radial flux, electric-field coordinate conversion, bracketing/refinement, all retained roots, slope/stability convention, continuation and selection. Compare with SFINCS root utilities and PENTA in matching models. |
| Sensitivities | Supported RHSMode 4/5 and adjoint controls: distinguish actual solves from input validation. Compare primal/adjoint equations, gradients of moments, geometry and profile parameters, and nonlinear Phi1/root derivatives where differentiable. |
| Geometry | Analytic and file-backed geometry schemes 1–5, external VMEC/Boozer families 11–13, asymmetric equilibria, signs/orientation, radial interpolation, field periods and Fourier truncation. Match source domains rather than treating equal integer settings as equal grids. |
| Resolution | `Ntheta`, `Nzeta`, `Nxi`, `Nx`, `NL`, speed maximum/grid scheme, interpolation/quadrature, active pitch layout, boundary conditions and Rosenbluth resolution. Joint refinement must close angular/pitch/speed coupling. |
| Execution and export | Iterative/direct routes, tolerances, preconditioners, nonlinear controls, all export fields, dump formats and restart semantics. Compare complete successful outputs; never accept a partial HDF5 simply because it opens. |
| Research workflows | Surface/profile scans, finite-Er databases, impurity transport, full-FP ambipolar calculations, bootstrap/equilibrium iteration and differentiable objectives. Document model validity and tested parameter envelope for each. |

### 3.2 Evidence hierarchy and certificates

Separate four questions: (1) does the code implement the stated equations, (2) are those equations solved accurately, (3) is the model appropriate and independently supported, and (4) does the workflow finish within its resource budget? Cross-code agreement answers none of these alone when both codes share a discretization error or incompatible normalization.

* **Mathematics/code verification:** manufactured distributions and forcing with analytic moments; polynomial/Gamma-function quadrature identities; periodic Fourier derivative symbols; active-grid indexing; adjoint dot products; block extraction/reconstruction and dense nonsymmetric referees; collision invariants and operator limits. Derive expected values independently of production helpers. Perturb a coefficient/sign/weight to confirm that important proofs fail.
* **Algebraic error:** retain original-system `r = b - A x`, absolute and relative norms and normwise/componentwise backward error. Check constraints separately. Establish rank and gauge before interpreting a condition estimate. A tiny global residual does not bound every small flux.
* **Observable error:** for a linear observable `Q = cᵀx`, solve `Aᵀλ = c`; `λᵀr` estimates the algebraic error, with adjoint error accounted for. Nonlinear observables need a linearization remainder or conservative refinement evidence. Do not call this a rigorous bound without its assumptions. Equilibration may improve numerical scaling; certification uses the original physical operator and units.
* **Numerical uncertainty:** refine each axis and jointly refine angular/pitch/speed/potential grids, Fourier truncation, radial interpolation, solver tolerances, and nonlinear/root tolerances. Track every published observable. Near-zero quantities use physically motivated absolute tolerances in their own units plus relative tolerances; never normalize current by a heat flux or discard an inconvenient species using another species' largest moment.
* **Model validation:** specify local ordering, collisionality, orbit-width and electric-field assumptions, collision model, magnetic drifts, Phi1, geometry regularity and boundary conditions. Model uncertainty is not a mesh error bar.
* **Differentiation:** JVP/VJP dot products, analytic sensitivities, central differences over a step-size window, and Taylor remainder rates for geometry, collisions, profiles, field and coupled objectives. Report primal and adjoint residuals, root branch, setup/gradient cost and peak memory. Stopping/reuse heuristics must not silently change the differentiated equations.

Existing `test_math.py`, `test_numerics.py`, `test_collision_physics_gates.py`, `test_transport_limits.py`, `test_shaing_callen.py` and solver/transport tests already close substantial proof work. In particular the thermal Lorentz coefficient `8/sqrt(pi)` in DKX normalization and manufactured Gamma-function thermal convolutions are proved; do not reopen them under a different normalization. Full-FP Spitzer–Härm and multispecies transport limits remain distinct from that Lorentz result. Onsager/Onsager–Casimir relations require the appropriate trajectory model, thermodynamic forces, magnetic-field reversal and sign conventions.

For a simple ambipolar root, differentiate `J_r(E_r,p)=0` using `dE_r/dp = -(∂J_r/∂p)/(∂J_r/∂E_r)`. A zero/uncertain slope is **marginal**, not stable; branch creation, tangency and selection switches are nonsmooth events. Sign samples cannot exclude even crossings or tangencies between samples. Axisymmetric local momentum-conserving neoclassical theory is intrinsically ambipolar: it does not select a unique tokamak Er. Use a prescribed field or an explicitly additional closure in that application.

Tier vocabulary, used everywhere: **Tier A code verification** (analytic limits, Onsager symmetry, adjoint versus finite difference); **Tier B solution verification** (convergence per axis with the criterion stated); **Tier C cross-code** (SFINCS parity with the same discretization is regression and is reported separately from MONKES/yancc independence at about 1 percent); **Tier D validation** (W7-X `E_r` against experiment with its uncertainty). Tier C is never called validation.

**Per-observable error budgets.** For each published quantity `Q`, define a physical scale and an application tolerance `atol_Q + rtol_Q |Q|`, and budget algebraic, grid/quadrature, root and geometry errors separately; start with at most 10 percent of the budget for algebraic error and calibrate rather than choosing a universal residual tolerance. For a linear observable `Q = cᵀx` with adjoint `Aᵀλ = c` and residual `r = b − Ax`, the exact discrete identity `Q_exact − Q_computed = λᵀr` gives the algebraic error bar at the cost of one transpose solve; approximate adjoints need an allowance. Richardson extrapolation over the convergence ladder gives the discretization bar. Use absolute scales for near-zero currents and fluxes. For a simple root, propagate current uncertainty through `|dJ_r/dE_r|`; a slope too small or an error overlapping another branch is reported as marginal, never as stable.

### 3.3 Benchmark families

Every family records pinned inputs, independent derivation/reference, model/normalization match, converged observable targets, tolerances with rationale, valid parameter range, CPU/GPU resource envelopes, and failure outcomes. Tiny smoke decks are not publication benchmarks. Extend the existing registry/runner instead of creating a script and JSON for each sweep point.

| ID | Case family | Evidence and use |
| --- | --- | --- |
| B0 | Manufactured operators/moments, grids and constrained linear systems | Analytic proof tests, backward error, conservation, derivatives; ordinary CI. |
| B1 | Uniform field, Lorentz conductivity, full-FP conductivity, high-collisionality limits | Independent normalized derivations; distinguish established Lorentz proofs from additional Spitzer–Härm and general-geometry transport work. |
| B2 | Large-aspect-ratio circular tokamak across collisionality | Banana/plateau/Pfirsch–Schlüter behavior, Shaing–Callen limits, flow/bootstrap and intrinsic ambipolarity; SFINCS plus appropriate analytic theory. Convergence toward the Shaing–Callen limit is slow and resolution-sensitive ([arXiv:2407.21599](https://arxiv.org/abs/2407.21599)); record the approach, not one point. |
| B3 | DSHAPE/NCSX/W7-X monoenergetic, finite Er | Existing MONKES/YANCC comparisons, matched DKES-like equations and Beidler conventions; extend resolution/collisionality coverage and compare full tables. Add [NTX](https://github.com/uwplasma/NTX), the JAX Legendre/block-tridiagonal monoenergetic solver with an embedded adjoint, as the sibling reference for monoenergetic derivatives. The external bar is yancc's reported agreement with MONKES within 1% across collisionality ([arXiv:2607.20861](https://arxiv.org/abs/2607.20861)). |
| B4 | LHD/HSX/W7-X full kinetic multispecies | SFINCS, impurities/high-Z/unequal temperatures, particle/heat/current outputs; finite-Er and collision terms separated. Include a direct yancc comparison on identical grids with a stated tolerance: it is open source, reports <1% against SFINCS on NCSX with about 5% on currents, and is the closest competitor (section 3.3). |
| B5 | Finite-Er trajectory and tangential-drift variants | SFINCS operator parity and converged moments; KNOSOS only inside its bounce-averaged asymptotic domain. |
| B6 | Linear/nonlinear Phi1 and impurity response | Quasineutrality/gauge proofs, full coupled residual and adjoint, independent SFINCS reference. |
| B7 | W7-X ambipolar profile and root events | Current uncertainty, seeded and discovery scopes, branch continuation, failed-point handling; broad-search completeness claimed only with an actual exclusion argument. |
| B8 | VMEC/Boozer conversion and asymmetric tokamak/stellarator | Coordinate/normalization and Fourier/radial errors; independently converged bootstrap rather than a tiny unresolved current. |
| B9 | Database → thermal response | Existing analytic convolution, withheld table points, interpolation/edge refusal and full-kinetic comparison under matched assumptions. |
| B10 | Linear, nonlinear and root sensitivities | Analytic/JVP/VJP/Taylor/finite-difference windows, SFINCS adjoints where supported, nonsmooth-event refusal. The gate is a derivative **through the ambipolar root and through geometry**, not a smooth PAS temperature derivative on a tiny deck: report the Taylor-remainder rate and the FD window for `dE_r/dp` and for a flux with respect to a boundary coefficient. Verify DKX's complete model and compare independently audited derivatives under matching model scope. |
| B11 | Restart and optimization hard cases | Cross-surface 2.46% regression, wide Er reuse, `Er=15` high-pitch case with `FSABjHat` = −3.77e-3 at `Nxi = 180` as a historical high-pitch comparison point pending joint refinement; cold equivalence in observables and bounded failure recovery. |
| B12 | CPU/GPU batch and state distribution | Strong/weak scaling, throughput/latency, real device occupancy, gradients, memory and communication; correctness before speed. |
| B13 | Actual VMEX and ESSOS design chain | Equilibrium → geometry → DKX objective → gradient → constrained design; final independently evaluated, refined design. |
| B14 | NEOPAX transport integration | Consistent profile/grid/flux units, conservation, ambipolar response and lagged-response refresh; prescribed versus evolved state. |

The four families Phase 1 actually runs, with their independent anchors; points are added only to resolve a concrete uncertainty:

Use **four representative families**, adding points only to resolve a concrete
uncertainty. Each has a fast verification grid and a separately converged research
grid; a smoke grid is never silently promoted.

| Family | Required physics and measurements | Independent anchor |
| --- | --- | --- |
| Analytic/axisymmetric tokamak | PAS and multispecies full-FP; prescribed Er; particle/heat flux, parallel flow, bootstrap/conductivity; NZeta=1 versus resolved symmetry | Collision invariants, Spitzer–Härm/full-FP and applicable tokamak limits; SFINCS v3 |
| One structured stellarator and one W7-X surface | PAS/DKES monoenergetic coefficients at zero and finite Er; sign/normalization/Onsager conventions; selected thermal convolution | MONKES/YANCC in matched equations; existing Beidler-normalized fixtures |
| Multispecies stellarator profile | Full-FP, independent n/T drives, finite Er, ion/electron currents and regular root branch | SFINCS/YANCC; native SI versus expert normalized path; independent cold roots |
| A bounded hard case | Existing finite-Er current sign-changing grid ladder and warm cross-surface discrepancy, then one Phi1/drift case when that model is admitted | Original inputs from #160–161, refined referee, appropriate trajectory/Phi1 literature |

For each published quantity Q, define a physical scale and an application tolerance
`atol_Q + rtol_Q * |Q|`. Budget algebraic, grid/quadrature, root and geometry errors
separately. A reasonable initial allocation is no more than 10% of the observable
budget to linear/nonlinear algebraic error; calibrate it, rather than choosing a
universal residual tolerance. Use absolute scales for near-zero current/flux.
For a simple root, propagate current uncertainty through `|dJr/dEr|`; if the slope
is too small or the error overlaps another branch, report unresolved/marginal.

Mathematical verification includes independently derived manufactured forcing and
moments, Fourier derivative symbols, Maxwell/Gamma quadrature identities, active
layout and border identities, nonsymmetric transpose dot tests, collision number,
combined momentum/energy conservation and nullspace/gauge checks. Preserve existing
proofs rather than rewrite them to mirror the implementation. Full-FP conductivity
is a different test from the existing Lorentz `8/sqrt(pi)` result. For linear
`Q=cᵀx`, `Aᵀlambda=c` and `r=b-Ax`, the exact discrete identity is
`Q_exact-Q_computed=lambdaᵀr`; approximate adjoints need an error allowance.
This neither proves grid convergence nor identifies the cause of every discrepancy.

Refine theta, zeta, pitch, speed, Rosenbluth resolution, geometry Fourier truncation
and relevant radial/Phi1 grids both separately and jointly. Compare current and
each species' flux independently. The historical Er=15 pitch ladder changes the
sign of current; it is not fixed by a tighter Krylov residual alone. Recover the
exact #161 source/target pair before attributing its 2.46% difference to conditioning.
If it cannot be recovered, keep it unresolved and construct a new identified
adversarial pair. Do not substitute the latter as a reproduction.

## 4. Phases

Each phase names the figure it produces, its entry and exit criteria, numbered steps, a kill criterion where the work is an experiment, and an effort range in person-weeks. Phases are ordered by dependency; Phase 2 and Phase 3 can run in parallel once Phase 1's error bars exist.

### Phase 0: land, freeze, and set up the working method (1 week)

Entry: this plan merged. Steps, in order:

1. Confirm main carries the integration: the reverts of #171 and #175, the stack #169–#188 squash-merged in order, #191 (`.test_durations`), #189 and this plan. `git log --oneline origin/main | head -25` should show them; `pytest -q -n 4 -m "not slow"` must pass on main and `python -m sphinx -b html -W docs _build` must be clean.
2. Confirm the open-PR list is empty of superseded work: #168 and #172 closed with dispositions; #190 merged as this plan. Anything else open must map to a phase figure or a bug, or be closed.
3. Create `docs/adr/0001-figure-first-planning.md`, `docs/adr/0002-durations-balanced-shards.md` (the reversal of the 2026-07-17 archive decision) and `docs/adr/0003-attribution.md` (every commit authored by the maintainer alone; the `commit-trailers` gate enforces it), using the Nygard template: title, status, context, decision, consequences.
4. Create `docs/experiments/README.md` with the one-page record template (hypothesis, admission test, result, decision) and `CHANGELOG.md` with a `v2.4.0-rc1` entry drafted from the stack's PR descriptions. This is where the execution diary goes from now on.
5. Draft `v2.4.0-rc1` in the changelog, but defer tagging and publication until the maintainer authorizes the release after the important goals are achieved. Freeze evidence tooling: no changes under `tools/benchmarks/` except bug fixes until Phase 1 needs one.
6. Fetch current main and work from a clean branch or isolated worktree. Preserve existing branches and uncommitted work; do not hard-reset an unrelated checkout.

Exit: main green at the integrated tree; ADRs 1–3 and the experiment template exist; the changelog holds the stack's story. A deferred release tag does not block research phases. Effort: one person-week, mostly review.

### Phase 1: the positioning figure (3–4 weeks)

**Why first.** The yancc paper ([Conlin & Landreman 2026](https://arxiv.org/abs/2607.20861)) is now the reference point every reviewer will hold DKX against: full 4D DKE on one A100, NCSX single-species at (n_x, n_α, n_θ, n_ζ) = (7, 121, 43, 65) in 6 GB, about 5× faster than SFINCS on 128 cores at moderate collisionality, agreement within 1% with SFINCS and MONKES. It does not do Phi1, does not find the ambipolar root, and reports no numerical verification of any derivative. Until DKX has a figure on the same problem, every DKX speed claim is unanchored and every "differentiable" claim is undifferentiated from yancc's. This phase produces that anchor and, as a by-product, forces every check #189 lists.

**Figure 1 (methods paper, Fig. "positioning").** One NCSX full-Fokker-Planck single-species case and one two-species case at yancc's published resolutions, plus the W7-X monoenergetic MONKES/yancc case at (N_L, N_θ, N_ζ) = (180, 39, 99). Panels: (a) fluxes and bootstrap current from DKX, SFINCS v3 and yancc with DKX's algebraic error bar `λᵀr` and its Richardson grid estimate drawn on the DKX points; (b) time to accepted observable, cold and warm, CPU (M3 Max) and GPU (A4000), each code at its own converged resolution; (c) peak device/host memory. Discretizations differ (yancc: finite differences in pitch and angles; DKX/SFINCS: Legendre in pitch), so the comparison is at *converged observables*, never at "identical grids" in the literal sense; the figure says so in its caption.

Steps:
1. Install `yancc` from PyPI (0.0.1; depends on lineax, equinox, interpax, orthax) and pin its commit; reproduce its NCSX (7,121,43,65) and W7-X monoenergetic numbers on the A4000 before touching DKX. If its published numbers do not reproduce within 2× on our hardware, record that and proceed with our measurement only.
2. Build the DKX cases from the same VMEC/Boozer files; converge each observable separately (theta, zeta, pitch, speed, jointly) with `dkx converge`; record the accepted grid per observable. Before advertising a collaborator-facing fix, replay the previously failing inputs at unchanged physics and tolerance, compare accepted SFINCS outputs, and retain unsuccessful attempts. Separately test that nested scans propagate failures and retry incomplete points without discarding successful results; passing orchestration tests does not close physics parity.
3. Run SFINCS v3 on the office isolated toolchain (PETSc 3.23.6 / MUMPS 5.8.1) at the accepted grid with the right-preconditioned GMRES/MUMPS configuration that #189's evidence found necessary for a valid 1e-10 original residual.
4. Compute `λᵀr` for every reported linear observable (one transpose solve with the operator's transpose, which every differentiable route already exposes through `_implicit_solve`) and the Richardson estimate from the convergence ladder; these become the error bars. The adjoint solve exists; what is added is the ~50-line utility that forms `λᵀr` per moment and writes it into `Result`, plus its manufactured-solution test.
5. Measure with the existing supervised runner, pinned SHAs, idle machine, five repetitions, medians with dispersion; cold and warm separately; never DKX solve-only against SFINCS whole-process.

Exit: Figure 1 rendered from tracked inputs by one runner selection; every DKX point carries both error bars; the caption states scope. Kill: none, this phase is mandatory; if DKX loses on time or memory the figure still ships, because a loss with error bars is publishable and a win without them is not.

Effort: 3–4 weeks, one person, mostly measurement and yancc onboarding. No new DKX solver code. Evidence tooling frozen at what exists.

**Three criteria this phase's campaign must satisfy, found the hard way.**

- **Audit the complete Legendre state.** The memory-saving structured route returns only low Legendre moments by default, and its zero-filled tail cannot certify the original equations. Request full recovery (`SolverOptions(keep_lowest=Nxi)`) for any original-equation audit, and include the recovery cost in the timing.
- **Bind benchmark child imports to the source the parent recorded**, including absolute dependency search paths. A relative `PYTHONPATH=src` is reinterpreted when the child's working directory changes and can select an unrelated editable installation; children must import the same package as the parent and record the source path they actually used.
- **Match radial coordinates, not only field arrays.** [yancc's DKES outputs](https://github.com/f0uriest/yancc/blob/33e1ce9b208f6d3209fdb55aeba8712e6d6a4223/yancc/solution.py) use `r = a_minor * rho`, while DKX's Beidler conversion uses `r_eff = sqrt(|Psi| / (pi B00)) * rho`. Transform each radial coefficient before comparing, or the cross-code numbers differ by a coordinate factor rather than by physics.

**Order of work from the 2026-09-13 review** (`docs/experiments/2026-09-13-preconditioner-cost-anatomy.md`, `…-sfincs-sparsify-threshold.md`). The NCSX baseline `(21,37,61,8)` completes in 164–187 s and 27–28 GiB on four office CPUs, but the peak is a transient of the dense band assembly (about 20 GiB of band-sized buffers that the exact linear map never needs) and the 70 s Krylov stage is 90% preconditioner apply through dense couplings with 9 nonzeros per row. The exact float64 information content of the preconditioner at this grid is 1.4 GiB, and at `(25,37,61,8)` 2–3 GiB. Therefore: do not launch memory-gated refinement points (theta25 "needs 56.1 GiB") or the baseline adjoint campaign against the dense-band route; run Phase 2 step 6 first, then resume the ladder (theta25, theta29, a pitch neighbour and a higher pitch order, one joint refinement) and the four baseline adjoints with `linear_observable_algebraic_error`. Do not infer error bars from an accepted residual.

**Cross-code admission criterion added by the same review.** SFINCS v3 inserts matrix entries through `sparsify.F90`, which drops every value with `|a_ij| <= 1e-12`. On decks with `sqrt(T_e m_i / (T_i m_e)) ≳ 30` that removes the ion→electron field-particle collision block; on the collaborator's HSX-like deck it accounts for the entire 12–19% bootstrap-current gap (DKX with the same threshold applied reproduces the SFINCS matrix to 6e-15 and its current to 2e-10). A SFINCS reference is admitted for parity on such decks only with the threshold disabled, or compared against DKX's opt-in threshold-compatibility mode, which exists for parity tests and is never a default. Report `E_* = (α Δ/2) Ĝ Φ̂' / (ι B̂_0 sqrt(T̂_i/m̂_i))` with every result; above `E_* ≈ 1/3` the `Nx`/`Nxi` ladder is mandatory for fluxes and current separately, and scan failures are plotted against `E_*`, not `(r, E_r)`.

**Positioning-figure scope.** Compare DKX, SFINCS v3 and yancc on the same equilibria at each code's converged resolution per observable, with both DKX error bars and the memory and time at those grids; state that the pitch discretizations differ. A DKX run at yancc's published NCSX grid `(43×65, Nxi 61–121)` needs 25–49 GiB of float64 factors even after step 6 and is a stretch item gated on the two-device split (Phase 2 step 7); it is reported with iteration counts if run in float32.

### Phase 2: why the Krylov route loses, and two bounded extensions (3 weeks, time-boxed)

**Correction to the first draft of this plan, and to a common assumption.** DKX already implements SFINCS's simplified-operator preconditioner, and it is the default on every deck that leaves the pitch-angle-scattering family: `build_coarse_preconditioner` (`coarse_precond.py:916`) mirrors the Fortran `preconditionerOptions` defaults (self-species, x-diagonal collisions; the Er and drift L±2 terms dropped), inverts that operator *exactly* with a batched block-Thomas factorization over (species, x), eliminates the constraint border exactly by a Schur complement, and is Phi1-aware. `multigrid.py` and `sparse_precond.py` are alternative inverses of the same simplified operator; the Krylov method is SOLVAX's GCROT with a recycle space; the adjoint solve is a cold-started GCROT on the transpose. So "use the exact PAS solve as the preconditioner" is not a proposal, it is the status quo, and the 7-of-23 record and six non-completions are *with* it. #189 was right to say measure first. What the numerics survey adds is a short list of specific, testable reasons the route still loses, and two extensions that follow from them.

**Step 1, one week: the diagnosis.** On the 23 Krylov-route decks and the six failures, log per solve: peak memory of the preconditioner bands and factors against the operator apply (five of the six failures were memory exhausted while `build_coarse_preconditioner` allocated dense (N_θN_ζ)² bands, 42.9 GB on `filteredW7XNetCDF_2species_magneticDrifts_noEr`; `docs/performance.rst`); GCROT iterations and restarts; the share of `A − M` carried by each dropped coupling; whether the preconditioner was rebuilt or reused at that point; and the true residual at every restart. Group the decks by which coupling dominates.

The attribution half is done and its result reorders what follows. `dropped_couplings` in `coarse_precond.py` splits `A − M` exactly by mechanism, matrix-free, and on every deck available the collision operator's dropped speed and species coupling carries essentially all of it, with the `E_r` terms at one to two per cent, robustly across three structurally different probes. The record is `docs/experiments/2026-09-07-dropped-coupling-attribution.md`. Two limits stand: no deck exercising tangential magnetic drifts was in that set, and an operator-norm attribution is not a convergence prediction, so the iteration counts and memory that decide whether a change pays are still to be measured.

**Step 2, one week: keep the collision operator's upper triangle in speed.** This is where the measured mass is, and the shape of it decides the method. The linearized Fokker–Planck operator is an integral operator in speed, so in this basis it is essentially upper triangular with its mass at the far corner: the strict lower triangle is four to five orders of magnitude below the whole matrix, the first superdiagonal carries about a tenth of a per cent, and the corner carries up to 92 per cent. Banding is therefore the wrong shape, and SFINCS's intermediate `preconditioner_x = 3` and `= 4` recover nothing measurable; `preconditioner_x = 2`, which keeps the upper triangle, is for this operator very nearly the full coupling and takes `||A − M||/||A||` from about 1.0 to 0.018 or below (`docs/experiments/2026-09-07-collision-speed-structure.md`).

Its cost is a back-substitution rather than a dense solve. With `M = D + U`, `D` the speed-diagonal blocks already factored and `U` strictly upper in `x`, the apply sweeps `x` downward as `y_x = D_x^{-1}(r_x − Σ_{x'>x} U_{x,x'} y_{x'})`, reusing the existing block-Thomas factors unchanged and adding no factorization.

Admission test: GCROT iterations and wall time against the current diagonal preconditioner on the full-FP decks at 2,804, 15,844 and 66,004 unknowns, CPU and A4000, byte-identical operators, idle machine, A/B/A/B, with the original-equation residual as the acceptance rather than the preconditioned norm. The apply becomes sequential over `Nx` where it is now one vmapped batch over `(species, x)`, so report the apply cost separately from the iteration count: this trades a much better preconditioner against `Nx` dependent steps and `Nx²/2` block matrix–vector products, and on GPU that sequencing may cost more than the iterations it saves.

Kill: less than a 1.5× reduction in iterations at the 15,844-unknown deck, or a net wall-time loss once the apply is charged, after one week. Record and stop.

Run and stopped on both devices. On an A4000 the exact triangle is three to five times slower with a 1.50 to 1.64 iteration gain, the same shape as CPU, so the accelerator does not rescue the sequential apply and the GPU caveat that motivated retesting is discharged (`docs/experiments/2026-09-07-gpu-answers.md`).

**Deferred, with a measured reason: the block-pentadiagonal structured solve.** The `E_r` xiDot/xDot and tangential-drift terms couple only |ΔL| ≤ 2, so extending the structured kernel from tridiagonal to pentadiagonal in `L` would make those decks exact again rather than merely preconditioned. Step 1 measured those terms at one to two per cent of `A − M` on every deck available, so the work addresses almost none of the dropped mass and is not the first thing to build. It reopens when a deck is measured in which the `E_r` or drift terms dominate, which the drift decks absent from that set may yet show, or when the collision experiment has been tried and stopped.

**Step 3, one week: qualify preconditioner reuse before implementing refresh.** The tiny-deck recycling experiment is stopped: it missed its iteration-gain threshold and did not improve CPU wall time. Holding the preconditioner fixed instead saved 2.4× on CPU and 2.6–2.8× on an A4000, on 1,178–3,930 unknowns; these are bounded experiment results, not a default-setting rule (`docs/experiments/2026-09-07-recycling-and-preconditioner-reuse.md`, `2026-09-07-gpu-answers.md`). A preconditioner built at `Er=0` eventually failed at `Er=80`, returning `converged=False` and original residual 0.35 against `1e-10`. Rejecting that result is necessary, but waiting 6,000 iterations is not bounded recovery.

The 73,444-unknown full-FP W7-X follow-up and its profiling are recorded in `docs/experiments/2026-09-12-reuse-admission.md`. Its grid is a scaling diagnostic, not a converged research grid. Separate transferring a recycle basis between solves (`recycle=None` disables it) from recycling inside GCROT (`recycle_dim=8` here; zero is rejected by DKX). Keep the initial-guess policy identical between arms. Do not attribute shared-device timings or compilation-cache loading to an algorithmic speedup.

The source audit also found a redundant principal-inverse application in SOLVAX's bordered preconditioner. [SOLVAX #101](https://github.com/uwplasma/SOLVAX/pull/101) reuses its already-computed transformed border, leaving the mathematical preconditioner unchanged. Use its pinned downstream qualification and post-change application costs in the experiment record before calibrating refresh. Do not infer root or optimization performance from field-sweep timing. The wider-root pilot in the record loses with a two-cycle reuse cap; keep that cap opt-in and count rejected trials before admitting a refresh threshold. An unstable sampled root is not the retained optimization branch.

Proceed in this order, within the existing step, and only after step 6 has cut the rebuild cost that any reuse policy is charged against (a policy can save at most one rebuild per point; at NCSX the rebuild is ~90 s on the dense route and a projected 20–30 s after step 6):

0. Add the tangent-predictor arm. `E_r` enters the operator linearly, so `d f / d E_r` is one extra solve with the **same** factors and recycle space (the one place recycling is valid), gives the warm start `f(E_r + ΔE_r) ≈ f + ΔE_r · df/dE_r` and the Newton slope `dJ_r/dE_r` for the bracketed root in `er.py`. Compare start residual and iterations against cold and previous-solution starts on the recorded W7-X 73,444-unknown sequence; alive at ≥ 30% fewer iterations. This targets the cold retries that made bounded reuse lose (#227), which a refresh policy does not.
1. Measure rebuilt versus lagged preconditioners on the **same accepted equation sequence** with per-species flux/flow comparisons. Retain five alternating-order repetitions, synchronized setup/solve times, cache scope, device/process occupancy and failures. A contaminated timing is excluded, not averaged into a speedup. Extend to a full-FP single-species case and the multispecies root workflow; report the mandatory independent cold final root check in both timing scopes. Close Phase 1's grid and observable error requirements before promoting these to publication points.
2. Only with a measured margin, add an **opt-in host refresh policy** to the existing `ErProblem`/`ErSolveState` path. Compare rebuild cost with measured extra solve work over the remaining horizon (§5.1), using a conservative bounded horizon for root searches. Iteration count alone is insufficient when per-iteration or cache costs change. Exercise the opt-in `find_ambipolar_er(..., max_restarts=..., reuse_max_restarts=...)` control: cap a reused solve's work; after failed original-equation or finite-current/flux admission, discard its candidate and retry once with rebuilt factors, then refuse on failure. A rejected trial never replaces the accepted physical state. Charge the pilot, refresh, retry and memory cost; do not silently invoke an unbounded solver-route escalation.
3. Replay forward, reverse and permuted sequences and an adversarial stale-factor case. Check every species independently, and retain the no-file-I/O interface. Keep timing/refresh decisions outside differentiation: gradients still solve the converged physical equations and their checked transpose, not a derivative of the cache policy. Physical admission belongs in DKX; changes to generic Krylov or factor algorithms belong in SOLVAX.

This separation of numerical-factor reuse and the initial guess follows [PETSc's preconditioner reuse](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/) and [nonzero-initial-guess controls](https://petsc.org/release/manualpages/KSP/KSPSetInitialGuessNonzero/). Neither control certifies the new equation or guarantees a time saving.

Admission: a sweep and a root on full-FP and multispecies cases, all original residuals within tolerance, cold observable agreement, bounded stale-state recovery, and a net wall-time gain with the policy charged. Kill default promotion if that gain is absent; preserve explicitly requested reuse where measured useful. Do not restart the killed triangular or recycling-only experiments to fill a missing admission row.

**Step 4, accuracy not speed: refinement precision on the exact route — run and closed.** The concern was right and the proposed remedy was not needed. Without refinement the structured route's forward error reaches 1.4e-7 at a conditioning of 3.3e6, which would make a 1e-10 agreement claim meaningless; with the single float64 sweep the route already performs it is 2.5e-14, bought for about 15 per cent of the solve. Those sweeps are ordinary float64 and they reach a reference refined with a 50-digit residual to 2.5e-14 after one sweep and 1.4e-15 after three, so a compensated double-double residual has no headroom to recover and is dropped (`docs/experiments/2026-09-07-refinement-precision.md`).

One sweep stays the default. The open thread is conditioning: every deck measurable this way reached 3.3e6, three orders below the 3e12 recorded for production pitch-angle decks, because the reference needs a dense operator. A single sweep appears to leave roughly `1e-20 × kappa` of forward error, which extrapolates to about 1e-8 at 3e12 and would matter, but that is an extrapolation and not evidence. Measuring it needs an iterative extended-precision residual against the matrix-free operator rather than a dense factorization, and is worth doing only if phase 1's observable error bars ever disagree with the residual.

**Step 5, memory-lean preconditioning — mostly already done; what is left needs the accelerator.** Step 1 did confirm memory as the binding loss, and most of this step was already implemented before it was written. The route choice is automatic by size (`_coarse_bands_fit`, `_coarse_factors_fit`), `DKX_COARSE_FACTOR_DTYPE=float32` exists opt-in, and `docs/performance.rst` records a completed campaign in which every deck the reusable route was built for completes.

Two parts of the step's premise do not survive contact with the code, and are withdrawn. There is no Ruiz equilibration in the solver path, only in `tools/benchmarks/operator_conditioning.py`, so "fp32 after Ruiz" cannot be done as written; the campaign shows `float32` working without it, and adding equilibration should be justified on its own evidence rather than smuggled in as a precondition. And `float32` is not free in general: on decks where the coarse preconditioner fits and is nearly exact it costs 40 to 76 times the iterations, because the factor error then dominates the preconditioner instead of disappearing into an approximation that is already larger (`docs/experiments/2026-09-07-float32-factor-scope.md`). It stays opt-in with float64 the default, which is what the code already does.

That admission has now been run on the accelerator, and it fails on the second half. All five decks build their coarse preconditioner within the 16 GB card once the factors are `float32`, at 8.94 to 11.28 GB, where float64 does not fit at all; the switch does exactly the job it was added for. The solve then asks for 7.96 GB more and dies inside the Krylov loop, whose working set XLA cannot rematerialize below 8.15 GB, for about 17 GB against a 16 GB device. Neither the checkpointed coarse route nor a Krylov workspace cut to `restart = 12` closes the gap (`docs/experiments/2026-09-07-step5-gpu-memory.md`).

So the preconditioner memory that motivated this step is solved, and the obstacle has moved to the solve's working set at 1.5 to 1.9 million unknowns. Closing it needs a larger accelerator or one system distributed across both A4000s rather than replicated, which is the state-decomposition work already under deferred; it is no longer a preconditioner problem.

**Step 6, two weeks: the exact, structure-preserving coarse preconditioner.** Same simplified operator, pins, floor, border elimination and forward/transpose contract as today; only storage and elimination bookkeeping change, so certification is map equality and identical iteration counts, never wall-clock (`docs/experiments/2026-09-13-preconditioner-cost-anatomy.md`). In order, each its own PR under the size cap, SOLVAX first where it owns the kernel:

1. Release SOLVAX 0.21.0 (#100–#104) and require `solvax >= 0.21.0`, so installed users and CI stop paying two coarse applies per GCROT iteration; re-run the W7-X sweep pair of `2026-09-12-reuse-admission.md` from the wheel.
2. Build the transposed preconditioner lazily (only for differentiable solves or on first transpose call); replace the discarded zero probe by `block_until_ready` on the factors; jit or close-form `_materialize_borders`; size `_coarse_bands_fit`/`_coarse_factors_fit` against device memory on accelerators; skip zero drift arrays on drift-free decks. Exit: NCSX baseline unchanged at 53 iterations with the same residual history, 10–20 s less build and no 6.5 GiB setup transient.
3. Generate `(L_k, D_k, U_k)` inside the factorization scan from the existing pinned generator, storing only the Schur LU and pivots; factor, store and apply per `Nxi_for_x` group (six chains at NCSX; identity on the padded right-hand sides, exact because the padded couplings are masked to zero). Exit: peak RSS at the NCSX baseline ≤ 8 GiB (projection 4–6).
4. SOLVAX: accept operator-valued couplings `(k, z) -> U_k z` and their transposes beside dense bands, and thread the right-hand side through a scan instead of whole-band slices (the regenerated-solve pattern), keeping `linear_transpose`/VJP working. DKX supplies the couplings as `x·c_L·S_s + x·c'_L·diag(mirror)` stencils, and the drift `(TZ,TZ)` terms as stencils too. Exit: apply agreement 1e-12 with the dense route on PAS, full-FP, improved-Sugama, `Phi1`-in-collision and magnetic-drift decks; GCROT at the NCSX baseline ≤ 30 s on four CPUs.
5. Device-appropriate Schur storage: LU on CPU; explicit `Δ_k^{-1}` on GPU, where the prototype applied 22× faster per block at 1.2× faster factorization, chosen by a per-device microbenchmark rather than by assumption, with a collisionless `l = 0` worst-case deck in the tests.

Exit for the step: the NCSX baseline in float64 on one A4000 and in ≤ 8 GiB / ≤ 70 s on four CPUs with identical observables; the W7-X 73,444-unknown sweep keeps its iteration counts; then re-test the speed triangle (`retain_speed_triangle=True`, killed in step 2 on apply cost alone) and float64 versus float32 on the five 1.5–1.9M decks. Kill per item: any iteration-count change or map difference above 1e-12.

**Step 7, in parallel, admission tests before any scale-out build (each ≤ 1 day).** (a) CPU batch parallelism: `lu_factor` on a `(B·L, 777, 777)` batch, default versus `--xla_force_host_platform_device_count = 4, 8` with BLAS threads capped, adopt at ≥ 1.5× on equal cores (open-source jaxlib runs the LAPACK batch serially). (b) GPU batched LU: the default `getrfBatched` path (taken for `n = 777` at batch ≥ 7) versus `lax.map(..., batch_size=6)` on an idle A4000. (c) Two-device split along `x` or species with `shard_map`: the preconditioner is block-diagonal in `(s, x)` and needs no communication, the matvec one all-gather per iteration (≤ 15 MB at 1.9M unknowns); emulate two CPU devices and require bitwise-identical iterations, count collectives in the jaxpr, then measure per-device peak on one 1.5–1.9M deck in float64. (d) An XLA buffer-assignment dump on a reduced HSX deck to attribute the ~6.8 GB of unexplained Krylov-loop working set, expected to be closed by step 6.4; host-stepped Arnoldi with donated buffers only if it is not. Exit: one of the five large decks completes in float64 on 2×A4000 or 1×A4000 with original residual and observables screened.

**Research bets, kill tests only until alive.** Nested-dissection sparse LU per `(s, x)`: SuperLU/COLAMD is already negative at `m ≈ 800` (2.4× the dense fill, 16× the time); one METIS-ordered fill count decides it, kill above 0.5× the dense LU. Angular two-grid with step 6's exact block-Thomas as the coarse solve on `(Nθ/2, Nζ/2)` and damped line block-Jacobi smoothing with the `L` chain kept, under flexible GCROT: alive at ≤ 4× iterations across three collisionalities, kill above 8× or divergence at the lowest ν. Block low-rank compression of `Δ_L` with field-aligned (`α = θ − ιζ`) clusters: an offline rank study of the off-diagonal blocks at `L ≈ 2` and `L ≈ Nxi/2` first. First-order `E_r` updates of the factors (Tebbens–Tůma) only if the tangent predictor in step 3 fails.

**Explicitly not in this phase.** Half precision anywhere in a factor is inadmissible for these operators. The κ ≈ 1e18 Er-xDot decks are numerically singular in double (κu ≈ 1e2); no preconditioner or refinement rescues them, and the fix is formulation (constraint-row and source-column scaling per SFINCS's block structure), diagnosed with a double-double factorization on a small deck. Semi-coarsened multigrid with the exact-in-L solve as plane smoother is built only if step 1 shows iterations growing with resolution; DKX's `multigrid.py` is the starting point. cuDSS through an XLA FFI call (the spineax pattern) is the SFINCS-style general LU fallback on GPU; it is proprietary and NVIDIA-only, and it is deferred until a deck needs it.

Deliverable if any step passes: Figure 2 of the methods paper, iterations and time versus size for the ablation (none / coarse exact / coarse with banded speed coupling / multigrid / fixed-M recycling), with yancc's published multigrid numbers as the external bar. Deliverable if all fail: the same figure with the losses, and one-page experiment records under `docs/experiments/`.

### Phase 2b: the production solver program (4–6 weeks, each step time-boxed)

**Why this phase exists.** Phase 2 asked why the Krylov route loses and tested two extensions. The 2026-09-19 cross-code study (`docs/experiments/2026-09-19-sfincs-on-the-gap-deck.md`) moved the question. On the collaborator's HSX-like deck at `Nxi = 120`, `Nx = 16`, 633,604 unknowns, SFINCS fails by every route on a 36 GiB host: its default iteration stagnates at `‖r‖/‖b‖ = 0.9955` after 66 iterations, and its direct factorization and its `preconditioner_x = 2` are both killed for memory. DKX's `coarse` route inverts the same simplified operator and fails the same way. The limit is the *method both codes share*, the simplified preconditioner, so parity with SFINCS is no longer the target on this deck class. The target is a production solve that is fast, certified and differentiable where SFINCS has none.

Three results make that reachable, and fix the order below.
- **The operator can be assembled exactly**, from products with whole groups of columns that share no row: 8,800 products for 633,600 unknowns, about 1.2 min, entrywise identical to sampling every column on the decks where that can be checked (`dkx.assembly`, `solvax.compression`).
- **The direct route then works**: on the collaborator grid it reaches `1.7e-13` after one refinement step. SuperLU needs 801 s and 389× fill where MUMPS needs 40 s on the same deck, so the factorization, not the assembly, is what is missing.
- **The speed coupling is where the iterations are**, and retaining it pays once they run away: the exact triangle costs 3.7× less than it did (`2026-09-18-speed-triangle-back-substitution.md`) and wins 1.85× end to end at `Nx = 16`, while losing below it.

**The method map.** One route per structure, chosen by what the operator is, not by a user flag. `method="auto"` owns the choice; every row names the certificate that admits its answer.

| Case | Structure the route exploits | Route | Certificate |
| --- | --- | --- | --- |
| Pitch-angle scattering; `RHSMode` 2 and 3; monoenergetic | Speeds uncouple, so each `(species, x)` is one block-tridiagonal chain in `l` | Structured direct: one factorization, every right-hand side and the adjoint share it | Original residual; factors reused, never rebuilt, across right-hand sides |
| Fokker–Planck, iterations bounded | Simplified operator is a good approximate inverse | `coarse` + GCROT (default) | Original residual |
| Fokker–Planck, iterations run away | The speed resolution, not the Legendre one and not the dropped collision coupling (`2026-09-19-nx-drives-the-iteration-growth.md`) | The triangle after a stall (#250); step 2 below tested the speed grid's conditioning as a diagonal scaling and killed it (`2026-09-20-the-balanced-solve.md`) | Original residual; same answer as `coarse` where both converge |
| Fokker–Planck, many solves of one operator (gradients, transport matrices, root iterations), up to a few hundred thousand unknowns | The matrix is sparse and can be assembled | Direct, through the assembly (step 1) | Backward error after refinement; assembly checked against the operator |
| Ambipolar root, `Phi1` Newton, profile and `E_r` scans, optimization line searches | A sequence of nearby operators | Factor once, precondition the neighbours with the frozen factors (§5.1) | Each accepted state passes its own original-equation residual |
| Impurities and electrons | Collisional coupling between species is one-way to leading order in the mass ratio | Species block substitution (step 3) | Same answer as the coupled solve on a deck where both run |
| Tokamak | `Nzeta = 1`: angular blocks are tiny | Structured direct at any resolution | As the first row |
| Large `Ntheta × Nzeta` | Dense angular blocks are the memory wall, `O((Ntheta Nzeta)^2)` per row | Sparse angular elimination (`sparse`); batched dense factors on an accelerator | Same map as `coarse` |

**Step 1, one week: a scaled supernodal factorization behind the assembly.** The assembly removed the reason the direct route was confined to `max_dense_size`; the factorization is what remains. Put a sparse direct backend in SOLVAX behind the interface `SpluFactorization` already defines: ordering on `A + Aᵀ`, scaling, iterative refinement to a backward-error target, transposed solves from the same factors, and a memory estimate before the factorization rather than a kill during it. The 2026-09-19 measurements settle two of the three candidates. A fill-reducing ordering is not it: on the assembled matrix SuperLU fills 389× in 801 s with COLAMD and 737× in 4,507 s with `MMD_AT_PLUS_A`. What closes the gap is a supernodal backend *and* a scaling: MKL PARDISO factors the same matrix in 201 s but answers with a relative residual of 9.4e-2, because static pivoting perturbs the pivots it cannot use; Ruiz equilibration first takes it to 103 s and 1.2e-10 refined, against SuperLU's 2,732 s and 19.1 GiB and MUMPS's 40 s and 3.8 GiB. So: land the equilibration in SOLVAX, choose the backend at run time — PARDISO is x86-64 only and the arm64 hosts must fall back to SuperLU — and keep an optional MUMPS binding as the remaining candidate for the last factor of 2.6. In DKX, replace the size guard of the direct route with a memory guard fed by that estimate, and let `method="auto"` choose it when the estimate fits and an iteration has stalled.
Admission: the 66,004-unknown collaborator deck in at most 2× MUMPS's 40 s and 3.8 GiB, residual at most `1e-12` after refinement, and the transposed solve at the cost of a forward one. Equilibrated PARDISO stands at 2.6× the time, 1.3× the memory and `1.2e-10`, so the bar is not met yet; the open moves are more refinement steps, `MKL_NUM_THREADS` above the four pinned cores measured, and the MUMPS binding.

**What this step does not buy, measured before it is built out.** The route's cost grows as the 2.8 power of the unknowns in time and the 1.6 power in memory: 66,004 unknowns factor in 103 s and 5.0 GiB, 158,404 in 1,107 s and 20.4 GiB. The gap deck's 633,604 would be of order fourteen hours and 190 GiB against the office host's 62 GiB, so **a general sparse direct factorization is for small and medium decks and does not answer the gap deck**. Its value there is exactness, a residual of `1.3e-14` end to end, and adjoints and extra right-hand sides at the price of a solve rather than a factorization — which is what steps 4 and 6 need. The gap deck belongs to step 2. Then the 633,604-unknown gap deck within 48 GiB on the office host, which gives point A its first converged `Nx = 16` value.
Kill: no candidate fits the gap deck in 48 GiB. Record the fill against `Nx` and `Nxi`, keep the route for decks where it fits, and rely on step 2 above that size.

**Step 2, one week: the speed grid's conditioning, not the collision coupling.** The step this replaces proposed keeping the whole collision coupling as a second factor, `M⁻¹ = (D_c + C_off)⁻¹ D_c (S + D_c)⁻¹`, on the premise that the dropped collision coupling is what the Krylov route is missing. `dropped_couplings` supports that premise — the Fokker–Planck share of `A − M` is 0.9999997 on the gap deck — and the premise is false. Three measurements retire it (`2026-09-19-collision-coupling-is-cross-species.md`, `2026-09-19-nx-drives-the-iteration-growth.md`):

- The factored splitting itself is unusable. Its error is `S D_c⁻¹ C_off`, and `D_c⁻¹ C_off` is not small: the collision block's off-diagonal exceeds its diagonal by up to 7.9e4, so `||M⁻¹ A v − v||/||v||` is 5.1e5 against 1.5 for `coarse`, across `nu_n` from 8.5e-3 to 1e2.
- Retaining the coupling *exactly* converges worse. On a two-species deck the cross-species block carries 99.9% of the collision norm, and keeping the speed triangle over every species pair rather than each species' own block takes `||A − M||/||A||` from 1.0 to 4.3e-4 at no cost in structure — and costs 17% *more* GCROT iterations, 6,786 against 5,799 at `(Nxi, Nx) = (40, 16)`. `D + U` is inverted exactly and that inverse is ill-conditioned. A closer `M` is not a better preconditioner here.
- The resolution that costs is `Nx`, not `Nxi`. At `Nx = 10`, doubling `Nxi` from 20 to 40 takes 174 iterations to 157; at `Nxi = 20`, raising `Nx` from 10 to 16 takes 174 to 2,788. The cost is tolerance-independent — `Nx = 16` needs 2,382, 2,589 and 2,788 iterations at `1e-8`, `1e-9` and `1e-10` — so it is a convergence rate, not an accuracy floor.

What `Nx` brings is conditioning, and that conditioning is a scaling artifact: `cond(ddx)` grows about a decade per speed grid point, 2.5e6 at `Nx = 10` to 2.5e12 at `Nx = 16`, and Ruiz equilibration takes every one of those to order ten.

Where the factors lose that conditioning is measured, and it is the elimination rather than the scaling. Against the pinned chain its own generators produce, the block-Thomas factors lose accuracy one speed at a time and worst for the light species, whose streaming coefficients carry the `sqrt(T/m)` its collision diagonal does not: at `Nx = 10` the backward error runs 2.1e-16 on the lowest-speed ion chain to 7.9e-8 on the highest-speed electron one, and at `Nx = 16` the top seven electron chains run 2.2e-10 to 3.2e-7, about a factor of three per speed point added. Assembling those chains separates cause from symptom. A dense LU of the worst chain, same matrix and same right-hand side, returns 1.4e-11 where block-Thomas returns 2.6e-7 — a growth factor of about 3e9, and a ratio of 1.9e4 where the chain is ill-conditioned (`cond` 1.79e6) against 0.6 where it is not (`cond` 6.5e2), which is what an elimination that does not pivot looks like. Ruiz equilibration of that chain moves its backward error from 1.4e-11 to 1.2e-11, i.e. nothing, although it does remove the chain's 9.9e6 entry spread. So if the factors' accuracy were the lever, a stable elimination would be the move and a scaling would not.
And that accuracy is not what costs the iterations either, which the same probe settles by making it far worse on purpose: `DKX_COARSE_FACTOR_DTYPE=float32` takes the worst chain's backward error from 2.6e-7 to 793, 9.4 decades, and takes GCROT at `(Nxi, Nx) = (20, 16)` from 2,788 iterations to 10,391 — a factor of 3.7. Over the interval that matters, `Nx = 10` to `16`, the float64 backward error moves 0.6 of a decade, from 7.9e-8 to 3.2e-7, so it cannot produce the 16× in iterations; recovering it entirely, 4.3 decades down to the dense LU's 1.4e-11, would by that measured sensitivity buy well under 2×. The `sparse` route agrees: a fill-reducing elimination of the same operator returns *exactly* 2,788 iterations at `(20, 16)`, with the two maps differing by 5.2e-9. So the factorization is exonerated and the growth is in the spectrum of `A M⁻¹`. That measurement is now done, on a grid small enough for a dense eigendecomposition (`Ntheta = 5`, `Nzeta = 5`, `Nxi = 8`, where the growth survives at 113 iterations against 340). The eigenvalues barely move between `Nx = 10` and `Nx = 16` — clustering within 0.5 of 1 goes 96.7% to 94.3% and the modulus ratio doubles — while the conditioning of the eigenvector basis goes 1.95e18 to 1.11e22, a factor of 5,700. The preconditioned operator is strongly non-normal and becomes far more so with `Nx`, which is why the operator-norm attribution that opened this program pointed the wrong way: an operator with an eigenvector conditioning of `1e22` is not governed by its eigenvalues.

And that collapse is a diagonal scaling. LAPACK balancing, a diagonal similarity that leaves the eigenvalues untouched, takes the two numbers to 9.82e10 and 1.57e11 — seven to eleven orders removed and the `Nx` dependence flattened from 5,700× to 1.6×. Those two figures come from `matrix_balance` at its `permute=True` default, and a permutation reorders rows, which is not available matrix-free; the diagonal a solve can actually use gives 2.07e11 and 7.49e11, a flattening to 3.6×. This does not contradict the earlier point that a scaling cannot move the spectrum; it cannot, and it does not. It moves the basis that spectrum is expressed in, and that is the quantity convergence depends on here.

**The balanced solve was built and it is killed on its own criterion** (`docs/experiments/2026-09-20-the-balanced-solve.md`). The admission test was at most half of `coarse`'s 2,788 iterations at `(Nxi, Nx) = (20, 16)` with `FSABjHat` unchanged to 1.6e-9 and no loss at `Nx = 10`; the kill was that the eigenvector conditioning falls as measured while the iterations do not move by 1.5×. The kill fired on the small assembled grid that step A of the work reserves for exactly this, `Ntheta = 5`, `Nzeta = 5`, `Nxi = 8`, where the growth survives at 113 iterations against 340, so the production grid was never reached.

Solving `(D⁻¹ A M⁻¹ D) z = D⁻¹ b` with `x = M⁻¹ D z` is verified against the assembled balanced matrix to 7.0e-10 and the recovered `x` does satisfy the original equation to 9.9e-11, so the algebra is not the issue. Against a matched acceptance on the original equation, one cycle and no restart, balancing costs 120 iterations against 115 at `Nx = 10` and reaches 220 at `Nx = 16` where the unbalanced arm stagnates at 1.9e-10 — a lower attainable floor, worth about 2× in residual and nothing in rate. In the restarted configuration it is far worse, 2,780 against 113 and 5,787 against 340, because its stopping test lives in the scaled norm and lands 27× to 51× above the bar. The residual curves the kill criterion names as the next measurement are taken there rather than deferred: each arm leads by two to three orders in its own norm, trails by two to three in the other, and the two meet at the same iterate by 240 iterations. **A diagonal similarity that removes 1.5e10 of eigenvector conditioning moves the iteration count by at most 1.05×, so that conditioning is not the quantity this solve's convergence depends on.** No matrix-free estimator of the balancing diagonal was built, which is what a negative step A is for.

Two attributions are withdrawn. That the eigenvector conditioning of `A M⁻¹` is the quantity the `Nx` growth acts through — true of the matrix, false of the solve. And that the offending directions are the high-speed rows of the light species: the exact balancing diagonal is a smooth Maxwellian-like function of speed alone, `d ≈ x^-1.7 exp(−0.6 x²)`, with no angular or Legendre structure, and it corrects the *low*-speed rows of the *heavy* species hardest. The row norms of `A M⁻¹` fall ten orders across the speed grid while its column norms rise ten orders, worst where the collision frequency diverges; that diagonal is measured and kept for any future scaling of this operator.

The `Nx` growth is therefore unexplained again, with the dropped collision coupling, the tolerance, chain equilibration, the factorization's accuracy, a second exact elimination and now the eigenvector conditioning all measured and all ruled out. The next candidate worth a probe is the transient rather than the asymptotic behaviour — the field of values or a pseudospectral radius of `A M⁻¹`, which bound GMRES where eigenvalues and eigenvector conditioning do not. The gap deck's point A returns to the queue for the routes that do not depend on a scaling.

**Step 3, three days: species by mass ratio.** Ion–electron collisions act on the electrons at order one and on the ions at order `√(m_e/m_i)`; the same holds for a light bulk against a heavy impurity. Order species by mass and keep the block-lower coupling exactly by substitution over species, reusing per-species factors, as the speed triangle does over `x`. SFINCS's `1d-12` threshold removed exactly this small block and moved the current 12–19% (`2026-09-13-sfincs-sparsify-threshold.md`), so it is small but not negligible: it stays in the operator and is dropped only from the preconditioner's upper block.
**Measured against, before it is built.** Step 2's cross-species arm is this step's mechanism taken to its limit: the cross-species coupling is strictly upper in speed as well, so the existing back-substitution keeps it exactly, over every species pair at once, for one more contraction index. That was built and it converges worse — 6,786 GCROT iterations against 5,799 for the self-species block alone, at `(Nxi, Nx) = (40, 16)` on the HSX-like deck (`2026-09-19-collision-coupling-is-cross-species.md`). Keeping a block-lower species coupling *exactly* therefore has a measured counter-example on the two-species case this step names, and the step stands only if a damped or truncated retention does what the exact one does not.
Admission: on the two-species HSX-like deck and on a three-species impurity deck, fewer iterations than the self-species triangle's 5,799 at `(40, 16)`, with an apply no more than 1.2× `coarse`. Kill: no retention beats the self-species triangle on either deck, exact or damped. Then the species coupling belongs in the operator and not in the preconditioner, and this step closes.

**Step 4, one week: one factorization, many solves.** Production use is never one solve. A transport matrix is three right-hand sides of one operator; a gradient is one transposed solve; an ambipolar root is five to ten operators that differ in `E_r` alone; a `Phi1` Newton iteration and an optimizer's line search are sequences of neighbours. Implement the reuse contract of §5.1 on the routes above: right-hand sides and adjoints share factors exactly, and neighbours are preconditioned by factors frozen at the first, refreshed when the iteration count passes a bound.
Admission: an ambipolar root on the W7-X deck in at most 1.5× the wall time of its first solve; a gradient in at most 1.3× the primal on every route; a transport matrix in at most 1.2× one right-hand side on the structured direct route. Kill per item, independently.

**Step 5, one week: resolution as a certified output.** A production answer carries its discretization error. Drive `Nxi`, `Nx`, `Ntheta` and `Nzeta` from the ladder and the adjoint-weighted residual already implemented, per observable, and let the `Nxi_for_x` ramp follow the measured Legendre tail rather than a fixed rule. Where a ladder is refused, as the speed ladder is on the HSX-like deck, the run reports that and quotes no value.
Admission: fluxes and bootstrap current with stated bars of at most 1% on NCSX and W7-X, and the refusal reproduced on the HSX-like deck. Kill: none; a value without a bar is not a production value.

**Step 6, two weeks, entry requires steps 1 and 4: fidelity tiers for optimization.** An optimizer needs hundreds of evaluations with gradients, and only some need the full operator.
- **Tier 0, proxies:** effective ripple and the analytic regime limits, for screening.
- **Tier 1, monoenergetic:** `RHSMode 3` coefficients on a `(ν*, E_r*)` grid, which is the pitch-angle-scattering row of the map: exact, cheap, differentiable, and free of every Krylov failure in this plan. Fluxes and current follow by energy convolution with a momentum-correction closure, whose error against Tier 2 is the closure accuracy map of Phase 3.
- **Tier 2, full Fokker–Planck:** the certified solve of steps 1–5, with gradients, used to correct Tier 1 inside a trust region and to certify every accepted design.
Admission: on one configuration, a Tier 1 gradient that agrees in sign and to 20% with Tier 2 for the bootstrap current and the ion heat flux, and an optimization that reaches its Tier 2-certified optimum with at most one Tier 2 solve per ten Tier 1 evaluations. Kill: Tier 1 gradients disagree in sign on more than one in five parameters; optimize on Tier 2 alone and record the cost.

**Verification that does not depend on another code.** Analytic limits pin each regime independently: the tokamak banana-regime bootstrap coefficients, the `1/ν` scaling with the effective ripple, the plateau and Pfirsch–Schlüter limits, and the momentum-conservation and ambipolarity identities, which hold to the solver tolerance at every resolution. Each production deck family carries one. SFINCS remains the cross-code reference where it converges, with its sparsification threshold disabled.

**Benchmarks.** Every step reports against the same matrix, on the office host, idle, A/B/A/B, cold and warm apart, with absolute times beside every ratio: tokamak (pitch-angle and Fokker–Planck); NCSX one species `(25, 37, 61, 8)`; W7-X two species; the HSX-like deck at points A and B for `Nx` 10, 13 and 16; one three-species impurity deck. Columns: wall time, peak RSS, iterations, final original residual, observables with bars, and the cost of one gradient. SFINCS columns where it runs, which requires rebuilding its toolchain on that host.

**What this phase does not do.** It does not pursue a drop-tolerance incomplete factorization, which diverged on the collaborator grid at 9.5× fill; nor a change of fill-reducing ordering for the *simplified* subsystems, where the default is already best; nor an earlier stall trigger, which costs four times the memory (`2026-09-18-speed-triangle-back-substitution.md`). Each is recorded with its measurement.

### Phase 2c: the differentiable optimization chain, and the case DKX has to be able to make (6–8 weeks, runs beside Phase 3)

Phase 2b asked whether DKX can solve one hard deck. This phase asks the two questions that decide whether anyone uses it: can a stellarator optimizer put a *kinetic* neoclassical objective in its loss function and get a gradient, and is DKX demonstrably the code that still runs when the alternatives do not.

**The chain already exists in pieces and has never been closed.** [VMEX](https://github.com/uwplasma/vmex) solves the equilibrium and supplies exact Jacobians and a reverse adjoint; [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax) is a JAX-native Boozer transform of a VMEC `wout` with a differentiable API; DKX reads Boozer geometry and returns the bootstrap current, the transport matrix and the monoenergetic coefficients, differentiably. The missing link is that nothing has ever carried a gradient from a DKX observable back to a VMEX boundary coefficient.

What exists today stops short of it. `examples/optimization/QA_optimization_bootstrap.py` in VMEX optimizes a self-consistent bootstrap current, but the current comes from the **analytic Redl fit** inside a Picard loop, not from a kinetic solve. That is the right example to mirror and the right thing to replace.

**Step 1, two weeks: one VMEX example whose loss is a DKX observable.** Mirror `QA_optimization.py` — same seed deck, same staged mode ladder, same `least_squares` driver, same residual-and-Jacobian structure so rows stay inspectable — and add one residual row that is a DKX quantity on a VMEX equilibrium, through `booz_xform_jax`. Take the bootstrap current first, because it is the observable this program has measured most and the one the Redl example already brackets: a Redl-fit row and a DKX row on the same configuration are directly comparable, and their difference is a result whether it is large or small.

Then the thermal transport coefficients, `Le1` and `Li1` in the notation of the neoclassical-optimization paper the owner referenced. **Pin the definition and the citation before implementing**: which normalization, which radial coordinate, which collisionality regime, and whether they are the monoenergetic coefficients convolved in energy or the thermal ones directly. DKX's `RHSMode 2` transport matrix and `RHSMode 3` monoenergetic coefficients both reach them, and the wrong choice is a silently different objective rather than a failure.

Admission: a finite-difference check of `d(observable)/d(boundary coefficient)` agreeing with the adjoint to 1% on at least three coefficients spanning two mode numbers, and one optimization that moves the objective and ends at an equilibrium VMEX still converges. Kill: the gradient is unusable and the adjoint cannot be made to agree with finite differences; then record which link in the chain loses it — VMEX, the Boozer transform, or DKX — which is itself the deliverable.

**Cost is the gate here, not correctness.** An optimizer evaluates hundreds of times. Today `jax.grad` through DKX costs 2.0–2.6× its primal on every differentiable route and the ambipolar root costs 9.72× its first solve, against gates of 1.3× and 1.5× (`2026-09-20-one-factorization-many-solves.md`). Factor reuse cannot close either: reverse mode re-executes the forward pass, and the adjoint already runs on the primal's factors. So this step needs the Tier 1 monoenergetic route of step 6 below, or a closure, not a faster linear solve. Decide that before building the example, not after it is too slow to run.

**Step 2, continuous: make DKX fast and small enough to be called in a loop.** The assembly costs 4,800 operator applications on the 633,604-unknown gap deck, down from 8,800 (#259), and the structural bound for that grouping is now the angular lattice rather than the row density. The 66,004-unknown collaborator deck factors in 846 s to a residual of `1.3e-14`, where MUMPS takes about 40 s at 3.8 GiB and SuperLU 2,732 s at 19.1 GiB. Factors are returned and reused, so three right-hand sides cost one factorization and an adjoint costs 0.15 of a primal.

The open items are named and ordered: the run-time backend choice, so a host with MUMPS uses it and an arm64 host falls back (SOLVAX 0.25.0 ships the adapter; DKX selects it explicitly and refuses rather than substituting); a memory *estimate* replacing the `max_dense_size` guard, so the route is chosen by what will fit rather than by a size constant; retaining only active rows in the reusable coarse factors; and the next factor available in the assembly, which is the Legendre direction, where columns at one angular point with different species and speed need only `|dl| > 2` where same-species pairs need `|dl| > 4`, taking the slots from 24 to 40. Report peak resident memory beside every timing; a route that fits is worth more here than a route that is fast on a host nobody has.

**Step 3, three weeks: the validation and benchmark matrix, across the cases production actually uses.** The matrix is the deliverable, not any single point in it. Every cell carries the original-equation residual, the observables with their resolution bars, peak memory, wall time and the host, and a cell that was stopped by a guard is recorded as stopped and never as a value.

The axes, and why each is there:

- **RHS mode.** `RHSMode 1` (a single drive), `2` (the transport matrix, three right-hand sides of one operator) and `3` (monoenergetic). These exercise different routes: `3` is the pitch-angle-scattering row where speeds uncouple and the structured direct solve is exact and cheap, and it is the Tier 1 candidate for optimization.
- **Collision operator.** Pitch-angle scattering, full linearized Fokker–Planck, improved Sugama. The Fokker–Planck decks are where the Krylov route's iteration count is unexplained.
- **Geometry.** Analytic, VMEC and Boozer, and within them a tokamak (`Nzeta = 1`, where the angular blocks are tiny and the structured route is exact at any resolution), a quasi-axisymmetric case, W7-X and the HSX-like deck. Geometry changes which term dominates and therefore which route wins.
- **Ambipolar root.** Prescribed `Er` against the solved root, including the ion and electron root branches and a case with multiple roots, since the root is differentiated by the implicit function theorem and each accepted state must pass its own original-equation residual.
- **`Phi1`.** Off, and on as the Newton iteration whose Jacobian the coarse preconditioner is `Phi1`-aware about.
- **Species.** One species, ion-electron, and a three-species impurity case, where the collisional coupling is one-way to leading order in the mass ratio and SFINCS's `1d-12` sparsify threshold was measured to move the current by 12–19%.

Admission: every cell either produces observables with a stated bar or a recorded refusal; no cell produces a number without an original-equation residual behind it.

**Step 4, two weeks: the comparison that is the actual claim.** "DKX converges where MONKES, SFINCS and yancc do not" is a strong claim and needs the discipline of one. It is true today on exactly one deck and by one route: on the HSX-like point A, `Nxi = 120`, `Nx = 16`, 633,604 unknowns, SFINCS v3 fails on a 36 GiB laptop by **every** route — the direct solve killed for memory at 22.5 minutes, the default iteration stagnant at `||r||/||b|| = 0.9955` after 66 iterations, `preconditioner_x = 2` killed at 30.6 minutes — while DKX assembles that operator exactly in 4,800 products (`2026-09-19-sfincs-on-the-gap-deck.md`). That is one point, on one host, against one code.

To make the claim properly, each comparison needs the same four things: the same physics, on the same geometry, at the same resolution, with the same convergence criterion applied to the original equation. Against **SFINCS** the model is the same, so the comparison is honest and the only care needed is the `1d-12` sparsify threshold, which changes the answer and must be disabled or reported. Against **MONKES** and **NTX**, which are monoenergetic, the comparison is `RHSMode 3` only, and comparing anything else is a category error. Against **yancc**, which is differentiable GPU full-DKE without `Phi1` and without the ambipolar root, the comparison is the cases it supports, and its absent features are a scope statement rather than a failure on its part. Where another code *cannot run at all*, report the resource it exhausted and the limit it hit, not a ratio against infinity.

Admission: at least three decks where DKX completes and a named alternative does not, each with the failure mode of the alternative recorded; and at least three where both complete and agree on the observables within their bars, because a code that only runs where others fail and never agrees where they succeed has not been validated. Kill: DKX and the alternative disagree outside their bars on a case both complete, and the disagreement is not resolved. That is then the most important open result in the project and it outranks everything else in this phase.

### Phase 3: verified derivatives, the closure accuracy map, and one real optimization (5–6 weeks)

**Why this and not more speed.** Derivatives are the claim yancc makes without evidence and NTX proves only for the monoenergetic problem (14× over finite differences at 32 parameters, agreement ~2e-14). DKX's #184 and #188 routed the ambipolar-root and profile derivatives through the differentiable solver with original-equation admission; what does not exist is the *published* check on a real configuration, and the accuracy map that design teams have asked for in print.

**Figure 3.** For one QI configuration (CIEMAT-QI4X or a Goodman-type QI) and one QA with large bootstrap current (Helios-like), at a W7-X-like surface: `dJ_bs/dp_k` and `dE_r/dp_k` for profile parameters and for a handful of boundary Fourier coefficients, from `jax.grad` through the ambipolar root by the implicit function theorem, against central finite differences over a step window and a Taylor-remainder slope near 2; cost ratio AD/FD versus parameter count; cold versus warm agreement. Where Paul et al. ([2019](https://arxiv.org/abs/1904.06430)) published sensitivities for the same quantities, overlay them.

**Figure 4 (the accuracy map).** Bootstrap current from Redl/Sauter via the quasisymmetry isomorphism (what DESC and SIMSOPT use; DESC's own tutorial states it does not apply to non-quasisymmetric fields), from PENTA-style momentum-corrected monoenergetic coefficients, and from DKX full-operator multispecies kinetics, across ν* and E_r on the same three or four real designs (Infinity Two, Helios, Stellaris, W7-X). Saxena et al. ([2025](https://arxiv.org/abs/2507.05166)) frame "how wrong is the analytic closure off-symmetry" as the open question; Infinity Two's design paper iterates SFINCS with VMEC by hand; Stellaris is QI and needs off-Redl verification. This figure is the physics letter, and it uses nothing DKX does not already have except the phase-1 error bars.

Verification design constraint from [Albert et al. 2024](https://arxiv.org/abs/2407.21599): at E_r = 0 the 1/ν off-set current does not converge and oscillates in log ν*; with finite E_r it decays as ν*^(3/5). Every low-collisionality bootstrap ladder in this programme is therefore run at finite E_r, and the ν*^(3/5) decay is itself a code-independent target (Figure 4 inset).

**Coordinate choices for the optimization deliverable** (decided; the default is the first row):

A coordinate change can simplify streaming or geometry preparation, but it also
changes grids, Jacobians, collision representation and boundary conditions. It is
not an algebraic cure for missing physics or an unresolved trapped/passing layer.

| Option | Potential benefit | Cost / decision |
| --- | --- | --- |
| Existing Boozer/general surface-angle geometry with Legendre pitch | Preserves working block structure and compatibility; existing VMEC geometry avoids requiring every input to be Boozer transformed | **Default.** Audit the geometry tensor/weight contract and its derivatives. |
| Direct VMEX/VMEC or DESC surface angles | Could avoid an expensive coordinate transform and its derivative in optimization | Use the existing general-geometry path where valid. Compare the same equilibrium, surface moments and shape derivative in two representations before choosing the adapter. No new solver backend is required merely to change the input representation. |
| Field-aligned `(alpha,l)` | Simplifies parallel streaming, may aid a line preconditioner | Global toroidal periodicity, rational surfaces and cross-field ExB coupling remain. Defer a solver rewrite; reconsider only if measured angular coupling dominates after P improvements. |
| Pitch angle `alpha=acos(xi)` with finite differences | YANCC-style regular endpoint handling and line smoothing | Loses the present simple Legendre collision/block structure and introduces new resolution/error tradeoffs. At most test it in P if justified; defer replacement of the fine operator. |
| Bounce coordinates / orbit averaging | Removes a fast coordinate for selected low-collisionality objectives | Model/order restrictions, well creation/merging and singular quadrature require separate evidence. Use an external reduced objective for screening if useful, then verify with DKX. Do not claim full-FP/Phi1 parity from it. |

The [differentiable bounce-averaging study](https://arxiv.org/html/2412.01724v2)
demonstrates that useful reduced objectives can be differentiated; it does not
make a bounce-averaged model equivalent to DKX's full local problem. The
[NEO-2 bootstrap-limit study](https://arxiv.org/abs/2407.21599) also cautions against
using a universal low-collisionality asymptote without the stated precession and
ripple conditions. Both support a validity-based choice, not more active branches.

**The optimization itself, in three steps within one example family:**

Deliver the real optimization in three steps within one existing example family:

1. Use a verified fixed equilibrium, explicit n/T profiles and prescribed Er.
   Validate profile and geometry derivatives independently, including radial
   coordinate, Fourier truncation and metric/Jacobian derivatives. Extend the
   adapter already used by VMEX; pin the actual dependency and equilibrium residual.
2. Optimize a small set of **boundary coefficients** through the equilibrium solve
   and DKX transport/current objective with explicit aspect-ratio, iota, field and
   geometric feasibility constraints. Report the full cost per accepted step,
   rejected steps and compilation. Compare AD with finite differences over several
   parameter counts at equal error; a harmonic field-amplitude descent is insufficient.
3. Add regular-branch stellarator ambipolar response, then self-consistent bootstrap
   current/equilibrium coupling. Differentiate the coupled fixed point or converge
   and validate its implicit Jacobian; freezing the equilibrium current omits part
   of the derivative. Keep a prescribed-Er tokamak example. Use
   [direct neoclassical optimization](https://arxiv.org/abs/2406.04147) and
   [bootstrap-consistent equilibrium optimization](https://arxiv.org/abs/2205.02914)
   as comparison designs, not claims that DKX already reproduces them.

**Exit:** objective improvement exceeds its numerical uncertainty, constraints are
satisfied, full-chain Taylor/FD tests pass on smooth branches, and the final design
is recomputed cold at finer resolution with an independent transport/current
reference. Charge the equilibrium and coordinate transformation to the timing.
Stop and narrow the parameter/physics domain if the derivative or model is invalid;
do not silently freeze it to preserve a descending objective.

Exit: Figures 3 and 4 rendered from tracked inputs; one optimization with objective improvement exceeding its numerical uncertainty, constraints satisfied, full-chain Taylor and finite-difference tests passing on smooth branches, and the final design recomputed cold at finer resolution with an independent SFINCS check on the converged equilibrium. Kill: if the root derivative fails its Taylor test on a regular branch after two weeks of diagnosis, narrow to prescribed-`E_r` derivatives and say so; if the derivative or model is invalid, stop and narrow the parameter domain rather than freezing anything to preserve a descending objective. Effort: 5–6 person-weeks.

### Phase 4: native Phi1 and a W7-X impurity result (4–6 weeks; entry requires Phase 1 error bars)

#189 defers native Phi1 behind "coupled/error contracts". This plan puts it on the roadmap with a named result, because it is the one full-kinetic capability with a live experimental audience that neither yancc nor MONKES/NTX has: W7-X impurity transport is neoclassically dominated in NBI-heated and turbulence-suppressed scenarios with peaking scaling with Z (Nucl. Fusion 2023; PPCF 2025), and the flux-surface variation of the potential is known to change impurity fluxes at the order-unity level ([Mollén et al. 2018](https://iopscience.iop.org/article/10.1088/1361-6587/aac700); García-Regaña et al. 2017), with the classical channel mattering in optimized stellarators ([Buller et al.](https://arxiv.org/abs/1903.12511)). KNOSOS's Phi1 is low-collisionality only.

**Figure 5.** W7-X standard configuration, bulk ions + electrons + one impurity (C or Fe), impurity particle flux and its convective/diffusive decomposition versus ν* with and without Phi1, DKX native against SFINCS with Phi1, and the impurity `E_r` root shift. GPU, warm scans.

Steps: (1) expose the existing compatibility-path Phi1 through the prepared native objects with the *coupled* residual (kinetic + quasineutrality + gauge) as the admission check, linearized quasineutrality first, block/Schur preconditioner, Newton with Eisenstat–Walker forcing as #189 proposes; (2) reproduce Mollén 2018's impurity result; (3) the scan. Exit: Figure 5 with error bars from the coupled residual. Kill: if the coupled Newton solve does not converge on the W7-X case with the linearized quasineutrality after three weeks, ship the frozen-Phi1 comparison labeled as such.

NEOPAX integration then needs a small in-memory protocol for species order,
radial centers/faces, SI fluxes and Jacobians, boundary conditions, validity and
refresh. Check transport conservation and lagged-response error before claiming a
transport simulation. NTX is a candidate monoenergetic database producer; avoid a
second database framework until its normalization/interpolation/restart contract
is compared with DKX's existing one. ESSOS first realizes coils for an accepted
target with field-error, length, curvature, distance and current constraints.
Arbitrary coil fields need not possess nested surfaces. Joint plasma/coils and
open-field-line mirror optimization follow their own physical validity gates.

### Phase 5: the papers, defined by the figures above

**Methods/software paper (CPC or JCP).** Figures: 1 (positioning), 2 (preconditioner ablation, if Phase 2 passes; otherwise the factor-reuse ablation), 3 (derivative verification), plus the convergence-order figures already in the validation matrix, the SFINCS field-by-field parity table, and the MONKES/YANCC monoenergetic table. The contribution statement is: SFINCS-v3 physics including Phi1 and ambipolar roots, on GPU, with every observable carrying an algebraic and a discretization error bar, and derivatives verified through the root. Not "differentiable GPU neoclassics", which yancc already owns as a phrase.

**Physics letter (Nuclear Fusion or JPP Letters).** Figure 4, the accuracy map of Redl/PENTA/monoenergetic closures against full kinetics on real designs, with the ν*^(3/5) inset, and Figure 5 if Phase 4 lands in time. This is the result three design teams have said in print they need.

Results the community would take up immediately, in the order they become available here: verified `dJ_bs/d(boundary)` inside DESC or SIMSOPT for a QI and a large-J_bs QA; the closure accuracy map; neoclassical Jacobians `∂(Γ_s, Q_s)/∂(∇n, ∇T, E_r)` and the root derivative as a T3D-compatible module (Infinity Two's T3D-GX-SFINCS pipeline has SFINCS as its CPU-only, derivative-free component); GPU Phi1 impurity scans for W7-X. No published neural surrogate of core stellarator neoclassical transport exists as of this survey; a DKX-generated table is a cheap by-product once the scans run, and is deliberately *not* a phase.

## 5. Engineering contracts: reuse, differentiation, measurement

These are the specifications Phases 2 and 3 implement against. They are adopted from #189 with one change: the root is already differentiated by the implicit function theorem, so the "design sketch" below describes what to expose, not what to build.

### 5.1 In-memory reuse contract

Extend the existing `ErProblem`, `ErSolveState`, `solve` and SOLVAX factor/recycle
objects. The intended expert interface is a prepared, immutable physical problem
plus explicit reusable state passed in and returned from Python/JAX. The following
is a **design sketch, not an implemented API**:

```python
prepared = prepare(case, layout=fixed_layout)
result, state = evaluate(prepared, profiles, er, state=state)
value, gradient = objective_and_grad(prepared, parameters, state=state)
```

File I/O is unnecessary for optimization. Disk checkpoints are an optional later
serialization of physical state and provenance, not the first implementation.
Keep reusable arrays as bounded pytrees; do not accumulate every iteration's
factors or diagnostics in a closure. State owns its device, dtype, layout,
physical dependency identity, constraints/gauge, previous iterate, recycle space
and preconditioner metadata. Rejected optimizer trial states must not overwrite
the accepted continuation state. A geometry/profile change can preserve layout
and compilation while still requiring all affected coefficients to refresh.

| Reused item | Validity / action |
| --- | --- |
| Compiled executable, quadrature and symbolic structure | Reuse for matching shapes, dtype, static model/layout and device contract. Changing n/T/Er is not automatically a retrace; changing active pitch topology is. |
| Numerical factors | Exact solver only for the same numerical operator, border and constraints, including changed collision/geometry dependencies. RHS-only changes can share factors. Otherwise use as an **approximate preconditioner**, or refactor. |
| Distribution / Phi1 / Er | Initial guess or branch predictor, never an accepted result by identity alone. Interpolate only through an explicit geometry/grid map and recheck physical constraints. |
| Recycle basis | Recompute its image under the new original operator; reorthogonalize and discard dependent/stale vectors. A small parameter step is not a validity certificate. |
| Lagged preconditioner | Permit nearby changed systems while true residuals converge; refresh on loss of convergence, measured extra work, memory pressure or changed constraints/layout. Keep forward/transpose use explicit. |

[PETSc's successive-system rules](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/)
and [maintainer explanation](https://lists.mcs.anl.gov/pipermail/petsc-users/2022-October/047018.html)
make this distinction explicit. [GCRO-DR](https://doi.org/10.1137/040607277) motivates
recycling; SOLVAX's implementation still needs calibration on DKX sequences.
Compare four ablations on exactly the same sequence: compilation only; plus x0;
plus recycle; plus lagged P. Separately test repeated RHSs with an unchanged A.
Record setup/apply/matvec/orthogonalization/adjoint work, cache rebuilds and peak
storage, including large rejected steps, branch changes and species/grid changes.

Select refresh by measured economics: if rebuilding costs `T_build`, compare it
with the expected extra iterations times `T_apply + T_matvec + T_orth` over the
remaining reuse horizon. Use a bounded diagnostic and a cold fallback, not a
universal distance threshold. At least one full-FP n/T sequence and one stellarator
Er/root sequence must agree with independent cold solves within the observable
budget. Replay forward, reverse and permuted sequences: the answer must not depend
on history. Short memory ownership tests must include failed/rejected evaluations.

### 5.2 Differentiation and the coupled potential

Differentiate the **converged equations**, not cache decisions or a fixed number
of unconverged iterations. For `F(u,p)=0`, the adjoint solves
`F_uᵀ lambda=Q_uᵀ`, giving `dQ/dp=Q_p-lambdaᵀF_p`. Warm guesses and lagged P can
be detached from AD while physical coefficients remain differentiable, provided
the primal and adjoint solve the intended equations accurately. This is the
[JAX custom-linear-solve contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
not permission to ignore residuals. Test JVP/VJP, two or more FD steps, quadratic
Taylor remainders above the noise floor, and cold/warm derivative agreement.
[Paul et al.'s neoclassical adjoint work](https://arxiv.org/abs/1904.06430)
already includes root acceleration; evaluate safeguarded Newton continuation
against the existing Brent search, charging every slope evaluation and retaining
bracket fallback and cold final verification. Root selection switches are not smooth.

For Phi1, the eventual state is `u=(f, Phi1, source/gauge variables)` and, when
appropriate, Er. Reuse both f and Phi1, but certify the **coupled** kinetic,
quasineutrality and gauge residual. A kinetic solve with frozen Phi1 is not the
coupled derivative. Start with linearized quasineutrality and a block/Schur
preconditioner; qualify nonlinear Newton/line-search behavior and potential gauge
before native promotion. Reuse the existing compatibility implementation first.
[PETSc SNES lagging](https://petsc.org/release/manualpages/SNES/SNESSetLagPreconditioner/)
and [Eisenstat–Walker forcing](https://users.wpi.edu/~walker/Papers/forcing_terms%2CSISC_17%2C1996%2C16-32.pdf)
suggest avoiding oversolved early Newton steps. Use existing SOLVAX support where
available; tighten terminal primal/adjoint accuracy to the observable budget.
This coupled extension follows the no-Phi1 reuse contract, not a parallel rewrite.

### 5.3 Measure the work users actually pay for

Time preparation → all kinetic/root/Newton evaluations → moments → backward pass
→ acceptance, synchronizing arrays **and diagnostic effects**. Also separate process
startup, empty-cache compile, persistent-cache load, warm solve and reuse modes.
Use at least five unprofiled repetitions/pairs after checking stable load; retain
samples, median and dispersion. Include failed solves, line searches and rebuilds.
Peak RSS, aggregate MPI memory, allocator peak VRAM and compiler temporary estimates
are separate quantities. Do not sum nested PETSc events or overlapping GPU intervals.

Follow [JAX profiling guidance](https://docs.jax.dev/en/latest/profiling.html):
coarse named regions first, then XPlane/Perfetto and HLO/XLA/kernel inspection of the
identified bottleneck. Check trace completeness/event caps; do not infer occupancy
or speed from a capped trace or HLO operation count. Keep TensorBoard/XProf/Perfetto
artifacts outside Git. Profiled runs diagnose; unprofiled runs establish runtime.

CPU policy compares thread counts and a small process pool with controlled BLAS/XLA
threads and per-process memory. Logical JAX CPU devices share resources; they are
not extra CPUs. GPU policy compares serial continuation against independent batches
on one/two physical A4000s, including compile and transfers, uneven sizes, gradient
placement and failure propagation. Nearby points may benefit more from serial warm
reuse than simultaneous cold solves; group related points into resident sequences
only if that measured tradeoff wins. Use strong/weak scaling for independent work;
state partitioning and multi-host collectives are deferred.

**Exit:** an installed native scan and a complete value/gradient/root workload
show a reproducible benefit at fixed accepted accuracy. As a decision target,
seek ≥20% end-to-end improvement or ≥2× lower measured peak memory on the identified
bottleneck, with no loss of admitted cases; otherwise keep the simpler baseline.
Targets are not promised results. Demonstrate cold fallback, bounded memory and
no unintended recompilation for supported parameter updates on CPU and GPU.

## 6. Documentation, examples and deliberate reduction

The README should contain one runnable start, a short feature/results summary and
an easy-to-advanced workflow map. Remove categorical SFINCS/AD/GPU claims and
multiple historical timing narratives. Keep scope beside each result. The docs
landing page should route readers by task; remove its duplicate performance table
and universal “runs in seconds” claim. Preserve existing URLs while consolidating.

| Existing material | Canonical destination / action |
| --- | --- |
| `installation`, `examples`, first-run parts of `usage` | Tutorials: first physical result, convergence, gradients; examples remain executable sources. |
| `case_files`, `applications`, `optimization`, `vmex_workflow`, `parallelism`, troubleshooting | How-to: one guide per user task; combine overlapping optimization instructions after the real adapter exists. |
| `physics_models`, `system_equations`, `physics_reference`, `theory_from_upstream`, `method`, `numerics` | Explanation: one model/units/constraints derivation and one numerical-method explanation; keep independent citations and applicability. |
| `api`, `cli`, `inputs`, `outputs`, `normalizations`, `capabilities`, `feature_matrix` | Reference: schemas/status generated from the existing definitions; link instead of copying a second capability matrix. |
| `performance`, `validation_matrix`, `parity`, `fortran_comparison`, `research_lanes` | Evidence: accepted results versus historical/experimental records; archive a superseded experiment, do not present it as a second roadmap. |

Reuse the nine numbered examples. Keep 01–03 for analytic/VMEC/Boozer profiles,
04 for the documented monoenergetic path, 05 for root evidence, 06 for convergence,
07 for gradients, 08 explicitly labeled geometry proxy until the real boundary
example replaces it, and 09 for qualified Phi1/impurity comparisons. Add warm/batch
options to the relevant examples and guides rather than another numbered gallery.
Each needs editable profiles/geometry/resolution, units, model scope, expected
qualitative result, a quick mode and a checksummed research case with measured
resource requirements. Do not imply all nine support native Case execution.

Students should reach a plot and a readable physical summary in one command, then
see why a small residual is not grid convergence. Researchers should be able to
replace equilibrium/profiles and run convergence, repeated solves and gradients
without copying internal code. Examples must expose rejected points, root scope,
error bars and SI conventions, not connect invalid entries as real data.

Audit duplication in `solve.py`, `coarse_precond.py`, `multigrid.py`, geometry and
workflow orchestration before creating helpers. Keep one physics assembly and one
solver policy; generic new algorithms belong in SOLVAX. Remove obsolete paths only
after preserving meaningful assertions, not by deleting difficult tests. Report
source/test/tool/doc file counts, physical lines, tracked bytes, fresh-clone size,
wheel and installed-owned size separately. Dependencies still count in user setup
cost. Preserve the <20 MiB owned-artifact/fresh-clone targets and soft 45k production
line target as visible debt where missed, without forcing unrelated modules into
one file or silently changing measurement definitions. No history rewrite is planned.

The 95% line/branch goal remains a ratchet for stable reachable code, not a reason
to manufacture tests or postpone an urgent scientific correction. Run focused
mathematical/physics tests per change, installed examples and warning-clean docs;
reserve large external/GPU campaigns for relevant changes. Keep compact inputs,
checksums, commands and result summaries in Git; raw states, traces and build trees
belong in an archive. Verify retrieval before pruning an artifact that underpins
an advertised result.

The README is governed by section 7 rule 12 and by the existing gates (`test_benchmark_doc_claims`, `test_readme_quickstart_runs`, `test_figure_provenance`): a self-contained quickstart that runs `dkx.run` and prints one flux and the solver route, one gradient example, the four-code capability table, the measured results table with every pinned number, both README figures and the cross-code figure, a BibTeX block, and no hedging sentences; hedges live in the docs quadrant Diátaxis assigns them.

## 7. Working method

The stack that produced #170–#188 was written in one day: 53 commits, 44 files, about 5,900 insertions. At the Cisco/SmartBear ceiling of roughly 500 reviewed lines per hour ([SmartBear](https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/)) that is 12 to 18 reviewer-hours, ten to twenty times over Google's "100 lines is usually reasonable, 1,000 is usually too large" ([Google eng-practices](https://google.github.io/eng-practices/review/developer/small-cls.html)). The work was good; the process made it unreviewable, and an unreviewable stack is a release risk however green its CI. The following rules replace volume with direction.

1. **Figure-first.** The roadmap is the ordered figure list of the two papers (Whitesides: a good outline for the paper is also a good plan for the research programme, [Adv. Mater. 2004](https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper)). Each roadmap item is a figure or table with an owner, a status and an acceptance criterion. Work that maps to no figure and no bug is not scheduled. Every week at least one merged PR adds or upgrades a paper figure.
2. **PR size cap: at most 400 changed lines and 10 files, one idea per PR, refactor never shares a PR with behaviour.** Above 800 lines the PR is split before review. Generated data and pinned artifacts go in their own PR.
3. **No stacks without tooling.** Stacks are allowed only with depth at most 3, auto-rebase tooling, and each PR independently mergeable ([Graphite](https://graphite.com/guides/stacked-diffs)). Otherwise finish and merge PR n before opening n+1; at most three open PRs per author. Branch lifetime at most 48 hours ([trunk-based development](https://trunkbaseddevelopment.com/)); incomplete features land behind a flag.
4. **Agent output is budgeted by review capacity.** One reviewer, two 60-minute sessions a day at 400 lines each is about 800 reviewed lines a day. The agent stops opening PRs when the review queue reaches that, whatever its generation speed.
5. **Four CI tiers; the PR gate is T1 and takes at most 20 minutes.** T0 lint and unit on every push. T1 touched-module tests, one small SFINCS parity deck, the README example. T2 nightly: the full 38-deck matrix and GPU. T3 on release tags: cross-code, figure regeneration, wheel, Zenodo. A change that cannot be trusted after T1 is too large.
6. **Three documents, three jobs, hard caps.** `plan.md` holds phases, figures and criteria and never status prose (this document; deletions weekly). `docs/adr/NNNN-*.md` holds decisions, one page each, immutable, superseded by a new ADR ([adr.github.io](https://adr.github.io/)). `CHANGELOG.md` holds what shipped per release ([Wilson et al. 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)). Execution detail lives in PR descriptions; review reports are dated write-once files under `docs/reviews/`.
7. **Experiments are time-boxed with a written kill criterion before they start.** The record is a one-page file under `docs/experiments/` with hypothesis, admission test, result, decision. Phase 2 is the template.
8. **Provenance is five fields, not a framework.** Every output records version, git SHA, JAX/jaxlib versions with the x64 flag, `case_id` and command line, device and host ([Taschuk & Wilson 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412), rule 10). Paper figures regenerate from `publications/<paper>/make_figures.py` in a pinned environment (the DESC pattern). The supervised runner, verifier, archive and provenance tooling that exist are frozen; nothing further until a reviewer asks.
9. **Benchmarks follow Hoefler and Belli** ([SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)) and the [JAX benchmarking page](https://docs.jax.dev/en/latest/benchmarking.html): absolute times beside every ratio, all decks including losses, median of at least five warm solves after `block_until_ready()`, compile time once per resolution, spread stated, hardware and versions in the caption. Pinned SHA, idle machine, A/B/A/B.
10. **Verification is labelled by tier everywhere.** Tier A code verification: analytic limits, Onsager symmetry, adjoint-versus-finite-difference. Tier B solution verification: convergence per axis with the criterion stated. Tier C cross-code: SFINCS parity (same discretization, regression) reported separately from MONKES/yancc independence. Tier D validation: W7-X `E_r` with experimental uncertainty. Tier C is never called validation ([Oberkampf & Roy 2010](https://www.cambridge.org/core/books/abs/verification-and-validation-in-scientific-computing/index/EE029CB068531D27); ASME V&V 20).
11. **Prepare releases at paper milestones**, with tag, changelog, `CITATION.cff` and a Zenodo DOI when the maintainer authorizes publication after important goals are achieved. Papers cite the tag, not `main`.
12. **README budget enforced by a test**: at most 160 lines and 1,000 words; at most two Python blocks of at most 16 lines each, the first a quickstart that runs `dkx.run` and the second ending in a gradient; at least one BibTeX block; the hero, parity and cross-code figures; and none of "has not", "cannot", "does not yet", "is not converged". Hedges live in the docs quadrant Diátaxis assigns them ([diataxis.fr](https://diataxis.fr/)). The numbers are set just above what the page currently needs, because the repository's own contracts force both code blocks and the measured-results table: `test_readme_quickstart_runs` requires a self-contained quickstart that prints a flux and a route, and `test_benchmark_doc_claims` pins fourteen measured tokens and the capability families. A budget the page cannot meet is not a budget. Measured exemplars: Diffrax 82 lines / 315 words, Optimistix 83 / 324, simsopt 85 / 403, DESC 137 / 623, jax-cfd 151 / 693; DKX was 250 / 1,508 with no citation block and no `jax.grad` before the 2026-09-06 rewrite, and had regrown from 158 lines three times.

13. **Definition of done for a PR, in at most ten lines:** what changed, why, which CI tier proves it, which figure, ADR or issue it serves; tests and docs in the same PR; a physics change regenerates the affected figure by script and attaches it.

## 8. Deferred work

Defer a new fine-grid coordinate/discretization backend; MUMPS reimplementation;
BLR/mixed precision without measured kinetic evidence; learned/Nyström/PINN
preconditioners; unqualified truncated-adjoint windows; multi-host/state-decomposed
execution; new database frameworks; joint coils/plasma optimization; and mirrors.
Existing useful experimental APIs can remain clearly labeled. A deferred item
reopens only with a named user calculation the active deliverables cannot serve,
a bounded experiment and a measurable decision. Native Phi1/full drifts are
scientific completeness work after the corresponding coupled/error contracts,
not abandoned physics or an excuse to claim SFINCS parity early.

Closed by the 2026-09-13 measurements (`docs/experiments/2026-09-13-*.md`): the SuperLU fill-reducing preconditioner route for `Ntheta·Nzeta ≈ 800` full-FP decks; Legendre-basis geometric multigrid (DKX's own smoother study); float32 factors as the route for decks whose float64 LU fits after Phase 2 step 6; Fourier-diagonal constant-coefficient preconditioners and low-rank tensor solvers (both drop the mirror force or need a separation rank trapped wells do not have); memory launch gates derived by scaling an unattributed peak.

Also deferred, from the 2026-09-06 numerics survey: fp32 factors before Ruiz equilibration or at scaled κ above about 1e10; half precision anywhere in a factor; differentiating through Krylov iterations; semi-coarsened multigrid with the exact-in-L plane smoother unless Phase 2 shows iterations growing with resolution; cuDSS through an XLA FFI call until a deck needs a general LU on GPU; a neural surrogate of core neoclassical transport (a cheap by-product of Phase 3 and 4 scans, not a phase).

## 9. History of this plan and disposition of its predecessors

| Predecessor | Where it is | Disposition |
| --- | --- | --- |
| Phase checklist and execution diary (to 2026-09-05) | git history before `be6506fe` | replaced by the R0–R11 queue |
| R0–R11 work queue with implementation checkpoint (`be6506fe`, #169–#188) | [`plan.md` at `be6506fe`](https://github.com/uwplasma/DKX/blob/be6506fedbf357ab7fbbc1c1eaa35195080afc9e/plan.md) | its contract, evidence hierarchy, benchmark families and retained decisions are sections 1 and 3 here; the checkpoint diary goes to `CHANGELOG.md` |
| Three-deliverable plan (#189) | merged; [its `plan.md`](https://github.com/uwplasma/DKX/pull/189/files) | its state tables, capability priorities, four families, reuse contract, differentiation and measurement specs, coordinate table, optimization steps, documentation table and deferred list are sections 2, 3.3, 4 and 5–8 here |
| Independent take (#190) | this branch's history | its figure-first phases, competitor positioning, Phase 2 diagnosis, working method and references are sections 1, 2.4, 4, 7 and 10 here |
| Survey reports behind #190 | branch `review/surveys-20260906` | audit trail for section 10 |
| Independent review of #227–#229 and the merged profiles (2026-09-13) | `docs/experiments/2026-09-13-preconditioner-cost-anatomy.md`, `…-sfincs-sparsify-threshold.md`; scripts and survey reports outside Git in `dkx-review-evidence-20260913/` | Phase 1 order of work and cross-code admission criterion, Phase 2 steps 6–7 and the research-bet kill tests, and the closed items in section 8 |
| Corrections accepted from #176 to #172 | `plan.md` at `be6506fe`, section 2.4 | an unknown-count threshold does not certify a reference; a saturated condition number attributes nothing; the Shaing–Callen paper concerns the collisionality limit, not a pitch ladder; clone-size arithmetic assumes an unchanged tree |

Merge order executed on 2026-09-06: #192 (reverts of #171 and #175) → #169 → #170 → #173 → #174 → #176 → #177 → #178 → #179 → #180 → #181 → #182 → #183 → #184 → #185 → #186 → #187 → #188 → #191 → #189 → this plan; #168 and #172 closed as superseded. Every commit is authored by the maintainer.

## 10. References

### Neoclassical codes, benchmarks and physics
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014), DOI 10.1063/1.4870077, arXiv 1312.6058 https://arxiv.org/abs/1312.6058
- SFINCS repository, landreman/sfincs; input.namelist symlink page https://github.com/landreman/sfincs/blob/master/fortran/version3/input.namelist ; issue #1 quoting preconditionerOptions (F) https://github.com/landreman/sfincs/issues/1
- Mollen, Landreman, Smith, Braun, Helander, "Impurities in a non-axisymmetric plasma: transport and effect on bootstrap current", arXiv 1504.04810 https://arxiv.org/abs/1504.04810
- Mollen, Landreman, Smith, Garcia-Regana, Nunami, "Flux-surface variations of the electrostatic potential in stellarators: impact on the radial electric field and neoclassical impurity transport", PPCF 60, 084001 (2018) https://www.osti.gov/biblio/1499870
- Garcia-Regana et al., "Electrostatic potential variation on the flux surface and its impact on impurity transport" (2017) https://www.researchgate.net/publication/271079728_Electrostatic_potential_variation_on_the_flux_surface_and_its_impact_on_impurity_tran
- Buller et al., "The importance of the classical channel in the impurity transport of optimized stellarators", J. Plasma Phys., arXiv 1903.12511 https://arxiv.org/html/1903.12511
- Paul, Abel, Landreman, Dorland, "An adjoint method for neoclassical stellarator optimization", arXiv 1904.06430 https://arxiv.org/abs/1904.06430
- Paul, Landreman, Antonsen, "Adjoint methods for stellarator shape optimization and sensitivity analysis", arXiv 2005.07633 https://arxiv.org/pdf/2005.07633
- Paul et al., "Adjoint approach to calculating shape gradients for three-dimensional magnetic confinement equilibria" https://www.osti.gov/pages/biblio/1597704
- Paul et al., "Adjoint methods for quasisymmetry of vacuum fields on a surface", arXiv 2108.11433 https://arxiv.org/pdf/2108.11433
- Conlin, Landreman, "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv 2607.20861 (F abstract and full text) https://arxiv.org/abs/2607.20861 ; https://arxiv.org/html/2607.20861
- yancc repository https://github.com/f0uriest/yancc
- Escoto, Velasco, Calvo, Landreman, Parra, "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", Nucl. Fusion, DOI 10.1088/1741-4326/ad3fc9, arXiv 2312.12248 (F abstract and full text) https://arxiv.org/abs/2312.12248 
- MONKES repository (F, from paper) https://github.com/JavierEscoto/MONKES/
- Escoto Lopez, PhD thesis, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas", arXiv 2510.27513 https://arxiv.org/abs/2510.27513
- "Evaluation of neoclassical transport in nearly quasi-isodynamic stellarator magnetic fields using MONKES", arXiv 2410.17836 https://arxiv.org/pdf/2410.17836
- NTX repository, uwplasma/NTX https://github.com/uwplasma/NTX
- Velasco, Calvo, Parra, Garcia-Regana, "KNOSOS: a fast orbit-averaging neoclassical code for stellarator geometry", J. Comput. Phys. (2020), DOI 10.1016/j.jcp.2020.109512, arXiv 1908.11615 https://arxiv.org/abs/1908.11615
- "Fast simulations for large aspect ratio stellarators with the neoclassical code KNOSOS", arXiv 2106.01727 https://arxiv.org/pdf/2106.01727
- Hirshman, Shaing, van Rij, Beasley, Crume, "Plasma transport coefficients for nonsymmetric toroidal confinement systems", Phys. Fluids 29, 2951 (1986) https://pubs.aip.org/aip/pfl/article-abstract/29/9/2951/944354/ ; OSTI (S) https://www.osti.gov/servlets/pu
- van Rij, Hirshman, "Variational bounds for transport coefficients in three-dimensional toroidal plasmas", Phys. Fluids B 1, 563 (1989) https://pubs.aip.org/aip/pfb/article-abstract/1/3/563/940728/
- "Modelling of relativistic electron transport with non-relativistic DKES solver", J. Plasma Phys. (2024) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/modelling-of-relativistic-electron-transport-with-nonrelativistic-dkes-solver/A
- Spong, "Generation and damping of neoclassical plasma flows in stellarators", Phys. Plasmas 12, 056114 (2005) https://pubs.aip.org/aip/pop/article/12/5/056114/1015589/
- "Three-dimensional equilibria and transport in RFX-mod: A description using stellarator tools", Phys. Plasmas 18, 062505 (2011) https://pubs.aip.org/aip/pop/article-abstract/18/6/062505/387754/
- Kernbichler et al., "Recent progress in NEO-2 - a code for neoclassical transport computations based on field line tracing", Plasma Fusion Res. 3, S1061 (2008) https://www.jstage.jst.go.jp/article/pfr/3/0/3_0_S1061/_article/-char/en
- Kernbichler, Kasilov, Kapper, Martitsch, Nemov, Albert, Heyn, "Solution of drift kinetic equation in stellarators and tokamaks with broken symmetry using the code NEO-2", PPCF 58, 104001 (2016) https://iopscience.iop.org/article/10.1088/0741-3335/58/10/10400
- Beurskens et al., "Demonstration of reduced neoclassical energy transport in Wendelstein 7-X", Nature (2021) (S; author list U) https://www.nature.com/articles/s41586-021-03687-w
- W7-X power balance study (NEOTRANSP use), PPCF (2025) https://iopscience.iop.org/article/10.1088/1361-6587/ade824
- EUTERPE vs NEOTRANSP Er benchmarking figure https://www.researchgate.net/figure/Benchmarking-of-the-global-neoclassical-radial-electric-field-calculated-with-EUTERPE_fig1_378516498
- Beidler et al., "Benchmarking of the mono-energetic transport coefficients - results from the ICNTS", Nucl. Fusion 51, 076001 (2011) https://iopscience.iop.org/article/10.1088/0029-5515/51/7/076001
- Redl, Angioni, Belli, Sauter, "A new set of analytical formulae for the computation of the bootstrap current and the neoclassical conductivity in tokamaks", Phys. Plasmas 28, 022502 (2021) https://pubs.aip.org/aip/pop/article/28/2/022502/124727/ ; open copy 
- Sauter, Angioni, Lin-Liu, Phys. Plasmas 6, 2834 (1999) - U (no URL fetched)
- Landreman, Buller, Drevlak, "Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement", Phys. Plasmas 29, 082501 (2022), DOI 10.1063/5.0098166, arXiv 2205.02914 https://arxiv.org/abs/2205.02914 ; h
- Albert, Beidler, Kapper, Kasilov, Kernbichler, "On the convergence of bootstrap current to the Shaing-Callen limit in stellarators", arXiv 2407.21599 https://arxiv.org/abs/2407.21599
- Saxena, Ferraro, Martin, Wright, "Bootstrap current modeling in M3D-C1", J. Plasma Phys. 91, E141 (2025), DOI 10.1017/S0022377825100834, arXiv 2507.05166 https://arxiv.org/abs/2507.05166 ; https://www.cambridge.org/core/journals/journal-of-plasma-physics/art
- DESC tutorial "Bootstrap Current Self-Consistency" https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html
- VMEX references page https://vmex.readthedocs.io/en/latest/project/references.html
- Goodman et al., "Quasi-isodynamic stellarators with low turbulence as fusion reactor candidates", PRX Energy 3, 023010 (2024), arXiv 2405.19860 https://arxiv.org/abs/2405.19860 ; https://link.aps.org/doi/10.1103/PRXEnergy.3.023010
- Goodman et al., "Constructing precisely quasi-isodynamic magnetic fields", J. Plasma Phys. (2023) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/constructing-precisely-quasiisodynamic-magnetic-fields/6601E449C8DD3B3FEB361DA2C5732EF
- Jorge et al., "A single-field-period quasi-isodynamic stellarator", J. Plasma Phys. https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/singlefieldperiod-quasiisodynamic-stellarator/9B2A5FDCCD7774E4F91BE45E75FDC6B0
- "CIEMAT-QI4X: a reactor-relevant quasi-isodynamic stellarator configuration compatible with an island divertor", Nucl. Fusion, arXiv 2512.08825 https://arxiv.org/pdf/2512.08825 ; https://iopscience.iop.org/article/10.1088/1741-4326/ae54ad
- "Near-axis quasi-isodynamic database", arXiv 2601.08400 https://arxiv.org/pdf/2601.08400
- "Optimization of nonlinear turbulence in stellarators", J. Plasma Phys. 90, 905900210 (2024) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/optimization-of-nonlinear-turbulence-in-stellarators/916FCC56452B5B166C14868F56D99AF5
- Type One Energy, "A comprehensive, unified baseline physics design for the Type One Energy stellarator fusion pilot power plant, 'Infinity Two'", J. Plasma Phys. 91, E65 (2025) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/compreh
- "Predictions of core plasma performance for the Infinity Two fusion pilot plant", J. Plasma Phys. (2025) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/predictions-of-core-plasma-performance-for-the-infinity-two-fusion-pilot-plant/
- Thea Energy, "Overview of the Helios Design: A Practical Planar Coil Stellarator Fusion Power Plant", arXiv 2512.08027 https://arxiv.org/abs/2512.08027 ; PDF https://thea.energy/wp-content/uploads/2025/12/20251210_FPP_Helios_overview_paper.pdf ; Fusion Eng. 
- "Equilibrium optimization of the Helios planar coil stellarator power plant", Fusion Eng. Des. (2026) https://sciencedirect.com/science/article/pii/S0920379626002905
- "Stellarator fusion systems enabled by arrays of planar coils", Nucl. Fusion (2025) https://iopscience.iop.org/article/10.1088/1741-4326/ada56c
- Proxima Fusion, "Stellaris: A high-field quasi-isodynamic stellarator for a prototypical fusion power plant", Fusion Eng. Des. (2025) https://www.sciencedirect.com/science/article/pii/S0920379625000705 ; press release https://www.proximafusion.com/press-news
- "Quantitative comparison of impurity transport in turbulence reduced and enhanced scenarios at Wendelstein 7-X", Nucl. Fusion (2023) https://iopscience.iop.org/article/10.1088/1741-4326/aceb76
- "The suppression of anomalous impurity transport above a critical normalized density gradient scale length in Wendelstein 7-X", PPCF (2025) https://iopscience.iop.org/article/10.1088/1361-6587/add597
- "Neural network-based surrogate model for 3D edge-plasma transport in the standard configuration of W7-X", Nucl. Fusion 66 (2025) https://iopscience.iop.org/article/10.1088/1741-4326/ae203d
- IPP abstract "Neoclassical transport simulations for stellarators" (DCOM/NNW description) https://pure.mpg.de/rest/items/item_2139735_1/component/file_2139734/content
- MMMnet surrogate (NSTX-U) https://www6.lehigh.edu/~eus204/per/publications/journals/tps24_MMMnetNSTXU.pdf
- "5D Neural Surrogates for Nonlinear Gyrokinetic Simulations of Plasma Turbulence", arXiv 2502.07469 https://arxiv.org/pdf/2502.07469
- "Efficient dataset construction using active learning and uncertainty-aware neural networks for plasma turbulent transport surrogate models", arXiv 2507.15976 https://arxiv.org/pdf/2507.15976

### Numerical methods
- "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv:2607.20861 (2026). https://arxiv.org/abs/2607.20861 ; full text https://arxiv.org/html/2607.20861v1 (fetched). Author list not extracted — UNVERI
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014). https://arxiv.org/pdf/1312.6058 (fetched, text extracted); https://pubs.a
- SFINCS repository and v3 manual. https://github.com/landreman/sfincs ; https://raw.githubusercontent.com/landreman/sfincs/master/doc/manual/version3/runs.tex (read). input.tex not found (404) — namelist option names UNVERIFIED.
- Escoto et al., "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", arXiv:2312.12248. https://arxiv.org/pdf/2312.12248 (URL seen)
- Escoto Lopez, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas" (thesis), arXiv:2510.27513 (2025). https://arxiv.org/abs/2510.27513 (fetched)
- Belli & Candy, "Full linearized Fokker-Planck collisions in neoclassical transport simulations", PPCF 54, 015015 (2012). https://iopscience.iop.org/article/10.1088/0741-3335/54/1/015015
- Landreman & Ernst, "New velocity-space discretization for continuum kinetic calculations and Fokker-Planck collisions", J. Comput. Phys. 243, 130-150 (2013). https://arxiv.org/abs/1210.5289 ; https://www.sciencedirect.com/science/article/abs/pii/S00219991130
- Velasco et al., KNOSOS. https://arxiv.org/pdf/2106.01727 ; https://github.com/joseluisvelasco/KNOSOS (URLs seen)
- PPPL-4775, "Numerical Calculation of Neoclassical Distribution Functions ..." https://bp-pub.pppl.gov/pub_report/2012/PPPL-4775.pdf (URL seen)
- DKX and SOLVAX repositories (given by the task; not fetched). https://github.com/uwplasma/DKX ; https://github.com/uwplasma/SOLVAX
- Dorf, Dorr, Ghosh, Umansky, Soukhanovskii, "Implicit full-F simulations of neoclassical ion transport", Phys. Plasmas 32(8) (2025). https://www.osti.gov/biblio/2588989 (fetched)
- "Axisymmetric Gyrokinetic Simulation of ASDEX-Upgrade Scrape-off Layer Using a Conservative Implicit BGK Collision Operator" (Gkeyll), arXiv:2507.22821. https://arxiv.org/abs/2507.22821
- Barnes, Abel, Dorland et al., "Linearized model Fokker-Planck collision operators for gyrokinetic simulations. II. Numerical implementation and tests", Phys. Plasmas 16, 072107 (2009). https://arxiv.org/abs/0809.3945
- GENE-X LBD collision operator (implementation/verification). https://www.researchgate.net/publication/357053643_Implementation_and_verification_of_a_conservative_multi-species_gyro-averaged_full-f_Lenard-Bernstein_Dougherty_collision_operator_in_the_gyrokine
- "An Angular Multigrid Preconditioner for the Radiation Transport Equation with Forward-Peaked Scatter", arXiv:2010.04559. https://arxiv.org/html/2010.04559 ; Fokker-Planck variant https://www.sciencedirect.com/science/article/pii/S0377042718306174
- "P-Multigrid Method for the Discontinuous Galerkin Discretization of Elliptic Problems", J. Sci. Comput. (2025). https://link.springer.com/article/10.1007/s10915-025-03105-7
- Parks, de Sturler, Mackey, Johnson, Maiti, "Recycling Krylov subspaces for sequences of linear systems", SIAM J. Sci. Comput. 28(5), 1651-1674 (2006), doi:10.1137/040607277. https://vtechworks.lib.vt.edu/items/590c07fe-a0c8-49b2-9494-be5061f5fbf7 ; https://w
- Soodhalter, de Sturler, Kilmer, "A survey of subspace recycling iterative methods", GAMM-Mitt. 43(4), e202000016 (2020), doi:10.1002/gamm.202000016. https://arxiv.org/abs/2001.10347 ; https://arxiv.org/pdf/2001.10347 (fetched, text extracted); https://online
- de Sturler, "Truncation strategies for optimal Krylov subspace methods", SIAM J. Numer. Anal. 36(3), 864-889 (1999), doi:10.1137/S0036142997315950 (DOI taken from the survey's reference list; not fetched separately)
- Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24(1), 20-37 (2002) (from the survey's reference list; not fetched separately)
- Kilmer & de Sturler, "Recycling subspace information for diffuse optical tomography", SIAM J. Sci. Comput. (2006) (from the survey's reference list)
- "Recycling Krylov Subspaces and Truncating Deflation Subspaces for Solving Sequence of Linear Systems", ACM TOMS (2021). https://dl.acm.org/doi/10.1145/3439746
- Applications: https://arxiv.org/pdf/1501.03358 (CFD); https://arxiv.org/pdf/2309.09925 (aerostructural adjoints); https://arxiv.org/pdf/2401.09516 (neural-operator data generation)
- Carson & Higham, "Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions", SIAM J. Sci. Comput. (2018); MIMS EPrint 2017.24. https://nhigham.com/2017/07/26/accelerating-the-solution-of-linear-systems-by-iterative-refinement-i
- Amestoy, Buttari, Higham, L'Excellent, Mary, Vieuble, "Five-precision GMRES-based iterative refinement", SIAM J. Matrix Anal. Appl. 45, 529-552 (2024); MIMS EPrint 2021.5. https://eprints.maths.manchester.ac.uk/2852/1/paper.pdf (fetched, text extracted); htt
- Higham & Mary, "Mixed precision algorithms in numerical linear algebra", Acta Numerica 31 (2022). https://eprints.maths.manchester.ac.uk/2841/ ; https://research.manchester.ac.uk/en/publications/mixed-precision-algorithms-in-numerical-linear-algebra/
- Abdelfattah et al., "A Survey of Numerical Methods Utilizing Mixed Precision Arithmetic". https://arxiv.org/pdf/2007.06674
- "Mixed Precision GMRES-based Iterative Refinement with Recycling". https://arxiv.org/pdf/2201.09827
- NVIDIA cuDSS documentation (v0.8.0, Preview). https://docs.nvidia.com/cuda/cudss/index.html (fetched); https://developer.nvidia.com/cudss
- nvmath-python sparse direct solver (cuDSS-backed). https://docs.nvidia.com/cuda/nvmath-python/0.5.0/host-apis/sparse/index.html ; https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver ; https://pypi.org/project/nvidia-cudss
- spineax (cuDSS in JAX via FFI). https://github.com/johnviljoen/spineax ; cudss_jax MWE https://github.com/stergiosba/cudss_jax ; JAX discussion https://github.com/jax-ml/jax/discussions/33205
- sparsax (SuiteSparse CHOLMOD/KLU via XLA FFI). https://github.com/knaaptime/sparsax/blob/main/README.md
- JAXMg (cuSOLVERMg multi-GPU dense via FFI). https://arxiv.org/pdf/2601.14466
- jax.experimental.sparse.linalg.spsolve docs. https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
- Ghysels & Synk, "High performance sparse multifrontal solvers on modern GPUs", Parallel Computing 110 (2022). https://www.osti.gov/pages/biblio/1960514 (fetched)
- Claus, Ghysels, Boukaram, Li, "A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression", Int. J. HPC Appl. (2025), doi:10.1177/10943420241288567. https://journals.sagepub.com/doi/10.1177/1094342024128856
- Li & Ghysels, ATPESC direct-solver lectures 2022/2023 (URLs seen). https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2023/08/ATPESC-2023-Track-5-Talk-3-Li-Ghysels-DirectSolvers.pdf
- Rader, Lyons, Kidger, "Lineax: unified linear solves and linear least-squares in JAX and Equinox", arXiv:2311.17283 (NeurIPS 2023 AI4Science). https://arxiv.org/abs/2311.17283 ; https://arxiv.org/pdf/2311.17283 (fetched, text extracted); https://github.com/p
- Rader et al., "Optimistix: modular optimisation in JAX and Equinox", arXiv:2402.09983. https://arxiv.org/pdf/2402.09983 ; adjoints doc https://docs.kidger.site/optimistix/api/adjoints/ (fetched); https://docs.kidger.site/optimistix/api/root_find/ ; https://g
- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-Lopez, Pedregosa, Vert, "Efficient and Modular Implicit Differentiation", NeurIPS 2022, arXiv:2105.15183. https://arxiv.org/pdf/2105.15183 ; https://ar5iv.labs.arxiv.org/html/2105.15183
- JAX issue #15837 "GMRES Fails Silently and Frequently from Stagnation". https://github.com/jax-ml/jax/issues/15837 ; gmres docs https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html ; JEP 18137 https://docs.jax.dev/en/latest/jep/1813
- torch-sla, "Differentiable Sparse Linear Algebra with Adjoint Solvers ...", arXiv:2601.13994. https://arxiv.org/pdf/2601.13994
- "Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation", arXiv:2608.08559. https://arxiv.org/html/2608.08559 (title only)
- "Automating Steady and Unsteady Adjoints: Efficiently Utilizing Implicit and Algorithmic Differentiation", arXiv:2306.15243. https://arxiv.org/html/2306.15243 (title only)
- Pierce & Giles, "Adjoint recovery of superconvergent functionals from PDE approximations", SIAM Review 42(2), 247-264 (2000) (metadata from search); Giles' error-analysis page https://people.maths.ox.ac.uk/gilesm/old/error.html
- Giles & Pierce, "Adjoint Error Correction for Integral Outputs", Springer (doi 10.1007/978-3-662-05189-4_2). https://link.springer.com/chapter/10.1007/978-3-662-05189-4_2
- Giles & Pierce, "Progress in adjoint error correction for integral functionals", Comput. Vis. Sci. https://people.maths.ox.ac.uk/~gilesm/files/cvs04.pdf ; https://link.springer.com/article/10.1007/s00791-003-0115-y
- Becker & Rannacher, "An optimal control approach to a posteriori error estimation in finite element methods", Acta Numerica (2001). https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation
- "Linearization Errors in Discrete Goal-Oriented Error Estimation", arXiv:2305.15285. https://arxiv.org/pdf/2305.15285 (title only)
- Roache, Grid Convergence Index (secondary sources). https://cfd.university/blog/how-to-manage-uncertainty-in-cfd-the-grid-convergence-index/ ; Roy, "Grid Convergence Error Analysis for Mixed-Order Numerical Schemes" https://www.aoe.vt.edu/content/dam/aoe_vt_
- Salari & Knupp, "Code Verification by the Method of Manufactured Solutions", SAND2000-1444 (2000). https://www.osti.gov/biblio/759450/
- Roache, "Code Verification by the Method of Manufactured Solutions", J. Fluids Eng. 124(1), 4 (2002). https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791/Code-Verification-by-the-Method-of-Manufactured
- ASME V&V 20-2009 (R2021), "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer". https://webstore.ansi.org/standards/asme/asme2020092021
- Oberkampf & Roy, "Verification and Validation in Scientific Computing", Cambridge University Press (2010), ISBN 9780521113601. https://books.google.com/books/about/Verification_and_Validation_in_Scientifi.html?id=7d26zLEJ1FUC
- "Accurate spectral numerical schemes for kinetic equations with energy diffusion", J. Comput. Phys. (2015). https://arxiv.org/pdf/1402.2971 ; https://www.sciencedirect.com/science/article/abs/pii/S0021999115001941
- "Pseudo spectral collocation with Maxwell polynomials for kinetic equations with energy diffusion". https://arxiv.org/pdf/1708.09031
- "A Spectral Transform Method for Singular Sturm-Liouville Problems with Applications to Energy Diffusion in Plasma Physics", SIAM J. Appl. Math. https://dx.doi.org/10.1137/130941948
- Tebbens & Tůma, "Efficient preconditioning of sequences of nonsymmetric linear systems", SIAM J. Sci. Comput. 29(5), 1918 (2007); matrix-free follow-up NLAA (2010). https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.695
- Hicken & Zingg, "A simplified and flexible variant of GCROT for solving nonsymmetric linear systems", SIAM J. Sci. Comput. 32(3), 1672 (2010). https://epubs.siam.org/doi/abs/10.1137/090754674
- Simoncini & Szyld, "Theory of inexact Krylov subspace methods and applications to scientific computing", SIAM J. Sci. Comput. 25, 454 (2003). https://dx.doi.org/10.1137/s1064827502406415
- Saunier et al., "Peclet-robust H-matrix approximation of convection-dominated inverses by convection tubes" (2025 preprint). https://arxiv.org/abs/2512.04824
- Claus, Ghysels, Boukaram & Li, "GPU-accelerated block low-rank sparse direct solvers in STRUMPACK", Int. J. HPC Appl. (2025). https://journals.sagepub.com/doi/10.1177/10943420241288567
- SFINCS v3 `sparsify.F90` (`threshholdForInclusion = 1d-12`) and `xGrid.F90` (QUADPACK Rosenbluth response). https://github.com/landreman/sfincs/tree/master/fortran/version3

- A. R. Curtis, M. J. D. Powell & J. K. Reid, [On the estimation of sparse Jacobian matrices](https://doi.org/10.1093/imamat/13.1.117), *J. Inst. Maths Applics* 13, 117 (1974): recovery from products with groups of columns that share no row.
- T. F. Coleman & J. J. Moré, [Estimation of sparse Jacobian matrices and graph coloring problems](https://doi.org/10.1137/0720013), *SIAM J. Numer. Anal.* 20, 187 (1983).
- P. R. Amestoy, I. S. Duff, J.-Y. L'Excellent & J. Koster, [A fully asynchronous multifrontal solver using distributed dynamic scheduling](https://doi.org/10.1137/S0895479899358194), *SIAM J. Matrix Anal. Appl.* 23, 15 (2001): MUMPS.
- I. S. Duff & J. Koster, [On algorithms for permuting large entries to the diagonal of a sparse matrix](https://doi.org/10.1137/S0895479899358443), *SIAM J. Matrix Anal. Appl.* 22, 973 (2001): the scaling and permutation a complete factorization applies and a drop-tolerance one lacks.
- O. Sauter, C. Angioni & Y. R. Lin-Liu, [Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime](https://doi.org/10.1063/1.873240), *Phys. Plasmas* 6, 2834 (1999): the tokamak limit used as a code-independent check.
- H. Sugama & S. Nishimura, [How to calculate the neoclassical viscosity, diffusion, and current coefficients in general toroidal plasmas](https://doi.org/10.1063/1.1512917), *Phys. Plasmas* 9, 4637 (2002): monoenergetic coefficients with momentum correction, the Tier 1 closure.

### Research-software practice
- Google Engineering Practices, "Small CLs" — https://google.github.io/eng-practices/review/developer/small-cls.html
- SmartBear, "Best Practices for Peer Code Review" (Cisco study) — https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/
- Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC '15, DOI 10.1145/2807591.2807644 — https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
- JAX documentation, "Benchmarking JAX code" — https://docs.jax.dev/en/latest/benchmarking.html (and FAQ https://docs.jax.dev/en/latest/faq.html)
- Wilson et al. 2017, "Good enough practices in scientific computing", DOI 10.1371/journal.pcbi.1005510 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510
- Taschuk & Wilson 2017, "Ten simple rules for making research software more robust", DOI 10.1371/journal.pcbi.1005412 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412
- JOSS review criteria — https://joss.readthedocs.io/en/latest/review_criteria.html
- Diátaxis — https://diataxis.fr/
- Graphite, "Stacked diffs" — https://graphite.com/guides/stacked-diffs
- Trunk Based Development — https://trunkbaseddevelopment.com/
- ADR GitHub organization — https://adr.github.io/
- FAIR4RS principles, RDA output page (metadata only), DOI 10.15497/RDA00068 — https://www.rd-alliance.org/group_output/fair-principles-for-research-software-fair4rs-principles/
- Conlin & Landreman, yancc, arXiv:2607.20861 — https://arxiv.org/html/2607.20861v1
- Escoto et al., MONKES, Nucl. Fusion 64, 076030 (2024), arXiv:2312.12248 — https://arxiv.org/html/2312.12248
- Panici et al., DESC Part I, JPP 2023, arXiv:2203.17173 — https://arxiv.org/abs/2203.17173
- READMEs (raw): DESC https://raw.githubusercontent.com/PlasmaControl/DESC/master/README.rst ; simsopt https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/README.md ; Diffrax https://raw.githubusercontent.com/patrick-kidger/diffrax/main/README.md
- Roache, "Code Verification by the Method of Manufactured Solutions", ASME J. Fluids Eng. 124, 4 (2002) — https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791
- Roy, "Review of Code and Solution Verification Procedures for Computational Simulation", JCP — https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/cjr_jcp.revise.final-accepted.pdf
- Oberkampf & Roy, Verification and Validation in Scientific Computing, CUP 2010 — https://www.cambridge.org/core/books/abs/verification-and-validation-in-scientific-computing/index/EE029CB068531D278AB2631911F8BE42
- Velasco et al., KNOSOS, J. Comput. Phys. 418, 109512 (2020) — https://www.sciencedirect.com/science/article/abs/pii/S0021999120302862 ; code https://github.com/joseluisvelasco/KNOSOS
- Landreman, Smith, Mollén, Helander, Phys. Plasmas 21, 042503 (2014) (SFINCS) — https://pubs.aip.org/aip/pop/article-abstract/21/4/042503/818401 ; https://github.com/landreman/sfincs
- Whitesides, "Whitesides' Group: Writing a Paper", Adv. Mater. 16, 1375 (2004), DOI 10.1002/adma.200400767 — https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper
- The Turing Way, "Software Citation with CITATION.cff" — https://book.the-turing-way.org/communication/citable/citable-cff/ ; Citation File Format — https://citation-file-format.github.io/
- ACM Artifact Review and Badging v1.1 — https://www.acm.org/publications/policies/artifact-review-and-badging-current (HTTP 403)
- FAIR4RS principle wording; Chue Hong et al., Sci. Data 9, 622 (2022), DOI 10.1038/s41597-022-01710-x (Nature redirect loop)
- ASME V&V 20-2009 scope statement (snippet only)
- GENE / GS2 / COGENT verification suites; Google test-size taxonomy; Keep a Changelog; CPC "Program summary" requirement; `jax.test_util.check_grads`
