# SFINCS_JAX Final Research-Grade Implementation Plan

Last updated: 2026-06-25 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review surface: PR #8, `refactor/v3-driver-architecture`, draft until the
Lane 1 consolidation acceptance gates pass

Status: this file is the controlling completion plan. `plan.md` remains the
execution log and historical record; future work should update this file only
when the target plan itself changes. The Lane 1 consolidation plan below
supersedes older iteration notes and is the only refactor plan to follow; avoid
new one-helper refactor tranches.

## One-Sentence Goal

Finish `sfincs_jax` as a compact, domain-organized, production-grade
neoclassical transport code: users provide a geometry and input file and get
accurate CPU/GPU results with automatic robust solver selection, while Python
users can opt into end-to-end differentiable residual, flux, ambipolar, and
optimization workflows with parity against SFINCS Fortran v3 wherever the
physics models overlap.

## Evidence Reviewed

### SFINCS Fortran v3 Source And Documentation

Local reference: `/Users/rogeriojorge/local/sfincs/fortran/version3`

Manual and paper references:

- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/equations.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/runs.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/inputParameters.tex`
- `/Users/rogeriojorge/local/sfincs/doc/manual/version3/outputParameters.tex`
- `/Users/rogeriojorge/local/sfincs/doc/sfincsPaper/sfincsPaper.tex`
- Public repository: <https://github.com/landreman/sfincs>

Reference decks and probe summaries for the ambipolar functionality live in
`benchmarks/fortran_v3_ambipolar_reference/`. These decks pin the source-code
behavior of `ambipolarSolveOption=1`, `2`, and `3` before the sfincs_jax
implementation begins.

Implementation progress on 2026-06-23:

- `sfincs_jax.problems.ambipolar` now owns the first-class ambipolar problem,
  iteration, and result contracts.
- The Fortran-compatible Brent root path is implemented and tested against the
  checked-in small and production Fortran v3 ambipolar summaries.
- The Fortran v3 ambipolar reference matrix now covers geometry-1 helical and
  geometry-4 W7-X-like decks at small and production tiers for options 1, 2,
  and 3, with PETSc/KSP/MUMPS profiling markers and `/usr/bin/time -lp` RSS
  summaries.
- The source-code validator for Fortran-compatible ambipolar restrictions is
  implemented for the reference decks and derivative-assisted option guards.
- The Brent owner can now evaluate real RHSMode 1 radial currents through
  in-process `write_sfincs_jax_output_h5` calls, records per-evaluation
  artifacts, and is exposed through `sfincs_jax ambipolar`.
- Each real ambipolar evaluation now records solver-trace provenance
  (selected path, residual, target, setup/solve/elapsed time, active size),
  uses an ambipolar-local geometry/output cache across `E_r` evaluations, and
  carries a shape-checked Krylov state file for warm starts/recycled basis
  reuse between nearby electric-field solves.
- RHSMode 1 active field-split symbolic orderings now use a package-level
  semantic fixed-shape key, so structured CSR and true-operator rescue paths
  can reuse active/full index maps across same-shape solves without reusing
  stale `E_r`-dependent numerical matrices or factors.
- A finite-difference `dJr/dEr` helper now provides the numerical derivative
  gate that implicit/adjoint derivatives must match.
- `sfincs_jax.sensitivity` now owns a reusable implicit linear-observable
  derivative certificate. It solves both the tangent equation and the adjoint
  equation, reports primal/tangent/adjoint residuals, and can compare the
  derivative against centered finite differences.
- The same module now exposes a fixed-shape `LinearObservableSystem` builder
  bridge. Concrete RHSMode 1/4/5 owners can provide the true operator, RHS,
  observable, and their scalar-parameter derivatives without coupling the
  ambipolar root solvers back to `v3_driver.py`.
- A bounded `probe_linear_observable_vector` helper can recover `c` and `J0`
  from an existing linear diagnostic `J(x) = c^T x + J0` on small validation
  decks. This is a validation bridge for pinning radial-current weights before
  replacing it with analytic production weights.
- `sfincs_jax.problems.transport_matrix.diagnostics` now exposes
  `radial_current_vm_psi_hat_from_state` and a chunked observable-vector helper
  for the magnetic-drift radial-current contribution, plus explicit
  `psiHat`/`rHat`/`rN` coordinate wrappers. A tiny RHSMode-1 deck checks the
  recovered vector against the existing diagnostic.
- `sfincs_jax.problems.ambipolar` now includes a size-limited dense RHSMode-1
  linear-observable builder that assembles the true small-deck operator/RHS,
  finite-differences their scalar-parameter derivatives, and feeds the result
  to the tangent/adjoint certificate. This pins the wiring contract before a
  production sparse/matrix-free implementation.
- `sfincs_jax.sensitivity` now also exposes a matrix-free
  `MatrixFreeLinearObservableSystem` certificate. Production owners can pass
  operator actions, transpose actions, derivative actions, and selected
  solve/transpose-solve closures without dense assembly. Ambipolar adapters
  route this certificate to the same `dJr/dEr` contract used by option-1/3
  Newton solvers.
- `sfincs_jax.problems.ambipolar` now exposes the first concrete RHSMode-1
  matrix-free radial-current builder. It uses the real full-system matrix-free
  operator action, caller-supplied transpose and solve closures, finite-
  difference operator/RHS derivative actions, and existing radial-current
  observable weights; this removes dense matrix assembly from the builder
  itself while keeping a finite-difference promotion gate.
- The same builder now accepts caller-supplied derivative actions or JAX
  operator tangents. `operator_tangent_from_centered_difference` constructs
  valid pytrees with `float0` tangents for integer/bool leaves, and a real
  electric-field `xDot` operator test verifies the JVP action against centered
  differences.
- `matrix_free_radial_current_derivative_provider` now bridges matrix-free
  implicit certificates into safeguarded Newton/bisection and pure Newton
  ambipolar root solvers. Fast option-1/3-style tests verify root convergence,
  derivative metadata, tangent/adjoint consistency, and finite-difference
  agreement through the public solver API.
- No-Phi1 existing-branch `Er` operator tangents now use the v3 radial
  conversion analytically. The helper updates stored `dphi_hat_dpsi_hat` leaves
  in the full operator and f-block suboperators, and a real electric-field
  `xDot` fixture verifies the JVP action against centered operator differences.
- `keep_zero_er_terms` now lets derivative gates retain zero-valued ExB and
  `Er` suboperators at `Er=0` without changing normal solve defaults. A real
  `xDot` fixture verifies that the opt-in zero-`Er` operator is numerically
  identical to the default operator, while its analytic JVP tangent matches
  nearby nonzero centered differences.
- `rhsmode1_radial_current_response_from_namelist` now provides the first
  namelist-backed RHSMode-1 radial-current response and derivative provider.
  A bounded real-deck gate uses fixed-shape zero-`Er` branches, the analytic
  JVP operator tangent, and dense validation closures to compare the implicit
  derivative against centered finite differences without exposing users to
  manual plus/minus operator assembly.
- The namelist-backed RHSMode-1 response now validates on Fortran-style active
  pitch-mode DOFs instead of the rectangular inactive-mode storage. It defaults
  to the Fortran ambipolar `particleFlux_vm_rN` current convention, infers the
  radial conversion from the namelist, and replays the checked
  `geometry1_helical_small_option1` Fortran option-1 current and Newton slope
  within `2e-5` relative tolerance.
- The same active provider also replays the checked small option-3 physical
  currents for `geometry1_helical_small_option3` and
  `geometry4_w7x_like_small_option3` within `2e-5` relative tolerance.
- `solve_rhsmode1_ambipolar_from_namelist` now wires that active provider into
  the real option-1/2/3 ambipolar root policies. Bounded small-deck helical
  option-1 and option-3 roots replay the checked Fortran v3 roots using the
  active `particleFlux_vm_rN` response rather than a synthetic table.
- `sfincs_jax.sensitivity` now exposes `jvp_flux`, `vjp_flux`, and
  `adjoint_dot_product_check`. The focused tests apply the dot-product identity
  to real RHSMode-1 particle-flux, heat-flux, flow, radial-current, and
  bootstrap-current diagnostics, which is the same consistency gate required
  before promoting RHSMode 4/5 adjoint outputs.
- `sfincs_jax.sensitivity` also owns the first RHSMode-4/5 Fortran source
  contract helpers. `validate_fortran_v3_adjoint_sensitivity_constraints`
  mirrors the `validateInput.F90` adjoint restrictions, and
  `fortran_v3_adjoint_sensitivity_output_fields` pins the HDF5 sensitivity
  fields written by `writeHDF5Output.F90`, including the source-code
  `dParallelFlowdLambda` gate.
- `benchmarks/fortran_v3_sensitivity_reference` now contains compact numerical
  RHSMode-4/5 Fortran summaries for tiny W7-X-like analytic radial-current,
  heat-flux, parallel-flow, bootstrap, and debug finite-difference sensitivity
  decks. Checked tests pin the HDF5 field names, tensor ranks, wall/RSS budgets,
  `dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`,
  `dTotalHeatFluxdLambda = sum_s dHeatFlux_s/dLambda`,
  `dBootstrapdLambda = sum_s Z_s dParallelFlow_s/dLambda`, the RHSMode-5
  `dPhidPsidLambda` constant-current output, and selected debug-adjoint
  finite-difference outputs without committing generated HDF5 files.
- Derivative-assisted safeguarded Newton/bisection and strict pure-Newton root
  solvers are implemented behind the same ambipolar owner. They accept a
  derivative provider, so finite-difference gates, direct implicit
  certificates, and builder-backed implicit certificates can be wired in
  without changing the root-solve contract.
- `docs/feature_matrix.rst` now records the audited Fortran-v3 feature owners,
  the matching `sfincs_jax` implementation owners, and the promotion gates for
  ambipolar option 1/3, RHSMode 4/5 sensitivities, solver backends, geometry,
  Phi1, outputs, and parallelism.
- Remaining Lane 3 work is deeper fixed-shape numerical operator/factor and
  preconditioner setup reuse behind that evaluator, plus larger-deck physical
  replay gates that run the namelist-backed derivative provider through real
  ambipolar root solves.

Important Fortran v3 implementation modules:

| Module | Functionality to mirror or compare |
| --- | --- |
| `sfincs_main.F90`, `sfincs.F90` | Program initialization, MPI/PETSc lifecycle, execution mode selection. |
| `readInput.F90`, `validateInput.F90` | Namelist schema, defaults, compatibility guards, automatic coercions. |
| `createGrids.F90`, `xGrid.F90`, `uniformDiffMatrices.F90`, `polynomialDiffMatrices.F90` | Velocity and angle grids, interpolation, differentiation matrices, monoenergetic overrides. |
| `geometry.F90`, `updateBoozerGeometry.F90`, `radialCoordinates.F90` | Geometry schemes 1, 2, 3, 4, 5, 11, 12, 13 and radial-coordinate conversions. |
| `populateMatrix.F90`, `evaluateResidual.F90`, `evaluateJacobian.F90`, `preallocateMatrix.F90`, `sparsify.F90` | Exact residual/Jacobian/Pmat construction, sparse preallocation, simplified preconditioner matrix. |
| `solver.F90` | SNES/KSP orchestration, direct/iterative choices, MUMPS/SuperLU_DIST/PETSc factor controls, adjoint solves. |
| `ambipolarSolver.F90` | In-solver ambipolar root solve using Brent, safeguarded Newton/bisection, or pure Newton. |
| `populateAdjointRHS.F90`, `populatedMatrixdLambda.F90`, `populatedRHSdLambda.F90`, `adjointDiagnostics.F90` | RHSMode 4/5 adjoint sensitivity system, `dL/dlambda f - dS/dlambda`, `dRadialCurrentdEr`. |
| `diagnostics.F90`, `writeHDF5Output.F90`, `export_f.F90` | Fluxes, flows, bootstrap current, Phi1 diagnostics, KSP/SNES status, HDF5 outputs, exported distribution functions. |
| `classicalTransport.F90` | Classical transport diagnostics. |
| `testingAdjointDiagnostics.F90` | Finite-difference checks for adjoint sensitivity implementation. |

Key Fortran v3 algorithm facts to preserve:

- The most general system solves for
  `{f_s1(theta,zeta,x,xi), Phi1(theta,zeta), S_s1, S_s2, lambda}`. Without
  self-consistent `Phi1`, the unknowns reduce to `{f_s1, S_s1, S_s2}`.
- RHSMode 1 solves one physical right-hand side. RHSMode 2 solves three
  right-hand sides for the energy-integrated transport matrix. RHSMode 3 solves
  two monoenergetic right-hand sides with its special `Nx=1` grid convention.
- RHSMode 4 computes adjoint sensitivities at fixed `E_r`. RHSMode 5 computes
  adjoint sensitivities at ambipolar `E_r`.
- `ambipolarSolveOption=1` uses safeguarded Newton/bisection,
  `ambipolarSolveOption=2` uses Brent, and `ambipolarSolveOption=3` uses pure
  Newton.
- Newton-based ambipolar solve options compute `dRadialCurrentdEr` with an
  adjoint solve. The derivative is based on
  `evaluateAdjointInnerProductFactor`, which forms
  `dL/dEr * f - dS/dEr`, followed by the free-energy inner product with the
  adjoint radial-current solution.
- Fortran v3 has a source/manual discrepancy for derivative-assisted
  ambipolar solves: the manual says options 1 and 3 require
  `magneticDriftScheme > 0`, but `validateInput.F90` rejects tangential
  magnetic drifts for `RHSMode>3` and for `ambipolarSolve=.true.` with
  `ambipolarSolveOption != 2`. The implementation plan follows the source:
  options 1 and 3 are validated first with `magneticDriftScheme == 0`, then
  any later tangential-drift extension must be an explicit new sfincs_jax
  capability with its own tests.
- The linear/nonlinear solver stack is PETSc SNES/KSP. Linear cases use one
  SNES step. `includePhi1=true` with self-consistent `Phi1` uses Newton line
  search and nested KSP solves.
- Iterative solves use GMRES with a factorized `PCLU` preconditioner. Direct
  solves use `KSPPREONLY + PCLU`.
- When parallel direct solvers are available, the factorization backend is
  MUMPS or SuperLU_DIST. Serial PETSc direct solves use a sparse ordering,
  typically RCM in v3, with diagonal-pivot safeguards.
- MUMPS robustness is improved by pivot threshold controls and by retrying with
  larger `ICNTL(14)` workspace when factorization fails. This is a solver
  policy and residual-admission pattern that `sfincs_jax` should mimic
  natively, not by depending on PETSc.
- For RHSMode 2/3, Fortran v3 builds the operator once, reuses the factorized
  operator/preconditioner across the right-hand sides, and only changes the
  drive terms.
- For adjoints, Fortran v3 reuses the transpose operator path through either a
  separately built adjoint matrix or `KSPSolveTranspose`.

### Ambipolar Probe Results From Fortran v3

Completed local probes on 2026-06-22 and 2026-06-23 used
`/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs` and the small decks
now checked into `benchmarks/fortran_v3_ambipolar_reference/namelists`. The
new reproducible summaries are
`small_profile_summary_2026-06-23.json` and
`production_profile_summary_2026-06-23.json`; raw PETSc/MUMPS logs stay in the
scratch paths recorded in those JSON files.

Observed facts to feed directly into implementation:

- `geometry4_w7x_like_small_option1`: safeguarded Newton/bisection completed
  with Er evaluations `[-20, 20, 0]`, radial currents about
  `[-1.09e-6, 1.43e-6, 2.51e-8]`, internal Fortran ambipolar time
  `0.143 s`, wall time `75.31 s`, and peak RSS about `135 MB`.
- `geometry4_w7x_like_small_option2`: Brent completed with the same bracket and
  initial point, internal Fortran ambipolar time `0.147 s`, wall time
  `75.66 s`, and peak RSS about `135 MB`.
- `geometry4_w7x_like_small_option3`: pure Newton completed from the initial
  guess, internal Fortran ambipolar time `0.063 s`, wall time `75.19 s`, and
  peak RSS about `120 MB`.
- `geometry1_helical_small_option2`: Brent completed on a distinct analytic
  helical geometry with Er values `[-20, 20, 0, -1.7273, -2.0106]` and final
  radial current about `1.7e-9`; internal Fortran ambipolar time was `0.373 s`,
  wall time was `75.54 s`, peak RSS was about `156 MB`, and the run reported an
  MPI finalization error after writing useful diagnostics.
- `geometry1_helical_small_option1` and `geometry1_helical_small_option3` now
  pin the same analytic helical problem for derivative-assisted Newton paths.
  Option 1 used four forward solves and four adjoint solves; option 3 used two
  forward solves and two adjoint solves. Both reached `|J_r| ~ 1.1e-9`.
- All small probes used MUMPS through PETSc. The logs show repeated full
  `whichMatrix=0` and `whichMatrix=1` setup per Er evaluation, followed by
  residual matrix assembly and diagnostics.
- For these tiny systems, Jacobian sizes alternated between about `19,566` and
  `26,750` nonzeros depending on the Er branch/active terms, while the
  preconditioner had about `15,436`, `19,356`, or `26,750` nonzeros depending
  on deck and preconditioner settings.
- `geometry4_w7x_like_production_option{1,2,3}` completed the larger reference
  deck with `Ntheta=13`, `Nzeta=19`, `Nxi=48`, and `Nx=5`. All three options
  converged to the same root near `Er=-3.57735`. Option 1 used six forward and
  six adjoint solves, option 2 used six forward solves, and option 3 used four
  forward and four adjoint solves. The profiled runs reported internal
  ambipolar times of about `18.7 s`, `13.8 s`, and `12.1 s`, respectively,
  and peak RSS around `1.35-1.39 GB`.
- `geometry1_helical_production_option1` and
  `geometry1_helical_production_option3` converged to `Er=-3.26189` with final
  `|J_r| ~ 6.5e-11`. The same production grid is substantially harder than the
  W7-X-like deck: peak RSS was `5.7-5.8 GB`, option 1 used ten forward plus ten
  adjoint solves, and option 3 used eight forward plus eight adjoint solves.
- `geometry1_helical_production_option2` reached `|J_r| ~ 8.1e-12` but did not
  print Fortran's Brent success marker before exhausting its 12 evaluation
  budget. This is an important policy gate: sfincs_jax should report both the
  best residual found and whether the declared convergence criterion was met,
  rather than treating a small final residual and a success marker as the same
  concept.
- The production decks used MUMPS and assembled matrices with about
  `1.57e6` to `2.28e6` Jacobian nonzeros and about `1.27e6` to `1.54e6`
  preconditioner nonzeros for W7-X-like geometry. The helical production decks
  used the same maximum Jacobian size but an exact preconditioner with up to
  `2.28e6` nonzeros, explaining their higher RSS.
- The large gap between Fortran's internal ambipolar time and process wall time
  comes from process/logging/finalization overhead and verbose MUMPS/PETSc
  diagnostics. sfincs_jax should not emulate this shell-style scan cost; it
  should keep fixed-shape geometry/operator/factor metadata alive across Er
  evaluations.

### Current sfincs_jax State

Current source size snapshot after the 2026-06-25 consolidation checkpoint and
the 2026-06-25 final consolidation-plan review:

- `sfincs_jax/v3_driver.py`: 47 lines in the current consolidation worktree,
  acting as a compatibility shim for the domain-owned solve modules.
- `sfincs_jax/problems/profile_response/solve.py`: 9,730 lines after the
  first profile-response ownership moves. This remains the largest
  structural debt and must be reduced by moving coherent sections into existing
  domain owners, not by adding many new helper files.
- `sfincs_jax/problems/profile_response/policies.py`: 6,876 lines after
  absorbing six former policy shards and current-backend RHSMode-1 policy
  wrappers. This is intentionally a temporary single policy owner during
  Tranche 1; do not split it back into micro-files.
- `sfincs_jax/problems/profile_response/sparse/handoff.py`: 3,761 lines after
  moving the former top-level sparse-PC handoff into the sparse package. Shared
  sparse-PC finalization remains in
  `problems/profile_response/sparse/finalization.py` because x-block and
  Fortran-reduced sparse owners both use the same result payloads.
- `sfincs_jax/problems/profile_response/sparse/direct.py`: 3,616 lines after
  taking ownership of sparse-factor cache keys, host memory probing, explicit
  sparse-pattern probes, sparse-JAX preconditioner materialization, and host
  sparse direct builder/polish wrappers.
- `sfincs_jax/problems/profile_response/dense.py`: 2,487 lines after taking
  ownership of profile linear-solve routing, dense-KSP, constraintScheme=0
  PETSc-compatible sparse-ILU, and SciPy rescue contracts.
- `sfincs_jax/problems/profile_response/auto_solve.py`: 550 lines after taking
  ownership of the explicit host structured-CSR RHSMode-1 solve entry point.
- `sfincs_jax/problems/transport_matrix/solve.py`: 1,763 lines after the
  RHSMode 2/3 solve-entry extraction.
- `sfincs_jax/operators/profile_response/full_system.py`: about 6.0k
  lines, moved from `rhs1_full_assembly.py` in Lane 1 Iteration 2.
- Top-level `transport_*` modules: 0 after Lane 1 Iteration 1.
- Top-level `rhs1_*` modules: 0 after the Lane 1 Iteration 3 ownership move.
  Solver-family implementation now lives under `solvers.preconditioners`.
- Package total is 227 Python files and about 163k package lines after the
  first consolidation checkpoints.
- Current concentration of complexity:
  `problems/profile_response` has 21 files and about 50k lines,
  `problems/transport_matrix` has 33 files and about 15k lines,
  `solvers/preconditioners` has 51 files and about 37k lines,
  `operators/profile_response` has 12 files and about 14k lines, and
  `io.py` remains about 4.3k lines.

Useful existing assets:

- JAX-native residual/operator code already exists for large portions of the
  v3 model.
- `sfincs_jax/implicit_solve.py` already wraps `jax.lax.custom_linear_solve`
  for implicit differentiation through linear solves.
- `sfincs_jax/problems/profile_response/phi1_newton.py` already uses
  `jax.linearize` to build Newton-Krylov JVPs.
- `sfincs_jax.problems.ambipolar` provides bounded/reference Fortran-v3-style
  ambipolar option 1/2/3 root solvers, derivative-provider hooks, setup-reuse
  metadata, and checked Fortran replay gates.
- Several profile-response and transport-matrix preconditioners exist, but the
  implementation surface is still too fragmented and too environment-variable
  driven.
- Docs and examples now include autodiff and optimization demonstrations, but
  they should be recast around a small public API and explicit validation
  contracts.

Important current gaps:

- RHSMode 4/5 bounded/reference output contracts are implemented and tested
  against compact Fortran-v3 fixtures. Production-grid parity remains an
  external release-refresh benchmark.
- Ambipolar option 1/2/3 bounded/reference solvers are implemented and tested;
  production-grid replay artifacts remain outside normal CI.
- The largest active gap is now structural complexity after the driver move:
  `profile_response/solve.py` is a mechanical owner, transport and
  preconditioner helper shards are over-split, and docs still mention
  historical `rhs1_*`, `transport_*`, `linear_solve.py`, and `v3_driver.py`
  internals.
- No complete derivative API for `dGamma/dEr`, `dQ/dEr`, `d<J.B>/dEr`,
  `dJr/dEr`, profile sensitivities, and geometry harmonic sensitivities across
  all supported solve lanes.
- Too many solver choices are exposed as low-level environment variables rather
  than automatic, tested policy decisions.
- Code ownership is better than before but still not final: the target is fewer
  domain packages with larger, coherent ownership boundaries, not a growing set
  of one-off files.

### External References For Differentiation And Workflows

Sources reviewed:

- Fast automated adjoints for spectral PDE solvers:
  <https://arxiv.org/abs/2506.14792>
- JAX `custom_linear_solve` documentation:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- JAX custom JVP/VJP documentation:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- Lineax documentation:
  <https://docs.kidger.site/lineax/>
- JAXopt implicit differentiation:
  <https://jaxopt.github.io/stable/implicit_diff.html>
- T3D documentation:
  <https://t3d.readthedocs.io/>
- NEOPAX repository and local checkout:
  <https://github.com/uwplasma/NEOPAX>

Derivative design implications:

- The right default is not to backpropagate through every Krylov iteration or
  every adaptive solver branch. The robust design is an explicit residual graph
  plus implicit differentiation at the converged solution.
- The primal equation is
  `R(u, p) = 0`, with solution `u(p)`.
- Forward sensitivity should use
  `R_u du/dp = -R_p`.
- Reverse sensitivity for scalar objective `J(u, p)` should solve
  `R_u^T lambda = J_u^T`, then use
  `dJ/dp = J_p - lambda^T R_p`.
- This matches the Fortran v3 adjoint structure and the spectral-PDE adjoint
  paper's message: build adjoints from the symbolic residual/operator graph and
  reuse efficient sparse/direct/iterative solves.
- `jax.lax.custom_linear_solve` is the base primitive for differentiating
  linear solve calls without tracing all iterations.
- `custom_jvp` and `custom_vjp` should wrap only stable public solve APIs and
  only when the default JAX transform would trace control flow or host-only
  operations.
- Lineax can be evaluated as an optional operator abstraction or least-squares
  backend, but it should be adopted only if benchmarks show lower compile time,
  lower runtime, lower memory, or cleaner derivative code for a concrete lane.
- T3D and NEOPAX need a profile/flux closure interface, not a file-only
  executable. `sfincs_jax` should expose shape-stable JAX-friendly functions for
  radial batches, Er scans, transport coefficients, ambipolar roots, and
  Jacobian/vector-Jacobian products.

## Product Requirements

### Default User Experience

- CLI users provide `input.namelist` and a geometry path. The default solver
  policy selects a robust and efficient method automatically.
- The terminal output reports phase timings, selected solver path, residual
  target, residual history summary, memory estimate, output file paths, and
  enough progress to judge whether a run will take seconds, minutes, or longer.
- Advanced users can override solver family, preconditioner, differentiability
  mode, output format, and profiling options, but routine runs should not
  require environment variables.

### Python And Optimization Experience

- Python users can call a compact public API:
  `solve_flux_surface`, `solve_transport_matrix`, `solve_ambipolar`,
  `make_flux_closure`, and `make_objective`.
- Differentiable mode is explicit and documented. It returns pytrees with
  solution, diagnostics, residual certificates, and derivative capabilities.
- Non-differentiable mode can use faster host/direct paths when called from the
  CLI or from Python with `differentiable=False`.
- The API should support optimization workflows such as:
  bootstrap-current minimization, ambipolar electron-root targeting, particle
  and heat flux minimization, impurity flux targeting, sensitivity analysis,
  inverse design, uncertainty quantification, and profile evolution coupling.

### Research-Grade Claim Boundary

Public claims require all of the following:

- Numerical residual passes the true operator residual gate.
- Shared outputs match SFINCS Fortran v3 within documented tolerances.
- CPU and GPU either match within tolerance or the difference is documented and
  gated.
- Runtime and peak memory are measured from fresh cold and warm runs.
- The exact resolution, device, commit, Python/JAX versions, and solver path
  are recorded.
- For derivative claims, JVP/VJP/adjoint results pass finite-difference checks
  on stable windows and transpose/adjoint residual checks.
- For optimization examples, the plotted improvement must come from a
  reproducible script, not a hand-edited artifact.

## Final Architecture Target

The final codebase should have a small number of domain packages. New files are
allowed only when they create a clear owner boundary. One-function wrappers and
historical compatibility facades should be deleted as their callers migrate.

Target package boundaries:

| Package | Ownership |
| --- | --- |
| `sfincs_jax.api` | Public Python API, stable return objects, high-level solve entry points. |
| `sfincs_jax.cli` | CLI parsing, user-facing progress, plotting, profiling switches. |
| `sfincs_jax.config` | Namelist/TOML parsing, defaults, validation, Fortran-compatible coercions. |
| `sfincs_jax.geometry` | Analytic, VMEC, Boozer, Miller, `vmec_jax`, and `booz_xform_jax` adapters. |
| `sfincs_jax.discretization` | Grids, quadrature, interpolation, finite differences, radial-coordinate conversion. |
| `sfincs_jax.operators` | Drift-kinetic residuals, Jacobians, RHS builders, collision/source/Phi1 blocks. |
| `sfincs_jax.solvers` | Linear/nonlinear solve policies, native factors, Krylov, preconditioners, implicit solves. |
| `sfincs_jax.problems` | Problem orchestration for profile response, transport matrices, monoenergetic, ambipolar, adjoint sensitivity. |
| `sfincs_jax.outputs` | HDF5/netCDF/NPZ writing, restart/recycle state, diagnostics schema, plotting payloads. |
| `sfincs_jax.sensitivity` | JVP/VJP, implicit differentiation, adjoint diagnostics, finite-difference checks. |
| `sfincs_jax.workflows` | Radial scans, optimization objectives, T3D/NEOPAX closures, publication figures. |
| `sfincs_jax.validation` | Parity reports, benchmark harnesses, physics gates, artifact policies. |

Structural rules:

- Keep domain files under about 1500 lines when practical. Exceptions require a
  short owner comment at the top explaining why the file is intentionally large.
- Prefer fewer coherent owner files over many small wrappers.
- Do not keep duplicate names like `v3_*`, `rhs1_*`, and `transport_*` when a
  domain name is clearer.
- Compatibility shims get a deletion issue/date in this plan and must have a
  focused import-contract test while they exist.
- Every moved owner boundary gets at least one test that imports the canonical
  owner and one test that exercises behavior through the public API.

## Lane 1 - Final Consolidation Pass

Goal: finish the refactor PR with fewer files, fewer names, and clearer domain
ownership without changing solver behavior. This is the only refactor plan to
follow. Do not add new one-helper modules or start side plans; every refactor
commit in this lane must finish one of the consolidation tranches below.

Current inventory from the 2026-06-25 final plan review:

| Area | Current state | Final target for this PR |
| --- | --- | --- |
| `sfincs_jax/v3_driver.py` | 47-line compatibility shim | Keep under 80 lines or delete after legacy imports migrate. |
| Package source files | 227 Python files, about 163k package lines | Below 205 files without deleting tested functionality. |
| `problems/profile_response` | 21 files, about 50k lines; `solve.py` is 9.7k lines | At most 18 files; `solve.py` below 3.5k lines; no recreated policy shards. |
| `problems/transport_matrix` | 33 files, about 15k lines; solve-loop and parallel micro-files dominate file count | At most 18 files; one solve owner, one policy owner, one diagnostics/finalization owner, compact parallel ownership. |
| `solvers/preconditioners` | 51 files, about 37k lines; QI and RHSMode-1 names remain over-fragmented | At most 36 files; QI organized by role, no implementation file starts with `rhs1_`. |
| `io.py` / `outputs` | `io.py` is 4.3k lines; output schema helpers already live in `outputs` | `io.py` below 800 lines as a compatibility shim, with real writers/readers in `outputs`. |
| Docs/API maps | 44 docs pages; several still document transient internal names | Docs expose public API and stable domain owners only. |
| Tests | 329 test files; many still import compatibility or historical owners directly | Tests import canonical owners except focused compatibility-contract tests. |

Source-tree findings from the final review:

- `profile_response/solve.py` is still the central bottleneck. Sparse-direct
  cache/materializer helpers have moved to `sparse/direct.py`, and
  current-backend policy wrappers have moved to `policies.py`; the next move is
  a larger solve-flow relocation into existing handoff/diagnostic owners plus
  sparse branch orchestration into sparse owners.
- `problems/transport_matrix` has a clear micro-file cluster:
  dense/direct helpers, host GMRES, iteration statistics, residual quality,
  solve policy, handoff policy, and parallel runtime shards. These can be
  collapsed without changing physics.
- `solvers/preconditioners` is over-fragmented by historical experiment names.
  The target is role-based owners: profile-response Schur, symbolic sparse,
  QI basis/correction/device/policy, PAS, x-block, and transport-matrix.
- `io.py` still owns too much real logic. Output writing and reading should be
  owned by `outputs`; `sfincs_jax.io` should become a small compatibility layer.
- The docs and tests need to follow the code move in the same tranche. A
  refactor commit is incomplete if source imports move but docs/source maps and
  canonical-owner tests still point to stale names.

Consolidation discipline:

- Each tranche must delete at least three files, or delete at least two files
  and reduce one large owner by at least 1,500 lines. Do not commit
  one-function moves unless they fix a failing test.
- New implementation files are allowed only if the same commit deletes or
  merges more files than it adds, and the new file has a durable domain name.
- Do not change physics formulas, tolerances, default solver choices, output
  schemas, benchmark claims, or public API semantics in this lane.
- Private compatibility aliases should be deleted in the same tranche that
  migrates callers. Public compatibility shims need a focused import-contract
  test while they remain.
- Before deleting code, verify imports with `rg`, run focused tests for the
  affected owner, run `py_compile`, run scoped `ruff`, and keep docs building
  with `-W`.

### Tranche 0 - Freeze The Boundary

Status: completed through the current clean branch. The latest validated
checkpoint folded `active_projection.py`, `qi_device_seed.py`, and
`strong_preconditioning.py` into canonical owners and reduced
`problems/profile_response` from 24 files to 21 files.

Completed boundary facts:

- Top-level `rhs1_*` and `transport_*` implementation files are gone.
- `v3_driver.py` is a small compatibility shim.
- `profile_response/linear_solve.py`, `transport_matrix/linear_solve.py`,
  former profile-response policy shards, top-level profile-response
  finalization/KSP shards, `sparse/krylov.py`, and top-level
  `profile_response/sparse_pc.py` are gone.
- The canonical sparse-PC handoff owner is
  `sfincs_jax.problems.profile_response.sparse.handoff`.
- `sfincs_jax.v3_driver` intentionally resolves to the profile-response solve
  compatibility surface and is covered by import-contract tests while it
  remains.

Exit gate:

- Keep the branch clean before each new tranche. If a tranche leaves broad lint
  ignores, stale docs imports, or temporary outputs, it is not complete.

### Tranche 1 - Profile-Response Sparse And Solve Collapse

This is the highest-priority consolidation because `profile_response/solve.py`
is still the largest non-reviewable file. This tranche must move whole
responsibility blocks into existing owners; it must not create new
profile-response helper files.

Completed inside Tranche 1 as of 2026-06-25:

1. Host sparse direct setup, sparse cache-key construction, active sparse
   pattern probing, sparse-JAX preconditioner materialization, and host sparse
   direct polish now live in `profile_response/sparse/direct.py`.
2. Current-backend RHSMode-1 dense/sparse/PAS/x-block admission wrappers now
   live in `profile_response/policies.py`; `solve.py` only imports the legacy
   private aliases needed by older call paths and compatibility tests.

Remaining move/delete targets:

1. Collapse duplicated env-token parsing. Keep one bool/int/float parser family
   and dataclass policy contracts in `policies.py`; delete duplicated parsers
   in `sparse/xblock.py`, `sparse/qi.py`, and sparse policy helpers when they
   can import the shared parser without circular dependencies.
2. Move sparse-PC branch orchestration from `solve.py` into the existing
   sparse owners:
   `sparse/handoff.py` for sparse-PC GMRES handoff,
   `sparse/xblock.py` for x-block stages,
   `sparse/qi.py` for QI/device stages,
   `sparse/fortran_reduced.py` for Fortran-reduced paths, and
   `sparse/finalization.py` only if it is still shared by more than one owner.
3. Move result payload assembly, progress-line replay, and final diagnostic
   normalization from `solve.py` into `handoff.py` and
   `solver_diagnostics.py`.
4. Delete compatibility aliases inside `solve.py` after tests and callers
   import the canonical owners. If `sfincs_jax.v3_driver` still needs a name,
   import it into the shim from the canonical owner rather than keeping its
   implementation in `solve.py`.
5. Leave `solve.py` with only public solve entry points, phase sequencing, and
   dependency injection. It should not own policy parsing, sparse pattern
   builders, QI stages, dense KSP/SciPy rescue internals, output schema, or
   final payload normalization.

Expected result:

- `problems/profile_response` drops from 21 to at most 18 files. Deletion
  candidates after call-graph checks are `sparse/finalization.py` if it becomes
  single-owner, `handoff.py` if it becomes only payload assembly, and any
  re-export-only sparse policy file left after sparse policy migration.
- `profile_response/solve.py` drops from 9.7k lines below 3.5k lines.
- `policies.py` may remain large, but it must no longer duplicate parser
  helpers or recreate the deleted policy-shard structure.

Exit gates:

- Focused tests pass for RHSMode 1 solve routing, dense fallback,
  sparse-PC handoff, direct-tail policy, host policy, QI admission, ambipolar,
  sensitivity, and output diagnostics.
- `python -m py_compile` passes for `problems/profile_response`,
  `operators/profile_response`, and `solvers/preconditioners`.
- Scoped `ruff` passes for touched modules and tests.
- `docs/api.rst` and `docs/source_map.rst` contain canonical
  profile-response owners only.

### Tranche 2 - Transport-Matrix And Output Collapse

This tranche should be one broad pass, not a sequence of small solve-loop
moves. The goal is to make RHSMode 2/3 reviewable and remove the remaining
output split between `io.py` and `outputs`.

Move/delete targets:

1. Merge `dense_lu.py`, `host_gmres.py`, `iteration_stats.py`, `loop.py`,
   `residual_quality.py`, `solve_policy.py`, and `handoff_policy.py` into
   `transport_matrix/solve.py` or `transport_matrix/policies.py`, depending on
   whether the code is phase execution or policy selection.
2. Merge `postsolve_diagnostics.py` into `finalize.py`.
3. Move `streaming_outputs.py` into `outputs/transport.py`; transport output
   streaming is an output responsibility.
4. Merge `active_factor.py` and `active_dense.py` into a single active-system
   owner. Prefer `active_dense.py` only if it remains the dominant import path;
   otherwise rename through delete/add only after callers are migrated.
5. Collapse `problems/transport_matrix/parallel` to three implementation
   files:
   `runtime.py` for process/device orchestration and subprocess collection,
   `worker.py` for payload execution,
   `sharding.py` for array partitioning if its tests justify separation.
   Delete `execution.py`, `payload.py`, `pool.py`, `policy.py`, `solve.py`,
   and `validation.py` after callers migrate.
6. Move non-orchestration preconditioner code from
   `problems.transport_matrix` into
   `solvers.preconditioners.transport_matrix` only when it reduces problem-file
   count and keeps tests at the solver owner.
7. Move remaining `io.py` write/read orchestration into existing output owners:
   `outputs/formats.py`, `outputs/rhsmode1.py`, `outputs/transport.py`, and
   `outputs/cache.py`. Keep `sfincs_jax.io` as a documented compatibility shim
   below 800 lines, or delete it after public imports migrate to `api` and
   `outputs`.
8. Delete stale private imports in benchmark scripts and docs that bypass the
   public output/API owners.

Expected result:

- `problems/transport_matrix` drops from 33 to at most 18 files.
- The transport parallel subpackage has at most three implementation files.
- `io.py` is below 800 lines or gone.

Exit gates:

- RHSMode 2/3 transport, monoenergetic transport, streaming output,
  parallel worker/runtime, CLI output, and HDF5/netCDF/NPZ tests pass.
- Runtime/benchmark scripts import canonical `problems.transport_matrix`,
  `outputs`, and `api` owners only.
- Docs explain the transport/output ownership once, not through historical
  internal file names.

### Tranche 3 - Solver/Preconditioner Naming And QI Collapse

This tranche removes experiment-history names while preserving tested solver
families. It is the only allowed solver/preconditioner package reshaping in
this PR; do not add more preconditioner micro-packages.

Move/delete targets:

1. Collapse `solvers/preconditioners/schur/rhs1*.py` into a durable
   profile-response Schur owner. Preferred final name:
   `solvers/preconditioners/schur/profile_response.py`. Delete the old
   `rhs1.py`, `rhs1_coarse_basis.py`, `rhs1_coarse_policy.py`, and
   `rhs1_full_csr.py` after imports and docs migrate.
2. Rename or merge `symbolic_sparse/rhs1_fortran_reduced.py` into
   `symbolic_sparse/profile_response.py` or the existing symbolic-sparse owner.
3. Collapse QI by role:
   `basis.py` for active-pattern/coarse/global/phase-space/residual-region
   basis construction,
   `corrections.py` for block-Schur/two-level/multilevel/deflation/residual
   equations,
   `device.py` for device-compatible setup/probes and former smoother code,
   `policy.py` for admission/promotion.
   Delete `device_smoother.py`, `promotion.py`, and any experiment-history
   module that becomes only a re-export.
4. Keep PAS and x-block files separate only when they represent independently
   tested algorithms. Delete compatibility aliases and historical names that
   are not public API.
5. Move solver-selection environment hooks behind policy objects where an
   automatic default already exists. Advanced environment variables may remain,
   but public docs should describe input-file and Python policy controls first.
6. Update all tests that currently import experiment modules to import the
   role-based owner. Keep one small compatibility test only for names that are
   intentionally public.

Expected result:

- `solvers/preconditioners` drops from 51 to at most 36 files.
- No implementation file starts with `rhs1_` or `transport_`; SFINCS RHSMode
  names may remain in functions/tests where they describe physics.
- Package source file count drops below 205.

Exit gates:

- Focused QI, PAS, x-block, Schur, symbolic-sparse, sparse-factor, and solver
  dispatch tests pass.
- Docs/API maps no longer document transient QI experiment files or
  `rhs1`-prefixed implementation modules.

### Tranche 4 - Public API, Docs, Tests, And Internal Deletion Sweep

This tranche makes the refactor reviewable to users and maintainers. It should
be done after the source tree stops moving so docs and tests do not churn.

Tasks:

1. Keep `sfincs_jax.api` as the public Python surface. Add or preserve stable
   high-level calls for output writing/reading, profile response, transport
   matrix, ambipolar solve, and derivative workflows without exposing internal
   sparse/preconditioner modules.
2. Reduce `docs/api.rst` to public API plus stable owner modules. Move detailed
   internals to source-map/developer docs only when they are durable.
3. Refresh `docs/source_map.rst` to show the final domain packages and remove
   historical `rhs1_*`, `transport_*`, `linear_solve.py`, and transient
   `v3_driver.py` implementation references.
4. Sweep for stale private imports in `tests`, `examples`, `docs`, and
   benchmark scripts. Tests should import canonical owners unless they are
   specifically testing a compatibility shim.
5. Delete dead code inside large files after ownership moves:
   duplicated env parsing, duplicate metadata key collectors, re-export-only
   aliases, unused debug branches, stale local-path helpers, obsolete output
   key builders, and compatibility constants whose callers migrated.
6. Consolidate tests only when behavior remains covered. Prefer moving
   duplicated import-contract assertions into a small number of canonical
   owner tests rather than keeping many per-file compatibility tests.
7. Keep the repository light: no generated HDF5/NPZ/profiler outputs, no large
   temporary plots, and only compressed docs figures that are part of current
   documentation.

Exit gates:

- `README.md`, docs, examples, and source-map pages describe the same package
  structure.
- Sphinx builds with `-W`.
- Import-contract tests cover any remaining public compatibility shims.
- `git diff --check` passes and no temporary outputs are tracked.

### Tranche 5 - Review-Ready Validation

Run this only after Tranches 1-4 are complete.

Required checks:

- Package source file count below 205.
- `v3_driver.py` deleted or below 80 lines.
- `profile_response/solve.py` below 3.5k lines.
- `io.py` deleted or below 800 lines.
- `problems/profile_response` at most 18 Python files.
- `problems/transport_matrix` at most 18 Python files.
- `solvers/preconditioners` at most 36 Python files.
- No top-level `rhs1_*` or `transport_*` implementation files.
- No broad new lint ignores in extracted modules.
- Focused tests pass for:
  profile response RHSMode 1,
  transport matrix RHSMode 2/3,
  ambipolar options 1/2/3,
  RHSMode 4/5 sensitivity contracts,
  sparse-PC and QI admission,
  output formats,
  CLI,
  public API imports,
  docs.
- Representative physics gates still pass for RHSMode 1/2/3, ambipolar, and
  sensitivity.

If these checks pass, PR #8 can move from draft to review-ready. If any
production-runtime or lower-memory solver optimization remains unresolved, it
belongs in the research-lane section below, not in the structural refactor
acceptance criteria.

## Lane 2 - Full Fortran v3 Functionality Matrix

Goal: make the Fortran v3 feature surface explicit and either implemented,
tested, or documented as intentionally unsupported.

Feature matrix:

| Feature | Current target |
| --- | --- |
| RHSMode 1 profile response | Keep parity, improve automatic solver selection, finish production-grid gates. |
| RHSMode 2 energy-integrated transport matrix | Keep parity, reuse operator/factors across RHS, production-floor CPU/GPU reports. |
| RHSMode 3 monoenergetic transport matrix | Keep parity, preserve special grid normalization, compare with DKES-style literature gates. |
| RHSMode 4 adjoint sensitivities at fixed Er | Implement first-class API and HDF5/netCDF outputs. |
| RHSMode 5 adjoint sensitivities at ambipolar Er | Implement after Lane 3 ambipolar and Lane 6 adjoint transpose solve are stable. |
| `ambipolarSolveOption=1` | Implement safeguarded Newton/bisection with `dJr/dEr` from implicit/adjoint solve. |
| `ambipolarSolveOption=2` | Implement Brent root solve using completed physical solves and scan reuse. |
| `ambipolarSolveOption=3` | Implement pure Newton with robust failure certificate and fallback suggestion. |
| `includePhi1` self-consistent | Keep nonlinear Newton-Krylov parity and output all SNES-like diagnostics. |
| `readExternalPhi1` | Add or audit file input parity and cheaper fixed-Phi1 solves. |
| `includePhi1InCollisionOperator` | Keep explicit caveat until parity and memory gates pass. |
| `geometryScheme=1/2/3/4` | Keep analytic geometry parity and physics gates. |
| `geometryScheme=5` | VMEC wout parity, `vmec_jax` differentiable adapter lane, finite-beta QA/QH gates. |
| `geometryScheme=11/12` | Boozer `.bc` parity and memory/runtime reports. |
| `geometryScheme=13` | Decide implementation scope for direct Boozer-spectrum optimization input. |
| `export_f` | Implement distribution-function export on requested grids with bounded output sizes. |
| Matrix/vector debug dumps | Provide NPZ/netCDF debug dumps for operators/RHS/solution/preconditioner metadata. |
| HDF5 output parity | Continue adding all Fortran v3 outputs plus sfincs_jax diagnostics. |
| Classical transport | Audit term-by-term with Fortran v3 and add physics tests. |
| Input validation | Mirror Fortran v3 compatibility errors and warnings, but phrase them for Python/CLI users. |

Acceptance gates:

- A generated feature matrix in docs marks each item as implemented, tested,
  intentionally unsupported, or deferred.
- Every implemented item has at least one unit test, one behavior/regression
  test, and one docs example or reference page.
- Every unsupported/deferred item has a precise physics or engineering reason,
  not a vague "not yet".

## Lane 3 - Ambipolar Solver And Er Derivatives

Goal: make ambipolar calculations first-class, fast, and differentiable when
requested.

Implementation steps:

1. Create `sfincs_jax.problems.ambipolar` as the canonical owner.
2. Define `AmbipolarProblem`, `AmbipolarResult`, and `AmbipolarIteration`
   pytrees with solver path, residual, root type, derivative, and output fields.
3. Implement an in-process fixed-shape ambipolar driver. It must reuse parsed
   input, geometry, grids, operator metadata, factor/preconditioner setup, and
   diagnostic allocation across Er evaluations whenever the shape is unchanged.
4. Reuse the existing scan postprocessor only as a compatibility backend for
   reading completed scan directories; do not use scan-style repeated process
   launches as the primary sfincs_jax algorithm.
5. Implement Fortran-compatible radial-current conventions:
   `J_r = sum_s Z_s Gamma_s`, with the correct flux variant for `Phi1`.
6. Implement Brent with bracket provenance, monotonic interpolation diagnostics,
   and no derivative requirement.
7. Implement safeguarded Newton/bisection using `dJr/dEr`.
8. Implement pure Newton with strict derivative and trust-region guards.
9. Mirror the Fortran v3 source-code validator first:
   derivative-assisted options 1 and 3 require no `Phi1`, no inductive
   electric field, FP collisions, constraint scheme `-1` or `1`, full or DKES
   trajectory compatibility, and no tangential magnetic drifts.
10. Add a separately gated future extension for tangential-drift
    derivative-assisted ambipolar solves if the physics/operator derivatives
    are implemented.
11. Add automatic policy:
   - use derivative-assisted safeguarded Newton when an adjoint/implicit
     derivative is available and bracket is valid;
   - use Brent for robust CLI default when derivative setup is expensive;
   - fail with partial scan artifact if no bracket is found.
12. Add radial-batch ambipolar solve for profile workflows.
13. Add solve-complete/finalization separation: if diagnostics are written but
    profiling or backend finalization fails, record a warning status rather
    than marking the physical solve as failed.
14. Add CLI:
    `sfincs_jax ambipolar input.namelist --er-min ... --er-max ...`.

Derivative formula:

Given `R(u, Er, p) = 0` and `Jr(u, Er, p)`, compute

```text
du/dEr = -R_u^{-1} R_Er
dJr/dEr = partial_Er Jr + Jr_u du/dEr
```

For reverse mode, solve

```text
R_u^T lambda = Jr_u^T
dJr/dEr = partial_Er Jr - lambda^T R_Er
```

Acceptance gates:

- Reproduce Fortran v3 `ambipolarSolveOption=1/2/3` on at least one tokamak,
  one W7-X-like analytic case, one VMEC QA case, and one QH case.
- First reference reproduction target is the checked-in small probe set:
  `geometry4_w7x_like_small_option{1,2,3}` and
  `geometry1_helical_small_option2`. The Brent option-2 sequence is now covered
  by `tests/test_ambipolar_problem.py`.
- The checked `geometry1_helical_small_option1` derivative-assisted Newton
  point is now covered by `tests/test_sensitivity.py` using the active
  Fortran-style operator, `particleFlux_vm_rN`, and the implicit
  tangent/adjoint derivative certificate. The same checked small deck is also
  covered by `tests/test_ambipolar_problem.py` through the real option-1
  safeguarded Newton root solve.
- The checked small option-3 current points for helical and W7-X-like analytic
  decks are now covered by `tests/test_sensitivity.py` using the same active
  namelist-backed provider, and the helical small deck is covered by the real
  pure-Newton option-3 root solve.
- Production reference target is the checked-in production decks under
  `benchmarks/fortran_v3_ambipolar_reference/namelists`, which must be run
  before public benchmark claims are regenerated. The checked-in production
  Brent summary is now covered by `tests/test_ambipolar_problem.py`.
- `dJr/dEr` matches finite differences on a stable step window and matches the
  checked small Fortran option-1 Newton slope.
- Brent and Newton return the same root within tolerance when both are valid.
- CPU/GPU roots and root types match within tolerance for bounded cases.
- Failed brackets write a useful partial artifact and do not claim success.
- The sfincs_jax in-process ambipolar driver reports per-evaluation solver
  trace provenance and currently reuses geometry/output setup through a scoped
  cache, shape-checked Krylov state through a private state file, and symbolic
  active field-split orderings through a fixed-shape key. The remaining
  implementation gate is to avoid repeated numerical operator and
  factor/preconditioner setup when Er updates do not change the problem shape.

## Lane 4 - Adjoint Sensitivities And Differentiable Solves

Goal: replace ad hoc gradient examples with a tested derivative system aligned
with Fortran RHSMode 4/5 and modern JAX implicit differentiation.

Public APIs:

- `linearize_solve(problem, params)`
- `jvp_flux(problem, tangent_params)`
- `vjp_flux(problem, cotangent_outputs)`
- `adjoint_sensitivity(problem, objective)`
- `differentiate_ambipolar_root(problem, objective)`
- `finite_difference_check(problem, parameter, step_window)`

Implementation steps:

1. Define residual graph objects for each problem: residual, matvec, transpose
   matvec, RHS, diagnostics, and parameter leaves.
2. Ensure each residual graph has shape-stable pytrees and no hidden global
   state.
3. Use `jax.lax.custom_linear_solve` for linear solve differentiation.
4. Use `jax.linearize` for Newton/Phi1 JVPs where the nonlinear residual is
   solved in JAX.
5. Use `custom_vjp` only around public solve functions that contain adaptive
   solver choices or host-only fast paths.
6. For CLI/non-differentiable paths, return derivative-unavailable metadata
   rather than silently tracing host operations.
7. Implement Fortran-v3-style adjoint RHS builders for particle flux, heat flux,
   parallel flow, total heat flux, radial current, and bootstrap current.
8. Implement `dL/dlambda f - dS/dlambda` for:
   - `Er`,
   - Boozer `B`,
   - contravariant/covariant Boozer components,
   - `iota`,
   - radial derivative/metric terms after the exact Fortran mapping is audited.
9. Audit the exact Fortran `whichLambda` mapping before public docs, because
   comments and case labels must be reconciled term-by-term.
10. Keep RHSMode 4 fixed-Er output-field contracts synchronized with
    Fortran-v3 fixture summaries.
11. Keep RHSMode 5 ambipolar-Er `dPhi/dPsi` contracts synchronized with
    Fortran-v3 fixture summaries.

Acceptance gates:

- Fortran-v3 RHSMode 4/5 input restrictions and sensitivity HDF5 field names
  are pinned against the source-code behavior.
- Small RHSMode-4 Fortran radial-current, heat-flux, parallel-flow, bootstrap,
  and debug finite-difference sensitivity summaries plus one RHSMode-5
  constant-current heat-flux summary are checked in and tested. Intermediate
  and production-grid parity are release-refresh benchmarks.
- `A^T lambda - J_u^T` adjoint residual passes for every derivative gate.
- JVP and VJP agree through dot-product tests:
  `<JVP(dp), y> = <dp, VJP(y)>`.
- Finite-difference checks pass on documented stable windows.
- Fortran v3 RHSMode 4/5 output contracts match on checked small grids.
- Derivative examples run under CI without full production solves.

## Lane 5 - Native Solver And Preconditioner Architecture

Goal: provide PETSc-like robustness natively in Python/JAX without depending on
PETSc, MUMPS, or SuperLU_DIST at runtime.

Fortran behavior to emulate:

- Exact operator and simplified Pmat are separate.
- Factorization is reused across RHS and adjoint solves when possible.
- Ordering and pivot safeguards are part of solver policy.
- Direct solve failure is diagnosed and retried with safer factor controls.
- Adjoints reuse transpose solves.

Implementation steps:

1. Standardize operator objects with:
   `matvec`, `transpose_matvec`, optional explicit sparse emission,
   block metadata, true residual, and parameter leaves.
2. Standardize preconditioner objects with:
   `apply`, optional `apply_transpose`, setup diagnostics, residual admission,
   memory estimate, and reuse key.
3. Keep two solver stacks:
   - differentiable stack: pure JAX matvecs, Krylov, custom linear solve,
     optional Lineax only if it improves a measured lane;
   - production stack: native Python/JAX sparse/direct/block factors and host
     rescue paths when `differentiable=False`.
4. Implement reusable symbolic ordering metadata for active-only operators.
5. Implement native sparse/block factor families:
   - line factors in `x`, `ell`, `theta`, and `zeta`;
   - block triangular/angular streaming factors;
   - active-only coupled kinetic block factors;
   - moment/source/constraint Schur complement;
   - additive Schwarz patches;
   - nested-dissection/multifrontal-inspired separator updates where beneficial.
6. Add setup-time true-residual admission for every approximate factor:
   no auto promotion without measuring residual reduction on the actual
   operator.
7. Add factor reuse across:
   - RHSMode 2/3 multiple RHS;
   - Er scans at fixed shape;
   - radial batches where geometry shape is fixed;
   - adjoint transpose solves.
8. Add fail-fast memory guards before expensive factor setup.
9. Add structured progress logging for setup and solve phases.
10. Keep lower-memory replacement as research/deferred if it cannot beat the
    existing robust path within the documented budget.

Acceptance gates:

- Auto policy selects the best passing method without environment variables.
- Strict true-residual gate passes before outputs are promoted as converged.
- Production-floor QA/QH RHSMode 1, geometry-rich RHSMode 2/3, and Phi1 cases
  either pass or are documented as deferred with exact residual/runtime/RSS.
- Runtime is no worse than 20x SFINCS Fortran v3 for public production
  comparison claims, unless a case is explicitly marked as research/deferred.
- Peak memory fits documented CPU/GPU budgets and records device memory where
  applicable.

## Lane 6 - T3D, NEOPAX, And Optimization Integration

Goal: make `sfincs_jax` useful as a transport closure in profile solvers and
stellarator optimization, not only as a standalone file-based code.

Public closures:

- `make_flux_surface_closure(geometry, species, resolution, mode)`
- `make_transport_matrix_closure(geometry, species, resolution)`
- `make_ambipolar_closure(geometry, species, resolution, er_policy)`
- `make_radial_profile_closure(geometry_provider, profile_provider, radii)`

Closure inputs:

- radius or radial grid,
- density and temperature profiles,
- density and temperature gradients,
- electric field or ambipolar policy,
- species charges/masses,
- geometry object or geometry file,
- solver/differentiability mode.

Closure outputs:

- particle fluxes,
- heat fluxes,
- parallel flows,
- bootstrap current,
- radial current,
- ambipolar roots and root type,
- transport matrices,
- derivatives/JVP/VJP when requested,
- solver certificates.

Optimization examples:

- QA nfp=2 bootstrap-current minimization.
- QA or QI electron-root targeting from ambipolarity.
- Heat and particle flux minimization with impurity-flux target.
- VMEC-JAX to Booz-Xform-JAX to SFINCS-JAX differentiable pipeline.
- T3D/NEOPAX-style radial closure using a fixed geometry and evolving profiles.

Acceptance gates:

- Examples run from a clean install without private local paths.
- Differentiable examples have finite-difference or adjoint validation.
- Non-differentiable production examples report solver path and residual.
- Docs explain when gradients are exact implicit derivatives, proxy gradients,
  or unavailable.

## Lane 7 - Validation, Physics Gates, And Coverage

Goal: reach high meaningful coverage through real physics, numerical, and
regression tests without making CI expensive.

Test tiers:

| Tier | Runtime budget | Purpose |
| --- | --- | --- |
| Unit | seconds | Grids, coordinates, geometry, operators, collision terms, output schema. |
| Numerical | seconds to 2 min | Residual identities, transpose tests, conservation, quadrature convergence. |
| Physics gates | 2 to 8 min | Literature and Fortran v3 anchored transport/ambipolar/bootstrap checks. |
| CI regression | 5 to 10 min total | Public API, CLI, small parity, derivative gates, docs build. |
| Nightly/optional | 30 to 120 min | Production-floor CPU, GPU, Fortran v3 parity, memory/runtime. |
| Release | manual or scheduled | Full public benchmark regeneration, figures, parity matrix, artifacts. |

Physics gates to keep or add:

- Axisymmetric tokamak banana/plateau/Pfirsch-Schlueter trend checks.
- Monoenergetic DKES-style transport coefficients for W7-X/LHD/HSX analytic
  cases.
- Simakov-Helander high-collisionality limit with a pinned high-nu scan.
- QA/QH bootstrap current comparison against SFINCS Fortran v3 and Redl formula
  at identical resolution, with convergence error bars.
- W7-X ambipolarity with checked equilibrium/profile provenance.
- Finite-beta QA profile-current lane once residual convergence is clean.
- Phi1 sanity checks: flux-surface averaged lowest-order radial particle flux
  cancellation and nonlinear residual convergence.
- Adjoint sensitivity checks against finite differences and Fortran RHSMode 4/5.
- CPU/GPU reproducibility checks for representative PAS, FP, Phi1, QA, QH, and
  QI cases.

Coverage strategy:

- Aim for 95 percent meaningful package coverage after refactor, not by adding
  slow full-solve tests.
- Prefer tests on extracted pure functions and operator blocks.
- Add synthetic manufactured operators for solver/preconditioner unit tests,
  but keep every manufactured test tied to an actual operator identity.
- Keep large equilibria out of git history; fetch release fixtures only in
  optional/nightly jobs.
- Coverage gaps in host-only profiling paths can be exempted only with explicit
  comments and separate integration tests.

## Lane 8 - Benchmarks, Figures, Docs, And Release Artifacts

Goal: regenerate public claims from reproducible scripts after implementation
lanes pass.

Benchmark matrix:

- Devices/geometries: tokamak, HSX, W7-X, LHD, QA, QH, QI nfp=1/2/3/4 when
  available.
- Geometry modes: analytic, VMEC, Boozer `.bc`, direct Boozer spectrum if
  implemented.
- Physics modes: PAS, full FP, magnetic drifts on/off, Er on/off, Phi1 on/off,
  RHSMode 1/2/3/4/5 where supported.
- Resolution tiers:
  - CI tier: small but physically meaningful;
  - public benchmark tier: production-floor resolution;
  - stress tier: collaborator/NTX-like large cases.
- Devices: CPU cold/warm, GPU cold/warm, Fortran v3 reference.

Metrics:

- wall time,
- compile/setup/solve/diagnostics/output phase times,
- peak RSS,
- device memory,
- residual norm and target,
- Krylov iterations and factor setup status,
- selected solver path,
- all shared physical outputs,
- derivative residuals and finite-difference errors when applicable.

Required public figures:

- Runtime and memory comparison plot: Fortran v3, sfincs_jax CPU cold/warm,
  sfincs_jax GPU cold/warm.
- QA/QH bootstrap current: sfincs_jax, Fortran v3, Redl formula, same
  resolution, convergence error bars.
- Ambipolar Er educational/validation plot.
- Autodiff sensitivity validation plot.
- Solver phase timing plot for at least one production-floor case.
- Optional transport profile closure plot for T3D/NEOPAX-style use.

Docs requirements:

- Landing page explains the drift-kinetic equation before the discretized
  system.
- Usage page shows one CLI command, one Python solve, one plot command, and one
  differentiable objective.
- Method pages explain grids, collisions, geometry, Phi1, solvers,
  preconditioners, ambipolar solve, and adjoints.
- API pages expose stable entry points only, not internal helper churn.
- Validation pages explain every physics gate and benchmark provenance.
- Research-lane page lists only honest deferred items with exact status.

## Lane 9 - Release And Branch Hygiene

Goal: ship from a clean, lightweight repository with one review PR.

Steps:

1. Keep the active refactor work in PR #8 until this plan is complete.
2. Do not open additional PRs for sub-lanes.
3. Commit and push coherent tranches frequently.
4. Keep large outputs and generated benchmark data out of git unless they are
   compressed public figures under the artifact-size policy.
5. Before review-ready:
   - full local focused tests pass;
   - docs build passes;
   - CI passes;
   - release benchmark scripts complete or deferred statuses are documented;
   - README figures and docs figures are regenerated from current reports;
   - `git diff --check` passes;
   - no temporary outputs or local paths remain.
6. After review-ready and merge:
   - tag a new version;
   - publish release notes;
   - confirm PyPI workflow;
   - archive release benchmark artifacts.

## Milestones

### M0 - Final Plan And Feature Matrix

Deliverables:

- `plan_final.md` committed.
- Generated Fortran-v3 feature matrix in docs.
- Current sfincs_jax gap table linked from docs.

Exit gate:

- Team agrees that `plan_final.md` is the single controlling plan.

### M1 - Refactor Skeleton

Deliverables:

- Public API and domain package boundaries finalized.
- `v3_driver.py` reduced to orchestration shim or eliminated.
- Compatibility shims marked and scheduled for deletion.

Exit gate:

- Focused refactor tests and docs build pass.

### M2 - Operator And Solver Ownership

Deliverables:

- Residual/operator objects standardized.
- Solver/preconditioner objects standardized.
- Auto policy no longer requires routine environment variables.

Exit gate:

- Representative RHSMode 1/2/3 cases pass CPU parity with automatic defaults.

### M3 - Ambipolar First-Class Solver

Deliverables:

- Brent, safeguarded Newton/bisection, and pure Newton implemented.
- `dJr/dEr` available via implicit/adjoint derivative.
- CLI and Python API documented.

Exit gate:

- Fortran-compatible roots and derivative checks pass on bounded cases.

### M4 - RHSMode 4/5 And General Sensitivities

Deliverables:

- Fixed-Er adjoint sensitivities implemented.
- Ambipolar-Er adjoint sensitivities implemented.
- Derivative outputs written to HDF5/netCDF.

Exit gate:

- Dot-product, finite-difference, and Fortran v3 adjoint parity gates pass.

### M5 - Optimization And Profile Closure

Deliverables:

- Stable closures for T3D/NEOPAX-style integration.
- QA bootstrap-current and electron-root optimization examples.
- VMEC-JAX/Booz-Xform-JAX/SFINCS-JAX workflow example.

Exit gate:

- Examples run and derivative/provenance gates pass.

### M6 - Production Benchmark Regeneration

Deliverables:

- Fresh CPU/GPU/Fortran benchmark reports.
- Runtime/memory plots regenerated.
- QA/QH bootstrap current plot regenerated.
- Parity matrix regenerated.

Exit gate:

- README and docs contain only current data.

### M7 - Review-Ready PR

Deliverables:

- CI green.
- Docs green.
- Coverage target met or remaining uncovered lines are justified.
- No temporary files or local-path artifacts.
- PR #8 ready for review.

Exit gate:

- The branch is merge-ready.

Status:

- Not complete. PR #8 stays draft until Lane 1 Tranches 1-5 pass, docs/source
  maps are refreshed, temporary lint suppressions are removed or narrowed, and
  focused refactor plus representative physics gates pass on the refactor
  branch.

### M8 - Release

Deliverables:

- Merge to main.
- Tag and release.
- PyPI workflow verified.
- Release notes include implemented features, limitations, and benchmark
  provenance.

## Immediate Next Steps

Current completion status:

- Lane 1 structural consolidation: about 45 percent. The compatibility-driver
  boundary is done and the first Tranche 1 ownership batch deleted three
  profile-response files; the remaining large blockers are
  `profile_response/solve.py`, transport/output consolidation,
  solver/preconditioner naming, and `io.py` ownership.
- Ambipolar bounded/reference functionality: about 85 percent. Small and
  bounded Fortran-compatible roots and derivatives are implemented; production
  refresh benchmarks remain outside normal CI.
- RHSMode 4/5 sensitivity contracts: about 75 percent. Small fixture contracts
  and derivative identities are implemented; production-grid parity refresh
  remains a release benchmark.
- Public docs/API stabilization for the refactor PR: about 45 percent. The
  source map exists, but it must be refreshed after the final module names
  settle.

Completed checkpoints that remain valid:

- Fortran-v3 feature matrix and current `sfincs_jax` status matrix are in the
  docs.
- Public lazy facades exist for `sfincs_jax.write_output`,
  `sfincs_jax.read_output`, and `sfincs_jax.run_ambipolar_brent`.
- Top-level `rhs1_*` and `transport_*` implementation files are gone.
- `sfincs_jax.v3_driver` is a small compatibility shim.
- Profile-response policy shards, old low-level linear-solve files, old
  finalization/KSP shards, and top-level sparse-PC handoff have been removed or
  moved into domain owners.
- The latest committed clean checkpoint, `76730df`, folded
  `active_projection.py`, `qi_device_seed.py`, and
  `strong_preconditioning.py` into canonical owners and passed focused owner
  tests, scoped ruff, py_compile, the broader sparse/RHSMode-1 test batch
  (`498 passed in 42.12s`), and `git diff --check`. This file now supersedes
  that checkpoint as the authoritative final consolidation plan.

Next ordered implementation steps:

1. Finish Lane 1 Tranche 1. Move sparse/policy/final handoff code out of
   `solve.py` into existing owners, delete any re-export-only owners that
   become obsolete, and reach `profile_response/solve.py < 3.5k` lines and
   `problems/profile_response <= 18` files.
2. Execute Lane 1 Tranche 2 as one transport/output consolidation. Collapse
   solve-loop shards and the transport parallel micro-files, then move
   transport output streaming and remaining `io.py` write/read logic into
   `outputs`. Target: `problems/transport_matrix <= 18` files,
   transport `parallel <= 3` files, and `io.py < 800` lines or deleted.
3. Execute Lane 1 Tranche 3 as one solver/preconditioner cleanup. Collapse
   Schur `rhs1*` files, symbolic-sparse RHSMode-1 names, and QI experiment
   files into role-based owners. Target:
   `solvers/preconditioners <= 36` files and package source count below 205.
4. Execute Lane 1 Tranche 4 docs/API cleanup. Refresh `docs/api.rst`,
   `docs/source_map.rst`, README/developer docs, examples, tests, and import
   contracts so all references use canonical owners and no transient names.
5. Execute Lane 1 Tranche 5 review-readiness validation. Run focused RHSMode
   1/2/3, ambipolar, sensitivity, sparse-PC, QI, output, CLI, public API,
   docs, and representative physics gates. Then update PR #8 from draft only
   if all structural and validation gates pass.
6. Keep production option-1/3 ambipolar reruns, production-grid RHSMode 4/5
   parity, large CPU/GPU benchmark regeneration, true device-QI promotion, and
   lower-memory production solver optimization as release-refresh or research
   lanes unless they block correctness.

## Known Risks And Explicit Deferred Items

- A fully native MUMPS/SuperLU_DIST-equivalent factorization stack is a large
  numerical project. The near-term target is robust native block/Schur/factor
  infrastructure with honest gates, not an overclaim that it matches mature
  sparse direct solvers on every production case.
- Full production-grid QA/QH RHSMode 1 can remain slower or higher-memory than
  Fortran v3 if residuals and outputs are correct and the limitation is
  documented. It cannot be used for favorable public performance claims unless
  the benchmark gate passes.
- True device-QI and single-case multi-GPU strong scaling remain research lanes
  until strict residual, runtime, and reproducibility gates pass.
- GeometryScheme 13 and full external-Phi1 parity require a scoped audit before
  implementation. They should not block core RHSMode 1/2/3/ambipolar/adjoint
  release milestones unless users require them.
- Lineax, JAXopt, Equinox, and other ecosystem libraries should be adopted only
  after a measured benefit. Avoid adding dependencies for conceptual elegance
  without runtime, memory, compile-time, or derivative-maintenance wins.
