# SFINCS_JAX Final Research-Grade Implementation Plan

Last updated: 2026-06-25 (America/Chicago)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review surface: PR #8, `refactor/v3-driver-architecture`, draft until the
Lane 1 consolidation acceptance gates pass

Status: this file is the controlling completion plan. `plan.md` remains the
execution log and historical record; future work should update this file only
when the target plan itself changes. The Lane 1 consolidation plan below
supersedes older iteration notes and is the only refactor plan to follow; avoid
new one-helper refactor commits.

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

Current source size snapshot after the 2026-06-25 consolidation checkpoint,
the 2026-06-25 final consolidation-plan review, the two root cleanup passes,
the first transport-parallel consolidation, the root solver/preconditioner
disposition move, the first profile-response sparse x-block handoff
extraction, the Schur-family consolidation, the full/reduced sparse retry
stage extraction, the SciPy rescue stage extraction, the final consolidation
planning audit, and the first Batch A validation-helper root cleanup:

- `sfincs_jax/v3_driver.py`: 47 lines in the current consolidation worktree,
  acting as a compatibility shim for the domain-owned solve modules.
- `sfincs_jax/problems/profile_response/solve.py`: 8,328 lines after the
  x-block sparse-PC branch, full/reduced sparse retry, and SciPy rescue stage
  extractions. This remains the largest
  structural debt and must be reduced by moving coherent sections into existing
  domain owners, not by adding many new helper files.
- `sfincs_jax/problems/profile_response/policies.py`: 6,885 lines after
  absorbing six former policy shards and current-backend RHSMode-1 policy
  wrappers. This is intentionally a temporary single policy owner during
  Lane 1 Batch B; do not split it back into micro-files.
- `sfincs_jax/problems/profile_response/preconditioner_build.py`: 2,683 lines
  after taking ownership of the current RHSMode-1 preconditioner builder
  registry, PAS-family compatibility bindings, Schur binding, x-block builder
  aliases, transport `tzfft` reuse, and strong fallback binding.
- `sfincs_jax/problems/profile_response/sparse/handoff.py`: 5,007 lines after
  moving the former top-level sparse-PC handoff into the sparse package and
  taking ownership of the driver-facing x-block sparse-PC GMRES branch
  orchestration and the full/reduced sparse retry stage. Shared generic
  sparse-PC finalization remains in
  `problems/profile_response/sparse/finalization.py`.
- `sfincs_jax/problems/profile_response/sparse/direct.py`: 3,569 lines after
  taking ownership of sparse-factor cache keys, host memory probing, explicit
  sparse-pattern probes, sparse-JAX preconditioner materialization, and host
  sparse direct builder/polish wrappers.
- `sfincs_jax/problems/profile_response/dense.py`: 2,751 lines after taking
  ownership of profile linear-solve routing, dense-KSP, constraintScheme=0
  PETSc-compatible sparse-ILU, SciPy rescue solve contracts, and SciPy rescue
  stage metadata/admission.
- `sfincs_jax/problems/profile_response/auto_solve.py`: 550 lines after taking
  ownership of the explicit host structured-CSR RHSMode-1 solve entry point.
- `sfincs_jax/problems/transport_matrix/solve.py`: 1,763 lines after the
  RHSMode 2/3 solve-entry extraction.
- `sfincs_jax/operators/profile_response/full_system.py`: about 6.0k
  lines, moved from `rhs1_full_assembly.py` during the earlier RHSMode-1
  ownership move.
- Top-level `transport_*` modules: 0.
- Top-level `rhs1_*` modules: 0. Solver-family implementation now lives under
  `solvers.preconditioners`.
- Package total is 209 Python files, 49 package-root files, and about 164k
  package lines after the first two root cleanup passes and the first
  transport-parallel consolidation, plus the validation-domain,
  workflow-domain, solver-utility, and solver/preconditioner implementation
  root-disposition moves. The root cleanups
  deleted the obsolete root modules `solver_runtime.py`,
  `matrix_reductions.py`, `solve_mode_policy.py`,
  `solver_progress_policy.py`, `phase_timing.py`, `linear_algebra.py`,
  `newton_krylov_diagnostics.py`, `phi1_line_search.py`, `sparse.py`, and
  `verbose.py` after moving their tested helpers into canonical owners.
  The transport-parallel consolidation merged `execution.py`, `payload.py`,
  `pool.py`, `solve.py`, and `validation.py` into
  `problems/transport_matrix/parallel/runtime.py`.
  The validation-domain move placed `validation_artifacts.py`,
  `validation_figures.py`, `validation_math.py`,
  `benchmark_artifact_policy.py`, `research_lane_policy.py`, and
  `qi_device_artifact_policy.py` under `sfincs_jax.validation`.
  The workflow-domain move placed `optimization_*`, `mapped_xgrid_*`, and
  `qi_res15_gpu_campaign.py` under `sfincs_jax.workflows`.
  The solver-utility move placed `solver_path_policy.py`,
  `solver_profile_compare.py`, `solver_progress.py`,
  `solver_selection_policy.py`, `solver_state.py`, `solver_trace.py`,
  `krylov_dispatch.py`, `implicit_solve.py`, `memory_model.py`, and
  `sparse_triangular.py` under `sfincs_jax.solvers`.
  The solver/preconditioner implementation move placed `explicit_sparse.py`,
  `explicit_sparse_factor_builder.py`, `explicit_sparse_factor_policy.py`,
  `native_block_factor.py`, `preconditioner_caches.py`,
  `preconditioner_context.py`, `preconditioner_operators.py`, and
  `preconditioner_setup.py` under `sfincs_jax.solvers`. The first Batch A
  validation-helper cleanup moved `fortran.py`, `fortran_profile.py`, and
  `h5_parity.py` under `sfincs_jax.validation`, reducing package-root files
  from 52 to 49 without keeping root compatibility shims.
- Current concentration of complexity after the Schur-family consolidation and
  full/reduced sparse retry and SciPy rescue stage extractions:
  `problems/profile_response` has 13 direct files plus 8 sparse subpackage
  files, for 21 files and about 50.7k lines;
  `problems/transport_matrix` has 23 direct files plus 5 parallel subpackage
  files, for 28 files and about 15k lines; `solvers/preconditioners` has
  47 files and about 37k lines; `operators/profile_response` has 11 files and
  about 14k lines; and `io.py` remains 4,263 lines.

Useful existing assets:

- JAX-native residual/operator code already exists for large portions of the
  v3 model.
- `sfincs_jax/solvers/implicit.py` already wraps `jax.lax.custom_linear_solve`
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

The final codebase should be organized around a small number of domain owners,
not around historical implementation names. This section supersedes all older
Lane 1 refactor notes. `plan.md` is only an execution log; the consolidation
steps below are the single authoritative plan for PR #8.

Target package boundaries:

| Package | Ownership |
| --- | --- |
| `sfincs_jax.api` | Stable Python entry points, return objects, differentiable closures, and documented high-level helpers. |
| `sfincs_jax.cli` | CLI parsing, progress reporting, profiling switches, plotting dispatch, and non-autodiff fast-path controls. |
| `sfincs_jax.input` plus thin `namelist.py` / `input_compat.py` shims | Namelist/TOML parsing, defaults, validation, Fortran-compatible coercions, and fixture-localization helpers. |
| `sfincs_jax.geometry` / `sfincs_jax.discretization` | VMEC/Boozer/Miller/analytic geometry adapters, radial coordinates, grids, quadrature, interpolation, and finite-difference stencils. |
| `sfincs_jax.operators` | Drift-kinetic residuals, Jacobians, RHS/source builders, collision operators, Phi1 terms, and reusable operator layouts. |
| `sfincs_jax.problems` | Problem orchestration only: profile response, transport matrices, ambipolar roots, monoenergetic cases, and solve-result certificates. |
| `sfincs_jax.solvers` | Krylov/direct/native factors, solver policies, reusable preconditioners, progress/trace state, implicit linear solves, and non-autodiff fast paths. |
| `sfincs_jax.outputs` | HDF5/netCDF/NPZ formats, output-cache state, diagnostics schema, streaming transport output, and plotting payloads. |
| `sfincs_jax.sensitivity` | JVP/VJP, implicit differentiation, adjoint checks, finite-difference certificates, and optimization derivative utilities. |
| `sfincs_jax.workflows` | Radial scans, optimization objectives, T3D/NEOPAX closures, examples that compose public APIs, and publication-figure scripts. |
| `sfincs_jax.validation` | Parity reports, benchmark harnesses, physics gates, artifact policies, and release-quality comparison plots. |

Structural rules:

- Prefer 5-10 durable modules in a domain package over dozens of algorithm-name
  shards. A file should exist because it has a domain owner, not because one
  helper was extracted from a monolith.
- Keep files below about 1500 lines when practical. Exceptions are allowed only
  for dense mathematical kernels or compatibility shims with an owner comment
  that explains why the file is intentionally large.
- Do not introduce new root-level implementation modules. Root modules are
  public API, compatibility shims, or long-standing physics kernels that have a
  documented deletion/migration reason.
- Avoid names that encode history instead of ownership: no new `v3_*`,
  `rhs1_*`, `transport_*`, `*_handoff`, `*_promotion`, or campaign-specific
  implementation files unless the file is a temporary compatibility shim with a
  deletion condition.
- Each consolidation commit must reduce net files or remove a large internal
  section. Moving code into a new file without deleting, merging, or shrinking
  another owner does not count as progress.
- Every moved owner boundary gets at least one canonical import test and at
  least one behavior test through the public API or problem-level entry point.
- The PR must stay light: no generated HDF5/NPZ/profiler outputs, no temporary
  plots, no local absolute-path artifacts, and no broad new lint ignores.

## Lane 1 - Final Consolidation Pass

Goal: finish PR #8 with a smaller, clearer source tree without changing
physics, outputs, tolerances, benchmark claims, public behavior, differentiable
Python paths, non-autodiff CLI fast paths, or current CPU/GPU parity gates.
This is the only active refactor plan. Do not add another one-helper module or
new refactor lane unless it replaces an existing file and reduces the final
review surface.

Current inventory from the final 2026-06-25 consolidation audit:

| Area | Current state | Final target for this PR |
| --- | --- | --- |
| `sfincs_jax/v3_driver.py` | 47-line compatibility shim; many tests/scripts still import it | Keep below 80 lines until the final compatibility sweep; delete only if all external imports migrate cleanly. |
| Package source files | 209 Python files | At most 195 files, with lower total package lines than the current branch baseline. |
| Package-root modules | 49 Python files | At most 48 root files unless a documented compatibility shim must remain; no new root implementation modules. |
| `problems/profile_response` | 13 direct files plus 8 sparse files; `solve.py` is 8,328 lines and `policies.py` is 6,885 lines | At most 16 total files; `solve.py` below 3,500 lines; policy/admission code owned by stable responsibilities, not experiment history. |
| `problems/transport_matrix` | 23 direct files plus 5 parallel files; many policy, loop, active-system, and postsolve shards | At most 16 total files; `parallel/` at most 3 implementation files; no policy/postsolve micro-files. |
| `solvers/preconditioners` | 47 files; QI, symbolic, x-block, PAS, full-FP, and domain-decomposition families are still over-fragmented | At most 32 files; no implementation file starts with `rhs1_` or `transport_`; QI files are role-based, not experiment-history based. |
| `io.py` / `outputs` | `io.py` is 4,263 lines; `outputs/` already owns formats, caches, RHSMode-1, and transport output | `io.py` below 800 lines as a compatibility shim, or gone; output implementation lives in `outputs`. |
| Docs/tests/examples | Many references still point to `sfincs_jax.io`, `sfincs_jax.v3_driver`, `sparse.finalization`, and symbolic `rhs1_fortran_reduced` | Public examples use `api`, `cli`, `outputs`, or stable workflows; private imports appear only in focused owner/shim tests. |

Non-negotiable consolidation rules:

- Consolidation commits should be batch-sized. A commit must delete files,
  merge files, or remove a large internal section from a monolith. Moving a
  helper into a new file without deleting another owner is not progress.
- Stable owner names are domain names: `setup`, `solve`, `policies`,
  `residual`, `diagnostics`, `sparse`, `dense`, `outputs`, `workflows`,
  `validation`, and role-based preconditioner families. Avoid historical names:
  no new `v3_*`, `rhs1_*`, `transport_*`, `*_handoff`, `*_promotion`, or
  campaign-specific implementation names.
- Compatibility surfaces are allowed temporarily only when they reduce risk:
  `sfincs_jax.io` and `sfincs_jax.v3_driver` can remain as small shims while
  docs/examples/tests migrate. They must not own new implementation.
- Keep shared lower-level owners when merging would create import cycles.
  `profile_response/sparse/finalization.py` is currently shared by
  `sparse/handoff.py`, `sparse/xblock.py`, and `sparse/fortran_reduced.py`; it
  should be deleted only after those dependencies are inverted or moved behind
  a lower-level payload owner.
- Preserve differentiability boundaries. JAX-native residual/operator/implicit
  derivative paths must stay reachable through Python APIs; host-only fast
  paths should stay behind explicit policy objects used by CLI/default solves.
- Every batch gets one owner import-contract test and at least one behavior
  test through the public API or problem-level entry point before commit.

### Batch A - Boundary Freeze And Compatibility Sweep

Purpose: stop conflicting plans, identify what can be deleted safely, and make
the remaining batches mechanical instead of exploratory.

Actions:

1. Create a root-disposition table in the execution log for all 52 root files:
   public API/CLI, compatibility shim, stable physics kernel, input/geometry
   helper, output helper, solver helper, workflow/validation helper, or delete.
2. Sweep imports in `sfincs_jax`, `tests`, `examples`, `scripts`, `benchmarks`,
   and docs for `sfincs_jax.io`, `sfincs_jax.v3_driver`, private
   `profile_response` owners, `sparse.finalization`, symbolic
   `rhs1_fortran_reduced`, and old docs references. Exclude generated
   `docs/_build` output from planning decisions.
3. Delete obvious re-export-only shims and aliases that have no external import
   contract, in one root cleanup batch.
4. Mark compatibility shims that must remain and add a deletion condition for
   each one.

Exit gates:

- Worktree clean before Batch B.
- Package file count does not increase.
- Root file count is at most 49 and trends toward 48.
- `docs/source_map.rst` does not reference files deleted in completed batches.

### Batch B - Profile-Response Owner Collapse

Purpose: make RHSMode 1 reviewable by reducing `profile_response/solve.py` to
phase sequencing and moving retry/finalization/result logic into existing
owners, not new shards.

Target final owner map:

- `setup.py`: namelist-to-problem setup and fixed-shape state construction.
- `solve.py`: public RHSMode-1 entry points and phase sequencing only.
- `policies.py`: auto solver selection, memory/runtime admission, and fallback
  policy. If split later, split into durable subdomains only; do not create
  experiment-specific policy files.
- `preconditioner_build.py`: registry, builder binding, setup admission, and
  reusable preconditioner context assembly.
- `dense.py`: dense, SciPy, constraintScheme=0, and host rescue paths.
- `residual.py`: true residual targets, convergence checks, and certificates.
- `diagnostics.py` and `solver_diagnostics.py`: physical output, progress
  replay, solver metadata, and final-result normalization.
- `phi1_newton.py`: Phi1 nonlinear accepted-iterate and line-search logic.
- `active_dof.py`: active-index construction.
- `sparse/`: sparse/direct factors, generic sparse-PC orchestration, x-block,
  QI, Fortran-reduced paths, and shared sparse payload/result types.

Actions:

1. Move the remaining generic sparse-PC/factor-preflight branch from
   `solve.py` into existing sparse owners as one batch. Do not add another
   helper-only file.
2. Move final result payload assembly, progress replay, solver metadata
   normalization, and sparse fallback summaries out of `solve.py`.
3. Replace long blocks of policy toggles with named policy dataclasses or
   compact decision functions in `policies.py` only when this removes duplicated
   branches from `solve.py`.
4. Keep `sparse/finalization.py` during this batch unless the import graph is
   inverted first. The current `handoff -> xblock` and `handoff ->
   fortran_reduced` imports make a direct merge into `handoff.py` unsafe.
5. Delete profile-response files that become re-export-only after the large
   moves, and update import contracts in the same commit.

Exit gates:

- `profile_response/solve.py` below 3,500 lines.
- `problems/profile_response` plus `problems/profile_response/sparse` at most
  16 total files.
- No new root-level, experiment-name, `rhs1_*`, or `*_handoff` files.
- Focused RHSMode-1 tests pass for solve routing, dense fallback, sparse-PC
  handoff, direct-tail policy, host policy, QI admission, Phi1, ambipolar,
  sensitivity, output diagnostics, and import contracts.
- Scoped `ruff`, `py_compile`, and `git diff --check` pass for touched owners.

### Batch C - Transport, Output, And Root Ownership Collapse

Purpose: remove the biggest file-count sources outside RHSMode 1 and make
output ownership unambiguous.

Transport actions:

1. Merge solve-loop micro-files into one durable transport execution owner:
   `dense_lu.py`, `dense_batch.py`, `host_gmres.py`, `loop.py`,
   `iteration_stats.py`, and `residual_quality.py` should not remain as
   separate files.
2. Merge `solve_policy.py` and `handoff_policy.py` into
   `transport_matrix/policies.py`.
3. Merge `postsolve_diagnostics.py` into `finalize.py`.
4. Merge `active_dense.py` and `active_factor.py` into one active-system owner.
5. Move reusable solver implementations out of `problems/transport_matrix`
   only if they naturally belong to `solvers/preconditioners/transport_matrix.py`.
   Problem orchestration stays in `problems`.
6. Keep only `parallel/runtime.py`, `parallel/worker.py`, and optionally
   `parallel/sharding.py`; merge `parallel/policy.py` into `runtime.py` unless
   the policy is a durable owner with direct tests.

Output/root actions:

1. Move solved-field schema assembly, output dictionary construction, and
   HDF5/netCDF/NPZ writing from `io.py` into `outputs/formats.py`,
   `outputs/rhsmode1.py`, `outputs/transport.py`, and `outputs/cache.py`.
2. Leave `sfincs_jax.io` as a shim below 800 lines only if examples, scripts,
   or external compatibility tests still require legacy imports.
3. Move output-related docs/examples to public `api`, `outputs`, or CLI usage.
   Private `io.py` helpers should appear only in focused shim tests.
4. Remove or migrate root-level helper modules that the root-disposition table
   marks as solver/output/workflow/validation implementation.

Exit gates:

- `problems/transport_matrix` plus `parallel` at most 16 total files.
- `problems/transport_matrix/parallel` at most 3 implementation files.
- `io.py` below 800 lines or deleted.
- Package root at most 48 files, unless a documented compatibility shim gate
  justifies staying at 52 until Batch E.
- RHSMode 2/3 transport, monoenergetic transport, streaming output,
  parallel runtime/worker, CLI output, HDF5/netCDF/NPZ, output-format, and
  public API output tests pass.

### Batch D - Solver And Preconditioner Family Collapse

Purpose: replace experiment-history files with role-based numerical families
and make automatic solver selection easier to review.

Actions:

1. Merge `symbolic_sparse/rhs1_fortran_reduced.py` into a stable symbolic
   sparse owner such as `symbolic_sparse/profile_response.py`, then update the
   symbolic package facade and tests.
2. Collapse QI into four role-based owners:
   `basis.py` for active-pattern, phase-space, global-moment, and
   residual-region bases;
   `corrections.py` for block-Schur, two-level, multilevel, deflation,
   coupled residual, and residual-Galerkin corrections;
   `device.py` for GPU/device-compatible setup and former smoother support;
   `policy.py` for admission and promotion.
3. Delete QI `device_smoother.py`, `promotion.py`, and any experiment-history
   file that becomes re-export-only after the role-based owners exist.
4. Collapse x-block files only where names describe implementation detail
   rather than stable concepts. Keep distinct families for active projected,
   low-l Schur, radial, and theta-zeta sparse only when focused tests prove
   they are independent owners.
5. Keep PAS, full-FP, symbolic sparse, Schur, transport matrix, and
   domain-decomposition separate where they are distinct numerical families.
   Merge inside each family when files are merely policy/detail shards.
6. Move environment-variable solver hooks behind input/Python policy objects
   when a documented input or API control exists. Advanced overrides may remain,
   but defaults must be automatic and documented as the normal user path.

Exit gates:

- `solvers/preconditioners` at most 32 files.
- QI at most 5 implementation files including `__init__.py`.
- No implementation file starts with `rhs1_` or `transport_`.
- Focused QI, PAS, x-block, Schur, symbolic-sparse, sparse-factor, full-FP,
  domain-decomposition, solver-dispatch, CPU/GPU admission, and import-contract
  tests pass.
- Docs/API maps no longer document transient QI experiment files or
  RHSMode-prefixed implementation modules.

### Batch E - Public API, Docs, Tests, And Review Gate

Purpose: make PR #8 review-ready after the source tree stops moving.

Actions:

1. Keep `sfincs_jax.api` as the public Python surface for output read/write,
   profile response, transport matrix, ambipolar solve, derivative workflows,
   validation helpers, plotting payloads, and optimization-ready closures.
2. Examples must use `api`, `cli`, `outputs`, or documented workflow modules,
   not internal sparse/preconditioner modules.
3. Reduce `docs/api.rst` to public API plus stable owner modules.
4. Refresh `docs/source_map.rst`, README developer notes, examples, and
   benchmark docs so they describe the same package structure and no
   historical `rhs1_*`, `transport_*`, `linear_solve.py`, or transient
   `v3_driver.py` internals.
5. Sweep tests, examples, docs, and benchmark scripts for stale private
   imports. Keep compatibility imports only where a focused shim test exists.
6. Consolidate duplicate import-contract tests into canonical owner tests while
   preserving physics, regression, numerical, output-format, CLI, autodiff, and
   docs coverage.
7. Run repository-size hygiene checks and remove temporary generated outputs.

Review-ready acceptance gates:

- Package source file count is at most 195.
- Package source lines are below the current branch baseline.
- Package root has at most 48 Python files, or at most 52 with written
  compatibility-shim deletion conditions.
- `v3_driver.py` is deleted or below 80 lines.
- `profile_response/solve.py` is below 3,500 lines.
- `io.py` is deleted or below 800 lines.
- `problems/profile_response` plus `sparse` has at most 16 Python files.
- `problems/transport_matrix` plus `parallel` has at most 16 Python files.
- `solvers/preconditioners` has at most 32 Python files.
- No top-level `rhs1_*` or `transport_*` implementation files exist.
- No broad new lint ignores exist in extracted modules.
- Focused tests pass for profile response RHSMode 1, transport matrix RHSMode
  2/3, ambipolar options 1/2/3, RHSMode 4/5 sensitivity contracts, sparse-PC,
  QI admission, output formats, CLI, public API imports, docs, representative
  physics gates, and differentiable linear-solve contracts.
- Sphinx builds with `-W`, `git diff --check` passes, and no temporary outputs
  are tracked.

If these gates pass, PR #8 can move from draft to review-ready. Any remaining
production-runtime optimization, true device-QI promotion, or lower-memory
native-factor work belongs in the research-lane sections below unless it blocks
correctness.

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
3. Commit and push coherent batches frequently.
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

- Not complete. PR #8 stays draft until Lane 1 Batches A-E pass, docs/source
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

- Lane 1 structural consolidation: about 78 percent. The compatibility-driver
  boundary is done, the first profile-response ownership checkpoint deleted
  three profile-response files, the current RHSMode-1 preconditioner registry is
  owned by `profile_response/preconditioner_build.py`, sparse env parsing is
  shared through `profile_response/sparse/policy.py`, x-block final payload
  builders live in `profile_response/sparse/xblock.py`, and the two root
  cleanup passes deleted ten obsolete top-level helper modules. The first
  transport-parallel consolidation deleted five transport micro-files. The
  first root-disposition checkpoint moved six validation and artifact-policy
  modules into `sfincs_jax.validation`, reducing package-root files from 85 to
  79. The second root-disposition checkpoint moved nine optimization,
  mapped-xgrid, and QI campaign modules into
  `sfincs_jax.workflows`, reducing package-root files to 70. The third
  root-disposition checkpoint moved ten solver utility modules into `sfincs_jax.solvers`,
  reducing package-root files to 60. The fourth root-disposition checkpoint moved
  eight solver/preconditioner implementation modules into `sfincs_jax.solvers`,
  reducing package-root files to 52 and meeting the earlier root-count gate.
  The first Batch A validation-helper cleanup moved `fortran.py`,
  `fortran_profile.py`, and `h5_parity.py` under `sfincs_jax.validation`,
  reducing package-root files to 49. The first profile-response sparse
  checkpoint moved the x-block sparse-PC GMRES branch
  from `profile_response/solve.py` into `profile_response/sparse/handoff.py`.
  The Schur-family consolidation then replaced four historical
  `solvers/preconditioners/schur/rhs1*` implementation files with the canonical
  `solvers/preconditioners/schur/profile_response.py` owner, reducing package
  files to 209 and preconditioner files to 47. The full/reduced sparse retry
  stage is now owned by `profile_response/sparse/handoff.py`, reducing
  `profile_response/solve.py` to 8,453 lines. The SciPy rescue stage is now
  owned by `profile_response/dense.py`, reducing `profile_response/solve.py`
  to 8,328 lines. The remaining large blockers are the generic
  sparse-PC/factor-preflight branch, final result/progress
  normalization, the rest of transport/output consolidation,
  solver/preconditioner naming, and `io.py` ownership. The next work follows
  Lane 1 Batches A-E only.
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
- The latest local consolidation checkpoint moved the CPU SciPy rescue
  stage from `profile_response/solve.py` into `profile_response/dense.py`,
  added direct tests for improving-rescue and active-size-cap metadata
  behavior, and passed focused owner tests, scoped ruff, py_compile, and
  sparse/RHSMode-1 coverage. The final consolidation audit also found that
  direct deletion of `profile_response/sparse/finalization.py` would currently
  create import-cycle risk because `handoff.py`, `xblock.py`, and
  `fortran_reduced.py` share its payload/result types. This file is the
  authoritative final consolidation plan.

Next ordered implementation steps:

1. Execute Lane 1 Batch A as a boundary freeze and compatibility sweep.
   Record the root-disposition table in `plan.md`, delete obvious
   re-export-only shims, refresh source maps for already-landed moves, and keep
   root files at or below 49 while trending toward 48.
2. Execute Lane 1 Batch B as one profile-response consolidation. Move the
   remaining generic sparse-PC/factor-preflight branch, result payload
   assembly, progress replay, and diagnostic normalization out of `solve.py`.
   Keep `sparse/finalization.py` until its shared payload/result dependencies
   are inverted safely; then delete re-export-only profile-response files and
   reach `profile_response/solve.py < 3.5k` with at most 16 profile-response
   files including `sparse`.
3. Execute Lane 1 Batch C as one transport/output/root consolidation. Collapse
   transport solve-loop shards, merge or justify parallel policy, move
   streaming/output logic into `outputs`, and finish canonical imports for
   root-level workflow/validation modules already moved. Targets:
   `problems/transport_matrix + parallel <= 16`, `parallel <= 3`,
   `io.py < 800` or deleted, and package root `<= 48` unless documented shim
   constraints require `<= 52`.
4. Execute Lane 1 Batch D as one solver/preconditioner domain consolidation.
   Collapse symbolic-sparse RHSMode-1 names and QI experiment files into
   role-based owners. Target: `solvers/preconditioners <= 32`, QI at most
   5 implementation files including `__init__.py`, and no implementation file
   starting with `rhs1_` or `transport_`.
5. Execute Lane 1 Batch E as the docs/API/tests/review gate. Refresh
   `docs/api.rst`, `docs/source_map.rst`, README/developer docs, examples,
   benchmark scripts, tests, and import contracts so all references use public
   API or canonical owners. Gate: package source files `<= 195`, current line
   baseline reduced, Sphinx `-W` clean, focused tests clean, and no temporary
   outputs tracked.
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
