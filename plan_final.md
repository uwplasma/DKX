# SFINCS_JAX Final Research-Grade Implementation Plan

Last updated: 2026-06-26 (final consolidation-pass plan refresh)

Active branch: `refactor/rhs1-full-assembly-preconditioners`

Review surface: PR #8, `refactor/v3-driver-architecture`, draft until the
Lane 1 consolidation acceptance gates pass

Status: this file is the controlling completion plan. `plan.md` remains the
execution log and historical record. The Lane 1 consolidation plan below is the
only refactor plan to follow; older iteration/tranche notes are historical.
Avoid one-helper refactor commits, new implementation shards, and new root
modules unless the same commit deletes or merges a larger owner.

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

Current source size snapshot after the 2026-06-26 consolidation audit and the
latest transport/profile-response owner moves:

- Whole package: 178 Python files after the completed
  profile-response solve-sequencer/handoff compression, output-writer move,
  transport/output payback, solver/preconditioner family compression, Batch A
  gate repair, and Batch B transport linear-system consolidation. The
  historical symbolic-sparse `rhs1_*` filename has been removed, QI has durable
  owner modules, and the preconditioner file-count gate is met. Package source
  lines are 165,992; this is above the previous line-count checkpoint but is
  justified by deleting ten QI shards, one domain-decomposition shard, and four
  transport linear-system implementation files while replacing them with
  durable owner modules.
- Package root: 43 Python files. No top-level `rhs1_*` or `transport_*`
  implementation files remain.
- `sfincs_jax/v3_driver.py`: 47-line compatibility shim. It must not regain
  implementation logic.
- Historical roots deleted and routed to owners:
  `v3_results.py`, `v3_sparse_pattern.py`, `v3_fblock.py`, `v3_system.py`,
  and `v3.py`.
- `sfincs_jax/problems/profile_response`: 18 files including `sparse/`, about
  52.7k lines. The largest files are `sparse/xblock.py` 7,725 lines,
  `policies.py` 7,425 lines, `solve.py` 5,420 lines,
  `sparse/handoff.py` 5,500 lines, `sparse/qi.py` 4,885 lines,
  `sparse/direct.py` 3,569 lines, `dense.py` 3,287 lines, and
  `preconditioner_build.py` 2,683 lines. The `solve.py <=5,500` and
  `handoff.py <=5,500` review gates are restored.
- `sfincs_jax/problems/transport_matrix`: 12 files including `parallel/`.
  The file-count gate is met. `postsolve_diagnostics.py` was merged into
  `finalize.py`, `streaming_outputs.py` was merged into `outputs/transport.py`,
  and active dense setup, active factors, direct reduced-``Pmat``, direct
  block-Schur setup, and Fortran-reduced LU setup were consolidated into
  `linear_system.py`.
- `sfincs_jax/solvers/preconditioners`: 35 files. QI now has five durable
  owner files: `basis.py`, `corrections.py`, `device.py`, `policy.py`, and
  `__init__.py`. The empty `coarse_space` package was deleted, the QI device
  smoother was merged into `qi/device.py`, the historical symbolic-sparse
  `rhs1_fortran_reduced.py` file was renamed to
  `symbolic_sparse/profile_response.py`, and the domain-decomposition line/block
  implementation was merged into its package owner.
- `sfincs_jax/operators/profile_response`: 14 files, about 18.4k lines.
  `full_system.py` is 5,978 lines. This package is large but domain coherent;
  do not split it in this PR unless a correctness bug requires it.
- `sfincs_jax/io.py`: 64-line compatibility facade. The concrete writer now
  lives in `sfincs_jax/outputs/writer.py` at 4,264 lines and is exported from
  `sfincs_jax.outputs`. `outputs/transport.py` now owns both streaming
  transport-output accumulation and streaming HDF5 writes.

The next consolidation pass must reduce these concentrated owners in a few
large batches. It must not add more one-off helper files, and it must not turn
the current 182-file tree into a larger but more fragmented tree.

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
physics, outputs, tolerances, benchmark claims, public behavior,
differentiable Python paths, non-autodiff CLI fast paths, or current CPU/GPU
parity gates. This is the only active refactor plan. Older tranche, sweep,
helper-extraction, and experiment-lane notes are historical and must not be
followed as parallel plans.

### Current Baseline And Locked Checkpoints

Audit commands for every batch:

```text
find sfincs_jax -name '*.py' -not -path '*/__pycache__/*' | wc -l
find sfincs_jax -maxdepth 1 -name '*.py' | wc -l
find sfincs_jax -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 wc -l | sort -nr | head -40
find sfincs_jax -name '*.py' -type f | sed 's#/[^/]*$##' | sort | uniq -c | sort -nr | head -50
find sfincs_jax/solvers/preconditioners -maxdepth 3 -type f -name '*.py' -print0 | xargs -0 wc -l | sort -nr
rg -n "sfincs_jax\.(v3_driver|problems\.transport_matrix\.postsolve_diagnostics|solvers\.preconditioners\.symbolic_sparse\.rhs1_fortran_reduced)" docs README.md examples tests sfincs_jax
```

Current source inventory from the final consolidation audit:

| Area | Current state | Review-ready target |
| --- | --- | --- |
| Whole package | 178 Python files, 165,992 package lines after Batch B. The file-count gate is met; the remaining line increase is explicitly justified by replacing multiple implementation shards with durable owner modules. | `<=190` Python files and below the previous 165,398-line checkpoint before review, or a documented line-count exception tied to deleted files plus clearer ownership. Stretch target: `<=175` files only if it improves clarity. |
| Package root | 43 Python files | `<=40` preferred, `<=44` maximum. Every remaining root file must be public API, stable physics kernel, or documented compatibility shim. |
| `v3_driver.py` | 47-line compatibility shim | Keep below 80 lines or delete after public imports migrate. It must not regain implementation logic. |
| `io.py` and `outputs/` | `io.py` is a 64-line compatibility facade; `outputs/writer.py` owns the 4,264-line writer; `outputs/transport.py` owns transport output accumulation and streaming writes. | `io.py` gate is met. Output implementation belongs in `outputs`, not root. |
| `problems/profile_response` | 18 files including `sparse/`; `solve.py` is 5,420 lines and `sparse/handoff.py` is 5,500 lines. | Review-ready line gates are restored. Do not add profile-response files. |
| `problems/transport_matrix` | 12 files including `parallel/`; active dense setup, active factors, direct reduced-``Pmat``, direct block-Schur setup, and Fortran-reduced LU setup now live in `linear_system.py`. | Review-ready file-count gate is met. Next consolidation target is `parallel/` if imports allow. |
| `solvers/preconditioners` | 35 files; QI has 5 files. QI is consolidated into `basis.py`, `corrections.py`, `device.py`, `policy.py`, and `__init__.py`; domain decomposition is consolidated into its package owner. | Review-ready gate met. Stretch target: `<=30` only if it improves clarity. No implementation file starts with `rhs1_` or `transport_`. |
| `solvers` | 19 root solver files; `explicit_sparse.py` is 4,039 lines and several small policy/context/cache files remain. | Consolidate only by domain owner: explicit sparse factors, preconditioning state, and diagnostics/progress. No new one-helper files. |
| Docs/tests/examples | Some private owner names still appear in docs/tests/examples. | Public examples use `api`, `cli`, `outputs`, or documented workflows. Private imports stay only in owner tests. |

Locked checkpoints:

- Top-level `rhs1_*` and `transport_*` implementation files are gone.
- Historical root implementation shims `v3_results.py`, `v3_sparse_pattern.py`,
  `v3_fblock.py`, `v3_system.py`, and `v3.py` are gone or routed to owners.
- `sfincs_jax.v3_driver` is only a compatibility shim and must not regain
  implementation logic.
- `profile_response/solve.py <=5,500`,
  `profile_response/sparse/handoff.py <=5,500`, `v3_driver.py <=80`, and
  `io.py <=800` are met. Keep them locked while later batches move owners.

### Non-Negotiable Consolidation Rules

- Do not create another plan. This file is the plan; `plan.md` is only the
  execution log.
- Do not add helper-only files. A refactor commit must delete files, merge
  files, remove a large internal section, or replace several historical names
  with one durable domain owner.
- Prefer fewer durable files over many thin files. File names must describe
  domain ownership, not extraction history.
- Avoid new historical or campaign names: no new `v3_*`, `rhs1_*`,
  `transport_*`, `*_handoff`, `*_promotion`, or experiment-specific
  implementation modules.
- A new file is allowed only when the same commit deletes at least two smaller
  files or replaces a worse historical owner name with a durable domain owner.
- Keep compatibility shims only when they protect users or tests. Each shim
  must have an owner and deletion condition in docs/source maps or tests.
- Preserve differentiability boundaries. JAX-native residual/operator/
  implicit-derivative paths remain available through Python APIs; host-only
  fast paths stay explicit policy choices for CLI/default solves.
- Do not restart broad solver research inside this consolidation pass. Solver
  performance work resumes only if a correctness regression blocks review
  readiness.
- Every consolidation batch gets import-contract coverage plus at least one
  behavior test through a public API or problem-level entry point before
  commit.
- Do not split a large owner into several new smaller files just to lower line
  count. A new owner is allowed only if the same commit deletes at least two
  old files and the new name is a durable domain name.
- Prefer registry/data-structure simplification inside existing owners for
  oversized files such as `profile_response/solve.py`, `policies.py`,
  `sparse/xblock.py`, and `sparse/handoff.py`.

### File Disposition Matrix

| Area | Keep | Merge or move | Delete or shim condition |
| --- | --- | --- | --- |
| Root public surface | `api.py`, `cli.py`, `namelist.py`, `input_compat.py`, `geometry.py`, `grids.py`, `sensitivity.py`, `plotting.py`, stable physics kernels | Classify `data_fetch.py`, `postprocess_upstream.py`, and `scans.py`. Move them only if public imports can migrate without adding shims; otherwise document them as public root workflows. | Delete no-op root shims after public imports migrate. |
| Compatibility roots | `v3_driver.py` only as temporary import shim | Result tests/docs import problem-owned result contracts directly | Delete `v3_driver.py` only if all public imports and tests move to domain APIs. |
| Profile response | Existing owners only: `setup.py`, `solve.py`, `policies.py`, `preconditioner_build.py`, `dense.py`, `residual.py`, `diagnostics.py`, `solver_diagnostics.py`, `phi1_newton.py`, `sparse/` | Restore `solve.py <=5,500` without creating files. Keep the `handoff.py` waiver only if documented as a compatibility re-export facade. Reduce oversized owners by deleting duplicate branch patterns, not by creating shards. | Do not create another profile-response file. |
| Transport matrix | `solve.py`, `setup.py`, `diagnostics.py`, `finalize.py`, `policies.py`, `linear_system.py`, `parallel/runtime.py`, and `parallel/worker.py` | Batch B merged `active_dense.py`, `active_factor.py`, `direct_block_schur.py`, `direct_pmat.py`, and `fortran_reduced_lu.py` into `linear_system.py`. Batch C merged internal `parallel/policy.py` and `parallel/sharding.py` into `parallel/runtime.py`. | Keep `parallel/worker.py` only as the documented `python -m sfincs_jax.problems.transport_matrix.parallel.worker` subprocess entry point. Delete tiny relay files after tests import the owner. Do not grow `transport_matrix/solve.py` into another monolith. |
| Outputs | `outputs/formats.py`, `outputs/cache.py`, `outputs/rhsmode1.py`, `outputs/transport.py`, `outputs/writer.py` | Continue moving schema/output-policy pieces into output owners only if total package complexity drops. | `io.py` remains a `<=800` line compatibility facade until public imports no longer need it. |
| Solver core | `explicit_sparse.py`, `implicit.py`, `krylov_dispatch.py`, `path_policy.py`, `selection_policy.py`, `memory_model.py`, native factor kernels, and one preconditioning-state owner | Batch D merged `explicit_sparse_factor_builder.py` and `explicit_sparse_factor_policy.py` into `explicit_sparse.py`. Next target: merge `preconditioner_caches.py`, `preconditioner_context.py`, `preconditioner_operators.py`, and `preconditioner_setup.py` into one clear owner only if imports stay acyclic. Merge `progress.py`, `state.py`, `trace.py`, and `profile_compare.py` into a diagnostics/progress owner if they are internal. | Delete old policy/context/cache files only after import-contract and solver-dispatch tests pass. |
| QI preconditioners | Durable owners are fixed: `qi/basis.py`, `qi/corrections.py`, `qi/device.py`, `qi/policy.py`, plus `qi/__init__.py` | No more QI file movement unless a correctness bug appears. Simplify internally by deleting dead `qi_*` compatibility symbols or duplicated basis/correction code. | Keep compatibility aliases only through `qi/__init__.py` or owner tests, not as files. |
| Symbolic sparse | `symbolic_sparse/active_factors.py`, `symbolic_sparse/host_factor.py`, `symbolic_sparse/policy.py`, `symbolic_sparse/profile_response.py` | Merge only if it removes a real boundary and keeps names clearer. | No symbolic-sparse implementation file may use an `rhs1_*` filename. |
| X-block/PAS/full-FP | Role-based family owners only | Merge policy/detail shards into family owners when they are not independent mathematical kernels. | No new smoother or experiment files. |
| Operators | Current owners are acceptable for this PR | Only merge very small term files if needed for file-count gates and if docs remain clearer. | Do not split `full_system.py` during this consolidation. |

### Final Consolidation Execution Plan

This sequence is the only implementation path for the consolidation pass. Each
batch is deliberately large enough to move an owner boundary, delete files, or
finish a review gate. Do not stop after a one-function helper move.

#### Batch A - Gate Repair And Compatibility Freeze

Purpose: repair the currently reopened review gate and freeze the compatibility
surface before further movement.

Actions:

1. Restore `profile_response/solve.py <=5,500` without creating files. Prefer
   compacting compatibility imports/re-export plumbing or deleting duplicated
   branch payload code; do not move one helper into a new module.
2. Audit `profile_response/sparse/handoff.py` without its local
   `F401,F811` waiver. If Ruff reports only intentional dynamic re-export and
   driver-scope shadowing errors, keep the waiver but document it in the file,
   `docs/source_map.rst`, and import-contract tests with a deletion condition.
3. Verify `profile_response/sparse/handoff.py <=5,500`, `v3_driver.py <=80`,
   and `io.py <=800`.

Exit gates:

- `profile_response/solve.py <=5,500`.
- `profile_response/sparse/handoff.py <=5,500`.
- `v3_driver.py <=80` and remains implementation-free.
- `io.py <=800` and remains a compatibility facade.
- Focused profile-response sparse-PC, active projection, QI admission, direct
  tail, dense fallback, output-diagnostic, import-contract, py_compile, scoped
  Ruff, and `git diff --check` pass.

#### Batch B - Transport Linear-System Owner Consolidation

Purpose: delete several small transport numerical shards in one owner-level
move without making `transport_matrix/solve.py` a second monolith.

Actions:

1. Create or reuse one durable owner named by the domain, preferably
   `sfincs_jax/problems/transport_matrix/linear_system.py`.
2. Move the coherent active-system/factor/direct-Pmat functionality from
   `active_dense.py`, `active_factor.py`, `direct_block_schur.py`,
   `direct_pmat.py`, and `fortran_reduced_lu.py` into that owner.
3. Delete those five old files in the same commit and update imports from
   `profile_response/solve.py`, transport tests, docs, and source maps.
4. Keep tests named for behavior, not old filenames. Owner tests should check
   active dense setup, active factorization, direct block-Schur/direct-Pmat
   paths, and Fortran-reduced LU behavior through the new owner.

Exit gates:

- Net package file count decreases by at least four.
- `problems/transport_matrix` remains `<=18` files including `parallel`.
- No `transport_*` or old algorithm-history filename is introduced.
- Transport RHSMode 2/3, monoenergetic, sparse/direct fallback, active-system,
  streaming-output, and import-contract tests pass.

#### Batch C - Transport Parallel Runtime Consolidation

Purpose: keep parallelization simple for users and reviewers by reducing the
parallel package to runtime ownership unless a file is a real independent
kernel.

Actions:

1. Audit `parallel/policy.py`, `parallel/sharding.py`, and `parallel/worker.py`
   imports and tests.
2. Merge purely internal policy and sharding helpers into
   `parallel/runtime.py`; delete the old files in the same commit if imports
   are internal.
3. Keep `parallel/worker.py` because it is a documented subprocess executable
   path, not a tiny implementation shard.
4. If any other public import must remain, expose it from a documented API
   owner, not by keeping a tiny implementation file.

Exit gates:

- `problems/transport_matrix/parallel` has two implementation files:
  `runtime.py` and `worker.py`.
- Parallel runtime, sharding, worker-payload, CPU/GPU admission, and import
  tests pass.

Status on 2026-06-26:

- Complete. `parallel/policy.py` and `parallel/sharding.py` were absorbed into
  `parallel/runtime.py`; `parallel/worker.py` stayed as the public executable
  wrapper. Live docs, examples, tests, and source no longer import the deleted
  modules.
- Current metrics after Batch C: `176` package Python files, `43` package-root
  files, `165,968` package source lines, `problems/transport_matrix` at `10`
  Python files including `parallel`, and `parallel` at `runtime.py`,
  `worker.py`, and `__init__.py`.
- Validation: scoped py_compile and Ruff passed; the focused transport-parallel
  suite passed with `101 passed`; the full `tests/test_transport_*.py` pattern
  passed with `273 passed`; Sphinx `-W` passed; `git diff --check` passed; and
  the live stale-reference audit found no references to deleted
  `parallel.policy` or `parallel.sharding` modules outside frozen static
  evidence artifacts.

#### Batch D - Solver Core And Preconditioner Surface Consolidation

Purpose: reduce solver-family fragmentation while preserving automatic solver
selection, differentiable JAX paths, and non-autodiff CLI fast paths.

Actions:

1. Merge `explicit_sparse_factor_builder.py` and
   `explicit_sparse_factor_policy.py` into `explicit_sparse.py` if imports are
   acyclic; otherwise document the exact cycle and keep them as the only
   explicit-sparse support files.
2. Merge `preconditioner_caches.py`, `preconditioner_context.py`,
   `preconditioner_operators.py`, and `preconditioner_setup.py` into one
   preconditioning-state owner if doing so deletes at least three files and
   keeps setup/application semantics clearer.
3. Merge `progress.py`, `state.py`, `trace.py`, and `profile_compare.py` into
   one diagnostics/progress owner if they are internal implementation support.
4. Keep QI frozen at its five durable owners. Only delete dead compatibility
   symbols or duplicate implementation blocks; do not create more QI files.

Exit gates:

- `sfincs_jax/solvers` root file count decreases or the plan documents the
  import-cycle blocker.
- `solvers/preconditioners <=35` remains true, with stretch `<=30` only if the
  names stay clearer.
- Solver-selection, explicit sparse, native factor, preconditioner setup,
  progress/trace, QI, PAS, x-block, Schur, symbolic-sparse, full-FP, and
  differentiable implicit-solve tests pass.

Status on 2026-06-26:

- Explicit-sparse consolidation is complete. `explicit_sparse_factor_policy.py`
  and `explicit_sparse_factor_builder.py` were absorbed into
  `sfincs_jax/solvers/explicit_sparse.py`; the old files were deleted and live
  source imports now target the consolidated owner.
- Current metrics after this substep: `174` package Python files, `43`
  package-root files, `165,929` package source lines, `17` solver-root files,
  and `35` preconditioner files. `profile_response/solve.py` remains `5,420`
  lines, `profile_response/sparse/handoff.py` remains exactly `5,500` lines,
  `v3_driver.py` remains `47` lines, and `io.py` remains `64` lines.
- Validation: scoped py_compile and Ruff passed; explicit-sparse, sparse-helper,
  and matrix-reduction tests passed with `29 passed`; preconditioner policy and
  context tests passed with `125 passed`; the broader sparse/profile-response
  gate passed with `618 passed`; import/API/docstring contracts passed with
  `20 passed`; Sphinx `-W` passed; and `git diff --check` passed.
- A sparse-PC branch regression was fixed during validation: requested
  sparse-PC solve setup now initializes the current residual vector even when
  optional factor preflight is disabled, and the auto-preflight retry stage now
  reuses the already-built structured layout instead of eagerly deriving a
  layout from lightweight mocked operators.

#### Batch E - Root/Public Surface And Workflow Classification

Purpose: reduce root-package ambiguity without breaking documented user imports.

Actions:

1. Classify every root module as public API, stable physics kernel,
   compatibility shim, or internal workflow. Record the classification in
   `docs/source_map.rst`.
2. For `scans.py`, `postprocess_upstream.py`, and `data_fetch.py`, move only if
   the public callers can migrate to `api`, `cli`, `workflows`, or
   `validation` without adding root shims. If public compatibility is needed,
   keep the root file and document it.
3. Ensure examples and docs import `sfincs_jax.api`, `sfincs_jax.cli`,
   `sfincs_jax.outputs`, or documented workflow modules rather than private
   sparse/preconditioner internals.

Exit gates:

- Package root is `<=40` files, or remains `<=44` with every remaining root
  file explicitly classified.
- No public example imports private sparse/preconditioner internals.
- CLI, API, scans, upstream postprocess, data-fetch, output-format, docs, and
  import-contract tests pass.

#### Batch F - Profile-Response Internal Line Paydown

Purpose: lower complexity in the largest remaining profile-response owners
without creating more files.

Actions:

1. In `policies.py`, replace duplicated environment/namelist/default-selection
   branches with small policy tables or dataclasses where behavior is identical.
2. In `sparse/xblock.py` and `sparse/qi.py`, delete dead compatibility paths
   and consolidate repeated residual/admission/candidate-selection payloads.
3. In `sparse/handoff.py`, keep only compatibility re-export and orchestration
   glue; move no new code into it unless another file is deleted in the same
   commit.

Exit gates:

- `profile_response/solve.py <=5,500` remains true.
- `profile_response/sparse/handoff.py <=5,500` remains true.
- No new profile-response files are created.
- Package source lines trend downward from 166,045, or any remaining increase
  is justified by deleted files and clearer owner boundaries.
- Focused RHSMode 1, QI, x-block, sparse-PC, Phi1, ambipolar, sensitivity, and
  output-diagnostic tests pass.

#### Batch G - Docs, Tests, Source Map, And Review Gate

Purpose: stop moving names, make the PR reviewable, and prove that the refactor
did not change physics or user behavior.

Actions:

1. Refresh `docs/source_map.rst`, `docs/api.rst`, README developer notes,
   examples, benchmark docs, testing docs, and release notes to the final
   module layout.
2. Consolidate duplicate import-contract tests into owner tests while keeping
   physics, regression, numerical, output-format, CLI, autodiff, docs, and
   representative benchmark gates.
3. Run repository-size hygiene: no generated HDF5/NPZ/profiler outputs, no
   temporary plots, no local absolute-path artifacts, and no accidental large
   files.
4. Run the review-ready validation set.

Review-ready acceptance gates:

- Package source file count is `<=190`; stretch target `<=175` only if clarity
  improves.
- Package source lines are below 165,398, or the remaining increase is
  explicitly justified by deleted files plus clearer ownership.
- Package root has `<=40` files preferred, `<=44` allowed only with explicit
  public/shim labels.
- `v3_driver.py` is deleted or below 80 lines.
- `profile_response/solve.py <=5,500`.
- `profile_response/sparse/handoff.py <=5,500`.
- `io.py <=800` or deleted.
- `problems/profile_response` plus `sparse` has `<=18` files.
- `problems/transport_matrix` plus `parallel` has `<=18` files.
- `solvers/preconditioners` has `<=35` files.
- No top-level `rhs1_*` or `transport_*` implementation files exist.
- No broad package-level lint ignores exist; the only allowed exception is a
  documented, file-local compatibility re-export waiver in
  `profile_response/sparse/handoff.py`.
- Focused tests pass for profile response RHSMode 1, transport matrix RHSMode
  2/3, ambipolar options 1/2/3, RHSMode 4/5 sensitivity contracts, sparse-PC,
  QI admission, output formats, CLI, public API imports, docs, representative
  physics gates, and differentiable linear-solve contracts.
- Sphinx builds with `-W`, `git diff --check` passes, and no temporary outputs
  are tracked.

If these gates pass, PR #8 can move from draft to review-ready. Any remaining
production-runtime optimization, true device-QI promotion, lower-memory native
factor work, or production benchmark refresh belongs in the research/release
lanes below unless it blocks correctness.

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

- Not complete. PR #8 stays draft until Lane 1 Batches A-G pass, docs/source
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

- Lane 1 structural consolidation: about 99 percent. The compatibility-driver
  boundary is done, historical `v3_*` implementation roots are routed or
  deleted, top-level `rhs1_*` and `transport_*` implementation files are gone,
  package-root count is 43, package file count is 174, and `v3_driver.py` is a
  47-line shim. Batch A restored the `profile_response/solve.py <=5,500`
  review gate at 5,420 lines and documented the
  `profile_response/sparse/handoff.py` compatibility waiver while keeping
  `handoff.py` at the 5,500-line gate. Batch B consolidated the active dense,
  active factor, direct reduced-``Pmat``, direct block-Schur, and
  Fortran-reduced LU transport files into
  `problems/transport_matrix/linear_system.py`. Batch C consolidated internal
  transport-parallel policy and sharding ownership into
  `problems/transport_matrix/parallel/runtime.py`, keeping only `worker.py` as
  the documented subprocess executable.
  The remaining blockers are concentrated and measurable:
  `profile_response/policies.py` is 7,425 lines,
  `profile_response/sparse/xblock.py` is 7,725 lines,
  `problems/profile_response` has 18 files including `sparse`,
  `problems/transport_matrix` has 10 files including `parallel`,
  solver-root files are down to 17, `solvers/preconditioners` has 35 files,
  QI preconditioners have 5 files,
  `io.py` is 64 lines,
  `outputs/writer.py` is 4,264 lines, `outputs/transport.py` is 935 lines,
  and package source lines are 165,929. Completed consolidation already removed
  the historical `v3_*`, `rhs1_*`, `transport_*`, transport-output, QI-shard,
  symbolic-sparse, and domain-decomposition file debt. The remaining active
  blockers proceed through larger owner-level batches only: solver
  core/preconditioner-surface consolidation, root/public-surface
  classification, profile-response internal line paydown, and final
  docs/tests/review validation.
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
- The latest local consolidation checkpoints moved the CPU SciPy rescue stage,
  fortran-reduced x-block backend, generic sparse-PC setup stage, and
  direct-tail structured/factor setup stage, plus direct-tail support/preflight
  rescue-policy setup stage, sparse factor-preflight execution, and direct-tail
  auto preflight retry plus true-operator coupled coarse execution and shared
  residual-candidate accept/update logic and the remaining true-active/
  true-window/residual-window correction execution out of
  `profile_response/solve.py` into existing dense/sparse owners. Focused
  owner tests, scoped ruff, py_compile, and sparse/RHSMode-1 coverage passed
  after each checkpoint. The
  final consolidation audit also found that direct deletion of
  `profile_response/sparse/finalization.py` would currently create import-cycle
  risk because `handoff.py`, `xblock.py`, and `fortran_reduced.py` share its
  payload/result types. This file is the authoritative final consolidation
  plan.

Next ordered implementation steps:

1. Execute Lane 1 Batch D: consolidate solver-core policy/context/cache/
   diagnostic files by durable owner without changing automatic solver
   selection, differentiable JAX paths, or CLI fast paths.
2. Execute Lane 1 Batch E: classify the root public surface and
   move only modules that can migrate without creating shims.
3. Execute Lane 1 Batch F and G: reduce oversized profile-response owners
   internally without new files, refresh docs/source maps/examples/tests, run
   the review-ready validation set, and move PR #8 out of draft only if gates
   pass.
4. Keep production option-1/3 ambipolar reruns, production-grid RHSMode 4/5
   parity, large CPU/GPU benchmark regeneration, true device-QI promotion, and
   lower-memory production solver optimization as release-refresh or research
   lanes unless they reveal a correctness regression in the refactor branch.

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
