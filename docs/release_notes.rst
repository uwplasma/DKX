Release notes
=============

Unreleased
----------

- Added the first domain-package skeletons for the active ``v3_driver.py``
  architecture refactor: input, physics, discretization, operators, problems,
  solvers/preconditioners, parallel, workflows, validation, benchmarks, and
  compatibility. Import-contract tests verify that the new packages are
  importable while legacy ``geometry.py`` and ``io.py`` module paths remain
  unchanged until their later migration. The post-skeleton local full suite
  passed with ``2662 passed in 549.92 s``.
- Added ``sfincs_jax/api.py`` with frozen public dataclass contracts for
  normalized solve inputs, geometry/grid/operator summaries, preconditioner and
  solver metadata, transport summaries, output schemas, and benchmark reports.
  These contracts are JAX-free orchestration boundaries; solver-specific pytrees
  remain in numerical modules. The post-contract local full suite passed with
  ``2667 passed in 565.10 s``.
- Extended ``sfincs_jax.api`` with lazy public facades for Python workflows:
  ``write_output``, ``read_output``, and ``run_ambipolar_brent``. These names are
  re-exported from ``sfincs_jax`` and are covered by fast monkeypatched routing
  tests so refactoring can move internal modules without changing public user
  code.
- Added explicit fixed-shape setup-reuse admission metadata to the in-process
  ambipolar evaluator and CLI summary JSON. Ambipolar runs now report whether a
  prior Krylov/setup state existed, whether it was actually admitted for the
  current same-shape solve, the fixed-shape signature used for the decision, and
  a cumulative reuse count.
- Added a matrix-free implicit linear-observable derivative contract in
  ``sfincs_jax.sensitivity`` plus ambipolar radial-current adapters. Production
  problem owners can now provide operator actions, transpose actions,
  parameter-derivative actions, and selected solve/transpose-solve closures
  without assembling dense matrices, while tests compare the matrix-free
  tangent/adjoint certificate against the dense certificate and centered finite
  differences.
- Added ``matrix_free_rhs1_vm_radial_current_linear_observable_system`` as the
  first RHSMode=1 production-facing radial-current derivative builder. It uses
  real full-system matrix-free operator actions, caller-supplied transpose and
  solve closures, finite-difference derivative actions, and the existing
  radial-current observable weights without dense matrix assembly inside the
  builder.
- Extended the RHSMode=1 matrix-free derivative builder with optional
  caller-supplied derivative actions and JAX ``jvp`` operator tangents. The new
  ``operator_tangent_from_centered_difference`` helper builds valid pytrees
  with ``float0`` tangents for static integer/bool leaves, and tests verify the
  JVP action against centered differences on a real electric-field ``xDot``
  operator block.
- Added ``matrix_free_radial_current_derivative_provider`` so ambipolar
  safeguarded Newton/bisection and pure Newton paths can consume matrix-free
  implicit derivative certificates directly. Fast option-1/3-style tests now
  verify root convergence, derivative metadata, tangent/adjoint consistency,
  and finite-difference agreement through the root-solver API.
- Added analytic no-Phi1 ``E_r`` operator-tangent helpers for fixed-shape
  RHSMode=1 operators. ``dphi_hat_dpsi_hat_er_derivative_from_namelist`` reuses
  the v3 radial-coordinate conversion, and
  ``er_operator_tangent_from_dphi_hat_dpsi_hat_derivative`` updates existing
  ``dphi_hat_dpsi_hat`` leaves in the full operator and f-block suboperators.
  Tests verify the analytic JVP action against centered operator differences
  on a real electric-field ``xDot`` fixture.
- Added an explicit ``keep_zero_er_terms`` option to the f-block and full-system
  operator builders. Normal solves keep the previous default branch behavior,
  while derivative/ambipolar gates can retain zero-valued ExB and ``E_r``
  suboperators at ``E_r=0`` so JVP tangents see the same fixed-shape operator
  family used at nearby nonzero fields.
- Added ``rhsmode1_radial_current_response_from_namelist`` as the first
  namelist-backed RHSMode=1 radial-current response/derivative provider. It
  keeps zero-field ``E_r`` branches for fixed-shape derivative gates, reuses
  the analytic/JVP operator tangent path, and is covered by a real small-deck
  implicit derivative versus centered finite-difference test.
- The namelist-backed RHSMode=1 response now uses a Fortran-style active
  pitch-mode dense validation path for small decks, defaults to the Fortran
  ambipolar ``particleFlux_vm_rN`` current convention, and infers radial
  conversion factors from the namelist. A checked
  ``geometry1_helical_small_option1`` regression now reproduces the Fortran v3
  option-1 current and Newton ``dJ_r/dE_r`` slope within ``2e-5`` relative
  tolerance.
- Added small option-3 physical replay coverage for the same active
  namelist-backed RHSMode=1 response. The helical and W7-X-like checked
  Fortran v3 option-3 radial-current points now match within ``2e-5`` relative
  tolerance.
- Added ``solve_rhsmode1_ambipolar_from_namelist`` as a bounded small-deck
  option-1/2/3 root driver over the active namelist-backed RHSMode=1 response.
  The checked helical option-1 and option-3 roots now run through the real
  active ``particleFlux_vm_rN`` response and replay the Fortran v3 roots within
  the documented current and electric-field tolerances.
- Added Fortran-v3 RHSMode=4/5 adjoint-sensitivity source contracts to
  ``sfincs_jax.sensitivity``. The new helpers validate the Fortran-compatible
  adjoint namelist restrictions and return the sensitivity HDF5 fields written
  by ``writeHDF5Output.F90``, including the documented source-code gate for
  ``dParallelFlowdLambda``.
- Added the first compact SFINCS Fortran v3 RHSMode=4 sensitivity reference
  summary: a tiny W7-X-like analytic radial-current sensitivity deck. Tests pin
  the output fields and the identity
  ``dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`` without
  checking in the generated HDF5 file.
- Added a second compact RHSMode=4 sensitivity reference for heat-flux and
  total-heat-flux adjoints, plus reusable output-surface validation for
  Fortran-v3 sensitivity field names and tensor ranks. Tests pin
  ``dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`` without committing the
  generated HDF5 file.
- Added a compact RHSMode=4 parallel-flow/bootstrap sensitivity reference.
  Tests pin the Fortran writer-gated ``dParallelFlowdLambda`` field and the
  identity ``dBootstrapdLambda = sum_s Z_s dParallelFlowdLambda_s``.
- Added the first compact RHSMode=5 constant-current sensitivity reference. It
  runs the small Fortran v3 Brent ambipolar solve, then pins the heat-flux
  adjoint outputs and the extra ``dPhidPsidLambda`` field emitted at ambipolar
  ``E_r``.
- Added a compact RHSMode=4 debug-adjoint finite-difference reference. The
  regression validates every debug output field name/rank, finite selected
  percent errors, and the Fortran NaN mask for lambda/mode entries that the
  finite-difference diagnostic does not fill.
- Added an aggregate RHSMode=4/5 fixture coverage gate so the checked
  references must collectively cover particle flux, heat flux, parallel flow,
  bootstrap current, total heat flux, radial current, constant-current
  ``dPhidPsidLambda``, and debug finite-difference percent-error outputs.
- Moved the first RHSMode=2/3 transport implementation cluster into
  ``sfincs_jax.problems.transport_matrix``: setup, active/dense setup, loop
  support, finalization, streaming outputs, and postsolve diagnostics. The old
  top-level module paths now alias the new modules so existing imports and
  monkeypatch-based debug tests keep working. The post-move local full suite
  passed with ``2668 passed in 562.12 s``.
- Moved the RHSMode=2/3 transport policy and solver-support cluster into
  ``sfincs_jax.problems.transport_matrix``: backend/sparse/recycle policies,
  solve policy, residual-quality gates, handoff policy, host-GMRES rescue,
  KSP iteration diagnostics, and linear-solve dispatch. The old top-level module
  paths remain aliases to preserve existing imports and monkeypatch seams. The
  post-move local full suite passed with ``2668 passed in 556.85 s``.
- Moved the RHSMode=2/3 dense/active/sparse solve-support cluster into
  ``sfincs_jax.problems.transport_matrix``: cached dense LU, batched all-RHS
  dense solves, active block-Schur/coarse factors, and sparse-direct rescue.
  The old top-level module paths remain aliases, so existing user scripts,
  tests, and debug monkeypatches keep working while the maintained source map
  uses the domain package. The post-move local full suite passed with
  ``2668 passed in 556.60 s``.
- Moved the transport parallelism cluster into
  ``sfincs_jax.problems.transport_matrix.parallel``: worker payloads, process/GPU
  execution, runtime merge/partition helpers, persistent-pool management,
  scaling/sharding policy, validation, and the subprocess worker entry point.
  The maintained worker entry point is now
  ``python -m sfincs_jax.problems.transport_matrix.parallel.worker`` for GPU
  worker subprocesses; top-level ``sfincs_jax.transport_parallel_*`` aliases
  were removed in the consolidation pass. Focused parallel/import tests passed with
  ``139 passed``, a broader transport/CLI slice passed with ``169 passed``, and
  the post-move local full suite passed with ``2668 passed in 552.24 s``.
- Moved the RHSMode=2/3 transport preconditioner/direct-operator cluster into
  ``sfincs_jax.problems.transport_matrix``: preconditioner-kind dispatch,
  direct reduced ``Pmat`` emission, direct active block-Schur setup, and
  Fortran-reduced sparse-factor preconditioning. Maintained imports now use the
  ``sfincs_jax.problems.transport_matrix`` package directly; top-level
  ``sfincs_jax.transport_*`` aliases were removed in the consolidation pass.
  Focused preconditioner/direct tests passed with ``117 passed`` and a broader
  transport/preconditioner slice passed with ``148 passed``. The post-move
  local full suite passed with
  ``2668 passed in 554.82 s``.
- Moved the RHSMode=1/2/3 transport diagnostics and transport-matrix assembly
  implementation into ``sfincs_jax.problems.transport_matrix.diagnostics``.
  Existing notebooks and scripts should import this maintained domain module
  directly.
- Moved RHSMode=1 host sparse ILU/LU matvec assembly, CSR factorization, cached
  dense/JAX triangular-factor materialization, and the full-system
  matrix-free adapter into
  ``sfincs_jax.solvers.preconditioners.symbolic_sparse.host_factor``. The
  historical ``v3_driver`` private helper names remain compatibility aliases,
  while the non-differentiable host-factor path now lives in the solver-domain
  package.
- Moved RHSMode=1 profile-response support utilities into
  ``sfincs_jax.problems.profile_response``: residual gates, active-DOF
  decisions, active full/reduced projection, accepted-solve handoff, and solver
  diagnostics. Maintained imports use the canonical
  ``sfincs_jax.problems.profile_response`` modules directly; the top-level
  ``sfincs_jax.rhs1_*`` aliases for these utilities were removed in the
  consolidation pass.
- Consolidated RHSMode=1 profile-response solve-routing and strong
  preconditioner controls into
  ``sfincs_jax.problems.profile_response.policies`` and
  ``sfincs_jax.problems.profile_response.preconditioner_build``. The old
  top-level ``sfincs_jax.rhs1_*`` policy aliases were removed in the
  consolidation pass; the maintained source map, API docs, and driver imports
  now use the domain package. Focused policy/import/driver validation passed with ``282`` tests
  and the post-consolidation local full suite passed with
  ``2670 passed in 579.02 s``.
- Moved the remaining small RHSMode=1 x-block/QI control helpers into
  ``sfincs_jax.problems.profile_response.policies``: guarded PAS-TZ structured
  levels, QI device extra-coarse controls, QI minres-step probe selection, and
  safe x-block fallback initial-guess admission. ``v3_driver.py`` keeps the old
  private names as imported aliases, and focused policy/driver validation
  passed with ``285`` tests. The post-move local full suite passed with
  ``2678 passed in 555.29 s``.
- Moved the constraintScheme=1 x-block moment-Schur wrapper into
  ``sfincs_jax.operators.profile_response.sources`` and the bounded host/device subspace
  residual-equation corrections into
  ``sfincs_jax.problems.profile_response.residual``. The driver now imports
  the historical private names as aliases, while direct algebraic tests cover
  the canonical helper modules.
- Moved the physics-aware x-block post-coarse direction builder into
  ``sfincs_jax.problems.profile_response.residual`` next to the correction
  kernels that consume it. The driver keeps the historical private alias, and
  the sparse-pattern tests now validate the canonical helper path directly.
- Moved residual-correction preconditioner composition, the safe
  non-finite/clipped preconditioner wrapper, and scalar preconditioned-minres
  polish into ``sfincs_jax.problems.profile_response.residual``. The driver
  retains the historical private aliases while direct tests exercise the
  canonical helpers.
- Moved the operator-derived x-block QI coarse-basis and block-metadata helpers
  from ``v3_driver.py`` into ``rhs1_qi_coarse.py``. The driver now imports the
  historical private names as aliases while the canonical implementation lives
  beside the QI coarse-space builders, Galerkin correction, and hard-seed
  basis logic. Focused QI validation passed with ``13`` tests and the broader
  RHSMode=1 QI/device/sparse policy slice passed with
  ``251 passed in 149.96 s``. The post-move local full suite passed with
  ``2680 passed in 555.56 s``.
- Moved the x-block global coarse/load-vector builders and smoothed-load QI
  basis construction from ``v3_driver.py`` into ``rhs1_qi_coarse.py``. The new
  direct tests cover RHS, tail, constraint-source, flux-surface-average, and
  low-angular load labels, smoothed-load rank gating, and driver alias
  compatibility. Focused validation passed with ``15`` tests and the broader
  RHSMode=1 QI/device/sparse policy slice passed with
  ``253 passed in 153.39 s``. The post-move local full suite passed with
  ``2682 passed in 527.23 s``.
- Moved the x-block fixed two-level, host smoothed global-coupling, and device
  global-coupling preconditioner builders from ``v3_driver.py`` into
  ``rhs1_qi_two_level.py``. The driver now imports the historical private
  names as aliases while the canonical implementation lives beside the QI
  two-level primitive. Focused wrapper validation passed with ``13`` tests and
  the broader RHSMode=1 QI/device/sparse policy slice passed with
  ``256 passed in 153.50 s``. The post-move local full suite passed with
  ``2685 passed in 575.33 s``.
- Continued the ``v3_driver.py`` refactor path by moving the coupled
  f/tail-moment and tail-only matrix-free residual-correction builders into
  ``rhs1_lowmode_coarse.py`` with direct tests for tail-selection policy,
  compact metadata, and bounded projection behavior.
- Extracted RHSMode=1 angular domain-decomposition sizing and patch-range
  helpers into ``rhs1_domain_decomposition.py`` with direct tests for
  shard-aware block sizing, overlap clamping, environment override handling,
  and multi-level coarse-block termination. The post-extraction local full
  suite passed with ``2520 passed in 537.83 s``.
- Started the dedicated v3-driver architecture branch by extracting v3 result
  dataclasses, small solver-runtime helpers, and matrix-reduction primitives
  into focused modules with direct tests while preserving compatibility imports
  from ``v3_driver.py``.
- Moved mutable preconditioner hint/context state out of ``v3_driver.py`` and
  into ``preconditioner_context.py``. The driver keeps the same private
  compatibility names, while dtype, structural-tolerance, and solver-JIT policy
  now have direct module tests. The post-extraction local full suite passed
  with ``2531 passed in 508.54 s``.
- Extracted Krylov dispatch, host-only SciPy method routing, concrete solver
  labels, and distributed-GMRES axis resolution into ``solvers/krylov_dispatch.py``.
  ``v3_driver.py`` retains thin compatibility wrappers for monkeypatch-based
  tests and local debugging while the extracted module has direct route-policy
  tests. The post-extraction local full suite passed with
  ``2537 passed in 534.28 s``.
- Moved passive RHSMode=1 and RHSMode=2/3 preconditioner cache dataclasses and
  global cache registries into ``preconditioner_caches.py``. The driver still
  re-exports the same registry objects under the existing private names, so
  existing cache-clearing tests and debugging scripts keep working while the
  containers now have direct lightweight tests. The post-extraction local full
  suite passed with ``2545 passed in 535.59 s``.
- Extracted JAX-native padded-row and compact-CSR triangular sparse-factor
  solves into ``sparse_triangular.py`` with dense-reference tests. The driver
  keeps the old private helper names by import, preserving sparse-preconditioner
  apply behavior while making the kernels independently testable. The
  post-extraction local full suite passed with ``2549 passed in 507.83 s``.
- Extracted preconditioner setup utilities into ``preconditioner_setup.py``:
  chunk-size policy, matrix-free selected submatrix probing, and stable array
  hashing for cache keys. ``v3_driver.py`` keeps a compatibility wrapper for
  submatrix probing so tests and debugging hooks can still monkeypatch the
  driver-level unsharded operator application. The post-extraction local full
  suite passed with ``2553 passed in 541.27 s``.
- Moved RHSMode=1 structured-preconditioner and RHSMode=2/3 transport
  preconditioner cache-key construction into ``preconditioner_setup.py``.
  Driver wrappers keep the historical private function names and live
  ``_precond_dtype()`` behavior, while direct tests now cover key stability,
  Phi1 participation, PAS/FP signatures, and dtype partitioning. The
  post-extraction local full suite passed with ``2556 passed in 543.16 s``.
- Extracted the backend-safe tiny regularized least-squares kernel; it now
  lives in ``solver.py`` with the recycled Krylov initial-guess helper. The
  driver keeps the historical private alias, while direct tests now cover
  dense-reference agreement, near-rank-deficient systems, empty coarse bases,
  and finite autodiff through the helper. The
  post-extraction local full suite passed with ``2558 passed in 542.63 s``.
- Moved host sparse-direct GMRES polish into ``host_refinement.py`` next to the
  existing host direct-refinement kernels. ``v3_driver.py`` keeps a wrapper that
  injects the monkeypatchable driver GMRES solver, while direct tests now cover
  the extracted polish helper with an injected solver and sparse-factor
  preconditioner. The post-extraction local full suite passed with
  ``2559 passed in 543.75 s``.
- Moved transport-worker XLA flag rewriting into the consolidated
  ``problems.transport_matrix.parallel.runtime`` owner next to the backend,
  environment, sharding, and process-pool policy helpers. The driver now
  imports the helper under the historical private name instead of carrying a
  forwarding wrapper, and focused tests cover stale XLA thread/device-cap
  replacement. The
  post-extraction local full suite passed with ``2562 passed in 550.40 s``.
- Extracted shared transport parallel payload handling into
  ``transport_parallel_payload.py``. CPU process workers and GPU subprocess
  workers now use the same injected-dependency payload parser, child-worker
  recursion guard, solve-call construction, merge-ready result packing, and
  NPZ conversion path, with direct tests covering non-contiguous
  ``whichRHS`` chunks. The post-extraction local full suite passed with
  ``2566 passed in 543.33 s``.
- Extracted RHSMode=1/transport constraint-source moment kernels into
  ``rhs1_constraint_sources.py``. The driver now imports the extracted kernels
  under the historical private names instead of carrying forwarding wrappers,
  while direct algebraic tests cover constraintScheme=1 and 2 flux-surface
  averages, density/pressure moments, source injection, and ``pointAtX0``
  handling. The post-extraction local full suite passed with
  ``2570 passed in 542.11 s``.
- Tightened the active refactor rule: extracted functions that need no
  dependency injection, live-global adaptation, or monkeypatch seam should be
  compatibility import aliases, not one-line wrapper bodies. This reduces
  ``v3_driver.py`` without adding new runtime layers and keeps future line-count
  reductions focused on whole solve-orchestration clusters.
- Consolidated the RHSMode=2/3 streamed transport-output accumulator into
  ``outputs.transport``. The driver now delegates per-``whichRHS`` diagnostic
  collection, NTV/source handling, final output-field assembly, and streaming
  HDF5 output to the same output-domain owner, with regression tests comparing
  streamed diagnostics against the established batched transport-output path.
- Consolidated the RHSMode=2/3 all-RHS dense batch solve path into
  ``problems.transport_matrix.solve``. The driver now builds a transport context and
  delegates dense matrix assembly, active-DOF projection, streamed diagnostics,
  residual bookkeeping, and progress logging to a focused helper with direct
  unit coverage plus the existing transport-output regression.
- Consolidated optional RHSMode=2/3 transport KSP iteration-count diagnostics
  into ``problems.transport_matrix.solve``. The production solve loop now calls
  a focused helper for small host SciPy history reruns while preserving the
  same skip/error messages and keeping diagnostic failures non-fatal.
- Consolidated RHSMode=2/3 transport Krylov dispatch into
  ``problems.transport_matrix.solve``. The driver delegates transport-specific
  solver-kind mapping, restart policy, implicit custom-solve routing,
  JIT/non-JIT selection, and distributed residual-solve routing to tested
  solve-owner helpers.
- Consolidated the RHSMode=2/3 sparse-direct rescue implementation into
  ``problems.transport_matrix.solve``. The driver now builds an explicit
  context for pattern probing, direct active FP factors, explicit sparse helper
  setup, fallback sparse-ILU setup, host refinement, float32 polish, and
  float64 retry, preserving the existing sparse rescue behavior with direct
  unit coverage and sparse-direct regressions.
- Extracted RHSMode=1 optional KSP residual-history replay and iteration-count
  diagnostics into ``rhs1_ksp_diagnostics.py``. The driver now keeps only thin
  wrappers that inject the active matvec, preconditioner, emit callback, and
  size/iteration guards, while the SciPy replay and non-fatal diagnostic
  failure paths have direct unit tests.
- Extracted optional Newton-Krylov/Phi1 GMRES history replay into the
  profile-response solver diagnostics owner with direct tests for disabled
  diagnostics, size/iteration skip gates, successful residual-history emission,
  and non-fatal replay failures.
- Extracted explicit sparse host-factor policy parsing into
  ``explicit_sparse.py``. Factor-kind aliases, numeric/boolean
  environment parsing, and monolithic LU/ILU guard sizing now have direct tests
  while the driver keeps the monkeypatch-sensitive operator/factorization seam.
- Moved the remaining explicit sparse host-factor environment bundle into the
  typed ``ExplicitSparseFactorSettings`` policy object. Default/override
  parsing for dense/CSR budgets, pattern probing, symbolic Schur/frontal/ND/BLR
  settings, SuperLU options, and ILU options is now tested in one focused
  owner.
- Extracted explicit sparse host-factor assembly/factorization orchestration
  into ``explicit_sparse.py``. ``v3_driver.py`` now keeps a
  compatibility wrapper that injects the current operator-build, pattern-build,
  factorization, backend, and guard callbacks, preserving existing debug and
  monkeypatch seams while removing another large block from the monolith.
- Extracted RHSMode=1 direct-tail structured-preconditioner cache and memory-cap
  policy into ``rhs1_direct_tail_policy.py``. The driver still re-exports the
  private compatibility names, but cache keys, cache metadata, direct reduced
  Pmat aliases, adaptive memory caps, and the structured host sparse adapter
  now have focused tests outside the main solve loop.
- Extracted RHSMode=1 true-operator rescue support bundles and helper routines
  into ``rhs1_true_operator_rescue.py``. The solver builders remain in
  ``v3_driver.py`` for this checkpoint, while residual-window/coarse bundle
  solves, reusable true-action column caching, graph expansion, sparse-factor
  storage estimates, and residual-window target selection now have direct tests
  outside the driver monolith.
- Extended ``rhs1_true_operator_rescue.py`` with the residual sparse-window and
  residual-coarse builder routines plus active reduced-residual diagnostics.
  Existing ``v3_driver`` private names remain import-compatible while the
  builder and diagnostic behavior is now directly testable in the focused
  module.
- Moved the remaining RHSMode=1 true-operator LSQ rescue builders into
  ``rhs1_true_operator_rescue.py``: residual-window LSQ, deterministic
  active-block LSQ, active-residual-block LSQ, active-submatrix, and
  coupled-coarse correction construction now live with the true-action cache and
  residual-window utilities they use.
- Extracted RHSMode=1 Fortran-reduced constraintScheme=1 direct-tail sparse
  operator materialization into ``rhs1_fortran_reduced_direct_tail.py``. The
  driver injects the structured full-CSR callback so monkeypatch/debug seams are
  preserved, while the source-column, moment-row, and active term-level
  ``whichMatrix=0`` assembly logic is no longer embedded in the main solve
  orchestrator.
- Extracted the RHSMode=1 structured full-CSR ``SparseOperatorBundle`` wrapper
  into ``rhs1_structured_full_csr.py``. The analytic full-CSR assembly still
  lives in ``rhs1_full_assembly.py``; the new module owns the runtime sparse-PC
  adapter, active projection, memory-budget admission, and no-probe metadata
  emission previously embedded in ``v3_driver.py``.
- Extracted RHSMode=1 and RHSMode=2/3 preconditioner-operator shaping into
  ``preconditioner_operators.py``. Point, theta/zeta line, theta/zeta
  domain-decomposition, and Fortran-reduced operator builders are now pure
  dataclass/JAX transformations outside the solve driver, with the existing
  driver private names preserved as import aliases.
- Consolidated RHSMode=2/3 active-system, direct reduced-``Pmat``, exact active
  transport-operator sparse emission, direct block-Schur setup, and full-FP
  Fortran-reduced LU preconditioner code into
  ``sfincs_jax.problems.transport_matrix.linear_system``. Focused tests verify
  the emitted CSR matrices against the matrix-free active operator, physics
  coarse-basis source/constraint columns, direct block-Schur callback path, and
  Fortran-reduced LU symbolic/BLR/ND metadata.
- Moved the RHSMode=1 full-FP sparse x-block/TZ preconditioner into
  ``sfincs_jax.solvers.preconditioners.xblock.tz_sparse``. The module owns
  host/JAX x-block factor setup, compact CSR and padded triangular apply,
  skipped-block diagonal fallback, and extra-variable Schur handling; the old
  top-level ``sfincs_jax.rhs1_xblock_tz_sparse`` alias was removed in the
  consolidation pass.
- Moved the PAS-only RHSMode=1 sparse x-block ILU/LU preconditioner into
  ``sfincs_jax.solvers.preconditioners.pas.xblock_ilu``. The module owns the
  per-``(species,x)`` Legendre/theta/zeta block assembly, PETSc-style
  ILU/exact-LU setup policy, padded triangular-factor apply, threaded factor
  build, cache storage, and extra-variable Schur solve; the old
  top-level ``sfincs_jax.rhs1_pas_xblock_ilu`` alias was removed in the
  consolidation pass.
- Consolidated RHSMode=2/3 post-solve diagnostic assembly into
  ``problems.transport_matrix.finalize``. The owner now covers streamed versus
  batched diagnostic selection, rematerialization/precompute/chunking policy,
  final flux-array assembly, optional output-field propagation, and transport
  matrix construction after the Krylov solve loop.
- Extracted the parent-side RHSMode=2/3 parallel solve branch into
  ``transport_parallel_solve.py``. The new module owns ``whichRHS`` partitioning,
  CPU/GPU worker launch through injected runtime hooks, worker payload merging,
  parallel-result diagnostics assembly, and early transport-matrix result
  construction.
- Consolidated the initial RHSMode=2/3 transport solve policy into
  ``problems.transport_matrix.policies``. The driver delegates geometryScheme parsing,
  low-memory output routing, streamed-diagnostic/state-vector retention,
  force-dense/force-Krylov handling, dense fallback admission, dense memory-cap
  blocking, and GMRES restart/max-iteration guards to a focused policy object
  before active-DOF and preconditioner setup.
- Consolidated RHSMode=2/3 sequential-loop matvec caching and recycle-basis
  bookkeeping into ``problems.transport_matrix.solve``. The driver delegates
  cached full/reduced matvec closure construction, recycle-size admission,
  stored-state recycle seeding, basis trimming, and recycled initial-guess
  construction to a focused helper before the per-``whichRHS`` solve branches.
- Moved the sequential RHSMode=2/3 post-``whichRHS`` residual/ETA bookkeeping
  into the same helper. The driver now records elapsed times, emits residual
  summaries, applies configured residual abort gates, and reports remaining-time
  estimates through ``TransportLoopProgress`` instead of carrying mutable
  progress state inline.
- Moved the RHSMode=2/3 per-``whichRHS`` loop policy into
  ``problems.transport_matrix.policies``. The driver delegates E_parallel loose/Krylov
  routing, constraint-nullspace projection admission, optional KSP iteration-stat
  settings, and dense-batch fallback admission to a focused policy object before
  entering the sequential solve branches.
- Extracted sequential RHSMode=2/3 branch-finalization bookkeeping into
  ``transport_solve_finalization.py``. The helper now owns reduced/full
  accepted-state recording, optional constraint projection, true-residual
  recomputation, streamed-output collection, recycle-basis updates, solver-method
  recording, and KSP iteration-stat dispatch, with direct unit coverage for dense
  fallback accepted-state overrides.
- Extracted the constraintScheme=1 nullspace/source-row projection into
  ``constraint_projection.py``. RHSMode=1 and RHSMode=2/3 solve paths keep the
  historical private driver helper names as compatibility aliases, while direct
  tests now cover no-op admission, environment disablement, transport roundoff
  skip behavior, source-row residual reduction, and the driver alias contract.
- Extracted RHSMode=2/3 entry setup into ``transport_solve_setup.py``. The
  driver now delegates transport max-iteration environment overrides, optional
  Krylov state checkpoint loading, ``whichRHS`` subset normalization, and
  CPU/GPU process-parallel worker request resolution to focused helpers with
  direct unit coverage.
- The transport linear-system owner also owns RHSMode=2/3 active-DOF and
  dense-path setup: initial output/restart policy, active-index compaction
  state, dense fallback and dense preconditioner admission, and ordered
  user-facing notes before preconditioner setup and the transport loop.
- The post-extraction strict docs build passed, the repo-size audit now has no
  reviewed files above 2 MiB after ``v3_driver.py`` dropped below the threshold,
  and the latest local full suite after the active/dense setup extraction passed
  with ``2659 passed in 551.86 s``.

v1.1.7
------

This patch release includes the regenerated validation artifacts from the
previous release pass and continues the ``v3_driver.py`` maintainability path
without changing the public solve interface. RHSMode=1 low-mode angular and
moment coarse-space helpers now live in ``rhs1_lowmode_coarse.py`` with direct
unit tests for feature construction, bounded matrix-free Galerkin correction
metadata, and projection behavior. The driver still owns solve orchestration,
but another algebraic preconditioner seam is now documented and independently
testable as part of the path toward higher meaningful coverage. The release
candidate passed the full local suite with coverage on 2026-06-13:
``2510 passed in 746.33 s`` with total package coverage at ``74%``.

v1.1.6
------

This patch release keeps the ``v1.1.5`` solver and documentation state, then
stabilizes the Linux/JAX CI gate for the active-ladder RHSMode=1 auto-selection
test. The affected test still requires a residual at the requested tolerance
scale and still checks the selected solver path; it no longer fails on a small
last-iteration roundoff difference across JAX builds.

v1.1.5
------

This patch release promotes the current residual-clean RHSMode=2/3 full-FP
transport preconditioner into ``auto``, refreshes the QA/QH bootstrap-current
documentation artifacts, and keeps production-resolution research lanes scoped
to their checked evidence.

Highlights
~~~~~~~~~~

- Promoted the residual-clean RHSMode=2/3 full-FP direct-Pmat LU
  preconditioner into ``auto`` for eligible non-Phi1 transport-matrix runs.
  ``auto`` now tries this PETSc-like route by default with strict residual
  admission and memory caps; set
  ``SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO=0`` to disable it for a
  benchmark campaign. Lower-memory symbolic/native replacements remain gated
  until production-floor CPU/GPU evidence passes.
- Fixed a multi-device RHSMode=1 transformed-matvec bug found by the next QI
  ``nfp=2`` kinetic single-point probe. Preconditioner submatrix setup and
  custom-linear-solve matvecs now avoid entering ``jax.set_mesh``/``pjit`` from
  inside JAX transforms, while top-level matvecs still use the sharded cached
  path. The checked ``13 x 13 x 15 x 4`` CPU probe at ``E_r=0.3`` converges
  with residual ``2.09e-18`` against target ``1.47e-11``. A bounded sparse-LU
  skip-primary policy then reduces the measured solver time from about
  ``108 s`` to about ``35 s`` with identical key observables and records that
  one-device and explicit full-system sparse-host routes are not promotion
  candidates for this rung.
- Added a bounded eight-point CPU solver-policy audit for the same QI
  ``13 x 13 x 15 x 4`` rung. The sparse-LU skip-primary policy completes the
  scan in ``263.1 s``, with mean solve time ``32.85 s``, max solve time
  ``35.72 s``, all residual gates passing, and a fixed-resolution electron root
  at ``E_r=2.2153427467``. This closes the CPU part of the rung and keeps the
  public production-resolution QI claim gated on backend/reference and
  resolution-ladder evidence.
- Added the matching fixed-resolution GPU and Fortran-v3 comparison for the QI
  ``13 x 13 x 15 x 4`` rung. CPU/GPU selected roots agree to ``4.8e-14`` and
  CPU/Fortran-v3 agrees to ``7.4e-9``. Checked CPU/GPU observables agree within
  ``5.2e-13`` relative; CPU/Fortran-v3 differs by at most ``1.8e-3`` on
  ``FSABFlow`` and below ``9e-6`` on particle/heat fluxes. The GPU route is
  correctness-clean but still performance-open because it safely enters host
  sparse LU and is slower than CPU at this size.
- Added the next QI ``15 x 15 x 17 x 4`` CPU/Fortran-v3 rung and fixed the
  sparse x-block policy cliff it exposed. The previous default spent about
  ``316.9 s`` on redundant x-block setup for the ``E_r=0.3`` point before
  falling through to the exact active sparse-LU solve. The new default skips
  that redundant setup for mid-size systems covered by the direct sparse-LU
  cap, reducing the point to about ``69.4 s`` with the same residual and key
  observables. The eight-point CPU scan completes in ``535.8 s`` with all
  residual gates passing, selects ``E_r=2.2132389239``, and agrees with the
  SFINCS Fortran v3 selected root to ``9.2e-7`` relative.
- Added a guarded matrix-free QI-device operator-reuse route for explicit
  RHSMode=1 x-block Krylov runs. When ``xblock_sparse_pc_gmres`` is requested
  with the QI-device matrix-free preconditioner installed in Krylov, the driver
  can now skip local sparse x-block factor construction and report the decision
  in solver metadata. The existing host-sparse fallback remains unchanged when
  the guarded route is not requested or cannot be built. This is infrastructure
  for the next one-GPU QI timing gate, not yet a production true-device-QI
  performance claim.
- Production-sized nonconverged RHSMode=1 ``write-output`` runs now write the
  requested JSON solver-trace sidecar before refusing to write HDF5/NetCDF/NPZ
  diagnostics. The physical output gate remains fail-closed, but failed large
  runs preserve solver path, residual, matvec, memory-estimate, and
  preconditioner metadata for debugging.
- JIT-cycle device Krylov metadata now reports internal restart-cycle
  iterations and estimated matvecs instead of only Python-visible wrapper
  matvecs. Solver traces keep ``python_matvecs`` separately so long GPU runs no
  longer look artificially cheap when the Krylov work is inside a compiled
  device loop.
- Added a checked fail-closed office-GPU artifact for the QI ``13 x 13 x 15 x 4``
  matrix-free operator-reuse route. It verifies operator-reuse activation,
  local x-block factor skipping, failure-safe trace writing, coupled-residual
  setup, and corrected device-cycle accounting, while explicitly keeping
  residual convergence failed and production true-device-QI performance
  deferred.
- Added the second refined QI ``nfp=2`` kinetic promotion rung at
  ``11 x 11 x 13 x 4`` after fixing a mid-size RHSMode=1 full-FP solver-policy
  cliff. The bounded dense policy now covers active sizes up to ``8000`` and
  ``scan-er`` writes per-point solver-trace sidecars. The checked CPU scan
  dropped from about ``326 s`` on the old automatic fallback to about ``23 s``;
  the matching office GPU0 scan also completes in about ``23 s``. CPU/GPU roots
  agree to ``2.5e-13`` and the Fortran-v3 selected root agrees within the
  documented ``2e-6`` refined-grid tolerance. The remaining resolution drift
  keeps production-resolution QI open.
- Added the first refined QI ``nfp=2`` kinetic promotion rung at
  ``9 x 9 x 11 x 4``. The two-species ion/electron scan passes CPU/GPU/Fortran
  fixed-resolution gates with CPU/GPU selected
  ``E_r = 2.2834299271``, CPU/GPU root difference ``4.3e-14``, and Fortran-v3
  selected ``E_r = 2.2834273232`` within the documented refined-grid
  tolerance. The low-to-refined root drift is still about ``0.155``, so
  production-resolution QI remains an open research lane.
- Fixed the promotion-audit default for two-species electron-root scans:
  ``--impurity-species-index`` is now optional, and no-impurity CPU/GPU
  comparisons automatically allow missing flux-objective scalars while still
  checking selected roots, bootstrap objectives, residuals, and backend
  agreement.
- Added residual-region/bounce-region QI coarse-reuse evidence plumbing and
  runtime documentation while keeping true differentiable device-QI
  production-resolution closure fail-closed.
- Added opt-in augmented-seed Krylov recycling and active-pattern coarse
  infrastructure with finite/shape guards and fail-closed evidence
  classification.
- Added a coupled residual-equation primitive for the next true device-QI
  architecture. It solves accepted coarse variables together instead of as a
  staged cascade, so Schur/multilevel cross-couplings can update earlier coarse
  coefficients without using smoother or restart tuning.
- Wired the coupled residual-equation path into RHSMode=1 driver progress,
  solver-trace metadata, and the ``coupled-residual-device-qi`` evidence
  preset. The path remains opt-in and fail-closed until a bounded CPU/GPU
  hard-seed artifact writes converged output and trace metadata.
- Added an opt-in Krylov-install mode for validated coupled residual-equation
  stages whose seed probe is rejected. This tests the preconditioner in the
  mathematically relevant Krylov context without relaxing convergence gates or
  changing the initial seed.
- Recorded the first scale-0.60 GPU coupled-residual Krylov-install evidence.
  It improves the coupled-probe runtime and host RSS versus the seed-gated
  attempt but still fails the residual/write gate, so true device-QI remains
  an explicit research lane.
- Hardened QI evidence extraction for failed long GPU runs so coupled
  residual-equation and install-in-Krylov progress lines remain visible in
  compact artifacts and fail-closed manifests even when no HDF5 output or
  solver trace is written.
- Added an opt-in post-Krylov residual-equation correction for RHSMode=1
  x-block solves. It reuses cached QI ``(U, A U)`` columns and final-residual
  physics directions in a bounded JAX least-squares solve, records metadata and
  output diagnostics, ships with a ``post-residual-equation-device-qi`` evidence
  preset, and remains fail-closed until hard-seed CPU/GPU evidence converges.
- Recorded the first post-residual-equation scale-0.60 CPU and GPU artifacts.
  They accept true-residual corrections but still refuse production output, so
  they are blocker evidence for the next coarse-space design rather than release
  promotion evidence.
- Updated QI evidence counts, multi-GPU wording, and source-map closure text so
  release-facing docs distinguish production host fallback from research
  device-QI probes.

v1.1.4
------

This patch release packages the 2026-05-16 research-lane safety and planning
push. It adds reusable QI/PAS/parallelism infrastructure while keeping public
claims scoped to checked release evidence.

Highlights
~~~~~~~~~~

- Added a standalone JAX-compatible RHSMode=1 QI block-Schur/angular/radial
  coarse-preconditioner primitive in ``sfincs_jax/rhs1_qi_block_schur.py``.
  The primitive builds deterministic block, radial, angular, and Schur-like
  basis directions; applies a local-plus-coarse action; and exposes a
  fail-closed true-residual probe with rank and conditioning metadata. It is
  not promoted as a production device-QI solve until wired into the driver and
  validated on the scale-0.60 CPU/GPU hard-seed gates.
- Strengthened PAS matrix-free memory guards with ``PasRuntimeChunkPlan`` and
  ``plan_pas_runtime_chunks``. PAS candidate and norm reductions can now derive
  bounded chunks from configured byte budgets, and tight budgets fail before
  launching a matvec or correction.
- Added release-safe single-case sharded-solve planning metadata in
  ``sfincs_jax/problems/transport_matrix/parallel/runtime.py``. The helper caps
  requested devices to available work, records per-device balance diagnostics,
  and fail-closes release scaling claims for experimental single-case sharding.
- Refreshed QI evidence metadata to include the scale-0.60 smoothed-load and
  probed moment-Schur CPU artifacts. The moment-Schur probe rejected itself
  after worsening the hard-seed residual, preserving the baseline x-block path.

Validation
~~~~~~~~~~

- Focused local validation for the new lanes passed:
  ``135 passed`` across QI block-Schur, PAS matrix-free/policy, PAS benchmark,
  sharding planner, sharded benchmark, and transport-parallel tests.
- Full local validation passed: ``1651 passed in 712.28 s``.
- Release-gate and research-lane checks remain green, and the Sphinx docs build
  passes with warnings treated as errors.
- Package build and distribution metadata checks passed with ``python -m build``
  and ``twine check dist/*``.
- The package version is ``1.1.4`` in both ``pyproject.toml`` and
  ``sfincs_jax.__version__`` so the PyPI workflow can validate the matching
  ``v1.1.4`` tag.

Remaining research lanes
~~~~~~~~~~~~~~~~~~~~~~~~

No new release-facing claim depends on true device-QI, production-resolution QI
ladders, or single-case multi-GPU strong scaling. Those lanes remain explicitly
deferred until checked CPU/GPU artifacts pass their promotion gates.

v1.1.3
------

This patch release candidate packages the 2026-05-15 release-narrative cleanup,
bounded large-QI fallback work, PAS probe gating, and release-claim hardening.

Highlights
~~~~~~~~~~

- Added an explicit release scope: current claims cover the audited example-suite
  parity, bounded large-QI non-autodiff host fallback, PAS/runtime probe gates,
  and transport-worker parallelism. Production-resolution QI CPU/GPU ladders,
  true differentiable device-QI, and single-case multi-GPU strong scaling remain
  deferred or experimental research lanes.

- Added an explicit ``pas_tzfft`` / ``pas_fft`` RHSMode=1 PAS preconditioner
  candidate and a guarded ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=tzfft``
  route for memory-unsafe PAS-TZ experiments.
- Guarded PAS-TZ fallbacks now skip stage-2 GMRES by default unless
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY=1`` is set. This keeps
  experimental memory fallback probes bounded when a candidate lowers the
  residual enough to avoid the generic high-ratio stage-2 skip but still misses
  the strict solve target.
- The bounded geometryScheme=4 PAS fallback smoke artifact now includes
  ``tzfft``. It improves the residual from the cheap-collision fallback
  (``~6.4e5``) to ``~1.9e-4`` in about ``3.3 s`` on the checked local smoke,
  but remains opt-in because it increases RSS and still misses the strict
  residual target.
- Added ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION=tzfft`` as an opt-in
  cheap-base plus matrix-free streaming-correction probe. It is bounded and
  modestly improves the cheap collision fallback, but the checked geometry4
  smoke does not meet the promotion gate.
- Large-QI explicit device-Krylov requests now have a metadata-visible
  non-autodiff host fallback that enters the measured host x-block auto policy
  before JAX factors are built. This is the current production escape hatch, not
  an end-to-end differentiable device-QI claim.
- PAS matrix-free production probes now require explicit candidate byte budgets
  by default, and the checked geometry4/HSX real-solve probes are documented as
  negative promotion evidence because they are residual-clean but do not improve
  runtime or memory.
- Release-gate and research-lane manifests now keep deferred validation lanes,
  device-QI work, and single-case scaling from silently blocking a tag while
  still preserving concrete promotion gates.

Validation
~~~~~~~~~~

- Release-gate and research-lane checks pass against the updated manifests.
- Sphinx documentation builds with warnings as errors after this release-note and
  scope refresh.
- Focused version metadata validation requires ``pyproject.toml`` and
  ``sfincs_jax.__version__`` to agree on ``1.1.3``.

Remaining research lanes
~~~~~~~~~~~~~~~~~~~~~~~~

No documented release-facing lane is blocked by true device-QI or single-case
strong-scaling work. Production-resolution QI CPU/GPU seed ladders, the
scale-0.60 one-GPU hard seed, and geometry-rich PAS runtime/memory wins remain
post-release promotion candidates until checked artifacts pass their gates.

v1.1.2
------

This patch release closes the post-v1.1.1 structured-PAS fallback push and
hardens the release workflow.

Highlights
~~~~~~~~~~

- Memory-unsafe PAS-TZ fallback routes are now bounded uniformly. The default
  route is the cheap collision fallback when available; the historical
  ``hybrid`` fallback remains available for A/B profiling with
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK=hybrid``. All guarded fallback
  routes skip the expensive automatic strong-preconditioner retry unless
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1`` is set.
- Guarded PAS-TZ and weak PAS forced/probe paths use accept-only matrix-free
  minimal-residual corrections before fail-fast classification. These
  corrections reuse the already-built preconditioner, store no dense angular
  patch inverses, and are kept only when the measured residual decreases.
- Forced weak PAS paths (``collision``, ``xmg``, and ``point``) now skip
  stage-2/strong retry only at enormous residual ratios by default, preventing
  minutes-long profiling stalls without changing moderate-residual polish
  behavior.
- Release metadata, production-benchmark workflow checks, and public
  runtime/validation figures were refreshed for the current ``main`` artifacts.
- The PyPI workflow now validates that ``pyproject.toml``,
  ``sfincs_jax.__version__``, and pushed release tags agree before publishing.

Validation
~~~~~~~~~~

- Current release-facing CPU/GPU suite artifacts remain ``39/39 parity_ok`` with
  zero strict mismatches, no ``jax_error``, and no ``max_attempts``.
- Bounded PAS-TZ fallback smoke now returns for ``collision``, ``hybrid``,
  ``zeta``, and ``theta`` under the 15 s local gate. These rows are
  intentionally documented as negative, non-promoted baselines because their
  residuals remain large.
- The bounded artifact is checked in at
  ``tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json``
  and guarded by ``tests/test_solver_path_artifacts.py``.
- Local release validation passed with ``1134 passed in 501.80 s``.

Remaining research lane
~~~~~~~~~~~~~~~~~~~~~~~

No release blocker remains in the documented workflows. The remaining
publication-scale optimization lane is still algorithmic: a genuinely stronger
matrix-free line/plane smoother or iterative chunked Schwarz correction for
geometry-rich PAS systems that clears the fixed gate of under 60 s, no measured
memory regression, and at least 100x residual reduction before any default
promotion.

v1.1.1
------

This patch release ships the final PAS/full-FP performance and memory closeout
after the v1.1.0 validation release.

Highlights
~~~~~~~~~~

- One-species PAS+Er sparse-PC defaults now use the measured
  ``MMD_AT_PLUS_A`` ordering and a bounded GMRES restart policy unless the user
  explicitly overrides the restart environment variable.
- Phi1 fast-explicit solves use a production-size restart helper that preserves
  output parity while reducing wasted Krylov storage on larger active systems.
- RHSMode 1 no-Phi1 single-state output avoids retaining an unnecessary stacked
  solved-distribution copy before diagnostic writeout.
- The README-facing runtime/memory and W7-X high-``nu`` performance figures were
  regenerated from the checked-in release artifacts.
- The production-resolution ``geometryScheme4_2species_PAS_noEr`` stress case is
  now explicitly closed for this release as ``no safe existing default
  promotion``. CPU and GPU candidate routes all hit the bounded 300 s gate, so no
  unsafe solver-path default is promoted.

Validation
~~~~~~~~~~

- Local full suite: ``1115 passed in 498.10 s``.
- GitHub Actions for the closeout commit: CI and Docs both passed.
- The large geometry-rich PAS closeout artifact is checked in at
  ``tests/reference_solver_path_artifacts/geometry4_large_pas_closeout_2026-05-09.json``
  and guarded by ``tests/test_solver_path_artifacts.py``.

Remaining research lane
~~~~~~~~~~~~~~~~~~~~~~~

No release blocker remains. The next research optimization target is a
structured/chunked geometry-aware PAS preconditioner for production-resolution
geometry-rich 3D cases; heuristic promotion of existing Schur, sparse-PC, or
PAS-lite paths is intentionally blocked until a measured route clears the
runtime, memory, residual, and Fortran-comparison gates.

v1.1.0
------

This release promotes the current CPU/GPU validation and performance work into the
first minor release after the 1.0 series.

Highlights
~~~~~~~~~~

- The audited 39-case example suite remains clean on CPU and GPU: no practical
  mismatches, no strict mismatches, no ``jax_error`` cases, and no ``max_attempts``
  cases in the release-facing artifacts.
- GPU solver-path selection is less aggressive for bounded full-collision and
  PAS systems. Moderate full-FP systems can stay on dense accelerator solves when
  that is faster and lower-memory, while bounded geometry-rich PAS examples now
  prefer the measured lower-memory ``pas_tz`` path. On CPU, audited 3D full-FP
  RHSMode 1 cases can auto-select sparse-PC GMRES inside the measured size
  window when it beats dense FP on both runtime and memory. On GPU/CUDA,
  production-floor tokamak full-FP no-Er/Er rows can auto-select sparse-PC GMRES
  inside narrow measured windows when the matrix-free route is not residual-clean
  and the faster theta-line route is too memory-heavy.
- Monoenergetic transport benchmarks now time the actual RHSMode 2/3 transport
  solve instead of only output-field assembly, and small bounded GPU cases can use
  dense accelerator transport when it is validated to be faster.
- The CLI and Python output paths support HDF5, NetCDF4, and NPZ by output suffix,
  and ``sfincs_jax --plot`` writes a PDF diagnostics panel from existing output
  files.
- Documentation covers the drift-kinetic equation being solved, geometry loading,
  normalizations, solver paths, output datasets, validation gates, performance
  techniques, and release-maintainer checks.

Validation artifacts
~~~~~~~~~~~~~~~~~~~~

The release-facing validation roots are:

- ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28``

The latest focused GPU performance pass measured:

- ``HSX_PASCollisions_fullTrajectories``: ``10.539 s`` / ``2042 MB`` to
  ``8.469 s`` / ``1577 MB``, with zero Fortran mismatches.
- ``sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories``:
  ``7.716 s`` / ``2098 MB`` to ``6.413 s`` / ``1609 MB``, with zero Fortran
  mismatches.
- ``monoenergetic_geometryScheme1``: ``13.039 s`` / ``996 MB`` to ``3.541 s`` /
  ``981 MB``, with zero Fortran mismatches.

Remaining research lanes
~~~~~~~~~~~~~~~~~~~~~~~~

No correctness blocker remains in the documented release-facing suite. The main
future optimization lane is allocator and work-array lifetime reduction for the
heaviest RHSMode 1 PAS Krylov/diagnostic paths. Single-case strong multi-GPU
scaling remains a research feature; release-facing parallel guidance continues to
prefer case-parallel and transport-worker throughput.
