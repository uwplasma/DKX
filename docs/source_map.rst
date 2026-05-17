Source-code map
===============

This page links the main pieces of the mathematics to the source files that implement
them. The goal is to shorten the path from an equation in the docs to the exact module
that evaluates it.

High-level flow
---------------

For a standard solve, the execution path is:

1. parse namelist and resolve equilibrium inputs,
2. build grids and geometry coefficients,
3. construct the operator / residual objects,
4. choose a solve path and preconditioner,
5. run the linear or nonlinear iteration,
6. postprocess diagnostics and write ``sfincsOutput.h5``.

Core modules
------------

``sfincs_jax/cli.py``
^^^^^^^^^^^^^^^^^^^^^

Public command-line interface:

- ``sfincs_jax input.namelist`` default solve mode,
- ``write-output``,
- ``transport-matrix-v3``,
- comparison and utility commands,
- parallel runtime/bootstrap flags.

``sfincs_jax/io.py``
^^^^^^^^^^^^^^^^^^^^

Input/output orchestration:

- reads namelists,
- resolves equilibrium overrides (including ``wout_path``),
- writes ``sfincsOutput.h5``,
- materializes output diagnostics,
- exposes the in-memory results API.

``sfincs_jax/input_compat.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compatibility and search-order logic for equilibrium files, input normalization, and
user overrides. This is the module to inspect first when a case fails to find a VMEC or
Boozer file.

``sfincs_jax/jax_geometry_adapters.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional structural adapters for JAX-native geometry producers such as ``vmec_jax``
and ``booz_xform_jax``. These helpers do not make either package a required
dependency; they normalize in-memory VMEC-like ``wout`` objects to the internal
``VmecWout`` layout used by ``geometryScheme=5``.

``sfincs_jax/grids.py`` and ``sfincs_jax/xgrid.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Velocity-space discretization:

- collocation points in :math:`x`,
- quadrature weights,
- modal transforms used by the collision operator,
- special handling for monoenergetic ``RHSMode=3``.

``sfincs_jax/geometry.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry loading and normalized coefficient generation:

- analytic model fields,
- VMEC-derived coefficients,
- Boozer ``.bc`` evaluation,
- surface metrics and scalar geometry diagnostics.

``sfincs_jax/vmec_geometry.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC ``geometryScheme=5`` Fourier-sum evaluator. The public file path
``vmec_geometry_from_wout_file(...)`` reads ``wout`` data and delegates to
``vmec_geometry_from_wout(...)`` so optional in-memory producers can exercise the
same formulas and parity tests without hidden file I/O.

``sfincs_jax/collisionless.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streaming and mirror-force contributions in the Legendre basis.

``sfincs_jax/collisionless_exb.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`E\times B` terms in the kinetic operator, including angular advection and
the radial-electric-field contributions to :math:`\dot \xi` and :math:`\dot x` where
supported.

``sfincs_jax/magnetic_drifts.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Magnetic-drift coefficient construction, angular advection terms, upwinding masks, and
associated :math:`\partial_\xi` couplings.

``sfincs_jax/collisions.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collision models:

- PAS,
- full linearized Fokker-Planck,
- field-particle terms,
- Phi1-modified collision coefficients.

``sfincs_jax/v3_system.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

System construction:

- state-vector ordering,
- operator block composition,
- transport-RHS rewrites,
- cached operator application,
- system metadata used by the driver and diagnostics.

``sfincs_jax/residual.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Residual and source-term helpers. This is where the thermodynamic drives and other RHS
pieces are assembled before being fed to the solve stack.

``sfincs_jax/v3_driver.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Top-level solve orchestration. This file controls:

- solver selection,
- preconditioner selection,
- bounded rescue paths,
- transport-worker parallelism,
- sharded experimental paths,
- output-field collection.

When a solve behaves differently on CPU and GPU, this is usually the first file to
inspect.

On the active refactor branch, the main policy layers are being split out of the
monolith into narrower modules while keeping ``v3_driver.py`` as the stable public seam
for debugging and monkeypatch-based tests. The first extracted layers are:

- ``sfincs_jax/rhs1_pas_policy.py``:
  PAS applicability, PAS-TZ memory safety, PAS fallback routing, and PAS
  adaptive-smoother eligibility.
- ``sfincs_jax/rhs1_pas_matrixfree.py``:
  bounded matrix-free PAS correction probes, streaming L2 norms, candidate
  byte-budget preflights, and ``PasRuntimeChunkPlan`` metadata for keeping
  PAS-heavy residual/correction reductions inside configured memory budgets
  before a matvec is launched.
- ``sfincs_jax/rhs1_qi_block_schur.py``:
  standalone JAX-compatible QI block-Schur/angular/radial coarse-preconditioner
  primitive. It builds deterministic global, radial, angular, and block-Schur
  basis directions, applies a local-plus-coarse action, and exposes a
  fail-closed true-residual probe for future device-QI expansion.
- ``sfincs_jax/rhs1_qi_deflation.py``:
  residual-deflated, device-compatible QI preconditioner primitive. It builds a
  bounded preconditioned-residual Krylov basis, optionally merges
  physics-informed block-Schur directions, applies a local-plus-deflated
  least-squares action, and fail-closes on true-residual probes. It also
  provides the seed-only cycle-minres helper used by QI hard-seed evidence:
  repeated fixed-basis residual corrections are combined by a small
  ``min ||A Z c - r||`` solve before Krylov starts. The production driver
  exposes it through the opt-in
  ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER`` hook.
- ``sfincs_jax/rhs1_qi_promotion.py``:
  pure promotion gates for QI hard-seed and production-ladder evidence. It
  requires complete seed/backend coverage, convergence, output and trace
  provenance, residual/observable bounds, and no host fallback before a true
  device-QI claim can be promoted.
- ``sfincs_jax/rhs1_preconditioner_dispatch.py``:
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/rhs1_preconditioner_auto_policy.py``:
  RHSMode=1 preconditioner environment alias normalization plus bounded
  automatic preconditioner policy predicates for PAS, DKES, tokamak, GPU sparse
  fallback, weak-default PAS promotion, PAS-family refinement, FP/DKES routing,
  large-FP near-zero-Er overrides, and sharded line-overrides.
- ``sfincs_jax/rhs1_schur_policy.py``:
  RHSMode=1 Schur base-preconditioner alias normalization and automatic
  geometry/PAS/DKES routing policy.
- ``sfincs_jax/rhs1_stage2_policy.py``:
  stage-2 trigger and skip rules.
- ``sfincs_jax/rhs1_strong_policy.py``, ``sfincs_jax/rhs1_strong_control.py``,
  ``sfincs_jax/rhs1_strong_auto_kind.py``:
  strong-preconditioner request mapping, enable/disable control, and automatic
  strong-kind selection.
- ``sfincs_jax/rhs1_sparse_rescue_policy.py`` and
  ``sfincs_jax/rhs1_sparse_polish_policy.py``:
  sparse-rescue ordering, skip logic, and sparse-polish env parsing.
- ``sfincs_jax/rhs1_sparse_exact_policy.py``:
  sparse exact-LU request policy, sparse-over-dense preference, and stage-2
  skip decisions for moderate RHSMode=1 full-FP systems.
- ``sfincs_jax/rhs1_handoff.py``:
  accepted-candidate handoff and Krylov replay-state updates.
- ``sfincs_jax/rhs1_acceptance_policy.py``:
  large-PAS fast-accept gates and host x-block factor-probe safety checks.
- ``sfincs_jax/rhs1_constraint0_policy.py``:
  RHSMode=1 constraint-scheme-0 sparse-first, PETSc-compatible sparse routing, and
  dense-fallback opt-in policy.
- ``sfincs_jax/rhs1_host_policy.py``:
  RHSMode=1 host dense fallback, host sparse-direct, sparse-preconditioned
  GMRES rescue, factor-dtype, and explicit sparse-helper policy.
- ``sfincs_jax/rhs1_large_cpu_policy.py``:
  large explicit full-FP CPU sparse rescue, x-block seed, exact-LU promotion,
  host x-block assembly, and species-x-block rescue policy.
- ``sfincs_jax/rhs1_post_xblock_policy.py``:
  post-x-block polish, targeted FP polish, and skip-global-sparse-after-xblock
  policy for large explicit full-FP CPU systems.
- ``sfincs_jax/solve_mode_policy.py``:
  shared implicit/differentiable solve-mode environment resolution.
- ``sfincs_jax/solver_path_policy.py``:
  pure solver/preconditioner path policy for JIT admission, RHSMode=1 rescue
  slack, DKES GMRES budget defaults, sparse-PC defaults, preconditioner dtype,
  and backend resource-exhaustion classification.
- ``sfincs_jax/solver_selection_policy.py``:
  measured candidate acceptance gates used by automatic solver/preconditioner
  promotions, including residual/parity checks and paired runtime/memory
  comparisons against an incumbent path.
- ``sfincs_jax/transport_policy.py``:
  pure transport backend, sparse-direct, host-GMRES, dtype, and recycle policy.
- ``sfincs_jax/transport_preconditioner_dispatch.py``:
  shared transport preconditioner-kind normalization, auto-selection, DD/sparse-JAX
  env parsing, and reduced/full preconditioner builder dispatch.
- ``sfincs_jax/transport_solve_policy.py``:
  shared active-DOF transport policy, active-index map construction, and dense
  fallback / dense-preconditioner policy used before the transport preconditioner and
  solve handoff layers.
- ``sfincs_jax/transport_handoff_policy.py``:
  shared transport retry residual metrics, better-candidate comparisons, and RHSMode=3
  polish threshold/restart/maxiter policy used by the reduced and full transport solve
  branches.
- ``sfincs_jax/transport_residual_quality.py``:
  fast transport worker residual-abort threshold parsing and failure-message
  formatting for absolute and RHS-normalized diagnostics.
- ``sfincs_jax/transport_dense_lu.py``:
  cached dense-LU solver and preconditioner construction used by bounded transport
  dense fallback and dense-preconditioner paths.
- ``sfincs_jax/transport_host_gmres.py``:
  host SciPy GMRES first-attempt/rescue solve helper for explicit transport paths,
  including PETSc-like preconditioned-residual acceptance for the relevant
  near-singular transport systems.
- ``sfincs_jax/transport_parallel_policy.py``:
  pure transport process-parallel backend selection, worker-count validation,
  benchmark scaling audits, process-pool cache keys, GPU-worker environment
  isolation, and multiprocessing fallback policy.
- ``sfincs_jax/transport_parallel_runtime.py``:
  transport parallel RHS partitioning, GPU worker subprocess launch, and parent-side
  merge of per-worker state/residual/elapsed-time results.
- ``sfincs_jax/transport_parallel_pool.py``:
  persistent transport process-pool caching, rebuild, and shutdown behavior used by the
  CPU process-parallel transport lane.
- ``sfincs_jax/transport_parallel_execution.py``:
  top-level transport process-parallel execution control, including run/no-run gating,
  per-worker payload construction, backend-specific execution, retry, and sequential
  fallback.
- ``sfincs_jax/transport_parallel_sharding.py``:
  pure single-case sharded-solve planning metadata. It caps requested device
  counts, records per-device workload balance, estimates whether setup and
  Krylov communication can be amortized, marks single-case sharding as
  experimental/non-release by default, and prevents malformed sharded payloads
  from becoming release scaling claims.
- ``sfincs_jax/validation_artifacts.py``:
  lightweight loaders and physics metrics for checked-in publication artifacts. This
  module is independent of the heavy solver path, so documentation and CI can verify
  collisionality, high-collisionality trend, trajectory-sweep, and dashboard artifacts
  plus frozen CPU/GPU Fortran-suite benchmark summaries without rerunning large scans
  or example-suite audits.
- ``sfincs_jax/phi1_newton_policy.py``:
  bounded nonlinear/Newton policy for Phi1 solves, including active-DOF mode
  selection, restart sizing, frozen-Jacobian cache policy, and line-search policy.
- ``sfincs_jax/phi1_newton_linear.py``:
  bounded nonlinear linear-step orchestration for Phi1 solves, including reduced/full
  routing, sparse-direct entry, KSP-history emission, and retry-without-preconditioner.
- ``sfincs_jax/phi1_line_search.py``:
  accepted-iterate update logic for the Newton path, including PETSc-like backtracking,
  fixed-candidate ``best`` search, and finite-state fallback handling.
- ``sfincs_jax/solver_progress.py``:
  user-facing duration formatting, coarse runtime hints, one-shot large RHSMode=1
  progress messages, and transport whichRHS ETA text. This module is intentionally
  lightweight so CLI progress can stay informative without importing heavy solver
  dependencies. It is solver-neutral: it improves observability without affecting
  numerical decisions.
- ``sfincs_jax/solver_progress_policy.py``:
  the pure formatting and RHSMode=1 progress-threshold policy re-exported by
  ``solver_progress.py`` so CLI observability decisions remain unit-testable.
- ``sfincs_jax/benchmark_artifact_policy.py``:
  fast schema, provenance, and release-blocking classification policy for checked-in
  benchmark JSON artifacts.
- ``sfincs_jax/memory_model.py``:
  conservative dense/CSR/Krylov/preconditioner memory estimates used by solver
  restart caps, benchmark manifests, and measured solver-candidate gates. This is
  the preflight layer that keeps future memory-saving defaults testable before
  expensive operators or preconditioners are materialized.
- ``sfincs_jax/rhs1_host_policy.py``:
  tested admission gates for RHSMode=1 host dense, sparse-host, constrained-PAS
  sparse-PC, CPU 3D full-FP sparse-PC, and GPU tokamak full-FP no-Er/Er
  sparse-PC auto lanes. These helpers keep solver path promotion rules explicit
  and unit-testable without assembling a kinetic operator.

``sfincs_jax/solver.py`` and ``sfincs_jax/implicit_solve.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear-algebra infrastructure:

- Krylov wrappers,
- host-direct and sparse rescues,
- differentiable linear solves,
- JAX-native linear solve utilities.

``sfincs_jax/transport_matrix.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RHSMode=2/3 postprocessing and transport-matrix assembly.

``sfincs_jax/diagnostics.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Moment integrals, flux-surface-averaged outputs, classical transport diagnostics, and
other quantities that end up in ``sfincsOutput.h5``.

Where the main equations live
-----------------------------

The conceptual mapping is:

- drift-kinetic model:
  :doc:`physics_models`, :doc:`system_equations`, :doc:`physics_reference`
- discretization:
  :doc:`method`, :doc:`numerics`
- geometry coefficients:
  :doc:`geometry`
- solve stack:
  ``sfincs_jax/v3_driver.py`` + ``sfincs_jax/solver.py``
- outputs and diagnostics:
  :doc:`outputs`, ``sfincs_jax/io.py``, ``sfincs_jax/diagnostics.py``

Tests that protect each layer
-----------------------------

The repository intentionally tests the code at several levels:

- unit tests for geometry, collision, and solve heuristics,
- parity/regression tests against frozen reference outputs,
- end-to-end output-writing tests,
- benchmark smoke tests for parallel and performance tooling.

See :doc:`testing` for the validation strategy and the most relevant test files.
