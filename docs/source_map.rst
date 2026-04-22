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
  PAS applicability, PAS-TZ memory safety, and PAS fallback routing.
- ``sfincs_jax/rhs1_preconditioner_dispatch.py``:
  shared RHSMode=1 preconditioner-kind dispatch.
- ``sfincs_jax/rhs1_stage2_policy.py``:
  stage-2 trigger and skip rules.
- ``sfincs_jax/rhs1_strong_policy.py``, ``sfincs_jax/rhs1_strong_control.py``,
  ``sfincs_jax/rhs1_strong_auto_kind.py``:
  strong-preconditioner request mapping, enable/disable control, and automatic
  strong-kind selection.
- ``sfincs_jax/rhs1_sparse_rescue_policy.py`` and
  ``sfincs_jax/rhs1_sparse_polish_policy.py``:
  sparse-rescue ordering, skip logic, and sparse-polish env parsing.
- ``sfincs_jax/rhs1_handoff.py``:
  accepted-candidate handoff and Krylov replay-state updates.
- ``sfincs_jax/transport_policy.py``:
  pure transport backend, sparse-direct, host-GMRES, dtype, and recycle policy.
- ``sfincs_jax/transport_parallel_policy.py``:
  transport process-parallel backend selection, worker env, GPU worker env, and pool
  policy.
- ``sfincs_jax/transport_parallel_runtime.py``:
  transport parallel RHS partitioning, GPU worker subprocess launch, and parent-side
  merge of per-worker state/residual/elapsed-time results.
- ``sfincs_jax/transport_parallel_pool.py``:
  persistent transport process-pool caching, rebuild, and shutdown behavior used by the
  CPU process-parallel transport lane.

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
