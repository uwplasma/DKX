Development Roadmap
===================

``sfincs_jax`` is a research neoclassical-transport code with production
CLI/API paths, optional differentiable Python paths, CPU/GPU solver paths,
benchmark generation, and explicitly gated research probes for difficult QI
and parallel-scaling cases. The code-health roadmap is to keep those
capabilities in a compact domain-organized package without changing validated
physics behavior.

The authoritative branch checklist lives in the repository root as
``plan_final.md``. This page summarizes stable development principles and does
not define a competing sequence of work.

Goals
-----

- Preserve release correctness: documented CPU/GPU examples remain clean, and
  SFINCS Fortran v3 comparisons remain reproducible.
- Keep public defaults simple: users provide an input file and geometry, then
  ``sfincs_jax`` chooses a safe solver path unless an advanced method is
  explicitly requested.
- Make research paths auditable: opt-in experiments must record solver method,
  residual history, memory, runtime, and failure reason.
- Keep CI practical: default CI should remain in the 5-10 minute range, while
  heavy CPU/GPU validation stays in scheduled or manual benchmark tiers.
- Keep autodiff explicit: differentiable paths should remain JAX-native and
  tested separately from faster non-autodiff production fallbacks.

Solver And Differentiation Contract
-----------------------------------

The refactor keeps two solver lanes because they have different technical
contracts.

1. Production CLI/non-autodiff lane
   This is the default for ``sfincs_jax input.namelist`` and for Python calls
   with ``differentiable=False``. It may use the fastest safe host or device
   path available for the requested model, including sparse factors,
   preconditioner caches, and other non-differentiable setup work. This lane
   must fail closed: accepted results need true-residual checks, phase timings,
   solver metadata, and output diagnostics.

2. Differentiable Python/API lane
   This lane is selected explicitly with ``differentiable=True``. It should use
   JAX-native operators and implicit differentiation for linear solves. The
   central contract is the same one used by ``jax.lax.custom_linear_solve``:
   gradients are defined by the linear equation and transpose/adjoint solve at
   the converged solution, not by differentiating through every Krylov or setup
   iteration. This keeps memory bounded and matches the adjoint strategy used
   in modern spectral-solver differentiation work.

3. Adaptive branch decisions
   The ``auto`` policy is a hard discrete decision and is not itself a smooth
   differentiable map. Gradients are valid through the selected accepted branch,
   away from branch boundaries. The code should therefore record a branch
   certificate: selected method, rejected candidates, predicates, residual
   margins, capability metadata, and warnings when the solve is near a branch
   boundary. Smooth or relaxed branch selectors should only be added as
   deliberate surrogate objectives, not as hidden replacements for production
   ``auto``.

4. Optional JAX ecosystem libraries
   ``lineax``, ``jaxopt``, ``equinox``, and ``optax`` remain measured optional
   lanes unless they clearly improve accuracy, runtime, memory, or code
   clarity on checked fixtures. ``lineax`` is most relevant to operator-based
   solves, ``jaxopt`` to implicit root/fixed-point examples, ``equinox`` to
   PyTree state boundaries, and ``optax`` to optimization workflows rather than
   core transport solves.

Refactoring Plan
----------------

The main code-health target is a flat package: implementation ownership lives
in single-purpose root modules directly under ``sfincs_jax/``, with only two
package folders (``validation`` and ``workflows``) for tooling that sits
outside the compute path. The legacy nested packages (``problems``,
``solvers``, ``operators``, ``outputs``, ``geometry``, ``discretization``,
``physics``) were deleted with the legacy pipeline; refactor work should remain
behavior-preserving, tested in gated slices, and should not reintroduce nested
implementation packages or historical helper names such as top-level
``rhs1_*`` files.

1. Package structure and import contract
   Every physics, discretization, and I/O owner is a flat root module
   (``drift_kinetic.py``, ``collisions.py``, ``magnetic_geometry.py``,
   ``phase_space.py``, ``xgrid.py``, ``moments.py``, ``writer.py``,
   ``inputs.py``, ``run.py``, ``solve.py``, ``er.py``, ``phi1.py``, ...; see
   the table in ``sfincs_jax/README.md``). ``validation/`` holds parity,
   release-gate, and data-fetch tooling; ``workflows/`` holds optimization and
   scan orchestration. Source-tree tests
   (``tests/fixtures/source_tree_expected.json``) pin this layout and the
   absence of nested implementation packages.

2. Problem owners
   RHSMode=1 profile/bootstrap-current orchestration, RHSMode=2/3
   transport-matrix and monoenergetic-response orchestration, and the
   namelist-level dispatch all belong to ``run.py`` (``run_profile``,
   ``run_transport_matrix``, ``run_from_namelist``). Do not reintroduce
   separate compatibility facades for these owners; user and contributor
   imports should point at ``run.py`` directly.

3. Physics and discretization owners
   Model terms and residual construction live in ``drift_kinetic.py`` (the
   matrix-free ``KineticOperator``), collision models in ``collisions.py``,
   grids and quadrature in ``phase_space.py``/``xgrid.py``, and geometry in
   ``magnetic_geometry.py``. Each owner should have shape, masking, zero-drive
   or constant-field, order-condition, conservation, or parity tests where
   those properties apply.

4. Solver and preconditioner architecture
   The three-tier solver policy (structured block elimination, preconditioned
   recycled Krylov, host direct referee), the coarse and bordered-Schur
   preconditioners, residual gates, recycling, and the
   implicit-differentiation contract all live in ``solve.py`` on top of the
   ``solvax`` library; reusable numerical kernels (block-Thomas, GCROT,
   implicit solves) belong upstream in ``solvax``, not in new local modules.
   Auto-selection remains the user-facing default, and solver decisions are
   reported through the versioned trace schema in ``solver_trace.py``.

5. I/O, workflow, validation, and benchmark boundaries
   HDF5/NPZ/NetCDF writes stay outside solver internals. Solver code returns
   structured diagnostics; output modules serialize those diagnostics without
   reinterpreting convergence. Benchmark JSON schemas are shared by README
   plots, documentation figures, and research-lane manifests so runtime,
   memory, residuals, backend, solver path, and comparison status use one
   contract.

Testing Plan
------------

Tests should be organized by cost and purpose rather than by historical file
location.

1. Unit tests
   Cover pure functions, grid transforms, geometry interpolation, operator
   shape/linearity, output schema, parser behavior, and solver-policy decisions.
   These must be fast and deterministic.

2. Numerical tests
   Exercise linearity, adjoint consistency where applicable, residual monotonic
   gates, fail-closed preconditioner behavior, active-DOF projection, and JAX
   JIT/vmap/grad compatibility.

3. Physics tests
   Maintain literature-anchored gates for collisionality trends, ambipolar
   roots, monoenergetic coefficient behavior, bootstrap-current signs/scales,
   and known symmetry limits. These should use small but meaningful fixtures.

4. Regression tests
   Keep a small public example set in normal CI and a larger release benchmark
   set outside normal CI. Regression artifacts should include solution
   tolerances, runtime windows, memory windows, and solver-path expectations.

5. GPU tests
   Keep tiny GPU smoke tests in optional CI where available. Larger GPU tests
   should be manual/scheduled and must write compact artifacts rather than raw
   profiler dumps.

Coverage target
---------------

The long-term target remains high meaningful coverage, but it should be reached
by refactoring and focused module tests, not by adding slow full-solve tests.
The practical sequence is:

- bring extracted policy/output/metadata modules to at least ``90%`` coverage,
- bring geometry/grid/normalization modules to at least ``90%`` coverage,
- cover solver primitives with synthetic operators and fixture-sized matrices,
- leave production full solves as regression/benchmark gates rather than unit
  coverage drivers.

Validation Plan
---------------

Validation should be layered and repeatable.

1. Release parity
   Continue comparing documented examples against SFINCS Fortran v3 artifacts
   where the model overlap is exact. Every comparison should record all output
   quantities that both codes expose, not only a short parity table.

2. Literature benchmarks
   Maintain publication-ready figures for high-collisionality limits,
   electric-field scans, monoenergetic coefficients, and bootstrap-current
   trends. Each figure should list the equation/model assumptions and the input
   fixture that generated it.

3. Cross-code comparisons
   Keep cross-code comparisons scoped by normalization and model contract. If
   profiles, equilibria, or collision models differ, the docs should state that
   explicitly instead of claiming direct agreement.

4. Autodiff validation
   Add finite-difference and complex-step style checks where possible for
   differentiable observables. For optimization examples, validate gradients on
   reduced fixtures before using them in larger design loops.

Benchmarking Plan
-----------------

Benchmarks should answer separate questions with separate tiers.

1. CI smoke tier
   Runtime target: seconds. Purpose: catch broken CLI/API paths, output schema
   regressions, and basic solver-policy mistakes.

2. Release tier
   Runtime target: minutes locally, not normal CI. Purpose: regenerate README
   plots and tables for SFINCS Fortran v3, cold/warm CPU, and cold/warm GPU.
   Include only cases whose Fortran reference runtime is large enough for a
   meaningful comparison.

3. Research tier
   Runtime target: bounded by explicit timeouts. Purpose: QI hard seeds,
   production-resolution PAS/geometry-rich offenders, high-nu campaigns, and
   scaling tests. Failed attempts are useful only if they are fail-closed and
   write compact evidence artifacts.

4. Profiling tier
   Runtime target: manual. Purpose: Perfetto/XLA/device-memory traces for one
   selected offender at a time. Large raw traces should stay off-repo; checked
   summaries should include phase timings, peak memory, kernel bottlenecks, and
   next code action.

Acceptance Gates
----------------

A change should not be promoted into public default behavior unless it passes
all relevant gates:

- correctness: no regression in shared SFINCS Fortran v3 outputs or checked
  physics gates,
- convergence: accepted solver paths satisfy true-residual gates, not only
  internal Krylov estimates,
- performance: runtime does not regress by more than the documented tolerance
  on representative CPU/GPU fixtures,
- memory: peak memory does not regress on known memory-sensitive cases unless
  the tradeoff is explicitly documented,
- diagnostics: CLI and Python users can see enough phase timing and progress to
  estimate whether a run will take seconds, minutes, or longer,
- documentation: new methods include equations, controls, failure modes, and
  reproducible benchmark commands.

QI Scope Boundary
-----------------

The documented scale-0.60 QI hard seed reaches residuals below ``3e-5`` on both
CPU and GPU with the post-residual-equation research path. This bounded result
confirms device safety for that checked path, but it is not a production
convergence claim. Promoting the lane requires residual-equation/coarse
variables built from the remaining Krylov error modes and from
current/constraint/profile moments.
