Development Roadmap
===================

``sfincs_jax`` has grown from a parity implementation into a larger research
codebase with production CLI/API paths, optional differentiable paths, GPU
solver paths, benchmark generation, and research-only QI/parallelism probes.
The next phase should improve maintainability without changing the validated
physics behavior. This page is the working plan for that phase.

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

Refactoring Plan
----------------

The main code-health target is to reduce the responsibility of
``sfincs_jax/v3_driver.py``. The refactor should be behavior-preserving and
land in small, gated slices.

1. Operator and state extraction
   Move RHSMode=1 operator construction, active-DOF projection, residual
   evaluation, and solver-state metadata into focused modules. Target modules:
   ``rhs1_operator.py``, ``rhs1_state.py``, and ``rhs1_metadata.py``. The
   active-DOF routing/index-map portion has started in
   ``rhs1_active_dof.py``.

2. Solver-policy extraction
   Move auto-selection heuristics, environment parsing, fallback policy, and
   promotion gates into a dedicated policy module. The policy layer should
   return a typed decision object rather than mutating driver-local flags.
   The first landed slice is ``sfincs_jax/rhs1_solver_policy.py``, which owns
   typed parsing for x-block probe-coarse, post-minres, post-coarse, and
   post-residual-equation controls.
   The second landed slice is ``sfincs_jax/rhs1_solver_diagnostics.py``, which
   owns x-block correction diagnostic records and output-visible metadata key
   assembly.
   The third landed slice is ``sfincs_jax/rhs1_active_dof.py``, which owns
   active-DOF routing and index-map construction for RHSMode=1 truncated pitch
   grids and PAS constraint-projection solves.
   The fourth landed slice is ``sfincs_jax/rhs1_active_projection.py``, which
   owns reusable JAX full/reduced vector gathers, one-based scatter expansion,
   and PAS constraint projection primitives used by sparse-PC and x-block
   active-DOF residual paths.
   The fifth landed slice is ``sfincs_jax/rhs1_residual.py``, which owns small
   residual norm/target/ratio helpers used by sparse-PC and x-block solver
   metadata.

3. Preconditioner registry
   Keep x-block, PAS-lite, Schur, QI, and post-residual-equation preconditioners
   behind a registry with explicit capability metadata: CPU/GPU safe,
   differentiable/non-differentiable, setup memory estimate, and expected
   operator shape.

4. Output and diagnostics boundary
   Keep HDF5/NPZ/NetCDF writes outside solver internals. Solver code should
   return structured diagnostics; output modules should serialize those
   diagnostics without reinterpreting convergence. The first concrete step is
   the RHSMode=1 x-block correction metadata extraction, which preserves the
   historical field names while making schema regressions unit-testable.

5. Benchmark and evidence schema
   Consolidate benchmark JSON schemas so README plots, documentation figures,
   and research-lane manifests consume the same fields for runtime, memory,
   residuals, backend, solver path, and comparison status.

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

Current QI Scope Boundary
-------------------------

The scale-0.60 QI hard seed is now below ``3e-5`` residual on both CPU and GPU
with the post-residual-equation research path. This is a useful bounded result
and confirms that the implementation is device-safe for the checked path. It is
not a production convergence claim. Closing that lane requires new
residual-equation/coarse variables built from the remaining Krylov error modes
and current/constraint/profile moments.
