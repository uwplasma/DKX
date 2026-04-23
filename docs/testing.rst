Testing, validation, and CI
===========================

`sfincs_jax` is validated with a layered testing strategy. The code is not trusted
because any single benchmark happens to pass; it is trusted because the operator,
solvers, output writer, and public workflows are all exercised at multiple levels.

Validation philosophy
---------------------

The validation stack is organized from local to global:

1. **Unit tests** for grids, geometry, collisions, solver heuristics, and CLI behavior.
2. **Regression tests** for previously broken branches and edge cases.
3. **Output tests** for ``sfincsOutput.h5`` generation and dataset integrity.
4. **Example-suite audits** that compare full case outputs against frozen reference
   artifacts.
5. **Benchmark smoke tests** for transport parallelism and scaling scripts.

This layered approach reduces the risk of shipping a numerically correct but operationally
fragile code, or a fast code that quietly changed the physics.

What is compared
----------------

The main release-facing checks compare:

- scalar diagnostics,
- arrays in ``sfincsOutput.h5``,
- transport matrices,
- selected terminal signals,
- and, where appropriate, strict tolerances on all datasets in the audited suite.

Comparisons against the mature Fortran implementation are used as a validation anchor,
not as the public identity of the code. The purpose of those checks is to show that the
standalone `sfincs_jax` implementation reproduces trusted neoclassical physics on the
supported audited scope.

Test categories
---------------

Unit and regression tests
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``tests/`` tree includes:

- physics-term tests (streaming, drifts, collisions),
- geometry/output tests for each supported geometry family,
- solve-path heuristic tests,
- CLI and input-override tests,
- parallel benchmark smoke tests,
- and output-writing end-to-end tests.

Representative examples:

- ``tests/test_output_h5_scheme5_parity.py``
- ``tests/test_transport_parallel.py``
- ``tests/test_cli_solve_mode.py``
- ``tests/test_full_system_gmres_solution_parity.py``
- ``tests/test_rhs1_schwarz_heuristic.py``

Full suite and release checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The release-facing example-suite artifacts on ``main`` are generated from the full
39-case CPU and GPU audits recorded in the repository. Those audits are summarized in
the README and the performance/validation pages.

For current release documentation, the important point is simple:

- all cases in the audited suite complete on CPU and GPU,
- no ``jax_error`` or ``max_attempts`` entries remain in the release artifacts,
- and the frozen-reference comparisons are clean on the documented scope.

Continuous integration
----------------------

The repository is kept buildable and testable through standard CI-style commands:

.. code-block:: bash

   pytest -q
   sphinx-build -W -b html docs docs/_build/html

The same checks are also represented in the repository CI/CD configuration:

- ``.github/workflows/ci.yml`` runs the test matrix and example smoke tests,
- the same CI workflow also runs the audited coverage job and uploads ``coverage.xml``
  through Codecov using GitHub OIDC,
- ``.github/workflows/docs.yml`` builds the Sphinx documentation,
- ``.github/workflows/publish-pypi.yml`` handles packaging/release publication.

The current audited full-suite command on ``main`` is:

.. code-block:: bash

   pytest -q --cov=sfincs_jax --cov-report=term --cov-report=xml

On the current audited local release tree this command yields ``579 tests collected``,
``579 passed`` in the stable chunked rerun, and
roughly ``55%`` package coverage. That number is materially higher than the Linux
CI runner floor, but it also makes the remaining gap explicit: the dominant uncovered
surface is still the large solver/geometry stack, especially ``v3_driver.py``,
``io.py``, ``geometry.py``, ``grids.py``, and ``vmec_geometry.py``. The latest
low-cost campaign improved the analytic geometry/grid surface materially
(``geometry.py`` to about ``88%``, ``grids.py`` to about ``82%``, and
``vmec_geometry.py`` to about ``97%``) and then added direct coverage for the
operational cache/policy seams in ``io.py`` and ``v3_driver.py`` plus bounded
HDF5/export and distributed-Krylov branches in ``io.py`` and ``solver.py``.
Those later heavy-module tests raised ``io.py`` from about ``65%`` to ``67%`` and
``solver.py`` from about ``57%`` to ``67%`` without opening another long solver-wide
campaign. The stencil-scheme campaign then raised ``grids.py`` from about
``46%`` to ``79%`` by exercising the unused finite-difference branches directly.
The latest literature-anchored numeric pass then pushed ``grids.py`` further to
about ``82%`` by checking the exact polynomial order conditions of the 3-point and
5-point SFINCS finite-difference formulas and by covering the remaining one-sided
five-point guard branches. In parallel, the bounded sparse-helper campaign covered
the explicit sparse-factor builder in ``v3_driver.py``, including environment parsing,
matrix-free operator assembly hooks, and host sparse factorization handoff on tiny
synthetic operators. These tests were chosen from the same identities used in the
SFINCS technical documentation and the 2014 SFINCS paper: periodic/spectral
differentiation exactness, finite-difference order conditions, Boozer-coordinate
field-component relations, VMEC half-mesh finite-difference conventions, and
deterministic cache / solver-policy behavior on bounded inputs. Reaching a
research-grade coverage target therefore still requires more focused tests on the
heavy solver modules rather than more trivial helper tests. The latest applied-math
pass then targeted the sparse/circulant derivative layer and the PAS residual-gate
metrics directly. That batch pushed ``periodic_stencil.py`` from about ``57%`` to
about ``67%`` by checking circulant/Fourier-mode exactness, sparse-row extraction
bounds, and the documented sharded-halo fallback behavior; it also pushed
``pas_smoother.py`` from about ``59%`` to about ``62%`` by checking
target/upward/plateau gate logic and bounded stationary-smoother convergence on
tiny analytic systems. These tests are anchored in standard numerical-analysis
invariants: Fourier modes as eigenvectors of circulant derivative operators, and
residual-history stopping rules consistent with minimal-residual / stagnation
monitoring in iterative methods. The latest diagnostics/output-reduction pass then
targeted the ``uHat`` assembly seam directly. Those tests pushed
``diagnostics.py`` to ``100%`` by checking FFT-vs-NumPy agreement on a frozen
scheme-4 fixture, differentiability with respect to Boozer harmonics, finite and
shape-correct loop behavior on even and odd periodic grids, resonant-denominator
safety in the explicit harmonic-loop reference implementation, and spatial
constancy in the constant-``B`` limit. That pass also found and fixed a real bug:
the resonant branch in ``_u_hat_loop()`` could still trigger a Python-side
division-by-zero before the masked ``jnp.where()`` path executed, so the loop now
guards the denominator explicitly instead of relying on masked evaluation.

After the diagnostics pass, the next bounded driver-side campaign targeted the
domain-decomposition and residual-correction layer directly. Those tests check
diagonal and block-diagonal reductions, overlapping patch-range construction,
coarse-level sizing and environment overrides, multilevel residual-correction
composition, safe-preconditioner clipping of nonfinite values, and finite-state
GMRES-result gating on tiny synthetic systems. These checks are anchored in
standard additive-Schwarz / block-Jacobi invariants and bounded multilevel
residual-correction ideas: local blocks must preserve only local couplings,
patches must cover the full discrete domain with controlled overlap, and
coarse corrections must apply in a deterministic order without creating
nonfinite iterates. The main measured result of that pass was not a dramatic
package-percentage jump but a tighter, more meaningful test net around the
``v3_driver.py`` decision layer while keeping the full tree at ``552/552``
green and the package coverage at roughly ``54%``. The latest bounded
solve-policy pass then moved the full tree to ``568/568`` and pushed package
coverage to about ``55%`` by exercising more of the driver’s actual control
logic: constraint-scheme routing, sparse-exact-LU selection, x-block/sxblock
rescue eligibility, transport sparse-direct and host-GMRES guard rails, host-only
SciPy Krylov dispatch, and distributed-incompatible GMRES rejection. These tests
do not attempt to prove convergence of every large solve; they prove that the
policy ladder surrounding those solves takes the right branch on bounded,
physically meaningful synthetic inputs. The latest bounded ``io.py`` pass then
extended that same strategy to the output side: cache-directory selection,
HDF5 read/write guards, Fortran-layout serialization, export-``f`` configuration
and mapping, and small output-policy helpers such as Newton-step selection and
scalar/list parsing. That moved the audited tree to ``579/579`` while nudging
``io.py`` from about ``66.6%`` to about ``66.8%``. The package percentage moved
only slightly because the remaining denominator is still dominated by the deep
solver body in ``v3_driver.py``, but the added tests cover genuine user-facing
behavior rather than synthetic dead branches. The latest bounded driver pass then
targeted the PAS tokamak / PAS-TZ preconditioner applicability ladder directly.
Those tests check zeta-invariant tokamak detection, rejection of zeta-varying or
drift-rich tokamak branches, PAS-only vs FP-only routing for the 3D PAS-TZ
preconditioner, invalid environment fallback for the PAS-TZ memory cap, build-byte
estimation, memory-safety gating, and the fallback to lighter hybrid or block
preconditioners when the heavier PAS builders are inapplicable or unsafe. The
follow-up pass then extended that same slice to the sharded memory-unsafe fallback
handoff itself, including the ``theta`` and ``zeta`` Schwarz branches and invalid
domain-decomposition environment parsing. That follow-up also fixed a real bug:
the PAS-TZ memory-unsafe sharded path always routed into
``_build_rhsmode1_theta_schwarz_preconditioner()`` even when the active shard axis
was ``zeta``. It now dispatches to the axis-correct Schwarz builder. Together these
passes moved the audited tree to ``590 tests collected`` and ``590 passed`` while
holding total package coverage at about ``55%``. That result is still useful because
it tightens the remaining driver decision surface without opening a new expensive
solve campaign, and it confirms again that the dominant denominator is the deep
execution body of ``v3_driver.py`` rather than the outer policy layer.
The latest follow-up then factored the RHSMode=1 preconditioner dispatch used by
the reduced and full solve paths into a single helper and added bounded tests on
that shared dispatch layer. Those tests cover DD-vs-Schwarz routing, the
``point_xdiag`` and ``xblock_tz_lmax`` branches, composition of the
``theta_line_xdiag`` collision wrapper, and the default block-preconditioner
fallback. This matters because it closed a real consistency gap: the reduced path
already supported ``point_xdiag`` and ``xblock_tz_lmax``, while the full-path copy
of the dispatch ladder had drifted away from it. After this refactor the audited
tree moved to ``596 tests collected`` and ``596 passed``, package coverage stayed
at about ``55%``, and ``v3_driver.py`` itself moved from about ``37%`` to about
``38%``.

The current refactor branch extends that strategy in two directions. First, it splits
RHSMode=1 preconditioner policy into directly tested helper modules, covering alias
canonicalization, PAS weak-default promotion, PAS-family refinement, FP/DKES routing,
GPU sparse fallback skipping, and sharded-line override safety without constructing
large operators. Second, it adds an explicit PAS physics gate: the pitch-angle
scattering collision operator must annihilate the isotropic ``L=0`` Legendre mode,
mask inactive Legendre slots, and scale active higher modes as ``L(L+1)/2`` when the
Krook term is zero. That gate is cheap enough for CI, but it is a real physics
invariant rather than coverage padding. A companion collision-kernel gate checks the
Chandrasekhar function small-``x`` limit, Coulomb deflection-frequency scaling with
density and charge, the identity and polynomial-exactness properties of the v3
barycentric interpolation matrix, and agreement between the analytic and quadrature
Rosenbluth-potential assembly paths on a bounded grid. That pass found and fixed a
real numerical issue:
the direct Chandrasekhar formula was used down to ``x≈1e-14``, which is below the
range where cancellation is safe in double precision. The JAX and NumPy paths now use
the analytic small-``x`` series for ``|x| < 1e-5``.

The geometry-integration gate now also includes an optional ``vmec_jax`` fixture
test. When ``vmec_jax`` and its example data are importable, the test reads a real
``vmec_jax.wout.WoutData`` object, converts it through
``vmec_wout_from_wout_like(...)``, compares all VMEC Fourier coefficient arrays
against the file reader, and verifies that ``vmec_geometry_from_wout(...)`` returns
the same scheme-5 geometry arrays. In normal CI environments where the optional
backend is not installed, the test skips rather than adding a hard dependency. The
same adapter file also has strict structural tests for mode/radius transposition,
metadata-only path overrides, required-table failures, optional zero-filled
magnetic-field coefficient tables, and invalid shapes, so lightweight CI still
protects the public adapter contract.

The differentiability gate starts with a cheaper analytic geometry check before
attempting heavier end-to-end optimization examples.  ``tests/test_geometry_autodiff_gates.py``
uses the scheme-4 harmonic-amplitude hook to form a scalar from
``mean(BHat**2)`` and ``mean(DHat)`` and compares ``jax.grad`` against central finite
differences.  This keeps CI fast while protecting the JAX-native geometry path from
silent regressions in array layout, dtype handling, or non-differentiable branches.

The refactor branch also treats documentation discoverability as testable behavior.
``tests/test_policy_module_docstrings.py`` imports the split RHSMode=1 and transport
policy modules and checks that their explanatory module docstrings are real
``__doc__`` strings rather than inert comments. This is intentionally small, but it
keeps the source map and generated API documentation useful as the large driver is
split into manageable pieces.

The latest driver split also extracts RHSMode=1 host dense/sparse-direct policy into
``sfincs_jax/rhs1_host_policy.py``. ``tests/test_rhs1_host_policy.py`` covers the
backend/env rules for host dense fallback, small accelerator dense shortcuts,
host sparse-direct enablement, sparse-preconditioned GMRES rescue, sparse factor
dtype selection, iterative-refinement step parsing, explicit sparse-helper
bounds, and the FP/PAS dense-fallback active-size ceiling.  The public driver
wrappers remain tested separately, so this is a
behavior-preserving refactor with a smaller directly testable policy surface.

The adjacent constraint-scheme-0 sparse-first policy now lives in
``sfincs_jax/rhs1_constraint0_policy.py``. ``tests/test_rhs1_constraint0_policy.py``
checks that the accelerator-default sparse-first lane, explicit
PETSc-compatible sparse mode, and dense-fallback opt-in preserve the same RHSMode,
``Phi1``, full-FP, solve-method, preconditioner, and size guards as the driver
wrappers in ``tests/test_rhs1_sparse_first_heuristic.py``.

The sparse exact-LU and sparse-over-dense preference decisions now live in
``sfincs_jax/rhs1_sparse_exact_policy.py``. ``tests/test_rhs1_sparse_exact_policy.py``
checks full-x CPU exact-LU routing, accelerator DKES and small-FP exact-LU
routing, PAS-only full-preconditioner opt-in, explicit enable/disable behavior,
size caps, dense-method rejection, moderate-FP sparse preference, and the
stage-2 skip guard. Existing driver-wrapper tests keep the `v3_driver` seam
stable for downstream users.

The large-CPU full-FP rescue ladder now has its own direct policy tests in
``tests/test_rhs1_large_cpu_policy.py``. Those tests cover global sparse rescue,
large-CPU exact-LU caps, sparse-rescue-first ordering, x-block exact-LU
promotion after a good seed, x-block sparse rescue, host x-block assembly,
primary-solve skipping, and the explicit multispecies species-x-block rescue
opt-in. This keeps the CPU runtime-offender routing testable without running a
large solve in CI.

The follow-up post-x-block policy split is covered by
``tests/test_rhs1_post_xblock_policy.py``. These tests check the residual and
active-size gates for fast post-x-block polish, targeted FP polish, and explicit
skip-global-sparse-after-xblock routing after a good x-block seed. The tests keep
large-case convergence handoff decisions visible while avoiding heavyweight CI
runs.

Small acceptance/probe gates are covered directly by
``tests/test_rhs1_acceptance_policy.py``. That file checks the large-PAS
fast-accept environment parsing and backend/implicit/Phi1/PAS guards, plus the
host x-block factor probe for exceptions, shape mismatches, nonfinite solves, and
excessive amplification. The PAS residual formula itself is shared with
``sfincs_jax/pas_smoother.py`` so the acceptance threshold is not duplicated.

The PAS adaptive-smoother gate and implicit-solve mode resolver also have direct
coverage in ``tests/test_rhs1_pas_policy.py`` and ``tests/test_solve_mode_policy.py``.
Those tests keep the PAS smoother activation threshold and
``SFINCS_JAX_IMPLICIT_SOLVE`` / differentiability precedence rules explicit while
the driver wrappers remain stable for compatibility tests.

The VMEC convention layer has its own bounded gate in
``tests/test_vmec_wout_conventions.py``. It checks the scheme-5 conventions that are
easy to break during refactors: ``psi_a_hat = phi[-1]/(2*pi)``, full- and half-mesh
radial interpolation weights, radial-option snapping, endpoint half-mesh
extrapolation, invalid radius/option rejection, and helicity-based
``rippleScale`` selection. The same gate also writes a tiny synthetic NetCDF
``wout`` file and checks the reader contract directly: ASCII path resolution to a
neighboring ``.nc`` file, radius/mode transposition, required-variable failures,
unsupported ``lasym=true`` rejection, and invalid first Fourier-mode metadata.
These tests protect the VMEC geometry path without loading a large equilibrium or
running a transport solve.

The documentation build is part of the release discipline, not a separate afterthought.
If a docs change breaks Sphinx or leaves pages internally inconsistent, it should be
treated as a real regression.

How to work safely
------------------

When changing physics, numerics, or performance-sensitive logic:

1. add or update a focused unit/regression test,
2. run the targeted tests for the touched functionality,
3. run the docs build if the public behavior changed,
4. rerun a representative case or benchmark if performance-sensitive code changed,
5. rerun broader validation before release.

For performance audits, keep benchmark instrumentation bounded: the suite runners
default to ``--jax-profile-marks off`` so runtime drift is measured without
profiler interference. Only opt into ``on`` or ``full`` when the goal is an
explicit profiling lane, and leave per-mark device-memory sampling off unless
you are doing targeted device-memory diagnosis. Kernel/XLA traces should go
through ``scripts/profile_write_output_trace.py`` or the transport-trace helpers,
not through always-on per-phase GPU memory polling. The runtime-drift audit also
prefers the solver's logged ``elapsed_s=...`` value when available, falling back
to subprocess wall time only for older artifacts that do not record it. The suite
subprocesses also pin ``SFINCS_JAX_PRECOMPILE=0`` unless explicitly overridden, and
they leave ``JAX_COMPILATION_CACHE_DIR`` unset unless ``--jax-cache-dir`` is requested,
so runtime drift is not polluted by eager precompile or persistent-cache write cost.

Research reproducibility
------------------------

The repository includes:

- frozen fixtures in ``tests/ref``,
- example inputs in ``examples/sfincs_examples`` and ``examples/upstream``,
- benchmark scripts in ``examples/performance``,
- and generated figures in ``docs/_static/figures``.

That structure is intended to make claims in the docs reproducible. If a figure or
table appears in the docs, there should be a script or artifact trail that explains how
it was produced.

Publication-facing validation lanes
-----------------------------------

The manuscript-oriented validation lanes are tracked separately from the general
unit/regression suite:

- ``examples/publication_figures/validation_manifest.json`` is the machine-readable
  map from literature claim to script and artifact.
- :doc:`validation_matrix` is the corresponding human-facing documentation page.
- ``tests/test_validation_manifest_schema.py`` enforces that every lane has explicit
  source-code anchors, protecting tests, and acceptance gates.

The first new lane on the refactor branch is the ``E_r`` trajectory-model sweep family:

- script: ``examples/publication_figures/generate_er_trajectory_sweep.py``
- pinned tokamak-like reference artifact:
  ``examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json``
- pinned tokamak-like reference figure:
  ``docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png``
- pinned stellarator-like fast artifact:
  ``examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json``
- pinned stellarator-like fast figure:
  ``docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png``

This lane is now used as:

- a branch-level regression target for the trajectory-model sweep script,
- a fixed tokamak-like reference lane with direct numerical assertions on zero-field
  agreement and finite-field model separation,
- a bounded stellarator-like branch lane that keeps the fixed input and figure stable
  while the full-resolution stellarator sweep remains a heavier validation target,
- and a figure/layout prototype for the eventual manuscript figure family.

Another important branch outcome is that the collisionality-figure generator
(``generate_sfincs_paper_figs.py``) is no longer trusted just because the old figure
files exist. On this branch, two real bugs were found in its scan-input writer:

- duplicate ``collisionOperator`` assignments could leave the later original value in
  force, collapsing FP and PAS scans onto the same physics,
- and the fast-path override could place missing keys such as ``NL`` outside the
  namelist group, while also choosing ``Nzeta=3``, which is below the current stencil
  floor for these runs.

Those writer bugs are now unit-tested and fixed. The corrected bounded fast reruns are
kept as branch-level regression scaffolds, and the full LHD/W7-X collisionality
summaries and figures have been regenerated from the fixed script and promoted as
audited validation artifacts.

The audited full artifacts from that lane are now checked in:

- summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_summary.json``
- figure:
  ``docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png``
- summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_summary.json``
- figure:
  ``docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality.png``

The bounded fast artifacts remain checked in for branch-level regression work:

- summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json``
- figure:
  ``docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png``
- summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json``
- figure:
  ``docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png``

The full artifacts are guarded by direct tests on the seven-point collisionality ladder
and FP/PAS label coverage. The corrected fast artifacts are guarded by lighter direct
tests on the four-point ladder and FP/PAS separation.

The collisionality generator writes structured JSON summaries for all reruns, with
top-level metadata that records the case, fast/full-resolution mode, scan ladder,
source input, and collision-operator labeling. That keeps future release reruns aligned
with the manifest's provenance and acceptance-gate expectations instead of relying on
plots alone.

The next planned literature lane now has an executable scaffold as well:

- script: ``examples/publication_figures/generate_w7x_ambipolar_validation.py``
- focused test: ``tests/test_generate_w7x_ambipolar_validation.py``

This keeps the W7-X ambipolar validation work out of the "purely aspirational" bucket:
the scan, ambipolar postprocessing, summary JSON, and figure generation paths are now
covered by a bounded end-to-end test on a tiny fixture, even though the heavy W7-X
reference artifact is still an explicit open lane in the manifest.

The same scaffold is now resumable for heavy runs: ``run_er_scan`` accepts
``skip_existing=True``, the ``sfincs_jax scan-er`` CLI exposes ``--skip-existing``,
and the publication script adds ``--skip-existing``, ``--scan-only``, and
``--index/--stride`` so the heavy W7-X reference ladder can be filled across multiple
devices before a final aggregation pass.

Further reading
---------------

For the current benchmark/performance state, see :doc:`performance` and
:doc:`parallelism`. For the external validation story, see :doc:`fortran_comparison`.
For the manuscript-facing literature/figure lanes, see :doc:`validation_matrix`.

Optional ecosystem gates
------------------------

External JAX ecosystem libraries are evaluated through benchmark gates before they are
allowed into production code. The current bounded examples are:

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py --backend all --suite all
   python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py --backend all

The Lineax gate always benchmarks the in-tree implicit solve and only runs the Lineax branch
when ``lineax`` is installed. The associated test
``tests/test_optional_lineax_implicit_gate.py`` verifies the deterministic
nonsymmetric stress system, the tiny real scheme-5 SFINCS implicit-diff lane, the
repeated-RHS reuse lane on that same tiny operator, and clean skip behavior when Lineax
is absent.

The measured conclusion from the current local Lineax run is intentionally conservative:
the synthetic system is faster and residual-clean with ``lineax``, but the tiny real
matrix-free SFINCS operator still returns Lineax failure statuses despite tiny residuals,
so the in-tree implicit solve remains the only admissible production path.

The Equinox/JAXopt gate is a separate objective-wrapper check on a real differentiable
``geometryScheme=4`` harmonic-fit problem. Its associated test
``tests/test_optional_eqx_jaxopt_scheme4_gate.py`` verifies deterministic problem
construction, directional-derivative agreement for an ``equinox.Module`` wrapper,
bounded loss reduction and parameter recovery for ``jaxopt.GradientDescent``, JSON
output, and clean skip behavior when either optional package is absent.
