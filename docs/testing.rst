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

Further reading
---------------

For the current benchmark/performance state, see :doc:`performance` and
:doc:`parallelism`. For the external validation story, see :doc:`fortran_comparison`.
For the manuscript-facing literature/figure lanes, see :doc:`validation_matrix`.
