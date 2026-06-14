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

Literature-artifact gates
^^^^^^^^^^^^^^^^^^^^^^^^^

Publication-facing validations use the same layered idea, but the unit under test is
the checked-in scientific artifact rather than a single function. The key files are:

- ``examples/publication_figures/validation_manifest.json``
- ``examples/publication_figures/artifacts/*.json``
- ``examples/publication_figures/generate_validation_dashboard.py``
- ``examples/publication_figures/generate_fortran_suite_benchmark_summary.py``
- ``sfincs_jax/validation_artifacts.py``

The dashboard tests are intentionally cheap enough for CI. They do not rerun the full
collisionality or electric-field scans; instead, they check the frozen artifacts for
physics invariants that come directly from the SFINCS validation literature:

- FP and PAS collision-operator scans must both be present on the audited
  collisionality grid.
- The high-collisionality ``L11`` FP/PAS separation must remain larger than the
  low-collisionality separation, matching the expected increased sensitivity to
  momentum conservation.
- PAS ``L11``/``L12`` tails must have positive high-collisionality power-law slopes,
  and any FP inverse-``nu`` claim must be recorded per geometry instead of inferred
  visually from a plot.
- The Simakov-Helander high-collisionality audit must recompute the Appendix-B
  geometry ingredients from checked-in ``sfincsOutput.h5`` fields and must keep the
  full analytic-limit gate closed unless the scan reaches the configured high-``nu``
  threshold.
- DKES, partial, and full trajectory models must agree exactly at ``E_r = 0`` on the
  pinned branch artifacts.
- Finite-``E_r`` trajectory sweeps must preserve nonzero model separation before the
  figure can be used as a manuscript-facing validation panel.
- The frozen CPU/GPU Fortran-suite benchmark artifact must retain ``39/39`` audited
  cases on both backends, with zero strict mismatches, zero ``jax_error`` cases, and
  zero ``max_attempts`` cases before the release comparison figure can be regenerated.
  The public runtime/memory figure then filters to reference-runtime-window rows
  with Fortran v3 runtime at least ``10 s``; the summary JSON records which
  legacy rows still need production-resolution reruns.

The corresponding tests are ``tests/test_validation_artifacts.py`` and
``tests/test_generate_validation_dashboard.py``. The high-collisionality plot smoke
tests are ``tests/test_generate_high_collisionality_trend_proxy.py`` and
``tests/test_generate_simakov_helander_limit_audit.py``. The frozen Fortran-suite
benchmark figure is protected by
``tests/test_generate_fortran_suite_benchmark_summary.py``.

Output-normalization gates
^^^^^^^^^^^^^^^^^^^^^^^^^^

The output layer also has physics-aware tests that avoid a full solve. In
``tests/test_output_coordinate_physics_gates.py``, a frozen multi-species output
fixture is checked for the radial-coordinate chain rules used when plotting
neoclassical particle, heat, and momentum fluxes versus ``psiHat``, ``psiN``,
``rHat``, and ``rN``. The same file checks density and temperature-gradient
conversion consistency. These gates are cheap enough for CI and catch a common
class of manuscript-facing mistakes: a plot can look reasonable while using an
inconsistent radial normalization.

The VMEC/JAX geometry adapter tests also avoid a solve when checking that the
normalized Boozer proxy transport objective is invariant under global
:math:`|B|` spectrum scaling and has zero value and gradient for constant
:math:`B`.

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
- ``external-data-smoke`` fetches the release-hosted W7-X/HSX/QI equilibrium
  archive into an isolated cache, then reruns VMEC-path output tests and the
  VMEC getting-started example with ``SFINCS_JAX_OFFLINE=1`` so CI proves that
  public release data is complete and usable without accidental network access,
- ``.github/workflows/docs.yml`` builds the Sphinx documentation,
- ``.github/workflows/publish-pypi.yml`` handles packaging/release publication.

Release-hosted data gates
^^^^^^^^^^^^^^^^^^^^^^^^^

Large public equilibrium fixtures are intentionally not tracked in git and are
not included in wheels. The CI contract for those files is:

- fetch the checksum-pinned ``sfincs-jax-data-v1`` release archive with
  ``python scripts/fetch_equilibria.py --quiet``;
- verify every manifest entry exists in the configured cache;
- rerun the public VMEC output path in offline mode;
- keep ``tests/test_data_fetch.py`` as the unit gate for manifest structure,
  checksum extraction, unknown-basename handling, and missing-cache offline
  failure behavior.

This prevents a common packaging regression: a repository can pass pure unit
tests while the release examples fail because a moved equilibrium file is
missing, renamed, or no longer checksum-compatible with the embedded manifest.

The current audited full-suite command on ``main`` is:

.. code-block:: bash

   pytest -q --cov=sfincs_jax --cov-report=term --cov-report=xml

The exact collected-test count changes as targeted regression tests are added, so
release notes should cite a dated local/CI artifact rather than hard-code a
permanent number here. The Linux CI coverage floor is intentionally conservative
until the remaining driver monolith is split further; the research-grade target
is ``95%`` meaningful package coverage, with the dominant uncovered surface still
the large solver/geometry stack, especially ``v3_driver.py``, ``io.py``,
``geometry.py``, ``grids.py``, and ``vmec_geometry.py``. The latest
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
synthetic operators. The next refactor extraction moved RHSMode=1 low-mode angular,
moment, coupled f/tail, and tail-only coarse-space construction into
``rhs1_lowmode_coarse.py`` with direct tests for feature normalization, rank
capping, compact matrix-free Galerkin and least-squares metadata, residual
projection, and tail-selection policies on bounded synthetic operators. The next
small extraction moved RHSMode=1 angular domain-decomposition patch and
two-level Schwarz sizing rules into ``rhs1_domain_decomposition.py`` with direct
tests for patch coverage, overlap clamping, shard-aware block sizing,
environment overrides, and coarse-level termination. These tests were
chosen from the same identities used in the
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
flow. The current QI coarse-basis pass adds focused tests for deterministic
rank gating, guarded least-squares residual reduction, callable matrix-free
operators, empty-basis guards, low-Fourier angular coarse directions, and
metadata from the opt-in pre-Krylov QI coarse seed path. These are still bounded
unit/regression gates. The documented large-QI non-autodiff host fallback now
covers production runs that need a robust solution today, while true
differentiable/device-QI closure still requires the documented scale-0.60 GPU
hard-seed artifact to pass within budget.
The follow-up Galerkin-policy tests make the opt-in device-QI path fail closed:
mode parsing, damping selection, nonfinite rejection, required-drop gates, and
production metadata are tested without launching a large solve. A Galerkin
preconditioner candidate is now used only when its probe reduces the true
residual; otherwise the base preconditioner remains active and the trace records
``probe_not_reduced``.
The runner also has a liveness gate for long GPU attempts: pass
``--heartbeat-s`` to write per-case ``runner_heartbeat.jsonl`` events outside
the solver trace. Tests cover both normal heartbeat completion and process-group
termination on timeout, so a hung GPU attempt leaves explicit evidence instead
of a silent SSH session.
The same policy-first strategy now covers solver-path logic: constraint-scheme
routing, sparse-exact-LU selection, x-block/sxblock rescue eligibility,
transport sparse-direct and host-GMRES guard rails, host-only SciPy Krylov
dispatch, and distributed-incompatible GMRES rejection. These tests do not
attempt to prove convergence of every large solve; they prove that the policy
ladder surrounding those solves takes the right branch on bounded, physically
meaningful synthetic inputs. The latest bounded ``io.py`` pass then extended
that same strategy to the output side: cache-directory selection,
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

The current release-preparation pass supersedes those historical counts. On
2026-06-13, the final local tree passed ``2510`` tests in ``562.68 s`` without
coverage instrumentation and ``2510`` tests in ``746.33 s`` with coverage
enabled. The final full coverage audit measured total package coverage at ``74%`` with
``v3_driver.py`` at ``56%`` and the newly extracted
``rhs1_lowmode_coarse.py`` at ``94%``. The remaining gap to ``95%`` is therefore
no longer broad helper coverage; it is concentrated in the largest solver,
I/O, sparse-pattern, and system-assembly bodies that still need further
behavior-preserving extraction.
The next main-branch refactor step extracted RHSMode=1 domain-decomposition
heuristics and passed the local full suite with ``2520`` tests in ``537.83 s``.
The dedicated architecture branch then started by moving v3 result dataclasses,
solver-runtime finite/synchronization helpers, matrix-reduction primitives,
preconditioner context state, and Krylov dispatch routing out of
``v3_driver.py`` with direct tests for
pytree round-trips, diagnostic metadata retention, finite-result rejection,
local-coupling preservation, small differentiable regularized least-squares
solves, sparse triangular factor solves, mutable
solver-hint state, dtype policy, and sparse structural-tolerance handling.
``preconditioner_setup.py`` adds direct tests for setup chunk sizing,
stable cache-key array hashing, injected matrix-free submatrix probing, and
RHSMode=1/transport cache-key stability for dtype, Phi1, PAS, and FP
signatures.
``krylov_dispatch.py`` adds direct tests
for host-only method rejection, JIT/plain/distributed dispatch, diagnostic
labels, and distributed-axis environment policy. ``preconditioner_caches.py``
adds direct tests for representative RHSMode=1, transport, sparse-ILU, Schur,
and PAS cache containers, plus identity checks proving that legacy
``v3_driver`` cache names still refer to the same registry objects.

The large refactor closure gate extends that strategy in two directions. First, it
splits RHSMode=1 preconditioner policy into directly tested helper modules, covering alias
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
Rosenbluth-potential assembly paths on a bounded grid. The same collision gate now
also checks the full Fokker-Planck apply path directly: dense speed-space matvecs,
inactive Legendre masking, shape guards, and the ``Phi1`` Boltzmann density factor
used when ``includePhi1InCollisionOperator`` is active. That pass found and fixed a
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
``tests/test_jax_geometry_adapters.py`` also checks the Boozer-spectrum objective
used by the optional ``vmec_jax -> booz_xform_jax -> sfincs_jax`` example.  The
mandatory unit part verifies the cosine-series evaluator and a directional
gradient against centered finite differences.  When both optional geometry
packages and a small VMEC fixture are available, the same file runs a real
``booz_xform_jax`` transform and checks the differentiable scalar gradient through
that transform.

The refactor branch also treats documentation discoverability as testable behavior.
``tests/test_policy_module_docstrings.py`` imports the split RHSMode=1 and transport
policy modules and checks that their explanatory module docstrings are real
``__doc__`` strings rather than inert comments. This is intentionally small, but it
keeps the source map and generated API documentation useful as the large driver is
split into manageable pieces.
``tests/test_rhs1_constraint_sources.py`` covers the extracted constraint-source
JAX kernels directly, including flux-surface averages, velocity-weighted
density/pressure moments, ``pointAtX0`` source injection, and the documented
constraintScheme=1 source basis. These are algebraic tests, not smoke tests, so
they protect the source/moment closure used by both RHSMode=1 and transport solves.

The docstring gate now discovers every ``sfincs_jax/*policy*.py`` module and also
checks public policy classes/functions, so new extraction seams must remain
discoverable. ``tests/test_transport_policy_coverage.py`` adds fast direct coverage
for transport backend/sparse-host/recycle policy, transport parallel
scaling-audit/environment helpers, and worker-local XLA flag rewriting without running
transport solves. ``tests/test_transport_parallel_payload.py`` separately validates the
transport worker payload contract: normalized solve kwargs, child-worker recursion
guards, merge-ready result packing, and GPU-worker NPZ conversion for non-contiguous
``whichRHS`` chunks.

QI seed-robustness artifacts are also checked as data, not only as scripts.
``tests/test_qi_seed_smoke_artifact.py`` verifies the one-seed smoke artifact,
the three-seed CPU artifact, the three-seed one-GPU artifact, the five-seed CPU
manifest-summary artifact, and the production-readiness evidence manifest. The
GPU artifact protects the moderate full-FP accelerator auto-dense policy that
avoids the slow sparse/fallback ladder for the fragile ``Nxi=25`` QI window while
keeping tiny GPU fixtures on the matrix-free path. The evidence manifest keeps
the lane at ``bounded_proxy`` until production-resolution CPU and GPU artifacts
exist and pass the same residual/convergence gates.
``tests/test_qi_device_research_lane_artifacts.py`` protects the separate
true-device-QI claim boundary. It checks that the best checked one-GPU
device-QI hard-seed residual is still recorded as fail-closed evidence, that no
nonconverged device-QI artifact writes HDF5 output or a solver trace, and that
the manifest keeps true device-QI marked ``closed_deferred`` rather than
promoted. This is intentionally a data gate: it prevents documentation or
release notes from turning a residual-improving but nonconverged hard-seed
probe into a production-resolution claim.
The bounded ``13 x 13 x 15 x 4`` office-GPU operator-reuse artifact is checked
separately by ``tests/test_qi_kinetic_res13_probe_artifact.py``. That gate
requires route activation, skipped local x-block factors, host fallback disabled,
fail-safe trace metadata, corrected device-cycle iteration/matvec accounting,
and explicit residual-gate failure. It is route-level evidence, not production
true-device-QI promotion.

The scan/CLI progress surface is also guarded without running expensive scan
points.  ``tests/test_scans_progress_and_recycle.py`` covers duration formatting,
ETA/reused-output messages, radial-gradient variable selection, namelist scalar
patching, stride/index subsetting, and serial scan-recycle state handoff.  These
tests protect user-facing runtime-estimate behavior and cross-run Krylov reuse
while keeping CI cost below one second for the new file.

The latest driver split also extracts RHSMode=1 host dense/sparse-direct policy into
``sfincs_jax/rhs1_host_policy.py``. ``tests/test_rhs1_host_policy.py`` covers the
backend/env rules for host dense fallback, small accelerator dense shortcuts,
host sparse-direct enablement, sparse-preconditioned GMRES rescue, sparse factor
dtype selection, iterative-refinement step parsing, explicit sparse-helper
bounds, and the FP/PAS dense-fallback active-size ceiling.  The public driver
wrappers remain tested separately, so this is a
behavior-preserving refactor with a smaller directly testable policy surface.
The adjacent ``tests/test_host_refinement.py`` covers the host direct
refinement kernels and sparse-direct GMRES polish helper directly, including
solver injection and sparse-factor preconditioner application.

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

The sparse helper coverage in ``tests/test_v3_driver_sparse_helper_coverage.py``
also protects the host full-FP x-block exact-LU cap. It verifies that only the
non-differentiable full-FP x-block path defaults to the larger production-floor
cap, while PAS and JAX-factor paths retain the lower memory-conservative cap.
``tests/test_explicit_sparse_factor_policy.py`` now covers the typed
``ExplicitSparseFactorSettings`` bundle consumed by that sparse builder:
defaults, environment overrides, bounds, factor-kind aliases, SuperLU
permutation fallback, ILU options, monolithic LU/ILU guards, and symbolic
Schur/frontal/ND/BLR knobs are checked without launching a large solve.
``tests/test_explicit_sparse_factor_builder.py`` then tests the extracted
builder itself with injected operator-build, pattern-build, backend, guard, and
factorization callbacks. The existing driver sparse-helper tests remain in
place to prove that monkeypatching the ``v3_driver`` compatibility seam still
reaches the extracted implementation.
``tests/test_rhs1_direct_tail_policy.py`` covers the next adjacent extraction:
direct-tail cache-key hashing, support-mode/env sensitivity, cache-hit metadata,
direct reduced-Pmat aliases, adaptive memory-cap policy, and the structured
host sparse adapter without running a production solve.
``tests/test_rhs1_true_operator_rescue.py`` covers the next RHSMode=1
true-operator rescue split: residual/coarse bundle application, reusable
true-action column caching, additive-rescue storage accounting, sparse-factor
storage estimates, graph expansion, residual-window parsing, and residual-window
selection with tail variables. It also covers residual sparse-window/coarse
builder application and active residual component diagnostics. Existing
sparse-pattern tests still exercise the ``v3_driver`` compatibility aliases and
the higher-level builder paths.
``tests/test_rhs1_xblock_sparse_host_policy.py`` also covers the production
host x-block factor cap: by default, very large local sparse factors are skipped
instead of spending the full runtime budget on singular ILU attempts, while an
explicit ``SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX=0`` keeps the
historical uncapped experiment available. ``tests/test_v3_sparse_pattern.py``
then checks that singular local ILU attempts escalate diagonal regularization
before failing closed, which makes high-resolution blocker evidence more
diagnostic without changing the physical operator.

The follow-up post-x-block policy split is covered by
``tests/test_rhs1_post_xblock_policy.py``. These tests check the residual and
active-size gates for fast post-x-block polish, targeted FP polish, and explicit
disable/override behavior for skip-global-sparse-after-xblock routing after a
good x-block seed. The tests keep
large-case convergence handoff decisions visible while avoiding heavyweight CI
runs. ``tests/test_rhs1_sparse_polish_policy.py`` also covers the large-system
post-x-block polish budget: for active systems above the configured floor, the
default SciPy polish is a short ``restart=10, maxiter=1`` probe rather than a
second long solve, unless the user explicitly overrides the restart or maxiter
environment variables.

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

The solver-path refactor continues this policy-first testing style in
``sfincs_jax/solver_path_policy.py`` and ``tests/test_solver_path_policy.py``.
The direct tests cover JIT eligibility, preconditioner dtype selection, the narrow
automatic FP32 PAS geometry-4 preconditioner gate, residual-rescue slack,
DKES GMRES budget preservation for explicit user budgets, sparse-PC default
permutation/restart choices, sparse structural tolerance parsing, and
resource-exhaustion error classification. These tests do not prove convergence of
new large cases; they keep solver-path decisions reproducible and inspectable so a
future performance promotion is not hidden inside driver control flow.

The measured solver-candidate gates are covered separately in
``tests/test_solver_selection_policy.py``. Those tests exercise residual/parity
rejection, baseline-clean promotion requirements, failed-baseline rescue
allowances, paired memory metrics, tie-breaking, and missing-measurement guards.
The adjacent RHSMode=1 handoff seam is covered in ``tests/test_rhs1_handoff.py``:
the strict finite residual-improvement predicate is tested directly, nonfinite
candidates are rejected even before measured gates run, finite rescues after a
nonfinite incumbent are accepted, and measured runtime/memory regressions still
block an otherwise lower residual when the incumbent is already clean. The
docstring audit in ``tests/test_policy_module_docstrings.py`` now also covers
this source-mapped non-policy control module so the handoff contract remains
discoverable while ``v3_driver.py`` is split further.
Transport worker residual abort formatting is covered in
``tests/test_transport_residual_quality.py``, including custom environment names,
negative/invalid threshold normalization, nonfinite residual diagnostics, and
array-to-message collection. These are intentionally fast policy checks; they
protect solver-path diagnostics without launching transport solves.

The output/helper layer is kept under small, direct tests rather than only
end-to-end HDF5 comparisons. ``tests/test_io_export_and_h5_coverage.py`` covers
Fortran-layout HDF5 round trips, export-``f`` identity, nearest-neighbor, periodic
linear wrapping, single-zeta, and invalid-option branches. ``tests/test_phi1_history_alignment.py``
checks the accepted-iterate padding rules used when writing ``Phi1`` diagnostics,
and ``tests/test_input_compat.py`` checks equilibrium-file localization for quoted,
unquoted, missing, VMEC, Boozer, and non-stellarator-symmetric input conventions.
These tests protect user-facing CLI/output behavior without adding long solve
cases to CI.

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

Coverage-to-95 plan
-------------------

The ``95%`` target is useful only if it reduces real scientific and operational
bugs. Literature on scientific-software testing repeatedly highlights the oracle
problem, and empirical software-engineering work warns that coverage alone is a
weak proxy for test effectiveness. The project therefore treats coverage as a
gap-finding metric, not as the final quality metric.

The staged path is:

1. **Refactor before raising the floor.** Continue extracting ``v3_driver.py`` and
   large I/O branches into pure policy, residual, normalization, output-schema,
   and preconditioner modules. Each extracted module must land with module-level
   docstrings, source-map documentation, direct unit tests, and one driver-wrapper
   regression so behavior stays unchanged.
2. **Add cheap physics oracles.** Prefer tests based on conservation, symmetry,
   limiting behavior, and normalization identities: PAS ``L=0`` null modes,
   collision positivity/symmetry where applicable, zero-drive flux limits,
   finite-difference order conditions, Fourier/circulant exactness, VMEC
   interpolation conventions, Boozer-coordinate field-component identities,
   radial-coordinate chain rules, and trajectory-model equivalences at
   ``E_r=0``.
3. **Use metamorphic and property-based tests for hard oracles.** For input
   parsing, scan orchestration, output-format selection, path localization,
   environment variables, and solver-policy branch selection, generate families
   of small cases and assert invariants under harmless transformations instead
   of storing many large fixtures.
4. **Keep backend equivalence cheap.** CPU/GPU and ``jit``/eager equivalence tests
   should use tiny bounded fixtures, synthetic operators, or frozen artifacts.
   Production CPU/GPU and Fortran comparisons remain release or nightly gates,
   not every-commit CI gates.
5. **Gate every fixed bug.** Any bug found in profiling, solver selection,
   output writing, release-data lookup, or geometry loading gets a regression
   test at the smallest level that reproduces it, plus a higher-level test only
   when the bug was caused by orchestration.
6. **Raise CI thresholds in steps.** Move the fail-under gate only after each
   extraction batch makes the denominator meaningful: ``43 -> 60 -> 75 -> 85 ->
   90 -> 95``. Each increase requires full CI, strict docs, release-data smoke,
   and the fast release gates to pass within the target wall-time budget.
7. **Keep size and runtime bounded.** New tests should use generated synthetic
   fixtures, release-hosted data, or compact JSON artifacts. Do not add
   multi-megabyte tracked fixtures to increase coverage; the repo-size gate and
   release-data manifest are part of the testing strategy.

The practical completion criterion is not only ``95%`` line coverage. A release
is considered research-grade when the coverage floor, physics gates, output-key
coverage, CPU/GPU equivalence gates, release-data gates, docs build, and
runtime/memory benchmark artifacts all agree.

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
When a run changes solver branches unexpectedly, summarize the emitted profiling
marks and preconditioner ladder with ``scripts/summarize_solver_paths.py``; this
is the lightweight audit used to close the full-FP dense/Krylov GPU policy issue.
For long RHSMode=1 profiling runs, keep the trace helper's phase log in the artifact
bundle. Its default ``profile_write_output_trace_phases.json`` sidecar makes it
clear whether a nonzero wrapper status came from the solve itself or from profiler
finalization after a valid output file was already written.
``tests/test_profiling_helpers.py`` keeps the lightweight profiler directly
covered: environment opt-ins, resource fallback units, unavailable OS/JAX memory
sampling, and ``na`` formatting are checked without launching a solver run.

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
- ``scripts/check_release_gates.py`` and ``tests/test_release_gate_metadata.py`` add a
  CI-fast release gate over the manifest's ``release_gate`` metadata. Each lane must be
  ``release_ready``, ``regression_scaffold``, ``bounded_proxy``, or
  ``closed_deferred``; none may block the current release without being explicitly
  closed or removed from the release manifest. The release checker now also validates
  the record status/kind, the non-empty literature/claim/source/test/gate lists, and
  all listed source, test, script, and artifact paths, including paths on
  ``closed_deferred`` lanes. A deferred lane is therefore allowed to stay out of the
  tagged release claim, but it is not allowed to rot silently.
- ``docs/_static/research_lane_completion_2026_05_12.json`` records the active
  research/performance lanes, evidence artifacts, completion estimates, gates,
  and next actions for the current large-push cycle. ``scripts/check_research_lanes.py``
  and ``tests/test_research_lane_policy.py`` enforce that those percentages are
  evidence-backed and that active lanes record substantial measured progress
  before their completion estimate is increased. The policy is target-capped:
  if a lane has fewer percentage points remaining than the current push target,
  it must reach its checked target rather than overclaiming beyond it.
- ``scripts/check_qi_device_artifacts.py`` and
  ``tests/test_qi_device_artifact_policy.py`` add the QI-specific offline gate
  for device/operator-reuse artifacts. The gate checks provenance and
  fail-closed metadata only; it is not a convergence certificate. The release
  metadata checker now runs the same policy over ``docs/_static`` so legacy or
  newly checked QI-device artifacts cannot silently overclaim GPU/device status.

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

Mapped x-grid and QI integration smoke gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mapped speed-grid work is intentionally split between cheap primitive tests and
bounded solve-facing evidence. The primitive and objective tests are:

- ``tests/test_adaptive_maps.py``
- ``tests/test_mapped_xgrid_objectives.py``
- ``tests/test_mapped_xgrid_v3.py``
- ``tests/test_mapped_xgrid_transport_evidence.py``
- ``tests/test_run_mapped_xgrid_transport_evidence.py``

The checked reviewer artifacts are bounded PAS RHSMode=2 comparisons, not default
production claims:

- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.json``
- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.csv``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.json``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.csv``

The reduced PAS tokamak artifact compares mapped ``Nx=7`` candidates against an
``Nx=13`` reference with active-DOF reduction. It is useful evidence that the
mapped-grid machinery can run through a real transport-matrix solve, but it is not
evidence for full-FP compatibility, a default-grid replacement, or a
production-resolution speedup.

The QI seed-robustness runner is guarded by
``tests/test_run_qi_seed_robustness.py``. It materializes deterministic
neighboring cases, localizes the VMEC equilibrium beside each generated
``input.namelist``, perturbs ``nu_n`` and ``Er`` by seed, and can optionally run
``sfincs_jax write-output`` while recording stdout, stderr, and solver-trace paths.
The checked summaries in ``docs/_static/qi_seed_robustness_smoke.json``,
``docs/_static/qi_seed_robustness_multiseed.json``, and
``docs/_static/qi_seed_robustness_multiseed5_cpu.json`` record low-resolution
default CLI evidence. The three-seed CPU/GPU artifacts run neighboring seeds at
``7 x 13 x 25 x 4`` and record ``process_failed=0``, public solver method
``auto``, all seeds ``converged=true``, and maximum residual ratio below
``1e-6``. The five-seed CPU artifact extends the CPU ladder to seeds ``0..4``
from the reusable manifest-summary writer with ``timed_out=0``,
``outputs_written=5``, ``solver_traces_written=5``, and maximum residual ratio
``7.88e-7``. The larger
``docs/_static/qi_seed_robustness_scale035_cpu_gpu.json`` artifact records the
bounded ``9 x 19 x 35 x 4`` CPU/GPU gate that caught and fixed the accelerator
Krylov-tail failure: the GPU case moved from a ``195 s`` rejected solve with
residual ratio ``53.9`` to a ``42.8 s`` converged solve with residual ratio
``4.49e-7``.

The bounded ``docs/_static/qi_seed_robustness_scale045_cpu_probe.json`` artifact
records the largest checked passing CPU probe in this lane so far:
``11 x 23 x 45 x 4`` completed in ``106.1 s`` with public ``auto`` solver
selection, output and solver trace written, ``converged=true``, and residual
ratio ``4.96e-7``. It raises the passing per-axis readiness estimate while still
remaining below the production CPU/GPU five-seed requirement.

The bounded ``docs/_static/qi_seed_robustness_scale050_cpu_probe.json`` artifact
records a deliberately timeout-capped CPU probe at ``13 x 27 x 50 x 4``. It
timed out after ``180 s`` before writing an output or solver trace, so it is
blocker evidence only. The evidence manifest records this as the largest
attempted size but excludes it from the checked-size and lane-completion
estimate.

The follow-up
``docs/_static/qi_seed_robustness_scale050_solver_matrix_2026_05_12.json``
artifact keeps the scale-0.50 blocker actionable without checking in copied VMEC
run directories. It compares eight bounded CPU routes at the same
``13 x 27 x 50 x 4`` resolution. The public ``auto`` route still times out after
``360 s`` after building an explicit FP x-block seed; ``sparse_host_safe`` fails
host sparse LU on a ``126365616``-entry conservative pattern; ``sparse_lsmr``
finishes in ``125.9 s`` but stalls at residual ``5.09e-6`` against target
``2.51e-11``; and ``xblock_sparse_pc_gmres`` reaches the same residual floor
after ``32000`` GMRES iterations. An opt-in initial x-block seed probe was also
tested; it was rejected because the seed residual was slightly worse than the
RHS norm, and the run still stalled at ``5.41e-6``. A first opt-in post-GMRES
matrix-free minimum-residual hook was then tested with four requested steps; it
accepted two corrections but only changed the true residual from
``5.413504e-6`` to ``5.409759e-6``. A stronger opt-in 10-direction coarse
least-squares correction improved the same floor only to ``5.401187e-6``. An
opt-in LGMRES variant was also tested as a restart-robust Krylov alternative; it
stalled at ``5.577462e-6``, fell back to GMRES, doubled the matrix-vector count
to ``65204``, and ended at the original ``5.413504e-6`` floor after about
``300 s``. This
makes the next required algorithmic step a different global coupling strategy,
not a larger timeout, full sparse materialization, default initial-seed probe,
scalar post-minres cleanup, the current small residual subspace, or a Krylov
method toggle alone.
The next implementation step now lives behind
``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION``: it reuses the final
Krylov residual and cached QI ``(U, A U)`` columns in a bounded JAX
least-squares residual equation. Its unit and driver tests currently validate
fail-closed residual reduction and metadata/output visibility; promotion still
requires a converged hard-seed CPU/GPU artifact.

The successor
``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json`` artifact
and ``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json``
artifacts close the same scale-0.50 CPU and one-GPU seed through the promoted
right-preconditioned explicit ``xblock_sparse_pc_gmres`` route. That route uses
exact sparse LU for medium non-differentiable full-FP host x-block factors with
the cap raised to ``30000``. The checked CPU run converges at
``13 x 27 x 50 x 4`` in ``~12 s`` with true residual ``1.04e-12`` against target
``2.51e-11`` and residual ratio ``4.16e-2``. The clean-clone one-GPU run on
``office`` converges in ``~44.5 s`` with residual ``1.58e-11`` and residual
ratio ``0.63``. The companion solver trace
``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu_solver_trace.json``
records ``precondition_side=right``, ``gmres_restart=80``, ``81`` Krylov
iterations, ``85`` matvecs, and exact ``sparse_lu`` block factors; the GPU trace
records the same policy with ``69`` iterations and ``72`` matvecs.

The follow-up
``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json``
and
``docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json``
artifacts extend that route to seeds ``0..4`` on CPU and one GPU. Both ladders
use the public ``solve_method=auto`` path, select ``xblock_sparse_pc_gmres``
internally, write all five outputs and solver traces, have zero process failures
and zero timeouts, and keep ``max_residual_ratio < 1``. The CPU ladder completes
with maximum elapsed time ``11.58 s`` and maximum residual ratio ``0.966``. The
one-GPU ``office`` ladder completes with maximum elapsed time ``41.18 s`` and
maximum residual ratio ``0.963``. These artifacts close the bounded public-auto
solver-route robustness blocker, but they are not yet a production-resolution QI
claim.

The next-scale
``docs/_static/qi_seed_robustness_scale055_auto_cpu_blocker.json`` artifact keeps
that boundary honest. It raises the public-auto CPU grid to
``15 x 29 x 55 x 4`` with the bounded x-block sparse-PC window widened for the
probe, but the old exact-LU cap sent the largest x-block into ILU and timed out
after ``360 s`` before writing output or a solver trace. The successor
``docs/_static/qi_seed_robustness_scale055_xblock_lu_right_cpu.json`` artifact
keeps the same widened auto window while using the new full-FP host exact-LU cap
of ``30000``. It writes output and solver trace in ``~21.5 s`` with active size
``52637`` and residual ratio ``8.25e-3``. This closes the CPU setup cliff at
scale ``0.55``; it is still bounded evidence, not a production QI claim, because
the matching GPU and wider multi-seed ladders have not been checked in.

The hard-seed follow-up
``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json``
checks the same ``15 x 29 x 55 x 4`` bounded scale on seed ``3``. That seed was
the right-preconditioned slow-mode outlier in the five-seed CPU/GPU probes, so
the default policy now keeps right-PC only below the measured 3D full-FP
active-size window and switches this larger 3D full-FP case to left-PC. The
artifact records ``precondition_side=left``, zero process failures, output and
solver trace written, residual ratio ``2.98e-3``, and elapsed time ``~47 s``.
This is a solver-policy robustness gate; it does not by itself promote the
production-resolution QI target.

The matching
``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json``
and
``docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json``
artifacts extend the adaptive-side policy to seeds ``0..4`` on CPU and one
``office`` GPU. Both use the public ``solve_method=auto`` path, select
left-preconditioned ``xblock_sparse_pc_gmres`` internally, write all five
outputs and solver traces, and have zero process failures or timeouts. The CPU
ladder completes with maximum elapsed time ``44.5 s`` and maximum residual
ratio ``5.88e-3``; the one-GPU ladder completes with maximum elapsed time
``206.7 s`` and maximum residual ratio ``8.28e-3``.

The next-size seed-0 artifacts
``docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json`` and
``docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json``
raise the bounded grid to ``15 x 31 x 60 x 5`` with active size ``81377`` and
total size ``139502``. Both pass with left-preconditioned exact-xblock-LU GMRES,
zero process failures, output and solver trace written, residual ratios below
``4.7e-3``, and elapsed times ``42.2 s`` on CPU and ``145.1 s`` on one GPU. This
is used only to advance the next-size readiness estimate; production promotion
still requires multi-seed CPU/GPU evidence at the larger target.

The scale-0.60 seed-3 follow-up
``docs/_static/qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json``
keeps the GPU hard-seed blocker actionable without promoting another solver
toggle. It records rejected two-level x-block, GCROT(m,k), BiCGStab fallback,
post-correction-only BiCGStab, and experimental JAX-factor/device-Krylov probes.
The only retained code policy from that pass is the safe GMRES fallback guard:
a non-GMRES candidate may seed fallback GMRES only when it strictly improves the
finite RHS norm and is not a right-preconditioned coordinate state.

The subsequent global-coupling/operator-reuse probe summary
``docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json``
adds another negative gate. It verifies that the newly implemented opt-in
smoothed global-coupling preconditioner, assembled-operator matvec reuse
preflight, side-probe keep-left guard, and JAX-factor switch are wired and
metadata-covered, but it rejects all of them for default promotion on the
scale-0.60 seed-3 hard case. The important behavioral result is that the GPU
left side probe reached a finite near residual, while both the old right-switch
continuation and the new keep-left continuation timed out at ``620 s``. This
keeps the lane honest: the next closing step must be a genuinely device-resident
or differently structured preconditioner/Krylov formulation, not another
threshold-only side-selection tweak.

The device-operator rejection artifact has since been extended with the
cycle-JIT evidence. It records that full-solver JIT is not viable for this hard
seed because it reaches ``56.6 GB`` RSS, while cycle-JIT and recycled cycle-JIT
keep the one-GPU memory footprint near ``13.9 GB`` and write diagnostics within
the bounded window. They still fail the strict true-residual gate, so tests treat
them as infrastructure coverage and negative physics evidence rather than a
closed QI validation.

The current development branch adds that next formulation as an opt-in test
surface, not a promoted claim: ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres``
selects a JAX-native flexible GMRES primitive, while ``gmres-jax`` selects the
same fixed-shape Arnoldi/least-squares primitive with a fixed left
preconditioner so left-preconditioned device probes can be tested without SciPy
Krylov. Both routes force JAX x-block factors and use a device-resident
global-coupling coarse correction when global coupling is enabled. The checked
device-Krylov rejection artifact shows a useful robustness improvement on the
scale-0.60 GPU hard seed: the route finishes before the timeout and avoids the
earlier CUDA illegal-address failure, but it still fails strict true-residual
acceptance. Unit tests cover the solver primitive, JIT tracing, policy parsing,
small full-system metadata, and host-transfer-free metadata boundaries, while
the production QI gate above remains unchanged until the scale-0.60 seed-3 and
production-resolution CPU/GPU ladders pass.

The 2026-05-15 compact-factor apply diagnostics add another negative gate. The
runner now includes diagonal, exact-LU diagonal, row-cap-16 exact triangular,
forced-left FGMRES, and left ``gmres_jax`` artifacts in the QI evidence manifest.
These probes demonstrate that a GPU-cheap diagonal apply can return restart
cycles without the old timeout, but it does not reduce the physical residual.
Tests therefore classify these artifacts as non-passing blocker evidence and
keep the completion estimate tied to the largest passing measured artifact.

The follow-up device-operator artifact exercises the next operator-reuse step on
the same ``office`` GPU hard seed. Full-restart device FGMRES builds the active
device CSR operator but times out after ``400`` device matvecs. The
short-recurrence ``bicgstab-jax`` variant reduces peak RSS to ``13.6 GB`` and
finishes before timeout, but diverges and therefore remains rejected. This is
valuable blocker evidence because it separates memory pressure from
preconditioner/Krylov stability.

The same blocker artifact now includes the final conditioning probes for this
push. Row equilibration, two-sided row/column equilibration, and a larger
x-block JAX factor row cap all complete the bounded GPU hard seed but leave the
physical residual at the same ``3.02e-5`` floor. A closer device analogue of the
CPU-closing route, exact per-x sparse LU with left device GMRES, reaches the
intended factors but times out inside the bounded GPU window. Tests therefore
keep these routes as negative infrastructure evidence. The compact-CSR
exact-factor replacement is now tested too: it stores actual SuperLU factor
nonzeros rather than padded rows and builds the full exact factors, but the
bounded GPU hard seed still times out before a solver trace. The remaining QI
closure therefore requires a cheaper exact/block-Schur application or a
different residual-reducing coarse operator, not another scaling or restart-only
knob.

The latest QI coarse-seed follow-up adds bounded liveness evidence for that
same one-GPU hard seed. The CPU artifact
``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json``
passes at ``15 x 31 x 60 x 5`` with an accepted residual ratio below
``1.4e-3``. The matching GPU heartbeat artifact
``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json``
records ``31`` heartbeat events over ``420 s`` and preserves the active/total
matrix sizes even though no solver trace was written. It is negative evidence:
that diagnostic launch forced the host-oriented LGMRES rescue on GPU and timed
out, so it does not promote the GPU policy. The follow-up no-LGMRES artifact
``docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json``
confirms the GPU-compatible branch: the side probe switches left-to-right with
plain GMRES, probe-coarse improves the seed to ``2.83e-6``, and the solve reaches
``900`` matvecs by ``412.7 s`` before timeout. It is still blocker evidence
because it writes no output or solver trace.

``docs/_static/qi_seed_robustness_evidence_manifest.json`` rolls those artifacts
into the current production-readiness gate. It records the production target
``25 x 51 x 100 x 4`` with estimated total size ``510002``, the largest checked
passing bounded grid ``139502``, the largest attempted grid ``510002``,
32 passing artifacts and 80 non-passing blocker artifacts, a ``60%``
per-axis lane-completion estimate based only on passing artifacts, and
``72.65%`` of production total size still uncovered. The production acceptance
gate requires five seeds on both CPU and one GPU with ``public_cli_default_path``,
``solve_method=auto``, ``process_failed=0``, ``timed_out=0``,
``outputs_written=5``, ``solver_traces_written=5``, ``converged=5``, and
``max_residual_ratio <= 1``. Treat these as bounded runner and solver-policy
evidence, not as a production-resolution QI robustness claim. The separate
kinetic-promotion lane has now closed the first QI ``nfp=2`` low-resolution
CPU/GPU/Fortran artifact, a refined ``9 x 9 x 11 x 4`` CPU/GPU/Fortran rung,
and a second ``11 x 11 x 13 x 4`` CPU/GPU/Fortran rung that also verifies the
bounded dense-policy fix for mid-size RHSMode=1 full-FP systems. The
resolution-ladder root drift remains large enough that the ladder is still
open.
Promote QI robustness only after the next bounded scale and production-resolution
CPU/GPU seed ladders are checked with solver traces, and promote the QI
electron-root kinetic claim only after the CPU/GPU/Fortran resolution ladder is
stable and the evidence manifest is regenerated.

The active-pattern GPU probe
``docs/_static/qi_seed_robustness_scale060_active_pattern_device_qi_gpu0.json``
is also included as fail-closed evidence. It observes the residual-selected
pitch/angular/radial/species coarse path, but the solve still refuses output at
residual ``1.622338e-5`` against the hard-seed write gate. It is tracked so the
negative result is reproducible and not promoted accidentally.

The next non-smoother probe is ``coupled-residual-device-qi``. It asks the
driver to build multilevel, residual-snapshot, block-Schur, and flat coarse
sources, then solve one joint cached ``A Q`` residual equation. If the coupled
stage is internally accepted but the seed probe is too weak, the opt-in preset
can install the stage as the Krylov preconditioner without changing ``x0``.
This is the current reviewer-facing test of the Schur/coarse-residual
hypothesis: a passing artifact must report
``xblock_qi_device_preconditioner_coupled_residual_equation=True``, write HDF5
and solver trace, satisfy the residual gate, and remain on the device-QI path
without host fallback. Until such an artifact exists, the evidence manifest
keeps the lane fail-closed.
The first one-GPU Krylov-install artifact,
``docs/_static/qi_seed_robustness_scale060_coupled_residual_krylov_install_device_qi_gpu1.json``,
does report observed coupled residual-equation setup and installation inside
Krylov, and it reduces runtime/RSS relative to the seed-gated coupled attempt.
It still refuses output because the residual remains above the write gate, so
the manifest records it as fail-closed blocker evidence. The runner tests also
assert that these coupled setup/install progress lines survive compacting even
when a long GPU run fails before writing HDF5 or solver trace metadata.
The follow-on post-Krylov residual-equation CPU artifact,
``docs/_static/qi_seed_robustness_scale060_post_residual_equation_device_qi_cpu_2026_05_22.json``,
is likewise fail-closed but machine-readable: the compact runner summary records
the accepted residual-equation correction, ``89`` directions, and the measured
true-residual reduction ``2.362283e-05 -> 2.105918e-05``. The matching GPU1
artifact,
``docs/_static/qi_seed_robustness_scale060_post_residual_equation_device_qi_gpu1_2026_05_22.json``,
records the same hook and reduces ``2.450895e-05 -> 2.142936e-05``. The gate
remains open until a CPU and GPU artifact write converged HDF5 plus solver trace
metadata.
For the current release documentation this is a scoped research result, not a
production claim: this hard seed is now below ``3e-5`` on CPU and GPU, while
additional algorithmic work is still required to reach the production write
tolerance.

The current QI device-preconditioner unit gate now checks one more non-smoother
piece of infrastructure: coupled residual-equation setup batches the ``A Q``
operator-action construction with ``jax.vmap`` when the operator has a batching
rule, and the installed device preconditioner reuses that cached action instead
of recomputing it.  The same state records ``jax_default_backend``,
``jax_available_platforms``, operator array devices/platforms, and reuse versus
recompute stage counts.  ``tests/test_rhs1_qi_coupled_residual.py`` and
``tests/test_rhs1_qi_device_preconditioner.py`` lock these metadata fields on
small systems so the large GPU lane can be audited without launching a long
solve.

QI device artifacts also have a CI-fast overclaiming gate:

.. code-block:: bash

   python scripts/check_qi_device_artifacts.py \
     docs/_static --min-relevant 1

This checker is intentionally narrower than the PAS benchmark artifact policy.
It classifies QI/device JSON artifacts, requires backend/provenance and claim
boundary fields, rejects operator-reuse routes that silently fall back to host
x-block factors, and requires fail-closed artifacts to refuse nonconverged
outputs. Older fail-closed GPU blocker artifacts may use the ``gpu0``/``gpu1``
file name as provenance only when they write no output and fail their gates.

The latest residual-weighted angular probe-coarse artifact
``docs/_static/qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json``
is the accepted CPU hard-seed reference for this bounded scale: it passes in
``170.7 s`` with residual ratio ``2.14e-3``. The enriched QI basis,
lower-fill local ILU, no-LGMRES GPU, and compact-factor device-Krylov follow-up
artifacts are deliberately kept as rejected evidence because they either regress
CPU time/memory or fail to write GPU HDF5/trace output.

Large device-Krylov QI solves now have a tested non-autodiff host fallback
policy. ``tests/test_rhs1_xblock_policy.py`` covers the auto/force/disable
decision gates, and
``tests/test_v3_sparse_pattern.py::test_xblock_sparse_pc_device_host_fallback_records_non_autodiff_host_policy``
checks that an explicit device-Krylov request can be rewritten to the host
x-block auto policy before JAX factor arrays are built. Large QI runs then
retain the measured side-probe seed plus host SciPy ``lgmres`` rescue rather
than launching direct LGMRES from a weak zero seed. This is a production-safe
route for large RHSMode=1 QI runs that need a solution today; it is deliberately
metadata-visible and does not count as an end-to-end differentiable device
Krylov closure.

The high-collisionality Simakov-Helander lane now has a bounded normalization audit:

- script: ``examples/publication_figures/generate_simakov_helander_limit_audit.py``
- artifact:
  ``examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_audit_summary.json``
- focused test: ``tests/test_generate_simakov_helander_limit_audit.py``

This audit checks that the checked-in geometry output fields are sufficient for the
Appendix-B comparison and that ``FSABHat2`` is reproduced from ``BHat`` and ``DHat``.
It intentionally does not close the full analytic-limit reproduction, because the
current audited collisionality scans stop near ``nu'=10``.

The deferred Simakov-Helander panel-data scaffold is also executable in
``sfincs_jax.validation_figures`` and guarded by
``tests/test_validation_figures.py``. It consumes a compact payload of
``nuprime``, computed value, and analytic-limit rows, then records:

- sorted panel data and normalized distance to the analytic limit,
- tail log-log slope and monotonic approach metadata,
- high-``nu`` range gates for threshold, point count, and decade span,
- provenance completeness scores,
- matching checked-in source-artifact status,
- and explicit ``deferred_reasons``.

The scaffold only marks a panel as literature-ready when all numerical,
high-``nu`` range, provenance, and checked-in-artifact gates pass. Otherwise it
keeps the label as a deferred scaffold and reports whether the blocker is the
scan range, the asymptotic trend, provenance, or source-artifact status.
The panel metadata now also carries a ``publication_figure`` block so downstream
plotting code can distinguish a checked-in converged artifact from a proxy or
deferred scaffold without inferring that status from the title text.

The executable high-``nu`` run plan is also gated as a run plan, not as evidence:
its summary records ``run_plan_only_not_completed_validation``,
``commands_require_residual_gates``, and ``ready_for_literature_claim=false``.
This keeps the Simakov-Helander lane closed until the generated commands produce
checked-in converged scan artifacts and the audit gate flips.

The W7-X high-``nu`` preconditioner performance figure is intentionally narrower
than a physics-validation lane. Its summary supports a single-point performance
claim only when the factor-reuse route is residual-clean, faster than no-reuse,
uses fewer sparse factorizations, matches the no-reuse residual series, and the
failed bounded Krylov route is explicitly rejected. The same metadata marks
``ready_for_physics_validation_claim=false`` so this performance artifact cannot be
mistaken for a closed W7-X or Simakov-Helander physics validation.

The W7-X ambipolar literature lane has an executable scaffold as well:

- script: ``examples/publication_figures/generate_w7x_ambipolar_validation.py``
- focused test: ``tests/test_generate_w7x_ambipolar_validation.py``

This keeps the W7-X ambipolar validation work out of the "purely aspirational" bucket:
the scan, ambipolar postprocessing, summary JSON, and figure generation paths are now
covered by a bounded end-to-end test on a tiny fixture. The heavy W7-X reference
artifact is closed in the manifest as ``deferred_post_release`` until a defensible
profile/equilibrium reconstruction is pinned.

The deferred panel data also records explicit ``deferred_reasons`` and provenance
completeness scores. This keeps manuscript-facing labels conservative: a W7-X
ambipolar figure remains a scaffold until the numerical root gates pass and the
matching W7-X provenance artifact is complete and checked in.
The summary generator mirrors those gates directly: it now requires finite distinct
``E_r`` scan points, a radial-current sign-change bracket, reported roots inside
the scanned range, root consistency with that bracket, a resolved local current
slope, an ion-root candidate, complete provenance, and checked-in source-artifact
status before ``ready_for_literature_claim`` can become true.

The same scaffold is now resumable for heavy runs: ``run_er_scan`` accepts
``skip_existing=True``, the ``sfincs_jax scan-er`` CLI exposes ``--skip-existing``,
and the publication script adds ``--skip-existing``, ``--scan-only``, and
``--index/--stride`` so the heavy W7-X reference ladder can be filled across multiple
devices before a final aggregation pass. Each non-skipped scan point now also
writes ``sfincsOutput.solver_trace.json`` beside ``sfincsOutput.h5``. Tests assert
that the sidecar path is forwarded in serial, recycled, and end-to-end scan
paths, so future promotion artifacts cannot silently lose solver-path timing and
residual provenance.

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
so the in-tree implicit solve remains the only admissible production path. The summary
gate records these as ``residual_clean_status_mismatch_rows`` and keeps the adoption
decision at ``do_not_promote_lineax_status_mismatch`` for real SFINCS rows.

The Equinox/JAXopt gate is a separate objective-wrapper check on a real differentiable
``geometryScheme=4`` harmonic-fit problem. Its associated test
``tests/test_optional_eqx_jaxopt_scheme4_gate.py`` verifies deterministic problem
construction, directional-derivative agreement for an ``equinox.Module`` wrapper,
the opt-in JAXopt gradient-descent row when that package is installed, JSON
output, and clean skip behavior when either optional package is absent. Default
CI installs ``equinox`` but not ``jaxopt`` so the maintained wrapper path and the
historical JAXopt skip path are both exercised without making JAXopt a release
dependency.
