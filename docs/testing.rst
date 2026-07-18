Testing, validation, and CI
===========================

`dkx` is validated with a layered testing strategy. The code is not trusted
because any single benchmark happens to pass; it is trusted because the operator,
solvers, output writer, and public workflows are all exercised at multiple levels.

Validation philosophy
---------------------

The validation stack is organized from local to global:

1. **Unit tests** for grids, geometry, collisions, solver heuristics, and CLI behavior.
2. **Regression tests** for fixed branches and edge cases.
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
standalone `dkx` implementation reproduces trusted neoclassical physics on the
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
- ``dkx/validation/artifacts.py``

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
  archived rows still need production-resolution reruns.

The corresponding tests are ``tests/test_validation_artifacts.py`` and
``tests/test_generate_validation_dashboard.py``. The high-collisionality proxy
and Simakov-Helander readiness audit are package-level artifact tests, not
stable example-generator tests. The frozen Fortran-suite benchmark figure is protected by
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

For release documentation, the important point is simple:

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
  VMEC getting-started example with ``DKX_OFFLINE=1`` so CI proves that
  public release data is complete and usable without accidental network access,
- ``.github/workflows/docs.yml`` builds the Sphinx documentation,
- ``.github/workflows/publish-pypi.yml`` handles packaging/release publication.

Release-hosted data gates
^^^^^^^^^^^^^^^^^^^^^^^^^

Large public equilibrium fixtures are intentionally not tracked in git and are
not included in wheels. The CI contract for those files is:

- fetch the checksum-pinned ``dkx-data-v1`` release archive with
  ``python -m dkx.validation.data_fetch --quiet``;
- verify every manifest entry exists in the configured cache;
- rerun the public VMEC output path in offline mode;
- keep ``tests/test_data_fetch.py`` as the unit gate for manifest structure,
  checksum extraction, unknown-basename handling, and missing-cache offline
  failure behavior.

This prevents a common packaging regression: a repository can pass pure unit
tests while the release examples fail because a moved equilibrium file is
missing, renamed, or no longer checksum-compatible with the embedded manifest.

Coverage audits use the full test suite with package instrumentation:

.. code-block:: bash

   pytest -q --cov=dkx --cov-report=term --cov-report=xml

The exact collected-test count changes as targeted regression tests are added,
so release notes and the refactor plan cite dated local/CI artifacts rather
than hard-code a permanent collected-test count here. The active refactor lane
records about ``91%`` measured package coverage, with ``95%`` meaningful package
coverage as the research-grade target and CI wall time kept below ten minutes.

The coverage gap is concentrated in large, risk-bearing owners rather than in
trivial helper functions: high-level solver orchestration, output-format and
cache branches, profile-response setup/solve finalization, sparse-factor
admission paths, and geometry/loading edge cases. Direct tests should be added
at those seams first. The preferred oracles are the same invariants used in the
SFINCS technical documentation and the 2014 SFINCS paper: periodic/spectral
differentiation exactness, finite-difference order conditions,
Boozer-coordinate field-component relations, VMEC half-mesh finite-difference
conventions, conservation/nullspace identities for collision operators,
Onsager/positive-semidefinite transport-matrix checks, and deterministic
cache/solver-policy behavior on bounded inputs.

Coverage tests must still be scientific tests. A new branch test is acceptable
when it verifies residual admission, output-key completeness, normalization,
fixture checksum handling, solver-policy invariants, or a recorded bug boundary. A
test that only calls a function to cover a line is not acceptable.

The required CI jobs are also part of the test contract. Coverage shards,
coverage-report generation, example smoke tests, release-data smoke tests, and
optional ecosystem gates are each capped at ten minutes; the final required-job
aggregator is capped at five minutes. ``tests/test_benchmark_doc_claims.py``
parses the workflow and fails if these required-job budgets drift upward.

QI device artifacts are route-level evidence, not production claims unless the
same run satisfies the documented residual, output, runtime, and provenance
gates. A residual-improving device run is not true-device-QI promotion until
those gates pass.

The documentation build is part of the release discipline, not a separate
afterthought. If a docs change breaks Sphinx or leaves pages internally
inconsistent, it should be treated as a real regression.

Coverage-to-95 plan
-------------------

The ``95%`` target is useful only if it reduces real scientific and operational
bugs. Literature on scientific-software testing repeatedly highlights the oracle
problem, and empirical software-engineering work warns that coverage alone is a
weak proxy for test effectiveness. The project therefore treats coverage as a
gap-finding metric, not as the final quality metric.

The staged path is:

1. **Refactor before raising the floor.** Continue simplifying the canonical
   profile, transport, solver, and output owners into pure policy, residual,
   normalization, output-schema, and preconditioner modules. Each extracted
   module must land with module-level docstrings, source-map documentation,
   direct unit tests, and an orchestration regression so behavior stays
   unchanged.
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
6. **Raise CI thresholds in steps.** The CI fail-under gate is ``80%``. Move the
   gate only after each extraction/test batch makes the denominator meaningful:
   ``80 -> 85 -> 90 -> 95``. Each increase requires full CI, strict docs,
   release-data smoke, and the fast release gates to pass within the target
   wall-time budget.
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
through ``python -m dkx.validation.release write-output-trace`` or the transport-trace helpers,
not through always-on per-phase GPU memory polling. The runtime-drift audit also
prefers the solver's logged ``elapsed_s=...`` value when available, falling back
to subprocess wall time only for archived artifacts that do not record it. The suite
subprocesses leave ``JAX_COMPILATION_CACHE_DIR`` unset unless ``--jax-cache-dir`` is
requested, so runtime drift is not polluted by persistent-cache write cost.
When a run changes solver branches unexpectedly, keep the emitted profiling
marks and preconditioner ladder in the research-branch artifact bundle.
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
- example inputs in ``examples/sfincs_examples``,
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
- ``python -m dkx.validation.release check-gates`` and ``tests/test_release_gate_metadata.py`` add a
  CI-fast release gate over the manifest's ``release_gate`` metadata. Each lane must be
  ``release_ready``, ``regression_scaffold``, ``bounded_proxy``, or
  ``closed_deferred``; no lane may remain ambiguous in the release manifest. The
  release checker also validates
  the record status/kind, the non-empty literature/claim/source/test/gate lists, and
  all listed source, test, script, and artifact paths, including paths on
  ``closed_deferred`` lanes. A deferred lane is therefore allowed to stay out of the
  tagged release claim, but it is not allowed to rot silently.
- ``docs/_static/research_lane_completion_2026_05_12.json`` records the active
  research/performance lanes, evidence artifacts, completion estimates, gates,
  and next actions for the checked research-lane cycle. ``python -m dkx.validation.release check-research-lanes``
  and ``tests/test_research_lane_policy.py`` enforce that those percentages are
  evidence-backed and that active lanes record substantial measured progress
  before their completion estimate is increased. The policy is target-capped:
  if a lane has fewer percentage points remaining than the current push target,
  it must reach its checked target rather than overclaiming beyond it.
- ``python -m dkx.validation.release check-research-lanes`` and
  ``tests/test_research_lane_policy.py`` provide the offline gate for deferred
  QI/device-QI evidence. The gate checks provenance, fail-closed metadata, and
  evidence-backed completion percentages only; it is not a convergence
  certificate. Historical or checked-in research artifacts therefore cannot
  silently overclaim GPU/device status.

The ``E_r`` trajectory-model sweep family is a release-facing validation lane:

- script: ``examples/publication_figures/generate_er_trajectory_sweep.py``
- pinned tokamak-like reference artifact:
  ``examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json``
- pinned tokamak-like reference figure:
  ``docs/_static/figures/paper/dkx_er_trajectory_sweep_tokamak_reference.png``
- pinned stellarator-like fast artifact:
  ``examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json``
- pinned stellarator-like fast figure:
  ``docs/_static/figures/paper/dkx_er_trajectory_sweep_stellarator_fast_reference.png``

This lane is used as:

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

Those writer bugs are unit-tested and fixed. The corrected bounded fast reruns are
kept as branch-level regression scaffolds, and the full LHD/W7-X collisionality
summaries and figures have been regenerated from the fixed script and promoted as
audited validation artifacts.

The audited full artifacts from that lane are checked in:

- summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_summary.json``
- figure:
  ``docs/_static/figures/paper/dkx_fig1_lhd_collisionality.png``
- summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_summary.json``
- figure:
  ``docs/_static/figures/paper/dkx_fig2_w7x_collisionality.png``

The bounded fast artifacts remain checked in for branch-level regression work:

- summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json``
- figure:
  ``docs/_static/figures/paper/dkx_fig1_lhd_collisionality_reaudit_fast.png``
- summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json``
- figure:
  ``docs/_static/figures/paper/dkx_fig2_w7x_collisionality_reaudit_fast.png``

The full artifacts are guarded by direct tests on the seven-point collisionality ladder
and FP/PAS label coverage. The corrected fast artifacts are guarded by lighter direct
tests on the four-point ladder and FP/PAS separation.

The collisionality generator writes structured JSON summaries for all reruns, with
top-level metadata that records the case, fast/full-resolution mode, scan ladder,
source input, and collision-operator labeling. That keeps future release reruns aligned
with the manifest's provenance and acceptance-gate expectations instead of relying on
plots alone.

Mapped x-grid and QI integration smoke gates (retired)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mapped speed-grid research owners and their tests were deleted with the
legacy pipeline (see :doc:`adaptive_speed_grid`). The bounded PAS RHSMode=2
comparison artifacts under ``docs/_static/`` are retained as a historical
record only.

The reduced PAS tokamak artifact compares mapped ``Nx=7`` candidates against an
``Nx=13`` reference with active-DOF reduction. It is useful evidence that the
mapped-grid machinery can run through a real transport-matrix solve, but it is not
evidence for full-FP compatibility, a default-grid replacement, or a
production-resolution speedup.

QI/device-QI solver research is preserved on the ``research/qi-device-hard-seed``
branch. The stable test suite keeps only general solver-policy and output-schema
contracts; it does not require checked QI seed-robustness JSON artifacts, GPU
hard-seed campaign outputs, or QI promotion figures. Before any QI/device-QI
route returns to stable, it must provide compact evidence for strict residual
acceptance, CPU/GPU agreement, output/solver-trace writing, and runtime/memory
gates at the documented production grid.

The high-collisionality Simakov-Helander lane has a bounded normalization audit:

- artifact:
  ``examples/publication_figures/artifacts/dkx_simakov_helander_limit_audit_summary.json``
- focused test: ``tests/test_validation_artifacts.py``

This audit checks that the checked-in geometry output fields are sufficient for the
Appendix-B comparison and that ``FSABHat2`` is reproduced from ``BHat`` and ``DHat``.
It intentionally does not close the full analytic-limit reproduction, because the
current audited collisionality scans stop near ``nu'=10``.

The deferred Simakov-Helander panel-data scaffold is also executable in
``dkx.validation.artifacts`` and guarded by
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
The panel metadata also carries a ``publication_figure`` block so downstream
plotting code can distinguish a checked-in converged artifact from a proxy or
deferred scaffold without inferring that status from the title text.

The executable high-``nu`` run-plan artifact is also gated as a run plan, not as
evidence: its summary records ``run_plan_only_not_completed_validation``,
``commands_require_residual_gates``, and ``ready_for_literature_claim=false``.
The generator is preserved outside the stable core; the stable branch keeps the
checked artifact and the validation gates that keep the lane closed until
converged scan artifacts exist.

The W7-X high-``nu`` preconditioner performance figure is intentionally narrower
than a physics-validation lane. Its summary supports a single-point performance
claim only when the factor-reuse route is residual-clean, faster than no-reuse,
uses fewer sparse factorizations, matches the no-reuse residual series, and the
failed bounded Krylov route is explicitly rejected. The same metadata marks
``ready_for_physics_validation_claim=false`` so this performance artifact cannot be
mistaken for a closed W7-X or Simakov-Helander physics validation.

The W7-X ambipolar literature lane is kept as a deferred artifact gate, not as a
stable long-run figure generator. The retained package-level validation is
``dkx.validation.artifacts.build_w7x_ambipolar_root_provenance_panel`` with
focused coverage in ``tests/test_validation_figures.py`` and the core ambipolar
scan/solve tests. The gate records explicit ``deferred_reasons``, provenance
completeness scores, finite-root checks, current-bracket checks, and matching
checked-artifact status. A W7-X ambipolar figure remains deferred until the
numerical root gates pass and the W7-X provenance/source artifact is complete
and checked in.

Further reading
---------------

For the current benchmark/performance state, see :doc:`performance` and
:doc:`parallelism`. For the external validation story, see :doc:`fortran_comparison`.
For the manuscript-facing literature/figure lanes, see :doc:`validation_matrix`.

Optional ecosystem research lanes
---------------------------------

External JAX ecosystem libraries are not production dependencies. Lineax,
Equinox-wrapper, and JAXopt benchmark drivers are preserved as research-lane
material until they satisfy the accuracy, runtime, memory, differentiability, and
dependency-policy gates. Stable tests instead exercise the retained in-tree
paths directly: implicit solve gradients, full-system residual JVPs,
sensitivity checks, optimization proxies, and VMEC/Boozer geometry proxy
gradients.
