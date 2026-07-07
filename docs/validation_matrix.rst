Validation Matrix
=================

This page tracks the publication-facing validation lanes for ``sfincs_jax``. The goal
is to connect each physics claim or benchmark figure to:

- a literature anchor,
- the script or workflow that generates it,
- the expected output artifact,
- and the claim status recorded in the release manifest.

Machine-readable manifest
-------------------------

The corresponding machine-readable manifest lives in:

- ``examples/publication_figures/validation_manifest.json``

That file is the stable spine for:

- future manuscript figure generation,
- reproducible benchmark reruns,
- and test/benchmark dashboards that distinguish implemented release lanes from
  deferred post-release research lanes.

Each manifest lane carries explicit research gates:

- ``source_code``: the implementation files that define the lane,
- ``tests``: the tests that protect the lane or its scaffold,
- ``acceptance_gates``: the concrete criteria required before the lane can support a
  manuscript or release claim.
- ``release_gate``: the release-facing claim status, evidence level, nonblocking
  release decision, and promotion gate for the lane.

The schema is enforced by ``tests/test_validation_manifest_schema.py``. Implemented
release lanes must point to existing scripts, artifacts, source files, and tests.
Deferred post-release lanes are closed for the tagged release but retain literature
anchors, implementation targets, tests, and acceptance criteria so follow-up research
work is not lost. ``python -m sfincs_jax.validation.release check-gates`` applies the same path
hygiene to deferred lanes: listed source files, tests, scripts, and artifacts must
exist even when the claim status is ``closed_deferred``.

Release claim gate metadata
---------------------------

Every manifest record has a ``release_gate`` block checked by
``python -m sfincs_jax.validation.release check-gates`` and ``tests/test_release_gate_metadata.py``.
The allowed ``claim_status`` values are:

- ``release_ready``: checked-in artifacts support the documented release-scope
  claim, and the listed tests are the fast gate for that claim.
- ``regression_scaffold``: checked-in bounded artifacts are useful for CI,
  branch validation, or manuscript layout, but a broader/full-resolution claim is
  intentionally not being made.
- ``bounded_proxy``: checked-in artifacts support a narrower proxy or
  normalization claim, while the corresponding full literature reproduction stays
  closed until its promotion gate is met.
- ``closed_deferred``: the lane is explicitly closed for the tagged release as
  post-release or nightly research work.

No manifest lane may set ``blocks_current_release=true`` unless the release
process intentionally stops on that lane. A lane that is not ready must
therefore be either absent from the release manifest or recorded as
``closed_deferred`` with a concrete reason and promotion gate. This prevents
scaffold scripts, run plans, or proxy figures from being mistaken for closed
publication evidence.

Release decision
----------------

The release is shippable only for the documented release-ready and bounded-proxy
claims. Production-resolution QI CPU/GPU seed ladders, true differentiable
device-QI closure, and single-case multi-device strong scaling are not release
blockers because they are explicitly scoped as bounded or deferred research
lanes. They should be promoted only after checked artifacts satisfy the listed
residual, output, trace, parity, and scaling gates.

Implemented literature reproductions
------------------------------------

These lanes already have scripts and figure artifacts in the repository.

Publication validation dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchor:

- `Landreman et al. 2014 <https://doi.org/10.1063/1.4870077>`_
- `Open PDF mirror <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`_

Script:

- ``examples/publication_figures/generate_validation_dashboard.py``

Artifacts:

- ``examples/publication_figures/artifacts/sfincs_jax_publication_validation_dashboard_summary.json``
- ``docs/_static/figures/paper/sfincs_jax_publication_validation_dashboard.png``
- ``docs/_static/figures/paper/sfincs_jax_publication_validation_dashboard.pdf``

.. figure:: _static/figures/paper/sfincs_jax_publication_validation_dashboard.png
   :alt: Literature-anchored sfincs_jax validation dashboard
   :width: 92%

   Dashboard assembled from checked-in validation artifacts rather than hand-edited
   plot data. The acceptance tests assert that the collisionality scans contain both
   FP and PAS rows on the seven-point grid, that the high-collisionality ``L11``
   separation remains larger than the low-collisionality separation, and that the
   trajectory sweeps retain exact zero-field agreement while resolving finite-field
   model separation.

Fortran v3 CPU/GPU suite benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature and reference anchors:

- `Landreman et al. 2014 <https://doi.org/10.1063/1.4870077>`_
- `Open PDF mirror <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`_
- `SFINCS Fortran repository <https://github.com/landreman/sfincs>`_

Script:

- ``examples/publication_figures/generate_fortran_suite_benchmark_summary.py``

Artifacts:

- ``examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json``
- ``docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png``
- ``docs/_static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.pdf``

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: Frozen CPU and GPU suite benchmark against SFINCS Fortran v3
   :width: 92%

   Cross-code release benchmark generated from frozen CPU/GPU suite reports. The
   plotted bars show wall-clock runtime and active solver memory for SFINCS
   Fortran v3, ``sfincs_jax`` CPU cold/warm, and ``sfincs_jax`` GPU cold/warm
   across the reference-runtime-window rows whose Fortran v3 reference runtime
   is at least ``10 s``. The summary JSON records which frozen rows are excluded
   from public performance claims until production-resolution reruns exist. JAX active memory subtracts the fixed Python/JAX/XLA runtime
   baseline using profiler RSS deltas while preserving full process RSS in the
   JSON audit fields. Cases are ordered by best warm ``sfincs_jax`` speedup over the
   Fortran v3 runtime. The acceptance tests require all 39 audited cases to remain
   ``parity_ok`` on both backends, with zero strict mismatches and no
   ``jax_error`` or ``max_attempts`` failures. Absolute runtime, memory, ratios,
   top offenders, warm timing-source counts, and the excluded short-reference
   rows are recomputed from the checked-in reports and stored in the JSON summary
   for manuscript tables and regression triage. The excluded short-reference
   rows remain CI parity/smoke checks until rerun at production-comparison
   resolution.

SFINCS 2014 collisionality figures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchor:

- [Landreman et al. 2014](https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf)

Scripts:

- ``examples/publication_figures/generate_sfincs_paper_figs.py --case lhd``
- ``examples/publication_figures/generate_sfincs_paper_figs.py --case w7x``

Artifacts:

- ``docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png``
- ``docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality.png``
- ``docs/_static/figures/paper/sfincs_jax_fig3_simakov_helander.png``

The standard LHD and W7-X collisionality figures are generated from the
corrected scan-input writer and recorded as audited full-resolution validation
artifacts. They are regression and manuscript-scaffold figures, not a claim that
every plotted point reproduces the original paper image digit-for-digit.

Status note:

- the scan writer in ``generate_sfincs_paper_figs.py`` rejects duplicate
  namelist assignments that would otherwise override the intended
  ``collisionOperator`` and fast-resolution settings
- the generator emits machine-readable collisionality summaries with top-level
  metadata and sorted rows so full-resolution reruns have pinned provenance
  instead of relying only on figure files
- the checked-in full LHD and W7-X summaries each contain 14 rows: both FP and PAS
  labels on a seven-point collisionality ladder
- corrected bounded fast reruns are retained as branch-level regression scaffolds, but
  the main LHD/W7-X figure family points at the full audited artifacts

Audited full artifacts:

- full LHD summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_summary.json``
- full LHD figure:
  ``docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality.png``
- full W7-X summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_summary.json``
- full W7-X figure:
  ``docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality.png``

Corrected bounded branch artifacts:

- bounded corrected LHD summary:
  ``examples/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json``
- bounded corrected LHD figure:
  ``docs/_static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png``

.. figure:: _static/figures/paper/sfincs_jax_fig1_lhd_collisionality_reaudit_fast.png
   :alt: Corrected bounded LHD collisionality scan for sfincs_jax
   :width: 85%

   Corrected bounded LHD collisionality rerun with the guarded scan-input writer.
   This artifact resolves the expected FP/PAS separation and is backed by direct
   JSON-based assertions, but it is a bounded fast branch lane rather than the
   final audited paper figure.

- bounded corrected W7-X summary:
  ``examples/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json``
- bounded corrected W7-X figure:
  ``docs/_static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png``

.. figure:: _static/figures/paper/sfincs_jax_fig2_w7x_collisionality_reaudit_fast.png
   :alt: Corrected bounded W7-X collisionality scan for sfincs_jax
   :width: 85%

   Corrected bounded W7-X collisionality rerun after fixing the scan-input writer.
   This lane also resolves clean FP/PAS separation and is light enough for branch-level
   validation, but it remains a bounded fast artifact rather than the final audited
   paper figure.

Autodiff / sensitivity validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `Paul et al. 2019 adjoint optimization <https://arxiv.org/abs/1904.06430>`_
- `APS adjoint optimization abstract <https://meetings-archive.aps.org/dpp/2018/bp11/36/>`_

Current script:

- ``examples/publication_figures/generate_autodiff_sensitivity_validation.py``

Current artifacts:

- ``examples/publication_figures/artifacts/sfincs_jax_autodiff_sensitivity_validation_summary.json``
- ``docs/_static/figures/paper/sfincs_jax_autodiff_gradient_check.png``
- ``docs/_static/figures/paper/sfincs_jax_autodiff_gradient_check.pdf``
- ``docs/_static/figures/paper/sfincs_jax_autodiff_sensitivity_map.png``
- ``docs/_static/figures/paper/sfincs_jax_autodiff_sensitivity_map.pdf``

Fortran-v3 RHSMode 4/5 source-contract gates:

- ``sfincs_jax.sensitivity.validate_fortran_v3_adjoint_sensitivity_constraints``
  mirrors the source-code restrictions from ``validateInput.F90`` for adjoint
  sensitivity decks.
- ``sfincs_jax.sensitivity.fortran_v3_adjoint_sensitivity_output_fields`` pins
  the sensitivity HDF5 field names emitted by ``writeHDF5Output.F90`` before
  the numerical Fortran replay fixtures are promoted.
- ``sfincs_jax.sensitivity.fortran_v3_adjoint_sensitivity_output_ranks`` and
  ``validate_fortran_v3_adjoint_sensitivity_output_surface`` validate the
  required RHSMode=4/5 field names and tensor ranks against either HDF5-like
  arrays or lightweight JSON summaries.
- ``tests/test_sensitivity.py`` checks valid and invalid RHSMode 4/5 decks,
  including the Fortran source-code gate that writes ``dParallelFlowdLambda``
  from ``adjointParticleFluxOption`` or ``debugAdjoint``.
- ``tests/fixtures/fortran_v3_reference_fixture.json`` contains compact
  RHSMode=4/5 reference summaries and embedded namelist text. The checked
  W7-X-like analytic decks pin radial-current, heat-flux, total-heat-flux,
  parallel-flow, bootstrap, and RHSMode=5 ``dPhidPsidLambda`` sensitivity
  outputs from SFINCS Fortran v3 without committing generated HDF5 files,
  including
  ``dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`` and
  ``dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`` and
  ``dBootstrapdLambda = sum_s Z_s dParallelFlowdLambda_s``.
- ``small_rhsmode4_debug_summary_2026-06-25.json`` records a bounded
  debug-adjoint finite-difference run. The regression validates every debug
  field name/rank, selected analytic/finite-difference values, finite percent
  errors below the checked tolerance, and the Fortran NaN mask for unfilled
  lambda/mode entries.

.. figure:: _static/figures/paper/sfincs_jax_autodiff_gradient_check.png
   :alt: Autodiff gradient validation for sfincs_jax
   :width: 92%

   Bounded manuscript-grade autodiff validation. The checked-in summary records
   centered finite-difference comparisons, primal residuals, and adjoint residuals
   for custom-linear-solve gradients. The SFINCS full-system panel uses a pinned
   tiny PAS fixture and validates the implicit-differentiation path without changing
   production solver defaults.

.. figure:: _static/figures/paper/sfincs_jax_autodiff_sensitivity_map.png
   :alt: Boozer harmonic sensitivity maps for sfincs_jax
   :width: 92%

   Differentiable ``geometryScheme=4`` Boozer-harmonic sensitivity maps. This
   artifact validates the public analytic-Boozer geometry path used by examples and
   optimization scaffolds; it does not claim full VMEC-boundary optimization.

Bounded integration lanes
-------------------------

These lanes are useful for integration review, but they are not current-release
publication claims unless and until they are added to
``examples/publication_figures/validation_manifest.json`` with explicit
``release_gate`` metadata.

Open lane board
^^^^^^^^^^^^^^^

- QI/device-QI solver research: QI seed-robustness, hard-seed GPU
  campaigns, and device-QI operator-reuse promotion evidence are preserved on
  the ``research/qi-device-hard-seed`` branch. They are not release-facing
  validation artifacts in the stable core. Any future QI/device-QI promotion
  must restore or regenerate compact artifacts from the candidate branch and
  pass residual, output, runtime, memory, CPU/GPU parity, solver-trace, and
  documentation gates before appearing in this matrix.
- PAS memory/runtime: guarded ``tzfft`` and weak-PAS fail-fast routes are bounded
  diagnostics. The byte-budgeted geometry4 and HSX real-solve probes are
  residual-clean and solver-path stable, but they are not promoted because they
  regress runtime, memory, or both against the checked baselines. Promotion still
  requires residual-clean CPU/GPU evidence with no parity loss and a measured
  runtime or memory win on geometry-rich PAS floors.
- Single-case scaling: transport-worker case/RHS throughput is separately gated,
  but single-case multi-device strong scaling remains experimental until a warm,
  compile-amortized, device-covered artifact shows a real speedup.
- Coverage/refactor: policy seams and solver helpers have focused tests, but the
  package-wide ``95%`` target still requires more owner-module tests for
  profile solves, transport solves, operator assembly, output writing, and a
  JAX-safe coverage environment.
- VMEC/Boozer workflow: validation checks cover workflow provenance, optional
  ecosystem gates, and proxy-gradient consistency. Full VMEC-boundary-to-SFINCS
  kinetic transport gradients remain deferred.
- Deferred validations: W7-X ambipolar validation, high-``nu`` analytic-limit
  extension, broader MONKES/KNOSOS overlap, production-resolution QI ladders, and
  large geometry-rich PAS claims remain deferred until checked-in numerically
  gated artifacts and release-gate metadata exist. Production-resolution QI
  ladders should not launch until the GPU hard-seed gate writes output through a
  true device route.

Mapped x-grid PAS transport evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current scripts and source anchors:

- ``sfincs_jax/discretization/adaptive_maps.py``
- ``sfincs_jax/workflows/mapped_xgrid.py``
- opt-in ``xGridScheme = 50`` construction in ``sfincs_jax/discretization/v3.py``

Current bounded artifacts:

- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.json``
- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.csv``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.json``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.csv``

Current tests:

- ``tests/test_adaptive_maps.py``
- ``tests/test_mapped_xgrid_objectives.py``
- ``tests/test_mapped_xgrid_v3.py``
- ``tests/test_mapped_xgrid_transport_evidence.py``

Scope and status:

- The tiny artifact is a smoke comparison against a small RHSMode=2 PAS fixture.
- The reduced PAS tokamak artifact compares mapped ``Nx=7`` candidates against an
  ``Nx=13`` reference and records residuals, active-DOF counts, elapsed time,
  moment-objective diagnostics, and transport-matrix error.
- The best reduced PAS tokamak candidate by transport error is a bounded evidence
  point for the opt-in mapped-grid machinery, not a claim that mapped grids should
  replace default SFINCS-v3-compatible grids.
- Full-FP mapped-grid compatibility remains open because the current full-FP
  collision precompute path still has assumptions that are not yet mapped-grid
  compatible.

Promotion gates:

- add the lane to the manifest with ``claim_status`` no stronger than
  ``bounded_proxy`` until production-resolution evidence exists,
- compare against higher-resolution default-grid references, not only
  same-resolution smoke solves,
- demonstrate residual-clean CPU/GPU behavior on at least one representative PAS
  transport case,
- and keep default ``xGridScheme`` behavior unchanged unless full-suite parity and
  runtime/memory gates justify promotion.

QI/device-QI research boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QI seed-robustness scripts, hard-seed campaign artifacts, and device-QI
promotion tests are preserved on the ``research/qi-device-hard-seed`` branch.
The stable core intentionally keeps only general solver-policy and output-schema
contracts. It does not ship QI seed-robustness JSON artifacts, QI promotion
figures, or QI-only example inputs as release evidence.

Promotion gates for any future QI/device-QI return to stable are:

- regenerate compact artifacts from the candidate branch,
- pass strict true-residual and output-write gates on CPU and GPU,
- record solver traces, runtime, and peak-memory budgets,
- compare supported observables against SFINCS Fortran v3 where the models
  overlap,
- document the differentiability scope, and
- add only the minimal stable source/tests/docs needed for the admitted default.

Solver-path policy refactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current source and tests:

- ``sfincs_jax/solvers/path_policy.py``
- ``tests/test_solver_path_policy.py``

Scope and status:

- The refactor centralizes directly testable policy decisions for solver JIT
  eligibility, preconditioner dtype selection, PAS geometry-4 FP32 gating,
  residual-rescue slack, DKES GMRES budget preservation, sparse-PC defaults,
  structural-tolerance parsing, and resource-exhaustion classification.
- This is a maintainability and reproducibility gate for solver-path selection.
  It does not by itself support a new performance or physics claim.

Promotion gates:

- keep policy tests green alongside the driver-wrapper tests,
- verify no solver-path branch change is promoted without residual-clean and
  parity-clean artifacts,
- and summarize solver-path provenance in release artifacts before using a new
  branch as a documented default.

Closed post-release research lanes
----------------------------------

The following lanes are not release blockers. They are closed in the manifest as
``deferred_post_release`` with explicit criteria for reopening them in a later
research/nightly cycle.

1. Electric-field sweeps
^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- [Landreman et al. 2014](https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf)

Publication target:

- one tokamak-like case,
- one stellarator case,
- fluxes, flows, and bootstrap current versus normalized radial electric field,
- clear comparison of partial, DKES-like, and full-trajectory models.

Current scaffold:

- ``examples/publication_figures/generate_er_trajectory_sweep.py``

This script already implements the correct upstream trajectory-model switches and
produces JSON summaries plus 2x2 publication-style figures.

Current fixed artifacts:

- audited tokamak-like reference summary:
  ``examples/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json``
- audited tokamak-like reference figure:
  ``docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png``
- bounded stellarator-like fast summary:
  ``examples/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json``
- bounded stellarator-like fast figure:
  ``docs/_static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png``

.. figure:: _static/figures/paper/sfincs_jax_er_trajectory_sweep_tokamak_reference.png
   :alt: Tokamak-like electric-field trajectory-model sweep for sfincs_jax
   :width: 85%

   Fixed tokamak-like ``E_r`` sweep across DKES, partial, and full trajectory
   models. This lane is pinned to checked-in JSON and figure artifacts, and it
   is backed by direct numerical assertions on zero-field agreement and
   finite-field model separation.

.. figure:: _static/figures/paper/sfincs_jax_er_trajectory_sweep_stellarator_fast_reference.png
   :alt: Stellarator-like electric-field trajectory-model sweep for sfincs_jax
   :width: 85%

   Fixed stellarator-like fast branch scaffold across DKES, partial, and full
   trajectory models. This is intentionally a bounded branch-validation lane:
   it resolves the expected model separation on the selected input, but the
   full-resolution stellarator sweep remains a heavier validation target.

Validation goal:

- verify small-field agreement and large-field separation behavior,
- make the ordering and crossover behavior explicit in both assertions and figures,
- promote the stellarator-like branch scaffold to a full-resolution audited lane only
  after the runtime/cost tradeoff is acceptable for the release/nightly workflow.

2. High-collisionality proxy after collisionality audit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- [Landreman et al. 2014](https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf)

Closed branch evidence:

- bounded LHD and W7-X fast reruns resolve FP/PAS separation on the same
  four-point ``\\nu'`` ladders without label collapse in the stored outputs
- audited full LHD and W7-X collisionality summaries resolve both FP and PAS labels
  on seven-point ``\\nu'`` ladders
- a checked-in trend proxy records high-collisionality tail slopes from those
  corrected artifacts:
  ``examples/publication_figures/artifacts/sfincs_jax_high_collisionality_trend_proxy_summary.json``
- a checked-in Simakov-Helander normalization audit records the Appendix-B
  geometry ingredients, ``FSABHat2`` recomputation, inverse-``nu`` slope gates, and
  explicit readiness status:
  ``examples/publication_figures/artifacts/sfincs_jax_simakov_helander_limit_audit_summary.json``

.. figure:: _static/figures/paper/sfincs_jax_high_collisionality_trend_proxy.png
   :alt: High-collisionality trend proxy from checked-in collisionality artifacts
   :width: 92%

   Trend proxy for the ``L11`` and ``L12`` tails. The SFINCS 2014 paper states that
   PAS ``L11``/``L12`` scale like ``+nu`` at high collisionality, while
   momentum-conserving FP/model-operator results should approach inverse-``nu``
   scaling in the ``nu' >> 1`` limit. The checked-in LHD artifact satisfies the
   loose inverse-tail proxy, but the W7-X artifact does not yet. The stricter
   Simakov-Helander audit therefore keeps both geometries deferred until wider
   high-``nu`` scans are pinned, so this figure is kept as an implemented trend gate
   rather than the final analytic-limit reproduction.

.. figure:: _static/figures/paper/sfincs_jax_simakov_helander_limit_audit.png
   :alt: Simakov-Helander high-collisionality readiness audit
   :width: 92%

   Normalization and readiness audit for the full Simakov-Helander lane. The audit
   confirms that checked-in ``sfincsOutput.h5`` files contain the geometry quantities
   needed for an Appendix-B comparison, but it keeps the full analytic-limit
   reproduction closed because the current full collisionality summaries stop near
   ``nu'=10`` rather than a wider ``nu' >> 1`` range. The JSON summary also
   carries a recommended logarithmic high-``nu'`` extension grid for each case,
   ending near ``nu'`` of ``100``, so the next heavy run is pinned and reviewable.

Post-release acceptance criteria:

- keep machine-readable summary artifacts for each full scan,
- keep the Simakov-Helander audit artifact in CI as the parent gate for future
  high-collisionality scan work,
- use
  ``examples/publication_figures/artifacts/sfincs_jax_simakov_helander_high_nu_run_plan.json``
  as the executable high-``nu'`` extension plan; it is generated from the audit
  and pins LHD and W7-X extension commands ending near ``nu'=100``,
- run each plan entry's ``pilot_command`` first; the first LHD FP pilot at
  ``nu'=17.78`` on the office GPU took about ``569 s`` for one transport point,
  so the complete FP/PAS LHD+W7-X extension is a nightly/workstation campaign,
- and only promote the deferred full analytic-limit reproduction after wider
  high-``nu`` LHD and W7-X scans are regenerated and the readiness gate flips true.

The run-plan artifact is explicitly labelled as a deferred executable plan. Its
machine-readable gates record that residual thresholds are wired into every command
and that ``ready_for_literature_claim`` remains false because no completed high-``nu``
scan artifact is present yet. Publication panel summaries use the same convention:
``publication_figure.claim_status`` is ``proxy_or_deferred`` unless the source JSON is
both numerically gated and checked in as a converged artifact.

A separate W7-X high-``nu`` preconditioner/performance figure is available for the
single first FP point, but it is not a physics-validation lane. Its summary gates
only support the bounded claim that sparse-helper factor reuse is residual-clean,
faster than no-reuse, uses fewer sparse factorizations, and rejects the failed
bounded Krylov route. The figure metadata therefore keeps
``ready_for_physics_validation_claim=false``.

3. W7-X ambipolar-field validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- [Pablant et al. 2020 ion-root context](https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf)
- [Pablant et al. 2018 W7-X core radial electric field](https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf)
- [Nature 2021 W7-X neoclassical validation context](https://www.nature.com/articles/s41586-021-03687-w)

Publication target:

- one figure comparing neoclassical ``E_r`` and/or heat-flux trends against the
  published W7-X validation context,
- one table documenting exactly which approximations and reconstructed inputs were used.

Validation goal:

- make any profile reconstruction assumptions explicit,
- use this lane only if the reconstructed input set is scientifically defensible.

Stable artifact gate:

- ``sfincs_jax.validation.artifacts.build_w7x_ambipolar_root_provenance_panel``
- ``examples/publication_figures/provenance/w7x_ambipolar_provenance_template.json``

The stable branch keeps the ambipolar solver API, scan/readback tests, and a
fail-closed provenance panel builder. Long W7-X scan and figure generation is a
publication-audits research workflow until a defensible equilibrium/profile
reconstruction is supplied and the resulting source artifact is checked in.

The deferred panel includes explicit ``acceptance_gates``:

- finite distinct ``E_r``/current scan points,
- finite ambipolar roots,
- radial-current sign bracketing,
- roots inside the scanned ``E_r`` range,
- root consistency with a sign-change bracket,
- a resolved local current slope at the accepted root,
- an ion-root candidate,
- complete equilibrium/profile/discharge/literature provenance,
- checked-in source-artifact status,
- and the combined ``ready_for_literature_claim`` gate.

Without a provenance JSON containing ``equilibrium_source``, ``profile_source``,
``configuration_or_shot``, and ``literature_reference``, generated artifacts remain
``w7x_like_scaffold`` rather than ``w7x_literature_validation``.
Even with complete provenance, generated summaries remain
``w7x_literature_candidate_deferred`` until the matching W7-X summary artifact is
checked in; this prevents an exploratory rerun from being labelled as a closed
literature comparison.
Start from
``examples/publication_figures/provenance/w7x_ambipolar_provenance_template.json``;
it is intentionally incomplete and should be copied/finalized for a specific
equilibrium/profile reconstruction before any literature-facing W7-X claim.

Closure note:

- the stable core keeps the ambipolar solver and provenance/artifact gate tests,
- the checked-in literature artifact and long generator are intentionally absent,
- this lane is classified as ``deferred_post_release`` until a
  defensible W7-X input reconstruction is run and its summary/figure are pinned in
  the repository.

4. MONKES / KNOSOS overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- [MONKES paper](https://arxiv.org/abs/2312.12248)
- [KNOSOS paper](https://arxiv.org/abs/1908.11615)

Publication target:

- coefficient overlap on monoenergetic shared-model subsets,
- low-collisionality trend comparison where the models are not exactly identical.

Validation goal:

- separate exact overlap claims from qualitative trend/ordering claims,
- keep this lane focused on the model subset that is genuinely comparable.

How this page should evolve
---------------------------

Each time a new figure lane is implemented, update both:

- this page,
- and ``examples/publication_figures/validation_manifest.json``.

That keeps the manuscript-facing validation story synchronized with the code structure
and the test/benchmark infrastructure.
