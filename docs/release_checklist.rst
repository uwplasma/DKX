Release checklist
=================

This page is intended for maintainers preparing a tagged release (PyPI + Read the Docs).

What this project can and cannot claim
--------------------------------------

On the current ``main`` branch, `sfincs_jax` can claim:

- full CPU and GPU parity for the vendored 39-case example suite, including
  ``examples/additional_examples/input.namelist``,
- no ``jax_error`` or ``max_attempts`` in the current release-facing suite artifacts,
- matching ``sfincsOutput.h5`` common numeric datasets, zero missing Fortran top-level
  output keys in JAX, and the required terminal-output signals for the supported examples.
- a bounded, metadata-visible non-autodiff host fallback for explicit large-QI
  device-Krylov requests, backed by the checked scale-0.60 CPU hard-seed artifact.
- transport-worker GPU parallelism for independent RHS/case throughput on the
  checked two-GPU benchmark lane.

The authoritative release-facing artifacts for this state are:

- ``tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas``

What should still be stated carefully:

- the CLI defaults are explicit and performance-oriented, not differentiable by default,
- differentiable solve paths are available from Python when requested,
- production-resolution QI CPU/GPU seed ladders remain bounded-proxy evidence,
- true differentiable device-QI closure is deferred research, not a release blocker,
- single-case multi-GPU strong scaling remains experimental and must not be
  presented as the release scaling story,
- and remaining work is concentrated on runtime and memory optimization of the
  heaviest PAS and geometry-rich cases.

Before shipping a release, make sure `README.md`, `docs/fortran_comparison.rst`, and
the performance/parallelism pages accurately reflect the current state of the code.

Local validation (recommended)
------------------------------

From the repository root:

.. code-block:: bash

   pytest -q
   python scripts/check_release_gates.py
   python scripts/check_research_lanes.py
   python scripts/check_qi_device_artifacts.py docs/_static/figures/optimization --min-relevant 1
   sphinx-build -W -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*

CI/CD also enforces this through:

- ``.github/workflows/ci.yml``
- ``.github/workflows/docs.yml``
- ``.github/workflows/publish-pypi.yml``

For a fast claim-scope check without running the whole suite, use:

.. code-block:: bash

   pytest -q tests/test_validation_manifest_schema.py tests/test_release_gate_metadata.py
   pytest -q tests/test_benchmark_doc_claims.py tests/test_generate_fortran_suite_benchmark_summary.py
   python scripts/check_release_gates.py
   python scripts/check_research_lanes.py
   python scripts/check_qi_device_artifacts.py docs/_static/figures/optimization --min-relevant 1

This validates that publication-facing lanes are either implemented for the
documented current claim, kept as bounded scaffolds/proxies, or explicitly closed as
post-release work. No manifest lane may silently remain an open release blocker.
The research-lane check additionally validates that active large-push completion
estimates are tied to checked-in evidence and next actions rather than informal
status text. When a push asks for a larger absolute movement than a lane has
remaining, the lane may pass only by saturating its checked target percentage;
the gate does not allow percentages to exceed the target just to satisfy a
requested point increase.
The benchmark-doc checks make the README/docs runtime and memory claims fail if
they drift from the checked-in CPU/GPU suite reports or benchmark summary JSON.
The QI device-artifact check is a no-solve guard against accidentally promoting
nonconverged GPU/operator-reuse evidence into a production true-device-QI claim.

For the current mapped-grid / QI / solver-path integration branch, also run the
bounded integration checks before promoting any of those lanes into release-facing
metadata:

.. code-block:: bash

   pytest -q \
     tests/test_adaptive_maps.py \
     tests/test_mapped_xgrid_objectives.py \
     tests/test_mapped_xgrid_v3.py \
     tests/test_mapped_xgrid_transport_evidence.py \
     tests/test_run_mapped_xgrid_transport_evidence.py \
     tests/test_run_qi_seed_robustness.py \
     tests/test_solver_path_policy.py

Those tests are not a substitute for full-suite parity or production-resolution
benchmark evidence. They only prove that the opt-in mapped grid, QI seed runner,
and solver-path policy seams are wired and reproducible on bounded inputs.

For a bounded QI smoke rerun, use the default CLI solver policy first:

.. code-block:: bash

   python scripts/run_qi_seed_robustness.py \
     --input examples/additional_examples/input.namelist \
     --out-root tests/qi_seed_robustness_smoke \
     --seeds 0 1 2 \
     --resolution-scale 0.25 \
     --min-ntheta 7 --min-nzeta 11 --min-nx 4 --min-nxi 16 \
     --execute \
     --max-residual-ratio 1 \
     --require-converged \
     --clean

The checked summaries in ``docs/_static/qi_seed_robustness_smoke.json`` and
``docs/_static/qi_seed_robustness_multiseed.json`` record bounded passing
default-CLI seeds. Only claim production QI robustness after production-resolution
CPU/GPU ladders record passing executions and the solver-trace/output checks
needed for the claim.

If claiming mapped x-grid transport evidence, regenerate the bounded artifacts and
review their residuals and transport-matrix errors before updating docs:

.. code-block:: bash

   python scripts/run_mapped_xgrid_transport_evidence.py \
     --case reduced_pas_tokamak_rhsmode2 \
     --json-out docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.json \
     --csv-out docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.csv

Keep the claim scoped to PAS RHSMode=2 bounded evidence unless a broader
production-resolution comparison is checked in and gated.

Smoke-run the examples that do not require optional dependencies:

.. code-block:: bash

   python examples/getting_started/build_grids_and_geometry.py
   python examples/getting_started/apply_collisionless_operator.py
   python examples/getting_started/write_sfincs_output_python.py
   python examples/getting_started/write_sfincs_output_cli.py
   python examples/autodiff/matrix_free_residual_and_jvp.py

Regenerate the exact-input upstream fixture audit if upstream inputs or support levels change:

.. code-block:: bash

   python scripts/generate_fortran_example_audit.py

Release-facing full suite run (vendored upstream inputs):

.. code-block:: bash

   python scripts/run_scaled_example_suite.py \
     --examples-root examples/sfincs_examples \
     --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
     --reference-results-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix \
     --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak \
     --scale-factor 1.0 \
     --runtime-target-basis fortran \
     --fortran-min-runtime-s 0.0 \
     --runtime-adjustment-iters 0 \
     --runtime-baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
     --jax-profile-marks on

Each suite run now writes:

- ``suite_output_key_coverage.json``
- ``suite_output_key_coverage_summary.json``
- and, when ``--runtime-baseline-report`` is provided,
  ``suite_runtime_drift.json`` plus ``suite_runtime_drift_summary.json``.

For release promotion, require:

- ``suite_output_key_coverage_summary.json`` reports ``missing_total = 0``
- and the candidate runtime lane is audited against the previous frozen CPU lane.

Manual audit commands:

.. code-block:: bash

   python scripts/fetch_equilibria.py --quiet
   SFINCS_JAX_OFFLINE=1 python examples/getting_started/write_sfincs_output_vmec.py \
     --out /tmp/sfincs_jax_vmec_release_data_smoke.h5

   python scripts/audit_suite_output_keys.py \
     --suite-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak \
     --fail-on-missing

   python scripts/audit_suite_runtime_drift.py \
     --baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
     --candidate-report tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak/suite_report.json \
     --threshold-ratio 1.25 \
     --min-baseline-runtime-s 1.0

Sync or regenerate the matching GPU lane against that CPU root before updating
release-facing README or docs claims. The current GPU root is
``tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas``.

Bounded artifact refreshes
--------------------------

These commands refresh release-facing plots and tables without launching new
full-suite solves:

.. code-block:: bash

   python examples/publication_figures/generate_fortran_suite_benchmark_summary.py

   python scripts/generate_readme_fast_branch_audit.py \
     --out-root tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak \
     --gpu-out-root tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas \
     --baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
     --min-fortran-runtime-s 10

   python examples/publication_figures/generate_w7x_high_nu_performance.py

   python examples/performance/benchmark_transport_parallel_scaling.py \
     --from-json examples/performance/output/transport_parallel_scaling_gpu.json \
     --out-dir docs/_static/figures/parallel \
     --figure-name transport_parallel_scaling_gpu.png

   python scripts/benchmark_pas_tz_memory_fallback.py \
     --input examples/sfincs_examples/geometryScheme4_2species_PAS_noEr/input.namelist \
     --out tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json \
     --variants hybrid zeta theta \
     --timeout-s 15 \
     --maxiter 4 \
     --restart 8 \
     --tol 1e-6

The manual GitHub workflow ``Production Benchmark Inputs`` should also pass
before a release. It validates that the generated SFINCS_JAX-owned benchmark
manifest still has 39 cases, uses the documented ``25 x 51 x 4 x 100`` 3D floor
and ``33 x 1 x 12 x 140`` tokamak floor, applies the calibrated
``89 x 1 x 24 x 300`` RHSMode=1 PAS/no-``E_r`` tokamak floor, and does not
include downstream project decks.

Packaging sanity check
----------------------

Before tagging:

- update ``pyproject.toml`` and ``sfincs_jax.__version__`` together,
- verify ``pytest -q tests/test_package_metadata.py`` passes,
- ensure the intended tag is exactly ``v<package-version>``,
- ensure GitHub CI and Docs are green on the commit being tagged.

The PyPI workflow validates tag/version consistency and runs ``twine check`` on
the built distributions before publishing.

CI uses an isolated build environment. Locally, if you are in an offline/sandboxed environment,
an isolated build may fail due to missing network access. In that case, use:

.. code-block:: bash

   python -m build --no-isolation

Fixture generation note (Fortran v3)
------------------------------------

Many parity tests rely on **frozen Fortran v3 fixtures** (PETSc binaries and/or `sfincsOutput.h5`).
Generating new fixtures requires a working v3 executable and an MPI/PETSc runtime environment that
can complete `MPI_Init`. Some sandboxed CI environments can block network endpoints and cause the
Fortran executable to fail at startup; generate fixtures on a normal workstation/HPC environment
and commit the resulting reference files under `tests/ref/`.
