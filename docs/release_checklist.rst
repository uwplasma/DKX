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

The authoritative release-facing artifacts for this state are:

- ``tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas``

What should still be stated carefully:

- the CLI defaults are explicit and performance-oriented, not differentiable by default,
- differentiable solve paths are available from Python when requested,
- and remaining work is concentrated on runtime and memory optimization of the heaviest PAS and geometry-rich cases.

Before shipping a release, make sure `README.md`, `docs/fortran_comparison.rst`, and
the performance/parallelism pages accurately reflect the current state of the code.

Local validation (recommended)
------------------------------

From the repository root:

.. code-block:: bash

   pytest -q
   sphinx-build -W -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*

CI/CD also enforces this through:

- ``.github/workflows/ci.yml``
- ``.github/workflows/docs.yml``
- ``.github/workflows/publish-pypi.yml``

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
and ``25 x 1 x 4 x 100`` tokamak floor, and does not include downstream project
decks.

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
