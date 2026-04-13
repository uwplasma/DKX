Release checklist
=================

This page is intended for maintainers preparing a tagged release (PyPI + Read the Docs).

What this project can and cannot claim
--------------------------------------

On the current ``main`` branch, `sfincs_jax` can claim:

- full CPU and GPU parity for the vendored 39-case example suite, including
  ``examples/additional_examples/input.namelist``,
- no ``jax_error`` or ``max_attempts`` in the current release-facing suite artifacts,
- and matching ``sfincsOutput.h5`` datasets and required terminal-output signals for the supported examples.

The authoritative release-facing artifacts for this state are:

- ``tests/scaled_example_suite_fast_cpu_full_v7_refresh``
- ``tests/scaled_example_suite_fast_gpu_full_v11_refresh``

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

CI/CD also enforces this through:

- ``.github/workflows/ci.yml``
- ``.github/workflows/docs.yml``
- ``.github/workflows/publish.yml``

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
     --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
     --out-root tests/scaled_example_suite_fast_cpu_full_v7_refresh \
     --scale-factor 1.0 \
     --runtime-target-basis fortran \
     --fortran-min-runtime-s 1.0 \
     --fortran-max-runtime-s 20.0 \
     --runtime-adjustment-iters 3

Sync or regenerate the matching frozen-reference GPU lane against that CPU root before updating
release-facing README or docs claims.

Packaging sanity check
----------------------

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
