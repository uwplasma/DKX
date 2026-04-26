:orphan:

Validation against reference implementations
============================================

`sfincs_jax` validates outputs and solver behavior against a mature Fortran SFINCS implementation as a
reference implementation.

On the current ``main`` branch, the release-facing comparison is the full vendored
example-suite audit:

- CPU: ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106``
- GPU: ``tests/scaled_example_suite_release_gpu_2026-04-25_v106``

Those artifacts currently report:

- ``39/39 parity_ok`` on CPU,
- ``39/39 parity_ok`` on GPU,
- no strict mismatches,
- no ``jax_error``,
- no ``max_attempts``.

The frozen reports also generate a publication-facing benchmark dashboard:

.. code-block:: bash

   python examples/publication_figures/generate_fortran_suite_benchmark_summary.py

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: sfincs_jax frozen CPU/GPU suite benchmark against SFINCS Fortran v3
   :width: 92%

   Release benchmark summary generated from the profiled CPU/GPU suite reports. The
   current artifacts have median JAX/Fortran wall-clock ratios of about ``0.035x`` on
   CPU and ``0.058x`` on GPU for the audited suite, while median maximum-RSS ratios are
   about ``4.92x`` on CPU and ``9.20x`` on GPU because JAX/XLA keeps compiled kernels
   and device buffers resident. The top runtime and memory cases are recorded in
   ``examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json``.

Use :doc:`parity` for the scope map and comparison policy, :doc:`performance` for CPU/GPU
runtime and memory context, and :doc:`fortran_examples` for the exact-input frozen-fixture audit.
