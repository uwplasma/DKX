:orphan:

Fortran Comparison
==================

This project validates outputs and solver behavior against SFINCS Fortran v3 as a
reference implementation.

On the current ``main`` branch, the release-facing comparison is the full vendored
example-suite audit:

- CPU: ``tests/scaled_example_suite_fast_cpu_full_v6_merged``
- GPU: ``tests/scaled_example_suite_fast_gpu_full_v8``

Those artifacts currently report:

- ``39/39 parity_ok`` on CPU,
- ``39/39 parity_ok`` on GPU,
- no strict mismatches,
- no ``jax_error``,
- no ``max_attempts``.

Use :doc:`parity` for the scope map and comparison policy, :doc:`performance` for CPU/GPU
runtime and memory context, and :doc:`fortran_examples` for the exact-input frozen-fixture audit.
