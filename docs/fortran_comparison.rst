:orphan:

Validation against reference implementations
============================================

`sfincs_jax` validates outputs and solver behavior against a mature Fortran SFINCS implementation as a
reference implementation.

On the current ``main`` branch, the release-facing comparison is the full vendored
example-suite audit:

- CPU: ``tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix``
- GPU: ``tests/scaled_example_suite_recheck_gpu_frozen_2026-04-23_postruntimefix_mem``

Those artifacts currently report:

- ``39/39 parity_ok`` on CPU,
- ``39/39 parity_ok`` on GPU,
- no strict mismatches,
- no ``jax_error``,
- no ``max_attempts``.

Use :doc:`parity` for the scope map and comparison policy, :doc:`performance` for CPU/GPU
runtime and memory context, and :doc:`fortran_examples` for the exact-input frozen-fixture audit.
