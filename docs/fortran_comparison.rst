:orphan:

Validation against reference implementations
============================================

`sfincs_jax` validates outputs and solver behavior against a mature Fortran SFINCS implementation as a
reference implementation.

On the current ``main`` branch, the release-facing comparison is the full vendored
example-suite audit:

- CPU: ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106``
- GPU: ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28``

Those artifacts currently report:

- ``39/39 parity_ok`` on CPU,
- ``39/39 parity_ok`` on GPU,
- no strict mismatches,
- no ``jax_error``,
- no ``max_attempts``.

The frozen reports also generate a publication-facing runtime and memory
comparison. The plotted rows are restricted to cases whose SFINCS Fortran v3
reference runtime is at least ``10 s``; shorter rows remain CI parity/smoke
checks unless they are rerun at production-comparison resolution.

.. code-block:: bash

   python examples/publication_figures/generate_fortran_suite_benchmark_summary.py

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: sfincs_jax frozen CPU/GPU suite benchmark against SFINCS Fortran v3
   :width: 92%

   Release benchmark generated from the profiled CPU/GPU suite reports. Panel A
   compares wall-clock runtime and Panel B compares active solver memory for the
   production-scale subset, with separate ``sfincs_jax`` cold and warm bars for
   CPU and GPU.
   Cases are ordered by best warm ``sfincs_jax`` speedup over the Fortran v3
   runtime.
   The current artifacts have median cold JAX/Fortran wall-clock ratios of about
   ``0.012x`` on CPU and ``0.021x`` on GPU for the plotted production-scale
   subset. Median process maximum-RSS ratios remain available in the JSON audit
   fields, while the public memory bars use profiler active RSS deltas over the
   fixed Python/JAX/XLA baseline; the median active-memory ratios are about
   ``2.79x`` on CPU and ``3.61x`` on GPU. The top runtime and memory cases are recorded in
   ``examples/publication_figures/artifacts/sfincs_jax_fortran_suite_benchmark_summary.json``.

Use :doc:`parity` for the scope map and comparison policy, :doc:`performance` for CPU/GPU
runtime and memory context, and :doc:`fortran_examples` for the exact-input frozen-fixture audit.
