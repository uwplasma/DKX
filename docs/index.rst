sfincs_jax
==========

`sfincs_jax` is a JAX implementation of **SFINCS Fortran v3** focused on
output compatibility, matrix-free performance, and differentiability.

Current release snapshot
------------------------

On the current ``main`` branch:

- the full vendored 39-case example suite runs cleanly on CPU and GPU,
- the default CLI and ``write-output`` path match Fortran v3 outputs with no practical or strict mismatches,
- the Python API can switch to differentiable solve paths when end-to-end sensitivities are needed,
- and the remaining open work is performance and memory tuning on the heaviest PAS and geometry-rich cases, not correctness.

.. figure:: _static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png
   :alt: Relative L11 difference and runtime comparison across four monoenergetic cases.
   :align: center
   :width: 90%

   Relative ``ΔL11`` (``(JAX − Fortran) / Fortran``) and runtime comparison for
   four monoenergetic fixtures. ``sfincs_jax`` runtime excludes JIT compilation
   (warm-up not timed). Reproduce with
   ``examples/performance/benchmark_transport_l11_vs_fortran.py``.

.. figure:: _static/figures/transport_compile_runtime_cache_2x2.png
   :alt: Compile/runtime split with the persistent JAX cache across four reference cases.
   :align: center
   :width: 90%

   Compile-time versus warm steady-state runtime for representative transport cases.
   Reproduce with ``examples/performance/profile_transport_compile_runtime_cache.py``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   normalizations
   method
   physics_models
   physics_reference
   paper_figures
   system_equations
   parallelism
   performance_techniques
   usage
   inputs
   outputs
   performance
   upstream_docs
   fortran_examples
   examples
   utils
   api
   fortran_comparison
   references
   contributing
   release_checklist
