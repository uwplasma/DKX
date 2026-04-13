sfincs_jax
==========

`sfincs_jax` is a production neoclassical transport code for radially local
drift-kinetic calculations in stellarator and tokamak geometry. It combines
high-fidelity kinetic models, CPU/GPU execution, matrix-free numerics, and optional
differentiable solve paths in one codebase.

Current release snapshot
------------------------

On the current ``main`` branch:

- the full audited 39-case example suite runs cleanly on CPU and GPU,
- the default CLI and ``write-output`` path are validated across the release-facing scope with no practical or strict mismatches,
- the Python API can switch to differentiable solve paths when end-to-end sensitivities are needed,
- and the remaining open work is performance and memory tuning on the heaviest cases, not correctness of the documented workflows.

What this documentation covers
------------------------------

This manual is organized around the actual user and developer workflows:

- :doc:`installation`, :doc:`usage`, :doc:`examples`
- :doc:`physics_models`, :doc:`system_equations`, :doc:`geometry`
- :doc:`method`, :doc:`numerics`, :doc:`source_map`
- :doc:`inputs`, :doc:`outputs`, :doc:`applications`
- :doc:`parallelism`, :doc:`performance`, :doc:`testing`
- :doc:`fortran_comparison` and :doc:`references`

.. figure:: _static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png
   :alt: Relative L11 difference and runtime comparison across four monoenergetic cases.
   :align: center
   :width: 90%

   Representative transport validation and runtime snapshot for four monoenergetic
   cases. ``sfincs_jax`` runtime excludes first-time compilation. Reproduce with
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
   applications
   examples
   usage
   inputs
   outputs
   normalizations
   geometry
   method
   numerics
   source_map
   theory_from_upstream
   physics_models
   physics_reference
   system_equations
   parallelism
   performance
   performance_techniques
   testing
   paper_figures
   upstream_docs
   fortran_examples
   utils
   api
   fortran_comparison
   references
   contributing
   release_checklist
