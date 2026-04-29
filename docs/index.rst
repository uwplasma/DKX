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

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: Runtime and peak-memory comparison for SFINCS Fortran v3 and sfincs_jax cold/warm CPU/GPU.
   :align: center
   :width: 90%

   Release benchmark for every audited example-suite case. Panel A compares
   wall-clock runtime and Panel B compares peak resident memory for SFINCS
   Fortran v3, ``sfincs_jax`` CPU cold/warm, and ``sfincs_jax`` GPU cold/warm.
   Cases are ordered by best warm ``sfincs_jax`` speedup over the Fortran v3
   runtime. Reproduce with
   ``examples/publication_figures/generate_fortran_suite_benchmark_summary.py``.

.. figure:: _static/figures/paper/sfincs_jax_publication_validation_dashboard.png
   :alt: Literature-anchored validation dashboard for sfincs_jax.
   :align: center
   :width: 90%

   Publication-facing validation dashboard from checked-in collisionality and
   electric-field sweep artifacts. Reproduce with
   ``examples/publication_figures/generate_validation_dashboard.py``.

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
   validation_matrix
   paper_figures
   upstream_docs
   fortran_examples
   utils
   api
   fortran_comparison
   references
   contributing
   release_notes
   release_checklist
