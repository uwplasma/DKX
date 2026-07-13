sfincs_jax
==========

`sfincs_jax` solves the radially local, linearized drift-kinetic equation on a
flux surface — the same physics as SFINCS Fortran v3 — in pure JAX. One
``input.namelist`` plus one geometry file gives neoclassical particle/heat
fluxes, parallel flows, bootstrap current, and transport matrices for
stellarators and tokamaks, on CPU or GPU, with end-to-end automatic
differentiation for sensitivities and optimization.

Quickstart
----------

.. code-block:: bash

   pip install sfincs_jax

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.run import run_profile

   run = run_profile(Path("input.namelist"), solve_method="auto",
                     out_path=Path("sfincsOutput.h5"))
   print(float(run.moments["particleFlux_vm_psiHat"][0]))
   print(float(run.moments["FSABjHat"]))  # bootstrap current <j.B>

``run_profile`` prints the Fortran-parity console flow, writes
``sfincsOutput.h5``/``.nc`` keyed by the SFINCS output names, and returns the
state vector, solver statistics, and all velocity-space moments in memory. The
CLI equivalent is ``sfincs_jax input.namelist --out sfincsOutput.h5``;
``sfincs_jax --plot sfincsOutput.h5`` builds a PDF diagnostics panel. See
:doc:`installation` for the optional ``solvax`` structured-solver extra, GPU
wheels, and the Fortran reference build.

Examples
--------

Six pedagogic scripts on the canonical API sit at the top of ``examples/``
(no ``main()``, parameters at the top, printed progress, a plot, outputs
written and read back); :doc:`examples` walks through each one:

- ``examples/run_tokamak.py`` — build a namelist in Python, solve a circular
  tokamak, read HDF5/NetCDF back.
- ``examples/run_w7x.py`` — W7-X Boozer geometry with full Fokker-Planck
  collisions (tier-2 recycled Krylov).
- ``examples/transport_coefficients.py`` — monoenergetic transport matrices
  and a collisionality scan.
- ``examples/ambipolar_er_scan.py`` — scan the radial electric field and
  solve the ambipolar root.
- ``examples/gradients_tour.py`` — ``jax.grad`` through the kinetic solve,
  verified against finite differences.
- ``examples/optimize_QA_bootstrap.py`` — flagship gradient-based QA
  stellarator optimization with kinetic ``<j.B>`` in the objective.

Performance and parity evidence
-------------------------------

:doc:`performance` records the measured canonical-stack evidence: on the
744k-unknown HSX PAS/DKES case, the tier-1 structured solve completes in
``27.2 s`` at ``0.93 GB`` on a MacBook M4, versus ``463.6 s`` / ``3.98 GB``
for 1-rank SFINCS Fortran v3 and ``229.5 s`` / ``2.86 GB`` at its measured
2-rank parallel floor. Parity referees pin RHSMode=1 output tables to
``8e-14``, state vectors to ``1e-11``, and transport matrices to
``6e-13 .. 9e-9`` against Fortran golden data.

A broader example-suite benchmark complements that single case: it runs the
full 39-case CPU/GPU example suite against SFINCS Fortran v3 and plots every
row whose Fortran reference runtime clears a ``10 s`` reference-runtime-window,
so process-launch and JIT-amortization noise does not dominate the bars.

.. figure:: _static/figures/paper/sfincs_jax_fortran_suite_benchmark_summary.png
   :alt: Runtime and active-memory comparison for SFINCS Fortran v3 and sfincs_jax across the example suite.
   :align: center
   :width: 90%

   Example-suite benchmark for rows whose SFINCS Fortran v3 reference runtime is
   at least ``10 s``. Fortran memory is process maximum RSS; JAX memory uses
   profiler RSS deltas over the fixed runtime baseline. Reproduce with
   ``examples/publication_figures/generate_fortran_suite_benchmark_summary.py``.

What this documentation covers
------------------------------

- getting started: :doc:`installation`, :doc:`usage`, :doc:`examples`
- physics and numerics: :doc:`physics_models`, :doc:`system_equations`,
  :doc:`geometry`, :doc:`method`, :doc:`numerics`, :doc:`differentiability`,
  :doc:`capabilities`
- references: :doc:`inputs`, :doc:`outputs`, :doc:`normalizations`,
  :doc:`source_map`, :doc:`api`
- evidence: :doc:`performance`, :doc:`parity`, :doc:`feature_matrix`,
  :doc:`fortran_comparison`, :doc:`validation_matrix`
- workflows: :doc:`applications`, :doc:`optimization`, :doc:`parallelism`,
  :doc:`vmec_jax_workflow`

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   applications
   optimization
   examples
   usage
   inputs
   outputs
   normalizations
   geometry
   vmec_jax_workflow
   method
   numerics
   differentiability
   capabilities
   source_map
   feature_matrix
   theory_from_upstream
   physics_models
   physics_reference
   system_equations
   parallelism
   research_lanes
   performance
   development_roadmap
   adaptive_speed_grid
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
