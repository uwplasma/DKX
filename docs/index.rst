DKX
===

`dkx` solves the radially local, linearized drift-kinetic equation on a
flux surface, in pure JAX. The physics is the same as SFINCS Fortran v3. One
``input.namelist`` plus one geometry file gives neoclassical particle/heat
fluxes, parallel flows, bootstrap current, and transport matrices for
stellarators and tokamaks, on CPU or GPU, with end-to-end automatic
differentiation for sensitivities and optimization.

Quickstart
----------

.. code-block:: bash

   pip install dkx

.. code-block:: python

   from pathlib import Path
   from dkx.run import run_profile

   run = run_profile(Path("input.namelist"), solve_method="auto",
                     out_path=Path("sfincsOutput.h5"))
   print(float(run.moments["particleFlux_vm_psiHat"][0]))
   print(float(run.moments["FSABjHat"]))  # bootstrap current <j.B>

``run_profile`` prints the Fortran-parity console flow, writes
``sfincsOutput.h5``/``.nc`` keyed by the SFINCS output names, and returns the
state vector, solver statistics, and all velocity-space moments in memory. The
CLI equivalent is ``dkx input.namelist --out sfincsOutput.h5``;
``dkx --plot sfincsOutput.h5`` builds a PDF diagnostics panel. See
:doc:`installation` for the ``solvax`` structured-solver core dependency, GPU
wheels, and the Fortran reference build.

Examples
--------

Six pedagogic scripts on the canonical API sit at the top of ``examples/``
(no ``main()``, parameters at the top, printed progress, a plot, outputs
written and read back); :doc:`examples` walks through each one:

- ``examples/getting_started/run_tokamak.py`` — build a namelist in Python, solve a circular
  tokamak, read HDF5/NetCDF back.
- ``examples/getting_started/run_w7x.py`` — W7-X Boozer geometry with full Fokker-Planck
  collisions (the recycled Krylov route).
- ``examples/transport/transport_coefficients.py`` — monoenergetic transport matrices
  and a collisionality scan.
- ``examples/vmex_finite_beta/ambipolar_er_scan.py`` — scan the radial electric field and
  solve the ambipolar root.
- ``examples/autodiff/gradients_tour.py`` — ``jax.grad`` through the kinetic solve,
  verified against finite differences.
- ``examples/optimization/optimize_QA_bootstrap.py`` — gradient-based QA
  stellarator optimization with kinetic ``<j.B>`` in the objective.

Performance and parity evidence
-------------------------------

:doc:`performance` records the measured canonical-stack evidence. On the
744k-unknown HSX PAS/DKES case:

.. list-table:: 744k-unknown HSX PAS/DKES case
   :header-rows: 1
   :widths: 46 20 20

   * - Configuration
     - Solve time
     - Peak RSS
   * - ``dkx`` structured direct solve, MacBook M4
     - ``27.2 s``
     - ``0.93 GB``
   * - SFINCS Fortran v3, 1 rank
     - ``463.6 s``
     - ``3.98 GB``
   * - SFINCS Fortran v3, measured 2-rank parallel floor
     - ``229.5 s``
     - ``2.86 GB``

Cross-check tests pin three envelopes against Fortran golden data: RHSMode=1
output tables to ``8e-14``, state vectors to ``1e-11``, and transport matrices
to ``6e-13 .. 9e-9``.

A broader benchmark covers more than that single case. It runs the full 39-case
CPU/GPU example suite against SFINCS Fortran v3, and plots every row whose
Fortran reference runtime clears a ``10 s`` reference-runtime-window, so
process-launch and JIT-amortization noise does not dominate the bars.

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png
   :alt: Runtime and active-memory comparison for SFINCS Fortran v3 and dkx across the example suite.
   :align: center
   :width: 90%

   Example-suite benchmark for rows whose SFINCS Fortran v3 reference runtime is
   at least ``10 s``. Fortran memory is process maximum RSS; JAX memory uses
   profiler RSS deltas over the fixed runtime baseline. Reproduce with
   ``tools/publication_figures/generate_fortran_suite_benchmark_summary.py``.

Documentation map
-----------------

- getting started: :doc:`installation`, :doc:`usage`, :doc:`case_files`, :doc:`examples`
- physics and numerics: :doc:`physics_models`, :doc:`system_equations`,
  :doc:`geometry`, :doc:`method`, :doc:`numerics`, :doc:`differentiability`,
  :doc:`capabilities`
- references: :doc:`inputs`, :doc:`outputs`, :doc:`normalizations`,
  :doc:`source_map`, :doc:`api`
- evidence: :doc:`performance`, :doc:`parity`, :doc:`feature_matrix`,
  :doc:`fortran_comparison`, :doc:`validation_matrix`
- workflows: :doc:`applications`, :doc:`optimization`, :doc:`parallelism`,
  :doc:`vmex_workflow`

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   applications
   optimization
   examples
   usage
   case_files
   inputs
   outputs
   normalizations
   geometry
   vmex_workflow
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
