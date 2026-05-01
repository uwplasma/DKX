Examples
========

The repository includes a structured `examples/` tree:

- `examples/getting_started/`: basic API usage (no external reference code required)
- `examples/parity/`: focused validation scripts against frozen reference fixtures
- `examples/transport/`: `RHSMode=2/3` transport-matrix workflows + upstream scanplot scripts
- `examples/autodiff/`: autodiff + implicit-diff demonstrations
- `examples/optimization/`: optimization patterns (may require extras)
- `examples/performance/`: JIT/performance microbenchmarks
- `examples/publication_figures/`: publication-style figure generation
- `examples/vmec_jax_finite_beta/`: finite-beta ``vmec_jax`` to ``sfincs_jax`` radial bootstrap-current and ambipolar-``E_r`` workflow

Run from the repo root:

.. code-block:: bash

   cd sfincs_jax
   python examples/getting_started/build_grids_and_geometry.py

Writing `sfincsOutput.h5` (Python + CLI):

.. code-block:: bash

   python examples/getting_started/write_sfincs_output_python.py
   python examples/getting_started/write_sfincs_output_cli.py

Geometry-specific write-output examples:

.. code-block:: bash

   python examples/getting_started/write_sfincs_output_tokamak.py
   python examples/getting_started/write_sfincs_output_vmec.py

Finite-beta VMEC-JAX to kinetic transport
-----------------------------------------

The finite-beta example is a single Python script that reads the bundled
``input.nfp2_QA_finite_beta`` VMEC input, runs ``vmec_jax`` for a bounded number
of fixed-boundary iterations, writes a VMEC-style ``wout`` file, and uses that
file directly in ``sfincs_jax`` with ``geometryScheme=5``.  It then scans the
normalized radial electric field on several flux surfaces, computes ambipolar
radial-current roots when each scan brackets them, selects a continuous root
branch, and writes a polished PNG/PDF panel with the radial electric-field
profile, bootstrap-current radial profile, representative ambipolarity and flux
scans in ``Er``, and a VMEC magnetic-field contour plot using a ``jet`` colormap.
The radial-profile x-axis is normalized toroidal flux,
:math:`\psi_N = r_N^2`, not the square-root radial label used in the input file.

.. code-block:: bash

   export SFINCS_JAX_VMEC_JAX_PATH=/path/to/vmec_jax  # optional for source checkouts
   python examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py

.. figure:: _static/figures/finite_beta_vmec_jax_sfincs_bootstrap_er.png
   :alt: Finite-beta VMEC-JAX to sfincs_jax radial profiles, Er scans, fluxes, and magnetic-field contour.
   :width: 100%

   Finite-beta VMEC-JAX to ``sfincs_jax`` example panel.  The top row shows the
   ambipolar radial-electric-field profile versus normalized toroidal flux, the
   bootstrap-current profile versus normalized toroidal flux, and the sampled
   VMEC magnetic-field-strength contour.  The bottom row shows the representative
   ambipolarity scan, bootstrap response, and ion/electron particle fluxes versus
   normalized ``Er`` at the selected surface.  Open markers show all bracketed
   roots; filled markers show the selected continuous branch; dashed black markers
   show the tighter root-bracket convergence scan at the same kinetic grid.

The direct VMEC path does not require a Boozer transform: ``sfincs_jax`` consumes
the generated ``wout`` through the same scheme-5 geometry implementation used by
file-based VMEC runs.  ``booz_xform_jax`` remains useful for the separate
differentiable Boozer-spectrum handoff described below.

By default the radial profile uses ``r_N = 0.15, 0.30, 0.50, 0.70, 0.85`` and
root-neighborhood ``Er = -9, -7, -5, -3, -1``.  This spans core-to-edge behavior
while avoiding the exact magnetic axis and VMEC boundary, and it spends the
expensive solves near the ambipolar branch instead of at far-off electric fields.
The checked documentation figure is run at ``Ntheta=7``, ``Nzeta=7``,
``Nxi=8``, ``NL=6``, and ``Nx=6`` with adaptive midpoint refinement of bracketed
ambipolar roots until the local ``Er`` bracket is no wider than ``1.25``.  The
dashed overlay in the top panels uses the same kinetic-space grid on every
plotted surface, but refines each bracketed ambipolar root to a local ``Er``
bracket width of ``0.625``.  The checked documentation figure passes the default
gate, ``max |Delta Er| <= 0.1`` and ``max |Delta Jbs| <= 5e-4``; the measured
values are ``max |Delta Er| = 2.1e-4`` and ``max |Delta Jbs| = 6.9e-7`` in the
cached documentation run.  Use ``--quick`` for the smoke-test resolution, and
increase ``--r-n-values``, ``--er-values``, and the resolution flags for denser
production-quality scans.

The convergence-scan helper reads the cached ``sfincsOutput.h5`` files from the
finite-beta example and summarizes which numerical knobs move the ambipolar root
and bootstrap current:

.. code-block:: bash

   python examples/vmec_jax_finite_beta/plot_convergence_scan.py

.. figure:: _static/figures/finite_beta_vmec_jax_sfincs_convergence_scan.png
   :alt: Finite-beta VMEC-JAX to sfincs_jax convergence scan over kinetic resolution, angular resolution, and Er-root bracket width.
   :width: 100%

   Convergence scan for the finite-beta example.  The top panels show that the
   cached documentation campaign is still sensitive to kinetic-space resolution:
   moving from ``7/6/6`` to ``8/6/6`` changes the selected ambipolar root and
   bootstrap current, especially at the representative surfaces.  The bottom-left
   panel shows that, once the practical ``8/6/6`` kinetic grid is fixed,
   tightening the local ambipolar-root bracket from ``1.25`` to ``0.625`` changes
   the full radial profile by only ``max |Delta Er| = 2.1e-4`` and
   ``max |Delta Jbs| = 6.9e-7``.  The bottom-right panel records one-parameter
   probes at ``r_N=0.50``: ``NL=7`` is stable, while ``Nxi=8`` and ``Nx=7`` both
   move the root.  Combined ``Nxi=8,Nx=7`` and ``Nxi=9`` probes were too expensive
   on a single RTX A4000 for this bounded documentation campaign, so this figure
   is an explicit resolution audit rather than a claim of asymptotic
   kinetic-space convergence.

Plotting a generated or frozen output file:

.. code-block:: bash

   sfincs_jax --plot sfincsOutput.h5
   python examples/getting_started/plot_sfincs_output.py

Matrix-free linear solve demo (using frozen PETSc binaries):

.. code-block:: bash

   python examples/parity/solve_fortran_matrix_with_gmres.py
   python examples/autodiff/autodiff_gradient_nu_n_residual.py

Transport matrices (RHSMode=2/3)
--------------------------------

Upstream v3 uses ``RHSMode=2`` and ``RHSMode=3`` to compute transport matrices by looping over multiple
right-hand sides (``whichRHS``) and assembling a matrix from diagnostic moments of the solved distribution.

`sfincs_jax` provides both a Python driver and a CLI:

.. code-block:: bash

   python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py
   sfincs_jax transport-matrix-v3 --input input.namelist --out-matrix transportMatrix.npy

Upstream postprocessing (utils/)
--------------------------------

The mature SFINCS ecosystem includes a set of plotting scripts under `utils/`. `sfincs_jax` vendors these scripts
in `examples/sfincs_examples/utils/` and can run them non-interactively:

.. code-block:: bash

   sfincs_jax postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf

There is also a small end-to-end demo that generates PDF figures for a tiny transport-matrix case:

.. code-block:: bash

   python examples/transport/postprocess_upstream_scanplot_1_transport_matrix.py

Some advanced examples require optional dependencies:

.. code-block:: bash

   pip install optax

Optimization + figures
----------------------

Two examples that showcase autodiff-driven optimization (and write publication-style figures when `matplotlib`
is available):

.. code-block:: bash

   pip install optax
   python examples/optimization/optimize_scheme4_harmonics_publication_figures.py
   python examples/optimization/calibrate_nu_n_to_fortran_residual_fixture.py

For bounded optional ecosystem checks around differentiable objective wrappers:

.. code-block:: bash

   pip install equinox jaxopt
   python examples/optimization/benchmark_optional_eqx_jaxopt_scheme4_gate.py --backend all

Implicit differentiation through solves
---------------------------------------

An important differentiability capability is **implicit differentiation** through a linear solve
(``A x = b``) without backpropagating through Krylov iterations. `sfincs_jax` provides a small helper
based on `jax.lax.custom_linear_solve` and demonstrates it here:

.. code-block:: bash

   python examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver gmres
   python examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver bicgstab

VMEC-to-Boozer differentiable geometry handoff
----------------------------------------------

For optional ``vmec_jax`` and ``booz_xform_jax`` installations, this example
checks a public differentiable geometry handoff into ``sfincs_jax``:

.. code-block:: bash

   python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py \
     --wout /path/to/wout_circular_tokamak.nc \
     --mboz 3 \
     --nboz 3 \
     --surface 0.5

The script reports a Boozer-spectrum geometry proxy, its JAX gradient, a centered
finite-difference gradient, and a few scalar optimization steps.  It is a fast
research workflow gate for JAX-native geometry coupling; it is not a full kinetic
transport optimization solve.

JIT-compiled optimization with implicit gradients
--------------------------------------------------

This example performs a fully JIT-compiled objective evaluation and gradient-based
optimization loop using implicit differentiation through the linear solve:

.. code-block:: bash

   python examples/autodiff/optimize_nu_n_implicit.py

Parallel and scaling examples
-----------------------------

For transport-matrix throughput on CPUs:

.. code-block:: bash

   python examples/performance/benchmark_transport_parallel_scaling.py \
     --input examples/performance/transport_parallel_2min.input.namelist \
     --workers 1 2 4

For transport-matrix throughput on a 2-GPU node:

.. code-block:: bash

   PYTHONPATH=. python examples/performance/benchmark_transport_parallel_scaling.py \
     --backend gpu \
     --input examples/performance/transport_parallel_2min.input.namelist \
     --workers 1 2

For sharded single-RHS solves on CPU or GPU:

.. code-block:: bash

   python examples/performance/benchmark_sharded_solve_scaling.py \
     --backend cpu \
     --input examples/performance/rhsmode1_sharded.input.namelist \
     --devices 1 2 4 8 \
     --inner-warmup-solves 1 \
     --sample-timeout-s 300 \
     --rhs1-precond theta_schwarz \
     --schwarz-coarse-levels 2

For the current one-GPU-per-case throughput benchmark on a 2-GPU node:

.. code-block:: bash

   PYTHONPATH=. python examples/performance/benchmark_multi_gpu_case_throughput.py \
     --input examples/performance/rhsmode1_sharded_scaling.input.namelist \
     --nsolve 4

.. note::

   ``geometryScheme=5`` (VMEC) and analytic tokamak ``geometryScheme=1`` are
   supported public examples today. `sfincs_jax` does not currently expose a
   separate Miller-parameter geometry mode in the public CLI/API, so tokamak
   examples use the supported analytic Boozer tokamak path instead.

It builds a cached operator once, treats :math:`\\nu_n` as a differentiable parameter,
and minimizes :math:`0.5\\|x(\\nu_n)\\|^2` where :math:`A(\\nu_n)x=b(\\nu_n)` is solved
with `custom_linear_solve`. This is the recommended pattern for fast, memory-efficient
gradients without backpropagating through Krylov iterations.

Transport-matrix recycling warm starts
--------------------------------------

To reuse recent Krylov solutions across ``whichRHS`` solves (RHSMode=2/3), use:

.. code-block:: bash

   python examples/transport/transport_matrix_recycle_demo.py --recycle-k 4

Upstream SFINCS example inputs
--------------------------------

For convenience, `sfincs_jax` also vendors the original example-input families (multi-species,
and MATLAB v3) in `examples/upstream/`. These files are intended as recognizable reference points for
SFINCS users; not all of them are runnable end-to-end in `sfincs_jax` yet.

The full upstream-style example suite (plus the upstream postprocessing scripts) is also vendored in
`examples/sfincs_examples/`. A best-effort runner is provided:

.. code-block:: bash

   python examples/sfincs_examples/run_sfincs_jax.py --write-output
