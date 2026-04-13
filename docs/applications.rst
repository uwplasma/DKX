Applications and research workflows
===================================

`sfincs_jax` is designed for practical neoclassical analysis, benchmarking, and
optimization-oriented workflows. This page summarizes the main ways the code is used in
practice and points to the repository examples that exercise those paths.

Core application areas
----------------------

Neoclassical transport coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard use case is to compute:

- particle fluxes,
- heat fluxes,
- flows,
- bootstrap-current-related moments,
- and transport matrices / monoenergetic coefficients.

These quantities feed transport studies, ambipolar-field calculations, and cross-code
benchmarking.

Examples:

- ``examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py``
- ``examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py``

Geometry studies
^^^^^^^^^^^^^^^^

`sfincs_jax` supports analytic, VMEC, and Boozer geometry workflows. This makes it
useful for:

- controlled analytic tokamak studies,
- stellarator transport studies on VMEC equilibria,
- Boozer-coordinate benchmark and scan workflows,
- geometry-sensitivity studies when differentiated or repeated solves are needed.

Examples:

- ``examples/getting_started/write_sfincs_output_tokamak.py``
- ``examples/getting_started/write_sfincs_output_vmec.py``

Differentiable and inverse-design workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Python API can retain differentiability through selected solve paths. This is useful
for:

- sensitivity analysis,
- gradient-based optimization,
- calibration / inverse problems,
- and coupling to higher-level design loops.

Examples:

- ``examples/autodiff/autodiff_gradient_nu_n_residual.py``
- ``examples/autodiff/autodiff_sensitivity_nu_n_scheme5.py``
- ``examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py``

Parallel production workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two different parallel patterns matter in practice:

- **case/RHS parallelism** for throughput,
- **single-case sharding** for very large solves.

The production-ready GPU scaling story on the documented audited scope is the
transport-worker lane for RHSMode=2/3 runs. The sharded single-case path remains
available for research and benchmarking, but is still more sensitive to hardware and
problem structure.

Examples:

- ``examples/performance/benchmark_transport_parallel_scaling.py``
- ``examples/performance/benchmark_multi_gpu_case_throughput.py``
- ``examples/performance/benchmark_sharded_solve_scaling.py``

Plotting and postprocessing
---------------------------

The repository includes simple plotting scripts for users who want to inspect or publish
results quickly:

- ``examples/getting_started/plot_sfincs_output.py``
- ``examples/publication_figures/magnetic_drifts_publication_figures.py``

These examples are intentionally lightweight. They are meant to show how to read
``sfincsOutput.h5`` and how to turn the stored diagnostics into publication figures or
analysis notebooks.

Typical research workflow
-------------------------

A common end-to-end workflow is:

1. choose a geometry source and flux surface,
2. run a baseline transport or single-RHS solve,
3. verify resolution sensitivity in the key axes,
4. scan profiles, collisionality, or electric field,
5. postprocess fluxes / flows / bootstrap current,
6. optionally differentiate or optimize with respect to selected inputs.

The examples directory is organized to mirror those tasks:

- ``examples/getting_started`` for first runs,
- ``examples/transport`` for transport coefficients,
- ``examples/autodiff`` for derivative-aware workflows,
- ``examples/performance`` for timing and scaling studies.

Trust and external comparison
-----------------------------

For users who need external validation before applying the code to a new device or
profile set, the repository also documents comparison against frozen reference outputs
from a mature Fortran implementation on the audited scope. That material is kept as a
validation section, not as the primary identity of the code.

See :doc:`fortran_comparison` for that trust-building material and :doc:`testing` for
the broader validation strategy.
