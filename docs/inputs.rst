Inputs (namelist) reference
===========================

`sfincs_jax` reads the same Fortran-style namelist files used by upstream SFINCS v3 (typically named
``input.namelist``).

Because upstream v3 supports a very large configuration space, this page focuses on:

1) **Where to find the complete upstream parameter definitions**, and
2) **What subset of inputs is currently implemented end-to-end in `sfincs_jax`**.

Full upstream parameter documentation
-------------------------------------

For the authoritative definitions of all namelist groups/parameters and their meaning, see the vendored
upstream technical documentation linked from ``docs/upstream_docs.rst`` (PDFs and TeX sources in
``docs/upstream/``).

Current `sfincs_jax` support (high level)
-----------------------------------------

At a high level:

- **Geometry**: `geometryScheme` in `{1,2,4,5,11,12}` is supported for grid/geometry construction and for
  writing `sfincsOutput.h5` parity fixtures.
- **Full-system solve parity**: matrix-free matvec/RHS/residual/GMRES parity is available for a growing
  subset of fixtures (see ``docs/parity.rst``).

Geometry examples you can run immediately:

- analytic tokamak ``geometryScheme=1``:
  ``examples/getting_started/write_sfincs_output_tokamak.py``
- VMEC ``geometryScheme=5``:
  ``examples/getting_started/write_sfincs_output_vmec.py``

The public CLI/API does not currently expose a separate Miller-parameter
geometry mode. Tokamak examples therefore use the supported analytic Boozer
tokamak inputs that mirror the upstream v3 example family.

Practical notes for users
-------------------------

- If you are starting from an upstream example input, the quickest way to see whether it is supported
  end-to-end is:

  - Try `sfincs_jax write-output ...` and compare the resulting ``sfincsOutput.h5`` with upstream.
  - For solve parity status, consult ``docs/parity.rst`` (fixtures) and ``docs/fortran_examples.rst``
    (the example audit).

- If you want differentiability, prefer workflows that construct a `V3FullSystemOperator` once and then
  treat its fields as differentiable parameters (see ``docs/performance.rst``).

- Equilibrium-file resolution uses the same practical search order in the CLI and Python API:

  - absolute path from the namelist or override,
  - relative to the input namelist directory,
  - relative to the current working directory,
  - directories listed in ``SFINCS_JAX_EQUILIBRIA_DIRS``,
  - and then bundled reference/data directories used by tests and packaged examples.

- If you need to point a run at a different equilibrium without editing ``input.namelist``,
  use:

  - CLI: ``--equilibrium-file /path/to/equilibrium`` or ``--wout-path /path/to/wout.nc``
  - Python: ``equilibrium_file=...`` or ``wout_path=...`` in
    :func:`sfincs_jax.io.write_sfincs_jax_output_h5`

Transport-matrix modes (``RHSMode=2/3``)
----------------------------------------------------------------------

In upstream v3, ``RHSMode=2`` and ``RHSMode=3`` (transport-matrix modes) run a loop over ``whichRHS`` and
overwrite the equilibrium gradients and/or inductive field **internally** before building each RHS via
``evaluateResidual(f=0)``. `sfincs_jax` exposes the same behavior via
:func:`sfincs_jax.v3_system.with_transport_rhs_settings` so parity fixtures can reproduce the v3 solver
RHS exactly.

For ``RHSMode=3`` (monoenergetic coefficients), v3 also overwrites the speed grid to a single point at
``x=1`` with ``xWeights=exp(1)`` (see v3 ``createGrids.F90``). `sfincs_jax` matches this behavior in
:func:`sfincs_jax.v3.grids_from_namelist`.
