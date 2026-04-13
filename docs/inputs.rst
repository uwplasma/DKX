Inputs (namelist) reference
===========================

`sfincs_jax` reads Fortran-style namelist files, typically named ``input.namelist``.
The input surface is intentionally compatible with the mature SFINCS-style ecosystem so
users can reuse equilibria, scans, and case descriptions, but this page documents the
public `sfincs_jax` interface in its own right.

This page focuses on:

1) the most important public knobs,
2) geometry and equilibrium-file handling,
3) supported runtime overrides,
4) where to find the exhaustive historical parameter definitions when needed.

Primary parameter families
--------------------------

The most important parameter groups in practice are:

- **general / solve mode**:
  ``RHSMode``, ``collisionOperator``, ``constraintScheme``, ``includePhi1``
- **geometry**:
  ``geometryScheme``, ``equilibriumFile``, radial-coordinate selectors
- **resolution**:
  ``Ntheta``, ``Nzeta``, ``Nxi``, ``Nx`` and related grid settings
- **profiles and drives**:
  densities, temperatures, gradients, electric-field quantities
- **trajectory / physics switches**:
  ``magneticDriftScheme``, ``useDKESExBDrift``, ``includeXDotTerm``,
  ``includePhi1InKineticEquation``, ``includePhi1InCollisionOperator``

For full historical parameter descriptions, see the bundled upstream manual and notes.

Geometry and equilibrium inputs
-------------------------------

The public supported geometry families are documented in :doc:`geometry`. In practice,
the most common input choices are:

- analytic tokamak-style geometry with ``geometryScheme=1``,
- VMEC with ``geometryScheme=5`` and ``equilibriumFile`` or ``wout_path``,
- Boozer ``.bc`` geometry with ``geometryScheme=11`` or ``12``.

The CLI and Python API use the same equilibrium search order:

- absolute path from the namelist or explicit override,
- relative to the input namelist directory,
- relative to the current working directory,
- directories listed in ``SFINCS_JAX_EQUILIBRIA_DIRS``,
- then bundled test/example data directories.

Runtime overrides
-----------------

Without editing the namelist, users can override the equilibrium source through:

- CLI: ``--equilibrium-file ...`` or ``--wout-path ...``
- Python:
  :func:`sfincs_jax.io.write_sfincs_jax_output_h5` with ``equilibrium_file=...`` or
  ``wout_path=...``

When an override is used, the effective configuration is also reflected in the embedded
``input.namelist`` dataset written to ``sfincsOutput.h5``.

Current `sfincs_jax` support (high level)
-----------------------------------------

At a high level:

- **Geometry**: `geometryScheme` in `{1,2,4,5,11,12}` is supported for grid/geometry construction and for
  writing `sfincsOutput.h5` parity fixtures.
- **Solve/output validation**: the release-facing audited scope is documented in
  :doc:`fortran_comparison` and :doc:`testing`.

Geometry examples you can run immediately:

- analytic tokamak ``geometryScheme=1``:
  ``examples/getting_started/write_sfincs_output_tokamak.py``
- VMEC ``geometryScheme=5``:
  ``examples/getting_started/write_sfincs_output_vmec.py``

There is not currently a separate public Miller-parameter geometry interface. Tokamak
examples therefore use the supported analytic straight-field-line model family.

Practical notes for users
-------------------------

- If you are starting from an existing case, the quickest first check is usually
  ``sfincs_jax write-output ...`` followed by inspection of ``sfincsOutput.h5``.

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
----------------------------------------

For ``RHSMode=2`` and ``RHSMode=3`` (transport-matrix modes), `sfincs_jax` runs a loop
over ``whichRHS`` and overwrites the relevant drives internally before building each RHS.
This behavior is exposed via
:func:`sfincs_jax.v3_system.with_transport_rhs_settings` so parity fixtures can reproduce the v3 solver
RHS exactly.

For ``RHSMode=3`` (monoenergetic coefficients), v3 also overwrites the speed grid to a single point at
``x=1`` with ``xWeights=exp(1)`` (see v3 ``createGrids.F90``). `sfincs_jax` matches this behavior in
:func:`sfincs_jax.v3.grids_from_namelist`.
