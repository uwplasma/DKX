Outputs (sfincsOutput.h5)
=========================

`sfincs_jax` writes results to an HDF5 file named ``sfincsOutput.h5``. The file layout
is designed to remain compatible with the established SFINCS-style postprocessing
ecosystem while also serving as the native public results format of `sfincs_jax`.

Writing output with `sfincs_jax`
--------------------------------

CLI
^^^

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5

.. code-block:: bash

   sfincs_jax write-output \
     --input input.namelist \
     --out sfincsOutput.h5 \
     --wout-path /path/to/wout.nc

For a compact summary figure from an existing output file:

.. code-block:: bash

   sfincs_jax --plot sfincsOutput.h5

.. code-block:: bash

   sfincs_jax plot-output --input-h5 sfincsOutput.h5 --out sfincsOutput_summary.png

Use ``--equilibrium-file`` for a generic Boozer or VMEC override, or ``--wout-path``
as a compatibility alias for VMEC-centered workflows.

For transport-matrix runs (``RHSMode=2`` or ``RHSMode=3``), the Fortran code loops over
multiple right-hand sides (``whichRHS``) and assembles a ``transportMatrix`` in the output.
To replicate that end-to-end behavior in `sfincs_jax`, enable:

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5 --compute-transport-matrix

In this mode, `sfincs_jax` also writes the RHSMode>1 diagnostics used by upstream scan plotting scripts:
``FSABFlow``, ``particleFlux_vm_psiHat``, and ``heatFlux_vm_psiHat``.

The default output uses a Fortran-compatible array layout. This is useful both for
existing postprocessing tools and for external validation with ``sfincs_jax compare-h5``.

Python
^^^^^^

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.io import write_sfincs_jax_output_h5

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
   )

.. code-block:: python

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
       wout_path=Path("/path/to/wout.nc"),
   )

.. code-block:: python

   out_path, results = write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
       return_results=True,
   )
   print(out_path)
   print(results["Ntheta"])

When an equilibrium override is supplied, ``sfincs_jax`` updates the embedded
``input.namelist`` dataset in ``sfincsOutput.h5`` to match the effective run
configuration.

Current coverage
----------------

At the moment, `sfincs_jax` output writing supports:

- ``geometryScheme = 4`` (simplified W7-X Boozer model)
- ``geometryScheme = 5`` (VMEC ``wout_*.nc`` netCDF workflow)
- ``geometryScheme = 11/12`` (Boozer `.bc` files for W7-X / general non-stellarator-symmetric equilibria)
- ``geometryScheme = 1/2`` (analytic Boozer models used by several v3 examples)
- v3 grids: ``theta``, ``zeta``, ``x`` and ``Nxi_for_x``
- core geometry fields: ``BHat``, ``DHat`` and derivatives available in `sfincs_jax.geometry`
- basic scalar integrals: ``VPrimeHat`` and ``FSABHat2`` (see `sfincs_jax.diagnostics`)
- selected run parameters, radial-coordinate conversions, and species arrays (e.g. ``Delta``, ``alpha``, ``Er``, ``dPhiHatdpsiHat``,
  ``psiAHat``, ``aHat``, ``rN``, ``Zs``, ``THats``)
- `NTV`-related geometry diagnostic ``uHat`` (computed from harmonics of :math:`1/\hat B^2`)
- transport-matrix output fields for RHSMode=2/3 runs: ``transportMatrix`` and the minimal
  diagnostics needed by upstream plotting scripts (see above)
- v3 classical transport fluxes (`calculateClassicalFlux`) for geometries with `gpsiHatpsiHat` support
  (VMEC `geometryScheme=5` and `.bc` `geometryScheme=11/12`), written as:
  ``classicalParticleFluxNoPhi1_*`` / ``classicalHeatFluxNoPhi1_*`` (static) and
  ``classicalParticleFlux_*`` / ``classicalHeatFlux_*`` (per-iteration diagnostics)

Output-writing regression tests live in:

- ``tests/test_output_h5_scheme4_parity.py`` (scheme 4)
- ``tests/test_output_h5_scheme1_parity.py`` (scheme 1)
- ``tests/test_output_h5_scheme2_parity.py`` (scheme 2)
- ``tests/test_output_h5_scheme11_parity.py`` (scheme 11)
- ``tests/test_output_h5_scheme5_parity.py`` (scheme 5)
- ``tests/test_transport_matrix_write_output_end_to_end.py`` (transport matrices, including geometryScheme=11/12 fixtures)

and compare the datasets above against frozen Fortran v3 fixtures in ``tests/ref``.

There is also a multi-species regression against the established reference output for
``quick_2species_FPCollisions_noEr``, implemented in
``tests/test_output_h5_scheme4_quick2species_parity.py``.

Plotting output files
---------------------

The CLI supports direct plotting from any existing ``sfincsOutput.h5``:

.. code-block:: bash

   sfincs_jax --plot sfincsOutput.h5

By default this writes ``<input>_summary.png`` next to the HDF5 file. Use
``plot-output --out`` to choose a different filename.

For a minimal end-to-end plotting example from the repository, run:

.. code-block:: bash

   python examples/getting_started/plot_sfincs_output.py

The script and the CLI both call the same plotting helper and write a compact
PNG summary showing representative radial diagnostics and the geometry field
``BHat(theta, zeta)``. The figure includes:

- ``FSABFlow_vs_x``
- ``heatFlux_vm_psiHat_vs_x``
- ``BHat(theta, zeta)``

.. note::

   ``uHat`` depends on many transcendental evaluations (cos/sin) and long floating-point
   reductions. In practice we observe tiny platform-dependent differences vs the frozen
   Fortran fixture (absolute errors :math:`\sim 10^{-9}` in the small scheme-4 test case),
   so the parity test compares ``uHat`` with a slightly looser tolerance than most other
   datasets.

Fortran vs Python array layout
------------------------------

Fortran writes arrays in column-major order. When those HDF5 datasets are read back in
Python, multi-dimensional arrays often appear with axes reversed relative to the
``(itheta, izeta, ...)`` indexing used in the Fortran source.

To make it easy to do *file-to-file* comparisons in Python, `sfincs_jax` writes arrays
using the same convention by default (see `sfincs_jax.io.write_sfincs_h5`).
