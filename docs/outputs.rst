Outputs (HDF5, NetCDF4, and NPZ)
================================

`sfincs_jax` writes results to the format selected by the output filename:

- ``.h5`` / ``.hdf5``: Fortran-compatible HDF5, the parity and regression-test default.
- ``.nc`` / ``.netcdf``: NetCDF4, useful for xarray, climate/space-physics tooling,
  and long-lived metadata-aware archives.
- ``.npz``: fast uncompressed NumPy archive, useful for lightweight Python workflows
  and rapid local sweeps.

The HDF5 layout is designed to remain compatible with the established SFINCS-style
postprocessing ecosystem while also serving as the native public results format of
`sfincs_jax`.

Writing output with `sfincs_jax`
--------------------------------

CLI
^^^

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5
   sfincs_jax write-output --input input.namelist --out sfincsOutput.nc
   sfincs_jax write-output --input input.namelist --out sfincsOutput.npz

.. code-block:: bash

   sfincs_jax write-output \
     --input input.namelist \
     --out sfincsOutput.h5 \
     --wout-path /path/to/wout.nc

For RHSMode=1 solves, the output includes solver-convergence metadata in the
main file:

- ``linearSolverMethod`` records the requested solve method passed by the CLI,
  Python API, or environment policy. ``linearSolverRequestedMethod`` repeats the
  requested method when the selected solver reports it explicitly.
- ``linearSolverPath`` and ``linearSolverKind`` record the selected
  implementation route when the solver exposes it.
- ``linearSolverPreconditionerKind`` records the selected preconditioner when
  available, for example ``pas_tz`` or ``collision``. (Fields describing the
  deleted sparse-PC/structured-CSR lanes no longer appear in new outputs.)
- ``linearSolverResidualNorm`` and ``linearSolverResidualTarget`` record the
  true residual norm and requested target used by the output safety gate.
- ``linearSolverResidualTargetRatio`` is residual divided by target.
- ``linearSolverConverged`` is a Fortran-style logical flag, ``+1`` when the
  residual target was met and ``-1`` otherwise.
- ``linearSolverAccepted`` and ``linearSolverAcceptanceCriterion`` record the
  branch acceptance actually used by the writer. In constrained-PAS
  PETSc-compatible minimum-norm runs, true-residual convergence can be false
  while the branch is still accepted and labeled explicitly.
- ``linearSolverIterations`` and ``linearSolverMatvecs`` record iteration work
  when the selected solver exposes those counters.
- ``linearSolverSetupTime``, ``linearSolverSolveTime``, and
  ``linearSolverElapsedTime`` split preconditioner setup from Krylov iteration
  time for large production solves. (The sparse-pattern/CSR/factor-estimate
  metadata of the deleted sparse-PC lanes no longer appears in new outputs.)
- The optional solver-trace sidecar records string-valued solver policy
  metadata. Use this sidecar when auditing why two runs with the same input used different
  sparse-factor ordering or precision.

For a publication-style PDF diagnostics panel from an existing output file:

.. code-block:: bash

   sfincs_jax --plot sfincsOutput.h5

.. code-block:: bash

   sfincs_jax plot-output --input-h5 sfincsOutput.h5 --out sfincsOutput_summary.pdf

Use ``--equilibrium-file`` for a generic Boozer or VMEC override, or ``--wout-path``
as a compatibility alias for VMEC-centered workflows.

To time writer/readback overhead independently from JAX compile and solve cost:

.. code-block:: bash

   python examples/performance/benchmark_output_formats.py --repeats 5

For transport-matrix runs (``RHSMode=2`` or ``RHSMode=3``), the Fortran code loops over
multiple right-hand sides (``whichRHS``) and assembles a ``transportMatrix`` in the output.
To replicate that end-to-end behavior in `sfincs_jax`, enable:

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5 --compute-transport-matrix

In this mode, `sfincs_jax` also writes the RHSMode>1 diagnostics used by upstream scan plotting scripts:
``FSABFlow``, ``particleFlux_vm_psiHat``, and ``heatFlux_vm_psiHat``.

The default HDF5 output uses a Fortran-compatible array layout. This is useful both
for existing postprocessing tools and for external validation with
``sfincs_jax compare-h5``. NetCDF and NPZ use the same array layout policy so
Python-level values match HDF5 readback.

Python
^^^^^^

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.api import write_output

   write_output(Path("input.namelist"), Path("sfincsOutput.h5"))

.. code-block:: python

   # The suffix chooses the writer; the solve and diagnostics are unchanged.
   write_output(Path("input.namelist"), Path("sfincsOutput.nc"))

   write_output(Path("input.namelist"), Path("sfincsOutput.npz"))

.. code-block:: python

   write_output(
       Path("input.namelist"),
       Path("sfincsOutput.h5"),
       wout_path=Path("/path/to/wout.nc"),
   )

.. code-block:: python

   from sfincs_jax.io import read_sfincs_h5

   out_path = write_output(Path("input.namelist"), Path("sfincsOutput.h5"))
   results = read_sfincs_h5(out_path)
   print(out_path)
   print(results["Ntheta"])

When an equilibrium override is supplied, ``sfincs_jax`` updates the embedded
``input.namelist`` dataset/variable in the output file to match the effective run
configuration. Use ``sfincs_jax.io.read_sfincs_output_file(...)`` to load HDF5,
NetCDF, or NPZ outputs with the same dictionary interface.

Output-variable reference
-------------------------

The writer emits a **base** field set for every ``RHSMode`` plus a per-iteration
set (``RHSMode=1`` profile diagnostics, or ``RHSMode=2/3`` transport columns).
The moment producers are :func:`sfincs_jax.moments.rhsmode1_moments` and
``transport_moments_table``. In the shapes below, **S** = species, **T** =
Ntheta, **Z** = Nzeta, **X** = Nx, **N** = number of RHS columns (1 for
RHSMode 1). Base :math:`(\theta,\zeta)` geometry arrays are stored transposed and
read back as :math:`(Z, T)` in ``h5py`` (:ref:`fortran-layout`).

Grids and geometry (base, all modes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Grids: ``theta`` (T), ``zeta`` (Z), ``x`` (X speed nodes), ``Nxi_for_x`` (X;
  Legendre modes kept per speed node).
- Scalars: ``NPeriods``, ``B0OverBBar``, ``GHat``, ``IHat``, ``iota``,
  ``VPrimeHat`` (:math:`\hat V'`), ``FSABHat2`` (:math:`\langle\hat B^2\rangle`),
  ``diotadpsiHat``.
- Field arrays (Z,T): ``BHat``, ``DHat``, ``dBHatdtheta``, ``dBHatdzeta``,
  ``dBHatdpsiHat``, the covariant/contravariant components
  ``BHat_sub_{theta,zeta,psi}`` and ``BHat_sup_{theta,zeta}`` with their
  derivatives, ``BDotCurlB``, and ``uHat`` (NTV geometry potential, zero for VMEC
  scheme 5). ``gpsiHatpsiHat`` (:math:`|\nabla\hat\psi|^2`) is populated for
  Boozer/VMEC input and zero for the analytic schemes.
- Scheme-1 only: ``epsilon_t``, ``epsilon_h``, ``epsilon_antisymm``,
  ``helicity_l``, ``helicity_n``, ``helicity_antisymm_l``,
  ``helicity_antisymm_n``.

Normalization scalars and species profiles (base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Ordering parameters ``Delta``, ``alpha``, ``nu_n`` (:doc:`normalizations`);
  ``EParallelHat``; the radial-electric-field drive ``Er`` and
  ``dPhiHatd{psiHat,psiN,rHat,rN}``.
- Surface label in four coordinates: ``psiHat``, ``psiN``, ``rHat``, ``rN``,
  plus ``psiAHat``, ``aHat``.
- Species arrays (S): ``Zs``, ``mHats``, ``THats``, ``nHats``, and the gradients
  ``dnHatd{psiHat,psiN,rHat,rN}`` / ``dTHatd{...}``.
- ``RHSMode=3`` adds ``nuPrime`` and ``EStar``.

Per-species radial fluxes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Surface-integrated fluxes are ``(S,N)`` (or ``(S,1)`` for RHSMode 1):

- ``particleFlux_vm_psiHat``, ``heatFlux_vm_psiHat``, ``momentumFlux_vm_psiHat``
  — the magnetic-drift radial fluxes (the ``FSABjHat``/flow diagnostics and the
  transport matrix are built from these).
- ``*_vm0_psiHat`` variants use only the leading-order :math:`f_{s0}`.
- Per-speed decompositions ``particleFlux_vm_psiHat_vs_x``,
  ``heatFlux_vm_psiHat_vs_x`` (X,S,N), and the ``*BeforeSurfaceIntegral_vm/_vm0``
  integrands (Z,T,S,N).
- Each ``_psiHat`` flux is also emitted in ``_psiN``, ``_rHat``, ``_rN``
  (see :ref:`flux-legend`).

Parallel flows and bootstrap current
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Grid moments (Z,T,S,N): ``densityPerturbation``, ``pressurePerturbation``,
  ``pressureAnisotropy``, ``flow`` (parallel flow), ``totalDensity``,
  ``totalPressure``, ``velocityUsingFSADensity``, ``velocityUsingTotalDensity``,
  ``MachUsingFSAThermalSpeed``.
- Flux-surface averages (S,N): ``FSADensityPerturbation``,
  ``FSAPressurePerturbation``, ``FSABFlow`` (:math:`\langle\hat B\hat
  V_{\parallel s}\rangle`) and its ``FSABVelocity*`` normalizations;
  ``FSABFlow_vs_x`` (X,S,N).
- **Bootstrap current**: ``FSABjHat`` :math:`=\langle\mathbf{j}\cdot\mathbf{B}\rangle
  =\sum_s Z_s\,\mathrm{FSABFlow}_s` (N), with ``FSABjHatOverB0`` and
  ``FSABjHatOverRootFSAB2``; ``jHat`` :math:`=\sum_s Z_s\,\mathrm{flow}_s(\theta,\zeta)`
  (Z,T,N), the parallel current density on the grid.

Transport matrix (``RHSMode=2/3`` only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``transportMatrix`` — the :math:`3\times3` Onsager matrix (RHSMode 2) or the
  :math:`2\times2` monoenergetic/DKES matrix (RHSMode 3), stored transposed
  (Fortran column-major); ``NIterations`` records the matrix dimension.

:math:`\Phi_1` / quasineutrality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``Phi1Hat`` — :math:`\Phi_1(\theta,\zeta)` (Z,T,1), written for ``RHSMode=1``
  with ``includePhi1`` set. The ``includePhi1``,
  ``includePhi1InKineticEquation``, ``includePhi1InCollisionOperator`` logical
  flags are in the base set.

Classical fluxes, NTV, sources, and metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Classical (collisional) fluxes for geometries with the
  :math:`|\nabla\hat\psi|^2` metric (VMEC scheme 5 and Boozer schemes 11/12):
  ``classicalParticleFluxNoPhi1_*`` / ``classicalHeatFluxNoPhi1_*`` (at input
  gradients) and ``classicalParticleFlux_*`` / ``classicalHeatFlux_*``
  (per-iteration), each in the four radial coordinates.
- ``NTV`` and ``NTVBeforeSurfaceIntegral`` — neoclassical toroidal viscosity
  (zero for VMEC scheme 5).
- ``sources`` — constraint-scheme particle/heat source unknowns.
- ``elapsed time (s)`` (N), and the raw ``input.namelist`` dataset (NetCDF:
  ``input_namelist`` global attribute), plus the resolution/option integers and
  run-config logicals (v3 ±1 encoding for booleans).

.. _flux-legend:

Flux-flavor and radial-coordinate legend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Drift flavor** (suffix on flux names):

- ``_vm`` — magnetic (:math:`\nabla B` + curvature) drift on the full
  distribution; geometric factor
  :math:`(\hat B_\theta\,\partial_\zeta\hat B - \hat B_\zeta\,\partial_\theta\hat B)/\hat B^3`.
- ``_vm0`` — the same operator on :math:`f_{s0}` only.
- ``_vE`` / ``_vE0`` — :math:`E\times B` drift (factor with :math:`/\hat B^2` and
  :math:`\partial\Phi_1`). **Deferred in the file**: the moment functions exist
  (``electric_drift_flux_moments``), but only zero ``BeforeSurfaceIntegral_vE``
  arrays are written — the surface-integrated ``_vE`` / total-drift ``_vd``
  scalars are not emitted.

**Radial coordinate** (a :math:`\nabla\hat\psi` flux is converted by
:math:`\times\,d(\text{coord})/d\hat\psi`):

- ``_psiHat`` — native normalized poloidal flux (all fluxes computed here first);
- ``_psiN`` = ``_psiHat`` / ``psiAHat``; ``_rHat`` = :math:`r/\bar R`;
  ``_rN`` = :math:`r/a`.

There is **no explicit** ``radialCurrent`` **dataset**: ambipolarity is the
post-processing condition :math:`\sum_s Z_s\,\Gamma_s = 0` on the per-species
``particleFlux_*`` outputs (:mod:`sfincs_jax.er`, :doc:`physics_reference`).

Regression coverage
-------------------

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

The CLI supports direct plotting from any existing ``sfincsOutput.h5``,
``sfincsOutput.nc``, or ``sfincsOutput.npz``:

.. code-block:: bash

   sfincs_jax --plot sfincsOutput.h5

By default this writes ``<input>_summary.pdf`` next to the output file. Use
``plot-output --out`` to choose a different filename.

For a minimal end-to-end plotting example from the repository, run:

.. code-block:: bash

   python examples/getting_started/plot_sfincs_output.py

The script and the CLI both call the same plotting helper. PDF output writes a
multi-page panel following the diagnostics most often used in SFINCS and
neoclassical-transport papers:

- ``FSABFlow_vs_x``
- ``particleFlux_vm_psiHat`` and ``particleFlux_vm_psiHat_vs_x``
- ``heatFlux_vm_psiHat_vs_x``
- ``momentumFlux_vm_psiHat``
- ``NTV`` and ``NTVBeforeSurfaceIntegral``
- ``densityPerturbation``, ``pressurePerturbation``, ``flow`` and ``jHat``
- ``transportMatrix`` when present
- ``BHat(theta, zeta)``

.. note::

   ``uHat`` depends on many transcendental evaluations (cos/sin) and long floating-point
   reductions. In practice we observe tiny platform-dependent differences vs the frozen
   Fortran fixture (absolute errors :math:`\sim 10^{-9}` in the small scheme-4 test case),
   so the parity test compares ``uHat`` with a slightly looser tolerance than most other
   datasets.

.. _fortran-layout:

Fortran vs Python array layout
------------------------------

Fortran writes arrays in column-major order. When those HDF5 datasets are read back in
Python, multi-dimensional arrays often appear with axes reversed relative to the
``(itheta, izeta, ...)`` indexing used in the Fortran source.

To make it easy to do *file-to-file* comparisons in Python, `sfincs_jax` writes arrays
using the same convention by default (see `sfincs_jax.io.write_sfincs_h5`).
