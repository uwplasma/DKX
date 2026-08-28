Source-code map
===============

This page links the main pieces of the mathematics to the source files that
implement them. The goal is to shorten the path from an equation in the docs
to the exact module that evaluates it.

High-level flow
---------------

For a standard solve, the execution path is:

1. parse namelist and resolve equilibrium inputs (``inputs.py``,
   ``namelist.py``, ``input_compat.py``, ``paths.py``),
2. build grids and geometry coefficients (``phase_space.py``, ``xgrid.py``,
   ``magnetic_geometry.py``),
3. construct the drift-kinetic operator (``drift_kinetic.py`` with the
   collision operators from ``collisions.py`` and species data from
   ``species.py``),
4. solve the linear (or Phi1 Newton) system (``solve.py``, ``phi1.py``),
5. compute moments and diagnostics (``moments.py``),
6. write ``sfincsOutput.h5``/``.nc``/``.npz`` plus the Fortran-parity console
   flow (``writer.py``, ``console.py``, ``solver_trace.py``), orchestrated end
   to end by ``run.py``.

Package layout
--------------

The package is a flat set of canonical, physics-named root modules, plus
one level of domain folders below ``dkx/`` for orchestration-only
code:

- ``dkx/validation`` for frozen-reference loading, Fortran/PETSc
  fixture readers, release-data manifests, validation artifacts, and the
  release/benchmark command-line tooling.
- ``dkx/workflows`` for scan orchestration (``scans.py``), optimization
  support (``optimization.py``), and JAX-native geometry adapters for external
  equilibrium producers (``geometry_adapters.py``).

Canonical root modules
----------------------

Physics and numerics:

- ``constants.py``: v3 normalizations, radial-coordinate conversions.
- ``species.py``: species pytrees, gradients, collisionality helpers.
- ``phase_space.py``: theta/zeta/x grids, derivative matrices, Legendre pitch
  machinery, ``Nxi_for_x`` ramps (``createGrids.F90``,
  ``uniformDiffMatrices.F90``, ``polynomialDiffMatrices.F90``).
- ``xgrid.py``: the Landreman–Ernst polynomial speed-grid kernel consumed by
  the collision operators (``xGrid.F90``).
- ``magnetic_geometry.py``: every supported geometry scheme, VMEC ``wout`` and
  Boozer ``.bc`` readers, the differentiable Fourier constructor
  (``geometry.F90``).
- ``collisions.py``: pitch-angle scattering and full Fokker–Planck with
  Rosenbluth potentials.
- ``drift_kinetic.py``: the matrix-free ``KineticOperator`` — streaming,
  mirror, ExB, Er xDot/xiDot, tangential magnetic drifts, collisions, sources,
  constraints, RHS drives (``populateMatrix.F90``, ``evaluateResidual.F90``).
- ``solve.py``: the three-tier solve policy (structured block elimination,
  recycled Krylov with a coarse-operator preconditioner, host direct referee)
  on the external ``solvax`` library; implicit differentiation.
- ``coarse_precond.py``: the tier-2 coarse preconditioner ``solve.py`` uses by
  default — the SFINCS-simplified operator, the three pins that make its chain
  invertible, and the routing between its dense-band, reusable Schur-LU and
  one-shot storage policies by what fits in memory
  (``preconditioner.F90``).  ``multigrid.py`` and ``sparse_precond.py`` are its
  drop-in alternatives and reproduce its pins.
- ``multigrid.py``: the semicoarsened geometric-multigrid tier-2
  preconditioner selected by ``solve(preconditioner="multigrid")``:
  rediscretized coarse operators on coarsened ``(theta, zeta[, xi])`` grids,
  pitch-collocation and Legendre-plane relaxations, and the block-Thomas solve
  on the coarsest grid; plus the pitch-basis diagnostics
  (``pitch_collocation_surrogate``, ``line_diagonal_dominance``,
  ``line_smoother_spectral_radius``) that measure why that route cannot reach
  tolerance on a Legendre-modal pitch discretization.
- ``sparse_precond.py``: the same simplified operator as ``multigrid.py``, but
  inverted exactly in a fill-reducing elimination order instead of eliminating
  ``L`` first — CSR assembly, host SuperLU, and the Sherman-Morrison
  correction that keeps the dense ``l = 0`` pin out of the sparsity pattern.
  Selected by ``solve(preconditioner="sparse")``.
- ``vmec_ascii.py``: the LIBSTELL text form of a VMEC ``wout``, which
  upstream accepts for ``geometryScheme 5`` alongside NetCDF;
  ``magnetic_geometry.read_vmec_wout`` routes here by file signature.
- ``phi1.py``: the Phi1/quasineutrality Newton solve.
- ``moments.py``: velocity-space moments, flux families, transport matrices,
  NTV, classical transport (``diagnostics.F90``,
  ``classicalTransport.F90``).
- ``er.py``: ambipolar radial-electric-field root solves.

Input/output and orchestration:

- ``inputs.py`` / ``namelist.py`` / ``input_compat.py``: typed namelist with
  Fortran-cited defaults and validation, parsing, alias handling.
- ``config.py``: immutable native ``Case`` with versioned TOML/JSON readers,
  deterministic semantic IDs, declarative scan bounds, and schema generation.
- ``execution.py``: native physical normalization and ``Case`` execution;
  consumes typed fields directly and never serializes a SFINCS namelist.
- ``result.py``: immutable named-array ``Result``, reviewer certificate, and
  the versioned native NetCDF reader/writer.
- ``run.py``: end-to-end RHSMode 1/2/3 drivers and ``run_from_namelist``.
- ``writer.py``: the canonical ``sfincsOutput`` writer (all formats,
  geometry-only output, export_f, solver-trace sidecars).
- ``console.py``: byte-parity Fortran stdout blocks.
- ``io.py``: output-file reading plus generic dict serializers.
- ``solver_trace.py``: the versioned solver-trace schema.
- ``api.py``, ``cli.py``, ``__main__.py``: the thin public surface.
- ``_version.py``: the single package and release version source.
- ``ambipolar.py``: scanplot-compatible ambipolar post-processing.
- ``sensitivity.py``: JVP/VJP, adjoint, and implicit differentiation helpers.
- ``compare.py``: HDF5 comparison and parity gates.
- ``plotting.py``: output plotting for the CLI and examples.
- ``paths.py`` / ``profiling.py``: path resolution and timing/memory probes.

Physics layers built on the canonical stack rather than inside it.  Each is a
separate owner because each answers a question the drift-kinetic solve does not:

- ``batch.py``: batched multi-surface / multi-``E_r`` solves, with the
  ``jax.lax.map`` chunk size set from the solver route's own memory model so a
  scan neither exhausts the device nor serializes to one solve at a time.
- ``monoenergetic.py``: the ``(nuPrime, EStar)`` monoenergetic database and its
  Maxwellian energy convolution, in the Beidler et al. benchmark normalization.
- ``bounce_averaged.py``: the collisionless ``1/nu`` limit -- effective ripple
  and the trapped-particle bounce integrals -- as a fast differentiable
  surrogate for the full solve.
- ``shaing_callen.py``: the collisionless bootstrap-coefficient limit, which
  anchors the low-collisionality physics tests.
- ``momentum_correction.py``: the Sugama-Nishimura parallel-momentum correction
  that restores inter-species momentum conservation lost by a pitch-angle
  collision operator.
- ``impurity.py``: classical impurity transport, screening, and the
  ``vmap``-over-charge-state API.
- ``variational.py``: entropy-production functionals that bound the
  monoenergetic coefficient from both sides -- a convergence certificate rather
  than a convergence claim.
- ``validity.py``: local-validity diagnostics (finite-orbit-width parameter,
  ``E x B`` resonance ratios, collisionality-regime classifier) that say when a
  radially-local answer should not be trusted.

Fortran-to-module correspondence
--------------------------------

===============================  =====================================
SFINCS v3 Fortran file           Canonical owner
===============================  =====================================
``globalVariables.F90``          ``constants.py``, ``species.py``
``createGrids.F90``              ``phase_space.py``
``xGrid.F90``                    ``xgrid.py``, ``phase_space.py``
``geometry.F90``                 ``magnetic_geometry.py``
``populateMatrix.F90``           ``drift_kinetic.py``, ``collisions.py``
``evaluateResidual.F90``         ``drift_kinetic.py``
``preconditioner.F90``           ``coarse_precond.py``
``solver.F90``                   ``solve.py``
``diagnostics.F90``              ``moments.py``, ``writer.py``
``classicalTransport.F90``       ``moments.py``
``readInput.F90``                ``inputs.py``, ``namelist.py``
``writeHDF5Output.F90``          ``writer.py``, ``io.py``
===============================  =====================================

The retired legacy pipeline (the transitional ``problems``, ``operators``,
``solvers``, ``outputs``, ``discretization``, ``geometry``, and ``physics``
packages) was deleted once every physics family became canonical; its parity
coverage lives on as Fortran-golden referees under ``tests/``.
