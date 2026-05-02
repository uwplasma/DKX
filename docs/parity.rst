.. This page is kept for backward compatibility with existing links and scripts.

:orphan:

Parity status
=============

High-level summary (parity-tested)
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Area
     - Status
     - Notes
   * - Grids (``theta``, ``zeta``, ``x``)
     - Yes
     - Includes monoenergetic ``x=1`` / ``xWeights=exp(1)`` special-case
   * - Geometry schemes ``1/2/4``
     - Yes
     - Output parity fixtures
   * - Geometry scheme ``5`` (VMEC ``wout_*.nc``)
     - Yes
     - End-to-end output parity in the current full example-suite audit, plus frozen fixture coverage
   * - Geometry schemes ``11/12`` (Boozer ``.bc``)
     - Yes
     - Geometry + transport-matrix end-to-end fixtures
   * - Linear runs (RHSMode=1)
     - Yes
     - Explicit CPU/GPU release lanes are parity-clean across the current vendored example suite
   * - Transport matrices (RHSMode=2/3)
     - Yes
     - End-to-end ``sfincsOutput.h5`` parity for 2×2 and 3×3 cases
   * - Full upstream v3 example suite
     - Yes
     - Current ``main`` release audit is ``39/39 parity_ok`` on CPU and ``39/39 parity_ok`` on GPU, with no strict mismatches, no ``jax_error``, no ``max_attempts``, and zero missing Fortran top-level output keys in JAX.

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
  and the monoenergetic (``RHSMode=3``) special-case ``x=1`` / ``xWeights=exp(1)`` grid used in v3 ``createGrids.F90``.
- Boozer geometryScheme=1 (three-helicity analytic model): ``BHat`` and derivatives (via output parity fixtures)
- Boozer geometryScheme=2 (simplified LHD model): ``BHat`` and derivatives (via output parity fixtures)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives
- Boozer geometryScheme=11/12 from `.bc` file inputs: ``BHat``, ``DHat``, ``BHat_sub_psi``, and derivatives (parity vs frozen fixture)
- VMEC geometryScheme=5 from ``wout_*.nc`` inputs: core geometry arrays (``BHat``, ``DHat``, covariant/contravariant components)
  and ``gpsiHatpsiHat`` (parity vs frozen fixture)
- ``sfincsOutput.h5`` writing for ``geometryScheme in {1,2,4,5,11,12}`` with dataset-by-dataset parity against frozen
  Fortran v3 fixtures (see ``docs/outputs.rst``). ``uHat`` is compared with a looser tolerance due to tiny
  platform-dependent transcendental/reduction differences.
- Classical transport (`calculateClassicalFlux`) for geometries with `gpsiHatpsiHat` support:
  `geometryScheme=5` (VMEC) and `geometryScheme=11/12` (.bc) — parity-tested via frozen `sfincsOutput.h5` fixtures.
- Collisionless v3 operator slice: streaming + mirror (parity vs PETSc binaries for one example)
- Collisionless v3 Er terms:

  - non-standard ``d/dxi`` term (``includeElectricFieldTermInXiDot = .true.``): ΔL = ±2 parity vs Fortran Jacobian
  - collisionless ``d/dx`` term (``includeXDotTerm = .true.``): ΔL = ±2 parity vs Fortran Jacobian

- ExB drift term (``useDKESExBDrift = .false.``): ``d/dtheta`` parity vs Fortran Jacobian (geometryScheme=4)
- Magnetic drift terms (``magneticDriftScheme=1``): parity-tested as ΔL = ±2 slices vs Fortran Jacobian (geometryScheme=11)
- Pitch-angle scattering collisions (``collisionOperator=1`` without Phi1): diagonal parity vs PETSc binaries for one small example
- Full linearized Fokker-Planck collision operator (``collisionOperator=0`` without Phi1): F-block matvec parity vs a frozen
  PETSc matrix for a 2-species ``geometryScheme=4`` fixture (``tests/ref/quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin``).
- Full-system matvec parity (includePhi1=false, constraint schemes 1/2) vs frozen PETSc matrices for:
  ``pas_1species_PAS_noEr_tiny`` and ``quick_2species_FPCollisions_noEr``.
- Full-system matvec + RHS + residual + GMRES-solution parity for VMEC ``geometryScheme=5`` (tiny PAS case):
  ``pas_1species_PAS_noEr_tiny_scheme5``.
- Full-system matvec + RHS + residual + GMRES-solution parity for VMEC ``geometryScheme=5`` with Phi1 QN/lambda blocks:
  ``pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear``.
- Full-system matvec + RHS + residual + GMRES-solution parity for ``geometryScheme=1`` (tokamak-like, Nzeta=1):
  ``pas_1species_PAS_noEr_tiny_scheme1``.
- Transport-matrix modes (``RHSMode=2/3``):

  - v3 internal ``whichRHS`` RHS settings parity (RHS/residual at ``f=0``) for ``RHSMode=2`` and ``RHSMode=3``.
  - Monoenergetic special-case ``x=1`` / ``xWeights=exp(1)`` (v3 ``createGrids.F90``) parity.
  - Full-system matvec parity vs v3 solver matrix (``whichMatrix=1``) for tiny monoenergetic fixtures in:
    ``monoenergetic_PAS_tiny_scheme1``, ``monoenergetic_PAS_tiny_scheme11``, and ``monoenergetic_PAS_tiny_scheme5_filtered``.
  - ``transportMatrix`` assembly parity vs frozen Fortran v3 ``sfincsOutput.h5`` for:
    ``monoenergetic_PAS_tiny_scheme{1,11,12,5_filtered}`` (``RHSMode=3``) and
    ``transportMatrix_PAS_tiny_rhsMode2_scheme2`` (``RHSMode=2``).
- Phi1/QN/lambda block parity (includePhi1=true, includePhi1InKineticEquation=false):
  full-system matvec + GMRES solution parity vs frozen PETSc binaries for
  ``pas_1species_PAS_noEr_tiny_withPhi1_linear``.
- Phi1 in kinetic equation parity (includePhi1=true, includePhi1InKineticEquation=true):
  full-system matvec + GMRES solution parity vs frozen PETSc binaries for
  ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Nonlinear end-to-end solve (experimental Newton–Krylov) parity for:
  ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Full-system RHS and residual assembly parity vs frozen Fortran v3 `evaluateResidual.F90` binaries for:
  ``pas_1species_PAS_noEr_tiny``, ``quick_2species_FPCollisions_noEr``,
  ``pas_1species_PAS_noEr_tiny_withPhi1_linear``, and ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Full linearized Fokker-Planck collisions with Phi1 in the collision operator
  (``collisionOperator=0``, ``includePhi1InCollisionOperator=true``) parity-tested as full-system matvec + residual for:
  ``fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision``.

Current scope limits
--------------------

- The release-facing parity claim is the current full example-suite audit:

  - ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106``
  - ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28``

  The older reduced-suite artifacts remain useful for debugging, fixture history, and faster local
  triage, but they are no longer the primary release status.
- The unconstrained ``constraintScheme=0`` branch is rank-deficient, so different solvers can select different nullspace
  components. For comparisons, sfincs_jax treats a small set of density/pressure-like outputs as gauge-dependent and
  skips them when ``constraintScheme=0`` (see ``sfincs_jax/compare.py``).
- Constrained PAS systems can also be branch-sensitive in current/flow
  diagnostics if a reference stops on a preconditioned residual while the true
  residual is still large.  These rows are treated as reference-quality blockers,
  not generic solver failures.  The compact regression fixture
  ``tests/reference_solver_path_artifacts/constrained_pas_branch_probe_2026-05-02.json``
  guards that exact true-residual, PETSc-compatible minimum-norm, and weak
  preconditioned-residual branches remain explicitly labeled.
- The default CLI and ``write-output`` path use an explicit performance-oriented solve strategy.
  End-to-end differentiable solves remain available from Python via the implicit/differentiable path when requested.
- Full Phi1 coupling end-to-end (nonlinear residual assembly + collision operator contributions) is still being expanded beyond the currently parity-tested subset.
- VMEC-based geometry schemes beyond the current ``geometryScheme=5`` parity subset.
- Rosenbluth response matrices for FP cross-species coupling are computed with QUADPACK (matching v3). We added strict
  scalar-order accumulation for the collocation-to-modal projection, but the remaining ~1e-10 deltas appear dominated by
  quadrature rounding differences rather than matrix-ordering effects.
- VMEC geometryScheme=5 full Fokker–Planck fixtures exhibit small (~1e-6 absolute) differences in local flow/Mach
  diagnostics at isolated grid points. These deltas are well below the physics tolerance but can trip strict relative
  checks when the true value is near zero, so we apply a dedicated absolute-floor override for the VMEC FP subset in
  ``sfincs_jax/compare.py``. The current release-facing CPU and GPU example-suite audits remain strict-clean.

Near-zero tolerances
--------------------

Some diagnostics are expected to be very close to zero in specific regimes, so strict relative
tolerances can overstate differences. ``sfincs_jax.compare.compare_sfincs_outputs`` applies small
absolute floors for near-zero fields in these cases (e.g., RHSMode=1 constraintScheme=1/2 flow,
pressure, and ``delta_f`` diagnostics; monoenergetic density/pressure moments; and RHSMode=2/3
``sources`` terms).
These built-in floors are documented in ``sfincs_jax/compare.py`` and are always active in
practical parity checks; strict mode ignores per-case JSON overrides but still respects these
near-zero safeguards.

For ``includePhi1 = .true.`` runs, parity compares the converged (last) Newton iterate when
datasets include an iteration axis, even if Fortran and sfincs_jax record different
``NIterations`` metadata. This keeps parity focused on the final physical state instead of
intermediate Newton-history bookkeeping.

Release-facing parity status (source of truth)
----------------------------------------------

The release-facing parity inventory is the full current example-suite audit:

- ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106/suite_report.json``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28/suite_report.json``
- ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106/suite_output_key_coverage_summary.json``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28/suite_output_key_coverage_summary.json``

Use these artifacts for README and release claims. The reduced upstream parity inventory remains
useful for faster debugging and historical comparison:

- ``docs/_generated/reduced_upstream_suite_status.rst``
- ``docs/_generated/reduced_upstream_suite_status_strict.rst``
- ``tests/reduced_upstream_examples/suite_report.json``
- ``tests/reduced_upstream_examples/suite_report_strict.json``

Regenerate the full release-facing suite:

.. code-block:: bash

   python scripts/run_scaled_example_suite.py \
     --examples-root examples/sfincs_examples \
     --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
     --reference-results-root tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix \
     --out-root tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106 \
     --scale-factor 1.0 \
     --runtime-target-basis fortran \
     --fortran-min-runtime-s 0.0 \
     --runtime-adjustment-iters 0 \
     --runtime-baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
     --jax-profile-marks on

After a suite refresh, verify the structural output coverage explicitly:

.. code-block:: bash

   python scripts/audit_suite_output_keys.py \
     --suite-root tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106 \
     --fail-on-missing

When refreshing a frozen CPU lane, compare runtime against the previously promoted lane:

.. code-block:: bash

   python scripts/audit_suite_runtime_drift.py \
     --baseline-report tests/scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix/suite_report.json \
     --candidate-report tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106/suite_report.json \
     --threshold-ratio 1.25 \
     --min-baseline-runtime-s 1.0

For faster targeted debugging, regenerate the reduced-suite files:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py --timeout-s 120 --max-attempts 1

Reduced-suite default tolerances are ``rtol=5e-4`` and ``atol=1e-9``; override with ``--rtol``/``--atol``.

Target a single case family:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py \
     --pattern 'HSX_FPCollisions|filteredW7XNetCDF_2species_magneticDrifts|geometryScheme4_2species' \
     --timeout-s 120 --max-attempts 1

Matrix/operator parity diagnosis (Fortran PETSc matrix vs JAX matvec):

.. code-block:: bash

   python scripts/compare_fortran_matrix_to_jax_operator.py \
     --input /path/to/input.namelist \
     --fortran-matrix /path/to/sfincsBinary_iteration_000_whichMatrix_3 \
     --fortran-state /path/to/sfincsBinary_iteration_000_stateVector \
     --project-active-dofs \
     --out-json matrix_compare.json

Frozen-state diagnostics isolation (solver-vs-diagnostics for RHSMode=1 moment families):

.. code-block:: bash

   python scripts/compare_rhsmode1_diagnostics_from_state.py \
     --input /path/to/input.namelist \
     --state /path/to/sfincsBinary_iteration_000_stateVector \
     --fortran-h5 /path/to/sfincsOutput.h5 \
     --out-json diagnostics_from_frozen_state.json
