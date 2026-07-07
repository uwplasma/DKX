Fortran v3 example suite status
===============================

This page tracks the vendored upstream example suite
(``examples/sfincs_examples``) from the perspective of **exact-input frozen-fixture
validation**.

For release-facing runtime, memory, and parity claims, use the full scaled
example-suite audit instead:

- ``tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas``

That audit is the one reflected in ``README.md`` and the release-facing parity
claims. This page answers a narrower question: whether a vendored upstream input
has an exact content match to a frozen in-repo Fortran fixture.

For each upstream example input we report:

1. whether ``sfincs_jax`` successfully writes ``sfincsOutput.h5`` for that exact input,
2. whether that exact input has a frozen Fortran output fixture in-repo and matches it,
3. and the reason when exact-input frozen-fixture parity is not verified.

How this table is maintained
----------------------------

The checked table is a release artifact. Its maintenance logic is:

- Runs ``sfincs_jax write-output`` semantics for each upstream example input.
- Matches each input against ``tests/ref/*.input.namelist`` by exact file content.
- If a matching frozen ``tests/ref/*.sfincsOutput.h5`` exists, compares datasets with
  ``compare_sfincs_outputs``.
- Otherwise marks the case as ``unverified`` with a concrete reason.

.. note::

   ``unverified`` on this page does not mean the example fails. It only
   means there is no exact content-matched frozen Fortran ``sfincsOutput.h5`` fixture for that
   specific vendored input file. The release-facing scaled suite runs all vendored
   examples plus ``additional_examples`` cleanly on CPU and GPU.

Output parity table
-------------------

.. include:: _generated/fortran_examples_output_status.rst

Complementary audit
-------------------

A broader capability audit (grid/geometry/write-output support classification)
is checked below as a release artifact. Regeneration helpers are kept on the
publication-audit branch rather than in the stable core.

.. include:: _generated/fortran_examples_table.rst

Archived reduced-suite artifacts
--------------------------------

The reduced-suite runner is still kept for faster local debugging, solver triage, and
historical milestone comparison. It is not the release-facing status page.

.. include:: _generated/reduced_suite_archive_note.rst

Historical reduced-runtime parity sweep (case-by-case)
------------------------------------------------------

For rapid parity iteration, we also keep a reduced-resolution sweep generated with:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py --timeout-s 30 --max-attempts 6

This workflow is still useful for fast local debugging and solver triage:

1. copies each upstream input into ``tests/reduced_upstream_examples/<case>/input.namelist``,
2. halves resolution axes adaptively until both Fortran and ``sfincs_jax`` runs are under 30s,
3. compares resulting ``sfincsOutput.h5`` files.

The practical table with per-case tolerance overrides is:

.. include:: _generated/reduced_upstream_suite_status.rst

The strict table with tolerance overrides ignored is:

.. include:: _generated/reduced_upstream_suite_status_strict.rst

Field-wise reduced-suite tolerances are stored in
``tests/reduced_inputs/*.compare_tolerances.json`` and applied automatically in
the practical report. The strict report ignores all of these overrides.

Detailed matrix-dump and PETSc operator triage is preserved on the research
audit branches. The stable repository keeps the reduced-suite output gates and
small frozen fixtures needed for reviewable parity evidence.

Promoted reduced-input fixtures
-------------------------------

Whenever a case reaches ``parity_ok`` in the reduced runner, its adapted input is promoted to:

``tests/reduced_inputs/<case>.input.namelist``

Promoted fixtures:

- ``tests/reduced_inputs/HSX_FPCollisions_DKESTrajectories.input.namelist``
- ``tests/reduced_inputs/filteredW7XNetCDF_2species_magneticDrifts_noEr.input.namelist``
- ``tests/reduced_inputs/filteredW7XNetCDF_2species_magneticDrifts_withEr.input.namelist``
- ``tests/reduced_inputs/geometryScheme4_2species_noEr.input.namelist``
- ``tests/reduced_inputs/inductiveE_noEr.input.namelist``
- ``tests/reduced_inputs/monoenergetic_geometryScheme11.input.namelist``
- ``tests/reduced_inputs/monoenergetic_geometryScheme5_ASCII.input.namelist``
- ``tests/reduced_inputs/monoenergetic_geometryScheme5_netCDF.input.namelist``
- ``tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist``

These are intended as fast, reusable parity gates for CI and local development, not as the
release-facing source of truth.

Historical blocker typing used for reduced-case triage
------------------------------------------------------

The reduced runner classifies non-parity cases into:

- ``unsupported physics/path``
- ``geometry parsing mismatch``
- ``solver branch mismatch``
- ``output field mismatch``

and also records a compact print-parity score (shared runtime-log signals between
Fortran and ``sfincs_jax``) to track progress toward terminal-output parity.
