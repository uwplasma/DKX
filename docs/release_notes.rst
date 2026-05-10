Release notes
=============

v1.1.2
------

This patch release closes the post-v1.1.1 structured-PAS fallback push and
hardens the release workflow.

Highlights
~~~~~~~~~~

- Memory-unsafe PAS-TZ fallback routes are now bounded uniformly. The default
  ``hybrid`` fallback is marked with the same guarded metadata as opt-in
  ``theta``/``zeta`` structured fallbacks, so it skips the expensive automatic
  strong-preconditioner retry unless
  ``SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1`` is set.
- Guarded PAS-TZ and weak PAS forced/probe paths use accept-only matrix-free
  minimal-residual corrections before fail-fast classification. These
  corrections reuse the already-built preconditioner, store no dense angular
  patch inverses, and are kept only when the measured residual decreases.
- Forced weak PAS paths (``collision``, ``xmg``, and ``point``) now skip
  stage-2/strong retry only at enormous residual ratios by default, preventing
  minutes-long profiling stalls without changing moderate-residual polish
  behavior.
- Release metadata, production-benchmark workflow checks, and public
  runtime/validation figures were refreshed for the current ``main`` artifacts.
- The PyPI workflow now validates that ``pyproject.toml``,
  ``sfincs_jax.__version__``, and pushed release tags agree before publishing.

Validation
~~~~~~~~~~

- Current release-facing CPU/GPU suite artifacts remain ``39/39 parity_ok`` with
  zero strict mismatches, no ``jax_error``, and no ``max_attempts``.
- Bounded PAS-TZ fallback smoke now returns for ``hybrid``, ``zeta``, and
  ``theta`` under the 15 s local gate. These rows are intentionally documented as
  negative, non-promoted baselines because their residuals remain large.
- The bounded artifact is checked in at
  ``tests/reference_solver_path_artifacts/pas_tz_memory_fallback_geometry4_smoke_2026-05-10.json``
  and guarded by ``tests/test_solver_path_artifacts.py``.
- Local release validation passed with ``1131 passed in 506.22 s``.

Remaining research lane
~~~~~~~~~~~~~~~~~~~~~~~

No release blocker remains in the documented workflows. The remaining
publication-scale optimization lane is still algorithmic: a genuinely stronger
matrix-free line/plane smoother or iterative chunked Schwarz correction for
geometry-rich PAS systems that clears the fixed gate of under 60 s, no measured
memory regression, and at least 100x residual reduction before any default
promotion.

v1.1.1
------

This patch release ships the final PAS/full-FP performance and memory closeout
after the v1.1.0 validation release.

Highlights
~~~~~~~~~~

- One-species PAS+Er sparse-PC defaults now use the measured
  ``MMD_AT_PLUS_A`` ordering and a bounded GMRES restart policy unless the user
  explicitly overrides the restart environment variable.
- Phi1 fast-explicit solves use a production-size restart helper that preserves
  output parity while reducing wasted Krylov storage on larger active systems.
- RHSMode 1 no-Phi1 single-state output avoids retaining an unnecessary stacked
  solved-distribution copy before diagnostic writeout.
- The README-facing runtime/memory and W7-X high-``nu`` performance figures were
  regenerated from the checked-in release artifacts.
- The production-resolution ``geometryScheme4_2species_PAS_noEr`` stress case is
  now explicitly closed for this release as ``no safe existing default
  promotion``. CPU and GPU candidate routes all hit the bounded 300 s gate, so no
  unsafe solver-path default is promoted.

Validation
~~~~~~~~~~

- Local full suite: ``1115 passed in 498.10 s``.
- GitHub Actions for the closeout commit: CI and Docs both passed.
- The large geometry-rich PAS closeout artifact is checked in at
  ``tests/reference_solver_path_artifacts/geometry4_large_pas_closeout_2026-05-09.json``
  and guarded by ``tests/test_solver_path_artifacts.py``.

Remaining research lane
~~~~~~~~~~~~~~~~~~~~~~~

No release blocker remains. The next research optimization target is a
structured/chunked geometry-aware PAS preconditioner for production-resolution
geometry-rich 3D cases; heuristic promotion of existing Schur, sparse-PC, or
PAS-lite paths is intentionally blocked until a measured route clears the
runtime, memory, residual, and Fortran-comparison gates.

v1.1.0
------

This release promotes the current CPU/GPU validation and performance work into the
first minor release after the 1.0 series.

Highlights
~~~~~~~~~~

- The audited 39-case example suite remains clean on CPU and GPU: no practical
  mismatches, no strict mismatches, no ``jax_error`` cases, and no ``max_attempts``
  cases in the release-facing artifacts.
- GPU solver-path selection is less aggressive for bounded full-collision and
  PAS systems. Moderate full-FP systems can stay on dense accelerator solves when
  that is faster and lower-memory, while bounded geometry-rich PAS examples now
  prefer the measured lower-memory ``pas_tz`` path. On CPU, audited 3D full-FP
  RHSMode 1 cases can auto-select sparse-PC GMRES inside the measured size
  window when it beats dense FP on both runtime and memory. On GPU/CUDA,
  production-floor tokamak full-FP no-Er/Er rows can auto-select sparse-PC GMRES
  inside narrow measured windows when the matrix-free route is not residual-clean
  and the faster theta-line route is too memory-heavy.
- Monoenergetic transport benchmarks now time the actual RHSMode 2/3 transport
  solve instead of only output-field assembly, and small bounded GPU cases can use
  dense accelerator transport when it is validated to be faster.
- The CLI and Python output paths support HDF5, NetCDF4, and NPZ by output suffix,
  and ``sfincs_jax --plot`` writes a PDF diagnostics panel from existing output
  files.
- Documentation covers the drift-kinetic equation being solved, geometry loading,
  normalizations, solver paths, output datasets, validation gates, performance
  techniques, and release-maintainer checks.

Validation artifacts
~~~~~~~~~~~~~~~~~~~~

The release-facing validation roots are:

- ``tests/scaled_example_suite_release_cpu_frozen_2026-04-25_v106``
- ``tests/scaled_example_suite_gpu_bounded_default_2026-04-28``

The latest focused GPU performance pass measured:

- ``HSX_PASCollisions_fullTrajectories``: ``10.539 s`` / ``2042 MB`` to
  ``8.469 s`` / ``1577 MB``, with zero Fortran mismatches.
- ``sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories``:
  ``7.716 s`` / ``2098 MB`` to ``6.413 s`` / ``1609 MB``, with zero Fortran
  mismatches.
- ``monoenergetic_geometryScheme1``: ``13.039 s`` / ``996 MB`` to ``3.541 s`` /
  ``981 MB``, with zero Fortran mismatches.

Remaining research lanes
~~~~~~~~~~~~~~~~~~~~~~~~

No correctness blocker remains in the documented release-facing suite. The main
future optimization lane is allocator and work-array lifetime reduction for the
heaviest RHSMode 1 PAS Krylov/diagnostic paths. Single-case strong multi-GPU
scaling remains a research feature; release-facing parallel guidance continues to
prefer case-parallel and transport-worker throughput.
