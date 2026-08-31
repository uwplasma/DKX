Speed grids (canonical) and the retired adaptive-map research topic
===================================================================

Canonical speed grids
---------------------

Every SFINCS v3 speed-grid option is canonical in :mod:`dkx.phase_space`
(``make_speed_grid``, ``xdot_diff_matrices``, ``make_grids``): ``xGridScheme``
1-8 (the Landreman–Ernst polynomial grids with and without a node at ``x=0``,
the uniform grids, and the Chebyshev variants) together with
``xDotDerivativeScheme`` -2..11 for the upwinded ``xDot`` derivative pairs.
The polynomial x-grid kernel consumed by the collision operators lives in
:mod:`dkx.xgrid`. Parity is pinned by Fortran goldens in
``tests/test_output_h5_xgrid_schemes_parity.py`` and the phase-space unit
tests.

Retired research topic
----------------------

The differentiable adaptive-map primitives (monotone maps
``x = g_theta(eta)`` with positive Jacobians, the opt-in ``xGridScheme = 50``
construction, and the mapped-grid transport-evidence workflow) were research
material hosted by the legacy ``discretization``/``workflows.mapped_xgrid``
owners. They were deleted with the legacy pipeline; the topic is preserved in
git history and may return through the checks recorded in
:doc:`research_lanes` as a differentiable-grid optimization workflow on the
canonical stack.
