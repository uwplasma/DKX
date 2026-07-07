Differentiable adaptive speed grids
===================================

``sfincs_jax.discretization.adaptive_maps`` contains opt-in research primitives for
differentiable maps of the normalized speed coordinate ``x = v / v_th``.
They do not change the default SFINCS-v3-compatible grids or solver paths.

Scope
-----

The first implementation is intentionally standalone. It provides:

- fixed reference nodes ``eta`` on ``[0, 1]``;
- monotone maps ``x = g_theta(eta)`` with positive Jacobian;
- plain ``dx`` quadrature weights;
- first and second derivative matrices on the mapped nodes;
- smoothness, width-ratio, and tail diagnostics;
- automatic-differentiation checks against finite differences in tests.

The current production grid construction in ``sfincs_jax.discretization.v3`` remains unchanged.
An opt-in research construction path is available through ``xGridScheme = 50``.
This path only changes the speed grid when explicitly requested in the namelist.
It does not change the default SFINCS-v3-compatible grids.

Map families
------------

The module includes three map families plus an affine test map:

``AffineXMap``
  Finite-interval map ``x = x0 + scale * eta`` used for identity-style tests.

``SoftplusCellXMap``
  Positive-width cell-center map. This is close to finite-cell velocity-grid
  optimization and is useful for bounded speed intervals.

``RationalTailXMap``
  Semi-infinite map ``x = x0 + L eta / (1 - eta + eps)`` for Maxwellian tails.

``SplineDensityXMap``
  Smooth monotone map built from a positive density sampled on reference nodes.

Opt-in namelist keys
--------------------

The mapped-grid path is selected with:

.. code-block:: text

   &otherNumericalParameters
     xGridScheme = 50
     mappedXGridFamily = 'rational_tail'
   /

Supported families are ``affine``, ``rational_tail``, ``softplus_cell``, and
``spline_density``. Common optional keys include:

``mappedXGridEtaKind``
  ``gauss`` or ``uniform`` reference nodes on ``[0, 1]``.

``mappedXGridDerivative``
  ``barycentric`` or ``chain-rule`` derivative construction.

``mappedXGridLogLength`` / ``mappedXGridLogScale``
  Scale parameters for rational-tail and affine maps.

``mappedXGridXMax`` and ``mappedXGridParam(s)``
  Bounded extent and cell parameters for ``softplus_cell``.

``mappedXGridParams``
  Polynomial density coefficients for ``spline_density``.

For ``RHSMode = 3`` the existing monoenergetic override is preserved:
``x = 1`` and ``xWeights = exp(1)``.

Derivative matrices
-------------------

Two derivative constructions are provided:

``barycentric``
  Builds collocation derivative matrices directly on the mapped ``x`` nodes.

``chain-rule``
  Uses ``d/dx = J^{-1} d/deta`` and the corresponding second-derivative
  transform. This is useful for smooth maps with reliable Jacobians.

Transport moment objectives
---------------------------

``sfincs_jax.workflows.mapped_xgrid`` adds the first transport-facing objective
layer. It does not solve the drift-kinetic equation. Instead, it scores mapped
speed grids by how accurately they integrate Maxwellian speed moments,

.. math::

   I_p = \int_0^\infty x^p e^{-x^2}\,dx
       = {1 \over 2}\Gamma\!\left({p+1 \over 2}\right),

which are the building blocks of velocity-weighted neoclassical transport
integrals. The default proxy minimizes relative errors in low-order moments such
as ``p = 2, 4, 6`` and can add smoothness, Jacobian-roughness, tail, and
conditioning penalties from the mapped-grid diagnostics.

This objective gives fast AD/finite-difference and baseline-tuning tests before
attempting expensive implicit-solve objectives. It should be interpreted as a
screening metric for speed-grid candidates, not as evidence that a mapped grid
improves a full SFINCS transport coefficient until the solve-level branch
demonstrates that directly.

PAS transport-matrix evidence
-----------------------------

``sfincs_jax.workflows.mapped_xgrid`` adds the first solve-facing
comparison layer for the opt-in mapped grid. It copies a namelist, sets
``xGridScheme = 50`` with rational-tail map parameters, runs a transport-matrix
solve for each candidate, and compares each result against a reference solve.
The report records the proxy moment objective, mapped-grid conditioning
diagnostics, residual norms, elapsed time, and transport-matrix error,

.. math::

   \epsilon_T =
   {\left\|T_{\mathrm{mapped}} - T_{\mathrm{ref}}\right\|_F
    \over
    \max\left(\left\|T_{\mathrm{ref}}\right\|_F, 10^{-300}\right)} .

This layer is intentionally conservative. It is designed to test whether the
cheap moment objective predicts a useful transport-matrix grid, not to replace
the transport solve itself. It targets the PAS path because mapped
``xGridScheme = 50`` is not yet compatible with the full-FP collision
precompute assumptions.

For reviewer-facing evidence, use ``reference_nml`` or a precomputed
``reference_result`` from a higher-resolution default-grid solve, then compare
lower-resolution mapped candidates against that reference. Comparing only
same-resolution grids is a useful smoke test but is not enough to support a
resolution-reduction claim.

QI seed-robustness smoke
------------------------

QI seed-robustness runner code, data, and production-promotion commands are
preserved on the ``research/qi-device-hard-seed`` branch. The stable core keeps
historical JSON artifacts under ``docs/_static`` as research evidence only; they
are not default CI commands and should not be cited as stable-core mapped-grid
acceptance gates.

Limitations
-----------

These primitives and the ``xGridScheme = 50`` construction path do not make the
full Fokker-Planck collision operator differentiable. The current full-FP setup
contains NumPy/SciPy precomputations, including Rosenbluth-potential quadrature
and interpolation matrices. A true mapped full-FP path requires a separate
JAX-native collision-precompute research branch.

Near-term acceptance tests
--------------------------

Before solver integration, the primitive layer must pass:

- monotonicity and positive-Jacobian tests;
- polynomial derivative-matrix tests;
- quadrature sanity checks;
- AD/finite-difference checks for nodes, weights, derivative matrices, and
  regularization diagnostics;
- conditioning checks for clustered-node maps.
