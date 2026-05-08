Differentiable adaptive speed grids
===================================

``sfincs_jax.adaptive_maps`` contains opt-in research primitives for
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

The current production grid construction in ``sfincs_jax.v3`` remains unchanged.
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
