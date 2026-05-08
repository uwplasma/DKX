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
Solver integration should happen only in later opt-in branches after these
primitives are trusted.

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

These primitives do not make the full Fokker-Planck collision operator
differentiable. The current full-FP setup contains NumPy/SciPy precomputations,
including Rosenbluth-potential quadrature and interpolation matrices. A true
mapped full-FP path requires a separate JAX-native collision-precompute
research branch.

Near-term acceptance tests
--------------------------

Before solver integration, the primitive layer must pass:

- monotonicity and positive-Jacobian tests;
- polynomial derivative-matrix tests;
- quadrature sanity checks;
- AD/finite-difference checks for nodes, weights, derivative matrices, and
  regularization diagnostics;
- conditioning checks for clustered-node maps.
