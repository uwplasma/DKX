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
the transport solve itself. It currently targets the PAS path because mapped
``xGridScheme = 50`` is not yet compatible with the full-FP collision
precompute assumptions.

For reviewer-facing evidence, use ``reference_nml`` or a precomputed
``reference_result`` from a higher-resolution default-grid solve, then compare
lower-resolution mapped candidates against that reference. Comparing only
same-resolution grids is a useful smoke test but is not enough to support a
resolution-reduction claim.

QI seed-robustness smoke
------------------------

``scripts/run_qi_seed_robustness.py`` materializes deterministic neighboring
cases from ``examples/additional_examples/input.namelist``. Each case localizes
the QI VMEC equilibrium beside its generated ``input.namelist``, applies
seed-derived ``nu_n`` and ``Er`` perturbations, and records the exact
``sfincs_jax write-output`` command in ``manifest.json``.

Use ``--execute`` for a bounded real solve smoke. For a promotable bounded
ladder, keep the default ``auto`` CLI policy, add residual/convergence gates,
and write a compact artifact from the generated ``manifest.json``:

.. code-block:: bash

   JAX_PLATFORM_NAME=cpu python scripts/run_qi_seed_robustness.py \
     --out-root tests/qi_seed_robustness_multiseed5_cpu \
     --seeds 0 1 2 3 4 \
     --resolution-scale 0.25 \
     --min-nzeta 13 \
     --min-nxi 25 \
     --execute \
     --timeout-s 120 \
     --max-residual-ratio 1 \
     --require-converged \
     --summary-output docs/_static/qi_seed_robustness_multiseed5_cpu.json \
     --clean

Refresh the production-readiness manifest without rerunning solves:

.. code-block:: bash

   python scripts/run_qi_seed_robustness.py \
     --summarize-artifacts-only \
     --evidence-manifest-output docs/_static/qi_seed_robustness_evidence_manifest.json

The QI lane defaults to the public ``auto`` CLI solver policy. The bounded
checked multi-seed CPU and one-GPU artifacts cover three neighboring seeds at
``7 x 13 x 25 x 4`` and verify that ``auto`` uses the fast dense full-FP path
before entering the sparse/fallback ladder. The current five-seed CPU artifact
extends that same low-resolution ladder to seeds ``0..4`` with ``process_failed=0``,
``timed_out=0``, ``outputs_written=5``, ``solver_traces_written=5``,
``converged=5``, and maximum residual ratio ``7.88e-7``. A larger single-seed
``9 x 19 x 35 x 4`` artifact in
``docs/_static/qi_seed_robustness_scale035_cpu_gpu.json`` validates the next
policy tier: at ``13169`` active unknowns, CPU converges in ``29.8 s`` and one
RTX A4000 GPU converges in ``42.8 s`` through the bounded accelerator
host-sparse rescue. Before that rescue was enabled, the same GPU case spent
``195 s`` in the Krylov/fallback tail and was rejected with residual ratio
``53.9``. The manifest records stdout/stderr paths, return codes, output and
solver-trace presence, and a compact solver-trace summary including residual
norm, residual target, residual ratio, and convergence flags.

The checked production-readiness manifest is
``docs/_static/qi_seed_robustness_evidence_manifest.json``. It keeps the lane at
``bounded_proxy`` because the largest checked passing grid is still ``139502``
estimated unknowns versus ``510002`` at the exact production-floor seed target
``25 x 51 x 100 x 4``. The largest attempted grid is the exact production-floor
``510002``-unknown seed-0 CPU/GPU timeout probe; it includes passing scale-0.60
CPU/GPU seed-0 evidence, passing scale-0.60 CPU five-seed evidence, rejected
scale-0.60 GPU hard-seed solver/global-coupling/device-Krylov probes, and the
exact-floor blocker runs. The bounded lane-completion estimate is therefore
``60%`` by the smallest per-axis resolution fraction of the largest passing
artifact, while ``72.65%`` of the production total-size estimate remains
uncovered.

Production-resolution promotion requires both scheduled ladders below to pass
before changing the gate status:

.. code-block:: bash

   JAX_PLATFORM_NAME=cpu python scripts/run_qi_seed_robustness.py \
     --out-root tests/qi_seed_robustness_prod_cpu \
     --seeds 0 1 2 3 4 \
     --resolution-scale 1.0 \
     --target-ntheta 25 \
     --target-nzeta 51 \
     --target-nx 4 \
     --target-nxi 100 \
     --execute \
     --timeout-s 3600 \
     --max-residual-ratio 1 \
     --require-converged \
     --summary-output docs/_static/qi_seed_robustness_prod_cpu.json \
     --clean

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=gpu python scripts/run_qi_seed_robustness.py \
     --out-root tests/qi_seed_robustness_prod_gpu0 \
     --seeds 0 1 2 3 4 \
     --resolution-scale 1.0 \
     --target-ntheta 25 \
     --target-nzeta 51 \
     --target-nx 4 \
     --target-nxi 100 \
     --execute \
     --timeout-s 3600 \
     --max-residual-ratio 1 \
     --require-converged \
     --summary-output docs/_static/qi_seed_robustness_prod_gpu0.json \
     --clean

Acceptance is machine-readable: each production artifact must report
``public_cli_default_path=true``, ``solve_methods=["auto"]``, ``process_failed=0``,
``timed_out=0``, ``outputs_written=5``, ``solver_traces_written=5``,
``converged=5``, and ``max_residual_ratio <= 1``. Treat all current artifacts as
bounded integration evidence; do not claim full QI robustness until both CPU and
GPU production-resolution artifacts are checked in and the evidence manifest is
regenerated.

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
