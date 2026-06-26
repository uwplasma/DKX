Numerics and algorithms
=======================

`sfincs_jax` solves a large structured linear system that comes from discretizing the
radially local drift-kinetic equation and its auxiliary constraints on a single flux
surface. This page focuses on the numerical representation of that system and the
algorithms used to solve it efficiently on CPU and GPU.

Discrete unknowns
-----------------

For each kinetic species, the unknown first-order distribution is represented on the
tensor grid

.. math::

   (x_i, L, \theta_j, \zeta_k),

where:

- :math:`x=v/v_\mathrm{th}` is the normalized speed coordinate,
- :math:`L` is the Legendre index in :math:`\xi=v_\parallel/v`,
- :math:`\theta` and :math:`\zeta` are straight-field-line angular coordinates.

The unknown vector therefore contains, schematically,

.. math::

   f_{s1}(x_i, L, \theta_j, \zeta_k),

plus optional :math:`\Phi_1(\theta,\zeta)` unknowns and constraint/source coefficients.

The total number of degrees of freedom scales like

.. math::

   N_\mathrm{dof}
   \sim
   N_\mathrm{species}\, N_x\, N_L\, N_\theta\, N_\zeta
   + N_\theta N_\zeta
   + N_\mathrm{constraints}.

This scaling is why solver design, preconditioning, and memory layout matter.

Angular discretization
----------------------

The angular coordinates are discretized on periodic grids. First derivatives are
represented by dense differentiation matrices acting along the corresponding axis:

.. math::

   \partial_\theta f \approx D_\theta f,
   \qquad
   \partial_\zeta f \approx D_\zeta f.

For advection-dominated magnetic-drift terms, `sfincs_jax` can also use directional
upwind derivative matrices selected pointwise from the sign of the local drift
coefficient. This is important at low collisionality, where the trapped-passing
boundary and narrow angular structures can otherwise produce poor Krylov behavior.

Velocity-space discretization
-----------------------------

The speed coordinate uses the same polynomial-grid philosophy as the mature SFINCS
formulation: collocation points in :math:`x` together with quadrature weights and
modal transforms tailored to Maxwellian-weighted integrals. The pitch-angle dependence
is expanded in Legendre modes,

.. math::

   f(x,\xi,\theta,\zeta)
   =
   \sum_{L=0}^{N_L-1} f_L(x,\theta,\zeta) P_L(\xi).

This representation has two major numerical consequences:

- streaming and mirror terms couple :math:`L \leftrightarrow L\pm 1`,
- several drift terms couple :math:`L \leftrightarrow L\pm 2`,

while collision operators are diagonal in :math:`L` for PAS and dense in :math:`x`
for full linearized Fokker-Planck.

Linear-system structure
-----------------------

After discretization, the linear problem can be written as

.. math::

   A u = b,

with the block structure

.. math::

   A =
   \begin{bmatrix}
     A_{ff} & A_{f\Phi} & A_{fc} \\
     A_{\Phi f} & A_{\Phi\Phi} & A_{\Phi c} \\
     A_{cf} & A_{c\Phi} & A_{cc}
   \end{bmatrix},

where:

- :math:`A_{ff}` is the kinetic block,
- :math:`A_{\Phi\Phi}` is the quasineutrality / potential block when active,
- :math:`A_{cc}` and the off-diagonal constraint blocks impose density, energy,
  and gauge conditions.

The explicit matrix is usually too large or too wasteful to assemble densely, so the
main production path uses matrix-free operator application.

Matrix-free operator evaluation
-------------------------------

The central numerical design choice in `sfincs_jax` is to express the kinetic operator
as a composition of tensor contractions, sparse/dense directional derivative
applications, collision blocks, and constraint evaluations rather than as one global
assembled sparse matrix.

The advantages are:

- the operator can be JIT-compiled for CPU or GPU,
- the same operator can be differentiated when the differentiable path is selected,
- and large solve branches can stay on device until a rescue path is actually needed.

In the source tree, the core operator assembly and cached application live in
``sfincs_jax/operators/profile_system.py``. RHSMode-1 solve orchestration lives in
``sfincs_jax/problems/profile_solve.py``; RHSMode-2/3 transport orchestration lives in
``sfincs_jax/problems/transport_solve.py``.

Solve modes
-----------

`sfincs_jax` intentionally separates two use cases:

- **CLI / production explicit path**:
  tuned for robustness, throughput, and bounded memory use.
- **Python differentiable path**:
  preserves JAX-native solve structure when end-to-end derivatives matter.

This split avoids forcing the public executable into the same algorithmic constraints
as the differentiable research workflow.

Krylov methods and preconditioners
----------------------------------

The dominant linear solves are nonsymmetric and often ill-conditioned, so Krylov
methods are the baseline:

- GMRES and closely related variants for the main implicit solve path,
- bounded host sparse-direct or host dense-direct rescues when the problem is small
  enough or badly conditioned,
- structured PAS and collision-based preconditioners for the hard branches.

The practical preconditioner family includes:

- simplified PAS angular/velocity blocks,
- collision-only approximations,
- block-structured ``xblock`` and Schur-style reductions,
- sparse host factorizations for selected medium and large branches,
- and multilevel Schwarz corrections for sharded multi-device experiments.

The preconditioner is not required to be a physically exact operator. Its job is to
reduce the Krylov iteration count and stabilize convergence while preserving the
solution of the full system.

JAX-specific implementation choices
-----------------------------------

JAX is used where it gives concrete numerical leverage:

- **JIT compilation** removes Python overhead from repeated operator applications.
- **XLA fusion** reduces intermediate allocations for tensor-heavy kernels.
- **Device portability** keeps the same math on CPU and GPU.
- **Automatic differentiation** is available when the solve path stays within the
  supported differentiable subset.
- **Sharding / distributed execution** can be used for selected transport-worker and
  experimental sharded solves.

At the same time, `sfincs_jax` does not force every solve through a pure-JAX path.
When a bounded host sparse or dense solve is the right tool for the CLI, the code uses
it.

Code locations
--------------

The most important numerical modules are:

- ``sfincs_jax/operators/profile_system.py``: system definition, cached operators, block structure.
- ``sfincs_jax/problems/profile_solve.py``: RHSMode-1 solve orchestration,
  solver/preconditioner selection, and rescue policy.
- ``sfincs_jax/problems/transport_solve.py`` and
  ``sfincs_jax/problems/transport_parallel_runtime.py``: RHSMode-2/3 transport
  solves and parallel transport execution.
- ``sfincs_jax/operators/profile_linear_systems.py``: residual and
  right-hand-side evaluation.
- ``sfincs_jax/solver.py``: linear-solver wrappers and Krylov helpers.
- ``sfincs_jax/solvers/implicit.py``: differentiable linear solve path.
- ``sfincs_jax/physics/collisions.py``: PAS and full FP operator kernels.
- ``sfincs_jax/grids.py`` and ``sfincs_jax/discretization/xgrid.py``: collocation grids, quadrature,
  modal transforms.

Resolution guidance
-------------------

The most important practical resolution knobs are

.. math::

   N_\theta, \qquad N_\zeta, \qquad N_\xi, \qquad N_x.

Low-collisionality runs are especially sensitive to :math:`N_\zeta` and :math:`N_\xi`
because of the trapped-passing boundary layer. In contrast, :math:`N_x` often changes
more slowly with collisionality. This is why the automated suite and examples choose
resolution changes by axis, not by a blind global scaling factor.

For user-facing guidance and benchmark examples, see :doc:`performance`,
:doc:`parallelism`, and :doc:`examples`.
