Numerics and algorithms
=======================

`sfincs_jax` solves a large structured linear (or, with :math:`\Phi_1`,
nonlinear) system that comes from discretizing the radially local drift-kinetic
equation of :doc:`physics_reference` on a single flux surface. This page
describes the discretization, the three-tier solver policy, and the
implicit-differentiation adjoint.

Discrete unknowns
-----------------

For each kinetic species the first-order distribution is represented on the
tensor grid :math:`(x_i, L, \theta_j, \zeta_k)`:

- :math:`x = v/v_\mathrm{th}` — normalized speed (collocation nodes);
- :math:`L` — Legendre index in :math:`\xi = v_\parallel/v`;
- :math:`\theta,\zeta` — periodic straight-field-line angles.

The total number of degrees of freedom scales as

.. math::

   N_\mathrm{dof} \sim
   N_\mathrm{species}\,N_x\,N_\xi\,N_\theta\,N_\zeta
   + N_\theta N_\zeta + N_\mathrm{constraints},

where the last two terms are the optional :math:`\Phi_1(\theta,\zeta)` unknowns
and the constraint/source coefficients. This scaling is why memory layout,
truncation, and preconditioning matter — a production HSX case is
:math:`\sim 7.4\times10^5` unknowns.

Angular discretization
----------------------

The angles use periodic grids with dense differentiation matrices acting along
each axis, :math:`\partial_\theta f \approx D_\theta f`,
:math:`\partial_\zeta f \approx D_\zeta f` (``thetaDerivativeScheme`` /
``zetaDerivativeScheme``). For the advection-dominated magnetic-drift terms the
code can use directional **upwind** derivative matrices selected pointwise from
the sign of the local drift coefficient, which stabilizes the trapped-passing
boundary layer at low collisionality.

Velocity-space discretization
-----------------------------

**Pitch angle** is expanded in Legendre modes,
:math:`f = \sum_{L=0}^{N_\xi-1} f_L(x,\theta,\zeta)\,P_L(\xi)`. This has two
structural consequences that drive the solver design:

- streaming and mirror couple :math:`L\leftrightarrow L\pm1`;
- the :math:`E_r` energy/pitch drifts couple :math:`L\leftrightarrow L\pm2`;
- collisions are diagonal in :math:`L` for pitch-angle scattering and dense in
  :math:`x` for the full Fokker--Planck operator.

**Speed** uses the Landreman--Ernst grid: collocation nodes of the non-classical
orthogonal polynomials for the weight :math:`e^{-x^2}x^{k}` (``xGrid_k``),
constructed by a Stieltjes three-term recurrence and Golub--Welsch
eigendecomposition. This gives spectral accuracy for Maxwellian-weighted moments
with few nodes, and the matching spectral differentiation matrices
:math:`d/dx`, :math:`d^2/dx^2` for the energy-drift and Fokker--Planck terms
(`Landreman & Ernst, J. Comput. Phys. 243 (2013) <https://arxiv.org/abs/1210.5289>`_).

**The** :math:`N_\xi`-**for-**:math:`x` **ramp** keeps fewer Legendre modes at
high speed, where the distribution is smoother in pitch: ``Nxi_for_x_option``
sets :math:`N_\xi(x)` to ramp from a floor (the Rosenbluth :math:`N_L`) up to
:math:`N_\xi`. On the 744k-unknown HSX case this ramp is the difference between a
warm solve at ``0.93 GB`` (ramp) and ``1.16 GB`` (uniform :math:`N_\xi`) — see
:doc:`performance`.

.. admonition:: Where in the code

   Legendre couplings and Lorentz eigenvalues:
   :func:`sfincs_jax.phase_space.legendre_coupling_upper` /
   ``legendre_coupling_lower`` / ``lorentz_eigenvalues``. Speed grid:
   :func:`sfincs_jax.phase_space.make_speed_grid` and
   ``speed_grid_diff_matrices``. Ramp:
   :func:`sfincs_jax.phase_space.n_xi_for_x_ramp`. All are collected in
   :class:`sfincs_jax.phase_space.Grids` via ``make_grids``.

Linear-system structure
-----------------------

After discretization the problem is :math:`A u = b` with the block structure

.. math::

   A =
   \begin{bmatrix}
     A_{ff} & A_{f\Phi} & A_{fc} \\
     A_{\Phi f} & A_{\Phi\Phi} & A_{\Phi c} \\
     A_{cf} & A_{c\Phi} & A_{cc}
   \end{bmatrix},

with :math:`A_{ff}` the kinetic block, :math:`A_{\Phi\Phi}` the quasineutrality
block (when :math:`\Phi_1` is active), and the :math:`c` rows/columns imposing
density, energy, and gauge constraints. The operator is applied **matrix-free**
as a composition of tensor contractions and directional derivatives rather than
an assembled sparse matrix, so it JIT-compiles for CPU or GPU and differentiates
cleanly.

.. admonition:: Where in the code

   The matrix-free action is :meth:`sfincs_jax.drift_kinetic.KineticOperator.apply`;
   the right-hand side is ``KineticOperator.rhs``. The analytic
   block-tridiagonal-in-:math:`L` extraction is
   ``KineticOperator.to_block_tridiagonal``.

The three solver tiers
----------------------

The solve policy (:func:`sfincs_jax.solve.solve`, ``solve_method="auto"``) picks
the cheapest adequate tier over a :class:`~sfincs_jax.drift_kinetic.KineticOperator`.

Tier 1 — structured direct (block-tridiagonal Legendre elimination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the operator reduces to the **DKES-trajectory / pitch-angle-scattering
family** — streaming and mirror couple :math:`L\pm1`, :math:`E\times B` and PAS
are diagonal in :math:`L`, no :math:`E_r` :math:`L\pm2` terms, and no
Fokker--Planck :math:`(\text{species},x)` coupling — the Legendre-mode
representation of the drift-kinetic operator is **block tridiagonal** in
:math:`L`. In that case the :math:`(\text{species}, x)` axes are mutually
uncoupled, so the full system splits into :math:`N_\mathrm{species}\times N_x`
independent block-tridiagonal systems of :math:`N_\xi` dense
:math:`(N_\theta N_\zeta)` blocks, each with a rank-one constraint border. The
border is absorbed exactly with a rank-one update, and the batch is solved by a
``vmap``-ed block-Thomas factor/solve; multiple right-hand sides share one
elimination.

The tridiagonal structure and its block elimination follow the Legendre
analysis of `Escoto, PhD thesis (2025), arXiv:2510.27513
<https://arxiv.org/abs/2510.27513>`_, and rest on the classical block-tridiagonal
and variational treatment of the monoenergetic drift-kinetic equation by
`Hirshman, Shaing, van Rij, Beasley & Crume, Phys. Fluids 29, 2951 (1986)
<https://doi.org/10.1063/1.865495>`_ (with monoenergetic normalizations as in
`Beidler et al., Nucl. Fusion 51, 076001 (2011)
<https://doi.org/10.1088/0029-5515/51/7/076001>`_). The implementation adds a
**truncated-storage** back-substitution: the forward elimination visits all
:math:`N_\xi` blocks, but only the lowest ``keep`` blocks — the ones the
right-hand side and the physical moments actually touch — are retained, so peak
memory is bounded by the truncation depth instead of the full Legendre chain.

Concretely, the tier-1 peak memory is

.. math::

   \mathcal{O}\!\left(K\,m^2\right),
   \qquad m = N_\theta N_\zeta,\quad K = \texttt{tier1\_keep\_lowest},

i.e. it scales with the retained keep-depth :math:`K` (default 3) times the
square of the dense angular block dimension :math:`m`, and is **independent of**
:math:`N_\xi` and :math:`N_x`. One :math:`2875^2` block at the 744k-unknown HSX
resolution is about 66 MB, so the truncated route needs :math:`\sim 0.3` GB where
a full-band factorization of the same operator would need :math:`\sim 91` GB
(:doc:`performance`). This is the origin of the tier-1 memory advantage over an
assembled sparse factorization. The same discrete operator also yields an
a-posteriori convergence certificate: the variational transport-coefficient
bounds (:doc:`capabilities`) bracket the monoenergetic :math:`D_{11}` from above
and below.

Tier 2 — preconditioned, recycled Krylov
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When tier 1 does not apply (full Fokker--Planck, or the full-trajectory
:math:`E_r` terms), the code runs matrix-free FGMRES with subspace recycling
(GCROT) on ``KineticOperator.apply``, right-preconditioned by an **exact tier-1
solve of a SFINCS-simplified coarse operator**. The coarse operator uses the
Fortran ``preconditionerOptions`` idiom — ``preconditioner_species=1``
(self-collisions only) and ``preconditioner_x=1`` (:math:`x`-diagonal
collisions) reduce Fokker--Planck to a PAS-like :math:`L`-diagonal coefficient,
and the :math:`E_r` :math:`L\pm2` terms are dropped — so the preconditioner is
itself a tier-1 direct solve. The recycle pair :math:`(C,U)` is returned for
warm-starting continuation, which makes neighbouring points in an :math:`E_r`
scan or Newton :math:`\Phi_1` iteration converge in a handful of iterations.

Tier 3 — host sparse-direct fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an escape hatch the operator is materialized (vmapped unit vectors, guarded
by ``max_dense_size``) into CSR and factored by SuperLU on the host. This tier
is non-differentiable and non-jittable and prints a one-line notice; it is used
on explicit request (``method="direct"``) or when tier 2 breaches its iteration
cap under ``method="auto"``.

.. admonition:: Where in the code

   :func:`sfincs_jax.solve.solve` (auto policy, solve.py); tier-1 build
   ``build_tier1_solver`` and the truncated variant ``_solve_tier1_truncated``;
   tier-2 ``_solve_tier2`` with ``build_coarse_preconditioner``; tier-3
   ``_solve_tier3``. The structured factorization, recycled Krylov, and host
   direct solves are provided by the ``solvax`` library
   (`github.com/uwplasma/SOLVAX <https://github.com/uwplasma/SOLVAX>`_,
   `PyPI <https://pypi.org/project/solvax/>`_).

Implicit differentiation (IFT adjoint)
--------------------------------------

For gradient-based workflows tiers 1 and 2 are wrapped with the implicit
function theorem (``jax.lax.custom_linear_solve`` via
``solvax.implicit.linear_solve``). Rather than differentiating through the
solver iterations, the adjoint of a linear solve :math:`Au=b` is one **transposed
solve**, which reuses the same tier-1 block-Thomas factors
(``block_thomas_solve(transpose=True)``) or a transposed-preconditioner GCROT
solve. The cost of a gradient is therefore one additional solve, independent of
the iteration count of the forward solve. The ambipolar :math:`E_r` root and the
nonlinear :math:`\Phi_1` Newton solve are differentiated the same way at the
outer (root) level (:func:`sfincs_jax.er.ambipolar_er`,
:func:`sfincs_jax.phi1.phi1_state`).

Numerical building blocks — the structured factorizations, the recycled Krylov,
the mixed-precision block-Thomas, and the implicit-solve wrappers — live in the
standalone ``solvax`` package so they can be tested and reused independently.
The mixed-precision block-Thomas path is GPU-gated (it is faster on GPU FP64 but
slower on CPU), so the CPU path uses the plain block-Thomas factorization.

When to use which tier
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 30 24

   * - Case
     - Auto tier
     - Why
     - Differentiable
   * - DKES trajectories + PAS (RHSMode 3, monoenergetic)
     - Tier 1
     - Block-tridiagonal in :math:`L`; :math:`N_s N_x` independent chains
     - yes (transposed factors)
   * - PAS, full profile solve (RHSMode 1)
     - Tier 1
     - Same structure; multi-RHS shares one elimination
     - yes
   * - Full Fokker--Planck collisions
     - Tier 2
     - Dense :math:`x`/species coupling breaks tridiagonality
     - yes (transposed preconditioner)
   * - Full-trajectory :math:`E_r` (:math:`L\pm2` terms)
     - Tier 2
     - :math:`L\pm2` coupling breaks tridiagonality
     - yes
   * - Ill-conditioned / small, or tier-2 stall
     - Tier 3
     - Host SuperLU direct
     - no (loud escape hatch)

Resolution guidance
-------------------

The practical knobs are :math:`N_\theta, N_\zeta, N_\xi, N_x`. Low-collisionality
runs are especially sensitive to :math:`N_\zeta` and :math:`N_\xi` because of the
trapped-passing boundary layer, while :math:`N_x` changes more slowly with
collisionality. Convergence is therefore best checked by refining one axis at a
time rather than by a blind global scale factor; the examples and audited suite
choose resolution changes per axis. For measured runtime/memory and parity
evidence see :doc:`performance` and :doc:`parity`.
