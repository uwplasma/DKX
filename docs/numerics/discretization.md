# Discretization

DKX discretizes the drift-kinetic equation exactly as SFINCS version 3 does, so
that a namelist deck produces the same discrete operator in both codes and the
parity suites can compare matrices entry by entry. This page describes that
discretization and the structure it gives the linear system, because the
structure is what the solver routes of {doc}`solver_routes` exploit.

## Unknowns

For each kinetic species $s$ the non-adiabatic part of the distribution
function is stored on the tensor grid

$$
f = f(s,\, x_i,\, L,\, \theta_j,\, \zeta_k),
$$

where $x = v/v_\mathrm{th}$ is the normalized speed, $L$ the Legendre index in
the pitch variable $\xi = v_\parallel/v$, and $\theta,\zeta$ the periodic
flux-surface angles. The number of unknowns is

$$
N \simeq N_\mathrm{species}\,N_x\,N_\xi\,N_\theta\,N_\zeta
      + N_\theta N_\zeta + N_\mathrm{constraints},
$$

where the second term is present only when $\Phi_1(\theta,\zeta)$ is solved
for and the last term counts the source and constraint coefficients. The
`HSX_PASCollisions_DKESTrajectories` deck used throughout
{doc}`../benchmarks/performance` has 744,610 unknowns; the production HSX
resolution `Ntheta=25, Nzeta=115, Nxi=149, Nx=5` has 2,512,760.

## Angular grids

The angles use uniform periodic grids with dense differentiation matrices,
$\partial_\theta f \approx D_\theta f$ and $\partial_\zeta f \approx D_\zeta f$,
selected by `thetaDerivativeScheme` and `zetaDerivativeScheme` with the
upstream meanings (centered finite differences of several orders, or spectral
collocation). `forceOddNthetaAndNzeta` rounds even resolutions up to odd ones,
as upstream does.

Boozer geometries given as Fourier tables (`geometryScheme` 11 and 12) are
truncated to modes the grid can represent,

$$
0 \le m \le \lfloor N_\theta/2 \rfloor, \qquad |n| \le \lfloor N_\zeta/2 \rfloor,
$$

with the upstream Nyquist exclusions for sine components, so the geometry
arrays agree with the version-3 Jacobian at every resolution. Geometry itself
is described in {doc}`../physics/geometry`.

### Upwinded magnetic-drift derivatives

The magnetic-drift advection (`magneticDriftScheme` 1 to 9) can use directional
upwind matrices `ddtheta_magneticDrift_plus/minus` and
`ddzeta_magneticDrift_plus/minus`. When `magneticDriftDerivativeScheme` is
nonzero the direction is chosen at every grid point from the sign of
$g_1 \hat D(1,1)/Z$, as in `populateMatrix.F90`. The selection is parity-tested
against frozen PETSc matrices on a `geometryScheme=11` fixture.

### Widened upwind stencils

A centered difference has no diagonal weight ($c_0 = 0$), so an advection
operator built from it has an empty diagonal and neither a relaxation smoother
nor a diagonal preconditioner can act on it. DKX offers two opt-in periodic
upwind stencils chosen for diagonal dominance rather than for the smallest
truncation constant (`dkx.phase_space.widened_upwind_stencil`):

```{list-table}
:header-rows: 1

* - Namelist value
  - Order
  - Offsets (positive wind)
  - Coefficients $\times \Delta$
  - $|c_0| / \sum_{j\ne0}|c_j|$
  - Leading truncation
* - `±103`
  - 3
  - $(-3,-1,0,+2)$
  - $(1/15,\,-1,\,5/6,\,1/10)$
  - $5/7 \approx 0.714$
  - $+\tfrac14 \Delta^3 f^{(4)}$
* - `±104`
  - 4
  - $(-4,-3,-1,0,+2)$
  - $(-1/12,\,4/15,\,-4/3,\,13/12,\,1/15)$
  - $13/21 \approx 0.619$
  - $+\tfrac15 \Delta^4 f^{(5)}$
```

The same diagonal measure is $0$ for the centered schemes and for spectral
collocation, and $1/3$, $5/14$, $2/11$ for the compact upwind-biased stencils of
`magneticDriftDerivativeScheme` 1, 2, 3. Each offset set is the unique maximizer
of the diagonal measure over $N$-point stencils in a window of width $N+1$ that
are exact to order $N-1$ and dissipative for the given wind
($\mathrm{Re}\,\hat c(k) \ge 0$ for the symbol $\hat c(k) = \sum_j c_j e^{ik o_j}$).
Reversing the wind mirrors the stencil; the wrong orientation flips the sign of
$\mathrm{Re}\,\hat c$ and is unstable. The arbitrary-offset weights follow
Fornberg, *Math. Comput.* **51**, 699 (1988); the construction is in the
tradition of Fromm (1968), Warming & Beam (1976) and Tam & Webb (1993).

These stencils are a DKX extension. They are never a default, they are not
bit-parity with the Fortran code, and the parity suites pin the centered
schemes. The 100 block of the namelist numbering keeps them clear of any value
upstream may assign. `thetaDerivativeScheme` and `zetaDerivativeScheme` take
`±103` and `±104` with the sign giving the wind direction;
`magneticDriftDerivativeScheme` takes the same codes with its usual
pair-swapping sign convention.

## Pitch angle: Legendre modes

The pitch dependence is expanded in Legendre polynomials,

$$
f = \sum_{L=0}^{N_\xi - 1} f_L(x,\theta,\zeta)\, P_L(\xi).
$$

This basis gives each physical term a fixed reach in $L$, and that reach
decides which solver can be used:

```{list-table}
:header-rows: 1

* - Term
  - Coupling in $L$
  - Coupling in speed and species
* - Parallel streaming, mirror force
  - $L \pm 1$ only (no diagonal)
  - none
* - $E\times B$ advection
  - diagonal
  - none
* - Pitch-angle scattering
  - diagonal, eigenvalue $L(L+1)/2$
  - none
* - Full Fokker–Planck, improved Sugama (`collisionOperator` 0, 3)
  - diagonal
  - dense in $x$, couples species
* - Non-standard $E_r$ terms in $\dot\xi$ and $\dot x$ (`includeXDotTerm`, `includeElectricFieldTermInXiDot`)
  - diagonal and $L \pm 2$
  - $\dot x$ term dense in $x$
* - Magnetic drifts (`magneticDriftScheme` > 0)
  - diagonal and $L \pm 2$
  - none
```

With only the first three rows, the operator is block tridiagonal in $L$ and
the $(s, x)$ pairs decouple: that is the family the structured direct route
solves exactly. Any of the last three rows takes the operator off that
structure and onto the recycled Krylov route. The collision operators are
described in {doc}`../physics/collisions` and the $E_r$ terms in
{doc}`../physics/electric_field`.

The coupling coefficients and Lorentz eigenvalues are
`dkx.phase_space.legendre_coupling_upper`, `legendre_coupling_lower` and
`lorentz_eigenvalues`. The streaming and mirror terms having no diagonal in
$L$ is also why no multigrid smoother exists in this basis; see
{doc}`krylov_and_preconditioners`.

## Speed: Landreman–Ernst grid

Speed uses collocation at the nodes of the polynomials orthogonal on
$[0,\infty)$ with weight $e^{-x^2}x^k$ (`xGrid_k`), built by a Stieltjes
three-term recurrence and a Golub–Welsch eigendecomposition. Maxwellian-weighted
moments are then spectrally accurate with few nodes, and the matching spectral
differentiation matrices $d/dx$ and $d^2/dx^2$ serve the energy-drift and
Fokker–Planck terms (Landreman & Ernst, *J. Comput. Phys.* **243**, 130 (2013)).

Every upstream option is implemented: `xGridScheme` 1 to 8 (the polynomial
grids with and without a node at $x = 0$, uniform grids, and Chebyshev
variants) and `xDotDerivativeScheme` −2 to 11 for the upwinded $\dot x$
derivative pairs. Parity is pinned by Fortran goldens in
`tests/test_output_h5_xgrid_schemes_parity.py`. The grid constructors are
`dkx.phase_space.make_speed_grid`, `speed_grid_diff_matrices` and
`xdot_diff_matrices`; the polynomial kernel the collision operators consume is
`dkx.xgrid`.

## The $N_\xi$-for-$x$ ramp

`Nxi_for_x_option` sets how many Legendre modes are active at each speed node.
Option 0 keeps all $N_\xi$ everywhere. Options 1 to 3 port the ramp of
`createGrids.F90`: the slowest particles keep fewer modes, and the count rises
with $x$ from a floor of `NL` (the Rosenbluth-potential resolution) to $N_\xi$.
Option 1 is the upstream default, so a deck that does not set the option gets
the ramp (`dkx.phase_space.n_xi_for_x_ramp`).

In the rectangular state layout the inactive $(x, L)$ entries are exact zero
rows of the operator. Upstream never stores them (packed indexing in
`indices.F90`); DKX keeps the rectangular layout and solves the pinned system

$$
(A M + I - M)\, u = b,
$$

with $M$ the projector onto active entries
(`KineticOperator.active_dof_mask`). The pinned system is nonsingular, agrees
with $A$ on the physical subspace, and forces the inactive entries to zero, so
solutions, residuals and adjoints match the packed Fortran system.

On the 744,610-unknown HSX case the ramp lowers the peak memory of a warm solve
from 1.16 GB (uniform $N_\xi$) to 0.93 GB, with physics outputs within 0.9 %
(the HSX head-to-head of {doc}`../benchmarks/performance`).

## Bordered block structure

After discretization the problem is $A u = b$ with

$$
A =
\begin{bmatrix}
  A_{ff} & A_{f\Phi} & A_{fc} \\
  A_{\Phi f} & A_{\Phi\Phi} & A_{\Phi c} \\
  A_{cf} & A_{c\Phi} & A_{cc}
\end{bmatrix},
$$

where $A_{ff}$ is the kinetic block, $A_{\Phi\Phi}$ the quasineutrality block
(present only with $\Phi_1$), and the $c$ rows and columns impose the density,
energy and gauge constraints of `constraintScheme`. The border is thin (at most
$N_\theta N_\zeta$ plus a few rows) and dense.

The operator is applied matrix-free: `KineticOperator.apply` is a composition
of tensor contractions and directional derivatives, never an assembled sparse
matrix. That keeps it compilable for CPU and GPU and differentiable with
`jax.grad`. The right-hand side is `KineticOperator.rhs`, and the analytic
block-tridiagonal extraction used by the structured route is
`KineticOperator.to_block_tridiagonal`. Only the sparse direct route forms a
matrix, and it does so from products with the matrix-free action
({doc}`solver_routes`).

## A second pitch discretization

`dkx.collocation` holds an experimental second discretization of the same
equation with pitch on a grid (half-index uniform pitch angle, upwinded
streaming and mirror) rather than a Legendre spectrum. On a pitch grid, $\xi$
is a multiplication operator, streaming is diagonal in pitch and the mirror
force is an upwindable advection, so a semicoarsened multigrid cycle over
$(\alpha, \theta, \zeta)$ converges: on the W7-X operator used in
{doc}`krylov_and_preconditioners`, first-order upwinding gives a two-grid factor
of 0.24 where the Legendre basis gives $4.0\times10^{13}$
(`tools/benchmarks/tier2_pitch_basis_study.py`).

Its scope is one species, `collisionOperator = 1`, RHSMode 1 and DKES-like
trajectories. It changes the answers at fixed resolution, it has no Fortran
matrix parity, and it does not touch `dkx.drift_kinetic` or `dkx.solve`.

## Resolution

The practical resolution knobs are $N_\theta$, $N_\zeta$, $N_\xi$ and $N_x$.
Low-collisionality runs are most sensitive to $N_\zeta$ and $N_\xi$ because of
the trapped–passing boundary layer, and $N_x$ changes more slowly with
collisionality. Refine one axis at a time and check each observable; the
workflow and its error bars are in {doc}`../user_guide/convergence`. Finite
$E_r$ needs its own check: above $|E_*| \approx 1/3$ the trajectory models
separate and the pitch and speed resolutions must be refined with $E_r$
({doc}`../physics/electric_field`).
