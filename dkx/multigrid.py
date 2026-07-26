"""Geometric-multigrid preconditioner for the tier-2 Krylov solve.

The tier-2 solver (:func:`dkx.solve.solve` with ``method="gmres"``) is the
only route open to the physics that has no block-tridiagonal-in-L structure:
full Fokker-Planck and improved-Sugama collisions, ``Phi1``/quasineutrality,
tangential magnetic drifts, and the ``E_r`` ``xDot``/``xiDot`` terms.  Its
classical preconditioner (:func:`dkx.solve.build_coarse_preconditioner`)
strips the operator to the Fortran ``preconditionerOptions`` simplification
-- block tridiagonal over Legendre index ``L``, uncoupled over
``(species, x)`` -- and inverts it *exactly* with a batched block-Thomas
factorization whose dense blocks are ``(Ntheta*Nzeta)`` square.  That is
``O(Nxi * Nspecies * Nx * (Ntheta*Nzeta)**3)`` work and
``O(Nxi * Nspecies * Nx * (Ntheta*Nzeta)**2)`` memory *per solve*, so it stops
being affordable exactly where the interesting physics lives: at
``Ntheta x Nzeta = 21 x 41`` the bands alone are ~10 GB.

This module replaces the exact inversion of that same simplified operator by
a geometric multigrid V-cycle, keeping everything else -- which operator is
simplified, and how the bordered constraint / ``Phi1`` rows are eliminated --
byte-for-byte identical.  A preconditioner cannot change the answer, and the
parity tests pin that: the tier-2 solution is the same to solver tolerance
whichever inner inverse is used.

Hierarchy
---------
Coarsening is *semicoarsening* over ``(theta, zeta[, xi])`` only
(Trottenberg et al., ch. 5).  The remaining phase-space axes are never
coarsened:

``x`` (speed)
    The Fokker-Planck / improved-Sugama collision operators couple speed
    nodes **densely** (they are spectral in ``x``, not local), and the
    ``E_r`` ``xDot`` term applies a dense ``x d/dx`` matrix.  A coarse ``x``
    grid is a different quadrature rule, not a coarser stencil, so no local
    smoother is complementary to it.
``species``
    Two to a handful of entries with dense cross-species coupling; there is
    no grid to coarsen.

``theta`` and ``zeta`` are periodic and carry the local finite-difference
stencils of ``createGrids.F90`` (``thetaDerivativeScheme`` 1/2 -> 3- or
5-point centered), which is exactly the setting geometric multigrid is built
for.  SFINCS forces ``Ntheta`` and ``Nzeta`` **odd** (so that no grid point is
conjugate to the Nyquist mode, which would put a null vector in the centered
first-derivative matrix), and an odd periodic axis has no factor-2 nested
subgrid.  The transfers here are therefore the *non-nested* uniform periodic
linear interpolation between an ``n_fine``-point and an ``n_coarse``-point
uniform periodic grid, with full weighting as its scaled adjoint,
``R = (n_coarse / n_fine) P^T``.  For the even, nested case ``n_fine =
2 n_coarse`` these reduce exactly to the classical ``[1/2, 1, 1/2]`` /
``[1/4, 1/2, 1/4]`` stencils of :func:`solvax.transfer.prolongation_matrix`
and :func:`solvax.transfer.restriction_matrix` -- which is what
``tests/test_multigrid.py`` checks them against.  Coarse levels stay odd for
the same Nyquist reason as the fine one.

The ``xi`` decision
-------------------
``xi`` is **not** a spatial grid: the state holds Legendre *coefficients*
``f_l``, so "coarsening ``xi``" means truncating the spectrum, ``Nxi ->
Nxi/2``.  That is p-coarsening, and its transfers are the natural ones for a
hierarchical modal basis: restriction keeps the first ``Nxi_c`` coefficients
(spectral truncation) and prolongation zero-pads (injection).  They are exact
adjoints, ``R = P^T``, and cost nothing -- full weighting and linear
interpolation would be *wrong* here, because averaging neighbouring Legendre
coefficients is not an averaging of anything physical.  The coarse operator is
then simply the same discrete DKE at a lower ``Nxi``, which is a legitimate
physical discretization (it is what a lower-resolution run solves).

**Measured, it is not safe, so it is off by default.**  On the NCSX
``11 x 21 x 41 x 5`` operator, restricting a smooth error, solving the coarse
operator exactly and prolonging back recovers that error to within

======================================  ==========================
coarsening                              ``||e - P e_c|| / ||e||``
======================================  ==========================
``Nzeta`` 21 -> 11                      0.056
``Ntheta`` 11 -> 5                      0.415
both angles, ``Nxi`` untouched          0.440
both angles **and** ``Nxi`` 41 -> 21    1.023
======================================  ==========================

i.e. the angular coarsenings are useful coarse-grid corrections and the pitch
p-coarsening is *worse than doing nothing*: truncating the Legendre spectrum
changes the streaming closure of the coarse operator (mode ``Nxi_c - 1`` loses
its partner), and the zero-padded prolongation puts nothing back on the modes
that were dropped.  ``coarsen_xi=True`` remains available for experiments.

Smoother
--------
Two families are implemented, and the measurement that chooses between them is
the point of this module.

The operator's two dominant terms live in *different bases*.  Parallel
streaming, ``x xi v_th b.grad``, is an advection along the field line whose
sign reverses with the sign of the pitch; in the Legendre basis it is the
tridiagonal ``xi``-multiplication matrix ``Y``
(``legendre_coupling_lower/upper``) acting on ``L``, tensored with the angular
derivative.  ``Y`` is similar to a symmetric tridiagonal matrix with positive
off-diagonals, so ``Y = V diag(lambda) V^-1`` with ``lambda_k`` the
Gauss-Legendre nodes of order ``Nxi``: ``V`` is the discrete Legendre
transform, and in *that* basis streaming is ``x lambda_k (v_theta d/dtheta +
v_zeta d/dzeta)`` -- a plain advection with a definite wind per pitch node,
exactly what an upwind-ordered relaxation wants (Brandt 1977, section 3; Brandt
& Yavneh 1993; Trottenberg et al., section 7.4).  Collisions, however, are
*diagonal in L* and **dense in pitch collocation**: measured on the same
operator, keeping only the collocation diagonal of the reduced collision
operator discards 65% of it in the 2-norm, and even a lumped tridiagonal-in-
pitch approximation discards 31%.  At the low speed nodes the collision
operator is by far the largest term (``||C||_2 = 678`` at the lowest speed
node of the NCSX ``11 x 21 x 41 x 5`` deck, against an angular-advection scale
of order one), so a relaxation that misrepresents it cannot converge.

``smoother="legendre_plane"`` (the default) therefore stays in the Legendre
basis and relaxes the ``(L, angle)`` planes *exactly*
(:func:`_legendre_plane_blocks`): collisions, mirror force, streaming and ExB
along one angle are all kept, the blocks are ``N x N`` with ``N`` one angular
length instead of ``Ntheta*Nzeta``, and only the other angle's derivative is
left to the coarse grid.  ``smoother="plane"``/``"alternating"``/``"theta"``/
``"zeta"`` are the pitch-collocation family, kept because they are the
measurement that justifies the default.

Measured smoothing factors
--------------------------
:func:`measure_smoothing_factor` runs
:func:`solvax.smoothers.smoothing_factor` -- the empirical counterpart of
Brandt's local Fourier analysis ``mu`` -- on the **real** simplified NCSX
``11 x 21 x 41 x 5`` operator, restricted to the high-frequency modes each
coarsening cannot represent:

===================================  ==========  =========  ==========
smoother                             ``theta``   ``zeta``   ``xi``
===================================  ==========  =========  ==========
``legendre_plane`` (zeta in plane)   214         0.87       196
pitch-collocation plane, upwind      5.27        5.17       4.27
...  with ``omega = 0.5``            2.77        1.31       1.72
...  with ``omega = 0.2``            1.06        0.98       0.98
upwind zeta line                     ---         6.55       ---
alternating upwind lines             5.63        ---        5.87
===================================  ==========  =========  ==========

Only the direction a relaxation resolves exactly is smoothed: the
``(L, zeta)``-plane sweep has ``mu = 0.87`` across ``zeta`` and ``mu = 214``
across ``theta``.  No cheap sweep is complementary to angular coarsening in
*both* angles at once, and the near-null directions -- distributions nearly
constant along the field line, which is neither a ``theta`` mode nor a ``zeta``
mode -- are represented by no coarse grid this hierarchy can build.
:func:`line_diagonal_dominance` and :func:`line_smoother_spectral_radius` say
why in one number, and the next section is that measurement.

The consequence, measured end to end and reported in ``docs/performance.rst``:
on the full-Fokker-Planck ladder the multigrid route is affordable where the
classical preconditioner is not, but it does not reach the tier-2 tolerance,
while the exact block-Thomas of the same simplified operator does so in 21
iterations.  The route is therefore opt-in
(``solve(preconditioner="multigrid")``) and the default is unchanged.

Why no relaxation smooths in a Legendre-modal pitch basis
---------------------------------------------------------
The stalls above are not a bad choice of cycle, transfer or sweep ordering.
They follow from one structural fact about the discretization, which
``tests/test_multigrid.py`` pins:

**Parallel streaming and the mirror force are strictly off-diagonal in the
Legendre index.**  They couple ``L -> L +- 1`` and contribute *nothing* to the
``(L, L)`` block (see :meth:`KineticOperator.legendre_blocks`: the diagonal
block is ExB plus the collision diagonal and nothing else).  Every line of an
alternating line relaxation therefore misses the operator's dominant term:

* a ``theta``- or ``zeta``-line block, taken at fixed ``L``, contains no
  streaming at all -- so no angular stencil, upwinded or not, can give that
  block diagonal weight from the term that dominates the operator;
* the ``L``-line block, taken at fixed ``(theta, zeta)``, does contain the
  ``L +- 1`` coupling, but only its angle-diagonal part, which is the mirror
  force: a tridiagonal matrix with ``-L(L-1)/(2L-1)`` below and
  ``(L+1)(L+2)/(2L+3)`` above the diagonal, i.e. near-*skew*-symmetric with a
  diagonal of ``nu_D l(l+1)/2`` that vanishes with the collisionality.

Measured with :func:`line_smoother_spectral_radius` on the real simplified
operator (one ``(species, speed)`` block, W7-X standard configuration,
``9 x 11 x 13``, alternating exact line solves in all three coordinates,
``omega = 1``):

===========================  =========  =========  =========
``nu_n``                     ``1e-1``   ``8.3e-3``  ``1e-4``
===========================  =========  =========  =========
Legendre-modal ``rho(S)``    3.8e3      5.9e6      1.7e12
pitch grid + upwind          0.88       0.97       1.00
pitch grid + centered        11.5       2.2e2      6.3e5
===========================  =========  =========  =========

``rho(S) > 1`` is fatal: a coarse-grid correction cannot rescue a divergent
relaxation.  The modal basis has no convergent line relaxation at any
collisionality, and the gap grows as ``nu`` falls.  Correspondingly the
two-grid factor ``rho(TG)`` of a full ``V(1,1)`` cycle with rediscretized
coarse operators and linear-interpolation transfers, coarsening ``(theta,
zeta)`` in the modal basis and ``(alpha, theta, zeta)`` on the pitch grid:

==========================  =========  ==========  =========  ==========
discretization              ``d``      ``1e-1``   ``8.3e-3``  ``1e-4``
==========================  =========  ==========  =========  ==========
Legendre-modal              ---        1.5e7      4.0e13     3.3e24
collocation, ``up1``        1.00       0.39       0.24       0.74
collocation, widened 2nd    0.88       0.49       0.36       0.80
collocation, textbook 2nd   0.60       0.65       0.48       1.09
collocation, widened 4th    0.62       0.91       0.49       1.25
collocation, textbook 3rd   0.33       2.68       0.89       3.68
collocation, centered       0.00       1.3e2      4.7e4      5.1e11
==========================  =========  ==========  =========  ==========

``d`` is the stencil's diagonal dominance (:func:`stencil_matrices`).  The
convergence factor is a monotone function of it, and the widened stencils --
which skip near neighbours to keep diagonal weight at a given formal order --
beat the textbook upwind-biased ones of the *same or higher* order.  That is
the whole of the recipe: relaxation smooths an advection only where the
discretization is diagonally dominant, and a modal pitch basis is nowhere
diagonally dominant in the streaming term because the streaming term has no
diagonal there at all (Brandt 1977, section 3; Trottenberg et al., sections 2.1
and 7.4; the widening trade is Leonard's, Comput. Methods Appl. Mech. Eng.
**19**, 59 (1979); the difficulty of smoothing an accurate drift-kinetic
discretization is stated in M. Landreman, Bull. Am. Phys. Soc. **62**, JP11.128
(2017)).

Why changing basis inside the preconditioner does not rescue it
--------------------------------------------------------------
The obvious escape is to keep dkx's Legendre discretization for the *solve*
(and with it the Fortran parity) and change basis only inside the
preconditioner: transform the residual to a pitch grid with the Legendre
Vandermonde ``V``, smooth there, transform back with ``V^+``.  The transform
is cheap and exact -- ``Nxi**2`` per angular point, ``cond(V) ~ 7`` at
``Nalpha = Nxi`` on the half-index uniform-``alpha`` grid -- and
:func:`pitch_collocation_surrogate` builds exactly that surrogate.  Coupling a
low-order finite-difference discretization to a spectral one this way is
standard practice (S. A. Orszag, J. Comput. Phys. **37**, 70 (1980); M. O.
Deville & E. H. Mund, J. Comput. Phys. **60**, 517 (1985)).

It does not work, for a reason that is quantitative rather than structural.
Two requirements pull in opposite directions:

*P1*
    the surrogate must be spectrally close to the modal operator, or the outer
    Krylov solve pays the difference;
*P2*
    the surrogate must admit a convergent relaxation, or its own inverse is no
    cheaper than the one it replaces.

Upwinding buys ``P2`` and costs ``P1``.  Measured on the same ``9 x 11 x 13``
deck at ``nu_n = 8.3e-3`` as the tables above: GMRES on the modal operator
preconditioned by the surrogate's *exact* inverse needs

======================================  ================  ============
surrogate                               GMRES iterations  ``rho(TG)``
======================================  ================  ============
centered angles, centered pitch         18                4.7e4
centered angles, upwind pitch           21                4.0e4
widened-2nd everywhere                  199               0.37
first-order upwind everywhere           201               0.24
(no preconditioner)                     > 400             ---
======================================  ================  ============

-- and the upwind column degrades further with angular resolution (261 at
``9 x 15``, > 400 at ``13 x 21``), while the centered column barely moves (18,
24, 41 at ``9 x 15 x 17``, ``13 x 21 x 17``, ``17 x 25 x 33``).  The half that
is a good preconditioner is exactly the half no relaxation smooths.  Note
which axis does the damage: upwinding the *pitch* direction alone is nearly
free (18 -> 21), because dkx's pitch operator is spectral anyway; it is
upwinding the *angles* -- where dkx's own stencils are centered by
construction, SFINCS ``thetaDerivativeScheme`` 1/2 -- that costs the order of
magnitude.

Nor does double discretisation (accurate operator on the fine level, upwinded
smoother and coarse operators; Brandt 1981; Trottenberg et al., section 7.4)
close the gap: with the centered surrogate as ``A`` and the ``up1`` surrogate
supplying the line blocks and the coarse level, ``rho(TG)`` is 1.46, 1.94 and
2.27 at ``nu_n = 1e-1``, ``8.3e-3`` and ``1e-4`` -- divergent at every
collisionality.  Over-resolving the pitch grid does not help either
(``Nalpha = 2 Nxi + 1`` takes the centered surrogate from 18 iterations to
310, because the extra nodes are outside the modal operator's range and the
pseudo-inverse projects them away).

What *would* be required
------------------------
A convergent cycle needs a discretization that is diagonally dominant in the
streaming term, and streaming has a diagonal only when pitch is a *grid*: on a
pitch-angle collocation grid ``xi`` is a multiplication operator, so
``xi b.grad`` is diagonal in pitch and its upwind angular stencil puts weight
on the matrix diagonal, while the mirror force becomes an advection in
``alpha`` whose upwind discretization is diagonally dominant in turn.  Then
all of ``(alpha, theta, zeta)`` can be coarsened together, which is what keeps
the advection direction fixed relative to the mesh on every level.

That is a change to dkx's *discretization*, not to its preconditioner: it
changes the answers at fixed resolution, breaks the Fortran matrix parity the
repository is gated on, and requires the Fokker-Planck and improved-Sugama
collision operators -- which are built in the Legendre basis, where they are
``L``-diagonal -- to be re-derived on a pitch grid, where they are dense
(keeping only the collocation diagonal discards 65% of the reduced operator in
the 2-norm; a lumped tridiagonal-in-pitch approximation still discards 31%).
It is a project, not a patch, and it buys a preconditioner that is *worse*
than the classical block-Thomas wherever the classical one fits in memory (21
iterations to ``1e-11``).  The honest scope of the multigrid route is
therefore unchanged: opt-in, for the grids where the exact factorization does
not fit.

Coarsest level
--------------
The recursion bottoms out on the existing exact batched block-Thomas solve of
the same simplified operator -- a cubic cost is harmless where the grid is
tiny (a ``5 x 11`` coarsest angular grid is ``55`` unknowns per block, not
``861``).

References
----------
- A. Brandt, "Multi-level adaptive solutions to boundary-value problems",
  Math. Comp. **31**, 333 (1977).
- A. Brandt & I. Yavneh, "Accelerated multigrid convergence and
  high-Reynolds recirculating flows", SIAM J. Sci. Comput. **14**, 607
  (1993).
- U. Trottenberg, C. W. Oosterlee & A. Schuller, *Multigrid*, Academic Press
  (2001) -- ch. 2 (transfers, rediscretized coarse operators), ch. 5
  (semicoarsening), sections 2.1, 5.1 and 7.4 (line relaxation, double
  discretisation and downstream relaxation for convection).
- A. Brandt, "Guide to multigrid development", in *Multigrid Methods*, Lecture
  Notes in Mathematics **960**, Springer (1982) -- double discretisation.
- B. P. Leonard, "A stable and accurate convective modelling procedure based on
  quadratic upstream interpolation", Comput. Methods Appl. Mech. Eng. **19**,
  59 (1979) -- trading stencil width for boundedness/diagonal weight.
- S. A. Orszag, "Spectral methods for problems in complex geometries",
  J. Comput. Phys. **37**, 70 (1980); M. O. Deville & E. H. Mund, "Chebyshev
  pseudospectral solution of second-order elliptic equations with finite
  element preconditioning", J. Comput. Phys. **60**, 517 (1985) -- low-order
  discretizations as preconditioners for spectral ones.
- M. L. Adams & E. W. Larsen, "Fast iterative methods for discrete-ordinates
  particle transport calculations", Prog. Nucl. Energy **40**, 3 (2002) -- the
  same near-null-space obstruction in neutral-particle transport, and the
  synthetic-acceleration family of remedies for it.
- M. Landreman, "A multigrid method for drift-kinetic calculations in
  stellarators and rippled tokamaks", Bull. Am. Phys. Soc. **62**, JP11.128
  (2017) -- multigrid smoothers are unstable for accurate discretizations of
  this equation.

Availability
------------
The cycle driver, transfers and smoothers come from ``solvax`` >
0.8.7 (``solvax.precond.multigrid``, ``solvax.smoothers.upwind_smoother``,
``solvax.transfer``).  Those landed after the currently pinned PyPI release,
so this module feature-detects them (:func:`multigrid_available`) exactly as
:mod:`dkx.solve` feature-detects ``schur_projected_precond(d_block=...)``;
with an older solvax installed the ``preconditioner="multigrid"`` route
raises a clear error and every other route is unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dkx.drift_kinetic import KineticOperator  # noqa: E402
from dkx.phase_space import uniform_periodic_diff_matrices  # noqa: E402

try:  # noqa: E402
    from solvax.direct import block_thomas_factor, block_thomas_solve
    from solvax.precond import MultigridLevel, multigrid as solvax_multigrid
    from solvax.smoothers import (
        alternating_smoother,
        plane_smoother,
        relaxation,
        smoothing_factor,
        tridiagonal_smoother,
        upwind_smoother,
    )

    _MULTIGRID_IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:  # pragma: no cover - exercised by old solvax installs
    block_thomas_factor = None  # type: ignore[assignment]
    block_thomas_solve = None  # type: ignore[assignment]
    MultigridLevel = None  # type: ignore[assignment, misc]
    solvax_multigrid = None  # type: ignore[assignment]
    alternating_smoother = None  # type: ignore[assignment]
    plane_smoother = None  # type: ignore[assignment]
    relaxation = None  # type: ignore[assignment]
    smoothing_factor = None  # type: ignore[assignment]
    tridiagonal_smoother = None  # type: ignore[assignment]
    upwind_smoother = None  # type: ignore[assignment]
    _MULTIGRID_IMPORT_ERROR = _exc

__all__ = [
    "MultigridSettings",
    "PitchCollocationSurrogate",
    "UPWIND_STENCILS",
    "build_multigrid_f_inverse",
    "coarsen_operator",
    "dense_simplified_block",
    "hierarchy_shapes",
    "line_diagonal_dominance",
    "measure_smoothing_factor",
    "multigrid_available",
    "periodic_transfer_matrices",
    "pitch_collocation_surrogate",
    "simplified_operator",
    "stencil_matrices",
    "xi_transfer_matrices",
]


# Relative floor added to every diagonal block so a collisionless, drift-free
# coarse f-block (whose diagonal is EXACTLY zero) still factors; mirrors
# ``dkx.solve.build_coarse_preconditioner``.
_DIAGONAL_FLOOR = 1e-8


def multigrid_available() -> tuple[bool, str]:
    """Whether the installed ``solvax`` exposes the multigrid API this needs.

    Returns:
        ``(True, "")`` when :mod:`solvax.precond`, :mod:`solvax.smoothers` and
        :mod:`solvax.transfer` provide the cycle driver, the upwind line
        smoother and the transfer builders; otherwise ``(False, reason)``.
    """
    if _MULTIGRID_IMPORT_ERROR is not None:
        return False, (
            "the installed solvax has no multigrid API "
            f"({_MULTIGRID_IMPORT_ERROR}); dkx.multigrid needs solvax > 0.8.7 "
            "(pip install git+https://github.com/uwplasma/SOLVAX)"
        )
    return True, ""


def _require_multigrid() -> None:
    ok, reason = multigrid_available()
    if not ok:
        raise ImportError(f"the multigrid tier-2 preconditioner is unavailable: {reason}")


@dataclass(frozen=True)
class MultigridSettings:
    """Knobs of the tier-2 multigrid preconditioner.

    Attributes:
        levels: maximum number of coarsening steps (the hierarchy stops
            earlier when every coarsenable axis has reached its floor).
        coarsen_xi: p-coarsen the Legendre index too (spectral truncation /
            zero-padding transfers).  ``False`` semicoarsens ``(theta, zeta)``
            only.
        min_angle: smallest ``Ntheta``/``Nzeta`` a coarse level may reach.
        min_xi: smallest ``Nxi`` a coarse level may reach.
        cycle: ``"v"``, ``"w"`` or ``"f"`` (see :func:`solvax.precond.multigrid`).
        pre_smooth: pre-smoothing sweeps per level.
        post_smooth: post-smoothing sweeps per level.
        cycles: cycles per preconditioner application.
        omega: relaxation weight of the angular (pitch-collocation) sweep.
        omega_xi: relaxation weight of the Legendre-basis sweep (the
            ``"legendre_plane"`` plane solve, or the ``L``-line when
            ``smooth_xi``).
        order: sweep ordering of the angular line relaxation --
            ``"upwind"`` (the default, downstream-ordered), ``"downwind"``,
            ``"forward"`` or ``"backward"``.  The last three exist to measure
            what the ordering is worth.  Ignored by ``smoother="plane"``,
            which is exact in the plane and needs no ordering.
        smoother: relaxation sweep -- ``"legendre_plane"`` (the default:
            exact ``(L, angle)`` plane solve in the Legendre basis) or, in the
            pitch-collocation basis, ``"plane"`` (exact ``(theta, zeta)`` plane
            solve), ``"alternating"`` (upwind theta-line then zeta-line),
            ``"theta"`` or ``"zeta"`` (one upwind line direction).
            ``"hybrid"`` composes the collocation plane sweep with the
            Legendre plane sweep multiplicatively.
        smooth_xi: add the multiplicative Legendre ``L``-line (mirror +
            collision) relaxation.  Off by default: the mirror force is
            skew-symmetric in ``L``, so that line operator is near-singular and
            measures far *worse* than the angular sweep alone.
        shift: weight of the uniform diagonal regularization
            (:func:`_shift`), relative to the operator's own diagonal scale.
            The default ``1e-8`` is the invertibility floor of
            :func:`dkx.solve.build_coarse_preconditioner` and is what a
            preconditioner may spend: measured on the NCSX ladder, raising it
            to ``0.01`` costs the tier-2 solve 60 iterations instead of 21, and
            ``1.0`` never converges at all.
        plane_pin: weight of the per-line rank-one pin in the
            ``"legendre_plane"`` smoother (see :func:`_legendre_plane_blocks`),
            in units of the mean collision diagonal.  ``0`` disables it.
        stencil: angular discretization used *inside the smoother* --
            ``"upwind"`` (first order, dissipative at high frequency, the
            default) or ``"centered"`` (the level operator's own stencil).
            See :func:`_advection_bands`.
        absolute_reaction: keep only the magnitude of the pitch-space reaction
            (collisions and mirror force projected on the collocation
            diagonal), so the smoother's plane operator stays diagonally
            dominant where their signed projection would not be.
    """

    levels: int = 3
    coarsen_xi: bool = False
    min_angle: int = 5
    min_xi: int = 6
    cycle: str = "v"
    pre_smooth: int = 1
    post_smooth: int = 1
    cycles: int = 1
    omega: float = 1.0
    omega_xi: float = 1.0
    order: str = "upwind"
    smoother: str = "legendre_plane"
    smooth_xi: bool = False
    shift: float = _DIAGONAL_FLOOR
    stencil: str = "upwind"
    plane_pin: float = 1.0
    absolute_reaction: bool = True


# =============================================================================
# The simplified (SFINCS ``preconditionerOptions``) operator, as an operator
# =============================================================================


def _collision_diagonal(op: KineticOperator) -> jnp.ndarray | None:
    """``(S, X, L)`` self-species x-diagonal reduction of the dense collisions."""
    from dkx.solve import _collision_phi1_diagonal, _dense_collision_diagonal  # noqa: PLC0415

    total = None
    for coll in (op.fp, op.sugama):
        if coll is not None:
            total = _dense_collision_diagonal(coll.mat)
    if op.fp_phi1 is not None:
        extra = _collision_phi1_diagonal(op)
        total = extra if total is None else total + extra
    return total


def simplified_operator(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> KineticOperator:
    """The Fortran ``preconditionerOptions`` operator, as a :class:`KineticOperator`.

    Identical in content to the block-tridiagonal system
    :func:`dkx.solve.build_coarse_preconditioner` factors -- the dense
    ``(species, x)``-coupled Fokker-Planck / improved-Sugama collision
    operators are reduced to their PAS-like self-species x-diagonal, the
    ``E_r`` ``L +- 2`` terms and the tangential magnetic drifts are dropped,
    and (with ``drop_l_coupling``, the ``preconditioner_xi=1`` knob) the
    ``L +- 1`` streaming coupling goes too -- but expressed as a *matrix-free*
    operator rather than as dense bands, so it can be rediscretized on coarse
    grids and applied at ``O(f_size)`` cost.

    The reduced collision diagonal is folded into the ``pas`` slot: pitch-angle
    scattering *is* an arbitrary ``(S, X, L)`` diagonal coefficient, so the
    result is a plain PAS-family operator that both
    :meth:`KineticOperator.apply_f` and
    :meth:`KineticOperator.to_block_tridiagonal` accept.

    Args:
        op: the full tier-2 operator.
        drop_l_coupling: drop the ``L +- 1`` streaming/mirror coupling.

    Returns:
        A PAS-family :class:`KineticOperator` on the same grid.
    """
    from dkx.collisions import PitchAngleScatteringV3Operator  # noqa: PLC0415

    n_s, n_x, n_xi = op.n_species, op.n_x, op.n_xi
    mask = op._mask()  # (X, L)
    coef = jnp.zeros((n_s, n_x, n_xi), dtype=jnp.float64)
    if op.pas is not None:
        coef = coef + op.pas.coef
    dense = _collision_diagonal(op)
    if dense is not None:
        coef = coef + dense
    coef = coef * mask[None, :, :]

    nu_d_hat = (
        op.pas.nu_d_hat
        if op.pas is not None
        else jnp.zeros((n_s, n_x), dtype=jnp.float64)
    )
    pas = PitchAngleScatteringV3Operator(
        nu_n=jnp.asarray(1.0),
        krook=jnp.asarray(0.0),
        nu_d_hat=nu_d_hat,
        n_xi_for_x=op.n_xi_for_x,
        coef=coef,
        mask_xi=mask,
    )
    simplified = replace(
        op,
        pas=pas,
        fp=None,
        sugama=None,
        fp_phi1=None,
        with_er_xidot=False,
        with_er_xdot=False,
        with_magnetic_drifts=False,
        external_phi1_hat=None,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        phi1_lin_state=None,
        ddx_xdot_plus=None,
        ddx_xdot_minus=None,
    )
    if drop_l_coupling:
        zeros_l = jnp.zeros((n_xi,), dtype=jnp.float64)
        simplified = replace(
            simplified, xi_coupling_lower=zeros_l, xi_coupling_upper=zeros_l
        )
    return simplified


# =============================================================================
# Grid transfers
# =============================================================================


def periodic_transfer_matrices(
    n_fine: int, n_coarse: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Restriction/prolongation between two uniform periodic grids.

    Prolongation is periodic linear interpolation from the ``n_coarse``-point
    grid onto the ``n_fine``-point grid of the same period; restriction is its
    row-normalized scaled adjoint ``R ~ (n_coarse / n_fine) P^T``, i.e. full
    weighting.  Rows of both sum to one, so a constant transfers to the same
    constant -- which also makes ``R`` the second-order interpolation used to
    carry the flux-surface geometry down the hierarchy.  On nested grids
    (``n_fine = 2 n_coarse``) the column sums of ``P`` are already exactly
    ``2``, so the normalization is a no-op there and the variational relation
    ``R = 2^-1 P^T`` holds exactly; off the nested case it is satisfied to
    within a few parts in a thousand.  That relation is a *Galerkin* property
    anyway, and the coarse operators here are rediscretized.

    Unlike :func:`solvax.transfer.grid_transfer`, the two grids need not be
    nested: SFINCS forces ``Ntheta``/``Nzeta`` odd, and an odd periodic axis
    has no factor-2 subgrid.  For the nested case ``n_fine = 2 * n_coarse``
    these matrices coincide exactly with
    :func:`solvax.transfer.prolongation_matrix` (``"linear"``) and
    :func:`solvax.transfer.restriction_matrix` (``"full_weighting"``) with the
    ``"periodic"`` closure.

    Args:
        n_fine: fine-grid points on the period.
        n_coarse: coarse-grid points on the same period (``<= n_fine``).

    Returns:
        ``(restrict, prolong)`` of shapes ``(n_coarse, n_fine)`` and
        ``(n_fine, n_coarse)``.
    """
    n_fine, n_coarse = int(n_fine), int(n_coarse)
    if not 1 <= n_coarse <= n_fine:
        raise ValueError(f"need 1 <= n_coarse <= n_fine, got {n_coarse} and {n_fine}")
    if n_coarse == n_fine:
        eye = jnp.eye(n_fine, dtype=jnp.float64)
        return eye, eye
    position = np.arange(n_fine, dtype=np.float64) * (n_coarse / n_fine)
    left = np.floor(position).astype(int) % n_coarse
    frac = position - np.floor(position)
    prolong = np.zeros((n_fine, n_coarse), dtype=np.float64)
    rows = np.arange(n_fine)
    np.add.at(prolong, (rows, left), 1.0 - frac)
    np.add.at(prolong, (rows, (left + 1) % n_coarse), frac)
    restrict = prolong.T / prolong.sum(axis=0)[:, None]
    return jnp.asarray(restrict), jnp.asarray(prolong)


def xi_transfer_matrices(l_fine: int, l_coarse: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Legendre p-coarsening transfers: spectral truncation and zero-padding.

    The state holds Legendre coefficients, so the coarse grid is the *same*
    hierarchical basis truncated to its first ``l_coarse`` modes.  Restriction
    keeps those coefficients and drops the rest; prolongation embeds them back
    and zero-pads.  They are exact adjoints (``R = P^T``), unlike the
    grid-transfer stencils, which have no meaning between modal coefficients.

    Args:
        l_fine: fine-level ``Nxi``.
        l_coarse: coarse-level ``Nxi`` (``<= l_fine``).

    Returns:
        ``(restrict, prolong)`` of shapes ``(l_coarse, l_fine)`` and
        ``(l_fine, l_coarse)``.
    """
    l_fine, l_coarse = int(l_fine), int(l_coarse)
    if not 1 <= l_coarse <= l_fine:
        raise ValueError(f"need 1 <= l_coarse <= l_fine, got {l_coarse} and {l_fine}")
    restrict = jnp.eye(l_fine, dtype=jnp.float64)[:l_coarse, :]
    return restrict, restrict.T


def _separable_transfer(
    matrices: dict[int, jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """One ``(out, in)`` contraction per transferred axis; untouched axes emit none.

    The same separable application as :mod:`solvax.transfer`: the N-D transfer
    of a tensor-product grid is the Kronecker product of its per-axis matrices,
    so it is applied axis by axis and never formed.
    """
    items = tuple(sorted(matrices.items()))

    def apply(x: jnp.ndarray) -> jnp.ndarray:
        for axis, matrix in items:
            x = jnp.moveaxis(jnp.tensordot(matrix, x, axes=[[1], [axis]]), 0, axis)
        return x

    return apply


def _next_odd(n: int, minimum: int) -> int | None:
    """Next coarser odd size of a periodic axis, or ``None`` when it must stop.

    Odd is not cosmetic: on an even periodic grid the centered first-derivative
    matrix annihilates the alternating Nyquist mode, which would put an exact
    null vector into every coarse operator (the reason ``createGrids.F90``
    forces ``Ntheta``/``Nzeta`` odd in the first place).
    """
    if n <= minimum:
        return None
    coarse = (n + 1) // 2
    if coarse % 2 == 0:
        coarse -= 1
    return coarse if coarse >= minimum else None


def hierarchy_shapes(
    op: KineticOperator, settings: MultigridSettings = MultigridSettings()
) -> tuple[tuple[int, int, int], ...]:
    """``(Ntheta, Nzeta, Nxi)`` of every level, finest first.

    An axis stops coarsening at its floor while the others continue; the
    hierarchy ends when no axis can coarsen further or ``settings.levels``
    steps have been taken.
    """
    shapes = [(int(op.n_theta), int(op.n_zeta), int(op.n_xi))]
    for _ in range(int(settings.levels)):
        n_t, n_z, n_l = shapes[-1]
        next_t = _next_odd(n_t, settings.min_angle)
        next_z = _next_odd(n_z, settings.min_angle)
        next_l = None
        if settings.coarsen_xi and n_l > settings.min_xi:
            next_l = max(settings.min_xi, (n_l + 1) // 2)
            if next_l >= n_l:
                next_l = None
        if next_t is None and next_z is None and next_l is None:
            break
        shapes.append(
            (
                n_t if next_t is None else next_t,
                n_z if next_z is None else next_z,
                n_l if next_l is None else next_l,
            )
        )
    return tuple(shapes)


# =============================================================================
# Rediscretization on a coarse grid
# =============================================================================

#: ``createGrids.F90`` periodic first-derivative stencils, keyed by their
#: half-width: ``(scheme, leading super-diagonal coefficient times spacing)``.
#: The centered first derivative has no diagonal entry, so the half-width --
#: not the nonzero count -- is what identifies the scheme.
_STENCIL_BY_RADIUS = {1: (0, 0.5), 2: (10, 2.0 / 3.0)}


def _angular_stencil(dd: jnp.ndarray) -> tuple[int, float]:
    """``(scheme, spacing)`` of a uniform periodic first-derivative matrix.

    ``uniform_periodic_diff_matrices`` builds circulant matrices whose
    half-width identifies the scheme (1 -> ``thetaDerivativeScheme=1``, the
    3-point centered stencil; 2 -> the 4th-order 5-point default) and whose
    leading super-diagonal entry fixes the grid spacing.  Reading them back is
    what lets a coarse level be rediscretized with the *same* scheme without
    carrying the namelist around.
    """
    row = np.asarray(dd[0])
    n = row.shape[0]
    nonzero = np.flatnonzero(np.abs(row) > 0.0)
    offsets = ((nonzero + n // 2) % n) - n // 2
    radius = int(np.max(np.abs(offsets))) if offsets.size else 0
    if radius not in _STENCIL_BY_RADIUS:
        raise NotImplementedError(
            "the multigrid preconditioner needs a local (finite-difference) "
            f"angular stencil; got a half-width of {radius} on a {n}-point "
            "grid, i.e. the spectral collocation scheme "
            "(thetaDerivativeScheme=0). Use preconditioner='coarse' for "
            "spectral-angle decks."
        )
    scheme, lead = _STENCIL_BY_RADIUS[radius]
    spacing = float(lead / row[1])
    return scheme, spacing


def _coarse_diff_matrix(dd: jnp.ndarray, n_fine: int, n_coarse: int) -> jnp.ndarray:
    """Rediscretize a periodic derivative matrix on a coarser uniform grid."""
    if n_coarse == n_fine:
        return dd
    scheme, spacing = _angular_stencil(dd)
    span = spacing * n_fine
    _, _, coarse, _ = uniform_periodic_diff_matrices(
        n=n_coarse, x_min=0.0, x_max=span, scheme=scheme
    )
    return coarse


def coarsen_operator(
    level: KineticOperator, n_theta: int, n_zeta: int, n_xi: int
) -> KineticOperator:
    """Rediscretize a simplified operator on a coarser ``(theta, zeta, xi)`` grid.

    *Rediscretization*, not a Galerkin triple product ``R A P``: the coarse
    operator is the same discrete drift-kinetic equation evaluated on the
    coarse grid, which costs nothing beyond restricting the geometry, keeps
    every coarse level in the block-tridiagonal-in-L / banded-in-angle form the
    smoothers and the coarsest block-Thomas solve need, and is the standard
    choice for non-symmetric convection-dominated operators (Trottenberg
    et al., section 2.8.3).

    The angular differentiation matrices are rebuilt by
    :func:`dkx.phase_space.uniform_periodic_diff_matrices` at the coarse size
    with the *same* ``createGrids.F90`` scheme and the same physical period;
    the flux-surface geometry ``(theta, zeta)`` fields are transferred with the
    full-weighting restriction, which for a smooth field is second-order
    accurate interpolation onto the coarse nodes; the quadrature weights are
    rebuilt (``2 pi / N`` per node in both angles, independent of the number of
    field periods); and ``xi`` is truncated, which is exactly what a
    lower-``Nxi`` discretization is.

    Args:
        level: a PAS-family operator from :func:`simplified_operator`.
        n_theta: coarse poloidal resolution.
        n_zeta: coarse toroidal resolution.
        n_xi: coarse Legendre resolution.

    Returns:
        The rediscretized :class:`KineticOperator`.
    """
    n_t, n_z, n_l = int(level.n_theta), int(level.n_zeta), int(level.n_xi)
    n_theta, n_zeta, n_xi = int(n_theta), int(n_zeta), int(n_xi)
    if (n_theta, n_zeta, n_xi) == (n_t, n_z, n_l):
        return level

    restrict_t, _ = periodic_transfer_matrices(n_t, n_theta)
    restrict_z, _ = periodic_transfer_matrices(n_z, n_zeta)
    restrict_tz = _separable_transfer({0: restrict_t, 1: restrict_z})

    fields = {
        name: restrict_tz(getattr(level, name))
        for name in (
            "b_hat",
            "db_hat_dtheta",
            "db_hat_dzeta",
            "d_hat",
            "b_hat_sup_theta",
            "b_hat_sup_zeta",
            "b_hat_sub_theta",
            "b_hat_sub_zeta",
        )
    }
    two_pi = 2.0 * np.pi
    pas = level.pas
    coarse_pas = replace(
        pas,
        coef=pas.coef[:, :, :n_xi],
        mask_xi=pas.mask_xi[:, :n_xi],
        n_xi_for_x=jnp.minimum(level.n_xi_for_x, n_xi),
    )
    return replace(
        level,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        ddtheta=_coarse_diff_matrix(level.ddtheta, n_t, n_theta),
        ddzeta=_coarse_diff_matrix(level.ddzeta, n_z, n_zeta)
        if n_z > 1
        else level.ddzeta,
        theta_weights=jnp.full((n_theta,), two_pi / n_theta, dtype=jnp.float64),
        zeta_weights=jnp.full((n_zeta,), two_pi / n_zeta, dtype=jnp.float64)
        if n_z > 1
        else level.zeta_weights,
        n_xi_for_x=jnp.minimum(level.n_xi_for_x, n_xi),
        xi_coupling_lower=level.xi_coupling_lower[:n_xi],
        xi_coupling_upper=level.xi_coupling_upper[:n_xi],
        pas=coarse_pas,
        **fields,
    )


# =============================================================================
# Level operator: pinned matvec + null-space pin
# =============================================================================


def _band_scale(level: KineticOperator) -> jnp.ndarray:
    """``(S, X)`` magnitude of the level operator's streaming/mirror band."""
    stream = (
        jnp.max(jnp.abs(level.b_hat_sup_theta / level.b_hat))
        * jnp.max(jnp.sum(jnp.abs(level.ddtheta), axis=1))
        + jnp.max(jnp.abs(level.b_hat_sup_zeta / level.b_hat))
        * jnp.max(jnp.sum(jnp.abs(level.ddzeta), axis=1))
        + jnp.max(jnp.abs(level.db_hat_dtheta) + jnp.abs(level.db_hat_dzeta))
        / jnp.max(level.b_hat)
    )
    ell = jnp.arange(level.n_xi, dtype=jnp.float64)
    couple = jnp.max(level.xi_coupling_upper * (ell + 2.0) + level.xi_coupling_lower)
    return (
        jnp.sqrt(level.t_hat / level.m_hat)[:, None] * level.x[None, :] * stream * couple
    )


def _shift(level: KineticOperator, weight: float) -> jnp.ndarray:
    """``(S, X)`` regularizing diagonal shift ``delta`` of the level operator.

    The simplified f-block has an *exact* null vector: a distribution constant
    on the flux surface with only the ``l = 0`` Legendre coefficient is
    annihilated by streaming (the angular derivative of a constant vanishes),
    by the mirror force (whose ``l = 1 <- l = 0`` factor carries ``l - 1 = 0``),
    by ExB, and by pitch-angle scattering (``nu l(l+1)/2 = 0`` at ``l = 0``).
    Something must remove it or the coarsest block-Thomas divides by zero.

    :func:`dkx.solve.build_coarse_preconditioner` removes it with a ``1e-8``
    relative diagonal floor plus an *adaptive* rank-one pin of that constant on
    the ``l = 0`` block (:func:`dkx.solve._l0_pin_gamma`), sized to the same
    ``1e-8`` level and applied only where that block really is singular.  Here
    only the floor is used -- a *uniform* shift ``A + delta I`` -- for two
    measured reasons:

    * ``delta I`` commutes with every basis change, so it is represented
      exactly in the pitch-collocation basis one smoother family works in, in
      the Legendre basis the other works in, and in the coarsest block-Thomas
      bands; a rank-one pin of a Legendre-``l = 0`` object is dense in
      collocation and would sit entirely in the smoother's remainder.  On the
      tiny NCSX operator the rank-one pin drives a smoother's error propagation
      from ``rho = 2.0`` to ``rho = 38``.
    * The magnitude matters far more than the form.  With the exact
      block-Thomas of this simplified operator as the tier-2 preconditioner on
      the NCSX ``11 x 21 x 41 x 5`` deck, GCROT needs **21** iterations at
      ``delta = 0``, 60 at ``1e-2`` of the mean collision diagonal, and never
      converges at ``1.0``.  (The *unconditional* full-strength rank-one pin
      that :func:`dkx.solve.build_coarse_preconditioner` used to apply cost 87
      on the same deck; sizing it by the floor instead is what recovered the
      21.)  The default weight is therefore the ``1e-8`` invertibility floor and
      nothing more -- enough to keep a collisionless, drift-free f-block, whose
      diagonal is *exactly* zero, out of a zero pivot.

    The scale is the mean collision diagonal, falling back to the streaming
    band magnitude on a collisionless deck.
    """
    scale = jnp.mean(jnp.abs(level.pas.coef), axis=2)  # (S, X)
    band = _band_scale(level)
    scale = jnp.where(scale > 0.0, scale, jnp.where(band > 0.0, band, 1.0))
    return float(weight) * scale


def _level_matvec(
    level: KineticOperator, weight: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Matrix-free apply of one level operator, regularized and pinned.

    Two regularizations are applied identically at every level (and in the
    coarsest block-Thomas solve) so the whole hierarchy is consistent:

    * the uniform diagonal shift ``delta`` of :func:`_shift`, which removes
      the exact null vector of the simplified f-block;
    * identity rows/columns on the ``Nxi_for_x``-truncated DOFs, which are
      exact zero rows of :meth:`KineticOperator.apply_f` (the same pinning
      :func:`dkx.solve._pinned_matvecs` applies to the full system).
    """
    mask = level._mask()[None, :, :, None, None]  # (1, X, L, 1, 1)
    shift = _shift(level, weight)[:, :, None, None, None]  # (S, X, 1, 1, 1)

    def matvec(f: jnp.ndarray) -> jnp.ndarray:
        active = f * mask
        return level.apply_f(active) + shift * active + f * (1.0 - mask)

    return matvec


def _coarse_solve(
    level: KineticOperator, weight: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Exact batched block-Thomas solve of the coarsest level operator.

    The classical tier-2 preconditioner, kept where it belongs: its
    ``O(Nxi Nspecies Nx (Ntheta Nzeta)**3)`` cost and
    ``O((Ntheta Nzeta)**2)`` storage are harmless once the angular grid is a
    handful of points across, and it makes the coarse-grid correction exact
    for every mode the coarse grid can represent.
    """
    n_s, n_x, n_xi, n_t, n_z = level.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x
    blocks = level.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower, diag, upper = (jnp.transpose(a, (1, 2, 0, 3, 4)) for a in blocks)

    eye = jnp.eye(n_tz, dtype=jnp.float64)
    mask = level._mask()  # (X, L)
    diag = diag + _shift(level, weight)[:, :, None, None, None] * eye[None, None, None, :, :]
    diag = diag + (1.0 - mask)[None, :, :, None, None] * eye[None, None, None, :, :]
    d4 = diag.reshape(batch, n_xi, n_tz, n_tz)

    factors = jax.vmap(block_thomas_factor)(
        lower.reshape(batch, n_xi, n_tz, n_tz), d4, upper.reshape(batch, n_xi, n_tz, n_tz)
    )

    def solve(r: jnp.ndarray) -> jnp.ndarray:
        g = r.reshape(batch, n_xi, n_tz)
        sol = jax.vmap(lambda f, v: block_thomas_solve(f, v))(factors, g)
        return sol.reshape(r.shape)

    return solve


# =============================================================================
# Smoother
# =============================================================================


def _streaming_eigenbasis(level: KineticOperator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(V, V^-1, lambda)`` diagonalizing the Legendre ``xi``-multiplication matrix.

    The streaming term couples ``L`` through the tridiagonal ``Y`` with
    ``Y[l, l-1] = l / (2l - 1)`` and ``Y[l, l+1] = (l + 1) / (2l + 3)``
    (``legendre_coupling_lower/upper``).  Its off-diagonal products are
    positive, so ``Y = D J D^-1`` with ``J`` symmetric tridiagonal, and the
    eigenvalues ``lambda_k`` are the Gauss-Legendre nodes of order ``Nxi``:
    ``V = D Q`` is the discrete Legendre transform and ``V^-1 = Q^T D^-1``.
    Going through the symmetric ``J`` keeps this real and well-conditioned at
    large ``Nxi``, which a direct nonsymmetric eigendecomposition would not.
    """
    lower = np.asarray(level.xi_coupling_lower, dtype=np.float64)
    upper = np.asarray(level.xi_coupling_upper, dtype=np.float64)
    n_xi = lower.shape[0]
    if n_xi == 1 or not np.any(np.abs(upper[:-1]) > 0.0):
        eye = np.eye(n_xi)
        return eye, eye, np.zeros((n_xi,))
    off = np.sqrt(upper[:-1] * lower[1:])
    scale = np.concatenate([[1.0], np.cumprod(np.sqrt(lower[1:] / upper[:-1]))])
    values, vectors = np.linalg.eigh(np.diag(off, 1) + np.diag(off, -1))
    v = scale[:, None] * vectors
    v_inv = vectors.T / scale[None, :]
    return v, v_inv, values


def _advection_bands(
    wind: jnp.ndarray, dd: jnp.ndarray, stencil: str
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(sub, main, super)`` bands of ``wind * d/dangle`` for the smoother.

    ``"upwind"`` (the default) discretizes the smoother's advection with the
    **first-order upwind** stencil rather than copying the level operator's
    centered one, and that is the single choice that makes this smoother work.
    A centered first derivative has the symbol ``i w sin(k h) / h``, which
    *vanishes* as ``k h -> pi``: its highest-frequency modes are almost in its
    null space, so a relaxation built on it cannot damp them and the coarse
    grid cannot represent them -- multigrid's complementarity fails on both
    sides at once.  The upwind symbol ``w (1 - e^{-i k h}) / h`` has modulus
    ``2 |w| |sin(k h / 2)| / h``, largest exactly at the highest frequency,
    which is why upwind (or otherwise dissipative) operators are the standard
    choice for convection-dominated multigrid (Brandt 1977, section 3.4;
    Trottenberg et al., section 7.4).  It also makes the plane operator
    diagonally dominant -- ``|w|/h`` on the diagonal against ``|w|/h`` off it
    -- so the banded factorization is stable without pivoting.

    ``"centered"`` reproduces the level operator's own tridiagonal part
    instead; it exists to measure what the upwind choice is worth.  On the NCSX
    ``11 x 21 x 41 x 5`` operator the pitch-collocation plane sweep measures
    ``mu = 5.09`` upwind against ``mu = 7.73`` centered over the angular high
    frequencies.

    Args:
        wind: iterate-shaped advection velocity along the axis.
        dd: the level's periodic derivative matrix for that angle.
        stencil: ``"upwind"`` or ``"centered"``.

    Returns:
        ``(sub, main, super)`` iterate-shaped bands.
    """
    row = np.asarray(dd[0])
    if row.shape[0] == 1:
        zero = jnp.zeros_like(wind)
        return zero, zero, zero
    if stencil == "upwind":
        _, spacing = _angular_stencil(dd)
        return (
            -jnp.maximum(wind, 0.0) / spacing,
            jnp.abs(wind) / spacing,
            jnp.minimum(wind, 0.0) / spacing,
        )
    if stencil != "centered":
        raise ValueError(f"unknown stencil {stencil!r}; expected 'upwind' or 'centered'")
    return wind * float(row[-1]), wind * float(row[0]), wind * float(row[1])


def _plane_angle(level: KineticOperator) -> int:
    """Which angle the ``"legendre_plane"`` smoother resolves exactly: 3 or 4.

    The one whose advective coupling ``max|v| / h`` is stronger.  On a
    stellarator flux surface that is normally ``zeta``: the toroidal grid
    covers only one field period (``2 pi / Nperiods``) so its spacing is
    several times finer, and ``B^zeta > B^theta`` by roughly ``1 / iota``.
    Line/plane relaxation goes in the strongly coupled direction and
    coarsening in the weakly coupled one -- the semicoarsening rule
    (Trottenberg et al., ch. 5).
    """
    if level.n_zeta < 3:
        return 3
    _, h_theta = _angular_stencil(level.ddtheta)
    _, h_zeta = _angular_stencil(level.ddzeta)
    strength_theta = float(jnp.max(jnp.abs(level.b_hat_sup_theta / level.b_hat))) / h_theta
    strength_zeta = float(jnp.max(jnp.abs(level.b_hat_sup_zeta / level.b_hat))) / h_zeta
    return 3 if strength_theta > strength_zeta else 4


def _legendre_plane_blocks(
    level: KineticOperator, axis: int, weight: float, pin: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Block-tridiagonal-in-L bands of the level operator on one angular line.

    The exact analogue of :meth:`KineticOperator.to_block_tridiagonal` with the
    *other* angle's derivative dropped: every remaining term -- the collision
    diagonal, the mirror force, the parallel streaming along this angle, the
    ExB drift along this angle, the regularizing shift -- is kept exactly, and
    the blocks are ``N x N`` with ``N`` one angular length instead of
    ``Ntheta*Nzeta``.

    That is the whole point of the smoother.  The two dominant terms of the
    drift-kinetic operator live in *different* bases: collisions are diagonal
    in the Legendre index and dense in pitch collocation, while streaming is
    diagonal in pitch collocation and ``L +- 1`` in Legendre.  Measured on the
    NCSX operator, keeping only the pitch-collocation diagonal of the collision
    operator discards 65% of it in the 2-norm (31% even with a lumped
    tridiagonal-in-pitch approximation), which no relaxation can survive: at
    the low speed nodes the collision operator is by far the largest term.
    Eliminating over ``L`` instead keeps both exactly, and the only term left
    to the coarse grid is the derivative along the *other* angle -- roughly 8%
    of the angular coupling on the NCSX ladder, which is exactly the anisotropy
    semicoarsening is designed to exploit.

    Returns ``(lower, diag, upper)`` of shape ``(B, L, N, N)`` with
    ``B = Nspecies * Nx * N_other``, ready for a batched block-Thomas
    factorization over ``L``.
    """
    n_s, n_x, n_xi, n_t, n_z = level.f_shape
    n_plane = n_t if axis == 3 else n_z
    n_other = n_z if axis == 3 else n_t
    eye = jnp.eye(n_plane, dtype=jnp.float64)

    sqrt_t_over_m = jnp.sqrt(level.t_hat / level.m_hat)  # (S,)
    if axis == 3:  # theta lines: (other, plane) = (zeta, theta)
        speed = jnp.transpose(level.b_hat_sup_theta / level.b_hat)  # (Z, T)
        derivative = level.ddtheta
        exb = jnp.transpose(level._exb_coefficients()[0]) if level.with_exb else None
        mirror_geom = jnp.transpose(
            level.b_hat_sup_theta * level.db_hat_dtheta
            + level.b_hat_sup_zeta * level.db_hat_dzeta
        )
        b_hat = jnp.transpose(level.b_hat)
    else:  # zeta lines: (other, plane) = (theta, zeta)
        speed = level.b_hat_sup_zeta / level.b_hat  # (T, Z)
        derivative = level.ddzeta
        exb = level._exb_coefficients()[1] if level.with_exb else None
        mirror_geom = (
            level.b_hat_sup_theta * level.db_hat_dtheta
            + level.b_hat_sup_zeta * level.db_hat_dzeta
        )
        b_hat = level.b_hat

    # Streaming along this angle, row-scaled: (S, other, plane, plane).
    stream = (
        sqrt_t_over_m[:, None, None, None]
        * speed[None, :, :, None]
        * derivative[None, None, :, :]
    )
    mirror_diag = -sqrt_t_over_m[:, None, None] * mirror_geom[None] / (2.0 * b_hat**2)
    mirror = mirror_diag[:, :, :, None] * eye[None, None, :, :]  # (S, other, plane, plane)

    ell = jnp.arange(n_xi, dtype=jnp.float64)
    mask = level._mask()  # (X, L)
    x = level.x  # (X,)

    def band(coupling: jnp.ndarray, mirror_factor: jnp.ndarray, shift_l: int) -> jnp.ndarray:
        """``(S, X, other, L, plane, plane)`` band from an L-coupling vector."""
        block = (
            stream[:, None, :, None, :, :] * coupling[None, None, None, :, None, None]
            + mirror[:, None, :, None, :, :] * mirror_factor[None, None, None, :, None, None]
        )
        rows = mask  # (X, L)
        columns = jnp.roll(mask, -shift_l, axis=1)
        if shift_l > 0:
            columns = columns.at[:, -shift_l:].set(0.0)
        elif shift_l < 0:
            columns = columns.at[:, :-shift_l].set(0.0)
        scale = x[:, None] * rows * columns  # (X, L)
        return block * scale[None, :, None, :, None, None]

    lower_coupling = level.xi_coupling_lower
    upper_coupling = level.xi_coupling_upper
    lower = band(lower_coupling, -lower_coupling * (ell - 1.0), -1)
    upper = band(upper_coupling, upper_coupling * (ell + 2.0), +1)

    diag_coef = level.pas.coef * mask[None] + _shift(level, weight)[:, :, None]  # (S,X,L)
    diag = diag_coef[:, :, None, :, None, None] * eye[None, None, None, None, :, :]
    diag = diag + (1.0 - mask)[None, :, None, :, None, None] * eye[None, None, None, None, :, :]
    if exb is not None:
        exb_block = exb[:, :, None] * derivative[None, :, :]  # (other, plane, plane)
        diag = diag + (
            exb_block[None, None, :, None, :, :] * mask[None, :, None, :, None, None]
        )

    shape = (n_s, n_x, n_other, n_xi, n_plane, n_plane)
    lower, diag, upper = (jnp.broadcast_to(a, shape) for a in (lower, diag, upper))

    # Pin the constant-along-the-line null vector of the l = 0 block, ONCE PER
    # LINE.  The full two-dimensional operator has exactly one such vector (a
    # distribution constant over the whole flux surface at l = 0), removed by
    # the bordered constraint rows.  Slicing the surface into independent lines
    # manufactures one *per line* -- ``N_other`` spurious near-null directions
    # that are absent from the operator being preconditioned -- and the
    # smoother's inverse explodes along every one of them: measured on the
    # NCSX 11x21x41x5 operator the theta-direction smoothing factor is 602
    # without this pin and 0.4 with it.  The pin lives only in the smoother's
    # ``M``, never in the level operator or the coarse solve, so it changes the
    # relaxation and nothing else.
    if pin > 0.0:
        gamma = pin * jnp.mean(jnp.abs(level.pas.coef), axis=2)  # (S, X)
        gamma = jnp.where(gamma > 0.0, gamma, pin * _band_scale(level))
        rank_one = jnp.ones((n_plane, n_plane), dtype=jnp.float64) / n_plane
        diag = diag.at[:, :, :, 0].add(
            gamma[:, :, None, None, None] * rank_one[None, None, None, :, :]
        )

    flat = (n_s * n_x * n_other, n_xi, n_plane, n_plane)
    return tuple(a.reshape(flat) for a in (lower, diag, upper))


def _legendre_plane_solve(
    level: KineticOperator, axis: int, weight: float, pin: float = 1.0
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Batched exact solve of the ``(L, angle)`` planes of the level operator."""
    n_s, n_x, n_xi, n_t, n_z = level.f_shape
    n_plane = n_t if axis == 3 else n_z
    n_other = n_z if axis == 3 else n_t
    lower, diag, upper = _legendre_plane_blocks(level, axis, weight, pin)
    factors = jax.vmap(block_thomas_factor)(lower, diag, upper)

    #: (S, X, L, T, Z) <-> (S, X, other, L, plane) for each plane orientation.
    forward = (0, 1, 4, 2, 3) if axis == 3 else (0, 1, 3, 2, 4)
    backward = (0, 1, 3, 4, 2) if axis == 3 else (0, 1, 3, 2, 4)

    def solve(r: jnp.ndarray) -> jnp.ndarray:
        g = jnp.transpose(r, forward).reshape(n_s * n_x * n_other, n_xi, n_plane)
        y = jax.vmap(block_thomas_solve)(factors, g)
        y = y.reshape(n_s, n_x, n_other, n_xi, n_plane)
        return jnp.transpose(y, backward)

    return solve


def _level_smoother(level: KineticOperator, settings: MultigridSettings, weight: float):
    """The relaxation sweep of one level.

    ``smoother="legendre_plane"`` (the default) relaxes the ``(L, angle)``
    planes exactly with :func:`_legendre_plane_solve`, wrapped by
    :func:`solvax.smoothers.relaxation` so the sweep refreshes its residual
    against the true level operator.  Everything except the *other* angle's
    derivative is inverted exactly, which is the plane relaxation Trottenberg
    et al. (section 5.2) prescribe when an operator is strongly coupled along
    two directions at once.

    The remaining options relax in the pitch-collocation basis ``V`` that
    diagonalizes the streaming ``L``-coupling, where the operator is, per pitch
    node ``k``, the advection-reaction

        ``[x lambda_k v_theta + u_theta] d/dtheta``
        ``+ [x lambda_k v_zeta + u_zeta] d/dzeta + reaction``

    with a definite wind direction, relaxed by
    :func:`solvax.smoothers.plane_smoother` or by upwind-ordered cyclic line
    sweeps (:func:`solvax.smoothers.upwind_smoother`) composed with
    :func:`solvax.smoothers.alternating_smoother`.  They are exact for
    streaming and ExB but see the collision operator only through its
    collocation *diagonal*, and are kept because that is the measurement that
    justifies the default (see :func:`_legendre_plane_blocks`).
    """
    n_s, n_x, n_xi, n_t, n_z = level.f_shape
    mask_only = level._mask()[None, :, :, None, None]
    uniform_mask = bool(np.all(np.asarray(level._mask()) == 1.0))

    def wrap(core: Callable[[jnp.ndarray], jnp.ndarray]):
        """Identity on the ``Nxi_for_x``-truncated (identity-row) DOFs."""
        if uniform_mask:
            return core

        def solve(r: jnp.ndarray) -> jnp.ndarray:
            return core(r * mask_only) * mask_only + r * (1.0 - mask_only)

        return solve

    legendre_part = None
    if settings.smoother in ("legendre_plane", "hybrid"):
        axis = _plane_angle(level)
        legendre_part = relaxation(
            wrap(_legendre_plane_solve(level, axis, weight, settings.plane_pin)),
            omega=settings.omega_xi,
        )
        if settings.smoother == "legendre_plane":
            return legendre_part

    v, v_inv, lam = _streaming_eigenbasis(level)
    v_j, v_inv_j, lam_j = (jnp.asarray(a) for a in (v, v_inv, lam))

    sqrt_t_over_m = jnp.sqrt(level.t_hat / level.m_hat)  # (S,)
    v_theta = sqrt_t_over_m[:, None, None] * (level.b_hat_sup_theta / level.b_hat)
    v_zeta = sqrt_t_over_m[:, None, None] * (level.b_hat_sup_zeta / level.b_hat)
    mirror_geom = (
        level.b_hat_sup_theta * level.db_hat_dtheta
        + level.b_hat_sup_zeta * level.db_hat_dzeta
    )
    mirror = -sqrt_t_over_m[:, None, None] * mirror_geom[None] / (2.0 * level.b_hat**2)

    if level.with_exb:
        exb_theta, exb_zeta = level._exb_coefficients()
    else:
        zero_tz = jnp.zeros((n_t, n_z), dtype=jnp.float64)
        exb_theta = exb_zeta = zero_tz

    # (S, X, L, T, Z) winds: streaming (pitch node lambda_k) plus ExB.
    x_lam = level.x[:, None] * lam_j[None, :]  # (X, L)
    wind_t = (
        x_lam[None, :, :, None, None] * v_theta[:, None, None, :, :]
        + exb_theta[None, None, None, :, :]
    )
    wind_z = (
        x_lam[None, :, :, None, None] * v_zeta[:, None, None, :, :]
        + exb_zeta[None, None, None, :, :]
    )

    # Reaction: collision diagonal and mirror force, both projected onto the
    # pitch-collocation diagonal (they are dense in that basis).
    ell = jnp.arange(n_xi, dtype=jnp.float64)
    mirror_l = jnp.diag(level.xi_coupling_upper[:-1] * (ell[:-1] + 2.0), 1) + jnp.diag(
        -level.xi_coupling_lower[1:] * (ell[1:] - 1.0), -1
    )
    mirror_k = jnp.sum((v_inv_j @ mirror_l) * v_j.T, axis=1)  # (L,)
    project = v_inv_j * v_j.T  # (L, L): W[k, l] = V^-1[k, l] V[l, k]
    coll_k = jnp.einsum("kl,sxl->sxk", project, level.pas.coef)  # (S, X, L)

    shift = _shift(level, weight)  # (S, X)
    sub_t, main_t, sup_t = _advection_bands(wind_t, level.ddtheta, settings.stencil)
    sub_z, main_z, sup_z = _advection_bands(wind_z, level.ddzeta, settings.stencil)
    mirror_react = (
        level.x[None, :, None] * mirror_k[None, None, :]
    )[:, :, :, None, None] * mirror[:, None, None, :, :]
    if settings.absolute_reaction:
        # Only the magnitude of the pitch-space reaction is kept, so the plane
        # operator stays diagonally dominant even where the mirror force's
        # collocation diagonal is negative.
        mirror_react = jnp.abs(mirror_react)
        coll_k = jnp.abs(coll_k)
    react = coll_k[:, :, :, None, None] + mirror_react + shift[:, :, None, None, None]
    diag = react + main_t + main_z

    def line(axis: int, wind: jnp.ndarray, sub: jnp.ndarray, sup: jnp.ndarray):
        return upwind_smoother(
            wind, sub, diag, sup, axis=axis, order=settings.order, periodic=True
        )

    kind = "plane" if settings.smoother == "hybrid" else settings.smoother
    if kind not in ("plane", "alternating", "theta", "zeta"):
        raise ValueError(f"unknown smoother {settings.smoother!r}")
    if n_z < 3:  # axisymmetric: there is no zeta line to relax
        kind = "theta"
    if kind == "plane":
        # Order the plane axes so the SHORTER one is second: the banded
        # factors are O(n_first * n_second**2).
        if n_t <= n_z:
            axes, first, second = (4, 3), (sub_z, sup_z), (sub_t, sup_t)
        else:
            axes, first, second = (3, 4), (sub_t, sup_t), (sub_z, sup_z)
        angular = plane_smoother(
            diag, first, second, axes=axes, periodic=(True, True)
        )
    elif kind == "theta":
        angular = line(3, wind_t, sub_t, sup_t)
    elif kind == "zeta":
        angular = line(4, wind_z, sub_z, sup_z)
    else:
        angular = alternating_smoother(
            [line(3, wind_t, sub_t, sup_t), line(4, wind_z, sub_z, sup_z)]
        )

    def collocation_matvec(g: jnp.ndarray) -> jnp.ndarray:
        out = diag * g + sup_t * jnp.roll(g, -1, axis=3) + sub_t * jnp.roll(g, 1, axis=3)
        if n_z > 1:
            out = out + sup_z * jnp.roll(g, -1, axis=4) + sub_z * jnp.roll(g, 1, axis=4)
        return out

    def angular_solve(r: jnp.ndarray) -> jnp.ndarray:
        g = jnp.einsum("kl,sxltz->sxktz", v_inv_j, r)
        y = angular(collocation_matvec, jnp.zeros_like(g), g)
        return jnp.einsum("lk,sxktz->sxltz", v_j, y)

    parts = [relaxation(wrap(angular_solve), omega=settings.omega)]
    if legendre_part is not None:
        parts.append(legendre_part)
    if settings.smooth_xi and n_xi > 1:
        shape = (n_s, n_x, n_xi, n_t, n_z)
        lower_l = jnp.broadcast_to(
            (level.x[None, :, None] * (-level.xi_coupling_lower * (ell - 1.0))[None, None, :])[
                :, :, :, None, None
            ]
            * mirror[:, None, None, :, :],
            shape,
        )
        couple_up = level.xi_coupling_upper * (ell + 2.0)
        couple_up = couple_up.at[-1].set(0.0)  # row Nxi-1 has no l+1 partner
        upper_l = jnp.broadcast_to(
            (level.x[None, :, None] * couple_up[None, None, :])[:, :, :, None, None]
            * mirror[:, None, None, :, :],
            shape,
        )
        diag_l = jnp.broadcast_to(
            (level.pas.coef + shift[:, :, None])[:, :, :, None, None], shape
        )
        pitch = tridiagonal_smoother(lower_l, diag_l, upper_l, axis=2, omega=1.0)
        zero = lambda v: jnp.zeros_like(v)  # noqa: E731
        parts.append(
            relaxation(
                wrap(lambda r: pitch(zero, jnp.zeros_like(r), r)),
                omega=settings.omega_xi,
            )
        )
    return alternating_smoother(parts) if len(parts) > 1 else parts[0]


# =============================================================================
# Assembly
# =============================================================================


def _levels(op: KineticOperator, settings: MultigridSettings):
    """Rediscretized level operators, finest first (the coarsest one last)."""
    shapes = hierarchy_shapes(op, settings)
    return tuple(coarsen_operator(op, *shape) for shape in shapes)


def build_multigrid_f_inverse(
    op: KineticOperator,
    *,
    drop_l_coupling: bool = False,
    settings: MultigridSettings = MultigridSettings(),
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], tuple[tuple[int, int, int], ...]]:
    """Multigrid approximate inverse of the simplified f-block.

    Args:
        op: the full tier-2 operator.
        drop_l_coupling: the ``preconditioner_xi=1`` knob.
        settings: cycle/hierarchy/smoother knobs.

    Returns:
        ``(a_inv, shapes)`` where ``a_inv`` maps a flat ``(f_size,)`` residual
        to an approximate solution of the simplified f-block, and ``shapes``
        lists the ``(Ntheta, Nzeta, Nxi)`` of every level.
    """
    _require_multigrid()
    simplified = simplified_operator(op, drop_l_coupling=drop_l_coupling)
    levels = _levels(simplified, settings)
    shapes = tuple((int(a.n_theta), int(a.n_zeta), int(a.n_xi)) for a in levels)
    f_shape = simplified.f_shape

    built = []
    for fine, coarse in zip(levels[:-1], levels[1:]):
        matrices: dict[int, jnp.ndarray] = {}
        prolong_matrices: dict[int, jnp.ndarray] = {}
        if coarse.n_xi != fine.n_xi:
            matrices[2], prolong_matrices[2] = xi_transfer_matrices(fine.n_xi, coarse.n_xi)
        if coarse.n_theta != fine.n_theta:
            matrices[3], prolong_matrices[3] = periodic_transfer_matrices(
                fine.n_theta, coarse.n_theta
            )
        if coarse.n_zeta != fine.n_zeta:
            matrices[4], prolong_matrices[4] = periodic_transfer_matrices(
                fine.n_zeta, coarse.n_zeta
            )
        built.append(
            MultigridLevel(
                matvec=_level_matvec(fine, settings.shift),
                smoother=_level_smoother(fine, settings, settings.shift),
                restrict=_separable_transfer(matrices),
                prolong=_separable_transfer(prolong_matrices),
            )
        )

    cycle = solvax_multigrid(
        built,
        _coarse_solve(levels[-1], settings.shift),
        cycle=settings.cycle,
        pre_smooth=settings.pre_smooth,
        post_smooth=settings.post_smooth,
        cycles=settings.cycles,
    )

    def a_inv(r: jnp.ndarray) -> jnp.ndarray:
        return cycle(r.reshape(f_shape)).reshape(r.shape)

    return a_inv, shapes


def build_multigrid_preconditioner(
    op: KineticOperator,
    *,
    drop_l_coupling: bool = False,
    settings: MultigridSettings = MultigridSettings(),
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Drop-in multigrid replacement for :func:`dkx.solve.build_coarse_preconditioner`.

    Same contract, same simplified operator, same exact elimination of the
    bordered constraint / ``Phi1`` rows -- only the inner approximate inverse
    of the f-block changes, from a batched block-Thomas factorization with
    dense ``(Ntheta*Nzeta)`` blocks to a multigrid V-cycle.

    The transposed preconditioner is obtained with :func:`jax.linear_transpose`
    of the cycle itself.  The cycle is a *linear* function of its argument
    (fixed transfers, fixed smoother factors, fixed coarse factorization), so
    its transpose is exact -- no second hierarchy, and no risk of the two
    drifting apart, which is what the tier-2 adjoint guard would otherwise
    catch.

    Args:
        op: the full tier-2 operator.
        drop_l_coupling: the ``preconditioner_xi=1`` knob.
        settings: cycle/hierarchy/smoother knobs.

    Returns:
        ``(precond, precond_t)`` — approximate inverses of ``K`` and ``K^T``
        on flat ``(total_size,)`` vectors.
    """
    from dkx.solve import (  # noqa: PLC0415
        _SCHUR_ACCEPTS_D_BLOCK,
        _bordered_schur_precond,
        _materialize_borders,
        _materialize_full_border,
        schur_projected_precond,
    )

    a_inv, _ = build_multigrid_f_inverse(
        op, drop_l_coupling=drop_l_coupling, settings=settings
    )
    template = jnp.zeros((op.f_size,), dtype=jnp.float64)
    transposed = jax.linear_transpose(a_inv, template)

    def a_inv_t(r: jnp.ndarray) -> jnp.ndarray:
        return transposed(r)[0]

    if op.include_phi1:
        b_cols, c_rows, d_block = _materialize_full_border(op)
        if _SCHUR_ACCEPTS_D_BLOCK:
            return (
                schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block),
                schur_projected_precond(a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T),
            )
        return (
            _bordered_schur_precond(a_inv, b_cols, c_rows, d_block),
            _bordered_schur_precond(a_inv_t, c_rows.T, b_cols.T, d_block.T),
        )
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = _materialize_borders(op)
    return (
        schur_projected_precond(a_inv, b_cols, c_rows),
        schur_projected_precond(a_inv_t, c_rows.T, b_cols.T),
    )


def measure_smoothing_factor(
    op: KineticOperator,
    *,
    settings: MultigridSettings = MultigridSettings(),
    seed: int = 0,
    steps: int = 24,
) -> float:
    """Measured smoothing factor ``mu`` of the finest-level smoother.

    :func:`solvax.smoothers.smoothing_factor` power-iterates the error
    propagation operator ``I - M^-1 A`` restricted to the high-frequency modes
    the coarsening cannot represent -- the empirical counterpart of Brandt's
    local Fourier analysis ``mu``.  Measured on the *real* simplified DKX
    operator, not a model problem.

    Values well below one mean the smoother and the coarse grid are
    complementary; ``mu`` near one means the cycle will not converge no matter
    what cycle index or transfers are used.
    """
    _require_multigrid()
    simplified = simplified_operator(op)
    smoother = _level_smoother(simplified, settings, settings.shift)
    coarsen = (False, False, bool(settings.coarsen_xi), True, True)
    value = smoothing_factor(
        smoother,
        _level_matvec(simplified, settings.shift),
        simplified.f_shape,
        key=jax.random.PRNGKey(seed),
        coarsen=coarsen,
        steps=steps,
    )
    return float(value)


# =============================================================================
# Pitch-basis diagnostics: why no relaxation smooths in a Legendre-modal basis
# =============================================================================
#
# Everything below is *measurement*, not a solver path.  It exists so the
# negative result in this module's docstring -- that the multigrid family cannot
# be made to work on a Legendre-modal pitch discretization -- is reproducible
# from the repository rather than asserted, and so a future lane can re-measure
# it on its own deck before spending a week on the alternative.


# Backward-biased (wind blowing from smaller index) first-derivative stencils,
# as offsets in units of the grid spacing.  ``up*`` are the textbook one-sided
# and upwind-biased schemes; ``wide*`` skip near neighbours to buy diagonal
# dominance at the same formal order -- the same trade Leonard's QUICK family
# makes for boundedness (B. P. Leonard, Comput. Methods Appl. Mech. Eng. 19, 59
# (1979)).  ``ctr2`` is the centered scheme SFINCS and dkx actually use on the
# angles (``thetaDerivativeScheme`` 1), included as the zero-dissipation
# reference.
UPWIND_STENCILS: dict[str, tuple[int, ...]] = {
    "up1": (-1, 0),
    "up2": (-2, -1, 0),
    "up3": (-2, -1, 0, 1),
    "wide2": (-4, -1, 0),
    "wide4": (-4, -3, -1, 0, 2),
    "ctr2": (-1, 0, 1),
}


def _stencil_weights(offsets: tuple[int, ...], order: int = 1) -> np.ndarray:
    """Finite-difference weights of the ``order``-th derivative at ``offsets*h``."""
    nodes = np.asarray(offsets, dtype=np.float64)
    vander = np.vander(nodes, nodes.size, increasing=True).T
    rhs = np.zeros(nodes.size)
    rhs[order] = float(math.factorial(order))
    return np.linalg.solve(vander, rhs)


def stencil_matrices(
    n: int, h: float, name: str, *, periodic: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(backward-biased, forward-biased)`` first-derivative pair of a stencil.

    An upwinded advection ``w df/dx`` uses the backward-biased matrix wherever
    ``w > 0`` and the forward-biased one wherever ``w <= 0``.  ``periodic=False``
    reflects the stencil at the ends (an even extension), which is the right
    closure for the pitch angle: ``f`` is a function of ``xi = cos(alpha)`` and
    is therefore even about ``alpha = 0`` and ``alpha = pi``, where the mirror
    advection speed vanishes anyway.

    Args:
        n: number of grid points.
        h: uniform spacing.
        name: key of :data:`UPWIND_STENCILS`.
        periodic: wrap (angles) rather than reflect (pitch).

    Returns:
        Two ``(n, n)`` arrays.
    """
    offsets = UPWIND_STENCILS[name]

    def build(offs: tuple[int, ...]) -> np.ndarray:
        w = _stencil_weights(offs, 1) / h
        mat = np.zeros((n, n))
        for j, off in enumerate(offs):
            for i in range(n):
                k = i + off
                if periodic:
                    k %= n
                else:
                    if k < 0:
                        k = -1 - k
                    if k > n - 1:
                        k = 2 * n - 1 - k
                    k = min(max(k, 0), n - 1)
                mat[i, k] += w[j]
        return mat

    return build(offsets), build(tuple(-o for o in offsets)[::-1])


def line_diagonal_dominance(
    matrix: np.ndarray, shape: tuple[int, ...], axis: int
) -> tuple[float, float]:
    """``(min, median)`` diagonal dominance of the line blocks along ``axis``.

    An alternating line block-Jacobi smoother inverts, for each coordinate in
    turn, the block of ``matrix`` that runs along that coordinate with all the
    others frozen.  The relevant question about such a block is whether it is
    diagonally dominant, ``d = min_i |a_ii| / sum_{j != i} |a_ij|``: a damped
    Jacobi/block-Jacobi sweep is a smoother when ``d`` is order one and
    *amplifies* when ``d`` is near zero (Trottenberg et al., section 2.1;
    Brandt, Math. Comp. **31**, 333 (1977), section 3).

    Args:
        matrix: dense operator on the flattened ``shape`` grid.
        shape: grid shape the operator acts on.
        axis: which axis the line runs along.

    Returns:
        The minimum and median of ``d`` over all lines and rows.
    """
    n_dim = len(shape)
    m = shape[axis]
    rest = tuple(s for i, s in enumerate(shape) if i != axis)
    n_rest = int(np.prod(rest)) if rest else 1
    blocks = np.moveaxis(
        matrix.reshape(tuple(shape) + tuple(shape)), (axis, n_dim + axis), (0, 1)
    ).reshape((m, m, n_rest, n_rest))
    blocks = np.einsum("abrr->rab", blocks)
    diag = np.abs(np.einsum("rii->ri", blocks))
    off = np.abs(blocks).sum(axis=2) - diag
    finite = off > 1e-14
    ratio = np.where(finite, diag / np.where(finite, off, 1.0), np.inf)
    return float(ratio.min()), float(np.median(ratio))


def dense_simplified_block(
    op: KineticOperator, *, species: int = 0, speed: int = 0
) -> np.ndarray:
    """Dense ``(Nxi*Ntheta*Nzeta)`` square block of the simplified operator.

    The simplified operator (:func:`simplified_operator`) is uncoupled over
    ``(species, x)``, so one such block is the whole thing the classical tier-2
    preconditioner factors, for one speed node of one species.  Index order is
    ``(L, theta, zeta)`` with ``zeta`` fastest, matching
    :meth:`KineticOperator.to_block_tridiagonal`.

    Only for analysis and tests -- it materializes ``(Nxi*Ntheta*Nzeta)**2``
    entries.
    """
    blocks = simplified_operator(op).to_block_tridiagonal()
    lower = np.asarray(blocks.lower[:, species, speed])
    diag = np.asarray(blocks.diag[:, species, speed])
    upper = np.asarray(blocks.upper[:, species, speed])
    n_l, n_tz, _ = diag.shape
    out = np.zeros((n_l * n_tz, n_l * n_tz))
    for ell in range(n_l):
        row = slice(ell * n_tz, (ell + 1) * n_tz)
        out[row, row] = diag[ell]
        if ell >= 1:
            out[row, (ell - 1) * n_tz : ell * n_tz] = lower[ell]
        if ell + 1 < n_l:
            out[row, (ell + 1) * n_tz : (ell + 2) * n_tz] = upper[ell]
    return out


@dataclass(frozen=True)
class PitchCollocationSurrogate:
    """A finite-difference discretization of the simplified operator on a pitch grid.

    Attributes:
        matrix: dense ``(Nalpha*Ntheta*Nzeta)`` square operator, index order
            ``(alpha, theta, zeta)``.
        to_nodal: ``(Nalpha, Nxi)`` Legendre Vandermonde ``P_L(xi_k)`` -- turns
            dkx's Legendre coefficients into values on the pitch grid.
        to_modal: its pseudo-inverse, ``(Nxi, Nalpha)``.
        shape: ``(Nalpha, Ntheta, Nzeta)``.
        alpha: the pitch-angle nodes.
    """

    matrix: np.ndarray
    to_nodal: np.ndarray
    to_modal: np.ndarray
    shape: tuple[int, int, int]
    alpha: np.ndarray

    def nodal(self, modal: np.ndarray) -> np.ndarray:
        """Legendre coefficients ``(Nxi, T, Z)`` -> grid values ``(Nalpha, T, Z)``."""
        n_xi = self.to_nodal.shape[1]
        v = np.asarray(modal).reshape(n_xi, self.shape[1], self.shape[2])
        return np.einsum("al,ltz->atz", self.to_nodal, v).reshape(-1)

    def modal(self, nodal: np.ndarray) -> np.ndarray:
        """Grid values ``(Nalpha, T, Z)`` -> Legendre coefficients ``(Nxi, T, Z)``."""
        g = np.asarray(nodal).reshape(self.shape)
        return np.einsum("la,atz->ltz", self.to_modal, g).reshape(-1)


def _legendre_vandermonde(xi: np.ndarray, n_l: int) -> np.ndarray:
    out = np.zeros((xi.size, n_l))
    out[:, 0] = 1.0
    if n_l > 1:
        out[:, 1] = xi
    for ell in range(1, n_l - 1):
        out[:, ell + 1] = ((2 * ell + 1) * xi * out[:, ell] - ell * out[:, ell - 1]) / (ell + 1)
    return out


def _uniform_spacing(ddx: np.ndarray, n: int) -> float:
    """Recover the grid spacing of a centered periodic first-derivative matrix.

    ``uniform_periodic_diff_matrices`` builds circulant matrices; the centered
    schemes dkx uses have first-neighbour weight ``1/2`` (3-point) or ``2/3``
    (5-point), so the spacing follows from the ``+1`` entry.
    """
    if n == 1:
        return 1.0
    row = np.asarray(ddx)[0]
    n_nonzero = int(np.count_nonzero(np.abs(row) > 1e-13))
    if n_nonzero == 2:
        return 0.5 / float(row[1])
    if n_nonzero == 4:
        return (2.0 / 3.0) / float(row[1])
    raise ValueError(
        "the pitch-collocation surrogate needs a 3- or 5-point centered angular "
        f"scheme to recover the grid spacing from, got {n_nonzero} nonzero weights"
    )


def pitch_collocation_surrogate(
    op: KineticOperator,
    *,
    species: int = 0,
    speed: int = 0,
    n_alpha: int | None = None,
    angular_stencil: str = "up1",
    pitch_stencil: str | None = None,
) -> PitchCollocationSurrogate:
    r"""Discretize the *same continuum operator* on a pitch-angle collocation grid.

    Per ``(species, speed)`` the simplified operator is, in continuum form,

    .. math::

        K f = a \left[ \xi\, \mathbf{b}\cdot\nabla f
                       - \frac{1-\xi^2}{2}\,
                         (\mathbf{b}\cdot\nabla \ln B)\, \partial_\xi f \right]
              + w^{E}_\theta \partial_\theta f + w^{E}_\zeta \partial_\zeta f
              + C f ,

    with ``a = x sqrt(That/mHat)`` and ``C`` diagonal in the Legendre index.
    dkx discretizes it Legendre-*modally* in pitch; this builds the alternative
    in which pitch is a *grid*, uniform in ``alpha`` with ``xi = cos alpha``
    (half-index, so the ends where the mirror speed vanishes are not nodes), and
    the streaming/mirror advection is upwind-differenced in all three of
    ``(alpha, theta, zeta)``.  ``C`` is carried across exactly, as
    ``V diag(c_L) V^+`` with ``V`` the Legendre Vandermonde -- dense in pitch but
    tiny, and it keeps the comparison about the *advection* discretization
    alone.

    Coupled to dkx's Legendre state by ``V`` and ``V^+``, this is the classical
    "low-order finite-difference preconditioner for a spectral operator"
    construction (S. A. Orszag, J. Comput. Phys. **37**, 70 (1980); M. O. Deville
    & E. H. Mund, J. Comput. Phys. **60**, 517 (1985)).  What it is *for* here is
    the measurement in this module's docstring: it is the discretization in
    which an alternating line block-Jacobi smoother converges, and dkx's is not.

    Args:
        op: the full tier-2 operator (only the simplified part is used).
        species: species index.
        speed: speed-node index.
        n_alpha: pitch-grid size; defaults to ``op.n_xi``.
        angular_stencil: key of :data:`UPWIND_STENCILS` for ``theta``/``zeta``.
        pitch_stencil: key for ``alpha``; defaults to ``angular_stencil``.

    Returns:
        A :class:`PitchCollocationSurrogate`.  Dense: only for analysis on small
        decks.
    """
    simplified = simplified_operator(op)
    n_theta, n_zeta, n_xi = op.n_theta, op.n_zeta, op.n_xi
    n_a = int(n_alpha or n_xi)

    a_scale = float(
        np.asarray(op.x)[speed]
        * np.sqrt(np.asarray(op.t_hat)[species] / np.asarray(op.m_hat)[species])
    )
    b_hat = np.asarray(op.b_hat)
    v_theta = np.asarray(op.b_hat_sup_theta) / b_hat
    v_zeta = np.asarray(op.b_hat_sup_zeta) / b_hat
    b_grad_ln_b = (
        np.asarray(op.b_hat_sup_theta) * np.asarray(op.db_hat_dtheta)
        + np.asarray(op.b_hat_sup_zeta) * np.asarray(op.db_hat_dzeta)
    ) / b_hat**2
    if op.with_exb:
        coef_theta, coef_zeta = (np.asarray(c) for c in op._exb_coefficients())
    else:
        coef_theta = np.zeros_like(b_hat)
        coef_zeta = np.zeros_like(b_hat)
    coll = np.asarray(simplified.pas.coef)[species, speed][:n_xi]

    alpha = np.pi * (2 * np.arange(n_a) + 1) / (2 * n_a)
    xi = np.cos(alpha)
    h_alpha = np.pi / n_a
    h_theta = 2.0 * np.pi / n_theta
    h_zeta = _uniform_spacing(np.asarray(op.ddzeta), n_zeta)

    # advection velocities; d(alpha)/ds follows from d(xi)/ds and xi = cos alpha
    w_alpha = a_scale * (np.sin(alpha)[:, None, None] / 2.0) * b_grad_ln_b[None]
    w_theta = a_scale * xi[:, None, None] * v_theta[None] + coef_theta[None]
    w_zeta = a_scale * xi[:, None, None] * v_zeta[None] + coef_zeta[None]

    pitch_stencil = pitch_stencil or angular_stencil
    pairs = {
        0: stencil_matrices(n_a, h_alpha, pitch_stencil, periodic=False),
        1: stencil_matrices(n_theta, h_theta, angular_stencil, periodic=True),
        2: stencil_matrices(n_zeta, h_zeta, angular_stencil, periodic=True),
    }
    eye = {0: np.eye(n_a), 1: np.eye(n_theta), 2: np.eye(n_zeta)}

    def lift(axis: int, mat: np.ndarray) -> np.ndarray:
        factors = [eye[0], eye[1], eye[2]]
        factors[axis] = mat
        return np.kron(factors[0], np.kron(factors[1], factors[2]))

    shape = (n_a, n_theta, n_zeta)
    size = n_a * n_theta * n_zeta
    matrix = np.zeros((size, size))
    for axis, field in ((0, w_alpha), (1, w_theta), (2, w_zeta)):
        flat = np.broadcast_to(field, shape).reshape(-1)
        back, fwd = pairs[axis]
        matrix += (flat * (flat > 0))[:, None] * lift(axis, back)
        matrix += (flat * (flat <= 0))[:, None] * lift(axis, fwd)

    vander = _legendre_vandermonde(xi, n_xi)
    pinv = np.linalg.pinv(vander)
    matrix += lift(0, vander @ np.diag(coll) @ pinv)
    return PitchCollocationSurrogate(
        matrix=matrix, to_nodal=vander, to_modal=pinv, shape=shape, alpha=alpha
    )


def line_smoother_spectral_radius(
    matrix: np.ndarray, shape: tuple[int, ...], *, omega: float = 1.0, floor: float = 1e-12
) -> float:
    """Spectral radius of an alternating line block-Jacobi error propagator.

    Builds ``S = prod_axes (I - omega M_axis^{-1} A)`` with ``M_axis`` the block
    diagonal of ``A`` whose blocks run along that axis -- one exact line solve
    per coordinate, applied multiplicatively, which is the standard remedy for
    an advection whose direction is not grid-aligned (Trottenberg et al.,
    sections 5.1 and 7.4) -- and returns ``rho(S)``.

    ``rho(S) < 1`` is necessary for *any* multigrid cycle built on this
    relaxation: a coarse-grid correction cannot rescue a divergent smoother.

    Dense, ``O(N^3)``: for analysis on small decks only.
    """
    size = matrix.shape[0]
    prop = np.eye(size)
    for axis in range(len(shape)):
        m = shape[axis]
        rest = tuple(s for i, s in enumerate(shape) if i != axis)
        n_rest = int(np.prod(rest)) if rest else 1
        blocks = np.moveaxis(
            matrix.reshape(tuple(shape) + tuple(shape)),
            (axis, len(shape) + axis),
            (0, 1),
        ).reshape((m, m, n_rest, n_rest))
        blocks = np.einsum("abrr->rab", blocks)
        blocks = blocks + floor * np.abs(blocks).sum(axis=(1, 2)).max() * np.eye(m)[None]
        rhs = (matrix @ prop).reshape(tuple(shape) + (size,))
        rhs = np.moveaxis(rhs, axis, 0).reshape(m, n_rest, size).transpose(1, 0, 2)
        step = np.linalg.solve(blocks, rhs).transpose(1, 0, 2)
        step = np.moveaxis(step.reshape((m,) + rest + (size,)), 0, axis)
        prop = prop - omega * step.reshape(size, size)
    return float(np.abs(np.linalg.eigvals(prop)).max())
