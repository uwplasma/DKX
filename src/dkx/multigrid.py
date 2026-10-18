"""Geometric-multigrid preconditioner for the recycled Krylov solve.

The recycled Krylov route (:func:`dkx.solve.solve` with ``method="gmres"``) is the
only route open to the physics that has no block-tridiagonal-in-L structure:
full Fokker-Planck and improved-Sugama collisions, ``Phi1``/quasineutrality,
tangential magnetic drifts, and the ``E_r`` ``xDot``/``xiDot`` terms.  Its
classical preconditioner (:func:`dkx.coarse_precond.build_coarse_preconditioner`)
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
parity tests pin that: the Krylov solution is the same to solver tolerance
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

The consequence, measured end to end and reported in ``docs/benchmarks/performance.md``:
on the full-Fokker-Planck ladder the multigrid route is affordable where the
classical preconditioner is not, but it does not reach the Krylov tolerance,
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
The cycle driver, transfers and smoothers come from ``solvax`` >= 0.9.1
(``solvax.precond.multigrid``, ``solvax.smoothers.upwind_smoother``,
``solvax.transfer``), which :mod:`dkx` requires outright, so this module
imports them unconditionally.
"""

from __future__ import annotations

from typing import Callable

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from solvax.precond import MultigridLevel  # noqa: E402
from solvax.precond import multigrid as solvax_multigrid
from solvax.smoothers import (  # noqa: E402
    smoothing_factor,
)

from dkx.drift_kinetic import KineticOperator  # noqa: E402

from .multigrid_diagnostics import (  # noqa: F401
    UPWIND_STENCILS,
    PitchCollocationSurrogate,
    _legendre_vandermonde,
    _stencil_weights,
    _uniform_spacing,
    dense_simplified_block,
    line_diagonal_dominance,
    line_smoother_spectral_radius,
    pitch_collocation_surrogate,
    stencil_matrices,
)
from .multigrid_levels import (  # noqa: F401
    _DIAGONAL_FLOOR,
    _STENCIL_BY_RADIUS,
    MultigridSettings,
    _angular_stencil,
    _coarse_diff_matrix,
    _collision_diagonal,
    _next_odd,
    _separable_transfer,
    coarsen_operator,
    hierarchy_shapes,
    periodic_transfer_matrices,
    simplified_operator,
    xi_transfer_matrices,
)
from .multigrid_smoothers import (  # noqa: F401
    _advection_bands,
    _band_scale,
    _coarse_solve,
    _legendre_plane_blocks,
    _legendre_plane_solve,
    _level_matvec,
    _level_smoother,
    _plane_angle,
    _shift,
    _streaming_eigenbasis,
)

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
    "periodic_transfer_matrices",
    "pitch_collocation_surrogate",
    "simplified_operator",
    "stencil_matrices",
    "xi_transfer_matrices",
]

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
        op: the full operator the recycled Krylov route solves.
        drop_l_coupling: drop the ``L +- 1`` streaming and mirror coupling
            (see :func:`simplified_operator`).  Not Fortran's
            ``preconditioner_xi``, which drops ``L +- 2``.
        settings: cycle/hierarchy/smoother knobs.

    Returns:
        ``(a_inv, shapes)`` where ``a_inv`` maps a flat ``(f_size,)`` residual
        to an approximate solution of the simplified f-block, and ``shapes``
        lists the ``(Ntheta, Nzeta, Nxi)`` of every level.
    """
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
    """Drop-in multigrid replacement for :func:`dkx.coarse_precond.build_coarse_preconditioner`.

    Same contract, same simplified operator, same exact elimination of the
    bordered constraint / ``Phi1`` rows -- only the inner approximate inverse
    of the f-block changes, from a batched block-Thomas factorization with
    dense ``(Ntheta*Nzeta)`` blocks to a multigrid V-cycle.

    The transposed preconditioner is obtained with :func:`jax.linear_transpose`
    of the cycle itself.  The cycle is a *linear* function of its argument
    (fixed transfers, fixed smoother factors, fixed coarse factorization), so
    its transpose is exact -- no second hierarchy, and no risk of the two
    drifting apart, which is what the Krylov adjoint guard would otherwise
    catch.

    Args:
        op: the full operator the recycled Krylov route solves.
        drop_l_coupling: drop the ``L +- 1`` streaming and mirror coupling
            (see :func:`simplified_operator`).  Not Fortran's
            ``preconditioner_xi``, which drops ``L +- 2``.
        settings: cycle/hierarchy/smoother knobs.

    Returns:
        ``(precond, precond_t)`` — approximate inverses of ``K`` and ``K^T``
        on flat ``(total_size,)`` vectors.
    """
    from dkx.coarse_precond import (  # noqa: PLC0415
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
        return (
            schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block),
            schur_projected_precond(a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T),
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
