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
The operator's two dominant terms live in *different bases*: parallel streaming
is diagonal in pitch collocation and ``L +- 1`` in Legendre, collisions are
diagonal in ``L`` and dense in pitch collocation (keeping only the collocation
diagonal discards 65% of the reduced collision operator in the 2-norm).  The
smoother therefore stays in the Legendre basis and relaxes the ``(L, angle)``
planes *exactly* (:func:`_legendre_plane_blocks`): collisions, mirror force,
streaming and ExB along one angle are all kept, the blocks are ``N x N`` with
``N`` one angular length, and only the other angle's derivative is left to the
coarse grid.  Pitch-collocation line, plane and alternating sweeps were
measured (smoothing factors ``mu`` of 4 to 7 against ``0.87`` for the plane
sweep across ``zeta`` on NCSX ``11 x 21 x 41 x 5``) and removed with their
diagnostics; they are in the git history.

No cheap sweep is complementary to angular coarsening in *both* angles at once,
and the near-null directions -- distributions nearly constant along the field
line -- are represented by no coarse grid this hierarchy can build.  Measured
end to end (``docs/benchmarks/performance.md``), on the full-Fokker-Planck
ladder the multigrid route is affordable where the classical preconditioner is
not, but it does not reach the Krylov tolerance, while the exact block-Thomas
of the same simplified operator does so in 21 iterations.  The route is
therefore opt-in (``solve(preconditioner="multigrid")``).

The cause is structural: parallel streaming and the mirror force are strictly
off-diagonal in the Legendre index (they couple ``L -> L +- 1`` and contribute
nothing to the ``(L, L)`` block), so no line relaxation has diagonal weight from
the dominant term.  A pitch-*grid* discretization would be diagonally dominant,
but it changes answers at fixed resolution and breaks Fortran matrix parity;
that experiment (two-grid factor 0.24 against 4.0e13 for the modal basis) was
removed from the tree and is in the git history.

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
- M. Landreman, "A multigrid method for drift-kinetic calculations in
  stellarators and rippled tokamaks", Bull. Am. Phys. Soc. **62**, JP11.128
  (2017) -- multigrid smoothers are unstable for accurate discretizations of
  this equation.

Availability
------------
The cycle driver, transfers and smoothers come from ``solvax`` >= 0.9.1
(``solvax.precond.multigrid``, ``solvax.smoothers.relaxation``,
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

from dkx.drift_kinetic import KineticOperator  # noqa: E402

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
    _band_scale,
    _coarse_solve,
    _legendre_plane_blocks,
    _legendre_plane_solve,
    _level_matvec,
    _level_smoother,
    _plane_angle,
    _shift,
)

__all__ = [
    "MultigridSettings",
    "build_multigrid_f_inverse",
    "coarsen_operator",
    "hierarchy_shapes",
    "periodic_transfer_matrices",
    "simplified_operator",
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

    The transpose is obtained with :func:`jax.vjp` at a zero input. Fixed
    transfers, smoother factors and the coarse factorization make the cycle
    linear, so the pullback is its exact transpose without a second hierarchy.
    Linearization keeps the cyclic solves' constant correction channels
    separate from the varying residual, including when their intermediate
    right-hand sides are packed together.

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
    _, transposed = jax.vjp(a_inv, template)

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
