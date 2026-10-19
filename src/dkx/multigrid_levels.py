"""Level operators: settings, simplified operator, grid transfers and coarsening."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax.numpy as jnp  # noqa: E402

from .drift_kinetic import KineticOperator  # noqa: E402
from .phase_space import uniform_periodic_diff_matrices  # noqa: E402

# Relative floor added to every diagonal block so a collisionless, drift-free
# coarse f-block (whose diagonal is EXACTLY zero) still factors; mirrors
# ``dkx.coarse_precond.build_coarse_preconditioner``.
_DIAGONAL_FLOOR = 1e-8

@dataclass(frozen=True)
class MultigridSettings:
    """Knobs of the recycled Krylov multigrid preconditioner.

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
            :func:`dkx.coarse_precond.build_coarse_preconditioner` and is what a
            preconditioner may spend: measured on the NCSX ladder, raising it
            to ``0.01`` costs the Krylov solve 60 iterations instead of 21, and
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
    from dkx.coarse_precond import (  # noqa: PLC0415
        _collision_phi1_diagonal,
        _dense_collision_diagonal,
    )

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
    :func:`dkx.coarse_precond.build_coarse_preconditioner` factors -- the dense
    ``(species, x)``-coupled Fokker-Planck / improved-Sugama collision
    operators are reduced to their PAS-like self-species x-diagonal, the
    ``E_r`` ``L +- 2`` terms and the tangential magnetic drifts are dropped
    -- and, with ``drop_l_coupling``, the ``L +- 1`` streaming coupling goes
    too -- but expressed as a *matrix-free* operator rather than as dense
    bands, so it can be rediscretized on coarse grids and applied at
    ``O(f_size)`` cost.

    Dropping the ``L +- 2`` terms is what Fortran's ``preconditioner_xi=1``
    does (``populateMatrix.F90`` lines 625, 772, 872, 933 and 1046: "Drop the
    off-by-2 diagonal terms in L if this is the preconditioner"), and DKX does
    it unconditionally.  ``drop_l_coupling`` is a *different*, stronger knob --
    see its argument note.

    The reduced collision diagonal is folded into the ``pas`` slot: pitch-angle
    scattering *is* an arbitrary ``(S, X, L)`` diagonal coefficient, so the
    result is a plain PAS-family operator that both
    :meth:`KineticOperator.apply_f` and
    :meth:`KineticOperator.to_block_tridiagonal` accept.

    Args:
        op: the full operator the recycled Krylov route solves.
        drop_l_coupling: drop the ``L +- 1`` streaming and mirror coupling,
            leaving the simplified operator block-*diagonal* in ``L``.  This is
            not Fortran's ``preconditioner_xi``, which drops ``L +- 2``; it is
            strictly more aggressive, and it removes the dominant term.  It is
            off by default and measured expensive: on the Krylov truncation
            study it takes 6000 iterations to a residual of 0.77 where keeping
            the coupling converges in 19 (``docs/benchmarks/performance.md``).  Kept as a
            knob because a multigrid smoother at fixed ``L`` cannot see the
            coupling anyway.

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

