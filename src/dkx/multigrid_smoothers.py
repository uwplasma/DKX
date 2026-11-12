"""Level matvec, coarse solve and the Legendre-plane smoother."""

from __future__ import annotations

from typing import Callable

import numpy as np

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from solvax.direct import block_thomas_factor, block_thomas_solve  # noqa: E402
from solvax.smoothers import relaxation  # noqa: E402

from .drift_kinetic import KineticOperator  # noqa: E402
from .multigrid_levels import MultigridSettings, _angular_stencil  # noqa: E402

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

    :func:`dkx.coarse_precond.build_coarse_preconditioner` removes it with a ``1e-8``
    relative diagonal floor plus an *adaptive* rank-one pin of that constant on
    the ``l = 0`` block (:func:`dkx.coarse_precond._l0_pin_gamma`), sized to the same
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
      block-Thomas of this simplified operator as the Krylov preconditioner on
      the NCSX ``11 x 21 x 41 x 5`` deck, GCROT needs **21** iterations at
      ``delta = 0``, 60 at ``1e-2`` of the mean collision diagonal, and never
      converges at ``1.0``.  (The *unconditional* full-strength rank-one pin
      that :func:`dkx.coarse_precond.build_coarse_preconditioner` used to apply cost 87
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

    The classical Krylov preconditioner, kept where it belongs: its
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

    Relaxes the ``(L, angle)`` planes exactly with :func:`_legendre_plane_solve`,
    wrapped by :func:`solvax.smoothers.relaxation` so the sweep refreshes its
    residual against the true level operator.  Everything except the *other*
    angle's derivative is inverted exactly, which is the plane relaxation
    Trottenberg et al. (section 5.2) prescribe when an operator is strongly
    coupled along two directions at once.
    """
    mask_only = level._mask()[None, :, :, None, None]
    uniform_mask = bool(np.all(np.asarray(level._mask()) == 1.0))
    core = _legendre_plane_solve(level, _plane_angle(level), weight, settings.plane_pin)
    if not uniform_mask:
        plane = core

        def core(r: jnp.ndarray) -> jnp.ndarray:
            """Identity on the ``Nxi_for_x``-truncated (identity-row) DOFs."""
            return plane(r * mask_only) * mask_only + r * (1.0 - mask_only)

    return relaxation(core, omega=settings.omega_xi)
