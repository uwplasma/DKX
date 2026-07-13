"""Differentiable bounce-averaged 1/nu low-collisionality radial transport.

This module implements a radially-local, bounce-averaged surrogate for the
dominant low-collisionality neoclassical radial-transport channel of a
stellarator -- the ``1/nu`` regime -- as a fast, fully differentiable JAX
objective whose high-fidelity referee is the full drift-kinetic solve of the
rest of the package.  The physics content is the effective ripple
``epsilon_eff`` and the underlying bounce integrals of the trapped-particle
radial magnetic drift.

Formulation
===========

On a single flux surface the ``1/nu`` radial transport is set by the
bounce-averaged radial magnetic drift of trapped particles.  Following the
bounce-averaged low-collisionality formulation of J. L. Velasco et al.,
J. Comput. Phys. 418, 109512 (2020) and Nucl. Fusion 61, 116059 (2021), the
effective ripple is the field-line functional

    epsilon_eff^{3/2} = (pi R_0^2) / (8 sqrt(2) <|grad psi|>^2) * Gamma_c,

    Gamma_c = < ( int dl/|B| )^{-1}  int_{B_min}^{B_max} d(rho)
                 sum_wells  H_j^2 / I_j >_field-line,

with, on each trapping well (the connected field-line interval where a particle
reflecting at field ``B_0 rho`` is trapped, ``lambda = 1/(B_0 rho)``),

    I_j   = int_well  sqrt(1 - |B|/(B_0 rho))                     dl/|B|,
    H_j   = rho^{-3/2} int_well sqrt(1 - |B|/(B_0 rho)) (4 B_0 rho/|B| - 1)
                 |grad psi| kappa_G                                dl/|B|.

Here ``dl/|B| = (G + iota I)/|B|^2 d(zeta)`` is the field-line-following measure
in Boozer coordinates and ``rho = B_reflect/B_0``.  The geometry factor
``|grad psi| kappa_G`` is the radial magnetic drive; in Boozer coordinates it
is exactly (independent of the flux-surface metric ``|grad psi|`` itself)

    |grad psi| kappa_G = -(G d|B|/d(theta) - I d|B|/d(zeta)) / (G + iota I),

the same geodesic-curvature drive that appears in the collisionless bootstrap
limit (:mod:`sfincs_jax.shaing_callen`).  ``R_0 = G/B_0`` and, for the analytic
Boozer schemes, ``<|grad psi|> = B_0 r_eff`` in the large-aspect-ratio circular
model (``psiHat = B_0 r^2/2``); for file geometries the flux-surface metric is
used when available.  ``Gamma_c`` is the ``|grad psi|``-free physics core; only
the outer ``<|grad psi|>^2`` normalization needs the metric.

Differentiable bounce points (the crux)
=======================================

The bounce points (turning points where ``v_parallel = 0``) are treated as
smooth functions of the geometry, not a non-differentiable root-find, following
the spectrally-accurate reverse-mode differentiable bounce-averaging algorithm
of arXiv:2412.01724.  ``|B|`` along a field line is evaluated spectrally
(exactly, from its Boozer spectrum), the connected trapping wells are located by
smooth threshold crossings (which correctly captures well merging over saddles),
each bounce point is refined by a differentiable Newton iteration on the
spectral ``|B|(zeta) = B_0 rho``, and every well integral is mapped to
``x in [-1, 1]`` by the sine substitution

    zeta = zeta_1 + (zeta_2 - zeta_1) (1 + sin(pi x/2))/2,

whose Jacobian ``(pi/4)(zeta_2 - zeta_1) cos(pi x/2)`` vanishes at the turning
points and cancels the square-root singularity, leaving a smooth integrand for
Gauss-Legendre quadrature.  ``jax.grad`` therefore flows through the whole
pipeline (spectrum -> bounce points -> bounce integrals).

Two public entry points:

* :func:`deep_well_bounce_integrals` returns the second adiabatic invariant and
  the bounce-averaged radial drift of the deepest trapping well at a fixed
  reflecting field.  These primitives are smooth and match central finite
  differences to ``~1e-10``.
* :func:`bounce_averaged_transport` assembles the full ``epsilon_eff`` /
  ``Gamma_c`` metric.  Its gradient flows (a valid descent direction for
  optimization); exact finite-difference agreement of the assembled metric is
  limited by the pitch integral's integrable jump discontinuities at trapped-
  well bifurcations (a known feature of bounce transport), improving only as
  ``1/sqrt(n_field_lines)``.  A bifurcation-adaptive pitch quadrature that would
  make the assembled metric finite-difference-exact is deferred.

Everything is pure JAX (``jit``/``grad``/``vmap`` safe) and uses only the
``|B|(theta, zeta)`` field plus the Boozer flux functions ``G``, ``I``,
``iota`` of a :class:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

__all__ = [
    "BounceAveragedTransport",
    "bounce_averaged_transport",
    "deep_well_bounce_integrals",
    "effective_ripple",
]


# =============================================================================
# Container
# =============================================================================


@dataclass(frozen=True)
class BounceAveragedTransport:
    """Bounce-averaged 1/nu transport metrics on one flux surface.

    Attributes:
        gamma_c: the ``|grad psi|``-free reduced 1/nu functional (the geometry
            core of the transport; ``0`` for an omnigeneous/quasisymmetric
            field).
        epsilon_eff_32: the effective ripple to the 3/2 power,
            ``(pi R_0^2/(8 sqrt(2) <|grad psi|>^2)) gamma_c``.
        epsilon_eff: the effective ripple ``epsilon_eff``.
        b_min: minimum ``|B|`` sampled on the surface.
        b_max: maximum ``|B|`` sampled on the surface.
        grad_psi_avg: the ``<|grad psi|>`` used in the normalization.
        r_major: ``R_0 = G/B_0`` used in the normalization.
    """

    gamma_c: jnp.ndarray
    epsilon_eff_32: jnp.ndarray
    epsilon_eff: jnp.ndarray
    b_min: jnp.ndarray
    b_max: jnp.ndarray
    grad_psi_avg: jnp.ndarray
    r_major: jnp.ndarray


# =============================================================================
# Spectral field-line evaluation
# =============================================================================


def _mode_indices(n_theta: int, n_zeta: int, m_keep: int, n_keep: int):
    """Static indices/mode numbers of the kept low ``(m, n)`` band of the FFT grid."""
    m = np.fft.fftfreq(n_theta, 1.0 / n_theta).astype(int)
    n = np.fft.fftfreq(n_zeta, 1.0 / n_zeta).astype(int)
    mm, nn = np.meshgrid(m, n, indexing="ij")
    mm = mm.ravel()
    nn = nn.ravel()
    keep = (np.abs(mm) <= m_keep) & (np.abs(nn) <= n_keep)
    return np.where(keep)[0], mm[keep].astype(float), nn[keep].astype(float)


def _line_spectrum(b_hat, n_periods, iota, alpha0, m_keep, n_keep):
    """Band-limited along-field-line spectrum of ``|B|`` at label ``alpha0``.

    Returns ``(c_j, m_j, nph_j, w_j)``: complex amplitude (carrying the
    ``exp(i m alpha0)`` field-line-label phase), poloidal mode number, physical
    toroidal mode number ``N_periods n``, and the along-line frequency
    ``w_j = m_j iota + nph_j`` (so ``|B|(zeta) = Re sum_j c_j exp(i w_j zeta)``
    along ``theta = alpha0 + iota zeta``).
    """
    n_theta, n_zeta = b_hat.shape
    idx, m_int, n_int = _mode_indices(n_theta, n_zeta, m_keep, n_keep)
    m_j = jnp.asarray(m_int, dtype=jnp.float64)
    nph_j = jnp.asarray(n_int, dtype=jnp.float64) * n_periods
    fb = (jnp.fft.fft2(b_hat) / (n_theta * n_zeta)).ravel()[idx]
    c_j = fb * jnp.exp(1j * m_j * alpha0)
    w_j = m_j * iota + nph_j
    return c_j, m_j, nph_j, w_j


def _eval_line(c_j, m_j, nph_j, w_j, z):
    """Spectral ``|B|`` and its ``theta``/``zeta`` partials along the line at ``z``."""
    e = jnp.exp(1j * z[..., None] * w_j)
    b = jnp.real(e @ c_j)
    db_dtheta = jnp.real(e @ (1j * m_j * c_j))
    db_dzeta = jnp.real(e @ (1j * nph_j * c_j))
    return b, db_dtheta, db_dzeta


def _geometry_fields(geometry):
    """Extract ``(b_hat, g_hat, i_hat, iota, n_periods, b0)`` from a geometry."""
    b_hat = jnp.asarray(geometry.b_hat, dtype=jnp.float64)
    g_hat = jnp.asarray(geometry.g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(geometry.i_hat, dtype=jnp.float64)
    iota = jnp.asarray(geometry.iota, dtype=jnp.float64)
    b0 = jnp.asarray(geometry.b0_over_bbar, dtype=jnp.float64)
    return b_hat, g_hat, i_hat, iota, int(geometry.n_periods), b0


def _default_bandwidth(n_theta: int, n_zeta: int, m_keep, n_keep):
    if m_keep is None:
        m_keep = min(n_theta // 2, 12)
    if n_keep is None:
        n_keep = min(n_zeta // 2, 8)
    return int(m_keep), int(n_keep)


# =============================================================================
# Differentiable bounce primitive: the deepest well
# =============================================================================


def deep_well_bounce_integrals(
    geometry: Any,
    reflect_fraction: float = 0.4,
    *,
    n_scan: int = 6000,
    n_periods_scan: int = 3,
    n_quad: int = 24,
    m_keep: int | None = None,
    n_keep: int | None = None,
):
    """Second adiabatic invariant and bounce-averaged radial drift, deepest well.

    A smooth, differentiable bounce-integral primitive: locates the global
    ``|B|`` minimum along the ``alpha = 0`` field line (refined by Newton on
    ``d|B|/dzeta = 0``), reflects at ``B_reflect = B_min + reflect_fraction
    (B_saddle - B_min)`` with ``B_saddle`` the flanking barrier, finds the two
    bounce points by Newton on the spectral ``|B|(zeta) = B_reflect``, and
    evaluates the sine-substitution bounce integrals.  Both outputs match
    central finite differences of a Boozer amplitude to ``~1e-10`` -- this is
    the differentiable bounce-averaging kernel of arXiv:2412.01724.

    Args:
        geometry: a :class:`~sfincs_jax.magnetic_geometry.FluxSurfaceGeometry`.
        reflect_fraction: trapping depth of the reflecting field between the
            well minimum (``0``) and the flanking barrier (``1``).
        n_scan, n_periods_scan: pre-scan resolution / field-period span used to
            bracket the deepest well.
        n_quad: Gauss-Legendre nodes of the sine-substitution quadrature.
        m_keep, n_keep: retained ``(m, n)`` Boozer bandwidth (defaults cover the
            analytic schemes exactly).

    Returns:
        ``(second_adiabatic_invariant, bounce_averaged_radial_drift)``: the
        well's ``int sqrt(1 - |B|/B_reflect) dl/|B|`` and the drift-weighted
        analogue ``int sqrt(1 - |B|/B_reflect) (|grad psi| kappa_G) dl/|B|``
        divided by the former (the bounce-averaged radial magnetic drift, hat
        units).
    """
    b_hat, g_hat, i_hat, iota, n_periods, b0 = _geometry_fields(geometry)
    m_keep, n_keep = _default_bandwidth(b_hat.shape[0], b_hat.shape[1], m_keep, n_keep)
    return _deep_well(
        b_hat, g_hat, i_hat, iota, n_periods, jnp.asarray(reflect_fraction, dtype=jnp.float64),
        int(n_scan), int(n_periods_scan), int(n_quad), m_keep, n_keep,
    )  # fmt: skip


@functools.partial(jax.jit, static_argnums=(4, 6, 7, 8, 9, 10))
def _deep_well(b_hat, g_hat, i_hat, iota, n_periods, reflect_fraction,
               n_scan, n_periods_scan, n_quad, m_keep, n_keep):  # fmt: skip
    c_j, m_j, nph_j, w_j = _line_spectrum(b_hat, n_periods, iota, 0.0, m_keep, n_keep)
    g_plus = g_hat + iota * i_hat

    def bfield(z):
        return _eval_line(c_j, m_j, nph_j, w_j, z)

    def d_along(z):  # d|B|/dl-direction and its zeta-derivative
        b, dt, dzt = bfield(z)
        dl = iota * dt + dzt
        e = jnp.exp(1j * z[..., None] * w_j)
        d2 = jnp.real(e @ (-(w_j**2) * c_j))
        return b, dl, d2

    span = 2.0 * jnp.pi / n_periods * n_periods_scan
    zc = jnp.linspace(0.0, span, n_scan, endpoint=False)
    bc, _, _ = bfield(zc)
    z0 = zc[jnp.argmin(bc)]
    for _ in range(5):  # refine the well minimum
        _, dl, d2 = d_along(jnp.atleast_1d(z0))
        z0 = z0 - (dl / jnp.where(jnp.abs(d2) > 1e-9, d2, 1e-9))[0]
    b_min = bfield(jnp.atleast_1d(z0))[0][0]
    # flanking barrier: the higher of the two neighbouring maxima along the line
    z_left_bar = z0 - jnp.pi / n_periods
    z_right_bar = z0 + jnp.pi / n_periods
    b_bar = jnp.minimum(bfield(jnp.atleast_1d(z_left_bar))[0][0],
                        bfield(jnp.atleast_1d(z_right_bar))[0][0])  # fmt: skip
    b_reflect = b_min + reflect_fraction * (b_bar - b_min)

    def bounce(sign):
        z = z0 + sign * 0.5 * (jnp.pi / n_periods)
        for _ in range(8):
            b, dt, dzt = bfield(jnp.atleast_1d(z))
            dl = iota * dt[0] + dzt[0]
            z = z - (b[0] - b_reflect) / jnp.where(jnp.abs(dl) > 1e-8, dl, sign * 1e-8)
        return z

    z1 = bounce(-1.0)
    z2 = bounce(1.0)

    xq, wq = np.polynomial.legendre.leggauss(int(n_quad))
    xq = jnp.asarray(xq, dtype=jnp.float64)
    wq = jnp.asarray(wq, dtype=jnp.float64)
    u = 0.5 * (1.0 + jnp.sin(0.5 * jnp.pi * xq))
    dudx = 0.25 * jnp.pi * jnp.cos(0.5 * jnp.pi * xq)
    zq = z1 + (z2 - z1) * u
    bq, dtq, dzq = bfield(zq)
    root = jnp.sqrt(jnp.clip(1.0 - bq / b_reflect, 0.0, None))
    meas = g_plus / (bq * bq)
    jac = (z2 - z1) * dudx
    d_psi = -(g_hat * dtq - i_hat * dzq) / g_plus
    j_inv = jnp.sum(wq * jac * root * meas)
    drift_int = jnp.sum(wq * jac * root * d_psi * meas)
    return j_inv, drift_int / j_inv


# =============================================================================
# The 1/nu effective-ripple functional
# =============================================================================


@functools.partial(jax.jit, static_argnums=(4, 7, 8, 9, 10, 11, 12, 13))
def _gamma_c_line(b_hat, g_hat, i_hat, iota, n_periods, b0, alpha0,
                  m_keep, n_keep, npi, ppp, n_rho, n_q, w_max):  # fmt: skip
    """Reduced 1/nu functional ``Gamma_c`` on one ergodic field line at ``alpha0``."""
    c_j, m_j, nph_j, w_j = _line_spectrum(b_hat, n_periods, iota, alpha0, m_keep, n_keep)
    g_plus = g_hat + iota * i_hat

    def bfield(z):
        return _eval_line(c_j, m_j, nph_j, w_j, z)

    k = ppp * npi
    zmax = 2.0 * jnp.pi / n_periods * npi
    zg = jnp.linspace(0.0, zmax, k, endpoint=False)
    dz = zg[1] - zg[0]
    bg, dtg, dzg = bfield(zg)
    # roll so index 0 is the global maximum -> the line starts untrapped, so
    # enter/exit crossings alternate and pair cleanly.
    shift = jnp.argmax(bg)
    ridx = (jnp.arange(k) + shift) % k
    bg = bg[ridx]
    zg = zg[ridx]
    meas_g = g_plus / (bg * bg)
    l_norm = jnp.sum(meas_g) * dz  # int dl/B (permutation-invariant)

    b_min = jnp.min(bg)
    b_max = jnp.max(bg)
    rho = jnp.linspace(b_min / b0, b_max / b0, n_rho + 2)[1:-1]

    xq, wq = np.polynomial.legendre.leggauss(int(n_q))
    xq = jnp.asarray(xq, dtype=jnp.float64)
    wq = jnp.asarray(wq, dtype=jnp.float64)
    u = 0.5 * (1.0 + jnp.sin(0.5 * jnp.pi * xq))
    dudx = 0.25 * jnp.pi * jnp.cos(0.5 * jnp.pi * xq)
    zprev = jnp.concatenate([zg[:1], zg[:-1]])

    def per_rho(r):
        xref = b0 * r
        s = xref - bg  # > 0 trapped
        tr = s > 0.0
        prev = jnp.concatenate([jnp.array([False]), tr[:-1]])
        nextt = jnp.concatenate([tr[1:], jnp.array([False])])
        enter = tr & (~prev)
        exit_ = tr & (~nextt)
        # linear guesses AT crossings only (double-where keeps the reverse pass
        # finite where the finite-difference denominator would vanish).
        s_prev = jnp.concatenate([s[:1], s[:-1]])
        s_next = jnp.concatenate([s[1:], s[:1]])
        den_e = jnp.where(enter, s - s_prev, 1.0)
        frac_e = jnp.where(enter, jnp.clip(-s_prev / den_e, 0.0, 1.0), 0.0)
        z_enter0 = zprev + dz * frac_e
        den_x = jnp.where(exit_, s_next - s, 1.0)
        frac_x = jnp.where(exit_, jnp.clip(-s / den_x, 0.0, 1.0), 0.0)
        z_exit0 = zg + dz * frac_x

        e_slot = jnp.where(enter, jnp.cumsum(enter) - 1, w_max)
        x_slot = jnp.where(exit_, jnp.cumsum(exit_) - 1, w_max)
        ent0 = jax.ops.segment_sum(jnp.where(enter, z_enter0, 0.0), e_slot, num_segments=w_max + 1)[:w_max]
        ext0 = jax.ops.segment_sum(jnp.where(exit_, z_exit0, 0.0), x_slot, num_segments=w_max + 1)[:w_max]
        n_pair = jnp.minimum(jnp.sum(enter), jnp.sum(exit_))
        vpre = jnp.arange(w_max) < n_pair

        def newton(z0):
            z = z0
            for _ in range(3):
                b, dt, dzt = bfield(z)
                dl = iota * dt + dzt
                dl_safe = jnp.where(jnp.abs(dl) > 1e-3, dl, 1e-3)
                step = jnp.clip((b - xref) / dl_safe, -0.5, 0.5)
                z = z - jnp.where(vpre, step, 0.0)
            return z

        ent = newton(ent0)
        ext = newton(ext0)
        valid = vpre & (ext > ent)

        z1 = ent[:, None]
        z2 = ext[:, None]
        width = jnp.where(valid[:, None], z2 - z1, 0.0)
        zq = z1 + width * u[None, :]
        bq, dtq, dzq = bfield(zq)
        arg = jnp.where(valid[:, None], 1.0 - bq / xref, 1.0)
        safe = jnp.where(arg > 0.0, arg, 1.0)
        rootq = jnp.where(arg > 0.0, jnp.sqrt(safe), 0.0)
        measq = g_plus / (bq * bq)
        d_psi = -(g_hat * dtq - i_hat * dzq) / g_plus
        jac = width * dudx[None, :]
        wgt = wq[None, :] * jac
        i_w = jnp.sum(wgt * rootq * measq, axis=1)
        h_w = jnp.sum(wgt * rootq * (4.0 * xref / bq - 1.0) * d_psi * measq, axis=1) * r ** (-1.5)
        good = valid & (i_w > 1e-300)
        ratio = h_w * h_w / jnp.where(good, i_w, 1.0)
        return jnp.sum(jnp.where(good, ratio, 0.0))

    integ = jax.vmap(per_rho)(rho)
    return jnp.trapezoid(integ, rho) / l_norm, b_min, b_max


@functools.partial(jax.jit, static_argnums=(4, 6, 7, 8, 9, 10, 11, 12, 13))
def _gamma_c(b_hat, g_hat, i_hat, iota, n_periods, b0,
            m_keep, n_keep, npi, ppp, n_rho, n_q, w_max, n_lines):  # fmt: skip
    """Field-line averaged ``Gamma_c`` (mean over ``n_lines`` ergodic lines)."""
    alphas = jnp.linspace(0.0, 2.0 * jnp.pi, n_lines, endpoint=False)

    def one(a):
        gc, bmn, bmx = _gamma_c_line(
            b_hat, g_hat, i_hat, iota, n_periods, b0, a,
            m_keep, n_keep, npi, ppp, n_rho, n_q, w_max,
        )  # fmt: skip
        return gc, bmn, bmx

    gcs, bmins, bmaxs = jax.vmap(one)(alphas)
    return jnp.mean(gcs), jnp.min(bmins), jnp.max(bmaxs)


def bounce_averaged_transport(
    geometry: Any,
    *,
    r_eff: float | None = None,
    grad_psi_avg: float | None = None,
    n_field_periods: int = 160,
    points_per_period: int = 48,
    n_pitch: int = 128,
    n_quad: int = 14,
    max_wells: int = 224,
    n_field_lines: int = 1,
    m_keep: int | None = None,
    n_keep: int | None = None,
) -> BounceAveragedTransport:
    """Bounce-averaged ``1/nu`` effective-ripple transport for one flux surface.

    Pure JAX and differentiable in the ``|B|`` Boozer spectrum of ``geometry``
    (use :meth:`FluxSurfaceGeometry.from_fourier` for a traceable spectrum);
    ``jit``/``vmap`` safe.  The reduced functional ``Gamma_c`` and hence
    ``epsilon_eff`` are the ``nu -> 0`` asymptote of the full-DKE monoenergetic
    radial coefficient ``D11``.

    Args:
        geometry: a :class:`~sfincs_jax.magnetic_geometry.FluxSurfaceGeometry`.
        r_eff: effective minor radius ``r`` of the surface; the normalization
            uses the large-aspect-ratio circular value ``<|grad psi|> = B_0
            r_eff`` when the flux-surface metric is unavailable (analytic
            schemes).  Overridden by ``grad_psi_avg`` or the geometry metric.
        grad_psi_avg: explicit ``<|grad psi|>`` for the ``epsilon_eff``
            normalization (takes precedence).
        n_field_periods, points_per_period: ergodic field-line length / pre-scan
            resolution (more periods -> better flux-surface coverage).
        n_pitch: pitch (reflecting-field) quadrature nodes.
        n_quad: sine-substitution Gauss-Legendre nodes per well.
        max_wells: static upper bound on trapping wells per field line (must
            exceed the number of ripple wells over ``n_field_periods``).
        n_field_lines: number of field lines averaged; ``> 1`` reduces the
            pitch-bifurcation staircase in the gradient as ``1/sqrt``.
        m_keep, n_keep: retained Boozer ``(m, n)`` bandwidth.

    Returns:
        A :class:`BounceAveragedTransport`.  ``epsilon_eff`` is ``nan`` if no
        ``|grad psi|`` normalization can be resolved (then use ``gamma_c``).
    """
    b_hat, g_hat, i_hat, iota, n_periods, b0 = _geometry_fields(geometry)
    m_keep, n_keep = _default_bandwidth(b_hat.shape[0], b_hat.shape[1], m_keep, n_keep)

    gamma_c, b_min, b_max = _gamma_c(
        b_hat, g_hat, i_hat, iota, n_periods, b0,
        m_keep, n_keep, int(n_field_periods), int(points_per_period),
        int(n_pitch), int(n_quad), int(max_wells), int(n_field_lines),
    )  # fmt: skip

    r_major = g_hat / b0
    gpsi = _resolve_grad_psi_avg(geometry, grad_psi_avg, r_eff, b0)
    if gpsi is None:
        eps32 = jnp.asarray(jnp.nan)
        eps = jnp.asarray(jnp.nan)
        gpsi_out = jnp.asarray(jnp.nan)
    else:
        gpsi_out = jnp.asarray(gpsi, dtype=jnp.float64)
        eps32 = (jnp.pi * r_major * r_major) / (8.0 * jnp.sqrt(2.0) * gpsi_out * gpsi_out) * gamma_c
        eps = jnp.clip(eps32, 0.0, None) ** (2.0 / 3.0)

    return BounceAveragedTransport(
        gamma_c=gamma_c,
        epsilon_eff_32=eps32,
        epsilon_eff=eps,
        b_min=b_min,
        b_max=b_max,
        grad_psi_avg=gpsi_out,
        r_major=r_major,
    )


def effective_ripple(geometry: Any, **kwargs) -> jnp.ndarray:
    """Convenience wrapper returning only ``epsilon_eff`` (see
    :func:`bounce_averaged_transport`)."""
    return bounce_averaged_transport(geometry, **kwargs).epsilon_eff


def _resolve_grad_psi_avg(geometry, grad_psi_avg, r_eff, b0):
    """``<|grad psi|>`` from an explicit value, the geometry metric, or LAR.

    Returns a (possibly traced) scalar or ``None``; never forces a host value,
    so it stays differentiable through ``from_fourier`` geometries.
    """
    if grad_psi_avg is not None:
        return jnp.asarray(grad_psi_avg, dtype=jnp.float64)
    gpsipsi = getattr(geometry, "gpsipsi", None)
    if gpsipsi is not None:
        return jnp.mean(jnp.sqrt(jnp.asarray(gpsipsi, dtype=jnp.float64)))
    if r_eff is not None:
        return jnp.asarray(b0, dtype=jnp.float64) * jnp.asarray(r_eff, dtype=jnp.float64)
    return None
