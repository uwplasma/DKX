"""Collisionless (Shaing-Callen) limit of the monoenergetic bootstrap coefficient.

At asymptotically low collisionality the monoenergetic bootstrap-current
coefficient (the RHSMode=3 ``transportMatrix[1][0]`` entry) approaches a
collisionality-independent value determined purely by the flux-surface
geometry.  The primary result is K.C. Shaing and J.D. Callen, Phys. Fluids
26, 3315 (1983); the axisymmetric reduction is the tokamak banana-regime
value also obtained by A.H. Boozer and H.J. Gardner, Phys. Fluids B 2, 2408
(1990).  This module evaluates the dimensionless geometric factor in the
closed form of C.G. Albert, C.D. Beidler, G. Kapper, S.V. Kasilov, and
W. Kernbichler, arXiv:2407.21599 (2024), eq. (42),

    lambda_bB = < 2 B^2 Y + (3<B^2>/8) int_0^{eta_b} deta eta^2
                 (|lambda| / <|lambda|>) W_eta >,

where ``lambda = sqrt(1 - eta B)``, ``eta_b = 1/B_max``, ``Y`` and ``W_eta``
are the along-field-line integrals of the geodesic-curvature drive that
vanish at the global maximum of ``B``,

    (iota d/dtheta + d/dzeta) Y     = -(G dB/dtheta - I dB/dzeta) / B^3,
    (iota d/dtheta + d/dzeta) W_eta = -(G dB/dtheta - I dB/dzeta) / lambda^3,

and ``< . >`` is the flux-surface average.  Everything is expressed in
SFINCS "Hat" units with the flux label ``psiHat`` (so no ``dr/dpsi``
factors), and the magnetic differential equations are solved spectrally on a
Fourier-upsampled ``(theta, zeta)`` grid — only geometry arrays are needed.

For an axisymmetric field the expression collapses analytically to

    lambda_bB = (G / iota) f_t,

with ``f_t`` the trapped-particle fraction (the Boozer-Gardner tokamak
value), which :func:`trapped_fraction` provides as an independent
cross-check.

The conversion to the SFINCS RHSMode=3 ``transportMatrix[1][0]``
normalization follows from the monoenergetic coefficient definitions
(S.P. Hirshman, K.C. Shaing, W.I. van Rij, C.O. Beasley, and E.C. Crume,
Phys. Fluids 29, 2951 (1986); conventions as in C.D. Beidler et al., Nucl.
Fusion 51, 076001 (2011)): with ``Gamma_31 = <B int dxi xi f_1>`` the
flux-surface parallel-flow response of the radial-drive solution and
``lambda_bB = -(3/2) Gamma_31`` in the collisionless limit,

    transportMatrix[1][0] -> -(8 / (3 sqrt(pi))) (w_0 x_0^4 e^{-x_0^2} / GHat)
                             lambda_bB,

where ``(x_0, w_0)`` are the single monoenergetic speed node and weight.

Convergence toward this limit is slow and in general non-monotonic: in the
1/nu regime the offset from the limit oscillates with log(nu) and true
saturation requires orbit precession (Albert et al., arXiv:2407.21599).
Physics tests should therefore assert a monotone *approach* within a
measured envelope, not equality.

This is a diagnostics/physics-test evaluator: plain NumPy, not jitted (the
global-maximum search and Fourier upsampling are data dependent).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "ShaingCallenLimit",
    "shaing_callen_d31_limit",
    "shaing_callen_lambda_bb",
    "trapped_fraction",
]


class ShaingCallenLimit(NamedTuple):
    """Shaing-Callen limit pieces (all dimensionless, SFINCS Hat units).

    Attributes:
        d31: the collisionless limit in ``transportMatrix[1][0]`` units.
        lambda_bb: the geometric factor of Albert et al. eq. (42).
        term_passing: the ``<2 B^2 Y>`` (passing, Pfirsch-Schlueter-like) part.
        term_trapped: the pitch-integral (trapped-boundary) part.
        b_max: global maximum of ``BHat`` on the surface.
    """

    d31: float
    lambda_bb: float
    term_passing: float
    term_trapped: float
    b_max: float


def _fourier_upsample(b_hat: np.ndarray, n_up: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-padded Fourier coefficients of ``BHat`` and the fine-grid mode numbers."""
    b_hat = np.asarray(b_hat, dtype=np.float64)
    n_theta, n_zeta = b_hat.shape
    if n_up < max(n_theta, n_zeta):
        raise ValueError(f"upsample={n_up} must be >= the input grid ({n_theta}, {n_zeta}).")
    fb_small = np.fft.fft2(b_hat) / (n_theta * n_zeta)
    fb = np.zeros((n_up, n_up), dtype=complex)
    for i, m in enumerate(np.fft.fftfreq(n_theta, 1.0 / n_theta).astype(int)):
        for j, k in enumerate(np.fft.fftfreq(n_zeta, 1.0 / n_zeta).astype(int)):
            fb[m % n_up, k % n_up] = fb_small[i, j]
    m_up = np.fft.fftfreq(n_up, 1.0 / n_up)
    return fb, m_up, m_up.copy()


def _newton_refine_max(
    fb: np.ndarray, mm: np.ndarray, kk: np.ndarray, theta0: float, zeta0: float
) -> tuple[float, float, float]:
    """Refine the location of the global maximum of the Fourier series of B."""
    coeffs = fb.ravel()
    m_flat = mm.ravel()
    k_flat = kk.ravel()
    theta, zeta = float(theta0), float(zeta0)
    for _ in range(10):
        phase = np.exp(1j * (m_flat * theta + k_flat * zeta))
        g_t = float(np.real(phase @ (1j * m_flat * coeffs)))
        g_z = float(np.real(phase @ (1j * k_flat * coeffs)))
        h_tt = float(np.real(phase @ (-(m_flat**2) * coeffs)))
        h_tz = float(np.real(phase @ (-(m_flat * k_flat) * coeffs)))
        h_zz = float(np.real(phase @ (-(k_flat**2) * coeffs)))
        det = h_tt * h_zz - h_tz * h_tz
        if abs(det) < 1e-300:
            break
        d_theta = -(h_zz * g_t - h_tz * g_z) / det
        d_zeta = -(-h_tz * g_t + h_tt * g_z) / det
        theta += d_theta
        zeta += d_zeta
        if abs(d_theta) + abs(d_zeta) < 1e-14:
            break
    phase = np.exp(1j * (m_flat * theta + k_flat * zeta))
    return theta, zeta, float(np.real(phase @ coeffs))


def trapped_fraction(b_hat: np.ndarray, *, n_pitch: int = 400) -> float:
    """Flux-surface trapped-particle fraction ``f_t``.

    ``f_t = 1 - (3/4) <B^2> int_0^{1/B_max} eta deta / <sqrt(1 - eta B)>``
    with the ``1/B^2``-weighted flux-surface average of Boozer coordinates;
    the substitution ``eta = (1 - t^2)/B_max`` regularizes the trapped
    boundary.  Used as the analytic axisymmetric cross-check
    ``lambda_bB = (G/iota) f_t``.
    """
    b = np.asarray(b_hat, dtype=np.float64)
    weight = 1.0 / (b * b)
    weight_sum = float(np.sum(weight))

    def fsa(values: np.ndarray) -> float:
        return float(np.sum(values * weight)) / weight_sum

    b2_avg = fsa(b * b)
    eta_b = 1.0 / float(np.max(b))
    nodes, gauss_w = np.polynomial.legendre.leggauss(n_pitch)
    t = 0.5 * (nodes + 1.0)
    w_t = 0.5 * gauss_w
    passing = 0.0
    for t_i, w_i in zip(t, w_t):
        eta = eta_b * (1.0 - t_i * t_i)
        deta_dt = 2.0 * eta_b * t_i
        passing += w_i * deta_dt * eta / fsa(np.sqrt(np.maximum(1.0 - eta * b, 0.0)))
    return 1.0 - 0.75 * b2_avg * passing


def shaing_callen_lambda_bb(
    b_hat: np.ndarray,
    *,
    g_hat: float,
    i_hat: float,
    iota: float,
    n_periods: int,
    upsample: int = 384,
    n_eta: int = 128,
) -> ShaingCallenLimit:
    """Evaluate the geometric factor ``lambda_bB`` of Albert et al. eq. (42).

    Args:
        b_hat: ``BHat(theta, zeta)`` on the uniform grid covering one field
            period in zeta (band-limited; e.g. ``KineticOperator.b_hat``).
        g_hat / i_hat / iota: Boozer flux functions (Hat units).
        n_periods: number of field periods of the zeta grid.
        upsample: fine-grid size for the spectral magnetic-differential-equation
            solves (the pitch-boundary layer of ``1/lambda^3`` needs more
            resolution than ``BHat`` itself).
        n_eta: Gauss nodes of the regularized pitch integral.

    Returns:
        A :class:`ShaingCallenLimit` with ``d31`` left as ``nan`` (use
        :func:`shaing_callen_d31_limit` for the transport-matrix units).
    """
    fb, m_up, k_base = _fourier_upsample(b_hat, upsample)
    mm, kk = np.meshgrid(m_up, k_base * n_periods, indexing="ij")
    n_up = fb.shape[0]

    b = np.real(np.fft.ifft2(fb) * n_up * n_up)
    db_dtheta = np.real(np.fft.ifft2(1j * mm * fb) * n_up * n_up)
    db_dzeta = np.real(np.fft.ifft2(1j * kk * fb) * n_up * n_up)

    weight = 1.0 / (b * b)
    weight_sum = float(np.sum(weight))

    def fsa(values: np.ndarray) -> float:
        return float(np.sum(values * weight)) / weight_sum

    b2_avg = fsa(b * b)

    idx = np.unravel_index(int(np.argmax(b)), b.shape)
    theta_m, zeta_m, b_max = _newton_refine_max(
        fb, mm, kk, 2.0 * np.pi * idx[0] / n_up, 2.0 * np.pi * idx[1] / (n_up * n_periods)
    )
    eta_b = 1.0 / b_max

    # spectral solve of (iota d/dtheta + d/dzeta) w = src with w(theta_m, zeta_m) = 0;
    # the sources below are exact (theta, zeta) derivatives of functions of B, so
    # their resonant (m = k = 0) component vanishes identically.
    denom = 1j * (iota * mm + kk)
    inv = np.where(np.abs(denom) < 1e-12, 0.0, 1.0 / np.where(denom == 0, 1.0, denom))
    phase_m = np.exp(1j * (mm.ravel() * theta_m + kk.ravel() * zeta_m))

    def mde_solve(src: np.ndarray) -> np.ndarray:
        f_src = np.fft.fft2(src) / (n_up * n_up)
        f_w = f_src * inv
        w = np.real(np.fft.ifft2(f_w) * n_up * n_up)
        return w - float(np.real(phase_m @ f_w.ravel()))

    src_geom = -(g_hat * db_dtheta - i_hat * db_dzeta)
    y = mde_solve(src_geom / b**3)
    term_passing = fsa(2.0 * b * b * y)

    nodes, gauss_w = np.polynomial.legendre.leggauss(n_eta)
    t = 0.5 * (nodes + 1.0)
    w_t = 0.5 * gauss_w
    term_trapped = 0.0
    for t_i, w_i in zip(t, w_t):
        eta = eta_b * (1.0 - t_i * t_i)
        deta_dt = 2.0 * eta_b * t_i
        lam = np.sqrt(np.maximum(1.0 - eta * b, 0.0))
        w_eta = mde_solve(src_geom / np.maximum(lam, 1e-300) ** 3)
        term_trapped += w_i * deta_dt * eta * eta * fsa(lam * w_eta) / fsa(lam)
    term_trapped *= 3.0 * b2_avg / 8.0

    lam_bb = term_passing + term_trapped
    return ShaingCallenLimit(
        d31=float("nan"),
        lambda_bb=lam_bb,
        term_passing=term_passing,
        term_trapped=term_trapped,
        b_max=b_max,
    )


def shaing_callen_d31_limit(
    b_hat: np.ndarray,
    *,
    g_hat: float,
    i_hat: float,
    iota: float,
    n_periods: int,
    x: np.ndarray,
    x_weights: np.ndarray,
    upsample: int = 384,
    n_eta: int = 128,
) -> ShaingCallenLimit:
    """Shaing-Callen limit of the RHSMode=3 ``transportMatrix[1][0]`` entry.

    Args:
        b_hat / g_hat / i_hat / iota / n_periods: geometry as in
            :func:`shaing_callen_lambda_bb`.
        x / x_weights: the monoenergetic speed grid of the operator
            (``Nx = 1``): the node/weight pair fixes the speed-moment
            normalization of the transport matrix.

    Returns:
        A :class:`ShaingCallenLimit` with ``d31`` filled in.
    """
    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    if x.shape != (1,) or x_weights.shape != (1,):
        raise ValueError("shaing_callen_d31_limit expects the monoenergetic Nx=1 speed grid.")
    result = shaing_callen_lambda_bb(
        b_hat, g_hat=g_hat, i_hat=i_hat, iota=iota, n_periods=n_periods,
        upsample=upsample, n_eta=n_eta,
    )  # fmt: skip
    x0 = float(x[0])
    w0 = float(x_weights[0])
    speed_norm = w0 * x0**4 * np.exp(-(x0**2))
    d31 = -(8.0 / (3.0 * np.sqrt(np.pi))) * speed_norm * result.lambda_bb / float(g_hat)
    return result._replace(d31=d31)
