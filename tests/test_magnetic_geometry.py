"""Tests for ``dkx.magnetic_geometry``.

The analytic schemes are pinned against independent evaluations of the
``geometry.F90`` formulas; the VMEC/Boozer file paths are pinned against the
recorded Fortran output goldens (``tests/test_output_h5_scheme*_parity.py``
and ``tests/test_geometry_scheme11_parity.py``).  Includes a
differentiability gate for the ``from_fourier`` (geometryScheme 13)
constructor used by optimization.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from dkx.magnetic_geometry import FluxSurfaceGeometry, read_vmec_wout, _u_and_bsubpsi

_NONSTELSYM_WOUT = Path(__file__).parent / "ref" / "wout_up_down_asymmetric_tokamak.nc"

# Stellarator-asymmetric (lasym=T) complementary-parity tables read by
# ``read_vmec_wout``; ``None`` for stellarator-symmetric wout files.
_COMPLEMENTARY_TABLES = (
    "bmns", "gmns", "bsubumns", "bsubvmns", "bsubsmnc",
    "bsupumns", "bsupvmns", "rmns", "zmnc", "lmnc",
)  # fmt: skip

# Tiny grid matrix (24 GB laptop budget): includes an axisymmetric Nzeta=1
# column and odd/even mixes that exercise the Nyquist bookkeeping.
_GRIDS = [(7, 5), (9, 8), (8, 1)]


def _grid(n_theta: int, n_zeta: int, n_periods: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = jnp.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False, dtype=jnp.float64)
    if n_zeta == 1:
        zeta = jnp.asarray([0.0], dtype=jnp.float64)
    else:
        zeta = jnp.linspace(0.0, 2.0 * math.pi / n_periods, n_zeta, endpoint=False, dtype=jnp.float64)
    return theta, zeta


# ---------------------------------------------------------------------------
# Analytic schemes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_theta", "n_zeta"), _GRIDS)
@pytest.mark.parametrize(
    "params",
    [
        dict(),  # namelist defaults
        dict(epsilon_antisymm=0.01, i_hat=0.1, helicity_antisymm_l=1, helicity_antisymm_n=10),
        dict(helicity_n=0, helicity_antisymm_n=3, epsilon_antisymm=0.02),  # tokamak-like NPeriods=1
    ],
)
def test_scheme1_matches_analytic_three_helicity_model(n_theta: int, n_zeta: int, params: dict) -> None:
    defaults = dict(
        epsilon_t=-0.07053,
        epsilon_h=0.05067,
        epsilon_antisymm=0.0,
        iota=0.4542,
        g_hat=3.7481,
        i_hat=0.0,
        b0_over_bbar=1.0,
        helicity_l=2,
        helicity_n=10,
        helicity_antisymm_l=1,
        helicity_antisymm_n=0,
    )
    defaults.update(params)
    n_periods = max(1, int(defaults["helicity_n"]))
    theta, zeta = _grid(n_theta, n_zeta, n_periods)
    geom = FluxSurfaceGeometry.from_scheme(1, theta=theta, zeta=zeta, **defaults)

    # Independent evaluation of the geometry.F90 three-helicity model:
    #   BHat = B0 (1 + eps_t cos(theta) + eps_h cos(l theta - n zeta)
    #                + eps_a sin(la theta - na zeta)),
    # with the helical n expressed in field-period units (NPeriods = helicity_n).
    th = np.asarray(theta)[:, None]
    ze = np.asarray(zeta)[None, :]
    b0 = float(defaults["b0_over_bbar"])
    hel_n = int(defaults["helicity_n"])
    n2 = 0 if hel_n == 0 else 1
    n3 = int(defaults["helicity_antisymm_n"]) if hel_n == 0 else int(defaults["helicity_antisymm_n"]) // hel_n
    expected = b0 * (
        1.0
        + float(defaults["epsilon_t"]) * np.cos(th - 0.0 * ze)
        + float(defaults["epsilon_h"]) * np.cos(int(defaults["helicity_l"]) * th - n_periods * n2 * ze)
        + float(defaults["epsilon_antisymm"])
        * np.sin(int(defaults["helicity_antisymm_l"]) * th - n_periods * n3 * ze)
    )
    np.testing.assert_allclose(np.asarray(geom.b_hat), expected, rtol=1.0e-14, atol=1.0e-14)
    assert geom.n_periods == n_periods
    assert float(geom.iota) == pytest.approx(float(defaults["iota"]), abs=0.0)
    # DHat = BHat^2 / (GHat + iota IHat) for Boozer coordinates.
    denom = float(defaults["g_hat"]) + float(defaults["iota"]) * float(defaults["i_hat"])
    np.testing.assert_allclose(
        np.asarray(geom.d_hat), expected**2 / denom, rtol=1.0e-13, atol=1.0e-14
    )
    # Analytic angular derivatives of the three-helicity model.
    l_h = int(defaults["helicity_l"])
    l_a = int(defaults["helicity_antisymm_l"])
    d_expected_dtheta = b0 * (
        -float(defaults["epsilon_t"]) * np.sin(th - 0.0 * ze)
        - float(defaults["epsilon_h"]) * l_h * np.sin(l_h * th - n_periods * n2 * ze)
        + float(defaults["epsilon_antisymm"]) * l_a * np.cos(l_a * th - n_periods * n3 * ze)
    )
    d_expected_dzeta = b0 * (
        float(defaults["epsilon_h"]) * (n_periods * n2) * np.sin(l_h * th - n_periods * n2 * ze)
        - float(defaults["epsilon_antisymm"]) * (n_periods * n3) * np.cos(l_a * th - n_periods * n3 * ze)
    )
    np.testing.assert_allclose(np.asarray(geom.db_hat_dtheta), d_expected_dtheta, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dzeta), d_expected_dzeta, rtol=1.0e-13, atol=1.0e-13)


def test_scheme3_matches_fortran_table() -> None:
    """Scheme 3 (LHD inward-shifted; new in v2) against an independent evaluation."""
    theta, zeta = _grid(9, 8, 10)
    geom = FluxSurfaceGeometry.from_scheme(3, theta=theta, zeta=zeta)
    assert geom.n_periods == 10
    assert geom.iota == pytest.approx(0.4692, abs=0.0)
    assert geom.g_hat == pytest.approx(1.0 * 3.6024, abs=0.0)
    assert geom.i_hat == 0.0
    assert geom.b0_over_bbar == 1.0

    th = np.asarray(theta)[:, None]
    ze = np.asarray(zeta)[None, :]
    expected = 1.0 + sum(
        amp * np.cos(m * th - 10.0 * n * ze)
        for (m, n, amp) in [(1, 0, -0.05927), (2, 1, 0.05267), (1, 1, -0.04956), (0, 1, 0.01045)]
    )
    np.testing.assert_allclose(np.asarray(geom.b_hat), expected, rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(
        np.asarray(geom.d_hat), expected**2 / 3.6024, rtol=1.0e-14, atol=1.0e-15
    )


def test_flux_surface_average_methods() -> None:
    """VPrimeHat / FSABHat2 methods match the computeBIntegrals definitions."""
    theta, zeta = _grid(9, 7, 5)
    tw = jnp.full((9,), 2.0 * math.pi / 9.0)
    zw = jnp.full((7,), (2.0 * math.pi / 5.0 / 7.0) * 5.0)
    geom = FluxSurfaceGeometry.from_scheme(4, theta=theta, zeta=zeta)
    w = np.asarray(tw)[:, None] * np.asarray(zw)[None, :]
    d_hat = np.asarray(geom.d_hat)
    b_hat = np.asarray(geom.b_hat)
    vprime = np.sum(w / d_hat)
    fsab2 = np.sum(w * b_hat**2 / d_hat) / vprime
    np.testing.assert_allclose(float(geom.vprime_hat(theta_weights=tw, zeta_weights=zw)), vprime, rtol=1e-15)
    np.testing.assert_allclose(float(geom.fsab_hat2(theta_weights=tw, zeta_weights=zw)), fsab2, rtol=1e-15)


# ---------------------------------------------------------------------------
# VMEC wout path (geometryScheme 5) — end-to-end Fortran parity lives in
# tests/test_output_h5_scheme5_parity.py and the nonstelsym scheme-5 golden.
# ---------------------------------------------------------------------------


def test_read_vmec_wout_loads_complementary_tables_when_lasym() -> None:
    """A lasym=T wout exposes every complementary-parity table with matching shapes."""
    assert _NONSTELSYM_WOUT.exists(), f"Missing lasym wout fixture: {_NONSTELSYM_WOUT}"
    w = read_vmec_wout(_NONSTELSYM_WOUT)
    assert w.lasym is True
    for name in _COMPLEMENTARY_TABLES:
        arr = getattr(w, name)
        assert arr is not None, f"complementary table {name} not loaded"
        assert arr.ndim == 2, name
    # Nyquist tables share the bmnc layout; shape tables share the rmnc layout.
    assert w.bmns.shape == w.bmnc.shape
    assert w.gmns.shape == w.gmnc.shape
    assert w.bsubsmnc.shape == w.bsubsmns.shape
    assert w.rmns.shape == w.rmnc.shape
    assert w.zmnc.shape == w.zmns.shape


def test_read_vmec_wout_stellarator_symmetric_returns_none(tmp_path: Path) -> None:
    """A stellarator-symmetric wout leaves the complementary tables ``None`` (path unchanged)."""
    from scipy.io import netcdf_file

    path = tmp_path / "wout_synthetic_symmetric.nc"
    ns, mnmax, mnmax_nyq = 5, 2, 3
    with netcdf_file(path, "w") as f:
        f.createDimension("radius", ns)
        f.createDimension("mnmax", mnmax)
        f.createDimension("mnmax_nyq", mnmax_nyq)
        f.createVariable("nfp", "i", ())[...] = 5
        f.createVariable("ns", "i", ())[...] = ns
        f.createVariable("mpol", "i", ())[...] = 2
        f.createVariable("ntor", "i", ())[...] = 1
        f.createVariable("mnmax", "i", ())[...] = mnmax
        f.createVariable("mnmax_nyq", "i", ())[...] = mnmax_nyq
        f.createVariable("lasym__logical__", "i", ())[...] = 0
        f.createVariable("Aminor_p", "d", ())[...] = 0.5
        f.createVariable("phi", "d", ("radius",))[...] = np.linspace(0.0, 2.0 * np.pi, ns)
        f.createVariable("xm", "i", ("mnmax",))[...] = np.asarray([0, 1], dtype=np.int32)
        f.createVariable("xn", "i", ("mnmax",))[...] = np.asarray([0, 5], dtype=np.int32)
        f.createVariable("xm_nyq", "i", ("mnmax_nyq",))[...] = np.asarray([0, 1, 2], dtype=np.int32)
        f.createVariable("xn_nyq", "i", ("mnmax_nyq",))[...] = np.asarray([0, 5, 10], dtype=np.int32)
        nyq = np.arange(ns * mnmax_nyq, dtype=np.float64).reshape(ns, mnmax_nyq) + 1.0
        full = np.arange(ns * mnmax, dtype=np.float64).reshape(ns, mnmax) + 1.0
        for name in ("bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc"):
            f.createVariable(name, "d", ("radius", "mnmax_nyq"))[...] = nyq
        f.createVariable("rmnc", "d", ("radius", "mnmax"))[...] = full
        f.createVariable("zmns", "d", ("radius", "mnmax"))[...] = full + 100.0
        f.createVariable("lmns", "d", ("radius", "mnmax"))[...] = full
        f.createVariable("iotas", "d", ("radius",))[...] = np.linspace(0.4, 0.6, ns)
        f.createVariable("presf", "d", ("radius",))[...] = np.linspace(1.0, 0.0, ns)

    w = read_vmec_wout(path)
    assert w.lasym is False
    for name in _COMPLEMENTARY_TABLES:
        assert getattr(w, name) is None, f"{name} should be None for a stell-sym wout"
    # The symmetric-parity tables are still populated.
    assert w.bmnc is not None
    assert w.rmnc is not None


# ---------------------------------------------------------------------------
# Differentiable Fourier path (geometryScheme 13 / from_fourier)
# ---------------------------------------------------------------------------


def test_from_fourier_reproduces_scheme4_table() -> None:
    theta, zeta = _grid(11, 9, 5)
    m = jnp.asarray([0, 0, 1, 1])
    n = jnp.asarray([0, 1, 1, 0])
    bmnc = jnp.asarray([3.089, 0.04645 * 3.089, -0.04351 * 3.089, -0.01902 * 3.089])
    g13 = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=bmnc, m=m, n=n, n_periods=5, iota=0.87, g_hat=-17.885, i_hat=0.0
    )
    g4 = FluxSurfaceGeometry.from_scheme(4, theta=theta, zeta=zeta)
    np.testing.assert_allclose(np.asarray(g13.b_hat), np.asarray(g4.b_hat), rtol=1.0e-14)
    np.testing.assert_allclose(np.asarray(g13.d_hat), np.asarray(g4.d_hat), rtol=1.0e-14)
    assert float(g13.b0_over_bbar) == pytest.approx(3.089, abs=0.0)


def test_from_fourier_gradient_of_fsab2_matches_finite_difference() -> None:
    """jax.grad of <B^2> w.r.t. a bmnc coefficient: finite and matches FD at 1e-5."""
    n_theta, n_zeta, n_periods = 11, 9, 5
    theta, zeta = _grid(n_theta, n_zeta, n_periods)
    theta_weights = jnp.full((n_theta,), 2.0 * math.pi / n_theta)
    zeta_weights = jnp.full((n_zeta,), (2.0 * math.pi / n_periods / n_zeta) * n_periods)
    m = jnp.asarray([0, 0, 1, 1, 2])
    n = jnp.asarray([0, 1, 1, 0, 1])
    coeff0 = jnp.asarray([3.089, 0.1435, -0.1344, -0.0588, 0.0210], dtype=jnp.float64)

    def fsab2(coeff: jnp.ndarray) -> jnp.ndarray:
        geom = FluxSurfaceGeometry.from_fourier(
            theta=theta,
            zeta=zeta,
            bmnc=coeff,
            m=m,
            n=n,
            n_periods=n_periods,
            iota=0.87,
            g_hat=-17.885,
            i_hat=0.0,
        )
        return geom.fsab_hat2(theta_weights=theta_weights, zeta_weights=zeta_weights)

    value, gradient = jax.value_and_grad(fsab2)(coeff0)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 1.0e-8

    eps = 1.0e-5
    for k in range(int(coeff0.size)):
        e_k = jnp.zeros_like(coeff0).at[k].set(1.0)
        fd = float((fsab2(coeff0 + eps * e_k) - fsab2(coeff0 - eps * e_k)) / (2.0 * eps))
        np.testing.assert_allclose(float(gradient[k]), fd, rtol=5.0e-7, atol=1.0e-10)

    # The constructor must also be jit-safe (static shapes, no python branches
    # on traced values):
    np.testing.assert_allclose(float(jax.jit(fsab2)(coeff0)), float(value), rtol=0.0, atol=0.0)


def test_from_fourier_with_sine_spectrum_is_finite_and_truncated() -> None:
    """bmns handling plus grid truncation of unrepresentable modes."""
    theta, zeta = _grid(7, 5, 5)
    m = jnp.asarray([0, 1, 30])  # m=30 is unrepresentable on Ntheta=7
    n = jnp.asarray([0, 1, 0])
    bmnc = jnp.asarray([1.0, 0.05, 0.7])
    bmns = jnp.asarray([0.0, 0.01, 0.0])
    geom = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=bmnc, m=m, n=n, bmns=bmns, n_periods=5, iota=0.9, g_hat=1.1, i_hat=0.0
    )
    assert np.all(np.isfinite(np.asarray(geom.b_hat)))
    # The truncated m=30 cosine (amplitude 0.7) must not contribute:
    assert float(jnp.max(jnp.abs(geom.b_hat))) < 1.1
    # And its gradient path is exactly zeroed:
    grad = jax.grad(
        lambda c: jnp.sum(
            FluxSurfaceGeometry.from_fourier(
                theta=theta, zeta=zeta, bmnc=c, m=m, n=n, bmns=bmns, n_periods=5, iota=0.9, g_hat=1.1, i_hat=0.0
            ).b_hat
        )
    )(bmnc)
    assert float(grad[2]) == 0.0


# ---------------------------------------------------------------------------
# _u_and_bsubpsi vectorization: bit-identity vs the scalar-loop reference
# ---------------------------------------------------------------------------


def _u_and_bsubpsi_reference(
    *, theta, zeta, n_periods, b_hat, iota, g_hat, i_hat, p_prime_hat, non_stel_sym
):
    """Pre-vectorization scalar-loop reference for ``_u_and_bsubpsi``.

    The production routine hoists the per-harmonic trig to a single 2-D
    ``cos``/``sin`` and does the ``b_sub_psi`` accumulation as whole-grid
    in-place adds; both are order-preserving, so its output must stay
    bit-for-bit equal to this literal transcription of the original loop
    (which keeps the ``h_amp`` reduction as sequential per-``itheta``
    ``np.dot``).  Any drift here would silently change the Boozer
    magnetic-drift operator.
    """
    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])
    h_hat = 1.0 / (b_hat * b_hat)
    b_sub_psi = np.zeros_like(b_hat)
    db_sub_psi_dtheta = np.zeros_like(b_hat)
    db_sub_psi_dzeta = np.zeros_like(b_hat)
    m_max = int(ntheta / 2.0)
    n_max = int(nzeta / 2.0)
    theta_half = ntheta / 2.0
    zeta_half = nzeta / 2.0
    zeta_is_even = float(int(zeta_half)) == zeta_half
    for m in range(m_max + 1):
        if m == 0:
            startn = 1
        elif float(m) == theta_half:
            startn = 0
        elif zeta_is_even:
            startn = -n_max + 1
        else:
            startn = -n_max
        for n in range(startn, n_max + 1):
            for is_cos in (True, False):
                if not is_cos and not non_stel_sym:
                    continue
                nyquist = (
                    (m == 0 and float(n) == zeta_half)
                    or (float(m) == theta_half and n == 0)
                    or (float(m) == theta_half and float(n) == zeta_half)
                )
                h_amp = 0.0
                if is_cos:
                    for itheta in range(ntheta):
                        ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                        w = (1.0 if nyquist else 2.0) / float(ntheta * nzeta)
                        h_amp += w * float(np.dot(np.cos(ang), h_hat[itheta, :]))
                elif not nyquist:
                    for itheta in range(ntheta):
                        ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                        h_amp += (2.0 / float(ntheta * nzeta)) * float(np.dot(np.sin(ang), h_hat[itheta, :]))
                denom = float(n * n_periods) - float(iota) * float(m)
                numer = float(iota) * (float(g_hat) * float(m) + float(i_hat) * float(n * n_periods))
                u_amp = 0.0 if denom == 0.0 else (numer / denom) * h_amp
                d_dtheta_amp = -float(p_prime_hat) / float(iota) * (u_amp - float(iota) * float(i_hat) * h_amp)
                d_dzeta_amp = float(p_prime_hat) * (u_amp + float(g_hat) * h_amp)
                for itheta in range(ntheta):
                    ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                    c = np.cos(ang)
                    s = np.sin(ang)
                    if is_cos:
                        db_sub_psi_dtheta[itheta, :] += d_dtheta_amp * c
                        db_sub_psi_dzeta[itheta, :] += d_dzeta_amp * c
                        if n == 0:
                            b_sub_psi[itheta, :] += (d_dtheta_amp / float(m)) * s
                        else:
                            b_sub_psi[itheta, :] += -(d_dzeta_amp / float(n) / float(n_periods)) * s
                    else:
                        db_sub_psi_dtheta[itheta, :] += d_dtheta_amp * s
                        db_sub_psi_dzeta[itheta, :] += d_dzeta_amp * s
                        if n == 0:
                            b_sub_psi[itheta, :] += -(d_dtheta_amp / float(m)) * c
                        else:
                            b_sub_psi[itheta, :] += (d_dzeta_amp / float(n) / float(n_periods)) * c
    return b_sub_psi, db_sub_psi_dtheta, db_sub_psi_dzeta


def _u_bsubpsi_inputs(n_theta: int, n_zeta: int, n_periods: int, non_stel_sym: bool):
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False, dtype=np.float64)
    zeta = np.linspace(0.0, 2.0 * math.pi / n_periods, n_zeta, endpoint=False, dtype=np.float64)
    th = theta[:, None]
    ze = zeta[None, :]
    # A smooth, strictly positive |B| with a helical ripple (and, when
    # non_stel_sym, a sine component so the odd-parity harmonic branch runs).
    b_hat = 1.0 + 0.12 * np.cos(th) + 0.06 * np.cos(th - n_periods * ze) - 0.03 * np.cos(2.0 * n_periods * ze)
    if non_stel_sym:
        b_hat = b_hat + 0.04 * np.sin(th - n_periods * ze) + 0.02 * np.sin(2.0 * th)
    return np.ascontiguousarray(theta), np.ascontiguousarray(zeta), np.ascontiguousarray(b_hat)


@pytest.mark.parametrize("n_theta,n_zeta", [(12, 12), (13, 11), (17, 33), (8, 6)])
@pytest.mark.parametrize("non_stel_sym", [False, True])
def test_u_and_bsubpsi_matches_scalar_reference(n_theta, n_zeta, non_stel_sym) -> None:
    """The vectorized ``_u_and_bsubpsi`` must be bit-for-bit the scalar loop.

    Covers even/odd Ntheta,Nzeta (the theta_half / zeta_is_even / nyquist
    branches) and both the stellarator-symmetric and non-symmetric spectra.
    """
    n_periods = 4
    theta, zeta, b_hat = _u_bsubpsi_inputs(n_theta, n_zeta, n_periods, non_stel_sym)
    kwargs = dict(
        theta=theta,
        zeta=zeta,
        n_periods=n_periods,
        b_hat=b_hat,
        iota=0.87,
        g_hat=1.13,
        i_hat=0.021,
        p_prime_hat=-0.35,
        non_stel_sym=non_stel_sym,
    )
    got = _u_and_bsubpsi(**kwargs)
    ref = _u_and_bsubpsi_reference(**kwargs)
    for g, r in zip(got, ref):
        assert np.array_equal(g, r)
