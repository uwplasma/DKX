"""Flux-surface magnetic geometry for every SFINCS geometry scheme.

This module consolidates the flux-surface geometry package
(``sfincs_jax/geometry/{__init__,boozer,vmec,vmec_wout}.py``) into one file,
mirroring the geometry part of the Fortran v3 code base (``geometry.F90``):

* ``initializeGeometry`` / ``computeBHat_Boozer`` — analytic Boozer models
  (geometryScheme 1 = three-helicity model, 2 = LHD standard, 3 = LHD
  inward-shifted, 4 = W7-X standard; harmonic tables from Beidler et al.,
  Nuclear Fusion 51, 076001 (2011), Table 1), Boozer ``.bc`` equilibrium files
  (geometryScheme 11/12), and namelist-supplied Boozer spectra
  (geometryScheme 13, the STELLOPT/optimization path).
* ``computeBHat_VMEC`` — VMEC ``wout`` files (geometryScheme 5), including the
  half/full-mesh radial interpolation and finite-difference conventions.
* ``setBoozerCoordinates`` — the map from ``BHat`` and the Boozer flux
  functions ``GHat``, ``IHat``, ``iota`` to ``DHat`` and the co/contravariant
  field components.
* ``computeBIntegrals`` — ``VPrimeHat`` and ``FSABHat2`` flux-surface averages.
* the ``gpsiHatpsiHat`` metric reconstruction used by Sugama magnetic drifts.

All quantities carry SFINCS normalizations (hats): lengths in ``RBar``, fields
in ``BBar``, and ``psiHat = psi / (BBar RBar^2)``.

Differentiability: :meth:`FluxSurfaceGeometry.from_scheme` and
:meth:`FluxSurfaceGeometry.from_fourier` are pure JAX and safe to ``jit`` /
``grad`` (``from_fourier`` is the geometry entry point for optimization
loops).  File readers (``read_vmec_wout``, ``read_boozer_bc``) are plain-NumPy
pure functions kept separate from geometry construction.  This file is the
single geometry owner (it replaced the retired ``sfincs_jax/geometry/``
package).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .paths import resolve_existing_path  # noqa: E402

__all__ = [
    "BoozerBcHeader",
    "BoozerBcSurface",
    "FluxSurfaceGeometry",
    "VmecWout",
    "VmecRadialInterpolation",
    "psi_a_hat_from_wout",
    "read_boozer_bc",
    "read_vmec_wout",
    "selected_r_n_from_bc",
    "vmec_radial_interpolation",
]

_MU0 = 4.0 * math.pi * 1e-7


# =============================================================================
# Container
# =============================================================================


@dataclass(frozen=True)
class FluxSurfaceGeometry:
    """Normalized single-surface geometry on a ``(n_theta, n_zeta)`` grid.

    Field names follow the SFINCS v3 variables written to ``sfincsOutput.h5``
    (``BHat`` -> ``b_hat``, ``DHat`` -> ``d_hat``, ...).  The container is flat
    and explicit so parity tests, finite-difference checks, and JAX
    differentiation gates are straightforward to write and audit.  Scalars may
    be Python floats or traced JAX scalars (``from_fourier`` keeps them
    traced so geometry-dependent objectives stay differentiable).
    """

    # Scalars:
    n_periods: int
    b0_over_bbar: float | jnp.ndarray
    iota: float | jnp.ndarray
    g_hat: float | jnp.ndarray
    i_hat: float | jnp.ndarray

    # (n_theta, n_zeta) arrays:
    b_hat: jnp.ndarray
    db_hat_dtheta: jnp.ndarray
    db_hat_dzeta: jnp.ndarray

    d_hat: jnp.ndarray
    b_hat_sup_theta: jnp.ndarray
    b_hat_sup_zeta: jnp.ndarray
    b_hat_sub_theta: jnp.ndarray
    b_hat_sub_zeta: jnp.ndarray

    # Radial/drift-related arrays (zero for the analytic schemes, populated
    # from nearby surfaces for .bc/VMEC input):
    b_hat_sub_psi: jnp.ndarray
    db_hat_dpsi_hat: jnp.ndarray
    db_hat_sub_psi_dtheta: jnp.ndarray
    db_hat_sub_psi_dzeta: jnp.ndarray
    db_hat_sub_theta_dpsi_hat: jnp.ndarray
    db_hat_sub_zeta_dpsi_hat: jnp.ndarray
    db_hat_sub_theta_dzeta: jnp.ndarray
    db_hat_sub_zeta_dtheta: jnp.ndarray
    db_hat_sup_theta_dpsi_hat: jnp.ndarray
    db_hat_sup_theta_dzeta: jnp.ndarray
    db_hat_sup_zeta_dpsi_hat: jnp.ndarray
    db_hat_sup_zeta_dtheta: jnp.ndarray

    # |grad psiHat|^2 metric (Sugama drifts); populated on request only:
    gpsipsi: jnp.ndarray | None = field(default=None)

    # Radial flux-function derivatives (geometry.F90 ``pPrimeHat``/``diotadpsiHat``;
    # zero for the analytic schemes, from nearby surfaces for .bc/VMEC input).
    # ``pPrimeHat`` drives magneticDriftScheme 6, ``diotadpsiHat`` the shear terms
    # of magneticDriftScheme 4/8:
    p_prime_hat: float | jnp.ndarray = field(default=0.0)
    diota_dpsi_hat: float | jnp.ndarray = field(default=0.0)

    # Sugama normal-curvature factor ``(grad psiHat . grad BHat)/gpsiHatpsiHat``
    # (geometry.F90 ``gradpsidotgradB_overgpsipsi``; magneticDriftScheme 5/6);
    # populated on request only:
    grad_psi_dot_grad_b_over_gpsipsi: jnp.ndarray | None = field(default=None)

    # ---- flux-surface averages (geometry.F90::computeBIntegrals) ----

    def vprime_hat(self, *, theta_weights: jnp.ndarray, zeta_weights: jnp.ndarray) -> jnp.ndarray:
        """``VPrimeHat = sum_ij w_theta_i w_zeta_j / DHat_ij`` (v3 computeBIntegrals)."""
        w = jnp.asarray(theta_weights)[:, None] * jnp.asarray(zeta_weights)[None, :]
        return jnp.sum(w / self.d_hat)

    def fsab_hat2(self, *, theta_weights: jnp.ndarray, zeta_weights: jnp.ndarray) -> jnp.ndarray:
        """``FSABHat2 = <BHat^2>`` with the Jacobian-weighted flux-surface average."""
        w = jnp.asarray(theta_weights)[:, None] * jnp.asarray(zeta_weights)[None, :]
        vprime = jnp.sum(w / self.d_hat)
        return jnp.sum(w * (self.b_hat**2) / self.d_hat) / vprime

    # Constructors ``from_scheme``, ``from_fourier``, ``from_boozer``, and
    # ``from_vmec`` are module-level functions bound as classmethods at the
    # end of this file (keeping the dataclass body a pure data contract).


def _geometry_from_boozer_bhat(
    *,
    n_periods: int,
    b0_over_bbar,
    iota,
    g_hat,
    i_hat,
    b_hat: jnp.ndarray,
    db_hat_dtheta: jnp.ndarray,
    db_hat_dzeta: jnp.ndarray,
    **radial_fields,
) -> FluxSurfaceGeometry:
    """v3 ``setBoozerCoordinates``: derive DHat and B components from BHat.

    In Boozer coordinates ``DHat = BHat^2 / (GHat + iota IHat)``,
    ``BHat^theta = iota DHat``, ``BHat^zeta = DHat``, ``BHat_theta = IHat``,
    ``BHat_zeta = GHat``.  Radial-derivative arrays default to zero (as in v3
    when no nearby surfaces are available) and can be overridden by keyword.
    """
    denom = g_hat + iota * i_hat
    d_hat = (b_hat * b_hat) / denom
    zeros = jnp.zeros_like(b_hat)
    fields = {
        "b_hat_sub_psi": zeros,
        "db_hat_dpsi_hat": zeros,
        "db_hat_sub_psi_dtheta": zeros,
        "db_hat_sub_psi_dzeta": zeros,
        "db_hat_sub_theta_dpsi_hat": zeros,
        "db_hat_sub_zeta_dpsi_hat": zeros,
        "db_hat_sub_theta_dzeta": zeros,
        "db_hat_sub_zeta_dtheta": zeros,
        "db_hat_sup_theta_dpsi_hat": zeros,
        "db_hat_sup_theta_dzeta": zeros,
        "db_hat_sup_zeta_dpsi_hat": zeros,
        "db_hat_sup_zeta_dtheta": zeros,
        "gpsipsi": None,
        "p_prime_hat": 0.0,
        "diota_dpsi_hat": 0.0,
        "grad_psi_dot_grad_b_over_gpsipsi": None,
    }
    fields.update(radial_fields)
    return FluxSurfaceGeometry(
        n_periods=n_periods,
        b0_over_bbar=b0_over_bbar,
        iota=iota,
        g_hat=g_hat,
        i_hat=i_hat,
        b_hat=b_hat,
        db_hat_dtheta=db_hat_dtheta,
        db_hat_dzeta=db_hat_dzeta,
        d_hat=d_hat,
        b_hat_sup_theta=iota * d_hat,
        b_hat_sup_zeta=d_hat,
        b_hat_sub_theta=jnp.full_like(b_hat, i_hat),
        b_hat_sub_zeta=jnp.full_like(b_hat, g_hat),
        **fields,
    )


# =============================================================================
# Fourier evaluation of BHat (shared by the analytic schemes and scheme 13)
# =============================================================================


def _harmonics_bhat_jax(
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    n_periods: int,
    b0: jnp.ndarray | float,
    m: jnp.ndarray,
    n: jnp.ndarray,
    parity: jnp.ndarray,
    amp: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate ``BHat = b0 + sum_h amp_h {cos|sin}(m_h theta - NPeriods n_h zeta)``.

    ``parity`` is True for cosine harmonics, False for sine.  Returns
    ``(BHat, dBHat/dtheta, dBHat/dzeta)`` on the tensor-product grid.  Pure
    JAX; the single evaluator behind geometry schemes 1-4 and 13.
    """
    theta2 = jnp.asarray(theta, dtype=jnp.float64)[:, None]
    zeta2 = jnp.asarray(zeta, dtype=jnp.float64)[None, :]
    m3 = jnp.asarray(m, dtype=jnp.float64)[:, None, None]
    n3 = jnp.asarray(n, dtype=jnp.float64)[:, None, None]
    parity3 = jnp.asarray(parity, dtype=bool)[:, None, None]
    amp3 = jnp.asarray(amp, dtype=jnp.float64)[:, None, None]

    angle = m3 * theta2[None, :, :] - float(n_periods) * n3 * zeta2[None, :, :]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    basis = jnp.where(parity3, cos_a, sin_a)
    b_hat = b0 + jnp.sum(amp3 * basis, axis=0)

    dtheta_basis = jnp.where(parity3, -m3 * sin_a, m3 * cos_a)
    db_hat_dtheta = jnp.sum(amp3 * dtheta_basis, axis=0)

    dzeta_factor = float(n_periods) * n3
    dzeta_basis = jnp.where(parity3, dzeta_factor * sin_a, -dzeta_factor * cos_a)
    db_hat_dzeta = jnp.sum(amp3 * dzeta_basis, axis=0)
    return b_hat, db_hat_dtheta, db_hat_dzeta


def _grid_mode_mask(
    *,
    n_theta: int,
    n_zeta: int,
    m: jnp.ndarray,
    n: jnp.ndarray,
    parity: jnp.ndarray,
) -> jnp.ndarray:
    """v3 ``computeBHat_Boozer`` truncation of harmonics to grid-representable modes.

    Cosine modes need ``m <= Ntheta/2`` and ``|n| <= Nzeta/2``; sine modes at a
    theta Nyquist point (``m = 0`` or ``m = Ntheta/2``) combined with a zeta
    Nyquist point (``n = 0`` or ``|n| = Nzeta/2``) are excluded because sine is
    not representable there.  For ``Nzeta = 1`` every harmonic is kept.
    """
    m = jnp.asarray(m)
    n = jnp.asarray(n)
    parity = jnp.asarray(parity, dtype=bool)
    if int(n_zeta) == 1:
        return jnp.ones(m.shape, dtype=bool)
    m_max = int(n_theta / 2.0)
    n_max = int(n_zeta / 2.0)
    include = (jnp.abs(n) <= n_max) & (m <= m_max)
    at_m_nyq = (m == 0) | (m.astype(jnp.float64) == (n_theta / 2.0))
    at_n_nyq = (n == 0) | (jnp.abs(n.astype(jnp.float64)) == (n_zeta / 2.0))
    return include & ~((~parity) & at_m_nyq & at_n_nyq)


# =============================================================================
# Analytic schemes 1-4 and namelist spectra (scheme 13)
# =============================================================================

# (m, n, amplitude/B0) cosine tables, Beidler et al. NF 51, 076001 (2011), Table 1.
_SCHEME2_TABLE = ((1, 0, -0.07053), (2, 1, 0.05067), (1, 1, -0.01476))  # LHD standard
_SCHEME3_TABLE = ((1, 0, -0.05927), (2, 1, 0.05267), (1, 1, -0.04956), (0, 1, 0.01045))  # LHD inward-shifted
_SCHEME4_TABLE = ((0, 1, 0.04645), (1, 1, -0.04351), (1, 0, -0.01902))  # W7-X standard

# Per-scheme flux functions hard-coded in geometry.F90 (GHat = B0OverBBar * R0
# for schemes 2/3; scheme 4 uses the measured GHat = -17.885).
_SCHEME_CONSTANTS = {
    2: dict(n_periods=10, iota=0.4542, b0_over_bbar=1.0, g_hat=1.0 * 3.7481, i_hat=0.0, table=_SCHEME2_TABLE),
    3: dict(n_periods=10, iota=0.4692, b0_over_bbar=1.0, g_hat=1.0 * 3.6024, i_hat=0.0, table=_SCHEME3_TABLE),
    4: dict(n_periods=5, iota=0.8700, b0_over_bbar=3.089, g_hat=-17.885, i_hat=0.0, table=_SCHEME4_TABLE),
}


def _from_scheme(
    cls,
    scheme: int,
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    epsilon_t: float = -0.07053,
    epsilon_h: float = 0.05067,
    epsilon_antisymm: float = 0.0,
    iota: float = 0.4542,
    g_hat: float = 3.7481,
    i_hat: float = 0.0,
    b0_over_bbar: float = 1.0,
    helicity_l: int = 2,
    helicity_n: int = 10,
    helicity_antisymm_l: int = 1,
    helicity_antisymm_n: int = 0,
) -> FluxSurfaceGeometry:
    """Analytic Boozer geometry for ``geometryScheme`` 1, 2, 3, or 4.

    Scheme 1 is the three-helicity model of ``geometry.F90``::

      BHat = B0OverBBar * (1 + epsilon_t cos(theta)
                             + epsilon_h cos(helicity_l theta - helicity_n zeta)
                             + epsilon_antisymm sin(helicity_antisymm_l theta
                                                    - helicity_antisymm_n zeta))

    with ``NPeriods = max(1, helicity_n)``; only scheme 1 consumes the keyword
    parameters (namelist defaults shown).  Schemes 2/3/4 are the fixed LHD
    standard, LHD inward-shifted, and W7-X standard harmonic tables.
    """
    if scheme == 1:
        n_periods = max(1, int(helicity_n))
        # v3 stores harmonics as cos(m theta - NPeriods n zeta); with
        # NPeriods = helicity_n the helical ripple term uses n = 1.
        n2 = 0 if int(helicity_n) == 0 else 1
        if int(helicity_n) == 0:
            n3 = int(helicity_antisymm_n)
        else:
            n3 = int(helicity_antisymm_n) // int(helicity_n)
        m = jnp.asarray([1, int(helicity_l), int(helicity_antisymm_l)], dtype=jnp.float64)
        n = jnp.asarray([0, int(n2), int(n3)], dtype=jnp.float64)
        parity = jnp.asarray([True, True, False], dtype=bool)
        amp = jnp.asarray([float(epsilon_t), float(epsilon_h), float(epsilon_antisymm)], dtype=jnp.float64) * float(
            b0_over_bbar
        )
        iota_v, g_v, i_v, b0_v = float(iota), float(g_hat), float(i_hat), float(b0_over_bbar)
    elif scheme in _SCHEME_CONSTANTS:
        c = _SCHEME_CONSTANTS[scheme]
        n_periods = int(c["n_periods"])
        table = np.asarray(c["table"], dtype=np.float64)
        m = jnp.asarray(table[:, 0])
        n = jnp.asarray(table[:, 1])
        parity = jnp.ones((table.shape[0],), dtype=bool)
        amp = jnp.asarray(table[:, 2]) * float(c["b0_over_bbar"])
        iota_v, g_v, i_v, b0_v = float(c["iota"]), float(c["g_hat"]), float(c["i_hat"]), float(c["b0_over_bbar"])
    else:
        raise ValueError(f"from_scheme covers geometryScheme 1-4; got {scheme}")

    b_hat, db_dtheta, db_dzeta = _harmonics_bhat_jax(
        theta=theta, zeta=zeta, n_periods=n_periods, b0=b0_v, m=m, n=n, parity=parity, amp=amp
    )
    return _geometry_from_boozer_bhat(
        n_periods=n_periods,
        b0_over_bbar=b0_v,
        iota=iota_v,
        g_hat=g_v,
        i_hat=i_v,
        b_hat=b_hat,
        db_hat_dtheta=db_dtheta,
        db_hat_dzeta=db_dzeta,
    )


def _from_fourier(
    cls,
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    bmnc: jnp.ndarray,
    m: jnp.ndarray,
    n: jnp.ndarray,
    bmns: jnp.ndarray | None = None,
    n_periods: int,
    iota,
    g_hat,
    i_hat,
    apply_grid_truncation: bool = True,
) -> FluxSurfaceGeometry:
    """Differentiable Boozer geometry from a ``|B|`` spectrum (``geometryScheme = 13``).

    ``bmnc`` (and optionally ``bmns``) hold the cosine (sine) amplitudes of
    ``BHat`` in ``BBar`` units for mode numbers ``(m, n)`` with the angle
    convention ``m theta - NPeriods n zeta`` (``n`` does NOT include the field
    period factor, matching the Fortran ``boozer_bmnc(m,n)`` namelist arrays).
    ``B0OverBBar`` is the ``(0, 0)`` cosine amplitude.  ``iota``, ``g_hat``,
    ``i_hat`` come from the namelist as in v3.

    This constructor is pure JAX and traceable in the amplitudes and the
    scalar flux functions: it is the geometry path used by optimization loops
    (gradients w.r.t. ``bmnc``/``bmns`` flow through ``BHat``, ``DHat``, and
    the flux-surface averages).  Grid truncation of unrepresentable modes
    (v3 ``include_mn``) is applied by zeroing amplitudes, which keeps shapes
    static under ``jit``; disable with ``apply_grid_truncation=False``.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    coeff_c = jnp.asarray(bmnc, dtype=jnp.float64)
    m_arr = jnp.asarray(m)
    n_arr = jnp.asarray(n)
    if coeff_c.ndim != 1 or m_arr.shape != coeff_c.shape or n_arr.shape != coeff_c.shape:
        raise ValueError("bmnc, m, and n must be 1-D arrays of equal length")

    b0_over_bbar = jnp.sum(jnp.where((m_arr == 0) & (n_arr == 0), coeff_c, 0.0))

    if bmns is None:
        m_all, n_all = m_arr, n_arr
        parity = jnp.ones(coeff_c.shape, dtype=bool)
        amp = coeff_c
    else:
        coeff_s = jnp.asarray(bmns, dtype=jnp.float64)
        if coeff_s.shape != coeff_c.shape:
            raise ValueError("bmns must have the same shape as bmnc")
        m_all = jnp.concatenate([m_arr, m_arr])
        n_all = jnp.concatenate([n_arr, n_arr])
        parity = jnp.concatenate([jnp.ones(coeff_c.shape, dtype=bool), jnp.zeros(coeff_s.shape, dtype=bool)])
        amp = jnp.concatenate([coeff_c, coeff_s])

    if apply_grid_truncation:
        include = _grid_mode_mask(
            n_theta=int(theta.shape[0]), n_zeta=int(zeta.shape[0]), m=m_all, n=n_all, parity=parity
        )
        amp = jnp.where(include, amp, 0.0)

    # The (0,0) cosine term is evaluated inside the sum (cos(0) = 1), so b0=0.
    b_hat, db_dtheta, db_dzeta = _harmonics_bhat_jax(
        theta=theta, zeta=zeta, n_periods=int(n_periods), b0=0.0, m=m_all, n=n_all, parity=parity, amp=amp
    )
    return _geometry_from_boozer_bhat(
        n_periods=int(n_periods),
        b0_over_bbar=b0_over_bbar,
        iota=iota,
        g_hat=g_hat,
        i_hat=i_hat,
        b_hat=b_hat,
        db_hat_dtheta=db_dtheta,
        db_hat_dzeta=db_dzeta,
    )


# =============================================================================
# Boozer .bc files (geometryScheme 11/12): reader
# =============================================================================


@dataclass(frozen=True)
class BoozerBcHeader:
    """Header of a ``.bc`` Boozer file, with v3 sign conventions applied."""

    n_periods: int
    psi_a_hat: float
    a_hat: float
    turkin_sign: int


@dataclass(frozen=True)
class BoozerBcSurface:
    """One flux surface of a ``.bc`` file: scalars plus the Fourier tables."""

    r_n: float
    iota: float
    g_hat: float
    i_hat: float
    p_prime_hat: float
    b0_over_bbar: float
    r0: float
    m: np.ndarray  # (H,) int32
    n: np.ndarray  # (H,) int32
    parity: np.ndarray  # (H,) bool, True=cos, False=sin
    b_amp: np.ndarray  # (H,) float64
    r_amp: np.ndarray  # (H,) float64
    z_amp: np.ndarray  # (H,) float64
    dz_amp: np.ndarray  # (H,) float64


def _parse_bc_header_line(line: str) -> tuple[list[int], list[float]]:
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Unexpected .bc header line (too short): {line!r}")
    return [int(x) for x in parts[:4]], [float(x.replace("D", "E").replace("d", "E")) for x in parts[4:]]


def _try_parse_floats(tokens: list[str], count: int) -> list[float] | None:
    if len(tokens) < count:
        return None
    out: list[float] = []
    for t in tokens[:count]:
        try:
            out.append(float(t.replace("D", "E").replace("d", "E")))
        except ValueError:
            return None
    return out


def _try_parse_ints(tokens: list[str], count: int) -> list[int] | None:
    if len(tokens) < count:
        return None
    out: list[int] = []
    for t in tokens[:count]:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out


def _iter_bc_surfaces(
    *, fh: IO[str], geometry_scheme: int, n_periods: int, psi_a_hat: float
) -> Iterator[BoozerBcSurface]:
    """Yield surfaces in order, following v3's list-directed read patterns."""
    _ = fh.readline()  # v3 reads and discards one line after the header.
    while True:
        line = fh.readline()
        if not line:
            return
        if "s" not in line:  # scan to the next surface marker line
            continue
        surf_header: list[float] | None = None
        while surf_header is None:
            line = fh.readline()
            if not line:
                return
            surf_header = _try_parse_floats(line.split(), 5)
        s, iota, g_raw, i_raw, pprime_raw = surf_header[:5]
        # G and I pick up a minus sign from Ampere's law in the left-handed
        # (r, pol, tor) system; amperes -> normalized units via mu0/(2 pi).
        g_hat = -float(g_raw) * float(n_periods) / (2.0 * math.pi) * _MU0
        i_hat = -float(i_raw) / (2.0 * math.pi) * _MU0
        p_prime_hat = float(pprime_raw) / float(psi_a_hat) * _MU0

        _ = fh.readline()  # units line
        m_list: list[int] = []
        n_list: list[int] = []
        parity_list: list[bool] = []
        b_list: list[float] = []
        r_list: list[float] = []
        z_list: list[float] = []
        dz_list: list[float] = []
        b0_over_bbar = 0.0
        r0 = 0.0
        found_b00 = False

        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if "s" in line:
                fh.seek(pos)  # push back for the next surface
                break
            tokens = line.split()
            ij = _try_parse_ints(tokens, 2)
            if ij is None:
                continue
            m, n = int(ij[0]), int(ij[1])
            if geometry_scheme == 11:
                vals = _try_parse_floats(tokens[2:], 4)
                if vals is None:
                    continue
                rmn, zmn, dzmn, bmn = vals
                if m == 0 and n == 0:
                    b0_over_bbar, r0, found_b00 = float(bmn), float(rmn), True
                else:
                    m_list.append(m)
                    n_list.append(n)
                    parity_list.append(True)  # cosine-only format
                    r_list.append(float(rmn))
                    z_list.append(float(zmn))
                    dz_list.append(float(dzmn))
                    b_list.append(float(bmn))
            else:
                vals8 = _try_parse_floats(tokens[2:], 8)
                if vals8 is None:
                    continue
                if m == 0 and n == 0:
                    b0_over_bbar, r0, found_b00 = float(vals8[6]), float(vals8[0]), True
                else:
                    # Non-stellarator-symmetric format: each (m,n) expands into
                    # a cosine entry and a sine entry (v3 geometryScheme=12).
                    rcos, rsin, zcos, zsin, dzcos, dzsin, bcos, bsin = vals8
                    m_list += [m, m]
                    n_list += [n, n]
                    parity_list += [True, False]
                    r_list += [float(rcos), float(rsin)]
                    z_list += [float(zsin), float(zcos)]
                    dz_list += [float(dzsin), float(dzcos)]
                    b_list += [float(bcos), float(bsin)]

        if not found_b00:
            raise ValueError("No (0,0) mode found in Boozer .bc file surface block.")
        yield BoozerBcSurface(
            r_n=math.sqrt(float(s)),
            iota=float(iota),
            g_hat=g_hat,
            i_hat=i_hat,
            p_prime_hat=p_prime_hat,
            b0_over_bbar=float(b0_over_bbar),
            r0=float(r0),
            m=np.asarray(m_list, dtype=np.int32),
            n=np.asarray(n_list, dtype=np.int32),
            parity=np.asarray(parity_list, dtype=bool),
            b_amp=np.asarray(b_list, dtype=np.float64),
            r_amp=np.asarray(r_list, dtype=np.float64),
            z_amp=np.asarray(z_list, dtype=np.float64),
            dz_amp=np.asarray(dz_list, dtype=np.float64),
        )


def read_boozer_bc(
    path: str | Path, *, geometry_scheme: int
) -> tuple[BoozerBcHeader, tuple[BoozerBcSurface, ...]]:
    """Read a v3 ``.bc`` Boozer equilibrium file (geometryScheme 11/12).

    Pure function returning plain NumPy arrays; the sign switches applied in
    v3 ``geometry.F90`` (left- to right-handed system, and the ``CStconfig``
    files that need an extra flip for scheme 11) are folded into the header
    ``psi_a_hat``.  Not differentiable; downstream construction is.
    """
    if geometry_scheme not in {11, 12}:
        raise ValueError(f"geometry_scheme must be 11 or 12, got {geometry_scheme}")
    p = Path(path).expanduser()
    if not p.exists():
        p = resolve_existing_path(p).path

    turkin_sign = 1
    with p.open("r") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {str(p)!r}")
            if line.startswith("CC"):
                # v3: this substring marks files saved by Yu. Turkin's
                # CStconfig, which use an extra sign convention.
                if geometry_scheme == 11 and "CStconfig" in line:
                    turkin_sign = -1
                continue
            try:
                header_ints, header_reals = _parse_bc_header_line(line)
            except Exception:  # noqa: BLE001 - column-name lines precede the numeric header
                continue
            n_periods = int(header_ints[3])
            psi_a_hat = float(header_reals[0]) / (2.0 * math.pi)
            a_hat = float(header_reals[1])
            break
        # Left-handed -> right-handed (radial, poloidal, toroidal):
        if geometry_scheme == 11:
            psi_a_hat = psi_a_hat * (-1.0) * float(turkin_sign)
        else:
            psi_a_hat = psi_a_hat * (-1.0)
        header = BoozerBcHeader(n_periods=n_periods, psi_a_hat=psi_a_hat, a_hat=a_hat, turkin_sign=turkin_sign)
        surfaces = tuple(
            _iter_bc_surfaces(fh=f, geometry_scheme=geometry_scheme, n_periods=n_periods, psi_a_hat=psi_a_hat)
        )
    return header, surfaces


def _bracketing_surfaces(
    surfaces: tuple[BoozerBcSurface, ...], r_n_wish: float
) -> tuple[BoozerBcSurface, BoozerBcSurface]:
    """Return the pair of surfaces bracketing ``r_n_wish`` (v3 selection order)."""
    old: BoozerBcSurface | None = None
    new: BoozerBcSurface | None = None
    for s in surfaces:
        if new is None:
            new = s
        if new.r_n < float(r_n_wish):
            old = new
            new = None
            continue
        if old is None:
            old = s
        new = s
        break
    if old is None or new is None:
        raise ValueError(f"Failed to locate surfaces bracketing rN_wish={r_n_wish}")
    return old, new


def selected_r_n_from_bc(
    *, path: str | Path, geometry_scheme: int, r_n_wish: float, vmec_radial_option: int = 1
) -> float:
    """Effective radius used by v3 for geometryScheme 11/12 output metadata.

    ``vmec_radial_option=1`` snaps to the nearest surface (v3 tie-breaking);
    otherwise v3 interpolates linearly in ``s = rN^2`` and this returns the
    corresponding effective ``rN``.
    """
    _header, surfaces = read_boozer_bc(path, geometry_scheme=int(geometry_scheme))
    surf_old, surf_new = _bracketing_surfaces(surfaces, float(r_n_wish))
    r_old, r_new = float(surf_old.r_n), float(surf_new.r_n)
    if r_new == r_old:
        return r_old
    if int(vmec_radial_option) == 1:
        return r_old if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)) else r_new
    s_old, s_new, s_wish = r_old * r_old, r_new * r_new, float(r_n_wish) ** 2
    radial_weight = (s_new - s_wish) / (s_new - s_old)
    return float(math.sqrt(max(radial_weight * s_old + (1.0 - radial_weight) * s_new, 0.0)))


# =============================================================================
# Boozer .bc files: geometry construction
# =============================================================================


def _harmonics_bhat_np(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    b0_over_bbar: float,
    m: np.ndarray,
    n: np.ndarray,
    parity: np.ndarray,
    b_amp: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy twin of :func:`_harmonics_bhat_jax` with v3 grid truncation.

    Kept in NumPy (sequential chunked accumulation) so the file-backed paths
    reproduce the historical sfincs_jax results bit-for-bit; the truncation
    mask matches ``computeBHat_Boozer``'s ``include_mn`` logic.
    """
    theta1 = theta[None, :, None]
    zeta1 = zeta[None, None, :]
    out = np.full((theta.shape[0], zeta.shape[0]), float(b0_over_bbar), dtype=np.float64)
    dbdtheta = np.zeros_like(out)
    dbdzeta = np.zeros_like(out)

    include = np.asarray(
        _grid_mode_mask(n_theta=int(theta.shape[0]), n_zeta=int(zeta.shape[0]), m=m, n=n, parity=parity)
    )
    m = m[include]
    n = n[include]
    parity = parity[include]
    b_amp = b_amp[include]

    for i0 in range(0, int(m.shape[0]), chunk):
        i1 = min(int(m.shape[0]), i0 + chunk)
        mc = m[i0:i1].astype(np.float64)[:, None, None]
        nc = n[i0:i1].astype(np.float64)[:, None, None]
        bc = b_amp[i0:i1].astype(np.float64)[:, None, None]
        pc = parity[i0:i1].astype(bool)[:, None, None]
        angle = mc * theta1 - float(n_periods) * nc * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        out = out + np.sum(bc * np.where(pc, cos_a, sin_a), axis=0)
        dbdtheta = dbdtheta + np.sum(bc * np.where(pc, -mc * sin_a, mc * cos_a), axis=0)
        dzeta_factor = float(n_periods) * nc
        dbdzeta = dbdzeta + np.sum(bc * np.where(pc, dzeta_factor * sin_a, -dzeta_factor * cos_a), axis=0)
    return out, dbdtheta, dbdzeta


def _u_and_bsubpsi(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    b_hat: np.ndarray,
    iota: float,
    g_hat: float,
    i_hat: float,
    p_prime_hat: float,
    non_stel_sym: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``BHat_sub_psi`` and derivatives via v3's harmonic projection of ``1/BHat^2``.

    Solves the magnetic differential equation for Shaing's ``u`` function
    harmonic-by-harmonic (skipping resonant denominators exactly as v3 does)
    and assembles ``BHat_sub_psi`` from the pressure-gradient drive.
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


def _from_boozer(
    cls,
    path: str | Path,
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    r_n_wish: float,
    vmec_radial_option: int = 1,
    geometry_scheme: int = 11,
    compute_gpsipsi: bool = False,
    compute_grad_psi_dot_grad_b: bool = False,
) -> FluxSurfaceGeometry:
    """Boozer geometry from a ``.bc`` equilibrium file (geometryScheme 11/12).

    Follows the sign switches and nearby-surface interpolation of v3
    ``geometry.F90``: the two surfaces bracketing ``r_n_wish`` supply radial
    derivatives, ``vmec_radial_option=1`` snaps to the nearest surface, and
    the toroidal-direction sign switch is applied to ``GHat`` and ``iota``.
    The file read is not differentiable; the returned arrays are JAX arrays.
    """
    header, surfaces = read_boozer_bc(path, geometry_scheme=int(geometry_scheme))
    surf_old, surf_new = _bracketing_surfaces(surfaces, float(r_n_wish))

    r_old, r_new = float(surf_old.r_n), float(surf_new.r_n)
    if r_new == r_old:
        radial_weight = 1.0
    elif int(vmec_radial_option) == 1:
        radial_weight = 1.0 if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)) else 0.0
    else:
        radial_weight = (r_new * r_new - float(r_n_wish) ** 2) / (r_new * r_new - r_old * r_old)

    iota_old, iota_new = float(surf_old.iota), float(surf_new.iota)
    g_old, g_new = float(surf_old.g_hat), float(surf_new.g_hat)
    i_old, i_new = float(surf_old.i_hat), float(surf_new.i_hat)
    b0_old, b0_new = float(surf_old.b0_over_bbar), float(surf_new.b0_over_bbar)

    iota = iota_old * radial_weight + iota_new * (1.0 - radial_weight)
    g_hat = g_old * radial_weight + g_new * (1.0 - radial_weight)
    i_hat = i_old * radial_weight + i_new * (1.0 - radial_weight)
    b0_over_bbar = b0_old * radial_weight + b0_new * (1.0 - radial_weight)
    p_prime_hat = float(surf_old.p_prime_hat) * radial_weight + float(surf_new.p_prime_hat) * (1.0 - radial_weight)

    delta_psi_hat = float(header.psi_a_hat) * (r_new * r_new - r_old * r_old)
    if delta_psi_hat == 0.0:
        raise ValueError("delta_psi_hat is zero; cannot compute radial derivatives from nearby radii.")

    if geometry_scheme == 11 and (g_hat * float(header.psi_a_hat) > 0.0):
        g_hat, g_old, g_new = -g_hat, -g_old, -g_new
        i_hat, i_old, i_new = -i_hat, -i_old, -i_new
    # Toroidal direction sign switch:
    g_hat, g_old, g_new = -g_hat, -g_old, -g_new
    iota, iota_old, iota_new = -iota, -iota_old, -iota_new
    n_old = -np.asarray(surf_old.n, dtype=np.int32)
    n_new = -np.asarray(surf_new.n, dtype=np.int32)

    theta_np = np.asarray(theta, dtype=np.float64)
    zeta_np = np.asarray(zeta, dtype=np.float64)
    b_l, dbdtheta_l, dbdzeta_l = _harmonics_bhat_np(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b0_over_bbar=b0_old,
        m=np.asarray(surf_old.m, dtype=np.int32),
        n=n_old,
        parity=np.asarray(surf_old.parity, dtype=bool),
        b_amp=np.asarray(surf_old.b_amp, dtype=np.float64),
    )
    b_h, dbdtheta_h, dbdzeta_h = _harmonics_bhat_np(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b0_over_bbar=b0_new,
        m=np.asarray(surf_new.m, dtype=np.int32),
        n=n_new,
        parity=np.asarray(surf_new.parity, dtype=bool),
        b_amp=np.asarray(surf_new.b_amp, dtype=np.float64),
    )

    b_hat = b_l * radial_weight + b_h * (1.0 - radial_weight)
    db_hat_dtheta = dbdtheta_l * radial_weight + dbdtheta_h * (1.0 - radial_weight)
    db_hat_dzeta = dbdzeta_l * radial_weight + dbdzeta_h * (1.0 - radial_weight)
    db_hat_dpsi_hat = (b_h - b_l) / float(delta_psi_hat)

    db_sub_theta_dpsi = np.full_like(b_hat, (i_new - i_old) / float(delta_psi_hat))
    db_sub_zeta_dpsi = np.full_like(b_hat, (g_new - g_old) / float(delta_psi_hat))

    denom = float(g_hat) + float(iota) * float(i_hat)
    d_hat = (b_hat * b_hat) / denom
    diotadpsi_hat = (float(iota_new) - float(iota_old)) / float(delta_psi_hat)

    d_bsup_zeta_dpsi = (
        2.0 * b_hat * db_hat_dpsi_hat / denom
        - (db_sub_zeta_dpsi + float(iota) * db_sub_theta_dpsi + float(diotadpsi_hat) * float(i_hat)) / (denom * denom)
    )
    d_bsup_zeta_dtheta = 2.0 * b_hat * db_hat_dtheta / denom
    d_bsup_theta_dpsi = float(iota) * d_bsup_zeta_dpsi + float(diotadpsi_hat) * d_hat
    d_bsup_theta_dzeta = float(iota) * 2.0 * b_hat * db_hat_dzeta / denom

    non_stel_sym = bool(np.any(~np.asarray(surf_old.parity)) or np.any(~np.asarray(surf_new.parity)))
    b_sub_psi, db_sub_psi_dtheta, db_sub_psi_dzeta = _u_and_bsubpsi(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b_hat=b_hat,
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        p_prime_hat=float(p_prime_hat),
        non_stel_sym=non_stel_sym,
    )

    gpsipsi = None
    grad_psi_dot_grad_b = None
    if compute_gpsipsi or compute_grad_psi_dot_grad_b:
        gpsipsi_np, grad_psi_dot_grad_b_np = _boozer_psi_metrics(
            header=header,
            surf_old=surf_old,
            surf_new=surf_new,
            theta=theta_np,
            zeta=zeta_np,
            b_hat=b_hat,
            d_hat=d_hat,
            iota=float(iota),
            p_prime_hat=float(p_prime_hat),
            radial_weight=radial_weight,
            curvature=compute_grad_psi_dot_grad_b,
        )
        if compute_gpsipsi:
            gpsipsi = jnp.asarray(gpsipsi_np)
        if compute_grad_psi_dot_grad_b:
            grad_psi_dot_grad_b = jnp.asarray(grad_psi_dot_grad_b_np)

    return _geometry_from_boozer_bhat(
        n_periods=int(header.n_periods),
        b0_over_bbar=float(b0_over_bbar),
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        b_hat=jnp.asarray(b_hat),
        db_hat_dtheta=jnp.asarray(db_hat_dtheta),
        db_hat_dzeta=jnp.asarray(db_hat_dzeta),
        b_hat_sub_psi=jnp.asarray(b_sub_psi),
        db_hat_dpsi_hat=jnp.asarray(db_hat_dpsi_hat),
        db_hat_sub_psi_dtheta=jnp.asarray(db_sub_psi_dtheta),
        db_hat_sub_psi_dzeta=jnp.asarray(db_sub_psi_dzeta),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(db_sub_theta_dpsi),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(db_sub_zeta_dpsi),
        db_hat_sup_theta_dpsi_hat=jnp.asarray(d_bsup_theta_dpsi),
        db_hat_sup_theta_dzeta=jnp.asarray(d_bsup_theta_dzeta),
        db_hat_sup_zeta_dpsi_hat=jnp.asarray(d_bsup_zeta_dpsi),
        db_hat_sup_zeta_dtheta=jnp.asarray(d_bsup_zeta_dtheta),
        gpsipsi=gpsipsi,
        p_prime_hat=float(p_prime_hat),
        diota_dpsi_hat=float(diotadpsi_hat),
        grad_psi_dot_grad_b_over_gpsipsi=grad_psi_dot_grad_b,
    )


def _boozer_rzd_series(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    surf: BoozerBcSurface,
    n_signed: np.ndarray,
    dz_scale: float,
    chunk: int = 256,
) -> tuple[np.ndarray, ...]:
    """Evaluate R, Z, and the cylindrical-angle difference Dz with derivatives.

    Used by the ``gpsipsi`` metric and the Sugama normal-curvature factor
    (geometry.F90 ``gradpsidotgradB_overgpsipsi``).  Returns the 17-tuple
    ``(R, dR/dtheta, dR/dzeta, dZ/dtheta, dZ/dzeta, Dz, dDz/dtheta, dDz/dzeta,
    d2R/dtheta2, d2R/dthetadzeta, d2R/dzeta2,
    d2Z/dtheta2, d2Z/dthetadzeta, d2Z/dzeta2,
    d2Dz/dtheta2, d2Dz/dthetadzeta, d2Dz/dzeta2)``.
    """
    theta1 = theta[None, :, None]
    zeta1 = zeta[None, None, :]
    include = np.asarray(
        _grid_mode_mask(
            n_theta=int(theta.shape[0]), n_zeta=int(zeta.shape[0]), m=surf.m, n=n_signed, parity=surf.parity
        )
    )
    m = surf.m[include].astype(np.float64)
    n = n_signed[include].astype(np.float64)
    parity = surf.parity[include].astype(bool)
    r_amp = surf.r_amp[include].astype(np.float64)
    z_amp = surf.z_amp[include].astype(np.float64)
    dz_amp = surf.dz_amp[include].astype(np.float64) * float(dz_scale)

    r = np.full((theta.shape[0], zeta.shape[0]), float(surf.r0), dtype=np.float64)
    dr_dt = np.zeros_like(r)
    dr_dz = np.zeros_like(r)
    z_dt = np.zeros_like(r)
    z_dz = np.zeros_like(r)
    dz_field = np.zeros_like(r)
    ddz_dt = np.zeros_like(r)
    ddz_dz = np.zeros_like(r)
    d2r_dt2 = np.zeros_like(r)
    d2r_dtz = np.zeros_like(r)
    d2r_dz2 = np.zeros_like(r)
    d2z_dt2 = np.zeros_like(r)
    d2z_dtz = np.zeros_like(r)
    d2z_dz2 = np.zeros_like(r)
    d2dz_dt2 = np.zeros_like(r)
    d2dz_dtz = np.zeros_like(r)
    d2dz_dz2 = np.zeros_like(r)

    for i0 in range(0, int(m.shape[0]), chunk):
        i1 = min(int(m.shape[0]), i0 + chunk)
        mc = m[i0:i1][:, None, None]
        nc = n[i0:i1][:, None, None]
        rc = r_amp[i0:i1][:, None, None]
        zc = z_amp[i0:i1][:, None, None]
        dzc = dz_amp[i0:i1][:, None, None]
        pc = parity[i0:i1][:, None, None]
        angle = mc * theta1 - float(n_periods) * nc * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dzeta_factor = float(n_periods) * nc

        # R has cosine parity where B does; Z and Dz have the opposite parity.
        basis_r = np.where(pc, cos_a, sin_a)
        r = r + np.sum(rc * basis_r, axis=0)
        dr_dt = dr_dt + np.sum(rc * np.where(pc, -mc * sin_a, mc * cos_a), axis=0)
        dr_dz = dr_dz + np.sum(rc * np.where(pc, dzeta_factor * sin_a, -dzeta_factor * cos_a), axis=0)
        # Second derivatives of A*cos/sin(m theta - Nn zeta) are the original
        # basis times -m^2 (theta^2), +m*Nn (theta zeta), -Nn^2 (zeta^2).
        d2r_dt2 = d2r_dt2 + np.sum(-mc * mc * rc * basis_r, axis=0)
        d2r_dtz = d2r_dtz + np.sum(mc * dzeta_factor * rc * basis_r, axis=0)
        d2r_dz2 = d2r_dz2 + np.sum(-dzeta_factor * dzeta_factor * rc * basis_r, axis=0)

        basis_z = np.where(pc, sin_a, cos_a)
        dtheta_basis_z = np.where(pc, mc * cos_a, -mc * sin_a)
        dzeta_basis_z = np.where(pc, -dzeta_factor * cos_a, dzeta_factor * sin_a)
        z_dt = z_dt + np.sum(zc * dtheta_basis_z, axis=0)
        z_dz = z_dz + np.sum(zc * dzeta_basis_z, axis=0)
        d2z_dt2 = d2z_dt2 + np.sum(-mc * mc * zc * basis_z, axis=0)
        d2z_dtz = d2z_dtz + np.sum(mc * dzeta_factor * zc * basis_z, axis=0)
        d2z_dz2 = d2z_dz2 + np.sum(-dzeta_factor * dzeta_factor * zc * basis_z, axis=0)
        dz_field = dz_field + np.sum(dzc * basis_z, axis=0)
        ddz_dt = ddz_dt + np.sum(dzc * dtheta_basis_z, axis=0)
        ddz_dz = ddz_dz + np.sum(dzc * dzeta_basis_z, axis=0)
        d2dz_dt2 = d2dz_dt2 + np.sum(-mc * mc * dzc * basis_z, axis=0)
        d2dz_dtz = d2dz_dtz + np.sum(mc * dzeta_factor * dzc * basis_z, axis=0)
        d2dz_dz2 = d2dz_dz2 + np.sum(-dzeta_factor * dzeta_factor * dzc * basis_z, axis=0)
    return (
        r, dr_dt, dr_dz, z_dt, z_dz, dz_field, ddz_dt, ddz_dz,
        d2r_dt2, d2r_dtz, d2r_dz2,
        d2z_dt2, d2z_dtz, d2z_dz2,
        d2dz_dt2, d2dz_dtz, d2dz_dz2,
    )  # fmt: skip


def _boozer_psi_metrics(
    *,
    header: BoozerBcHeader,
    surf_old: BoozerBcSurface,
    surf_new: BoozerBcSurface,
    theta: np.ndarray,
    zeta: np.ndarray,
    b_hat: np.ndarray,
    d_hat: np.ndarray,
    iota: float,
    p_prime_hat: float,
    radial_weight: float,
    curvature: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(gpsiHatpsiHat, gradpsidotgradB_overgpsipsi)`` from the .bc shape harmonics.

    The first element is ``|grad psiHat|^2`` (v3 metric branch); the second is the
    Sugama normal-curvature factor of geometry.F90 (the ``magneticDriftScheme`` 5/6
    drive), or ``None`` unless ``curvature`` is requested.  ``iota`` must be the
    final (sign-switched) rotational transform and ``d_hat`` the final Jacobian
    factor ``BHat^2/(GHat + iota IHat)``, exactly as geometry.F90 uses them.
    """
    dz_scale = float(2.0 * math.pi / float(header.n_periods)) * (-1.0)
    old = _boozer_rzd_series(
        theta=theta,
        zeta=zeta,
        n_periods=int(header.n_periods),
        surf=surf_old,
        n_signed=-np.asarray(surf_old.n, dtype=np.int32),
        dz_scale=dz_scale,
    )
    new = _boozer_rzd_series(
        theta=theta,
        zeta=zeta,
        n_periods=int(header.n_periods),
        surf=surf_new,
        n_signed=-np.asarray(surf_new.n, dtype=np.int32),
        dz_scale=dz_scale,
    )
    (
        r, dr_dt, dr_dz, dz_dt, dz_dz, dz_field, ddz_dt, ddz_dz,
        d2r_dt2, d2r_dtz, d2r_dz2,
        d2z_dt2, d2z_dtz, d2z_dz2,
        d2dz_dt2, d2dz_dtz, d2dz_dz2,
    ) = (o * radial_weight + h * (1.0 - radial_weight) for o, h in zip(old, new))  # fmt: skip

    geomang = dz_field - zeta[None, :]
    cosg = np.cos(geomang)
    sing = np.sin(geomang)
    dgeomang_dtheta = ddz_dt
    dgeomang_dzeta = ddz_dz - 1.0

    dx_dt = dr_dt * cosg - r * dgeomang_dtheta * sing
    dx_dz = dr_dz * cosg - r * dgeomang_dzeta * sing
    dy_dt = dr_dt * sing + r * dgeomang_dtheta * cosg
    dy_dz = dr_dz * sing + r * dgeomang_dzeta * cosg

    d_hat = np.asarray(d_hat, dtype=np.float64)
    gradpsi_x = d_hat * (dy_dt * dz_dz - dz_dt * dy_dz)
    gradpsi_y = d_hat * (dz_dt * dx_dz - dx_dt * dz_dz)
    gradpsi_z = d_hat * (dx_dt * dy_dz - dy_dt * dx_dz)
    gpsipsi = gradpsi_x * gradpsi_x + gradpsi_y * gradpsi_y + gradpsi_z * gradpsi_z
    if not curvature:
        return gpsipsi, None

    # Second derivatives of X = R cos(geomang), Y = R sin(geomang)
    # (geometry.F90 normal-curvature block, d2Xdtheta2 ... d2Ydzeta2):
    d2x_dt2 = (
        d2r_dt2 * cosg
        - 2.0 * dr_dt * dgeomang_dtheta * sing
        - r * d2dz_dt2 * sing
        - r * dgeomang_dtheta * dgeomang_dtheta * cosg
    )
    d2x_dtz = (
        d2r_dtz * cosg
        - (dr_dt * dgeomang_dzeta + dr_dz * dgeomang_dtheta) * sing
        - r * d2dz_dtz * sing
        - r * dgeomang_dtheta * dgeomang_dzeta * cosg
    )
    d2x_dz2 = (
        d2r_dz2 * cosg
        - 2.0 * dr_dz * dgeomang_dzeta * sing
        - r * d2dz_dz2 * sing
        - r * dgeomang_dzeta * dgeomang_dzeta * cosg
    )
    d2y_dt2 = (
        d2r_dt2 * sing
        + 2.0 * dr_dt * dgeomang_dtheta * cosg
        + r * d2dz_dt2 * cosg
        - r * dgeomang_dtheta * dgeomang_dtheta * sing
    )
    d2y_dtz = (
        d2r_dtz * sing
        + (dr_dt * dgeomang_dzeta + dr_dz * dgeomang_dtheta) * cosg
        + r * d2dz_dtz * cosg
        - r * dgeomang_dtheta * dgeomang_dzeta * sing
    )
    d2y_dz2 = (
        d2r_dz2 * sing
        + 2.0 * dr_dz * dgeomang_dzeta * cosg
        + r * d2dz_dz2 * cosg
        - r * dgeomang_dzeta * dgeomang_dzeta * sing
    )

    # C = (d2/dzeta2 + 2 iota d2/dthetadzeta + iota^2 d2/dtheta2) * DHat^2 per
    # Cartesian component (the field-line-following second derivative):
    iota_f = float(iota)
    d_hat2 = d_hat * d_hat
    cx = (d2x_dz2 + 2.0 * iota_f * d2x_dtz + iota_f * iota_f * d2x_dt2) * d_hat2
    cy = (d2y_dz2 + 2.0 * iota_f * d2y_dtz + iota_f * iota_f * d2y_dt2) * d_hat2
    cz = (d2z_dz2 + 2.0 * iota_f * d2z_dtz + iota_f * iota_f * d2z_dt2) * d_hat2

    b_hat = np.asarray(b_hat, dtype=np.float64)
    grad_psi_dot_grad_b = (cx * gradpsi_x + cy * gradpsi_y + cz * gradpsi_z) / (
        b_hat * gpsipsi
    ) - float(p_prime_hat) / b_hat
    return gpsipsi, grad_psi_dot_grad_b


# =============================================================================
# VMEC wout files (geometryScheme 5): reader and radial interpolation
# =============================================================================


@dataclass(frozen=True)
class VmecWout:
    """VMEC ``wout`` data normalized to the internal scheme-5 layout.

    Fourier tables use the ``(mode, radius)`` convention (netCDF files store
    ``(radius, mode)``; the reader transposes).  Half-mesh arrays keep VMEC's
    dummy element at radius index 0, matching v3's indexing rules.
    """

    path: Path
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnmax_nyq: int
    lasym: bool
    aminor_p: float
    phi: np.ndarray  # (ns,)
    xm: np.ndarray  # (mnmax,)
    xn: np.ndarray  # (mnmax,)
    xm_nyq: np.ndarray  # (mnmax_nyq,)
    xn_nyq: np.ndarray  # (mnmax_nyq,) = n * nfp
    bmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    gmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubumnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubvmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsubsmns: np.ndarray  # (mnmax_nyq, ns) full mesh
    bsupumnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    bsupvmnc: np.ndarray  # (mnmax_nyq, ns) half mesh
    rmnc: np.ndarray  # (mnmax, ns) full mesh
    zmns: np.ndarray  # (mnmax, ns) full mesh
    lmns: np.ndarray  # (mnmax, ns) half mesh
    iotas: np.ndarray  # (ns,) half mesh
    presf: np.ndarray  # (ns,) full mesh
    # Stellarator-asymmetric (lasym=T) complementary-parity tables.  ``None`` for
    # stellarator-symmetric equilibria, so the stell-sym build is unaffected.  The
    # parity of each is flipped relative to its symmetric counterpart above: the
    # cosine tables gain sine partners (bmns/gmns/bsub{u,v}mns/bsup{u,v}mns/rmns),
    # and the sine tables gain cosine partners (bsubsmnc, zmnc).
    bmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    gmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubumns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubvmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsubsmnc: np.ndarray | None = None  # (mnmax_nyq, ns) full mesh
    bsupumns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    bsupvmns: np.ndarray | None = None  # (mnmax_nyq, ns) half mesh
    rmns: np.ndarray | None = None  # (mnmax, ns) full mesh
    zmnc: np.ndarray | None = None  # (mnmax, ns) full mesh
    lmnc: np.ndarray | None = None  # (mnmax, ns) half mesh


def read_vmec_wout(path: str | Path) -> VmecWout:
    """Read a VMEC ``wout_*.nc`` file.

    Pure function returning plain NumPy arrays; not differentiable.  The
    returned tables feed :meth:`FluxSurfaceGeometry.from_vmec`.  For
    stellarator-asymmetric equilibria (``lasym=T``) the complementary-parity
    tables (``bmns``, ``gmns``, ``bsub{u,v}mns``, ``bsubsmnc``, ``bsup{u,v}mns``,
    ``rmns``, ``zmnc``, ``lmnc``) are also loaded; they stay ``None`` for
    stellarator-symmetric files so that path is unchanged.
    """
    from scipy.io import netcdf_file  # noqa: PLC0415 - keep scipy import lazy

    p = Path(path).expanduser().resolve()
    if not p.exists():
        p = resolve_existing_path(path).path.resolve()

    with netcdf_file(p, "r", mmap=False) as f:
        def var(name: str) -> np.ndarray:
            if name not in f.variables:
                raise KeyError(f"Missing variable {name!r} in wout file.")
            return np.array(f.variables[name].data)

        def opt_var(name: str) -> np.ndarray | None:
            if name not in f.variables:
                return None
            return np.array(f.variables[name].data).astype(np.float64).T

        nfp = int(var("nfp"))
        ns = int(var("ns"))
        lasym = bool(int(np.asarray(var("lasym__logical__")).reshape(())))
        out = VmecWout(
            path=p,
            nfp=nfp,
            ns=ns,
            mpol=int(var("mpol")),
            ntor=int(var("ntor")),
            mnmax=int(var("mnmax")),
            mnmax_nyq=int(var("mnmax_nyq")),
            lasym=lasym,
            aminor_p=float(np.asarray(var("Aminor_p")).reshape(())),
            phi=var("phi").astype(np.float64),
            xm=var("xm").astype(np.int32),
            xn=var("xn").astype(np.int32),
            xm_nyq=var("xm_nyq").astype(np.int32),
            xn_nyq=var("xn_nyq").astype(np.int32),
            bmnc=var("bmnc").astype(np.float64).T,
            gmnc=var("gmnc").astype(np.float64).T,
            bsubumnc=var("bsubumnc").astype(np.float64).T,
            bsubvmnc=var("bsubvmnc").astype(np.float64).T,
            bsubsmns=var("bsubsmns").astype(np.float64).T,
            bsupumnc=var("bsupumnc").astype(np.float64).T,
            bsupvmnc=var("bsupvmnc").astype(np.float64).T,
            rmnc=var("rmnc").astype(np.float64).T,
            zmns=var("zmns").astype(np.float64).T,
            lmns=var("lmns").astype(np.float64).T,
            iotas=var("iotas").astype(np.float64),
            presf=var("presf").astype(np.float64),
            # Complementary-parity tables (present only when lasym=T).
            bmns=opt_var("bmns") if lasym else None,
            gmns=opt_var("gmns") if lasym else None,
            bsubumns=opt_var("bsubumns") if lasym else None,
            bsubvmns=opt_var("bsubvmns") if lasym else None,
            bsubsmnc=opt_var("bsubsmnc") if lasym else None,
            bsupumns=opt_var("bsupumns") if lasym else None,
            bsupvmns=opt_var("bsupvmns") if lasym else None,
            rmns=opt_var("rmns") if lasym else None,
            zmnc=opt_var("zmnc") if lasym else None,
            lmnc=opt_var("lmnc") if lasym else None,
        )
    if out.xm[0] != 0 or out.xn[0] != 0 or out.xm_nyq[0] != 0 or out.xn_nyq[0] != 0:
        raise ValueError("Expected the first VMEC mode to be (0,0).")
    return out


def psi_a_hat_from_wout(w: VmecWout) -> float:
    """``psiAHat = phi(ns) / (2 pi)`` as in v3."""
    return float(w.phi[-1]) / (2.0 * math.pi)


@dataclass(frozen=True)
class VmecRadialInterpolation:
    """Resolved full- and half-mesh interpolation state for one VMEC radius."""

    index_full: tuple[int, int]  # 0-based
    weight_full: tuple[float, float]
    index_half: tuple[int, int]  # 0-based; half arrays have a dummy 0 at index 0
    weight_half: tuple[float, float]
    psi_n: float
    psi_n_full: np.ndarray  # (ns,)
    psi_n_half: np.ndarray  # (ns-1,)


def vmec_radial_interpolation(*, w: VmecWout, psi_n_wish: float, vmec_radial_option: int) -> VmecRadialInterpolation:
    """v3's radius selection and interpolation index/weight logic (scheme 5)."""
    psi_n_full = np.asarray(w.phi, dtype=np.float64) / float(w.phi[-1])
    psi_n_half = 0.5 * (psi_n_full[:-1] + psi_n_full[1:])
    if not (0.0 <= float(psi_n_wish) <= 1.0):
        raise ValueError("psiN_wish must be in [0,1].")

    if int(vmec_radial_option) == 0:
        psi_n = float(psi_n_wish)
    elif int(vmec_radial_option) == 1:
        psi_n = float(psi_n_half[int(np.argmin((psi_n_half - float(psi_n_wish)) ** 2))])
    elif int(vmec_radial_option) == 2:
        psi_n = float(psi_n_full[int(np.argmin((psi_n_full - float(psi_n_wish)) ** 2))])
    else:
        raise ValueError(f"Invalid VMECRadialOption={vmec_radial_option}")

    ns = int(w.ns)
    if psi_n == 1.0:
        i0, i1, w0 = ns - 2, ns - 1, 0.0
    else:
        i0 = int(math.floor(psi_n * (ns - 1)))
        i1 = i0 + 1
        w0 = float(i0 + 1) - float(psi_n) * float(ns - 1)

    # Half-mesh: file arrays have length ns with a dummy 0 at index 0, so real
    # half indices start at python index 1 (Fortran index 2).
    if float(psi_n) < float(psi_n_half[0]):
        j0, j1 = 1, 2
        wh0 = (float(psi_n_half[1]) - float(psi_n)) / (float(psi_n_half[1]) - float(psi_n_half[0]))
    elif float(psi_n) > float(psi_n_half[-1]):
        j0, j1 = ns - 2, ns - 1
        wh0 = (float(psi_n_half[-1]) - float(psi_n)) / (float(psi_n_half[-1]) - float(psi_n_half[-2]))
    elif float(psi_n) == float(psi_n_half[-1]):
        j0, j1, wh0 = ns - 2, ns - 1, 0.0
    else:
        j_for = max(int(math.floor(float(psi_n) * float(ns - 1) + 0.5)) + 1, 2)  # v3, 1-based
        j0 = j_for - 1
        j1 = j0 + 1
        wh0 = float(j_for) - float(psi_n) * float(ns - 1) - 0.5

    return VmecRadialInterpolation(
        index_full=(i0, i1),
        weight_full=(w0, 1.0 - w0),
        index_half=(j0, j1),
        weight_half=(float(wh0), float(1.0 - wh0)),
        psi_n=float(psi_n),
        psi_n_full=psi_n_full,
        psi_n_half=psi_n_half,
    )


def _scale_factors(*, m: np.ndarray, n_over_nfp: np.ndarray, helicity_n: int, helicity_l: int, ripple_scale: float) -> np.ndarray:
    """v3 ``setScaleFactor``: keep the quasisymmetric family, scale the rest by rippleScale."""
    n = np.asarray(n_over_nfp, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    if int(helicity_n) == 0:
        scaled = n != 0
    else:
        scaled = ((n != 0) & (n * int(helicity_l) != m * int(helicity_n))) | (n == 0)
    return np.where(scaled, float(ripple_scale), 1.0)


def _finite_diff_full_from_half(arr_half: np.ndarray, dpsi: float) -> np.ndarray:
    """v3 finite-difference: half-mesh table -> radial derivative on the full mesh.

    For interior full points ``dQ/dpsiHat(j) = (Q(j+1) - Q(j)) / dpsi``;
    endpoints copy the adjacent interior value.
    """
    arr_half = np.asarray(arr_half, dtype=np.float64)
    n_mode, ns = arr_half.shape
    out = np.zeros((n_mode, ns), dtype=np.float64)
    out[:, 1 : ns - 1] = (arr_half[:, 2:ns] - arr_half[:, 1 : ns - 1]) / float(dpsi)
    if ns >= 3:
        out[:, 0] = out[:, 1]
        out[:, ns - 1] = out[:, ns - 2]
    return out


def _vmec_included_modes(
    *,
    w: VmecWout,
    interp: VmecRadialInterpolation,
    vmec_nyquist_option: int,
    min_bmn_to_load: float,
    ripple_scale: float,
    helicity_n: int,
    helicity_l: int,
    amplitude_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nyquist-mode selection shared by ``from_vmec`` and the gpsipsi metric.

    Returns ``(idx, scale_all)``: the included indices into the Nyquist mode
    tables and the per-mode rippleScale factors.  Mirrors v3's
    ``min_Bmn_to_load`` filter (applied after scaling) and the
    ``VMEC_Nyquist_option=1`` truncation to ``|m| < mpol``, ``|n| <= ntor``.

    ``amplitude_table`` selects which ``|B|`` Fourier table drives the
    amplitude filter: ``bmnc`` (default) for the stellarator-symmetric modes and
    ``bmns`` for the stellarator-asymmetric modes.  The ``(0,0)`` reference
    ``b00`` is always taken from ``bmnc`` (v3's ``geometry.F90``).
    """
    (j0, j1) = interp.index_half
    (wh0, wh1) = interp.weight_half
    b00 = float(w.bmnc[0, j0] * wh0 + w.bmnc[0, j1] * wh1)
    if b00 == 0.0:
        raise ValueError("VMEC bmnc(0,0) is zero; cannot apply min_Bmn_to_load filter.")
    xm = np.asarray(w.xm_nyq, dtype=np.float64)
    xn = np.asarray(w.xn_nyq, dtype=np.float64)
    n_over_nfp = np.round(xn / float(w.nfp)).astype(np.int64)
    scale_all = _scale_factors(
        m=np.round(xm).astype(np.int64),
        n_over_nfp=n_over_nfp,
        helicity_n=int(helicity_n),
        helicity_l=int(helicity_l),
        ripple_scale=float(ripple_scale),
    )
    table = w.bmnc if amplitude_table is None else amplitude_table
    b_mode = table[:, j0] * wh0 + table[:, j1] * wh1
    include = np.abs((b_mode * scale_all) / b00) >= float(min_bmn_to_load)

    option = int(vmec_nyquist_option)
    if option == 0:
        option = 1  # early-prototype compatibility: treat 0 as the v3 default
    if option not in {1, 2}:
        raise ValueError("VMEC_Nyquist_option must be 1 (skip Nyquist) or 2 (include Nyquist).")
    if option == 1:
        include = include & (np.abs(xm) < float(w.mpol)) & (np.abs(xn / float(w.nfp)) <= float(w.ntor))
    idx = np.nonzero(include)[0].astype(np.int32)
    if idx.size == 0:
        raise ValueError("No VMEC modes were included (min_Bmn_to_load too large?).")
    return idx, scale_all


def _from_vmec(
    cls,
    wout: str | Path | VmecWout,
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    psi_n_wish: float,
    vmec_radial_option: int = 0,
    vmec_nyquist_option: int = 1,
    min_bmn_to_load: float = 0.0,
    ripple_scale: float = 1.0,
    helicity_n: int = 0,
    helicity_l: int = 0,
    chunk: int = 256,
    compute_gpsipsi: bool = False,
    compute_grad_psi_dot_grad_b: bool = False,
) -> FluxSurfaceGeometry:
    """VMEC geometry (``geometryScheme = 5``), from a wout path or preloaded arrays.

    Direct translation of v3 ``geometry.F90::computeBHat_VMEC`` for
    stellarator-symmetric equilibria: half/full-mesh interpolation at the
    requested radius, cosine sums for ``BHat``/Jacobian/covariant/contravariant
    components, sine sums for ``BHat_sub_psi``, and the v3 finite-difference
    convention for radial derivatives.  As in v3, ``B0OverBBar``, ``GHat``,
    and ``IHat`` are left as 0.0 placeholders (they are flux-surface moments
    computed later, e.g. via :meth:`FluxSurfaceGeometry.fsab_hat2`).

    Accepting a preloaded :class:`VmecWout` keeps file I/O separate from
    geometry evaluation (the seam used by in-memory equilibrium producers).
    """
    w = wout if isinstance(wout, VmecWout) else read_vmec_wout(wout)
    psi_a_hat = psi_a_hat_from_wout(w)
    n_periods = int(w.nfp)
    interp = vmec_radial_interpolation(w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=int(vmec_radial_option))
    (i_full0, i_full1) = interp.index_full
    (w_full0, w_full1) = interp.weight_full
    (i_half0, i_half1) = interp.index_half
    (w_half0, w_half1) = interp.weight_half

    # VMEC spacing in psiHat: dpsi = phi(2)/(2 pi) = psiAHat/(ns-1).
    dpsi = float(w.phi[1]) / (2.0 * math.pi)

    theta_np = np.asarray(theta, dtype=np.float64)
    zeta_np = np.asarray(zeta, dtype=np.float64)
    theta1 = theta_np[None, :, None]
    zeta1 = zeta_np[None, None, :]

    idx, scale_all = _vmec_included_modes(
        w=w,
        interp=interp,
        vmec_nyquist_option=int(vmec_nyquist_option),
        min_bmn_to_load=float(min_bmn_to_load),
        ripple_scale=float(ripple_scale),
        helicity_n=int(helicity_n),
        helicity_l=int(helicity_l),
    )
    # Stellarator-asymmetric (lasym=T) modes carry a separate min_Bmn_to_load
    # filter driven by the |B| sine table (v3 geometry.F90 antisymmetric block).
    lasym = bool(w.lasym) and w.bmns is not None
    idx_asym = None
    if lasym:
        idx_asym, _ = _vmec_included_modes(
            w=w,
            interp=interp,
            vmec_nyquist_option=int(vmec_nyquist_option),
            min_bmn_to_load=float(min_bmn_to_load),
            ripple_scale=float(ripple_scale),
            helicity_n=int(helicity_n),
            helicity_l=int(helicity_l),
            amplitude_table=w.bmns,
        )
    xm = np.asarray(w.xm_nyq, dtype=np.float64)
    xn = np.asarray(w.xn_nyq, dtype=np.float64)

    shape = (int(theta_np.shape[0]), int(zeta_np.shape[0]))
    acc = {
        name: np.zeros(shape, dtype=np.float64)
        for name in (
            "b_hat",
            "d_hat",
            "db_dtheta",
            "db_dzeta",
            "db_dpsi",
            "b_sub_theta",
            "b_sub_zeta",
            "b_sub_psi",
            "db_sub_theta_dpsi",
            "db_sub_zeta_dpsi",
            "db_sub_theta_dzeta",
            "db_sub_zeta_dtheta",
            "db_sub_psi_dtheta",
            "db_sub_psi_dzeta",
            "b_sup_theta",
            "b_sup_zeta",
            "db_sup_theta_dzeta",
            "db_sup_zeta_dtheta",
        )
    }

    def half(table: np.ndarray, sel: np.ndarray) -> np.ndarray:
        return table[sel, i_half0] * w_half0 + table[sel, i_half1] * w_half1

    def full(table: np.ndarray, sel: np.ndarray) -> np.ndarray:
        return table[sel, i_full0] * w_full0 + table[sel, i_full1] * w_full1

    for i0 in range(0, int(idx.size), int(chunk)):
        sel = idx[i0 : min(int(idx.size), i0 + int(chunk))]
        m = xm[sel][:, None, None]
        n_nyq = xn[sel][:, None, None]  # equals n * NPeriods
        scale = scale_all[sel][:, None, None]

        angle = m * theta1 - n_nyq * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        b = half(w.bmnc, sel)[:, None, None] * scale
        jac = half(w.gmnc, sel)[:, None, None] * scale / float(psi_a_hat)
        bsupu = half(w.bsupumnc, sel)[:, None, None] * scale
        bsupv = half(w.bsupvmnc, sel)[:, None, None] * scale
        bsubu = half(w.bsubumnc, sel)[:, None, None] * scale
        bsubv = half(w.bsubvmnc, sel)[:, None, None] * scale
        bsubs = full(w.bsubsmns, sel)[:, None, None] * scale / float(psi_a_hat)

        d_b_dpsi = full(_finite_diff_full_from_half(w.bmnc[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
        d_bsubu_dpsi = (
            full(_finite_diff_full_from_half(w.bsubumnc[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
        )
        d_bsubv_dpsi = (
            full(_finite_diff_full_from_half(w.bsubvmnc[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
        )

        acc["b_hat"] += np.sum(b * cos_a, axis=0)
        acc["db_dtheta"] += np.sum(-m * b * sin_a, axis=0)
        acc["db_dzeta"] += np.sum(n_nyq * b * sin_a, axis=0)
        acc["d_hat"] += np.sum(jac * cos_a, axis=0)
        acc["b_sup_theta"] += np.sum(bsupu * cos_a, axis=0)
        acc["db_sup_theta_dzeta"] += np.sum(n_nyq * bsupu * sin_a, axis=0)
        acc["b_sup_zeta"] += np.sum(bsupv * cos_a, axis=0)
        acc["db_sup_zeta_dtheta"] += np.sum(-m * bsupv * sin_a, axis=0)
        acc["b_sub_theta"] += np.sum(bsubu * cos_a, axis=0)
        acc["db_sub_theta_dzeta"] += np.sum(n_nyq * bsubu * sin_a, axis=0)
        acc["b_sub_zeta"] += np.sum(bsubv * cos_a, axis=0)
        acc["db_sub_zeta_dtheta"] += np.sum(-m * bsubv * sin_a, axis=0)
        acc["b_sub_psi"] += np.sum(bsubs * sin_a, axis=0)
        acc["db_sub_psi_dtheta"] += np.sum(m * bsubs * cos_a, axis=0)
        acc["db_sub_psi_dzeta"] += np.sum(-n_nyq * bsubs * cos_a, axis=0)
        acc["db_dpsi"] += np.sum(d_b_dpsi * cos_a, axis=0)
        acc["db_sub_theta_dpsi"] += np.sum(d_bsubu_dpsi * cos_a, axis=0)
        acc["db_sub_zeta_dpsi"] += np.sum(d_bsubv_dpsi * cos_a, axis=0)

    if lasym:
        # Stellarator-asymmetric terms (v3 geometry.F90 antisymmetric block):
        # the cosine tables above gain sine partners, and the sine-parity field
        # BHat_sub_psi (bsubsmns) gains a cosine partner (bsubsmnc).  Angle,
        # rippleScale, and mesh interpolation reuse the symmetric machinery.
        for i0 in range(0, int(idx_asym.size), int(chunk)):
            sel = idx_asym[i0 : min(int(idx_asym.size), i0 + int(chunk))]
            m = xm[sel][:, None, None]
            n_nyq = xn[sel][:, None, None]  # equals n * NPeriods
            scale = scale_all[sel][:, None, None]

            angle = m * theta1 - n_nyq * zeta1
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            b = half(w.bmns, sel)[:, None, None] * scale
            jac = half(w.gmns, sel)[:, None, None] * scale / float(psi_a_hat)
            bsupu = half(w.bsupumns, sel)[:, None, None] * scale
            bsupv = half(w.bsupvmns, sel)[:, None, None] * scale
            bsubu = half(w.bsubumns, sel)[:, None, None] * scale
            bsubv = half(w.bsubvmns, sel)[:, None, None] * scale
            bsubs = full(w.bsubsmnc, sel)[:, None, None] * scale / float(psi_a_hat)

            d_b_dpsi = full(_finite_diff_full_from_half(w.bmns[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
            d_bsubu_dpsi = (
                full(_finite_diff_full_from_half(w.bsubumns[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
            )
            d_bsubv_dpsi = (
                full(_finite_diff_full_from_half(w.bsubvmns[sel, :], dpsi), np.arange(sel.size))[:, None, None] * scale
            )

            acc["b_hat"] += np.sum(b * sin_a, axis=0)
            acc["db_dtheta"] += np.sum(m * b * cos_a, axis=0)
            acc["db_dzeta"] += np.sum(-n_nyq * b * cos_a, axis=0)
            acc["d_hat"] += np.sum(jac * sin_a, axis=0)
            acc["b_sup_theta"] += np.sum(bsupu * sin_a, axis=0)
            acc["db_sup_theta_dzeta"] += np.sum(-n_nyq * bsupu * cos_a, axis=0)
            acc["b_sup_zeta"] += np.sum(bsupv * sin_a, axis=0)
            acc["db_sup_zeta_dtheta"] += np.sum(m * bsupv * cos_a, axis=0)
            acc["b_sub_theta"] += np.sum(bsubu * sin_a, axis=0)
            acc["db_sub_theta_dzeta"] += np.sum(-n_nyq * bsubu * cos_a, axis=0)
            acc["b_sub_zeta"] += np.sum(bsubv * sin_a, axis=0)
            acc["db_sub_zeta_dtheta"] += np.sum(m * bsubv * cos_a, axis=0)
            acc["b_sub_psi"] += np.sum(bsubs * cos_a, axis=0)
            acc["db_sub_psi_dtheta"] += np.sum(-m * bsubs * sin_a, axis=0)
            acc["db_sub_psi_dzeta"] += np.sum(n_nyq * bsubs * sin_a, axis=0)
            acc["db_dpsi"] += np.sum(d_b_dpsi * sin_a, axis=0)
            acc["db_sub_theta_dpsi"] += np.sum(d_bsubu_dpsi * sin_a, axis=0)
            acc["db_sub_zeta_dpsi"] += np.sum(d_bsubv_dpsi * sin_a, axis=0)

    d_hat = 1.0 / acc["d_hat"]  # accumulated the Jacobian; DHat is its inverse
    iota = float(w.iotas[i_half0] * w_half0 + w.iotas[i_half1] * w_half1)

    # Radial flux-function derivatives (geometry.F90 computeBHat_VMEC):
    # diotadpsiHat from the bracketing half-mesh iotas, pPrimeHat = mu0 dp/dpsiHat
    # from the full-mesh pressure finite difference indexed on the half mesh.
    psi_n_half_of = np.concatenate(([0.0], interp.psi_n_half))  # dummy 0 as in the file tables
    dpsi_n_half = float(psi_n_half_of[i_half1] - psi_n_half_of[i_half0])
    diotadpsi_hat = 0.0
    if dpsi_n_half != 0.0:
        diotadpsi_hat = float(w.iotas[i_half1] - w.iotas[i_half0]) / dpsi_n_half / float(psi_a_hat)
    presf = np.asarray(w.presf, dtype=np.float64)
    dp_dpsi_hat = np.zeros_like(presf)
    dp_dpsi_hat[1:] = (presf[1:] - presf[:-1]) / float(dpsi)
    p_prime_hat = float(_MU0 * (dp_dpsi_hat[i_half0] * w_half0 + dp_dpsi_hat[i_half1] * w_half1))

    gpsipsi = None
    grad_psi_dot_grad_b = None
    if compute_gpsipsi or compute_grad_psi_dot_grad_b:
        gpsipsi_np, (g_tt, g_tz, g_zz, g_pt, g_pz) = _gpsipsi_vmec(
            w=w,
            interp=interp,
            idx=idx,
            idx_asym=idx_asym,
            theta=theta_np,
            zeta=zeta_np,
            dpsi=dpsi,
            ripple_scale=float(ripple_scale),
            helicity_n=int(helicity_n),
            helicity_l=int(helicity_l),
            chunk=int(chunk),
        )
        if compute_gpsipsi:
            gpsipsi = jnp.asarray(gpsipsi_np)
        if compute_grad_psi_dot_grad_b:
            # geometry.F90 computeBHat_VMEC: gradpsidotgradB_overgpsipsi from the
            # covariant metric and the BHat derivatives.
            metric_denom = g_tt * g_zz - g_tz * g_tz
            grad_psi_dot_grad_b = jnp.asarray(
                acc["db_dpsi"]
                + (
                    (g_tz * g_pz - g_pt * g_zz) * acc["db_dtheta"]
                    + (g_pt * g_tz - g_tt * g_pz) * acc["db_dzeta"]
                )
                / metric_denom
            )

    zeros = np.zeros(shape, dtype=np.float64)
    return FluxSurfaceGeometry(
        n_periods=n_periods,
        b0_over_bbar=0.0,
        iota=iota,
        g_hat=0.0,
        i_hat=0.0,
        b_hat=jnp.asarray(acc["b_hat"]),
        db_hat_dtheta=jnp.asarray(acc["db_dtheta"]),
        db_hat_dzeta=jnp.asarray(acc["db_dzeta"]),
        d_hat=jnp.asarray(d_hat),
        b_hat_sup_theta=jnp.asarray(acc["b_sup_theta"]),
        b_hat_sup_zeta=jnp.asarray(acc["b_sup_zeta"]),
        b_hat_sub_theta=jnp.asarray(acc["b_sub_theta"]),
        b_hat_sub_zeta=jnp.asarray(acc["b_sub_zeta"]),
        b_hat_sub_psi=jnp.asarray(acc["b_sub_psi"]),
        db_hat_dpsi_hat=jnp.asarray(acc["db_dpsi"]),
        db_hat_sub_psi_dtheta=jnp.asarray(acc["db_sub_psi_dtheta"]),
        db_hat_sub_psi_dzeta=jnp.asarray(acc["db_sub_psi_dzeta"]),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(acc["db_sub_theta_dpsi"]),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(acc["db_sub_zeta_dpsi"]),
        db_hat_sub_theta_dzeta=jnp.asarray(acc["db_sub_theta_dzeta"]),
        db_hat_sub_zeta_dtheta=jnp.asarray(acc["db_sub_zeta_dtheta"]),
        db_hat_sup_theta_dpsi_hat=jnp.asarray(zeros),  # v3 sets these to 0 for VMEC
        db_hat_sup_theta_dzeta=jnp.asarray(acc["db_sup_theta_dzeta"]),
        db_hat_sup_zeta_dpsi_hat=jnp.asarray(zeros),
        db_hat_sup_zeta_dtheta=jnp.asarray(acc["db_sup_zeta_dtheta"]),
        gpsipsi=gpsipsi,
        p_prime_hat=p_prime_hat,
        diota_dpsi_hat=diotadpsi_hat,
        grad_psi_dot_grad_b_over_gpsipsi=grad_psi_dot_grad_b,
    )


def _gpsipsi_vmec(
    *,
    w: VmecWout,
    interp: VmecRadialInterpolation,
    idx: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
    dpsi: float,
    ripple_scale: float,
    helicity_n: int,
    helicity_l: int,
    chunk: int,
    idx_asym: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """``gpsiHatpsiHat`` for VMEC input, from the R/Z shape tables (v3 metric branch).

    Returns ``(gpsipsi, (g_tt, g_tz, g_zz, g_pt, g_pz))`` — the metric components
    are what geometry.F90 combines with the ``BHat`` derivatives to form the
    Sugama normal-curvature factor ``gradpsidotgradB_overgpsipsi``.

    For stellarator-asymmetric equilibria, ``idx_asym`` supplies the antisymmetric
    mode set whose ``rmns``/``zmnc`` partners are added to R/Z and their
    derivatives (v3 ``geometry.F90`` antisymmetric block)."""
    (i_full0, i_full1) = interp.index_full
    (w_full0, w_full1) = interp.weight_full
    (i_half0, i_half1) = interp.index_half
    (w_half0, w_half1) = interp.weight_half

    theta1 = theta[None, :, None]
    zeta1 = zeta[None, None, :]
    mode_to_index = {(int(w.xm[k]), int(w.xn[k])): int(k) for k in range(int(w.xm.shape[0]))}

    rmnc = np.asarray(w.rmnc, dtype=np.float64)
    zmns = np.asarray(w.zmns, dtype=np.float64)
    d_rmnc_dpsi = np.zeros_like(rmnc)
    d_zmns_dpsi = np.zeros_like(zmns)
    d_rmnc_dpsi[:, 1:] = (rmnc[:, 1:] - rmnc[:, :-1]) / float(dpsi)
    d_zmns_dpsi[:, 1:] = (zmns[:, 1:] - zmns[:, :-1]) / float(dpsi)

    shape = (int(theta.shape[0]), int(zeta.shape[0]))
    r = np.zeros(shape, dtype=np.float64)
    dr_dt = np.zeros_like(r)
    dr_dz = np.zeros_like(r)
    dr_dp = np.zeros_like(r)
    dz_dt = np.zeros_like(r)
    dz_dz = np.zeros_like(r)
    dz_dp = np.zeros_like(r)

    for i0 in range(0, int(idx.size), chunk):
        sel_nyq = idx[i0 : min(int(idx.size), i0 + chunk)]
        non_sel = np.array(
            [mode_to_index.get((int(w.xm_nyq[k]), int(w.xn_nyq[k])), -1) for k in sel_nyq.tolist()], dtype=np.int32
        )
        non_sel = non_sel[non_sel >= 0]
        if non_sel.size == 0:
            continue
        m = np.asarray(w.xm[non_sel], dtype=np.float64)[:, None, None]
        n_nyq = np.asarray(w.xn[non_sel], dtype=np.float64)[:, None, None]
        scale = _scale_factors(
            m=np.asarray(w.xm[non_sel], dtype=np.int64),
            n_over_nfp=np.round(np.asarray(w.xn[non_sel], dtype=np.float64) / float(w.nfp)).astype(np.int64),
            helicity_n=int(helicity_n),
            helicity_l=int(helicity_l),
            ripple_scale=float(ripple_scale),
        )[:, None, None]

        angle = m * theta1 - n_nyq * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        r_coef = (rmnc[non_sel, i_full0] * w_full0 + rmnc[non_sel, i_full1] * w_full1)[:, None, None] * scale
        z_coef = (zmns[non_sel, i_full0] * w_full0 + zmns[non_sel, i_full1] * w_full1)[:, None, None] * scale
        dr_dp_coef = (d_rmnc_dpsi[non_sel, i_half0] * w_half0 + d_rmnc_dpsi[non_sel, i_half1] * w_half1)[
            :, None, None
        ] * scale
        dz_dp_coef = (d_zmns_dpsi[non_sel, i_half0] * w_half0 + d_zmns_dpsi[non_sel, i_half1] * w_half1)[
            :, None, None
        ] * scale

        r += np.sum(r_coef * cos_a, axis=0)
        dr_dt += np.sum(-m * r_coef * sin_a, axis=0)
        dr_dz += np.sum(n_nyq * r_coef * sin_a, axis=0)
        dr_dp += np.sum(dr_dp_coef * cos_a, axis=0)
        dz_dt += np.sum(m * z_coef * cos_a, axis=0)
        dz_dz += np.sum(-n_nyq * z_coef * cos_a, axis=0)
        dz_dp += np.sum(dz_dp_coef * sin_a, axis=0)

    if idx_asym is not None and w.rmns is not None:
        # Stellarator-asymmetric shape: R gains rmns*sin, Z gains zmnc*cos.
        rmns = np.asarray(w.rmns, dtype=np.float64)
        zmnc = np.asarray(w.zmnc, dtype=np.float64)
        d_rmns_dpsi = np.zeros_like(rmns)
        d_zmnc_dpsi = np.zeros_like(zmnc)
        d_rmns_dpsi[:, 1:] = (rmns[:, 1:] - rmns[:, :-1]) / float(dpsi)
        d_zmnc_dpsi[:, 1:] = (zmnc[:, 1:] - zmnc[:, :-1]) / float(dpsi)
        for i0 in range(0, int(idx_asym.size), chunk):
            sel_nyq = idx_asym[i0 : min(int(idx_asym.size), i0 + chunk)]
            non_sel = np.array(
                [mode_to_index.get((int(w.xm_nyq[k]), int(w.xn_nyq[k])), -1) for k in sel_nyq.tolist()], dtype=np.int32
            )
            non_sel = non_sel[non_sel >= 0]
            if non_sel.size == 0:
                continue
            m = np.asarray(w.xm[non_sel], dtype=np.float64)[:, None, None]
            n_nyq = np.asarray(w.xn[non_sel], dtype=np.float64)[:, None, None]
            scale = _scale_factors(
                m=np.asarray(w.xm[non_sel], dtype=np.int64),
                n_over_nfp=np.round(np.asarray(w.xn[non_sel], dtype=np.float64) / float(w.nfp)).astype(np.int64),
                helicity_n=int(helicity_n),
                helicity_l=int(helicity_l),
                ripple_scale=float(ripple_scale),
            )[:, None, None]

            angle = m * theta1 - n_nyq * zeta1
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            r_coef = (rmns[non_sel, i_full0] * w_full0 + rmns[non_sel, i_full1] * w_full1)[:, None, None] * scale
            z_coef = (zmnc[non_sel, i_full0] * w_full0 + zmnc[non_sel, i_full1] * w_full1)[:, None, None] * scale
            dr_dp_coef = (d_rmns_dpsi[non_sel, i_half0] * w_half0 + d_rmns_dpsi[non_sel, i_half1] * w_half1)[
                :, None, None
            ] * scale
            dz_dp_coef = (d_zmnc_dpsi[non_sel, i_half0] * w_half0 + d_zmnc_dpsi[non_sel, i_half1] * w_half1)[
                :, None, None
            ] * scale

            r += np.sum(r_coef * sin_a, axis=0)
            dr_dt += np.sum(m * r_coef * cos_a, axis=0)
            dr_dz += np.sum(-n_nyq * r_coef * cos_a, axis=0)
            dr_dp += np.sum(dr_dp_coef * sin_a, axis=0)
            dz_dt += np.sum(-m * z_coef * sin_a, axis=0)
            dz_dz += np.sum(n_nyq * z_coef * sin_a, axis=0)
            dz_dp += np.sum(dz_dp_coef * cos_a, axis=0)

    cosz = np.cos(zeta)[None, :]
    sinz = np.sin(zeta)[None, :]
    dx_dt = dr_dt * cosz
    dx_dz = dr_dz * cosz - r * sinz
    dx_dp = dr_dp * cosz
    dy_dt = dr_dt * sinz
    dy_dz = dr_dz * sinz + r * cosz
    dy_dp = dr_dp * sinz

    g_tt = dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt
    g_tz = dx_dt * dx_dz + dy_dt * dy_dz + dz_dt * dz_dz
    g_zz = dx_dz * dx_dz + dy_dz * dy_dz + dz_dz * dz_dz
    g_pt = dx_dp * dx_dt + dy_dp * dy_dt + dz_dp * dz_dt
    g_pz = dx_dp * dx_dz + dy_dp * dy_dz + dz_dp * dz_dz
    g_pp = dx_dp * dx_dp + dy_dp * dy_dp + dz_dp * dz_dp

    denom = g_tt * g_zz - g_tz * g_tz
    gpsipsi = 1.0 / (g_pp + (g_pt * (g_tz * g_pz - g_pt * g_zz) + g_pz * (g_pt * g_tz - g_tt * g_pz)) / denom)
    return gpsipsi, (g_tt, g_tz, g_zz, g_pt, g_pz)


# Bind constructors (defined as module-level functions to keep the dataclass
# body focused on the data contract):
FluxSurfaceGeometry.from_scheme = classmethod(_from_scheme)
FluxSurfaceGeometry.from_fourier = classmethod(_from_fourier)
FluxSurfaceGeometry.from_boozer = classmethod(_from_boozer)
FluxSurfaceGeometry.from_vmec = classmethod(_from_vmec)
