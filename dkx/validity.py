"""Local-validity diagnostics for radially-local drift-kinetic transport.

A monoenergetic / radially-local neoclassical result -- the RHSMode=3
transport matrix, the (nuPrime, EStar) database, or the bounce-averaged
``1/nu`` effective-ripple surrogate -- is built on two orderings that this
module makes quantitative and returns as pass / marginal / fail flags:

1. **Radial locality (finite orbit width).**  The local ansatz drops the
   radial excursion of the guiding-centre orbit against the radial scale of
   the gradients.  It fails when the trapped-particle banana / orbit width
   ``w_b`` is no longer small compared with the shortest gradient scale
   length ``L`` of the profiles (density, temperature, or the radial
   electric field).  The controlling ratio is ``delta_FOW = w_b / L``
   (finite-orbit-width parameter; F.L. Hinton and R.D. Hazeltine, Rev. Mod.
   Phys. 48, 239 (1976); P. Helander and D.J. Sigmar, *Collisional Transport
   in Magnetized Plasmas*, CUP (2002)).

2. **Collisionality regime and the E x B resonance.**  Which asymptotic
   transport regime the surface is in -- Pfirsch-Schlueter, plateau,
   banana / ``1/nu``, ``sqrt(nu)``, or superbanana-plateau -- is fixed by the
   normalized collisionality ``nu_star`` and the E x B precession set by the
   radial electric field.  Two dimensionless ratios organize the
   long-mean-free-path corner:

   * ``k_ExB = omega_E / nu_eff`` -- the poloidal E x B precession frequency
     over the effective (de-trapping) collision frequency ``nu_eff = nu /
     eps``.  The ``1/nu -> sqrt(nu)`` boundary is ``k_ExB ~ 1``: below it
     collisions de-trap ripple-trapped particles before the electric field
     rotates them (``1/nu``); above it the E x B precession takes over and
     the radial transport crosses to the ``sqrt(nu)`` scaling
     (K.C. Shaing, Phys. Fluids 27, 1567 (1984); D.-I. Ho and R.M. Kulsrud,
     Phys. Fluids 30, 442 (1987)).  In the DKES-database variables this is the
     ``nu_star`` vs ``EStar`` crossing ``nu_star ~ eps_t EStar / x0``
     (C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)).
   * ``k_res = omega_E / omega_d`` -- the E x B precession over the magnetic
     (grad-B) precession drift.  The superbanana-plateau resonance is
     ``k_res ~ 1`` (the two precessions cancel for the resonant particles),
     where the transport develops a collisionless plateau independent of
     ``nu`` (K.C. Shaing, Phys. Fluids 27, 1567 (1984)).

Normalizations and units
========================

Everything is in SFINCS "Hat" units and reuses the flux-surface constants of
the monoenergetic database (:mod:`dkx.monoenergetic`) and the benchmark
conventions of Beidler et al. 2011.  For a species ``s`` at speed
``v = x vth_s`` the normalized thermal gyroradius (over ``RBar``) is

    rho_hat = Delta x sqrt(mHat_s THat_s) / (|Z_s| B0),

with ``Delta = mBar vBar/(e BBar RBar)`` the reference ``rho*`` and
``vth_s = sqrt(THat_s/mHat_s)``.  The trapped-particle banana / orbit width
is the poloidal gyroradius enhanced by the bounce geometry,

    w_b = rho_hat / (|iota| sqrt(eps_t))          (= rho q / sqrt(eps)),

``eps_t = r/R0`` the inverse aspect ratio and ``q = 1/iota`` the safety
factor (Helander and Sigmar 2002, ch. 7).  The collisionality-regime
boundaries follow the standard neoclassical ordering written in the database
``nu_star = R0 nu / (iota v)`` (so ``nu_star = eps_t^{3/2} nu_star_HS``):

    Pfirsch-Schlueter : nu_star > 1
    plateau           : eps_t^{3/2} < nu_star <= 1
    long-mfp          : nu_star <= eps_t^{3/2}   (banana / 1/nu / sqrt-nu / SBP)

The E x B and drift ratios expressed in the database variables are

    v_E   = |E_r/(v B0)| = |EStar| |iota| eps_t / x    (the db ``v_e``),
    k_ExB = v_E / (|iota| nu_star),
    k_res = v_E |GHat| / Delta                          (at the reference speed),

where ``k_ExB`` uses ``nu_eff = nu/eps`` with the trapping depth ``eps``
defaulting to ``eps_t`` (pass an effective ripple ``eps`` for ripple-trapped
particles; V.V. Nemov et al., Phys. Plasmas 6, 4622 (1999); J.L. Velasco et
al., J. Comput. Phys. 418, 109512 (2020)).

The scalar ratios (:func:`thermal_gyroradius_hat`,
:func:`banana_orbit_width_hat`, :func:`finite_orbit_width_parameter`,
:func:`exb_collision_ratio`, :func:`drift_resonance_ratio`) are pure ``jnp``
and differentiable; the regime classification and the pass/marginal/fail
flags are host-side reporting (Python branches on the computed ratios).  No
environment-variable routes: every option is an explicit argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = [
    "Regime",
    "RegimeThresholds",
    "ValidityFlag",
    "LocalValidityReport",
    "banana_orbit_width_hat",
    "classify_collisionality_regime",
    "drift_resonance_ratio",
    "exb_collision_ratio",
    "finite_orbit_width_parameter",
    "local_validity_report",
    "thermal_gyroradius_hat",
]


class Regime(str, Enum):
    """Neoclassical radial-transport regime of a flux surface.

    The three high-collisionality regimes are the standard tokamak ordering
    (Helander and Sigmar 2002); the long-mean-free-path corner splits by the
    E x B precession into the collisional ``BANANA``/``ONE_OVER_NU`` branch
    and the electric-field branches ``SQRT_NU``/``SUPERBANANA_PLATEAU``
    (Shaing 1984; Ho and Kulsrud 1987; Beidler et al. 2011).
    """

    PFIRSCH_SCHLUETER = "pfirsch-schlueter"
    PLATEAU = "plateau"
    BANANA = "banana"
    ONE_OVER_NU = "one-over-nu"
    SQRT_NU = "sqrt-nu"
    SUPERBANANA_PLATEAU = "superbanana-plateau"


class ValidityFlag(str, Enum):
    """Traffic-light verdict for a local-validity diagnostic."""

    PASS = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"


@dataclass(frozen=True)
class RegimeThresholds:
    """Dimensionless O(1) boundaries used by the classifier and the flags.

    The collisionality boundaries are exact neoclassical orderings; the E x B
    and finite-orbit-width thresholds carry an intentional marginal band (a
    factor between the ``*_marginal`` and ``*_fail`` values) because the
    physical crossovers are gradual (Beidler et al. 2011; Shaing 1984).

    Attributes:
        nu_star_ps: plateau / Pfirsch-Schlueter boundary in ``nu_star`` (1).
        nu_star_pb_factor: plateau / long-mfp boundary as
            ``nu_star_pb = nu_star_pb_factor * eps_t**1.5`` (1).
        k_exb_marginal: ``k_ExB`` above which the E x B precession is no longer
            negligible (the ``sqrt(nu)`` onset marginal band lower edge).
        k_exb_sqrt_nu: ``k_ExB`` above which the surface is classified
            ``sqrt(nu)`` / superbanana-plateau (the ``omega_E ~ nu_eff``
            crossing, 1).
        k_res_window: multiplicative half-width of the superbanana-plateau
            resonance window around ``k_res = 1`` (resonant when
            ``1/k_res_window <= k_res <= k_res_window``).
        eps_eff_qs: effective-ripple value below which the long-mfp regime is
            labelled ``banana`` (quasisymmetric / omnigeneous) rather than
            ``one-over-nu``.
        fow_pass: ``delta_FOW`` below which radial locality passes.
        fow_marginal: ``delta_FOW`` below which radial locality is marginal
            (fail above it).
    """

    nu_star_ps: float = 1.0
    nu_star_pb_factor: float = 1.0
    k_exb_marginal: float = 0.3
    k_exb_sqrt_nu: float = 1.0
    k_res_window: float = 3.0
    eps_eff_qs: float = 1.0e-3
    fow_pass: float = 0.1
    fow_marginal: float = 0.3


DEFAULT_THRESHOLDS = RegimeThresholds()


@dataclass(frozen=True)
class LocalValidityReport:
    """Structured local-validity verdict for one radially-local result.

    Attributes:
        regime: the :class:`Regime` the surface is in.
        nu_star: normalized collisionality ``R0 nu/(iota v)`` (positive).
        e_star: the DKES ``EStar`` of the point.
        v_e: normalized E x B drift ``|E_r/(v B0)|`` (the db ``v_e``).
        k_exb: ``omega_E/nu_eff`` (E x B precession over de-trapping rate).
        k_res: ``omega_E/omega_d`` (E x B over magnetic precession).
        orbit_width_hat: trapped-particle banana / orbit width ``w_b`` (RBar).
        grad_scale_length_hat: shortest gradient scale length ``L`` used (RBar).
        delta_fow: ``w_b / L`` finite-orbit-width parameter.
        eps_t: inverse aspect ratio of the surface.
        radial_locality_flag: pass/marginal/fail from ``delta_fow``.
        one_over_nu_surrogate_flag: whether the bounce-averaged ``1/nu``
            effective-ripple surrogate is trustworthy at this point.
        overall_flag: the worst of the individual flags.
        notes: short human-readable reasons behind the flags.
    """

    regime: Regime
    nu_star: float
    e_star: float
    v_e: float
    k_exb: float
    k_res: float
    orbit_width_hat: float
    grad_scale_length_hat: float
    delta_fow: float
    eps_t: float
    radial_locality_flag: ValidityFlag
    one_over_nu_surrogate_flag: ValidityFlag
    overall_flag: ValidityFlag
    notes: tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# Differentiable scalar ratios
# =============================================================================


def thermal_gyroradius_hat(
    *, delta: Any, z: Any, m_hat: Any, t_hat: Any, b0_over_bbar: Any, x: Any = 1.0
) -> jnp.ndarray:
    """Normalized gyroradius ``rho_hat = Delta x sqrt(mHat THat)/(|Z| B0)`` (RBar units).

    ``rho_hat`` is the guiding-centre Larmor radius over the reference length
    ``RBar`` at speed ``v = x sqrt(THat/mHat)`` (``x = x0 = 1`` gives the
    thermal value).  Pure ``jnp``; differentiable in every argument.
    """
    delta = jnp.asarray(delta, dtype=jnp.float64)
    z = jnp.abs(jnp.asarray(z, dtype=jnp.float64))
    m_hat = jnp.asarray(m_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    b0 = jnp.abs(jnp.asarray(b0_over_bbar, dtype=jnp.float64))
    x = jnp.asarray(x, dtype=jnp.float64)
    return delta * x * jnp.sqrt(m_hat * t_hat) / (z * b0)


def banana_orbit_width_hat(*, rho_hat: Any, iota: Any, eps_t: Any) -> jnp.ndarray:
    """Trapped-particle banana / orbit width ``w_b = rho_hat/(|iota| sqrt(eps_t))``.

    The radial excursion of a trapped bounce orbit is the poloidal gyroradius
    ``rho q`` enhanced by ``1/sqrt(eps)`` from the slow bounce motion
    (Helander and Sigmar 2002, ch. 7; ``q = 1/iota``).  Pure ``jnp``;
    differentiable.
    """
    rho_hat = jnp.asarray(rho_hat, dtype=jnp.float64)
    iota = jnp.abs(jnp.asarray(iota, dtype=jnp.float64))
    eps_t = jnp.asarray(eps_t, dtype=jnp.float64)
    return rho_hat / (iota * jnp.sqrt(eps_t))


def finite_orbit_width_parameter(*, orbit_width_hat: Any, grad_scale_length_hat: Any) -> jnp.ndarray:
    """Finite-orbit-width parameter ``delta_FOW = w_b / L``.

    The radially-local ansatz requires ``delta_FOW << 1``; when the orbit
    width approaches the gradient scale length the local result is
    invalidated by finite-orbit-width physics (Hinton and Hazeltine 1976).
    Pure ``jnp``; differentiable.
    """
    w_b = jnp.asarray(orbit_width_hat, dtype=jnp.float64)
    length = jnp.asarray(grad_scale_length_hat, dtype=jnp.float64)
    return w_b / length


def exb_collision_ratio(*, v_e: Any, iota: Any, nu_star: Any) -> jnp.ndarray:
    """E x B precession over de-trapping collisions ``k_ExB = v_E/(|iota| nu_star)``.

    Equals ``omega_E/nu_eff`` with ``omega_E = E_r/(r B0)`` the poloidal E x B
    precession and ``nu_eff = nu/eps_t`` the effective de-trapping frequency;
    the ``1/nu -> sqrt(nu)`` boundary is ``k_ExB ~ 1`` (Shaing 1984; Ho and
    Kulsrud 1987).  Pure ``jnp``; differentiable.
    """
    v_e = jnp.abs(jnp.asarray(v_e, dtype=jnp.float64))
    iota = jnp.abs(jnp.asarray(iota, dtype=jnp.float64))
    nu_star = jnp.abs(jnp.asarray(nu_star, dtype=jnp.float64))
    return v_e / (iota * nu_star)


def drift_resonance_ratio(*, v_e: Any, g_hat: Any, delta: Any) -> jnp.ndarray:
    """E x B over magnetic precession ``k_res = v_E |GHat|/Delta`` (reference speed).

    ``omega_E/omega_d`` for the reference-speed trapped particle; the
    superbanana-plateau resonance -- where the E x B and grad-B precessions
    cancel -- is ``k_res ~ 1`` (Shaing 1984).  Pure ``jnp``; differentiable.
    """
    v_e = jnp.abs(jnp.asarray(v_e, dtype=jnp.float64))
    g_hat = jnp.abs(jnp.asarray(g_hat, dtype=jnp.float64))
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return v_e * g_hat / delta


# =============================================================================
# Regime classification (host-side)
# =============================================================================


def classify_collisionality_regime(
    *,
    nu_star: float,
    k_exb: float = 0.0,
    k_res: float = 0.0,
    eps_t: float,
    epsilon_eff: float | None = None,
    thresholds: RegimeThresholds = DEFAULT_THRESHOLDS,
) -> Regime:
    """Classify the radial-transport regime from ``nu_star`` and the E x B ratios.

    High collisionality is the exact tokamak ordering in ``nu_star``; the
    long-mean-free-path corner (``nu_star <= eps_t**1.5``) splits on the E x B
    precession:

    * ``k_exb < k_exb_sqrt_nu`` (E x B negligible): ``banana`` when the field
      is (quasi)symmetric -- ``epsilon_eff`` given and below ``eps_eff_qs`` --
      else ``one-over-nu``;
    * otherwise ``superbanana-plateau`` when ``k_res`` sits inside the
      resonance window ``[1/k_res_window, k_res_window]`` (the drift
      resonance), else ``sqrt-nu``.

    Args:
        nu_star: normalized collisionality ``R0 nu/(iota v)`` (positive).
        k_exb: ``exb_collision_ratio`` (0 for ``E_r = 0``).
        k_res: ``drift_resonance_ratio`` (0 for ``E_r = 0``).
        eps_t: inverse aspect ratio of the surface.
        epsilon_eff: optional effective ripple; ``< eps_eff_qs`` selects the
            ``banana`` label over ``one-over-nu`` in the collisional branch.
        thresholds: the O(1) boundaries.
    """
    nu = abs(float(nu_star))
    eps = abs(float(eps_t))
    nu_pb = thresholds.nu_star_pb_factor * eps**1.5
    if nu > thresholds.nu_star_ps:
        return Regime.PFIRSCH_SCHLUETER
    if nu > nu_pb:
        return Regime.PLATEAU
    # Long mean free path.
    if abs(float(k_exb)) < thresholds.k_exb_sqrt_nu:
        quasisymmetric = epsilon_eff is not None and abs(float(epsilon_eff)) < thresholds.eps_eff_qs
        return Regime.BANANA if quasisymmetric else Regime.ONE_OVER_NU
    kr = abs(float(k_res))
    if kr > 0.0 and (1.0 / thresholds.k_res_window) <= kr <= thresholds.k_res_window:
        return Regime.SUPERBANANA_PLATEAU
    return Regime.SQRT_NU


def _fow_flag(delta_fow: float, thresholds: RegimeThresholds) -> ValidityFlag:
    value = abs(float(delta_fow))
    if value < thresholds.fow_pass:
        return ValidityFlag.PASS
    if value < thresholds.fow_marginal:
        return ValidityFlag.MARGINAL
    return ValidityFlag.FAIL


def _worst(flags: tuple[ValidityFlag, ...]) -> ValidityFlag:
    order = {ValidityFlag.PASS: 0, ValidityFlag.MARGINAL: 1, ValidityFlag.FAIL: 2}
    return max(flags, key=lambda flag: order[flag])


# =============================================================================
# The assembled report
# =============================================================================


def local_validity_report(
    *,
    nu_star: float,
    e_star: float,
    delta: float,
    g_hat: float,
    iota: float,
    b0_over_bbar: float,
    eps_t: float,
    grad_scale_length_hat: float,
    z: float = 1.0,
    m_hat: float = 1.0,
    t_hat: float = 1.0,
    x: float = 1.0,
    epsilon_eff: float | None = None,
    thresholds: RegimeThresholds = DEFAULT_THRESHOLDS,
) -> LocalValidityReport:
    """Assemble the full local-validity verdict for one (nu_star, EStar) point.

    Combines the finite-orbit-width parameter (radial locality) with the
    collisionality-regime classification and the E x B ratios into a
    :class:`LocalValidityReport`.  The two verdicts answer distinct questions:

    * ``radial_locality_flag`` -- is *any* radially-local model valid here
      (finite-orbit-width ordering ``w_b/L << 1``)?
    * ``one_over_nu_surrogate_flag`` -- is the bounce-averaged ``1/nu``
      effective-ripple surrogate (:mod:`dkx.bounce_averaged`) trustworthy?  It
      is the ``nu -> 0`` asymptote of ``D11`` at ``E_r = 0``, so it passes only
      in the ``1/nu`` (or ``banana``) regime with negligible E x B precession
      and small orbit width; it fails once the E x B precession switches the
      surface to ``sqrt(nu)`` / superbanana-plateau, or in the collisional
      (plateau / Pfirsch-Schlueter) regimes.

    Args:
        nu_star: normalized collisionality of the point (positive).
        e_star: the DKES ``EStar`` of the point (signed; only ``|EStar|``
            enters the E x B magnitudes).
        delta, g_hat, iota, b0_over_bbar, eps_t: flux-surface constants of the
            monoenergetic database (SFINCS hat units).
        grad_scale_length_hat: shortest profile gradient scale length ``L``
            (RBar units) -- e.g. ``min(L_n, L_T, L_Er)`` or, as a conservative
            upper bound, the minor radius ``rHat``.
        z, m_hat, t_hat, x: species charge / mass / temperature and speed
            factor of the orbit whose width is tested (v3 monoenergetic runs
            use ``1, 1, 1``; ``x = x0 = 1`` is the thermal reference).
        epsilon_eff: optional effective ripple (selects ``banana`` vs
            ``one-over-nu`` and is not otherwise used).
        thresholds: the O(1) boundaries.

    Returns:
        A :class:`LocalValidityReport`.
    """
    v_e = abs(float(e_star)) * abs(float(iota)) * abs(float(eps_t)) / abs(float(x))
    rho_hat = float(
        thermal_gyroradius_hat(delta=delta, z=z, m_hat=m_hat, t_hat=t_hat, b0_over_bbar=b0_over_bbar, x=x)
    )
    w_b = float(banana_orbit_width_hat(rho_hat=rho_hat, iota=iota, eps_t=eps_t))
    delta_fow = float(
        finite_orbit_width_parameter(orbit_width_hat=w_b, grad_scale_length_hat=grad_scale_length_hat)
    )
    k_exb = float(exb_collision_ratio(v_e=v_e, iota=iota, nu_star=nu_star))
    k_res = float(drift_resonance_ratio(v_e=v_e, g_hat=g_hat, delta=delta))

    regime = classify_collisionality_regime(
        nu_star=nu_star, k_exb=k_exb, k_res=k_res, eps_t=eps_t,
        epsilon_eff=epsilon_eff, thresholds=thresholds,
    )  # fmt: skip

    radial_locality_flag = _fow_flag(delta_fow, thresholds)

    notes: list[str] = []
    # 1/nu surrogate verdict.
    if regime in (Regime.ONE_OVER_NU, Regime.BANANA):
        if k_exb >= thresholds.k_exb_marginal:
            surrogate_flag = ValidityFlag.MARGINAL
            notes.append(
                f"E x B precession is turning on (k_ExB={k_exb:.2g}); the E_r=0 1/nu "
                "surrogate underestimates the sqrt(nu) crossover."
            )
        else:
            surrogate_flag = ValidityFlag.PASS
            notes.append(f"1/nu regime with negligible E x B precession (k_ExB={k_exb:.2g}).")
    elif regime in (Regime.SQRT_NU, Regime.SUPERBANANA_PLATEAU):
        surrogate_flag = ValidityFlag.FAIL
        notes.append(
            f"{regime.value} regime (k_ExB={k_exb:.2g}, k_res={k_res:.2g}); the E_r=0 "
            "1/nu surrogate does not describe the E x B-controlled transport."
        )
    else:  # plateau / Pfirsch-Schlueter
        surrogate_flag = ValidityFlag.FAIL
        notes.append(
            f"{regime.value} regime (nu_star={float(nu_star):.2g}); the 1/nu surrogate is a "
            "low-collisionality asymptote and does not apply."
        )

    if radial_locality_flag is ValidityFlag.PASS:
        notes.append(f"radially local (delta_FOW={delta_fow:.2g}).")
    else:
        notes.append(
            f"finite-orbit-width {radial_locality_flag.value} "
            f"(delta_FOW={delta_fow:.2g} = w_b/L, w_b={w_b:.2g}, L={float(grad_scale_length_hat):.2g})."
        )

    overall_flag = _worst((radial_locality_flag, surrogate_flag))

    return LocalValidityReport(
        regime=regime,
        nu_star=abs(float(nu_star)),
        e_star=float(e_star),
        v_e=float(v_e),
        k_exb=k_exb,
        k_res=k_res,
        orbit_width_hat=w_b,
        grad_scale_length_hat=float(grad_scale_length_hat),
        delta_fow=delta_fow,
        eps_t=abs(float(eps_t)),
        radial_locality_flag=radial_locality_flag,
        one_over_nu_surrogate_flag=surrogate_flag,
        overall_flag=overall_flag,
        notes=tuple(notes),
    )
