"""Classical impurity transport: algebraic fluxes, screening, charge-state scans.

Roadmap item 4 of ``plan_final.md``.  The *classical* (Pfirsch-Schlueter-independent)
radial particle and heat fluxes of a magnetized multi-species plasma are purely
local and algebraic: they follow from the Braginskii inter-species friction
moments evaluated with the leading-order Maxwellians and a single flux-surface
geometry number, with no kinetic solve.  This module packages that classical
piece for impurity studies -- building a bulk plasma plus a (trace or non-trace)
impurity, decomposing the impurity flux into its thermodynamic drives, exposing
the classical diffusion coefficient, and evaluating the ion-temperature
*screening* term -- and provides the differentiable, ``vmap``-over-charge-state
API used by the mixed-collisionality high-Z benchmark.

Physics and normalization
--------------------------
For species ``a`` the classical radial particle flux projected on ``grad psiHat``
is the sum over collision partners ``b`` of the Braginskii friction moments

    Gamma_a^cl = Z_a (Delta^2 nu_n) sqrt(m_a) / (2 T_a^{3/2})
                 * G * sum_b Z_b^2 n_a n_b * [ M^{ab}_{00} (u^a_n - u^b_n)
                       + (M^{ab}_{00} - M^{ab}_{01}) u^a_T
                       - (M^{ab}_{00} - x_{ab}^2 M^{ab}_{01}) u^b_T ]

with the drive combinations ``u^a_n = T_a n_a'/(n_a Z_a)`` and
``u^a_T = T_a'/Z_a`` (primes are ``d/dpsiHat``), the mass/temperature ratio
``x_{ab}^2 = m_a T_b/(m_b T_a)``, and the friction-matrix elements
``M^{ab}_{jk}`` of Braginskii (Rev. Plasma Phys. 1, 205 (1965)).  The heat flux
uses the ``M^{ab}_{01}``/``M^{ab}_{11}``/``N^{ab}_{11}`` moments and the
convective ``5/2 T_a Gamma_a`` piece.  **All geometry enters through the single
scalar** ``G = <|grad psiHat|^2 / BHat^2>`` (the flux-surface average), which is
what makes the classical flux "nearly free" -- see :func:`classical_geometry_factor`.

Every quantity is the dimensionless SFINCS v3 "Hat" quantity: ``THat = T/TBar``,
``nHat = n/nBar``, ``mHat = m/mBar``, ``Z`` in proton charges, gradients w.r.t.
``psiHat``; fluxes are normalized to ``nBar vBar RBar^2`` projected on
``grad psiHat``.  ``Delta = mBar vBar/(e BBar RBar)`` and ``nu_n`` are the v3
scalars.  Signs follow the v3 convention: ``psiHat`` increases outward, so a
positive flux is radially outward and an impurity *pinch* (accumulation) is a
negative particle flux.

Temperature screening
---------------------
Written with logarithmic gradients ``g_x = (1/x) dx/dpsiHat`` the trace-impurity
classical particle flux reduces to the textbook form (Rutherford, Phys. Fluids
17, 1782 (1974); Hinton & Hazeltine, Rev. Mod. Phys. 48, 239 (1976); Helander &
Sigmar, *Collisional Transport in Magnetized Plasmas*, CUP (2002), Sec. 11)

    Gamma_z/n_z = -D_z^cl [ g_{n_z} - Z_z g_{n_i} + H Z_z g_{T_i} + g_{T_z} ]

with the ion-density peaking coefficient ``-Z_z`` (impurity accumulation) and the
ion-temperature *screening* coefficient ``H`` that this module returns.  In the
collisional (classical/Pfirsch-Schlueter) regime and the heavy-impurity limit
``H -> 1/2`` (as opposed to ``3/2`` in the banana regime; Wenzel & Sigmar,
Nucl. Fusion 30, 1117 (1990)).  A peaked ion temperature (``T_i' < 0``) then
opposes the density pinch -- the screening that keeps high-Z impurities out of
the core.  :func:`temperature_screening_diagnostic` reports ``H``, the exact
``-Z_z`` peaking coefficient, and whether the ion-temperature contribution
opposes the pinch for the supplied gradients.

SFINCS Fortran counterpart
--------------------------
The multi-species algebra reproduces ``classicalTransport.F90:calculateClassicalFlux``
(the ``Phi1 = 0`` branch) exactly; :func:`classical_species_fluxes` and
:func:`dkx.moments.classical_fluxes` agree to machine precision (the former is
the geometry-collapsed form of the latter).  All functions are pure and
jit/vmap/grad friendly and take explicit arrays -- no namelist parsing, no IO.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from dkx.species import SpeciesSet  # noqa: E402

__all__ = [
    "classical_geometry_factor",
    "classical_species_fluxes",
    "build_impurity_plasma",
    "ImpurityClassicalFlux",
    "classical_impurity_flux",
    "classical_diffusion_coefficient",
    "TemperatureScreening",
    "temperature_screening_diagnostic",
    "classical_impurity_flux_over_charge_states",
]


# ---------------------------------------------------------------------------
# Geometry: the single flux-surface scalar the classical flux depends on
# ---------------------------------------------------------------------------


def classical_geometry_factor(
    *,
    theta_weights: jnp.ndarray,
    zeta_weights: jnp.ndarray,
    d_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    gpsipsi: jnp.ndarray,
) -> jnp.ndarray:
    """Flux-surface average ``G = <|grad psiHat|^2 / BHat^2>`` (a scalar).

    This is the only geometry quantity the classical flux needs.  It equals
    ``geom1`` of the ``Phi1 = 0`` branch of
    :func:`dkx.moments.classical_fluxes`,

        G = [sum_ij w_i w_j gpsiHatpsiHat_ij / (DHat_ij BHat_ij^2)]
            / [sum_ij w_i w_j / DHat_ij],

    with ``gpsiHatpsiHat`` the ``|grad psiHat|^2`` metric (v3
    ``geometry.F90``; nonzero for Boozer ``.bc`` and VMEC geometries,
    a placeholder zero for the analytic schemes 1/2/4/13).

    Args:
        theta_weights: ``(T,)`` poloidal quadrature weights.
        zeta_weights: ``(Z,)`` toroidal quadrature weights.
        d_hat: ``(T, Z)`` Jacobian factor ``DHat`` (``dV ~ dtheta dzeta / DHat``).
        b_hat: ``(T, Z)`` field strength ``BHat``.
        gpsipsi: ``(T, Z)`` metric ``gpsiHatpsiHat = |grad psiHat|^2``.

    Returns:
        The dimensionless scalar ``G``.
    """
    tw = jnp.asarray(theta_weights, dtype=jnp.float64)
    zw = jnp.asarray(zeta_weights, dtype=jnp.float64)
    d_hat = jnp.asarray(d_hat, dtype=jnp.float64)
    b_hat = jnp.asarray(b_hat, dtype=jnp.float64)
    gpsipsi = jnp.asarray(gpsipsi, dtype=jnp.float64)
    w = tw[:, None] * zw[None, :] / d_hat
    vprime = jnp.sum(w)
    return jnp.sum(w * gpsipsi / (b_hat * b_hat)) / vprime


# ---------------------------------------------------------------------------
# The classical multi-species flux, reduced to the geometry scalar G
# ---------------------------------------------------------------------------


def classical_species_fluxes(
    *,
    z: jnp.ndarray,
    m_hat: jnp.ndarray,
    n_hat: jnp.ndarray,
    t_hat: jnp.ndarray,
    dn_hat_dpsi_hat: jnp.ndarray,
    dt_hat_dpsi_hat: jnp.ndarray,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Classical particle and heat fluxes ``(Gamma_a, q_a)`` for every species.

    The geometry-collapsed form of ``classicalTransport.F90`` (``Phi1 = 0``):
    identical to :func:`dkx.moments.classical_fluxes` with ``use_phi1=False``
    but parameterized by the single scalar ``geometry_factor`` ``G`` (see
    :func:`classical_geometry_factor`) instead of the full ``(theta, zeta)``
    geometry arrays.  Pure, differentiable, and ``vmap``-friendly over the
    species leaves.

    Args:
        z, m_hat, n_hat, t_hat: ``(S,)`` charge / mass / density / temperature.
        dn_hat_dpsi_hat, dt_hat_dpsi_hat: ``(S,)`` ``d/dpsiHat`` gradients.
        geometry_factor: scalar ``G = <|grad psiHat|^2/BHat^2>``.
        delta: the v3 ``Delta`` scalar.
        nu_n: the v3 collisionality scalar ``nu_n``.

    Returns:
        ``(particle_flux, heat_flux)`` each ``(S,)`` (``psiHat`` projection).
        The heat flux is the total definition including the convective
        ``5/2 T_a Gamma_a`` term, matching v3.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    m_hat = jnp.asarray(m_hat, dtype=jnp.float64)
    n_hat = jnp.asarray(n_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    dn = jnp.asarray(dn_hat_dpsi_hat, dtype=jnp.float64)
    dt = jnp.asarray(dt_hat_dpsi_hat, dtype=jnp.float64)
    g = jnp.asarray(geometry_factor, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)

    # Braginskii mass/temperature ratios and friction-matrix elements M^{ab}.
    xab2 = (m_hat[:, None] * t_hat[None, :]) / (m_hat[None, :] * t_hat[:, None])
    m_ratio = m_hat[:, None] / m_hat[None, :]
    one_plus_x = 1.0 + xab2
    denom = one_plus_x ** 2.5
    mab00 = -((1.0 + m_ratio) * one_plus_x) / denom
    mab01 = -1.5 * (1.0 + m_ratio) / denom
    mab11 = -(13.0 + 16.0 * xab2 + 30.0 * (xab2 ** 2)) / 4.0 / denom
    nab11 = (27.0 * m_ratio) / 4.0 / denom

    geom1 = g * (n_hat[:, None] * n_hat[None, :])  # (S,S); Phi1=0 collapses to G

    u_dn = (t_hat * dn) / (n_hat * z)  # (S,)
    u_dt_over_z = dt / z  # (S,)
    term_dn = u_dn[:, None] - u_dn[None, :]

    pf_ab = geom1 * (
        mab00 * term_dn
        + (mab00 - mab01) * u_dt_over_z[:, None]
        - (mab00 - xab2 * mab01) * u_dt_over_z[None, :]
    )
    hf_ab = geom1 * (
        mab01 * term_dn
        + (mab01 - mab11) * u_dt_over_z[:, None]
        - (mab01 + nab11) * u_dt_over_z[None, :]
    )

    z2_b = z[None, :] ** 2
    pf_a = jnp.sum(z2_b * pf_ab, axis=1)
    hf_a = jnp.sum(z2_b * hf_ab, axis=1)

    pf_a = z * (delta ** 2) * nu_n * jnp.sqrt(m_hat) * pf_a / (2.0 * (t_hat ** 1.5))
    hf_a = -z * (delta ** 2) * nu_n * jnp.sqrt(m_hat) * hf_a / (4.0 * jnp.sqrt(t_hat))
    hf_a = hf_a + 1.25 * t_hat * pf_a
    return pf_a, hf_a


def _classical_species_fluxes_from_set(
    species: SpeciesSet,
    *,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return classical_species_fluxes(
        z=species.z,
        m_hat=species.m_hat,
        n_hat=species.n_hat,
        t_hat=species.t_hat,
        dn_hat_dpsi_hat=species.dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=species.dt_hat_dpsi_hat,
        geometry_factor=geometry_factor,
        delta=delta,
        nu_n=nu_n,
    )


# ---------------------------------------------------------------------------
# Building a bulk-plus-impurity deck
# ---------------------------------------------------------------------------


def build_impurity_plasma(
    bulk: SpeciesSet,
    *,
    impurity_z: float,
    impurity_m_hat: float,
    impurity_n_hat: float,
    impurity_t_hat: float | None = None,
    impurity_dn_hat_dpsi_hat: float | None = None,
    impurity_dt_hat_dpsi_hat: float | None = None,
    match_bulk_logarithmic_gradients: bool = False,
) -> SpeciesSet:
    """Append an impurity species to a bulk plasma, returning a combined set.

    The impurity is added as the *last* species (``impurity_index = -1``).  When
    a gradient argument is left ``None`` and ``match_bulk_logarithmic_gradients``
    is set, the impurity inherits the bulk species-0 logarithmic gradient
    (``(1/n) dn/dpsiHat`` and ``(1/T) dT/dpsiHat``) -- a common modeling default
    for a passively-transported impurity -- otherwise the missing gradient
    defaults to zero.  The impurity temperature defaults to the bulk species-0
    temperature (thermal equilibration).

    "Trace" versus "non-trace" is a physical regime, not a code switch: an
    impurity is trace when its charge-weighted density ``Z_z^2 n_z`` is
    negligible next to the bulk ``sum_i Z_i^2 n_i`` (so it neither perturbs the
    bulk nor screens itself).  Set ``impurity_n_hat`` accordingly; the strength
    parameter is reported by :func:`classical_impurity_flux` as
    ``impurity_strength``.

    Args:
        bulk: the bulk plasma (one or more species) as a :class:`SpeciesSet`.
        impurity_z, impurity_m_hat, impurity_n_hat: impurity charge, mass, density.
        impurity_t_hat: impurity temperature (default: bulk species-0 ``THat``).
        impurity_dn_hat_dpsi_hat, impurity_dt_hat_dpsi_hat: impurity gradients.
        match_bulk_logarithmic_gradients: inherit bulk-0 log-gradients when a
            gradient argument is ``None``.

    Returns:
        A :class:`SpeciesSet` with the impurity appended.
    """
    n0 = float(bulk.n_hat[0])
    t0 = float(bulk.t_hat[0])
    dn0 = float(bulk.dn_hat_dpsi_hat[0])
    dt0 = float(bulk.dt_hat_dpsi_hat[0])

    t_imp = t0 if impurity_t_hat is None else float(impurity_t_hat)

    if impurity_dn_hat_dpsi_hat is None:
        if match_bulk_logarithmic_gradients:
            dn_imp = (dn0 / n0) * float(impurity_n_hat)  # equal (1/n) dn/dpsi
        else:
            dn_imp = 0.0
    else:
        dn_imp = float(impurity_dn_hat_dpsi_hat)

    if impurity_dt_hat_dpsi_hat is None:
        if match_bulk_logarithmic_gradients:
            dt_imp = (dt0 / t0) * t_imp  # equal (1/T) dT/dpsi
        else:
            dt_imp = 0.0
    else:
        dt_imp = float(impurity_dt_hat_dpsi_hat)

    def _append(arr: jnp.ndarray, value: float) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(arr, dtype=jnp.float64), jnp.asarray([value], dtype=jnp.float64)])

    return SpeciesSet(
        z=_append(bulk.z, float(impurity_z)),
        m_hat=_append(bulk.m_hat, float(impurity_m_hat)),
        n_hat=_append(bulk.n_hat, float(impurity_n_hat)),
        t_hat=_append(bulk.t_hat, t_imp),
        dn_hat_dpsi_hat=_append(bulk.dn_hat_dpsi_hat, dn_imp),
        dt_hat_dpsi_hat=_append(bulk.dt_hat_dpsi_hat, dt_imp),
    )


# ---------------------------------------------------------------------------
# Impurity flux, decomposition, and classical diffusion coefficient
# ---------------------------------------------------------------------------


def _resolve_index(index: int, n_species: int) -> int:
    return index + n_species if index < 0 else index


class ImpurityClassicalFlux(NamedTuple):
    """Classical impurity transport summary (all quantities ``psiHat``-projected).

    Attributes:
        particle_flux: impurity classical radial particle flux ``Gamma_z``.
        heat_flux: impurity classical radial heat flux ``q_z`` (total definition).
        diffusion_coefficient: classical particle diffusion ``D_z^cl`` defined by
            ``Gamma_z = -D_z^cl (dn_z/dpsiHat) + (other drives)``.
        all_particle_fluxes: ``(S,)`` classical particle flux of every species.
        all_heat_fluxes: ``(S,)`` classical heat flux of every species.
        impurity_strength: ``Z_z^2 n_z / sum_i Z_i^2 n_i`` over the non-impurity
            species (small for a trace impurity).
        impurity_index: resolved impurity species index.
    """

    particle_flux: jnp.ndarray
    heat_flux: jnp.ndarray
    diffusion_coefficient: jnp.ndarray
    all_particle_fluxes: jnp.ndarray
    all_heat_fluxes: jnp.ndarray
    impurity_strength: jnp.ndarray
    impurity_index: int


def classical_diffusion_coefficient(
    species: SpeciesSet,
    *,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
    impurity_index: int = -1,
) -> jnp.ndarray:
    """Classical particle diffusion coefficient ``D_z^cl`` of the impurity.

    Defined operationally by ``D_z^cl = -d Gamma_z / d(dn_z/dpsiHat)``: the flux
    driven by the impurity's own density gradient with all other drives held
    fixed is ``Gamma_z = -D_z^cl (dn_z/dpsiHat)``.  Positive for a stable
    plasma.  Because the classical flux is linear in the gradients this equals
    the flux evaluated at a unit impurity density gradient (with a sign flip),
    which is exact and differentiable.
    """
    s = species.n_species
    idx = _resolve_index(impurity_index, s)
    zero = jnp.zeros((s,), dtype=jnp.float64)
    unit_dn = zero.at[idx].set(1.0)
    pf, _ = classical_species_fluxes(
        z=species.z, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
        dn_hat_dpsi_hat=unit_dn, dt_hat_dpsi_hat=zero,
        geometry_factor=geometry_factor, delta=delta, nu_n=nu_n,
    )  # fmt: skip
    return -pf[idx]


def classical_impurity_flux(
    species: SpeciesSet,
    *,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
    impurity_index: int = -1,
) -> ImpurityClassicalFlux:
    """Classical impurity particle/heat flux and diffusion coefficient.

    Wraps :func:`classical_species_fluxes` and isolates the impurity species.
    See :class:`ImpurityClassicalFlux` for the returned fields.
    """
    s = species.n_species
    idx = _resolve_index(impurity_index, s)
    pf, hf = _classical_species_fluxes_from_set(
        species, geometry_factor=geometry_factor, delta=delta, nu_n=nu_n
    )
    d_cl = classical_diffusion_coefficient(
        species, geometry_factor=geometry_factor, delta=delta, nu_n=nu_n, impurity_index=idx
    )
    z2n = species.z ** 2 * species.n_hat
    mask = jnp.arange(s) != idx
    bulk_z2n = jnp.sum(jnp.where(mask, z2n, 0.0))
    strength = (species.z[idx] ** 2 * species.n_hat[idx]) / bulk_z2n
    return ImpurityClassicalFlux(
        particle_flux=pf[idx],
        heat_flux=hf[idx],
        diffusion_coefficient=d_cl,
        all_particle_fluxes=pf,
        all_heat_fluxes=hf,
        impurity_strength=strength,
        impurity_index=idx,
    )


# ---------------------------------------------------------------------------
# Temperature-screening diagnostic
# ---------------------------------------------------------------------------


class TemperatureScreening(NamedTuple):
    """Ion-temperature screening diagnostic for a classical impurity flux.

    Attributes:
        ion_temperature_flux_coefficient: ``d Gamma_z / d(dT_i/dpsiHat)`` -- the
            raw (AD-consistent) sensitivity of the impurity particle flux to the
            bulk-ion temperature gradient.  This is the quantity a
            screening-aware gradient should recover.
        ion_density_peaking_coefficient: ``(d Gamma_z/d g_{n_i}) /
            (d Gamma_z/d g_{n_z})`` in logarithmic gradients ``g = (1/x)dx/dpsiHat``;
            exactly ``-Z_z / Z_i`` for the classical flux (the accumulation drive).
        screening_coefficient: ``H = (d Gamma_z/d g_{T_i}) /
            (d Gamma_z/d g_{n_z}) / Z_z`` -- the normalized ion-temperature
            screening coefficient, ``-> 1/2`` in the collisional heavy-impurity
            limit (``3/2`` would be the banana-regime value).
        density_pinch_flux: contribution to ``Gamma_z`` from *all* density
            gradients (the impurity accumulation drive).
        screening_flux: contribution to ``Gamma_z`` from the bulk-ion
            temperature gradient alone.
        screens: ``True`` when the ion-temperature contribution opposes the
            density-driven pinch for the supplied gradients (``screening_flux``
            and ``density_pinch_flux`` have opposite signs).
    """

    ion_temperature_flux_coefficient: jnp.ndarray
    ion_density_peaking_coefficient: jnp.ndarray
    screening_coefficient: jnp.ndarray
    density_pinch_flux: jnp.ndarray
    screening_flux: jnp.ndarray
    screens: jnp.ndarray


def temperature_screening_diagnostic(
    species: SpeciesSet,
    *,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
    bulk_index: int = 0,
    impurity_index: int = -1,
) -> TemperatureScreening:
    """Screening-aware decomposition of the classical impurity particle flux.

    Extracts the linear response of ``Gamma_z`` to the bulk-ion temperature
    gradient (``ion_temperature_flux_coefficient``), the exact ``-Z_z/Z_i``
    density peaking coefficient, and the normalized screening coefficient ``H``
    (``-> 1/2`` collisional).  See :class:`TemperatureScreening`.

    The ``screens`` flag answers, for the *supplied* gradients, whether the
    ion-temperature-gradient contribution to the impurity flux opposes the
    density-driven pinch -- the physical statement of temperature screening.
    """
    s = species.n_species
    z_idx = _resolve_index(impurity_index, s)
    i_idx = _resolve_index(bulk_index, s)
    zero = jnp.zeros((s,), dtype=jnp.float64)

    def particle_flux(dn: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        pf, _ = classical_species_fluxes(
            z=species.z, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
            dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt,
            geometry_factor=geometry_factor, delta=delta, nu_n=nu_n,
        )  # fmt: skip
        return pf[z_idx]

    # Raw AD-consistent sensitivity to the ion temperature gradient.
    unit_dti = zero.at[i_idx].set(1.0)
    coeff_dti = particle_flux(zero, unit_dti)

    # Logarithmic-gradient coefficients: perturb g = (1/x) dx/dpsiHat by unity,
    # i.e. set dx/dpsiHat = x for the target species only.
    c_lnz = particle_flux(zero.at[z_idx].set(float(species.n_hat[z_idx])), zero)
    c_lni = particle_flux(zero.at[i_idx].set(float(species.n_hat[i_idx])), zero)
    c_lti = particle_flux(zero, zero.at[i_idx].set(float(species.t_hat[i_idx])))
    peaking = c_lni / c_lnz
    screening = (c_lti / c_lnz) / species.z[z_idx]

    # Contributions for the supplied gradients.
    density_only = species.dn_hat_dpsi_hat
    temp_ion_only = zero.at[i_idx].set(species.dt_hat_dpsi_hat[i_idx])
    pinch = particle_flux(density_only, zero)
    screen = particle_flux(zero, temp_ion_only)
    screens = (pinch * screen) < 0.0

    return TemperatureScreening(
        ion_temperature_flux_coefficient=coeff_dti,
        ion_density_peaking_coefficient=peaking,
        screening_coefficient=screening,
        density_pinch_flux=pinch,
        screening_flux=screen,
        screens=screens,
    )


# ---------------------------------------------------------------------------
# vmap over charge states
# ---------------------------------------------------------------------------


def classical_impurity_flux_over_charge_states(
    bulk: SpeciesSet,
    *,
    impurity_charges: jnp.ndarray,
    impurity_masses: jnp.ndarray,
    impurity_n_hat: float,
    geometry_factor: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
    impurity_t_hat: float | None = None,
    match_bulk_logarithmic_gradients: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched classical impurity flux over a sweep of charge states (``jax.vmap``).

    Builds a bulk-plus-impurity :class:`SpeciesSet` for each ``(Z, m)`` pair and
    maps :func:`classical_impurity_flux` across them in one batched call, so the
    ``Z``-scaling of the classical flux comes out in a single vectorized
    evaluation.  ``impurity_masses`` must have the same shape as
    ``impurity_charges`` (pass e.g. ``2 * impurity_charges`` for a rough
    fully-stripped ion ``A ~ 2Z``).

    Returns:
        ``(particle_flux, heat_flux, diffusion_coefficient)`` each with the
        shape of ``impurity_charges``.
    """
    charges = jnp.asarray(impurity_charges, dtype=jnp.float64)
    masses = jnp.asarray(impurity_masses, dtype=jnp.float64)

    def one(z_imp: jnp.ndarray, m_imp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        species = _build_impurity_plasma_traced(
            bulk,
            impurity_z=z_imp,
            impurity_m_hat=m_imp,
            impurity_n_hat=impurity_n_hat,
            impurity_t_hat=impurity_t_hat,
            match_bulk_logarithmic_gradients=match_bulk_logarithmic_gradients,
        )
        res = classical_impurity_flux(
            species, geometry_factor=geometry_factor, delta=delta, nu_n=nu_n, impurity_index=-1
        )
        return res.particle_flux, res.heat_flux, res.diffusion_coefficient

    return lax.map(lambda zm: one(zm[0], zm[1]), jnp.stack([charges, masses], axis=1))


def _build_impurity_plasma_traced(
    bulk: SpeciesSet,
    *,
    impurity_z: jnp.ndarray,
    impurity_m_hat: jnp.ndarray,
    impurity_n_hat: float,
    impurity_t_hat: float | None,
    match_bulk_logarithmic_gradients: bool,
) -> SpeciesSet:
    """Traceable :func:`build_impurity_plasma` (impurity ``Z``/``m`` may be tracers)."""
    n0 = bulk.n_hat[0]
    t0 = bulk.t_hat[0]
    dn0 = bulk.dn_hat_dpsi_hat[0]
    dt0 = bulk.dt_hat_dpsi_hat[0]
    n_imp = jnp.asarray(impurity_n_hat, dtype=jnp.float64)
    t_imp = t0 if impurity_t_hat is None else jnp.asarray(impurity_t_hat, dtype=jnp.float64)
    if match_bulk_logarithmic_gradients:
        dn_imp = (dn0 / n0) * n_imp
        dt_imp = (dt0 / t0) * t_imp
    else:
        dn_imp = jnp.asarray(0.0, dtype=jnp.float64)
        dt_imp = jnp.asarray(0.0, dtype=jnp.float64)

    def _append(arr: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(arr, dtype=jnp.float64), jnp.reshape(jnp.asarray(value, dtype=jnp.float64), (1,))])

    return SpeciesSet(
        z=_append(bulk.z, impurity_z),
        m_hat=_append(bulk.m_hat, impurity_m_hat),
        n_hat=_append(bulk.n_hat, n_imp),
        t_hat=_append(bulk.t_hat, t_imp),
        dn_hat_dpsi_hat=_append(bulk.dn_hat_dpsi_hat, dn_imp),
        dt_hat_dpsi_hat=_append(bulk.dt_hat_dpsi_hat, dt_imp),
    )
