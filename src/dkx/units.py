"""SFINCS reference values and the conversion of "Hat" outputs to SI.

Every quantity SFINCS computes is dimensionless: the code works with ratios to
a fixed set of reference values (the "Bar" quantities of the v3 technical
documentation).  Reporting a bootstrap current as ``-2.4e-02`` is therefore
correct but unreadable, and it cannot be compared against a VMEC ``jdotb``
profile or against a number in a paper without the conversion below.

Fortran correspondence (``sfincs/fortran/version3``) and the v3 technical
documentation ("20150507-01 Technical documentation for version 3 of SFINCS"):

- ``globalVariables.F90:133-135`` — the reference set is pinned, not free.
  ``Delta = mBar*vBar/(e*BBar*RBar) = 4.5694e-3`` and
  ``nu_n = nuBar*RBar/vBar = 8.330e-3`` hold simultaneously only for
  ``nBar = 1e20 m^-3``, ``TBar = 1 keV``, ``mBar = m_p``, ``BBar = 1 T``,
  ``RBar = 1 m`` and ``ln(Lambda) = 17``.  :func:`reference_delta` and
  :func:`reference_nu_n` recompute both from the SI values here, and
  ``tests/test_units.py`` pins them against
  :data:`dkx.constants.DEFAULT_DELTA` and :data:`dkx.constants.DEFAULT_NU_N`,
  so the set cannot drift silently.
- documentation eq. (98), ``f_s = (nBar/vBar**3) fHat_s`` — the single
  normalization from which every factor below follows.
- eq. (194)-(196) — ``FSABjHat`` and ``FSABjHatOverRootFSAB2`` carry
  ``e nBar vBar BBar`` and ``e nBar vBar``.
- eq. (201) — ``particleFlux_*`` carries ``nBar vBar / RBar``.
- eq. (220)-(221) — ``heatFlux_*`` carries ``nBar mBar vBar**3 / RBar``, which
  is ``2 nBar TBar vBar / RBar`` because ``mBar vBar**2 = 2 TBar``.
- eq. (175) and ``diagnostics.F90:703`` — a flux in the ``rHat`` coordinate is
  ``ddrHat2ddpsiHat`` times the ``psiHat`` one, and ``rHat = r/RBar``, so the
  ``RBar`` in the two factors above cancels: the SI radial flux densities are
  exactly :data:`PARTICLE_FLUX` and :data:`HEAT_FLUX` times the ``rHat`` value.
"""

from __future__ import annotations

import math

__all__ = [
    "ATOMIC_MASS_UNIT",
    "B_BAR",
    "COULOMB_LOGARITHM",
    "CURRENT_DENSITY",
    "ELEMENTARY_CHARGE",
    "HEAT_FLUX",
    "M_BAR",
    "N_BAR",
    "PARALLEL_CURRENT",
    "PARTICLE_FLUX",
    "PROTON_MASS",
    "R_BAR",
    "T_BAR",
    "V_BAR",
    "density_m3_from_n_hat",
    "electric_field_kv_m_from_er_hat",
    "er_hat_from_electric_field_kv_m",
    "flux_psi_hat_to_r_hat",
    "m_hat_from_mass_amu",
    "mass_amu_from_m_hat",
    "n_hat_from_density_m3",
    "reference_delta",
    "reference_nu_n",
    "t_hat_from_temperature_kev",
    "temperature_kev_from_t_hat",
]

#: SI defining constants (2019 redefinition; exact) and the proton mass.
ELEMENTARY_CHARGE: float = 1.602176634e-19  # C
PROTON_MASS: float = 1.67262192369e-27  # kg
VACUUM_PERMITTIVITY: float = 8.8541878128e-12  # F/m
#: The unified atomic mass unit (2022 CODATA), the unit of ``mass_amu``.
ATOMIC_MASS_UNIT: float = 1.66053906892e-27  # kg

#: The SFINCS reference set (see the module docstring for how it is pinned).
N_BAR: float = 1.0e20  # m^-3
T_BAR: float = 1.0e3 * ELEMENTARY_CHARGE  # J (1 keV)
M_BAR: float = PROTON_MASS  # kg
B_BAR: float = 1.0  # T
R_BAR: float = 1.0  # m
COULOMB_LOGARITHM: float = 17.0

#: ``vBar = sqrt(2 TBar / mBar)`` (documentation eq. 91), 4.3769e5 m/s.
V_BAR: float = math.sqrt(2.0 * T_BAR / M_BAR)

#: ``FSABjHatOverRootFSAB2`` -> ``<j.B>/sqrt(<B^2>)`` in A/m^2 (7.0126e6).
CURRENT_DENSITY: float = ELEMENTARY_CHARGE * N_BAR * V_BAR
#: ``FSABjHat`` -> ``<j.B>`` in A T/m^2, the unit of the VMEC ``jdotb`` profile.
PARALLEL_CURRENT: float = CURRENT_DENSITY * B_BAR
#: ``particleFlux_*_rHat`` -> ``<Gamma.grad r>`` in m^-2 s^-1 (4.3769e25).
PARTICLE_FLUX: float = N_BAR * V_BAR
#: ``heatFlux_*_rHat`` -> ``<Q.grad r>`` in W/m^2 (1.4025e10).
HEAT_FLUX: float = N_BAR * M_BAR * V_BAR**3


def reference_delta() -> float:
    """``Delta = mBar vBar / (e BBar RBar)`` from the SI reference values."""
    return M_BAR * V_BAR / (ELEMENTARY_CHARGE * B_BAR * R_BAR)


def reference_nu_n() -> float:
    """``nu_n = nuBar RBar / vBar`` from the SI reference values.

    ``nuBar`` is documentation eq. (95), the SI collisionality at the reference
    parameters, which is where ``ln(Lambda) = 17`` enters.
    """
    nu_bar = (
        4.0 * math.sqrt(2.0 * math.pi) * N_BAR * ELEMENTARY_CHARGE**4 * COULOMB_LOGARITHM
        / (3.0 * (4.0 * math.pi * VACUUM_PERMITTIVITY) ** 2 * math.sqrt(M_BAR) * T_BAR**1.5)
    )
    return nu_bar * R_BAR / V_BAR


#: --- Dimensionless "Hat" inputs <-> the SI quantities a native case declares.
#:
#: A SFINCS deck states ``nHats``, ``THats``, ``mHats`` and ``Er`` as ratios to
#: the reference set above; a DKX case states ``density_m3``, ``temperature_keV``,
#: ``mass_amu`` and ``value_kV_m``.  The two descriptions are the same physics
#: only through the *pinned* reference values, so both directions live here
#: rather than being re-derived at each boundary that crosses them.


def density_m3_from_n_hat(n_hat: float) -> float:
    """``n = nHat * nBar`` in m^-3."""
    return float(n_hat) * N_BAR


def n_hat_from_density_m3(density_m3: float) -> float:
    """``nHat = n / nBar``, the dimensionless density a deck declares."""
    return float(density_m3) / N_BAR


def temperature_kev_from_t_hat(t_hat: float) -> float:
    """``T = THat * TBar`` in keV (numerically ``THat``, since ``TBar = 1 keV``)."""
    return float(t_hat) * T_BAR / (1.0e3 * ELEMENTARY_CHARGE)


def t_hat_from_temperature_kev(temperature_kev: float) -> float:
    """``THat = T / TBar``, the dimensionless temperature a deck declares."""
    return float(temperature_kev) * 1.0e3 * ELEMENTARY_CHARGE / T_BAR


def mass_amu_from_m_hat(m_hat: float) -> float:
    """``m/u = mHat * mBar / u``.

    ``mHat`` is a ratio to the *proton* mass while a case declares a ratio to
    the unified atomic mass unit, so the two differ by 0.14% -- enough to move
    a collision frequency, and exactly the kind of silent factor this module
    exists to keep in one place.
    """
    return float(m_hat) * M_BAR / ATOMIC_MASS_UNIT


def m_hat_from_mass_amu(mass_amu: float) -> float:
    """``mHat = m / mBar`` from the atomic-mass-unit ratio a case declares."""
    return float(mass_amu) * ATOMIC_MASS_UNIT / M_BAR


def electric_field_kv_m_from_er_hat(er_hat: float) -> float:
    """``Er`` in kV/m from the dimensionless ``Er = -dPhiHat/drHat`` of a deck.

    ``PhiHat = e Phi / TBar`` and ``rHat = r / RBar``, so with ``TBar = 1 keV``
    and ``RBar = 1 m`` the two are numerically equal; the conversion is written
    out anyway so a change to the reference set cannot silently invalidate it.
    """
    return float(er_hat) * T_BAR / (1.0e3 * ELEMENTARY_CHARGE * R_BAR)


def er_hat_from_electric_field_kv_m(value_kv_m: float) -> float:
    """Inverse of :func:`electric_field_kv_m_from_er_hat`."""
    return float(value_kv_m) * 1.0e3 * ELEMENTARY_CHARGE * R_BAR / T_BAR


def flux_psi_hat_to_r_hat(*, psi_a_hat: float, a_hat: float, r_n: float) -> float:
    """``ddrHat2ddpsiHat = aHat / (2 psiAHat sqrt(psiN))`` (eq. 175).

    Multiply a ``*_psiHat`` flux by this to obtain the ``*_rHat`` one, exactly
    as ``diagnostics.F90:703`` does.  ``r_n = sqrt(psiN)`` is the normalized
    effective minor radius of the surface.

    **The factor carries a sign, and that is the point.**  ``psiAHat`` follows
    the wout's toroidal-flux orientation: it is ``+0.083`` for the precise-QA
    reference and ``-0.385`` for W7-X standard configuration.  A flux reported
    against ``psiHat`` therefore changes sign with a convention rather than with
    the physics, while the ``rHat`` one is outward-positive on both.  Reporting
    the ``rHat`` flux is what makes "the particle flux is outward" a statement
    about the device instead of about the file.
    """
    from dkx.constants import RadialCoordinates  # noqa: PLC0415

    return RadialCoordinates(
        psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=float(r_n)
    ).d_dr_hat_to_d_dpsi_hat
