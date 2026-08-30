"""SFINCS normalization constants and radial-coordinate conversions.

This module consolidates the v3 normalization bookkeeping previously scattered
across ``dkx/outputs/transport.py`` (``conversion_factors_to_from_dpsi_hat``),
``dkx/outputs/writer.py`` (physics-parameter defaults), and
``dkx/operators/profile_*.py`` (``_V3_DEFAULT_DELTA`` / ``_V3_DEFAULT_NU_N``).
It becomes ``dkx/constants.py`` at the v2 purge.

Fortran correspondence (all paths relative to ``sfincs/fortran/version3``):

- ``globalVariables.F90`` — reference constants and input defaults:
  ``pi``/``sqrtpi`` (lines 16-17), ``Delta``/``alpha``/``nu_n`` (lines 133-135),
  ``nuPrime``/``EStar`` (line 155), adiabatic-species defaults (line 97).
- ``radialCoordinates.F90`` — flux-surface labels ``psiHat = psiAHat*rN**2``,
  ``psiN = rN**2``, ``rHat = aHat*rN`` (lines 143-145) and the six derivative
  conversion factors to/from ``d/dpsiHat`` (lines 157-164) plus the
  ``inputRadialCoordinateForGradients`` select block (lines 167-224).
- ``sfincs_main.F90`` — the RHSMode=3 (monoenergetic) overwrite of ``nu_n`` and
  ``dPhiHatdpsiHat`` from ``nuPrime`` and ``EStar`` (lines 151-154).

SFINCS normalization conventions: barred quantities (``BBar``, ``RBar``, ``nBar``,
``TBar``, ``mBar``, ``phiBar``, ``vBar = sqrt(2*TBar/mBar)``) are the dimensional
reference values; all code-level quantities are the dimensionless "Hat" ratios.
``Delta = mBar*vBar/(e*BBar*RBar)`` (~ rho*/R), ``alpha = e*phiBar/TBar``, and
``nu_n = nuBar*RBar/vBar`` is the collisionality at reference parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

# --- Reference numerical constants (globalVariables.F90 lines 16-17). -------------------
# v3 hard-codes 15-digit literals rather than computing pi to machine precision;
# strict-parity kernels (collisions.py) reuse exactly these values.
PI_V3: float = 3.14159265358979
SQRT_PI_V3: float = 1.77245385090552

# --- Normalization-parameter defaults (globalVariables.F90 lines 133-135). --------------
DEFAULT_DELTA: float = 4.5694e-3  # Delta = mBar*vBar/(e*BBar*RBar), i.e. rho*_ref
DEFAULT_ALPHA: float = 1.0  # alpha = e*phiBar/TBar
DEFAULT_NU_N: float = 8.330e-3  # nu_n = nuBar*RBar/vBar

# --- Monoenergetic (RHSMode=3) input defaults (globalVariables.F90 line 155). -----------
DEFAULT_NU_PRIME: float = 1.0
DEFAULT_E_STAR: float = 0.0

# --- Adiabatic-species defaults (globalVariables.F90 line 97). --------------------------
# adiabaticMHat is the electron/proton mass ratio m_e/m_p = 5.446170214e-4.
DEFAULT_ADIABATIC_Z: float = -1.0
DEFAULT_ADIABATIC_M_HAT: float = 5.446170214e-4
DEFAULT_ADIABATIC_N_HAT: float = 1.0
DEFAULT_ADIABATIC_T_HAT: float = 1.0

# ``inputRadialCoordinateForGradients`` codes (radialCoordinates.F90 lines 171-224):
#   0 = psiHat, 1 = psiN, 2 = rHat, 3 = rN, 4 = rHat for n/T but Er for Phi.
RADIAL_GRADIENT_COORDINATES: tuple[int, ...] = (0, 1, 2, 3, 4)


class RadialGradients(NamedTuple):
    """One radial gradient expressed in all four v3 radial coordinates.

    Mirrors the final block of ``radialCoordinates.F90`` (lines 227-236), which
    converts each input gradient from ``d/dpsiHat`` to every other coordinate.
    Entries may be scalars or per-species arrays.
    """

    d_dpsi_hat: object
    d_dpsi_n: object
    d_dr_hat: object
    d_dr_n: object


@dataclass(frozen=True)
class RadialCoordinates:
    """Flux-surface labels and derivative conversion factors for one surface.

    Replicates ``radialCoordinates.F90:setInputRadialCoordinate`` given the
    normalized toroidal-flux label ``r_n`` selected by the geometry module,
    ``psi_a_hat`` (boundary poloidal flux over 2*pi, normalized), and ``a_hat``
    (normalized effective minor radius).
    """

    psi_a_hat: float
    a_hat: float
    r_n: float

    # Flux-surface labels (radialCoordinates.F90 lines 143-145).
    @property
    def psi_n(self) -> float:
        return float(self.r_n) * float(self.r_n)

    @property
    def psi_hat(self) -> float:
        return float(self.psi_a_hat) * self.psi_n

    @property
    def r_hat(self) -> float:
        return float(self.a_hat) * float(self.r_n)

    # Conversion factors TO d/dpsiHat (radialCoordinates.F90 lines 157-159).
    @property
    def d_dpsi_n_to_d_dpsi_hat(self) -> float:
        return 1.0 / float(self.psi_a_hat)

    @property
    def d_dr_hat_to_d_dpsi_hat(self) -> float:
        return float(self.a_hat) / (2.0 * float(self.psi_a_hat) * math.sqrt(self.psi_n))

    @property
    def d_dr_n_to_d_dpsi_hat(self) -> float:
        return 1.0 / (2.0 * float(self.psi_a_hat) * math.sqrt(self.psi_n))

    # Conversion factors FROM d/dpsiHat (radialCoordinates.F90 lines 162-164).
    @property
    def d_dpsi_hat_to_d_dpsi_n(self) -> float:
        return float(self.psi_a_hat)

    @property
    def d_dpsi_hat_to_d_dr_hat(self) -> float:
        return (2.0 * float(self.psi_a_hat) * math.sqrt(self.psi_n)) / float(self.a_hat)

    @property
    def d_dpsi_hat_to_d_dr_n(self) -> float:
        return 2.0 * float(self.psi_a_hat) * math.sqrt(self.psi_n)

    def to_d_dpsi_hat(self, value, *, coordinate: int):
        """Convert a gradient given w.r.t. ``coordinate`` into ``d/dpsiHat``.

        ``coordinate`` follows the v3 ``inputRadialCoordinateForGradients`` codes;
        code 4 uses the rHat factor for n/T gradients (radialCoordinates.F90
        lines 211-214).  For Phi under code 4 the caller passes ``value = -Er``.
        """
        if coordinate == 0:
            return value
        if coordinate == 1:
            return self.d_dpsi_n_to_d_dpsi_hat * value
        if coordinate in (2, 4):
            return self.d_dr_hat_to_d_dpsi_hat * value
        if coordinate == 3:
            return self.d_dr_n_to_d_dpsi_hat * value
        raise ValueError(f"Invalid inputRadialCoordinateForGradients={coordinate}")

    def gradients_from_d_dpsi_hat(self, value) -> RadialGradients:
        """Express a ``d/dpsiHat`` gradient in all four radial coordinates.

        Mirrors radialCoordinates.F90 lines 227-236 (and the identical species
        blocks in ``outputs/writer.py``).
        """
        return RadialGradients(
            d_dpsi_hat=value,
            d_dpsi_n=self.d_dpsi_hat_to_d_dpsi_n * value,
            d_dr_hat=self.d_dpsi_hat_to_d_dr_hat * value,
            d_dr_n=self.d_dpsi_hat_to_d_dr_n * value,
        )


def nu_n_from_nu_prime(*, nu_prime: float, b0_over_bbar: float, g_hat: float, i_hat: float, iota: float) -> float:
    """RHSMode=3 collisionality overwrite: ``nu_n = nuPrime*B0OverBBar/(GHat+iota*IHat)``.

    Matches ``sfincs_main.F90`` line 153 (and ``outputs/writer.py``'s RHSMode=3 branch).
    """
    denom = float(g_hat) + float(iota) * float(i_hat)
    if denom == 0.0:
        raise ZeroDivisionError("nu_n_from_nu_prime: (GHat + iota*IHat) == 0")
    return float(nu_prime) * float(b0_over_bbar) / denom


def nu_prime_from_nu_n(*, nu_n: float, b0_over_bbar: float, g_hat: float, i_hat: float, iota: float) -> float:
    """Inverse of :func:`nu_n_from_nu_prime` (same Fortran definition, solved for nuPrime)."""
    if float(b0_over_bbar) == 0.0:
        raise ZeroDivisionError("nu_prime_from_nu_n: B0OverBBar == 0")
    return float(nu_n) * (float(g_hat) + float(iota) * float(i_hat)) / float(b0_over_bbar)


def d_phi_hat_d_psi_hat_from_e_star(
    *, e_star: float, alpha: float, delta: float, iota: float, b0_over_bbar: float, g_hat: float
) -> float:
    """RHSMode=3 radial-electric-field overwrite.

    ``dPhiHatdpsiHat = 2/(alpha*Delta) * EStar * iota * B0OverBBar / GHat``
    (``sfincs_main.F90`` line 154).
    """
    return 2.0 / (float(alpha) * float(delta)) * float(e_star) * float(iota) * float(b0_over_bbar) / float(g_hat)
