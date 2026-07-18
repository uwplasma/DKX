"""Species containers for DKX: charges, masses, profiles, gradients, collisionality.

This module consolidates the per-species bookkeeping previously scattered across
``dkx/operators/profile_system.py`` (``full_system_operator_from_namelist``
species parsing), ``dkx/outputs/writer.py`` (species arrays and the four
radial-gradient output variants), and ``dkx/collisions.py``
(``nu_d_hat_pitch_angle_scattering_v3``).  It becomes ``dkx/species.py``
at the v2 purge.

Fortran correspondence (paths relative to ``sfincs/fortran/version3``):

- ``globalVariables.F90`` lines 89-98 — ``Zs``, ``mHats``, ``nHats``, ``THats``,
  the gradient arrays in all four radial coordinates, and the adiabatic-species
  parameters ``adiabaticZ/MHat/NHat/THat`` + ``withAdiabatic``.
- ``readInput.F90`` lines 124-135 — single-species defaults
  ``Zs(1)=mHats(1)=nHats(1)=THats(1)=1`` and zero gradients.
- ``radialCoordinates.F90`` lines 167-238 — selection of the input gradient
  coordinate and conversion of ``dnHatd*``/``dTHatd*`` to/from ``d/dpsiHat``.
- ``populateMatrix.F90`` (collisionOperator=1, no-Phi1 branch) — the deflection
  frequency ``nuDHat`` reproduced by :meth:`SpeciesSet.nu_d_hat`.

Normalization: ``THat = T/TBar``, ``nHat = n/nBar``, ``mHat = m/mBar``, ``Z`` in
units of the proton charge; the species thermal speed is ``v_a = vBar*sqrt(THat/mHat)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import tree_util as jtu  # noqa: E402
from jax.scipy.special import erf  # noqa: E402

from dkx.constants import (  # noqa: E402
    DEFAULT_ADIABATIC_M_HAT,
    DEFAULT_ADIABATIC_N_HAT,
    DEFAULT_ADIABATIC_T_HAT,
    DEFAULT_ADIABATIC_Z,
    RadialCoordinates,
    RadialGradients,
    SQRT_PI_V3,
)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class SpeciesSet:
    """Kinetic species: charges, normalized profiles, and psiHat-gradients.

    All fields are float64 arrays of shape ``(n_species,)``.  Gradients are stored
    w.r.t. ``psiHat`` only (the v3 internal convention after
    ``radialCoordinates.F90:setInputRadialCoordinate`` runs); the other three
    coordinate variants are derived via :meth:`density_gradients` /
    :meth:`temperature_gradients`.
    """

    z: jnp.ndarray  # (S,) charge in units of e (Fortran `Zs`)
    m_hat: jnp.ndarray  # (S,) mass / mBar (Fortran `mHats`)
    n_hat: jnp.ndarray  # (S,) density / nBar (Fortran `nHats`)
    t_hat: jnp.ndarray  # (S,) temperature / TBar (Fortran `THats`)
    dn_hat_dpsi_hat: jnp.ndarray  # (S,) (Fortran `dNHatdpsiHats`)
    dt_hat_dpsi_hat: jnp.ndarray  # (S,) (Fortran `dTHatdpsiHats`)

    def tree_flatten(self):
        return (
            (self.z, self.m_hat, self.n_hat, self.t_hat, self.dn_hat_dpsi_hat, self.dt_hat_dpsi_hat),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        z, m_hat, n_hat, t_hat, dn, dt = children
        return cls(z=z, m_hat=m_hat, n_hat=n_hat, t_hat=t_hat, dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt)

    @property
    def n_species(self) -> int:
        return int(self.z.shape[0])

    @property
    def v_hat(self) -> jnp.ndarray:
        """Thermal speed ``sqrt(THat/mHat)`` (v_a/vBar); cf. `sqrt_t_over_m` in operators/."""
        return jnp.sqrt(self.t_hat / self.m_hat)

    def density_gradients(self, radial: RadialCoordinates) -> RadialGradients:
        """``dnHatdpsiHat/psiN/rHat/rN`` per species (radialCoordinates.F90 lines 233-238)."""
        return radial.gradients_from_d_dpsi_hat(self.dn_hat_dpsi_hat)

    def temperature_gradients(self, radial: RadialCoordinates) -> RadialGradients:
        """``dTHatdpsiHat/psiN/rHat/rN`` per species (radialCoordinates.F90 lines 233-238)."""
        return radial.gradients_from_d_dpsi_hat(self.dt_hat_dpsi_hat)

    def nu_d_hat(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pitch-angle deflection frequency ``nuDHat(species, x)``.

        Matches the "WITHOUT PHI1" ``collisionOperator=1`` branch of v3
        ``populateMatrix.F90`` and reproduces
        ``collisions.nu_d_hat_pitch_angle_scattering_v3`` bit-for-bit:

          nuDHat_a(x) = (3*sqrt(pi)/4) * Za^2 / (THat_a*sqrt(THat_a*mHat_a))
                        * sum_b Zb^2 * nHat_b * (erf(x_b) - Psi(x_b)) / x^3

        with ``x_b = x*sqrt(THat_a*mHat_b/(THat_b*mHat_a))`` and Psi the
        Chandrasekhar function.  The multiplicative ``nu_n`` is NOT included,
        matching the Fortran variable.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        z2 = self.z * self.z  # (S,)
        t32m = self.t_hat * jnp.sqrt(self.t_hat * self.m_hat)  # (S,)
        species_factor = jnp.sqrt(
            (self.t_hat[:, None] * self.m_hat[None, :]) / (self.t_hat[None, :] * self.m_hat[:, None])
        )  # (S,S)
        xb = x[None, None, :] * species_factor[:, :, None]  # (S,S,X)
        psi = _chandrasekhar(xb)
        term = erf(xb) - psi  # (S,S,X)
        x3 = x * x * x  # (X,) — Fortran divides by the base grid x^3, not xb^3.
        x3 = jnp.where(x3 == 0, jnp.asarray(jnp.inf, dtype=jnp.float64), x3)
        term = term / x3[None, None, :]
        prefac = (3.0 * jnp.asarray(SQRT_PI_V3, dtype=jnp.float64) / 4.0) / t32m  # (S,)
        sum_b = jnp.sum((z2[None, :, None] * self.n_hat[None, :, None]) * term, axis=1)  # (S,X)
        return prefac[:, None] * z2[:, None] * sum_b


def _chandrasekhar(x: jnp.ndarray) -> jnp.ndarray:
    """Chandrasekhar function Psi(x) = (erf(x) - (2/sqrt(pi)) x exp(-x^2)) / (2 x^2).

    Identical to ``collisions._psi_chandra`` (v3 ``populateMatrix.F90``),
    including the small-x series switch at |x| < 1e-5.
    """
    x = x.astype(jnp.float64)
    sqrt_pi = jnp.asarray(SQRT_PI_V3, dtype=jnp.float64)
    num = erf(x) - (2.0 / sqrt_pi) * x * jnp.exp(-(x * x))
    den = 2.0 * x * x
    eps = jnp.asarray(1e-5, dtype=jnp.float64)
    small = jnp.abs(x) < eps
    x2 = x * x
    series = ((2.0 / 3.0) * x - (2.0 / 5.0) * x * x2 + (1.0 / 7.0) * x * x2 * x2) / sqrt_pi
    return jnp.where(small, series, num / den)


@dataclass(frozen=True)
class AdiabaticSpecies:
    """The optional adiabatic species (``withAdiabatic``; globalVariables.F90 line 97).

    Defaults are the v3 electron values: ``Z=-1``, ``mHat = m_e/m_p``,
    ``nHat = THat = 1``.
    """

    z: float = DEFAULT_ADIABATIC_Z
    m_hat: float = DEFAULT_ADIABATIC_M_HAT
    n_hat: float = DEFAULT_ADIABATIC_N_HAT
    t_hat: float = DEFAULT_ADIABATIC_T_HAT


def _as_1d_array(value: Any, default: float) -> jnp.ndarray:
    if value is None:
        value = [default]
    elif not isinstance(value, list):
        value = [value]
    return jnp.asarray(value, dtype=jnp.float64)


def infer_gradient_coordinate(
    *, geom_params: Mapping[str, Any], species_params: Mapping[str, Any], default: int = 4
) -> int:
    """Determine ``inputRadialCoordinateForGradients`` for the species gradients.

    An explicit setting in &geometryParameters wins (as in ``readInput.F90``);
    otherwise infer the coordinate from which gradient arrays are present.  This
    reproduces ``input_compat.infer_species_input_radial_coordinate_for_gradients``
    (a convenience beyond Fortran, which always uses the default, 4).
    """
    explicit = geom_params.get("INPUTRADIALCOORDINATEFORGRADIENTS", None)
    if explicit is not None:
        return int(explicit)
    for keys, code in (
        (("DNHATDRHATS", "DTHATDRHATS"), 2),
        (("DNHATDPSIHATS", "DTHATDPSIHATS"), 0),
        (("DNHATDPSINS", "DTHATDPSINS"), 1),
        (("DNHATDRNS", "DTHATDRNS"), 3),
    ):
        if any(species_params.get(k, None) is not None for k in keys):
            return code
    return int(default)


def species_set_from_namelist(
    nml: Any,
    *,
    radial: RadialCoordinates,
    input_radial_coordinate_for_gradients: int | None = None,
) -> SpeciesSet:
    """Build a :class:`SpeciesSet` from a parsed SFINCS input namelist.

    ``nml`` is a ``dkx.namelist.Namelist``.  Defaults follow
    ``readInput.F90`` lines 124-135 (unit charges/masses/profiles, zero
    gradients).  The gradient arrays given in the input coordinate are converted
    to ``d/dpsiHat`` exactly as ``radialCoordinates.F90`` lines 167-224 (and the
    existing ``operators/profile_system.py`` / ``outputs/writer.py`` paths) do.
    """
    species = nml.group("speciesParameters")
    geom_params = nml.group("geometryParameters")

    z = _as_1d_array(species.get("ZS", None), 1.0)
    m_hat = _as_1d_array(species.get("MHATS", None), 1.0)
    n_hat = _as_1d_array(species.get("NHATS", None), 1.0)
    t_hat = _as_1d_array(species.get("THATS", None), 1.0)
    s = int(z.shape[0])
    for name, arr in (("mHats", m_hat), ("nHats", n_hat), ("THats", t_hat)):
        if int(arr.shape[0]) != s:
            raise ValueError(f"{name} has {int(arr.shape[0])} entries but Zs has {s}")

    coord = (
        int(input_radial_coordinate_for_gradients)
        if input_radial_coordinate_for_gradients is not None
        else infer_gradient_coordinate(geom_params=geom_params, species_params=species)
    )
    key_by_coord = {0: "PSIHATS", 1: "PSINS", 2: "RHATS", 3: "RNS", 4: "RHATS"}
    try:
        suffix = key_by_coord[coord]
    except KeyError:
        raise ValueError(f"Invalid inputRadialCoordinateForGradients={coord}") from None

    dn_in = _as_1d_array(species.get(f"DNHATD{suffix}", None), 0.0)
    dt_in = _as_1d_array(species.get(f"DTHATD{suffix}", None), 0.0)
    if dn_in.shape[0] == 1 and s > 1:
        dn_in = jnp.broadcast_to(dn_in, (s,))
    if dt_in.shape[0] == 1 and s > 1:
        dt_in = jnp.broadcast_to(dt_in, (s,))
    if int(dn_in.shape[0]) != s or int(dt_in.shape[0]) != s:
        raise ValueError(f"Gradient arrays must have {s} entries (got {int(dn_in.shape[0])}, {int(dt_in.shape[0])})")

    return SpeciesSet(
        z=z,
        m_hat=m_hat,
        n_hat=n_hat,
        t_hat=t_hat,
        dn_hat_dpsi_hat=jnp.asarray(radial.to_d_dpsi_hat(dn_in, coordinate=coord), dtype=jnp.float64),
        dt_hat_dpsi_hat=jnp.asarray(radial.to_d_dpsi_hat(dt_in, coordinate=coord), dtype=jnp.float64),
    )


def adiabatic_species_from_namelist(nml: Any) -> AdiabaticSpecies | None:
    """Return the adiabatic species if ``withAdiabatic`` is set, else ``None``.

    Defaults match ``globalVariables.F90`` line 97 (and ``outputs/writer.py``).
    """
    species = nml.group("speciesParameters")
    if not bool(species.get("WITHADIABATIC", False)):
        return None
    return AdiabaticSpecies(
        z=float(species.get("ADIABATICZ", DEFAULT_ADIABATIC_Z)),
        m_hat=float(species.get("ADIABATICMHAT", DEFAULT_ADIABATIC_M_HAT)),
        n_hat=float(species.get("ADIABATICNHAT", DEFAULT_ADIABATIC_N_HAT)),
        t_hat=float(species.get("ADIABATICTHAT", DEFAULT_ADIABATIC_T_HAT)),
    )
