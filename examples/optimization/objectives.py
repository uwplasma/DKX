"""Composable neoclassical objectives for the dkx optimization examples.

This is a *small* shared library for the optimization example family
(``optimize_QH_bootstrap.py``, ``optimize_electron_root.py``,
``optimize_impurity_screening.py``, and the flagship
``examples/optimization/optimize_QA_bootstrap.py`` keeps its own inline copies so it stays
self-contained).  Every function here is a pure ``jax`` function of a solved
moment table (the dict returned by
:func:`dkx.run.profile_moments_from_operator`) or of a Boozer ``|B|``
spectrum, so ``jax.grad`` / ``jax.value_and_grad`` flow through them.  The user
writes the *scalar* objective (weights, penalties, which figure of merit) in
each script by composing these building blocks.

Two of the helpers here are plumbing rather than figures of merit
(:func:`operator_with_boozer_geometry` and :func:`solve_and_moments`).  They
wrap the canonical public API (``FluxSurfaceGeometry.from_fourier`` +
``dataclasses.replace`` of the kinetic operator's geometry leaves, then
``solve`` + ``profile_moments_from_operator``).  They are duplicated in the
flagship on purpose; if this pattern proves broadly useful it is a candidate to
promote into the package as a ``KineticOperator.with_boozer_surface(...)``
convenience (flagged in the example's report, not hidden here).

References:
  - bootstrap current & quasisymmetry optimization: Landreman & Buller,
    J. Plasma Phys. (2022); Landreman, Buller & Drevlak, arXiv:2205.02914.
  - ambipolar Er / electron root: Turkin et al., Phys. Plasmas 18, 022505
    (2011); Maassberg, Beidler & Turkin, Phys. Plasmas 16, 072504 (2009).
  - impurity temperature screening: Helander & Sigmar, *Collisional Transport*
    (2002); Newton, Helander et al., J. Plasma Phys. 83 (2017);
    Mollen et al., Plasma Phys. Control. Fusion 60, 084001 (2018).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.run import profile_moments_from_operator
from dkx.solve import solve as kinetic_solve

jax.config.update("jax_enable_x64", True)

# geometry leaves of a KineticOperator replaced when we swap in a new surface.
_GEOMETRY_LEAVES = (
    "b_hat", "db_hat_dtheta", "db_hat_dzeta", "d_hat",
    "b_hat_sup_theta", "b_hat_sup_zeta", "b_hat_sub_theta", "b_hat_sub_zeta",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Plumbing: put a differentiable Boozer |B| spectrum onto a kinetic operator
# ---------------------------------------------------------------------------
def operator_with_boozer_geometry(op_template, *, bmnc, m, n, nfp, iota, g_hat,
                                  i_hat, theta, zeta, theta_weights, zeta_weights):
    """Return ``op_template`` with its geometry replaced by a ``|B|`` spectrum.

    ``bmnc`` are the cosine amplitudes for mode numbers ``(m, n)`` (``n`` WITHOUT
    the field-period factor) in ``BBar`` units; ``iota``/``g_hat``/``i_hat`` are
    the surface's rotational transform and Boozer ``G``/``I``.  Traceable in
    every one of those (the ``geometryScheme = 13`` pure-JAX path), so gradients
    w.r.t. the spectrum flow through the returned operator.
    """
    geom = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=jnp.asarray(bmnc), m=jnp.asarray(m),
        n=jnp.asarray(n), n_periods=int(nfp), iota=iota, g_hat=g_hat, i_hat=i_hat,
    )
    fsab2 = geom.fsab_hat2(theta_weights=theta_weights, zeta_weights=zeta_weights)
    leaves = {name: getattr(geom, name) for name in _GEOMETRY_LEAVES}
    return dataclasses.replace(op_template, fsab_hat2=fsab2, **leaves)


def solve_and_moments(op, *, tol=1e-9, x0=None, recycle=None,
                      differentiable=True, method="gmres"):
    """Solve the drift-kinetic system and return ``(moment_table, SolveResult)``.

    Threads ``x0`` (warm start) and the GCROT ``recycle`` pair through the
    canonical :func:`dkx.solve.solve`; the returned moment table is the
    differentiable :func:`dkx.run.profile_moments_from_operator` output.
    """
    result = kinetic_solve(op, op.rhs(), method=method, tol=tol,
                           differentiable=differentiable, x0=x0, recycle=recycle)
    return profile_moments_from_operator(op, result.x), result


# ---------------------------------------------------------------------------
# Figures of merit on a solved moment table (all pure jax, minimize-oriented)
# ---------------------------------------------------------------------------
def bootstrap_current(mom):
    """Normalized bootstrap current ``<j.B> / sqrt(<B^2>)`` (signed scalar)."""
    return mom["FSABjHatOverRootFSAB2"]


def bootstrap_jbs2(mom):
    """``(<j.B>/sqrt(<B^2>))^2`` -> drive the bootstrap current to zero."""
    return mom["FSABjHatOverRootFSAB2"] ** 2


def _smooth_abs(x, eps):
    return jnp.sqrt(x * x + eps * eps)


def particle_flux_l1(mom, *, eps=1e-12):
    """Smooth ``sum_s |Gamma_s|`` of the radial particle fluxes (L1)."""
    return jnp.sum(_smooth_abs(mom["particleFlux_vm_psiHat"], eps))


def particle_flux_l2(mom):
    """``sum_s Gamma_s^2`` of the radial particle fluxes (L2)."""
    return jnp.sum(mom["particleFlux_vm_psiHat"] ** 2)


def heat_flux_l1(mom, *, eps=1e-12):
    """Smooth ``sum_s |Q_s|`` of the radial heat fluxes (L1)."""
    return jnp.sum(_smooth_abs(mom["heatFlux_vm_psiHat"], eps))


def heat_flux_l2(mom):
    """``sum_s Q_s^2`` of the radial heat fluxes (L2)."""
    return jnp.sum(mom["heatFlux_vm_psiHat"] ** 2)


def species_particle_flux(mom, s):
    """Radial particle flux of species ``s`` (``+`` = outward, up grad psi)."""
    return mom["particleFlux_vm_psiHat"][s]


def species_heat_flux(mom, s):
    """Radial heat flux of species ``s``."""
    return mom["heatFlux_vm_psiHat"][s]


def impurity_screening_metric(mom, impurity_index):
    """Outward radial flux of the impurity species (``+`` = screening).

    Temperature screening drives impurities *out* (up the flux-surface gradient,
    ``Gamma_z > 0``); accumulation is inward (``Gamma_z < 0``).  Maximize this
    (i.e. minimize its negative) to shape the field for impurity screening.
    """
    return mom["particleFlux_vm_psiHat"][impurity_index]


def radial_transport_coefficient(mom, gradient, s):
    """Effective radial diffusion ``D_s ~ -Gamma_s / (dn_s/dr)`` (D11-like proxy).

    A cheap RHSMode=1 proxy for the monoenergetic ``D11`` (which needs a
    dedicated RHSMode=3 monoenergetic solve): the ratio of the radial particle
    flux to the driving density gradient.  Positive for ordinary (down-gradient)
    transport.
    """
    return -mom["particleFlux_vm_psiHat"][s] / gradient


# ---------------------------------------------------------------------------
# Objectives on the ambipolar radial electric field (a scalar from er.py)
# ---------------------------------------------------------------------------
def root_offset_sq(er, target):
    """``(E_r - target)^2`` -- steer the ambipolar root to ``target``.

    Set ``target >= 0`` to push toward the electron-root side ``E_r > 0``.
    """
    return (er - target) ** 2


# ---------------------------------------------------------------------------
# Quasisymmetry residual straight from a Boozer |B| spectrum (composable)
# ---------------------------------------------------------------------------
def qs_symmetric_mask(xm_b, xn_b, nfp, helicity_m, helicity_n):
    """Static boolean mask of quasisymmetry-preserving Boozer modes.

    A mode ``(m, n)`` (``xn_b`` including the ``nfp`` factor) preserves the
    helicity ``(helicity_m, helicity_n)`` iff ``(n/nfp) * helicity_m ==
    m * helicity_n``.  QA is ``(1, 0)`` (symmetric <=> ``n == 0``); QH nfp=4 is
    ``(1, -1)`` (symmetric <=> ``n/nfp == -m``).  The ``(0, 0)`` term is always
    kept.
    """
    xm_b = np.asarray(xm_b, dtype=int)
    n_perfp = np.rint(np.asarray(xn_b, dtype=float) / float(nfp)).astype(int)
    return (n_perfp * int(helicity_m)) == (xm_b * int(helicity_n))


def boozer_qs_residual(bmnc_b, symmetric_mask):
    """Fraction of ``|B|`` spectral energy in symmetry-breaking modes.

    ``sum_{breaking} bmnc_b^2 / sum_all bmnc_b^2`` with ``symmetric_mask`` from
    :func:`qs_symmetric_mask`.  Zero for a perfectly quasisymmetric field.
    """
    bmnc_b = jnp.asarray(bmnc_b)
    breaking = jnp.where(jnp.asarray(symmetric_mask), 0.0, bmnc_b)
    return jnp.sum(breaking ** 2) / jnp.sum(bmnc_b ** 2)


# A registry so the CI test can prove every metric evaluates on a solved table.
MOMENT_METRICS = {
    "bootstrap_current": bootstrap_current,
    "bootstrap_jbs2": bootstrap_jbs2,
    "particle_flux_l1": particle_flux_l1,
    "particle_flux_l2": particle_flux_l2,
    "heat_flux_l1": heat_flux_l1,
    "heat_flux_l2": heat_flux_l2,
}
