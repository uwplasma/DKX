"""Sugama-Nishimura parallel-momentum correction for monoenergetic transport.

The monoenergetic (RHSMode=3) transport coefficients produced by
:mod:`dkx.monoenergetic` are computed with the pitch-angle-scattering
(Lorentz) collision operator, which conserves particle number but *not*
parallel momentum.  Single- and multi-species neoclassical parallel flows and
the bootstrap current built directly from those coefficients are therefore
momentum-deficient: the full linearized Fokker-Planck operator restores the
parallel momentum through its field-particle (Rosenbluth) term, which the
pitch-angle operator omits.

This module implements the *moment* (Sugama-Nishimura) momentum-correction
technique, which restores inter-species parallel-momentum conservation as a
post-processing step on the monoenergetic coefficients.  The parallel flows of
all species are coupled through the parallel friction-coefficient matrix and a
small linear system is solved for the corrected flows, from which a corrected
flux-surface-averaged bootstrap current follows.

References
==========

- H. Sugama and S. Nishimura, Phys. Plasmas 9, 4637 (2002).
- H. Sugama and S. Nishimura, Phys. Plasmas 15, 042502 (2008).
- H. Maassberg, C. D. Beidler, and Y. Turkin, Phys. Plasmas 16, 072504 (2009).

Formulation
===========

For each species ``a`` the flux-surface-averaged parallel momentum balance in
the single-moment (parallel particle-flow) truncation reads

    M_a V_a  -  sum_b l_ab V_b  =  M_a^(0) V_a^unc ,

with ``V_a = <B V_{||a}>`` the parallel-flow moment, ``M_a`` the parallel
viscosity coefficient (Sugama-Nishimura 2002, obtained by a Maxwellian energy
convolution of the monoenergetic parallel-conductivity coefficient ``D33``),
``l_ab`` the parallel friction-coefficient matrix (self-adjoint,
``sum_a l_ab = 0`` so that like-particle collisions conserve momentum), and
``V_a^unc`` the uncorrected monoenergetic parallel flow.  The right-hand side
``M_a^(0) V_a^unc`` is the parallel viscous drive that reproduces the
uncorrected flow when the momentum coupling is switched off, so the corrected
flow reduces to ``V_a^unc`` exactly in that limit.

Writing the momentum-conserving friction as ``l_ab = gamma_ab`` for ``a != b``
and ``l_aa = -sum_{b!=a} gamma_ab`` with symmetric ``gamma_ab = gamma_ba``, the
balance becomes the linear system

    (diag(M_a) + Lambda) V = diag(M_a^(0)) V^unc ,
    Lambda_ab = delta_ab (sum_c gamma_ac) - gamma_ab ,

solved with :func:`jax.numpy.linalg.solve` (differentiable, jit/vmap-safe).
The corrected bootstrap current is ``<B j_||> = sum_a Z_a V_a`` (hat units,
per-species flow moments already carry the density).

The single-species momentum-restoring factor is the energy-convolution ratio
``M_a^(0) / M_a`` (the parallel viscosity computed without / with the
Sugama-Nishimura like-particle momentum-restoration factor); with a single
species the friction coupling vanishes and the corrected flow is exactly
``V_a = (M_a^(0)/M_a) V_a^unc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from .collisions import nu_d_hat_pitch_angle_scattering_v3  # noqa: E402
from .monoenergetic import (  # noqa: E402
    MonoenergeticDatabase,
    _dstar_lookup,
    energy_convolution,
)

__all__ = [
    "MomentumCorrectionResult",
    "ParallelViscosity",
    "momentum_corrected_bootstrap",
    "momentum_restoring_factor",
    "parallel_friction_matrix",
    "parallel_viscosity",
    "solve_corrected_flows",
]

# Beidler et al. (2011) large-aspect-ratio circulating fraction constant,
# shared with :mod:`dkx.monoenergetic` (1 - f_c = 1.46 sqrt(eps_t)).
_FT_LARGE_ASPECT = 1.46
_FRICTION_PREFACTOR = 3.0 * np.sqrt(np.pi) / 4.0


@dataclass(frozen=True)
class ParallelViscosity:
    """Energy-convolved parallel viscosity coefficients, per species.

    Attributes:
        uncorrected: ``(S,)`` viscosity ``M_a^(0)`` without the like-particle
            momentum-restoration factor (pure pitch-angle result).
        corrected: ``(S,)`` viscosity ``M_a`` with the Sugama-Nishimura
            momentum-restoration factor applied in the energy convolution.
        restoring_factor: ``(S,)`` ratio ``M_a^(0) / M_a`` (the single-species
            momentum-restoring factor).
    """

    uncorrected: Any
    corrected: Any
    restoring_factor: Any


@dataclass(frozen=True)
class MomentumCorrectionResult:
    """Result of the parallel-momentum correction.

    Attributes:
        corrected_flows: ``(S,)`` momentum-corrected parallel-flow moments
            ``<B V_{||a}>`` (hat units, density included).
        uncorrected_flows: ``(S,)`` input uncorrected parallel-flow moments.
        corrected_bootstrap: scalar corrected ``<B j_||>`` (hat units).
        uncorrected_bootstrap: scalar uncorrected ``<B j_||>``.
        delta_bootstrap: ``corrected_bootstrap - uncorrected_bootstrap``.
        friction_matrix: ``(S, S)`` symmetric parallel friction matrix
            ``gamma_ab``.
        viscosity: the :class:`ParallelViscosity` used.
    """

    corrected_flows: Any
    uncorrected_flows: Any
    corrected_bootstrap: Any
    uncorrected_bootstrap: Any
    delta_bootstrap: Any
    friction_matrix: Any
    viscosity: ParallelViscosity


def _as_species_arrays(z_s, m_hats, n_hats, t_hats):
    z = jnp.atleast_1d(jnp.asarray(z_s, dtype=jnp.float64))
    m = jnp.atleast_1d(jnp.asarray(m_hats, dtype=jnp.float64))
    n = jnp.atleast_1d(jnp.asarray(n_hats, dtype=jnp.float64))
    t = jnp.atleast_1d(jnp.asarray(t_hats, dtype=jnp.float64))
    if not (z.shape == m.shape == n.shape == t.shape):
        raise ValueError("z_s, m_hats, n_hats, t_hats must share the (S,) shape.")
    return z, m, n, t


def parallel_friction_matrix(
    *,
    z_s: Any,
    m_hats: Any,
    n_hats: Any,
    t_hats: Any,
    nu_n: Any,
) -> jnp.ndarray:
    """Symmetric, momentum-conserving parallel friction matrix ``gamma_ab``.

    Implements the single-moment parallel friction-coefficient matrix of the
    moment method (Sugama-Nishimura 2002): the flux-surface-averaged parallel
    friction on species ``a`` is ``<B F_a> = sum_b gamma_ab (V_b - V_a)`` with
    ``gamma_ab = gamma_ba`` the inter-species parallel momentum-exchange rate.
    The symmetric combination ``m_a n_a nu_ab = m_b n_b nu_ba`` makes total
    parallel momentum conserved by like- and unlike-particle collisions
    (``sum_a sum_b gamma_ab (V_b - V_a) = 0``).

    In SFINCS "hat" units the momentum-exchange rate is

        gamma_ab = (3 sqrt(pi)/4) nu_n Z_a^2 Z_b^2 n_a n_b
                   * sqrt(m_a m_b) (m_a + m_b) / (m_b T_a + m_a T_b)^{3/2} ,

    which is manifestly symmetric under ``a <-> b`` and reduces to the standard
    Braginskii parallel-friction weight ``(1 + m_a/m_b)(1 + x_ab^2)^{-3/2}``
    (``x_ab^2 = m_a T_b / (m_b T_a)``) times ``m_a n_a nu`` for each pair.

    Args:
        z_s, m_hats, n_hats, t_hats: species parameters, shape ``(S,)``.
        nu_n: deck normalized collisionality ``nu_n``.

    Returns:
        ``(S, S)`` symmetric friction matrix (``jnp`` array).
    """
    z, m, n, t = _as_species_arrays(z_s, m_hats, n_hats, t_hats)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)

    z2 = z * z
    num = jnp.sqrt(m[:, None] * m[None, :]) * (m[:, None] + m[None, :])
    denom = (m[None, :] * t[:, None] + m[:, None] * t[None, :]) ** 1.5
    gamma = (
        _FRICTION_PREFACTOR
        * nu_n
        * (z2[:, None] * z2[None, :])
        * (n[:, None] * n[None, :])
        * num
        / denom
    )
    return gamma


def _physical_d31_d33(
    db: MonoenergeticDatabase,
    *,
    z_s: jnp.ndarray,
    m_hats: jnp.ndarray,
    n_hats: jnp.ndarray,
    t_hats: jnp.ndarray,
    nu_n: jnp.ndarray,
    x_q: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Speed-resolved physical monoenergetic ``D31``, ``D33`` per species.

    Returns ``(D31, d33_star, d33_ps, nu_d)`` each of shape ``(S, X)`` in the
    SFINCS hat normalization, reconstructed exactly as in
    :func:`dkx.monoenergetic.energy_convolution` (same database lookup
    and un-normalization).  ``d33_star`` is the Beidler-normalized coefficient
    (``-> 1`` collisionally) and ``d33_ps`` its collisional (Spitzer) value, so
    the Sugama-Nishimura conductivity coefficient is
    ``D33_SN = d33_ps (1 - d33_star)``.
    """
    delta = jnp.asarray(db.delta, dtype=jnp.float64)
    g_hat = jnp.asarray(db.g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(db.i_hat, dtype=jnp.float64)
    iota = jnp.asarray(db.iota, dtype=jnp.float64)
    b0 = jnp.asarray(db.b0_over_bbar, dtype=jnp.float64)
    fsab2 = jnp.asarray(db.fsab_hat2, dtype=jnp.float64)
    x0 = jnp.asarray(db.x0, dtype=jnp.float64)
    nu_d_ref = jnp.asarray(db.nu_d_hat_x0, dtype=jnp.float64)
    g_plus = g_hat + iota * i_hat
    r_major = g_hat / b0
    eps_t = jnp.asarray(db.r_hat, dtype=jnp.float64) / r_major

    nu_d_hat = nu_d_hat_pitch_angle_scattering_v3(
        x=x_q, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
    )  # (S, X)

    d31_rows = []
    d33star_rows = []
    d33ps_rows = []
    nud_rows = []
    for s in range(int(z_s.shape[0])):
        z = z_s[s]
        m_hat = m_hats[s]
        t_hat = t_hats[s]
        vth = jnp.sqrt(t_hat / m_hat)
        v = x_q * vth
        nu_prime_x = g_plus / b0 * (x0 / nu_d_ref) * nu_n * nu_d_hat[s] / v
        _d11, _d13, d31s, d33s = _dstar_lookup(db, nu_prime_x, jnp.zeros_like(nu_prime_x))
        v_d = delta * m_hat * v * v / (2.0 * z * r_major * b0)
        d31_b = (2.0 / 3.0) * (v_d * r_major / (iota * eps_t)) * (
            _FT_LARGE_ASPECT * jnp.sqrt(eps_t)
        )
        nu_d = nu_n * nu_d_hat[s]
        d33_ps = v * v * fsab2 / (3.0 * nu_d * b0 * b0)
        d31_rows.append(d31s * d31_b)
        d33star_rows.append(d33s)
        d33ps_rows.append(d33_ps)
        nud_rows.append(nu_d)
    return (
        jnp.stack(d31_rows),
        jnp.stack(d33star_rows),
        jnp.stack(d33ps_rows),
        jnp.stack(nud_rows),
    )


def _quadrature(db: MonoenergeticDatabase, x, x_weights, n_x, x_max):
    if x is None or x_weights is None:
        if (x is None) != (x_weights is None):
            raise ValueError("Provide both x and x_weights, or neither.")
        nodes, weights = np.polynomial.legendre.leggauss(int(n_x))
        x_q = jnp.asarray(0.5 * float(x_max) * (nodes + 1.0), dtype=jnp.float64)
        w_q = jnp.asarray(0.5 * float(x_max) * weights, dtype=jnp.float64)
    else:
        x_q = jnp.asarray(x, dtype=jnp.float64)
        w_q = jnp.asarray(x_weights, dtype=jnp.float64)
    return x_q, w_q


def parallel_viscosity(
    db: MonoenergeticDatabase,
    *,
    z_s: Any,
    m_hats: Any,
    n_hats: Any,
    t_hats: Any,
    nu_n: Any,
    x: Any = None,
    x_weights: Any = None,
    n_x: int = 64,
    x_max: float = 5.0,
) -> ParallelViscosity:
    """Energy-convolved parallel viscosity coefficients (Sugama-Nishimura 2002).

    The single-moment parallel viscosity of species ``a`` is the Maxwellian
    energy convolution

        M_a = n_a (2/sqrt(pi)) int_0^inf dK sqrt(K) e^-K (m_a^2/T_a) nu_D^2
              D33_SN(K) F_a(K) ,

    with ``D33_SN = d33_ps (1 - d33_star)`` the Sugama-Nishimura parallel
    conductivity coefficient (the neoclassical, trapped-particle part of the
    monoenergetic ``D33``) and the like-particle momentum-restoration factor

        F_a(K) = [1 - (3 m_a nu_D D33_SN) / (2 T_a K <B^2>)]^{-1} .

    ``uncorrected`` sets ``F_a = 1`` (pure pitch-angle viscosity);
    ``corrected`` keeps ``F_a`` (the momentum-restored viscosity).  The ratio
    ``M_a^(0)/M_a`` is the single-species momentum-restoring factor.

    Args:
        db: the monoenergetic database.
        z_s, m_hats, n_hats, t_hats: species parameters, shape ``(S,)``.
        nu_n: deck normalized collisionality ``nu_n``.
        x, x_weights: optional speed quadrature (default Gauss-Legendre).
        n_x, x_max: default quadrature parameters.

    Returns:
        A :class:`ParallelViscosity`.
    """
    z, m, n, t = _as_species_arrays(z_s, m_hats, n_hats, t_hats)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)
    fsab2 = jnp.asarray(db.fsab_hat2, dtype=jnp.float64)
    x_q, w_q = _quadrature(db, x, x_weights, n_x, x_max)

    d31, d33_star, d33_ps, nu_d = _physical_d31_d33(
        db, z_s=z, m_hats=m, n_hats=n, t_hats=t, nu_n=nu_n, x_q=x_q
    )
    k = x_q * x_q  # (X,)
    quad = (4.0 / jnp.sqrt(jnp.pi)) * w_q * k * jnp.exp(-k)  # (X,)

    d33_sn = d33_ps * (1.0 - d33_star)  # (S, X)
    arg = (3.0 * m[:, None] * nu_d * d33_sn) / (2.0 * t[:, None] * k[None, :] * fsab2)
    factor = 1.0 / (1.0 - arg)  # (S, X)

    base = (m[:, None] ** 2 / t[:, None]) * nu_d * nu_d * d33_sn  # (S, X)
    m0 = n * jnp.sum(quad[None, :] * base, axis=1)
    mc = n * jnp.sum(quad[None, :] * base * factor, axis=1)
    return ParallelViscosity(uncorrected=m0, corrected=mc, restoring_factor=m0 / mc)


def momentum_restoring_factor(
    db: MonoenergeticDatabase,
    *,
    z_s: Any,
    m_hats: Any,
    n_hats: Any,
    t_hats: Any,
    nu_n: Any,
    x: Any = None,
    x_weights: Any = None,
    n_x: int = 64,
    x_max: float = 5.0,
) -> jnp.ndarray:
    """Single-species Sugama-Nishimura momentum-restoring factor ``M^(0)/M``.

    Convenience wrapper returning :attr:`ParallelViscosity.restoring_factor`.
    """
    return parallel_viscosity(
        db, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats, nu_n=nu_n,
        x=x, x_weights=x_weights, n_x=n_x, x_max=x_max,
    ).restoring_factor  # fmt: skip


def solve_corrected_flows(
    uncorrected_flows: Any,
    *,
    viscosity: ParallelViscosity,
    friction_matrix: Any,
    z_s: Any,
) -> MomentumCorrectionResult:
    """Solve the coupled momentum balance for the corrected parallel flows.

    Assembles and solves the linear system

        (diag(M_a) + Lambda) V = diag(M_a^(0)) V^unc ,
        Lambda_ab = delta_ab (sum_c gamma_ac) - gamma_ab ,

    with ``jnp.linalg.solve`` (differentiable).  ``M_a`` is the corrected
    viscosity, ``M_a^(0)`` the uncorrected one, so the corrected flow reduces
    to ``V^unc`` when ``M_a = M_a^(0)`` and the friction vanishes, and to the
    single-species momentum-restoring factor ``(M^(0)/M) V^unc`` when only the
    friction coupling vanishes.

    Args:
        uncorrected_flows: ``(S,)`` uncorrected parallel-flow moments.
        viscosity: :class:`ParallelViscosity` (uncorrected / corrected).
        friction_matrix: ``(S, S)`` symmetric friction matrix ``gamma_ab``.
        z_s: species charges, shape ``(S,)``.

    Returns:
        A :class:`MomentumCorrectionResult`.
    """
    v_unc = jnp.atleast_1d(jnp.asarray(uncorrected_flows, dtype=jnp.float64))
    z = jnp.atleast_1d(jnp.asarray(z_s, dtype=jnp.float64))
    gamma = jnp.asarray(friction_matrix, dtype=jnp.float64)
    m0 = jnp.asarray(viscosity.uncorrected, dtype=jnp.float64)
    mc = jnp.asarray(viscosity.corrected, dtype=jnp.float64)

    lam = jnp.diag(jnp.sum(gamma, axis=1)) - gamma
    a_mat = jnp.diag(mc) + lam
    rhs = m0 * v_unc
    v_corr = jnp.linalg.solve(a_mat, rhs)

    boot_unc = jnp.sum(z * v_unc)
    boot_corr = jnp.sum(z * v_corr)
    return MomentumCorrectionResult(
        corrected_flows=v_corr,
        uncorrected_flows=v_unc,
        corrected_bootstrap=boot_corr,
        uncorrected_bootstrap=boot_unc,
        delta_bootstrap=boot_corr - boot_unc,
        friction_matrix=gamma,
        viscosity=viscosity,
    )


def momentum_corrected_bootstrap(
    db: MonoenergeticDatabase,
    *,
    z_s: Any,
    m_hats: Any,
    n_hats: Any,
    t_hats: Any,
    nu_n: Any,
    dn_hat_dpsi_hat: Any,
    dt_hat_dpsi_hat: Any,
    dphi_hat_dpsi_hat: Any = 0.0,
    e_par_b: Any = 0.0,
    uncorrected_flows: Any = None,
    x: Any = None,
    x_weights: Any = None,
    n_x: int = 64,
    x_max: float = 5.0,
) -> MomentumCorrectionResult:
    """Momentum-corrected bootstrap current from a monoenergetic database.

    Thin facade wiring the monoenergetic energy convolution to the parallel
    friction/viscosity momentum correction:

    1. energy-convolve the database (:func:`monoenergetic.energy_convolution`)
       for the thermal transport matrices per species;
    2. build the uncorrected parallel-flow moments ``V_a^unc = I_3^a`` from the
       thermodynamic forces (unless ``uncorrected_flows`` is supplied
       directly, e.g. from a kinetic solve);
    3. form the parallel viscosity (:func:`parallel_viscosity`) and friction
       (:func:`parallel_friction_matrix`);
    4. solve the coupled momentum balance (:func:`solve_corrected_flows`).

    The thermodynamic forces follow the monoenergetic convention
    (``I_i = -n sum_j L_ij A_j``) with ``A_1 = n'/n - Z alpha Phi'/T
    - (3/2) T'/T``, ``A_2 = T'/T``, ``A_3 = -Z alpha B0 <E_par B>/(T <B^2>)``,
    all radial derivatives with respect to ``psiHat``.

    Args:
        db: the monoenergetic database.
        z_s, m_hats, n_hats, t_hats: species parameters, shape ``(S,)``.
        nu_n: deck normalized collisionality.
        dn_hat_dpsi_hat, dt_hat_dpsi_hat: radial density/temperature gradients
            ``dn/dpsiHat``, ``dT/dpsiHat`` per species, shape ``(S,)``.
        dphi_hat_dpsi_hat: radial-electric-field input ``dPhiHat/dpsiHat``.
        e_par_b: inductive parallel field moment ``<E_par B>`` (default 0).
        uncorrected_flows: optional ``(S,)`` override for the uncorrected
            parallel-flow moments (bypasses steps 1-2).
        x, x_weights, n_x, x_max: speed quadrature controls.

    Returns:
        A :class:`MomentumCorrectionResult`.
    """
    z, m, n, t = _as_species_arrays(z_s, m_hats, n_hats, t_hats)
    visc = parallel_viscosity(
        db, z_s=z, m_hats=m, n_hats=n, t_hats=t, nu_n=nu_n,
        x=x, x_weights=x_weights, n_x=n_x, x_max=x_max,
    )  # fmt: skip
    gamma = parallel_friction_matrix(z_s=z, m_hats=m, n_hats=n, t_hats=t, nu_n=nu_n)

    if uncorrected_flows is None:
        v_unc = _uncorrected_flows_from_forces(
            db, z_s=z, m_hats=m, n_hats=n, t_hats=t, nu_n=nu_n,
            dn_hat_dpsi_hat=dn_hat_dpsi_hat, dt_hat_dpsi_hat=dt_hat_dpsi_hat,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat, e_par_b=e_par_b,
            x=x, x_weights=x_weights, n_x=n_x, x_max=x_max,
        )  # fmt: skip
    else:
        v_unc = jnp.atleast_1d(jnp.asarray(uncorrected_flows, dtype=jnp.float64))

    return solve_corrected_flows(v_unc, viscosity=visc, friction_matrix=gamma, z_s=z)


def _uncorrected_flows_from_forces(
    db: MonoenergeticDatabase,
    *,
    z_s: jnp.ndarray,
    m_hats: jnp.ndarray,
    n_hats: jnp.ndarray,
    t_hats: jnp.ndarray,
    nu_n: jnp.ndarray,
    dn_hat_dpsi_hat: Any,
    dt_hat_dpsi_hat: Any,
    dphi_hat_dpsi_hat: Any,
    e_par_b: Any,
    x: Any,
    x_weights: Any,
    n_x: int,
    x_max: float,
) -> jnp.ndarray:
    """Uncorrected parallel-flow moments ``I_3^a`` from the thermodynamic forces."""
    thermal = energy_convolution(
        db, z_s=z_s, m_hats=m_hats, t_hats=t_hats, n_hats=n_hats, nu_n=nu_n,
        dphi_hat_dpsi_hat=dphi_hat_dpsi_hat, x=x, x_weights=x_weights,
        n_x=n_x, x_max=x_max,
    )  # fmt: skip
    l31 = thermal.entry(3, 1)
    l32 = thermal.entry(3, 2)
    l33 = thermal.entry(3, 3)

    dn = jnp.atleast_1d(jnp.asarray(dn_hat_dpsi_hat, dtype=jnp.float64))
    dt = jnp.atleast_1d(jnp.asarray(dt_hat_dpsi_hat, dtype=jnp.float64))
    alpha = jnp.asarray(db.alpha, dtype=jnp.float64)
    dphi = jnp.asarray(dphi_hat_dpsi_hat, dtype=jnp.float64)
    fsab2 = jnp.asarray(db.fsab_hat2, dtype=jnp.float64)
    b0 = jnp.asarray(db.b0_over_bbar, dtype=jnp.float64)
    e_par_b = jnp.asarray(e_par_b, dtype=jnp.float64)

    a1 = dn / n_hats - z_s * alpha * dphi / t_hats - 1.5 * dt / t_hats
    a2 = dt / t_hats
    a3 = -z_s * alpha * b0 * e_par_b / (t_hats * fsab2)
    return -n_hats * (l31 * a1 + l32 * a2 + l33 * a3)
