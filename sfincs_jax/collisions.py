"""Collision operators for the SFINCS drift-kinetic equation.

This module consolidates all collision-operator physics of SFINCS v3 into one
place, mirroring (paths relative to ``sfincs/fortran/version3``):

* ``populateMatrix.F90`` — the ``collisionOperator = 1`` pitch-angle-scattering
  (Lorentz) entries and the ``collisionOperator = 0`` full linearized
  Fokker-Planck entries (test-particle energy/pitch scattering plus the field
  term), both in the "WITHOUT PHI1" branch.
* ``xGrid.F90:computeRosenbluthPotentialResponse`` — the response matrices of
  the Rosenbluth potentials H and G expanded in the collocation polynomials of
  the speed grid (the field-term kernel for xGridScheme 5/6).
* ``polynomialInterpolationMatrix.F90`` — interpolation of the distribution
  between species-specific speed variables (via
  :func:`sfincs_jax.phase_space.polynomial_interpolation_matrix`).

Methods: the speed discretization and the modal treatment of the linearized
Fokker-Planck operator follow Landreman & Ernst, *J. Comput. Phys.* **243**,
130 (2013), and the SFINCS technical note "20150402-01 Implementation of the
Fokker-Planck operator".  The pitch coordinate is a Legendre modal expansion:
both operators are diagonal in the Legendre index ``l`` (and in theta/zeta),
with the Lorentz operator contributing the eigenvalue ``l(l+1)/2`` per mode
(:func:`sfincs_jax.phase_space.lorentz_eigenvalues`); the Fokker-Planck operator
is additionally dense in speed ``x`` and couples species pairs.

Normalization: as in ``constants``/``species``, all inputs are the
dimensionless "Hat" quantities and the assembled matrices carry the overall
collisionality ``nu_n = nuBar*RBar/vBar`` so that entries equal the Fortran
PETSc Jacobian entries (the sign convention is "as added to the row residual",
i.e. the Fokker-Planck blocks include the factor ``-nu_n``).

This canonical module replaces ``sfincs_jax/physics/collisions.py`` (PAS +
full FP, no-Phi1 branches) at the purge.  The ``includePhi1InCollisionOperator`` variant
(``FokkerPlanckV3Phi1Operator``) stays in the old module until the Phi1 track
(plan section 4, ``phi1.py``) migrates it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import tree_util as jtu  # noqa: E402
from jax.scipy.special import erf  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from sfincs_jax.constants import PI_V3, SQRT_PI_V3  # noqa: E402
from sfincs_jax.phase_space import (  # noqa: E402
    SpeedGrid,
    lorentz_eigenvalues,
    make_speed_grid,
    polynomial_interpolation_matrix,
)
from sfincs_jax.species import SpeciesSet  # noqa: E402

__all__ = [
    "CollisionMatrices",
    "FokkerPlanck",
    "PitchAngleScattering",
    "apply_fokker_planck",
    "apply_pitch_angle_scattering",
    "chandrasekhar",
    "collision_matrices_from_fokker_planck",
    "collision_matrices_from_pitch_angle_scattering",
    "make_fokker_planck",
    "make_pitch_angle_scattering",
    "rosenbluth_potential_terms",
]


# ----------------------------------------------------------------------------
# 1. Scalar kernels
# ----------------------------------------------------------------------------


def chandrasekhar(x: jnp.ndarray) -> jnp.ndarray:
    """Chandrasekhar function ``Psi(x) = (erf(x) - (2/sqrt(pi)) x e^{-x^2}) / (2 x^2)``.

    Matches the inline definition in ``populateMatrix.F90`` (both collision
    operators), including the series switch for ``|x| < 1e-5`` that avoids the
    cancellation at the origin.  Identical to the private helper used by
    :meth:`sfincs_jax.species.SpeciesSet.nu_d_hat`.
    """
    x = x.astype(jnp.float64)
    sqrt_pi = jnp.asarray(SQRT_PI_V3, dtype=jnp.float64)
    num = erf(x) - (2.0 / sqrt_pi) * x * jnp.exp(-(x * x))
    den = 2.0 * x * x
    small = jnp.abs(x) < jnp.asarray(1e-5, dtype=jnp.float64)
    x2 = x * x
    series = ((2.0 / 3.0) * x - (2.0 / 5.0) * x * x2 + (1.0 / 7.0) * x * x2 * x2) / sqrt_pi
    return jnp.where(small, series, num / den)


def _erf_np(x: np.ndarray) -> np.ndarray:
    """libm-based erf for closer parity with the Fortran intrinsic."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    """Active-Legendre-mode mask, shape ``(n_x, n_xi_max)`` (``Nxi_for_x`` ramps)."""
    ell = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return ell < n_xi_for_x[:, None]


# ----------------------------------------------------------------------------
# 2. Pitch-angle scattering (collisionOperator = 1, Lorentz operator)
# ----------------------------------------------------------------------------


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class PitchAngleScattering:
    """Pure pitch-angle-scattering collision operator in the Legendre basis.

    ``collisionOperator = 1`` without Phi1 (``populateMatrix.F90``): diagonal
    in (theta, zeta), in Legendre index ``l``, and in speed ``x``, with per-mode
    coefficient ``nu_n * nuDHat_a(x) * (l(l+1) + 2*Krook)/2``.

    Attributes:
      nu_n: Scalar collisionality normalization (namelist ``nu_n``).
      krook: Scalar Krook-operator addition (namelist ``Krook``).
      nu_d_hat: Deflection frequency per species and speed node, ``(S, X)``.
      n_xi_for_x: Active Legendre modes per speed node, ``(X,)`` int32.
      coef: Precomputed diagonal ``nu_n * nuDHat * (l(l+1)+2K)/2``, ``(S, X, L)``.
      mask_xi: Active-mode mask, ``(X, L)`` bool.
    """

    nu_n: jnp.ndarray
    krook: jnp.ndarray
    nu_d_hat: jnp.ndarray
    n_xi_for_x: jnp.ndarray
    coef: jnp.ndarray
    mask_xi: jnp.ndarray

    def tree_flatten(self):
        return (self.nu_n, self.krook, self.nu_d_hat, self.n_xi_for_x, self.coef, self.mask_xi), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        nu_n, krook, nu_d_hat, n_xi_for_x, coef, mask_xi = children
        return cls(nu_n=nu_n, krook=krook, nu_d_hat=nu_d_hat, n_xi_for_x=n_xi_for_x, coef=coef, mask_xi=mask_xi)


def make_pitch_angle_scattering(
    *,
    species: SpeciesSet,
    x: jnp.ndarray,
    nu_n: float,
    krook: float = 0.0,
    n_xi_for_x: jnp.ndarray,
    n_xi: int,
) -> PitchAngleScattering:
    """Build the ``collisionOperator = 1`` operator (no Phi1).

    The deflection frequency comes from
    :meth:`sfincs_jax.species.SpeciesSet.nu_d_hat` and the Legendre
    eigenvalue ``l(l+1)`` from :func:`sfincs_jax.phase_space.lorentz_eigenvalues`,
    reproducing ``physics.collisions.make_pitch_angle_scattering_v3_operator``
    exactly.

    Args:
      species: Kinetic species (charges, masses, densities, temperatures).
      x: Speed-grid nodes, shape ``(n_x,)``.
      nu_n: Collisionality normalization ``nu_n`` (namelist).
      krook: Krook addition (namelist ``Krook``).
      n_xi_for_x: Active Legendre modes per speed node, ``(n_x,)``.
      n_xi: Total number of Legendre modes.

    Returns:
      A :class:`PitchAngleScattering` operator.
    """
    nu_d_hat = species.nu_d_hat(jnp.asarray(x, dtype=jnp.float64))
    n_xi_int = int(n_xi)
    factor_l = 0.5 * (
        jnp.asarray(lorentz_eigenvalues(n_xi_int), dtype=jnp.float64) + 2.0 * krook
    )
    coef = jnp.asarray(nu_n, dtype=jnp.float64) * nu_d_hat[:, :, None] * factor_l[None, None, :]
    mask = _mask_xi(jnp.asarray(n_xi_for_x, dtype=jnp.int32), n_xi_int)
    return PitchAngleScattering(
        nu_n=jnp.asarray(nu_n, dtype=jnp.float64),
        krook=jnp.asarray(krook, dtype=jnp.float64),
        nu_d_hat=nu_d_hat,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        coef=coef,
        mask_xi=mask,
    )


def apply_pitch_angle_scattering(op: PitchAngleScattering, f: jnp.ndarray) -> jnp.ndarray:
    """Apply pitch-angle-scattering collisions to ``f``.

    Args:
      op: Operator from :func:`make_pitch_angle_scattering`.
      f: Distribution, shape ``(n_species, n_x, n_xi, n_theta, n_zeta)``.

    Returns:
      Array of the same shape (rows with inactive ``l`` are zeroed).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    n_species, n_x, n_xi, _, _ = f.shape
    if op.nu_d_hat.shape != (n_species, n_x):
        raise ValueError(f"op.nu_d_hat has shape {op.nu_d_hat.shape}, expected {(n_species, n_x)}")

    if op.coef.shape[-1] != n_xi:
        # Rebuild for the runtime number of Legendre modes (padded vectors).
        ell = jnp.arange(n_xi, dtype=jnp.float64)
        factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * op.krook)
        coef = op.nu_n * op.nu_d_hat[:, :, None] * factor_l[None, None, :]
        mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(coef.dtype)
    else:
        coef = op.coef
        mask = op.mask_xi.astype(coef.dtype)

    out = coef[:, :, :, None, None] * f
    return out * mask[None, :, :, None, None]


# ----------------------------------------------------------------------------
# 3. Rosenbluth-potential response (xGrid.F90:computeRosenbluthPotentialResponse)
# ----------------------------------------------------------------------------


def _evaluate_polynomial(x: float, *, j: int, a: np.ndarray, b: np.ndarray) -> float:
    """Evaluate the orthogonal polynomial ``p_j(x)`` by the 3-term recurrence.

    Mirrors ``xGrid.F90:evaluatePolynomial`` (1-based ``j``); ``a``/``b`` are
    the :class:`~sfincs_jax.phase_space.SpeedGrid` recurrence coefficients.
    """
    if j == 1:
        return 1.0
    pj_minus1 = 0.0
    pj = 1.0
    y = 0.0
    for ii in range(1, j):
        y = (x - float(a[ii])) * pj - float(b[ii]) * pj_minus1
        pj_minus1, pj = pj, y
    return float(y)


def rosenbluth_potential_terms(
    *,
    x: np.ndarray,
    x_weights: np.ndarray,
    x_grid_k: float,
    speed_grid: SpeedGrid,
    species: SpeciesSet,
    nl: int,
) -> np.ndarray:
    """Field-term response matrices of the linearized Fokker-Planck operator.

    Port of ``xGrid.F90:computeRosenbluthPotentialResponse`` for the "new"
    xGridScheme 5/6 path (``RosenbluthPotentialTerms`` in ``populateMatrix.F90``):
    the perturbed distribution of species B is expanded in the speed-grid
    collocation polynomials, the Rosenbluth potentials H and G and the needed
    derivatives are evaluated analytically per Legendre mode ``l < NL``, and
    the result is contracted back to collocation values of species A.  All
    integrals use QUADPACK (``scipy.integrate.quad``) with
    ``epsabs = epsrel = 1e-13`` and the same semi-infinite split at
    ``partition = max(10, 2*xb)`` as the Fortran, so entries match the Fortran
    matrices to quadrature accuracy.  See Landreman & Ernst, JCP 243, 130
    (2013), section 4, and technical note 20150402-01.

    This routine is NumPy/SciPy precomputation (static grids); it is not part
    of the differentiable JAX graph.

    Args:
      x: Speed nodes, shape ``(n_x,)``.
      x_weights: Plain ``dx`` speed quadrature weights, shape ``(n_x,)``.
      x_grid_k: Weight exponent ``k`` (namelist ``xGrid_k``).
      speed_grid: The :class:`~sfincs_jax.phase_space.SpeedGrid` carrying the
        orthogonal-polynomial recurrence for these nodes.
      species: Kinetic species set.
      nl: Number of Legendre modes retained in the potentials (namelist ``NL``).

    Returns:
      Array of shape ``(S, S, NL, n_x, n_x)`` indexed
      ``(species_row, species_col, l, x_row, x_col)``.
    """
    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    z_s = np.asarray(species.z, dtype=np.float64)
    m_hats = np.asarray(species.m_hat, dtype=np.float64)
    n_hats = np.asarray(species.n_hat, dtype=np.float64)
    t_hats = np.asarray(species.t_hat, dtype=np.float64)

    n_x = int(x.size)
    n_species = int(z_s.size)

    expx2 = np.exp(-(x * x))
    a = np.asarray(speed_grid.poly_a, dtype=np.float64)
    b = np.asarray(speed_grid.poly_b, dtype=np.float64)
    poly_c = np.asarray(speed_grid.poly_c, dtype=np.float64)

    # collocation2modal(j,i) in the Fortran code:
    pvals = np.zeros((n_x, n_x), dtype=np.float64)  # (j,i)
    for j in range(1, n_x + 1):
        for i in range(n_x):
            pvals[j - 1, i] = _evaluate_polynomial(float(x[i]), j=j, a=a, b=b)
    collocation2modal = (x_weights[None, :] * (x[None, :] ** float(x_grid_k)) * pvals) / (
        poly_c[1 : n_x + 1, None]
    )

    pi = float(PI_V3)
    epsabs = 1e-13
    epsrel = 1e-13
    limit = 5000

    terms = np.zeros((n_species, n_species, int(nl), n_x, n_x), dtype=np.float64)
    for ell in range(int(nl)):
        alpha = -float(2 * ell - 1) / float(2 * ell + 3)
        denom_h = float(2 * ell + 1)
        denom_g = float(4 * ell * ell - 1)

        for ia in range(n_species):
            for ib in range(n_species):
                species_factor = float(
                    math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
                )
                species_factor2 = float(3.0 / (2.0 * pi)) * float(n_hats[ia]) * float(z_s[ia] ** 2) * float(
                    z_s[ib] ** 2
                )
                species_factor2 *= float(t_hats[ib] * m_hats[ia]) / float(t_hats[ia] * m_hats[ib])
                species_factor2 /= float(t_hats[ia] * math.sqrt(t_hats[ia] * m_hats[ia]))

                temp_h = np.zeros((n_x, n_x), dtype=np.float64)
                temp_dh = np.zeros((n_x, n_x), dtype=np.float64)
                temp_d2g = np.zeros((n_x, n_x), dtype=np.float64)

                for ix in range(n_x):
                    xb = float(x[ix] * species_factor)
                    xb_safe = xb if xb > 0 else 1e-14

                    # v3 splits semi-infinite integrals at partition = max(10, 2*xb).
                    partition = float(max(10.0, 2.0 * xb_safe))

                    for j in range(1, n_x + 1):

                        def integrand(t: float, power: int) -> float:
                            # v3 excludes t**xGrid_k in these integrals.
                            return (t**power) * _evaluate_polynomial(t, j=j, a=a, b=b) * math.exp(-(t * t))

                        def quad_finite(power: int, a0: float, b0: float) -> float:
                            val, _ = quad(
                                lambda tt: integrand(tt, power),
                                a0,
                                b0,
                                epsabs=epsabs,
                                epsrel=epsrel,
                                limit=limit,
                            )
                            return float(val)

                        def quad_semiinf(power: int, a0: float) -> float:
                            val, _ = quad(
                                lambda tt: integrand(tt, power),
                                a0,
                                np.inf,
                                epsabs=epsabs,
                                epsrel=epsrel,
                                limit=limit,
                            )
                            return float(val)

                        i_2pl = quad_finite(ell + 2, 0.0, xb_safe)
                        i_4pl = quad_finite(ell + 4, 0.0, xb_safe)
                        i_1ml = quad_finite(1 - ell, xb_safe, partition) + quad_semiinf(1 - ell, partition)
                        i_3ml = quad_finite(3 - ell, xb_safe, partition) + quad_semiinf(3 - ell, partition)

                        xb_pow_l = xb_safe**ell
                        xb_pow_lm1 = xb_safe ** (ell - 1) if ell >= 1 else xb_safe ** (-1)
                        xb_pow_lm2 = xb_safe ** (ell - 2) if ell >= 2 else xb_safe ** (-2)

                        temp_h[ix, j - 1] = (4.0 * pi / denom_h) * (
                            i_2pl / (xb_safe ** (ell + 1)) + xb_pow_l * i_1ml
                        )
                        temp_dh[ix, j - 1] = (4.0 * pi / denom_h) * (
                            -(ell + 1) * i_2pl / (xb_safe ** (ell + 2)) + ell * xb_pow_lm1 * i_1ml
                        )
                        temp_d2g[ix, j - 1] = (-4.0 * pi / denom_g) * (
                            ell * (ell - 1) * xb_pow_lm2 * i_3ml
                            + alpha * (ell + 1) * (ell + 2) * xb_pow_l * i_1ml
                            + alpha * (ell + 1) * (ell + 2) * i_4pl / (xb_safe ** (ell + 3))
                            + ell * (ell - 1) * i_2pl / (xb_safe ** (ell + 1))
                        )

                temp_combined = np.zeros((n_x, n_x), dtype=np.float64)
                mass_ratio = float(m_hats[ia] / m_hats[ib])
                for i in range(n_x):
                    xb = float(x[i] * species_factor)
                    temp_combined[i, :] = species_factor2 * expx2[i] * (
                        -temp_h[i, :]
                        - (1.0 - mass_ratio) * xb * temp_dh[i, :]
                        + float(x[i] * x[i]) * temp_d2g[i, :]
                    )

                terms[ia, ib, ell, :, :] = temp_combined @ collocation2modal

    return terms


# ----------------------------------------------------------------------------
# 4. Full linearized Fokker-Planck (collisionOperator = 0)
# ----------------------------------------------------------------------------


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class FokkerPlanck:
    """Full linearized Fokker-Planck collision operator (no Phi1).

    ``collisionOperator = 0`` in ``populateMatrix.F90``: diagonal in
    (theta, zeta) and in Legendre index ``l``, dense in speed ``x`` and in the
    species pair.  ``mat`` already carries the overall ``-nu_n`` so its entries
    equal the Fortran PETSc Jacobian entries.

    Attributes:
      mat: Dense x-blocks per (species pair, l), ``(S, S, L, X, X)``.
      n_xi_for_x: Active Legendre modes per speed node, ``(X,)`` int32.
      mask_xi: Active-mode mask, ``(X, L)`` bool.
    """

    mat: jnp.ndarray
    n_xi_for_x: jnp.ndarray
    mask_xi: jnp.ndarray

    def tree_flatten(self):
        return (self.mat, self.n_xi_for_x, self.mask_xi), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mat, n_xi_for_x, mask_xi = children
        return cls(mat=mat, n_xi_for_x=n_xi_for_x, mask_xi=mask_xi)


def make_fokker_planck(
    *,
    x: np.ndarray,
    x_weights: np.ndarray,
    ddx: np.ndarray,
    d2dx2: np.ndarray,
    x_grid_k: float,
    species: SpeciesSet,
    nu_n: float,
    krook: float = 0.0,
    n_xi: int,
    nl: int,
    n_xi_for_x: np.ndarray,
    strict_parity: bool = False,
) -> FokkerPlanck:
    """Build the ``collisionOperator = 0`` operator (no Phi1).

    Assembles, per species pair and Legendre mode, the dense-in-x matrix
    ``-nu_n * (C_E + C_D + Rosenbluth field term - (l(l+1)+2K)/2 * nuDHat)``
    exactly as ``populateMatrix.F90`` does for xGridScheme 5/6 (see technical
    note 20150402-01):

    * ``C_E``: test-particle energy scattering — first/second speed-derivative
      collocation matrices weighted by the Chandrasekhar function, plus the
      local Maxwellian diagonal term.
    * ``C_D``: the part of the field term independent of the Rosenbluth
      potentials, using barycentric interpolation between the species speed
      variables (``polynomialInterpolationMatrix.F90``).
    * Rosenbluth field term for ``l < nl`` from
      :func:`rosenbluth_potential_terms`.
    * The pitch-angle deflection diagonal ``-(l(l+1)+2*Krook)/2 * nuDHat``.

    Args:
      x: Speed nodes, ``(n_x,)``.
      x_weights: Plain ``dx`` quadrature weights, ``(n_x,)``.
      ddx: d/dx collocation matrix, ``(n_x, n_x)``.
      d2dx2: d^2/dx^2 collocation matrix, ``(n_x, n_x)``.
      x_grid_k: Weight exponent (namelist ``xGrid_k``).
      species: Kinetic species set.
      nu_n: Collisionality normalization (namelist ``nu_n``).
      krook: Krook addition (namelist ``Krook``).
      n_xi: Number of Legendre modes.
      nl: Legendre modes kept in the Rosenbluth potentials (namelist ``NL``).
      n_xi_for_x: Active Legendre modes per speed node, ``(n_x,)``.
      strict_parity: Use Fortran-ordered scalar loops for bit-level parity with
        the Fortran sums (used for multispecies RHSMode=1 golden tests).

    Returns:
      A :class:`FokkerPlanck` operator.
    """
    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    ddx = np.asarray(ddx, dtype=np.float64)
    d2dx2 = np.asarray(d2dx2, dtype=np.float64)
    z_s = np.asarray(species.z, dtype=np.float64)
    m_hats = np.asarray(species.m_hat, dtype=np.float64)
    n_hats = np.asarray(species.n_hat, dtype=np.float64)
    t_hats = np.asarray(species.t_hat, dtype=np.float64)
    n_xi_for_x = np.asarray(n_xi_for_x, dtype=np.int32)

    n_species = int(z_s.size)
    n_x = int(x.size)
    sqrt_pi = float(SQRT_PI_V3)
    expx2 = np.exp(-(x * x))
    x2 = x * x
    x3 = x2 * x

    # Rosenbluth response matrices (xGridScheme 5/6 "new" scheme).
    speed_grid = make_speed_grid(n_x=n_x, k=float(x_grid_k), include_point_at_x0=False)
    rosen = rosenbluth_potential_terms(
        x=x,
        x_weights=x_weights,
        x_grid_k=float(x_grid_k),
        speed_grid=speed_grid,
        species=species,
        nl=int(nl),
    )  # (S,S,NL,X,X)

    # nuDHat and CECD (= C_E + C_D) both omit the overall nu_n, matching v3.
    nu_d_hat = np.zeros((n_species, n_x), dtype=np.float64)
    cecd = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)

    strict_fp = bool(strict_parity)

    for ia in range(n_species):
        t32m = float(t_hats[ia]) * math.sqrt(float(t_hats[ia]) * float(m_hats[ia]))
        for ib in range(n_species):
            species_factor = float(
                math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
            )
            xb = x * species_factor
            if strict_fp:
                expxb2 = np.empty((n_x,), dtype=np.float64)
                erfs = np.empty((n_x,), dtype=np.float64)
                psi = np.empty((n_x,), dtype=np.float64)
                for ix in range(n_x):
                    xb_val = float(xb[ix])
                    exp_val = math.exp(-(xb_val * xb_val))
                    erf_val = math.erf(xb_val)
                    expxb2[ix] = exp_val
                    erfs[ix] = erf_val
                    if abs(xb_val) < 1e-14:
                        psi[ix] = (2.0 / sqrt_pi) * xb_val / 3.0
                    else:
                        psi[ix] = (erf_val - (2.0 / sqrt_pi) * xb_val * exp_val) / (2.0 * xb_val * xb_val)
            else:
                expxb2 = np.exp(-(xb * xb))
                erfs = _erf_np(xb)
                psi = (erfs - (2.0 / sqrt_pi) * xb * expxb2) / (2.0 * xb * xb)

            # nuDHat: base x-grid x^3 in the denominator (matching Fortran).
            nu_factor = (3.0 * sqrt_pi / 4.0) / t32m * float(z_s[ia] ** 2) * float(
                z_s[ib] ** 2
            ) * float(n_hats[ib])
            if strict_fp:
                for ix in range(n_x):
                    nu_d_hat[ia, ix] += nu_factor * (erfs[ix] - psi[ix]) / x3[ix]
            else:
                nu_d_hat[ia, :] += nu_factor * (erfs - psi) / x3

            # Interpolate species-B f(x_b) onto the species-A x grid.
            if ia == ib:
                f_to_f = np.eye(n_x, dtype=np.float64)
            else:
                alpxk = expx2 * (x ** float(x_grid_k))
                alpx = expxb2 * (xb ** float(x_grid_k))
                f_to_f = polynomial_interpolation_matrix(xk=x, x=xb, alpxk=alpxk, alpx=alpx)

            # CD: field term independent of the Rosenbluth potentials.
            species_factor_cd = (
                3.0
                * float(n_hats[ia])
                * float(m_hats[ia] / m_hats[ib])
                * float(z_s[ia] ** 2)
                * float(z_s[ib] ** 2)
                / t32m
            )
            if strict_fp:
                for ix in range(n_x):
                    for jx in range(n_x):
                        cecd[ia, ib, ix, jx] += species_factor_cd * expx2[ix] * f_to_f[ix, jx]
            else:
                cecd[ia, ib, :, :] += (species_factor_cd * expx2)[:, None] * f_to_f

            # CE: energy scattering (diagonal in species, depends on species B).
            species_factor_ce = (
                3.0
                * sqrt_pi
                / 4.0
                * float(n_hats[ib])
                * float(z_s[ia] ** 2)
                * float(z_s[ib] ** 2)
                / t32m
            )
            coef_d2 = (psi / x)[:, None] * d2dx2
            coef_dx = (
                (
                    -2.0
                    * float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia]))
                    * psi
                    * (1.0 - float(m_hats[ia] / m_hats[ib]))
                    + (erfs - psi) / x2
                )[:, None]
                * ddx
            )
            if strict_fp:
                for ix in range(n_x):
                    for jx in range(n_x):
                        cecd[ia, ia, ix, jx] += species_factor_ce * (coef_d2[ix, jx] + coef_dx[ix, jx])
            else:
                cecd[ia, ia, :, :] += species_factor_ce * (coef_d2 + coef_dx)

            diag_extra = (
                species_factor_ce
                * 4.0
                / sqrt_pi
                * float(t_hats[ia] / t_hats[ib])
                * math.sqrt(float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia])))
                * expxb2
            )
            if strict_fp:
                for ix in range(n_x):
                    cecd[ia, ia, ix, ix] += diag_extra[ix]
            else:
                cecd[ia, ia, range(n_x), range(n_x)] += diag_extra

    # Per-L matrices including the overall (-nu_n), matching PETSc Jacobian entries.
    mat = np.zeros((n_species, n_species, int(n_xi), n_x, n_x), dtype=np.float64)
    for ell in range(int(n_xi)):
        m11 = cecd.copy()
        diag = -0.5 * nu_d_hat * (float(ell * (ell + 1)) + 2.0 * float(krook))
        for s in range(n_species):
            m11[s, s, range(n_x), range(n_x)] += diag[s, :]
        if ell < int(nl):
            m11 = m11 + rosen[:, :, ell, :, :]
        mat[:, :, ell, :, :] = -float(nu_n) * m11

    return FokkerPlanck(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        mask_xi=_mask_xi(jnp.asarray(n_xi_for_x, dtype=jnp.int32), int(n_xi)),
    )


def _apply_dense_x_blocks(
    mat: jnp.ndarray, n_xi_for_x: jnp.ndarray, mask_xi: jnp.ndarray, f: jnp.ndarray
) -> jnp.ndarray:
    """Apply per-(species pair, l) dense x-blocks to ``f`` and mask inactive rows."""
    n_xi = f.shape[2]
    # y[a,i,l,t,z] = sum_{b,j} mat[a,b,l,i,j] * f[b,j,l,t,z]
    f2 = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
    y2 = jnp.einsum("abLij,bLjtz->aLitz", mat, f2)  # (S,L,X,T,Z)
    y = jnp.transpose(y2, (0, 2, 1, 3, 4))  # (S,X,L,T,Z)
    if mask_xi.shape[-1] != n_xi:
        mask = _mask_xi(n_xi_for_x.astype(jnp.int32), n_xi).astype(y.dtype)
    else:
        mask = mask_xi.astype(y.dtype)
    return y * mask[None, :, :, None, None]


def apply_fokker_planck(op: FokkerPlanck, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the ``collisionOperator = 0`` collision operator to ``f`` (no Phi1).

    Args:
      op: Operator from :func:`make_fokker_planck`.
      f: Distribution, shape ``(n_species, n_x, n_xi, n_theta, n_zeta)``.

    Returns:
      Array of the same shape (rows with inactive ``l`` are zeroed; input
      columns are deliberately NOT masked, matching the padded-vector behavior
      of the production evaluators).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    n_species, n_x, n_xi, _, _ = f.shape
    if op.mat.shape != (n_species, n_species, n_xi, n_x, n_x):
        raise ValueError(f"op.mat has shape {op.mat.shape}, expected {(n_species, n_species, n_xi, n_x, n_x)}")
    return _apply_dense_x_blocks(op.mat, op.n_xi_for_x, op.mask_xi, f)


# ----------------------------------------------------------------------------
# 5. CollisionMatrices: one source of truth, multiple consumers
# ----------------------------------------------------------------------------


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class CollisionMatrices:
    """Precomputed collision x-blocks indexed by (species pair, Legendre mode).

    Uniform container for both collision operators (plan section 2.2: one
    source of truth, multiple consumers):

    * :meth:`apply` — the matrix-free path used by Krylov solvers.
    * :meth:`blocks_for_l` — the per-``l`` dense block extractor used by
      structured (Legendre-block) direct solvers and preconditioners.

    ``mat`` uses the "apply" sign convention: for ``collisionOperator = 0`` the
    entries are ``-nu_n * C`` (the Fortran PETSc Jacobian entries), and for
    ``collisionOperator = 1`` the Lorentz diagonal
    ``+nu_n * nuDHat * (l(l+1)+2K)/2`` — the same sign either way, since the
    Lorentz diagonal enters ``C`` with a minus sign.

    Attributes:
      mat: Dense x-blocks, ``(S, S, L, X, X)``.
      n_xi_for_x: Active Legendre modes per speed node, ``(X,)`` int32.
      mask_xi: Active-mode mask, ``(X, L)`` bool.
    """

    mat: jnp.ndarray
    n_xi_for_x: jnp.ndarray
    mask_xi: jnp.ndarray

    def tree_flatten(self):
        return (self.mat, self.n_xi_for_x, self.mask_xi), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mat, n_xi_for_x, mask_xi = children
        return cls(mat=mat, n_xi_for_x=n_xi_for_x, mask_xi=mask_xi)

    @property
    def n_species(self) -> int:
        return int(self.mat.shape[0])

    @property
    def n_xi(self) -> int:
        return int(self.mat.shape[2])

    @property
    def n_x(self) -> int:
        return int(self.mat.shape[3])

    def apply(self, f: jnp.ndarray) -> jnp.ndarray:
        """Matrix-free apply: ``f`` has shape ``(S, X, L, theta, zeta)``."""
        if f.ndim != 5:
            raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
        n_species, n_x, n_xi, _, _ = f.shape
        if (n_species, n_x) != (self.n_species, self.n_x) or n_xi != self.n_xi:
            raise ValueError(
                f"f has (S, X, L) = {(n_species, n_x, n_xi)}, "
                f"expected {(self.n_species, self.n_x, self.n_xi)}"
            )
        return _apply_dense_x_blocks(self.mat, self.n_xi_for_x, self.mask_xi, f)

    def blocks_for_l(self, ell: int) -> jnp.ndarray:
        """Dense x-blocks for one Legendre mode, ``(S, S, X, X)``.

        Rows of inactive speed nodes (``ell >= n_xi_for_x[ix]``) are zeroed so
        that stacking these blocks materializes exactly the operator that
        :meth:`apply` implements.
        """
        ell = int(ell)
        if not 0 <= ell < self.n_xi:
            raise ValueError(f"ell must be in [0, {self.n_xi}), got {ell}")
        row_mask = self.mask_xi[:, ell].astype(self.mat.dtype)  # (X,)
        return self.mat[:, :, ell, :, :] * row_mask[None, None, :, None]


def collision_matrices_from_fokker_planck(op: FokkerPlanck) -> CollisionMatrices:
    """Wrap a :class:`FokkerPlanck` operator (its blocks are already dense in x)."""
    return CollisionMatrices(mat=op.mat, n_xi_for_x=op.n_xi_for_x, mask_xi=op.mask_xi)


def collision_matrices_from_pitch_angle_scattering(op: PitchAngleScattering) -> CollisionMatrices:
    """Expand a :class:`PitchAngleScattering` operator to (diagonal) x-blocks.

    The Lorentz operator is diagonal in species and x, so the dense blocks are
    ``mat[a, a, l, i, i] = coef[a, i, l]`` and zero elsewhere;
    :meth:`CollisionMatrices.apply` then reproduces
    :func:`apply_pitch_angle_scattering` exactly (the extra terms in the block
    contraction are exact zeros).
    """
    coef = np.asarray(op.coef, dtype=np.float64)  # (S, X, L)
    n_species, n_x, n_xi = coef.shape
    mat = np.zeros((n_species, n_species, n_xi, n_x, n_x), dtype=np.float64)
    for a in range(n_species):
        for i in range(n_x):
            mat[a, a, :, i, i] = coef[a, i, :]
    return CollisionMatrices(
        mat=jnp.asarray(mat),
        n_xi_for_x=op.n_xi_for_x,
        mask_xi=op.mask_xi,
    )
