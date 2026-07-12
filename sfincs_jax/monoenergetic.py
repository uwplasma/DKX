"""Monoenergetic-database mode: (nu'/v, E_r/v) scans and energy convolution.

This module turns the RHSMode=3 monoenergetic transport-matrix solve into the
community-standard database workflow: scan the normalized collisionality
(``nuPrime``) / radial-electric-field (``EStar``) plane, store the four
monoenergetic transport coefficients in the benchmark normalization of
C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011), and reconstruct the
thermal (energy-integrated) transport matrices per species by convolution
with a local Maxwellian.  The monoenergetic formulation is that of
S.P. Hirshman, K.C. Shaing, W.I. van Rij, C.O. Beasley, and E.C. Crume,
Phys. Fluids 29, 2951 (1986).

Normalization conventions
=========================

The RHSMode=3 ``transportMatrix`` computed by :mod:`sfincs_jax.moments`
(``diagnostics.F90`` normalization) is converted to the dimensionless
coefficients used throughout the stellarator benchmark literature
(Beidler et al. 2011, section 5):

- ``D11* = D11 / D11^p`` with the equivalent-tokamak plateau value
  ``D11^p = (pi/4) v_d^2 R0 / (v iota)``,
- ``D31* = D31 / D31^b`` and ``D13* = D13 / D31^b`` with the banana-regime
  bootstrap value ``D31^b = (2/3) (v_d R0 / (iota eps_t)) (1 - f_c)``,
  ``f_c = 1 - 1.46 sqrt(eps_t)`` (the large-aspect-ratio circulating
  fraction used by the benchmark paper),
- ``D33* = D33 / D33^PS`` with the collisional parallel-conductivity value
  ``D33^PS = (v^2 / (3 nu)) <B^2> / B0^2``,

where ``v_d = m v^2 / (2 Z e R0 B0)``, ``R0 = GHat / B0OverBBar`` (Boozer
``G = R B_t`` evaluated with the reference field), ``eps_t = r / R0``, and
the effective radius follows the benchmark convention ``psiHat = B0 r^2 / 2``
(i.e. ``dr/dpsiHat = 1/(B0 r)``; for the analytic geometry schemes this is
exactly ``rHat = aHat sqrt(psiN)``).  ``nu`` is the pitch-angle deflection
frequency actually applied by the collision operator at the monoenergetic
speed node, ``nu = nu_n nuDHat(x0) vBar/RBar``, so ``D33* -> 1`` exactly in
the collisional limit.  The normalized collisionality reported alongside is
``nu_star = R0 nu / (iota v)`` and the normalized electric field is
``v_E = |E_r| / (v B0) = |EStar| iota eps_t / x0``.

The conversion from the ``transportMatrix`` entries ``L^S`` to the physical
matrix ``L^B`` defined by ``I_i = -n sum_j L^B_ij A_j`` (Beidler et al. 2011,
eq. (4), with thermodynamic forces ``A_1 = n'/n - Z e E_r/T - (3/2) T'/T``,
``A_2 = T'/T``, ``A_3 = -Z e B0 <E_par B>/(T <B^2>)``) is, in SFINCS "hat"
units (RBar vBar diffusivity units, single species, indices ``{1,3}`` for
RHSMode=3):

    L^B_11 = -(Delta^2/4) (dr/dpsiHat)^2 THat^2 GHat^2
             / (Z^2 B0 vtHat (GHat + iota IHat)) L^S_11
    L^B_13 = +(Delta/2) (THat GHat / (Z B0)) (dr/dpsiHat) L^S_13
    L^B_31 = -(Delta/2) (THat GHat / (Z B0)) (dr/dpsiHat) L^S_31
    L^B_33 = vtHat (GHat + iota IHat) / B0 L^S_33

with ``vtHat = sqrt(THat/mHat)``.  A monoenergetic (``Nx = 1``) run
evaluates the energy convolution below with the single node ``(x0, w0)``,
so the monoenergetic coefficients follow by dividing out the quadrature
factor ``(4/sqrt(pi)) w0 x0^2 exp(-x0^2)``.

Energy convolution
==================

The thermal transport matrix per species is the standard energy integral over
the local Maxwellian (Beidler et al. 2011, eq. before (5); ``K = x^2``):

    L_ij = (2/sqrt(pi)) int_0^inf dK sqrt(K) e^-K D_ij(K) h_i h_j
         = (4/sqrt(pi)) int_0^inf dx x^2 e^-x^2 D_ij(v(x)) h_i h_j,

with ``h_1 = h_3 = 1`` and ``h_2 = K = x^2``, and the monoenergetic
identities ``D12 = D21 = D22 = D11``, ``D23 = D13``, ``D32 = D31`` (the
energy weights, not the kernels, distinguish the entries).  At each speed
``x`` the kernel is looked up in the database at the equivalent
monoenergetic inputs

    nuPrime(x) = (GHat + iota IHat)/B0 * (x0/nuDHat_mono(x0))
                 * nuDHat_s(x) nu_n / (x vtHat_s),
    EStar(x)   = (alpha Delta / 2) (GHat/(iota B0)) dPhiHatdpsiHat
                 * x0 / (x vtHat_s),

which match the physical dimensionless groups ``nu(v)/v`` and ``E_r/(v B0)``
of the mono-energetic kinetic equation.  For pitch-angle-scattering
collisions with DKES trajectories the kinetic equation is exactly diagonal
in speed, so this reconstruction reproduces a full RHSMode=2 solve to
solver precision (see ``tests/test_monoenergetic_database.py``); with
energy-scattering (Fokker-Planck) collisions it is the standard
monoenergetic approximation.

Everything in the conversion and convolution path is pure JAX (jit/grad
safe): with a database built from a
:meth:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_fourier`
geometry and ``differentiable=True`` solves, ``jax.grad`` of any convolved
``L_ij`` entry with respect to a Boozer amplitude ``B_mn`` flows through the
scan, the conversion, and the convolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from .collisions import (  # noqa: E402
    make_pitch_angle_scattering_v3_operator,
    nu_d_hat_pitch_angle_scattering_v3,
)
from .constants import (  # noqa: E402
    RadialCoordinates,
    d_phi_hat_d_psi_hat_from_e_star,
    nu_n_from_nu_prime,
)
from .drift_kinetic import (  # noqa: E402
    KineticOperator,
    _geometry_and_radial,
    kinetic_operator_from_namelist,
)
from .inputs import SfincsInput, load_sfincs_input  # noqa: E402
from .moments import transport_matrix_from_state_vectors  # noqa: E402
from .solve import SolveResult, solve  # noqa: E402
from .writer import operator_containers  # noqa: E402

__all__ = [
    "DATABASE_SCHEMA",
    "MonoenergeticDatabase",
    "ThermalTransportMatrices",
    "energy_convolution",
    "load_database",
    "monoenergetic_database",
    "monoenergetic_database_from_operator",
    "monoenergetic_dstar_from_transport_matrix",
    "save_database",
]

DATABASE_SCHEMA = "sfincs_jax.monoenergetic_database.v1"

_FT_LARGE_ASPECT = 1.46  # 1 - f_c = 1.46 sqrt(eps_t), Beidler et al. 2011.


class DStarPoint(NamedTuple):
    """The four normalized monoenergetic coefficients at one (nuPrime, EStar).

    Attributes:
        d11_star: radial diffusion over the equivalent-tokamak plateau value.
        d13_star: Ware-pinch coefficient over the banana bootstrap value.
        d31_star: bootstrap coefficient over the banana bootstrap value
            (Onsager symmetry gives ``d13_star = -d31_star`` at ``EStar=0``).
        d33_star: parallel conductivity over its collisional limit.
        nu_star: ``R0 nu/(iota v)`` with ``nu`` the applied deflection rate.
    """

    d11_star: jnp.ndarray
    d13_star: jnp.ndarray
    d31_star: jnp.ndarray
    d33_star: jnp.ndarray
    nu_star: jnp.ndarray


@dataclass(frozen=True)
class MonoenergeticDatabase:
    """A (nuPrime, EStar) grid of normalized monoenergetic coefficients.

    Arrays may be NumPy or traced JAX values; the geometry scalars are the
    flux-surface constants of the deck the scan was run on.  ``deck_text``
    carries the input namelist for provenance (empty when built directly
    from an operator).

    Attributes:
        nu_prime: scan grid, shape ``(n_nu,)`` (SFINCS ``nuPrime``).
        e_star: scan grid, shape ``(n_er,)`` (SFINCS ``EStar``).
        d11_star: ``(n_nu, n_er)`` normalized radial-diffusion coefficient.
        d13_star: ``(n_nu, n_er)`` normalized Ware-pinch coefficient.
        d31_star: ``(n_nu, n_er)`` normalized bootstrap coefficient.
        d33_star: ``(n_nu, n_er)`` normalized parallel conductivity.
        x0: monoenergetic speed node (v3 uses ``x0 = 1``).
        w0: speed-quadrature weight of the node.
        nu_d_hat_x0: deflection-frequency shape factor ``nuDHat(x0)`` of the
            monoenergetic reference species (enters the ``nu`` conversion).
        delta: ``Delta`` (rhoBar/RBar) of the deck.
        alpha: ``alpha`` (e PhiBar/TBar) of the deck.
        g_hat, i_hat, iota, b0_over_bbar: Boozer flux functions.
        fsab_hat2: flux-surface average ``<BHat^2>``.
        r_hat: effective radius of the surface (``sqrt(2 psiHat/B0)``).
        deck_text: input namelist text (provenance).
    """

    nu_prime: Any
    e_star: Any
    d11_star: Any
    d13_star: Any
    d31_star: Any
    d33_star: Any
    x0: float
    w0: float
    nu_d_hat_x0: float
    delta: float
    alpha: float
    g_hat: float
    i_hat: float
    iota: float
    b0_over_bbar: float
    fsab_hat2: Any
    r_hat: float
    deck_text: str = ""

    @property
    def r_major(self) -> float:
        """Beidler's ``R0`` in RBar units: ``GHat / B0OverBBar``."""
        return float(self.g_hat) / float(self.b0_over_bbar)

    @property
    def eps_t(self) -> float:
        """Equivalent-tokamak inverse aspect ratio ``r/R0`` of the surface."""
        return float(self.r_hat) / self.r_major

    @property
    def nu_star(self) -> np.ndarray:
        """``R0 nu/(iota v)`` for each ``nu_prime`` grid value."""
        nu_eff = (
            np.asarray(self.nu_prime, dtype=np.float64)
            * float(self.b0_over_bbar)
            / (float(self.g_hat) + float(self.iota) * float(self.i_hat))
            * float(self.nu_d_hat_x0)
        )
        return self.r_major * nu_eff / (float(self.iota) * float(self.x0))

    @property
    def v_e(self) -> np.ndarray:
        """Normalized electric field ``|E_r/(v B0)|`` per ``e_star`` value."""
        return np.abs(
            np.asarray(self.e_star, dtype=np.float64)
            * float(self.iota)
            * self.eps_t
            / float(self.x0)
        )


@dataclass(frozen=True)
class ThermalTransportMatrices:
    """Energy-convolved thermal transport matrices, one per species.

    ``l_matrix[s, i, j]`` is ``L_ij`` of species ``s`` in the convention
    ``I_i = -n sum_j L_ij A_j`` (Beidler et al. 2011, eq. (4)) with the flows
    ``I_1 = <Gamma . grad r>``, ``I_2 = <Q . grad r>/T``,
    ``I_3 = <J . B>/(Z e B0)`` and forces ``A_1``, ``A_2``, ``A_3`` given in
    the module docstring.  Everything is in SFINCS hat units: the 2x2
    ``(1,2)`` block carries RBar vBar (diffusivity) units, the ``(i,3)`` /
    ``(3,j)`` couplings and ``L_33`` follow from the same normalization.

    Attributes:
        l_matrix: shape ``(n_species, 3, 3)``.
        x: speed nodes used for the convolution quadrature.
        x_weights: quadrature weights for ``int_0^inf dx``.
    """

    l_matrix: Any
    x: Any
    x_weights: Any

    def entry(self, i: int, j: int) -> Any:
        """1-based ``L_ij`` accessor, shape ``(n_species,)``."""
        return self.l_matrix[:, i - 1, j - 1]

    @property
    def l11(self) -> Any:
        return self.l_matrix[:, 0, 0]

    @property
    def l13(self) -> Any:
        return self.l_matrix[:, 0, 2]

    @property
    def l31(self) -> Any:
        return self.l_matrix[:, 2, 0]

    @property
    def l33(self) -> Any:
        return self.l_matrix[:, 2, 2]


# =============================================================================
# transportMatrix -> normalized monoenergetic coefficients
# =============================================================================


def monoenergetic_dstar_from_transport_matrix(
    transport_matrix: Any,
    *,
    nu_prime: Any,
    delta: Any,
    g_hat: Any,
    i_hat: Any,
    iota: Any,
    b0_over_bbar: Any,
    fsab_hat2: Any,
    r_hat: Any,
    x0: Any,
    w0: Any,
    nu_d_hat_x0: Any,
    z: Any = 1.0,
    m_hat: Any = 1.0,
    t_hat: Any = 1.0,
) -> DStarPoint:
    """Convert one RHSMode=3 ``transportMatrix`` to Beidler-normalized D*.

    Implements the hat-unit conversion chain documented in the module
    docstring: ``transportMatrix`` (rows/columns = radial, parallel) ->
    physical ``L^B`` -> monoenergetic ``D_ij`` (single-node de-convolution)
    -> normalized ``D*``.  Pure ``jnp``; safe under ``jit``/``grad``.

    Args:
        transport_matrix: shape ``(2, 2)`` RHSMode=3 matrix.
        nu_prime: the ``nuPrime`` the matrix was solved at.
        delta, g_hat, i_hat, iota, b0_over_bbar, fsab_hat2: deck constants
            (``fsab_hat2 = <BHat^2>``).
        r_hat: effective radius (``sqrt(2 psiHat / B0OverBBar)``).
        x0, w0: monoenergetic speed node and quadrature weight.
        nu_d_hat_x0: ``nuDHat(x0)`` of the monoenergetic species (the shape
            factor of the applied deflection frequency).
        z, m_hat, t_hat: species charge/mass/temperature of the run
            (v3 monoenergetic runs force ``1, 1, 1``).

    Returns:
        A :class:`DStarPoint`.
    """
    tm = jnp.asarray(transport_matrix, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    g_hat = jnp.asarray(g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(i_hat, dtype=jnp.float64)
    iota = jnp.asarray(iota, dtype=jnp.float64)
    b0 = jnp.asarray(b0_over_bbar, dtype=jnp.float64)
    fsab2 = jnp.asarray(fsab_hat2, dtype=jnp.float64)
    r_hat = jnp.asarray(r_hat, dtype=jnp.float64)
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    w0 = jnp.asarray(w0, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    m_hat = jnp.asarray(m_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)

    dr_dpsi = 1.0 / (b0 * r_hat)
    r_major = g_hat / b0
    eps_t = r_hat / r_major
    g_plus = g_hat + iota * i_hat
    vth = jnp.sqrt(t_hat / m_hat)
    v = x0 * vth

    # Single-node de-convolution of the Maxwellian energy integral.
    qfac = jnp.sqrt(jnp.pi) / (4.0 * w0 * x0 * x0 * jnp.exp(-x0 * x0))

    # transportMatrix -> L^B (hat units; module docstring).
    c11 = -(delta * delta / 4.0) * dr_dpsi**2 * t_hat**2 * g_hat**2 / (z * z * b0 * vth * g_plus)
    c13 = +(delta / 2.0) * (t_hat * g_hat / (z * b0)) * dr_dpsi
    c31 = -(delta / 2.0) * (t_hat * g_hat / (z * b0)) * dr_dpsi
    c33 = vth * g_plus / b0

    d11 = qfac * c11 * tm[0, 0]
    d13 = qfac * c13 * tm[0, 1]
    d31 = qfac * c31 * tm[1, 0]
    d33 = qfac * c33 * tm[1, 1]

    # Beidler et al. 2011 reference values at the node speed.
    v_d = delta * m_hat * v * v / (2.0 * z * r_major * b0)
    d11_p = (jnp.pi / 4.0) * v_d * v_d * r_major / (v * iota)
    d31_b = (2.0 / 3.0) * (v_d * r_major / (iota * eps_t)) * (_FT_LARGE_ASPECT * jnp.sqrt(eps_t))
    nu_eff = jnp.asarray(nu_prime, dtype=jnp.float64) * b0 / g_plus * jnp.asarray(
        nu_d_hat_x0, dtype=jnp.float64
    )
    d33_ps = v * v * fsab2 / (3.0 * nu_eff * b0 * b0)
    nu_star = r_major * nu_eff / (iota * v)

    return DStarPoint(
        d11_star=d11 / d11_p,
        d13_star=d13 / d31_b,
        d31_star=d31 / d31_b,
        d33_star=d33 / d33_ps,
        nu_star=nu_star,
    )


# =============================================================================
# The (nuPrime, EStar) scan
# =============================================================================


def _operator_at(
    op: KineticOperator,
    *,
    nu_prime: float,
    e_star: float,
    g_hat: float,
    i_hat: float,
    iota: float,
    b0_over_bbar: float,
) -> KineticOperator:
    """The deck operator with only the (nuPrime, EStar) coefficients replaced.

    Rebuilds the pitch-angle-scattering block with the overwritten ``nu_n``
    (``sfincs_main.F90``'s RHSMode=3 rule) and the ExB coefficients with the
    overwritten ``dPhiHatdpsiHat``; every other field (geometry, grids,
    species, layout) is shared, so a scan re-solves the same discretization
    with new coefficients only.
    """
    if op.rhs_mode != 3 or op.pas is None:
        raise ValueError("_operator_at expects an RHSMode=3 operator with PAS collisions.")
    nu_n = nu_n_from_nu_prime(
        nu_prime=float(nu_prime), b0_over_bbar=b0_over_bbar, g_hat=g_hat, i_hat=i_hat, iota=iota
    )
    pas = make_pitch_angle_scattering_v3_operator(
        x=op.x,
        z_s=op.z_s,
        m_hats=op.m_hat,
        n_hats=op.n_hat,
        t_hats=op.t_hat,
        nu_n=nu_n,
        krook=float(op.pas.krook),
        n_xi_for_x=op.n_xi_for_x,
        n_xi=op.n_xi,
    )
    dphi = d_phi_hat_d_psi_hat_from_e_star(
        e_star=float(e_star),
        alpha=float(op.alpha),
        delta=float(op.delta),
        iota=iota,
        b0_over_bbar=b0_over_bbar,
        g_hat=g_hat,
    )
    return replace(
        op,
        pas=pas,
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi, dtype=jnp.float64),
        with_exb=bool(dphi != 0.0),
    )


def monoenergetic_database_from_operator(
    op: KineticOperator,
    nu_prime_grid: Iterable[float],
    e_star_grid: Iterable[float] = (0.0,),
    *,
    g_hat: float,
    i_hat: float,
    iota: float,
    b0_over_bbar: float,
    r_hat: float,
    solve_method: str = "auto",
    tol: float = 1e-10,
    differentiable: bool = False,
    deck_text: str = "",
    emit: Callable[[str], None] | None = None,
) -> MonoenergeticDatabase:
    """Scan (nuPrime, EStar) on a prepared RHSMode=3 operator.

    The geometry/grid fields of ``op`` are reused for every point; only the
    collisionality and ExB coefficients are replaced per point
    (:func:`_operator_at`).  With ``differentiable=True`` and an operator
    whose geometry fields are traced JAX values (e.g. via
    ``FluxSurfaceGeometry.from_fourier`` + ``dataclasses.replace``), the
    returned coefficient arrays are traced end to end, enabling
    ``jax.grad`` of any database entry (or of an
    :func:`energy_convolution` output) with respect to geometry inputs.

    Args:
        op: RHSMode=3 operator built from the deck (PAS collisions).
        nu_prime_grid: ``nuPrime`` values (must be nonzero).
        e_star_grid: ``EStar`` values (default: the single value 0).
        g_hat, i_hat, iota, b0_over_bbar: deck flux functions (static
            Python floats; also used for the ``nu_n``/``dPhiHat`` rules).
        r_hat: effective radius of the surface (``sqrt(2 psiHat/B0)``).
        solve_method, tol: forwarded to :func:`sfincs_jax.solve.solve`.
        differentiable: use implicit-differentiation solves and skip host
            convergence checks (keeps the scan traceable).
        deck_text: provenance string stored on the database.
        emit: optional per-point progress sink (e.g. ``print``).

    Returns:
        A :class:`MonoenergeticDatabase` (arrays are ``jnp`` values; wrap
        with :func:`save_database` to store as ``.npz``).
    """
    nu_values = [float(v) for v in nu_prime_grid]
    er_values = [float(v) for v in e_star_grid]
    if not nu_values or not er_values:
        raise ValueError("nu_prime_grid and e_star_grid must be non-empty.")
    if any(v == 0.0 for v in nu_values):
        raise ValueError("RHSMode=3 requires nonzero nuPrime at every scan point.")

    x0 = float(np.asarray(op.x)[0])
    w0 = float(np.asarray(op.x_weights)[0])
    nu_d_hat_x0 = float(
        np.asarray(
            nu_d_hat_pitch_angle_scattering_v3(
                x=op.x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
            )
        )[0, 0]
    )

    layout, vgrid, _surface, species = operator_containers(op)
    rows: dict[str, list[list[jnp.ndarray]]] = {k: [] for k in ("d11", "d13", "d31", "d33")}
    warm: SolveResult | None = None
    for nu_prime in nu_values:
        cols: dict[str, list[jnp.ndarray]] = {k: [] for k in rows}
        for e_star in er_values:
            op_point = _operator_at(
                op,
                nu_prime=nu_prime,
                e_star=e_star,
                g_hat=g_hat,
                i_hat=i_hat,
                iota=iota,
                b0_over_bbar=b0_over_bbar,
            )
            rhs = jnp.stack([op_point.rhs(which_rhs) for which_rhs in (1, 2)], axis=1)
            result = solve(
                op_point,
                rhs,
                method=solve_method,
                tol=tol,
                differentiable=differentiable,
                x0=None if warm is None else warm.x,
                recycle=None if warm is None else warm.recycle,
            )
            if not (differentiable or result.converged):
                raise RuntimeError(
                    f"monoenergetic scan point (nuPrime={nu_prime}, EStar={e_star}) did not "
                    f"converge (method={result.method}, "
                    f"residuals={np.asarray(result.residual_norms)!r})"
                )
            warm = None if differentiable else result
            _, _, surface, _ = operator_containers(op_point)
            tm = transport_matrix_from_state_vectors(
                layout, vgrid, surface, species, jnp.asarray(result.x).T,
                rhs_mode=3, delta=op_point.delta, alpha=op_point.alpha,
                g_hat=g_hat, i_hat=i_hat, iota=iota, b0_over_bbar=b0_over_bbar,
            )  # fmt: skip
            point = monoenergetic_dstar_from_transport_matrix(
                tm,
                nu_prime=nu_prime,
                delta=op.delta,
                g_hat=g_hat,
                i_hat=i_hat,
                iota=iota,
                b0_over_bbar=b0_over_bbar,
                fsab_hat2=op.fsab_hat2,
                r_hat=r_hat,
                x0=x0,
                w0=w0,
                nu_d_hat_x0=nu_d_hat_x0,
                z=op.z_s[0],
                m_hat=op.m_hat[0],
                t_hat=op.t_hat[0],
            )
            cols["d11"].append(point.d11_star)
            cols["d13"].append(point.d13_star)
            cols["d31"].append(point.d31_star)
            cols["d33"].append(point.d33_star)
            if emit is not None:
                emit(
                    f"  nuPrime={nu_prime:12.5e}  EStar={e_star:12.5e}  "
                    f"D11*={float(point.d11_star):12.5e}  D31*={float(point.d31_star):12.5e}  "
                    f"D13*={float(point.d13_star):12.5e}  D33*={float(point.d33_star):12.5e}"
                )
        for key in rows:
            rows[key].append(cols[key])

    def _grid(key: str) -> jnp.ndarray:
        return jnp.stack([jnp.stack(col) for col in rows[key]])

    return MonoenergeticDatabase(
        nu_prime=jnp.asarray(nu_values, dtype=jnp.float64),
        e_star=jnp.asarray(er_values, dtype=jnp.float64),
        d11_star=_grid("d11"),
        d13_star=_grid("d13"),
        d31_star=_grid("d31"),
        d33_star=_grid("d33"),
        x0=x0,
        w0=w0,
        nu_d_hat_x0=nu_d_hat_x0,
        delta=float(op.delta),
        alpha=float(op.alpha),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        iota=float(iota),
        b0_over_bbar=float(b0_over_bbar),
        fsab_hat2=op.fsab_hat2,
        r_hat=float(r_hat),
        deck_text=deck_text,
    )


def _mono_raw_namelist(inp: SfincsInput, raw: Any) -> Any:
    """The deck's raw namelist with the RHSMode=3 monoenergetic forcing applied."""
    groups = {name: dict(values) for name, values in raw.groups.items()}
    groups.setdefault("general", {})["RHSMODE"] = 3
    phys = groups.setdefault("physicsparameters", {})
    phys.setdefault("NUPRIME", 1.0)
    phys["COLLISIONOPERATOR"] = 1
    phys["USEDKESEXBDRIFT"] = True
    phys["INCLUDEXDOTTERM"] = False
    phys["INCLUDEELECTRICFIELDTERMINXIDOT"] = False
    phys["INCLUDEPHI1"] = False
    groups.setdefault("resolutionparameters", {})["NX"] = 1
    groups.setdefault("othernumericalparameters", {})["NXI_FOR_X_OPTION"] = 0
    return replace(raw, groups=groups)


def _grids_for_mono(inp: SfincsInput, raw: Any) -> Any:
    """Monoenergetic grids for the deck (RHSMode=3 forcing: ``Nx=1``)."""
    from .run import _grids_from_input  # noqa: PLC0415 - avoid import cycle at module load

    res = replace(inp.resolution, n_x=1)
    gen = replace(inp.general, rhs_mode=3)
    other = replace(inp.other, n_xi_for_x_option=0)
    return _grids_from_input(replace(inp, resolution=res, general=gen, other=other), raw)


def monoenergetic_database(
    input_namelist: str | Path,
    nu_prime_grid: Iterable[float],
    e_star_grid: Iterable[float] = (0.0,),
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    emit: Callable[[str], None] | None = None,
) -> MonoenergeticDatabase:
    """Build a monoenergetic database from a SFINCS input namelist.

    The deck provides the geometry, resolution, and numerics; it does not
    need ``RHSMode = 3`` itself — the v3 monoenergetic forcing (``Nx = 1``,
    pitch-angle scattering, DKES ExB drift, no ``xDot``/Er-``xiDot`` terms)
    is applied automatically, exactly as ``validateInput.F90`` does for
    RHSMode=3 decks.  Each grid point then re-solves the same discretization
    with the (nuPrime, EStar) coefficient overwrites only.

    Args:
        input_namelist: path to the input namelist.
        nu_prime_grid: ``nuPrime`` values (nonzero).
        e_star_grid: ``EStar`` values (default ``(0.0,)``).
        solve_method, tol: forwarded to :func:`sfincs_jax.solve.solve`.
        emit: optional progress sink (e.g. ``print``).

    Returns:
        A :class:`MonoenergeticDatabase` with the deck text as provenance.
    """
    path = Path(input_namelist)
    inp = load_sfincs_input(path)
    if inp.raw is None:
        raise ValueError("monoenergetic_database requires an input parsed from a namelist file.")
    raw = _mono_raw_namelist(inp, inp.raw)
    op = kinetic_operator_from_namelist(raw)
    grids = _grids_for_mono(inp, raw)
    geom, radial = _geometry_and_radial(nml=raw, grids=grids)

    # VMEC-placeholder flux functions are replaced by the surface averages,
    # mirroring the operator builder and the transport-matrix assembly.
    from .writer import _effective_flux_functions  # noqa: PLC0415

    b0_eff, g_eff, i_eff = _effective_flux_functions(op, geom)

    r_hat = _effective_r_hat(radial, b0_eff)
    return monoenergetic_database_from_operator(
        op,
        nu_prime_grid,
        e_star_grid,
        g_hat=g_eff,
        i_hat=i_eff,
        iota=float(geom.iota),
        b0_over_bbar=b0_eff,
        r_hat=r_hat,
        solve_method=solve_method,
        tol=tol,
        deck_text=path.read_text(),
        emit=emit,
    )


def _effective_r_hat(radial: RadialCoordinates, b0_over_bbar: float) -> float:
    """Benchmark effective radius ``sqrt(2 psiHat / B0)`` of the surface."""
    return float(np.sqrt(2.0 * abs(float(radial.psi_hat)) / abs(float(b0_over_bbar))))


# =============================================================================
# Energy convolution: thermal transport matrices
# =============================================================================


def _bilinear(
    grid_x: jnp.ndarray, grid_y: jnp.ndarray, table: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Clamped bilinear interpolation of ``table[(x, y)]`` (pure jnp).

    ``grid_x``/``grid_y`` must be strictly increasing.  1-length axes are
    treated as constant.  ``x``/``y`` may be arrays (broadcast together).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    def axis_weights(grid: jnp.ndarray, value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        n = grid.shape[0]
        if n == 1:
            zero = jnp.zeros_like(value, dtype=jnp.int32)
            return zero, jnp.zeros_like(value)
        idx = jnp.clip(jnp.searchsorted(grid, value, side="right") - 1, 0, n - 2)
        left = grid[idx]
        right = grid[idx + 1]
        frac = jnp.clip((value - left) / (right - left), 0.0, 1.0)
        return idx.astype(jnp.int32), frac

    ix, fx = axis_weights(grid_x, x)
    iy, fy = axis_weights(grid_y, y)
    ix1 = jnp.minimum(ix + 1, grid_x.shape[0] - 1)
    iy1 = jnp.minimum(iy + 1, grid_y.shape[0] - 1)
    v00 = table[ix, iy]
    v01 = table[ix, iy1]
    v10 = table[ix1, iy]
    v11 = table[ix1, iy1]
    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v10 * fx * (1.0 - fy)
        + v11 * fx * fy
    )


def _dstar_lookup(
    db: MonoenergeticDatabase, nu_prime: jnp.ndarray, e_star: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interpolate the D* tables at (nuPrime, EStar) points.

    ``d11_star``/``d33_star`` are positive and near power laws in
    collisionality, so they are interpolated log-log in ``nuPrime``;
    ``d31_star``/``d13_star`` change sign and are interpolated linearly in
    ``log(nuPrime)``.  The ``EStar`` axis is linear.  Queries are clamped to
    the grid (the tests exercise in-range lookups; extrapolation is
    deliberately flat).
    """
    log_nu = jnp.log(jnp.asarray(db.nu_prime, dtype=jnp.float64))
    er_grid = jnp.asarray(db.e_star, dtype=jnp.float64)
    q_nu = jnp.log(jnp.asarray(nu_prime, dtype=jnp.float64))
    q_er = jnp.asarray(e_star, dtype=jnp.float64)

    tiny = jnp.asarray(1e-300, dtype=jnp.float64)
    d11 = jnp.exp(
        _bilinear(log_nu, er_grid, jnp.log(jnp.maximum(jnp.asarray(db.d11_star), tiny)), q_nu, q_er)
    )
    d33 = jnp.exp(
        _bilinear(log_nu, er_grid, jnp.log(jnp.maximum(jnp.asarray(db.d33_star), tiny)), q_nu, q_er)
    )
    d13 = _bilinear(log_nu, er_grid, jnp.asarray(db.d13_star, dtype=jnp.float64), q_nu, q_er)
    d31 = _bilinear(log_nu, er_grid, jnp.asarray(db.d31_star, dtype=jnp.float64), q_nu, q_er)
    return d11, d13, d31, d33


def energy_convolution(
    db: MonoenergeticDatabase,
    *,
    z_s: Any,
    m_hats: Any,
    t_hats: Any,
    n_hats: Any,
    nu_n: Any,
    dphi_hat_dpsi_hat: Any = 0.0,
    x: Any = None,
    x_weights: Any = None,
    n_x: int = 64,
    x_max: float = 5.0,
) -> ThermalTransportMatrices:
    """Thermal transport matrices from the monoenergetic database.

    Evaluates the Maxwellian energy integrals of the module docstring for
    each species: at every quadrature speed ``x`` the database is queried at
    the equivalent (nuPrime, EStar) point, un-normalized to physical
    (hat-unit) monoenergetic coefficients for that species and speed, and
    integrated with weights ``h = (1, x^2, 1)``.

    Args:
        db: the monoenergetic database (its geometry constants are used).
        z_s, m_hats, t_hats, n_hats: species parameters, shape ``(S,)``
            (SFINCS hat units).  The full species set enters the deflection
            frequency of each species (field-particle sum).
        nu_n: the deck normalized collisionality ``nu_n`` (reference).
        dphi_hat_dpsi_hat: radial-electric-field input ``dPhiHatdpsiHat``
            (default 0; converted per species/speed to the ``EStar`` axis).
        x, x_weights: optional speed nodes/weights for ``int_0^inf dx``
            (e.g. a deck's speed grid).  Default: Gauss-Legendre on
            ``[0, x_max]`` with ``n_x`` nodes.
        n_x, x_max: default quadrature parameters.

    Returns:
        A :class:`ThermalTransportMatrices` with ``l_matrix`` of shape
        ``(S, 3, 3)``.
    """
    z_arr = jnp.atleast_1d(jnp.asarray(z_s, dtype=jnp.float64))
    m_arr = jnp.atleast_1d(jnp.asarray(m_hats, dtype=jnp.float64))
    t_arr = jnp.atleast_1d(jnp.asarray(t_hats, dtype=jnp.float64))
    n_arr = jnp.atleast_1d(jnp.asarray(n_hats, dtype=jnp.float64))
    if not (z_arr.shape == m_arr.shape == t_arr.shape == n_arr.shape):
        raise ValueError("z_s, m_hats, t_hats, n_hats must share the (S,) shape.")

    if x is None or x_weights is None:
        if (x is None) != (x_weights is None):
            raise ValueError("Provide both x and x_weights, or neither.")
        nodes, weights = np.polynomial.legendre.leggauss(int(n_x))
        x_q = jnp.asarray(0.5 * float(x_max) * (nodes + 1.0), dtype=jnp.float64)
        w_q = jnp.asarray(0.5 * float(x_max) * weights, dtype=jnp.float64)
    else:
        x_q = jnp.asarray(x, dtype=jnp.float64)
        w_q = jnp.asarray(x_weights, dtype=jnp.float64)

    delta = jnp.asarray(db.delta, dtype=jnp.float64)
    alpha = jnp.asarray(db.alpha, dtype=jnp.float64)
    g_hat = jnp.asarray(db.g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(db.i_hat, dtype=jnp.float64)
    iota = jnp.asarray(db.iota, dtype=jnp.float64)
    b0 = jnp.asarray(db.b0_over_bbar, dtype=jnp.float64)
    fsab2 = jnp.asarray(db.fsab_hat2, dtype=jnp.float64)
    g_plus = g_hat + iota * i_hat
    r_major = g_hat / b0
    eps_t = jnp.asarray(db.r_hat, dtype=jnp.float64) / r_major
    x0 = jnp.asarray(db.x0, dtype=jnp.float64)
    nu_d_ref = jnp.asarray(db.nu_d_hat_x0, dtype=jnp.float64)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)
    dphi = jnp.asarray(dphi_hat_dpsi_hat, dtype=jnp.float64)

    # Deflection frequency shape per species over the quadrature grid, (S, X).
    nu_d_hat = nu_d_hat_pitch_angle_scattering_v3(
        x=x_q, z_s=z_arr, m_hats=m_arr, n_hats=n_arr, t_hats=t_arr
    )

    quad = (4.0 / jnp.sqrt(jnp.pi)) * w_q * x_q * x_q * jnp.exp(-x_q * x_q)  # (X,)

    matrices = []
    for s in range(int(z_arr.shape[0])):
        z = z_arr[s]
        m_hat = m_arr[s]
        t_hat = t_arr[s]
        vth = jnp.sqrt(t_hat / m_hat)
        v = x_q * vth  # (X,)

        nu_prime_x = g_plus / b0 * (x0 / nu_d_ref) * nu_n * nu_d_hat[s] / v
        e_star_x = (alpha * delta / 2.0) * (g_hat / (iota * b0)) * dphi * x0 / v

        d11s, d13s, d31s, d33s = _dstar_lookup(db, nu_prime_x, e_star_x)

        v_d = delta * m_hat * v * v / (2.0 * z * r_major * b0)
        d11_p = (jnp.pi / 4.0) * v_d * v_d * r_major / (v * iota)
        d31_b = (2.0 / 3.0) * (v_d * r_major / (iota * eps_t)) * (
            _FT_LARGE_ASPECT * jnp.sqrt(eps_t)
        )
        d33_ps = v * v * fsab2 / (3.0 * (nu_n * nu_d_hat[s]) * b0 * b0)

        d11 = d11s * d11_p
        d13 = d13s * d31_b
        d31 = d31s * d31_b
        d33 = d33s * d33_ps

        h2 = x_q * x_q
        def integral(kernel: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(quad * kernel * weight)

        one = jnp.ones_like(x_q)
        l_s = jnp.stack(
            [
                jnp.stack([integral(d11, one), integral(d11, h2), integral(d13, one)]),
                jnp.stack([integral(d11, h2), integral(d11, h2 * h2), integral(d13, h2)]),
                jnp.stack([integral(d31, one), integral(d31, h2), integral(d33, one)]),
            ]
        )
        matrices.append(l_s)

    return ThermalTransportMatrices(
        l_matrix=jnp.stack(matrices), x=x_q, x_weights=w_q
    )


# =============================================================================
# Save / load (compact npz database format)
# =============================================================================


def save_database(path: str | Path, db: MonoenergeticDatabase, *, overwrite: bool = True) -> Path:
    """Write a database to ``.npz`` (grids, coefficients, metadata, deck).

    Uses the generic :func:`sfincs_jax.io.write_sfincs_npz` serializer with
    Python-native array layout.  The schema is versioned via the
    ``database_schema`` field; :func:`load_database` round-trips exactly.
    """
    from .io import write_sfincs_npz  # noqa: PLC0415

    path = Path(path)
    scalars = {
        "x0": db.x0,
        "w0": db.w0,
        "nu_d_hat_x0": db.nu_d_hat_x0,
        "delta": db.delta,
        "alpha": db.alpha,
        "g_hat": db.g_hat,
        "i_hat": db.i_hat,
        "iota": db.iota,
        "b0_over_bbar": db.b0_over_bbar,
        "fsab_hat2": float(np.asarray(db.fsab_hat2)),
        "r_hat": db.r_hat,
    }
    data: dict[str, Any] = {
        "database_schema": np.asarray(DATABASE_SCHEMA),
        "nu_prime": np.asarray(db.nu_prime, dtype=np.float64),
        "e_star": np.asarray(db.e_star, dtype=np.float64),
        "d11_star": np.asarray(db.d11_star, dtype=np.float64),
        "d13_star": np.asarray(db.d13_star, dtype=np.float64),
        "d31_star": np.asarray(db.d31_star, dtype=np.float64),
        "d33_star": np.asarray(db.d33_star, dtype=np.float64),
        "nu_star": np.asarray(db.nu_star, dtype=np.float64),
        "v_e": np.asarray(db.v_e, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(scalars, sort_keys=True)),
        "deck_text": np.asarray(db.deck_text),
    }
    write_sfincs_npz(path=path, data=data, fortran_layout=False, overwrite=overwrite)
    return path


def load_database(path: str | Path) -> MonoenergeticDatabase:
    """Read a database written by :func:`save_database`."""
    from .io import decode_if_bytes  # noqa: PLC0415

    with np.load(Path(path), allow_pickle=False) as data:
        schema = str(decode_if_bytes(data["database_schema"][()]))
        if schema != DATABASE_SCHEMA:
            raise ValueError(f"Unsupported database schema {schema!r} (expected {DATABASE_SCHEMA!r}).")
        meta = json.loads(str(decode_if_bytes(data["metadata_json"][()])))
        return MonoenergeticDatabase(
            nu_prime=np.asarray(data["nu_prime"], dtype=np.float64),
            e_star=np.asarray(data["e_star"], dtype=np.float64),
            d11_star=np.asarray(data["d11_star"], dtype=np.float64),
            d13_star=np.asarray(data["d13_star"], dtype=np.float64),
            d31_star=np.asarray(data["d31_star"], dtype=np.float64),
            d33_star=np.asarray(data["d33_star"], dtype=np.float64),
            x0=float(meta["x0"]),
            w0=float(meta["w0"]),
            nu_d_hat_x0=float(meta["nu_d_hat_x0"]),
            delta=float(meta["delta"]),
            alpha=float(meta["alpha"]),
            g_hat=float(meta["g_hat"]),
            i_hat=float(meta["i_hat"]),
            iota=float(meta["iota"]),
            b0_over_bbar=float(meta["b0_over_bbar"]),
            fsab_hat2=float(meta["fsab_hat2"]),
            r_hat=float(meta["r_hat"]),
            deck_text=str(decode_if_bytes(data["deck_text"][()])),
        )
