"""The v3 drift-kinetic operator as a single consolidated ``KineticOperator``.

This module is the Phase-3.2 consolidation target for the drift-kinetic-equation
(DKE) physics that today lives across ``sfincs_jax/operators/profile_*.py``
(collisionless streaming/mirror, ExB, radial-electric-field xDot/xiDot terms,
collision wiring, constraint/source bordering, and the RHS drives).  It becomes
``sfincs_jax/dke.py`` at the v2 purge.

Fortran correspondence (paths relative to ``sfincs/fortran/version3``):

- ``populateMatrix.F90`` — every operator term applied by :meth:`KineticOperator.apply`:
  streaming (``ddtheta``/``ddzeta`` blocks coupling L±1), the mirror term (L±1),
  the ExB drift d/dtheta and d/dzeta terms, the non-standard d/dxi term
  associated with E_r (L, L±2), the collisionless d/dx term associated with E_r
  (L, L±2, dense in x), pitch-angle-scattering and Fokker–Planck collisions
  (diagonal in L), and the source/constraint rows and columns
  (``constraintScheme`` 1 and 2).
- ``evaluateResidual.F90`` — the inhomogeneous drives produced by
  :meth:`KineticOperator.rhs` (the ``dot(psi) dfM/dpsi`` gradient drive on
  L=0/L=2 and the inductive ``EParallelHat`` drive on L=1).
- ``indices.F90`` — the state-vector layout (see below).
- ``sfincs_main.F90`` lines 151-154 — the RHSMode=3 (monoenergetic)
  ``nuPrime``/``EStar`` overwrites of ``nu_n`` and ``dPhiHatdpsiHat``.

State-vector layout (matches v3 ``indices.F90`` with ``BLOCK_F`` first)::

    [ f(species, x, L, theta, zeta) row-major  |  constraint unknowns ]

- The distribution block flattens ``(n_species, n_x, n_xi, n_theta, n_zeta)``
  in C order.
- ``constraintScheme=1`` (default for Fokker-Planck): two source unknowns per
  species (particle, energy), constraint rows enforce zero density and pressure
  moments.
- ``constraintScheme=2`` (default for pitch-angle scattering): one L=0 source
  unknown per (species, x), constraint rows enforce ``<f1>=0`` at each x.
- ``includePhi1`` rows (quasineutrality + lambda) are **not** yet part of this
  consolidated operator; :meth:`KineticOperator.from_namelist` raises
  ``NotImplementedError`` for them (see the "deferred" list below).

Coefficient provenance: everything is built from the committed consolidated
modules — :mod:`sfincs_jax.phase_space` (grids, differentiation matrices, Legendre
couplings), :mod:`sfincs_jax.magnetic_geometry` (flux-surface geometry for
geometrySchemes 1/2/3/4/5/11/12), :mod:`sfincs_jax.species` (charges, profiles,
psiHat-gradients), and :mod:`sfincs_jax.constants` (normalizations and radial
conversions).  Collision matrices are built by the stable
:mod:`sfincs_jax.physics.collisions` (they migrate to ``collisions`` when
that consolidation lands).

Three consumers, one source of truth (plan §2.2):

1. :meth:`KineticOperator.apply` — matrix-free matvec for Krylov solvers,
   bit-compatible with ``operators.profile_system.apply_v3_full_system_operator``.
2. :meth:`KineticOperator.legendre_blocks` / :meth:`to_block_tridiagonal` — the
   analytic (probing-free) block-tridiagonal-in-L representation used by the
   structured direct solver and preconditioners.  The terms know their own
   L-coupling: streaming/mirror couple L±1 with the
   :func:`sfincs_jax.phase_space.legendre_coupling_lower` /
   :func:`~sfincs_jax.phase_space.legendre_coupling_upper` factors, ExB is diagonal
   in L, and pitch-angle scattering is diagonal in L (eigenvalues
   ``l(l+1)/2``).
3. :meth:`KineticOperator.rhs` — the v3 drives, including the internal
   ``whichRHS`` gradient/E_parallel overwrites used by RHSMode 2 and 3
   transport-matrix loops.

Deferred (raise ``NotImplementedError`` in :meth:`from_namelist`, tracked for a
follow-up pass; the old ``operators/profile_*`` code paths remain authoritative
for them until then):

- ``includePhi1`` (quasineutrality block, lambda row, Phi1-in-kinetic-equation
  and Phi1-in-collision-operator couplings, ``readExternalPhi1``);
- ``magneticDriftScheme != 0`` (tangential magnetic drifts and their upwinded
  stencils);
- ``constraintScheme`` 3 and 4;
- ``collisionOperator`` other than 0 (Fokker-Planck) and 1 (pitch-angle
  scattering);
- mapped x-grids (``xGridScheme >= 50``) and ``xDotDerivativeScheme != 0``;
- ``geometryScheme`` 13 (namelist Boozer spectrum): the differentiable
  :meth:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_fourier` builds
  this geometry, but :meth:`from_namelist` does not yet parse ``bmnc``/``bmns``
  from the deck and route them (the analytic schemes {1,2,3,4} and the
  file-based schemes {5,11,12} are the namelist-wired set).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import tree_util as jtu  # noqa: E402

from sfincs_jax.constants import (  # noqa: E402
    DEFAULT_DELTA,
    DEFAULT_NU_N,
    RadialCoordinates,
    d_phi_hat_d_psi_hat_from_e_star,
    nu_n_from_nu_prime,
)
from sfincs_jax.magnetic_geometry import (  # noqa: E402
    FluxSurfaceGeometry,
    psi_a_hat_from_wout,
    read_boozer_bc,
    read_vmec_wout,
    selected_r_n_from_bc,
    vmec_radial_interpolation,
)
from sfincs_jax.phase_space import (  # noqa: E402
    Grids,
    legendre_coupling_lower,
    legendre_coupling_upper,
    make_grids,
)
from sfincs_jax.input_compat import (  # noqa: E402
    effective_equilibrium_file,
    effective_psi_a_hat,
    effective_psi_n_wish,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
)
from sfincs_jax.paths import resolve_existing_path  # noqa: E402

# Collision matrices: the stable, committed implementation.  This import moves
# to `sfincs_jax.collisions` when that consolidation lands (same public API).
from sfincs_jax.physics.collisions import (  # noqa: E402
    FokkerPlanckV3Operator,
    PitchAngleScatteringV3Operator,
    apply_fokker_planck_v3,
    apply_pitch_angle_scattering_v3,
    make_fokker_planck_v3_operator,
    make_pitch_angle_scattering_v3_operator,
)
from sfincs_jax.species import SpeciesSet, species_set_from_namelist  # noqa: E402

__all__ = [
    "KineticOperator",
    "LegendreBlocks",
    "kinetic_operator_from_namelist",
]


# =============================================================================
# Small namelist-access helpers (same conventions as readInput.F90 defaults).
# =============================================================================


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def _get_bool(group: dict, key: str, default: bool = False) -> bool:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return bool(v)


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi: int) -> jnp.ndarray:
    """(X, L) float mask of retained Legendre modes (createGrids.F90 Nxi_for_x)."""
    ell = jnp.arange(int(n_xi), dtype=jnp.int32)[None, :]
    return (ell < n_xi_for_x.astype(jnp.int32)[:, None]).astype(jnp.float64)


def _ix_min(point_at_x0: bool) -> int:
    """First speed row carrying DKE equations (populateMatrix.F90 ixMin)."""
    return 1 if point_at_x0 else 0


class LegendreBlocks(NamedTuple):
    """Dense (theta*zeta) blocks of one Legendre row of the f-block operator.

    ``lower`` maps mode ``l-1`` to ``l``, ``diag`` maps ``l`` to ``l``, and
    ``upper`` maps ``l+1`` to ``l``.  All have shape
    ``(n_species, n_x, n_theta*n_zeta, n_theta*n_zeta)``; ``lower`` is zero for
    ``l=0`` and ``upper`` is zero for ``l=n_xi-1``.
    """

    lower: jnp.ndarray
    diag: jnp.ndarray
    upper: jnp.ndarray


# =============================================================================
# The operator
# =============================================================================


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class KineticOperator:
    """Matrix-free v3 drift-kinetic operator with structured-block extractors.

    One instance holds the per-term coefficient arrays for a single flux
    surface and applies the linear operator of the v3 system
    ``populateMatrix.F90`` (whichMatrix=3 content for the supported subset:
    no Phi1, no tangential magnetic drifts).

    The supported term inventory (see module docstring for the Fortran map):

    ====================  =========  ==========================================
    term                  L-coupling  notes
    ====================  =========  ==========================================
    streaming             L±1        ``x_j (l±-coupling) v_|| b·∇``; diagonal in x
    mirror                L±1        ``-x_j (b·∇B)/(2B)``; diagonal in x
    ExB (d/dtheta,dzeta)  diag       present iff the kinetic dPhiHat/dpsiHat ≠ 0
    Er xiDot              L, L±2     ``includeElectricFieldTermInXiDot``
    Er xDot               L, L±2     ``includeXDotTerm``; dense in x (x ddx)
    PAS collisions        diag       ``collisionOperator=1``; ``nu_n nuD l(l+1)/2``
    FP collisions         diag       ``collisionOperator=0``; dense in (species,x)
    sources/constraints   n/a        bordered rows/cols, ``constraintScheme`` 1/2
    ====================  =========  ==========================================
    """

    # ---- static layout / option flags (pytree aux data) ----
    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    rhs_mode: int
    constraint_scheme: int
    point_at_x0: bool
    use_dkes_exb: bool
    with_exb: bool
    with_er_xidot: bool
    with_er_xdot: bool

    # ---- speed / angle grids (phase_space) ----
    x: jnp.ndarray  # (X,)
    x_weights: jnp.ndarray  # (X,)
    ddx: jnp.ndarray  # (X,X)
    ddtheta: jnp.ndarray  # (T,T)
    ddzeta: jnp.ndarray  # (Z,Z)
    theta_weights: jnp.ndarray  # (T,)
    zeta_weights: jnp.ndarray  # (Z,)
    n_xi_for_x: jnp.ndarray  # (X,) int32
    xi_coupling_lower: jnp.ndarray  # (L,) l/(2l-1)
    xi_coupling_upper: jnp.ndarray  # (L,) (l+1)/(2l+3)

    # ---- flux-surface geometry (magnetic_geometry) ----
    b_hat: jnp.ndarray  # (T,Z)
    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)
    d_hat: jnp.ndarray  # (T,Z)
    b_hat_sup_theta: jnp.ndarray  # (T,Z)
    b_hat_sup_zeta: jnp.ndarray  # (T,Z)
    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)
    fsab_hat2: jnp.ndarray  # scalar <BHat^2>

    # ---- species (species) ----
    z_s: jnp.ndarray  # (S,)
    m_hat: jnp.ndarray  # (S,)
    t_hat: jnp.ndarray  # (S,)
    n_hat: jnp.ndarray  # (S,)
    dn_hat_dpsi_hat: jnp.ndarray  # (S,)
    dt_hat_dpsi_hat: jnp.ndarray  # (S,)

    # ---- normalization / drive scalars (constants conventions) ----
    alpha: jnp.ndarray  # scalar e*phiBar/TBar
    delta: jnp.ndarray  # scalar rho*_ref
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar; RHS-drive value (phi gradient coordinate)
    dphi_hat_dpsi_hat_kinetic: jnp.ndarray  # scalar; value multiplying ExB/Er terms
    e_parallel_hat: jnp.ndarray  # scalar
    e_parallel_hat_spec: jnp.ndarray  # (S,)

    # ---- collisions (physics.collisions; -> collisions later) ----
    pas: PitchAngleScatteringV3Operator | None
    fp: FokkerPlanckV3Operator | None

    # ------------------------------------------------------------------
    # pytree protocol
    # ------------------------------------------------------------------

    _AUX_FIELDS = (
        "n_species",
        "n_x",
        "n_xi",
        "n_theta",
        "n_zeta",
        "rhs_mode",
        "constraint_scheme",
        "point_at_x0",
        "use_dkes_exb",
        "with_exb",
        "with_er_xidot",
        "with_er_xdot",
    )
    _CHILD_FIELDS = (
        "x",
        "x_weights",
        "ddx",
        "ddtheta",
        "ddzeta",
        "theta_weights",
        "zeta_weights",
        "n_xi_for_x",
        "xi_coupling_lower",
        "xi_coupling_upper",
        "b_hat",
        "db_hat_dtheta",
        "db_hat_dzeta",
        "d_hat",
        "b_hat_sup_theta",
        "b_hat_sup_zeta",
        "b_hat_sub_theta",
        "b_hat_sub_zeta",
        "fsab_hat2",
        "z_s",
        "m_hat",
        "t_hat",
        "n_hat",
        "dn_hat_dpsi_hat",
        "dt_hat_dpsi_hat",
        "alpha",
        "delta",
        "dphi_hat_dpsi_hat",
        "dphi_hat_dpsi_hat_kinetic",
        "e_parallel_hat",
        "e_parallel_hat_spec",
        "pas",
        "fp",
    )

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._CHILD_FIELDS)
        aux = tuple(getattr(self, name) for name in self._AUX_FIELDS)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(cls._AUX_FIELDS, aux))
        kwargs.update(dict(zip(cls._CHILD_FIELDS, children)))
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        """Distribution-block shape ``(n_species, n_x, n_xi, n_theta, n_zeta)``."""
        return (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)

    @property
    def f_size(self) -> int:
        return int(np.prod(self.f_shape))

    @property
    def extra_size(self) -> int:
        """Number of bordered source unknowns (= number of constraint rows)."""
        if self.constraint_scheme == 0:
            return 0
        if self.constraint_scheme == 1:
            return 2 * self.n_species
        if self.constraint_scheme == 2:
            return self.n_species * self.n_x
        raise NotImplementedError(f"constraintScheme={self.constraint_scheme} is not supported.")

    @property
    def total_size(self) -> int:
        return self.f_size + self.extra_size

    # ------------------------------------------------------------------
    # matrix-free apply
    # ------------------------------------------------------------------

    def _mask(self) -> jnp.ndarray:
        return _mask_xi(self.n_xi_for_x, self.n_xi)  # (X,L)

    def active_dof_mask(self) -> jnp.ndarray | None:
        """Flat 0/1 mask of the DOFs retained by the ``Nxi_for_x`` truncation.

        The rectangular ``(species, x, L, theta, zeta)`` state layout carries
        entries for Legendre modes ``l >= Nxi_for_x(ix)`` that Fortran v3
        excludes from the matrix entirely (packed indexing, ``indices.F90``
        ``DKE_size``).  Those rows of :meth:`apply` are identically zero, so
        the rectangular embedding of the operator is structurally singular:
        one exact zero singular value per truncated DOF.  Solvers must pin
        them (``A M + (I - M)`` with ``M = diag(mask)``) to recover the
        nonsingular packed system — see :func:`sfincs_jax.solve.solve`.

        Returns:
            ``None`` when every speed node keeps the full Legendre resolution
            (no truncation, operator nonsingular as embedded); otherwise a
            ``(total_size,)`` float64 vector with 1.0 on active DOFs (all
            bordered source unknowns are active) and 0.0 on truncated ones.
        """
        if int(np.min(np.asarray(self.n_xi_for_x))) >= self.n_xi:
            return None
        m = jnp.broadcast_to(self._mask()[None, :, :, None, None], self.f_shape)
        return jnp.concatenate(
            [m.reshape((-1,)), jnp.ones((self.extra_size,), dtype=jnp.float64)]
        )

    def _streaming_mirror(self, f: jnp.ndarray) -> jnp.ndarray:
        """Parallel streaming + mirror terms (populateMatrix.F90 ddtheta/ddzeta/ddxi blocks).

        Couples Legendre rows L to columns L±1 with the
        ``legendre_coupling_lower/upper`` factors; diagonal in x with an overall
        factor ``x sqrt(THat/mHat)``.
        """
        mask = self._mask()
        # Fortran excludes the truncated-L DOFs entirely; mask columns first so
        # the L±1 coupling cannot pull from them.
        fm = f * mask[None, :, :, None, None]

        sqrt_t_over_m = jnp.sqrt(self.t_hat / self.m_hat)  # (S,)
        v_theta_s = sqrt_t_over_m[:, None, None] * (self.b_hat_sup_theta / self.b_hat)[None, :, :]
        v_zeta_s = sqrt_t_over_m[:, None, None] * (self.b_hat_sup_zeta / self.b_hat)[None, :, :]

        dtheta_f = jnp.einsum("ij,sxljz->sxliz", self.ddtheta, fm) * v_theta_s[:, None, None, :, :]
        dzeta_f = jnp.einsum("ij,sxltj->sxlti", self.ddzeta, fm) * v_zeta_s[:, None, None, :, :]

        coef_plus_x = self.x[:, None] * self.xi_coupling_upper[None, :]  # (X,L)
        coef_minus_x = self.x[:, None] * self.xi_coupling_lower[None, :]  # (X,L)

        def couple_l(g: jnp.ndarray) -> jnp.ndarray:
            term_plus = coef_plus_x[None, :, :-1, None, None] * g[:, :, 1:, :, :]
            term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
            term_minus = coef_minus_x[None, :, 1:, None, None] * g[:, :, :-1, :, :]
            term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
            return term_plus + term_minus

        out = couple_l(dtheta_f) + couple_l(dzeta_f)

        # Mirror term: -x sqrt(T/m) (b·∇B)/(2B^2) with (l+2)(l+1)/(2l+3) and
        # -l(l-1)/(2l-1) factors (= the streaming couplings times (l+2), -(l-1)).
        mirror_geom = self.b_hat_sup_theta * self.db_hat_dtheta + self.b_hat_sup_zeta * self.db_hat_dzeta
        mirror_factor = -sqrt_t_over_m[:, None, None] * mirror_geom[None, :, :] / (2.0 * self.b_hat**2)

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        coef_mirror_plus_x = self.x[:, None] * (self.xi_coupling_upper * (ell + 2.0))[None, :]
        coef_mirror_minus_x = self.x[:, None] * (-self.xi_coupling_lower * (ell - 1.0))[None, :]

        term_plus = coef_mirror_plus_x[None, :, :-1, None, None] * fm[:, :, 1:, :, :]
        term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        term_minus = coef_mirror_minus_x[None, :, 1:, None, None] * fm[:, :, :-1, :, :]
        term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
        out = out + (term_plus + term_minus) * mirror_factor[:, None, None, :, :]

        return out * mask[None, :, :, None, None]

    def _exb_coefficients(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Row-scaling ``(T,Z)`` coefficients of the ExB d/dtheta and d/dzeta terms."""
        denom = self.fsab_hat2 if self.use_dkes_exb else self.b_hat**2
        factor = self.alpha * self.delta * 0.5 * self.dphi_hat_dpsi_hat_kinetic
        coef_theta = factor * self.d_hat * self.b_hat_sub_zeta / denom
        coef_zeta = -factor * self.d_hat * self.b_hat_sub_theta / denom
        return coef_theta, coef_zeta

    def _exb(self, f: jnp.ndarray) -> jnp.ndarray:
        """ExB drift terms (diagonal in L and x; ``useDKESExBDrift`` selects <B^2>)."""
        coef_theta, coef_zeta = self._exb_coefficients()
        dtheta_f = jnp.einsum("ij,sxljz->sxliz", self.ddtheta, f)
        dzeta_f = jnp.einsum("ij,sxltj->sxlti", self.ddzeta, f)
        out = dtheta_f * coef_theta[None, None, None, :, :] + dzeta_f * coef_zeta[None, None, None, :, :]
        return out * self._mask()[None, :, :, None, None]

    def _er_xidot(self, f: jnp.ndarray) -> jnp.ndarray:
        """Non-standard d/dxi term associated with E_r (L diagonal and L±2)."""
        temp = self.b_hat_sub_zeta * self.db_hat_dtheta - self.b_hat_sub_theta * self.db_hat_dzeta
        factor = (
            self.alpha
            * self.delta
            * self.dphi_hat_dpsi_hat_kinetic
            / (4.0 * self.b_hat**3)
            * self.d_hat
            * temp
        )  # (T,Z)

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        denom0 = (2.0 * ell - 1.0) * (2.0 * ell + 3.0)
        diag_coef = (ell + 1.0) * ell / denom0

        out = (factor[None, None, None, :, :] * diag_coef[None, None, :, None, None]) * f

        sup2 = (ell + 3.0) * (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
        sub2 = -ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0))

        term_sup2 = sup2[None, None, :-2, None, None] * f[:, :, 2:, :, :]
        term_sup2 = jnp.pad(term_sup2, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
        out = out + factor[None, None, None, :, :] * term_sup2

        term_sub2 = sub2[None, None, 2:, None, None] * f[:, :, :-2, :, :]
        term_sub2 = jnp.pad(term_sub2, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
        out = out + factor[None, None, None, :, :] * term_sub2

        return out * self._mask()[None, :, :, None, None]

    def _er_xdot(self, f: jnp.ndarray) -> jnp.ndarray:
        """Collisionless d/dx term associated with E_r (dense in x; L and L±2).

        ``xDotDerivativeScheme=0`` only (centered ddx for both signs);
        ``force0RadialCurrentInEquilibrium=.true.`` (v3 default) so xDotFactor2=0.
        """
        factor0 = -(self.alpha * self.delta * self.dphi_hat_dpsi_hat_kinetic) / 4.0
        xdot_factor = (
            factor0
            * self.d_hat
            / self.b_hat**3
            * (self.b_hat_sub_theta * self.db_hat_dzeta - self.b_hat_sub_zeta * self.db_hat_dtheta)
        )  # (T,Z)

        x_ddx = self.x[:, None] * self.ddx  # (X,X)

        def x_apply(g: jnp.ndarray) -> jnp.ndarray:
            # g (S,X,L',T,Z) -> x d/dx along the X axis.
            g_xlast = jnp.transpose(g, (0, 2, 3, 4, 1))
            y = jnp.einsum("ij,...j->...i", x_ddx, g_xlast)
            return jnp.transpose(y, (0, 4, 1, 2, 3))

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom  # (L,)

        out = x_apply(f) * (diag_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

        if self.n_xi >= 3:
            l0 = ell[:-2]
            sup = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
            y_sup = x_apply(f[:, :, 2:, :, :])
            y_sup = jnp.pad(y_sup, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
            sup_coef = jnp.pad(sup, (0, 2))
            out = out + y_sup * (sup_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

            l2 = ell[2:]
            sub = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))
            y_sub = x_apply(f[:, :, :-2, :, :])
            y_sub = jnp.pad(y_sub, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
            sub_coef = jnp.pad(sub, (2, 0))
            out = out + y_sub * (sub_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

        return out * self._mask()[None, :, :, None, None]

    def apply_f(self, f: jnp.ndarray) -> jnp.ndarray:
        """Apply the f-block (kinetic) part of the operator to a 5-D ``f``."""
        f = jnp.asarray(f, dtype=jnp.float64)
        if f.shape != self.f_shape:
            raise ValueError(f"f must have shape {self.f_shape}, got {f.shape}")
        out = self._streaming_mirror(f)
        if self.with_exb:
            out = out + self._exb(f)
        if self.with_er_xidot:
            out = out + self._er_xidot(f)
        if self.with_er_xdot:
            out = out + self._er_xdot(f)
        if self.pas is not None:
            out = out + apply_pitch_angle_scattering_v3(self.pas, f)
        if self.fp is not None:
            out = out + apply_fokker_planck_v3(self.fp, f)
        return out

    def _fs_average_factor(self) -> jnp.ndarray:
        """(T,Z) weights of the flux-surface average (w_theta w_zeta / DHat)."""
        return (self.theta_weights[:, None] * self.zeta_weights[None, :]) / self.d_hat

    def _source_basis_constraint_scheme_1(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(xPartOfSource1, xPartOfSource2) for constraintScheme=1 (whichMatrix != 4,5)."""
        x2 = self.x * self.x
        coef = jnp.exp(-x2) / (jnp.pi * jnp.sqrt(jnp.pi))
        return (-x2 + 2.5) * coef, ((2.0 / 3.0) * x2 - 1.0) * coef

    def apply(self, v: jnp.ndarray) -> jnp.ndarray:
        """Apply the full bordered operator ``[[A, B], [C, 0]]`` to a flat state.

        ``A`` is the kinetic f-block, ``B`` injects the constraint-scheme source
        shapes into the L=0 DKE rows, and ``C`` evaluates the flux-surface-average
        moments (populateMatrix.F90 source/constraint blocks).
        """
        v = jnp.asarray(v, dtype=jnp.float64)
        if v.shape != (self.total_size,):
            raise ValueError(f"v must have shape {(self.total_size,)}, got {v.shape}")

        f = v[: self.f_size].reshape(self.f_shape)
        extra = v[self.f_size :]
        y_f = self.apply_f(f)

        factor = self._fs_average_factor()
        ix0 = _ix_min(self.point_at_x0)

        if self.constraint_scheme == 0:
            y_extra = jnp.zeros((0,), dtype=jnp.float64)

        elif self.constraint_scheme == 1:
            src = extra.reshape((self.n_species, 2))
            xpart1, xpart2 = self._source_basis_constraint_scheme_1()
            y_f = y_f.at[:, ix0:, 0, :, :].add(
                xpart1[ix0:][None, :, None, None] * src[:, 0, None, None, None]
                + xpart2[ix0:][None, :, None, None] * src[:, 1, None, None, None]
            )
            x2 = self.x * self.x
            w2 = x2 * self.x_weights
            w4 = x2 * x2 * self.x_weights
            y_dens = jnp.einsum("x,tz,sxtz->s", w2, factor, f[:, :, 0, :, :])
            y_pres = jnp.einsum("x,tz,sxtz->s", w4, factor, f[:, :, 0, :, :])
            y_extra = jnp.stack([y_dens, y_pres], axis=1).reshape((-1,))

        elif self.constraint_scheme == 2:
            src = extra.reshape((self.n_species, self.n_x))
            y_f = y_f.at[:, ix0:, 0, :, :].add(src[:, ix0:, None, None])
            y_avg = jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])
            if self.point_at_x0:
                y_avg = y_avg.at[:, 0].set(src[:, 0])
            y_extra = y_avg.reshape((-1,))

        else:
            raise NotImplementedError(f"constraintScheme={self.constraint_scheme} is not supported.")

        return jnp.concatenate([y_f.reshape((-1,)), y_extra], axis=0)

    # ------------------------------------------------------------------
    # RHS drives (evaluateResidual.F90 with f = 0)
    # ------------------------------------------------------------------

    def _with_rhs_settings(self, which_rhs: int | None) -> "KineticOperator":
        """Apply v3's internal whichRHS gradient/E_parallel overwrites.

        RHSMode=2: whichRHS 1..3 (dn drive, dT drive at fixed pressure drive,
        E_parallel drive); RHSMode=3: whichRHS 1..2 (radial drive, parallel
        drive).  Matches ``solver.F90``'s overwrites before evaluateResidual.
        """
        if which_rhs is None:
            return self
        w = int(which_rhs)
        if self.rhs_mode == 3:
            if w == 1:
                dn, dt, epar = jnp.ones_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 2:
                dn, dt, epar = jnp.zeros_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 1.0
            else:
                raise ValueError("RHSMode=3 expects which_rhs in {1,2}.")
        elif self.rhs_mode == 2:
            if w == 1:
                dn, dt, epar = jnp.ones_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 2:
                # (1/n) dn/dpsi + (3/2) dT/dpsi = 0 with dT/dpsi = 1.
                dn_val = 1.5 * self.n_hat[0] * self.t_hat[0]
                dn = jnp.broadcast_to(dn_val, self.dn_hat_dpsi_hat.shape)
                dt, epar = jnp.ones_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 3:
                dn, dt, epar = jnp.zeros_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 1.0
            else:
                raise ValueError("RHSMode=2 expects which_rhs in {1,2,3}.")
        else:
            if w != 1:
                raise ValueError("RHSMode=1 has a single RHS (which_rhs=1 or None).")
            return self
        return replace(
            self,
            dn_hat_dpsi_hat=dn,
            dt_hat_dpsi_hat=dt,
            e_parallel_hat=jnp.asarray(epar, dtype=jnp.float64),
        )

    def rhs(self, which_rhs: int | None = None) -> jnp.ndarray:
        """Assemble the v3 RHS vector (evaluateResidual.F90 drives, f-independent).

        - the ``dot(psi) dfM/dpsi`` gradient drive (L=0 with weight 4/3 and L=2
          with weight 2/3);
        - the inductive ``EParallelHat`` drive (L=1);
        - zeros on the constraint rows.

        ``which_rhs`` selects the RHSMode 2/3 transport-matrix drive column.
        """
        op = self._with_rhs_settings(which_rhs)

        f_rhs = jnp.zeros(op.f_shape, dtype=jnp.float64)
        ix0 = _ix_min(op.point_at_x0)
        x2 = op.x * op.x
        expx2 = jnp.exp(-x2)
        x2_expx2 = x2 * expx2
        sqrt_pi = jnp.sqrt(jnp.pi)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)

        # For RHSMode 2/3 the electrostatic term is excluded from the drive.
        dphi_to_use = jnp.where(
            (op.rhs_mode == 1) | (op.rhs_mode > 3),
            op.dphi_hat_dpsi_hat,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        geom2 = (
            (op.b_hat_sub_zeta * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dzeta)
            * op.d_hat
            / op.b_hat**3
        )  # (T,Z)

        mask_x = (jnp.arange(op.n_x) >= ix0).astype(jnp.float64)

        x_part = x2_expx2[None, :] * (
            (op.dn_hat_dpsi_hat / op.n_hat)[:, None]
            + (op.alpha * op.z_s / op.t_hat)[:, None] * dphi_to_use
            + (x2[None, :] - 1.5) * (op.dt_hat_dpsi_hat / op.t_hat)[:, None]
        )  # (S,X)

        pref = (
            op.delta
            * op.n_hat
            * op.m_hat
            * jnp.sqrt(op.m_hat)
            / (two_pi * sqrt_pi * op.z_s * jnp.sqrt(op.t_hat))
        )  # (S,)

        factor = pref[:, None, None, None] * geom2[None, None, :, :] * x_part[:, :, None, None]
        factor = factor * mask_x[None, :, None, None]

        if op.n_xi > 0:
            mask_l0 = (op.n_xi_for_x > 0).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 0, :, :].add((4.0 / 3.0) * factor * mask_l0[None, :, None, None])
        if op.n_xi > 2:
            mask_l2 = (op.n_xi_for_x > 2).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 2, :, :].add((2.0 / 3.0) * factor * mask_l2[None, :, None, None])

        if op.n_xi > 1:
            epar = op.e_parallel_hat + op.e_parallel_hat_spec  # (S,)
            factor_e = (
                op.alpha
                * op.z_s[:, None]
                * op.x[None, :]
                * expx2[None, :]
                * epar[:, None]
                * op.n_hat[:, None]
                * op.m_hat[:, None]
                / (jnp.pi * sqrt_pi * (op.t_hat * op.t_hat)[:, None] * op.fsab_hat2)
            )  # (S,X)
            factor_e = factor_e * mask_x[None, :]
            f_rhs = f_rhs.at[:, :, 1, :, :].add(factor_e[:, :, None, None] * op.b_hat[None, None, :, :])

        rhs_extra = jnp.zeros((op.extra_size,), dtype=jnp.float64)
        return jnp.concatenate([f_rhs.reshape((-1,)), rhs_extra], axis=0)

    # ------------------------------------------------------------------
    # analytic Legendre-block extraction (probing-free)
    # ------------------------------------------------------------------

    def _check_block_extraction_supported(self) -> None:
        if self.with_er_xidot or self.with_er_xdot:
            raise NotImplementedError(
                "legendre_blocks requires DKES trajectories: the Er xiDot/xDot terms "
                "couple L±2 and break the block-tridiagonal-in-L structure."
            )
        if self.fp is not None:
            raise NotImplementedError(
                "legendre_blocks currently supports pitch-angle-scattering collisions only; "
                "Fokker-Planck couples (species, x) densely within each L "
                "(its per-L blocks live in KineticOperator.fp.mat)."
            )

    def legendre_blocks(self, ell: int) -> LegendreBlocks:
        """Dense (theta*zeta) blocks of Legendre row ``ell`` of the f-block.

        Built analytically from the term coefficients — no operator probing:
        streaming/mirror provide ``lower``/``upper`` (L±1 with the phase_space
        coupling factors), ExB and pitch-angle scattering provide ``diag``.
        Supported for the DKES-trajectory PAS family (the tier-1 structured
        solver family of plan §2.3); raises otherwise.

        Returns blocks of shape ``(n_species, n_x, T*Z, T*Z)``; the truncated-L
        masks are baked in, so the block-tridiagonal matvec reproduces
        :meth:`apply_f` exactly.
        """
        self._check_block_extraction_supported()
        if not (0 <= int(ell) < self.n_xi):
            raise ValueError(f"ell must be in [0, {self.n_xi}), got {ell}")
        ell = int(ell)

        n_tz = self.n_theta * self.n_zeta
        eye_t = jnp.eye(self.n_theta, dtype=jnp.float64)
        eye_z = jnp.eye(self.n_zeta, dtype=jnp.float64)
        d_theta_tz = jnp.kron(self.ddtheta, eye_z)  # (TZ,TZ)
        d_zeta_tz = jnp.kron(eye_t, self.ddzeta)  # (TZ,TZ)

        mask = np.asarray(self._mask())  # (X,L)
        row_mask = jnp.asarray(mask[:, ell])  # (X,)

        sqrt_t_over_m = jnp.sqrt(self.t_hat / self.m_hat)  # (S,)
        v_theta = (self.b_hat_sup_theta / self.b_hat).reshape((-1,))  # (TZ,)
        v_zeta = (self.b_hat_sup_zeta / self.b_hat).reshape((-1,))
        # Streaming operator per species: rowscale(v)·D, summed over both angles.
        stream_tz = (
            sqrt_t_over_m[:, None, None]
            * (v_theta[None, :, None] * d_theta_tz[None, :, :] + v_zeta[None, :, None] * d_zeta_tz[None, :, :])
        )  # (S,TZ,TZ)

        mirror_geom = self.b_hat_sup_theta * self.db_hat_dtheta + self.b_hat_sup_zeta * self.db_hat_dzeta
        mirror_diag = (
            -sqrt_t_over_m[:, None] * (mirror_geom / (2.0 * self.b_hat**2)).reshape((-1,))[None, :]
        )  # (S,TZ)
        mirror_tz = mirror_diag[:, :, None] * jnp.eye(n_tz, dtype=jnp.float64)[None, :, :]  # (S,TZ,TZ)

        def _shaped(block_s: jnp.ndarray, x_factor: jnp.ndarray, col_mask: jnp.ndarray) -> jnp.ndarray:
            # block_s (S,TZ,TZ); x_factor (X,); col_mask (X,) -> (S,X,TZ,TZ)
            scale = x_factor * row_mask * col_mask  # (X,)
            return block_s[:, None, :, :] * scale[None, :, None, None]

        # ---- lower: row ell receives column ell-1 ----
        if ell >= 1:
            coef_stream = float(np.asarray(self.xi_coupling_lower)[ell])
            coef_mirror = -coef_stream * (ell - 1.0)
            col_mask = jnp.asarray(mask[:, ell - 1])
            lower = _shaped(coef_stream * stream_tz + coef_mirror * mirror_tz, self.x, col_mask)
        else:
            lower = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)

        # ---- upper: row ell receives column ell+1 ----
        if ell + 1 < self.n_xi:
            coef_stream = float(np.asarray(self.xi_coupling_upper)[ell])
            coef_mirror = coef_stream * (ell + 2.0)
            col_mask = jnp.asarray(mask[:, ell + 1])
            upper = _shaped(coef_stream * stream_tz + coef_mirror * mirror_tz, self.x, col_mask)
        else:
            upper = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)

        # ---- diag: ExB (species-independent) + PAS (diagonal in theta/zeta) ----
        diag = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)
        if self.with_exb:
            coef_theta, coef_zeta = self._exb_coefficients()
            exb_tz = (
                coef_theta.reshape((-1,))[:, None] * d_theta_tz
                + coef_zeta.reshape((-1,))[:, None] * d_zeta_tz
            )  # (TZ,TZ)
            diag = diag + exb_tz[None, None, :, :] * row_mask[None, :, None, None]
        if self.pas is not None:
            pas_coef = self.pas.coef[:, :, ell]  # (S,X)
            diag = diag + (pas_coef * row_mask[None, :])[:, :, None, None] * jnp.eye(
                n_tz, dtype=jnp.float64
            )[None, None, :, :]

        return LegendreBlocks(lower=lower, diag=diag, upper=upper)

    def to_block_tridiagonal(self) -> LegendreBlocks:
        """Stack :meth:`legendre_blocks` over all L.

        Returns arrays of shape ``(n_xi, n_species, n_x, T*Z, T*Z)`` ready for a
        block-Thomas recursion over L (SOLVAX tier-1 kernel).
        """
        self._check_block_extraction_supported()
        blocks = [self.legendre_blocks(ell) for ell in range(self.n_xi)]
        return LegendreBlocks(
            lower=jnp.stack([b.lower for b in blocks]),
            diag=jnp.stack([b.diag for b in blocks]),
            upper=jnp.stack([b.upper for b in blocks]),
        )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_namelist(cls, nml: Any) -> "KineticOperator":
        """Build the operator from a parsed SFINCS input namelist.

        See :func:`kinetic_operator_from_namelist` for details.
        """
        return kinetic_operator_from_namelist(nml)


# =============================================================================
# Namelist construction (readInput.F90 defaults; createGrids.F90 overrides)
# =============================================================================


def _flux_surface_averages_effective(
    *, grids: Grids, geom: FluxSurfaceGeometry
) -> tuple[float, float, float]:
    """Effective ``(B0OverBBar, GHat, IHat)`` from surface averages.

    For VMEC geometry (scheme 5) the flux functions are stored as 0.0
    placeholders and v3 recomputes them in ``computeBIntegrals``:
    ``B0OverBBar = <B^3>/<B^2>``, ``GHat = <B_sub_zeta>``, ``IHat = <B_sub_theta>``
    (the latter two averaged over 4*pi^2 with the quadrature weights).
    """
    w = np.asarray(grids.theta_weights)[:, None] * np.asarray(grids.zeta_weights)[None, :]
    d_hat = np.asarray(geom.d_hat)
    b_hat = np.asarray(geom.b_hat)
    vprime = float(np.sum(w / d_hat))
    fsab2 = float(np.sum(w * b_hat**2 / d_hat) / vprime)
    b0_eff = float(np.sum(w * b_hat**3 / d_hat) / (vprime * fsab2))
    denom = 4.0 * math.pi * math.pi
    g_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_zeta)) / denom)
    i_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_theta)) / denom)
    return b0_eff, g_eff, i_eff


def _geometry_and_radial(
    *, nml: Any, grids: Grids
) -> tuple[FluxSurfaceGeometry, RadialCoordinates]:
    """Geometry + radial-coordinate conversions for the supported geometrySchemes."""
    geom_params = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)

    if scheme == 1:
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys, default=0.15596)
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        r_n = math.sqrt(float(psi_n_wish))
        geom = FluxSurfaceGeometry.from_scheme(
            1,
            theta=grids.theta,
            zeta=grids.zeta,
            epsilon_t=_get_float(geom_params, "epsilon_t", -0.07053),
            epsilon_h=_get_float(geom_params, "epsilon_h", 0.05067),
            epsilon_antisymm=_get_float(geom_params, "epsilon_antisymm", 0.0),
            iota=_get_float(geom_params, "iota", 0.4542),
            g_hat=_get_float(geom_params, "GHat", 3.7481),
            i_hat=_get_float(geom_params, "IHat", 0.0),
            b0_over_bbar=_get_float(geom_params, "B0OverBBar", 1.0),
            helicity_l=_get_int(geom_params, "helicity_l", 2),
            helicity_n=_get_int(geom_params, "helicity_n", 10),
            helicity_antisymm_l=_get_int(geom_params, "helicity_antisymm_l", 1),
            helicity_antisymm_n=_get_int(geom_params, "helicity_antisymm_n", 0),
        )
        return geom, RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    if scheme == 2:
        # v3 fixed simplified LHD model; rN forced to 0.5.
        a_hat = 0.5585
        return (
            FluxSurfaceGeometry.from_scheme(2, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 3:
        # v3 fixed LHD inward-shifted analytic Boozer model (geometry.F90 case 3);
        # aHat=0.5400, psiAHat=aHat^2/2, rN forced to 0.5 (rN_wish ignored).  The
        # B-field harmonics/flux functions come from from_scheme(3).
        a_hat = 0.5400
        return (
            FluxSurfaceGeometry.from_scheme(3, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 4:
        # v3 built-in W7-X standard model; rN forced to 0.5.
        return (
            FluxSurfaceGeometry.from_scheme(4, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=-0.384935, a_hat=0.5109, r_n=0.5),
        )

    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _surfaces = read_boozer_bc(path, geometry_scheme=scheme)
        psi_n_wish = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        r_n_wish = math.sqrt(float(psi_n_wish))
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        r_n = float(
            selected_r_n_from_bc(
                path=str(path),
                geometry_scheme=scheme,
                r_n_wish=r_n_wish,
                vmec_radial_option=vmec_radial_option,
            )
        )
        geom = FluxSurfaceGeometry.from_boozer(
            path,
            theta=grids.theta,
            zeta=grids.zeta,
            r_n_wish=r_n_wish,
            vmec_radial_option=vmec_radial_option,
            geometry_scheme=scheme,
        )
        return geom, RadialCoordinates(psi_a_hat=float(header.psi_a_hat), a_hat=float(header.a_hat), r_n=r_n)

    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        w = read_vmec_wout(path)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        interp = vmec_radial_interpolation(
            w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=vmec_radial_option
        )
        r_n = float(interp.psi_n) ** 0.5
        geom = FluxSurfaceGeometry.from_vmec(
            w,
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=float(psi_n_wish),
            vmec_radial_option=vmec_radial_option,
            vmec_nyquist_option=_get_int(geom_params, "VMEC_NYQUIST_OPTION", 1),
            min_bmn_to_load=_get_float(geom_params, "MIN_BMN_TO_LOAD", 0.0),
            ripple_scale=_get_float(geom_params, "RIPPLESCALE", 1.0),
            helicity_n=_get_int(geom_params, "HELICITY_N", 0),
            helicity_l=_get_int(geom_params, "HELICITY_L", 0),
        )
        return geom, RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)

    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12}}; got {scheme}."
    )


def _resolve_equilibrium_path(*, nml: Any, geom_params: dict, vmec: bool = False) -> Path:
    eq = effective_equilibrium_file(geom_params=geom_params)
    if eq is None:
        raise ValueError("This geometryScheme requires equilibriumFile in geometryParameters.")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
    try:
        return resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path
    except FileNotFoundError:
        if vmec:
            # Allow `.txt -> .nc` fallback for VMEC wout files (matches the old reader).
            p2 = Path(str(eq).strip().strip('"').strip("'")).with_suffix(".nc")
            return resolve_existing_path(str(p2), base_dir=base_dir, extra_search_dirs=extra).path
        raise


def _n_periods_from_namelist(*, nml: Any) -> int:
    """NPeriods for grid construction (createGrids.F90 / geometry.F90)."""
    geom_params = nml.group("geometryParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)
    if scheme == 1:
        return max(1, _get_int(geom_params, "helicity_n", 10))
    if scheme in {2, 3}:
        return 10
    if scheme == 4:
        return 5
    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _ = read_boozer_bc(path, geometry_scheme=scheme)
        return int(header.n_periods)
    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        return int(read_vmec_wout(path).nfp)
    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12}}; got {scheme}."
    )


def kinetic_operator_from_namelist(nml: Any) -> KineticOperator:
    """Build a :class:`KineticOperator` from a parsed SFINCS input namelist.

    ``nml`` is a :class:`sfincs_jax.namelist.Namelist`.  Grids come from
    :func:`sfincs_jax.phase_space.make_grids`, geometry from
    :class:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry`, species from
    :func:`sfincs_jax.species.species_set_from_namelist`, radial-coordinate
    and monoenergetic conversions from :mod:`sfincs_jax.constants`, and
    collision matrices from :mod:`sfincs_jax.physics.collisions`.

    Supports ``geometryScheme`` in {1, 2, 3, 4, 5, 11, 12} (analytic schemes
    1/2/3/4 and file-based 5/11/12).  Raises ``NotImplementedError`` for the
    deferred features listed in the module docstring (Phi1, magnetic drifts,
    constraintScheme 3/4, mapped x-grids, and the namelist Boozer-spectrum
    geometryScheme 13).
    """
    general = nml.group("general")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")
    res = nml.group("resolutionParameters")
    geom_params = nml.group("geometryParameters")

    # ---- reject deferred physics up front ----
    if _get_bool(phys, "includePhi1", False):
        raise NotImplementedError(
            "includePhi1 (quasineutrality/lambda rows, Phi1 kinetic/collision couplings) "
            "is not yet consolidated into KineticOperator; use the operators/profile_* path."
        )
    if _get_bool(phys, "readExternalPhi1", False):
        raise NotImplementedError("readExternalPhi1 is not supported.")
    magnetic_drift_scheme = _get_int(phys, "magneticDriftScheme", 0)
    if magnetic_drift_scheme != 0:
        raise NotImplementedError(
            "magneticDriftScheme != 0 (tangential magnetic drifts) is not yet consolidated "
            "into KineticOperator; use the operators/profile_* path."
        )
    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator not in {0, 1}:
        raise NotImplementedError(f"collisionOperator={collision_operator} is not supported.")

    rhs_mode = _get_int(general, "RHSMode", 1)
    constraint_scheme = _get_int(phys, "constraintScheme", -1)
    if constraint_scheme < 0:
        constraint_scheme = 1 if collision_operator == 0 else 2
    if constraint_scheme not in {0, 1, 2}:
        raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported.")

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    if x_grid_scheme not in {1, 2, 5, 6}:
        raise NotImplementedError(f"Only xGridScheme in {{1,2,5,6}} is supported (got {x_grid_scheme}).")
    if _get_int(other, "xDotDerivativeScheme", 0) != 0:
        raise NotImplementedError("Only xDotDerivativeScheme=0 is supported.")
    point_at_x0 = x_grid_scheme in {2, 6}

    # ---- grids (phase_space) ----
    grids = make_grids(
        n_theta=_get_int(res, "Ntheta", 15),
        n_zeta=_get_int(res, "Nzeta", 15),
        n_xi=_get_int(res, "Nxi", 16),
        n_x=_get_int(res, "Nx", 5),
        n_l=_get_int(res, "NL", 4),
        n_periods=_n_periods_from_namelist(nml=nml),
        theta_derivative_scheme=_get_int(other, "thetaDerivativeScheme", 2),
        zeta_derivative_scheme=_get_int(other, "zetaDerivativeScheme", 2),
        magnetic_drift_derivative_scheme=_get_int(other, "magneticDriftDerivativeScheme", 3),
        x_grid_scheme=x_grid_scheme,
        x_grid_k=_get_float(other, "xGrid_k", 0.0),
        n_xi_for_x_option=_get_int(other, "Nxi_for_x_option", 1),
        monoenergetic=(rhs_mode == 3),
    )

    # ---- geometry + radial conversions (magnetic_geometry / constants) ----
    geom, radial = _geometry_and_radial(nml=nml, grids=grids)
    fsab_hat2 = float(geom.fsab_hat2(theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights))

    # ---- species (species) ----
    species_grad_coord = infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params, species_params=nml.group("speciesParameters"), default=4
    )
    species: SpeciesSet = species_set_from_namelist(
        nml, radial=radial, input_radial_coordinate_for_gradients=species_grad_coord
    )
    n_species = species.n_species

    alpha = _get_float(phys, "alpha", 1.0)
    delta = _get_float(phys, "Delta", DEFAULT_DELTA)
    nu_n = _get_float(phys, "nu_n", DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)
    er = _get_float(phys, "Er", 0.0)

    # ---- flux functions (effective values for VMEC placeholders) ----
    b0_eff = float(geom.b0_over_bbar)
    g_eff = float(geom.g_hat)
    i_eff = float(geom.i_hat)
    if abs(g_eff) < 1e-30 or abs(b0_eff) < 1e-30:
        b0_eff, g_eff, i_eff = _flux_surface_averages_effective(grids=grids, geom=geom)

    # ---- dPhiHat/dpsiHat: drive value (phi gradient coordinate) and kinetic value ----
    phi_grad_coord = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=geom_params, phys_params=phys, default=4
    )
    if phi_grad_coord == 0:
        dphi_rhs = _get_float(phys, "dPhiHatdpsiHat", 0.0)
    elif phi_grad_coord == 1:
        dphi_rhs = radial.d_dpsi_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdpsiN", 0.0)
    elif phi_grad_coord == 2:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrHat", 0.0)
    elif phi_grad_coord == 3:
        dphi_rhs = radial.d_dr_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrN", 0.0)
    elif phi_grad_coord == 4:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * (-er)
    else:
        raise NotImplementedError(f"Unsupported inputRadialCoordinateForGradients={phi_grad_coord} for Phi.")

    if rhs_mode == 3:
        # sfincs_main.F90: EStar overwrites dPhiHatdpsiHat for monoenergetic runs.
        e_star = _get_float(phys, "EStar", 0.0)
        dphi_kinetic = d_phi_hat_d_psi_hat_from_e_star(
            e_star=e_star, alpha=alpha, delta=delta, iota=float(geom.iota), b0_over_bbar=b0_eff, g_hat=g_eff
        )
        dphi_rhs = dphi_kinetic
    else:
        # The ExB/Er kinetic terms follow the Er input (v3 examples convention).
        if er != 0.0 and phi_grad_coord != 4:
            raise NotImplementedError(
                "Er != 0 with a non-Er inputRadialCoordinateForGradients is not supported."
            )
        dphi_kinetic = radial.d_dr_hat_to_d_dpsi_hat * (-er)

    include_xdot = _get_bool(phys, "includeXDotTerm", False)
    include_er_xidot = _get_bool(phys, "includeElectricFieldTermInXiDot", False)
    use_dkes_exb = _get_bool(phys, "useDKESExBDrift", False)
    has_er = float(dphi_kinetic) != 0.0

    # ---- collisions (physics.collisions builders) ----
    pas = None
    fp = None
    if collision_operator == 1:
        nu_n_use = nu_n
        if rhs_mode == 3:
            nu_n_use = nu_n_from_nu_prime(
                nu_prime=_get_float(phys, "nuPrime", 1.0),
                b0_over_bbar=b0_eff,
                g_hat=g_eff,
                i_hat=i_eff,
                iota=float(geom.iota),
            )
        pas = make_pitch_angle_scattering_v3_operator(
            x=grids.x,
            z_s=species.z,
            m_hats=species.m_hat,
            n_hats=species.n_hat,
            t_hats=species.t_hat,
            nu_n=float(nu_n_use),
            krook=krook,
            n_xi_for_x=grids.n_xi_for_x,
            n_xi=grids.n_xi,
        )
    else:
        import os  # noqa: PLC0415

        strict_env = os.environ.get("SFINCS_JAX_FP_STRICT_PARITY", "").strip().lower()
        if strict_env in {"0", "false", "no", "off"}:
            strict_parity = False
        elif strict_env in {"1", "true", "yes", "on"}:
            strict_parity = True
        else:
            strict_parity = bool(rhs_mode == 1 and n_species > 1)
        fp = make_fokker_planck_v3_operator(
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            x_grid_k=_get_float(other, "xGrid_k", 0.0),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
            strict_parity=strict_parity,
        )

    # ---- inductive E_parallel ----
    e_parallel_hat = _get_float(phys, "EParallelHat", 0.0)
    epar_spec_raw = phys.get("EPARALLELHATSPEC", None)
    if epar_spec_raw is None:
        e_parallel_hat_spec = jnp.zeros((n_species,), dtype=jnp.float64)
    else:
        e_parallel_hat_spec = jnp.atleast_1d(jnp.asarray(epar_spec_raw, dtype=jnp.float64))
        if e_parallel_hat_spec.shape == (1,) and n_species > 1:
            e_parallel_hat_spec = jnp.broadcast_to(e_parallel_hat_spec, (n_species,))
        if e_parallel_hat_spec.shape != (n_species,):
            raise ValueError(
                f"EParallelHatSpec must have {n_species} entries, got {e_parallel_hat_spec.shape}"
            )

    return KineticOperator(
        n_species=n_species,
        n_x=grids.n_x,
        n_xi=grids.n_xi,
        n_theta=grids.n_theta,
        n_zeta=grids.n_zeta,
        rhs_mode=rhs_mode,
        constraint_scheme=constraint_scheme,
        point_at_x0=point_at_x0,
        use_dkes_exb=use_dkes_exb,
        with_exb=has_er,
        with_er_xidot=bool(include_er_xidot and has_er),
        with_er_xdot=bool(include_xdot and has_er),
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        n_xi_for_x=grids.n_xi_for_x,
        xi_coupling_lower=jnp.asarray(legendre_coupling_lower(grids.n_xi)),
        xi_coupling_upper=jnp.asarray(legendre_coupling_upper(grids.n_xi)),
        b_hat=geom.b_hat,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        d_hat=geom.d_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        z_s=species.z,
        m_hat=species.m_hat,
        t_hat=species.t_hat,
        n_hat=species.n_hat,
        dn_hat_dpsi_hat=species.dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=species.dt_hat_dpsi_hat,
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi_rhs, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi_kinetic, dtype=jnp.float64),
        e_parallel_hat=jnp.asarray(e_parallel_hat, dtype=jnp.float64),
        e_parallel_hat_spec=e_parallel_hat_spec,
        pas=pas,
        fp=fp,
    )


apply_kinetic_operator_jit = jax.jit(lambda op, v: op.apply(v))
