"""Profile-response Schur and coarse preconditioners.

This module owns the RHSMode-1 Schur family as one domain unit: base-kind
selection, active coarse residual bases, bounded native-stack policy, and the
host-side structured full-CSR Schur factors. The setup helpers here may use
SciPy for non-autodiff CLI/production solves; differentiable Python workflows
route through JAX-native solve policies before they reach these host factors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import os
import time

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_response.sources import (
    constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f,
)
from sfincs_jax.solvers.preconditioner_caches import (
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_SCHUR_CACHE,
)
from sfincs_jax.solvers.preconditioner_context import precond_dtype as _precond_dtype
from sfincs_jax.solvers.preconditioner_setup import rhs_mode1_precond_cache_key
from ....v3_system import V3FullSystemOperator, _ix_min

__all__ = (
    "ActiveNativeFieldSplitSparseCoarsePolicy",
    "ActiveNativeStackPolicy",
    "ActiveSparseCoarseResidualPolicy",
    "RHS1SchurPreconditionerBuilders",
    "RHS1StructuredFullCSRPreconditioner",
    "append_adaptive_residual_basis_csc",
    "build_active_native_xell_coarse_window_basis_csc",
    "build_block_schur_preconditioner",
    "build_coarse_residual_basis_csc",
    "build_diagonal_schur_preconditioner",
    "build_jacobi_preconditioner",
    "build_rhs1_schur_preconditioner",
    "build_x_xi_block_schur_preconditioner",
    "build_xi_block_schur_preconditioner",
    "canonical_schur_base_kind",
    "coarse_residual_config",
    "coarse_surface_mode_count",
    "coarse_surface_modes",
    "estimate_coarse_residual_nbytes",
    "estimate_x_xi_block_inverse_nbytes",
    "estimate_xblock_tz_low_l_factor_nbytes",
    "estimate_xi_block_inverse_nbytes",
    "estimate_zeta_block_inverse_nbytes",
    "resolve_active_native_field_split_sparse_coarse_policy",
    "resolve_active_native_stack_policy",
    "resolve_active_sparse_coarse_residual_policy",
    "resolve_rhs1_schur_base_kind",
    "safe_inverse_diagonal",
    "xblock_tz_low_l_config",)

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]
Builder = Callable[..., Preconditioner]
ApplicabilityPredicate = Callable[[V3FullSystemOperator], bool]


# Schur base-selection policy.
_SCHUR_BASE_ALIASES: dict[str, str] = {
    "theta": "theta_line",
    "theta_line": "theta_line",
    "line_theta": "theta_line",
    "theta_dd": "theta_dd",
    "theta_block": "theta_dd",
    "dd_theta": "theta_dd",
    "dd_t": "theta_dd",
    "zeta": "zeta_line",
    "zeta_line": "zeta_line",
    "line_zeta": "zeta_line",
    "zeta_dd": "zeta_dd",
    "zeta_block": "zeta_dd",
    "dd_zeta": "zeta_dd",
    "dd_z": "zeta_dd",
    "adi": "adi",
    "adi_line": "adi",
    "theta_zeta": "adi",
    "zeta_theta": "adi",
    "species": "species_block",
    "species_block": "species_block",
    "speciesblock": "species_block",
    "sxblock_tz": "sxblock_tz",
    "sxblock_theta_zeta": "sxblock_tz",
    "species_xblock_tz": "sxblock_tz",
    "sx_tz": "sxblock_tz",
    "xblock_tz": "xblock_tz",
    "xblock": "xblock_tz",
    "x_tz": "xblock_tz",
    "xtz": "xblock_tz",
    "xblock_theta_zeta": "xblock_tz",
    "xblock_tz_lmax": "xblock_tz_lmax",
    "xblock_lmax": "xblock_tz_lmax",
    "xtz_lmax": "xblock_tz_lmax",
    "xblock_theta_zeta_lmax": "xblock_tz_lmax",
    "xmg": "xmg",
    "multigrid": "xmg",
    "x_coarse": "xmg",
    "coarse_x": "xmg",
    "pas_lite": "pas_lite",
    "pas_light": "pas_lite",
    "pas_xmg": "pas_lite",
    "pas_xmg_lite": "pas_lite",
    "pas_hybrid": "pas_hybrid",
    "pas_xline_xcoarse": "pas_hybrid",
    "pas_line_xcoarse": "pas_hybrid",
    "pas_xcoarse_line": "pas_hybrid",
    "pas_schur": "pas_schur",
    "pas_block_schur": "pas_schur",
    "pas_xmg_l": "pas_schur",
    "pas_ilu": "pas_ilu",
    "pas_block_ilu": "pas_ilu",
    "pas_xblock_ilu": "pas_ilu",
    "block_ilu": "pas_ilu",
    "pas_tz": "pas_tz",
    "pas_theta_zeta": "pas_tz",
    "pas_tz_l": "pas_tz",
    "pas_l_tz": "pas_tz",
    "tz_l": "pas_tz",
    "tz_lblock": "pas_tz",
    "pas_tokamak_theta": "pas_tokamak_theta",
    "pas_tokamak": "pas_tokamak_theta",
    "pas_theta": "pas_tokamak_theta",
    "tokamak_theta": "pas_tokamak_theta",
    "theta_tokamak": "pas_tokamak_theta",
    "point": "point",
    "block": "point",
    "jacobi": "point",
}


def canonical_schur_base_kind(base_kind_env: str | None) -> str | None:
    """Return the canonical Schur-base kind for an explicit environment value."""
    key = str(base_kind_env or "").strip().lower()
    if not key:
        return None
    return _SCHUR_BASE_ALIASES.get(key)


def _env_int(name: str, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def resolve_rhs1_schur_base_kind(
    *,
    base_kind_env: str | None,
    n_theta: int,
    n_zeta: int,
    n_species: int,
    total_size: int,
    nxi_for_x: Sequence[int],
    has_pas: bool,
    has_fp: bool,
    has_er_xdot: bool,
    has_er_xidot: bool,
    use_dkes_exb: bool,
    pas_tokamak_theta_applicable: bool,
    pas_tz_applicable: bool,
    geom_scheme: int | None = None,
) -> str:
    """Resolve the base preconditioner used inside RHSMode=1 Schur.

    Explicit ``SFINCS_JAX_RHSMODE1_SCHUR_BASE`` aliases win. Otherwise the policy
    mirrors the historical v3-driver ordering: large PAS+Er uses x-coarse,
    DKES PAS prefers dense x-blocks only while bounded, tokamak PAS uses the
    dedicated theta/L base, and large 3D PAS uses the structured PAS-TZ base.
    """
    explicit = canonical_schur_base_kind(base_kind_env)
    if explicit is not None:
        return explicit

    n_theta_i = int(n_theta)
    n_zeta_i = int(n_zeta)
    n_species_i = int(n_species)
    total_size_i = int(total_size)
    geom_scheme_i = int(geom_scheme or 0)
    nxi = np.asarray(nxi_for_x, dtype=np.int64)
    max_l = int(np.max(nxi)) if nxi.size else 0
    local_per_species = int(np.sum(nxi))
    dke_size = int(local_per_species * n_theta_i * n_zeta_i)

    if bool(has_pas) and bool(has_er_xdot) and (not bool(has_fp)):
        xmg_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", 50000)
        if total_size_i >= max(1, int(xmg_min)):
            return "xmg"

    if n_theta_i <= 1 and n_zeta_i <= 1:
        return "point"

    tz_max = _env_int("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", 128)
    default_xblock_tz_max = 2000 if bool(has_pas) else 1200
    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", default_xblock_tz_max)
    block_size = int(max_l * n_theta_i * n_zeta_i)
    pas_tz_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", 800)

    if bool(has_pas) and (not bool(has_fp)) and bool(use_dkes_exb) and (not bool(has_er_xdot)) and (not bool(has_er_xidot)):
        dense_bytes_max = max(0, _env_int("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", 512 * 1024 * 1024))
        block_sizes_x = (nxi.astype(np.int64, copy=False) * n_theta_i * n_zeta_i).astype(np.int64, copy=False)
        dense_bytes = int(n_species_i * int(np.sum(block_sizes_x * block_sizes_x)) * 8)
        xblock_tz_small_default = max(0, int(xblock_tz_max))
        xblock_tz_small_max = max(
            0,
            _env_int("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_SMALL_MAX", xblock_tz_small_default),
        )
        if (
            xblock_tz_small_max > 0
            and xblock_tz_max > 0
            and block_size <= xblock_tz_max
            and block_size <= xblock_tz_small_max
            and dense_bytes <= dense_bytes_max
        ):
            return "xblock_tz"
        return "pas_ilu"

    if bool(has_pas) and bool(pas_tokamak_theta_applicable):
        return "pas_tokamak_theta"

    if bool(has_pas) and (not bool(has_fp)) and bool(pas_tz_applicable) and block_size >= pas_tz_min:
        return "pas_tz"

    species_block_max = _env_int("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", 1600)
    if (
        bool(has_pas)
        and n_theta_i > 1
        and n_zeta_i > 1
        and species_block_max > 0
        and dke_size <= species_block_max
    ):
        return "species_block"

    if bool(has_pas) and n_theta_i > 1 and xblock_tz_max > 0 and block_size <= xblock_tz_max:
        return "xblock_tz"

    if bool(has_pas) and n_theta_i > 1 and n_zeta_i > 1 and n_theta_i * n_zeta_i <= tz_max:
        return "theta_zeta"

    if bool(has_pas) and (geom_scheme_i == 1 or n_zeta_i <= 5):
        return "pas_schur"

    return "theta_line" if n_theta_i >= n_zeta_i else "zeta_line"

@dataclass(frozen=True)
class RHS1SchurPreconditionerBuilders:
    """Builder bundle used by the RHSMode=1 Schur compatibility facade."""

    pas_tokamak_theta_applicable: ApplicabilityPredicate
    pas_tz_applicable: ApplicabilityPredicate
    theta_line_builder: Builder
    theta_dd_builder: Builder
    species_block_builder: Builder
    sxblock_tz_builder: Builder
    xblock_tz_builder: Builder
    xblock_tz_lmax_builder: Builder
    pas_xblock_ilu_builder: Builder
    xmg_builder: Builder
    pas_lite_builder: Builder
    pas_hybrid_builder: Builder
    pas_schur_builder: Builder
    pas_tokamak_theta_builder: Builder
    pas_tz_builder: Builder
    theta_zeta_builder: Builder
    zeta_line_builder: Builder
    zeta_dd_builder: Builder
    block_builder: Builder


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=_precond_dtype())


def build_rhs1_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    builders: RHS1SchurPreconditionerBuilders,
    geom_scheme: int = 0,
) -> Preconditioner:
    """Approximate Schur-complement preconditioner for constraintScheme=2 RHSMode=1 solves."""
    precond_dtype = _precond_dtype()
    geom_scheme = int(geom_scheme or 0)
    base_xblock_tz_lmax = 0
    base_kind_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    use_dkes_exb = bool(getattr(op.fblock.exb_theta, "use_dkes_exb_drift", False)) or bool(
        getattr(op.fblock.exb_zeta, "use_dkes_exb_drift", False)
    )
    base_kind = resolve_rhs1_schur_base_kind(
        base_kind_env=base_kind_env,
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        n_species=int(op.n_species),
        total_size=int(op.total_size),
        nxi_for_x=nxi_for_x,
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        has_er_xdot=op.fblock.er_xdot is not None,
        has_er_xidot=op.fblock.er_xidot is not None,
        use_dkes_exb=use_dkes_exb,
        pas_tokamak_theta_applicable=builders.pas_tokamak_theta_applicable(op),
        pas_tz_applicable=builders.pas_tz_applicable(op),
        geom_scheme=geom_scheme,
    )

    if base_kind == "theta_line":
        base_precond = builders.theta_line_builder(op=op)
    elif base_kind == "theta_dd":
        dd_block_env = os.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_T", "").strip()
        try:
            dd_block = int(dd_block_env) if dd_block_env else 8
        except ValueError:
            dd_block = 8
        base_precond = builders.theta_dd_builder(op=op, block=dd_block)
    elif base_kind == "species_block":
        base_precond = builders.species_block_builder(op=op)
    elif base_kind == "sxblock_tz":
        base_precond = builders.sxblock_tz_builder(op=op)
    elif base_kind == "xblock_tz":
        base_precond = builders.xblock_tz_builder(op=op)
    elif base_kind == "xblock_tz_lmax":
        lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_XBLOCK_TZ_LMAX", "").strip()
        if not lmax_env:
            lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "").strip()
        try:
            lmax_use = int(lmax_env) if lmax_env else int(base_xblock_tz_lmax)
        except ValueError:
            lmax_use = int(base_xblock_tz_lmax)
        base_precond = builders.xblock_tz_lmax_builder(op=op, lmax=int(lmax_use))
    elif base_kind == "pas_ilu":
        base_precond = builders.pas_xblock_ilu_builder(op=op)
    elif base_kind == "xmg":
        base_precond = builders.xmg_builder(op=op)
    elif base_kind == "pas_lite":
        base_precond = builders.pas_lite_builder(op=op, safe=False)
    elif base_kind == "pas_hybrid":
        base_precond = builders.pas_hybrid_builder(op=op, safe=False)
    elif base_kind == "pas_schur":
        base_precond = builders.pas_schur_builder(op=op, safe=False)
    elif base_kind == "pas_tokamak_theta":
        base_precond = builders.pas_tokamak_theta_builder(op=op)
    elif base_kind == "pas_tz":
        base_precond = builders.pas_tz_builder(op=op)
    elif base_kind == "theta_zeta":
        base_precond = builders.theta_zeta_builder(op=op)
    elif base_kind == "zeta_line":
        base_precond = builders.zeta_line_builder(op=op)
    elif base_kind == "zeta_dd":
        dd_block_env = os.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_Z", "").strip()
        try:
            dd_block = int(dd_block_env) if dd_block_env else 8
        except ValueError:
            dd_block = 8
        base_precond = builders.zeta_dd_builder(op=op, block=dd_block)
    elif base_kind == "adi":
        pre_theta = builders.theta_line_builder(op=op)
        pre_zeta = builders.zeta_line_builder(op=op)
        sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
        try:
            sweeps = int(sweeps_env) if sweeps_env else 2
        except ValueError:
            sweeps = 2
        sweeps = max(1, sweeps)

        def base_precond(v: jnp.ndarray) -> jnp.ndarray:
            out = v
            for _ in range(sweeps):
                out = pre_zeta(pre_theta(out))
            return out
    else:
        base_precond = builders.block_builder(op=op)
    f_size = int(op.f_size)
    extra_size = int(op.extra_size)
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_constraints = int(extra_size)
    ix0 = _ix_min(bool(op.point_at_x0))

    theta_w = np.asarray(op.theta_weights, dtype=np.float64)
    zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
    d_hat = np.asarray(op.d_hat, dtype=np.float64)
    fs_sum = float(np.sum((theta_w[:, None] * zeta_w[None, :]) / d_hat))

    # Large PAS+Er constrained systems are expensive to precondition if we assemble a dense Schur
    # complement via repeated full-size preconditioner applications. For the important PAS branch,
    # the constraint operator only touches the L=0 flux-surface average, so we can approximate
    # S^{-1} using an x-coupled L=0 block (dense in x) built from collisions + averaged Er xDot.
    #
    # This is a "block-Schur with x-coarsening" strategy:
    # - Base preconditioner: x-coarse (xmg) for the kinetic block M^{-1}.
    # - Schur inverse approximation: (1/<1>) * M_L0, where <1>=fs_sum and M_L0 is an x-coupled
    #   approximation to the L=0 rows/cols.
    xschur_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_XMG_MIN", "").strip()
    try:
        xschur_min = int(xschur_min_env) if xschur_min_env else 50000
    except ValueError:
        xschur_min = 50000
    use_xschur = bool(
        base_kind in {"xmg", "pas_lite", "pas_hybrid", "pas_tz", "pas_schur"}
        and op.fblock.pas is not None
        and op.fblock.fp is None
        and op.fblock.er_xdot is not None
        and int(op.total_size) >= xschur_min
        and fs_sum != 0.0
    )
    a0_cached: jnp.ndarray | None = None  # (S,X,X)
    if use_xschur:
        try:
            er = op.fblock.er_xdot
            assert er is not None
            pas = op.fblock.pas
            assert pas is not None

            alpha = float(np.asarray(er.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(er.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(er.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            d_hat_er = np.asarray(er.d_hat, dtype=np.float64)  # (T,Z)
            b_hat = np.asarray(er.b_hat, dtype=np.float64)  # (T,Z)
            b_sub_theta = np.asarray(er.b_hat_sub_theta, dtype=np.float64)  # (T,Z)
            b_sub_zeta = np.asarray(er.b_hat_sub_zeta, dtype=np.float64)  # (T,Z)
            db_dtheta = np.asarray(er.db_hat_dtheta, dtype=np.float64)  # (T,Z)
            db_dzeta = np.asarray(er.db_hat_dzeta, dtype=np.float64)  # (T,Z)

            factor0 = -(alpha * delta * dphi) / 4.0
            xdot_factor = factor0 * d_hat_er / (b_hat**3) * (b_sub_theta * db_dzeta - b_sub_zeta * db_dtheta)  # (T,Z)
            fs_factor = (theta_w[:, None] * zeta_w[None, :]) / d_hat  # (T,Z)
            fs_sum_safe = float(np.sum(fs_factor))
            if fs_sum_safe > 0.0:
                xdot_rms = float(np.sqrt(np.sum(fs_factor * (xdot_factor * xdot_factor)) / fs_sum_safe))
            else:
                xdot_rms = 0.0

            # Diagonal-in-L coefficient for L=0 from `apply_er_xdot_v3`:
            # diag_coef = 2*(3L^2+3L-2) / ((2L+3)(2L-1)) -> 4/3 at L=0.
            diag_coef0 = 4.0 / 3.0
            xdot_coef0 = diag_coef0 * xdot_rms

            x_arr = np.asarray(er.x, dtype=np.float64)
            ddx = np.asarray(er.ddx_plus, dtype=np.float64)
            x_part = x_arr[:, None] * ddx  # (X,X)

            pas_coef = np.asarray(pas.coef, dtype=np.float64)  # (S,X,L)
            identity_shift = float(op.fblock.identity_shift)
            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_XMG_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-12
            except ValueError:
                reg = 1e-12

            nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
            inactive_x = np.where(nxi_for_x <= 0)[0]
            a0 = np.zeros((n_species, n_x, n_x), dtype=np.float64)
            for s in range(n_species):
                a = np.array(xdot_coef0 * x_part, dtype=np.float64, copy=True)
                a[np.arange(n_x), np.arange(n_x)] += identity_shift + reg + pas_coef[s, :, 0]
                if inactive_x.size:
                    for ix in inactive_x:
                        a[ix, :] = 0.0
                        a[:, ix] = 0.0
                        a[ix, ix] = 1.0
                a0[s, :, :] = a
            a0_cached = jnp.asarray(a0, dtype=precond_dtype)
        except Exception:  # noqa: BLE001
            use_xschur = False
            a0_cached = None

    def _schur_inv_diag() -> jnp.ndarray:
        cache_key = _rhsmode1_precond_cache_key(op, "schur_diag")
        cached = _RHSMODE1_SCHUR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Ensure base block preconditioner cache exists.
        builders.block_builder(
            op=op,
            preconditioner_species=1,
            preconditioner_x=0,
            preconditioner_xi=1,
        )
        block_key = _rhsmode1_precond_cache_key(op, "point")
        block_cached = _RHSMODE1_PRECOND_CACHE.get(block_key)
        if block_cached is None:
            raise RuntimeError("Schur preconditioner requires block preconditioner cache.")
        block_inv = np.asarray(block_cached.block_inv_jnp, dtype=np.float64)
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(nxi_for_x)])
        idx = offsets[:-1]
        diag = np.zeros((n_species, n_x), dtype=np.float64)
        for s in range(n_species):
            diag[s, :] = block_inv[s, idx, idx]
        theta_w = np.asarray(op.theta_weights, dtype=np.float64)
        zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
        d_hat = np.asarray(op.d_hat, dtype=np.float64)
        fs_sum = float(np.sum((theta_w[:, None] * zeta_w[None, :]) / d_hat))
        diag = diag * fs_sum
        eps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_EPS", "").strip()
        try:
            eps = float(eps_env) if eps_env else 1e-14
        except ValueError:
            eps = 1e-14
        inv_diag = np.zeros_like(diag)
        mask = np.abs(diag) > eps
        inv_diag[mask] = 1.0 / diag[mask]
        ix0 = _ix_min(bool(op.point_at_x0))
        if ix0 > 0:
            inv_diag[:, :ix0] = 0.0
        inv_diag_jnp = jnp.asarray(inv_diag, dtype=precond_dtype)
        _RHSMODE1_SCHUR_CACHE[cache_key] = inv_diag_jnp
        return inv_diag_jnp

    def _schur_inv_full() -> jnp.ndarray:
        cache_key = _rhsmode1_precond_cache_key(op, f"schur_full_{n_constraints}")
        cached = _RHSMODE1_SCHUR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Build a dense approximate Schur complement: S ~= C M^{-1} B.
        # M^{-1} is the block preconditioner for the f-block.
        if n_constraints <= 0:
            inv = np.zeros((0, 0), dtype=np.float64)
            inv_jnp = jnp.asarray(inv, dtype=jnp.float64)
            _RHSMODE1_SCHUR_CACHE[cache_key] = inv_jnp
            return inv_jnp
        if base_kind == "pas_ilu":
            # For block-Jacobi (species,x) ILU bases, S ~= C M^{-1} B is diagonal in the
            # (species,x) constraint index. Building the full Schur matrix by repeatedly
            # applying M^{-1} is wasteful and can dominate runtime for PAS DKES cases.
            #
            # Instead, compute the diagonal using the ILU factors in NumPy (no JAX calls),
            # then return a diagonal dense inverse for compatibility with the existing
            # preconditioner application.
            factors = getattr(base_precond, "_sfincs_pas_ilu_factors", None)
            block_size_max = getattr(base_precond, "_sfincs_pas_ilu_block_size_max", None)
            if factors is not None and block_size_max is not None and int(n_constraints) == int(n_species) * int(n_x):
                (
                    inv_perm_r_sx_jnp,
                    perm_c_sx_jnp,
                    lower_idx_sx_jnp,
                    lower_val_sx_jnp,
                    upper_idx_sx_jnp,
                    upper_val_sx_jnp,
                    upper_diag_sx_jnp,
                ) = factors
                inv_perm_r_sx_np = np.asarray(inv_perm_r_sx_jnp, dtype=np.int32)
                perm_c_sx_np = np.asarray(perm_c_sx_jnp, dtype=np.int32)
                lower_idx_sx_np = np.asarray(lower_idx_sx_jnp, dtype=np.int32)
                lower_val_sx_np = np.asarray(lower_val_sx_jnp, dtype=np.float64)
                upper_idx_sx_np = np.asarray(upper_idx_sx_jnp, dtype=np.int32)
                upper_val_sx_np = np.asarray(upper_val_sx_jnp, dtype=np.float64)
                upper_diag_sx_np = np.asarray(upper_diag_sx_jnp, dtype=np.float64)

                n_theta = int(op.n_theta)
                n_zeta = int(op.n_zeta)
                n_tz = int(n_theta * n_zeta)
                factor_tz = ((theta_w[:, None] * zeta_w[None, :]) / d_hat).reshape((-1,))  # (TZ,)

                rhs_unit = np.zeros((int(block_size_max),), dtype=np.float64)
                rhs_unit[:n_tz] = 1.0

                def _solve_block_np(
                    rhs: np.ndarray,
                    inv_perm_r: np.ndarray,
                    perm_c: np.ndarray,
                    lower_idx: np.ndarray,
                    lower_val: np.ndarray,
                    upper_idx: np.ndarray,
                    upper_val: np.ndarray,
                    upper_diag: np.ndarray,
                ) -> np.ndarray:
                    rhs_perm = rhs[inv_perm_r]
                    n = int(rhs_perm.shape[0])
                    y = np.zeros((n,), dtype=np.float64)
                    if int(lower_idx.shape[1]) > 0:
                        for i in range(n):
                            cols = lower_idx[i]
                            vals = lower_val[i]
                            mask = cols >= 0
                            if np.any(mask):
                                contrib = float(np.dot(vals[mask], y[cols[mask]]))
                            else:
                                contrib = 0.0
                            y[i] = rhs_perm[i] - contrib
                    else:
                        y = rhs_perm
                    x = np.zeros((n,), dtype=np.float64)
                    if int(upper_idx.shape[1]) > 0:
                        for i in range(n - 1, -1, -1):
                            cols = upper_idx[i]
                            vals = upper_val[i]
                            mask = cols >= 0
                            if np.any(mask):
                                contrib = float(np.dot(vals[mask], x[cols[mask]]))
                            else:
                                contrib = 0.0
                            x[i] = (y[i] - contrib) / float(upper_diag[i])
                    else:
                        x = y / upper_diag
                    return x[perm_c]

                diag = np.zeros((n_constraints,), dtype=np.float64)
                for j in range(int(n_constraints)):
                    s = int(j // int(n_x))
                    ix = int(j - s * int(n_x))
                    if ix0 > 0 and ix < int(ix0):
                        # For pointAtX0, the x=0 constraint rows are enforced directly.
                        diag[j] = 1.0
                        continue
                    sol = _solve_block_np(
                        rhs_unit,
                        inv_perm_r_sx_np[s, ix],
                        perm_c_sx_np[s, ix],
                        lower_idx_sx_np[s, ix],
                        lower_val_sx_np[s, ix],
                        upper_idx_sx_np[s, ix],
                        upper_val_sx_np[s, ix],
                        upper_diag_sx_np[s, ix],
                    )
                    diag[j] = float(np.dot(factor_tz, sol[:n_tz]))

                reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_REG", "").strip()
                try:
                    reg = float(reg_env) if reg_env else 1e-12
                except ValueError:
                    reg = 1e-12
                diag = diag + float(reg)
                eps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_EPS", "").strip()
                try:
                    eps = float(eps_env) if eps_env else 1e-14
                except ValueError:
                    eps = 1e-14
                inv_diag = np.zeros_like(diag)
                mask = np.abs(diag) > float(eps)
                inv_diag[mask] = 1.0 / diag[mask]
                s_inv = np.diag(inv_diag)
                s_inv_jnp = jnp.asarray(s_inv, dtype=precond_dtype)
                _RHSMODE1_SCHUR_CACHE[cache_key] = s_inv_jnp
                return s_inv_jnp
        # Fast path: if constraints are per-(species,x) and the base preconditioner is
        # block-diagonal in x (i.e. the operator has no x-coupling), then the approximate
        # Schur complement S ~= C M^{-1} B is diagonal in the (species,x) index.
        #
        # Instead of building S column-by-column via n_constraints preconditioner applications,
        # recover the diagonal with a *single* base-preconditioner apply by injecting all
        # constraint sources simultaneously. This is a large win for tokamak-like PAS runs
        # with many x points.
        base_has_x_coupling = bool(
            op.fblock.fp is not None or op.fblock.er_xdot is not None or op.fblock.er_xidot is not None
        )
        # Some bases (point/species/sxblock) deliberately introduce x-coupling even if the
        # underlying operator is x-block-diagonal. Exclude them from the diagonal-Schur shortcut.
        base_is_x_coupled_kind = base_kind in {
            "xmg",
            "pas_lite",
            "pas_hybrid",
            "pas_tz",
            "point",
            "species_block",
            "sxblock_tz",
        }
        if (
            int(n_constraints) == int(n_species) * int(n_x)
            and (not base_has_x_coupling)
            and (not base_is_x_coupled_kind)
        ):
            zeros_e = jnp.zeros((extra_size,), dtype=jnp.float64)
            src = np.ones((n_species, n_x), dtype=np.float64)
            if ix0 > 0:
                src[:, :ix0] = 0.0
            f_src = np.zeros(op.fblock.f_shape, dtype=np.float64)
            f_src[:, ix0:, 0, :, :] = src[:, ix0:, None, None]
            y_full = base_precond(
                jnp.concatenate([jnp.asarray(f_src.reshape((-1,)), dtype=jnp.float64), zeros_e], axis=0)
            )
            y_f = np.asarray(y_full[:f_size], dtype=np.float64).reshape(op.fblock.f_shape)
            factor = ((theta_w[:, None] * zeta_w[None, :]) / d_hat).astype(np.float64, copy=False)  # (T,Z)
            diag_sx = np.einsum("tz,sxtz->sx", factor, y_f[:, :, 0, :, :])  # (S,X)
            diag = diag_sx.reshape((-1,)).copy()
            if ix0 > 0:
                # For pointAtX0, the x=0 constraint rows are enforced directly.
                for j in range(int(n_constraints)):
                    s = int(j // int(n_x))
                    ix = int(j - s * int(n_x))
                    if ix < int(ix0):
                        diag[j] = 1.0
            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-12
            except ValueError:
                reg = 1e-12
            diag = diag + float(reg)
            eps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_EPS", "").strip()
            try:
                eps = float(eps_env) if eps_env else 1e-14
            except ValueError:
                eps = 1e-14
            inv_diag = np.zeros_like(diag)
            mask = np.abs(diag) > float(eps)
            inv_diag[mask] = 1.0 / diag[mask]
            s_inv = np.diag(inv_diag)
            s_inv_jnp = jnp.asarray(s_inv, dtype=precond_dtype)
            _RHSMODE1_SCHUR_CACHE[cache_key] = s_inv_jnp
            return s_inv_jnp
        zeros_e = jnp.zeros((extra_size,), dtype=jnp.float64)
        s_mat = np.zeros((n_constraints, n_constraints), dtype=np.float64)
        # Build columns of S by applying M^{-1} to constraint injections.
        for j in range(n_constraints):
            basis = np.zeros((n_species, n_x), dtype=np.float64)
            basis.reshape(-1)[j] = 1.0
            f_src = constraint_scheme2_inject_source(op, basis).reshape((-1,))
            y_full = base_precond(jnp.concatenate([jnp.asarray(f_src, dtype=jnp.float64), zeros_e], axis=0))
            y_f = np.asarray(y_full[:f_size], dtype=np.float64).reshape(op.fblock.f_shape)
            c_y = np.asarray(constraint_scheme2_source_from_f(op, y_f), dtype=np.float64).reshape((-1,))
            s_mat[:, j] = c_y
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-12
        except ValueError:
            reg = 1e-12
        s_mat = s_mat + reg * np.eye(n_constraints, dtype=np.float64)
        try:
            s_inv = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_mat, rcond=1e-12)
        if not np.all(np.isfinite(s_inv)):
            s_inv = np.linalg.pinv(s_mat, rcond=1e-12)
        s_inv_jnp = jnp.asarray(s_inv, dtype=precond_dtype)
        _RHSMODE1_SCHUR_CACHE[cache_key] = s_inv_jnp
        return s_inv_jnp

    schur_mode_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_MODE", "").strip().lower()
    if schur_mode_env in {"full", "dense"}:
        schur_mode = "full"
    elif schur_mode_env in {"diag", "diagonal"}:
        schur_mode = "diag"
    elif schur_mode_env in {"auto", ""}:
        schur_mode = "auto"
    else:
        schur_mode = "diag"
    schur_full_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX", "").strip()
    try:
        schur_full_max = int(schur_full_max_env) if schur_full_max_env else 256
    except ValueError:
        schur_full_max = 256
    use_full_schur = bool(schur_mode == "full" or (schur_mode == "auto" and n_constraints <= schur_full_max))

    # Building the diagonal Schur approximation currently requires constructing the
    # (theta,zeta)-point block preconditioner cache, which can be expensive for large
    # active DOF systems. If we are going to use the dense (small) Schur complement,
    # skip the diagonal approximation entirely.
    if use_xschur:
        inv_diag_cached = None
        inv_schur_cached = None
    else:
        inv_diag_cached = _schur_inv_diag() if (not use_full_schur) else None
        inv_schur_cached = _schur_inv_full() if use_full_schur else None

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        if int(op.rhs_mode) != 1 or int(op.constraint_scheme) != 2 or int(op.phi1_size) != 0 or extra_size == 0:
            return base_precond(r_full)
        r_f = r_full[:f_size]
        r_e = r_full[f_size:]
        zeros_e = jnp.zeros((extra_size,), dtype=r_full.dtype)
        y_full = base_precond(jnp.concatenate([r_f, zeros_e], axis=0))
        y_f = y_full[:f_size]
        f = y_f.reshape(op.fblock.f_shape)
        c_y = constraint_scheme2_source_from_f(op, f).reshape((-1,))
        if use_xschur and a0_cached is not None:
            y = (c_y - r_e).reshape((n_species, n_x))
            x_e = jnp.einsum("sij,sj->si", a0_cached, y) / float(fs_sum)
            if ix0 > 0:
                # For pointAtX0, the x=0 constraint row is `y_extra = src`, so treat it as identity.
                r_e_mat = r_e.reshape((n_species, n_x))
                x_e = x_e.at[:, :ix0].set(r_e_mat[:, :ix0])
            x_e = x_e.reshape((-1,))
        elif use_full_schur and inv_schur_cached is not None:
            x_e = inv_schur_cached @ (c_y - r_e)
        else:
            inv_diag = inv_diag_cached.reshape((-1,)) if inv_diag_cached is not None else jnp.zeros_like(r_e)
            x_e = (c_y - r_e) * inv_diag
        f_corr = constraint_scheme2_inject_source(op, x_e.reshape((n_species, n_x)))
        r_corr = r_f - f_corr
        y_corr = base_precond(jnp.concatenate([r_corr, zeros_e], axis=0))
        x_f = y_corr[:f_size]
        return jnp.concatenate([x_f, x_e], axis=0)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


@dataclass(frozen=True)
class ActiveNativeStackPolicy:
    """Resolved controls for the bounded native-line/SCHWARZ/coarse stack."""

    base_budget_fraction: float
    base_budget_nbytes: int
    schwarz_requested: bool
    schwarz_max_size: int
    max_coarse_size: int
    coarse_solver_mode: str


@dataclass(frozen=True)
class ActiveNativeFieldSplitSparseCoarsePolicy:
    """Resolved controls for native field-split plus sparse coarse correction."""

    requested_kind: str
    requested_kind_normalized: str
    output_kind: str
    requested_base_kind: str
    is_multiline: bool
    is_angular_only: bool
    is_coupled_kinetic: bool
    max_coarse_size: int
    coarse_solver_mode: str
    admission_probes: int
    admission_max_relative_residual: float
    admission_min_improvement: float


@dataclass(frozen=True)
class ActiveSparseCoarseResidualPolicy:
    """Resolved controls for tail/Schwarz/filtered sparse coarse correction."""

    requested_kind: str
    requested_kind_normalized: str
    base_kind: str
    output_kind: str
    max_coarse_size: int
    coarse_solver_mode: str


def resolve_active_native_stack_policy(
    *,
    max_factor_nbytes: int,
    env: Mapping[str, str] | None = None,
) -> ActiveNativeStackPolicy:
    """Resolve bounded native-stack memory and coarse-equation controls."""

    env_map = os.environ if env is None else env
    base_budget_fraction = min(
        max(
            float(
                _policy_env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION",
                    0.75,
                )
            ),
            0.05,
        ),
        1.0,
    )
    default_coarse_size = _policy_env_int(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE",
        640,
    )
    return ActiveNativeStackPolicy(
        base_budget_fraction=float(base_budget_fraction),
        base_budget_nbytes=max(1, int(float(max_factor_nbytes) * float(base_budget_fraction))),
        schwarz_requested=_policy_env_bool(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ", False),
        schwarz_max_size=int(
            _policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE", 100_000)
        ),
        max_coarse_size=max(
            1,
            int(
                _policy_env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE",
                    int(default_coarse_size),
                )
            ),
        ),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER", "least_squares"),
            galerkin_aliases=("galerkin", "petrov_galerkin", "ztaz"),
            least_squares_aliases=(),
            default="least_squares",
        ),
    )


def resolve_active_native_field_split_sparse_coarse_policy(
    *,
    requested_kind: str,
    env: Mapping[str, str] | None = None,
) -> ActiveNativeFieldSplitSparseCoarsePolicy:
    """Resolve routing and admission controls for native field-split coarse paths."""

    env_map = os.environ if env is None else env
    requested_kind_l = _normalize_token(requested_kind)
    is_multiline = "multiline" in requested_kind_l or "xell_angular" in requested_kind_l
    is_angular_only = "angular" in requested_kind_l and not bool(is_multiline)
    is_coupled_kinetic = "coupled_kinetic" in requested_kind_l or "dominant_kinetic" in requested_kind_l

    output_kind = "active_native_xell_field_split_sparse_coarse"
    if bool(is_coupled_kinetic):
        output_kind = "active_coupled_kinetic_field_split_sparse_coarse"
    if bool(is_angular_only):
        output_kind = "active_angular_line_field_split_sparse_coarse"
    if bool(is_multiline):
        output_kind = "active_multiline_field_split_sparse_coarse"

    requested_base_kind = "active_multiline_xell_angular"
    if bool(is_coupled_kinetic):
        requested_base_kind = "active_coupled_kinetic_block"
    elif not bool(is_multiline):
        requested_base_kind = (
            "active_angular_line"
            if bool(is_angular_only)
            else _normalize_token(
                env_map.get(
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LINE_FIELD_SPLIT_BASE",
                    "active_native_xell",
                )
            )
            or "active_native_xell"
        )

    sparse_default = _policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640)
    coarse_size_key = (
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE"
        if bool(is_coupled_kinetic)
        else "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE"
    )

    return ActiveNativeFieldSplitSparseCoarsePolicy(
        requested_kind=str(requested_kind),
        requested_kind_normalized=str(requested_kind_l),
        output_kind=str(output_kind),
        requested_base_kind=str(requested_base_kind),
        is_multiline=bool(is_multiline),
        is_angular_only=bool(is_angular_only),
        is_coupled_kinetic=bool(is_coupled_kinetic),
        max_coarse_size=max(1, int(_policy_env_int(env_map, coarse_size_key, int(sparse_default)))),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER", "least_squares"),
            galerkin_aliases=("galerkin", "petrov_galerkin", "ztaz"),
            least_squares_aliases=(),
            default="least_squares",
        ),
        admission_probes=max(
            1,
            int(
                _policy_env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES",
                    4,
                )
            ),
        ),
        admission_max_relative_residual=max(
            0.0,
            float(
                _policy_env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        admission_min_improvement=max(
            0.0,
            float(
                _policy_env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
    )


def resolve_active_sparse_coarse_residual_policy(
    *,
    requested_kind: str,
    env: Mapping[str, str] | None = None,
) -> ActiveSparseCoarseResidualPolicy:
    """Resolve base-preconditioner and coarse-equation controls for sparse coarse paths."""

    env_map = os.environ if env is None else env
    requested_norm = _normalize_token(requested_kind)
    if "scaled_ilu" in requested_norm or "equilibrated_ilu" in requested_norm or "rowcol_ilu" in requested_norm:
        base_kind = "active_scaled_ilu"
    elif "schwarz" in requested_norm or "ras" in requested_norm:
        base_kind = "active_overlap_schwarz"
    elif "filtered" in requested_norm:
        base_kind = "active_filtered_sparse_factor"
    elif "xblock" in requested_norm:
        base_kind = "active_xblock"
    else:
        base_kind = "active_diagonal_schur"

    default_coarse_solver = "least_squares" if str(base_kind) == "active_filtered_sparse_factor" else "galerkin"
    return ActiveSparseCoarseResidualPolicy(
        requested_kind=str(requested_kind),
        requested_kind_normalized=str(requested_norm),
        base_kind=str(base_kind),
        output_kind=(
            "active_filtered_sparse_coarse"
            if str(base_kind) == "active_filtered_sparse_factor"
            else "active_tail_sparse_coarse"
        ),
        max_coarse_size=max(
            1,
            int(_policy_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640)),
        ),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER", default_coarse_solver),
            galerkin_aliases=(),
            least_squares_aliases=("least_squares", "normal", "normal_equations"),
            default=str(default_coarse_solver),
        ),
    )


def _normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _coarse_solver_mode(
    value: object,
    *,
    galerkin_aliases: tuple[str, ...],
    least_squares_aliases: tuple[str, ...],
    default: str,
) -> str:
    token = _normalize_token(value)
    if not token:
        token = _normalize_token(default)
    if token in set(galerkin_aliases):
        return "galerkin"
    if token in set(least_squares_aliases):
        return "least_squares"
    if _normalize_token(default) == "galerkin":
        return "galerkin"
    return "least_squares"


def _policy_env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _policy_env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def _policy_env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)

def xblock_tz_low_l_config(layout: Any) -> dict[str, object]:
    """Resolve low-L ``x``-block sparse-factor controls for coarse estimates."""

    lmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_LMAX", 8)
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    factor_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FACTOR_KIND", "splu").strip().lower()
    if factor_kind not in {"splu", "spilu"}:
        factor_kind = "splu"
    return {
        "lmax": int(lmax),
        "drop_tol": float(_basis_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_DROP_TOL", 0.0)),
        "fill_factor": float(_basis_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FILL_FACTOR", 8.0)),
        "factor_kind": factor_kind,
    }


def coarse_residual_config(layout: Any) -> dict[str, object]:
    """Resolve the physics low-mode basis used by RHSMode=1 coarse residual solves."""

    config = xblock_tz_low_l_config(layout)
    coarse_lmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_LMAX", min(4, int(layout.n_xi)))
    coarse_lmax = max(1, min(int(layout.n_xi), int(coarse_lmax)))
    angular_mmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_MMAX", 1)
    angular_nmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_NMAX", 1)
    helical_mmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_MMAX", 1)
    helical_nmax = _basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_NMAX", min(4, int(layout.n_zeta) // 2))
    angular_mmax = max(0, min(int(layout.n_theta) // 2, int(angular_mmax)))
    angular_nmax = max(0, min(int(layout.n_zeta) // 2, int(angular_nmax)))
    helical_mmax = max(0, min(int(layout.n_theta) // 2, int(helical_mmax)))
    helical_nmax = max(0, min(int(layout.n_zeta) // 2, int(helical_nmax)))
    has_angular_modes = any((angular_mmax, angular_nmax, helical_mmax and helical_nmax))
    config.update(
        {
            "coarse_lmax": int(coarse_lmax),
            "coarse_include_tail": True,
            "coarse_angular_mmax": int(angular_mmax),
            "coarse_angular_nmax": int(angular_nmax),
            "coarse_helical_mmax": int(helical_mmax),
            "coarse_helical_nmax": int(helical_nmax),
            "coarse_basis": (
                "flux_surface_low_l_angular_plus_tail"
                if has_angular_modes
                else "flux_surface_low_l_plus_tail"
            ),
        }
    )
    return config


def estimate_xblock_tz_low_l_factor_nbytes(*, layout: Any, config: dict[str, object]) -> int:
    """Return a conservative sparse-factor memory estimate for low-L x-blocks."""

    block_size = int(config["lmax"]) * int(layout.n_theta) * int(layout.n_zeta)
    n_blocks = int(layout.n_species) * int(layout.n_x)
    # Sparse factors should be much smaller than dense inverse blocks. This
    # conservative cap estimate prevents accidental full-resolution promotion.
    return int(n_blocks * block_size * min(block_size, 64) * np.dtype(np.float64).itemsize)


def estimate_coarse_residual_nbytes(*, layout: Any, config: dict[str, object]) -> int:
    """Estimate sparse basis plus dense coarse-equation storage in bytes."""

    coarse_lmax = int(config["coarse_lmax"])
    surface_mode_count = coarse_surface_mode_count(layout=layout, config=config)
    coarse_kinetic = int(layout.n_species) * int(layout.n_x) * int(coarse_lmax) * int(surface_mode_count)
    coarse_tail = int(layout.total_size) - int(layout.f_size)
    coarse_size = int(coarse_kinetic + coarse_tail)
    basis_nnz = int(coarse_kinetic * layout.n_theta * layout.n_zeta + coarse_tail)
    sparse_bytes = int(basis_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
    sparse_bytes += int((coarse_size + 1) * np.dtype(np.int32).itemsize)
    dense_bytes = int(coarse_size * coarse_size * np.dtype(np.float64).itemsize)
    return int(sparse_bytes + dense_bytes)


def build_coarse_residual_basis_csc(*, layout: Any, config: dict[str, object]) -> Any:
    """Build the sparse full-space low-mode coarse residual basis."""

    coarse_lmax = int(config["coarse_lmax"])
    surface_modes = coarse_surface_modes(layout=layout, config=config)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    col = 0
    ntz = int(layout.n_theta) * int(layout.n_zeta)
    theta = np.arange(int(layout.n_theta), dtype=np.int64)
    zeta = np.arange(int(layout.n_zeta), dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            for ell in range(coarse_lmax):
                idx = (
                    (((int(species) * int(layout.n_x) + int(x)) * int(layout.n_xi) + int(ell)) * int(layout.n_theta) + theta_grid)
                    * int(layout.n_zeta)
                    + zeta_grid
                ).astype(np.int64, copy=False).reshape((-1,))
                for _mode_name, surface_values in surface_modes:
                    rows.append(idx)
                    cols.append(np.full((ntz,), col, dtype=np.int64))
                    data.append(surface_values)
                    col += 1
    if bool(config.get("coarse_include_tail", True)):
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0:
            tail_rows = int(layout.f_size) + np.arange(tail_size, dtype=np.int64)
            rows.append(tail_rows)
            cols.append(np.arange(col, col + tail_size, dtype=np.int64))
            data.append(np.ones((tail_size,), dtype=np.float64))
            col += tail_size
    if not rows:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64)
    row = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(data)
    basis = sp.coo_matrix((values, (row, col_idx)), shape=(int(layout.total_size), int(col))).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    return basis


def build_active_native_xell_coarse_window_basis_csc(
    *,
    layout: Any,
) -> tuple[Any, dict[str, object]]:
    """Return optional identity columns for targeted active-native coarse windows."""

    spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS", "").strip()
    max_columns = max(
        0,
        int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_MAX_COLUMNS", 8192)),
    )
    metadata = {
        "window_basis_requested": bool(spec),
        "window_basis_specs": str(spec),
        "window_basis_columns": 0,
        "window_basis_max_columns": int(max_columns),
    }
    if not spec or int(max_columns) <= 0:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    ell_radius = max(
        0,
        int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_ELL_RADIUS", 1)),
    )
    x_radius = max(
        0,
        int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_X_RADIUS", 0)),
    )

    def parse_axis(raw: str, stop: int) -> tuple[int, ...]:
        item = str(raw).strip().lower()
        if item in {"*", "all"}:
            return tuple(range(int(stop)))
        try:
            value = int(item)
        except ValueError:
            return ()
        if value < 0 or value >= int(stop):
            return ()
        return (int(value),)

    selected: list[int] = []
    skipped_specs = 0
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.replace("/", ":").split(":")
        if len(parts) != 3:
            skipped_specs += 1
            continue
        species_values = parse_axis(parts[0], int(layout.n_species))
        x_centers = parse_axis(parts[1], int(layout.n_x))
        ell_centers = parse_axis(parts[2], int(layout.n_xi))
        if not species_values or not x_centers or not ell_centers:
            skipped_specs += 1
            continue
        for species in species_values:
            for x_center in x_centers:
                x_min = max(0, int(x_center) - int(x_radius))
                x_max = min(int(layout.n_x) - 1, int(x_center) + int(x_radius))
                for ell_center in ell_centers:
                    ell_min = max(0, int(ell_center) - int(ell_radius))
                    ell_max = min(int(layout.n_xi) - 1, int(ell_center) + int(ell_radius))
                    for x_index in range(x_min, x_max + 1):
                        for ell in range(ell_min, ell_max + 1):
                            for theta in range(int(layout.n_theta)):
                                for zeta in range(int(layout.n_zeta)):
                                    selected.append(
                                        layout.kinetic_flat_index(
                                            species=species,
                                            x=x_index,
                                            ell=ell,
                                            theta=theta,
                                            zeta=zeta,
                                        )
                                    )
                                    if len(selected) >= int(max_columns):
                                        break
                                if len(selected) >= int(max_columns):
                                    break
                            if len(selected) >= int(max_columns):
                                break
                        if len(selected) >= int(max_columns):
                            break
                    if len(selected) >= int(max_columns):
                        break
                if len(selected) >= int(max_columns):
                    break
            if len(selected) >= int(max_columns):
                break
        if len(selected) >= int(max_columns):
            break

    if not selected:
        metadata.update(
            {
                "window_basis_skipped_specs": int(skipped_specs),
                "window_basis_truncated": False,
            }
        )
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    rows = np.unique(np.asarray(selected, dtype=np.int64))
    if int(rows.size) > int(max_columns):
        rows = rows[: int(max_columns)]
    cols = np.arange(int(rows.size), dtype=np.int64)
    basis = sp.coo_matrix(
        (np.ones((int(rows.size),), dtype=np.float64), (rows, cols)),
        shape=(int(layout.total_size), int(rows.size)),
    ).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    metadata.update(
        {
            "window_basis_columns": int(basis.shape[1]),
            "window_basis_nnz": int(basis.nnz),
            "window_basis_ell_radius": int(ell_radius),
            "window_basis_x_radius": int(x_radius),
            "window_basis_skipped_specs": int(skipped_specs),
            "window_basis_truncated": bool(len(selected) >= int(max_columns)),
        }
    )
    return basis, metadata


def append_adaptive_residual_basis_csc(
    *,
    matrix: Any,
    base_operator: Any,
    basis: Any,
    max_total_columns: int,
) -> tuple[Any, dict[str, object]]:
    """Append bounded residual-derived coarse columns ``z - A M z``.

    The generated vectors are construction-time snapshots of the mismatch
    between the true active operator ``A`` and the selected base preconditioner
    ``M``. They are independent of the Krylov right-hand side, so the resulting
    preconditioner remains linear. Columns are sparsified by relative magnitude
    and capped by both column count and per-column nonzeros.
    """

    enabled = _basis_env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", False)
    max_columns = max(0, int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_COLUMNS", 32)))
    max_seed_columns = max(
        0,
        int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_SEED_COLUMNS", 64)),
    )
    max_nnz_per_column = max(
        1,
        int(_basis_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_NNZ_PER_COLUMN", 4096)),
    )
    drop_rel = max(
        0.0,
        float(_basis_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_DROP_REL", 1.0e-3)),
    )
    min_rel_norm = max(
        0.0,
        float(_basis_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MIN_REL_NORM", 1.0e-8)),
    )
    metadata = {
        "adaptive_residual_basis_enabled": bool(enabled),
        "adaptive_residual_basis_columns": 0,
        "adaptive_residual_basis_seed_columns": 0,
        "adaptive_residual_basis_max_columns": int(max_columns),
        "adaptive_residual_basis_max_seed_columns": int(max_seed_columns),
        "adaptive_residual_basis_max_nnz_per_column": int(max_nnz_per_column),
        "adaptive_residual_basis_drop_rel": float(drop_rel),
        "adaptive_residual_basis_min_rel_norm": float(min_rel_norm),
    }
    if not bool(enabled) or int(max_columns) <= 0 or int(max_seed_columns) <= 0 or int(basis.shape[1]) <= 0:
        return basis, metadata

    matrix_csr = matrix.tocsr()
    basis_csc = basis.tocsc()
    remaining = max(0, int(max_total_columns) - int(basis_csc.shape[1]))
    max_columns_use = min(int(max_columns), int(remaining))
    if max_columns_use <= 0:
        metadata["adaptive_residual_basis_truncated_by_total_cap"] = True
        return basis_csc, metadata

    seed_count = min(int(basis_csc.shape[1]), int(max_seed_columns))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    residual_norms: list[float] = []
    skipped_small = 0
    skipped_zero = 0
    for seed_col in range(seed_count):
        if len(rows) >= max_columns_use:
            break
        z = np.asarray(basis_csc[:, seed_col].toarray(), dtype=np.float64).reshape((-1,))
        z_norm = max(float(np.linalg.norm(z)), np.finfo(np.float64).tiny)
        mz = np.asarray(base_operator.matvec(z), dtype=np.float64).reshape((-1,))
        residual = z - np.asarray(matrix_csr @ mz, dtype=np.float64).reshape((-1,))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or residual_norm <= 0.0:
            skipped_zero += 1
            continue
        if residual_norm / z_norm < float(min_rel_norm):
            skipped_small += 1
            continue
        abs_residual = np.abs(residual)
        threshold = float(drop_rel) * max(float(np.max(abs_residual)), np.finfo(np.float64).tiny)
        keep = np.flatnonzero(abs_residual >= threshold)
        if keep.size > int(max_nnz_per_column):
            order = np.argpartition(abs_residual[keep], -int(max_nnz_per_column))[-int(max_nnz_per_column) :]
            keep = keep[order]
            keep.sort()
        if keep.size == 0:
            skipped_zero += 1
            continue
        values = residual[keep]
        values_norm = float(np.linalg.norm(values))
        if not np.isfinite(values_norm) or values_norm <= 0.0:
            skipped_zero += 1
            continue
        rows.append(keep.astype(np.int64, copy=False))
        cols.append(np.full((int(keep.size),), len(rows) - 1, dtype=np.int64))
        data.append((values / values_norm).astype(np.float64, copy=False))
        residual_norms.append(float(residual_norm))

    if not rows:
        metadata.update(
            {
                "adaptive_residual_basis_seed_columns": int(seed_count),
                "adaptive_residual_basis_skipped_small": int(skipped_small),
                "adaptive_residual_basis_skipped_zero": int(skipped_zero),
                "adaptive_residual_basis_truncated_by_total_cap": False,
            }
        )
        return basis_csc, metadata

    adaptive = sp.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(int(matrix_csr.shape[0]), int(len(rows))),
    ).tocsc()
    adaptive.sum_duplicates()
    adaptive.eliminate_zeros()
    combined = sp.hstack([basis_csc, adaptive], format="csc")
    metadata.update(
        {
            "adaptive_residual_basis_columns": int(adaptive.shape[1]),
            "adaptive_residual_basis_seed_columns": int(seed_count),
            "adaptive_residual_basis_nnz": int(adaptive.nnz),
            "adaptive_residual_basis_skipped_small": int(skipped_small),
            "adaptive_residual_basis_skipped_zero": int(skipped_zero),
            "adaptive_residual_basis_residual_norm_max": float(max(residual_norms)),
            "adaptive_residual_basis_residual_norm_min": float(min(residual_norms)),
            "adaptive_residual_basis_truncated_by_total_cap": bool(len(rows) >= max_columns_use),
        }
    )
    return combined, metadata


def coarse_surface_mode_count(*, layout: Any, config: dict[str, object]) -> int:
    """Return the number of retained normalized angular/helical surface modes."""

    return int(len(coarse_surface_modes(layout=layout, config=config)))


def coarse_surface_modes(*, layout: Any, config: dict[str, object]) -> tuple[tuple[str, np.ndarray], ...]:
    """Return normalized low-angle modes for the RHSMode=1 coarse residual space."""

    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    theta = 2.0 * np.pi * np.arange(n_theta, dtype=np.float64) / max(1, n_theta)
    zeta = 2.0 * np.pi * np.arange(n_zeta, dtype=np.float64) / max(1, n_zeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    modes: list[tuple[str, np.ndarray]] = []

    def add_mode(name: str, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape((-1,))
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm) or norm <= 0.0:
            return
        flat = flat / norm
        for _existing_name, existing in modes:
            # Avoid exact duplicate modes on tiny grids; the coarse solve still
            # has regularization, but removing duplicates keeps conditioning sane.
            if flat.shape == existing.shape and float(abs(np.dot(flat, existing))) > 1.0 - 1.0e-12:
                return
        modes.append((name, flat))

    add_mode("constant", np.ones((n_theta, n_zeta), dtype=np.float64))
    angular_mmax = int(config.get("coarse_angular_mmax", 0) or 0)
    angular_nmax = int(config.get("coarse_angular_nmax", 0) or 0)
    helical_mmax = int(config.get("coarse_helical_mmax", 0) or 0)
    helical_nmax = int(config.get("coarse_helical_nmax", 0) or 0)

    for m in range(1, max(0, angular_mmax) + 1):
        add_mode(f"cos_theta_{m}", np.cos(float(m) * theta_grid))
        add_mode(f"sin_theta_{m}", np.sin(float(m) * theta_grid))
    for n in range(1, max(0, angular_nmax) + 1):
        add_mode(f"cos_zeta_{n}", np.cos(float(n) * zeta_grid))
        add_mode(f"sin_zeta_{n}", np.sin(float(n) * zeta_grid))
    for m in range(1, max(0, helical_mmax) + 1):
        for n in range(1, max(0, helical_nmax) + 1):
            phase = float(m) * theta_grid - float(n) * zeta_grid
            add_mode(f"cos_helical_{m}_{n}", np.cos(phase))
            add_mode(f"sin_helical_{m}_{n}", np.sin(phase))
    return tuple(modes)


def _basis_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or int(default))
    except ValueError:
        return int(default)


def _basis_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or float(default))
    except ValueError:
        return float(default)


def _basis_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RHS1StructuredFullCSRPreconditioner:
    """Host-side inverse preconditioner used by the explicit CSR solve lane."""

    operator: Any | None
    selected: bool
    kind: str
    reason: str
    setup_s: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly preconditioner metadata."""

        return {
            "selected": bool(self.selected),
            "kind": str(self.kind),
            "reason": str(self.reason),
            "setup_s": float(self.setup_s),
            "metadata": dict(self.metadata),
        }


def build_jacobi_preconditioner(
    *,
    matrix: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
    reason: str,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a diagonal inverse fallback with regularized zero pivots."""

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    inv_diag, diag_meta = safe_inverse_diagonal(matrix.diagonal(), regularization=regularization)
    operator = LinearOperator(
        matrix.shape,
        matvec=lambda x: inv_diag * np.asarray(x, dtype=np.float64).reshape((-1,)),
        dtype=np.float64,
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="jacobi",
        reason=reason,
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={"requested_kind": str(requested_kind), **diag_meta},
    )


def build_diagonal_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a diagonal kinetic inverse plus exact dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    if tail_size <= 0:
        return build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="no_global_tail",
        )

    diag = matrix.diagonal()
    inv_f, diag_meta = safe_inverse_diagonal(diag[:n_f], regularization=regularization)
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()
    scaled_u = u.multiply(inv_f[:, None])
    schur = (w - v @ scaled_u).toarray()
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = inv_f * arr[:n_f]
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - inv_f * np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="diagonal_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **diag_meta,
        },
    )


def build_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a zeta-line kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    block_size = int(layout.n_zeta)
    n_blocks = n_f // block_size
    inverse_blocks, block_meta = build_zeta_diagonal_inverse_blocks(
        matrix=matrix,
        n_f=n_f,
        block_size=block_size,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((n_blocks, block_size))
        out = np.einsum("bij,bj->bi", inverse_blocks, flat, optimize=True)
        return out.reshape((-1,))

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(block_size),
            "n_blocks": int(n_blocks),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def build_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a pitch-line kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = build_xi_diagonal_inverse_blocks(
        matrix=matrix,
        layout=layout,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((-1,))
        gathered = flat[block_indices]
        block_values = np.einsum("bij,bj->bi", inverse_blocks, gathered, optimize=True)
        out = np.zeros((n_f,), dtype=np.float64)
        out[block_indices] = block_values
        return out

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="xi_block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(layout.n_xi),
            "n_blocks": int(block_indices.shape[0]),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def build_x_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build an x-pitch kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = build_x_xi_diagonal_inverse_blocks(
        matrix=matrix,
        layout=layout,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((-1,))
        gathered = flat[block_indices]
        block_values = np.einsum("bij,bj->bi", inverse_blocks, gathered, optimize=True)
        out = np.zeros((n_f,), dtype=np.float64)
        out[block_indices] = block_values
        return out

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="x_xi_block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(layout.n_x * layout.n_xi),
            "n_blocks": int(block_indices.shape[0]),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def build_zeta_diagonal_inverse_blocks(
    *,
    matrix: Any,
    n_f: int,
    block_size: int,
    regularization: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build dense inverses for contiguous zeta-line kinetic blocks."""

    n_blocks = int(n_f) // int(block_size)
    inverse_blocks = np.empty((n_blocks, int(block_size), int(block_size)), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block in range(n_blocks):
        start = block * int(block_size)
        stop = start + int(block_size)
        dense = np.asarray(matrix[start:stop, start:stop].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(int(block_size), dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, metadata


def build_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: Any,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dense inverses for active pitch-line kinetic blocks."""

    block_size = int(layout.n_xi)
    index_rows: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            for theta in range(int(layout.n_theta)):
                for zeta in range(int(layout.n_zeta)):
                    indices = [
                        layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
                        for ell in range(int(layout.n_xi))
                    ]
                    index_rows.append(np.asarray(indices, dtype=np.int64))
    block_indices = np.asarray(index_rows, dtype=np.int64)
    inverse_blocks = np.empty((block_indices.shape[0], block_size, block_size), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block, indices in enumerate(block_indices):
        dense = np.asarray(matrix[indices[:, None], indices].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(block_size, dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_index_nbytes_actual": int(block_indices.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, block_indices, metadata


def build_x_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: Any,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dense inverses for combined radial-pitch kinetic blocks."""

    block_size = int(layout.n_x) * int(layout.n_xi)
    index_rows: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for theta in range(int(layout.n_theta)):
            for zeta in range(int(layout.n_zeta)):
                indices = [
                    layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
                    for x in range(int(layout.n_x))
                    for ell in range(int(layout.n_xi))
                ]
                index_rows.append(np.asarray(indices, dtype=np.int64))
    block_indices = np.asarray(index_rows, dtype=np.int64)
    inverse_blocks = np.empty((block_indices.shape[0], block_size, block_size), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block, indices in enumerate(block_indices):
        dense = np.asarray(matrix[indices[:, None], indices].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(block_size, dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_index_nbytes_actual": int(block_indices.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, block_indices, metadata


def estimate_zeta_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for zeta-line dense inverse blocks."""

    block_size = int(layout.n_zeta)
    n_blocks = int(layout.f_size) // block_size
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def estimate_xi_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for pitch-line dense inverse blocks."""

    block_size = int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_x) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def estimate_x_xi_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for combined radial-pitch dense inverse blocks."""

    block_size = int(layout.n_x) * int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def safe_inverse_diagonal(diagonal: Any, *, regularization: float) -> tuple[np.ndarray, dict[str, object]]:
    """Invert a diagonal with a scale-aware floor and return pivot metadata."""

    diag = np.asarray(diagonal, dtype=np.float64).reshape((-1,))
    abs_diag = np.abs(diag)
    scale = max(float(np.max(abs_diag)) if abs_diag.size else 0.0, 1.0)
    floor = float(abs(regularization)) * scale
    if floor == 0.0:
        floor = np.finfo(np.float64).tiny
    safe = diag.copy()
    small = abs_diag <= floor
    signs = np.where(safe < 0.0, -1.0, 1.0)
    safe[small] = signs[small] * floor
    inv = 1.0 / safe
    metadata = {
        "diagonal_size": int(diag.size),
        "diagonal_abs_max": float(np.max(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_abs_min": float(np.min(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_floor": float(floor),
        "diagonal_regularized_count": int(np.count_nonzero(small)),
    }
    return inv, metadata
