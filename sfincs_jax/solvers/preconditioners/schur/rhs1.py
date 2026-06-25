"""RHSMode=1 constraint-source Schur preconditioner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os

import jax.numpy as jnp
import numpy as np

from ....preconditioner_caches import (
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_SCHUR_CACHE,
)
from ....preconditioner_context import precond_dtype as _precond_dtype
from ....preconditioner_setup import rhs_mode1_precond_cache_key
from sfincs_jax.operators.profile_response.sources import (
    constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f,
)
from ....v3_system import V3FullSystemOperator, _ix_min

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



__all__ = (
    "RHS1SchurPreconditionerBuilders",
    "canonical_schur_base_kind",
    "resolve_rhs1_schur_base_kind",
    "build_rhs1_schur_preconditioner",
)


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

