"""RHSMode=1 PAS applicability and memory policy helpers.

This module holds the small, pure policy functions that decide whether the
specialized PAS tokamak-theta and PAS-TZ preconditioners are eligible to run.
They are intentionally isolated from the large solve orchestration in
``v3_driver.py`` so they can be tested directly and reused from multiple
dispatch paths without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Callable
import os

import numpy as np

from .pas_smoother import adaptive_pas_smoother_allowed


def pas_tokamak_theta_preconditioner_applicable(op) -> bool:
    """Return whether the PAS tokamak theta/L preconditioner is applicable.

    The tokamak-theta branch is intended for PAS-only RHSMode=1 systems with no
    drift/X coupling terms. ``n_zeta == 1`` is the direct tokamak case, but we
    also admit effectively tokamak-like multi-zeta grids when the geometry is
    zeta-invariant to within ``SFINCS_JAX_PAS_TOKAMAK_TZ_TOL``.
    """
    if int(op.rhs_mode) != 1:
        return False
    if int(op.n_zeta) != 1:
        cl = op.fblock.collisionless
        if cl is None:
            return False
        try:
            b_hat = np.asarray(cl.b_hat, dtype=np.float64)
            b_sup_theta = np.asarray(cl.b_hat_sup_theta, dtype=np.float64)
            b_sup_zeta = np.asarray(cl.b_hat_sup_zeta, dtype=np.float64)
            db_dtheta = np.asarray(cl.db_hat_dtheta, dtype=np.float64)
            db_dzeta = np.asarray(cl.db_hat_dzeta, dtype=np.float64)
        except Exception:
            return False
        tol_env = os.environ.get("SFINCS_JAX_PAS_TOKAMAK_TZ_TOL", "").strip()
        try:
            tol = float(tol_env) if tol_env else 1e-12
        except ValueError:
            tol = 1e-12
        if (
            np.max(np.abs(b_hat - b_hat[:, :1])) > tol
            or np.max(np.abs(b_sup_theta - b_sup_theta[:, :1])) > tol
            or np.max(np.abs(b_sup_zeta - b_sup_zeta[:, :1])) > tol
            or np.max(np.abs(db_dtheta - db_dtheta[:, :1])) > tol
            or np.max(np.abs(db_dzeta - db_dzeta[:, :1])) > tol
        ):
            return False
    fb = op.fblock
    if fb.collisionless is None or fb.pas is None:
        return False
    if (
        fb.exb_theta is not None
        or fb.exb_zeta is not None
        or fb.magdrift_theta is not None
        or fb.magdrift_zeta is not None
        or fb.magdrift_xidot is not None
        or fb.er_xdot is not None
        or fb.er_xidot is not None
        or fb.fp is not None
        or fb.fp_phi1 is not None
    ):
        return False
    return True


def pas_tz_preconditioner_applicable(op) -> bool:
    """Return whether the PAS 3D (theta,zeta)/L preconditioner is applicable."""
    if int(op.rhs_mode) != 1:
        return False
    if int(op.n_theta) <= 1 or int(op.n_zeta) <= 1:
        return False
    if int(op.n_theta) * int(op.n_zeta) < 64:
        return False
    if int(op.n_xi) < 2:
        return False
    fb = op.fblock
    if fb.collisionless is None or fb.pas is None:
        return False
    if fb.fp is not None or fb.fp_phi1 is not None:
        return False
    return True


def rhs1_pas_tz_max_bytes() -> int:
    """Parse the PAS-TZ memory ceiling from the environment with a safe default."""
    env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES", "").strip()
    try:
        return int(env) if env else 2 * 1024 * 1024 * 1024
    except ValueError:
        return 2 * 1024 * 1024 * 1024


def estimate_rhs1_pas_tz_build_bytes(op) -> int:
    """Estimate the dense PAS-TZ builder memory footprint in bytes.

    This is intentionally conservative. It counts the dominant dense arrays built
    by the block-tridiagonal PAS-TZ approximation, which is enough for the
    routing layer to avoid obviously unsafe builder choices.
    """
    if not pas_tz_preconditioner_applicable(op):
        return 0
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l_full = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_tz = int(n_theta * n_zeta)
    if n_tz <= 1 or n_l_full < 2:
        return 0

    pas_tz_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", "").strip()
    try:
        pas_tz_lmax = int(pas_tz_lmax_env) if pas_tz_lmax_env else 0
    except ValueError:
        pas_tz_lmax = 0
    if pas_tz_lmax <= 0:
        if n_tz <= 192:
            pas_tz_lmax = n_l_full
        elif n_tz >= 256:
            pas_tz_lmax = 6
        elif n_tz >= 128:
            pas_tz_lmax = 8
        else:
            pas_tz_lmax = 12
        if (
            op.fblock.exb_theta is None
            and op.fblock.exb_zeta is None
            and op.fblock.magdrift_theta is None
            and op.fblock.magdrift_zeta is None
            and op.fblock.magdrift_xidot is None
            and op.fblock.er_xdot is None
            and op.fblock.er_xidot is None
            and n_tz <= 256
        ):
            pas_tz_lmax = n_l_full
    n_l_use = min(n_l_full, max(2, int(pas_tz_lmax)))
    tz = int(n_tz)
    twotz = int(2 * tz)

    inv_a01 = n_species * n_x * twotz * twotz
    g01 = n_species * n_x * twotz * tz
    inv_a = n_species * n_x * max(n_l_use - 2, 0) * tz * tz
    g = n_species * n_x * max(n_l_use - 3, 0) * tz * tz
    return int((inv_a01 + g01 + inv_a + g) * 8)


def pas_tz_preconditioner_memory_safe(op) -> bool:
    """Return whether the PAS-TZ builder estimate fits within the memory ceiling."""
    estimate = estimate_rhs1_pas_tz_build_bytes(op)
    if estimate <= 0:
        return True
    return estimate <= max(0, rhs1_pas_tz_max_bytes())


def rhs1_pas_adaptive_smoother_allowed(
    *,
    op,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
) -> bool:
    """Return whether the adaptive PAS smoother should run before stronger solves."""
    env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "").strip().lower()
    enabled = env not in {"0", "false", "no", "off"}
    min_env = os.environ.get("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", "").strip()
    try:
        min_size = int(min_env) if min_env else 2000
    except ValueError:
        min_size = 2000
    return adaptive_pas_smoother_allowed(
        enabled=enabled,
        use_implicit=bool(use_implicit),
        has_pas=op.fblock.pas is not None,
        include_phi1=bool(op.include_phi1),
        residual_norm=float(residual_norm),
        target=float(target),
        active_size=int(active_size),
        min_size=int(min_size),
    )


def build_pas_tz_memory_fallback(
    *,
    op,
    matvec_shard_axis: Callable[[object], str | None],
    device_count: Callable[[], int],
    theta_schwarz_builder: Callable[..., Callable],
    zeta_schwarz_builder: Callable[..., Callable],
    hybrid_builder: Callable[..., Callable],
    reduce_full=None,
    expand_reduced=None,
) -> Callable:
    """Build the fallback preconditioner for memory-unsafe PAS-TZ requests.

    On multi-device sharded runs we use the shard-axis-specific Schwarz builder.
    Otherwise we fall back to the lighter PAS hybrid preconditioner.
    """
    shard_axis = matvec_shard_axis(op)
    if shard_axis in {"theta", "zeta"} and device_count() > 1:
        dd_block_env = os.environ.get(f"SFINCS_JAX_RHSMODE1_{shard_axis.upper()}_DD_BLOCK", "").strip()
        dd_overlap_env = os.environ.get(f"SFINCS_JAX_RHSMODE1_{shard_axis.upper()}_DD_OVERLAP", "").strip()
        try:
            dd_block = int(dd_block_env) if dd_block_env else 64
        except ValueError:
            dd_block = 64
        try:
            dd_overlap = int(dd_overlap_env) if dd_overlap_env else 1
        except ValueError:
            dd_overlap = 1
        schwarz_builder = theta_schwarz_builder if shard_axis == "theta" else zeta_schwarz_builder
        return schwarz_builder(
            op=op,
            block=dd_block,
            overlap=dd_overlap,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    return hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)


__all__ = [
    "build_pas_tz_memory_fallback",
    "estimate_rhs1_pas_tz_build_bytes",
    "pas_tokamak_theta_preconditioner_applicable",
    "pas_tz_preconditioner_applicable",
    "pas_tz_preconditioner_memory_safe",
    "rhs1_pas_adaptive_smoother_allowed",
    "rhs1_pas_tz_max_bytes",
]
