"""RHSMode=1 PAS applicability and memory policy helpers.

This module holds the small, pure policy functions that decide whether the
specialized PAS tokamak-theta and PAS-TZ preconditioners are eligible to run.
They are intentionally isolated from the large solve orchestration in
``v3_driver.py`` so they can be tested directly and reused from multiple
dispatch paths without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Callable
import math
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
    collision_builder: Callable[..., Callable] | None = None,
    reduce_full=None,
    expand_reduced=None,
) -> Callable:
    """Build the fallback preconditioner for memory-unsafe PAS-TZ requests.

    On multi-device sharded runs we use the shard-axis-specific Schwarz builder.
    Otherwise we fall back to the lighter PAS hybrid preconditioner unless the
    user explicitly requests a structured Schwarz fallback with
    ``SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK``. This keeps release defaults
    unchanged while giving the next geometry-rich PAS optimization lane a
    measured, opt-in benchmark hook. Structured Schwarz builders still allocate
    dense patch inverses, so they are guarded by an explicit patch-work estimate;
    unsafe requests fall back to the cheap collision preconditioner when it is
    available, or to the historical hybrid fallback otherwise.
    """
    shard_axis = matvec_shard_axis(op)
    axis = resolve_pas_tz_memory_fallback_axis(
        op=op,
        requested=os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", ""),
        shard_axis=shard_axis,
        n_devices=device_count(),
    )
    if axis in {"theta", "zeta"}:
        dd_block = _parse_pas_tz_fallback_int(
            f"SFINCS_JAX_RHSMODE1_{axis.upper()}_DD_BLOCK",
            fallback_name="SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK",
            default=64,
        )
        dd_overlap = _parse_pas_tz_fallback_int(
            f"SFINCS_JAX_RHSMODE1_{axis.upper()}_DD_OVERLAP",
            fallback_name="SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP",
            default=1,
        )
        if not pas_tz_schwarz_fallback_memory_safe(
            op,
            axis=axis,
            block=dd_block,
            overlap=dd_overlap,
        ):
            if collision_builder is not None:
                precond = collision_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
            else:
                precond = hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
            _mark_pas_tz_guarded_fallback(precond, axis=axis)
            return precond
        schwarz_builder = theta_schwarz_builder if axis == "theta" else zeta_schwarz_builder
        return schwarz_builder(
            op=op,
            block=dd_block,
            overlap=dd_overlap,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    return hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)


def _mark_pas_tz_guarded_fallback(precond: Callable, *, axis: str) -> None:
    """Attach best-effort metadata to a guarded PAS-TZ fallback callable."""
    try:
        setattr(precond, "_sfincs_jax_pas_tz_guarded_fallback", True)
        setattr(precond, "_sfincs_jax_pas_tz_guarded_axis", str(axis))
    except Exception:
        pass


def estimate_pas_tz_schwarz_fallback_work(
    op,
    *,
    axis: str,
    block: int,
    overlap: int,
) -> dict[str, int]:
    """Estimate dense-patch work for an opt-in PAS-TZ Schwarz fallback.

    The theta/zeta Schwarz builders currently precompute dense inverses for
    every species and orthogonal angular line. The dominant memory term is the
    number of stored inverse entries, while the dominant setup-time term scales
    cubically with the largest patch unknown count. This estimator is deliberately
    simple and conservative so routing tests can reject known-bad production
    shapes before they enter a long JAX/XLA build.
    """
    axis_l = str(axis).strip().lower()
    if axis_l not in {"theta", "zeta"}:
        axis_l = preferred_pas_tz_schwarz_axis(op)
    n_species = max(1, int(getattr(op, "n_species", 1)))
    n_theta = max(1, int(getattr(op, "n_theta", 1)))
    n_zeta = max(1, int(getattr(op, "n_zeta", 1)))
    block_i = max(1, int(block))
    overlap_i = max(0, int(overlap))
    n_axis = n_theta if axis_l == "theta" else n_zeta
    n_lines = n_zeta if axis_l == "theta" else n_theta
    n_patches_per_line = max(1, int(math.ceil(float(n_axis) / float(block_i))))
    max_patch_extent = min(n_axis, block_i + 2 * overlap_i)
    local_velocity_dof = _pas_tz_local_velocity_dof(op)
    max_patch_unknowns = int(max_patch_extent * local_velocity_dof)
    patch_count = int(n_species * n_lines * n_patches_per_line)
    inverse_entries = int(patch_count * max_patch_unknowns * max_patch_unknowns)
    return {
        "axis": 0 if axis_l == "theta" else 1,
        "block": int(block_i),
        "overlap": int(overlap_i),
        "patch_count": int(patch_count),
        "max_patch_extent": int(max_patch_extent),
        "local_velocity_dof": int(local_velocity_dof),
        "max_patch_unknowns": int(max_patch_unknowns),
        "inverse_entries": int(inverse_entries),
        "inverse_bytes_float64": int(inverse_entries * 8),
    }


def pas_tz_schwarz_fallback_memory_safe(
    op,
    *,
    axis: str,
    block: int,
    overlap: int,
) -> bool:
    """Return whether a structured PAS-TZ Schwarz fallback should be attempted."""
    work = estimate_pas_tz_schwarz_fallback_work(op, axis=axis, block=block, overlap=overlap)
    max_patch_unknowns = _parse_nonnegative_env_int(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS",
        default=8192,
    )
    max_inverse_entries = _parse_nonnegative_env_int(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES",
        default=100_000_000,
    )
    if max_patch_unknowns > 0 and int(work["max_patch_unknowns"]) > max_patch_unknowns:
        return False
    if max_inverse_entries > 0 and int(work["inverse_entries"]) > max_inverse_entries:
        return False
    return True


def _pas_tz_local_velocity_dof(op) -> int:
    """Return the per-angular-line velocity unknown count used by Schwarz patches."""
    try:
        collisionless = getattr(getattr(op, "fblock", None), "collisionless", None)
        values = getattr(collisionless, "n_xi_for_x", None)
        if values is not None:
            arr = np.asarray(values, dtype=np.int64).reshape(-1)
            if arr.size:
                return max(1, int(np.sum(arr)))
    except Exception:
        pass
    n_x = max(1, int(getattr(op, "n_x", 1)))
    n_xi = max(1, int(getattr(op, "n_xi", 1)))
    return int(n_x * n_xi)


def _parse_nonnegative_env_int(name: str, *, default: int) -> int:
    """Parse a non-negative integer env var; non-positive values disable a cap."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(0, int(raw))
    except ValueError:
        return int(default)


def _parse_pas_tz_fallback_int(name: str, *, fallback_name: str, default: int) -> int:
    """Parse a structured PAS fallback integer env var with a shared fallback."""
    for env_name in (name, fallback_name):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return int(default)


def preferred_pas_tz_schwarz_axis(op) -> str:
    """Choose the structured Schwarz axis with the richer angular direction."""
    try:
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
    except Exception:
        return "theta"
    return "zeta" if n_zeta >= n_theta else "theta"


def resolve_pas_tz_memory_fallback_axis(
    *,
    op,
    requested: str,
    shard_axis: str | None,
    n_devices: int,
) -> str | None:
    """Resolve the memory-unsafe PAS-TZ fallback axis.

    Empty/default requests preserve the historical behavior: only already-sharded
    multi-device runs select a Schwarz fallback automatically. Explicit
    ``theta``, ``zeta``, or ``schwarz`` requests enable the structured route for
    bounded single-device experiments without widening production defaults.
    """
    req = str(requested or "").strip().lower().replace("-", "_")
    if req in {"hybrid", "pas_hybrid", "off", "0", "false", "no"}:
        return None
    if req in {"theta", "theta_schwarz"}:
        return "theta"
    if req in {"zeta", "zeta_schwarz"}:
        return "zeta"
    if req in {"schwarz", "structured", "structured_schwarz", "auto_schwarz"}:
        return preferred_pas_tz_schwarz_axis(op)
    if req:
        return None
    if shard_axis in {"theta", "zeta"} and int(n_devices) > 1:
        return str(shard_axis)
    return None


__all__ = [
    "build_pas_tz_memory_fallback",
    "estimate_rhs1_pas_tz_build_bytes",
    "estimate_pas_tz_schwarz_fallback_work",
    "pas_tokamak_theta_preconditioner_applicable",
    "pas_tz_preconditioner_applicable",
    "pas_tz_schwarz_fallback_memory_safe",
    "pas_tz_preconditioner_memory_safe",
    "preferred_pas_tz_schwarz_axis",
    "resolve_pas_tz_memory_fallback_axis",
    "rhs1_pas_adaptive_smoother_allowed",
    "rhs1_pas_tz_max_bytes",
]
