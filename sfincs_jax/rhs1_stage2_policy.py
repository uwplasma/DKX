"""Policy helpers for RHSMode=1 stage-2 solve triggering."""

from __future__ import annotations

import os


_PAS_STAGE2_SKIP_BASE_KINDS = frozenset(
    {
        "pas_lite",
        "pas_hybrid",
        "pas_tz",
        "pas_schur",
        "pas_tokamak_theta",
    }
)

_PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS = frozenset(
    {
        "pas_ilu",
        "schur",
        "xblock_tz",
        "xblock_tz_lmax",
    }
)


def rhs1_stage2_ratio(*, use_dkes: bool) -> float:
    """Return the stage-2 residual-ratio trigger with DKES tightening."""
    stage2_ratio_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RATIO", "").strip()
    try:
        stage2_ratio = float(stage2_ratio_env) if stage2_ratio_env else 1.0e2
    except ValueError:
        stage2_ratio = 1.0e2
    if use_dkes:
        stage2_ratio = min(float(stage2_ratio), 1.0)
    return float(stage2_ratio)


def rhs1_stage2_trigger(*, res_ratio: float, use_dkes: bool) -> bool:
    """Return whether stage-2 should be considered from the residual ratio."""
    ratio = rhs1_stage2_ratio(use_dkes=use_dkes)
    return bool(res_ratio > ratio) if ratio > 0 else True


def rhs1_fp_force_stage2(*, has_fp: bool, include_phi1: bool, residual_norm: float) -> bool:
    """Return whether FP runs should force a stage-2 polish based on absolute residual."""
    fp_stage2_abs_env = os.environ.get("SFINCS_JAX_FP_STAGE2_ABS", "").strip()
    try:
        fp_stage2_abs = float(fp_stage2_abs_env) if fp_stage2_abs_env else 1.0e-6
    except ValueError:
        fp_stage2_abs = 1.0e-6
    return bool(
        has_fp
        and (not include_phi1)
        and float(residual_norm) > float(fp_stage2_abs)
    )


def rhs1_pas_stage2_skip(
    *,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    res_ratio: float,
) -> bool:
    """Return whether PAS runs should skip stage-2 and move to later rescue logic.

    Stage-2 GMRES is useful as a polish when the first residual is close enough
    to target. For the historical PAS-lite/hybrid/tz family, very large
    residual ratios should move directly to later rescue logic. Broader skips
    for Schur/xblock/PAS-ILU routes are opt-in only because production-floor
    tests show they can produce faster but non-parity-clean completed outputs.
    """
    if not has_pas:
        return False
    if rhs1_precond_kind not in _PAS_STAGE2_SKIP_BASE_KINDS:
        extended_env = os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED", "").strip().lower()
        if extended_env not in {"1", "true", "yes", "on"}:
            return False
        if rhs1_precond_kind not in _PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS:
            return False
    pas_stage2_skip_env = os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "").strip()
    try:
        pas_stage2_skip_ratio = float(pas_stage2_skip_env) if pas_stage2_skip_env else 1.0e6
    except ValueError:
        pas_stage2_skip_ratio = 1.0e6
    return float(res_ratio) >= float(pas_stage2_skip_ratio)


__all__ = [
    "rhs1_fp_force_stage2",
    "rhs1_pas_stage2_skip",
    "rhs1_stage2_ratio",
    "rhs1_stage2_trigger",
]
