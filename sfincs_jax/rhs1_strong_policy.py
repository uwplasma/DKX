"""Policy helpers for RHSMode=1 strong-preconditioner selection."""

from __future__ import annotations

import os


_PAS_WEAK_STRONG_SKIP_KINDS = frozenset({"collision", "point", "xmg"})


def requested_rhs1_strong_preconditioner_kind(
    strong_precond_env: str,
    *,
    mode: str,
) -> str | None:
    """Map the env string to a strong-preconditioner kind for the requested mode."""
    env = str(strong_precond_env).strip().lower()
    if env in {"theta", "theta_line", "line_theta"}:
        return "theta_line"
    if env in {"theta_schwarz", "schwarz_theta", "ras_theta", "theta_ras"}:
        return "theta_schwarz"
    if env in {"theta_line_xdiag", "theta_xdiag", "theta_line_diagx"}:
        return "theta_line_xdiag"
    if env in {"species", "species_block", "speciesblock"}:
        return "species_block"
    if env in {"sxblock", "species_xblock", "species_x"}:
        return "sxblock"
    if env in {"sxblock_tz", "sxblock_theta_zeta", "species_xblock_tz", "sx_tz"}:
        return "sxblock_tz"
    if env in {"zeta", "zeta_line", "line_zeta"}:
        return "zeta_line"
    if env in {"zeta_schwarz", "schwarz_zeta", "ras_zeta", "zeta_ras"}:
        return "zeta_schwarz"
    if env in {"xblock_tz", "xblock", "x_tz", "xtz", "xblock_theta_zeta"}:
        return "xblock_tz"
    if env in {"xmg", "multigrid", "x_coarse", "coarse_x"}:
        return "xmg"
    if env in {"pas_lite", "pas_light", "pas_xmg", "pas_xmg_lite"}:
        return "pas_lite"
    if env in {"pas_hybrid", "pas_xline_xcoarse", "pas_line_xcoarse", "pas_xcoarse_line"}:
        return "pas_hybrid"
    if env in {"schur", "schur_complement", "constraint_schur"}:
        return "schur"
    if env == "auto":
        return None

    mode_norm = str(mode).strip().lower()
    if mode_norm == "reduced":
        if env in {"point_xdiag"}:
            return "point_xdiag"
        if env in {"xblock_tz_lmax", "xblock_tz_trunc", "xblock_tz_cut"}:
            return "xblock_tz_lmax"
        if env in {"pas_tz", "pas_3d", "pas_tz_l"}:
            return "pas_tz"
        if env in {"theta_zeta", "theta_zeta_line", "tz", "tz_line"}:
            return "theta_zeta"
        if env in {"adi", "adi_line", "line_adi", "zeta_theta"}:
            return "adi"
    else:
        if env in {"adi", "adi_line", "line_adi", "theta_zeta", "zeta_theta"}:
            return "adi"

    return None


def rhs1_pas_weak_strong_retry_skip(
    *,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    res_ratio: float,
) -> bool:
    """Return whether a weak PAS base should skip expensive strong retries.

    Collision/point/xmg preconditioners are useful bounded baselines, but when
    their first residual ratio is astronomically large, the automatic strong
    retry tends to spend minutes in setup without producing a releasable solve.
    The high default threshold keeps normal polish behavior intact while making
    known-bad forced paths fail fast and auditable.
    """
    if not bool(has_pas) or rhs1_precond_kind not in _PAS_WEAK_STRONG_SKIP_KINDS:
        return False
    env = os.environ.get("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", "").strip()
    try:
        threshold = float(env) if env else 1.0e12
    except ValueError:
        threshold = 1.0e12
    if threshold <= 0.0:
        return False
    return float(res_ratio) >= float(threshold)


def rhs1_pas_weak_minres_steps(
    *,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    res_ratio: float,
) -> int:
    """Return bounded minres correction steps for weak PAS base solves.

    This is intentionally limited to the same weak forced/probe paths guarded by
    ``rhs1_pas_weak_strong_retry_skip``. The correction is later accepted only
    if the measured residual improves, so the policy here only controls whether
    the driver should spend a few extra matrix-free matvecs before giving up on
    a weak baseline.
    """
    if not bool(has_pas) or rhs1_precond_kind not in _PAS_WEAK_STRONG_SKIP_KINDS:
        return 0
    ratio_env = os.environ.get("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", "").strip()
    try:
        ratio = float(ratio_env) if ratio_env else 1.0e6
    except ValueError:
        ratio = 1.0e6
    if ratio <= 0.0 or float(res_ratio) < float(ratio):
        return 0
    steps_env = os.environ.get("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", "").strip()
    try:
        steps = int(steps_env) if steps_env else 2
    except ValueError:
        steps = 2
    return max(0, int(steps))


__all__ = [
    "requested_rhs1_strong_preconditioner_kind",
    "rhs1_pas_weak_minres_steps",
    "rhs1_pas_weak_strong_retry_skip",
]
