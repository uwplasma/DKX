"""RHSMode=1 solver diagnostic assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class RHS1SubspaceCorrectionDiagnostics:
    """Diagnostics for one residual/coarse subspace correction hook."""

    steps_requested: int
    direction_counts: Sequence[int] = ()
    direction_names: Sequence[str] = ()
    residual_before: float | None = None
    residual_after: float | None = None
    history: Sequence[float] = ()
    fsavg_lmax: int = 0
    angular_lmax: int = -1
    angular_residual: bool = False
    seed_initialized: bool | None = None
    setup_s: float | None = None
    include_qi_basis: bool | None = None


@dataclass(frozen=True)
class RHS1PostMinresDiagnostics:
    """Diagnostics for the scalar post-minres cleanup hook."""

    steps_requested: int
    alphas: Sequence[float] = ()
    history: Sequence[float] = ()
    residual_before: float | None = None
    residual_after: float | None = None


@dataclass(frozen=True)
class RHS1PreflightDiagnostics:
    """Diagnostics for the optional x-block seed preflight gate."""

    min_improvement: float
    required: bool
    residual_norm: float | None = None
    improvement: float | None = None
    passed: bool | None = None


def _subspace_count(values: Sequence[int]) -> int:
    return int(sum(int(value) for value in values))


def build_rhs1_xblock_correction_metadata(
    *,
    probe_coarse: RHS1SubspaceCorrectionDiagnostics,
    preflight: RHS1PreflightDiagnostics,
    post_minres: RHS1PostMinresDiagnostics,
    post_coarse: RHS1SubspaceCorrectionDiagnostics,
    post_residual_equation: RHS1SubspaceCorrectionDiagnostics,
) -> dict[str, object]:
    """Build solver-trace metadata for x-block correction hooks.

    Keeping this field assembly out of ``v3_driver.py`` makes output/trace
    compatibility independently testable. The returned keys intentionally match
    the historical solver metadata names.
    """

    metadata: dict[str, object] = {
        "xblock_probe_coarse_steps_requested": int(probe_coarse.steps_requested),
        "xblock_probe_coarse_steps_accepted": int(len(probe_coarse.direction_counts)),
        "xblock_probe_coarse_direction_count": _subspace_count(probe_coarse.direction_counts),
        "xblock_probe_coarse_residual_before": probe_coarse.residual_before,
        "xblock_probe_coarse_residual_after": probe_coarse.residual_after,
        "xblock_probe_coarse_seed_initialized": bool(probe_coarse.seed_initialized),
        "xblock_probe_coarse_s": float(probe_coarse.setup_s or 0.0),
        "xblock_probe_coarse_history": tuple(probe_coarse.history),
        "xblock_probe_coarse_direction_counts": tuple(probe_coarse.direction_counts),
        "xblock_probe_coarse_direction_names": tuple(probe_coarse.direction_names),
        "xblock_probe_coarse_fsavg_lmax": int(probe_coarse.fsavg_lmax),
        "xblock_probe_coarse_angular_lmax": int(probe_coarse.angular_lmax),
        "xblock_probe_coarse_angular_residual": bool(probe_coarse.angular_residual),
        "xblock_preflight_min_improvement": float(preflight.min_improvement),
        "xblock_preflight_required": bool(preflight.required),
        "xblock_preflight_residual_norm": preflight.residual_norm,
        "xblock_preflight_improvement": preflight.improvement,
        "xblock_preflight_passed": preflight.passed,
        "xblock_post_minres_steps_requested": int(post_minres.steps_requested),
        "xblock_post_minres_steps_accepted": int(len(post_minres.alphas)),
        "xblock_post_minres_residual_before": post_minres.residual_before,
        "xblock_post_minres_residual_after": post_minres.residual_after,
        "xblock_post_minres_alphas": tuple(post_minres.alphas),
        "xblock_post_minres_history": tuple(post_minres.history),
        "xblock_post_coarse_steps_requested": int(post_coarse.steps_requested),
        "xblock_post_coarse_steps_accepted": int(len(post_coarse.direction_counts)),
        "xblock_post_coarse_direction_count": _subspace_count(post_coarse.direction_counts),
        "xblock_post_coarse_residual_before": post_coarse.residual_before,
        "xblock_post_coarse_residual_after": post_coarse.residual_after,
        "xblock_post_coarse_history": tuple(post_coarse.history),
        "xblock_post_coarse_direction_counts": tuple(post_coarse.direction_counts),
        "xblock_post_coarse_direction_names": tuple(post_coarse.direction_names),
        "xblock_post_coarse_fsavg_lmax": int(post_coarse.fsavg_lmax),
        "xblock_post_coarse_angular_lmax": int(post_coarse.angular_lmax),
        "xblock_post_coarse_angular_residual": bool(post_coarse.angular_residual),
        "xblock_post_residual_equation_steps_requested": int(
            post_residual_equation.steps_requested
        ),
        "xblock_post_residual_equation_steps_accepted": int(
            len(post_residual_equation.direction_counts)
        ),
        "xblock_post_residual_equation_direction_count": _subspace_count(
            post_residual_equation.direction_counts
        ),
        "xblock_post_residual_equation_residual_before": (
            post_residual_equation.residual_before
        ),
        "xblock_post_residual_equation_residual_after": (
            post_residual_equation.residual_after
        ),
        "xblock_post_residual_equation_history": tuple(post_residual_equation.history),
        "xblock_post_residual_equation_direction_counts": tuple(
            post_residual_equation.direction_counts
        ),
        "xblock_post_residual_equation_direction_names": tuple(
            post_residual_equation.direction_names
        ),
        "xblock_post_residual_equation_fsavg_lmax": int(
            post_residual_equation.fsavg_lmax
        ),
        "xblock_post_residual_equation_angular_lmax": int(
            post_residual_equation.angular_lmax
        ),
        "xblock_post_residual_equation_angular_residual": bool(
            post_residual_equation.angular_residual
        ),
        "xblock_post_residual_equation_include_qi_basis": bool(
            post_residual_equation.include_qi_basis
        ),
    }
    return metadata
