"""Policy helpers for RHSMode=1 strong-preconditioner control."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RHS1StrongPreconditionerControl:
    """Resolved strong-preconditioner control state for a solve branch."""

    min_size: int
    disabled: bool
    auto: bool
    reason_cs0_sparse_first: bool = False
    reason_large_cpu_sparse_first: bool = False
    reason_pas_auto_skip: bool = False
    reason_pas_fast_accept: bool = False
    reason_collision_probe_skip: bool = False


def rhs1_strong_preconditioner_min_size() -> int:
    """Parse the minimum size threshold for auto strong-preconditioning."""
    strong_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "").strip()
    try:
        return int(strong_precond_min_env) if strong_precond_min_env else 800
    except ValueError:
        return 800


def rhs1_resolved_strong_preconditioner_control(
    *,
    strong_precond_env: str,
    has_extra_constraint_block: bool,
    has_fp: bool,
    has_pas: bool,
    size: int,
    n_theta: int,
    n_zeta: int,
    pas_large_bicgstab_fastpath: bool = False,
    cs0_sparse_first: bool = False,
    large_cpu_sparse_rescue_first: bool = False,
    pas_auto_skip: bool = False,
    pas_fast_accept: bool = False,
    pas_precond_force_collision: bool = False,
    residual_norm: float = 0.0,
    target: float = 0.0,
) -> RHS1StrongPreconditionerControl:
    """Resolve disabled/auto strong-preconditioner control without solver side effects."""
    strong_precond_min = rhs1_strong_preconditioner_min_size()
    env = str(strong_precond_env).strip().lower()
    disabled = env in {"0", "false", "no", "off"}
    auto = env == "auto"

    reason_cs0_sparse_first = False
    reason_large_cpu_sparse_first = False
    reason_pas_auto_skip = False
    reason_pas_fast_accept = False
    reason_collision_probe_skip = False

    if pas_large_bicgstab_fastpath and env == "":
        disabled = True
        auto = False
    if cs0_sparse_first and env in {"", "auto"}:
        disabled = True
        auto = False
        reason_cs0_sparse_first = True
    if large_cpu_sparse_rescue_first and env in {"", "auto"}:
        disabled = True
        auto = False
        reason_large_cpu_sparse_first = True
    if pas_auto_skip:
        disabled = True
        auto = False
        reason_pas_auto_skip = True
    if pas_fast_accept and env in {"", "auto"}:
        disabled = True
        auto = False
        reason_pas_fast_accept = True
    if pas_precond_force_collision and env in {"", "auto"}:
        pas_force_strong_ratio_env = os.environ.get("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "").strip()
        try:
            pas_force_strong_ratio = float(pas_force_strong_ratio_env) if pas_force_strong_ratio_env else 50.0
        except ValueError:
            pas_force_strong_ratio = 50.0
        if float(residual_norm) <= float(target) * float(pas_force_strong_ratio):
            disabled = True
            auto = False
            reason_collision_probe_skip = True

    if env == "" and has_extra_constraint_block:
        auto = True
    if env == "" and has_fp and int(size) >= int(strong_precond_min) and (int(n_theta) > 1 or int(n_zeta) > 1):
        auto = True
    if env == "" and has_pas and int(size) >= int(strong_precond_min) and (int(n_theta) > 1 or int(n_zeta) > 1):
        auto = True

    if disabled:
        auto = False

    return RHS1StrongPreconditionerControl(
        min_size=int(strong_precond_min),
        disabled=bool(disabled),
        auto=bool(auto),
        reason_cs0_sparse_first=bool(reason_cs0_sparse_first),
        reason_large_cpu_sparse_first=bool(reason_large_cpu_sparse_first),
        reason_pas_auto_skip=bool(reason_pas_auto_skip),
        reason_pas_fast_accept=bool(reason_pas_fast_accept),
        reason_collision_probe_skip=bool(reason_collision_probe_skip),
    )


__all__ = [
    "RHS1StrongPreconditionerControl",
    "rhs1_resolved_strong_preconditioner_control",
    "rhs1_strong_preconditioner_min_size",
]
