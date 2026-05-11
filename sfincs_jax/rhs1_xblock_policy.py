"""Pure RHSMode=1 x-block sparse-PC routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RHS1XBlockSparsePCPolicy:
    """Resolved x-block sparse preconditioned Krylov policy for one solve."""

    precondition_side: str
    default_right_preconditioned: bool
    krylov_method: str
    ignored_krylov_env: bool
    gmres_restart: int
    restart_capped: bool


def rhs1_xblock_precondition_side(
    *,
    env_value: str,
    tokamak_fp_er_pc: bool,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> tuple[str, bool]:
    """Return the x-block sparse-PC side and whether right-PC was auto-selected.

    The measured production-floor GPU tokamak full-FP Er full-trajectory row is
    Krylov dominated and benefits from right preconditioning. DKES-trajectory
    Er rows do not, so the default is deliberately narrow and remains
    overrideable through ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``.
    """
    env_side = str(env_value).strip().lower()
    if env_side in {"left", "right", "none"}:
        return env_side, False
    default_right = bool(
        tokamak_fp_er_pc
        and (not bool(use_dkes))
        and (bool(include_xdot) or bool(include_electric_field_xi))
    )
    return ("right" if default_right else "left"), default_right


def rhs1_xblock_krylov_method(env_value: str) -> tuple[str, bool]:
    """Canonicalize the x-block sparse-PC Krylov method env value.

    Returns ``(method, ignored_unknown)`` so the driver can preserve its
    historical warning while keeping this normalization pure and directly
    testable.
    """
    env_method = str(env_value).strip().lower()
    method = env_method.replace("-", "_") if env_method else "gmres"
    if method in {"default", "auto"}:
        return "gmres", False
    if method in {"short_recurrence", "shortrecurrence"}:
        return "bicgstab", False
    if method == "lgmres_scipy":
        return "lgmres", False
    if method in {"gmres", "lgmres", "bicgstab"}:
        return method, False
    return "gmres", bool(env_method)


def rhs1_xblock_gmres_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    krylov_method: str,
    default_right_preconditioned: bool,
) -> tuple[int, bool]:
    """Return the x-block sparse-PC GMRES restart and whether it was auto-capped.

    The production-floor GPU full-FP Er full-trajectory row converges faster
    with a short restarted GMRES basis once the x-block preconditioner is applied
    on the right. Keep this cap restricted to the measured auto-selected policy;
    explicit user restart overrides and other trajectory branches remain
    untouched.
    """
    restart_use = max(1, int(requested_restart))
    if str(restart_env_value).strip():
        return restart_use, False
    if str(krylov_method).strip().lower() != "gmres":
        return restart_use, False
    if not bool(default_right_preconditioned):
        return restart_use, False
    capped = min(restart_use, 20)
    return capped, bool(capped != restart_use)


def resolve_rhs1_xblock_sparse_pc_policy(
    *,
    precondition_side_env_value: str,
    krylov_env_value: str,
    requested_restart: int,
    restart_env_value: str,
    tokamak_fp_er_pc: bool,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> RHS1XBlockSparsePCPolicy:
    """Resolve the full x-block sparse-PC policy used by ``v3_driver.py``."""
    precondition_side, default_right_preconditioned = rhs1_xblock_precondition_side(
        env_value=precondition_side_env_value,
        tokamak_fp_er_pc=tokamak_fp_er_pc,
        use_dkes=use_dkes,
        include_xdot=include_xdot,
        include_electric_field_xi=include_electric_field_xi,
    )
    krylov_method, ignored_krylov_env = rhs1_xblock_krylov_method(krylov_env_value)
    gmres_restart, restart_capped = rhs1_xblock_gmres_restart(
        requested_restart=requested_restart,
        restart_env_value=restart_env_value,
        krylov_method=krylov_method,
        default_right_preconditioned=default_right_preconditioned,
    )
    return RHS1XBlockSparsePCPolicy(
        precondition_side=precondition_side,
        default_right_preconditioned=default_right_preconditioned,
        krylov_method=krylov_method,
        ignored_krylov_env=ignored_krylov_env,
        gmres_restart=gmres_restart,
        restart_capped=restart_capped,
    )


__all__ = [
    "RHS1XBlockSparsePCPolicy",
    "resolve_rhs1_xblock_sparse_pc_policy",
    "rhs1_xblock_gmres_restart",
    "rhs1_xblock_krylov_method",
    "rhs1_xblock_precondition_side",
]
