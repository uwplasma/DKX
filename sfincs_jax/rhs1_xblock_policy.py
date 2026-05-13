"""Pure RHSMode=1 x-block sparse-PC routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE = 45_000
DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE = 80_000
DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO = 5_000.0
DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER = 80
DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K = 10


@dataclass(frozen=True)
class RHS1XBlockSparsePCPolicy:
    """Resolved x-block sparse preconditioned Krylov policy for one solve."""

    precondition_side: str
    default_right_preconditioned: bool
    krylov_method: str
    ignored_krylov_env: bool
    gmres_restart: int
    restart_capped: bool


def _full_fp_3d_right_pc_max_active_size(env_value: str) -> int:
    """Return the full-FP 3D active-size limit for default right preconditioning."""
    raw = str(env_value).strip()
    if not raw:
        return DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE


def _active_size_allows_full_fp_3d_right_pc(active_size: int | None, max_active_size: int) -> bool:
    """Gate right-PC defaults to the measured full-FP 3D window."""
    if active_size is None:
        return True
    try:
        return int(active_size) <= int(max_active_size)
    except (TypeError, ValueError):
        return True


def rhs1_xblock_side_probe_min_active_size(env_value: str) -> int:
    """Return the active-size floor for the 3D full-FP side probe."""
    raw = str(env_value).strip()
    if not raw:
        return DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE


def rhs1_xblock_side_probe_enabled(
    *,
    env_value: str,
    explicit_side_env_value: str,
    full_fp_3d_pc: bool,
    active_size: int | None,
    min_active_size_env_value: str,
    krylov_method: str,
    precondition_side: str,
) -> bool:
    """Return whether to run the bounded precondition-side probe.

    The probe is deliberately scoped to larger 3D full-FP QI-like systems where
    bounded evidence has shown seed-dependent left/right slow modes. Explicit
    user side overrides are always respected and disable the automatic probe.
    """
    raw = str(env_value).strip().lower()
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    explicit_side = str(explicit_side_env_value).strip().lower()
    if explicit_side in {"left", "right", "none"}:
        return False
    method = str(krylov_method).strip().lower()
    side = str(precondition_side).strip().lower()
    if method != "gmres" or side not in {"left", "right"}:
        return False
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw not in {"", "auto", "default"}:
        return False
    if not bool(full_fp_3d_pc):
        return False
    min_active_size = rhs1_xblock_side_probe_min_active_size(min_active_size_env_value)
    try:
        return int(active_size) >= int(min_active_size)
    except (TypeError, ValueError):
        return False


def rhs1_xblock_side_probe_should_switch(
    *,
    residual_ratio: float | None,
    switch_ratio_env_value: str,
) -> bool:
    """Return whether a default-side probe is weak enough to try the other side."""
    raw = str(switch_ratio_env_value).strip()
    try:
        threshold = float(raw) if raw else DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO
    except ValueError:
        threshold = DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO
    threshold = max(1.0, float(threshold))
    if residual_ratio is None:
        return False
    try:
        value = float(residual_ratio)
    except (TypeError, ValueError):
        return False
    return bool(value == value and value not in {float("inf"), float("-inf")} and value > threshold)


def rhs1_xblock_lgmres_rescue_enabled(*, env_value: str, krylov_env_value: str) -> bool:
    """Return whether a weak large-QI GMRES probe may switch to LGMRES.

    Explicit Krylov method requests are treated as user intent and are not
    rewritten by the automatic rescue. Users can still force this rescue with
    ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE=1``.
    """
    raw = str(env_value).strip().lower()
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw not in {"", "auto", "default"}:
        return False
    method_env = str(krylov_env_value).strip().lower().replace("-", "_")
    return method_env in {"", "auto", "default"}


def rhs1_xblock_lgmres_rescue_maxiter(env_value: str, current_maxiter: int) -> tuple[int, bool]:
    """Return the bounded LGMRES-rescue outer-iteration limit and cap flag."""
    try:
        requested = int(current_maxiter)
    except (TypeError, ValueError):
        requested = DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER
    requested = max(1, int(requested))
    raw = str(env_value).strip()
    if raw:
        try:
            selected = max(1, int(raw))
        except ValueError:
            selected = min(requested, DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER)
    else:
        selected = min(requested, DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER)
    return selected, bool(selected != requested)


def rhs1_xblock_lgmres_rescue_outer_k(env_value: str) -> int:
    """Return the LGMRES augmentation-space size for the large-QI rescue."""
    raw = str(env_value).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K
    return DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K


def rhs1_xblock_precondition_side(
    *,
    env_value: str,
    tokamak_fp_er_pc: bool,
    full_fp_3d_pc: bool = False,
    active_size: int | None = None,
    full_fp_3d_right_pc_max_env_value: str = "",
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> tuple[str, bool]:
    """Return the x-block sparse-PC side and whether right-PC was auto-selected.

    The measured production-floor GPU tokamak full-FP Er full-trajectory row
    and the bounded scale-0.50 3D full-FP QI lane are Krylov dominated and
    benefit from right preconditioning. Larger 3D full-FP QI cases can enter a
    seed-dependent right-PC slow mode, so the 3D default is capped by active
    system size and remains overrideable through
    ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``.
    """
    env_side = str(env_value).strip().lower()
    if env_side in {"left", "right", "none"}:
        return env_side, False
    full_trajectory = bool(include_xdot) or bool(include_electric_field_xi)
    base_path = bool((not bool(use_dkes)) and full_trajectory)
    full_fp_3d_right_pc_max = _full_fp_3d_right_pc_max_active_size(full_fp_3d_right_pc_max_env_value)
    default_right = bool(
        base_path
        and (
            bool(tokamak_fp_er_pc)
            or (
                bool(full_fp_3d_pc)
                and _active_size_allows_full_fp_3d_right_pc(active_size, full_fp_3d_right_pc_max)
            )
        )
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
    if method in {"gcrot", "gcrotmk", "gcrot_mk"}:
        return "gcrotmk", False
    if method in {"gmres", "lgmres", "bicgstab"}:
        return method, False
    return "gmres", bool(env_method)


def rhs1_xblock_gmres_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    krylov_method: str,
    default_right_preconditioned: bool,
    short_restart_default: bool | None = None,
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
    short_restart_default = bool(default_right_preconditioned) if short_restart_default is None else bool(
        short_restart_default
    )
    if not short_restart_default:
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
    full_fp_3d_pc: bool = False,
    active_size: int | None = None,
    full_fp_3d_right_pc_max_env_value: str = "",
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> RHS1XBlockSparsePCPolicy:
    """Resolve the full x-block sparse-PC policy used by ``v3_driver.py``."""
    precondition_side, default_right_preconditioned = rhs1_xblock_precondition_side(
        env_value=precondition_side_env_value,
        tokamak_fp_er_pc=tokamak_fp_er_pc,
        full_fp_3d_pc=full_fp_3d_pc,
        active_size=active_size,
        full_fp_3d_right_pc_max_env_value=full_fp_3d_right_pc_max_env_value,
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
        short_restart_default=bool(tokamak_fp_er_pc),
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
    "DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE",
    "DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER",
    "DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K",
    "DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE",
    "DEFAULT_FULL_FP_3D_SIDE_PROBE_SWITCH_RATIO",
    "RHS1XBlockSparsePCPolicy",
    "resolve_rhs1_xblock_sparse_pc_policy",
    "rhs1_xblock_gmres_restart",
    "rhs1_xblock_krylov_method",
    "rhs1_xblock_lgmres_rescue_enabled",
    "rhs1_xblock_lgmres_rescue_maxiter",
    "rhs1_xblock_lgmres_rescue_outer_k",
    "rhs1_xblock_precondition_side",
    "rhs1_xblock_side_probe_enabled",
    "rhs1_xblock_side_probe_min_active_size",
    "rhs1_xblock_side_probe_should_switch",
]
