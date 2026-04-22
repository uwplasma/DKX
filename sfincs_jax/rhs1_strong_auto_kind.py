from __future__ import annotations

"""Automatic RHSMode=1 strong-preconditioner kind selection helpers."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RHS1StrongAutoSelection:
    """Resolved automatic strong-preconditioner choice."""

    kind: str | None
    xblock_tz_lmax: int | None = None


def _int_env(name: str, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def rhs1_pas_lite_min() -> int:
    return _int_env("SFINCS_JAX_PAS_LITE_MIN", 20000)


def rhs1_tz_precond_max() -> int:
    return _int_env("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", 128)


def rhs1_xblock_tz_max(*, default: int) -> int:
    return _int_env("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", default)


def rhs1_schwarz_auto_min() -> int:
    return _int_env("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", 4000)


def rhs1_pas_xmg_min() -> int:
    return _int_env("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", 50000)


def rhs1_theta_line_max() -> int:
    return _int_env("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", 0)


def rhs1_pas_strong_lmax() -> int:
    return _int_env("SFINCS_JAX_PAS_STRONG_LMAX", 2)


def auto_rhs1_reduced_strong_kind(
    *,
    has_pas: bool,
    has_fp: bool,
    geom_scheme: int,
    use_dkes: bool,
    active_size: int,
    strong_precond_min: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    shard_axis: str | None,
    device_count: int,
) -> RHS1StrongAutoSelection:
    """Choose the automatic reduced-space strong-preconditioner kind."""
    if has_pas:
        if int(active_size) >= max(1, rhs1_pas_lite_min()):
            return RHS1StrongAutoSelection(kind="pas_lite")
        return RHS1StrongAutoSelection(kind="pas_hybrid")

    if has_fp and int(active_size) >= int(strong_precond_min) and (int(n_theta) > 1 or int(n_zeta) > 1):
        tz_max = rhs1_tz_precond_max()
        xblock_default = 1200
        xblock_tz_max = rhs1_xblock_tz_max(default=xblock_default)
        lmax_auto = 0
        if int(n_theta) > 0 and int(n_zeta) > 0:
            lmax_auto = int(xblock_tz_max // (int(n_theta) * int(n_zeta)))
        lmax_auto = max(0, min(int(max_l), int(lmax_auto)))
        if (
            int(n_theta) > 1
            and xblock_tz_max > 0
            and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if lmax_auto >= 1:
            return RHS1StrongAutoSelection(kind="xblock_tz_lmax", xblock_tz_lmax=int(lmax_auto))
        if int(n_theta) > 1 and int(n_zeta) > 1 and int(n_theta) * int(n_zeta) <= int(tz_max):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and int(active_size) >= max(1, rhs1_schwarz_auto_min())
        ):
            return RHS1StrongAutoSelection(kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz")
        return RHS1StrongAutoSelection(kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line")

    if has_pas and int(active_size) >= int(strong_precond_min) and (int(n_theta) > 1 or int(n_zeta) > 1):
        tz_max = rhs1_tz_precond_max()
        xblock_default = 2000 if int(geom_scheme) == 1 or bool(use_dkes) else 1200
        xblock_tz_max = rhs1_xblock_tz_max(default=xblock_default)
        if (
            int(n_theta) > 1
            and xblock_tz_max > 0
            and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if int(n_theta) > 1 and int(n_zeta) > 1 and int(n_theta) * int(n_zeta) <= int(tz_max):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and int(active_size) >= max(1, rhs1_schwarz_auto_min())
        ):
            return RHS1StrongAutoSelection(kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz")
        return RHS1StrongAutoSelection(kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line")

    return RHS1StrongAutoSelection(kind=None)


def auto_rhs1_full_strong_kind(
    *,
    has_pas: bool,
    has_fp: bool,
    rhs1_precond_kind: str | None,
    total_size: int,
    strong_precond_min: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    shard_axis: str | None,
    device_count: int,
) -> RHS1StrongAutoSelection:
    """Choose the automatic full-space strong-preconditioner kind."""
    if has_pas:
        if int(total_size) >= max(1, rhs1_pas_lite_min()):
            return RHS1StrongAutoSelection(kind="pas_lite")
        return RHS1StrongAutoSelection(kind="pas_hybrid")

    if (
        rhs1_precond_kind == "point"
        and has_pas
        and int(total_size) >= int(strong_precond_min)
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
        if int(total_size) >= rhs1_pas_xmg_min():
            return RHS1StrongAutoSelection(kind="xmg")
        xblock_tz_max = rhs1_xblock_tz_max(default=1200)
        if (
            int(n_theta) > 1
            and xblock_tz_max > 0
            and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and int(total_size) >= max(1, rhs1_schwarz_auto_min())
        ):
            return RHS1StrongAutoSelection(kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz")
        return RHS1StrongAutoSelection(kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line")

    if has_fp and int(total_size) >= int(strong_precond_min) and (int(n_theta) > 1 or int(n_zeta) > 1):
        tz_max = rhs1_tz_precond_max()
        xblock_tz_max = rhs1_xblock_tz_max(default=1200)
        if (
            int(n_theta) > 1
            and int(n_zeta) > 1
            and xblock_tz_max > 0
            and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if int(n_theta) > 1 and int(n_zeta) > 1 and int(n_theta) * int(n_zeta) <= int(tz_max):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and int(total_size) >= max(1, rhs1_schwarz_auto_min())
        ):
            return RHS1StrongAutoSelection(kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz")
        return RHS1StrongAutoSelection(kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line")

    return RHS1StrongAutoSelection(kind=None)


def adjust_rhs1_reduced_auto_kind(
    *,
    kind: str | None,
    has_pas: bool,
    geom_scheme: int,
    n_zeta: int,
    strong_precond_trigger: bool,
    max_l: int,
    n_theta: int,
) -> RHS1StrongAutoSelection:
    """Apply post-selection adjustments to the reduced-space auto strong kind."""
    selected = kind
    selected_lmax: int | None = None
    if selected == "pas_lite" and has_pas and (int(n_zeta) == 1 or int(geom_scheme) == 1):
        selected = "pas_hybrid"
    if (
        selected in {"pas_lite", "pas_hybrid", "pas_tz"}
        and has_pas
        and (int(n_zeta) == 1 or int(geom_scheme) == 1)
        and strong_precond_trigger
    ):
        xblock_tz_max = rhs1_xblock_tz_max(default=2000)
        if (
            int(n_theta) > 0
            and int(n_zeta) > 0
            and xblock_tz_max > 0
            and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
        ):
            selected = "xblock_tz"
        else:
            lmax_fallback = min(int(max_l), max(1, rhs1_pas_strong_lmax()))
            if lmax_fallback > 0:
                selected = "xblock_tz_lmax"
                selected_lmax = int(lmax_fallback)
    return RHS1StrongAutoSelection(kind=selected, xblock_tz_lmax=selected_lmax)


def adjust_rhs1_theta_line_auto_kind(
    *,
    kind: str | None,
    n_theta: int,
    nxi_for_x_sum: int,
) -> RHS1StrongAutoSelection:
    """Apply the theta-line size guard that promotes to theta_line_xdiag."""
    if kind != "theta_line":
        return RHS1StrongAutoSelection(kind=kind)
    line_size = int(nxi_for_x_sum) * int(n_theta)
    theta_line_max = rhs1_theta_line_max()
    if theta_line_max > 0 and line_size > theta_line_max:
        return RHS1StrongAutoSelection(kind="theta_line_xdiag")
    return RHS1StrongAutoSelection(kind=kind)


__all__ = [
    "RHS1StrongAutoSelection",
    "adjust_rhs1_reduced_auto_kind",
    "adjust_rhs1_theta_line_auto_kind",
    "auto_rhs1_full_strong_kind",
    "auto_rhs1_reduced_strong_kind",
    "rhs1_pas_lite_min",
    "rhs1_pas_strong_lmax",
    "rhs1_pas_xmg_min",
    "rhs1_schwarz_auto_min",
    "rhs1_theta_line_max",
    "rhs1_tz_precond_max",
    "rhs1_xblock_tz_max",
]
