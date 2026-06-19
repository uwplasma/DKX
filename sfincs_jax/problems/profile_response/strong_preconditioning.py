"""RHSMode=1 strong-preconditioner policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

# From sfincs_jax.rhs1_strong_policy
_PAS_WEAK_STRONG_SKIP_KINDS = frozenset({"collision", "point", "xmg"})
_PAS_STRONG_DELAY_BASE_KINDS = frozenset(
    {
        "theta_line",
        "theta_line_xdiag",
        "theta_dd",
        "theta_schwarz",
        "xblock_tz",
        "xblock_tz_lmax",
        "pas_hybrid",
        "pas_lite",
        "pas_tz",
        "pas_schur",
        "pas_tokamak_theta",
    }
)
_FP_STRONG_SIZE_GUARD_KINDS = frozenset(
    {
        "theta_line",
        "theta_line_xdiag",
        "zeta_line",
        "theta_zeta",
        "xblock_tz",
        "xblock_tz_lmax",
        "species_block",
        "sxblock",
        "sxblock_tz",
    }
)


@dataclass(frozen=True)
class RHS1StrongTriggerControls:
    """Resolved residual triggers for RHSMode=1 strong-preconditioner retries."""

    res_ratio: float
    ratio_threshold: float
    trigger: bool
    fp_force: bool
    fp_abs_threshold: float


@dataclass(frozen=True)
class RHS1StrongRetryControls:
    """Krylov bounds for a strong-preconditioner fallback solve."""

    restart: int
    maxiter: int


@dataclass(frozen=True)
class RHS1FPStrongSizeGuard:
    """Admission result for FP-only strong preconditioners on large systems."""

    skip: bool
    max_active_size: int


@dataclass(frozen=True)
class RHS1MinresCorrectionControls:
    """Controls for bounded preconditioned-MINRES residual correction."""

    steps: int
    alpha_clip: float
    min_improvement: float


def requested_rhs1_strong_preconditioner_kind(
    strong_precond_env: str, *, mode: str
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
    if env in {
        "pas_hybrid",
        "pas_xline_xcoarse",
        "pas_line_xcoarse",
        "pas_xcoarse_line",
    }:
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
    elif env in {"adi", "adi_line", "line_adi", "theta_zeta", "zeta_theta"}:
        return "adi"
    return None


def rhs1_strong_preconditioner_env_from_env() -> str:
    """Return the normalized strong-preconditioner request token."""

    return os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "").strip().lower()


def rhs1_pas_force_strong_ratio_from_env() -> float:
    """Return the PAS collision-probe ratio that allows strong fallback."""

    env = os.environ.get("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "").strip()
    try:
        return float(env) if env else 50.0
    except ValueError:
        return 50.0


def rhs1_strong_trigger_controls_from_env(
    *,
    residual_norm: float,
    target: float,
    has_fp: bool,
    include_phi1: bool,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    delay_pas_base_retries: bool,
) -> RHS1StrongTriggerControls:
    """Resolve strong-preconditioner residual trigger thresholds."""

    res_ratio = float(residual_norm) / max(float(target), 1e-300)
    ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "").strip()
    try:
        ratio_threshold = float(ratio_env) if ratio_env else 1.0
    except ValueError:
        ratio_threshold = 1.0
    if (
        not ratio_env
        and delay_pas_base_retries
        and has_pas
        and rhs1_precond_kind in _PAS_STRONG_DELAY_BASE_KINDS
    ):
        ratio_threshold = max(float(ratio_threshold), 1.0e2)
        if rhs1_precond_kind == "pas_tokamak_theta":
            # Large tokamak PAS runs usually converge with the theta
            # preconditioner; delaying the heavy fallback avoids wasted setup.
            ratio_threshold = max(float(ratio_threshold), 1.0e4)
    trigger = bool(res_ratio > ratio_threshold) if ratio_threshold > 0 else True

    fp_abs_env = os.environ.get("SFINCS_JAX_FP_STRONG_ABS", "").strip()
    try:
        fp_abs_threshold = float(fp_abs_env) if fp_abs_env else 1.0e-6
    except ValueError:
        fp_abs_threshold = 1.0e-6
    fp_force = bool(
        has_fp
        and (not bool(include_phi1))
        and float(residual_norm) > float(fp_abs_threshold)
    )
    return RHS1StrongTriggerControls(
        res_ratio=float(res_ratio),
        ratio_threshold=float(ratio_threshold),
        trigger=bool(trigger),
        fp_force=bool(fp_force),
        fp_abs_threshold=float(fp_abs_threshold),
    )


def rhs1_strong_retry_controls_from_env(*, restart: int, maxiter: int | None) -> RHS1StrongRetryControls:
    """Resolve strong-preconditioner retry Krylov bounds."""

    restart_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "").strip()
    maxiter_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "").strip()
    try:
        restart_use = int(restart_env) if restart_env else max(120, int(restart))
    except ValueError:
        restart_use = max(120, int(restart))
    try:
        maxiter_use = int(maxiter_env) if maxiter_env else max(800, int(maxiter or 400) * 2)
    except ValueError:
        maxiter_use = max(800, int(maxiter or 400) * 2)
    return RHS1StrongRetryControls(restart=int(restart_use), maxiter=int(maxiter_use))


def rhs1_collision_retry_allowed(
    *,
    residual_norm: float,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
    rhs1_precond_kind: str | None,
    has_fp: bool,
    has_pas: bool,
    strong_precond_trigger: bool,
) -> bool:
    """Return whether the point-preconditioned solve should retry with collisions."""

    return bool(
        float(residual_norm) > float(target)
        and int(rhs_mode) == 1
        and (not bool(include_phi1))
        and rhs1_precond_kind == "point"
        and (bool(has_fp) or bool(has_pas))
        and bool(strong_precond_trigger)
    )


def rhs1_fp_strong_size_guard_from_env(
    *,
    active_size: int,
    strong_precond_kind: str | None,
    has_fp: bool,
    has_pas: bool,
) -> RHS1FPStrongSizeGuard:
    """Return whether an FP-only strong preconditioner exceeds its size cap."""

    raw = os.environ.get("SFINCS_JAX_RHSMODE1_FP_STRONG_PRECOND_MAX", "").strip()
    try:
        max_active_size = int(raw) if raw else 120000
    except ValueError:
        max_active_size = 120000
    cap = max(0, int(max_active_size))
    skip = bool(
        has_fp
        and not bool(has_pas)
        and strong_precond_kind in _FP_STRONG_SIZE_GUARD_KINDS
        and int(active_size) > cap
    )
    return RHS1FPStrongSizeGuard(skip=bool(skip), max_active_size=int(max_active_size))


def rhs1_pas_weak_strong_retry_skip(
    *, has_pas: bool, rhs1_precond_kind: str | None, res_ratio: float
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
        threshold = float(env) if env else 1000000000000.0
    except ValueError:
        threshold = 1000000000000.0
    if threshold <= 0.0:
        return False
    return float(res_ratio) >= float(threshold)


def rhs1_pas_weak_minres_steps(
    *, has_pas: bool, rhs1_precond_kind: str | None, res_ratio: float
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
        ratio = float(ratio_env) if ratio_env else 1000000.0
    except ValueError:
        ratio = 1000000.0
    if ratio <= 0.0 or float(res_ratio) < float(ratio):
        return 0
    steps_env = os.environ.get("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", "").strip()
    try:
        steps = int(steps_env) if steps_env else 2
    except ValueError:
        steps = 2
    return max(0, int(steps))


def rhs1_pas_tz_guarded_minres_controls_from_env() -> RHS1MinresCorrectionControls:
    """Return guarded PAS-TZ correction controls with historical defaults."""

    steps_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS", "").strip()
    clip_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP", "").strip()
    improve_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT", ""
    ).strip()
    try:
        steps = int(steps_env) if steps_env else 2
    except ValueError:
        steps = 2
    try:
        alpha_clip = float(clip_env) if clip_env else 10.0
    except ValueError:
        alpha_clip = 10.0
    try:
        min_improvement = float(improve_env) if improve_env else 0.0
    except ValueError:
        min_improvement = 0.0
    return RHS1MinresCorrectionControls(
        steps=int(steps),
        alpha_clip=float(alpha_clip),
        min_improvement=float(min_improvement),
    )


def rhs1_pas_weak_minres_controls_from_env(*, steps: int) -> RHS1MinresCorrectionControls:
    """Return weak PAS correction controls for an admitted weak-MINRES pass."""

    clip_env = os.environ.get("SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP", "").strip()
    improve_env = os.environ.get("SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT", "").strip()
    try:
        alpha_clip = float(clip_env) if clip_env else 10.0
    except ValueError:
        alpha_clip = 10.0
    try:
        min_improvement = float(improve_env) if improve_env else 0.0
    except ValueError:
        min_improvement = 0.0
    return RHS1MinresCorrectionControls(
        steps=int(steps),
        alpha_clip=float(alpha_clip),
        min_improvement=float(min_improvement),
    )


# From sfincs_jax.rhs1_strong_control
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
    strong_precond_min_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", ""
    ).strip()
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
        pas_force_strong_ratio = rhs1_pas_force_strong_ratio_from_env()
        if float(residual_norm) <= float(target) * float(pas_force_strong_ratio):
            disabled = True
            auto = False
            reason_collision_probe_skip = True
    if env == "" and has_extra_constraint_block:
        auto = True
    if (
        env == ""
        and has_fp
        and (int(size) >= int(strong_precond_min))
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
        auto = True
    if (
        env == ""
        and has_pas
        and (int(size) >= int(strong_precond_min))
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
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


# From sfincs_jax.rhs1_strong_auto_kind
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
    """Return the active-size floor for automatic PAS-lite promotion."""
    return _int_env("SFINCS_JAX_PAS_LITE_MIN", 20000)


def rhs1_tz_precond_max() -> int:
    """Return the theta-zeta line preconditioner size cap."""
    return _int_env("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", 128)


def rhs1_xblock_tz_max(*, default: int) -> int:
    """Return the x-block theta-zeta preconditioner size cap."""
    return _int_env("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", default)


def rhs1_schwarz_auto_min() -> int:
    """Return the active-size floor for automatic Schwarz line selection."""
    return _int_env("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", 4000)


def rhs1_pas_xmg_min() -> int:
    """Return the active-size floor for automatic PAS x-grid coarse selection."""
    return _int_env("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", 50000)


def rhs1_theta_line_max() -> int:
    """Return the theta-line size cap before x-diagonal promotion."""
    return _int_env("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", 0)


def rhs1_pas_strong_lmax() -> int:
    """Return the maximum Legendre index retained by PAS strong fallbacks."""
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
    if (
        has_fp
        and int(active_size) >= int(strong_precond_min)
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
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
            and (int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max))
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if lmax_auto >= 1:
            return RHS1StrongAutoSelection(
                kind="xblock_tz_lmax", xblock_tz_lmax=int(lmax_auto)
            )
        if (
            int(n_theta) > 1
            and int(n_zeta) > 1
            and (int(n_theta) * int(n_zeta) <= int(tz_max))
        ):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and (int(active_size) >= max(1, rhs1_schwarz_auto_min()))
        ):
            return RHS1StrongAutoSelection(
                kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
            )
        return RHS1StrongAutoSelection(
            kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line"
        )
    if (
        has_pas
        and int(active_size) >= int(strong_precond_min)
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
        tz_max = rhs1_tz_precond_max()
        xblock_default = 2000 if int(geom_scheme) == 1 or bool(use_dkes) else 1200
        xblock_tz_max = rhs1_xblock_tz_max(default=xblock_default)
        if (
            int(n_theta) > 1
            and xblock_tz_max > 0
            and (int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max))
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if (
            int(n_theta) > 1
            and int(n_zeta) > 1
            and (int(n_theta) * int(n_zeta) <= int(tz_max))
        ):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and (int(active_size) >= max(1, rhs1_schwarz_auto_min()))
        ):
            return RHS1StrongAutoSelection(
                kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
            )
        return RHS1StrongAutoSelection(
            kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line"
        )
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
        and (int(total_size) >= int(strong_precond_min))
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
        if int(total_size) >= rhs1_pas_xmg_min():
            return RHS1StrongAutoSelection(kind="xmg")
        xblock_tz_max = rhs1_xblock_tz_max(default=1200)
        if (
            int(n_theta) > 1
            and xblock_tz_max > 0
            and (int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max))
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and (int(total_size) >= max(1, rhs1_schwarz_auto_min()))
        ):
            return RHS1StrongAutoSelection(
                kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
            )
        return RHS1StrongAutoSelection(
            kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line"
        )
    if (
        has_fp
        and int(total_size) >= int(strong_precond_min)
        and (int(n_theta) > 1 or int(n_zeta) > 1)
    ):
        tz_max = rhs1_tz_precond_max()
        xblock_tz_max = rhs1_xblock_tz_max(default=1200)
        if (
            int(n_theta) > 1
            and int(n_zeta) > 1
            and (xblock_tz_max > 0)
            and (int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max))
        ):
            return RHS1StrongAutoSelection(kind="xblock_tz")
        if (
            int(n_theta) > 1
            and int(n_zeta) > 1
            and (int(n_theta) * int(n_zeta) <= int(tz_max))
        ):
            return RHS1StrongAutoSelection(kind="theta_zeta")
        if (
            shard_axis in {"theta", "zeta"}
            and int(device_count) > 1
            and (int(total_size) >= max(1, rhs1_schwarz_auto_min()))
        ):
            return RHS1StrongAutoSelection(
                kind="theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
            )
        return RHS1StrongAutoSelection(
            kind="theta_line" if int(n_theta) >= int(n_zeta) else "zeta_line"
        )
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
    if (
        selected == "pas_lite"
        and has_pas
        and (int(n_zeta) == 1 or int(geom_scheme) == 1)
    ):
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
            and (xblock_tz_max > 0)
            and (int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max))
        ):
            selected = "xblock_tz"
        else:
            lmax_fallback = min(int(max_l), max(1, rhs1_pas_strong_lmax()))
            if lmax_fallback > 0:
                selected = "xblock_tz_lmax"
                selected_lmax = int(lmax_fallback)
    return RHS1StrongAutoSelection(kind=selected, xblock_tz_lmax=selected_lmax)


def adjust_rhs1_pas_schur_strong_kind_from_env(
    *,
    kind: str | None,
    has_pas: bool,
    base_kind: str | None,
    residual_norm: float,
    active_size: int,
) -> str | None:
    """Downgrade oversized reduced PAS Schur retries to the bounded PAS hybrid."""

    if not (
        kind == "schur"
        and bool(has_pas)
        and base_kind in {"pas_lite", "pas_hybrid", "pas_tz"}
        and math.isfinite(float(residual_norm))
    ):
        return kind
    raw = os.environ.get("SFINCS_JAX_PAS_SCHUR_SMALL_MAX", "").strip()
    try:
        max_active_size = int(raw) if raw else 2000
    except ValueError:
        max_active_size = 2000
    if int(active_size) > max(0, int(max_active_size)):
        return "pas_hybrid"
    return kind


def adjust_rhs1_theta_line_auto_kind(
    *, kind: str | None, n_theta: int, nxi_for_x_sum: int
) -> RHS1StrongAutoSelection:
    """Apply the theta-line size guard that promotes to theta_line_xdiag."""
    if kind != "theta_line":
        return RHS1StrongAutoSelection(kind=kind)
    line_size = int(nxi_for_x_sum) * int(n_theta)
    theta_line_max = rhs1_theta_line_max()
    if theta_line_max > 0 and line_size > theta_line_max:
        return RHS1StrongAutoSelection(kind="theta_line_xdiag")
    return RHS1StrongAutoSelection(kind=kind)


__all__ = (
    "RHS1StrongAutoSelection",
    "RHS1FPStrongSizeGuard",
    "RHS1MinresCorrectionControls",
    "RHS1StrongPreconditionerControl",
    "RHS1StrongRetryControls",
    "RHS1StrongTriggerControls",
    "adjust_rhs1_reduced_auto_kind",
    "adjust_rhs1_pas_schur_strong_kind_from_env",
    "adjust_rhs1_theta_line_auto_kind",
    "auto_rhs1_full_strong_kind",
    "auto_rhs1_reduced_strong_kind",
    "requested_rhs1_strong_preconditioner_kind",
    "rhs1_collision_retry_allowed",
    "rhs1_fp_strong_size_guard_from_env",
    "rhs1_pas_force_strong_ratio_from_env",
    "rhs1_pas_tz_guarded_minres_controls_from_env",
    "rhs1_pas_weak_minres_controls_from_env",
    "rhs1_pas_lite_min",
    "rhs1_pas_strong_lmax",
    "rhs1_pas_weak_minres_steps",
    "rhs1_pas_weak_strong_retry_skip",
    "rhs1_pas_xmg_min",
    "rhs1_resolved_strong_preconditioner_control",
    "rhs1_schwarz_auto_min",
    "rhs1_strong_preconditioner_env_from_env",
    "rhs1_strong_preconditioner_min_size",
    "rhs1_strong_retry_controls_from_env",
    "rhs1_strong_trigger_controls_from_env",
    "rhs1_theta_line_max",
    "rhs1_tz_precond_max",
    "rhs1_xblock_tz_max",
)
