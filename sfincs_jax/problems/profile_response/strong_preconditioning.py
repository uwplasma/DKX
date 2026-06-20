"""RHSMode=1 strong-preconditioner policy helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import os
from typing import Any

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


@dataclass(frozen=True)
class RHS1PostPrimaryMinresCorrectionContext:
    """State for guarded/weak MinRes correction after the primary RHSMode=1 solve."""

    result: Any
    residual_vec: Any
    residual_norm_true: float
    target: float
    matvec: Callable[[Any], Any]
    rhs: Any
    preconditioner: Callable[[Any], Any] | None
    has_pas: bool
    rhs1_precond_kind: str | None
    pas_tz_guarded_fallback: bool
    pas_tz_guarded_axis: str | None
    pas_tz_guarded_stream_requested: bool
    use_pas_projection: bool
    metadata: dict[str, object]
    requested_guarded_correction: str
    build_tzfft_preconditioner: Callable[[], Callable[[Any], Any]]
    wrap_pas_preconditioner: Callable[[Callable[[Any], Any]], Callable[[Any], Any]]
    minres_correction: Callable[..., tuple[Any, Any, tuple[float, ...], tuple[float, ...]]]
    result_factory: Callable[[Any, float], Any]
    resolve_guarded_correction_kind: Callable[..., str | None]
    guarded_controls_factory: Callable[[], RHS1MinresCorrectionControls]
    weak_steps_policy: Callable[..., int]
    weak_controls_factory: Callable[..., RHS1MinresCorrectionControls]


@dataclass(frozen=True)
class RHS1PostPrimaryMinresCorrectionOutcome:
    """Updated result and diagnostics after optional post-primary MinRes corrections."""

    result: Any
    residual_vec: Any
    residual_norm_true: float
    accepted_guarded: bool
    accepted_weak: bool


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


def run_rhs1_post_primary_minres_corrections(
    context: RHS1PostPrimaryMinresCorrectionContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1PostPrimaryMinresCorrectionOutcome:
    """Apply guarded PAS-TZ and weak-PAS MinRes corrections if they improve residuals."""

    result = context.result
    residual_vec = context.residual_vec
    residual_norm_true = float(context.residual_norm_true)
    accepted_guarded = False
    accepted_weak = False

    if (
        bool(context.pas_tz_guarded_fallback)
        and context.preconditioner is not None
        and float(result.residual_norm) > float(context.target)
    ):
        correction_preconditioner = context.preconditioner
        correction_kind = context.resolve_guarded_correction_kind(
            requested=str(context.requested_guarded_correction)
        )
        if correction_kind is not None or bool(context.pas_tz_guarded_stream_requested):
            context.metadata.update(
                {
                    "pas_tz_guarded_correction_kind": correction_kind,
                    "pas_tz_guarded_correction_stream_requested": bool(
                        context.pas_tz_guarded_stream_requested
                    ),
                    "pas_tz_guarded_correction_streamed": False,
                    "pas_tz_guarded_correction_full_update_materialized": False,
                }
            )
        if context.pas_tz_guarded_stream_requested:
            blocker = "production-pas-tz-minres-correction-requires-full-residual-direction"
            context.metadata["pas_tz_guarded_correction_stream_blocker"] = blocker
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS-TZ guarded streamed "
                    "correction requested but unavailable; using dense minres "
                    "correction because the production preconditioner requires "
                    "a full residual and full preconditioned direction",
                )
        if correction_kind == "tzfft" and str(context.pas_tz_guarded_axis) != "tzfft":
            try:
                correction_preconditioner = context.build_tzfft_preconditioner()
                if context.use_pas_projection:
                    correction_preconditioner = context.wrap_pas_preconditioner(
                        correction_preconditioner
                    )
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS-TZ guarded "
                        "matrix-free correction=tzfft",
                    )
            except Exception as exc:  # noqa: BLE001
                correction_preconditioner = context.preconditioner
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS-TZ guarded "
                        f"matrix-free correction=tzfft unavailable ({type(exc).__name__}); "
                        "using base fallback",
                    )
        guarded_minres = context.guarded_controls_factory()
        if guarded_minres.steps > 0:
            if context.metadata:
                context.metadata["pas_tz_guarded_correction_full_update_materialized"] = True
                context.metadata["pas_tz_guarded_correction_minres_steps"] = int(
                    guarded_minres.steps
                )
            x_minres, residual_minres, minres_history, minres_alphas = context.minres_correction(
                matvec=context.matvec,
                rhs=context.rhs,
                x0=result.x,
                preconditioner=correction_preconditioner,
                steps=int(guarded_minres.steps),
                alpha_clip=float(guarded_minres.alpha_clip),
                min_improvement=float(guarded_minres.min_improvement),
            )
            if minres_history and float(minres_history[-1]) < float(result.residual_norm):
                old_residual = float(result.residual_norm)
                residual_norm_true = float(minres_history[-1])
                residual_vec = residual_minres
                result = context.result_factory(x_minres, residual_norm_true)
                accepted_guarded = True
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS-TZ guarded minres correction "
                        f"accepted {len(minres_alphas)} step(s), residual="
                        f"{old_residual:.3e}->{residual_norm_true:.3e}",
                    )

    weak_minres_ratio = float(residual_norm_true) / max(float(context.target), 1e-300)
    weak_minres_steps = context.weak_steps_policy(
        has_pas=bool(context.has_pas),
        rhs1_precond_kind=context.rhs1_precond_kind,
        res_ratio=float(weak_minres_ratio),
    )
    weak_minres = context.weak_controls_factory(steps=int(weak_minres_steps))
    if (
        (not bool(context.pas_tz_guarded_fallback))
        and context.preconditioner is not None
        and weak_minres.steps > 0
        and float(result.residual_norm) > float(context.target)
    ):
        x_minres, residual_minres, minres_history, minres_alphas = context.minres_correction(
            matvec=context.matvec,
            rhs=context.rhs,
            x0=result.x,
            preconditioner=context.preconditioner,
            steps=int(weak_minres.steps),
            alpha_clip=float(weak_minres.alpha_clip),
            min_improvement=float(weak_minres.min_improvement),
        )
        if minres_history and float(minres_history[-1]) < float(result.residual_norm):
            old_residual = float(result.residual_norm)
            residual_norm_true = float(minres_history[-1])
            residual_vec = residual_minres
            result = context.result_factory(x_minres, residual_norm_true)
            accepted_weak = True
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: weak PAS minres correction "
                    f"accepted {len(minres_alphas)} step(s), residual="
                    f"{old_residual:.3e}->{residual_norm_true:.3e}",
                )
    return RHS1PostPrimaryMinresCorrectionOutcome(
        result=result,
        residual_vec=residual_vec,
        residual_norm_true=float(residual_norm_true),
        accepted_guarded=bool(accepted_guarded),
        accepted_weak=bool(accepted_weak),
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


def rhs1_strong_preconditioner_control_messages(
    control: RHS1StrongPreconditionerControl,
    *,
    residual_norm: float,
    target: float,
    rhs1_precond_kind: str | None,
    pas_auto_strong_ratio: float,
    pas_collision_probe_allows_strong: bool = False,
    pas_force_strong_ratio: float | None = None,
    sparse_rescue_label: str = "large CPU",
) -> tuple[str, ...]:
    """Return user-facing progress messages for strong-preconditioner gates."""

    messages: list[str] = []
    if control.reason_cs0_sparse_first:
        messages.append(
            "solve_v3_full_system_linear_gmres: constraintScheme=0 sparse-first "
            "auto mode -> defer strong preconditioner until after sparse ILU"
        )
    if control.reason_large_cpu_sparse_first:
        messages.append(
            f"solve_v3_full_system_linear_gmres: {sparse_rescue_label} rescue-first "
            "auto mode -> defer strong preconditioner until after sparse LU"
        )
    if control.reason_pas_auto_skip:
        messages.append(
            "solve_v3_full_system_linear_gmres: PAS auto strong preconditioner skipped "
            f"after base={rhs1_precond_kind} "
            f"(residual={float(residual_norm):.3e} <= {float(pas_auto_strong_ratio):.1f}x target)"
        )
    if control.reason_pas_fast_accept:
        messages.append(
            "solve_v3_full_system_linear_gmres: PAS fast-accept "
            f"(residual={float(residual_norm):.3e}) -> skip strong preconditioner tail"
        )
    if control.reason_collision_probe_skip:
        messages.append(
            "solve_v3_full_system_linear_gmres: PAS collision probe disabled strong preconditioner auto"
        )
    elif pas_collision_probe_allows_strong:
        ratio = (
            rhs1_pas_force_strong_ratio_from_env()
            if pas_force_strong_ratio is None
            else float(pas_force_strong_ratio)
        )
        if float(residual_norm) > float(target) * float(ratio):
            messages.append(
                "solve_v3_full_system_linear_gmres: PAS collision probe allows strong preconditioner "
                f"(residual={float(residual_norm):.3e} > {float(ratio):.1f}x target)"
            )
    return tuple(messages)


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


@dataclass(frozen=True)
class RHS1ReducedStrongPreconditionerSelection:
    """Resolved reduced-space strong-preconditioner route.

    This is intentionally policy-only: builders, caches, Krylov solves, and
    residual admission stay in the driver because they depend on live operator
    state. The returned skip flags let the driver keep existing progress
    messages without duplicating the selection logic.
    """

    kind: str | None
    candidate_kind_before_skips: str | None
    xblock_tz_lmax: int | None
    trigger: bool
    skipped_weak_pas: bool = False
    skipped_guarded_pas_tz: bool = False
    skipped_qi_device: bool = False


@dataclass(frozen=True)
class RHS1FullStrongPreconditionerSelection:
    """Resolved full-space strong-preconditioner route."""

    kind: str | None
    xblock_tz_lmax: int | None


def rhs1_reduced_strong_selection_skip_messages(
    selection: RHS1ReducedStrongPreconditionerSelection,
) -> tuple[str, ...]:
    """Return progress messages for reduced strong-preconditioner skip gates."""

    if selection.candidate_kind_before_skips is None:
        return ()
    messages: list[str] = []
    if selection.skipped_weak_pas:
        messages.append(
            "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
            "after weak PAS base residual exceeded skip threshold; set "
            "SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO=0 to retry"
        )
    if selection.skipped_guarded_pas_tz:
        messages.append(
            "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
            "after guarded PAS-TZ fallback; set "
            "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1 to retry"
        )
    if selection.skipped_qi_device:
        messages.append(
            "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
            "for QI device preconditioner experiment"
        )
    return tuple(messages)


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


def resolve_rhs1_reduced_strong_preconditioner_selection(
    *,
    strong_precond_env: str,
    control: RHS1StrongPreconditionerControl,
    has_extra_constraint_block: bool,
    has_fp: bool,
    has_pas: bool,
    geom_scheme: int,
    use_dkes: bool,
    active_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    nxi_for_x_sum: int,
    shard_axis: str | None,
    device_count: int,
    strong_precond_trigger: bool,
    rhs1_precond_kind: str | None,
    res_ratio: float,
    pas_tz_guarded_fallback: bool,
    pas_tz_guarded_strong_retry: bool,
    qi_device_skip_strong: bool,
) -> RHS1ReducedStrongPreconditionerSelection:
    """Resolve the reduced-space strong-preconditioner kind and skip gates.

    The driver has several late guards that disable strong retries for PAS,
    guarded PAS-TZ, and QI-device experiments. Centralizing the pure routing
    here keeps the solve orchestration focused on building and testing the
    chosen preconditioner.
    """

    kind: str | None = None
    xblock_tz_lmax: int | None = None
    trigger = bool(strong_precond_trigger)

    if not bool(control.disabled):
        kind = requested_rhs1_strong_preconditioner_kind(
            strong_precond_env,
            mode="reduced",
        )

    if kind is None and (not bool(control.disabled)) and bool(control.auto):
        if bool(has_extra_constraint_block):
            if has_pas:
                auto_sel = auto_rhs1_reduced_strong_kind(
                    has_pas=True,
                    has_fp=False,
                    geom_scheme=int(geom_scheme),
                    use_dkes=bool(use_dkes),
                    active_size=int(active_size),
                    strong_precond_min=int(control.min_size),
                    n_theta=int(n_theta),
                    n_zeta=int(n_zeta),
                    max_l=int(max_l),
                    shard_axis=shard_axis,
                    device_count=int(device_count),
                )
                kind = auto_sel.kind
                xblock_tz_lmax = auto_sel.xblock_tz_lmax
            else:
                kind = "schur"
        else:
            auto_sel = auto_rhs1_reduced_strong_kind(
                has_pas=bool(has_pas),
                has_fp=bool(has_fp),
                geom_scheme=int(geom_scheme),
                use_dkes=bool(use_dkes),
                active_size=int(active_size),
                strong_precond_min=int(control.min_size),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                max_l=int(max_l),
                shard_axis=shard_axis,
                device_count=int(device_count),
            )
            kind = auto_sel.kind
            xblock_tz_lmax = auto_sel.xblock_tz_lmax

    auto_sel = adjust_rhs1_reduced_auto_kind(
        kind=kind,
        has_pas=bool(has_pas),
        geom_scheme=int(geom_scheme),
        n_zeta=int(n_zeta),
        strong_precond_trigger=bool(trigger),
        max_l=int(max_l),
        n_theta=int(n_theta),
    )
    kind = auto_sel.kind
    if auto_sel.xblock_tz_lmax is not None:
        xblock_tz_lmax = auto_sel.xblock_tz_lmax

    auto_sel = adjust_rhs1_theta_line_auto_kind(
        kind=kind,
        n_theta=int(n_theta),
        nxi_for_x_sum=int(nxi_for_x_sum),
    )
    kind = auto_sel.kind

    candidate_kind_before_skips = kind

    skipped_weak_pas = rhs1_pas_weak_strong_retry_skip(
        has_pas=bool(has_pas),
        rhs1_precond_kind=rhs1_precond_kind,
        res_ratio=float(res_ratio),
    )
    if skipped_weak_pas:
        kind = None
        trigger = False

    skipped_guarded_pas_tz = bool(pas_tz_guarded_fallback) and not bool(
        pas_tz_guarded_strong_retry
    )
    if skipped_guarded_pas_tz:
        kind = None
        trigger = False

    skipped_qi_device = bool(qi_device_skip_strong)
    if skipped_qi_device:
        kind = None
        trigger = False

    return RHS1ReducedStrongPreconditionerSelection(
        kind=kind,
        candidate_kind_before_skips=candidate_kind_before_skips,
        xblock_tz_lmax=xblock_tz_lmax,
        trigger=bool(trigger),
        skipped_weak_pas=bool(skipped_weak_pas),
        skipped_guarded_pas_tz=bool(skipped_guarded_pas_tz),
        skipped_qi_device=bool(skipped_qi_device),
    )


def resolve_rhs1_full_strong_preconditioner_selection(
    *,
    strong_precond_env: str,
    control: RHS1StrongPreconditionerControl,
    has_extra_constraint_block: bool,
    has_fp: bool,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    geom_scheme: int,
    total_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    nxi_for_x_sum: int,
    shard_axis: str | None,
    device_count: int,
) -> RHS1FullStrongPreconditionerSelection:
    """Resolve the full-space strong-preconditioner kind without side effects."""

    kind: str | None = None
    xblock_tz_lmax: int | None = None

    if not bool(control.disabled):
        kind = requested_rhs1_strong_preconditioner_kind(
            strong_precond_env,
            mode="full",
        )

    if kind is None and (not bool(control.disabled)) and bool(control.auto):
        if bool(has_extra_constraint_block):
            if has_pas:
                auto_sel = auto_rhs1_full_strong_kind(
                    has_pas=True,
                    has_fp=False,
                    rhs1_precond_kind=rhs1_precond_kind,
                    total_size=int(total_size),
                    strong_precond_min=int(control.min_size),
                    n_theta=int(n_theta),
                    n_zeta=int(n_zeta),
                    max_l=int(max_l),
                    shard_axis=shard_axis,
                    device_count=int(device_count),
                )
                kind = auto_sel.kind
                xblock_tz_lmax = auto_sel.xblock_tz_lmax
            else:
                kind = "schur"
        else:
            auto_sel = auto_rhs1_full_strong_kind(
                has_pas=bool(has_pas),
                has_fp=bool(has_fp),
                rhs1_precond_kind=rhs1_precond_kind,
                total_size=int(total_size),
                strong_precond_min=int(control.min_size),
                n_theta=int(n_theta),
                n_zeta=int(n_zeta),
                max_l=int(max_l),
                shard_axis=shard_axis,
                device_count=int(device_count),
            )
            kind = auto_sel.kind
            xblock_tz_lmax = auto_sel.xblock_tz_lmax

    auto_sel = adjust_rhs1_reduced_auto_kind(
        kind=kind,
        has_pas=bool(has_pas),
        geom_scheme=int(geom_scheme),
        n_zeta=int(n_zeta),
        strong_precond_trigger=True,
        max_l=int(max_l),
        n_theta=int(n_theta),
    )
    kind = auto_sel.kind
    if auto_sel.xblock_tz_lmax is not None:
        xblock_tz_lmax = auto_sel.xblock_tz_lmax

    auto_sel = adjust_rhs1_theta_line_auto_kind(
        kind=kind,
        n_theta=int(n_theta),
        nxi_for_x_sum=int(nxi_for_x_sum),
    )
    return RHS1FullStrongPreconditionerSelection(
        kind=auto_sel.kind,
        xblock_tz_lmax=xblock_tz_lmax,
    )


__all__ = (
    "RHS1FullStrongPreconditionerSelection",
    "RHS1StrongAutoSelection",
    "RHS1FPStrongSizeGuard",
    "RHS1MinresCorrectionControls",
    "RHS1PostPrimaryMinresCorrectionContext",
    "RHS1PostPrimaryMinresCorrectionOutcome",
    "RHS1ReducedStrongPreconditionerSelection",
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
    "rhs1_reduced_strong_selection_skip_messages",
    "rhs1_pas_weak_minres_controls_from_env",
    "rhs1_pas_lite_min",
    "rhs1_pas_strong_lmax",
    "rhs1_pas_weak_minres_steps",
    "rhs1_pas_weak_strong_retry_skip",
    "rhs1_pas_xmg_min",
    "rhs1_resolved_strong_preconditioner_control",
    "resolve_rhs1_full_strong_preconditioner_selection",
    "resolve_rhs1_reduced_strong_preconditioner_selection",
    "run_rhs1_post_primary_minres_corrections",
    "rhs1_strong_preconditioner_control_messages",
    "rhs1_schwarz_auto_min",
    "rhs1_strong_preconditioner_env_from_env",
    "rhs1_strong_preconditioner_min_size",
    "rhs1_strong_retry_controls_from_env",
    "rhs1_strong_trigger_controls_from_env",
    "rhs1_theta_line_max",
    "rhs1_tz_precond_max",
    "rhs1_xblock_tz_max",
)
