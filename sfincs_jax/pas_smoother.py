from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
import math

import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

__all__ = [
    "PAS_AUTO_STRONG_BASE_KINDS",
    "AdaptiveStationaryResult",
    "AdaptivePassSmootherResult",
    "PasSmootherConfig",
    "PasResidualTrend",
    "PasSmootherDecision",
    "pas_auto_skip_strong_retry",
    "pas_fast_accept",
    "should_stop_adaptive_smoother",
    "run_adaptive_stationary_smoother",
    "adaptive_pas_smoother_allowed",
    "adaptive_pas_smoother",
    "append_residual",
    "summarize_residual_history",
    "decide_pas_smoother_action",
    "advance_pas_smoother",
]


PAS_AUTO_STRONG_BASE_KINDS = frozenset(
    {
        "schur",
        "xblock_tz",
        "xblock_tz_lmax",
        "sxblock_tz",
        "species_block",
        "theta_zeta",
        "pas_lite",
        "pas_hybrid",
        "pas_schur",
        "pas_tz",
        "pas_tokamak_theta",
    }
)


@dataclass(frozen=True)
class PasSmootherConfig:
    window: int = 3
    accept_ratio: float = 0.999999
    worsen_ratio: float = 1.01
    stagnation_ratio: float = 0.999

    def __post_init__(self) -> None:
        if int(self.window) <= 0:
            raise ValueError("PasSmootherConfig.window must be positive")
        if float(self.accept_ratio) <= 0.0 or float(self.accept_ratio) >= 1.0:
            raise ValueError("PasSmootherConfig.accept_ratio must lie in (0, 1)")
        if float(self.worsen_ratio) <= 1.0:
            raise ValueError("PasSmootherConfig.worsen_ratio must exceed 1")
        if float(self.accept_ratio) >= float(self.worsen_ratio):
            raise ValueError("PasSmootherConfig.accept_ratio must be smaller than worsen_ratio")
        if float(self.stagnation_ratio) <= 0.0:
            raise ValueError("PasSmootherConfig.stagnation_ratio must be positive")


@dataclass(frozen=True)
class PasSmootherTrend:
    history: tuple[float, ...]
    latest: float
    previous: float
    best_so_far: float
    best_before_latest: float
    latest_ratio: float
    best_before_latest_ratio: float
    window_reference: float
    window_ratio: float
    window_log_slope: float
    consecutive_increases: int
    has_nonfinite: bool


@dataclass(frozen=True)
class PasSmootherDecision:
    accept: bool
    stop: bool
    reason: str
    trend: PasSmootherTrend


def append_residual(history: Sequence[float], residual: float) -> tuple[float, ...]:
    return tuple(float(v) for v in history) + (float(residual),)


def summarize_residual_history(history: Sequence[float], *, window: int = 3) -> PasSmootherTrend:
    history_t = tuple(float(v) for v in history)
    if not history_t:
        raise ValueError("history must be non-empty")
    arr = np.asarray(history_t, dtype=np.float64)
    has_nonfinite = not np.all(np.isfinite(arr))
    latest = float(arr[-1])
    previous = float(arr[-2]) if arr.size >= 2 else latest
    best_so_far = float(np.min(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else float("inf")
    if arr.size >= 2 and np.any(np.isfinite(arr[:-1])):
        best_before_latest = float(np.min(arr[:-1][np.isfinite(arr[:-1])]))
    else:
        best_before_latest = latest
    latest_ratio = latest / max(abs(previous), 1.0e-300)
    best_before_latest_ratio = latest / max(abs(best_before_latest), 1.0e-300)
    ref_index = max(0, arr.size - int(window) - 1)
    window_reference = float(arr[ref_index])
    window_ratio = latest / max(abs(window_reference), 1.0e-300)
    window_steps = max(1, arr.size - 1 - ref_index)
    window_log_slope = np.log(max(window_ratio, 1.0e-300)) / float(window_steps)
    consecutive_increases = 0
    for idx in range(arr.size - 1, 0, -1):
        if float(arr[idx]) > float(arr[idx - 1]):
            consecutive_increases += 1
        else:
            break
    return PasSmootherTrend(
        history=history_t,
        latest=latest,
        previous=previous,
        best_so_far=best_so_far,
        best_before_latest=best_before_latest,
        latest_ratio=latest_ratio,
        best_before_latest_ratio=best_before_latest_ratio,
        window_reference=window_reference,
        window_ratio=window_ratio,
        window_log_slope=float(window_log_slope),
        consecutive_increases=consecutive_increases,
        has_nonfinite=has_nonfinite,
    )


def decide_pas_smoother_action(
    history: Sequence[float],
    *,
    config: PasSmootherConfig | None = None,
) -> PasSmootherDecision:
    cfg = config if config is not None else PasSmootherConfig()
    trend = summarize_residual_history(history, window=cfg.window)
    if trend.has_nonfinite:
        return PasSmootherDecision(accept=False, stop=True, reason="nonfinite", trend=trend)
    if len(trend.history) < 2:
        return PasSmootherDecision(accept=False, stop=False, reason="insufficient-history", trend=trend)
    accept = bool(trend.latest_ratio < 1.0)
    if trend.latest_ratio >= float(cfg.worsen_ratio):
        return PasSmootherDecision(accept=False, stop=True, reason="single-step-worsened", trend=trend)
    if len(trend.history) >= cfg.window + 1 and trend.window_ratio >= float(cfg.stagnation_ratio):
        return PasSmootherDecision(accept=accept, stop=True, reason="window-stagnation", trend=trend)
    if accept:
        return PasSmootherDecision(accept=True, stop=False, reason="improved", trend=trend)
    return PasSmootherDecision(accept=False, stop=False, reason="continue", trend=trend)


def advance_pas_smoother(
    history: Sequence[float],
    residual: float,
    *,
    config: PasSmootherConfig | None = None,
) -> PasSmootherDecision:
    return decide_pas_smoother_action(append_residual(history, residual), config=config)


def pas_auto_skip_strong_retry(
    *,
    has_pas: bool,
    strong_precond_env: str,
    rhs1_precond_kind: str | None,
    residual_norm: float,
    target: float,
    ratio: float,
) -> bool:
    if not has_pas or ratio <= 0.0:
        return False
    if strong_precond_env not in {"", "auto"}:
        return False
    if rhs1_precond_kind not in PAS_AUTO_STRONG_BASE_KINDS:
        return False
    return float(residual_norm) <= float(target) * float(ratio)


def pas_fast_accept(
    *,
    active_size: int,
    residual_norm: float,
    target: float,
    min_size: int,
    ratio: float,
    abs_floor: float,
) -> bool:
    if int(active_size) < max(1, int(min_size)):
        return False
    if not np.isfinite(float(residual_norm)):
        return False
    accept_thresh = max(float(target) * max(1.0, float(ratio)), max(0.0, float(abs_floor)))
    return float(residual_norm) <= float(accept_thresh)


def should_stop_adaptive_smoother(
    residual_history: Sequence[float],
    *,
    target: float,
    target_ratio: float,
    abs_floor: float,
    upward_ratio: float,
    patience: int,
    min_steps: int,
) -> tuple[bool, str]:
    history = [float(v) for v in residual_history if np.isfinite(float(v))]
    if not history:
        return True, "empty"
    if len(history) != len(residual_history):
        return True, "nonfinite"
    threshold = max(float(target) * max(1.0, float(target_ratio)), max(0.0, float(abs_floor)))
    if history[-1] <= threshold:
        return True, "target"
    patience_use = max(1, int(patience))
    min_steps_use = max(1, int(min_steps))
    if len(history) < max(2, min_steps_use + 1):
        return False, "continue"
    best = min(history[:-1])
    if history[-1] <= best:
        return False, "continue"
    tail = history[-patience_use:]
    if all(val >= best * max(1.0, float(upward_ratio)) for val in tail):
        return True, "upward"
    return False, "continue"


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveStationaryResult:
    x_best: jnp.ndarray
    best_residual_norm: float
    residual_history: tuple[float, ...]
    steps_completed: int
    stop_reason: str
    improved: bool

    def tree_flatten(self):
        children = (
            self.x_best,
            jnp.asarray(self.best_residual_norm, dtype=jnp.float64),
            jnp.asarray(self.residual_history, dtype=jnp.float64),
        )
        aux = (self.steps_completed, self.stop_reason, self.improved)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        steps_completed, stop_reason, improved = aux
        x_best, best_residual_norm, residual_history = children
        return cls(
            x_best=x_best,
            best_residual_norm=float(best_residual_norm),
            residual_history=tuple(float(v) for v in np.asarray(residual_history, dtype=np.float64)),
            steps_completed=steps_completed,
            stop_reason=stop_reason,
            improved=improved,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class AdaptivePassSmootherResult:
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    history: jnp.ndarray
    accepted_sweeps: int
    stop_reason: str

    def tree_flatten(self):
        children = (self.x, self.residual_norm, self.history)
        aux = (self.accepted_sweeps, self.stop_reason)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        accepted_sweeps, stop_reason = aux
        x, residual_norm, history = children
        return cls(
            x=x,
            residual_norm=residual_norm,
            history=history,
            accepted_sweeps=accepted_sweeps,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True)
class PasSmootherConfig:
    window: int = 3
    accept_ratio: float = 1.0
    worsen_ratio: float = 1.05
    stagnation_ratio: float = 0.995
    max_consecutive_increases: int = 1

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise ValueError("window must be >= 1")
        if not math.isfinite(float(self.accept_ratio)) or float(self.accept_ratio) <= 0.0:
            raise ValueError("accept_ratio must be finite and > 0")
        if not math.isfinite(float(self.worsen_ratio)) or float(self.worsen_ratio) <= 0.0:
            raise ValueError("worsen_ratio must be finite and > 0")
        if float(self.accept_ratio) > float(self.worsen_ratio):
            raise ValueError("accept_ratio must not exceed worsen_ratio")
        if not math.isfinite(float(self.stagnation_ratio)) or float(self.stagnation_ratio) <= 0.0:
            raise ValueError("stagnation_ratio must be finite and > 0")
        if int(self.max_consecutive_increases) < 1:
            raise ValueError("max_consecutive_increases must be >= 1")


@dataclass(frozen=True)
class PasResidualTrend:
    history: tuple[float, ...]
    latest: float
    previous: float | None
    best_so_far: float
    best_before_latest: float | None
    worst_so_far: float
    latest_ratio: float | None
    best_before_latest_ratio: float | None
    window_reference: float | None
    window_ratio: float | None
    window_log_slope: float | None
    consecutive_increases: int
    has_nonfinite: bool


@dataclass(frozen=True)
class PasSmootherDecision:
    accept: bool
    stop: bool
    reason: str
    trend: PasResidualTrend


def append_residual(history: Sequence[float], residual: float) -> tuple[float, ...]:
    return tuple(float(value) for value in history) + (float(residual),)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.inf
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        return math.inf
    return float(numerator) / float(denominator)


def _count_consecutive_increases(history: tuple[float, ...]) -> int:
    if len(history) < 2:
        return 0
    count = 0
    for idx in range(len(history) - 1, 0, -1):
        prev = history[idx - 1]
        curr = history[idx]
        if not (math.isfinite(prev) and math.isfinite(curr)):
            break
        if curr > prev:
            count += 1
            continue
        break
    return count


def summarize_residual_history(
    history: Sequence[float],
    *,
    window: int = 3,
) -> PasResidualTrend:
    values = tuple(float(value) for value in history)
    if not values:
        raise ValueError("history must contain at least one residual")
    if int(window) < 1:
        raise ValueError("window must be >= 1")

    latest = values[-1]
    previous = values[-2] if len(values) >= 2 else None
    has_nonfinite = any(not math.isfinite(value) for value in values)
    finite_values = tuple(value for value in values if math.isfinite(value))
    if finite_values:
        best_so_far = min(finite_values)
        worst_so_far = max(finite_values)
        best_before_latest = (
            min(value for value in values[:-1] if math.isfinite(value))
            if len(values) >= 2 and any(math.isfinite(value) for value in values[:-1])
            else None
        )
    else:
        best_so_far = math.nan
        worst_so_far = math.nan
        best_before_latest = None
    latest_ratio = _safe_ratio(latest, previous) if previous is not None else None
    best_before_latest_ratio = (
        _safe_ratio(latest, best_before_latest) if best_before_latest is not None else None
    )

    pair_count = min(int(window), max(0, len(values) - 1))
    window_reference = None
    window_ratio = None
    window_log_slope = None
    if pair_count > 0:
        window_start = len(values) - pair_count - 1
        if window_start >= 0:
            window_reference = values[window_start]
            window_ratio = _safe_ratio(latest, window_reference)
        pairwise_logs: list[float] = []
        for idx in range(len(values) - pair_count, len(values)):
            prev = values[idx - 1]
            curr = values[idx]
            ratio = _safe_ratio(curr, prev)
            if not math.isfinite(ratio) or ratio <= 0.0:
                pairwise_logs = []
                break
            pairwise_logs.append(math.log(ratio))
        if pairwise_logs:
            window_log_slope = float(sum(pairwise_logs)) / float(len(pairwise_logs))

    return PasResidualTrend(
        history=values,
        latest=latest,
        previous=previous,
        best_so_far=best_so_far,
        best_before_latest=best_before_latest,
        worst_so_far=worst_so_far,
        latest_ratio=latest_ratio,
        best_before_latest_ratio=best_before_latest_ratio,
        window_reference=window_reference,
        window_ratio=window_ratio,
        window_log_slope=window_log_slope,
        consecutive_increases=_count_consecutive_increases(values),
        has_nonfinite=has_nonfinite,
    )


def decide_pas_smoother_action(
    history: Sequence[float],
    *,
    config: PasSmootherConfig = PasSmootherConfig(),
) -> PasSmootherDecision:
    trend = summarize_residual_history(history, window=config.window)
    if trend.has_nonfinite:
        return PasSmootherDecision(accept=False, stop=True, reason="nonfinite-residual", trend=trend)
    if len(trend.history) == 1:
        return PasSmootherDecision(accept=True, stop=False, reason="seed-history", trend=trend)
    if trend.latest == 0.0:
        return PasSmootherDecision(accept=True, stop=True, reason="zero-residual", trend=trend)
    if trend.latest_ratio is not None and trend.latest_ratio > config.worsen_ratio:
        return PasSmootherDecision(accept=False, stop=True, reason="single-step-worsened", trend=trend)
    if trend.window_log_slope is not None and trend.window_log_slope > math.log(config.worsen_ratio):
        return PasSmootherDecision(accept=False, stop=True, reason="window-trend-worsened", trend=trend)
    if trend.consecutive_increases >= config.max_consecutive_increases:
        return PasSmootherDecision(
            accept=False,
            stop=True,
            reason="consecutive-increases",
            trend=trend,
        )
    if trend.window_ratio is not None and trend.window_ratio >= config.stagnation_ratio:
        return PasSmootherDecision(accept=True, stop=True, reason="window-stagnation", trend=trend)
    if trend.best_before_latest_ratio is not None and trend.best_before_latest_ratio <= config.accept_ratio:
        return PasSmootherDecision(accept=True, stop=False, reason="improved", trend=trend)
    if trend.latest_ratio is not None and trend.latest_ratio <= config.accept_ratio:
        return PasSmootherDecision(accept=True, stop=False, reason="improved", trend=trend)
    return PasSmootherDecision(accept=False, stop=True, reason="not-improving", trend=trend)


def advance_pas_smoother(
    history: Sequence[float],
    residual: float,
    *,
    config: PasSmootherConfig = PasSmootherConfig(),
) -> PasSmootherDecision:
    return decide_pas_smoother_action(append_residual(history, residual), config=config)


def adaptive_pas_smoother_allowed(
    *,
    enabled: bool,
    use_implicit: bool,
    has_pas: bool,
    include_phi1: bool,
    residual_norm: float,
    target: float,
    active_size: int,
    min_size: int,
) -> bool:
    if not bool(enabled):
        return False
    if bool(use_implicit) or (not bool(has_pas)) or bool(include_phi1):
        return False
    if int(active_size) < max(1, int(min_size)):
        return False
    if not np.isfinite(float(residual_norm)):
        return False
    return float(residual_norm) > float(target)


def adaptive_pas_smoother(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    target: float,
    omega: float = 1.0,
    max_sweeps: int = 3,
    min_rel_improvement: float = 2.5e-2,
    worsen_factor: float = 1.05,
    plateau_patience: int = 1,
) -> AdaptivePassSmootherResult:
    x = jnp.asarray(x0, dtype=jnp.float64)
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    residual = rhs - jnp.asarray(matvec(x), dtype=jnp.float64)
    residual_norm = float(jnp.linalg.norm(residual))
    best_x = x
    best_norm = residual_norm
    history: list[float] = [residual_norm]
    accepted = 0
    plateau_count = 0
    stop_reason = "max_sweeps"
    tiny = 1.0e-300

    for _ in range(max(0, int(max_sweeps))):
        correction = jnp.asarray(preconditioner(residual), dtype=jnp.float64)
        trial_x = x + float(omega) * correction
        trial_residual = rhs - jnp.asarray(matvec(trial_x), dtype=jnp.float64)
        trial_norm = float(jnp.linalg.norm(trial_residual))
        history.append(trial_norm)

        if np.isfinite(trial_norm) and trial_norm < best_norm:
            best_x = trial_x
            best_norm = trial_norm

        if not np.isfinite(trial_norm):
            stop_reason = "nonfinite"
            break
        if trial_norm <= float(target):
            accepted += 1
            best_x = trial_x
            best_norm = trial_norm
            stop_reason = "target"
            break
        if trial_norm > residual_norm * float(worsen_factor):
            stop_reason = "worsened"
            break

        rel_improvement = (residual_norm - trial_norm) / max(abs(residual_norm), tiny)
        if rel_improvement < float(min_rel_improvement):
            plateau_count += 1
        else:
            plateau_count = 0

        if trial_norm < residual_norm:
            accepted += 1
            x = trial_x
            residual = trial_residual
            residual_norm = trial_norm

        if plateau_count > int(plateau_patience):
            stop_reason = "plateau"
            break

    return AdaptivePassSmootherResult(
        x=jnp.asarray(best_x, dtype=jnp.float64),
        residual_norm=jnp.asarray(best_norm, dtype=jnp.float64),
        history=jnp.asarray(history, dtype=jnp.float64),
        accepted_sweeps=int(accepted),
        stop_reason=stop_reason,
    )


def run_adaptive_stationary_smoother(
    *,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_vec: jnp.ndarray,
    x0_vec: jnp.ndarray,
    smoother_fn: Callable[[jnp.ndarray], jnp.ndarray],
    target: float,
    max_steps: int,
    omega: float,
    upward_ratio: float,
    patience: int,
    min_steps: int,
    target_ratio: float,
    abs_floor: float,
) -> AdaptiveStationaryResult:
    x_curr = jnp.asarray(x0_vec, dtype=jnp.float64)
    x_best = x_curr
    history: list[float] = []
    best_residual = float("inf")
    stop_reason = "max_steps"
    max_steps_use = max(1, int(max_steps))
    omega_use = float(omega)

    for step in range(max_steps_use + 1):
        residual_vec = rhs_vec - matvec_fn(x_curr)
        residual_norm = float(jnp.linalg.norm(residual_vec))
        history.append(residual_norm)
        if np.isfinite(residual_norm) and residual_norm < best_residual:
            best_residual = residual_norm
            x_best = x_curr
        stop, reason = should_stop_adaptive_smoother(
            history,
            target=float(target),
            target_ratio=float(target_ratio),
            abs_floor=float(abs_floor),
            upward_ratio=float(upward_ratio),
            patience=int(patience),
            min_steps=int(min_steps),
        )
        if stop:
            stop_reason = reason
            break
        if step >= max_steps_use:
            break
        delta = smoother_fn(jnp.asarray(residual_vec, dtype=jnp.float64))
        if not bool(jnp.all(jnp.isfinite(delta))):
            stop_reason = "nonfinite_update"
            break
        x_curr = x_curr + omega_use * jnp.asarray(delta, dtype=jnp.float64)

    initial = history[0] if history else float("inf")
    return AdaptiveStationaryResult(
        x_best=jnp.asarray(x_best, dtype=jnp.float64),
        best_residual_norm=float(best_residual),
        residual_history=tuple(float(v) for v in history),
        steps_completed=max(0, len(history) - 1),
        stop_reason=stop_reason,
        improved=np.isfinite(float(best_residual)) and float(best_residual) < float(initial),
    )


def adaptive_pas_smoother_allowed(
    *,
    enabled: bool,
    use_implicit: bool,
    has_pas: bool,
    include_phi1: bool,
    residual_norm: float,
    target: float,
    active_size: int,
    min_size: int,
) -> bool:
    if not bool(enabled):
        return False
    if bool(use_implicit) or (not bool(has_pas)) or bool(include_phi1):
        return False
    if int(active_size) < max(1, int(min_size)):
        return False
    if not np.isfinite(float(residual_norm)):
        return False
    return float(residual_norm) > float(target)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class AdaptivePassSmootherResult:
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    history: jnp.ndarray
    accepted_sweeps: int
    stop_reason: str

    def tree_flatten(self):
        children = (self.x, self.residual_norm, self.history)
        aux = (self.accepted_sweeps, self.stop_reason)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        accepted_sweeps, stop_reason = aux
        x, residual_norm, history = children
        return cls(
            x=x,
            residual_norm=residual_norm,
            history=history,
            accepted_sweeps=accepted_sweeps,
            stop_reason=stop_reason,
        )


def adaptive_pas_smoother(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    target: float,
    omega: float = 1.0,
    max_sweeps: int = 3,
    min_rel_improvement: float = 2.5e-2,
    worsen_factor: float = 1.05,
    plateau_patience: int = 1,
) -> AdaptivePassSmootherResult:
    result = run_adaptive_stationary_smoother(
        matvec_fn=matvec,
        rhs_vec=jnp.asarray(rhs, dtype=jnp.float64),
        x0_vec=jnp.asarray(x0, dtype=jnp.float64),
        smoother_fn=preconditioner,
        target=float(target),
        max_steps=int(max_sweeps),
        omega=float(omega),
        upward_ratio=float(worsen_factor),
        patience=max(1, int(plateau_patience)),
        min_steps=1,
        target_ratio=1.0,
        abs_floor=max(0.0, float(target)) * max(0.0, float(min_rel_improvement)),
    )
    initial = float(result.residual_history[0]) if result.residual_history else float("inf")
    accepted = 0
    prev = initial
    for val in result.residual_history[1:]:
        if float(val) < float(prev):
            accepted += 1
            prev = float(val)
    return AdaptivePassSmootherResult(
        x=jnp.asarray(result.x_best, dtype=jnp.float64),
        residual_norm=jnp.asarray(result.best_residual_norm, dtype=jnp.float64),
        history=jnp.asarray(result.residual_history, dtype=jnp.float64),
        accepted_sweeps=int(accepted),
        stop_reason=result.stop_reason,
    )
