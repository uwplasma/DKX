"""Matrix-free RHSMode=1 PAS correction helpers.

The routines here are intentionally opt-in building blocks. They do not build
dense angular patch inverses and only accept a correction when a fresh matrix-
free residual check shows a finite improvement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import math

import jax.numpy as jnp

ArrayLike = object


@dataclass(frozen=True)
class Rhs1PasMatrixFreeConfig:
    """Acceptance controls for a bounded matrix-free PAS correction."""

    max_steps: int = 1
    omega: float = 1.0
    min_residual_reduction: float = 1.0e-3
    block_size: int | None = None
    max_update_norm_ratio: float | None = None

    def __post_init__(self) -> None:
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be >= 1")
        if not math.isfinite(float(self.omega)):
            raise ValueError("omega must be finite")
        if not math.isfinite(float(self.min_residual_reduction)):
            raise ValueError("min_residual_reduction must be finite")
        if float(self.min_residual_reduction) < 0.0:
            raise ValueError("min_residual_reduction must be >= 0")
        if self.block_size is not None and int(self.block_size) < 1:
            raise ValueError("block_size must be >= 1 when provided")
        if self.max_update_norm_ratio is not None:
            ratio = float(self.max_update_norm_ratio)
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("max_update_norm_ratio must be finite and > 0 when provided")


@dataclass(frozen=True)
class Rhs1PasMatrixFreeResult:
    """Result from a guarded matrix-free PAS correction attempt."""

    x: jnp.ndarray
    residual_norm: float
    initial_residual_norm: float
    residual_history: tuple[float, ...]
    accepted_steps: int
    accepted: bool
    reason: str
    diagnostics: dict[str, object] = field(default_factory=dict)


def _finite_or_none(value: float) -> float | None:
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _gate_diagnostics(
    *,
    reason: str,
    initial_residual_norm: float,
    candidate_residual_norm: float | None = None,
    min_residual_reduction: float,
    accepted_steps: int,
) -> dict[str, object]:
    initial = _finite_or_none(initial_residual_norm)
    candidate = None if candidate_residual_norm is None else _finite_or_none(candidate_residual_norm)
    diagnostics: dict[str, object] = {
        "reason": reason,
        "accepted_steps": int(accepted_steps),
        "initial_residual_norm": initial,
        "candidate_residual_norm": candidate,
        "min_residual_reduction": float(min_residual_reduction),
    }
    if initial is not None:
        diagnostics["required_residual_norm"] = initial * max(0.0, 1.0 - float(min_residual_reduction))
    if initial is not None and candidate is not None and initial > 0.0:
        diagnostics["residual_reduction"] = (initial - candidate) / initial
    diagnostics["initial_residual_finite"] = initial is not None
    diagnostics["candidate_residual_finite"] = candidate is not None if candidate_residual_norm is not None else None
    return diagnostics


def streaming_l2_norm(value: ArrayLike, *, block_size: int | None = None) -> float:
    """Return an L2 norm using optional flat chunks for bounded accumulation."""

    arr = jnp.ravel(jnp.asarray(value))
    if arr.size == 0:
        return 0.0
    if block_size is None:
        norm = jnp.linalg.norm(arr)
        return float(norm)
    block = max(1, int(block_size))
    total = 0.0
    for start in range(0, int(arr.size), block):
        chunk = arr[start : start + block]
        chunk_total = float(jnp.real(jnp.vdot(chunk, chunk)))
        if not math.isfinite(chunk_total):
            return chunk_total
        total += chunk_total
        if not math.isfinite(total):
            return total
    return math.sqrt(max(0.0, total))


def rhs1_pas_matrixfree_acceptance_gate(
    *,
    initial_residual_norm: float,
    candidate_residual_norm: float,
    min_residual_reduction: float,
) -> tuple[bool, str]:
    """Return whether a candidate residual is finite and sufficiently improved."""

    initial = float(initial_residual_norm)
    candidate = float(candidate_residual_norm)
    if not math.isfinite(initial):
        return False, "nonfinite-initial-residual"
    if not math.isfinite(candidate):
        return False, "nonfinite-candidate-residual"
    if initial < 0.0 or candidate < 0.0:
        return False, "negative-residual"
    required = initial * max(0.0, 1.0 - float(min_residual_reduction))
    if candidate < required:
        return True, "accepted"
    return False, "insufficient-residual-improvement"


def rhs1_pas_matrixfree_correction(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: ArrayLike,
    x0: ArrayLike,
    correction: Callable[[jnp.ndarray], jnp.ndarray],
    config: Rhs1PasMatrixFreeConfig = Rhs1PasMatrixFreeConfig(),
) -> Rhs1PasMatrixFreeResult:
    """Apply a bounded matrix-free PAS correction only when residual gates pass.

    ``correction`` receives the current residual and returns an update direction.
    The update is never trusted by construction: every proposed step is checked
    with ``matvec`` and rejected on non-finite values, shape changes, or failure
    to improve the residual by ``min_residual_reduction``.
    """

    x_initial = jnp.asarray(x0)
    rhs_arr = jnp.asarray(rhs)
    if rhs_arr.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    x_best = x_initial
    residual = rhs_arr - matvec(x_best)
    if jnp.asarray(residual).shape != x_initial.shape:
        raise ValueError("matvec(x0) must have the same shape as x0")
    initial_norm = streaming_l2_norm(residual, block_size=config.block_size)
    history: list[float] = [initial_norm]
    if not math.isfinite(initial_norm):
        return Rhs1PasMatrixFreeResult(
            x=x_initial,
            residual_norm=initial_norm,
            initial_residual_norm=initial_norm,
            residual_history=tuple(history),
            accepted_steps=0,
            accepted=False,
            reason="nonfinite-initial-residual",
            diagnostics=_gate_diagnostics(
                reason="nonfinite-initial-residual",
                initial_residual_norm=initial_norm,
                min_residual_reduction=float(config.min_residual_reduction),
                accepted_steps=0,
            ),
        )

    accepted_steps = 0
    best_norm = initial_norm
    current_residual = jnp.asarray(residual)
    for _step in range(max(1, int(config.max_steps))):
        update = jnp.asarray(correction(current_residual))
        if update.shape != x_initial.shape:
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="update-shape-mismatch",
                diagnostics={
                    "reason": "update-shape-mismatch",
                    "accepted_steps": int(accepted_steps),
                    "expected_shape": tuple(int(dim) for dim in x_initial.shape),
                    "observed_shape": tuple(int(dim) for dim in update.shape),
                },
            )
        update_norm = streaming_l2_norm(update, block_size=config.block_size)
        if not math.isfinite(update_norm):
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="nonfinite-update",
                diagnostics={
                    "reason": "nonfinite-update",
                    "accepted_steps": int(accepted_steps),
                    "update_norm": None,
                    "update_norm_finite": False,
                },
            )
        if config.max_update_norm_ratio is not None:
            x_scale = max(streaming_l2_norm(x_best, block_size=config.block_size), 1.0)
            if update_norm > x_scale * float(config.max_update_norm_ratio):
                update_limit = x_scale * float(config.max_update_norm_ratio)
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason="update-norm-too-large",
                    diagnostics={
                        "reason": "update-norm-too-large",
                        "accepted_steps": int(accepted_steps),
                        "update_norm": float(update_norm),
                        "update_norm_limit": float(update_limit),
                        "x_scale": float(x_scale),
                        "max_update_norm_ratio": float(config.max_update_norm_ratio),
                    },
                )

        candidate = x_best + jnp.asarray(float(config.omega) * update, dtype=x_initial.dtype)
        candidate = jnp.asarray(candidate, dtype=x_initial.dtype)
        candidate_residual = rhs_arr - matvec(candidate)
        if jnp.asarray(candidate_residual).shape != x_initial.shape:
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason="candidate-residual-shape-mismatch",
                diagnostics={
                    "reason": "candidate-residual-shape-mismatch",
                    "accepted_steps": int(accepted_steps),
                    "expected_shape": tuple(int(dim) for dim in x_initial.shape),
                    "observed_shape": tuple(int(dim) for dim in jnp.asarray(candidate_residual).shape),
                },
            )
        candidate_norm = streaming_l2_norm(candidate_residual, block_size=config.block_size)
        history.append(candidate_norm)
        accepted, reason = rhs1_pas_matrixfree_acceptance_gate(
            initial_residual_norm=best_norm,
            candidate_residual_norm=candidate_norm,
            min_residual_reduction=float(config.min_residual_reduction),
        )
        if not accepted:
            return Rhs1PasMatrixFreeResult(
                x=x_best,
                residual_norm=best_norm,
                initial_residual_norm=initial_norm,
                residual_history=tuple(history),
                accepted_steps=accepted_steps,
                accepted=accepted_steps > 0,
                reason=reason,
                diagnostics=_gate_diagnostics(
                    reason=reason,
                    initial_residual_norm=best_norm,
                    candidate_residual_norm=candidate_norm,
                    min_residual_reduction=float(config.min_residual_reduction),
                    accepted_steps=accepted_steps,
                ),
            )
        x_best = candidate
        current_residual = jnp.asarray(candidate_residual)
        best_norm = candidate_norm
        accepted_steps += 1

    return Rhs1PasMatrixFreeResult(
        x=x_best,
        residual_norm=best_norm,
        initial_residual_norm=initial_norm,
        residual_history=tuple(history),
        accepted_steps=accepted_steps,
        accepted=accepted_steps > 0,
        reason="accepted",
        diagnostics=_gate_diagnostics(
            reason="accepted",
            initial_residual_norm=initial_norm,
            candidate_residual_norm=best_norm,
            min_residual_reduction=float(config.min_residual_reduction),
            accepted_steps=accepted_steps,
        ),
    )


__all__ = [
    "Rhs1PasMatrixFreeConfig",
    "Rhs1PasMatrixFreeResult",
    "rhs1_pas_matrixfree_acceptance_gate",
    "rhs1_pas_matrixfree_correction",
    "streaming_l2_norm",
]
