"""Matrix-free RHSMode=1 PAS correction helpers.

The routines here are intentionally opt-in building blocks. They do not build
dense angular patch inverses and only accept a correction when a fresh matrix-
free residual check shows a finite improvement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
        total += float(jnp.real(jnp.vdot(chunk, chunk)))
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
            )
        if config.max_update_norm_ratio is not None:
            x_scale = max(streaming_l2_norm(x_best, block_size=config.block_size), 1.0)
            if update_norm > x_scale * float(config.max_update_norm_ratio):
                return Rhs1PasMatrixFreeResult(
                    x=x_best,
                    residual_norm=best_norm,
                    initial_residual_norm=initial_norm,
                    residual_history=tuple(history),
                    accepted_steps=accepted_steps,
                    accepted=accepted_steps > 0,
                    reason="update-norm-too-large",
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
    )


__all__ = [
    "Rhs1PasMatrixFreeConfig",
    "Rhs1PasMatrixFreeResult",
    "rhs1_pas_matrixfree_acceptance_gate",
    "rhs1_pas_matrixfree_correction",
    "streaming_l2_norm",
]
