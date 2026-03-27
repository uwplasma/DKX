from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np


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
