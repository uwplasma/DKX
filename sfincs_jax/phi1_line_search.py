from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np


def advance_phi1_newton_iterate(
    *,
    x: jnp.ndarray,
    step_direction: jnp.ndarray,
    residual_norm0: float,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    accepted: Sequence[jnp.ndarray],
    mode: str,
    step_scale: float,
    factor: float | None,
    c1: float,
    maxiter: int,
) -> jnp.ndarray:
    """Apply the bounded Newton line-search/update policy for Phi1 solves.

    The acceptance logic follows the current production semantics:
    - ``basic`` / ``full`` take the scaled full step immediately,
    - ``best`` evaluates a fixed candidate list and takes the best finite trial,
    - the PETSc-like modes use a simple Armijo/factor test with backtracking.
    """
    if mode in {"basic", "full"}:
        return x + (float(step_scale) * step_direction)

    step = 1.0
    best_x = None
    best_rnorm = float("inf")
    step_candidates = [1.0, 1.5, 2.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] if mode == "best" else None

    for _ in range(int(maxiter)):
        if mode == "best":
            try_step = step_candidates.pop(0) if step_candidates else step
        else:
            try_step = step

        x_try = x + (float(try_step) * float(step_scale)) * step_direction
        r_try = residual_fn(x_try)
        rnorm_try = float(jnp.linalg.norm(r_try))
        if not np.isfinite(rnorm_try):
            if mode != "best":
                step *= 0.5
            continue

        if rnorm_try < best_rnorm:
            best_rnorm = rnorm_try
            best_x = x_try

        if mode != "best":
            if factor is not None:
                accept = rnorm_try <= float(factor) * float(residual_norm0)
            else:
                accept = rnorm_try <= (1.0 - float(c1) * float(step)) * float(residual_norm0)
            if accept:
                return x_try
            step *= 0.5

    if mode == "best" and best_x is not None and best_rnorm < float(residual_norm0):
        return best_x
    if best_x is not None and np.isfinite(best_rnorm):
        return best_x
    if accepted:
        return accepted[-1]
    return x + (1.0 / 64.0) * step_direction
