"""Small runtime helpers shared by v3 solve orchestration code."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .solver import GMRESSolveResult


def gmres_result_is_finite(result: GMRESSolveResult) -> bool:
    """Return True when GMRES returned finite state and residual."""

    return bool(jnp.all(jnp.isfinite(result.x)) and jnp.isfinite(result.residual_norm))


def block_gmres_result_ready(result: GMRESSolveResult) -> GMRESSolveResult:
    """Synchronize a GMRES result so timing/profiling marks include XLA work."""

    try:
        jax.block_until_ready((result.x, result.residual_norm))
    except Exception:
        pass
    return result


__all__ = ["block_gmres_result_ready", "gmres_result_is_finite"]
