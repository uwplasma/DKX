"""Small RHSMode=1 residual norm and gate helpers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def l2_norm_float(values: jnp.ndarray) -> float:
    """Return a host ``float`` L2 norm for JAX/NumPy-like vectors."""

    return float(jax.device_get(jnp.linalg.norm(jnp.asarray(values))))


def residual_target(*, atol: float, tol: float, rhs_norm: float) -> float:
    """Return the absolute residual target used by PETSc-style relative gates."""

    return max(float(atol), float(tol) * float(rhs_norm))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator`` only for finite positive denominators."""

    den = float(denominator)
    num = float(numerator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return None
    return num / den


def residual_converged(residual_norm: float, target: float) -> bool:
    """Return whether a finite residual satisfies a finite absolute target."""

    residual = float(residual_norm)
    target_use = float(target)
    return bool(math.isfinite(residual) and math.isfinite(target_use) and residual <= target_use)


__all__ = [
    "l2_norm_float",
    "residual_converged",
    "residual_target",
    "safe_ratio",
]
