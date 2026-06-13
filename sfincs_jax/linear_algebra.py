"""Small JAX-native linear algebra kernels used by solver infrastructure."""

from __future__ import annotations

import os

import jax.numpy as jnp


def small_regularized_lstsq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve a small least-squares problem with regularized normal equations.

    This pure-JAX path avoids GPU backends that lack the LAPACK/SVD custom calls
    used by ``jnp.linalg.lstsq`` while remaining differentiable. It is intended
    for tiny coarse/residual correction systems, not for large dense solves.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b, dtype=a.dtype)
    gram = a.T @ a
    rhs = a.T @ b
    n = int(gram.shape[0])
    if n == 0:
        return jnp.zeros((0,), dtype=a.dtype)

    reg_env = os.environ.get("SFINCS_JAX_SMALL_LSTSQ_REG", "").strip()
    try:
        reg_base = float(reg_env) if reg_env else 1e-12
    except ValueError:
        reg_base = 1e-12
    diag = jnp.diag(gram)
    scale = jnp.maximum(jnp.max(jnp.abs(diag)), jnp.asarray(1.0, dtype=a.dtype))
    reg = jnp.asarray(reg_base, dtype=a.dtype) * scale
    diag_reg = diag + reg
    eps = jnp.asarray(1e-30, dtype=a.dtype)
    inv_diag = 1.0 / jnp.maximum(jnp.abs(diag_reg), eps)

    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return gram @ x + reg * x

    x = jnp.zeros_like(rhs)
    r = rhs - matvec(x)
    z = inv_diag * r
    p = z
    rz = jnp.vdot(r, z)
    max_iters = max(8, 2 * n)

    for _ in range(max_iters):
        ap = matvec(p)
        denom = jnp.vdot(p, ap)
        denom_safe = jnp.where(jnp.abs(denom) > eps, denom, eps)
        alpha = rz / denom_safe
        x = x + alpha * p
        r = r - alpha * ap
        z = inv_diag * r
        rz_new = jnp.vdot(r, z)
        beta = rz_new / jnp.where(jnp.abs(rz) > eps, rz, 1.0)
        p = z + beta * p
        rz = rz_new

    return x
