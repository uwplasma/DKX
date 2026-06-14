"""Constraint-source moment helpers for RHSMode=1 and transport solves.

SFINCS constraint schemes add source amplitudes that enforce density/pressure
or flux-surface-average constraints. These helpers are small JAX kernels that
convert between kinetic ``f`` blocks and source amplitudes. They are shared by
RHSMode=1 preconditioners and RHSMode=2/3 transport residual corrections.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .v3_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1


def constraint_scheme2_source_from_f(op: Any, f: jnp.ndarray) -> jnp.ndarray:
    """Return constraintScheme=2 source terms from L=0 flux-surface averages."""
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)
    return jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])


def constraint_scheme2_inject_source(op: Any, src: jnp.ndarray) -> jnp.ndarray:
    """Inject constraintScheme=2 source terms into the L=0 rows of the f block."""
    f = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
    ix0 = _ix_min(bool(op.point_at_x0))
    f = f.at[:, ix0:, 0, :, :].set(src[:, ix0:, None, None])
    return f.reshape((-1,))


def constraint_scheme1_moments_from_f(op: Any, f: jnp.ndarray) -> jnp.ndarray:
    """Return constraintScheme=1 density/pressure moments from the L=0 block."""
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)
    x2 = op.x * op.x
    x4 = x2 * x2
    w2 = x2 * op.x_weights
    w4 = x4 * op.x_weights
    y_dens = jnp.einsum("x,tz,sxtz->s", w2, factor, f[:, :, 0, :, :])
    y_pres = jnp.einsum("x,tz,sxtz->s", w4, factor, f[:, :, 0, :, :])
    return jnp.stack([y_dens, y_pres], axis=1)


def constraint_scheme1_inject_source(op: Any, src: jnp.ndarray) -> jnp.ndarray:
    """Inject constraintScheme=1 particle/energy source amplitudes into L=0 rows."""
    src = jnp.asarray(src, dtype=jnp.float64).reshape((int(op.n_species), 2))
    xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
    ix0 = _ix_min(bool(op.point_at_x0))
    f = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
    f = f.at[:, ix0:, 0, :, :].set(
        xpart1[ix0:][None, :, None, None] * src[:, 0, None, None, None]
        + xpart2[ix0:][None, :, None, None] * src[:, 1, None, None, None]
    )
    return f.reshape((-1,))


__all__ = [
    "constraint_scheme1_inject_source",
    "constraint_scheme1_moments_from_f",
    "constraint_scheme2_inject_source",
    "constraint_scheme2_source_from_f",
]
