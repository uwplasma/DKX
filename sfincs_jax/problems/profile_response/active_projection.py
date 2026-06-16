"""Reusable RHSMode=1 full/reduced active-DOF projection primitives."""

from __future__ import annotations

import jax.numpy as jnp


def reduce_full_with_indices(v_full: jnp.ndarray, active_idx: jnp.ndarray) -> jnp.ndarray:
    """Gather the active entries from a full vector."""

    return jnp.asarray(v_full)[jnp.asarray(active_idx, dtype=jnp.int32)]


def expand_reduced_with_map(v_reduced: jnp.ndarray, full_to_active: jnp.ndarray) -> jnp.ndarray:
    """Scatter a reduced vector into full ordering using a one-based index map.

    ``full_to_active[i] == 0`` denotes an inactive full-system row. Positive
    entries select ``v_reduced[full_to_active[i] - 1]``. This one-based map
    matches the historical in-driver implementation and avoids a separate mask
    allocation in JAX.
    """

    v_reduced = jnp.asarray(v_reduced)
    z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
    padded = jnp.concatenate([z0, v_reduced], axis=0)
    return padded[jnp.asarray(full_to_active, dtype=jnp.int32)]


def project_pas_constraint_f(
    f_flat: jnp.ndarray,
    *,
    f_shape: tuple[int, ...],
    fs_factor: jnp.ndarray,
    fs_sum_safe: jnp.ndarray,
    mask_x: jnp.ndarray,
) -> jnp.ndarray:
    """Project PAS ``l=0`` density-like rows to zero flux-surface average."""

    f = jnp.asarray(f_flat).reshape(f_shape)
    avg = jnp.einsum("tz,sxtz->sx", fs_factor, f[:, :, 0, :, :])
    avg = avg * mask_x[None, :]
    avg = avg / fs_sum_safe
    f = f.at[:, :, 0, :, :].add(-avg[:, :, None, None])
    return f.reshape((-1,))


__all__ = [
    "expand_reduced_with_map",
    "project_pas_constraint_f",
    "reduce_full_with_indices",
]
