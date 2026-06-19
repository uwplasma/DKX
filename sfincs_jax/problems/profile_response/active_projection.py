"""Reusable RHSMode=1 full/reduced active-DOF projection primitives."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


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


def fp_pitch_mode_active_indices(
    *,
    n_species: int,
    n_x: int,
    n_xi: int,
    n_theta: int,
    n_zeta: int,
    nxi_for_x: np.ndarray,
    l_min: int,
    l_max: int,
    full_to_active: np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray:
    """Return active reduced indices for FP pitch modes in a Legendre band.

    The FP distribution is stored in flattened
    ``(species, x, l, theta, zeta)`` order. When ``full_to_active`` is supplied,
    it is the historical one-based full-to-reduced map where zero means
    inactive; the returned indices are zero-based reduced indices.
    """

    nxi_for_x_np = np.asarray(nxi_for_x, dtype=np.int32)
    full_to_active_np = (
        None
        if full_to_active is None
        else np.asarray(full_to_active, dtype=np.int32)
    )
    l_min_use = max(0, int(l_min))
    l_max_use = min(max(l_min_use, int(l_max)), int(n_xi) - 1)
    selected: list[int] = []
    for s_idx in range(int(n_species)):
        for ix in range(int(n_x)):
            if ix >= int(nxi_for_x_np.size):
                continue
            lmax_x = min(int(nxi_for_x_np[ix]) - 1, int(l_max_use))
            if lmax_x < l_min_use:
                continue
            for il in range(l_min_use, lmax_x + 1):
                for it in range(int(n_theta)):
                    for iz in range(int(n_zeta)):
                        full_idx = int(
                            (
                                (
                                    ((s_idx * int(n_x) + ix) * int(n_xi) + il)
                                    * int(n_theta)
                                    + it
                                )
                                * int(n_zeta)
                                + iz
                            )
                        )
                        if full_to_active_np is not None:
                            if full_idx >= int(full_to_active_np.size):
                                continue
                            active_idx = int(full_to_active_np[full_idx]) - 1
                            if active_idx >= 0:
                                selected.append(active_idx)
                        else:
                            selected.append(full_idx)
    if not selected:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.asarray(selected, dtype=np.int32))


__all__ = [
    "expand_reduced_with_map",
    "fp_pitch_mode_active_indices",
    "project_pas_constraint_f",
    "reduce_full_with_indices",
]
