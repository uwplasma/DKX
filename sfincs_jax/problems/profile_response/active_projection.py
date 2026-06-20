"""Reusable RHSMode=1 full/reduced active-DOF projection primitives."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from ...constraint_projection import project_constraint_scheme1_nullspace_solution_with_residual
from ...solver import GMRESSolveResult
from .policies import rhs1_pas_source_zero_tolerance_from_env


ProjectWithResidualFn = Callable[
    ...,
    tuple[jnp.ndarray, jnp.ndarray],
]

_FALSE_TOKENS = {"0", "false", "no", "off"}
_TRUE_TOKENS = {"1", "true", "yes", "on"}


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


def finalize_rhs1_linear_solution_cleanup(
    *,
    op: Any,
    result: GMRESSolveResult,
    rhs: jnp.ndarray,
    residual_vec: jnp.ndarray | None,
    project_solution_with_residual: ProjectWithResidualFn = (
        project_constraint_scheme1_nullspace_solution_with_residual
    ),
    source_zero_tolerance: float | None = None,
) -> GMRESSolveResult:
    """Apply final RHSMode=1 projection/source cleanup to a linear solve result.

    The cleanup preserves the historical driver behavior: constraintScheme=1
    nullspace projection is enabled by default for linear no-Phi1 solves, and
    tiny constraintScheme=2 PAS source rows are zeroed only when all source
    entries fall below the configured tolerance.
    """

    if int(op.rhs_mode) != 1:
        return result

    result_use = result
    if _rhs1_project_nullspace_enabled(op):
        x_projected, residual_projected = project_solution_with_residual(
            op=op,
            x_vec=result_use.x,
            rhs_vec=rhs,
            matvec_op=op,
            enabled_env_var="SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE",
            residual_vec=(
                residual_vec
                if residual_vec is not None and residual_vec.shape == rhs.shape
                else None
            ),
        )
        if not bool(jnp.allclose(x_projected, result_use.x)):
            result_use = GMRESSolveResult(
                x=x_projected,
                residual_norm=jnp.linalg.norm(residual_projected),
            )

    if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
        zero_tol = (
            rhs1_pas_source_zero_tolerance_from_env()
            if source_zero_tolerance is None
            else float(source_zero_tolerance)
        )
        if zero_tol > 0.0:
            extra = result_use.x[-int(op.extra_size) :]
            max_abs = jnp.max(jnp.abs(extra))
            extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
            x_new = jnp.concatenate([result_use.x[: -int(op.extra_size)], extra], axis=0)
            result_use = GMRESSolveResult(x=x_new, residual_norm=result_use.residual_norm)

    return result_use


def _rhs1_project_nullspace_enabled(op: Any) -> bool:
    project_env = os.environ.get("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "").strip().lower()
    if project_env in _FALSE_TOKENS:
        return False
    if project_env in _TRUE_TOKENS:
        return True
    # Default parity-first behavior: enforce constraintScheme=1 nullspace
    # projection for linear RHSMode=1 solves without Phi1.
    return bool(int(op.constraint_scheme) == 1 and not bool(op.include_phi1))


__all__ = [
    "expand_reduced_with_map",
    "finalize_rhs1_linear_solution_cleanup",
    "fp_pitch_mode_active_indices",
    "project_pas_constraint_f",
    "reduce_full_with_indices",
]
