"""ConstraintScheme=1 nullspace projection helpers.

RHSMode=1 and RHSMode=2/3 constraintScheme=1 systems include source rows that
can leave a gauge-like nullspace component in iterative solves.  These helpers
apply the small source-basis correction used by the driver so the constraint
rows are enforced without changing the public solve API.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from .linear_algebra import small_regularized_lstsq
from .v3_system import _source_basis_constraint_scheme_1, apply_v3_full_system_operator_cached


ApplyOperatorFn = Callable[[Any, jnp.ndarray], jnp.ndarray]


def project_constraint_scheme1_nullspace_solution_with_residual(
    *,
    op: Any,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: Any,
    enabled_env_var: str,
    residual_vec: jnp.ndarray | None = None,
    apply_operator: ApplyOperatorFn = apply_v3_full_system_operator_cached,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a constraintScheme=1 solution and return its corrected residual."""
    if int(op.constraint_scheme) != 1:
        return _unchanged_or_residual(
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=matvec_op,
            residual_vec=residual_vec,
            apply_operator=apply_operator,
        )
    if int(op.phi1_size) != 0:
        return _unchanged_or_residual(
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=matvec_op,
            residual_vec=residual_vec,
            apply_operator=apply_operator,
        )
    if int(op.extra_size) == 0:
        return _unchanged_or_residual(
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=matvec_op,
            residual_vec=residual_vec,
            apply_operator=apply_operator,
        )

    project_env = os.environ.get(enabled_env_var, "").strip().lower()
    if project_env in {"0", "false", "no", "off"}:
        return _unchanged_or_residual(
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=matvec_op,
            residual_vec=residual_vec,
            apply_operator=apply_operator,
        )

    basis = _constraint_scheme1_source_basis(op)
    r = (
        residual_vec
        if residual_vec is not None and residual_vec.shape == rhs_vec.shape
        else apply_operator(matvec_op, x_vec) - rhs_vec
    )
    r_extra = r[-int(op.extra_size) :]
    proj_atol = _projection_atol(enabled_env_var)
    if proj_atol > 0.0:
        max_res = float(np.max(np.abs(np.asarray(r_extra, dtype=np.float64))))
        if max_res <= proj_atol:
            return x_vec, r

    cols_full = [apply_operator(matvec_op, v) for v in basis]
    cols_extra = [col[-int(op.extra_size) :] for col in cols_full]
    m = jnp.stack(cols_extra, axis=1)
    c_res = small_regularized_lstsq(m, -r_extra)
    x_corr = sum(v * c_res[i] for i, v in enumerate(basis))
    r_corr = sum(col * c_res[i] for i, col in enumerate(cols_full))
    # For constraintScheme=1, enforce the source rows directly and keep the corrected
    # solution. Projecting out the basis reintroduces the constraint residuals.
    return x_vec + x_corr, r + r_corr


def project_constraint_scheme1_nullspace_solution(
    *,
    op: Any,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: Any,
    enabled_env_var: str,
    apply_operator: ApplyOperatorFn = apply_v3_full_system_operator_cached,
) -> jnp.ndarray:
    """Project a constraintScheme=1 solution and return the corrected state."""
    x_proj, _r = project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x_vec,
        rhs_vec=rhs_vec,
        matvec_op=matvec_op,
        enabled_env_var=enabled_env_var,
        apply_operator=apply_operator,
    )
    return x_proj


def _unchanged_or_residual(
    *,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: Any,
    residual_vec: jnp.ndarray | None,
    apply_operator: ApplyOperatorFn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
        return x_vec, residual_vec
    return x_vec, apply_operator(matvec_op, x_vec) - rhs_vec


def _constraint_scheme1_source_basis(op: Any) -> list[jnp.ndarray]:
    xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
    ix0 = 1 if bool(op.point_at_x0) else 0
    f_shape = op.fblock.f_shape
    n_s, _, _, _, _ = f_shape

    def _basis(species_index: int, src_index: int, xpart: jnp.ndarray) -> jnp.ndarray:
        f = jnp.zeros(f_shape, dtype=jnp.float64)
        f = f.at[species_index, ix0:, 0, :, :].set(xpart[ix0:][:, None, None])
        extra = jnp.zeros((n_s, 2), dtype=jnp.float64)
        extra = extra.at[species_index, src_index].set(-1.0)
        return jnp.concatenate([f.reshape((-1,)), extra.reshape((-1,))], axis=0)

    basis: list[jnp.ndarray] = []
    for species_index in range(n_s):
        basis.append(_basis(species_index, 0, xpart1))
        basis.append(_basis(species_index, 1, xpart2))
    return basis


def _projection_atol(enabled_env_var: str) -> float:
    proj_atol_env = os.environ.get(f"{enabled_env_var}_ATOL", "").strip()
    if proj_atol_env:
        try:
            return float(proj_atol_env)
        except ValueError:
            return 0.0
    # For transport-matrix solves, skip projection when constraint residuals are
    # already at roundoff. Keep RHSMode=1 default behavior unchanged.
    return 0.0 if "RHSMODE1" in enabled_env_var else 1e-9


__all__ = [
    "project_constraint_scheme1_nullspace_solution",
    "project_constraint_scheme1_nullspace_solution_with_residual",
]
