"""Constraint-source moment helpers for RHSMode=1 and transport solves.

SFINCS constraint schemes add source amplitudes that enforce density/pressure
or flux-surface-average constraints. These helpers are small JAX kernels that
convert between kinetic ``f`` blocks and source amplitudes. They are shared by
RHSMode=1 preconditioners and RHSMode=2/3 transport residual corrections.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.operators.profile_response.system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1


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


def build_rhs1_xblock_constraint1_moment_schur_preconditioner(
    *,
    op: Any,
    base_preconditioner: Callable[[jnp.ndarray], jnp.ndarray],
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    rcond: float = 1.0e-12,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], dict[str, object], dict[str, int]]:
    """Wrap an x-block preconditioner with a constraintScheme=1 moment Schur solve.

    ConstraintScheme=1 introduces source unknowns that enforce density and
    pressure moments. Per-x block preconditioners leave these global rows as
    slow saddle-point modes. This wrapper builds the small dense approximation
    ``S = C M^{-1} B``, where ``B`` injects particle/energy sources and ``C``
    returns density/pressure moments, then applies the block inverse with the
    supplied preconditioner as ``M^{-1}``.
    """
    from scipy.linalg import qr  # noqa: PLC0415

    if int(op.rhs_mode) != 1 or int(op.constraint_scheme) != 1:
        raise RuntimeError(
            "constraint1 moment Schur requires RHSMode=1 constraintScheme=1"
        )
    if int(op.phi1_size) != 0:
        raise RuntimeError(
            "constraint1 moment Schur currently requires includePhi1=false"
        )
    extra_size = int(op.extra_size)
    n_species = int(op.n_species)
    if extra_size != 2 * n_species or extra_size <= 0:
        raise RuntimeError(
            "constraint1 moment Schur expected two source unknowns per species"
        )

    reduced_probe = (
        jnp.zeros((op.total_size,), dtype=jnp.float64)
        if reduce_full is None
        else reduce_full(jnp.zeros((op.total_size,), dtype=jnp.float64))
    )
    expected_size = int(reduced_probe.size)
    f_size = int(op.f_size)
    zeros_extra = jnp.zeros((extra_size,), dtype=jnp.float64)
    rcond_use = max(0.0, float(rcond))

    def _to_full(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64).reshape((-1,))
        if expand_reduced is None:
            return v
        return jnp.asarray(expand_reduced(v), dtype=jnp.float64).reshape((-1,))

    def _from_full(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64).reshape((-1,))
        if reduce_full is None:
            return v
        return jnp.asarray(reduce_full(v), dtype=jnp.float64).reshape((-1,))

    def _base_full(v_full: jnp.ndarray) -> jnp.ndarray:
        v_full = jnp.asarray(v_full, dtype=jnp.float64).reshape((-1,))
        if reduce_full is None or expand_reduced is None:
            return jnp.asarray(base_preconditioner(v_full), dtype=jnp.float64).reshape(
                (-1,)
            )
        z_reduced = base_preconditioner(reduce_full(v_full))
        return jnp.asarray(expand_reduced(z_reduced), dtype=jnp.float64).reshape((-1,))

    s_mat = np.zeros((extra_size, extra_size), dtype=np.float64)
    for j in range(extra_size):
        src = np.zeros((n_species, 2), dtype=np.float64)
        src.reshape((-1,))[j] = 1.0
        f_src = constraint_scheme1_inject_source(
            op, jnp.asarray(src, dtype=jnp.float64)
        )
        y_full = _base_full(jnp.concatenate([f_src, zeros_extra], axis=0))
        y_f = y_full[:f_size].reshape(op.fblock.f_shape)
        s_col = constraint_scheme1_moments_from_f(op, y_f).reshape((-1,))
        s_mat[:, j] = np.asarray(jax.device_get(s_col), dtype=np.float64)

    if not np.all(np.isfinite(s_mat)):
        raise RuntimeError("constraint1 moment Schur matrix contains non-finite values")
    _q, r, _piv = qr(s_mat, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise RuntimeError("constraint1 moment Schur QR is empty")
    threshold = rcond_use * max(float(diag[0]), 1.0)
    rank = int(np.count_nonzero(diag > threshold))
    if rank <= 0:
        raise RuntimeError("constraint1 moment Schur is rank deficient")
    allow_rank_deficient_env = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_ALLOW_RANK_DEFICIENT",
            "",
        )
        .strip()
        .lower()
    )
    allow_rank_deficient = allow_rank_deficient_env in {
        "1",
        "true",
        "t",
        "yes",
        "on",
        ".true.",
        ".t.",
    }
    if rank < extra_size and not allow_rank_deficient:
        raise RuntimeError(
            "constraint1 moment Schur is numerically rank deficient "
            f"(rank={rank} < extra={extra_size}); refusing unstable pseudo-inverse"
        )
    if rank == extra_size:
        try:
            schur_inv_np = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            schur_inv_np = np.linalg.pinv(s_mat, rcond=max(rcond_use, 1.0e-12))
    else:
        schur_inv_np = np.linalg.pinv(s_mat, rcond=max(rcond_use, 1.0e-12))
    if not np.all(np.isfinite(schur_inv_np)):
        raise RuntimeError(
            "constraint1 moment Schur inverse contains non-finite values"
        )
    schur_inv = jnp.asarray(schur_inv_np, dtype=jnp.float64)
    stats = {"applies": 0, "base_applies": 0}

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        stats["applies"] += 1
        r_full = _to_full(v)
        r_f = r_full[:f_size]
        r_e = r_full[f_size : f_size + extra_size]
        y_full = _base_full(jnp.concatenate([r_f, zeros_extra], axis=0))
        stats["base_applies"] += 1
        y_f = y_full[:f_size].reshape(op.fblock.f_shape)
        c_y = constraint_scheme1_moments_from_f(op, y_f).reshape((-1,))
        x_e = schur_inv @ (c_y - r_e)
        f_corr = constraint_scheme1_inject_source(op, x_e.reshape((n_species, 2)))
        y_corr = _base_full(jnp.concatenate([r_f - f_corr, zeros_extra], axis=0))
        stats["base_applies"] += 1
        out_full = jnp.concatenate([y_corr[:f_size], x_e], axis=0)
        return _from_full(out_full)

    metadata: dict[str, object] = {
        "mode": "constraint1_moment_schur",
        "extra_size": int(extra_size),
        "rank": int(rank),
        "rcond": float(rcond_use),
        "expected_size": int(expected_size),
        "singular_value_proxy": tuple(
            float(v) for v in diag[: min(int(diag.size), 16)]
        ),
        "device_resident": True,
        "rank_deficient_allowed": bool(allow_rank_deficient),
    }
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            f"constraint1 moment-Schur built extra={extra_size} rank={rank}",
        )
    return _apply, metadata, stats


__all__ = [
    "build_rhs1_xblock_constraint1_moment_schur_preconditioner",
    "constraint_scheme1_inject_source",
    "constraint_scheme1_moments_from_f",
    "constraint_scheme2_inject_source",
    "constraint_scheme2_source_from_f",
]
