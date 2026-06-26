"""Setup utilities shared by sparse and block preconditioner builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np


def precond_chunk_cols(
    total_size: int,
    n_cols: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Choose how many basis columns to probe at once during setup.

    The explicit column override wins over the memory-budget estimate. Invalid
    environment values deliberately fall back to conservative defaults, matching
    the historical driver behavior.
    """

    env = os.environ if environ is None else environ
    env_cols = env.get("SFINCS_JAX_PRECOND_CHUNK", "").strip()
    if env_cols:
        try:
            cols = int(env_cols)
            if cols > 0:
                return min(cols, n_cols)
        except ValueError:
            pass
    env_max_mb = env.get("SFINCS_JAX_PRECOND_MAX_MB", "").strip()
    try:
        max_mb = float(env_max_mb) if env_max_mb else 256.0
    except ValueError:
        max_mb = 256.0
    if max_mb <= 0:
        return n_cols
    bytes_per_row = int(total_size) * 8
    if bytes_per_row <= 0:
        return n_cols
    max_cols = max(1, int((max_mb * 1e6) // bytes_per_row))
    return min(n_cols, max_cols)


def matvec_submatrix(
    op_pc: object,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
    apply_operator_fn: Callable[..., jnp.ndarray],
) -> np.ndarray:
    """Assemble selected rows of selected operator columns by batched probes."""

    col_idx = np.asarray(col_idx, dtype=np.int32)
    row_idx_jnp = jnp.asarray(row_idx, dtype=jnp.int32)
    blocks: list[np.ndarray] = []
    for start in range(0, int(col_idx.shape[0]), int(chunk_cols)):
        idx = col_idx[start : start + int(chunk_cols)]
        basis = jax.nn.one_hot(jnp.asarray(idx, dtype=jnp.int32), total_size, dtype=jnp.float64)
        y = jax.vmap(
            lambda v: apply_operator_fn(
                op_pc,
                v,
                include_jacobian_terms=True,
                allow_sharding=False,
            )
        )(basis)
        y_sub = y[:, row_idx_jnp]
        blocks.append(np.asarray(y_sub, dtype=np.float64))
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=0)


def matvec_submatrix_v3_unsharded(
    op_pc: object,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
) -> np.ndarray:
    """Assemble selected V3 operator rows with the unsharded operator apply."""

    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator  # noqa: PLC0415

    return matvec_submatrix(
        op_pc,
        col_idx=col_idx,
        row_idx=row_idx,
        total_size=total_size,
        chunk_cols=chunk_cols,
        apply_operator_fn=apply_v3_full_system_operator,
    )


def hash_array(arr: jnp.ndarray | np.ndarray) -> str:
    """Stable short hash for numeric arrays used in preconditioner cache keys."""

    arr_np = np.asarray(arr, dtype=np.float64)
    return hashlib.blake2b(arr_np.tobytes(), digest_size=8).hexdigest()


def rhs_mode1_precond_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
) -> tuple[object, ...]:
    """Build the operator-only cache key for RHSMode=1 preconditioners.

    RHS-only gradients are deliberately excluded so preconditioners can be
    reused across whichRHS/profile scan points that share the same linear
    operator.
    """

    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    return (
        kind,
        str(precond_dtype),
        int(op.rhs_mode),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        int(op.constraint_scheme),
        int(op.quasineutrality_option),
        bool(op.include_phi1),
        bool(op.include_phi1_in_kinetic),
        bool(op.with_adiabatic),
        float(op.alpha),
        float(op.delta),
        float(op.dphi_hat_dpsi_hat),
        hash_array(op.adiabatic_z),
        hash_array(op.adiabatic_nhat),
        hash_array(op.adiabatic_that),
        hash_array(op.z_s),
        hash_array(op.m_hat),
        hash_array(op.t_hat),
        hash_array(op.n_hat),
        hash_array(op.theta_weights),
        hash_array(op.zeta_weights),
        hash_array(op.b_hat),
        hash_array(op.d_hat),
        hash_array(op.b_hat_sub_theta),
        hash_array(op.b_hat_sub_zeta),
        hash_array(op.x),
        hash_array(op.x_weights),
        tuple(nxi_for_x.tolist()),
    )


def rhs_mode1_structured_fblock_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
    params: tuple[object, ...] = (),
) -> tuple[object, ...]:
    """Build a cache key for structured RHSMode=1 f-block preconditioners."""

    phi1_hash = None
    if getattr(op, "phi1_hat_base", None) is not None:
        phi1_hash = hash_array(op.phi1_hat_base)
    return (
        *rhs_mode1_precond_cache_key(
            op,
            f"structured_fblock_{kind}",
            precond_dtype=precond_dtype,
        ),
        phi1_hash,
        *tuple(params),
    )


def transport_precond_cache_key(
    op: object,
    kind: str,
    *,
    precond_dtype: object,
) -> tuple[object, ...]:
    """Build the cache key for RHSMode=2/3 transport preconditioners."""

    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    pas = op.fblock.pas
    fp = op.fblock.fp
    return (
        kind,
        str(precond_dtype),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        float(op.fblock.identity_shift),
        bool(pas is not None),
        float(pas.nu_n) if pas is not None else None,
        float(pas.krook) if pas is not None else None,
        hash_array(pas.nu_d_hat) if pas is not None else None,
        bool(fp is not None),
        hash_array(fp.mat) if fp is not None else None,
        tuple(nxi_for_x.tolist()),
    )
