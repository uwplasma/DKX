"""Angular line and block preconditioners for RHSMode=1 solves."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_caches import (
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_SCHWARZ_PRECOND_CACHE,
    _RHSMODE1_THETA_LINE_DIAGX_CACHE,
    _RHSMode1PrecondCache,
    _RHSMode1SchwarzPrecondCache,
    _RHSMode1ThetaLineDiagXCache,
)
from sfincs_jax.solvers.preconditioner_context import precond_dtype
from sfincs_jax.solvers.preconditioner_operators import (
    _build_rhsmode1_preconditioner_operator_theta_dd,
    _build_rhsmode1_preconditioner_operator_theta_line,
    _build_rhsmode1_preconditioner_operator_zeta_dd,
    _build_rhsmode1_preconditioner_operator_zeta_line,
)
from sfincs_jax.solvers.preconditioner_setup import (
    matvec_submatrix_v3_unsharded,
    precond_chunk_cols,
    rhs_mode1_precond_cache_key,
)
from ....problems.profile_response.residual import (
    compose_multilevel_residual_correction_preconditioner,
    safe_preconditioner,
)
from sfincs_jax.operators.profile_response.system import V3FullSystemOperator, _matvec_shard_axis, apply_v3_full_system_operator_cached

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]


# Domain-decomposition sizing policy helpers.
def _dd_core_patch_ranges(n: int, block: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return ``(core_start, core_end, patch_start, patch_end)`` DD/RAS blocks."""

    n = int(n)
    block = max(1, int(block))
    overlap = max(0, int(overlap))
    ranges: list[tuple[int, int, int, int]] = []
    for core_start in range(0, n, block):
        core_end = min(n, core_start + block)
        patch_start = max(0, core_start - overlap)
        patch_end = min(n, core_end + overlap)
        ranges.append((core_start, core_end, patch_start, patch_end))
    return ranges


def _rhs1_dd_auto_block_size(
    *,
    n: int,
    n_dev: int,
    sum_nxi: int,
    dof_target: int,
) -> int:
    """Choose a shard-local Schwarz block that spans more than one local shard."""

    n = max(1, int(n))
    n_dev = max(1, int(n_dev))
    sum_nxi = max(1, int(sum_nxi))
    dof_target = max(128, int(dof_target))
    local_n = max(1, (n + n_dev - 1) // n_dev)
    dof_cap = max(2, int(dof_target) // int(sum_nxi))
    block = max(dof_cap, local_n + dof_cap)
    return max(1, min(n, int(block)))


def _rhs1_dd_coarse_block_size(*, n: int, block: int, overlap: int) -> int:
    """Choose a wider coarse theta/zeta block for two-level Schwarz correction."""

    n = max(1, int(n))
    block = max(1, min(n, int(block)))
    overlap = max(0, int(overlap))
    coarse = block + max(8, block // 2, 2 * overlap)
    return max(1, min(n, int(coarse)))


def _rhs1_dd_coarse_level_count(*, n_dev: int) -> int:
    """Choose how many coarse residual-correction levels to apply."""

    n_dev = max(1, int(n_dev))
    coarse_levels_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "").strip()
    if coarse_levels_env:
        try:
            return max(0, int(coarse_levels_env))
        except ValueError:
            return 1
    if n_dev >= 8:
        return 2
    if n_dev >= 4:
        return 1
    return 0


def _rhs1_dd_coarse_block_sizes(*, n: int, block: int, overlap: int, levels: int) -> tuple[int, ...]:
    """Choose one or more increasingly wider coarse theta/zeta block sizes."""

    n = max(1, int(n))
    current = max(1, min(n, int(block)))
    out: list[int] = []
    for _ in range(max(0, int(levels))):
        next_block = _rhs1_dd_coarse_block_size(n=n, block=current, overlap=overlap)
        if next_block <= current:
            break
        out.append(int(next_block))
        current = int(next_block)
        if current >= n:
            break
    return tuple(out)


__all__ = (
    "_dd_core_patch_ranges",
    "_rhs1_dd_auto_block_size",
    "_rhs1_dd_coarse_block_size",
    "_rhs1_dd_coarse_block_sizes",
    "_rhs1_dd_coarse_level_count",
    "build_rhs1_theta_dd_preconditioner",
    "build_rhs1_theta_line_preconditioner",
    "build_rhs1_theta_schwarz_preconditioner",
    "build_rhs1_theta_line_xdiag_preconditioner",
    "build_rhs1_theta_zeta_preconditioner",
    "build_rhs1_zeta_dd_preconditioner",
    "build_rhs1_zeta_line_preconditioner",
    "build_rhs1_zeta_schwarz_preconditioner",
)


def _cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=precond_dtype())


def _regularization_from_env() -> np.float64:
    reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
    reg_val = float(reg_env) if reg_env else 1e-10
    return np.float64(reg_val)


def _extra_inverse(
    *,
    op_pc: V3FullSystemOperator,
    op: V3FullSystemOperator,
    total_size: int,
    dtype: jnp.dtype,
    reg: np.float64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
    extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
    extra_inv_jnp: jnp.ndarray | None = None
    if extra_size > 0:
        chunk_cols = precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
        y_sub = matvec_submatrix_v3_unsharded(
            op_pc,
            col_idx=extra_idx_np,
            row_idx=extra_idx_np,
            total_size=total_size,
            chunk_cols=chunk_cols,
        )
        ee = np.asarray(y_sub.T, dtype=np.float64)
        ee = ee + reg * np.eye(extra_size, dtype=np.float64)
        try:
            ee_inv = np.linalg.inv(ee)
        except np.linalg.LinAlgError:
            ee_inv = np.linalg.pinv(ee, rcond=1e-12)
        if not np.all(np.isfinite(ee_inv)):
            ee_inv = np.linalg.pinv(ee, rcond=1e-12)
        extra_inv_jnp = jnp.asarray(ee_inv, dtype=dtype)
    return extra_idx_jnp, extra_inv_jnp


def _axis_line_index_map(
    *,
    op: V3FullSystemOperator,
    axis: str,
) -> tuple[np.ndarray, int]:
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))

    if axis == "theta":
        line_size = int(n_theta * local_per_species)
        idx_map = np.zeros((n_species, n_zeta, line_size), dtype=np.int32)
        for s in range(n_species):
            for iz in range(n_zeta):
                k = 0
                for it in range(n_theta):
                    for ix in range(n_x):
                        for il in range(int(nxi_for_x[ix])):
                            idx_map[s, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
        return idx_map, line_size

    if axis == "zeta":
        line_size = int(n_zeta * local_per_species)
        idx_map = np.zeros((n_species, n_theta, line_size), dtype=np.int32)
        for s in range(n_species):
            for it in range(n_theta):
                k = 0
                for iz in range(n_zeta):
                    for ix in range(n_x):
                        for il in range(int(nxi_for_x[ix])):
                            idx_map[s, it, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
        return idx_map, line_size

    raise ValueError(f"unknown line-block axis {axis!r}")


def _build_axis_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    kind: str,
    axis: str,
    op_pc: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> Preconditioner:
    cache_key = _cache_key(op, kind)
    dtype = precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        total_size = int(op.total_size)
        idx_map, line_size = _axis_line_index_map(op=op, axis=axis)
        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        reg = _regularization_from_env()

        block_inv = np.zeros((*idx_map.shape[:2], line_size, line_size), dtype=np.float64)
        for i0 in range(idx_map.shape[0]):
            for i1 in range(idx_map.shape[1]):
                rep_idx = np.asarray(idx_map[i0, i1, :], dtype=np.int32)
                chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                y_sub = matvec_submatrix_v3_unsharded(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)
                a = a + reg * np.eye(line_size, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                block_inv[i0, i1, :, :] = inv

        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op_pc=op_pc,
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )
        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=jnp.asarray(block_inv, dtype=dtype),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    line_size = int(cached.block_inv_jnp.shape[-1])
    outer0 = int(cached.block_inv_jnp.shape[0])
    outer1 = int(cached.block_inv_jnp.shape[1])
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        r_loc = r_full[flat_idx_jnp].reshape((outer0, outer1, line_size))
        z_loc = jnp.einsum("ijab,ijb->ija", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_theta_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build blocks coupling all theta points for each species/zeta line."""

    return _build_axis_line_preconditioner(
        op=op,
        kind="theta_line",
        axis="theta",
        op_pc=_build_rhsmode1_preconditioner_operator_theta_line(op),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_theta_dd_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build theta-domain-decomposition blocks over each species/zeta line."""

    return _build_axis_line_preconditioner(
        op=op,
        kind=f"theta_dd_{int(block)}",
        axis="theta",
        op_pc=_build_rhsmode1_preconditioner_operator_theta_dd(op, block=int(block)),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_zeta_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build blocks coupling all zeta points for each species/theta line."""

    return _build_axis_line_preconditioner(
        op=op,
        kind="zeta_line",
        axis="zeta",
        op_pc=_build_rhsmode1_preconditioner_operator_zeta_line(op),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_zeta_dd_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build zeta-domain-decomposition blocks over each species/theta line."""

    return _build_axis_line_preconditioner(
        op=op,
        kind=f"zeta_dd_{int(block)}",
        axis="zeta",
        op_pc=_build_rhsmode1_preconditioner_operator_zeta_dd(op, block=int(block)),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )



def build_rhs1_theta_schwarz_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    overlap: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Restricted additive Schwarz preconditioner on theta lines.

    Uses local theta patches with configurable overlap and writes only each
    patch's non-overlapped core region (RAS), avoiding duplicate writes while
    still capturing nearest-neighbor line coupling across block boundaries.
    """
    block = max(1, min(int(op.n_theta), int(block)))
    overlap = max(0, min(int(op.n_theta) - 1, int(overlap)))
    cache_key = _cache_key(op, f"theta_schwarz_b{block}_o{overlap}")
    dtype = precond_dtype()
    cached = _RHSMODE1_SCHWARZ_PRECOND_CACHE.get(cache_key)
    if cached is None:
        block_pc = max(block, block + 2 * overlap)
        op_pc = _build_rhsmode1_preconditioner_operator_theta_dd(op, block=block_pc)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        inv_list: list[np.ndarray] = []
        patch_idx_list: list[np.ndarray] = []
        core_idx_list: list[np.ndarray] = []
        core_local_list: list[np.ndarray] = []

        ranges = _dd_core_patch_ranges(n=n_theta, block=block, overlap=overlap)
        for s in range(n_species):
            for iz in range(n_zeta):
                for core_start, core_end, patch_start, patch_end in ranges:
                    patch_idx: list[int] = []
                    core_idx: list[int] = []
                    core_local: list[int] = []
                    k = 0
                    for it in range(patch_start, patch_end):
                        for ix in range(n_x):
                            max_l = int(nxi_for_x[ix])
                            for il in range(max_l):
                                idx = int(((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz))
                                patch_idx.append(idx)
                                if core_start <= it < core_end:
                                    core_idx.append(idx)
                                    core_local.append(k)
                                k += 1
                    if not patch_idx:
                        continue
                    patch_idx_np = np.asarray(patch_idx, dtype=np.int32)
                    core_idx_np = np.asarray(core_idx, dtype=np.int32)
                    core_local_np = np.asarray(core_local, dtype=np.int32)
                    chunk_cols = precond_chunk_cols(total_size, int(patch_idx_np.shape[0]))
                    y_sub = matvec_submatrix_v3_unsharded(
                        op_pc,
                        col_idx=patch_idx_np,
                        row_idx=patch_idx_np,
                        total_size=total_size,
                        chunk_cols=chunk_cols,
                    )
                    a = np.asarray(y_sub.T, dtype=np.float64)
                    a = a + reg * np.eye(int(patch_idx_np.shape[0]), dtype=np.float64)
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_list.append(inv)
                    patch_idx_list.append(patch_idx_np)
                    core_idx_list.append(core_idx_np)
                    core_local_list.append(core_local_np)

        if not inv_list:
            return build_rhs1_theta_dd_preconditioner(
                op=op,
                block=block,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

        n_patch = len(inv_list)
        max_patch = max(int(v.shape[0]) for v in patch_idx_list)
        max_core = max(int(v.shape[0]) for v in core_idx_list)
        inv_padded = np.zeros((n_patch, max_patch, max_patch), dtype=np.float64)
        patch_idx_padded = np.zeros((n_patch, max_patch), dtype=np.int32)
        core_idx_padded = np.zeros((n_patch, max_core), dtype=np.int32)
        core_local_padded = np.zeros((n_patch, max_core), dtype=np.int32)
        core_mask_padded = np.zeros((n_patch, max_core), dtype=np.float64)
        for i in range(n_patch):
            patch_n = int(patch_idx_list[i].shape[0])
            core_n = int(core_idx_list[i].shape[0])
            inv_padded[i, :patch_n, :patch_n] = inv_list[i]
            patch_idx_padded[i, :patch_n] = patch_idx_list[i]
            core_idx_padded[i, :core_n] = core_idx_list[i]
            core_local_padded[i, :core_n] = core_local_list[i]
            core_mask_padded[i, :core_n] = 1.0

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = matvec_submatrix_v3_unsharded(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=dtype)

        cached = _RHSMode1SchwarzPrecondCache(
            inv_padded_jnp=jnp.asarray(inv_padded, dtype=dtype),
            patch_idx_padded_jnp=jnp.asarray(patch_idx_padded, dtype=jnp.int32),
            core_idx_padded_jnp=jnp.asarray(core_idx_padded, dtype=jnp.int32),
            core_local_padded_jnp=jnp.asarray(core_local_padded, dtype=jnp.int32),
            core_mask_padded_jnp=jnp.asarray(core_mask_padded, dtype=dtype),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_SCHWARZ_PRECOND_CACHE[cache_key] = cached

    inv_padded_jnp = cached.inv_padded_jnp
    patch_idx_padded_jnp = cached.patch_idx_padded_jnp
    core_idx_padded_jnp = cached.core_idx_padded_jnp
    core_local_padded_jnp = cached.core_local_padded_jnp
    core_mask_padded_jnp = cached.core_mask_padded_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_patch = int(inv_padded_jnp.shape[0])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        z_full = jnp.zeros_like(r_full)
        for ip in range(n_patch):
            r_patch = r_full[patch_idx_padded_jnp[ip]]
            z_patch = inv_padded_jnp[ip] @ r_patch
            z_core = z_patch[core_local_padded_jnp[ip]] * core_mask_padded_jnp[ip]
            z_full = z_full.at[core_idx_padded_jnp[ip]].add(z_core)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    coarse_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE", "").strip().lower()
    coarse_auto = coarse_env not in {"0", "false", "no", "off"}
    coarse_steps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS", "").strip()
    try:
        coarse_steps = int(coarse_steps_env) if coarse_steps_env else 1
    except ValueError:
        coarse_steps = 1
    coarse_damp_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP", "").strip()
    try:
        coarse_damp = float(coarse_damp_env) if coarse_damp_env else 1.0
    except ValueError:
        coarse_damp = 1.0

    apply_full_raw = _apply_full
    n_dev = max(1, int(jax.device_count()))
    if coarse_auto and n_dev >= 4 and _matvec_shard_axis(op) == "theta":
        coarse_blocks = _rhs1_dd_coarse_block_sizes(
            n=int(op.n_theta),
            block=int(block),
            overlap=int(overlap),
            levels=_rhs1_dd_coarse_level_count(n_dev=n_dev),
        )
        if coarse_blocks:
            coarse_levels = tuple(
                build_rhs1_theta_dd_preconditioner(op=op, block=coarse_block)
                for coarse_block in coarse_blocks
            )
            apply_full_raw = compose_multilevel_residual_correction_preconditioner(
                base=apply_full_raw,
                coarse_levels=coarse_levels,
                matvec=lambda v: apply_v3_full_system_operator_cached(op, v),
                damping=coarse_damp,
                steps=coarse_steps,
            )

    apply_full_safe = safe_preconditioner(apply_full_raw)
    if reduce_full is None or expand_reduced is None:
        return apply_full_safe

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full_safe(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_zeta_schwarz_preconditioner(
    *,
    op: V3FullSystemOperator,
    block: int,
    overlap: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Restricted additive Schwarz preconditioner on zeta lines."""
    block = max(1, min(int(op.n_zeta), int(block)))
    overlap = max(0, min(int(op.n_zeta) - 1, int(overlap)))
    cache_key = _cache_key(op, f"zeta_schwarz_b{block}_o{overlap}")
    dtype = precond_dtype()
    cached = _RHSMODE1_SCHWARZ_PRECOND_CACHE.get(cache_key)
    if cached is None:
        block_pc = max(block, block + 2 * overlap)
        op_pc = _build_rhsmode1_preconditioner_operator_zeta_dd(op, block=block_pc)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        inv_list: list[np.ndarray] = []
        patch_idx_list: list[np.ndarray] = []
        core_idx_list: list[np.ndarray] = []
        core_local_list: list[np.ndarray] = []

        ranges = _dd_core_patch_ranges(n=n_zeta, block=block, overlap=overlap)
        for s in range(n_species):
            for it in range(n_theta):
                for core_start, core_end, patch_start, patch_end in ranges:
                    patch_idx: list[int] = []
                    core_idx: list[int] = []
                    core_local: list[int] = []
                    k = 0
                    for iz in range(patch_start, patch_end):
                        for ix in range(n_x):
                            max_l = int(nxi_for_x[ix])
                            for il in range(max_l):
                                idx = int(((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz))
                                patch_idx.append(idx)
                                if core_start <= iz < core_end:
                                    core_idx.append(idx)
                                    core_local.append(k)
                                k += 1
                    if not patch_idx:
                        continue
                    patch_idx_np = np.asarray(patch_idx, dtype=np.int32)
                    core_idx_np = np.asarray(core_idx, dtype=np.int32)
                    core_local_np = np.asarray(core_local, dtype=np.int32)
                    chunk_cols = precond_chunk_cols(total_size, int(patch_idx_np.shape[0]))
                    y_sub = matvec_submatrix_v3_unsharded(
                        op_pc,
                        col_idx=patch_idx_np,
                        row_idx=patch_idx_np,
                        total_size=total_size,
                        chunk_cols=chunk_cols,
                    )
                    a = np.asarray(y_sub.T, dtype=np.float64)
                    a = a + reg * np.eye(int(patch_idx_np.shape[0]), dtype=np.float64)
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_list.append(inv)
                    patch_idx_list.append(patch_idx_np)
                    core_idx_list.append(core_idx_np)
                    core_local_list.append(core_local_np)

        if not inv_list:
            return build_rhs1_zeta_dd_preconditioner(
                op=op,
                block=block,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

        n_patch = len(inv_list)
        max_patch = max(int(v.shape[0]) for v in patch_idx_list)
        max_core = max(int(v.shape[0]) for v in core_idx_list)
        inv_padded = np.zeros((n_patch, max_patch, max_patch), dtype=np.float64)
        patch_idx_padded = np.zeros((n_patch, max_patch), dtype=np.int32)
        core_idx_padded = np.zeros((n_patch, max_core), dtype=np.int32)
        core_local_padded = np.zeros((n_patch, max_core), dtype=np.int32)
        core_mask_padded = np.zeros((n_patch, max_core), dtype=np.float64)
        for i in range(n_patch):
            patch_n = int(patch_idx_list[i].shape[0])
            core_n = int(core_idx_list[i].shape[0])
            inv_padded[i, :patch_n, :patch_n] = inv_list[i]
            patch_idx_padded[i, :patch_n] = patch_idx_list[i]
            core_idx_padded[i, :core_n] = core_idx_list[i]
            core_local_padded[i, :core_n] = core_local_list[i]
            core_mask_padded[i, :core_n] = 1.0

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = matvec_submatrix_v3_unsharded(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=dtype)

        cached = _RHSMode1SchwarzPrecondCache(
            inv_padded_jnp=jnp.asarray(inv_padded, dtype=dtype),
            patch_idx_padded_jnp=jnp.asarray(patch_idx_padded, dtype=jnp.int32),
            core_idx_padded_jnp=jnp.asarray(core_idx_padded, dtype=jnp.int32),
            core_local_padded_jnp=jnp.asarray(core_local_padded, dtype=jnp.int32),
            core_mask_padded_jnp=jnp.asarray(core_mask_padded, dtype=dtype),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_SCHWARZ_PRECOND_CACHE[cache_key] = cached

    inv_padded_jnp = cached.inv_padded_jnp
    patch_idx_padded_jnp = cached.patch_idx_padded_jnp
    core_idx_padded_jnp = cached.core_idx_padded_jnp
    core_local_padded_jnp = cached.core_local_padded_jnp
    core_mask_padded_jnp = cached.core_mask_padded_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_patch = int(inv_padded_jnp.shape[0])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        z_full = jnp.zeros_like(r_full)
        for ip in range(n_patch):
            r_patch = r_full[patch_idx_padded_jnp[ip]]
            z_patch = inv_padded_jnp[ip] @ r_patch
            z_core = z_patch[core_local_padded_jnp[ip]] * core_mask_padded_jnp[ip]
            z_full = z_full.at[core_idx_padded_jnp[ip]].add(z_core)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    coarse_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE", "").strip().lower()
    coarse_auto = coarse_env not in {"0", "false", "no", "off"}
    coarse_steps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS", "").strip()
    try:
        coarse_steps = int(coarse_steps_env) if coarse_steps_env else 1
    except ValueError:
        coarse_steps = 1
    coarse_damp_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP", "").strip()
    try:
        coarse_damp = float(coarse_damp_env) if coarse_damp_env else 1.0
    except ValueError:
        coarse_damp = 1.0

    apply_full_raw = _apply_full
    n_dev = max(1, int(jax.device_count()))
    if coarse_auto and n_dev >= 4 and _matvec_shard_axis(op) == "zeta":
        coarse_blocks = _rhs1_dd_coarse_block_sizes(
            n=int(op.n_zeta),
            block=int(block),
            overlap=int(overlap),
            levels=_rhs1_dd_coarse_level_count(n_dev=n_dev),
        )
        if coarse_blocks:
            coarse_levels = tuple(
                build_rhs1_zeta_dd_preconditioner(op=op, block=coarse_block)
                for coarse_block in coarse_blocks
            )
            apply_full_raw = compose_multilevel_residual_correction_preconditioner(
                base=apply_full_raw,
                coarse_levels=coarse_levels,
                matvec=lambda v: apply_v3_full_system_operator_cached(op, v),
                damping=coarse_damp,
                steps=coarse_steps,
            )

    apply_full_safe = safe_preconditioner(apply_full_raw)
    if reduce_full is None or expand_reduced is None:
        return apply_full_safe

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full_safe(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_theta_line_xdiag_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build independent theta-line blocks for each species/x/L/zeta tuple."""

    cache_key = _cache_key(op, "theta_line_xdiag")
    dtype = precond_dtype()
    cached = _RHSMODE1_THETA_LINE_DIAGX_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_theta_line(op)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        reg = _regularization_from_env()

        block_inv_list: list[np.ndarray] = []
        block_idx_list: list[np.ndarray] = []
        for s in range(n_species):
            for ix in range(n_x):
                for il in range(int(nxi_for_x[ix])):
                    for iz in range(n_zeta):
                        rep_idx = np.zeros((n_theta,), dtype=np.int32)
                        for it in range(n_theta):
                            rep_idx[it] = int(((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz))
                        chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                        y_sub = matvec_submatrix_v3_unsharded(
                            op_pc,
                            col_idx=rep_idx,
                            row_idx=rep_idx,
                            total_size=total_size,
                            chunk_cols=chunk_cols,
                        )
                        a = np.asarray(y_sub.T, dtype=np.float64)
                        a = a + reg * np.eye(n_theta, dtype=np.float64)
                        try:
                            inv = np.linalg.inv(a)
                        except np.linalg.LinAlgError:
                            inv = np.linalg.pinv(a, rcond=1e-12)
                        if not np.all(np.isfinite(inv)):
                            inv = np.linalg.pinv(a, rcond=1e-12)
                        block_inv_list.append(inv)
                        block_idx_list.append(rep_idx)

        if block_inv_list:
            block_inv = jnp.asarray(np.stack(block_inv_list), dtype=dtype)
            block_idx = jnp.asarray(np.stack(block_idx_list), dtype=jnp.int32)
        else:
            block_inv = jnp.zeros((0, n_theta, n_theta), dtype=dtype)
            block_idx = jnp.zeros((0, n_theta), dtype=jnp.int32)

        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op_pc=op_pc,
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )
        cached = _RHSMode1ThetaLineDiagXCache(
            block_inv=block_inv,
            block_idx=block_idx,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_THETA_LINE_DIAGX_CACHE[cache_key] = cached

    block_inv = cached.block_inv
    block_idx = cached.block_idx
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        z_full = jnp.zeros_like(r_full)
        if block_inv.shape[0] > 0:
            r_loc = r_full[block_idx]
            z_loc = jnp.einsum("bij,bj->bi", block_inv, r_loc)
            z_full = z_full.at[block_idx.reshape((-1,))].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_theta_zeta_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build full angular ``(theta,zeta)`` blocks for each species/x/L tuple."""

    cache_key = _cache_key(op, "theta_zeta")
    dtype = precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        total_size = int(op.total_size)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))
        block_count = int(n_species * local_per_species)
        block_size = int(n_theta * n_zeta)
        reg = _regularization_from_env()

        idx_map = np.zeros((block_count, block_size), dtype=np.int32)
        block_inv = np.zeros((block_count, block_size, block_size), dtype=np.float64)

        block_idx = 0
        for s in range(n_species):
            for ix in range(n_x):
                for il in range(int(nxi_for_x[ix])):
                    k = 0
                    for it in range(n_theta):
                        for iz in range(n_zeta):
                            idx_map[block_idx, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
                    rep_idx = idx_map[block_idx, :]
                    chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                    y_sub = matvec_submatrix_v3_unsharded(
                        op,
                        col_idx=rep_idx,
                        row_idx=rep_idx,
                        total_size=total_size,
                        chunk_cols=chunk_cols,
                    )
                    a = np.asarray(y_sub.T, dtype=np.float64)
                    a = a + reg * np.eye(block_size, dtype=np.float64)
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    block_inv[block_idx, :, :] = inv
                    block_idx += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op_pc=op,
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )
        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=jnp.asarray(block_inv, dtype=dtype),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    block_size = int(block_inv_jnp.shape[-1])
    block_count = int(block_inv_jnp.shape[0])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        r_loc = r_full[flat_idx_jnp].reshape((block_count, block_size))
        z_loc = jnp.einsum("bij,bj->bi", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced
