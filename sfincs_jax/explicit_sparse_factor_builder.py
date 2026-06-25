"""Host explicit-sparse operator assembly and factorization orchestration."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .explicit_sparse import (
    SparseOperatorBundle,
    build_operator_from_matvec,
    build_operator_from_pattern,
    factorize_host_sparse_operator,
)
from .explicit_sparse_factor_policy import (
    explicit_sparse_factor_settings_from_env,
    explicit_sparse_monolithic_max_size,
)
from .profiling import Timer


def build_host_sparse_direct_factor_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    factor_dtype: np.dtype,
    pattern=None,
    operator_bundle_override: SparseOperatorBundle | None = None,
    emit: Callable[[int, str], None] | None = None,
    default_diag_pivot_thresh: float = 1.0,
    default_permc_spec: str = "COLAMD",
    default_factor_kind: str = "lu",
    default_ilu_fill_factor: float = 10.0,
    default_ilu_drop_tol: float = 1.0e-4,
    default_pattern_color_batch: int = 1,
    default_symbolic_ordering_kind: str = "rcm",
    default_symbolic_block_size: int = 4096,
    default_symbolic_block_overlap: int = 0,
    default_symbolic_coarse_max_cols: int = 256,
    default_symbolic_coarse_probe_cols: int = 4,
    default_symbolic_coarse_damping: float = 1.0,
    default_symbolic_coarse_regularization_rel: float = 1.0e-10,
    default_symbolic_schur_max_separator_cols: int = 256,
    default_symbolic_schur_tail_size: int = 0,
    default_symbolic_schur_boundary_width: int = 1,
    default_symbolic_schur_high_degree_cols: int = 64,
    default_symbolic_schur_regularization_rel: float = 1.0e-12,
    default_symbolic_frontal_max_separator_cols: int = 1024,
    default_symbolic_frontal_tail_size: int = 0,
    default_symbolic_frontal_boundary_width: int = 1,
    default_symbolic_frontal_high_degree_cols: int = 128,
    default_symbolic_frontal_max_superblock_size: int = 8192,
    default_symbolic_frontal_max_superblock_blocks: int = 8,
    default_symbolic_frontal_min_cross_nnz: int = 1,
    default_symbolic_frontal_min_cross_separator_fraction: float = 0.0,
    default_symbolic_frontal_regularization_rel: float = 1.0e-12,
    default_symbolic_frontal_max_dense_rhs_entries: int = 0,
    default_symbolic_frontal_max_dense_rhs_cols_per_block: int = 0,
    default_symbolic_blr_frontal_tol: float = 1.0e-6,
    default_symbolic_blr_frontal_max_rank: int = 64,
    default_symbolic_blr_frontal_min_cols: int = 8,
    default_symbolic_blr_frontal_gmres_rtol: float = 1.0e-6,
    default_symbolic_blr_frontal_gmres_atol: float = 0.0,
    default_symbolic_blr_frontal_gmres_maxiter: int = 50,
    default_symbolic_blr_frontal_gmres_restart: int = 64,
    default_symbolic_blr_frontal_woodbury_max_rank: int = 512,
    default_symbolic_blr_frontal_woodbury_max_condition: float = 1.0e8,
    default_symbolic_nd_max_leaf_size: int = 4096,
    default_symbolic_nd_max_terminal_factor_size: int = 32768,
    default_symbolic_nd_max_depth: int = 4,
    default_symbolic_nd_separator_width: int = 64,
    default_symbolic_nd_max_separator_cols: int = 4096,
    default_symbolic_nd_high_degree_cols: int = 64,
    default_symbolic_nd_regularization_rel: float = 1.0e-12,
    default_symbolic_nd_max_dense_rhs_entries: int = 0,
    default_symbolic_nd_max_dense_rhs_entries_per_child: int = 0,
    default_symbolic_nd_max_dense_rhs_cols_per_child: int = 0,
    default_symbolic_nd_max_setup_s: float = 0.0,
    default_symbolic_nd_compress_updates: bool = False,
    default_symbolic_nd_parallel_update_workers: int = 1,
    default_symbolic_nd_residual_polish_steps: int = 0,
    default_symbolic_nd_residual_polish_damping: float = 1.0,
    default_symbolic_superblock_max_size: int = 32768,
    default_symbolic_superblock_max_blocks: int = 8,
    default_symbolic_superblock_min_cross_nnz: int = 1,
    default_symbolic_superblock_min_retained_cross_fraction: float = 0.0,
    default_symbolic_superblock_regularization_rel: float = 1.0e-12,
    default_symbolic_numeric_parallel_workers: int = 1,
    default_symbolic_max_permutation_size: int = 250_000,
    default_monolithic_guard_enabled: bool = True,
    build_operator_from_matvec_callback=build_operator_from_matvec,
    build_operator_from_pattern_callback=build_operator_from_pattern,
    factorize_host_sparse_operator_callback=factorize_host_sparse_operator,
    default_backend_callback=jax.default_backend,
    monolithic_max_size_callback=explicit_sparse_monolithic_max_size,
):
    """Build a host sparse operator and factor through dependency-injected seams."""

    factor_dtype_np = np.dtype(factor_dtype)
    sparse_settings = explicit_sparse_factor_settings_from_env(
        default_diag_pivot_thresh=default_diag_pivot_thresh,
        default_permc_spec=default_permc_spec,
        default_factor_kind=default_factor_kind,
        default_ilu_fill_factor=default_ilu_fill_factor,
        default_ilu_drop_tol=default_ilu_drop_tol,
        default_pattern_color_batch=default_pattern_color_batch,
        default_symbolic_block_overlap=default_symbolic_block_overlap,
        default_symbolic_coarse_max_cols=default_symbolic_coarse_max_cols,
        default_symbolic_coarse_probe_cols=default_symbolic_coarse_probe_cols,
        default_symbolic_coarse_damping=default_symbolic_coarse_damping,
        default_symbolic_coarse_regularization_rel=default_symbolic_coarse_regularization_rel,
        default_symbolic_schur_max_separator_cols=default_symbolic_schur_max_separator_cols,
        default_symbolic_schur_tail_size=default_symbolic_schur_tail_size,
        default_symbolic_schur_boundary_width=default_symbolic_schur_boundary_width,
        default_symbolic_schur_high_degree_cols=default_symbolic_schur_high_degree_cols,
        default_symbolic_schur_regularization_rel=default_symbolic_schur_regularization_rel,
        default_symbolic_frontal_max_separator_cols=default_symbolic_frontal_max_separator_cols,
        default_symbolic_frontal_tail_size=default_symbolic_frontal_tail_size,
        default_symbolic_frontal_boundary_width=default_symbolic_frontal_boundary_width,
        default_symbolic_frontal_high_degree_cols=default_symbolic_frontal_high_degree_cols,
        default_symbolic_frontal_max_superblock_size=default_symbolic_frontal_max_superblock_size,
        default_symbolic_frontal_max_superblock_blocks=default_symbolic_frontal_max_superblock_blocks,
        default_symbolic_frontal_min_cross_nnz=default_symbolic_frontal_min_cross_nnz,
        default_symbolic_frontal_min_cross_separator_fraction=default_symbolic_frontal_min_cross_separator_fraction,
        default_symbolic_frontal_regularization_rel=default_symbolic_frontal_regularization_rel,
        default_symbolic_frontal_max_dense_rhs_entries=default_symbolic_frontal_max_dense_rhs_entries,
        default_symbolic_frontal_max_dense_rhs_cols_per_block=default_symbolic_frontal_max_dense_rhs_cols_per_block,
        default_symbolic_blr_frontal_tol=default_symbolic_blr_frontal_tol,
        default_symbolic_blr_frontal_max_rank=default_symbolic_blr_frontal_max_rank,
        default_symbolic_blr_frontal_min_cols=default_symbolic_blr_frontal_min_cols,
        default_symbolic_blr_frontal_gmres_rtol=default_symbolic_blr_frontal_gmres_rtol,
        default_symbolic_blr_frontal_gmres_atol=default_symbolic_blr_frontal_gmres_atol,
        default_symbolic_blr_frontal_gmres_maxiter=default_symbolic_blr_frontal_gmres_maxiter,
        default_symbolic_blr_frontal_gmres_restart=default_symbolic_blr_frontal_gmres_restart,
        default_symbolic_blr_frontal_woodbury_max_rank=default_symbolic_blr_frontal_woodbury_max_rank,
        default_symbolic_blr_frontal_woodbury_max_condition=default_symbolic_blr_frontal_woodbury_max_condition,
        default_symbolic_nd_max_leaf_size=default_symbolic_nd_max_leaf_size,
        default_symbolic_nd_max_terminal_factor_size=default_symbolic_nd_max_terminal_factor_size,
        default_symbolic_nd_max_depth=default_symbolic_nd_max_depth,
        default_symbolic_nd_separator_width=default_symbolic_nd_separator_width,
        default_symbolic_nd_max_separator_cols=default_symbolic_nd_max_separator_cols,
        default_symbolic_nd_high_degree_cols=default_symbolic_nd_high_degree_cols,
        default_symbolic_nd_regularization_rel=default_symbolic_nd_regularization_rel,
        default_symbolic_nd_max_dense_rhs_entries=default_symbolic_nd_max_dense_rhs_entries,
        default_symbolic_nd_max_dense_rhs_entries_per_child=default_symbolic_nd_max_dense_rhs_entries_per_child,
        default_symbolic_nd_max_dense_rhs_cols_per_child=default_symbolic_nd_max_dense_rhs_cols_per_child,
        default_symbolic_nd_max_setup_s=default_symbolic_nd_max_setup_s,
        default_symbolic_nd_compress_updates=default_symbolic_nd_compress_updates,
        default_symbolic_nd_parallel_update_workers=default_symbolic_nd_parallel_update_workers,
        default_symbolic_nd_residual_polish_steps=default_symbolic_nd_residual_polish_steps,
        default_symbolic_nd_residual_polish_damping=default_symbolic_nd_residual_polish_damping,
        default_symbolic_superblock_max_size=default_symbolic_superblock_max_size,
        default_symbolic_superblock_max_blocks=default_symbolic_superblock_max_blocks,
        default_symbolic_superblock_min_cross_nnz=default_symbolic_superblock_min_cross_nnz,
        default_symbolic_superblock_min_retained_cross_fraction=default_symbolic_superblock_min_retained_cross_fraction,
        default_symbolic_superblock_regularization_rel=default_symbolic_superblock_regularization_rel,
        default_symbolic_numeric_parallel_workers=default_symbolic_numeric_parallel_workers,
        default_monolithic_guard_enabled=default_monolithic_guard_enabled,
    )

    def _matvec_np(x_np: np.ndarray) -> np.ndarray:
        return np.asarray(matvec(jnp.asarray(x_np, dtype=dtype)), dtype=np.float64)

    def _matmat_np(cols_np: np.ndarray) -> np.ndarray:
        cols = jnp.asarray(cols_np, dtype=dtype)
        out = jax.vmap(matvec, in_axes=1, out_axes=1)(cols)
        return np.asarray(out, dtype=np.float64)

    factor_kind = sparse_settings.factor_kind
    permc_spec = sparse_settings.permc_spec
    diag_pivot_thresh = sparse_settings.diag_pivot_thresh
    log_operator_phase = int(n) >= 10_000 or pattern is not None or operator_bundle_override is not None
    operator_build_timer = Timer()
    if emit is not None and log_operator_phase:
        operator_source = (
            "override"
            if operator_bundle_override is not None
            else ("pattern" if pattern is not None else "matvec")
        )
        emit(
            1,
            "explicit_sparse: operator assembly start "
            f"source={operator_source} n={int(n)} factor_dtype={factor_dtype_np.name}",
        )
    if operator_bundle_override is not None:
        operator_bundle = operator_bundle_override
    elif pattern is None:
        operator_bundle = build_operator_from_matvec_callback(
            _matvec_np,
            n=int(n),
            dtype=factor_dtype_np,
            backend=default_backend_callback(),
            block_cols=int(sparse_settings.block_cols),
            dense_max_mb=float(sparse_settings.dense_max_mb),
            csr_max_mb=float(sparse_settings.csr_max_mb),
            prefer_sparse_on_gpu=True,
            drop_tol=float(sparse_settings.drop_tol),
            matmat=_matmat_np,
            allow_operator_only=False,
        )
    else:
        operator_bundle = build_operator_from_pattern_callback(
            _matvec_np,
            pattern=pattern,
            dtype=factor_dtype_np,
            backend=default_backend_callback(),
            csr_max_mb=float(sparse_settings.csr_max_mb),
            drop_tol=float(sparse_settings.drop_tol),
            allow_operator_only=False,
            color_batch=int(sparse_settings.pattern_color_batch),
            matmat=_matmat_np,
            progress_callback=(
                None
                if emit is None
                else lambda message: emit(1, f"explicit_sparse: {message}")
            ),
        )
    operator_metadata = getattr(operator_bundle, "metadata", None)
    operator_nnz = getattr(operator_metadata, "nnz_estimate", None)
    operator_csr_nbytes = getattr(operator_metadata, "csr_nbytes_estimate", None)
    operator_csr_mb = None if operator_csr_nbytes is None else float(operator_csr_nbytes) / 1.0e6
    operator_csr_mb_text = "unknown" if operator_csr_mb is None else f"{operator_csr_mb:.3f}"
    operator_shape = getattr(operator_metadata, "shape", (int(n), int(n)))
    if emit is not None:
        if log_operator_phase:
            emit(
                1,
                "explicit_sparse: operator assembly complete "
                f"elapsed_s={operator_build_timer.elapsed_s():.3f} "
                f"shape={operator_shape} operator_nnz={operator_nnz} operator_csr_mb={operator_csr_mb_text}",
            )
        emit(
            1,
            "explicit_sparse: "
            f"storage={getattr(operator_metadata, 'storage_kind', 'unknown')} "
            f"reason={getattr(operator_metadata, 'reason', 'unknown')} factor_kind={factor_kind} "
            f"factor_dtype={factor_dtype_np.name} "
            f"permc={permc_spec} diag_pivot={float(diag_pivot_thresh):.3g} "
            f"operator_nnz={operator_nnz} operator_csr_mb={operator_csr_mb_text}",
        )
    if bool(sparse_settings.monolithic_guard_enabled) and factor_kind in {"lu", "ilu"}:
        max_n = monolithic_max_size_callback(factor_kind)
        operator_rows = int(operator_shape[0]) if operator_shape is not None else int(n)
        if max_n > 0 and operator_rows > max_n:
            message = (
                "explicit_sparse: monolithic factor preflight rejected "
                f"factor_kind={factor_kind} n={operator_rows} max_n={max_n} "
                "set SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND=symbolic_block_lu_coarse "
                "or raise the monolithic guard only for explicit diagnostics"
            )
            if emit is not None:
                emit(1, message)
            raise MemoryError(message)
    if emit is not None:
        emit(
            1,
            "explicit_sparse: factorization start "
            f"factor_kind={factor_kind} permc={permc_spec} "
            f"shape={operator_shape}",
        )
    factor_timer = Timer()
    try:
        factor_bundle = factorize_host_sparse_operator_callback(
            operator_bundle,
            kind=factor_kind,
            fill_factor=float(sparse_settings.ilu_fill_factor),
            drop_tol=float(sparse_settings.ilu_drop_tol),
            permc_spec=permc_spec,
            diag_pivot_thresh=float(diag_pivot_thresh),
            symbolic_ordering_kind=str(default_symbolic_ordering_kind),
            symbolic_block_size=int(default_symbolic_block_size),
            symbolic_block_overlap=int(sparse_settings.symbolic_block_overlap),
            symbolic_coarse_max_cols=int(sparse_settings.symbolic_coarse_max_cols),
            symbolic_coarse_probe_cols=int(sparse_settings.symbolic_coarse_probe_cols),
            symbolic_coarse_damping=float(sparse_settings.symbolic_coarse_damping),
            symbolic_coarse_regularization_rel=float(sparse_settings.symbolic_coarse_regularization_rel),
            symbolic_schur_max_separator_cols=int(sparse_settings.symbolic_schur_max_separator_cols),
            symbolic_schur_tail_size=int(sparse_settings.symbolic_schur_tail_size),
            symbolic_schur_boundary_width=int(sparse_settings.symbolic_schur_boundary_width),
            symbolic_schur_high_degree_cols=int(sparse_settings.symbolic_schur_high_degree_cols),
            symbolic_schur_regularization_rel=float(sparse_settings.symbolic_schur_regularization_rel),
            symbolic_frontal_max_separator_cols=int(sparse_settings.symbolic_frontal_max_separator_cols),
            symbolic_frontal_tail_size=int(sparse_settings.symbolic_frontal_tail_size),
            symbolic_frontal_boundary_width=int(sparse_settings.symbolic_frontal_boundary_width),
            symbolic_frontal_high_degree_cols=int(sparse_settings.symbolic_frontal_high_degree_cols),
            symbolic_frontal_max_superblock_size=int(sparse_settings.symbolic_frontal_max_superblock_size),
            symbolic_frontal_max_superblock_blocks=int(sparse_settings.symbolic_frontal_max_superblock_blocks),
            symbolic_frontal_min_cross_nnz=int(sparse_settings.symbolic_frontal_min_cross_nnz),
            symbolic_frontal_min_cross_separator_fraction=float(
                sparse_settings.symbolic_frontal_min_cross_separator_fraction
            ),
            symbolic_frontal_regularization_rel=float(sparse_settings.symbolic_frontal_regularization_rel),
            symbolic_frontal_max_dense_rhs_entries=int(sparse_settings.symbolic_frontal_max_dense_rhs_entries),
            symbolic_frontal_max_dense_rhs_cols_per_block=int(
                sparse_settings.symbolic_frontal_max_dense_rhs_cols_per_block
            ),
            symbolic_blr_frontal_tol=float(sparse_settings.symbolic_blr_frontal_tol),
            symbolic_blr_frontal_max_rank=int(sparse_settings.symbolic_blr_frontal_max_rank),
            symbolic_blr_frontal_min_cols=int(sparse_settings.symbolic_blr_frontal_min_cols),
            symbolic_blr_frontal_gmres_rtol=float(sparse_settings.symbolic_blr_frontal_gmres_rtol),
            symbolic_blr_frontal_gmres_atol=float(sparse_settings.symbolic_blr_frontal_gmres_atol),
            symbolic_blr_frontal_gmres_maxiter=int(sparse_settings.symbolic_blr_frontal_gmres_maxiter),
            symbolic_blr_frontal_gmres_restart=int(sparse_settings.symbolic_blr_frontal_gmres_restart),
            symbolic_blr_frontal_woodbury_max_rank=int(sparse_settings.symbolic_blr_frontal_woodbury_max_rank),
            symbolic_blr_frontal_woodbury_max_condition=float(
                sparse_settings.symbolic_blr_frontal_woodbury_max_condition
            ),
            symbolic_nd_max_leaf_size=int(sparse_settings.symbolic_nd_max_leaf_size),
            symbolic_nd_max_terminal_factor_size=int(sparse_settings.symbolic_nd_max_terminal_factor_size),
            symbolic_nd_max_depth=int(sparse_settings.symbolic_nd_max_depth),
            symbolic_nd_separator_width=int(sparse_settings.symbolic_nd_separator_width),
            symbolic_nd_max_separator_cols=int(sparse_settings.symbolic_nd_max_separator_cols),
            symbolic_nd_high_degree_cols=int(sparse_settings.symbolic_nd_high_degree_cols),
            symbolic_nd_regularization_rel=float(sparse_settings.symbolic_nd_regularization_rel),
            symbolic_nd_max_dense_rhs_entries=int(sparse_settings.symbolic_nd_max_dense_rhs_entries),
            symbolic_nd_max_dense_rhs_entries_per_child=int(
                sparse_settings.symbolic_nd_max_dense_rhs_entries_per_child
            ),
            symbolic_nd_max_dense_rhs_cols_per_child=int(sparse_settings.symbolic_nd_max_dense_rhs_cols_per_child),
            symbolic_nd_max_setup_s=float(sparse_settings.symbolic_nd_max_setup_s),
            symbolic_nd_compress_updates=bool(sparse_settings.symbolic_nd_compress_updates),
            symbolic_nd_parallel_update_workers=int(sparse_settings.symbolic_nd_parallel_update_workers),
            symbolic_nd_residual_polish_steps=int(sparse_settings.symbolic_nd_residual_polish_steps),
            symbolic_nd_residual_polish_damping=float(sparse_settings.symbolic_nd_residual_polish_damping),
            symbolic_superblock_max_size=int(sparse_settings.symbolic_superblock_max_size),
            symbolic_superblock_max_blocks=int(sparse_settings.symbolic_superblock_max_blocks),
            symbolic_superblock_min_cross_nnz=int(sparse_settings.symbolic_superblock_min_cross_nnz),
            symbolic_superblock_min_retained_cross_fraction=float(
                sparse_settings.symbolic_superblock_min_retained_cross_fraction
            ),
            symbolic_superblock_regularization_rel=float(sparse_settings.symbolic_superblock_regularization_rel),
            symbolic_numeric_parallel_workers=int(sparse_settings.symbolic_numeric_parallel_workers),
            symbolic_max_permutation_size=int(default_symbolic_max_permutation_size),
        )
    except Exception as exc:
        if emit is not None:
            emit(
                1,
                "explicit_sparse: factorization failed "
                f"factor_kind={factor_kind} elapsed_s={factor_timer.elapsed_s():.3f} "
                f"({type(exc).__name__}: {exc})",
            )
        raise
    if emit is not None:
        factor_nbytes = getattr(factor_bundle, "factor_nbytes_estimate", None)
        factor_nnz = getattr(factor_bundle, "factor_nnz_estimate", None)
        factor_elapsed_s = getattr(factor_bundle, "factor_s", None)
        if factor_elapsed_s is None:
            factor_elapsed_s = factor_timer.elapsed_s()
        factor_mb = None if factor_nbytes is None else float(factor_nbytes) / 1.0e6
        factor_mb_text = "unknown" if factor_mb is None else f"{factor_mb:.3f}"
        emit(
            1,
            "explicit_sparse: factorization complete "
            f"factor_kind={factor_bundle.kind} elapsed_s={float(factor_elapsed_s or 0.0):.3f} "
            f"factor_nnz={factor_nnz} factor_mb={factor_mb_text}",
        )
    return operator_bundle, factor_bundle


__all__ = ["build_host_sparse_direct_factor_from_matvec"]
