"""Explicit sparse operator assembly, host factors, and residual polish helpers."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
import time
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu, splu

import jax
import jax.numpy as jnp

from ..profiling import Timer
from .krylov import gmres_solve_with_history_scipy

StorageKind = Literal["dense", "csr", "linear_operator"]
FactorKind = Literal[
    "lu",
    "ilu",
    "jacobi",
    "symbolic_block_lu",
    "symbolic_block_lu_coarse",
    "symbolic_block_schur_lu",
    "symbolic_superblock_lu",
    "symbolic_frontal_schur_lu",
]


# Explicit sparse host-factor policy.
EnvMapping = Mapping[str, str]
_VALID_PERMC_SPECS = frozenset({"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"})


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    """Return the inverse of a zero-based permutation."""

    p = np.asarray(p, dtype=np.int32).reshape((-1,))
    inv = np.empty_like(p)
    inv[p] = np.arange(int(p.size), dtype=np.int32)
    return inv


def triangular_solve_lower_padded(
    *,
    lower_idx: jnp.ndarray,
    lower_val: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``L y = b`` for a unit-lower triangular factor in padded rows.

    Padding entries use index ``-1`` and are ignored. This representation is
    used by JAX-compatible sparse preconditioner apply paths that cannot call
    host triangular solves inside a compiled Krylov iteration.
    """

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    y = jnp.zeros_like(b)
    if lower_idx.size == 0:
        return b

    def _body(i, y_vec):
        idx = lower_idx[i]
        val = lower_val[i]
        mask = idx >= 0
        idx_safe = jnp.where(mask, idx, 0)
        contrib = jnp.sum(jnp.where(mask, val * y_vec[idx_safe], 0.0))
        yi = b[i] - contrib
        return y_vec.at[i].set(yi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, y)


def triangular_solve_upper_padded(
    *,
    upper_idx: jnp.ndarray,
    upper_val: jnp.ndarray,
    upper_diag: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``U x = b`` for an upper triangular factor in padded rows."""

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    x = jnp.zeros_like(b)
    if upper_idx.size == 0:
        return b / upper_diag

    def _body(i, x_vec):
        row = n - 1 - i
        idx = upper_idx[row]
        val = upper_val[row]
        mask = idx >= 0
        idx_safe = jnp.where(mask, idx, 0)
        contrib = jnp.sum(jnp.where(mask, val * x_vec[idx_safe], 0.0))
        xi = (b[row] - contrib) / upper_diag[row]
        return x_vec.at[row].set(xi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, x)


def triangular_solve_lower_csr_rows(
    *,
    indptr: jnp.ndarray,
    indices: jnp.ndarray,
    data: jnp.ndarray,
    b: jnp.ndarray,
    row_base: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``L y = b`` for one compact-CSR unit-lower block.

    ``row_base`` points to the first row pointer for this block inside a
    concatenated per-block CSR table. Column indices are local to the block.
    """

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    y = jnp.zeros_like(b)
    if data.size == 0:
        return b

    def _body(i, y_vec):
        row = row_base + i
        start = indptr[row]
        end = indptr[row + 1]

        def _accumulate(k, acc):
            return acc + data[k] * y_vec[indices[k]]

        contrib = jax.lax.fori_loop(start, end, _accumulate, jnp.asarray(0.0, dtype=b.dtype))
        return y_vec.at[i].set(b[i] - contrib, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, y)


def triangular_solve_upper_csr_rows(
    *,
    indptr: jnp.ndarray,
    indices: jnp.ndarray,
    data: jnp.ndarray,
    upper_diag: jnp.ndarray,
    b: jnp.ndarray,
    row_base: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``U x = b`` for one compact-CSR upper triangular block."""

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    x = jnp.zeros_like(b)
    if data.size == 0:
        return b / upper_diag

    def _body(i, x_vec):
        row_local = n - 1 - i
        row = row_base + row_local
        start = indptr[row]
        end = indptr[row + 1]

        def _accumulate(k, acc):
            return acc + data[k] * x_vec[indices[k]]

        contrib = jax.lax.fori_loop(start, end, _accumulate, jnp.asarray(0.0, dtype=b.dtype))
        xi = (b[row_local] - contrib) / upper_diag[row_local]
        return x_vec.at[row_local].set(xi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, x)


_FACTOR_KIND_ALIASES = {
    "jacobi": "jacobi",
    "diagonal": "jacobi",
    "diag": "jacobi",
    "none": "jacobi",
    "symbolic_block_schur_lu": "symbolic_block_schur_lu",
    "block_schur_lu": "symbolic_block_schur_lu",
    "native_block_schur_lu": "symbolic_block_schur_lu",
    "symbolic_schur_lu": "symbolic_block_schur_lu",
    "symbolic_frontal_schur_lu": "symbolic_frontal_schur_lu",
    "frontal_schur_lu": "symbolic_frontal_schur_lu",
    "native_frontal_schur_lu": "symbolic_frontal_schur_lu",
    "multifrontal_schur_lu": "symbolic_frontal_schur_lu",
    "symbolic_superblock_lu": "symbolic_superblock_lu",
    "superblock_lu": "symbolic_superblock_lu",
    "native_superblock_lu": "symbolic_superblock_lu",
    "block_edge_lu": "symbolic_superblock_lu",
    "symbolic_block_lu_coarse": "symbolic_block_lu_coarse",
    "block_lu_coarse": "symbolic_block_lu_coarse",
    "native_block_lu_coarse": "symbolic_block_lu_coarse",
    "symbolic_lu_coarse": "symbolic_block_lu_coarse",
    "symbolic_block_lu": "symbolic_block_lu",
    "block_lu": "symbolic_block_lu",
    "native_block_lu": "symbolic_block_lu",
    "symbolic_lu": "symbolic_block_lu",
    "ilu": "ilu",
    "spilu": "ilu",
    "lu": "lu",
    "splu": "lu",
}


def parse_explicit_sparse_int(value: str, default: int, *, minimum: int = 0) -> int:
    """Parse an integer explicit-sparse option with fail-closed bounds."""

    try:
        parsed = int(value) if value else int(default)
    except ValueError:
        parsed = int(default)
    return max(int(minimum), int(parsed))


def parse_explicit_sparse_float(value: str, default: float, *, minimum: float = 0.0) -> float:
    """Parse a floating-point explicit-sparse option with fail-closed bounds."""

    try:
        parsed = float(value) if value else float(default)
    except ValueError:
        parsed = float(default)
    return max(float(minimum), float(parsed))


def parse_explicit_sparse_bool(value: str, default: bool) -> bool:
    """Parse Fortran/Python-style boolean explicit-sparse options."""

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on", ".true.", ".t."}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ".false.", ".f."}:
        return False
    return bool(default)


def canonical_explicit_sparse_factor_kind(kind: str | None, *, default: str = "lu") -> str:
    """Normalize explicit-sparse factor aliases to canonical factor kinds."""

    kind_l = str(kind or "").strip().lower()
    if kind_l in _FACTOR_KIND_ALIASES:
        return _FACTOR_KIND_ALIASES[kind_l]
    default_l = str(default or "").strip().lower()
    return _FACTOR_KIND_ALIASES.get(default_l, "lu")


def explicit_sparse_factor_kind_from_env(
    default_factor_kind: str,
    *,
    env: EnvMapping | None = None,
) -> str:
    """Resolve the explicit sparse factor kind from env, then default aliases."""

    env_map = os.environ if env is None else env
    override = str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "")).strip().lower()
    if override:
        return canonical_explicit_sparse_factor_kind(override, default=default_factor_kind)
    return canonical_explicit_sparse_factor_kind(default_factor_kind, default="lu")


def explicit_sparse_monolithic_guard_enabled(
    default_enabled: bool,
    *,
    env: EnvMapping | None = None,
) -> bool:
    """Resolve whether monolithic LU/ILU preflight guards are enabled."""

    env_map = os.environ if env is None else env
    return parse_explicit_sparse_bool(
        str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_GUARD", "")).strip(),
        bool(default_enabled),
    )


def explicit_sparse_monolithic_max_size(
    factor_kind: str,
    *,
    env: EnvMapping | None = None,
    default: int = 250_000,
) -> int:
    """Resolve factor-specific monolithic LU/ILU maximum active size."""

    env_map = os.environ if env is None else env
    max_n_name = (
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_LU_MAX_SIZE"
        if str(factor_kind).strip().lower() == "lu"
        else "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_ILU_MAX_SIZE"
    )
    max_n_env = str(env_map.get(max_n_name, "")).strip()
    max_n_fallback_env = str(env_map.get("SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_MAX_SIZE", "")).strip()
    return parse_explicit_sparse_int(max_n_env or max_n_fallback_env, int(default), minimum=0)


@dataclass(frozen=True)
class ExplicitSparseFactorSettings:
    """Parsed host explicit-sparse assembly and factorization settings."""

    block_cols: int
    dense_max_mb: float
    csr_max_mb: float
    drop_tol: float
    pattern_color_batch: int
    symbolic_block_overlap: int
    symbolic_coarse_max_cols: int
    symbolic_coarse_probe_cols: int
    symbolic_coarse_damping: float
    symbolic_coarse_regularization_rel: float
    symbolic_schur_max_separator_cols: int
    symbolic_schur_tail_size: int
    symbolic_schur_boundary_width: int
    symbolic_schur_high_degree_cols: int
    symbolic_schur_regularization_rel: float
    symbolic_frontal_max_separator_cols: int
    symbolic_frontal_tail_size: int
    symbolic_frontal_boundary_width: int
    symbolic_frontal_high_degree_cols: int
    symbolic_frontal_max_superblock_size: int
    symbolic_frontal_max_superblock_blocks: int
    symbolic_frontal_min_cross_nnz: int
    symbolic_frontal_min_cross_separator_fraction: float
    symbolic_frontal_regularization_rel: float
    symbolic_frontal_max_dense_rhs_entries: int
    symbolic_frontal_max_dense_rhs_cols_per_block: int
    symbolic_superblock_max_size: int
    symbolic_superblock_max_blocks: int
    symbolic_superblock_min_cross_nnz: int
    symbolic_superblock_min_retained_cross_fraction: float
    symbolic_superblock_regularization_rel: float
    symbolic_numeric_parallel_workers: int
    factor_kind: str
    monolithic_guard_enabled: bool
    ilu_fill_factor: float
    ilu_drop_tol: float
    permc_spec: str
    diag_pivot_thresh: float


def _env_value(env: EnvMapping, name: str) -> str:
    return str(env.get(name, "")).strip()


def _explicit_sparse_permc_spec_from_env(
    default_permc_spec: str,
    *,
    env: EnvMapping,
) -> str:
    default_permc_spec_use = str(default_permc_spec).strip().upper()
    if default_permc_spec_use not in _VALID_PERMC_SPECS:
        default_permc_spec_use = "COLAMD"
    permc_spec = _env_value(env, "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC").upper()
    return permc_spec if permc_spec in _VALID_PERMC_SPECS else default_permc_spec_use


def explicit_sparse_factor_settings_from_env(
    *,
    env: EnvMapping | None = None,
    default_diag_pivot_thresh: float = 1.0,
    default_permc_spec: str = "COLAMD",
    default_factor_kind: str = "lu",
    default_ilu_fill_factor: float = 10.0,
    default_ilu_drop_tol: float = 1.0e-4,
    default_pattern_color_batch: int = 1,
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
    default_symbolic_superblock_max_size: int = 32768,
    default_symbolic_superblock_max_blocks: int = 8,
    default_symbolic_superblock_min_cross_nnz: int = 1,
    default_symbolic_superblock_min_retained_cross_fraction: float = 0.0,
    default_symbolic_superblock_regularization_rel: float = 1.0e-12,
    default_symbolic_numeric_parallel_workers: int = 1,
    default_monolithic_guard_enabled: bool = True,
) -> ExplicitSparseFactorSettings:
    """Resolve explicit-sparse settings from environment variables and defaults."""

    env_map = os.environ if env is None else env
    return ExplicitSparseFactorSettings(
        block_cols=parse_explicit_sparse_int(_env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS"), 32),
        dense_max_mb=parse_explicit_sparse_float(_env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB"), 128.0),
        csr_max_mb=parse_explicit_sparse_float(_env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB"), 512.0),
        drop_tol=parse_explicit_sparse_float(_env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL"), 0.0),
        pattern_color_batch=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH"),
            int(default_pattern_color_batch),
            minimum=1,
        ),
        symbolic_block_overlap=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLOCK_OVERLAP"),
            int(default_symbolic_block_overlap),
            minimum=0,
        ),
        symbolic_coarse_max_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_COARSE_MAX_COLS"),
            int(default_symbolic_coarse_max_cols),
            minimum=1,
        ),
        symbolic_coarse_probe_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_COARSE_PROBE_COLS"),
            int(default_symbolic_coarse_probe_cols),
            minimum=0,
        ),
        symbolic_coarse_damping=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_COARSE_DAMPING"),
            float(default_symbolic_coarse_damping),
            minimum=0.0,
        ),
        symbolic_coarse_regularization_rel=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_COARSE_REG_REL"),
            float(default_symbolic_coarse_regularization_rel),
            minimum=0.0,
        ),
        symbolic_schur_max_separator_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS"),
            int(default_symbolic_schur_max_separator_cols),
            minimum=0,
        ),
        symbolic_schur_tail_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SCHUR_TAIL_SIZE"),
            int(default_symbolic_schur_tail_size),
            minimum=0,
        ),
        symbolic_schur_boundary_width=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SCHUR_BOUNDARY_WIDTH"),
            int(default_symbolic_schur_boundary_width),
            minimum=0,
        ),
        symbolic_schur_high_degree_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SCHUR_HIGH_DEGREE_COLS"),
            int(default_symbolic_schur_high_degree_cols),
            minimum=0,
        ),
        symbolic_schur_regularization_rel=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SCHUR_REG_REL"),
            float(default_symbolic_schur_regularization_rel),
            minimum=0.0,
        ),
        symbolic_frontal_max_separator_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS"),
            int(default_symbolic_frontal_max_separator_cols),
            minimum=0,
        ),
        symbolic_frontal_tail_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_TAIL_SIZE"),
            int(default_symbolic_frontal_tail_size),
            minimum=0,
        ),
        symbolic_frontal_boundary_width=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH"),
            int(default_symbolic_frontal_boundary_width),
            minimum=0,
        ),
        symbolic_frontal_high_degree_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS"),
            int(default_symbolic_frontal_high_degree_cols),
            minimum=0,
        ),
        symbolic_frontal_max_superblock_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE"),
            int(default_symbolic_frontal_max_superblock_size),
            minimum=1,
        ),
        symbolic_frontal_max_superblock_blocks=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS"),
            int(default_symbolic_frontal_max_superblock_blocks),
            minimum=1,
        ),
        symbolic_frontal_min_cross_nnz=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MIN_CROSS_NNZ"),
            int(default_symbolic_frontal_min_cross_nnz),
            minimum=1,
        ),
        symbolic_frontal_min_cross_separator_fraction=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION"),
            float(default_symbolic_frontal_min_cross_separator_fraction),
            minimum=0.0,
        ),
        symbolic_frontal_regularization_rel=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_REG_REL"),
            float(default_symbolic_frontal_regularization_rel),
            minimum=0.0,
        ),
        symbolic_frontal_max_dense_rhs_entries=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_ENTRIES"),
            int(default_symbolic_frontal_max_dense_rhs_entries),
            minimum=0,
        ),
        symbolic_frontal_max_dense_rhs_cols_per_block=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_COLS_PER_BLOCK"),
            int(default_symbolic_frontal_max_dense_rhs_cols_per_block),
            minimum=0,
        ),
        symbolic_superblock_max_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_MAX_SIZE"),
            int(default_symbolic_superblock_max_size),
            minimum=1,
        ),
        symbolic_superblock_max_blocks=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS"),
            int(default_symbolic_superblock_max_blocks),
            minimum=1,
        ),
        symbolic_superblock_min_cross_nnz=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_MIN_CROSS_NNZ"),
            int(default_symbolic_superblock_min_cross_nnz),
            minimum=1,
        ),
        symbolic_superblock_min_retained_cross_fraction=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_MIN_RETAINED_CROSS_FRACTION"),
            float(default_symbolic_superblock_min_retained_cross_fraction),
            minimum=0.0,
        ),
        symbolic_superblock_regularization_rel=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_REG_REL"),
            float(default_symbolic_superblock_regularization_rel),
            minimum=0.0,
        ),
        symbolic_numeric_parallel_workers=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_NUMERIC_PARALLEL_WORKERS"),
            int(default_symbolic_numeric_parallel_workers),
            minimum=1,
        ),
        factor_kind=explicit_sparse_factor_kind_from_env(default_factor_kind, env=env_map),
        monolithic_guard_enabled=explicit_sparse_monolithic_guard_enabled(
            default_monolithic_guard_enabled,
            env=env_map,
        ),
        ilu_fill_factor=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_ILU_FILL_FACTOR"),
            float(default_ilu_fill_factor),
        ),
        ilu_drop_tol=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_ILU_DROP_TOL"),
            float(default_ilu_drop_tol),
        ),
        permc_spec=_explicit_sparse_permc_spec_from_env(default_permc_spec, env=env_map),
        diag_pivot_thresh=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH"),
            float(default_diag_pivot_thresh),
            minimum=0.0,
        ),
    )


def _backend_name(backend: str | None = None) -> str:
    if backend is not None and str(backend).strip():
        return str(backend).strip().lower()
    try:
        return jax.default_backend().strip().lower()
    except Exception:  # pragma: no cover - defensive fallback
        return "cpu"


def _host_array(value, *, dtype=None, copy: bool = False) -> np.ndarray:
    arr = np.asarray(jax.device_get(value), dtype=dtype)
    return np.array(arr, copy=True) if copy else arr


def csr_matvec(
    *,
    data: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    x: jnp.ndarray,
    n_rows: int | None = None,
) -> jnp.ndarray:
    """Return the sparse CSR matrix-vector product ``A @ x``.

    The implementation is JAX-native and keeps ``nnz`` static under JIT by
    passing ``total_repeat_length`` to ``jnp.repeat``.
    """

    data = jnp.asarray(data)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    x = jnp.asarray(x)

    if indptr.ndim != 1:
        raise ValueError("indptr must be 1D")
    if indices.ndim != 1 or data.ndim != 1:
        raise ValueError("data and indices must be 1D")

    if n_rows is None:
        n_rows = int(indptr.shape[0] - 1)
    if int(indptr.shape[0]) != int(n_rows) + 1:
        raise ValueError("indptr has incompatible length")

    counts = indptr[1:] - indptr[:-1]
    nnz = int(data.shape[0])
    row_ids = jnp.repeat(
        jnp.arange(int(n_rows), dtype=indices.dtype),
        counts,
        total_repeat_length=nnz,
    )
    y_vals = data * x[indices]
    return jax.ops.segment_sum(y_vals, row_ids, int(n_rows))


def estimate_dense_nbytes(shape: tuple[int, int], dtype=np.float64) -> int:
    dtype_np = np.dtype(dtype)
    return int(shape[0]) * int(shape[1]) * int(dtype_np.itemsize)


def estimate_csr_nbytes(
    shape: tuple[int, int],
    nnz: int,
    *,
    data_dtype=np.float64,
    index_dtype=np.int32,
) -> int:
    data_itemsize = np.dtype(data_dtype).itemsize
    index_itemsize = np.dtype(index_dtype).itemsize
    return int(nnz) * (data_itemsize + index_itemsize) + (int(shape[0]) + 1) * index_itemsize


def estimate_multifrontal_direct_lu_nbytes(
    nnz: int,
    *,
    fill_ratio: float = 104.0,
    data_dtype=np.float64,
    index_dtype=np.int32,
    overhead: float = 1.15,
) -> int:
    """Estimate sparse-direct LU storage from profiled nested-dissection fill.

    Production SFINCS Fortran v3 FP transport profiles on geometry-rich
    ``whichMatrix=0`` matrices show roughly 100x nonzero growth between the
    assembled preconditioner matrix and the MUMPS factors.  This helper keeps
    SFINCS-JAX admission honest: if a monolithic or near-monolithic sparse
    direct fallback would require that level of fill, reject it before spending
    minutes in setup unless the caller explicitly raises the memory cap.
    """

    nnz_use = max(0, int(nnz))
    fill_use = max(1.0, float(fill_ratio))
    overhead_use = max(1.0, float(overhead))
    entries = int(np.ceil(float(nnz_use) * fill_use))
    bytes_per_entry = int(np.dtype(data_dtype).itemsize + np.dtype(index_dtype).itemsize)
    return int(np.ceil(float(entries * bytes_per_entry) * overhead_use))


@dataclass(frozen=True)
class SparseDecision:
    storage_kind: StorageKind
    reason: str
    backend: str
    shape: tuple[int, int]
    dense_nbytes: int
    csr_nbytes_estimate: int
    nnz_estimate: int | None
    block_cols: int | None = None
    drop_tol: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "storage_kind": self.storage_kind,
            "reason": self.reason,
            "backend": self.backend,
            "shape": self.shape,
            "dense_nbytes": self.dense_nbytes,
            "csr_nbytes_estimate": self.csr_nbytes_estimate,
            "nnz_estimate": self.nnz_estimate,
            "block_cols": self.block_cols,
            "drop_tol": self.drop_tol,
        }


@dataclass(frozen=True)
class SparseOperatorBundle:
    matrix: np.ndarray | sp.spmatrix | None
    operator: LinearOperator
    metadata: SparseDecision

    def matvec(self, x) -> np.ndarray:
        return np.asarray(self.operator.matvec(np.asarray(x)))


@dataclass(frozen=True)
class SparseFactorBundle:
    factor: object
    operator: SparseOperatorBundle
    metadata: SparseDecision
    kind: FactorKind
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None

    def solve(self, rhs) -> np.ndarray:
        # Factor implementations are allowed to use in-place work arrays. Keep
        # user input isolated while avoiding extra copies during assembly paths.
        rhs_host = _host_array(
            rhs,
            dtype=self.operator.matrix.dtype if self.operator.matrix is not None else None,
            copy=True,
        )
        sol = self.factor.solve(rhs_host)
        return np.asarray(sol)


@dataclass(frozen=True)
class SparseSymbolicAnalysis:
    """Reusable structural metadata for sparse operator/factor plans.

    This deliberately stores the symbolic facts that matter for production
    preconditioner decisions without owning a PETSc/MUMPS-style symbolic object:
    pattern fingerprint, row/column density, diagonal coverage, bandwidth,
    profile, and a bounded block plan over an optional structural permutation.
    Numeric factorization can then be gated and compared across runs using a
    stable structural key before a stronger native sparse factor is introduced.
    """

    shape: tuple[int, int]
    nnz: int
    pattern_hash: str
    ordering_kind: str
    ordering_hash: str
    bandwidth: int
    profile: int
    permuted_bandwidth: int
    permuted_profile: int
    diagonal_present: int
    diagonal_missing: int
    row_nnz_min: int
    row_nnz_max: int
    row_nnz_mean: float
    row_nnz_p95: float
    col_nnz_min: int
    col_nnz_max: int
    col_nnz_mean: float
    col_nnz_p95: float
    block_size_target: int
    block_count: int
    block_size_max: int
    block_nnz_max: int
    permutation: np.ndarray | None = None
    inverse_permutation: np.ndarray | None = None

    def cache_key(self) -> tuple[object, ...]:
        return (
            tuple(int(v) for v in self.shape),
            int(self.nnz),
            str(self.pattern_hash),
            str(self.ordering_kind),
            str(self.ordering_hash),
            int(self.block_size_target),
        )

    def to_dict(self, *, include_permutation: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "shape": tuple(int(v) for v in self.shape),
            "nnz": int(self.nnz),
            "pattern_hash": str(self.pattern_hash),
            "ordering_kind": str(self.ordering_kind),
            "ordering_hash": str(self.ordering_hash),
            "bandwidth": int(self.bandwidth),
            "profile": int(self.profile),
            "permuted_bandwidth": int(self.permuted_bandwidth),
            "permuted_profile": int(self.permuted_profile),
            "diagonal_present": int(self.diagonal_present),
            "diagonal_missing": int(self.diagonal_missing),
            "row_nnz_min": int(self.row_nnz_min),
            "row_nnz_max": int(self.row_nnz_max),
            "row_nnz_mean": float(self.row_nnz_mean),
            "row_nnz_p95": float(self.row_nnz_p95),
            "col_nnz_min": int(self.col_nnz_min),
            "col_nnz_max": int(self.col_nnz_max),
            "col_nnz_mean": float(self.col_nnz_mean),
            "col_nnz_p95": float(self.col_nnz_p95),
            "block_size_target": int(self.block_size_target),
            "block_count": int(self.block_count),
            "block_size_max": int(self.block_size_max),
            "block_nnz_max": int(self.block_nnz_max),
        }
        if include_permutation and self.permutation is not None and self.inverse_permutation is not None:
            out["permutation"] = np.asarray(self.permutation, dtype=np.int64).tolist()
            out["inverse_permutation"] = np.asarray(self.inverse_permutation, dtype=np.int64).tolist()
        return out


@dataclass(frozen=True)
class SparseFactorAdmission:
    """Setup-time quality gate for an approximate sparse factor."""

    accepted: bool
    max_relative_residual: float
    median_relative_residual: float
    min_improvement_vs_identity: float
    probe_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "max_relative_residual": float(self.max_relative_residual),
            "median_relative_residual": float(self.median_relative_residual),
            "min_improvement_vs_identity": float(self.min_improvement_vs_identity),
            "probe_count": int(self.probe_count),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class _JacobiFactor:
    inverse_diagonal: np.ndarray

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.inverse_diagonal.dtype)
        if rhs_np.ndim == 2:
            return rhs_np * self.inverse_diagonal[:, None]
        return rhs_np * self.inverse_diagonal


@dataclass(frozen=True)
class _DenseInverseFactor:
    inverse: np.ndarray

    def solve(self, rhs) -> np.ndarray:
        return np.asarray(self.inverse @ np.asarray(rhs, dtype=self.inverse.dtype), dtype=self.inverse.dtype)


@dataclass(frozen=True)
class _SymbolicBlockFactor:
    """Block-diagonal sparse factor in a reusable symbolic ordering."""

    blocks: tuple[tuple[int, int, int, int, object], ...]
    analysis: SparseSymbolicAnalysis
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    dtype: np.dtype
    overlap_size: int = 0

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.permutation.size),))
        rhs_perm = rhs_np[np.asarray(self.permutation, dtype=np.int64)]
        sol_perm = np.array(rhs_perm, copy=True)
        for center_start, center_stop, local_start, local_stop, factor in self.blocks:
            center_sl = slice(int(center_start), int(center_stop))
            local_sl = slice(int(local_start), int(local_stop))
            offset0 = int(center_start) - int(local_start)
            offset1 = int(center_stop) - int(local_start)
            try:
                local_sol = np.asarray(factor.solve(rhs_perm[local_sl]), dtype=self.dtype)
                sol_perm[center_sl] = local_sol[offset0:offset1]
            except Exception:
                # Fail soft: this is a preconditioner, not the residual gate.
                sol_perm[center_sl] = rhs_perm[center_sl]
        out = np.empty_like(sol_perm)
        out[np.asarray(self.permutation, dtype=np.int64)] = sol_perm
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicBlockCoarseFactor:
    """Block factor plus sparse Galerkin residual coarse correction."""

    local_factor: _SymbolicBlockFactor
    matrix: sp.csr_matrix
    coarse_basis: sp.csr_matrix
    coarse_factor: object
    coarse_matrix_nnz: int
    dtype: np.dtype
    coarse_damping: float = 1.0

    @property
    def analysis(self) -> SparseSymbolicAnalysis:
        return self.local_factor.analysis

    @property
    def overlap_size(self) -> int:
        return int(self.local_factor.overlap_size)

    @property
    def coarse_size(self) -> int:
        return int(self.coarse_basis.shape[1])

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.matrix.shape[0]),))
        z0 = np.asarray(self.local_factor.solve(rhs_np), dtype=self.dtype).reshape(rhs_np.shape)
        if self.coarse_basis.shape[1] == 0:
            return z0
        residual = rhs_np - np.asarray(self.matrix @ z0, dtype=self.dtype)
        coarse_rhs = np.asarray(self.coarse_basis.T @ residual, dtype=self.dtype).reshape((self.coarse_size,))
        try:
            coarse_delta = np.asarray(self.coarse_factor.solve(coarse_rhs), dtype=self.dtype).reshape((self.coarse_size,))
        except Exception:
            coarse_delta = np.zeros((self.coarse_size,), dtype=self.dtype)
        correction = np.asarray(self.coarse_basis @ coarse_delta, dtype=self.dtype).reshape(rhs_np.shape)
        out = z0 + float(self.coarse_damping) * correction
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, z0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SparseCoarseCorrectionFactor:
    """Wrap a sparse factor with a caller-provided residual coarse basis."""

    base_factor: object
    matrix: sp.csr_matrix
    coarse_basis: sp.csr_matrix
    coarse_factor: object
    dtype: np.dtype
    damping: float = 1.0

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.matrix.shape[0]),))
        try:
            z0 = np.asarray(self.base_factor.solve(rhs_np), dtype=self.dtype).reshape(rhs_np.shape)
        except Exception:
            z0 = np.array(rhs_np, copy=True)
        z0_finite = np.isfinite(z0)
        if not np.all(z0_finite):
            z0 = np.where(z0_finite, z0, rhs_np)
        if self.coarse_basis.shape[1] == 0:
            return z0
        residual = rhs_np - np.asarray(self.matrix @ z0, dtype=self.dtype)
        coarse_rhs = np.asarray(self.coarse_basis.T @ residual, dtype=self.dtype).reshape((self.coarse_basis.shape[1],))
        try:
            coarse_delta = np.asarray(self.coarse_factor.solve(coarse_rhs), dtype=self.dtype).reshape((self.coarse_basis.shape[1],))
        except Exception:
            coarse_delta = np.zeros((self.coarse_basis.shape[1],), dtype=self.dtype)
        correction = np.asarray(self.coarse_basis @ coarse_delta, dtype=self.dtype).reshape(rhs_np.shape)
        out = z0 + float(self.damping) * correction
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, z0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SparseResidualPolishFactor:
    """Apply bounded residual-equation refinement around an approximate factor."""

    base_factor: object
    matrix: sp.csr_matrix
    dtype: np.dtype
    steps: int
    damping: float = 1.0
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype)
        was_vector = rhs_np.ndim == 1
        if was_vector:
            rhs_work = rhs_np.reshape((int(self.matrix.shape[0]), 1))
        else:
            rhs_work = rhs_np.reshape((int(self.matrix.shape[0]), -1))
        try:
            x = np.asarray(self.base_factor.solve(rhs_work), dtype=self.dtype)
        except Exception:
            x = np.array(rhs_work, dtype=self.dtype, copy=True)
        if x.ndim == 1:
            x = x.reshape(rhs_work.shape)
        for _ in range(max(0, int(self.steps))):
            residual = np.asarray(rhs_work - self.matrix @ x, dtype=self.dtype)
            try:
                correction = np.asarray(self.base_factor.solve(residual), dtype=self.dtype)
            except Exception:
                break
            if correction.ndim == 1:
                correction = correction.reshape(rhs_work.shape)
            candidate = np.asarray(x + float(self.damping) * correction, dtype=self.dtype)
            if not np.all(np.isfinite(candidate)):
                break
            x = candidate
        finite = np.isfinite(x)
        if not np.all(finite):
            x = np.where(finite, x, 0.0)
        return np.asarray(x[:, 0] if was_vector else x, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicSchurBlock:
    indices: np.ndarray
    factor: object
    b_to_separator: sp.csr_matrix
    c_from_separator: sp.csr_matrix

@dataclass(frozen=True)
class _SymbolicBlockSchurFactor:
    """Block sparse factor with an explicit separator Schur complement.

    This mirrors the analysis/factor/solve split used by sparse direct solvers
    at a bounded scale: local block factors eliminate interior unknowns, while a
    compact separator system retains source/constraint and high-connectivity
    couplings that simple block Jacobi drops.
    """

    blocks: tuple[_SymbolicSchurBlock, ...]
    separator_indices: np.ndarray
    schur_factor: object
    dtype: np.dtype
    n: int
    analysis: SparseSymbolicAnalysis
    separator_count: int
    frontal_block_count: int = 0
    total_cross_nnz: int = 0
    selected_cross_nnz: int = 0
    cross_separator_fraction: float = 1.0
    dense_rhs_entries: int = 0
    peak_dense_rhs_entries: int = 0
    separator_update_columns: int = 0
    factor_failures: int = 0
    metadata: dict[str, object] | None = None

    @property
    def overlap_size(self) -> int:
        return 0

    @property
    def coarse_size(self) -> int:
        return int(self.separator_count)

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.n),))
        sep = np.asarray(self.separator_indices, dtype=np.int64)
        sep_count = int(sep.size)
        out = np.zeros((int(self.n),), dtype=self.dtype)
        if sep_count == 0:
            for block in self.blocks:
                idx = np.asarray(block.indices, dtype=np.int64)
                try:
                    out[idx] = np.asarray(block.factor.solve(rhs_np[idx]), dtype=self.dtype)
                except Exception:
                    out[idx] = rhs_np[idx]
            return out
        schur_rhs = np.array(rhs_np[sep], dtype=self.dtype, copy=True)
        block_solutions: list[tuple[np.ndarray, np.ndarray]] = []
        for block in self.blocks:
            idx = np.asarray(block.indices, dtype=np.int64)
            rhs_i = rhs_np[idx]
            try:
                y_i = np.asarray(block.factor.solve(rhs_i), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                y_i = np.array(rhs_i, dtype=self.dtype, copy=True)
            block_solutions.append((idx, y_i))
            if block.c_from_separator.shape[0] == sep_count:
                schur_rhs -= np.asarray(block.c_from_separator @ y_i, dtype=self.dtype).reshape((sep_count,))
        try:
            y_sep = np.asarray(self.schur_factor.solve(schur_rhs), dtype=self.dtype).reshape((sep_count,))
        except Exception:
            y_sep = np.zeros((sep_count,), dtype=self.dtype)
        out[sep] = y_sep
        for block, (idx, y_i) in zip(self.blocks, block_solutions, strict=True):
            rhs_corr = np.asarray(block.b_to_separator @ y_sep, dtype=self.dtype).reshape((idx.size,))
            try:
                delta = np.asarray(block.factor.solve(rhs_corr), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                delta = np.zeros((idx.size,), dtype=self.dtype)
            out[idx] = y_i - delta
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicSuperblock:
    indices: np.ndarray
    base_blocks: tuple[int, ...]
    factor: object


@dataclass(frozen=True)
class _SymbolicSuperblockFactor:
    """Bounded sparse-direct factor over grouped symbolic blocks.

    This is a native analogue of the first useful layer in multifrontal/sparse
    direct solvers: reuse a symbolic ordering, merge strongly coupled base
    blocks subject to a hard size cap, and factor each merged block exactly.
    Cross-superblock couplings are intentionally dropped, then measured by the
    setup residual admission gate before the factor is allowed into Krylov.
    """

    blocks: tuple[_SymbolicSuperblock, ...]
    analysis: SparseSymbolicAnalysis
    dtype: np.dtype
    n: int
    max_superblock_size: int
    max_superblock_blocks: int
    retained_cross_nnz: int
    dropped_cross_nnz: int
    retained_cross_fraction: float
    factor_failures: int
    parallel_workers: int = 1
    numeric_factor_tasks: int = 0

    @property
    def superblock_count(self) -> int:
        return int(len(self.blocks))

    @property
    def base_block_count(self) -> int:
        return int(self.analysis.block_count)

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.n),))
        out = np.array(rhs_np, dtype=self.dtype, copy=True)
        for block in self.blocks:
            idx = np.asarray(block.indices, dtype=np.int64)
            if idx.size == 0:
                continue
            try:
                out[idx] = np.asarray(block.factor.solve(rhs_np[idx]), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                out[idx] = rhs_np[idx]
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


class _DisjointSet:
    def __init__(self, sizes: np.ndarray) -> None:
        self.parent = np.arange(int(sizes.size), dtype=np.int64)
        self.rows = np.asarray(sizes, dtype=np.int64).copy()
        self.blocks = np.ones((int(sizes.size),), dtype=np.int64)

    def find(self, item: int) -> int:
        idx = int(item)
        parent = self.parent
        while int(parent[idx]) != idx:
            parent[idx] = parent[int(parent[idx])]
            idx = int(parent[idx])
        return idx

    def union_if_fits(
        self,
        a: int,
        b: int,
        *,
        max_rows: int,
        max_blocks: int,
    ) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        rows = int(self.rows[root_a] + self.rows[root_b])
        blocks = int(self.blocks[root_a] + self.blocks[root_b])
        if rows > int(max_rows) or blocks > int(max_blocks):
            return False
        if self.rows[root_a] < self.rows[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.rows[root_a] = rows
        self.blocks[root_a] = blocks
        return True


def _build_symbolic_superblock_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_superblock_size: int = 32768,
    max_superblock_blocks: int = 8,
    min_cross_nnz: int = 1,
    min_retained_cross_fraction: float = 0.0,
    regularization_rel: float = 1.0e-12,
    parallel_workers: int = 1,
) -> tuple[_SymbolicSuperblockFactor, int, int]:
    """Build a bounded grouped-block sparse factor retaining dominant couplings."""

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    if n != int(matrix_csr.shape[1]):
        raise ValueError("symbolic superblock factor requires a square matrix")
    dtype = np.dtype(matrix_csr.dtype)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    base_block_count = int(analysis.block_count)
    worker_count = max(1, int(parallel_workers))
    if base_block_count <= 0:
        return (
            _SymbolicSuperblockFactor(
                blocks=tuple(),
                analysis=analysis,
                dtype=dtype,
                n=n,
                max_superblock_size=max(1, int(max_superblock_size)),
                max_superblock_blocks=max(1, int(max_superblock_blocks)),
                retained_cross_nnz=0,
                dropped_cross_nnz=0,
                retained_cross_fraction=1.0,
                factor_failures=0,
                parallel_workers=worker_count,
                numeric_factor_tasks=0,
            ),
            0,
            0,
        )

    block_sizes = np.asarray(
        [
            min(block_size, max(0, n - int(block) * block_size))
            for block in range(base_block_count)
        ],
        dtype=np.int64,
    )
    dsu = _DisjointSet(block_sizes)
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    coo = matrix_perm.tocoo()
    row_block = np.asarray(coo.row // block_size, dtype=np.int64)
    col_block = np.asarray(coo.col // block_size, dtype=np.int64)
    cross = row_block != col_block
    edge_counts: list[tuple[int, int, int]] = []
    if np.any(cross):
        lo = np.minimum(row_block[cross], col_block[cross])
        hi = np.maximum(row_block[cross], col_block[cross])
        valid = (lo >= 0) & (hi < base_block_count)
        if np.any(valid):
            pair_id = lo[valid] * int(base_block_count) + hi[valid]
            unique, counts = np.unique(pair_id, return_counts=True)
            for pair, count in zip(unique, counts, strict=True):
                c = int(count)
                if c < int(min_cross_nnz):
                    continue
                a = int(pair // int(base_block_count))
                b = int(pair % int(base_block_count))
                edge_counts.append((c, a, b))
    edge_counts.sort(key=lambda item: (-item[0], item[1], item[2]))
    max_rows = max(1, int(max_superblock_size))
    max_blocks = max(1, int(max_superblock_blocks))
    for _, a, b in edge_counts:
        dsu.union_if_fits(a, b, max_rows=max_rows, max_blocks=max_blocks)

    groups: dict[int, list[int]] = {}
    for block in range(base_block_count):
        groups.setdefault(dsu.find(block), []).append(int(block))
    ordered_groups = sorted(groups.values(), key=lambda values: (min(values), len(values)))

    retained_cross = 0
    dropped_cross = 0
    if np.any(cross):
        for rb, cb in zip(row_block[cross], col_block[cross], strict=True):
            if dsu.find(int(rb)) == dsu.find(int(cb)):
                retained_cross += 1
            else:
                dropped_cross += 1
    cross_total = int(retained_cross + dropped_cross)
    retained_fraction = 1.0 if cross_total == 0 else float(retained_cross) / float(cross_total)
    min_retained = max(0.0, min(1.0, float(min_retained_cross_fraction)))
    if retained_fraction < min_retained:
        raise RuntimeError(
            "symbolic_superblock_lu retained insufficient cross-block coupling "
            f"({retained_fraction:.6g} < {min_retained:.6g}; "
            f"retained={int(retained_cross)} dropped={int(dropped_cross)} total={int(cross_total)})"
        )

    blocks: list[_SymbolicSuperblock] = []
    total_nbytes = 0
    total_nnz = 0
    factor_failures = 0
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))

    def _factor_group(group: list[int]) -> tuple[_SymbolicSuperblock | None, int, int, int]:
        perm_positions: list[np.ndarray] = []
        for block in group:
            start = int(block) * block_size
            stop = min(n, start + block_size)
            if stop > start:
                perm_positions.append(np.arange(start, stop, dtype=np.int64))
        if not perm_positions:
            return None, 0, 0, 0
        positions = np.concatenate(perm_positions).astype(np.int64, copy=False)
        indices = np.asarray(permutation[positions], dtype=np.int64)
        local = matrix_csr[indices, :][:, indices].tocsc()
        local_failures = 0
        try:
            factor = splu(local, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            local_failures += 1
            local_reg = (local + reg * sp.eye(local.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(local_reg, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                local_failures += 1
                diagonal = np.asarray(local.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        return (
            _SymbolicSuperblock(
                indices=indices,
                base_blocks=tuple(int(v) for v in group),
                factor=factor,
            ),
            int(nbytes or 0),
            int(nnz or 0),
            int(local_failures),
        )

    if worker_count > 1 and len(ordered_groups) > 1:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(ordered_groups))) as executor:
            factor_results = list(executor.map(_factor_group, ordered_groups))
    else:
        factor_results = [_factor_group(group) for group in ordered_groups]
    for block, nbytes, nnz, failures in factor_results:
        if block is None:
            continue
        blocks.append(block)
        total_nbytes += int(nbytes)
        total_nnz += int(nnz)
        factor_failures += int(failures)

    factor = _SymbolicSuperblockFactor(
        blocks=tuple(blocks),
        analysis=analysis,
        dtype=dtype,
        n=n,
        max_superblock_size=int(max_rows),
        max_superblock_blocks=int(max_blocks),
        retained_cross_nnz=int(retained_cross),
        dropped_cross_nnz=int(dropped_cross),
        retained_cross_fraction=float(retained_fraction),
        factor_failures=int(factor_failures),
        parallel_workers=int(min(worker_count, max(1, len(ordered_groups)))),
        numeric_factor_tasks=int(len(factor_results)),
    )
    return factor, int(total_nbytes), int(total_nnz)


def _build_symbolic_frontal_schur_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_separator_cols: int = 1024,
    tail_size: int = 0,
    boundary_width: int = 1,
    high_degree_cols: int = 128,
    max_superblock_size: int = 8192,
    max_superblock_blocks: int = 8,
    min_cross_nnz: int = 1,
    min_cross_separator_fraction: float = 0.0,
    regularization_rel: float = 1.0e-12,
    max_dense_rhs_entries: int = 0,
    max_dense_rhs_cols_per_block: int = 0,
) -> tuple[_SymbolicBlockSchurFactor, int, int]:
    """Build a bounded frontal/Schur elimination over symbolic block groups.

    Unlike ``symbolic_superblock_lu``, cross-group couplings are not simply
    dropped.  Their endpoints are promoted into a bounded separator, local
    interiors are eliminated, and the separator Schur complement retains the
    global coupling that local block factors miss.
    """

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    if n != int(matrix_csr.shape[1]):
        raise ValueError("symbolic frontal Schur factor requires a square matrix")
    dtype = np.dtype(matrix_csr.dtype)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    base_block_count = int(analysis.block_count)
    max_cols = max(0, min(int(max_separator_cols), n))
    if n == 0 or base_block_count <= 0:
        empty = _SymbolicBlockSchurFactor(
            blocks=tuple(),
            separator_indices=np.asarray([], dtype=np.int64),
            schur_factor=_DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype)),
            dtype=dtype,
            n=n,
            analysis=analysis,
            separator_count=0,
        )
        return empty, 0, 0

    block_sizes = np.asarray(
        [
            min(block_size, max(0, n - int(block) * block_size))
            for block in range(base_block_count)
        ],
        dtype=np.int64,
    )
    dsu = _DisjointSet(block_sizes)
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    coo = matrix_perm.tocoo()
    row_block = np.asarray(coo.row // block_size, dtype=np.int64)
    col_block = np.asarray(coo.col // block_size, dtype=np.int64)
    cross = row_block != col_block
    edge_counts: list[tuple[int, int, int]] = []
    if np.any(cross):
        lo = np.minimum(row_block[cross], col_block[cross])
        hi = np.maximum(row_block[cross], col_block[cross])
        valid = (lo >= 0) & (hi < base_block_count)
        if np.any(valid):
            pair_id = lo[valid] * int(base_block_count) + hi[valid]
            unique, counts = np.unique(pair_id, return_counts=True)
            for pair, count in zip(unique, counts, strict=True):
                c = int(count)
                if c < int(min_cross_nnz):
                    continue
                a = int(pair // int(base_block_count))
                b = int(pair % int(base_block_count))
                edge_counts.append((c, a, b))
    edge_counts.sort(key=lambda item: (-item[0], item[1], item[2]))
    max_rows = max(1, int(max_superblock_size))
    max_blocks = max(1, int(max_superblock_blocks))
    for _, a, b in edge_counts:
        dsu.union_if_fits(a, b, max_rows=max_rows, max_blocks=max_blocks)

    unresolved_cross = np.zeros((0,), dtype=bool)
    if np.any(cross):
        unresolved_cross = np.asarray(
            [dsu.find(int(rb)) != dsu.find(int(cb)) for rb, cb in zip(row_block[cross], col_block[cross], strict=True)],
            dtype=bool,
        )
    total_cross_nnz = int(np.count_nonzero(unresolved_cross))

    candidates: dict[int, tuple[int, int]] = {}

    def _add_candidate(index: int, priority: int, score: int = 0) -> None:
        idx = int(index)
        if idx < 0 or idx >= n or len(candidates) >= max(8 * max(1, max_cols), n):
            return
        value = (int(priority), int(score))
        old = candidates.get(idx)
        if old is None or value > old:
            candidates[idx] = value

    tail = max(0, min(int(tail_size), n))
    if tail:
        for idx in range(n - tail, n):
            _add_candidate(idx, 1000, n - idx)

    if total_cross_nnz:
        cross_rows = np.asarray(coo.row[cross], dtype=np.int64)[unresolved_cross]
        cross_cols = np.asarray(coo.col[cross], dtype=np.int64)[unresolved_cross]
        endpoints = np.concatenate((permutation[cross_rows], permutation[cross_cols])).astype(np.int64, copy=False)
        counts = np.bincount(endpoints, minlength=n)
        for idx in np.flatnonzero(counts):
            _add_candidate(int(idx), 900, int(counts[int(idx)]))

    high_degree = max(0, min(int(high_degree_cols), n))
    if high_degree:
        row_degree = np.diff(matrix_csr.indptr).astype(np.int64, copy=False)
        col_degree = np.diff(matrix_csr.tocsc().indptr).astype(np.int64, copy=False)
        degree = row_degree + col_degree
        top = np.argpartition(-degree, kth=high_degree - 1)[:high_degree]
        for idx in top:
            _add_candidate(int(idx), 700, int(degree[int(idx)]))

    boundary = max(0, int(boundary_width))
    if boundary:
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            edge = min(boundary, stop - start)
            for local in range(start, start + edge):
                _add_candidate(int(permutation[local]), 500, edge - (local - start))
            for local in range(max(start, stop - edge), stop):
                _add_candidate(int(permutation[local]), 500, local - max(start, stop - edge) + 1)

    if candidates and max_cols:
        ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
        separator = np.asarray(sorted(idx for idx, _ in ordered[:max_cols]), dtype=np.int64)
    else:
        separator = np.asarray([], dtype=np.int64)
    separator_set = set(int(v) for v in separator)
    sep_count = int(separator.size)
    selected_cross = 0
    if total_cross_nnz and sep_count:
        cross_rows_orig = np.asarray(coo.row[cross], dtype=np.int64)[unresolved_cross]
        cross_cols_orig = np.asarray(coo.col[cross], dtype=np.int64)[unresolved_cross]
        row_orig = permutation[cross_rows_orig]
        col_orig = permutation[cross_cols_orig]
        selected_cross = int(
            np.count_nonzero(
                np.fromiter(
                    ((int(r) in separator_set) or (int(c) in separator_set) for r, c in zip(row_orig, col_orig, strict=True)),
                    dtype=bool,
                    count=int(total_cross_nnz),
                )
            )
        )
    cross_fraction = 1.0 if total_cross_nnz == 0 else float(selected_cross) / float(total_cross_nnz)
    min_fraction = max(0.0, min(1.0, float(min_cross_separator_fraction)))
    if cross_fraction < min_fraction:
        raise RuntimeError(
            "symbolic_frontal_schur_lu selected insufficient cross-block separator coverage "
            f"({cross_fraction:.6g} < {min_fraction:.6g}; "
            f"selected={int(selected_cross)} total={int(total_cross_nnz)} separator={int(sep_count)})"
        )

    groups: dict[int, list[int]] = {}
    for block in range(base_block_count):
        groups.setdefault(dsu.find(block), []).append(int(block))
    ordered_groups = sorted(groups.values(), key=lambda values: (min(values), len(values)))

    if sep_count:
        separator_mask = np.zeros((n,), dtype=bool)
        separator_mask[separator] = True
        dense_rhs_entries = 0
        peak_dense_rhs_entries = 0
        separator_update_columns = 0
        for group in ordered_groups:
            positions: list[np.ndarray] = []
            for block in group:
                start = int(block) * block_size
                stop = min(n, start + block_size)
                if stop <= start:
                    continue
                positions.append(np.arange(start, stop, dtype=np.int64))
            if not positions:
                continue
            idx_all = np.asarray(permutation[np.concatenate(positions)], dtype=np.int64)
            idx = idx_all[~separator_mask[idx_all]]
            if idx.size == 0:
                continue
            b_pattern = matrix_csr[idx, :][:, separator].tocsr()
            local_cols = int(np.unique(b_pattern.indices).size) if b_pattern.nnz else 0
            separator_update_columns += local_cols
            entries = int(idx.size) * int(local_cols)
            dense_rhs_entries += entries
            peak_dense_rhs_entries = max(peak_dense_rhs_entries, entries)
        if int(max_dense_rhs_entries) > 0 and dense_rhs_entries > int(max_dense_rhs_entries):
            raise RuntimeError(
                "symbolic_frontal_schur_lu dense separator RHS work budget exceeded "
                f"({int(dense_rhs_entries)}>{int(max_dense_rhs_entries)}; "
                f"peak_block_entries={int(peak_dense_rhs_entries)} separator={int(sep_count)} "
                f"groups={int(len(ordered_groups))})"
            )
    else:
        dense_rhs_entries = 0
        peak_dense_rhs_entries = 0
        separator_update_columns = 0

    schur_base_sparse = matrix_csr[separator, :][:, separator].tocsc().astype(dtype, copy=False) if sep_count else sp.csc_matrix((0, 0), dtype=dtype)
    schur = schur_base_sparse.toarray().astype(dtype, copy=False)
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))
    blocks: list[_SymbolicSchurBlock] = []
    total_nbytes = 0
    total_nnz = 0
    factor_failures = 0

    for group in ordered_groups:
        positions: list[np.ndarray] = []
        for block in group:
            start = int(block) * block_size
            stop = min(n, start + block_size)
            if stop > start:
                positions.append(np.arange(start, stop, dtype=np.int64))
        if not positions:
            continue
        idx_all = np.asarray(permutation[np.concatenate(positions)], dtype=np.int64)
        if sep_count:
            idx = np.asarray([int(v) for v in idx_all if int(v) not in separator_set], dtype=np.int64)
        else:
            idx = idx_all
        if idx.size == 0:
            continue
        local = matrix_csr[idx, :][:, idx].tocsc()
        try:
            factor = splu(local, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            factor_failures += 1
            local_reg = (local + reg * sp.eye(local.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(local_reg, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                diagonal = np.asarray(local.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += int(nbytes or 0)
        total_nnz += int(nnz or 0)
        if sep_count:
            b_mat = matrix_csr[idx, :][:, separator].tocsr().astype(dtype, copy=False)
            c_mat = matrix_csr[separator, :][:, idx].tocsr().astype(dtype, copy=False)
            if b_mat.nnz and c_mat.nnz:
                local_cols = np.unique(b_mat.indices).astype(np.int64, copy=False)
                max_cols_per_chunk = int(max_dense_rhs_cols_per_block)
                if max_cols_per_chunk <= 0:
                    max_cols_per_chunk = int(local_cols.size)
                max_cols_per_chunk = max(1, int(max_cols_per_chunk))
                for col_start in range(0, int(local_cols.size), max_cols_per_chunk):
                    col_chunk = local_cols[col_start : col_start + max_cols_per_chunk]
                    b_dense = b_mat[:, col_chunk].toarray().astype(dtype, copy=False)
                    try:
                        eliminated = np.asarray(factor.solve(b_dense), dtype=dtype)
                    except Exception:
                        eliminated = np.zeros((idx.size, int(col_chunk.size)), dtype=dtype)
                    update = np.asarray(c_mat @ eliminated, dtype=dtype)
                    schur[:, col_chunk] -= update
            total_nbytes += estimate_csr_nbytes(b_mat.shape, int(b_mat.nnz), data_dtype=b_mat.dtype, index_dtype=b_mat.indices.dtype)
            total_nbytes += estimate_csr_nbytes(c_mat.shape, int(c_mat.nnz), data_dtype=c_mat.dtype, index_dtype=c_mat.indices.dtype)
            total_nnz += int(b_mat.nnz) + int(c_mat.nnz)
        else:
            b_mat = sp.csr_matrix((idx.size, 0), dtype=dtype)
            c_mat = sp.csr_matrix((0, idx.size), dtype=dtype)
        blocks.append(_SymbolicSchurBlock(indices=idx, factor=factor, b_to_separator=b_mat, c_from_separator=c_mat))

    if sep_count:
        schur_csc = sp.csc_matrix(schur)
        schur_csc.sum_duplicates()
        schur_max = float(np.max(np.abs(schur_csc.data))) if schur_csc.nnz else 0.0
        schur_reg = max(1.0e-14, float(regularization_rel) * max(1.0, schur_max))
        schur_reg_csc = (schur_csc + schur_reg * sp.eye(sep_count, dtype=dtype, format="csc")).tocsc()
        try:
            base_schur_factor = splu(schur_reg_csc, permc_spec="COLAMD", diag_pivot_thresh=1.0)
        except RuntimeError:
            base_schur_factor = _DenseInverseFactor(inverse=np.linalg.pinv(schur_reg_csc.toarray()))
        schur_factor = base_schur_factor
        schur_nbytes, schur_nnz = estimate_superlu_factor_storage(base_schur_factor)
        if schur_nbytes is None and isinstance(base_schur_factor, _DenseInverseFactor):
            schur_nbytes = int(base_schur_factor.inverse.nbytes)
            schur_nnz = int(base_schur_factor.inverse.size)
        total_nbytes += estimate_csr_nbytes(schur_csc.shape, int(schur_csc.nnz), data_dtype=schur_csc.dtype, index_dtype=schur_csc.indices.dtype)
        total_nbytes += int(schur_nbytes or 0)
        total_nnz += int(schur_csc.nnz) + int(schur_nnz or 0)
    else:
        schur_factor = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype))

    factor = _SymbolicBlockSchurFactor(
        blocks=tuple(blocks),
        separator_indices=separator,
        schur_factor=schur_factor,
        dtype=dtype,
        n=n,
        analysis=analysis,
        separator_count=sep_count,
        frontal_block_count=len(blocks),
        total_cross_nnz=int(total_cross_nnz),
        selected_cross_nnz=int(selected_cross),
        cross_separator_fraction=float(cross_fraction),
        dense_rhs_entries=int(dense_rhs_entries),
        peak_dense_rhs_entries=int(peak_dense_rhs_entries),
        separator_update_columns=int(separator_update_columns),
        factor_failures=int(factor_failures),
    )
    return factor, int(total_nbytes), int(total_nnz)


def _factor_csc_with_regularized_fallback(
    matrix_csc: sp.csc_matrix,
    *,
    dtype: np.dtype,
    diag_pivot_thresh: float,
    regularization_rel: float,
    permc_spec: str = "COLAMD",
) -> tuple[object, int, int, int]:
    """Factor one sparse frontal matrix with a bounded diagonal fallback."""

    local = matrix_csc.astype(dtype, copy=False).tocsc()
    failures = 0
    try:
        factor = splu(local, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot_thresh))
    except RuntimeError:
        failures += 1
        max_abs = float(np.max(np.abs(local.data))) if local.nnz else 0.0
        reg = 0.0 if float(regularization_rel) <= 0.0 else max(
            1.0e-14,
            float(regularization_rel) * max(1.0, max_abs),
        )
        if reg > 0.0:
            try:
                local_reg = (local + reg * sp.eye(local.shape[0], dtype=dtype, format="csc")).tocsc()
                factor = splu(local_reg, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                failures += 1
                factor = None
        else:
            factor = None
        if factor is None:
            diagonal = np.asarray(local.diagonal(), dtype=np.float64)
            scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
            floor = max(1.0e-14, abs(float(regularization_rel)) * scale)
            sign = np.where(diagonal < 0.0, -1.0, 1.0)
            diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
            factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
    nbytes, nnz = estimate_superlu_factor_storage(factor)
    if nbytes is None and isinstance(factor, _JacobiFactor):
        nbytes = int(factor.inverse_diagonal.nbytes)
        nnz = int(factor.inverse_diagonal.size)
    if nbytes is None and isinstance(factor, _DenseInverseFactor):
        nbytes = int(factor.inverse.nbytes)
        nnz = int(factor.inverse.size)
    return factor, int(nbytes or 0), int(nnz or 0), int(failures)


def wrap_sparse_factor_with_coarse_correction(
    factor_bundle: SparseFactorBundle,
    coarse_basis: sp.spmatrix,
    *,
    damping: float = 1.0,
    regularization_rel: float = 1.0e-10,
) -> SparseFactorBundle:
    """Return a factor bundle with a supplied sparse Galerkin residual correction."""

    matrix = factor_bundle.operator.matrix
    if matrix is None:
        return factor_bundle
    matrix_csr = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))
    basis = coarse_basis.tocsr() if sp.issparse(coarse_basis) else sp.csr_matrix(np.asarray(coarse_basis))
    if basis.shape[0] != matrix_csr.shape[0] or basis.shape[1] == 0:
        return factor_bundle
    basis.sum_duplicates()
    basis.eliminate_zeros()
    coarse_matrix = (basis.T @ matrix_csr @ basis).tocsc()
    coarse_matrix.sum_duplicates()
    max_abs = float(np.max(np.abs(coarse_matrix.data))) if coarse_matrix.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))
    coarse_matrix_reg = (coarse_matrix + reg * sp.eye(coarse_matrix.shape[0], dtype=matrix_csr.dtype, format="csc")).tocsc()
    try:
        coarse_factor = splu(coarse_matrix_reg, permc_spec="COLAMD", diag_pivot_thresh=1.0)
    except RuntimeError:
        coarse_factor = _DenseInverseFactor(inverse=np.linalg.pinv(coarse_matrix_reg.toarray()))
    coarse_nbytes, coarse_nnz = estimate_superlu_factor_storage(coarse_factor)
    if coarse_nbytes is None and isinstance(coarse_factor, _DenseInverseFactor):
        coarse_nbytes = int(coarse_factor.inverse.nbytes)
        coarse_nnz = int(coarse_factor.inverse.size)
    basis_nbytes = estimate_csr_nbytes(
        basis.shape,
        int(basis.nnz),
        data_dtype=basis.dtype,
        index_dtype=basis.indices.dtype,
    )
    coarse_matrix_nbytes = estimate_csr_nbytes(
        coarse_matrix.shape,
        int(coarse_matrix.nnz),
        data_dtype=coarse_matrix.dtype,
        index_dtype=coarse_matrix.indices.dtype,
    )
    wrapper = _SparseCoarseCorrectionFactor(
        base_factor=factor_bundle.factor,
        matrix=matrix_csr,
        coarse_basis=basis,
        coarse_factor=coarse_factor,
        dtype=np.dtype(matrix_csr.dtype),
        damping=float(damping),
    )
    nbytes = None if factor_bundle.factor_nbytes_estimate is None else int(factor_bundle.factor_nbytes_estimate)
    nnz = None if factor_bundle.factor_nnz_estimate is None else int(factor_bundle.factor_nnz_estimate)
    total_nbytes = None if nbytes is None else int(nbytes + basis_nbytes + coarse_matrix_nbytes + int(coarse_nbytes or 0))
    total_nnz = None if nnz is None else int(nnz + basis.nnz + coarse_matrix.nnz + int(coarse_nnz or 0))
    return SparseFactorBundle(
        factor=wrapper,
        operator=factor_bundle.operator,
        metadata=factor_bundle.metadata,
        kind=factor_bundle.kind,
        factor_nbytes_estimate=total_nbytes,
        factor_nnz_estimate=total_nnz,
        factor_s=factor_bundle.factor_s,
    )


def _build_symbolic_block_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    overlap_size: int = 0,
    regularization_rel: float = 1.0e-12,
) -> tuple[_SymbolicBlockFactor, int, int]:
    matrix_csr = matrix.tocsr()
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    inverse_permutation = np.asarray(analysis.inverse_permutation, dtype=np.int64)
    if permutation.size != matrix_csr.shape[0]:
        raise ValueError("symbolic permutation size does not match matrix")
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    blocks: list[tuple[int, int, int, int, object]] = []
    total_nbytes = 0
    total_nnz = 0
    dtype = np.dtype(matrix_perm.dtype)
    block_size = max(1, int(analysis.block_size_target))
    overlap = max(0, int(overlap_size))
    n = int(matrix_perm.shape[0])
    for start in range(0, int(matrix_perm.shape[0]), block_size):
        stop = min(int(matrix_perm.shape[0]), start + block_size)
        local_start = max(0, int(start) - overlap)
        local_stop = min(n, int(stop) + overlap)
        block = matrix_perm[local_start:local_stop, local_start:local_stop].tocsc()
        if block.shape[0] == 0:
            continue
        try:
            factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            diagonal = np.asarray(block.diagonal(), dtype=np.float64)
            scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
            floor = max(1.0e-14, float(regularization_rel) * scale)
            sign = np.where(diagonal < 0.0, -1.0, 1.0)
            diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
            factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += 0 if nbytes is None else int(nbytes)
        total_nnz += 0 if nnz is None else int(nnz)
        blocks.append((int(start), int(stop), int(local_start), int(local_stop), factor))
    factor = _SymbolicBlockFactor(
        blocks=tuple(blocks),
        analysis=analysis,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        dtype=dtype,
        overlap_size=overlap,
    )
    return factor, int(total_nbytes), int(total_nnz)


def _build_symbolic_block_coarse_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    overlap_size: int = 0,
    coarse_max_cols: int = 256,
    coarse_probe_cols: int = 4,
    coarse_damping: float = 1.0,
    coarse_regularization_rel: float = 1.0e-10,
) -> tuple[_SymbolicBlockCoarseFactor, int, int]:
    """Build a bounded local block factor with block-indicator Galerkin coarse solve."""

    matrix_csr = matrix.tocsr()
    local_factor, local_nbytes, local_nnz = _build_symbolic_block_factor(
        matrix_csr,
        analysis=analysis,
        diag_pivot_thresh=float(diag_pivot_thresh),
        overlap_size=int(overlap_size),
    )
    n = int(matrix_csr.shape[0])
    if n == 0 or not local_factor.blocks:
        basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)
        coarse_factor: object = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=matrix_csr.dtype))
        return (
            _SymbolicBlockCoarseFactor(
                local_factor=local_factor,
                matrix=matrix_csr,
                coarse_basis=basis,
                coarse_factor=coarse_factor,
                coarse_matrix_nnz=0,
                dtype=np.dtype(matrix_csr.dtype),
                coarse_damping=float(coarse_damping),
            ),
            int(local_nbytes),
            int(local_nnz),
        )

    max_cols = max(1, int(coarse_max_cols))
    block_count = len(local_factor.blocks)
    coarse_cols = min(block_count, max_cols)
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    group_edges = np.linspace(0, block_count, coarse_cols + 1, dtype=np.int64)
    for coarse_col in range(coarse_cols):
        block_start = int(group_edges[coarse_col])
        block_stop = int(group_edges[coarse_col + 1])
        if block_stop <= block_start:
            continue
        owned_indices: list[np.ndarray] = []
        for block_index in range(block_start, block_stop):
            center_start, center_stop, _, _, _ = local_factor.blocks[block_index]
            owned_indices.append(permutation[int(center_start) : int(center_stop)])
        if not owned_indices:
            continue
        rows = np.concatenate(owned_indices).astype(np.int64, copy=False)
        if rows.size == 0:
            continue
        rows_parts.append(rows)
        cols_parts.append(np.full(rows.shape, int(coarse_col), dtype=np.int64))
        data_parts.append(np.full(rows.shape, 1.0 / np.sqrt(float(rows.size)), dtype=matrix_csr.dtype))
    if rows_parts:
        block_basis = sp.coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
            shape=(n, coarse_cols),
            dtype=matrix_csr.dtype,
        ).tocsr()
    else:
        block_basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)

    basis_parts: list[sp.csr_matrix] = [block_basis] if block_basis.shape[1] else []
    probe_cols = max(0, int(coarse_probe_cols))
    if probe_cols:
        probes = deterministic_sparse_probe_matrix(n, count=probe_cols, dtype=matrix_csr.dtype)
        residual_cols: list[np.ndarray] = []
        for col in range(probes.shape[1]):
            probe = np.asarray(probes[:, col], dtype=matrix_csr.dtype)
            local_sol = np.asarray(local_factor.solve(probe), dtype=matrix_csr.dtype)
            residual = probe - np.asarray(matrix_csr @ local_sol, dtype=matrix_csr.dtype)
            if basis_parts:
                # Remove the already represented block-average component so
                # residual modes carry only the missing fine/global coupling.
                block_projection = block_basis @ np.asarray(block_basis.T @ residual, dtype=matrix_csr.dtype)
                residual = residual - np.asarray(block_projection, dtype=matrix_csr.dtype)
            for prev in residual_cols:
                residual = residual - prev * float(np.dot(prev, residual))
            norm = float(np.linalg.norm(residual.astype(np.float64, copy=False)))
            if np.isfinite(norm) and norm > 1.0e-14:
                residual_cols.append(np.asarray(residual / norm, dtype=matrix_csr.dtype))
        if residual_cols:
            residual_basis = sp.csr_matrix(np.stack(residual_cols, axis=1))
            basis_parts.append(residual_basis)

    if basis_parts:
        basis = sp.hstack(basis_parts, format="csr")
    else:
        basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)

    coarse_matrix = (basis.T @ matrix_csr @ basis).tocsc()
    coarse_matrix.sum_duplicates()
    max_abs = float(np.max(np.abs(coarse_matrix.data))) if coarse_matrix.nnz else 0.0
    reg = max(1.0e-14, float(coarse_regularization_rel) * max(1.0, max_abs))
    coarse_matrix_reg = (coarse_matrix + reg * sp.eye(coarse_matrix.shape[0], dtype=matrix_csr.dtype, format="csc")).tocsc()
    try:
        coarse_factor = splu(coarse_matrix_reg, permc_spec="COLAMD", diag_pivot_thresh=1.0)
    except RuntimeError:
        coarse_factor = _DenseInverseFactor(inverse=np.linalg.pinv(coarse_matrix_reg.toarray()))
    coarse_nbytes, coarse_nnz = estimate_superlu_factor_storage(coarse_factor)
    if coarse_nbytes is None and isinstance(coarse_factor, _DenseInverseFactor):
        coarse_nbytes = int(coarse_factor.inverse.nbytes)
        coarse_nnz = int(coarse_factor.inverse.size)
    basis_nbytes = estimate_csr_nbytes(
        basis.shape,
        int(basis.nnz),
        data_dtype=basis.dtype,
        index_dtype=basis.indices.dtype,
    )
    coarse_matrix_nbytes = estimate_csr_nbytes(
        coarse_matrix.shape,
        int(coarse_matrix.nnz),
        data_dtype=coarse_matrix.dtype,
        index_dtype=coarse_matrix.indices.dtype,
    )
    factor = _SymbolicBlockCoarseFactor(
        local_factor=local_factor,
        matrix=matrix_csr,
        coarse_basis=basis,
        coarse_factor=coarse_factor,
        coarse_matrix_nnz=int(coarse_matrix.nnz),
        dtype=np.dtype(matrix_csr.dtype),
        coarse_damping=float(coarse_damping),
    )
    total_nbytes = int(local_nbytes) + int(basis_nbytes) + int(coarse_matrix_nbytes) + int(coarse_nbytes or 0)
    total_nnz = int(local_nnz) + int(basis.nnz) + int(coarse_matrix.nnz) + int(coarse_nnz or 0)
    return factor, int(total_nbytes), int(total_nnz)


def _select_symbolic_schur_separator(
    matrix: sp.csr_matrix,
    *,
    analysis: SparseSymbolicAnalysis,
    max_separator_cols: int,
    tail_size: int,
    boundary_width: int,
    high_degree_cols: int,
) -> np.ndarray:
    n = int(matrix.shape[0])
    max_cols = max(0, min(int(max_separator_cols), n))
    if n == 0 or max_cols == 0:
        return np.asarray([], dtype=np.int64)
    candidates: dict[int, tuple[int, int]] = {}

    def _add(index: int, priority: int, score: int = 0) -> None:
        idx = int(index)
        if idx < 0 or idx >= n:
            return
        old = candidates.get(idx)
        value = (int(priority), int(score))
        if old is None or value > old:
            candidates[idx] = value

    tail = max(0, min(int(tail_size), n))
    if tail:
        for idx in range(n - tail, n):
            _add(idx, 100, n - idx)

    boundary = max(0, int(boundary_width))
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    inverse_permutation = np.asarray(analysis.inverse_permutation, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    if permutation.size == n and boundary:
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            edge = min(boundary, stop - start)
            for local in range(start, start + edge):
                _add(int(permutation[local]), 40, edge - (local - start))
            for local in range(max(start, stop - edge), stop):
                _add(int(permutation[local]), 40, local - max(start, stop - edge) + 1)

    if permutation.size == n and inverse_permutation.size == n and block_size < n:
        coo = matrix.tocoo()
        row_pos = inverse_permutation[np.asarray(coo.row, dtype=np.int64)]
        col_pos = inverse_permutation[np.asarray(coo.col, dtype=np.int64)]
        row_block = row_pos // block_size
        col_block = col_pos // block_size
        cross = row_block != col_block
        if np.any(cross):
            endpoints = np.concatenate(
                [
                    np.asarray(coo.row[cross], dtype=np.int64),
                    np.asarray(coo.col[cross], dtype=np.int64),
                ]
            )
            counts = np.bincount(endpoints, minlength=n)
            cross_indices = np.flatnonzero(counts)
            for idx in cross_indices:
                _add(int(idx), 80, int(counts[int(idx)]))

    high_degree = max(0, int(high_degree_cols))
    if high_degree:
        row_degree = np.diff(matrix.indptr).astype(np.int64, copy=False)
        col_degree = np.diff(matrix.tocsc().indptr).astype(np.int64, copy=False)
        degree = row_degree + col_degree
        count = min(high_degree, n)
        if count:
            top = np.argpartition(-degree, kth=count - 1)[:count]
            for idx in top:
                _add(int(idx), 60, int(degree[int(idx)]))

    if not candidates:
        return np.asarray([], dtype=np.int64)
    ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
    selected = [idx for idx, _ in ordered[:max_cols]]
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


def _build_symbolic_block_schur_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_separator_cols: int = 256,
    tail_size: int = 0,
    boundary_width: int = 1,
    high_degree_cols: int = 64,
    regularization_rel: float = 1.0e-12,
) -> tuple[_SymbolicBlockSchurFactor, int, int]:
    """Build a bounded block-elimination factor with an explicit separator Schur solve."""

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    dtype = np.dtype(matrix_csr.dtype)
    separator = _select_symbolic_schur_separator(
        matrix_csr,
        analysis=analysis,
        max_separator_cols=int(max_separator_cols),
        tail_size=int(tail_size),
        boundary_width=int(boundary_width),
        high_degree_cols=int(high_degree_cols),
    )
    separator_set = set(int(v) for v in separator)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    interior_order = np.asarray([int(v) for v in permutation if int(v) not in separator_set], dtype=np.int64)
    sep_count = int(separator.size)
    block_size = max(1, int(analysis.block_size_target))
    blocks: list[_SymbolicSchurBlock] = []
    total_nbytes = 0
    total_nnz = 0
    schur = matrix_csr[separator, :][:, separator].toarray().astype(dtype, copy=False) if sep_count else np.zeros((0, 0), dtype=dtype)
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))

    for start in range(0, int(interior_order.size), block_size):
        idx = np.asarray(interior_order[start : start + block_size], dtype=np.int64)
        if idx.size == 0:
            continue
        block = matrix_csr[idx, :][:, idx].tocsc()
        if block.shape[0] == 0:
            continue
        try:
            factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            block = (block + reg * sp.eye(block.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                diagonal = np.asarray(block.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += int(nbytes or 0)
        total_nnz += int(nnz or 0)
        if sep_count:
            b_mat = matrix_csr[idx, :][:, separator].tocsr().astype(dtype, copy=False)
            c_mat = matrix_csr[separator, :][:, idx].tocsr().astype(dtype, copy=False)
            if b_mat.nnz:
                b_dense = b_mat.toarray().astype(dtype, copy=False)
                try:
                    eliminated = np.asarray(factor.solve(b_dense), dtype=dtype)
                except Exception:
                    eliminated = np.zeros((idx.size, sep_count), dtype=dtype)
                if c_mat.nnz:
                    schur -= np.asarray(c_mat @ eliminated, dtype=dtype)
            total_nbytes += estimate_csr_nbytes(b_mat.shape, int(b_mat.nnz), data_dtype=b_mat.dtype, index_dtype=b_mat.indices.dtype)
            total_nbytes += estimate_csr_nbytes(c_mat.shape, int(c_mat.nnz), data_dtype=c_mat.dtype, index_dtype=c_mat.indices.dtype)
            total_nnz += int(b_mat.nnz) + int(c_mat.nnz)
        else:
            b_mat = sp.csr_matrix((idx.size, 0), dtype=dtype)
            c_mat = sp.csr_matrix((0, idx.size), dtype=dtype)
        blocks.append(_SymbolicSchurBlock(indices=idx, factor=factor, b_to_separator=b_mat, c_from_separator=c_mat))

    if sep_count:
        schur_csc = sp.csc_matrix(schur)
        schur_csc.sum_duplicates()
        schur_max = float(np.max(np.abs(schur_csc.data))) if schur_csc.nnz else 0.0
        schur_reg = max(1.0e-14, float(regularization_rel) * max(1.0, schur_max))
        schur_reg_csc = (schur_csc + schur_reg * sp.eye(sep_count, dtype=dtype, format="csc")).tocsc()
        try:
            schur_factor = splu(schur_reg_csc, permc_spec="COLAMD", diag_pivot_thresh=1.0)
        except RuntimeError:
            schur_factor = _DenseInverseFactor(inverse=np.linalg.pinv(schur_reg_csc.toarray()))
        schur_nbytes, schur_nnz = estimate_superlu_factor_storage(schur_factor)
        if schur_nbytes is None and isinstance(schur_factor, _DenseInverseFactor):
            schur_nbytes = int(schur_factor.inverse.nbytes)
            schur_nnz = int(schur_factor.inverse.size)
        total_nbytes += estimate_csr_nbytes(schur_csc.shape, int(schur_csc.nnz), data_dtype=schur_csc.dtype, index_dtype=schur_csc.indices.dtype)
        total_nbytes += int(schur_nbytes or 0)
        total_nnz += int(schur_csc.nnz) + int(schur_nnz or 0)
    else:
        schur_factor = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype))

    factor = _SymbolicBlockSchurFactor(
        blocks=tuple(blocks),
        separator_indices=separator,
        schur_factor=schur_factor,
        dtype=dtype,
        n=n,
        analysis=analysis,
        separator_count=sep_count,
    )
    return factor, int(total_nbytes), int(total_nnz)


def _hash_int_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        arr_np = np.ascontiguousarray(arr)
        digest.update(str(arr_np.dtype).encode("ascii"))
        digest.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
        digest.update(arr_np.tobytes())
    return digest.hexdigest()


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float64, copy=False), float(percentile)))


def deterministic_sparse_probe_matrix(
    size: int,
    *,
    count: int = 4,
    dtype=np.float64,
) -> np.ndarray:
    """Return deterministic probe RHS columns for sparse factor admission."""

    n = max(0, int(size))
    probe_count = max(1, int(count))
    dtype_np = np.dtype(dtype)
    if n == 0:
        return np.zeros((0, probe_count), dtype=dtype_np)
    probes: list[np.ndarray] = []
    probes.append(np.ones((n,), dtype=np.float64))
    probes.append(np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float64))
    if n == 1:
        probes.append(np.ones((1,), dtype=np.float64))
    else:
        probes.append(np.linspace(-1.0, 1.0, n, dtype=np.float64))
    rng = np.random.default_rng(8675309)
    while len(probes) < probe_count:
        probes.append(rng.standard_normal(n).astype(np.float64))
    normalized: list[np.ndarray] = []
    for probe in probes[:probe_count]:
        norm = float(np.linalg.norm(probe))
        if norm > 0.0 and np.isfinite(norm):
            probe = probe / norm
        normalized.append(np.asarray(probe, dtype=dtype_np))
    return np.stack(normalized, axis=1)


def admit_sparse_factor_against_operator(
    operator: SparseOperatorBundle | np.ndarray | sp.spmatrix,
    factor: SparseFactorBundle | object,
    *,
    probes: np.ndarray | None = None,
    probe_count: int = 4,
    max_relative_residual: float = 1.0e-2,
    min_improvement_vs_identity: float = 10.0,
) -> SparseFactorAdmission:
    """Accept an approximate factor only if it reduces the true setup residual.

    This is a cheap PETSc-style setup safeguard for native Python/JAX sparse
    preconditioners: before a candidate enters Krylov, apply it to deterministic
    RHS probes and measure ``||P M^{-1} b - b|| / ||b||`` against the actual
    materialized preconditioner matrix ``P``.  Weak block factors are rejected
    before they can amplify the nonlinear solve residual.
    """

    if isinstance(operator, SparseOperatorBundle):
        matrix = operator.matrix
    else:
        matrix = operator
    if matrix is None:
        return SparseFactorAdmission(
            accepted=False,
            max_relative_residual=float("inf"),
            median_relative_residual=float("inf"),
            min_improvement_vs_identity=0.0,
            probe_count=0,
            reason="missing_operator_matrix",
        )
    if not sp.issparse(matrix):
        matrix_csr = sp.csr_matrix(np.asarray(matrix))
    else:
        matrix_csr = matrix.tocsr()
    n_rows, n_cols = int(matrix_csr.shape[0]), int(matrix_csr.shape[1])
    if n_rows != n_cols:
        return SparseFactorAdmission(
            accepted=False,
            max_relative_residual=float("inf"),
            median_relative_residual=float("inf"),
            min_improvement_vs_identity=0.0,
            probe_count=0,
            reason="operator_not_square",
        )
    if probes is None:
        probes_use = deterministic_sparse_probe_matrix(n_rows, count=int(probe_count), dtype=matrix_csr.dtype)
    else:
        probes_use = np.asarray(probes, dtype=matrix_csr.dtype)
        if probes_use.ndim == 1:
            probes_use = probes_use.reshape((n_rows, 1))
    if probes_use.shape[0] != n_rows:
        raise ValueError(f"probe rows {probes_use.shape[0]} do not match operator size {n_rows}")

    if isinstance(factor, SparseFactorBundle):
        solve = factor.solve
    else:
        solve = factor.solve

    tiny = np.finfo(np.float64).tiny
    rel_residuals: list[float] = []
    improvements: list[float] = []
    for col in range(int(probes_use.shape[1])):
        rhs = np.asarray(probes_use[:, col], dtype=matrix_csr.dtype).reshape((n_rows,))
        rhs_norm = max(tiny, float(np.linalg.norm(rhs.astype(np.float64, copy=False))))
        try:
            y = np.asarray(solve(rhs), dtype=matrix_csr.dtype).reshape((n_rows,))
            residual = np.asarray(matrix_csr @ y - rhs, dtype=np.float64)
            identity_residual = np.asarray(matrix_csr @ rhs - rhs, dtype=np.float64)
            rel = float(np.linalg.norm(residual) / rhs_norm)
            identity_rel = float(np.linalg.norm(identity_residual) / rhs_norm)
        except Exception:
            rel = float("inf")
            identity_rel = 0.0
        if not np.isfinite(rel):
            rel = float("inf")
        rel_residuals.append(float(rel))
        if rel <= tiny:
            improvement = float("inf") if identity_rel > tiny else 1.0
        else:
            improvement = float(identity_rel / rel) if np.isfinite(identity_rel) else 0.0
        if not np.isfinite(improvement) and improvement != float("inf"):
            improvement = 0.0
        improvements.append(float(improvement))

    residuals_np = np.asarray(rel_residuals, dtype=np.float64)
    improvements_np = np.asarray(improvements, dtype=np.float64)
    max_rel = float(np.max(residuals_np)) if residuals_np.size else float("inf")
    median_rel = float(np.median(residuals_np)) if residuals_np.size else float("inf")
    min_improvement = float(np.min(improvements_np)) if improvements_np.size else 0.0
    accepted = bool(
        np.isfinite(max_rel)
        and max_rel <= float(max_relative_residual)
        and min_improvement >= float(min_improvement_vs_identity)
    )
    reason = "accepted" if accepted else "residual_or_improvement_gate_failed"
    return SparseFactorAdmission(
        accepted=accepted,
        max_relative_residual=max_rel,
        median_relative_residual=median_rel,
        min_improvement_vs_identity=min_improvement,
        probe_count=int(probes_use.shape[1]),
        reason=reason,
    )


def _bandwidth_and_profile(pattern_csr: sp.csr_matrix) -> tuple[int, int]:
    if pattern_csr.nnz == 0:
        return 0, 0
    rows = np.repeat(np.arange(pattern_csr.shape[0], dtype=np.int64), np.diff(pattern_csr.indptr))
    cols = pattern_csr.indices.astype(np.int64, copy=False)
    bandwidth = int(np.max(np.abs(rows - cols))) if rows.size else 0
    profile = 0
    for row in range(pattern_csr.shape[0]):
        start, end = int(pattern_csr.indptr[row]), int(pattern_csr.indptr[row + 1])
        if start == end:
            continue
        min_col = int(np.min(pattern_csr.indices[start:end]))
        if min_col < row:
            profile += int(row - min_col)
    return bandwidth, int(profile)


def _nested_dissection_like_permutation(
    pattern_csr: sp.csr_matrix,
    *,
    leaf_size: int,
    max_depth: int,
) -> np.ndarray:
    """Return a bounded graph-bisection ordering for native sparse factors.

    MUMPS/SuperLU_DIST get most of their production-grid robustness from a
    symbolic analysis phase driven by graph orderings such as SCOTCH,
    PT-SCOTCH, ParMETIS, METIS, or RCM.  SFINCS-JAX cannot depend on those
    packages in the default install, so this helper provides the deterministic
    native fallback used by the symbolic factors: recursively order the local
    graph with RCM, split it, promote cross-edge endpoints into a separator,
    and emit ``left, separator, right`` so the current frontal-Schur builder
    sees the separator near the middle of each active interval.
    """

    n = int(pattern_csr.shape[0])
    if n <= 0:
        return np.arange(0, dtype=np.int64)
    graph = (pattern_csr + pattern_csr.T).astype(np.int8, copy=False).tocsr()
    graph.sum_duplicates()
    if graph.nnz:
        graph.data = np.ones_like(graph.data, dtype=np.int8)
    graph.setdiag(0)
    graph.eliminate_zeros()
    leaf = max(2, int(leaf_size))
    depth_cap = max(0, int(max_depth))
    try:
        from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415
    except Exception:
        reverse_cuthill_mckee = None  # type: ignore[assignment]

    def _rcm_local(nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes, dtype=np.int64).reshape((-1,))
        if nodes.size <= 2 or reverse_cuthill_mckee is None:
            return nodes
        try:
            local = graph[nodes, :][:, nodes]
            order = np.asarray(reverse_cuthill_mckee(local, symmetric_mode=True), dtype=np.int64)
            if order.size == nodes.size and np.unique(order).size == nodes.size:
                return nodes[order]
        except Exception:
            pass
        return nodes

    def _order(nodes: np.ndarray, depth: int) -> np.ndarray:
        nodes = _rcm_local(nodes)
        node_count = int(nodes.size)
        if node_count <= leaf or int(depth) >= depth_cap:
            return nodes
        midpoint = node_count // 2
        left = nodes[:midpoint]
        right = nodes[midpoint:]
        if left.size == 0 or right.size == 0:
            return nodes
        local = graph[nodes, :][:, nodes].tocoo()
        side = np.zeros((node_count,), dtype=np.int8)
        side[:midpoint] = 1
        side[midpoint:] = 2
        row_side = side[np.asarray(local.row, dtype=np.int64)]
        col_side = side[np.asarray(local.col, dtype=np.int64)]
        cross = (row_side > 0) & (col_side > 0) & (row_side != col_side)
        if not np.any(cross):
            return np.concatenate([_order(left, depth + 1), _order(right, depth + 1)])
        separator_local = np.unique(
            np.concatenate(
                [
                    np.asarray(local.row[cross], dtype=np.int64),
                    np.asarray(local.col[cross], dtype=np.int64),
                ]
            )
        )
        separator_mask = np.zeros((node_count,), dtype=bool)
        separator_mask[separator_local] = True
        left_keep = nodes[(np.arange(node_count) < midpoint) & (~separator_mask)]
        right_keep = nodes[(np.arange(node_count) >= midpoint) & (~separator_mask)]
        separator = nodes[separator_mask]
        if left_keep.size == 0 or right_keep.size == 0 or separator.size >= node_count:
            return nodes
        return np.concatenate(
            [
                _order(left_keep, depth + 1),
                separator,
                _order(right_keep, depth + 1),
            ]
        )

    perm = _order(np.arange(n, dtype=np.int64), 0)
    if perm.size != n or np.unique(perm).size != n:
        return np.arange(n, dtype=np.int64)
    return np.asarray(perm, dtype=np.int64)


def analyze_sparse_symbolic_structure(
    matrix: np.ndarray | sp.spmatrix,
    *,
    ordering_kind: str = "rcm",
    block_size_target: int = 4096,
    max_permutation_size: int = 250_000,
) -> SparseSymbolicAnalysis:
    """Analyze a sparse pattern for reusable symbolic factor planning.

    The returned metadata is intentionally independent of any numerical factor.
    It can be used as a cache/admission key for Python/JAX-native sparse
    preconditioners and for comparing how close SFINCS-JAX is to
    PETSc/MUMPS/SuperLU-style symbolic reuse.
    """

    if sp.issparse(matrix):
        pattern = matrix.tocsr(copy=True)
    else:
        pattern = sp.csr_matrix(np.asarray(matrix))
    pattern.sum_duplicates()
    if pattern.nnz:
        pattern.data = np.ones_like(pattern.data, dtype=np.int8)
    pattern = pattern.astype(np.int8, copy=False)
    pattern.eliminate_zeros()
    n_rows, n_cols = int(pattern.shape[0]), int(pattern.shape[1])
    if n_rows != n_cols:
        raise ValueError(f"symbolic analysis requires a square matrix, got {pattern.shape}")

    row_counts = np.diff(pattern.indptr).astype(np.int64, copy=False)
    col_counts = np.diff(pattern.tocsc().indptr).astype(np.int64, copy=False)
    diagonal = pattern.diagonal()
    diagonal_present = int(np.count_nonzero(diagonal))
    diagonal_missing = int(n_rows - diagonal_present)
    bandwidth, profile = _bandwidth_and_profile(pattern)
    pattern_hash = _hash_int_arrays(
        np.asarray(pattern.shape, dtype=np.int64),
        pattern.indptr.astype(np.int64, copy=False),
        pattern.indices.astype(np.int64, copy=False),
    )

    ordering_norm = str(ordering_kind).strip().lower()
    permutation: np.ndarray | None
    if ordering_norm in {"natural", "none", "identity"} or n_rows == 0:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)
    elif ordering_norm in {"rcm", "reverse_cuthill_mckee", "reverse-cuthill-mckee"} and n_rows <= int(max_permutation_size):
        try:
            from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415

            permutation = np.asarray(reverse_cuthill_mckee(pattern, symmetric_mode=False), dtype=np.int64)
            ordering_norm = "rcm"
        except Exception:
            ordering_norm = "natural"
            permutation = np.arange(n_rows, dtype=np.int64)
    elif ordering_norm in {
        "nd",
        "nested_dissection",
        "nested-dissection",
        "mumps",
        "mumps_like",
        "mumps-like",
        "scotch",
        "ptscotch",
        "pt-scotch",
        "parmetis",
        "metis",
    } and n_rows <= int(max_permutation_size):
        permutation = _nested_dissection_like_permutation(
            pattern,
            leaf_size=max(2, int(block_size_target)),
            max_depth=max(1, int(np.ceil(np.log2(max(2, n_rows // max(2, int(block_size_target)))))) + 1),
        )
        ordering_norm = "nested_dissection"
    else:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)

    if permutation.size != n_rows or np.unique(permutation).size != n_rows:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(n_rows, dtype=np.int64)
    ordering_hash = _hash_int_arrays(permutation.astype(np.int64, copy=False))
    pattern_perm = pattern[permutation, :][:, permutation].tocsr()
    permuted_bandwidth, permuted_profile = _bandwidth_and_profile(pattern_perm)

    block_size_target = max(1, int(block_size_target))
    block_count = int((n_rows + block_size_target - 1) // block_size_target) if n_rows else 0
    block_size_max = 0
    block_nnz_max = 0
    for start in range(0, n_rows, block_size_target):
        stop = min(n_rows, start + block_size_target)
        block_size_max = max(block_size_max, int(stop - start))
        block_nnz_max = max(block_nnz_max, int(pattern_perm[start:stop, start:stop].nnz))

    return SparseSymbolicAnalysis(
        shape=(n_rows, n_cols),
        nnz=int(pattern.nnz),
        pattern_hash=pattern_hash,
        ordering_kind=ordering_norm,
        ordering_hash=ordering_hash,
        bandwidth=int(bandwidth),
        profile=int(profile),
        permuted_bandwidth=int(permuted_bandwidth),
        permuted_profile=int(permuted_profile),
        diagonal_present=int(diagonal_present),
        diagonal_missing=int(diagonal_missing),
        row_nnz_min=int(np.min(row_counts)) if row_counts.size else 0,
        row_nnz_max=int(np.max(row_counts)) if row_counts.size else 0,
        row_nnz_mean=float(np.mean(row_counts)) if row_counts.size else 0.0,
        row_nnz_p95=_safe_percentile(row_counts, 95.0),
        col_nnz_min=int(np.min(col_counts)) if col_counts.size else 0,
        col_nnz_max=int(np.max(col_counts)) if col_counts.size else 0,
        col_nnz_mean=float(np.mean(col_counts)) if col_counts.size else 0.0,
        col_nnz_p95=_safe_percentile(col_counts, 95.0),
        block_size_target=int(block_size_target),
        block_count=int(block_count),
        block_size_max=int(block_size_max),
        block_nnz_max=int(block_nnz_max),
        permutation=permutation,
        inverse_permutation=inverse_permutation,
    )


def estimate_superlu_factor_storage(factor: object) -> tuple[int | None, int | None]:
    """Estimate SuperLU/ILU storage from materialized ``L`` and ``U`` factors."""

    total_nbytes = 0
    total_nnz = 0
    saw_factor = False
    for name in ("L", "U"):
        matrix = getattr(factor, name, None)
        if matrix is None:
            continue
        if sp.issparse(matrix):
            matrix_csr = matrix.tocsr()
            total_nbytes += estimate_csr_nbytes(
                tuple(matrix_csr.shape),
                int(matrix_csr.nnz),
                data_dtype=matrix_csr.dtype,
                index_dtype=matrix_csr.indices.dtype,
            )
            total_nnz += int(matrix_csr.nnz)
            saw_factor = True
            continue
        arr = np.asarray(matrix)
        if arr.size:
            total_nbytes += int(arr.nbytes)
            total_nnz += int(np.count_nonzero(arr))
            saw_factor = True
    if not saw_factor:
        return None, None
    return int(total_nbytes), int(total_nnz)


def choose_storage_kind(
    *,
    shape: tuple[int, int],
    nnz_estimate: int | None,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    block_cols: int | None = None,
    drop_tol: float = 0.0,
) -> SparseDecision:
    backend_norm = _backend_name(backend)
    dense_nbytes = estimate_dense_nbytes(shape, np.float64)
    csr_nnz = int(nnz_estimate) if nnz_estimate is not None else int(shape[0] * shape[1])
    csr_nbytes = estimate_csr_nbytes(shape, csr_nnz)
    dense_cap = int(max(0.0, float(dense_max_mb)) * 1e6)
    csr_cap = int(max(0.0, float(csr_max_mb)) * 1e6)

    dense_fits = dense_nbytes <= dense_cap if dense_cap > 0 else False
    csr_fits = csr_nbytes <= csr_cap if csr_cap > 0 else False
    sparse_smaller = csr_nbytes <= dense_nbytes

    if force_dense and force_sparse:
        raise ValueError("force_dense and force_sparse cannot both be true")

    if force_dense:
        return SparseDecision(
            storage_kind="dense",
            reason="forced dense materialization",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if force_sparse:
        if csr_fits or not dense_fits:
            return SparseDecision(
                storage_kind="csr",
                reason="forced sparse materialization",
                backend=backend_norm,
                shape=shape,
                dense_nbytes=dense_nbytes,
                csr_nbytes_estimate=csr_nbytes,
                nnz_estimate=nnz_estimate,
                block_cols=block_cols,
                drop_tol=drop_tol,
            )
        return SparseDecision(
            storage_kind="linear_operator",
            reason="forced sparse but CSR budget unavailable",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if backend_norm == "gpu" and prefer_sparse_on_gpu and csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="preferred sparse host materialization on GPU",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if sparse_smaller and csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="CSR smaller than dense and within budget",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if dense_fits:
        return SparseDecision(
            storage_kind="dense",
            reason="dense within budget",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="dense budget exceeded but CSR fits",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    return SparseDecision(
        storage_kind="linear_operator",
        reason="dense and CSR budgets exceeded; keep operator-only fallback",
        backend=backend_norm,
        shape=shape,
        dense_nbytes=dense_nbytes,
        csr_nbytes_estimate=csr_nbytes,
        nnz_estimate=nnz_estimate,
        block_cols=block_cols,
        drop_tol=drop_tol,
    )


def _drop_tol_dense(a: np.ndarray, drop_tol: float) -> np.ndarray:
    if float(drop_tol) <= 0.0:
        return np.array(a, copy=True)
    out = np.array(a, copy=True)
    out[np.abs(out) <= float(drop_tol)] = 0.0
    return out


def _operator_from_matrix(matrix: np.ndarray | sp.spmatrix) -> LinearOperator:
    n_rows, n_cols = matrix.shape

    def _matvec(x):
        return np.asarray(matrix @ np.asarray(x))

    dtype = matrix.dtype if sp.issparse(matrix) else np.asarray(matrix).dtype
    return LinearOperator((n_rows, n_cols), matvec=_matvec, dtype=dtype)


def _normalize_dense_input(a, *, dtype=None) -> np.ndarray:
    arr = _host_array(a, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    return arr


def build_operator_from_dense(
    a,
    *,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    drop_tol: float = 0.0,
) -> SparseOperatorBundle:
    dense = _normalize_dense_input(a)
    nnz = int(np.count_nonzero(np.abs(dense) > float(drop_tol))) if float(drop_tol) > 0.0 else int(np.count_nonzero(dense))
    decision = choose_storage_kind(
        shape=dense.shape,
        nnz_estimate=nnz,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=force_sparse,
        force_dense=force_dense,
        drop_tol=drop_tol,
    )
    if decision.storage_kind == "csr":
        matrix = sp.csr_matrix(_drop_tol_dense(dense, drop_tol))
    else:
        matrix = np.array(dense, copy=True)
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def _coerce_block(block, *, dtype=None):
    if block is None:
        return None
    arr = _host_array(block, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D block, got shape {arr.shape}")
    return arr


def build_operator_from_blocks(
    blocks,
    *,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    drop_tol: float = 0.0,
) -> SparseOperatorBundle:
    block_rows = [[_coerce_block(block, dtype=None) for block in row] for row in blocks]
    matrix = sp.bmat(block_rows, format="csr")
    if float(drop_tol) > 0.0:
        matrix = matrix.copy()
        matrix.data[np.abs(matrix.data) <= float(drop_tol)] = 0.0
        matrix.eliminate_zeros()
    nnz = int(matrix.nnz)
    decision = choose_storage_kind(
        shape=matrix.shape,
        nnz_estimate=nnz,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=True,
        drop_tol=drop_tol,
    )
    if decision.storage_kind != "csr":
        decision = SparseDecision(
            storage_kind="csr",
            reason="block assembly is explicitly sparse",
            backend=decision.backend,
            shape=matrix.shape,
            dense_nbytes=decision.dense_nbytes,
            csr_nbytes_estimate=decision.csr_nbytes_estimate,
            nnz_estimate=nnz,
            block_cols=decision.block_cols,
            drop_tol=drop_tol,
        )
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def _matvec_to_dense(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    dtype=np.float64,
    block_cols: int = 32,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    n = int(n)
    block_cols = max(1, int(block_cols))
    dtype_np = np.dtype(dtype)
    dense = np.empty((n, n), dtype=dtype_np)
    for start in range(0, n, block_cols):
        width = min(block_cols, n - start)
        cols = np.zeros((n, width), dtype=dtype_np)
        cols[np.arange(start, start + width), np.arange(width)] = 1
        if matmat is not None:
            out = _host_array(matmat(cols), dtype=dtype)
        else:
            out_cols = [_host_array(matvec(cols[:, j]), dtype=dtype) for j in range(cols.shape[1])]
            out = np.column_stack(out_cols)
        dense[:, start : start + width] = out
    return dense


def build_operator_from_matvec(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    dtype=np.float64,
    backend: str | None = None,
    block_cols: int = 32,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    drop_tol: float = 0.0,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
    allow_operator_only: bool = True,
) -> SparseOperatorBundle:
    dense_nbytes = estimate_dense_nbytes((int(n), int(n)), dtype)
    backend_norm = _backend_name(backend)
    if allow_operator_only and dense_nbytes > int(max(0.0, float(dense_max_mb)) * 1e6):
        decision = SparseDecision(
            storage_kind="linear_operator",
            reason="dense assembly would exceed budget; keep operator-only fallback",
            backend=backend_norm,
            shape=(int(n), int(n)),
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=estimate_csr_nbytes((int(n), int(n)), int(n) * int(n), data_dtype=dtype),
            nnz_estimate=None,
            block_cols=int(block_cols),
            drop_tol=float(drop_tol),
        )

        def _op_matvec(x):
            return np.asarray(matvec(np.asarray(x)))

        operator = LinearOperator((int(n), int(n)), matvec=_op_matvec, dtype=np.dtype(dtype))
        return SparseOperatorBundle(matrix=None, operator=operator, metadata=decision)

    dense = _matvec_to_dense(matvec, int(n), dtype=dtype, block_cols=block_cols, matmat=matmat)
    return build_operator_from_dense(
        dense,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=force_sparse,
        force_dense=force_dense,
        drop_tol=drop_tol,
    )


def _normalize_pattern(pattern, *, shape: tuple[int, int] | None = None) -> sp.csr_matrix:
    if sp.issparse(pattern):
        pattern_csr = pattern.tocsr(copy=True)
    else:
        pattern_csr = sp.csr_matrix(np.asarray(pattern))
    if shape is not None and tuple(pattern_csr.shape) != tuple(shape):
        raise ValueError(f"pattern shape {pattern_csr.shape} does not match expected shape {shape}")
    if pattern_csr.ndim != 2:
        raise ValueError(f"expected a 2D sparsity pattern, got shape {pattern_csr.shape}")
    pattern_csr.sum_duplicates()
    if pattern_csr.nnz:
        pattern_csr.data = np.ones_like(pattern_csr.data, dtype=bool)
    pattern_csr = pattern_csr.astype(bool)
    pattern_csr.eliminate_zeros()
    return pattern_csr


def color_pattern_columns(pattern, *, max_colors: int | None = None) -> list[list[int]]:
    """Greedily group columns whose declared row supports do not overlap.

    A single matvec with ones in all columns of a color recovers every value in
    that color when their row supports are disjoint. This is the sparse analogue
    of column-by-column probing, but it avoids materializing a dense identity or
    dense operator before converting to CSR.
    """

    pattern_csc = _normalize_pattern(pattern).tocsc()
    max_colors_use = None if max_colors is None else max(1, int(max_colors))
    color_rows: list[set[int]] = []
    color_cols: list[list[int]] = []
    for col in range(pattern_csc.shape[1]):
        start, end = pattern_csc.indptr[col], pattern_csc.indptr[col + 1]
        rows = set(int(row) for row in pattern_csc.indices[start:end])
        if not rows:
            continue
        for color_index, used_rows in enumerate(color_rows):
            if rows.isdisjoint(used_rows):
                used_rows.update(rows)
                color_cols[color_index].append(col)
                break
        else:
            color_rows.append(set(rows))
            color_cols.append([col])
            if max_colors_use is not None and len(color_cols) > max_colors_use:
                raise ValueError(
                    f"pattern probing would require more than max_colors={max_colors_use} colors"
                )
    return color_cols


def build_operator_from_pattern(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    pattern,
    dtype=np.float64,
    backend: str | None = None,
    csr_max_mb: float = 512.0,
    drop_tol: float = 0.0,
    allow_operator_only: bool = False,
    max_colors: int | None = None,
    color_batch: int = 1,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SparseOperatorBundle:
    """Materialize a sparse operator by probing a known sparsity pattern.

    The pattern must be a conservative structural superset of all nonzeros. Each
    color is evaluated with one combined seed vector; entries are then unpacked
    using the declared row supports. If the pattern misses a true nonzero, that
    value cannot be recovered, so callers should validate this path against the
    matrix-free operator before using it as a production backend.
    """

    dtype_np = np.dtype(dtype)
    pattern_csr = _normalize_pattern(pattern)
    n_rows, n_cols = pattern_csr.shape
    backend_norm = _backend_name(backend)
    dense_nbytes = estimate_dense_nbytes((n_rows, n_cols), dtype_np)
    csr_nbytes_estimate = estimate_csr_nbytes((n_rows, n_cols), int(pattern_csr.nnz), data_dtype=dtype_np)
    csr_cap = int(max(0.0, float(csr_max_mb)) * 1e6)
    if (not allow_operator_only) and (csr_cap <= 0 or csr_nbytes_estimate > csr_cap):
        raise MemoryError(
            "pattern CSR estimate would exceed budget "
            f"({csr_nbytes_estimate / 1.0e6:.3g} MB > {float(csr_max_mb):.3g} MB)"
        )
    if allow_operator_only and (csr_cap <= 0 or csr_nbytes_estimate > csr_cap):
        decision = SparseDecision(
            storage_kind="linear_operator",
            reason="pattern CSR estimate would exceed budget; keep operator-only fallback",
            backend=backend_norm,
            shape=(n_rows, n_cols),
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes_estimate,
            nnz_estimate=int(pattern_csr.nnz),
            drop_tol=float(drop_tol),
        )

        def _op_matvec(x):
            return np.asarray(matvec(np.asarray(x, dtype=dtype_np)))

        operator = LinearOperator((n_rows, n_cols), matvec=_op_matvec, dtype=dtype_np)
        return SparseOperatorBundle(matrix=None, operator=operator, metadata=decision)

    if progress_callback is not None:
        progress_callback(
            "pattern-probe preflight "
            f"shape={n_rows}x{n_cols} pattern_nnz={int(pattern_csr.nnz)} "
            f"csr_estimate_mb={csr_nbytes_estimate / 1.0e6:.3g}"
        )
    colors = color_pattern_columns(pattern_csr, max_colors=max_colors)
    color_batch_use = max(1, int(color_batch))
    if matmat is None:
        color_batch_use = 1
    if progress_callback is not None:
        progress_callback(
            f"pattern-probe coloring complete colors={len(colors)} columns={n_cols} "
            f"color_batch={color_batch_use}"
        )

    pattern_csc = pattern_csr.tocsc()
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    def _append_color_values(cols: list[int], out: np.ndarray) -> None:
        if out.shape != (n_rows,):
            raise ValueError(f"matvec returned shape {out.shape}; expected {(n_rows,)}")
        for col in cols:
            start, end = pattern_csc.indptr[col], pattern_csc.indptr[col + 1]
            rows = pattern_csc.indices[start:end]
            values = np.asarray(out[rows], dtype=dtype_np)
            if float(drop_tol) > 0.0:
                keep = np.abs(values) > float(drop_tol)
            else:
                keep = values != 0
            if not np.any(keep):
                continue
            rows_kept = np.asarray(rows[keep], dtype=np.int32)
            values_kept = np.asarray(values[keep], dtype=dtype_np)
            row_parts.append(rows_kept)
            col_parts.append(np.full(rows_kept.shape, int(col), dtype=np.int32))
            data_parts.append(values_kept)

    color_index = 0
    for batch_start in range(0, len(colors), color_batch_use):
        batch_colors = colors[batch_start : batch_start + color_batch_use]
        if color_batch_use == 1:
            cols = batch_colors[0]
            seed = np.zeros(n_cols, dtype=dtype_np)
            seed[np.asarray(cols, dtype=np.intp)] = 1
            out = _host_array(matvec(seed), dtype=dtype_np)
            _append_color_values(cols, np.asarray(out, dtype=dtype_np))
            color_index += 1
        else:
            seeds = np.zeros((n_cols, len(batch_colors)), dtype=dtype_np)
            for batch_col, cols in enumerate(batch_colors):
                seeds[np.asarray(cols, dtype=np.intp), batch_col] = 1
            assert matmat is not None
            out_batch = _host_array(matmat(seeds), dtype=dtype_np)
            expected_shape = (n_rows, len(batch_colors))
            if out_batch.shape != expected_shape:
                raise ValueError(f"matmat returned shape {out_batch.shape}; expected {expected_shape}")
            for batch_col, cols in enumerate(batch_colors):
                _append_color_values(cols, np.asarray(out_batch[:, batch_col], dtype=dtype_np))
            color_index += len(batch_colors)
        if progress_callback is not None and (color_index == len(colors) or color_index % 10 == 0):
            progress_callback(f"pattern-probe colors_done={color_index}/{len(colors)}")

    if data_parts:
        data = np.concatenate(data_parts)
        row_indices = np.concatenate(row_parts)
        col_indices = np.concatenate(col_parts)
    else:
        data = np.asarray([], dtype=dtype_np)
        row_indices = np.asarray([], dtype=np.int32)
        col_indices = np.asarray([], dtype=np.int32)
    matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols), dtype=dtype_np).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if progress_callback is not None:
        progress_callback(f"pattern-probe csr built nnz={int(matrix.nnz)}")
    decision = SparseDecision(
        storage_kind="csr",
        reason=(
            f"pattern-probed sparse materialization ({len(colors)} colors for {n_cols} columns; "
            f"color_batch={color_batch_use})"
        ),
        backend=backend_norm,
        shape=(n_rows, n_cols),
        dense_nbytes=dense_nbytes,
        csr_nbytes_estimate=estimate_csr_nbytes((n_rows, n_cols), int(matrix.nnz), data_dtype=dtype_np),
        nnz_estimate=int(matrix.nnz),
        block_cols=len(colors),
        drop_tol=float(drop_tol),
    )
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def factorize_host_sparse_operator(
    operator: SparseOperatorBundle | np.ndarray | sp.spmatrix,
    *,
    kind: FactorKind = "lu",
    fill_factor: float = 10.0,
    drop_tol: float = 1.0e-4,
    permc_spec: str = "COLAMD",
    diag_pivot_thresh: float = 1.0,
    symbolic_analysis: SparseSymbolicAnalysis | None = None,
    symbolic_ordering_kind: str = "rcm",
    symbolic_block_size: int = 4096,
    symbolic_block_overlap: int = 0,
    symbolic_coarse_max_cols: int = 256,
    symbolic_coarse_probe_cols: int = 4,
    symbolic_coarse_damping: float = 1.0,
    symbolic_coarse_regularization_rel: float = 1.0e-10,
    symbolic_schur_max_separator_cols: int = 256,
    symbolic_schur_tail_size: int = 0,
    symbolic_schur_boundary_width: int = 1,
    symbolic_schur_high_degree_cols: int = 64,
    symbolic_schur_regularization_rel: float = 1.0e-12,
    symbolic_frontal_max_separator_cols: int = 1024,
    symbolic_frontal_tail_size: int = 0,
    symbolic_frontal_boundary_width: int = 1,
    symbolic_frontal_high_degree_cols: int = 128,
    symbolic_frontal_max_superblock_size: int = 8192,
    symbolic_frontal_max_superblock_blocks: int = 8,
    symbolic_frontal_min_cross_nnz: int = 1,
    symbolic_frontal_min_cross_separator_fraction: float = 0.0,
    symbolic_frontal_regularization_rel: float = 1.0e-12,
    symbolic_frontal_max_dense_rhs_entries: int = 0,
    symbolic_frontal_max_dense_rhs_cols_per_block: int = 0,
    symbolic_superblock_max_size: int = 32768,
    symbolic_superblock_max_blocks: int = 8,
    symbolic_superblock_min_cross_nnz: int = 1,
    symbolic_superblock_min_retained_cross_fraction: float = 0.0,
    symbolic_superblock_regularization_rel: float = 1.0e-12,
    symbolic_numeric_parallel_workers: int = 1,
    symbolic_max_permutation_size: int = 250_000,
) -> SparseFactorBundle:
    if isinstance(operator, SparseOperatorBundle):
        matrix = operator.matrix
        metadata = operator.metadata
    else:
        matrix = operator
        if sp.issparse(matrix):
            nnz = int(matrix.nnz)
            dense_dtype = matrix.dtype
            shape = tuple(matrix.shape)
        else:
            matrix = np.asarray(matrix)
            nnz = int(np.count_nonzero(matrix))
            dense_dtype = matrix.dtype
            shape = tuple(matrix.shape)
        metadata = SparseDecision(
            storage_kind="csr" if sp.issparse(matrix) else "dense",
            reason="factorized direct matrix input",
            backend="cpu",
            shape=shape,
            dense_nbytes=estimate_dense_nbytes(shape, dense_dtype),
            csr_nbytes_estimate=estimate_csr_nbytes(shape, nnz, data_dtype=dense_dtype),
            nnz_estimate=nnz,
        )

    if matrix is None:
        raise ValueError("factorize_host_sparse_operator requires a materialized matrix")

    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix))

    csc = matrix.tocsc()
    factor_start_s = time.perf_counter()
    max_abs = float(np.max(np.abs(csc.data))) if csc.nnz else 0.0
    attempts_env = os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_ILU_ATTEMPTS", "").strip()
    try:
        ilu_attempts = int(attempts_env) if attempts_env else 1
    except ValueError:
        ilu_attempts = 1
    ilu_attempts = max(1, int(ilu_attempts))
    singular_reg_env = os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_ILU_SINGULAR_REG_REL", "").strip()
    try:
        singular_reg_rel = float(singular_reg_env) if singular_reg_env else 1.0e-10
    except ValueError:
        singular_reg_rel = 1.0e-10
    singular_reg_rel = max(0.0, float(singular_reg_rel))
    ilu_drop_tol_eff = float(drop_tol)
    fill_factor_eff = float(fill_factor)
    last_factor_error: RuntimeError | None = None

    def _regularized_csc(attempt: int) -> sp.csc_matrix:
        if attempt <= 0 or singular_reg_rel <= 0.0:
            return csc
        reg = max(1.0e-12, float(singular_reg_rel) * (10.0 ** (attempt - 1)) * max(1.0, max_abs))
        return (csc + reg * sp.eye(csc.shape[0], csc.shape[1], dtype=csc.dtype, format="csc")).tocsc()

    try:
        if kind == "lu":
            factor = splu(csc, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh)
        elif kind == "ilu":
            factor = None
            for attempt in range(int(ilu_attempts)):
                try:
                    factor = spilu(
                        _regularized_csc(attempt),
                        fill_factor=fill_factor_eff,
                        drop_tol=ilu_drop_tol_eff,
                        permc_spec=permc_spec,
                        diag_pivot_thresh=diag_pivot_thresh,
                    )
                    break
                except RuntimeError as exc:
                    last_factor_error = exc
                    msg = str(exc).lower()
                    if not (
                        attempt + 1 < int(ilu_attempts)
                        and (("singular" in msg) or ("pivot" in msg) or ("dpivot" in msg) or ("zero" in msg))
                    ):
                        raise
                    ilu_drop_tol_eff = max(0.0, float(ilu_drop_tol_eff) * 0.1)
                    fill_factor_eff = max(float(fill_factor_eff), float(fill_factor) * 2.0, 20.0)
            if factor is None:
                assert last_factor_error is not None
                raise last_factor_error
        elif kind == "jacobi":
            matrix_dtype = np.dtype(matrix.dtype)
            factor_dtype = np.dtype(matrix_dtype if np.issubdtype(matrix_dtype, np.floating) else np.float64)
            diagonal = np.asarray(matrix.diagonal(), dtype=factor_dtype)
            if diagonal.size != int(matrix.shape[0]):
                raise RuntimeError("Jacobi preconditioner requires a square matrix diagonal")
            if np.any(~np.isfinite(diagonal)):
                raise RuntimeError("Jacobi preconditioner diagonal is non-finite")
            scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
            floor = 1.0e-12 * scale
            sign = np.where(diagonal < 0.0, -1.0, 1.0).astype(factor_dtype, copy=False)
            diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
            factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=factor_dtype))
        elif kind in {
            "symbolic_block_lu",
            "symbolic_block_lu_coarse",
            "symbolic_block_schur_lu",
            "symbolic_superblock_lu",
            "symbolic_frontal_schur_lu",
        }:
            analysis = symbolic_analysis
            if analysis is None:
                analysis = analyze_sparse_symbolic_structure(
                    matrix,
                    ordering_kind=str(symbolic_ordering_kind),
                    block_size_target=int(symbolic_block_size),
                    max_permutation_size=int(symbolic_max_permutation_size),
                )
            if kind == "symbolic_block_schur_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_schur_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_separator_cols=int(symbolic_schur_max_separator_cols),
                    tail_size=int(symbolic_schur_tail_size),
                    boundary_width=int(symbolic_schur_boundary_width),
                    high_degree_cols=int(symbolic_schur_high_degree_cols),
                    regularization_rel=float(symbolic_schur_regularization_rel),
                )
            elif kind == "symbolic_frontal_schur_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_frontal_schur_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_separator_cols=int(symbolic_frontal_max_separator_cols),
                    tail_size=int(symbolic_frontal_tail_size),
                    boundary_width=int(symbolic_frontal_boundary_width),
                    high_degree_cols=int(symbolic_frontal_high_degree_cols),
                    max_superblock_size=int(symbolic_frontal_max_superblock_size),
                    max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                    min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                    min_cross_separator_fraction=float(symbolic_frontal_min_cross_separator_fraction),
                    regularization_rel=float(symbolic_frontal_regularization_rel),
                    max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                    max_dense_rhs_cols_per_block=int(symbolic_frontal_max_dense_rhs_cols_per_block),
                )
            elif kind == "symbolic_superblock_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_superblock_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_superblock_size=int(symbolic_superblock_max_size),
                    max_superblock_blocks=int(symbolic_superblock_max_blocks),
                    min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                    min_retained_cross_fraction=float(symbolic_superblock_min_retained_cross_fraction),
                    regularization_rel=float(symbolic_superblock_regularization_rel),
                    parallel_workers=int(symbolic_numeric_parallel_workers),
                )
            elif kind == "symbolic_block_lu_coarse":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_coarse_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    overlap_size=int(symbolic_block_overlap),
                    coarse_max_cols=int(symbolic_coarse_max_cols),
                    coarse_probe_cols=int(symbolic_coarse_probe_cols),
                    coarse_damping=float(symbolic_coarse_damping),
                    coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                )
            else:
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    overlap_size=int(symbolic_block_overlap),
                )
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown factorization kind: {kind}")
    except RuntimeError as exc:
        detail = str(exc).strip()
        raise RuntimeError(
            "Host sparse factorization failed. The assembled RHSMode=1 operator may be singular, "
            "ill-conditioned, or missing a pinned gauge/nullspace constraint for this solver branch. "
            "Use the default solver for parity runs, try solve_method='sparse_lsmr' only for diagnostic "
            "minimum-norm probes, or adjust SFINCS_JAX_EXPLICIT_SPARSE_* factorization controls. "
            f"Underlying factorization error: {detail}"
        ) from exc
    factor_s = time.perf_counter() - factor_start_s
    if kind == "jacobi":
        factor_nbytes, factor_nnz = int(factor.inverse_diagonal.nbytes), int(factor.inverse_diagonal.size)
    elif kind in {
        "symbolic_block_lu",
        "symbolic_block_lu_coarse",
            "symbolic_block_schur_lu",
            "symbolic_superblock_lu",
            "symbolic_frontal_schur_lu",
    }:
        factor_nbytes, factor_nnz = int(symbolic_nbytes), int(symbolic_nnz)
    else:
        factor_nbytes, factor_nnz = estimate_superlu_factor_storage(factor)
    return SparseFactorBundle(
        factor=factor,
        operator=SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=metadata),
        metadata=metadata,
        kind=kind,
        factor_nbytes_estimate=factor_nbytes,
        factor_nnz_estimate=factor_nnz,
        factor_s=float(factor_s),
    )


# Host explicit-sparse operator assembly and factorization orchestration.
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


def host_sparse_direct_solve_with_refinement(
    *,
    ilu,
    a_csr_full,
    rhs_vec,
    factor_dtype: np.dtype,
    refine_steps: int,
) -> tuple[np.ndarray, float]:
    """Solve with a sparse host factor and monotone iterative refinement."""

    rhs64 = np.asarray(rhs_vec, dtype=np.float64).reshape((-1,))
    rhs_factor = np.asarray(rhs_vec, dtype=factor_dtype).reshape((-1,))
    x_np = np.asarray(ilu.solve(rhs_factor), dtype=np.float64)
    residual_np = rhs64 - a_csr_full @ x_np
    residual_norm = float(np.linalg.norm(residual_np))
    for _ in range(max(0, int(refine_steps))):
        if not np.isfinite(residual_norm) or residual_norm == 0.0:
            break
        dx_np = np.asarray(ilu.solve(np.asarray(residual_np, dtype=factor_dtype)), dtype=np.float64)
        x_trial = x_np + dx_np
        residual_trial = rhs64 - a_csr_full @ x_trial
        residual_norm_trial = float(np.linalg.norm(residual_trial))
        if not np.isfinite(residual_norm_trial) or residual_norm_trial >= residual_norm:
            break
        x_np = x_trial
        residual_np = residual_trial
        residual_norm = residual_norm_trial
    return x_np, residual_norm


def host_direct_solve_with_refinement(
    *,
    factor_solve: Callable[[np.ndarray], np.ndarray],
    operator_matrix,
    rhs_vec,
    factor_dtype: np.dtype,
    refine_steps: int,
) -> tuple[np.ndarray, float]:
    """Solve with a host direct factor callback and monotone refinement."""

    rhs64 = np.asarray(rhs_vec, dtype=np.float64).reshape((-1,))
    rhs_factor = np.asarray(rhs_vec, dtype=factor_dtype).reshape((-1,))
    x_np = np.asarray(factor_solve(rhs_factor), dtype=np.float64)
    residual_np = rhs64 - operator_matrix @ x_np
    residual_norm = float(np.linalg.norm(residual_np))
    for _ in range(max(0, int(refine_steps))):
        if not np.isfinite(residual_norm) or residual_norm == 0.0:
            break
        dx_np = np.asarray(factor_solve(np.asarray(residual_np, dtype=factor_dtype)), dtype=np.float64)
        x_trial = x_np + dx_np
        residual_trial = rhs64 - operator_matrix @ x_trial
        residual_norm_trial = float(np.linalg.norm(residual_trial))
        if not np.isfinite(residual_norm_trial) or residual_norm_trial >= residual_norm:
            break
        x_np = x_trial
        residual_np = residual_trial
        residual_norm = residual_norm_trial
    return x_np, residual_norm


def host_sparse_direct_polish(
    *,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_vec: jnp.ndarray,
    x0_np: np.ndarray,
    ilu,
    factor_dtype: np.dtype,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    gmres_solver: Callable[..., tuple[np.ndarray, float, list[float]]] | None = None,
) -> tuple[np.ndarray, float]:
    """Polish a host sparse-direct solution with preconditioned SciPy GMRES."""

    def _precond_sparse(v: jnp.ndarray) -> jnp.ndarray:
        v_np = np.asarray(v, dtype=factor_dtype).reshape((-1,))
        y_np = ilu.solve(v_np)
        return jnp.asarray(y_np, dtype=jnp.float64)

    solver = gmres_solve_with_history_scipy if gmres_solver is None else gmres_solver
    x_np, _rn_sparse, _history = solver(
        matvec=matvec_fn,
        b=rhs_vec,
        preconditioner=_precond_sparse,
        x0=jnp.asarray(x0_np, dtype=jnp.float64),
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        precondition_side=precondition_side,
    )
    x_polish = np.asarray(x_np, dtype=np.float64)
    residual_vec = rhs_vec - matvec_fn(jnp.asarray(x_polish, dtype=jnp.float64))
    residual_norm = float(jnp.linalg.norm(residual_vec))
    return x_polish, residual_norm
