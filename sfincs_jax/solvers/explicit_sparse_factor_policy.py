"""Environment and factor-kind policy for explicit sparse host factors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


EnvMapping = Mapping[str, str]
_VALID_PERMC_SPECS = frozenset({"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"})


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
    "symbolic_blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "native_blr_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "compressed_frontal_schur_lu": "symbolic_blr_frontal_schur_lu",
    "symbolic_nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "nested_dissection_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "native_nd_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
    "multilevel_frontal_schur_lu": "symbolic_nd_frontal_schur_lu",
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
    symbolic_blr_frontal_tol: float
    symbolic_blr_frontal_max_rank: int
    symbolic_blr_frontal_min_cols: int
    symbolic_blr_frontal_gmres_rtol: float
    symbolic_blr_frontal_gmres_atol: float
    symbolic_blr_frontal_gmres_maxiter: int
    symbolic_blr_frontal_gmres_restart: int
    symbolic_blr_frontal_woodbury_max_rank: int
    symbolic_blr_frontal_woodbury_max_condition: float
    symbolic_nd_max_leaf_size: int
    symbolic_nd_max_terminal_factor_size: int
    symbolic_nd_max_depth: int
    symbolic_nd_separator_width: int
    symbolic_nd_max_separator_cols: int
    symbolic_nd_high_degree_cols: int
    symbolic_nd_regularization_rel: float
    symbolic_nd_max_dense_rhs_entries: int
    symbolic_nd_max_dense_rhs_entries_per_child: int
    symbolic_nd_max_dense_rhs_cols_per_child: int
    symbolic_nd_max_setup_s: float
    symbolic_nd_compress_updates: bool
    symbolic_nd_parallel_update_workers: int
    symbolic_nd_residual_polish_steps: int
    symbolic_nd_residual_polish_damping: float
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
        symbolic_blr_frontal_tol=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_TOL"),
            float(default_symbolic_blr_frontal_tol),
            minimum=0.0,
        ),
        symbolic_blr_frontal_max_rank=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_MAX_RANK"),
            int(default_symbolic_blr_frontal_max_rank),
            minimum=1,
        ),
        symbolic_blr_frontal_min_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_MIN_COLS"),
            int(default_symbolic_blr_frontal_min_cols),
            minimum=1,
        ),
        symbolic_blr_frontal_gmres_rtol=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_GMRES_RTOL"),
            float(default_symbolic_blr_frontal_gmres_rtol),
            minimum=0.0,
        ),
        symbolic_blr_frontal_gmres_atol=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_GMRES_ATOL"),
            float(default_symbolic_blr_frontal_gmres_atol),
            minimum=0.0,
        ),
        symbolic_blr_frontal_gmres_maxiter=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_GMRES_MAXITER"),
            int(default_symbolic_blr_frontal_gmres_maxiter),
            minimum=1,
        ),
        symbolic_blr_frontal_gmres_restart=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_GMRES_RESTART"),
            int(default_symbolic_blr_frontal_gmres_restart),
            minimum=1,
        ),
        symbolic_blr_frontal_woodbury_max_rank=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_RANK"),
            int(default_symbolic_blr_frontal_woodbury_max_rank),
            minimum=0,
        ),
        symbolic_blr_frontal_woodbury_max_condition=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_CONDITION"),
            float(default_symbolic_blr_frontal_woodbury_max_condition),
            minimum=1.0,
        ),
        symbolic_nd_max_leaf_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_LEAF_SIZE"),
            int(default_symbolic_nd_max_leaf_size),
            minimum=1,
        ),
        symbolic_nd_max_terminal_factor_size=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_TERMINAL_FACTOR_SIZE"),
            int(default_symbolic_nd_max_terminal_factor_size),
            minimum=1,
        ),
        symbolic_nd_max_depth=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_DEPTH"),
            int(default_symbolic_nd_max_depth),
            minimum=0,
        ),
        symbolic_nd_separator_width=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_SEPARATOR_WIDTH"),
            int(default_symbolic_nd_separator_width),
            minimum=1,
        ),
        symbolic_nd_max_separator_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_SEPARATOR_COLS"),
            int(default_symbolic_nd_max_separator_cols),
            minimum=1,
        ),
        symbolic_nd_high_degree_cols=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_HIGH_DEGREE_COLS"),
            int(default_symbolic_nd_high_degree_cols),
            minimum=0,
        ),
        symbolic_nd_regularization_rel=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_REG_REL"),
            float(default_symbolic_nd_regularization_rel),
            minimum=0.0,
        ),
        symbolic_nd_max_dense_rhs_entries=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES"),
            int(default_symbolic_nd_max_dense_rhs_entries),
            minimum=0,
        ),
        symbolic_nd_max_dense_rhs_entries_per_child=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES_PER_CHILD"),
            int(default_symbolic_nd_max_dense_rhs_entries_per_child),
            minimum=0,
        ),
        symbolic_nd_max_dense_rhs_cols_per_child=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_DENSE_RHS_COLS_PER_CHILD"),
            int(default_symbolic_nd_max_dense_rhs_cols_per_child),
            minimum=0,
        ),
        symbolic_nd_max_setup_s=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_MAX_SETUP_S"),
            float(default_symbolic_nd_max_setup_s),
            minimum=0.0,
        ),
        symbolic_nd_compress_updates=parse_explicit_sparse_bool(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_COMPRESS_UPDATES"),
            bool(default_symbolic_nd_compress_updates),
        ),
        symbolic_nd_parallel_update_workers=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_PARALLEL_UPDATE_WORKERS"),
            int(default_symbolic_nd_parallel_update_workers),
            minimum=1,
        ),
        symbolic_nd_residual_polish_steps=parse_explicit_sparse_int(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_RESIDUAL_POLISH_STEPS"),
            int(default_symbolic_nd_residual_polish_steps),
            minimum=0,
        ),
        symbolic_nd_residual_polish_damping=parse_explicit_sparse_float(
            _env_value(env_map, "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_ND_RESIDUAL_POLISH_DAMPING"),
            float(default_symbolic_nd_residual_polish_damping),
            minimum=0.0,
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


__all__ = [
    "ExplicitSparseFactorSettings",
    "canonical_explicit_sparse_factor_kind",
    "explicit_sparse_factor_kind_from_env",
    "explicit_sparse_factor_settings_from_env",
    "explicit_sparse_monolithic_guard_enabled",
    "explicit_sparse_monolithic_max_size",
    "parse_explicit_sparse_bool",
    "parse_explicit_sparse_float",
    "parse_explicit_sparse_int",
]
