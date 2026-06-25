"""Policies for RHSMode=1 symbolic sparse active preconditioners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ActiveSymbolicSuperblockPolicy:
    """Resolved controls for the active symbolic superblock factor."""

    max_active_size: int
    ordering_kind: str
    block_size: int
    max_permutation_size: int
    max_superblock_size: int
    max_superblock_blocks: int
    min_cross_nnz: int
    min_retained_cross_fraction: float
    regularization_rel: float
    prefill_safety_factor: float
    admission_probes: int
    admission_max_relative_residual: float
    admission_min_improvement: float


@dataclass(frozen=True)
class ActiveSymbolicBlockSchurPolicy:
    """Resolved controls for the active symbolic separator-Schur factor."""

    max_active_size: int
    ordering_kind: str
    block_size: int
    max_permutation_size: int
    separator_cols: int
    boundary_width: int
    high_degree_cols: int
    regularization_rel: float
    prefill_safety_factor: float
    admission_probes: int
    admission_max_relative_residual: float
    admission_min_improvement: float


def resolve_active_symbolic_superblock_policy(
    *,
    active_size: int,
    regularization: float,
    env: Mapping[str, str] | None = None,
) -> ActiveSymbolicSuperblockPolicy:
    """Resolve environment controls for symbolic grouped-block factors."""

    env_map = os.environ if env is None else env
    large_retained_default = 0.25 if int(active_size) > 300_000 else 0.0
    return ActiveSymbolicSuperblockPolicy(
        max_active_size=int(
            _env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_ACTIVE_SIZE", 300_000)
        ),
        ordering_kind=str(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ORDERING", "rcm")
        ).strip().lower(),
        block_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_BLOCK_SIZE", 1024)),
        ),
        max_permutation_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_PERMUTATION_SIZE",
                    300_000,
                )
            ),
        ),
        max_superblock_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_SIZE", 32768)),
        ),
        max_superblock_blocks=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS", 8)),
        ),
        min_cross_nnz=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MIN_CROSS_NNZ", 1)),
        ),
        min_retained_cross_fraction=max(
            0.0,
            min(
                1.0,
                float(
                    _env_float(
                        env_map,
                        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MIN_RETAINED_CROSS_FRACTION",
                        large_retained_default,
                    )
                ),
            ),
        ),
        regularization_rel=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_REGULARIZATION_REL",
                    max(float(abs(regularization)), 1.0e-12),
                )
            ),
        ),
        prefill_safety_factor=max(
            1.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_PREFILL_SAFETY_FACTOR",
                    8.0,
                )
            ),
        ),
        admission_probes=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_PROBES", 4)),
        ),
        admission_max_relative_residual=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        admission_min_improvement=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
    )


def resolve_active_symbolic_block_schur_policy(
    *,
    regularization: float,
    env: Mapping[str, str] | None = None,
) -> ActiveSymbolicBlockSchurPolicy:
    """Resolve environment controls for symbolic separator-Schur factors."""

    env_map = os.environ if env is None else env
    return ActiveSymbolicBlockSchurPolicy(
        max_active_size=int(
            _env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_ACTIVE_SIZE", 300_000)
        ),
        ordering_kind=str(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ORDERING", "rcm")
        ).strip().lower(),
        block_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BLOCK_SIZE", 2048)),
        ),
        max_permutation_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_PERMUTATION_SIZE",
                    300_000,
                )
            ),
        ),
        separator_cols=max(
            0,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_SEPARATOR_COLS",
                    512,
                )
            ),
        ),
        boundary_width=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BOUNDARY_WIDTH", 1)),
        ),
        high_degree_cols=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_HIGH_DEGREE_COLS", 128)),
        ),
        regularization_rel=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_REGULARIZATION_REL",
                    max(float(abs(regularization)), 1.0e-12),
                )
            ),
        ),
        prefill_safety_factor=max(
            1.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_PREFILL_SAFETY_FACTOR",
                    4.0,
                )
            ),
        ),
        admission_probes=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_PROBES", 4)),
        ),
        admission_max_relative_residual=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        admission_min_improvement=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
    )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or int(default))
    except ValueError:
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, "")).strip() or float(default))
    except ValueError:
        return float(default)


# Consolidated from frontal_policy.py.
@dataclass(frozen=True)
class ActiveSymbolicFrontalPolicy:
    """Resolved controls for the active symbolic frontal/Schur factor."""

    requested_kind: str
    use_nd_frontal: bool
    active_symbolic_kind: str
    active_architecture: str
    max_active_size: int
    ordering_kind: str
    block_size: int
    max_permutation_size: int
    separator_cols: int
    max_superblock_size: int
    max_superblock_blocks: int
    boundary_width: int
    high_degree_cols: int
    min_cross_nnz: int
    max_dense_rhs_entries: int
    max_dense_rhs_cols_per_block: int
    min_cross_separator_fraction: float
    regularization_rel: float
    prefill_safety_factor: float
    admission_probes: int
    admission_max_relative_residual: float
    admission_min_improvement: float
    nd_max_leaf_size: int
    nd_max_terminal_factor_size: int
    nd_max_depth: int
    nd_separator_width: int
    nd_max_separator_cols: int
    nd_high_degree_cols: int
    nd_max_dense_rhs_entries: int
    nd_max_dense_rhs_entries_per_child: int
    nd_max_dense_rhs_cols_per_child: int
    nd_max_setup_s: float
    nd_residual_polish_steps: int
    nd_residual_polish_damping: float


def resolve_active_symbolic_frontal_policy(
    *,
    requested_kind: str,
    active_size: int,
    regularization: float,
    env: Mapping[str, str] | None = None,
) -> ActiveSymbolicFrontalPolicy:
    """Resolve env-driven controls for the symbolic frontal/ND preconditioner."""

    env_map = os.environ if env is None else env
    requested_kind_l = str(requested_kind).strip().lower()
    use_nd_frontal = (
        "nd_frontal" in requested_kind_l
        or "nested_dissection" in requested_kind_l
        or "multilevel" in requested_kind_l
    )
    active_symbolic_kind = (
        "active_symbolic_nd_frontal_schur_lu"
        if bool(use_nd_frontal)
        else "active_symbolic_frontal_schur_lu"
    )
    active_architecture = (
        "active_true_operator_symbolic_nd_frontal_schur_lu"
        if bool(use_nd_frontal)
        else "active_true_operator_symbolic_frontal_schur_lu"
    )
    separator_cols = max(
        0,
        int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS", 1024)),
    )
    high_degree_cols = max(
        0,
        int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS", 128)),
    )
    max_dense_rhs_entries = max(
        0,
        int(
            _env_int(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_ENTRIES",
                160_000_000,
            )
        ),
    )
    large_separator_default = 0.20 if int(active_size) > 300_000 else 0.0
    return ActiveSymbolicFrontalPolicy(
        requested_kind=str(requested_kind),
        use_nd_frontal=bool(use_nd_frontal),
        active_symbolic_kind=str(active_symbolic_kind),
        active_architecture=str(active_architecture),
        max_active_size=int(
            _env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_ACTIVE_SIZE", 300_000)
        ),
        ordering_kind=str(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ORDERING", "rcm")
        ).strip().lower(),
        block_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BLOCK_SIZE", 1024)),
        ),
        max_permutation_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_PERMUTATION_SIZE",
                    300_000,
                )
            ),
        ),
        separator_cols=int(separator_cols),
        max_superblock_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE",
                    8192,
                )
            ),
        ),
        max_superblock_blocks=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS",
                    8,
                )
            ),
        ),
        boundary_width=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH", 1)),
        ),
        high_degree_cols=int(high_degree_cols),
        min_cross_nnz=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MIN_CROSS_NNZ", 1)),
        ),
        max_dense_rhs_entries=int(max_dense_rhs_entries),
        max_dense_rhs_cols_per_block=max(
            0,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_COLS_PER_BLOCK",
                    256,
                )
            ),
        ),
        min_cross_separator_fraction=max(
            0.0,
            min(
                1.0,
                float(
                    _env_float(
                        env_map,
                        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION",
                        large_separator_default,
                    )
                ),
            ),
        ),
        regularization_rel=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_REGULARIZATION_REL",
                    max(float(abs(regularization)), 1.0e-12),
                )
            ),
        ),
        prefill_safety_factor=max(
            1.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_PREFILL_SAFETY_FACTOR",
                    4.0,
                )
            ),
        ),
        admission_probes=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ADMISSION_PROBES", 4)),
        ),
        admission_max_relative_residual=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        admission_min_improvement=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
        nd_max_leaf_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_LEAF_SIZE", 4096)),
        ),
        nd_max_terminal_factor_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_TERMINAL_FACTOR_SIZE",
                    32768,
                )
            ),
        ),
        nd_max_depth=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_DEPTH", 1)),
        ),
        nd_separator_width=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_SEPARATOR_WIDTH", 128)),
        ),
        nd_max_separator_cols=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_SEPARATOR_COLS",
                    max(1, int(separator_cols)),
                )
            ),
        ),
        nd_high_degree_cols=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_HIGH_DEGREE_COLS", high_degree_cols)),
        ),
        nd_max_dense_rhs_entries=max(
            0,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES",
                    max_dense_rhs_entries,
                )
            ),
        ),
        nd_max_dense_rhs_entries_per_child=max(
            0,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES_PER_CHILD",
                    0,
                )
            ),
        ),
        nd_max_dense_rhs_cols_per_child=max(
            0,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_DENSE_RHS_COLS_PER_CHILD",
                    0,
                )
            ),
        ),
        nd_max_setup_s=max(
            0.0,
            float(_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_MAX_SETUP_S", 0.0)),
        ),
        nd_residual_polish_steps=max(
            0,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_RESIDUAL_POLISH_STEPS", 2)),
        ),
        nd_residual_polish_damping=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_ND_RESIDUAL_POLISH_DAMPING",
                    1.0,
                )
            ),
        ),
    )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or int(default))
    except ValueError:
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, "")).strip() or float(default))
    except ValueError:
        return float(default)


# Consolidated from reduced_policy.py.
@dataclass(frozen=True)
class ActiveFortranV3ReducedFactorPolicy:
    """Resolved host factor settings for the reduced RHSMode=1 Pmat."""

    requested: str
    factor_kind: str
    large_matrix: bool
    ilu_max_size: int
    ilu_size_exceeded: bool
    fill_factor: float
    drop_tol: float
    diag_pivot: float
    permc_requested: str
    permc_candidates: tuple[str, ...]
    permc_spec: str
    scale_norm: str
    max_scale: float
    progress: bool
    lu_large_prefill_size: int
    lu_prefill_safety_factor: float


def resolve_active_fortran_v3_reduced_factor_policy(
    *,
    requested_kind: str,
    matrix_size: int,
    env: Mapping[str, str] | None = None,
) -> ActiveFortranV3ReducedFactorPolicy:
    """Resolve factorization defaults for the Fortran-v3-reduced active Pmat."""

    env_map = os.environ if env is None else env
    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind = str(
        env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "")
    ).strip().lower()
    if not factor_kind:
        factor_kind = "ilu" if "ilu" in requested or requested.endswith("pc_matrix") else "lu"
    if factor_kind not in {"ilu", "spilu", "lu", "splu"}:
        factor_kind = "ilu"
    factor_kind = "lu" if factor_kind in {"lu", "splu"} else "ilu"

    n = int(matrix_size)
    large_matrix = n >= int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_SIZE", 300_000))
    ilu_max_size = int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_ILU_MAX_SIZE", 350_000))
    ilu_size_exceeded = bool(factor_kind == "ilu" and int(ilu_max_size) > 0 and n > int(ilu_max_size))

    fill_factor_default = 3.0 if factor_kind == "ilu" else 12.0
    drop_tol_default = 3.0e-3 if factor_kind == "ilu" else 0.0
    if bool(large_matrix) and factor_kind == "ilu":
        fill_factor_default = float(
            _env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_FILL_FACTOR", 1.2)
        )
        drop_tol_default = float(
            _env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_DROP_TOL", 5.0e-2)
        )

    fill_factor = max(
        1.0,
        float(_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FILL_FACTOR", fill_factor_default)),
    )
    drop_tol = float(_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DROP_TOL", drop_tol_default))
    diag_pivot = float(
        _env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAG_PIVOT_THRESH", 0.0)
    )
    permc_requested = str(
        env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", "")
    ).strip().upper()
    permc_candidates = active_fortran_v3_reduced_permc_candidates(
        requested=permc_requested,
        factor_kind=factor_kind,
    )
    scale_norm = str(env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SCALE_NORM", "l1")).strip().lower()
    if scale_norm not in {"l1", "l2", "max"}:
        scale_norm = "l1"
    max_scale = max(
        1.0,
        float(_env_float(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_MAX_SCALE", 1.0e6)),
    )
    lu_large_prefill_size = int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_SIZE", 300_000))
    lu_prefill_default = 4.5
    if factor_kind == "lu" and n >= int(lu_large_prefill_size):
        lu_prefill_default = float(
            _env_float(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_PREFILL_SAFETY_FACTOR",
                32.0,
            )
        )
    lu_prefill_safety_factor = max(
        1.0,
        float(
            _env_float(
                env_map,
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_PREFILL_SAFETY_FACTOR",
                lu_prefill_default,
            )
        ),
    )
    progress = _env_bool(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PROGRESS",
        bool(large_matrix),
    )
    return ActiveFortranV3ReducedFactorPolicy(
        requested=str(requested),
        factor_kind=str(factor_kind),
        large_matrix=bool(large_matrix),
        ilu_max_size=int(ilu_max_size),
        ilu_size_exceeded=bool(ilu_size_exceeded),
        fill_factor=float(fill_factor),
        drop_tol=float(drop_tol),
        diag_pivot=float(diag_pivot),
        permc_requested=str(permc_requested),
        permc_candidates=tuple(str(candidate) for candidate in permc_candidates),
        permc_spec=str(permc_candidates[0]),
        scale_norm=str(scale_norm),
        max_scale=float(max_scale),
        progress=bool(progress),
        lu_large_prefill_size=int(lu_large_prefill_size),
        lu_prefill_safety_factor=float(lu_prefill_safety_factor),
    )


def active_fortran_v3_reduced_permc_candidates(*, requested: str, factor_kind: str) -> tuple[str, ...]:
    """Return SuperLU ordering candidates for the active Fortran-v3 factor.

    ``RCM`` is implemented by an explicit symmetric permutation before calling
    SuperLU with ``NATURAL`` ordering. This mirrors SFINCS Fortran v3's PETSc
    serial sparse-direct fallback, where ``MATORDERINGRCM`` is requested for
    the preconditioner factor.
    """

    valid = ("RCM", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
    requested_use = str(requested or "").strip().upper()
    if requested_use in valid:
        return (requested_use,)
    if requested_use and requested_use not in {"AUTO", "DEFAULT"}:
        return ("COLAMD",)
    if str(factor_kind).strip().lower() == "lu":
        return ("NATURAL", "COLAMD")
    return ("COLAMD",)


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or int(default))
    except ValueError:
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, "")).strip() or float(default))
    except ValueError:
        return float(default)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


__all__ = [
    "resolve_active_fortran_v3_reduced_factor_policy",
    "active_fortran_v3_reduced_permc_candidates",
    "ActiveFortranV3ReducedFactorPolicy",
    "resolve_active_symbolic_frontal_policy",
    "ActiveSymbolicFrontalPolicy",
    "ActiveSymbolicBlockSchurPolicy",
    "ActiveSymbolicSuperblockPolicy",
    "resolve_active_symbolic_block_schur_policy",
    "resolve_active_symbolic_superblock_policy",
]
