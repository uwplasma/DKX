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


__all__ = [
    "ActiveSymbolicBlockSchurPolicy",
    "ActiveSymbolicSuperblockPolicy",
    "resolve_active_symbolic_block_schur_policy",
    "resolve_active_symbolic_superblock_policy",
]
