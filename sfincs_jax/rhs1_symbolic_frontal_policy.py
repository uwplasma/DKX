"""Policy for RHSMode=1 symbolic frontal/Schur active preconditioners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


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


__all__ = [
    "ActiveSymbolicFrontalPolicy",
    "resolve_active_symbolic_frontal_policy",
]
