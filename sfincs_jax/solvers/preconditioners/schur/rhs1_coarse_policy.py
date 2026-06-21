"""Policy controls for RHSMode=1 active coarse preconditioners.

The numerical builders in :mod:`sfincs_jax.rhs1_full_assembly` own sparse
matrix assembly, factorization, and true-residual admission. This module owns
only the side-effect-free environment parsing and branch naming used by the
native-line plus coarse-residual preconditioner family.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

__all__ = (
    "ActiveNativeFieldSplitSparseCoarsePolicy",
    "ActiveNativeStackPolicy",
    "ActiveSparseCoarseResidualPolicy",
    "resolve_active_native_field_split_sparse_coarse_policy",
    "resolve_active_native_stack_policy",
    "resolve_active_sparse_coarse_residual_policy",
)


@dataclass(frozen=True)
class ActiveNativeStackPolicy:
    """Resolved controls for the bounded native-line/SCHWARZ/coarse stack."""

    base_budget_fraction: float
    base_budget_nbytes: int
    schwarz_requested: bool
    schwarz_max_size: int
    max_coarse_size: int
    coarse_solver_mode: str


@dataclass(frozen=True)
class ActiveNativeFieldSplitSparseCoarsePolicy:
    """Resolved controls for native field-split plus sparse coarse correction."""

    requested_kind: str
    requested_kind_normalized: str
    output_kind: str
    requested_base_kind: str
    is_multiline: bool
    is_angular_only: bool
    is_coupled_kinetic: bool
    max_coarse_size: int
    coarse_solver_mode: str
    admission_probes: int
    admission_max_relative_residual: float
    admission_min_improvement: float


@dataclass(frozen=True)
class ActiveSparseCoarseResidualPolicy:
    """Resolved controls for tail/Schwarz/filtered sparse coarse correction."""

    requested_kind: str
    requested_kind_normalized: str
    base_kind: str
    output_kind: str
    max_coarse_size: int
    coarse_solver_mode: str


def resolve_active_native_stack_policy(
    *,
    max_factor_nbytes: int,
    env: Mapping[str, str] | None = None,
) -> ActiveNativeStackPolicy:
    """Resolve bounded native-stack memory and coarse-equation controls."""

    env_map = os.environ if env is None else env
    base_budget_fraction = min(
        max(
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION",
                    0.75,
                )
            ),
            0.05,
        ),
        1.0,
    )
    default_coarse_size = _env_int(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE",
        640,
    )
    return ActiveNativeStackPolicy(
        base_budget_fraction=float(base_budget_fraction),
        base_budget_nbytes=max(1, int(float(max_factor_nbytes) * float(base_budget_fraction))),
        schwarz_requested=_env_bool(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ", False),
        schwarz_max_size=int(
            _env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE", 100_000)
        ),
        max_coarse_size=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE",
                    int(default_coarse_size),
                )
            ),
        ),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER", "least_squares"),
            galerkin_aliases=("galerkin", "petrov_galerkin", "ztaz"),
            least_squares_aliases=(),
            default="least_squares",
        ),
    )


def resolve_active_native_field_split_sparse_coarse_policy(
    *,
    requested_kind: str,
    env: Mapping[str, str] | None = None,
) -> ActiveNativeFieldSplitSparseCoarsePolicy:
    """Resolve routing and admission controls for native field-split coarse paths."""

    env_map = os.environ if env is None else env
    requested_kind_l = _normalize_token(requested_kind)
    is_multiline = "multiline" in requested_kind_l or "xell_angular" in requested_kind_l
    is_angular_only = "angular" in requested_kind_l and not bool(is_multiline)
    is_coupled_kinetic = "coupled_kinetic" in requested_kind_l or "dominant_kinetic" in requested_kind_l

    output_kind = "active_native_xell_field_split_sparse_coarse"
    if bool(is_coupled_kinetic):
        output_kind = "active_coupled_kinetic_field_split_sparse_coarse"
    if bool(is_angular_only):
        output_kind = "active_angular_line_field_split_sparse_coarse"
    if bool(is_multiline):
        output_kind = "active_multiline_field_split_sparse_coarse"

    requested_base_kind = "active_multiline_xell_angular"
    if bool(is_coupled_kinetic):
        requested_base_kind = "active_coupled_kinetic_block"
    elif not bool(is_multiline):
        requested_base_kind = (
            "active_angular_line"
            if bool(is_angular_only)
            else _normalize_token(
                env_map.get(
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LINE_FIELD_SPLIT_BASE",
                    "active_native_xell",
                )
            )
            or "active_native_xell"
        )

    sparse_default = _env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640)
    coarse_size_key = (
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE"
        if bool(is_coupled_kinetic)
        else "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE"
    )

    return ActiveNativeFieldSplitSparseCoarsePolicy(
        requested_kind=str(requested_kind),
        requested_kind_normalized=str(requested_kind_l),
        output_kind=str(output_kind),
        requested_base_kind=str(requested_base_kind),
        is_multiline=bool(is_multiline),
        is_angular_only=bool(is_angular_only),
        is_coupled_kinetic=bool(is_coupled_kinetic),
        max_coarse_size=max(1, int(_env_int(env_map, coarse_size_key, int(sparse_default)))),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER", "least_squares"),
            galerkin_aliases=("galerkin", "petrov_galerkin", "ztaz"),
            least_squares_aliases=(),
            default="least_squares",
        ),
        admission_probes=max(
            1,
            int(
                _env_int(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES",
                    4,
                )
            ),
        ),
        admission_max_relative_residual=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        admission_min_improvement=max(
            0.0,
            float(
                _env_float(
                    env_map,
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
    )


def resolve_active_sparse_coarse_residual_policy(
    *,
    requested_kind: str,
    env: Mapping[str, str] | None = None,
) -> ActiveSparseCoarseResidualPolicy:
    """Resolve base-preconditioner and coarse-equation controls for sparse coarse paths."""

    env_map = os.environ if env is None else env
    requested_norm = _normalize_token(requested_kind)
    if "scaled_ilu" in requested_norm or "equilibrated_ilu" in requested_norm or "rowcol_ilu" in requested_norm:
        base_kind = "active_scaled_ilu"
    elif "schwarz" in requested_norm or "ras" in requested_norm:
        base_kind = "active_overlap_schwarz"
    elif "filtered" in requested_norm:
        base_kind = "active_filtered_sparse_factor"
    elif "xblock" in requested_norm:
        base_kind = "active_xblock"
    else:
        base_kind = "active_diagonal_schur"

    default_coarse_solver = "least_squares" if str(base_kind) == "active_filtered_sparse_factor" else "galerkin"
    return ActiveSparseCoarseResidualPolicy(
        requested_kind=str(requested_kind),
        requested_kind_normalized=str(requested_norm),
        base_kind=str(base_kind),
        output_kind=(
            "active_filtered_sparse_coarse"
            if str(base_kind) == "active_filtered_sparse_factor"
            else "active_tail_sparse_coarse"
        ),
        max_coarse_size=max(
            1,
            int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640)),
        ),
        coarse_solver_mode=_coarse_solver_mode(
            env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER", default_coarse_solver),
            galerkin_aliases=(),
            least_squares_aliases=("least_squares", "normal", "normal_equations"),
            default=str(default_coarse_solver),
        ),
    )


def _normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _coarse_solver_mode(
    value: object,
    *,
    galerkin_aliases: tuple[str, ...],
    least_squares_aliases: tuple[str, ...],
    default: str,
) -> str:
    token = _normalize_token(value)
    if not token:
        token = _normalize_token(default)
    if token in set(galerkin_aliases):
        return "galerkin"
    if token in set(least_squares_aliases):
        return "least_squares"
    if _normalize_token(default) == "galerkin":
        return "galerkin"
    return "least_squares"


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)
