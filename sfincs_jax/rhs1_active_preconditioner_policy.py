"""Policy helpers for active-projected RHSMode=1 preconditioner selection.

The production RHSMode=1 active system has several host-side preconditioner
families. This module owns only the environment-driven auto-candidate policy so
the matrix builder can focus on dispatching candidates and measuring residuals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

_DEFAULT_AUTO_CANDIDATES = (
    "active_fortran_v3_reduced_lu",
    "active_fortran_v3_reduced_native_stack",
    "active_symbolic_frontal_schur_lu",
    "active_symbolic_superblock_lu",
    "active_symbolic_block_schur_lu",
    "active_schwarz_sparse_coarse",
    "active_global_field_split_schur",
    "active_xblock_ell_band_schur",
    "active_ell_band_schur",
    "active_bounded_native_stack",
    "active_xblock",
    "active_diagonal_schur",
    "active_spilu",
    "jacobi",
)

_DEFAULT_LARGE_AUTO_CANDIDATES = (
    "active_fortran_v3_reduced_native_stack",
    "active_symbolic_frontal_schur_lu",
    "active_symbolic_superblock_lu",
    "active_coupled_kinetic_field_split_sparse_coarse",
    "active_symbolic_block_schur_lu",
    "active_fortran_v3_reduced_lu",
)

_LARGE_FALLBACK_CANDIDATES = frozenset(
    {
        "active_diagonal_schur",
        "active_diag_schur",
        "active_tail_schur",
        "active_constraint_tail_schur",
        "active_field_split",
        "active_field_split_tail",
        "jacobi",
        "diagonal",
    }
)


@dataclass(frozen=True)
class ActiveProjectedPreconditionerAutoPolicy:
    """Resolved auto-policy for active projected RHSMode=1 preconditioners."""

    candidates: tuple[str, ...]
    candidates_requested: tuple[str, ...]
    skipped_large_fallbacks: tuple[str, ...]
    large_fallback_size: int
    large_default_used: bool
    log_progress: bool


def resolve_active_projected_preconditioner_auto_policy(
    *,
    matrix_size: int,
    env: Mapping[str, str] | None = None,
) -> ActiveProjectedPreconditionerAutoPolicy:
    """Resolve the active-system preconditioner auto ladder.

    Large systems avoid diagonal/Jacobi fallbacks by default because those paths
    can spend substantial time while providing weak residual reduction. Users
    can still opt in through the documented environment override.
    """

    env_map = os.environ if env is None else env
    size = int(matrix_size)
    large_fallback_size = max(
        1,
        int(_env_int(env_map, "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", 300000)),
    )
    candidate_env_override = env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES")
    candidate_env = (
        candidate_env_override
        if candidate_env_override is not None
        else ",".join(_DEFAULT_AUTO_CANDIDATES)
    )
    large_default_used = False
    if candidate_env_override is None and size >= int(large_fallback_size):
        candidate_env = env_map.get(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES",
            ",".join(_DEFAULT_LARGE_AUTO_CANDIDATES),
        )
        large_default_used = True

    candidates = _parse_candidates(candidate_env)
    if not candidates:
        candidates = ("active_diagonal_schur", "jacobi")
    candidates_requested = tuple(candidates)

    allow_large_fallback = _env_bool(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_ALLOW_LARGE_DIAGONAL_FALLBACK",
        False,
    )
    skipped_large_fallbacks: tuple[str, ...] = ()
    if size >= int(large_fallback_size) and not bool(allow_large_fallback):
        skipped_large_fallbacks = tuple(
            candidate for candidate in candidates if candidate in _LARGE_FALLBACK_CANDIDATES
        )
        candidates = tuple(
            candidate for candidate in candidates if candidate not in _LARGE_FALLBACK_CANDIDATES
        )

    log_progress = _env_bool(
        env_map,
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_PROGRESS",
        bool(large_default_used) or size >= int(large_fallback_size),
    )
    return ActiveProjectedPreconditionerAutoPolicy(
        candidates=tuple(candidates),
        candidates_requested=tuple(candidates_requested),
        skipped_large_fallbacks=tuple(skipped_large_fallbacks),
        large_fallback_size=int(large_fallback_size),
        large_default_used=bool(large_default_used),
        log_progress=bool(log_progress),
    )


def _parse_candidates(candidate_env: str) -> tuple[str, ...]:
    return tuple(
        item.strip().lower().replace("-", "_")
        for item in str(candidate_env).split(",")
        if item.strip()
    )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or int(default))
    except ValueError:
        return int(default)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


__all__ = [
    "ActiveProjectedPreconditionerAutoPolicy",
    "resolve_active_projected_preconditioner_auto_policy",
]
