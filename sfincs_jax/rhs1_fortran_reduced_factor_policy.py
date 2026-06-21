"""Factorization policy for RHSMode=1 Fortran-v3-reduced preconditioners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


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
    "ActiveFortranV3ReducedFactorPolicy",
    "active_fortran_v3_reduced_permc_candidates",
    "resolve_active_fortran_v3_reduced_factor_policy",
]
