"""Setup helpers for RHSMode=1 profile-response solves.

The solve driver still owns the numerical solve loop.  This module keeps the
early setup decisions pure and directly testable: GMRES environment overrides,
geometry hints, tolerance tightening, and solve-method classification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


SPARSE_HOST_DIRECT_SOLVE_METHODS = frozenset({"sparse_host", "host_sparse", "sparse_host_lu"})
SPARSE_HOST_SAFE_SOLVE_METHODS = frozenset(
    {
        "sparse_host_safe",
        "safe_sparse_host",
        "sparse_host_or_petsc_compat",
    }
)
SPARSE_HOST_PC_GMRES_SOLVE_METHODS = frozenset(
    {
        "sparse_pc_gmres",
        "sparse_host_gmres",
        "sparse_host_pc",
        "host_sparse_pc_gmres",
        "petsc_host",
        "petsc_host_gmres",
        "fortran_reduced_pc_gmres",
        "fortran_reduced_sparse_pc_gmres",
        "fortran_like_pc_gmres",
        "petsc_like_pc_gmres",
    }
)
SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS = frozenset(
    {
        "fortran_reduced_pc_gmres",
        "fortran_reduced_sparse_pc_gmres",
        "fortran_like_pc_gmres",
        "petsc_like_pc_gmres",
    }
)
SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS = frozenset(
    {
        "xblock_sparse_pc_gmres",
        "sparse_xblock_pc_gmres",
        "xblock_host_pc_gmres",
        "host_xblock_pc_gmres",
    }
)
STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS = frozenset(
    {
        "structured_csr",
        "structured_full_csr",
        "host_structured_csr",
        "host_full_csr",
        "no_probe_csr",
        "full_csr_host_gmres",
        "structured_full_csr_host_gmres",
    }
)
SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS = frozenset(
    {
        "sparse_lsmr",
        "sparse_host_lsmr",
        "sparse_lsqr",
        "sparse_host_lsqr",
        "minimum_norm",
        "sparse_minimum_norm",
        "petsc_compat",
        "sparse_petsc_compat",
        "petsc_minimum_norm",
    }
)
SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS = frozenset(
    {
        "petsc_compat",
        "sparse_petsc_compat",
        "petsc_minimum_norm",
    }
)


@dataclass(frozen=True)
class RHS1GmresBudgetSetup:
    """GMRES restart/maxiter after explicit environment overrides."""

    restart: int
    maxiter: int | None
    restart_env_forced: bool
    maxiter_env_forced: bool


@dataclass(frozen=True)
class RHS1ToleranceSetup:
    """Tolerance state after RHSMode=1 FP/PAS tightening rules."""

    tol: float
    fp_tol: float
    fp_tol_min_size: int
    fp_tightened: bool
    fp_previous_tol: float | None
    pas_tol: float | None
    pas_tightened: bool
    pas_previous_tol: float | None


@dataclass(frozen=True)
class SolveMethodRequestFlags:
    """Normalized solve-method token plus coarse branch classifications."""

    kind: str
    sparse_host_requested: bool
    sparse_host_safe_requested: bool
    sparse_pc_gmres_requested: bool
    sparse_minimum_norm_requested: bool
    sparse_host_like_requested: bool
    xblock_active_dof_requested: bool
    structured_full_csr_explicit_requested: bool


@dataclass(frozen=True)
class RHS1PreconditionerOptionSetup:
    """Parsed RHSMode=1 preconditioner options and PAS projection admission."""

    preconditioner_species: int
    preconditioner_x: int
    preconditioner_x_min_l: int
    preconditioner_xi: int
    full_preconditioner_requested: bool
    geom_scheme: int
    pas_project_mode: str
    pas_project_enabled: bool
    use_pas_projection: bool
    use_active_dof_mode: bool


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    source = env if env is not None else {}
    return str(source.get(key, "")).strip()


def _read_float(env: Mapping[str, str] | None, key: str, default: float) -> float:
    raw = _env_value(env, key)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _read_int(env: Mapping[str, str] | None, key: str, default: int) -> int:
    raw = _env_value(env, key)
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _nml_get(group: Mapping[str, object], key: str, default: object | None = None) -> object | None:
    if key in group:
        return group[key]
    key_upper = key.upper()
    if key_upper in group:
        return group[key_upper]
    key_lower = key.lower()
    if key_lower in group:
        return group[key_lower]
    return default


def _preconditioner_option_int(options: Mapping[str, object], key: str, default: int) -> int:
    value = options.get(key, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def resolve_rhs1_gmres_budget_setup(
    *,
    restart: int,
    maxiter: int | None,
    env: Mapping[str, str] | None = None,
) -> RHS1GmresBudgetSetup:
    """Apply explicit GMRES restart/maxiter environment overrides."""

    restart_use = int(restart)
    maxiter_use = None if maxiter is None else int(maxiter)
    restart_forced = False
    raw_restart = _env_value(env, "SFINCS_JAX_GMRES_RESTART")
    if raw_restart:
        try:
            restart_use = int(raw_restart)
            restart_forced = True
        except ValueError:
            pass
    maxiter_forced = False
    raw_maxiter = _env_value(env, "SFINCS_JAX_GMRES_MAXITER")
    if raw_maxiter:
        try:
            maxiter_use = int(raw_maxiter)
            maxiter_forced = True
        except ValueError:
            pass
    return RHS1GmresBudgetSetup(
        restart=int(restart_use),
        maxiter=maxiter_use,
        restart_env_forced=bool(restart_forced),
        maxiter_env_forced=bool(maxiter_forced),
    )


def geometry_scheme_hint_from_namelist(nml: Any) -> int:
    """Return the integer geometryScheme hint without building an operator."""

    geom_params = nml.group("geometryParameters")
    return int(
        geom_params.get(
            "GEOMETRYSCHEME",
            geom_params.get("geometryScheme", geom_params.get("geometryscheme", 0)),
        )
        or 0
    )


def equilibrium_name_hint_from_namelist(nml: Any) -> str:
    """Return a user-facing equilibrium-file basename for progress messages."""

    from pathlib import Path

    geom_params = nml.group("geometryParameters")
    eq_hint = geom_params.get(
        "EQUILIBRIUMFILE",
        geom_params.get("equilibriumFile", geom_params.get("equilibriumfile", "")),
    )
    return Path(str(eq_hint)).name if eq_hint else "VMEC equilibrium"


def resolve_rhs1_tolerance_setup(
    *,
    op: Any,
    tol: float,
    env: Mapping[str, str] | None = None,
) -> RHS1ToleranceSetup:
    """Apply RHSMode=1 FP/PAS tolerance-tightening rules.

    The returned ``fp_tol`` is kept because a later DKES full-FP rule reuses the
    same configured floor after active-DOF setup.
    """

    tol_use = float(tol)
    fp_tol = _read_float(env, "SFINCS_JAX_RHSMODE1_FP_TOL", 1.0e-8)
    fp_tol_min = _read_int(env, "SFINCS_JAX_RHSMODE1_FP_TOL_MIN_SIZE", 80000)
    fp_tightened = False
    fp_previous: float | None = None
    if (
        int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and int(op.total_size) >= max(1, int(fp_tol_min))
        and fp_tol > 0.0
    ):
        fp_previous = float(tol_use)
        tol_use = min(float(tol_use), float(fp_tol))
        fp_tightened = bool(float(tol_use) < float(fp_previous))

    pas_raw = _env_value(env, "SFINCS_JAX_RHSMODE1_PAS_TOL")
    try:
        pas_tol = float(pas_raw) if pas_raw else None
    except ValueError:
        pas_tol = None
    pas_tightened = False
    pas_previous: float | None = None
    if (
        int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and int(op.constraint_scheme) == 2
        and pas_tol is not None
        and pas_tol > 0.0
    ):
        pas_previous = float(tol_use)
        tol_use = min(float(tol_use), float(pas_tol))
        pas_tightened = bool(float(tol_use) < float(pas_previous))

    return RHS1ToleranceSetup(
        tol=float(tol_use),
        fp_tol=float(fp_tol),
        fp_tol_min_size=int(fp_tol_min),
        fp_tightened=bool(fp_tightened),
        fp_previous_tol=fp_previous,
        pas_tol=pas_tol,
        pas_tightened=bool(pas_tightened),
        pas_previous_tol=pas_previous,
    )


def normalize_profile_solve_method_kind(solve_method: str) -> str:
    """Normalize user solve-method tokens to the internal underscore style."""

    return str(solve_method).strip().lower().replace("-", "_")


def resolve_solve_method_request_flags(
    *,
    solve_method: str,
    xblock_active_dof_env: str = "",
) -> SolveMethodRequestFlags:
    """Classify a profile-response solve method into coarse solver lanes."""

    kind = normalize_profile_solve_method_kind(solve_method)
    sparse_host_requested = kind in SPARSE_HOST_DIRECT_SOLVE_METHODS
    sparse_host_safe_requested = kind in SPARSE_HOST_SAFE_SOLVE_METHODS
    sparse_pc_gmres_requested = (
        kind in SPARSE_HOST_PC_GMRES_SOLVE_METHODS or kind in SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
    )
    sparse_minimum_norm_requested = kind in SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS
    sparse_host_like_requested = bool(
        sparse_host_requested
        or sparse_host_safe_requested
        or sparse_pc_gmres_requested
        or sparse_minimum_norm_requested
    )
    active_env = str(xblock_active_dof_env or "").strip().lower()
    xblock_active_dof_requested = bool(
        kind in SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
        and active_env in {"1", "true", "yes", "on"}
    )
    return SolveMethodRequestFlags(
        kind=str(kind),
        sparse_host_requested=bool(sparse_host_requested),
        sparse_host_safe_requested=bool(sparse_host_safe_requested),
        sparse_pc_gmres_requested=bool(sparse_pc_gmres_requested),
        sparse_minimum_norm_requested=bool(sparse_minimum_norm_requested),
        sparse_host_like_requested=bool(sparse_host_like_requested),
        xblock_active_dof_requested=bool(xblock_active_dof_requested),
        structured_full_csr_explicit_requested=bool(kind in STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS),
    )


def resolve_rhs1_preconditioner_option_setup(
    *,
    nml: Any,
    op: Any,
    sparse_host_like_requested: bool,
    use_active_dof_mode: bool,
    env: Mapping[str, str] | None = None,
) -> RHS1PreconditionerOptionSetup:
    """Parse preconditioner options and PAS projection admission.

    This is intentionally pure: it does not build active indices or
    preconditioners.  It only mirrors the driver's historical setup decisions.
    """

    precond_opts = nml.group("preconditionerOptions")
    preconditioner_species = _preconditioner_option_int(precond_opts, "PRECONDITIONER_SPECIES", 1)
    preconditioner_x = _preconditioner_option_int(precond_opts, "PRECONDITIONER_X", 1)
    preconditioner_x_min_l = _preconditioner_option_int(precond_opts, "PRECONDITIONER_X_MIN_L", 0)
    preconditioner_xi = _preconditioner_option_int(precond_opts, "PRECONDITIONER_XI", 1)
    full_precond_requested = bool(
        preconditioner_species == 0 and preconditioner_x == 0 and preconditioner_xi == 0
    )
    pas_project_env = _env_value(env, "SFINCS_JAX_PAS_PROJECT_CONSTRAINTS").lower()
    if pas_project_env in {"1", "true", "yes", "on"}:
        pas_project_mode = "on"
    elif pas_project_env in {"0", "false", "no", "off"}:
        pas_project_mode = "off"
    elif pas_project_env in {"", "auto"}:
        pas_project_mode = "auto"
    else:
        pas_project_mode = "off"
    geom_params = nml.group("geometryParameters")
    geom_scheme = int(_nml_get(geom_params, "geometryScheme", -1) or -1)
    pas_project_enabled = bool(
        pas_project_mode == "on"
        or (
            pas_project_mode == "auto"
            and not full_precond_requested
            and geom_scheme != 1
        )
    )
    use_pas_projection = bool(
        (not sparse_host_like_requested)
        and pas_project_enabled
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.constraint_scheme) == 2
        and op.fblock.pas is not None
        and int(op.phi1_size) == 0
    )
    if use_pas_projection:
        pas_project_min = _read_int(env, "SFINCS_JAX_PAS_PROJECT_MIN", 2000)
        if int(op.total_size) < max(0, int(pas_project_min)):
            use_pas_projection = False
    use_active = bool(use_active_dof_mode or use_pas_projection)
    return RHS1PreconditionerOptionSetup(
        preconditioner_species=int(preconditioner_species),
        preconditioner_x=int(preconditioner_x),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
        preconditioner_xi=int(preconditioner_xi),
        full_preconditioner_requested=bool(full_precond_requested),
        geom_scheme=int(geom_scheme),
        pas_project_mode=str(pas_project_mode),
        pas_project_enabled=bool(pas_project_enabled),
        use_pas_projection=bool(use_pas_projection),
        use_active_dof_mode=bool(use_active),
    )


__all__ = (
    "RHS1GmresBudgetSetup",
    "RHS1PreconditionerOptionSetup",
    "RHS1ToleranceSetup",
    "SPARSE_HOST_DIRECT_SOLVE_METHODS",
    "SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS",
    "SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS",
    "SPARSE_HOST_PC_GMRES_SOLVE_METHODS",
    "SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS",
    "SPARSE_HOST_SAFE_SOLVE_METHODS",
    "SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS",
    "STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS",
    "SolveMethodRequestFlags",
    "equilibrium_name_hint_from_namelist",
    "geometry_scheme_hint_from_namelist",
    "normalize_profile_solve_method_kind",
    "resolve_rhs1_gmres_budget_setup",
    "resolve_rhs1_preconditioner_option_setup",
    "resolve_rhs1_tolerance_setup",
    "resolve_solve_method_request_flags",
)
