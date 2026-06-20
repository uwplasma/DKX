"""Setup helpers for RHSMode=1 profile-response solves.

The solve driver still owns the numerical solve loop.  This module keeps the
early setup decisions pure and directly testable: GMRES environment overrides,
geometry hints, tolerance tightening, and solve-method classification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
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
class RHS1PhysicsFlagSetup:
    """Physics flags that affect RHSMode=1 sparse/preconditioner policy."""

    use_dkes: bool
    include_xdot_sparse_pc: bool
    include_electric_field_xi_sparse_pc: bool
    er_abs_sparse_pc: float


@dataclass(frozen=True)
class RHS1DKESAdjustmentSetup:
    """Tolerance and GMRES-budget adjustments for DKES RHSMode=1 lanes."""

    tol: float
    restart: int
    maxiter: int | None
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class RHS1PostActiveSolvePolicySetup:
    """Solve method and budget after active-DOF size is known."""

    restart: int
    maxiter: int | None
    solve_method: str
    tokamak_pas: bool
    pas_large_bicgstab_fastpath: bool
    pas_large_fastpath_min: int
    messages: tuple[tuple[int, str], ...]


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
class RHS1InitialRouteSetup:
    """Early RHSMode=1 solver routing state used before active-DOF setup."""

    method_flags: SolveMethodRequestFlags
    use_implicit_requested: bool
    structured_eparallel_abs: float
    structured_auto_allowed: bool
    structured_sharded_multidevice: bool


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


@dataclass(frozen=True)
class RHS1DomainDecompositionSetup:
    """Parsed block/overlap sizes for RHSMode=1 line-Schwarz preconditioners."""

    sharded_axis: str | None
    patch_dof_target: int
    sum_nxi: int
    block_theta: int
    block_zeta: int
    overlap_theta: int | None
    overlap_zeta: int | None
    n_theta: int
    n_zeta: int

    def block(self, axis: str) -> int:
        """Return the clamped block size for ``theta`` or ``zeta``."""

        if axis == "theta":
            return int(self.block_theta)
        if axis == "zeta":
            return int(self.block_zeta)
        raise ValueError(f"unsupported RHSMode=1 DD axis: {axis!r}")

    def overlap(self, axis: str, *, default: int) -> int:
        """Return the clamped overlap, using ``default`` when no override exists."""

        if axis == "theta":
            n = int(self.n_theta)
            value = self.overlap_theta
        elif axis == "zeta":
            n = int(self.n_zeta)
            value = self.overlap_zeta
        else:
            raise ValueError(f"unsupported RHSMode=1 DD axis: {axis!r}")
        overlap = int(default) if value is None else int(value)
        return max(0, min(max(0, n - 1), int(overlap)))


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


def _nml_bool(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return bool(int(value))
    if isinstance(value, Real):
        return bool(float(value))
    if isinstance(value, str):
        return value.strip().lower() in {"t", "true", "1", "yes", ".true.", ".t."}
    return False


def _nml_abs_float(group: Mapping[str, object], key: str) -> float:
    value = _nml_get(group, key, None)
    try:
        return abs(float(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


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


def resolve_rhs1_physics_flag_setup(nml: Any) -> RHS1PhysicsFlagSetup:
    """Resolve RHSMode=1 physics flags from namelist spelling variants."""

    phys_params = nml.group("physicsParameters")
    use_dkes = _nml_bool(
        _nml_get(
            phys_params,
            "useDKESExBDrift",
            _nml_get(
                phys_params,
                "useDKESExBdrift",
                _nml_get(phys_params, "use_dkes_exb_drift", None),
            ),
        )
    )
    include_xdot = _nml_bool(_nml_get(phys_params, "includeXDotTerm", None))
    include_electric_field_xi = _nml_bool(_nml_get(phys_params, "includeElectricFieldTermInXiDot", None))
    er_abs = max(
        _nml_abs_float(phys_params, "Er"),
        _nml_abs_float(phys_params, "dPhiHatdpsiHat"),
        _nml_abs_float(phys_params, "dPhiHatdpsiN"),
        _nml_abs_float(phys_params, "dPhiHatdrHat"),
        _nml_abs_float(phys_params, "dPhiHatdrN"),
    )
    return RHS1PhysicsFlagSetup(
        use_dkes=bool(use_dkes),
        include_xdot_sparse_pc=bool(include_xdot),
        include_electric_field_xi_sparse_pc=bool(include_electric_field_xi),
        er_abs_sparse_pc=float(er_abs),
    )


def resolve_rhs1_dkes_adjustment_setup(
    *,
    op: Any,
    tol: float,
    fp_tol: float,
    restart: int,
    maxiter: int | None,
    restart_env_forced: bool,
    maxiter_env_forced: bool,
    use_dkes: bool,
    dkes_gmres_budget: Any,
    env: Mapping[str, str] | None = None,
) -> RHS1DKESAdjustmentSetup:
    """Apply DKES-specific tolerance and GMRES-budget rules."""

    tol_use = float(tol)
    restart_use = int(restart)
    maxiter_use = None if maxiter is None else int(maxiter)
    messages: list[tuple[int, str]] = []

    if (
        int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and bool(use_dkes)
        and float(fp_tol) > 0.0
    ):
        tol_old = float(tol_use)
        tol_use = min(float(tol_use), float(fp_tol))
        if float(tol_use) < tol_old:
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: FP DKES tol tightened "
                    f"{tol_old:.1e} -> {float(tol_use):.1e}",
                )
            )

    if op.fblock.pas is not None and bool(use_dkes):
        restart_use, maxiter_use, restart_defaulted, maxiter_defaulted = dkes_gmres_budget(
            restart=int(restart_use),
            maxiter=maxiter_use,
            restart_forced=bool(restart_env_forced),
            maxiter_forced=bool(maxiter_env_forced),
            restart_cap_env=_env_value(env, "SFINCS_JAX_RHSMODE1_DKES_RESTART_CAP"),
        )
        if bool(restart_env_forced) or bool(maxiter_env_forced):
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: PAS DKES respecting explicit GMRES budget "
                    f"restart={int(restart_use)} maxiter={maxiter_use}",
                )
            )
        elif bool(restart_defaulted) or bool(maxiter_defaulted):
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: PAS DKES default GMRES budget "
                    f"restart={int(restart_use)} maxiter={maxiter_use}",
                )
            )

    return RHS1DKESAdjustmentSetup(
        tol=float(tol_use),
        restart=int(restart_use),
        maxiter=maxiter_use,
        messages=tuple(messages),
    )


def resolve_rhs1_post_active_solve_policy_setup(
    *,
    op: Any,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    active_size: int,
    use_active_dof_mode: bool,
    full_precond_requested: bool,
    geom_scheme: int,
    dense_backend_allowed: bool,
    backend: str,
    sharded_axis_hint: str | None,
    device_count: int,
    env: Mapping[str, str] | None = None,
) -> RHS1PostActiveSolvePolicySetup:
    """Resolve active-size-dependent RHSMode=1 solve-method policy."""

    restart_use = int(restart)
    maxiter_use = None if maxiter is None else int(maxiter)
    solve_method_use = str(solve_method)
    messages: list[tuple[int, str]] = []

    pas_full_gmres_max = _read_int(env, "SFINCS_JAX_PAS_FULL_GMRES_MAX", 1200)
    if op.fblock.pas is not None and int(active_size) <= max(0, int(pas_full_gmres_max)):
        restart_use = max(int(restart_use), int(active_size))
        if maxiter_use is None or int(maxiter_use) < int(active_size):
            maxiter_use = int(active_size)

    full_precond_env = _env_value(env, "SFINCS_JAX_RHSMODE1_FULL_PRECOND").lower()
    if full_precond_env in {"0", "false", "no", "off"}:
        full_precond_mode = "off"
    elif full_precond_env in {"dense", "dense_ksp"}:
        full_precond_mode = full_precond_env
    else:
        full_precond_mode = "auto"

    full_precond_dense_max = _read_int(env, "SFINCS_JAX_RHSMODE1_FULL_PRECOND_DENSE_MAX", 2500)
    full_precond_size = int(active_size) if bool(use_active_dof_mode) else int(op.total_size)
    method_auto = str(solve_method_use).strip().lower() in {"auto", "default"}
    auto_dense_full_precond = bool(
        full_precond_mode == "auto"
        and bool(full_precond_requested)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.constraint_scheme) != 0
        and int(full_precond_dense_max) > 0
        and int(full_precond_size) <= int(full_precond_dense_max)
        and bool(dense_backend_allowed)
        and method_auto
    )
    if (
        bool(full_precond_requested)
        and (full_precond_mode in {"dense", "dense_ksp"} or auto_dense_full_precond)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(full_precond_dense_max) > 0
        and int(full_precond_size) <= int(full_precond_dense_max)
        and method_auto
    ):
        solve_method_use = "dense" if (full_precond_mode != "dense_ksp" or bool(use_active_dof_mode)) else "dense_ksp"
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: full preconditioner requested; "
                f"using solve_method={solve_method_use} (size={int(full_precond_size)})",
            )
        )
    elif (
        bool(full_precond_requested)
        and full_precond_mode == "auto"
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.constraint_scheme) != 0
        and int(full_precond_dense_max) > 0
        and int(full_precond_size) <= int(full_precond_dense_max)
        and (not bool(dense_backend_allowed))
        and method_auto
    ):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: full preconditioner requested; "
                f"skipping dense auto mode on backend={backend}",
            )
        )

    tokamak_pas = bool(
        op.fblock.pas is not None
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and (int(geom_scheme) == 1 or int(op.n_zeta) <= 5)
    )
    pas_large_fastpath_env = _env_value(env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH").lower()
    pas_large_fastpath_min = max(1, _read_int(env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MIN", 80000))
    pas_large_fastpath_max = _read_int(env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MAX", 300000)
    pas_large_fastpath_auto = pas_large_fastpath_env in {"", "auto"}
    pas_large_fastpath_on = pas_large_fastpath_env in {"1", "true", "yes", "on"}
    pas_large_fastpath_off = pas_large_fastpath_env in {"0", "false", "no", "off"}
    pas_large_bicgstab_fastpath = bool(
        (pas_large_fastpath_on or pas_large_fastpath_auto)
        and (not pas_large_fastpath_off)
        and tokamak_pas
        and int(geom_scheme) == 1
        and int(op.n_species) == 1
        and int(op.n_zeta) == 1
        and int(active_size) >= int(pas_large_fastpath_min)
        and (int(pas_large_fastpath_max) <= 0 or int(active_size) <= int(pas_large_fastpath_max))
    )

    if int(op.rhs_mode) == 1 and str(solve_method_use).strip().lower() in {"auto", "default"}:
        sharded_multidevice_hint = sharded_axis_hint in {"theta", "zeta"} and int(device_count) > 1
        if pas_large_bicgstab_fastpath:
            solve_method_use = "bicgstab"
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: enabling PAS-large BiCGStab fastpath "
                    f"(active_size={int(active_size)} >= {int(pas_large_fastpath_min)})",
                )
            )
        elif sharded_multidevice_hint:
            solve_method_use = "auto"
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: preserving auto solver selection for "
                    f"multi-device sharded axis={sharded_axis_hint}",
                )
            )
        else:
            solve_method_use = "incremental"

    return RHS1PostActiveSolvePolicySetup(
        restart=int(restart_use),
        maxiter=maxiter_use,
        solve_method=str(solve_method_use),
        tokamak_pas=bool(tokamak_pas),
        pas_large_bicgstab_fastpath=bool(pas_large_bicgstab_fastpath),
        pas_large_fastpath_min=int(pas_large_fastpath_min),
        messages=tuple(messages),
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


def resolve_rhs1_initial_route_setup(
    *,
    nml: Any,
    op: Any,
    solve_method: str,
    xblock_active_dof_env: str,
    use_implicit: bool,
    force_krylov: bool,
    sharded_axis: str | None,
    backend: str,
    device_count: int,
    structured_auto_allowed: Callable[..., bool],
) -> RHS1InitialRouteSetup:
    """Resolve early RHSMode=1 routing shared by auto and explicit host paths.

    The solve driver still builds the operator and owns the selected solve
    branches. This helper only classifies the user request and evaluates the
    structured full-CSR auto-admission policy from already available inputs.
    """

    method_flags = resolve_solve_method_request_flags(
        solve_method=solve_method,
        xblock_active_dof_env=xblock_active_dof_env,
    )
    structured_eparallel_abs = 0.0
    structured_sharded_multidevice = False
    structured_auto_allowed_value = False
    if not method_flags.structured_full_csr_explicit_requested:
        phys_params = nml.group("physicsParameters")
        values: list[float] = []
        for key in ("EParallelHat", "eParallelHat", "EPARALLELHAT"):
            value = phys_params.get(key, phys_params.get(key.upper(), None))
            try:
                values.append(abs(float(value)) if value is not None else 0.0)
            except (TypeError, ValueError):
                values.append(0.0)
        structured_eparallel_abs = max(values, default=0.0)
        structured_sharded_multidevice = (
            sharded_axis in {"theta", "zeta"} and int(device_count) > 1
        )
        structured_auto_allowed_value = bool(
            (not bool(force_krylov))
            and structured_auto_allowed(
                op=op,
                active_size=int(op.total_size),
                use_implicit=bool(use_implicit),
                solve_method_kind=method_flags.kind,
                backend=str(backend),
                eparallel_abs=float(structured_eparallel_abs),
            )
        )
    return RHS1InitialRouteSetup(
        method_flags=method_flags,
        use_implicit_requested=bool(use_implicit),
        structured_eparallel_abs=float(structured_eparallel_abs),
        structured_auto_allowed=bool(structured_auto_allowed_value),
        structured_sharded_multidevice=bool(structured_sharded_multidevice),
    )


def _read_int_value(raw: object, default: int) -> int:
    try:
        text = str(raw).strip()
        return int(text) if text else int(default)
    except (TypeError, ValueError):
        return int(default)


def resolve_rhs1_domain_decomposition_setup(
    *,
    n_theta: int,
    n_zeta: int,
    sum_nxi: int,
    distributed_env: str,
    device_count: int,
    auto_axis: str | None,
    theta_block_env: str,
    zeta_block_env: str,
    theta_overlap_env: str,
    zeta_overlap_env: str,
    overlap_env: str,
    patch_dof_target_env: str,
) -> RHS1DomainDecompositionSetup:
    """Resolve RHSMode=1 domain-decomposition block and overlap settings."""

    from ...rhs1_domain_decomposition import _rhs1_dd_auto_block_size

    dist = str(distributed_env or "").strip().lower()
    if dist in {"0", "false", "no", "off"} or int(device_count) <= 1:
        axis: str | None = None
    elif dist in {"theta", "zeta"}:
        axis = dist
    else:
        axis_auto = str(auto_axis or "").strip().lower()
        axis = axis_auto if axis_auto in {"theta", "zeta"} else None

    patch_dof_target = max(128, _read_int_value(patch_dof_target_env, 1200))
    n_dev = max(1, int(device_count))
    sum_nxi_use = max(1, int(sum_nxi))

    def _block(axis_name: str, *, raw: str, n: int) -> int:
        block = _read_int_value(raw, 0)
        if block <= 0 and axis == axis_name:
            block = _rhs1_dd_auto_block_size(
                n=int(n),
                n_dev=n_dev,
                sum_nxi=sum_nxi_use,
                dof_target=patch_dof_target,
            )
        if block <= 0:
            block = 8
        return max(1, min(max(1, int(n)), int(block)))

    block_theta = _block("theta", raw=theta_block_env, n=int(n_theta))
    block_zeta = _block("zeta", raw=zeta_block_env, n=int(n_zeta))

    def _overlap(axis_name: str, *, raw_axis: str, raw_generic: str, block: int) -> int | None:
        raw = str(raw_axis or "").strip() or str(raw_generic or "").strip()
        overlap = _read_int_value(raw, -1)
        if overlap < 0 and axis == axis_name:
            overlap = 2 if int(block) >= 4 else 1
            while overlap > 1 and int(block + 2 * overlap) * sum_nxi_use > patch_dof_target:
                overlap -= 1
            return max(1, int(overlap))
        if overlap < 0:
            return None
        return int(overlap)

    return RHS1DomainDecompositionSetup(
        sharded_axis=axis,
        patch_dof_target=int(patch_dof_target),
        sum_nxi=int(sum_nxi_use),
        block_theta=int(block_theta),
        block_zeta=int(block_zeta),
        overlap_theta=_overlap("theta", raw_axis=theta_overlap_env, raw_generic=overlap_env, block=block_theta),
        overlap_zeta=_overlap("zeta", raw_axis=zeta_overlap_env, raw_generic=overlap_env, block=block_zeta),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
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
    "RHS1DomainDecompositionSetup",
    "RHS1GmresBudgetSetup",
    "RHS1InitialRouteSetup",
    "RHS1PhysicsFlagSetup",
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
    "resolve_rhs1_domain_decomposition_setup",
    "resolve_rhs1_gmres_budget_setup",
    "resolve_rhs1_initial_route_setup",
    "resolve_rhs1_physics_flag_setup",
    "resolve_rhs1_preconditioner_option_setup",
    "resolve_rhs1_tolerance_setup",
    "resolve_solve_method_request_flags",
)
