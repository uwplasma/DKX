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

import os

import jax.numpy as jnp
import numpy as np

from ...solvers.preconditioning import (
    project_constraint_scheme1_nullspace_solution_with_residual,
)
from ...solver import GMRESSolveResult
from .policies import rhs1_pas_source_zero_tolerance_from_env


SPARSE_HOST_DIRECT_SOLVE_METHODS = frozenset(
    {"sparse_host", "host_sparse", "sparse_host_lu"}
)
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
class RHS1RecycleBasisSetup:
    """Filtered recycled Krylov basis compatible with the current operator."""

    recycle_k: int
    basis: tuple[Any, ...]


@dataclass(frozen=True)
class RHS1ReducedModeShapeSetup:
    """Pitch-mode shape summary used by active-DOF admission policy."""

    nxi_for_x: np.ndarray
    max_l: int
    has_reduced_modes: bool


@dataclass(frozen=True)
class RHS1ActiveDOFDecision:
    """Resolved active-DOF routing decision for the RHSMode=1 solve."""

    use_active_dof_mode: bool
    reason: str | None = None


@dataclass(frozen=True)
class RHS1ActiveDOFState:
    """Index maps used to reduce a full RHSMode=1 system to active pitch modes."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int


@dataclass(frozen=True)
class RHS1ActiveProblemSetup:
    """Resolved active-system setup after RHSMode=1 physics policy gates."""

    tol: float
    restart: int
    maxiter: int | None
    messages: tuple[tuple[int, str], ...]
    use_dkes: bool
    include_xdot_sparse_pc: bool
    include_electric_field_xi_sparse_pc: bool
    er_abs_sparse_pc: float
    preconditioner_species: int
    preconditioner_x: int
    preconditioner_x_min_l: int
    preconditioner_xi: int
    full_preconditioner_requested: bool
    geom_scheme: int
    use_pas_projection: bool
    use_active_dof_mode: bool
    active_idx_jnp: Any
    full_to_active_jnp: Any
    active_size: int


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


@dataclass(frozen=True)
class ProfileResponseLinearProblemSetupContext:
    """Injected dependencies for initial v3 full-system problem materialization."""

    nml: Any
    op: Any | None
    which_rhs: int | None
    restart: int
    maxiter: int | None
    tol: float
    identity_shift: float
    phi1_hat_base: Any
    emit: Callable[[int, str], None] | None
    mark: Callable[[str], None]
    env: Mapping[str, str]
    timer_factory: Callable[[], Any]
    build_operator: Callable[..., Any]
    rhs_builder: Callable[[Any], Any]
    norm: Callable[[Any], Any]
    with_transport_rhs_settings: Callable[..., Any]
    set_precond_size_hint: Callable[[int], None]
    set_precond_policy_hints: Callable[..., None]


@dataclass(frozen=True)
class ProfileResponseLinearProblemSetup:
    """Initial v3 full-system operator/RHS setup result."""

    op: Any
    which_rhs: int | None
    rhs: Any
    rhs_norm: Any
    tol: float
    fp_tol: float
    restart: int
    maxiter: int | None
    restart_env_forced: bool
    maxiter_env_forced: bool
    geom_scheme_hint: int
    tolerance_setup: RHS1ToleranceSetup


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


def _nml_get(
    group: Mapping[str, object], key: str, default: object | None = None
) -> object | None:
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


def _preconditioner_option_int(
    options: Mapping[str, object], key: str, default: int
) -> int:
    value = options.get(key, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def materialize_profile_response_linear_problem(
    context: ProfileResponseLinearProblemSetupContext,
) -> ProfileResponseLinearProblemSetup:
    """Build the initial operator/RHS pair and setup metadata for a linear solve."""

    gmres_budget_setup = resolve_rhs1_gmres_budget_setup(
        restart=int(context.restart),
        maxiter=context.maxiter,
        env=context.env,
    )
    restart = int(gmres_budget_setup.restart)
    maxiter = gmres_budget_setup.maxiter

    geom_scheme_hint = geometry_scheme_hint_from_namelist(context.nml)
    vmec_operator_timer = None
    if context.emit is not None:
        context.emit(1, "solve_v3_full_system_linear_gmres: building operator")
        if geom_scheme_hint == 5:
            eq_name = equilibrium_name_hint_from_namelist(context.nml)
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: VMEC operator build start ({eq_name})",
            )
            vmec_operator_timer = context.timer_factory()

    op = (
        context.build_operator(
            nml=context.nml,
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
        )
        if context.op is None
        else context.op
    )
    if context.emit is not None and vmec_operator_timer is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: VMEC operator build done "
            f"elapsed_s={float(vmec_operator_timer.elapsed_s()):.3f}",
        )

    context.mark("operator_built")
    context.set_precond_size_hint(int(op.total_size))
    context.set_precond_policy_hints(
        geom_scheme=geom_scheme_hint,
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
    )

    tolerance_setup = resolve_rhs1_tolerance_setup(
        op=op,
        tol=float(context.tol),
        env=context.env,
    )
    tol = float(tolerance_setup.tol)
    if context.emit is not None and tolerance_setup.fp_tightened:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: FP tol tightened "
            f"{float(tolerance_setup.fp_previous_tol):.1e} -> {float(tol):.1e}",
        )
    if context.emit is not None and tolerance_setup.pas_tightened:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: PAS tol tightened "
            f"{float(tolerance_setup.pas_previous_tol):.1e} -> {float(tol):.1e}",
        )

    which_rhs = context.which_rhs
    if int(op.rhs_mode) in {2, 3}:
        if which_rhs is None:
            which_rhs = 1
        op = context.with_transport_rhs_settings(op, which_rhs=int(which_rhs))
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: applied transport RHS settings "
                f"whichRHS={int(which_rhs)}",
            )

    if context.emit is not None:
        context.emit(
            1, f"solve_v3_full_system_linear_gmres: total_size={int(op.total_size)}"
        )
        context.emit(1, "solve_v3_full_system_linear_gmres: assembling RHS")
    rhs = context.rhs_builder(op)
    context.mark("rhs_assembled")
    rhs_norm = context.norm(rhs)
    if context.emit is not None:
        context.emit(
            2,
            f"solve_v3_full_system_linear_gmres: rhs_norm={float(rhs_norm):.6e}",
        )

    return ProfileResponseLinearProblemSetup(
        op=op,
        which_rhs=which_rhs,
        rhs=rhs,
        rhs_norm=rhs_norm,
        tol=float(tol),
        fp_tol=float(tolerance_setup.fp_tol),
        restart=int(restart),
        maxiter=maxiter,
        restart_env_forced=bool(gmres_budget_setup.restart_env_forced),
        maxiter_env_forced=bool(gmres_budget_setup.maxiter_env_forced),
        geom_scheme_hint=int(geom_scheme_hint),
        tolerance_setup=tolerance_setup,
    )


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
    include_electric_field_xi = _nml_bool(
        _nml_get(phys_params, "includeElectricFieldTermInXiDot", None)
    )
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
        restart_use, maxiter_use, restart_defaulted, maxiter_defaulted = (
            dkes_gmres_budget(
                restart=int(restart_use),
                maxiter=maxiter_use,
                restart_forced=bool(restart_env_forced),
                maxiter_forced=bool(maxiter_env_forced),
                restart_cap_env=_env_value(env, "SFINCS_JAX_RHSMODE1_DKES_RESTART_CAP"),
            )
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
    if op.fblock.pas is not None and int(active_size) <= max(
        0, int(pas_full_gmres_max)
    ):
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

    full_precond_dense_max = _read_int(
        env, "SFINCS_JAX_RHSMODE1_FULL_PRECOND_DENSE_MAX", 2500
    )
    full_precond_size = (
        int(active_size) if bool(use_active_dof_mode) else int(op.total_size)
    )
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
        solve_method_use = (
            "dense"
            if (full_precond_mode != "dense_ksp" or bool(use_active_dof_mode))
            else "dense_ksp"
        )
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
    pas_large_fastpath_env = _env_value(
        env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH"
    ).lower()
    pas_large_fastpath_min = max(
        1, _read_int(env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MIN", 80000)
    )
    pas_large_fastpath_max = _read_int(
        env, "SFINCS_JAX_PAS_LARGE_BICGSTAB_FASTPATH_MAX", 300000
    )
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
        and (
            int(pas_large_fastpath_max) <= 0
            or int(active_size) <= int(pas_large_fastpath_max)
        )
    )

    if int(op.rhs_mode) == 1 and str(solve_method_use).strip().lower() in {
        "auto",
        "default",
    }:
        sharded_multidevice_hint = (
            sharded_axis_hint in {"theta", "zeta"} and int(device_count) > 1
        )
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
        kind in SPARSE_HOST_PC_GMRES_SOLVE_METHODS
        or kind in SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
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
        structured_full_csr_explicit_requested=bool(
            kind in STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS
        ),
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


def resolve_rhs1_recycle_basis_setup(
    *,
    recycle_basis: Any,
    total_size: int,
    recycle_k_env: str,
    asarray: Callable[[Any], Any],
) -> RHS1RecycleBasisSetup:
    """Clamp and filter recycled Krylov vectors for a fixed linear-system size."""

    raw = str(recycle_k_env or "").strip()
    try:
        recycle_k = int(raw) if raw else 4
    except ValueError:
        recycle_k = 4
    recycle_k = max(0, int(recycle_k))
    if recycle_k <= 0 or not recycle_basis:
        return RHS1RecycleBasisSetup(recycle_k=int(recycle_k), basis=())

    selected: list[Any] = []
    expected_shape = (int(total_size),)
    for vec in recycle_basis:
        candidate = asarray(vec)
        if getattr(candidate, "shape", None) == expected_shape:
            selected.append(candidate)
    if len(selected) > recycle_k:
        selected = selected[-recycle_k:]
    return RHS1RecycleBasisSetup(recycle_k=int(recycle_k), basis=tuple(selected))


def resolve_rhs1_reduced_mode_shape_setup(
    *,
    nxi_for_x: Any,
    n_xi: int,
) -> RHS1ReducedModeShapeSetup:
    """Summarize velocity-grid pitch truncation for active-DOF decisions."""

    nxi_array = np.asarray(nxi_for_x, dtype=np.int32)
    max_l = int(np.max(nxi_array)) if nxi_array.size else 0
    has_reduced_modes = bool(np.any(nxi_array < int(n_xi)))
    return RHS1ReducedModeShapeSetup(
        nxi_for_x=nxi_array,
        max_l=int(max_l),
        has_reduced_modes=bool(has_reduced_modes),
    )


def _env_on(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_off(raw: str) -> bool:
    return str(raw).strip().lower() in {"0", "false", "no", "off"}


def resolve_rhs1_active_dof_mode(
    *,
    active_dof_env: str,
    dkes_active_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_reduced_modes: bool,
    sparse_host_like_requested: bool,
    xblock_active_dof_requested: bool,
    has_pas: bool,
    use_dkes: bool,
) -> RHS1ActiveDOFDecision:
    """Resolve the default RHSMode=1/transport active-DOF compaction policy."""

    env = str(active_dof_env).strip().lower()
    if _env_on(env):
        return RHS1ActiveDOFDecision(use_active_dof_mode=True, reason="env")
    if _env_off(env):
        return RHS1ActiveDOFDecision(use_active_dof_mode=False, reason="env")

    use_active_dof_mode = bool(
        has_reduced_modes
        and (int(rhs_mode) in {2, 3} or (int(rhs_mode) == 1 and not bool(include_phi1)))
    )
    reason = "auto" if use_active_dof_mode else None
    if sparse_host_like_requested and not bool(xblock_active_dof_requested):
        use_active_dof_mode = False
        reason = "sparse_host"
    if (
        _env_off(dkes_active_env)
        and use_active_dof_mode
        and int(rhs_mode) == 1
        and bool(has_pas)
        and bool(use_dkes)
    ):
        use_active_dof_mode = False
        reason = "dkes_env"
    return RHS1ActiveDOFDecision(
        use_active_dof_mode=bool(use_active_dof_mode),
        reason=reason,
    )


def build_rhs1_active_dof_state(
    *,
    op: Any,
    use_active_dof_mode: bool,
    use_pas_projection: bool,
    active_dof_indices: Callable[[Any], np.ndarray],
) -> RHS1ActiveDOFState:
    """Build active-DOF index maps for full-system or PAS-projected solves."""

    if not bool(use_active_dof_mode):
        return RHS1ActiveDOFState(
            active_idx_np=None,
            active_idx_jnp=None,
            full_to_active_jnp=None,
            active_size=int(op.total_size),
        )

    active_idx_np = np.asarray(active_dof_indices(op), dtype=np.int32)
    map_size = int(op.total_size)
    if bool(use_pas_projection):
        active_idx_np = active_idx_np[active_idx_np < int(op.f_size)]
        map_size = int(op.f_size)

    active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
    full_to_active_np = np.zeros((map_size,), dtype=np.int32)
    full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
        1,
        int(active_idx_np.shape[0]) + 1,
        dtype=np.int32,
    )
    full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
    return RHS1ActiveDOFState(
        active_idx_np=active_idx_np,
        active_idx_jnp=active_idx_jnp,
        full_to_active_jnp=full_to_active_jnp,
        active_size=int(active_idx_np.shape[0]),
    )


ProjectWithResidualFn = Callable[
    ...,
    tuple[jnp.ndarray, jnp.ndarray],
]

_FALSE_TOKENS = {"0", "false", "no", "off"}
_TRUE_TOKENS = {"1", "true", "yes", "on"}


def reduce_full_with_indices(
    v_full: jnp.ndarray, active_idx: jnp.ndarray
) -> jnp.ndarray:
    """Gather the active entries from a full vector."""

    return jnp.asarray(v_full)[jnp.asarray(active_idx, dtype=jnp.int32)]


def expand_reduced_with_map(
    v_reduced: jnp.ndarray, full_to_active: jnp.ndarray
) -> jnp.ndarray:
    """Scatter a reduced vector into full ordering using a one-based index map."""

    v_reduced = jnp.asarray(v_reduced)
    z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
    padded = jnp.concatenate([z0, v_reduced], axis=0)
    return padded[jnp.asarray(full_to_active, dtype=jnp.int32)]


def project_pas_constraint_f(
    f_flat: jnp.ndarray,
    *,
    f_shape: tuple[int, ...],
    fs_factor: jnp.ndarray,
    fs_sum_safe: jnp.ndarray,
    mask_x: jnp.ndarray,
) -> jnp.ndarray:
    """Project PAS ``l=0`` density-like rows to zero flux-surface average."""

    f = jnp.asarray(f_flat).reshape(f_shape)
    avg = jnp.einsum("tz,sxtz->sx", fs_factor, f[:, :, 0, :, :])
    avg = avg * mask_x[None, :]
    avg = avg / fs_sum_safe
    f = f.at[:, :, 0, :, :].add(-avg[:, :, None, None])
    return f.reshape((-1,))


def fp_pitch_mode_active_indices(
    *,
    n_species: int,
    n_x: int,
    n_xi: int,
    n_theta: int,
    n_zeta: int,
    nxi_for_x: np.ndarray,
    l_min: int,
    l_max: int,
    full_to_active: np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray:
    """Return active reduced indices for FP pitch modes in a Legendre band."""

    nxi_for_x_np = np.asarray(nxi_for_x, dtype=np.int32)
    full_to_active_np = (
        None if full_to_active is None else np.asarray(full_to_active, dtype=np.int32)
    )
    l_min_use = max(0, int(l_min))
    l_max_use = min(max(l_min_use, int(l_max)), int(n_xi) - 1)
    selected: list[int] = []
    for s_idx in range(int(n_species)):
        for ix in range(int(n_x)):
            if ix >= int(nxi_for_x_np.size):
                continue
            lmax_x = min(int(nxi_for_x_np[ix]) - 1, int(l_max_use))
            if lmax_x < l_min_use:
                continue
            for il in range(l_min_use, lmax_x + 1):
                for it in range(int(n_theta)):
                    for iz in range(int(n_zeta)):
                        full_idx = int(
                            (
                                (
                                    ((s_idx * int(n_x) + ix) * int(n_xi) + il)
                                    * int(n_theta)
                                    + it
                                )
                                * int(n_zeta)
                                + iz
                            )
                        )
                        if full_to_active_np is not None:
                            if full_idx >= int(full_to_active_np.size):
                                continue
                            active_idx = int(full_to_active_np[full_idx]) - 1
                            if active_idx >= 0:
                                selected.append(active_idx)
                        else:
                            selected.append(full_idx)
    if not selected:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.asarray(selected, dtype=np.int32))


def finalize_rhs1_linear_solution_cleanup(
    *,
    op: Any,
    result: GMRESSolveResult,
    rhs: jnp.ndarray,
    residual_vec: jnp.ndarray | None,
    project_solution_with_residual: ProjectWithResidualFn = (
        project_constraint_scheme1_nullspace_solution_with_residual
    ),
    source_zero_tolerance: float | None = None,
) -> GMRESSolveResult:
    """Apply final RHSMode=1 projection/source cleanup to a linear solve result."""

    if int(op.rhs_mode) != 1:
        return result

    result_use = result
    if _rhs1_project_nullspace_enabled(op):
        x_projected, residual_projected = project_solution_with_residual(
            op=op,
            x_vec=result_use.x,
            rhs_vec=rhs,
            matvec_op=op,
            enabled_env_var="SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE",
            residual_vec=(
                residual_vec
                if residual_vec is not None and residual_vec.shape == rhs.shape
                else None
            ),
        )
        if not bool(jnp.allclose(x_projected, result_use.x)):
            result_use = GMRESSolveResult(
                x=x_projected,
                residual_norm=jnp.linalg.norm(residual_projected),
            )

    if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
        zero_tol = (
            rhs1_pas_source_zero_tolerance_from_env()
            if source_zero_tolerance is None
            else float(source_zero_tolerance)
        )
        if zero_tol > 0.0:
            extra = result_use.x[-int(op.extra_size) :]
            max_abs = jnp.max(jnp.abs(extra))
            extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
            x_new = jnp.concatenate(
                [result_use.x[: -int(op.extra_size)], extra], axis=0
            )
            result_use = GMRESSolveResult(
                x=x_new, residual_norm=result_use.residual_norm
            )

    return result_use


def _rhs1_project_nullspace_enabled(op: Any) -> bool:
    project_env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "").strip().lower()
    )
    if project_env in _FALSE_TOKENS:
        return False
    if project_env in _TRUE_TOKENS:
        return True
    return bool(int(op.constraint_scheme) == 1 and not bool(op.include_phi1))


def resolve_rhs1_active_problem_setup(
    *,
    nml: Any,
    op: Any,
    tol: float,
    fp_tol: float,
    restart: int,
    maxiter: int | None,
    restart_env_forced: bool,
    maxiter_env_forced: bool,
    has_reduced_modes: bool,
    sparse_host_like_requested: bool,
    xblock_active_dof_requested: bool,
    dkes_gmres_budget: Callable[..., tuple[int, int | None, bool, bool]],
    active_dof_indices: Callable[[Any], np.ndarray],
    env: Mapping[str, str] | None = None,
) -> RHS1ActiveProblemSetup:
    """Resolve RHSMode=1 active-system policy and index maps.

    This combines the setup decisions that must happen after the operator and
    RHS are available but before any solver branch is selected.
    """

    physics = resolve_rhs1_physics_flag_setup(nml)
    dkes = resolve_rhs1_dkes_adjustment_setup(
        op=op,
        tol=float(tol),
        fp_tol=float(fp_tol),
        restart=int(restart),
        maxiter=maxiter,
        restart_env_forced=bool(restart_env_forced),
        maxiter_env_forced=bool(maxiter_env_forced),
        use_dkes=bool(physics.use_dkes),
        dkes_gmres_budget=dkes_gmres_budget,
        env=env,
    )
    active_env = _env_value(env, "SFINCS_JAX_ACTIVE_DOF").lower()
    dkes_active_env = _env_value(env, "SFINCS_JAX_ACTIVE_DOF_DKES").lower()
    active_decision = resolve_rhs1_active_dof_mode(
        active_dof_env=active_env,
        dkes_active_env=dkes_active_env,
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        has_reduced_modes=bool(has_reduced_modes),
        sparse_host_like_requested=bool(sparse_host_like_requested),
        xblock_active_dof_requested=bool(xblock_active_dof_requested),
        has_pas=op.fblock.pas is not None,
        use_dkes=bool(physics.use_dkes),
    )
    precond = resolve_rhs1_preconditioner_option_setup(
        nml=nml,
        op=op,
        sparse_host_like_requested=bool(sparse_host_like_requested),
        use_active_dof_mode=bool(active_decision.use_active_dof_mode),
        env=env,
    )
    active_state = build_rhs1_active_dof_state(
        op=op,
        use_active_dof_mode=bool(precond.use_active_dof_mode),
        use_pas_projection=bool(precond.use_pas_projection),
        active_dof_indices=active_dof_indices,
    )
    return RHS1ActiveProblemSetup(
        tol=float(dkes.tol),
        restart=int(dkes.restart),
        maxiter=dkes.maxiter,
        messages=tuple(dkes.messages),
        use_dkes=bool(physics.use_dkes),
        include_xdot_sparse_pc=bool(physics.include_xdot_sparse_pc),
        include_electric_field_xi_sparse_pc=bool(
            physics.include_electric_field_xi_sparse_pc
        ),
        er_abs_sparse_pc=float(physics.er_abs_sparse_pc),
        preconditioner_species=int(precond.preconditioner_species),
        preconditioner_x=int(precond.preconditioner_x),
        preconditioner_x_min_l=int(precond.preconditioner_x_min_l),
        preconditioner_xi=int(precond.preconditioner_xi),
        full_preconditioner_requested=bool(precond.full_preconditioner_requested),
        geom_scheme=int(precond.geom_scheme),
        use_pas_projection=bool(precond.use_pas_projection),
        use_active_dof_mode=bool(precond.use_active_dof_mode),
        active_idx_jnp=active_state.active_idx_jnp,
        full_to_active_jnp=active_state.full_to_active_jnp,
        active_size=int(active_state.active_size),
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

    from sfincs_jax.solvers.preconditioners.domain_decomposition import (
        _rhs1_dd_auto_block_size,
    )

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

    def _overlap(
        axis_name: str, *, raw_axis: str, raw_generic: str, block: int
    ) -> int | None:
        raw = str(raw_axis or "").strip() or str(raw_generic or "").strip()
        overlap = _read_int_value(raw, -1)
        if overlap < 0 and axis == axis_name:
            overlap = 2 if int(block) >= 4 else 1
            while (
                overlap > 1
                and int(block + 2 * overlap) * sum_nxi_use > patch_dof_target
            ):
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
        overlap_theta=_overlap(
            "theta",
            raw_axis=theta_overlap_env,
            raw_generic=overlap_env,
            block=block_theta,
        ),
        overlap_zeta=_overlap(
            "zeta", raw_axis=zeta_overlap_env, raw_generic=overlap_env, block=block_zeta
        ),
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
    preconditioner_species = _preconditioner_option_int(
        precond_opts, "PRECONDITIONER_SPECIES", 1
    )
    preconditioner_x = _preconditioner_option_int(precond_opts, "PRECONDITIONER_X", 1)
    preconditioner_x_min_l = _preconditioner_option_int(
        precond_opts, "PRECONDITIONER_X_MIN_L", 0
    )
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
    "RHS1ActiveDOFDecision",
    "RHS1ActiveDOFState",
    "RHS1DomainDecompositionSetup",
    "RHS1ActiveProblemSetup",
    "RHS1GmresBudgetSetup",
    "RHS1InitialRouteSetup",
    "RHS1PhysicsFlagSetup",
    "RHS1PreconditionerOptionSetup",
    "RHS1RecycleBasisSetup",
    "RHS1ReducedModeShapeSetup",
    "RHS1ToleranceSetup",
    "ProfileResponseLinearProblemSetup",
    "ProfileResponseLinearProblemSetupContext",
    "SPARSE_HOST_DIRECT_SOLVE_METHODS",
    "SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS",
    "SPARSE_HOST_MINIMUM_NORM_SOLVE_METHODS",
    "SPARSE_HOST_PC_GMRES_SOLVE_METHODS",
    "SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS",
    "SPARSE_HOST_SAFE_SOLVE_METHODS",
    "SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS",
    "STRUCTURED_FULL_CSR_HOST_SOLVE_METHODS",
    "SolveMethodRequestFlags",
    "build_rhs1_active_dof_state",
    "equilibrium_name_hint_from_namelist",
    "expand_reduced_with_map",
    "finalize_rhs1_linear_solution_cleanup",
    "fp_pitch_mode_active_indices",
    "geometry_scheme_hint_from_namelist",
    "materialize_profile_response_linear_problem",
    "normalize_profile_solve_method_kind",
    "project_pas_constraint_f",
    "reduce_full_with_indices",
    "resolve_rhs1_active_dof_mode",
    "resolve_rhs1_active_problem_setup",
    "resolve_rhs1_domain_decomposition_setup",
    "resolve_rhs1_gmres_budget_setup",
    "resolve_rhs1_initial_route_setup",
    "resolve_rhs1_physics_flag_setup",
    "resolve_rhs1_preconditioner_option_setup",
    "resolve_rhs1_recycle_basis_setup",
    "resolve_rhs1_reduced_mode_shape_setup",
    "resolve_rhs1_tolerance_setup",
    "resolve_solve_method_request_flags",
)
