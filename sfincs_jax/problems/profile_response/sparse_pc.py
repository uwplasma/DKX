"""Host sparse-PC Krylov helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .setup import (
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
)


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class SparsePCGMRESContext:
    """Solve-local dependencies for one sparse-PC GMRES attempt."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    restart: int
    tol: float
    atol: float
    precondition_side: str
    factor_dtype: np.dtype
    progress_every: int
    stagnation_abort: bool
    stagnation_min_iter: int
    stagnation_window: int
    stagnation_rel_improvement: float
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


@dataclass(frozen=True)
class SparsePCGMRESResult:
    """Measured result from one sparse-PC GMRES attempt."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    solve_s: float


@dataclass(frozen=True)
class SparsePCPostMinresContext:
    """Solve-local dependencies for the optional sparse-PC residual polish."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    steps: int
    alpha_clip: float
    min_improvement: float
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]]


@dataclass(frozen=True)
class SparsePCPostMinresResult:
    """Result of the optional sparse-PC post-minres polish."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    alphas: tuple[float, ...]
    residual_before: float
    residual_after: float | None
    error: str | None
    solve_s: float


@dataclass(frozen=True)
class SparsePCEntryPolicySetup:
    """Physics classification and GMRES budget for RHSMode=1 sparse-PC paths."""

    constrained_pas_pc: bool
    tokamak_pas_noer_pc: bool
    tokamak_pas_er_pc: bool
    tokamak_fp_er_pc: bool
    tokamak_fp_noer_pc: bool
    tokamak_fp_pc: bool
    xblock_sparse_pc: bool
    fortran_reduced_sparse_pc: bool
    sparse_pc_use_active_dof: bool
    xblock_use_active_dof: bool
    sparse_pc_fp_dense_velocity_block: bool | None
    pc_restart_env: str
    pc_restart: int
    pc_maxiter: int


@dataclass(frozen=True)
class XBlockSparsePCSetup:
    """Setup controls for RHSMode=1 x-block sparse-PC solves."""

    xblock_drop_tol: float
    xblock_drop_rel: float
    xblock_ilu_drop_tol: float
    xblock_fill_factor: float
    xblock_lower_fill_mode: str
    xblock_lower_fill_ignored_env: bool
    xblock_preconditioner_xi: int
    force_assembled_host_fp: bool
    xblock_assembled_host_fp: bool
    xblock_krylov_env_requested: str
    xblock_krylov_env: str
    xblock_krylov_requested: str
    xblock_device_fgmres_requested: bool
    xblock_device_gmres_requested: bool
    xblock_device_bicgstab_requested: bool
    xblock_device_tfqmr_requested: bool
    xblock_device_krylov_requested: bool
    xblock_device_host_fallback_decision: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: bool
    qi_device_preconditioner_requested_for_fallback: bool
    qi_device_matrix_free_requested_for_fallback: bool
    qi_device_use_in_krylov_requested_for_fallback: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockSparsePCSidePolicySetup:
    """JAX-factor and side-preconditioner policy for x-block sparse-PC solves."""

    xblock_jax_factors_env: str
    xblock_jax_factors_requested: bool
    xblock_jax_factors: bool
    xblock_jax_factor_format: str
    xblock_jax_factor_apply: str
    xblock_device_krylov_forced_jax_factors: bool
    full_fp_3d_pc: bool
    side_env: str
    precondition_side: str
    xblock_default_right_pc: bool
    xblock_krylov_method: str
    xblock_device_fgmres_forced_right_pc: bool
    pc_restart: int
    xblock_default_restart_capped: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockQIDeviceOperatorReuseSetup:
    """QI-device operator-reuse admission and x-block factor-build routing."""

    decision: object
    skip_xblock_factors: bool
    xblock_jax_factors: bool
    xblock_device_krylov_forced_jax_factors: bool
    factor_backend: str
    factor_reason: str
    messages: tuple[tuple[int, str], ...]


class MatvecCounter:
    """Mutable matvec counter that preserves ``int(counter)`` call sites."""

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def increment(self) -> None:
        self.value += 1

    def __iadd__(self, increment: int) -> "MatvecCounter":
        self.value += int(increment)
        return self

    def __int__(self) -> int:
        return int(self.value)

    def __mod__(self, divisor: int) -> int:
        return int(self.value) % int(divisor)


@dataclass(frozen=True)
class XBlockKrylovMatvecSetup:
    """Active-DOF reduction and true-matvec context for x-block Krylov solves."""

    progress_every: int
    mv_count: MatvecCounter
    xblock_linear_size: int
    xblock_active_idx_np: np.ndarray | None
    xblock_rhs: jnp.ndarray
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    matvec_no_count: ArrayFn
    matvec: ArrayFn
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockAssembledEquilibrationSetup:
    """Row/column equilibration state for an assembled x-block operator."""

    row_enabled: bool
    row_built: bool
    row_metadata: dict[str, object]
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_enabled: bool
    col_built: bool
    col_metadata: dict[str, object]
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None
    messages: tuple[tuple[int, str], ...]


class XBlockAssembledPreflightMemoryError(MemoryError):
    """Preflight rejection that carries metadata for solver diagnostics."""

    def __init__(self, message: str, metadata: Mapping[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


XBlockAssembledPreflightError = XBlockAssembledPreflightMemoryError


@dataclass(frozen=True)
class XBlockAssembledOperatorPreflightSetup:
    """Memory-budget and structural-pattern preflight for assembled x-block operators."""

    csr_max_mb: float
    drop_tol: float
    device_enabled: bool
    device_required: bool
    max_colors: int
    csr_cap_nbytes: int
    pattern: object
    summary: object
    metadata: dict[str, object]


@dataclass(frozen=True)
class XBlockAssembledDeviceSetup:
    """Optional device-resident CSR operator setup for assembled x-block matvecs."""

    device_operator: object | None
    device_resident: bool
    validation_errors: tuple[float, ...]
    error: str | None
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockAssembledMatvecSetup:
    """Matvec closure for an assembled x-block operator."""

    matvec: ArrayFn
    location: str


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    source = env if env is not None else {}
    return str(source.get(key, "")).strip()


def _env_float(env: Mapping[str, str] | None, key: str, default: float) -> float:
    raw = _env_value(env, key)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(env: Mapping[str, str] | None, key: str, default: int, minimum: int | None = None) -> int:
    raw = _env_value(env, key)
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), int(value))
    return int(value)


def _env_bool(env: Mapping[str, str] | None, key: str, default: bool = False) -> bool:
    raw = _env_value(env, key).lower()
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    return bool(default)


def resolve_sparse_pc_entry_policy(
    *,
    op: object,
    solve_method_kind: str,
    has_reduced_modes: bool,
    use_active_dof_mode: bool,
    xblock_active_dof_requested: bool,
    active_maps_available: bool,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    er_abs_sparse_pc: float,
    restart: int,
    maxiter: int | None,
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    sparse_pc_default_restart: Callable[..., int],
    env: Mapping[str, str] | None = None,
) -> SparsePCEntryPolicySetup:
    """Resolve the entry policy for host sparse-PC GMRES RHSMode=1 solves."""

    constrained_pas_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 2
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and op.fblock.fp is None
    )
    tokamak_pas_noer_pc = bool(
        constrained_pas_pc
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) == 0.0
    )
    tokamak_pas_er_pc = bool(
        constrained_pas_pc
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) > 0.0
        and (bool(use_dkes) or bool(include_xdot_sparse_pc) or bool(include_electric_field_xi_sparse_pc))
    )
    tokamak_fp_er_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) > 0.0
        and (bool(use_dkes) or bool(include_xdot_sparse_pc) or bool(include_electric_field_xi_sparse_pc))
    )
    tokamak_fp_noer_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 0
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) == 0.0
    )
    tokamak_fp_pc = bool(tokamak_fp_er_pc or tokamak_fp_noer_pc)
    xblock_sparse_pc = solve_method_kind in SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
    fortran_reduced_sparse_pc = solve_method_kind in SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS

    sparse_pc_active_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF").lower()
    sparse_pc_active_forced_on = sparse_pc_active_env in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    sparse_pc_active_forced_off = sparse_pc_active_env in {"0", "false", "f", "no", "off", ".false.", ".f."}
    sparse_pc_active_auto = sparse_pc_active_env in {"", "auto"}
    sparse_pc_use_active_dof = bool(
        (not xblock_sparse_pc)
        and bool(has_reduced_modes)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and (
            sparse_pc_active_forced_on
            or (
                sparse_pc_active_auto
                and (tokamak_pas_er_pc or tokamak_pas_noer_pc or fortran_reduced_sparse_pc)
            )
        )
        and (not sparse_pc_active_forced_off)
    )
    xblock_use_active_dof = bool(
        xblock_sparse_pc
        and bool(use_active_dof_mode)
        and bool(xblock_active_dof_requested)
        and bool(active_maps_available)
    )
    if bool(use_active_dof_mode) and not (sparse_pc_use_active_dof or xblock_use_active_dof):
        raise NotImplementedError(
            "solve_method='sparse_pc_gmres'/'xblock_sparse_pc_gmres' active-DOF mode is only implemented "
            "for the generic sparse_pc_gmres branch or opt-in x-block branch. Set "
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF=1, "
            "SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1, or SFINCS_JAX_ACTIVE_DOF=0."
        )

    fp_dense_velocity_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK").lower()
    if fp_dense_velocity_env in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        sparse_pc_fp_dense_velocity_block: bool | None = False
    elif fp_dense_velocity_env in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        sparse_pc_fp_dense_velocity_block = True
    else:
        sparse_pc_fp_dense_velocity_block = None

    pc_restart_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART")
    pc_restart, pc_maxiter = parse_polish_gmres_config(
        restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART",
        maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER",
        default_restart=max(20, int(restart)),
        default_maxiter=max(100, int(maxiter) if maxiter is not None else 400),
        min_restart=2,
        min_maxiter=1,
    )
    pc_restart = sparse_pc_default_restart(
        requested_restart=int(pc_restart),
        restart_env_value=pc_restart_env,
        tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
        n_species=int(op.n_species),
    )

    return SparsePCEntryPolicySetup(
        constrained_pas_pc=bool(constrained_pas_pc),
        tokamak_pas_noer_pc=bool(tokamak_pas_noer_pc),
        tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        tokamak_fp_noer_pc=bool(tokamak_fp_noer_pc),
        tokamak_fp_pc=bool(tokamak_fp_pc),
        xblock_sparse_pc=bool(xblock_sparse_pc),
        fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
        sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
        xblock_use_active_dof=bool(xblock_use_active_dof),
        sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        pc_restart_env=str(pc_restart_env),
        pc_restart=int(pc_restart),
        pc_maxiter=int(pc_maxiter),
    )


def _xblock_device_flags(method: str) -> tuple[bool, bool, bool, bool, bool]:
    method_s = str(method)
    fgmres = method_s == "fgmres_jax"
    gmres = method_s == "gmres_jax"
    bicgstab = method_s == "bicgstab_jax"
    tfqmr = method_s == "tfqmr_jax"
    return fgmres, gmres, bicgstab, tfqmr, bool(fgmres or gmres or bicgstab or tfqmr)


def resolve_xblock_sparse_pc_setup(
    *,
    op: object,
    preconditioner_species: int,
    preconditioner_xi: int,
    active_size: int,
    lower_fill_mode: Callable[[str], tuple[str, bool]],
    species_decoupled_for_host_assembly: Callable[..., bool],
    assembled_host_allowed: Callable[..., bool],
    krylov_method: Callable[[str], tuple[str, bool]],
    device_host_fallback_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSetup:
    """Resolve x-block sparse-PC setup controls before factor construction."""

    if op.fblock.fp is None or op.fblock.pas is not None:
        raise NotImplementedError("solve_method='xblock_sparse_pc_gmres' currently targets full-FP RHSMode=1 systems.")

    drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL", 0.0)
    drop_rel = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_REL", 1.0e-8)
    ilu_drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ILU_DROP_TOL", 1.0e-4)
    fill_factor = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR", 10.0)
    lower_fill_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL")
    lower_fill_mode_value, lower_fill_ignored_env = lower_fill_mode(lower_fill_env)

    xblock_preconditioner_xi = int(preconditioner_xi)
    if xblock_preconditioner_xi == 0:
        xblock_preconditioner_xi = 1

    force_assembled_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ASSEMBLED_HOST").lower()
    force_assembled_host_fp = force_assembled_env not in {"0", "false", "f", "no", "off", ".false.", ".f."}
    xblock_assembled_host_fp = bool(
        (
            bool(force_assembled_host_fp)
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
            and species_decoupled_for_host_assembly(
                op=op,
                preconditioner_species=int(preconditioner_species),
            )
            and int(xblock_preconditioner_xi) == 1
            and (not bool(op.point_at_x0))
        )
        or assembled_host_allowed(
            op=op,
            preconditioner_species=int(preconditioner_species),
            preconditioner_xi=int(xblock_preconditioner_xi),
            use_implicit=False,
        )
    )

    krylov_env_requested = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV").lower()
    krylov_env = str(krylov_env_requested)
    krylov_requested, _unknown = krylov_method(krylov_env)
    (
        device_fgmres,
        device_gmres,
        device_bicgstab,
        device_tfqmr,
        device_krylov,
    ) = _xblock_device_flags(str(krylov_requested))

    fallback_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK")
    fallback_auto_disabled_by_qi_device = False
    qi_device_preconditioner = _env_bool(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", default=False)
    qi_device_matrix_free = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
        default=False,
    )
    qi_device_use_in_krylov = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV",
        default=False,
    )
    precondition_side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    fallback_env_token = fallback_env.strip().lower().replace("-", "_")
    if (
        bool(device_krylov)
        and bool(qi_device_preconditioner)
        and bool(qi_device_matrix_free)
        and bool(qi_device_use_in_krylov)
        and precondition_side_env != "none"
        and fallback_env_token in {"", "auto", "default"}
    ):
        fallback_env = "off"
        fallback_auto_disabled_by_qi_device = True

    fallback_decision = device_host_fallback_decision(
        env_value=fallback_env,
        requested_krylov_method=str(krylov_requested),
        active_size=int(active_size),
        min_active_size_env_value=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE"),
        rhs_mode=int(op.rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
        n_zeta=int(getattr(op, "n_zeta", 1)),
    )
    messages: list[tuple[int, str]] = []
    if bool(fallback_decision.used):
        krylov_env = str(fallback_decision.effective_krylov_env_value)
        krylov_requested, _unknown = krylov_method(krylov_env)
        (
            device_fgmres,
            device_gmres,
            device_bicgstab,
            device_tfqmr,
            _device_krylov_after_fallback,
        ) = _xblock_device_flags(str(krylov_requested))
        device_krylov = False
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "using non-autodiff host x-block fallback for requested device Krylov "
                f"method={fallback_decision.requested_method} "
                f"reason={fallback_decision.reason} "
                f"active_size={int(active_size)}",
            )
        )
    elif bool(fallback_decision.ignored_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK value; "
                f"using auto policy reason={fallback_decision.reason}",
            )
        )
    elif bool(fallback_auto_disabled_by_qi_device):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "automatic non-autodiff host fallback disabled by explicit matrix-free "
                "QI-device Krylov preconditioner request",
            )
        )

    return XBlockSparsePCSetup(
        xblock_drop_tol=float(drop_tol),
        xblock_drop_rel=float(drop_rel),
        xblock_ilu_drop_tol=float(ilu_drop_tol),
        xblock_fill_factor=float(fill_factor),
        xblock_lower_fill_mode=str(lower_fill_mode_value),
        xblock_lower_fill_ignored_env=bool(lower_fill_ignored_env),
        xblock_preconditioner_xi=int(xblock_preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        xblock_assembled_host_fp=bool(xblock_assembled_host_fp),
        xblock_krylov_env_requested=str(krylov_env_requested),
        xblock_krylov_env=str(krylov_env),
        xblock_krylov_requested=str(krylov_requested),
        xblock_device_fgmres_requested=bool(device_fgmres),
        xblock_device_gmres_requested=bool(device_gmres),
        xblock_device_bicgstab_requested=bool(device_bicgstab),
        xblock_device_tfqmr_requested=bool(device_tfqmr),
        xblock_device_krylov_requested=bool(device_krylov),
        xblock_device_host_fallback_decision=fallback_decision,
        xblock_device_host_fallback_auto_disabled_by_qi_device=bool(fallback_auto_disabled_by_qi_device),
        qi_device_preconditioner_requested_for_fallback=bool(qi_device_preconditioner),
        qi_device_matrix_free_requested_for_fallback=bool(qi_device_matrix_free),
        qi_device_use_in_krylov_requested_for_fallback=bool(qi_device_use_in_krylov),
        messages=tuple(messages),
    )


def _normalize_jax_factor_format(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"csr", "compact", "compact_csr", "ragged_csr"}:
        return "csr"
    return "padded"


def _normalize_jax_factor_apply(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"diag", "diagonal", "jacobi", "factor_diag", "factor_diagonal"}:
        return "diagonal"
    if token in {"identity", "none", "skip"}:
        return "identity"
    if token in {"upper", "upper_only", "u", "u_only"}:
        return "upper"
    if token in {"lower", "lower_only", "l", "l_only"}:
        return "lower"
    return "exact"


def resolve_xblock_sparse_pc_side_policy_setup(
    *,
    op: object,
    xblock_device_krylov_requested: bool,
    xblock_device_host_fallback_decision: object,
    xblock_krylov_env: str,
    pc_restart: int,
    pc_restart_env: str,
    tokamak_fp_er_pc: bool,
    active_size: int,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    resolve_xblock_policy: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSidePolicySetup:
    """Resolve x-block factor format and preconditioner-side policy."""

    jax_factors_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS").lower()
    jax_factors_requested = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS",
        default=False,
    )
    fallback_used = bool(getattr(xblock_device_host_fallback_decision, "used", False))
    jax_factors = bool(jax_factors_requested or bool(xblock_device_krylov_requested)) and not fallback_used

    messages: list[tuple[int, str]] = []
    if fallback_used and bool(jax_factors_requested):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS=1 because "
                "the non-autodiff host fallback requires host sparse factors",
            )
        )

    jax_factor_format = _normalize_jax_factor_format(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT") or "padded"
    )
    jax_factor_apply = _normalize_jax_factor_apply(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY") or "exact"
    )
    device_krylov_forced_jax_factors = bool(
        xblock_device_krylov_requested
        and jax_factors_env not in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    )

    side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    full_fp_3d_right_pc_max_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_RIGHT_PC_MAX")
    full_fp_3d_pc = bool(
        op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) > 1
    )
    xblock_policy = resolve_xblock_policy(
        precondition_side_env_value=side_env,
        krylov_env_value=str(xblock_krylov_env),
        requested_restart=int(pc_restart),
        restart_env_value=str(pc_restart_env),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        active_size=int(active_size),
        full_fp_3d_right_pc_max_env_value=str(full_fp_3d_right_pc_max_env),
        use_dkes=bool(use_dkes),
        include_xdot=bool(include_xdot_sparse_pc),
        include_electric_field_xi=bool(include_electric_field_xi_sparse_pc),
    )
    precondition_side = str(xblock_policy.precondition_side)
    xblock_default_right_pc = bool(xblock_policy.default_right_preconditioned)
    xblock_krylov_method = str(xblock_policy.krylov_method)
    device_fgmres_forced_right_pc = False
    if xblock_krylov_method == "fgmres_jax" and precondition_side == "left":
        precondition_side = "right"
        device_fgmres_forced_right_pc = True
    if bool(xblock_policy.ignored_krylov_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV={xblock_krylov_env!r}",
            )
        )

    return XBlockSparsePCSidePolicySetup(
        xblock_jax_factors_env=str(jax_factors_env),
        xblock_jax_factors_requested=bool(jax_factors_requested),
        xblock_jax_factors=bool(jax_factors),
        xblock_jax_factor_format=str(jax_factor_format),
        xblock_jax_factor_apply=str(jax_factor_apply),
        xblock_device_krylov_forced_jax_factors=bool(device_krylov_forced_jax_factors),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        side_env=str(side_env),
        precondition_side=str(precondition_side),
        xblock_default_right_pc=bool(xblock_default_right_pc),
        xblock_krylov_method=str(xblock_krylov_method),
        xblock_device_fgmres_forced_right_pc=bool(device_fgmres_forced_right_pc),
        pc_restart=int(xblock_policy.gmres_restart),
        xblock_default_restart_capped=bool(xblock_policy.restart_capped),
        messages=tuple(messages),
    )


def resolve_xblock_qi_device_operator_reuse_setup(
    *,
    op: object,
    xblock_krylov_method: str,
    xblock_device_host_fallback_decision: object,
    qi_device_preconditioner_requested: bool,
    qi_device_matrix_free_requested: bool,
    qi_device_use_in_krylov_requested: bool,
    precondition_side: str,
    xblock_jax_factors: bool,
    xblock_device_krylov_forced_jax_factors: bool,
    xblock_preconditioner_xi: int,
    reuse_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceOperatorReuseSetup:
    """Resolve QI-device reuse admission before local x-block factor setup."""

    decision = reuse_decision(
        env_value=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_QI_DEVICE_OPERATOR_REUSE"),
        requested_krylov_method=str(xblock_krylov_method),
        host_fallback_used=bool(getattr(xblock_device_host_fallback_decision, "used", False)),
        rhs_mode=int(op.rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
        n_zeta=int(getattr(op, "n_zeta", 1)),
        qi_device_preconditioner_requested=bool(qi_device_preconditioner_requested),
        qi_device_matrix_free_requested=bool(qi_device_matrix_free_requested),
        qi_device_use_in_krylov_requested=bool(qi_device_use_in_krylov_requested),
        precondition_side=str(precondition_side),
    )
    skip_factors = bool(getattr(decision, "skip_xblock_factors", False))
    jax_factors = bool(xblock_jax_factors)
    forced_jax_factors = bool(xblock_device_krylov_forced_jax_factors)
    messages: list[tuple[int, str]] = []
    if skip_factors:
        jax_factors = False
        forced_jax_factors = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "using matrix-free QI-device operator reuse; skipping local x-block factors",
            )
        )
    else:
        factor_backend = "jax" if bool(jax_factors) else "host"
        factor_reason = " device-krylov" if bool(forced_jax_factors) else ""
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres building "
                f"{factor_backend} x-block preconditioner preconditioner_xi={int(xblock_preconditioner_xi)}"
                f"{factor_reason}",
            )
        )

    factor_backend = "jax" if bool(jax_factors) else "host"
    factor_reason = " device-krylov" if bool(forced_jax_factors) else ""
    return XBlockQIDeviceOperatorReuseSetup(
        decision=decision,
        skip_xblock_factors=bool(skip_factors),
        xblock_jax_factors=bool(jax_factors),
        xblock_device_krylov_forced_jax_factors=bool(forced_jax_factors),
        factor_backend=str(factor_backend),
        factor_reason=str(factor_reason),
        messages=tuple(messages),
    )


def build_xblock_krylov_matvec_setup(
    *,
    op: object,
    rhs: jnp.ndarray,
    xblock_use_active_dof: bool,
    active_idx: jnp.ndarray | None,
    full_to_active: jnp.ndarray | None,
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    operator_matvec: ArrayFn,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    env: Mapping[str, str] | None = None,
) -> XBlockKrylovMatvecSetup:
    """Build reduced/full matvec closures and progress accounting."""

    progress_every_env = _env_value(env, "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY")
    try:
        progress_every = int(progress_every_env) if progress_every_env else 25
    except ValueError:
        progress_every = 25
    progress_every = max(0, int(progress_every))
    mv_count = MatvecCounter(0)

    linear_size = int(op.total_size)
    active_idx_np: np.ndarray | None = None
    xblock_rhs = rhs
    messages: list[tuple[int, str]] = []
    if bool(xblock_use_active_dof):
        if active_idx is None or full_to_active is None:
            raise ValueError("x-block active-DOF matvec setup requires active_idx and full_to_active maps.")
        active_idx_np = np.asarray(jax.device_get(active_idx), dtype=np.int32)
        linear_size = int(active_idx_np.shape[0])
        xblock_rhs = rhs[active_idx]
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres active-DOF reduction "
                f"enabled (size={int(linear_size)}/{int(op.total_size)})",
            )
        )

    def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        if not bool(xblock_use_active_dof):
            return v_full
        assert active_idx is not None
        return reduce_full_with_indices(v_full, active_idx)

    def expand_reduced(v_vec: jnp.ndarray) -> jnp.ndarray:
        if not bool(xblock_use_active_dof):
            return v_vec
        assert full_to_active is not None
        return expand_reduced_with_map(v_vec, full_to_active)

    def matvec_no_count(v: jnp.ndarray) -> jnp.ndarray:
        x_full = expand_reduced(jnp.asarray(v, dtype=rhs.dtype))
        y_full = operator_matvec(x_full)
        return reduce_full(y_full)

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        mv_count.increment()
        if emit is not None and progress_every > 0 and int(mv_count) % progress_every == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"matvecs={int(mv_count)} elapsed_s={float(elapsed_s()):.3f}",
            )
        return matvec_no_count(v)

    return XBlockKrylovMatvecSetup(
        progress_every=int(progress_every),
        mv_count=mv_count,
        xblock_linear_size=int(linear_size),
        xblock_active_idx_np=active_idx_np,
        xblock_rhs=jnp.asarray(xblock_rhs, dtype=rhs.dtype),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        matvec_no_count=matvec_no_count,
        matvec=matvec,
        messages=tuple(messages),
    )


def _normalized_equilibration_norm(value: str) -> str:
    norm = str(value).strip().lower().replace("-", "_")
    if norm in {"inf", "max", "maximum"}:
        return "linf"
    if norm in {"linf", "l1", "l2"}:
        return norm
    return "linf"


def build_xblock_assembled_equilibration_setup(
    *,
    assembled_matrix: object,
    xblock_linear_size: int,
    elapsed_s: Callable[[], float],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledEquilibrationSetup:
    """Build optional row/column scaling for assembled x-block Krylov operators."""

    col_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE",
        default=False,
    )
    row_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE",
        default=bool(col_enabled),
    )
    row_metadata: dict[str, object] = {}
    col_metadata: dict[str, object] = {}
    row_scale_jnp: jnp.ndarray | None = None
    inv_row_scale_jnp: jnp.ndarray | None = None
    col_scale_jnp: jnp.ndarray | None = None
    inv_col_scale_jnp: jnp.ndarray | None = None
    messages: list[tuple[int, str]] = []
    row_built = False
    col_built = False
    if not bool(row_enabled):
        return XBlockAssembledEquilibrationSetup(
            row_enabled=bool(row_enabled),
            row_built=False,
            row_metadata=row_metadata,
            row_scale=None,
            inv_row_scale=None,
            col_enabled=bool(col_enabled),
            col_built=False,
            col_metadata=col_metadata,
            col_scale=None,
            inv_col_scale=None,
            messages=(),
        )

    row_start_s = float(elapsed_s())
    norm = _normalized_equilibration_norm(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM") or "linf"
    )
    floor = _env_float(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_FLOOR",
        default=1.0e-14,
    )
    floor = max(0.0, float(floor))
    max_scale = max(
        1.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_MAX_SCALE",
            default=1.0e8,
        ),
    )
    assembled_csr = assembled_matrix.tocsr()
    abs_csr = abs(assembled_csr)
    if norm == "l1":
        row_norm = np.asarray(abs_csr.sum(axis=1), dtype=np.float64).reshape((-1,))
    elif norm == "l2":
        squared_csr = assembled_csr.copy()
        squared_csr.data = np.asarray(np.abs(squared_csr.data) ** 2, dtype=np.float64)
        row_norm = np.sqrt(np.asarray(squared_csr.sum(axis=1), dtype=np.float64).reshape((-1,)))
    else:
        row_norm = np.asarray(abs_csr.max(axis=1).toarray(), dtype=np.float64).reshape((-1,))
    row_norm = np.asarray(row_norm, dtype=np.float64)
    finite_positive = np.isfinite(row_norm) & (row_norm > float(floor))
    raw_scale = np.ones_like(row_norm, dtype=np.float64)
    raw_scale[finite_positive] = 1.0 / row_norm[finite_positive]
    row_scale_np = np.clip(raw_scale, 1.0 / float(max_scale), float(max_scale))
    inv_row_scale_np = 1.0 / row_scale_np
    expected_shape = (int(xblock_linear_size),)
    if (
        row_scale_np.shape != expected_shape
        or not np.all(np.isfinite(row_scale_np))
        or not np.all(np.isfinite(inv_row_scale_np))
    ):
        raise RuntimeError("assembled x-block row equilibration produced invalid row scales")
    row_scale_jnp = jnp.asarray(row_scale_np, dtype=jnp.float64)
    inv_row_scale_jnp = jnp.asarray(inv_row_scale_np, dtype=jnp.float64)
    row_built = True

    if bool(col_enabled):
        col_start_s = float(elapsed_s())
        row_scaled_abs = abs_csr.multiply(row_scale_np[:, None])
        if norm == "l1":
            col_norm = np.asarray(row_scaled_abs.sum(axis=0), dtype=np.float64).reshape((-1,))
        elif norm == "l2":
            row_scaled_squared = assembled_csr.copy()
            row_scaled_squared.data = np.asarray(row_scaled_squared.data, dtype=np.float64) ** 2
            row_scaled_squared = row_scaled_squared.multiply((row_scale_np**2)[:, None])
            col_norm = np.sqrt(np.asarray(row_scaled_squared.sum(axis=0), dtype=np.float64).reshape((-1,)))
        else:
            col_norm = np.asarray(row_scaled_abs.max(axis=0).toarray(), dtype=np.float64).reshape((-1,))
        col_norm = np.asarray(col_norm, dtype=np.float64)
        col_finite_positive = np.isfinite(col_norm) & (col_norm > float(floor))
        raw_col_scale = np.ones_like(col_norm, dtype=np.float64)
        raw_col_scale[col_finite_positive] = 1.0 / col_norm[col_finite_positive]
        col_scale_np = np.clip(raw_col_scale, 1.0 / float(max_scale), float(max_scale))
        inv_col_scale_np = 1.0 / col_scale_np
        if (
            col_scale_np.shape != expected_shape
            or not np.all(np.isfinite(col_scale_np))
            or not np.all(np.isfinite(inv_col_scale_np))
        ):
            raise RuntimeError("assembled x-block column equilibration produced invalid column scales")
        col_scale_jnp = jnp.asarray(col_scale_np, dtype=jnp.float64)
        inv_col_scale_jnp = jnp.asarray(inv_col_scale_np, dtype=jnp.float64)
        col_built = True
        col_norm_positive = col_norm[col_finite_positive]
        col_metadata = {
            "enabled": True,
            "built": True,
            "norm": norm,
            "floor": float(floor),
            "max_scale": float(max_scale),
            "setup_s": float(elapsed_s()) - col_start_s,
            "zero_or_tiny_columns": int(col_norm.size - np.count_nonzero(col_finite_positive)),
            "col_norm_min": float(np.min(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_norm_max": float(np.max(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_scale_min": float(np.min(col_scale_np)) if col_scale_np.size else 0.0,
            "col_scale_max": float(np.max(col_scale_np)) if col_scale_np.size else 0.0,
        }

    row_norm_positive = row_norm[finite_positive]
    row_metadata = {
        "enabled": True,
        "built": True,
        "norm": norm,
        "floor": float(floor),
        "max_scale": float(max_scale),
        "setup_s": float(elapsed_s()) - row_start_s,
        "zero_or_tiny_rows": int(row_norm.size - np.count_nonzero(finite_positive)),
        "row_norm_min": float(np.min(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_norm_max": float(np.max(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_scale_min": float(np.min(row_scale_np)) if row_scale_np.size else 0.0,
        "row_scale_max": float(np.max(row_scale_np)) if row_scale_np.size else 0.0,
        "column_equilibration": bool(col_built),
    }
    messages.append(
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "assembled row equilibration built "
            f"norm={norm} "
            f"scale_range=[{float(np.min(row_scale_np)):.3e}, {float(np.max(row_scale_np)):.3e}]",
        )
    )
    if bool(col_built):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled column equilibration built "
                f"norm={norm} "
                f"scale_range=[{col_metadata['col_scale_min']:.3e}, {col_metadata['col_scale_max']:.3e}]",
            )
        )

    return XBlockAssembledEquilibrationSetup(
        row_enabled=bool(row_enabled),
        row_built=bool(row_built),
        row_metadata=row_metadata,
        row_scale=row_scale_jnp,
        inv_row_scale=inv_row_scale_jnp,
        col_enabled=bool(col_enabled),
        col_built=bool(col_built),
        col_metadata=col_metadata,
        col_scale=col_scale_jnp,
        inv_col_scale=inv_col_scale_jnp,
        messages=tuple(messages),
    )


def _csr_storage_nbytes(*, nnz: int, n_rows: int) -> int:
    return int(
        int(nnz) * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
        + (int(n_rows) + 1) * np.dtype(np.int32).itemsize
    )


def build_xblock_assembled_operator_preflight_setup(
    *,
    op: object,
    xblock_active_idx_np: np.ndarray | None,
    sparse_pc_fp_dense_velocity_block: bool | None,
    xblock_krylov_method: str,
    estimate_summary: Callable[..., object],
    full_pattern: Callable[..., object],
    active_pattern: Callable[..., object],
    summarize_pattern: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledOperatorPreflightSetup:
    """Resolve assembled-operator memory budget and structural pattern."""

    csr_max_mb = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB",
            default=2048.0,
        ),
    )
    drop_tol = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DROP_TOL",
            default=0.0,
        ),
    )
    device_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE",
        default=str(xblock_krylov_method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"},
    )
    device_required = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED",
        default=False,
    )
    max_colors = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS",
        default=512,
        minimum=1,
    )
    full_preflight = estimate_summary(
        op,
        fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
    )
    full_csr_nbytes = _csr_storage_nbytes(
        nnz=int(full_preflight.nnz),
        n_rows=int(full_preflight.shape[0]),
    )
    preflight_csr_nbytes = int(full_csr_nbytes)
    preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
    csr_cap_nbytes = int(float(csr_max_mb) * 1.0e6)
    pattern = None
    preflight_scope = "full"
    metadata: dict[str, object] = {
        "active_dof": bool(xblock_active_idx_np is not None),
        "preflight_scope": preflight_scope,
        "preflight_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_pattern_max_row_nnz_estimate": int(full_preflight.max_row_nnz),
        "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
        "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
        "preflight_full_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_full_csr_nbytes_estimate": int(full_csr_nbytes),
        "preflight_csr_max_mb": float(csr_max_mb),
        "preflight_rejected": False,
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": False,
    }
    if int(csr_cap_nbytes) <= 0:
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected non-positive CSR memory budget "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if xblock_active_idx_np is not None:
        pattern = active_pattern(
            op,
            xblock_active_idx_np,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
        active_preflight = summarize_pattern(op, pattern)
        preflight_scope = "active_dof"
        preflight_csr_nbytes = _csr_storage_nbytes(
            nnz=int(active_preflight.nnz),
            n_rows=int(active_preflight.shape[0]),
        )
        preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
        metadata.update(
            {
                "preflight_scope": preflight_scope,
                "preflight_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_pattern_max_row_nnz_estimate": int(active_preflight.max_row_nnz),
                "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
                "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
                "preflight_active_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_active_csr_nbytes_estimate": int(preflight_csr_nbytes),
            }
        )
    if int(preflight_csr_nbytes) > int(csr_cap_nbytes):
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected "
            f"{preflight_scope} CSR estimate "
            f"{int(preflight_csr_nbytes) / 1.0e6:.3g} MB > "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if pattern is None:
        pattern = full_pattern(
            op,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
    summary = summarize_pattern(op, pattern)
    return XBlockAssembledOperatorPreflightSetup(
        csr_max_mb=float(csr_max_mb),
        drop_tol=float(drop_tol),
        device_enabled=bool(device_enabled),
        device_required=bool(device_required),
        max_colors=int(max_colors),
        csr_cap_nbytes=int(csr_cap_nbytes),
        pattern=pattern,
        summary=summary,
        metadata=metadata,
    )


def build_xblock_assembled_device_setup(
    *,
    assembled_matrix: object,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    csr_cap_nbytes: int,
    device_enabled: bool,
    device_required: bool,
    validation_samples: int,
    validation_tol: float,
    device_csr_from_matrix: Callable[..., object],
    validate_device_csr_matvec: Callable[..., Sequence[float]],
) -> XBlockAssembledDeviceSetup:
    """Optionally build and validate a device CSR matvec for an assembled operator."""

    if not bool(device_enabled):
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=None,
            messages=(),
        )
    messages: list[tuple[int, str]] = []
    try:
        device_operator = device_csr_from_matrix(
            assembled_matrix,
            dtype=np.float64,
            max_nbytes=int(csr_cap_nbytes),
        )
        validation_errors = validate_device_csr_matvec(
            device_operator,
            assembled_matvec,
            samples=int(validation_samples),
            rtol=float(validation_tol),
            seed=1730,
        )
        return XBlockAssembledDeviceSetup(
            device_operator=device_operator,
            device_resident=True,
            validation_errors=tuple(float(v) for v in validation_errors),
            error=None,
            messages=(),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if bool(device_required):
            raise RuntimeError(f"assembled x-block device CSR operator failed ({error})") from exc
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled device operator disabled after build failure "
                f"({error})",
            )
        )
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=error,
            messages=tuple(messages),
    )


def build_xblock_assembled_matvec_setup(
    *,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    device_operator: object | None,
    mv_count: MatvecCounter,
    progress_every: int,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
) -> XBlockAssembledMatvecSetup:
    """Select host or device matvec closure for assembled x-block Krylov solves."""

    if device_operator is not None:
        device_matvec = device_operator.jitted_matvec()

        def matvec(v: jnp.ndarray) -> jnp.ndarray:
            mv_count.increment()
            if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"assembled_device_matvecs={int(mv_count)} "
                    f"elapsed_s={float(elapsed_s()):.3f}",
                )
            return device_matvec(jnp.asarray(v, dtype=jnp.float64))

        return XBlockAssembledMatvecSetup(matvec=matvec, location="device")

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        mv_count.increment()
        if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"assembled_host_matvecs={int(mv_count)} "
                f"elapsed_s={float(elapsed_s()):.3f}",
            )
        v_np = np.asarray(jax.device_get(v), dtype=np.float64).reshape((-1,))
        return jnp.asarray(assembled_matvec(v_np), dtype=jnp.float64)

    return XBlockAssembledMatvecSetup(matvec=matvec, location="host")


def finalize_xblock_assembled_operator_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
    assembled_matrix: object,
    assembled_summary: object,
    assembled_bundle_metadata: object,
    max_colors: int,
    validation_errors: Sequence[float],
    device_enabled: bool,
    device_required: bool,
    device_resident: bool,
    device_operator: object | None,
    device_validation_errors: Sequence[float],
    device_error: str | None,
) -> dict[str, object]:
    """Return normalized metadata after assembled x-block operator construction."""

    if hasattr(assembled_matrix, "nnz"):
        matrix_nnz = int(assembled_matrix.nnz)
    else:
        matrix_nnz = int(np.count_nonzero(np.asarray(assembled_matrix)))
    return {
        **dict(metadata),
        "setup_s": float(setup_s),
        "pattern_nnz": int(assembled_summary.nnz),
        "pattern_avg_row_nnz": float(assembled_summary.avg_row_nnz),
        "pattern_max_row_nnz": int(assembled_summary.max_row_nnz),
        "storage_kind": assembled_bundle_metadata.storage_kind,
        "reason": assembled_bundle_metadata.reason,
        "matrix_nnz": int(matrix_nnz),
        "csr_nbytes_estimate": int(assembled_bundle_metadata.csr_nbytes_estimate),
        "max_colors": int(max_colors),
        "validation_rel_errors": tuple(float(v) for v in validation_errors),
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": bool(device_resident),
        "device_nnz": int(device_operator.nnz) if device_operator is not None else None,
        "device_csr_nbytes_estimate": (
            int(device_operator.nbytes_estimate) if device_operator is not None else None
        ),
        "device_validation_rel_errors": tuple(float(v) for v in device_validation_errors),
        "device_error": device_error,
    }


def run_sparse_pc_gmres_once(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> SparsePCGMRESResult:
    """Run one host sparse-PC GMRES attempt and recompute the true residual."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres solve start "
            f"form={context.pc_form} restart={int(context.restart)} maxiter={int(maxiter)} "
            f"precondition_side={context.precondition_side} "
            f"factor_dtype={np.dtype(context.factor_dtype).name}",
        )

    solve_start_s = float(context.elapsed_s())
    stagnation_best = float("inf")
    stagnation_best_iter = 0

    def _progress_callback(iteration: int, residual_norm: float) -> None:
        nonlocal stagnation_best, stagnation_best_iter
        iteration_i = int(iteration)
        residual_f = float(residual_norm)
        if np.isfinite(residual_f) and (
            not np.isfinite(stagnation_best)
            or residual_f < stagnation_best * (1.0 - float(context.stagnation_rel_improvement))
        ):
            stagnation_best = float(residual_f)
            stagnation_best_iter = int(iteration_i)
        if (
            bool(context.stagnation_abort)
            and iteration_i >= int(context.stagnation_min_iter)
            and iteration_i - int(stagnation_best_iter) >= int(context.stagnation_window)
        ):
            raise RuntimeError(
                "sparse_pc_gmres stagnation detected: "
                f"iters={iteration_i} best_iter={int(stagnation_best_iter)} "
                f"best_ksp_residual={float(stagnation_best):.6e} "
                f"current_ksp_residual={residual_f:.6e} "
                f"window={int(context.stagnation_window)} "
                f"rel_improvement={float(context.stagnation_rel_improvement):.3e}"
            )
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if iteration_i % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres "
            f"iters={iteration_i} ksp_residual={residual_f:.6e} "
            f"elapsed_s={float(context.elapsed_s()):.3f}",
        )

    preconditioned_residual_norm = float("nan")
    if context.pc_form in {"explicit_left", "petsc_left"}:
        x_np, residual_norm, preconditioned_residual_norm, history = context.explicit_left_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            progress_callback=_progress_callback,
        )
    else:
        x_np, residual_norm, history = context.gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            precondition_side=context.precondition_side,
            progress_callback=_progress_callback,
        )

    solve_s = float(context.elapsed_s()) - solve_start_s
    try:
        residual_true = np.asarray(context.rhs, dtype=np.float64) - np.asarray(
            jax.device_get(context.matvec(jnp.asarray(x_np, dtype=jnp.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(residual_norm)

    return SparsePCGMRESResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(preconditioned_residual_norm),
        history=tuple(float(v) for v in (history or ())),
        solve_s=float(solve_s),
    )


def apply_sparse_pc_post_minres(
    *,
    context: SparsePCPostMinresContext,
    x: np.ndarray,
    residual_norm: float,
    preconditioned_residual_norm: float,
) -> SparsePCPostMinresResult:
    """Apply the optional sparse-PC minimum-residual polish and gate acceptance."""

    residual_before = float(residual_norm)
    post_minres_start_s = float(context.elapsed_s())
    history: tuple[float, ...] = ()
    alphas: tuple[float, ...] = ()
    residual_after: float | None = None
    error: str | None = None
    x_out = np.asarray(x, dtype=np.float64)
    rn_out = float(residual_norm)
    rn_pc_out = float(preconditioned_residual_norm)

    try:
        x_post_minres, residual_post_minres, post_history, post_alphas = context.minres_correction(
            matvec=context.matvec,
            rhs=context.rhs,
            x0=jnp.asarray(x_out, dtype=jnp.float64),
            preconditioner=context.preconditioner,
            steps=int(context.steps),
            alpha_clip=float(context.alpha_clip),
            min_improvement=float(context.min_improvement),
        )
        history = tuple(float(v) for v in post_history)
        alphas = tuple(float(v) for v in post_alphas)
        residual_after = float(jnp.linalg.norm(residual_post_minres))
        if np.isfinite(float(residual_after)) and float(residual_after) < float(rn_out):
            x_out = np.asarray(x_post_minres, dtype=np.float64)
            rn_out = float(residual_after)
            if context.pc_form in {"explicit_left", "petsc_left"}:
                try:
                    residual_pc = context.preconditioner(
                        context.rhs - context.matvec(jnp.asarray(x_out, dtype=jnp.float64))
                    )
                    rn_pc_out = float(jnp.linalg.norm(residual_pc))
                except Exception:
                    pass
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres "
                    f"improved residual {residual_before:.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(accepted_steps={len(alphas)})",
                )
        elif context.emit is not None:
            after = float(residual_after) if residual_after is not None else float("nan")
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres "
                f"rejected residual {residual_before:.6e} -> {after:.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres failed "
                f"({error})",
            )

    return SparsePCPostMinresResult(
        x=x_out,
        residual_norm=float(rn_out),
        preconditioned_residual_norm=float(rn_pc_out),
        history=history,
        alphas=alphas,
        residual_before=float(residual_before),
        residual_after=residual_after,
        error=error,
        solve_s=float(context.elapsed_s()) - post_minres_start_s,
    )


__all__ = [
    "SparsePCGMRESContext",
    "SparsePCGMRESResult",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "apply_sparse_pc_post_minres",
    "run_sparse_pc_gmres_once",
]
