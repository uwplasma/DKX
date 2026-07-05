"""Fortran-reduced sparse-PC x-block policy and solve helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .profile_diagnostics import fortran_reduced_xblock_result_metadata
from .profile_residual import (
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
)
from .profile_sparse_finalization import SparsePCGMRESFinalPayload, SparsePCGMRESResult
from .profile_sparse_xblock import (
    MatvecCounter,
    XBlockGlobalCouplingPolicySetup,
    XBlockInitialGuessSetup,
    XBlockMomentSchurPolicySetup,
    build_xblock_krylov_matvec_setup,
)
from .profile_sparse_policy import _env_bool, _env_float, _env_int, _env_value


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class FortranReducedSparsePCBackendSetup:
    """Backend and direct-tail policy for fortran-reduced sparse-PC solves."""

    backend_raw: str
    xblock_min_size: int
    backend_ignored_env: bool
    direct_tail_pc_env: str
    direct_tail_pc_explicit: bool
    direct_tail_structured_pc_required: bool
    direct_tail_default: bool
    direct_tail_enabled: bool
    direct_tail_structured_pc_forces_global: bool
    direct_tail_auto_forces_global: bool
    direct_tail_auto_structured_pc_forces_global: bool
    backend: str
    reason: str
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class FortranReducedXBlockFactorPolicySetup:
    """Local x-block factor controls for fortran-reduced sparse-PC solves."""

    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    preconditioner_xi: int
    promote_xi: bool
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class FortranReducedXBlockKrylovPolicySetup:
    """Krylov-side, method, progress, and matvec-counter controls."""

    side_env: str
    precondition_side: str
    pc_form: str
    krylov_method: str
    progress_every: int
    mv_count: "MatvecCounter"
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class FortranReducedXBlockInitialSeedPolicySetup:
    """Initial-seed refinement controls for fortran-reduced x-block solves."""

    enabled: bool
    refine_steps: int
    accept_ratio: float

@dataclass(frozen=True)
class FortranReducedXBlockInitialSeedResult:
    """Accepted initial seed and diagnostics for fortran-reduced x-block solves."""

    x0: jnp.ndarray | None
    used: bool
    residual_norm: float | None
    improvement_ratio: float | None
    refines_performed: int
    elapsed_s: float
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class FortranReducedXBlockKrylovSetupContext:
    """Dependencies for fortran-reduced x-block Krylov policy and matvec setup."""

    op: object
    rhs: jnp.ndarray
    xblock_use_active_dof: bool
    active_idx: jnp.ndarray | None
    full_to_active: jnp.ndarray | None
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    operator_matvec: ArrayFn
    base_preconditioner: ArrayFn
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None

@dataclass(frozen=True)
class FortranReducedXBlockKrylovSetupResult:
    """Policy values and closures for fortran-reduced x-block Krylov solves."""

    side_env: str
    precondition_side: str
    pc_form: str
    krylov_method: str
    progress_every: int
    mv_count: MatvecCounter
    matvec_no_count: ArrayFn
    matvec: ArrayFn
    preconditioner: ArrayFn

@dataclass(frozen=True)
class FortranReducedXBlockKrylovSolveContext:
    """Solve-local Krylov dispatch dependencies for fortran-reduced x-block solves."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    method: str
    pc_form: str
    restart: int
    maxiter: int
    tol: float
    atol: float
    target: float
    precondition_side: str
    progress_every: int
    mv_count: MatvecCounter
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    lgmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    gcrotmk_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    bicgstab_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]

@dataclass(frozen=True)
class FortranReducedXBlockFactorBuildContext:
    """Dependencies for the fortran-reduced x-block factor build stage."""

    op_pc: object
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    preconditioner_species: int
    preconditioner_xi: int
    sparse_pc_linear_size: int
    backend_reason: str
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None
    assembled_host_allowed: Callable[..., bool]
    builder: Callable[..., ArrayFn]

@dataclass(frozen=True)
class FortranReducedXBlockFactorBuildResult:
    """Result from building the local fortran-reduced x-block preconditioner."""

    preconditioner: ArrayFn
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    preconditioner_xi: int
    force_assembled_host_fp: bool
    factor_s: float

@dataclass(frozen=True)
class FortranReducedXBlockMomentSchurStageContext:
    """Dependencies for the optional fortran-reduced moment-Schur stage."""

    op: object
    base_preconditioner: ArrayFn
    reduce_full: ArrayFn | None
    expand_reduced: ArrayFn | None
    policy: XBlockMomentSchurPolicySetup
    precondition_side: str
    rhs: jnp.ndarray
    matvec_no_count: ArrayFn
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]

@dataclass(frozen=True)
class FortranReducedXBlockMomentSchurStageResult:
    """Result from the optional fortran-reduced moment-Schur stage."""

    preconditioner: ArrayFn
    built: bool
    used: bool
    reason: str | None
    metadata: dict[str, object]
    stats: dict[str, int]
    probe_residual_before: float | None
    probe_residual_after: float | None
    probe_improvement_ratio: float | None
    setup_s: float

@dataclass(frozen=True)
class FortranReducedXBlockGlobalCouplingStageContext:
    """Dependencies for the optional fortran-reduced global-coupling stage."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockGlobalCouplingPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None

@dataclass(frozen=True)
class FortranReducedXBlockGlobalCouplingStageResult:
    """Result from the optional fortran-reduced global-coupling stage."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float

@dataclass(frozen=True)
class FortranReducedXBlockFinalPayloadContext:
    """Explicit inputs for final fortran-reduced xblock sparse-PC payloads."""

    diagnostic_state: Mapping[str, object]
    result: SparsePCGMRESResult
    atol: float
    tol: float
    rhs_norm: float
    target: float

def resolve_fortran_reduced_sparse_pc_backend(
    *,
    op: object,
    env: Mapping[str, str] | None,
    fortran_reduced_sparse_pc: bool,
    sparse_pc_linear_size: int,
) -> FortranReducedSparsePCBackendSetup:
    """Resolve backend routing for fortran-reduced sparse-PC solves."""

    backend_raw = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND")
        .lower()
        .replace("-", "_")
    )
    xblock_min_size = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE",
        100000,
        minimum=1,
    )
    direct_tail_pc_env = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        )
        .lower()
        .replace("-", "_")
    )
    direct_tail_pc_explicit = direct_tail_pc_env not in {
        "",
        "auto",
        "active_auto",
        "structured",
        "factor",
        "host_factor",
        "legacy",
        "default",
    }
    direct_tail_structured_pc_required = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED",
        default=bool(direct_tail_pc_explicit),
    )
    direct_tail_default = bool(
        fortran_reduced_sparse_pc
        and int(sparse_pc_linear_size) >= 100000
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )
    direct_tail_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL",
        default=direct_tail_default,
    )
    direct_tail_structured_pc_forces_global = bool(
        fortran_reduced_sparse_pc
        and direct_tail_enabled
        and direct_tail_structured_pc_required
        and (
            direct_tail_pc_explicit
            or direct_tail_pc_env in {"auto", "active_auto", "structured"}
            or direct_tail_default
        )
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )
    direct_tail_auto_forces_global = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_FORCES_GLOBAL",
        default=True,
    )
    direct_tail_auto_structured_pc_forces_global = bool(
        fortran_reduced_sparse_pc
        and direct_tail_enabled
        and bool(direct_tail_auto_forces_global)
        and (
            direct_tail_pc_env in {"", "auto", "active_auto", "structured"}
            or direct_tail_default
        )
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )

    backend_ignored_env = False
    if backend_raw in {"xblock", "x_block", "local_xblock", "block", "blocked"}:
        backend = "xblock"
        reason = "env"
    elif backend_raw in {"global", "monolithic", "csr", "full"}:
        backend = "global"
        reason = "env"
    else:
        backend_ignored_env = bool(backend_raw not in {"", "auto"})
        if direct_tail_structured_pc_forces_global:
            backend = "global"
            reason = "required_direct_tail_structured_pc"
        elif direct_tail_auto_structured_pc_forces_global:
            backend = "global"
            reason = "auto_direct_tail_structured_pc"
        else:
            fblock = getattr(op, "fblock", None)
            auto_xblock_backend = bool(
                fortran_reduced_sparse_pc
                and int(sparse_pc_linear_size) >= int(xblock_min_size)
                and int(getattr(op, "rhs_mode", 0)) == 1
                and (not bool(getattr(op, "include_phi1", False)))
                and getattr(fblock, "fp", None) is not None
                and getattr(fblock, "pas", None) is None
            )
            backend = "xblock" if auto_xblock_backend else "global"
            reason = (
                f"auto_large_full_fp_size>={int(xblock_min_size)}"
                if auto_xblock_backend
                else "auto_global"
            )

    messages: tuple[tuple[int, str], ...] = ()
    if backend_ignored_env:
        messages = (
            (
                1,
                "solve_v3_full_system_linear_gmres: ignoring unknown "
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND="
                f"{backend_raw!r}; using {backend}",
            ),
        )

    return FortranReducedSparsePCBackendSetup(
        backend_raw=backend_raw,
        xblock_min_size=int(xblock_min_size),
        backend_ignored_env=bool(backend_ignored_env),
        direct_tail_pc_env=direct_tail_pc_env,
        direct_tail_pc_explicit=bool(direct_tail_pc_explicit),
        direct_tail_structured_pc_required=bool(
            direct_tail_structured_pc_required
        ),
        direct_tail_default=bool(direct_tail_default),
        direct_tail_enabled=bool(direct_tail_enabled),
        direct_tail_structured_pc_forces_global=bool(
            direct_tail_structured_pc_forces_global
        ),
        direct_tail_auto_forces_global=bool(direct_tail_auto_forces_global),
        direct_tail_auto_structured_pc_forces_global=bool(
            direct_tail_auto_structured_pc_forces_global
        ),
        backend=backend,
        reason=reason,
        messages=messages,
    )

def _env_float_first(
    env: Mapping[str, str] | None,
    names: Sequence[str],
    default: float,
) -> float:
    for name in names:
        raw = _env_value(env, name)
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            return float(default)
    return float(default)

def resolve_fortran_reduced_xblock_factor_policy(
    *,
    env: Mapping[str, str] | None,
    preconditioner_xi: int,
) -> FortranReducedXBlockFactorPolicySetup:
    """Resolve x-block factor tolerances for fortran-reduced sparse-PC solves."""

    drop_tol = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_TOL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL",
        ),
        0.0,
    )
    drop_rel = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_REL",
        ),
        1.0e-8,
    )
    ilu_drop_tol = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_ILU_DROP_TOL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ILU_DROP_TOL",
        ),
        1.0e-4,
    )
    fill_factor = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_FILL_FACTOR",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR",
        ),
        10.0,
    )
    xblock_preconditioner_xi = int(preconditioner_xi)
    promote_xi_raw = _env_value(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PROMOTE_XI",
    ).lower()
    promote_xi = promote_xi_raw not in {
        "0",
        "false",
        "f",
        "no",
        "off",
        ".false.",
        ".f.",
    }

    messages: tuple[tuple[int, str], ...] = ()
    if xblock_preconditioner_xi == 0 and bool(promote_xi):
        xblock_preconditioner_xi = 1
        messages = (
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
                "promoting x-block backend preconditioner_xi 0 -> 1 for stronger FP block factors",
            ),
        )

    return FortranReducedXBlockFactorPolicySetup(
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        ilu_drop_tol=float(ilu_drop_tol),
        fill_factor=float(fill_factor),
        preconditioner_xi=int(xblock_preconditioner_xi),
        promote_xi=bool(promote_xi),
        messages=messages,
    )

def resolve_fortran_reduced_xblock_krylov_policy(
    *,
    env: Mapping[str, str] | None,
) -> FortranReducedXBlockKrylovPolicySetup:
    """Resolve fortran-reduced x-block Krylov method and progress controls."""

    side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    precondition_side = side_env if side_env in {"left", "right", "none"} else "left"

    pc_form = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM").lower()
    if pc_form not in {"", "scipy_left", "scipy", "explicit_left", "petsc_left"}:
        pc_form = ""
    pc_form = pc_form or "scipy_left"

    krylov_method = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV")
        or "gmres"
    )
    krylov_method = krylov_method.lower().replace("-", "_")
    messages: list[tuple[int, str]] = []
    if krylov_method in {"lgmres_scipy"}:
        krylov_method = "lgmres"
    elif krylov_method in {"gcrot", "gcrotmk_scipy"}:
        krylov_method = "gcrotmk"
    elif krylov_method in {"bicgstab_scipy", "bi_cgstab"}:
        krylov_method = "bicgstab"
    elif krylov_method not in {"gmres", "lgmres", "gcrotmk", "bicgstab"}:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                "ignoring unknown SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV="
                f"{krylov_method!r}; using gmres",
            )
        )
        krylov_method = "gmres"

    progress_every_env = _env_value(env, "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY")
    try:
        progress_every = int(progress_every_env) if progress_every_env else 25
    except ValueError:
        progress_every = 25
    progress_every = max(0, int(progress_every))

    return FortranReducedXBlockKrylovPolicySetup(
        side_env=side_env,
        precondition_side=precondition_side,
        pc_form=pc_form,
        krylov_method=krylov_method,
        progress_every=int(progress_every),
        mv_count=MatvecCounter(0),
        messages=tuple(messages),
    )

def resolve_fortran_reduced_xblock_initial_seed_policy(
    *,
    env: Mapping[str, str] | None,
) -> FortranReducedXBlockInitialSeedPolicySetup:
    """Resolve initial-seed controls for fortran-reduced x-block solves."""

    seed_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_INITIAL_SEED",
        default=True,
    )
    refine_steps = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES",
        2,
        minimum=0,
    )
    accept_ratio = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO",
            default=1.0,
        ),
    )
    return FortranReducedXBlockInitialSeedPolicySetup(
        enabled=bool(seed_enabled),
        refine_steps=int(refine_steps),
        accept_ratio=float(accept_ratio),
    )

def apply_fortran_reduced_xblock_initial_seed(
    *,
    policy: FortranReducedXBlockInitialSeedPolicySetup,
    rhs: jnp.ndarray,
    rhs_norm: float,
    x0: jnp.ndarray | None,
    preconditioner: ArrayFn,
    matvec_no_count: ArrayFn,
    elapsed_s: Callable[[], float],
) -> FortranReducedXBlockInitialSeedResult:
    """Apply and refine the fortran-reduced x-block initial preconditioner seed."""

    if (not bool(policy.enabled)) or x0 is not None:
        return FortranReducedXBlockInitialSeedResult(
            x0=x0,
            used=False,
            residual_norm=None,
            improvement_ratio=None,
            refines_performed=0,
            elapsed_s=0.0,
            messages=(),
        )

    seed_start_s = float(elapsed_s())
    x_seed = jnp.asarray(preconditioner(rhs), dtype=jnp.float64)
    residual_vec_seed = rhs - matvec_no_count(x_seed)
    seed_residual_norm = float(jnp.linalg.norm(residual_vec_seed))
    seed_improvement_ratio: float | None = None
    if np.isfinite(seed_residual_norm) and seed_residual_norm > 0.0:
        seed_improvement_ratio = float(rhs_norm) / float(seed_residual_norm)
    elif np.isfinite(seed_residual_norm):
        seed_improvement_ratio = float("inf")

    refines_performed = 0
    for refine_index in range(int(policy.refine_steps)):
        if not np.isfinite(seed_residual_norm) or seed_residual_norm == 0.0:
            break
        dx_seed = jnp.asarray(preconditioner(residual_vec_seed), dtype=jnp.float64)
        x_next = x_seed + dx_seed
        residual_vec_next = rhs - matvec_no_count(x_next)
        residual_norm_next = float(jnp.linalg.norm(residual_vec_next))
        if not np.isfinite(residual_norm_next) or residual_norm_next >= float(seed_residual_norm):
            break
        x_seed = x_next
        residual_vec_seed = residual_vec_next
        seed_residual_norm = float(residual_norm_next)
        refines_performed = int(refine_index) + 1
        if seed_residual_norm > 0.0:
            seed_improvement_ratio = float(rhs_norm) / float(seed_residual_norm)
        else:
            seed_improvement_ratio = float("inf")

    seed_used = bool(
        np.isfinite(seed_residual_norm)
        and seed_residual_norm
        <= (float(rhs_norm) * max(float(policy.accept_ratio), 1.0e-300))
    )
    elapsed = float(elapsed_s()) - seed_start_s
    messages = (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "initial seed "
            f"residual={float(seed_residual_norm):.6e} "
            f"rhs_norm={float(rhs_norm):.6e} "
            f"improvement={float(seed_improvement_ratio or 0.0):.6e} "
            f"refines={int(refines_performed)}/{int(policy.refine_steps)} "
            f"accepted={bool(seed_used)} elapsed_s={elapsed:.3f}",
        ),
    )
    return FortranReducedXBlockInitialSeedResult(
        x0=x_seed if bool(seed_used) else x0,
        used=bool(seed_used),
        residual_norm=float(seed_residual_norm),
        improvement_ratio=seed_improvement_ratio,
        refines_performed=int(refines_performed),
        elapsed_s=float(elapsed),
        messages=messages,
    )

def build_fortran_reduced_xblock_krylov_setup(
    *,
    context: FortranReducedXBlockKrylovSetupContext,
) -> FortranReducedXBlockKrylovSetupResult:
    """Resolve Krylov controls and closures for fortran-reduced x-block solves."""

    policy = resolve_fortran_reduced_xblock_krylov_policy(env=context.env)
    if context.emit is not None:
        for level, message in policy.messages:
            context.emit(level, message)

    matvec_setup = build_xblock_krylov_matvec_setup(
        op=context.op,
        rhs=context.rhs,
        xblock_use_active_dof=bool(context.xblock_use_active_dof),
        active_idx=context.active_idx,
        full_to_active=context.full_to_active,
        reduce_full_with_indices=context.reduce_full_with_indices,
        expand_reduced_with_map=context.expand_reduced_with_map,
        operator_matvec=context.operator_matvec,
        elapsed_s=context.elapsed_s,
        emit=context.emit,
        env=context.env,
        progress_every=int(policy.progress_every),
        mv_count=policy.mv_count,
        progress_label="fortran_reduced_pc_gmres xblock",
        emit_active_message=False,
    )

    def preconditioner(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            context.base_preconditioner(jnp.asarray(v, dtype=context.rhs.dtype)),
            dtype=jnp.float64,
        )

    return FortranReducedXBlockKrylovSetupResult(
        side_env=policy.side_env,
        precondition_side=policy.precondition_side,
        pc_form=policy.pc_form,
        krylov_method=policy.krylov_method,
        progress_every=int(policy.progress_every),
        mv_count=policy.mv_count,
        matvec_no_count=matvec_setup.matvec_no_count,
        matvec=matvec_setup.matvec,
        preconditioner=preconditioner,
    )

def run_fortran_reduced_xblock_krylov_solve(
    *,
    context: FortranReducedXBlockKrylovSolveContext,
    x0: jnp.ndarray | np.ndarray | None,
) -> SparsePCGMRESResult:
    """Run the host Krylov method for a fortran-reduced x-block solve."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock solve start "
            f"method={context.method} form={context.pc_form} "
            f"restart={int(context.restart)} maxiter={int(context.maxiter)} "
            f"precondition_side={context.precondition_side}",
        )
    solve_start_s = float(context.elapsed_s())

    def _progress_callback(iteration: int, residual_norm: float) -> None:
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if int(iteration) % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            f"iters={int(iteration)} ksp_residual={float(residual_norm):.6e} "
            f"elapsed_s={float(context.elapsed_s()):.3f}",
        )

    rn_pc = float("nan")
    if context.method == "lgmres":
        x_np, residual_norm, history = context.lgmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.method == "gcrotmk":
        x_np, residual_norm, history = context.gcrotmk_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.method == "bicgstab":
        x_np, residual_norm, history = context.bicgstab_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.pc_form in {"explicit_left", "petsc_left"}:
        x_np, residual_norm, rn_pc, history = context.explicit_left_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
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
            maxiter=int(context.maxiter),
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

    history_tuple = tuple(float(value) for value in (history or ()))
    if context.emit is not None:
        pc_suffix = (
            f" preconditioned_residual={float(rn_pc):.6e}"
            if np.isfinite(rn_pc)
            else ""
        )
        if history_tuple:
            pc_suffix = f"{pc_suffix} ksp_residual={float(history_tuple[-1]):.6e}"
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock complete "
            f"elapsed_s={float(context.elapsed_s()):.3f} iters={len(history_tuple)} "
            f"matvecs={int(context.mv_count)} residual={float(residual_norm):.6e} "
            f"target={float(context.target):.6e}{pc_suffix}",
        )

    return SparsePCGMRESResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(rn_pc),
        history=history_tuple,
        solve_s=float(solve_s),
    )

def build_fortran_reduced_xblock_factor_stage(
    *,
    context: FortranReducedXBlockFactorBuildContext,
) -> FortranReducedXBlockFactorBuildResult:
    """Resolve and build the fortran-reduced x-block local preconditioner."""

    policy = resolve_fortran_reduced_xblock_factor_policy(
        env=context.env,
        preconditioner_xi=int(context.preconditioner_xi),
    )
    if context.emit is not None:
        for level, message in policy.messages:
            context.emit(level, message)
    force_assembled_host_fp = bool(
        context.assembled_host_allowed(
            op=context.op_pc,
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_xi=int(policy.preconditioner_xi),
            use_implicit=False,
            active_size=int(context.sparse_pc_linear_size),
        )
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
            "using x-block backend instead of monolithic CSR factor "
            f"(reason={context.backend_reason} "
            f"size={int(context.sparse_pc_linear_size)} "
            f"preconditioner_xi={int(policy.preconditioner_xi)} "
            f"assembled_host_fp={bool(force_assembled_host_fp)})",
        )
    factor_start_s = float(context.elapsed_s())
    preconditioner = context.builder(
        op=context.op_pc,
        reduce_full=context.reduce_full,
        expand_reduced=context.expand_reduced,
        build_jax_factors=False,
        preconditioner_species=int(context.preconditioner_species),
        preconditioner_xi=int(policy.preconditioner_xi),
        drop_tol=float(policy.drop_tol),
        drop_rel=float(policy.drop_rel),
        ilu_drop_tol=float(policy.ilu_drop_tol),
        fill_factor=float(policy.fill_factor),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        emit=context.emit,
    )
    return FortranReducedXBlockFactorBuildResult(
        preconditioner=preconditioner,
        drop_tol=float(policy.drop_tol),
        drop_rel=float(policy.drop_rel),
        ilu_drop_tol=float(policy.ilu_drop_tol),
        fill_factor=float(policy.fill_factor),
        preconditioner_xi=int(policy.preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        factor_s=float(context.elapsed_s()) - factor_start_s,
    )

def apply_fortran_reduced_xblock_moment_schur_stage(
    *,
    context: FortranReducedXBlockMomentSchurStageContext,
) -> FortranReducedXBlockMomentSchurStageResult:
    """Build and optionally probe the fortran-reduced moment-Schur stage."""

    if (not bool(context.policy.enabled)) or str(context.precondition_side) == "none":
        return FortranReducedXBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=None,
            metadata={},
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    if context.emit is not None:
        for level, message in context.policy.messages:
            context.emit(level, message)
    try:
        candidate, metadata, stats = context.builder(
            op=context.op,
            base_preconditioner=context.base_preconditioner,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            rcond=context.policy.rcond,
            emit=context.emit,
        )
        built = True
        used = True
        reason: str | None = "built"
        probe_residual_before: float | None = None
        probe_residual_after: float | None = None
        probe_improvement_ratio: float | None = None
        if bool(context.policy.probe_enabled):
            seed_candidate = jnp.asarray(candidate(context.rhs), dtype=jnp.float64)
            seed_residual = context.rhs - jnp.asarray(
                context.matvec_no_count(seed_candidate),
                dtype=jnp.float64,
            )
            probe_residual_after = float(jnp.linalg.norm(seed_residual))
            probe_residual_before = float(jnp.linalg.norm(context.rhs))
            if probe_residual_before > 0.0:
                probe_improvement_ratio = probe_residual_after / probe_residual_before
                required = float(probe_residual_before) * max(
                    0.0,
                    1.0 - float(context.policy.probe_min_improvement),
                )
                used = bool(
                    np.isfinite(float(probe_residual_after))
                    and float(probe_residual_after) < float(required)
                )
            else:
                probe_improvement_ratio = (
                    0.0 if probe_residual_after == 0.0 else float("inf")
                )
                used = bool(
                    np.isfinite(float(probe_residual_after))
                    and float(probe_residual_after) <= 0.0
                )
            reason = "probe_reduced" if bool(used) else "probe_not_reduced"
            if context.emit is not None:
                context.emit(
                    0 if bool(used) else 1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                    "constraint1 moment-Schur "
                    f"{'accepted' if bool(used) else 'rejected'} "
                    f"seed residual {float(probe_residual_before):.6e} "
                    f"-> {float(probe_residual_after):.6e} "
                    f"(ratio={float(probe_improvement_ratio):.6e})",
                )
        preconditioner = candidate if bool(used) else context.base_preconditioner
        setup_s = float(context.elapsed_s()) - start_s
        metadata = dict(metadata)
        metadata["setup_s"] = float(setup_s)
        return FortranReducedXBlockMomentSchurStageResult(
            preconditioner=preconditioner,
            built=bool(built),
            used=bool(used),
            reason=reason,
            metadata=metadata,
            stats=stats,
            probe_residual_before=probe_residual_before,
            probe_residual_after=probe_residual_after,
            probe_improvement_ratio=probe_improvement_ratio,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"constraint1 moment-Schur disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return FortranReducedXBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=reason,
            metadata={"error": reason, "setup_s": float(setup_s)},
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=float(setup_s),
        )

def apply_fortran_reduced_xblock_global_coupling_stage(
    *,
    context: FortranReducedXBlockGlobalCouplingStageContext,
) -> FortranReducedXBlockGlobalCouplingStageResult:
    """Build the optional fortran-reduced global-coupling stage."""

    if not bool(context.policy.should_build):
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "global-coupling build start",
        )
    try:
        if context.builder is None:
            from sfincs_jax.solvers.preconditioner_qi_corrections import (
                build_rhs1_xblock_smoothed_global_coupling_preconditioner,
            )

            builder = build_rhs1_xblock_smoothed_global_coupling_preconditioner
        else:
            builder = context.builder
        preconditioner, metadata, stats = builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=context.policy.fsavg_lmax,
            angular_lmax=context.policy.angular_lmax,
            max_extra_units=context.policy.max_extra_units,
            max_directions=context.policy.max_directions,
            rcond=context.policy.rcond,
            include_rhs=context.policy.include_rhs,
            max_setup_s=context.policy.setup_max_s,
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        metadata = dict(metadata)
        metadata["setup_s"] = float(setup_s)
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=metadata,
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"global-coupling disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={"error": error, "setup_s": float(setup_s)},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )

def resolve_fortran_reduced_xblock_moment_schur_policy(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None,
) -> XBlockMomentSchurPolicySetup:
    """Resolve moment-Schur controls for fortran-reduced x-block solves."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR",
        default=False,
    )
    generic_rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND",
            default=1.0e-12,
        ),
    )
    rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND",
            default=generic_rcond,
        ),
    )
    probe_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE",
        default=False,
    )
    probe_min_improvement = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_MIN_IMPROVEMENT",
            default=0.0,
        ),
    )

    messages: tuple[tuple[int, str], ...] = ()
    if bool(enabled) and str(precondition_side) != "none":
        messages = (
            (
                0,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                "constraint1 moment-Schur build start",
            ),
        )

    return XBlockMomentSchurPolicySetup(
        default_candidate=False,
        default_blocked_by_compact_factors=False,
        enabled=bool(enabled),
        rcond=float(rcond),
        probe_enabled=bool(probe_enabled),
        probe_min_improvement=float(probe_min_improvement),
        messages=messages,
    )

def resolve_fortran_reduced_xblock_global_coupling_policy(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None,
) -> XBlockGlobalCouplingPolicySetup:
    """Resolve global-coupling controls for fortran-reduced x-block solves."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING",
        default=False,
    )
    mode = _env_value(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MODE",
    ) or (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE")
        or "additive"
    )
    return XBlockGlobalCouplingPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        use_device_builder=False,
        mode=mode,
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS",
            default=96,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_FSAVG_LMAX",
            default=12,
            minimum=0,
        ),
        angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_ANGULAR_LMAX",
            default=2,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_INCLUDE_RHS",
            default=True,
        ),
        setup_max_s=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S",
                default=0.0,
            ),
        ),
    )

def prepare_fortran_reduced_xblock_initial_guess(
    *,
    x0: object | None,
    sparse_pc_rhs: jnp.ndarray,
    full_rhs: jnp.ndarray,
    reduce_full: ArrayFn,
) -> XBlockInitialGuessSetup:
    """Route user-provided x0 into the fortran-reduced x-block solve space."""

    if x0 is None:
        return XBlockInitialGuessSetup(x0_full=None, messages=())
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    if x0_arr.shape == sparse_pc_rhs.shape:
        return XBlockInitialGuessSetup(x0_full=x0_arr, messages=())
    if x0_arr.shape == full_rhs.shape:
        return XBlockInitialGuessSetup(
            x0_full=jnp.asarray(reduce_full(x0_arr), dtype=jnp.float64),
            messages=(),
        )
    return XBlockInitialGuessSetup(
        x0_full=None,
        messages=(
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"ignoring incompatible x0 shape={tuple(x0_arr.shape)} "
                f"expected={tuple(sparse_pc_rhs.shape)} or {tuple(full_rhs.shape)}",
            ),
        ),
    )

def fortran_reduced_xblock_final_payload_from_solve_state(
    state: Mapping[str, object],
    *,
    result: SparsePCGMRESResult,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the fortran-reduced x-block branch from state."""

    return fortran_reduced_xblock_final_payload(
        FortranReducedXBlockFinalPayloadContext(
            diagnostic_state=state,
            result=result,
            atol=float(state["atol"]),
            tol=float(state["tol"]),
            rhs_norm=float(state["rhs_norm"]),
            target=float(state["target"]),
        ),
        expand_reduced=expand_reduced,
    )

def fortran_reduced_xblock_final_payload(
    context: FortranReducedXBlockFinalPayloadContext,
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the fortran-reduced x-block sparse-PC branch.

    The x-block branch has its own metadata schema, but its final convergence
    gates are the same true-residual gates used by the generic sparse-PC path.
    Keeping that acceptance bookkeeping here avoids duplicating target logic in
    the driver while preserving the historical metadata keys.
    """

    result = context.result
    residual_norm = float(result.residual_norm)
    metadata_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    metadata_state.update(
        {
            "x_np": np.asarray(result.x, dtype=np.float64),
            "residual_norm_sparse_pc": residual_norm,
            "history": tuple(result.history),
            "solve_s": float(result.solve_s),
            "fortran_reduced_xblock_accepted_converged": profile_residual_converged(
                residual_norm,
                profile_residual_target(
                    atol=float(context.atol),
                    tol=float(context.tol),
                    rhs_norm=float(context.rhs_norm),
                ),
            ),
            "fortran_reduced_xblock_factor_quality_rejected": not profile_residual_converged(
                residual_norm,
                float(context.target),
            ),
        }
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(result.x, dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=fortran_reduced_xblock_result_metadata(metadata_state),
    )


__all__ = (
    "FortranReducedSparsePCBackendSetup",
    "FortranReducedXBlockFactorPolicySetup",
    "FortranReducedXBlockKrylovPolicySetup",
    "FortranReducedXBlockInitialSeedPolicySetup",
    "FortranReducedXBlockInitialSeedResult",
    "FortranReducedXBlockKrylovSetupContext",
    "FortranReducedXBlockKrylovSetupResult",
    "FortranReducedXBlockKrylovSolveContext",
    "FortranReducedXBlockFactorBuildContext",
    "FortranReducedXBlockFactorBuildResult",
    "FortranReducedXBlockMomentSchurStageContext",
    "FortranReducedXBlockMomentSchurStageResult",
    "FortranReducedXBlockGlobalCouplingStageContext",
    "FortranReducedXBlockGlobalCouplingStageResult",
    "FortranReducedXBlockFinalPayloadContext",
    "resolve_fortran_reduced_sparse_pc_backend",
    "resolve_fortran_reduced_xblock_factor_policy",
    "resolve_fortran_reduced_xblock_krylov_policy",
    "resolve_fortran_reduced_xblock_initial_seed_policy",
    "apply_fortran_reduced_xblock_initial_seed",
    "build_fortran_reduced_xblock_krylov_setup",
    "run_fortran_reduced_xblock_krylov_solve",
    "build_fortran_reduced_xblock_factor_stage",
    "apply_fortran_reduced_xblock_moment_schur_stage",
    "apply_fortran_reduced_xblock_global_coupling_stage",
    "resolve_fortran_reduced_xblock_moment_schur_policy",
    "resolve_fortran_reduced_xblock_global_coupling_policy",
    "prepare_fortran_reduced_xblock_initial_guess",
    "fortran_reduced_xblock_final_payload_from_solve_state",
    "fortran_reduced_xblock_final_payload",
)
