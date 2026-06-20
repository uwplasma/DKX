"""Explicit host-sparse and minimum-norm solve helpers for profile response."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..residual import (
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
)
from ..setup import SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS
from ....solver import GMRESSolveResult
from ....sparse_triangular import (
    triangular_solve_lower_padded,
    triangular_solve_upper_padded,
)


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


def _env_float(env: Mapping[str, str] | None, key: str, default: float) -> float:
    if env is None:
        return float(default)
    value = str(env.get(key, "")).strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(
    env: Mapping[str, str] | None,
    key: str,
    default: int,
    minimum: int | None = None,
) -> int:
    if env is None:
        value = int(default)
    else:
        raw = str(env.get(key, "")).strip()
        try:
            value = int(raw) if raw else int(default)
        except ValueError:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), int(value))
    return int(value)


def _env_bool(env: Mapping[str, str] | None, key: str, default: bool = False) -> bool:
    if env is None:
        return bool(default)
    value = str(env.get(key, "")).strip().lower()
    if value in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if value in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    return bool(default)


@dataclass(frozen=True)
class SparseMinimumNormPolicy:
    """Host LSQR/LSMR controls for sparse minimum-norm solves."""

    solver_name: str
    atol: float
    btol: float
    conlim: float
    damp: float
    maxiter: int
    show: bool
    petsc_compat_requested: bool


@dataclass(frozen=True)
class SparseMinimumNormPayload:
    """Driver-independent payload for a sparse minimum-norm solve."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]
    start_message: str
    completion_message: str


@dataclass(frozen=True)
class ExplicitSparseMinimumNormBranchContext:
    """Driver callbacks and controls for the explicit sparse LSQR/LSMR branch."""

    op: Any
    rhs: jnp.ndarray
    solve_method_kind: str
    differentiable: bool | None
    use_active_dof: bool
    tol: float
    atol: float
    maxiter: int | None
    rhs_norm: float
    backend: str
    env: Mapping[str, str]
    emit: EmitFn | None
    build_pattern: Callable[[Any], object]
    summarize_pattern: Callable[[Any, object], object]
    apply_cached_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    build_operator_from_pattern: Callable[..., object]


@dataclass(frozen=True)
class SparseHostDirectPayload:
    """Driver-independent payload for an explicit host sparse direct solve."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]
    completion_message: str


@dataclass(frozen=True)
class SparseHostDirectFactorSolvePayload:
    """Host direct-solve result from an explicit factor or fallback ILU factor."""

    x: np.ndarray
    residual_norm: float
    used_explicit_factor: bool


@dataclass(frozen=True)
class SparseHostDirectPolishPayload:
    """Post-direct-solve polish result for host sparse direct fallback solves."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    attempted: bool
    accepted: bool
    restart: int | None
    maxiter: int | None


@dataclass(frozen=True)
class SparseHostDirectFallbackPayload:
    """Complete host sparse direct fallback result with its true residual."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    residual_vec: jnp.ndarray
    used_explicit_factor: bool
    polish_attempted: bool
    polish_accepted: bool
    polish_restart: int | None
    polish_maxiter: int | None


@dataclass(frozen=True)
class ExplicitSparseHostDirectBranchContext:
    """Driver callbacks and controls for the explicit sparse host-LU branch."""

    op: Any
    rhs: jnp.ndarray
    differentiable: bool | None
    use_active_dof: bool
    tol: float
    atol: float
    rhs_norm: float
    refine_steps: int
    emit: EmitFn | None
    build_pattern: Callable[[Any], object]
    summarize_pattern: Callable[[Any, object], object]
    apply_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]]
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]


@dataclass(frozen=True)
class SparseHostOrILUFactorBuildContext:
    """Inputs for choosing explicit host sparse direct factorization or ILU."""

    matvec: ArrayFn
    n: int
    dtype: object
    cache_key: object
    factor_dtype: np.dtype
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    build_dense_factors: bool
    build_jax_factors: bool
    store_dense: bool
    factorization: str
    emit: EmitFn | None
    host_sparse_direct_wanted: bool
    explicit_sparse_allowed: bool
    explicit_sparse_pattern: object | None = None
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]] | None = None
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, bool]] | None = None


@dataclass(frozen=True)
class SparseHostOrILUFactorControls:
    """Resolved routing controls for a host sparse direct/ILU factor build."""

    host_sparse_direct_wanted: bool
    factor_dtype: np.dtype
    cache_key_use: object
    build_dense_factors: bool
    build_jax_factors: bool
    store_dense: bool
    explicit_sparse_allowed: bool


@dataclass(frozen=True)
class SparseHostOrILUFactorBuildResult:
    """Factor objects and matrix caches returned by sparse host/ILU setup."""

    explicit_sparse_operator: object | None
    explicit_sparse_factor: object | None
    a_csr_full: object
    a_csr_drop: object
    ilu: object
    a_dense_cache: object | None
    l_dense: object | None
    u_dense: object | None
    l_unit_diag: bool
    used_explicit_sparse: bool


@dataclass(frozen=True)
class SparseILUPreconditionerBuildContext:
    """Cached ILU factors needed to build a JAX-side sparse preconditioner."""

    cache_entry: object | None
    l_dense: object | None
    u_dense: object | None
    l_unit_diag: bool
    require_lower_diag: bool = False


@dataclass(frozen=True)
class SparseILUPreconditionerBuildResult:
    """JAX preconditioner selected from cached dense or padded ILU factors."""

    preconditioner: ArrayFn | None
    used_dense_triangular: bool
    used_padded_triangular: bool


@dataclass(frozen=True)
class SparseHostScipyPreconditionerBuildContext:
    """Host ILU factor and optional explicit matrix used by SciPy Krylov."""

    ilu: object | None
    a_csr_full: object
    base_matvec: ArrayFn
    sparse_use_matvec: bool
    unavailable_message: str = "sparse_ilu: ILU factors unavailable"


@dataclass(frozen=True)
class SparseHostScipyPreconditionerBuildResult:
    """Host preconditioner and matrix-vector product for SciPy Krylov fallback."""

    preconditioner: ArrayFn
    matvec: ArrayFn


@dataclass(frozen=True)
class SparseHostScipyGMRESContext:
    """Inputs for one host SciPy GMRES sparse fallback solve."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    x0: jnp.ndarray
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    residual_matvec: ArrayFn | None = None


@dataclass(frozen=True)
class SparseHostRetryCandidateContext:
    """Inputs for choosing one sparse-host retry candidate after factor setup."""

    factor_build: SparseHostOrILUFactorBuildResult
    host_sparse_direct: bool
    host_direct_operator_pc: bool
    use_implicit: bool
    matvec: ArrayFn
    rhs: jnp.ndarray
    x0: jnp.ndarray
    factor_dtype: np.dtype
    refine_steps: int
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    emit: EmitFn | None
    backend_name: str
    sparse_use_matvec: bool
    sparse_exact_lu: bool
    cache_entry: object | None
    require_lower_diag: bool
    polish_enabled: Callable[..., bool]
    parse_polish_gmres_config: Callable[..., tuple[int, int]]
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    implicit_solver: Callable[[ArrayFn], tuple[GMRESSolveResult | None, jnp.ndarray | None]]
    operator_pc_restart: int | None = None
    operator_pc_maxiter: int | None = None
    compute_scipy_residual_vec: bool = True


@dataclass(frozen=True)
class SparseHostRetryCandidateResult:
    """Sparse retry candidate plus callbacks needed by the replay accept gate."""

    result: GMRESSolveResult | None
    residual_vec: jnp.ndarray | None
    matvec: ArrayFn
    preconditioner: ArrayFn | None
    solve_s: float
    host_sparse_direct_used: bool


@dataclass(frozen=True)
class SparseJAXRetryPreconditionerBuildContext:
    """Inputs for building the sparse-JAX retry preconditioner."""

    matvec: ArrayFn
    n: int
    dtype: object
    cache_key: object
    drop_tol: float
    drop_rel: float
    reg: float
    omega: float
    sweeps: int
    emit: EmitFn | None
    builder: Callable[..., ArrayFn]


@dataclass(frozen=True)
class ExplicitSparseOperatorBuildPolicy:
    """Materialization controls shared by explicit host sparse solve paths."""

    csr_max_mb: float
    drop_tol: float


@dataclass(frozen=True)
class ExplicitSparseOperatorBuildResult:
    """Materialized explicit sparse operator and stable progress messages."""

    operator_bundle: object
    policy: ExplicitSparseOperatorBuildPolicy
    messages: tuple[tuple[int, str], ...]


def explicit_sparse_pattern_progress_messages(
    *,
    solver_label: str,
    summary: object,
) -> tuple[tuple[int, str], ...]:
    """Return stable progress lines for conservative sparse-pattern setup."""

    return (
        (
            1,
            f"solve_v3_full_system_linear_gmres: {solver_label} building conservative pattern",
        ),
        (
            1,
            f"solve_v3_full_system_linear_gmres: {solver_label} pattern "
            f"nnz={int(summary.nnz)} avg_row_nnz={float(summary.avg_row_nnz):.3g} "
            f"max_row_nnz={int(summary.max_row_nnz)}",
        ),
    )


def resolve_explicit_sparse_operator_build_policy(
    env: Mapping[str, str] | None,
) -> ExplicitSparseOperatorBuildPolicy:
    """Resolve explicit sparse operator materialization controls."""

    return ExplicitSparseOperatorBuildPolicy(
        csr_max_mb=_env_float(env, "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", 512.0),
        drop_tol=_env_float(env, "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", 0.0),
    )


def build_explicit_sparse_operator_from_pattern(
    *,
    matvec_np: Callable[[np.ndarray], np.ndarray],
    pattern: object,
    dtype: object,
    backend: str,
    env: Mapping[str, str] | None,
    build_operator_from_pattern: Callable[..., object],
    allow_operator_only: bool = False,
) -> ExplicitSparseOperatorBuildResult:
    """Materialize an explicit sparse operator using shared host controls."""

    policy = resolve_explicit_sparse_operator_build_policy(env)
    operator_bundle = build_operator_from_pattern(
        matvec_np,
        pattern=pattern,
        dtype=dtype,
        backend=backend,
        csr_max_mb=float(policy.csr_max_mb),
        drop_tol=float(policy.drop_tol),
        allow_operator_only=bool(allow_operator_only),
    )
    return ExplicitSparseOperatorBuildResult(
        operator_bundle=operator_bundle,
        policy=policy,
        messages=(
            (
                1,
                "explicit_sparse: "
                f"storage={operator_bundle.metadata.storage_kind} "
                f"reason={operator_bundle.metadata.reason}",
            ),
        ),
    )


def validate_explicit_sparse_host_request(
    *,
    solve_method_label: str,
    differentiable: bool | None,
    rhs_mode: int,
    use_active_dof: bool,
    path_description: str,
) -> None:
    """Validate that an explicit host sparse solve is on the non-autodiff lane."""

    if differentiable is True:
        raise ValueError(
            f"solve_method='{solve_method_label}' is a non-differentiable {path_description}."
        )
    if int(rhs_mode) != 1:
        raise NotImplementedError(
            f"solve_method='{solve_method_label}' is currently implemented for RHSMode=1 only."
        )
    if bool(use_active_dof):
        raise NotImplementedError(
            f"solve_method='{solve_method_label}' currently targets the full system; "
            "set SFINCS_JAX_ACTIVE_DOF=0 or use the default matrix-free solver for "
            "active-DOF runs."
        )


def resolve_sparse_minimum_norm_policy(
    env: Mapping[str, str],
    *,
    solve_method_kind: str,
    tol: float,
    maxiter: int | None,
    emit_enabled: bool,
) -> SparseMinimumNormPolicy:
    """Parse host sparse minimum-norm controls from environment values."""

    maxiter_default = max(1000, int(maxiter or 400))
    kind = str(solve_method_kind)
    return SparseMinimumNormPolicy(
        solver_name="lsqr" if kind in {"sparse_lsqr", "sparse_host_lsqr"} else "lsmr",
        atol=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_ATOL", float(tol)),
        btol=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_BTOL", float(tol)),
        conlim=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_CONLIM", 1.0e8),
        damp=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_DAMP", 0.0),
        maxiter=max(1, _env_int(env, "SFINCS_JAX_SPARSE_LSMR_MAXITER", maxiter_default)),
        show=bool(emit_enabled and _env_bool(env, "SFINCS_JAX_SPARSE_LSMR_SHOW")),
        petsc_compat_requested=kind in SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS,
    )


def sparse_minimum_norm_start_message(policy: SparseMinimumNormPolicy) -> str:
    """Return the stable progress line emitted before the LSQR/LSMR solve."""

    return (
        "solve_v3_full_system_linear_gmres: sparse_lsmr solve start "
        f"solver={policy.solver_name} atol={policy.atol:.1e} btol={policy.btol:.1e} "
        f"damp={policy.damp:.1e} conlim={policy.conlim:.1e} maxiter={int(policy.maxiter)}"
    )


def sparse_minimum_norm_solve_payload(
    *,
    matrix: Any,
    rhs: jnp.ndarray,
    policy: SparseMinimumNormPolicy,
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
) -> SparseMinimumNormPayload:
    """Solve a materialized host sparse system with LSQR/LSMR and gate residuals."""

    import scipy.sparse.linalg as _spla  # noqa: PLC0415

    rhs_np = np.asarray(rhs, dtype=np.float64).reshape((-1,))
    if policy.solver_name == "lsqr":
        ls_result = _spla.lsqr(
            matrix,
            rhs_np,
            damp=float(policy.damp),
            atol=float(policy.atol),
            btol=float(policy.btol),
            conlim=float(policy.conlim),
            iter_lim=int(policy.maxiter),
            show=bool(policy.show),
        )
    else:
        ls_result = _spla.lsmr(
            matrix,
            rhs_np,
            damp=float(policy.damp),
            atol=float(policy.atol),
            btol=float(policy.btol),
            conlim=float(policy.conlim),
            maxiter=int(policy.maxiter),
            show=bool(policy.show),
        )

    x_np = np.asarray(ls_result[0], dtype=np.float64)
    istop = int(ls_result[1])
    iters = int(ls_result[2])
    solver_reported_residual = float(ls_result[3])
    residual_true = rhs_np - np.asarray(matrix @ x_np, dtype=np.float64)
    residual_norm = float(np.linalg.norm(residual_true))
    target = profile_residual_target(
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
    )
    true_residual_converged = profile_residual_converged(residual_norm, target)
    compatibility_converged = bool(istop in {1, 2})
    accepted_converged = bool(
        true_residual_converged
        or (policy.petsc_compat_requested and compatibility_converged)
    )
    acceptance_criterion = (
        "true_residual"
        if true_residual_converged
        else "petsc_compatible_minimum_norm"
        if policy.petsc_compat_requested and compatibility_converged
        else "not_converged"
    )
    completion_message = (
        "solve_v3_full_system_linear_gmres: sparse_lsmr complete "
        f"elapsed_s={float(elapsed_s()):.3f} iters={iters} istop={istop} "
        f"reported_residual={solver_reported_residual:.6e} "
        f"residual={residual_norm:.6e} target={float(target):.6e} "
        f"accepted={accepted_converged} criterion={acceptance_criterion}"
    )
    return SparseMinimumNormPayload(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata={
            "solver_kind": "sparse_lsmr",
            "residual_kind": "least_squares_true_residual",
            "reported_residual_norm": float(solver_reported_residual),
            "iterations": int(iters),
            "info_code": int(istop),
            "least_squares_converged": bool(compatibility_converged),
            "true_residual_converged": bool(true_residual_converged),
            "accepted_converged": bool(accepted_converged),
            "acceptance_criterion": str(acceptance_criterion),
            "petsc_compat_requested": bool(policy.petsc_compat_requested),
        },
        start_message=sparse_minimum_norm_start_message(policy),
        completion_message=completion_message,
    )


def sparse_minimum_norm_solve_from_pattern(
    *,
    matvec_np: Callable[[np.ndarray], np.ndarray],
    pattern: object,
    summary: object,
    rhs: jnp.ndarray,
    solve_method_kind: str,
    tol: float,
    atol: float,
    maxiter: int | None,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    backend: str,
    env: Mapping[str, str],
    emit: EmitFn | None,
    build_operator_from_pattern: Callable[..., object],
) -> SparseMinimumNormPayload:
    """Materialize the explicit sparse matrix and run the host minimum-norm solve."""

    if emit is not None:
        for level, message in explicit_sparse_pattern_progress_messages(
            solver_label="sparse_lsmr",
            summary=summary,
        ):
            emit(level, message)
    sparse_operator_build = build_explicit_sparse_operator_from_pattern(
        matvec_np=matvec_np,
        pattern=pattern,
        dtype=np.float64,
        backend=backend,
        env=env,
        build_operator_from_pattern=build_operator_from_pattern,
        allow_operator_only=False,
    )
    if emit is not None:
        for level, message in sparse_operator_build.messages:
            emit(level, message)
    matrix = sparse_operator_build.operator_bundle.matrix
    if matrix is None:
        raise RuntimeError("sparse_lsmr requires a materialized sparse matrix.")

    policy = resolve_sparse_minimum_norm_policy(
        env,
        solve_method_kind=solve_method_kind,
        tol=float(tol),
        maxiter=maxiter,
        emit_enabled=emit is not None,
    )
    if emit is not None:
        emit(0, sparse_minimum_norm_start_message(policy))
    payload = sparse_minimum_norm_solve_payload(
        matrix=matrix,
        rhs=rhs,
        policy=policy,
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
        elapsed_s=elapsed_s,
    )
    if emit is not None:
        emit(0, payload.completion_message)
    return payload


def _elapsed_since_now() -> Callable[[], float]:
    """Return a cheap elapsed-time callback for explicit host sparse branches."""

    start_s = perf_counter()
    return lambda: perf_counter() - start_s


def solve_explicit_sparse_minimum_norm_branch(
    context: ExplicitSparseMinimumNormBranchContext,
) -> SparseMinimumNormPayload:
    """Run the explicit sparse LSQR/LSMR branch from driver-provided callbacks."""

    validate_explicit_sparse_host_request(
        solve_method_label="sparse_lsmr",
        differentiable=context.differentiable,
        rhs_mode=int(context.op.rhs_mode),
        use_active_dof=bool(context.use_active_dof),
        path_description="host sparse minimum-norm path",
    )
    pattern = context.build_pattern(context.op)
    summary = context.summarize_pattern(context.op, pattern)
    rhs_dtype = context.rhs.dtype

    def matvec_np(x_np: np.ndarray) -> np.ndarray:
        x_device = jnp.asarray(np.asarray(x_np, dtype=np.float64), dtype=rhs_dtype)
        return np.asarray(
            context.apply_cached_operator(context.op, x_device),
            dtype=np.float64,
        )

    return sparse_minimum_norm_solve_from_pattern(
        matvec_np=matvec_np,
        pattern=pattern,
        summary=summary,
        rhs=context.rhs,
        solve_method_kind=context.solve_method_kind,
        tol=float(context.tol),
        atol=float(context.atol),
        maxiter=context.maxiter,
        rhs_norm=float(context.rhs_norm),
        elapsed_s=_elapsed_since_now(),
        backend=str(context.backend),
        env=context.env,
        emit=context.emit,
        build_operator_from_pattern=context.build_operator_from_pattern,
    )


def sparse_host_direct_solve_payload(
    *,
    factor_solve: Callable[[Any], Any],
    operator_matrix: Any,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    matvec: Callable[[np.ndarray], jnp.ndarray],
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPayload:
    """Solve with a host sparse direct factor and return stable result metadata."""

    x_np, residual_norm = direct_solve_with_refinement(
        factor_solve=factor_solve,
        operator_matrix=operator_matrix,
        rhs_vec=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
    )
    try:
        residual_true = np.asarray(rhs, dtype=np.float64) - np.asarray(
            jax.device_get(matvec(np.asarray(x_np, dtype=np.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(residual_norm)

    target = profile_residual_target(
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
    )
    accepted_converged = profile_residual_converged(float(residual_norm), target)
    completion_message = (
        "solve_v3_full_system_linear_gmres: sparse_host complete "
        f"elapsed_s={float(elapsed_s()):.3f} residual={float(residual_norm):.6e}"
    )
    return SparseHostDirectPayload(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata={
            "solver_kind": "sparse_host",
            "residual_kind": "true_residual",
            "accepted_converged": bool(accepted_converged),
            "acceptance_criterion": "true_residual",
        },
        completion_message=completion_message,
    )


def sparse_host_direct_solve_from_pattern(
    *,
    matvec: Callable[[np.ndarray], jnp.ndarray],
    pattern: object,
    summary: object,
    n: int,
    dtype: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPayload:
    """Build an explicit host sparse factor and solve the full RHSMode=1 system."""

    if emit is not None:
        for level, message in explicit_sparse_pattern_progress_messages(
            solver_label="sparse_host",
            summary=summary,
        ):
            emit(level, message)
    operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
        matvec=matvec,
        n=int(n),
        dtype=dtype,
        factor_dtype=factor_dtype,
        pattern=pattern,
        emit=emit,
    )
    payload = sparse_host_direct_solve_payload(
        factor_solve=factor_bundle.solve,
        operator_matrix=operator_bundle.matrix,
        rhs=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
        matvec=matvec,
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
        elapsed_s=elapsed_s,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )
    if emit is not None:
        emit(0, payload.completion_message)
    return payload


def solve_explicit_sparse_host_direct_branch(
    context: ExplicitSparseHostDirectBranchContext,
) -> SparseHostDirectPayload:
    """Run the explicit sparse host-LU branch from driver-provided callbacks."""

    validate_explicit_sparse_host_request(
        solve_method_label="sparse_host",
        differentiable=context.differentiable,
        rhs_mode=int(context.op.rhs_mode),
        use_active_dof=bool(context.use_active_dof),
        path_description="host sparse LU path",
    )
    pattern = context.build_pattern(context.op)
    summary = context.summarize_pattern(context.op, pattern)
    rhs_dtype = context.rhs.dtype

    def matvec(x_np: np.ndarray) -> jnp.ndarray:
        x_device = jnp.asarray(x_np, dtype=rhs_dtype)
        return context.apply_operator(context.op, x_device)

    return sparse_host_direct_solve_from_pattern(
        matvec=matvec,
        pattern=pattern,
        summary=summary,
        n=int(context.op.total_size),
        dtype=rhs_dtype,
        factor_dtype=np.dtype(np.float64),
        rhs=context.rhs,
        refine_steps=int(context.refine_steps),
        atol=float(context.atol),
        tol=float(context.tol),
        rhs_norm=float(context.rhs_norm),
        elapsed_s=_elapsed_since_now(),
        emit=context.emit,
        build_host_sparse_direct_factor_from_matvec=(
            context.build_host_sparse_direct_factor_from_matvec
        ),
        direct_solve_with_refinement=context.direct_solve_with_refinement,
    )


def solve_sparse_host_direct_from_available_factor(
    *,
    explicit_sparse_factor: object | None,
    explicit_sparse_operator: object | None,
    ilu: object,
    a_csr_full: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectFactorSolvePayload:
    """Solve with an explicit host factor when present, otherwise with ILU/CSR."""

    if explicit_sparse_factor is not None and explicit_sparse_operator is not None:
        x_np, residual_norm = direct_solve_with_refinement(
            factor_solve=explicit_sparse_factor.solve,
            operator_matrix=explicit_sparse_operator.matrix,
            rhs_vec=rhs,
            factor_dtype=factor_dtype,
            refine_steps=int(refine_steps),
        )
        return SparseHostDirectFactorSolvePayload(
            x=np.asarray(x_np, dtype=np.float64),
            residual_norm=float(residual_norm),
            used_explicit_factor=True,
        )

    x_np, residual_norm = ilu_solve_with_refinement(
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs_vec=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
    )
    return SparseHostDirectFactorSolvePayload(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        used_explicit_factor=False,
    )


def apply_sparse_host_direct_polish_if_needed(
    *,
    x: np.ndarray,
    residual_norm: float,
    factor_dtype: np.dtype,
    target: float,
    matvec: ArrayFn,
    rhs: jnp.ndarray,
    ilu: object,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    emit: EmitFn | None,
    polish_enabled: Callable[..., bool],
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPolishPayload:
    """Optionally polish a float32 host sparse direct solve with GMRES."""

    x_current = np.asarray(x, dtype=np.float64)
    residual_current = float(residual_norm)
    if np.dtype(factor_dtype) != np.dtype(np.float32) or residual_current <= float(target):
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_current, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
            attempted=False,
            accepted=False,
            restart=None,
            maxiter=None,
        )
    if not polish_enabled(env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH"):
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_current, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
            attempted=False,
            accepted=False,
            restart=None,
            maxiter=None,
        )

    polish_restart, polish_maxiter = parse_polish_gmres_config(
        restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH_RESTART",
        maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH_MAXITER",
        default_restart=min(int(restart), 40),
        default_maxiter=min(max(40, int(maxiter or 120)), 120),
    )
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            f"restart={polish_restart} maxiter={polish_maxiter}",
        )
    x_polish, residual_norm_polish = host_sparse_direct_polish(
        matvec_fn=matvec,
        rhs_vec=rhs,
        x0_np=x_current,
        ilu=ilu,
        factor_dtype=factor_dtype,
        tol=tol,
        atol=atol,
        restart=polish_restart,
        maxiter=polish_maxiter,
        precondition_side=precondition_side,
    )
    if np.isfinite(residual_norm_polish) and float(residual_norm_polish) < residual_current:
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_polish, dtype=jnp.float64),
            residual_norm=jnp.asarray(float(residual_norm_polish), dtype=jnp.float64),
            attempted=True,
            accepted=True,
            restart=int(polish_restart),
            maxiter=int(polish_maxiter),
        )
    return SparseHostDirectPolishPayload(
        x=jnp.asarray(x_current, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
        attempted=True,
        accepted=False,
        restart=int(polish_restart),
        maxiter=int(polish_maxiter),
    )


def sparse_host_direct_fallback_payload(
    *,
    explicit_sparse_factor: object | None,
    explicit_sparse_operator: object | None,
    ilu: object,
    a_csr_full: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    matvec: ArrayFn,
    target: float,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    emit: EmitFn | None,
    backend_name: str | None = None,
    polish_enabled: Callable[..., bool],
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectFallbackPayload:
    """Run a host sparse direct fallback, optional polish, and true residual check."""

    if emit is not None and backend_name is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: host sparse LU direct fallback "
            f"on backend={backend_name}",
        )
    factor_payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=explicit_sparse_factor,
        explicit_sparse_operator=explicit_sparse_operator,
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
        direct_solve_with_refinement=direct_solve_with_refinement,
        ilu_solve_with_refinement=ilu_solve_with_refinement,
    )
    polish_payload = apply_sparse_host_direct_polish_if_needed(
        x=factor_payload.x,
        residual_norm=float(factor_payload.residual_norm),
        factor_dtype=factor_dtype,
        target=float(target),
        matvec=matvec,
        rhs=rhs,
        ilu=ilu,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        precondition_side=precondition_side,
        emit=emit,
        polish_enabled=polish_enabled,
        parse_polish_gmres_config=parse_polish_gmres_config,
        host_sparse_direct_polish=host_sparse_direct_polish,
    )
    residual_vec = jnp.asarray(rhs, dtype=jnp.float64) - matvec(polish_payload.x)
    return SparseHostDirectFallbackPayload(
        x=polish_payload.x,
        residual_norm=polish_payload.residual_norm,
        residual_vec=residual_vec,
        used_explicit_factor=bool(factor_payload.used_explicit_factor),
        polish_attempted=bool(polish_payload.attempted),
        polish_accepted=bool(polish_payload.accepted),
        polish_restart=polish_payload.restart,
        polish_maxiter=polish_payload.maxiter,
    )


def build_sparse_host_or_ilu_factor(
    context: SparseHostOrILUFactorBuildContext,
) -> SparseHostOrILUFactorBuildResult:
    """Build either an explicit host sparse direct factor or the ILU fallback."""

    if bool(context.host_sparse_direct_wanted) and bool(context.explicit_sparse_allowed):
        if context.build_host_sparse_direct_factor_from_matvec is None:
            raise ValueError("explicit sparse host factor requested without a build callback")
        explicit_sparse_operator, explicit_sparse_factor = (
            context.build_host_sparse_direct_factor_from_matvec(
                matvec=context.matvec,
                n=int(context.n),
                dtype=context.dtype,
                factor_dtype=context.factor_dtype,
                pattern=context.explicit_sparse_pattern,
                emit=context.emit,
            )
        )
        return SparseHostOrILUFactorBuildResult(
            explicit_sparse_operator=explicit_sparse_operator,
            explicit_sparse_factor=explicit_sparse_factor,
            a_csr_full=explicit_sparse_operator.matrix,
            a_csr_drop=explicit_sparse_operator.matrix,
            ilu=explicit_sparse_factor.factor,
            a_dense_cache=None,
            l_dense=None,
            u_dense=None,
            l_unit_diag=False,
            used_explicit_sparse=True,
        )

    if context.build_sparse_ilu_from_matvec is None:
        raise ValueError("ILU factor requested without a build callback")
    a_csr_full, a_csr_drop, ilu, a_dense_cache, l_dense, u_dense, l_unit_diag = (
        context.build_sparse_ilu_from_matvec(
            matvec=context.matvec,
            n=int(context.n),
            dtype=context.dtype,
            cache_key=context.cache_key,
            factor_dtype=context.factor_dtype,
            drop_tol=float(context.drop_tol),
            drop_rel=float(context.drop_rel),
            ilu_drop_tol=float(context.ilu_drop_tol),
            fill_factor=float(context.fill_factor),
            build_dense_factors=bool(context.build_dense_factors),
            build_jax_factors=bool(context.build_jax_factors),
            build_ilu=True,
            store_dense=bool(context.store_dense),
            factorization=str(context.factorization),
            emit=context.emit,
        )
    )
    return SparseHostOrILUFactorBuildResult(
        explicit_sparse_operator=None,
        explicit_sparse_factor=None,
        a_csr_full=a_csr_full,
        a_csr_drop=a_csr_drop,
        ilu=ilu,
        a_dense_cache=a_dense_cache,
        l_dense=l_dense,
        u_dense=u_dense,
        l_unit_diag=bool(l_unit_diag),
        used_explicit_sparse=False,
    )


def resolve_sparse_host_or_ilu_factor_controls(
    *,
    n: int,
    cache_key: object,
    sparse_exact_lu: bool,
    use_implicit: bool,
    force_host_sparse_direct: bool,
    sparse_ilu_dense_max: int,
    sparse_dense_cache_max: int,
    host_sparse_direct_wanted: bool | None = None,
    host_sparse_direct_allowed: Callable[..., bool],
    host_sparse_factor_dtype: Callable[..., np.dtype],
    sparse_factor_cache_key: Callable[..., object],
    explicit_sparse_host_direct_allowed: Callable[..., bool],
) -> SparseHostOrILUFactorControls:
    """Resolve host sparse direct/ILU build controls shared by reduced/full paths."""

    direct_wanted = (
        bool(host_sparse_direct_wanted)
        if host_sparse_direct_wanted is not None
        else bool(
            host_sparse_direct_allowed(
                sparse_exact_lu=bool(sparse_exact_lu),
                use_implicit=bool(use_implicit),
            )
        )
    )
    if bool(force_host_sparse_direct) and bool(sparse_exact_lu):
        direct_wanted = True
    factorization = "lu" if bool(sparse_exact_lu) else "ilu"
    factor_dtype = (
        host_sparse_factor_dtype(
            size=int(n),
            factorization=factorization,
            use_implicit=bool(use_implicit),
        )
        if direct_wanted
        else np.dtype(np.float64)
    )
    cache_key_use = sparse_factor_cache_key(cache_key, factor_dtype) if direct_wanted else cache_key
    build_dense_factors = bool(use_implicit) and (not direct_wanted) and int(n) <= int(sparse_ilu_dense_max)
    build_jax_factors = bool(use_implicit) and (not direct_wanted)
    store_dense = int(n) <= int(sparse_dense_cache_max)
    explicit_sparse_allowed = direct_wanted and bool(
        explicit_sparse_host_direct_allowed(
            sparse_exact_lu=bool(sparse_exact_lu),
            use_implicit=bool(use_implicit),
            active_size=int(n),
        )
    )
    return SparseHostOrILUFactorControls(
        host_sparse_direct_wanted=bool(direct_wanted),
        factor_dtype=np.dtype(factor_dtype),
        cache_key_use=cache_key_use,
        build_dense_factors=bool(build_dense_factors),
        build_jax_factors=bool(build_jax_factors),
        store_dense=bool(store_dense),
        explicit_sparse_allowed=bool(explicit_sparse_allowed),
    )


def build_sparse_ilu_preconditioner_from_cache(
    context: SparseILUPreconditionerBuildContext,
) -> SparseILUPreconditionerBuildResult:
    """Build a JAX ILU preconditioner from cached permutations and factors."""

    cache_entry = context.cache_entry
    perm_r = None if cache_entry is None else getattr(cache_entry, "perm_r", None)
    inv_perm_c = (
        None if cache_entry is None else getattr(cache_entry, "inv_perm_c", None)
    )
    lower_idx = None if cache_entry is None else getattr(cache_entry, "lower_idx", None)
    lower_val = None if cache_entry is None else getattr(cache_entry, "lower_val", None)
    lower_diag = None if cache_entry is None else getattr(cache_entry, "lower_diag", None)
    upper_idx = None if cache_entry is None else getattr(cache_entry, "upper_idx", None)
    upper_val = None if cache_entry is None else getattr(cache_entry, "upper_val", None)
    upper_diag = None if cache_entry is None else getattr(cache_entry, "upper_diag", None)

    if (
        context.l_dense is not None
        and context.u_dense is not None
        and perm_r is not None
        and inv_perm_c is not None
    ):
        import jax.scipy.linalg as jla  # noqa: PLC0415

        l_jnp = jnp.asarray(context.l_dense, dtype=jnp.float64)
        u_jnp = jnp.asarray(context.u_dense, dtype=jnp.float64)

        def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            v = jnp.asarray(v, dtype=jnp.float64)
            v_perm = v[perm_r]
            y = jla.solve_triangular(
                l_jnp,
                v_perm,
                lower=True,
                unit_diagonal=bool(context.l_unit_diag),
            )
            z = jla.solve_triangular(u_jnp, y, lower=False)
            return z[inv_perm_c]

        return SparseILUPreconditionerBuildResult(
            preconditioner=_preconditioner,
            used_dense_triangular=True,
            used_padded_triangular=False,
        )

    if (
        perm_r is not None
        and inv_perm_c is not None
        and lower_idx is not None
        and lower_val is not None
        and (lower_diag is not None or not bool(context.require_lower_diag))
        and upper_idx is not None
        and upper_val is not None
        and upper_diag is not None
    ):

        def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            v = jnp.asarray(v, dtype=jnp.float64)
            v_perm = v[perm_r]
            y = triangular_solve_lower_padded(
                lower_idx=lower_idx,
                lower_val=lower_val,
                b=v_perm,
            )
            z = triangular_solve_upper_padded(
                upper_idx=upper_idx,
                upper_val=upper_val,
                upper_diag=upper_diag,
                b=y,
            )
            return z[inv_perm_c]

        return SparseILUPreconditionerBuildResult(
            preconditioner=_preconditioner,
            used_dense_triangular=False,
            used_padded_triangular=True,
        )

    return SparseILUPreconditionerBuildResult(
        preconditioner=None,
        used_dense_triangular=False,
        used_padded_triangular=False,
    )


def build_sparse_host_scipy_preconditioner(
    context: SparseHostScipyPreconditionerBuildContext,
) -> SparseHostScipyPreconditionerBuildResult:
    """Build host callbacks for SciPy Krylov sparse fallback solves."""

    if context.ilu is None:
        raise RuntimeError(str(context.unavailable_message))

    def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
        x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
        y_np = context.ilu.solve(x_np)
        return jnp.asarray(y_np, dtype=jnp.float64)

    if bool(context.sparse_use_matvec):

        def _matvec(v: jnp.ndarray) -> jnp.ndarray:
            x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
            y_np = context.a_csr_full @ x_np
            return jnp.asarray(y_np, dtype=jnp.float64)

    else:
        _matvec = context.base_matvec

    return SparseHostScipyPreconditionerBuildResult(
        preconditioner=_preconditioner,
        matvec=_matvec,
    )


def run_sparse_host_scipy_gmres(
    context: SparseHostScipyGMRESContext,
) -> tuple[GMRESSolveResult, jnp.ndarray | None]:
    """Run host SciPy GMRES and wrap the result for RHSMode=1 retry gates."""

    x_np, residual_norm, _history = context.gmres_solver(
        matvec=context.matvec,
        b=context.rhs,
        preconditioner=context.preconditioner,
        x0=context.x0,
        tol=float(context.tol),
        atol=float(context.atol),
        restart=int(context.restart),
        maxiter=context.maxiter,
        precondition_side=str(context.precondition_side),
    )
    result = GMRESSolveResult(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
    )
    residual_vec = None
    if context.residual_matvec is not None:
        residual_vec = jnp.asarray(context.rhs, dtype=jnp.float64) - jnp.asarray(
            context.residual_matvec(result.x),
            dtype=jnp.float64,
        )
    return result, residual_vec


def run_sparse_host_retry_candidate(
    context: SparseHostRetryCandidateContext,
) -> SparseHostRetryCandidateResult:
    """Run one sparse-host retry candidate from already-built factors."""

    start_s = perf_counter()
    factor_build = context.factor_build
    matvec_for_accept = context.matvec
    preconditioner_for_accept: ArrayFn | None = None
    result: GMRESSolveResult | None = None
    residual_vec: jnp.ndarray | None = None
    host_sparse_direct_used = False
    label = "sparse LU" if bool(context.sparse_exact_lu) else "sparse ILU"

    if bool(context.host_sparse_direct) and factor_build.ilu is not None:
        if bool(context.host_direct_operator_pc):
            scipy_sparse_build = build_sparse_host_scipy_preconditioner(
                SparseHostScipyPreconditionerBuildContext(
                    ilu=factor_build.ilu,
                    a_csr_full=factor_build.a_csr_full,
                    base_matvec=context.matvec,
                    sparse_use_matvec=False,
                )
            )
            preconditioner_for_accept = scipy_sparse_build.preconditioner
            restart = (
                int(context.operator_pc_restart)
                if context.operator_pc_restart is not None
                else int(context.restart)
            )
            maxiter = (
                int(context.operator_pc_maxiter)
                if context.operator_pc_maxiter is not None
                else context.maxiter
            )
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: sparse LU operator-preconditioned "
                    f"GMRES fallback restart={int(restart)} maxiter={int(maxiter or 0)}",
                )
            result, residual_vec = run_sparse_host_scipy_gmres(
                SparseHostScipyGMRESContext(
                    matvec=context.matvec,
                    rhs=context.rhs,
                    preconditioner=preconditioner_for_accept,
                    x0=context.x0,
                    tol=float(context.tol),
                    atol=float(context.atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    precondition_side=context.precondition_side,
                    gmres_solver=context.gmres_solver,
                    residual_matvec=context.matvec,
                )
            )
        else:
            host_sparse_direct_used = True
            direct_payload = sparse_host_direct_fallback_payload(
                explicit_sparse_factor=factor_build.explicit_sparse_factor,
                explicit_sparse_operator=factor_build.explicit_sparse_operator,
                ilu=factor_build.ilu,
                a_csr_full=factor_build.a_csr_full,
                rhs=context.rhs,
                factor_dtype=context.factor_dtype,
                refine_steps=int(context.refine_steps),
                matvec=context.matvec,
                target=float(context.target),
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                precondition_side=context.precondition_side,
                emit=context.emit,
                backend_name=context.backend_name,
                polish_enabled=context.polish_enabled,
                parse_polish_gmres_config=context.parse_polish_gmres_config,
                direct_solve_with_refinement=context.direct_solve_with_refinement,
                ilu_solve_with_refinement=context.ilu_solve_with_refinement,
                host_sparse_direct_polish=context.host_sparse_direct_polish,
            )
            result = GMRESSolveResult(
                x=direct_payload.x,
                residual_norm=direct_payload.residual_norm,
            )
            residual_vec = direct_payload.residual_vec
    elif bool(context.use_implicit):
        precond_build = build_sparse_ilu_preconditioner_from_cache(
            SparseILUPreconditionerBuildContext(
                cache_entry=context.cache_entry,
                l_dense=factor_build.l_dense,
                u_dense=factor_build.u_dense,
                l_unit_diag=factor_build.l_unit_diag,
                require_lower_diag=bool(context.require_lower_diag),
            )
        )
        preconditioner_for_accept = precond_build.preconditioner
        if preconditioner_for_accept is None:
            if context.emit is not None:
                context.emit(
                    1,
                    f"{'sparse_lu' if context.sparse_exact_lu else 'sparse_ilu'}: "
                    "implicit preconditioner factors unavailable; skipping",
                )
        else:
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: "
                    f"{label} (implicit) fallback",
                )
            result, residual_vec = context.implicit_solver(preconditioner_for_accept)
    else:
        scipy_sparse_build = build_sparse_host_scipy_preconditioner(
            SparseHostScipyPreconditionerBuildContext(
                ilu=factor_build.ilu,
                a_csr_full=factor_build.a_csr_full,
                base_matvec=context.matvec,
                sparse_use_matvec=bool(context.sparse_use_matvec),
            )
        )
        preconditioner_for_accept = scipy_sparse_build.preconditioner
        matvec_for_accept = scipy_sparse_build.matvec
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: "
                f"{label} GMRES fallback",
            )
        result, residual_vec = run_sparse_host_scipy_gmres(
            SparseHostScipyGMRESContext(
                matvec=matvec_for_accept,
                rhs=context.rhs,
                preconditioner=preconditioner_for_accept,
                x0=context.x0,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                precondition_side=context.precondition_side,
                gmres_solver=context.gmres_solver,
                residual_matvec=(
                    matvec_for_accept
                    if bool(context.compute_scipy_residual_vec)
                    else None
                ),
            )
        )

    return SparseHostRetryCandidateResult(
        result=result,
        residual_vec=residual_vec,
        matvec=matvec_for_accept,
        preconditioner=preconditioner_for_accept,
        solve_s=perf_counter() - start_s,
        host_sparse_direct_used=bool(host_sparse_direct_used),
    )


def build_sparse_jax_retry_preconditioner(
    context: SparseJAXRetryPreconditionerBuildContext,
) -> ArrayFn:
    """Build the sparse-JAX retry preconditioner and emit its progress line."""

    preconditioner = context.builder(
        matvec=context.matvec,
        n=int(context.n),
        dtype=context.dtype,
        cache_key=context.cache_key,
        drop_tol=float(context.drop_tol),
        drop_rel=float(context.drop_rel),
        reg=float(context.reg),
        omega=float(context.omega),
        sweeps=int(context.sweeps),
        emit=context.emit,
    )
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse JAX Jacobi fallback "
            f"(sweeps={int(context.sweeps)} omega={float(context.omega):.2f})",
        )
    return preconditioner


__all__ = (
    "ExplicitSparseHostDirectBranchContext",
    "ExplicitSparseMinimumNormBranchContext",
    "ExplicitSparseOperatorBuildPolicy",
    "ExplicitSparseOperatorBuildResult",
    "SparseHostDirectFactorSolvePayload",
    "SparseHostDirectFallbackPayload",
    "SparseHostDirectPayload",
    "SparseHostDirectPolishPayload",
    "SparseHostOrILUFactorBuildContext",
    "SparseHostOrILUFactorBuildResult",
    "SparseHostOrILUFactorControls",
    "SparseHostRetryCandidateContext",
    "SparseHostRetryCandidateResult",
    "SparseHostScipyGMRESContext",
    "SparseHostScipyPreconditionerBuildContext",
    "SparseHostScipyPreconditionerBuildResult",
    "SparseILUPreconditionerBuildContext",
    "SparseILUPreconditionerBuildResult",
    "SparseJAXRetryPreconditionerBuildContext",
    "SparseMinimumNormPayload",
    "SparseMinimumNormPolicy",
    "apply_sparse_host_direct_polish_if_needed",
    "build_explicit_sparse_operator_from_pattern",
    "build_sparse_host_or_ilu_factor",
    "build_sparse_host_scipy_preconditioner",
    "build_sparse_ilu_preconditioner_from_cache",
    "build_sparse_jax_retry_preconditioner",
    "explicit_sparse_pattern_progress_messages",
    "resolve_explicit_sparse_operator_build_policy",
    "resolve_sparse_host_or_ilu_factor_controls",
    "resolve_sparse_minimum_norm_policy",
    "run_sparse_host_retry_candidate",
    "run_sparse_host_scipy_gmres",
    "solve_explicit_sparse_minimum_norm_branch",
    "solve_explicit_sparse_host_direct_branch",
    "solve_sparse_host_direct_from_available_factor",
    "sparse_host_direct_fallback_payload",
    "sparse_host_direct_solve_from_pattern",
    "sparse_host_direct_solve_payload",
    "sparse_minimum_norm_solve_from_pattern",
    "sparse_minimum_norm_solve_payload",
    "sparse_minimum_norm_start_message",
    "validate_explicit_sparse_host_request",
)
