"""Explicit host-sparse and minimum-norm solve helpers for profile response."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..residual import (
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
)
from ..setup import SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS


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


__all__ = (
    "ExplicitSparseMinimumNormBranchContext",
    "ExplicitSparseOperatorBuildPolicy",
    "ExplicitSparseOperatorBuildResult",
    "SparseMinimumNormPayload",
    "SparseMinimumNormPolicy",
    "build_explicit_sparse_operator_from_pattern",
    "explicit_sparse_pattern_progress_messages",
    "resolve_explicit_sparse_operator_build_policy",
    "resolve_sparse_minimum_norm_policy",
    "solve_explicit_sparse_minimum_norm_branch",
    "sparse_minimum_norm_solve_from_pattern",
    "sparse_minimum_norm_solve_payload",
    "sparse_minimum_norm_start_message",
    "validate_explicit_sparse_host_request",
)
