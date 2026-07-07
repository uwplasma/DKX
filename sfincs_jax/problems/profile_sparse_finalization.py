"""Generic sparse-PC GMRES finalization and retry helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .profile_diagnostics import (
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    sparse_pc_direct_tail_result_metadata,
    sparse_pc_factor_preflight_result_metadata,
    sparse_pc_factor_preflight_result_metadata_from_context,
    sparse_pc_gmres_result_metadata,
    sparse_pc_gmres_static_metadata,
    sparse_pc_gmres_static_metadata_from_context,
    sparse_pc_pattern_result_metadata,
    sparse_pc_pattern_result_metadata_from_context,
)
from .profile_residual import (
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
)
from .profile_sparse_direct import (
    SparsePCDirectTailFinalMetadataContext,
    sparse_pc_direct_tail_final_metadata,
)


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class SparsePCGMRESResult:
    """Measured result from one sparse-PC GMRES attempt."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    solve_s: float


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


def run_sparse_pc_gmres_once_for_retry(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> tuple[np.ndarray, float, float, tuple[float, ...], float]:
    """Run sparse-PC GMRES and return the tuple contract used by dtype retry."""

    result = run_sparse_pc_gmres_once(
        context=context,
        x0=x0,
        maxiter=int(maxiter),
    )
    return (
        result.x,
        float(result.residual_norm),
        float(result.preconditioned_residual_norm),
        tuple(float(value) for value in result.history),
        float(result.solve_s),
    )


@dataclass(frozen=True)
class SparsePCGMRESFinalPayload:
    """Driver-independent payload for constructing the final sparse-PC result."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]

@dataclass(frozen=True)
class SparsePCPostMinresFinalizationContext:
    """Dependencies for final optional sparse-PC post-MinRes polishing."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    steps: int
    alpha_clip: float
    min_improvement: float
    target: float

@dataclass(frozen=True)
class SparsePCFactorDtypeRetryFinalizationContext:
    """Dependencies for optional sparse-PC factor dtype retry."""

    factor_matvec: ArrayFn
    linear_size: int
    rhs_dtype: np.dtype
    pattern: object
    emit: EmitFn | None
    constrained_pas_pc: bool
    tokamak_fp_pc: bool
    fortran_reduced_sparse_pc: bool
    default_permc_spec: str
    default_factor_kind: str
    default_ilu_fill_factor: float
    default_ilu_drop_tol: float
    default_pattern_color_batch: int
    x0_fallback: jnp.ndarray
    pc_maxiter: int
    elapsed_s: Callable[[], float]

@dataclass(frozen=True)
class SparsePCGMRESFinalizationContext:
    """Explicit inputs for final sparse-PC GMRES retry, polish, and payload."""

    diagnostic_state: Mapping[str, object]
    result: SparsePCGMRESResult
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    pc_factor_s: float
    setup_s: float | None
    post_minres: SparsePCPostMinresFinalizationContext | None = None
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext | None = None

@dataclass(frozen=True)
class SparsePCGMRESFinalResultContext:
    """Result and setup timing from the first sparse-PC GMRES attempt."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: Sequence[float] | None
    solve_s: float
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    pc_factor_s: float
    setup_s: float


@dataclass(frozen=True)
class SparsePCGMRESFinalizationBundleContext:
    """Typed sparse-PC finalization inputs that the driver passes as one bundle."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    direct_tail: SparsePCDirectTailFinalMetadataContext
    factor_preflight: SparsePCFactorPreflightMetadataContext
    pattern: SparsePCPatternMetadataContext
    static: SparsePCGMRESStaticMetadataContext
    result: SparsePCGMRESFinalResultContext
    post_minres: SparsePCPostMinresFinalizationContext
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext


def _unique_state_keys(*groups: Sequence[str]) -> tuple[str, ...]:
    """Return state keys in first-seen order without duplicate diagnostics."""

    keys: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for key in group:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


_SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS = (
    "atol",
    "mv_count",
    "rhs_norm",
    "target",
    "tol",
)

_SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS = (
    "fortran_reduced_sparse_pc",
    "fortran_reduced_sparse_pc_backend",
    "fortran_reduced_sparse_pc_backend_reason",
    "fortran_reduced_xblock_min_size",
    "op",
    "pc_maxiter",
    "pc_restart",
    "pc_shift",
    "preconditioner_species",
    "preconditioner_x",
    "preconditioner_x_min_l",
    "preconditioner_xi",
    "sparse_pc_default_factor_kind",
    "sparse_pc_default_ilu_drop_tol",
    "sparse_pc_default_ilu_fill_factor",
    "sparse_pc_default_pattern_color_batch",
    "sparse_pc_default_permc_spec",
    "sparse_pc_factor_dtype_initial",
    "sparse_pc_factorization",
    "sparse_pc_first_attempt_maxiter",
    "sparse_pc_fp_dense_velocity_block",
    "sparse_pc_linear_size",
    "sparse_pc_permc_spec",
    "sparse_pc_preconditioner_operator",
    "sparse_pc_use_active_dof",
)

_SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
)

_SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
    _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
)


def sparse_pc_gmres_finalization_solve_state_keys() -> tuple[str, ...]:
    """Return finalizer keys copied from solve scope before metadata injection."""

    return _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS


def sparse_pc_gmres_finalization_solve_scope_keys() -> tuple[str, ...]:
    """Return raw solve-scope keys needed to build sparse-PC finalization state."""

    return _SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS


@dataclass(frozen=True)
class SparsePCGMRESFinalizationStateContext:
    """Explicit sparse-PC final metadata state inputs."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    sparse_pc_direct_tail_metadata: object
    sparse_pc_factor_preflight_metadata: object
    sparse_pc_pattern_metadata: object
    sparse_pc_static_metadata: object


def sparse_pc_gmres_finalization_state_from_context(
    context: SparsePCGMRESFinalizationStateContext,
) -> dict[str, object]:
    """Build sparse-PC finalization metadata state from typed inputs."""

    return {
        "atol": context.atol,
        "mv_count": context.mv_count,
        "rhs_norm": context.rhs_norm,
        "target": context.target,
        "tol": context.tol,
        "sparse_pc_direct_tail_metadata": context.sparse_pc_direct_tail_metadata,
        "sparse_pc_factor_preflight_metadata": context.sparse_pc_factor_preflight_metadata,
        "sparse_pc_pattern_metadata": context.sparse_pc_pattern_metadata,
        "sparse_pc_static_metadata": context.sparse_pc_static_metadata,
    }


def sparse_pc_gmres_finalization_state_from_solve_scope(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Copy only sparse-PC finalizer state and precompute direct-tail metadata."""

    required_keys = _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS
    if "sparse_pc_static_metadata" not in scope:
        required_keys = _unique_state_keys(
            required_keys,
            _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
        )
    missing = tuple(key for key in required_keys if key not in scope)
    if missing:
        joined = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise KeyError(f"sparse-PC GMRES finalization state missing: {joined}{suffix}")
    state = {key: scope[key] for key in _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS}
    if "sparse_pc_direct_tail_metadata" in scope:
        direct_tail_metadata = scope["sparse_pc_direct_tail_metadata"]
    else:
        direct_tail_metadata = sparse_pc_direct_tail_result_metadata(scope)
    if "sparse_pc_factor_preflight_metadata" in scope:
        factor_preflight_metadata = scope["sparse_pc_factor_preflight_metadata"]
    else:
        factor_preflight_metadata = sparse_pc_factor_preflight_result_metadata(scope)
    if "sparse_pc_pattern_metadata" in scope:
        pattern_metadata = scope["sparse_pc_pattern_metadata"]
    else:
        pattern_metadata = sparse_pc_pattern_result_metadata(scope)
    if "sparse_pc_static_metadata" in scope:
        static_metadata = scope["sparse_pc_static_metadata"]
    else:
        static_metadata = sparse_pc_gmres_static_metadata(scope)
    return sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=state["atol"],
            mv_count=state["mv_count"],
            rhs_norm=state["rhs_norm"],
            target=state["target"],
            tol=state["tol"],
            sparse_pc_direct_tail_metadata=direct_tail_metadata,
            sparse_pc_factor_preflight_metadata=factor_preflight_metadata,
            sparse_pc_pattern_metadata=pattern_metadata,
            sparse_pc_static_metadata=static_metadata,
        )
    )


def sparse_pc_gmres_finalization_bundle_from_solve_scope(
    scope: Mapping[str, object],
    *,
    result: SparsePCGMRESFinalResultContext,
    post_minres: SparsePCPostMinresFinalizationContext,
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the typed sparse-PC finalization bundle from driver-local names."""

    return SparsePCGMRESFinalizationBundleContext(
        atol=scope["atol"],
        mv_count=scope["mv_count"],
        rhs_norm=scope["rhs_norm"],
        target=scope["target"],
        tol=scope["tol"],
        direct_tail=SparsePCDirectTailFinalMetadataContext(
            structured_pc_preflight_required=bool(
                scope["structured_pc_preflight_required"]
            ),
            structured_pc_preflight_required_min_size=int(
                scope["structured_pc_preflight_required_min_size"]
            ),
            materialization=scope["direct_tail_materialization"],
            structured_admission=scope["direct_tail_structured_admission"],
            structured_max_nbytes=scope["direct_tail_structured_max_nbytes"],
            structured_pc_selected=bool(scope["direct_tail_structured_pc_selected"]),
            structured_pc_reason=scope["direct_tail_structured_pc_reason"],
            structured_pc_error=scope["direct_tail_structured_pc_error"],
            structured_pc_metadata=scope["direct_tail_structured_pc_metadata"],
            support_mode_preflight_requested=bool(
                scope["direct_tail_support_mode_preflight_requested"]
            ),
            support_mode_preflight_selected=bool(
                scope["direct_tail_support_mode_preflight_selected"]
            ),
            support_mode_preflight_error=scope[
                "direct_tail_support_mode_preflight_error"
            ],
            support_mode_preflight_metadata=scope[
                "direct_tail_support_mode_preflight_metadata"
            ],
        ),
        factor_preflight=SparsePCFactorPreflightMetadataContext(
            enabled=bool(scope["factor_preflight_enabled"]),
            required=bool(scope["factor_preflight_required"]),
            seed_enabled=bool(scope["factor_preflight_seed_enabled"]),
            seed_used=bool(scope["factor_preflight_seed_used"]),
            passed=scope["factor_preflight_passed"],
            error=scope["factor_preflight_error"],
            residual_before=scope["factor_preflight_residual_before"],
            residual_after=scope["factor_preflight_residual_after"],
            improvement_ratio=scope["factor_preflight_improvement_ratio"],
            target_ratio=scope["factor_preflight_target_ratio"],
            max_target_ratio=float(scope["factor_preflight_max_target_ratio"]),
            residual_diagnostics=scope["factor_preflight_residual_diagnostics"],
        ),
        pattern=SparsePCPatternMetadataContext(
            summary=scope["summary"],
            scope=scope["sparse_pattern_scope"],
            build_s=float(scope["pattern_build_s"]),
        ),
        static=SparsePCGMRESStaticMetadataContext(
            op=scope["op"],
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            fortran_reduced_sparse_pc_backend=scope["fortran_reduced_sparse_pc_backend"],
            fortran_reduced_sparse_pc_backend_reason=scope[
                "fortran_reduced_sparse_pc_backend_reason"
            ],
            fortran_reduced_xblock_min_size=scope["fortran_reduced_xblock_min_size"],
            pc_restart=int(scope["pc_restart"]),
            pc_maxiter=int(scope["pc_maxiter"]),
            sparse_pc_first_attempt_maxiter=int(scope["sparse_pc_first_attempt_maxiter"]),
            pc_shift=float(scope["pc_shift"]),
            sparse_pc_factor_dtype_initial=scope["sparse_pc_factor_dtype_initial"],
            sparse_pc_preconditioner_operator=scope["sparse_pc_preconditioner_operator"],
            sparse_pc_factorization=scope["sparse_pc_factorization"],
            sparse_pc_default_factor_kind=scope["sparse_pc_default_factor_kind"],
            sparse_pc_default_ilu_fill_factor=float(scope["sparse_pc_default_ilu_fill_factor"]),
            sparse_pc_default_ilu_drop_tol=float(scope["sparse_pc_default_ilu_drop_tol"]),
            sparse_pc_default_pattern_color_batch=int(scope["sparse_pc_default_pattern_color_batch"]),
            preconditioner_x=int(scope["preconditioner_x"]),
            preconditioner_x_min_l=int(scope["preconditioner_x_min_l"]),
            preconditioner_xi=int(scope["preconditioner_xi"]),
            preconditioner_species=int(scope["preconditioner_species"]),
            sparse_pc_permc_spec=scope["sparse_pc_permc_spec"],
            sparse_pc_default_permc_spec=scope["sparse_pc_default_permc_spec"],
            sparse_pc_use_active_dof=bool(scope["sparse_pc_use_active_dof"]),
            sparse_pc_linear_size=int(scope["sparse_pc_linear_size"]),
            sparse_pc_fp_dense_velocity_block=scope["sparse_pc_fp_dense_velocity_block"],
        ),
        result=result,
        post_minres=post_minres,
        dtype_retry=dtype_retry,
    )


def sparse_pc_gmres_finalization_bundle_from_solve_result(
    scope: Mapping[str, object],
    *,
    x: np.ndarray,
    residual_norm: float,
    preconditioned_residual_norm: float,
    history: Sequence[float] | None,
    solve_s: float,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the full sparse-PC finalization bundle from the first GMRES result."""

    return sparse_pc_gmres_finalization_bundle_from_solve_scope(
        scope,
        result=SparsePCGMRESFinalResultContext(
            x=np.asarray(x, dtype=np.float64),
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(preconditioned_residual_norm),
            history=tuple(float(v) for v in (history or ())),
            solve_s=float(solve_s),
            factor_dtype_used=np.dtype(scope["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=scope["sparse_pc_factor_dtype_retry"],
            operator_bundle=scope["_operator_bundle_pc"],
            factor_bundle=scope["factor_bundle_pc"],
            pc_factor_s=float(scope["pc_factor_s"]),
            setup_s=float(scope["setup_s"]),
        ),
        post_minres=SparsePCPostMinresFinalizationContext(
            matvec=scope["_mv_true"],
            rhs=scope["sparse_pc_rhs"],
            preconditioner=scope["_precond_sparse"],
            emit=scope["emit"],
            elapsed_s=scope["sparse_timer"].elapsed_s,
            pc_form=scope["pc_form"],
            steps=int(scope["sparse_pc_post_minres_steps"]),
            alpha_clip=float(scope["sparse_pc_post_minres_alpha_clip"]),
            min_improvement=float(scope["sparse_pc_post_minres_min_improvement"]),
            target=float(scope["target"]),
        ),
        dtype_retry=SparsePCFactorDtypeRetryFinalizationContext(
            factor_matvec=scope["_sparse_pc_factor_mv"],
            linear_size=int(scope["sparse_pc_linear_size"]),
            rhs_dtype=np.dtype(scope["rhs"].dtype),
            pattern=scope["pattern"],
            emit=scope["emit"],
            constrained_pas_pc=bool(scope["constrained_pas_pc"]),
            tokamak_fp_pc=bool(scope["tokamak_fp_pc"]),
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            default_permc_spec=scope["sparse_pc_default_permc_spec"],
            default_factor_kind=scope["sparse_pc_default_factor_kind"],
            default_ilu_fill_factor=float(scope["sparse_pc_default_ilu_fill_factor"]),
            default_ilu_drop_tol=float(scope["sparse_pc_default_ilu_drop_tol"]),
            default_pattern_color_batch=int(scope["sparse_pc_default_pattern_color_batch"]),
            x0_fallback=scope["x0_sparse"],
            pc_maxiter=int(scope["pc_maxiter"]),
            elapsed_s=scope["sparse_timer"].elapsed_s,
        ),
    )


@dataclass(frozen=True)
class SparsePCGMRESCompletionMessageContext:
    """Fields used to format the sparse-PC GMRES completion progress line."""

    elapsed_s: float
    iterations: int
    matvecs: int
    residual_norm: float
    target: float
    preconditioned_residual_norm: float
    history: Sequence[float]

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
    solver_label: str = "sparse_pc_gmres"

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
class SparsePCPostMinresUpdateContext:
    """Current sparse-PC solve state for optional post-minres polishing."""

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
    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    solve_s: float
    target: float
    solver_label: str = "sparse_pc_gmres"

@dataclass(frozen=True)
class SparsePCPostMinresUpdateResult:
    """Updated sparse-PC state and diagnostics after optional post-minres."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    alphas: tuple[float, ...]
    residual_before: float | None
    residual_after: float | None
    error: str | None
    solve_s: float

@dataclass(frozen=True)
class SparsePCFactorDtypeRetryDecision:
    """Decision for retrying a sparse-PC factor with higher precision."""

    retry: bool
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None

@dataclass(frozen=True)
class SparsePCFactorDtypeRetryContext:
    """Callbacks and state for retrying a sparse-PC factor in higher precision."""

    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    residual_norm: float
    preconditioned_residual_norm: float
    history: Sequence[float]
    target: float
    x: np.ndarray
    x0_fallback: jnp.ndarray
    solve_s: float
    pc_maxiter: int
    operator_bundle: Any
    factor_bundle: Any
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    build_factor: Callable[[np.dtype], tuple[Any, Any]]
    run_gmres_once: Callable[[jnp.ndarray, int], tuple[np.ndarray, float, float, Sequence[float], float]]

@dataclass(frozen=True)
class SparsePCFactorDtypeRetryResult:
    """Sparse-PC factor dtype retry result and updated solve state."""

    retried: bool
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    factor_s_increment: float
    setup_s: float | None
    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    solve_s: float

def evaluate_sparse_pc_factor_dtype_retry(
    *,
    factor_dtype_used: np.dtype,
    residual_norm: float,
    target: float,
) -> SparsePCFactorDtypeRetryDecision:
    """Decide whether an FP32 sparse-PC factor should retry in FP64."""

    dtype_used = np.dtype(factor_dtype_used)
    should_retry = bool(
        dtype_used == np.dtype(np.float32)
        and (
            not np.isfinite(float(residual_norm))
            or float(residual_norm) > float(target)
        )
    )
    if not should_retry:
        return SparsePCFactorDtypeRetryDecision(
            retry=False,
            factor_dtype_used=dtype_used,
            factor_dtype_retry=None,
        )
    return SparsePCFactorDtypeRetryDecision(
        retry=True,
        factor_dtype_used=np.dtype(np.float64),
        factor_dtype_retry="float64",
    )

def sparse_pc_factor_dtype_retry_initial_guess(
    x_candidate: np.ndarray,
    fallback: jnp.ndarray,
) -> jnp.ndarray:
    """Use the first solve as the retry seed only if it is finite."""

    x_np = np.asarray(x_candidate)
    if np.all(np.isfinite(x_np)):
        return jnp.asarray(x_np, dtype=jnp.float64)
    return fallback

def retry_sparse_pc_factor_dtype_if_needed(
    context: SparsePCFactorDtypeRetryContext,
) -> SparsePCFactorDtypeRetryResult:
    """Retry an FP32 sparse-PC factor in FP64 when the probe residual fails."""

    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=context.factor_dtype_used,
        residual_norm=float(context.residual_norm),
        target=float(context.target),
    )
    if not bool(decision.retry):
        return SparsePCFactorDtypeRetryResult(
            retried=False,
            factor_dtype_used=np.dtype(context.factor_dtype_used),
            factor_dtype_retry=context.factor_dtype_retry,
            operator_bundle=context.operator_bundle,
            factor_bundle=context.factor_bundle,
            factor_s_increment=0.0,
            setup_s=None,
            x=np.asarray(context.x, dtype=np.float64),
            residual_norm=float(context.residual_norm),
            preconditioned_residual_norm=float(context.preconditioned_residual_norm),
            history=tuple(float(v) for v in (context.history or ())),
            solve_s=float(context.solve_s),
        )

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres retrying preconditioner "
            f"with factor_dtype={decision.factor_dtype_used.name} "
            f"after residual={float(context.residual_norm):.6e} target={float(context.target):.6e}",
        )
    retry_factor_start_s = float(context.elapsed_s())
    operator_bundle, factor_bundle = context.build_factor(decision.factor_dtype_used)
    factor_s_increment = float(context.elapsed_s()) - retry_factor_start_s
    setup_s = float(context.elapsed_s())
    x0_retry = sparse_pc_factor_dtype_retry_initial_guess(context.x, context.x0_fallback)
    x, residual_norm, rn_pc, history, solve_s_retry = context.run_gmres_once(
        x0_retry,
        int(context.pc_maxiter),
    )
    return SparsePCFactorDtypeRetryResult(
        retried=True,
        factor_dtype_used=np.dtype(decision.factor_dtype_used),
        factor_dtype_retry=decision.factor_dtype_retry,
        operator_bundle=operator_bundle,
        factor_bundle=factor_bundle,
        factor_s_increment=float(factor_s_increment),
        setup_s=float(setup_s),
        x=np.asarray(x, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(rn_pc),
        history=tuple(float(v) for v in (history or ())),
        solve_s=float(context.solve_s) + float(solve_s_retry),
    )

def retry_sparse_pc_factor_dtype_from_solve_state(
    state: Mapping[str, object],
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
) -> SparsePCFactorDtypeRetryResult:
    """Retry sparse-PC factor precision using stored sparse-PC solve state."""

    def build_factor(factor_dtype_arg: np.dtype) -> tuple[Any, Any]:
        return build_host_sparse_direct_factor_from_matvec(
            matvec=state["_sparse_pc_factor_mv"],
            n=int(state["sparse_pc_linear_size"]),
            dtype=state["rhs"].dtype,
            factor_dtype=np.dtype(factor_dtype_arg),
            pattern=state["pattern"],
            emit=state["emit"],
            default_diag_pivot_thresh=(
                0.0
                if (
                    bool(state["constrained_pas_pc"])
                    or bool(state["tokamak_fp_pc"])
                    or bool(state["fortran_reduced_sparse_pc"])
                )
                else 1.0
            ),
            default_permc_spec=state["sparse_pc_default_permc_spec"],
            default_factor_kind=state["sparse_pc_default_factor_kind"],
            default_ilu_fill_factor=float(state["sparse_pc_default_ilu_fill_factor"]),
            default_ilu_drop_tol=float(state["sparse_pc_default_ilu_drop_tol"]),
            default_pattern_color_batch=int(state["sparse_pc_default_pattern_color_batch"]),
        )

    return retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(state["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=state["sparse_pc_factor_dtype_retry"],
            residual_norm=float(state["residual_norm_sparse_pc"]),
            preconditioned_residual_norm=float(state["rn_pc"]),
            history=state["history"],
            target=float(state["target"]),
            x=np.asarray(state["x_np"], dtype=np.float64),
            x0_fallback=state["x0_sparse"],
            solve_s=float(state["solve_s"]),
            pc_maxiter=int(state["pc_maxiter"]),
            operator_bundle=state["_operator_bundle_pc"],
            factor_bundle=state["factor_bundle_pc"],
            elapsed_s=state["sparse_timer"].elapsed_s,
            emit=state["emit"],
            build_factor=build_factor,
            run_gmres_once=lambda x0, maxiter: run_sparse_pc_gmres_once_callback(
                x0,
                maxiter_arg=int(maxiter),
            ),
        )
    )

def retry_sparse_pc_factor_dtype_from_finalization_context(
    context: SparsePCFactorDtypeRetryFinalizationContext,
    *,
    factor_dtype_used: np.dtype,
    factor_dtype_retry: str | None,
    residual_norm: float,
    preconditioned_residual_norm: float,
    history: Sequence[float],
    target: float,
    x: np.ndarray,
    solve_s: float,
    operator_bundle: Any,
    factor_bundle: Any,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
) -> SparsePCFactorDtypeRetryResult:
    """Retry sparse-PC factor precision from explicit finalization inputs."""

    def build_factor(factor_dtype_arg: np.dtype) -> tuple[Any, Any]:
        return build_host_sparse_direct_factor_from_matvec(
            matvec=context.factor_matvec,
            n=int(context.linear_size),
            dtype=np.dtype(context.rhs_dtype),
            factor_dtype=np.dtype(factor_dtype_arg),
            pattern=context.pattern,
            emit=context.emit,
            default_diag_pivot_thresh=(
                0.0
                if (
                    bool(context.constrained_pas_pc)
                    or bool(context.tokamak_fp_pc)
                    or bool(context.fortran_reduced_sparse_pc)
                )
                else 1.0
            ),
            default_permc_spec=context.default_permc_spec,
            default_factor_kind=context.default_factor_kind,
            default_ilu_fill_factor=float(context.default_ilu_fill_factor),
            default_ilu_drop_tol=float(context.default_ilu_drop_tol),
            default_pattern_color_batch=int(context.default_pattern_color_batch),
        )

    return retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(factor_dtype_used),
            factor_dtype_retry=factor_dtype_retry,
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(preconditioned_residual_norm),
            history=tuple(float(v) for v in (history or ())),
            target=float(target),
            x=np.asarray(x, dtype=np.float64),
            x0_fallback=context.x0_fallback,
            solve_s=float(solve_s),
            pc_maxiter=int(context.pc_maxiter),
            operator_bundle=operator_bundle,
            factor_bundle=factor_bundle,
            elapsed_s=context.elapsed_s,
            emit=context.emit,
            build_factor=build_factor,
            run_gmres_once=lambda x0, maxiter: run_sparse_pc_gmres_once_callback(
                x0,
                maxiter_arg=int(maxiter),
            ),
        )
    )

def sparse_pc_gmres_completion_message(
    context: SparsePCGMRESCompletionMessageContext,
) -> str:
    """Format the final sparse-PC GMRES progress message."""

    pc_suffix = (
        f" preconditioned_residual={float(context.preconditioned_residual_norm):.6e}"
        if np.isfinite(float(context.preconditioned_residual_norm))
        else ""
    )
    history = tuple(float(v) for v in (context.history or ()))
    if history:
        pc_suffix = f"{pc_suffix} ksp_residual={float(history[-1]):.6e}"
    return (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        f"elapsed_s={float(context.elapsed_s):.3f} iters={int(context.iterations)} "
        f"matvecs={int(context.matvecs)} residual={float(context.residual_norm):.6e} "
        f"target={float(context.target):.6e}{pc_suffix}"
    )

def emit_sparse_pc_gmres_completion_from_solve_state(
    state: Mapping[str, object],
) -> None:
    """Emit the sparse-PC GMRES completion line from stored solve state."""

    emit = state["emit"]
    if emit is None:
        return
    emit(
        0,
        sparse_pc_gmres_completion_message(
            SparsePCGMRESCompletionMessageContext(
                elapsed_s=float(state["sparse_timer"].elapsed_s()),
                iterations=int(len(state["history"] or ())),
                matvecs=int(state["mv_count"]),
                residual_norm=float(state["residual_norm_sparse_pc"]),
                target=float(state["target"]),
                preconditioned_residual_norm=float(state["rn_pc"]),
                history=state["history"],
            )
        ),
    )

def sparse_pc_gmres_final_payload_from_solve_state(
    state: Mapping[str, object],
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final sparse-PC solve payload from stored solve state."""

    residual_norm = float(state["residual_norm_sparse_pc"])
    metadata_state = state if isinstance(state, MutableMapping) else dict(state)
    metadata_state["sparse_pc_accepted_converged"] = profile_residual_converged(
        residual_norm,
        profile_residual_target(
            atol=float(state["atol"]),
            tol=float(state["tol"]),
            rhs_norm=float(state["rhs_norm"]),
        ),
    )
    metadata_state["sparse_pc_factor_quality_rejected"] = not profile_residual_converged(
        residual_norm,
        float(state["target"]),
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(state["x_np"], dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=sparse_pc_gmres_result_metadata(metadata_state),
    )

def finalize_sparse_pc_gmres_from_solve_state(
    state: Mapping[str, object],
    *,
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Apply final sparse-PC polish, emit completion, and build solve payload.

    This helper keeps the driver from manually copying the post-minres result
    back into its local variables before constructing the final metadata. The
    broad metadata schema is still mapping-backed for compatibility, but the
    mutation is isolated to a copied state map instead of scattered through the
    solve loop.
    """
    post_minres = apply_sparse_pc_post_minres_from_solve_state(
        state,
        minres_correction=minres_correction,
    )
    final_state = state.__class__(state) if isinstance(state, MutableMapping) else dict(state)
    final_state.update(
        {
            "x_np": post_minres.x,
            "residual_norm_sparse_pc": float(post_minres.residual_norm),
            "rn_pc": float(post_minres.preconditioned_residual_norm),
            "sparse_pc_post_minres_history": post_minres.history,
            "sparse_pc_post_minres_alphas": post_minres.alphas,
            "sparse_pc_post_minres_residual_before": post_minres.residual_before,
            "sparse_pc_post_minres_residual_after": post_minres.residual_after,
            "sparse_pc_post_minres_error": post_minres.error,
            "solve_s": float(post_minres.solve_s),
        }
    )
    emit_sparse_pc_gmres_completion_from_solve_state(final_state)
    return sparse_pc_gmres_final_payload_from_solve_state(
        final_state,
        expand_reduced=expand_reduced,
    )

def finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state(
    state: Mapping[str, object],
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Retry factor dtype if needed, then build the final sparse-PC payload."""

    return finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=state,
            result=SparsePCGMRESResult(
                x=np.asarray(state["x_np"], dtype=np.float64),
                residual_norm=float(state["residual_norm_sparse_pc"]),
                preconditioned_residual_norm=float(state["rn_pc"]),
                history=tuple(float(v) for v in (state["history"] or ())),
                solve_s=float(state["solve_s"]),
            ),
            factor_dtype_used=np.dtype(state["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=state["sparse_pc_factor_dtype_retry"],
            operator_bundle=state["_operator_bundle_pc"],
            factor_bundle=state["factor_bundle_pc"],
            pc_factor_s=float(state["pc_factor_s"]),
            setup_s=float(state["setup_s"]) if "setup_s" in state else None,
        ),
        build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
    )


def finalize_sparse_pc_gmres_bundle(
    context: SparsePCGMRESFinalizationBundleContext,
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build typed sparse-PC final metadata, apply retry/polish, and return payload."""

    diagnostic_state = sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=context.atol,
            mv_count=context.mv_count,
            rhs_norm=context.rhs_norm,
            target=context.target,
            tol=context.tol,
            sparse_pc_direct_tail_metadata=sparse_pc_direct_tail_final_metadata(
                context.direct_tail
            ),
            sparse_pc_factor_preflight_metadata=(
                sparse_pc_factor_preflight_result_metadata_from_context(
                    context.factor_preflight
                )
            ),
            sparse_pc_pattern_metadata=sparse_pc_pattern_result_metadata_from_context(
                context.pattern
            ),
            sparse_pc_static_metadata=sparse_pc_gmres_static_metadata_from_context(
                context.static
            ),
        )
    )
    result = context.result
    return finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=diagnostic_state,
            result=SparsePCGMRESResult(
                x=np.asarray(result.x, dtype=np.float64),
                residual_norm=float(result.residual_norm),
                preconditioned_residual_norm=float(
                    result.preconditioned_residual_norm
                ),
                history=tuple(float(v) for v in (result.history or ())),
                solve_s=float(result.solve_s),
            ),
            factor_dtype_used=np.dtype(result.factor_dtype_used),
            factor_dtype_retry=result.factor_dtype_retry,
            operator_bundle=result.operator_bundle,
            factor_bundle=result.factor_bundle,
            pc_factor_s=float(result.pc_factor_s),
            setup_s=float(result.setup_s),
            post_minres=context.post_minres,
            dtype_retry=context.dtype_retry,
        ),
        build_host_sparse_direct_factor_from_matvec=(
            build_host_sparse_direct_factor_from_matvec
        ),
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
    )


def finalize_sparse_pc_gmres_with_dtype_retry(
    context: SparsePCGMRESFinalizationContext,
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Retry factor dtype if needed from explicit solve state, then finalize."""

    initial_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    initial_state.update(
        {
            "sparse_pc_factor_dtype_used": np.dtype(context.factor_dtype_used),
            "sparse_pc_factor_dtype_retry": context.factor_dtype_retry,
            "_operator_bundle_pc": context.operator_bundle,
            "factor_bundle_pc": context.factor_bundle,
            "pc_factor_s": float(context.pc_factor_s),
            "x_np": np.asarray(context.result.x, dtype=np.float64),
            "residual_norm_sparse_pc": float(context.result.residual_norm),
            "rn_pc": float(context.result.preconditioned_residual_norm),
            "history": tuple(float(v) for v in (context.result.history or ())),
            "solve_s": float(context.result.solve_s),
        }
    )
    if context.setup_s is not None:
        initial_state["setup_s"] = float(context.setup_s)
    if context.dtype_retry is None:
        retry_result = retry_sparse_pc_factor_dtype_from_solve_state(
            initial_state,
            build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
            run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        )
    else:
        retry_result = retry_sparse_pc_factor_dtype_from_finalization_context(
            context.dtype_retry,
            factor_dtype_used=np.dtype(context.factor_dtype_used),
            factor_dtype_retry=context.factor_dtype_retry,
            residual_norm=float(context.result.residual_norm),
            preconditioned_residual_norm=float(
                context.result.preconditioned_residual_norm
            ),
            history=context.result.history,
            target=float(initial_state["target"]),
            x=np.asarray(context.result.x, dtype=np.float64),
            solve_s=float(context.result.solve_s),
            operator_bundle=context.operator_bundle,
            factor_bundle=context.factor_bundle,
            build_host_sparse_direct_factor_from_matvec=(
                build_host_sparse_direct_factor_from_matvec
            ),
            run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        )
    final_state = (
        initial_state.__class__(initial_state)
        if isinstance(initial_state, MutableMapping)
        else dict(initial_state)
    )
    final_state.update(
        {
            "sparse_pc_factor_dtype_used": retry_result.factor_dtype_used,
            "sparse_pc_factor_dtype_retry": retry_result.factor_dtype_retry,
            "_operator_bundle_pc": retry_result.operator_bundle,
            "factor_bundle_pc": retry_result.factor_bundle,
            "pc_factor_s": float(context.pc_factor_s) + float(retry_result.factor_s_increment),
            "x_np": retry_result.x,
            "residual_norm_sparse_pc": float(retry_result.residual_norm),
            "rn_pc": float(retry_result.preconditioned_residual_norm),
            "history": retry_result.history,
            "solve_s": float(retry_result.solve_s),
        }
    )
    if retry_result.setup_s is not None:
        final_state["setup_s"] = float(retry_result.setup_s)
    if context.post_minres is not None:
        post_context = context.post_minres
        post_minres = apply_sparse_pc_post_minres_if_needed(
            SparsePCPostMinresUpdateContext(
                matvec=post_context.matvec,
                rhs=post_context.rhs,
                preconditioner=post_context.preconditioner,
                emit=post_context.emit,
                elapsed_s=post_context.elapsed_s,
                pc_form=str(post_context.pc_form),
                steps=int(post_context.steps),
                alpha_clip=float(post_context.alpha_clip),
                min_improvement=float(post_context.min_improvement),
                minres_correction=minres_correction,
                x=np.asarray(retry_result.x, dtype=np.float64),
                residual_norm=float(retry_result.residual_norm),
                preconditioned_residual_norm=float(
                    retry_result.preconditioned_residual_norm
                ),
                solve_s=float(retry_result.solve_s),
                target=float(post_context.target),
            )
        )
        final_state.update(
            {
                "x_np": post_minres.x,
                "residual_norm_sparse_pc": float(post_minres.residual_norm),
                "rn_pc": float(post_minres.preconditioned_residual_norm),
                "sparse_pc_post_minres_steps": int(post_context.steps),
                "sparse_pc_post_minres_alpha_clip": float(post_context.alpha_clip),
                "sparse_pc_post_minres_min_improvement": float(
                    post_context.min_improvement
                ),
                "sparse_pc_post_minres_history": post_minres.history,
                "sparse_pc_post_minres_alphas": post_minres.alphas,
                "sparse_pc_post_minres_residual_before": (
                    post_minres.residual_before
                ),
                "sparse_pc_post_minres_residual_after": post_minres.residual_after,
                "sparse_pc_post_minres_error": post_minres.error,
                "solve_s": float(post_minres.solve_s),
                "sparse_pc_elapsed_s": float(post_context.elapsed_s()),
            }
        )
        if post_context.emit is not None:
            post_context.emit(
                0,
                sparse_pc_gmres_completion_message(
                    SparsePCGMRESCompletionMessageContext(
                        elapsed_s=float(final_state["sparse_pc_elapsed_s"]),
                        iterations=int(len(final_state["history"] or ())),
                        matvecs=int(final_state["mv_count"]),
                        residual_norm=float(final_state["residual_norm_sparse_pc"]),
                        target=float(final_state["target"]),
                        preconditioned_residual_norm=float(final_state["rn_pc"]),
                        history=final_state["history"],
                    )
                ),
            )
        return sparse_pc_gmres_final_payload_from_solve_state(
            final_state,
            expand_reduced=expand_reduced,
        )
    return finalize_sparse_pc_gmres_from_solve_state(
        final_state,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
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
                    f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres "
                    f"improved residual {residual_before:.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(accepted_steps={len(alphas)})",
                )
        elif context.emit is not None:
            after = float(residual_after) if residual_after is not None else float("nan")
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres "
                f"rejected residual {residual_before:.6e} -> {after:.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres failed "
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

def apply_sparse_pc_post_minres_if_needed(
    context: SparsePCPostMinresUpdateContext,
) -> SparsePCPostMinresUpdateResult:
    """Apply sparse-PC post-minres only when requested and still above target."""

    if (
        int(context.steps) <= 0
        or not np.isfinite(float(context.residual_norm))
        or float(context.residual_norm) <= float(context.target)
    ):
        return SparsePCPostMinresUpdateResult(
            x=np.asarray(context.x, dtype=np.float64),
            residual_norm=float(context.residual_norm),
            preconditioned_residual_norm=float(context.preconditioned_residual_norm),
            history=(),
            alphas=(),
            residual_before=None,
            residual_after=None,
            error=None,
            solve_s=float(context.solve_s),
        )

    post_minres = apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=context.matvec,
            rhs=context.rhs,
            preconditioner=context.preconditioner,
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            pc_form=context.pc_form,
            steps=int(context.steps),
            alpha_clip=float(context.alpha_clip),
            min_improvement=float(context.min_improvement),
            minres_correction=context.minres_correction,
            solver_label=str(context.solver_label),
        ),
        x=np.asarray(context.x, dtype=np.float64),
        residual_norm=float(context.residual_norm),
        preconditioned_residual_norm=float(context.preconditioned_residual_norm),
    )
    return SparsePCPostMinresUpdateResult(
        x=post_minres.x,
        residual_norm=float(post_minres.residual_norm),
        preconditioned_residual_norm=float(post_minres.preconditioned_residual_norm),
        history=post_minres.history,
        alphas=post_minres.alphas,
        residual_before=post_minres.residual_before,
        residual_after=post_minres.residual_after,
        error=post_minres.error,
        solve_s=float(context.solve_s) + float(post_minres.solve_s),
    )

def apply_sparse_pc_post_minres_from_solve_state(
    state: Mapping[str, object],
    *,
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
) -> SparsePCPostMinresUpdateResult:
    """Apply sparse-PC post-minres using stored sparse-PC solve state."""

    return apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=state["_mv_true"],
            rhs=state["sparse_pc_rhs"],
            preconditioner=state["_precond_sparse"],
            emit=state["emit"],
            elapsed_s=state["sparse_timer"].elapsed_s,
            pc_form=str(state["pc_form"]),
            steps=int(state["sparse_pc_post_minres_steps"]),
            alpha_clip=float(state["sparse_pc_post_minres_alpha_clip"]),
            min_improvement=float(state["sparse_pc_post_minres_min_improvement"]),
            minres_correction=minres_correction,
            x=np.asarray(state["x_np"], dtype=np.float64),
            residual_norm=float(state["residual_norm_sparse_pc"]),
            preconditioned_residual_norm=float(state["rn_pc"]),
            solve_s=float(state["solve_s"]),
            target=float(state["target"]),
        )
    )


__all__ = (
    "SparsePCGMRESResult",
    "SparsePCGMRESContext",
    "SparsePCGMRESFinalPayload",
    "SparsePCPostMinresFinalizationContext",
    "SparsePCFactorDtypeRetryFinalizationContext",
    "SparsePCGMRESFinalizationContext",
    "SparsePCGMRESFinalResultContext",
    "SparsePCGMRESFinalizationBundleContext",
    "SparsePCGMRESFinalizationStateContext",
    "SparsePCGMRESCompletionMessageContext",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "SparsePCPostMinresUpdateContext",
    "SparsePCPostMinresUpdateResult",
    "run_sparse_pc_gmres_once",
    "run_sparse_pc_gmres_once_for_retry",
    "sparse_pc_gmres_finalization_solve_state_keys",
    "sparse_pc_gmres_finalization_solve_scope_keys",
    "sparse_pc_gmres_finalization_state_from_context",
    "sparse_pc_gmres_finalization_state_from_solve_scope",
    "sparse_pc_gmres_finalization_bundle_from_solve_scope",
    "sparse_pc_gmres_finalization_bundle_from_solve_result",
    "SparsePCFactorDtypeRetryDecision",
    "SparsePCFactorDtypeRetryContext",
    "SparsePCFactorDtypeRetryResult",
    "evaluate_sparse_pc_factor_dtype_retry",
    "sparse_pc_factor_dtype_retry_initial_guess",
    "retry_sparse_pc_factor_dtype_if_needed",
    "retry_sparse_pc_factor_dtype_from_solve_state",
    "retry_sparse_pc_factor_dtype_from_finalization_context",
    "sparse_pc_gmres_completion_message",
    "emit_sparse_pc_gmres_completion_from_solve_state",
    "sparse_pc_gmres_final_payload_from_solve_state",
    "finalize_sparse_pc_gmres_from_solve_state",
    "finalize_sparse_pc_gmres_with_dtype_retry_from_solve_state",
    "finalize_sparse_pc_gmres_bundle",
    "finalize_sparse_pc_gmres_with_dtype_retry",
    "apply_sparse_pc_post_minres",
    "apply_sparse_pc_post_minres_if_needed",
    "apply_sparse_pc_post_minres_from_solve_state",
)
