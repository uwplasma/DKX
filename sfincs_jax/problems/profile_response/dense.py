"""Host dense reduced-system helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    dense_krylov_solve_from_matrix_with_residual,
    dense_solve_from_matrix,
    dense_solve_from_matrix_row_scaled,
)
from .residual import result_with_true_residual


@dataclass(frozen=True)
class HostDenseReducedSolveContext:
    """Solve-local inputs for a host dense reduced RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    active_size: int
    constraint_scheme: int
    has_fp: bool
    dense_matrix_cache: np.ndarray | None = None


@dataclass(frozen=True)
class HostDenseFullSolveContext:
    """Solve-local inputs for a host dense full-system RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    total_size: int


@dataclass(frozen=True)
class RHS1ReducedDenseFallbackCandidateContext:
    """Inputs for the reduced RHSMode=1 dense fallback candidate.

    This is the richer post-primary fallback used by the v3 driver after a
    matrix-free reduced solve stalls. It intentionally supports both
    host/non-autodiff LU and JAX-visible dense Krylov lanes so the CLI can use a
    fast host path while implicit-differentiation callers still have a JAX
    custom-linear-solve contract.
    """

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray
    active_size: int
    constraint_scheme: int
    has_fp: bool
    has_pas: bool
    dense_matrix_cache: np.ndarray | jnp.ndarray | None
    dense_backend_allowed: bool
    use_implicit: bool
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    gmres_precond_side: str
    backend: str | None = None


@dataclass(frozen=True)
class RHS1DenseProbeAdmission:
    """Whether the reduced-system dense probe should run."""

    enabled: bool


@dataclass(frozen=True)
class RHS1DenseProbeShortcutDecision:
    """Dense-probe shortcut decision after the probe residual is known."""

    accept_shortcut: bool
    seed_x0_if_missing: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1DenseShortcutSetup:
    """Dense shortcut/fallback controls after env and backend gates."""

    dense_shortcut_ratio: float
    dense_fallback_max: int
    disable_dense_pas: bool
    messages: tuple[tuple[int, str], ...] = ()


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def rhs1_dense_shortcut_setup_from_env(
    *,
    has_pas: bool,
    include_phi1: bool,
    constraint_scheme: int,
    active_size: int,
    dense_fallback_max: int,
    dense_backend_allowed: bool,
    host_dense_fallback_allowed: bool,
    dense_krylov_allowed: bool,
    backend: str,
) -> RHS1DenseShortcutSetup:
    """Resolve dense shortcut/fallback controls with legacy PAS/backend guards."""

    dense_shortcut_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO",
        1.0e6,
    )
    disable_dense_pas = (
        bool(has_pas) and (not bool(include_phi1)) and int(constraint_scheme) != 0
    )
    pas_dense_allow_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_DENSE_ALLOW_MAX", 4000)
    if disable_dense_pas and int(active_size) <= max(0, int(pas_dense_allow_max)):
        disable_dense_pas = False
    if disable_dense_pas or bool(has_pas):
        dense_shortcut_ratio = 0.0

    dense_fallback_max_use = int(dense_fallback_max)
    if disable_dense_pas:
        dense_fallback_max_use = 0

    messages: list[tuple[int, str]] = []
    if not bool(dense_backend_allowed):
        dense_shortcut_ratio = 0.0
        if not bool(host_dense_fallback_allowed) and not bool(dense_krylov_allowed):
            dense_fallback_max_use = 0
        dense_note = "dense shortcut/fallback"
        if bool(host_dense_fallback_allowed):
            dense_note = "dense shortcut (host dense fallback kept)"
        elif bool(dense_krylov_allowed):
            dense_note = "dense shortcut disabled (dense Krylov fallback kept)"
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: disabling RHSMode=1 "
                f"{dense_note} on backend={backend}",
            )
        )

    return RHS1DenseShortcutSetup(
        dense_shortcut_ratio=float(dense_shortcut_ratio),
        dense_fallback_max=int(dense_fallback_max_use),
        disable_dense_pas=bool(disable_dense_pas),
        messages=tuple(messages),
    )


def rhs1_dense_probe_enabled_from_env() -> bool:
    """Return whether the reduced dense probe is globally enabled."""

    probe_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PROBE", "").strip().lower()
    return probe_env not in {"0", "false", "no", "off"}


def rhs1_dense_probe_admission(
    *,
    probe_enabled: bool,
    probe_shortcut: bool,
    cs0_petsc_compat: bool,
    cs0_sparse_first: bool,
    cs0_dense_fallback_allowed: bool,
    constraint_scheme: int,
    has_preconditioner: bool,
    solve_method_kind: str,
) -> RHS1DenseProbeAdmission:
    """Apply cheap guards before evaluating a reduced dense fallback probe."""

    enabled = (
        bool(probe_enabled)
        and (not bool(probe_shortcut))
        and (not bool(cs0_petsc_compat))
        and (not bool(cs0_sparse_first))
        and (bool(cs0_dense_fallback_allowed) or int(constraint_scheme) != 0)
        and bool(has_preconditioner)
        and str(solve_method_kind) not in {"dense", "dense_ksp"}
    )
    return RHS1DenseProbeAdmission(enabled=bool(enabled))


def rhs1_dense_probe_shortcut_decision(
    *,
    dense_shortcut_ratio: float,
    probe_ratio: float,
    dense_fallback_max: int,
    active_size: int,
    sparse_prefer_over_dense_shortcut: bool,
) -> RHS1DenseProbeShortcutDecision:
    """Resolve whether a dense probe should become an early dense shortcut."""

    if float(dense_shortcut_ratio) <= 0.0 or float(probe_ratio) < float(
        dense_shortcut_ratio
    ):
        return RHS1DenseProbeShortcutDecision(
            accept_shortcut=False,
            seed_x0_if_missing=True,
        )

    allow_probe_shortcut = int(dense_fallback_max) > 0 and int(active_size) <= int(
        dense_fallback_max
    )
    if allow_probe_shortcut and (not bool(sparse_prefer_over_dense_shortcut)):
        return RHS1DenseProbeShortcutDecision(
            accept_shortcut=True,
            seed_x0_if_missing=False,
            messages=(
                (
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback shortcut (probe) "
                    f"(ratio={float(probe_ratio):.3e} >= {float(dense_shortcut_ratio):.1e})",
                ),
            ),
        )

    if bool(sparse_prefer_over_dense_shortcut) and allow_probe_shortcut:
        message = (
            "solve_v3_full_system_linear_gmres: probe shortcut skipped "
            "(preferring sparse rescue over dense shortcut)"
        )
    else:
        message = (
            "solve_v3_full_system_linear_gmres: probe shortcut skipped "
            f"(size={int(active_size)} > dense_max={int(dense_fallback_max)})"
        )
    return RHS1DenseProbeShortcutDecision(
        accept_shortcut=False,
        seed_x0_if_missing=True,
        messages=((1, message),),
    )


def solve_host_dense_reduced(
    *,
    context: HostDenseReducedSolveContext,
    x0: jnp.ndarray | None = None,
) -> GMRESSolveResult:
    """Solve the reduced system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    use_row_scaled = bool(int(context.constraint_scheme) == 0 or (int(context.constraint_scheme) == 1 and context.has_fp))
    if context.dense_matrix_cache is not None:
        a_np = np.asarray(context.dense_matrix_cache, dtype=np.float64)
    else:
        a_dense_jnp = assemble_dense_matrix_from_matvec(
            matvec=context.matvec,
            n=int(context.active_size),
            dtype=context.rhs.dtype,
        )
        a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)

    matvec_residual = context.matvec
    b_dense = jnp.asarray(context.rhs, dtype=jnp.float64)
    if use_row_scaled:
        diag_floor = 1e-12
        diag = np.diag(a_np).astype(np.float64, copy=False)
        diag_abs = np.abs(diag)
        diag_safe = np.where(diag_abs > diag_floor, diag, np.sign(diag) * diag_floor)
        diag_safe = np.where(diag_safe != 0.0, diag_safe, diag_floor)
        scale = (1.0 / diag_safe).astype(np.float64, copy=False)
        a_np = a_np * scale[:, None]
        scale_jnp = jnp.asarray(scale, dtype=jnp.float64)
        b_dense = b_dense * scale_jnp

        def matvec_residual(x_vec: jnp.ndarray) -> jnp.ndarray:
            return scale_jnp * context.matvec(x_vec)

    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(b_dense, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(b_dense, dtype=np.float64)), dtype=np.float64)
        if x0 is not None and x0.shape == context.rhs.shape:
            x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)

    r_dense = b_dense - matvec_residual(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))


def solve_host_dense_full(
    *,
    context: HostDenseFullSolveContext,
    x0: jnp.ndarray | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve the full system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    a_dense_jnp = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.total_size),
        dtype=context.rhs.dtype,
    )
    a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)
    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(context.rhs, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(context.rhs, dtype=np.float64)), dtype=np.float64)
    if x0 is not None and x0.shape == context.rhs.shape:
        x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
    x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    residual_vec = context.rhs - context.matvec(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_vec)), residual_vec


def solve_rhs1_reduced_dense_fallback_candidate(
    *,
    context: RHS1ReducedDenseFallbackCandidateContext,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[GMRESSolveResult, float]:
    """Run one dense fallback candidate for a reduced RHSMode=1 system.

    The caller remains responsible for residual/runtime/memory admission. This
    keeps the policy gate in the driver while moving the dense solve mechanics
    out of the monolithic solve function.
    """

    started = time.perf_counter()
    use_row_scaled = bool(
        int(context.constraint_scheme) == 0
        or (int(context.constraint_scheme) == 1 and bool(context.has_fp))
    )
    host_dense_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", ""
    ).strip().lower()
    backend = context.backend or jax.default_backend()
    if host_dense_env in {"0", "false", "no", "off"}:
        use_host_dense = False
    elif host_dense_env in {"1", "true", "yes", "on"}:
        use_host_dense = True
    else:
        # Default: avoid backend LAPACK/SVD paths on accelerators, and avoid
        # XLA dense-solve scratch allocations for medium/large CPU systems.
        use_host_dense = backend != "cpu" or (
            bool(context.use_implicit) and int(context.active_size) >= 2000
        )
    if bool(context.has_pas) and int(context.active_size) <= 2000:
        use_host_dense = True

    if use_host_dense:
        result = _solve_rhs1_reduced_dense_fallback_host_candidate(
            context=context,
            backend=backend,
            host_dense_env=host_dense_env,
            use_row_scaled=use_row_scaled,
            emit=emit,
        )
    elif context.dense_backend_allowed and context.dense_matrix_cache is not None:
        a_dense_jnp = jnp.asarray(context.dense_matrix_cache, dtype=context.rhs.dtype)
        if use_row_scaled:
            x_dense, _rn = dense_solve_from_matrix_row_scaled(
                a=a_dense_jnp,
                b=context.rhs,
            )
        else:
            x_dense, _rn = dense_solve_from_matrix(a=a_dense_jnp, b=context.rhs)
        result, _residual = result_with_true_residual(
            x=x_dense,
            rhs=context.rhs,
            matvec=context.matvec,
        )
    else:
        if context.dense_matrix_cache is not None:
            a_dense_jnp = jnp.asarray(context.dense_matrix_cache, dtype=context.rhs.dtype)
        else:
            a_dense_jnp = assemble_dense_matrix_from_matvec(
                matvec=context.matvec,
                n=int(context.active_size),
                dtype=context.rhs.dtype,
            )
        if emit is not None and jax.default_backend() != "cpu":
            emit(
                0,
                "solve_v3_full_system_linear_gmres: dense fallback using explicit dense Krylov "
                f"on backend={jax.default_backend()}",
            )
        result, _residual = dense_krylov_solve_from_matrix_with_residual(
            a=a_dense_jnp,
            b=context.rhs,
            x0=context.x0,
            preconditioner=None,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            solve_method="incremental",
            precondition_side=(
                "none" if use_row_scaled else str(context.gmres_precond_side)
            ),
            row_scaled=use_row_scaled,
        )

    return result, time.perf_counter() - started


def _solve_rhs1_reduced_dense_fallback_host_candidate(
    *,
    context: RHS1ReducedDenseFallbackCandidateContext,
    backend: str,
    host_dense_env: str,
    use_row_scaled: bool,
    emit: Callable[[int, str], None] | None,
) -> GMRESSolveResult:
    """Host LU/least-squares branch for the reduced dense fallback."""

    import scipy.linalg as sla  # noqa: PLC0415

    if emit is not None and backend != "cpu" and host_dense_env in {"", "auto"}:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: dense fallback using host LU "
            f"on backend={backend}",
        )

    if context.dense_matrix_cache is not None:
        a_np = np.asarray(context.dense_matrix_cache, dtype=np.float64)
    else:
        a_dense_jnp = assemble_dense_matrix_from_matvec(
            matvec=context.matvec,
            n=int(context.active_size),
            dtype=context.rhs.dtype,
        )
        a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)

    mv_dense = context.matvec
    b_dense = jnp.asarray(context.rhs, dtype=jnp.float64)
    if use_row_scaled:
        diag_floor = 1e-12
        diag = np.diag(a_np).astype(np.float64, copy=False)
        diag_abs = np.abs(diag)
        diag_safe = np.where(
            diag_abs > diag_floor,
            diag,
            np.sign(diag) * diag_floor,
        )
        diag_safe = np.where(diag_safe != 0.0, diag_safe, diag_floor)
        scale = (1.0 / diag_safe).astype(np.float64, copy=False)
        a_np = a_np * scale[:, None]
        scale_jnp = jnp.asarray(scale, dtype=jnp.float64)
        b_dense = b_dense * scale_jnp

        def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
            return scale_jnp * context.matvec(x)

    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: dense fallback "
                f"non-square matrix shape={a_np.shape}; using least-squares host solve",
            )
        if not context.use_implicit:
            x_np = np.asarray(
                np.linalg.lstsq(
                    a_np,
                    np.asarray(b_dense, dtype=np.float64),
                    rcond=None,
                )[0],
                dtype=np.float64,
            )
            x_dense = jnp.asarray(x_np, dtype=jnp.float64)
        else:

            def _solve_cb(rhs_np: np.ndarray) -> np.ndarray:
                rhs_np = np.asarray(rhs_np, dtype=np.float64)
                return np.asarray(
                    np.linalg.lstsq(a_np, rhs_np, rcond=None)[0],
                    dtype=np.float64,
                )

            out_spec = jax.ShapeDtypeStruct(b_dense.shape, jnp.float64)
            x_dense = jax.pure_callback(_solve_cb, out_spec, b_dense)
        result, _residual = result_with_true_residual(
            x=x_dense,
            rhs=context.rhs,
            matvec=context.matvec,
        )
        return result

    lu, piv = sla.lu_factor(a_np)
    refine_steps = 0
    if bool(context.has_pas) and int(context.active_size) <= 2000:
        refine_steps = 2
    if not context.use_implicit:
        rhs_np = np.asarray(b_dense, dtype=np.float64)
        x_np = np.asarray(sla.lu_solve((lu, piv), rhs_np), dtype=np.float64)
        for _ in range(int(refine_steps)):
            r_np = rhs_np - a_np @ x_np
            dx_np = np.asarray(sla.lu_solve((lu, piv), r_np), dtype=np.float64)
            x_np = x_np + dx_np
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    else:
        out_spec = jax.ShapeDtypeStruct(b_dense.shape, jnp.float64)

        def _solve_cb(rhs_np: np.ndarray) -> np.ndarray:
            rhs_np = np.asarray(rhs_np, dtype=np.float64)
            x_np = np.asarray(sla.lu_solve((lu, piv), rhs_np), dtype=np.float64)
            for _ in range(int(refine_steps)):
                r_np = rhs_np - a_np @ x_np
                dx_np = np.asarray(sla.lu_solve((lu, piv), r_np), dtype=np.float64)
                x_np = x_np + dx_np
            return x_np

        def _solveT_cb(rhs_np: np.ndarray) -> np.ndarray:
            rhs_np = np.asarray(rhs_np, dtype=np.float64)
            x_np = np.asarray(
                sla.lu_solve((lu, piv), rhs_np, trans=1),
                dtype=np.float64,
            )
            for _ in range(int(refine_steps)):
                r_np = rhs_np - a_np.T @ x_np
                dx_np = np.asarray(
                    sla.lu_solve((lu, piv), r_np, trans=1),
                    dtype=np.float64,
                )
                x_np = x_np + dx_np
            return x_np

        def _solve_host(_mv, rhs: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(_solve_cb, out_spec, rhs)

        def _transpose_solve_host(_mv_t, rhs: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(_solveT_cb, out_spec, rhs)

        x_dense = jax.lax.custom_linear_solve(
            mv_dense,
            b_dense,
            solve=_solve_host,
            transpose_solve=_transpose_solve_host,
            symmetric=False,
        )
    result, _residual = result_with_true_residual(
        x=x_dense,
        rhs=context.rhs,
        matvec=context.matvec,
    )
    return result


__all__ = [
    "RHS1DenseProbeAdmission",
    "RHS1DenseProbeShortcutDecision",
    "RHS1DenseShortcutSetup",
    "HostDenseFullSolveContext",
    "HostDenseReducedSolveContext",
    "RHS1ReducedDenseFallbackCandidateContext",
    "rhs1_dense_probe_admission",
    "rhs1_dense_probe_enabled_from_env",
    "rhs1_dense_probe_shortcut_decision",
    "rhs1_dense_shortcut_setup_from_env",
    "solve_rhs1_reduced_dense_fallback_candidate",
    "solve_host_dense_full",
    "solve_host_dense_reduced",
]
