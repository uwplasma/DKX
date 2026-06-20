"""Linear-solver dispatch for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import numpy as np
import jax.scipy.linalg as jla
import jax.numpy as jnp

from sfincs_jax.implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from sfincs_jax.krylov_dispatch import gmres_solve_dispatch, rhs_krylov_method_for_context
from sfincs_jax.solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_history_scipy,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy,
)
from sfincs_jax.v3_system import sharding_constraints

from .residual import result_with_true_residual


@dataclass(frozen=True)
class ProfileLinearSolveContext:
    """Routing state shared by RHSMode=1 linear-solve attempts."""

    rhs_mode: int
    total_size: int
    use_implicit: bool
    use_solver_jit: bool
    distributed_axis: str | None
    distributed_auto_solver: str
    small_gmres_max: int


@dataclass(frozen=True)
class RHS1ScipyRescueContext:
    """Host-only SciPy rescue solve inputs for stalled RHSMode=1 systems."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    method: str
    tol: float
    atol: float
    restart: int
    maxiter: int
    precond_side: str


@dataclass(frozen=True)
class RHS1ScipyRescueOutcome:
    """Result payload and measured diagnostics from a SciPy rescue attempt."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray
    reported_residual: float
    history_len: int
    preconditioned_residual: float | None = None


@dataclass(frozen=True)
class RHS1DenseKSPFullSolveContext:
    """Inputs for the full-system RHSMode=1 dense-KSP solve path."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    total_size: int
    phi1_size: int
    n_species: int
    n_theta: int
    n_zeta: int
    nxi_for_x: object
    extra_size: int
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_linear: Callable[..., GMRESSolveResult]


@dataclass(frozen=True)
class RHS1DenseKSPReducedSolveContext:
    """Inputs for the reduced active-DOF RHSMode=1 dense-KSP solve path."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    active_size: int
    phi1_size: int
    n_species: int
    n_theta: int
    n_zeta: int
    nxi_for_x: object
    extra_size: int
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_linear: Callable[..., GMRESSolveResult]
    result_ready: Callable[[GMRESSolveResult], GMRESSolveResult]


@dataclass(frozen=True)
class RHS1DenseKSPFullSolveOutcome:
    """Physical result plus the preconditioned replay system."""

    result: GMRESSolveResult
    replay_matvec: Callable[[jnp.ndarray], jnp.ndarray]
    replay_rhs: jnp.ndarray


@dataclass(frozen=True)
class RHS1DenseKSPReducedSolveOutcome:
    """Reduced dense-KSP result plus the preconditioned replay system."""

    result: GMRESSolveResult
    replay_matvec: Callable[[jnp.ndarray], jnp.ndarray]
    replay_rhs: jnp.ndarray


def rhs1_small_gmres_max_from_env(*, default: int = 600) -> int:
    """Return the size cutoff for small-system GMRES auto routing."""

    env = os.environ.get("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def profile_solver_kind(method: str, *, context: ProfileLinearSolveContext) -> tuple[str, str]:
    """Map RHSMode=1 solve-method tokens to the concrete Krylov family."""

    method_l = str(method).strip().lower()
    if method_l in {"auto", "default"}:
        if (
            context.distributed_axis is not None
            and int(context.rhs_mode) == 1
            and context.distributed_auto_solver == "bicgstab"
        ):
            return "bicgstab", "batched"
        if int(context.rhs_mode) in {2, 3}:
            return "gmres", "incremental"
        if int(context.small_gmres_max) > 0 and int(context.total_size) <= int(context.small_gmres_max):
            return "gmres", "incremental"
        return "gmres", "incremental"
    if method_l in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab", "batched"
    return "gmres", method_l


def solve_profile_linear(
    *,
    context: ProfileLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    precond_side: str,
) -> GMRESSolveResult:
    """Solve an RHSMode=1 linear system without returning an explicit residual."""

    solver_kind, gmres_method = profile_solver_kind(solve_method_val, context=context)
    if context.use_implicit:
        return linear_custom_solve(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precond_side,
            size_hint=int(b_vec.shape[0]),
        )
    solve_method_dispatch = "bicgstab" if solver_kind == "bicgstab" else rhs_krylov_method_for_context(
        gmres_method=gmres_method,
        use_implicit=bool(context.use_implicit),
        distributed_axis=context.distributed_axis,
        solver_jit=bool(context.use_solver_jit),
    )
    return gmres_solve_dispatch(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=solve_method_dispatch,
        distributed_axis=context.distributed_axis,
        precondition_side=precond_side,
        use_solver_jit_fn=lambda _size_hint: bool(context.use_solver_jit),
    )


def solve_profile_linear_with_residual(
    *,
    context: ProfileLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    precond_side: str,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve an RHSMode=1 linear system and return the explicit residual."""

    solver_kind, gmres_method = profile_solver_kind(solve_method_val, context=context)
    if context.use_implicit:
        return linear_custom_solve_with_residual(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precond_side,
            size_hint=int(b_vec.shape[0]),
        )
    if solver_kind == "bicgstab":
        if context.distributed_axis is not None:
            with sharding_constraints(True):
                return gmres_solve_with_residual_distributed(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=precond_fn,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    restart=restart_val,
                    maxiter=maxiter_val,
                    solve_method="bicgstab",
                    precondition_side=precond_side,
                    axis_name=context.distributed_axis,
                )
        solver_fn = bicgstab_solve_with_residual_jit if context.use_solver_jit else bicgstab_solve_with_residual
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
    gmres_method_dispatch = rhs_krylov_method_for_context(
        gmres_method=gmres_method,
        use_implicit=bool(context.use_implicit),
        distributed_axis=context.distributed_axis,
        solver_jit=bool(context.use_solver_jit),
    )
    if context.distributed_axis is not None:
        with sharding_constraints(True):
            return gmres_solve_with_residual_distributed(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method_dispatch,
                precondition_side=precond_side,
                axis_name=context.distributed_axis,
            )
    solver_fn = gmres_solve_with_residual_jit if context.use_solver_jit else gmres_solve_with_residual
    return solver_fn(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=gmres_method_dispatch,
        precondition_side=precond_side,
    )


def run_rhs1_scipy_rescue(
    *,
    context: RHS1ScipyRescueContext,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1ScipyRescueOutcome:
    """Run the host-only SciPy rescue and recompute its true residual.

    This is intentionally non-differentiable and should only be called by
    CLI/host production lanes. The driver owns the size, timeout, and residual
    admission policy; this helper only executes the selected SciPy Krylov
    method and returns a true-residual payload.
    """

    method = str(context.method).strip().lower()
    if method not in {"gmres", "bicgstab"}:
        method = "gmres"
    side = str(context.precond_side).strip().lower()
    if method == "bicgstab":
        x_np, reported_residual, history = bicgstab_solve_with_history_scipy(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=int(context.maxiter),
            precondition_side=context.precond_side,
        )
        preconditioned_residual = None
    elif context.preconditioner is not None and side == "left":
        x_np, reported_residual, preconditioned_residual, history = (
            explicit_left_preconditioned_gmres_scipy(
                matvec=context.matvec,
                b=context.rhs,
                preconditioner=context.preconditioner,
                x0=context.x0,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=int(context.maxiter),
            )
        )
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: SciPy rescue residuals "
                f"true={float(reported_residual):.3e} "
                f"preconditioned={float(preconditioned_residual):.3e}",
            )
    else:
        x_np, reported_residual, history = gmres_solve_with_history_scipy(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precond_side,
        )
        preconditioned_residual = None
    x_scipy = jnp.asarray(x_np, dtype=jnp.float64)
    result, residual_vec = result_with_true_residual(
        x=x_scipy,
        rhs=context.rhs,
        matvec=context.matvec,
    )
    return RHS1ScipyRescueOutcome(
        result=result,
        residual_vec=residual_vec,
        reported_residual=float(reported_residual),
        history_len=len(history or []),
        preconditioned_residual=(
            None
            if preconditioned_residual is None
            else float(preconditioned_residual)
        ),
    )


def solve_rhs1_dense_ksp_full(
    context: RHS1DenseKSPFullSolveContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1DenseKSPFullSolveOutcome:
    """Run the host dense-KSP branch for full RHSMode=1 systems.

    This path assembles the full operator, builds PETSc-like species blocks,
    solves the left-preconditioned dense system, and reports the physical
    residual. Replay-state mutation remains in the driver.
    """

    if int(context.phi1_size) != 0:
        raise NotImplementedError(
            "dense_ksp is only supported for includePhi1=false RHSMode=1 solves."
        )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: assembling dense full matrix for dense_ksp",
        )
    a_dense = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.total_size),
        dtype=context.rhs.dtype,
    )

    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: building PETSc-like species-block "
            "preconditioner (dense_ksp)",
        )

    n_species = int(context.n_species)
    n_theta = int(context.n_theta)
    n_zeta = int(context.n_zeta)
    local_per_species = int(np.sum(np.asarray(context.nxi_for_x, dtype=np.int64)))
    dke_size = int(local_per_species * n_theta * n_zeta)
    extra_size = int(context.extra_size)
    extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
    if extra_size and (extra_per_species * n_species != extra_size):
        extra_per_species = 0

    f_size = int(n_species * dke_size)
    expected_size = int(f_size + int(context.phi1_size) + extra_size)
    if int(context.total_size) != expected_size:
        raise RuntimeError(
            f"dense_ksp expects total_size={expected_size}, got {int(context.total_size)}"
        )

    lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    idx_blocks: list[jnp.ndarray] = []
    for species_index in range(n_species):
        f_idx = np.arange(
            species_index * dke_size,
            (species_index + 1) * dke_size,
            dtype=np.int32,
        )
        extra_idx = np.arange(
            f_size + species_index * extra_per_species,
            f_size + (species_index + 1) * extra_per_species,
            dtype=np.int32,
        )
        block_idx_np = (
            np.concatenate([f_idx, extra_idx], axis=0)
            if extra_per_species
            else f_idx
        )
        block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
        a_block = a_dense[jnp.ix_(block_idx, block_idx)]
        lu, piv = jla.lu_factor(a_block)
        lu_factors.append((lu, piv))
        idx_blocks.append(block_idx)

    def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(v)
        for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
            rhs_block = v[block_idx]
            sol_block = jla.lu_solve((lu, piv), rhs_block)
            out = out.at[block_idx].set(sol_block, unique_indices=True)
        return out

    def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
        return a_dense @ x

    rhs_pc = preconditioner_dense(context.rhs)

    def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
        return preconditioner_dense(mv_dense(x))

    res_pc = context.solve_linear(
        matvec_fn=mv_pc,
        b_vec=rhs_pc,
        precond_fn=None,
        x0_vec=context.x0,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        restart_val=int(context.restart),
        maxiter_val=context.maxiter,
        solve_method_val="incremental",
        precond_side="none",
    )
    residual_norm_full = jnp.linalg.norm(context.matvec(res_pc.x) - context.rhs)
    result = GMRESSolveResult(x=res_pc.x, residual_norm=residual_norm_full)
    return RHS1DenseKSPFullSolveOutcome(
        result=result,
        replay_matvec=mv_pc,
        replay_rhs=rhs_pc,
    )


def solve_rhs1_dense_ksp_reduced(
    context: RHS1DenseKSPReducedSolveContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1DenseKSPReducedSolveOutcome:
    """Run the reduced active-DOF dense-KSP branch.

    This mirrors the full-system dense-KSP path but preserves the reduced
    branch's historical residual semantics: the returned result is the
    left-preconditioned solve result after the driver's result-ready hook.
    """

    if int(context.phi1_size) != 0:
        raise NotImplementedError(
            "dense_ksp is only supported for includePhi1=false RHSMode=1 solves."
        )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: assembling dense reduced matrix for dense_ksp",
        )
    a_dense = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.active_size),
        dtype=context.rhs.dtype,
    )

    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: building PETSc-like species-block "
            "preconditioner (dense_ksp)",
        )

    n_species = int(context.n_species)
    n_theta = int(context.n_theta)
    n_zeta = int(context.n_zeta)
    local_per_species = int(np.sum(np.asarray(context.nxi_for_x, dtype=np.int64)))
    dke_size = int(local_per_species * n_theta * n_zeta)
    extra_size = int(context.extra_size)
    extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
    if extra_size and (extra_per_species * n_species != extra_size):
        extra_per_species = 0

    f_size = int(n_species * dke_size)
    expected_active = int(f_size + int(context.phi1_size) + extra_size)
    if int(context.active_size) != expected_active:
        raise RuntimeError(
            f"dense_ksp expects active_size={expected_active}, got {int(context.active_size)}"
        )

    lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    idx_blocks: list[jnp.ndarray] = []
    for species_index in range(n_species):
        f_idx = np.arange(
            species_index * dke_size,
            (species_index + 1) * dke_size,
            dtype=np.int32,
        )
        extra_idx = np.arange(
            f_size + species_index * extra_per_species,
            f_size + (species_index + 1) * extra_per_species,
            dtype=np.int32,
        )
        block_idx_np = (
            np.concatenate([f_idx, extra_idx], axis=0)
            if extra_per_species
            else f_idx
        )
        block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
        a_block = a_dense[jnp.ix_(block_idx, block_idx)]
        lu, piv = jla.lu_factor(a_block)
        lu_factors.append((lu, piv))
        idx_blocks.append(block_idx)

    def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(v)
        for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
            rhs_block = v[block_idx]
            sol_block = jla.lu_solve((lu, piv), rhs_block)
            out = out.at[block_idx].set(sol_block, unique_indices=True)
        return out

    def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
        return a_dense @ x

    rhs_pc = preconditioner_dense(context.rhs)

    def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
        return preconditioner_dense(mv_dense(x))

    result = context.solve_linear(
        matvec_fn=mv_pc,
        b_vec=rhs_pc,
        precond_fn=None,
        x0_vec=context.x0,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        restart_val=int(context.restart),
        maxiter_val=context.maxiter,
        solve_method_val="incremental",
        precond_side="none",
    )
    result = context.result_ready(result)
    return RHS1DenseKSPReducedSolveOutcome(
        result=result,
        replay_matvec=mv_pc,
        replay_rhs=rhs_pc,
    )


__all__ = [
    "ProfileLinearSolveContext",
    "RHS1DenseKSPFullSolveContext",
    "RHS1DenseKSPFullSolveOutcome",
    "RHS1DenseKSPReducedSolveContext",
    "RHS1DenseKSPReducedSolveOutcome",
    "RHS1ScipyRescueContext",
    "RHS1ScipyRescueOutcome",
    "profile_solver_kind",
    "rhs1_small_gmres_max_from_env",
    "run_rhs1_scipy_rescue",
    "solve_rhs1_dense_ksp_full",
    "solve_rhs1_dense_ksp_reduced",
    "solve_profile_linear",
    "solve_profile_linear_with_residual",
]
