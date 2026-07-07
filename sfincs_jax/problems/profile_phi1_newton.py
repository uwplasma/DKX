"""Nonlinear Phi1 Newton-Krylov solves for RHSMode=1 profile response.

This module owns the nonlinear Phi1 solve loop for the profile-response
problem. It keeps the production/non-autodiff sparse-direct rescue explicit
while preserving the JAX-native Newton and linearization path used by
differentiable Python workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..solvers.explicit_sparse import (
    host_sparse_direct_polish,
    host_sparse_direct_solve_with_refinement,
)
from ..solvers.krylov_dispatch import gmres_solve_dispatch
from ..namelist import Namelist
from ..solvers.preconditioning import (
    set_precond_policy_hints,
    set_precond_size_hint,
    use_solver_jit,
)
from sfincs_jax.operators.profile_layout import build_rhs1_compressed_pitch_layout
from sfincs_jax.problems.profile_policies import (
    host_sparse_direct_refine_steps,
    host_sparse_factor_dtype,
)
from sfincs_jax.problems.profile_solver_diagnostics import (
    V3NewtonKrylovResult,
    emit_newton_krylov_ksp_history,
    rhs1_fortran_stdout_from_env,
    rhs1_ksp_history_limits_from_env,
)
from ..solvers.krylov import (
    GMRESSolveResult,
    distributed_gmres_enabled,
    gmres_solve,
    gmres_solve_distributed,
    gmres_solve_jit,
)
from ..solvers.krylov import gmres_result_is_finite
from ..solvers.preconditioner_full_fp_kinetic import (
    build_rhs1_block_preconditioner,
    build_rhs1_collision_preconditioner,
)
from ..solvers.preconditioner_symbolic_host import build_sparse_ilu_from_matvec
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator,
    apply_v3_full_system_jacobian_jit,
    apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system_jit,
)


def _transport_active_dof_indices(op: V3FullSystemOperator) -> np.ndarray:
    return build_rhs1_compressed_pitch_layout(op).active_full_indices.astype(np.int32, copy=False)


def _dispatch_gmres(**kwargs):
    return gmres_solve_dispatch(
        gmres_solve_fn=gmres_solve,
        gmres_solve_jit_fn=gmres_solve_jit,
        gmres_solve_distributed_fn=gmres_solve_distributed,
        distributed_gmres_enabled_fn=distributed_gmres_enabled,
        use_solver_jit_fn=use_solver_jit,
        **kwargs,
    )


def phi1_use_active_dof_mode(
    *,
    rhs_mode: int,
    include_phi1: bool,
    has_reduced_modes: bool,
    env_value: str,
) -> bool:
    """Return whether Phi1 solves should compact to active DOFs."""
    env = str(env_value).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return bool(int(rhs_mode) == 1 and bool(include_phi1) and bool(has_reduced_modes))


def phi1_gmres_restart(active_size: int, gmres_restart: int) -> int:
    """Cap GMRES restart for small Phi1 active systems."""
    restart = int(gmres_restart)
    if int(active_size) <= 1000:
        restart = min(restart, 200)
    return max(1, restart)


@dataclass(frozen=True)
class Phi1FrozenJacobianPolicy:
    """Resolved frozen-Jacobian cache policy for the nonlinear Phi1 path."""

    mode: str
    use_cache: bool
    every: int


def phi1_frozen_jacobian_policy(
    *,
    include_phi1: bool,
) -> Phi1FrozenJacobianPolicy:
    """Resolve the frozen-Jacobian mode and cache cadence from env settings."""
    jac_mode = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "").strip().lower()
    if jac_mode not in {"frozen", "frozen_rhs", "frozen_op"}:
        jac_mode = "frozen" if bool(include_phi1) else "frozen_rhs"

    cache_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE", "").strip().lower()
    if cache_env in {"0", "false", "no", "off"}:
        use_cache = False
    elif cache_env in {"1", "true", "yes", "on"}:
        use_cache = True
    else:
        use_cache = True

    every_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY", "").strip()
    try:
        every = int(every_env) if every_env else 1
    except ValueError:
        every = 1
    return Phi1FrozenJacobianPolicy(mode=jac_mode, use_cache=use_cache, every=max(1, int(every)))


@dataclass(frozen=True)
class Phi1LineSearchPolicy:
    """Resolved nonlinear Phi1 line-search constants and mode."""

    step_scale: float
    factor: float | None
    c1: float
    mode: str
    maxiter: int


def phi1_line_search_policy(
    *,
    use_frozen_linearization: bool,
    include_phi1: bool,
) -> Phi1LineSearchPolicy:
    """Resolve the Phi1 line-search mode, constants, and iteration cap."""
    step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
    try:
        step_scale = float(step_scale_env) if step_scale_env else 1.0
    except ValueError:
        step_scale = 1.0

    ls_factor_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_FACTOR", "").strip()
    ls_c1_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_C1", "").strip()
    try:
        ls_factor = float(ls_factor_env) if ls_factor_env else None
    except ValueError:
        ls_factor = None
    try:
        ls_c1 = float(ls_c1_env) if ls_c1_env else 1.0e-4
    except ValueError:
        ls_c1 = 1.0e-4

    ls_mode_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MODE", "").strip().lower()
    if ls_mode_env:
        ls_mode = ls_mode_env
    else:
        ls_mode = "petsc" if (bool(use_frozen_linearization) and bool(include_phi1)) else "best"

    max_ls_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MAXITER", "").strip()
    try:
        max_ls = int(max_ls_env) if max_ls_env else (40 if ls_mode == "petsc" else 12)
    except ValueError:
        max_ls = 40 if ls_mode == "petsc" else 12

    return Phi1LineSearchPolicy(
        step_scale=float(step_scale),
        factor=ls_factor,
        c1=float(ls_c1),
        mode=str(ls_mode),
        maxiter=int(max_ls),
    )


def solve_phi1_newton_linear_step(
    *,
    use_active_dof_mode: bool,
    solve_method_linear: str,
    matvec,
    residual_vec: jnp.ndarray,
    preconditioner,
    gmres_tol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
    sparse_direct_solve: Callable[..., GMRESSolveResult],
    gmres_dispatch: Callable[..., GMRESSolveResult],
    gmres_result_is_finite: Callable[[GMRESSolveResult], bool],
    emit_ksp_history: Callable[..., None],
    emit: Callable[[int, str], None] | None,
    newton_iter: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_size: int | None = None,
    total_size: int | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray, jnp.ndarray]:
    """Solve one Newton linearization step for the Phi1 nonlinear path."""
    if use_active_dof_mode:
        assert reduce_full is not None
        assert expand_reduced is not None
        assert active_size is not None
        rhs_reduced = reduce_full(-residual_vec)

        def matvec_reduced(dx_reduced: jnp.ndarray) -> jnp.ndarray:
            return reduce_full(matvec(expand_reduced(dx_reduced)))

        if solve_method_linear == "sparse_direct":
            lin = sparse_direct_solve(
                matvec_fn=matvec_reduced,
                b_vec=rhs_reduced,
                n=int(active_size),
                cache_tag=("reduced", int(newton_iter), int(active_size)),
                tol_val=float(gmres_tol),
                atol_val=0.0,
                restart_val=int(gmres_restart),
                maxiter_val=gmres_maxiter,
            )
        else:
            lin = gmres_dispatch(
                matvec=matvec_reduced,
                b=rhs_reduced,
                preconditioner=preconditioner,
                tol=float(gmres_tol),
                restart=int(gmres_restart),
                maxiter=gmres_maxiter,
                solve_method=solve_method_linear,
            )
            emit_ksp_history(
                matvec_fn=matvec_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner,
                x0_vec=None,
                tol_val=float(gmres_tol),
                atol_val=0.0,
                restart_val=int(gmres_restart),
                maxiter_val=gmres_maxiter,
                precond_side="left",
            )
            if preconditioner is not None and (not gmres_result_is_finite(lin)):
                if emit is not None:
                    emit(
                        0,
                        "newton_iter="
                        f"{newton_iter}: preconditioned GMRES returned non-finite result; retrying without preconditioner",
                    )
                lin = gmres_dispatch(
                    matvec=matvec_reduced,
                    b=rhs_reduced,
                    preconditioner=None,
                    tol=float(gmres_tol),
                    restart=int(gmres_restart),
                    maxiter=gmres_maxiter,
                    solve_method=solve_method_linear,
                )
                emit_ksp_history(
                    matvec_fn=matvec_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=None,
                    x0_vec=None,
                    tol_val=float(gmres_tol),
                    atol_val=0.0,
                    restart_val=int(gmres_restart),
                    maxiter_val=gmres_maxiter,
                    precond_side="left",
                )
        step_vec = expand_reduced(lin.x)
        linear_residual_norm = jnp.linalg.norm(matvec(step_vec) + residual_vec)
        return lin, step_vec, linear_residual_norm

    assert total_size is not None
    if solve_method_linear == "sparse_direct":
        lin = sparse_direct_solve(
            matvec_fn=matvec,
            b_vec=-residual_vec,
            n=int(total_size),
            cache_tag=("full", int(newton_iter), int(total_size)),
            tol_val=float(gmres_tol),
            atol_val=0.0,
            restart_val=int(gmres_restart),
            maxiter_val=gmres_maxiter,
        )
    else:
        lin = gmres_dispatch(
            matvec=matvec,
            b=-residual_vec,
            preconditioner=preconditioner,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=solve_method_linear,
        )
        emit_ksp_history(
            matvec_fn=matvec,
            b_vec=-residual_vec,
            precond_fn=preconditioner,
            x0_vec=None,
            tol_val=float(gmres_tol),
            atol_val=0.0,
            restart_val=int(gmres_restart),
            maxiter_val=gmres_maxiter,
            precond_side="left",
        )
        if preconditioner is not None and (not gmres_result_is_finite(lin)):
            if emit is not None:
                emit(
                    0,
                    "newton_iter="
                    f"{newton_iter}: preconditioned GMRES returned non-finite result; retrying without preconditioner",
                )
            lin = gmres_dispatch(
                matvec=matvec,
                b=-residual_vec,
                preconditioner=None,
                tol=float(gmres_tol),
                restart=int(gmres_restart),
                maxiter=gmres_maxiter,
                solve_method=solve_method_linear,
            )
            emit_ksp_history(
                matvec_fn=matvec,
                b_vec=-residual_vec,
                precond_fn=None,
                x0_vec=None,
                tol_val=float(gmres_tol),
                atol_val=0.0,
                restart_val=int(gmres_restart),
                maxiter_val=gmres_maxiter,
                precond_side="left",
            )
    return lin, lin.x, lin.residual_norm


def build_phi1_newton_preconditioner(
    *,
    use_preconditioner: bool,
    use_frozen_linearization: bool,
    rhs_mode: int,
    include_phi1: bool,
    use_active_dof_mode: bool,
    op,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None,
    preconditioner_options: dict[str, object],
    collision_builder: Callable[..., object],
    block_builder: Callable[..., object],
    emit: Callable[[int, str], None] | None = None,
) -> object | None:
    """Build the bounded Newton preconditioner used by Phi1 solves."""

    def _opt_int(key: str, default: int) -> int:
        val = preconditioner_options.get(key, None)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    if not (use_preconditioner and use_frozen_linearization and int(rhs_mode) == 1):
        return None

    precond_kind_env = os.environ.get("SFINCS_JAX_PHI1_PRECOND_KIND", "").strip().lower()
    if not precond_kind_env:
        precond_kind = "collision" if bool(include_phi1) else "block"
    elif precond_kind_env in {"collision", "diag"}:
        precond_kind = "collision"
    elif precond_kind_env in {"block", "block_jacobi", "point"}:
        precond_kind = "block"
    else:
        precond_kind = "block"

    if emit is not None:
        emit(1, f"solve_v3_full_system_newton_krylov_history: preconditioner={precond_kind}")

    if precond_kind == "collision":
        if use_active_dof_mode:
            return collision_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        return collision_builder(op=op)

    kwargs = {
        "op": op,
        "preconditioner_species": _opt_int("PRECONDITIONER_SPECIES", 1),
        "preconditioner_x": _opt_int("PRECONDITIONER_X", 1),
        "preconditioner_xi": _opt_int("PRECONDITIONER_XI", 1),
    }
    if use_active_dof_mode:
        kwargs["reduce_full"] = reduce_full
        kwargs["expand_reduced"] = expand_reduced
    return block_builder(**kwargs)


def _phi1_host_sparse_factor_dtype(*, size: int) -> np.dtype:
    return host_sparse_factor_dtype(
        size=int(size),
        factorization="lu",
        use_implicit=False,
        backend=jax.default_backend(),
    )


def advance_phi1_newton_iterate(
    *,
    x: jnp.ndarray,
    step_direction: jnp.ndarray,
    residual_norm0: float,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    accepted: Sequence[jnp.ndarray],
    mode: str,
    step_scale: float,
    factor: float | None,
    c1: float,
    maxiter: int,
) -> jnp.ndarray:
    """Apply the bounded Newton line-search/update policy for Phi1 solves."""

    if mode in {"basic", "full"}:
        return x + (float(step_scale) * step_direction)

    step = 1.0
    best_x = None
    best_rnorm = float("inf")
    step_candidates = [1.0, 1.5, 2.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] if mode == "best" else None

    for _ in range(int(maxiter)):
        if mode == "best":
            try_step = step_candidates.pop(0) if step_candidates else step
        else:
            try_step = step

        x_try = x + (float(try_step) * float(step_scale)) * step_direction
        r_try = residual_fn(x_try)
        rnorm_try = float(jnp.linalg.norm(r_try))
        if not np.isfinite(rnorm_try):
            if mode != "best":
                step *= 0.5
            continue

        if rnorm_try < best_rnorm:
            best_rnorm = rnorm_try
            best_x = x_try

        if mode != "best":
            if factor is not None:
                accept = rnorm_try <= float(factor) * float(residual_norm0)
            else:
                accept = rnorm_try <= (1.0 - float(c1) * float(step)) * float(residual_norm0)
            if accept:
                return x_try
            step *= 0.5

    if mode == "best" and best_x is not None and best_rnorm < float(residual_norm0):
        return best_x
    if best_x is not None and np.isfinite(best_rnorm):
        return best_x
    if accepted:
        return accepted[-1]
    return x + (1.0 / 64.0) * step_direction


def solve_v3_full_system_newton_krylov(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
) -> V3NewtonKrylovResult:
    """Solve ``residual_v3_full_system(op, x) = 0`` with Newton-Krylov.

    This small path is intended for parity fixtures and developer experiments.
    Production output writing uses :func:`solve_v3_full_system_newton_krylov_history`
    so accepted Newton iterates can be serialized.
    """

    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    set_precond_size_hint(int(op.total_size))
    set_precond_policy_hints(
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
    )
    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for k in range(int(max_newton)):
        r, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
        rnorm = jnp.linalg.norm(r)
        if float(rnorm) < float(tol):
            return V3NewtonKrylovResult(
                op=op,
                x=x,
                residual_norm=rnorm,
                n_newton=k,
                last_linear_residual_norm=last_linear_resid,
            )

        lin = _dispatch_gmres(
            matvec=jvp,
            b=-r,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=str(solve_method),
        )
        s = lin.x
        last_linear_resid = lin.residual_norm

        step = 1.0
        step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
        try:
            step_scale = float(step_scale_env) if step_scale_env else 1.0
        except ValueError:
            step_scale = 1.0
        rnorm0 = float(rnorm)
        for _ in range(12):
            x_try = x + (step * step_scale) * s
            r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if rnorm_try <= 0.9 * rnorm0:
                x = x_try
                break
            step *= 0.5
        else:
            x = x + (1.0 / 64.0) * s

    r = residual_v3_full_system(op, x)
    return V3NewtonKrylovResult(
        op=op,
        x=x,
        residual_norm=jnp.linalg.norm(r),
        n_newton=int(max_newton),
        last_linear_residual_norm=last_linear_resid,
    )


def solve_v3_full_system_newton_krylov_history(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
    nonlinear_rtol: float = 0.0,
    use_frozen_linearization: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[V3NewtonKrylovResult, list[jnp.ndarray]]:
    """Newton-Krylov solve that returns the accepted Newton-state history."""

    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    if emit is not None:
        emit(1, f"solve_v3_full_system_newton_krylov_history: total_size={int(op.total_size)}")
    fortran_stdout = rhs1_fortran_stdout_from_env(emit=emit)
    env_gmres_tol = os.environ.get("SFINCS_JAX_PHI1_GMRES_TOL", "").strip()
    if env_gmres_tol:
        gmres_tol = float(env_gmres_tol)

    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    active_env = os.environ.get("SFINCS_JAX_PHI1_ACTIVE_DOF", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    has_reduced_modes = bool(np.any(nxi_for_x < int(op.n_xi)))
    use_active_dof_mode = phi1_use_active_dof_mode(
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        has_reduced_modes=has_reduced_modes,
        env_value=active_env,
    )

    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op.total_size)
    if use_active_dof_mode:
        active_idx_np = _transport_active_dof_indices(op)
        active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
        full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
        full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
            1,
            int(active_idx_np.shape[0]) + 1,
            dtype=np.int32,
        )
        full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
        active_size = int(active_idx_np.shape[0])
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_newton_krylov_history: active-DOF mode enabled "
                f"(size={active_size}/{int(op.total_size)})",
            )
    gmres_restart_use = phi1_gmres_restart(active_size=active_size, gmres_restart=int(gmres_restart))

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        assert active_idx_jnp is not None
        return v_full[active_idx_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        assert full_to_active_jnp is not None
        z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
        padded = jnp.concatenate([z0, v_reduced], axis=0)
        return padded[full_to_active_jnp]

    pc_env = os.environ.get("SFINCS_JAX_PHI1_USE_PRECONDITIONER", "").strip().lower()
    use_preconditioner = pc_env not in {"0", "false", "no", "off"}
    dense_cutoff_env = os.environ.get("SFINCS_JAX_PHI1_NK_DENSE_CUTOFF", "").strip()
    try:
        dense_cutoff = int(dense_cutoff_env) if dense_cutoff_env else 5000
    except ValueError:
        dense_cutoff = 5000
    linear_size = active_size if use_active_dof_mode else int(op.total_size)
    solve_method_in = str(solve_method).strip().lower()
    use_sparse_direct_linear = solve_method_in == "sparse_direct"
    use_dense_linear = solve_method_in in {"dense", "dense_row_scaled"} or (
        use_frozen_linearization and int(linear_size) <= int(dense_cutoff)
    )
    if use_dense_linear or use_sparse_direct_linear:
        use_preconditioner = False
    preconditioner = build_phi1_newton_preconditioner(
        use_preconditioner=bool(use_preconditioner),
        use_frozen_linearization=bool(use_frozen_linearization),
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        use_active_dof_mode=bool(use_active_dof_mode),
        op=op,
        reduce_full=_reduce_full if use_active_dof_mode else None,
        expand_reduced=_expand_reduced if use_active_dof_mode else None,
        preconditioner_options=nml.group("preconditionerOptions"),
        collision_builder=build_rhs1_collision_preconditioner,
        block_builder=build_rhs1_block_preconditioner,
        emit=emit,
    )

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)
    accepted: list[jnp.ndarray] = []
    rnorm_initial: float | None = None
    cached_jvp = None
    cached_jvp_iter = -1
    frozen_jac_policy = phi1_frozen_jacobian_policy(include_phi1=bool(op.include_phi1))
    use_frozen_jac_cache = bool(frozen_jac_policy.use_cache)
    frozen_jac_every = int(frozen_jac_policy.every)
    ksp_history_limits = rhs1_ksp_history_limits_from_env()
    ksp_history_max_size = ksp_history_limits.max_size
    ksp_history_max_iter = ksp_history_limits.max_iter

    def _emit_ksp_history_nk(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
    ) -> None:
        emit_newton_krylov_ksp_history(
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            precond_side=precond_side,
            emit=emit,
            fortran_stdout=bool(fortran_stdout),
            max_size=ksp_history_max_size,
            max_history_iter=ksp_history_max_iter,
        )

    def _phi1_sparse_direct_solve(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        n: int,
        cache_tag: tuple[object, ...],
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
    ) -> GMRESSolveResult:
        factor_dtype = _phi1_host_sparse_factor_dtype(size=int(n))
        cache_key_use = ("phi1_nk_sparse_direct", *cache_tag, str(factor_dtype))
        a_csr_full, _a_csr_drop, ilu, _a_dense, _l_dense, _u_dense, _l_unit = build_sparse_ilu_from_matvec(
            matvec=matvec_fn,
            n=int(n),
            dtype=jnp.float64,
            cache_key=cache_key_use,
            factor_dtype=factor_dtype,
            drop_tol=0.0,
            drop_rel=0.0,
            ilu_drop_tol=0.0,
            fill_factor=1.0,
            build_dense_factors=False,
            build_jax_factors=False,
            build_ilu=True,
            store_dense=False,
            factorization="lu",
            emit=emit,
        )
        if ilu is None:
            raise RuntimeError("phi1 sparse_direct: factors unavailable")
        x_np, residual_norm = host_sparse_direct_solve_with_refinement(
            ilu=ilu,
            a_csr_full=a_csr_full,
            rhs_vec=b_vec,
            factor_dtype=factor_dtype,
            refine_steps=host_sparse_direct_refine_steps("SFINCS_JAX_PHI1_SPARSE_DIRECT_REFINE", default=2),
        )
        target_true = max(float(atol_val), float(tol_val) * float(jnp.linalg.norm(b_vec)))
        if factor_dtype == np.dtype(np.float32) and residual_norm > target_true:
            polish_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH", "").strip().lower()
            if polish_env not in {"0", "false", "no", "off"}:
                polish_restart_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH_RESTART", "").strip()
                polish_maxiter_env = os.environ.get("SFINCS_JAX_PHI1_SPARSE_DIRECT_POLISH_MAXITER", "").strip()
                try:
                    polish_restart = int(polish_restart_env) if polish_restart_env else min(int(restart_val), 40)
                except ValueError:
                    polish_restart = min(int(restart_val), 40)
                try:
                    polish_maxiter = (
                        int(polish_maxiter_env)
                        if polish_maxiter_env
                        else min(max(40, int(maxiter_val or 120)), 120)
                    )
                except ValueError:
                    polish_maxiter = min(max(40, int(maxiter_val or 120)), 120)
                x_polish, residual_norm_polish = host_sparse_direct_polish(
                    matvec_fn=matvec_fn,
                    rhs_vec=b_vec,
                    x0_np=x_np,
                    ilu=ilu,
                    factor_dtype=factor_dtype,
                    tol=tol_val,
                    atol=atol_val,
                    restart=max(5, int(polish_restart)),
                    maxiter=max(5, int(polish_maxiter)),
                    precondition_side="left",
                )
                if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm:
                    x_np = x_polish
                    residual_norm = residual_norm_polish
        return GMRESSolveResult(
            x=jnp.asarray(x_np, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        )

    for k in range(int(max_newton)):
        if emit is not None:
            emit(1, f"newton_iter={k}: evaluateResidual called")
        op_use = op
        if bool(op.include_phi1):
            phi1_flat = x[op.f_size : op.f_size + op.n_theta * op.n_zeta]
            phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
            op_use = replace(op, phi1_hat_base=phi1)

        r = apply_v3_full_system_operator_cached(op_use, x, include_jacobian_terms=False) - rhs_v3_full_system_jit(op_use)
        rnorm = jnp.linalg.norm(r)
        rnorm_f = float(rnorm)
        if rnorm_initial is None:
            rnorm_initial = max(rnorm_f, 1e-300)
        if emit is not None:
            emit(0, f"newton_iter={k}: residual_norm={rnorm_f:.6e}")
        if emit is not None and fortran_stdout:
            emit(0, f"{k:4d} SNES Function norm {rnorm_f: .12e} ")
        if not np.isfinite(rnorm_f):
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        converged_abs = rnorm_f < float(tol)
        converged_rel = rnorm_f <= float(nonlinear_rtol) * float(rnorm_initial)
        if converged_abs or converged_rel:
            if not accepted:
                accepted.append(x)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x,
                    residual_norm=rnorm,
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        if use_frozen_linearization:
            jac_mode = frozen_jac_policy.mode
            if jac_mode == "frozen_rhs":

                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_rhs_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_rhs_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_use, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_rhs_x)
                    )

                reuse_cached = (
                    use_frozen_jac_cache
                    and cached_jvp is not None
                    and (k - cached_jvp_iter) < frozen_jac_every
                )
                if reuse_cached:
                    matvec = cached_jvp
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian reused (frozen_rhs cache)")
                else:
                    _r_lin, jvp = jax.linearize(residual_for_jac, x)
                    matvec = jvp
                    if use_frozen_jac_cache:
                        cached_jvp = jvp
                        cached_jvp_iter = k
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian called (frozen operator + dynamic RHS)")
            elif jac_mode == "frozen_op":

                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_mat_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_mat_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_mat_x, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_use)
                    )

                _r_lin, jvp = jax.linearize(residual_for_jac, x)
                matvec = jvp
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (dynamic operator + frozen RHS)")
            else:
                def matvec(dx: jnp.ndarray) -> jnp.ndarray:
                    return apply_v3_full_system_jacobian_jit(op_use, x, dx)

                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (fully frozen linearization)")
        else:
            _r_lin, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
            matvec = jvp
            if emit is not None:
                emit(1, f"newton_iter={k}: evaluateJacobian called (autodiff linearization)")

        solve_method_linear = str(solve_method)
        if use_frozen_linearization and int(linear_size) <= int(dense_cutoff):
            solve_method_linear = "dense"

        lin, s, linear_resid_norm = solve_phi1_newton_linear_step(
            use_active_dof_mode=bool(use_active_dof_mode),
            solve_method_linear=solve_method_linear,
            matvec=matvec,
            residual_vec=r,
            preconditioner=preconditioner,
            gmres_tol=float(gmres_tol),
            gmres_restart=int(gmres_restart_use),
            gmres_maxiter=gmres_maxiter,
            sparse_direct_solve=_phi1_sparse_direct_solve,
            gmres_dispatch=_dispatch_gmres,
            gmres_result_is_finite=gmres_result_is_finite,
            emit_ksp_history=_emit_ksp_history_nk,
            emit=emit,
            newton_iter=int(k),
            reduce_full=_reduce_full if use_active_dof_mode else None,
            expand_reduced=_expand_reduced if use_active_dof_mode else None,
            active_size=int(active_size),
            total_size=int(op.total_size),
        )

        if emit is not None:
            emit(1, f"newton_iter={k}: gmres_residual={float(linear_resid_norm):.6e}")
        if not gmres_result_is_finite(lin):
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )
        last_linear_resid = linear_resid_norm

        ls_policy = phi1_line_search_policy(
            use_frozen_linearization=bool(use_frozen_linearization),
            include_phi1=bool(op.include_phi1),
        )
        x = advance_phi1_newton_iterate(
            x=x,
            step_direction=s,
            residual_norm0=float(rnorm),
            residual_fn=lambda x_try: residual_v3_full_system(op, x_try),
            accepted=accepted,
            mode=str(ls_policy.mode),
            step_scale=float(ls_policy.step_scale),
            factor=ls_policy.factor,
            c1=float(ls_policy.c1),
            maxiter=int(ls_policy.maxiter),
        )
        accepted.append(x)

    r = residual_v3_full_system(op, x)
    return (
        V3NewtonKrylovResult(
            op=op,
            x=x,
            residual_norm=jnp.linalg.norm(r),
            n_newton=int(max_newton),
            last_linear_residual_norm=last_linear_resid,
        ),
        accepted,
    )


__all__ = [
    "solve_v3_full_system_newton_krylov",
    "solve_v3_full_system_newton_krylov_history",
]
