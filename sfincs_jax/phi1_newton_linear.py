from __future__ import annotations

from collections.abc import Callable
import os

import jax.numpy as jnp
import numpy as np

from .solver import GMRESSolveResult


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
    """Solve one Newton linearization step for the Phi1 nonlinear path.

    The helper preserves the current reduced/full routing and the bounded retry
    semantics used in the production Newton-Krylov path. It does not choose the
    Jacobian or line-search policy; it only executes the linear solve for a
    given linear operator and residual.
    """
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
