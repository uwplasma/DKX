from __future__ import annotations

import jax.numpy as jnp

from sfincs_jax.problems.profile_response.linear_solve import (
    ProfileLinearSolveContext,
    profile_solver_kind,
    solve_profile_linear_with_residual,
)


def _context(
    *,
    rhs_mode: int = 1,
    total_size: int = 1000,
    use_implicit: bool = False,
    use_solver_jit: bool = False,
    distributed_axis: str | None = None,
    distributed_auto_solver: str = "bicgstab",
    small_gmres_max: int = 600,
) -> ProfileLinearSolveContext:
    return ProfileLinearSolveContext(
        rhs_mode=rhs_mode,
        total_size=total_size,
        use_implicit=use_implicit,
        use_solver_jit=use_solver_jit,
        distributed_axis=distributed_axis,
        distributed_auto_solver=distributed_auto_solver,
        small_gmres_max=small_gmres_max,
    )


def test_profile_solver_kind_preserves_rhs1_auto_defaults() -> None:
    assert profile_solver_kind("auto", context=_context(total_size=100)) == ("gmres", "incremental")
    assert profile_solver_kind("default", context=_context(total_size=10000)) == ("gmres", "incremental")
    assert profile_solver_kind("bicgstab", context=_context()) == ("bicgstab", "batched")
    assert profile_solver_kind("lgmres", context=_context()) == ("gmres", "lgmres")


def test_profile_solver_kind_prefers_bicgstab_for_distributed_auto() -> None:
    context = _context(distributed_axis="theta", distributed_auto_solver="bicgstab")

    assert profile_solver_kind("auto", context=context) == ("bicgstab", "batched")


def test_solve_profile_linear_with_residual_solves_tiny_identity_system() -> None:
    b = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    result, residual = solve_profile_linear_with_residual(
        context=_context(total_size=2, small_gmres_max=10),
        matvec_fn=lambda x: x,
        b_vec=b,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=4,
        maxiter_val=8,
        solve_method_val="auto",
        precond_side="left",
    )

    assert jnp.linalg.norm(result.x - b) < 1.0e-10
    assert jnp.linalg.norm(residual) < 1.0e-10
