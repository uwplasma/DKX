from __future__ import annotations

import jax.numpy as jnp

from sfincs_jax.problems.profile_response.linear_solve import (
    ProfileLinearSolveContext,
    RHS1ScipyRescueContext,
    profile_solver_kind,
    rhs1_small_gmres_max_from_env,
    run_rhs1_scipy_rescue,
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


def test_rhs1_small_gmres_max_env_preserves_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", raising=False)
    assert rhs1_small_gmres_max_from_env() == 600

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "42")
    assert rhs1_small_gmres_max_from_env() == 42

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "bad")
    assert rhs1_small_gmres_max_from_env(default=17) == 17


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


def test_run_rhs1_scipy_rescue_gmres_recomputes_true_residual() -> None:
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    outcome = run_rhs1_scipy_rescue(
        context=RHS1ScipyRescueContext(
            matvec=lambda x: a @ x,
            rhs=b,
            x0=jnp.zeros_like(b),
            preconditioner=None,
            method="gmres",
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=12,
            precond_side="left",
        )
    )

    assert jnp.linalg.norm(a @ outcome.result.x - b) < 1.0e-10
    assert jnp.linalg.norm(outcome.residual_vec) < 1.0e-10
    assert outcome.reported_residual < 1.0e-10
    assert outcome.history_len >= 1
    assert outcome.preconditioned_residual is None


def test_run_rhs1_scipy_rescue_bicgstab_recomputes_true_residual() -> None:
    a = jnp.asarray([[5.0, 0.5], [0.25, 2.0]], dtype=jnp.float64)
    b = jnp.asarray([2.0, -1.0], dtype=jnp.float64)

    outcome = run_rhs1_scipy_rescue(
        context=RHS1ScipyRescueContext(
            matvec=lambda x: a @ x,
            rhs=b,
            x0=jnp.zeros_like(b),
            preconditioner=None,
            method="bicgstab",
            tol=1.0e-12,
            atol=1.0e-12,
            restart=4,
            maxiter=20,
            precond_side="left",
        )
    )

    assert jnp.linalg.norm(a @ outcome.result.x - b) < 1.0e-10
    assert jnp.linalg.norm(outcome.residual_vec) < 1.0e-10
    assert outcome.reported_residual < 1.0e-10
    assert outcome.history_len >= 1
