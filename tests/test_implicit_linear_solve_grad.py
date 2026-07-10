from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.krylov import GMRESSolveResult
import sfincs_jax.solvers.implicit as implicit_module
from sfincs_jax.solvers.implicit import (
    ImplicitGMRESSolveResult,
    ImplicitLinearSolveResult,
    _use_solver_jit,
    gmres_custom_linear_solve,
    implicit_solve_method_for_custom_linear_solve,
    linear_custom_solve,
    linear_custom_solve_with_residual,
)


def test_implicit_solve_method_for_custom_linear_solve_downgrades_host_only_methods() -> None:
    assert implicit_solve_method_for_custom_linear_solve("lgmres") == "incremental"
    assert implicit_solve_method_for_custom_linear_solve(" LGMRES-SCIPY ") == "incremental"
    assert implicit_solve_method_for_custom_linear_solve("batched") == "batched"


def test_implicit_solver_jit_admission_respects_env_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", raising=False)
    assert _use_solver_jit(size_hint=None)
    assert _use_solver_jit(size_hint=2000)
    assert not _use_solver_jit(size_hint=2001)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "bad")
    assert _use_solver_jit(size_hint=2000)
    assert not _use_solver_jit(size_hint=2001)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "3")
    assert _use_solver_jit(size_hint=3)
    assert not _use_solver_jit(size_hint=4)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "true")
    assert _use_solver_jit(size_hint=10_000_000)

    monkeypatch.setenv("SFINCS_JAX_SOLVER_JIT", "off")
    assert not _use_solver_jit(size_hint=1)


def test_implicit_result_wrappers_are_pytrees() -> None:
    linear = ImplicitLinearSolveResult(
        x=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(0.25, dtype=jnp.float64),
    )
    leaves, treedef = jax.tree_util.tree_flatten(linear)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.x), [1.0, 2.0])
    assert float(rebuilt.residual_norm) == 0.25

    gmres = ImplicitGMRESSolveResult(
        x=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        gmres=GMRESSolveResult(
            x=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
            residual_norm=jnp.asarray(1e-12, dtype=jnp.float64),
        ),
    )
    leaves, treedef = jax.tree_util.tree_flatten(gmres)
    rebuilt_gmres = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt_gmres.x), [3.0, 4.0])
    np.testing.assert_allclose(np.asarray(rebuilt_gmres.gmres.x), [3.0, 4.0])
    assert float(rebuilt_gmres.gmres.residual_norm) == pytest.approx(1e-12)


def test_linear_custom_solve_uses_transpose_preconditioner_under_vjp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_gmres_solve(**kwargs):
        preconditioner = kwargs.get("preconditioner")
        rhs = kwargs["b"]
        calls.append((kwargs.get("solve_method"), preconditioner))
        x = preconditioner(rhs) if preconditioner is not None else rhs
        return GMRESSolveResult(x=x, residual_norm=jnp.asarray(0.0, dtype=jnp.float64))

    monkeypatch.setattr(implicit_module, "gmres_solve", fake_gmres_solve)

    def matvec(x):
        return x

    def preconditioner(rhs):
        return rhs

    def transpose_preconditioner(rhs):
        return 2.0 * rhs

    b = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    def objective(rhs):
        return jnp.vdot(
            jnp.asarray([0.25, -0.5], dtype=jnp.float64),
            linear_custom_solve(
                matvec=matvec,
                b=rhs,
                solver="gmres",
                solve_method="lgmres",
                preconditioner=preconditioner,
                preconditioner_transpose=transpose_preconditioner,
                solver_jit=False,
            ).x,
        )

    grad = jax.grad(objective)(b)
    np.testing.assert_allclose(np.asarray(grad), [0.5, -1.0])
    assert calls[0] == ("incremental", preconditioner)
    assert calls[-1] == ("incremental", transpose_preconditioner)


def test_linear_custom_solve_with_residual_returns_vector_residual() -> None:
    a = jnp.asarray([[3.0, 1.0], [0.5, 2.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, -0.25], dtype=jnp.float64)

    def matvec(x):
        return a @ x

    result, residual = linear_custom_solve_with_residual(
        matvec=matvec,
        b=b,
        tol=1e-12,
        restart=8,
        maxiter=20,
        solver="gmres",
        solver_jit=False,
    )

    np.testing.assert_allclose(np.asarray(matvec(result.x) + residual), np.asarray(b), atol=1e-10)
    assert float(result.residual_norm) < 1e-10
    assert residual.shape == b.shape


def test_custom_linear_solve_grad_matches_finite_difference() -> None:
    # A small, well-conditioned linear system A(p) x = b with A(p) = A0 + p*I.
    a0 = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float64)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=jnp.float64)
        a = a0 + p * jnp.eye(4, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        x = gmres_custom_linear_solve(matvec=mv, b=b, tol=1e-12, restart=20, maxiter=50).x
        return 0.5 * jnp.vdot(x, x)

    p0 = jnp.asarray(0.2, dtype=jnp.float64)
    g = float(jax.grad(objective)(p0))

    eps = 1e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)

    assert np.isfinite(g)
    assert abs(g - fd) < 5e-6


def test_custom_linear_solve_bicgstab_grad_matches_finite_difference() -> None:
    a0 = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float64)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=jnp.float64)
        a = a0 + p * jnp.eye(4, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        x = linear_custom_solve(
            matvec=mv,
            b=b,
            tol=1e-12,
            maxiter=100,
            solver="bicgstab",
        ).x
        return 0.5 * jnp.vdot(x, x)

    p0 = jnp.asarray(0.2, dtype=jnp.float64)
    g = float(jax.grad(objective)(p0))

    eps = 1e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)

    assert np.isfinite(g)
    assert abs(g - fd) < 5e-6


def test_custom_linear_solve_lgmres_falls_back_to_traced_safe_solver() -> None:
    a0 = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float64)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=jnp.float64)
        a = a0 + p * jnp.eye(4, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        x = linear_custom_solve(
            matvec=mv,
            b=b,
            tol=1e-12,
            restart=20,
            maxiter=50,
            solver="lgmres",
        ).x
        return 0.5 * jnp.vdot(x, x)

    p0 = jnp.asarray(0.2, dtype=jnp.float64)
    g = float(jax.grad(objective)(p0))

    eps = 1e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)

    assert np.isfinite(g)
    assert abs(g - fd) < 5e-6


def test_custom_linear_solve_host_only_solve_method_falls_back_under_grad() -> None:
    """Guard the differentiable lane against CLI-only SciPy Krylov choices."""

    a0 = jnp.asarray(
        [
            [5.0, -0.5, 0.25, 0.0],
            [0.75, 4.0, -0.5, 0.25],
            [0.0, 0.5, 3.5, -0.75],
            [0.25, 0.0, 0.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([0.25, -1.0, 2.0, 0.5], dtype=jnp.float64)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=jnp.float64)
        a = a0 + p * jnp.diag(jnp.asarray([1.0, 0.5, 1.5, 2.0], dtype=jnp.float64))

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        x = linear_custom_solve(
            matvec=mv,
            b=b,
            tol=1e-12,
            restart=20,
            maxiter=80,
            solve_method="lgmres_scipy",
            solver="gmres",
        ).x
        return jnp.sum(jnp.sin(x) + 0.1 * x * x)

    p0 = jnp.asarray(0.15, dtype=jnp.float64)
    g = float(jax.grad(objective)(p0))

    eps = 1e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)

    assert np.isfinite(g)
    assert abs(g - fd) < 5e-6
