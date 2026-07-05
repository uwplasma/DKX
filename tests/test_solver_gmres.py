from __future__ import annotations

import os
import numpy as np
import subprocess
import sys
import jax
import jax.numpy as jnp
import pytest

import sfincs_jax.solver as solver_module
from sfincs_jax.solvers.krylov_dispatch import rhs_krylov_method_for_context as _rhs_krylov_method_for_context
from sfincs_jax.solver import (
    _materialize_distributed_input,
    _distributed_solver_kind,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve,
    bicgstab_solve_with_residual,
    dense_krylov_solve_from_matrix,
    dense_solve_from_matrix,
    explicit_left_preconditioned_gmres_scipy,
    fgmres_cycle_jit_solve_with_residual,
    fgmres_solve_with_residual,
    fgmres_solve_with_residual_jit,
    gcrotmk_solve_with_history_scipy,
    gmres_solve_distributed,
    gmres_solve_with_history_scipy,
    gmres_solve_jit,
    gmres_solve,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    lgmres_solve_with_history_scipy,
    tfqmr_solve,
    tfqmr_solve_with_residual,
    tfqmr_solve_with_residual_jit,
)


def test_gmres_solve_matches_numpy_for_spd_matrix() -> None:
    rng = np.random.default_rng(0)
    n = 24
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.5 * np.eye(n)  # SPD, well-conditioned enough.
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result = gmres_solve(matvec=mv, b=b_j, tol=1e-12, restart=30, maxiter=200)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-8)
    assert float(result.residual_norm) < 1e-8


def test_solver_result_wrappers_are_jax_pytrees() -> None:
    """Solver result dataclasses must remain transform-safe public contracts."""

    flexible = solver_module.FlexibleGMRESSolveResult(
        x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0e-8, dtype=jnp.float64),
        residual_history=jnp.asarray([1.0, 1.0e-8], dtype=jnp.float64),
        n_iterations=jnp.asarray(1, dtype=jnp.int32),
        n_restarts=jnp.asarray(0, dtype=jnp.int32),
        converged=jnp.asarray(True),
    )
    bicgstab = solver_module.BiCGSTABSolveResult(
        x=flexible.x,
        residual_norm=flexible.residual_norm,
        residual_history=flexible.residual_history,
        n_iterations=flexible.n_iterations,
        converged=flexible.converged,
    )
    tfqmr = solver_module.TFQMRSolveResult(
        x=flexible.x,
        residual_norm=flexible.residual_norm,
        residual_history=flexible.residual_history,
        n_iterations=flexible.n_iterations,
        converged=flexible.converged,
    )

    for result in (flexible, bicgstab, tfqmr):
        leaves, treedef = jax.tree_util.tree_flatten(result)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(np.asarray(rebuilt.x), np.asarray(result.x), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(rebuilt.residual_history),
            np.asarray(result.residual_history),
            rtol=0.0,
            atol=0.0,
        )
        assert bool(rebuilt.converged) is True


def test_gcrotmk_solve_with_history_scipy_matches_numpy_for_small_system() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.0],
            [1.0, 3.0, -1.0],
            [0.0, -0.5, 2.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)

    def mv(x):
        return jnp.asarray(a) @ x

    x, residual_norm, history = gcrotmk_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        tol=1.0e-12,
        restart=3,
        maxiter=20,
    )

    np.testing.assert_allclose(x, x_ref, rtol=1.0e-10, atol=1.0e-10)
    assert residual_norm < 1.0e-10
    assert history


def test_dense_solve_from_matrix_supports_multiple_rhs() -> None:
    rng = np.random.default_rng(3)
    n = 18
    k = 3
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.3 * np.eye(n)
    b = rng.normal(size=(n, k)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_solve_from_matrix(a=jnp.asarray(a), b=jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-10, atol=1e-10)
    assert np.asarray(rn).shape == (k,)
    assert float(np.max(np.asarray(rn))) < 1e-9


def test_dense_solve_from_matrix_regularizes_singular_system() -> None:
    a = np.array(
        [
            [2.0, -1.0, 0.0],
            [4.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, 2.0, -3.0], dtype=np.float64)

    x, rn = dense_solve_from_matrix(a=jnp.asarray(a), b=jnp.asarray(b))
    x_np = np.asarray(x)

    assert np.all(np.isfinite(x_np))
    np.testing.assert_allclose(a @ x_np, b, rtol=1e-8, atol=1e-8)
    assert float(rn) < 1e-8


def test_dense_solve_modes_and_shape_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin dense direct-solve guardrails without running production systems."""

    with pytest.raises(ValueError, match="square matrix"):
        dense_solve_from_matrix(a=jnp.ones((2, 3), dtype=jnp.float64), b=jnp.ones(2, dtype=jnp.float64))
    with pytest.raises(ValueError, match="b.ndim"):
        dense_solve_from_matrix(a=jnp.eye(2, dtype=jnp.float64), b=jnp.ones((2, 1, 1), dtype=jnp.float64))
    with pytest.raises(ValueError, match="shape mismatch"):
        dense_solve_from_matrix(a=jnp.eye(3, dtype=jnp.float64), b=jnp.ones(2, dtype=jnp.float64))
    with pytest.raises(ValueError, match="square matrix"):
        solver_module.dense_solve_from_matrix_row_scaled(
            a=jnp.ones((2, 3), dtype=jnp.float64),
            b=jnp.ones(2, dtype=jnp.float64),
        )

    singular = jnp.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float64)
    rhs = jnp.asarray([2.0, 4.0], dtype=jnp.float64)

    monkeypatch.setenv("SFINCS_JAX_DENSE_FORCE_REG", "1")
    monkeypatch.setenv("SFINCS_JAX_DENSE_REG", "1e-8")
    x_reg, rn_reg = dense_solve_from_matrix(a=singular, b=rhs)
    assert np.isfinite(np.asarray(x_reg)).all()
    assert float(rn_reg) < 1.0e-6

    monkeypatch.delenv("SFINCS_JAX_DENSE_FORCE_REG", raising=False)
    monkeypatch.setenv("SFINCS_JAX_DENSE_SINGULAR_MODE", "lstsq")
    x_lstsq, rn_lstsq = dense_solve_from_matrix(a=singular, b=rhs)
    assert np.isfinite(np.asarray(x_lstsq)).all()
    assert float(rn_lstsq) < 1.0e-8


def test_gmres_dense_dispatch_and_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense and row-scaled dense dispatch must remain explicit and bounded."""

    a = np.asarray(
        [
            [2.0, -1.0, 0.25],
            [0.5, 3.0, -0.75],
            [0.0, 0.5, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    for method in ("dense", "dense_row_scaled"):
        result, residual = gmres_solve_with_residual(
            matvec=mv,
            b=jnp.asarray(b),
            solve_method=method,
        )
        np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-12, atol=1.0e-12)
        assert float(result.residual_norm) < 1.0e-10

    monkeypatch.setenv("SFINCS_JAX_DENSE_MAX", "2")
    with pytest.raises(ValueError, match="too large"):
        gmres_solve_with_residual(
            matvec=mv,
            b=jnp.asarray(b),
            solve_method="dense",
        )

    monkeypatch.setenv("SFINCS_JAX_DENSE_MAX", "bad")
    result = gmres_solve(matvec=mv, b=jnp.asarray(b), solve_method="dense")
    assert float(result.residual_norm) < 1.0e-10


def test_dense_krylov_solve_from_matrix_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    n = 20
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.25 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_krylov_solve_from_matrix(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        tol=1e-12,
        restart=n,
        maxiter=8,
        solve_method="incremental",
    )

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8, atol=1e-8)
    assert float(rn) < 1e-8


def test_dense_krylov_env_shape_and_row_scaled_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row-scaled Krylov solves disable incompatible user preconditioners."""

    with pytest.raises(ValueError, match="square matrix"):
        solver_module.dense_krylov_solve_from_matrix_with_residual(
            a=jnp.ones((2, 3), dtype=jnp.float64),
            b=jnp.ones(2, dtype=jnp.float64),
        )
    with pytest.raises(ValueError, match="b.shape"):
        solver_module.dense_krylov_solve_from_matrix_with_residual(
            a=jnp.eye(3, dtype=jnp.float64),
            b=jnp.ones((3, 1), dtype=jnp.float64),
        )

    monkeypatch.setenv("SFINCS_JAX_DENSE_KRYLOV_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_DENSE_KRYLOV_MAXITER", "bad")
    a = jnp.diag(jnp.asarray([1.0e-8, 2.0, 5.0e3], dtype=jnp.float64))
    b = jnp.asarray([2.0e-8, -4.0, 1.0e4], dtype=jnp.float64)

    def forbidden_preconditioner(x):
        raise AssertionError("row-scaled dense Krylov must ignore external preconditioners")

    result, residual = solver_module.dense_krylov_solve_from_matrix_with_residual(
        a=a,
        b=b,
        preconditioner=forbidden_preconditioner,
        row_scaled=True,
        tol=1.0e-12,
        solve_method="incremental",
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray([2.0, -2.0, 2.0]), rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), np.zeros(3), atol=1.0e-8)
    assert float(result.residual_norm) < 1.0e-8


def test_dense_krylov_lgmres_matches_numpy_on_nonsymmetric_matrix() -> None:
    rng = np.random.default_rng(17)
    n = 18
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 6.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_krylov_solve_from_matrix(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        tol=1e-12,
        restart=8,
        maxiter=40,
        solve_method="lgmres",
    )

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8, atol=1e-8)
    assert float(rn) < 1e-8


def test_dense_krylov_row_scaled_handles_diagonal_imbalance() -> None:
    diag = np.array([1.0e-8, 2.0, 5.0e4, 7.0], dtype=np.float64)
    a = np.diag(diag)
    a[0, 1] = -3.0e-7
    a[1, 0] = 2.0e-7
    b = np.array([2.0e-8, -4.0, 1.5e5, 3.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_krylov_solve_from_matrix(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        tol=1e-12,
        restart=a.shape[0],
        maxiter=4,
        solve_method="incremental",
        row_scaled=True,
    )

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-7, atol=1e-9)
    assert float(rn) < 1e-7


def test_assemble_dense_matrix_from_matvec_recovers_operator() -> None:
    rng = np.random.default_rng(7)
    n = 13
    a = rng.normal(size=(n, n)).astype(np.float64)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    assembled = assemble_dense_matrix_from_matvec(matvec=mv, n=n, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(assembled), a, rtol=0.0, atol=1e-12)


def test_gmres_solve_with_residual_matches_matvec() -> None:
    rng = np.random.default_rng(11)
    n = 16
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.4 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result, residual = gmres_solve_with_residual(matvec=mv, b=b_j, tol=1e-12, restart=30, maxiter=200)
    r_expected = b_j - mv(result.x)

    np.testing.assert_allclose(np.asarray(residual), np.asarray(r_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        float(result.residual_norm),
        float(jnp.linalg.norm(residual)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_gmres_right_preconditioning_preserves_physical_initial_guess() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.25],
            [1.5, 3.0, -0.5],
            [0.0, 0.75, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    x0 = x_ref + np.array([0.25, -0.1, 0.05], dtype=np.float64)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = gmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        x0=jnp.asarray(x0, dtype=jnp.float64),
        tol=1.0e-12,
        restart=3,
        maxiter=12,
        solve_method="incremental",
        precondition_side="right",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-10


def test_fgmres_solve_with_residual_matches_numpy_for_nonsymmetric_matrix() -> None:
    rng = np.random.default_rng(101)
    n = 18
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 7.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = fgmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-12,
        restart=8,
        maxiter=80,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-8
    assert bool(result.converged)
    assert int(result.n_iterations) <= 80
    assert np.asarray(result.residual_history).ndim == 1
    assert float(np.asarray(result.residual_history)[-1]) == pytest.approx(float(result.residual_norm), rel=1.0e-10)


def test_fgmres_true_residual_is_preconditioner_side_invariant() -> None:
    """Converged FGMRES diagnostics must be physical, not preconditioned residuals."""

    a = np.asarray(
        [
            [6.0, -1.0, 0.5, 0.0],
            [1.5, 5.0, -0.25, 0.75],
            [0.0, 0.5, 4.5, -1.0],
            [0.25, 0.0, 1.0, 3.5],
        ],
        dtype=np.float64,
    )
    b = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    results = {}
    residuals = {}
    for side, preconditioner in (("none", None), ("left", precond), ("right", precond)):
        result, residual = fgmres_solve_with_residual(
            matvec=mv,
            b=b_j,
            preconditioner=preconditioner,
            tol=1.0e-12,
            restart=4,
            maxiter=8,
            precondition_side=side,
        )
        results[side] = result
        residuals[side] = residual

    for side, result in results.items():
        np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(np.asarray(residuals[side]), b - a @ np.asarray(result.x), rtol=1.0e-11, atol=1.0e-11)
        assert bool(result.converged)
        assert float(result.residual_norm) < 1.0e-10

    np.testing.assert_allclose(np.asarray(results["left"].x), np.asarray(results["right"].x), rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(residuals["left"]), np.asarray(residuals["none"]), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(residuals["right"]), np.asarray(residuals["none"]), rtol=1.0e-10, atol=1.0e-10)


def test_fgmres_right_preconditioner_is_transpose_safe() -> None:
    """Guard the device-QI installed-Krylov path against scatter transpose failures."""

    a = jnp.asarray(
        [
            [4.0, -1.0, 0.2],
            [0.5, 3.0, -0.1],
            [0.0, 0.25, 2.5],
        ],
        dtype=jnp.float64,
    )
    inv_diag = jnp.asarray(1.0 / np.diag(np.asarray(a)), dtype=jnp.float64)
    weights = jnp.asarray([0.3, -0.2, 0.7], dtype=jnp.float64)

    def mv(x):
        return a @ x

    def precond(x):
        return inv_diag * x

    def objective(rhs):
        result, _residual = fgmres_solve_with_residual(
            matvec=mv,
            b=rhs,
            preconditioner=precond,
            tol=1.0e-12,
            restart=3,
            maxiter=6,
            precondition_side="right",
        )
        return jnp.vdot(weights, result.x)

    rhs = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    value, pullback = jax.vjp(objective, rhs)
    (grad_rhs,) = pullback(jnp.asarray(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad_rhs)).all()
    assert np.linalg.norm(np.asarray(grad_rhs)) > 0.0


def test_fgmres_cycle_synchronization_preserves_solution() -> None:
    rng = np.random.default_rng(102)
    n = 10
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 5.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    synced, synced_residual = fgmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-10,
        restart=3,
        maxiter=30,
        block_between_cycles=True,
    )
    unsynced, _unsynced_residual = fgmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-10,
        restart=3,
        maxiter=30,
        block_between_cycles=False,
    )

    np.testing.assert_allclose(np.asarray(synced.x), x_ref, rtol=1.0e-7, atol=5.0e-8)
    np.testing.assert_allclose(np.asarray(synced.x), np.asarray(unsynced.x), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(synced_residual), b - a @ np.asarray(synced.x), rtol=1.0e-10, atol=1.0e-10)
    assert float(synced.residual_norm) < 1.0e-6


def test_fgmres_accepts_iteration_dependent_preconditioner() -> None:
    a = np.array(
        [
            [5.0, -1.0, 0.5],
            [2.0, 4.0, -0.25],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)
    calls: list[int] = []

    def mv(x):
        return a_j @ x

    def precond(x, iteration: int):
        calls.append(int(iteration))
        # Vary the smoothing weight to exercise the flexible-preconditioner path.
        omega = 0.85 + 0.05 * (int(iteration) % 3)
        return omega * inv_diag * x

    result, _residual = fgmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-12,
        restart=3,
        maxiter=12,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    assert calls
    assert calls == sorted(calls)
    assert int(result.n_iterations) >= 1
    assert int(result.n_restarts) >= 0


def test_fgmres_jit_matches_numpy_without_host_iteration_state() -> None:
    rng = np.random.default_rng(103)
    n = 6
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 4.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = fgmres_solve_with_residual_jit(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-12,
        restart=n,
        maxiter=n,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-7, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)


def test_fgmres_cycle_jit_matches_numpy_with_bounded_restart() -> None:
    rng = np.random.default_rng(104)
    n = 7
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 5.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-11,
        restart=4,
        maxiter=24,
        precondition_side="right",
        outer_k=2,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-7, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)
    assert int(result.n_iterations) <= 24
    assert np.asarray(result.residual_history).size <= 16


def test_fgmres_cycle_jit_fixed_augmentation_removes_restart_slow_mode() -> None:
    a = jnp.diag(jnp.asarray([1.0e-4, 2.0, 3.0, 4.0], dtype=jnp.float64))
    b = jnp.asarray([1.0, 1.0, -0.5, 0.25], dtype=jnp.float64)
    slow_mode = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64).reshape((4, 1))
    action = a @ slow_mode

    def mv(x):
        return a @ x

    baseline, _baseline_residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b,
        tol=1.0e-14,
        restart=1,
        maxiter=1,
        precondition_side="none",
    )
    augmented, augmented_residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b,
        augmentation_basis=slow_mode,
        operator_on_augmentation=action,
        tol=1.0e-14,
        restart=1,
        maxiter=1,
        precondition_side="none",
    )

    assert float(augmented.residual_norm) < 0.75 * float(baseline.residual_norm)
    np.testing.assert_allclose(float(augmented.x[0]), 1.0e4, rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(augmented_residual), np.asarray(b - mv(augmented.x)), atol=1.0e-12)
    history = np.asarray(augmented.residual_history)
    assert history[1] < history[0]


def test_fgmres_cycle_jit_combined_augmentation_couples_coarse_and_krylov_spaces() -> None:
    a = jnp.asarray(
        [
            [1.0, 10.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    coarse = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64).reshape((3, 1))
    coarse_action = a @ coarse

    def mv(x):
        return a @ x

    projected, _projected_residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b,
        augmentation_basis=coarse,
        operator_on_augmentation=coarse_action,
        augmentation_mode="projected",
        tol=1.0e-14,
        restart=1,
        maxiter=1,
        precondition_side="none",
    )
    combined, combined_residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b,
        augmentation_basis=coarse,
        operator_on_augmentation=coarse_action,
        augmentation_mode="combined",
        tol=1.0e-14,
        restart=1,
        maxiter=1,
        precondition_side="none",
    )

    assert float(projected.residual_norm) > 0.9
    assert float(combined.residual_norm) < 1.0e-11
    np.testing.assert_allclose(np.asarray(combined.x), np.asarray([-10.0, 1.0, 0.0]), atol=1.0e-11)
    np.testing.assert_allclose(np.asarray(combined_residual), np.asarray(b - mv(combined.x)), atol=1.0e-12)


def test_fgmres_full_jit_fixed_augmentation_is_trace_safe() -> None:
    a = jnp.diag(jnp.asarray([1.0e-5, 1.5, 2.0], dtype=jnp.float64))
    slow_mode = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64).reshape((3, 1))
    action = a @ slow_mode

    def solve_objective(rhs):
        result, _residual = fgmres_solve_with_residual_jit(
            matvec=lambda x: a @ x,
            b=rhs,
            augmentation_basis=slow_mode,
            operator_on_augmentation=action,
            tol=1.0e-13,
            restart=1,
            maxiter=2,
            precondition_side="none",
        )
        return result.residual_norm + 1.0e-8 * jnp.vdot(result.x, result.x)

    rhs = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)
    value, grad = jax.value_and_grad(solve_objective)(rhs)

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad)).all()
    assert np.linalg.norm(np.asarray(grad)) > 0.0


def test_fgmres_cycle_jit_reports_progress_at_restart_boundaries() -> None:
    a = jnp.asarray(
        [
            [4.0, -1.0, 0.0],
            [0.5, 3.0, -0.25],
            [0.0, 0.25, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    events: list[dict[str, float | int]] = []

    def mv(x):
        return a @ x

    result, _residual = fgmres_cycle_jit_solve_with_residual(
        matvec=mv,
        b=b,
        tol=1.0e-14,
        restart=1,
        maxiter=3,
        precondition_side="none",
        progress_callback=lambda **event: events.append(event),
    )

    assert events
    assert events[0]["cycle"] == 1
    assert events[0]["iterations"] == 1
    assert events[-1]["iterations"] <= 3
    assert all(np.isfinite(float(event["residual_norm"])) for event in events)
    assert all(float(event["target"]) > 0.0 for event in events)
    assert np.isfinite(float(result.residual_norm))


def test_fgmres_reports_first_converged_iteration_for_identity() -> None:
    b = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    result, residual = fgmres_solve_with_residual(
        matvec=lambda x: x,
        b=b,
        preconditioner=None,
        tol=1.0e-12,
        restart=3,
        maxiter=9,
        precondition_side="none",
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(b), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(residual), np.zeros(3), rtol=1.0e-12, atol=1.0e-12)
    assert bool(result.converged)
    assert int(result.n_iterations) == 1


def test_fgmres_left_preconditioning_matches_numpy() -> None:
    a = np.array(
        [
            [5.0, -1.0, 0.5, 0.0],
            [2.0, 4.0, -0.25, 0.5],
            [0.0, 1.0, 3.0, -1.0],
            [0.25, 0.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = fgmres_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-12,
        restart=4,
        maxiter=8,
        precondition_side="left",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)


def test_bicgstab_solve_with_residual_matches_matvec() -> None:
    rng = np.random.default_rng(23)
    n = 14
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.6 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result, residual = bicgstab_solve_with_residual(matvec=mv, b=b_j, tol=1e-10, maxiter=400)
    r_expected = b_j - mv(result.x)

    np.testing.assert_allclose(np.asarray(residual), np.asarray(r_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        float(result.residual_norm),
        float(jnp.linalg.norm(residual)),
        rtol=1e-12,
        atol=1e-12,
    )
    assert int(result.n_iterations) >= 1
    history = np.asarray(result.residual_history)
    assert np.isfinite(history[: int(result.n_iterations) + 1]).all()
    assert bool(result.converged)


def test_bicgstab_solve_wrapper_matches_numpy_for_diagonal_system() -> None:
    diagonal = jnp.asarray([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)
    b = jnp.asarray([1.0, -2.0, 4.0, 0.5], dtype=jnp.float64)

    result = bicgstab_solve(
        matvec=lambda x: diagonal * x,
        b=b,
        tol=1.0e-12,
        maxiter=20,
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(b / diagonal), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-10
    assert bool(result.converged)


def test_bicgstab_right_preconditioning_preserves_physical_initial_guess() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.25],
            [1.5, 3.0, -0.5],
            [0.0, 0.75, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    x0 = x_ref + np.array([0.25, -0.1, 0.05], dtype=np.float64)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = bicgstab_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        x0=jnp.asarray(x0, dtype=jnp.float64),
        tol=1.0e-12,
        maxiter=40,
        precondition_side="right",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-10
    assert int(result.n_iterations) >= 1
    assert bool(result.converged)


def test_tfqmr_solve_with_residual_matches_numpy_for_nonsymmetric_matrix() -> None:
    rng = np.random.default_rng(121)
    n = 14
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 7.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = tfqmr_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        tol=1.0e-12,
        maxiter=80,
        precondition_side="left",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-8
    assert bool(result.converged)


def test_tfqmr_solve_wrapper_matches_numpy_for_diagonal_system() -> None:
    diagonal = jnp.asarray([2.5, 4.0, 6.0, 9.0], dtype=jnp.float64)
    b = jnp.asarray([1.5, -1.0, 3.0, -4.5], dtype=jnp.float64)

    result = tfqmr_solve(
        matvec=lambda x: diagonal * x,
        b=b,
        tol=1.0e-12,
        maxiter=20,
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(b / diagonal), rtol=1.0e-10, atol=1.0e-10)
    assert float(result.residual_norm) < 1.0e-10
    assert bool(result.converged)


def test_tfqmr_right_preconditioning_preserves_physical_initial_guess() -> None:
    a = np.array(
        [
            [5.0, -1.0, 0.5, 0.0],
            [2.0, 4.0, -0.25, 0.5],
            [0.0, 1.0, 3.0, -1.0],
            [0.25, 0.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    x0 = np.array([0.05, -0.02, 0.03, 0.01], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)
    x0_j = jnp.asarray(x0)
    inv_diag = jnp.asarray(1.0 / np.diag(a), dtype=jnp.float64)

    def mv(x):
        return a_j @ x

    def precond(x):
        return inv_diag * x

    result, residual = tfqmr_solve_with_residual(
        matvec=mv,
        b=b_j,
        preconditioner=precond,
        x0=x0_j,
        tol=1.0e-12,
        maxiter=80,
        precondition_side="right",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)


def test_tfqmr_jit_matches_numpy() -> None:
    a = jnp.asarray(
        [
            [4.0, -1.0, 0.5],
            [0.25, 3.5, -0.75],
            [0.0, 1.0, 2.5],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    x_ref = np.linalg.solve(np.asarray(a), np.asarray(b))

    def mv(x):
        return a @ x

    result, residual = tfqmr_solve_with_residual_jit(
        matvec=mv,
        b=b,
        tol=1.0e-12,
        maxiter=30,
        precondition_side="none",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), np.zeros(3), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)


def test_tfqmr_residual_replacement_preserves_solution() -> None:
    rng = np.random.default_rng(122)
    n = 8
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 6.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    result, residual = tfqmr_solve_with_residual(
        matvec=lambda x: a_j @ x,
        b=jnp.asarray(b),
        tol=1.0e-9,
        maxiter=60,
        precondition_side="none",
        residual_replacement_interval=3,
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)
    assert bool(result.converged)


def test_gmres_wrapper_accepts_tfqmr_solve_method() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.5],
            [0.25, 3.5, -0.75],
            [0.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    result, residual = gmres_solve_with_residual(
        matvec=lambda x: a_j @ x,
        b=jnp.asarray(b),
        tol=1.0e-12,
        maxiter=30,
        solve_method="tfqmr-jax",
        precondition_side="none",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), b - a @ np.asarray(result.x), rtol=1.0e-10, atol=1.0e-10)


def test_bicgstab_rejects_explosive_preconditioned_steps() -> None:
    a = jnp.asarray([[1.0, 2.0], [3.0, 4.1]], dtype=jnp.float64)
    b = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    def mv(x):
        return a @ x

    def pathological_preconditioner(x):
        return jnp.asarray([1.0e120 * x[0], 1.0e-120 * x[1]], dtype=x.dtype)

    result, residual = bicgstab_solve_with_residual(
        matvec=mv,
        b=b,
        preconditioner=pathological_preconditioner,
        tol=1.0e-12,
        maxiter=10,
        precondition_side="right",
    )

    history = np.asarray(result.residual_history)
    assert np.isfinite(float(result.residual_norm))
    assert np.isfinite(np.asarray(residual)).all()
    assert np.nanmax(history) < 10.0 * float(jnp.linalg.norm(b))
    assert not bool(result.converged)


def test_gmres_solve_jit_bicgstab_method_matches_numpy() -> None:
    rng = np.random.default_rng(231)
    n = 12
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.4 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result = gmres_solve_jit(
        matvec=mv,
        b=b_j,
        tol=1e-12,
        restart=30,
        maxiter=200,
        solve_method="bicgstab",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1e-8, atol=1e-8)
    assert float(result.residual_norm) < 1e-8


def test_explicit_left_preconditioned_gmres_scipy_matches_numpy() -> None:
    a = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, -1.0],
            [0.0, -1.0, 2.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    inv_diag = 1.0 / np.diag(a)

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        tol=1e-12,
        restart=6,
        maxiter=20,
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-10)
    assert rn_true < 1e-10
    assert rn_pc < 1e-10
    assert history


def test_gmres_solve_with_history_scipy_progress_matches_history() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.25],
            [0.5, 3.0, -0.5],
            [0.0, 0.75, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)
    events: list[tuple[int, float]] = []

    def mv(x):
        return a_j @ x

    x, rn, history = gmres_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        tol=1.0e-12,
        restart=3,
        maxiter=8,
        progress_callback=lambda iteration, residual: events.append((iteration, residual)),
    )

    np.testing.assert_allclose(x, x_ref, rtol=1.0e-10, atol=1.0e-10)
    assert rn < 1.0e-10
    assert events
    assert [iteration for iteration, _residual in events] == list(range(1, len(history) + 1))
    np.testing.assert_allclose([residual for _iteration, residual in events], history, rtol=0.0, atol=0.0)


def test_gmres_solve_with_history_scipy_right_preconditioned_x0_is_physical() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.25],
            [1.5, 3.0, -0.5],
            [0.0, 0.75, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    x0 = x_ref + np.array([0.25, -0.1, 0.05], dtype=np.float64)
    inv_diag = 1.0 / np.diag(a)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn, history = gmres_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        x0=jnp.asarray(x0, dtype=jnp.float64),
        tol=1.0e-12,
        restart=3,
        maxiter=8,
        precondition_side="right",
    )

    np.testing.assert_allclose(x, x_ref, rtol=1.0e-10, atol=1.0e-10)
    assert rn < 1.0e-10
    assert history


def test_explicit_left_preconditioned_gmres_scipy_progress_matches_history() -> None:
    a = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, -1.0],
            [0.0, -1.0, 2.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    a_j = jnp.asarray(a)
    inv_diag = 1.0 / np.diag(a)
    events: list[tuple[int, float]] = []

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        tol=1.0e-12,
        restart=6,
        maxiter=20,
        progress_callback=lambda iteration, residual: events.append((iteration, residual)),
    )

    np.testing.assert_allclose(a @ x, b, rtol=1.0e-10, atol=1.0e-10)
    assert rn_true < 1.0e-10
    assert rn_pc < 1.0e-10
    assert events
    assert [iteration for iteration, _residual in events] == list(range(1, len(history) + 1))
    np.testing.assert_allclose([residual for _iteration, residual in events], history, rtol=0.0, atol=0.0)


def test_explicit_left_preconditioned_gmres_scipy_zero_preconditioned_rhs_is_finite() -> None:
    a = np.eye(3, dtype=np.float64)
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def zero_precond(x):
        return jnp.zeros_like(x)

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=zero_precond,
        tol=1e-12,
        restart=4,
        maxiter=4,
    )

    np.testing.assert_allclose(x, np.zeros_like(b), rtol=0.0, atol=0.0)
    assert rn_true == pytest.approx(np.linalg.norm(b))
    assert rn_pc == 0.0
    assert history == [0.0]


def test_lgmres_solve_with_history_matches_numpy_on_nonsymmetric_matrix() -> None:
    rng = np.random.default_rng(29)
    n = 15
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += 5.0 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    x, rn, history = lgmres_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        tol=1e-12,
        restart=7,
        maxiter=30,
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-8)
    assert rn < 1e-8
    assert history


def test_lgmres_right_preconditioning_matches_true_solution() -> None:
    a = np.array(
        [
            [4.0, -2.0, 1.0],
            [1.0, 3.5, -0.5],
            [0.0, 2.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    inv_diag = 1.0 / np.diag(a)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    result = gmres_solve(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        tol=1e-12,
        restart=6,
        maxiter=20,
        solve_method="lgmres",
        precondition_side="right",
    )

    np.testing.assert_allclose(np.asarray(result.x), x_ref, rtol=1e-10, atol=1e-10)
    assert float(result.residual_norm) < 1e-10


def test_distributed_solver_kind_auto_defaults_to_bicgstab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", raising=False)
    kind, method = _distributed_solver_kind("auto")
    assert kind == "bicgstab"
    assert method == "batched"


def test_distributed_solver_kind_auto_can_force_gmres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", "gmres")
    kind, method = _distributed_solver_kind("auto")
    assert kind == "gmres"
    assert method == "incremental"


def test_distributed_solver_kind_rejects_host_only_lgmres() -> None:
    with pytest.raises(ValueError, match="host-only"):
        _distributed_solver_kind("lgmres")


def test_gmres_solve_jit_rejects_host_only_lgmres() -> None:
    a = jnp.asarray([[2.0, 1.0], [0.0, 1.5]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    def mv(x):
        return a @ x

    with pytest.raises(ValueError, match="host-only"):
        gmres_solve_jit(
            matvec=mv,
            b=b,
            tol=1e-12,
            restart=4,
            maxiter=4,
            solve_method="lgmres",
        )


def test_rhs_krylov_method_for_context_keeps_lgmres_on_plain_host_path() -> None:
    assert _rhs_krylov_method_for_context(
        gmres_method="lgmres",
        use_implicit=False,
        distributed_axis=None,
        solver_jit=False,
    ) == "lgmres"


@pytest.mark.parametrize(
    ("use_implicit", "distributed_axis", "solver_jit"),
    [
        (True, None, False),
        (False, "theta", False),
        (False, None, True),
    ],
)
def test_rhs_krylov_method_for_context_downgrades_host_only_lgmres(
    use_implicit: bool,
    distributed_axis: str | None,
    solver_jit: bool,
) -> None:
    assert _rhs_krylov_method_for_context(
        gmres_method="lgmres",
        use_implicit=use_implicit,
        distributed_axis=distributed_axis,
        solver_jit=solver_jit,
    ) == "incremental"


def test_materialize_distributed_input_preserves_values_and_dtype() -> None:
    arr = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    out = _materialize_distributed_input(arr, dtype=jnp.float64)
    assert out is not None
    np.testing.assert_allclose(np.asarray(out), np.asarray(arr), rtol=0.0, atol=0.0)
    assert out.dtype == jnp.float64


def test_distributed_solver_pjit_factories_reuse_wrappers() -> None:
    if solver_module._pjit is None:  # pragma: no cover - depends on optional JAX internals
        pytest.skip("sharded JIT unavailable")
    solver_module._get_distributed_solve_pjit.cache_clear()
    solver_module._get_distributed_solve_with_residual_pjit.cache_clear()

    assert solver_module._get_distributed_solve_pjit("p") is solver_module._get_distributed_solve_pjit("p")
    assert solver_module._get_distributed_solve_with_residual_pjit(
        "p"
    ) is solver_module._get_distributed_solve_with_residual_pjit("p")


def test_distributed_gmres_wrappers_fall_back_to_host_without_axis() -> None:
    a = jnp.asarray(
        [
            [3.0, 0.5, 0.0],
            [0.25, 4.0, -0.5],
            [0.0, 1.0, 2.5],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    ref = np.linalg.solve(np.asarray(a), np.asarray(b))

    def mv(x):
        return a @ x

    result = gmres_solve_distributed(
        matvec=mv,
        b=b,
        axis_name=None,
        tol=1.0e-12,
        restart=4,
        maxiter=20,
    )
    result_with_residual, residual = gmres_solve_with_residual_distributed(
        matvec=mv,
        b=b,
        axis_name=None,
        tol=1.0e-12,
        restart=4,
        maxiter=20,
    )

    np.testing.assert_allclose(np.asarray(result.x), ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(result_with_residual.x), ref, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(residual), np.asarray(b - mv(result_with_residual.x)), rtol=1.0e-10)


def test_distributed_solver_sharded_jit_smoke_two_cpu_devices() -> None:
    code = r"""
import numpy as np
import jax
import jax.numpy as jnp
from sfincs_jax.solver import (
    distributed_gmres_enabled,
    gmres_solve_distributed,
    gmres_solve_with_residual_distributed,
)

assert len(jax.local_devices()) == 2
assert distributed_gmres_enabled()

A = jnp.asarray(
    [
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.5, 1.0],
        [0.0, 0.0, 1.0, 2.0],
    ],
    dtype=jnp.float64,
)
b = jnp.asarray([1.0, 2.0, 0.5, -1.0], dtype=jnp.float64)
ref = np.linalg.solve(np.asarray(A), np.asarray(b))

def mv(x):
    return A @ x

res = gmres_solve_distributed(matvec=mv, b=b, axis_name="p", tol=1e-12, restart=4, maxiter=16)
res.x.block_until_ready()
np.testing.assert_allclose(np.asarray(res.x), ref, rtol=1e-8, atol=1e-8)

res2, r = gmres_solve_with_residual_distributed(
    matvec=mv,
    b=b,
    axis_name="p",
    tol=1e-12,
    restart=4,
    maxiter=16,
)
r.block_until_ready()
np.testing.assert_allclose(np.asarray(res2.x), ref, rtol=1e-8, atol=1e-8)
np.testing.assert_allclose(np.asarray(r), np.asarray(b - mv(res2.x)), rtol=1e-8, atol=1e-8)
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_ENABLE_X64"] = "True"
    env["SFINCS_JAX_GMRES_DISTRIBUTED"] = "on"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
