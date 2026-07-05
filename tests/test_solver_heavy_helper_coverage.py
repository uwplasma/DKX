from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import sfincs_jax.solver as solver
from sfincs_jax.solvers.memory_model import gmres_basis_nbytes
from sfincs_jax.solver import (
    _distributed_krylov_preference,
    _distributed_gmres_axis,
    _materialize_distributed_input,
    _maybe_limit_restart,
    _normalize_krylov_method,
    _preconditioner_accepts_iteration,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_history_scipy,
    distributed_gmres_enabled,
    explicit_left_preconditioned_gmres_scipy,
    gcrotmk_solve_with_history_scipy,
    gmres_solve_with_history_scipy,
    lgmres_solve_with_history_scipy,
)


def test_normalize_krylov_method_and_restart_limit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _normalize_krylov_method("auto") == "bicgstab"
    assert _normalize_krylov_method("DEFAULT") == "bicgstab"
    assert _normalize_krylov_method("lgmres") == "lgmres"

    monkeypatch.delenv("SFINCS_JAX_GMRES_AUTO_RESTART", raising=False)
    monkeypatch.setenv("SFINCS_JAX_GMRES_MAX_MB", "1")
    capped = _maybe_limit_restart(1000, 500, jnp.float64)
    assert capped == 120
    assert gmres_basis_nbytes(1000, capped, dtype=np.float64) <= 1_000_000
    assert gmres_basis_nbytes(1000, capped + 1, dtype=np.float64) > 1_000_000

    monkeypatch.setenv("SFINCS_JAX_GMRES_AUTO_RESTART", "off")
    assert _maybe_limit_restart(1000, 500, jnp.float64) == 500

    monkeypatch.setenv("SFINCS_JAX_GMRES_AUTO_RESTART", "on")
    monkeypatch.setenv("SFINCS_JAX_GMRES_MAX_MB", "bad")
    assert _maybe_limit_restart(0, 500, jnp.float64) == 500
    assert _maybe_limit_restart(1000, 1, jnp.float64) == 1
    assert _maybe_limit_restart(1000, 500, jnp.float64) == 500


def test_materialize_distributed_input_returns_host_array_with_requested_dtype() -> None:
    assert _materialize_distributed_input(None) is None

    arr = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    materialized = _materialize_distributed_input(arr, dtype=jnp.float64)

    assert materialized is not None
    assert materialized.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(materialized), np.asarray([1.0, 2.0, 3.0]), rtol=0.0, atol=0.0)


def test_distributed_axis_and_enablement_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    assert _distributed_gmres_axis() == "theta"

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "on")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "flat")
    assert _distributed_gmres_axis() == "p"

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "off")
    assert _distributed_gmres_axis() is None

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(solver, "_pjit", object())
    monkeypatch.setattr(solver, "PartitionSpec", object())
    monkeypatch.setattr(solver, "_get_gmres_mesh", lambda axis_name: object())
    assert distributed_gmres_enabled() is True

    monkeypatch.setattr(solver, "_get_gmres_mesh", lambda axis_name: None)
    assert distributed_gmres_enabled() is False


def test_distributed_krylov_preference_and_iteration_aware_preconditioner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", raising=False)
    assert _distributed_krylov_preference() == "bicgstab"

    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", "gmres")
    assert _distributed_krylov_preference() == "gmres"

    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", "unexpected")
    assert _distributed_krylov_preference() == "bicgstab"

    def plain_preconditioner(x):
        return x

    def iteration_preconditioner(x, iteration):
        return x / (1.0 + iteration)

    class CallablePreconditioner:
        def __call__(self, *args):
            return args[0]

    assert not _preconditioner_accepts_iteration(None)
    assert not _preconditioner_accepts_iteration(plain_preconditioner)
    assert _preconditioner_accepts_iteration(iteration_preconditioner)
    assert _preconditioner_accepts_iteration(CallablePreconditioner())


def test_dense_matrix_assembly_obeys_block_and_jit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    a = np.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.25, 3.0, -0.75],
            [1.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    monkeypatch.setenv("SFINCS_JAX_DENSE_BLOCK", "2")
    monkeypatch.setenv("SFINCS_JAX_DENSE_ASSEMBLE_JIT", "off")
    blocked = assemble_dense_matrix_from_matvec(matvec=mv, n=3, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(blocked), a, rtol=0.0, atol=1.0e-12)

    monkeypatch.setenv("SFINCS_JAX_DENSE_BLOCK", "bad")
    monkeypatch.setenv("SFINCS_JAX_DENSE_ASSEMBLE_JIT", "on")
    jitted = assemble_dense_matrix_from_matvec(matvec=mv, n=3, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(jitted), a, rtol=0.0, atol=1.0e-12)


def test_gmres_history_scipy_supports_right_preconditioning() -> None:
    a = np.array(
        [
            [5.0, -1.0, 0.5],
            [0.0, 4.0, -1.0],
            [1.0, 0.5, 3.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, 2.0, -1.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
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
        tol=1e-12,
        restart=6,
        maxiter=20,
        precondition_side="right",
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-10)
    assert rn < 1e-10
    assert history


def test_gmres_history_scipy_right_preconditioning_uses_physical_x0() -> None:
    a = np.array(
        [
            [4.0, -1.0, 0.25],
            [0.5, 3.0, -0.75],
            [0.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    inv_diag = 1.0 / np.diag(a)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn, _history = gmres_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        x0=jnp.asarray(x_ref),
        tol=1e-12,
        restart=1,
        maxiter=1,
        precondition_side="right",
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-12, atol=1e-12)
    assert rn < 1e-12


def test_lgmres_and_gcrotmk_right_preconditioning_use_physical_x0(monkeypatch: pytest.MonkeyPatch) -> None:
    a = np.asarray(
        [
            [4.0, -1.0, 0.25],
            [0.5, 3.0, -0.75],
            [0.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    inv_diag = 1.0 / np.diag(a)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    monkeypatch.setenv("SFINCS_JAX_LGMRES_OUTER_K", "bad")
    x_lgmres, rn_lgmres, _history_lgmres = lgmres_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        x0=jnp.asarray(x_ref),
        tol=1e-12,
        restart=1,
        maxiter=1,
        precondition_side="right",
    )
    np.testing.assert_allclose(x_lgmres, x_ref, rtol=1e-12, atol=1e-12)
    assert rn_lgmres < 1e-12

    monkeypatch.setenv("SFINCS_JAX_GCROTMK_OUTER_K", "bad")
    x_gcrot, rn_gcrot, _history_gcrot = gcrotmk_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        x0=jnp.asarray(x_ref),
        tol=1e-12,
        restart=1,
        maxiter=1,
        precondition_side="right",
    )
    np.testing.assert_allclose(x_gcrot, x_ref, rtol=1e-12, atol=1e-12)
    assert rn_gcrot < 1e-12


def test_explicit_left_preconditioned_gmres_handles_zero_preconditioned_rhs() -> None:
    a = jnp.eye(3, dtype=jnp.float64)
    b = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    def zero_preconditioner(x):
        return jnp.zeros_like(x)

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=lambda x: a @ x,
        b=b,
        preconditioner=zero_preconditioner,
        tol=1e-12,
        restart=3,
        maxiter=3,
    )

    np.testing.assert_allclose(x, np.zeros(3), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rn_true, float(np.linalg.norm(np.asarray(b))), rtol=1e-12)
    assert rn_pc == 0.0
    assert history == [0.0]


def test_bicgstab_history_scipy_returns_solution_and_history() -> None:
    a = np.array(
        [
            [3.0, 0.5, 0.0],
            [0.0, 2.5, -0.25],
            [0.1, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    x, rn, history = bicgstab_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        tol=1e-12,
        maxiter=50,
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-10)
    assert rn < 1e-10
    assert history


def test_bicgstab_history_scipy_right_preconditioning_uses_physical_x0() -> None:
    a = np.array(
        [
            [3.0, -0.5, 0.0],
            [0.25, 2.0, -0.25],
            [0.0, 0.5, 2.5],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, 0.25, -0.75], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    inv_diag = 1.0 / np.diag(a)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn, _history = bicgstab_solve_with_history_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        x0=jnp.asarray(x_ref),
        tol=1e-12,
        maxiter=1,
        precondition_side="right",
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-12, atol=1e-12)
    assert rn < 1e-12
