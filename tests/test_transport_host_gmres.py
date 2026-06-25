from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.transport_matrix.host_gmres import transport_host_gmres_solve


def _op(*, rhs_mode: int = 2, has_fp: bool = True, n_x: int = 4):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        n_x=n_x,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_transport_host_gmres_solve_without_preconditioner() -> None:
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    def mv(x):
        return a @ x

    result, residual = transport_host_gmres_solve(
        op=_op(),
        matvec_fn=mv,
        b_vec=b,
        x0_vec=None,
        preconditioner_fn=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=20,
        precondition_side_val="left",
    )
    np.testing.assert_allclose(a @ result.x, b, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(residual, b - a @ result.x, rtol=1.0e-12, atol=1.0e-12)
    assert float(result.residual_norm) < 1.0e-10


def test_transport_host_gmres_solve_with_left_preconditioner() -> None:
    a = jnp.asarray([[5.0, 0.0], [0.0, 2.0]], dtype=jnp.float64)
    b = jnp.asarray([10.0, 4.0], dtype=jnp.float64)

    def mv(x):
        return a @ x

    def precond(v):
        return jnp.asarray([v[0] / 5.0, v[1] / 2.0], dtype=jnp.float64)

    result, residual = transport_host_gmres_solve(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        matvec_fn=mv,
        b_vec=b,
        x0_vec=None,
        preconditioner_fn=precond,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=20,
        precondition_side_val="left",
    )
    np.testing.assert_allclose(result.x, jnp.asarray([2.0, 2.0]), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(residual, b - a @ result.x, rtol=1.0e-12, atol=1.0e-12)


def test_transport_host_gmres_reports_progress() -> None:
    a = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    messages: list[tuple[int, str]] = []

    def mv(x):
        return a @ x

    result, _residual = transport_host_gmres_solve(
        op=_op(),
        matvec_fn=mv,
        b_vec=b,
        x0_vec=None,
        preconditioner_fn=None,
        tol_val=1.0e-12,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=20,
        precondition_side_val="left",
        emit=lambda level, message: messages.append((level, message)),
        which_rhs=2,
        progress_every=1,
    )

    np.testing.assert_allclose(a @ result.x, b, rtol=1.0e-10, atol=1.0e-10)
    assert messages
    assert "whichRHS=2" in messages[0][1]
    assert "iter=1" in messages[0][1]
