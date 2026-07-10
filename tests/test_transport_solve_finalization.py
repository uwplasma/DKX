from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.krylov import GMRESSolveResult
from sfincs_jax.problems.transport_finalize import (
    TransportKSPIterationRequest,
    TransportRHSFinalizationContext,
    finalize_full_transport_rhs,
    finalize_reduced_transport_rhs,
)


class _Recycle:
    def __init__(self) -> None:
        self.full: list[tuple[jnp.ndarray, jnp.ndarray]] = []
        self.reduced: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []

    def append_full(self, x_full, ax_full) -> None:
        self.full.append((x_full, ax_full))

    def append_reduced(self, x_reduced, ax_reduced, *, x_full, ax_full) -> None:
        self.reduced.append((x_reduced, ax_reduced, x_full, ax_full))


def _context(*, recycle=None, stream=True, store=True, iter_stats=True):
    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    solver_kinds: dict[int, str] = {}
    solve_methods: dict[int, str] = {}
    collected: list[tuple[int, jnp.ndarray]] = []
    stats_calls: list[dict[str, object]] = []

    def apply_operator(op, x):
        return op.scale * x

    def emit_iteration_stats(**kwargs):
        stats_calls.append(dict(kwargs))

    context = TransportRHSFinalizationContext(
        state_vectors=state_vectors,
        residual_norms=residual_norms,
        solver_kinds_by_rhs=solver_kinds,
        solve_methods_by_rhs=solve_methods,
        store_state_vectors=store,
        stream_diagnostics=stream,
        collect_transport_outputs=lambda which_rhs, x: collected.append((which_rhs, x)),
        recycle_state=recycle,
        apply_operator=apply_operator,
        emit_iteration_stats=emit_iteration_stats,
        emit=None,
        iter_stats_enabled=iter_stats,
        iter_stats_max_size=99,
        atol=1.0e-12,
        maxiter=11,
        precond_side="right",
    )
    return context, state_vectors, residual_norms, solver_kinds, solve_methods, collected, stats_calls


def _ksp_request():
    return TransportKSPIterationRequest(
        matvec_fn=lambda x: 2.0 * x,
        b_vec=jnp.asarray([1.0, 2.0]),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=7,
        maxiter_val=11,
        precond_side="right",
        solver_kind="gmres",
    )


def test_finalize_full_transport_rhs_reuses_supplied_residual_vector() -> None:
    recycle = _Recycle()
    context, state, residuals, solver_kinds, solve_methods, collected, stats = _context(recycle=recycle)
    op = SimpleNamespace(scale=999.0)
    rhs = jnp.asarray([4.0, 9.0])
    residual_vec = jnp.asarray([0.5, -0.25])
    result = GMRESSolveResult(x=jnp.asarray([1.0, 2.0]), residual_norm=jnp.asarray(0.75))

    finalized = finalize_full_transport_rhs(
        context=context,
        which_rhs=2,
        result=result,
        rhs_full=rhs,
        op_matvec=op,
        solver_kind="gmres",
        solve_method="incremental",
        dense_used=False,
        projection_needed=False,
        residual_vec=residual_vec,
        maybe_project_constraint_nullspace=lambda x, **_: x + 100.0,
        ksp_request=_ksp_request(),
    )

    np.testing.assert_allclose(np.asarray(finalized.x_full), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(finalized.ax_full), np.asarray([3.5, 9.25]))
    assert float(residuals[2]) == 0.75
    np.testing.assert_allclose(np.asarray(state[2]), np.asarray([1.0, 2.0]))
    assert solver_kinds[2] == "gmres"
    assert solve_methods[2] == "incremental"
    assert collected[0][0] == 2
    assert len(recycle.full) == 1
    assert stats and stats[0]["which_rhs"] == 2


def test_finalize_full_transport_rhs_recomputes_after_projection() -> None:
    context, state, residuals, *_ = _context(recycle=None)
    op = SimpleNamespace(scale=3.0)
    rhs = jnp.asarray([1.0, 2.0])
    result = GMRESSolveResult(x=jnp.asarray([1.0, 1.0]), residual_norm=jnp.asarray(123.0))

    finalized = finalize_full_transport_rhs(
        context=context,
        which_rhs=3,
        result=result,
        rhs_full=rhs,
        op_matvec=op,
        solver_kind="gmres",
        solve_method="incremental",
        dense_used=True,
        projection_needed=True,
        residual_vec=jnp.asarray([0.0, 0.0]),
        maybe_project_constraint_nullspace=lambda x, **_: x.at[1].set(2.0),
        ksp_request=_ksp_request(),
    )

    np.testing.assert_allclose(np.asarray(state[3]), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(finalized.ax_full), np.asarray([3.0, 6.0]))
    np.testing.assert_allclose(float(residuals[3]), np.linalg.norm(np.asarray([2.0, 4.0])))


def test_finalize_reduced_transport_rhs_uses_accepted_full_override() -> None:
    recycle = _Recycle()
    context, state, residuals, solver_kinds, solve_methods, collected, stats = _context(recycle=recycle)
    op = SimpleNamespace(scale=5.0)
    rhs = jnp.asarray([10.0, 20.0, 30.0])
    result = GMRESSolveResult(x=jnp.asarray([7.0, 8.0]), residual_norm=jnp.asarray(999.0))
    accepted_x = jnp.asarray([1.0, 0.0, 2.0])
    accepted_ax = jnp.asarray([9.0, 20.0, 33.0])
    accepted_residual = jnp.asarray(4.0)

    finalized = finalize_reduced_transport_rhs(
        context=context,
        which_rhs=1,
        result=result,
        rhs_full=rhs,
        op_matvec=op,
        solver_kind="dense",
        solve_method="dense",
        dense_used=True,
        expand_reduced=lambda x: jnp.asarray([x[0], 0.0, x[1]]),
        reduce_full=lambda x: x[jnp.asarray([0, 2])],
        maybe_project_constraint_nullspace=lambda x, **_: x + 1000.0,
        ksp_request=_ksp_request(),
        accepted_x_full=accepted_x,
        accepted_ax_full=accepted_ax,
        accepted_residual_norm=accepted_residual,
    )

    np.testing.assert_allclose(np.asarray(finalized.x_full), np.asarray(accepted_x))
    np.testing.assert_allclose(np.asarray(state[1]), np.asarray(accepted_x))
    assert float(residuals[1]) == 4.0
    assert solver_kinds[1] == "dense"
    assert solve_methods[1] == "dense"
    assert collected[0][0] == 1
    np.testing.assert_allclose(np.asarray(recycle.reduced[0][0]), np.asarray([7.0, 8.0]))
    np.testing.assert_allclose(np.asarray(recycle.reduced[0][1]), np.asarray([9.0, 33.0]))
    assert not stats


def test_finalize_reduced_transport_rhs_computes_true_residual_without_override() -> None:
    context, _state, residuals, *_rest, stats = _context(recycle=None, stream=False, store=False)
    op = SimpleNamespace(scale=2.0)
    rhs = jnp.asarray([2.0, 3.0, 8.0])
    result = GMRESSolveResult(x=jnp.asarray([1.0, 4.0]), residual_norm=jnp.asarray(99.0))

    finalized = finalize_reduced_transport_rhs(
        context=context,
        which_rhs=4,
        result=result,
        rhs_full=rhs,
        op_matvec=op,
        solver_kind="gmres",
        solve_method="incremental",
        dense_used=False,
        expand_reduced=lambda x: jnp.asarray([x[0], 0.0, x[1]]),
        reduce_full=lambda x: x[jnp.asarray([0, 2])],
        maybe_project_constraint_nullspace=lambda x, **_: x,
        ksp_request=_ksp_request(),
    )

    np.testing.assert_allclose(np.asarray(finalized.x_full), np.asarray([1.0, 0.0, 4.0]))
    np.testing.assert_allclose(np.asarray(finalized.ax_full), np.asarray([2.0, 0.0, 8.0]))
    np.testing.assert_allclose(float(residuals[4]), 3.0)
    assert stats and stats[0]["which_rhs"] == 4
