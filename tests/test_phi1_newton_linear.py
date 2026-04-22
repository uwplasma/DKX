from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.phi1_newton_linear import (
    build_phi1_newton_preconditioner,
    solve_phi1_newton_linear_step,
)
from sfincs_jax.solver import GMRESSolveResult


def test_build_phi1_newton_preconditioner_collision_and_block_modes() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def collision_builder(**kwargs):
        calls.append(("collision", kwargs))
        return "collision_precond"

    def block_builder(**kwargs):
        calls.append(("block", kwargs))
        return "block_precond"

    reduce_full = lambda x: x  # noqa: E731
    expand_reduced = lambda x: x  # noqa: E731

    precond = build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        rhs_mode=1,
        include_phi1=True,
        use_active_dof_mode=True,
        op="op",
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        preconditioner_options={"PRECONDITIONER_SPECIES": "2", "PRECONDITIONER_X": "3", "PRECONDITIONER_XI": "4"},
        collision_builder=collision_builder,
        block_builder=block_builder,
    )
    assert precond == "collision_precond"
    assert calls[-1][0] == "collision"
    assert calls[-1][1]["reduce_full"] is reduce_full
    assert calls[-1][1]["expand_reduced"] is expand_reduced

    precond = build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        rhs_mode=1,
        include_phi1=False,
        use_active_dof_mode=False,
        op="op",
        reduce_full=None,
        expand_reduced=None,
        preconditioner_options={"PRECONDITIONER_SPECIES": "2", "PRECONDITIONER_X": "3", "PRECONDITIONER_XI": "4"},
        collision_builder=collision_builder,
        block_builder=block_builder,
    )
    assert precond == "block_precond"
    assert calls[-1][0] == "block"
    assert calls[-1][1]["preconditioner_species"] == 2
    assert calls[-1][1]["preconditioner_x"] == 3
    assert calls[-1][1]["preconditioner_xi"] == 4


def test_solve_phi1_newton_linear_step_retries_without_preconditioner() -> None:
    dispatch_calls: list[tuple[object, jnp.ndarray]] = []
    history_calls: list[object] = []
    emitted: list[tuple[int, str]] = []

    def gmres_dispatch(**kwargs):
        dispatch_calls.append((kwargs["preconditioner"], kwargs["b"]))
        if kwargs["preconditioner"] is not None:
            return GMRESSolveResult(
                x=jnp.array([jnp.nan, 0.0]),
                residual_norm=jnp.array(jnp.nan),
            )
        return GMRESSolveResult(
            x=jnp.array([1.0, 2.0]),
            residual_norm=jnp.array(3.0),
        )

    def emit_ksp_history(**kwargs):
        history_calls.append(kwargs["precond_fn"])

    lin, step_vec, linear_resid_norm = solve_phi1_newton_linear_step(
        use_active_dof_mode=False,
        solve_method_linear="batched",
        matvec=lambda x: x,
        residual_vec=jnp.array([-1.0, -2.0]),
        preconditioner="P",
        gmres_tol=1e-10,
        gmres_restart=80,
        gmres_maxiter=300,
        sparse_direct_solve=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sparse direct")),
        gmres_dispatch=gmres_dispatch,
        gmres_result_is_finite=lambda res: bool(np.isfinite(float(res.residual_norm))),
        emit_ksp_history=emit_ksp_history,
        emit=lambda lvl, msg: emitted.append((lvl, msg)),
        newton_iter=2,
        total_size=2,
    )

    assert len(dispatch_calls) == 2
    assert dispatch_calls[0][0] == "P"
    assert dispatch_calls[1][0] is None
    assert history_calls == ["P", None]
    assert np.allclose(np.asarray(lin.x), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(step_vec), np.array([1.0, 2.0]))
    assert float(linear_resid_norm) == 3.0
    assert any("retrying without preconditioner" in msg for _, msg in emitted)


def test_solve_phi1_newton_linear_step_reduced_sparse_direct_expands_and_recomputes_residual() -> None:
    sparse_calls: list[dict[str, object]] = []

    def sparse_direct_solve(**kwargs):
        sparse_calls.append(kwargs)
        return GMRESSolveResult(
            x=jnp.array([1.0, 2.0]),
            residual_norm=jnp.array(99.0),
        )

    def reduce_full(v):
        return jnp.asarray([v[0], v[2]])

    def expand_reduced(v):
        return jnp.asarray([v[0], 0.0, v[1], 0.0])

    lin, step_vec, linear_resid_norm = solve_phi1_newton_linear_step(
        use_active_dof_mode=True,
        solve_method_linear="sparse_direct",
        matvec=lambda x: x,
        residual_vec=jnp.array([-1.0, 0.0, -2.0, 0.0]),
        preconditioner=None,
        gmres_tol=1e-10,
        gmres_restart=80,
        gmres_maxiter=300,
        sparse_direct_solve=sparse_direct_solve,
        gmres_dispatch=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected gmres dispatch")),
        gmres_result_is_finite=lambda res: True,
        emit_ksp_history=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected history emission")),
        emit=None,
        newton_iter=1,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_size=2,
    )

    assert sparse_calls
    assert sparse_calls[0]["cache_tag"] == ("reduced", 1, 2)
    assert np.allclose(np.asarray(lin.x), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(step_vec), np.array([1.0, 0.0, 2.0, 0.0]))
    assert float(linear_resid_norm) == 0.0
