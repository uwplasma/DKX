from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.profile_phi1_newton import (
    build_phi1_newton_preconditioner,
    solve_phi1_newton_linear_step,
)
from sfincs_jax.solvers.krylov import GMRESSolveResult


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


def test_build_phi1_newton_preconditioner_gates_env_aliases_and_defaults(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    emitted: list[tuple[int, str]] = []

    def collision_builder(**kwargs):
        calls.append(("collision", kwargs))
        return "collision"

    def block_builder(**kwargs):
        calls.append(("block", kwargs))
        return "block"

    common = dict(
        rhs_mode=1,
        include_phi1=True,
        use_active_dof_mode=True,
        op="op",
        reduce_full="reduce",
        expand_reduced="expand",
        preconditioner_options={
            "PRECONDITIONER_SPECIES": "bad",
            "PRECONDITIONER_X": None,
            "PRECONDITIONER_XI": "6",
        },
        collision_builder=collision_builder,
        block_builder=block_builder,
        emit=lambda level, message: emitted.append((level, message)),
    )

    assert build_phi1_newton_preconditioner(
        use_preconditioner=False,
        use_frozen_linearization=True,
        **common,
    ) is None
    assert build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=False,
        **common,
    ) is None
    assert build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        **{**common, "rhs_mode": 2},
    ) is None

    monkeypatch.setenv("SFINCS_JAX_PHI1_PRECOND_KIND", "diag")
    assert build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        **common,
    ) == "collision"
    assert calls[-1][0] == "collision"
    assert calls[-1][1]["reduce_full"] == "reduce"
    assert calls[-1][1]["expand_reduced"] == "expand"

    monkeypatch.setenv("SFINCS_JAX_PHI1_PRECOND_KIND", "point")
    assert build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        **common,
    ) == "block"
    assert calls[-1][0] == "block"
    assert calls[-1][1]["preconditioner_species"] == 1
    assert calls[-1][1]["preconditioner_x"] == 1
    assert calls[-1][1]["preconditioner_xi"] == 6
    assert calls[-1][1]["reduce_full"] == "reduce"
    assert calls[-1][1]["expand_reduced"] == "expand"
    assert any("preconditioner=block" in message for _, message in emitted)

    monkeypatch.setenv("SFINCS_JAX_PHI1_PRECOND_KIND", "unknown")
    assert build_phi1_newton_preconditioner(
        use_preconditioner=True,
        use_frozen_linearization=True,
        **{**common, "include_phi1": False, "use_active_dof_mode": False},
    ) == "block"
    assert calls[-1][0] == "block"


def test_build_phi1_newton_preconditioner_collision_without_active_reduction(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def collision_builder(**kwargs):
        calls.append(kwargs)
        return "collision"

    monkeypatch.delenv("SFINCS_JAX_PHI1_PRECOND_KIND", raising=False)
    assert (
        build_phi1_newton_preconditioner(
            use_preconditioner=True,
            use_frozen_linearization=True,
            rhs_mode=1,
            include_phi1=True,
            use_active_dof_mode=False,
            op="op",
            reduce_full=None,
            expand_reduced=None,
            preconditioner_options={},
            collision_builder=collision_builder,
            block_builder=lambda **kwargs: "unexpected",
        )
        == "collision"
    )
    assert calls == [{"op": "op"}]


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


def test_solve_phi1_newton_linear_step_active_gmres_retries_and_recomputes_true_residual() -> None:
    dispatch_calls: list[object] = []
    history_calls: list[object] = []

    def gmres_dispatch(**kwargs):
        dispatch_calls.append(kwargs["preconditioner"])
        if kwargs["preconditioner"] is not None:
            return GMRESSolveResult(x=jnp.asarray([jnp.nan, 0.0]), residual_norm=jnp.asarray(jnp.nan))
        return GMRESSolveResult(x=jnp.asarray([1.0, 2.0]), residual_norm=jnp.asarray(0.0))

    def reduce_full(v):
        return jnp.asarray([v[0], v[2]])

    def expand_reduced(v):
        return jnp.asarray([v[0], 0.0, v[1], 0.0])

    lin, step_vec, linear_resid_norm = solve_phi1_newton_linear_step(
        use_active_dof_mode=True,
        solve_method_linear="batched",
        matvec=lambda x: 2.0 * x,
        residual_vec=jnp.asarray([-2.0, 0.0, -4.0, 0.0]),
        preconditioner="P",
        gmres_tol=1e-9,
        gmres_restart=5,
        gmres_maxiter=10,
        gmres_dispatch=gmres_dispatch,
        gmres_result_is_finite=lambda res: bool(np.isfinite(float(res.residual_norm))),
        emit_ksp_history=lambda **kwargs: history_calls.append(kwargs["precond_fn"]),
        emit=None,
        newton_iter=4,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        active_size=2,
    )

    assert dispatch_calls == ["P", None]
    assert history_calls == ["P", None]
    np.testing.assert_allclose(np.asarray(lin.x), np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(step_vec), np.array([1.0, 0.0, 2.0, 0.0]))
    assert float(linear_resid_norm) == 0.0


