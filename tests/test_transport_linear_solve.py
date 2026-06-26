from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import jax.numpy as jnp

import sfincs_jax.problems.transport_solve as transport_linear
from sfincs_jax.problems.transport_solve import (
    TransportLinearSolveCallbacks,
    TransportLinearSolveContext,
    solve_transport_linear,
    solve_transport_linear_with_residual,
    transport_restart_for_method,
    transport_solver_kind,
)


def _context(**overrides) -> TransportLinearSolveContext:
    values = {
        "rhs_mode": 2,
        "size_hint": 5,
        "use_implicit": False,
        "use_solver_jit": False,
        "distributed_axis": None,
    }
    values.update(overrides)
    return TransportLinearSolveContext(**values)


def test_transport_solver_kind_and_restart_policy() -> None:
    assert transport_solver_kind("auto", rhs_mode=2) == ("bicgstab", "batched")
    assert transport_solver_kind("bicgstab_jax", rhs_mode=3) == ("bicgstab", "batched")
    assert transport_solver_kind("incremental", rhs_mode=2) == ("gmres", "incremental")

    assert transport_restart_for_method("auto", rhs_mode=2, gmres_restart=30, restart=80) == 80
    assert transport_restart_for_method("incremental", rhs_mode=2, gmres_restart=30, restart=80) == 30


def test_solve_transport_linear_uses_nonjit_or_jit_gmres(monkeypatch) -> None:
    calls: list[str] = []

    def fake_gmres(**kwargs):
        calls.append(f"gmres:{kwargs['solve_method']}")
        return "plain"

    def fake_gmres_jit(**kwargs):
        calls.append(f"jit:{kwargs['solve_method']}")
        return "jit"

    monkeypatch.setattr(transport_linear, "gmres_solve", fake_gmres)
    monkeypatch.setattr(transport_linear, "gmres_solve_jit", fake_gmres_jit)
    args = dict(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    assert solve_transport_linear(context=_context(use_solver_jit=False), **args) == "plain"
    assert solve_transport_linear(context=_context(use_solver_jit=True), **args) == "jit"
    assert calls == ["gmres:incremental", "jit:incremental"]


def test_transport_linear_solve_callbacks_bind_context(monkeypatch) -> None:
    captured = {}

    def fake_gmres(**kwargs):
        captured.update(kwargs)
        return "bound"

    monkeypatch.setattr(transport_linear, "gmres_solve", fake_gmres)
    callbacks = TransportLinearSolveCallbacks(context=_context(use_solver_jit=False))

    result = callbacks.solve(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert result == "bound"
    assert captured["solve_method"] == "incremental"
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_implicit_routes_to_custom_solve(monkeypatch) -> None:
    captured = {}

    def fake_custom(**kwargs):
        captured.update(kwargs)
        return "implicit"

    monkeypatch.setattr(transport_linear, "linear_custom_solve", fake_custom)

    assert (
        solve_transport_linear(
            context=_context(use_implicit=True, size_hint=17),
            matvec_fn=lambda x: x,
            b_vec=jnp.ones((2,)),
            x0_vec=None,
            tol_val=1e-8,
            atol_val=1e-12,
            restart_val=10,
            maxiter_val=20,
            solve_method_val="auto",
            preconditioner_val=None,
            precondition_side_val="right",
        )
        == "implicit"
    )
    assert captured["solver"] == "bicgstab"
    assert captured["solve_method"] == "batched"
    assert captured["size_hint"] == 17
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_with_residual_bicgstab_route(monkeypatch) -> None:
    captured = {}

    def fake_bicgstab(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(transport_linear, "bicgstab_solve_with_residual", fake_bicgstab)
    result, residual = solve_transport_linear_with_residual(
        context=_context(use_solver_jit=False),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="left",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["maxiter"] == 20
    assert "restart" not in captured


def test_solve_transport_linear_with_residual_implicit_routes_to_custom_solve(monkeypatch) -> None:
    captured = {}

    def fake_custom(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(transport_linear, "linear_custom_solve_with_residual", fake_custom)

    result, residual = solve_transport_linear_with_residual(
        context=_context(use_implicit=True, size_hint=23),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["solver"] == "bicgstab"
    assert captured["solve_method"] == "batched"
    assert captured["size_hint"] == 23
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_with_residual_uses_plain_or_jit_gmres(monkeypatch) -> None:
    calls: list[str] = []

    def fake_gmres(**kwargs):
        calls.append(f"gmres:{kwargs['solve_method']}")
        return SimpleNamespace(residual_norm=jnp.asarray(1.0)), jnp.ones((2,))

    def fake_gmres_jit(**kwargs):
        calls.append(f"jit:{kwargs['solve_method']}")
        return SimpleNamespace(residual_norm=jnp.asarray(2.0)), 2.0 * jnp.ones((2,))

    monkeypatch.setattr(transport_linear, "gmres_solve_with_residual", fake_gmres)
    monkeypatch.setattr(transport_linear, "gmres_solve_with_residual_jit", fake_gmres_jit)
    args = dict(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    plain, plain_residual = solve_transport_linear_with_residual(context=_context(use_solver_jit=False), **args)
    jit, jit_residual = solve_transport_linear_with_residual(context=_context(use_solver_jit=True), **args)

    assert float(plain.residual_norm) == 1.0
    assert float(jit.residual_norm) == 2.0
    assert calls == ["gmres:incremental", "jit:incremental"]
    assert plain_residual.tolist() == [1.0, 1.0]
    assert jit_residual.tolist() == [2.0, 2.0]


def test_solve_transport_linear_with_residual_distributed_axis_routes(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    @contextmanager
    def fake_sharding_constraints(enabled: bool):
        calls.append({"context_enabled": enabled})
        yield

    def fake_distributed(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(transport_linear, "sharding_constraints", fake_sharding_constraints)
    monkeypatch.setattr(transport_linear, "gmres_solve_with_residual_distributed", fake_distributed)

    solve_transport_linear_with_residual(
        context=_context(distributed_axis="theta"),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
    )
    solve_transport_linear_with_residual(
        context=_context(distributed_axis="zeta"),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=11,
        maxiter_val=21,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    distributed_calls = [call for call in calls if "axis_name" in call]
    assert [call["axis_name"] for call in distributed_calls] == ["theta", "zeta"]
    assert [call["solve_method"] for call in distributed_calls] == ["bicgstab", "incremental"]
    assert sum(1 for call in calls if call == {"context_enabled": True}) == 2


def test_transport_linear_solve_callbacks_bind_residual_context(monkeypatch) -> None:
    captured = {}

    def fake_bicgstab(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(transport_linear, "bicgstab_solve_with_residual", fake_bicgstab)
    callbacks = TransportLinearSolveCallbacks(context=_context(use_solver_jit=False))

    result, residual = callbacks.solve_with_residual(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["precondition_side"] == "right"
