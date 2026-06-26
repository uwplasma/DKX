from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.problems.profile_phi1_newton as phi1
from sfincs_jax.solver import GMRESSolveResult


@dataclass(frozen=True)
class _FakeCollisionless:
    n_xi_for_x: np.ndarray


@dataclass(frozen=True)
class _FakeFBlock:
    collisionless: _FakeCollisionless = field(
        default_factory=lambda: _FakeCollisionless(n_xi_for_x=np.asarray([2, 2], dtype=np.int32))
    )
    pas: object | None = None
    fp: object | None = None


@dataclass(frozen=True)
class _FakeOperator:
    total_size: int
    n_xi: int = 2
    rhs_mode: int = 1
    include_phi1: bool = False
    f_size: int = 0
    n_theta: int = 0
    n_zeta: int = 0
    phi1_hat_base: object | None = None
    fblock: _FakeFBlock = field(default_factory=_FakeFBlock)


class _FakeNamelist:
    def group(self, _name: str) -> dict[str, object]:
        return {}


def _install_linear_problem(monkeypatch: pytest.MonkeyPatch, *, target: jnp.ndarray) -> _FakeOperator:
    op = _FakeOperator(total_size=int(target.size))
    monkeypatch.setattr(phi1, "full_system_operator_from_namelist", lambda **_kwargs: op)
    monkeypatch.setattr(phi1, "residual_v3_full_system", lambda _op, x: x - target)
    monkeypatch.setattr(phi1, "apply_v3_full_system_operator_cached", lambda _op, x, **_kwargs: x)
    monkeypatch.setattr(phi1, "rhs_v3_full_system_jit", lambda _op: target)
    monkeypatch.setattr(
        phi1,
        "_dispatch_gmres",
        lambda **kwargs: GMRESSolveResult(
            x=jnp.asarray(kwargs["b"], dtype=jnp.float64),
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
        ),
    )
    return op


def test_newton_krylov_small_linear_problem_uses_dispatch_and_line_search(monkeypatch: pytest.MonkeyPatch) -> None:
    target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    _install_linear_problem(monkeypatch, target=target)

    result = phi1.solve_v3_full_system_newton_krylov(
        nml=_FakeNamelist(),
        max_newton=1,
        gmres_tol=1e-12,
        gmres_restart=8,
        gmres_maxiter=5,
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target))
    assert float(result.residual_norm) == pytest.approx(0.0)
    assert result.n_newton == 1
    assert float(result.last_linear_residual_norm) == pytest.approx(0.0)


def test_newton_krylov_rejects_wrong_initial_state_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_linear_problem(monkeypatch, target=jnp.asarray([1.0, 2.0], dtype=jnp.float64))

    with pytest.raises(ValueError, match="x0 must have shape"):
        phi1.solve_v3_full_system_newton_krylov(
            nml=_FakeNamelist(),
            x0=jnp.zeros((3,), dtype=jnp.float64),
        )


def test_newton_krylov_history_converges_and_records_accepted_state(monkeypatch: pytest.MonkeyPatch) -> None:
    target = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    _install_linear_problem(monkeypatch, target=target)
    emitted: list[str] = []

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        max_newton=3,
        gmres_tol=1e-12,
        gmres_restart=8,
        gmres_maxiter=5,
        emit=lambda _level, message: emitted.append(message),
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target))
    assert float(result.residual_norm) == pytest.approx(0.0)
    assert len(history) == 1
    np.testing.assert_allclose(np.asarray(history[-1]), np.asarray(target))
    assert any("evaluateJacobian called (autodiff linearization)" in msg for msg in emitted)


def test_newton_krylov_history_active_frozen_rhs_uses_reduced_linear_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = jnp.asarray([1.0, 0.0, -3.0, 0.0], dtype=jnp.float64)
    op = _FakeOperator(
        total_size=4,
        fblock=_FakeFBlock(
            collisionless=_FakeCollisionless(n_xi_for_x=np.asarray([1, 2], dtype=np.int32)),
        ),
    )
    monkeypatch.setenv("SFINCS_JAX_PHI1_ACTIVE_DOF", "1")
    monkeypatch.setattr(phi1, "full_system_operator_from_namelist", lambda **_kwargs: op)
    monkeypatch.setattr(phi1, "_transport_active_dof_indices", lambda _op: np.asarray([0, 2], dtype=np.int32))
    monkeypatch.setattr(phi1, "apply_v3_full_system_operator_cached", lambda _op, x, **_kwargs: x)
    monkeypatch.setattr(phi1, "rhs_v3_full_system_jit", lambda _op: target)
    monkeypatch.setattr(phi1, "residual_v3_full_system", lambda _op, x: x - target)
    calls: list[dict[str, object]] = []

    def _linear_step(**kwargs):
        calls.append(kwargs)
        residual_vec = kwargs["residual_vec"]
        step = -residual_vec
        return (
            GMRESSolveResult(x=step, residual_norm=jnp.asarray(0.0, dtype=jnp.float64)),
            step,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

    monkeypatch.setattr(phi1, "solve_phi1_newton_linear_step", _linear_step)

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        max_newton=3,
        use_frozen_linearization=True,
        gmres_restart=400,
        emit=lambda _level, _message: None,
    )

    assert calls
    assert calls[0]["use_active_dof_mode"] is True
    assert calls[0]["active_size"] == 2
    assert calls[0]["gmres_restart"] == 200
    reduced = calls[0]["reduce_full"](target)
    expanded = calls[0]["expand_reduced"](reduced)
    np.testing.assert_allclose(np.asarray(reduced), np.asarray([1.0, -3.0]))
    np.testing.assert_allclose(np.asarray(expanded), np.asarray(target))
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target))
    assert len(history) == 1


def test_newton_krylov_history_returns_last_accepted_on_nonfinite_linear_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    _install_linear_problem(monkeypatch, target=target)

    def _bad_linear_step(**_kwargs):
        return (
            GMRESSolveResult(
                x=jnp.asarray([jnp.nan, jnp.nan], dtype=jnp.float64),
                residual_norm=jnp.asarray(jnp.nan, dtype=jnp.float64),
            ),
            jnp.asarray([jnp.nan, jnp.nan], dtype=jnp.float64),
            jnp.asarray(jnp.nan, dtype=jnp.float64),
        )

    monkeypatch.setattr(phi1, "solve_phi1_newton_linear_step", _bad_linear_step)

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        x0=jnp.zeros_like(target),
        tol=1e-30,
        max_newton=1,
    )

    assert history == []
    np.testing.assert_allclose(np.asarray(result.x), np.zeros_like(np.asarray(target)))
    assert float(result.residual_norm) == pytest.approx(float(jnp.linalg.norm(target)))


def test_newton_krylov_history_rejects_wrong_initial_state_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_linear_problem(monkeypatch, target=jnp.asarray([1.0, 2.0], dtype=jnp.float64))

    with pytest.raises(ValueError, match="x0 must have shape"):
        phi1.solve_v3_full_system_newton_krylov_history(
            nml=_FakeNamelist(),
            x0=jnp.zeros((3,), dtype=jnp.float64),
        )


def test_newton_krylov_history_relative_tolerance_can_accept_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    _install_linear_problem(monkeypatch, target=target)

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        x0=jnp.zeros_like(target),
        tol=1e-30,
        nonlinear_rtol=2.0,
        max_newton=4,
    )

    np.testing.assert_allclose(np.asarray(result.x), np.zeros_like(np.asarray(target)))
    assert len(history) == 1
    np.testing.assert_allclose(np.asarray(history[0]), np.zeros_like(np.asarray(target)))
    assert float(result.residual_norm) == pytest.approx(float(jnp.linalg.norm(target)))
    assert result.n_newton == 0


def test_newton_krylov_history_nonfinite_residual_returns_fallback_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _FakeOperator(total_size=2)
    monkeypatch.setattr(phi1, "full_system_operator_from_namelist", lambda **_kwargs: op)
    monkeypatch.setattr(
        phi1,
        "apply_v3_full_system_operator_cached",
        lambda _op, x, **_kwargs: jnp.asarray([jnp.nan, x[1]], dtype=jnp.float64),
    )
    monkeypatch.setattr(phi1, "rhs_v3_full_system_jit", lambda _op: jnp.zeros((2,), dtype=jnp.float64))
    monkeypatch.setattr(phi1, "residual_v3_full_system", lambda _op, x: x + 1.0)

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        x0=jnp.asarray([2.0, 3.0], dtype=jnp.float64),
        max_newton=3,
    )

    assert history == []
    np.testing.assert_allclose(np.asarray(result.x), np.asarray([2.0, 3.0]))
    assert float(result.residual_norm) == pytest.approx(float(jnp.linalg.norm(jnp.asarray([3.0, 4.0]))))
    assert result.n_newton == 0


def test_newton_krylov_history_frozen_op_include_phi1_passes_dynamic_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = jnp.asarray([1.0, -2.0, 0.25], dtype=jnp.float64)
    op = _FakeOperator(
        total_size=3,
        include_phi1=True,
        f_size=2,
        n_theta=1,
        n_zeta=1,
        phi1_hat_base=jnp.asarray([[0.0]], dtype=jnp.float64),
    )
    seen_phi1_shapes: list[tuple[int, ...]] = []
    monkeypatch.setenv("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "frozen_op")
    monkeypatch.setattr(phi1, "full_system_operator_from_namelist", lambda **_kwargs: op)
    monkeypatch.setattr(phi1, "rhs_v3_full_system_jit", lambda _op: target)
    monkeypatch.setattr(phi1, "residual_v3_full_system", lambda _op, x: x - target)

    def _apply(op_arg, x, **_kwargs):
        if getattr(op_arg, "phi1_hat_base", None) is not None:
            seen_phi1_shapes.append(tuple(getattr(op_arg.phi1_hat_base, "shape", ())))
        return x

    monkeypatch.setattr(phi1, "apply_v3_full_system_operator_cached", _apply)

    def _linear_step(**kwargs):
        step = -kwargs["residual_vec"]
        matvec_sample = kwargs["matvec"](jnp.ones_like(step))
        np.testing.assert_allclose(np.asarray(matvec_sample), np.ones_like(np.asarray(step)))
        return (
            GMRESSolveResult(x=step, residual_norm=jnp.asarray(0.0, dtype=jnp.float64)),
            step,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

    monkeypatch.setattr(phi1, "solve_phi1_newton_linear_step", _linear_step)

    result, history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        max_newton=3,
        use_frozen_linearization=True,
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target))
    assert len(history) == 1
    assert (1, 1) in seen_phi1_shapes


def test_newton_krylov_history_fully_frozen_linearization_uses_jacobian_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    _install_linear_problem(monkeypatch, target=target)
    monkeypatch.setenv("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "frozen")
    jacobian_calls: list[np.ndarray] = []

    def _jacobian(_op, _x, dx):
        jacobian_calls.append(np.asarray(dx))
        return 2.0 * dx

    monkeypatch.setattr(phi1, "apply_v3_full_system_jacobian_jit", _jacobian)

    def _linear_step(**kwargs):
        probe = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(kwargs["matvec"](probe)), np.asarray([2.0, 4.0]))
        step = -kwargs["residual_vec"]
        return (
            GMRESSolveResult(x=step, residual_norm=jnp.asarray(0.0, dtype=jnp.float64)),
            step,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

    monkeypatch.setattr(phi1, "solve_phi1_newton_linear_step", _linear_step)

    result, _history = phi1.solve_v3_full_system_newton_krylov_history(
        nml=_FakeNamelist(),
        max_newton=2,
        use_frozen_linearization=True,
    )

    assert jacobian_calls
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target))
