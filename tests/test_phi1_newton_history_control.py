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
