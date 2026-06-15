from __future__ import annotations

from types import SimpleNamespace

from jax import config as jax_config
import jax.numpy as jnp
import numpy as np

import sfincs_jax.v3_driver as v3_driver
from sfincs_jax.constraint_projection import (
    project_constraint_scheme1_nullspace_solution,
    project_constraint_scheme1_nullspace_solution_with_residual,
)

jax_config.update("jax_enable_x64", True)


def _op(
    *,
    constraint_scheme: int = 1,
    phi1_size: int = 0,
    extra_size: int = 2,
    point_at_x0: bool = False,
):
    return SimpleNamespace(
        constraint_scheme=constraint_scheme,
        phi1_size=phi1_size,
        extra_size=extra_size,
        point_at_x0=point_at_x0,
        x=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        fblock=SimpleNamespace(f_shape=(1, 2, 1, 1, 1)),
    )


def _identity_operator(_op, x):
    return x


def test_projection_reduces_constraint_source_residual() -> None:
    op = _op()
    x = jnp.zeros((4,), dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 0.0, -2.0, 3.0], dtype=jnp.float64)

    x_projected, residual_projected = project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x,
        rhs_vec=rhs,
        matvec_op=op,
        enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        apply_operator=_identity_operator,
    )

    np.testing.assert_allclose(np.asarray(residual_projected[-2:]), np.zeros((2,)), atol=1e-10)
    assert np.linalg.norm(np.asarray(residual_projected)) < np.linalg.norm(np.asarray(x - rhs))
    assert bool(jnp.any(x_projected != x))


def test_projection_returns_supplied_residual_for_ineligible_system() -> None:
    op = _op(constraint_scheme=2)
    x = jnp.ones((4,), dtype=jnp.float64)
    rhs = jnp.zeros((4,), dtype=jnp.float64)
    supplied_residual = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    def _raising_operator(_op, _x):
        raise AssertionError("ineligible projection should reuse supplied residual")

    x_projected, residual_projected = project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x,
        rhs_vec=rhs,
        matvec_op=op,
        enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        residual_vec=supplied_residual,
        apply_operator=_raising_operator,
    )

    np.testing.assert_allclose(np.asarray(x_projected), np.asarray(x))
    np.testing.assert_allclose(np.asarray(residual_projected), np.asarray(supplied_residual))


def test_projection_respects_disabled_environment(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE", "off")
    op = _op()
    x = jnp.zeros((4,), dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 0.0, -2.0, 3.0], dtype=jnp.float64)

    x_projected, residual_projected = project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x,
        rhs_vec=rhs,
        matvec_op=op,
        enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        apply_operator=_identity_operator,
    )

    np.testing.assert_allclose(np.asarray(x_projected), np.asarray(x))
    np.testing.assert_allclose(np.asarray(residual_projected), np.asarray(x - rhs))


def test_transport_projection_skips_roundoff_constraint_residual() -> None:
    op = _op()
    x = jnp.zeros((4,), dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 0.0, -1.0e-11, 2.0e-11], dtype=jnp.float64)

    x_projected, residual_projected = project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x,
        rhs_vec=rhs,
        matvec_op=op,
        enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        apply_operator=_identity_operator,
    )

    np.testing.assert_allclose(np.asarray(x_projected), np.asarray(x))
    np.testing.assert_allclose(np.asarray(residual_projected), np.asarray(x - rhs))


def test_projection_state_wrapper_and_driver_alias() -> None:
    op = _op()
    x = jnp.zeros((4,), dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 0.0, -2.0, 3.0], dtype=jnp.float64)

    x_projected = project_constraint_scheme1_nullspace_solution(
        op=op,
        x_vec=x,
        rhs_vec=rhs,
        matvec_op=op,
        enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        apply_operator=_identity_operator,
    )

    assert bool(jnp.any(x_projected != x))
    assert v3_driver._project_constraint_scheme1_nullspace_solution is project_constraint_scheme1_nullspace_solution
