from __future__ import annotations

import math

import jax.numpy as jnp

from sfincs_jax.rhs1_residual import (
    l2_norm_float,
    recompute_true_residual_result,
    residual_converged,
    residual_target,
    safe_ratio,
)
from sfincs_jax.solver import GMRESSolveResult


def test_rhs1_residual_target_matches_petsc_style_gate() -> None:
    assert residual_target(atol=1.0e-12, tol=1.0e-6, rhs_norm=3.0) == 3.0e-6
    assert residual_target(atol=1.0e-4, tol=1.0e-6, rhs_norm=3.0) == 1.0e-4


def test_rhs1_l2_norm_float_and_safe_ratio_are_host_scalars() -> None:
    norm = l2_norm_float(jnp.asarray([3.0, 4.0]))

    assert isinstance(norm, float)
    assert norm == 5.0
    assert safe_ratio(2.0, 4.0) == 0.5
    assert safe_ratio(2.0, 0.0) is None
    assert safe_ratio(math.nan, 4.0) is None
    assert safe_ratio(2.0, math.inf) is None


def test_rhs1_residual_converged_requires_finite_residual_and_target() -> None:
    assert residual_converged(1.0e-8, 1.0e-7) is True
    assert residual_converged(1.0e-6, 1.0e-7) is False
    assert residual_converged(math.nan, 1.0e-7) is False
    assert residual_converged(1.0e-8, math.inf) is False


def test_recompute_true_residual_result_replaces_reported_krylov_norm() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        residual_vec=None,
        update_residual_vec=False,
    )

    assert updated is not result
    assert float(updated.residual_norm) == 4.0
    assert residual_vec is None
    assert residual_norm == 4.0


def test_recompute_true_residual_result_can_keep_computed_residual_vector() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )
    supplied_residual = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=supplied_residual,
        update_residual_vec=True,
    )

    assert float(updated.residual_norm) == 5.0
    assert residual_vec is not None
    assert jnp.array_equal(residual_vec, supplied_residual)
    assert residual_norm == 5.0


def test_recompute_true_residual_result_keeps_incumbent_on_nonfinite_true_norm() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    updated, residual_vec, residual_norm = recompute_true_residual_result(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        residual_vec=None,
        update_residual_vec=True,
    )

    assert updated is result
    assert residual_vec is None
    assert residual_norm == 7.0
