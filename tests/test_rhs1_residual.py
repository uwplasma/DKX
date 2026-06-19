from __future__ import annotations

import math

import jax.numpy as jnp

from sfincs_jax.rhs1_residual import (
    l2_norm_float,
    recompute_true_residual_result,
    replay_left_preconditioned_residual_norms,
    residual_converged,
    residual_target,
    safe_ratio,
    true_residual_norm_or_inf,
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


def test_true_residual_norm_or_inf_returns_finite_norm() -> None:
    norm = true_residual_norm_or_inf(
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
    )

    assert norm == 4.0


def test_true_residual_norm_or_inf_maps_nonfinite_norm_to_infinity() -> None:
    norm = true_residual_norm_or_inf(
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        x=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert norm == math.inf


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


def test_replay_left_preconditioned_residual_norms_noops_without_left_preconditioner() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=None,
        preconditioner=None,
        precondition_side="none",
        update_residual_vec=True,
    )

    assert residual_vec is None
    assert true_norm == 7.0
    assert check_norm == 7.0


def test_replay_left_preconditioned_residual_norms_reports_true_and_preconditioned_norms() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        matvec=lambda x: jnp.asarray([x[0], 2.0 * x[1]], dtype=jnp.float64),
        residual_vec=None,
        preconditioner=lambda r: 0.5 * r,
        precondition_side="left",
        update_residual_vec=True,
    )

    assert residual_vec is not None
    assert jnp.array_equal(residual_vec, jnp.asarray([0.0, 4.0], dtype=jnp.float64))
    assert true_norm == 4.0
    assert check_norm == 2.0


def test_replay_left_preconditioned_residual_norms_keeps_supplied_residual_without_update() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(99.0, dtype=jnp.float64),
    )
    supplied_residual = jnp.asarray([3.0, 4.0], dtype=jnp.float64)

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        matvec=lambda _x: (_ for _ in ()).throw(RuntimeError("should not be called")),
        residual_vec=supplied_residual,
        preconditioner=lambda r: 2.0 * r,
        precondition_side="left",
        update_residual_vec=False,
    )

    assert residual_vec is supplied_residual
    assert true_norm == 5.0
    assert check_norm == 10.0


def test_replay_left_preconditioned_residual_norms_preserves_current_check_when_preconditioned_nonfinite() -> None:
    result = GMRESSolveResult(
        x=jnp.asarray([1.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(7.0, dtype=jnp.float64),
    )

    residual_vec, true_norm, check_norm = replay_left_preconditioned_residual_norms(
        result=result,
        rhs=jnp.asarray([0.0], dtype=jnp.float64),
        matvec=lambda _x: jnp.asarray([jnp.nan], dtype=jnp.float64),
        residual_vec=None,
        preconditioner=lambda r: r,
        precondition_side="left",
        update_residual_vec=True,
    )

    assert residual_vec is not None
    assert true_norm == math.inf
    assert check_norm == 7.0
