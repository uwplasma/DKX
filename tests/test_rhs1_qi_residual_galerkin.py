from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.preconditioners.qi.corrections import (
    RHS1QIResidualGalerkinConfig,
    setup_rhs1_qi_residual_galerkin,
)


def _fixed_global_residual_after(operator: jnp.ndarray, residual: jnp.ndarray) -> float:
    q = jnp.ones_like(residual)
    q = q / jnp.linalg.norm(q)
    aq = operator @ q
    denominator = jnp.vdot(aq, aq)
    coefficient = jnp.where(denominator > 0.0, jnp.vdot(aq, residual) / denominator, 0.0)
    return float(jnp.linalg.norm(residual - aq * coefficient))


def test_residual_derived_stages_recover_modes_missed_by_fixed_global_basis() -> None:
    operator = jnp.eye(9, dtype=jnp.float64)
    residual = jnp.asarray(
        [
            2.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            1.0,
            2.0,
        ],
        dtype=jnp.float64,
    )
    residual_before = float(jnp.linalg.norm(residual))

    fixed_after = _fixed_global_residual_after(operator, residual)
    state = setup_rhs1_qi_residual_galerkin(
        operator,
        residual,
        block_sizes=(3, 3, 3),
        config=RHS1QIResidualGalerkinConfig(
            max_stages=2,
            max_stage_rank=1,
            include_global_residual=False,
            include_block_residuals=True,
            regularization_rcond=1.0e-14,
        ),
    )

    assert fixed_after == pytest.approx(residual_before)
    assert state.metadata.accepted is True
    assert state.metadata.reason == "residual_reduced"
    assert state.metadata.rank == 2
    assert state.metadata.stage_count == 2
    assert state.metadata.stage_ranks == (1, 1)
    assert state.metadata.candidate_count == 6
    assert set(state.metadata.labels) == {
        "stage:0:block:2:residual",
        "stage:1:block:0:residual",
    }
    assert state.metadata.residual_after < residual_before * 1.0e-10
    np.testing.assert_allclose(
        residual - operator @ state.apply(residual),
        state.residual_after_apply(residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_rank_deficient_and_no_improvement_paths_fail_closed() -> None:
    operator = jnp.eye(4, dtype=jnp.float64)
    residual = jnp.asarray([1.0, -1.0, 0.0, 0.0], dtype=jnp.float64)

    rank_deficient = setup_rhs1_qi_residual_galerkin(
        operator,
        residual,
        block_sizes=(4,),
        config=RHS1QIResidualGalerkinConfig(
            max_stages=1,
            max_stage_rank=1,
            min_rank=2,
            regularization_rcond=1.0e-14,
        ),
    )

    assert rank_deficient.metadata.accepted is False
    assert rank_deficient.metadata.reason == "rank_deficient"
    assert rank_deficient.metadata.rank == 0
    assert rank_deficient.metadata.stage_ranks == (1,)
    assert rank_deficient.metadata.residual_after == pytest.approx(rank_deficient.metadata.residual_before)
    np.testing.assert_allclose(rank_deficient.apply(residual), jnp.zeros_like(residual), atol=0.0)

    zero_operator = jnp.zeros((4, 4), dtype=jnp.float64)
    no_improvement = setup_rhs1_qi_residual_galerkin(
        zero_operator,
        residual,
        block_sizes=(2, 2),
        config=RHS1QIResidualGalerkinConfig(
            max_stages=1,
            max_stage_rank=2,
            regularization_rcond=1.0e-14,
        ),
    )

    assert no_improvement.metadata.accepted is False
    assert no_improvement.metadata.reason == "not_reduced"
    assert no_improvement.metadata.rank == 0
    assert no_improvement.metadata.candidate_count == 2
    assert no_improvement.metadata.residual_after == pytest.approx(no_improvement.metadata.residual_before)
    np.testing.assert_allclose(no_improvement.apply(residual), jnp.zeros_like(residual), atol=0.0)


def test_residual_galerkin_apply_path_is_jit_compatible() -> None:
    operator = jnp.diag(jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float64))
    exact = jnp.asarray([1.0, -0.5, 0.25, -0.75], dtype=jnp.float64)
    residual = operator @ exact

    state = setup_rhs1_qi_residual_galerkin(
        operator,
        residual,
        block_sizes=(2, 2),
        config=RHS1QIResidualGalerkinConfig(
            max_stages=1,
            max_stage_rank=2,
            include_operator_images=True,
            include_operator_preimages=True,
            regularization_rcond=1.0e-14,
        ),
    )
    eager = state.apply(residual)
    compiled = jax.jit(state.as_preconditioner())(residual)
    remaining = state.residual_after_apply(residual)

    assert state.metadata.accepted is True
    assert state.metadata.solver == "action_lstsq"
    assert state.metadata.rank == 2
    assert state.metadata.candidate_count == 6
    assert np.isfinite(state.metadata.condition_estimate)
    assert float(jnp.linalg.norm(remaining)) == pytest.approx(state.metadata.residual_after)
    np.testing.assert_allclose(compiled, eager, rtol=1.0e-12, atol=1.0e-12)
