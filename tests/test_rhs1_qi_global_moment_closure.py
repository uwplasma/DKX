from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.preconditioner_qi_basis import (
    RHS1QIGlobalMomentLayout,
    build_rhs1_qi_global_moment_candidates,
    build_rhs1_qi_global_moment_closure,
)


def _coupled_layout() -> RHS1QIGlobalMomentLayout:
    return RHS1QIGlobalMomentLayout(
        block_sizes=(5, 5, 5, 5),
        block_x=(0.0, 1.0, 2.0, 3.0),
        block_species=(0, 0, 1, 1),
    )


def _unit(values: jnp.ndarray) -> jnp.ndarray:
    return values / jnp.linalg.norm(values)


def test_global_moment_closure_reduces_coupled_residual_where_block_only_fails() -> None:
    layout = _coupled_layout()
    candidates, labels = build_rhs1_qi_global_moment_candidates(
        layout,
        max_candidates=24,
        include_blocks=False,
    )
    constraint = _unit(candidates[:, labels.index("constraint:m1")])
    profile_constraint = _unit(candidates[:, labels.index("profile:x_ramp*constraint:m1")])
    n = layout.total_size
    identity = jnp.eye(n, dtype=jnp.float64)
    coupling = (
        0.20 * jnp.outer(constraint, constraint)
        + 0.30 * jnp.outer(profile_constraint, profile_constraint)
        + 0.25 * (jnp.outer(constraint, profile_constraint) + jnp.outer(profile_constraint, constraint))
    )
    operator_matrix = identity + coupling
    exact = 0.70 * constraint - 0.45 * profile_constraint
    rhs = operator_matrix @ exact

    def operator(x: jnp.ndarray) -> jnp.ndarray:
        return operator_matrix @ x

    block_solution, block_closure = build_rhs1_qi_global_moment_closure(
        operator=operator,
        rhs=rhs,
        layout=layout,
        solver="galerkin",
        include_current=False,
        include_profile=False,
        include_constraint=False,
        include_cross_moments=False,
        include_blocks=True,
    )
    solution, closure = build_rhs1_qi_global_moment_closure(
        operator=operator,
        rhs=rhs,
        layout=layout,
        solver="galerkin",
        max_rank=12,
    )

    assert block_closure.metadata.accepted is False
    assert block_closure.metadata.reason == "not_reduced"
    assert block_closure.metadata.residual_after_norm == pytest.approx(
        block_closure.metadata.residual_before_norm,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(block_solution, jnp.zeros_like(rhs), atol=1.0e-12)

    assert closure.metadata.accepted is True
    assert closure.metadata.reason == "residual_reduced"
    assert closure.metadata.solver == "galerkin"
    assert closure.metadata.rank >= 4
    assert closure.metadata.numerical_rank == closure.metadata.rank
    assert closure.metadata.candidate_count >= closure.metadata.rank
    assert "constraint:m1" in closure.metadata.labels
    assert "profile:x_ramp*constraint:m1" in closure.metadata.labels
    assert closure.metadata.condition_estimate >= 1.0
    assert np.isfinite(closure.metadata.condition_estimate)
    assert closure.metadata.residual_after_norm < closure.metadata.residual_before_norm * 1.0e-9
    np.testing.assert_allclose(operator(solution), rhs, atol=1.0e-9)


def test_global_moment_closure_rank_deficient_candidates_fail_closed() -> None:
    candidate = jnp.asarray([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
    candidates = jnp.stack((candidate, 2.0 * candidate), axis=1)
    rhs = candidate
    operator_matrix = jnp.eye(4, dtype=jnp.float64)

    solution, closure = build_rhs1_qi_global_moment_closure(
        operator=operator_matrix,
        rhs=rhs,
        candidates=candidates,
        labels=("duplicate:a", "duplicate:b"),
        require_independent_candidates=True,
    )

    assert closure.metadata.accepted is False
    assert closure.metadata.reason == "candidate_rank_deficient"
    assert closure.metadata.candidate_count == 2
    assert closure.metadata.rank == 1
    assert closure.metadata.discarded_count == 1
    assert closure.metadata.residual_after_norm == pytest.approx(closure.metadata.residual_before_norm)
    np.testing.assert_allclose(solution, jnp.zeros_like(rhs), atol=0.0)
    np.testing.assert_allclose(closure.apply(rhs), jnp.zeros_like(rhs), atol=0.0)


def test_global_moment_closure_apply_is_jittable_and_differentiable() -> None:
    layout = RHS1QIGlobalMomentLayout(
        block_sizes=(4, 4, 4),
        block_x=(-1.0, 0.0, 1.0),
        block_species=(0, 1, 1),
    )
    candidates, labels = build_rhs1_qi_global_moment_candidates(layout, max_candidates=18)
    profile_constraint = _unit(candidates[:, labels.index("profile:x_ramp*constraint:m1")])
    constraint = _unit(candidates[:, labels.index("constraint:m1")])
    diag = jnp.linspace(1.2, 2.4, layout.total_size, dtype=jnp.float64)
    operator_matrix = (
        jnp.diag(diag)
        + 0.15 * jnp.outer(profile_constraint, profile_constraint)
        + 0.10 * jnp.outer(constraint, constraint)
    )
    exact = 0.4 * profile_constraint + 0.2 * constraint
    rhs = operator_matrix @ exact

    solution, closure = build_rhs1_qi_global_moment_closure(
        operator=operator_matrix,
        rhs=rhs,
        layout=layout,
        solver="action_lstsq",
        max_rank=10,
    )
    residual = jnp.linspace(-1.0, 1.0, layout.total_size, dtype=jnp.float64)

    eager = closure.apply(residual)
    compiled = jax.jit(closure.apply)(residual)

    def action_energy(scale: jnp.ndarray) -> jnp.ndarray:
        value = closure.apply(scale * residual)
        return jnp.vdot(value, value)

    gradient = jax.grad(action_energy)(jnp.asarray(1.25, dtype=jnp.float64))

    assert closure.metadata.accepted is True
    assert closure.metadata.solver == "action_lstsq"
    assert solution.shape == rhs.shape
    np.testing.assert_allclose(compiled, eager, rtol=2.0e-5, atol=2.0e-5)
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0
