from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.preconditioner_qi_basis import orthonormalize_rhs1_qi_coarse_basis
from sfincs_jax.solvers.preconditioner_qi_corrections import (
    RHS1QICoupledResidualEquationConfig,
    RHS1QICoupledResidualEquationMetadata,
    RHS1QICoupledResidualEquationState,
    setup_rhs1_qi_coupled_residual_equation,
)
from sfincs_jax.solvers.preconditioner_qi_device import (
    RHS1QIDevicePreconditionerConfig,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)


def test_coupled_residual_equation_updates_earlier_stage_coefficients() -> None:
    operator_matrix = jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    basis0 = orthonormalize_rhs1_qi_coarse_basis(
        jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        labels=("x0",),
    )
    basis1 = orthonormalize_rhs1_qi_coarse_basis(
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64),
        labels=("x1",),
    )

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    # A staged action least-squares cascade freezes the first coefficient at 0,
    # so the second stage cannot later choose the coupled coefficient pair
    # (-1, +1) that solves the residual exactly.
    coeff0 = jnp.linalg.lstsq(matvec(basis0.vectors), rhs, rcond=None)[0]
    remaining = rhs - matvec(basis0.vectors) @ coeff0
    coeff1 = jnp.linalg.lstsq(matvec(basis1.vectors), remaining, rcond=None)[0]
    staged_residual = rhs - matvec(basis0.vectors) @ coeff0 - matvec(basis1.vectors) @ coeff1

    state = setup_rhs1_qi_coupled_residual_equation(
        operator=matvec,
        residual=rhs,
        bases=(basis0, basis1),
        config=RHS1QICoupledResidualEquationConfig(
            solver="action_lstsq",
            regularization_rcond=0.0,
        ),
    )
    correction = state.apply(rhs)
    compiled = jax.jit(state.apply)(rhs)
    tangent = jax.vjp(state.apply, rhs)[1](rhs)[0]

    assert isinstance(state, RHS1QICoupledResidualEquationState)
    assert isinstance(state.metadata, RHS1QICoupledResidualEquationMetadata)
    assert jnp.linalg.norm(staged_residual) > 1.0e-1
    assert state.metadata.accepted is True
    assert state.metadata.rank == 2
    assert state.metadata.to_dict()["solver"] == "action_lstsq"
    assert state.metadata.source_stage_ranks == (1, 1)
    assert state.metadata.residual_after < 1.0e-12
    np.testing.assert_allclose(matvec(correction), rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(compiled, correction, rtol=1.0e-12, atol=1.0e-12)
    assert bool(jnp.all(jnp.isfinite(tangent)))


def test_coupled_residual_equation_batches_operator_on_basis_setup() -> None:
    operator_matrix = jnp.asarray(
        [[2.0, 0.3, 0.0], [0.1, 1.5, -0.2], [0.0, 0.4, 1.2]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        jnp.eye(3, dtype=jnp.float64),
        labels=("x0", "x1", "x2"),
    )
    call_count = 0

    def matvec(x):
        nonlocal call_count
        call_count += 1
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_coupled_residual_equation(
        operator=matvec,
        residual=rhs,
        bases=(basis,),
        config=RHS1QICoupledResidualEquationConfig(
            solver="action_lstsq",
            regularization_rcond=0.0,
        ),
    )

    assert state.metadata.accepted is True
    assert state.metadata.rank == 3
    assert call_count == 1
    np.testing.assert_allclose(
        state.operator_on_basis,
        operator_matrix @ basis.vectors,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_coupled_residual_equation_empty_rank_and_rejected_paths_fail_closed() -> None:
    residual = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        labels=("x0",),
    )

    zero_state = setup_rhs1_qi_coupled_residual_equation(
        operator=lambda x: x,
        residual=jnp.zeros_like(residual),
        bases=(basis,),
    )
    assert zero_state.metadata.reason == "zero_residual"
    assert zero_state.solve_coefficients(residual).shape == (0,)
    np.testing.assert_allclose(zero_state.apply(residual), jnp.zeros_like(residual), atol=0.0)
    np.testing.assert_allclose(zero_state.residual_after_apply(residual), residual, atol=0.0)

    empty_state = setup_rhs1_qi_coupled_residual_equation(
        operator=lambda x: x,
        residual=residual,
        bases=(),
    )
    assert empty_state.metadata.reason == "empty_basis"
    assert empty_state.metadata.to_dict()["rank"] == 0

    rank_deficient = setup_rhs1_qi_coupled_residual_equation(
        operator=lambda x: x,
        residual=residual,
        bases=(basis,),
        config=RHS1QICoupledResidualEquationConfig(max_rank=0),
    )
    assert rank_deficient.metadata.reason == "rank_deficient"
    assert rank_deficient.metadata.candidate_count == 1

    rejected = setup_rhs1_qi_coupled_residual_equation(
        operator=lambda x: jnp.zeros_like(x),
        residual=residual,
        bases=(basis,),
        config=RHS1QICoupledResidualEquationConfig(
            min_relative_improvement=0.1,
            acceptance_atol=0.0,
        ),
    )
    assert rejected.metadata.reason == "no_residual_reduction"
    assert rejected.metadata.rank == 0
    assert rejected.metadata.candidate_labels == ("coupled:stage0:x0",)
    np.testing.assert_allclose(rejected.apply(residual), jnp.zeros_like(residual), atol=0.0)


def test_coupled_residual_equation_galerkin_aliases_and_errors() -> None:
    residual = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    operator_matrix = jnp.asarray([[2.0, 0.25], [0.25, 1.5]], dtype=jnp.float64)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        jnp.eye(2, dtype=jnp.float64),
        labels=("x0", "x1"),
    )

    state = setup_rhs1_qi_coupled_residual_equation(
        operator=lambda x: operator_matrix @ jnp.asarray(x, dtype=jnp.float64),
        residual=residual,
        bases=(basis,),
        config=RHS1QICoupledResidualEquationConfig(
            solver="Schur",
            regularization_rcond=0.0,
        ),
    )

    assert state.solver == "galerkin"
    assert state.metadata.accepted is True
    np.testing.assert_allclose(
        state.residual_after_apply(residual),
        residual - operator_matrix @ state.apply(residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="solver"):
        setup_rhs1_qi_coupled_residual_equation(
            operator=lambda x: x,
            residual=residual,
            bases=(basis,),
            config=RHS1QICoupledResidualEquationConfig(solver="not-a-solver"),
        )

    mismatched = orthonormalize_rhs1_qi_coarse_basis(
        jnp.eye(3, dtype=jnp.float64),
        labels=("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="same row count"):
        setup_rhs1_qi_coupled_residual_equation(
            operator=lambda x: x,
            residual=residual,
            bases=(mismatched,),
        )


def test_device_preconditioner_coupled_residual_equation_can_include_flat_basis() -> None:
    operator_matrix = jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    flat_basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float64)
    geometry_metadata = {
        "qi_block_sizes": (1, 1),
        "qi_block_x": (0, 1),
        "n_theta": 1,
        "n_zeta": 1,
    }

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    staged_state = setup_rhs1_qi_device_preconditioner(
        operator=matvec,
        total_size=2,
        coarse_basis=flat_basis,
        coarse_labels=("flat:x0",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_residual_equation=True,
            residual_snapshot_residual_equation_max_rank=1,
            residual_snapshot_residual_equation_include_global=False,
            regularization_rcond=0.0,
            max_rank=1,
        ),
    )
    _, staged_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=staged_state,
        min_relative_improvement=0.0,
    )

    coupled_state = setup_rhs1_qi_device_preconditioner(
        operator=matvec,
        total_size=2,
        coarse_basis=flat_basis,
        coarse_labels=("flat:x0",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_residual_equation=True,
            residual_snapshot_residual_equation_max_rank=1,
            residual_snapshot_residual_equation_include_global=False,
            coupled_residual_equation=True,
            coupled_residual_equation_include_flat=True,
            coupled_residual_equation_max_rank=2,
            coupled_residual_equation_solver="action_lstsq",
            regularization_rcond=0.0,
            max_rank=1,
        ),
    )
    x, coupled_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=coupled_state,
        min_relative_improvement=0.0,
    )

    assert staged_probe.residual_after_norm > 1.0e-1
    assert coupled_probe.accepted is True
    assert coupled_state.metadata.reason == "built_with_coupled_residual_equation"
    assert coupled_state.metadata.coupled_residual_equation_enabled is True
    assert coupled_state.metadata.coupled_residual_equation_rank == 2
    assert coupled_state.metadata.coupled_residual_equation_source_stage_ranks == (1, 1)
    assert coupled_state.metadata.coupled_residual_equation_include_flat is True
    assert coupled_probe.residual_after_norm < 1.0e-12
    np.testing.assert_allclose(matvec(x), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_reuses_coupled_residual_operator_action() -> None:
    operator_matrix = jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    flat_basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float64)
    geometry_metadata = {
        "qi_block_sizes": (1, 1),
        "qi_block_x": (0, 1),
        "n_theta": 1,
        "n_zeta": 1,
    }
    call_count = 0

    def matvec(x):
        nonlocal call_count
        call_count += 1
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=matvec,
        total_size=2,
        coarse_basis=flat_basis,
        coarse_labels=("flat:x0",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_residual_equation=True,
            residual_snapshot_residual_equation_max_rank=1,
            residual_snapshot_residual_equation_include_global=False,
            coupled_residual_equation=True,
            coupled_residual_equation_include_flat=True,
            coupled_residual_equation_max_rank=2,
            coupled_residual_equation_solver="action_lstsq",
            regularization_rcond=0.0,
            max_rank=1,
        ),
    )

    assert state.metadata.reason == "built_with_coupled_residual_equation"
    assert state.metadata.coupled_residual_equation_rank == 2
    assert state.metadata.residual_equation_operator_reuse_stage_count == 1
    assert state.metadata.residual_equation_operator_recomputed_stage_count == 0
    assert call_count == 3
