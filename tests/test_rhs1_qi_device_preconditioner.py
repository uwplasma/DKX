from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.rhs1_device_operator import device_csr_from_scipy_csr
from sfincs_jax.rhs1_qi_coarse import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata
from sfincs_jax.rhs1_qi_device_preconditioner import (
    RHS1QIDevicePreconditionerConfig,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)
from sfincs_jax.rhs1_qi_device_smoother import build_rhs1_qi_device_jacobi_smoother


def _basis(vectors: jnp.ndarray, label: str = "coarse") -> RHS1QICoarseBasis:
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    labels = tuple(f"{label}:{idx}" for idx in range(int(vectors.shape[1])))
    return RHS1QICoarseBasis(
        vectors=vectors,
        metadata=RHS1QICoarseBasisMetadata(
            total_size=int(vectors.shape[0]),
            candidate_count=int(vectors.shape[1]),
            rank=int(vectors.shape[1]),
            discarded_count=0,
            candidate_labels=labels,
            accepted_labels=labels,
            candidate_norms=tuple(1.0 for _ in labels),
            accepted_norms=tuple(1.0 for _ in labels),
            rank_rtol=1.0e-12,
            rank_atol=0.0,
        ),
    )


def _rank_one_coupled_operator(n: int = 6, strength: float = 3.0):
    diagonal = jnp.linspace(1.2, 2.4, n, dtype=jnp.float64)
    mode = jnp.ones((n,), dtype=jnp.float64)
    mode = mode / jnp.linalg.norm(mode)
    dense = jnp.diag(diagonal) + float(strength) * jnp.outer(mode, mode)
    return dense, mode, device_csr_from_scipy_csr(sp.csr_matrix(np.asarray(dense)), max_csr_mb=1.0)


def test_device_preconditioner_builds_field_split_state_and_metadata() -> None:
    dense, mode, device_operator = _rank_one_coupled_operator()
    smoother = build_rhs1_qi_device_jacobi_smoother(device_operator, damping=0.8, sweeps=1)
    basis = _basis(mode.reshape((-1, 1)), label="global")

    state = setup_rhs1_qi_device_preconditioner(
        operator=device_operator,
        coarse_basis=basis,
        local_smoother=smoother,
        operator_metadata={"source": "unit-test"},
        geometry_metadata={"n_theta": 3, "n_zeta": 2},
        config=RHS1QIDevicePreconditionerConfig(
            regularization_rcond=1.0e-14,
            damping=1.0,
            coarse_solver="action_lstsq",
        ),
    )
    residual = jnp.asarray([1.0, -0.25, 0.4, 0.6, -0.5, 0.9], dtype=jnp.float64)

    local = smoother.apply(residual)
    remaining = residual - dense @ local
    coefficients = state.solve_coarse(remaining)
    expected = local + basis.vectors @ coefficients

    assert state.metadata.reason == "built_with_coarse"
    assert state.metadata.rank == 1
    assert state.metadata.operator_source == "device_csr"
    assert state.metadata.device_resident is True
    assert state.metadata.host_fallback_used is False
    assert state.metadata.host_callback_free is True
    assert state.metadata.operator_metadata_keys == ("source",)
    assert state.metadata.geometry_metadata_keys == ("n_theta", "n_zeta")
    assert state.metadata.local_smoother_reason == "built"
    np.testing.assert_allclose(state.operator_on_basis, dense @ basis.vectors, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(state.apply(residual), expected, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_probe_accepts_true_residual_drop_and_fails_closed() -> None:
    dense, mode, device_operator = _rank_one_coupled_operator(strength=5.0)
    basis = _basis(mode.reshape((-1, 1)))
    rhs = jnp.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    state = setup_rhs1_qi_device_preconditioner(
        operator=device_operator,
        coarse_basis=basis,
        config=RHS1QIDevicePreconditionerConfig(
            jacobi_damping=0.4,
            jacobi_sweeps=1,
            regularization_rcond=1.0e-14,
            coarse_solver="action_lstsq",
        ),
    )
    local_state = setup_rhs1_qi_device_preconditioner(
        operator=device_operator,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(jacobi_damping=0.4, jacobi_sweeps=1),
    )

    local_x, local_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=local_state,
        min_relative_improvement=0.0,
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.05,
    )
    rejected_x, rejected_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.999999,
    )

    assert local_probe.accepted is True
    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.residual_after_norm < 0.5 * local_probe.residual_after_norm
    assert float(jnp.linalg.norm(rhs - dense @ x)) == pytest.approx(probe.residual_after_norm)
    assert rejected_probe.accepted is False
    assert rejected_probe.reason == "residual_not_reduced"
    np.testing.assert_allclose(rejected_x, x0, atol=0.0)
    assert float(jnp.linalg.norm(rhs - dense @ local_x)) == pytest.approx(local_probe.residual_after_norm)


def test_device_preconditioner_is_jittable_and_differentiable_in_residual() -> None:
    _, mode, device_operator = _rank_one_coupled_operator(n=5, strength=2.0)
    state = setup_rhs1_qi_device_preconditioner(
        operator=device_operator,
        coarse_basis=mode.reshape((-1, 1)),
        coarse_labels=("global",),
        config=RHS1QIDevicePreconditionerConfig(jacobi_step_policy="residual_minimizing"),
    )
    residual = jnp.linspace(-0.5, 0.75, 5, dtype=jnp.float64)

    eager = state.apply(residual)
    compiled = jax.jit(state.as_preconditioner())(residual)
    gradient = jax.grad(lambda scale: jnp.sum(state.apply(scale * residual) ** 2))(1.0)

    np.testing.assert_allclose(compiled, eager, rtol=1.0e-12, atol=1.0e-12)
    assert np.isfinite(float(gradient))
    assert state.metadata.accepted_basis_labels == ("global",)


def test_device_preconditioner_matrix_free_coarse_only_path_avoids_csr() -> None:
    mode = jnp.ones((5,), dtype=jnp.float64)
    mode = mode / jnp.linalg.norm(mode)
    dense = 2.0 * jnp.eye(5, dtype=jnp.float64) + 4.0 * jnp.outer(mode, mode)
    rhs = mode.copy()
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=5,
        coarse_basis=mode.reshape((-1, 1)),
        coarse_labels=("global",),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(rhs=rhs, x0=x0, state=state, min_relative_improvement=0.0)
    compiled = jax.jit(state.as_preconditioner())(rhs)

    assert state.operator is None
    assert state.local_smoother is None
    assert state.metadata.reason == "built_matrix_free_coarse_only"
    assert state.metadata.operator_source == "matrix_free"
    assert state.metadata.local_smoother_kind == "none"
    assert state.metadata.nnz == 0
    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(compiled, state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_matrix_free_residual_smoother_reduces_residual() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_minres",
            matrix_free_smoother_sweeps=1,
            matrix_free_smoother_damping=1.0,
            matrix_free_smoother_step_policy="residual_minimizing",
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
    )
    compiled = jax.jit(state.as_preconditioner())(rhs)

    assert state.operator is None
    assert state.metadata.reason == "built_local_only"
    assert state.metadata.operator_source == "matrix_free"
    assert state.metadata.local_smoother_kind == "matrix_free_residual"
    assert state.metadata.local_smoother_reason == "built"
    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.residual_after_norm < probe.residual_before_norm
    np.testing.assert_allclose(compiled, state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(float(jnp.linalg.norm(rhs - mv(x))), probe.residual_after_norm)


def test_device_preconditioner_matrix_free_stationary_smoother_is_bounded() -> None:
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_richardson",
            matrix_free_smoother_sweeps=1,
            matrix_free_smoother_damping=0.5,
            matrix_free_smoother_step_policy="stationary",
        ),
    )
    dx = state.apply(rhs)

    assert state.metadata.local_smoother_kind == "matrix_free_residual"
    np.testing.assert_allclose(dx, 0.5 * rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(rhs - mv(dx), 0.5 * rhs, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_matrix_free_block_smoother_solves_projected_groups() -> None:
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=None,
        geometry_metadata={"qi_block_sizes": (2, 2)},
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_block_minres",
            matrix_free_smoother_sweeps=1,
            matrix_free_smoother_damping=1.0,
            matrix_free_block_smoother_max_groups=4,
            matrix_free_block_smoother_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
    )
    compiled = jax.jit(state.as_preconditioner())(rhs)

    assert state.metadata.reason == "built_local_only"
    assert state.metadata.local_smoother_kind == "matrix_free_block_minres"
    assert state.local_smoother is not None
    assert state.local_smoother.metadata.block_count == 2
    assert state.local_smoother.metadata.group_count == 2
    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(compiled, rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mv(x), rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_matrix_free_block_smoother_adds_x_species_aggregates() -> None:
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=None,
        geometry_metadata={
            "qi_block_sizes": (1, 1, 1, 1),
            "qi_block_x": (0, 0, 1, 1),
            "qi_block_species": (0, 1, 0, 1),
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_block_minres",
            matrix_free_block_smoother_grouping="block_x_species",
            matrix_free_block_smoother_max_groups=8,
            matrix_free_block_smoother_rcond=1.0e-14,
        ),
    )
    correction = state.apply(rhs)

    assert state.local_smoother is not None
    assert state.local_smoother.metadata.grouping == "block_x_species"
    assert state.local_smoother.metadata.group_count == 8
    assert state.local_smoother.metadata.group_partitions[:4] == (
        ((0, 1),),
        ((1, 2),),
        ((2, 3),),
        ((3, 4),),
    )
    assert ((0, 1), (1, 2)) in state.local_smoother.metadata.group_partitions
    assert ((0, 1), (2, 3)) in state.local_smoother.metadata.group_partitions
    np.testing.assert_allclose(mv(correction), rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_matrix_free_block_smoother_requires_block_metadata() -> None:
    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    with pytest.raises(ValueError, match="qi_block_sizes"):
        setup_rhs1_qi_device_preconditioner(
            operator=mv,
            total_size=4,
            coarse_basis=None,
            config=RHS1QIDevicePreconditionerConfig(local_smoother_kind="matrix_free_block_minres"),
        )


def test_device_preconditioner_probe_runs_bounded_residual_reduction_cycles() -> None:
    dense = jnp.eye(2, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=jnp.eye(2, dtype=jnp.float64),
        coarse_labels=("a", "b"),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            damping=0.5,
            regularization_rcond=1.0e-14,
        ),
    )
    one_cycle_x, one_cycle_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
        max_cycles=1,
    )
    multi_cycle_x, multi_cycle_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
        max_cycles=3,
    )

    assert one_cycle_probe.accepted is True
    assert one_cycle_probe.cycles == 1
    assert multi_cycle_probe.accepted is True
    assert multi_cycle_probe.cycles == 3
    assert len(multi_cycle_probe.residual_history) == 4
    assert multi_cycle_probe.residual_history == tuple(sorted(multi_cycle_probe.residual_history, reverse=True))
    assert multi_cycle_probe.residual_after_norm < one_cycle_probe.residual_after_norm
    np.testing.assert_allclose(mv(one_cycle_x) - rhs, -0.5 * rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mv(multi_cycle_x) - rhs, -0.125 * rhs, rtol=1.0e-12, atol=1.0e-12)
    assert multi_cycle_probe.to_dict()["cycles"] == 3


def test_device_preconditioner_probe_reports_last_rejected_candidate_residual() -> None:
    dense = jnp.eye(2, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=jnp.eye(2, dtype=jnp.float64),
        coarse_labels=("a", "b"),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            damping=3.0,
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
        max_cycles=4,
    )

    assert probe.accepted is False
    assert probe.reason == "residual_not_reduced"
    assert probe.cycles == 0
    assert probe.residual_history == pytest.approx((float(jnp.linalg.norm(rhs)),))
    assert probe.residual_after_norm == pytest.approx(2.0 * float(jnp.linalg.norm(rhs)))
    assert probe.improvement_ratio == pytest.approx(2.0)
    np.testing.assert_allclose(x, x0, atol=0.0)


def test_device_preconditioner_probe_minres_step_accepts_overscaled_direction() -> None:
    dense = jnp.eye(2, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=jnp.eye(2, dtype=jnp.float64),
        coarse_labels=("a", "b"),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            damping=3.0,
            regularization_rcond=1.0e-14,
        ),
    )
    rejected_x, rejected_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.0,
        residual_minimizing_step=True,
    )

    assert rejected_probe.accepted is False
    np.testing.assert_allclose(rejected_x, x0, atol=0.0)
    assert probe.accepted is True
    assert probe.cycles == 1
    assert probe.step_history == pytest.approx((1.0 / 3.0,))
    assert probe.residual_after_norm < 1.0e-10
    assert probe.to_dict()["step_history"] == pytest.approx((1.0 / 3.0,))
    np.testing.assert_allclose(mv(x), rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_residual_enrichment_adds_matrix_free_directions() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float64))
    rhs = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    irrelevant_basis = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64).reshape((-1, 1))

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=irrelevant_basis,
        coarse_labels=("irrelevant",),
        config=RHS1QIDevicePreconditionerConfig(local_smoother_kind="none"),
    )
    _, base_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=base_state,
        min_relative_improvement=0.0,
    )
    enriched_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=irrelevant_basis,
        coarse_labels=("irrelevant",),
        residual_seed=rhs,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_enrichment=True,
            residual_enrichment_depth=1,
            max_rank=3,
            regularization_rcond=1.0e-14,
        ),
    )
    x, enriched_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=enriched_state,
        min_relative_improvement=0.05,
    )

    assert base_probe.accepted is False
    assert enriched_probe.accepted is True
    assert enriched_state.metadata.operator_source == "matrix_free"
    assert enriched_state.metadata.residual_enrichment_enabled is True
    assert enriched_state.metadata.residual_enrichment_depth == 1
    assert enriched_state.metadata.residual_enrichment_candidate_count == 2
    assert "residual:0" in enriched_state.metadata.accepted_basis_labels
    assert enriched_probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_recycle_enrichment_targets_remaining_residual() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    first_component_only = jnp.asarray([1.0, 0.0], dtype=jnp.float64).reshape((-1, 1))

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=first_component_only,
        coarse_labels=("first",),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )
    _, base_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=base_state,
        min_relative_improvement=0.05,
    )
    recycle_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=first_component_only,
        coarse_labels=("first",),
        residual_seed=rhs,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            recycle_enrichment=True,
            recycle_enrichment_cycles=1,
            max_rank=2,
            regularization_rcond=1.0e-14,
        ),
    )
    x, recycle_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=recycle_state,
        min_relative_improvement=0.05,
    )

    assert base_probe.accepted is True
    assert base_probe.residual_after_norm == pytest.approx(1.0)
    assert recycle_probe.accepted is True
    assert recycle_probe.residual_after_norm < 1.0e-10
    assert recycle_state.metadata.recycle_enrichment_enabled is True
    assert recycle_state.metadata.recycle_enrichment_cycles == 1
    assert recycle_state.metadata.recycle_enrichment_candidate_count == 1
    assert "recycle_residual:0" in recycle_state.metadata.accepted_basis_labels
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_residual_enrichment_requires_seed() -> None:
    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    with pytest.raises(ValueError, match="residual_seed is required"):
        setup_rhs1_qi_device_preconditioner(
            operator=mv,
            total_size=3,
            coarse_basis=None,
            config=RHS1QIDevicePreconditionerConfig(
                local_smoother_kind="none",
                residual_enrichment=True,
            ),
        )


def test_device_preconditioner_recycle_enrichment_requires_seed() -> None:
    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    with pytest.raises(ValueError, match="residual_seed is required"):
        setup_rhs1_qi_device_preconditioner(
            operator=mv,
            total_size=3,
            coarse_basis=None,
            config=RHS1QIDevicePreconditionerConfig(
                local_smoother_kind="none",
                recycle_enrichment=True,
                recycle_enrichment_cycles=1,
            ),
        )


def test_device_preconditioner_local_only_path_has_empty_coarse_state() -> None:
    _, _, device_operator = _rank_one_coupled_operator(n=4, strength=1.0)
    state = setup_rhs1_qi_device_preconditioner(operator=device_operator, coarse_basis=None)
    residual = jnp.asarray([1.0, -1.0, 0.5, 0.25], dtype=jnp.float64)

    assert state.metadata.reason == "built_local_only"
    assert state.metadata.rank == 0
    assert state.coarse_operator.shape == (0, 0)
    assert state.operator_on_basis.shape == (4, 0)
    assert state.local_smoother is not None
    np.testing.assert_allclose(
        state.apply(residual),
        state.local_smoother.apply(residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_device_preconditioner_rejects_mismatched_basis_shape() -> None:
    _, _, device_operator = _rank_one_coupled_operator(n=4, strength=1.0)

    with pytest.raises(ValueError, match="coarse basis must have shape"):
        setup_rhs1_qi_device_preconditioner(
            operator=device_operator,
            coarse_basis=jnp.ones((3, 1), dtype=jnp.float64),
        )
