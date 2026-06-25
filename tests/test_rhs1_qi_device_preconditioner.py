from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.operators.profile_response.device_sparse import device_csr_from_scipy_csr
from sfincs_jax.solvers.preconditioners.qi.coarse import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata, RHS1QICoarseBlockLayout
from sfincs_jax.solvers.preconditioners.qi.device import (
    RHS1QIDevicePreconditionerConfig,
    probe_rhs1_qi_device_augmented_seed,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)
from sfincs_jax.solvers.preconditioners.qi.device_smoother import build_rhs1_qi_device_jacobi_smoother
from sfincs_jax.solvers.preconditioners.qi.active_pattern import (
    RHS1QIActivePatternCoarseConfig,
    build_rhs1_qi_active_pattern_coarse_basis,
)
from sfincs_jax.solvers.preconditioners.qi.multilevel import (
    RHS1QIMultilevelCoarseConfig,
    build_rhs1_qi_multilevel_coarse_basis,
    build_rhs1_qi_multilevel_coarse_candidates,
)
from sfincs_jax.solvers.preconditioners.qi.phase_space import (
    RHS1QIPhaseSpaceCoarseConfig,
    build_rhs1_qi_phase_space_coarse_basis,
)
from sfincs_jax.solvers.preconditioners.qi.residual_regions import (
    RHS1QIResidualRegionCoarseConfig,
    build_rhs1_qi_residual_region_coarse_basis,
)


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


def _angular_radial_layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(8, 8, 8, 8),
        n_theta=4,
        n_zeta=1,
        block_x=(0, 1, 2, 3),
    )


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
    assert state.metadata.jax_default_backend == device_operator.metadata.default_backend
    assert state.metadata.jax_available_platforms == device_operator.metadata.available_platforms
    assert state.metadata.operator_array_devices == device_operator.metadata.array_devices
    assert state.metadata.operator_array_platforms == device_operator.metadata.array_platforms
    assert state.metadata.operator_arrays_same_device is True
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


def test_device_preconditioner_batches_matrix_free_coarse_basis_action() -> None:
    dense = jnp.asarray(
        [
            [2.0, 0.5, 0.0],
            [0.0, 3.0, -0.25],
            [1.0, 0.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    calls = 0

    def mv(x):
        nonlocal calls
        calls += 1
        vector = jnp.asarray(x, dtype=jnp.float64)
        if vector.ndim != 1:
            raise ValueError("test operator only accepts one vector")
        return dense @ vector

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=jnp.eye(3, dtype=jnp.float64),
        coarse_labels=("e0", "e1", "e2"),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )

    assert calls == 1
    assert state.metadata.operator_source == "matrix_free"
    assert state.metadata.operator_on_basis_shape == (3, 3)
    np.testing.assert_allclose(state.operator_on_basis, dense, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_multilevel_coarse_candidate_reduces_low_mode_residual() -> None:
    layout = _angular_radial_layout()
    multilevel_config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=20,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=multilevel_config)
    mode = candidates[:, labels.index("level:0:radial:p1:angular:theta_cos1")]
    mode = mode / jnp.linalg.norm(mode)
    dense = jnp.eye(layout.total_size, dtype=jnp.float64) + 4.0 * jnp.outer(mode, mode)
    rhs = dense @ mode
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            multilevel_coarse=True,
            multilevel_max_levels=3,
            multilevel_aggregate_factor=2,
            multilevel_max_rank=20,
            multilevel_max_angular_mode=1,
            multilevel_max_radial_degree=1,
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.05,
    )
    apply_preconditioner = state.as_preconditioner()
    compiled = jax.jit(apply_preconditioner)(rhs)
    cotangent = jnp.linspace(0.25, 1.25, layout.total_size, dtype=jnp.float64)
    transpose_action = jax.vjp(apply_preconditioner, rhs)[1](cotangent)[0]

    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    assert state.metadata.reason == "built_with_multilevel_coarse"
    assert state.metadata.multilevel_coarse_enabled is True
    assert state.metadata.multilevel_coarse_level_count == 3
    assert state.metadata.multilevel_coarse_candidate_count == len(labels)
    assert state.metadata.multilevel_coarse_rank > 0
    assert any(label.startswith("multilevel:") for label in state.metadata.accepted_basis_labels)
    np.testing.assert_allclose(compiled, state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)
    assert bool(jnp.all(jnp.isfinite(transpose_action)))


def test_device_preconditioner_multilevel_residual_equation_recovers_flat_rank_discard() -> None:
    layout = _angular_radial_layout()
    selector_config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=1,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=selector_config)
    flat_basis, _ = build_rhs1_qi_multilevel_coarse_basis(layout, config=selector_config)
    target_label = next(
        label
        for label in labels
        if ":angular:" in label and label not in flat_basis.metadata.accepted_labels
    )
    mode = candidates[:, labels.index(target_label)]
    mode = mode / jnp.linalg.norm(mode)
    dense = jnp.eye(layout.total_size, dtype=jnp.float64)
    rhs = dense @ mode
    x0 = jnp.zeros_like(rhs)
    geometry_metadata = {
        "qi_block_sizes": layout.block_sizes,
        "qi_block_x": layout.block_x,
        "n_theta": layout.n_theta,
        "n_zeta": layout.n_zeta,
    }

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    flat_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            multilevel_coarse=True,
            multilevel_max_levels=3,
            multilevel_aggregate_factor=2,
            multilevel_max_rank=1,
            multilevel_max_angular_mode=1,
            multilevel_max_radial_degree=1,
            regularization_rcond=1.0e-14,
        ),
    )
    residual_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            multilevel_coarse=True,
            multilevel_max_levels=3,
            multilevel_aggregate_factor=2,
            multilevel_max_rank=1,
            multilevel_max_angular_mode=1,
            multilevel_max_radial_degree=1,
            multilevel_residual_equation=True,
            multilevel_residual_equation_max_level_rank=64,
            multilevel_residual_equation_order="coarse_to_fine",
            regularization_rcond=1.0e-14,
        ),
    )

    _, flat_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=flat_state,
        min_relative_improvement=0.0,
    )
    x, residual_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=residual_state,
        min_relative_improvement=0.05,
    )
    compiled = jax.jit(residual_state.as_preconditioner())(rhs)

    assert flat_probe.residual_after_norm > 1.0e-2
    assert residual_probe.accepted is True
    assert residual_probe.residual_after_norm < 1.0e-10
    assert residual_state.metadata.reason == "built_with_multilevel_residual_equation"
    assert residual_state.metadata.multilevel_residual_equation_enabled is True
    assert residual_state.metadata.multilevel_residual_equation_stage_count == 3
    assert residual_state.metadata.multilevel_residual_equation_rank > residual_state.metadata.rank
    assert residual_state.metadata.multilevel_residual_equation_stage_ranks == (3, 6, 12)
    assert residual_state.metadata.multilevel_residual_equation_order == "coarse_to_fine"
    np.testing.assert_allclose(compiled, residual_state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_multilevel_current_moments_reach_device_state() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(8, 12, 8, 12, 2),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )
    multilevel_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        max_rank=16,
        max_angular_mode=0,
        max_radial_degree=0,
        include_angular=False,
        include_radial_angular=False,
        include_current_moments=True,
        max_current_pitch_degree=1,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=multilevel_config)
    mode = candidates[:, labels.index("current:global:p1")]
    mode = mode / jnp.linalg.norm(mode)
    dense = 1.1 * jnp.eye(layout.total_size, dtype=jnp.float64) + 2.5 * jnp.outer(mode, mode)
    rhs = dense @ mode

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "qi_block_species": layout.block_species,
            "qi_block_tail_size": 2,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            multilevel_coarse=True,
            multilevel_max_levels=2,
            multilevel_max_rank=16,
            multilevel_max_angular_mode=0,
            multilevel_max_radial_degree=0,
            multilevel_current_moments=True,
            multilevel_current_max_pitch_degree=1,
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        state=state,
        min_relative_improvement=0.0,
    )

    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    assert "multilevel:current:global:p1" in state.metadata.accepted_basis_labels
    assert "multilevel:constraint_tail:aggregate" in state.metadata.accepted_basis_labels
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_global_moment_residual_equation_closes_current_tail_mode() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(8, 12, 8, 12, 2),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )
    moment_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        max_rank=16,
        max_angular_mode=0,
        max_radial_degree=0,
        include_angular=False,
        include_radial_angular=False,
        include_current_moments=True,
        include_tail_constraint_moments=True,
        max_current_pitch_degree=1,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=moment_config)
    current_mode = candidates[:, labels.index("current:global:p1")]
    tail_mode = candidates[:, labels.index("constraint_tail:aggregate")]
    mode = current_mode + 0.35 * tail_mode
    mode = mode / jnp.linalg.norm(mode)
    dense = 1.2 * jnp.eye(layout.total_size, dtype=jnp.float64) + 2.0 * jnp.outer(mode, mode)
    rhs = dense @ mode

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "qi_block_species": layout.block_species,
            "qi_block_tail_size": 2,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            global_moment_residual_equation=True,
            global_moment_residual_equation_max_rank=8,
            global_moment_residual_equation_solver="galerkin",
            global_moment_residual_equation_include_profile=False,
            global_moment_residual_equation_include_current=True,
            global_moment_residual_equation_include_tail=True,
            multilevel_current_moments=True,
            multilevel_current_max_pitch_degree=1,
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        state=state,
        min_relative_improvement=0.0,
    )
    compiled = jax.jit(state.as_preconditioner())(rhs)

    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    assert state.metadata.reason == "built_with_global_moment_residual_equation"
    assert state.metadata.global_moment_residual_equation_enabled is True
    assert state.metadata.global_moment_residual_equation_rank >= 2
    assert state.metadata.global_moment_residual_equation_candidate_count >= 2
    assert state.metadata.global_moment_residual_equation_solver == "galerkin"
    assert np.isfinite(state.metadata.global_moment_residual_equation_condition_estimate)
    assert "global_moment:current:global:p1" in state.residual_equation_bases[0].metadata.accepted_labels
    assert (
        "global_moment:constraint_tail:aggregate"
        in state.residual_equation_bases[0].metadata.accepted_labels
    )
    np.testing.assert_allclose(compiled, state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_residual_galerkin_equation_solves_block_residual_modes() -> None:
    block_sizes = (3, 3, 2)
    scales = jnp.asarray([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 3.0, 3.0], dtype=jnp.float64)
    dense = jnp.diag(scales)
    expected = jnp.asarray([1.0, -0.5, 0.25, 0.6, -0.2, 0.4, 0.3, -0.1], dtype=jnp.float64)
    rhs = dense @ expected

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=int(rhs.shape[0]),
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": block_sizes,
            "n_theta": 1,
            "n_zeta": 1,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_galerkin_equation=True,
            residual_galerkin_equation_max_stages=2,
            residual_galerkin_equation_max_stage_rank=4,
            residual_galerkin_equation_max_rank=8,
            residual_galerkin_equation_solver="action_lstsq",
            residual_galerkin_equation_include_global_residual=True,
            residual_galerkin_equation_include_block_residuals=True,
            regularization_rcond=1.0e-14,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        state=state,
        min_relative_improvement=0.0,
    )

    assert probe.accepted is True
    assert probe.residual_after_norm < 1.0e-10
    assert state.metadata.reason == "built_with_residual_galerkin_equation"
    assert state.metadata.residual_galerkin_equation_enabled is True
    assert state.metadata.residual_galerkin_equation_rank >= 3
    assert state.metadata.residual_galerkin_equation_stage_count >= 1
    assert state.metadata.residual_galerkin_equation_candidate_count >= 3
    assert state.metadata.residual_galerkin_equation_solver == "action_lstsq"
    assert np.isfinite(state.metadata.residual_galerkin_equation_condition_estimate)
    assert "residual_galerkin:stage:0:block:0:residual" in (
        state.residual_equation_bases[0].metadata.accepted_labels
    )
    np.testing.assert_allclose(x, expected, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_phase_space_residual_equation_solves_pitch_mode() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(10, 10, 10, 10, 3),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )
    phase_basis = build_rhs1_qi_phase_space_coarse_basis(
        layout,
        config=RHS1QIPhaseSpaceCoarseConfig(
            include_radial=False,
            include_species=False,
            max_rank=8,
        ),
    )
    mode = jnp.asarray(phase_basis.vectors[:, 0], dtype=jnp.float64)
    dense = jnp.eye(layout.total_size, dtype=jnp.float64) + 2.0 * jnp.outer(mode, mode)
    rhs = dense @ mode
    operator = device_csr_from_scipy_csr(sp.csr_matrix(np.asarray(dense)), max_csr_mb=1.0)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=operator,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(
            regularization_rcond=1.0e-14,
            local_smoother_kind="none",
        ),
    )
    base_residual = rhs - dense @ base_state.apply(rhs)
    phase_state = setup_rhs1_qi_device_preconditioner(
        operator=operator,
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "qi_block_species": layout.block_species,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            regularization_rcond=1.0e-14,
            local_smoother_kind="none",
            phase_space_residual_equation=True,
            phase_space_residual_equation_max_rank=8,
            phase_space_residual_equation_include_global=True,
            phase_space_residual_equation_include_radial=False,
            phase_space_residual_equation_include_species=False,
        ),
    )

    corrected = phase_state.apply(rhs)
    residual_after = rhs - dense @ corrected
    jitted = jax.jit(phase_state.apply)(rhs)

    assert float(jnp.linalg.norm(base_residual)) > 1.0e-2
    assert float(jnp.linalg.norm(residual_after)) < 1.0e-10
    assert phase_state.metadata.phase_space_residual_equation_enabled is True
    assert phase_state.metadata.phase_space_residual_equation_rank > 0
    assert phase_state.metadata.phase_space_residual_equation_stage_count == 1
    assert phase_state.metadata.phase_space_residual_equation_candidate_count >= (
        phase_state.metadata.phase_space_residual_equation_rank
    )
    assert phase_state.metadata.phase_space_residual_equation_include_global is True
    assert np.isfinite(phase_state.metadata.phase_space_residual_equation_condition_estimate)
    assert all(label.startswith("phase_space:") for label in phase_state.metadata.accepted_basis_labels)
    np.testing.assert_allclose(jitted, corrected, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_residual_region_bounce_coarse_solves_local_mode() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12, 12, 3),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )
    seed = jnp.zeros((layout.total_size,), dtype=jnp.float64)
    seed = seed.at[layout.block_offsets[2] + 4 : layout.block_offsets[2] + 8].set(
        jnp.asarray([2.0, -1.0, 1.5, -0.5], dtype=jnp.float64)
    )
    region_basis = build_rhs1_qi_residual_region_coarse_basis(
        layout,
        seed,
        config=RHS1QIResidualRegionCoarseConfig(
            max_rank=8,
            min_region_energy_fraction=0.0,
            include_global_active_region=True,
        ),
    )
    mode = jnp.asarray(region_basis.vectors[:, 0], dtype=jnp.float64)
    dense = jnp.eye(layout.total_size, dtype=jnp.float64) + 2.0 * jnp.outer(mode, mode)
    rhs = dense @ mode
    operator = device_csr_from_scipy_csr(sp.csr_matrix(np.asarray(dense)), max_csr_mb=1.0)

    state = setup_rhs1_qi_device_preconditioner(
        operator=operator,
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "qi_block_species": layout.block_species,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            regularization_rcond=1.0e-14,
            local_smoother_kind="none",
            residual_region_bounce_coarse=True,
            residual_region_bounce_coarse_max_rank=8,
            residual_region_bounce_coarse_min_region_energy_fraction=0.0,
            residual_region_bounce_coarse_include_global=True,
        ),
    )

    corrected = state.apply(rhs)
    residual_after = rhs - dense @ corrected
    jitted = jax.jit(state.apply)(rhs)

    assert float(jnp.linalg.norm(residual_after)) < 1.0e-10
    assert state.metadata.residual_region_bounce_coarse_enabled is True
    assert state.metadata.residual_region_bounce_coarse_rank > 0
    assert state.metadata.residual_region_bounce_coarse_stage_count == 1
    assert state.metadata.residual_region_bounce_coarse_candidate_count >= (
        state.metadata.residual_region_bounce_coarse_rank
    )
    assert state.metadata.residual_region_bounce_coarse_include_global is True
    assert state.metadata.residual_region_bounce_coarse_bounce_boundary == pytest.approx(0.35)
    assert np.isfinite(state.metadata.residual_region_bounce_coarse_condition_estimate)
    assert all(
        label.startswith("residual_region:")
        for label in state.residual_equation_bases[0].metadata.accepted_labels
    )
    np.testing.assert_allclose(jitted, corrected, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_active_pattern_coarse_solves_chunk_mode() -> None:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12, 12, 3),
        n_theta=2,
        n_zeta=1,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )
    seed = jnp.zeros((layout.total_size,), dtype=jnp.float64)
    seed = seed.at[layout.block_offsets[1] + 2 : layout.block_offsets[1] + 8].set(
        jnp.asarray([1.0, -2.0, 0.75, -1.25, 1.5, -0.5], dtype=jnp.float64)
    )
    active_basis = build_rhs1_qi_active_pattern_coarse_basis(
        layout,
        seed,
        config=RHS1QIActivePatternCoarseConfig(
            max_rank=8,
            max_candidates=16,
            min_chunk_energy_fraction=0.0,
        ),
    )
    mode = jnp.asarray(active_basis.vectors[:, 0], dtype=jnp.float64)
    dense = jnp.eye(layout.total_size, dtype=jnp.float64) + 1.75 * jnp.outer(mode, mode)
    rhs = dense @ mode
    operator = device_csr_from_scipy_csr(sp.csr_matrix(np.asarray(dense)), max_csr_mb=1.0)

    state = setup_rhs1_qi_device_preconditioner(
        operator=operator,
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": layout.block_sizes,
            "qi_block_x": layout.block_x,
            "qi_block_species": layout.block_species,
            "n_theta": layout.n_theta,
            "n_zeta": layout.n_zeta,
        },
        config=RHS1QIDevicePreconditionerConfig(
            regularization_rcond=1.0e-14,
            local_smoother_kind="none",
            active_pattern_coarse=True,
            active_pattern_coarse_max_rank=8,
            active_pattern_coarse_max_candidates=16,
            active_pattern_coarse_min_chunk_energy_fraction=0.0,
            active_pattern_coarse_include_global=True,
        ),
    )

    corrected = state.apply(rhs)
    residual_after = rhs - dense @ corrected
    jitted = jax.jit(state.apply)(rhs)

    assert float(jnp.linalg.norm(residual_after)) < 1.0e-10
    assert state.metadata.active_pattern_coarse_enabled is True
    assert state.metadata.active_pattern_coarse_rank > 0
    assert state.metadata.active_pattern_coarse_stage_count == 1
    assert state.metadata.active_pattern_coarse_candidate_count >= (
        state.metadata.active_pattern_coarse_rank
    )
    assert state.metadata.active_pattern_coarse_include_global is True
    assert state.metadata.active_pattern_coarse_max_candidates_requested == 16
    assert np.isfinite(state.metadata.active_pattern_coarse_condition_estimate)
    assert all(
        label.startswith("active_pattern:")
        for label in state.residual_equation_bases[0].metadata.accepted_labels
    )
    np.testing.assert_allclose(jitted, corrected, rtol=1.0e-12, atol=1.0e-12)


def test_device_preconditioner_multilevel_coarse_requires_layout_metadata() -> None:
    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    with pytest.raises(ValueError, match="qi_block_sizes"):
        setup_rhs1_qi_device_preconditioner(
            operator=mv,
            total_size=4,
            coarse_basis=None,
            config=RHS1QIDevicePreconditionerConfig(
                local_smoother_kind="none",
                multilevel_coarse=True,
            ),
        )


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


def test_device_preconditioner_accepts_block_minres_hybrid_alias() -> None:
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=None,
        geometry_metadata={"qi_block_sizes": (2, 2)},
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_block_minres_hybrid",
            matrix_free_smoother_sweeps=1,
            matrix_free_block_smoother_max_groups=4,
            matrix_free_block_smoother_rcond=1.0e-14,
        ),
    )

    assert state.metadata.local_smoother_kind == "matrix_free_block_minres"
    np.testing.assert_allclose(state.apply(rhs), rhs, rtol=1.0e-11, atol=1.0e-11)


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


def test_device_preconditioner_adaptive_residual_equation_uses_multilevel_groups() -> None:
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0, -0.25, 0.75], dtype=jnp.float64)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=6,
        coarse_basis=None,
        geometry_metadata={"qi_block_sizes": (1, 1, 1, 1, 1, 1), "n_theta": 1, "n_zeta": 1},
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="adaptive_residual_equation",
            matrix_free_smoother_sweeps=1,
            matrix_free_block_smoother_max_groups=6,
            matrix_free_block_smoother_rcond=1.0e-14,
            multilevel_max_levels=3,
            multilevel_aggregate_factor=2,
        ),
    )
    correction = state.apply(rhs)
    compiled = jax.jit(state.as_preconditioner())(rhs)

    assert state.local_smoother is not None
    assert state.metadata.local_smoother_kind == "adaptive_residual_equation"
    assert state.local_smoother.metadata.grouping == "block_hierarchy"
    assert state.local_smoother.metadata.group_partitions[0] == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    )
    assert ((0, 1), (1, 2), (2, 3), (3, 4)) in state.local_smoother.metadata.group_partitions
    np.testing.assert_allclose(compiled, correction, rtol=1.0e-12, atol=1.0e-12)
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


def test_device_preconditioner_matrix_free_block_smoother_is_transpose_safe() -> None:
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=None,
        geometry_metadata={"qi_block_sizes": (1, 1, 1, 1)},
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="matrix_free_block_minres",
            matrix_free_block_smoother_max_groups=4,
            matrix_free_block_smoother_rcond=1.0e-14,
        ),
    )
    apply_preconditioner = state.as_preconditioner()
    compiled = jax.jit(apply_preconditioner)(rhs)
    pullback = jax.vjp(apply_preconditioner, rhs)[1](rhs)[0]

    np.testing.assert_allclose(compiled, rhs, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.all(jnp.isfinite(pullback))


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


def test_device_preconditioner_residual_snapshot_enrichment_solves_block_residual() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float64))
    rhs = jnp.asarray([0.0, 0.0, 4.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    irrelevant_basis = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64).reshape((-1, 1))

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=irrelevant_basis,
        coarse_labels=("irrelevant",),
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": (1, 1, 1, 1),
            "qi_block_x": (0, 1, 2, 3),
            "n_theta": 1,
            "n_zeta": 1,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_enrichment=True,
            residual_snapshot_max_rank=4,
            residual_snapshot_include_blocks=True,
            residual_snapshot_include_aggregates=False,
            regularization_rcond=1.0e-14,
            max_rank=5,
        ),
    )
    x, probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=state,
        min_relative_improvement=0.05,
    )

    assert probe.accepted is True
    assert state.metadata.residual_snapshot_enrichment_enabled is True
    assert state.metadata.residual_snapshot_candidate_count == 1
    assert state.metadata.residual_snapshot_rank == 1
    assert state.metadata.residual_snapshot_group_count == 4
    assert any("residual_snapshot_primal:2" in label for label in state.metadata.accepted_basis_labels)
    assert probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_residual_snapshot_requires_seed() -> None:
    with pytest.raises(ValueError, match="residual_seed is required"):
        setup_rhs1_qi_device_preconditioner(
            operator=lambda x: jnp.asarray(x, dtype=jnp.float64),
            total_size=2,
            geometry_metadata={"qi_block_sizes": (1, 1), "n_theta": 1, "n_zeta": 1},
            config=RHS1QIDevicePreconditionerConfig(
                local_smoother_kind="none",
                residual_snapshot_enrichment=True,
            ),
        )


def test_device_preconditioner_residual_snapshot_residual_equation_stages_blocks() -> None:
    dense = jnp.eye(4, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0, 0.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    irrelevant_basis = jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64).reshape((-1, 1))

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    flat_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=irrelevant_basis,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": (1, 1, 1, 1),
            "qi_block_x": (0, 1, 2, 3),
            "n_theta": 1,
            "n_zeta": 1,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_enrichment=True,
            residual_snapshot_max_rank=1,
            residual_snapshot_include_blocks=True,
            residual_snapshot_include_aggregates=False,
            regularization_rcond=1.0e-14,
            max_rank=1,
        ),
    )
    staged_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=irrelevant_basis,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": (1, 1, 1, 1),
            "qi_block_x": (0, 1, 2, 3),
            "n_theta": 1,
            "n_zeta": 1,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_residual_equation=True,
            residual_snapshot_residual_equation_max_rank=2,
            residual_snapshot_residual_equation_include_global=False,
            residual_snapshot_include_blocks=True,
            residual_snapshot_include_aggregates=False,
            regularization_rcond=1.0e-14,
            max_rank=1,
        ),
    )
    flat_x, flat_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=flat_state,
        min_relative_improvement=0.0,
    )
    staged_x, staged_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=staged_state,
        min_relative_improvement=0.0,
    )

    assert flat_probe.residual_after_norm > 1.0
    assert staged_probe.accepted is True
    assert staged_state.metadata.reason == "built_with_residual_snapshot_residual_equation"
    assert staged_state.metadata.residual_snapshot_residual_equation_enabled is True
    assert staged_state.metadata.residual_snapshot_residual_equation_group_count == 4
    assert staged_state.metadata.residual_snapshot_residual_equation_candidate_count == 2
    assert staged_state.metadata.residual_snapshot_residual_equation_rank == 2
    assert staged_state.metadata.residual_snapshot_residual_equation_stage_ranks == (1, 1)
    assert any(
        "residual_snapshot_equation_primal:0" in label
        for label in staged_state.residual_equation_bases[0].metadata.accepted_labels
    )
    assert any(
        "residual_snapshot_equation_primal:1" in label
        for label in staged_state.residual_equation_bases[1].metadata.accepted_labels
    )
    assert staged_probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(dense @ staged_x, rhs, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.linalg.norm(dense @ staged_x - rhs) < jnp.linalg.norm(dense @ flat_x - rhs)


def test_device_preconditioner_residual_snapshot_can_add_adjoint_normal_directions() -> None:
    dense = jnp.asarray([[1.0, 0.0], [3.0, 2.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 2.0], dtype=jnp.float64)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        residual_seed=rhs,
        geometry_metadata={"qi_block_sizes": (2,), "n_theta": 1, "n_zeta": 1},
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_enrichment=True,
            residual_snapshot_max_rank=2,
            residual_snapshot_include_global=True,
            residual_snapshot_include_blocks=False,
            residual_snapshot_include_aggregates=False,
            residual_snapshot_use_adjoint=True,
            regularization_rcond=1.0e-14,
        ),
    )

    assert state.metadata.residual_snapshot_enrichment_enabled is True
    assert state.metadata.residual_snapshot_candidate_count == 2
    assert state.metadata.residual_snapshot_include_primal is True
    assert state.metadata.residual_snapshot_use_adjoint is True
    assert any("residual_snapshot_adjoint:0" in label for label in state.metadata.accepted_basis_labels)


def test_device_preconditioner_block_schur_residual_equation_solves_coupled_mode_snapshots_miss() -> None:
    dense = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    block0_residual = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64).reshape((-1, 1))
    geometry_metadata = {
        "qi_block_sizes": (2, 2),
        "qi_block_x": (0, 1),
        "n_theta": 1,
        "n_zeta": 1,
    }

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=block0_residual,
        coarse_labels=("block0_residual",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )
    _, base_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=base_state,
        min_relative_improvement=0.0,
    )
    snapshot_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=block0_residual,
        coarse_labels=("block0_residual",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            residual_snapshot_enrichment=True,
            residual_snapshot_max_rank=4,
            residual_snapshot_include_blocks=True,
            residual_snapshot_include_aggregates=True,
            regularization_rcond=1.0e-14,
            max_rank=4,
        ),
    )
    _, snapshot_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=snapshot_state,
        min_relative_improvement=0.0,
    )
    schur_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=block0_residual,
        coarse_labels=("block0_residual",),
        residual_seed=rhs,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            block_schur_residual_equation=True,
            block_schur_residual_equation_max_rank=2,
            block_schur_residual_equation_include_blocks=True,
            block_schur_residual_equation_include_aggregates=False,
            regularization_rcond=1.0e-14,
            max_rank=3,
        ),
    )
    x, schur_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=schur_state,
        min_relative_improvement=0.05,
    )
    compiled = jax.jit(schur_state.as_preconditioner())(rhs)
    tangent = jax.vjp(schur_state.as_preconditioner(), rhs)[1](rhs)[0]

    assert base_probe.residual_after_norm > 1.0e-1
    assert snapshot_probe.accepted is False
    assert snapshot_probe.residual_after_norm > 1.0e-1
    assert schur_probe.accepted is True
    assert schur_state.metadata.reason == "built_with_block_schur_residual_equation"
    assert schur_state.metadata.block_schur_residual_equation_enabled is True
    assert schur_state.metadata.block_schur_residual_equation_candidate_count >= 2
    assert schur_state.metadata.block_schur_residual_equation_rank >= 1
    assert schur_state.metadata.block_schur_residual_equation_group_count == 2
    assert sum(schur_state.metadata.block_schur_residual_equation_stage_ranks) == (
        schur_state.metadata.block_schur_residual_equation_rank
    )
    assert any(
        "block_schur_coupled:block:0" in label
        for label in schur_state.residual_equation_bases[0].metadata.accepted_labels
    )
    assert schur_probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(compiled, schur_state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)
    assert bool(jnp.all(jnp.isfinite(tangent)))


def test_device_preconditioner_block_schur_residual_requires_seed() -> None:
    with pytest.raises(ValueError, match="residual_seed is required"):
        setup_rhs1_qi_device_preconditioner(
            operator=lambda x: jnp.asarray(x, dtype=jnp.float64),
            total_size=2,
            geometry_metadata={"qi_block_sizes": (1, 1), "n_theta": 1, "n_zeta": 1},
            config=RHS1QIDevicePreconditionerConfig(
                local_smoother_kind="none",
                block_schur_residual_equation=True,
            ),
        )


def test_device_preconditioner_operator_action_enrichment_builds_reused_coarse_space() -> None:
    dense = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    first_component_only = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64).reshape((-1, 1))

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
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
        min_relative_improvement=0.0,
    )

    action_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=first_component_only,
        coarse_labels=("first",),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            operator_action_enrichment=True,
            operator_action_enrichment_depth=1,
            regularization_rcond=1.0e-14,
        ),
    )
    x, action_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=action_state,
        min_relative_improvement=0.05,
    )
    compiled = jax.jit(action_state.as_preconditioner())(rhs)

    assert base_probe.accepted is True
    assert base_probe.residual_after_norm > 1.0e-2
    assert action_probe.accepted is True
    assert action_probe.residual_after_norm < 1.0e-10
    assert action_state.metadata.operator_source == "matrix_free"
    assert action_state.metadata.operator_action_enrichment_enabled is True
    assert action_state.metadata.operator_action_enrichment_depth == 1
    assert action_state.metadata.operator_action_enrichment_candidate_count == 1
    assert action_state.metadata.rank == 2
    assert "operator_action:1:first" in action_state.metadata.accepted_basis_labels
    assert action_state.operator_on_basis.shape == (3, 2)
    np.testing.assert_allclose(compiled, action_state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_operator_krylov_enrichment_solves_residual_subspace() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 5.0, 11.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    base_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(local_smoother_kind="none"),
    )
    _, base_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=base_state,
        min_relative_improvement=0.0,
    )

    krylov_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=None,
        residual_seed=rhs,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            operator_krylov_enrichment=True,
            operator_krylov_depth=1,
            regularization_rcond=1.0e-14,
        ),
    )
    x, krylov_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=krylov_state,
        min_relative_improvement=0.05,
    )

    assert base_probe.accepted is False
    assert krylov_probe.accepted is True
    assert krylov_probe.residual_after_norm < 1.0e-10
    assert krylov_state.metadata.operator_source == "matrix_free"
    assert krylov_state.metadata.operator_krylov_enrichment_enabled is True
    assert krylov_state.metadata.operator_krylov_depth == 1
    assert krylov_state.metadata.operator_krylov_candidate_count == 2
    assert krylov_state.metadata.rank == 2
    assert "operator_krylov:0" in krylov_state.metadata.accepted_basis_labels
    assert "operator_krylov:1" in krylov_state.metadata.accepted_basis_labels
    np.testing.assert_allclose(dense @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_device_preconditioner_adjoint_krylov_targets_nonnormal_left_error_mode() -> None:
    dense = jnp.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    residual_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=None,
        residual_seed=rhs,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            operator_krylov_enrichment=True,
            operator_krylov_depth=0,
            regularization_rcond=1.0e-14,
        ),
    )
    _, residual_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=residual_state,
        min_relative_improvement=0.0,
    )

    adjoint_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=2,
        coarse_basis=None,
        residual_seed=rhs,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            adjoint_krylov_enrichment=True,
            adjoint_krylov_depth=0,
            regularization_rcond=1.0e-14,
        ),
    )
    x, adjoint_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=adjoint_state,
        min_relative_improvement=0.0,
    )
    compiled = jax.jit(adjoint_state.as_preconditioner())(rhs)
    tangent = jax.vjp(adjoint_state.as_preconditioner(), rhs)[1](rhs)[0]

    assert residual_probe.accepted is False
    assert residual_probe.residual_after_norm == pytest.approx(residual_probe.residual_before_norm)
    assert adjoint_probe.accepted is True
    assert adjoint_probe.residual_after_norm < 1.0e-10
    assert adjoint_state.metadata.adjoint_krylov_enrichment_enabled is True
    assert adjoint_state.metadata.adjoint_krylov_depth == 0
    assert adjoint_state.metadata.adjoint_krylov_candidate_count == 1
    assert adjoint_state.metadata.adjoint_krylov_transpose_source == "autodiff"
    assert "adjoint_krylov:0" in adjoint_state.metadata.accepted_basis_labels
    np.testing.assert_allclose(mv(x), rhs, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(compiled, adjoint_state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    assert bool(jnp.all(jnp.isfinite(tangent)))


def test_device_preconditioner_operator_krylov_and_multilevel_apply_are_transpose_safe() -> None:
    dense = jnp.diag(jnp.asarray([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, -0.5, 0.25, 0.75], dtype=jnp.float64)

    def mv(x):
        return dense @ jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=4,
        coarse_basis=None,
        residual_seed=rhs,
        geometry_metadata={
            "qi_block_sizes": (2, 2),
            "qi_block_x": (0, 1),
            "qi_block_species": (0, 0),
            "n_theta": 1,
            "n_zeta": 2,
        },
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            operator_krylov_enrichment=True,
            operator_krylov_depth=1,
            multilevel_coarse=True,
            multilevel_max_levels=2,
            multilevel_max_rank=4,
            multilevel_max_angular_mode=1,
            multilevel_max_radial_degree=1,
            regularization_rcond=1.0e-14,
        ),
    )
    vector = jnp.asarray([0.2, -0.1, 0.4, 0.3], dtype=jnp.float64)
    _, pullback = jax.vjp(lambda x: state.apply(x), vector)
    tangent = pullback(jnp.ones_like(vector))[0]

    assert state.metadata.operator_krylov_enrichment_enabled is True
    assert state.metadata.multilevel_coarse_enabled is True
    assert state.metadata.multilevel_coarse_rank > 0
    assert jnp.all(jnp.isfinite(tangent))


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


def test_device_augmented_seed_probe_returns_rank_gated_correction_space() -> None:
    diagonal = jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64)

    def mv(x):
        return diagonal * jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=jnp.eye(3, dtype=jnp.float64),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )
    rhs = jnp.asarray([2.0, 0.0, 0.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    result = probe_rhs1_qi_device_augmented_seed(
        rhs=rhs,
        x0=x0,
        state=state,
        operator=mv,
        max_cycles=2,
        max_rank=1,
        min_relative_improvement=0.0,
        residual_minimizing_step=True,
    )

    assert result.probe.accepted is True
    assert result.probe.reason == "augmented_residual_reduced"
    assert result.rank == 1
    assert result.augmentation_basis.shape == (3, 1)
    assert result.operator_on_augmentation.shape == (3, 1)
    assert result.accepted_labels == ("augmented_seed:0",)
    assert result.projection_residual_norm is not None
    assert result.projection_residual_norm < 1.0e-12
    np.testing.assert_allclose(mv(result.solution), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_device_augmented_seed_probe_fails_closed_without_residual_reduction() -> None:
    def mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=3,
        coarse_basis=None,
        config=RHS1QIDevicePreconditionerConfig(local_smoother_kind="none"),
    )
    rhs = jnp.asarray([1.0, -1.0, 0.5], dtype=jnp.float64)
    result = probe_rhs1_qi_device_augmented_seed(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        state=state,
        operator=mv,
        max_cycles=2,
        max_rank=2,
    )

    assert result.probe.accepted is False
    assert result.reason == "residual_not_reduced"
    assert result.rank == 0
    assert result.augmentation_basis.shape == (3, 0)
    assert result.operator_on_augmentation.shape == (3, 0)


def test_device_augmented_seed_probe_rejects_nonfinite_augmentation_action() -> None:
    def setup_mv(x):
        return jnp.asarray(x, dtype=jnp.float64)

    state = setup_rhs1_qi_device_preconditioner(
        operator=setup_mv,
        total_size=3,
        coarse_basis=jnp.eye(3, dtype=jnp.float64),
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            regularization_rcond=1.0e-14,
        ),
    )
    call_count = 0

    def probe_mv(x):
        nonlocal call_count
        call_count += 1
        vector = jnp.asarray(x, dtype=jnp.float64)
        if call_count >= 3:
            return jnp.full_like(vector, jnp.nan)
        return vector

    rhs = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    result = probe_rhs1_qi_device_augmented_seed(
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        state=state,
        operator=probe_mv,
        max_cycles=1,
        max_rank=1,
    )

    assert result.probe.accepted is True
    assert result.reason == "residual_reduced_invalid_augmentation"
    assert result.rank == 0
    assert result.accepted_labels == ()
    assert result.augmentation_basis.shape == (3, 0)
    assert result.operator_on_augmentation.shape == (3, 0)
    assert result.projection_residual_norm is None


def test_device_preconditioner_rejects_mismatched_basis_shape() -> None:
    _, _, device_operator = _rank_one_coupled_operator(n=4, strength=1.0)

    with pytest.raises(ValueError, match="coarse basis must have shape"):
        setup_rhs1_qi_device_preconditioner(
            operator=device_operator,
            coarse_basis=jnp.ones((3, 1), dtype=jnp.float64),
        )
