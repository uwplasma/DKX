from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.solvers.preconditioner_qi_basis import RHS1QICoarseBlockLayout
from sfincs_jax.solvers.preconditioner_qi_device import (
    RHS1QIDevicePreconditionerConfig,
    probe_rhs1_qi_device_preconditioner,
    setup_rhs1_qi_device_preconditioner,
)
from sfincs_jax.solvers.preconditioner_qi_corrections import (
    RHS1QIMultilevelCoarseConfig,
    RHS1QIMultilevelCoarseLevelMetadata,
    RHS1QIMultilevelCoarseMetadata,
    RHS1QIMultilevelCoarsePreconditioner,
    RHS1QIMultilevelCoarseProbe,
    build_rhs1_qi_multilevel_coarse_basis,
    build_rhs1_qi_multilevel_coarse_candidates,
    build_rhs1_qi_multilevel_coarse_preconditioner,
    build_rhs1_qi_multilevel_residual_level_bases,
    probe_rhs1_qi_multilevel_coarse_correction,
)


def _angular_radial_layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(8, 8, 8, 8),
        n_theta=4,
        n_zeta=1,
        block_x=(0, 1, 2, 3),
    )


def _angular_pitch_layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1, 2),
    )


def _current_tail_layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(8, 12, 8, 12, 3),
        n_theta=2,
        n_zeta=2,
        block_x=(0, 1, 0, 1, -1),
        block_species=(0, 0, 1, 1, -1),
    )


def _two_stage_galerkin_counterexample() -> tuple[
    RHS1QICoarseBlockLayout,
    RHS1QIMultilevelCoarseConfig,
    jnp.ndarray,
    jnp.ndarray,
]:
    layout = RHS1QICoarseBlockLayout(
        block_sizes=(1, 1, 1, 1),
        n_theta=1,
        n_zeta=1,
        block_x=(0, 1, 2, 3),
    )
    config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        aggregate_factor=2,
        max_rank=0,
        max_angular_mode=0,
        max_radial_degree=0,
        include_angular=False,
        include_radial=False,
        include_radial_angular=False,
        include_pitch=False,
        include_radial_pitch=False,
        nested_residual_correction=True,
        nested_level_max_rank=1,
        nested_order="coarse_to_fine",
        nested_include_global=False,
        regularization_rcond=0.0,
    )
    bases, _ = build_rhs1_qi_multilevel_residual_level_bases(layout, config=config)
    assert len(bases) == 2
    q1 = jnp.asarray(bases[0].vectors[:, 0], dtype=jnp.float64)
    q2 = jnp.asarray(bases[1].vectors[:, 0], dtype=jnp.float64)

    y2 = q2 - q1 * jnp.vdot(q1, q2)
    y2 = y2 / jnp.linalg.norm(y2)
    y1 = q1 + y2

    complement: list[jnp.ndarray] = []
    span = [q1, y2]
    for index in range(layout.total_size):
        candidate = jnp.eye(layout.total_size, dtype=jnp.float64)[:, index]
        for vector in (*span, *complement):
            candidate = candidate - vector * jnp.vdot(vector, candidate)
        norm = float(jnp.linalg.norm(candidate))
        if norm > 1.0e-12:
            complement.append(candidate / norm)

    domain_basis = jnp.stack((q1, q2, *complement), axis=1)
    image_basis = jnp.stack((y1, y2, *complement), axis=1)
    operator_matrix = image_basis @ jnp.linalg.inv(domain_basis)
    mode = q1 + q2
    return layout, config, operator_matrix, mode


def test_multilevel_coarse_candidates_are_deterministic_and_hierarchical() -> None:
    layout = _angular_radial_layout()
    config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=18,
        max_angular_mode=1,
        max_radial_degree=1,
    )

    candidates, labels, levels = build_rhs1_qi_multilevel_coarse_candidates(layout, config=config)
    repeated_candidates, repeated_labels, repeated_levels = build_rhs1_qi_multilevel_coarse_candidates(
        layout,
        config=config,
    )
    basis, basis_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=config)

    assert candidates.shape == (layout.total_size, len(labels))
    assert labels == repeated_labels
    assert len(levels) == 3
    assert all(isinstance(level, RHS1QIMultilevelCoarseLevelMetadata) for level in levels)
    assert tuple(level.aggregate_size for level in levels) == (1, 2, 4)
    assert levels[1].block_groups == ((0, 1), (2, 3))
    assert levels[2].block_groups == ((0, 1, 2, 3),)
    assert "level:0:radial:p1:angular:theta_cos1" in labels
    assert "level:1:aggregate:0:angular:theta_cos1" in labels
    assert basis.metadata.rank <= config.max_rank
    assert basis.metadata.candidate_count == len(labels)
    assert basis_levels[0].rank > 0
    assert levels[0].accepted_labels == repeated_levels[0].accepted_labels
    np.testing.assert_allclose(candidates, repeated_candidates, atol=0.0)


def test_multilevel_coarse_reduces_coupled_low_frequency_error_missed_by_local_blocks() -> None:
    layout = _angular_radial_layout()
    config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=20,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=config)
    basis, levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=config)
    mode = candidates[:, labels.index("level:0:radial:p1:angular:theta_cos1")]
    mode = mode / jnp.linalg.norm(mode)
    block_diagonal = 2.0 * jnp.eye(layout.total_size, dtype=jnp.float64)
    operator_matrix = block_diagonal + 4.0 * jnp.outer(mode, mode)
    rhs = operator_matrix @ mode
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return operator_matrix @ x

    def exact_local_block_smoother(r):
        return 0.5 * r

    local_seed = exact_local_block_smoother(rhs)
    local_residual_norm = float(jnp.linalg.norm(rhs - operator(local_seed)))
    preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=basis,
        level_metadata=levels,
        local_smoother=exact_local_block_smoother,
        config=config,
    )
    solution, probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=0.1,
    )

    assert isinstance(preconditioner, RHS1QIMultilevelCoarsePreconditioner)
    assert isinstance(preconditioner.metadata, RHS1QIMultilevelCoarseMetadata)
    assert isinstance(probe, RHS1QIMultilevelCoarseProbe)
    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.metadata.reason == "built_with_multilevel_coarse"
    assert probe.metadata.level_count == 3
    assert probe.metadata.device_resident is True
    assert probe.metadata.host_callback_free is True
    assert "level:0:radial:p1:angular:theta_cos1" in probe.metadata.candidate_labels
    assert local_residual_norm > 1.0
    assert probe.residual_after_norm < local_residual_norm * 1.0e-10
    assert probe.residual_after_norm < probe.residual_before_norm * 1.0e-10
    np.testing.assert_allclose(solution, mode, rtol=1.0e-10, atol=1.0e-10)


def test_nested_residual_equation_recovers_mode_discarded_by_flat_rank_gate() -> None:
    layout = _angular_radial_layout()
    flat_config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=1,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
    )
    nested_config = RHS1QIMultilevelCoarseConfig(
        max_levels=3,
        aggregate_factor=2,
        max_rank=1,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
        nested_residual_correction=True,
        nested_level_max_rank=64,
        nested_order="coarse_to_fine",
        nested_include_global=True,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=nested_config)
    flat_basis, flat_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=flat_config)
    target_label = next(
        label
        for label in labels
        if ":angular:" in label and label not in flat_basis.metadata.accepted_labels
    )
    mode = candidates[:, labels.index(target_label)]
    mode = mode / jnp.linalg.norm(mode)
    operator_matrix = jnp.eye(layout.total_size, dtype=jnp.float64)
    rhs = operator_matrix @ mode

    def operator(x):
        return operator_matrix @ x

    flat_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        layout=layout,
        basis=flat_basis,
        level_metadata=flat_levels,
        config=flat_config,
    )
    nested_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        layout=layout,
        basis=flat_basis,
        level_metadata=flat_levels,
        config=nested_config,
    )
    _, flat_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=flat_preconditioner,
        min_relative_improvement=0.0,
    )
    nested_solution, nested_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=nested_preconditioner,
        min_relative_improvement=0.05,
    )

    assert flat_probe.residual_after_norm > 1.0e-2
    assert nested_probe.accepted is True
    assert nested_probe.metadata.reason == "built_with_nested_residual_equation"
    assert nested_probe.metadata.nested_residual_correction_enabled is True
    assert nested_probe.metadata.nested_level_count == 3
    assert nested_probe.metadata.nested_rank > nested_probe.metadata.rank
    assert nested_probe.metadata.nested_level_ranks == (3, 6, 12)
    assert nested_probe.metadata.nested_order == "coarse_to_fine"
    assert nested_probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(nested_solution, mode, rtol=1.0e-10, atol=1.0e-10)


def test_nested_galerkin_residual_equation_recovers_mode_missed_by_action_staging() -> None:
    layout, action_config, operator_matrix, mode = _two_stage_galerkin_counterexample()
    galerkin_config = replace(action_config, nested_solver="galerkin")
    rhs = operator_matrix @ mode

    def operator(x):
        return operator_matrix @ x

    action_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        layout=layout,
        config=action_config,
    )
    galerkin_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        layout=layout,
        config=galerkin_config,
    )

    action_solution, action_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=action_preconditioner,
        min_relative_improvement=0.0,
    )
    galerkin_solution, galerkin_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=galerkin_preconditioner,
        min_relative_improvement=0.05,
    )
    compiled = jax.jit(galerkin_preconditioner.as_preconditioner())(rhs)

    assert action_probe.accepted is True
    assert action_probe.residual_after_norm > 0.25
    assert float(jnp.linalg.norm(operator_matrix @ action_solution - rhs)) > 0.25
    assert galerkin_probe.accepted is True
    assert galerkin_probe.metadata.reason == "built_with_nested_galerkin_residual_equation"
    assert galerkin_probe.metadata.nested_solver == "galerkin"
    assert galerkin_probe.metadata.nested_coarse_operator_shapes == ((1, 1), (1, 1))
    assert galerkin_probe.residual_after_norm < 1.0e-10
    np.testing.assert_allclose(compiled, galerkin_preconditioner.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(galerkin_solution, mode, rtol=1.0e-10, atol=1.0e-10)


def test_device_multilevel_galerkin_residual_equation_uses_cached_projected_stages() -> None:
    layout, _action_config, operator_matrix, mode = _two_stage_galerkin_counterexample()
    rhs = operator_matrix @ mode
    x0 = jnp.zeros_like(rhs)
    geometry_metadata = {
        "qi_block_sizes": layout.block_sizes,
        "qi_block_x": layout.block_x,
        "n_theta": layout.n_theta,
        "n_zeta": layout.n_zeta,
    }

    def mv(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    action_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            max_rank=0,
            multilevel_residual_equation=True,
            multilevel_max_levels=2,
            multilevel_aggregate_factor=2,
            multilevel_max_angular_mode=0,
            multilevel_max_radial_degree=0,
            multilevel_residual_equation_max_level_rank=1,
            multilevel_residual_equation_include_global=False,
            regularization_rcond=0.0,
        ),
    )
    galerkin_state = setup_rhs1_qi_device_preconditioner(
        operator=mv,
        total_size=layout.total_size,
        coarse_basis=None,
        geometry_metadata=geometry_metadata,
        config=RHS1QIDevicePreconditionerConfig(
            local_smoother_kind="none",
            max_rank=0,
            multilevel_residual_equation=True,
            multilevel_max_levels=2,
            multilevel_aggregate_factor=2,
            multilevel_max_angular_mode=0,
            multilevel_max_radial_degree=0,
            multilevel_residual_equation_max_level_rank=1,
            multilevel_residual_equation_solver="galerkin",
            multilevel_residual_equation_include_global=False,
            regularization_rcond=0.0,
        ),
    )

    _, action_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=action_state,
        min_relative_improvement=0.0,
    )
    x, galerkin_probe = probe_rhs1_qi_device_preconditioner(
        rhs=rhs,
        x0=x0,
        state=galerkin_state,
        min_relative_improvement=0.05,
    )
    compiled = jax.jit(galerkin_state.as_preconditioner())(rhs)

    assert action_probe.residual_after_norm > 0.25
    assert galerkin_probe.accepted is True
    assert galerkin_probe.residual_after_norm < 1.0e-10
    assert galerkin_state.metadata.reason == "built_with_multilevel_galerkin_residual_equation"
    assert galerkin_state.metadata.multilevel_residual_equation_solver == "galerkin"
    assert galerkin_state.residual_equation_stage_solvers == ("galerkin", "galerkin")
    assert tuple(operator.shape for operator in galerkin_state.residual_equation_coarse_operators) == ((1, 1), (1, 1))
    np.testing.assert_allclose(compiled, galerkin_state.apply(rhs), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(operator_matrix @ x, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_multilevel_coarse_without_angular_space_cannot_fix_angular_radial_mode() -> None:
    layout = _angular_radial_layout()
    full_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        max_rank=16,
        max_angular_mode=1,
        max_radial_degree=1,
        regularization_rcond=1.0e-14,
    )
    radial_only_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        max_rank=16,
        max_angular_mode=0,
        max_radial_degree=1,
        include_angular=False,
        include_radial_angular=False,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=full_config)
    mode = candidates[:, labels.index("level:0:radial:p1:angular:theta_cos1")]
    mode = mode / jnp.linalg.norm(mode)
    operator_matrix = jnp.eye(layout.total_size, dtype=jnp.float64) + 3.0 * jnp.outer(mode, mode)
    rhs = operator_matrix @ mode

    def operator(x):
        return operator_matrix @ x

    def zero_local_smoother(r):
        return jnp.zeros_like(r)

    radial_basis, radial_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=radial_only_config)
    preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=radial_basis,
        level_metadata=radial_levels,
        local_smoother=zero_local_smoother,
        config=radial_only_config,
    )
    solution, probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=preconditioner,
    )

    assert probe.accepted is False
    assert probe.reason == "not_reduced"
    assert "level:0:radial:p1:angular:theta_cos1" not in radial_basis.metadata.accepted_labels
    assert probe.residual_after_norm == pytest.approx(probe.residual_before_norm)
    np.testing.assert_allclose(solution, jnp.zeros_like(rhs), atol=0.0)


def test_multilevel_coarse_pitch_moment_reduces_pitch_coupled_error() -> None:
    layout = _angular_pitch_layout()
    pitch_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        aggregate_factor=2,
        max_rank=20,
        max_angular_mode=0,
        max_radial_degree=1,
        max_pitch_degree=1,
        include_angular=False,
        include_radial_angular=False,
        regularization_rcond=1.0e-14,
    )
    no_pitch_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        aggregate_factor=2,
        max_rank=20,
        max_angular_mode=0,
        max_radial_degree=1,
        max_pitch_degree=0,
        include_angular=False,
        include_radial_angular=False,
        include_pitch=False,
        include_radial_pitch=False,
        regularization_rcond=1.0e-14,
    )
    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=pitch_config)
    pitch_label = "level:0:aggregate:0:pitch:p1"
    mode = candidates[:, labels.index(pitch_label)]
    mode = mode / jnp.linalg.norm(mode)
    operator_matrix = 1.4 * jnp.eye(layout.total_size, dtype=jnp.float64) + 3.5 * jnp.outer(mode, mode)
    rhs = operator_matrix @ mode

    def operator(x):
        return operator_matrix @ x

    def zero_local_smoother(r):
        return jnp.zeros_like(r)

    pitch_basis, pitch_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=pitch_config)
    pitch_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=pitch_basis,
        level_metadata=pitch_levels,
        local_smoother=zero_local_smoother,
        config=pitch_config,
    )
    pitch_solution, pitch_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=pitch_preconditioner,
        min_relative_improvement=0.1,
    )
    no_pitch_basis, no_pitch_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=no_pitch_config)
    no_pitch_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=no_pitch_basis,
        level_metadata=no_pitch_levels,
        local_smoother=zero_local_smoother,
        config=no_pitch_config,
    )
    _, no_pitch_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=no_pitch_preconditioner,
        min_relative_improvement=0.0,
    )

    assert pitch_label in labels
    assert pitch_label in pitch_basis.metadata.accepted_labels
    assert pitch_label not in no_pitch_basis.metadata.accepted_labels
    assert pitch_probe.accepted is True
    assert pitch_probe.residual_after_norm < 1.0e-10
    assert no_pitch_probe.accepted is False
    assert no_pitch_probe.residual_after_norm == pytest.approx(no_pitch_probe.residual_before_norm)
    np.testing.assert_allclose(pitch_solution, mode, rtol=1.0e-10, atol=1.0e-10)


def test_multilevel_current_constraint_family_prioritizes_flow_and_tail_modes() -> None:
    layout = _current_tail_layout()
    current_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        aggregate_factor=2,
        max_rank=18,
        max_angular_mode=0,
        max_radial_degree=1,
        max_pitch_degree=0,
        include_angular=False,
        include_radial_angular=False,
        include_pitch=False,
        include_radial_pitch=False,
        include_current_moments=True,
        max_current_pitch_degree=1,
        include_tail_constraint_moments=True,
        regularization_rcond=1.0e-14,
    )
    no_current_config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        aggregate_factor=2,
        max_rank=18,
        max_angular_mode=0,
        max_radial_degree=1,
        max_pitch_degree=0,
        include_angular=False,
        include_radial_angular=False,
        include_pitch=False,
        include_radial_pitch=False,
        include_current_moments=False,
        include_tail_constraint_moments=False,
        regularization_rcond=1.0e-14,
    )

    candidates, labels, _ = build_rhs1_qi_multilevel_coarse_candidates(layout, config=current_config)
    current_label = "current:species:0:radial:p1:pitch:p1"
    tail_label = "constraint_tail:aggregate"
    current_mode = candidates[:, labels.index(current_label)]
    tail_mode = candidates[:, labels.index(tail_label)]
    mode = current_mode / jnp.linalg.norm(current_mode) + 0.25 * tail_mode / jnp.linalg.norm(tail_mode)
    mode = mode / jnp.linalg.norm(mode)
    operator_matrix = 1.2 * jnp.eye(layout.total_size, dtype=jnp.float64) + 4.0 * jnp.outer(mode, mode)
    rhs = operator_matrix @ mode

    def operator(x):
        return operator_matrix @ x

    def zero_local_smoother(r):
        return jnp.zeros_like(r)

    current_basis, current_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=current_config)
    current_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=current_basis,
        level_metadata=current_levels,
        local_smoother=zero_local_smoother,
        config=current_config,
    )
    current_solution, current_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=current_preconditioner,
        min_relative_improvement=0.1,
    )
    no_current_basis, no_current_levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=no_current_config)
    no_current_preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=no_current_basis,
        level_metadata=no_current_levels,
        local_smoother=zero_local_smoother,
        config=no_current_config,
    )
    _, no_current_probe = probe_rhs1_qi_multilevel_coarse_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=no_current_preconditioner,
        min_relative_improvement=0.0,
    )

    assert current_label in labels
    assert tail_label in labels
    assert current_label in current_basis.metadata.accepted_labels
    assert tail_label in current_basis.metadata.accepted_labels
    assert current_probe.accepted is True
    assert current_probe.residual_after_norm < 1.0e-10
    assert current_label not in no_current_basis.metadata.accepted_labels
    assert tail_label not in no_current_basis.metadata.accepted_labels
    assert no_current_probe.residual_after_norm > 1.0e-2
    np.testing.assert_allclose(current_solution, mode, rtol=1.0e-10, atol=1.0e-10)


def test_multilevel_coarse_action_is_jittable_and_differentiable() -> None:
    layout = _angular_radial_layout()
    config = RHS1QIMultilevelCoarseConfig(
        max_levels=2,
        max_rank=12,
        max_angular_mode=1,
        max_radial_degree=1,
    )
    basis, levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=config)
    q = basis.vectors
    operator_matrix = 1.5 * jnp.eye(layout.total_size, dtype=jnp.float64) + 0.35 * (q[:, :4] @ q[:, :4].T)

    def operator(x):
        return operator_matrix @ x

    def local_smoother(r):
        return r / 1.5

    preconditioner = build_rhs1_qi_multilevel_coarse_preconditioner(
        operator=operator,
        basis=basis,
        level_metadata=levels,
        local_smoother=local_smoother,
        config=config,
    )
    residual = jnp.linspace(-1.0, 1.0, layout.total_size, dtype=jnp.float64)

    eager = preconditioner.apply(residual)
    compiled = jax.jit(preconditioner.as_preconditioner())(residual)
    cotangent = jnp.linspace(0.5, 1.5, layout.total_size, dtype=jnp.float64)
    transpose_action = jax.vjp(preconditioner.as_preconditioner(), residual)[1](cotangent)[0]

    def squared_action_norm(value):
        corrected = preconditioner.apply(value)
        return jnp.vdot(corrected, corrected)

    gradient = jax.grad(squared_action_norm)(residual)

    np.testing.assert_allclose(compiled, eager, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        jnp.vdot(eager, cotangent),
        jnp.vdot(residual, transpose_action),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert gradient.shape == residual.shape
    assert transpose_action.shape == residual.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(transpose_action)))
