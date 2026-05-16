from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_qi_block_schur import (
    build_rhs1_qi_block_schur_basis,
    build_rhs1_qi_block_schur_candidates,
    build_rhs1_qi_block_schur_preconditioner,
    probe_rhs1_qi_block_schur_correction,
)
from sfincs_jax.rhs1_qi_coarse import RHS1QICoarseBlockLayout


def _coupled_layout() -> RHS1QICoarseBlockLayout:
    return RHS1QICoarseBlockLayout(
        block_sizes=(12, 12, 12, 12),
        n_theta=3,
        n_zeta=2,
        block_x=(0, 1, 2, 3),
    )


def test_block_schur_qi_reduces_coupled_angular_radial_global_residual() -> None:
    layout = _coupled_layout()
    basis = build_rhs1_qi_block_schur_basis(
        layout,
        max_candidates=96,
        max_rank=28,
        max_angular_mode=1,
        max_radial_degree=2,
    )
    n = layout.total_size
    diag = jnp.linspace(1.2, 2.6, n, dtype=jnp.float64)
    q = basis.vectors
    coupling_cols = q[:, : min(10, int(q.shape[1]))]
    coupling = 0.55 * (coupling_cols @ coupling_cols.T)
    a = jnp.diag(diag) + coupling
    coefficients = jnp.linspace(0.4, -0.25, int(q.shape[1]), dtype=jnp.float64)
    exact = q @ coefficients
    rhs = a @ exact

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / diag

    local_seed = local_smoother(rhs)
    local_residual_norm = float(jnp.linalg.norm(rhs - operator(local_seed)))
    preconditioner = build_rhs1_qi_block_schur_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
        regularization_rcond=1.0e-11,
    )
    solution, probe = probe_rhs1_qi_block_schur_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=preconditioner,
    )

    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.metadata.rank == basis.metadata.rank
    assert probe.metadata.stable_rank > 1.0
    assert probe.residual_after_norm < local_residual_norm * 5.0e-2
    assert probe.residual_after_norm < probe.residual_before_norm * 1.0e-2
    assert float(jnp.linalg.norm(rhs - operator(solution))) == pytest.approx(probe.residual_after_norm)


def test_block_schur_qi_rejects_when_no_true_residual_reduction_is_possible() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(2, 2), n_theta=1, n_zeta=1, block_x=(0, 1))
    basis = build_rhs1_qi_block_schur_basis(
        layout,
        max_candidates=1,
        max_rank=1,
        include_radial=False,
        include_angular=False,
        include_radial_angular=False,
        include_block_schur=False,
        include_block_schur_angular=False,
        include_blocks=False,
    )
    a = jnp.eye(4, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)

    def operator(x):
        return a @ x

    preconditioner = build_rhs1_qi_block_schur_preconditioner(operator=operator, basis=basis)
    solution, probe = probe_rhs1_qi_block_schur_correction(
        operator=operator,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        preconditioner=preconditioner,
    )

    assert basis.metadata.candidate_labels == ("global",)
    assert basis.metadata.rank == 1
    assert probe.accepted is False
    assert probe.reason == "not_reduced"
    assert probe.residual_after_norm == pytest.approx(probe.residual_before_norm)
    np.testing.assert_allclose(solution, jnp.zeros_like(rhs), atol=0.0)


def test_block_schur_qi_metadata_has_stable_rank_and_conditioning() -> None:
    layout = _coupled_layout()
    candidates, labels = build_rhs1_qi_block_schur_candidates(
        layout,
        max_candidates=80,
        max_angular_mode=1,
        max_radial_degree=2,
    )
    basis = build_rhs1_qi_block_schur_basis(
        layout,
        max_candidates=80,
        max_rank=24,
        max_angular_mode=1,
        max_radial_degree=2,
    )
    repeated = build_rhs1_qi_block_schur_basis(
        layout,
        max_candidates=80,
        max_rank=24,
        max_angular_mode=1,
        max_radial_degree=2,
    )
    diag = jnp.linspace(1.0, 1.8, layout.total_size, dtype=jnp.float64)

    def operator(x):
        return diag * x

    preconditioner = build_rhs1_qi_block_schur_preconditioner(
        operator=operator,
        basis=basis,
        regularization_rcond=1.0e-10,
    )
    repeated_preconditioner = build_rhs1_qi_block_schur_preconditioner(
        operator=operator,
        basis=repeated,
        regularization_rcond=1.0e-10,
    )
    metadata = preconditioner.metadata

    assert candidates.shape == (layout.total_size, len(labels))
    assert "radial:p1*angular:theta_cos1" in labels
    assert "schur:x_diff:0->1*angular:theta_cos1" in labels
    assert metadata.candidate_count == len(labels)
    assert 0 < metadata.rank <= 24
    assert metadata.numerical_rank == metadata.rank
    assert metadata.discarded_count == metadata.candidate_count - metadata.rank
    assert metadata.condition_estimate >= 1.0
    assert metadata.stable_rank > 1.0
    assert np.isfinite(metadata.condition_estimate)
    assert metadata.accepted_labels == repeated_preconditioner.metadata.accepted_labels
    assert metadata.condition_estimate == pytest.approx(repeated_preconditioner.metadata.condition_estimate)


def test_block_schur_qi_action_is_jittable_and_differentiable() -> None:
    layout = RHS1QICoarseBlockLayout(block_sizes=(6, 6, 6), n_theta=3, n_zeta=2, block_x=(0, 1, 2))
    basis = build_rhs1_qi_block_schur_basis(
        layout,
        max_candidates=40,
        max_rank=12,
        max_angular_mode=1,
        max_radial_degree=1,
    )
    diag = jnp.linspace(1.0, 2.0, layout.total_size, dtype=jnp.float64)
    q = basis.vectors
    a = jnp.diag(diag) + 0.2 * (q[:, :3] @ q[:, :3].T)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / diag

    preconditioner = build_rhs1_qi_block_schur_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
    )
    residual = jnp.linspace(-1.0, 1.0, layout.total_size, dtype=jnp.float64)

    eager = preconditioner.apply(residual)
    compiled = jax.jit(preconditioner.apply)(residual)

    def squared_action_norm(r):
        value = preconditioner.apply(r)
        return jnp.vdot(value, value)

    gradient = jax.grad(squared_action_norm)(residual)

    np.testing.assert_allclose(compiled, eager, rtol=2.0e-5, atol=2.0e-5)
    assert gradient.shape == residual.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
