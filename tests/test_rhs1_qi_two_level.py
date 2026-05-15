from __future__ import annotations

import jax
import jax.numpy as jnp

from sfincs_jax.rhs1_qi_coarse import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata
from sfincs_jax.rhs1_qi_two_level import (
    build_rhs1_qi_two_level_preconditioner,
    probe_rhs1_qi_two_level_correction,
)


def _basis(vectors: jnp.ndarray) -> RHS1QICoarseBasis:
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    return RHS1QICoarseBasis(
        vectors=vectors,
        metadata=RHS1QICoarseBasisMetadata(
            total_size=int(vectors.shape[0]),
            candidate_count=int(vectors.shape[1]),
            rank=int(vectors.shape[1]),
            discarded_count=0,
            candidate_labels=("global",),
            accepted_labels=("global",),
            candidate_norms=(1.0,),
            accepted_norms=(1.0,),
            rank_rtol=1.0e-12,
            rank_atol=1.0e-14,
        ),
    )


def test_two_level_qi_preconditioner_reduces_low_rank_residual() -> None:
    u = jnp.ones((6,), dtype=jnp.float64)
    diag = jnp.linspace(1.0, 2.0, 6, dtype=jnp.float64)
    a = jnp.diag(diag) + 0.35 * jnp.outer(u, u)
    q = (u / jnp.linalg.norm(u)).reshape((-1, 1))
    rhs = jnp.arange(1, 7, dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / diag

    local_residual = rhs - operator(local_smoother(rhs))
    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=_basis(q),
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
    )

    assert probe.accepted is True
    assert probe.reason == "residual_reduced"
    assert probe.metadata.rank == 1
    assert probe.residual_after_norm < float(jnp.linalg.norm(local_residual))
    assert float(jnp.linalg.norm(rhs - operator(x))) == probe.residual_after_norm


def test_two_level_qi_preconditioner_action_is_jittable() -> None:
    a = jnp.asarray(
        [
            [3.0, 0.5, 0.0],
            [0.5, 2.0, 0.25],
            [0.0, 0.25, 1.5],
        ],
        dtype=jnp.float64,
    )
    q = jnp.asarray([[1.0], [1.0], [1.0]], dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)

    def operator(x):
        return a @ x

    def local_smoother(r):
        return r / jnp.diag(a)

    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=_basis(q),
    )
    residual = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)

    eager = preconditioner.apply(residual)
    compiled = jax.jit(preconditioner.apply)(residual)

    assert jnp.allclose(compiled, eager)


def test_two_level_qi_probe_fails_closed_without_required_improvement() -> None:
    a = jnp.eye(4, dtype=jnp.float64)
    q = jnp.ones((4, 1), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def operator(x):
        return a @ x

    def exact_local_smoother(r):
        return r

    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=operator,
        local_smoother=exact_local_smoother,
        basis=_basis(q),
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=operator,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=1.1,
    )

    assert probe.accepted is False
    assert probe.reason == "residual_not_reduced"
    assert jnp.allclose(x, x0)
