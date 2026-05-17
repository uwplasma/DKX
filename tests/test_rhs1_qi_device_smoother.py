from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.rhs1_device_operator import device_csr_from_scipy_csr
from sfincs_jax.rhs1_qi_coarse import RHS1QICoarseBasis, RHS1QICoarseBasisMetadata
from sfincs_jax.rhs1_qi_device_smoother import (
    build_rhs1_qi_device_jacobi_smoother,
    extract_device_csr_diagonal,
)
from sfincs_jax.rhs1_qi_two_level import (
    build_rhs1_qi_two_level_preconditioner,
    probe_rhs1_qi_two_level_correction,
)


def _basis(vectors: jnp.ndarray, label: str = "global") -> RHS1QICoarseBasis:
    vectors = jnp.asarray(vectors, dtype=jnp.float64)
    labels = tuple(f"{label}:{i}" for i in range(int(vectors.shape[1])))
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
            rank_atol=1.0e-14,
        ),
    )


def test_device_jacobi_smoother_reuses_csr_operator_and_is_jittable() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [-1.0, 3.0, 0.5],
            [0.0, 0.25, 2.0],
        ],
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=0.6,
        sweeps=3,
    )
    residual = jnp.asarray([1.0, -0.5, 0.75], dtype=jnp.float64)

    correction = jnp.zeros_like(residual)
    remaining = residual
    diagonal = jnp.asarray(matrix.diagonal(), dtype=jnp.float64)
    dense = jnp.asarray(matrix.toarray(), dtype=jnp.float64)
    for _ in range(3):
        step = 0.6 * remaining / diagonal
        correction = correction + step
        remaining = remaining - dense @ step

    eager = smoother.apply(residual)
    compiled = jax.jit(smoother.as_preconditioner())(residual)

    assert smoother.metadata.device_resident is True
    assert smoother.metadata.reason == "built"
    assert smoother.metadata.valid_diagonal_count == 3
    np.testing.assert_allclose(smoother.diagonal, diagonal)
    np.testing.assert_allclose(eager, correction, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(compiled, eager, rtol=1.0e-12, atol=1.0e-12)


def test_device_jacobi_smoother_rejects_invalid_diagonal_by_default() -> None:
    matrix = sp.csr_matrix(
        [
            [0.0, 2.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)

    diagonal, hit_count = extract_device_csr_diagonal(device_operator)
    np.testing.assert_allclose(diagonal, jnp.asarray([0.0, 4.0], dtype=jnp.float64))
    np.testing.assert_array_equal(hit_count, jnp.asarray([0, 1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="invalid diagonal"):
        build_rhs1_qi_device_jacobi_smoother(device_operator)

    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=0.5,
        require_all_diagonal=False,
    )

    assert smoother.metadata.reason == "partial_diagonal"
    assert smoother.metadata.missing_diagonal_count == 1
    np.testing.assert_allclose(
        smoother.apply(jnp.asarray([2.0, 8.0], dtype=jnp.float64)),
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
    )


def test_device_jacobi_smoother_feeds_two_level_qi_probe() -> None:
    n = 6
    diag = jnp.linspace(1.5, 3.0, n, dtype=jnp.float64)
    global_mode = jnp.ones((n,), dtype=jnp.float64)
    global_mode = global_mode / jnp.linalg.norm(global_mode)
    dense = jnp.diag(diag) + 0.85 * jnp.outer(global_mode, global_mode)
    matrix = sp.csr_matrix(np.asarray(dense), dtype=np.float64)
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    smoother = build_rhs1_qi_device_jacobi_smoother(
        device_operator,
        damping=1.0,
        sweeps=2,
    )
    rhs = jnp.asarray([1.0, 0.4, -0.6, 0.8, -0.2, 0.5], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)
    basis = _basis(global_mode.reshape((-1, 1)))

    local_seed = smoother.apply(rhs)
    local_residual_norm = float(jnp.linalg.norm(rhs - device_operator.matvec(local_seed)))
    preconditioner = build_rhs1_qi_two_level_preconditioner(
        operator=device_operator.matvec,
        local_smoother=smoother.apply,
        basis=basis,
        coarse_solver="action_lstsq",
    )
    x, probe = probe_rhs1_qi_two_level_correction(
        operator=device_operator.matvec,
        rhs=rhs,
        x0=x0,
        preconditioner=preconditioner,
        min_relative_improvement=0.05,
    )

    assert probe.accepted is True
    assert probe.residual_after_norm < 0.75 * local_residual_norm
    assert probe.metadata.rank == 1
    np.testing.assert_allclose(
        jnp.linalg.norm(rhs - device_operator.matvec(x)),
        probe.residual_after_norm,
    )
