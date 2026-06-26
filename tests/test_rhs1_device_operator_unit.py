from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.solvers.explicit_sparse import estimate_csr_nbytes
from sfincs_jax.operators.profile_device_sparse import (
    assert_device_matvec_matches,
    device_csr_from_operator,
    device_csr_from_scipy_csr,
    estimate_device_csr_nbytes,
    materialized_operator_to_csr,
    validate_device_matvec,
)
from sfincs_jax.solvers.preconditioners.qi.basis import (
    RHS1QICoarseBlockLayout,
    build_rhs1_qi_galerkin_preconditioner,
    build_rhs1_qi_coarse_basis,
)


def test_device_csr_from_scipy_csr_exposes_arrays_and_jitted_matvec() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 0.0, 1.0],
            [0.0, -2.0, 0.0],
            [3.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )

    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)

    assert device_operator.shape == matrix.shape
    assert device_operator.metadata.nnz == matrix.nnz
    assert device_operator.metadata.csr_nbytes == estimate_csr_nbytes(matrix.shape, matrix.nnz)
    assert device_operator.metadata.csr_nbytes == estimate_device_csr_nbytes(matrix.shape, matrix.nnz)
    assert device_operator.metadata.row_indices_nbytes == 0
    assert device_operator.row_indices is None
    assert device_operator.indices.dtype == jnp.int32
    assert device_operator.metadata.default_backend == jax.default_backend()
    assert device_operator.metadata.array_devices
    assert device_operator.metadata.array_platforms
    assert device_operator.metadata.all_arrays_same_device is True
    assert device_operator.metadata.array_platforms[0] in device_operator.metadata.available_platforms
    metadata_dict = device_operator.metadata.to_dict()
    assert metadata_dict["array_devices"] == device_operator.metadata.array_devices
    assert metadata_dict["array_platforms"] == device_operator.metadata.array_platforms
    assert metadata_dict["all_arrays_same_device"] is True

    x = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(device_operator.matvec(x)), matrix @ np.asarray(x))
    np.testing.assert_allclose(np.asarray(jax.jit(device_operator.matvec)(x)), matrix @ np.asarray(x))


def test_device_csr_from_operator_slices_active_indices_without_dense_materialization() -> None:
    matrix = sp.csr_matrix(
        [
            [10.0, 1.0, 2.0, 0.0, 3.0],
            [4.0, 11.0, 5.0, 0.0, 0.0],
            [6.0, 0.0, 12.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 13.0, 0.0],
            [14.0, 0.0, 15.0, 0.0, 16.0],
        ],
        dtype=np.float64,
    )
    active = np.asarray([0, 2, 4], dtype=np.int32)
    expected = matrix[active, :][:, active].tocsr()

    device_operator = device_csr_from_operator(SimpleNamespace(matrix=matrix), active_indices=active, max_csr_mb=1.0)

    assert device_operator.shape == expected.shape
    assert device_operator.metadata.active_size == int(active.size)
    assert device_operator.metadata.source_shape == matrix.shape
    assert device_operator.metadata.active_mapping_nbytes == matrix.shape[1] * np.dtype(np.int32).itemsize
    np.testing.assert_array_equal(np.asarray(device_operator.active_indices), active)

    x = jnp.asarray([1.0, -1.0, 2.0], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(device_operator.matvec(x)), expected @ np.asarray(x))


def test_validate_device_matvec_and_budget_gate() -> None:
    matrix = sp.eye(4, format="csr", dtype=np.float64)
    needed = estimate_csr_nbytes(matrix.shape, matrix.nnz)
    removed_row_indices_nbytes = matrix.nnz * np.dtype(np.int32).itemsize
    assert estimate_device_csr_nbytes(matrix.shape, matrix.nnz) == needed
    assert needed + removed_row_indices_nbytes > needed
    with pytest.raises(MemoryError, match="device CSR operator exceeds memory budget"):
        device_csr_from_scipy_csr(matrix, max_csr_nbytes=needed - 1)

    device_operator = device_csr_from_scipy_csr(matrix, max_csr_nbytes=needed)
    result = validate_device_matvec(
        device_operator,
        lambda x: jnp.asarray(x, dtype=jnp.float64),
        probes=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        samples=1,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert result.passed
    assert_device_matvec_matches(
        device_operator,
        lambda x: jnp.asarray(x, dtype=jnp.float64),
        samples=1,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_materialized_operator_to_csr_rejects_operator_only_and_dense_by_default() -> None:
    with pytest.raises(ValueError, match="materialized matrix"):
        materialized_operator_to_csr(SimpleNamespace(matrix=None))

    with pytest.raises(TypeError, match="allow_dense=True"):
        materialized_operator_to_csr(np.eye(2))

    csr = materialized_operator_to_csr(np.eye(2), allow_dense=True)
    assert sp.isspmatrix_csr(csr)
    np.testing.assert_allclose(csr.toarray(), np.eye(2))


def test_device_csr_matvec_supports_jitted_qi_galerkin_preconditioner() -> None:
    matrix = sp.csr_matrix(
        np.diag([2.0, 2.0, 3.5, 3.5]),
        dtype=np.float64,
    )
    device_operator = device_csr_from_scipy_csr(matrix, max_csr_mb=1.0)
    layout = RHS1QICoarseBlockLayout(block_sizes=(2, 2), block_x=(0, 1))
    basis = build_rhs1_qi_coarse_basis(layout, include_angular=False)
    preconditioner = build_rhs1_qi_galerkin_preconditioner(device_operator.matvec, basis=basis)
    coefficients = jnp.asarray([0.75, -0.25], dtype=jnp.float64)[: basis.metadata.rank]
    exact = basis.vectors @ coefficients
    rhs = device_operator.matvec(exact)

    got = jax.jit(preconditioner.as_preconditioner())(rhs)

    assert preconditioner.metadata.rank == basis.metadata.rank
    assert preconditioner.metadata.coarse_operator_shape == (basis.metadata.rank, basis.metadata.rank)
    np.testing.assert_allclose(got, exact, atol=1.0e-10)
