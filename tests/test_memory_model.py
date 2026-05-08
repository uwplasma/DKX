from __future__ import annotations

import numpy as np

from sfincs_jax.memory_model import (
    csr_matrix_nbytes,
    dense_matrix_nbytes,
    estimate_linear_solve_memory,
    estimate_sparse_pc_memory,
    gmres_basis_nbytes,
    gmres_restart_for_budget,
)


def test_dense_and_csr_estimates_are_byte_exact() -> None:
    assert dense_matrix_nbytes(10, dtype=np.float64) == 10 * 10 * 8
    assert dense_matrix_nbytes(10, 3, dtype=np.float32) == 10 * 3 * 4
    assert csr_matrix_nbytes(10, 25, data_dtype=np.float64, index_dtype=np.int32) == 25 * (8 + 4) + 11 * 4


def test_gmres_restart_budget_counts_work_vectors() -> None:
    n = 1_000
    # restart=50 would need (50 + 1 + 4) * 1000 * 8 bytes.
    assert gmres_basis_nbytes(n, 50, dtype=np.float64) == 55 * 1_000 * 8
    max_bytes = 15 * n * 8
    assert gmres_restart_for_budget(n, 50, dtype=np.float64, max_bytes=max_bytes) == 10


def test_linear_solve_memory_estimate_reports_per_device_totals() -> None:
    estimate = estimate_linear_solve_memory(
        unknowns=100,
        gmres_restart=10,
        csr_nnz=400,
        preconditioner_nbytes=1_000,
        compiled_temp_nbytes=2_000,
        device_count=2,
    )
    data = estimate.to_dict()
    assert data["dense_operator_nbytes"] == 100 * 100 * 8
    assert data["csr_operator_nbytes"] == 400 * (8 + 4) + 101 * 4
    assert data["dense_total_nbytes"] == estimate.dense_total_nbytes
    assert data["dense_per_device_nbytes"] == estimate.dense_per_device_nbytes
    assert estimate.csr_per_device_nbytes is not None


def test_sparse_pc_memory_estimate_includes_factor_fill() -> None:
    estimate = estimate_sparse_pc_memory(
        unknowns=100,
        gmres_restart=20,
        csr_nnz=500,
        factor_fill_estimate=3.0,
    )

    assert estimate.csr_operator_nbytes == csr_matrix_nbytes(100, 500)
    assert estimate.preconditioner_nbytes == 3 * estimate.csr_operator_nbytes
    assert estimate.csr_total_nbytes == (
        estimate.csr_operator_nbytes + estimate.gmres_basis_nbytes + estimate.preconditioner_nbytes
    )
