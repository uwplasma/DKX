from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

import sfincs_jax.io as io_module
import sfincs_jax.v3_driver as v3_driver_module
from sfincs_jax.explicit_sparse import SparseDecision, SparseOperatorBundle, build_operator_from_pattern
from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij
from sfincs_jax.rhs1_block_operator import RHS1BlockLayout
from sfincs_jax.rhs1_xblock_policy import resolve_rhs1_xblock_sparse_pc_policy
from sfincs_jax.solver import FlexibleGMRESSolveResult
from sfincs_jax.v3_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary,
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
)
from sfincs_jax.v3_driver import (
    _apply_device_subspace_residual_equation_correction,
    _rhs1_additive_rescue_nbytes,
    _rhs1_xblock_gmres_restart,
    _rhs1_xblock_precondition_side,
    _rhs1_xblock_post_coarse_directions,
    _triangular_solve_lower_csr_rows,
    _triangular_solve_upper_csr_rows,
    solve_v3_full_system_linear_gmres,
)
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _csr_from_petsc(path: Path) -> sp.csr_matrix:
    a = read_petsc_mat_aij(path)
    return sp.csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def _assert_pattern_covers_matrix(pattern: sp.spmatrix, matrix: sp.spmatrix) -> None:
    pattern_bool = pattern.tocsr().astype(bool)
    matrix_bool = matrix.tocsr().astype(bool)
    covered = matrix_bool.multiply(pattern_bool)
    missing = matrix_bool.astype(np.int8) - covered.astype(np.int8)
    missing.eliminate_zeros()
    assert missing.nnz == 0


def _fast_device_krylov_result(**kwargs):
    """Return a converged device-Krylov result for solver-path metadata tests."""

    b = jnp.asarray(kwargs["b"], dtype=jnp.float64)
    x = jnp.zeros_like(b)
    history = jnp.asarray([jnp.linalg.norm(b), 0.0], dtype=jnp.float64)
    return (
        FlexibleGMRESSolveResult(
            x=x,
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
            residual_history=history,
            n_iterations=jnp.asarray(1, dtype=jnp.int32),
            n_restarts=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(True),
        ),
        jnp.zeros_like(b),
    )


def _fast_device_cycle_krylov_result(**kwargs):
    """Return a converged cycle-JIT FGMRES result with many internal iterations."""

    b = jnp.asarray(kwargs["b"], dtype=jnp.float64)
    x = jnp.zeros_like(b)
    history = jnp.asarray([jnp.linalg.norm(b), 1.0e-6, 0.0], dtype=jnp.float64)
    return (
        FlexibleGMRESSolveResult(
            x=x,
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
            residual_history=history,
            n_iterations=jnp.asarray(80, dtype=jnp.int32),
            n_restarts=jnp.asarray(1, dtype=jnp.int32),
            converged=jnp.asarray(True),
        ),
        jnp.zeros_like(b),
    )


def test_rhs1_additive_rescue_nbytes_treats_cap_as_incremental_budget() -> None:
    factor = SimpleNamespace(factor_nbytes_estimate=5_000_000_000)

    assert _rhs1_additive_rescue_nbytes(factor, 512.0) == 5_000_000_000 + 512 * 1024 * 1024
    assert _rhs1_additive_rescue_nbytes(factor, 0.0) == 0


def test_residual_coarse_host_preconditioner_solves_adaptive_identity_residual() -> None:
    matrix = sp.eye(4, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit test",
            backend="cpu",
            shape=(4, 4),
            dense_nbytes=4 * 4 * 8,
            csr_nbytes_estimate=128,
            nnz_estimate=4,
        ),
    )

    class HalfFactor:
        kind = "half_identity"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 4
        factor_s = 0.0

        def solve(self, rhs):
            return 0.5 * np.asarray(rhs, dtype=np.float64)

    rhs = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    failed_residual = rhs - operator.matvec(HalfFactor().solve(rhs))

    bundle = v3_driver_module._try_build_residual_coarse_host_sparse_preconditioner(
        operator_bundle=operator,
        factor_bundle=HalfFactor(),
        residual=failed_residual,
        max_rank=2,
        max_nbytes=1024 * 1024,
        regularization=0.0,
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["rank"] == 1
    corrected = bundle.solve(rhs)
    np.testing.assert_allclose(operator.matvec(corrected), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_residual_window_host_preconditioner_solves_targeted_kinetic_window() -> None:
    matrix = sp.eye(4, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit test",
            backend="cpu",
            shape=(4, 4),
            dense_nbytes=4 * 4 * 8,
            csr_nbytes_estimate=128,
            nnz_estimate=4,
        ),
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class HalfFactor:
        kind = "half_identity"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 4
        factor_s = 0.0

        def solve(self, rhs):
            return 0.5 * np.asarray(rhs, dtype=np.float64)

    rhs = np.asarray([0.0, 2.0, 0.0, 0.0], dtype=np.float64)
    failed_residual = rhs - operator.matvec(HalfFactor().solve(rhs))

    bundle = v3_driver_module._try_build_residual_window_host_sparse_preconditioner(
        operator_bundle=operator,
        factor_bundle=HalfFactor(),
        residual=failed_residual,
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        coefficient_mode="least_squares",
        combine_mode="independent",
        interface_depth=0,
        max_window_size=16,
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["window_count"] == 1
    corrected = bundle.solve(rhs)
    np.testing.assert_allclose(operator.matvec(corrected), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_true_operator_residual_window_lsq_reduces_global_residual() -> None:
    matrix = sp.csr_matrix(
        [
            [1.0, 8.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -4.0, 1.0],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 0
        factor_s = 0.0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    rhs = np.asarray([1.0, 0.25, -0.5], dtype=np.float64)
    bundle = v3_driver_module._try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=1.0e-12,
        max_window_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        explicit_specs=((0, 0, 1),),
        damping=True,
        beta_max=0.0,
    )

    assert bundle is not None
    corrected = bundle.solve(rhs)
    assert np.linalg.norm(rhs - matrix @ corrected) < np.linalg.norm(rhs)
    assert bundle.metadata is not None
    assert bundle.metadata["window_size"] == 1
    assert bundle.metadata["a_window_nnz"] == 3


def test_true_operator_residual_window_lsq_is_linear_without_damping() -> None:
    matrix = sp.eye(3, format="csr", dtype=np.float64)
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    bundle = v3_driver_module._try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=np.asarray([1.0, -2.0, 0.5], dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=1,
        max_nbytes=1024 * 1024,
        regularization=1.0e-12,
        max_window_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        explicit_specs=((0, 0, 1),),
        damping=False,
        beta_max=10.0,
    )

    assert bundle is not None
    lhs = bundle.solve(np.asarray([1.0, 2.0, 3.0], dtype=np.float64)) + bundle.solve(
        np.asarray([-0.5, 0.25, 1.0], dtype=np.float64)
    )
    rhs = bundle.solve(np.asarray([0.5, 2.25, 4.0], dtype=np.float64))
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)
    assert bundle.metadata is not None
    assert bundle.metadata["damping"] is False


def test_true_operator_active_block_lsq_solves_deterministic_active_block() -> None:
    matrix = sp.csr_matrix(
        [
            [2.0, -0.5, 0.25],
            [0.1, 1.8, -0.4],
            [-0.2, 0.3, 1.6],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 0
        factor_s = 0.0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    rhs = np.asarray([1.0, -0.25, 0.5], dtype=np.float64)
    bundle = v3_driver_module._try_build_true_operator_active_block_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        x_count=1,
        ell_count=3,
        max_nbytes=1024 * 1024,
        regularization=1.0e-14,
        max_block_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        max_tail=0,
    )

    assert bundle is not None
    corrected = bundle.solve(rhs)
    assert np.linalg.norm(rhs - matrix @ corrected) < 1.0e-10
    assert bundle.metadata is not None
    assert bundle.metadata["block_size"] == 3
    assert bundle.metadata["kinetic_selected"] == 3
    assert bundle.metadata["tail_selected"] == 0


def test_true_operator_active_residual_block_lsq_solves_dominant_true_residual() -> None:
    matrix = sp.csr_matrix(
        [
            [2.0, -0.5, 0.25, 0.0],
            [0.1, 1.8, -0.4, 0.2],
            [-0.2, 0.3, 1.6, -0.1],
            [0.0, 0.2, -0.3, 1.4],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=1,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 0
        factor_s = 0.0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    rhs = np.asarray([0.1, 2.0, -3.0, 4.0], dtype=np.float64)
    bundle = v3_driver_module._try_build_true_operator_active_residual_block_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_nbytes=1024 * 1024,
        regularization=1.0e-14,
        max_block_size=4,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        max_tail=1,
        kinetic_only=True,
    )

    assert bundle is not None
    corrected = bundle.solve(rhs)
    assert np.linalg.norm(rhs - matrix @ corrected) < 1.0e-10
    assert bundle.metadata is not None
    assert bundle.metadata["block_size"] == 4
    assert bundle.metadata["kinetic_selected"] == 3
    assert bundle.metadata["tail_selected"] == 1
    assert bundle.metadata["selection"] == "top_residual_active_positions"
    assert bundle.metadata["residual_energy_fraction"] == pytest.approx(1.0)


def test_true_operator_active_submatrix_solves_deterministic_active_block() -> None:
    matrix = sp.csr_matrix(
        [
            [2.0, -0.5, 0.25],
            [0.1, 1.8, -0.4],
            [-0.2, 0.3, 1.6],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 0
        factor_s = 0.0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    rhs = np.asarray([1.0, -0.25, 0.5], dtype=np.float64)
    bundle = v3_driver_module._try_build_true_operator_active_submatrix_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        x_count=1,
        ell_count=3,
        max_nbytes=1024 * 1024,
        regularization=1.0e-14,
        max_block_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        max_tail=0,
        damping=False,
    )

    assert bundle is not None
    corrected = bundle.solve(rhs)
    assert np.linalg.norm(rhs - matrix @ corrected) < 1.0e-10
    assert bundle.metadata is not None
    assert bundle.metadata["block_size"] == 3
    assert bundle.metadata["a_block_nnz"] == matrix.nnz
    assert bundle.metadata["local_solver"] == "splu"


def test_reusable_true_action_column_cache_reuses_batched_columns() -> None:
    matrix = np.asarray(
        [
            [2.0, -0.5, 0.25],
            [0.1, 1.8, -0.4],
            [-0.2, 0.3, 1.6],
        ],
        dtype=np.float64,
    )
    calls = {"matmat": 0}

    def true_matmat(x):
        calls["matmat"] += 1
        return matrix @ np.asarray(x, dtype=np.float64)

    cache = v3_driver_module._ReusableTrueActionColumnCache(
        true_matvec=lambda x: matrix @ np.asarray(x, dtype=np.float64),
        true_matmat=true_matmat,
        n=3,
        max_nbytes=1024 * 1024,
        enabled=True,
    )
    basis_a = np.zeros((3, 2), dtype=np.float64)
    basis_a[[0, 2], [0, 1]] = 1.0
    basis_b = np.zeros((3, 2), dtype=np.float64)
    basis_b[[2, 1], [0, 1]] = 1.0

    out_a = cache.matmat(basis_a)
    out_b = cache.matmat(basis_b)

    np.testing.assert_allclose(out_a, matrix @ basis_a)
    np.testing.assert_allclose(out_b, matrix @ basis_b)
    metadata = cache.metadata()
    assert metadata["hits"] == 1
    assert metadata["misses"] == 3
    assert metadata["stored_columns"] == 3
    assert metadata["batches"] == 2
    assert calls["matmat"] == 1


def test_active_residual_block_reuses_true_action_column_cache() -> None:
    matrix = np.asarray(
        [
            [2.0, -0.5, 0.25],
            [0.1, 1.8, -0.4],
            [-0.2, 0.3, 1.6],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    calls = {"matmat": 0}

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 0
        factor_s = 0.0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    def true_matmat(x):
        calls["matmat"] += 1
        return matrix @ np.asarray(x, dtype=np.float64)

    cache = v3_driver_module._ReusableTrueActionColumnCache(
        true_matvec=lambda x: matrix @ np.asarray(x, dtype=np.float64),
        true_matmat=true_matmat,
        n=3,
        max_nbytes=1024 * 1024,
        enabled=True,
    )
    rhs = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)

    first = v3_driver_module._try_build_true_operator_active_block_lsq_preconditioner(
        true_matvec=cache.matvec,
        true_matmat=cache.matmat,
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        x_count=1,
        ell_count=3,
        max_nbytes=1024 * 1024,
        regularization=1.0e-14,
        max_block_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        max_tail=0,
    )
    second = v3_driver_module._try_build_true_operator_active_residual_block_lsq_preconditioner(
        true_matvec=cache.matvec,
        true_matmat=cache.matmat,
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_nbytes=1024 * 1024,
        regularization=1.0e-14,
        max_block_size=3,
        column_batch=3,
        drop_tol=0.0,
        include_tail=False,
        max_tail=0,
        kinetic_only=True,
    )

    assert first is not None
    assert second is not None
    np.testing.assert_allclose(cache.matmat(np.eye(3, dtype=np.float64)), matrix)
    metadata = cache.metadata()
    assert metadata["misses"] == 3
    assert metadata["hits"] >= 5
    assert metadata["stored_columns"] == 3
    assert calls["matmat"] == 1


def test_true_operator_residual_window_specs_skip_invalid_indices() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=0,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    specs = v3_driver_module._parse_true_operator_window_specs(
        "0:0:1, 4:0:0, -1:0:0, 0:5:0, 0:0:9, 0/1/2",
        layout=layout,
    )

    assert specs == ((0, 0, 1), (0, 1, 2))


def test_true_operator_residual_window_lsq_supports_matvec_only_short_batches() -> None:
    matrix = sp.csr_matrix(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    rhs = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)
    bundle = v3_driver_module._try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=None,
        factor_bundle=ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=1,
        max_nbytes=1024 * 1024,
        regularization=1.0e-12,
        max_window_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        explicit_specs=((0, 0, 1),),
        damping=False,
        beta_max=10.0,
    )

    assert bundle is not None
    corrected = bundle.solve(rhs)
    np.testing.assert_allclose(matrix @ corrected, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_true_operator_residual_window_lsq_is_memory_gated() -> None:
    matrix = sp.eye(3, format="csr")
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class ZeroFactor:
        kind = "zero"
        factor_nbytes_estimate = 0

        def solve(self, rhs):
            return np.zeros_like(np.asarray(rhs, dtype=np.float64))

    bundle = v3_driver_module._try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        true_matmat=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        factor_bundle=ZeroFactor(),
        residual=np.ones(3, dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1,
        regularization=1.0e-12,
        max_window_size=3,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        explicit_specs=((0, 0, 0),),
        damping=False,
        beta_max=10.0,
    )

    assert bundle is None


def test_residual_window_host_preconditioner_can_combine_windows() -> None:
    matrix = sp.eye(8, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit test",
            backend="cpu",
            shape=(8, 8),
            dense_nbytes=8 * 8 * 8,
            csr_nbytes_estimate=256,
            nnz_estimate=8,
        ),
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=0,
        extra_size=0,
        total_size=8,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )

    class HalfFactor:
        kind = "half_identity"
        factor_nbytes_estimate = 0
        factor_nnz_estimate = 8
        factor_s = 0.0

        def solve(self, rhs):
            return 0.5 * np.asarray(rhs, dtype=np.float64)

    rhs = np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0], dtype=np.float64)
    failed_residual = rhs - operator.matvec(HalfFactor().solve(rhs))

    bundle = v3_driver_module._try_build_residual_window_host_sparse_preconditioner(
        operator_bundle=operator,
        factor_bundle=HalfFactor(),
        residual=failed_residual,
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        coefficient_mode="least_squares",
        combine_mode="union",
        interface_depth=0,
        max_window_size=16,
    )

    assert bundle is not None
    assert len(bundle.window_positions) == 1
    assert bundle.metadata is not None
    assert bundle.metadata["combine_mode"] == "union"
    corrected = bundle.solve(rhs)
    np.testing.assert_allclose(operator.matvec(corrected), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_sparse_host_ilu_escalates_regularization_after_singular_factor(monkeypatch) -> None:
    v3_driver_module._RHSMODE1_SPARSE_ILU_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", "2")
    messages: list[str] = []

    _a_full, _a_drop, ilu = v3_driver_module._factorize_sparse_matrix_csr_host(
        a_csr_full=sp.csr_matrix([[0.0, 0.0], [0.0, 1.0]]),
        cache_key=("singular-regularization-test",),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        factorization="ilu",
        emit=lambda _level, msg: messages.append(msg),
    )

    assert ilu is not None
    assert any("increasing diagonal regularization" in msg for msg in messages)


def test_xblock_precondition_side_defaults_right_only_for_full_fp_er() -> None:
    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=False,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=39_314,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=52_637,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=False,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="left",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=True,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("left", False)


def test_xblock_gmres_restart_caps_only_auto_right_preconditioned_path() -> None:
    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (20, True)

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="40",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (80, False)

    policy = resolve_rhs1_xblock_sparse_pc_policy(
        precondition_side_env_value="",
        krylov_env_value="",
        requested_restart=80,
        restart_env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=39_314,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert policy.precondition_side == "right"
    assert policy.default_right_preconditioned is True
    assert policy.gmres_restart == 80
    assert policy.restart_capped is False

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="lgmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (80, False)

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=False,
    )
    assert (restart, capped) == (80, False)


def test_compact_csr_triangular_solves_match_dense_reference() -> None:
    lower_indptr = jnp.asarray([0, 0, 1, 3], dtype=jnp.int32)
    lower_indices = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    lower_data = jnp.asarray([2.0, -1.0, 0.5], dtype=jnp.float64)
    upper_indptr = jnp.asarray([0, 2, 3, 3], dtype=jnp.int32)
    upper_indices = jnp.asarray([1, 2, 2], dtype=jnp.int32)
    upper_data = jnp.asarray([-0.25, 0.5, 1.5], dtype=jnp.float64)
    upper_diag = jnp.asarray([4.0, -3.0, 2.0], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0, 0.25], dtype=jnp.float64)

    y = _triangular_solve_lower_csr_rows(
        indptr=lower_indptr,
        indices=lower_indices,
        data=lower_data,
        b=rhs,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )
    z = _triangular_solve_upper_csr_rows(
        indptr=upper_indptr,
        indices=upper_indices,
        data=upper_data,
        upper_diag=upper_diag,
        b=y,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )

    lower = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.5, 1.0]])
    upper = np.array([[4.0, -0.25, 0.5], [0.0, -3.0, 1.5], [0.0, 0.0, 2.0]])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, np.asarray(rhs)))
    np.testing.assert_allclose(np.asarray(z), expected, rtol=1.0e-12, atol=1.0e-12)


def test_conservative_sparse_pattern_covers_pas_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.nnz == pattern.nnz
    assert summary.has_pas
    assert not summary.has_fp


def test_conservative_sparse_pattern_covers_fp_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.has_fp
    assert summary.avg_row_nnz > 0.0


def test_conservative_sparse_pattern_preflight_estimate_bounds_materialized_pattern() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    summary = summarize_v3_sparse_pattern(op, pattern)
    estimate = estimate_v3_full_system_conservative_sparsity_summary(op)

    assert estimate.shape == summary.shape
    assert estimate.nnz >= summary.nnz
    assert estimate.max_row_nnz >= summary.max_row_nnz
    assert estimate.has_fp is True


def test_fp_sparse_pc_can_use_local_velocity_pattern() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    dense_velocity = v3_full_system_conservative_sparsity_pattern(op, fp_dense_velocity_block=True)
    local_velocity = v3_full_system_conservative_sparsity_pattern(op, fp_dense_velocity_block=False)

    assert local_velocity.shape == dense_velocity.shape == (op.total_size, op.total_size)
    assert local_velocity.nnz < dense_velocity.nnz
    assert summarize_v3_sparse_pattern(op, local_velocity).has_fp


def test_conservative_sparse_pattern_covers_phi1_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.include_phi1
    assert summary.max_row_nnz >= op.n_theta * op.n_zeta


def test_pattern_probe_recovers_pas_tiny_matrix_free_operator() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    fortran_matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.whichMatrix_3.petscbin")

    def mv(x):
        return apply_v3_full_system_operator(op, jnp.asarray(x, dtype=jnp.float64))

    bundle = build_operator_from_pattern(mv, pattern=pattern, backend="cpu")

    assert sp.isspmatrix_csr(bundle.matrix)
    assert bundle.metadata.block_cols < op.total_size
    np.testing.assert_allclose(bundle.matrix.toarray(), fortran_matrix.toarray(), rtol=0, atol=3e-12)


def test_sparse_host_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_host",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-10
    assert any("sparse_host complete" in msg for msg in messages)


def test_sparse_pc_gmres_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_pc_gmres",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-10
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "sparse_pc_gmres"
    assert result.metadata["setup_s"] >= 0.0
    assert result.metadata["solve_s"] >= 0.0
    assert result.metadata["elapsed_s"] >= result.metadata["setup_s"]
    assert result.metadata["sparse_pc_factor_dtype"] == "float64"
    assert result.metadata["sparse_pc_initial_factor_dtype"] == "float64"
    assert result.metadata["sparse_pc_factor_dtype_retry"] is None
    assert result.metadata["sparse_pc_first_attempt_maxiter"] == result.metadata["gmres_maxiter"]
    assert result.metadata["sparse_pc_permc_spec"] in {"COLAMD", "MMD_ATA"}
    assert result.metadata["sparse_pc_default_permc_spec"] in {"COLAMD", "MMD_ATA"}
    assert result.metadata["sparse_pattern_nnz"] > 0
    assert result.metadata["sparse_pattern_max_row_nnz"] > 0
    assert any("sparse_pc_gmres complete" in msg for msg in messages)


def test_fortran_reduced_pc_operator_preserves_angular_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    point = v3_driver_module._build_rhsmode1_preconditioner_operator_point(op)
    reduced = v3_driver_module._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddtheta),
        np.asarray(op.fblock.collisionless.ddtheta),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddzeta),
        np.asarray(op.fblock.collisionless.ddzeta),
    )
    np.testing.assert_allclose(
        np.asarray(point.fblock.collisionless.ddtheta),
        np.diag(np.diag(np.asarray(op.fblock.collisionless.ddtheta))),
    )
    np.testing.assert_allclose(
        np.asarray(point.fblock.collisionless.ddzeta),
        np.diag(np.diag(np.asarray(op.fblock.collisionless.ddzeta))),
    )


def test_fortran_reduced_pc_pattern_keeps_global_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    point = v3_driver_module._build_rhsmode1_preconditioner_operator_point(op)
    reduced = v3_driver_module._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    true_pattern = v3_full_system_conservative_sparsity_pattern(op).astype(bool).tocsr()
    point_pattern = v3_full_system_conservative_sparsity_pattern(point).astype(bool).tocsr()
    reduced_pattern = v3_full_system_conservative_sparsity_pattern(reduced).astype(bool).tocsr()

    missing_from_true = reduced_pattern.astype(np.int8) - reduced_pattern.multiply(true_pattern).astype(np.int8)
    missing_from_true.eliminate_zeros()

    assert point_pattern.nnz < reduced_pattern.nnz
    assert reduced_pattern.nnz <= true_pattern.nnz
    assert missing_from_true.nnz == 0


def test_fortran_reduced_structural_pattern_drops_fp_x_species_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    conservative = v3_full_system_conservative_sparsity_pattern(op).astype(bool).tocsr()
    reduced = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    ).astype(bool).tocsr()

    def idx(s: int, x: int, ell: int, theta: int, zeta: int) -> int:
        return (((s * op.n_x + x) * op.n_xi + ell) * op.n_theta + theta) * op.n_zeta + zeta

    row = idx(0, 0, 0, 2, 2)
    off_species_col = idx(1, 0, 0, 2, 2)
    off_x_col = idx(0, 1, 0, 2, 2)
    theta_coupled_col = idx(0, 0, 1, 1, 2)
    zeta_coupled_col = idx(0, 0, 1, 2, 1)
    off_xi2_col = idx(0, 0, 2, 2, 2)

    assert reduced.shape == conservative.shape
    assert reduced.nnz < conservative.nnz
    assert conservative[row, off_species_col]
    assert conservative[row, off_x_col]
    assert not reduced[row, off_species_col]
    assert not reduced[row, off_x_col]
    assert not reduced[row, off_xi2_col]
    assert reduced[row, theta_coupled_col]
    assert reduced[row, zeta_coupled_col]


def test_fortran_reduced_structural_pattern_respects_preconditioner_x_min_l() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    reduced = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=1,
    ).astype(bool).tocsr()

    def idx(s: int, x: int, ell: int, theta: int, zeta: int) -> int:
        return (((s * op.n_x + x) * op.n_xi + ell) * op.n_theta + theta) * op.n_zeta + zeta

    low_l_row = idx(0, 0, 0, 2, 2)
    high_l_row = idx(0, 0, 1, 2, 2)
    low_l_off_x = idx(0, 1, 0, 2, 2)
    high_l_off_x = idx(0, 1, 1, 2, 2)

    assert reduced[low_l_row, low_l_off_x]
    assert not reduced[high_l_row, high_l_off_x]


def test_fortran_reduced_pc_gmres_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "64")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced"] is True
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_global"
    assert result.metadata["sparse_pc_fortran_reduced_keeps_theta_zeta"] is True
    assert result.metadata["sparse_pc_fortran_reduced_preconditioner_x"] == 1
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_full"
    assert result.metadata["sparse_pattern_nnz"] > 0
    assert any("fortran_reduced_pc_gmres using global angular-coupled" in msg for msg in messages)
    assert any("sparse_pc_gmres complete" in msg for msg in messages)


def test_auto_selects_fortran_reduced_pc_gmres_for_large_full_fp_rhs1(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "64")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="auto",
        differentiable=False,
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert result.metadata is not None
    assert result.metadata["auto_solver_selected"] is True
    assert result.metadata["auto_solver_policy"] == "fortran_reduced_pc_gmres"
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced"] is True
    assert any("auto selecting Fortran-reduced sparse-PC GMRES" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_enabled"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_error"] is None
    assert "structured direct-tail CSR" in result.metadata[
        "sparse_pc_fortran_reduced_direct_tail_operator_reason"
    ]
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] > 0
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_csr_nbytes_estimate"] > 0
    assert result.metadata["sparse_pc_operator_nnz_estimate"] > 0
    assert result.metadata["sparse_pc_operator_csr_nbytes_estimate"] > 0
    assert result.metadata["sparse_pc_factor_elapsed_s"] >= 0.0
    assert result.metadata["sparse_pc_residual_target"] > 0.0
    assert result.metadata["sparse_pc_residual_ratio_to_target"] < 1.0
    assert result.metadata["sparse_pc_factor_quality_rejected"] is False
    assert result.metadata["sparse_pc_factor_preflight_enabled"] is True
    assert result.metadata["sparse_pc_factor_preflight_required"] is False
    assert result.metadata["sparse_pc_factor_preflight_residual_before"] > 0.0
    assert result.metadata["sparse_pc_factor_preflight_residual_after"] >= 0.0
    assert result.metadata["sparse_pc_factor_preflight_improvement_ratio"] is not None
    assert result.metadata["sparse_pc_factor_preflight_target_ratio"] is not None
    assert any("fortran_reduced direct-tail structured csr built" in msg for msg in messages)
    assert any("explicit_sparse: factorization complete" in msg for msg in messages)
    assert any("sparse_pc_gmres factor preflight" in msg for msg in messages)


def test_sparse_pc_post_minres_records_true_residual_improvement(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_STEPS", "2")

    def fake_gmres_solve_with_history_scipy(**kwargs):
        b = np.asarray(kwargs["b"], dtype=np.float64)
        x = np.zeros_like(b)
        return x, float(np.linalg.norm(b)), [float(np.linalg.norm(b))]

    monkeypatch.setattr(
        v3_driver_module,
        "gmres_solve_with_history_scipy",
        fake_gmres_solve_with_history_scipy,
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-12,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.metadata["sparse_pc_post_minres_steps_requested"] == 2
    assert result.metadata["sparse_pc_post_minres_steps_accepted"] > 0
    assert result.metadata["sparse_pc_post_minres_error"] is None
    assert result.metadata["sparse_pc_post_minres_residual_before"] is not None
    assert result.metadata["sparse_pc_post_minres_residual_after"] is not None
    assert (
        result.metadata["sparse_pc_post_minres_residual_after"]
        < result.metadata["sparse_pc_post_minres_residual_before"]
    )
    assert any("sparse_pc_gmres post-minres improved residual" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_can_fallback_to_pattern_probe(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert "direct-tail materialization" in result.metadata[
        "sparse_pc_fortran_reduced_direct_tail_operator_reason"
    ]
    assert any("fortran_reduced direct-tail materialization csr built" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_whichmatrix0_active_terms_solve_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "whichMatrix0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    reason = result.metadata["sparse_pc_fortran_reduced_direct_tail_operator_reason"]
    assert "whichMatrix=0 active term-level" in reason
    assert "no kinetic probing" in reason
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] > 0
    assert any("whichMatrix=0 active term CSR built" in msg for msg in messages)


def test_fortran_reduced_direct_tail_auto_preconditioner_uses_active_ladder(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES", "jacobi")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    # CI/Linux JAX releases can differ at the last Krylov iteration by a few
    # ulps around the requested tolerance. This test is about auto path
    # selection, so keep the residual gate tight without making it bit-fragile.
    assert float(result.residual_norm) < 1.2e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "auto_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "auto"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "jacobi"
    assert structured_metadata["metadata"]["auto_selected_kind"] == "jacobi"


def test_fortran_reduced_direct_pmat_preconditioner_skips_active_csr_materialization(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_direct_pmat_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_XI", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_SPECIES", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X_MIN_L", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_pmat_requested"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is False
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_fortran_v3_reduced_direct_pmat_lu"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["metadata"]["direct_reduced_pmat_emission"] is True
    assert structured_metadata["metadata"]["direct_reduced_pmat_avoids_full_active_true_csr"] is True
    assert any("materialization skipped; direct reduced-Pmat preconditioner requested" in msg for msg in messages)
    assert not any("whichMatrix=0 active term CSR built" in msg for msg in messages)


def test_fortran_reduced_direct_tail_auto_retries_active_lu_after_native_preflight_failure(
    monkeypatch,
) -> None:
    v3_driver_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES",
        "active_fortran_v3_reduced_native_stack,active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "auto_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "auto"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"]
        == "auto_retry_selected_no_required_preflight:complete"
    )
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    pc_metadata = structured_metadata["metadata"]
    assert pc_metadata["auto_preflight_retry_selected"] is True
    attempts = pc_metadata["auto_preflight_retry_attempts"]
    assert attempts[0]["kind"] == "active_fortran_v3_reduced_lu"
    assert attempts[0]["preflight_required"] is False
    assert attempts[0]["preflight_passed"] is False
    assert attempts[0]["preflight_policy_passed"] is True
    assert any(
        "structured preconditioner selected kind=active_fortran_v3_reduced_native_stack" in msg
        for msg in messages
    )
    assert any(
        "auto preflight retry accepted kind=active_fortran_v3_reduced_lu required=False" in msg
        for msg in messages
    )


def test_fortran_reduced_direct_tail_large_auto_fails_closed_before_host_factor_fallback(
    monkeypatch,
) -> None:
    v3_driver_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "1e-6")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_FAIL_CLOSED_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES", "active_fortran_v3_reduced_lu")
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="direct-tail structured preconditioner was explicitly requested"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )

    assert any("structured preconditioner not selected" in msg for msg in messages)
    assert not any("sparse_pc_gmres host sparse factor built" in msg for msg in messages)


def test_structured_direct_tail_uses_actual_csr_budget_instead_of_preflight() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    op = full_system_operator_from_namelist(nml=nml)
    messages: list[str] = []

    bundle = v3_driver_module._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=None,
        csr_max_mb=1.0e-4,
        drop_tol=0.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert bundle is None
    assert any("rejected actual CSR budget" in msg for msg in messages)
    assert not any("csr_budget_preflight_exceeded" in msg for msg in messages)


def test_structured_direct_tail_skips_large_project_after_build(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    op = full_system_operator_from_namelist(nml=nml)
    active = np.arange(int(op.total_size) - 1, dtype=np.int32)
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_PROJECT_AFTER_BUILD_MAX_SIZE", "1")

    bundle = v3_driver_module._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=active,
        csr_max_mb=100.0,
        drop_tol=0.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert bundle is None
    assert any("skipped full build before active projection" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_diagonal_schur_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_diagonal_schur",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "active_diagonal_schur"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_diagonal_schur"
    assert structured_metadata["metadata"]["tail_size"] > 0
    assert structured_metadata["metadata"]["factor_nbytes_actual"] > 0
    assert result.metadata["sparse_pc_factor_elapsed_s"] >= 0.0
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_diagonal_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_global_field_split_schur_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_global_field_split_schur",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_global_field_split_schur"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_global_field_split_schur"
    assert structured_metadata["metadata"]["architecture"] == "active_kinetic_global_field_split_schur"
    assert structured_metadata["metadata"]["tail_size"] > 0
    assert structured_metadata["metadata"]["factor_nbytes_actual"] > 0
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_global_field_split_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_fortran_v3_reduced_ilu_fails_fast_when_preflight_worsens(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_ilu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FILL_FACTOR", "8")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DROP_TOL", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_CANDIDATES", "current,x0")

    with pytest.raises(RuntimeError, match="direct-tail structured preconditioner preflight failed") as excinfo:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )

    message = str(excinfo.value)
    assert "active_fortran_v3_reduced_ilu" in message
    assert "target_ratio=" in message


def test_fortran_reduced_pc_gmres_direct_tail_active_xblock_ilu_low_l_schur_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_xblock_ilu_low_l_schur",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_ALLOW_SINGULAR_FALLBACK", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_DROP_TOL", "1e-3")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=120,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_xblock_ilu_low_l_schur"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_low_l_schur"
    assert structured_metadata["metadata"]["factor_kind"] == "spilu"
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["factor_kind"] == "spilu"
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["allow_block_fallback"] is True
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_low_l_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_native_xell_coarse_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_native_xell_field_split_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "128")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_native_xell_field_split_sparse_coarse"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_native_xell_field_split_sparse_coarse"
    assert (
        structured_metadata["metadata"]["architecture"]
        == "active_native_xell_global_field_split_sparse_coarse"
    )
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["requested_base_kind"] == "active_native_xell"
    assert structured_metadata["metadata"]["coarse_size"] > 0
    assert structured_metadata["metadata"]["az_basis_nnz"] > 0
    assert any(
        "structured preconditioner selected kind=active_native_xell_field_split_sparse_coarse" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_angular_line_coarse_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_angular_line_field_split_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_angular_line_field_split_sparse_coarse"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_angular_line_field_split_sparse_coarse"
    assert (
        structured_metadata["metadata"]["architecture"]
        == "active_angular_line_global_field_split_sparse_coarse"
    )
    assert structured_metadata["metadata"]["requested_base_kind"] == "active_angular_line"
    assert structured_metadata["metadata"]["adaptive_residual_basis_enabled"] is True
    assert any(
        "structured preconditioner selected kind=active_angular_line_field_split_sparse_coarse" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_multiline_coarse_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_multiline_field_split_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_multiline_field_split_sparse_coarse"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_multiline_field_split_sparse_coarse"
    assert (
        structured_metadata["metadata"]["architecture"]
        == "active_multiline_xell_angular_global_field_split_sparse_coarse"
    )
    assert structured_metadata["metadata"]["requested_base_kind"] == "active_multiline_xell_angular"
    assert (
        structured_metadata["metadata"]["base_preconditioner"]["metadata"]["architecture"]
        == "active_multiline_xell_angular_field_split_residual"
    )
    assert structured_metadata["metadata"]["adaptive_residual_basis_enabled"] is True
    assert any(
        "structured preconditioner selected kind=active_multiline_field_split_sparse_coarse" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_bounded_native_stack_fails_fast_when_preflight_worsens(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_bounded_native_stack",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_bounded_native_stack" in str(exc)
    else:  # pragma: no cover - the bounded stack must not enter GMRES if it worsens residual.
        raise AssertionError("bounded native stack should fail fast when preflight worsens the residual")
    assert any(
        "structured preconditioner selected kind=active_bounded_native_stack" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_native_stack_production_alias_fails_fast(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_native_stack",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE",
        "1",
    )
    true_coupled_called = False

    def _unexpected_true_coupled_builder(*_args, **_kwargs):
        nonlocal true_coupled_called
        true_coupled_called = True
        raise AssertionError("native-stack true-coupled auto rescue should be opt-in")

    monkeypatch.setattr(
        v3_driver_module,
        "_try_build_true_operator_coupled_coarse_lsq_preconditioner",
        _unexpected_true_coupled_builder,
    )
    messages: list[str] = []

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_fortran_v3_reduced_native_stack" in str(exc)
    else:  # pragma: no cover - the production alias must stay residual-gated.
        raise AssertionError("production native stack should fail fast when preflight worsens the residual")
    assert any(
        "structured preconditioner selected kind=active_fortran_v3_reduced_native_stack" in msg
        for msg in messages
    )
    assert true_coupled_called is False


def test_fortran_reduced_pc_gmres_direct_tail_active_symbolic_coupled_schur_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_symbolic_coupled_schur",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_X_COUNT", "3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ELL_COUNT", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_COARSE_SIZE", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_IDENTITY_COLUMNS", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ACCEPT_BASE_IMPROVEMENT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE", "256")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS", "16")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS",
        "32",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ",
        "4096",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_symbolic_coupled_schur"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_symbolic_coupled_schur"
    assert structured_metadata["metadata"]["requires_preflight"] is True
    assert structured_metadata["metadata"]["symbolic_kinetic_basis_columns"] > 0
    assert structured_metadata["metadata"]["symbolic_identity_basis_covers_active"] is True
    assert structured_metadata["metadata"]["base_kind"] == "zero_symbolic_schur_base"
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_selected"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_base_improvement_override_used"] is True
    true_coupled_metadata = result.metadata["sparse_pc_direct_tail_true_coupled_coarse_metadata"]
    assert true_coupled_metadata["residual_after"] < true_coupled_metadata["base_residual_after"]
    assert any(
        "structured preconditioner selected kind=active_symbolic_coupled_schur" in msg
        for msg in messages
    )
    assert any("true coupled coarse accepted" in msg for msg in messages)


def test_fortran_reduced_direct_tail_symbolic_schur_can_use_coupled_kinetic_base(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_symbolic_coupled_schur",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_BASE", "active_coupled_kinetic_block")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_X_COUNT", "3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ELL_COUNT", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_COARSE_SIZE", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_IDENTITY_COLUMNS", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "2048")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FACTOR_KIND", "splu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ACCEPT_BASE_IMPROVEMENT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE", "256")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS", "16")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS",
        "32",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ",
        "4096",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_symbolic_coupled_schur"
    assert structured_metadata["metadata"]["base_kind"] == "active_coupled_kinetic_block"
    base_metadata = structured_metadata["metadata"]["base_preconditioner"]["metadata"]
    assert base_metadata["architecture"] == "active_dominant_kinetic_sparse_coupled_factor"
    assert base_metadata["block_covers_active"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_selected"] is True
    assert any(
        "structured preconditioner selected kind=active_symbolic_coupled_schur" in msg
        for msg in messages
    )


def test_fortran_reduced_direct_tail_structured_pc_preflight_can_fail_fast(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_diagonal_schur",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_diagonal_schur" in str(exc)
    else:  # pragma: no cover - the preflight must reject this weak one-step factor.
        raise AssertionError("structured direct-tail preflight should fail fast when explicitly required")


def test_fortran_reduced_pc_gmres_direct_tail_sparse_coarse_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_tail_sparse_coarse",
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "active_tail_sparse_coarse"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_tail_sparse_coarse"
    assert structured_metadata["metadata"]["base_kind"] == "active_diagonal_schur"
    assert structured_metadata["metadata"]["coarse_size"] > 0
    assert structured_metadata["metadata"]["az_basis_nnz"] > 0


def test_fortran_reduced_direct_tail_structured_pc_cache_reuses_candidate(monkeypatch) -> None:
    v3_driver_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_schwarz_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", "32")

    first = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )
    second = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    first_metadata = first.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    second_metadata = second.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert first_metadata["metadata"]["direct_tail_structured_pc_cache_hit"] is False
    assert second_metadata["metadata"]["direct_tail_structured_pc_cache_hit"] is True
    assert first_metadata["metadata"]["direct_tail_structured_pc_cache_key_digest"] == second_metadata["metadata"][
        "direct_tail_structured_pc_cache_key_digest"
    ]
    assert second_metadata["metadata"]["architecture"] == "additive_schwarz_global_sparse_coarse"


def test_fortran_reduced_direct_tail_true_coupled_coarse_records_bounded_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_diagonal_schur",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE", "96")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS", "16")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ", "1024")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_requested"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_error"] is None
    metadata = result.metadata["sparse_pc_direct_tail_true_coupled_coarse_metadata"]
    assert metadata["coarse_size"] <= 96
    assert metadata["window_count"] >= 1
    assert "dominant_kinetic_residual_window" in metadata["basis_names"]
    assert any(name.startswith("profile_flow_current_moment_") for name in metadata["basis_names"])
    assert any(name.startswith("angular_residual_") for name in metadata["basis_names"])
    assert any(name.startswith("preconditioned_") for name in metadata["basis_names"])
    assert metadata["tail_included"] is True
    assert metadata["constraint_sources_included"] is True
    assert metadata["fsavg_included"] is True
    assert metadata["profile_moments_included"] is True
    assert metadata["angular_residual_included"] is True
    assert metadata["preconditioned_loads_included"] is True
    assert 1 <= metadata["preconditioned_load_column_count"] <= 16
    assert metadata["a_basis_nnz"] > 0
    assert metadata["z_basis_nnz"] > 0
    assert metadata["residual_after"] <= metadata["base_residual_after"]
    assert any("true coupled coarse built" in msg for msg in messages)


def test_fortran_reduced_direct_tail_true_coupled_coarse_auto_promotes_active_lu(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE", "96")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS", "8")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS",
        "16",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ",
        "1024",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_explicit_requested"] is False
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_auto_enabled"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_auto_min_size"] == 1
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_auto_selected"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_requested"] is True
    assert result.metadata["sparse_pc_direct_tail_true_coupled_coarse_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb"] >= 512.0
    metadata = result.metadata["sparse_pc_direct_tail_true_coupled_coarse_metadata"]
    assert metadata["residual_after"] < metadata["base_residual_after"]
    assert metadata["coarse_size"] <= 96
    assert any("true coupled coarse accepted" in msg for msg in messages)


def test_fortran_reduced_direct_tail_active_lu_preflight_stays_diagnostic_under_size_gate(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    assert result.metadata["sparse_pc_factor_preflight_enabled"] is True
    assert result.metadata["sparse_pc_factor_preflight_required"] is False
    assert result.metadata["sparse_pc_direct_tail_structured_pc_preflight_required"] is False
    assert result.metadata["sparse_pc_factor_preflight_residual_before"] > 0.0
    assert result.metadata["sparse_pc_factor_preflight_residual_after"] >= 0.0
    assert any("sparse_pc_gmres factor preflight" in msg for msg in messages)


def test_fortran_reduced_direct_tail_pc_default_cap_is_adaptive_for_active_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_BASE_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MAX_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MB_PER_UNKNOWN", raising=False)

    small = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=604,
    )
    mid = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=110_000,
    )
    production = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=900_000,
    )
    fullgrid_qa_qh = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=507_004,
    )
    upper_midgrid = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=169_264,
    )
    auto_upper_midgrid = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="auto",
        active_size=169_264,
    )
    non_exact = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_xblock",
        active_size=900_000,
    )

    assert small == pytest.approx(521.664)
    assert mid == pytest.approx(2272.0)
    assert upper_midgrid == pytest.approx(3220.224)
    assert auto_upper_midgrid == pytest.approx(3220.224)
    assert fullgrid_qa_qh == pytest.approx(14708.112)
    assert production == pytest.approx(16384.0)
    assert non_exact == pytest.approx(512.0)


def test_fortran_reduced_direct_tail_explicit_structured_pc_rejection_is_fast(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_xblock",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "0")
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="explicitly requested but not selected"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )

    assert not any("explicit_sparse: factorization start" in msg for msg in messages)


def test_fortran_reduced_direct_tail_required_pc_forces_global_backend(monkeypatch) -> None:
    v3_driver_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "required_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_required"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto"] is True
    expected_cap_mb = v3_driver_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=int(result.metadata["sparse_pc_linear_size"]),
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb"] == pytest.approx(
        expected_cap_mb
    )
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_fortran_v3_reduced_lu"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    assert structured_metadata["reason"] == "complete"
    pc_metadata = structured_metadata["metadata"]
    assert pc_metadata["factor_kind"] == "lu"
    assert pc_metadata["requires_preflight"] is False
    assert pc_metadata["max_factor_nbytes"] == int(expected_cap_mb * 1024.0 * 1024.0)
    assert pc_metadata["permc_spec_requested"] == "AUTO"
    assert not any("using x-block backend instead of monolithic CSR factor" in msg for msg in messages)


def test_sparse_pc_gmres_stagnation_guard_aborts_mocked_krylov(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_ABORT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_MIN_ITER", "3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_WINDOW", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_REL_IMPROVEMENT", "0")

    def _stagnating_gmres(*, b, progress_callback, **_kwargs):
        for iteration in range(1, 5):
            progress_callback(iteration, 1.0)
        return np.zeros_like(np.asarray(b, dtype=np.float64)), float(np.linalg.norm(b)), [1.0] * 4

    monkeypatch.setattr(v3_driver_module, "gmres_solve_with_history_scipy", _stagnating_gmres)

    with pytest.raises(RuntimeError, match="sparse_pc_gmres stagnation detected"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="sparse_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )


def test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    messages: list[str] = []

    def _forbidden_global_pattern(*_args, **_kwargs):
        raise AssertionError("x-block backend must not build the monolithic Fortran-reduced pattern")

    monkeypatch.setattr(
        v3_driver_module,
        "v3_full_system_fortran_reduced_preconditioner_sparsity_pattern",
        _forbidden_global_pattern,
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_xblock"
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_xblock_no_global_pattern"
    assert result.metadata["sparse_pattern_nnz"] == 0
    assert result.metadata["sparse_pc_xblock_moment_schur_enabled"] is False
    assert result.metadata["sparse_pc_xblock_global_coupling_enabled"] is False
    assert any("using x-block backend instead of monolithic CSR factor" in msg for msg in messages)
    assert any("fortran_reduced_pc_gmres xblock complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_xblock_backend_accepts_lgmres(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV", "lgmres")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_xblock_krylov_method"] == "lgmres"


def test_fortran_reduced_pc_gmres_xblock_backend_moment_schur_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_xblock_moment_schur_enabled"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_built"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_used"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_mode"] == "constraint1_moment_schur"
    assert result.metadata["sparse_pc_xblock_moment_schur_extra_size"] == 4
    assert result.metadata["sparse_pc_xblock_moment_schur_rank"] == 4
    assert result.metadata["sparse_pc_xblock_moment_schur_base_applies"] >= (
        2 * result.metadata["sparse_pc_xblock_moment_schur_applies"]
    )
    assert any("fortran_reduced_pc_gmres xblock constraint1 moment-Schur" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_xblock_backend_global_coupling_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS", "8")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_xblock_global_coupling_enabled"] is True
    assert result.metadata["sparse_pc_xblock_global_coupling_built"] is True
    assert result.metadata["sparse_pc_xblock_global_coupling_rank"] >= 1
    assert result.metadata["sparse_pc_xblock_global_coupling_basis_size"] <= 8
    assert result.metadata["sparse_pc_xblock_global_coupling_applies"] > 0
    assert any("fortran_reduced_pc_gmres xblock global-coupling build start" in msg for msg in messages)


def test_sparse_pc_gmres_active_dof_reduces_truncated_pas_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 8
    nml.group("resolutionParameters")["NL"] = 4
    nml.group("resolutionParameters")["NX"] = 4
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    nml.group("physicsParameters")["ER"] = 0.1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert result.metadata is not None
    assert result.metadata["sparse_pc_active_dof"] is True
    assert result.metadata["sparse_pattern_scope"] == "active_dof"
    assert result.metadata["sparse_pc_linear_size"] < result.metadata["sparse_pc_full_size"]
    active_idx = v3_driver_module._transport_active_dof_indices(result.op)
    inactive_idx = np.setdiff1d(np.arange(int(result.op.total_size), dtype=np.int32), active_idx)
    assert np.allclose(np.asarray(result.x)[inactive_idx], 0.0)
    residual = result.rhs[active_idx] - apply_v3_full_system_operator(result.op, result.x)[active_idx]
    target = 1.0e-8 * float(jnp.linalg.norm(result.rhs[active_idx]))
    assert float(jnp.linalg.norm(residual)) <= target
    assert any("active-DOF reduction enabled" in msg for msg in messages)


def test_fortran_reduced_pc_auto_uses_active_dof_for_truncated_modes(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 8
    nml.group("resolutionParameters")["NL"] = 4
    nml.group("resolutionParameters")["NX"] = 4
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    nml.group("physicsParameters")["ER"] = 0.1
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", raising=False)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_active_dof"] is True
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_active_dof"
    assert result.metadata["sparse_pc_linear_size"] < result.metadata["sparse_pc_full_size"]


def test_xblock_sparse_pc_gmres_solve_method_solves_fp_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-2
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["setup_s"] >= 0.0
    assert result.metadata["solve_s"] >= 0.0
    assert result.metadata["elapsed_s"] >= result.metadata["setup_s"]
    assert result.metadata["xblock_initial_seed_used"] in {True, False}
    assert result.metadata["xblock_initial_seed_residual_norm"] >= 0.0
    assert any("initial x-block seed" in msg for msg in messages)
    assert any("xblock_sparse_pc_gmres complete" in msg for msg in messages)


def test_xblock_sparse_pc_gmres_initial_seed_can_be_disabled(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-2
    assert result.metadata["xblock_initial_seed_used"] is False
    assert result.metadata["xblock_initial_seed_residual_norm"] is None
    assert result.metadata["xblock_initial_seed_residual_ratio"] is None
    assert result.metadata["xblock_post_minres_steps_requested"] == 2
    assert result.metadata["xblock_post_minres_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_steps_requested"] == 1
    assert result.metadata["xblock_post_coarse_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_direction_count"] == 0


def test_xblock_sparse_pc_post_residual_equation_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres_jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_MAX_DIRECTIONS", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_FSAVG_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_ANGULAR_LMAX", "-1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-12,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.metadata["xblock_post_residual_equation_steps_requested"] == 1
    assert result.metadata["xblock_post_residual_equation_residual_before"] is not None
    assert result.metadata["xblock_post_residual_equation_residual_after"] is not None
    assert (
        result.metadata["xblock_post_residual_equation_residual_after"]
        < result.metadata["xblock_post_residual_equation_residual_before"]
    )
    assert result.metadata["xblock_post_residual_equation_direction_count"] > 0
    assert any("post-residual-equation improved residual" in msg for msg in messages)


def test_xblock_sparse_pc_two_level_preconditioner_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "10")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX", "2")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["xblock_two_level_enabled"] is True
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_two_level_mode"] == "additive"
    assert 1 <= result.metadata["xblock_two_level_rank"] <= result.metadata["xblock_two_level_basis_size"] <= 10
    assert result.metadata["xblock_two_level_applies"] > 0
    assert result.metadata["xblock_two_level_coarse_applies"] == result.metadata["xblock_two_level_applies"]
    assert any("two-level coarse built" in msg for msg in messages)


def test_xblock_sparse_pc_global_coupling_preconditioner_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["xblock_global_coupling_enabled"] is True
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_mode"] == "additive"
    assert 1 <= result.metadata["xblock_global_coupling_rank"] <= result.metadata["xblock_global_coupling_basis_size"] <= 12
    assert result.metadata["xblock_global_coupling_smoother"] == "base"
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert result.metadata["xblock_global_coupling_applies"] > 0
    assert result.metadata["xblock_global_coupling_coarse_applies"] == result.metadata["xblock_global_coupling_applies"]
    assert any("global-coupling built" in msg for msg in messages)


def test_xblock_sparse_pc_global_coupling_setup_budget_uses_partial_basis(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S", "1e-12")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_smoother"] == "base"
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == pytest.approx(1.0e-12)
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is True
    assert result.metadata["xblock_global_coupling_basis_size"] < result.metadata["xblock_global_coupling_load_basis_size"]
    assert any("global-coupling setup budget reached" in msg for msg in messages)


def test_xblock_sparse_pc_constraint1_moment_schur_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_moment_schur_enabled"] is True
    assert result.metadata["xblock_moment_schur_built"] is True
    assert result.metadata["xblock_moment_schur_mode"] == "constraint1_moment_schur"
    assert result.metadata["xblock_moment_schur_extra_size"] == 4
    assert result.metadata["xblock_moment_schur_rank"] == 4
    assert result.metadata["xblock_moment_schur_device_resident"] is True
    assert result.metadata["xblock_moment_schur_used"] is True
    assert result.metadata["xblock_moment_schur_reason"] == "built"
    assert result.metadata["xblock_moment_schur_base_applies"] == 2 * result.metadata["xblock_moment_schur_applies"]
    assert result.metadata["xblock_moment_schur_seed_residual_norm"] is not None
    assert any("constraint1 moment-Schur built" in msg for msg in messages)


def test_xblock_sparse_pc_constraint1_moment_schur_probe_fails_closed(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT", "1.0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_moment_schur_enabled"] is True
    assert result.metadata["xblock_moment_schur_built"] is True
    assert result.metadata["xblock_moment_schur_used"] is False
    assert result.metadata["xblock_moment_schur_reason"] == "probe_not_reduced"
    assert result.metadata["xblock_moment_schur_probe_residual_before"] is not None
    assert result.metadata["xblock_moment_schur_probe_residual_after"] is not None
    assert result.metadata["xblock_moment_schur_probe_improvement_ratio"] is not None
    assert result.metadata["xblock_moment_schur_seed_used"] is False
    assert any("constraint1 moment-Schur rejected" in msg for msg in messages)


def test_xblock_sparse_pc_preflight_required_rejects_weak_seed(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_MIN_IMPROVEMENT", "0.9")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_REQUIRED", "1")

    with pytest.raises(RuntimeError, match="preflight gate failed"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="xblock_sparse_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )


def test_xblock_sparse_pc_assembled_operator_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_matrix_nnz"] > 0
    assert result.metadata["xblock_assembled_operator_error"] is None
    assert any("assembled operator built" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_can_use_device_csr(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_enabled"] is True
    assert result.metadata["xblock_assembled_operator_device_required"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_assembled_operator_device_nnz"] == result.metadata["xblock_assembled_operator_matrix_nnz"]
    assert result.metadata["xblock_assembled_operator_device_csr_nbytes_estimate"] > 0
    assert result.metadata["xblock_assembled_operator_device_error"] is None
    assert any("assembled operator built location=device" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_row_equilibration_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM", "linf")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_enabled"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_norm"] == "linf"
    assert result.metadata["xblock_assembled_operator_row_equilibration_setup_s"] >= 0.0
    assert result.metadata["xblock_assembled_operator_row_equilibration_scale_min"] > 0.0
    assert result.metadata["xblock_assembled_operator_row_equilibration_scale_max"] > 0.0
    assert any("assembled row equilibration built" in msg for msg in messages)
    assert any("using row-equilibrated assembled operator" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_row_col_equilibration_maps_solution(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_enabled"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_scale_min"] > 0.0
    assert result.metadata["xblock_assembled_operator_col_equilibration_scale_max"] > 0.0
    assert any("assembled column equilibration built" in msg for msg in messages)
    assert any("using row/column-equilibrated assembled operator" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_records_budget_rejection(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is False
    assert "MemoryError" in str(result.metadata["xblock_assembled_operator_error"])
    assert result.metadata["xblock_assembled_operator_preflight_rejected"] is True
    assert result.metadata["xblock_assembled_operator_preflight_pattern_nnz_estimate"] > 0
    assert any("assembled operator disabled after build failure" in msg for msg in messages)


def test_xblock_sparse_pc_active_dof_opt_in_records_reduced_size(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_linear_size"] < result.metadata["xblock_full_size"]
    assert result.gmres.x.shape == result.rhs.shape


def test_xblock_sparse_pc_probe_coarse_uses_active_projected_directions(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_FSAVG_LMAX", "2")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_probe_coarse_steps_requested"] == 1
    assert result.metadata["xblock_probe_coarse_steps_accepted"] == 1
    assert result.metadata["xblock_probe_coarse_direction_count"] == 8
    assert result.metadata["xblock_probe_coarse_angular_lmax"] == -1
    assert result.metadata["xblock_probe_coarse_seed_initialized"] is True
    assert result.metadata["xblock_probe_coarse_residual_after"] < result.metadata["xblock_probe_coarse_residual_before"]
    assert any("probe-coarse improved seed residual" in msg for msg in messages)


def test_xblock_post_coarse_directions_can_include_angular_modes() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    residual = jnp.ones((op.total_size,), dtype=jnp.float64)

    directions = _rhs1_xblock_post_coarse_directions(
        op=op,
        residual=residual,
        preconditioner=lambda v: jnp.asarray(v, dtype=jnp.float64),
        include_raw=False,
        fsavg_lmax=0,
        angular_lmax=1,
        max_extra_units=0,
        max_directions=16,
    )

    names = tuple(name for name, _direction in directions)
    assert any(name.startswith("fsavg_l") for name in names)
    assert any(name.startswith("angular_") for name in names)
    assert len(directions) <= 16


def test_xblock_post_coarse_directions_can_include_residual_weighted_angular_modes() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    theta = jnp.arange(int(op.n_theta), dtype=jnp.float64)
    zeta = jnp.arange(int(op.n_zeta), dtype=jnp.float64)
    pattern = jnp.cos(2.0 * jnp.pi * theta[:, None] / float(op.n_theta)) + 0.25 * jnp.sin(
        2.0 * jnp.pi * zeta[None, :] / float(op.n_zeta)
    )
    f_res = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
    f_res = f_res.at[0, :, 0, :, :].set(pattern[None, :, :])
    residual = jnp.concatenate(
        [f_res.reshape((-1,)), jnp.zeros((int(op.total_size) - int(op.f_size),), dtype=jnp.float64)]
    )

    directions = _rhs1_xblock_post_coarse_directions(
        op=op,
        residual=residual,
        preconditioner=lambda v: jnp.asarray(v, dtype=jnp.float64),
        include_raw=False,
        fsavg_lmax=0,
        angular_lmax=0,
        include_angular_residual=True,
        max_extra_units=0,
        max_directions=16,
    )

    names = tuple(name for name, _direction in directions)
    assert any(name.startswith("angular_residual_") for name in names)
    assert len(directions) <= 16


def test_device_subspace_residual_equation_reuses_cached_operator_basis() -> None:
    operator_matrix = jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    cached_basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float64)
    cached_action = operator_matrix @ cached_basis

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    def direction_builder(_residual):
        return (("missing_mode", jnp.asarray([0.0, 1.0], dtype=jnp.float64)),)

    x, residual, history, counts, names = _apply_device_subspace_residual_equation_correction(
        matvec=matvec,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        direction_builder=direction_builder,
        steps=1,
        max_directions=2,
        cached_basis=cached_basis,
        cached_operator_on_basis=cached_action,
        cached_labels=("flat_x0",),
        rcond=0.0,
    )

    np.testing.assert_allclose(matvec(x), rhs, rtol=1.0e-12, atol=1.0e-12)
    assert float(jnp.linalg.norm(residual)) < 1.0e-12
    assert history[-1] < 1.0e-12
    assert counts == (2,)
    assert names == ("cached_qi:flat_x0", "missing_mode")


def test_device_subspace_residual_equation_fails_closed_without_improvement() -> None:
    operator_matrix = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    def direction_builder(_residual):
        return (("zero_mode", jnp.zeros_like(rhs)),)

    x, residual, history, counts, names = _apply_device_subspace_residual_equation_correction(
        matvec=matvec,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        direction_builder=direction_builder,
        steps=1,
        max_directions=4,
    )

    np.testing.assert_allclose(x, jnp.zeros_like(rhs), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(residual, rhs, rtol=1.0e-12, atol=1.0e-12)
    assert len(history) == 1
    assert history[0] == pytest.approx(float(jnp.linalg.norm(rhs)))
    assert counts == ()
    assert names == ()


def test_xblock_sparse_pc_probe_coarse_records_angular_mode_usage(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_FSAVG_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_RESIDUAL", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_probe_coarse_angular_lmax"] == 1
    assert result.metadata["xblock_probe_coarse_angular_residual"] is True
    assert any(
        str(name).startswith("angular_")
        for name in result.metadata["xblock_probe_coarse_direction_names"]
    )


def test_xblock_sparse_pc_qi_coarse_seed_records_residual_reduction(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "10")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_coarse_seed_enabled"] is True
    assert result.metadata["xblock_qi_coarse_seed_used"] is True
    assert result.metadata["xblock_qi_coarse_seed_rank"] > 0
    assert result.metadata["xblock_qi_coarse_seed_residual_after"] < result.metadata[
        "xblock_qi_coarse_seed_residual_before"
    ]
    assert result.metadata["xblock_qi_coarse_seed_basis"] == "enriched"
    assert result.metadata["xblock_qi_coarse_seed_candidate_count"] <= 24
    assert result.metadata["xblock_qi_coarse_seed_max_candidates"] == 24
    assert result.metadata["xblock_qi_coarse_seed_max_angular_mode"] == 2
    assert "global" in result.metadata["xblock_qi_coarse_seed_labels"]
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is False
    assert any("QI coarse seed improved residual" in msg for msg in messages)


def test_xblock_sparse_pc_qi_galerkin_preconditioner_fails_closed_when_probe_worsens(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_MODE", "multiplicative")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPINGS", "1.0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_used"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_reason"] == "probe_not_reduced"
    assert result.metadata["xblock_qi_galerkin_preconditioner_mode"] == "multiplicative"
    assert result.metadata["xblock_qi_galerkin_preconditioner_basis_reused_from_seed"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_rank"] > 0
    assert result.metadata["xblock_qi_galerkin_preconditioner_candidate_count"] <= 24
    assert result.metadata["xblock_qi_galerkin_preconditioner_coarse_operator_shape"][0] == result.metadata[
        "xblock_qi_galerkin_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_galerkin_preconditioner_coarse_applies"] == 0
    assert result.metadata["xblock_qi_galerkin_preconditioner_base_applies"] == 0
    assert np.isfinite(float(result.metadata["xblock_qi_galerkin_preconditioner_residual_before"]))
    assert np.isfinite(float(result.metadata["xblock_qi_galerkin_preconditioner_residual_after"]))
    assert result.metadata["xblock_qi_galerkin_preconditioner_probe_reduced"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_selected_index"] is None
    assert result.metadata["xblock_qi_galerkin_preconditioner_probe_candidates"]
    assert all(
        candidate["residual_norm"] >= result.metadata["xblock_qi_galerkin_preconditioner_residual_before"]
        for candidate in result.metadata["xblock_qi_galerkin_preconditioner_probe_candidates"]
    )
    assert any("QI Galerkin preconditioner built" in msg for msg in messages)


def test_xblock_sparse_pc_qi_two_level_preconditioner_fails_closed_when_probe_worsens(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_used"] is False
    assert result.metadata["xblock_qi_two_level_preconditioner_reason"] == "residual_not_reduced"
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] > 0
    assert result.metadata["xblock_qi_two_level_preconditioner_candidate_count"] <= 24
    assert result.metadata["xblock_qi_two_level_preconditioner_coarse_solver"] == "action_lstsq"
    assert result.metadata["xblock_qi_two_level_preconditioner_coarse_operator_shape"][0] == result.metadata[
        "xblock_qi_two_level_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_two_level_preconditioner_operator_on_basis_shape"][1] == result.metadata[
        "xblock_qi_two_level_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_two_level_preconditioner_probe_candidates"]
    assert result.metadata["xblock_qi_two_level_preconditioner_selected_index"] is not None
    assert result.metadata["xblock_qi_two_level_preconditioner_improvement_ratio"] >= 0.95
    assert result.metadata["xblock_qi_two_level_preconditioner_applies"] == 0
    assert result.metadata["xblock_qi_two_level_preconditioner_local_applies"] >= 1
    assert any("QI two-level preconditioner rejected" in msg for msg in messages)


def test_xblock_sparse_pc_qi_two_level_residual_augmentation_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_MAX_EXTRA", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augmented"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_rank_before_augmentation"] > 0
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_max_extra"] == 2
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_steps"] == 1
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_include_residuals"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_augmentation_labels"]
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] >= result.metadata[
        "xblock_qi_two_level_preconditioner_rank_before_augmentation"
    ]


def test_xblock_sparse_pc_qi_two_level_smoothed_load_basis_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS", "1"
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS_COMBINE", "0"
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_RANK", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_ANGULAR_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "12")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_smoothed_load_basis"] is True
    smoothed = result.metadata["xblock_qi_two_level_preconditioner_smoothed_load_metadata"]
    assert smoothed["smoothed_candidate_count"] > 0
    assert smoothed["rank"] == result.metadata["xblock_qi_two_level_preconditioner_rank"]
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] <= 4


def test_xblock_sparse_pc_lower_fill_local_policy_is_wired(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL", "force")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_FACTOR", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ILU_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ROW_NNZ_MAX", "16")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-4,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-4
    assert result.metadata["xblock_lower_fill_mode"] == "force"
    assert result.metadata["xblock_lower_fill_requested"] is True
    assert result.metadata["xblock_lower_fill_ignored_env"] is False
    assert any("lower-fill local factor" in msg for msg in messages)


def test_xblock_side_probe_switch_preserves_physical_seed_for_right_pc(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("physicsParameters")["includeXDotTerm"] = False
    nml.group("physicsParameters")["includeElectricFieldTermInXiDot"] = False
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE_RESTART", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE", "0")
    monkeypatch.setattr(
        v3_driver_module._rhs1_xblock_policy,
        "rhs1_xblock_side_probe_should_switch",
        lambda *, residual_ratio, switch_ratio_env_value: True,
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_side_probe_used"] is True
    assert result.metadata["xblock_side_probe_switched"] is True
    assert result.metadata["xblock_side_probe_initial_side"] == "left"
    assert result.metadata["xblock_side_probe_selected_side"] == "right"
    assert result.metadata["xblock_side_probe_physical_seed_preserved_after_switch"] is True
    assert result.metadata["xblock_side_probe_seed_used"] is True
    assert any("preserved_physical_seed=1" in msg for msg in messages)


def test_xblock_sparse_pc_two_level_active_dof_projects_coarse_basis(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX", "2")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_two_level_enabled"] is True
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_two_level_active_projected"] is True
    assert result.metadata["xblock_two_level_expected_size"] == result.metadata["xblock_linear_size"]
    assert result.metadata["xblock_two_level_applies"] > 0
    assert any("two-level coarse built" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_active_dof_uses_sliced_budget(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active_idx = v3_driver_module._transport_active_dof_indices(op)
    full_summary = estimate_v3_full_system_conservative_sparsity_summary(op)
    full_csr_nbytes = int(full_summary.nnz * 12 + (full_summary.shape[0] + 1) * 4)
    full_sliced_pattern = v3_full_system_conservative_sparsity_pattern(op)[active_idx, :][:, active_idx].tocsr()
    active_pattern = v3_full_system_conservative_sparsity_pattern_for_indices(op, active_idx)
    assert (active_pattern != full_sliced_pattern).nnz == 0
    active_csr_nbytes = int(active_pattern.nnz * 12 + (active_pattern.shape[0] + 1) * 4)
    assert active_csr_nbytes < full_csr_nbytes

    cap_mb = 1.2 * active_csr_nbytes / 1.0e6
    assert full_csr_nbytes > cap_mb * 1.0e6
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", f"{cap_mb:.6f}")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_active_dof"] is True
    assert result.metadata["xblock_assembled_operator_preflight_scope"] == "active_dof"
    assert result.metadata["xblock_assembled_operator_preflight_active_csr_nbytes_estimate"] <= cap_mb * 1.0e6
    assert result.metadata["xblock_assembled_operator_preflight_full_csr_nbytes_estimate"] > cap_mb * 1.0e6
    assert result.metadata["xblock_assembled_operator_error"] is None


@pytest.mark.parametrize(
    ("method", "expected_solver_kind"),
    [
        ("gmres", "xblock_sparse_pc_gmres"),
        ("lgmres", "xblock_sparse_pc_lgmres"),
    ],
)
def test_xblock_sparse_pc_gmres_opt_in_krylov_method_records_realized_solver(
    monkeypatch,
    method: str,
    expected_solver_kind: str,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", method)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == expected_solver_kind
    assert result.metadata["krylov_method"] == method
    assert result.metadata["candidate_krylov_method"] == method
    assert result.metadata["fallback_from_krylov_method"] is None
    assert result.metadata["matvecs"] >= result.metadata["candidate_matvecs"]


@pytest.mark.parametrize(
    ("method", "expected_kind", "expected_metadata_key"),
    [
        ("fgmres", "xblock_sparse_pc_fgmres_jax", "xblock_device_fgmres_enabled"),
        ("gmres-jax", "xblock_sparse_pc_gmres_jax", "xblock_device_gmres_enabled"),
    ],
)
def test_xblock_sparse_pc_device_krylov_records_experimental_metadata(
    monkeypatch,
    method: str,
    expected_kind: str,
    expected_metadata_key: str,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", method)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "4")

    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    expected_method = {
        "fgmres": "fgmres_jax",
        "gmres-jax": "gmres_jax",
    }[method]
    assert result.metadata["solver_kind"] == expected_kind
    assert result.metadata["krylov_method"] == expected_method
    assert result.metadata["candidate_krylov_method"] == expected_method
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_device_krylov_method"] == expected_method
    assert result.metadata[expected_metadata_key] is True
    assert result.metadata["xblock_device_fgmres_forced_jax_factors"] is True
    if method == "fgmres":
        assert result.metadata["precondition_side"] == "right"
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_device_resident"] is True
    assert result.metadata["xblock_global_coupling_coarse_solver"] == "qr"
    assert result.metadata["xblock_global_coupling_smoother"] == "identity"
    assert result.metadata["xblock_global_coupling_ridge"] == 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 180.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert len(result.metadata["xblock_global_coupling_singular_values"]) >= 1


def test_xblock_sparse_pc_device_cycle_jit_reports_internal_iterations(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE", "cycle")

    monkeypatch.setattr(
        v3_driver_module,
        "fgmres_cycle_jit_solve_with_residual",
        _fast_device_cycle_krylov_result,
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_fgmres_jax"
    assert result.metadata["iterations"] == 80
    assert result.metadata["matvecs"] >= 82
    assert result.metadata["device_cycle_estimated_matvecs"] == result.metadata["matvecs"]
    assert result.metadata["python_matvecs"] < result.metadata["matvecs"]
    assert result.metadata["candidate_iterations"] == 80
    assert result.metadata["candidate_matvecs"] == result.metadata["matvecs"]


def test_xblock_sparse_pc_device_host_fallback_records_non_autodiff_host_policy(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK", "force")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=40,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["krylov_method"] == "gmres"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is False
    assert result.metadata["xblock_device_krylov_method"] is None
    assert result.metadata["xblock_device_host_fallback_used"] is True
    assert result.metadata["xblock_device_host_fallback_reason"] == "forced"
    assert result.metadata["xblock_device_host_fallback_requested_method"] == "gmres_jax"
    assert result.metadata["xblock_device_host_fallback_effective_krylov_env_value"] == "auto"
    assert result.metadata["xblock_device_host_fallback_non_autodiff"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_reason"] == "disabled_by_device_host_fallback"
    assert any("using non-autodiff host x-block fallback" in msg for msg in messages)


def test_xblock_sparse_pc_qi_device_krylov_request_disables_auto_host_fallback(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_WITH_BASE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK",
        "4",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER",
        "galerkin",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _forbidden_xblock_factor_build(**_kwargs):
        raise AssertionError("device-QI operator reuse should bypass local x-block factors")

    monkeypatch.setattr(
        v3_driver_module,
        "_build_rhsmode1_xblock_tz_sparse_preconditioner",
        _forbidden_xblock_factor_build,
    )
    messages: list[str] = []

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.operator_krylov_enrichment_enabled is True
        assert state.metadata.operator_krylov_depth == 1
        assert state.metadata.operator_krylov_candidate_count > 0
        assert state.metadata.adjoint_krylov_enrichment_enabled is True
        assert state.metadata.adjoint_krylov_depth == 0
        assert state.metadata.adjoint_krylov_transpose_source == "autodiff"
        assert state.metadata.multilevel_coarse_enabled is True
        assert state.metadata.multilevel_coarse_rank > 0
        assert state.metadata.multilevel_coarse_candidate_count > 0
        assert state.metadata.global_moment_residual_equation_solver == "galerkin"
        assert state.metadata.global_moment_residual_equation_include_current is True
        assert state.metadata.global_moment_residual_equation_include_tail is True
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=5.0,
            improvement_ratio=0.5,
            metadata=state.metadata,
            cycles=1,
            residual_history=(10.0, 5.0),
            step_history=(1.0,),
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-2
    assert result.metadata["xblock_device_host_fallback_used"] is False
    assert result.metadata["xblock_device_host_fallback_reason"] == "disabled"
    assert result.metadata["xblock_device_host_fallback_auto_disabled_by_qi_device"] is True
    assert result.metadata["sparse_pc_xblock_preconditioner_built"] is False
    assert result.metadata["sparse_pc_xblock_jax_factors"] is False
    assert result.metadata["xblock_qi_device_operator_reuse_enabled"] is True
    assert result.metadata["xblock_qi_device_operator_reuse_skip_xblock_factors"] is True
    assert result.metadata["xblock_qi_device_operator_reuse_reason"] == "matrix-free-qi-device-krylov"
    assert result.metadata["xblock_device_krylov_method"] == "gmres_jax"
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used_in_krylov"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_source"] == "matrix_free"
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["use_in_krylov"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["compose_with_base"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["compose_mode"] == "multiplicative"
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["residual_enrichment_enabled"] is False
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_enrichment_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_candidate_count"] > 0
    assert (
        result.metadata["xblock_qi_device_preconditioner_metadata"][
            "global_moment_residual_equation_requested"
        ]
        is True
    )
    assert (
        result.metadata["xblock_qi_device_preconditioner_metadata"][
            "global_moment_residual_equation_solver_requested"
        ]
        == "galerkin"
    )
    assert "xblock_qi_device_preconditioner_global_moment_residual_equation" in result.metadata
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["adjoint_krylov_enrichment_enabled"] is True
    assert (
        result.metadata["xblock_qi_device_preconditioner_metadata"]["adjoint_krylov_transpose_source"]
        == "autodiff"
    )
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["multilevel_coarse_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_coarse_reuse"] is True
    assert any("global moment residual equation" in msg for msg in messages)
    assert any("host fallback disabled by explicit matrix-free QI-device" in msg for msg in messages)
    assert any("skipping local x-block factors" in msg for msg in messages)


def test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "csr")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "100000")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e3,
        maxiter=1,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["sparse_pc_xblock_jax_factor_format"] == "csr"
    assert result.metadata["xblock_moment_schur_default_blocked_by_compact_factors"] is True
    assert result.metadata["xblock_moment_schur_built"] is False
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert any("xblock_sparse_csr: built compact JAX factors" in msg for msg in messages)
    assert any("moment-Schur default disabled for compact JAX factors" in msg for msg in messages)


def test_xblock_sparse_pc_device_krylov_can_use_compact_diagonal_factor_apply(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "csr")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY", "diagonal")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "100000")
    messages: list[str] = []

    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["sparse_pc_xblock_jax_factor_format"] == "csr"
    assert result.metadata["sparse_pc_xblock_jax_factor_apply"] == "diagonal"
    assert any("xblock_sparse_csr: using approximate compact JAX factor apply mode=diagonal" in msg for msg in messages)


def test_xblock_sparse_pc_device_global_coupling_can_use_normal_equations(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER", "normal-equations")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_device_resident"] is True
    assert result.metadata["xblock_global_coupling_coarse_solver"] == "normal_equations"
    assert result.metadata["xblock_global_coupling_smoother"] == "identity"
    assert result.metadata["xblock_global_coupling_ridge"] > 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 180.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False


def test_xblock_sparse_pc_device_krylov_with_device_assembled_operator_is_transfer_free(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES", "1")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_fgmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_block_between_cycles"] is True


def test_xblock_sparse_pc_qi_device_preconditioner_opt_in_records_acceptance(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.05")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_REQUIRE_ALL_DIAGONAL",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["xblock_qi_device_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used_in_krylov"] is True
    assert result.metadata["xblock_qi_device_preconditioner_reason"] == "residual_reduced"
    assert result.metadata["xblock_qi_device_preconditioner_residual_before"] == pytest.approx(10.0)
    assert result.metadata["xblock_qi_device_preconditioner_residual_after"] == pytest.approx(8.0)
    assert result.metadata["xblock_qi_device_preconditioner_improvement_ratio"] == pytest.approx(0.8)
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["device_resident"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["host_fallback_used"] is False
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["host_callback_free"] is True
    assert result.metadata["xblock_qi_device_preconditioner_use_in_krylov"] is True


def test_xblock_sparse_pc_qi_device_preconditioner_fails_closed_without_device_operator(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["xblock_qi_device_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_built"] is False
    assert result.metadata["xblock_qi_device_preconditioner_used"] is False
    assert result.metadata["xblock_qi_device_preconditioner_used_in_krylov"] is False
    assert result.metadata["xblock_qi_device_preconditioner_reason"] == "disabled_missing_assembled_device_operator"
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["assembled_operator_enabled"] is False


def test_xblock_sparse_pc_qi_device_preconditioner_matrix_free_fallback(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.05")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["xblock_assembled_operator_built"] is False
    assert result.metadata["xblock_qi_device_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used_in_krylov"] is False
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_source"] == "matrix_free"
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["local_smoother_kind"] == "none"
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["matrix_free_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["use_in_krylov"] is False
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["residual_enrichment_requested"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["residual_enrichment_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["residual_enrichment_depth"] == 2
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["residual_enrichment_candidate_count"] > 0
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_enrichment_requested"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_enrichment_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_depth"] == 1
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_krylov_candidate_count"] >= 0
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_action_enrichment_requested"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_action_enrichment_enabled"] is True
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_action_enrichment_depth"] == 1
    assert result.metadata["xblock_qi_device_preconditioner_metadata"]["operator_action_enrichment_candidate_count"] > 0


def test_xblock_sparse_pc_qi_device_multilevel_coarse_env_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.multilevel_coarse_enabled is True
        assert state.metadata.multilevel_coarse_level_count >= 1
        assert state.metadata.multilevel_coarse_candidate_count > 0
        assert state.metadata.multilevel_coarse_rank > 0
        assert any("current:" in label for label in state.metadata.accepted_basis_labels)
        assert "qi_block_sizes" in state.metadata.geometry_metadata_keys
        assert "qi_block_tail_included" in state.metadata.geometry_metadata_keys
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert metadata["multilevel_coarse_requested"] is True
    assert metadata["multilevel_coarse_enabled"] is True
    assert metadata["multilevel_coarse_level_count"] >= 1
    assert metadata["multilevel_coarse_candidate_count"] > 0
    assert metadata["multilevel_coarse_rank"] > 0
    assert metadata["multilevel_max_levels_requested"] == 2
    assert metadata["multilevel_max_rank_requested"] == 8
    assert metadata["multilevel_max_pitch_degree_requested"] == 1
    assert metadata["multilevel_current_moments_requested"] is True
    assert metadata["multilevel_current_max_pitch_degree_requested"] == 1
    assert "qi_block_sizes" in metadata["geometry_metadata_keys"]
    assert "qi_block_tail_included" in metadata["geometry_metadata_keys"]


def test_xblock_sparse_pc_qi_device_augmented_krylov_reuses_operator_basis(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres_jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    captured: dict[str, object] = {}

    def _capturing_device_krylov_result(**kwargs):
        captured.update(kwargs)
        return _fast_device_krylov_result(**kwargs)

    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _capturing_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _capturing_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    augmentation_basis = captured["augmentation_basis"]
    operator_on_augmentation = captured["operator_on_augmentation"]
    assert augmentation_basis is not None
    assert operator_on_augmentation is not None
    assert tuple(augmentation_basis.shape) == tuple(operator_on_augmentation.shape)
    assert result.metadata["xblock_device_fgmres_qi_augmented_krylov_requested"] is True
    assert result.metadata["xblock_device_fgmres_qi_augmented_krylov_used"] is True
    assert result.metadata["xblock_device_fgmres_qi_augmented_krylov_mode"] == "combined"
    assert result.metadata["xblock_qi_device_preconditioner_augmented_krylov_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_augmented_krylov_rank"] == int(
        augmentation_basis.shape[1]
    )
    assert result.metadata["xblock_qi_device_preconditioner_augmented_krylov_mode"] == "combined"


def test_xblock_sparse_pc_qi_device_augmented_krylov_can_reuse_probe_seed_space(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres_jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    captured: dict[str, object] = {}

    def _capturing_device_krylov_result(**kwargs):
        captured.update(kwargs)
        return _fast_device_krylov_result(**kwargs)

    def _accepted_augmented_seed(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        basis = jnp.zeros((int(x0.shape[0]), 1), dtype=jnp.float64).at[0, 0].set(1.0)
        action = 2.0 * basis
        probe = SimpleNamespace(
            accepted=True,
            reason="augmented_residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=1.0,
            improvement_ratio=0.1,
            metadata=state.metadata,
            cycles=1,
            residual_history=(10.0, 1.0),
            step_history=(1.0,),
        )
        return SimpleNamespace(
            solution=x0,
            probe=probe,
            augmentation_basis=basis,
            operator_on_augmentation=action,
            rank=1,
            reason="augmented_residual_reduced",
            accepted_labels=("augmented_seed:0",),
            projection_residual_norm=1.0,
        )

    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _capturing_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _capturing_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_augmented_seed", _accepted_augmented_seed)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    augmentation_basis = captured["augmentation_basis"]
    operator_on_augmentation = captured["operator_on_augmentation"]
    assert augmentation_basis is not None
    assert operator_on_augmentation is not None
    assert tuple(augmentation_basis.shape) == tuple(operator_on_augmentation.shape)
    assert augmentation_basis.shape[1] == 1
    assert result.metadata["xblock_device_fgmres_qi_augmented_seed_requested"] is True
    assert result.metadata["xblock_device_fgmres_qi_augmented_seed_used"] is True
    assert result.metadata["xblock_device_fgmres_qi_augmented_seed_rank"] == 1
    assert result.metadata["xblock_device_fgmres_qi_augmented_krylov_reason"] == "enabled_from_augmented_seed"
    assert result.metadata["xblock_qi_device_preconditioner_augmented_seed_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_augmented_seed_rank"] == 1
    assert (
        result.metadata["xblock_qi_device_preconditioner_metadata"]["augmented_seed_labels"]
        == ("augmented_seed:0",)
    )


def test_xblock_sparse_pc_qi_device_multilevel_residual_equation_env_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK",
        "5",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER",
        "fine-to-coarse",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER",
        "galerkin",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.multilevel_residual_equation_enabled is True
        assert state.metadata.multilevel_residual_equation_stage_count >= 1
        assert state.metadata.multilevel_residual_equation_rank > 0
        assert state.metadata.multilevel_residual_equation_order == "fine_to_coarse"
        assert state.metadata.multilevel_residual_equation_solver == "galerkin"
        assert state.metadata.multilevel_residual_equation_include_global is False
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert metadata["multilevel_residual_equation_requested"] is True
    assert metadata["multilevel_residual_equation_enabled"] is True
    assert metadata["multilevel_residual_equation_max_level_rank_requested"] == 5
    assert metadata["multilevel_residual_equation_order_requested"] == "fine_to_coarse"
    assert metadata["multilevel_residual_equation_solver_requested"] == "galerkin"
    assert metadata["multilevel_residual_equation_include_global_requested"] is False
    assert metadata["multilevel_residual_equation_stage_count"] >= 1
    assert metadata["multilevel_residual_equation_rank"] > 0


def test_xblock_sparse_pc_qi_device_residual_snapshot_env_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_MAX_RANK",
        "7",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_GLOBAL",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_AGGREGATES",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)
    messages: list[str] = []

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.residual_snapshot_enrichment_enabled is True
        assert state.metadata.residual_snapshot_rank > 0
        assert state.metadata.residual_snapshot_candidate_count > 0
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_residual_snapshot_enrichment"] is True
    assert metadata["residual_snapshot_enrichment_requested"] is True
    assert metadata["residual_snapshot_enrichment_enabled"] is True
    assert metadata["residual_snapshot_max_rank_requested"] == 7
    assert metadata["residual_snapshot_include_primal_requested"] is True
    assert metadata["residual_snapshot_use_adjoint_requested"] is False
    assert metadata["residual_snapshot_include_global_requested"] is True
    assert metadata["residual_snapshot_include_aggregates_requested"] is False
    assert metadata["residual_snapshot_rank"] > 0
    assert any("residual-snapshot coarse enrichment" in msg for msg in messages)


def test_xblock_sparse_pc_qi_device_block_schur_residual_equation_env_records_metadata(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK",
        "7",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_AGGREGATES",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)
    messages: list[str] = []

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.block_schur_residual_equation_enabled is True
        assert state.metadata.block_schur_residual_equation_rank > 0
        assert state.metadata.block_schur_residual_equation_candidate_count > 0
        assert state.metadata.block_schur_residual_equation_include_global is True
        assert state.metadata.block_schur_residual_equation_include_aggregates is False
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_block_schur_residual_equation"] is True
    assert result.metadata["xblock_qi_device_preconditioner_block_schur_residual_equation_rank"] > 0
    assert metadata["block_schur_residual_equation_requested"] is True
    assert metadata["block_schur_residual_equation_enabled"] is True
    assert metadata["block_schur_residual_equation_max_rank_requested"] == 7
    assert metadata["block_schur_residual_equation_include_global_requested"] is True
    assert metadata["block_schur_residual_equation_include_aggregates_requested"] is False
    assert metadata["block_schur_residual_equation_rank"] > 0
    assert any("block-Schur residual equation" in msg for msg in messages)


def test_xblock_sparse_pc_qi_device_residual_snapshot_equation_env_records_metadata(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_MAX_RANK",
        "6",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER",
        "galerkin",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_AGGREGATES",
        "0",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)
    messages: list[str] = []

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.metadata.residual_snapshot_residual_equation_solver == "galerkin"
        assert state.metadata.residual_snapshot_residual_equation_include_global is True
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert (
        result.metadata["xblock_qi_device_preconditioner_residual_snapshot_residual_equation"]
        is metadata["residual_snapshot_residual_equation_enabled"]
    )
    assert metadata["residual_snapshot_residual_equation_requested"] is True
    assert metadata["residual_snapshot_residual_equation_max_rank_requested"] == 6
    assert metadata["residual_snapshot_residual_equation_solver_requested"] == "galerkin"
    assert metadata["residual_snapshot_residual_equation_include_global_requested"] is True
    assert metadata["residual_snapshot_include_aggregates_requested"] is False
    if metadata["residual_snapshot_residual_equation_enabled"]:
        assert metadata["residual_snapshot_residual_equation_rank"] > 0
    else:
        assert metadata["residual_snapshot_residual_equation_rank"] == 0
    assert any("residual-snapshot residual equation" in msg for msg in messages)


def test_xblock_sparse_pc_qi_device_matrix_free_local_smoother_routing(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER",
        "matrix_free_minres",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS",
        "2",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_DAMPING",
        "0.75",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.05")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.local_smoother is not None
        assert state.local_smoother.metadata.sweeps == 2
        assert state.local_smoother.metadata.damping == pytest.approx(0.75)
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=7.0,
            improvement_ratio=0.7,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert metadata["operator_source"] == "matrix_free"
    assert metadata["local_smoother_kind_requested"] == "matrix_free_minres"
    assert metadata["local_smoother_kind"] == "matrix_free_residual"
    assert metadata["local_smoother_reason"] == "built"
    assert metadata["matrix_free_enabled"] is True


def test_xblock_sparse_pc_qi_device_matrix_free_block_smoother_routing(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER",
        "matrix_free_block_minres",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS",
        "3",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING",
        "block-x-species",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT", "0.05")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        assert state.local_smoother is not None
        assert state.local_smoother.metadata.group_count == 3
        assert state.local_smoother.metadata.block_count >= 1
        assert state.local_smoother.metadata.grouping == "block_x_species"
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=7.0,
            improvement_ratio=0.7,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert metadata["operator_source"] == "matrix_free"
    assert metadata["local_smoother_kind_requested"] == "matrix_free_block_minres"
    assert metadata["local_smoother_kind"] == "matrix_free_block_minres"
    assert metadata["local_smoother_reason"] == "built"
    assert metadata["matrix_free_enabled"] is True
    assert metadata["local_smoother_metadata"]["grouping"] == "block_x_species"
    assert metadata["local_smoother_metadata"]["group_count"] == 3


def test_xblock_sparse_pc_qi_device_matrix_free_seed_runs_with_precondition_side_none(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_GMRES_PRECONDITION_SIDE", "none")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    def _accepted_probe(**kwargs):
        state = kwargs["state"]
        x0 = jnp.asarray(kwargs["x0"], dtype=jnp.float64)
        return x0, SimpleNamespace(
            accepted=True,
            reason="residual_reduced",
            residual_before_norm=10.0,
            residual_after_norm=8.0,
            improvement_ratio=0.8,
            metadata=state.metadata,
        )

    monkeypatch.setattr(v3_driver_module, "probe_rhs1_qi_device_preconditioner", _accepted_probe)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    metadata = result.metadata["xblock_qi_device_preconditioner_metadata"]
    assert result.metadata["xblock_qi_device_preconditioner_built"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used"] is True
    assert result.metadata["xblock_qi_device_preconditioner_used_in_krylov"] is False
    assert metadata["precondition_side"] == "none"
    assert metadata["use_in_krylov_requested"] is True
    assert metadata["use_in_krylov"] is False
    assert metadata["use_in_krylov_blocked_by_precondition_side_none"] is True


def test_xblock_sparse_pc_device_bicgstab_uses_device_assembled_operator(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "bicgstab-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setattr(v3_driver_module, "bicgstab_solve_with_residual", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_bicgstab_jax"
    assert result.metadata["xblock_device_krylov_method"] == "bicgstab_jax"
    assert result.metadata["xblock_device_bicgstab_enabled"] is True
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_bicgstab_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False
    assert result.metadata["candidate_iterations"] >= 1
    assert result.metadata["xblock_estimated_bicgstab_work_nbytes"] < result.metadata["xblock_estimated_gmres_basis_nbytes"]


def test_xblock_sparse_pc_device_tfqmr_uses_device_assembled_operator(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "tfqmr-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TFQMR_REPLACE_INTERVAL", "2")
    monkeypatch.setattr(v3_driver_module, "tfqmr_solve_with_residual", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_tfqmr_jax"
    assert result.metadata["xblock_device_krylov_method"] == "tfqmr_jax"
    assert result.metadata["xblock_device_tfqmr_enabled"] is True
    assert result.metadata["xblock_device_tfqmr_replacement_interval"] == 2
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_tfqmr_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False
    assert result.metadata["candidate_iterations"] >= 1
    assert result.metadata["xblock_estimated_tfqmr_work_nbytes"] < result.metadata["xblock_estimated_gmres_basis_nbytes"]


def test_xblock_sparse_pc_device_krylov_marks_host_two_level_transfer(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "2")
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(v3_driver_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["xblock_device_krylov_method"] == "fgmres_jax"
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is False
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False


def test_xblock_sparse_pc_candidate_falls_back_to_gmres_when_residual_is_bad(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "bicgstab")

    def fake_bicgstab(*, b, **_kwargs):
        return np.zeros(int(b.size), dtype=np.float64), float("inf"), [float("inf")]

    monkeypatch.setattr(v3_driver_module, "bicgstab_solve_with_history_scipy", fake_bicgstab)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["krylov_method"] == "gmres"
    assert result.metadata["candidate_krylov_method"] == "bicgstab"
    assert result.metadata["fallback_from_krylov_method"] == "bicgstab"
    assert result.metadata["candidate_residual_norm"] > 1.0e-8
    assert result.metadata["candidate_iterations"] == 1


def test_sparse_lsmr_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_lsmr",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert any("sparse_lsmr complete" in msg for msg in messages)


def test_petsc_compat_solve_method_labels_minimum_norm_branch(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="petsc_compat",
        tol=1.0e-10,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "sparse_lsmr"
    assert result.metadata["petsc_compat_requested"] is True
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["acceptance_criterion"] in {
        "true_residual",
        "petsc_compatible_minimum_norm",
    }


def test_write_output_preserves_explicit_sparse_host_solve_method(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method="sparse_host",
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "sparse_host"
    assert trace["converged"] is True


def test_write_output_preserves_sparse_pc_gmres_solve_method(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method="sparse_pc_gmres",
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "sparse_pc_gmres"
    assert trace["converged"] is True
    assert trace["setup_s"] is not None
    assert trace["solve_s"] is not None
    assert trace["metadata"]["solver_metadata"]["sparse_pattern_nnz"] > 0


def test_write_output_auto_tokamak_fp_noer_policy_uses_xblock_sparse_pc(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_ASSEMBLED_HOST", "1")
    monkeypatch.setattr(io_module, "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed", lambda **_kwargs: True)
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method=None,
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "xblock_sparse_pc_gmres"
    assert trace["converged"] is True
    assert trace["metadata"]["solver_metadata"]["solver_kind"] == "xblock_sparse_pc_gmres"
    assert trace["metadata"]["solver_metadata"]["sparse_pc_xblock_preconditioner_xi"] == 1
