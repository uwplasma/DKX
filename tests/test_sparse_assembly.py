from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import sfincs_jax.solvers.preconditioner_xblock_tz_sparse as tz_sparse
import sfincs_jax.solvers.preconditioner_symbolic_host as symbolic_host
from sfincs_jax.solvers.preconditioner_symbolic_host import build_sparse_ilu_from_matvec
from sfincs_jax.solvers.preconditioner_xblock_tz_sparse import (
    assemble_selected_theta_tz_operator,
    assemble_rhsmode1_fp_xblock_tz_sparse_matrix,
    assemble_selected_zeta_tz_operator,
    get_rhsmode1_fp_xblock_assembled_host_cache,
    rhsmode1_fp_xblock_tz_sparse_diagonal,
    safe_inverse_diagonal_np,
)


def test_chunked_sparse_assembly_matches_dense_operator(monkeypatch) -> None:
    a = jnp.array(
        [
            [4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "1")

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = build_sparse_ilu_from_matvec(
        matvec=lambda x: a @ x,
        n=6,
        dtype=jnp.float64,
        cache_key=("chunked_sparse_assembly_test", 6),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=1.0e-4,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=False,
        store_dense=False,
        emit=None,
    )

    a_np = np.asarray(a)
    np.testing.assert_allclose(np.asarray(a_csr_full.toarray()), a_np, rtol=0.0, atol=0.0)
    drop_np = np.asarray(a_csr_drop.toarray())
    np.testing.assert_allclose(np.diag(drop_np) - np.diag(a_np), np.full((6,), 4.0e-12), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(drop_np - np.diag(np.diag(drop_np)), a_np - np.diag(np.diag(a_np)), rtol=0.0, atol=0.0)
    assert ilu is None
    assert a_dense is None
    assert l_dense is None
    assert u_dense is None
    assert l_unit_diag is True


def test_safe_inverse_diagonal_uses_floor_and_rejects_nonfinite() -> None:
    inv = safe_inverse_diagonal_np(np.asarray([2.0, 0.0, -1.0e-12]), floor=1.0e-6)
    assert inv is not None
    np.testing.assert_allclose(inv, np.asarray([0.5, 1.0e6, -1.0e6]))
    assert safe_inverse_diagonal_np(np.asarray([], dtype=np.float64), floor=1.0e-6) is None
    assert safe_inverse_diagonal_np(np.asarray([1.0, np.inf]), floor=1.0e-6) is None
    assert safe_inverse_diagonal_np(np.asarray([1.0, np.nan]), floor=1.0e-6) is None
    assert safe_inverse_diagonal_np(np.asarray([1.0, 0.0]), floor=0.0) is None


def test_chunked_sparse_assembly_applies_fortran_structural_threshold(monkeypatch) -> None:
    a = jnp.array(
        [
            [4.0, 5.0e-13, 0.0],
            [5.0e-13, 3.0, -2.0],
            [0.0, -2.0, 3.0],
        ],
        dtype=jnp.float64,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "1")
    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "1e-12")

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = build_sparse_ilu_from_matvec(
        matvec=lambda x: a @ x,
        n=3,
        dtype=jnp.float64,
        cache_key=("chunked_sparse_assembly_threshold_test", 3),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=1.0e-4,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=False,
        store_dense=False,
        emit=None,
    )

    expected = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [0.0, 3.0, -2.0],
            [0.0, -2.0, 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(a_csr_full.toarray()), expected, rtol=0.0, atol=0.0)
    drop_np = np.asarray(a_csr_drop.toarray())
    np.testing.assert_allclose(np.diag(drop_np) - np.diag(expected), np.full((3,), 4.0e-12), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(drop_np - np.diag(np.diag(drop_np)), expected - np.diag(np.diag(expected)), rtol=0.0, atol=0.0)
    assert ilu is None
    assert a_dense is None
    assert l_dense is None
    assert u_dense is None
    assert l_unit_diag is True


def test_symbolic_host_row_cap_and_regularization_settings_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ROW_NNZ_MAX", raising=False)
    assert symbolic_host._row_nnz_cap(None) == 256
    assert symbolic_host._row_nnz_cap(3) == 3
    assert symbolic_host._row_nnz_cap(-5) == 0

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ROW_NNZ_MAX", "bad")
    assert symbolic_host._row_nnz_cap(None) == 256
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ROW_NNZ_MAX", "-9")
    assert symbolic_host._row_nnz_cap(None) == 0

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_SINGULAR_REG_REL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", raising=False)
    reg, singular_reg_rel, attempts = symbolic_host._regularization_settings(2.0)
    assert reg == pytest.approx(2.0e-12)
    assert singular_reg_rel == pytest.approx(1.0e-10)
    assert attempts == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_SINGULAR_REG_REL", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", "bad")
    reg, singular_reg_rel, attempts = symbolic_host._regularization_settings(4.0)
    assert reg == pytest.approx(4.0e-12)
    assert singular_reg_rel == pytest.approx(1.0e-10)
    assert attempts == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", "-1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_SINGULAR_REG_REL", "-2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", "0")
    reg, singular_reg_rel, attempts = symbolic_host._regularization_settings(4.0)
    assert reg == 0.0
    assert singular_reg_rel == 0.0
    assert attempts == 1


def test_symbolic_host_drop_and_regularize_matches_dense_and_csr_paths() -> None:
    matrix_np = np.asarray([[1.0, 0.1], [0.2, 2.0]], dtype=np.float64)
    matrix_csr = sp.csr_matrix(matrix_np)

    dense_path = symbolic_host._drop_and_regularize_csr(
        a_csr_full=matrix_csr,
        a_np_full=matrix_np,
        factor_dtype=np.dtype(np.float64),
        thresh=0.15,
        reg=0.5,
    )
    csr_path = symbolic_host._drop_and_regularize_csr(
        a_csr_full=matrix_csr,
        a_np_full=None,
        factor_dtype=np.dtype(np.float64),
        thresh=0.15,
        reg=0.5,
    )
    expected = np.asarray([[1.5, 0.0], [0.2, 2.5]], dtype=np.float64)

    np.testing.assert_allclose(np.asarray(dense_path.toarray()), expected)
    np.testing.assert_allclose(np.asarray(csr_path.toarray()), expected)


def test_symbolic_host_cache_can_add_dense_and_jax_factors_lazily() -> None:
    """Pin the bounded setup path that avoids building all factors on first use."""

    symbolic_host._RHSMODE1_SPARSE_ILU_CACHE.clear()
    messages: list[str] = []
    matrix = jnp.asarray(
        [
            [4.0, 0.5, 0.0],
            [0.25, 3.0, -0.2],
            [0.0, 0.1, 2.5],
        ],
        dtype=jnp.float64,
    )
    key = ("symbolic-host-lazy-factor-test",)

    first = build_sparse_ilu_from_matvec(
        matvec=lambda x: matrix @ x,
        n=3,
        dtype=jnp.float64,
        cache_key=key,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=True,
        store_dense=False,
        factorization="lu",
        emit=lambda _level, message: messages.append(message),
    )
    cached_first = symbolic_host._RHSMODE1_SPARSE_ILU_CACHE[key]
    assert first[2] is cached_first.ilu
    assert cached_first.l_dense is None
    assert cached_first.perm_r is None

    second = build_sparse_ilu_from_matvec(
        matvec=lambda x: matrix @ x,
        n=3,
        dtype=jnp.float64,
        cache_key=key,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        build_dense_factors=True,
        build_jax_factors=True,
        build_ilu=True,
        store_dense=False,
        factorization="lu",
        row_nnz_cap=1,
        emit=lambda _level, message: messages.append(message),
    )
    cached_second = symbolic_host._RHSMODE1_SPARSE_ILU_CACHE[key]

    assert second[2] is first[2]
    assert second[4] is not None
    assert second[5] is not None
    assert cached_second.l_dense is not None
    assert cached_second.u_dense is not None
    assert cached_second.perm_r is not None
    assert cached_second.inv_perm_c is not None
    assert cached_second.lower_idx is not None
    assert cached_second.upper_idx is not None
    assert int(cached_second.lower_idx.shape[1]) <= 1
    assert int(cached_second.upper_idx.shape[1]) <= 1
    assert any("cached JAX factors" in message for message in messages)
    assert any("factorization cache hit" in message for message in messages)


def test_symbolic_host_dense_assembly_stores_thresholded_operator(monkeypatch) -> None:
    """Dense assembly should keep the optional stored matrix small and deterministic."""

    symbolic_host._RHSMODE1_SPARSE_ILU_CACHE.clear()
    matrix = jnp.asarray(
        [
            [3.0, 5.0e-10, 0.0],
            [0.5, 2.0, -0.25],
            [0.0, 0.75, 4.0],
        ],
        dtype=jnp.float64,
    )
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "100")
    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "1e-8")

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, _l_unit_diag = build_sparse_ilu_from_matvec(
        matvec=lambda x: matrix @ x,
        n=3,
        dtype=jnp.float64,
        cache_key=("symbolic-host-dense-store-test",),
        factor_dtype=np.dtype(np.float32),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=False,
        store_dense=True,
        emit=lambda _level, message: messages.append(message),
    )

    expected = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.5, 2.0, -0.25],
            [0.0, 0.75, 4.0],
        ],
        dtype=np.float32,
    )
    assert ilu is None
    assert l_dense is None
    assert u_dense is None
    assert a_dense is not None
    assert np.asarray(a_dense).dtype == np.float32
    np.testing.assert_allclose(np.asarray(a_csr_full.toarray()), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(a_dense), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(a_csr_drop.toarray()), expected, rtol=0.0, atol=2.0e-11)
    assert any("mode=dense" in message for message in messages)


def test_symbolic_host_chunked_empty_operator_can_store_dense_zero_matrix(monkeypatch) -> None:
    """Chunked assembly must handle all-zero operators without materializing bogus rows."""

    symbolic_host._RHSMODE1_SPARSE_ILU_CACHE.clear()
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "1")
    monkeypatch.setenv("SFINCS_JAX_DENSE_ASSEMBLE_JIT", "off")

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = build_sparse_ilu_from_matvec(
        matvec=lambda x: jnp.zeros_like(x),
        n=4,
        dtype=jnp.float64,
        cache_key=("symbolic-host-zero-chunked-test",),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=False,
        store_dense=True,
        emit=lambda _level, message: messages.append(message),
    )

    assert ilu is None
    assert l_dense is None
    assert u_dense is None
    assert l_unit_diag is True
    assert a_csr_full.nnz == 0
    assert a_csr_drop.nnz == 0
    assert a_dense is not None
    np.testing.assert_allclose(np.asarray(a_dense), np.zeros((4, 4), dtype=np.float64), rtol=0.0, atol=0.0)
    assert any("mode=column_blocks" in message for message in messages)
    assert any("nnz=0" in message for message in messages)


def test_selected_theta_tz_operator_matches_expected_rows() -> None:
    dd_plus = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dd_minus = np.asarray([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False, True], [False, True, False]])

    op = assemble_selected_theta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            [-3.0, 0.0, 0.0, -4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, -3.0, 0.0, 0.0, -4.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(op.toarray()), expected, rtol=0.0, atol=0.0)


def test_selected_theta_tz_operator_drops_structural_noise_and_handles_empty_rows(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "1e-8")
    dd_plus = np.asarray([[1.0e-10, 0.0], [0.0, 0.0]], dtype=np.float64)
    dd_minus = np.asarray([[0.0, -2.0], [3.0e-10, 0.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False], [False, True]])

    op = assemble_selected_theta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[1, 3] = -2.0

    np.testing.assert_allclose(np.asarray(op.toarray()), expected, rtol=0.0, atol=0.0)

    empty = assemble_selected_theta_tz_operator(
        dd_plus=np.zeros((2, 2), dtype=np.float64),
        dd_minus=np.zeros((2, 2), dtype=np.float64),
        use_plus=np.ones((2, 2), dtype=bool),
    )
    assert empty.nnz == 0
    assert empty.shape == (4, 4)


def test_selected_zeta_tz_operator_matches_expected_rows() -> None:
    dd_plus = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]], dtype=np.float64)
    dd_minus = np.asarray([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0], [0.0, -6.0, -7.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False, True], [False, True, False]])

    op = assemble_selected_zeta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
    expected = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [-3.0, -4.0, -5.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 7.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, -6.0, -7.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(op.toarray()), expected, rtol=0.0, atol=0.0)


def test_selected_zeta_tz_operator_drops_structural_noise_and_handles_empty_rows(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", "1e-8")
    dd_plus = np.asarray([[0.0, 1.0e-10], [5.0, 0.0]], dtype=np.float64)
    dd_minus = np.asarray([[0.0, -4.0], [2.0e-10, 0.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False], [False, True]])

    op = assemble_selected_zeta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[2, 3] = -4.0
    expected[3, 2] = 5.0

    np.testing.assert_allclose(np.asarray(op.toarray()), expected, rtol=0.0, atol=0.0)

    empty = assemble_selected_zeta_tz_operator(
        dd_plus=np.zeros((2, 2), dtype=np.float64),
        dd_minus=np.zeros((2, 2), dtype=np.float64),
        use_plus=np.ones((2, 2), dtype=bool),
    )
    assert empty.nnz == 0
    assert empty.shape == (4, 4)


def test_xblock_sparse_policy_wrappers_preserve_species_and_lu_defaults() -> None:
    fp_op = SimpleNamespace(fblock=SimpleNamespace(fp=object(), pas=None))
    pas_op = SimpleNamespace(fblock=SimpleNamespace(fp=object(), pas=object()))

    assert tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=SimpleNamespace(n_species=1),
        preconditioner_species=0,
    )
    assert not tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=SimpleNamespace(n_species=2),
        preconditioner_species=0,
    )
    assert tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=SimpleNamespace(n_species=2),
        preconditioner_species=1,
    )
    assert tz_sparse.rhsmode1_xblock_sparse_lu_default_max(fp_op, build_jax_factors=False) == 30000
    assert tz_sparse.rhsmode1_xblock_sparse_lu_default_max(fp_op, build_jax_factors=True) == 2000
    assert tz_sparse.rhsmode1_xblock_sparse_lu_default_max(pas_op, build_jax_factors=False) == 2000


def test_sxblock_active_indices_follow_v3_flattening_and_skip_inactive_x() -> None:
    indices_l0 = tz_sparse._sxblock_active_indices_for_l(
        n_species=2,
        n_x=3,
        n_l=4,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([1, 3, 0], dtype=np.int32),
        ell=0,
    )
    indices_l2 = tz_sparse._sxblock_active_indices_for_l(
        n_species=2,
        n_x=3,
        n_l=4,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([1, 3, 0], dtype=np.int32),
        ell=2,
    )
    indices_l3 = tz_sparse._sxblock_active_indices_for_l(
        n_species=2,
        n_x=3,
        n_l=4,
        n_theta=2,
        n_zeta=2,
        nxi_for_x=np.asarray([1, 3, 0], dtype=np.int32),
        ell=3,
    )

    np.testing.assert_array_equal(indices_l0, np.asarray([0, 1, 2, 3, 16, 17, 18, 19, 48, 49, 50, 51, 64, 65, 66, 67]))
    np.testing.assert_array_equal(indices_l2, np.asarray([24, 25, 26, 27, 72, 73, 74, 75]))
    assert indices_l3.size == 0


def test_explicit_fp_xblock_matrix_and_diagonal_share_domain_implementation() -> None:
    """A minimal full-FP x-block keeps the diagonal fallback consistent with CSR assembly."""

    ddtheta = np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    ddzeta = np.zeros((2, 2), dtype=np.float64)
    n_tz = 4
    colless = SimpleNamespace(
        x=np.asarray([1.0], dtype=np.float64),
        n_xi_for_x=np.asarray([2], dtype=np.int32),
        ddtheta=ddtheta,
        ddzeta=ddzeta,
        b_hat=np.ones((2, 2), dtype=np.float64),
        b_hat_sup_theta=np.ones((2, 2), dtype=np.float64),
        b_hat_sup_zeta=np.zeros((2, 2), dtype=np.float64),
        db_hat_dtheta=np.zeros((2, 2), dtype=np.float64),
        db_hat_dzeta=np.zeros((2, 2), dtype=np.float64),
        t_hats=np.asarray([1.0], dtype=np.float64),
        m_hats=np.asarray([1.0], dtype=np.float64),
    )
    op = SimpleNamespace(
        rhs_mode=1,
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=2,
        constraint_scheme=1,
        quasineutrality_option=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        with_adiabatic=False,
        alpha=1.0,
        delta=1.0,
        dphi_hat_dpsi_hat=0.0,
        adiabatic_z=np.zeros((0,), dtype=np.float64),
        adiabatic_nhat=np.zeros((0,), dtype=np.float64),
        adiabatic_that=np.zeros((0,), dtype=np.float64),
        z_s=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([1.0], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        n_hat=np.asarray([1.0], dtype=np.float64),
        theta_weights=np.ones((2,), dtype=np.float64),
        zeta_weights=np.ones((2,), dtype=np.float64),
        b_hat=np.ones((2, 2), dtype=np.float64),
        d_hat=np.ones((2, 2), dtype=np.float64),
        b_hat_sub_theta=np.ones((2, 2), dtype=np.float64),
        b_hat_sub_zeta=np.zeros((2, 2), dtype=np.float64),
        x=np.asarray([1.0], dtype=np.float64),
        x_weights=np.ones((1,), dtype=np.float64),
        point_at_x0=False,
        fblock=SimpleNamespace(
            collisionless=colless,
            fp=SimpleNamespace(mat=np.asarray([[[[[5.0]], [[6.0]]]]], dtype=np.float64)),
            pas=None,
            identity_shift=2.0,
            exb_theta=None,
            exb_zeta=None,
            magdrift_theta=None,
            magdrift_zeta=None,
            magdrift_xidot=None,
            er_xidot=None,
            er_xdot=None,
        ),
    )

    host = get_rhsmode1_fp_xblock_assembled_host_cache(op=op)
    matrix = assemble_rhsmode1_fp_xblock_tz_sparse_matrix(
        op=op,
        species=0,
        ix=0,
        preconditioner_xi=1,
        host_cache=host,
    )
    diagonal = rhsmode1_fp_xblock_tz_sparse_diagonal(
        op=op,
        species=0,
        ix=0,
        preconditioner_xi=1,
        host_cache=host,
    )

    assert matrix.shape == (2 * n_tz, 2 * n_tz)
    np.testing.assert_allclose(matrix.diagonal(), diagonal, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(diagonal, np.asarray([7.0] * n_tz + [8.0] * n_tz), rtol=0.0, atol=0.0)
