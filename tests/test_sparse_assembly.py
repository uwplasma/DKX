from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.v3_driver as v3_driver
from sfincs_jax.solvers.preconditioner_xblock_tz_sparse import (
    assemble_rhsmode1_fp_xblock_tz_sparse_matrix,
    get_rhsmode1_fp_xblock_assembled_host_cache,
    rhsmode1_fp_xblock_tz_sparse_diagonal,
)
from sfincs_jax.v3_driver import (
    _assemble_selected_theta_tz_operator,
    _assemble_selected_zeta_tz_operator,
    _build_sparse_ilu_from_matvec,
    _safe_inverse_diagonal_np,
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

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = _build_sparse_ilu_from_matvec(
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
    inv = _safe_inverse_diagonal_np(np.asarray([2.0, 0.0, -1.0e-12]), floor=1.0e-6)
    assert inv is not None
    np.testing.assert_allclose(inv, np.asarray([0.5, 1.0e6, -1.0e6]))
    assert _safe_inverse_diagonal_np(np.asarray([1.0, np.nan]), floor=1.0e-6) is None
    assert _safe_inverse_diagonal_np(np.asarray([1.0, 0.0]), floor=0.0) is None


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

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = _build_sparse_ilu_from_matvec(
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


def test_selected_theta_tz_operator_matches_expected_rows() -> None:
    dd_plus = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dd_minus = np.asarray([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False, True], [False, True, False]])

    op = _assemble_selected_theta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
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


def test_selected_zeta_tz_operator_matches_expected_rows() -> None:
    dd_plus = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]], dtype=np.float64)
    dd_minus = np.asarray([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0], [0.0, -6.0, -7.0]], dtype=np.float64)
    use_plus = np.asarray([[True, False, True], [False, True, False]])

    op = _assemble_selected_zeta_tz_operator(dd_plus=dd_plus, dd_minus=dd_minus, use_plus=use_plus)
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

    assert v3_driver._get_rhsmode1_fp_xblock_assembled_host_cache is get_rhsmode1_fp_xblock_assembled_host_cache
    assert v3_driver._assemble_rhsmode1_fp_xblock_tz_sparse_matrix is assemble_rhsmode1_fp_xblock_tz_sparse_matrix
    assert v3_driver._rhsmode1_fp_xblock_tz_sparse_diagonal is rhsmode1_fp_xblock_tz_sparse_diagonal

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
