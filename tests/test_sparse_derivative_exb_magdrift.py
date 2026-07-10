from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.operators.profile_exb import (
    ExBThetaV3Operator,
    ExBZetaV3Operator,
    apply_exb_theta_v3,
    apply_exb_zeta_v3,
)
from sfincs_jax.operators.profile_magnetic_drifts import (
    MagneticDriftThetaV3Operator,
    MagneticDriftXiDotV3Operator,
    MagneticDriftZetaV3Operator,
    apply_magnetic_drift_theta_v3,
    apply_magnetic_drift_theta_v3_offdiag2,
    apply_magnetic_drift_xidot_v3,
    apply_magnetic_drift_xidot_v3_offdiag2,
    apply_magnetic_drift_zeta_v3,
    apply_magnetic_drift_zeta_v3_offdiag2,
)
from sfincs_jax.discretization.periodic_stencil import extract_sparse_row_stencil


def _periodic_first_derivative_matrix(n: int) -> np.ndarray:
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        d[i, (i + 1) % n] = 0.5
        d[i, (i - 1) % n] = -0.5
    return d


def _random_f(*, n_x: int, n_xi: int, n_theta: int, n_zeta: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    f = rng.normal(size=(1, n_x, n_xi, n_theta, n_zeta)).astype(np.float64)
    return jnp.asarray(f)


def test_exb_sparse_theta_matches_dense() -> None:
    n_theta, n_zeta = 9, 7
    ddtheta = _periodic_first_derivative_matrix(n_theta)
    cols, vals = extract_sparse_row_stencil(ddtheta)
    f = _random_f(n_x=3, n_xi=5, n_theta=n_theta, n_zeta=n_zeta)

    op_dense = ExBThetaV3Operator(
        alpha=jnp.asarray(1.0),
        delta=jnp.asarray(0.2),
        dphi_hat_dpsi_hat=jnp.asarray(0.1),
        ddtheta=jnp.asarray(ddtheta),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat_sub_zeta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64) * 0.3,
        use_dkes_exb_drift=jnp.asarray(False),
        fsab_hat2=jnp.asarray(1.0),
        n_xi_for_x=jnp.asarray([5, 4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddtheta_sparse_cols=jnp.asarray(cols, dtype=jnp.int32),
        ddtheta_sparse_vals=jnp.asarray(vals, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_exb_theta_v3(op_dense, f))
    y_sparse = np.asarray(apply_exb_theta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_exb_sparse_zeta_matches_dense() -> None:
    n_theta, n_zeta = 7, 9
    ddzeta = _periodic_first_derivative_matrix(n_zeta)
    cols, vals = extract_sparse_row_stencil(ddzeta)
    f = _random_f(n_x=2, n_xi=4, n_theta=n_theta, n_zeta=n_zeta)

    op_dense = ExBZetaV3Operator(
        alpha=jnp.asarray(1.0),
        delta=jnp.asarray(0.2),
        dphi_hat_dpsi_hat=jnp.asarray(0.1),
        ddzeta=jnp.asarray(ddzeta),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat_sub_theta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64) * -0.2,
        use_dkes_exb_drift=jnp.asarray(False),
        fsab_hat2=jnp.asarray(1.0),
        n_xi_for_x=jnp.asarray([4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddzeta_sparse_cols=jnp.asarray(cols, dtype=jnp.int32),
        ddzeta_sparse_vals=jnp.asarray(vals, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_exb_zeta_v3(op_dense, f))
    y_sparse = np.asarray(apply_exb_zeta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_magnetic_drift_sparse_theta_matches_dense() -> None:
    n_theta, n_zeta = 9, 7
    dd_plus = _periodic_first_derivative_matrix(n_theta)
    dd_minus = -0.7 * dd_plus
    cols_plus, vals_plus = extract_sparse_row_stencil(dd_plus)
    cols_minus, vals_minus = extract_sparse_row_stencil(dd_minus)
    f = _random_f(n_x=3, n_xi=5, n_theta=n_theta, n_zeta=n_zeta)
    base_2d = jnp.ones((n_theta, n_zeta), dtype=jnp.float64)

    op_dense = MagneticDriftThetaV3Operator(
        delta=jnp.asarray(0.1),
        t_hat=jnp.asarray(1.0),
        z=jnp.asarray(1.0),
        x=jnp.asarray(np.linspace(0.1, 2.0, 3), dtype=jnp.float64),
        ddtheta_plus=jnp.asarray(dd_plus),
        ddtheta_minus=jnp.asarray(dd_minus),
        d_hat=base_2d,
        b_hat=base_2d,
        b_hat_sub_zeta=base_2d * 0.2,
        b_hat_sub_psi=base_2d * 0.1,
        db_hat_dzeta=base_2d * 0.4,
        db_hat_dpsi_hat=base_2d * -0.3,
        db_hat_sub_psi_dzeta=base_2d * 0.6,
        db_hat_sub_zeta_dpsi_hat=base_2d * -0.2,
        n_xi_for_x=jnp.asarray([5, 4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddtheta_plus_sparse_cols=jnp.asarray(cols_plus, dtype=jnp.int32),
        ddtheta_plus_sparse_vals=jnp.asarray(vals_plus, dtype=jnp.float64),
        ddtheta_minus_sparse_cols=jnp.asarray(cols_minus, dtype=jnp.int32),
        ddtheta_minus_sparse_vals=jnp.asarray(vals_minus, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_magnetic_drift_theta_v3(op_dense, f))
    y_sparse = np.asarray(apply_magnetic_drift_theta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_magnetic_drift_sparse_zeta_matches_dense() -> None:
    n_theta, n_zeta = 7, 9
    dd_plus = _periodic_first_derivative_matrix(n_zeta)
    dd_minus = -0.5 * dd_plus
    cols_plus, vals_plus = extract_sparse_row_stencil(dd_plus)
    cols_minus, vals_minus = extract_sparse_row_stencil(dd_minus)
    f = _random_f(n_x=2, n_xi=4, n_theta=n_theta, n_zeta=n_zeta)
    base_2d = jnp.ones((n_theta, n_zeta), dtype=jnp.float64)

    op_dense = MagneticDriftZetaV3Operator(
        delta=jnp.asarray(0.1),
        t_hat=jnp.asarray(1.0),
        z=jnp.asarray(1.0),
        x=jnp.asarray(np.linspace(0.1, 2.0, 2), dtype=jnp.float64),
        ddzeta_plus=jnp.asarray(dd_plus),
        ddzeta_minus=jnp.asarray(dd_minus),
        d_hat=base_2d,
        b_hat=base_2d,
        b_hat_sub_theta=base_2d * -0.3,
        b_hat_sub_psi=base_2d * 0.2,
        db_hat_dtheta=base_2d * 0.7,
        db_hat_dpsi_hat=base_2d * -0.1,
        db_hat_sub_theta_dpsi_hat=base_2d * 0.4,
        db_hat_sub_psi_dtheta=base_2d * -0.8,
        n_xi_for_x=jnp.asarray([4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddzeta_plus_sparse_cols=jnp.asarray(cols_plus, dtype=jnp.int32),
        ddzeta_plus_sparse_vals=jnp.asarray(vals_plus, dtype=jnp.float64),
        ddzeta_minus_sparse_cols=jnp.asarray(cols_minus, dtype=jnp.int32),
        ddzeta_minus_sparse_vals=jnp.asarray(vals_minus, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_magnetic_drift_zeta_v3(op_dense, f))
    y_sparse = np.asarray(apply_magnetic_drift_zeta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def _small_exb_theta_operator() -> ExBThetaV3Operator:
    ddtheta = _periodic_first_derivative_matrix(3)
    shape = (3, 2)
    return ExBThetaV3Operator(
        alpha=jnp.asarray(1.3, dtype=jnp.float64),
        delta=jnp.asarray(0.2, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(-0.7, dtype=jnp.float64),
        ddtheta=jnp.asarray(ddtheta),
        d_hat=jnp.ones(shape, dtype=jnp.float64),
        b_hat=2.0 * jnp.ones(shape, dtype=jnp.float64),
        b_hat_sub_zeta=0.25 * jnp.ones(shape, dtype=jnp.float64),
        use_dkes_exb_drift=False,
        fsab_hat2=jnp.asarray(5.0, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
    )


def _small_exb_zeta_operator() -> ExBZetaV3Operator:
    ddzeta = _periodic_first_derivative_matrix(2)
    shape = (3, 2)
    return ExBZetaV3Operator(
        alpha=jnp.asarray(1.3, dtype=jnp.float64),
        delta=jnp.asarray(0.2, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(-0.7, dtype=jnp.float64),
        ddzeta=jnp.asarray(ddzeta),
        d_hat=jnp.ones(shape, dtype=jnp.float64),
        b_hat=2.0 * jnp.ones(shape, dtype=jnp.float64),
        b_hat_sub_theta=-0.35 * jnp.ones(shape, dtype=jnp.float64),
        use_dkes_exb_drift=False,
        fsab_hat2=jnp.asarray(5.0, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
    )


def _small_magnetic_theta_operator(*, drop_l2_couplings: bool = False) -> MagneticDriftThetaV3Operator:
    shape = (3, 2)
    ones = jnp.ones(shape, dtype=jnp.float64)
    return MagneticDriftThetaV3Operator(
        delta=jnp.asarray(0.4, dtype=jnp.float64),
        t_hat=jnp.asarray(1.2, dtype=jnp.float64),
        z=jnp.asarray(1.0, dtype=jnp.float64),
        x=jnp.asarray([0.3, 0.9], dtype=jnp.float64),
        ddtheta_plus=jnp.asarray(_periodic_first_derivative_matrix(3)),
        ddtheta_minus=-0.5 * jnp.asarray(_periodic_first_derivative_matrix(3)),
        d_hat=ones,
        b_hat=1.8 * ones,
        b_hat_sub_zeta=0.2 * ones,
        b_hat_sub_psi=-0.3 * ones,
        db_hat_dzeta=0.4 * ones,
        db_hat_dpsi_hat=-0.6 * ones,
        db_hat_sub_psi_dzeta=0.15 * ones,
        db_hat_sub_zeta_dpsi_hat=-0.25 * ones,
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
        drop_l2_couplings=drop_l2_couplings,
    )


def _small_magnetic_zeta_operator(*, drop_l2_couplings: bool = False) -> MagneticDriftZetaV3Operator:
    shape = (3, 2)
    ones = jnp.ones(shape, dtype=jnp.float64)
    return MagneticDriftZetaV3Operator(
        delta=jnp.asarray(0.4, dtype=jnp.float64),
        t_hat=jnp.asarray(1.2, dtype=jnp.float64),
        z=jnp.asarray(1.0, dtype=jnp.float64),
        x=jnp.asarray([0.3, 0.9], dtype=jnp.float64),
        ddzeta_plus=jnp.asarray(_periodic_first_derivative_matrix(2)),
        ddzeta_minus=-0.75 * jnp.asarray(_periodic_first_derivative_matrix(2)),
        d_hat=ones,
        b_hat=1.8 * ones,
        b_hat_sub_theta=0.35 * ones,
        b_hat_sub_psi=-0.3 * ones,
        db_hat_dtheta=0.45 * ones,
        db_hat_dpsi_hat=-0.6 * ones,
        db_hat_sub_theta_dpsi_hat=0.2 * ones,
        db_hat_sub_psi_dtheta=-0.1 * ones,
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
        drop_l2_couplings=drop_l2_couplings,
    )


def _small_magnetic_xidot_operator(*, drop_l2_couplings: bool = False) -> MagneticDriftXiDotV3Operator:
    shape = (3, 2)
    ones = jnp.ones(shape, dtype=jnp.float64)
    return MagneticDriftXiDotV3Operator(
        delta=jnp.asarray(0.4, dtype=jnp.float64),
        t_hat=jnp.asarray(1.2, dtype=jnp.float64),
        z=jnp.asarray(1.0, dtype=jnp.float64),
        x=jnp.asarray([0.3, 0.9], dtype=jnp.float64),
        d_hat=ones,
        b_hat=1.8 * ones,
        db_hat_dtheta=0.45 * ones,
        db_hat_dzeta=-0.35 * ones,
        db_hat_sub_psi_dzeta=0.15 * ones,
        db_hat_sub_zeta_dpsi_hat=-0.25 * ones,
        db_hat_sub_theta_dpsi_hat=0.2 * ones,
        db_hat_sub_psi_dtheta=-0.1 * ones,
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
        drop_l2_couplings=drop_l2_couplings,
    )


def test_exb_apply_validates_tensor_shape_and_operator_axes() -> None:
    theta = _small_exb_theta_operator()
    zeta = _small_exb_zeta_operator()

    with pytest.raises(ValueError, match="shape"):
        apply_exb_theta_v3(theta, jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="theta axis"):
        apply_exb_theta_v3(theta, jnp.zeros((1, 2, 3, 2, 2), dtype=jnp.float64))
    with pytest.raises(ValueError, match="zeta axis"):
        apply_exb_theta_v3(theta, jnp.zeros((1, 2, 3, 3, 3), dtype=jnp.float64))
    with pytest.raises(ValueError, match="x axis"):
        apply_exb_theta_v3(theta, jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float64))

    with pytest.raises(ValueError, match="shape"):
        apply_exb_zeta_v3(zeta, jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="zeta axis"):
        apply_exb_zeta_v3(zeta, jnp.zeros((1, 2, 3, 3, 3), dtype=jnp.float64))
    with pytest.raises(ValueError, match="theta axis"):
        apply_exb_zeta_v3(zeta, jnp.zeros((1, 2, 3, 2, 2), dtype=jnp.float64))
    with pytest.raises(ValueError, match="x axis"):
        apply_exb_zeta_v3(zeta, jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float64))


def test_exb_pytree_unflatten_accepts_historical_boolean_aux() -> None:
    theta = _small_exb_theta_operator()
    zeta = _small_exb_zeta_operator()

    theta_children, _ = theta.tree_flatten()
    restored_theta = ExBThetaV3Operator.tree_unflatten(True, theta_children)
    assert restored_theta.use_dkes_exb_drift is True
    assert restored_theta.ddtheta_stencil_shifts == ()
    assert restored_theta.ddtheta_stencil_coeffs == ()

    zeta_children, _ = zeta.tree_flatten()
    restored_zeta = ExBZetaV3Operator.tree_unflatten(True, zeta_children)
    assert restored_zeta.use_dkes_exb_drift is True
    assert restored_zeta.ddzeta_stencil_shifts == ()
    assert restored_zeta.ddzeta_stencil_coeffs == ()


def test_magnetic_drift_pytree_unflatten_accepts_historical_aux_layouts() -> None:
    theta = _small_magnetic_theta_operator(drop_l2_couplings=True)
    theta_children, theta_aux = theta.tree_flatten()
    restored_theta = MagneticDriftThetaV3Operator.tree_unflatten(theta_aux[:4], theta_children)
    assert restored_theta.drop_l2_couplings is False
    restored_theta_none = MagneticDriftThetaV3Operator.tree_unflatten(None, theta_children)
    assert restored_theta_none.drop_l2_couplings is False

    zeta = _small_magnetic_zeta_operator(drop_l2_couplings=True)
    zeta_children, zeta_aux = zeta.tree_flatten()
    restored_zeta = MagneticDriftZetaV3Operator.tree_unflatten(zeta_aux[:4], zeta_children)
    assert restored_zeta.drop_l2_couplings is False
    restored_zeta_none = MagneticDriftZetaV3Operator.tree_unflatten(None, zeta_children)
    assert restored_zeta_none.drop_l2_couplings is False

    xidot = _small_magnetic_xidot_operator(drop_l2_couplings=True)
    xidot_children, _ = xidot.tree_flatten()
    restored_xidot = MagneticDriftXiDotV3Operator.tree_unflatten(None, xidot_children)
    assert restored_xidot.drop_l2_couplings is False


def test_magnetic_drift_drop_l2_equals_full_minus_offdiagonal_piece() -> None:
    """The diagonal-L magnetic drift is the full operator minus the |Delta L|=2 piece."""

    f = _random_f(n_x=2, n_xi=3, n_theta=3, n_zeta=2)

    theta = _small_magnetic_theta_operator()
    theta_drop = _small_magnetic_theta_operator(drop_l2_couplings=True)
    np.testing.assert_allclose(
        np.asarray(apply_magnetic_drift_theta_v3(theta, f)),
        np.asarray(apply_magnetic_drift_theta_v3(theta_drop, f) + apply_magnetic_drift_theta_v3_offdiag2(theta, f)),
        rtol=0,
        atol=1e-12,
    )

    zeta = _small_magnetic_zeta_operator()
    zeta_drop = _small_magnetic_zeta_operator(drop_l2_couplings=True)
    np.testing.assert_allclose(
        np.asarray(apply_magnetic_drift_zeta_v3(zeta, f)),
        np.asarray(apply_magnetic_drift_zeta_v3(zeta_drop, f) + apply_magnetic_drift_zeta_v3_offdiag2(zeta, f)),
        rtol=0,
        atol=1e-12,
    )

    xidot = _small_magnetic_xidot_operator()
    xidot_drop = _small_magnetic_xidot_operator(drop_l2_couplings=True)
    np.testing.assert_allclose(
        np.asarray(apply_magnetic_drift_xidot_v3(xidot, f)),
        np.asarray(apply_magnetic_drift_xidot_v3(xidot_drop, f) + apply_magnetic_drift_xidot_v3_offdiag2(xidot, f)),
        rtol=0,
        atol=1e-12,
    )


def test_magnetic_drift_apply_validates_tensor_shape_and_operator_axes() -> None:
    theta = _small_magnetic_theta_operator()
    zeta = _small_magnetic_zeta_operator()
    xidot = _small_magnetic_xidot_operator()

    with pytest.raises(ValueError, match="shape"):
        apply_magnetic_drift_theta_v3_offdiag2(theta, jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="theta axis"):
        apply_magnetic_drift_theta_v3_offdiag2(theta, jnp.zeros((1, 2, 3, 2, 2), dtype=jnp.float64))
    theta_short_x = replace(theta, n_xi_for_x=jnp.asarray([3], dtype=jnp.int32))
    with pytest.raises(ValueError, match="x axis"):
        apply_magnetic_drift_theta_v3_offdiag2(theta_short_x, jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float64))

    with pytest.raises(ValueError, match="shape"):
        apply_magnetic_drift_zeta_v3_offdiag2(zeta, jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="zeta axis"):
        apply_magnetic_drift_zeta_v3_offdiag2(zeta, jnp.zeros((1, 2, 3, 3, 3), dtype=jnp.float64))
    zeta_short_x = replace(zeta, n_xi_for_x=jnp.asarray([3], dtype=jnp.int32))
    with pytest.raises(ValueError, match="x axis"):
        apply_magnetic_drift_zeta_v3_offdiag2(zeta_short_x, jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float64))

    with pytest.raises(ValueError, match="shape"):
        apply_magnetic_drift_xidot_v3_offdiag2(xidot, jnp.zeros((2, 3)))
    xidot_short_x = replace(xidot, n_xi_for_x=jnp.asarray([3], dtype=jnp.int32))
    with pytest.raises(ValueError, match="x axis"):
        apply_magnetic_drift_xidot_v3_offdiag2(xidot_short_x, jnp.zeros((1, 1, 3, 3, 2), dtype=jnp.float64))
