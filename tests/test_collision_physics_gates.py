from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.collisions import (
    _V3_SQRTPI,
    _psi_chandra,
    apply_pitch_angle_scattering_v3,
    make_pitch_angle_scattering_v3_operator,
    nu_d_hat_pitch_angle_scattering_v3,
    polynomial_interpolation_matrix_np,
    rosenbluth_potential_terms_v3_np,
)
from sfincs_jax.xgrid import make_x_grid


def _pas_operator():
    return make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray([0.35, 0.9, 1.7], dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        m_hats=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        t_hats=jnp.asarray([1.0], dtype=jnp.float64),
        nu_n=0.7,
        krook=0.0,
        n_xi_for_x=jnp.asarray([4, 3, 2], dtype=jnp.int32),
        n_xi=5,
    )


def test_pas_l0_is_null_and_inactive_legendre_slots_are_masked() -> None:
    op = _pas_operator()
    f = np.ones((1, 3, 5, 2, 2), dtype=np.float64)
    out = np.asarray(apply_pitch_angle_scattering_v3(op, jnp.asarray(f)))

    np.testing.assert_allclose(out[:, :, 0, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 0, 4, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 1, 3:, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 2, 2:, :, :], 0.0, atol=0.0, rtol=0.0)
    assert np.all(out[:, :, 1, :, :] > 0.0)


def test_pas_legendre_eigenvalues_follow_l_lplus1_over_two() -> None:
    op = _pas_operator()
    coef = np.asarray(op.coef[0])

    # With krook=0, L=1 has factor 1, so higher active-L coefficients should
    # follow L(L+1)/2 relative to L=1 at each x.
    for ix, n_l_active in enumerate([4, 3, 2]):
        base = coef[ix, 1]
        assert base > 0.0
        for ell in range(1, n_l_active):
            expected = 0.5 * ell * (ell + 1.0)
            np.testing.assert_allclose(coef[ix, ell] / base, expected, rtol=2e-15, atol=2e-15)


def test_chandrasekhar_function_matches_small_x_limit() -> None:
    x = jnp.asarray([1.0e-12, 1.0e-10, 1.0e-8], dtype=jnp.float64)
    psi = np.asarray(_psi_chandra(x))
    expected_ratio = 2.0 / (3.0 * float(_V3_SQRTPI))
    np.testing.assert_allclose(psi / np.asarray(x), expected_ratio, rtol=1.0e-8, atol=1.0e-12)
    assert np.all(psi > 0.0)


def test_polynomial_interpolation_matrix_is_identity_on_matching_nodes() -> None:
    xk = np.asarray([0.2, 0.7, 1.4, 2.2], dtype=np.float64)
    alpxk = np.exp(-(xk * xk))
    mat = polynomial_interpolation_matrix_np(xk=xk, x=xk.copy(), alpxk=alpxk, alpx=alpxk.copy())
    np.testing.assert_allclose(mat, np.eye(xk.size), rtol=0.0, atol=1.0e-13)


def test_pas_deflection_frequency_has_coulomb_scaling() -> None:
    x = jnp.asarray([0.4, 1.0, 2.0], dtype=jnp.float64)
    common = {
        "x": x,
        "m_hats": jnp.asarray([1.0], dtype=jnp.float64),
        "t_hats": jnp.asarray([1.0], dtype=jnp.float64),
    }

    base = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([1.0], dtype=jnp.float64),
            n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        )
    )
    double_density = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([1.0], dtype=jnp.float64),
            n_hats=jnp.asarray([2.0], dtype=jnp.float64),
        )
    )
    double_charge = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([2.0], dtype=jnp.float64),
            n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        )
    )

    assert np.all(np.isfinite(base))
    assert np.all(base > 0.0)
    np.testing.assert_allclose(double_density / base, 2.0, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(double_charge / base, 16.0, rtol=2.0e-15, atol=2.0e-15)


def test_weighted_barycentric_interpolation_is_exact_for_polynomials() -> None:
    xk = np.asarray([0.15, 0.7, 1.4, 2.3], dtype=np.float64)
    x = np.asarray([0.2, 0.9, 1.8], dtype=np.float64)
    alpxk = np.exp(-(xk * xk))
    alpx = np.exp(-(x * x))
    mat = polynomial_interpolation_matrix_np(xk=xk, x=x, alpxk=alpxk, alpx=alpx)

    def polynomial(y: np.ndarray) -> np.ndarray:
        return 1.0 - 2.0 * y + 0.5 * y * y + 0.1 * y * y * y

    source = alpxk * polynomial(xk)
    expected = alpx * polynomial(x)
    np.testing.assert_allclose(mat @ source, expected, rtol=2.0e-15, atol=2.0e-15)


def test_rosenbluth_analytic_terms_match_quadpack_reference() -> None:
    xg = make_x_grid(n=3, k=0.0, include_point_at_x0=False)
    kwargs = {
        "x": xg.x,
        "x_weights": xg.dx_weights(),
        "x_grid_k": 0.0,
        "xg": xg,
        "z_s": np.asarray([1.0], dtype=np.float64),
        "m_hats": np.asarray([1.0], dtype=np.float64),
        "n_hats": np.asarray([1.0], dtype=np.float64),
        "t_hats": np.asarray([1.0], dtype=np.float64),
        "nl": 2,
    }

    quadpack = rosenbluth_potential_terms_v3_np(**kwargs, method="quadpack")
    analytic = rosenbluth_potential_terms_v3_np(**kwargs, method="analytic")

    assert quadpack.shape == (1, 1, 2, 3, 3)
    assert np.all(np.isfinite(quadpack))
    assert np.all(np.isfinite(analytic))
    np.testing.assert_allclose(analytic, quadpack, rtol=2.0e-13, atol=2.0e-13)
