from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.collisions import apply_pitch_angle_scattering_v3, make_pitch_angle_scattering_v3_operator


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
