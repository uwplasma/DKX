from __future__ import annotations

import numpy as np
import pytest

from dkx.moments import (
    FluxSurface,
    StateLayout,
    VelocityGrid,
    legendre_tail_relative_l2,
    legendre_tail_relative_l2_batch,
)


def _containers():
    import jax.numpy as jnp

    layout = StateLayout(n_species=1, n_x=2, n_xi=4, n_theta=1, n_zeta=1)
    vgrid = VelocityGrid(
        x=jnp.asarray([0.5, 1.5]),
        x_weights=jnp.ones(2),
        n_xi_for_x=jnp.asarray([3, 4], dtype=jnp.int32),
    )
    surface = FluxSurface(
        theta_weights=jnp.ones(1),
        zeta_weights=jnp.ones(1),
        b_hat=jnp.ones((1, 1)),
        d_hat=-jnp.ones((1, 1)),
        db_hat_dtheta=jnp.zeros((1, 1)),
        db_hat_dzeta=jnp.zeros((1, 1)),
        b_hat_sub_theta=jnp.zeros((1, 1)),
        b_hat_sub_zeta=jnp.zeros((1, 1)),
        fsab_hat2=jnp.asarray(1.0),
    )
    return layout, vgrid, surface


def test_legendre_tail_uses_last_two_active_modes_and_ignores_padding() -> None:
    layout, vgrid, surface = _containers()
    f = np.asarray(
        [
            [3.0, 4.0, 12.0, 999.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ).reshape(layout.f_shape)
    state = f.reshape(-1)

    actual = np.asarray(legendre_tail_relative_l2(layout, vgrid, surface, state))[:, 0]
    weights = 2.0 / (2.0 * np.arange(4) + 1.0)
    expected = np.asarray(
        [
            np.sqrt(
                np.sum((f[0, 0, 1:3, 0, 0] ** 2) * weights[1:3])
                / np.sum((f[0, 0, :3, 0, 0] ** 2) * weights[:3])
            ),
            np.sqrt(
                np.sum((f[0, 1, 2:4, 0, 0] ** 2) * weights[2:4])
                / np.sum((f[0, 1, :4, 0, 0] ** 2) * weights)
            ),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14)
    tiny = np.asarray(
        legendre_tail_relative_l2(layout, vgrid, surface, state * 1.0e-200)
    )[:, 0]
    np.testing.assert_allclose(tiny, expected, rtol=1.0e-14)


def test_legendre_tail_batch_zero_state_is_finite() -> None:
    layout, vgrid, surface = _containers()
    states = np.zeros((2, layout.total_size))
    actual = np.asarray(legendre_tail_relative_l2_batch(layout, vgrid, surface, states))
    assert actual.shape == (2, 2, 1)
    np.testing.assert_array_equal(actual, 0.0)


def test_legendre_tail_rejects_invalid_contracts() -> None:
    layout, vgrid, surface = _containers()
    with pytest.raises(ValueError, match="tail_modes"):
        legendre_tail_relative_l2_batch(
            layout, vgrid, surface, np.zeros((1, layout.total_size)), tail_modes=0
        )
    with pytest.raises(ValueError, match="x_full_stack"):
        legendre_tail_relative_l2_batch(
            layout, vgrid, surface, np.zeros((layout.total_size,))
        )
