import warnings

import numpy as np
from scipy.integrate import IntegrationWarning

from sfincs_jax.collisions import _monomial_int_upper, rosenbluth_potential_terms_v3_np
from sfincs_jax.xgrid import make_x_grid


def _east_three_species_case(nl: int) -> tuple[dict[str, object], object]:
    xg = make_x_grid(n=12, k=0.0, include_point_at_x0=False)
    kwargs = {
        "x": xg.x,
        "x_weights": xg.dx_weights(),
        "x_grid_k": 0.0,
        "xg": xg,
        "z_s": np.array([-1.0, 1.0, 6.0]),
        "m_hats": np.array([5.4461702149014566e-4, 2.0, 12.0]),
        "n_hats": np.array([0.17326127575229972, 0.13860902060183977, 0.005775375858409991]),
        "t_hats": np.full(3, 1.7221796790605068),
        "nl": nl,
    }
    return kwargs, xg


def test_negative_power_upper_moments_cover_small_and_large_species_speed() -> None:
    # 80-digit mpmath references for Gamma((n+1)/2, xb**2)/2.  The first
    # case exercises the sharply peaked small-x electron/ion integral and the
    # second the exponentially small large-x continuation.
    cases = (
        (4.929789984040181e-4, -14, 7.573333596728916e41),
        (22.88985968775449, -14, 5.642165067724352e-249),
    )
    for xb, power, expected in cases:
        got = _monomial_int_upper(xb, power)
        assert np.isfinite(got)
        assert got > 0.0
        assert np.isclose(got, expected, rtol=2e-13, atol=0.0)


def test_hybrid_rosenbluth_is_warning_free_and_keeps_low_l_quadpack_parity() -> None:
    kwargs, _ = _east_three_species_case(nl=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hybrid = rosenbluth_potential_terms_v3_np(**kwargs, method="hybrid")

    assert np.isfinite(hybrid).all()
    assert not any(isinstance(item.message, IntegrationWarning) for item in caught)

    low_l_kwargs = dict(kwargs)
    low_l_kwargs["nl"] = 4
    quadpack = rosenbluth_potential_terms_v3_np(**low_l_kwargs, method="quadpack")
    assert np.array_equal(hybrid[:, :, :4], quadpack)
