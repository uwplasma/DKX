import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning

from dkx.collisions import (
    ROSENBLUTH_METHODS,
    _monomial_int_upper,
    resolve_rosenbluth_method,
    rosenbluth_potential_terms_v3_np,
)
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text
from dkx.xgrid import make_x_grid

REF = Path(__file__).parent / "ref"
_FP_DECK = "fp_1species_FPCollisions_noEr_tiny_cs3"


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


# --- selection routes -------------------------------------------------------
#
# The repo rule is that a solver route is reachable from a namelist key or an
# API argument; the environment variable is an override, never the only way in.


def _fp_operator(rosenbluth_line: str = "") -> KineticOperator:
    # NL = Nxi = 6 so the assembled operator actually reaches the hybrid
    # route's analytic L >= 4 blocks (below that hybrid is QUADPACK by
    # construction and the comparison would be vacuous).
    text = (
        (REF / f"{_FP_DECK}.input.namelist")
        .read_text()
        .replace("NL = 3", "NL = 6")
        .replace("Nxi = 4", "Nxi = 6")
    )
    if rosenbluth_line:
        text = text.replace(
            "  Nxi_for_x_option = 0", f"  Nxi_for_x_option = 0\n  {rosenbluth_line}"
        )
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


def test_rosenbluth_method_resolution_prefers_the_explicit_route() -> None:
    assert resolve_rosenbluth_method(None) == "quadpack"
    for name in ROSENBLUTH_METHODS:
        assert resolve_rosenbluth_method(name.upper()) == name
        assert resolve_rosenbluth_method(f"  {name}  ") == name
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        resolve_rosenbluth_method("quadpak")


def test_env_var_overrides_only_the_unset_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DKX_ROSENBLUTH_METHOD", "hybrid")
    assert resolve_rosenbluth_method(None) == "hybrid"
    # An explicit namelist/API selection wins over the environment.
    assert resolve_rosenbluth_method("quadpack") == "quadpack"
    monkeypatch.setenv("DKX_ROSENBLUTH_METHOD", "not-a-method")
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        resolve_rosenbluth_method(None)


def test_builder_api_argument_selects_the_hybrid_rosenbluth_path() -> None:
    from dkx.collisions import make_fokker_planck_v3_operator
    from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices

    sg = make_speed_grid(n_x=4, k=0.0)
    x = np.asarray(sg.x, dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=0.0)
    common = {
        "x": x,
        "x_weights": np.asarray(sg.dx_weights(0.0), dtype=np.float64),
        "ddx": ddx,
        "d2dx2": d2dx2,
        "x_grid_k": 0.0,
        "z_s": np.array([1.0]),
        "m_hats": np.array([1.0]),
        "n_hats": np.array([1.0]),
        "t_hats": np.array([1.0]),
        "nu_n": 0.01,
        "krook": 0.0,
        "n_xi": 6,
        "nl": 6,
        "n_xi_for_x": np.full(4, 6, dtype=np.int32),
    }
    base = make_fokker_planck_v3_operator(**common)
    same = make_fokker_planck_v3_operator(**common, rosenbluth_method="quadpack")
    hybrid = make_fokker_planck_v3_operator(**common, rosenbluth_method="hybrid")

    # The default and an explicit 'quadpack' are the same operator; 'hybrid'
    # reaches a different quadrature -- with nl = 6 the L >= 4 blocks take the
    # analytic moments -- without the environment variable ever being set.
    np.testing.assert_array_equal(np.asarray(same.mat), np.asarray(base.mat))
    assert not np.array_equal(np.asarray(hybrid.mat), np.asarray(base.mat))
    np.testing.assert_allclose(
        np.asarray(hybrid.mat), np.asarray(base.mat), rtol=1e-6, atol=1e-10
    )

    with pytest.raises(ValueError, match="RosenbluthMethod"):
        make_fokker_planck_v3_operator(**common, rosenbluth_method="quadpak")


def test_namelist_key_selects_the_hybrid_rosenbluth_path() -> None:
    base = _fp_operator()
    hybrid = _fp_operator("RosenbluthMethod = 'hybrid'")
    quadpack = _fp_operator("RosenbluthMethod = 'quadpack'")

    np.testing.assert_array_equal(np.asarray(quadpack.fp.mat), np.asarray(base.fp.mat))
    assert not np.array_equal(np.asarray(hybrid.fp.mat), np.asarray(base.fp.mat))
    np.testing.assert_allclose(
        np.asarray(hybrid.fp.mat), np.asarray(base.fp.mat), rtol=1e-6, atol=1e-10
    )


def test_namelist_key_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        _fp_operator("RosenbluthMethod = 'quadpak'")
