"""Equivalence tests: sfincs_jax.phase_space must reproduce the old implementations.

The referee for the Phase 3.1a consolidation is exact (1e-15 / bitwise)
agreement with the modules being replaced:

* ``sfincs_jax.grids.uniform_diff_matrices``
* ``sfincs_jax.discretization.xgrid`` (``make_x_grid``,
  ``make_x_polynomial_diff_matrices``)
* ``sfincs_jax.discretization.v3.grids_from_namelist``
* ``sfincs_jax.physics.collisions.polynomial_interpolation_matrix_np``
* the Legendre coupling formulas inlined in ``operators/profile_collisionless``

plus a few absolute checks (quadrature exactness, hardcoded nodes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sfincs_jax import phase_space
from sfincs_jax.discretization.v3 import grids_from_namelist
from sfincs_jax.discretization.xgrid import make_x_grid, make_x_polynomial_diff_matrices
from sfincs_jax.grids import uniform_diff_matrices
from sfincs_jax.namelist import parse_sfincs_input_text
from sfincs_jax.physics.collisions import polynomial_interpolation_matrix_np

TOL = dict(rtol=0.0, atol=1e-15)


def assert_exact(new, old) -> None:
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), **TOL)


# ---------------------------------------------------------------------------
# Uniform periodic diff matrices vs old sfincs_jax.grids.uniform_diff_matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", sorted(phase_space._PERIODIC_SCHEMES))
@pytest.mark.parametrize("n", [5, 6, 9, 16])
@pytest.mark.parametrize("x_max", [2 * math.pi, 2 * math.pi / 5, 1.0])
def test_uniform_periodic_diff_matrices_match_old(scheme: int, n: int, x_max: float) -> None:
    x_old, w_old, ddx_old, d2_old = uniform_diff_matrices(
        n=n, x_min=0.0, x_max=x_max, scheme=scheme
    )
    x_new, w_new, ddx_new, d2_new = phase_space.uniform_periodic_diff_matrices(
        n=n, x_min=0.0, x_max=x_max, scheme=scheme
    )
    assert_exact(x_new, x_old)
    assert_exact(w_new, w_old)
    assert_exact(ddx_new, ddx_old)
    assert_exact(d2_new, d2_old)


def test_uniform_periodic_diff_matrices_rejects_dropped_schemes() -> None:
    for scheme in (1, 2, 3, 11, 12, 13, 21, 30, 42, 52, 81, 92, 101, 112, 121, 131):
        with pytest.raises(ValueError):
            phase_space.uniform_periodic_diff_matrices(n=8, x_min=0.0, x_max=1.0, scheme=scheme)


def test_spectral_theta_derivative_is_exact_on_fourier_modes() -> None:
    """Absolute check: spectral collocation differentiates sin/cos exactly."""
    n = 15
    theta, w, ddtheta, d2 = phase_space.uniform_periodic_diff_matrices(
        n=n, x_min=0.0, x_max=2 * math.pi, scheme=20
    )
    theta = np.asarray(theta)
    for m in (1, 2, 5):
        np.testing.assert_allclose(
            np.asarray(ddtheta) @ np.sin(m * theta),
            m * np.cos(m * theta),
            rtol=0.0,
            atol=1e-11,
        )
    # Weights integrate trig polynomials exactly: int_0^{2pi} (1 + cos 3t) dt = 2 pi.
    np.testing.assert_allclose(
        float(np.asarray(w) @ (1.0 + np.cos(3 * theta))), 2 * math.pi, rtol=1e-14
    )


# ---------------------------------------------------------------------------
# Legendre pitch machinery vs the formulas inlined in the operators modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_xi", [1, 4, 16, 33])
def test_legendre_couplings_match_operator_inline_formulas(n_xi: int) -> None:
    # Same expressions as operators/profile_collisionless.py lines 275-276.
    ell = np.arange(n_xi, dtype=np.float64)
    coef_plus = (ell + 1.0) / (2.0 * ell + 3.0)
    coef_minus = np.where(ell > 0, ell / (2.0 * ell - 1.0), 0.0)
    assert_exact(phase_space.legendre_coupling_upper(n_xi), coef_plus)
    assert_exact(phase_space.legendre_coupling_lower(n_xi), coef_minus)
    # Lorentz eigenvalues l(l+1) as in operators/profile_collisions.py line 201.
    assert_exact(phase_space.lorentz_eigenvalues(n_xi), ell * (ell + 1.0))


def test_lorentz_eigenvalues_absolute_values() -> None:
    assert phase_space.lorentz_eigenvalues(4).tolist() == [0.0, 2.0, 6.0, 12.0]


# ---------------------------------------------------------------------------
# Speed grid vs old discretization/xgrid.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_x", [1, 4, 5, 8])
@pytest.mark.parametrize("k", [0.0, 1.0])
@pytest.mark.parametrize("include_x0", [False, True])
def test_speed_grid_matches_old_xgrid(n_x: int, k: float, include_x0: bool) -> None:
    old = make_x_grid(n=n_x, k=k, include_point_at_x0=include_x0)
    new = phase_space.make_speed_grid(n_x=n_x, k=k, include_point_at_x0=include_x0)
    assert_exact(new.x, old.x)
    assert_exact(new.gaussian_weights, old.gaussian_weights)
    assert_exact(new.poly_a, old.poly_a)
    assert_exact(new.poly_b, old.poly_b)
    assert_exact(new.poly_c, old.poly_c)
    assert_exact(new.dx_weights(k), old.dx_weights(k))


@pytest.mark.parametrize("n_x", [4, 7])
@pytest.mark.parametrize("k", [0.0, 2.0])
def test_speed_grid_diff_matrices_match_old(n_x: int, k: float) -> None:
    xg = make_x_grid(n=n_x, k=k, include_point_at_x0=False)
    x = np.asarray(xg.x, dtype=np.float64)
    ddx_old, d2_old = make_x_polynomial_diff_matrices(x, k=k)
    ddx_new, d2_new = phase_space.speed_grid_diff_matrices(x, k=k)
    assert_exact(ddx_new, ddx_old)
    assert_exact(d2_new, d2_old)


def test_speed_grid_nodes_hardcoded_reference() -> None:
    """Absolute check: n_x=5, k=0 nodes/weights computed once with the old code."""
    ref_x = np.array(
        [
            0.10024215196821541,
            0.48281396604620047,
            1.060949821525717,
            1.7797294185202617,
            2.6697603560876564,
        ]
    )
    ref_w = np.array(
        [
            0.24840615202844266,
            0.3923310666523999,
            0.2114181930760567,
            0.0332466603513439,
            0.00082485334451563,
        ]
    )
    grid = phase_space.make_speed_grid(n_x=5, k=0.0)
    np.testing.assert_allclose(grid.x, ref_x, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(grid.gaussian_weights, ref_w, rtol=0.0, atol=1e-14)


def test_speed_grid_quadrature_is_exact_for_polynomials() -> None:
    """Absolute check: n-point Gauss rule integrates x^m exp(-x^2) exactly
    for m <= 2n-1 against Gamma((m+1)/2)/2."""
    n_x = 6
    grid = phase_space.make_speed_grid(n_x=n_x, k=0.0)
    for m in range(2 * n_x):
        exact = 0.5 * math.gamma(0.5 * (m + 1))
        approx = float(np.sum(grid.gaussian_weights * grid.x**m))
        np.testing.assert_allclose(approx, exact, rtol=1e-12)


def test_polynomial_interpolation_matrix_matches_collisions_port() -> None:
    xg = phase_space.make_speed_grid(n_x=5, k=0.0)
    xk = np.asarray(xg.x)
    species_factor = math.sqrt(2.0 / 3.0)
    xb = xk * species_factor
    alpxk = np.exp(-(xk**2))
    alpx = np.exp(-(xb**2))
    old = polynomial_interpolation_matrix_np(xk=xk, x=xb, alpxk=alpxk, alpx=alpx)
    new = phase_space.polynomial_interpolation_matrix(xk=xk, x=xb, alpxk=alpxk, alpx=alpx)
    assert_exact(new, old)


def test_rosenbluth_potential_grid_size_rules() -> None:
    x = np.asarray(phase_space.make_speed_grid(n_x=5, k=0.0).x)
    # Default rule: ceil(max(x[-1], xMax) * NxPotentialsPerVth), createGrids.F90.
    expected = int(math.ceil(max(float(x[-1]), 5.0) * 40.0))
    assert phase_space.rosenbluth_potential_grid_size(x=x, n_x=5) == expected == 200
    assert (
        phase_space.rosenbluth_potential_grid_size(x=x, n_x=5, x_potentials_grid_scheme=3) == 6
    )
    assert phase_space.rosenbluth_potential_grid_size(x=x, n_x=5, monoenergetic=True) == 1


# ---------------------------------------------------------------------------
# Nxi_for_x ramps and full Grids container vs discretization/v3.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("option", [0, 1, 2, 3])
def test_n_xi_for_x_ramp_options(option: int) -> None:
    x = np.asarray(phase_space.make_speed_grid(n_x=7, k=0.0).x)
    n_xi, n_l = 16, 4
    # Reference: the loop in discretization/v3.py grids_from_namelist.
    ref = np.zeros((7,), dtype=int)
    if option == 0:
        ref[:] = n_xi
    elif option == 1:
        for j in range(7):
            temp = n_xi * (0.1 + 0.9 * x[j] / 2.0)
            ref[j] = max(4, n_l, min(int(temp), n_xi))
    elif option == 2:
        for j in range(7):
            temp = n_xi * (0.1 + 0.9 * ((x[j] / 2.0) ** 2))
            ref[j] = max(4, n_l, min(int(temp), n_xi))
    else:
        for j in range(7):
            temp = n_xi * (0.1 + 0.9 * x[j] / 2.0)
            ref[j] = max(3, n_l, int(temp))
    out = phase_space.n_xi_for_x_ramp(x=x, n_xi=n_xi, n_l=n_l, option=option)
    assert out.tolist() == ref.tolist()


def _v3_namelist(
    *,
    n_theta: int,
    n_zeta: int,
    n_x: int,
    n_xi: int,
    n_l: int,
    theta_scheme: int,
    zeta_scheme: int,
    drift_scheme: int,
    x_grid_scheme: int,
    n_xi_for_x_option: int,
    rhs_mode: int = 1,
):
    text = f"""
&general
  RHSMode = {rhs_mode}
/
&geometryParameters
  geometryScheme = 4
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nx = {n_x}
  Nxi = {n_xi}
  NL = {n_l}
/
&otherNumericalParameters
  thetaDerivativeScheme = {theta_scheme}
  zetaDerivativeScheme = {zeta_scheme}
  magneticDriftDerivativeScheme = {drift_scheme}
  xGridScheme = {x_grid_scheme}
  Nxi_for_x_option = {n_xi_for_x_option}
/
"""
    return parse_sfincs_input_text(text)


_GRIDS_FIELDS = (
    "theta",
    "zeta",
    "x",
    "theta_weights",
    "zeta_weights",
    "x_weights",
    "ddtheta",
    "ddzeta",
    "ddx",
    "d2dx2",
    "ddtheta_magdrift_plus",
    "ddtheta_magdrift_minus",
    "ddzeta_magdrift_plus",
    "ddzeta_magdrift_minus",
    "n_xi_for_x",
)


@pytest.mark.parametrize(
    (
        "n_theta",
        "n_zeta",
        "n_x",
        "n_xi",
        "theta_scheme",
        "zeta_scheme",
        "drift_scheme",
        "x_grid_scheme",
        "n_xi_for_x_option",
    ),
    [
        (7, 9, 4, 8, 2, 2, 3, 5, 1),  # v3 defaults, small
        (8, 6, 5, 12, 2, 2, 3, 5, 0),  # even Ntheta/Nzeta exercise force-odd
        (9, 7, 5, 12, 0, 0, 0, 5, 2),  # spectral, centered drifts
        (7, 7, 4, 10, 1, 1, 1, 6, 3),  # 2nd-order FD, scheme-1 upwinding, x0 point
        (7, 7, 4, 10, 2, 2, -2, 2, 1),  # negative drift scheme, xGridScheme 2
        (7, 1, 4, 8, 2, 2, 3, 5, 1),  # axisymmetric Nzeta = 1
    ],
)
def test_make_grids_matches_v3_grids_from_namelist(
    n_theta: int,
    n_zeta: int,
    n_x: int,
    n_xi: int,
    theta_scheme: int,
    zeta_scheme: int,
    drift_scheme: int,
    x_grid_scheme: int,
    n_xi_for_x_option: int,
) -> None:
    n_l = 4
    nml = _v3_namelist(
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_x=n_x,
        n_xi=n_xi,
        n_l=n_l,
        theta_scheme=theta_scheme,
        zeta_scheme=zeta_scheme,
        drift_scheme=drift_scheme,
        x_grid_scheme=x_grid_scheme,
        n_xi_for_x_option=n_xi_for_x_option,
    )
    old = grids_from_namelist(nml)
    new = phase_space.make_grids(
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        n_x=n_x,
        n_l=n_l,
        n_periods=5,  # geometryScheme=4
        theta_derivative_scheme=theta_scheme,
        zeta_derivative_scheme=zeta_scheme,
        magnetic_drift_derivative_scheme=drift_scheme,
        x_grid_scheme=x_grid_scheme,
        n_xi_for_x_option=n_xi_for_x_option,
    )
    for field in _GRIDS_FIELDS:
        assert_exact(getattr(new, field), getattr(old, field))
    assert new.n_xi == old.n_xi
    assert new.n_l == old.n_l
    assert new.n_theta == int(np.asarray(old.theta).size)
    assert new.n_zeta == int(np.asarray(old.zeta).size)


def test_make_grids_monoenergetic_matches_v3_rhsmode3() -> None:
    nml = _v3_namelist(
        n_theta=7,
        n_zeta=9,
        n_x=4,
        n_xi=8,
        n_l=4,
        theta_scheme=2,
        zeta_scheme=2,
        drift_scheme=3,
        x_grid_scheme=5,
        n_xi_for_x_option=1,
        rhs_mode=3,
    )
    old = grids_from_namelist(nml)
    new = phase_space.make_grids(
        n_theta=7,
        n_zeta=9,
        n_xi=8,
        n_x=4,
        n_l=4,
        n_periods=5,
        monoenergetic=True,
    )
    for field in _GRIDS_FIELDS:
        assert_exact(getattr(new, field), getattr(old, field))
    assert new.n_x == 1
    assert float(np.asarray(new.x)[0]) == 1.0
    assert float(np.asarray(new.x_weights)[0]) == math.exp(1.0)


def test_grids_container_exposes_legendre_machinery() -> None:
    grids = phase_space.make_grids(
        n_theta=7, n_zeta=7, n_xi=8, n_x=4, n_l=4, n_periods=5
    )
    assert_exact(grids.xi_coupling_upper, phase_space.legendre_coupling_upper(8))
    assert_exact(grids.xi_coupling_lower, phase_space.legendre_coupling_lower(8))
    assert_exact(grids.lorentz_eigenvalues, phase_space.lorentz_eigenvalues(8))
