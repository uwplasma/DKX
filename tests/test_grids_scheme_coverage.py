from __future__ import annotations

import numpy as np
import pytest

from sfincs_jax.grids import uniform_diff_matrices


@pytest.mark.parametrize(
    ("scheme", "row", "entries"),
    [
        (30, 3, {3: 1.0, 2: -1.0}),
        (40, 2, {2: -1.0, 3: 1.0}),
        (50, 4, {4: 1.5, 3: -2.0, 2: 0.5}),
        (60, 1, {1: -1.5, 2: 2.0, 3: -0.5}),
        (80, 0, {1: 1.0 / 3.0, 0: 0.5, 5: -1.0, 4: 1.0 / 6.0}),
        (81, 0, {1: 1.0 / 3.0, 0: 0.5, 5: -1.0, 4: 1.0 / 6.0}),
        (90, 0, {5: -1.0 / 3.0, 0: -0.5, 1: 1.0, 2: -1.0 / 6.0}),
        (91, 0, {5: -1.0 / 3.0, 0: -0.5, 1: 1.0, 2: -1.0 / 6.0}),
        (100, 0, {1: 0.25, 0: 5.0 / 6.0, 5: -1.5, 4: 0.5, 3: -1.0 / 12.0}),
        (101, 0, {1: 0.25, 0: 5.0 / 6.0, 5: -1.5, 4: 0.5, 3: -1.0 / 12.0}),
        (110, 0, {5: -0.25, 0: -5.0 / 6.0, 1: 1.5, 2: -0.5, 3: 1.0 / 12.0}),
        (111, 0, {5: -0.25, 0: -5.0 / 6.0, 1: 1.5, 2: -0.5, 3: 1.0 / 12.0}),
        (120, 0, {2: -1.0 / 20.0, 1: 0.5, 0: 1.0 / 3.0, 5: -1.0, 4: 0.25, 3: -1.0 / 30.0}),
        (121, 0, {2: -1.0 / 20.0, 1: 0.5, 0: 1.0 / 3.0, 5: -1.0, 4: 0.25, 3: -1.0 / 30.0}),
        (130, 0, {4: 1.0 / 20.0, 5: -0.5, 0: -1.0 / 3.0, 1: 1.0, 2: -0.25, 3: 1.0 / 30.0}),
        (131, 0, {4: 1.0 / 20.0, 5: -0.5, 0: -1.0 / 3.0, 1: 1.0, 2: -0.25, 3: 1.0 / 30.0}),
    ],
)
def test_uniform_diff_matrices_stencil_schemes_have_expected_rows(
    scheme: int,
    row: int,
    entries: dict[int, float],
) -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=6, x_min=0.0, x_max=6.0, scheme=scheme)
    row_vals = np.asarray(ddx)[row]
    for col, coeff in entries.items():
        assert row_vals[col] == pytest.approx(coeff)


@pytest.mark.parametrize(
    ("scheme", "row", "entries"),
    [
        (82, 2, {3: 1.0 / 3.0, 2: 0.5, 1: -1.0, 0: 1.0 / 6.0}),
        (92, 2, {1: -1.0 / 3.0, 2: -0.5, 3: 1.0, 4: -1.0 / 6.0}),
        (102, 3, {4: 0.25, 3: 5.0 / 6.0, 2: -1.5, 1: 0.5, 0: -1.0 / 12.0}),
        (112, 1, {0: -0.25, 1: -5.0 / 6.0, 2: 1.5, 3: -0.5, 4: 1.0 / 12.0}),
    ],
)
def test_uniform_diff_matrices_aperiodic_one_sided_rows(
    scheme: int,
    row: int,
    entries: dict[int, float],
) -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=6, x_min=0.0, x_max=5.0, scheme=scheme)
    row_vals = np.asarray(ddx)[row]
    for col, coeff in entries.items():
        assert row_vals[col] == pytest.approx(coeff)


@pytest.mark.parametrize(
    ("scheme", "expected_x"),
    [
        (80, np.arange(6.0)),
        (81, np.arange(1.0, 7.0)),
        (82, np.linspace(0.0, 5.0, 6)),
    ],
)
def test_uniform_diff_matrices_grid_placement_matches_v3_radial_conventions(
    scheme: int,
    expected_x: np.ndarray,
) -> None:
    x, weights, _, _ = uniform_diff_matrices(n=6, x_min=0.0, x_max=6.0 if scheme != 82 else 5.0, scheme=scheme)
    np.testing.assert_allclose(np.asarray(x), expected_x)
    if scheme == 82:
        np.testing.assert_allclose(np.asarray(weights), np.asarray([0.5, 1.0, 1.0, 1.0, 1.0, 0.5]))
    else:
        np.testing.assert_allclose(np.asarray(weights), np.ones(6))


def test_uniform_diff_matrices_odd_periodic_spectral_exactness() -> None:
    x, _, ddx, d2dx2 = uniform_diff_matrices(n=15, x_min=0.0, x_max=2.0 * np.pi, scheme=20)
    x = np.asarray(x)
    ddx = np.asarray(ddx)
    d2dx2 = np.asarray(d2dx2)

    f = np.sin(2.0 * x) + 0.3 * np.cos(3.0 * x)
    df = 2.0 * np.cos(2.0 * x) - 0.9 * np.sin(3.0 * x)
    d2f = -4.0 * np.sin(2.0 * x) - 2.7 * np.cos(3.0 * x)

    np.testing.assert_allclose(ddx @ f, df, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(d2dx2 @ f, d2f, rtol=0.0, atol=1e-11)


def test_uniform_diff_matrices_aperiodic_endpoint_coefficients_for_high_order_schemes() -> None:
    _, _, ddx12, d2dx212 = uniform_diff_matrices(n=6, x_min=0.0, x_max=6.0, scheme=12)
    ddx12 = np.asarray(ddx12)
    d2dx212 = np.asarray(d2dx212)
    dx = 6.0 / 5.0
    np.testing.assert_allclose(
        ddx12[0, :5],
        np.asarray([-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -0.25]) / dx,
    )
    np.testing.assert_allclose(
        d2dx212[0, :5],
        np.asarray([35.0 / 12.0, -26.0 / 3.0, 19.0 / 2.0, -14.0 / 3.0, 11.0 / 12.0]) / (dx * dx),
    )

    _, _, ddx13, d2dx213 = uniform_diff_matrices(n=6, x_min=0.0, x_max=6.0, scheme=13)
    ddx13 = np.asarray(ddx13)
    d2dx213 = np.asarray(d2dx213)
    np.testing.assert_allclose(ddx13[1, :4], np.asarray([-1.0 / 3.0, -0.5, 1.0, -1.0 / 6.0]) / dx)
    np.testing.assert_allclose(d2dx213[0, :3], np.asarray([1.0, -2.0, 1.0]) / (dx * dx))


@pytest.mark.parametrize(("scheme", "poly_degree"), [(2, 2), (12, 4), (13, 2)])
def test_uniform_diff_matrices_polynomial_exactness_matches_stencil_order(
    scheme: int,
    poly_degree: int,
) -> None:
    x, _, ddx, d2dx2 = uniform_diff_matrices(n=7, x_min=-1.0, x_max=2.0, scheme=scheme)
    x = np.asarray(x)
    ddx = np.asarray(ddx)
    d2dx2 = np.asarray(d2dx2)

    coeffs = np.arange(poly_degree + 1, dtype=np.float64) + 1.0
    f = sum(coeffs[k] * x**k for k in range(poly_degree + 1))
    df = sum(k * coeffs[k] * x ** (k - 1) for k in range(1, poly_degree + 1))
    d2f = sum(k * (k - 1) * coeffs[k] * x ** (k - 2) for k in range(2, poly_degree + 1))

    first_err = np.max(np.abs(ddx @ f - df))
    second_err = np.max(np.abs(d2dx2 @ f - d2f))

    assert first_err < 1.0e-12
    assert second_err < 1.0e-12


def test_uniform_diff_matrices_scheme3_matches_expected_low_order_exactness() -> None:
    x, _, ddx, d2dx2 = uniform_diff_matrices(n=7, x_min=-1.0, x_max=2.0, scheme=3)
    x = np.asarray(x)
    ddx = np.asarray(ddx)
    d2dx2 = np.asarray(d2dx2)

    linear = 2.0 + 3.0 * x
    quadratic = 1.0 - 2.0 * x + 4.0 * x**2

    np.testing.assert_allclose(ddx @ linear, np.full_like(x, 3.0), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(d2dx2 @ quadratic, np.full_like(x, 8.0), atol=1.0e-12, rtol=0.0)


def test_uniform_diff_matrices_minimum_n_guards_for_one_sided_five_point_schemes() -> None:
    for scheme in (82, 92, 100, 101, 102, 110, 111, 112, 120, 121, 130, 131):
        with pytest.raises(ValueError, match="scheme|4 point stencil"):
            uniform_diff_matrices(n=4, x_min=0.0, x_max=1.0, scheme=scheme)


def test_uniform_diff_matrices_notimplemented_branches_raise() -> None:
    with pytest.raises(NotImplementedError, match="scheme 122"):
        uniform_diff_matrices(n=6, x_min=0.0, x_max=1.0, scheme=122)
    with pytest.raises(NotImplementedError, match="scheme 132"):
        uniform_diff_matrices(n=6, x_min=0.0, x_max=1.0, scheme=132)
