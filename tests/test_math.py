"""Proofs for the speed-grid math kernels.

plan.md section 7.3 asks for direct proofs of the discretization primitives:
quadrature that integrates its declared moments exactly, differentiation
matrices that differentiate what they claim to, and stated limits where
exactness stops. Until now the speed grid was covered only indirectly, through
solves that would still pass if a rule were exact to one order less than
advertised.

Everything here is checked against a closed form, not against a recorded
output, so a regression cannot be absorbed by refreshing a fixture.

The convention worth knowing before reading the derivative tests: the matrices
from ``make_x_polynomial_diff_matrices`` act on *weighted* values. The vector
they take is ``u = w(x) p(x)`` for the grid's own weight
``w(x) = exp(-x^2) x^k``, and what comes back is ``u'``. Feeding them bare
polynomial values gives a large, meaningless answer -- which is exactly what a
reader assuming the other convention would produce, so both directions are
pinned below.
"""

from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from dkx.xgrid import (
    make_x_grid,
    make_x_polynomial_diff_matrices,
    x_weight_d1_over_weight_np,
    x_weight_d2_over_weight_np,
    x_weight_np,
)

#: Grid sizes and weight exponents to prove over. k=0 is the plain Gaussian
#: weight; k=2 is the one the speed grid actually uses for velocity moments.
GRIDS = [(4, 0.0), (5, 0.0), (6, 2.0), (8, 2.0)]


def analytic_moment(k: float, m: int) -> float:
    """int_0^inf exp(-x^2) x^(k+m) dx, in closed form."""
    return 0.5 * gamma((k + m + 1) / 2.0)


@pytest.mark.parametrize(("n", "k"), GRIDS)
def test_gauss_quadrature_is_exact_to_its_theoretical_degree(n: int, k: float) -> None:
    """An n-node Gauss rule integrates degree 2n-1 exactly, and this one does.

    This is the property the whole speed discretization rests on: the collision
    and moment operators assume their integrands are captured exactly up to that
    degree. A rule that was merely accurate, rather than exact, would degrade
    every velocity moment by an amount no solve-level test would isolate.
    """
    grid = make_x_grid(n=n, k=k)
    for degree in range(2 * n):
        got = float(np.sum(grid.gaussian_weights * grid.x**degree))
        want = analytic_moment(k, degree)
        assert got == pytest.approx(want, rel=1e-13), f"degree {degree}"


@pytest.mark.parametrize(("n", "k"), GRIDS)
def test_gauss_quadrature_stops_being_exact_one_degree_later(n: int, k: float) -> None:
    """Degree 2n is not exact, which is what makes the previous test a proof.

    Without this, a rule that happened to integrate everything -- a bug that
    silently used far more nodes than requested -- would pass the exactness test
    and go unnoticed.
    """
    grid = make_x_grid(n=n, k=k)
    got = float(np.sum(grid.gaussian_weights * grid.x ** (2 * n)))
    want = analytic_moment(k, 2 * n)
    assert got != pytest.approx(want, rel=1e-13)


@pytest.mark.parametrize(("n", "k"), GRIDS)
def test_gauss_nodes_and_weights_have_the_shape_a_gauss_rule_must(n: int, k: float) -> None:
    """Positive weights and strictly increasing interior nodes.

    A negative weight is the signature of a rule constructed from an ill-posed
    moment problem; it integrates polynomials correctly and amplifies noise on
    everything else.
    """
    grid = make_x_grid(n=n, k=k)
    assert grid.x.shape == (n,)
    assert np.all(np.diff(grid.x) > 0.0)
    assert np.all(grid.x > 0.0)
    assert np.all(grid.gaussian_weights > 0.0)


@pytest.mark.parametrize(("n", "k"), GRIDS)
def test_first_derivative_matrix_is_exact_on_weighted_polynomials(n: int, k: float) -> None:
    """D1 (w p) = (w p)' to roundoff, for every polynomial the grid resolves."""
    grid = make_x_grid(n=n, k=k)
    x = grid.x
    d1, _ = make_x_polynomial_diff_matrices(x, k=k)
    weight = x_weight_np(x, k)
    log_derivative = x_weight_d1_over_weight_np(x, k)

    for degree in range(n):
        p = x**degree
        dp = degree * x ** (degree - 1) if degree >= 1 else np.zeros_like(x)
        expected = weight * (dp + log_derivative * p)
        assert np.allclose(d1 @ (weight * p), expected, rtol=0, atol=1e-12), f"degree {degree}"


@pytest.mark.parametrize(("n", "k"), GRIDS)
def test_second_derivative_matrix_is_exact_on_weighted_polynomials(n: int, k: float) -> None:
    """D2 (w p) = (w p)'' to roundoff, by the same product rule."""
    grid = make_x_grid(n=n, k=k)
    x = grid.x
    _, d2 = make_x_polynomial_diff_matrices(x, k=k)
    weight = x_weight_np(x, k)
    first = x_weight_d1_over_weight_np(x, k)
    second = x_weight_d2_over_weight_np(x, k)

    for degree in range(n):
        p = x**degree
        dp = degree * x ** (degree - 1) if degree >= 1 else np.zeros_like(x)
        ddp = degree * (degree - 1) * x ** (degree - 2) if degree >= 2 else np.zeros_like(x)
        expected = weight * (ddp + 2.0 * first * dp + second * p)
        assert np.allclose(d2 @ (weight * p), expected, rtol=0, atol=1e-11), f"degree {degree}"


def test_the_derivative_matrices_take_weighted_values_not_bare_ones() -> None:
    """Pin the calling convention, because getting it wrong is silent.

    Passing bare polynomial values returns a large finite vector rather than
    raising, so a caller who assumes the unweighted convention gets plausible
    numbers that are wrong by orders of magnitude. This test states which
    convention holds, so the next reader does not have to infer it.
    """
    grid = make_x_grid(n=5, k=0.0)
    x = grid.x
    d1, _ = make_x_polynomial_diff_matrices(x, k=0.0)

    weighted = d1 @ (x_weight_np(x, 0.0) * np.ones_like(x))
    assert np.allclose(weighted, x_weight_np(x, 0.0) * x_weight_d1_over_weight_np(x, 0.0))

    bare = d1 @ np.ones_like(x)
    assert np.abs(bare).max() > 1.0, "bare values must not accidentally look correct"


@pytest.mark.parametrize("n", [6, 8])
def test_plain_dx_weights_integrate_what_the_weight_leaves_polynomial(n: int) -> None:
    """``dx_weights`` divides the weight out, so the reduced integrand must be a polynomial.

    With exponent ``k``, integrating ``exp(-x^2) x^m dx`` reduces to a Gauss sum
    over ``x^(m-k)``. That is exact when ``m >= k``, and not otherwise: the
    reduced integrand is then a negative power, which no polynomial rule
    integrates. Both halves are pinned, because the failing half is a real
    limit of the method rather than an accuracy shortfall to be tightened away.
    """
    k = 2.0
    grid = make_x_grid(n=n, k=k)
    weights = grid.dx_weights()
    integrand = np.exp(-grid.x**2)

    for m in range(int(k), int(k) + 2 * n - 1):
        got = float(np.sum(weights * integrand * grid.x**m))
        assert got == pytest.approx(analytic_moment(0.0, m), rel=1e-12), f"m={m}"

    for m in range(int(k)):
        got = float(np.sum(weights * integrand * grid.x**m))
        assert got != pytest.approx(analytic_moment(0.0, m), rel=1e-6), (
            f"m={m} reduces to a negative power; exactness here would mean the "
            "weights are not what dx_weights documents"
        )


def test_the_grid_is_cached_by_value_not_rebuilt_per_call() -> None:
    """Repeated construction returns the same object, so grids stay identity-stable.

    Callers compare grids to decide whether an operator can be reused; a fresh
    equal-but-distinct grid each call would silently defeat that.
    """
    first = make_x_grid(n=6, k=2.0)
    second = make_x_grid(n=6, k=2.0)
    assert first is second
    assert make_x_grid(n=6, k=0.0) is not first


@pytest.mark.parametrize("bad", [0, -1])
def test_a_grid_smaller_than_one_point_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        make_x_grid(n=bad)


def test_a_negative_weight_exponent_is_refused() -> None:
    with pytest.raises(ValueError, match="k must be >= 0"):
        make_x_grid(n=4, k=-1.0)
