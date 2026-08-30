"""Proofs for the ambipolar bracket search and root classification.

plan.md section 7.3 asks for these to be proved against analytic current
functions with known roots, rather than only exercised through W7-X solves that
take minutes each. The bracket finder and the classifier are pure functions of
sampled ``(E_r, J_r)`` pairs, so every case below is exact and instant.

Two of these tests assert that the search *fails* to find something. They are
the load-bearing ones. Every admitted W7-X result in this repository is scoped
to explicit intervals and says "unsampled crossings not excluded"; that caveat
is only honest if the search genuinely cannot see a root that does not change
sign between two samples. Proving the limitation is what entitles the claim to
be stated the narrow way it is.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dkx.phase_space import uniform_periodic_diff_matrices
from dkx.workflows.ambipolar_native import _brackets, _classify_root

TWO_PI = 2.0 * math.pi

#: A uniform sampled field grid in kV/m, the shape a scan produces.
FIELDS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


def brackets_of(currents: list[float]) -> list[tuple[int, int]]:
    return _brackets(FIELDS, np.asarray(currents, dtype=float))


# --------------------------------------------------------------------------
# What the search must find
# --------------------------------------------------------------------------


def test_a_single_sign_change_is_one_bracket() -> None:
    assert brackets_of([-1, -1, -1, -1, 1, 1, 1]) == [(3, 4)]


def test_every_sign_change_is_reported_not_just_the_first() -> None:
    """A three-root S-curve must yield three brackets.

    The standard stellarator topology is ion / unstable / electron. Returning
    only the first would silently reduce a three-root surface to one and change
    which root the profile selects.
    """
    assert brackets_of([-1, 1, -1, -1, 1, 1, 1]) == [(0, 1), (1, 2), (3, 4)]


def test_a_root_landing_exactly_on_a_node_is_a_degenerate_bracket() -> None:
    """Zero at a sample is reported as ``(i, i)``, not skipped and not widened.

    ``left * right < 0`` is false when one endpoint is exactly zero, so without
    the explicit equality branch an exactly-resolved root would vanish.
    """
    assert brackets_of([-1, -1, 0, 1, 1, 1, 1]) == [(2, 2)]


@pytest.mark.parametrize(
    ("currents", "expected"),
    [
        ([0, 1, 1, 1, 1, 1, 1], [(0, 0)]),
        ([-1, -1, -1, -1, -1, -1, 0], [(6, 6)]),
    ],
)
def test_a_root_on_either_endpoint_is_still_found(
    currents: list[float], expected: list[tuple[int, int]]
) -> None:
    """The first and last samples are covered.

    The loop runs over pairs, so the final node is only reachable through the
    separate trailing check; dropping it would lose a root at the scan edge,
    which is where seeded intervals deliberately put their endpoints.
    """
    assert brackets_of(currents) == expected


def test_closely_spaced_roots_in_adjacent_intervals_stay_separate() -> None:
    """Two roots one interval apart must not merge into a single bracket."""
    assert brackets_of([-1, 1, -1, -1, -1, -1, -1]) == [(0, 1), (1, 2)]


# --------------------------------------------------------------------------
# What the search cannot find, and must not pretend to
# --------------------------------------------------------------------------


def test_no_sign_change_yields_no_bracket() -> None:
    assert brackets_of([1, 1, 1, 1, 1, 1, 1]) == []


def test_a_tangential_root_that_does_not_cross_zero_is_invisible() -> None:
    """A current that touches zero without changing sign is not found.

    This is a real root of ``J_r`` and sign sampling cannot see it. The search
    is correct to return nothing -- inventing a bracket here would be worse --
    but it means "no bracket" never means "no root". Every claim built on this
    search has to carry that caveat, which is why it is pinned rather than
    left as folklore.
    """
    assert brackets_of([1, 1, 0.5, 1e-12, 0.5, 1, 1]) == []


def test_an_even_number_of_crossings_between_samples_is_invisible() -> None:
    """Two roots inside one interval leave the endpoint signs equal.

    plan.md is explicit that a finite sign-sampled grid cannot exclude an even
    number of unresolved crossings. Both endpoints are negative here while the
    true current dips positive in between; the search reports nothing, exactly
    as a sign test must. This is the formal reason the admitted W7-X results
    are interval-scoped and never claim to have found every root.
    """
    assert brackets_of([-1, -1, -1, -1, -1, -1, -1]) == []


def test_refining_the_grid_is_what_exposes_a_hidden_pair() -> None:
    """The same physical current, sampled finely enough, does yield brackets.

    Pairs the previous test with its remedy so the limitation reads as a
    resolution property rather than a defect in the search.
    """
    coarse = np.array([-1.0, 1.0])
    fine = np.linspace(-1.0, 1.0, 9)
    # J_r(E) = 0.04 - E^2 has two roots at +-0.2, both inside the coarse interval.
    current = lambda field: 0.04 - field**2  # noqa: E731

    assert _brackets(coarse, current(coarse)) == []
    assert len(_brackets(fine, current(fine))) == 2


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_the_unstable_root_is_the_one_with_negative_slope() -> None:
    """``dE_r/dt ~ -J_r``, so a root is stable iff ``dJ_r/dE_r > 0``.

    The sign of the field does not decide stability; the slope does. Reversing
    this would relabel the middle branch of every S-curve as a physical root.
    """
    assert _classify_root(1.0, -1.0) == "unstable"
    assert _classify_root(-1.0, -1.0) == "unstable"


def test_a_stable_root_is_named_by_the_sign_of_the_field() -> None:
    assert _classify_root(1.0, 1.0) == "electron"
    assert _classify_root(-1.0, 1.0) == "ion"


def test_zero_slope_is_treated_as_stable_not_unstable() -> None:
    """The boundary is ``slope < 0``, so a flat root classifies by field sign.

    Pinned because it is a boundary a refactor could flip to ``<=`` without any
    other test noticing, which would relabel marginal roots as unstable.
    """
    assert _classify_root(1.0, 0.0) == "electron"
    assert _classify_root(-1.0, 0.0) == "ion"


def test_a_root_exactly_at_zero_field_classifies_as_ion() -> None:
    """``E_r > 0`` is strict, so the origin falls to the ion side.

    Physically arbitrary at exactly zero, but it must be deterministic: branch
    identity across a radial profile depends on the label not flickering.
    """
    assert _classify_root(0.0, 1.0) == "ion"


def test_classification_is_a_pure_function_of_field_and_slope() -> None:
    """No hidden state: the same inputs give the same label every time.

    Branch continuation compares labels between surfaces, so a classifier with
    memory would produce spurious branch-change events.
    """
    labels = {_classify_root(0.5, 2.0) for _ in range(5)}
    assert labels == {"electron"}


# --------------------------------------------------------------------------
# Angular differentiation
#
# The existing angular tests assert that these matrices equal a reference
# construction, which pins the port but not the mathematics: a scheme that was
# wrong in both places would pass. These check what the matrices do to
# manufactured periodic functions whose derivatives are known in closed form.
# --------------------------------------------------------------------------

FOURIER_SCHEME = 20
#: (scheme, order the docstring declares) for the first-derivative stencils.
CENTERED_SCHEMES = [(0, 2), (10, 4)]
UPWIND_SCHEMES = [
    (80, 2), (90, 2), (100, 3), (110, 3), (120, 4), (130, 4),
    (200, 3), (210, 3), (220, 4), (230, 4),
]


def periodic_grid(n: int, scheme: int):
    grid, weights, first, second = uniform_periodic_diff_matrices(
        n=n, x_min=0.0, x_max=TWO_PI, scheme=scheme
    )
    return (np.asarray(grid), np.asarray(weights), np.asarray(first), np.asarray(second))


def test_the_periodic_grid_excludes_its_right_endpoint_and_weights_uniformly() -> None:
    """A grid that included both ends would double-count one point in every integral."""
    grid, weights, _, _ = periodic_grid(16, FOURIER_SCHEME)
    assert grid[0] == 0.0
    assert grid[-1] < TWO_PI
    assert np.allclose(np.diff(grid), TWO_PI / 16)
    assert np.allclose(weights, TWO_PI / 16)


def test_fourier_differentiation_is_exact_on_every_representable_mode() -> None:
    """Spectral collocation differentiates sin(mx) and cos(mx) to roundoff.

    This is the property that makes the spectral option worth having: not
    "accurate", exact. An implementation that was merely high-order would pass
    a tolerance test and fail this one.
    """
    n = 16
    grid, _, first, _ = periodic_grid(n, FOURIER_SCHEME)
    for mode in range(1, n // 2):
        assert np.allclose(first @ np.sin(mode * grid), mode * np.cos(mode * grid), atol=1e-12)
        assert np.allclose(first @ np.cos(mode * grid), -mode * np.sin(mode * grid), atol=1e-12)


@pytest.mark.parametrize("scheme", [FOURIER_SCHEME] + [s for s, _ in CENTERED_SCHEMES])
def test_every_periodic_first_derivative_annihilates_constants(scheme: int) -> None:
    """d/dx of a constant is zero, so every row must sum to zero.

    A non-zero row sum is a spurious source term that a solve would quietly
    absorb into the answer.
    """
    _, _, first, _ = periodic_grid(16, scheme)
    assert np.abs(first @ np.ones(16)).max() < 1e-12


def test_the_fourier_first_derivative_is_exactly_skew_symmetric() -> None:
    """d/dx is anti-self-adjoint under the periodic inner product.

    The spectral matrix satisfies this identically, not approximately, so the
    check is on exact equality rather than a tolerance.
    """
    _, _, first, _ = periodic_grid(16, FOURIER_SCHEME)
    assert np.array_equal(first, -first.T)


@pytest.mark.parametrize(("scheme", "order"), CENTERED_SCHEMES)
def test_centered_stencils_recover_their_declared_order(scheme: int, order: int) -> None:
    """Manufactured solution: f = exp(sin x), f' = cos(x) exp(sin x).

    Halving the spacing must reduce the error by 2**order. The observed rate is
    required to reach within 0.3 of the claim, which separates a correct
    stencil from one that has silently lost an order.
    """
    errors = []
    for n in (32, 64, 128):
        grid, _, first, _ = periodic_grid(n, scheme)
        exact = np.cos(grid) * np.exp(np.sin(grid))
        errors.append(np.abs(first @ np.exp(np.sin(grid)) - exact).max())
    observed = math.log2(errors[-2] / errors[-1])
    assert observed > order - 0.3, f"scheme {scheme}: observed {observed:.2f} < claimed {order}"


@pytest.mark.parametrize(("scheme", "order"), UPWIND_SCHEMES)
def test_upwind_stencils_reach_at_least_their_declared_order(scheme: int, order: int) -> None:
    """The declared order is a floor, not an equality.

    On this manufactured function the SFINCS-ported stencils (80 through 130)
    converge one order faster than their docstring claims, while the dkx-only
    widened ones (200 through 230) land on theirs. A single test function can
    superconverge, so asserting equality here would pin an accident; asserting
    the floor catches the failure that matters, a stencil that has degraded
    below what it promises.
    """
    errors = []
    for n in (32, 64, 128):
        grid, _, first, _ = periodic_grid(n, scheme)
        exact = np.cos(grid) * np.exp(np.sin(grid))
        errors.append(np.abs(first @ np.exp(np.sin(grid)) - exact).max())
    observed = math.log2(errors[-2] / errors[-1])
    assert observed > order - 0.3, f"scheme {scheme}: observed {observed:.2f} < claimed {order}"
