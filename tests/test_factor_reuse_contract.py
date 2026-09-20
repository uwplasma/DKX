"""One factorization, many solves: what the stored factors are allowed to do.

Production use is never one solve. A transport matrix is three right-hand sides
of one operator, a gradient is one transposed solve, and a line search is a
sequence of neighbours. The two direct routes return the factorization they
built (:attr:`dkx.solve.SolveResult.factors`) and take it back, so each of those
costs a triangular substitution rather than a factorization.

What is *not* claimed here is that stale factors are safe. They are accepted as
an approximate inverse and the route checks the original residual; a solve that
misses factorizes once and repeats. Both halves are tested, because a reuse
policy without a bounded recovery is how a solve at one field spends thousands
of iterations failing at another
(``docs/experiments/2026-09-07-recycling-and-preconditioner-reuse.md``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import dkx
from dkx.solve import DirectFactors, Tier1Solver, _pinned_matvecs, solve

_DECK = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.02, iota=0.4542,
    GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=5, Nzeta=5, Nxi=6, NL=4, Nx=4,
    Delta=4.5694e-3, alpha=1.0, nu_n=0.01, Nxi_for_x_option=0, RHSMode=2,
)

#: ``(label, method, extra solve kwargs)`` for the two routes that factor.
_ROUTES = [
    ("structured", "block_tridiagonal", {}),
    ("sparse", "direct", {"max_dense_size": 32}),
]


@pytest.fixture(scope="module")
def operators():
    # Pitch-angle scattering for the structured route (the only family its
    # Legendre bands support); Fokker-Planck for the sparse one.
    return {
        "block_tridiagonal": dkx.run(**dict(_DECK, collisionOperator=1)).operator,
        "direct": dkx.run(**dict(_DECK, collisionOperator=0)).operator,
    }


def _columns(op, k=3):
    return np.stack([np.asarray(op.rhs(i + 1)) for i in range(k)], axis=1)


def _solve(op, rhs, method, kw, **over):
    return solve(op, rhs, method=method, tol=1e-10, emit=None, **kw, **over)


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_a_factoring_route_returns_its_factorization(operators, label, method, kw):
    op = operators[method]
    result = _solve(op, _columns(op, 1), method, kw)
    assert result.converged
    assert isinstance(
        result.factors, Tier1Solver if method == "block_tridiagonal" else DirectFactors
    )


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_another_right_hand_side_reuses_them_exactly(operators, label, method, kw):
    """The transport matrix's columns need not arrive together."""
    op = operators[method]
    together = _solve(op, _columns(op), method, kw)
    assert together.converged
    for j in range(3):
        apart = _solve(op, _columns(op)[:, j], method, kw,
                       factors=together.factors)
        assert apart.converged
        # Reuse is not an approximation of the same operator: it is the same
        # elimination, so the two answers differ by round-off alone.
        reference = np.asarray(together.x)[:, j]
        assert np.max(np.abs(np.asarray(apart.x) - reference)) <= 1e-8 * np.max(
            np.abs(reference)
        )
        assert apart.timings["build"] == 0.0, "it factorized again"


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_the_adjoint_comes_from_the_same_factors(operators, label, method, kw):
    """A gradient's linear algebra is one transposed substitution."""
    op = operators[method]
    primal = _solve(op, _columns(op, 1), method, kw)
    cotangent = np.asarray(op.rhs(1))
    adjoint = _solve(op, cotangent, method, kw, factors=primal.factors,
                     transpose=True)
    assert adjoint.converged
    assert adjoint.timings["build"] == 0.0, "the adjoint factorized again"

    apply, apply_t = _pinned_matvecs(op)
    residual = np.asarray(apply_t(adjoint.x)) - cotangent
    assert np.linalg.norm(residual) <= 1e-9 * np.linalg.norm(cotangent)

    # <y, A x> == <A^T y, x>: that the transposed solve solves the transpose,
    # checked against the operator rather than against the factors.
    x = np.asarray(primal.x)[:, 0]
    y = np.asarray(adjoint.x)
    left = float(np.dot(y, np.asarray(apply(x))))
    right = float(np.dot(np.asarray(apply_t(y)), x))
    assert abs(left - right) <= 1e-9 * abs(left)


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_a_near_neighbour_is_solved_without_factoring_again(
    operators, label, method, kw
):
    op = operators[method]
    first = _solve(op, _columns(op, 1), method, kw)
    near = replace(op, t_hat=op.t_hat * (1.0 + 1e-6))
    result = _solve(near, np.asarray(near.rhs(1)), method, kw,
                    factors=first.factors)
    assert result.converged
    assert result.timings["build"] == 0.0
    # It is the neighbour's answer, not the first operator's: the defect is
    # measured with the operator that was passed in.
    cold = _solve(near, np.asarray(near.rhs(1)), method, kw)
    reference = np.asarray(cold.x)
    assert np.max(np.abs(np.asarray(result.x) - reference)) <= 1e-7 * np.max(
        np.abs(reference)
    )


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_a_distant_one_is_refused_and_recovered_in_one_factorization(
    operators, label, method, kw
):
    """The staleness test is the original residual and the recovery is bounded.

    Rejecting stale factors is necessary but not sufficient: the measured
    failure mode is a solve that waits, and this route cannot, because the
    recovery is a factorization rather than more iterations.
    """
    op = operators[method]
    first = _solve(op, _columns(op, 1), method, kw)
    far = replace(op, t_hat=op.t_hat * 4.0)
    result = _solve(far, np.asarray(far.rhs(1)), method, kw, factors=first.factors)
    assert result.converged, "the bounded recovery did not recover"
    assert result.timings["build"] > 0.0, "it accepted factors that do not solve this"
    assert result.factors is not first.factors, "it returned the stale factors"
    cold = _solve(far, np.asarray(far.rhs(1)), method, kw)
    reference = np.asarray(cold.x)
    assert np.max(np.abs(np.asarray(result.x) - reference)) <= 1e-7 * np.max(
        np.abs(reference)
    )


@pytest.mark.parametrize(("label", "method", "kw"), _ROUTES)
def test_factors_of_another_size_are_refused(operators, label, method, kw):
    op = operators[method]
    first = _solve(op, _columns(op, 1), method, kw)
    wider = dkx.run(**dict(_DECK, collisionOperator=1 if method ==
                           "block_tridiagonal" else 0, Nxi=8)).operator
    with pytest.raises(ValueError, match="size"):
        _solve(wider, np.asarray(wider.rhs(1)), method, kw, factors=first.factors)


def test_the_route_a_factorization_belongs_to_is_the_one_that_runs(operators):
    """``auto`` cannot choose a route that would then ignore the factors."""
    op = operators["direct"]
    first = solve(op, np.asarray(op.rhs(1)), method="direct", tol=1e-10,
                  emit=None, max_dense_size=32)
    again = solve(op, np.asarray(op.rhs(1)), method="auto", tol=1e-10,
                  emit=None, max_dense_size=32, factors=first.factors)
    assert again.method == "direct"
    assert again.timings["build"] == 0.0


def test_the_krylov_route_reuses_a_preconditioner_and_not_factors(operators):
    """Two different things with two different validity rules."""
    op = operators["direct"]
    with pytest.raises(ValueError, match="precond"):
        solve(op, np.asarray(op.rhs(1)), method="gmres", tol=1e-10, emit=None,
              factors=object())


def test_the_explicit_transpose_and_a_differentiable_solve_are_not_combined(
    operators,
):
    op = operators["block_tridiagonal"]
    with pytest.raises(ValueError, match="transpose"):
        solve(op, np.asarray(op.rhs(1)), method="block_tridiagonal", tol=1e-10,
              emit=None, transpose=True, differentiable=True)


def test_a_factorization_of_the_wrong_kind_is_named(operators):
    op = operators["direct"]
    first = solve(op, np.asarray(op.rhs(1)), method="direct", tol=1e-10,
                  emit=None, max_dense_size=32)
    pas = operators["block_tridiagonal"]
    with pytest.raises(TypeError, match="Tier1Solver"):
        solve(pas, np.asarray(pas.rhs(1)), method="block_tridiagonal", tol=1e-10,
              emit=None, factors=first.factors)
