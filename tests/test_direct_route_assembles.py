"""The direct route builds its matrix instead of sampling it.

Sampling costs one operator application per column, which is what
``max_dense_size`` bounds and why this route refused every production deck: the
collaborator grid alone would need 66,004 applications. The assembly costs one
application per group of columns that share no row, so the bound does not apply
to it, and the factorization decides what the route can afford.
"""

from __future__ import annotations

import numpy as np
import pytest

import dkx
from dkx.solve import solve

CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0, iota=0.4542,
    GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=5, Nzeta=1, Nxi=6, NL=4, Nx=4,
    collisionOperator=0, Delta=4.5694e-3, alpha=1.0, nu_n=0.01,
)


@pytest.fixture(scope="module")
def operator():
    return dkx.run(**CASE).operator


def _spy(monkeypatch, module, name) -> list:
    calls: list = []
    real = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_it_assembles_rather_than_sampling_above_the_bound(operator, monkeypatch) -> None:
    """The bound counts operator applications per column; the assembly does not
    pay them, so it is not what the bound is about."""
    from dkx import solve as solve_module

    sampled = _spy(monkeypatch, solve_module, "materialize_csr")
    result = solve(
        operator, np.asarray(operator.rhs()), method="direct",
        tol=1e-10, max_dense_size=10,
    )
    assert result.converged
    assert sampled == [], "the route sampled the matrix column by column"
    assert float(np.asarray(result.residual_norms)[-1]) < 1e-10


def test_it_scales_the_matrix_before_factoring_it(operator, monkeypatch) -> None:
    """These rows carry streaming, collision and constraint terms at once, and a
    factorization picks its pivots from what it is handed."""
    from dkx import solve as solve_module

    scaled = _spy(monkeypatch, solve_module, "equilibrate")
    solve(
        operator, np.asarray(operator.rhs()), method="direct",
        tol=1e-10, max_dense_size=10,
    )
    assert len(scaled) == 1


def test_it_answers_what_the_iterative_route_answers(operator) -> None:
    """A different route to the same equation must reach the same solution."""
    rhs = np.asarray(operator.rhs())
    direct = solve(operator, rhs, method="direct", tol=1e-10, max_dense_size=10)
    iterative = solve(operator, rhs, method="gmres", tol=1e-12)
    a = np.asarray(direct.x).reshape(-1)
    b = np.asarray(iterative.x).reshape(-1)
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-8


def test_it_refuses_clearly_when_it_can_neither_assemble_nor_sample(
    operator, monkeypatch
) -> None:
    """An operator whose pattern is unavailable falls back to the bound, and the
    refusal has to name both halves: the applications sampling would cost, and
    why the assembly that avoids them did not run."""
    import dkx.assembly as assembly

    def unavailable(*args, **kwargs):
        raise NotImplementedError("no pattern for this operator")

    monkeypatch.setattr(assembly, "assemble_operator", unavailable)
    with pytest.raises(RuntimeError, match="no pattern for this operator") as excinfo:
        solve(
            operator, np.asarray(operator.rhs()), method="direct",
            tol=1e-10, max_dense_size=10,
        )
    message = str(excinfo.value)
    assert "max_dense_size=10" in message
    assert "column by column" in message


def test_a_small_deck_still_samples(operator, monkeypatch) -> None:
    """Below the bound nothing changes: sampling is exact and needs no pattern."""
    from dkx import solve as solve_module

    sampled = _spy(monkeypatch, solve_module, "materialize_csr")
    solve(
        operator, np.asarray(operator.rhs()), method="direct",
        tol=1e-10, max_dense_size=10**6,
    )
    assert len(sampled) == 1
