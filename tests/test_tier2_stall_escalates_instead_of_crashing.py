"""A stalled Krylov solve must escalate, not crash on a fallback that cannot run.

Reported from a scanType=5 run (Er scan inside a radius scan) on a 66004-DOF
deck.  Recycled Krylov GCROT hit its iteration cap at the largest |Er|, DKX announced a
fall back to the sparse direct host solve, and that route refused immediately
because 66004 > max_dense_size=8192.  The RuntimeError propagated out through
the scan driver and killed every remaining Er point at that radius: one radius
folder finished with zero outputs, another with three of a hundred.

Sparse direct obtains its matrix by applying the operator to n identity columns, so
at the sizes where recycled Krylov actually stalls it can never run -- the advice in the
old message, to raise max_dense_size, would have asked for 32.5 GB.  A stalled
Krylov solve is a preconditioner problem, and DKX already ships the strong
preconditioner (``sparse``) that SFINCS's MUMPS LU is the analogue of; it was
simply never tried.
"""

import numpy as np
import pytest

import dkx
from dkx.solve import SolveResult, _escalate_after_tier2_stall

CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0, iota=0.4542,
    GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=9, Nzeta=1, Nxi=8, NL=4, Nx=4,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=0.01,
)


def _operator():
    return dkx.run(**CASE).operator


def _stalled(op) -> SolveResult:
    """What recycled Krylov hands back when it breaches its cap."""
    return SolveResult(
        x=np.zeros((op.total_size, 1)),
        method="gmres",
        iterations=6000,
        residual_norms=np.asarray([1.0, 0.5, 0.2]),
        converged=False,
        recycle=None,
        timings={},
        adjoint=None,
    )


def _rhs2d(op):
    """Recycled Krylov works in (n, nrhs); op.rhs() is (n,), as solve() reshapes it."""
    return np.asarray(op.rhs()).reshape(op.total_size, -1)


def _escalate(op, **overrides):
    kwargs = dict(
        stalled=_stalled(op), tol=1e-10, atol=0.0, x0=None, recycle=None,
        preconditioner="coarse", drop_l_coupling_in_precond=False,
        restart=20, recycle_dim=8, max_restarts=200,
        check_adjoint=False, adjoint_residual_factor=1e3,
        max_dense_size=8192,
    )
    kwargs.update(overrides)
    return _escalate_after_tier2_stall(op, _rhs2d(op), **kwargs)


def test_escalation_recovers_a_stalled_solve() -> None:
    """The ladder must actually produce a converged answer, not just try."""
    op = _operator()
    result = _escalate(op)
    assert result.converged, "escalation should recover a solvable system"
    assert float(np.asarray(result.residual_norms)[-1]) < 1e-8


def test_a_deck_too_large_for_tier3_reports_the_real_problem() -> None:
    """The failure the user hit: n > max_dense_size, so sparse direct cannot help.

    The message must not send them to max_dense_size, and must name what was
    tried and what actually helps.
    """
    op = _operator()
    with pytest.raises(RuntimeError) as excinfo:
        # Force every rung to fail by demanding a tolerance nothing can meet,
        # with sparse direct excluded exactly as it is at 66004 DOFs.
        _escalate(op, tol=1e-300, max_dense_size=1)
    message = str(excinfo.value)
    assert "did not converge" in message
    assert "Tried:" in message
    assert "raising max_dense_size would not help" in message
    assert "Er" in message, "the remedy that matters should name the Er dependence"


def test_the_old_misleading_advice_is_gone() -> None:
    """`raise max_dense_size explicitly` at 66004 DOFs asks for 32.5 GB."""
    op = _operator()
    with pytest.raises(RuntimeError) as excinfo:
        _escalate(op, tol=1e-300, max_dense_size=1)
    assert "raise max_dense_size explicitly if you really want this" not in str(excinfo.value)
