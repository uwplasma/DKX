"""A stalled Krylov solve must escalate, not crash on a fallback that cannot run.

Reported from a scanType=5 run (Er scan inside a radius scan) on a 66004-DOF
deck.  Recycled Krylov GCROT hit its iteration cap at the largest |Er|, DKX announced a
fall back to the sparse direct host solve, and that route refused immediately
because 66004 > max_dense_size=8192.  The RuntimeError propagated out through
the scan driver and killed every remaining Er point at that radius: one radius
folder finished with zero outputs, another with three of a hundred.

Escalation must preserve the requested equation and report failed attempts
without claiming that nonconvergence establishes a particular physical cause.
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
    assert "max_dense_size=1" in message
    assert "does not establish that the physical model has no solution" in message
    assert "fixed physics" in message
    assert "small linear residual does not establish resolution convergence" in message
    assert "Keep Er and the requested tolerance fixed" in message


def _spy_on_rungs(monkeypatch) -> list[str]:
    """The preconditioner each rung asks for, in order."""
    from dkx import solve as solve_module

    kinds: list[str] = []
    real = solve_module._solve_tier2

    def spy(op, rhs2d, **kwargs):
        kinds.append(str(kwargs.get("preconditioner")))
        return real(op, rhs2d, **kwargs)

    monkeypatch.setattr(solve_module, "_solve_tier2", spy)
    return kinds


def test_the_ladder_restores_the_dropped_speed_coupling_first(monkeypatch) -> None:
    """``sparse`` and ``multigrid`` invert the same simplified operator that
    stalled, so neither can answer a stall caused by the Fokker-Planck speed
    coupling that operator drops. Retaining its upper triangle is the rung that
    changes the operator, and it reuses the factors already built, so it goes
    first on a deck that has a dense collision operator to retain."""
    op = dkx.run(**{**CASE, "collisionOperator": 0}).operator
    assert op.fp is not None or op.sugama is not None
    kinds = _spy_on_rungs(monkeypatch)
    _escalate(op)
    assert kinds[0] == "coarse_triangle", kinds


def test_pitch_angle_scattering_skips_the_triangle_rung(monkeypatch) -> None:
    """Pitch-angle scattering is already speed-diagonal, so retaining the
    triangle would re-run the stalled solve under another name."""
    op = _operator()
    assert op.fp is None and op.sugama is None
    kinds = _spy_on_rungs(monkeypatch)
    _escalate(op)
    assert "coarse_triangle" not in kinds, kinds


def _spy_on_starts(monkeypatch) -> list:
    """The initial guess each rung is handed."""
    from dkx import solve as solve_module

    starts: list = []
    real = solve_module._solve_tier2

    def spy(op, rhs2d, **kwargs):
        starts.append(kwargs.get("x0"))
        return real(op, rhs2d, **kwargs)

    monkeypatch.setattr(solve_module, "_solve_tier2", spy)
    return starts


def test_escalation_continues_from_the_stalled_iterate(monkeypatch) -> None:
    """Every rung solves the same equation, so the iterate the stalled solve
    reached is the work already paid for. Starting the next rung from the
    original guess throws it away."""
    op = _operator()
    reached = np.full((op.total_size, 1), 1e-3)
    stalled = SolveResult(
        x=reached, method="gmres", iterations=6000,
        residual_norms=np.asarray([1.0, 0.5, 0.2]), converged=False,
        recycle=None, timings={}, adjoint=None,
    )
    starts = _spy_on_starts(monkeypatch)
    _escalate(op, stalled=stalled)
    assert starts, "the ladder ran no rung"
    np.testing.assert_allclose(np.asarray(starts[0]), reached)


def test_a_diverged_iterate_is_not_used_as_a_start(monkeypatch) -> None:
    """An iterate worse than the right-hand side it started from would begin
    the next rung further away than the original guess."""
    op = _operator()
    diverged = SolveResult(
        x=np.full((op.total_size, 1), 1e6), method="gmres", iterations=6000,
        residual_norms=np.asarray([1.0, 5.0]), converged=False,
        recycle=None, timings={}, adjoint=None,
    )
    starts = _spy_on_starts(monkeypatch)
    _escalate(op, stalled=diverged)
    assert starts and starts[0] is None, starts[:1]


def test_the_old_misleading_advice_is_gone() -> None:
    """`raise max_dense_size explicitly` at 66004 DOFs asks for 32.5 GB."""
    op = _operator()
    with pytest.raises(RuntimeError) as excinfo:
        _escalate(op, tol=1e-300, max_dense_size=1)
    assert "raise max_dense_size explicitly if you really want this" not in str(excinfo.value)


@pytest.mark.parametrize('restart,cap,size,prebuilt,nonfinite,expected', [
    (30, 30, 4, False, False, [(30, 5), (100, 7)]),
    (30, 30, 4, False, True, [(30, 5), (100, 7)]),
    (30, 5, 4, False, False, [(30, 5)]),
    (1, 6, 4, False, False, [(1, 6)]),
    (100, 30, 4, False, False, [(100, 30)]),
    (30, 30, 167773, False, False, [(30, 30)]),
    (30, 30, 4, True, False, [(30, 30)]),
])
def test_auto_restart_budget_and_current_factors(
    monkeypatch, restart, cap, size, prebuilt, nonfinite, expected,
):
    """Probe/retry costs, factor identity, and unsafe-state cold fallback."""
    import importlib
    from types import SimpleNamespace
    import jax.numpy as jnp
    module = importlib.import_module('dkx.solve')
    op = SimpleNamespace(total_size=size)
    pair = (lambda x: x, lambda x: x)
    builds, calls = [], []
    monkeypatch.setattr(module, '_pinned_matvecs', lambda op: pair)
    monkeypatch.setattr(module, 'build_tier2_preconditioner',
                        lambda *a, **k: builds.append(True) or pair)
    def gcrot(mv, b, **kwargs):
        calls.append(kwargs)
        x = jnp.full_like(b, jnp.nan if nonfinite and len(calls) == 1 else 0.5)
        return SimpleNamespace(x=x, recycle=(x[:, None], x[:, None]),
                               iterations=kwargs['m'] * kwargs['max_restarts'],
                               converged=len(calls) > 1, residual_norm=jnp.array(1.))
    monkeypatch.setattr(module, 'gcrot', gcrot)
    result = module._solve_tier2(
        op, jnp.ones((size, 1)), tol=1e-7, atol=0., x0=None, recycle=None,
        preconditioner='coarse', drop_l_coupling_in_precond=False,
        restart=restart, recycle_dim=8, max_restarts=cap, differentiable=False,
        check_adjoint=False, prebuilt_precond=pair if prebuilt else None,
        auto_restart_recovery=True,
    )
    assert [(c['m'], c['max_restarts']) for c in calls] == expected
    assert result.iterations == sum(m * n for m, n in expected) <= restart * cap
    assert len(builds) == (0 if prebuilt else 1)
    assert all(c['precond'] is pair[0] and c['rtol'] == 1e-7 for c in calls)
    if len(calls) == 2:
        assert (calls[1]['x0'] is None) == nonfinite
        assert (calls[1]['recycle'] is None) == nonfinite
        if not nonfinite:
            np.testing.assert_array_equal(calls[1]['x0'], 0.5)


@pytest.mark.parametrize('method,differentiable,recovery', [
    ('auto', False, True), ('auto', True, False), ('gmres', False, False),
])
def test_restart_recovery_is_only_auto_host_policy(monkeypatch, method, differentiable, recovery):
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module('dkx.solve')
    op = SimpleNamespace(total_size=4)
    monkeypatch.setattr(module, '_auto_route', lambda *a: 'gmres')
    monkeypatch.setattr(module, '_resolve_solve_device', lambda *a: None)
    def tier2(op, rhs, **kwargs):
        assert kwargs['auto_restart_recovery'] is recovery
        return SolveResult(rhs, 'gcrot', 0, np.array([0.]), True, None, {})
    monkeypatch.setattr(module, '_solve_tier2', tier2)
    assert module.solve(op, np.ones(4), method=method, differentiable=differentiable).converged
