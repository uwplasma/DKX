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


@pytest.mark.parametrize("flaw", ["nonfinite", "unreadable"])
def test_an_unusable_iterate_is_not_used_as_a_start(monkeypatch, flaw: str) -> None:
    """A stalled solve can report a residual it no longer has: GCROT hands back
    whatever it reached, which may hold NaNs, and a cap breached before the
    first cycle leaves no residual history to judge by. Neither is a start."""
    op = _operator()
    reached = np.full((op.total_size, 1), 1e-3)
    norms = np.asarray([1.0, 0.2])
    if flaw == "nonfinite":
        reached[0, 0] = np.nan
    else:
        norms = np.asarray([])
    stalled = SolveResult(
        x=reached, method="gmres", iterations=6000, residual_norms=norms,
        converged=False, recycle=None, timings={}, adjoint=None,
    )
    starts = _spy_on_starts(monkeypatch)
    _escalate(op, stalled=stalled)
    assert starts and starts[0] is None, starts[:1]


def test_the_old_misleading_advice_is_gone() -> None:
    """`raise max_dense_size explicitly` at 66004 DOFs asks for 32.5 GB."""
    op = _operator()
    with pytest.raises(RuntimeError) as excinfo:
        _escalate(op, tol=1e-300, max_dense_size=1)
    assert "raise max_dense_size explicitly if you really want this" not in str(excinfo.value)


_BIG = 1.0e3  # GB: the basis budget never binds


@pytest.mark.parametrize('restart,cap,size,budget,prebuilt,nonfinite,expected', [
    (30, 200, 4000, _BIG, False, False, [(30, 5), (100, 2), (1000, 5)]),
    (30, 200, 4000, _BIG, False, True, [(30, 5), (100, 2), (1000, 5)]),
    # The budget binds: 400 steps of a 4,000-unknown V and Z basis.
    (30, 200, 4000, 400 * 2 * 4000 * 8 / 2**30, False, False,
     [(30, 5), (100, 2), (400, 14)]),
    (30, 200, 4000, 20 * 2 * 4000 * 8 / 2**30, False, False, [(30, 200)]),
    (30, 30, 4000, _BIG, False, False, [(30, 5), (100, 2)]),
    (30, 5, 4000, _BIG, False, False, [(30, 5)]),
    (1, 6, 4000, _BIG, False, False, [(1, 6)]),
    # Never wider than the system.
    (30, 200, 400, _BIG, False, False, [(30, 5), (100, 2), (400, 14)]),
    (30, 200, 60, _BIG, False, False, [(30, 5), (60, 97)]),
    (30, 200, 4, _BIG, False, False, [(30, 200)]),
    (30, 200, 4000, _BIG, True, False, [(30, 200)]),
])
def test_auto_restart_budget_and_current_factors(
    monkeypatch, restart, cap, size, budget, prebuilt, nonfinite, expected,
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
                               converged=len(calls) == len(expected),
                               residual_norm=jnp.array(1.))
    monkeypatch.setattr(module, 'gcrot', gcrot)
    result = module._solve_tier2(
        op, jnp.ones((size, 1)), tol=1e-7, atol=0., x0=None, recycle=None,
        preconditioner='coarse', drop_l_coupling_in_precond=False,
        restart=restart, recycle_dim=8, max_restarts=cap, differentiable=False,
        check_adjoint=False, prebuilt_precond=pair if prebuilt else None,
        auto_restart_recovery=True, krylov_memory_budget_gb=budget,
    )
    assert [(c['m'], c['max_restarts']) for c in calls] == expected
    assert result.iterations == sum(m * n for m, n in expected) <= restart * cap
    assert len(builds) == (0 if prebuilt else 1)
    assert all(c['precond'] is pair[0] and c['rtol'] == 1e-7 for c in calls)
    if len(calls) >= 2:
        assert (calls[1]['x0'] is None) == nonfinite
        assert (calls[1]['recycle'] is None) == nonfinite
        if not nonfinite:
            np.testing.assert_array_equal(calls[1]['x0'], 0.5)


def test_tier2_profile_events_cover_build_and_outer_gcrot(monkeypatch, capsys):
    """Completion events follow existing build and returned GCROT leaves."""
    import importlib
    from types import SimpleNamespace

    import jax.numpy as jnp

    module = importlib.import_module('dkx.solve')
    monkeypatch.setenv('DKX_PROFILE', '1')
    op = SimpleNamespace(total_size=4)
    pair = (lambda x: x, lambda x: x)
    monkeypatch.setattr(module, '_pinned_matvecs', lambda _op: pair)
    monkeypatch.setattr(module, 'build_tier2_preconditioner', lambda *_a, **_k: pair)
    monkeypatch.setattr(module, 'gcrot', lambda _mv, b, **_k: SimpleNamespace(
        x=b, recycle=(b[:, None], b[:, None]), iterations=1,
        converged=jnp.array(True), residual_norm=jnp.array(0.),
    ))

    result = module._solve_tier2(
        op, jnp.ones((4, 1)), tol=1e-7, atol=0., x0=None, recycle=None,
        preconditioner='coarse', drop_l_coupling_in_precond=False,
        restart=4, recycle_dim=1, max_restarts=1, differentiable=False,
        check_adjoint=False,
    )

    assert result.converged
    labels = [line.split()[1] for line in capsys.readouterr().err.splitlines()]
    assert labels == [
        'preconditioner_build.start',
        'preconditioner_build.complete',
        'krylov_compile_and_execute.start',
        'krylov_compile_and_execute.complete',
    ]


@pytest.mark.parametrize(
    ('failure', 'start_label', 'complete_label'),
    [
        ('preconditioner', 'preconditioner_build.start', 'preconditioner_build.complete'),
        ('gcrot', 'krylov_compile_and_execute.start', 'krylov_compile_and_execute.complete'),
    ],
)
def test_tier2_profile_start_survives_phase_failure(
    monkeypatch, capsys, failure, start_label, complete_label,
):
    import importlib
    from types import SimpleNamespace

    import jax.numpy as jnp

    module = importlib.import_module('dkx.solve')
    monkeypatch.setenv('DKX_PROFILE', '1')
    monkeypatch.setattr(module, '_pinned_matvecs', lambda _op: (lambda x: x, lambda x: x))
    target = 'build_tier2_preconditioner' if failure == 'preconditioner' else 'gcrot'
    monkeypatch.setattr(
        module, target,
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('failed')),
    )

    with pytest.raises(RuntimeError, match='failed'):
        module._solve_tier2(
            SimpleNamespace(total_size=4), jnp.ones((4, 1)), tol=1e-7,
            atol=0., x0=None, recycle=None,
            preconditioner='coarse' if failure == 'preconditioner' else 'none',
            drop_l_coupling_in_precond=False, restart=4, recycle_dim=1,
            max_restarts=1, differentiable=False, check_adjoint=False,
        )

    lines = capsys.readouterr().err.splitlines()
    assert any(start_label in line for line in lines)
    assert not any(complete_label in line for line in lines)


def test_tier2_traced_execution_does_not_create_profiler(monkeypatch):
    import importlib
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    module = importlib.import_module('dkx.solve')
    pair = (lambda x: x, lambda x: x)
    monkeypatch.setenv('DKX_PROFILE', '1')
    monkeypatch.setattr(
        module, 'maybe_profiler',
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError('profiler created')),
    )
    monkeypatch.setattr(module, '_pinned_matvecs', lambda _op: pair)
    monkeypatch.setattr(module, 'gcrot', lambda _mv, b, **_k: SimpleNamespace(
        x=b, recycle=(b[:, None], b[:, None]), iterations=jnp.array(1),
        converged=jnp.array(True), residual_norm=jnp.array(0.),
    ))

    def traced(rhs):
        return module._solve_tier2(
            SimpleNamespace(total_size=4), rhs, tol=1e-7, atol=0., x0=None,
            recycle=None, preconditioner='coarse',
            drop_l_coupling_in_precond=False, restart=4, recycle_dim=1,
            max_restarts=1, differentiable=False, check_adjoint=False,
            prebuilt_precond=pair,
        ).x

    jax.make_jaxpr(traced)(jnp.ones((4, 1)))


@pytest.mark.parametrize('method,differentiable,restart,recovery', [
    ('auto', False, None, True), ('auto', True, None, False),
    ('gmres', False, None, False), ('auto', False, 30, False),
])
def test_restart_recovery_is_only_auto_host_policy(
    monkeypatch, method, differentiable, restart, recovery,
):
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module('dkx.solve')
    op = SimpleNamespace(total_size=4)
    monkeypatch.setattr(module, '_auto_route', lambda *a: 'gmres')
    monkeypatch.setattr(module, '_resolve_solve_device', lambda *a: None)
    def tier2(op, rhs, **kwargs):
        assert kwargs['auto_restart_recovery'] is recovery
        assert kwargs['restart'] == 30  # the policy's first cycles, or the explicit size
        return SolveResult(rhs, 'gcrot', 0, np.array([0.]), True, None, {})
    monkeypatch.setattr(module, '_solve_tier2', tier2)
    assert module.solve(
        op, np.ones(4), method=method, differentiable=differentiable, restart=restart,
    ).converged


@pytest.mark.parametrize('restart', [0, -3, 2.5, True])
def test_restart_must_be_none_or_a_positive_integer(restart):
    import importlib
    module = importlib.import_module('dkx.solve')
    with pytest.raises(ValueError, match='restart must be None or a positive integer'):
        module.solve(_operator(), np.ones(1), restart=restart)


def test_wide_restart_budget_resolution(monkeypatch):
    """Argument, then environment, then a quarter of available memory, capped."""
    import importlib
    module = importlib.import_module('dkx.solve')
    monkeypatch.delenv('DKX_KRYLOV_MEMORY_BUDGET_GB', raising=False)
    per_step = 2 * 100_000 * 8
    monkeypatch.setattr(module, '_available_memory_bytes', lambda: 4.0 * 600 * per_step)
    assert module._auto_wide_restart(100_000, 8, None) == 600
    monkeypatch.setattr(module, '_available_memory_bytes', lambda: 1e15)
    assert module._auto_wide_restart(100_000, 8, None) == 1000
    assert module._auto_wide_restart(500, 8, None) == 500
    monkeypatch.setattr(module, '_available_memory_bytes', lambda: None)
    assert module._auto_wide_restart(100_000, 8, None) == (256 * 1024**2) // per_step
    monkeypatch.setenv('DKX_KRYLOV_MEMORY_BUDGET_GB', str(300 * per_step / 2**30))
    assert module._auto_wide_restart(100_000, 8, None) == 300
    assert module._auto_wide_restart(100_000, 8, 700 * per_step / 2**30) == 700
    with pytest.raises(ValueError, match='krylov_memory_budget_gb'):
        module._auto_wide_restart(100_000, 8, -1.0)


@pytest.mark.parametrize('auto_restart,expected', [(True, (1000, 24)), (False, (30, 800))])
def test_larger_budget_rung_uses_the_wide_restart(monkeypatch, auto_restart, expected):
    """The stall ladder's larger-budget rung spends it at the wide restart."""
    import importlib
    module = importlib.import_module('dkx.solve')
    op = _operator()
    monkeypatch.setattr(module, '_auto_wide_restart', lambda *a: 1000)
    windows = []

    def tier2(op, rhs, **kwargs):
        windows.append((kwargs['restart'], kwargs['max_restarts']))
        return SolveResult(rhs, 'gcrot', 0, np.array([1.]), False, None, {})

    monkeypatch.setattr(module, '_solve_tier2', tier2)
    monkeypatch.setattr(module, '_solve_tier3', lambda *a, **k: 'direct')
    stalled = SolveResult(np.zeros((op.total_size, 1)), 'gcrot', 0, np.array([1.]),
                          False, None, {})
    assert module._escalate_after_tier2_stall(
        op, np.ones((op.total_size, 1)), stalled=stalled, tol=1e-10, atol=0.,
        x0=None, recycle=None, preconditioner='coarse',
        drop_l_coupling_in_precond=False, restart=30, recycle_dim=8,
        max_restarts=200, check_adjoint=False, adjoint_residual_factor=1.,
        max_dense_size=10**9, auto_restart=auto_restart,
    ) == 'direct'
    assert windows[-1] == expected
