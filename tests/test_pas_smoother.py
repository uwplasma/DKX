from __future__ import annotations

import jax.numpy as jnp
import math

import pytest

from sfincs_jax.pas_smoother import (
    AdaptivePassSmootherResult,
    PasSmootherConfig,
    adaptive_pas_smoother,
    adaptive_pas_smoother_allowed,
    advance_pas_smoother,
    append_residual,
    decide_pas_smoother_action,
    summarize_residual_history,
)


def test_adaptive_pas_smoother_reaches_target_with_exact_preconditioner() -> None:
    a = jnp.diag(jnp.asarray([4.0, 2.0], dtype=jnp.float64))
    rhs = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return a @ x

    def preconditioner(r):
        return jnp.asarray([r[0] / 4.0, r[1] / 2.0], dtype=jnp.float64)

    result = adaptive_pas_smoother(
        matvec=matvec,
        rhs=rhs,
        preconditioner=preconditioner,
        x0=x0,
        target=1.0e-12,
        max_sweeps=2,
    )

    assert result.stop_reason == "target"
    assert result.accepted_sweeps == 1
    assert float(result.residual_norm) <= 1.0e-12
    assert result.history.shape[0] >= 2


def test_adaptive_pas_smoother_stops_when_worsening() -> None:
    a = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    rhs = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return a @ x

    def bad_preconditioner(r):
        return -r

    result = adaptive_pas_smoother(
        matvec=matvec,
        rhs=rhs,
        preconditioner=bad_preconditioner,
        x0=x0,
        target=1.0e-12,
        max_sweeps=3,
        worsen_factor=1.01,
    )

    assert result.stop_reason in {"worsened", "upward"}
    assert result.accepted_sweeps == 0
    assert float(result.residual_norm) == float(result.history[0])


def test_adaptive_pas_smoother_allowed_respects_guards() -> None:
    assert adaptive_pas_smoother_allowed(
        enabled=True,
        use_implicit=False,
        has_pas=True,
        include_phi1=False,
        residual_norm=1.0,
        target=1.0e-6,
        active_size=4000,
        min_size=2000,
    )
    assert not adaptive_pas_smoother_allowed(
        enabled=False,
        use_implicit=False,
        has_pas=True,
        include_phi1=False,
        residual_norm=1.0,
        target=1.0e-6,
        active_size=4000,
        min_size=2000,
    )
    assert not adaptive_pas_smoother_allowed(
        enabled=True,
        use_implicit=True,
        has_pas=True,
        include_phi1=False,
        residual_norm=1.0,
        target=1.0e-6,
        active_size=4000,
        min_size=2000,
    )
    assert not adaptive_pas_smoother_allowed(
        enabled=True,
        use_implicit=False,
        has_pas=False,
        include_phi1=False,
        residual_norm=1.0,
        target=1.0e-6,
        active_size=4000,
        min_size=2000,
    )
    assert not adaptive_pas_smoother_allowed(
        enabled=True,
        use_implicit=False,
        has_pas=True,
        include_phi1=False,
        residual_norm=1.0e-8,
        target=1.0e-6,
        active_size=4000,
        min_size=2000,
    )


def test_append_residual_returns_new_tuple() -> None:
    history = (10.0, 5.0)
    updated = append_residual(history, 2.5)
    assert updated == (10.0, 5.0, 2.5)
    assert history == (10.0, 5.0)


def test_summarize_residual_history_reports_ratios_and_slope() -> None:
    trend = summarize_residual_history((10.0, 5.0, 2.5, 1.25), window=3)
    assert trend.latest == pytest.approx(1.25)
    assert trend.previous == pytest.approx(2.5)
    assert trend.best_so_far == pytest.approx(1.25)
    assert trend.best_before_latest == pytest.approx(2.5)
    assert trend.latest_ratio == pytest.approx(0.5)
    assert trend.best_before_latest_ratio == pytest.approx(0.5)
    assert trend.window_reference == pytest.approx(10.0)
    assert trend.window_ratio == pytest.approx(0.125)
    assert trend.window_log_slope == pytest.approx(math.log(0.5))
    assert trend.consecutive_increases == 0
    assert not trend.has_nonfinite


def test_decide_pas_smoother_action_accepts_or_stops_as_expected() -> None:
    improved = decide_pas_smoother_action((10.0, 4.0, 1.0))
    assert improved.accept
    assert not improved.stop
    assert improved.reason == "improved"

    worsening = decide_pas_smoother_action((10.0, 9.0, 9.9))
    assert not worsening.accept
    assert worsening.stop
    assert worsening.reason == "single-step-worsened"

    stagnating = decide_pas_smoother_action(
        (10.0, 9.98, 9.97, 9.96),
        config=PasSmootherConfig(window=3, stagnation_ratio=0.995),
    )
    assert stagnating.accept
    assert stagnating.stop
    assert stagnating.reason == "window-stagnation"


def test_advance_pas_smoother_appends_then_decides() -> None:
    decision = advance_pas_smoother((10.0, 5.0), 2.0)
    assert decision.accept
    assert not decision.stop
    assert decision.trend.history == (10.0, 5.0, 2.0)


def test_pas_smoother_config_validation() -> None:
    with pytest.raises(ValueError):
        PasSmootherConfig(window=0)
    with pytest.raises(ValueError):
        PasSmootherConfig(accept_ratio=1.1, worsen_ratio=1.0)
