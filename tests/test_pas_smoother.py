from __future__ import annotations

import jax.numpy as jnp
import math
import numpy as np

import pytest

from sfincs_jax.pas_smoother import (
    AdaptivePassSmootherResult,
    PasSmootherConfig,
    adaptive_pas_smoother,
    adaptive_pas_smoother_allowed,
    advance_pas_smoother,
    append_residual,
    decide_pas_smoother_action,
    run_adaptive_stationary_smoother,
    should_stop_adaptive_smoother,
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


def test_should_stop_adaptive_smoother_handles_empty_nonfinite_target_and_upward() -> None:
    assert should_stop_adaptive_smoother(
        (),
        target=1.0e-8,
        target_ratio=1.0,
        abs_floor=0.0,
        upward_ratio=1.05,
        patience=2,
        min_steps=1,
    ) == (True, "empty")

    assert should_stop_adaptive_smoother(
        (10.0, float("nan")),
        target=1.0e-8,
        target_ratio=1.0,
        abs_floor=0.0,
        upward_ratio=1.05,
        patience=2,
        min_steps=1,
    ) == (True, "nonfinite")

    assert should_stop_adaptive_smoother(
        (10.0, 1.0e-10),
        target=1.0e-8,
        target_ratio=1.0,
        abs_floor=0.0,
        upward_ratio=1.05,
        patience=2,
        min_steps=1,
    ) == (True, "target")

    assert should_stop_adaptive_smoother(
        (10.0, 9.0, 9.6, 9.8),
        target=1.0e-8,
        target_ratio=1.0,
        abs_floor=0.0,
        upward_ratio=1.05,
        patience=2,
        min_steps=1,
    ) == (True, "upward")

    assert should_stop_adaptive_smoother(
        (10.0, 9.0),
        target=1.0e-8,
        target_ratio=1.0,
        abs_floor=0.0,
        upward_ratio=1.05,
        patience=2,
        min_steps=3,
    ) == (False, "continue")


def test_summarize_residual_history_handles_nonfinite_and_zero_denominator() -> None:
    trend = summarize_residual_history((0.0, 0.0, float("inf")), window=2)
    assert trend.has_nonfinite
    assert trend.latest_ratio == math.inf
    assert trend.best_before_latest == pytest.approx(0.0)
    assert trend.best_before_latest_ratio == math.inf


def test_decide_pas_smoother_action_zero_residual_and_consecutive_increases() -> None:
    zero = decide_pas_smoother_action((3.0, 0.0))
    assert zero.accept
    assert zero.stop
    assert zero.reason == "zero-residual"

    worsening = decide_pas_smoother_action(
        (5.0, 5.2, 5.5),
        config=PasSmootherConfig(max_consecutive_increases=2, worsen_ratio=10.0, stagnation_ratio=10.0),
    )
    assert not worsening.accept
    assert worsening.stop
    assert worsening.reason == "consecutive-increases"


def test_run_adaptive_stationary_smoother_reaches_target_and_tracks_best_state() -> None:
    a = jnp.diag(jnp.asarray([4.0, 2.0], dtype=jnp.float64))
    rhs = jnp.asarray([2.0, 4.0], dtype=jnp.float64)

    def matvec(x):
        return a @ x

    def smoother(r):
        return jnp.asarray([r[0] / 4.0, r[1] / 2.0], dtype=jnp.float64)

    result = run_adaptive_stationary_smoother(
        matvec_fn=matvec,
        rhs_vec=rhs,
        x0_vec=jnp.zeros_like(rhs),
        smoother_fn=smoother,
        target=1.0e-12,
        max_steps=2,
        omega=1.0,
        upward_ratio=1.05,
        patience=1,
        min_steps=1,
        target_ratio=1.0,
        abs_floor=0.0,
    )

    assert result.stop_reason == "target"
    assert result.improved
    assert result.steps_completed == 1
    assert result.best_residual_norm <= 1.0e-12
    np.testing.assert_allclose(np.asarray(result.x_best), np.asarray([0.5, 2.0]), rtol=0.0, atol=1e-12)


def test_run_adaptive_stationary_smoother_stops_on_nonfinite_update() -> None:
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    result = run_adaptive_stationary_smoother(
        matvec_fn=lambda x: x,
        rhs_vec=rhs,
        x0_vec=jnp.zeros_like(rhs),
        smoother_fn=lambda r: jnp.asarray([jnp.inf, r[1]], dtype=jnp.float64),
        target=1.0e-12,
        max_steps=2,
        omega=1.0,
        upward_ratio=1.05,
        patience=1,
        min_steps=1,
        target_ratio=1.0,
        abs_floor=0.0,
    )

    assert result.stop_reason == "nonfinite_update"
    assert result.steps_completed == 0
