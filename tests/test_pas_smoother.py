from __future__ import annotations

import math

import pytest

from sfincs_jax.pas_smoother import (
    PasSmootherConfig,
    advance_pas_smoother,
    append_residual,
    decide_pas_smoother_action,
    summarize_residual_history,
)


def test_append_residual_returns_new_immutable_history() -> None:
    history = (10.0, 5.0)
    updated = append_residual(history, 2.5)
    assert updated == (10.0, 5.0, 2.5)
    assert history == (10.0, 5.0)


def test_summarize_residual_history_reports_window_ratios() -> None:
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


def test_decide_pas_smoother_action_accepts_monotone_improvement() -> None:
    decision = decide_pas_smoother_action((10.0, 4.0, 1.0))
    assert decision.accept
    assert not decision.stop
    assert decision.reason == "improved"
    assert decision.trend.latest_ratio == pytest.approx(0.25)


def test_decide_pas_smoother_action_stops_on_worsening() -> None:
    decision = decide_pas_smoother_action((10.0, 9.0, 9.9))
    assert not decision.accept
    assert decision.stop
    assert decision.reason == "single-step-worsened"


def test_decide_pas_smoother_action_stops_on_stagnation_and_nonfinite() -> None:
    stagnating = decide_pas_smoother_action(
        (10.0, 9.98, 9.97, 9.96),
        config=PasSmootherConfig(window=3, stagnation_ratio=0.995),
    )
    assert stagnating.accept
    assert stagnating.stop
    assert stagnating.reason == "window-stagnation"

    nonfinite = decide_pas_smoother_action((10.0, float("nan")))
    assert not nonfinite.accept
    assert nonfinite.stop
    assert nonfinite.reason == "nonfinite-residual"


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
