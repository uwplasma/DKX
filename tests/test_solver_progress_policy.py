from __future__ import annotations

import pytest

from sfincs_jax.solver_progress import (
    format_duration as legacy_format_duration,
    rhs1_large_progress_enabled as legacy_rhs1_large_progress_enabled,
    runtime_scale_hint as legacy_runtime_scale_hint,
)
from sfincs_jax.solver_progress import (
    PROGRESS_SIZE_MIN_ENV,
    format_duration,
    rhs1_large_progress_enabled,
    rhs1_progress_size_min,
    runtime_scale_hint,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-5.0, "0.0s"),
        (59.94, "59.9s"),
        (59.95, "60.0s"),
        (60.0, "1m00s"),
        (3599.0, "59m59s"),
        (3600.0, "1h00m"),
        (86399.0, "23h59m"),
        (86400.0, "1d00h"),
    ],
)
def test_format_duration_boundary_conventions(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_runtime_scale_hint_clamps_transport_rhs_count() -> None:
    assert (
        runtime_scale_hint(rhs_mode_hint=2, total_size_hint=39_999, n_rhs_hint=None)
        == "usually seconds to a few minutes"
    )
    assert (
        runtime_scale_hint(rhs_mode_hint=2, total_size_hint=40_000, n_rhs_hint=0)
        == "often minutes"
    )
    assert runtime_scale_hint(rhs_mode_hint=3, total_size_hint=125_000, n_rhs_hint=2) == (
        "often many minutes or longer"
    )


def test_rhs1_progress_threshold_parsing_is_pure_and_guarded() -> None:
    assert rhs1_progress_size_min(environ={PROGRESS_SIZE_MIN_ENV: " 17 "}) == 17
    assert rhs1_progress_size_min(environ={PROGRESS_SIZE_MIN_ENV: ""}, default=25) == 25
    assert rhs1_progress_size_min(environ={PROGRESS_SIZE_MIN_ENV: "bad"}, default=25) == 25
    assert rhs1_large_progress_enabled(rhs_mode=1, total_size=1, environ={PROGRESS_SIZE_MIN_ENV: "0"})
    assert rhs1_large_progress_enabled(rhs_mode=1, total_size=1, environ={PROGRESS_SIZE_MIN_ENV: "-4"})
    assert not rhs1_large_progress_enabled(rhs_mode=1, total_size=0, environ={PROGRESS_SIZE_MIN_ENV: "-4"})


def test_solver_progress_reexports_policy_helpers_for_backwards_compatibility() -> None:
    assert legacy_format_duration is format_duration
    assert legacy_runtime_scale_hint is runtime_scale_hint
    assert legacy_rhs1_large_progress_enabled is rhs1_large_progress_enabled
