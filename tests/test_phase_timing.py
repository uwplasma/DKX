from __future__ import annotations

from sfincs_jax.validation.artifacts import PhaseTimer, maxrss_mb


def test_maxrss_mb_normalizes_linux_and_macos_units() -> None:
    assert maxrss_mb(platform="linux", raw_value=2048) == 2.0
    assert maxrss_mb(platform="darwin", raw_value=2 * 1024 * 1024) == 2.0


def test_phase_timer_records_json_friendly_phases() -> None:
    timer = PhaseTimer()
    with timer.phase("work", case="demo"):
        pass

    summary = timer.summary()

    assert summary["phase_count"] == 1
    assert summary["elapsed_s"] >= 0.0
    phase = summary["phases"][0]
    assert phase["name"] == "work"
    assert phase["status"] == "ok"
    assert phase["elapsed_s"] >= 0.0
    assert phase["maxrss_mb"] >= 0.0
    assert phase["metadata"] == {"case": "demo"}
