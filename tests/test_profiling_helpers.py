from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest

from sfincs_jax import profiling


def test_make_emit_respects_quiet_verbosity_prefix_and_flush() -> None:
    stream = io.StringIO()
    emit = profiling.make_emit(verbose=1, quiet=False, stream=stream, prefix="[sfincs] ")

    emit(2, "hidden")
    assert stream.getvalue() == ""

    emit(1, "visible")
    assert stream.getvalue() == "[sfincs] visible\n"

    quiet_stream = io.StringIO()
    quiet_emit = profiling.make_emit(verbose=99, quiet=True, stream=quiet_stream)
    quiet_emit(0, "suppressed")
    assert quiet_stream.getvalue() == ""


def test_timer_elapsed_uses_monotone_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([10.0, 12.5, 13.25])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    timer = profiling.Timer()

    assert timer.elapsed_s() == pytest.approx(2.5)
    assert timer.elapsed_s() == pytest.approx(3.25)


def test_profile_env_flags_enable_profiler_and_device_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiling, "_rss_mb", lambda: 1.0)
    monkeypatch.setattr(profiling, "_peak_rss_mb", lambda: 2.0)
    monkeypatch.delenv("SFINCS_JAX_PROFILE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PROFILE_DEVICE_MEM", raising=False)

    assert profiling.maybe_profiler() is None

    monkeypatch.setenv("SFINCS_JAX_PROFILE", "trace")
    profiler = profiling.maybe_profiler()
    assert profiler is not None
    assert profiler.sample_device_mem is False

    monkeypatch.setenv("SFINCS_JAX_PROFILE", "full")
    profiler = profiling.maybe_profiler()
    assert profiler is not None
    assert profiler.sample_device_mem is True

    monkeypatch.setenv("SFINCS_JAX_PROFILE_DEVICE_MEM", "0")
    profiler = profiling.maybe_profiler()
    assert profiler is not None
    assert profiler.sample_device_mem is False

    monkeypatch.setenv("SFINCS_JAX_PROFILE_DEVICE_MEM", "yes")
    profiler = profiling.maybe_profiler()
    assert profiler is not None
    assert profiler.sample_device_mem is True


def test_rss_falls_back_to_resource_units(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = SimpleNamespace(
        RUSAGE_SELF=object(),
        getrusage=lambda _: SimpleNamespace(ru_maxrss=2 * 1024 * 1024),
    )
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(profiling.sys, "platform", "darwin")

    assert profiling._rss_mb() == pytest.approx(2.0)

    monkeypatch.setattr(profiling.sys, "platform", "linux")

    assert profiling._rss_mb() == pytest.approx(2048.0)
    assert profiling._resource_maxrss_to_mb(2048.0, platform="linux") == pytest.approx(2.0)


def test_rss_and_peak_rss_return_none_when_os_sampling_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "resource", None)

    assert profiling._rss_mb() is None
    assert profiling._peak_rss_mb() is None


def test_device_mem_handles_missing_jax_bad_stats_and_empty_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jax", None)
    assert profiling._device_mem_mb() is None

    jax = SimpleNamespace(
        devices=lambda: [
            SimpleNamespace(memory_stats=lambda: {"bytes_in_use": "not-a-number", "bytes_active": 2_500_000})
        ]
    )
    monkeypatch.setitem(sys.modules, "jax", jax)

    assert profiling._device_mem_mb() == pytest.approx(2.5)

    jax.devices = lambda: [SimpleNamespace(memory_stats=lambda: {})]

    assert profiling._device_mem_mb() is None


def test_profiler_emit_formats_unavailable_memory_samples_as_na(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[int, str]] = []
    monkeypatch.setattr(profiling, "_rss_mb", lambda: None)
    monkeypatch.setattr(profiling, "_peak_rss_mb", lambda: None)
    monkeypatch.setattr(profiling, "_device_mem_mb", lambda: None)
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: 13.0)

    profiler = profiling.SimpleProfiler(
        emit=lambda level, message: emitted.append((level, message)),
        sample_device_mem=True,
        t0=10.0,
        last=12.0,
        rss0_mb=None,
        peak_rss0_mb=None,
    )

    profiler.mark("phase")

    assert profiler.entries == [
        {
            "label": "phase",
            "dt_s": 1.0,
            "total_s": 3.0,
            "rss_mb": None,
            "drss_mb": None,
            "peak_rss_mb": None,
            "dpeak_rss_mb": None,
            "device_mb": None,
        }
    ]
    assert emitted == [
        (
            0,
            "profiling: phase dt_s=1.000 total_s=3.000 "
            "rss_mb=na drss_mb=na peak_rss_mb=na dpeak_rss_mb=na device_mb=na",
        )
    ]


def test_profiler_emit_formats_available_memory_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[int, str]] = []
    monkeypatch.setattr(profiling, "_rss_mb", lambda: 15.0)
    monkeypatch.setattr(profiling, "_peak_rss_mb", lambda: 30.0)
    monkeypatch.setattr(profiling, "_device_mem_mb", lambda: 7.5)
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: 13.0)

    profiler = profiling.SimpleProfiler(
        emit=lambda level, message: emitted.append((level, message)),
        sample_device_mem=True,
        t0=10.0,
        last=12.0,
        rss0_mb=10.0,
        peak_rss0_mb=20.0,
    )

    profiler.mark("phase")

    assert profiler.entries[0]["drss_mb"] == pytest.approx(5.0)
    assert profiler.entries[0]["dpeak_rss_mb"] == pytest.approx(10.0)
    assert emitted == [
        (
            0,
            "profiling: phase dt_s=1.000 total_s=3.000 "
            "rss_mb=15.0 drss_mb=5.0 peak_rss_mb=30.0 dpeak_rss_mb=10.0 device_mb=7.5",
        )
    ]
