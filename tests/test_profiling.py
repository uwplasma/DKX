from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import sfincs_jax.profiling as profiling
from sfincs_jax.profiling import (
    SimpleProfiler,
    Timer,
    _device_mem_mb,
    _peak_rss_mb,
    _resource_maxrss_to_mb,
    _rss_mb,
    make_emit,
    maybe_profiler,
)


def test_resource_maxrss_to_mb_handles_darwin_bytes() -> None:
    assert _resource_maxrss_to_mb(1024.0 * 1024.0, platform="darwin") == 1.0


def test_resource_maxrss_to_mb_handles_linux_kib() -> None:
    assert _resource_maxrss_to_mb(1024.0, platform="linux") == 1.0


def test_timer_uses_perf_counter_delta(monkeypatch) -> None:
    ticks = iter([2.0, 2.75])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    timer = Timer()

    assert timer.elapsed_s() == 0.75


def test_rss_helpers_cover_resource_fallback_and_missing_resource(monkeypatch) -> None:
    def raise_psutil() -> None:
        raise RuntimeError("psutil unavailable")

    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: SimpleNamespace(ru_maxrss=2048.0),
    )
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=raise_psutil))
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(profiling.sys, "platform", "linux")

    assert _rss_mb() == 2.0
    assert _peak_rss_mb() == 2.0

    fake_broken_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: (_ for _ in ()).throw(RuntimeError("resource unavailable")),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_broken_resource)
    assert _rss_mb() is None
    assert _peak_rss_mb() is None


def test_device_memory_uses_first_numeric_jax_stat(monkeypatch) -> None:
    fake_jax = SimpleNamespace(
        devices=lambda: [
            SimpleNamespace(
                memory_stats=lambda: {
                    "bytes_in_use": "not-a-number",
                    "bytes_active": 5_000_000,
                }
            )
        ]
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    assert _device_mem_mb() == 5.0

    fake_empty_jax = SimpleNamespace(devices=lambda: [SimpleNamespace(memory_stats=lambda: {})])
    monkeypatch.setitem(sys.modules, "jax", fake_empty_jax)
    assert _device_mem_mb() is None

    fake_broken_jax = SimpleNamespace(devices=lambda: (_ for _ in ()).throw(RuntimeError("no device")))
    monkeypatch.setitem(sys.modules, "jax", fake_broken_jax)
    assert _device_mem_mb() is None


def test_make_emit_respects_verbosity_quiet_and_prefix() -> None:
    stream = io.StringIO()
    emit = make_emit(verbose=1, stream=stream, prefix="[sfincs] ")

    emit(0, "always")
    emit(1, "visible")
    emit(2, "hidden")

    assert stream.getvalue().splitlines() == [
        "[sfincs] always",
        "[sfincs] visible",
    ]

    quiet_stream = io.StringIO()
    quiet_emit = make_emit(verbose=10, quiet=True, stream=quiet_stream)
    quiet_emit(0, "suppressed")
    assert quiet_stream.getvalue() == ""


def test_maybe_profiler_is_environment_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PROFILE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PROFILE_DEVICE_MEM", raising=False)
    assert maybe_profiler() is None

    monkeypatch.setenv("SFINCS_JAX_PROFILE", "timings")
    profiler = maybe_profiler()
    assert isinstance(profiler, SimpleProfiler)
    assert profiler.sample_device_mem is False

    monkeypatch.setenv("SFINCS_JAX_PROFILE", "full")
    profiler = maybe_profiler()
    assert isinstance(profiler, SimpleProfiler)
    assert profiler.sample_device_mem is True

    monkeypatch.setenv("SFINCS_JAX_PROFILE_DEVICE_MEM", "0")
    profiler = maybe_profiler()
    assert isinstance(profiler, SimpleProfiler)
    assert profiler.sample_device_mem is False


def test_simple_profiler_mark_records_phase_memory_and_emits(monkeypatch) -> None:
    stream = io.StringIO()
    emit = make_emit(verbose=0, stream=stream)

    monkeypatch.setattr(profiling.time, "perf_counter", lambda: 11.25)
    monkeypatch.setattr(profiling, "_rss_mb", lambda: 130.0)
    monkeypatch.setattr(profiling, "_peak_rss_mb", lambda: 170.0)
    monkeypatch.setattr(profiling, "_device_mem_mb", lambda: 42.0)

    profiler = SimpleProfiler(
        emit=emit,
        sample_device_mem=True,
        t0=10.0,
        last=10.5,
        rss0_mb=100.0,
        peak_rss0_mb=120.0,
    )

    profiler.mark("operator_build")

    assert len(profiler.entries) == 1
    entry = profiler.entries[0]
    assert entry["label"] == "operator_build"
    assert entry["dt_s"] == 0.75
    assert entry["total_s"] == 1.25
    assert entry["rss_mb"] == 130.0
    assert entry["drss_mb"] == 30.0
    assert entry["peak_rss_mb"] == 170.0
    assert entry["dpeak_rss_mb"] == 50.0
    assert entry["device_mb"] == 42.0
    assert profiler.last == 11.25

    line = stream.getvalue()
    assert "profiling: operator_build" in line
    assert "dt_s=0.750" in line
    assert "device_mb=42.0" in line
