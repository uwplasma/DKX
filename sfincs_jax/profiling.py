from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable


def _env_flag(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / 1e6
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _resource_maxrss_to_mb(raw_maxrss: float, platform: str | None = None) -> float:
    """Convert ``resource.ru_maxrss`` to MB-like units.

    Linux reports KiB while macOS reports bytes. The profiler historically uses
    MB labels with psutil's decimal bytes/1e6 value; this helper keeps the same
    practical scale while preserving the platform-specific resource semantics.
    """

    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return float(raw_maxrss) / (1024.0 * 1024.0)
    return float(raw_maxrss) / 1024.0


def _peak_rss_mb() -> float | None:
    """Return the process high-water RSS when the OS exposes it."""

    try:
        import resource

        return _resource_maxrss_to_mb(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except Exception:
        return None


def _device_mem_mb() -> float | None:
    try:
        import jax  # noqa: PLC0415

        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    for key in ("bytes_in_use", "bytes_active", "bytes_limit", "peak_bytes_in_use"):
        if key in stats:
            try:
                return float(stats[key]) / 1e6
            except Exception:
                continue
    return None


def _profile_enabled() -> bool:
    flag = _env_flag("SFINCS_JAX_PROFILE")
    return flag in {"1", "true", "yes", "on", "timings", "full", "trace"}


def _profile_device_mem_enabled() -> bool:
    explicit = _env_flag("SFINCS_JAX_PROFILE_DEVICE_MEM")
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    flag = _env_flag("SFINCS_JAX_PROFILE")
    return flag in {"full", "device", "device_mem"}


@dataclass
class SimpleProfiler:
    emit: Callable[[int, str], None] | None = None
    sample_device_mem: bool = False
    t0: float = field(default_factory=time.perf_counter)
    last: float = field(default_factory=time.perf_counter)
    rss0_mb: float | None = field(default_factory=_rss_mb)
    peak_rss0_mb: float | None = field(default_factory=_peak_rss_mb)
    entries: list[dict[str, float | str | None]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        rss_mb = _rss_mb()
        peak_rss_mb = _peak_rss_mb()
        dev_mb = _device_mem_mb() if self.sample_device_mem else None
        entry = {
            "label": label,
            "dt_s": now - self.last,
            "total_s": now - self.t0,
            "rss_mb": rss_mb,
            "drss_mb": (rss_mb - self.rss0_mb) if (rss_mb is not None and self.rss0_mb is not None) else None,
            "peak_rss_mb": peak_rss_mb,
            "dpeak_rss_mb": (
                peak_rss_mb - self.peak_rss0_mb
                if (peak_rss_mb is not None and self.peak_rss0_mb is not None)
                else None
            ),
            "device_mb": dev_mb,
        }
        self.entries.append(entry)
        if self.emit is not None:
            rss_txt = f"{rss_mb:.1f}" if rss_mb is not None else "na"
            drss_txt = f"{entry['drss_mb']:.1f}" if entry["drss_mb"] is not None else "na"
            peak_txt = f"{peak_rss_mb:.1f}" if peak_rss_mb is not None else "na"
            dpeak_txt = f"{entry['dpeak_rss_mb']:.1f}" if entry["dpeak_rss_mb"] is not None else "na"
            dev_txt = f"{dev_mb:.1f}" if dev_mb is not None else "na"
            self.emit(
                0,
                f"profiling: {label} dt_s={entry['dt_s']:.3f} total_s={entry['total_s']:.3f} "
                f"rss_mb={rss_txt} drss_mb={drss_txt} "
                f"peak_rss_mb={peak_txt} dpeak_rss_mb={dpeak_txt} device_mb={dev_txt}",
            )
        self.last = now


def maybe_profiler(emit: Callable[[int, str], None] | None = None) -> SimpleProfiler | None:
    if _profile_enabled():
        return SimpleProfiler(emit=emit, sample_device_mem=_profile_device_mem_enabled())
    return None
