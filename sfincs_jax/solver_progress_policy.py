"""Pure formatting and threshold policy for solver progress messages."""

from __future__ import annotations

from collections.abc import Mapping
import os


PROGRESS_SIZE_MIN_ENV = "SFINCS_JAX_PROGRESS_SIZE_MIN"


def format_duration(seconds: float) -> str:
    """Format elapsed time for user-facing progress messages."""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def runtime_scale_hint(*, rhs_mode_hint: int, total_size_hint: int, n_rhs_hint: int | None = None) -> str:
    """Return a coarse runtime class for CLI progress hints.

    The thresholds are intentionally conservative and human-readable. They are
    not used for solver decisions, so changing this helper cannot affect parity.
    """
    size = int(total_size_hint)
    if int(rhs_mode_hint) in {2, 3}:
        n_rhs_use = max(1, int(n_rhs_hint or 1))
        work = size * n_rhs_use
        if work < 40_000:
            return "usually seconds to a few minutes"
        if work < 250_000:
            return "often minutes"
        return "often many minutes or longer"
    if size < 8_000:
        return "usually seconds"
    if size < 50_000:
        return "often tens of seconds to a few minutes"
    if size < 200_000:
        return "often minutes"
    return "often many minutes or longer"


def rhs1_progress_size_min(*, environ: Mapping[str, str] | None = None, default: int = 20_000) -> int:
    """Read the RHSMode=1 progress-note size threshold from the environment."""
    env = os.environ if environ is None else environ
    raw = str(env.get(PROGRESS_SIZE_MIN_ENV, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def rhs1_large_progress_enabled(
    *, rhs_mode: int, total_size: int, environ: Mapping[str, str] | None = None
) -> bool:
    """Whether to emit extra progress notes for a large RHSMode=1 solve."""
    threshold = max(1, rhs1_progress_size_min(environ=environ))
    return int(rhs_mode) == 1 and int(total_size) >= threshold


__all__ = [
    "PROGRESS_SIZE_MIN_ENV",
    "format_duration",
    "rhs1_large_progress_enabled",
    "rhs1_progress_size_min",
    "runtime_scale_hint",
]
