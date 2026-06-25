from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass, field
import os


EmitFn = Callable[[int, str], None]
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


@dataclass
class RHS1ProgressNotes:
    """One-shot progress-note emitter for large RHSMode=1 solves."""

    emit: EmitFn | None
    enabled: bool
    emitted: set[str] = field(default_factory=set)

    def preconditioner_build(self, kind: str | None) -> None:
        if self.emit is None or not self.enabled or "precond" in self.emitted:
            return
        self.emit(
            0,
            " solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner "
            f"({kind}); this stage can take a while for large systems.",
        )
        self.emitted.add("precond")

    def krylov_start(self) -> None:
        if self.emit is None or not self.enabled or "solve" in self.emitted:
            return
        self.emit(0, " solve_v3_full_system_linear_gmres: starting Krylov iterations.")
        self.emitted.add("solve")


def transport_progress_message(*, completed: int, total: int, avg_rhs_s: float, elapsed_s: float) -> str:
    """Build the standard transport whichRHS progress line."""
    completed_i = int(completed)
    total_i = int(total)
    remaining_rhs = max(0, total_i - completed_i)
    return (
        f"solve_v3_transport_matrix_linear_gmres: progress {completed_i}/{total_i} "
        f"avg_rhs={format_duration(avg_rhs_s)} elapsed={format_duration(elapsed_s)} "
        f"est_remaining={format_duration(float(avg_rhs_s) * remaining_rhs)}"
    )


__all__ = [
    "EmitFn",
    "PROGRESS_SIZE_MIN_ENV",
    "RHS1ProgressNotes",
    "format_duration",
    "rhs1_large_progress_enabled",
    "rhs1_progress_size_min",
    "runtime_scale_hint",
    "transport_progress_message",
]
