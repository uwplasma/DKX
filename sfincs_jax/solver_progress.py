from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sfincs_jax.solver_progress_policy import (
    format_duration as format_duration,
    rhs1_large_progress_enabled as rhs1_large_progress_enabled,
    rhs1_progress_size_min as rhs1_progress_size_min,
    runtime_scale_hint as runtime_scale_hint,
)


EmitFn = Callable[[int, str], None]


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
    "RHS1ProgressNotes",
    "format_duration",
    "rhs1_large_progress_enabled",
    "rhs1_progress_size_min",
    "runtime_scale_hint",
    "transport_progress_message",
]
