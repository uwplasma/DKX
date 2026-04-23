from __future__ import annotations

from sfincs_jax.solver_progress import (
    RHS1ProgressNotes,
    format_duration,
    rhs1_large_progress_enabled,
    rhs1_progress_size_min,
    runtime_scale_hint,
    transport_progress_message,
)


def test_format_duration_matches_cli_progress_conventions() -> None:
    assert format_duration(-1.0) == "0.0s"
    assert format_duration(12.34) == "12.3s"
    assert format_duration(65.0) == "1m05s"
    assert format_duration(3600.0 + 125.0) == "1h02m"
    assert format_duration(26.0 * 3600.0) == "1d02h"


def test_runtime_scale_hint_keeps_rhs_mode_thresholds_stable() -> None:
    assert runtime_scale_hint(rhs_mode_hint=1, total_size_hint=1000) == "usually seconds"
    assert (
        runtime_scale_hint(rhs_mode_hint=1, total_size_hint=20_000)
        == "often tens of seconds to a few minutes"
    )
    assert runtime_scale_hint(rhs_mode_hint=1, total_size_hint=80_000) == "often minutes"
    assert runtime_scale_hint(rhs_mode_hint=1, total_size_hint=250_000) == "often many minutes or longer"
    assert (
        runtime_scale_hint(rhs_mode_hint=2, total_size_hint=10_000, n_rhs_hint=3)
        == "usually seconds to a few minutes"
    )
    assert runtime_scale_hint(rhs_mode_hint=3, total_size_hint=80_000, n_rhs_hint=4) == "often many minutes or longer"


def test_rhs1_progress_threshold_is_environment_guarded() -> None:
    assert rhs1_progress_size_min(environ={}) == 20_000
    assert rhs1_progress_size_min(environ={"SFINCS_JAX_PROGRESS_SIZE_MIN": "12"}) == 12
    assert rhs1_progress_size_min(environ={"SFINCS_JAX_PROGRESS_SIZE_MIN": "bad"}) == 20_000
    assert rhs1_large_progress_enabled(rhs_mode=1, total_size=12, environ={"SFINCS_JAX_PROGRESS_SIZE_MIN": "12"})
    assert not rhs1_large_progress_enabled(rhs_mode=2, total_size=12, environ={"SFINCS_JAX_PROGRESS_SIZE_MIN": "12"})
    assert not rhs1_large_progress_enabled(rhs_mode=1, total_size=11, environ={"SFINCS_JAX_PROGRESS_SIZE_MIN": "12"})


def test_rhs1_progress_notes_are_one_shot() -> None:
    messages: list[tuple[int, str]] = []
    notes = RHS1ProgressNotes(emit=lambda level, msg: messages.append((level, msg)), enabled=True)

    notes.preconditioner_build("collision")
    notes.preconditioner_build("xmg")
    notes.krylov_start()
    notes.krylov_start()

    assert messages == [
        (
            0,
            " solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner "
            "(collision); this stage can take a while for large systems.",
        ),
        (0, " solve_v3_full_system_linear_gmres: starting Krylov iterations."),
    ]


def test_transport_progress_message_estimates_remaining_time() -> None:
    assert transport_progress_message(completed=2, total=5, avg_rhs_s=61.0, elapsed_s=125.0) == (
        "solve_v3_transport_matrix_linear_gmres: progress 2/5 "
        "avg_rhs=1m01s elapsed=2m05s est_remaining=3m03s"
    )
