"""Tests for ``dkx doctor`` and the subcommand dispatch it exposed.

``doctor`` reports whether an install can actually run. The checks that earn
their place are the ones whose failure is otherwise silent: float64 quietly
off, or a solvax below the floor that fails deep inside a solve rather than at
import. Both produce plausible wrong numbers or a traceback naming neither dkx
nor the cause.
"""

from __future__ import annotations

import json

import pytest

from dkx import cli


def test_doctor_reports_a_healthy_install_and_exits_zero(capsys) -> None:
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "python" in out
    assert "float64" in out


def test_doctor_json_is_machine_readable(capsys) -> None:
    assert cli.main(["doctor", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["float64"] == {"status": "ok", "detail": "active"}
    assert report["python"]["status"] == "ok"
    assert set(report["devices"]) == {"status", "detail"}


def test_every_check_reports_a_known_status() -> None:
    rows = cli._doctor_checks()
    assert rows, "doctor must report at least one check"
    assert {status for status, _, _ in rows} <= {"ok", "warn", "fail"}
    names = [name for _, name, _ in rows]
    assert len(names) == len(set(names)), f"duplicate check names: {names}"


def test_doctor_observes_float64_rather_than_trusting_the_environment() -> None:
    """The float64 row must come from an allocated array, not from an env var.

    ``JAX_ENABLE_X64`` being set is not evidence that float64 is active: a
    backend initialised by an earlier import ignores it. Under this suite the
    runtime is configured, so the observed answer is ``active`` -- and the row
    says ``active`` only because an array came back ``float64``, which is what
    this pins.
    """
    import jax.numpy as jnp

    status, _, detail = next(row for row in cli._doctor_checks() if row[1] == "float64")
    observed = str(jnp.zeros(1, dtype=jnp.float64).dtype)
    assert (status, detail) == (("ok", "active") if observed == "float64"
                                else ("fail", f"arrays materialize as {observed}; results will be wrong"))


def test_a_failing_check_makes_doctor_exit_non_zero(monkeypatch, capsys) -> None:
    """Exit status is the machine-readable half of the report.

    A doctor that printed a red row and still exited 0 would pass in any CI
    that only checks the return code, which is the place the report is most
    likely to be read by a machine rather than a person.
    """
    monkeypatch.setattr(
        cli, "_doctor_checks", lambda: [("ok", "python", "3.11.14"), ("fail", "solvax", "missing")]
    )
    assert cli.main(["doctor"]) == 1
    assert "solvax" in capsys.readouterr().err


def test_a_warning_check_alone_does_not_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "_doctor_checks", lambda: [("ok", "python", "3.11.14"), ("warn", "netCDF4", "missing")]
    )
    assert cli.main(["doctor"]) == 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(cli._KNOWN_COMMANDS))
def test_a_registered_command_is_never_swallowed_as_a_file_path(command: str) -> None:
    """Regression: a subcommand missing from the dispatch set became a filename.

    ``dkx`` lets the namelist path be given with no command, which it
    implements by inserting ``write-output`` when the first token is not a
    known command. When ``doctor`` was added and the hand-kept command set was
    not updated, ``dkx doctor`` was read as ``dkx write-output doctor`` and
    died with ``FileNotFoundError: .../doctor`` -- a file the user never named.
    """
    assert cli._normalize_default_argv([command]) == [command]


def test_an_unknown_first_token_is_still_treated_as_a_namelist() -> None:
    """The implicit path is the reason the dispatch set exists; keep it working."""
    assert cli._normalize_default_argv(["input.namelist"])[0] == "write-output"
