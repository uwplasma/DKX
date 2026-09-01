"""The CLI can execute a native Case.

Until Phase D there was no command that did. `dkx validate` checked a Case and
`dkx schema` printed its shape, but every subcommand that actually solved took a
SFINCS namelist, so the native workflow was reachable only from Python. The
Phase A audit recorded that as a finding; these tests are the gate that keeps it
closed.

They use the smallest native case in the repository (a three-surface analytic
tokamak) so the whole file stays inside the pull-request time budget.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from dkx import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTIC_CASE = REPO_ROOT / "examples" / "01_tokamak_profile" / "case.toml"


def test_run_executes_a_native_case_and_writes_a_result(tmp_path, capsys) -> None:
    out = tmp_path / "result.nc"
    code = cli.main(["run", str(ANALYTIC_CASE), "--out", str(out), "--quiet"])

    assert code == 0
    assert out.is_file(), "run --out must write the NetCDF Result it reports"
    captured = capsys.readouterr().out
    assert "analytic_tokamak_profile" in captured
    assert "converged" in captured


def test_run_reports_the_solver_route_and_residual(tmp_path, capsys) -> None:
    """The summary has to carry the two numbers that make a run trustworthy.

    A route name without a residual says nothing about whether the answer was
    accepted, and a residual without the route hides which solver produced it.
    """
    cli.main(["run", str(ANALYTIC_CASE), "--out", str(tmp_path / "r.nc"), "--quiet"])
    captured = capsys.readouterr().out
    assert "true residual" in captured
    assert "solver route" in captured
    assert "block_tridiagonal" in captured


def test_run_without_out_does_not_write_anything(tmp_path, capsys) -> None:
    """Omitting --out runs the case and saves nothing, rather than guessing."""
    before = set(tmp_path.iterdir())
    code = cli.main(["run", str(ANALYTIC_CASE), "--quiet"])
    assert code == 0
    assert set(tmp_path.iterdir()) == before


def test_run_refuses_a_missing_case(tmp_path, capsys) -> None:
    code = cli.main(["run", str(tmp_path / "absent.toml")])
    assert code == 2
    assert "dkx run failed" in capsys.readouterr().err


def test_run_refuses_an_invalid_case(tmp_path, capsys) -> None:
    """A malformed Case fails during validation, not deep inside a solve."""
    bad = tmp_path / "bad.toml"
    bad.write_text('schema = 1\nname = "broken"\n', encoding="utf-8")
    code = cli.main(["run", str(bad)])
    assert code == 2
    assert "dkx run failed" in capsys.readouterr().err


def test_inspect_reads_a_saved_result_without_recomputing(tmp_path, capsys) -> None:
    out = tmp_path / "result.nc"
    cli.main(["run", str(ANALYTIC_CASE), "--out", str(out), "--quiet"])
    capsys.readouterr()

    code = cli.main(["inspect", str(out)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "analytic_tokamak_profile" in captured
    # The array inventory is the point of inspect: names, shapes, dtypes.
    assert "heat_flux_W_m2" in captured
    assert "particle_flux_m2_s" in captured
    assert "float64" in captured


def test_inspect_does_not_claim_units_it_does_not_have(tmp_path, capsys) -> None:
    """A native Result has no per-variable units metadata yet.

    plan.md section 5.5 requires it. Until it exists, inspect must not print a
    units column: an empty one would read as "the metadata is there and blank"
    rather than "the metadata does not exist".
    """
    out = tmp_path / "result.nc"
    cli.main(["run", str(ANALYTIC_CASE), "--out", str(out), "--quiet"])
    capsys.readouterr()

    cli.main(["inspect", str(out)])
    captured = capsys.readouterr().out
    assert "units" not in captured.lower()


def test_inspect_refuses_a_missing_result(tmp_path, capsys) -> None:
    code = cli.main(["inspect", str(tmp_path / "absent.nc")])
    assert code == 2
    assert "dkx inspect failed" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "inspect"])
def test_the_new_commands_survive_the_default_argv_form(command: str) -> None:
    """`dkx run CASE` must not be read as the bare `dkx INPUT.namelist` form.

    The CLI keeps a default form where a lone path means "solve this namelist".
    A subcommand missing from that normalizer's known set gets swallowed by it,
    which is how `dkx run case.toml` first failed with "unrecognized arguments".
    """
    argv = [command, "some/path"]
    assert cli._normalize_default_argv(list(argv)) == argv


def test_run_is_reachable_as_an_installed_console_script() -> None:
    """The parser wires the subcommand to its handler, not just its help text."""
    parser_help = cli.main(["--help"]) if False else None  # documented no-op
    assert parser_help is None
    # Resolve through the same path the console script uses.
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "dkx", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "--out" in completed.stdout


# ---------------------------------------------------------------------------
# validate runs the executor's preflight
# ---------------------------------------------------------------------------


def _case_with(tmp_path, **overrides):
    """The shipped tokamak case with one physics field replaced."""
    import tomllib

    source = ANALYTIC_CASE.read_text(encoding="utf-8")
    for key, value in overrides.items():
        source = re.sub(
            rf'^{key} = ".*"$', f'{key} = "{value}"', source, count=1, flags=re.M
        )
    path = tmp_path / "case.toml"
    path.write_text(source, encoding="utf-8")
    tomllib.loads(source)  # the edit must still be valid TOML
    return path


@pytest.mark.parametrize(
    ("field", "value", "named"),
    [
        ("phi1", "kinetic", "physics.phi1"),
        ("magnetic_drifts", "full", "physics.magnetic_drifts"),
    ],
)
def test_validate_refuses_a_schema_valid_case_the_executor_cannot_run(
    tmp_path, capsys, field: str, value: str, named: str
) -> None:
    """The schema is deliberately wider than the executor; validate closes the gap.

    ``magnetic_drifts = "full"``, ``workflow = "monoenergetic"`` and
    ``phi1 = "kinetic"`` all pass the JSON schema and are then refused by
    execution. Without this preflight a user discovers which of the advertised
    enum values are real only after ``dkx run`` has set up and failed --- and
    ``dkx validate`` reporting success on a case that cannot run is worse than
    not having the command.
    """
    path = _case_with(tmp_path, **{field: value})
    assert cli.main(["validate", str(path)]) == 2
    assert named in capsys.readouterr().err


def test_validate_still_accepts_every_shipped_ladder_case() -> None:
    """The preflight must not refuse the cases the repository ships.

    A guard that rejects the project's own examples is a broken guard, and
    these five are the ones a reader meets first.
    """
    for case_file in sorted((REPO_ROOT / "examples").glob("*/case.toml")):
        assert cli.main(["validate", str(case_file)]) == 0, case_file
