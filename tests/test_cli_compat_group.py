"""The SFINCS commands live under ``dkx sfincs`` without breaking old scripts.

plan.md section 5.6 wants SFINCS-specific operations in a compatibility group
rather than as unrelated top-level commands. They are registered twice: once
under ``sfincs``, where they are documented, and once at the top level with
suppressed help so existing invocations keep working.

Both halves need pinning. Losing the group makes the help output 21 entries
again; losing the aliases silently breaks every script and CI job that calls
the old spelling.
"""

from __future__ import annotations

import argparse

import pytest

from dkx import cli

#: The SFINCS commands, as they were spelled before the group existed.
COMPAT_COMMANDS = (
    "solve-v3", "scan-er", "ambipolar", "ambipolar-solve", "run-fortran",
    "write-output", "transport-matrix-v3", "monoenergetic-database",
    "dump-h5", "plot-output", "compare-h5", "postprocess-upstream",
)


def build_parser() -> argparse.ArgumentParser:
    """The real parser, built the way ``main`` builds it."""
    parser = argparse.ArgumentParser(prog="dkx")
    cli._add_common_cli_args(parser)
    cli._add_parallel_cli_args(parser)
    sub = parser.add_subparsers(
        dest="cmd", required=True, metavar="{" + ",".join(cli._USER_COMMANDS) + "}"
    )
    p_sfincs = sub.add_parser("sfincs")
    cli._add_compat_parsers(p_sfincs.add_subparsers(dest="sfincs_cmd", required=True))
    cli._add_compat_parsers(cli._HiddenAliases(sub))
    return parser, sub, p_sfincs


@pytest.mark.parametrize("command", COMPAT_COMMANDS)
def test_every_sfincs_command_is_reachable_under_the_group(command: str) -> None:
    _, _, p_sfincs = build_parser()
    group_sub = next(
        action for action in p_sfincs._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert command in group_sub.choices


@pytest.mark.parametrize("command", COMPAT_COMMANDS)
def test_every_sfincs_command_still_works_at_the_top_level(command: str) -> None:
    """The old spelling must keep running.

    These names appear in existing scripts, CI jobs and the upstream
    comparison harness; removing them to tidy the help output would break work
    that has nothing to do with this rename.
    """
    _, sub, _ = build_parser()
    assert command in sub.choices


@pytest.mark.parametrize("command", COMPAT_COMMANDS)
def test_the_top_level_alias_is_hidden_from_help(command: str) -> None:
    """Registered but not advertised.

    If the aliases carried their help strings, the compatibility group would
    make ``dkx --help`` longer rather than shorter, which is the opposite of
    what it is for.
    """
    _, sub, _ = build_parser()
    action = next(
        a for a in sub._get_subactions() if a.dest == command
    ) if any(a.dest == command for a in sub._get_subactions()) else None
    assert action is None or action.help is argparse.SUPPRESS


def test_help_advertises_only_the_user_commands() -> None:
    parser, _, _ = build_parser()
    usage = parser.format_usage()
    assert "{" + ",".join(cli._USER_COMMANDS) + "}" in usage
    for command in COMPAT_COMMANDS:
        assert f",{command}," not in usage, f"{command} leaked into the usage line"


@pytest.mark.parametrize("command", cli._USER_COMMANDS)
def test_every_advertised_command_is_really_dispatchable(command: str) -> None:
    """``_USER_COMMANDS`` is hand-kept and drives the metavar; this stops it lying.

    Checked against the parser ``main`` actually builds, not against a fixture:
    an earlier version of this test rebuilt only the compatibility half and so
    asserted nothing about the user commands. Advertising a command that is not
    registered would send it down the implicit-namelist path and fail on a file
    the user never named -- the bug that ``dkx doctor`` hit.
    """
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0


def test_the_metavar_lists_exactly_the_user_commands(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    usage = capsys.readouterr().out
    assert "{" + ",".join(cli._USER_COMMANDS) + "}" in usage


def test_the_two_registrations_are_independent_parsers() -> None:
    """argparse requires distinct parser objects; they must still share handlers."""
    _, sub, p_sfincs = build_parser()
    group_sub = next(
        a for a in p_sfincs._actions if isinstance(a, argparse._SubParsersAction)
    )
    top = sub.choices["dump-h5"]
    grouped = group_sub.choices["dump-h5"]
    assert top is not grouped
    assert top.get_default("func") is grouped.get_default("func")


def test_sfincs_is_dispatched_not_swallowed_as_a_filename() -> None:
    assert cli._normalize_default_argv(["sfincs", "dump-h5"]) == ["sfincs", "dump-h5"]


def test_a_bare_namelist_still_routes_to_write_output() -> None:
    """The implicit path predates the group and must survive it."""
    assert cli._normalize_default_argv(["input.namelist"])[0] == "write-output"
