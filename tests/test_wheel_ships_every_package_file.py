"""Every non-Python file the package needs must be declared as package data.

``dkx wout_*.nc`` shipped broken twice for pip users, both times for the same
reason: :mod:`dkx.representative` read its monoenergetic deck from a file that
existed in a source checkout and not in the wheel.  Every CI job runs from a
checkout, so every CI job was green.

The deck is a module-level string now, which fixes that one file.  This module
guards the *class* of bug.  ``[tool.setuptools.package-data]`` declares exactly
one pattern, ``dkx = ["validation/*.json"]``, so the next runtime file added
under ``src/dkx/`` with any other name or in any other directory is excluded from
the wheel by default and silently -- the build prints nothing, and only an
installed user finds out.

The companion check lives in the ``wheel-install`` CI job, which asserts the
same files are present in the built artifact.  This one is the fast gate: it
needs no build, so it fails in the pull request that adds the file.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import pytest

try:  # tomllib is 3.11+; pyproject.toml declares requires-python >= 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on the declared floor
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Files under ``src/dkx/`` that are deliberately absent from the wheel.
#:
#: ``src/dkx/README.md`` is the package-directory readme: rendered on GitHub and by
#: the docs build, never opened by the code.  Anything added here has to be
#: something the package never reads at runtime -- if it is read, it belongs in
#: ``package-data`` instead, because that is the whole failure this file exists
#: to stop.
NOT_SHIPPED = frozenset({"src/dkx/README.md"})


def _tracked_package_files() -> list[str]:
    """Every git-tracked path under ``src/dkx/``, as repo-relative POSIX strings.

    Asking git rather than walking the tree keeps ``__pycache__``, editable
    installs' ``.egg-info`` and stray local outputs out of the answer.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/dkx"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout; the wheel-install CI job covers this")
    return [name for name in out.stdout.split("\0") if name]


def _package_data_patterns() -> dict[str, list[str]]:
    if tomllib is None:
        pytest.skip("tomllib needs Python 3.11; the other CI jobs run 3.11")
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tool = config.get("tool", {}).get("setuptools", {}).get("package-data", {})
    return {package: list(patterns) for package, patterns in tool.items()}


def _matches(relative: str, pattern: str) -> bool:
    """setuptools-style match: ``*`` covers one path segment, not a subtree.

    ``fnmatch`` alone lets ``*`` swallow separators, so ``validation/*.json``
    would claim to cover ``validation/decks/anything.json`` -- which the build
    does not, in fact, include.  A per-segment comparison is what makes a green
    test mean the file ships.
    """
    rel_parts = relative.split("/")
    pat_parts = pattern.split("/")
    if len(rel_parts) != len(pat_parts):
        return False
    return all(fnmatch.fnmatchcase(r, p) for r, p in zip(rel_parts, pat_parts))


#: The src layout puts the package one directory down, but package-data
#: patterns are relative to the package itself, so the prefix has to come off
#: before matching.
SRC_PREFIX = "src/"


def declared_by_package_data(path: str) -> bool:
    """Is this repo-relative path covered by a ``package-data`` pattern?"""
    if path.startswith(SRC_PREFIX):
        path = path[len(SRC_PREFIX) :]
    for package, patterns in _package_data_patterns().items():
        prefix = package.replace(".", "/") + "/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        if any(_matches(relative, pattern) for pattern in patterns):
            return True
    return False


def test_every_tracked_non_python_file_under_dkx_ships_or_is_listed_as_excluded():
    """The gate itself: no undeclared data file may reach ``src/dkx/`` unnoticed."""
    undeclared = [
        path
        for path in _tracked_package_files()
        if not path.endswith(".py")
        and path not in NOT_SHIPPED
        and not declared_by_package_data(path)
    ]
    assert not undeclared, (
        "these files live under src/dkx/ but no [tool.setuptools.package-data] "
        f"pattern includes them, so the wheel will not carry them: {undeclared}. "
        "Add a pattern, or add the path to NOT_SHIPPED if the package never "
        "opens it at runtime."
    )


def test_the_matcher_does_not_credit_a_file_no_pattern_names():
    """Guard the matcher, or the gate above passes for the wrong reason.

    A matcher that says yes too easily turns the whole file green while the
    wheel stays empty --- the same silence that let 2.2.0 ship.  A real
    declared file must match, and a directory nothing declares must not.
    """
    assert declared_by_package_data("dkx/validation/equilibria_manifest.json")
    assert not declared_by_package_data("dkx/decks/representative.namelist")


def test_a_star_covers_one_path_segment_not_a_subtree():
    """The rule setuptools follows, and the one ``fnmatch`` alone gets wrong.

    ``fnmatch`` lets ``*`` swallow ``/``, so a bare ``fnmatch`` would call a
    file one level below ``validation/`` declared.  It would not be in the
    wheel, so the gate has to agree with the build, not with the glob.
    """
    assert _matches("validation/inner.json", "validation/*.json")
    assert not _matches("validation/decks/inner.json", "validation/*.json")
    assert not _matches("validation/inner.namelist", "validation/*.json")


def test_every_excluded_file_actually_exists():
    """A stale ``NOT_SHIPPED`` entry is an exemption nobody is watching."""
    missing = [name for name in sorted(NOT_SHIPPED) if not (REPO_ROOT / name).is_file()]
    assert not missing, f"NOT_SHIPPED names files that are gone: {missing}"


def test_package_data_is_declared_at_all():
    """Guard the guard: an empty table would make every assertion above vacuous."""
    assert _package_data_patterns(), "[tool.setuptools.package-data] is empty"
