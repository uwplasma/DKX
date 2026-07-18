from __future__ import annotations

import re
from pathlib import Path

import dkx


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"$', pyproject.read_text(), re.MULTILINE)
    assert match is not None
    assert dkx.__version__ == match.group(1)
