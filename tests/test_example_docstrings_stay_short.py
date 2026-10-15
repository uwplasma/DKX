"""Tutorial docstrings stay short enough that the code is still on screen.

Each header explains the physics, the equations and how to read the output,
plus the run/CLI/smoke lines; past ~26 lines it starts displacing the code it
introduces.  Rationale for a specific line belongs in a comment beside it.
"""

import ast
from pathlib import Path

import pytest

MAX_LINES = 26
TUTORIALS = sorted((Path(__file__).resolve().parents[1] / "examples" / "tutorials").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_module_docstring_is_short(path: Path) -> None:
    docstring = ast.get_docstring(ast.parse(path.read_text()))
    assert docstring is not None, f"{path.name} has no module docstring"
    assert len(docstring.splitlines()) <= MAX_LINES, f"{path.name}: keep the docstring to {MAX_LINES} lines"
