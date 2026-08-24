"""Example module docstrings stay short enough that the code is still on screen.

A header that runs past ~20 lines stops orienting the reader and starts
displacing the thing it introduces.  This is a style budget, not a physics
claim, so it is a cheap check rather than a review comment every time.
"""

import ast
from pathlib import Path

import pytest

MAX_LINES = 20
EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples" / "1_basics").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_module_docstring_is_at_most_twenty_lines(path: Path) -> None:
    docstring = ast.get_docstring(ast.parse(path.read_text()))
    assert docstring is not None, f"{path.name} has no module docstring"
    lines = len(docstring.splitlines())
    assert lines <= MAX_LINES, (
        f"{path.name} docstring is {lines} lines; keep it to {MAX_LINES} so the "
        f"code is still visible when the file opens.  Move the rationale for a "
        f"specific line down to a comment beside that line."
    )
