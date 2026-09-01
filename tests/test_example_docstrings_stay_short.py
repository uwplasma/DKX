"""Example module docstrings stay short enough that the code is still on screen.

A header that runs past ~20 lines stops orienting the reader and starts
displacing the thing it introduces.  This is a style budget, not a physics
claim, so it is a cheap check rather than a review comment every time.
"""

import ast
from pathlib import Path

import pytest

# 24 rather than 20: plan.md section 9.2 requires the header to carry the
# physical regime, the expected runtime, and the equivalent CLI command, and a
# rung with no case.toml has to say why.  That is a real floor, so the budget
# is set just above it rather than pretending those lines are optional.
MAX_LINES = 24
EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"
# The numbered ladder is the curated path users are pointed at; the older topic
# folders predate this budget and are being retired rather than reformatted.
EXAMPLES = sorted(EXAMPLES_ROOT.glob("0[1-9]_*/run.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_module_docstring_is_at_most_twenty_lines(path: Path) -> None:
    docstring = ast.get_docstring(ast.parse(path.read_text()))
    assert docstring is not None, f"{path.name} has no module docstring"
    lines = len(docstring.splitlines())
    assert lines <= MAX_LINES, (
        f"{path.name} docstring is {lines} lines; keep it to {MAX_LINES} so the "
        f"code is still visible when the file opens.  Move the rationale for a "
        f"specific line down to a comment beside that line."
    )
