"""The README's Python quickstart must actually run, verbatim.

A user pasted it and got a NameError from inside numpy, because their
environment was wrong -- but nothing in the suite would have caught the snippet
itself drifting out of date, because no test ever executed it.  Documentation
that is never run is documentation that is eventually wrong.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def _quickstart_snippet() -> str:
    """The first ```python block that calls dkx.run -- the one users paste."""
    for block in re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S):
        if "dkx.run(" in block and "import dkx" in block:
            return block
    pytest.fail("no `import dkx` + `dkx.run(` block found in the README quickstart")


def test_the_quickstart_snippet_runs_and_prints_finite_numbers(tmp_path):
    snippet = _quickstart_snippet()
    script = tmp_path / "quickstart.py"
    script.write_text(snippet, encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          cwd=tmp_path, timeout=1800)  # fmt: skip
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]

    values = [float(line.split(":")[-1]) for line in done.stdout.splitlines()
              if line.startswith("particle flux:")]  # fmt: skip
    assert len(values) == 1, f"expected the printed flux, got {done.stdout!r}"
    assert all(v == v and abs(v) < float("inf") for v in values), values
    # A run that silently produced nothing would still print zeros.
    assert any(v != 0.0 for v in values), "the quickstart solved nothing"
    # The route is printed rather than a second physics number: the quickstart
    # resolution is deliberately unconverged, so showing which solver ran is
    # honest where showing a bootstrap current invited it to be quoted.
    assert any(line.startswith("solver route:") and line.split(":")[-1].strip()
               for line in done.stdout.splitlines()), done.stdout


def test_the_quickstart_uses_the_entry_point_we_document():
    """If the API is renamed, the README must move with it."""
    snippet = _quickstart_snippet()
    assert "dkx.run(" in snippet
    # The old five-entry-point surface is what the single `run` replaced; the
    # quickstart is the one place that must not drift back to it.
    assert "run_profile(" not in snippet
    assert "argparse" not in snippet and "__main__" not in snippet
