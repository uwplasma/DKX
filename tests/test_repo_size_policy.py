from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_tracked_large_files_are_reviewed() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_repo_size.py"],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
