from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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


def test_compress_images_reduces_compressible_png(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image

    image_path = tmp_path / "compressible.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(image_path, compress_level=0)
    before = image_path.stat().st_size

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/compress_images.py", "--apply", str(tmp_path)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert image_path.stat().st_size < before
