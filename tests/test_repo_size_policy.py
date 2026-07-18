from __future__ import annotations

from pathlib import Path

import pytest

from dkx.validation import release


def test_tracked_large_files_are_reviewed() -> None:
    assert release.check_size_main([]) == 0


def test_compress_images_reduces_compressible_png(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image

    image_path = tmp_path / "compressible.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(image_path, compress_level=0)
    before = image_path.stat().st_size

    assert release.compress_images_main(["--apply", str(tmp_path)]) == 0
    assert image_path.stat().st_size < before
