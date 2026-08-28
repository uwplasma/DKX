from __future__ import annotations

from pathlib import Path

import pytest

from tools import release_contracts


def test_artifact_sizes_requires_one_of_each(tmp_path: Path) -> None:
    (tmp_path / "dkx-1.0.whl").write_bytes(b"wheel")
    with pytest.raises(ValueError, match="one wheel and one sdist"):
        release_contracts.artifact_sizes(tmp_path)

    (tmp_path / "dkx-1.0.tar.gz").write_bytes(b"sdist")
    assert release_contracts.artifact_sizes(tmp_path) == {"wheel": 5, "sdist": 5}


def test_measurement_uses_a_strict_below_limit_contract() -> None:
    below = release_contracts.Measurement("wheel", 19, 20, True)
    equal = release_contracts.Measurement("wheel", 20, 20, True)

    assert below.passes
    assert not equal.passes
    assert equal.payload()["passes"] is False


def test_repository_measurements_count_only_tracked_files(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "code.py").write_bytes(b"1234")
    (tmp_path / "figure.png").write_bytes(b"123456")
    (tmp_path / "ignored.png").write_bytes(b"not tracked")
    subprocess.run(["git", "add", "code.py", "figure.png"], cwd=tmp_path, check=True)

    assert release_contracts.tracked_worktree_size(tmp_path) == 10
    assert release_contracts.tracked_media_size(tmp_path) == 6


def test_collect_rejects_an_unrequested_enforcement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not requested"):
        release_contracts.collect_measurements(
            repository=None,
            dist_dir=None,
            full_clone=None,
            include_installed=False,
            enforced={"wheel"},
            limit_bytes=20,
        )
