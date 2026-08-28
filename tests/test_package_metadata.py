from __future__ import annotations

from pathlib import Path
import tomllib

import dkx


def test_package_version_has_one_literal_source() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text())

    assert "version" not in metadata["project"]
    assert "version" in metadata["project"]["dynamic"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "dkx._version.__version__"
    }
    assert dkx.__version__ == "2.3.1"
