from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 gate
    import tomli as tomllib

import dkx


def test_package_version_has_one_literal_source() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text())

    assert "version" not in metadata["project"]
    assert "version" in metadata["project"]["dynamic"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "dkx._version.__version__"
    }

    assignments: list[tuple[Path, ast.Assign]] = []
    for path in (root / "dkx").rglob("*.py"):
        tree = ast.parse(path.read_text())
        assignments.extend(
            (path, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        )
    assert [path.relative_to(root).as_posix() for path, _ in assignments] == [
        "dkx/_version.py"
    ]
    assert dkx.__version__ == ast.literal_eval(assignments[0][1].value)
