from __future__ import annotations

import json
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "sfincs_jax"
EXPECTED_TREE = REPO_ROOT / "tests" / "fixtures" / "source_tree_expected.json"


def _expected_tree() -> dict[str, list[str]]:
    with EXPECTED_TREE.open(encoding="utf-8") as stream:
        return json.load(stream)


def _package_dirs() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    )


def _relative_dir(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def test_source_tree_does_not_gain_new_root_modules_or_packages() -> None:
    expected = _expected_tree()

    root_modules = sorted(path.name for path in PACKAGE_ROOT.glob("*.py"))
    root_packages = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )

    assert root_modules == expected["allowed_root_modules"]
    assert root_packages == expected["allowed_root_packages"]


def test_source_tree_nested_packages_are_explicit_refactor_debt() -> None:
    expected = _expected_tree()

    nested_packages = sorted(
        _relative_dir(path)
        for path in _package_dirs()
        if len(path.relative_to(PACKAGE_ROOT).parts) > 1
    )

    assert nested_packages == expected["temporary_nested_packages"]


def test_source_tree_init_only_packages_are_explicit_refactor_debt() -> None:
    expected = _expected_tree()

    init_only_packages: list[str] = []
    for path in _package_dirs():
        files = sorted(child.name for child in path.iterdir() if child.is_file())
        dirs = sorted(
            child.name
            for child in path.iterdir()
            if child.is_dir() and child.name != "__pycache__"
        )
        if set(files) <= {"__init__.py"} and len(dirs) <= 1:
            init_only_packages.append(_relative_dir(path))

    assert init_only_packages == expected["temporary_init_only_packages"]


def test_source_tree_consolidation_target_is_stricter_than_current_tree() -> None:
    expected = _expected_tree()

    assert set(expected["target_root_modules"]) < set(expected["allowed_root_modules"])
    assert set(expected["target_root_packages"]) < set(expected["allowed_root_packages"])
    assert expected["temporary_nested_packages"], "temporary nested packages should be reduced by later tranches"


def test_flattened_operator_legacy_imports_resolve_to_canonical_modules() -> None:
    assert not (PACKAGE_ROOT / "operators" / "profile_response").exists()

    for name in ("collisionless", "fblock", "full_system", "layout", "system"):
        legacy = importlib.import_module(f"sfincs_jax.operators.profile_response.{name}")
        canonical = importlib.import_module(f"sfincs_jax.operators.profile_{name}")
        assert legacy is canonical
