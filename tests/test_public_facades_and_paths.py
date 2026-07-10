from __future__ import annotations

from pathlib import Path

import pytest

import sfincs_jax.io as io_module
from sfincs_jax.outputs import writer as output_writer
from sfincs_jax.paths import resolve_existing_path
from sfincs_jax.validation import data_fetch


def test_resolve_existing_path_uses_extra_search_dirs_and_basename(tmp_path: Path) -> None:
    """Stale absolute equilibrium paths should resolve by basename in search dirs."""

    extra_root = tmp_path / "equilibria"
    target = extra_root / "nested" / "wout_unit.nc"
    target.parent.mkdir(parents=True)
    target.write_text("fixture\n", encoding="utf-8")

    resolved_relative = resolve_existing_path(
        "nested/wout_unit.nc",
        base_dir=tmp_path / "missing-base",
        env_search_var="SFINCS_JAX_TEST_EMPTY_SEARCH",
        extra_search_dirs=(extra_root,),
    )
    assert resolved_relative.path == target

    stale_absolute = tmp_path / "old-machine" / "missing" / "wout_unit.nc"
    resolved_stale = resolve_existing_path(
        stale_absolute,
        env_search_var="SFINCS_JAX_TEST_EMPTY_SEARCH",
        extra_search_dirs=(target.parent,),
    )
    assert resolved_stale.path == target
    assert stale_absolute in resolved_stale.tried


def test_resolve_existing_path_uses_release_data_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "cache" / "known.bc"
    external.parent.mkdir()
    external.write_text("external\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_resolve(path: str | Path) -> Path:
        calls.append(Path(path))
        return external

    monkeypatch.setattr(data_fetch, "resolve_external_equilibrium", fake_resolve)

    resolved = resolve_existing_path(
        "known.bc",
        base_dir=tmp_path / "empty",
        env_search_var="SFINCS_JAX_TEST_EMPTY_SEARCH",
    )

    assert resolved.path == external
    assert calls == [Path("known.bc")]
    assert resolved.tried[-1] == external


def test_resolve_existing_path_failure_preserves_attempted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_fetch, "resolve_external_equilibrium", lambda _path: None)

    missing = tmp_path / "missing" / "wout_absent.nc"
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_existing_path(
            missing,
            base_dir=tmp_path,
            env_search_var="SFINCS_JAX_TEST_EMPTY_SEARCH",
            extra_search_dirs=(tmp_path / "extra",),
        )

    message = str(excinfo.value)
    assert "Unable to resolve existing path" in message
    assert "wout_absent.nc" in message
    assert str(missing) in message


def test_io_facade_delegates_legacy_private_getattr_and_setattr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility imports should keep working while implementations move."""

    original = output_writer._should_precompile_v3_full_system
    assert io_module._should_precompile_v3_full_system is original

    def replacement(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(io_module, "_should_precompile_v3_full_system", replacement)

    assert io_module._should_precompile_v3_full_system is replacement
    assert output_writer._should_precompile_v3_full_system is replacement


def test_io_facade_unknown_private_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(io_module, "_definitely_not_a_real_compat_name")
