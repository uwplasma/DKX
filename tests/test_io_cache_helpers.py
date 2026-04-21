from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import (
    _OUTPUT_CACHE_FIELDS,
    _decode_if_bytes,
    _equilibrium_cache_identity,
    _file_content_identity,
    _fortran_h5_layout,
    _group_subset_key,
    _hashable_value,
    _load_output_cache,
    _output_cache_enabled,
    _output_cache_path,
    _save_output_cache,
    _to_numpy_for_h5,
)


def test_output_cache_enabled_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE_PERSIST", raising=False)
    assert _output_cache_enabled() is True

    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "off")
    assert _output_cache_enabled() is False

    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE", "on")
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_PERSIST", "off")
    assert _output_cache_enabled() is False


def test_output_cache_path_and_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_OUTPUT_CACHE_PERSIST", raising=False)

    key = ("a", 1, ("b", 2))
    path = _output_cache_path(key)
    assert path is not None
    assert path.parent == (tmp_path / "cache")

    payload = {
        "uHat": np.asarray([[1.0, 2.0]]),
        "VPrimeHat": np.asarray(3.0),
        "ignored": np.asarray(99.0),
    }
    _save_output_cache(key, payload)
    loaded = _load_output_cache(key)
    assert loaded is not None
    assert set(loaded).issubset(set(_OUTPUT_CACHE_FIELDS))
    np.testing.assert_allclose(loaded["uHat"], payload["uHat"])
    np.testing.assert_allclose(loaded["VPrimeHat"], payload["VPrimeHat"])
    assert "ignored" not in loaded


def test_output_cache_load_rejects_missing_or_bad_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_OUTPUT_CACHE_DIR", str(tmp_path / "cache"))
    key = ("x",)
    assert _load_output_cache(key) is None

    path = _output_cache_path(key)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cache_version=np.asarray(7, dtype=np.int32), uHat=np.asarray([1.0]))
    assert _load_output_cache(key) is None


def test_hashable_group_and_file_identity_helpers(tmp_path: Path) -> None:
    nested = {"b": [1, {"c": 2}], "a": 3}
    hv = _hashable_value(nested)
    assert hv == (("a", 3), ("b", (1, (("c", 2),))))

    group = {"A": 1, "B": [2, 3]}
    subset = _group_subset_key(group, ("B", "A", "MISSING"))
    assert subset == (("B", (2, 3)), ("A", 1), ("MISSING", None))

    data_path = tmp_path / "eq.dat"
    data_path.write_bytes(b"abc")
    st = data_path.stat()
    ident = _file_content_identity(str(data_path.resolve()), int(st.st_mtime_ns), int(st.st_size))
    assert ident[0] == 3
    assert isinstance(ident[1], str) and len(ident[1]) == 32
    assert _equilibrium_cache_identity(data_path) == ident
    assert _equilibrium_cache_identity(tmp_path / "missing") is None


def test_decode_and_h5_layout_helpers() -> None:
    assert _decode_if_bytes(b"abc") == "abc"
    assert _decode_if_bytes(np.asarray([b"abc"])) == "abc"
    arr = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(_fortran_h5_layout(arr), arr.T)
    np.testing.assert_array_equal(_fortran_h5_layout(np.arange(3)), np.arange(3))

    class _ArrayLike:
        def __array__(self):
            return np.asarray([[1.0, 2.0]])

    np.testing.assert_array_equal(_to_numpy_for_h5(_ArrayLike()), np.asarray([[1.0, 2.0]]))
    assert _to_numpy_for_h5("x") == "x"
