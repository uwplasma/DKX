from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from sfincs_jax.validation import data_fetch


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_external_data_manifest_lists_release_fixtures() -> None:
    manifest = data_fetch.external_data_manifest()

    assert manifest["release_tag"] == "sfincs-jax-data-v1"
    assert manifest["archive_sha256"]
    assert "wout_w7x_standardConfig.nc" in data_fetch.known_external_equilibrium_names()
    assert "hsx3free.bc" in data_fetch.known_external_equilibrium_names()
    for item in manifest["files"]:
        assert Path(item["path"]).name
        assert int(item["size"]) > 0
        assert len(item["sha256"]) == 64


def test_unknown_external_equilibrium_does_not_fetch() -> None:
    assert data_fetch.resolve_external_equilibrium("not_a_known_fixture.nc", fetch=False) is None


def test_external_data_cache_root_honors_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "explicit"))
    assert data_fetch.data_cache_root() == tmp_path / "explicit"

    monkeypatch.delenv("SFINCS_JAX_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert data_fetch.data_cache_root() == tmp_path / "xdg" / "sfincs_jax" / "data"


def test_external_data_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"unsafe"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive, "r:gz") as tf:
        try:
            data_fetch._safe_extract(tf, tmp_path / "target")
        except RuntimeError as exc:
            assert "Unsafe path" in str(exc)
        else:  # pragma: no cover - defensive assertion branch
            raise AssertionError("unsafe archive member should be rejected")


def test_external_data_download_rejects_checksum_mismatch(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    archive = release_dir / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("payload.txt")
        payload = b"payload"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "cache" / "payload.tar.gz"
    manifest = {
        "release_url": release_dir.as_uri(),
        "asset": archive.name,
        "archive_sha256": "0" * 64,
    }

    try:
        data_fetch._download_archive(manifest, destination)
    except RuntimeError as exc:
        assert "Checksum mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("checksum mismatch should fail")
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()


def test_external_equilibrium_download_extract_and_resolve(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_OFFLINE", raising=False)
    payload = b"small release-hosted fixture\n"
    rel_path = Path("sfincs_jax/data/equilibria/tiny.bc")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    archive_path = release_dir / "tiny-data.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tf:
        info = tarfile.TarInfo(rel_path.as_posix())
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": release_dir.as_uri(),
        "asset": archive_path.name,
        "archive_sha256": data_fetch._sha256(archive_path),
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        ],
    }

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)

    target = data_fetch.ensure_external_equilibrium_data(quiet=True)
    resolved = data_fetch.resolve_external_equilibrium("tiny.bc", fetch=False)

    assert target == tmp_path / "cache" / "test-v1"
    assert resolved == target / rel_path
    assert resolved.read_bytes() == payload
    assert (target / ".complete").read_text(encoding="ascii") == "ok\n"


def test_external_equilibrium_cached_data_short_circuits_download(tmp_path: Path, monkeypatch) -> None:
    payload = b"already cached\n"
    rel_path = Path("sfincs_jax/data/equilibria/tiny_cached.bc")
    target = tmp_path / "cache" / "test-v1"
    cached = target / rel_path
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    (target / ".complete").write_text("ok\n", encoding="ascii")
    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": "https://example.invalid/release",
        "asset": "missing.tar.gz",
        "archive_sha256": "0" * 64,
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        ],
    }

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SFINCS_JAX_OFFLINE", "1")
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)

    assert data_fetch.ensure_external_equilibrium_data(quiet=True) == target
    assert data_fetch.resolve_external_equilibrium("'tiny_cached.bc'", fetch=True) == cached


def test_external_equilibrium_resolver_fetches_known_basename_on_cache_miss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"created by fake fetch\n"
    rel_path = Path("sfincs_jax/data/equilibria/tiny_fetch.nc")
    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": "https://example.invalid/release",
        "asset": "missing.tar.gz",
        "archive_sha256": "0" * 64,
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        ],
    }
    cache_root = tmp_path / "cache"
    target = cache_root / "test-v1"
    calls: list[str] = []

    def _fake_fetch() -> Path:
        calls.append("fetch")
        fixture = target / rel_path
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(payload)
        return target

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(cache_root))
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(data_fetch, "ensure_external_equilibrium_data", _fake_fetch)

    assert data_fetch.resolve_external_equilibrium('"tiny_fetch.nc"', fetch=True) == target / rel_path
    assert calls == ["fetch"]


def test_external_equilibrium_resolver_rejects_corrupt_cached_file_without_fetch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    good_payload = b"expected fixture\n"
    bad_payload = b"wrong fixture\n"
    rel_path = Path("sfincs_jax/data/equilibria/tiny_corrupt.nc")
    target = tmp_path / "cache" / "test-v1"
    cached = target / rel_path
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(bad_payload)
    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": "https://example.invalid/release",
        "asset": "missing.tar.gz",
        "archive_sha256": "0" * 64,
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": len(good_payload),
                "sha256": _sha256_bytes(good_payload),
            }
        ],
    }

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)

    assert data_fetch.resolve_external_equilibrium("tiny_corrupt.nc", fetch=False) is None


def test_external_equilibrium_resolver_returns_none_if_fetch_does_not_produce_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rel_path = Path("sfincs_jax/data/equilibria/tiny_missing_after_fetch.nc")
    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": "https://example.invalid/release",
        "asset": "missing.tar.gz",
        "archive_sha256": "0" * 64,
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": 7,
                "sha256": "1" * 64,
            }
        ],
    }
    calls: list[str] = []

    def _fake_fetch() -> Path:
        calls.append("fetch")
        return tmp_path / "cache" / "test-v1"

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(data_fetch, "ensure_external_equilibrium_data", _fake_fetch)

    assert data_fetch.resolve_external_equilibrium("tiny_missing_after_fetch.nc", fetch=True) is None
    assert calls == ["fetch"]


def test_external_equilibrium_file_verification_checks_missing_size_and_hash(
    tmp_path: Path,
) -> None:
    rel_path = Path("sfincs_jax/data/equilibria/tiny_verify.bc")
    payload = b"reference fixture\n"
    manifest = {
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        ],
    }

    assert not data_fetch._verify_files(tmp_path, manifest)

    fixture = tmp_path / rel_path
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(payload + b"extra")
    assert not data_fetch._verify_files(tmp_path, manifest)

    fixture.write_bytes(b"wrong-length-ok!")
    bad_hash_manifest = {
        "files": [
            {
                "path": rel_path.as_posix(),
                "size": fixture.stat().st_size,
                "sha256": _sha256_bytes(payload),
            }
        ],
    }
    assert not data_fetch._verify_files(tmp_path, bad_hash_manifest)

    fixture.write_bytes(payload)
    assert data_fetch._verify_files(tmp_path, manifest)


def test_download_archive_accepts_matching_checksum_and_replaces_temp_file(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    archive = release_dir / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("payload.txt")
        payload = b"payload"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "cache" / "payload.tar.gz"
    manifest = {
        "release_url": release_dir.as_uri() + "/",
        "asset": archive.name,
        "archive_sha256": data_fetch._sha256(archive),
    }

    data_fetch._download_archive(manifest, destination)

    assert destination.read_bytes() == archive.read_bytes()
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()


def test_external_equilibrium_offline_missing_cache_raises(tmp_path: Path, monkeypatch) -> None:
    manifest = {
        "version": "test-v1",
        "release_tag": "test-data",
        "release_url": "https://example.invalid/release",
        "asset": "missing.tar.gz",
        "archive_sha256": "0" * 64,
        "files": [],
    }

    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SFINCS_JAX_OFFLINE", "1")
    monkeypatch.setattr(data_fetch, "_load_manifest", lambda: manifest)

    with pytest.raises(FileNotFoundError, match="SFINCS_JAX_OFFLINE"):
        data_fetch.ensure_external_equilibrium_data(quiet=True)
