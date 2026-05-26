from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile

from sfincs_jax import data_fetch


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

    try:
        data_fetch.ensure_external_equilibrium_data(quiet=True)
    except FileNotFoundError as exc:
        assert "SFINCS_JAX_OFFLINE" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("missing offline cache should raise FileNotFoundError")
