from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen


_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "equilibria_manifest.json"


def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def data_cache_root() -> Path:
    """Return the writable cache root used for optional external data files."""

    override = os.environ.get("SFINCS_JAX_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "sfincs_jax" / "data"


def external_data_version() -> str:
    return str(_load_manifest()["version"])


def external_data_dir() -> Path:
    return data_cache_root() / external_data_version()


def external_data_manifest() -> dict:
    """Return the embedded manifest for release-hosted equilibrium fixtures."""

    return _load_manifest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _release_asset_url(manifest: dict) -> str:
    return f"{manifest['release_url'].rstrip('/')}/{manifest['asset']}"


def _safe_extract(tf: tarfile.TarFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in tf.getmembers():
        member_target = (target / member.name).resolve()
        if os.path.commonpath((str(target_resolved), str(member_target))) != str(target_resolved):
            raise RuntimeError(f"Unsafe path in data archive: {member.name!r}")
    tf.extractall(target)


def _download_archive(manifest: dict, destination: Path) -> None:
    url = _release_asset_url(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    digest = _sha256(tmp)
    if digest != manifest["archive_sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {url}: got {digest}, expected {manifest['archive_sha256']}"
        )
    tmp.replace(destination)


def _file_records_by_basename(manifest: dict) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for item in manifest["files"]:
        records.setdefault(Path(item["path"]).name, []).append(item)
    return records


def known_external_equilibrium_names() -> set[str]:
    manifest = _load_manifest()
    return set(_file_records_by_basename(manifest))


def _verify_files(target: Path, manifest: dict) -> bool:
    for item in manifest["files"]:
        path = target / item["path"]
        if not path.exists() or path.stat().st_size != int(item["size"]):
            return False
        if _sha256(path) != item["sha256"]:
            return False
    return True


def ensure_external_equilibrium_data(*, quiet: bool = False) -> Path:
    """Download and verify release-hosted equilibrium fixtures if needed.

    The package intentionally does not ship multi-megabyte VMEC/Boozer fixtures.
    This function restores them into a user cache for tests, examples, and
    compatibility with upstream-style ``equilibriumFile`` namelist paths.
    """

    manifest = _load_manifest()
    target = external_data_dir()
    complete_marker = target / ".complete"
    if complete_marker.exists() and _verify_files(target, manifest):
        return target

    target.mkdir(parents=True, exist_ok=True)
    archive = data_cache_root() / manifest["asset"]
    if not archive.exists() or _sha256(archive) != manifest["archive_sha256"]:
        if os.environ.get("SFINCS_JAX_OFFLINE", "").strip() in {"1", "true", "True"}:
            raise FileNotFoundError(
                "External SFINCS-JAX equilibrium data is not cached and SFINCS_JAX_OFFLINE is set."
            )
        if not quiet:
            print(f"Downloading SFINCS-JAX external equilibrium data from {_release_asset_url(manifest)}")
        _download_archive(manifest, archive)

    with tempfile.TemporaryDirectory(prefix="sfincs_jax_data_", dir=str(data_cache_root())) as tmpdir:
        tmp_target = Path(tmpdir) / manifest["version"]
        tmp_target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            _safe_extract(tf, tmp_target)
        if not _verify_files(tmp_target, manifest):
            raise RuntimeError("External SFINCS-JAX equilibrium data verification failed after extraction.")
        for item in manifest["files"]:
            rel = Path(item["path"])
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tmp_target / rel
            src.replace(dst)
    complete_marker.write_text("ok\n", encoding="ascii")
    return target


def resolve_external_equilibrium(path: str | Path, *, fetch: bool = True) -> Path | None:
    """Resolve a known external equilibrium path by basename from the cache."""

    manifest = _load_manifest()
    p = Path(str(path).strip().strip('"').strip("'"))
    records = _file_records_by_basename(manifest).get(p.name)
    if not records:
        return None
    target = external_data_dir()
    for item in records:
        candidate = target / item["path"]
        if candidate.exists() and candidate.stat().st_size == int(item["size"]):
            if _sha256(candidate) == item["sha256"]:
                return candidate
    if not fetch:
        return None
    ensure_external_equilibrium_data()
    for item in records:
        candidate = target / item["path"]
        if candidate.exists():
            return candidate
    return None
