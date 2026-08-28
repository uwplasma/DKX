"""Measure DKX-owned release artifacts without importing DKX or JAX."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


MIB = 1024 * 1024
DEFAULT_LIMIT_MIB = 20.0
MEDIA_SUFFIXES = {".gif", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}


@dataclass(frozen=True)
class Measurement:
    """One independently enforceable DKX-owned size measurement."""

    name: str
    bytes: int
    limit_bytes: int
    enforced: bool

    @property
    def passes(self) -> bool:
        return self.bytes < self.limit_bytes

    def payload(self) -> dict[str, object]:
        result = asdict(self)
        result["mib"] = self.bytes / MIB
        result["limit_mib"] = self.limit_bytes / MIB
        result["passes"] = self.passes
        return result


def _file_tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / name.decode() for name in raw.split(b"\0") if name]


def tracked_worktree_size(root: Path) -> int:
    """Return the bytes in tracked working-tree files, excluding Git history."""

    return sum(path.stat().st_size for path in _tracked_files(root) if path.is_file())


def tracked_media_size(root: Path) -> int:
    """Return the bytes in tracked documentation and publication media."""

    return sum(
        path.stat().st_size
        for path in _tracked_files(root)
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    )


def installed_distribution_size(distribution: str = "dkx") -> int:
    """Return all installed DKX package and dist-info bytes, including caches."""

    dist = metadata.distribution(distribution)
    site_packages = Path(dist.locate_file("")).resolve()
    package = site_packages / distribution.replace("-", "_")
    candidates = [package, *site_packages.glob(f"{distribution.replace('-', '_')}-*.dist-info")]
    return sum(_file_tree_size(path) for path in candidates if path.is_dir())


def artifact_sizes(dist_dir: Path) -> dict[str, int]:
    """Require and measure exactly one wheel and one source distribution."""

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"expected one wheel and one sdist in {dist_dir}; "
            f"found wheels={wheels}, sdists={sdists}"
        )
    return {"wheel": wheels[0].stat().st_size, "sdist": sdists[0].stat().st_size}


def collect_measurements(
    *,
    repository: Path | None,
    dist_dir: Path | None,
    full_clone: Path | None,
    include_installed: bool,
    enforced: set[str],
    limit_bytes: int,
) -> list[Measurement]:
    """Collect requested measurements in stable display order."""

    sizes: dict[str, int] = {}
    if repository is not None:
        sizes["tracked_worktree"] = tracked_worktree_size(repository)
        sizes["tracked_media"] = tracked_media_size(repository)
    if dist_dir is not None:
        sizes.update(artifact_sizes(dist_dir))
    if include_installed:
        sizes["installed"] = installed_distribution_size()
    if full_clone is not None:
        sizes["full_clone"] = _file_tree_size(full_clone)
    unknown = enforced - sizes.keys()
    if unknown:
        raise ValueError(f"cannot enforce measurements that were not requested: {sorted(unknown)}")
    return [
        Measurement(name=name, bytes=value, limit_bytes=limit_bytes, enforced=name in enforced)
        for name, value in sizes.items()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--full-clone", type=Path)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--limit-mib", type=float, default=DEFAULT_LIMIT_MIB)
    parser.add_argument("--enforce", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    try:
        measurements = collect_measurements(
            repository=args.repository,
            dist_dir=args.dist_dir,
            full_clone=args.full_clone,
            include_installed=args.installed,
            enforced=set(args.enforce),
            limit_bytes=int(args.limit_mib * MIB),
        )
    except (ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    payload = {"schema_version": 1, "measurements": [item.payload() for item in measurements]}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    failed = [item.name for item in measurements if item.enforced and not item.passes]
    if failed:
        print(f"size contract failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
