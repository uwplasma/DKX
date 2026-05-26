#!/usr/bin/env python3
"""Compress documentation raster images when a smaller encoding is available.

The script uses Pillow, a pip-installable package, and intentionally keeps the
policy conservative: PNGs are recompressed losslessly, while JPEGs are reencoded
with a high quality setting suitable for documentation screenshots and plots.
Files are replaced only when the optimized payload is smaller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile


RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class CompressionResult:
    path: Path
    before: int
    after: int
    changed: bool

    @property
    def saved(self) -> int:
        return max(0, self.before - self.after) if self.changed else 0


def _iter_images(roots: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = [p for p in root.rglob("*") if p.is_file()]
        images.extend(p for p in candidates if p.suffix.lower() in RASTER_EXTENSIONS)
    return sorted(set(images))


def _save_optimized(image, source: Path, target: Path, *, jpeg_quality: int) -> None:
    suffix = source.suffix.lower()
    if suffix == ".png":
        image.save(target, format="PNG", optimize=True, compress_level=9)
        return
    if suffix in {".jpg", ".jpeg"}:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(
            target,
            format="JPEG",
            optimize=True,
            progressive=True,
            quality=jpeg_quality,
        )
        return
    raise ValueError(f"Unsupported image suffix: {source}")


def compress_image(path: Path, *, apply: bool, jpeg_quality: int) -> CompressionResult:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit("Pillow is required: python -m pip install pillow") from exc

    before = path.stat().st_size
    with Image.open(path) as image:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".",
            suffix=path.suffix,
            dir=str(path.parent),
            delete=False,
        ) as tmp_file:
            tmp = Path(tmp_file.name)
        try:
            _save_optimized(image, path, tmp, jpeg_quality=jpeg_quality)
            after = tmp.stat().st_size
            if after < before:
                if apply:
                    tmp.replace(path)
                else:
                    tmp.unlink(missing_ok=True)
                return CompressionResult(path=path, before=before, after=after, changed=True)
            tmp.unlink(missing_ok=True)
            return CompressionResult(path=path, before=before, after=before, changed=False)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("docs")],
        help="Image files or directories to scan. Defaults to docs/.",
    )
    parser.add_argument("--apply", action="store_true", help="Replace files with smaller optimized payloads.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when reencoding .jpg/.jpeg files.",
    )
    args = parser.parse_args(argv)

    if not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg-quality must be between 1 and 100")

    images = _iter_images(args.roots)
    results = [
        compress_image(path, apply=args.apply, jpeg_quality=args.jpeg_quality)
        for path in images
    ]

    changed = [item for item in results if item.changed]
    saved = sum(item.saved for item in changed)
    mode = "applied" if args.apply else "dry-run"
    print(
        f"Image compression {mode}: {len(changed)}/{len(results)} files smaller, "
        f"saved {saved / 1024 / 1024:.3f} MiB."
    )
    for item in changed:
        print(
            f"  {item.before / 1024:.1f} KiB -> {item.after / 1024:.1f} KiB  "
            f"{item.path.as_posix()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
