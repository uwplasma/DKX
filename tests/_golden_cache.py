"""Transparent lzma (de)compression of the large Fortran v3 reference goldens.

To keep the repository lightweight, the heavy binary parity fixtures under
``tests/ref/`` (the HDF5 ``sfincsOutput`` goldens, the PETSc matrix/vector
binaries, and any NetCDF ``wout`` files) are committed **only** in their
lzma-compressed ``*.xz`` form.  The uncompressed originals are git-ignored and
materialised on demand at pytest start-up by :func:`ensure_decompressed`, so
every parity test still reads a byte-identical golden while the tracked golden
footprint shrinks by roughly 15x (HDF5, dominated by per-dataset metadata,
compresses ~20x; the PETSc binaries ~7x).

``compress_all`` is the developer-side inverse used when regenerating or adding
goldens: drop the raw ``foo.sfincsOutput.h5`` into ``tests/ref/`` and run it to
produce the committed ``foo.sfincsOutput.h5.xz``.
"""

from __future__ import annotations

import lzma
import os
import tempfile
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent / "ref"

# The heavy binary golden types managed here.  Small text fixtures
# (``*.input.namelist``, ``*.json``) stay uncompressed and human-diffable.
COMPRESSED_SUFFIXES = (".h5", ".petscbin", ".nc")


def _managed_target(xz_path: Path) -> Path | None:
    """Return the uncompressed path an ``*.xz`` restores to, if we manage it."""

    if xz_path.suffix != ".xz":
        return None
    target = xz_path.with_suffix("")  # strip the trailing ``.xz``
    return target if target.suffix in COMPRESSED_SUFFIXES else None


def ensure_decompressed(ref_dir: Path = REF_DIR) -> int:
    """Materialise ``tests/ref/<name>`` for every committed ``<name>.xz``.

    Idempotent and cheap on repeat runs: a target is rewritten only when it is
    missing or older than its ``.xz`` source.  Returns the number of files
    (re)written, so callers can report cold-clone materialisation if they wish.
    """

    written = 0
    for xz_path in sorted(ref_dir.glob("*.xz")):
        target = _managed_target(xz_path)
        if target is None:
            continue
        if target.exists() and target.stat().st_mtime >= xz_path.stat().st_mtime:
            continue
        payload = lzma.decompress(xz_path.read_bytes())
        # Atomic write: decompress to a unique temp file in the same directory,
        # then rename into place.  Under ``pytest -n auto`` several xdist workers
        # run this concurrently; the temp+rename keeps any reader from ever
        # seeing a half-written golden (last writer wins, content identical).
        fd, tmp_name = tempfile.mkstemp(dir=str(ref_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        written += 1
    return written


def compress_all(ref_dir: Path = REF_DIR, preset: int = 9) -> list[Path]:
    """Developer tool: (re)compress every raw golden to ``<name>.xz``.

    Returns the list of ``.xz`` files written.  Use after adding or
    regenerating a reference fixture so only the compressed form is committed.
    """

    written: list[Path] = []
    for target in sorted(ref_dir.iterdir()):
        if target.suffix not in COMPRESSED_SUFFIXES:
            continue
        xz_path = target.with_name(target.name + ".xz")
        xz_path.write_bytes(lzma.compress(target.read_bytes(), preset=preset))
        written.append(xz_path)
    return written
