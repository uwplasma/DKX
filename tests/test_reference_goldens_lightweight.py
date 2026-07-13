"""Guard the lightweight-repository invariant for the Fortran v3 goldens.

The heavy binary parity fixtures under ``tests/ref`` are committed only as
lzma-compressed ``*.xz`` and materialised on demand (see
``tests/_golden_cache.py`` and ``tests/conftest.py``).  These checks fail loudly
if a raw golden is ever committed uncompressed (which ``.gitignore`` would
silently drop) or if the compressed golden footprint regresses, so the repo
stays small without anyone having to remember the convention.
"""

from __future__ import annotations

import importlib.util
import lzma
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent / "ref"
COMPRESSED_SUFFIXES = (".h5", ".petscbin", ".nc")
# Regression tripwire: the whole compressed golden set is ~2.5 MB today; a
# generous ceiling catches accidental re-bloat without churning on new fixtures.
MAX_COMPRESSED_GOLDEN_BYTES = 5_000_000


def _load_golden_cache():
    helper = Path(__file__).resolve().parent / "_golden_cache.py"
    spec = importlib.util.spec_from_file_location("_sfincs_golden_cache_test", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_raw_golden_has_a_committed_compressed_source() -> None:
    """No uncompressed golden may exist without its ``.xz`` (else it is untracked)."""

    orphans = [
        path.name
        for path in REF_DIR.iterdir()
        if path.suffix in COMPRESSED_SUFFIXES
        and not path.with_name(path.name + ".xz").exists()
    ]
    assert not orphans, (
        "uncompressed goldens without a committed .xz source (they are gitignored "
        f"and would be lost): {sorted(orphans)} -- run compress_all() in "
        "tests/_golden_cache.py"
    )


def test_compressed_goldens_materialise_and_round_trip() -> None:
    """Every committed ``.xz`` decompresses to a non-empty, stable payload."""

    xz_files = sorted(REF_DIR.glob("*.xz"))
    assert xz_files, "no compressed goldens found under tests/ref"
    for xz_path in xz_files:
        if xz_path.with_suffix("").suffix not in COMPRESSED_SUFFIXES:
            continue
        payload = lzma.decompress(xz_path.read_bytes())
        assert payload, f"empty payload decompressed from {xz_path.name}"

    # ensure_decompressed is idempotent and leaves every managed golden present.
    _load_golden_cache().ensure_decompressed()
    for xz_path in xz_files:
        target = xz_path.with_suffix("")
        if target.suffix in COMPRESSED_SUFFIXES:
            assert target.exists(), f"{target.name} was not materialised"


def test_compressed_golden_footprint_stays_small() -> None:
    """Tripwire against golden re-bloat past the lightweight-repo budget."""

    total = sum(p.stat().st_size for p in REF_DIR.glob("*.xz"))
    assert total <= MAX_COMPRESSED_GOLDEN_BYTES, (
        f"compressed goldens total {total / 1e6:.1f} MB exceeds the "
        f"{MAX_COMPRESSED_GOLDEN_BYTES / 1e6:.1f} MB budget"
    )
