"""Small helpers shared across the dkx teaching examples.

These are deliberately minimal utilities factored out of several
``examples/`` scripts so the teaching scripts do not copy-paste the same
boilerplate.  ``examples/`` is not an installable package, so each example
puts the ``examples/`` directory on ``sys.path`` and imports by module name::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/
    from _example_utils import output_dir, print_dataset_summary
"""

from __future__ import annotations

import lzma
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def output_dir(script_file: str | Path, name: str | None = None) -> Path:
    """Return (and create) the standard output directory for an example.

    The layout is ``examples/output/<name>/`` where ``name`` defaults to the
    calling script's stem.  Pass ``__file__`` as ``script_file``.
    """

    script_path = Path(script_file).resolve()
    stem = name if name is not None else script_path.stem
    out = script_path.parents[1] / "output" / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def print_dataset_summary(
    data: Mapping[str, Any], keys: Iterable[str], *, indent: str = "  "
) -> None:
    """Print a few named datasets from a loaded ``sfincsOutput`` mapping."""

    for key in keys:
        print(f"{indent}{key} = {np.asarray(data[key])}")


def ensure_uncompressed(path: str | Path) -> Path:
    """Materialize ``path`` from a sibling ``<path>.xz`` if it is missing.

    The heavy ``tests/ref`` fixtures ship lzma-compressed; decompress on
    demand so an example runs from a fresh checkout.
    """

    path = Path(path)
    if not path.exists():
        compressed = path.with_name(path.name + ".xz")
        if compressed.exists():
            path.write_bytes(lzma.decompress(compressed.read_bytes()))
    return path
