#!/usr/bin/env python
"""Index selected benchmark JSON artifacts for release gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sfincs_jax.validation.artifacts import (
    ARTIFACT_CLASSES,
    BenchmarkArtifactIndex,
    BenchmarkArtifactIndexEntry,
    index_benchmark_artifact_files,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify selected JSON artifacts as schema-v2 compliant, historical "
            "legacy, Fortran suite summary, unrelated non-PAS, or release-blocking."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSON artifact file(s) or directories to scan recursively.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable index instead of CI text output.",
    )
    return parser


def expand_json_paths(paths: list[Path]) -> list[Path]:
    """Expand selected files/directories to a stable list of JSON files."""

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
            )
        else:
            expanded.append(path)
    return expanded


def _entry_to_json(entry: BenchmarkArtifactIndexEntry) -> dict[str, object]:
    return {
        "path": str(entry.path),
        "classification": entry.classification,
        "release_blocking": entry.release_blocking,
        "errors": list(entry.errors),
    }


def _index_to_json(index: BenchmarkArtifactIndex) -> dict[str, object]:
    return {
        "summary": index.counts,
        "release_blocking": len(index.release_blocking),
        "artifacts": [_entry_to_json(entry) for entry in index.entries],
    }


def _print_text_index(index: BenchmarkArtifactIndex) -> None:
    for entry in index.entries:
        print(f"{entry.path}: {entry.classification}")
        for error in entry.errors:
            print(f"  {error}", file=sys.stderr)

    counts = index.counts
    summary = ", ".join(
        f"{artifact_class}={counts[artifact_class]}" for artifact_class in ARTIFACT_CLASSES
    )
    print(f"summary: total={len(index.entries)}, {summary}")

    if index.release_blocking:
        print(f"release gate: fail ({len(index.release_blocking)} blocking artifact(s))", file=sys.stderr)
    else:
        print("release gate: pass")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = expand_json_paths(args.paths)
    index = index_benchmark_artifact_files(paths)

    if args.json:
        print(json.dumps(_index_to_json(index), indent=2, sort_keys=True))
    else:
        _print_text_index(index)

    return 1 if index.release_blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
