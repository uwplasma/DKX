#!/usr/bin/env python
"""Validate release-facing claim metadata without running expensive solves."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "publication_figures" / "validation_manifest.json"
DEFAULT_DOCS = (REPO_ROOT / "docs" / "validation_matrix.rst",)

VALID_CLAIM_STATUSES = {
    "release_ready",
    "regression_scaffold",
    "bounded_proxy",
    "closed_deferred",
}
IMPLEMENTED_CLAIM_STATUSES = {
    "release_ready",
    "regression_scaffold",
    "bounded_proxy",
}
DEFERRED_STATUS = "deferred_post_release"

REQUIRED_DOC_PHRASES = (
    "Release claim gate metadata",
    "release_ready",
    "regression_scaffold",
    "bounded_proxy",
    "closed_deferred",
)


def _load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path}: manifest must be a list")
    return payload


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def release_gate_errors(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    docs_paths: Iterable[Path] = DEFAULT_DOCS,
) -> list[str]:
    """Return release-gate metadata errors for CI-fast checks."""

    errors: list[str] = []
    records = _load_manifest(manifest_path)
    for index, record in enumerate(records):
        record_id = str(record.get("id", f"<record {index}>"))
        status = str(record.get("status", ""))
        gate = record.get("release_gate")
        if not isinstance(gate, dict):
            errors.append(f"{record_id}: missing release_gate object")
            continue

        claim_status = str(gate.get("claim_status", ""))
        if claim_status not in VALID_CLAIM_STATUSES:
            errors.append(
                f"{record_id}: release_gate.claim_status must be one of "
                f"{sorted(VALID_CLAIM_STATUSES)}, got {claim_status!r}"
            )

        blocks_current_release = gate.get("blocks_current_release")
        if not isinstance(blocks_current_release, bool):
            errors.append(f"{record_id}: release_gate.blocks_current_release must be a bool")
        elif blocks_current_release:
            errors.append(f"{record_id}: no manifest lane may block the current release")

        for key in ("evidence", "promotion_gate"):
            if not _nonempty_string(gate.get(key)):
                errors.append(f"{record_id}: release_gate.{key} must be a non-empty string")

        if status == DEFERRED_STATUS:
            if claim_status != "closed_deferred":
                errors.append(
                    f"{record_id}: deferred lanes must use claim_status='closed_deferred'"
                )
            reason = str(gate.get("closed_or_deferred_reason", "")).lower()
            if "closed" not in reason or "post-release" not in reason:
                errors.append(
                    f"{record_id}: deferred lanes must record a closed post-release reason"
                )
        elif claim_status not in IMPLEMENTED_CLAIM_STATUSES:
            errors.append(
                f"{record_id}: implemented lanes must use an implemented claim status, "
                f"got {claim_status!r}"
            )

    for doc_path in docs_paths:
        text = doc_path.read_text(encoding="utf-8")
        for phrase in REQUIRED_DOC_PHRASES:
            if phrase not in text:
                errors.append(f"{doc_path}: missing release-gate docs phrase {phrase!r}")

    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate release claim metadata for validation/doc gates."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Validation manifest to check.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Do not require release-gate terminology in validation docs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    docs_paths: tuple[Path, ...] = () if args.skip_docs else DEFAULT_DOCS
    errors = release_gate_errors(args.manifest, docs_paths=docs_paths)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    records = _load_manifest(args.manifest)
    counts = Counter(str(record["release_gate"]["claim_status"]) for record in records)  # type: ignore[index]
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"{args.manifest}: release gates ok ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
