#!/usr/bin/env python
"""Validate release-facing claim metadata without running expensive solves."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

from sfincs_jax.validation.qi_device import check_qi_device_artifact_files


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "publication_figures" / "validation_manifest.json"
DEFAULT_DOCS = (REPO_ROOT / "docs" / "validation_matrix.rst",)
DEFAULT_QI_DEVICE_ARTIFACT_PATHS = (REPO_ROOT / "docs" / "_static",)

DEFERRED_STATUS = "deferred_post_release"
VALID_RECORD_STATUSES = {"implemented", DEFERRED_STATUS}
VALID_RECORD_KINDS = {
    "literature_reproduction",
    "literature_validation",
    "profile_validation",
    "cross_code_validation",
    "autodiff_validation",
}
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

REQUIRED_DOC_PHRASES = (
    "Release claim gate metadata",
    "release_ready",
    "regression_scaffold",
    "bounded_proxy",
    "closed_deferred",
)
REQUIRED_RECORD_LIST_FIELDS = (
    "literature",
    "claims",
    "source_code",
    "tests",
    "acceptance_gates",
)


def _load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path}: manifest must be a list")
    return payload


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    rows = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return rows if len(rows) == len(value) else None


def _first_command_path(command: str) -> str:
    return command.split()[0]


def release_gate_errors(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    docs_paths: Iterable[Path] = DEFAULT_DOCS,
    qi_device_paths: Iterable[Path] = DEFAULT_QI_DEVICE_ARTIFACT_PATHS,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Return release-gate metadata errors for CI-fast checks."""

    errors: list[str] = []
    records = _load_manifest(manifest_path)
    for index, record in enumerate(records):
        record_id = str(record.get("id", f"<record {index}>"))
        if not _nonempty_string(record.get("id")):
            errors.append(f"<record {index}>: id must be a non-empty string")

        status = str(record.get("status", ""))
        if status not in VALID_RECORD_STATUSES:
            errors.append(
                f"{record_id}: status must be one of {sorted(VALID_RECORD_STATUSES)}, got {status!r}"
            )
        kind = str(record.get("kind", ""))
        if kind not in VALID_RECORD_KINDS:
            errors.append(
                f"{record_id}: kind must be one of {sorted(VALID_RECORD_KINDS)}, got {kind!r}"
            )

        for key in REQUIRED_RECORD_LIST_FIELDS:
            values = _string_list(record.get(key))
            if values is None or not values:
                errors.append(f"{record_id}: field {key} must be a non-empty list of strings")

        for key in ("source_code", "tests", "artifacts"):
            values = _string_list(record.get(key))
            if values is None:
                continue
            for value in values:
                if not (repo_root / value).exists():
                    errors.append(f"{record_id}: {key} path does not exist: {value}")

        scripts = _string_list(record.get("scripts"))
        if scripts is None and "scripts" in record:
            errors.append(f"{record_id}: field scripts must be a list of strings")
        elif scripts is not None:
            for command in scripts:
                path = _first_command_path(command)
                if path and not (repo_root / path).exists():
                    errors.append(f"{record_id}: script path does not exist: {path}")

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

    qi_paths = [
        candidate
        for path in qi_device_paths
        for candidate in (sorted(path.rglob("*.json")) if path.is_dir() else [path])
    ]
    qi_checks = check_qi_device_artifact_files(qi_paths)
    relevant_qi_checks = [check for check in qi_checks if check.relevant]
    if not relevant_qi_checks:
        errors.append("QI device artifact release gate found no relevant artifacts")
    for check in relevant_qi_checks:
        errors.extend(check.errors)

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
