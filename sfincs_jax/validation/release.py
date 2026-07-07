"""Release and suite-audit helpers for lightweight validation gates.

This module owns checks that are useful for releases but do not launch physics
solves. Keeping them in the package makes the behavior importable from tests
and avoids a collection of small root-level scripts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable

import h5py

from sfincs_jax.validation.artifacts import check_research_lane_completion_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_MANIFEST = REPO_ROOT / "examples" / "publication_figures" / "validation_manifest.json"
DEFAULT_VALIDATION_DOCS = (REPO_ROOT / "docs" / "validation_matrix.rst",)
DEFAULT_RESEARCH_MANIFEST = REPO_ROOT / "docs" / "_static" / "research_lane_completion_2026_05_12.json"
DEFAULT_SIZE_THRESHOLD_MIB = 2.0
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REVIEWED_LARGE_FILES: dict[str, str] = {}

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


@dataclass(frozen=True)
class CaseKeyCoverage:
    """Top-level HDF5 key comparison for one SFINCS/JAX suite case."""

    case: str
    fortran_h5: str | None
    jax_h5: str | None
    missing_in_jax: list[str]
    extra_in_jax: list[str]
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class RuntimeDrift:
    """Runtime ratio for a case whose candidate run exceeded a threshold."""

    case: str
    baseline_runtime_s: float
    candidate_runtime_s: float
    ratio: float


@dataclass(frozen=True)
class CompressionResult:
    """Image-compression outcome for one raster file."""

    path: Path
    before: int
    after: int
    changed: bool

    @property
    def saved(self) -> int:
        return max(0, self.before - self.after) if self.changed else 0


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
    manifest_path: Path = DEFAULT_VALIDATION_MANIFEST,
    *,
    docs_paths: Iterable[Path] = DEFAULT_VALIDATION_DOCS,
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

    return errors


def _resolve_report_path(path_str: str | None, *, suite_root: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    suite_rel = suite_root / path
    if suite_rel.exists():
        return suite_rel
    repo_rel = REPO_ROOT / path
    if repo_rel.exists():
        return repo_rel
    return path


def _top_level_h5_keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as h5:
        return set(h5.keys())


def audit_suite_output_keys(*, suite_root: Path) -> list[CaseKeyCoverage]:
    """Compare top-level Fortran and JAX HDF5 keys for a suite report."""

    report_path = Path(suite_root) / "suite_report.json"
    rows = json.loads(report_path.read_text(encoding="utf-8"))
    coverage: list[CaseKeyCoverage] = []
    for row in rows:
        fortran_h5 = _resolve_report_path(row.get("fortran_h5"), suite_root=Path(suite_root))
        jax_h5 = _resolve_report_path(row.get("jax_h5"), suite_root=Path(suite_root))
        skip_reason: str | None = None
        if fortran_h5 is None or jax_h5 is None:
            skip_reason = "missing_h5_path"
        elif not fortran_h5.exists() or not jax_h5.exists():
            skip_reason = "missing_h5_file"
        if skip_reason is not None:
            coverage.append(
                CaseKeyCoverage(
                    case=str(row["case"]),
                    fortran_h5=str(fortran_h5) if fortran_h5 is not None else None,
                    jax_h5=str(jax_h5) if jax_h5 is not None else None,
                    missing_in_jax=[],
                    extra_in_jax=[],
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
            continue
        fortran_keys = _top_level_h5_keys(fortran_h5)
        jax_keys = _top_level_h5_keys(jax_h5)
        coverage.append(
            CaseKeyCoverage(
                case=str(row["case"]),
                fortran_h5=str(fortran_h5),
                jax_h5=str(jax_h5),
                missing_in_jax=sorted(fortran_keys - jax_keys),
                extra_in_jax=sorted(jax_keys - fortran_keys),
            )
        )
    return coverage


def _load_rows(path: Path) -> dict[str, dict[str, object]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["case"]): row for row in rows}


def _preferred_runtime(row: dict[str, object]) -> float | None:
    logged = row.get("jax_logged_elapsed_s")
    if logged not in (None, 0):
        return float(logged)
    runtime = row.get("jax_runtime_s")
    if runtime in (None, 0):
        return None
    return float(runtime)


def audit_suite_runtime_drift(
    *,
    baseline_report: Path,
    candidate_report: Path,
    threshold_ratio: float = 1.25,
    min_baseline_runtime_s: float = 0.0,
) -> list[RuntimeDrift]:
    """Return suite cases whose runtime ratio exceeds ``threshold_ratio``."""

    baseline = _load_rows(Path(baseline_report))
    candidate = _load_rows(Path(candidate_report))
    flagged: list[RuntimeDrift] = []
    for case, base_row in baseline.items():
        cand_row = candidate.get(case)
        if cand_row is None:
            continue
        base_runtime = _preferred_runtime(base_row)
        cand_runtime = _preferred_runtime(cand_row)
        if base_runtime in (None, 0) or cand_runtime is None:
            continue
        base_runtime_f = float(base_runtime)
        cand_runtime_f = float(cand_runtime)
        if base_runtime_f < float(min_baseline_runtime_s):
            continue
        ratio = cand_runtime_f / base_runtime_f
        if ratio > float(threshold_ratio):
            flagged.append(
                RuntimeDrift(
                    case=case,
                    baseline_runtime_s=base_runtime_f,
                    candidate_runtime_s=cand_runtime_f,
                    ratio=ratio,
                )
            )
    flagged.sort(key=lambda item: item.ratio, reverse=True)
    return flagged


def _tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / item.decode() for item in raw.split(b"\0") if item]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def large_tracked_files(*, root: Path = REPO_ROOT, threshold_bytes: int) -> dict[str, int]:
    """Return tracked files larger than ``threshold_bytes``."""

    large: dict[str, int] = {}
    for path in _tracked_files(root):
        if not path.exists():
            continue
        size = path.stat().st_size
        if size > threshold_bytes:
            large[_relative(path, root)] = int(size)
    return large


def _iter_images(roots: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        candidates = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
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
    """Compress one raster image and replace it only when the payload shrinks."""

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


def check_release_gates_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate release claim metadata for validation/doc gates."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST,
        help="Validation manifest to check.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Do not require release-gate terminology in validation docs.",
    )
    args = parser.parse_args(argv)
    docs_paths: tuple[Path, ...] = () if args.skip_docs else DEFAULT_VALIDATION_DOCS
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


def check_research_lanes_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the machine-readable open-lane completion manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RESEARCH_MANIFEST,
        help="Research-lane completion JSON to validate.",
    )
    args = parser.parse_args(argv)
    errors = check_research_lane_completion_file(args.manifest, repo_root=REPO_ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    lanes = payload["lanes"]
    avg_current = sum(float(lane["current_percent"]) for lane in lanes) / len(lanes)
    avg_delta = sum(
        float(lane["current_percent"]) - float(lane["before_percent"]) for lane in lanes
    ) / len(lanes)
    print(
        f"{args.manifest}: research lanes ok "
        f"(lanes={len(lanes)}, avg_current={avg_current:.1f}%, avg_delta={avg_delta:.1f}pp)"
    )
    return 0


def audit_output_keys_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit top-level HDF5 key coverage for a suite_report.json root.")
    parser.add_argument("--suite-root", type=Path, required=True, help="Suite root containing suite_report.json.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON audit artifact.")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit nonzero if any Fortran top-level key is missing in the JAX HDF5 output.",
    )
    args = parser.parse_args(argv)

    coverage = audit_suite_output_keys(suite_root=Path(args.suite_root))
    total_missing = sum(len(item.missing_in_jax) for item in coverage)
    total_extra = sum(len(item.extra_in_jax) for item in coverage)
    skipped = [item for item in coverage if item.skipped]
    cases_with_missing = [item for item in coverage if item.missing_in_jax]
    cases_with_extra = [item for item in coverage if item.extra_in_jax]

    print(f"cases={len(coverage)} missing_total={total_missing} extra_total={total_extra} skipped={len(skipped)}")
    if skipped:
        print("skipped cases:")
        for item in skipped:
            print(f"  {item.case}: reason={item.skip_reason}")
    if cases_with_missing:
        print("cases with missing keys:")
        for item in cases_with_missing:
            print(f"  {item.case}: missing={len(item.missing_in_jax)} sample={item.missing_in_jax[:8]}")
    if cases_with_extra:
        print("cases with extra keys:")
        for item in cases_with_extra:
            print(f"  {item.case}: extra={len(item.extra_in_jax)} sample={item.extra_in_jax[:8]}")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps([asdict(item) for item in coverage], indent=2),
            encoding="utf-8",
        )

    if args.fail_on_missing and cases_with_missing:
        return 1
    return 0


def audit_runtime_drift_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit runtime drift between two suite_report.json files.")
    parser.add_argument("--baseline-report", type=Path, required=True, help="Baseline suite_report.json.")
    parser.add_argument("--candidate-report", type=Path, required=True, help="Candidate suite_report.json.")
    parser.add_argument("--threshold-ratio", type=float, default=1.25, help="Flag candidate/base ratios above this threshold.")
    parser.add_argument(
        "--min-baseline-runtime-s",
        type=float,
        default=0.0,
        help="Ignore baseline cases faster than this threshold to reduce noise on tiny runs.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON audit artifact.")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit nonzero if any drift case exceeds the threshold.")
    args = parser.parse_args(argv)

    flagged = audit_suite_runtime_drift(
        baseline_report=args.baseline_report,
        candidate_report=args.candidate_report,
        threshold_ratio=float(args.threshold_ratio),
        min_baseline_runtime_s=float(args.min_baseline_runtime_s),
    )
    print(f"flagged_cases={len(flagged)} threshold_ratio={args.threshold_ratio} min_baseline_runtime_s={args.min_baseline_runtime_s}")
    for item in flagged:
        print(
            f"  {item.case}: baseline={item.baseline_runtime_s:.3f}s "
            f"candidate={item.candidate_runtime_s:.3f}s ratio={item.ratio:.2f}x"
        )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps([asdict(item) for item in flagged], indent=2), encoding="utf-8")

    if args.fail_on_drift and flagged:
        return 1
    return 0


def check_size_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if large tracked files have not been explicitly reviewed."
    )
    parser.add_argument(
        "--threshold-mib",
        type=float,
        default=DEFAULT_SIZE_THRESHOLD_MIB,
        help="Tracked-file review threshold in MiB.",
    )
    args = parser.parse_args(argv)

    threshold_bytes = int(float(args.threshold_mib) * 1024 * 1024)
    large = large_tracked_files(root=REPO_ROOT, threshold_bytes=threshold_bytes)
    missing = sorted(path for path in large if path not in REVIEWED_LARGE_FILES)
    stale = sorted(path for path in REVIEWED_LARGE_FILES if path not in large)

    if missing or stale:
        print("Repository size audit failed.", file=sys.stderr)
        if missing:
            print("\nTracked files above threshold without review:", file=sys.stderr)
            for path in missing:
                print(f"  {large[path] / 1024 / 1024:7.2f} MiB  {path}", file=sys.stderr)
        if stale:
            print("\nReviewed-large-file entries that no longer exist or are below threshold:", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
        return 1

    print(f"Repository size audit passed: {len(large)} reviewed files above {args.threshold_mib:g} MiB.")
    for path in sorted(large):
        print(f"  {large[path] / 1024 / 1024:7.2f} MiB  {path} - {REVIEWED_LARGE_FILES[path]}")
    return 0


def compress_images_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compress documentation raster images when a smaller encoding is available."
    )
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

    results = [
        compress_image(path, apply=args.apply, jpeg_quality=args.jpeg_quality)
        for path in _iter_images(args.roots)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight SFINCS_JAX release validation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-gates", help="Validate validation-manifest release gates.")
    subparsers.add_parser("check-research-lanes", help="Validate research-lane completion metadata.")
    subparsers.add_parser("audit-output-keys", help="Audit Fortran/JAX HDF5 output key coverage.")
    subparsers.add_parser("audit-runtime-drift", help="Audit runtime drift between suite reports.")
    subparsers.add_parser("check-size", help="Audit tracked files above the reviewed size threshold.")
    subparsers.add_parser("compress-images", help="Compress documentation raster images.")

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 2
    command = argv[0]
    rest = argv[1:]
    if command == "check-gates":
        return check_release_gates_main(rest)
    if command == "check-research-lanes":
        return check_research_lanes_main(rest)
    if command == "audit-output-keys":
        return audit_output_keys_main(rest)
    if command == "audit-runtime-drift":
        return audit_runtime_drift_main(rest)
    if command == "check-size":
        return check_size_main(rest)
    if command == "compress-images":
        return compress_images_main(rest)
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
