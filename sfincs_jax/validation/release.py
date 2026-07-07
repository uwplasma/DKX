"""Release, suite-audit, and bounded profiling helpers for validation gates.

This module owns release commands that are useful from CI, docs, and local
audits. Keeping them together makes the behavior importable from tests and
avoids a collection of small root-level scripts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

import h5py
import jax
from jax import profiler as jax_profiler

from sfincs_jax.io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5
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




# Production benchmark input generation.
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "production_resolution_inputs_2026-05-04"
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "examples" / "sfincs_examples"
DEFAULT_ADDITIONAL_INPUT = REPO_ROOT / "examples" / "data" / "qi_nfp2_reference.input.namelist"
# Archived downstream NTX decks can still be imported explicitly for local
# reproduction, but the production benchmark tier is SFINCS_JAX-owned by
# default and should not depend on another repository.
DEFAULT_ARCHIVED_NTX_INPUTS = (
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_profile_current_audit/"
        "finite_beta_qa_pressure_current/rho_0p142857/nu_n_0p00831565/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "sfincs_jax_rhsmode1_profile_current_profiling/cpu_17x21x12_deck/"
        "finite_beta_qa_pressure_current/rho_0p142857/nu_n_0p00831565/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs/finite_beta_qa_pressure_current/"
        "rho_0p5/nuPrime_0p01/EStar_0/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs/finite_beta_nfp3_qh_stage1/"
        "rho_0p5/nuPrime_0p01/EStar_0/input.namelist"
    ),
    Path(
        "/Users/rogeriojorge/local/NTX/examples/outputs/"
        "owned_finite_beta_sfincs_jax_inputs_production_probe/grid_35_43_48/"
        "finite_beta_qa_pressure_current/rho_0p142857/nuPrime_0p01/EStar_0/input.namelist"
    ),
)

RESOLUTION_KEYS = ("NTHETA", "NZETA", "NX", "NXI")
DEFAULT_3D_MINIMUM = {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100}
DEFAULT_TOKAMAK_MINIMUM = {"NTHETA": 33, "NX": 12, "NXI": 140}
DEFAULT_TOKAMAK_PAS_NOER_MINIMUM = {"NTHETA": 89, "NX": 24, "NXI": 300}
DEFAULT_TARGET_FORTRAN_MIN_RUNTIME_S = 10.0


def _read_resolution(text: str) -> dict[str, int]:
    resolution: dict[str, int] = {}
    for key in RESOLUTION_KEYS:
        match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
        if match is None:
            continue
        token = match.group(1).replace("D", "E").replace("d", "e")
        try:
            resolution[key] = int(float(token))
        except ValueError:
            continue
    return resolution


def _read_int_parameter(text: str, key: str, default: int | None = None) -> int | None:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return default
    token = match.group(1).replace("D", "E").replace("d", "e")
    try:
        return int(float(token))
    except ValueError:
        return default


def _read_float_parameter(text: str, key: str, default: float | None = None) -> float | None:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return default
    token = match.group(1).replace("D", "E").replace("d", "e")
    try:
        return float(token)
    except ValueError:
        return default


def _read_logical_parameter(text: str, key: str, default: bool = False) -> bool:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*(\.[TtFf][A-Za-z]*\.)", text)
    if match is None:
        return bool(default)
    return match.group(1).lower().startswith(".t")


def _count_parameter_values(text: str, key: str, default: int = 1) -> int:
    match = re.search(rf"(?im)^\s*{key}\s*=\s*([^!\n/]+)", text)
    if match is None:
        return int(default)
    tokens = [token for token in re.split(r"[,\s]+", match.group(1).strip()) if token]
    return max(1, len(tokens))


def _replace_or_append_resolution(text: str, updates: dict[str, int]) -> str:
    updated = text
    missing: dict[str, int] = {}
    for key, value in updates.items():
        pattern = re.compile(rf"(?im)^(\s*{key}\s*=\s*)([-+0-9.eEdD]+)(.*)$")
        if pattern.search(updated):
            updated = pattern.sub(rf"\g<1>{int(value)}\3", updated, count=1)
        else:
            missing[key] = int(value)
    if not missing:
        return updated
    block = "\n&resolutionParameters\n" + "".join(f"  {key} = {value}\n" for key, value in missing.items()) + "/\n"
    return updated.rstrip() + block


def _replace_or_append_namelist_parameter(text: str, *, group: str, key: str, value: str) -> str:
    """Replace a scalar namelist parameter or append it to the requested group."""
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n/]+)(.*)$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}\3", text, count=1)

    group_pattern = re.compile(rf"(?ims)(^\s*&{re.escape(group)}\b.*?)(^\s*/\s*$)")
    match = group_pattern.search(text)
    if match is not None:
        return text[: match.start(2)] + f"  {key} = {value}\n" + text[match.start(2) :]

    return text.rstrip() + f"\n\n&{group}\n  {key} = {value}\n/\n"


def _apply_benchmark_solver_hints(text: str) -> str:
    """Apply non-physics solver hints needed for robust production benchmarks.

    The public benchmark floor lifts several old small tokamak FP decks far above
    their authored resolution. SFINCS v3 itself warns that ``preconditioner_x = 1``
    is usually the best option; without it, the full ``Nxi=100`` no-Er FP deck can
    run to the PETSc iteration ceiling and abort before a reference comparison is
    possible. This changes only the linear-solver preconditioner, not the physics
    model or requested outputs.
    """

    rhs_mode = int(_read_int_parameter(text, "RHSMode", 1) or 1)
    collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
    if rhs_mode == 1 and collision_operator == 0:
        text = _replace_or_append_namelist_parameter(
            text,
            group="preconditionerOptions",
            key="preconditioner_x",
            value="1",
        )
        text = _replace_or_append_namelist_parameter(
            text,
            group="otherNumericalParameters",
            key="whichParallelSolverToFactorPreconditioner",
            value="1",
        )
        resolution = _read_resolution(text)
        phase_points = _phase_points(resolution) or 0
        tokamak_like = int(resolution.get("NZETA", 1)) == 1
        include_phi1 = _read_logical_parameter(text, "includePhi1", False)
        if tokamak_like and (not include_phi1) and int(phase_points) <= 30_000:
            text = _replace_or_append_namelist_parameter(
                text,
                group="otherNumericalParameters",
                key="useIterativeLinearSolver",
                value=".false.",
            )
    return text


def _normalize_generated_text(text: str) -> str:
    """Return deterministic text for checked-in benchmark input artifacts."""
    return "\n".join(line.rstrip() for line in str(text).splitlines()).rstrip() + "\n"


def _normalize_text_file_if_possible(path: Path) -> None:
    if path.suffix.lower() in {".h5", ".hdf5", ".nc", ".npz", ".npy"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    path.write_text(_normalize_generated_text(text), encoding="utf-8")


def _safe_case_name(prefix: str, input_path: Path) -> str:
    parts = [part for part in input_path.resolve().parts if part not in {"", "/"}]
    tail = "_".join(parts[-6:-1])
    raw = f"{prefix}_{tail}" if tail else f"{prefix}_{input_path.parent.name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _case_kind(case: str, resolution: dict[str, int]) -> str:
    n_zeta = int(resolution.get("NZETA", 1))
    if "tokamak" in case.lower() or n_zeta == 1:
        return "tokamak"
    return "3d"


def _benchmark_resolution(
    resolution: dict[str, int],
    *,
    kind: str,
    text: str,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
) -> dict[str, int]:
    updated = dict(resolution)
    minimum = min_tokamak if kind == "tokamak" else min_3d
    if kind == "tokamak":
        rhs_mode = int(_read_int_parameter(text, "RHSMode", 1) or 1)
        collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
        if rhs_mode == 1 and collision_operator == 1 and not _effective_xdot_requested(text):
            minimum = {
                key: max(int(minimum.get(key, 0)), int(value))
                for key, value in min_tokamak_pas_noer.items()
            }
    for key, value in minimum.items():
        if key in updated:
            updated[key] = max(int(updated[key]), int(value))
        else:
            updated[key] = int(value)
    return updated


def _copy_parent_files(src_input: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_input.parent.iterdir():
        if item.is_file():
            target = dst_dir / item.name
            if item.resolve() == target.resolve():
                continue
            shutil.copy2(item, target)
            _normalize_text_file_if_possible(target)


def _copy_referenced_absolute_files(text: str, dst_dir: Path) -> None:
    """Copy absolute file references in an input deck when they exist locally."""
    for match in re.finditer(r"['\"](?P<path>/[^'\"]+?)['\"]", text):
        path = Path(match.group("path"))
        if not path.is_file():
            continue
        target = dst_dir / path.name
        if path.resolve() == target.resolve():
            continue
        shutil.copy2(path, target)
        _normalize_text_file_if_possible(target)


def _localize_copied_absolute_paths(text: str, dst_dir: Path) -> str:
    """Replace absolute file paths with copied sibling basenames when possible."""

    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote")
        raw = match.group("path")
        path = Path(raw)
        if not path.is_absolute():
            return match.group(0)
        sibling = dst_dir / path.name
        if not sibling.exists():
            return match.group(0)
        return f"{quote}{path.name}{quote}"

    return re.sub(
        r"(?P<quote>['\"])(?P<path>/[^'\"]+?)(?P=quote)",
        repl,
        text,
    )


def _phase_points(resolution: dict[str, int]) -> int | None:
    try:
        return int(resolution["NTHETA"]) * int(resolution.get("NZETA", 1)) * int(resolution["NX"]) * int(resolution["NXI"])
    except KeyError:
        return None


def _average_legendre_bandwidth(n_xi: int, *, radius: int = 2) -> float:
    n_xi = max(1, int(n_xi))
    total = 0
    for ell in range(n_xi):
        total += min(n_xi, ell + int(radius) + 1) - max(0, ell - int(radius))
    return float(total) / float(n_xi)


def _effective_xdot_requested(text: str) -> bool:
    """Return whether `includeXDotTerm` implies a nonzero electric-field x-coupling."""
    if not _read_logical_parameter(text, "includeXDotTerm", False):
        return False
    for key in ("dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN", "Er"):
        value = _read_float_parameter(text, key, None)
        if value is not None and abs(float(value)) > 1.0e-15:
            return True
    return False


def _estimate_case_size(text: str, resolution: dict[str, int]) -> dict[str, object]:
    """Return a conservative matrix-size estimate for benchmark scheduling.

    This is a preflight estimate only. It intentionally over-approximates the
    sparse pattern in the same direction as the sparse-host builder so that
    benchmark launchers can avoid unsafe dense or sparse-direct runs.
    """

    n_theta = int(resolution["NTHETA"])
    n_zeta = int(resolution.get("NZETA", 1))
    n_x = int(resolution["NX"])
    n_xi = int(resolution["NXI"])
    n_species = _count_parameter_values(text, "Zs")
    collision_operator = int(_read_int_parameter(text, "collisionOperator", 0) or 0)
    include_xdot_requested = _read_logical_parameter(text, "includeXDotTerm", False)
    include_xdot = _effective_xdot_requested(text)
    include_phi1 = _read_logical_parameter(text, "includePhi1", False)
    constraint_scheme = _read_int_parameter(text, "constraintScheme", None)
    if constraint_scheme is None:
        constraint_scheme = 1 if collision_operator == 0 else 2

    f_unknowns = int(n_species * n_x * n_xi * n_theta * n_zeta)
    phi1_unknowns = int(n_theta * n_zeta) if include_phi1 else 0
    if constraint_scheme == 2:
        extra_unknowns = int(n_species * n_x)
    elif constraint_scheme in {1, 3, 4}:
        extra_unknowns = int(2 * n_species)
    else:
        extra_unknowns = 0
    total_unknowns = int(f_unknowns + phi1_unknowns + extra_unknowns)

    spatial_stencil = int(n_theta + n_zeta - 1)
    if collision_operator == 0:
        avg_row_nnz = float(n_species * n_x * n_xi * spatial_stencil)
    else:
        x_band = n_x if include_xdot else 1
        avg_row_nnz = float(x_band * _average_legendre_bandwidth(n_xi) * spatial_stencil)
    conservative_nnz = int(f_unknowns * avg_row_nnz + extra_unknowns * n_theta * n_zeta)
    dense_nbytes = int(total_unknowns * total_unknowns * 8)
    csr_nbytes = int(conservative_nnz * (8 + 4) + (total_unknowns + 1) * 4)

    run_recommendation = "bounded_remote"
    if total_unknowns < 100_000 and csr_nbytes < 2_000_000_000:
        run_recommendation = "bounded_local_ok"
    elif csr_nbytes > 8_000_000_000 or total_unknowns > 250_000:
        run_recommendation = "remote_or_cluster_only"

    return {
        "species_count": int(n_species),
        "collision_operator": int(collision_operator),
        "include_xdot_requested": bool(include_xdot_requested),
        "include_xdot": bool(include_xdot),
        "include_xdot_effective": bool(include_xdot),
        "include_phi1": bool(include_phi1),
        "constraint_scheme": int(constraint_scheme),
        "f_unknowns_estimate": int(f_unknowns),
        "phi1_unknowns_estimate": int(phi1_unknowns),
        "extra_unknowns_estimate": int(extra_unknowns),
        "total_unknowns_estimate": int(total_unknowns),
        "dense_matrix_nbytes_estimate": int(dense_nbytes),
        "conservative_sparse_nnz_estimate": int(conservative_nnz),
        "conservative_csr_nbytes_estimate": int(csr_nbytes),
        "run_recommendation": run_recommendation,
    }


def _resolution_policy_text(
    *,
    enforce_minimum: bool,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
    target_fortran_min_runtime_s: float,
) -> str:
    if not enforce_minimum:
        return "preserve authored external resolution"
    min_3d_text = (
        f"{int(min_3d['NTHETA'])}x{int(min_3d['NZETA'])}x"
        f"{int(min_3d['NX'])}x{int(min_3d['NXI'])}"
    )
    min_tokamak_text = (
        f"{int(min_tokamak['NTHETA'])}x1x"
        f"{int(min_tokamak['NX'])}x{int(min_tokamak['NXI'])}"
    )
    min_tokamak_pas_noer_text = (
        f"{int(min_tokamak_pas_noer['NTHETA'])}x1x"
        f"{int(min_tokamak_pas_noer['NX'])}x{int(min_tokamak_pas_noer['NXI'])}"
    )
    return (
        f"preserve nominal grid, but enforce 3D >= {min_3d_text} "
        f"and tokamak >= {min_tokamak_text} "
        f"(RHSMode=1 PAS/no-Er tokamak >= {min_tokamak_pas_noer_text}); "
        f"production timing rows target "
        f"Fortran v3 >= {float(target_fortran_min_runtime_s):g} s"
    )


def _case_name_with_benchmark_resolution(case: str, resolution: dict[str, int]) -> str:
    try:
        token = (
            f"{int(resolution['NTHETA'])}x{int(resolution['NZETA'])}x"
            f"{int(resolution['NX'])}x{int(resolution['NXI'])}"
        )
    except KeyError:
        return case
    return re.sub(r"(?i)cpu_\d+x\d+x\d+(?:x\d+)?_deck", f"cpu_{token}_deck", case)


def _iter_example_inputs(examples_root: Path, additional_input: Path | None) -> Iterable[tuple[str, Path]]:
    for path in sorted(Path(examples_root).glob("*/input.namelist")):
        yield path.parent.name, path
    if additional_input is not None and Path(additional_input).exists():
        yield "additional_examples", Path(additional_input)


def _build_entry(
    *,
    case: str,
    src_input: Path,
    dst_root: Path,
    source_group: str,
    min_3d: dict[str, int],
    min_tokamak: dict[str, int],
    min_tokamak_pas_noer: dict[str, int],
    enforce_minimum: bool,
    target_fortran_min_runtime_s: float,
) -> dict[str, object] | None:
    src_input = Path(src_input).resolve()
    if not src_input.exists():
        return None
    text = src_input.read_text(encoding="utf-8", errors="replace")
    original = _read_resolution(text)
    kind = _case_kind(case, original)
    if source_group == "examples" and kind not in {"tokamak", "3d"}:
        return None
    benchmark = (
        _benchmark_resolution(
            original,
            kind=kind,
            text=text,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
        )
        if enforce_minimum
        else dict(original)
    )
    case = _case_name_with_benchmark_resolution(case, benchmark)
    dst_dir = dst_root / "inputs" / case
    _copy_parent_files(src_input, dst_dir)
    _copy_referenced_absolute_files(text, dst_dir)
    dst_input = dst_dir / "input.namelist"
    text = _localize_copied_absolute_paths(text, dst_dir)
    generated_text = _replace_or_append_resolution(text, benchmark)
    generated_text = _apply_benchmark_solver_hints(generated_text)
    dst_input.write_text(_normalize_generated_text(generated_text), encoding="utf-8")
    size_estimate = _estimate_case_size(text, benchmark)
    return {
        "case": case,
        "source_group": source_group,
        "kind": kind,
        "source_input": str(src_input),
        "input": str(dst_input.relative_to(dst_root)),
        "original_resolution": original,
        "benchmark_resolution": benchmark,
        "phase_points": _phase_points(benchmark),
        "size_estimate": size_estimate,
        "resolution_policy": _resolution_policy_text(
            enforce_minimum=enforce_minimum,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            target_fortran_min_runtime_s=float(target_fortran_min_runtime_s),
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", type=Path, default=DEFAULT_EXAMPLES_ROOT)
    parser.add_argument("--additional-input", type=Path, default=DEFAULT_ADDITIONAL_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--external-input",
        type=Path,
        action="append",
        default=[],
        help="Additional production-resolution input deck to copy into the manifest.",
    )
    parser.add_argument(
        "--include-archived-ntx-defaults",
        action="store_true",
        help=(
            "Compatibility/debug option: also import archived downstream NTX decks "
            "when they exist locally. Not used by the default SFINCS_JAX benchmark tier."
        ),
    )
    parser.add_argument(
        "--include-ntx-defaults",
        dest="include_archived_ntx_defaults",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ntx-input",
        dest="external_input",
        type=Path,
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enforce-minimum-on-external",
        action="store_true",
        default=True,
        help="Also lift external inputs to the production minimum instead of preserving authored grids.",
    )
    parser.add_argument(
        "--enforce-minimum-on-ntx",
        dest="enforce_minimum_on_external",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preserve-external-resolution",
        dest="enforce_minimum_on_external",
        action="store_false",
        help="Compatibility/debug mode: keep authored external grids instead of lifting them to the production minimum.",
    )
    parser.add_argument(
        "--preserve-ntx-resolution",
        dest="enforce_minimum_on_external",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-3d-ntheta", type=int, default=DEFAULT_3D_MINIMUM["NTHETA"])
    parser.add_argument("--min-3d-nzeta", type=int, default=DEFAULT_3D_MINIMUM["NZETA"])
    parser.add_argument("--min-3d-nx", type=int, default=DEFAULT_3D_MINIMUM["NX"])
    parser.add_argument("--min-3d-nxi", type=int, default=DEFAULT_3D_MINIMUM["NXI"])
    parser.add_argument("--min-tokamak-ntheta", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NTHETA"])
    parser.add_argument("--min-tokamak-nx", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NX"])
    parser.add_argument("--min-tokamak-nxi", type=int, default=DEFAULT_TOKAMAK_MINIMUM["NXI"])
    parser.add_argument(
        "--min-tokamak-pas-noer-ntheta",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NTHETA"],
    )
    parser.add_argument(
        "--min-tokamak-pas-noer-nx",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NX"],
    )
    parser.add_argument(
        "--min-tokamak-pas-noer-nxi",
        type=int,
        default=DEFAULT_TOKAMAK_PAS_NOER_MINIMUM["NXI"],
    )
    parser.add_argument(
        "--target-fortran-min-runtime-s",
        type=float,
        default=DEFAULT_TARGET_FORTRAN_MIN_RUNTIME_S,
        help=(
            "Documented Fortran v3 lower runtime target for production benchmark rows. "
            "The fixed resolution floor is chosen to make public timing cases nontrivial; "
            "benchmark runners should still report the measured per-case runtime."
        ),
    )
    parser.add_argument("--clean", action="store_true")
    return parser


def production_inputs_main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    min_3d = {
        "NTHETA": int(args.min_3d_ntheta),
        "NZETA": int(args.min_3d_nzeta),
        "NX": int(args.min_3d_nx),
        "NXI": int(args.min_3d_nxi),
    }
    min_tokamak = {
        "NTHETA": int(args.min_tokamak_ntheta),
        "NX": int(args.min_tokamak_nx),
        "NXI": int(args.min_tokamak_nxi),
    }
    min_tokamak_pas_noer = {
        "NTHETA": int(args.min_tokamak_pas_noer_ntheta),
        "NX": int(args.min_tokamak_pas_noer_nx),
        "NXI": int(args.min_tokamak_pas_noer_nxi),
    }
    target_fortran_min_runtime_s = float(args.target_fortran_min_runtime_s)
    entries: list[dict[str, object]] = []
    for case, src_input in _iter_example_inputs(Path(args.examples_root), Path(args.additional_input)):
        entry = _build_entry(
            case=case,
            src_input=src_input,
            dst_root=out_root,
            source_group="examples",
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            enforce_minimum=True,
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        )
        if entry is not None:
            entries.append(entry)
    external_inputs = list(args.external_input or [])
    if args.include_archived_ntx_defaults:
        external_inputs.extend(path for path in DEFAULT_ARCHIVED_NTX_INPUTS if path.exists())
    for src_input in external_inputs:
        case = _safe_case_name("external", Path(src_input))
        entry = _build_entry(
            case=case,
            src_input=Path(src_input),
            dst_root=out_root,
            source_group="external",
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            enforce_minimum=bool(args.enforce_minimum_on_external),
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        )
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda item: (str(item["source_group"]), int(item["phase_points"] or 0), str(item["case"])))
    manifest = {
        "schema_version": 1,
        "minimum_3d_resolution": min_3d,
        "minimum_tokamak_resolution": min_tokamak,
        "minimum_tokamak_pas_noer_resolution": min_tokamak_pas_noer,
        "target_fortran_min_runtime_s": target_fortran_min_runtime_s,
        "resolution_policy": _resolution_policy_text(
            enforce_minimum=True,
            min_3d=min_3d,
            min_tokamak=min_tokamak,
            min_tokamak_pas_noer=min_tokamak_pas_noer,
            target_fortran_min_runtime_s=target_fortran_min_runtime_s,
        ),
        "case_count": len(entries),
        "cases": entries,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Cases: {len(entries)}")
    if entries:
        largest = sorted(entries, key=lambda item: int(item["phase_points"] or 0), reverse=True)[:8]
        print("Largest cases by Ntheta*Nzeta*Nx*Nxi:")
        for item in largest:
            print(f"- {item['case']}: {item['benchmark_resolution']} phase_points={item['phase_points']}")
    return 0




# README runtime\/memory audit regeneration.
README = REPO_ROOT / "README.md"
README_AUDIT_DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak"
README_AUDIT_DEFAULT_GPU_OUT_ROOT = REPO_ROOT / "tests" / "scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas"
README_AUDIT_BASELINE_REPORT = (
    REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak" / "suite_report.json"
)
README_AUDIT_EXAMPLES_ROOT = REPO_ROOT / "examples" / "sfincs_examples"
README_AUDIT_EXTRA_INPUT = REPO_ROOT / "examples" / "data" / "qi_nfp2_reference.input.namelist"
README_AUDIT_EXTRA_CASE_NAME = "additional_examples"
DEFAULT_PUBLIC_MIN_FORTRAN_RUNTIME_S = 10.0

BEGIN = "<!-- BEGIN EXAMPLE_SUITE_AUDIT -->"
END = "<!-- END EXAMPLE_SUITE_AUDIT -->"
LEGACY_BEGIN = "<!-- BEGIN FAST_BRANCH_AUDIT -->"
LEGACY_END = "<!-- END FAST_BRANCH_AUDIT -->"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _load_optional_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return _load_json(path)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def _case_names_for_inputs(inputs: list[Path], *, base_root: Path | None = None) -> dict[Path, str]:
    parent_counts: dict[str, int] = {}
    for input_path in inputs:
        parent_counts[input_path.parent.name] = parent_counts.get(input_path.parent.name, 0) + 1

    names: dict[Path, str] = {}
    for input_path in inputs:
        try:
            if input_path.resolve() == README_AUDIT_EXTRA_INPUT.resolve():
                names[input_path] = README_AUDIT_EXTRA_CASE_NAME
                continue
        except Exception:  # noqa: BLE001
            pass
        parent_name = input_path.parent.name
        if parent_counts[parent_name] == 1:
            names[input_path] = parent_name
            continue
        if base_root is not None:
            try:
                rel = input_path.parent.resolve().relative_to(base_root.resolve())
                names[input_path] = "__".join(rel.parts)
                continue
            except Exception:  # noqa: BLE001
                pass
        names[input_path] = "__".join(input_path.parent.parts[-2:])
    return names


def _expected_cases() -> list[str]:
    inputs = sorted(README_AUDIT_EXAMPLES_ROOT.rglob("input.namelist"))
    if README_AUDIT_EXTRA_INPUT.exists():
        inputs.append(README_AUDIT_EXTRA_INPUT)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    case_names = _case_names_for_inputs(deduped, base_root=README_AUDIT_EXAMPLES_ROOT.parent)
    return sorted(case_names.values())


def _fmt_float(value: object | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_runtime_ratio(row: dict[str, object]) -> str:
    jax = row.get("jax_runtime_s")
    fort = row.get("fortran_runtime_s")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _warm_or_logged_runtime(row: dict[str, object] | None) -> object | None:
    if row is None:
        return None
    warm = row.get("jax_runtime_s_warm")
    if warm is not None:
        return warm
    return row.get("jax_logged_elapsed_s")


def _fmt_memory_ratio(row: dict[str, object]) -> str:
    jax = _jax_public_memory(row)
    fort = row.get("fortran_max_rss_mb")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _jax_public_memory(row: dict[str, object] | None) -> object | None:
    if row is None:
        return None
    active = row.get("jax_incremental_max_rss_mb")
    if active is not None:
        return active
    return row.get("jax_max_rss_mb")


def _fmt_ratio(jax: object | None, fort: object | None) -> str:
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _fmt_status(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    status = str(row.get("status", "")).strip()
    return status or "-"


def _fmt_mismatch_pair(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    return (
        f"{int(row.get('n_mismatch_common', 0))}/{int(row.get('n_common_keys', 0))}"
        f" (strict {int(row.get('strict_n_mismatch_common', 0))}/{int(row.get('strict_n_common_keys', 0))})"
    )


def _fmt_print_parity(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    total = int(row.get("print_parity_total", 0))
    if total <= 0:
        return "-"
    return f"{int(row.get('print_parity_signals', 0))}/{total}"


def _reference_fortran_runtime(
    case: str,
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
) -> float | None:
    row = rows_by_case.get(case) or gpu_rows_by_case.get(case)
    if row is None:
        return None
    value = row.get("fortran_runtime_s")
    if value is None:
        return None
    return float(value)


def _public_comparison_cases(
    case_order: list[str],
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
    *,
    min_fortran_runtime_s: float,
) -> tuple[list[str], list[dict[str, object]]]:
    included: list[str] = []
    excluded: list[dict[str, object]] = []
    threshold = float(min_fortran_runtime_s)
    for case in case_order:
        runtime = _reference_fortran_runtime(case, rows_by_case, gpu_rows_by_case)
        if runtime is not None and runtime >= threshold:
            included.append(case)
        else:
            excluded.append({"case": case, "fortran_runtime_s": runtime})
    return included, excluded


def _format_excluded_public_cases(excluded_cases: list[dict[str, object]]) -> str:
    if not excluded_cases:
        return "none"
    return ", ".join(
        f"`{row['case']}` ({_fmt_float(row.get('fortran_runtime_s'), 3)}s)" for row in excluded_cases
    )


def _status_counts(rows: list[dict[str, object]], prefix: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(prefix, row.get("status", "unknown")))] += 1
    return counts


def _strict_status_counts(rows: list[dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        strict_mismatch = int(row.get("strict_n_mismatch_common", 0))
        strict_common = int(row.get("strict_n_common_keys", 0))
        strict_status = "parity_ok"
        if strict_common > 0 and strict_mismatch > 0:
            strict_status = "parity_mismatch"
        elif row.get("status") not in {"parity_ok", "parity_mismatch"}:
            strict_status = str(row.get("status"))
        counts[str(strict_status)] += 1
    return counts


def _top_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get(key) is not None]
    return sorted(valid, key=lambda row: float(row[key]), reverse=True)[:limit]


def _format_row_summary(row: dict[str, object], *, metric_key: str, digits: int = 1) -> str:
    value = _fmt_float(row.get(metric_key), digits)
    fort_key = "fortran_runtime_s" if metric_key == "jax_runtime_s" else "fortran_max_rss_mb"
    ratio = _fmt_runtime_ratio(row) if metric_key == "jax_runtime_s" else _fmt_memory_ratio(row)
    final_resolution = row.get("final_resolution")
    res_str = f", res={final_resolution}" if final_resolution else ""
    return (
        f"- `{row['case']}`: jax={value}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"fortran={_fmt_float(row.get(fort_key), digits)}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"ratio={ratio} status={row.get('status', '-')}{res_str}"
    )


def _format_mismatch(row: dict[str, object]) -> str:
    return (
        f"- `{row['case']}`: status={row.get('status', '-')}, "
        f"practical={row.get('n_mismatch_common', 0)}/{row.get('n_common_keys', 0)}, "
        f"strict={row.get('strict_n_mismatch_common', 0)}/{row.get('strict_n_common_keys', 0)}, "
        f"sample={','.join(row.get('mismatch_keys_sample', [])[:4]) or '-'}"
    )


def _format_case_table(
    case_order: list[str],
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
) -> list[str]:
    lines = [
        "| Case | Fortran CPU(s) | JAX CPU cold(s) | CPU cold x | JAX CPU warm/logged(s) | CPU warm/logged x | JAX GPU cold(s) | GPU cold x | JAX GPU warm/logged(s) | GPU warm/logged x | Fortran MB | JAX CPU active MB | CPU MB x | JAX GPU active MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for case in case_order:
        cpu_row = rows_by_case.get(case)
        gpu_row = gpu_rows_by_case.get(case)
        if cpu_row is None and gpu_row is None:
            continue
        reference_row = cpu_row or gpu_row
        fort_runtime = reference_row.get("fortran_runtime_s") if reference_row else None
        fort_memory = reference_row.get("fortran_max_rss_mb") if reference_row else None
        cpu_runtime = cpu_row.get("jax_runtime_s") if cpu_row else None
        gpu_runtime = gpu_row.get("jax_runtime_s") if gpu_row else None
        cpu_warm_runtime = _warm_or_logged_runtime(cpu_row)
        gpu_warm_runtime = _warm_or_logged_runtime(gpu_row)
        cpu_memory = _jax_public_memory(cpu_row)
        gpu_memory = _jax_public_memory(gpu_row)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case}`",
                    _fmt_float(fort_runtime, 3),
                    _fmt_float(cpu_runtime, 3),
                    _fmt_ratio(cpu_runtime, fort_runtime),
                    _fmt_float(cpu_warm_runtime, 3),
                    _fmt_ratio(cpu_warm_runtime, fort_runtime),
                    _fmt_float(gpu_runtime, 3),
                    _fmt_ratio(gpu_runtime, fort_runtime),
                    _fmt_float(gpu_warm_runtime, 3),
                    _fmt_ratio(gpu_warm_runtime, fort_runtime),
                    _fmt_float(fort_memory, 1),
                    _fmt_float(cpu_memory, 1),
                    _fmt_ratio(cpu_memory, fort_memory),
                    _fmt_float(gpu_memory, 1),
                    _fmt_ratio(gpu_memory, fort_memory),
                    _fmt_mismatch_pair(cpu_row),
                    _fmt_mismatch_pair(gpu_row),
                    _fmt_print_parity(cpu_row),
                    _fmt_print_parity(gpu_row),
                    _fmt_status(cpu_row),
                    _fmt_status(gpu_row),
                ]
            )
            + " |"
        )
    return lines


def _format_improvement(
    current_row: dict[str, object],
    baseline_row: dict[str, object],
    *,
    metric_key: str,
    digits: int = 1,
) -> str:
    if metric_key == "jax_incremental_max_rss_mb":
        current_value = _jax_public_memory(current_row)
        baseline_value = _jax_public_memory(baseline_row)
    else:
        current_value = current_row[metric_key]
        baseline_value = baseline_row[metric_key]
    current = float(current_value)
    baseline = float(baseline_value)
    delta = baseline - current
    unit = "s" if metric_key == "jax_runtime_s" else " MB"
    return (
        f"- `{current_row['case']}`: "
        f"{_fmt_float(baseline, digits)}{unit} -> {_fmt_float(current, digits)}{unit} "
        f"(delta={_fmt_float(delta, digits)}{unit})"
    )


def _format_runtime_drift_summary(prefix: str, summary: dict[str, object]) -> str:
    """Format only same-resolution runtime drift as a gate."""
    status = str(summary.get("status", "")).strip().lower()
    if status in {"not_applicable", "skipped"}:
        reason = str(summary.get("reason", "")).strip()
        for stale_reason in (
            "production-floor reruns are not same-resolution with the older frozen smoke baseline",
            "production-floor reruns are not same-resolution with the frozen smoke baseline",
        ):
            reason = reason.replace(
                stale_reason,
                "suite rows are not same-resolution with the optional runtime baseline",
            )
        suffix = f": {reason}" if reason else ""
        return f"- {prefix} runtime drift gate: not applicable{suffix}"

    flagged = int(summary.get("flagged_cases", 0))
    threshold = summary.get("threshold_ratio", "-")
    baseline = summary.get("baseline_report", "-")
    cases = [str(case) for case in summary.get("cases", [])]
    if flagged > 0:
        return (
            f"- {prefix} runtime drift gate vs `{baseline}`: "
            f"`{flagged}` cases above `{threshold}x` "
            f"({', '.join(cases[:4])})"
        )
    return f"- {prefix} runtime drift gate vs `{baseline}`: none"


def readme_audit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update README full example-suite audit block.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=README_AUDIT_DEFAULT_OUT_ROOT,
        help="Suite output root containing suite_report.json and run_manifest.json.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=README_AUDIT_BASELINE_REPORT,
        help="Optional baseline report used for improvement summaries.",
    )
    parser.add_argument(
        "--gpu-out-root",
        type=Path,
        default=README_AUDIT_DEFAULT_GPU_OUT_ROOT,
        help="Optional GPU suite output root containing suite_report.json.",
    )
    parser.add_argument(
        "--min-fortran-runtime-s",
        type=float,
        default=DEFAULT_PUBLIC_MIN_FORTRAN_RUNTIME_S,
        help="Minimum Fortran v3 runtime for public runtime/memory comparison rows.",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    report_path = out_root / "suite_report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")
    manifest_path = out_root / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    gpu_out_root = Path(args.gpu_out_root)
    gpu_report_path = gpu_out_root / "suite_report.json"
    gpu_rows = list(_load_json(gpu_report_path)) if gpu_report_path.exists() else []
    cpu_key_summary = _load_optional_json(out_root / "suite_output_key_coverage_summary.json")
    gpu_key_summary = _load_optional_json(gpu_out_root / "suite_output_key_coverage_summary.json")
    cpu_runtime_drift_summary = _load_optional_json(out_root / "suite_runtime_drift_summary.json")
    gpu_runtime_drift_summary = _load_optional_json(gpu_out_root / "suite_runtime_drift_summary.json")

    rows = list(_load_json(report_path))
    case_order = _expected_cases()
    total_cases = len(case_order)
    rows_by_case = {str(row["case"]): row for row in rows}
    missing_cases = [case for case in case_order if case not in rows_by_case]

    strict_counts = _strict_status_counts(rows)
    gpu_status_counts = Counter(str(row.get("status", "unknown")) for row in gpu_rows)
    gpu_strict_counts = _strict_status_counts(gpu_rows)

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    mismatches = [row for row in rows if str(row.get("status")) != "parity_ok"]
    gpu_mismatches = [row for row in gpu_rows if str(row.get("status")) != "parity_ok"]

    improvements_runtime: list[str] = []
    improvements_memory: list[str] = []
    baseline_report = Path(args.baseline_report)
    if baseline_report.exists():
        baseline_rows = {str(row["case"]): row for row in _load_json(baseline_report)}
        paired_runtime = []
        paired_memory = []
        for case, row in rows_by_case.items():
            reference_runtime = _reference_fortran_runtime(case, rows_by_case, {})
            if reference_runtime is None or reference_runtime < float(args.min_fortran_runtime_s):
                continue
            base = baseline_rows.get(case)
            if not base:
                continue
            if row.get("jax_runtime_s") is not None and base.get("jax_runtime_s") is not None:
                paired_runtime.append((float(base["jax_runtime_s"]) - float(row["jax_runtime_s"]), row, base))
            if row.get("jax_max_rss_mb") is not None and base.get("jax_max_rss_mb") is not None:
                paired_memory.append((float(base["jax_max_rss_mb"]) - float(row["jax_max_rss_mb"]), row, base))
        for _delta, row, base in sorted(paired_runtime, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_runtime.append(_format_improvement(row, base, metric_key="jax_runtime_s", digits=1))
        for _delta, row, base in sorted(paired_memory, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_memory.append(_format_improvement(row, base, metric_key="jax_max_rss_mb", digits=1))

    lines = [
        BEGIN,
        f"CPU audit source: `{_repo_rel(out_root)}`.",
        (
            f"GPU audit source: `{_repo_rel(gpu_out_root)}`."
            if gpu_rows
            else "GPU audit source: not available."
        ),
        "",
        f"- Recorded cases: `{len(rows)}/{total_cases}`",
        f"- Practical status counts: `{', '.join(f'{k}={status_counts[k]}' for k in sorted(status_counts))}`",
        f"- Strict status counts: `{', '.join(f'{k}={strict_counts[k]}' for k in sorted(strict_counts))}`",
    ]
    if gpu_rows:
        lines.append(
            f"- GPU practical status counts: `{', '.join(f'{k}={gpu_status_counts[k]}' for k in sorted(gpu_status_counts))}`"
        )
        lines.append(
            f"- GPU strict status counts: `{', '.join(f'{k}={gpu_strict_counts[k]}' for k in sorted(gpu_strict_counts))}`"
        )
    if cpu_key_summary is not None:
        lines.append(
            "- CPU output-key coverage: "
            f"`missing_total={cpu_key_summary.get('missing_total') or 0}, "
            f"extra_total={cpu_key_summary.get('extra_total') or '-'}, "
            f"audited_cases={cpu_key_summary.get('audited_cases') or '-'}, "
            f"skipped_cases={cpu_key_summary.get('skipped_cases') or 0}`"
        )
    if gpu_key_summary is not None:
        lines.append(
            "- GPU output-key coverage: "
            f"`missing_total={gpu_key_summary.get('missing_total') or 0}, "
            f"extra_total={gpu_key_summary.get('extra_total') or '-'}, "
            f"audited_cases={gpu_key_summary.get('audited_cases') or '-'}, "
            f"skipped_cases={gpu_key_summary.get('skipped_cases') or 0}`"
        )
    if cpu_runtime_drift_summary is not None:
        lines.append(_format_runtime_drift_summary("CPU", cpu_runtime_drift_summary))
    if gpu_runtime_drift_summary is not None:
        lines.append(_format_runtime_drift_summary("GPU", gpu_runtime_drift_summary))
    if manifest:
        resolution_policy = manifest.get("resolution_policy")
        scale_factor = manifest.get("scale_factor")
        runtime_basis = manifest.get("runtime_target_basis")
        runtime_floor = manifest.get("fortran_min_runtime_s")
        runtime_cap = manifest.get("fortran_max_runtime_s")
        adjust_iters = manifest.get("runtime_adjustment_iters")
        lines.append(
            "- Resolution policy: "
            f"`{resolution_policy}, scale_factor={scale_factor}, runtime_basis={runtime_basis}, "
            f"fortran_min={runtime_floor}, fortran_max={runtime_cap}, adjust_iters={adjust_iters}`"
        )
    if missing_cases:
        lines.append(f"- Remaining cases: `{', '.join(missing_cases)}`")
    else:
        lines.append("- Remaining cases: none")
    cpu_additional = rows_by_case.get("additional_examples")
    gpu_rows_by_case = {str(row["case"]): row for row in gpu_rows}
    public_case_order, excluded_public_cases = _public_comparison_cases(
        case_order,
        rows_by_case,
        gpu_rows_by_case,
        min_fortran_runtime_s=float(args.min_fortran_runtime_s),
    )
    gpu_additional = gpu_rows_by_case.get("additional_examples")
    if cpu_additional is not None and gpu_additional is not None:
        lines.append(
            f"- Additional example: `{cpu_additional.get('status', '-')}` on CPU and `{gpu_additional.get('status', '-')}` on GPU"
        )

    gpu_rows_by_case = {str(row["case"]): row for row in gpu_rows}

    if mismatches:
        lines.extend(
            [
                "",
                "Mismatches:",
                *[_format_mismatch(row) for row in mismatches],
            ]
        )
    elif gpu_rows:
        lines.extend(
            [
                "",
                "Mismatches:",
                "- CPU practical mismatches: none",
                (
                    "- CPU strict-only survivor: "
                    f"`{next(row['case'] for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}` "
                    f"(`{next(int(row.get('strict_n_mismatch_common', 0)) for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}/"
                    f"{next(int(row.get('strict_n_common_keys', 0)) for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}`)"
                    if any(int(row.get("strict_n_mismatch_common", 0)) > 0 for row in rows)
                    else "- CPU strict mismatches: none"
                ),
                "- GPU practical/strict mismatches: none" if not gpu_mismatches else _format_mismatch(gpu_mismatches[0]),
            ]
        )

    lines.extend(
        [
            "",
            "Runtime columns match the summary plot: cold is `jax_runtime_s`; warm/logged is "
            "`jax_runtime_s_warm` when available, otherwise `jax_logged_elapsed_s`. "
            "The JAX memory columns match the plot and use profiler active RSS deltas "
            "(`jax_incremental_max_rss_mb`) when present; full process peak RSS remains "
            "available as `jax_max_rss_mb` in the merged JSON reports.",
            "The benchmark summary JSON records production-resolution floor violations for "
            "frozen reference rows, so the table is a reference-runtime-window comparison "
            "unless a row is also marked as satisfying the production-resolution floor.",
            (
                f"The public runtime/memory table is restricted to cases where the "
                f"SFINCS Fortran v3 reference runtime is at least `{float(args.min_fortran_runtime_s):g} s`. "
                f"Excluded lower-resolution CI parity/smoke rows: "
                f"{_format_excluded_public_cases(excluded_public_cases)}."
            ),
            "",
            "Full per-case runtime / memory table:",
            *_format_case_table(public_case_order, rows_by_case, gpu_rows_by_case),
        ]
    )

    if improvements_runtime:
        lines.extend(
            [
                "",
                f"Largest CPU runtime improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_runtime,
            ]
        )
    if improvements_memory:
        lines.extend(
            [
                "",
                f"Largest CPU process peak-RSS improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_memory,
            ]
        )

    lines.append(END)

    readme = README.read_text()
    begin = BEGIN if BEGIN in readme else LEGACY_BEGIN
    end = END if END in readme else LEGACY_END
    if begin not in readme or end not in readme:
        raise SystemExit("README example-suite audit markers not found.")
    prefix, rest = readme.split(begin, 1)
    _old, suffix = rest.split(end, 1)
    README.write_text(prefix + "\n".join(lines) + suffix)
    print("Updated README full example-suite audit block.")
    return 0




# One-case write-output tracing and phase logging.
def _prepare_input(input_path: Path, *, equilibrium_file: str | None = None, wout_path: str | None = None) -> tuple[Path, Path]:
    tmpdir = Path(tempfile.mkdtemp(prefix="sfincs_jax_write_output_trace_"))
    dst = tmpdir / "input.namelist"
    shutil.copy2(input_path, dst)
    if equilibrium_file or wout_path:
        return dst, tmpdir
    old_search = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS")
    search_dirs = [str(input_path.parent.resolve())]
    if old_search:
        search_dirs.append(old_search)
    os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = os.pathsep.join(search_dirs)
    try:
        localize_equilibrium_file_in_place(input_namelist=dst, overwrite=False)
    finally:
        if old_search is None:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)
        else:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = old_search
    return dst, tmpdir


def _run_write_output(
    *,
    input_path: Path,
    output_path: Path,
    compute_solution: bool,
    compute_transport_matrix: bool,
    equilibrium_file: str | None,
    wout_path: str | None,
    solver_trace_path: Path | None = None,
    differentiable: bool = False,
) -> None:
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=output_path,
        compute_solution=bool(compute_solution),
        compute_transport_matrix=bool(compute_transport_matrix),
        equilibrium_file=equilibrium_file,
        wout_path=wout_path,
        overwrite=True,
        verbose=True,
        solver_trace_path=solver_trace_path,
        differentiable=bool(differentiable),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON sidecar atomically so timeout/debug runs keep usable state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def write_output_trace_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a full sfincs_jax write-output run with JAX trace capture. "
            "Use --warmup 0 to include compile/lowering time, or --warmup 1 to focus on steady-state kernels."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input.namelist.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory for the JAX/XProf trace output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sfincsOutput_profiled.h5"),
        help="Output H5 written by the profiled run.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of untraced warmup runs.")
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Also emit perfetto_trace.json.gz for upload to ui.perfetto.dev.",
    )
    parser.add_argument(
        "--no-jax-trace",
        action="store_true",
        help=(
            "Skip the JAX/XProf trace context while keeping the phase log, "
            "output solve, and optional device-memory snapshot. Use this for "
            "long low-overhead production audits."
        ),
    )
    parser.add_argument(
        "--device-memory-profile",
        type=Path,
        default=None,
        help="Optional pprof-format device-memory snapshot written after the traced run.",
    )
    parser.add_argument(
        "--phase-log",
        type=Path,
        default=None,
        help=(
            "Optional JSON sidecar for timeout-safe phase timings. Defaults to "
            "<trace-dir>/profile_write_output_trace_phases.json."
        ),
    )
    parser.add_argument(
        "--phase-log-interval-s",
        type=float,
        default=10.0,
        help="Heartbeat interval for refreshing the phase log while a long solve is running; use 0 to disable.",
    )
    parser.add_argument(
        "--solver-trace",
        type=Path,
        default=None,
        help="Optional JSON solver-trace sidecar written by write_sfincs_jax_output_h5.",
    )
    parser.add_argument(
        "--differentiable",
        action="store_true",
        help=(
            "Profile the differentiable implicit-solve path. By default this wrapper "
            "matches the CLI write-output fast path and uses differentiable=False."
        ),
    )
    parser.add_argument(
        "--strict-profiler",
        action="store_true",
        help=(
            "Return a nonzero status if profiler finalization or device-memory "
            "snapshotting fails after the solve. By default, a completed output "
            "file is treated as solve success and profiler teardown failures are "
            "recorded in the phase log."
        ),
    )
    parser.add_argument(
        "--compute-solution",
        action="store_true",
        help="Force solution arrays into the output file.",
    )
    parser.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Force transport-matrix arrays into the output file.",
    )
    parser.add_argument(
        "--equilibrium-file",
        default=None,
        help="Optional equilibrium file override, matching the CLI.",
    )
    parser.add_argument(
        "--wout-path",
        default=None,
        help="Compatibility alias for --equilibrium-file.",
    )
    args = parser.parse_args(argv)

    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    phase_log = (
        args.phase_log.resolve()
        if args.phase_log is not None
        else trace_dir / "profile_write_output_trace_phases.json"
    )
    overall_t0 = time.perf_counter()
    phase_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "input": str(args.input.resolve()),
        "trace_dir": str(trace_dir),
        "output": str(args.out.resolve()),
        "warmup": int(args.warmup),
        "perfetto": bool(args.perfetto),
        "jax_trace": not bool(args.no_jax_trace),
        "compute_solution": bool(args.compute_solution),
        "compute_transport_matrix": bool(args.compute_transport_matrix),
        "differentiable": bool(args.differentiable),
        "phase_log_interval_s": float(args.phase_log_interval_s),
        "solver_trace": str(args.solver_trace.resolve()) if args.solver_trace is not None else None,
        "device_memory_profile": str(args.device_memory_profile.resolve())
        if args.device_memory_profile is not None
        else None,
        "phases": [],
    }
    phase_log_lock = threading.RLock()

    def _flush(status: str | None = None) -> None:
        with phase_log_lock:
            if status is not None:
                phase_payload["status"] = status
            phase_payload["elapsed_s"] = time.perf_counter() - overall_t0
            _atomic_write_json(phase_log, phase_payload)

    def _start_phase(name: str, **extra: Any) -> dict[str, Any]:
        phase = {
            "name": name,
            "status": "running",
            "started_s": time.perf_counter() - overall_t0,
            **extra,
        }
        with phase_log_lock:
            phase_payload["phases"].append(phase)
        _flush("running")
        return phase

    def _finish_phase(phase: dict[str, Any], status: str = "ok", **extra: Any) -> None:
        with phase_log_lock:
            if phase.get("status") != "running":
                return
            phase["status"] = status
            phase["elapsed_s"] = time.perf_counter() - overall_t0 - float(phase.get("started_s", 0.0))
            phase.update(extra)
        _flush()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    cache_dir = (
        Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
        if os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
        else (trace_dir / ".jax_compilation_cache").resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    output_path = args.out.resolve()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    phase_payload["output"] = str(output_path)
    _flush("initializing")

    phase = _start_phase("prepare_input")
    try:
        work_input, tmpdir = _prepare_input(
            args.input.resolve(),
            equilibrium_file=args.equilibrium_file,
            wout_path=args.wout_path,
        )
    except Exception as exc:  # noqa: BLE001
        _finish_phase(phase, "failed", exception=_exception_summary(exc))
        _flush("failed")
        print(f"Input preparation failed: {_exception_summary(exc)}", flush=True)
        return 1
    _finish_phase(phase, localized_input=str(work_input), localized_dir=str(tmpdir))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for warmup_index in range(max(0, int(args.warmup))):
        warmup_out = output_path.with_name(output_path.stem + ".warmup.h5")
        phase = _start_phase("warmup", index=warmup_index, output=str(warmup_out))
        try:
            _run_write_output(
                input_path=work_input,
                output_path=warmup_out,
                compute_solution=bool(args.compute_solution),
                compute_transport_matrix=bool(args.compute_transport_matrix),
                equilibrium_file=args.equilibrium_file,
                wout_path=args.wout_path,
                solver_trace_path=None,
                differentiable=bool(args.differentiable),
            )
            if warmup_out.exists():
                warmup_out.unlink()
        except Exception as exc:  # noqa: BLE001
            _finish_phase(phase, "failed", exception=_exception_summary(exc))
            _flush("failed")
            print(f"Warmup {warmup_index} failed: {_exception_summary(exc)}", flush=True)
            return 1
        _finish_phase(phase)

    try:
        heartbeat_interval_s = max(0.0, float(args.phase_log_interval_s))
    except (TypeError, ValueError):
        heartbeat_interval_s = 10.0
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(heartbeat_interval_s):
            _flush()

    heartbeat_thread: threading.Thread | None = None
    if heartbeat_interval_s > 0.0:
        heartbeat_thread = threading.Thread(target=_heartbeat, name="sfincs_jax_phase_log_heartbeat", daemon=True)
        heartbeat_thread.start()

    trace_phase = _start_phase(
        "jax_trace",
        perfetto=bool(args.perfetto),
        enabled=not bool(args.no_jax_trace),
    )
    solve_phase: dict[str, Any] | None = None
    block_phase: dict[str, Any] | None = None
    solve_completed = False
    profiler_error: BaseException | None = None
    try:
        if args.no_jax_trace:
            solve_phase = _start_phase("write_output_solve", output=str(output_path))
            _run_write_output(
                input_path=work_input,
                output_path=output_path,
                compute_solution=bool(args.compute_solution),
                compute_transport_matrix=bool(args.compute_transport_matrix),
                equilibrium_file=args.equilibrium_file,
                wout_path=args.wout_path,
                solver_trace_path=args.solver_trace.resolve() if args.solver_trace is not None else None,
                differentiable=bool(args.differentiable),
            )
            solve_completed = True
            _finish_phase(solve_phase, output_exists=output_path.exists())
            block_phase = _start_phase("block_until_ready")
            jax.block_until_ready(0)
            _finish_phase(block_phase)
        else:
            with jax_profiler.trace(
                str(trace_dir),
                create_perfetto_trace=bool(args.perfetto),
            ):
                solve_phase = _start_phase("write_output_solve", output=str(output_path))
                _run_write_output(
                    input_path=work_input,
                    output_path=output_path,
                    compute_solution=bool(args.compute_solution),
                    compute_transport_matrix=bool(args.compute_transport_matrix),
                    equilibrium_file=args.equilibrium_file,
                    wout_path=args.wout_path,
                    solver_trace_path=args.solver_trace.resolve() if args.solver_trace is not None else None,
                    differentiable=bool(args.differentiable),
                )
                solve_completed = True
                _finish_phase(solve_phase, output_exists=output_path.exists())
                block_phase = _start_phase("block_until_ready")
                jax.block_until_ready(0)
                _finish_phase(block_phase)
    except Exception as exc:  # noqa: BLE001
        profiler_error = exc
        if solve_phase is not None and solve_phase.get("status") == "running":
            _finish_phase(solve_phase, "failed", exception=_exception_summary(exc))
        if block_phase is not None and block_phase.get("status") == "running":
            _finish_phase(block_phase, "failed", exception=_exception_summary(exc))
        _finish_phase(trace_phase, "failed", exception=_exception_summary(exc))
    else:
        _finish_phase(trace_phase)
    heartbeat_stop.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)

    device_memory_error: BaseException | None = None
    if args.device_memory_profile is not None:
        mem_phase = _start_phase("device_memory_profile", output=str(args.device_memory_profile.resolve()))
        try:
            args.device_memory_profile.parent.mkdir(parents=True, exist_ok=True)
            jax_profiler.save_device_memory_profile(str(args.device_memory_profile))
        except Exception as exc:  # noqa: BLE001
            device_memory_error = exc
            _finish_phase(mem_phase, "failed", exception=_exception_summary(exc))
        else:
            _finish_phase(mem_phase)

    output_exists = output_path.exists()
    profiler_failed_after_solve = profiler_error is not None and solve_completed and output_exists
    device_profile_failed_after_solve = device_memory_error is not None and solve_completed and output_exists
    if profiler_error is not None and not profiler_failed_after_solve:
        _flush("failed")
        print(f"Trace failed before solve completion: {_exception_summary(profiler_error)}", flush=True)
        return 1
    if profiler_failed_after_solve and args.strict_profiler:
        _flush("failed")
        print(f"Profiler finalization failed: {_exception_summary(profiler_error)}", flush=True)
        return 1
    if device_memory_error is not None and args.strict_profiler:
        _flush("failed")
        print(f"Device-memory profiling failed: {_exception_summary(device_memory_error)}", flush=True)
        return 1

    status = "completed"
    if profiler_failed_after_solve or device_profile_failed_after_solve:
        status = "solve_completed_profile_incomplete"
    _flush(status)

    print(f"Wrote trace -> {trace_dir}")
    print(f"Wrote output -> {output_path}")
    print(f"Wrote phase log -> {phase_log}")
    print(f"Trace elapsed {float(trace_phase.get('elapsed_s', 0.0)):.3f}s")
    print(f"Localized input dir -> {tmpdir}")
    if profiler_failed_after_solve:
        print(
            "Profiler finalization failed after the output file was written; "
            f"recorded and returning solve success: {_exception_summary(profiler_error)}",
            flush=True,
        )
    if device_profile_failed_after_solve:
        print(
            "Device-memory profiling failed after the output file was written; "
            f"recorded and returning solve success: {_exception_summary(device_memory_error)}",
            flush=True,
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
    subparsers.add_parser("production-inputs", help="Generate production-resolution benchmark inputs.")
    subparsers.add_parser("readme-audit", help="Regenerate the README runtime/memory audit block.")
    subparsers.add_parser("write-output-trace", help="Profile a single write-output run with phase logs.")

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
    if command == "production-inputs":
        return production_inputs_main(rest)
    if command == "readme-audit":
        return readme_audit_main(rest)
    if command == "write-output-trace":
        return write_output_trace_main(rest)
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
