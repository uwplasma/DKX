#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseKeyCoverage:
    case: str
    fortran_h5: str | None
    jax_h5: str | None
    missing_in_jax: list[str]
    extra_in_jax: list[str]
    skipped: bool = False
    skip_reason: str | None = None


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


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
