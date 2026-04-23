#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RuntimeDrift:
    case: str
    baseline_runtime_s: float
    candidate_runtime_s: float
    ratio: float


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


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
