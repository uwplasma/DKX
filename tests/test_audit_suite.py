from __future__ import annotations

import json
from pathlib import Path

import h5py

from tools.release import release


def test_audit_suite_runtime_drift_flags_only_cases_above_threshold(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            [
                {"case": "slow_case", "jax_runtime_s": 2.0},
                {"case": "tiny_case", "jax_runtime_s": 0.2},
                {"case": "stable_case", "jax_runtime_s": 5.0},
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                {"case": "slow_case", "jax_runtime_s": 3.0},
                {"case": "tiny_case", "jax_runtime_s": 0.5},
                {"case": "stable_case", "jax_runtime_s": 5.4},
            ]
        ),
        encoding="utf-8",
    )

    flagged = release.audit_suite_runtime_drift(
        baseline_report=baseline,
        candidate_report=candidate,
        threshold_ratio=1.25,
        min_baseline_runtime_s=1.0,
    )
    assert [(item.case, round(item.ratio, 2)) for item in flagged] == [("slow_case", 1.5)]


def test_audit_suite_runtime_drift_cli_can_fail(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps([{"case": "case", "jax_runtime_s": 1.0}]), encoding="utf-8")
    candidate.write_text(json.dumps([{"case": "case", "jax_runtime_s": 1.4}]), encoding="utf-8")
    rc = release.audit_runtime_drift_main(
        [
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--threshold-ratio",
            "1.25",
            "--fail-on-drift",
        ]
    )
    assert rc == 1


def test_audit_suite_runtime_drift_prefers_logged_elapsed_when_available(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps([{"case": "case", "jax_runtime_s": 10.0, "jax_logged_elapsed_s": 8.0}]),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps([{"case": "case", "jax_runtime_s": 11.0, "jax_logged_elapsed_s": 8.5}]),
        encoding="utf-8",
    )

    flagged = release.audit_suite_runtime_drift(
        baseline_report=baseline,
        candidate_report=candidate,
        threshold_ratio=1.10,
        min_baseline_runtime_s=1.0,
    )

    assert flagged == []


def _write_h5(path: Path, keys: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in keys.items():
            h5[key] = float(value)


def test_audit_suite_output_keys_reports_missing_and_extra(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    report_path = suite_root / "suite_report.json"

    fortran_a = suite_root / "case_a" / "fortran.h5"
    jax_a = suite_root / "case_a" / "jax.h5"
    fortran_b = suite_root / "case_b" / "fortran.h5"
    jax_b = suite_root / "case_b" / "jax.h5"
    _write_h5(fortran_a, {"a": 1.0, "b": 2.0})
    _write_h5(jax_a, {"a": 1.0})
    _write_h5(fortran_b, {"c": 3.0})
    _write_h5(jax_b, {"c": 3.0, "d": 4.0})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            [
                {"case": "case_a", "fortran_h5": str(fortran_a), "jax_h5": str(jax_a)},
                {"case": "case_b", "fortran_h5": str(fortran_b), "jax_h5": str(jax_b)},
            ]
        ),
        encoding="utf-8",
    )

    coverage = release.audit_suite_output_keys(suite_root=suite_root)
    by_case = {item.case: item for item in coverage}
    assert by_case["case_a"].missing_in_jax == ["b"]
    assert by_case["case_a"].extra_in_jax == []
    assert by_case["case_b"].missing_in_jax == []
    assert by_case["case_b"].extra_in_jax == ["d"]


def test_audit_suite_output_keys_cli_can_fail_on_missing(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    report_path = suite_root / "suite_report.json"
    fortran_h5 = suite_root / "case" / "fortran.h5"
    jax_h5 = suite_root / "case" / "jax.h5"
    _write_h5(fortran_h5, {"a": 1.0, "b": 2.0})
    _write_h5(jax_h5, {"a": 1.0})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([{"case": "case", "fortran_h5": str(fortran_h5), "jax_h5": str(jax_h5)}]),
        encoding="utf-8",
    )

    rc = release.audit_output_keys_main(["--suite-root", str(suite_root), "--fail-on-missing"])
    assert rc == 1


def test_audit_suite_output_keys_skips_cases_without_h5_paths(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    report_path = suite_root / "suite_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            [
                {"case": "missing_paths", "fortran_h5": None, "jax_h5": None},
            ]
        ),
        encoding="utf-8",
    )

    coverage = release.audit_suite_output_keys(suite_root=suite_root)
    assert len(coverage) == 1
    assert coverage[0].case == "missing_paths"
    assert coverage[0].skipped is True
    assert coverage[0].skip_reason == "missing_h5_path"
