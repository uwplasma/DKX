from __future__ import annotations

import json
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_FOCUSED_ROOT = _REPO / "tests" / "gpu_solver_path_cases_2026-04-28"
_GPU_SUITE_ROOT = _REPO / "tests" / "scaled_example_suite_gpu_bounded_default_2026-04-28"


def _rows(name: str) -> dict[str, dict[str, object]]:
    payload = json.loads((_FOCUSED_ROOT / name).read_text())
    return {str(row["variant"]): row for row in payload}


def test_focused_gpu_full_fp_dense_default_avoids_krylov_cliff() -> None:
    for filename in (
        "tokamak_fp_ntheta13_nxi20_gpu_bounded_default.json",
        "tokamak_fp_ntheta13_nxi40_gpu_bounded_default.json",
    ):
        rows = _rows(filename)
        default = rows["default"]
        forced_krylov = rows["forced_krylov"]

        assert default["vs_fortran"]["count"] == 0
        assert forced_krylov["vs_fortran"]["count"] == 0
        assert default["solver_path"]["used_dense_auto"] is True
        assert forced_krylov["solver_path"]["used_dense_auto"] is False
        assert float(default["elapsed_s"]) < 0.5 * float(forced_krylov["elapsed_s"])
        assert float(default["ru_maxrss_mb"]) < float(forced_krylov["ru_maxrss_mb"])


def test_bounded_gpu_suite_artifact_remains_release_clean() -> None:
    rows = json.loads((_GPU_SUITE_ROOT / "suite_report.json").read_text())
    strict_rows = json.loads((_GPU_SUITE_ROOT / "suite_report_strict.json").read_text())
    drift = json.loads((_GPU_SUITE_ROOT / "suite_runtime_drift_summary.json").read_text())
    key_coverage = json.loads((_GPU_SUITE_ROOT / "suite_output_key_coverage_summary.json").read_text())

    assert len(rows) == 39
    assert len(strict_rows) == 39
    assert {row["status"] for row in rows} == {"parity_ok"}
    assert {row["status"] for row in strict_rows} == {"parity_ok"}
    assert drift["flagged_cases"] == 0
    assert key_coverage["missing_total"] == 0


def test_bounded_gpu_pas_geometry11_defaults_to_pas_tz() -> None:
    audit_rows = json.loads((_GPU_SUITE_ROOT / "solver_path_audit.json").read_text())
    report_rows = json.loads((_GPU_SUITE_ROOT / "suite_report.json").read_text())
    audits = {str(row["case"]): row for row in audit_rows}
    reports = {str(row["case"]): row for row in report_rows}

    expected_pas_tz = {
        "HSX_PASCollisions_fullTrajectories": (9.0, 1700.0),
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories": (7.0, 1700.0),
    }
    for case, (max_logged_s, max_rss_mb) in expected_pas_tz.items():
        assert audits[case]["last_preconditioner"] == "pas_tz"
        assert reports[case]["n_mismatch_common"] == 0
        assert reports[case]["strict_n_mismatch_common"] == 0
        assert float(reports[case]["jax_logged_elapsed_s"]) < max_logged_s
        assert float(reports[case]["jax_max_rss_mb"]) < max_rss_mb

    # The tokamak PAS+Er probe timed out on the alternative paths, so its safe
    # bounded default remains Schur.
    assert audits["tokamak_2species_PASCollisions_withEr_fullTrajectories"]["last_preconditioner"] == "schur"
