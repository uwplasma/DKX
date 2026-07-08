from __future__ import annotations

import json
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_FOCUSED_ROOT = _REPO / "tests" / "gpu_solver_path_cases_2026-04-28"
_GPU_SUITE_SUMMARY = (
    _REPO
    / "tests"
    / "reference_solver_path_artifacts"
    / "gpu_bounded_default_summary_2026-04-28.json"
)


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
    suite = json.loads(_GPU_SUITE_SUMMARY.read_text())["suite"]

    assert suite["row_count"] == 39
    assert suite["strict_row_count"] == 39
    assert suite["statuses"] == ["parity_ok"]
    assert suite["strict_statuses"] == ["parity_ok"]
    assert suite["runtime_drift_flagged_cases"] == 0
    assert suite["missing_output_key_total"] == 0


def test_bounded_gpu_pas_geometry11_defaults_to_pas_tz() -> None:
    reports = json.loads(_GPU_SUITE_SUMMARY.read_text())["case_checks"]

    expected_pas_tz = {
        "HSX_PASCollisions_fullTrajectories": (9.0, 1700.0),
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories": (7.0, 1700.0),
    }
    for case, (max_logged_s, max_rss_mb) in expected_pas_tz.items():
        assert reports[case]["last_preconditioner"] == "pas_tz"
        assert reports[case]["n_mismatch_common"] == 0
        assert reports[case]["strict_n_mismatch_common"] == 0
        assert float(reports[case]["jax_logged_elapsed_s"]) < max_logged_s
        assert float(reports[case]["jax_max_rss_mb"]) < max_rss_mb

    # The tokamak PAS+Er probe timed out on the alternative paths, so its safe
    # bounded default remains Schur.
    assert reports["tokamak_2species_PASCollisions_withEr_fullTrajectories"]["last_preconditioner"] == "schur"


def test_bounded_gpu_monoenergetic_geometry1_uses_dense_transport_gate() -> None:
    reports = json.loads(_GPU_SUITE_SUMMARY.read_text())["case_checks"]
    row = reports["monoenergetic_geometryScheme1"]

    assert row["n_mismatch_common"] == 0
    assert row["strict_n_mismatch_common"] == 0
    assert float(row["jax_logged_elapsed_s"]) < 4.0
    assert float(row["jax_max_rss_mb"]) < 1100.0
