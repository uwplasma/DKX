from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "reference_solver_path_artifacts"


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def test_cpu_solver_path_audit_stays_parity_clean() -> None:
    report = _load_json(FIXTURES / "cpu_audit_snapshot.json")["strict_report"]

    assert report["len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0


def test_cpu_full_fp_cliff_cases_stay_on_dense_audit_path() -> None:
    audit = _load_json(FIXTURES / "cpu_audit_snapshot.json")["full_fp_cases"]
    full_fp_cases = [
        "tokamak_1species_FPCollisions_noEr",
        "tokamak_1species_FPCollisions_withEr_DKESTrajectories",
        "tokamak_1species_FPCollisions_withEr_fullTrajectories",
        "quick_2species_FPCollisions_noEr",
        "inductiveE_noEr",
    ]

    for case in full_fp_cases:
        row = audit[case]
        assert row["dense_auto"] is True
        assert row["default_krylov"] is False
        assert row["dense_fallback"] is False
        assert row["sparse_fallback"] is False
        assert row["profile_peak_rss_mb"] < 650.0
        assert row["rhs1_solve_s"] < 1.0


def test_gpu_neighboring_full_fp_artifacts_reject_slow_krylov_cliff() -> None:
    audits = _load_json(FIXTURES / "gpu_neighboring_full_fp_snapshot.json")
    for rows in audits.values():
        default = rows["default"]
        forced_krylov = rows["forced_krylov"]

        assert default["status"] == "ok"
        assert forced_krylov["status"] == "ok"
        assert default["vs_fortran_count"] == 0
        assert forced_krylov["vs_fortran_count"] == 0
        assert default["used_dense_auto"] is True
        assert default["used_sparse_fallback"] is False
        assert float(default["elapsed_s"]) < 0.5 * float(forced_krylov["elapsed_s"])
        assert float(default["ru_maxrss_mb"]) < float(forced_krylov["ru_maxrss_mb"])


def test_cpu_pas_heavy_audit_cases_stay_on_structured_paths() -> None:
    audit = _load_json(FIXTURES / "cpu_audit_snapshot.json")["pas_heavy_cases"]
    expected_preconditioners = {
        "HSX_PASCollisions_DKESTrajectories": "pas_tz",
        "HSX_PASCollisions_fullTrajectories": "pas_tz",
        "geometryScheme4_1species_PAS_withEr_DKESTrajectories": "pas_tz",
        "geometryScheme4_2species_PAS_noEr": "pas_tz",
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories": "pas_tz",
        "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories": "pas_tz",
        "tokamak_1species_PASCollisions_noEr": "pas_tokamak_theta",
        "tokamak_1species_PASCollisions_noEr_Nx1": "xblock_tz",
        "tokamak_1species_PASCollisions_withEr_fullTrajectories": "xblock_tz",
        "tokamak_2species_PASCollisions_noEr": "schur",
        "tokamak_2species_PASCollisions_withEr_fullTrajectories": "schur",
    }

    for case, expected in expected_preconditioners.items():
        row = audit[case]
        assert row["last_preconditioner"] == expected
        assert row["dense_auto"] is False
        assert row["dense_fallback"] is False
        assert row["sparse_fallback"] is False
        assert row["rhs1_solve_s"] < 2.0
        assert row["profile_peak_rss_mb"] < 2100.0


def test_cpu_pas_heavy_audit_cases_remain_strict_parity_clean() -> None:
    report = _load_json(FIXTURES / "cpu_audit_snapshot.json")["strict_report"]["pas_cases"]
    pas_cases = list(report)

    assert len(pas_cases) >= 13
    for case in pas_cases:
        row = report[case]
        assert row["status"] == "parity_ok"
        assert row["strict_n_mismatch_common"] == 0
        assert row["print_missing_signals"] == []


def test_trace_backed_full_cpu_suite_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["cpu_full_trace"]

    assert report["report_len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 39


def test_trace_backed_gpu_probe_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["gpu_probe"]

    assert report["report_len"] == 7
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 7
    assert set(report["backends"]) == {"gpu"}
    assert set(report["selected_paths"]) == {"rhsmode1_solution"}


def test_trace_backed_full_gpu_suite_report_is_strict_clean() -> None:
    report = _load_json(FIXTURES / "trace_artifact_summary.json")["gpu_full"]

    assert report["report_len"] == 39
    assert set(report["statuses"]) == {"parity_ok"}
    assert report["strict_n_mismatch_common_sum"] == 0
    assert report["print_missing_signals_total"] == 0
    assert report["trace_count"] == 39
    assert set(report["backends"]) == {"gpu"}
    assert {"rhsmode1_solution", "transport_matrix"}.issuperset(
        set(report["selected_paths"])
    )
