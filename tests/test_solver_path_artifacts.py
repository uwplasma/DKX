from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CPU_AUDIT = REPO / "tests" / "scaled_example_suite_solver_path_audit_2026-04-27"
GPU_FOCUSED = REPO / "tests" / "gpu_solver_path_cases_2026-04-28"
CPU_FULL_TRACE = REPO / "tests" / "solver_policy_trace_gate_cpu_full_2026-04-30"
GPU_TRACE_PROBE = REPO / "tests" / "solver_policy_trace_gate_gpu_probe_2026-04-30"
GPU_FULL_TRACE = REPO / "tests" / "solver_policy_trace_gate_gpu_full_2026-04-30"


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def test_cpu_solver_path_audit_stays_parity_clean() -> None:
    report = _load_json(CPU_AUDIT / "suite_report_strict.json")

    assert len(report) == 39
    assert {row["status"] for row in report} == {"parity_ok"}
    assert sum(int(row["strict_n_mismatch_common"]) for row in report) == 0
    assert all(not row["print_missing_signals"] for row in report)


def test_cpu_full_fp_cliff_cases_stay_on_dense_audit_path() -> None:
    audit = {row["case"]: row for row in _load_json(CPU_AUDIT / "solver_path_audit.json")}
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
        assert row["profile_stage_durations_s"]["rhs1_solve"] < 1.0


def test_gpu_neighboring_full_fp_artifacts_reject_slow_krylov_cliff() -> None:
    for path in sorted(GPU_FOCUSED.glob("tokamak_fp_ntheta13_nxi*_gpu_bounded_default.json")):
        rows = {row["variant"]: row for row in _load_json(path)}
        default = rows["default"]
        forced_krylov = rows["forced_krylov"]

        assert default["status"] == "ok"
        assert forced_krylov["status"] == "ok"
        assert default["vs_fortran"]["count"] == 0
        assert forced_krylov["vs_fortran"]["count"] == 0
        assert default["solver_path"]["used_dense_auto"] is True
        assert default["solver_path"]["used_sparse_fallback"] is False
        assert float(default["elapsed_s"]) < 0.5 * float(forced_krylov["elapsed_s"])
        assert float(default["ru_maxrss_mb"]) < float(forced_krylov["ru_maxrss_mb"])


def test_cpu_pas_heavy_audit_cases_stay_on_structured_paths() -> None:
    audit = {row["case"]: row for row in _load_json(CPU_AUDIT / "solver_path_audit.json")}
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
        assert row["profile_stage_durations_s"]["rhs1_solve"] < 2.0
        assert row["profile_peak_rss_mb"] < 2100.0


def test_cpu_pas_heavy_audit_cases_remain_strict_parity_clean() -> None:
    report = {row["case"]: row for row in _load_json(CPU_AUDIT / "suite_report_strict.json")}
    pas_cases = [
        case
        for case in report
        if "PAS" in case or case == "monoenergetic_geometryScheme11"
    ]

    assert len(pas_cases) >= 13
    for case in pas_cases:
        row = report[case]
        assert row["status"] == "parity_ok"
        assert row["strict_n_mismatch_common"] == 0
        assert row["print_missing_signals"] == []


def test_trace_backed_full_cpu_suite_report_is_strict_clean() -> None:
    report = _load_json(CPU_FULL_TRACE / "suite_report_strict.json")

    assert len(report) == 39
    assert {row["status"] for row in report} == {"parity_ok"}
    assert sum(int(row["strict_n_mismatch_common"]) for row in report) == 0
    assert sum(len(row.get("print_missing_signals") or []) for row in report) == 0
    assert len(list(CPU_FULL_TRACE.glob("*/*solver_trace*.json"))) == 39


def test_trace_backed_gpu_probe_report_is_strict_clean() -> None:
    report = _load_json(GPU_TRACE_PROBE / "suite_report_strict.json")

    assert len(report) == 7
    assert {row["status"] for row in report} == {"parity_ok"}
    assert sum(int(row["strict_n_mismatch_common"]) for row in report) == 0
    assert sum(len(row.get("print_missing_signals") or []) for row in report) == 0
    assert len(list(GPU_TRACE_PROBE.glob("*/*solver_trace*.json"))) == 7

    traces = [
        _load_json(path)
        for path in sorted(GPU_TRACE_PROBE.glob("*/*solver_trace*.json"))
    ]
    assert {trace["backend"] for trace in traces} == {"gpu"}
    assert {trace["selected_path"] for trace in traces} == {"rhsmode1_solution"}


def test_trace_backed_full_gpu_suite_report_is_strict_clean() -> None:
    report = _load_json(GPU_FULL_TRACE / "suite_report_strict.json")

    assert len(report) == 39
    assert {row["status"] for row in report} == {"parity_ok"}
    assert sum(int(row["strict_n_mismatch_common"]) for row in report) == 0
    assert sum(len(row.get("print_missing_signals") or []) for row in report) == 0
    assert len(list(GPU_FULL_TRACE.glob("*/*solver_trace*.json"))) == 39

    traces = [
        _load_json(path)
        for path in sorted(GPU_FULL_TRACE.glob("*/*solver_trace*.json"))
    ]
    assert {trace["backend"] for trace in traces} == {"gpu"}
    assert {"rhsmode1_solution", "transport_matrix"}.issuperset(
        {trace["selected_path"] for trace in traces}
    )
