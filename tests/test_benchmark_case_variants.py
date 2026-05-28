from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_benchmark_module():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "benchmark_case_variants_under_test",
        repo / "scripts" / "benchmark_case_variants.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tail_text_normalizes_timeout_bytes() -> None:
    module = _load_benchmark_module()

    assert module._tail_text(b"alpha\nbeta", 4) == "beta"
    assert module._tail_text("alpha\nbeta", 5) == "\nbeta"
    assert module._tail_text(None, 4) == ""


def test_rhs_mode_from_input_controls_transport_matrix(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "input.namelist"

    path.write_text("&general\n RHSMode = 3\n/\n")
    assert module._rhs_mode_from_input(path) == 3

    path.write_text("&general\n RHSMode = 1\n/\n")
    assert module._rhs_mode_from_input(path) == 1


def test_last_rhs1_preconditioner_parses_final_solver_line() -> None:
    module = _load_benchmark_module()

    stdout = "\n".join(
        [
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=xblock_tz (active-DOF)",
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=pas_tz (active-DOF)",
        ]
    )
    assert module._last_rhs1_preconditioner(stdout) == "pas_tz"
    assert module._last_rhs1_preconditioner("no preconditioner line") is None


def test_solver_path_summary_parses_profile_marks_and_memory_units() -> None:
    module = _load_benchmark_module()

    stdout = "\n".join(
        [
            "profiling: rhs1_sparse_precond_build_start total_s=1.0 delta_s=0.2 rss_mb=500.0",
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=theta_line (active-DOF)",
            "profiling: rhs1_sparse_precond_build_done total_s=2.5 delta_s=1.5 rss_mb=900.0",
            "host sparse LU direct fallback on backend=cpu",
        ]
    )
    summary = module._solver_path_summary(stdout)

    assert summary["preconditioners"] == ["theta_line"]
    assert summary["profile_stage_durations_s"]["rhs1_sparse_precond_build"] == 1.5
    assert summary["profile_peak_rss_mb"] == 900.0
    assert summary["used_sparse_fallback"]
    assert module._resource_maxrss_mb(1024 * 1024, platform="darwin") == 1.0
    assert module._resource_maxrss_mb(1024, platform="linux") == 1.0


def test_timeout_budget_requires_explicit_long_run_opt_in() -> None:
    module = _load_benchmark_module()

    module._validate_timeout_budget(600.0, allow_long_run=False)
    module._validate_timeout_budget(601.0, allow_long_run=True)

    with pytest.raises(ValueError, match="capped at 600s"):
        module._validate_timeout_budget(601.0, allow_long_run=False)


def test_benchmark_progress_summary_records_profile_progress_and_budget() -> None:
    module = _load_benchmark_module()

    stdout = "\n".join(
        [
            "profiling: rhs1_sparse_precond_build_start total_s=1.0 delta_s=0.1 rss_mb=100.0",
            "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=pas_tz (active-DOF)",
            "profiling: rhs1_sparse_precond_build_done total_s=1.5 delta_s=0.5 rss_mb=150.0",
            '@@RESULT@@{"elapsed_s": 0.25}',
        ]
    )

    summary = module._benchmark_progress_summary(
        stdout,
        "",
        profile_requested=True,
        timeout_s=300.0,
        status="ok",
        wall_s=1.23456,
    )

    assert summary["status"] == "ok"
    assert summary["wall_s"] == 1.235
    assert summary["timeout_s"] == 300.0
    assert summary["default_timeout_cap_s"] == 600.0
    assert summary["within_default_timeout_cap"] is True
    assert summary["profile_requested"] is True
    assert summary["profile_event_count"] == 2
    assert summary["profile_stage_count"] == 1
    assert summary["rhs1_preconditioner_count"] == 1
    assert summary["last_rhs1_preconditioner"] == "pas_tz"
    assert summary["result_marker_count"] == 1
    assert summary["progress_markers_seen"] is True


def test_benchmark_case_variants_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    case_dir = tmp_path / "case"
    json_out = tmp_path / "bench.json"
    case_dir.mkdir()
    (case_dir / "input.namelist").write_text(source.read_text())

    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "benchmark_case_variants.py"),
            "--case-dir",
            str(case_dir),
            "--timeout-s",
            "120",
            "--json-out",
            str(json_out),
            "--variant",
            "incremental=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=incremental",
            "--variant",
            "lgmres=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=lgmres",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "## running default" in proc.stdout
    rows = json.loads(json_out.read_text())
    assert len(rows) == 3
    assert rows[0]["variant"] == "default"
    assert rows[1]["variant"] == "incremental"
    assert rows[2]["variant"] == "lgmres"
    assert rows[0]["status"] == "ok", rows[0]
    assert rows[1]["status"] == "ok", rows[1]
    assert rows[2]["status"] == "ok", rows[2]
    assert rows[1]["vs_default"]["count"] == 0
    assert rows[2]["vs_default"]["count"] == 0
    assert not rows[1]["used_lgmres"]
    assert rows[2]["used_lgmres"]
    assert rows[0]["benchmark_progress"]["timeout_s"] == 120.0
    assert rows[0]["benchmark_progress"]["within_default_timeout_cap"] is True


def test_benchmark_case_variants_no_default_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    case_dir = tmp_path / "case"
    json_out = tmp_path / "bench.json"
    case_dir.mkdir()
    (case_dir / "input.namelist").write_text(source.read_text())

    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "benchmark_case_variants.py"),
            "--case-dir",
            str(case_dir),
            "--timeout-s",
            "120",
            "--json-out",
            str(json_out),
            "--no-default",
            "--variant",
            "incremental=SFINCS_JAX_RHSMODE1_SOLVE_METHOD=incremental",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "## running default" not in proc.stdout
    assert "## running incremental" in proc.stdout
    rows = json.loads(json_out.read_text())
    assert [row["variant"] for row in rows] == ["incremental"]
    assert rows[0]["status"] == "ok", rows[0]
