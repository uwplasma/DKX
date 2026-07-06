from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "summarize_solver_paths_under_test",
        repo / "scripts" / "summarize_solver_paths.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_summarize_log_extracts_solver_path(tmp_path: Path) -> None:
    module = _load_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    log_path = case_dir / "sfincs_jax.log"
    log_path.write_text(
        "\n".join(
            [
                "write_sfincs_jax_output_h5: FP RHSMode=1 bounded system -> using dense solve",
                "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner=theta_line",
                "profiling: rhs1_solve_start dt_s=0.1 total_s=0.2 rss_mb=100.0 drss_mb=1.0 device_mb=na",
                "profiling: rhs1_solve_done dt_s=1.5 total_s=1.7 rss_mb=300.0 drss_mb=2.0 device_mb=na",
                "ksp_iterations=7 solver=gmres",
            ]
        )
    )

    summary = module.summarize_log(log_path)

    assert summary["case"] == "case"
    assert summary["dense_auto"]
    assert not summary["host_dense_shortcut"]
    assert summary["last_preconditioner"] == "theta_line"
    assert summary["profile_peak_rss_mb"] == 300.0
    assert summary["profile_stage_durations_s"]["rhs1_solve"] == 1.5
    assert summary["ksp_iterations"] == [{"solver": "gmres", "iterations": 7}]


def test_summarize_log_extracts_host_dense_shortcut(tmp_path: Path) -> None:
    module = _load_module()
    case_dir = tmp_path / "gpu_case"
    case_dir.mkdir()
    log_path = case_dir / "sfincs_jax.log"
    log_path.write_text(
        "\n".join(
            [
                "write_sfincs_jax_output_h5: FP RHSMode=1 bounded system -> using host dense shortcut on backend=gpu",
                "solve_v3_full_system_linear_gmres: accelerator FP bounded system -> using host dense shortcut (size=4096)",
                "profiling: rhs1_host_dense_shortcut_start dt_s=0.1 total_s=0.1 rss_mb=200.0 drss_mb=2.0 device_mb=na",
                "profiling: rhs1_host_dense_shortcut_done dt_s=2.5 total_s=2.6 rss_mb=240.0 drss_mb=3.0 device_mb=na",
            ]
        )
    )

    summary = module.summarize_log(log_path)

    assert summary["case"] == "gpu_case"
    assert not summary["dense_auto"]
    assert summary["host_dense_shortcut"]
    assert not summary["default_krylov"]


def test_summarize_solver_paths_cli_writes_outputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    case_dir = tmp_path / "suite" / "case"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincs_jax.log").write_text("profiling: write_h5_done dt_s=0.2 total_s=0.2 rss_mb=42.0")
    json_out = tmp_path / "solver_paths.json"
    md_out = tmp_path / "solver_paths.md"

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "summarize_solver_paths.py"),
            "--suite-root",
            str(tmp_path / "suite"),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = json.loads(json_out.read_text())
    assert rows[0]["case"] == "case"
    markdown = md_out.read_text()
    assert "| Host dense shortcut |" in markdown
    assert "| case |" in markdown
