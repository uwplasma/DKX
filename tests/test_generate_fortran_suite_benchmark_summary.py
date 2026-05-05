from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_fortran_suite_benchmark_summary.py"
    spec = importlib.util.spec_from_file_location("generate_fortran_suite_benchmark_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_report(path: Path, *, backend_scale: float) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(39):
        rows.append(
            {
                "case": f"case_{idx:02d}",
                "status": "parity_ok",
                "blocker_type": "none",
                "fortran_runtime_s": 10.0 + idx,
                "jax_runtime_s": backend_scale * (0.7 + 0.1 * idx),
                "jax_runtime_s_cold": backend_scale * (1.2 + 0.1 * idx),
                "jax_runtime_s_warm": backend_scale * (0.7 + 0.1 * idx),
                "jax_logged_elapsed_s": backend_scale * (0.8 + 0.1 * idx),
                "fortran_max_rss_mb": 100.0 + idx,
                "jax_max_rss_mb": backend_scale * (500.0 + idx),
                "jax_incremental_max_rss_mb": backend_scale * (200.0 + idx),
                "jax_rss_baseline_mb": backend_scale * 300.0,
                "jax_memory_metric_source": "drss_mb",
                "final_resolution": {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100},
                "n_mismatch_common": 0,
                "n_mismatch_physics": 0,
                "n_mismatch_solver": 0,
                "strict_n_mismatch_common": 0,
                "strict_n_mismatch_physics": 0,
                "strict_n_mismatch_solver": 0,
            }
        )
    path.write_text(json.dumps(rows, indent=2) + "\n")


def test_generate_fortran_suite_benchmark_summary_from_reports(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"
    cpu_report = tmp_path / "cpu_suite_report.json"
    gpu_report = tmp_path / "gpu_suite_report.json"
    _write_synthetic_report(cpu_report, backend_scale=1.0)
    _write_synthetic_report(gpu_report, backend_scale=1.4)

    rc = mod.main(
        [
            "--cpu-report",
            str(cpu_report),
            "--gpu-report",
            str(gpu_report),
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "suite_benchmark_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "suite_benchmark_test.png").exists()
    assert (out_dir / "suite_benchmark_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "fortran_v3_suite_benchmark_summary"
    assert payload["reports"]["cpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["gpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["cpu"]["strict_mismatch_total"] == 0
    assert payload["reports"]["gpu"]["strict_mismatch_total"] == 0
    assert payload["reports"]["cpu"]["cold_runtime_ratio_summary"]["count"] == 39
    assert payload["reports"]["gpu"]["warm_runtime_ratio_summary"]["count"] == 39
    assert payload["reports"]["gpu"]["warm_or_logged_runtime_ratio_summary"]["count"] == 39
    assert payload["reports"]["gpu"]["warm_or_logged_runtime_source_counts"] == {"jax_runtime_s_warm": 39}


def test_generate_fortran_suite_benchmark_summary_filters_short_reference_runs(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"
    cpu_report = tmp_path / "cpu_suite_report.json"
    gpu_report = tmp_path / "gpu_suite_report.json"
    _write_synthetic_report(cpu_report, backend_scale=1.0)
    _write_synthetic_report(gpu_report, backend_scale=1.2)

    rows = json.loads(cpu_report.read_text())
    rows[0]["fortran_runtime_s"] = 0.5
    cpu_report.write_text(json.dumps(rows, indent=2) + "\n")
    rows = json.loads(gpu_report.read_text())
    rows[0]["fortran_runtime_s"] = 0.5
    gpu_report.write_text(json.dumps(rows, indent=2) + "\n")

    rc = mod.main(
        [
            "--cpu-report",
            str(cpu_report),
            "--gpu-report",
            str(gpu_report),
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "suite_benchmark_filtered_test",
        ]
    )

    assert rc == 0
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["source_case_counts"] == {"cpu": 39, "gpu": 39}
    assert payload["metadata"]["reported_case_counts"] == {"cpu": 38, "gpu": 38}
    assert payload["metadata"]["min_fortran_runtime_s"] == 10.0
    assert payload["metadata"]["excluded_low_fortran_runtime_cases"] == [
        {"case": "case_00", "fortran_runtime_s": 0.5}
    ]
    assert payload["reports"]["cpu"]["parity_ok_cases"] == 38
    assert payload["reports"]["gpu"]["warm_or_logged_runtime_ratio_summary"]["count"] == 38


def test_case_order_prioritizes_warm_jax_speedup() -> None:
    mod = _load_module()

    def row(case: str, *, fortran: float, warm: float) -> dict[str, object]:
        return {
            "case": case,
            "status": "parity_ok",
            "blocker_type": "none",
            "fortran_runtime_s": fortran,
            "jax_runtime_s": warm,
            "jax_runtime_s_cold": warm * 2.0,
            "jax_runtime_s_warm": warm,
            "jax_logged_elapsed_s": warm,
            "fortran_max_rss_mb": 100.0,
            "jax_max_rss_mb": 500.0,
            "jax_incremental_max_rss_mb": 200.0,
            "jax_rss_baseline_mb": 300.0,
            "jax_memory_metric_source": "drss_mb",
            "final_resolution": {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100},
            "n_mismatch_common": 0,
            "n_mismatch_physics": 0,
            "n_mismatch_solver": 0,
            "strict_n_mismatch_common": 0,
            "strict_n_mismatch_physics": 0,
            "strict_n_mismatch_solver": 0,
        }

    metrics = mod.suite_case_metrics(
        [
            row("slow_jax", fortran=1.0, warm=4.0),
            row("medium_speedup", fortran=20.0, warm=5.0),
            row("largest_speedup", fortran=100.0, warm=10.0),
        ]
    )

    assert mod._case_order(metrics, metrics) == ["largest_speedup", "medium_speedup", "slow_jax"]
