from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sfincs_jax.validation_artifacts import (
    build_high_collisionality_trend_proxy_summary,
    build_publication_validation_summary,
    collisionality_power_law_slope,
    collisionality_grid,
    collisionality_labels,
    er_nonzero_model_spread,
    er_zero_field_spread,
    fp_pas_l11_separation,
    high_collisionality_trend_summary,
    load_collisionality_records,
    load_er_sweep_records,
    load_suite_report,
    suite_case_metrics,
    suite_report_summary,
)


def _artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "publication_figures" / "artifacts"


def _synthetic_suite_rows(n: int = 39) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(int(n)):
        rows.append(
            {
                "case": f"case_{idx:02d}",
                "status": "parity_ok",
                "blocker_type": "none",
                "fortran_runtime_s": 10.0 + idx,
                "jax_runtime_s": 1.0 + 0.1 * idx,
                "jax_logged_elapsed_s": 0.8 + 0.1 * idx,
                "fortran_max_rss_mb": 100.0 + idx,
                "jax_max_rss_mb": 500.0 + idx,
                "n_mismatch_common": 0,
                "n_mismatch_physics": 0,
                "n_mismatch_solver": 0,
                "strict_n_mismatch_common": 0,
                "strict_n_mismatch_physics": 0,
                "strict_n_mismatch_solver": 0,
            }
        )
    return rows


def test_collisionality_artifact_metrics_are_literature_consistent() -> None:
    for name in ("lhd_collisionality_summary.json", "w7x_collisionality_summary.json"):
        records = load_collisionality_records(_artifact_dir() / name)
        assert collisionality_labels(records) == ["Fokker-Planck", "PAS"]
        assert len(collisionality_grid(records)) == 7

        separation = fp_pas_l11_separation(records)
        assert len(separation) == 7
        assert separation[-1]["relative_to_fp"] > separation[0]["relative_to_fp"]
        assert separation[-1]["relative_to_fp"] > 5.0


def test_er_sweep_artifact_metrics_pin_zero_field_and_finite_field_behavior() -> None:
    for name in ("er_sweep_tokamak_reference_summary.json", "er_sweep_stellarator_fast_reference_summary.json"):
        records = load_er_sweep_records(_artifact_dir() / name)
        zero_spread = er_zero_field_spread(records)
        assert all(value <= 1e-12 for value in zero_spread.values())

        jhat_spreads = er_nonzero_model_spread(records, field="fsab_jhat")
        assert jhat_spreads
        assert all(np.isfinite(value) for value in jhat_spreads.values())
        assert all(value > 0.0 for value in jhat_spreads.values())


def test_publication_validation_summary_has_research_gate_payload() -> None:
    payload = build_publication_validation_summary(artifact_dir=_artifact_dir())
    assert payload["metadata"]["kind"] == "publication_validation_dashboard"
    assert "https://doi.org/10.1063/1.4870077" in payload["metadata"]["literature"]
    assert payload["collisionality"]["lhd"]["l11_high_to_low_relative_separation_ratio"] > 10.0
    assert payload["collisionality"]["w7x"]["l11_high_to_low_relative_separation_ratio"] > 10.0
    assert payload["trajectory_sweeps"]["tokamak"]["models"] == ["dkes", "full", "partial"]
    assert payload["trajectory_sweeps"]["stellarator"]["models"] == ["dkes", "full", "partial"]


def test_high_collisionality_tail_slopes_match_expected_proxy_behavior() -> None:
    lhd = load_collisionality_records(_artifact_dir() / "lhd_collisionality_summary.json")
    w7x = load_collisionality_records(_artifact_dir() / "w7x_collisionality_summary.json")

    for records in (lhd, w7x):
        pas_l11 = collisionality_power_law_slope(records, label="PAS", element=(0, 0), n_fit=3)
        pas_l12 = collisionality_power_law_slope(records, label="PAS", element=(0, 1), n_fit=3)
        assert pas_l11 > 0.65
        assert pas_l12 > 0.65

    w7x_fp_l11 = collisionality_power_law_slope(w7x, label="Fokker-Planck", element=(0, 0), n_fit=3)
    w7x_fp_l12 = collisionality_power_law_slope(w7x, label="Fokker-Planck", element=(0, 1), n_fit=3)
    assert w7x_fp_l11 < -1.0
    assert w7x_fp_l12 < -1.0

    lhd_summary = high_collisionality_trend_summary(lhd, n_fit=3)
    assert lhd_summary["gates"]["pas_l11_l12_positive"] is True
    assert lhd_summary["gates"]["fp_l11_l12_inverse_like"] is False
    assert lhd_summary["state"] == "needs_wider_high_nu_scan"


def test_high_collisionality_proxy_summary_keeps_analytic_limit_lane_honest() -> None:
    payload = build_high_collisionality_trend_proxy_summary(artifact_dir=_artifact_dir(), n_fit=3)
    assert payload["metadata"]["kind"] == "high_collisionality_trend_proxy"
    assert "nu' >> 1" in " ".join(payload["metadata"]["notes"])
    assert payload["cases"]["lhd"]["state"] == "needs_wider_high_nu_scan"
    assert payload["cases"]["w7x"]["state"] == "asymptotic_trend_proxy"


def test_fortran_suite_report_summary_closes_cpu_gpu_release_gate_on_synthetic_rows(tmp_path: Path) -> None:
    report = tmp_path / "suite_report.json"
    report.write_text(json.dumps(_synthetic_suite_rows(), indent=2) + "\n")
    rows = load_suite_report(report)
    metrics = suite_case_metrics(rows)
    payload = suite_report_summary(rows, label="CPU")

    assert len(metrics) == 39
    assert payload["total_cases"] == 39
    assert payload["parity_ok_cases"] == 39
    assert payload["jax_error_cases"] == 0
    assert payload["max_attempts_cases"] == 0
    assert payload["practical_mismatch_cases"] == 0
    assert payload["strict_mismatch_cases"] == 0
    assert payload["runtime_ratio_summary"]["count"] == 39
    assert payload["memory_ratio_summary"]["count"] == 39
    assert all(metric.runtime_ratio is not None and metric.runtime_ratio > 0.0 for metric in metrics)
    assert all(metric.memory_ratio is not None and metric.memory_ratio > 0.0 for metric in metrics)


def test_fortran_suite_benchmark_summary_records_source_reports_and_gates() -> None:
    payload = json.loads((_artifact_dir() / "sfincs_jax_fortran_suite_benchmark_summary.json").read_text())

    assert payload["metadata"]["kind"] == "fortran_v3_suite_benchmark_summary"
    assert "https://github.com/landreman/sfincs" in payload["metadata"]["literature"]
    assert payload["reports"]["cpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["gpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["cpu"]["strict_mismatch_total"] == 0
    assert payload["reports"]["gpu"]["strict_mismatch_total"] == 0
