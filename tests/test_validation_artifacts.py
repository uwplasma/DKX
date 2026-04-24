from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from sfincs_jax.validation_artifacts import (
    appendix_b_geometry_audit_from_h5,
    autodiff_gradient_error_summary,
    load_autodiff_sensitivity_summary,
    build_high_collisionality_trend_proxy_summary,
    build_publication_validation_summary,
    build_simakov_helander_limit_audit_summary,
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


def _write_appendix_b_geometry_fixture(path: Path) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 5.0, 5, endpoint=False)
    theta_2d = theta[:, None]
    zeta_2d = zeta[None, :]
    b_hat = 1.0 + 0.08 * np.cos(theta_2d) + 0.04 * np.cos(5.0 * zeta_2d)
    g_hat = 3.0
    i_hat = 0.2
    iota = 0.45
    d_hat = b_hat * b_hat / (g_hat + iota * i_hat)
    u_hat = 0.12 * np.sin(theta_2d) + 0.03 * np.sin(5.0 * zeta_2d)
    weights = 1.0 / d_hat
    fsab_hat2 = float(np.sum(weights * b_hat * b_hat) / np.sum(weights))

    with h5py.File(path, "w") as h5:
        h5["BHat"] = b_hat
        h5["DHat"] = d_hat
        h5["uHat"] = u_hat
        h5["BHat_sup_theta"] = iota * d_hat
        h5["BHat_sup_zeta"] = d_hat
        h5["dBHatdtheta"] = -0.08 * np.sin(theta_2d) + np.zeros_like(zeta_2d)
        h5["dBHatdzeta"] = np.zeros_like(theta_2d) - 0.2 * np.sin(5.0 * zeta_2d)
        h5["theta"] = theta
        h5["zeta"] = zeta
        h5["GHat"] = np.asarray(g_hat)
        h5["IHat"] = np.asarray(i_hat)
        h5["iota"] = np.asarray(iota)
        h5["FSABHat2"] = np.asarray(fsab_hat2)


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


def test_simakov_helander_audit_records_geometry_and_keeps_full_gate_closed(tmp_path: Path) -> None:
    fixture_h5 = tmp_path / "appendix_b_fixture.h5"
    _write_appendix_b_geometry_fixture(fixture_h5)

    geometry = appendix_b_geometry_audit_from_h5(fixture_h5)
    assert geometry["geometry_scalars"]["FSABHat2_relative_error"] < 1.0e-12
    assert "G1" in geometry["appendix_b_discrete_quantities"]
    assert "L11" in geometry["transport_matrix_coefficients_over_nuprime"]

    pinned = json.loads((_artifact_dir() / "sfincs_jax_simakov_helander_limit_audit_summary.json").read_text())
    precomputed = {
        case: pinned["cases"][case]["appendix_b_geometry_audit"]
        for case in ("lhd", "w7x")
    }
    payload = build_simakov_helander_limit_audit_summary(
        artifact_dir=_artifact_dir(),
        precomputed_geometry_audits=precomputed,
        n_fit=3,
    )
    assert payload["metadata"]["kind"] == "simakov_helander_limit_audit"
    assert payload["cases"]["lhd"]["gates"]["scan_extends_to_required_high_nu"] is False
    assert payload["cases"]["w7x"]["gates"]["fp_l11_l12_target_inverse_slope"] is True
    assert payload["gates"]["all_cases_ready_for_full_overlay"] is False
    assert payload["gates"]["full_simakov_helander_reproduction_closed"] is True


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


def test_autodiff_sensitivity_summary_records_gradient_and_residual_gates() -> None:
    payload = load_autodiff_sensitivity_summary(
        _artifact_dir() / "sfincs_jax_autodiff_sensitivity_validation_summary.json"
    )
    errors = autodiff_gradient_error_summary(payload)

    assert errors["count"] == 3
    assert errors["max_relative_error"] < 1.0e-4
    assert payload["gates"]["gradient_relative_error_ok"] is True
    assert payload["gates"]["primal_residual_ok"] is True
    assert payload["gates"]["adjoint_residual_ok"] is True
    assert payload["geometry_sensitivity"]["kind"] == "scheme4_boozer_harmonic_map"
    assert payload["geometry_sensitivity"]["gradient_relative_error"] < 1.0e-8
    assert payload["cost_scaling"][-1]["centered_finite_difference_solve_count_model"] > payload["cost_scaling"][-1][
        "implicit_solve_count_model"
    ]
