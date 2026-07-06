from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import sfincs_jax.validation.artifacts as artifacts
from sfincs_jax.validation.artifacts import (
    appendix_b_geometry_audit_from_h5,
    autodiff_gradient_error_summary,
    benchmark_artifact_policy_errors,
    build_fortran_suite_benchmark_summary,
    benchmark_resolution_floor_violations,
    check_benchmark_artifact_file,
    check_benchmark_artifact_files,
    classify_benchmark_artifact_file,
    load_autodiff_sensitivity_summary,
    build_high_collisionality_trend_proxy_summary,
    build_publication_validation_summary,
    build_simakov_helander_limit_audit_summary,
    collisionality_power_law_slope,
    collisionality_grid,
    collisionality_labels,
    er_nonzero_model_spread,
    er_zero_field_spread,
    fortran_suite_benchmark_schema_errors,
    fortran_suite_benchmark_summary_errors,
    fp_pas_l11_separation,
    high_collisionality_trend_summary,
    index_benchmark_artifact_files,
    load_collisionality_records,
    load_er_sweep_records,
    load_suite_report,
    recommended_high_collisionality_nuprime_grid,
    suite_case_metrics,
    suite_report_summary,
)
from sfincs_jax.validation.data_fetch import external_data_dir, external_data_version


def _artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "publication_figures" / "artifacts"


def test_external_data_version_and_dir_follow_manifest_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_DATA_DIR", str(tmp_path / "data"))

    version = external_data_version()
    path = external_data_dir()

    assert version
    assert path == tmp_path / "data" / version


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
                "jax_incremental_max_rss_mb": 200.0 + idx,
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

    lhd_summary = high_collisionality_trend_summary(lhd, n_fit=3)
    assert lhd_summary["gates"]["pas_l11_l12_positive"] is True
    assert lhd_summary["gates"]["fp_l11_l12_inverse_like"] is True
    assert lhd_summary["state"] == "asymptotic_trend_proxy"

    w7x_summary = high_collisionality_trend_summary(w7x, n_fit=3)
    assert w7x_summary["gates"]["pas_l11_l12_positive"] is True
    assert w7x_summary["gates"]["fp_l11_l12_inverse_like"] is False
    assert w7x_summary["state"] == "needs_wider_high_nu_scan"


def test_high_collisionality_proxy_summary_keeps_analytic_limit_lane_honest() -> None:
    payload = build_high_collisionality_trend_proxy_summary(artifact_dir=_artifact_dir(), n_fit=3)
    assert payload["metadata"]["kind"] == "high_collisionality_trend_proxy"
    assert "nu' >> 1" in " ".join(payload["metadata"]["notes"])
    assert payload["cases"]["lhd"]["state"] == "asymptotic_trend_proxy"
    assert payload["cases"]["w7x"]["state"] == "needs_wider_high_nu_scan"
    assert payload["gates"]["all_fp_l11_l12_inverse_like"] is False
    assert payload["gates"]["ready_for_literature_claim"] is False


def test_recommended_high_collisionality_nuprime_grid_extends_beyond_current_tail() -> None:
    extension = recommended_high_collisionality_nuprime_grid(
        [0.1, 1.0, 10.0],
        min_nuprime_for_full_limit=50.0,
    )
    assert extension[0] > 10.0
    assert extension[-1] >= 100.0
    assert all(b > a for a, b in zip(extension, extension[1:]))

    assert (
        recommended_high_collisionality_nuprime_grid(
            [0.1, 10.0, 100.0],
            min_nuprime_for_full_limit=50.0,
        )
        == []
    )


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
    assert payload["cases"]["lhd"]["recommended_high_nuprime_extension"][0] > payload["cases"]["lhd"]["max_nuprime"]
    assert payload["cases"]["lhd"]["gates"]["fp_l11_l12_target_inverse_slope"] is False
    assert payload["cases"]["w7x"]["gates"]["fp_l11_l12_target_inverse_slope"] is False
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
    assert payload["active_memory_ratio_summary"]["count"] == 39
    assert all(metric.runtime_ratio is not None and metric.runtime_ratio > 0.0 for metric in metrics)
    assert all(metric.memory_ratio is not None and metric.memory_ratio > 0.0 for metric in metrics)
    assert all(metric.active_memory_ratio is not None and metric.active_memory_ratio > 0.0 for metric in metrics)
    assert payload["highest_active_jax_memory_cases"][0]["active_jax_memory_mb"] == 238.0


def test_fortran_suite_benchmark_summary_can_filter_short_reference_runs(tmp_path: Path) -> None:
    cpu_report = tmp_path / "cpu_report.json"
    gpu_report = tmp_path / "gpu_report.json"
    rows = _synthetic_suite_rows(n=3)
    rows[0]["fortran_runtime_s"] = 0.2
    cpu_report.write_text(json.dumps(rows, indent=2) + "\n")
    gpu_report.write_text(json.dumps(rows, indent=2) + "\n")

    payload = build_fortran_suite_benchmark_summary(
        cpu_report=cpu_report,
        gpu_report=gpu_report,
        min_fortran_runtime_s=10.0,
    )

    assert payload["metadata"]["source_case_counts"] == {"cpu": 3, "gpu": 3}
    assert payload["metadata"]["reported_case_counts"] == {"cpu": 2, "gpu": 2}
    assert payload["metadata"]["excluded_low_fortran_runtime_cases"] == [
        {"case": "case_00", "fortran_runtime_s": 0.2}
    ]
    assert payload["metadata"]["resolution_floor_violations"] == {"cpu": [], "gpu": []}
    assert fortran_suite_benchmark_schema_errors(payload) == []
    assert payload["reports"]["cpu"]["total_cases"] == 2


def test_fortran_suite_benchmark_summary_rejects_below_floor_public_rows(tmp_path: Path) -> None:
    cpu_report = tmp_path / "cpu_report.json"
    gpu_report = tmp_path / "gpu_report.json"
    rows = _synthetic_suite_rows(n=2)
    rows[0]["final_resolution"] = {"NTHETA": 5, "NZETA": 5, "NX": 2, "NXI": 4}
    cpu_report.write_text(json.dumps(rows, indent=2) + "\n")
    gpu_report.write_text(json.dumps(rows, indent=2) + "\n")

    violations = benchmark_resolution_floor_violations(rows)
    assert violations[0]["case"] == "case_00"
    assert violations[0]["reason"] == "below_public_benchmark_resolution_floor"

    with pytest.raises(ValueError, match="below_public_benchmark_resolution_floor"):
        build_fortran_suite_benchmark_summary(
            cpu_report=cpu_report,
            gpu_report=gpu_report,
            min_fortran_runtime_s=10.0,
        )


def test_fortran_suite_benchmark_schema_validator_fails_closed() -> None:
    payload = {
        "metadata": {"schema_version": 99, "kind": "wrong"},
        "reports": {"cpu": {}, "gpu": {"total_cases": 0}},
    }

    errors = fortran_suite_benchmark_schema_errors(payload)

    assert "metadata.kind must be fortran_v3_suite_benchmark_summary" in errors
    assert "metadata.schema_version must be 1" in errors
    assert "reports.cpu.total_cases missing" in errors
    assert "reports.gpu.parity_ok_cases missing" in errors


def test_fortran_suite_benchmark_summary_records_source_reports_and_gates() -> None:
    payload = json.loads((_artifact_dir() / "sfincs_jax_fortran_suite_benchmark_summary.json").read_text())

    assert payload["metadata"]["kind"] == "fortran_v3_suite_benchmark_summary"
    assert fortran_suite_benchmark_schema_errors(payload) == []
    assert "https://github.com/landreman/sfincs" in payload["metadata"]["literature"]
    assert payload["metadata"]["source_case_counts"] == {"cpu": 39, "gpu": 39}
    assert payload["metadata"]["source_reports"] == {
        "cpu": "tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak/suite_report.json",
        "gpu": "tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas/suite_report.json",
    }
    assert payload["metadata"]["reported_case_counts"] == {"cpu": 24, "gpu": 24}
    assert payload["metadata"]["min_fortran_runtime_s"] == 10.0
    assert len(payload["metadata"]["excluded_low_fortran_runtime_cases"]) == 15
    assert payload["reports"]["cpu"]["parity_ok_cases"] == 24
    assert payload["reports"]["gpu"]["parity_ok_cases"] == 24
    assert payload["reports"]["cpu"]["strict_mismatch_total"] == 0
    assert payload["reports"]["gpu"]["strict_mismatch_total"] == 0


def test_production_gpu_report_preserves_trace_backed_solver_metadata() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas"
        / "suite_report.json"
    )
    rows = {str(row["case"]): row for row in json.loads(report.read_text())}

    expected = {
        "tokamak_1species_FPCollisions_noEr": ("xblock_sparse_pc_gmres", 60),
        "tokamak_1species_FPCollisions_withEr_DKESTrajectories": ("xblock_sparse_pc_gmres", 250),
        "tokamak_1species_FPCollisions_withEr_fullTrajectories": ("xblock_sparse_pc_gmres", 150),
        "tokamak_2species_PASCollisions_withEr_fullTrajectories": ("sparse_pc_gmres", 20),
    }
    for case, (solver_kind, max_matvecs) in expected.items():
        row = rows[case]
        assert row["status"] == "parity_ok"
        assert row["strict_n_mismatch_common"] == 0
        assert row["jax_solver_kinds"] == [solver_kind]
        assert row["jax_solver_iters_n"] == 1
        assert row["jax_solver_iters_max"] <= max_matvecs


def test_production_cpu_report_uses_xblock_for_tokamak_fp_er_rows() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak"
        / "suite_report.json"
    )
    rows = {str(row["case"]): row for row in json.loads(report.read_text())}

    expected = {
        "tokamak_1species_FPCollisions_withEr_DKESTrajectories": 160,
        "tokamak_1species_FPCollisions_withEr_fullTrajectories": 120,
    }
    for case, max_matvecs in expected.items():
        row = rows[case]
        assert row["status"] == "parity_ok"
        assert row["strict_n_mismatch_common"] == 0
        assert row["jax_solver_kinds"] == ["xblock_sparse_pc_gmres"]
        assert row["jax_solver_iters_n"] == 1
        assert row["jax_solver_iters_max"] <= max_matvecs
        assert row["jax_max_rss_mb"] < 500.0


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


def test_validation_artifact_small_helper_edges(tmp_path: Path) -> None:
    assert recommended_high_collisionality_nuprime_grid(
        [100.0],
        min_nuprime_for_full_limit=50.0,
    ) == []
    with pytest.raises(ValueError, match="positive finite"):
        recommended_high_collisionality_nuprime_grid([0.0, float("nan")], min_nuprime_for_full_limit=50.0)

    assert artifacts.maxrss_mb(platform="darwin", raw_value=1024 * 1024) == pytest.approx(1.0)
    assert artifacts.maxrss_mb(platform="linux", raw_value=1024) == pytest.approx(1.0)

    timer = artifacts.PhaseTimer()
    with pytest.raises(RuntimeError, match="boom"):
        with timer.phase("failing-phase", step="synthetic"):
            raise RuntimeError("boom")
    summary = timer.summary()
    assert summary["phase_count"] == 1
    assert summary["phases"][0]["status"] == "error"
    assert summary["phases"][0]["metadata"] == {"step": "synthetic"}

    ok_timer = artifacts.PhaseTimer()
    with ok_timer.phase("ok-phase"):
        pass
    assert ok_timer.records[0].to_json()["status"] == "ok"

    suite_path = tmp_path / "bad_suite.json"
    suite_path.write_text(json.dumps({"rows": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of case rows"):
        load_suite_report(suite_path)

    autodiff_path = tmp_path / "bad_autodiff.json"
    autodiff_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_autodiff_sensitivity_summary(autodiff_path)
    autodiff_path.write_text(json.dumps({"metadata": {"kind": "wrong"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata.kind"):
        load_autodiff_sensitivity_summary(autodiff_path)

    with pytest.raises(ValueError, match="gradient_checks"):
        autodiff_gradient_error_summary({"gradient_checks": object()})
    summary = autodiff_gradient_error_summary({"gradient_checks": [{"relative_error": "bad"}, "skip"]})
    assert summary["count"] == 0
    assert np.isnan(summary["max_relative_error"])


def test_validation_artifact_resolution_and_geometry_fail_closed_edges(tmp_path: Path) -> None:
    rows = [
        {"case": "tokamak_case", "final_resolution": {"NTHETA": 25, "NZETA": 1, "NX": 4, "NXI": 100}},
        {"case": "stellarator_missing_resolution"},
        {"case": "stellarator_bad_resolution", "final_resolution": {"NTHETA": "bad", "NZETA": 51, "NX": 4}},
    ]

    violations = benchmark_resolution_floor_violations(rows)

    assert violations[0]["case"] == "stellarator_missing_resolution"
    assert violations[0]["reason"] == "missing_final_resolution"
    assert violations[1]["case"] == "stellarator_bad_resolution"
    assert violations[1]["fields"]["NTHETA"]["actual"] is None
    assert violations[1]["fields"]["NXI"]["actual"] is None

    derivative = artifacts._periodic_central_derivative(
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([0.0, 1.0, 2.0]),
        axis=0,
    )
    np.testing.assert_allclose(derivative, np.asarray([-1.0, 1.5, -0.5]))
    np.testing.assert_allclose(
        artifacts._periodic_central_derivative(np.asarray([1.0]), np.asarray([0.0]), axis=0),
        np.asarray([0.0]),
    )
    with pytest.raises(ValueError, match="finite nonzero spacing"):
        artifacts._periodic_central_derivative(np.asarray([1.0, 2.0]), np.asarray([0.0, 0.0]), axis=0)
    with pytest.raises(ValueError, match="two-dimensional"):
        artifacts._theta_zeta_axes((2, 3, 4), n_theta=2, n_zeta=3)
    with pytest.raises(ValueError, match="theta/zeta sizes"):
        artifacts._theta_zeta_axes((4, 4), n_theta=2, n_zeta=3)

    missing_h5 = tmp_path / "missing_geometry.h5"
    with h5py.File(missing_h5, "w") as h5:
        h5["BHat"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="missing Appendix-B audit fields"):
        appendix_b_geometry_audit_from_h5(missing_h5)

    zero_dhat = tmp_path / "zero_dhat.h5"
    _write_appendix_b_geometry_fixture(zero_dhat)
    with h5py.File(zero_dhat, "a") as h5:
        del h5["DHat"]
        h5["DHat"] = np.zeros((6, 5))
    with pytest.raises(ValueError, match="zero DHat"):
        appendix_b_geometry_audit_from_h5(zero_dhat)


def test_benchmark_artifact_policy_and_file_classification_fail_closed(tmp_path: Path) -> None:
    assert benchmark_artifact_policy_errors(["not an object"], source="artifact.json") == [
        "artifact.json: artifact must be a JSON object"
    ]

    bad_payload = {
        "schema_version": True,
        "kind": "pas_runtime_benchmark",
        "plan": {
            "variant_methods": [
                {"variant": "baseline"},
                {"variant": "baseline"},
                "not a mapping",
            ],
            "gates": {
                "default_promotion_required": True,
                "baseline_elapsed_s": 0.0,
                "baseline_rss_mb": None,
                "min_runtime_speedup": 0.5,
                "min_memory_reduction": "bad",
            },
        },
        "summary": {"all_gates_passed": False},
        "results": [
            {
                "variant": "baseline",
                "status": "ok",
                "variant_provenance": "not a mapping",
                "solver_provenance": {},
                "phase_metadata": "not a list",
                "gates": {"stall": {"status": "fail"}},
            },
            {
                "variant": "baseline",
                "status": "timeout",
                "variant_provenance": {},
                "tail_metadata": "not a mapping",
            },
            "not a mapping",
        ],
    }

    errors = benchmark_artifact_policy_errors(bad_payload, source="bad_pas.json")

    assert any("schema_version must be a number" in error for error in errors)
    assert any("duplicate variant 'baseline' in plan.variant_methods" in error for error in errors)
    assert any("plan.gates.baseline_elapsed_s" in error for error in errors)
    assert any("summary.all_gates_passed" in error for error in errors)
    assert any("phase_metadata must be a list" in error for error in errors)
    assert any("gates.residual" in error for error in errors)
    assert any("tail_metadata must be a JSON object" in error for error in errors)
    assert any("results[2] must be a JSON object" in error for error in errors)
    assert any("duplicate variant 'baseline' in results" in error for error in errors)

    missing_errors = check_benchmark_artifact_file(tmp_path / "missing.json")
    assert "could not read JSON file" in missing_errors[0]
    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert "invalid JSON" in check_benchmark_artifact_file(malformed)[0]
    assert check_benchmark_artifact_files([malformed, tmp_path / "missing_again.json"])

    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    scalar_entry = classify_benchmark_artifact_file(scalar)
    assert scalar_entry.classification == artifacts.ARTIFACT_CLASS_NON_PAS
    assert scalar_entry.release_blocking is False

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"schema_version": 1, "kind": "pas_runtime_benchmark"}), encoding="utf-8")
    legacy_entry = classify_benchmark_artifact_file(legacy)
    assert legacy_entry.classification == artifacts.ARTIFACT_CLASS_LEGACY

    bad_pas = tmp_path / "bad_pas.json"
    bad_pas.write_text(json.dumps(bad_payload), encoding="utf-8")
    bad_entry = classify_benchmark_artifact_file(bad_pas)
    assert bad_entry.classification == artifacts.ARTIFACT_CLASS_RELEASE_BLOCKING
    assert bad_entry.release_blocking is True

    missing_entry = classify_benchmark_artifact_file(tmp_path / "missing_classify.json")
    assert missing_entry.classification == artifacts.ARTIFACT_CLASS_RELEASE_BLOCKING
    malformed_entry = classify_benchmark_artifact_file(malformed)
    assert malformed_entry.classification == artifacts.ARTIFACT_CLASS_RELEASE_BLOCKING

    valid_pas = tmp_path / "valid_pas.json"
    valid_pas.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "pas_runtime_benchmark",
                "plan": {"variant_methods": [{"variant": "baseline"}]},
                "results": [
                    {
                        "variant": "baseline",
                        "status": "ok",
                        "variant_provenance": {},
                        "solver_provenance": {},
                        "phase_metadata": [],
                        "tail_metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert classify_benchmark_artifact_file(valid_pas).classification == artifacts.ARTIFACT_CLASS_SCHEMA_V2

    index = index_benchmark_artifact_files([scalar, legacy, bad_pas])
    assert index.counts[artifacts.ARTIFACT_CLASS_NON_PAS] == 1
    assert index.counts[artifacts.ARTIFACT_CLASS_LEGACY] == 1
    assert index.counts[artifacts.ARTIFACT_CLASS_RELEASE_BLOCKING] == 1
    assert index.release_blocking == (bad_entry,)


def test_fortran_suite_benchmark_summary_release_policy_fail_closed() -> None:
    assert fortran_suite_benchmark_summary_errors(["not an object"], source="suite.json") == [
        "suite.json: Fortran suite benchmark summary must be a JSON object"
    ]
    assert fortran_suite_benchmark_summary_errors({"metadata": []}) == ["missing field metadata"]
    assert fortran_suite_benchmark_summary_errors(
        {
            "metadata": {
                "kind": "wrong",
                "min_fortran_runtime_s": None,
                "excluded_low_fortran_runtime_cases": "not a list",
            }
        }
    ) == [
        "field metadata.kind must be 'fortran_v3_suite_benchmark_summary'",
        "field metadata.min_fortran_runtime_s must be a finite number",
        "field metadata.excluded_low_fortran_runtime_cases must be a list",
        "missing field reports",
    ]

    payload = {
        "metadata": {
            "kind": "fortran_v3_suite_benchmark_summary",
            "min_fortran_runtime_s": 10.0,
            "reported_case_counts": {"cpu": 2, "gpu": 1},
            "excluded_low_fortran_runtime_cases": [
                {"case": "", "fortran_runtime_s": 20.0},
                "not a row",
            ],
        },
        "reports": {
            "cpu": {
                "total_cases": 1,
                "parity_ok_cases": 0,
                "strict_mismatch_total": 1,
                "cold_runtime_ratio_summary": {"count": 0},
                "active_memory_ratio_summary": {},
                "warm_or_logged_runtime_source_counts": {"unknown": -1},
                "fastest_jax_vs_fortran_cases": [
                    {
                        "case": "",
                        "status": "jax_error",
                        "fortran_runtime_s": 5.0,
                        "jax_runtime_s_cold": 0.0,
                        "warm_or_logged_runtime_s": None,
                        "warm_or_logged_runtime_source": "bad",
                        "active_jax_memory_mb": 0.0,
                        "runtime_ratio": 2.0,
                    },
                    {
                        "case": "later",
                        "status": "parity_ok",
                        "fortran_runtime_s": 20.0,
                        "jax_runtime_s_cold": 2.0,
                        "warm_or_logged_runtime_s": 1.0,
                        "warm_or_logged_runtime_source": "jax_runtime_s_warm",
                        "jax_runtime_s_warm": 1.0,
                        "active_jax_memory_mb": 3.0,
                        "jax_incremental_max_rss_mb": 4.0,
                        "runtime_ratio": 1.0,
                    },
                ],
                "slowest_jax_vs_fortran_cases": ["not a row"],
                "highest_active_jax_memory_cases": [
                    {
                        "case": "mem",
                        "status": "parity_ok",
                        "fortran_runtime_s": 20.0,
                        "jax_runtime_s_cold": 2.0,
                        "warm_or_logged_runtime_s": 1.0,
                        "warm_or_logged_runtime_source": "jax_logged_elapsed_s",
                        "jax_logged_elapsed_s": 1.0,
                        "jax_incremental_max_rss_mb": 4.0,
                        "active_jax_memory_mb": 4.0,
                    }
                ],
            },
            "gpu": {"total_cases": 0, "parity_ok_cases": 0, "strict_mismatch_total": 0},
        },
        "canonical_rows": {
            "cpu": [
                {
                    "case": "b",
                    "status": "parity_ok",
                    "fortran_runtime_s": 20.0,
                    "jax_runtime_s_cold": 2.0,
                    "warm_or_logged_runtime_s": 1.0,
                    "warm_or_logged_runtime_source": "jax_logged_elapsed_s",
                    "jax_logged_elapsed_s": 1.0,
                    "active_jax_memory_mb": 4.0,
                    "jax_incremental_max_rss_mb": 4.0,
                },
                {
                    "case": "b",
                    "status": "parity_ok",
                    "fortran_runtime_s": 20.0,
                    "jax_runtime_s_cold": 2.0,
                    "warm_or_logged_runtime_s": 1.0,
                    "warm_or_logged_runtime_source": "jax_logged_elapsed_s",
                    "jax_logged_elapsed_s": 1.0,
                    "active_jax_memory_mb": 4.0,
                    "jax_incremental_max_rss_mb": 4.0,
                },
            ],
            "gpu": "not a list",
        },
    }

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert any("excluded_low_fortran_runtime_cases[0].case" in error for error in errors)
    assert any("excluded_low_fortran_runtime_cases[1] must be a JSON object" in error for error in errors)
    assert any("total_cases must match metadata.reported_case_counts.cpu" in error for error in errors)
    assert any("parity_ok_cases must equal total_cases" in error for error in errors)
    assert any("strict_mismatch_total must be 0" in error for error in errors)
    assert any("warm_or_logged_runtime_ratio_summary" in error for error in errors)
    assert any("unknown source" in error for error in errors)
    assert any("must sum to total_cases" in error for error in errors)
    assert any("fastest_jax_vs_fortran_cases must be sorted ascending" in error for error in errors)
    assert any("fortran_runtime_s must be >= 10" in error for error in errors)
    assert any("warm_or_logged_runtime_source must be one of" in error for error in errors)
    assert any("active_jax_memory_mb must use" in error for error in errors)
    assert any("slowest_jax_vs_fortran_cases[0] must be a JSON object" in error for error in errors)
    assert any("canonical_rows.gpu must be a list" in error for error in errors)
    assert any("missing field metadata.canonical_case_order" in error for error in errors)


def test_validation_artifact_research_and_collisionality_fail_closed_edges() -> None:
    assert artifacts.research_lane_completion_errors({"schema_version": 1, "lanes": "not a list"}) == [
        "field lanes must be a non-empty list"
    ]
    errors = artifacts.research_lane_completion_errors(
        {
            "schema_version": 1,
            "lanes": [
                {
                    "title": "Missing id",
                    "status": "active",
                    "before_percent": 10,
                    "current_percent": 20,
                    "target_percent": 15,
                    "evidence": [{"path": "docs/performance_techniques.rst", "claim": "ok"}],
                    "gates": "not a list",
                    "next_actions": ["ok"],
                }
            ],
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert "lanes[0].id must be a non-empty string" in errors
    assert "lanes[0]: current_percent must be <= target_percent" in errors
    assert "lanes[0]: field gates must be a non-empty list" in errors

    records = [
        artifacts.CollisionalityRecord("Fokker-Planck", 1.0, np.eye(3)),
        artifacts.CollisionalityRecord("PAS", 1.0, 2.0 * np.eye(3)),
    ]
    nuprime, l11 = artifacts.l11_abs_series(records, label="PAS")
    np.testing.assert_allclose(nuprime, np.asarray([1.0]))
    np.testing.assert_allclose(l11, np.asarray([2.0]))
    with pytest.raises(ValueError, match="n_fit"):
        collisionality_power_law_slope(records, label="PAS", element=(0, 0), n_fit=1)
    with pytest.raises(ValueError, match="Need at least 2 records"):
        collisionality_power_law_slope(records, label="PAS", element=(0, 0), n_fit=2)
