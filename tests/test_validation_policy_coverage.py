from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.validation.artifacts import (
    CollisionalityRecord,
    autodiff_gradient_error_summary,
    build_autodiff_sensitivity_validation_summary,
    collisionality_power_law_slope,
    filter_suite_metrics_by_fortran_runtime,
    fp_pas_l11_separation,
    load_autodiff_sensitivity_summary,
    load_suite_report,
    suite_case_metrics,
    suite_report_summary,
)


def _row(
    case: str,
    *,
    fortran_runtime_s: float | None,
    jax_runtime_s: float | None = 1.0,
    jax_runtime_s_warm: float | None = None,
    jax_logged_elapsed_s: float | None = None,
    strict_n_mismatch_physics: int = 0,
) -> dict[str, object]:
    return {
        "case": case,
        "status": "parity_ok",
        "blocker_type": "none",
        "fortran_runtime_s": fortran_runtime_s,
        "jax_runtime_s": jax_runtime_s,
        "jax_runtime_s_warm": jax_runtime_s_warm,
        "jax_logged_elapsed_s": jax_logged_elapsed_s,
        "fortran_max_rss_mb": 100.0,
        "jax_max_rss_mb": 200.0,
        "strict_n_mismatch_physics": strict_n_mismatch_physics,
    }


def test_suite_filter_uses_cpu_reference_runtime_before_gpu_fallback() -> None:
    cpu_metrics = suite_case_metrics(
        [
            _row("shared_too_small_on_cpu", fortran_runtime_s=0.5),
            _row("shared_kept_by_cpu", fortran_runtime_s=20.0),
        ]
    )
    gpu_metrics = suite_case_metrics(
        [
            _row("gpu_only_kept", fortran_runtime_s=30.0),
            _row("shared_too_small_on_cpu", fortran_runtime_s=40.0),
            _row("shared_kept_by_cpu", fortran_runtime_s=0.2),
        ]
    )

    filtered_cpu, filtered_gpu, excluded = filter_suite_metrics_by_fortran_runtime(
        cpu_metrics,
        gpu_metrics,
        min_fortran_runtime_s=10.0,
    )

    assert [metric.case for metric in filtered_cpu] == ["shared_kept_by_cpu"]
    assert [metric.case for metric in filtered_gpu] == ["gpu_only_kept", "shared_kept_by_cpu"]
    assert excluded == [{"case": "shared_too_small_on_cpu", "fortran_runtime_s": 0.5}]


def test_suite_filter_without_runtime_floor_keeps_sorted_cpu_gpu_metrics() -> None:
    cpu_metrics = suite_case_metrics(
        [
            _row("b_cpu", fortran_runtime_s=None),
            _row("a_cpu", fortran_runtime_s=0.1),
        ]
    )
    gpu_metrics = suite_case_metrics(
        [
            _row("b_gpu", fortran_runtime_s=None),
            _row("a_gpu", fortran_runtime_s=0.1),
        ]
    )

    filtered_cpu, filtered_gpu, excluded = filter_suite_metrics_by_fortran_runtime(
        cpu_metrics,
        gpu_metrics,
        min_fortran_runtime_s=None,
    )

    assert [metric.case for metric in filtered_cpu] == ["a_cpu", "b_cpu"]
    assert [metric.case for metric in filtered_gpu] == ["a_gpu", "b_gpu"]
    assert excluded == []


def test_suite_summary_prefers_warm_runtime_but_falls_back_to_logged_elapsed() -> None:
    rows = [
        _row("logged_only", fortran_runtime_s=10.0, jax_logged_elapsed_s=2.0),
        _row("warm_wins", fortran_runtime_s=10.0, jax_runtime_s_warm=1.0, jax_logged_elapsed_s=9.0),
        _row("strict_delta", fortran_runtime_s=10.0, jax_logged_elapsed_s=3.0, strict_n_mismatch_physics=2),
    ]

    payload = suite_report_summary(rows, label="CPU", n_top=3)

    assert payload["warm_or_logged_runtime_source_counts"] == {
        "jax_logged_elapsed_s": 2,
        "jax_runtime_s_warm": 1,
    }
    assert payload["warm_or_logged_runtime_ratio_summary"]["max"] == pytest.approx(0.3)
    assert payload["strict_mismatch_cases"] == 1
    assert payload["strict_mismatch_total"] == 2


def test_autodiff_validation_gates_fail_closed_on_bad_gradient_or_residual(tmp_path: Path) -> None:
    payload = build_autodiff_sensitivity_validation_summary(
        gradient_checks=[
            {
                "relative_error": 2.0e-3,
                "absolute_error": 1.0e-6,
                "primal_residual_norm": 2.0e-7,
                "adjoint_residual_norm": 5.0e-9,
            },
            {
                "relative_error": "not-a-number",
                "absolute_error": float("inf"),
                "primal_residual_norm": None,
            },
        ],
        finite_difference_sweep=[],
        geometry_sensitivity={"parameter": "Boozer harmonic"},
        cost_scaling=[],
        relative_error_gate=1.0e-4,
        residual_gate=1.0e-8,
    )

    assert payload["gradient_error_summary"]["count"] == 1
    assert payload["gates"]["gradient_relative_error_ok"] is False
    assert payload["gates"]["primal_residual_ok"] is False
    assert payload["gates"]["adjoint_residual_ok"] is True

    path = tmp_path / "autodiff_summary.json"
    path.write_text(json.dumps(payload) + "\n")
    assert load_autodiff_sensitivity_summary(path)["metadata"]["kind"] == "autodiff_sensitivity_validation"

    bad_path = tmp_path / "bad_autodiff_summary.json"
    bad_path.write_text(json.dumps({"metadata": {"kind": "wrong"}}) + "\n")
    with pytest.raises(ValueError, match="unexpected metadata.kind"):
        load_autodiff_sensitivity_summary(bad_path)


def test_autodiff_artifact_loaders_reject_non_object_and_bad_gradient_shape(tmp_path: Path) -> None:
    bad_shape_path = tmp_path / "bad_shape.json"
    bad_shape_path.write_text("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_autodiff_sensitivity_summary(bad_shape_path)

    with pytest.raises(ValueError, match="gradient_checks must be a sequence"):
        autodiff_gradient_error_summary({"gradient_checks": 7})


def test_collisionality_slope_and_fp_pas_separation_are_scale_invariants() -> None:
    records: list[CollisionalityRecord] = []
    for nuprime in (1.0, 10.0, 100.0):
        fp_l11 = -4.0 / nuprime
        pas_l11 = 0.25 * nuprime
        records.append(
            CollisionalityRecord(
                label="Fokker-Planck",
                nuprime=nuprime,
                transport_matrix=np.diag([fp_l11, 2.0 / nuprime, 0.1]),
            )
        )
        records.append(
            CollisionalityRecord(
                label="PAS",
                nuprime=nuprime,
                transport_matrix=np.diag([pas_l11, 0.5 * nuprime, 0.1]),
            )
        )

    assert collisionality_power_law_slope(records, label="Fokker-Planck", element=(0, 0)) == pytest.approx(-1.0)
    assert collisionality_power_law_slope(records, label="PAS", element=(0, 0)) == pytest.approx(1.0)

    separation = fp_pas_l11_separation(records)
    assert separation[0]["relative_to_fp"] == pytest.approx(abs(-4.0 - 0.25) / 4.0)
    assert separation[-1]["relative_to_fp"] > separation[0]["relative_to_fp"]

    with pytest.raises(ValueError, match="No collisionality records"):
        collisionality_power_law_slope(records, label="missing", element=(0, 0))


def test_load_suite_report_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "suite_report.json"
    path.write_text(json.dumps({"rows": {"case": "not-a-list"}}) + "\n")

    with pytest.raises(ValueError, match="must contain a list"):
        load_suite_report(path)


def test_load_suite_report_accepts_wrapped_rows_and_ignores_non_mapping_entries(tmp_path: Path) -> None:
    path = tmp_path / "suite_report.json"
    path.write_text(
        json.dumps({"rows": [{"case": "kept"}, ["drop"], "drop", {"case": "also_kept"}]}) + "\n"
    )

    rows = load_suite_report(path)

    assert rows == [{"case": "kept"}, {"case": "also_kept"}]
