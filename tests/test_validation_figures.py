from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import dkx.validation.artifacts as validation_figures
from dkx.validation.artifacts import (
    build_simakov_helander_high_nu_panel,
    build_w7x_ambipolar_root_provenance_panel,
)


def _ambipolar_payload(
    *,
    root_er: float = -1.5,
    root_type: str = "ion",
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    # Intentionally unsorted: the helper should sort by Er before computing gates.
    runs = [
        {"er": 1.0, "radial_current": 2.5},
        {"er": -3.0, "radial_current": -1.5},
        {"er": -1.0, "radial_current": 0.5},
        {"er": -2.0, "radial_current": -0.5},
        {"er": 0.0, "radial_current": 1.5},
    ]
    return {
        "metadata": {
            "kind": "w7x_ambipolar_validation_scaffold",
            "validation_scope": "w7x_like_scaffold",
        },
        "provenance": dict(provenance or {}),
        "runs": runs,
        "ambipolar": {
            "roots_er": [float(root_er)],
            "root_types": [root_type],
        },
    }


def _complete_provenance() -> dict[str, object]:
    return {
        "equilibrium_source": "wout_w7x_reference.nc",
        "profile_source": "published W7-X ion-root profile table",
        "configuration_or_shot": "W7-X reference ion-root discharge",
        "literature_reference": "https://doi.org/10.1088/1741-4326/ab6ea8",
    }


def _complete_simakov_helander_provenance() -> dict[str, object]:
    return {
        "geometry_source": "checked-in Appendix-B geometry fixture",
        "scan_source": "high-nu transport scan artifact",
        "analytic_limit_reference": "Simakov-Helander high-collisionality limit",
        "normalization_reference": "SFINCS 2014 Appendix-B normalization audit",
    }


def _simakov_helander_payload(
    *,
    provenance: dict[str, object] | None = None,
    relative_errors: list[float] | None = None,
    nuprime: list[float] | None = None,
) -> dict[str, object]:
    nu_grid = nuprime or [10.0, 30.0, 100.0, 300.0, 1000.0]
    errors = relative_errors or [0.5, 0.2, 0.08, 0.02, 0.004]
    assert len(nu_grid) == len(errors)
    analytic_limit = 2.0
    runs = [
        {
            "nuprime": float(nu),
            "value": float(analytic_limit * (1.0 + error)),
            "analytic_limit": analytic_limit,
        }
        for nu, error in zip(nu_grid, errors, strict=True)
    ]
    # Intentionally unsorted: the helper should sort before fitting the tail.
    return {
        "metadata": {
            "kind": "simakov_helander_limit_audit",
            "validation_scope": "high_nu_panel_fixture",
        },
        "provenance": dict(provenance or _complete_simakov_helander_provenance()),
        "runs": list(reversed(runs)),
    }


def test_w7x_ambipolar_panel_builds_sorted_zero_bracket_and_deferred_label() -> None:
    panel = build_w7x_ambipolar_root_provenance_panel(_ambipolar_payload())

    assert panel["metadata"]["kind"] == "w7x_ambipolar_root_provenance_panel"
    assert panel["metadata"]["manuscript_lane"] == "w7x_ambipolar_er_validation"
    assert panel["metadata"]["validation_state"] == "scaffold_deferred"
    assert panel["metadata"]["figure_label"].startswith("DEFERRED SCAFFOLD")
    assert panel["metadata"]["publication_figure"] == {
        "claim_status": "proxy_or_deferred",
        "artifact_class": "deferred_w7x_ambipolar_scaffold",
        "checked_in_converged_artifact": False,
        "ready_for_physics_validation_claim": False,
        "manuscript_label": "deferred W7-X ambipolar-root scaffold",
    }
    assert panel["metadata"]["deferred_reason_codes"] == [
        "incomplete_provenance",
        "source_artifact_not_checked_in",
    ]
    assert panel["scan"]["er"] == [-3.0, -2.0, -1.0, 0.0, 1.0]

    brackets = panel["zero_crossing_brackets"]
    assert len(brackets) == 1
    assert brackets[0]["er_min"] == -2.0
    assert brackets[0]["er_max"] == -1.0
    assert brackets[0]["linear_interpolated_root_er"] == pytest.approx(-1.5)
    assert brackets[0]["local_radial_current_slope"] == pytest.approx(1.0)

    root = panel["roots"][0]
    assert root["matching_bracket_index"] == 1
    assert root["linear_interpolated_root_er"] == pytest.approx(-1.5)
    assert root["root_to_linear_delta"] == pytest.approx(0.0)
    assert root["local_radial_current_slope"] == pytest.approx(1.0)

    gates = panel["gates"]
    assert gates["finite_er_current_series"] is True
    assert gates["radial_current_brackets_zero"] is True
    assert gates["finite_ambipolar_roots"] is True
    assert gates["root_inside_scanned_er_range"] is True
    assert gates["root_consistent_with_sign_change"] is True
    assert gates["ambipolar_root_slope_resolved"] is True
    assert gates["ion_root_candidate"] is True
    assert gates["current_trend_monotonic"] is True
    assert gates["provenance_complete"] is False
    assert gates["source_artifact_checked_in"] is False
    assert gates["ready_for_literature_claim"] is False
    assert gates["checked_in_converged_artifact"] is False

    assert panel["provenance"]["present_required_fields"] == 0
    assert panel["provenance"]["total_required_fields"] == 4
    assert panel["provenance"]["completeness_score"] == 0.0
    assert panel["deferred_reasons"] == [
        {
            "code": "incomplete_provenance",
            "gate": "provenance_complete",
            "missing_fields": [
                "equilibrium_source",
                "profile_source",
                "configuration_or_shot",
                "literature_reference",
            ],
            "completeness_score": 0.0,
            "message": "Required W7-X provenance fields are incomplete.",
        },
        {
            "code": "source_artifact_not_checked_in",
            "gate": "source_artifact_checked_in",
            "artifact_status": "missing",
            "message": "The source JSON is not a matching Git-tracked W7-X ambipolar artifact.",
        },
    ]


def test_w7x_ambipolar_panel_records_provenance_but_keeps_untracked_artifact_deferred(tmp_path: Path) -> None:
    source_artifact = tmp_path / "dkx_w7x_ambipolar_validation_summary.json"
    payload = _ambipolar_payload(provenance=_complete_provenance())
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")

    panel = build_w7x_ambipolar_root_provenance_panel(
        payload,
        source_artifact=source_artifact,
    )

    assert panel["provenance"]["complete"] is True
    assert panel["provenance"]["missing_fields"] == []
    assert panel["provenance"]["present_required_fields"] == 4
    assert panel["provenance"]["total_required_fields"] == 4
    assert panel["provenance"]["completeness_score"] == 1.0
    assert panel["provenance"]["fields"]["profile_source"] == "published W7-X ion-root profile table"
    assert panel["source_artifact"]["exists"] is True
    assert panel["source_artifact"]["looks_like_w7x_ambipolar_artifact"] is True
    assert panel["source_artifact"]["payload_matches"] is True
    assert panel["source_artifact"]["checked_in"] is False
    assert panel["source_artifact"]["status"] == "untracked_or_outside_git"
    assert panel["gates"]["provenance_complete"] is True
    assert panel["gates"]["source_artifact_checked_in"] is False
    assert panel["gates"]["ready_for_literature_claim"] is False
    assert panel["metadata"]["validation_state"] == "scaffold_deferred"
    assert panel["metadata"]["deferred_reason_codes"] == ["source_artifact_not_checked_in"]
    assert panel["deferred_reasons"] == [
        {
            "code": "source_artifact_not_checked_in",
            "gate": "source_artifact_checked_in",
            "artifact_status": "untracked_or_outside_git",
            "message": "The source JSON is not a matching Git-tracked W7-X ambipolar artifact.",
        }
    ]


def test_w7x_ambipolar_panel_can_be_literature_ready_with_checked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _ambipolar_payload(provenance=_complete_provenance())
    source_artifact = tmp_path / "dkx_w7x_ambipolar_validation_summary.json"
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    panel = build_w7x_ambipolar_root_provenance_panel(
        payload,
        source_artifact=source_artifact,
    )

    assert panel["metadata"]["validation_state"] == "artifact_backed_literature_ready"
    assert panel["metadata"]["figure_label"].startswith("ARTIFACT-BACKED")
    assert panel["metadata"]["publication_figure"] == {
        "claim_status": "checked_in_converged_artifact",
        "artifact_class": "checked_in_w7x_ambipolar_literature_artifact",
        "checked_in_converged_artifact": True,
        "ready_for_physics_validation_claim": True,
        "manuscript_label": "checked-in W7-X ambipolar-root validation",
    }
    assert panel["source_artifact"]["status"] == "checked_in"
    assert panel["source_artifact"]["tracked"] is True
    assert panel["source_artifact"]["payload_matches"] is True
    assert panel["source_artifact"]["checked_in"] is True
    assert panel["gates"]["provenance_complete"] is True
    assert panel["gates"]["source_artifact_checked_in"] is True
    assert panel["gates"]["ready_for_literature_claim"] is True
    assert panel["gates"]["checked_in_converged_artifact"] is True
    assert panel["metadata"]["deferred_reason_codes"] == []
    assert panel["deferred_reasons"] == []


def test_w7x_ambipolar_panel_fails_closed_for_tracked_payload_or_name_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _ambipolar_payload(provenance=_complete_provenance())
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    wrong_name = tmp_path / "local_scan_summary.json"
    wrong_name.write_text(json.dumps(payload), encoding="utf-8")
    wrong_name_panel = build_w7x_ambipolar_root_provenance_panel(
        payload,
        source_artifact=wrong_name,
    )
    assert wrong_name_panel["source_artifact"]["tracked"] is True
    assert wrong_name_panel["source_artifact"]["looks_like_w7x_ambipolar_artifact"] is False
    assert wrong_name_panel["source_artifact"]["status"] == "tracked_non_w7x_ambipolar_artifact"
    assert wrong_name_panel["gates"]["source_artifact_checked_in"] is False
    assert wrong_name_panel["metadata"]["deferred_reason_codes"] == ["source_artifact_not_checked_in"]

    mismatched_payload = tmp_path / "dkx_w7x_ambipolar_validation_summary.json"
    mismatched_payload.write_text(
        json.dumps({**payload, "runs": payload["runs"][:-1]}),
        encoding="utf-8",
    )
    mismatch_panel = build_w7x_ambipolar_root_provenance_panel(
        payload,
        source_artifact=mismatched_payload,
    )
    assert mismatch_panel["source_artifact"]["tracked"] is True
    assert mismatch_panel["source_artifact"]["looks_like_w7x_ambipolar_artifact"] is True
    assert mismatch_panel["source_artifact"]["payload_matches"] is False
    assert mismatch_panel["source_artifact"]["status"] == "tracked_w7x_ambipolar_artifact_payload_mismatch"
    assert mismatch_panel["gates"]["source_artifact_checked_in"] is False
    assert mismatch_panel["metadata"]["deferred_reason_codes"] == ["source_artifact_not_checked_in"]


def test_w7x_ambipolar_panel_rejects_root_not_supported_by_current_bracket() -> None:
    payload = _ambipolar_payload(root_er=0.8, root_type="electron")
    panel = build_w7x_ambipolar_root_provenance_panel(payload)

    root = panel["roots"][0]
    assert root["inside_scanned_er_range"] is True
    assert root["matching_bracket_index"] is None
    assert root["ion_root_candidate"] is False
    assert panel["gates"]["radial_current_brackets_zero"] is True
    assert panel["gates"]["root_consistent_with_sign_change"] is False
    assert panel["gates"]["ambipolar_root_slope_resolved"] is False
    assert panel["gates"]["ion_root_candidate"] is False
    assert panel["gates"]["ready_for_literature_claim"] is False

    reason_codes = [reason["code"] for reason in panel["deferred_reasons"]]
    assert reason_codes == [
        "root_not_supported_by_sign_change",
        "ambipolar_root_slope_unresolved",
        "ion_root_candidate_missing",
        "incomplete_provenance",
        "source_artifact_not_checked_in",
    ]
    assert panel["metadata"]["deferred_reason_codes"] == reason_codes


def test_w7x_ambipolar_panel_fails_closed_for_degenerate_zero_bracket_and_metadata() -> None:
    payload = {
        "metadata": "local scratch metadata",
        "provenance": "not a provenance mapping",
        "runs": [
            {"er": -1.0, "radial_current": 0.0},
            {"er": 1.0, "radial_current": 0.0},
        ],
        "ambipolar": {"roots_er": [0.0], "root_types": []},
    }

    panel = build_w7x_ambipolar_root_provenance_panel(payload)

    bracket = panel["zero_crossing_brackets"][0]
    assert bracket["linear_interpolated_root_er"] is None
    assert bracket["local_radial_current_slope"] == 0.0
    root = panel["roots"][0]
    assert root["root_type"] == "unknown"
    assert root["linear_interpolated_root_er"] is None
    assert root["root_to_linear_delta"] is None
    assert root["ion_root_candidate"] is False
    assert panel["metadata"]["source_kind"] is None
    assert panel["metadata"]["source_validation_scope"] is None
    assert panel["provenance"]["present_required_fields"] == 0
    assert panel["gates"]["ambipolar_root_slope_resolved"] is False
    assert panel["gates"]["ion_root_candidate"] is False
    assert panel["metadata"]["deferred_reason_codes"] == [
        "ambipolar_root_slope_unresolved",
        "ion_root_candidate_missing",
        "incomplete_provenance",
        "source_artifact_not_checked_in",
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"runs": "not a scan", "ambipolar": {"roots_er": [0.0]}}, "runs"),
        ({"runs": ["not a mapping"], "ambipolar": {"roots_er": [0.0]}}, "run must be a mapping"),
        (
            {"runs": [{"er": 0.0}], "ambipolar": {"roots_er": [0.0]}},
            "finite er and radial_current",
        ),
        (
            {"runs": [{"er": 0.0, "radial_current": float("nan")}], "ambipolar": {"roots_er": [0.0]}},
            "finite er and radial_current",
        ),
        (
            {"runs": [{"er": 0.0, "radial_current": 1.0}], "ambipolar": {"roots_er": [0.0]}},
            "Need at least two",
        ),
        (
            {
                "runs": [
                    {"er": 0.0, "radial_current": -1.0},
                    {"er": 0.0, "radial_current": 1.0},
                ],
                "ambipolar": {"roots_er": [0.0]},
            },
            "distinct",
        ),
        ({"runs": [{"er": -1.0, "radial_current": -1.0}, {"er": 1.0, "radial_current": 1.0}]}, "ambipolar"),
        (
            {
                "runs": [{"er": -1.0, "radial_current": -1.0}, {"er": 1.0, "radial_current": 1.0}],
                "ambipolar": {"roots_er": "0.0"},
            },
            "roots_er",
        ),
        (
            {
                "runs": [{"er": -1.0, "radial_current": -1.0}, {"er": 1.0, "radial_current": 1.0}],
                "ambipolar": {"roots_er": ["not numeric"]},
            },
            "numeric",
        ),
    ],
)
def test_w7x_ambipolar_panel_rejects_malformed_payloads(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_w7x_ambipolar_root_provenance_panel(payload)


def test_w7x_ambipolar_artifact_summary_reports_missing_named_artifact(tmp_path: Path) -> None:
    payload = _ambipolar_payload(provenance=_complete_provenance())
    missing_artifact = tmp_path / "dkx_w7x_ambipolar_missing.json"

    panel = build_w7x_ambipolar_root_provenance_panel(
        payload,
        source_artifact=missing_artifact,
    )

    assert panel["source_artifact"]["exists"] is False
    assert panel["source_artifact"]["status"] == "missing"
    assert panel["source_artifact"]["checked_in"] is False


def test_simakov_helander_high_nu_panel_can_be_literature_ready_with_checked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _simakov_helander_payload()
    source_artifact = tmp_path / "dkx_simakov_helander_high_nu_panel.json"
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    panel = build_simakov_helander_high_nu_panel(payload, source_artifact=source_artifact)

    assert panel["metadata"]["kind"] == "simakov_helander_high_nu_panel"
    assert panel["metadata"]["manuscript_lane"] == "simakov_helander_high_collisionality_limit"
    assert panel["metadata"]["validation_state"] == "artifact_backed_literature_ready"
    assert panel["metadata"]["figure_label"].startswith("ARTIFACT-BACKED")
    assert panel["metadata"]["publication_figure"] == {
        "claim_status": "checked_in_converged_artifact",
        "artifact_class": "checked_in_simakov_helander_high_nu_artifact",
        "checked_in_converged_artifact": True,
        "ready_for_physics_validation_claim": True,
        "manuscript_label": "checked-in Simakov-Helander high-nu analytic-limit panel",
    }
    assert panel["metadata"]["deferred_reason_codes"] == []
    assert panel["scan"]["nuprime"] == [10.0, 30.0, 100.0, 300.0, 1000.0]
    assert panel["scan"]["relative_error"] == pytest.approx([0.5, 0.2, 0.08, 0.02, 0.004])

    approach = panel["asymptotic_approach"]
    assert approach["tail_nuprime"] == [100.0, 300.0, 1000.0]
    assert approach["tail_relative_error"] == pytest.approx([0.08, 0.02, 0.004])
    assert approach["slope"] == pytest.approx(-1.3015997080955761, rel=1.0e-12)
    assert approach["relative_error_nonincreasing"] is True
    assert approach["tail_error_reduction_factor"] == pytest.approx(20.0)

    high_nu = panel["high_nu_range"]
    assert high_nu["observed_high_nu_points"] == 3
    assert high_nu["observed_high_nu_min"] == 100.0
    assert high_nu["observed_high_nu_max"] == 1000.0
    assert high_nu["observed_high_nu_decades"] == pytest.approx(1.0)

    gates = panel["gates"]
    assert gates["finite_positive_nuprime_scan"] is True
    assert gates["finite_nonzero_analytic_limits"] is True
    assert gates["finite_asymptotic_error"] is True
    assert gates["enough_tail_points_for_slope"] is True
    assert gates["asymptotic_error_decreases"] is True
    assert gates["asymptotic_slope_matches_expected"] is True
    assert gates["high_nu_range_reaches_threshold"] is True
    assert gates["high_nu_tail_has_enough_points"] is True
    assert gates["high_nu_tail_spans_required_decades"] is True
    assert gates["provenance_complete"] is True
    assert gates["source_artifact_checked_in"] is True
    assert gates["ready_for_literature_claim"] is True
    assert gates["checked_in_converged_artifact"] is True
    assert panel["deferred_reasons"] == []


def test_simakov_helander_high_nu_panel_defaults_metadata_and_missing_artifact() -> None:
    payload = _simakov_helander_payload()
    payload["metadata"] = ["not", "a", "mapping"]

    panel = build_simakov_helander_high_nu_panel(payload)

    assert panel["metadata"]["source_kind"] is None
    assert panel["metadata"]["source_validation_scope"] is None
    assert panel["source_artifact"]["status"] == "missing"
    assert panel["metadata"]["deferred_reason_codes"] == ["source_artifact_not_checked_in"]


def test_simakov_helander_high_nu_panel_defers_when_high_nu_range_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _simakov_helander_payload(nuprime=[1.0, 3.0, 10.0, 30.0, 90.0])
    source_artifact = tmp_path / "dkx_simakov_helander_high_nu_panel.json"
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    panel = build_simakov_helander_high_nu_panel(payload, source_artifact=source_artifact)

    assert panel["metadata"]["validation_state"] == "scaffold_deferred"
    assert panel["gates"]["high_nu_range_reaches_threshold"] is False
    assert panel["gates"]["high_nu_tail_has_enough_points"] is False
    assert panel["gates"]["high_nu_tail_spans_required_decades"] is False
    assert panel["gates"]["ready_for_literature_claim"] is False
    assert panel["metadata"]["deferred_reason_codes"] == [
        "high_nu_threshold_not_reached",
        "insufficient_high_nu_points",
        "high_nu_span_too_narrow",
    ]


def test_simakov_helander_high_nu_panel_defers_on_wrong_slope_and_trend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _simakov_helander_payload(relative_errors=[0.5, 0.2, 0.02, 0.08, 0.1])
    source_artifact = tmp_path / "dkx_simakov_helander_high_nu_panel.json"
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    panel = build_simakov_helander_high_nu_panel(payload, source_artifact=source_artifact)

    assert panel["asymptotic_approach"]["slope"] > 0.0
    assert panel["gates"]["asymptotic_error_decreases"] is False
    assert panel["gates"]["asymptotic_slope_matches_expected"] is False
    assert panel["gates"]["ready_for_literature_claim"] is False
    assert panel["metadata"]["deferred_reason_codes"] == [
        "asymptotic_error_not_monotone",
        "asymptotic_slope_outside_expected_range",
    ]


def test_simakov_helander_high_nu_panel_reports_provenance_gaps_and_untracked_artifact(
    tmp_path: Path,
) -> None:
    payload = _simakov_helander_payload(provenance={"scan_source": "local scratch scan"})
    source_artifact = tmp_path / "dkx_simakov_helander_high_nu_panel.json"
    source_artifact.write_text(json.dumps(payload), encoding="utf-8")

    panel = build_simakov_helander_high_nu_panel(payload, source_artifact=source_artifact)

    assert panel["provenance"]["complete"] is False
    assert panel["provenance"]["present_required_fields"] == 1
    assert panel["provenance"]["total_required_fields"] == 4
    assert panel["provenance"]["completeness_score"] == 0.25
    assert panel["provenance"]["missing_fields"] == [
        "geometry_source",
        "analytic_limit_reference",
        "normalization_reference",
    ]
    assert panel["source_artifact"]["exists"] is True
    assert panel["source_artifact"]["looks_like_simakov_helander_artifact"] is True
    assert panel["source_artifact"]["payload_matches"] is True
    assert panel["source_artifact"]["checked_in"] is False
    assert panel["source_artifact"]["status"] == "untracked_or_outside_git"
    assert panel["gates"]["provenance_complete"] is False
    assert panel["gates"]["source_artifact_checked_in"] is False
    assert panel["gates"]["ready_for_literature_claim"] is False
    assert panel["metadata"]["deferred_reason_codes"] == [
        "incomplete_provenance",
        "source_artifact_not_checked_in",
    ]
    assert panel["deferred_reasons"] == [
        {
            "code": "incomplete_provenance",
            "gate": "provenance_complete",
            "missing_fields": [
                "geometry_source",
                "analytic_limit_reference",
                "normalization_reference",
            ],
            "completeness_score": 0.25,
            "message": "Required Simakov-Helander provenance fields are incomplete.",
        },
        {
            "code": "source_artifact_not_checked_in",
            "gate": "source_artifact_checked_in",
            "artifact_status": "untracked_or_outside_git",
            "message": "The source JSON is not a matching Git-tracked Simakov-Helander artifact.",
        },
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"runs": "not a scan"}, "runs"),
        ({"runs": ["not a mapping"]}, "run must be a mapping"),
        ({"runs": [{"nuprime": 1.0, "value": 2.0}]}, "nuprime, value, and analytic_limit"),
        (
            {"runs": [{"nuprime": float("nan"), "value": 2.0, "analytic_limit": 1.0}]},
            "positive finite nuprime",
        ),
        (
            {"runs": [{"nuprime": 1.0, "value": 2.0, "analytic_limit": 0.0}]},
            "positive finite nuprime",
        ),
        (
            {"runs": [{"nuprime": 1.0, "value": 2.0, "analytic_limit": 1.0}]},
            "Need at least two",
        ),
        (
            {
                "runs": [
                    {"nuprime": 1.0, "value": 2.0, "analytic_limit": 1.0},
                    {"nuprime": 1.0, "value": 1.5, "analytic_limit": 1.0},
                ],
            },
            "distinct",
        ),
    ],
)
def test_simakov_helander_high_nu_panel_rejects_malformed_payloads(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_simakov_helander_high_nu_panel(payload)


def test_simakov_helander_high_nu_panel_rejects_invalid_tail_fit() -> None:
    with pytest.raises(ValueError, match="n_tail_fit"):
        build_simakov_helander_high_nu_panel(_simakov_helander_payload(), n_tail_fit=1)


def test_simakov_helander_artifact_summary_fail_closed_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _simakov_helander_payload()
    monkeypatch.setattr(validation_figures, "_is_git_tracked_file", lambda path: True)

    wrong_name = tmp_path / "local_high_nu_scan.json"
    wrong_name.write_text(json.dumps(payload), encoding="utf-8")
    wrong_name_panel = build_simakov_helander_high_nu_panel(payload, source_artifact=wrong_name)
    assert wrong_name_panel["source_artifact"]["tracked"] is True
    assert wrong_name_panel["source_artifact"]["looks_like_simakov_helander_artifact"] is False
    assert wrong_name_panel["source_artifact"]["status"] == "tracked_non_simakov_helander_artifact"

    mismatched_payload = tmp_path / "dkx_simakov_helander_high_nu_panel.json"
    mismatched_payload.write_text(
        json.dumps({**payload, "runs": payload["runs"][:-1]}),
        encoding="utf-8",
    )
    mismatch_panel = build_simakov_helander_high_nu_panel(payload, source_artifact=mismatched_payload)
    assert mismatch_panel["source_artifact"]["tracked"] is True
    assert mismatch_panel["source_artifact"]["looks_like_simakov_helander_artifact"] is True
    assert mismatch_panel["source_artifact"]["payload_matches"] is False
    assert mismatch_panel["source_artifact"]["status"] == "tracked_simakov_helander_artifact_payload_mismatch"

    missing_artifact = tmp_path / "dkx_simakov_helander_missing.json"
    missing_panel = build_simakov_helander_high_nu_panel(payload, source_artifact=missing_artifact)
    assert missing_panel["source_artifact"]["exists"] is False
    assert missing_panel["source_artifact"]["status"] == "missing"


def test_private_validation_helpers_cover_git_and_tail_fail_closed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_json = tmp_path / "broken.json"
    malformed_json.write_text("{not valid json", encoding="utf-8")
    assert validation_figures._json_payload_matches(path=malformed_json, payload={}) is False

    outside_git_file = tmp_path / "outside.json"
    outside_git_file.write_text("{}", encoding="utf-8")
    assert validation_figures._find_git_root(tmp_path) is None
    assert validation_figures._is_git_tracked_file(outside_git_file) is False

    git_root = tmp_path / "repo"
    nested_file = git_root / "nested" / "data.json"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("{}", encoding="utf-8")
    (git_root / ".git").mkdir()

    def _raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="git ls-files", timeout=2.0)

    monkeypatch.setattr(validation_figures.subprocess, "run", _raise_timeout)
    assert validation_figures._is_git_tracked_file(nested_file) is False

    tail = validation_figures._tail_asymptotic_metadata(
        nuprime=np.asarray([1.0]),
        relative_error=np.asarray([0.25]),
        n_tail_fit=1,
        monotonic_tolerance=0.0,
    )
    assert tail["fit_point_count"] == 1
    assert tail["slope"] is None
    assert tail["intercept"] is None
    assert tail["relative_error_nonincreasing"] is False
    assert tail["tail_error_reduction_factor"] is None

    assert validation_figures._is_monotonic(np.asarray([0.0, 1.0, 0.5])) is False


def test_private_w7x_helpers_cover_fallbacks_and_reason_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validation_figures._root_types({}, n_roots=2) == ["unknown", "unknown"]
    assert validation_figures._root_types({"ambipolar": {"root_types": "ion"}}, n_roots=1) == ["unknown"]
    assert validation_figures._root_types({"ambipolar": {"root_types": ["electron"]}}, n_roots=2) == [
        "electron",
        "unknown",
    ]

    reasons = validation_figures._deferred_reasons(
        gates={
            "ready_for_literature_claim": False,
            "finite_er_current_series": False,
            "radial_current_brackets_zero": False,
            "finite_ambipolar_roots": False,
            "root_inside_scanned_er_range": False,
            "root_consistent_with_sign_change": True,
            "ambipolar_root_slope_resolved": True,
            "ion_root_candidate": True,
            "provenance_complete": True,
            "source_artifact_checked_in": True,
        },
        provenance={"missing_fields": [], "completeness_score": 1.0},
        source_artifact={"status": "checked_in"},
    )
    assert [reason["code"] for reason in reasons] == [
        "nonfinite_or_underresolved_scan",
        "missing_zero_current_bracket",
        "missing_finite_ambipolar_root",
        "root_outside_scanned_er_range",
    ]

    tracked = tmp_path / "tracked.json"
    tracked.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_find_git_root", lambda path: tmp_path / "other_repo")
    assert validation_figures._is_git_tracked_file(tracked) is False

    repo = tmp_path / "repo"
    nested = repo / "nested" / "tracked.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(validation_figures, "_find_git_root", lambda path: repo)
    monkeypatch.setattr(
        validation_figures.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=1),
    )
    assert validation_figures._is_git_tracked_file(nested) is False
