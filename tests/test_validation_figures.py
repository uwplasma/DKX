from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfincs_jax.validation_figures import build_w7x_ambipolar_root_provenance_panel


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


def test_w7x_ambipolar_panel_builds_sorted_zero_bracket_and_deferred_label() -> None:
    panel = build_w7x_ambipolar_root_provenance_panel(_ambipolar_payload())

    assert panel["metadata"]["kind"] == "w7x_ambipolar_root_provenance_panel"
    assert panel["metadata"]["manuscript_lane"] == "w7x_ambipolar_er_validation"
    assert panel["metadata"]["validation_state"] == "scaffold_deferred"
    assert panel["metadata"]["figure_label"].startswith("DEFERRED SCAFFOLD")
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
    source_artifact = tmp_path / "sfincs_jax_w7x_ambipolar_validation_summary.json"
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
