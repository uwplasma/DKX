from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

# The panel builders and their private helpers live in tools.release.panels;
# tools.release.artifacts re-exports the public ones, but a test that reaches
# into internals should name the module that owns them.
import tools.release.panels as validation_figures
from tools.release.artifacts import (
    build_simakov_helander_high_nu_panel,
)


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



def test_private_git_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
