from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.benchmark_rhs1_pas_matrixfree import (
    _build_parser,
    _child_payload,
    _run_child,
    build_plan,
    build_probe_cases,
    main,
)


def _parse_args(argv: list[str]):
    return _build_parser().parse_args(argv)


def test_dry_run_writes_json_schema_without_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "rhs1_pas_probe_plan.json"

    def fail_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not launch child subprocesses")

    monkeypatch.setattr(subprocess, "run", fail_run)

    rc = main(
        [
            "--dry-run",
            "--out",
            str(out),
            "--systems",
            "diagonal_keep",
            "zero_update_reject",
            "--metadata-inputs",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["kind"] == "rhs1_pas_matrixfree_probe"
    assert payload["results"] == []
    assert payload["summary"]["lane_state"] == "harness_only_no_solver_default_change"
    assert payload["summary"]["production_floor_probe_ready"] is False
    assert set(payload["plan"]["gates"]) == {"keep", "reject"}
    assert payload["plan"]["cases"][0]["case_id"] == "diagonal_keep"
    assert payload["plan"]["cases"][1]["expected_gate"] == "reject"


def test_child_payload_covers_keep_and_reject_gates() -> None:
    args = _parse_args(
        [
            "--systems",
            "diagonal_keep",
            "zero_update_reject",
            "--metadata-inputs",
        ]
    )
    cases = {case["case_id"]: case for case in build_probe_cases(args)}

    keep = _child_payload(cases["diagonal_keep"])
    reject = _child_payload(cases["zero_update_reject"])

    assert keep["status"] == "ok"
    assert keep["gate"] == "keep"
    assert keep["gate_reason"] == "accepted"
    assert keep["meets_expected_gate"] is True
    assert reject["status"] == "ok"
    assert reject["gate"] == "reject"
    assert reject["gate_reason"] == "insufficient-residual-improvement"
    assert reject["meets_expected_gate"] is True
    json.dumps({"results": [keep, reject]}, allow_nan=False)


def test_child_payload_rejects_nonfinite_candidate_with_json_safe_history() -> None:
    args = _parse_args(
        [
            "--systems",
            "nonfinite_candidate_reject",
            "--metadata-inputs",
            "--block-size",
            "2",
        ]
    )
    case = build_probe_cases(args)[0]

    row = _child_payload(case)

    assert row["status"] == "ok"
    assert row["gate"] == "reject"
    assert row["gate_reason"] == "nonfinite-candidate-residual"
    assert row["residual_history"][-1] is None
    assert row["residual_history_nonfinite_count"] == 1
    assert row["meets_expected_gate"] is True
    json.dumps(row, allow_nan=False)


def test_run_child_records_timeout_as_reject_gate(monkeypatch) -> None:
    args = _parse_args(
        [
            "--systems",
            "timeout_sleep",
            "--metadata-inputs",
            "--timeout-s",
            "0.25",
        ]
    )
    case = build_probe_cases(args)[0]

    def fake_run(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "child"],
            timeout=kwargs["timeout"],
            output="started",
            stderr="still running",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    row = _run_child(args, case)

    assert row["status"] == "timeout"
    assert row["gate"] == "reject"
    assert row["gate_reason"] == "timeout"
    assert row["meets_expected_gate"] is True
    assert row["stdout_tail"] == "started"
    assert row["stderr_tail"] == "still running"


def test_build_plan_includes_capped_geometry_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        """&geometryParameters
  geometryScheme = 11
  equilibriumFile = 'w7x-test.bc'
/
&physicsParameters
  collisionOperator = 1
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 19
  Nzeta = 59
  Nxi = 60
  Nx = 5
/
&speciesParameters
  Zs = 1 -1
/
"""
    )
    args = _parse_args(
        [
            "--dry-run",
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(input_path),
            "--max-size",
            "40",
        ]
    )

    plan = build_plan(args)

    metadata_case = plan["cases"][1]
    source = metadata_case["source_metadata"]
    assert metadata_case["system_kind"] == "metadata_coupled_jacobi"
    assert metadata_case["size"] == 40
    assert source["geometry_scheme"] == 11
    assert source["collision_operator"] == 1
    assert source["include_phi1"] is False
    assert source["equilibrium_file"] == "w7x-test.bc"
    assert source["species_count"] == 2
    assert source["estimated_full_unknowns"] == 19 * 59 * 60 * 5 * 2
