from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.benchmark_rhs1_pas_matrixfree import (
    _build_parser,
    _child_payload,
    _run_child,
    build_artifact_probe,
    build_plan,
    build_probe_cases,
    main,
    run_bounded_real_solve_probe,
)
from scripts.benchmark_pas_tz_memory_fallback import (
    result_gates as pas_tz_result_gates,
    summarize_results as summarize_pas_tz_results,
)


def _parse_args(argv: list[str]):
    return _build_parser().parse_args(argv)


def _write_input(
    path: Path,
    *,
    geometry_scheme: int,
    collision_operator: int = 1,
    case_name: str = "case",
    equilibrium_file: str | None = None,
) -> Path:
    case_dir = path / case_name
    case_dir.mkdir(parents=True)
    equilibrium_line = f"  equilibriumFile = '{equilibrium_file}'\n" if equilibrium_file else ""
    input_path = case_dir / "input.namelist"
    input_path.write_text(
        f"""&geometryParameters
  geometryScheme = {geometry_scheme}
{equilibrium_line}/
&physicsParameters
  collisionOperator = {collision_operator}
  includePhi1 = .false.
  useDKESExBDrift = .true.
  Er = 1.25
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 9
  Nxi = 11
  Nx = 3
/
&speciesParameters
  Zs = 1 -1
/
"""
    )
    return input_path


def _write_artifact(
    path: Path,
    *,
    target_input: str,
    residual_norm: object = 1.0e-4,
    elapsed_s: float = 4.5,
    max_rss_mb: float = 123.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pas_tz_memory_fallback_benchmark",
                "plan": {"input": target_input, "variants": ["tzfft_lgmres"]},
                "results": [
                    {
                        "status": "ok",
                        "variant": "tzfft_lgmres",
                        "residual_norm": residual_norm,
                        "max_rss_mb": max_rss_mb,
                        "elapsed_s": elapsed_s,
                        "messages_tail": [
                            "solve_v3_full_system_linear_gmres: total_size=1024",
                            "solve_v3_full_system_linear_gmres: PAS constraint projection enabled (size=512/1024)",
                            "solve_v3_full_system_linear_gmres: PAS-TZ guarded minres correction accepted 2 step(s)",
                        ],
                    }
                ],
            },
            allow_nan=True,
        )
    )
    return path


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
    assert payload["summary"]["next_real_solve_recommendation"] == "hold_for_missing_or_unexpected_probe_evidence"
    assert payload["plan"]["artifact_probe"]["mode"] == "checked_in_artifact_dry_run"
    assert payload["plan"]["production_floor_preflight"]["all_required_targets_ready"] is False
    assert payload["plan"]["bounded_real_solve_probe"]["run_requested"] is False
    assert payload["plan"]["bounded_real_solve_probe"]["parent_wall_timeout_s"] <= 600.0
    assert payload["plan"]["bounded_real_solve_probe"]["total_wall_timeout_budget_s"] <= 600.0
    assert payload["plan"]["bounded_real_solve_probe"]["gates"]["max_rss_mb"] == 4096.0
    assert payload["plan"]["bounded_real_solve_probe"]["gates"]["default_promotion_required"] is False
    assert payload["plan"]["bounded_real_solve_probe"]["safety_policy"]["invalid_targets_fail_closed"] is True
    assert set(payload["plan"]["gates"]) == {"keep", "reject"}
    assert payload["plan"]["cases"][0]["case_id"] == "diagonal_keep"
    assert payload["plan"]["cases"][0]["source_type"] == "synthetic"
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
    assert keep["gate_diagnostics"]["reason"] == "accepted"
    assert keep["gate_diagnostics"]["residual_reduction"] >= keep["gate_diagnostics"]["min_residual_reduction"]
    assert keep["meets_expected_gate"] is True
    assert reject["status"] == "ok"
    assert reject["gate"] == "reject"
    assert reject["gate_reason"] == "insufficient-residual-improvement"
    assert reject["gate_diagnostics"]["reason"] == "insufficient-residual-improvement"
    assert reject["gate_diagnostics"]["candidate_residual_norm"] > reject["gate_diagnostics"]["required_residual_norm"]
    assert reject["meets_expected_gate"] is True
    json.dumps({"results": [keep, reject]}, allow_nan=False)


def test_child_payload_tiny_update_rejects_before_candidate_matvec() -> None:
    args = _parse_args(
        [
            "--systems",
            "tiny_update_reject",
            "--metadata-inputs",
            "--min-update-norm-ratio",
            "1e-8",
        ]
    )
    case = build_probe_cases(args)[0]

    row = _child_payload(case)

    assert row["status"] == "ok"
    assert row["gate"] == "reject"
    assert row["gate_reason"] == "update-norm-too-small"
    assert row["meets_expected_gate"] is True
    assert row["metrics"]["matvec_calls"] == 1
    assert row["metrics"]["correction_calls"] == 1
    assert row["gate_diagnostics"]["matrix_free_metadata"]["candidate_matvecs"] == 0


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
    assert row["gate_diagnostics"]["candidate_residual_norm"] is None
    assert row["gate_diagnostics"]["candidate_residual_finite"] is False
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
    assert source["source_type"] == "production_floor_geometry_metadata"
    assert source["production_floor_target"] == "geometry11"
    assert source["species_count"] == 2
    assert source["estimated_full_unknowns"] == 19 * 59 * 60 * 5 * 2


def test_production_floor_preflight_ready_from_metadata_and_checked_in_artifacts(
    tmp_path: Path,
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    hsx = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="HSX_PASCollisions_DKESTrajectories",
        equilibrium_file="hsx3free.bc",
    )
    geom11 = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="sfincsPaperFigure3_geometryScheme11_PASCollisions",
        equilibrium_file="w7x-sc1.bc",
    )
    artifacts = [
        _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4)),
        _write_artifact(tmp_path / "artifacts" / "hsx.json", target_input=str(hsx)),
        _write_artifact(tmp_path / "artifacts" / "geom11.json", target_input=str(geom11)),
    ]
    out = tmp_path / "probe.json"

    rc = main(
        [
            "--dry-run",
            "--out",
            str(out),
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            str(hsx),
            str(geom11),
            "--artifact-inputs",
            *(str(path) for path in artifacts),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    preflight = payload["plan"]["production_floor_preflight"]
    assert preflight["all_required_targets_ready"] is True
    assert set(preflight["ready_targets"]) == {"geometry4", "hsx", "geometry11"}
    assert payload["summary"]["production_floor_probe_ready"] is True
    assert payload["summary"]["next_real_solve_recommendation"] == "proceed_to_short_real_solve_probe"
    real_solve = payload["plan"]["bounded_real_solve_probe"]
    assert real_solve["run_requested"] is False
    assert set(real_solve["targets"]) == {"geometry4", "hsx", "geometry11"}
    assert all(target["ready"] is True for target in real_solve["targets"].values())
    assert all(target["will_run"] is False for target in real_solve["targets"].values())
    assert real_solve["selected_targets"] == ["geometry4"]
    assert real_solve["targets"]["geometry4"]["selected_by_budget"] is True
    assert real_solve["targets"]["hsx"]["skip_reason"] == "production-solve-total-timeout-budget-exhausted"
    assert real_solve["targets"]["geometry4"]["command"][1].endswith(
        "scripts/benchmark_pas_tz_memory_fallback.py"
    )
    assert "--max-residual-norm" in real_solve["targets"]["geometry4"]["command"]
    assert {case["source_type"] for case in payload["plan"]["cases"]} == {
        "synthetic",
        "production_floor_geometry_metadata",
    }


def test_production_floor_preflight_fails_non_pas_metadata(tmp_path: Path) -> None:
    geom4 = _write_input(
        tmp_path,
        geometry_scheme=4,
        collision_operator=0,
        case_name="geometryScheme4_2species_FPCollisions",
    )
    artifact = _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4))
    args = _parse_args(
        [
            "--dry-run",
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            "--artifact-inputs",
            str(artifact),
        ]
    )

    plan = build_plan(args)

    geom4_preflight = plan["production_floor_preflight"]["targets"]["geometry4"]
    assert geom4_preflight["ready"] is False
    assert geom4_preflight["gates"]["pas_collision_operator"]["status"] == "fail"
    assert geom4_preflight["gates"]["checked_artifact_evidence"]["status"] == "pass"
    assert plan["production_floor_preflight"]["targets"]["hsx"]["gates"]["metadata_present"]["status"] == "fail"


def test_production_floor_preflight_requires_residual_runtime_memory_artifact_gates(
    tmp_path: Path,
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    artifact = _write_artifact(
        tmp_path / "artifacts" / "geometry4_high_residual.json",
        target_input=str(geom4),
        residual_norm=2.0e-2,
    )
    args = _parse_args(
        [
            "--dry-run",
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            "--artifact-inputs",
            str(artifact),
        ]
    )

    plan = build_plan(args)

    artifact_record = plan["artifact_probe"]["artifacts"][0]
    assert artifact_record["artifact_gates"]["residual"]["status"] == "fail"
    assert artifact_record["artifact_gates_passed"] is False
    assert artifact_record["ready_evidence"] is False
    geom4_preflight = plan["production_floor_preflight"]["targets"]["geometry4"]
    assert geom4_preflight["ready"] is False
    assert geom4_preflight["gates"]["checked_artifact_evidence"]["status"] == "fail"


def test_artifact_probe_is_json_safe_for_nonfinite_residual(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "artifacts" / "geometry4_nan.json",
        target_input="examples/sfincs_examples/geometryScheme4_2species_PAS_noEr/input.namelist",
        residual_norm=float("nan"),
    )

    probe = build_artifact_probe([artifact])

    row = probe["artifacts"][0]
    assert row["status"] == "ok"
    assert row["ready_evidence"] is False
    assert row["best_residual_norm"] is None
    assert row["guarded_pas_tz_seen"] is True
    json.dumps(probe, allow_nan=False)


def test_opt_in_production_real_solve_probe_runs_ready_targets_with_parent_bound(
    tmp_path: Path, monkeypatch
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    hsx = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="HSX_PASCollisions_DKESTrajectories",
        equilibrium_file="hsx3free.bc",
    )
    geom11 = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="sfincsPaperFigure3_geometryScheme11_PASCollisions",
        equilibrium_file="w7x-sc1.bc",
    )
    artifacts = [
        _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4)),
        _write_artifact(tmp_path / "artifacts" / "hsx.json", target_input=str(hsx)),
        _write_artifact(tmp_path / "artifacts" / "geom11.json", target_input=str(geom11)),
    ]
    args = _parse_args(
        [
            "--run-production-solve-probe",
            "--out",
            str(tmp_path / "probe.json"),
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            str(hsx),
            str(geom11),
            "--artifact-inputs",
            *(str(path) for path in artifacts),
            "--production-solve-timeout-s",
            "30",
        ]
    )
    plan = build_plan(args)
    captured: list[tuple[list[str], float]] = []

    def fake_run(cmd: list[str], *, text: bool, capture_output: bool, timeout: float):
        assert text is True
        assert capture_output is True
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(json.dumps({"summary": {"all_gates_passed": True}}) + "\n")
        captured.append((cmd, timeout))
        return subprocess.CompletedProcess(cmd, 0, stdout="wrote\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rows = run_bounded_real_solve_probe(args, plan["bounded_real_solve_probe"])

    assert len(rows) == 3
    assert {row["target"] for row in rows} == {"geometry4", "hsx", "geometry11"}
    assert all(row["all_gates_passed"] is True for row in rows)
    assert all(timeout <= 600.0 for _, timeout in captured)
    assert all("--timeout-s" in cmd for cmd, _ in captured)
    assert captured[0][0][captured[0][0].index("--maxiter") + 1] == "20"
    assert captured[0][0][captured[0][0].index("--restart") + 1] == "20"


def test_opt_in_production_real_solve_probe_caps_selected_targets_to_total_budget(
    tmp_path: Path,
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    hsx = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="HSX_PASCollisions_DKESTrajectories",
        equilibrium_file="hsx3free.bc",
    )
    artifacts = [
        _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4)),
        _write_artifact(tmp_path / "artifacts" / "hsx.json", target_input=str(hsx)),
    ]
    args = _parse_args(
        [
            "--run-production-solve-probe",
            "--out",
            str(tmp_path / "probe.json"),
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            str(hsx),
            "--artifact-inputs",
            *(str(path) for path in artifacts),
        ]
    )

    plan = build_plan(args)
    real_solve = plan["bounded_real_solve_probe"]

    assert real_solve["parent_wall_timeout_s"] == 510.0
    assert real_solve["total_wall_timeout_budget_s"] == 600.0
    assert real_solve["planned_wall_timeout_s"] == 510.0
    assert real_solve["selected_targets"] == ["geometry4"]
    assert real_solve["targets"]["geometry4"]["will_run"] is True
    assert real_solve["targets"]["hsx"]["will_run"] is False
    assert real_solve["targets"]["hsx"]["skip_reason"] == "production-solve-total-timeout-budget-exhausted"


def test_opt_in_production_real_solve_probe_selects_requested_target_aliases(
    tmp_path: Path,
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    hsx = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="HSX_PASCollisions_DKESTrajectories",
        equilibrium_file="hsx3free.bc",
    )
    geom11 = _write_input(
        tmp_path,
        geometry_scheme=11,
        case_name="sfincsPaperFigure3_geometryScheme11_PASCollisions",
        equilibrium_file="w7x-sc1.bc",
    )
    artifacts = [
        _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4)),
        _write_artifact(tmp_path / "artifacts" / "hsx.json", target_input=str(hsx)),
        _write_artifact(tmp_path / "artifacts" / "geom11.json", target_input=str(geom11)),
    ]

    for requested, selected in (("geometry4", "geometry4"), ("HSX", "hsx"), ("geom11", "geometry11")):
        args = _parse_args(
            [
                "--run-production-solve-probe",
                "--out",
                str(tmp_path / f"{selected}.json"),
                "--systems",
                "diagonal_keep",
                "--metadata-inputs",
                str(geom4),
                str(hsx),
                str(geom11),
                "--artifact-inputs",
                *(str(path) for path in artifacts),
                "--production-solve-targets",
                requested,
            ]
        )

        plan = build_plan(args)
        real_solve = plan["bounded_real_solve_probe"]

        assert real_solve["requested_targets"] == [selected]
        assert real_solve["selected_targets"] == [selected]
        assert real_solve["planned_wall_timeout_s"] <= 600.0
        assert real_solve["targets"][selected]["will_run"] is True
        assert all(
            record["will_run"] is (target == selected)
            for target, record in real_solve["targets"].items()
        )
        assert all(
            record["skip_reason"] in {None, "not-requested"}
            for record in real_solve["targets"].values()
        )


def test_production_real_solve_probe_propagates_default_promotion_gate(
    tmp_path: Path,
) -> None:
    geom4 = _write_input(tmp_path, geometry_scheme=4, case_name="geometryScheme4_2species_PAS_noEr")
    artifact = _write_artifact(tmp_path / "artifacts" / "geometry4.json", target_input=str(geom4))
    args = _parse_args(
        [
            "--run-production-solve-probe",
            "--out",
            str(tmp_path / "probe.json"),
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
            str(geom4),
            "--artifact-inputs",
            str(artifact),
            "--production-solve-targets",
            "geometry4",
            "--production-solve-require-default-promotion-gate",
            "--production-solve-baseline-elapsed-s",
            "50",
            "--production-solve-baseline-rss-mb",
            "4000",
            "--production-solve-min-runtime-speedup",
            "1.10",
            "--production-solve-min-memory-reduction",
            "1.20",
        ]
    )

    plan = build_plan(args)
    real_solve = plan["bounded_real_solve_probe"]
    command = real_solve["targets"]["geometry4"]["command"]

    assert real_solve["gates"]["default_promotion_required"] is True
    assert real_solve["gates"]["baseline_elapsed_s"] == 50.0
    assert real_solve["gates"]["baseline_rss_mb"] == 4000.0
    assert "--require-default-promotion-gate" in command
    assert command[command.index("--baseline-elapsed-s") + 1] == "50.0"
    assert command[command.index("--baseline-rss-mb") + 1] == "4000.0"
    assert command[command.index("--min-runtime-speedup") + 1] == "1.1"
    assert command[command.index("--min-memory-reduction") + 1] == "1.2"


def test_production_real_solve_promotion_gate_requires_baselines(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"

    try:
        main(
            [
                "--dry-run",
                "--out",
                str(out),
                "--production-solve-require-default-promotion-gate",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser failure for missing promotion baselines")


def test_production_real_solve_probe_rejects_unknown_target_without_fallback(
    tmp_path: Path,
) -> None:
    out = tmp_path / "probe.json"

    try:
        main(["--dry-run", "--out", str(out), "--production-solve-targets", "hxs"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser failure for unknown production solve target")


def test_production_real_solve_probe_rejects_long_timeout_without_opt_in(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"

    try:
        main(["--dry-run", "--out", str(out), "--production-solve-timeout-s", "601"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser failure for unbounded production solve timeout")


def test_dry_run_never_launches_opt_in_production_real_solve(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "probe.json"

    def fail_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not launch production real-solve subprocesses")

    monkeypatch.setattr(subprocess, "run", fail_run)

    rc = main(
        [
            "--dry-run",
            "--run-production-solve-probe",
            "--out",
            str(out),
            "--systems",
            "diagonal_keep",
            "--metadata-inputs",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["plan"]["bounded_real_solve_probe"]["run_requested"] is True
    assert all(
        target["will_run"] is False
        for target in payload["plan"]["bounded_real_solve_probe"]["targets"].values()
    )


def test_pas_tz_benchmark_requires_guarded_fallback_provenance() -> None:
    args = type(
        "Args",
        (),
        {
            "timeout_s": 9.0,
            "stall_s": 9.0,
            "max_rss_mb": 0.0,
            "max_residual_norm": 1.0e-3,
            "expected_backend": "auto",
            "allow_solver_churn": False,
            "solve_method": "incremental",
            "require_default_promotion_gate": False,
        },
    )()
    row = {
        "status": "ok",
        "elapsed_s": 1.0,
        "max_rss_mb": 256.0,
        "residual_norm": 1.0e-5,
        "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 1.0}],
        "solver_provenance": {
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
        },
        "metadata": {"accepted_converged": True},
        "messages_tail": ["solve_v3_full_system_linear_gmres: completed without PAS-TZ fallback"],
    }

    gates = pas_tz_result_gates(args, row, "tzfft")

    assert gates["guarded_pas_tz"]["status"] == "fail"
    assert gates["guarded_pas_tz"]["reason"] == "missing-guarded-pas-tz-evidence"

    row["messages_tail"] = [
        "solve_v3_full_system_linear_gmres: PAS-TZ structured fallback guarded out (axis=tzfft); using cheap fallback"
    ]
    gates = pas_tz_result_gates(args, row, "tzfft")

    assert gates["guarded_pas_tz"]["status"] == "pass"
    assert gates["guarded_pas_tz"]["reason"] == "guarded-pas-tz-evidence-recorded"


def test_pas_tz_summary_does_not_promote_when_any_gate_failed() -> None:
    row = {
        "variant": "tzfft",
        "status": "ok",
        "gate": "fail",
        "gate_failures": ["guarded_pas_tz:missing-guarded-pas-tz-evidence"],
        "gates": {
            "default_promotion": {
                "status": "pass",
                "reason": "promotion-win-recorded",
            },
            "guarded_pas_tz": {
                "status": "fail",
                "reason": "missing-guarded-pas-tz-evidence",
            },
        },
    }

    summary = summarize_pas_tz_results([row])

    assert summary["all_gates_passed"] is False
    assert summary["promotion_eligible_variants"] == []
