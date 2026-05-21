from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qi_seed_robustness.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_qi_seed_robustness", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qi_seed = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qi_seed
_SPEC.loader.exec_module(qi_seed)


def _write_qi_input(path: Path, *, equilibrium_name: str = "wout_QI_test.nc") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    equilibrium = path.parent / equilibrium_name
    equilibrium.write_bytes(b"netcdf")
    path.write_text(
        (
            "&general\n"
            "  RHSMode = 1\n"
            "/\n"
            "&geometryParameters\n"
            "  geometryScheme = 5\n"
            "  equilibriumFile = '../../../../equilibria/wout_QI_test.nc'\n"
            "/\n"
            "&physicsParameters\n"
            "  nu_n = 8.330e-3\n"
            "  Er = 0.0\n"
            "/\n"
            "&resolutionParameters\n"
            "  Ntheta = 25\n"
            "  Nzeta = 51\n"
            "  Nx = 8\n"
            "  Nxi = 100\n"
            "/\n"
        ),
        encoding="utf-8",
    )
    return equilibrium


def test_qi_seed_runner_materializes_deterministic_localized_cases(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    equilibrium = _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "0",
                "3",
                "--resolution-scale",
                "0.25",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lane"] == "qi_seed_robustness"
    assert manifest["source_equilibrium"] == str(equilibrium.resolve())
    assert manifest["case_count"] == 2
    cases = {case["seed"]: case for case in manifest["cases"]}
    assert set(cases) == {0, 3}

    first_input = out_root / cases[0]["input"]
    first_text = first_input.read_text(encoding="utf-8")
    assert "equilibriumFile = 'wout_QI_test.nc'" in first_text
    assert (first_input.parent / "wout_QI_test.nc").read_bytes() == b"netcdf"
    assert "Ntheta = 7" in first_text
    assert "Nzeta = 13" in first_text
    assert "Nx = 4" in first_text
    assert "Nxi = 25" in first_text
    assert cases[0]["solve_method"] == "auto"
    assert "--solve-method" not in cases[0]["command"]
    assert cases[0]["perturbations"]["nu_n"] != cases[3]["perturbations"]["nu_n"]
    assert cases[0]["perturbations"]["Er"] != cases[3]["perturbations"]["Er"]

    repeat_root = tmp_path / "repeat"
    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(repeat_root),
                "--seeds",
                "0",
                "3",
                "--resolution-scale",
                "0.25",
                "--clean",
            ]
        )
        == 0
    )
    repeat_manifest = json.loads((repeat_root / "manifest.json").read_text(encoding="utf-8"))
    assert [case["perturbations"] for case in manifest["cases"]] == [
        case["perturbations"] for case in repeat_manifest["cases"]
    ]


def test_qi_seed_runner_records_mocked_execution_results(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        commands.append([str(part) for part in command])
        stdout.write("ok\n")
        stderr.write("")
        assert cwd == qi_seed.REPO_ROOT
        assert timeout == 12.0
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--timeout-s",
                "12",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["passed"] == 1
    assert manifest["execution"]["failed"] == 0
    assert manifest["execution"]["summary"]["attempted"] == 1
    assert manifest["execution"]["summary"]["process_passed"] == 1
    assert manifest["execution"]["summary"]["solver_traces_written"] == 0
    assert manifest["execution"]["gates"]["passed"] is True
    assert len(commands) == 1
    assert commands[0][1:4] == ["-m", "sfincs_jax", "write-output"]
    assert "--solve-method" not in commands[0]
    result = manifest["execution"]["results"][0]
    assert (out_root / result["stdout"]).read_text(encoding="utf-8") == "ok\n"
    assert (out_root / result["stderr"]).read_text(encoding="utf-8") == ""
    assert result["heartbeat"] is None
    assert result["heartbeat_count"] == 0


def test_qi_seed_runner_heartbeat_records_liveness_and_timeout(tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    heartbeat_path = tmp_path / "runner_heartbeat.jsonl"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        returncode, timed_out, heartbeat_count = qi_seed._run_command_with_heartbeat(
            [
                sys.executable,
                "-c",
                "import time; print('start', flush=True); time.sleep(0.05); print('end', flush=True)",
            ],
            cwd=tmp_path,
            stdout=stdout,
            stderr=stderr,
            timeout_s=2.0,
            heartbeat_s=0.01,
            heartbeat_path=heartbeat_path,
        )

    events = [json.loads(line)["event"] for line in heartbeat_path.read_text(encoding="utf-8").splitlines()]
    assert returncode == 0
    assert timed_out is False
    assert heartbeat_count >= 2
    assert events[0] == "started"
    assert "completed" in events
    assert stdout_path.read_text(encoding="utf-8").splitlines() == ["start", "end"]

    timeout_stdout = tmp_path / "timeout_stdout.log"
    timeout_stderr = tmp_path / "timeout_stderr.log"
    timeout_heartbeat = tmp_path / "timeout_heartbeat.jsonl"
    with timeout_stdout.open("w", encoding="utf-8") as stdout, timeout_stderr.open(
        "w", encoding="utf-8"
    ) as stderr:
        returncode, timed_out, heartbeat_count = qi_seed._run_command_with_heartbeat(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            stdout=stdout,
            stderr=stderr,
            timeout_s=0.05,
            heartbeat_s=0.01,
            heartbeat_path=timeout_heartbeat,
        )

    timeout_events = [
        json.loads(line)["event"] for line in timeout_heartbeat.read_text(encoding="utf-8").splitlines()
    ]
    assert returncode == 124
    assert timed_out is True
    assert heartbeat_count >= 3
    assert "timeout" in timeout_events
    assert timeout_events[-1] == "terminated"
    assert "timed out" in timeout_stderr.read_text(encoding="utf-8")


def test_qi_seed_runner_infers_matrix_sizes_from_timeout_progress() -> None:
    events = [
        "The matrix is 81377 x 81377 elements.",
        "solve_v3_full_system_linear_gmres: active-DOF mode enabled (size=81377/139502)",
        "QI seed execution timed out after 420.000 s.",
    ]

    assert qi_seed._infer_sizes_from_progress_events(events) == (81377, 139502)

    output_refusal = [
        "sfincs_jax write-output failed: Refusing to write nonconverged RHSMode=1 diagnostics "
        "for a production-sized solve: active_size=81377 residual_norm=2.833435e-05 "
        "target=3.021487e-11 solve_method=xblock_sparse_pc_gmres."
    ]
    assert qi_seed._infer_sizes_from_progress_events(output_refusal) == (81377, None)


def test_qi_seed_runner_infers_latest_matvec_progress() -> None:
    events = [
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=875 elapsed_s=403.885",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=900 elapsed_s=412.719",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete method=gmres "
        "elapsed_s=417.536 iters=48 matvecs=925 residual=4.0e-13 target=3.0e-13",
    ]

    assert qi_seed._infer_last_matvec_progress(events) == (925, 417.536)


def test_qi_seed_runner_infers_side_probe_and_residual_progress() -> None:
    events = [
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI residual-deflated "
        "preconditioner accepted residual 3.021487e-05 -> 2.814560e-05 "
        "(rank=16 seed_solver=cycle_minres cycles=8 use_in_krylov=0 ratio=9.315148e-01)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI device "
        "preconditioner multilevel residual equation "
        "(levels=3 stage_rank=16 order=coarse_to_fine include_global=1)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI device "
        "preconditioner accepted residual 2.814560e-05 -> 2.533104e-05 "
        "(rank=12 use_in_krylov=1 operator_krylov=1 coarse_reuse=1 ratio=9.000000e-01)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe method_rescue "
        "side=left->left method=gmres->lgmres iters=20 matvecs=23 residual=4.565805e-06 "
        "ratio=1.511112e+07 seed_used=1",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete method=lgmres "
        "elapsed_s=247.000 iters=48 matvecs=2978 residual=4.122852e-14 "
        "target=3.021487e-13 ksp_residual=4.122852e-14",
    ]

    side_probe = qi_seed._infer_side_probe_progress(events)
    assert side_probe["precondition_side"] == "left"
    assert side_probe["xblock_side_probe_initial_method"] == "gmres"
    assert side_probe["xblock_side_probe_selected_method"] == "lgmres"
    assert side_probe["xblock_side_probe_lgmres_rescue"] is True
    assert side_probe["xblock_side_probe_iterations"] == 20
    assert side_probe["xblock_side_probe_matvecs"] == 23
    assert side_probe["xblock_side_probe_residual_norm"] == 4.565805e-06
    assert side_probe["xblock_side_probe_residual_ratio"] == 1.511112e7
    assert qi_seed._infer_lgmres_rescue_status(events, side_probe) == "used"

    qi_deflated = qi_seed._infer_qi_deflated_progress(events)
    assert qi_deflated["xblock_qi_deflated_preconditioner_used"] is True
    assert qi_deflated["xblock_qi_deflated_preconditioner_rank"] == 16
    assert qi_deflated["xblock_qi_deflated_preconditioner_seed_solver"] == "cycle_minres"
    assert qi_deflated["xblock_qi_deflated_preconditioner_cycles"] == 8
    assert qi_deflated["xblock_qi_deflated_preconditioner_use_in_krylov"] is False
    assert qi_deflated["xblock_qi_deflated_preconditioner_residual_before"] == 3.021487e-05
    assert qi_deflated["xblock_qi_deflated_preconditioner_residual_after"] == 2.814560e-05
    assert qi_deflated["xblock_qi_deflated_preconditioner_improvement_ratio"] == 9.315148e-01
    qi_device = qi_seed._infer_qi_device_progress(events)
    assert qi_device["xblock_qi_device_preconditioner_used"] is True
    assert qi_device["xblock_qi_device_preconditioner_rank"] == 12
    assert qi_device["xblock_qi_device_preconditioner_use_in_krylov"] is True
    assert qi_device["xblock_qi_device_preconditioner_operator_krylov_enrichment"] is True
    assert qi_device["xblock_qi_device_preconditioner_coarse_reuse"] is True
    assert qi_device["xblock_qi_device_preconditioner_multilevel_residual_equation"] is True
    assert qi_device["xblock_qi_device_preconditioner_multilevel_residual_equation_stage_rank"] == 16
    assert qi_device["xblock_qi_device_preconditioner_multilevel_residual_equation_order"] == "coarse_to_fine"
    assert qi_device["xblock_qi_device_preconditioner_multilevel_residual_equation_include_global"] is True
    assert qi_device["xblock_qi_device_preconditioner_residual_before"] == 2.814560e-05
    assert qi_device["xblock_qi_device_preconditioner_residual_after"] == 2.533104e-05
    assert qi_device["xblock_qi_device_preconditioner_improvement_ratio"] == 9.0e-01

    residual = qi_seed._infer_last_residual_progress(events)
    assert residual is not None
    assert residual["event"] == events[-1]
    assert residual["residual_norm"] == 4.122852e-14
    assert residual["residual_target"] == 3.021487e-13
    assert 0.0 < residual["residual_ratio"] < 1.0


def test_qi_seed_runner_records_timeout_attempt_from_synthetic_tails(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&resolutionParameters\n  Ntheta = 15\n/\n", encoding="utf-8")
    progress_events = [
        "solve_v3_full_system_linear_gmres: active-DOF mode enabled (size=81377/139502)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres LGMRES rescue disabled by explicit gmres method",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI residual-deflated "
        "preconditioner accepted residual 3.021487e-05 -> 2.814560e-05 "
        "(rank=16 seed_solver=cycle_minres cycles=8 use_in_krylov=0 ratio=9.315148e-01)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres QI device "
        "preconditioner accepted residual 2.814560e-05 -> 2.533104e-05 "
        "(rank=12 use_in_krylov=1 ratio=9.000000e-01)",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe switch "
        "side=left->right method=gmres->gmres iters=20 matvecs=23 residual=4.565805e-06 "
        "ratio=1.511112e+07 seed_used=1 preserved_physical_seed=1",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
        "probe-coarse improved seed residual 4.565805e-06 -> 2.830374e-06 (steps=1 directions=40)",
        "QI seed execution timed out after 420.000 s.",
    ]
    stdout_tail = [
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=875 elapsed_s=403.885",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=900 elapsed_s=412.719",
    ]
    manifest = {
        "source_input": str(input_path),
        "resolution_scale": 0.6,
        "case_count": 1,
        "solve_method": "xblock_sparse_pc_gmres",
        "nu_jitter": 0.05,
        "er_jitter": 0.02,
        "cases": [
            {
                "case": "qi_seed_0003",
                "resolution": {"NTHETA": 15, "NZETA": 31, "NX": 5, "NXI": 60},
            }
        ],
        "execution": {
            "summary": {"attempted": 1, "timed_out": 1},
            "gates": {"passed": False},
            "timeout_s": 420.0,
            "heartbeat_s": 15.0,
            "fail_fast": False,
            "results": [
                {
                    "case": "qi_seed_0003",
                    "seed": 3,
                    "returncode": 124,
                    "timed_out": True,
                    "output_exists": False,
                    "solver_trace_exists": False,
                    "solver_trace_summary": None,
                    "elapsed_s": 420.1,
                    "progress_events": progress_events,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": ["QI seed execution timed out after 420.000 s."],
                    "heartbeat": "qi_seed_0003/runner_heartbeat.jsonl",
                    "heartbeat_count": 31,
                }
            ],
        },
    }

    seed = qi_seed._compact_execution_artifact(manifest)["seeds"][0]
    assert seed["active_size"] == 81377
    assert seed["total_size"] == 139502
    assert seed["precondition_side"] == "right"
    assert seed["xblock_side_probe_used"] is True
    assert seed["xblock_side_probe_decision"] == "switch"
    assert seed["xblock_side_probe_switched"] is True
    assert seed["xblock_side_probe_selected_method"] == "gmres"
    assert seed["xblock_side_probe_lgmres_rescue"] is False
    assert seed["xblock_lgmres_rescue_status"] == "disabled"
    assert seed["xblock_side_probe_residual_norm"] == 4.565805e-06
    assert seed["xblock_side_probe_residual_ratio"] == 1.511112e7
    assert seed["xblock_qi_deflated_preconditioner_used"] is True
    assert seed["xblock_qi_deflated_preconditioner_reason"] == "residual_reduced"
    assert seed["xblock_qi_deflated_preconditioner_residual_before"] == 3.021487e-05
    assert seed["xblock_qi_deflated_preconditioner_residual_after"] == 2.814560e-05
    assert seed["xblock_qi_deflated_preconditioner_improvement_ratio"] == 9.315148e-01
    assert seed["xblock_qi_deflated_preconditioner_seed_solver"] == "cycle_minres"
    assert seed["xblock_qi_deflated_preconditioner_cycles"] == 8
    assert seed["xblock_qi_deflated_preconditioner_use_in_krylov"] is False
    assert seed["xblock_qi_device_preconditioner_used"] is True
    assert seed["xblock_qi_device_preconditioner_rank"] == 12
    assert seed["xblock_qi_device_preconditioner_residual_before"] == 2.814560e-05
    assert seed["xblock_qi_device_preconditioner_residual_after"] == 2.533104e-05
    assert seed["xblock_qi_device_preconditioner_improvement_ratio"] == 9.0e-01
    assert seed["xblock_qi_device_preconditioner_use_in_krylov"] is True
    assert seed["last_progress_residual_event"] == progress_events[5]
    assert seed["last_progress_residual_before"] == 4.565805e-06
    assert seed["last_progress_residual_norm"] == 2.830374e-06
    assert seed["last_matvecs"] == 900
    assert seed["last_matvec_elapsed_s"] == 412.719


def test_qi_seed_runner_preserves_forced_lgmres_visibility() -> None:
    events = [
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres LGMRES rescue forced by env",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe method_rescue "
        "side=left->left method=gmres->lgmres iters=20 matvecs=23 residual=4.565805e-06 "
        "ratio=1.511112e+07 seed_used=1",
    ]
    side_probe = qi_seed._infer_side_probe_progress(events)

    assert qi_seed._infer_lgmres_rescue_status(events, side_probe) == "forced"


def test_qi_seed_runner_records_solver_trace_summary(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "cpu",
                    "converged": False,
                    "elapsed_s": 1.25,
                    "residual_norm": 2.0e-6,
                    "residual_target": 1.0e-11,
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "dense",
                    "metadata": {
                        "solver_metadata": {
                            "acceptance_criterion": "not_converged",
                            "accepted_converged": False,
                            "iterations": 12,
                            "solver_kind": "dense",
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_coarse_operator_shape": [4, 4],
                            "xblock_device_host_fallback_used": True,
                            "xblock_device_host_fallback_reason": "large-qi-full-fp-3d",
                            "xblock_device_host_fallback_requested_method": "gmres_jax",
                            "xblock_device_host_fallback_effective_krylov_env_value": "auto",
                            "xblock_device_host_fallback_non_autodiff": True,
                            "xblock_qi_two_level_preconditioner_built": True,
                            "xblock_qi_two_level_preconditioner_smoothed_load_basis": True,
                            "xblock_qi_two_level_preconditioner_improvement_ratio": 0.99,
                            "xblock_qi_two_level_preconditioner_smoothed_load_metadata": {"rank": 4},
                            "xblock_moment_schur_built": True,
                            "xblock_moment_schur_used": False,
                            "xblock_moment_schur_reason": "probe_not_reduced",
                            "xblock_moment_schur_probe_improvement_ratio": 1.2,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    result = manifest["execution"]["results"][0]
    assert result["output_exists"] is True
    assert result["solver_trace_exists"] is True
    summary = result["solver_trace_summary"]
    assert summary["solve_method"] == "dense"
    assert summary["converged"] is False
    assert summary["accepted_converged"] is False
    assert summary["residual_norm"] == 2.0e-6
    assert summary["residual_target"] == 1.0e-11
    assert summary["residual_ratio"] == 2.0e5
    assert summary["iterations"] == 12
    assert summary["xblock_device_host_fallback_used"] is True
    assert summary["xblock_device_host_fallback_reason"] == "large-qi-full-fp-3d"
    assert summary["xblock_device_host_fallback_requested_method"] == "gmres_jax"
    assert summary["xblock_device_host_fallback_effective_krylov_env_value"] == "auto"
    assert summary["xblock_device_host_fallback_non_autodiff"] is True
    assert summary["xblock_qi_device_preconditioner_used"] is True
    assert summary["xblock_qi_device_preconditioner_use_in_krylov"] is True
    assert summary["xblock_qi_device_preconditioner_coarse_operator_shape"] == [4, 4]
    assert summary["xblock_qi_two_level_preconditioner_built"] is True
    assert summary["xblock_qi_two_level_preconditioner_smoothed_load_basis"] is True
    assert summary["xblock_qi_two_level_preconditioner_improvement_ratio"] == 0.99
    assert summary["xblock_qi_two_level_preconditioner_smoothed_load_metadata"] == {"rank": 4}
    assert summary["xblock_moment_schur_built"] is True
    assert summary["xblock_moment_schur_used"] is False
    assert summary["xblock_moment_schur_reason"] == "probe_not_reduced"
    assert summary["xblock_moment_schur_probe_improvement_ratio"] == 1.2
    assert manifest["execution"]["summary"]["max_residual_ratio"] == 2.0e5
    assert manifest["execution"]["summary"]["converged"] == 0
    assert manifest["execution"]["summary"]["solve_methods"] == ["dense"]
    assert manifest["execution"]["gates"]["passed"] is True


def test_qi_seed_runner_enforces_optional_trace_gates(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "cpu",
                    "converged": False,
                    "elapsed_s": 1.25,
                    "residual_norm": 2.0e-6,
                    "residual_target": 1.0e-11,
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "dense",
                    "metadata": {"solver_metadata": {"accepted_converged": False}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--max-residual-ratio",
                "1",
                "--require-converged",
                "--require-accepted-converged",
                "--clean",
            ]
        )
        == 1
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    gates = manifest["execution"]["gates"]
    assert gates["passed"] is False
    reasons = {failure["reason"] for failure in gates["failures"]}
    assert reasons == {"residual_ratio_exceeded", "not_converged", "not_accepted_converged"}


def test_qi_seed_runner_passes_optional_trace_gates(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "cpu",
                    "converged": True,
                    "elapsed_s": 0.75,
                    "residual_norm": 4.0e-12,
                    "residual_target": 1.0e-11,
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "dense",
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": True,
                            "iterations": 1,
                            "solver_kind": "dense",
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--max-residual-ratio",
                "1",
                "--require-converged",
                "--require-accepted-converged",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["gates"]["passed"] is True
    assert manifest["execution"]["summary"]["max_residual_ratio"] == 0.4
    assert manifest["execution"]["summary"]["converged"] == 1
    assert manifest["execution"]["summary"]["accepted_converged"] == 1


def test_qi_seed_runner_keeps_explicit_diagnostic_solve_method(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--solve-method",
                "dense",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    assert case["solve_method"] == "dense"
    assert case["command"][-2:] == ["--solve-method", "dense"]


def test_qi_seed_runner_operator_krylov_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "operator-krylov-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "operator-krylov-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert env["SFINCS_JAX_GMRES_PRECONDITION_SIDE"] == "right"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH"] == "64"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK"] == "64"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER"]
        == "matrix_free_block_minres_hybrid"
    )

    case = manifest["cases"][0]
    assert case["probe_preset"] == "operator-krylov-device-qi"
    assert case["env"] == env
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["requested_solve_method"] == "auto"
    assert case["solve_method"] == "xblock_sparse_pc_gmres"
    assert case["command"][0] == "env"
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH=64" in case[
        "command"
    ]
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE=1" in case[
        "command"
    ]
    assert case["command"][-2:] == ["--solve-method", "xblock_sparse_pc_gmres"]


def test_qi_seed_runner_current_constraint_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "current-constraint-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "current-constraint-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE"]
        == "1"
    )
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["cases"][0]["probe_preset"] == "current-constraint-device-qi"


def test_qi_seed_runner_residual_snapshot_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "residual-snapshot-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "residual-snapshot-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"]
        == "1"
    )
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_MAX_RANK"] == "48"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT"] == "1"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_residual_snapshot_equation_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "residual-snapshot-equation-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "residual-snapshot-equation-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION"]
        == "1"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "48"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT"] == "1"
    assert manifest["cases"][0]["probe_preset"] == "residual-snapshot-equation-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_assembled_reuse_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "assembled-reuse-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "assembled-reuse-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS"] == "4096"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR"] == "1"
    assert manifest["cases"][0]["probe_preset"] == "assembled-reuse-device-qi"
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR=1" in manifest["cases"][0]["command"]


def test_qi_seed_runner_composite_closure_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "composite-closure-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "composite-closure-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_OPERATOR_IMAGES"]
        == "1"
    )
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK"] == "256"
    assert manifest["cases"][0]["probe_preset"] == "composite-closure-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_global_moment_closure_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "global-moment-closure-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "global-moment-closure-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION"]
        == "1"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER"]
        == "galerkin"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_CURRENT"
        ]
        == "1"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE"]
        == "2"
    )
    assert manifest["cases"][0]["probe_preset"] == "global-moment-closure-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_residual_galerkin_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "residual-galerkin-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "residual-galerkin-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK"]
        == "8"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_SOLVER"]
        == "action_lstsq"
    )
    assert manifest["cases"][0]["probe_preset"] == "residual-galerkin-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_phase_space_coarse_reuse_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "phase-space-coarse-reuse-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "phase-space-coarse-reuse-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION"]
        == "1"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "32"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_BOUNDARY"]
        == "0.35"
    )
    assert manifest["cases"][0]["probe_preset"] == "phase-space-coarse-reuse-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_residual_bounce_region_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "residual-bounce-region-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "residual-bounce-region-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE"]
        == "1"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_MAX_RANK"
        ]
        == "48"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_REGION_BANDS"
        ]
        == "bounce,trapped,passing"
    )
    assert manifest["cases"][0]["probe_preset"] == "residual-bounce-region-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_block_schur_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "block-schur-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "block-schur-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"] == "1"
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"]
        == "1"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK"]
        == "64"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER"]
        == "galerkin"
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION=1"
        in manifest["cases"][0]["command"]
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER=galerkin"
        in manifest["cases"][0]["command"]
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_adaptive_residual_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "adaptive-residual-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "adaptive-residual-device-qi"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER"] == (
        "adaptive_residual_equation"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING"]
        == "block_hierarchy"
    )
    assert (
        env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"]
        == "1"
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER=adaptive_residual_equation"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_adjoint_krylov_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "adjoint-krylov-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "adjoint-krylov-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH"] == "2"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE"] == "autodiff"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["cases"][0]["probe_preset"] == "adjoint-krylov-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_augmented_krylov_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "augmented-krylov-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "augmented-krylov-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE"] == "cycle"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K"] == "0"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV_MODE"] == "combined"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["cases"][0]["probe_preset"] == "augmented-krylov-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV=1"
        in manifest["cases"][0]["command"]
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV_MODE=combined"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_recycled_augmented_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "recycled-augmented-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "recycled-augmented-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE"] == "cycle"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K"] == "32"
    assert env["SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER"] == "960"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV_MODE"] == "combined"
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["cases"][0]["probe_preset"] == "recycled-augmented-device-qi"
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K=32" in manifest["cases"][0]["command"]
    assert "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER=960" in manifest["cases"][0]["command"]


def test_qi_seed_runner_coarse_residual_device_qi_probe_records_env(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--resolution-scale",
                "0.60",
                "--probe-preset",
                "coarse-residual-device-qi",
                "--clean",
            ]
        )
        == 0
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["probe_env"]
    assert manifest["probe_preset"] == "coarse-residual-device-qi"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert env["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE"] == "1"
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK"
        ]
        == "16"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER"
        ]
        == "coarse_to_fine"
    )
    assert (
        env[
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert manifest["solve_method"] == "xblock_sparse_pc_gmres"
    assert manifest["cases"][0]["probe_preset"] == "coarse-residual-device-qi"
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION=1"
        in manifest["cases"][0]["command"]
    )


def test_qi_seed_runner_records_timeout_without_crashing(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--timeout-s",
                "0.01",
                "--clean",
            ]
        )
        == 1
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    result = manifest["execution"]["results"][0]
    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert result["solver_trace_exists"] is False
    assert result["solver_trace_summary"] is None
    assert "timed out" in (out_root / result["stderr"]).read_text(encoding="utf-8")


def test_qi_seed_runner_keeps_compact_failure_progress(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        stdout.write("setup line\n")
        stdout.write("solve_v3_full_system_linear_gmres: active matrix size=39314 (total=70202)\n")
        stdout.write("solve_v3_full_system_linear_gmres: strong preconditioner fallback kind=xblock_tz_lmax\n")
        stdout.write("sparse_lsmr complete elapsed_s=125.0 iters=1000 residual=5.0e-06 target=2.5e-11\n")
        stdout.write(
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "post-minres improved residual 5.4e-06 -> 5.3e-06\n"
        )
        stdout.write(
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "post-coarse improved residual 5.3e-06 -> 4.0e-06\n"
        )
        stderr.write("Refusing to write nonconverged RHSMode=1 diagnostics\n")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "5",
                "--execute",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 1
    )

    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    result = manifest["execution"]["results"][0]
    assert result["progress_events"] == [
        "solve_v3_full_system_linear_gmres: active matrix size=39314 (total=70202)",
        "solve_v3_full_system_linear_gmres: strong preconditioner fallback kind=xblock_tz_lmax",
        "sparse_lsmr complete elapsed_s=125.0 iters=1000 residual=5.0e-06 target=2.5e-11",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
        "post-minres improved residual 5.4e-06 -> 5.3e-06",
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
        "post-coarse improved residual 5.3e-06 -> 4.0e-06",
        "Refusing to write nonconverged RHSMode=1 diagnostics",
    ]
    assert result["stderr_tail"] == ["Refusing to write nonconverged RHSMode=1 diagnostics"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["progress_events"] == result["progress_events"]
    assert seed["stderr_tail"] == result["stderr_tail"]
    assert seed["run_outcome"] == "process_failed"
    assert seed["failed_before_summary_json"] is True
    assert "failed_before_solver_trace_summary" in seed["evidence_tags"]
    assert summary["evidence_classification"]["has_failed_before_summary_json"] is True


def test_extract_progress_events_preserves_qi_setup_when_device_cycles_are_long(tmp_path: Path) -> None:
    stdout_path = tmp_path / "sfincs_jax.stdout.log"
    stdout_path.write_text(
        "\n".join(
            [
                "solve_v3_full_system_linear_gmres: active matrix size=81377 (total=139502)",
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner operator-Krylov coarse enrichment (depth=64 max_rank=128)",
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner accepted residual 3.0e-05 -> 2.8e-05 "
                "(rank=13 operator_krylov=1 coarse_reuse=1)",
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI augmented Krylov enabled rank=13 mode=combined",
                *[
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"device-cycle cycle={cycle} iterations={40 * cycle} residual={1.0 / cycle:.6e} "
                    "target=3.0e-13"
                    for cycle in range(1, 30)
                ],
                "sfincs_jax write-output failed: Refusing to write nonconverged RHSMode=1 diagnostics",
            ]
        ),
        encoding="utf-8",
    )

    events = qi_seed._extract_progress_events(stdout_path, max_events=8)

    assert any("operator-Krylov coarse enrichment" in event for event in events)
    assert any("coarse_reuse=1" in event for event in events)
    assert any("QI augmented Krylov enabled" in event for event in events)
    assert any("device-cycle cycle=29" in event for event in events)
    assert any("Refusing to write nonconverged" in event for event in events)


def test_qi_seed_runner_classifies_installed_krylov_and_coarse_reuse_from_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": True,
                    "residual_norm": 1.0e-12,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": True,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                            "xblock_qi_device_preconditioner_coarse_operator_shape": [8, 8],
                            "xblock_qi_device_preconditioner_operator_on_basis_shape": [8, 8],
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "operator-krylov-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_installed_krylov_coarse_reuse"
    assert seed["promotion_eligible"] is True
    assert seed["requested_qi_device_installed_krylov"] is True
    assert seed["requested_qi_device_operator_krylov"] is True
    assert seed["observed_qi_device_installed_krylov"] is True
    assert seed["observed_qi_device_operator_krylov"] is True
    assert seed["observed_qi_device_coarse_reuse"] is True
    assert "observed_installed_krylov" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == ["device_qi_installed_krylov_coarse_reuse"]
    assert summary["public_cli_default_path"] is False


def test_qi_seed_runner_keeps_block_schur_probe_fail_closed_until_converged(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": False,
                    "residual_norm": 2.0e-5,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": False,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                                "xblock_qi_device_preconditioner_coarse_reuse": True,
                                "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                                "xblock_qi_device_preconditioner_block_schur_residual_equation": True,
                                "xblock_qi_device_preconditioner_metadata": {
                                    "block_schur_residual_equation_enabled": True,
                                    "block_schur_residual_equation_candidate_count": 12,
                                    "block_schur_residual_equation_rank": 8,
                                    "block_schur_residual_equation_group_count": 4,
                                },
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "block-schur-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_block_schur_residual_coarse_reuse"
    assert seed["run_outcome"] == "not_converged"
    assert seed["promotion_eligible"] is False
    assert seed["requested_qi_device_block_schur_residual"] is True
    assert seed["observed_qi_device_block_schur_residual"] is True
    assert seed["xblock_qi_device_preconditioner_block_schur_residual_equation_candidate_count"] == 12
    assert seed["xblock_qi_device_preconditioner_block_schur_residual_equation_rank"] == 8
    assert "not_converged" in seed["evidence_tags"]
    assert "not_accepted_converged" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == [
        "device_qi_block_schur_residual_coarse_reuse"
    ]
    assert summary["evidence_classification"]["outcomes"] == ["not_converged"]
    assert summary["evidence_classification"]["has_observed_block_schur_residual"] is True
    assert summary["evidence_classification"]["promotion_eligible_seed_count"] == 0


def test_qi_seed_runner_keeps_global_moment_probe_fail_closed_until_converged(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": False,
                    "residual_norm": 2.0e-5,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": False,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                            "xblock_qi_device_preconditioner_global_moment_residual_equation": True,
                            "xblock_qi_device_preconditioner_metadata": {
                                "global_moment_residual_equation_enabled": True,
                                "global_moment_residual_equation_candidate_count": 9,
                                "global_moment_residual_equation_rank": 5,
                                "global_moment_residual_equation_solver": "galerkin",
                                "global_moment_residual_equation_condition_estimate": 12.0,
                            },
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "global-moment-closure-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_global_moment_residual_equation_coarse_reuse"
    assert seed["run_outcome"] == "not_converged"
    assert seed["promotion_eligible"] is False
    assert seed["requested_qi_device_global_moment_residual_equation"] is True
    assert seed["observed_qi_device_global_moment_residual_equation"] is True
    assert seed["xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count"] == 9
    assert seed["xblock_qi_device_preconditioner_global_moment_residual_equation_rank"] == 5
    assert seed["xblock_qi_device_preconditioner_global_moment_residual_equation_solver"] == "galerkin"
    assert "observed_global_moment_residual_equation" in seed["evidence_tags"]
    assert "not_converged" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == [
        "device_qi_global_moment_residual_equation_coarse_reuse"
    ]
    assert summary["evidence_classification"]["has_observed_global_moment_residual_equation"] is True
    assert summary["evidence_classification"]["promotion_eligible_seed_count"] == 0


def test_qi_seed_runner_keeps_residual_galerkin_probe_fail_closed_until_converged(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": False,
                    "residual_norm": 2.0e-5,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": False,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                            "xblock_qi_device_preconditioner_residual_galerkin_equation": True,
                            "xblock_qi_device_preconditioner_metadata": {
                                "residual_galerkin_equation_enabled": True,
                                "residual_galerkin_equation_candidate_count": 12,
                                "residual_galerkin_equation_rank": 6,
                                "residual_galerkin_equation_stage_count": 2,
                                "residual_galerkin_equation_solver": "action_lstsq",
                                "residual_galerkin_equation_condition_estimate": 3.0,
                            },
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "residual-galerkin-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_residual_galerkin_equation_coarse_reuse"
    assert seed["run_outcome"] == "not_converged"
    assert seed["promotion_eligible"] is False
    assert seed["requested_qi_device_residual_galerkin_equation"] is True
    assert seed["observed_qi_device_residual_galerkin_equation"] is True
    assert seed["xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count"] == 12
    assert seed["xblock_qi_device_preconditioner_residual_galerkin_equation_rank"] == 6
    assert seed["xblock_qi_device_preconditioner_residual_galerkin_equation_stage_count"] == 2
    assert "observed_residual_galerkin_equation" in seed["evidence_tags"]
    assert "not_converged" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == [
        "device_qi_residual_galerkin_equation_coarse_reuse"
    ]
    assert summary["evidence_classification"]["has_observed_residual_galerkin_equation"] is True
    assert summary["evidence_classification"]["promotion_eligible_seed_count"] == 0


def test_qi_seed_runner_keeps_phase_space_probe_fail_closed_until_converged(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": False,
                    "residual_norm": 2.0e-5,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": False,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                            "xblock_qi_device_preconditioner_phase_space_residual_equation": True,
                            "xblock_qi_device_preconditioner_metadata": {
                                "phase_space_residual_equation_enabled": True,
                                "phase_space_residual_equation_candidate_count": 11,
                                "phase_space_residual_equation_rank": 7,
                                "phase_space_residual_equation_stage_count": 1,
                                "phase_space_residual_equation_solver": "action_lstsq",
                                "phase_space_residual_equation_condition_estimate": 4.0,
                            },
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "phase-space-coarse-reuse-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_phase_space_residual_equation_coarse_reuse"
    assert seed["run_outcome"] == "not_converged"
    assert seed["promotion_eligible"] is False
    assert seed["requested_qi_device_phase_space_residual_equation"] is True
    assert seed["observed_qi_device_phase_space_residual_equation"] is True
    assert seed["xblock_qi_device_preconditioner_phase_space_residual_equation_candidate_count"] == 11
    assert seed["xblock_qi_device_preconditioner_phase_space_residual_equation_rank"] == 7
    assert seed["xblock_qi_device_preconditioner_phase_space_residual_equation_stage_count"] == 1
    assert "observed_phase_space_residual_equation" in seed["evidence_tags"]
    assert "not_converged" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == [
        "device_qi_phase_space_residual_equation_coarse_reuse"
    ]
    assert summary["evidence_classification"]["has_observed_phase_space_residual_equation"] is True
    assert summary["evidence_classification"]["promotion_eligible_seed_count"] == 0


def test_qi_seed_runner_keeps_residual_bounce_region_probe_fail_closed_until_converged(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        trace_path = Path(command[command.index("--solver-trace") + 1])
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_bytes(b"h5")
        trace_path.write_text(
            json.dumps(
                {
                    "backend": "gpu",
                    "converged": False,
                    "residual_norm": 2.0e-5,
                    "residual_target": 1.0e-11,
                    "metadata": {
                        "solver_metadata": {
                            "accepted_converged": False,
                            "xblock_qi_device_preconditioner_used": True,
                            "xblock_qi_device_preconditioner_use_in_krylov": True,
                            "xblock_qi_device_preconditioner_coarse_reuse": True,
                            "xblock_qi_device_preconditioner_operator_krylov_enrichment": True,
                            "xblock_qi_device_preconditioner_residual_region_bounce_coarse": True,
                            "xblock_qi_device_preconditioner_metadata": {
                                "residual_region_bounce_coarse_enabled": True,
                                "residual_region_bounce_coarse_candidate_count": 13,
                                "residual_region_bounce_coarse_rank": 6,
                                "residual_region_bounce_coarse_stage_count": 2,
                                "residual_region_bounce_coarse_solver": "action_lstsq",
                                "residual_region_bounce_coarse_condition_estimate": 5.0,
                                "residual_region_bounce_coarse_bounce_boundary": 0.35,
                                "residual_region_bounce_coarse_region_bands": [
                                    "bounce",
                                    "trapped",
                                    "passing",
                                ],
                            },
                        }
                    },
                    "selected_path": "rhsmode1_solution",
                    "solve_method": "xblock_sparse_pc_gmres",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qi_seed.subprocess, "run", fake_run)

    assert (
        qi_seed.main(
            [
                "--input",
                str(input_path),
                "--out-root",
                str(out_root),
                "--seeds",
                "3",
                "--execute",
                "--probe-preset",
                "residual-bounce-region-device-qi",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = summary["seeds"][0]
    assert seed["evidence_class"] == "device_qi_residual_bounce_region_coarse_reuse"
    assert seed["run_outcome"] == "not_converged"
    assert seed["promotion_eligible"] is False
    assert seed["requested_qi_device_residual_bounce_region_coarse"] is True
    assert seed["observed_qi_device_residual_bounce_region_coarse"] is True
    assert seed["xblock_qi_device_preconditioner_residual_region_bounce_coarse_candidate_count"] == 13
    assert seed["xblock_qi_device_preconditioner_residual_region_bounce_coarse_rank"] == 6
    assert seed["xblock_qi_device_preconditioner_residual_region_bounce_coarse_stage_count"] == 2
    assert seed["xblock_qi_device_preconditioner_residual_region_bounce_coarse_solver"] == "action_lstsq"
    assert "observed_residual_bounce_region_coarse" in seed["evidence_tags"]
    assert "not_converged" in seed["evidence_tags"]
    assert summary["evidence_classification"]["classes"] == [
        "device_qi_residual_bounce_region_coarse_reuse"
    ]
    assert summary["evidence_classification"]["has_observed_residual_bounce_region_coarse"] is True
    assert summary["evidence_classification"]["promotion_eligible_seed_count"] == 0


def test_qi_seed_progress_parser_preserves_residual_bounce_region_bands() -> None:
    progress = qi_seed._infer_qi_device_progress(
        [
            (
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner residual-region/bounce coarse "
                "(max_rank=48 rank=7 candidates=12 stage_count=1 solver=action_lstsq "
                "condition_estimate=4.0 residual_before=3.0e-5 residual_after=2.8e-5 "
                "boundary=3.500e-01 include_global=1 radial=1 species=1 "
                "bands=bounce,trapped,passing)"
            ),
            (
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner accepted residual 3.0e-05 -> 2.8e-05 "
                "(rank=13 cycles=1 ratio=9.333333e-01 operator_krylov=1 coarse_reuse=1)"
            ),
        ]
    )

    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse"] is True
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank"] == 48
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_rank"] == 7
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_candidate_count"] == 12
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_stage_count"] == 1
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_solver"] == "action_lstsq"
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_condition_estimate"] == 4.0
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_before"] == 3.0e-5
    assert progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_after"] == 2.8e-5
    assert (
        progress["xblock_qi_device_preconditioner_residual_region_bounce_coarse_region_bands"]
        == "bounce,trapped,passing"
    )


def test_evidence_classification_does_not_treat_false_residual_bounce_as_observed(
    tmp_path: Path,
) -> None:
    artifact = {
        "schema_version": 2,
        "artifact_kind": "qi_seed_execution_summary",
        "seeds": [
            {
                "evidence_class": "requested_residual_bounce_region_coarse_device_qi",
                "requested_qi_device_residual_bounce_region_coarse": True,
                "observed_qi_device_residual_bounce_region_coarse": False,
            }
        ],
    }

    classification = qi_seed._artifact_evidence_classification(tmp_path / "artifact.json", artifact)

    assert classification["classes"] == ["requested_residual_bounce_region_coarse_device_qi"]
    assert classification["has_observed_residual_bounce_region_coarse"] is False


def test_evidence_manifest_does_not_promote_failed_larger_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    passing_path = tmp_path / "passing.json"
    failed_path = tmp_path / "failed.json"
    passing_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_kind": "qi_seed_execution_summary",
                "lane": "qi_seed_robustness",
                "case_count": 1,
                "public_cli_default_path": True,
                "resolution": {"NTHETA": 9, "NZETA": 19, "NX": 4, "NXI": 35},
                "total_size_estimate": 23942,
                "execution_summary": {
                    "backends": ["cpu"],
                    "max_residual_ratio": 1.0e-6,
                    "process_failed": 0,
                    "timed_out": 0,
                },
                "gates": {"passed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failed_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_kind": "qi_seed_execution_summary",
                "lane": "qi_seed_robustness",
                "case_count": 1,
                "public_cli_default_path": True,
                "resolution": {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50},
                "total_size_estimate": 70202,
                "execution_summary": {
                    "backends": [],
                    "max_residual_ratio": None,
                    "process_failed": 1,
                    "timed_out": 1,
                },
                "gates": {"passed": False, "failures": [{"reason": "process_failed"}]},
                "evidence_classification": {
                    "classes": ["requested_operator_krylov_device_qi"],
                    "outcomes": ["process_failed"],
                    "tags": [
                        "failed_before_solver_trace_summary",
                        "requested_device_qi",
                        "requested_installed_krylov",
                        "requested_operator_krylov",
                    ],
                    "has_failed_before_summary_json": True,
                    "has_observed_installed_krylov": False,
                    "has_observed_coarse_reuse": False,
                    "promotion_eligible_seed_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = qi_seed.build_evidence_manifest(
        artifact_paths=[passing_path, failed_path],
        source_input=input_path,
        production_seed_count=5,
        production_timeout_s=3600.0,
    )

    current = manifest["current_evidence"]
    assert current["artifact_count"] == 2
    assert current["passing_artifact_count"] == 1
    assert current["nonpassing_artifact_count"] == 1
    assert current["max_checked_total_size"] == 23942
    assert current["largest_attempted_total_size"] == 70202
    assert current["largest_nonpassing_total_size"] == 70202
    assert current["max_checked_per_axis_resolution_fraction"] == 0.35
    assert current["bounded_lane_completion_estimate_percent"] == 35.0
    assert current["completion_estimate_basis"] == "largest passing measured artifact only"
    assert current["failed_before_summary_json_count"] == 1
    assert current["evidence_class_counts"]["requested_operator_krylov_device_qi"] == 1
    assert current["evidence_tag_counts"]["failed_before_solver_trace_summary"] == 1
    assert manifest["release_gate"] == "bounded_proxy"
    failed_artifact = next(
        artifact for artifact in manifest["source_artifacts"] if artifact["path"].endswith("failed.json")
    )
    assert failed_artifact["evidence_classes"] == ["requested_operator_krylov_device_qi"]
    assert "requested_installed_krylov" in failed_artifact["evidence_tags"]
    assert failed_artifact["run_outcomes"] == ["process_failed"]
    preset = manifest["probe_presets"]["operator-krylov-device-qi"]
    assert (
        preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT"]
        == "1"
    )
    assert preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV"] == "fgmres-jax"
    assert preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE"] == "1"
    assert preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK"] == "64"
    assert (
        preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE"]
        == "1"
    )
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS" not in preset["env"]
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH=64"
        in preset["recommended_command"]
    )
    assert (
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE=1"
        in preset["recommended_command"]
    )
    assert "--probe-preset operator-krylov-device-qi" in preset["recommended_command"]
    assert "--solve-method xblock_sparse_pc_gmres" in preset["recommended_command"]
    assert "--timeout-s 900" in preset["recommended_command"]
    current_preset = manifest["probe_presets"]["current-constraint-device-qi"]
    assert (
        current_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS"]
        == "1"
    )
    assert (
        current_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE"
        ]
        == "1"
    )
    assert "--probe-preset current-constraint-device-qi" in current_preset["recommended_command"]
    adjoint_preset = manifest["probe_presets"]["adjoint-krylov-device-qi"]
    assert (
        adjoint_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT"]
        == "1"
    )
    assert (
        adjoint_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH"]
        == "2"
    )
    assert "--probe-preset adjoint-krylov-device-qi" in adjoint_preset["recommended_command"]
    augmented_preset = manifest["probe_presets"]["augmented-krylov-device-qi"]
    assert augmented_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT"] == "1"
    assert augmented_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE"] == "cycle"
    assert (
        augmented_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV"]
        == "1"
    )
    assert "--probe-preset augmented-krylov-device-qi" in augmented_preset["recommended_command"]
    recycled_preset = manifest["probe_presets"]["recycled-augmented-device-qi"]
    assert recycled_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K"] == "32"
    assert recycled_preset["env"]["SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER"] == "960"
    assert (
        recycled_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV"]
        == "1"
    )
    assert "--probe-preset recycled-augmented-device-qi" in recycled_preset["recommended_command"]
    residual_snapshot_equation_preset = manifest["probe_presets"]["residual-snapshot-equation-device-qi"]
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert "--probe-preset residual-snapshot-equation-device-qi" in residual_snapshot_equation_preset[
        "recommended_command"
    ]
    assert "fail-closed" in residual_snapshot_equation_preset["description"]
    assembled_reuse_preset = manifest["probe_presets"]["assembled-reuse-device-qi"]
    assert assembled_reuse_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR"] == "1"
    assert assembled_reuse_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE"] == "1"
    assert (
        assembled_reuse_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED"]
        == "1"
    )
    assert (
        assembled_reuse_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"
        ]
        == "1"
    )
    assert "--probe-preset assembled-reuse-device-qi" in assembled_reuse_preset["recommended_command"]
    assert "assembled/operator-reuse" in assembled_reuse_preset["description"]
    composite_closure_preset = manifest["probe_presets"]["composite-closure-device-qi"]
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"
        ]
        == "1"
    )
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"
        ]
        == "1"
    )
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert "--probe-preset composite-closure-device-qi" in composite_closure_preset["recommended_command"]
    assert "composite residual-snapshot" in composite_closure_preset["description"]
    global_moment_preset = manifest["probe_presets"]["global-moment-closure-device-qi"]
    assert (
        global_moment_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        global_moment_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER"
        ]
        == "galerkin"
    )
    assert (
        global_moment_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "64"
    )
    assert "--probe-preset global-moment-closure-device-qi" in global_moment_preset["recommended_command"]
    assert "fail-closed" in global_moment_preset["description"]
    residual_galerkin_preset = manifest["probe_presets"]["residual-galerkin-device-qi"]
    assert (
        residual_galerkin_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"
        ]
        == "1"
    )
    assert (
        residual_galerkin_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK"
        ]
        == "8"
    )
    assert (
        residual_galerkin_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert "--probe-preset residual-galerkin-device-qi" in residual_galerkin_preset["recommended_command"]
    assert "fail-closed" in residual_galerkin_preset["description"]
    phase_space_preset = manifest["probe_presets"]["phase-space-coarse-reuse-device-qi"]
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "32"
    )
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert "--probe-preset phase-space-coarse-reuse-device-qi" in phase_space_preset[
        "recommended_command"
    ]
    assert "fail-closed" in phase_space_preset["description"]
    residual_bounce_preset = manifest["probe_presets"]["residual-bounce-region-device-qi"]
    assert (
        residual_bounce_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE"
        ]
        == "1"
    )
    assert (
        residual_bounce_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_MAX_RANK"
        ]
        == "48"
    )
    assert (
        residual_bounce_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_REGION_BOUNCE_COARSE_REGION_BANDS"
        ]
        == "bounce,trapped,passing"
    )
    assert "--probe-preset residual-bounce-region-device-qi" in residual_bounce_preset[
        "recommended_command"
    ]
    assert (
        "tests/qi_seed_robustness_scale060_residual_bounce_region_device_qi_gpu0"
        in residual_bounce_preset["recommended_command"]
    )
    assert (
        "docs/_static/qi_seed_robustness_scale060_residual_bounce_region_device_qi_gpu0.json"
        in residual_bounce_preset["recommended_command"]
    )
    assert "runtime hook" in residual_bounce_preset["description"]
    assert "fail-closed" in residual_bounce_preset["description"]
    block_schur_preset = manifest["probe_presets"]["block-schur-device-qi"]
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "64"
    )
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER"
        ]
        == "galerkin"
    )
    assert "--probe-preset block-schur-device-qi" in block_schur_preset["recommended_command"]
    assert "fail-closed" in block_schur_preset["description"]
    adaptive_residual_preset = manifest["probe_presets"]["adaptive-residual-device-qi"]
    assert (
        adaptive_residual_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER"
        ]
        == "adaptive_residual_equation"
    )
    assert (
        adaptive_residual_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING"
        ]
        == "block_hierarchy"
    )
    assert (
        adaptive_residual_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert "--probe-preset adaptive-residual-device-qi" in adaptive_residual_preset["recommended_command"]
    recommended = json.dumps(
        {
            "regeneration_commands": manifest["regeneration_commands"],
            "open_blockers": manifest["open_blockers"],
            "true_device_qi": manifest["release_claims"]["true_device_qi"],
        },
        sort_keys=True,
    ).lower()
    assert "operator-krylov" in recommended
    assert "global-moment" in recommended
    assert "residual-galerkin" in recommended
    assert "phase-space" in recommended
    assert "residual-bounce-region" in recommended
    assert "block-schur" in recommended
    assert "adaptive-residual" in recommended
    assert "recycled-augmented" in recommended
    assert "projected smoother" not in recommended
    assert "projected-smoother" not in recommended
