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
    assert seed["last_progress_residual_event"] == progress_events[3]
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
    assert manifest["release_gate"] == "bounded_proxy"
