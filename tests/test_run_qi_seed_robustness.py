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
    assert manifest["execution"]["summary"]["max_residual_ratio"] == 2.0e5
    assert manifest["execution"]["summary"]["converged"] == 0
    assert manifest["execution"]["summary"]["solve_methods"] == ["dense"]
    assert manifest["execution"]["gates"]["passed"] is True


def test_qi_seed_runner_writes_compact_summary_artifact(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    input_path = tmp_path / "source" / "input.namelist"
    _write_qi_input(input_path)
    out_root = tmp_path / "lane"
    summary_path = tmp_path / "qi_seed_summary.json"

    def fake_run(command, *, cwd, stdout, stderr, timeout, check):  # noqa: ANN001
        case_dir = Path(command[command.index("--input") + 1]).parent
        if case_dir.name.endswith("0000"):
            output_path = Path(command[command.index("--out") + 1])
            trace_path = Path(command[command.index("--solver-trace") + 1])
            output_path.write_bytes(b"h5")
            trace_path.write_text(
                json.dumps(
                    {
                        "backend": "cpu",
                        "converged": True,
                        "elapsed_s": 0.5,
                        "residual_norm": 4.0e-12,
                        "residual_target": 1.0e-11,
                        "selected_path": "rhsmode1_solution",
                        "solve_method": "auto",
                        "metadata": {"solver_metadata": {"accepted_converged": True}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)
        stderr.write("mock failure\n")
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
                "0",
                "1",
                "--execute",
                "--summary-output",
                str(summary_path),
                "--clean",
            ]
        )
        == 1
    )

    artifact = json.loads(summary_path.read_text(encoding="utf-8"))
    assert artifact["artifact_kind"] == "qi_seed_execution_summary"
    assert artifact["execution_summary"]["process_passed"] == 1
    assert artifact["execution_summary"]["process_failed"] == 1
    assert artifact["execution_summary"]["outputs_written"] == 1
    assert artifact["execution_summary"]["solver_traces_written"] == 1
    assert artifact["seeds"][0]["returncode"] == 0
    assert artifact["seeds"][0]["solver_trace_exists"] is True
    assert artifact["seeds"][0]["residual_ratio"] == 0.4
    assert artifact["seeds"][1]["returncode"] == 2
    assert artifact["seeds"][1]["solver_trace_exists"] is False


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
