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
