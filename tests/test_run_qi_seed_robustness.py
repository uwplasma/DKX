from __future__ import annotations

import importlib.util
import json
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
    result = manifest["execution"]["results"][0]
    assert (out_root / result["stdout"]).read_text(encoding="utf-8") == "ok\n"
    assert (out_root / result["stderr"]).read_text(encoding="utf-8") == ""
