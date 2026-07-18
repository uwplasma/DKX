from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_OPTIMIZATION_DIR = _REPO / "examples" / "optimization"


def _local_vmex_root() -> Path | None:
    env_root = os.environ.get("DKX_VMEX_ROOT")
    candidates = [
        Path(env_root).expanduser() if env_root else None,
        Path("/Users/rogeriojorge/local/vmex"),
        Path("/Users/rogeriojorge/vmex"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        result_dir = candidate / "examples" / "optimization" / "results" / "qa_opt" / "ess"
        if (candidate / "vmex").is_dir() and (result_dir / "wout_final.nc").is_file():
            return candidate
    return None


def _run_script(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=_REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_artifacts(out_dir: Path, stem: str) -> dict:
    json_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    assert json_path.is_file()
    assert png_path.is_file()
    assert pdf_path.is_file()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
    return json.loads(json_path.read_text(encoding="utf-8"))


def test_public_optimization_scripts_show_help() -> None:
    scripts = {
        _OPTIMIZATION_DIR / "qa_nfp2_dkx_objectives.py": ["--out-dir", "--stem"],
        _OPTIMIZATION_DIR / "qa_nfp2_bootstrap_current_comparison.py": [
            "--vmex-root",
            "--comparison-result-dir",
        ],
        _OPTIMIZATION_DIR / "evaluate_dkx_promotion_scan.py": ["--out-dir", "--stem"],
        _OPTIMIZATION_DIR / "launch_dkx_candidate_scan.py": ["--out-dir", "--promotion-stem"],
        _OPTIMIZATION_DIR / "compare_dkx_promotion_runs.py": ["--out-dir", "--stem"],
    }

    for script, expected_flags in scripts.items():
        result = _run_script(script, ["--help"])

        assert "usage:" in result.stdout
        for flag in expected_flags:
            assert flag in result.stdout


def test_qa_nfp2_public_script_writes_fast_demo_artifacts(tmp_path: Path) -> None:
    stem = "qa_proxy_cli"
    script = _OPTIMIZATION_DIR / "qa_nfp2_dkx_objectives.py"

    _run_script(
        script,
        [
            "--steps",
            "0",
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
    )

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["workflow"] == "qa_nfp2_dkx_neoclassical_optimization_proxy"
    assert payload["nfp"] == 2
    assert payload["objective_preset"] == "balanced"
    assert payload["autodiff_gradient_gate"]["status"] == "pass"
    assert len(payload["history"]) == 1
    assert "required_high_fidelity_gates" in payload["promotion_plan"]


def test_qa_bootstrap_comparison_script_writes_fast_demo_artifacts(tmp_path: Path) -> None:
    stem = "qa_bootstrap_comparison_cli"
    script = _OPTIMIZATION_DIR / "qa_nfp2_bootstrap_current_comparison.py"
    vmex_root = _local_vmex_root()
    if vmex_root is None:
        pytest.skip("vmex QA_optimization.py result is not available")

    _run_script(
        script,
        [
            "--vmex-root",
            str(vmex_root),
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
    )

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["workflow"] == "dkx_vmex_qa_optimization_current_diagnostic"
    assert payload["nfp"] == 2
    assert payload["targets"] == {"aspect_ratio": 5.0, "iota": 0.41}
    assert payload["qa_optimization"]["gate"]["status"] == "pass"
    assert abs(payload["qa_optimization"]["metrics"]["mean_iota"] - 0.41) < 2.0e-2
    assert "not a completed high-fidelity dkx kinetic bootstrap-current claim" in payload["claim_boundary"]
    assert payload["comparison"]["status"] == "baseline_only"
    assert "dkx scan-er" in " ".join(payload["promotion_plan"]["required_gates"])


def test_vmex_bootstrap_optimization_script_is_reviewable_max_mode3() -> None:
    script = _OPTIMIZATION_DIR / "QA_optimization_bootstrap_current.py"
    text = script.read_text(encoding="utf-8")

    compile(text, str(script), "exec")
    assert "MAX_MODE = 3" in text
    assert "INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE = False" in text
    assert "vj.JDotB" in text
    assert "RedlBootstrapMismatch" in text
    assert "DKX_VMEX_ROOT" in text


def test_promotion_public_script_writes_fast_demo_artifacts(tmp_path: Path) -> None:
    stem = "promotion_cli"
    script = _OPTIMIZATION_DIR / "evaluate_dkx_promotion_scan.py"

    _run_script(
        script,
        [
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
            "--impurity-species-index",
            "2",
        ],
    )

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["workflow"] == "dkx_optimization_high_fidelity_promotion"
    assert payload["gate_status"] == "pass"
    assert payload["selected_root"]["root_type"] == "electron"
    assert payload["bootstrap_objective"] > 0.0
    assert payload["flux_objective"]["mean_impurity_flux"] > 0.0
    assert len(payload["runs"]) == 4


def test_promotion_public_script_allows_two_species_scan_without_impurity_objective(tmp_path: Path) -> None:
    stem = "promotion_two_species_cli"
    scan_dir = tmp_path / "scan"
    script = _OPTIMIZATION_DIR / "evaluate_dkx_promotion_scan.py"

    from dkx.io import write_sfincs_h5

    for er, current in [(-0.3, -1.0), (0.3, -0.2), (1.0, 0.3), (3.0, 1.4)]:
        run_dir = scan_dir / f"Er{er:g}"
        run_dir.mkdir(parents=True)
        write_sfincs_h5(
            path=run_dir / "sfincsOutput.h5",
            data={
                "Er": er,
                "Nspecies": 2,
                "Zs": [1.0, -1.0],
                "particleFlux_vm_rHat": [[0.1 * er], [0.1 * er - current]],
                "heatFlux_vm_rHat": [[0.01], [0.02]],
                "FSABjHatOverRootFSAB2": [0.01 * er],
                "linearSolverResidualNorm": 1.0e-10,
                "linearSolverResidualTarget": 1.0e-8,
            },
            overwrite=True,
        )

    _run_script(
        script,
        [
            "--scan-dir",
            str(scan_dir),
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
            "--require-electron-root",
        ],
    )

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["gate_status"] == "pass"
    assert payload["selected_root"]["root_type"] == "electron"
    assert payload["flux_objective"] is None


def test_candidate_scan_launcher_and_comparison_script_write_artifacts(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    input_path = tmp_path / "input.namelist"
    proxy.write_text(
        json.dumps(
            {
                "workflow": "qa_nfp2_dkx_neoclassical_optimization_proxy",
                "objective_preset": "balanced",
                "final_components": {"bootstrap": 0.1},
                "autodiff_gradient_gate": {"status": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    input_path.write_text("&physicsParameters\n/\n", encoding="utf-8")

    _run_script(
        _OPTIMIZATION_DIR / "launch_dkx_candidate_scan.py",
        [
            "--proxy-summary",
            str(proxy),
            "--input",
            str(input_path),
            "--out-dir",
            str(tmp_path / "candidate_scan"),
            "--er-min",
            "-1",
            "--er-max",
            "1",
            "--n-er",
            "3",
        ],
    )
    plan = json.loads((tmp_path / "candidate_scan" / "candidate_scan_plan.json").read_text(encoding="utf-8"))
    assert plan["workflow"] == "dkx_optimization_candidate_scan_plan"
    assert plan["er_values"] == [-1.0, 0.0, 1.0]

    promotion = {
        "workflow": "dkx_optimization_high_fidelity_promotion",
        "selected_root": {"er": 0.5, "root_type": "electron"},
        "bootstrap_objective": 0.01,
        "flux_objective": {"total": 0.02},
        "gate_status": "pass",
    }
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps(promotion) + "\n", encoding="utf-8")
    gpu.write_text(json.dumps({**promotion, "bootstrap_objective": 0.010000001}) + "\n", encoding="utf-8")
    stem = "comparison_cli"
    _run_script(
        _OPTIMIZATION_DIR / "compare_dkx_promotion_runs.py",
        [
            "--cpu",
            str(cpu),
            "--gpu",
            str(gpu),
            "--bootstrap-rtol",
            "1e-3",
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
    )
    comparison = _assert_artifacts(tmp_path, stem)
    assert comparison["workflow"] == "dkx_optimization_promotion_comparison"
    assert comparison["status"] == "pass"


def test_comparison_script_returns_two_when_cpu_gpu_gate_fails(tmp_path: Path) -> None:
    promotion = {
        "workflow": "dkx_optimization_high_fidelity_promotion",
        "selected_root": {"er": 0.5, "root_type": "electron"},
        "bootstrap_objective": 0.01,
        "flux_objective": {"total": 0.02},
        "gate_status": "pass",
    }
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps(promotion) + "\n", encoding="utf-8")
    gpu.write_text(
        json.dumps(
            {
                **promotion,
                "selected_root": {"er": 0.75, "root_type": "electron"},
                "bootstrap_objective": 0.02,
                "flux_objective": {"total": 0.05},
                "gate_status": "fail",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    stem = "comparison_cli_fail"
    result = subprocess.run(
        [
            sys.executable,
            str(_OPTIMIZATION_DIR / "compare_dkx_promotion_runs.py"),
            "--cpu",
            str(cpu),
            "--gpu",
            str(gpu),
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
        cwd=_REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 2
    assert "status:  fail" in result.stdout
    comparison = _assert_artifacts(tmp_path, stem)
    assert comparison["status"] == "fail"
    assert any("gpu gate_status" in failure for failure in comparison["failures"])
    assert any("selected_root_er differs" in failure for failure in comparison["failures"])
