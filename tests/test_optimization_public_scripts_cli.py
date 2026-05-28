from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_OPTIMIZATION_DIR = _REPO / "examples" / "optimization"


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
        timeout=20,
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
        _OPTIMIZATION_DIR / "qa_nfp2_sfincs_jax_objectives.py": ["--out-dir", "--stem"],
        _OPTIMIZATION_DIR / "screen_qi_electron_root_nfp.py": ["--candidates", "--target-electron-root-drive"],
        _OPTIMIZATION_DIR / "evaluate_sfincs_jax_promotion_scan.py": ["--out-dir", "--stem"],
        _OPTIMIZATION_DIR / "launch_sfincs_jax_candidate_scan.py": ["--out-dir", "--promotion-stem"],
        _OPTIMIZATION_DIR / "compare_sfincs_jax_promotion_runs.py": ["--out-dir", "--stem"],
        _OPTIMIZATION_DIR / "run_promotion_evidence_campaign.py": ["--run-cpu", "--run-gpu", "--run-fortran"],
    }

    for script, expected_flags in scripts.items():
        result = _run_script(script, ["--help"])

        assert "usage:" in result.stdout
        for flag in expected_flags:
            assert flag in result.stdout


def test_qa_nfp2_public_script_writes_fast_demo_artifacts(tmp_path: Path) -> None:
    stem = "qa_proxy_cli"
    script = _OPTIMIZATION_DIR / "qa_nfp2_sfincs_jax_objectives.py"

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
    assert payload["workflow"] == "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy"
    assert payload["nfp"] == 2
    assert payload["objective_preset"] == "balanced"
    assert payload["autodiff_gradient_gate"]["status"] == "pass"
    assert len(payload["history"]) == 1
    assert "required_high_fidelity_gates" in payload["promotion_plan"]


def test_qi_screen_public_script_pivots_to_qi_nfp2_when_qa_is_deferred(tmp_path: Path) -> None:
    stem = "qi_screen_cli"
    script = _OPTIMIZATION_DIR / "screen_qi_electron_root_nfp.py"

    _run_script(
        script,
        [
            "--steps",
            "2",
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
    )

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["workflow"] == "sfincs_jax_qi_qa_electron_root_nfp_screening_proxy"
    assert "not a kinetic SFINCS transport claim" in payload["claim_boundary"]
    assert payload["recommended_candidate"]["candidate"] == "qi:nfp2"
    assert payload["recommended_candidate"]["symmetry"] == "qi"
    assert payload["recommended_candidate"]["nfp"] == 2
    assert "sfincs_jax scan-er" in " ".join(payload["promotion_plan"]["next_commands"])


def test_promotion_public_script_writes_fast_demo_artifacts(tmp_path: Path) -> None:
    stem = "promotion_cli"
    script = _OPTIMIZATION_DIR / "evaluate_sfincs_jax_promotion_scan.py"

    _run_script(script, ["--out-dir", str(tmp_path), "--stem", stem])

    payload = _assert_artifacts(tmp_path, stem)
    assert payload["workflow"] == "sfincs_jax_optimization_high_fidelity_promotion"
    assert payload["gate_status"] == "pass"
    assert payload["selected_root"]["root_type"] == "electron"
    assert payload["bootstrap_objective"] > 0.0
    assert payload["flux_objective"]["mean_impurity_flux"] > 0.0
    assert len(payload["runs"]) == 4


def test_candidate_scan_launcher_and_comparison_script_write_artifacts(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    input_path = tmp_path / "input.namelist"
    proxy.write_text(
        json.dumps(
            {
                "workflow": "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy",
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
        _OPTIMIZATION_DIR / "launch_sfincs_jax_candidate_scan.py",
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
    assert plan["workflow"] == "sfincs_jax_optimization_candidate_scan_plan"
    assert plan["er_values"] == [-1.0, 0.0, 1.0]

    promotion = {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
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
        _OPTIMIZATION_DIR / "compare_sfincs_jax_promotion_runs.py",
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
    assert comparison["workflow"] == "sfincs_jax_optimization_promotion_comparison"
    assert comparison["status"] == "pass"


def test_comparison_script_returns_two_when_cpu_gpu_gate_fails(tmp_path: Path) -> None:
    promotion = {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
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
            str(_OPTIMIZATION_DIR / "compare_sfincs_jax_promotion_runs.py"),
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
