from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sfincs_jax.workflows.optimization_workflow import (
    build_candidate_scan_plan,
    er_values_from_bounds,
    write_candidate_scan_plan,
)


_REPO = Path(__file__).resolve().parents[1]


def test_candidate_scan_plan_builds_reproducible_commands(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    input_path = tmp_path / "input.namelist"
    proxy.write_text('{"workflow": "proxy", "objective_preset": "balanced"}\n', encoding="utf-8")
    input_path.write_text("&physicsParameters\n/\n", encoding="utf-8")

    plan = build_candidate_scan_plan(
        proxy_summary=proxy,
        input_namelist=input_path,
        out_dir=tmp_path / "scan",
        er_values=er_values_from_bounds(er_min=-2.0, er_max=2.0, n=3),
        jobs=2,
        impurity_species_index=2,
        target_impurity_flux=0.01,
    )

    payload = plan.as_dict()
    assert payload["er_values"] == [-2.0, 0.0, 2.0]
    assert "--compute-solution" in payload["scan_command"]
    assert "--skip-existing" in payload["scan_command"]
    assert payload["scan_command"][-2:] == ["--jobs", "2"]
    assert "--impurity-species-index" in payload["promotion_command"]


def test_candidate_scan_plan_serialization_is_deterministic(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    input_path = tmp_path / "input.namelist"
    scan_dir = tmp_path / "scan"
    proxy.write_text('{"workflow": "proxy"}\n', encoding="utf-8")
    input_path.write_text("&physicsParameters\n/\n", encoding="utf-8")
    er_values = er_values_from_bounds(er_min=-0.2, er_max=0.2, n=5)

    plan_a = build_candidate_scan_plan(
        proxy_summary=proxy,
        input_namelist=input_path,
        out_dir=scan_dir,
        er_values=er_values,
        compute_solution=False,
        compute_transport_matrix=True,
        jobs=1,
        skip_existing=False,
        require_electron_root=False,
    )
    plan_b = build_candidate_scan_plan(
        proxy_summary=proxy,
        input_namelist=input_path,
        out_dir=scan_dir,
        er_values=er_values,
        compute_solution=False,
        compute_transport_matrix=True,
        jobs=1,
        skip_existing=False,
        require_electron_root=False,
    )

    assert plan_a.as_dict() == plan_b.as_dict()
    assert plan_a.as_dict()["scan_command"] == [
        sys.executable,
        "-m",
        "sfincs_jax",
        "scan-er",
        "--input",
        str(input_path.resolve()),
        "--out-dir",
        str(scan_dir.resolve()),
        "--values",
        "-0.2",
        "-0.1",
        "0",
        "0.1",
        "0.2",
        "--compute-transport-matrix",
    ]
    assert "--compute-solution" not in plan_a.scan_command
    assert "--skip-existing" not in plan_a.scan_command
    assert "--allow-no-electron-root" in plan_a.promotion_command

    first = write_candidate_scan_plan(tmp_path / "first.json", plan_a, proxy_payload={"workflow": "proxy"})
    second = write_candidate_scan_plan(tmp_path / "second.json", plan_b, proxy_payload={"workflow": "proxy"})

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_write_candidate_scan_plan_embeds_proxy_metadata(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.json"
    input_path = tmp_path / "input.namelist"
    proxy.write_text("{}\n", encoding="utf-8")
    input_path.write_text("&physicsParameters\n/\n", encoding="utf-8")
    plan = build_candidate_scan_plan(
        proxy_summary=proxy,
        input_namelist=input_path,
        out_dir=tmp_path / "scan",
        er_values=(-1.0, 1.0),
    )
    out = write_candidate_scan_plan(
        tmp_path / "plan.json",
        plan,
        proxy_payload={
            "workflow": "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy",
            "objective_preset": "balanced",
            "final_components": {"bootstrap": 0.1},
            "autodiff_gradient_gate": {"status": "pass"},
        },
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["workflow"] == "sfincs_jax_optimization_candidate_scan_plan"
    assert payload["proxy_objective_preset"] == "balanced"
    assert payload["proxy_autodiff_gradient_gate"]["status"] == "pass"


def test_public_candidate_scan_launcher_dry_run(tmp_path: Path) -> None:
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
    script = _REPO / "examples" / "optimization" / "launch_sfincs_jax_candidate_scan.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--proxy-summary",
            str(proxy),
            "--input",
            str(input_path),
            "--out-dir",
            str(tmp_path / "scan"),
            "--er-min",
            "-1",
            "--er-max",
            "1",
            "--n-er",
            "3",
            "--impurity-species-index",
            "2",
            "--target-impurity-flux",
            "0.01",
        ],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((tmp_path / "scan" / "candidate_scan_plan.json").read_text(encoding="utf-8"))

    assert payload["er_values"] == [-1.0, 0.0, 1.0]
    assert payload["proxy_workflow"] == "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy"
