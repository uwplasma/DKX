from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "list_workflows.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("list_workflows", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_browser_filters_by_physics_topic() -> None:
    module = _load_module()
    catalog = module._load_catalog()

    bootstrap = module._matching_workflows(catalog, topic="bootstrap", search="")
    assert {workflow["id"] for workflow in bootstrap} >= {"bootstrap_redl", "qa_optimization_objective"}

    vmec_geometry = module._matching_workflows(catalog, topic="vmec", search="geometry")
    assert {workflow["id"] for workflow in vmec_geometry} >= {"vmec_wout_path", "vmec_boozer_jax_pipeline"}

    gpu = module._matching_workflows(catalog, topic="gpu", search="")
    assert {workflow["id"] for workflow in gpu} >= {"output_format_benchmark"}


def test_workflow_browser_json_cli_is_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--topic", "redl", "--json"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert [workflow["id"] for workflow in payload["workflows"]] == ["bootstrap_redl"]
    assert payload["workflows"][0]["command"].startswith("python examples/")


def test_workflow_browser_text_cli_guides_first_run() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--topic", "transport", "--long"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "transport_matrix" in result.stdout
    assert "python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py" in result.stdout
    assert "local SFINCS Fortran v3 required for first run: no" in result.stdout


def test_workflow_browser_lists_topics() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-topics"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "getting_started" in result.stdout
    assert "[learning]" in result.stdout
    assert "vmec_jax_finite_beta" in result.stdout
    assert "[capability]" in result.stdout
    assert "sfincs_examples" in result.stdout
    assert "[reference]" in result.stdout
    assert "Finite-beta VMEC" in result.stdout
