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
    assert {workflow["id"] for workflow in bootstrap} >= {"12_bootstrap_vs_redl", "qa_bootstrap_vmex_chain"}

    vmec_geometry = module._matching_workflows(catalog, topic="vmec", search="geometry")
    assert {workflow["id"] for workflow in vmec_geometry} >= {"04_vmec_geometry"}

    transport = module._matching_workflows(catalog, topic="transport", search="")
    assert {workflow["id"] for workflow in transport} >= {"10_transport_matrix"}
    assert module._matching_workflows(catalog, topic="gpu", search="") == []


def test_workflow_browser_json_cli_is_machine_readable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--topic", "ambipolar", "--json"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert [workflow["id"] for workflow in payload["workflows"]] == ["08_ambipolar_er"]
    assert payload["workflows"][0]["command"].startswith("python examples/")


def test_workflow_browser_text_cli_guides_first_run() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--topic", "onsager", "--long"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "10_transport_matrix" in result.stdout
    assert "python examples/tutorials/10_transport_matrix.py" in result.stdout
    assert "local SFINCS Fortran v3 required for first run: no" in result.stdout


def test_workflow_browser_lists_topics() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-topics"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "tutorials" in result.stdout
    assert "[learning]" in result.stdout
    assert "advanced" in result.stdout
    assert "[capability]" in result.stdout
    assert "sfincs_examples" in result.stdout
    assert "[reference]" in result.stdout
