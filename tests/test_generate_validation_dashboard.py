from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_validation_dashboard.py"
    spec = importlib.util.spec_from_file_location("generate_validation_dashboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_validation_dashboard_from_checked_in_artifacts(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    artifact_dir = repo / "examples" / "publication_figures" / "artifacts"
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"

    rc = mod.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "validation_dashboard_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "validation_dashboard_test.png").exists()
    assert (out_dir / "validation_dashboard_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "publication_validation_dashboard"
    assert payload["collisionality"]["lhd"]["labels"] == ["Fokker-Planck", "PAS"]
    assert payload["trajectory_sweeps"]["tokamak"]["zero_field_spread"]["fsab_flow"] == 0.0
