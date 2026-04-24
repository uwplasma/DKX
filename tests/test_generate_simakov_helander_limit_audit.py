from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_simakov_helander_limit_audit.py"
    spec = importlib.util.spec_from_file_location("generate_simakov_helander_limit_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_simakov_helander_limit_audit_from_artifacts(tmp_path: Path) -> None:
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
            "simakov_audit_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "simakov_audit_test.png").exists()
    assert (out_dir / "simakov_audit_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "simakov_helander_limit_audit"
    assert payload["gates"]["appendix_b_geometry_inputs_available"] is True
    assert payload["gates"]["all_cases_ready_for_full_overlay"] is False
    assert payload["cases"]["lhd"]["state"] == "needs_wider_high_nu_scan"
    assert payload["cases"]["lhd"]["appendix_b_geometry_audit"]["geometry_scalars"]["FSABHat2_relative_error"] < 1e-12


def test_generate_simakov_helander_limit_audit_uses_checked_in_geometry_fallback(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    artifact_dir = repo / "examples" / "publication_figures" / "artifacts"
    summary_json = tmp_path / "summary.json"

    rc = mod.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--out-dir",
            str(tmp_path / "figures"),
            "--summary-json",
            str(summary_json),
            "--stem",
            "simakov_audit_missing_h5_test",
            "--lhd-geometry-output",
            str(tmp_path / "missing_lhd.h5"),
            "--w7x-geometry-output",
            str(tmp_path / "missing_w7x.h5"),
        ]
    )

    assert rc == 0
    payload = json.loads(summary_json.read_text())
    assert payload["gates"]["appendix_b_geometry_inputs_available"] is True
    assert payload["cases"]["w7x"]["appendix_b_geometry_audit"]["source_output"].endswith("sfincsOutput.h5")
