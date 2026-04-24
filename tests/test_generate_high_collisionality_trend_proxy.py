from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_high_collisionality_trend_proxy.py"
    spec = importlib.util.spec_from_file_location("generate_high_collisionality_trend_proxy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_high_collisionality_trend_proxy_from_artifacts(tmp_path: Path) -> None:
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
            "high_collisionality_proxy_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "high_collisionality_proxy_test.png").exists()
    assert (out_dir / "high_collisionality_proxy_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "high_collisionality_trend_proxy"
    assert payload["cases"]["w7x"]["gates"]["fp_l11_l12_inverse_like"] is True
