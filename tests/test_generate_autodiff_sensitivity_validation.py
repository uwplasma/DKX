from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_autodiff_sensitivity_validation.py"
    spec = importlib.util.spec_from_file_location("generate_autodiff_sensitivity_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_autodiff_sensitivity_validation_plot_only_from_checked_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    summary_json = (
        repo
        / "examples"
        / "publication_figures"
        / "artifacts"
        / "sfincs_jax_autodiff_sensitivity_validation_summary.json"
    )
    out_dir = tmp_path / "figures"

    rc = mod.main(
        [
            "--plot-only",
            "--summary-json",
            str(summary_json),
            "--out-dir",
            str(out_dir),
            "--gradient-stem",
            "gradient_check_test",
            "--sensitivity-stem",
            "sensitivity_map_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "gradient_check_test.png").exists()
    assert (out_dir / "gradient_check_test.pdf").exists()
    assert (out_dir / "sensitivity_map_test.png").exists()
    assert (out_dir / "sensitivity_map_test.pdf").exists()

    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "autodiff_sensitivity_validation"
    assert payload["gates"]["gradient_relative_error_ok"] is True
    assert payload["gates"]["primal_residual_ok"] is True
    assert payload["gates"]["adjoint_residual_ok"] is True
    assert payload["geometry_sensitivity"]["gradient_relative_error"] < 1.0e-8
