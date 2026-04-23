from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from sfincs_jax.ambipolar import AmbipolarSolveResult


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_w7x_ambipolar_validation.py"
    spec = importlib.util.spec_from_file_location("generate_w7x_ambipolar_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_er_values_uses_scan_comment_bracket() -> None:
    mod = _load_module()
    values = mod.resolve_er_values(input_path=mod.DEFAULT_W7X_INPUT, er_values_arg="", n_points=5)
    np.testing.assert_allclose(np.asarray(values), np.asarray([1.0, 0.0, -1.0, -2.0, -3.0]))


def test_build_summary_payload_serializes_runs_and_roots(tmp_path: Path) -> None:
    mod = _load_module()
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    ambi = AmbipolarSolveResult(
        var_name="Er",
        var_values=np.asarray([1.0, 0.0, -1.0], dtype=np.float64),
        er_values=np.asarray([1.0, 0.0, -1.0], dtype=np.float64),
        radial_currents=np.asarray([2.0, 0.5, -1.0], dtype=np.float64),
        roots_var=np.asarray([-0.25], dtype=np.float64),
        roots_er=np.asarray([-0.25], dtype=np.float64),
        root_types=["ion"],
        outputs_labels=["heatFlux_vm_rHat", "FSABjHat"],
        outputs_by_run=np.asarray([[3.0, 0.2], [2.0, 0.1], [1.0, -0.3]], dtype=np.float64),
        outputs_at_roots=[np.asarray([1.5], dtype=np.float64), np.asarray([-0.1], dtype=np.float64)],
        radius_wish=0.5,
        radius_actual=0.49,
    )
    payload = mod.build_summary_payload(
        base_input=mod.DEFAULT_W7X_INPUT,
        scan_dir=scan_dir,
        requested_er_values=[1.0, 0.0, -1.0],
        ambipolar_result=ambi,
        fast=True,
    )
    assert payload["metadata"]["kind"] == "w7x_ambipolar_validation_scaffold"
    assert payload["metadata"]["fast"] is True
    assert len(payload["runs"]) == 3
    assert payload["runs"][0]["outputs"]["heatFlux_vm_rHat"] == 3.0
    assert payload["ambipolar"]["roots_er"] == [-0.25]
    assert payload["ambipolar"]["outputs_at_roots"]["FSABjHat"] == [-0.1]


def test_generate_w7x_ambipolar_validation_tiny_fixture_end_to_end(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    input_path = repo / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    out_dir = tmp_path / "figures"
    work_dir = tmp_path / "work"
    summary_json = tmp_path / "summary.json"
    rc = mod.main(
        [
            "--input",
            str(input_path),
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--er-values",
            "1e-3,0,-1e-3",
            "--stem",
            "tiny_w7x_ambi",
            "--title",
            "Tiny ambipolar validation",
        ]
    )
    assert rc == 0
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["source_script"] == "examples/publication_figures/generate_w7x_ambipolar_validation.py"
    assert payload["metadata"]["requested_er_values"] == [0.001, 0.0, -0.001]
    assert len(payload["runs"]) == 3
    assert "roots_er" in payload["ambipolar"]
    assert (out_dir / "tiny_w7x_ambi.png").exists()
    assert (out_dir / "tiny_w7x_ambi.pdf").exists()
