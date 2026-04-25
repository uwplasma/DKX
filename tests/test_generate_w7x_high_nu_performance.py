from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_w7x_high_nu_performance.py"
    spec = importlib.util.spec_from_file_location("generate_w7x_high_nu_performance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_w7x_high_nu_summary_tracks_residual_and_setup_gates() -> None:
    mod = _load_module()
    records = [
        {
            "id": "bounded_30k_krylov",
            "label": "failed",
            "elapsed_s": 400.0,
            "sparse_factorizations": 0,
            "relative_residuals": [0.7, 0.9, 1.0],
        },
        {
            "id": "sparse_lu_no_reuse",
            "label": "no reuse",
            "elapsed_s": 2000.0,
            "sparse_factorizations": 3,
            "max_relative_residual": 8.0e-7,
        },
        {
            "id": "sparse_lu_factor_reuse",
            "label": "reuse",
            "elapsed_s": 800.0,
            "sparse_factorizations": 1,
            "max_relative_residual": 7.0e-7,
        },
    ]

    payload = mod.build_w7x_high_nu_performance_summary(records, residual_gate=1.0e-6)

    assert payload["metadata"]["kind"] == "w7x_high_nu_preconditioning_performance"
    assert payload["gates"]["failed_route_rejected"] is True
    assert payload["gates"]["factor_reuse_residual_clean"] is True
    assert payload["gates"]["factor_reuse_fewer_factorizations"] is True
    assert payload["gates"]["factor_reuse_speedup_vs_no_reuse"] == 2.5
    assert payload["gates"]["factor_reuse_wall_time_saved_s"] == 1200.0


def test_generate_w7x_high_nu_performance_writes_figure_and_summary(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"

    rc = mod.main(
        [
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "w7x_high_nu_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "w7x_high_nu_test.png").exists()
    assert (out_dir / "w7x_high_nu_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "w7x_high_nu_preconditioning_performance"
    assert payload["gates"]["factor_reuse_present"] is True
