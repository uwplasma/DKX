from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mapped_xgrid_transport_evidence.py"
spec = importlib.util.spec_from_file_location("run_mapped_xgrid_transport_evidence", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _get(mapping: dict, key: str, default):
    for existing_key, value in mapping.items():
        if existing_key.lower() == key.lower():
            return value
    return default


def _fake_transport_solve(*, nml, **_kwargs):
    other = nml.group("otherNumericalParameters")
    resolution = nml.group("resolutionParameters")
    x_scheme = int(_get(other, "XGRIDSCHEME", 0))
    log_length = float(_get(other, "MAPPEDXGRIDLOGLENGTH", 0.0))
    scale = 1.0 if x_scheme != 50 else 1.0 + 0.2 * (log_length + 0.5) ** 2
    base = np.asarray([[1.0, 0.2], [0.1, 2.0]], dtype=np.float64)
    return SimpleNamespace(
        op0=SimpleNamespace(total_size=12, n_x=int(_get(resolution, "NX", 0))),
        transport_matrix=scale * base,
        state_vectors_by_rhs={1: np.ones(2), 2: np.ones(2)},
        residual_norms_by_rhs={1: 1.0e-10 * scale, 2: 2.0e-10 * scale},
        rhs_norms_by_rhs={1: 1.0, 2: 2.0},
        fsab_flow=np.zeros((1, 2)),
        particle_flux_vm_psi_hat=np.zeros((1, 2)),
        heat_flux_vm_psi_hat=np.zeros((1, 2)),
        elapsed_time_s=np.asarray([0.01, 0.02]) * scale,
        transport_output_fields=None,
        active_size=8,
        use_active_dof_mode=True,
        solver_kinds_by_rhs={1: "sparse_lu", 2: "gmres"},
        solve_methods_by_rhs={1: "sparse_lu", 2: "incremental"},
    )


def test_parse_log_lengths_accepts_commas_and_spaces():
    assert mod._parse_log_lengths("-1, -0.5 0.0") == (-1.0, -0.5, 0.0)


def test_main_writes_json_and_csv_artifacts(tmp_path: Path):
    input_path = Path(__file__).parent / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    json_path = tmp_path / "mapped_evidence.json"
    csv_path = tmp_path / "mapped_evidence.csv"

    rc = mod.main(
        [
            str(input_path),
            "--json-out",
            str(json_path),
            "--csv-out",
            str(csv_path),
            "--log-lengths=-0.5,0.0",
            "--reference-nx",
            "9",
        ],
        solve_fn=_fake_transport_solve,
    )

    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["reference_nx"] == 9
    assert payload["best_by_transport_error_log_length"] == -0.5
    assert payload["rows"][0]["solver_kinds"] == ["gmres", "sparse_lu"]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["reference_n_x"] == "9"
    assert rows[0]["solve_methods"] == "incremental;sparse_lu"


def test_main_writes_case_preset_artifacts(tmp_path: Path):
    json_path = tmp_path / "mapped_evidence.json"
    csv_path = tmp_path / "mapped_evidence.csv"

    rc = mod.main(
        [
            "--case",
            "reduced_pas_tokamak_rhsmode2",
            "--json-out",
            str(json_path),
            "--csv-out",
            str(csv_path),
            "--log-lengths=-1.0,0.5",
            "--candidate-nx",
            "6",
            "--reference-nx",
            "10",
        ],
        solve_fn=_fake_transport_solve,
    )

    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["case"] == "reduced_pas_tokamak_rhsmode2"
    assert payload["metadata"]["candidate_resolution"]["nx"] == 6
    assert payload["metadata"]["reference_nx"] == 10
    assert payload["metadata"]["log_lengths"] == [-1.0, 0.5]
    assert payload["reference_summary"]["n_x"] == 10
    assert [row["n_x"] for row in payload["rows"]] == [6, 6]
    assert payload["best_by_transport_error"]["log_length"] == -1.0


def test_main_writes_scorecard_artifacts(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    tiny = root / "docs" / "_static" / "mapped_xgrid_transport_evidence_rhsmode2_tiny.json"
    reduced = (
        root
        / "docs"
        / "_static"
        / "mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.json"
    )
    json_path = tmp_path / "mapped_scorecard.json"
    csv_path = tmp_path / "mapped_scorecard.csv"

    rc = mod.main(
        [
            "--scorecard",
            str(tiny),
            str(reduced),
            "--scorecard-json-out",
            str(json_path),
            "--scorecard-csv-out",
            str(csv_path),
        ],
        solve_fn=_fake_transport_solve,
    )

    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "mapped_xgrid_transport_scorecard"
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["useful_count"] == 1
    assert payload["summary"]["negative_count"] == 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mapped_classification"] for row in rows] == ["negative", "useful"]


def test_list_cases_prints_presets(capsys):
    rc = mod.main(["--list-cases"], solve_fn=_fake_transport_solve)

    assert rc == 0
    captured = capsys.readouterr()
    assert "tiny_pas_rhsmode2_scheme2" in captured.out
    assert "reduced_pas_tokamak_rhsmode2" in captured.out
