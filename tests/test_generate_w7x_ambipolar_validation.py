from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    assert payload["metadata"]["validation_scope"] == "w7x_like_scaffold"
    assert payload["acceptance_gates"]["finite_ambipolar_roots"] is True
    assert payload["acceptance_gates"]["radial_current_brackets_zero"] is True
    assert payload["acceptance_gates"]["ion_root_candidate"] is True
    assert payload["acceptance_gates"]["provenance_complete"] is False
    assert payload["acceptance_gates"]["ready_for_literature_claim"] is False
    assert len(payload["runs"]) == 3
    assert payload["runs"][0]["outputs"]["heatFlux_vm_rHat"] == 3.0
    assert payload["ambipolar"]["roots_er"] == [-0.25]
    assert payload["ambipolar"]["outputs_at_roots"]["FSABjHat"] == [-0.1]


def test_build_summary_payload_promotes_only_with_complete_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    ambi = AmbipolarSolveResult(
        var_name="Er",
        var_values=np.asarray([1.0, 0.0, -1.0], dtype=np.float64),
        er_values=np.asarray([1.0, 0.0, -1.0], dtype=np.float64),
        radial_currents=np.asarray([1.0, 0.0, -1.0], dtype=np.float64),
        roots_var=np.asarray([-0.2], dtype=np.float64),
        roots_er=np.asarray([-0.2], dtype=np.float64),
        root_types=["ion"],
        outputs_labels=["heatFlux_vm_rHat"],
        outputs_by_run=np.asarray([[3.0], [2.0], [1.0]], dtype=np.float64),
        outputs_at_roots=[np.asarray([1.5], dtype=np.float64)],
        radius_wish=0.5,
        radius_actual=0.49,
    )
    provenance = {
        "equilibrium_source": "wout_reference.nc",
        "profile_source": "published profile table",
        "configuration_or_shot": "W7-X reference discharge",
        "literature_reference": "https://doi.org/10.1088/1741-4326/ab6ea8",
    }

    payload = mod.build_summary_payload(
        base_input=mod.DEFAULT_W7X_INPUT,
        scan_dir=tmp_path / "scan",
        requested_er_values=[1.0, 0.0, -1.0],
        ambipolar_result=ambi,
        fast=False,
        provenance=provenance,
    )

    assert payload["metadata"]["validation_scope"] == "w7x_literature_validation"
    assert payload["acceptance_gates"]["ready_for_literature_claim"] is True
    assert payload["provenance"]["profile_source"] == "published profile table"


def test_checked_in_w7x_provenance_template_is_incomplete_by_design() -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    template = (
        repo
        / "examples"
        / "publication_figures"
        / "provenance"
        / "w7x_ambipolar_provenance_template.json"
    )
    payload = mod.load_provenance_json(template)
    assert payload["schema_version"] == 1
    for key in payload["required_fields"]:
        assert key in payload
    assert all(payload[key] == "" for key in payload["required_fields"])


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


def test_scan_only_forwards_resume_and_split_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    input_path = repo / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"

    scan_calls: list[dict[str, object]] = []

    def _fake_run_er_scan(**kwargs):
        scan_calls.append(dict(kwargs))
        scan_dir = Path(kwargs["out_dir"])
        scan_dir.mkdir(parents=True, exist_ok=True)
        for value in kwargs["values"]:
            run_dir = scan_dir / f"Er{float(value):.4g}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "sfincsOutput.h5").write_bytes(b"")
        return None

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("ambipolar postprocessing should be skipped in --scan-only mode")

    monkeypatch.setattr(mod, "run_er_scan", _fake_run_er_scan)
    monkeypatch.setattr(mod, "solve_ambipolar_from_scan_dir", _unexpected)
    monkeypatch.setattr(mod, "plot_w7x_ambipolar_summary", _unexpected)

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
            "--skip-existing",
            "--scan-only",
            "--jobs",
            "2",
            "--index",
            "1",
            "--stride",
            "3",
        ]
    )

    assert rc == 0
    assert len(scan_calls) == 1
    kwargs = scan_calls[0]
    assert kwargs["skip_existing"] is True
    assert kwargs["jobs"] == 2
    assert kwargs["index"] == 1
    assert kwargs["stride"] == 3
    assert not summary_json.exists()
    assert not out_dir.exists()


def test_main_rejects_scan_only_and_plot_only(tmp_path: Path) -> None:
    mod = _load_module()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text("{}\n")
    with pytest.raises(ValueError, match="Cannot combine --scan-only and --plot-only"):
        mod.main(
            [
                "--summary-json",
                str(summary_json),
                "--scan-only",
                "--plot-only",
            ]
        )
