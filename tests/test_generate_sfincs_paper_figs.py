from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_sfincs_paper_figs.py"
    spec = importlib.util.spec_from_file_location("generate_sfincs_paper_figs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_transport_output(
    run_dir: Path,
    *,
    nu_n: float,
    diagonal_scale: float = 1.0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(run_dir / "sfincsOutput.h5", "w") as h5:
        h5["nu_n"] = float(nu_n)
        h5["GHat"] = 1.0
        h5["IHat"] = 0.0
        h5["iota"] = 0.0
        h5["B0OverBBar"] = 1.0
        h5["transportMatrix"] = np.asarray(
            [[diagonal_scale * nu_n, 0.1 * diagonal_scale], [0.2 * diagonal_scale, 2.0 * diagonal_scale * nu_n]],
            dtype=float,
        )


def test_write_scan_input_replaces_collision_operator_and_fast_resolution(tmp_path: Path) -> None:
    mod = _load_module()
    base_input = tmp_path / "input.namelist"
    base_input.write_text(
        "!ss scanType = 1\n"
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "  Er = 0.0\n"
        "/\n"
        "&resolutionParameters\n"
        "  Ntheta = 13\n"
        "  Nzeta = 31\n"
        "  Nxi = 24\n"
        "  Nx = 6\n"
        "  solverTolerance = 1d-6\n"
        "/\n"
    )
    dest = tmp_path / "scan_input.namelist"
    mod._write_scan_input(
        base_input=base_input,
        dest=dest,
        nu_n_min=1e-2,
        nu_n_max=1e0,
        n_points=4,
        collision_operator=1,
        fast=True,
    )
    text = dest.read_text()
    assert text.count("collisionOperator = 1") == 1
    assert "collisionOperator = 0" not in text
    assert "Ntheta = 5" in text
    assert "Nzeta = 5" in text
    assert "Nxi = 3" in text
    assert "NL = 3" in text
    assert "Nx = 3" in text
    assert "solverTolerance = 1e-4" in text
    assert "Ntheta = 13" not in text
    assert "Nzeta = 31" not in text
    assert "&resolutionParameters\n" in text
    resolution_block = text.split("&resolutionParameters\n", 1)[1].split("/\n", 1)[0]
    assert "NL = 3" in resolution_block
    assert "!ss scanType = 3" not in resolution_block
    assert "scanVariable = nu_n" in text
    assert "scanVariableScale = log" in text


def test_write_scan_input_preserves_full_resolution_when_fast_disabled(tmp_path: Path) -> None:
    mod = _load_module()
    base_input = tmp_path / "input.namelist"
    base_input.write_text(
        "&physicsParameters\n"
        "  collisionOperator = 0\n"
        "/\n"
        "&resolutionParameters\n"
        "  Ntheta = 15\n"
        "  Nzeta = 13\n"
        "  Nxi = 16\n"
        "  NL = 9\n"
        "  Nx = 8\n"
        "  solverTolerance = 1d-6\n"
        "/\n"
    )
    dest = tmp_path / "scan_input.namelist"
    mod._write_scan_input(
        base_input=base_input,
        dest=dest,
        nu_n_min=2e-2,
        nu_n_max=2e0,
        n_points=7,
        collision_operator=1,
        fast=False,
    )
    text = dest.read_text()
    assert text.count("collisionOperator = 1") == 1
    assert "collisionOperator = 0" not in text
    assert "Ntheta = 15" in text
    assert "Nzeta = 13" in text
    assert "Nxi = 16" in text
    assert "Nx = 8" in text
    assert "solverTolerance = 1d-6" in text


def test_write_transport_scan_summary_json_sorts_and_serializes(tmp_path: Path) -> None:
    mod = _load_module()
    summary_path = tmp_path / "summary.json"
    datasets = {
        "PAS": (
            np.asarray([2.0, 1.0]),
            np.asarray(
                [
                    [[2.0, 0.0], [0.0, 3.0]],
                    [[4.0, 0.0], [0.0, 5.0]],
                ],
                dtype=float,
            ),
        ),
        "Fokker-Planck": (
            np.asarray([1.5]),
            np.asarray([[[1.0, 0.5], [0.25, 2.0]]], dtype=float),
        ),
    }
    mod.write_transport_scan_summary_json(summary_path, datasets)
    rows = json.loads(summary_path.read_text())
    assert [row["label"] for row in rows] == ["Fokker-Planck", "PAS", "PAS"]
    assert [row["nuprime"] for row in rows] == [1.5, 1.0, 2.0]
    assert rows[0]["transport_matrix"] == [[1.0, 0.5], [0.25, 2.0]]


def test_write_transport_scan_summary_json_can_include_metadata_payload(tmp_path: Path) -> None:
    mod = _load_module()
    summary_path = tmp_path / "summary_with_metadata.json"
    datasets = {
        "Fokker-Planck": (
            np.asarray([1.5]),
            np.asarray([[[1.0, 0.5], [0.25, 2.0]]], dtype=float),
        ),
    }
    metadata = {
        "case": "lhd",
        "fast": True,
        "labels_to_collision_operator": {"Fokker-Planck": 0},
        "schema_version": 1,
    }
    mod.write_transport_scan_summary_json(summary_path, datasets, metadata=metadata)
    payload = json.loads(summary_path.read_text())
    assert payload["metadata"] == metadata
    rows = payload["rows"]
    assert [row["label"] for row in rows] == ["Fokker-Planck"]
    assert [row["nuprime"] for row in rows] == [1.5]


def test_summary_metadata_records_case_resolution_and_paths(tmp_path: Path) -> None:
    mod = _load_module()
    work_dir = tmp_path / "work"
    summary_path = tmp_path / "summary.json"
    work_dir.mkdir()
    metadata = mod._summary_metadata(
        case="w7x",
        fast=False,
        n_points=7,
        nuprime_min=0.1,
        nuprime_max=10.0,
        work_dir=work_dir,
        summary_path=summary_path,
        base_input=mod.EXAMPLES / "transportMatrix_geometryScheme11" / "input.namelist",
        labels_to_collision_operator={"Fokker-Planck": 0, "PAS": 1},
    )
    assert metadata["case"] == "w7x"
    assert metadata["fast"] is False
    assert metadata["n_points"] == 7
    assert metadata["nuprime_min"] == 0.1
    assert metadata["nuprime_max"] == 10.0
    assert metadata["base_input"] == "examples/sfincs_examples/transportMatrix_geometryScheme11/input.namelist"
    assert metadata["source_script"] == "examples/publication_figures/generate_sfincs_paper_figs.py"
    assert metadata["labels_to_collision_operator"] == {"Fokker-Planck": 0, "PAS": 1}


def test_parse_collision_operators_deduplicates_and_preserves_order() -> None:
    mod = _load_module()
    assert mod._parse_collision_operators("1, 0, 1") == (1, 0)
    assert mod._parse_collision_operators("") == (0, 1)


def test_parse_collision_operators_rejects_unsupported_values() -> None:
    mod = _load_module()
    with pytest.raises(ValueError, match="Unsupported collision operator 2"):
        mod._parse_collision_operators("0,2")


def test_run_streams_child_output_to_terminal_and_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module()
    child = tmp_path / "child.py"
    child.write_text(
        "print('child-line-1')\n"
        "print('child-progress 1/2')\n"
        "print('child-line-2')\n"
    )
    mod._run([sys.executable, str(child)], cwd=tmp_path, timeout_s=5.0, label="child")
    captured = capsys.readouterr().out
    assert "[child] cwd=" in captured
    assert "child-line-1" in captured
    assert "child-progress 1/2" in captured
    assert "[child] completed elapsed=" in captured
    log_text = (tmp_path / "child.log").read_text()
    assert "child-line-1" in log_text
    assert "child-line-2" in log_text


def test_expected_scan_subdirs_matches_log_ladder_format() -> None:
    mod = _load_module()
    got = mod._expected_scan_subdirs(
        scan_variable="nu_n",
        min_value=0.1 * 0.2668018,
        max_value=10.0 * 0.2668018,
        n_points=4,
        scale="log",
    )
    assert got == ("nu_n_0.02668", "nu_n_0.1238", "nu_n_0.5748", "nu_n_2.668")


def test_main_skip_existing_skips_rerun_for_complete_selected_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    work_dir = tmp_path / "work"
    summary_dir = tmp_path / "summary"
    out_dir = tmp_path / "figures"
    for nu_n, scale in ((0.02668, 1.0), (0.1238, 2.0), (0.5748, 3.0), (2.668, 4.0)):
        _write_fake_transport_output(
            work_dir / "lhd_co1" / f"nu_n_{nu_n:.4g}",
            nu_n=nu_n,
            diagonal_scale=scale,
        )

    write_calls = 0
    run_calls = 0

    def _unexpected_write_scan_input(**kwargs):
        nonlocal write_calls
        write_calls += 1

    def _unexpected_run(*args, **kwargs):
        nonlocal run_calls
        run_calls += 1

    monkeypatch.setattr(mod, "_write_scan_input", _unexpected_write_scan_input)
    monkeypatch.setattr(mod, "_run", _unexpected_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_sfincs_paper_figs.py",
            "--case",
            "lhd",
            "--fast",
            "--scan-only",
            "--skip-existing",
            "--collision-operators",
            "1",
            "--work-dir",
            str(work_dir),
            "--summary-dir",
            str(summary_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    mod.main()

    assert write_calls == 0
    assert run_calls == 0
    payload = json.loads((summary_dir / "lhd_collisionality_fast_summary.json").read_text())
    assert payload["metadata"]["labels_to_collision_operator"] == {"PAS": 1}
    assert [row["label"] for row in payload["rows"]] == ["PAS", "PAS", "PAS", "PAS"]
    assert not (out_dir / "sfincs_jax_fig1_lhd_collisionality.png").exists()


def test_main_skip_existing_prunes_incomplete_dirs_and_reruns_partial_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    work_dir = tmp_path / "work"
    summary_dir = tmp_path / "summary"
    out_dir = tmp_path / "figures"
    case_dir = work_dir / "lhd_co1"
    _write_fake_transport_output(case_dir / "nu_n_0.1238", nu_n=0.1238, diagonal_scale=2.0)
    stale_dir = case_dir / "nu_n_0.5748"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "input.namelist").write_text("&dummy\n/\n")

    write_calls = 0
    run_calls = 0

    def _fake_write_scan_input(**kwargs):
        nonlocal write_calls
        write_calls += 1
        Path(kwargs["dest"]).write_text("&physicsParameters\n/\n")

    def _fake_run(*args, **kwargs):
        nonlocal run_calls
        run_calls += 1

    monkeypatch.setattr(mod, "_write_scan_input", _fake_write_scan_input)
    monkeypatch.setattr(mod, "_run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_sfincs_paper_figs.py",
            "--case",
            "lhd",
            "--scan-only",
            "--skip-existing",
            "--collision-operators",
            "1",
            "--fast",
            "--work-dir",
            str(work_dir),
            "--summary-dir",
            str(summary_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    mod.main()

    assert write_calls == 1
    assert run_calls == 1
    assert not stale_dir.exists()


def test_main_plot_only_allows_single_selected_operator_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    work_dir = tmp_path / "work"
    summary_dir = tmp_path / "summary"
    out_dir = tmp_path / "figures"
    for nu_n, scale in ((0.02668, 1.0), (0.1238, 2.0), (0.5748, 3.0), (2.668, 4.0)):
        _write_fake_transport_output(
            work_dir / "lhd_co1" / f"nu_n_{nu_n:.4g}",
            nu_n=nu_n,
            diagonal_scale=scale,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_sfincs_paper_figs.py",
            "--case",
            "lhd",
            "--plot-only",
            "--fast",
            "--collision-operators",
            "1",
            "--work-dir",
            str(work_dir),
            "--summary-dir",
            str(summary_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    mod.main()

    payload = json.loads((summary_dir / "lhd_collisionality_fast_summary.json").read_text())
    assert payload["metadata"]["labels_to_collision_operator"] == {"PAS": 1}
    assert [row["label"] for row in payload["rows"]] == ["PAS", "PAS", "PAS", "PAS"]
    assert (out_dir / "sfincs_jax_fig1_lhd_collisionality.png").exists()


def test_main_plot_only_rejects_incomplete_scan_before_rewriting_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    work_dir = tmp_path / "work"
    summary_dir = tmp_path / "summary"
    out_dir = tmp_path / "figures"
    _write_fake_transport_output(work_dir / "lhd_co1" / "nu_n_0.02668", nu_n=0.02668)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_sfincs_paper_figs.py",
            "--case",
            "lhd",
            "--plot-only",
            "--fast",
            "--collision-operators",
            "1",
            "--work-dir",
            str(work_dir),
            "--summary-dir",
            str(summary_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    with pytest.raises(RuntimeError, match=r"--plot-only requires a complete scan"):
        mod.main()

    assert not (summary_dir / "lhd_collisionality_fast_summary.json").exists()
    assert not (out_dir / "sfincs_jax_fig1_lhd_collisionality.png").exists()
