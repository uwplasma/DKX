from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_sfincs_paper_figs.py"
    spec = importlib.util.spec_from_file_location("generate_sfincs_paper_figs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
