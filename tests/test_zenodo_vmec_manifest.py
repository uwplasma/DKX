from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "build_zenodo_vmec_manifest.py"
    spec = importlib.util.spec_from_file_location("build_zenodo_vmec_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_input(path: Path, *, case: str, psi_n: float, ntheta: int, nzeta: int, nxi: int, nx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
&general
  RHSMode = 1
/

&geometryParameters
  psiN_wish = {psi_n}
  geometryScheme = 5
  inputRadialCoordinate = 1
  equilibriumFile = '/tmp/wout_{case}.nc'
/

&physicsParameters
  Er = 0.001
  nu_n = 8.31565d-3
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = {ntheta}
  Nzeta = {nzeta}
  Nxi = {nxi}
  Nx = {nx}
  solverTolerance = 1d-5
/
""",
        encoding="utf-8",
    )


def _write_output(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5["FSABjHat"] = -1.5
        h5["FSABjHatOverRootFSAB2"] = -1.25
        h5["NIterations"] = 4
        h5["RHSMode"] = 1


def test_build_zenodo_vmec_manifest_parses_cases_and_outputs(tmp_path: Path) -> None:
    mod = _load_module()
    root = tmp_path / "zenodo"
    qa_input = (
        root
        / "calculations"
        / "campaign"
        / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
        / "psiN_0.55"
        / "input.namelist"
    )
    qh_input = root / "calculations" / "campaign" / "QH_scan" / "psiN_0.70" / "input.namelist"
    _write_input(qa_input, case="qa", psi_n=0.55, ntheta=25, nzeta=39, nxi=60, nx=7)
    _write_input(qh_input, case="qh", psi_n=0.70, ntheta=13, nzeta=17, nxi=21, nx=5)
    _write_output(qa_input.parent / "sfincsOutput.h5")

    manifest = mod.build_zenodo_vmec_manifest(root)

    assert manifest["input_count"] == 2
    assert manifest["with_fortran_output_count"] == 1
    assert manifest["family_counts"] == {"qa": 1, "qh": 1}
    assert manifest["rhs_mode_counts"] == {"1": 2}
    assert manifest["resolution_counts"] == {"13x17x21x5": 1, "25x39x60x7": 1}

    qa = next(entry for entry in manifest["entries"] if entry["family"] == "qa")
    assert qa["surface_s"] == 0.55
    assert qa["resolution"]["label"] == "25x39x60x7"
    assert qa["fortran_output_summary"]["exists"] is True
    assert qa["fortran_output_summary"]["FSABjHatOverRootFSAB2"] == -1.25

    qh = next(entry for entry in manifest["entries"] if entry["family"] == "qh")
    assert qh["fortran_output_summary"] == {"exists": False}


def test_build_zenodo_vmec_manifest_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_module()
    root = tmp_path / "zenodo"
    input_path = root / "calculations" / "campaign" / "QA_case" / "psiN_0.25" / "input.namelist"
    _write_input(input_path, case="qa", psi_n=0.25, ntheta=7, nzeta=9, nxi=11, nx=3)

    out = tmp_path / "manifest.json"
    rc = mod.main(["--zenodo-root", str(root), "--out", str(out)])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["input_count"] == 1
    assert payload["entries"][0]["resolution"]["label"] == "7x9x11x3"
