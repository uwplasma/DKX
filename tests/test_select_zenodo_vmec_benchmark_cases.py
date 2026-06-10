from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "select_zenodo_vmec_benchmark_cases.py"
    spec = importlib.util.spec_from_file_location("select_zenodo_vmec_benchmark_cases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(*, family: str, label: str, s: float, has_output: bool = True, rhs_mode: int = 1) -> dict:
    ntheta, nzeta, nxi, nx = (int(part) for part in label.split("x"))
    token = str(s).replace(".", "p")
    return {
        "input": f"calculations/campaign/{family}_{label}/psiN_{s}/input.namelist",
        "fortran_output": f"calculations/campaign/{family}_{label}/psiN_{s}/sfincsOutput.h5",
        "family": family,
        "case": f"{family}_{label}",
        "surface_s": s,
        "rhs_mode": rhs_mode,
        "geometry_scheme": 5,
        "collision_operator": 0,
        "solver_tolerance": 1e-5,
        "nu_n": 0.01,
        "er": 0.0,
        "resolution": {"Ntheta": ntheta, "Nzeta": nzeta, "Nxi": nxi, "Nx": nx, "label": label},
        "fortran_output_summary": {"exists": has_output, "tag": token},
    }


def _manifest() -> dict:
    entries = []
    for family in ("qa", "qh"):
        for label in ("7x7x9x3", "13x13x21x5", "25x39x60x7"):
            for surface in (0.05, 0.5, 0.95):
                entries.append(_entry(family=family, label=label, s=surface))
    entries.append(_entry(family="qa", label="51x49x55x6", s=0.5, has_output=False))
    entries.append(_entry(family="qa", label="31x39x95x7", s=0.5, rhs_mode=3))
    entries.append(_entry(family="w7x", label="5x7x8x5", s=0.45))
    return {"input_count": len(entries), "with_fortran_output_count": 19, "entries": entries}


def test_select_zenodo_vmec_benchmark_cases_is_deterministic_and_filtered() -> None:
    mod = _load_module()

    selected = mod.select_zenodo_vmec_benchmark_cases(_manifest(), families=("qa", "qh"))

    assert selected["selected_count"] == 18
    assert selected["counts_by_family"] == {"qa": 9, "qh": 9}
    assert selected["counts_by_rung"] == {"intermediate": 6, "low": 6, "production": 6}
    assert {case["surface_role"] for case in selected["cases"]} == {"central", "inner_edge", "outer_edge"}
    assert all("51x49x55x6" not in case["input"] for case in selected["cases"])
    assert all("31x39x95x7" not in case["input"] for case in selected["cases"])

    qa_production = [
        case
        for case in selected["cases"]
        if case["family"] == "qa" and case["rung"] == "production" and case["surface_role"] == "central"
    ][0]
    assert qa_production["resolution"]["label"] == "25x39x60x7"
    assert qa_production["surface_s"] == 0.5


def test_select_zenodo_vmec_benchmark_cases_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_module()
    manifest_path = tmp_path / "manifest.json"
    out = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    rc = mod.main(["--manifest", str(manifest_path), "--out", str(out), "--family", "w7x"])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selected_count"] == 1
    assert payload["cases"][0]["family"] == "w7x"
    assert payload["cases"][0]["rung"] == "single"
