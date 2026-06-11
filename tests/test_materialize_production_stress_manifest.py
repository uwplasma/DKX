from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "materialize_production_stress_manifest.py"
)
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("materialize_production_stress_manifest", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
stress_manifest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stress_manifest
_SPEC.loader.exec_module(stress_manifest)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_fake_vmec_jax_root(tmp_path: Path) -> Path:
    root = tmp_path / "vmec_jax"
    for rel in (
        "docs/_build/html/_static/qi_readme_cases/nfp1/wout_final.nc",
        "docs/_build/html/_static/qi_readme_cases/nfp2_target_helicity/wout_final.nc",
        "docs/_build/html/_static/qi_readme_cases/nfp3_seed3127/wout_final.nc",
        "docs/_build/html/_static/qi_readme_cases/nfp4_minimal/wout_final.nc",
        "docs/_static/readme_best_cases/qi/wout_final.nc",
        "examples/data/wout_QI_stel_seed_3127.nc",
        "examples/data/wout_nfp3_QI_fixed_resolution_final.nc",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{rel}\n".encode())
    return root


def test_build_manifest_collects_current_production_gaps_and_research_lanes(tmp_path: Path) -> None:
    manifest = stress_manifest.build_manifest(vmec_jax_root=_make_fake_vmec_jax_root(tmp_path))

    assert manifest["kind"] == "sfincs_jax_production_stress_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["summary_counts"]["production_manifest_cases"] == 39
    assert manifest["summary_counts"]["benchmark_floor_gap_rows"] == {"cpu": 16, "gpu": 16}
    assert manifest["summary_counts"]["short_fortran_runtime_case_count"] == 15
    assert manifest["summary_counts"]["qi_nfps_covered"] == ["1", "2", "3", "4"]

    cpu_floor_cases = {
        row["case"]
        for row in manifest["benchmark_floor_gaps"]["cpu"]
        if row["production_case_found"]
    }
    assert "HSX_FPCollisions_DKESTrajectories" in cpu_floor_cases
    assert "geometryScheme4_2species_noEr_withPhi1InDKE" in cpu_floor_cases
    assert all(
        row["production_input"].endswith("/input.namelist")
        and row["production_resolution"] is not None
        for row in manifest["benchmark_floor_gaps"]["cpu"]
        if row["production_case_found"]
    )

    short_cases = {row["case"] for row in manifest["short_fortran_runtime_cases"]}
    assert {"transportMatrix_geometryScheme2", "transportMatrix_geometryScheme11"} <= short_cases

    bootstrap_by_case = {row["case"]: row for row in manifest["qa_qh_bootstrap"]}
    assert bootstrap_by_case["QA"]["metrics"]["completed_points"] == 11
    assert bootstrap_by_case["QH"]["metrics"]["completed_points"] == 11
    assert (
        bootstrap_by_case["QA"]["metrics"][
            "max_jax_relative_difference_vs_fortran_same_resolution"
        ]
        < 2e-3
    )
    assert (
        bootstrap_by_case["QH"]["metrics"][
            "max_jax_relative_difference_vs_fortran_same_resolution"
        ]
        < 5e-3
    )

    assert manifest["qi_evidence"]["release_gate"] == "bounded_proxy"
    assert len(manifest["open_lanes"]) >= 6
    assert len(manifest["execution_order"]) >= 6


def test_manifest_writer_outputs_json_with_qi_checksums(tmp_path: Path) -> None:
    out_root = tmp_path / "stress_manifest"
    vmec_jax_root = _make_fake_vmec_jax_root(tmp_path)

    assert (
        stress_manifest.main(
            [
                "--out-root",
                str(out_root),
                "--vmec-jax-root",
                str(vmec_jax_root),
                "--json",
            ]
        )
        == 0
    )

    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest["qi_candidates"]
    assert len(candidates) == 7
    assert all(candidate["exists"] for candidate in candidates)
    assert all(len(candidate["sha256"]) == 64 for candidate in candidates)
    assert all(candidate["size_bytes"] > 0 for candidate in candidates)
