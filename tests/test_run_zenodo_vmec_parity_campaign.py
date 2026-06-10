from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "run_zenodo_vmec_parity_campaign.py"
    spec = importlib.util.spec_from_file_location("run_zenodo_vmec_parity_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_h5(path: Path, *, current: float, niter: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5["FSABjHat"] = current
        h5["FSABjHatOverRootFSAB2"] = current / 2
        h5["NIterations"] = niter
        h5["particleFlux"] = np.array([1.0, 2.0])
        h5["text"] = b"metadata"


def _selection(root: Path, *, fortran_rel: str = "cases/qa/s0p5/sfincsOutput.h5") -> Path:
    payload = {
        "source_manifest_input_count": 2,
        "source_manifest_with_fortran_output_count": 1,
        "counts_by_family": {"qa": 1},
        "counts_by_rung": {"low": 1},
        "cases": [
            {
                "family": "qa",
                "rung": "low",
                "surface_role": "central",
                "surface_s": 0.5,
                "input": "cases/qa/s0p5/input.namelist",
                "fortran_output": fortran_rel,
                "case": "qa_case",
                "resolution": {"label": "7x7x9x3", "Ntheta": 7, "Nzeta": 7, "Nxi": 9, "Nx": 3},
            }
        ],
    }
    path = root / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_input(path: Path, *, equilibrium_file: str = "/ptmp/mlan/wout_demo.nc") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
&general
/

&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{equilibrium_file}"
/
""",
        encoding="utf-8",
    )


def test_run_campaign_reference_only_audits_fortran_outputs(tmp_path: Path) -> None:
    mod = _load_module()
    zenodo_root = tmp_path / "zenodo"
    selection = _selection(tmp_path)
    _write_h5(zenodo_root / "cases/qa/s0p5/sfincsOutput.h5", current=-1.0)

    report = mod.run_campaign(selection_path=selection, zenodo_root=zenodo_root, mode="reference-only")

    assert report["selected_count"] == 1
    assert report["status_counts"] == {"reference_ok": 1}
    row = report["cases"][0]
    assert row["reference"]["numeric_dataset_count"] == 4
    assert row["reference"]["selected_datasets"]["FSABjHat"]["value"] == -1.0
    assert row["candidate"] is None
    assert row["timing"]["phase_count"] == 1


def test_run_campaign_cached_compare_uses_candidate_map_and_reports_parity(tmp_path: Path) -> None:
    mod = _load_module()
    zenodo_root = tmp_path / "zenodo"
    selection = _selection(tmp_path)
    reference = zenodo_root / "cases/qa/s0p5/sfincsOutput.h5"
    candidate = tmp_path / "jax" / "sfincsOutput.h5"
    _write_h5(reference, current=-1.0)
    _write_h5(candidate, current=-1.0)
    candidate_map = tmp_path / "candidate_map.json"
    candidate_map.write_text(json.dumps({"cases/qa/s0p5/input.namelist": str(candidate)}), encoding="utf-8")

    report = mod.run_campaign(
        selection_path=selection,
        zenodo_root=zenodo_root,
        mode="cached-compare",
        candidate_map_path=candidate_map,
        keys=("FSABjHat", "FSABjHatOverRootFSAB2", "NIterations", "heatFlux"),
    )

    assert report["status_counts"] == {"parity_fail": 1}
    row = report["cases"][0]
    assert row["candidate"]["exists"] is True
    assert row["parity"]["overall_status"] == "fail"
    by_key = {item["key"]: item for item in row["parity"]["datasets"]}
    assert by_key["FSABjHat"]["status"] == "ok"
    assert by_key["heatFlux"]["status"] == "missing_in_reference"
    assert row["timing"]["phase_count"] == 3


def test_run_campaign_cached_compare_supports_template_paths(tmp_path: Path) -> None:
    mod = _load_module()
    zenodo_root = tmp_path / "zenodo"
    selection = _selection(tmp_path)
    reference = zenodo_root / "cases/qa/s0p5/sfincsOutput.h5"
    candidate = tmp_path / "candidates" / "qa" / "low" / "0p5" / "7x7x9x3" / "sfincsOutput.h5"
    _write_h5(reference, current=-1.0)
    _write_h5(candidate, current=-1.1)

    report = mod.run_campaign(
        selection_path=selection,
        zenodo_root=zenodo_root,
        mode="cached-compare",
        candidate_root=tmp_path / "candidates",
        candidate_template="{family}/{rung}/{surface_token}/{resolution_token}/sfincsOutput.h5",
        keys=("FSABjHat",),
    )

    assert report["status_counts"] == {"parity_fail": 1}
    assert report["cases"][0]["candidate"]["exists"] is True
    assert report["cases"][0]["parity"]["datasets"][0]["max_abs"] == 0.10000000000000009


def test_run_campaign_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_module()
    zenodo_root = tmp_path / "zenodo"
    selection = _selection(tmp_path)
    _write_h5(zenodo_root / "cases/qa/s0p5/sfincsOutput.h5", current=-1.0)
    out = tmp_path / "report.json"

    rc = mod.main(["--selection", str(selection), "--zenodo-root", str(zenodo_root), "--out", str(out)])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status_counts"] == {"reference_ok": 1}


def test_run_campaign_solve_dry_run_constructs_command_and_resolves_equilibrium(tmp_path: Path) -> None:
    mod = _load_module()
    zenodo_root = tmp_path / "zenodo"
    selection = _selection(tmp_path)
    _write_h5(zenodo_root / "cases/qa/s0p5/sfincsOutput.h5", current=-1.0)
    _write_input(zenodo_root / "cases/qa/s0p5/input.namelist")
    wout = zenodo_root / "configurations" / "wout_demo.nc"
    wout.parent.mkdir(parents=True, exist_ok=True)
    wout.write_bytes(b"not a real netcdf; dry-run only")

    report = mod.run_campaign(
        selection_path=selection,
        zenodo_root=zenodo_root,
        mode="solve",
        run_root=tmp_path / "runs",
        dry_run=True,
        max_cases=1,
    )

    assert report["status_counts"] == {"solve_dry_run": 1}
    row = report["cases"][0]
    assert row["candidate"]["exists"] is False
    assert row["solve"]["status"] == "dry_run"
    assert row["solve"]["equilibrium_path"] == str(wout)
    assert row["solve"]["cli_args"][0] == "-v"
    assert "--wout-path" in row["solve"]["cli_args"]
    assert str(wout) in row["solve"]["cli_args"]
    assert row["solve"]["trace_path"].endswith("solver_trace.json")
    assert report["solve_settings"]["dry_run"] is True
    assert report["solve_settings"]["quiet_solve"] is False
    assert report["solve_trace_summary"]["solver_traces_written"] == 0


def test_extract_solve_progress_summarizes_profile_and_solver_lines() -> None:
    mod = _load_module()

    progress = mod._extract_solve_progress(
        """
noise
profiling: read_namelist dt_s=0.1 total_s=0.1 rss_mb=10 drss_mb=0 peak_rss_mb=10 dpeak_rss_mb=0 device_mb=na
 VMEC: starting geometry build from wout.nc
 write_sfincs_jax_output_h5: 3D full-FP RHSMode=1 -> using sparse-PC GMRES host solve
 solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres iters=20 ksp_residual=3.0e-4 elapsed_s=12.5
""",
        "Solver note: large system detected; preconditioner setup and Krylov solve can take several minutes.",
    )

    assert progress["progress_line_count"] == 5
    assert progress["profile_labels"] == ["read_namelist"]
    assert progress["last_progress_lines"][-1].startswith("Solver note:")
    assert progress["krylov_progress_count"] == 1
    assert progress["last_krylov_progress"]["kind"] == "iters"
    assert progress["last_krylov_progress"]["count"] == 20
    assert progress["last_krylov_progress"]["residual"] == 3.0e-4
    assert progress["last_krylov_progress"]["elapsed_s"] == 12.5
    assert progress["min_krylov_residual"] == 3.0e-4


def test_extract_solve_progress_surfaces_matvec_timeout_progress() -> None:
    mod = _load_module()

    progress = mod._extract_solve_progress(
        """
solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=1980 elapsed_s=473.939
solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=1990 elapsed_s=476.059
solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=2000 elapsed_s=478.202
""",
        "",
    )

    assert progress["progress_line_count"] == 3
    assert progress["krylov_progress_count"] == 3
    assert progress["max_krylov_count"] == 2000
    assert progress["last_krylov_progress"]["kind"] == "matvecs"
    assert progress["last_krylov_progress"]["count"] == 2000
    assert progress["last_krylov_progress"]["elapsed_s"] == 478.202
    assert progress["last_krylov_residual"] is None


def test_solver_trace_summary_surfaces_capped_scipy_rescue(tmp_path: Path) -> None:
    mod = _load_module()
    trace = tmp_path / "solver_trace.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "cpu",
                "rhs_mode": 1,
                "geometry_scheme": 5,
                "selected_path": "rhsmode1_solution",
                "solve_method": "auto",
                "total_size": 819004,
                "active_size": 507004,
                "residual_norm": 1.239e6,
                "residual_target": 4.773e-10,
                "converged": False,
                "elapsed_s": 1.35,
                "peak_rss_mb": 1063.0,
                "metadata": {
                    "failure_reason": "nonconverged_rhsmode1_output",
                    "output_refused": True,
                    "profile_entries": [
                        {"label": "read_namelist", "dt_s": 0.01},
                        {"label": "rhs1_solve_done", "dt_s": 0.77},
                    ],
                    "solver_metadata": {
                        "sparse_xblock_rescue_active": True,
                        "sparse_xblock_rescue_attempted": True,
                        "sparse_xblock_rescue_reason": "exception",
                        "sparse_xblock_rescue_error": "MemoryError: budget",
                        "scipy_rescue_active_size": 507004,
                        "scipy_rescue_attempted": False,
                        "scipy_rescue_skip_reason": "active_size_cap",
                        "scipy_rescue_skipped": True,
                        "scipy_rescue_initial_residual": 1.239e6,
                        "scipy_rescue_target": 4.773e-13,
                        "fp_xblock_global_correction_allowed": True,
                        "fp_xblock_global_correction_attempted": True,
                        "fp_xblock_global_correction_accepted": False,
                        "fp_xblock_global_correction_reason": "no_improvement",
                        "fp_xblock_global_correction_residual_before": 4.96e-5,
                        "fp_xblock_global_correction_residual_after": 4.96e-5,
                        "fp_xblock_highx_residual_correction_attempted": True,
                        "fp_xblock_highx_residual_correction_accepted": True,
                        "fp_xblock_highx_residual_correction_reason": "accepted",
                        "fp_xblock_highx_residual_correction_direction_names": ["highx_all"],
                        "large_history": [1, 2, 3],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = mod._solver_trace_summary(trace)

    assert summary["exists"] is True
    assert summary["readable"] is True
    assert summary["active_size"] == 507004
    assert summary["failure_reason"] == "nonconverged_rhsmode1_output"
    assert summary["output_refused"] is True
    assert summary["max_profile_label"] == "rhs1_solve_done"
    assert summary["solver_metadata"]["scipy_rescue_skipped"] is True
    assert summary["solver_metadata"]["scipy_rescue_skip_reason"] == "active_size_cap"
    assert summary["solver_metadata"]["sparse_xblock_rescue_attempted"] is True
    assert summary["solver_metadata"]["sparse_xblock_rescue_reason"] == "exception"
    assert summary["solver_metadata"]["sparse_xblock_rescue_error"] == "MemoryError: budget"
    assert summary["solver_metadata"]["fp_xblock_global_correction_attempted"] is True
    assert summary["solver_metadata"]["fp_xblock_global_correction_reason"] == "no_improvement"
    assert summary["solver_metadata"]["fp_xblock_highx_residual_correction_attempted"] is True
    assert summary["solver_metadata"]["fp_xblock_highx_residual_correction_reason"] == "accepted"
    assert summary["solver_metadata"]["fp_xblock_highx_residual_correction_direction_names_len"] == 1
    assert "large_history_len" not in summary["solver_metadata"]


def test_solver_trace_summary_surfaces_xblock_assembled_device_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    trace = tmp_path / "solver_trace.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "gpu",
                "rhs_mode": 1,
                "selected_path": "xblock_sparse_pc_gmres",
                "active_size": 4096,
                "residual_norm": 2.0e-12,
                "residual_target": 1.0e-10,
                "converged": True,
                "candidate_decisions": [{"name": "assembled_device_csr", "accepted": True}],
                "metadata": {
                    "solver_metadata": {
                        "xblock_assembled_operator_enabled": True,
                        "xblock_assembled_operator_built": True,
                        "xblock_assembled_operator_device_resident": True,
                        "xblock_assembled_operator_matrix_nnz": 12345,
                        "xblock_assembled_operator_validation_rel_errors": [1.0e-13, 2.0e-13],
                        "xblock_active_dof": True,
                        "xblock_linear_size": 4096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = mod._solver_trace_summary(trace)

    assert summary["converged"] is True
    assert summary["residual_ratio"] == 0.02
    assert summary["accepted_candidates"] == ["assembled_device_csr"]
    metadata = summary["solver_metadata"]
    assert metadata["xblock_assembled_operator_enabled"] is True
    assert metadata["xblock_assembled_operator_built"] is True
    assert metadata["xblock_assembled_operator_device_resident"] is True
    assert metadata["xblock_assembled_operator_matrix_nnz"] == 12345
    assert metadata["xblock_assembled_operator_validation_rel_errors_len"] == 2
    assert metadata["xblock_active_dof"] is True
    assert metadata["xblock_linear_size"] == 4096


def test_solve_trace_campaign_summary_aggregates_solver_path_decisions() -> None:
    mod = _load_module()

    summary = mod._solve_trace_campaign_summary(
        [
            {
                "solve": {
                    "solver_trace": {
                        "exists": True,
                        "readable": True,
                        "backend": "cpu",
                        "selected_path": "rhsmode1_solution",
                        "converged": False,
                        "output_refused": True,
                        "residual_ratio": 100.0,
                        "solver_metadata": {
                            "scipy_rescue_skip_reason": "active_size_cap",
                            "sparse_xblock_rescue_reason": "sparse_disabled",
                            "fp_xblock_global_correction_reason": "disabled",
                            "fp_xblock_highx_residual_correction_reason": "disabled",
                            "xblock_assembled_operator_built": False,
                        },
                    }
                }
            },
            {
                "solve": {
                    "solver_trace": {
                        "exists": True,
                        "readable": True,
                        "backend": "gpu",
                        "selected_path": "xblock_sparse_pc_gmres",
                        "converged": True,
                        "residual_ratio": 0.01,
                        "solver_metadata": {
                            "fp_xblock_global_correction_reason": "accepted",
                            "fp_xblock_global_correction_accepted": True,
                            "fp_xblock_highx_residual_correction_reason": "accepted",
                            "fp_xblock_highx_residual_correction_accepted": True,
                            "xblock_assembled_operator_built": True,
                            "xblock_assembled_operator_device_resident": True,
                        },
                    }
                }
            },
        ]
    )

    assert summary["solver_traces_written"] == 2
    assert summary["solver_traces_readable"] == 2
    assert summary["converged"] == 1
    assert summary["output_refused"] == 1
    assert summary["selected_paths"] == {"rhsmode1_solution": 1, "xblock_sparse_pc_gmres": 1}
    assert summary["backends"] == {"cpu": 1, "gpu": 1}
    assert summary["scipy_rescue_skip_reasons"] == {"active_size_cap": 1}
    assert summary["sparse_xblock_rescue_reasons"] == {"sparse_disabled": 1}
    assert summary["fp_xblock_global_correction_reasons"] == {"accepted": 1, "disabled": 1}
    assert summary["fp_xblock_global_correction_accepted"] == 1
    assert summary["fp_xblock_highx_residual_correction_reasons"] == {"accepted": 1, "disabled": 1}
    assert summary["fp_xblock_highx_residual_correction_accepted"] == 1
    assert summary["xblock_assembled_operator_built"] == 1
    assert summary["xblock_assembled_operator_device_resident"] == 1
    assert summary["max_residual_ratio"] == 100.0
