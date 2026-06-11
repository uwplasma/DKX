from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_production_stress_campaign.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("summarize_production_stress_campaign", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
summary_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = summary_script
_SPEC.loader.exec_module(summary_script)


def _write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_campaign_summary_compacts_status_and_fortran_profile(tmp_path: Path) -> None:
    cpu_root = tmp_path / "production_stress_cpu_campaign"
    gpu_root = tmp_path / "production_stress_gpu_campaign"
    _write_report(
        cpu_root / "geom2" / "suite_report.json",
        [
            {
                "case": "geom2",
                "status": "reference timeout",
                "blocker_type": "reference timeout",
                "message": "Fortran reference timed out",
                "fortran_runtime_s": None,
                "fortran_max_rss_mb": 12288.0,
                "final_resolution": {"NTHETA": 25, "NZETA": 51, "NXI": 100, "NX": 4},
                "fortran_profile": {
                    "matrix_shape": [648977, 648977],
                    "matrix_nnz": 15165133,
                    "preconditioner_nnz": 12176533,
                    "solver_package": "mumps",
                    "timings_s": {"metis_reordering": 3.2, "unused": None},
                    "mumps": {
                        "n": 648977,
                        "nnz": 12176533,
                        "estimated_factor_entries": 1274005121,
                        "estimated_real_factor_space": 2200000000,
                        "ignored_verbose_field": "drop",
                    },
                },
            }
        ],
    )
    _write_report(
        gpu_root / "geom2" / "suite_report.json",
        [
            {
                "case": "geom2",
                "status": "parity_ok",
                "blocker_type": "none",
                "message": "matched",
                "fortran_runtime_s": 10.0,
                "jax_logged_elapsed_s": 25.0,
                "fortran_max_rss_mb": 1000.0,
                "jax_max_rss_mb": 3000.0,
                "jax_solver_kinds": ["fp_fortran_reduced_lu"],
                "n_common_keys": 190,
                "n_mismatch_common": 0,
                "strict_n_mismatch_common": 0,
            }
        ],
    )

    summary = summary_script.build_summary([cpu_root, gpu_root])

    assert summary["report_count"] == 2
    assert summary["row_count"] == 2
    assert summary["status_counts"] == {"parity_ok": 1, "reference timeout": 1}
    assert summary["backend_counts"] == {"cpu": 1, "gpu": 1}
    timeout = next(row for row in summary["rows"] if row["status"] == "reference timeout")
    assert timeout["fortran_profile"]["matrix_shape"] == [648977, 648977]
    assert timeout["fortran_profile"]["mumps"]["estimated_factor_entries"] == 1274005121
    assert "ignored_verbose_field" not in timeout["fortran_profile"]["mumps"]
    ok = next(row for row in summary["rows"] if row["status"] == "parity_ok")
    assert ok["runtime_ratio_jax_to_fortran"] == 2.5
    assert ok["memory_ratio_jax_to_fortran"] == 3.0


def test_campaign_summary_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "production_stress_gpu_campaign"
    _write_report(
        root / "case" / "suite_report.json",
        [
            {
                "case": "case",
                "status": "jax_error",
                "blocker_type": "solver failure",
                "message": "failed",
                "fortran_runtime_s": 1.0,
                "jax_runtime_s": 2.0,
            }
        ],
    )
    out = tmp_path / "summary.json"
    md = tmp_path / "summary.md"

    assert summary_script.main(["--root", str(root), "--out", str(out), "--markdown-out", str(md), "--json"]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status_counts"] == {"jax_error": 1}
    text = md.read_text(encoding="utf-8")
    assert "SFINCS-JAX production stress campaign summary" in text
    assert "| gpu | case | jax_error | solver failure |" in text


def test_campaign_summary_captures_partial_fortran_log_without_report(tmp_path: Path) -> None:
    root = tmp_path / "production_stress_gpu_campaign"
    log_path = root / "additional_examples" / "additional_examples" / "fortran_run" / "sfincs.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "The matrix is 648977 x 648977 elements.\n"
        "Running populateMatrix with whichMatrix = 0\n"
        "Time to pre-assemble Jacobian preconditioner matrix: 1.234 seconds.\n"
        "# of nonzeros in Jacobian preconditioner matrix: 12176533 , allocated: 48044552\n"
        "Running populateMatrix with whichMatrix = 1\n"
        "Time to pre-assemble Jacobian matrix: 2.345 seconds.\n"
        "# of nonzeros in Jacobian matrix: 15165133 , allocated: 48044552\n"
        "Entering DMUMPS 5.8.2 from C interface with JOB, N, NNZ =   1      648977       12176533\n"
        "      executing #MPI =      1 and #OMP =     36\n"
        "Ordering based on METIS\n"
        "ELAPSED TIME SPENT IN METIS reordering  =      8.3250\n"
        "ELAPSED TIME IN symbolic factorization  =      0.1550\n"
        "-- (20) Number of entries in factors (estim.)  =      1274005121\n"
        "--  (3) Real space for factors    (estimated)  =      1274005121\n"
        "--  (4) Integer space for factors (estimated)  =        14044192\n"
        "--  (5) Maximum frontal size      (estimated)  =            5330\n"
        "RINFOG(1) Operations during elimination (estim)= 3.078D+12\n",
        encoding="utf-8",
    )

    summary = summary_script.build_summary([root])

    assert summary["report_count"] == 0
    assert summary["partial_fortran_log_count"] == 1
    assert summary["status_counts"] == {"running_or_unreported": 1}
    row = summary["rows"][0]
    assert row["case"] == "additional_examples"
    assert row["backend"] == "gpu"
    assert row["fortran_profile"]["matrix_shape"] == [648977, 648977]
    assert row["fortran_profile"]["matrix_nnz"] == 15165133
    assert row["fortran_profile"]["preconditioner_nnz"] == 12176533
    assert row["fortran_profile"]["timings_s"]["metis_reordering"] == 8.325
    assert row["fortran_profile"]["mumps"]["n_omp_threads"] == 36
    assert row["fortran_profile"]["mumps"]["estimated_elimination_operations"] == 3.078e12


def test_campaign_summary_prefers_completed_report_over_partial_log(tmp_path: Path) -> None:
    root = tmp_path / "production_stress_cpu_campaign"
    log_path = root / "case" / "case" / "fortran_run" / "sfincs.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("The matrix is 10 x 10 elements.\n", encoding="utf-8")
    _write_report(root / "case" / "suite_report.json", [{"case": "case", "status": "parity_ok"}])

    summary = summary_script.build_summary([root])

    assert summary["report_count"] == 1
    assert summary["partial_fortran_log_count"] == 0
    assert summary["status_counts"] == {"parity_ok": 1}
