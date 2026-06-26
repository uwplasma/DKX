from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sfincs_jax.solvers.diagnostics import compare_solver_profiles


def _fortran_profile() -> dict:
    return {
        "n_mpi_processes": 40,
        "solver_package": "mumps",
        "matrix_shape": [507004, 507004],
        "matrix_nnz": {"nnz": 21400898, "allocated_nnz": 79146608},
        "preconditioner_nnz": {"nnz": 13483256, "allocated_nnz": 79146608},
        "timings_s": {
            "assemble_preconditioner": 0.187,
            "assemble_jacobian": 0.288,
            "mumps_analysis_driver": 11.01,
            "mumps_factorization_driver": 2.79,
        },
        "mumps": {
            "factor_entries": 796672990,
            "factor_memory_peak_mb": 555,
            "factor_memory_total_mb": 12787,
        },
        "ksp": {
            "iteration_count": 52,
            "initial_residual": 1.22e-3,
            "final_residual": 1.15e-8,
        },
    }


def _jax_campaign() -> dict:
    return {
        "cases": [
            {
                "solve": {
                    "status": "timeout",
                    "timeout_s": 480.0,
                    "result": {"elapsed_s": 479.0},
                    "progress": {
                        "max_krylov_count": 2000,
                        "last_krylov_progress": {"kind": "matvecs", "count": 2000, "elapsed_s": 478.2},
                    },
                    "solver_trace": {
                        "exists": True,
                        "active_size": 507004,
                        "total_size": 819004,
                        "solve_method": "xblock_sparse_pc_gmres",
                        "selected_path": "rhsmode1_solution",
                        "converged": False,
                        "residual_norm": 3.0e-5,
                        "residual_target": 4.8e-10,
                        "residual_ratio": 6.25e4,
                        "peak_rss_mb": 8300.0,
                    },
                }
            }
        ]
    }


def _load_script_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "compare_fortran_jax_solver_profiles.py"
    spec = importlib.util.spec_from_file_location("compare_fortran_jax_solver_profiles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_solver_profiles_reports_krylov_and_residual_gaps() -> None:
    comparison = compare_solver_profiles(fortran_profile=_fortran_profile(), jax_profile=_jax_campaign())

    assert comparison["fortran"]["ksp_iteration_count"] == 52
    assert comparison["jax"]["status"] == "timeout"
    assert comparison["jax"]["max_krylov_count"] == 2000
    assert comparison["comparison"]["same_active_size"] is True
    assert comparison["comparison"]["jax_krylov_count_per_fortran_ksp_iteration"] == 2000 / 52
    assert comparison["comparison"]["jax_residual_norm_per_fortran_final_ksp_residual"] == 3.0e-5 / 1.15e-8


def test_compare_solver_profiles_recovers_timeout_progress_from_stdout_tail() -> None:
    jax_profile = {
        "cases": [
            {
                "solve": {
                    "status": "timeout",
                    "stdout_tail": "\n".join(
                        [
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=1990 elapsed_s=476.059",
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres matvecs=2000 elapsed_s=478.202",
                        ]
                    ),
                    "progress": {"progress_line_count": 0},
                    "solver_trace": {"exists": False},
                }
            }
        ]
    }

    comparison = compare_solver_profiles(fortran_profile=_fortran_profile(), jax_profile=jax_profile)

    assert comparison["jax"]["status"] == "timeout"
    assert comparison["jax"]["max_krylov_count"] == 2000
    assert comparison["jax"]["last_krylov_progress"]["kind"] == "matvecs"
    assert comparison["jax"]["last_krylov_progress"]["elapsed_s"] == 478.202
    assert comparison["comparison"]["same_active_size"] is None
    assert comparison["comparison"]["jax_krylov_count_per_fortran_ksp_iteration"] == 2000 / 52


def test_compare_fortran_jax_solver_profiles_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_script_module()
    fortran_path = tmp_path / "fortran.json"
    jax_path = tmp_path / "jax.json"
    out = tmp_path / "comparison.json"
    fortran_path.write_text(json.dumps(_fortran_profile()), encoding="utf-8")
    jax_path.write_text(json.dumps(_jax_campaign()), encoding="utf-8")

    rc = mod.main(
        [
            "--fortran-profile",
            str(fortran_path),
            "--jax-profile",
            str(jax_path),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_fortran_profile"] == str(fortran_path)
    assert payload["source_jax_profile"] == str(jax_path)
    assert payload["comparison"]["same_active_size"] is True
