from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sfincs_jax.validation.fortran_profile import parse_fortran_v3_profile_text


SYNTHETIC_LOG = """
 Parallel job (  40 processes) detected.
 ---- Numerical parameters: ----
 Ntheta             =           25
 Nzeta              =           39
 Nxi                =           60
 NL                 =            4
 Nx                 =            7
 solverTolerance    =   1.000000000000000E-005
 The matrix is       507004 x      507004  elements.
 mumps detected
 Solver package which will be used:
 mumps
 --------- Residual function norm:  2.1248565E-05 -----------------------------
 Time to pre-assemble Jacobian preconditioner matrix:   0.141603469848633 seconds.
 Time to assemble Jacobian preconditioner matrix:   0.187377460300922 seconds.
 # of nonzeros in Jacobian preconditioner matrix:    13483256 , allocated:
    79146608 , mallocs:           0  (should be 0)
 Time to pre-assemble Jacobian matrix:   0.146416537463665 seconds.
 Time to assemble Jacobian matrix:   0.287799000740051 seconds.
 # of nonzeros in Jacobian matrix:    21400898 , allocated:    79146608
 Elapsed time in analysis driver=      11.0101
 INFOG(16) (estimated size in MB): 706
 INFOG(17) (estimated size sum in MB): 17313
 INFOG(21) (size in MB used peak): 555
 INFOG(22) (size in MB used total): 12787
 INFOG(29) (after factorization: effective number of entries): 796672990
 Elapsed time in factorization driver=       2.7922
 Elapsed time in solve driver=       0.0732
    0 KSP Residual norm 1.222910059169e-03
 Elapsed time in solve driver=       0.0688
   51 KSP Residual norm 1.148996017067e-08
 # of nonzeros in residual f1 matrix:    21400898 , allocated:    79146608
 Time to pre-assemble residual f1 matrix:   0.148629181087017 seconds.
 Time to assemble residual f1 matrix:   0.270098157227039 seconds.
 Linear iteration (KSP) converged.  KSPConvergedReason =            2
"""


def _load_script_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "summarize_fortran_v3_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_fortran_v3_profile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_fortran_v3_profile_text_extracts_petsc_mumps_metrics() -> None:
    profile = parse_fortran_v3_profile_text(SYNTHETIC_LOG)

    assert profile["n_mpi_processes"] == 40
    assert profile["solver_package"] == "mumps"
    assert profile["resolution"] == {"Ntheta": 25, "Nzeta": 39, "Nxi": 60, "NL": 4, "Nx": 7}
    assert profile["solver_tolerance"] == 1.0e-5
    assert profile["matrix_shape"] == [507004, 507004]
    assert profile["matrix_nnz"] == {"nnz": 21400898, "allocated_nnz": 79146608}
    assert profile["preconditioner_nnz"] == {"nnz": 13483256, "allocated_nnz": 79146608}
    assert profile["residual_matrix_nnz"] == {"nnz": 21400898, "allocated_nnz": 79146608}
    assert profile["timings_s"]["assemble_preconditioner"] == 0.187377460300922
    assert profile["timings_s"]["assemble_jacobian"] == 0.287799000740051
    assert profile["timings_s"]["mumps_analysis_driver"] == 11.0101
    assert profile["timings_s"]["mumps_factorization_driver"] == 2.7922
    assert profile["timings_s"]["last_solve_driver"] == 0.0688
    assert profile["mumps"]["detected"] is True
    assert profile["mumps"]["factor_entries"] == 796672990
    assert profile["mumps"]["factor_memory_peak_mb"] == 555
    assert profile["mumps"]["factor_memory_total_mb"] == 12787
    assert profile["ksp"]["iteration_count"] == 2
    assert profile["ksp"]["initial_residual"] == 1.222910059169e-3
    assert profile["ksp"]["final_residual"] == 1.148996017067e-8
    assert profile["ksp"]["converged_reason"] == 2
    assert profile["residual_function_norm_initial"] == 2.1248565e-5


def test_summarize_fortran_v3_profile_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_script_module()
    log = tmp_path / "slurm.out"
    out = tmp_path / "summary.json"
    log.write_text(SYNTHETIC_LOG, encoding="utf-8")

    rc = mod.main([str(log), "--out", str(out)])

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_log"] == str(log)
    assert payload["matrix_shape"] == [507004, 507004]
    assert payload["ksp"]["final_residual"] == 1.148996017067e-8
