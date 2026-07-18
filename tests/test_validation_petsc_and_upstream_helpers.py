from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

import dkx.validation.fortran as fortran_validation
from dkx.validation.fortran import (
    default_fortran_exe,
    parse_fortran_v3_profile_file,
    parse_fortran_v3_profile_text,
    read_petsc_mat_aij,
    read_petsc_vec,
    run_sfincs_fortran,
)
from dkx.workflows.scans import find_upstream_utils_dir, run_upstream_util


def test_petsc_vec_reader_roundtrips_big_endian_fixture(tmp_path: Path) -> None:
    path = tmp_path / "vec.dat"
    np.asarray([1211214, 3], dtype=">i4").tofile(path)
    with path.open("ab") as stream:
        np.asarray([1.5, -2.0, 4.25], dtype=">f8").tofile(stream)

    vec = read_petsc_vec(path)

    assert vec.size == 3
    np.testing.assert_allclose(vec.values, [1.5, -2.0, 4.25])


def test_petsc_vec_reader_rejects_short_and_wrong_class_files(tmp_path: Path) -> None:
    short = tmp_path / "short.vec"
    short.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="too small"):
        read_petsc_vec(short)

    wrong = tmp_path / "wrong.vec"
    np.asarray([999, 0], dtype=">i4").tofile(wrong)
    with pytest.raises(ValueError, match="Unexpected PETSc Vec classid"):
        read_petsc_vec(wrong)

    negative = tmp_path / "negative.vec"
    np.asarray([1211214, -1], dtype=">i4").tofile(negative)
    with pytest.raises(ValueError, match="negative size"):
        read_petsc_vec(negative)

    truncated = tmp_path / "truncated.vec"
    np.asarray([1211214, 2], dtype=">i4").tofile(truncated)
    with truncated.open("ab") as stream:
        np.asarray([1.0], dtype=">f8").tofile(stream)
    with pytest.raises(ValueError, match="Invalid PETSc Vec"):
        read_petsc_vec(truncated)


def test_petsc_aij_reader_roundtrips_sorted_csr_fixture(tmp_path: Path) -> None:
    path = tmp_path / "mat.dat"
    np.asarray([1211216, 2, 3, 3], dtype=">i4").tofile(path)
    with path.open("ab") as stream:
        np.asarray([2, 1], dtype=">i4").tofile(stream)
        np.asarray([0, 2, 1], dtype=">i4").tofile(stream)
        np.asarray([1.0, 3.0, -2.0], dtype=">f8").tofile(stream)

    mat = read_petsc_mat_aij(path)

    assert mat.shape == (2, 3)
    np.testing.assert_array_equal(mat.row_ptr, [0, 2, 3])
    np.testing.assert_array_equal(mat.col_ind, [0, 2, 1])
    np.testing.assert_allclose(mat.data, [1.0, 3.0, -2.0])
    assert mat.get(0, 0) == pytest.approx(1.0)
    assert mat.get(0, 1) == pytest.approx(0.0)
    assert mat.get(1, 1) == pytest.approx(-2.0)


def test_petsc_aij_reader_rejects_bad_headers_and_row_counts(tmp_path: Path) -> None:
    short = tmp_path / "short.mat"
    short.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="too small"):
        read_petsc_mat_aij(short)

    wrong = tmp_path / "wrong.mat"
    np.asarray([999, 0, 0, 0], dtype=">i4").tofile(wrong)
    with pytest.raises(ValueError, match="Unexpected PETSc Mat classid"):
        read_petsc_mat_aij(wrong)

    negative = tmp_path / "negative.mat"
    np.asarray([1211216, 1, 1, -1], dtype=">i4").tofile(negative)
    with pytest.raises(ValueError, match="negative dimension"):
        read_petsc_mat_aij(negative)

    bad_rows = tmp_path / "bad_rows.mat"
    np.asarray([1211216, 2, 2, 3], dtype=">i4").tofile(bad_rows)
    with bad_rows.open("ab") as stream:
        np.asarray([1, 1], dtype=">i4").tofile(stream)
        np.asarray([0, 1, 1], dtype=">i4").tofile(stream)
        np.asarray([1.0, 2.0, 3.0], dtype=">f8").tofile(stream)
    with pytest.raises(ValueError, match="row pointers do not sum"):
        read_petsc_mat_aij(bad_rows)

    truncated_rows = tmp_path / "truncated_rows.mat"
    np.asarray([1211216, 2, 2, 0], dtype=">i4").tofile(truncated_rows)
    with truncated_rows.open("ab") as stream:
        np.asarray([0], dtype=">i4").tofile(stream)
    with pytest.raises(ValueError, match="row counts"):
        read_petsc_mat_aij(truncated_rows)

    truncated_cols = tmp_path / "truncated_cols.mat"
    np.asarray([1211216, 1, 2, 2], dtype=">i4").tofile(truncated_cols)
    with truncated_cols.open("ab") as stream:
        np.asarray([2], dtype=">i4").tofile(stream)
        np.asarray([0], dtype=">i4").tofile(stream)
    with pytest.raises(ValueError, match="column indices"):
        read_petsc_mat_aij(truncated_cols)

    truncated_values = tmp_path / "truncated_values.mat"
    np.asarray([1211216, 1, 2, 2], dtype=">i4").tofile(truncated_values)
    with truncated_values.open("ab") as stream:
        np.asarray([2], dtype=">i4").tofile(stream)
        np.asarray([0, 1], dtype=">i4").tofile(stream)
        np.asarray([1.0], dtype=">f8").tofile(stream)
    with pytest.raises(ValueError, match="Mat values"):
        read_petsc_mat_aij(truncated_values)


def test_fortran_profile_parser_handles_d_exponents_and_optional_sections() -> None:
    profile = parse_fortran_v3_profile_text(
        "Parallel job ( 8 processes) detected\n"
        "Ntheta = 25\n"
        "Nzeta = 51\n"
        "Nxi = 100\n"
        "NL = 4\n"
        "Nx = 7\n"
        "solverTolerance = 1.0D-10\n"
        "Solver package which will be used:\n"
        "superlu_dist\n"
        "The matrix is 12 x 12 elements\n"
        "# of nonzeros in Jacobian matrix: 30, allocated: 40\n"
        "# of nonzeros in Jacobian preconditioner matrix: 18, allocated: 20\n"
        "# of nonzeros in residual f1 matrix: 9, allocated: 16\n"
        "Time to pre-assemble Jacobian preconditioner matrix: 1.5D-1\n"
        "Time to assemble Jacobian preconditioner matrix: 2.5D-1\n"
        "Time to pre-assemble Jacobian matrix: 3.5D-1\n"
        "Time to assemble Jacobian matrix: 4.5D-1\n"
        "Time to pre-assemble residual f1 matrix: 5.5D-1\n"
        "Time to assemble residual f1 matrix: 6.5D-1\n"
        "Elapsed time in solve driver= 1.0D+0\n"
        "Elapsed time in solve driver= 2.0D+0\n"
        " 0 KSP Residual norm 3.0D-2\n"
        " 1 KSP Residual norm 4.0D-8\n"
        "KSPConvergedReason = 4\n"
        "Residual function norm: 8.0D-3\n"
    )

    assert profile["n_mpi_processes"] == 8
    assert profile["solver_package"] == "superlu_dist"
    assert profile["resolution"] == {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 7}
    assert profile["solver_tolerance"] == pytest.approx(1.0e-10)
    assert profile["matrix_shape"] == [12, 12]
    assert profile["matrix_nnz"] == {"nnz": 30, "allocated_nnz": 40}
    assert profile["preconditioner_nnz"] == {"nnz": 18, "allocated_nnz": 20}
    assert profile["residual_matrix_nnz"] == {"nnz": 9, "allocated_nnz": 16}
    assert profile["timings_s"]["preassemble_preconditioner"] == pytest.approx(0.15)
    assert profile["timings_s"]["assemble_preconditioner"] == pytest.approx(0.25)
    assert profile["timings_s"]["preassemble_jacobian"] == pytest.approx(0.35)
    assert profile["timings_s"]["assemble_jacobian"] == pytest.approx(0.45)
    assert profile["timings_s"]["residual_preassemble_f1"] == pytest.approx(0.55)
    assert profile["timings_s"]["residual_assemble_f1"] == pytest.approx(0.65)
    assert profile["timings_s"]["last_solve_driver"] == pytest.approx(2.0)
    assert profile["ksp"]["iteration_count"] == 2
    assert profile["ksp"]["initial_residual"] == pytest.approx(3.0e-2)
    assert profile["ksp"]["final_residual"] == pytest.approx(4.0e-8)
    assert profile["ksp"]["converged_reason"] == 4
    assert profile["residual_function_norm_initial"] == pytest.approx(8.0e-3)
    assert profile["mumps"]["detected"] is False
    assert profile["mumps"]["infog"] == {}


def test_fortran_profile_parser_extracts_mumps_infog_and_tolerates_empty_logs() -> None:
    profile = parse_fortran_v3_profile_text(
        "mumps detected\n"
        "INFOG(16) = 11\n"
        "INFOG(17) = 12\n"
        "INFOG(21) = 21\n"
        "INFOG(22) = 22\n"
        "INFOG(29) = 99\n"
        "INFOG(30) = 100\n"
        "Elapsed time in analysis driver= 7.0e-1\n"
        "Elapsed time in factorization driver= 8.0e-1\n"
    )

    assert profile["mumps"]["detected"] is True
    assert profile["mumps"]["infog"]["16"] == 11
    assert profile["mumps"]["infog"]["29"] == 99
    assert profile["mumps"]["factor_entries"] == pytest.approx(99.0)
    assert profile["mumps"]["analysis_memory_peak_mb"] == 11
    assert profile["mumps"]["analysis_memory_total_mb"] == 12
    assert profile["mumps"]["factor_memory_peak_mb"] == 21
    assert profile["mumps"]["factor_memory_total_mb"] == 22
    assert profile["timings_s"]["mumps_analysis_driver"] == pytest.approx(0.7)
    assert profile["timings_s"]["mumps_factorization_driver"] == pytest.approx(0.8)

    empty = parse_fortran_v3_profile_text("")
    assert empty["solver_package"] is None
    assert empty["matrix_shape"] is None
    assert empty["ksp"]["history"] == []
    assert empty["timings_s"]["last_solve_driver"] is None


def test_fortran_profile_file_parser_and_default_executable_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "sfincs.log"
    log.write_text("Solver package which will be used:\n mumps\n", encoding="utf-8")
    assert parse_fortran_v3_profile_file(log)["solver_package"] == "mumps"

    monkeypatch.delenv("SFINCS_FORTRAN_EXE", raising=False)
    assert default_fortran_exe() is None
    monkeypatch.setenv("SFINCS_FORTRAN_EXE", str(tmp_path / "sfincs"))
    assert default_fortran_exe() == tmp_path / "sfincs"


def _write_fake_fortran_exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_fortran_runner_validates_inputs_and_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_FORTRAN_EXE", raising=False)
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing_input"):
        run_sfincs_fortran(input_namelist=tmp_path / "missing_input.namelist")
    with pytest.raises(ValueError, match="Fortran executable not specified"):
        run_sfincs_fortran(input_namelist=input_path)
    with pytest.raises(FileNotFoundError, match="missing_exe"):
        run_sfincs_fortran(input_namelist=input_path, exe=tmp_path / "missing_exe")


def test_fortran_runner_accepts_successful_output_and_mpi_finalize_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "input_source.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")

    success_exe = _write_fake_fortran_exe(
        tmp_path / "sfincs_success.sh",
        "printf 'Saving diagnostics to h5 file\\nGoodbye!\\n'\n"
        "printf 'fake h5' > sfincsOutput.h5\n"
        "exit 0\n",
    )
    output = run_sfincs_fortran(
        input_namelist=input_path,
        exe=success_exe,
        workdir=tmp_path / "success",
        localize_equilibrium=False,
    )
    assert output == (tmp_path / "success" / "sfincsOutput.h5").resolve()
    assert (tmp_path / "success" / "input.namelist").read_text(encoding="utf-8") == "&general\n/\n"

    mpi_finalize_exe = _write_fake_fortran_exe(
        tmp_path / "sfincs_mpi_finalize.sh",
        "printf 'Saving diagnostics to h5 file\\nGoodbye!\\nMPI_Finalize failed\\n'\n"
        "printf 'fake h5' > sfincsOutput.h5\n"
        "exit 7\n",
    )
    assert run_sfincs_fortran(
        input_namelist=input_path,
        exe=mpi_finalize_exe,
        workdir=tmp_path / "mpi_finalize",
        localize_equilibrium=False,
    ).exists()


def test_fortran_runner_rejects_failed_or_missing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input_source.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")

    no_output_exe = _write_fake_fortran_exe(
        tmp_path / "sfincs_no_output.sh",
        "printf 'Goodbye!\\n'\nexit 0\n",
    )
    with pytest.raises(RuntimeError, match="did not create"):
        run_sfincs_fortran(
            input_namelist=input_path,
            exe=no_output_exe,
            workdir=tmp_path / "no_output",
            localize_equilibrium=False,
        )

    hard_fail_no_output = _write_fake_fortran_exe(
        tmp_path / "sfincs_hard_fail_no_output.sh",
        "printf 'boom\\n'\nexit 5\n",
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_sfincs_fortran(
            input_namelist=input_path,
            exe=hard_fail_no_output,
            workdir=tmp_path / "hard_fail_no_output",
            localize_equilibrium=False,
        )

    hard_fail_with_output = _write_fake_fortran_exe(
        tmp_path / "sfincs_hard_fail_with_output.sh",
        "printf 'Saving diagnostics to h5 file\\nnot clean\\n'\n"
        "printf 'fake h5' > sfincsOutput.h5\n"
        "exit 6\n",
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_sfincs_fortran(
            input_namelist=input_path,
            exe=hard_fail_with_output,
            workdir=tmp_path / "hard_fail_with_output",
            localize_equilibrium=False,
        )


def test_fortran_runner_uses_default_executable_and_can_skip_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    input_path = workdir / "input.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    exe = _write_fake_fortran_exe(
        tmp_path / "sfincs_default.sh",
        "printf 'Saving diagnostics to h5 file\\nGoodbye!\\n'\n"
        "printf 'fake h5' > sfincsOutput.h5\n"
        "exit 0\n",
    )
    monkeypatch.setattr(fortran_validation, "default_fortran_exe", lambda: exe)

    assert run_sfincs_fortran(
        input_namelist=input_path,
        workdir=workdir,
        localize_equilibrium=False,
    ) == (workdir / "sfincsOutput.h5").resolve()


def test_fortran_runner_auto_workdir_localizes_and_merges_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dkx.io as io_module

    input_path = tmp_path / "input_source.namelist"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    calls: list[Path] = []

    def _fake_localize(*, input_namelist: Path, overwrite: bool) -> None:
        calls.append(input_namelist)
        assert overwrite is False

    monkeypatch.setattr(io_module, "localize_equilibrium_file_in_place", _fake_localize)
    exe = _write_fake_fortran_exe(
        tmp_path / "sfincs_env.sh",
        "test \"${DKX_UNIT_ENV:-}\" = expected\n"
        "printf 'Saving diagnostics to h5 file\\nGoodbye!\\n'\n"
        "printf 'fake h5' > sfincsOutput.h5\n"
        "exit 0\n",
    )

    output = run_sfincs_fortran(
        input_namelist=input_path,
        exe=exe,
        env={"DKX_UNIT_ENV": "expected"},
    )

    assert output.name == "sfincsOutput.h5"
    assert output.exists()
    assert calls == [output.parent / "input.namelist"]


def test_find_upstream_utils_dir_resolves_override_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    utils = tmp_path / "utils"
    utils.mkdir()
    assert find_upstream_utils_dir(override=utils) == utils

    with pytest.raises(FileNotFoundError, match="utils dir does not exist"):
        find_upstream_utils_dir(override=tmp_path / "missing")

    monkeypatch.setenv("DKX_UPSTREAM_UTILS_DIR", str(utils))
    assert find_upstream_utils_dir() == utils

    monkeypatch.setenv("DKX_UPSTREAM_UTILS_DIR", str(tmp_path / "missing_env"))
    with pytest.raises(FileNotFoundError, match="DKX_UPSTREAM_UTILS_DIR"):
        find_upstream_utils_dir()


def test_run_upstream_util_executes_noninteractive_script(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()
    script = utils / "make_marker.py"
    script.write_text(
        "from pathlib import Path\n"
        "import builtins\n"
        "import os\n"
        "import sys\n"
        "Path('marker.txt').write_text('|'.join([os.environ['MPLBACKEND'], builtins.input('x'), *sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    messages: list[str] = []

    run_upstream_util(
        util="make_marker.py",
        case_dir=case_dir,
        args=("a", "b"),
        utils_dir=utils,
        emit=lambda _level, message: messages.append(message),
    )

    assert (case_dir / "marker.txt").read_text(encoding="utf-8") == "Agg||a|b"
    assert messages and "make_marker.py" in messages[0]


def test_run_upstream_util_rejects_missing_script_or_case(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Upstream util not found"):
        run_upstream_util(util="missing.py", case_dir=case_dir, utils_dir=utils)

    script = utils / "ok.py"
    script.write_text("pass\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="case_dir does not exist"):
        run_upstream_util(util="ok.py", case_dir=tmp_path / "missing_case", utils_dir=utils)


def test_run_upstream_util_propagates_script_failure(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()
    script = utils / "fail.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        run_upstream_util(util="fail.py", case_dir=case_dir, utils_dir=utils)
