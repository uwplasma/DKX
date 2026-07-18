from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_VEC_CLASSID = 1211214
_MAT_CLASSID = 1211216


def _require_binary_size(*, payload: bytes, expected_bytes: int, label: str) -> None:
    """Fail closed before NumPy views a truncated PETSc binary payload."""

    if len(payload) < int(expected_bytes):
        raise ValueError(
            f"Invalid PETSc {label}: expected at least {int(expected_bytes)} bytes, "
            f"found {len(payload)}."
        )


@dataclass(frozen=True)
class PetscVec:
    """PETSc binary vector read from a SFINCS Fortran-v3 fixture."""

    values: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of vector entries."""

        return int(self.values.size)


@dataclass(frozen=True)
class PetscCSRMatrix:
    """PETSc AIJ matrix represented as CSR arrays."""

    shape: tuple[int, int]
    row_ptr: np.ndarray
    col_ind: np.ndarray
    data: np.ndarray

    def get(self, i: int, j: int) -> float:
        """Return ``A[i, j]`` for sorted CSR rows, or zero if absent."""

        i = int(i)
        j = int(j)
        start = int(self.row_ptr[i])
        end = int(self.row_ptr[i + 1])
        cols = self.col_ind[start:end]
        k = int(np.searchsorted(cols, j))
        if k < cols.size and int(cols[k]) == j:
            return float(self.data[start + k])
        return 0.0


def read_petsc_vec(path: str | Path) -> PetscVec:
    """Read a PETSc Vec binary file written by SFINCS Fortran v3."""

    b = Path(path).read_bytes()
    if len(b) < 2 * 4:
        raise ValueError("File too small to be a PETSc Vec.")
    header = np.frombuffer(b, dtype=">i4", count=2)
    if header.size != 2:
        raise ValueError("File too small to be a PETSc Vec.")
    classid, n = (int(header[0]), int(header[1]))
    if classid != _VEC_CLASSID:
        raise ValueError(f"Unexpected PETSc Vec classid={classid} (expected {_VEC_CLASSID}).")
    if n < 0:
        raise ValueError(f"Invalid PETSc Vec: negative size n={n}.")
    _require_binary_size(payload=b, expected_bytes=2 * 4 + int(n) * 8, label="Vec")
    values = np.frombuffer(b, dtype=">f8", offset=2 * 4, count=n).astype(np.float64, copy=False)
    return PetscVec(values=values)


def read_petsc_mat_aij(path: str | Path) -> PetscCSRMatrix:
    """Read a PETSc AIJ Mat binary file into CSR.

    The supported format is the one written by
    ``MatView(..., PETSC_VIEWER_BINARY_)`` in the Fortran-v3 parity fixtures.
    """

    b = Path(path).read_bytes()
    if len(b) < 4 * 4:
        raise ValueError("File too small to be a PETSc Mat.")
    header = np.frombuffer(b, dtype=">i4", count=4)
    if header.size != 4:
        raise ValueError("File too small to be a PETSc Mat.")
    classid, m, n, nnz = (int(header[0]), int(header[1]), int(header[2]), int(header[3]))
    if classid != _MAT_CLASSID:
        raise ValueError(f"Unexpected PETSc Mat classid={classid} (expected {_MAT_CLASSID}).")
    if m < 0 or n < 0 or nnz < 0:
        raise ValueError(f"Invalid PETSc Mat: negative dimension/count m={m}, n={n}, nnz={nnz}.")

    offset = 4 * 4
    _require_binary_size(payload=b, expected_bytes=offset + int(m) * 4, label="Mat row counts")
    row_nnz = np.frombuffer(b, dtype=">i4", offset=offset, count=m).astype(np.int32, copy=False)
    offset += m * 4
    _require_binary_size(payload=b, expected_bytes=offset + int(nnz) * 4, label="Mat column indices")
    col_ind = np.frombuffer(b, dtype=">i4", offset=offset, count=nnz).astype(np.int32, copy=False)
    offset += nnz * 4
    _require_binary_size(payload=b, expected_bytes=offset + int(nnz) * 8, label="Mat values")
    data = np.frombuffer(b, dtype=">f8", offset=offset, count=nnz).astype(np.float64, copy=False)

    row_ptr = np.zeros((m + 1,), dtype=np.int64)
    row_ptr[1:] = np.cumsum(row_nnz, dtype=np.int64)

    if int(row_ptr[-1]) != nnz:
        raise ValueError("Invalid PETSc Mat: row pointers do not sum to nnz.")

    return PetscCSRMatrix(
        shape=(m, n),
        row_ptr=row_ptr,
        col_ind=col_ind,
        data=data,
    )


def default_fortran_exe() -> Path | None:
    env = os.environ.get("SFINCS_FORTRAN_EXE")
    return Path(env) if env else None


def run_sfincs_fortran(
    *,
    input_namelist: Path,
    exe: Path | None = None,
    workdir: Path | None = None,
    env: dict[str, str] | None = None,
    localize_equilibrium: bool = True,
    timeout_s: float | None = None,
) -> Path:
    """Run the compiled Fortran SFINCS v3 executable.

    Notes
    -----
    - The Fortran executable is **not** shipped as part of this package.
    - The executable is expected to read `input.namelist` from the working directory and
      write `sfincsOutput.h5` there.
    """
    input_namelist = input_namelist.resolve()
    if not input_namelist.exists():
        raise FileNotFoundError(str(input_namelist))

    exe = (exe or default_fortran_exe())
    if exe is None:
        raise ValueError(
            "Fortran executable not specified. Pass `exe=...` or set SFINCS_FORTRAN_EXE."
        )
    exe = exe.resolve()
    if not exe.exists():
        raise FileNotFoundError(str(exe))

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="sfincs_fortran_run_"))
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dst_input = workdir / "input.namelist"
    if input_namelist != dst_input:
        shutil.copyfile(input_namelist, dst_input)

    if bool(localize_equilibrium):
        # Many upstream inputs set equilibriumFile relative to the upstream SFINCS repo.
        # When we run in a temporary workdir, localize the referenced file next to input.namelist.
        from dkx.io import localize_equilibrium_file_in_place  # noqa: PLC0415

        localize_equilibrium_file_in_place(input_namelist=dst_input, overwrite=False)

    log_path = workdir / "sfincs.log"
    output_path = workdir / "sfincsOutput.h5"
    with log_path.open("w") as log:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        completed = subprocess.run(
            [str(exe)],
            cwd=str(workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            check=False,
            timeout=timeout_s,
        )

    if completed.returncode != 0 and output_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        completed_cleanly = "Goodbye!" in log_text and "Saving diagnostics to h5 file" in log_text
        mpi_finalize_failure = "MPI_Finalize failed" in log_text or "internal_Finalize" in log_text
        if not (completed_cleanly and mpi_finalize_failure):
            raise subprocess.CalledProcessError(completed.returncode, [str(exe)])
    elif completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, [str(exe)])

    if not output_path.exists():
        raise RuntimeError(
            f"Fortran run finished but did not create {output_path}. See {log_path}."
        )
    return output_path


# Fortran v3 PETSc/MUMPS log profiling helpers.
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[dDeE][-+]?\d+)?"


def _to_float(text: str) -> float:
    return float(str(text).replace("D", "E").replace("d", "e"))


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return int(match.group(1)) if match is not None else None


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return _to_float(match.group(1)) if match is not None else None


def _last_float(pattern: str, text: str) -> float | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return _to_float(matches[-1].group(1)) if matches else None


def _matrix_stats(label: str, text: str) -> dict[str, int] | None:
    pattern = (
        rf"# of nonzeros in {re.escape(label)}:\s+(\d+)\s*,\s*allocated:\s*"
        rf"(\d+)"
    )
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return None
    return {"nnz": int(match.group(1)), "allocated_nnz": int(match.group(2))}


def _ksp_residual_history(text: str) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    for match in re.finditer(r"^\s*(\d+)\s+KSP Residual norm\s+(" + _FLOAT + r")\s*$", text, flags=re.MULTILINE):
        history.append({"iteration": int(match.group(1)), "residual": _to_float(match.group(2))})
    return history


def parse_fortran_v3_profile_text(text: str) -> dict[str, Any]:
    """Parse a Fortran-v3 stdout/PETSc log.

    The parser intentionally targets high-value production profiling fields and
    tolerates missing lines so it can be used for local logs, archived Slurm
    outputs, and abbreviated CI fixtures.
    """
    text = str(text)
    matrix_shape = re.search(r"The matrix is\s+(\d+)\s+x\s+(\d+)\s+elements", text)
    ksp_history = _ksp_residual_history(text)
    solve_driver_times = [_to_float(match.group(1)) for match in re.finditer(r"Elapsed time in solve driver=\s*(" + _FLOAT + r")", text)]

    mumps_infog: dict[str, int] = {}
    for key in (3, 4, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 29, 30, 31):
        match = re.search(rf"\bINFOG\({key}\).*?(?:=|:)\s*(\d+)", text)
        if match is not None:
            mumps_infog[str(key)] = int(match.group(1))

    profile: dict[str, Any] = {
        "schema_version": 1,
        "n_mpi_processes": _first_int(r"Parallel job \(\s*(\d+) processes\) detected", text),
        "solver_package": None,
        "resolution": {
            "Ntheta": _first_int(r"^\s*Ntheta\s+=\s+(\d+)", text),
            "Nzeta": _first_int(r"^\s*Nzeta\s+=\s+(\d+)", text),
            "Nxi": _first_int(r"^\s*Nxi\s+=\s+(\d+)", text),
            "NL": _first_int(r"^\s*NL\s+=\s+(\d+)", text),
            "Nx": _first_int(r"^\s*Nx\s+=\s+(\d+)", text),
        },
        "solver_tolerance": _first_float(r"^\s*solverTolerance\s+=\s+(" + _FLOAT + r")", text),
        "matrix_shape": [int(matrix_shape.group(1)), int(matrix_shape.group(2))] if matrix_shape is not None else None,
        "matrix_nnz": _matrix_stats("Jacobian matrix", text),
        "preconditioner_nnz": _matrix_stats("Jacobian preconditioner matrix", text),
        "residual_matrix_nnz": _matrix_stats("residual f1 matrix", text),
        "timings_s": {
            "preassemble_preconditioner": _first_float(
                r"Time to pre-assemble Jacobian preconditioner matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "assemble_preconditioner": _first_float(
                r"Time to assemble Jacobian preconditioner matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "preassemble_jacobian": _first_float(
                r"Time to pre-assemble Jacobian matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "assemble_jacobian": _first_float(
                r"Time to assemble Jacobian matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "mumps_analysis_driver": _first_float(r"Elapsed time in analysis driver=\s*(" + _FLOAT + r")", text),
            "mumps_factorization_driver": _first_float(
                r"Elapsed time in factorization driver=\s*(" + _FLOAT + r")",
                text,
            ),
            "residual_preassemble_f1": _first_float(
                r"Time to pre-assemble residual f1 matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "residual_assemble_f1": _first_float(
                r"Time to assemble residual f1 matrix:\s*(" + _FLOAT + r")",
                text,
            ),
            "last_solve_driver": solve_driver_times[-1] if solve_driver_times else None,
        },
        "mumps": {
            "detected": "mumps detected" in text or "Entering DMUMPS" in text,
            "infog": mumps_infog,
            "factor_entries": _last_float(r"INFOG\(29\).*?(?:=|:)\s*(" + _FLOAT + r")", text),
            "factor_memory_peak_mb": _first_int(r"INFOG\(21\).*?(?:=|:)\s*(\d+)", text),
            "factor_memory_total_mb": _first_int(r"INFOG\(22\).*?(?:=|:)\s*(\d+)", text),
            "analysis_memory_peak_mb": _first_int(r"INFOG\(16\).*?(?:=|:)\s*(\d+)", text),
            "analysis_memory_total_mb": _first_int(r"INFOG\(17\).*?(?:=|:)\s*(\d+)", text),
        },
        "ksp": {
            "history": ksp_history,
            "iteration_count": len(ksp_history),
            "initial_residual": ksp_history[0]["residual"] if ksp_history else None,
            "final_residual": ksp_history[-1]["residual"] if ksp_history else None,
            "converged_reason": _first_int(r"KSPConvergedReason\s+=\s+(-?\d+)", text),
        },
        "residual_function_norm_initial": _first_float(
            r"Residual function norm:\s*(" + _FLOAT + r")",
            text,
        ),
    }
    solver_match = re.search(r"Solver package which will be used:\s*\n\s*(\S+)", text)
    if solver_match is not None:
        profile["solver_package"] = solver_match.group(1).strip()
    return profile


def parse_fortran_v3_profile_file(path: str | Path) -> dict[str, Any]:
    """Read and parse a Fortran-v3 stdout/PETSc log file."""
    return parse_fortran_v3_profile_text(Path(path).read_text(encoding="utf-8", errors="replace"))
