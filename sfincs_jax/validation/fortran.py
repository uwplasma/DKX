from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


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
        from sfincs_jax.io import localize_equilibrium_file_in_place  # noqa: PLC0415

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
