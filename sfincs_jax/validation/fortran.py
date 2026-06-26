from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


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
