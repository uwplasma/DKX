from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "sfincs_fortran_mpi_wrapper.sh"


def test_fortran_mpi_wrapper_requires_executable_env() -> None:
    env = os.environ.copy()
    env.pop("SFINCS_FORTRAN_EXE", None)

    proc = subprocess.run(
        [str(WRAPPER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "SFINCS_FORTRAN_EXE must point" in proc.stderr


def test_fortran_mpi_wrapper_uses_mpirun_when_available(tmp_path: Path) -> None:
    fake_exe = tmp_path / "sfincs"
    fake_exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_exe.chmod(0o755)

    args_file = tmp_path / "mpirun_args.txt"
    fake_mpirun = tmp_path / "mpirun"
    fake_mpirun.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n",
        encoding="utf-8",
    )
    fake_mpirun.chmod(0o755)

    env = os.environ.copy()
    env["SFINCS_FORTRAN_EXE"] = str(fake_exe)
    env["SFINCS_FORTRAN_MPI_NP"] = "7"
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"

    proc = subprocess.run(
        [str(WRAPPER), "--example-arg"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "-np",
        "7",
        str(fake_exe),
        "--example-arg",
    ]


def test_fortran_mpi_wrapper_defaults_to_one_rank_for_reference_runs(tmp_path: Path) -> None:
    fake_exe = tmp_path / "sfincs"
    fake_exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_exe.chmod(0o755)

    args_file = tmp_path / "mpirun_args.txt"
    fake_mpirun = tmp_path / "mpirun"
    fake_mpirun.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n",
        encoding="utf-8",
    )
    fake_mpirun.chmod(0o755)

    env = os.environ.copy()
    env["SFINCS_FORTRAN_EXE"] = str(fake_exe)
    env.pop("SFINCS_FORTRAN_MPI_NP", None)
    env.pop("SFINCS_MPI_NP", None)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"

    proc = subprocess.run(
        [str(WRAPPER), "--example-arg"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "-np",
        "1",
        str(fake_exe),
        "--example-arg",
    ]
