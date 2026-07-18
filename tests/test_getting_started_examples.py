from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Converted teaching scripts write to examples/output/<script_stem>/ (their
# top-level OUTPUT_DIR parameter); each is now run with no CLI flags.
EXAMPLES_OUTPUT = REPO_ROOT / "examples" / "output"


def _run_script(script: Path, *args: str) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(script), *args], cwd=str(script.parents[2]), env=env, check=True)


def test_getting_started_tokamak_example() -> None:
    script = REPO_ROOT / "examples" / "getting_started" / "write_sfincs_output_tokamak.py"
    out_path = EXAMPLES_OUTPUT / "write_sfincs_output_tokamak" / "sfincsOutput_tokamak.h5"
    out_path.unlink(missing_ok=True)
    _run_script(script)
    assert out_path.exists()


def test_getting_started_vmec_example() -> None:
    script = REPO_ROOT / "examples" / "getting_started" / "write_sfincs_output_vmec.py"
    out_path = EXAMPLES_OUTPUT / "write_sfincs_output_vmec" / "sfincsOutput_vmec.h5"
    out_path.unlink(missing_ok=True)
    _run_script(script)
    assert out_path.exists()


def test_getting_started_plot_example() -> None:
    script = REPO_ROOT / "examples" / "getting_started" / "plot_sfincs_output.py"
    out_path = EXAMPLES_OUTPUT / "plot_sfincs_output" / "sfincsOutput_summary.pdf"
    out_path.unlink(missing_ok=True)
    _run_script(script)
    assert out_path.exists()


def test_getting_started_multiformat_output_example() -> None:
    script = REPO_ROOT / "examples" / "getting_started" / "write_and_plot_multiple_formats.py"
    out_dir = EXAMPLES_OUTPUT / "write_and_plot_multiple_formats"
    _run_script(script)
    assert (out_dir / "sfincsOutput_getting_started.h5").exists()
    assert (out_dir / "sfincsOutput_getting_started.nc").exists()
    assert (out_dir / "sfincsOutput_getting_started.npz").exists()
    assert (out_dir / "sfincsOutput_getting_started_summary.pdf").exists()


def test_tutorial_quick_output_and_plot_script() -> None:
    script = REPO_ROOT / "examples" / "tutorials" / "run_quick_output_and_plot.py"
    out_dir = EXAMPLES_OUTPUT / "run_quick_output_and_plot"
    _run_script(script)
    assert (out_dir / "sfincsOutput_tutorial.h5").exists()
    assert (out_dir / "sfincsOutput_tutorial.nc").exists()
    assert (out_dir / "sfincsOutput_tutorial.npz").exists()
    assert (out_dir / "sfincsOutput_tutorial_summary.pdf").exists()


def test_output_format_benchmark_example(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "performance" / "benchmark_output_formats.py"
    out_json = tmp_path / "output_benchmark.json"
    _run_script(script, "--repeats", "1", "--out-dir", str(tmp_path), "--json", str(out_json))
    text = out_json.read_text(encoding="utf-8")
    assert ".h5" in text
    assert ".nc" in text
    assert ".npz" in text


def test_cli_plot_shortcut_on_fixture(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    output_png = tmp_path / "sfincsOutput_summary.png"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, "-m", "dkx", "--plot", str(input_h5), "--out", str(output_png)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    assert output_png.exists()


def test_cli_plot_shortcut_accepts_netcdf(tmp_path: Path) -> None:
    from dkx.io import read_sfincs_h5, write_sfincs_output_file

    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    input_nc = tmp_path / "sfincsOutput.nc"
    output_pdf = tmp_path / "sfincsOutput_summary.pdf"
    write_sfincs_output_file(path=input_nc, data=read_sfincs_h5(input_h5), fortran_layout=False)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, "-m", "dkx", "--plot", str(input_nc), "--out", str(output_pdf)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    assert output_pdf.exists()


def test_readme_quick_solve_command_uses_public_auto_path(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_path = repo / "examples" / "sfincs_examples" / "quick_2species_FPCollisions_noEr" / "input.namelist"
    output_h5 = tmp_path / "sfincsOutput.h5"
    output_pdf = tmp_path / "sfincsOutput_summary.pdf"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.update(
        {
            "DKX_CPU_DEVICES": "1",
            "DKX_TRANSPORT_PARALLEL": "off",
            "DKX_TRANSPORT_PARALLEL_WORKERS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dkx",
            str(input_path),
            "--out",
            str(output_h5),
            "--quiet",
        ],
        cwd=str(repo),
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dkx",
            "--plot",
            str(output_h5),
            "--out",
            str(output_pdf),
            "--quiet",
        ],
        cwd=str(repo),
        env=env,
        check=True,
    )
    assert output_h5.exists()
    assert output_pdf.exists()
