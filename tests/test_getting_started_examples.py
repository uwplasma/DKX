from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_script(script: Path, *args: str) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(script), *args], cwd=str(script.parents[2]), env=env, check=True)


def test_getting_started_tokamak_example(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "getting_started" / "write_sfincs_output_tokamak.py"
    out_path = tmp_path / "sfincsOutput_tokamak.h5"
    _run_script(script, "--out", str(out_path))
    assert out_path.exists()


def test_getting_started_vmec_example(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "getting_started" / "write_sfincs_output_vmec.py"
    out_path = tmp_path / "sfincsOutput_vmec.h5"
    _run_script(script, "--out", str(out_path))
    assert out_path.exists()


def test_getting_started_plot_example(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "getting_started" / "plot_sfincs_output.py"
    out_path = tmp_path / "sfincsOutput_summary.pdf"
    _run_script(script, "--out", str(out_path))
    assert out_path.exists()


def test_getting_started_multiformat_output_example(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "getting_started" / "write_and_plot_multiple_formats.py"
    _run_script(script, "--out-dir", str(tmp_path))
    assert (tmp_path / "sfincsOutput_getting_started.h5").exists()
    assert (tmp_path / "sfincsOutput_getting_started.nc").exists()
    assert (tmp_path / "sfincsOutput_getting_started.npz").exists()
    assert (tmp_path / "sfincsOutput_getting_started_summary.pdf").exists()


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
        [sys.executable, "-m", "sfincs_jax", "--plot", str(input_h5), "--out", str(output_png)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    assert output_png.exists()


def test_cli_plot_shortcut_accepts_netcdf(tmp_path: Path) -> None:
    from sfincs_jax.io import read_sfincs_h5, write_sfincs_output_file

    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    input_nc = tmp_path / "sfincsOutput.nc"
    output_pdf = tmp_path / "sfincsOutput_summary.pdf"
    write_sfincs_output_file(path=input_nc, data=read_sfincs_h5(input_h5), fortran_layout=False)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, "-m", "sfincs_jax", "--plot", str(input_nc), "--out", str(output_pdf)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    assert output_pdf.exists()
