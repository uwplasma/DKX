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
    pytest.importorskip("matplotlib")
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "getting_started" / "plot_sfincs_output.py"
    out_path = tmp_path / "sfincsOutput_summary.png"
    _run_script(script, "--out", str(out_path))
    assert out_path.exists()
