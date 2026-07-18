"""Smoke tests for the flat pedagogic examples on the canonical stack.

Each flat pedagogic example listed in ``CASES`` (the DKX example style
contract: no ``main()``, parameters at the top, prints of setup/progress/
final results, at least one PNG plot, output-file writing plus read-back) is
run as a subprocess with ``DKX_CI=1`` (the scripts' shrunken-resolution
branch); the test asserts a zero exit code, the expected result lines in
stdout, and that the advertised plot file exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
OUT_DIR = EXAMPLES / "output"

# script name -> (expected stdout fragments, expected plot file)
CASES = {
    "getting_started/run_tokamak.py": (
        (
            "FSABFlow:",  # Fortran-parity species results table (single species)
            "FSABjHat (bootstrap current):",
            "particleFlux_vm_psiHat =",
            "FSABjHat",
            "read back from h5: FSABFlow =",
            "Wrote output files: run_tokamak.sfincsOutput.h5, run_tokamak.sfincsOutput.nc",
        ),
        OUT_DIR / "run_tokamak.png",
    ),
    "getting_started/run_w7x.py": (
        (
            "Solver tier used: gcrot",  # FP collisions must route to tier 2
            "tier-1 structured direct applicable: False",
            "particleFlux_vm_psiHat[ions] =",
            "particleFlux_vm_psiHat[electrons] =",
            "read back from h5: NPeriods = 5",
        ),
        OUT_DIR / "run_w7x.png",
    ),
    "transport/transport_coefficients.py": (
        (
            "L11 (D11-like)",
            "max Onsager asymmetry",
            "read back from h5: nuPrime =",
            "L11 at nuPrime=",
        ),
        OUT_DIR / "transport_coefficients.png",
    ),
    "vmex_finite_beta/ambipolar_er_scan.py": (
        (
            "ambipolar root via er.find_ambipolar_er",
            "ambipolar root: Er =",
            "root classified as:",
            "read back from h5: Er =",
            "Gamma_ions at the root =",
        ),
        OUT_DIR / "ambipolar_er_scan.png",
    ),
    "autodiff/gradients_tour.py": (
        (
            "all gradients verified against central finite differences",
            "d(FSABjHat)/d(THat)",
            "d(particleFlux)/d(dPhiHatdpsiHat)",
            "read back from h5: autodiff =",
        ),
        OUT_DIR / "gradients_tour.png",
    ),
}


@pytest.mark.parametrize("script", sorted(CASES))
def test_pedagogic_example_runs_and_reports(script: str) -> None:
    expected_lines, plot_path = CASES[script]
    if plot_path.exists():
        plot_path.unlink()

    env = dict(os.environ)
    env["DKX_CI"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    for fragment in expected_lines:
        assert fragment in proc.stdout, f"{script}: missing {fragment!r} in stdout"
    assert f"Done: examples/{script}" in proc.stdout
    assert "=== Final results ===" in proc.stdout
    assert plot_path.exists(), f"{script}: plot {plot_path} was not written"
    assert plot_path.stat().st_size > 0
