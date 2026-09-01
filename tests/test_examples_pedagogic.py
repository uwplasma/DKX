"""The examples run, and they keep saying what they are advertised to say.

Two groups.

``LADDER_CASES`` is the canonical nine-rung ladder (plan.md section 9.1).
Those scripts carry no ``DKX_CI`` branch at all -- plan.md section 9.2 forbids
mutating the user's scientific case for CI -- so each is sized to run in
seconds at the resolution it ships with, and the test runs exactly the code a
reader runs.  ``test_ladder_example_obeys_the_style_contract`` checks the rest
of 9.2 statically: no ``argparse``, no ``main()``, no ``__main__`` guard, no
helper functions beyond the differentiable objectives the physics needs,
parameters at the top, and the visible eight-step sequence.

``LEGACY_CASES`` is the older topic-folder set, kept because those folders are
still referenced by docs and other tooling.  They do shrink under ``DKX_CI=1``
and are run that way.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
OUT_DIR = EXAMPLES / "output"

# The eight comments plan.md section 9.2 requires, in order.
STYLE_CONTRACT_STEPS = (
    "# 1. Imports",
    "# 2. User-editable parameters",
    "# 3. Geometry and species construction",
    "# 4. Physics and numerical configuration",
    "# 5. Run",
    "# 6. Print a scientific summary and certificate",
    "# 7. Save native result",
    "# 8. Plot publication-ready outputs",
)

# rung -> (expected stdout fragments, expected NetCDF file, expected plot file)
LADDER_CASES = {
    "01_tokamak_profile": (
        (
            "case id (Python) = ",
            "case id (TOML)   = ",
            "workflow: profile on 3 surfaces, 1 species",
            "m^-2 s^-1",  # a printed physical result carries its units
            "W m^-2",
            "converged: True",
            "solver route: block_tridiagonal_truncated",
        ),
        OUT_DIR / "01_tokamak_profile" / "result.nc",
        OUT_DIR / "01_tokamak_profile" / "result.png",
    ),
    "02_vmec_stellarator": (
        (
            "run.py and case.toml agree",
            "wout_up_down_asymmetric_tokamak.nc",
            "geometry sha256: ",  # the equilibrium is pinned in the certificate
            "m^-2 s^-1",
            "converged: True",
        ),
        OUT_DIR / "02_vmec_stellarator" / "result.nc",
        OUT_DIR / "02_vmec_stellarator" / "result.png",
    ),
    "03_boozer_stellarator": (
        (
            "run.py and case.toml agree",
            "nonStelSym_tiny_geometryScheme12.bc",
            # Non-axisymmetric geometry: the operator's structure sends "auto"
            # to the recycled Krylov route, not the structured direct one.
            "solver route: gcrot",
            "route reason: ",
            "converged: True",
        ),
        OUT_DIR / "03_boozer_stellarator" / "result.nc",
        OUT_DIR / "03_boozer_stellarator" / "result.png",
    ),
    "04_monoenergetic_scan": (
        (
            "D11*",
            "D31*",
            "D33*",
            "collisional limit is 1",
            "D11* monotone in nu' at E*=0: True",
        ),
        OUT_DIR / "04_monoenergetic_scan" / "monoenergetic.nc",
        OUT_DIR / "04_monoenergetic_scan" / "monoenergetic.png",
    ),
    "05_ambipolar_profile": (
        (
            "root(s), selected #",
            "kV/m",
            "[ion]",  # the root is classified, not just returned
            "all surfaces bracketed: True",
            "selection rule: ",
        ),
        OUT_DIR / "05_ambipolar_profile" / "result.nc",
        OUT_DIR / "05_ambipolar_profile" / "result.png",
    ),
    "06_convergence_certificate": (
        (
            "(the linear solve, not the discretization)",
            "worst single-axis change:",
            "joint change:",
            "axes understate the joint change:",
            "converged at 2.0%: False",  # the shipped resolution is coarse on purpose
            "provenance: dkx ",
        ),
        OUT_DIR / "06_convergence_certificate" / "result.nc",
        OUT_DIR / "06_convergence_certificate" / "convergence.png",
    ),
    "07_gradients": (
        (
            "jax.grad           d<j.B>/dTHat",
            "central difference d<j.B>/dTHat",
            "all gradients verified against central finite differences",
            "reverse mode is one transposed solve",
        ),
        OUT_DIR / "07_gradients" / "gradients.nc",
        OUT_DIR / "07_gradients" / "gradients.png",
    ),
    "08_vmex_optimization": (
        (
            "optional geometry backends:",
            "vmex hand-off claim scope: geometry_proxy_gradient_only",
            "full VMEC-boundary transport gradients claimed: False",
            "all gradients verified against central finite differences",
            "physics check: removing helical ripple lowered the neoclassical flux",
            "kinetic solve executed: True",
        ),
        OUT_DIR / "08_vmex_optimization" / "optimization.nc",
        OUT_DIR / "08_vmex_optimization" / "optimization.png",
    ),
    "09_phi1_and_impurities": (
        (
            "Phi1 off: solver route gcrot",
            "Phi1 on:  solver route phi1_newton_krylov",
            "impurity particle flux changes by",
            "every species moved: Phi1 is not an impurity-only correction on this deck",
            "sum_s Z_s Gamma_s (Phi1 on)",
        ),
        OUT_DIR / "09_phi1_and_impurities" / "phi1_and_impurities.nc",
        OUT_DIR / "09_phi1_and_impurities" / "phi1_and_impurities.png",
    ),
}

# script name -> (expected stdout fragments, expected plot file)
LEGACY_CASES = {
    "getting_started/build_input_from_python.py": (
        (
            "SfincsInput round trip verified",
            "in-memory run matches the file-based run: particleFlux_vm_psiHat =",
            "solver route used: block_tridiagonal",  # SolverOptions(method="auto") on a PAS deck
            "nu_n = 3.0e-03:  particleFlux_vm_psiHat =",
        ),
        OUT_DIR / "build_input_from_python.png",
    ),
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
            "Solver route used: gcrot",  # FP collisions must take the recycled Krylov route
            "structured direct applicable: False",
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


def _run(script: Path, *, ci: bool) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if ci:
        env["DKX_CI"] = "1"
    else:
        env.pop("DKX_CI", None)
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=600,
    )


@pytest.mark.parametrize("rung", sorted(LADDER_CASES))
def test_ladder_example_runs_and_reports(rung: str) -> None:
    expected_lines, result_path, plot_path = LADDER_CASES[rung]
    for path in (result_path, plot_path):
        path.unlink(missing_ok=True)

    script = EXAMPLES / rung / "run.py"
    # No DKX_CI: the ladder is sized to run at the resolution it ships with, so
    # CI exercises exactly the code a reader runs (plan.md section 9.2).
    proc = _run(script, ci=False)

    assert proc.returncode == 0, f"{rung} failed:\n{proc.stdout}\n{proc.stderr}"
    for fragment in expected_lines:
        assert fragment in proc.stdout, f"{rung}: missing {fragment!r} in stdout"
    assert "=== Final results ===" in proc.stdout
    assert f"Done: examples/{rung}/run.py" in proc.stdout
    assert result_path.exists(), f"{rung}: NetCDF {result_path} was not written"
    assert result_path.stat().st_size > 0
    assert plot_path.exists(), f"{rung}: plot {plot_path} was not written"
    assert plot_path.stat().st_size > 0


@pytest.mark.parametrize("rung", sorted(LADDER_CASES))
def test_ladder_example_obeys_the_style_contract(rung: str) -> None:
    """plan.md section 9.2, checked rather than reviewed."""
    script = EXAMPLES / rung / "run.py"
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "argparse" not in source, f"{rung}: examples take no command-line arguments"
    assert "__main__" not in source, f"{rung}: no __main__ guard; the script is the program"
    assert "DKX_CI" not in source, f"{rung}: no CI branch may change the scientific case"

    # The eight steps, in order, each on its own line.
    lines = source.splitlines()
    positions = [
        next((index for index, line in enumerate(lines) if line.strip() == step), None)
        for step in STYLE_CONTRACT_STEPS
    ]
    missing = [
        step for step, position in zip(STYLE_CONTRACT_STEPS, positions) if position is None
    ]
    assert missing == [], f"{rung}: missing step comments {missing}"
    assert positions == sorted(positions), f"{rung}: the eight steps are out of order"

    # The only module-level functions allowed are the differentiable objectives
    # rungs 07 and 08 have to define to hand to jax.grad; anything else is a
    # helper that belongs in the library rather than in a teaching script.
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert "main" not in functions, f"{rung}: no main(); the script runs top to bottom"
    allowed = {"bootstrap_current", "particle_flux"}
    assert set(functions) <= allowed, f"{rung}: unexpected helper functions {functions}"

    # Parameters live at the top, above the marker every rung carries.
    assert "# end of parameters" in source, f"{rung}: no 'end of parameters' marker"

    docstring = ast.get_docstring(tree)
    assert docstring is not None
    assert "Expected runtime:" in docstring, f"{rung}: docstring omits the expected runtime"
    assert "Physics:" in docstring or "physics" in docstring.lower(), (
        f"{rung}: docstring omits the physical regime"
    )


@pytest.mark.parametrize("script", sorted(LEGACY_CASES))
def test_pedagogic_example_runs_and_reports(script: str) -> None:
    expected_lines, plot_path = LEGACY_CASES[script]
    if plot_path.exists():
        plot_path.unlink()

    proc = _run(EXAMPLES / script, ci=True)

    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    for fragment in expected_lines:
        assert fragment in proc.stdout, f"{script}: missing {fragment!r} in stdout"
    assert f"Done: examples/{script}" in proc.stdout
    assert "=== Final results ===" in proc.stdout
    assert plot_path.exists(), f"{script}: plot {plot_path} was not written"
    assert plot_path.stat().st_size > 0
