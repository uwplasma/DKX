"""The two VMEX-chain examples in ``examples/advanced``.

``QA_optimization_bootstrap_dkx.py`` is VMEX's QA bootstrap-current
optimization with the Redl row replaced by the kinetic one DKX computes on the
same equilibrium; ``KineticBootstrapMismatch`` is traced, so VMEX's implicit
Jacobian carries it.  ``ambipolar_bootstrap_dkx.py`` evaluates that row at the
ambipolar E_r and checks its derivative.  Running either at its committed
parameters takes tens of minutes, so the slow tests run a copy whose
parameters are overridden after the input block; the static tests pin the
parts that break silently.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EX_DIR = REPO_ROOT / "examples" / "advanced"
END_OF_INPUTS = "# End of input parameters."

# Parameter overrides that make each script a wiring check, not an optimization.
SMOKE = {
    "QA_optimization_bootstrap_dkx.py": """
SURFACES, KINETIC_SURFACES = np.linspace(0.2, 0.8, 4), [0.5]
KINETIC_RESOLUTION = dict(Ntheta=9, Nzeta=9, Nxi=12, NL=4, Nx=4)
MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE = [1], [2], [4]
PICARD_ITERATIONS, REDL_N_LAMBDA = 2, 12
FINAL_NS, FINAL_FTOL = 31, 1e-10
""",
    "ambipolar_bootstrap_dkx.py": """
KINETIC_SURFACES = (0.3,)
CHECK_DOFS, FD_TOLERANCE, NS, PICARD_ITERATIONS = CHECK_DOFS[:1], 0.2, 13, 2
""",
}


def _source(name: str) -> str:
    path = EX_DIR / name
    assert path.is_file(), path
    return path.read_text(encoding="utf-8")


def _run_smoke(name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    source = _source(name)
    assert END_OF_INPUTS in source
    script = tmp_path / name
    seed = REPO_ROOT / "examples" / "data" / "input.minimal_seed_nfp2"  # the copy runs from tmp_path
    overrides = f"INPUT_FILE = Path({str(seed)!r})\n" + SMOKE[name]
    script.write_text(source.replace(END_OF_INPUTS, overrides + END_OF_INPUTS), encoding="utf-8")
    return subprocess.run([sys.executable, "-u", str(script)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=5400, check=False)  # fmt: skip


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_script_is_flat_and_has_no_switches(name: str) -> None:
    source = _source(name)
    tree = ast.parse(source)
    assert not [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    for banned in ("argparse", "sys.argv", "os.environ", "__main__"):
        assert banned not in source, f"{name}: {banned}"
    docstring = ast.get_docstring(tree) or ""
    assert "Run:" in docstring and "Runtime:" in docstring


def test_the_qa_flagship_carries_the_traced_kinetic_row() -> None:
    """QA swaps the Redl row for the traced DKX row under VMEX's implicit Jacobian."""
    source = _source("QA_optimization_bootstrap_dkx.py")
    assert "from dkx.bootstrap import KineticBootstrapMismatch" in source
    assert "kinetic = KineticBootstrapMismatch(profiles" in source
    assert '"dkx": [(kinetic, 0.0, BOOTSTRAP_WEIGHT)]' in source
    assert '"redl": [(bootstrap, 0.0, BOOTSTRAP_WEIGHT)]' in source
    assert 'BOOTSTRAP_MODEL = "dkx"' in source
    # The traced row needs no finite-difference Jacobian, and both models are reported.
    assert "finite_difference" not in source
    assert '("f_boot Redl", bootstrap.total' in source and '("f_boot DKX", kinetic.total' in source
    # The seed deck ships with DKX, where VMEX's template looks for it.
    assert (REPO_ROOT / "examples" / "data" / "input.minimal_seed_nfp2").is_file()


def test_the_qa_script_stays_close_to_its_vmex_template() -> None:
    """"One term swapped" is the claim; a diverged script quietly breaks it."""
    source = " ".join(_source("QA_optimization_bootstrap_dkx.py").split())
    for marker in ("self_consistent_bootstrap", "KineticProfiles", "OptimizationMonitor",
                   "problem.residual, problem.x0, jac=problem.residual_jac",
                   "opt.VmecProblem.from_tuples", "monitor.save", "vj.write_wout"):  # fmt: skip
        assert marker in source, f"lost the vmex workflow marker {marker!r}"


@pytest.mark.slow  # a Picard seed, one implicit-Jacobian stage, and a final solve
def test_the_qa_flagship_runs_end_to_end(tmp_path) -> None:
    """Parsing is not running.  This is the only check that the chain closes."""
    pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax.jax_api")
    done = _run_smoke("QA_optimization_bootstrap_dkx.py", tmp_path)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    out = done.stdout
    assert "[self-consistent seed]" in out and "[final]" in out
    assert "f_boot DKX" in out and "f_boot Redl" in out
    assert "<j.B> [MA T/m^2], seed" in out and "<j.B> [MA T/m^2], final" in out
    for name in ("input.QA_bootstrap_dkx_optimized", "wout_QA_bootstrap_dkx_optimized.nc",
                 "QA_bootstrap_dkx_current.png"):  # fmt: skip
        assert (tmp_path / name).exists(), name


@pytest.mark.slow  # two equilibrium solves, the traced row and its Jacobian check
def test_the_ambipolar_example_runs_and_checks_its_own_derivative(tmp_path) -> None:
    pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax.jax_api")
    done = _run_smoke("ambipolar_bootstrap_dkx.py", tmp_path)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert "ambipolar E_r" in done.stdout and "relative" in done.stdout
    assert (tmp_path / "ambipolar_bootstrap_dkx.png").exists()
    assert (tmp_path / "ambipolar_bootstrap_dkx.csv").exists()


@pytest.mark.parametrize("include_redl", [False, True])
def test_bootstrap_plot_preserves_profile_units_and_finite_rms(tmp_path, monkeypatch, include_redl):
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np

    from dkx.bootstrap import plot_kinetic_bootstrap_current

    surfaces = np.array([0.25, 0.75])
    kinetic = SimpleNamespace(current_profile=lambda eq: (surfaces, np.array([2e6, np.nan])))
    redl = SimpleNamespace(current_profiles=lambda eq: (surfaces, [3e6, -3e6], [4e6, -4e6]))
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    try:
        out = plot_kinetic_bootstrap_current(
            tmp_path / "current.png", object(), kinetic, redl=redl if include_redl else None
        )
        axis = plt.gcf().axes[0]
        lines = {line.get_label(): line for line in axis.lines}
        np.testing.assert_array_equal(lines["DKX kinetic"].get_xdata(), surfaces)
        np.testing.assert_allclose(lines["DKX kinetic"].get_ydata(), [2.0, np.nan], equal_nan=True)
        assert "kinetic RMS = 2 " in axis.texts[0].get_text()
        assert "MA T" in axis.get_ylabel()
        if include_redl:
            np.testing.assert_allclose(lines["VMEC equilibrium"].get_ydata(), [3.0, -3.0])
            np.testing.assert_allclose(lines["Redl bootstrap"].get_ydata(), [4.0, -4.0])
        else:
            assert "VMEC equilibrium" not in lines and "Redl bootstrap" not in lines
        assert out.is_file()
    finally:
        close("all")
