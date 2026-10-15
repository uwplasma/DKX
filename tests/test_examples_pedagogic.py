"""The tutorial ladder runs, and every script keeps the one shape it promises.

Every ``examples/tutorials/NN_*.py`` is run in its smoke mode
(``DKX_EXAMPLES_CI=1``), which only shrinks grids and scan lengths; the
physics and the code path are the ones a reader runs.  Each must exit cleanly,
print a ``=== Summary ===`` block and write at least one figure under
``examples/output/<script stem>/``.  A few fragments per tutorial pin the
claims the docstrings make about the output.

The style check is static: imports, then input parameters, then the run,
then plot/save/print -- four numbered comments, in order -- and no
``argparse``, ``main()`` or ``__main__`` guard.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPO_ROOT / "examples" / "tutorials"
OUT_DIR = REPO_ROOT / "examples" / "output"
SCRIPTS = sorted(TUTORIALS.glob("[0-9][0-9]_*.py"))
STEPS = ("# 1. Imports", "# 2. Input parameters", "# 3. Run", "# 4. Plot, save and print")

# Fragments a tutorial's printed summary must contain.
EXPECTED = {
    "01_first_run": ("m^-2 s^-1", "W m^-2", "converged: True"),
    "02_cli_case": ("$ dkx run", "case ID from the file: 2ca9e2e8db56  stored in the result: 2ca9e2e8db56"),
    "03_sfincs_namelist": ("particleFlux_vm_psiHat=", "FSABjHat"),
    "04_vmec_geometry": ("wout_vmex_seed_nfp2_ns13.nc", "geometry sha256: "),
    "05_boozer_geometry": ("solver route: gcrot", "route reason: "),
    "06_fokker_planck_vs_pas": ("collisions = linearized_fokker_planck", "parallel_current_A_T_m2"),
    "07_species_and_impurity": ("carbon=", "electron="),
    "08_ambipolar_er": ("[ion]", "all surfaces bracketed: True"),
    "09_monoenergetic": ("D33*", "D11* monotone in nu' at E*=0: True"),
    "10_transport_matrix": ("RHSMode=2", "RHSMode=3", "Onsager check"),
    "11_phi1": ("solver route phi1_newton_krylov", "peak |dn/n|"),
    "12_bootstrap_vs_redl": ("Redl fit", "ratio to Redl"),
    "13_convergence": ("worst change", "converged at 2%: False"),
    "14_gradients": ("jax.grad", "max relative difference"),
    "15_optimization": ("lower)", "central difference"),
}


def test_the_ladder_is_numbered_without_gaps() -> None:
    numbers = [int(path.name[:2]) for path in SCRIPTS]
    assert numbers == list(range(1, len(SCRIPTS) + 1))
    assert set(EXPECTED) == {path.stem for path in SCRIPTS}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_tutorial_runs_in_smoke_mode(script: Path) -> None:
    out = OUT_DIR / script.stem
    for old in out.glob("*.png"):
        old.unlink()
    env = {**os.environ, "DKX_EXAMPLES_CI": "1", "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          cwd=REPO_ROOT, env=env, timeout=900, check=False)  # fmt: skip
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert "=== Summary" in proc.stdout
    for fragment in EXPECTED[script.stem]:
        assert fragment in proc.stdout, f"{script.name}: missing {fragment!r}"
    assert any(path.stat().st_size > 0 for path in out.glob("*.png")), f"{script.name} wrote no figure"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_tutorial_obeys_the_style_contract(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "argparse" not in source and "__main__" not in source
    lines = source.splitlines()
    positions = [next((i for i, line in enumerate(lines) if line.startswith(step)), None) for step in STEPS]
    assert None not in positions, f"{script.name}: missing step comments"
    assert positions == sorted(positions), f"{script.name}: steps out of order"
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" not in functions
    assert functions <= {"bootstrap_current", "particle_flux"}, functions
    docstring = ast.get_docstring(tree) or ""
    for required in ("Run:", "Smoke mode:", "Expected runtime:"):
        assert required in docstring, f"{script.name}: docstring lacks {required!r}"
    assert "DKX_EXAMPLES_CI" in source
