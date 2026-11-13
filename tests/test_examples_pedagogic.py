"""The tutorial ladder runs, and every example keeps the one shape it promises.

Every ``examples/tutorials/NN_*.py`` is run with the grid shrunk by a few
plain assignments inserted before its ``# 3. Run`` step (``SMOKE`` below), so
the suite stays fast while the script itself has no switches and the physics
and code path are the ones a reader runs.  Each must exit cleanly,
print a ``=== Summary ===`` block and write a result under
``examples/output/<script stem>/``.  A few fragments per tutorial pin the
claims the docstrings make about the output.

The style check is static and covers the tutorials and ``examples/advanced``:
a docstring with ``Run:`` and ``Runtime:`` lines, no ``argparse``,
``sys.argv``, ``os.environ``, ``main()`` or ``__main__`` guard; the tutorials
also keep four numbered step comments (imports, input parameters, run, plot).
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
ADVANCED = sorted((REPO_ROOT / "examples" / "advanced").glob("*.py"))
OPTIONAL = {"17_vmex_geometry": "vmex"}  # tutorials that need an optional package
STEPS = ("# 1. Imports", "# 2. Input parameters", "# 3. Run", "# 4. Plot, save and print")

# Assignments run before step 3: coarser grids and shorter scans, nothing else.
SMOKE = {
    "01_first_run": 'CASE["resolution"] = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}',
    "04_vmec_geometry": 'RESOLUTION = {"theta": 9, "zeta": 9, "pitch": 8, "speed": 4}',
    "05_boozer_geometry": 'RESOLUTION = {"theta": 5, "zeta": 5, "pitch": 8, "speed": 4}',
    "06_collision_operators": "PARAMETERS.update(Ntheta=9, Nxi=8, Nx=4)",
    "07_species_and_impurity": 'CASE["resolution"] = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}',
    "08_ambipolar_er": "ER_SCAN_KV_M, SEARCH_POINTS = ER_SCAN_KV_M[::4], 3",
    "09_monoenergetic": "NU_PRIME = NU_PRIME[::4]",
    "10_transport_matrix": ("for c in CASES.values():\n"
                            "    c.update(Ntheta=9, Nzeta=9, Nxi=6, NL=3, Nx=3 if c['RHSMode'] == 2 else 1)"),
    "11_phi1": "PARAMETERS.update(Ntheta=9, Nxi=8, Nx=4)",
    "12_bootstrap_vs_redl": "PARAMETERS.update(Ntheta=11, Nzeta=11, Nxi=16, Nx=4)",
    "14_gradients": "PARAMETERS.update(Nxi=12)\nSWEEP = SWEEP[::2]",
    "15_optimization": "N_STEPS = 2\nPARAMETERS.update(Nxi=10)",
}
RUNNER = (  # runs a script as __main__ with its smoke assignments inserted before "# 3. Run"
    "import sys\n"
    "path, smoke = sys.argv[1:3]\n"
    "source = open(path, encoding='utf-8').read()\n"
    "source = source.replace('# 3. Run', smoke + '\\n# 3. Run', 1) if smoke else source\n"
    "exec(compile(source, path, 'exec'), {'__file__': path, '__name__': '__main__'})\n"
)

# Fragments a tutorial's printed summary must contain.
EXPECTED = {
    "01_first_run": ("m^-2 s^-1", "W m^-2", "converged: True"),
    "02_cli_case": ("$ dkx run", "case ID from the file: 2ca9e2e8db56  stored in the result: 2ca9e2e8db56"),
    "03_sfincs_namelist": ("particleFlux_vm_psiHat=", "FSABjHat"),
    "04_vmec_geometry": ("wout_vmex_seed_nfp2_ns13.nc", "geometry sha256: "),
    "05_boozer_geometry": ("gcrot", "route reason: "),
    "06_collision_operators": ("Sugama", "pitch-angle scattering", "<j.B>"),
    "07_species_and_impurity": ("carbon=", "electron="),
    "08_ambipolar_er": ("[ion]", "all surfaces bracketed: True"),
    "09_monoenergetic": ("D33*", "D11* monotone in nu' at E*=0: True"),
    "10_transport_matrix": ("RHSMode=2", "RHSMode=3", "Onsager check"),
    "11_phi1": ("solver route phi1_newton_krylov", "peak |dn/n|"),
    "12_bootstrap_vs_redl": ("Redl fit", "ratio to Redl"),
    "13_convergence": ("worst change", "converged at 2%: False"),
    "14_gradients": ("jax.grad", "max relative difference"),
    "15_optimization": ("lower)", "central difference"),
    "16_full_trajectories": ("full/DKES", "particle flux DKES"),
    "17_vmex_geometry": ("VMEX: converged", "geometry sha256: "),
    "18_gpu": ("first", "repeat"),
    "19_plotting": ("panels.png", "heatFlux_vm_psiHat"),
}


def test_the_ladder_is_numbered_without_gaps() -> None:
    numbers = [int(path.name[:2]) for path in SCRIPTS]
    assert numbers == list(range(1, len(SCRIPTS) + 1))
    assert set(EXPECTED) == {path.stem for path in SCRIPTS}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_tutorial_runs(script: Path) -> None:
    if script.stem in OPTIONAL:
        pytest.importorskip(OPTIONAL[script.stem])
    out = OUT_DIR / script.stem
    for old in out.glob("*.png"):
        old.unlink()
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, "-c", RUNNER, str(script), SMOKE.get(script.stem, "")],
                          capture_output=True, text=True,
                          cwd=REPO_ROOT, env=env, timeout=900, check=False)  # fmt: skip
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert "=== Summary" in proc.stdout
    for fragment in EXPECTED[script.stem]:
        assert fragment in proc.stdout, f"{script.name}: missing {fragment!r}"
    assert any(path.stat().st_size > 0 for path in out.iterdir()), f"{script.name} wrote nothing"


@pytest.mark.parametrize("script", [*SCRIPTS, *ADVANCED], ids=lambda path: path.stem)
def test_example_obeys_the_style_contract(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for banned in ("argparse", "sys.argv", "os.environ", "__main__"):
        assert banned not in source, f"{script.name}: {banned}"
    assert "main" not in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    docstring = ast.get_docstring(tree) or ""
    for required in ("Run:", "Runtime:"):
        assert required in docstring, f"{script.name}: docstring lacks {required!r}"
    assert "@RT@" not in docstring, f"{script.name}: runtime was never filled in"
    if script.parent == TUTORIALS:
        lines = source.splitlines()
        positions = [next((i for i, line in enumerate(lines) if line.startswith(step)), None) for step in STEPS]
        assert None not in positions, f"{script.name}: missing step comments"
        assert positions == sorted(positions), f"{script.name}: steps out of order"
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert functions <= {"bootstrap_current", "particle_flux"}, functions
