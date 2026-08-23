"""The three ``*_optimization_bootstrap_dkx.py`` VMEX pairings.

Each is its ``vmex/examples/optimization`` counterpart with one objective term
swapped: the Redl analytic bootstrap current replaced by the kinetic one DKX
computes on the same equilibrium.  Running them to convergence is hours under a
finite-difference Jacobian, so this suite checks the parts that break silently:
the substitution is actually present, the term is wired the way
``vmex.core.optimize`` requires, and the surrounding script still parses and
matches its upstream template.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EX_DIR = REPO_ROOT / "examples" / "optimization"
CASES = ("QA", "QH", "QI")


def _source(case: str) -> str:
    path = EX_DIR / f"{case}_optimization_bootstrap_dkx.py"
    assert path.is_file(), path
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES)
def test_script_parses(case: str) -> None:
    ast.parse(_source(case))


@pytest.mark.parametrize("case", CASES)
def test_the_kinetic_term_replaces_the_redl_one_in_the_objective(case: str) -> None:
    """The whole point: minimize the kinetic current, keep Redl for reporting."""
    source = _source(case)
    assert "from dkx.bootstrap import KineticBootstrapCurrent" in source
    assert "kinetic = KineticBootstrapCurrent(profiles" in source
    # Target 0 with a weight -- the tuple form vmex.optimize consumes.
    assert "(kinetic, 0.0, KINETIC_WEIGHT)" in source
    # Redl stays, but only in the reporter, never in objective_function_terms.
    assert '("f_boot_redl", bootstrap.total' in source
    assert "(bootstrap, 0.0," not in source


@pytest.mark.parametrize("case", CASES)
def test_the_problem_uses_finite_differences(case: str) -> None:
    """DKX is a host code: the implicit lane has no traceable term to use.

    Without this the problem raises "objective term ... is not
    implicit-differentiable" the moment it is built.
    """
    source = _source(case)
    assert 'derivative_method="finite_difference"' in source


@pytest.mark.parametrize("case", CASES)
def test_the_seed_boundary_is_resolvable_or_explains_itself(case: str) -> None:
    """vmex from a wheel ships no examples/data; the script must say so."""
    source = _source(case)
    assert "DKX_VMEX_ROOT" in source
    assert "raise SystemExit" in source


@pytest.mark.parametrize(("case", "helicity"), [("QA", 0), ("QH", -1), ("QI", 0)])
def test_the_redl_helicity_matches_the_configuration(case: str, helicity: int) -> None:
    """QA and QI carry no helical symmetry for the isomorphism to shift; QH does."""
    assert f"helicity_n={helicity}, surfaces=SURFACES" in _source(case)


def test_qi_says_why_the_substitution_matters_most_there() -> None:
    """Redl is a fit to quasisymmetric calculations and QI is not quasisymmetric."""
    source = _source("QI")
    assert "quasi-isodynamic field is not" in source or "not\nquasisymmetric" in source
    assert "ConstructedQIResidual" in source


@pytest.mark.parametrize("case", CASES)
def test_the_scripts_stay_close_to_their_vmex_templates(case: str) -> None:
    """"One term swapped" is the claim; a diverged script quietly breaks it."""
    source = _source(case)
    for marker in ("self_consistent_bootstrap", "KineticProfiles", "OptimizationMonitor",
                   "least_squares(problem.residual", "opt.VmecProblem.from_tuples",
                   "monitor.save", "vj.write_wout"):  # fmt: skip
        assert marker in source, f"{case}: lost the vmex workflow marker {marker!r}"


def test_the_objective_term_construction_matches_the_scripts(tmp_path) -> None:
    """Build the term exactly as the scripts do, with no vmex import needed."""
    import numpy as np

    from dkx.bootstrap import KineticBootstrapCurrent

    class Profiles:
        ne_coeffs = 3.0e20 * np.array([1.0, 0, 0, 0, 0, -1.0])
        Te_coeffs = 15.0e3 * np.array([1.0, -1.0])
        Ti_coeffs = 15.0e3 * np.array([1.0, -1.0])

    surfaces = np.array([0.25, 0.5, 0.75])
    term = KineticBootstrapCurrent(Profiles(), surfaces=surfaces,
                                   resolution=dict(n_theta=13, n_zeta=19, n_xi=13, n_x=4))
    assert term.name == "j_boot_dkx"
    assert callable(term) and callable(term.total)
    # vmex's _call_term inspects the signature: one positional argument means
    # "hand it the Equilibrium", two would mean "(state, runtime)".
    import inspect

    positional = [p for p in inspect.signature(term).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]  # fmt: skip
    assert len(positional) == 1, positional
    # And _term_name reads .name off the instance for the monitor's column.
    assert getattr(term, "name") == "j_boot_dkx"


@pytest.mark.slow  # a Picard seed, a finite-difference stage, and a final solve
def test_the_qa_pairing_runs_end_to_end(tmp_path) -> None:
    """Parsing is not running.  This is the only check that the chain closes.

    Picard seed -> VmecProblem with a finite-difference Jacobian -> one DKX
    kinetic solve per residual evaluation -> least_squares -> final solve ->
    plots.  At DKX_EXAMPLES_CI=1 the ladder is one stage of two evaluations on a
    single surface at reduced resolution, which is a smoke test of the wiring
    and emphatically not a converged optimization: with 12 dofs and max_nfev=2
    the step is dominated by the quasisymmetry and beta terms, so the kinetic
    current is not expected to fall here.  Measured once at that setting: cost
    2.2060 -> 0.8593, j_boot_dkx 5.71e-04 -> 7.23e-04.
    """
    pytest.importorskip("vmex")
    vmex_root = Path(pytest.importorskip("vmex").__file__).resolve().parents[1]
    if not (vmex_root / "examples" / "data").is_dir():
        pytest.skip("vmex installed without its examples/data seed boundaries")

    script = EX_DIR / "QA_optimization_bootstrap_dkx.py"
    env = dict(os.environ, DKX_EXAMPLES_CI="1", DKX_VMEX_ROOT=str(vmex_root))
    done = subprocess.run([sys.executable, "-u", str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=3600)  # fmt: skip
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    out = done.stdout
    assert "[self-consistent seed]" in out and "[final]" in out
    # Both bootstrap estimates are reported side by side; that is the point of
    # keeping the Redl term in the reporter.
    assert "j_boot_dkx" in out and "f_boot_redl" in out
    costs = [float(line.split()[2]) for line in out.splitlines()
             if line.strip().startswith(("0 ", "1 ")) and len(line.split()) >= 3]  # fmt: skip
    assert costs and costs[-1] < costs[0], f"cost did not decrease: {costs}"
    for name in ("input.QA_bootstrap_dkx_optimized", "wout_QA_bootstrap_dkx_optimized.nc",
                 "QA_bootstrap_dkx_current.png"):  # fmt: skip
        assert (tmp_path / name).exists(), name
