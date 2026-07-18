"""Tests for the ``examples/optimization/`` gradient-based example family.

Each example is a simsopt-style single script that optimizes a neoclassical
objective through the canonical dkx kinetic solve with ``jax.grad``.  For
every example this suite asserts, at CI resolution (``DKX_CI=1``, 2-3
iterations):

1. the script runs end to end, the objective decreases, and the advertised
   outputs (a compressed before/after plot + a history JSON) exist;
2. warm starts + GCROT recycling reduce the kinetic Krylov iteration count;
3. the autodiff gradient matches central finite differences at the
   *honestly-achievable* tolerance for that objective (documented per test --
   ~1e-4/1e-3 for the pure-JAX kinetic and ambipolar-root objectives, a few
   percent for the QH full-VMEC-chain kinetic gradient whose FD is limited by
   the reactor-scale host equilibrium solver's ftol noise, not the autodiff);
4. every commented alternative objective (a dict-switch in the script)
   evaluates, so uncommenting it just works.

The shared ``objectives.py`` metric library is also unit-tested directly.

optimize_QH_bootstrap needs the optional companions vmex + booz_xform_jax;
optimize_electron_root and optimize_impurity_screening need only dkx.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

# These build full vmex->Boozer->kinetic optimization chains; too heavy for
# the fast coverage shards. Marked slow and excluded there (run in the full suite).
pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
EX_DIR = REPO_ROOT / "examples" / "optimization"


def _run_example(name: str, env: dict) -> dict:
    """Run an example via runpy at CI resolution; return its module globals."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    if str(EX_DIR) not in sys.path:
        sys.path.insert(0, str(EX_DIR))
    try:
        return runpy.run_path(str(EX_DIR / name), run_name=f"ex_{name}")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ===========================================================================
# optimize_electron_root.py  (needs only dkx)
# ===========================================================================
@pytest.fixture(scope="module")
def er_example() -> dict:
    return _run_example("optimize_electron_root.py",
                        {"DKX_CI": "1", "DKX_ER_MAXITER": "3"})


def test_electron_root_completes_and_decreases(er_example) -> None:
    g = er_example
    assert bool(g["objective_decreased"]), f"{float(g['J0'])} -> {float(g['J_final'])}"
    assert (g["OUT_DIR"] / f"{g['STEM']}.png").exists()
    assert (g["OUT_DIR"] / f"{g['STEM']}_history.json").exists()
    # warm start reduced the tier-2 Krylov iteration count on the E_r scan
    assert int(g["it_warm"]) < int(g["it_cold"])


def test_electron_root_ambipolar_gradient_matches_fd(er_example) -> None:
    """d(E_r objective)/d(shape dof): implicit-function-theorem AD vs central FD."""
    fd = er_example["fd_check"]
    assert np.isfinite(fd["ad"]) and np.isfinite(fd["fd"]) and abs(fd["fd"]) > 0.0
    assert fd["rel"] < 1e-3, f"ambipolar-root gradient rel {fd['rel']:.3e}"


def test_electron_root_alternative_objectives(er_example) -> None:
    """Every ROOT_OBJECTIVES entry (the commented alternatives) evaluates."""
    g = er_example
    er_seed = float(g["er_seed"])
    for name, metric in g["ROOT_OBJECTIVES"].items():
        assert np.isfinite(float(metric(er_seed))), f"root objective {name} not finite"
    # end-to-end: switch the active objective (what uncommenting does) and solve
    original = g["ROOT_OBJECTIVE"]
    try:
        for name in g["ROOT_OBJECTIVES"]:
            g["ROOT_OBJECTIVE"] = name
            value, er = g["_obj_and_er"](g["DOFS0"], er_seed)
            assert np.isfinite(float(value)), f"objective({name}) not finite"
    finally:
        g["ROOT_OBJECTIVE"] = original


# ===========================================================================
# optimize_impurity_screening.py  (needs only dkx)
# ===========================================================================
@pytest.fixture(scope="module")
def imp_example() -> dict:
    return _run_example("optimize_impurity_screening.py",
                        {"DKX_CI": "1", "DKX_IMP_MAXITER": "3"})


def test_impurity_completes_and_decreases(imp_example) -> None:
    g = imp_example
    assert bool(g["objective_decreased"]), f"{float(g['J0'])} -> {float(g['J_final'])}"
    assert (g["OUT_DIR"] / f"{g['STEM']}.png").exists()
    assert (g["OUT_DIR"] / f"{g['STEM']}_history.json").exists()
    assert int(g["it_warm"]) < int(g["it_cold"])          # tier-2 FP warm-start savings
    assert np.isfinite(float(g["screening_coeff"]))        # temperature-screening coefficient


def test_impurity_flux_gradient_matches_fd(imp_example) -> None:
    """d(impurity flux objective)/d(shape dof): AD vs central FD (pure-JAX solve)."""
    fd = imp_example["fd_check"]
    assert np.isfinite(fd["ad"]) and np.isfinite(fd["fd"]) and abs(fd["fd"]) > 0.0
    assert fd["rel"] < 1e-3, f"impurity-flux gradient rel {fd['rel']:.3e}"


def test_impurity_alternative_objectives(imp_example) -> None:
    """Every FLUX_OBJECTIVES entry (the commented alternatives) evaluates."""
    import jax.numpy as jnp

    g = imp_example
    aux0 = g["aux0"]
    mom_like = {"particleFlux_vm_psiHat": jnp.asarray(aux0["particle_flux"]),
                "heatFlux_vm_psiHat": jnp.asarray(aux0["heat_flux"])}
    for name, metric in g["FLUX_OBJECTIVES"].items():
        assert np.isfinite(float(metric(mom_like))), f"flux objective {name} not finite"
    # end-to-end: switching the active objective (what uncommenting does) solves
    original = g["FLUX_OBJECTIVE"]
    try:
        for name in ("impurity_flux_zero", "impurity_heat_flux"):
            g["FLUX_OBJECTIVE"] = name
            value, _aux = g["objective"](g["DOFS0"], g["warm_state"])
            assert np.isfinite(float(value)), f"objective({name}) not finite"
    finally:
        g["FLUX_OBJECTIVE"] = original


# ===========================================================================
# optimize_QH_bootstrap.py  (needs vmex + booz_xform_jax companions)
# ===========================================================================
@pytest.fixture(scope="module")
def qh_example() -> dict:
    pytest.importorskip("vmex")
    pytest.importorskip("vmex.core.boozer_tables")
    pytest.importorskip("booz_xform_jax")
    return _run_example("optimize_QH_bootstrap.py",
                        {"DKX_CI": "1", "DKX_QH_MAXITER": "2",
                         "DKX_QH_FD_CHECK": "1"})


def test_qh_completes_and_decreases(qh_example) -> None:
    g = qh_example
    assert bool(g["objective_decreased"]), f"{float(g['J0'])} -> {float(g['J_final'])}"
    assert (g["OUT_DIR"] / f"{g['STEM']}.png").exists()
    assert (g["OUT_DIR"] / f"{g['STEM']}_history.json").exists()
    assert int(g["it_warm"]) < int(g["it_cold"])          # x0 + recycle warm-start savings


def test_qh_kinetic_bootstrap_gradient_matches_fd(qh_example) -> None:
    """Kinetic <j.B> gradient through the full VMEC chain vs central FD.

    Gated at the script's resolution-dependent FD_GATE (a few percent at CI):
    an eps sweep shows this gradient converging to the autodiff value at the
    reactor-scale host equilibrium ftol-noise floor -- FD-limited, not autodiff.
    """
    g = qh_example
    fd = g["fd_check"]
    assert fd is not None and np.isfinite(fd["ad"]) and np.isfinite(fd["fd"])
    assert fd["rel"] < float(g["FD_GATE"]), f"kinetic-bootstrap gradient rel {fd['rel']:.3e}"


def test_qh_alternative_objectives(qh_example) -> None:
    """Every KINETIC_OBJECTIVES entry (the commented alternatives) evaluates."""
    import jax.numpy as jnp

    g = qh_example
    aux0 = g["aux0"]
    mom_like = {"FSABjHatOverRootFSAB2": jnp.asarray(aux0["jbs"]),
                "particleFlux_vm_psiHat": jnp.asarray(aux0["particle_flux"]),
                "heatFlux_vm_psiHat": jnp.asarray(aux0["heat_flux"])}
    for name, metric in g["KINETIC_OBJECTIVES"].items():
        assert np.isfinite(float(metric(mom_like))), f"kinetic objective {name} not finite"
    # end-to-end: switching the active objective (what uncommenting does) solves
    original = g["KINETIC_OBJECTIVE"]
    try:
        for name in ("particle_flux_l1", "heat_flux_l2"):
            g["KINETIC_OBJECTIVE"] = name
            value, _aux = g["objective"](g["dofs0"], g["warm_state"])
            assert np.isfinite(float(value)), f"objective({name}) not finite"
    finally:
        g["KINETIC_OBJECTIVE"] = original


# ===========================================================================
# objectives.py  (the shared metric library, unit-tested directly)
# ===========================================================================
def test_objectives_metrics_and_qs_residual() -> None:
    import jax.numpy as jnp

    if str(EX_DIR) not in sys.path:
        sys.path.insert(0, str(EX_DIR))
    import objectives as ob

    # every registered figure of merit is finite on a synthetic moment table
    mom = {
        "FSABjHatOverRootFSAB2": jnp.asarray(0.012),
        "particleFlux_vm_psiHat": jnp.asarray([3.0e-6, 8.0e-8, 1.0e-8]),
        "heatFlux_vm_psiHat": jnp.asarray([5.0e-6, 2.0e-7, 3.0e-8]),
    }
    for name, metric in ob.MOMENT_METRICS.items():
        assert np.isfinite(float(metric(mom))), f"{name} not finite"
    assert float(ob.species_particle_flux(mom, 2)) == pytest.approx(1.0e-8)
    assert float(ob.impurity_screening_metric(mom, 2)) == pytest.approx(1.0e-8)
    assert float(ob.root_offset_sq(0.3, 0.0)) == pytest.approx(0.09)

    # QA (helicity 1,0): only n==0 modes are quasisymmetric
    xm = np.array([0, 1, 1, 1])
    xn = np.array([0, 0, 4, -4])  # xn includes nfp=4
    mask_qa = ob.qs_symmetric_mask(xm, xn, nfp=4, helicity_m=1, helicity_n=0)
    assert list(mask_qa) == [True, True, False, False]
    # QH (helicity 1,-1): symmetric <=> n/nfp == -m
    mask_qh = ob.qs_symmetric_mask(xm, xn, nfp=4, helicity_m=1, helicity_n=-1)
    assert list(mask_qh) == [True, False, False, True]
    # residual is zero for a purely symmetric spectrum, positive otherwise
    bmnc = jnp.asarray([1.0, 0.3, 0.2, 0.1])
    assert float(ob.boozer_qs_residual(bmnc, mask_qh)) > 0.0
    assert float(ob.boozer_qs_residual(jnp.asarray([1.0, 0.0, 0.0, 0.2]), mask_qh)) == pytest.approx(0.0)
