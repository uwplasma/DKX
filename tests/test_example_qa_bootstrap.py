"""Tests for the flagship example ``examples/optimize_QA_bootstrap.py``.

Covered here:

1. the example runs end to end at toy resolution (``SFINCS_JAX_CI=1``) and
   the objective decreases;
2. autodiff accuracy, documented honestly at two levels:
   - the Boozer-spectrum -> kinetic <j.B> segment (pure JAX + implicit
     linear solve): d(objective)/d(spectrum scale) vs central finite
     differences at rtol 1e-4 (measured ~1e-6);
   - the full boundary-dof chain: the example's own in-script FD check on
     the dominant dof must pass its (resolution-dependent) gate.  The
     end-to-end FD comparison is limited by the *finite-difference* noise of
     the iterative host equilibrium solve (ftol-termination noise ~5e-3 on
     small gradient components), not by the autodiff path — the same floor
     the vmec_jax implicit-adjoint tests document;
3. every entry of ``KINETIC_OBJECTIVES`` (the commented alternative
   objectives in the example) evaluates to a finite scalar on the solved
   moments and participates in a full traced objective evaluation, so
   uncommenting the alternative lines just works;
4. the example's traceable VMEC->Boozer bridge reproduces the host wout
   engine + classic booz_xform run on the same surface.

Requires the optional companions vmec_jax (new core API) and booz_xform_jax;
skipped otherwise.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

# Dependency-based skips only: the example needs the optional companion
# packages vmec_jax (new core API with core.boozer_tables) and booz_xform_jax.
# The starting VMEC input ships with vmec_jax; the example resolves it from
# the installed package (override with SFINCS_JAX_QA_VMEC_INPUT).
pytest.importorskip("vmec_jax")
pytest.importorskip("vmec_jax.core.boozer_tables")
pytest.importorskip("booz_xform_jax")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "optimize_QA_bootstrap.py"


@pytest.fixture(scope="module")
def example():
    """Run the example once in-process at CI resolution; return its globals."""
    env = {
        "SFINCS_JAX_CI": "1",
        "SFINCS_JAX_QA_MAXITER": "2",
        "SFINCS_JAX_QA_FD_CHECK": "1",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return runpy.run_path(str(EXAMPLE), run_name="qa_bootstrap_example")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_example_completes_and_objective_decreases(example) -> None:
    g = example
    assert bool(g["objective_decreased"]), (
        f"objective did not decrease: {float(g['J0'])} -> {float(g['J_final'])}"
    )
    # advertised outputs exist
    out_dir = g["OUT_DIR"]
    assert (out_dir / f"{g['STEM']}.png").exists()
    assert (out_dir / f"{g['STEM']}_history.json").exists()
    assert g["wout_paths"]["initial"].exists() and g["wout_paths"]["final"].exists()
    # warm start actually saved kinetic iterations
    assert int(g["it_warm"]) < int(g["it_cold"])


def test_full_chain_fd_check_passed_at_documented_tolerance(example) -> None:
    fd = example["fd_check"]
    assert fd is not None
    assert np.isfinite(fd["fd"]) and np.isfinite(fd["ad"])
    # dominant-dof agreement at the example's resolution-dependent gate
    # (1.5e-2 at default resolution, measured ~5e-3; 5e-2 at CI's looser VMEC
    # ftol — the host solver's FD termination noise dominates, not autodiff)
    assert fd["rel"] < float(example["FD_GATE"])


def test_kinetic_segment_gradient_matches_fd_at_1e4(example) -> None:
    """d(kinetic objective)/d(Boozer spectrum scale): AD vs central FD."""
    import jax
    import jax.numpy as jnp

    g = example
    aux0 = g["aux0"]
    bmnc_b = jnp.asarray(aux0["bmnc_b"])
    non_00 = jnp.asarray((g["BOOZ_XM"] != 0) | (g["BOOZ_XN"] != 0))

    def seg(scale):
        booz = {
            "bmnc_b": jnp.where(non_00, bmnc_b * scale, bmnc_b)[None, :],
            "ixm_b": jnp.asarray(g["BOOZ_XM"]),
            "ixn_b": jnp.asarray(g["BOOZ_XN"]),
            "iota_b": jnp.asarray(aux0["booz_iota"])[None],
            "bvco_b": jnp.asarray(aux0["booz_G"])[None],
            "buco_b": jnp.asarray(aux0["booz_I"])[None],
        }
        mom, _ = g["kinetic_moments"](booz)
        return mom["FSABjHatOverRootFSAB2"] ** 2

    ad = float(jax.grad(seg)(1.0))
    eps = 1e-6
    fd = (float(seg(1.0 + eps)) - float(seg(1.0 - eps))) / (2.0 * eps)
    rel = abs(ad - fd) / max(abs(fd), 1e-300)
    assert np.isfinite(ad) and np.isfinite(fd) and abs(fd) > 0.0
    assert rel < 1e-4, f"kinetic-segment gradient rel diff {rel:.3e}"


def test_alternative_kinetic_objectives_evaluate(example) -> None:
    """All commented alternative objectives work when selected."""
    import jax.numpy as jnp

    g = example
    aux0 = g["aux0"]
    # metric-level: apply every registered figure of merit to solved moments
    mom_like = {
        "FSABjHatOverRootFSAB2": jnp.asarray(aux0["jbs"]),
        "particleFlux_vm_psiHat": jnp.asarray(aux0["particle_flux"]),
        "heatFlux_vm_psiHat": jnp.asarray(aux0["heat_flux"]),
    }
    for name, metric in g["KINETIC_OBJECTIVES"].items():
        value = float(metric(mom_like))
        assert np.isfinite(value), f"objective {name} not finite"
        assert value >= 0.0

    # end-to-end: switch the selected objective (what uncommenting does) and
    # run one full traced evaluation
    original = g["KINETIC_OBJECTIVE"]
    try:
        for name in g["KINETIC_OBJECTIVES"]:
            if name == original:
                continue
            g["KINETIC_OBJECTIVE"] = name
            value, aux = g["objective"](g["dofs0"], g["warm_state"])
            assert np.isfinite(float(value)), f"objective({name}) not finite"
    finally:
        g["KINETIC_OBJECTIVE"] = original


def test_boozer_bridge_matches_host_wout_engine(example) -> None:
    """The traceable VMEC->Boozer tables reproduce the host reference path."""
    from booz_xform_jax import Booz_xform
    from vmec_jax.core import solver as vmec_solver
    from vmec_jax.core.wout import wout_from_state

    g = example
    params0, inp0 = g["params0"], g["inp0"]
    j = int(g["S_KINETIC_ROW"])

    # The example's cfg hot-restarts every solve for optimizer speed; a parity
    # check wants a *fresh cold* solve of this exact boundary, since a warm
    # restart from the optimizer loop's last (far-away) boundary leaves the
    # equilibrium a ftol-noise level (~1e-4..1e-3) off the reference host solve.
    cfg = g["vmec_implicit"].make_config(
        inp0, adjoint_tol=1e-6, adjoint_maxiter=40)
    state = g["vmec_implicit"].solve_implicit(params0, cfg)
    rt = g["vmec_implicit"].runtime_from_params(params0, cfg)
    tabs = g["boozer_input_tables"](state, rt, j)

    res = vmec_solver.solve(inp0, cfg.resolution, ftol=cfg.ftol,
                            max_iterations=cfg.max_iterations, mode="cli")
    wout = wout_from_state(inp=inp0, state=res.state, fsqr=res.fsqr,
                           fsqz=res.fsqz, fsql=res.fsql)

    xm_nyq = np.asarray(wout.xm_nyq, dtype=int)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=int)

    def max_rel(name, ref_row, xm_ref, xn_ref):
        mine, ref = [], []
        for i, (m, n) in enumerate(zip(tabs["xm"], tabs["xn"])):
            hits = np.where((xm_ref == m) & (xn_ref == n))[0]
            if hits.size:
                mine.append(float(np.asarray(tabs[name])[i]))
                ref.append(float(ref_row[hits[0]]))
        mine, ref = np.asarray(mine), np.asarray(ref)
        return np.max(np.abs(mine - ref)) / max(np.max(np.abs(ref)), 1e-30)

    # |B| spectrum: identical quadrature -> near machine precision
    assert max_rel("bmnc", np.asarray(wout.bmnc)[j], xm_nyq, xn_nyq) < 1e-10
    # covariant fields: same half-mesh data, wout-engine grid differences only
    assert max_rel("bsubumnc", np.asarray(wout.bsubumnc)[j], xm_nyq, xn_nyq) < 5e-3
    assert max_rel("bsubvmnc", np.asarray(wout.bsubvmnc)[j], xm_nyq, xn_nyq) < 5e-3
    # lambda (reconstructed from angular derivatives, wout normalization;
    # dominated by the half-mesh finite-difference level at CI's tiny ns)
    assert max_rel("lmns", np.asarray(wout.lmns)[j],
                   np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int)) < 2e-2
    # iota at the surface
    assert abs(float(tabs["iota"]) - float(np.asarray(wout.iotas)[j])) < 1e-10

    # Boozer |B| spectrum vs the classic host transform on the same surface
    bx = Booz_xform(mboz=int(g["MBOZ"]), nboz=int(g["NBOZ"]))
    bx.read_wout_data(wout)
    bx.compute_surfs = [j - 1]  # 0-based half-mesh surface index
    bx.run(jit=False)
    bmnc_b_ref = np.asarray(bx.bmnc_b)[:, 0]
    xm_b_ref = np.asarray(bx.xm_b)
    xn_b_ref = np.asarray(bx.xn_b)

    bmnc_b_mine = np.asarray(g["aux0"]["bmnc_b"])
    mine, ref = [], []
    for i, (m, n) in enumerate(zip(g["BOOZ_XM"], g["BOOZ_XN"])):
        hits = np.where((xm_b_ref == m) & (xn_b_ref == n))[0]
        if hits.size:
            mine.append(bmnc_b_mine[i])
            ref.append(bmnc_b_ref[hits[0]])
    mine, ref = np.asarray(mine), np.asarray(ref)
    rel = np.max(np.abs(mine - ref)) / np.max(np.abs(ref))
    assert rel < 1e-3, f"Boozer |B| spectrum mismatch vs host path: {rel:.3e}"
