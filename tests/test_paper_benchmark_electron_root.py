"""CI-sized regression gate for the electron-root optimization benchmark.

Runs ``examples/paper_benchmarks/electron_root_optimization.py`` in its CI mode
(reduced Te/Ti ladder, coarse E_r grids, Nxi=24 scan / Nxi=8 differentiable
deck) into temporary output/figure directories, and pins the structural physics
the full benchmark relies on:

- the ``J_r(E_r)`` S-curve has at least one bracketed ambipolar root (>= 1 sign
  change) and every resolved root is classified ion / unstable / electron by the
  documented rule (dJr/dEr sign + E_r sign);
- the Te/Ti scan exhibits the multi-root ion-root -> electron-root transition:
  a stable ion root (E_r < 0) at low Te/Ti and a stable electron root
  (E_r > 0) at high Te/Ti both appear across the scan;
- the differentiable ambipolar Er reproduces its central finite difference
  through the implicit-function-theorem root solve (one frozen AD value at loose
  rtol; the AD-vs-FD relative deviation is tiny);
- the gradient-descent optimization drives the ambipolar Er toward the target
  and the objective strictly decreases;
- the figure + JSON artifacts are produced.

Measured on 2026-07-17 (float64, laptop CPU): CI run ~60 s;
d(Er)/d(electron dT/dr scale) AD = -1.974155e-01 (rel dev vs FD 2.4e-9).
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "paper_benchmarks" / "electron_root_optimization.py"

# Frozen at the CI differentiable resolution (Nxi=8, resolution-independent of
# the coarse scan); see the module docstring for the measurement.
AD_GRAD_FROZEN = -1.974155e-01
AD_GRAD_RTOL = 5.0e-3


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """Run the example once in CI mode into temp dirs; return its globals."""
    out_dir = tmp_path_factory.mktemp("eroot_out")
    fig_dir = tmp_path_factory.mktemp("eroot_fig")
    env = {
        "DKX_EROOT_CI": "1",
        "DKX_EROOT_FORCE": "1",
        "DKX_EROOT_OUT_DIR": str(out_dir),
        "DKX_EROOT_FIG_DIR": str(fig_dir),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return runpy.run_path(str(EXAMPLE), run_name="electron_root_benchmark")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_scurve_has_bracketed_root_and_valid_classification(example) -> None:
    scurve = example["scurve"]
    assert scurve["n_sign_changes"] >= 1
    assert scurve["roots"], "no ambipolar root resolved on the S-curve"
    for r in scurve["roots"]:
        assert r["root_type"] in {"ion", "unstable", "electron"}
        # classification is self-consistent with the documented rule
        if r["slope"] > 0:
            assert r["root_type"] == ("electron" if r["er"] > 0 else "ion")
        else:
            assert r["root_type"] == "unstable"


def test_te_scan_shows_ion_to_electron_transition(example) -> None:
    bif = example["bif"]
    has_ion = False
    has_electron = False
    for entry in bif["entries"]:
        for r in entry["roots"]:
            if r["slope"] > 0 and r["root_type"] == "ion" and r["er"] < 0:
                has_ion = True
            if r["slope"] > 0 and r["root_type"] == "electron" and r["er"] > 0:
                has_electron = True
    assert has_ion, "no stable ion root anywhere on the Te/Ti ladder"
    assert has_electron, "no stable electron root anywhere on the Te/Ti ladder"
    # the electron-root onset is bracketed within the scanned ladder
    onset = bif["electron_root_onset"]
    assert onset is not None
    tes = [e["te_ratio"] for e in bif["entries"]]
    assert min(tes) <= onset <= max(tes)


def test_differentiable_er_matches_finite_difference(example) -> None:
    diff = example["diff"]
    sens = diff["sensitivity"]
    # implicit-function-theorem gradient == central FD (no FD in the AD path)
    assert sens["rel_dev"] < 1.0e-4
    assert np.isfinite(sens["ad"]) and abs(sens["ad"]) > 1e-6
    # one frozen AD value at loose rtol
    assert np.isclose(sens["ad"], AD_GRAD_FROZEN, rtol=AD_GRAD_RTOL), sens["ad"]
    assert diff["base_root_type"] == "ion"


def test_optimization_objective_decreases(example) -> None:
    opt = example["diff"]["optimization"]
    hist = opt["history"]
    objs = [h["objective"] for h in hist]
    assert len(objs) >= 3
    assert objs[-1] < objs[0]
    for earlier, later in zip(objs[:-1], objs[1:]):
        assert later <= earlier, f"objective did not decrease monotonically: {objs}"


def test_artifacts_written(example) -> None:
    png = Path(example["PNG_PATH"])
    js = Path(example["JSON_PATH"])
    assert png.exists() and js.exists()
    record = json.loads(js.read_text())
    for key in ("benchmark", "references", "scurve", "bifurcation",
                "differentiable", "root_selection_rule", "modeling_note", "provenance"):
        assert key in record, key
