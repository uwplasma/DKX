"""CI-sized regression gate for the W7-X ambipolar-Er experimental benchmark.

Runs ``examples/paper_benchmarks/w7x_ambipolar_er.py`` in its CI mode (two flux
surfaces at coarse resolution) into temporary output/figure directories, and
pins the structural physics the full data-validation benchmark relies on:

- each solved surface has at least one bracketed ambipolar root (>= 1 sign
  change on the ``J_r(E_r)`` scan), and every resolved root is classified
  ion / unstable / electron by the documented rule (``dJr/dEr`` sign + ``E_r``
  sign);
- the core CERC surface (``rho = 0.30``) carries a stable electron root
  (``E_r > 0``, ``dJr/dEr > 0``) -- the published discharge's core-electron-root
  signature -- pinned to a frozen coarse-resolution ``E_r`` value at loose rtol;
- the reference-comparison table and the digitized-provenance record are
  emitted, and the figure + JSON artifacts are written.

Measured on 2026-07-17 (float64, laptop CPU): CI run ~95 s; coarse-grid
electron-root ``E_r`` at ``rho = 0.30`` = +6.87 kV/m (the production grid gives
~+10.4 kV/m; see the example's convergence note).
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

# The module-scoped ``example`` fixture runs several multi-solve surface scans;
# under coverage instrumentation that peaks a CI shard's memory, so keep it out
# of the "not slow" coverage shards (same policy as the electron-root and
# flagship integration gates). It still runs in the slow lane and locally.
pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "paper_benchmarks" / "w7x_ambipolar_er.py"

# Frozen at the CI (coarse) resolution; see the module docstring.
CI_CORE_ELECTRON_ER = 6.87
CI_CORE_ELECTRON_RTOL = 0.05


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """Run the example once in CI mode into temp dirs; return its globals."""
    out_dir = tmp_path_factory.mktemp("w7xamb_out")
    fig_dir = tmp_path_factory.mktemp("w7xamb_fig")
    env = {
        "DKX_W7XAMB_CI": "1",
        "DKX_W7XAMB_FORCE": "1",
        "DKX_W7XAMB_OUT_DIR": str(out_dir),
        "DKX_W7XAMB_FIG_DIR": str(fig_dir),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return runpy.run_path(str(EXAMPLE), run_name="w7x_ambipolar_benchmark")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_each_surface_has_bracketed_root_and_valid_classification(example) -> None:
    surfaces = example["surfaces"]
    assert len(surfaces) >= 2
    for surf in surfaces:
        assert surf["n_sign_changes"] >= 1, surf["rho"]
        assert surf["roots"], f"no ambipolar root at rho={surf['rho']}"
        for r in surf["roots"]:
            assert r["root_type"] in {"ion", "unstable", "electron"}
            # classification is self-consistent with the orientation-aware rule
            # (stable iff slope*sign(psiAHat) > 0; psiAHat<0 on this equilibrium)
            assert r["stable"] == (r["slope"] * r["orient"] > 0)
            if r["stable"]:
                assert r["root_type"] == ("electron" if r["er"] > 0 else "ion")
            else:
                assert r["root_type"] == "unstable"


def test_core_surface_carries_stable_electron_root(example) -> None:
    surfaces = {round(s["rho"], 2): s for s in example["surfaces"]}
    core = surfaces[0.30]
    electron = [r for r in core["roots"] if r["root_type"] == "electron" and r["stable"]]
    assert electron, "no stable electron root at the core CERC surface rho=0.30"
    er = max(r["er"] for r in electron)
    assert er > 0.0
    # one frozen coarse-resolution value at loose rtol
    assert np.isclose(er, CI_CORE_ELECTRON_ER, rtol=CI_CORE_ELECTRON_RTOL), er


def test_reference_comparison_and_provenance_present(example) -> None:
    record = example["record"]
    assert record["case_type"].startswith("REAL published discharge")
    assert "Pablant" in record["citation"]
    assert record["reference_comparison"], "empty reference comparison"
    for c in record["reference_comparison"]:
        assert set(c) >= {"rho", "dkx_er", "dkx_root_type", "ref_nc_er", "difference"}
    # digitized provenance is explicitly labelled approximate
    assert "digitized" in record["profiles"]["source"]
    assert "approximate" in record["profiles"]["source"]


def test_artifacts_written(example) -> None:
    png = Path(example["PNG_PATH"])
    js = Path(example["JSON_PATH"])
    assert png.exists() and js.exists()
    rec = json.loads(js.read_text())
    for key in (
        "benchmark", "case_type", "citation", "references", "equilibrium",
        "model", "profiles", "reference_er", "surfaces", "reference_comparison",
        "convergence", "root_selection_rule", "provenance",
    ):
        assert key in rec, key
