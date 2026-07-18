"""CI-sized regression gate for the kinetic-in-the-loop bootstrap benchmark.

Runs ``examples/paper_benchmarks/bootstrap_consistency_kinetic_loop.py`` in
its CI mode (truncated boundary max_mode = 2, ns = 7, two kinetic surfaces at
toy resolution, three Picard iterations, single-surface gradient quadrature)
into temporary output directories, and pins the structural physics the full
benchmark relies on:

- the Picard current-profile change ``delta`` strictly DECREASES between
  iterations (the damped fixed-point map contracts), starting from the
  zero-current ``delta_0 = 1``;
- one frozen value: the applied total current after the first refit
  (``history[1].curtor``) at the toy resolution, so silent regressions in
  the kinetic surface solve, the SI conversion, or the parallel-current
  identity inversion surface here;
- the bootstrap current has the QA-standard sign (negative curtor for this
  deck's orientation) and the kinetic <J.B> profile is finite and negative
  across the recorded surfaces;
- the Redl contrast, split resolution probes, gradient hook (finite,
  nonzero derivative through the differentiable equilibrium -> Boozer ->
  kinetic chain), and the figure + JSON artifacts are all produced.

Measured on 2026-07-17 (float64, laptop CPU): full CI run ~30 s;
``history[1].curtor = -199850.025 A``; deltas 1.0 -> 0.4877 -> 0.2349.

Requires the optional companions vmex (core API) and booz_xform_jax;
skipped otherwise.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vmex")
pytest.importorskip("vmex.core.bootstrap")
pytest.importorskip("booz_xform_jax")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "paper_benchmarks" / "bootstrap_consistency_kinetic_loop.py"

# Frozen at the CI resolution (see module docstring for the measurement).
CURTOR_ITERATION1_FROZEN = -199850.025
CURTOR_RTOL = 5e-3


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """Run the example once in CI mode into temp dirs; return its globals."""
    out_dir = tmp_path_factory.mktemp("bootloop_out")
    fig_dir = tmp_path_factory.mktemp("bootloop_fig")
    env = {
        "DKX_BOOT_LOOP_CI": "1",
        "DKX_BOOT_LOOP_OUT_DIR": str(out_dir),
        "DKX_BOOT_LOOP_FIG_DIR": str(fig_dir),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    os.environ.pop("DKX_BOOT_LOOP_MAX_NEW_STAGES", None)
    try:
        return runpy.run_path(str(EXAMPLE), run_name="bootstrap_loop_benchmark")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_picard_profile_change_decreases(example) -> None:
    history = example["history"]
    assert len(history) >= 3
    deltas = [float(row["delta"]) for row in history]
    assert deltas[0] == pytest.approx(1.0)
    for earlier, later in zip(deltas[:-1], deltas[1:]):
        assert later < earlier, f"delta did not decrease: {deltas}"


def test_frozen_first_refit_current(example) -> None:
    history = example["history"]
    curtor1 = float(history[1]["curtor"])
    assert np.isclose(curtor1, CURTOR_ITERATION1_FROZEN, rtol=CURTOR_RTOL), curtor1


def test_bootstrap_sign_and_profiles_finite(example) -> None:
    history = example["history"]
    for row in history:
        jdotb = np.asarray(row["jdotb_kinetic_si"], dtype=float)
        assert np.all(np.isfinite(jdotb))
        assert np.all(jdotb < 0.0), jdotb  # QA bootstrap direction for this deck
    assert float(history[-1]["curtor"]) < 0.0
    # the applied current raises iota (the genuine equilibrium feedback)
    assert float(history[-1]["iota_mid"]) > float(history[0]["iota_mid"])


def test_redl_contrast_and_probes_recorded(example) -> None:
    jdotb_redl = np.asarray(example["jdotb_redl"], dtype=float)
    assert jdotb_redl.shape == np.asarray(example["S_KIN"]).shape
    assert np.all(np.isfinite(jdotb_redl))
    assert np.all(jdotb_redl < 0.0)
    probe = example["probe"]
    for name in ("angular", "velocity"):
        assert np.isfinite(probe[name]["rel_dev"])
        assert probe[name]["rel_dev"] >= 0.0


def test_gradient_hook_finite_nonzero(example) -> None:
    gradient = example["gradient"]
    assert np.isfinite(gradient["value_A"]) and gradient["value_A"] > 0.0
    assert np.isfinite(gradient["grad_A_per_m"])
    assert gradient["grad_A_per_m"] != 0.0


def test_artifacts_written(example) -> None:
    json_path = Path(example["JSON_PATH"])
    png_path = Path(example["PNG_PATH"])
    assert json_path.exists() and png_path.exists()
    record = json.loads(json_path.read_text())
    for key in ("benchmark", "references", "picard", "redl_contrast",
                "kinetic_resolution_probe", "gradient_hook", "limitation",
                "claim_boundary", "finding"):
        assert key in record, key
    assert record["picard"]["history"][0]["delta"] == pytest.approx(1.0)
