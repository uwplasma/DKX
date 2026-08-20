"""The `dkx --plot` panels and the representative run.

Two properties matter more than pixel layout. The panels must render from
*Fortran* SFINCS output as well as DKX's -- both are ``sfincsOutput.h5`` in the
same layout, so a reader that only works on one of them is a bug. And a panel
that cannot be drawn must say so rather than leave an empty axis, which a reader
would take for zero.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from dkx.representative import (  # noqa: E402
    DEFAULT_RESOLUTION,
    _panel_bootstrap,
    _panel_modB,
    _panel_monoenergetic,
    plot_representative,
)

REF = Path(__file__).parent / "ref"


def test_the_default_resolution_is_the_measured_one():
    """Pinned to the convergence scan, not to taste.

    Nxi is converged to 1e-07 at the lowest value tested while Nzeta is the
    expensive axis, so a "reasonable-looking" bump to Nxi would cost time and
    buy nothing.  The scan is in the module docstring.
    """
    assert DEFAULT_RESOLUTION == {"n_theta": 25, "n_zeta": 41, "n_xi": 20}


def test_modB_labels_the_axis_that_actually_varies():
    """BHat is stored (zeta, theta); an axisymmetric run must not say zeta.

    The mismatch guard used to overwrite both coordinates with arange before the
    label was chosen, so every axisymmetric figure was labelled with the wrong
    angle -- correct-looking and wrong.
    """
    data = {"BHat": np.linspace(0.9, 1.1, 21).reshape(1, 21),
            "theta": np.linspace(0, 6.28, 21), "zeta": np.zeros(1)}  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_modB(ax, data)
        assert "theta" in ax.get_xlabel()
    finally:
        plt.close(fig)


def test_modB_handles_a_real_two_dimensional_surface():
    data = {"BHat": np.outer(np.linspace(0.9, 1.1, 8), np.linspace(1.0, 1.05, 6)),
            "theta": np.linspace(0, 6.28, 8), "zeta": np.linspace(0, 1.2, 6)}  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_modB(ax, data)
    finally:
        plt.close(fig)


def test_a_panel_with_no_data_reports_it_rather_than_drawing_nothing():
    fig, ax = plt.subplots()
    try:
        assert _panel_bootstrap(ax, {}) is False
    finally:
        plt.close(fig)


def test_the_monoenergetic_panel_needs_a_scan():
    fig, axes = plt.subplots(1, 3)
    try:
        assert _panel_monoenergetic(*axes, []) is False
        records = [{"nu_prime": 1e-2, "e_star": 0.0, "D11": -6.7e-4, "D31": -1e-4, "D33": 0.8},
                   {"nu_prime": 1e-1, "e_star": 0.0, "D11": -1.1e-3, "D31": -2e-4, "D33": 0.85}]  # fmt: skip
        assert _panel_monoenergetic(*axes, records) is True
    finally:
        plt.close(fig)


def test_the_figure_renders_with_nothing_at_all(tmp_path):
    """An empty input must still produce a figure that explains itself."""
    out = plot_representative(tmp_path / "empty.png")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("deck", ["tokamak_1species_FPCollisions_noEr"])
def test_panels_render_from_a_real_output_file(tmp_path, deck):
    from dkx.representative import plot_output_file  # noqa: PLC0415
    from dkx.run import run_profile  # noqa: PLC0415

    src = Path(__file__).resolve().parents[1] / "examples" / "sfincs_examples" / deck
    if not (src / "input.namelist").exists():
        pytest.skip(f"{deck} not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_profile(src / "input.namelist", out_path=tmp_path / "out.h5", emit=None)
    out = plot_output_file(tmp_path / "out.h5", tmp_path / "panels.png")
    assert out.exists() and out.stat().st_size > 0


def test_ambipolar_roots_are_bracketed_not_solved():
    """The panel reports sign changes, so a reader sees how MANY roots there are.

    A Brent solve from one guess returns a single root and hides whether the
    device is in the ion-root regime or the ion/unstable/electron triplet --
    which is the physics the panel exists to show.
    """
    from dkx.representative import _ambipolar_roots

    single = [{"er": -4.0, "J_r": 2.4e-08}, {"er": -2.0, "J_r": -5.4e-08}]
    assert len(_ambipolar_roots(single)) == 1
    assert -4.0 < _ambipolar_roots(single)[0] < -2.0

    triplet = [{"er": -6.0, "J_r": 1.0}, {"er": -3.0, "J_r": -1.0},
               {"er": 0.0, "J_r": 1.0}, {"er": 3.0, "J_r": -1.0}]  # fmt: skip
    assert len(_ambipolar_roots(triplet)) == 3

    assert _ambipolar_roots([{"er": -1.0, "J_r": 1.0}, {"er": 1.0, "J_r": 2.0}]) == []


def test_a_nan_current_does_not_invent_a_root():
    from dkx.representative import _ambipolar_roots

    assert _ambipolar_roots([{"er": -1.0, "J_r": float("nan")},
                             {"er": 1.0, "J_r": -1.0}]) == []  # fmt: skip


def test_the_figure_grows_a_row_for_the_ambipolarity_panel(tmp_path):
    from dkx.representative import plot_representative

    recs = [{"er": -4.0, "J_r": 2.4e-08}, {"er": -2.0, "J_r": -5.4e-08}]
    out = plot_representative(tmp_path / "full.png", ambipolar=recs)
    assert out.exists() and out.stat().st_size > 0
