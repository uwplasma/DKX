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


# ---------------------------------------------------------------------------
# A binary file handed to the namelist reader must say what it is
# ---------------------------------------------------------------------------
def test_a_netcdf_handed_to_the_namelist_reader_is_named(tmp_path):
    """A wout is the other file a DKX user has in the directory.

    Without this, `read_text` dies with `UnicodeDecodeError: 'utf-8' codec can't
    decode byte 0xc8 in position 55` -- which says nothing about what was wrong
    or what to do, and sent a real user debugging their equilibrium instead of
    their command.
    """
    from dkx.namelist import read_sfincs_input

    fake = tmp_path / "wout_thing.nc"
    fake.write_bytes(b"CDF\x02" + b"\x00" * 64 + b"\xc8rubbish")
    with pytest.raises(ValueError) as excinfo:
        read_sfincs_input(fake)
    message = str(excinfo.value)
    assert "NetCDF" in message
    assert "not a SFINCS input namelist" in message
    assert "geometryScheme = 5" in message   # tells the user what to do instead


@pytest.mark.parametrize(
    ("signature", "expected"),
    [(b"\x89HDF\r\n\x1a\n", "HDF5"), (b"\x93NUMPY", "NumPy"), (b"PK\x03\x04", "zip")],
)
def test_other_binary_formats_are_named_too(tmp_path, signature, expected):
    from dkx.namelist import read_sfincs_input

    blob = tmp_path / "thing.bin"
    blob.write_bytes(signature + b"\x00" * 40 + b"\xff\xfe")
    with pytest.raises(ValueError, match=expected):
        read_sfincs_input(blob)


def test_an_unrecognised_binary_still_fails_clearly(tmp_path):
    """No signature match must not mean falling back to the opaque decode error."""
    from dkx.namelist import read_sfincs_input

    blob = tmp_path / "mystery.dat"
    blob.write_bytes(b"\xc8\xc9\xca\xcb" * 20)
    with pytest.raises(ValueError, match="not a text file"):
        read_sfincs_input(blob)


def test_a_real_namelist_still_parses(tmp_path):
    from dkx.namelist import read_sfincs_input

    deck = tmp_path / "input.namelist"
    deck.write_text("&physicsParameters\n  nu_n = 0.01\n/\n")
    assert read_sfincs_input(deck) is not None


# ---------------------------------------------------------------------------
# Radial profiles, named species, and the output file
# ---------------------------------------------------------------------------
def test_the_flux_panel_names_species_rather_than_numbering_them():
    """"species 0" and "species 1" tell a reader nothing about which is which."""
    from dkx.representative import _species_labels

    assert _species_labels(2, [1.0, -1.0]) == ["ions", "electrons"]
    assert _species_labels(2, [-1.0, 1.0]) == ["electrons", "ions"]
    assert _species_labels(2) == ["ions", "electrons"]


def test_coincident_ambipolar_fluxes_are_drawn_once_and_explained():
    """At the root sum_s Z_s Gamma_s = 0, so a Z=+-1 pair has EQUAL fluxes.

    Two identical lines read as a rendering fault.  One labelled line says what
    is actually true.
    """
    from dkx.representative import _panel_radial_fluxes

    shared = [5.0e-9, 4.0e-9, 3.0e-9]
    profiles = [
        {"r": r, "particle_flux": [g, g], "heat_flux": [7.0e-8, 1.0e-8]}
        for r, g in zip((0.3, 0.5, 0.7), shared)
    ]
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_fluxes(ax, profiles)
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("ambipolar" in str(x) for x in labels), labels
        assert sum("Gamma" in str(x) or "\\Gamma" in str(x) for x in labels) == 1
    finally:
        plt.close(fig)


def test_distinct_fluxes_are_drawn_separately():
    """The collapse must not hide a genuine difference between species."""
    from dkx.representative import _panel_radial_fluxes

    profiles = [
        {"r": r, "particle_flux": [5.0e-9, 9.0e-9], "heat_flux": [7.0e-8, 1.0e-8]}
        for r in (0.3, 0.5)
    ]
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_fluxes(ax, profiles)
        labels = [str(line.get_label()) for line in ax.get_lines()]
        assert sum("Gamma" in x for x in labels) == 2, labels
    finally:
        plt.close(fig)


def test_the_bootstrap_panel_is_a_radial_profile_with_the_field_beside_it():
    from dkx.representative import _panel_radial_bootstrap

    profiles = [{"r": r, "bootstrap": -6e-3 + r * 1e-3, "er_ambipolar": -1.4 + r}
                for r in (0.25, 0.5, 0.75)]  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_bootstrap(ax, profiles)
        assert "r/a" in ax.get_xlabel()
        assert len(ax.figure.axes) == 2, "ambipolar Er needs its own twin axis"
    finally:
        plt.close(fig)


def test_a_run_always_leaves_the_numbers_behind(tmp_path):
    """A PNG cannot be re-plotted, re-scaled, or checked against another code."""
    from dkx.representative import write_representative_output

    scan = [{"nu_prime": 1e-2, "e_star": 0.0, "D11": -1e-3, "D31": -1e-4, "D33": 0.9}]
    profiles = [{"r": 0.5, "er_ambipolar": -1.1, "bootstrap": -6e-3,
                 "particle_flux": [5e-9, 5e-9], "heat_flux": [7e-8, 1e-8]}]  # fmt: skip
    out = write_representative_output(tmp_path / "run.h5", scan=scan, profiles=profiles)
    assert out.exists() and out.stat().st_size > 0
