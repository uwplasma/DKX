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


def test_nxi_is_at_least_nzeta():
    """The pitch-angle grid must not be the coarsest axis.

    An earlier default of 25/41/20 came from a convergence scan run at a single
    mid collisionality, where Nxi appeared converged to 1e-07 at the lowest
    value tested.  That is exactly where the scan is blind: at LOW collisionality
    the pitch-angle resolution is what limits the answer, and the 1/nu branch is
    the part of a monoenergetic figure a reader cares about.  A one-collisionality
    scan cannot see that, so the constraint is pinned as a relation rather than a
    number.
    """
    assert DEFAULT_RESOLUTION["n_xi"] >= DEFAULT_RESOLUTION["n_zeta"]
    assert DEFAULT_RESOLUTION == {"n_theta": 25, "n_zeta": 25, "n_xi": 41}


def test_the_estar_grid_separates_the_regimes():
    """0, 0.1 and 0.3 put two curves in the same regime.

    Between zero field and E*=0.1 the D11 curves are nearly indistinguishable,
    so an intermediate point belongs near 1e-3 where the E x B suppression
    actually begins.
    """
    from dkx.representative import DEFAULT_E_STAR

    assert DEFAULT_E_STAR[0] == 0.0
    assert DEFAULT_E_STAR[1] < 1.0e-2, "intermediate EStar must resolve the onset"


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


def test_the_bootstrap_panel_overlays_the_vmec_current_in_kA_per_m2():
    """The kinetic and equilibrium currents belong on one axis, in real units."""
    from dkx.representative import _panel_radial_bootstrap

    profiles = [
        {"r": r, "bootstrap": -6e-3 + r * 1e-3, "bootstrap_kA_m2": -42.0 + r * 7.0,
         "jdotb_vmec_kA_m2": -35.0 + r * 6.0, "er_ambipolar": -1.4 + r}
        for r in (0.25, 0.5, 0.75)
    ]  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_bootstrap(ax, profiles)
        assert "kA/m" in ax.get_ylabel()
        labels = [str(line.get_label()) for line in ax.get_lines()]
        assert any("DKX" in x for x in labels), labels
        assert any("VMEC" in x for x in labels), labels
    finally:
        plt.close(fig)


def test_the_bootstrap_panel_says_so_when_it_has_no_dimensional_values():
    """Without the equilibrium scalars the panel must not imply kA/m^2."""
    from dkx.representative import _panel_radial_bootstrap

    profiles = [{"r": r, "bootstrap": -6e-3, "er_ambipolar": -1.4} for r in (0.3, 0.6)]
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_bootstrap(ax, profiles)
        assert "kA/m" not in ax.get_ylabel()
        assert "SFINCS units" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_the_flux_panel_uses_si_units_when_every_surface_has_them():
    from dkx.representative import _panel_radial_fluxes

    profiles = [
        {"r": r, "particle_flux": [5e-9, 5e-9], "heat_flux": [7e-8, 1e-8],
         "particle_flux_si": [1.5e18, 1.5e18], "heat_flux_si": [22.0, 3.0]}
        for r in (0.3, 0.6)
    ]  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_fluxes(ax, profiles)
        assert "m$^{-2}$s$^{-1}$" in ax.get_ylabel()
        assert "kW/m" in ax.figure.axes[1].get_ylabel()
    finally:
        plt.close(fig)


def test_the_flux_panel_falls_back_rather_than_mixing_unit_systems():
    """One surface without the conversion must not put two systems on one axis."""
    from dkx.representative import _panel_radial_fluxes

    profiles = [
        {"r": 0.3, "particle_flux": [5e-9, 5e-9], "heat_flux": [7e-8, 1e-8],
         "particle_flux_si": [1.5e18, 1.5e18], "heat_flux_si": [22.0, 3.0]},
        {"r": 0.6, "particle_flux": [4e-9, 4e-9], "heat_flux": [6e-8, 9e-9]},
    ]  # fmt: skip
    fig, ax = plt.subplots()
    try:
        assert _panel_radial_fluxes(ax, profiles)
        assert "SFINCS units" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_the_output_file_records_the_dimensional_values_and_their_factors(tmp_path):
    from dkx import units
    from dkx.representative import write_representative_output

    profiles = [{"r": 0.5, "er_ambipolar": -1.1, "bootstrap": -6e-3,
                 "bootstrap_kA_m2": -42.0, "jdotb_vmec_kA_m2": -35.0,
                 "root_fsab2": 2.7, "particle_flux": [5e-9, 5e-9],
                 "particle_flux_si": [1.5e18, 1.5e18]}]  # fmt: skip
    out = write_representative_output(tmp_path / "run.h5", profiles=profiles)
    if out.suffix == ".h5":
        import h5py

        with h5py.File(out, "r") as handle:
            assert handle["profiles/bootstrap_kA_m2"][()] == pytest.approx([-42.0])
            assert handle["profiles/jdotb_vmec_kA_m2"][()] == pytest.approx([-35.0])
            assert handle.attrs["units/current_density_A_per_m2"] == pytest.approx(
                units.CURRENT_DENSITY)
    else:
        import json

        payload = json.loads(out.read_text())
        assert payload["profiles/bootstrap_kA_m2"] == pytest.approx([-42.0])
        assert payload["units"]["current_density_A_per_m2"] == pytest.approx(
            units.CURRENT_DENSITY)


def test_a_vacuum_equilibrium_does_not_become_a_1e9_density_plasma(tmp_path):
    """VMEC writes p ~ 1e-6 Pa for a vacuum run, not zero.

    Splitting that into n and T yields n ~ 1e9 m^-3: a collisionless deck that
    grinds for tens of minutes and reports transport for a plasma that is not
    there.  It cost 23 minutes on W7-X before this guard existed.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import FALLBACK_PLASMA, plasma_parameters, resolve_plasma

    path = tmp_path / "wout_vacuum.nc"
    with netCDF4.Dataset(path, "w") as handle:
        handle.createDimension("radius", 5)
        pres = handle.createVariable("presf", "f8", ("radius",))
        pres[:] = np.linspace(1.0e-6, 0.0, 5)

    assert plasma_parameters(path) == {}
    plasma, source = resolve_plasma(path)
    assert plasma == FALLBACK_PLASMA
    assert "vacuum" in source and "1e-06 Pa" in source


def test_a_real_pressure_profile_is_still_used(tmp_path):
    """The guard must not reject an actual finite-beta equilibrium."""
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import resolve_plasma

    path = tmp_path / "wout_beta.nc"
    with netCDF4.Dataset(path, "w") as handle:
        handle.createDimension("radius", 5)
        pres = handle.createVariable("presf", "f8", ("radius",))
        pres[:] = np.linspace(5.0e5, 0.0, 5)  # 0.5 MPa on axis

    plasma, source = resolve_plasma(path)
    assert source == "p(s) from the equilibrium"
    assert plasma["n_hat"] > 0.1  # 1e20-scale, not 1e-11


def test_the_pressure_split_gives_the_temperature_a_gradient(tmp_path):
    """A constant T zeroes dT/ds, which is most of the bootstrap drive.

    With T fixed the kinetic current comes out an order of magnitude under the
    equilibrium's own, and the overlay reads as a physics discrepancy instead of
    an artifact of the assumed profile.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import DEFAULT_T_AXIS_KEV, plasma_parameters

    path = tmp_path / "wout_beta.nc"
    with netCDF4.Dataset(path, "w") as handle:
        handle.createDimension("radius", 21)
        pres = handle.createVariable("presf", "f8", ("radius",))
        pres[:] = 7.0e5 * (1.0 - np.linspace(0.0, 1.0, 21) ** 2)

    plasma = plasma_parameters(path, 0.5)
    # Both profiles fall wherever the pressure falls: the power-law split cannot
    # invent a hollow density the way a linear T against a flat-topped p does.
    assert plasma["dt_ds"] < 0.0 and plasma["dn_ds"] < 0.0
    assert 0.0 < plasma["t_hat"] < DEFAULT_T_AXIS_KEV
    # T ~ p^(1/3) at s = r^2 = 0.25, where p/p(0) = 1 - 0.25^2.
    assert plasma["t_hat"] == pytest.approx(
        DEFAULT_T_AXIS_KEV * (1.0 - 0.25**2) ** (1.0 / 3.0), rel=1e-3)


def test_the_profile_deck_asks_for_the_gradient_coordinate_it_supplies():
    """dNHatdpsiNs needs inputRadialCoordinateForGradients = 1, in &geometryParameters.

    With the wrong coordinate the gradients are ignored and every moment comes
    back at ~1e-20; with the key in &physicsParameters it is not read at all.
    A d/drHat key fed a d/drN number is a factor of aHat -- 0.17 on a compact
    device, so a six-fold error in the drive rather than an obvious failure.
    """
    from dkx.inputs import sfincs_input_from_raw
    from dkx.namelist import parse_sfincs_input_text
    from dkx.representative import DEFAULT_RESOLUTION_PROFILE, FALLBACK_PLASMA, _PROFILE_TEMPLATE
    from dkx.representative import _plasma_keys

    text = _PROFILE_TEMPLATE.format(
        equilibrium="/w.nc", **DEFAULT_RESOLUTION_PROFILE, **_plasma_keys(dict(FALLBACK_PLASMA)))
    parsed = sfincs_input_from_raw(parse_sfincs_input_text(text))
    assert int(parsed.geometry.input_radial_coordinate_for_gradients) == 1
    assert len(parsed.species.d_n_hat_d_psi_ns) == 2
    assert len(parsed.species.d_t_hat_d_psi_ns) == 2
    assert all(v < 0.0 for v in parsed.species.d_t_hat_d_psi_ns)
    # The bootstrap current is the parallel-momentum moment; PAS has no
    # momentum-restoring term and runs 35-47% high against Redl.
    assert int(parsed.physics.collision_operator) == 0


def test_equilibrium_scalars_reads_the_wout_constants_the_units_need(tmp_path):
    """psiAHat, aHat and jdotb: without these there is no dimensional overlay.

    Nothing else in this module imports the function, so a refactor can delete
    it and every other test still passes -- which is exactly what happened once.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import equilibrium_scalars

    path = tmp_path / "wout_scalars.nc"
    with netCDF4.Dataset(path, "w") as handle:
        handle.createDimension("radius", 4)
        handle.createVariable("phi", "f8", ("radius",))[:] = [0.0, 0.5, 1.0, 2.0 * np.pi]
        handle.createVariable("Aminor_p", "f8", ())[...] = 0.42
        handle.createVariable("jdotb", "f8", ("radius",))[:] = [-1.0, -2.0, -3.0, -4.0]

    scalars = equilibrium_scalars(path)
    assert scalars["psi_a_hat"] == pytest.approx(1.0)  # phi(ns) / 2 pi
    assert scalars["a_hat"] == pytest.approx(0.42)
    assert scalars["jdotb"].tolist() == [-1.0, -2.0, -3.0, -4.0]
    assert scalars["jdotb_s"][0] == 0.0 and scalars["jdotb_s"][-1] == 1.0
    assert equilibrium_scalars(tmp_path / "absent.nc") == {}
