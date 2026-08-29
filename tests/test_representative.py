"""The `dkx --plot` panels and the representative run.

Two properties matter more than pixel layout. The panels must render from
*Fortran* SFINCS output as well as DKX's -- both are ``sfincsOutput.h5`` in the
same layout, so a reader that only works on one of them is a bug. And a panel
that cannot be drawn must say so rather than leave an empty axis, which a reader
would take for zero.
"""

from __future__ import annotations

import tempfile
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


def test_unbracketed_profile_keeps_closest_scan_observables(monkeypatch, tmp_path):
    """A missed root is explicit, but it no longer blanks Er and flux panels."""
    from types import SimpleNamespace

    from dkx import api
    from dkx import representative as rep

    moments = {
        "FSABjHat": np.asarray([-4.0, -3.0, -2.0]),
        "particleFlux_vm_psiHat": np.asarray(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ),
        "heatFlux_vm_psiHat": np.asarray(
            [[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]]
        ),
    }
    monkeypatch.setattr(
        api,
        "batched_er_scan",
        lambda *_args, **_kwargs: SimpleNamespace(
            radial_current=np.asarray([3.0, 1.0, 2.0]), moments=moments
        ),
    )
    monkeypatch.setattr(
        rep, "resolve_plasma", lambda _path: (dict(rep.FALLBACK_PLASMA), "test")
    )
    monkeypatch.setattr(rep, "equilibrium_scalars", lambda _path: {})

    profiles = rep.radial_profiles(
        tmp_path / "wout.nc", surfaces=[0.5], er_values=[-2.0, 0.0, 2.0]
    )
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["evaluation_status"] == "no_bracketed_root"
    assert profile["evaluation_is_root"] is False
    assert profile["er_evaluated"] == pytest.approx(0.0)
    assert profile["radial_current_evaluated"] == pytest.approx(1.0)
    assert "er_ambipolar" not in profile
    assert profile["particle_flux"] == pytest.approx([2.0, 5.0])
    assert profile["heat_flux"] == pytest.approx([20.0, 50.0])

    fig, (boot_ax, flux_ax) = plt.subplots(1, 2)
    try:
        assert rep._panel_radial_bootstrap(boot_ax, profiles)
        assert rep._panel_radial_fluxes(flux_ax, profiles)
        assert "closest scanned" in boot_ax.get_title()
        assert "closest scanned" in flux_ax.get_title()
    finally:
        plt.close(fig)

    output = rep.write_representative_output(tmp_path / "closest.h5", profiles=profiles)
    import h5py

    with h5py.File(output, "r") as handle:
        assert handle["profiles/er_evaluated"][()] == pytest.approx([0.0])
        assert handle["profiles/radial_current_evaluated"][()] == pytest.approx([1.0])
        assert handle["profiles/evaluation_is_root"][()] == pytest.approx([0.0])


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
    profiles = [{"r": 0.5, "er_ambipolar": -1.1, "er_evaluated": -1.1,
                 "evaluation_is_root": True, "bootstrap": -6e-3,
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
            assert handle["profiles/er_evaluated"][()] == pytest.approx([-1.1])
            assert handle["profiles/evaluation_is_root"][()] == pytest.approx([1.0])
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
        handle.createVariable("Aminor_p", "f8", ())[...] = 0.6

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
        handle.createVariable("Aminor_p", "f8", ())[...] = 0.6

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
        handle.createVariable("Aminor_p", "f8", ())[...] = 0.6

    plasma = plasma_parameters(path, 0.5)
    # Both profiles fall wherever the pressure falls: the power-law split cannot
    # invent a hollow density the way a linear T against a flat-topped p does.
    assert plasma["dt_drhat"] < 0.0 and plasma["dn_drhat"] < 0.0
    assert 0.0 < plasma["t_hat"] < DEFAULT_T_AXIS_KEV
    # T ~ p^(1/3) at s = r^2 = 0.25, where p/p(0) = 1 - 0.25^2.
    assert plasma["t_hat"] == pytest.approx(
        DEFAULT_T_AXIS_KEV * (1.0 - 0.25**2) ** (1.0 / 3.0), rel=1e-3)


def test_the_profile_deck_asks_for_the_gradient_coordinate_it_supplies():
    """dNHatdrHats needs inputRadialCoordinateForGradients = 4 -- the v3 default.

    Two failure modes, neither of which raises.  A key that does not match the
    coordinate leaves the gradients at ZERO and every moment comes back at
    ~1e-20.  And a d/drHat key fed a d/drN number is off by aHat -- 0.17 on a
    compact device, so a six-fold error in the drive.

    Code 4 is also the only one that drives the potential with Er; the
    ambipolar scan is an Er scan, so no other code is available here.
    """
    from dkx.inputs import sfincs_input_from_raw
    from dkx.namelist import parse_sfincs_input_text
    from dkx.representative import (DEFAULT_RESOLUTION_PROFILE, FALLBACK_PLASMA,
                                    _PROFILE_TEMPLATE, _plasma_keys)

    text = _PROFILE_TEMPLATE.format(
        equilibrium="/w.nc", **DEFAULT_RESOLUTION_PROFILE, **_plasma_keys(dict(FALLBACK_PLASMA)))
    parsed = sfincs_input_from_raw(parse_sfincs_input_text(text))
    assert int(parsed.geometry.input_radial_coordinate_for_gradients) == 4
    assert len(parsed.species.d_n_hat_d_r_hats) == 2
    assert len(parsed.species.d_t_hat_d_r_hats) == 2
    assert all(v < 0.0 for v in parsed.species.d_t_hat_d_r_hats)
    # The bootstrap current is the parallel-momentum moment; PAS has no
    # momentum-restoring term and runs 35-47% high against Redl.
    assert int(parsed.physics.collision_operator) == 0


def test_the_gradients_carry_the_aHat_chain_rule(tmp_path):
    """d/drHat = (1/aHat) d/drN, and aHat comes from the wout, not from 1.0."""
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import plasma_parameters

    def written(a_minor: float):
        path = tmp_path / f"wout_a{a_minor}.nc"
        with netCDF4.Dataset(path, "w") as handle:
            handle.createDimension("radius", 21)
            handle.createVariable("presf", "f8", ("radius",))[:] = (
                7.0e5 * (1.0 - np.linspace(0.0, 1.0, 21)))
            handle.createVariable("Aminor_p", "f8", ())[...] = a_minor
        return plasma_parameters(path, 0.5)

    wide, narrow = written(1.0), written(0.25)
    assert narrow["dn_drhat"] == pytest.approx(4.0 * wide["dn_drhat"])
    assert narrow["dt_drhat"] == pytest.approx(4.0 * wide["dt_drhat"])
    # The local values themselves do not depend on the minor radius.
    assert narrow["n_hat"] == pytest.approx(wide["n_hat"])


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


def test_the_plasma_summary_covers_every_key_the_deck_needs(tmp_path):
    """The log line, the caption and the namelist must agree on the key names.

    Two hand-written f-strings drifted from the template across a key rename and
    took down a 60-second run at its last line, twice.  One formatter, and a
    test that walks it with both the fallback and a real equilibrium.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    from dkx.representative import (DEFAULT_RESOLUTION_PROFILE, FALLBACK_PLASMA,
                                    _PROFILE_TEMPLATE, _plasma_keys, plasma_parameters,
                                    plasma_summary, resolve_plasma)

    path = tmp_path / "wout_real.nc"
    with netCDF4.Dataset(path, "w") as handle:
        handle.createDimension("radius", 21)
        handle.createVariable("presf", "f8", ("radius",))[:] = (
            7.0e5 * (1.0 - np.linspace(0.0, 1.0, 21)))
        handle.createVariable("Aminor_p", "f8", ())[...] = 0.6

    for plasma in (dict(FALLBACK_PLASMA), plasma_parameters(path, 0.5), resolve_plasma(path)[0]):
        assert plasma, "every branch must yield a usable plasma"
        summary = plasma_summary(plasma, "source phrase")
        assert "n=" in summary and "dT/drHat=" in summary and "(source phrase)" in summary
        # The same dict must fill the namelist without a KeyError.
        _PROFILE_TEMPLATE.format(
            equilibrium="/w.nc", **DEFAULT_RESOLUTION_PROFILE, **_plasma_keys(plasma))


def test_the_caption_warns_that_the_two_currents_need_not_coincide():
    """DKX runs an assumed n/T split, not the equilibrium's own profiles.

    On the precise-QA reference the same solver gives DKX/VMEC = 1.04-1.23 with
    that equilibrium's own T0 = 5 keV profiles and 0.42 with this figure's
    T0 = 2 keV closure: a factor of 2.7 from the temperature assumption alone.
    Without the caveat a reader reads that panel as a code disagreement.
    """
    from dkx.representative import figure_caption

    plasma = {"n_hat": 9.3, "t_hat": 1.8, "dn_drhat": -49.0, "dt_drhat": -4.8}
    caption = figure_caption(plasma, "p(s) from the equilibrium",
                             {"profiles": {"n_theta": 13, "n_zeta": 19}})  # fmt: skip
    assert "assumed split of p" in caption
    assert "need not coincide" in caption
    assert "p(s) from the equilibrium" in caption
    assert "profiles 13x19" in caption
    assert caption.count("\n") == 1, "the footer is two lines; the band reserves for two"
    # No plasma, no claim about one.
    assert "assumed split" not in figure_caption(None, "", {"mono": {"n_theta": 25}})


def test_every_preset_threads_its_grid_through_both_solve_stages(monkeypatch, tmp_path):
    """A preset must reach the profile solve and the radial scan, not just the scan.

    The monoenergetic grid is chosen in run_representative itself, so it is
    hard to get wrong; the two profile stages take their resolution from a
    module constant, and threading a flag to one but not the other would leave
    the preset quietly half-applied.
    """
    from dkx import representative as rep

    seen: dict[str, dict] = {}

    def fake_profile_data(equilibrium, *, full=False, quick=False, emit=None):
        seen["profile"] = rep._profile_resolution(full=full, quick=quick)
        return {}

    def fake_radial(equilibrium, *, full=False, quick=False, emit=None, **kwargs):
        seen["radial"] = rep._profile_resolution(full=full, quick=quick)
        return []

    monkeypatch.setattr(rep, "_profile_data", fake_profile_data)
    monkeypatch.setattr(rep, "radial_profiles", fake_radial)
    monkeypatch.setattr(rep, "monoenergetic_scan", lambda *a, **k: [])
    monkeypatch.setattr(rep, "resolve_plasma",
                        lambda eq: (dict(rep.FALLBACK_PLASMA), "generic reference"))  # fmt: skip

    wout = tmp_path / "wout_stub.nc"
    wout.write_text("")
    presets = (
        ({}, rep.DEFAULT_RESOLUTION_PROFILE),
        ({"full": True}, rep.FULL_RESOLUTION_PROFILE),
        ({"quick": True}, rep.QUICK_RESOLUTION_PROFILE),
    )
    for flags, expected in presets:
        label = "-".join(flags) or "default"
        rep.run_representative(wout, out_path=tmp_path / f"p{label}.png", emit=None, **flags)
        assert seen["profile"] == expected, f"profile stage ignored {flags}"
        assert seen["radial"] == expected, f"radial stage ignored {flags}"
    grids = [rep.DEFAULT_RESOLUTION_PROFILE, rep.FULL_RESOLUTION_PROFILE,
             rep.QUICK_RESOLUTION_PROFILE]  # fmt: skip
    assert len({tuple(sorted(g.items())) for g in grids}) == 3, "two presets share a grid"


def test_full_and_quick_together_are_refused():
    """They pull in opposite directions, so silently picking one is a trap."""
    from dkx import representative as rep

    with pytest.raises(ValueError, match="opposite presets"):
        rep._profile_resolution(full=True, quick=True)


def test_the_quick_speed_grid_keeps_the_ambipolar_root_alive():
    """``n_x`` is the one axis ``--quick`` may not cut, and this records why.

    Measured on ``tests/ref/wout_up_down_asymmetric_tokamak.nc``: at ``n_x = 3``
    the radial current is negative across the whole ``E_r`` bracket, so there is
    no sign change, no ambipolar root, and the bootstrap and flux panels come
    out empty. At ``n_x = 4`` with the same angular grid the roots reappear.
    Cutting the speed grid to buy CI seconds therefore costs half the figure.
    """
    from dkx import representative as rep

    assert rep.QUICK_RESOLUTION_PROFILE["n_x"] >= 4
    # The angular axes are the ones the preset is allowed to cut, and it must
    # actually cut them or it is not a quick preset at all.
    for axis in ("n_theta", "n_zeta", "n_xi"):
        assert rep.QUICK_RESOLUTION_PROFILE[axis] < rep.DEFAULT_RESOLUTION_PROFILE[axis], axis


def test_quick_keeps_enough_points_for_every_panel_to_be_a_curve():
    """One point is a dot, and a one-point axis stops exercising the code path."""
    from dkx import representative as rep

    assert len(rep.QUICK_NU_PRIME) >= 3, "the D11/D31/D33 panels need a curve"
    assert len(rep.QUICK_E_STAR) >= 2, "the legend claims one curve per EStar"
    assert len(rep.QUICK_SURFACES) >= 2, "a radial profile needs two radii"
    # A bracket that does not span zero cannot contain a sign change.
    assert min(rep.QUICK_ER_BRACKET) < 0.0 < max(rep.QUICK_ER_BRACKET)


def test_quick_reports_the_grid_it_actually_ran(monkeypatch, tmp_path):
    """The figure caption and the HDF5 must not name the default grid.

    ``write_representative_output`` used to hard-code ``DEFAULT_RESOLUTION`` for
    the monoenergetic entry, which was true while the only presets shared that
    grid and became a false label the moment one did not.
    """
    from dkx import representative as rep

    captured: dict[str, dict] = {}

    def fake_plot(out, **kwargs):
        captured["resolutions"] = kwargs["resolutions"]
        Path(out).write_bytes(b"")
        return Path(out)

    monkeypatch.setattr(rep, "_profile_data", lambda *a, **k: {})
    monkeypatch.setattr(rep, "radial_profiles", lambda *a, **k: [])
    monkeypatch.setattr(rep, "monoenergetic_scan", lambda *a, **k: [])
    monkeypatch.setattr(rep, "plot_representative", fake_plot)
    monkeypatch.setattr(rep, "resolve_plasma",
                        lambda eq: (dict(rep.FALLBACK_PLASMA), "generic reference"))  # fmt: skip

    wout = tmp_path / "wout_stub.nc"
    wout.write_text("")
    rep.run_representative(wout, out_path=tmp_path / "q.png", quick=True, emit=None)
    assert captured["resolutions"]["monoenergetic"] == rep.QUICK_RESOLUTION
    assert captured["resolutions"]["profiles"] == rep.QUICK_RESOLUTION_PROFILE

    # The HDF5 is the half that was wrong: plot_representative was always given
    # the grid, write_representative_output substituted the default for it.
    import h5py

    with h5py.File(tmp_path / "q.h5", "r") as handle:
        written = {
            key.split("/", 1)[1]: int(value)
            for key, value in handle.attrs.items()
            if key.startswith("resolution_monoenergetic/")
        }
    assert written == rep.QUICK_RESOLUTION


def test_the_out_of_memory_retry_grid_never_grows_an_axis():
    """A "reduced" grid that increases an axis retries a bigger, slower solve.

    The floors are there so the retry stays solvable, but a bare max() against
    the default n_x of 4 raised it to 5.
    """
    from dkx.representative import DEFAULT_RESOLUTION_PROFILE, FULL_RESOLUTION_PROFILE

    def reduced_of(resolution):
        return {k: min(v, max(int(v * 2 / 3), 3 if k == "n_x" else 9))
                for k, v in resolution.items()}  # fmt: skip

    for resolution in (DEFAULT_RESOLUTION_PROFILE, FULL_RESOLUTION_PROFILE):
        reduced = reduced_of(resolution)
        assert set(reduced) == set(resolution)
        for axis, value in reduced.items():
            assert value <= resolution[axis], f"{axis} grew: {resolution[axis]} -> {value}"
            assert value >= 3, f"{axis} collapsed to {value}"
        assert any(reduced[a] < resolution[a] for a in resolution), "nothing was reduced"
