"""``dkx convert``: a SFINCS deck becomes a native case that solves the same problem.

This is the migration path DKX's compatibility pitch rests on, so the test that
matters is not "the file parses" but "the converted case answers the deck's
question". Two decks are run twice -- once through the SFINCS-compatibility path
(:func:`dkx.run.run_profile`) and once as a converted case through
:func:`dkx.execution.run_case` -- and their fluxes and bootstrap current are
compared in SI.

Two structural facts make that conversion more than a rename, and both are
pinned below:

- A deck is **dimensionless**: ``Zs``/``mHats``/``nHats``/``THats``/``Er`` are
  ratios to the pinned SFINCS reference set. A case is **SI**. Every rescaling
  goes through :mod:`dkx.units`, whose reference values are fixed by
  ``globalVariables.F90:133-135`` rather than chosen.
- A deck states **one surface plus prescribed gradients**; a case states a
  **profile**, and :mod:`dkx.execution` recovers gradients by differentiating
  it. The converter therefore expands one surface into three carrying the
  profile linear in ``rHat``, which reproduces the deck's gradients exactly.

**No deck checked into this repository converts.** All 100 of them
(``tests/ref/*.input.namelist`` and ``examples/sfincs_examples/*/input.namelist``)
are refused, every one for a reason that is a real limit of the native route
rather than a converter defect: they override ``nu_n`` (which a case cannot
express: it is fixed by the pinned reference set and ln(Lambda) = 17), or ask
for a ``geometryScheme=1`` model the native analytic route builds from defaults
only, or snap to a stored VMEC/Boozer surface (``VMECRadialOption=1``), or run
``RHSMode`` 2/3. The two fixtures used here
(``tests/ref/convert_*_tiny.input.namelist``) were therefore written for this
test, inside the convertible subset. That is worth knowing on its own: the
converter is correct and the convertible subset is narrow.
"""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from dkx import cli
from dkx.config import Case, CaseValidationError
from dkx.constants import RadialCoordinates
from dkx.input_compat import (
    case_from_sfincs_namelist,
    convert_sfincs_namelist,
    write_case_file,
)
from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX

REF = Path(__file__).resolve().parent / "ref"
PAS_DECK = REF / "convert_pas_w7x_noEr_tiny.input.namelist"
FP_DECK = REF / "convert_fp_lhd_withEr_tiny.input.namelist"

#: Agreement required between the two paths. The two calculations are the same
#: physics discretized identically, so the only difference is float rounding in
#: the dimensionless -> SI -> dimensionless round trip, amplified by the solve.
#: Measured: 1.9e-14 (pitch-angle scattering, direct solve) and 9.0e-11
#: (Fokker-Planck, iterative solve at solverTolerance = 1e-10).
ROUND_TRIP_RTOL = 1.0e-8

#: ``(psiAHat, aHat, rN)`` that ``geometry.F90`` fixes for the built-in models
#: the two fixtures use; ``rN`` is forced to 0.5 upstream.
DECK_RADIAL = {
    PAS_DECK: (-0.384935, 0.5109, 0.5),
    FP_DECK: (0.5585 * 0.5585 / 2.0, 0.5585, 0.5),
}


def _quiet(_message: str) -> None:
    """Progress sink: the physics is the subject here, not the console flow."""


def _relative_difference(left, right) -> float:
    left = np.atleast_1d(np.asarray(left, dtype=np.float64))
    right = np.atleast_1d(np.asarray(right, dtype=np.float64))
    scale = np.maximum(np.abs(left), np.abs(right))
    return float(
        np.max(np.where(scale > 0.0, np.abs(left - right) / np.where(scale > 0.0, scale, 1.0), 0.0))
    )


def _deck_variant(
    tmp_path: Path,
    name: str,
    *,
    insert: dict[str, str] | None = None,
    replace: tuple[str, str] | None = None,
    source: Path = PAS_DECK,
) -> Path:
    """A convertible deck with one thing changed, for the refusal tests."""
    text = source.read_text(encoding="utf-8")
    if replace is not None:
        old, new = replace
        assert old in text, old
        text = text.replace(old, new)
    for group, lines in (insert or {}).items():
        marker = f"&{group}\n"
        assert marker in text, group
        text = text.replace(marker, f"{marker}{lines}\n")
    path = tmp_path / f"{name}.input.namelist"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The acceptance test: a converted case solves the deck's problem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck", [PAS_DECK, FP_DECK], ids=["pas_w7x_noEr", "fp_lhd_withEr"])
def test_the_converted_case_reproduces_the_deck_fluxes(deck: Path, tmp_path: Path) -> None:
    """Both paths must report the same transport, in SI, at the deck's surface.

    A converter that merely produced a valid case file would pass every other
    test in this module while quietly solving a different problem: a dropped
    gradient, an ``mHats``-as-``mass_amu`` confusion, or a collisionality taken
    from the wrong reference set all leave the schema satisfied. Only comparing
    the answers catches them.
    """
    from dkx.execution import run_case  # noqa: PLC0415
    from dkx.run import run_profile  # noqa: PLC0415

    psi_a_hat, a_hat, r_n = DECK_RADIAL[deck]
    radial = RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    # diagnostics.F90:703 -- a psiHat flux times d(rHat)/d(psiHat) is the
    # physical radial one, which is what run_case reports.
    radial_factor = radial.d_dr_hat_to_d_dpsi_hat

    deck_run = run_profile(deck, emit=None, tol=1.0e-11)
    deck_particle = np.asarray(deck_run.moments["particleFlux_vm_psiHat"]) * radial_factor * PARTICLE_FLUX
    deck_heat = np.asarray(deck_run.moments["heatFlux_vm_psiHat"]) * radial_factor * HEAT_FLUX
    deck_current = float(np.asarray(deck_run.moments["FSABjHat"])) * PARALLEL_CURRENT

    case, written = convert_sfincs_namelist(deck, tmp_path / "case.toml")
    result = run_case(Case.from_file(written), emit=_quiet)

    surfaces = np.asarray(case.geometry.surfaces)
    index = int(np.argmin(np.abs(surfaces - r_n * r_n)))
    assert surfaces[index] == pytest.approx(r_n * r_n, rel=1e-12), (
        "the deck's own surface must be one of the profile's surfaces"
    )

    assert _relative_difference(deck_particle, result["particle_flux_m2_s"][index]) < ROUND_TRIP_RTOL
    assert _relative_difference(deck_heat, result["heat_flux_W_m2"][index]) < ROUND_TRIP_RTOL
    assert _relative_difference(deck_current, result["parallel_current_A_T_m2"][index]) < ROUND_TRIP_RTOL
    # A comparison of two zeros would pass everything above without evidence.
    assert np.any(np.abs(deck_particle) > 0.0)
    assert abs(deck_current) > 0.0


def test_a_boozer_deck_converts_and_reproduces_its_fluxes(tmp_path: Path) -> None:
    """The file-geometry path, on the checked-in non-stellarator-symmetric ``.bc``.

    Worth its own case for three reasons the analytic fixtures cannot show: the
    equilibrium is recorded by path, the profile stencil has to stay inside the
    two surfaces the file actually stores (0.4 and 0.6), and the deck sits on the
    outermost of them -- so the stencil is one-sided and the deck's surface is
    the *last* of the three, not the middle one.

    The deck is the checked-in ``geometryScheme=12`` fixture with its two
    blockers removed, which is also a compact statement of what stands between a
    real deck and conversion.
    """
    from dkx.execution import run_case  # noqa: PLC0415
    from dkx.magnetic_geometry import read_native_boozer  # noqa: PLC0415
    from dkx.run import run_profile  # noqa: PLC0415

    text = (REF / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist").read_text()
    text = text.replace("  nu_n = 8.4774d-3\n", "").replace("  NL = 2\n", "")
    text = text.replace("geometryScheme = 12", "geometryScheme = 12\n  VMECRadialOption = 0")
    deck = tmp_path / "boozer.input.namelist"
    deck.write_text(text, encoding="utf-8")

    boozer = read_native_boozer(REF / "nonStelSym_tiny_geometryScheme12.bc")
    radial = RadialCoordinates(
        psi_a_hat=float(boozer.header.psi_a_hat),
        a_hat=float(boozer.header.a_hat),
        r_n=0.6,  # rN_wish in the deck
    )

    case, written = convert_sfincs_namelist(deck, tmp_path / "boozer.toml")
    assert case.geometry.format == "boozer"
    assert Case.from_file(written).geometry_path.exists()
    surfaces = np.asarray(case.geometry.surfaces)
    index = int(np.argmin(np.abs(surfaces - 0.36)))
    assert index == len(surfaces) - 1, "an outermost stored surface needs a one-sided stencil"
    assert float(np.min(np.sqrt(surfaces))) >= min(float(s.r_n) for s in boozer.surfaces)

    deck_run = run_profile(deck, emit=None, tol=1.0e-11)
    factor = radial.d_dr_hat_to_d_dpsi_hat
    result = run_case(Case.from_file(written), emit=_quiet)
    assert (
        _relative_difference(
            np.asarray(deck_run.moments["particleFlux_vm_psiHat"]) * factor * PARTICLE_FLUX,
            result["particle_flux_m2_s"][index],
        )
        < ROUND_TRIP_RTOL
    )
    assert (
        _relative_difference(
            float(np.asarray(deck_run.moments["FSABjHat"])) * PARALLEL_CURRENT,
            result["parallel_current_A_T_m2"][index],
        )
        < ROUND_TRIP_RTOL
    )


def test_the_profile_carries_the_decks_gradients_at_the_decks_surface() -> None:
    """The three surfaces exist to carry ``dNHatdrHats``; check they do.

    ``numpy.gradient`` against ``rHat`` is what ``dkx.execution`` differentiates
    with, so reproducing it here states the invariant the stencil is built for
    rather than restating the construction.
    """
    case = case_from_sfincs_namelist(PAS_DECK)
    _psi_a_hat, a_hat, r_n = DECK_RADIAL[PAS_DECK]
    r_hat = a_hat * np.sqrt(np.asarray(case.geometry.surfaces))
    index = int(np.argmin(np.abs(np.asarray(case.geometry.surfaces) - r_n * r_n)))

    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    assert n_hat[index] == pytest.approx(0.8, rel=1e-12)
    assert t_hat[index] == pytest.approx(1.2, rel=1e-12)
    # dNHatdrHats = -0.5, dTHatdrHats = -2.0 in the deck.
    assert np.gradient(n_hat, r_hat, edge_order=2)[index] == pytest.approx(-0.5, rel=1e-9)
    assert np.gradient(t_hat, r_hat, edge_order=2)[index] == pytest.approx(-2.0, rel=1e-9)


@pytest.mark.parametrize(
    ("coordinate", "key", "value"),
    [
        (3, "rN_wish", 0.4),
        (1, "psiN_wish", 0.16),
        (2, "rHat_wish", 0.5585 * 0.4),
        (0, "psiHat_wish", 0.15596 * 0.16),
    ],
    ids=["rN", "psiN", "rHat", "psiHat"],
)
def test_every_radial_wish_key_selects_the_same_surface(
    tmp_path: Path, coordinate: int, key: str, value: float
) -> None:
    """``inputRadialCoordinate`` picks which ``*_wish`` key names the surface.

    All four spell the same surface, and ``geometry.surfaces`` is normalized
    toroidal flux, so all four must convert to it. Checked on
    ``geometryScheme=1``: the built-in LHD/W7-X models ignore the wish keys
    upstream (``rN`` is forced to 0.5), so they cannot show this.
    """
    deck = _deck_variant(
        tmp_path,
        f"wish_{key}",
        replace=(
            "geometryScheme = 4",
            f"geometryScheme = 1\n  inputRadialCoordinate = {coordinate}\n"
            f"  {key} = {value!r}",
        ),
    )
    case = case_from_sfincs_namelist(deck)
    assert case.geometry.format == "analytic"
    assert case.geometry.file.name == "tokamak"
    surfaces = np.asarray(case.geometry.surfaces)
    assert surfaces[int(np.argmin(np.abs(surfaces - 0.16)))] == pytest.approx(0.16, rel=1e-9)


@pytest.mark.parametrize("coordinate", [0, 1, 2, 3], ids=["psiHat", "psiN", "rHat", "rN"])
def test_every_species_gradient_coordinate_gives_the_same_profile(
    tmp_path: Path, coordinate: int
) -> None:
    """``dNHatd{psiHat,psiN,rHat,rN}s`` are one gradient in four coordinates.

    The conversion runs through :class:`dkx.constants.RadialCoordinates`, the
    same helper ``radialCoordinates.F90`` is ported into, so a deck that states
    its gradients in any of the four must produce the identical SI profile.
    """
    psi_a_hat, a_hat, r_n = DECK_RADIAL[PAS_DECK]
    radial = RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    suffix = {0: "psiHats", 1: "psiNs", 2: "rHats", 3: "rNs"}[coordinate]
    # The deck states d/drHat; re-express it in `coordinate` so both spellings
    # reach the same d/dpsiHat, and check the SI profile did not move.
    to_psi_hat = {
        0: 1.0,
        1: radial.d_dpsi_n_to_d_dpsi_hat,
        2: radial.d_dr_hat_to_d_dpsi_hat,
        3: radial.d_dr_n_to_d_dpsi_hat,
    }[coordinate]
    scale = radial.d_dr_hat_to_d_dpsi_hat / to_psi_hat
    deck = _deck_variant(
        tmp_path,
        f"gradient_{suffix}",
        replace=(
            "  dNHatdrHats = -0.5d+0\n  dTHatdrHats = -2.0d+0",
            f"  dNHatd{suffix} = {-0.5 * scale!r}\n  dTHatd{suffix} = {-2.0 * scale!r}",
        ),
    )
    converted = case_from_sfincs_namelist(deck).species[0]
    reference = case_from_sfincs_namelist(PAS_DECK).species[0]
    assert converted.density_m3 == pytest.approx(reference.density_m3, rel=1e-12)
    assert converted.temperature_keV == pytest.approx(reference.temperature_keV, rel=1e-12)


def test_a_gradient_too_steep_for_a_positive_profile_refuses(tmp_path: Path) -> None:
    """A case declares positive densities at every surface; some decks cannot.

    The stencil shrinks to keep the linear profile physical, but a gradient
    steep enough to need a stencil narrower than float noise has no honest
    profile representation at all, and saying so beats emitting surfaces that
    collide once squared.
    """
    deck = _deck_variant(
        tmp_path,
        "steep_gradient",
        replace=("dNHatdrHats = -0.5d+0", "dNHatdrHats = -1.0d+12"),
    )
    with pytest.raises(CaseValidationError) as excinfo:
        case_from_sfincs_namelist(deck)
    assert "dNHatdrHats" in str(excinfo.value)


def test_dimensionless_species_inputs_become_si_through_the_pinned_references() -> None:
    """``mHats`` is a proton-mass ratio and ``mass_amu`` is not; 0.14% matters."""
    case = case_from_sfincs_namelist(FP_DECK)
    ion, electron = case.species
    assert ion.charge == 1.0 and electron.charge == -1.0
    # mHats = 2.0 -> 2 * m_p / u, not 2.0.
    assert ion.mass_amu == pytest.approx(2.01455, rel=1e-5)
    assert ion.mass_amu != pytest.approx(2.0, rel=1e-3)
    assert electron.mass_amu == pytest.approx(5.4858e-4, rel=1e-4)
    assert ion.name == "deuterium" and electron.name == "electron"
    # nHats = 0.6 -> 0.6 * nBar, THats = 0.9 -> 0.9 * TBar (1 keV).
    index = 1
    assert ion.density_m3[index] == pytest.approx(6.0e19, rel=1e-12)
    assert ion.temperature_keV[index] == pytest.approx(0.9, rel=1e-12)


def test_a_finite_er_becomes_the_prescribed_field_in_kv_per_metre() -> None:
    case = case_from_sfincs_namelist(FP_DECK)
    assert case.electric_field.mode == "prescribed"
    assert case.electric_field.value_kV_m == pytest.approx(0.6, rel=1e-12)


def test_an_ambipolar_deck_becomes_an_ambipolar_workflow_with_its_search_range(
    tmp_path: Path,
) -> None:
    deck = _deck_variant(
        tmp_path,
        "ambipolar",
        insert={
            "general": "  ambipolarSolve = .true.\n  Er_min = -12.0d+0\n"
            "  Er_max = 8.0d+0\n  NEr_ambipolarSolve = 7"
        },
    )
    case = case_from_sfincs_namelist(deck)
    assert case.run.workflow == "ambipolar_profile"
    assert case.electric_field.mode == "ambipolar"
    assert case.electric_field.search_kV_m == pytest.approx((-12.0, 8.0))
    assert case.electric_field.search_points == 7


# ---------------------------------------------------------------------------
# Serialization: the destination extension picks the format
# ---------------------------------------------------------------------------


def test_the_destination_extension_picks_the_format(tmp_path: Path) -> None:
    toml_case, toml_path = convert_sfincs_namelist(PAS_DECK, tmp_path / "case.toml")
    json_case, json_path = convert_sfincs_namelist(PAS_DECK, tmp_path / "case.json")

    toml_text = toml_path.read_text(encoding="utf-8")
    assert toml_text.lstrip().startswith("#")
    assert "[geometry]" in toml_text and "[[species]]" in toml_text
    assert tomllib.loads(toml_text)["geometry"]["format"] == "analytic"

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["geometry"]["format"] == "analytic"
    # Same calculation either way: the format is a serialization choice.
    assert toml_case.case_id == json_case.case_id


def test_an_unknown_destination_extension_is_refused_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(CaseValidationError) as excinfo:
        convert_sfincs_namelist(PAS_DECK, tmp_path / "case.yaml")
    assert "case.yaml" in str(excinfo.value)
    assert not (tmp_path / "case.yaml").exists()


def test_an_existing_destination_is_not_silently_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "case.toml"
    convert_sfincs_namelist(PAS_DECK, destination)
    with pytest.raises(CaseValidationError):
        convert_sfincs_namelist(PAS_DECK, destination)
    # Overwriting is available, it just has to be asked for.
    convert_sfincs_namelist(PAS_DECK, destination, overwrite=True)


@pytest.mark.parametrize("suffix", [".toml", ".json"])
def test_a_converted_case_passes_case_from_file_validation(tmp_path: Path, suffix: str) -> None:
    """Both serializations must reload through the single validation boundary.

    Absent optionals are the trap: ``search_kV_m`` is ``None`` for a prescribed
    field, TOML has no null, and the case parser rejects an explicit ``None``
    where it wants a number. Omission is the only correct spelling.
    """
    case, written = convert_sfincs_namelist(PAS_DECK, tmp_path / f"case{suffix}")
    reloaded = Case.from_file(written)
    assert reloaded.case_id == case.case_id
    assert reloaded.electric_field.search_kV_m is None
    assert reloaded.run.workflow == "profile"


def test_an_ambipolar_case_also_round_trips_through_both_formats(tmp_path: Path) -> None:
    """The other half of the optional fields: no ``value_kV_m``, a search range."""
    deck = _deck_variant(
        tmp_path,
        "ambipolar_rt",
        insert={"general": "  ambipolarSolve = .true.\n  Er_min = -5.0d+0\n  Er_max = 5.0d+0"},
    )
    case, written = convert_sfincs_namelist(deck, tmp_path / "amb.toml")
    reloaded = Case.from_file(written)
    assert reloaded.case_id == case.case_id
    assert reloaded.electric_field.value_kV_m is None
    assert reloaded.electric_field.search_kV_m == pytest.approx((-5.0, 5.0))


def test_write_case_file_refuses_an_unknown_extension(tmp_path: Path) -> None:
    case = case_from_sfincs_namelist(PAS_DECK)
    with pytest.raises(CaseValidationError):
        write_case_file(case, tmp_path / "case.ini")


# ---------------------------------------------------------------------------
# Refusals: one per category the case schema or native route cannot carry
# ---------------------------------------------------------------------------

#: ``(id, deck edit, the namelist key the message must name)``. Every entry is a
#: model the native case route does NOT implement, so converting it would
#: produce a case that runs and is wrong, or does not run at all -- plan.md
#: operating rule 11 requires each to refuse at convert time instead.
REFUSALS: tuple[tuple[str, dict, str], ...] = (
    (
        "normalization_nu_n",
        {"insert": {"physicsParameters": "  nu_n = 8.4774d-3"}},
        "nu_n",
    ),
    (
        "normalization_Delta",
        {"insert": {"physicsParameters": "  Delta = 1.0d-3"}},
        "Delta",
    ),
    (
        "normalization_alpha",
        {"insert": {"physicsParameters": "  alpha = 2.0d+0"}},
        "alpha",
    ),
    (
        "workflow_transport_matrix",
        {"replace": ("RHSMode = 1", "RHSMode = 2")},
        "RHSMode",
    ),
    (
        "workflow_monoenergetic",
        {"replace": ("RHSMode = 1", "RHSMode = 3\n  nuPrime = 1.0d+0")},
        "RHSMode",
    ),
    (
        "geometry_namelist_spectrum",
        {"replace": ("geometryScheme = 4", "geometryScheme = 13")},
        "geometryScheme",
    ),
    (
        "geometry_unknown_family",
        {"replace": ("geometryScheme = 4", "geometryScheme = 10")},
        "geometryScheme",
    ),
    (
        "geometry_modified_analytic_model",
        {"replace": ("geometryScheme = 4", "geometryScheme = 1\n  epsilon_t = 0.1d+0")},
        "epsilon_t",
    ),
    (
        "geometry_snapped_radial_surface",
        {
            "replace": (
                "geometryScheme = 4",
                'geometryScheme = 11\n  equilibriumFile = "missing.bc"\n  VMECRadialOption = 1',
            )
        },
        "VMECRadialOption",
    ),
    (
        "collisions_improved_sugama",
        {"replace": ("collisionOperator = 1", "collisionOperator = 3")},
        "collisionOperator",
    ),
    (
        "collisions_constraint_scheme",
        {"insert": {"physicsParameters": "  constraintScheme = 4"}},
        "constraintScheme",
    ),
    (
        "collisions_krook_operator",
        {"insert": {"physicsParameters": "  Krook = 0.1d+0"}},
        "Krook",
    ),
    (
        "physics_phi1",
        {"replace": ("includePhi1 = .false.", "includePhi1 = .true.")},
        "includePhi1",
    ),
    (
        "physics_magnetic_drifts",
        {"insert": {"physicsParameters": "  magneticDriftScheme = 1"}},
        "magneticDriftScheme",
    ),
    (
        "physics_full_er_trajectories",
        {
            "replace": (
                "Er = 0.0d+0",
                "Er = 0.5d+0\n  useDKESExBDrift = .false.",
            )
        },
        "useDKESExBDrift",
    ),
    (
        "physics_inductive_drive",
        {"insert": {"physicsParameters": "  EParallelHat = 0.01d+0"}},
        "EParallelHat",
    ),
    (
        "field_non_er_gradient_coordinate",
        {
            "insert": {
                "geometryParameters": "  inputRadialCoordinateForGradients = 0",
                "physicsParameters": "  dPhiHatdpsiHat = 0.3d+0",
            }
        },
        "dPhiHatdpsiHat",
    ),
    (
        "numerics_speed_grid",
        {"insert": {"otherNumericalParameters": "  xGridScheme = 2"}},
        "xGridScheme",
    ),
    (
        "numerics_speed_grid_weight",
        {"insert": {"otherNumericalParameters": "  xGrid_k = 1.0d+0"}},
        "xGrid_k",
    ),
    (
        "numerics_angular_stencil",
        {"insert": {"otherNumericalParameters": "  thetaDerivativeScheme = 0"}},
        "thetaDerivativeScheme",
    ),
    (
        "numerics_pitch_speed_ramp",
        {"replace": ("Nxi_for_x_option = 0", "Nxi_for_x_option = 3")},
        "Nxi_for_x_option",
    ),
    (
        "numerics_legendre_depth",
        {
            "replace": (
                "collisionOperator = 1",
                "collisionOperator = 0",
            ),
            "insert": {"resolutionParameters": "  NL = 2"},
        },
        "NL",
    ),
    (
        "geometry_missing_scheme",
        {"replace": ("  geometryScheme = 4\n", "")},
        "geometryScheme",
    ),
)


@pytest.mark.parametrize(
    ("edit", "key"),
    [(edit, key) for _name, edit, key in REFUSALS],
    ids=[name for name, _edit, _key in REFUSALS],
)
def test_an_unsupported_deck_refuses_and_names_the_key(
    tmp_path: Path, edit: dict, key: str
) -> None:
    """Every refusal names the namelist key and its value.

    A message that says only "unsupported" leaves the reader guessing which of
    forty namelist entries to change, and the whole point of refusing rather
    than silently substituting an adjacent model is that the reader can act on
    it.
    """
    deck = _deck_variant(tmp_path, key.lower(), **edit)
    with pytest.raises(CaseValidationError) as excinfo:
        case_from_sfincs_namelist(deck)
    message = str(excinfo.value)
    assert key in message, message
    assert "supplied" in message and "expected" in message, message


def test_a_deck_refusal_leaves_no_partial_case_file_behind(tmp_path: Path) -> None:
    deck = _deck_variant(tmp_path, "phi1", replace=("includePhi1 = .false.", "includePhi1 = .true."))
    destination = tmp_path / "case.toml"
    with pytest.raises(CaseValidationError):
        convert_sfincs_namelist(deck, destination)
    assert not destination.exists()


def test_the_inert_er_switches_do_not_refuse_a_zero_field_deck(tmp_path: Path) -> None:
    """Refusing must be reserved for options that change the answer.

    ``useDKESExBDrift``/``includeXDotTerm``/``includeElectricFieldTermInXiDot``
    only ever multiply the E_r terms (``drift_kinetic`` gates all three on
    ``has_er``), so at ``Er = 0`` they are inert -- and the checked-in decks all
    set them. Refusing there would make users edit working decks for nothing,
    which is the policy :func:`dkx.inputs.check_supported_options` already
    states.
    """
    deck = _deck_variant(
        tmp_path,
        "inert_switches",
        insert={
            "physicsParameters": "  useDKESExBDrift = .false.\n"
            "  includeXDotTerm = .true.\n  includeElectricFieldTermInXiDot = .true."
        },
    )
    assert case_from_sfincs_namelist(deck).physics.magnetic_drifts == "dkes"


def test_the_landed_namelist_guards_still_apply_at_convert_time(tmp_path: Path) -> None:
    """``dkx.inputs.check_supported_deck_options`` is reused, not reimplemented."""
    deck = _deck_variant(
        tmp_path,
        "temperature_equilibration",
        insert={"physicsParameters": "  includeTemperatureEquilibrationTerm = .true."},
    )
    with pytest.raises(CaseValidationError) as excinfo:
        case_from_sfincs_namelist(deck)
    assert "includeTemperatureEquilibrationTerm" in str(excinfo.value)


@pytest.mark.parametrize(
    "deck",
    [
        "pas_1species_PAS_noEr_tiny.input.namelist",
        "output_scheme1_tokamak_1species_tiny.input.namelist",
        "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist",
        "monoenergetic_PAS_tiny_scheme1.input.namelist",
    ],
)
def test_checked_in_decks_refuse_rather_than_convert_approximately(deck: str) -> None:
    """The narrow convertible subset is a fact about the native route, not a bug.

    Pinned so that widening the route (a case field for ``nu_n``, a
    ``VMECRadialOption`` choice, a configurable analytic model) is a deliberate
    change with a test to update, rather than something that silently starts
    accepting decks whose physics has quietly shifted.
    """
    with pytest.raises(CaseValidationError):
        case_from_sfincs_namelist(REF / deck)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_convert_is_registered_and_advertised() -> None:
    assert "convert" in cli._USER_COMMANDS
    assert "convert" in cli._KNOWN_COMMANDS
    # Without this, `dkx convert a b` falls through to the implicit namelist
    # path and fails on a file the user never named.
    assert cli._normalize_default_argv(["convert", "a", "b"]) == ["convert", "a", "b"]


def test_the_cli_writes_the_case_and_reports_it(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "case.toml"
    assert cli.main(["convert", str(PAS_DECK), str(destination)]) == 0
    assert destination.exists()
    out = capsys.readouterr().out
    assert "convert_pas_w7x_noEr_tiny" in out
    assert Case.from_file(destination).run.workflow == "profile"


def test_the_cli_reports_a_refusal_on_stderr_and_exits_non_zero(
    tmp_path: Path, capsys
) -> None:
    deck = _deck_variant(tmp_path, "nu_n_cli", insert={"physicsParameters": "  nu_n = 8.4774d-3"})
    assert cli.main(["convert", str(deck), str(tmp_path / "case.toml")]) == 2
    captured = capsys.readouterr()
    assert "dkx convert failed" in captured.err
    assert "nu_n" in captured.err
    assert not (tmp_path / "case.toml").exists()


def test_the_cli_requires_force_to_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "case.json"
    assert cli.main(["convert", str(PAS_DECK), str(destination)]) == 0
    assert cli.main(["convert", str(PAS_DECK), str(destination)]) == 2
    assert cli.main(["convert", str(PAS_DECK), str(destination), "--force"]) == 0


def test_the_cli_accepts_an_explicit_case_name(tmp_path: Path) -> None:
    destination = tmp_path / "named.toml"
    assert cli.main(["convert", str(PAS_DECK), str(destination), "--name", "w7x_probe"]) == 0
    assert Case.from_file(destination).name == "w7x_probe"
