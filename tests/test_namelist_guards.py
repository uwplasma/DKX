"""Unsupported-model guards on the SFINCS v3 namelist surface.

plan.md operating rule 11: where dkx does not implement a v3 option, reading a
deck that asks for it must RAISE rather than silently solve the adjacent model.
Each guarded key gets two tests -- the unsupported value is refused with the key
named in the message, and the value dkx *does* implement still runs -- plus a
regression that a defaults-only deck (and every checked-in upstream deck) is
completely unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dkx.config import CaseValidationError
from dkx.inputs import SfincsInput, parse_sfincs_input_text, sfincs_input_from_raw
from dkx.namelist import parse_sfincs_input_text as parse_raw_deck
from dkx.namelist import read_sfincs_input

# The guarded keys, with the group a v3 deck would spell them in.
GUARDED_KEYS = (
    ("geometryParameters", "force0RadialCurrentInEquilibrium"),
    ("physicsParameters", "includeTemperatureEquilibrationTerm"),
    ("speciesParameters", "withNBIspec"),
    ("otherNumericalParameters", "ExBDerivativeSchemeTheta"),
    ("otherNumericalParameters", "ExBDerivativeSchemeZeta"),
    ("geometryParameters", "EParallelHatSpec_bcdatFile"),
)

_BASE_DECK = """
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
{geometryParameters}/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1
  THats = 1
{speciesParameters}/
&physicsParameters
  collisionOperator = 1
{physicsParameters}/
&resolutionParameters
  Ntheta = 5
  Nzeta = 1
  Nxi = 4
  Nx = 3
{resolutionParameters}/
&otherNumericalParameters
{otherNumericalParameters}/
&preconditionerOptions
/
&export_f
/
"""


def deck(**extra: str) -> str:
    """The minimal well-formed deck, with extra lines injected per group."""
    groups = {
        "geometryParameters": "",
        "speciesParameters": "",
        "physicsParameters": "",
        "resolutionParameters": "",
        "otherNumericalParameters": "",
    }
    for group, lines in extra.items():
        groups[group] = "".join(f"  {line}\n" for line in lines.splitlines() if line.strip())
    return _BASE_DECK.format(**groups)


def load_typed(text: str) -> SfincsInput:
    """The typed deck-read path (``load_sfincs_input`` without touching disk)."""
    return sfincs_input_from_raw(parse_sfincs_input_text(text))


# --------------------------------------------------------------------------
# Defaults are never touched
# --------------------------------------------------------------------------


def test_default_deck_is_unaffected_by_the_guards() -> None:
    """A deck that names none of the guarded keys reads exactly as before."""
    text = deck()
    raw = parse_raw_deck(text)
    assert raw.group("general")["RHSMODE"] == 1
    typed = load_typed(text)
    assert typed.physics.include_temperature_equilibration_term is False
    assert typed.species.with_nbi_spec is False
    assert typed.other.exb_derivative_scheme_theta == 0
    assert typed.other.exb_derivative_scheme_zeta == 0


def test_every_checked_in_deck_still_reads() -> None:
    """No deck shipped with dkx sets a guarded key to an unsupported value."""
    root = Path(__file__).resolve().parents[1]
    decks = sorted(
        p
        for p in root.rglob("*.namelist")
        if ".claude" not in p.parts and "node_modules" not in p.parts
    )
    assert len(decks) > 30, "expected the checked-in upstream deck corpus"
    for path in decks:
        read_sfincs_input(path)


# --------------------------------------------------------------------------
# force0RadialCurrentInEquilibrium
# --------------------------------------------------------------------------


def test_force0_radial_current_false_is_refused() -> None:
    text = deck(geometryParameters="force0RadialCurrentInEquilibrium = .false.")
    for read in (parse_raw_deck, load_typed):
        with pytest.raises(CaseValidationError) as excinfo:
            read(text)
        message = str(excinfo.value)
        assert "force0RadialCurrentInEquilibrium" in message
        assert ".true." in message
        assert "radial current" in message


def test_force0_radial_current_true_still_reads() -> None:
    text = deck(geometryParameters="force0RadialCurrentInEquilibrium = .true.")
    assert parse_raw_deck(text).group("geometryParameters")["GEOMETRYSCHEME"] == 1
    assert load_typed(text).geometry.geometry_scheme == 1


# --------------------------------------------------------------------------
# includeTemperatureEquilibrationTerm
# --------------------------------------------------------------------------


def test_include_temperature_equilibration_term_true_is_refused() -> None:
    text = deck(physicsParameters="includeTemperatureEquilibrationTerm = .true.")
    for read in (parse_raw_deck, load_typed):
        with pytest.raises(CaseValidationError) as excinfo:
            read(text)
        message = str(excinfo.value)
        assert "includeTemperatureEquilibrationTerm" in message
        assert ".false." in message
        assert "collision operator" in message


def test_include_temperature_equilibration_term_false_still_reads() -> None:
    text = deck(physicsParameters="includeTemperatureEquilibrationTerm = .false.")
    assert load_typed(text).physics.include_temperature_equilibration_term is False


def test_include_temperature_equilibration_term_is_allowed_for_rhsmode3() -> None:
    """validateInput.F90:166-176 silently disables the term for RHSMode=3."""
    text = deck(physicsParameters="includeTemperatureEquilibrationTerm = .true.").replace(
        "RHSMode = 1", "RHSMode = 3\n  ambipolarSolve = .false."
    ).replace("&physicsParameters\n", "&physicsParameters\n  nuPrime = 0.1\n")
    typed = load_typed(text)
    assert typed.general.rhs_mode == 3
    assert typed.physics.include_temperature_equilibration_term is False


# --------------------------------------------------------------------------
# withNBIspec
# --------------------------------------------------------------------------


def test_with_nbi_spec_with_solved_phi1_is_refused() -> None:
    text = deck(
        speciesParameters="withNBIspec = .true.\nNBIspecZ = 1.0\nNBIspecNHat = 0.1",
        physicsParameters="includePhi1 = .true.",
    )
    for read in (parse_raw_deck, load_typed):
        with pytest.raises(CaseValidationError) as excinfo:
            read(text)
        message = str(excinfo.value)
        assert "withNBIspec" in message
        assert ".false." in message
        assert "quasineutrality" in message


def test_with_nbi_spec_false_still_reads_with_phi1() -> None:
    text = deck(
        speciesParameters="withNBIspec = .false.",
        physicsParameters="includePhi1 = .true.",
    )
    typed = load_typed(text)
    assert typed.species.with_nbi_spec is False
    assert typed.physics.include_phi1 is True


def test_with_nbi_spec_without_phi1_is_accepted() -> None:
    """SFINCS itself reports the NBI species has no impact (validateInput.F90:583)."""
    text = deck(speciesParameters="withNBIspec = .true.\nNBIspecNHat = 0.1")
    assert load_typed(text).species.with_nbi_spec is True


# --------------------------------------------------------------------------
# ExBDerivativeSchemeTheta / ExBDerivativeSchemeZeta
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ExBDerivativeSchemeTheta", "ExBDerivativeSchemeZeta"])
def test_upwinded_exb_derivative_scheme_is_refused(key: str) -> None:
    text = deck(otherNumericalParameters=f"{key} = 2")
    for read in (parse_raw_deck, load_typed):
        with pytest.raises(CaseValidationError) as excinfo:
            read(text)
        message = str(excinfo.value)
        assert key in message
        assert "upwinded" in message


@pytest.mark.parametrize("key", ["ExBDerivativeSchemeTheta", "ExBDerivativeSchemeZeta"])
def test_centered_exb_derivative_scheme_still_reads(key: str) -> None:
    text = deck(otherNumericalParameters=f"{key} = 0")
    typed = load_typed(text)
    assert typed.other.exb_derivative_scheme_theta == 0
    assert typed.other.exb_derivative_scheme_zeta == 0


# --------------------------------------------------------------------------
# EParallelHatSpec_bcdatFile
# --------------------------------------------------------------------------


def test_eparallel_bcdat_file_is_refused() -> None:
    text = deck(geometryParameters='EParallelHatSpec_bcdatFile = "bcdata.dat"')
    for read in (parse_raw_deck, load_typed):
        with pytest.raises(CaseValidationError) as excinfo:
            read(text)
        message = str(excinfo.value)
        assert "EParallelHatSpec_bcdatFile" in message
        assert "bcdata" in message


def test_empty_eparallel_bcdat_file_still_reads() -> None:
    text = deck(geometryParameters='EParallelHatSpec_bcdatFile = ""')
    assert load_typed(text).geometry.geometry_scheme == 1


# --------------------------------------------------------------------------
# Options deliberately NOT guarded
# --------------------------------------------------------------------------


def test_inert_options_are_not_guarded() -> None:
    """Keys that leave dkx's converged answer alone must never be refused.

    ``include_fDivVE_term`` is commented out in populateMatrix.F90 (v3 ignores
    it too, and three checked-in decks name it); ``forceOddNthetaAndNzeta``
    picks a grid size, not a model term; the rest is Fortran-only I/O, PETSc
    plumbing, the preconditioner, and the Rosenbluth auxiliary grid that dkx
    replaces with direct quadrature.
    """
    text = deck(
        physicsParameters="include_fDivVE_term = .true.",
        resolutionParameters=(
            "forceOddNthetaAndNzeta = .false.\n"
            "NxPotentialsPerVth = 40.0\n"
            "solverTolerance = 1e-7"
        ),
        otherNumericalParameters=(
            "useIterativeLinearSolver = .false.\n"
            "PETSCPreallocationStrategy = 0\n"
            "whichParallelSolverToFactorPreconditioner = 2\n"
            "xPotentialsGridScheme = 2"
        ),
    )
    parse_raw_deck(text)
    typed = load_typed(text)
    assert typed.physics.include_f_div_ve_term is True
    assert typed.resolution.force_odd_ntheta_and_nzeta is False


def test_guarded_error_is_a_value_error() -> None:
    """The refusal reuses the codebase's CaseValidationError (a ValueError)."""
    assert issubclass(CaseValidationError, ValueError)
    with pytest.raises(ValueError):
        parse_raw_deck(deck(physicsParameters="includeTemperatureEquilibrationTerm = .true."))


def test_every_guarded_key_is_named_by_its_own_message() -> None:
    """No guard borrows another key's message."""
    values = {
        "force0RadialCurrentInEquilibrium": ".false.",
        "includeTemperatureEquilibrationTerm": ".true.",
        "withNBIspec": ".true.",
        "ExBDerivativeSchemeTheta": "3",
        "ExBDerivativeSchemeZeta": "3",
        "EParallelHatSpec_bcdatFile": '"bcdata.dat"',
    }
    for group, key in GUARDED_KEYS:
        extra = {group: f"{key} = {values[key]}"}
        if key == "withNBIspec":
            extra["physicsParameters"] = "includePhi1 = .true."
        with pytest.raises(CaseValidationError) as excinfo:
            parse_raw_deck(deck(**extra))
        assert excinfo.value.path == f"{group}.{key}"


# ---------------------------------------------------------------------------
# Defaults inherited by decks that omit a key
# ---------------------------------------------------------------------------


def test_omitted_trajectory_switches_inherit_the_fortran_defaults() -> None:
    """A deck may omit a key; DKX must then assume what SFINCS assumes.

    ``includeXDotTerm`` and ``includeElectricFieldTermInXiDot`` are both
    ``.true.`` in globalVariables.F90:144-145. DKX defaulted them to False, so
    a compact deck that omitted them and set a finite ``Er`` silently lost both
    E_r trajectory terms while SFINCS kept them -- a wrong answer, not a
    missing feature. ``useDKESExBDrift`` genuinely is ``.false.`` upstream, so
    it is pinned here too to stop a well-meant sweep "fixing" it to match the
    other two.
    """
    from dkx.namelist import parse_sfincs_input_text

    deck = """
&general
/
&geometryParameters
  geometryScheme = 1
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
/
&physicsParameters
  Er = 1.0d+0
/
&resolutionParameters
  Ntheta = 5
  Nzeta = 1
  Nxi = 4
  Nx = 3
/
&otherNumericalParameters
/
&preconditionerOptions
/
&export_f
/
"""
    nml = parse_sfincs_input_text(deck)
    phys = nml.groups.get("physicsparameters", nml.groups.get("physicsParameters", {}))
    assert "includexdotterm" not in {k.lower() for k in phys}

    from dkx.input_compat import config_bool

    assert config_bool(nml, ("physicsParameters",), "includeXDotTerm", True) is True
    assert (
        config_bool(nml, ("physicsParameters",), "includeElectricFieldTermInXiDot", True)
        is True
    )
    assert config_bool(nml, ("physicsParameters",), "useDKESExBDrift", False) is False
