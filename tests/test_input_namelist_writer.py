"""SfincsInput namelist writer: round-trip fidelity and the in-memory run path.

The serializer contract: ``load_sfincs_input`` of :meth:`SfincsInput.to_namelist`
text reproduces every typed section field (tuples included), the ``export_f``
group, untyped raw keys (legacy aliases such as ``JGboozer_file``), and the
rank-2 ``boozer_bmnc(m,n)`` spectra — across the geometry families (scheme 1
analytic, 11/12 Boozer ``.bc``, 13 namelist spectrum, 4/5) and multi-species
Fokker-Planck decks.  The in-memory entry points must agree exactly with the
path-based runs.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from dkx.inputs import (
    SfincsInput,
    load_sfincs_input,
    parse_sfincs_input_text,
    sfincs_input_from_raw,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Deck coverage: scheme 1 (analytic tokamak + RHSMode=3 monoenergetic with
# validateInput hard overrides), scheme 11 (HSX .bc, multi-species FP), scheme
# 12 (non-stellarator-symmetric .bc), scheme 13 (namelist boozer_bmnc(m,n)
# spectrum), scheme 4, scheme 5 (3-species), and a legacy JGboozer_file alias.
ROUND_TRIP_DECKS = (
    "tests/reduced_inputs/tokamak_1species_PASCollisions_noEr_Nx1.input.namelist",
    "tests/reduced_inputs/monoenergetic_geometryScheme1.input.namelist",
    "tests/reduced_inputs/HSX_FPCollisions_DKESTrajectories.input.namelist",
    "tests/ref/pas_1species_PAS_noEr_tiny_scheme12.input.namelist",
    "tests/ref/pas_1species_PAS_noEr_tiny_scheme13.input.namelist",
    "tests/reduced_inputs/quick_2species_FPCollisions_noEr.input.namelist",
    "tests/reduced_inputs/geometryScheme5_3species_loRes.input.namelist",
    "tests/ref/multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist",
)

TYPED_SECTIONS = ("general", "geometry", "species", "physics", "resolution", "other", "preconditioner")


def _assert_typed_sections_equal(a: SfincsInput, b: SfincsInput) -> None:
    for attr in TYPED_SECTIONS:
        sec_a, sec_b = getattr(a, attr), getattr(b, attr)
        for fld in fields(type(sec_a)):
            assert getattr(sec_a, fld.name) == getattr(sec_b, fld.name), f"{attr}.{fld.name}"
    assert dict(a.export_f) == dict(b.export_f)


@pytest.mark.parametrize("include_defaults", [False, True], ids=["compact", "full"])
@pytest.mark.parametrize("deck", ROUND_TRIP_DECKS, ids=[Path(d).stem for d in ROUND_TRIP_DECKS])
def test_to_namelist_round_trips_every_typed_field(deck: str, include_defaults: bool) -> None:
    original = load_sfincs_input(REPO_ROOT / deck)
    text = original.to_namelist(include_defaults=include_defaults)
    reparsed = sfincs_input_from_raw(parse_sfincs_input_text(text))
    _assert_typed_sections_equal(original, reparsed)


def test_to_namelist_preserves_untyped_alias_keys_and_boozer_spectrum() -> None:
    # Legacy Boozer alias (JGboozer_file) lives only in ``raw``; the writer
    # must keep it or the written deck loses its equilibrium.
    alias = load_sfincs_input(REPO_ROOT / "tests/ref/multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist")
    alias_rt = parse_sfincs_input_text(alias.to_namelist())
    assert "JGBOOZER_FILE" in alias_rt.group("geometryParameters")
    assert alias_rt.group("geometryParameters")["JGBOOZER_FILE"] == alias.raw.group("geometryParameters")["JGBOOZER_FILE"]

    # geometryScheme=13 rank-2 boozer_bmnc(m,n) spectrum survives verbatim.
    spectrum = load_sfincs_input(REPO_ROOT / "tests/ref/pas_1species_PAS_noEr_tiny_scheme13.input.namelist")
    spectrum_rt = parse_sfincs_input_text(spectrum.to_namelist())
    assert spectrum_rt.indexed["geometryparameters"]["BOOZER_BMNC"] == (
        spectrum.raw.indexed["geometryparameters"]["BOOZER_BMNC"]
    )


def test_write_creates_a_loadable_file(tmp_path: Path) -> None:
    original = load_sfincs_input(
        REPO_ROOT / "tests/reduced_inputs/tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    )
    out = original.write(tmp_path / "input.namelist")
    assert out == tmp_path / "input.namelist"
    _assert_typed_sections_equal(original, load_sfincs_input(out))


def test_compact_deck_omits_defaults_but_keeps_the_geometry_dispatch_key() -> None:
    inp = SfincsInput.from_params(geometryScheme=1, Ntheta=17)
    text = inp.to_namelist()
    assert "Ntheta = 17" in text
    assert "geometryScheme = 1" in text  # structural dispatch key, always written
    assert "Nzeta" not in text  # Fortran default: omitted by the compact writer
    full = inp.to_namelist(include_defaults=True)
    assert "Nzeta = 15" in full
    assert "solverTolerance" in full


def test_from_params_flat_fortran_names_case_insensitive() -> None:
    inp = SfincsInput.from_params(
        geometryScheme=1, ntheta=17, NZETA=1, equilibriumFile="wout.nc",
        Zs=[1.0, 6.0], mHats=(1.0, 6.0),
    )  # fmt: skip
    assert inp.resolution.n_theta == 17
    assert inp.resolution.n_zeta == 1
    assert inp.geometry.equilibrium_file == "wout.nc"
    assert inp.species.z_s == (1.0, 6.0)
    assert inp.species.m_hats == (1.0, 6.0)

    with pytest.raises(ValueError, match="Unknown SFINCS input parameter"):
        SfincsInput.from_params(NoSuchKnob=3)

    with pytest.raises(ValueError, match="Ntheta"):
        SfincsInput.from_params(Ntheta=3)  # validation runs by default
    assert SfincsInput.from_params(Ntheta=3, validate=False).resolution.n_theta == 3


def test_from_params_round_trips_through_the_writer() -> None:
    inp = SfincsInput.from_params(
        geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, iota=1.31,
        epsilon_t=0.1, epsilon_h=0.0, psiAHat=0.045, aHat=0.1,
        Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[0.5],
        dNHatdrHats=[-6.0], dTHatdrHats=[-3.0],
        nu_n=8.4774e-3, collisionOperator=1,
        Ntheta=9, Nzeta=1, Nxi=6, NL=2, Nx=4, solverTolerance=1e-10,
        Nxi_for_x_option=0,
    )  # fmt: skip
    for include_defaults in (False, True):
        reparsed = sfincs_input_from_raw(
            parse_sfincs_input_text(inp.to_namelist(include_defaults=include_defaults))
        )
        _assert_typed_sections_equal(inp, reparsed)


def test_solver_options_kwargs_match_the_solve_signature() -> None:
    """Every SolverOptions knob must map onto a real dkx.solve.solve keyword."""
    from dkx.api import SolverOptions
    from dkx.solve import solve

    kwargs = SolverOptions().solve_kwargs()
    parameters = inspect.signature(solve).parameters
    unknown = set(kwargs) - set(parameters)
    assert unknown == set(), f"solve() has no keywords {sorted(unknown)}"
    assert "cores" not in kwargs  # threadpool width cannot act post-import
    # Defaults agree with solve()'s own defaults, so SolverOptions() is inert.
    for name, value in kwargs.items():
        if name == "tier1_memory_budget_gb":
            assert value is None
            continue
        assert parameters[name].default == value, name


def test_run_profile_in_memory_matches_path_run() -> None:
    deck = REPO_ROOT / "tests/reduced_inputs/tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    from dkx.run import run_profile

    by_path = run_profile(deck, emit=None)
    by_object = run_profile(load_sfincs_input(deck), emit=None)
    assert set(by_path.moments) == set(by_object.moments)
    for key in by_path.moments:
        np.testing.assert_array_equal(by_path.moments[key], by_object.moments[key], err_msg=key)


def test_run_profile_detects_replaced_typed_sections() -> None:
    """dataclasses.replace on a loaded input must win over the stale raw deck."""
    deck = REPO_ROOT / "tests/reduced_inputs/tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    from dkx.run import run_profile

    base = load_sfincs_input(deck)
    modified = replace(base, resolution=replace(base.resolution, n_theta=13))
    run = run_profile(modified, emit=None)
    assert run.operator.n_theta == 13
    assert run.input.resolution.n_theta == 13


def test_write_output_accepts_in_memory_input_and_stores_provenance(tmp_path: Path) -> None:
    import h5py

    from dkx.api import write_output

    inp = SfincsInput.from_params(
        geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, iota=1.31,
        epsilon_t=0.1, epsilon_h=0.0, psiAHat=0.045, aHat=0.1,
        Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[0.5],
        dNHatdrHats=[-6.0], dTHatdrHats=[-3.0],
        nu_n=8.4774e-3, collisionOperator=1,
        Ntheta=9, Nzeta=1, Nxi=6, NL=2, Nx=4, solverTolerance=1e-10,
        Nxi_for_x_option=0,
    )  # fmt: skip
    out = write_output(inp, tmp_path / "sfincsOutput.h5")
    assert out.exists()
    with h5py.File(out, "r") as f:
        stored = f["input.namelist"][()]
        text = stored.decode() if isinstance(stored, bytes) else str(stored)
    assert "SfincsInput.to_namelist" in text
    assert "geometryScheme = 1" in text

    with pytest.raises(ValueError, match="output_path is required"):
        write_output(inp)
