"""Tests for the consolidated inputs namelist reader and Fortran-parity prints.

- Equivalence referee: inputs parsing of upstream Fortran example decks must
  agree with the existing ``sfincs_jax.namelist`` reader.
- Defaults: an empty deck must yield the readInput.F90/globalVariables.F90
  defaults.
- Golden prints: rendered stdout blocks must match
  ``reference-data-v2/quick_2species_FPCollisions_noEr/stdout.log``
  line-by-line (trailing whitespace normalized; timings re-rendered from the
  golden values).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sfincs_jax.namelist import read_sfincs_input as legacy_read_sfincs_input
from sfincs_jax import console as prints
from sfincs_jax.inputs import (
    load_sfincs_input,
    parse_sfincs_input_text,
    read_sfincs_input,
    sfincs_input_from_raw,
)

_EXAMPLES = Path("/Users/rogerio/local/sfincs/fortran/version3/examples")
_REFERENCE = Path("/Users/rogerio/local/reference-data-v2")

# The upstream geometryScheme4_2species_noEr_withPhi1 example directory has no
# input.namelist (only job scripts); use the equivalent Phi1 deck captured in
# reference-data-v2.
_EXAMPLE_DECKS = (
    _EXAMPLES / "quick_2species_FPCollisions_noEr" / "input.namelist",
    _EXAMPLES / "monoenergetic_geometryScheme1" / "input.namelist",
    _EXAMPLES / "tokamak_1species_FPCollisions_noEr" / "input.namelist",
    _REFERENCE / "geometryScheme4_2species_noEr_withPhi1InDKE" / "input.namelist",
)

_GOLDEN_LOG = _REFERENCE / "quick_2species_FPCollisions_noEr" / "stdout.log"

_FLOAT_RE = re.compile(r"[+-]?\d+\.\d+(?:E[+-]\d+)?")


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"reference file not available: {path}")
    return path


# ---------------------------------------------------------------------------
# Equivalence referee: inputs vs the existing sfincs_jax.namelist reader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck", _EXAMPLE_DECKS, ids=lambda p: p.parent.name)
def test_inputs_parser_matches_legacy_reader(deck: Path) -> None:
    _require(deck)
    legacy = legacy_read_sfincs_input(deck)
    new_raw = read_sfincs_input(deck)

    # Strongest referee: the raw group/index structures are identical.
    assert new_raw.groups == legacy.groups
    assert new_raw.indexed == legacy.indexed

    # Typed fields agree with the legacy raw values where the deck sets them.
    inp = sfincs_input_from_raw(new_raw, validate=False)
    geom = legacy.group("geometryParameters")
    if "GEOMETRYSCHEME" in geom:
        assert inp.geometry.geometry_scheme == int(geom["GEOMETRYSCHEME"])
    species = legacy.group("speciesParameters")
    for key, attr in (("ZS", "z_s"), ("MHATS", "m_hats"), ("NHATS", "n_hats"), ("THATS", "t_hats")):
        if key in species:
            vals = species[key] if isinstance(species[key], list) else [species[key]]
            assert getattr(inp.species, attr) == tuple(float(v) for v in vals)
    physics = legacy.group("physicsParameters")
    for key, attr in (
        ("DELTA", "delta"),
        ("ALPHA", "alpha"),
        ("NU_N", "nu_n"),
        ("NUPRIME", "nu_prime"),
        ("ER", "er"),
    ):
        if key in physics:
            assert getattr(inp.physics, attr) == pytest.approx(float(physics[key]), abs=0.0)
    for key, attr in (("COLLISIONOPERATOR", "collision_operator"), ("CONSTRAINTSCHEME", "constraint_scheme")):
        if key in physics:
            assert getattr(inp.physics, attr) == int(physics[key])
    for key, attr in (
        ("INCLUDEXDOTTERM", "include_x_dot_term"),
        ("USEDKESEXBDRIFT", "use_dkes_exb_drift"),
        ("INCLUDEPHI1", "include_phi1"),
    ):
        if key in physics:
            assert getattr(inp.physics, attr) is bool(physics[key])
    res = legacy.group("resolutionParameters")
    for key, attr in (("NTHETA", "n_theta"), ("NZETA", "n_zeta"), ("NXI", "n_xi"), ("NL", "n_l"), ("NX", "n_x")):
        if key in res:
            assert getattr(inp.resolution, attr) == int(res[key])
    if "SOLVERTOLERANCE" in res:
        assert inp.resolution.solver_tolerance == float(res["SOLVERTOLERANCE"])
    other = legacy.group("otherNumericalParameters")
    if "NXI_FOR_X_OPTION" in other:
        assert inp.other.n_xi_for_x_option == int(other["NXI_FOR_X_OPTION"])
    pre = legacy.group("preconditionerOptions")
    if "PRECONDITIONER_X" in pre:
        assert inp.preconditioner.preconditioner_x == int(pre["PRECONDITIONER_X"])
    general = legacy.group("general")
    if "RHSMODE" in general:
        assert inp.general.rhs_mode == int(general["RHSMODE"])
    # export_f stays reachable through the raw fallback.
    assert inp.export_f == legacy.group("export_f")


# ---------------------------------------------------------------------------
# Defaults: readInput.F90 + globalVariables.F90.
# ---------------------------------------------------------------------------

_EMPTY_DECK = "\n".join(
    f"&{group}\n/"
    for group in (
        "general",
        "geometryParameters",
        "speciesParameters",
        "physicsParameters",
        "resolutionParameters",
        "otherNumericalParameters",
        "preconditionerOptions",
        "export_f",
    )
)


def test_empty_deck_yields_fortran_defaults() -> None:
    inp = sfincs_input_from_raw(parse_sfincs_input_text(_EMPTY_DECK))
    assert inp.resolution.n_theta == 15  # Ntheta, globalVariables.F90:186
    assert inp.resolution.n_zeta == 15  # Nzeta, gV:187
    assert inp.resolution.n_xi == 16  # Nxi, gV:188
    assert inp.resolution.n_l == 4  # NL, gV:189
    assert inp.resolution.n_x == 5  # Nx, gV:190
    assert inp.resolution.x_max == 5.0  # xMax, gV:192
    assert inp.resolution.solver_tolerance == 1.0e-6  # gV:193
    assert inp.physics.delta == 4.5694e-3  # Delta, gV:133
    assert inp.physics.alpha == 1.0  # alpha, gV:134
    assert inp.physics.nu_n == 8.330e-3  # nu_n, gV:135
    assert inp.general.rhs_mode == 1  # RHSMode, gV:33
    assert inp.geometry.geometry_scheme == 1  # geometryScheme, gV:53
    assert inp.physics.collision_operator == 0  # gV:140
    assert inp.other.theta_derivative_scheme == 2  # gV:171
    assert inp.other.x_grid_scheme == 5  # xGridScheme, gV:183
    assert inp.other.n_xi_for_x_option == 1  # gV:217
    assert inp.physics.constraint_scheme == -1  # gV:214
    assert inp.preconditioner.preconditioner_x == 1  # gV:208
    assert inp.preconditioner.reuse_preconditioner is True  # gV:211
    assert inp.species.n_species == 0  # Zs uninitialized sentinel, readInput.F90:103
    assert inp.warnings == ()


def test_rhsmode3_forcing_matches_validateinput() -> None:
    deck = (
        "&general\n  RHSMode = 3\n/\n"
        "&physicsParameters\n  nuPrime = 1.0\n/\n"
        "&resolutionParameters\n  Ntheta = 15\n  Nx = 5\n/\n"
    )
    inp = sfincs_input_from_raw(parse_sfincs_input_text(deck))
    assert inp.resolution.n_x == 1  # forced, validateInput.F90:91
    assert inp.other.n_xi_for_x_option == 0  # validateInput.F90:57
    assert inp.physics.collision_operator == 1  # pitch-angle scattering, validateInput.F90:163
    assert inp.physics.use_dkes_exb_drift is True  # validateInput.F90:103
    assert inp.physics.include_x_dot_term is False
    assert inp.physics.include_electric_field_term_in_xi_dot is False
    assert len(inp.warnings) >= 5

    with pytest.raises(ValueError, match="nuPrime"):
        sfincs_input_from_raw(
            parse_sfincs_input_text("&general\n  RHSMode = 3\n/\n&physicsParameters\n  nuPrime = 0.0\n/\n")
        )
    with pytest.raises(ValueError, match="Ntheta"):
        sfincs_input_from_raw(parse_sfincs_input_text("&resolutionParameters\n  Ntheta = 3\n/\n"))
    with pytest.raises(ValueError, match="RHSMode"):
        sfincs_input_from_raw(parse_sfincs_input_text("&general\n  RHSMode = 0\n/\n"))


def test_out_of_range_option_values_are_validation_errors() -> None:
    """RHSMode 4/5 and out-of-range option values raise ValueError at load, not legacy routes."""
    for rhs_mode in (4, 5):
        with pytest.raises(ValueError, match=r"jax\.grad"):
            sfincs_input_from_raw(parse_sfincs_input_text(f"&general\n  RHSMode = {rhs_mode}\n/\n"))
    with pytest.raises(ValueError, match="collisionOperator must be 0 .* or 1"):
        sfincs_input_from_raw(
            parse_sfincs_input_text("&physicsParameters\n  collisionOperator = 2\n/\n")
        )
    with pytest.raises(ValueError, match="constraintScheme must be -1"):
        sfincs_input_from_raw(
            parse_sfincs_input_text("&physicsParameters\n  constraintScheme = 5\n/\n")
        )
    with pytest.raises(ValueError, match="quasineutralityOption must be 1 .* or 2"):
        sfincs_input_from_raw(
            parse_sfincs_input_text(
                "&physicsParameters\n  includePhi1 = .true.\n  quasineutralityOption = 3\n/\n"
            )
        )
    # readExternalPhi1 bypasses quasineutrality entirely, so its option is inert then.
    inp = sfincs_input_from_raw(
        parse_sfincs_input_text(
            "&physicsParameters\n  includePhi1 = .true.\n  readExternalPhi1 = .true.\n"
            "  quasineutralityOption = 3\n/\n"
        )
    )
    assert inp.physics.quasineutrality_option == 3


def test_monoenergetic_example_deck_validates_with_overrides() -> None:
    deck = _require(_EXAMPLE_DECKS[1])
    inp = load_sfincs_input(deck)
    assert inp.general.rhs_mode == 3
    assert inp.resolution.n_x == 1
    assert inp.other.n_xi_for_x_option == 0  # deck leaves the default 1; forcing applies
    assert inp.physics.collision_operator == 1
    assert any("Nxi_for_x_option" in w for w in inp.warnings)


# ---------------------------------------------------------------------------
# Golden print parity vs reference-data-v2 stdout.log.
# ---------------------------------------------------------------------------


def _golden_lines() -> list[str]:
    return _require(_GOLDEN_LOG).read_text().splitlines()


def _floats(line: str) -> list[float]:
    return [float(tok) for tok in _FLOAT_RE.findall(line)]


def _assert_block_equal(rendered, golden) -> None:
    assert [ln.rstrip() for ln in rendered] == [ln.rstrip() for ln in golden]


def test_golden_banner_and_namelist_lines() -> None:
    golden = _golden_lines()
    _assert_block_equal(prints.banner_lines(n_procs=1), golden[:5])
    _assert_block_equal(prints.namelist_read_lines(input_name="input.namelist"), golden[5:13])


def test_golden_physics_parameter_block() -> None:
    golden = _golden_lines()
    start = golden.index(" ---- Physics parameters: ----")
    block = golden[start : start + 6]
    rendered = prints.physics_parameter_lines(
        n_species=2,
        delta=_floats(block[2])[0],
        alpha=_floats(block[3])[0],
        nu_n=_floats(block[4])[0],
        include_phi1=False,
    )
    _assert_block_equal(rendered, block)


def test_golden_grid_summary_block() -> None:
    golden = _golden_lines()
    start = golden.index(" ---- Numerical parameters: ----")
    end = next(i for i in range(start, len(golden)) if golden[i].startswith(" The matrix is"))
    block = golden[start : end + 1]
    x_line = next(ln for ln in block if ln.startswith(" x:"))
    rendered = prints.grid_summary_lines(
        n_theta=5,
        n_zeta=7,
        n_xi=8,
        n_l=4,
        n_x=5,
        solver_tolerance=1.0e-6,
        theta_derivative_scheme=2,
        zeta_derivative_scheme=2,
        use_iterative_linear_solver=True,
        n_xi_for_x_option=0,
        x=_floats(x_line),
        n_xi_for_x=[8] * 5,
        min_x_for_l=[1] * 8,
        matrix_size=2804,
    )
    _assert_block_equal(rendered, block)


def test_golden_species_results_block() -> None:
    golden = _golden_lines()
    start = golden.index(" Results for species            1 :")
    end = next(i for i in range(start, len(golden)) if golden[i].startswith(" FSABjHat"))
    block = golden[start : end + 1]

    keys = (
        "FSADensityPerturbation",
        "FSABFlow",
        "Mach",  # two values on one line
        "FSAPressurePerturbation",
        "NTV",
        "particleFlux_vm0_psiHat",
        "particleFlux_vm_psiHat",
        "classicalParticleFlux",
        "classicalHeatFlux",
        "momentumFlux_vm0_psiHat",
        "momentumFlux_vm_psiHat",
        "heatFlux_vm0_psiHat",
        "heatFlux_vm_psiHat",
        "particleSource",
        "heatSource",
    )
    species_results = []
    per_species = len(keys) + 1  # +1 for the "Results for species" line
    for s in range(2):
        values: dict[str, float] = {}
        for j, key in enumerate(keys):
            nums = _floats(block[s * per_species + 1 + j])
            if key == "Mach":
                values["MachMax"], values["MachMin"] = nums
            else:
                values[key] = nums[0]
        species_results.append(values)
    fsab_j_hat = _floats(block[-1])[0]

    rendered = prints.species_results_lines(
        species_results=species_results,
        fsab_j_hat=fsab_j_hat,
        include_phi1=False,
        constraint_scheme=1,
    )
    _assert_block_equal(rendered, block)


def test_golden_solver_banners_and_goodbye() -> None:
    golden = _golden_lines()
    assert prints.entering_solver_line() in golden
    assert prints.main_solve_begin_line() in golden
    assert prints.goodbye_line() in golden
    # The timing value differs run-to-run; re-rendering the golden value must
    # reproduce the golden line exactly (format parity, value-independent).
    done = next(ln for ln in golden if ln.startswith(" Done with the main solve."))
    assert prints.main_solve_done_line(seconds=_floats(done)[0]).rstrip() == done.rstrip()


def test_fortran_real_field_spot_values() -> None:
    # gfortran list-directed doubles, verified against golden logs.
    assert prints.fortran_real_field(4.5694e-3) == "   4.5694000000000004E-003"
    assert prints.fortran_real_field(1.0) == "   1.0000000000000000     "
    assert prints.fortran_real_field(0.0) == "   0.0000000000000000     "
    assert prints.fortran_real_field(-17.885) == "  -17.885000000000002     "
    assert prints.fortran_real_field(0.87) == "  0.87000000000000000     "
    assert prints.fortran_real_field(1e-6) == "   9.9999999999999995E-007"
    assert prints.fortran_int_field(2804) == "        2804"
