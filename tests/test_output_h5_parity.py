"""``sfincsOutput.h5`` parity against Fortran SFINCS v3 goldens, one deck per case.

Two routes write the file. The output writer (:func:`dkx.api.write_output`)
covers geometry schemes 1, 2, 4, 5 and 11 and the two-species quick example.
The canonical solve (:func:`dkx.run.run_profile`) covers constraintScheme 3/4
(Fokker-Planck sources whose x-shape differs from cs1 only in the L=0 rows,
``populateMatrix.F90`` 2915-2938), magneticDriftScheme 2-9 (recycled Krylov at
solverTolerance 1e-10 against Fortran MUMPS, measured differences below 2e-11),
stellarator-asymmetric VMEC (each Fourier sum gains its complementary-parity
partner), and the alternative speed grids and xDot schemes.

Every case embeds the deck verbatim, compares geometry and scalars at rounding
level, and compares the solution-derived moments at 1e-8. ``uHat`` uses 1e-8:
it is built from long transcendental reductions, O(1e-9) apart on scheme 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from dkx.api import write_output
from dkx.compare import compare_sfincs_outputs
from dkx.io import read_sfincs_h5

REF = Path(__file__).parent / "ref"

GRID = ["Nspecies", "Ntheta", "Nzeta", "Nxi", "NL", "Nx", "theta", "zeta", "x", "Nxi_for_x"]
PHI = ["dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN"]
RADIAL = ["psiAHat", "aHat", "psiHat", "psiN", "rHat", "rN"]
RADIAL_INPUT = ["inputRadialCoordinate", "inputRadialCoordinateForGradients"]
SWITCHES = ["includeXDotTerm", "includeElectricFieldTermInXiDot", "useDKESExBDrift"]
PHYSICS_FLAGS = [
    "force0RadialCurrentInEquilibrium", "includePhi1", "includePhi1InKineticEquation",
    "includePhi1InCollisionOperator", "includeTemperatureEquilibrationTerm",
    "include_fDivVE_Term", "withAdiabatic", "withNBIspec",
]  # fmt: skip
FLUX_FUNCTIONS = ["NPeriods", "B0OverBBar", "iota", "GHat", "IHat", "VPrimeHat", "FSABHat2"]
SPECIES = ["Zs", "mHats", "THats", "nHats"]
GRADIENTS = [
    "dnHatdrHat", "dnHatdrN", "dnHatdpsiN", "dnHatdpsiHat",
    "dTHatdrHat", "dTHatdrN", "dTHatdpsiN", "dTHatdpsiHat",
]  # fmt: skip
B_COMPONENTS = [
    "BHat_sub_psi", "dBHat_sub_psi_dtheta", "dBHat_sub_psi_dzeta",
    "BHat_sub_theta", "dBHat_sub_theta_dpsiHat", "dBHat_sub_theta_dzeta",
    "BHat_sub_zeta", "dBHat_sub_zeta_dpsiHat", "dBHat_sub_zeta_dtheta",
    "BHat_sup_theta", "dBHat_sup_theta_dpsiHat", "dBHat_sup_theta_dzeta",
    "BHat_sup_zeta", "dBHat_sup_zeta_dpsiHat", "dBHat_sup_zeta_dtheta",
]  # fmt: skip
B_FIELD = ["DHat", "BHat", "dBHatdpsiHat", "dBHatdtheta", "dBHatdzeta", "BDotCurlB", *B_COMPONENTS]
X_GRID = [
    "thetaDerivativeScheme", "zetaDerivativeScheme", "xGridScheme", "xGrid_k", "xMax",
    "xPotentialsGridScheme", "NxPotentialsPerVth", "pointAtX0", "Nxi_for_x_option",
    "solverTolerance",
]  # fmt: skip
SOLVER_ECHO = [
    "reusePreconditioner", "preconditioner_species", "preconditioner_x",
    "preconditioner_x_min_L", "preconditioner_xi", "preconditioner_theta",
    "preconditioner_zeta", "preconditioner_magnetic_drifts_max_L", "RHSMode",
    "useIterativeLinearSolver", "NIterations", "finished",
]  # fmt: skip
CLASSICAL = [
    f"classical{kind}FluxNoPhi1_{coord}"
    for kind in ("Particle", "Heat")
    for coord in ("psiHat", "psiN", "rHat", "rN")
]
TRANSPORT = [
    "FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat",
    "momentumFlux_vm_psiHat", "particleFlux_vm0_psiHat", "heatFlux_vm0_psiHat",
    "NTV", "FSABFlow_vs_x",
]  # fmt: skip
DRIVES = ["Delta", "alpha", "nu_n", "Er"]
GEOMETRY_ECHO = ["coordinateSystem", "rippleScale", "gpsiHatpsiHat", "diotadpsiHat", "EParallelHat"]

# Writer route, schemes 4, 5, 11: every dataset the writer echoes.
WRITER_FULL = [
    "integerToRepresentFalse", "integerToRepresentTrue", *GRID, "geometryScheme", *X_GRID,
    *DRIVES, *PHI, *RADIAL, *RADIAL_INPUT, *GEOMETRY_ECHO, "collisionOperator",
    "constraintScheme", *SOLVER_ECHO, *SWITCHES, "export_full_f", "export_delta_f",
    *PHYSICS_FLAGS, *CLASSICAL, *FLUX_FUNCTIONS, *SPECIES, *GRADIENTS, *B_FIELD,
]  # fmt: skip
# Canonical route: geometry, scalars and flags of a solved RHSMode=1 deck.
PROFILE_STRICT = [
    *GRID, "geometryScheme", *DRIVES, *PHI, *RADIAL, *RADIAL_INPUT, *GEOMETRY_ECHO,
    "collisionOperator", "constraintScheme", "RHSMode", *SWITCHES, *PHYSICS_FLAGS,
    *FLUX_FUNCTIONS, *SPECIES, *GRADIENTS, *B_FIELD,
]  # fmt: skip


@dataclass(frozen=True)
class Case:
    deck: str
    golden: str
    strict: list[str]
    strict_atol: float = 1e-12
    loose: dict[float, list[str]] = field(default_factory=dict)
    echoes_deck: bool = True
    rhs_mode: int | None = None
    species: int | None = None
    keys_must_exist: bool = False


WRITER_CASES = {
    "scheme1": Case(
        "output_scheme1_tokamak_1species_tiny", "output_scheme1_tokamak_1species_tiny",
        [*GRID, "geometryScheme", "RHSMode", *DRIVES, *PHI, *RADIAL, *RADIAL_INPUT,
         *GEOMETRY_ECHO, "collisionOperator", "constraintScheme", *SWITCHES, *PHYSICS_FLAGS,
         *FLUX_FUNCTIONS, *SPECIES, *GRADIENTS, *B_FIELD],
        loose={1e-8: ["uHat"]}, rhs_mode=1,
    ),
    "scheme2": Case(
        "output_scheme2_lhd_1species_tiny", "output_scheme2_lhd_1species_tiny",
        [*GRID, "geometryScheme", "RHSMode", *DRIVES, *RADIAL, "coordinateSystem",
         "gpsiHatpsiHat", "diotadpsiHat", "collisionOperator", "constraintScheme", *SWITCHES,
         *FLUX_FUNCTIONS, "DHat", "BHat", "dBHatdtheta", "dBHatdzeta"],
        loose={1e-8: ["uHat"]}, rhs_mode=2,
    ),
    "scheme4": Case("output_scheme4_1species_tiny", "output_scheme4_1species_tiny", WRITER_FULL,
                    loose={1e-8: ["uHat"]}),
    "scheme5": Case("output_scheme5_1species_tiny", "output_scheme5_1species_tiny",
                    [*WRITER_FULL, "uHat"]),
    "scheme11": Case("output_scheme11_1species_tiny", "output_scheme11_1species_tiny", WRITER_FULL,
                     loose={1e-8: ["uHat"]}),
    "scheme4_quick2species": Case(
        "quick_2species_FPCollisions_noEr", "output_scheme4_2species_quick",
        ["integerToRepresentFalse", "integerToRepresentTrue", *GRID, "geometryScheme",
         *X_GRID[:5], "Nxi_for_x_option", "solverTolerance", *DRIVES, *PHI, *RADIAL,
         *RADIAL_INPUT, *GEOMETRY_ECHO, "collisionOperator", *SWITCHES, "includePhi1",
         *FLUX_FUNCTIONS, *SPECIES, *GRADIENTS, *B_FIELD],
        loose={1e-8: ["uHat"]}, species=2, keys_must_exist=True,
    ),
}  # fmt: skip

_MAGDRIFT_STRICT = [
    *GRID, "geometryScheme", *DRIVES, "dPhiHatdpsiHat", *RADIAL, *RADIAL_INPUT,
    "coordinateSystem", "magneticDriftScheme", "collisionOperator", "constraintScheme",
    "RHSMode", *SWITCHES, "force0RadialCurrentInEquilibrium", "includePhi1", *FLUX_FUNCTIONS,
    *SPECIES, *B_FIELD, "diotadpsiHat",
]  # fmt: skip
_NONSTELSYM_STRICT = [
    *GRID[:9], "geometryScheme", *DRIVES, *RADIAL, "inputRadialCoordinate", "rippleScale",
    "gpsiHatpsiHat", "collisionOperator", "constraintScheme", "RHSMode", "NPeriods", "iota",
    "VPrimeHat", "FSABHat2", *SPECIES, "DHat", "BHat", "dBHatdpsiHat", "dBHatdtheta",
    "dBHatdzeta", *B_COMPONENTS,
]  # fmt: skip
_XGRID_STRICT = [
    *GRID, "pointAtX0", "xGridScheme", "xGrid_k", "xMax", *DRIVES, "dPhiHatdpsiHat",
    "collisionOperator", "constraintScheme", "RHSMode", "includeXDotTerm",
    "includeElectricFieldTermInXiDot", "BHat", "DHat",
]  # fmt: skip
# (deck tag, xGridScheme, xDotDerivativeScheme)
_XGRID_CASES = [
    ("xgrid2", 2, 0), ("xgrid3", 3, 0), ("xgrid3_xdot1", 3, 1), ("xgrid3_xdot2", 3, 2),
    ("xgrid3_xdot3", 3, 3), ("xgrid3_xdot9", 3, 9), ("xgrid4", 4, 0), ("xgrid4_xdot4", 4, 4),
    ("xgrid4_xdot5", 4, 5), ("xgrid4_xdot6", 4, 6), ("xgrid4_xdot7", 4, 7),
    ("xgrid4_xdot8", 4, 8), ("xgrid4_xdot10", 4, 10), ("xgrid5_xdotm2", 5, -2),
    ("xgrid5_xdot11", 5, 11), ("xgrid7", 7, 0), ("xgrid8", 8, 0),
]  # fmt: skip

PROFILE_CASES = {
    **{
        cs: Case(f"fp_1species_FPCollisions_noEr_tiny_{cs}", f"fp_1species_FPCollisions_noEr_tiny_{cs}",
                 PROFILE_STRICT, loose={1e-8: [*TRANSPORT, "uHat"]})
        for cs in ("cs3", "cs4")
    },
    **{
        f"magdrift{n}": Case(f"magdrift_1species_tiny_scheme{n}", f"magdrift_1species_tiny_scheme{n}",
                             _MAGDRIFT_STRICT, loose={5e-10: ["gpsiHatpsiHat"], 1e-8: TRANSPORT},
                             echoes_deck=False)
        for n in range(2, 10)
    },
    "nonstelsym_vmec_scheme5": Case(
        "nonstelsym_vmec_tiny_scheme5", "nonstelsym_vmec_tiny_scheme5", _NONSTELSYM_STRICT,
        strict_atol=1e-10, loose={1e-8: TRANSPORT},
    ),
    **{
        tag: Case(f"pas_1species_PAS_Er_tiny_{tag}", f"pas_1species_PAS_Er_tiny_{tag}",
                  _XGRID_STRICT, loose={1e-8: TRANSPORT})
        for tag, _, _ in _XGRID_CASES
    },
}  # fmt: skip
_X_GRID_SCHEME = {tag: scheme for tag, scheme, _ in _XGRID_CASES}


def _check(case: Case, out: Path, tag: str) -> None:
    golden = REF / f"{case.golden}.sfincsOutput.h5"
    assert golden.exists(), f"Missing Fortran fixture: {golden}"
    a = read_sfincs_h5(out)
    b = read_sfincs_h5(golden)
    if case.echoes_deck:
        assert a["input.namelist"] == b["input.namelist"]
    if case.keys_must_exist:
        for key in [*case.strict, *(k for keys in case.loose.values() for k in keys)]:
            assert key in a, f"Missing key in dkx output: {key}"
            assert key in b, f"Missing key in Fortran fixture: {key}"
    for atol, keys in [(case.strict_atol, case.strict), *case.loose.items()]:
        results = compare_sfincs_outputs(a_path=out, b_path=golden, keys=keys, rtol=0, atol=atol)
        bad = [(r.key, r.max_abs) for r in results if not r.ok]
        assert not bad, f"{tag}: mismatched keys at atol={atol}: {bad}"
    if case.rhs_mode is not None:
        assert int(np.asarray(a["RHSMode"])) == case.rhs_mode
    if case.species is not None:
        assert int(np.asarray(a["Nspecies"])) == case.species
    if tag in _X_GRID_SCHEME:
        assert int(a["xGridScheme"]) == _X_GRID_SCHEME[tag]


@pytest.mark.parametrize("tag", sorted(WRITER_CASES))
def test_writer_output_matches_fortran(tag: str, tmp_path: Path) -> None:
    case = WRITER_CASES[tag]
    out = tmp_path / "sfincsOutput.h5"
    write_output(REF / f"{case.deck}.input.namelist", out)
    _check(case, out, tag)


@pytest.mark.parametrize("tag", sorted(PROFILE_CASES))
def test_profile_output_matches_fortran(tag: str, tmp_path: Path, monkeypatch) -> None:
    from dkx.run import run_profile

    case = PROFILE_CASES[tag]
    if tag.startswith("magdrift"):
        monkeypatch.setenv("DKX_EQUILIBRIA_DIRS", str(REF))
    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(REF / f"{case.deck}.input.namelist", out_path=out, emit=None)
    assert run.solve_result.converged
    assert run.output_path is not None
    _check(case, out, tag)


def test_write_output_shapes(tmp_path: Path) -> None:
    """Scheme-4 output sizes; ``BHat`` reads back in Fortran layout (Nzeta, Ntheta)."""
    out = tmp_path / "sfincsOutput.h5"
    write_output(REF / "output_scheme4_1species_tiny.input.namelist", out)
    data = read_sfincs_h5(out)
    sizes = {"Nspecies": 1, "Ntheta": 5, "Nzeta": 5, "Nxi": 6, "Nx": 4}
    assert {k: int(np.asarray(data[k])) for k in sizes} == sizes
    assert [np.asarray(data[k]).shape for k in ("theta", "zeta", "x", "BHat")] == [
        (5,), (5,), (4,), (5, 5)
    ]


def test_nonstelsym_bhat_matches_fortran_golden() -> None:
    """Asymmetric VMEC ``BHat`` before any solve: a sine-term sign or parity bug shows here first."""
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import read_sfincs_input

    op = kinetic_operator_from_namelist(read_sfincs_input(REF / "nonstelsym_vmec_tiny_scheme5.input.namelist"))
    golden = read_sfincs_h5(REF / "nonstelsym_vmec_tiny_scheme5.sfincsOutput.h5")
    # Canonical stores (theta, zeta); the Fortran h5 stores (zeta, theta).
    np.testing.assert_allclose(np.asarray(op.b_hat), np.asarray(golden["BHat"]).T, rtol=0, atol=1e-10)
