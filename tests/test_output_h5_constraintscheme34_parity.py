"""Parity tests for constraintScheme 3 and 4 (Fokker-Planck, RHSMode=1).

constraintScheme 3 (constant + quartic sources) and 4 (quadratic + quartic
sources) differ from the default cs1 (constant + quadratic) ONLY in the x-shape
of the two bordered particle/heat source unknowns injected into the L=0 DKE rows
(``populateMatrix.F90`` lines 2915-2938); the flux-surface-averaged
density/pressure constraint rows are identical across 1/3/4.

The legacy ``operators``/``problems`` stack reuses the cs1 source basis for
cs3/4 (a bug that stays invisible whenever the sources solve to ~0), so these
fixtures pin the *canonical* :func:`dkx.drift_kinetic.kinetic_operator_from_namelist`
against a Fortran SFINCS v3 golden — NOT against the legacy writer — mirroring
the sibling ``test_output_h5_scheme*_parity`` tests.  The canonical side runs
through :func:`dkx.run.run_profile`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dkx.compare import compare_sfincs_outputs
from dkx.io import read_sfincs_h5

REF = Path(__file__).parent / "ref"

# Geometry + scalar datasets compared bit-for-bit (the built-in W7-X geometry is
# identical across the two decks; the constraintScheme value itself is refereed
# here too so a mis-parsed deck cannot pass).
_STRICT_KEYS = [
    "Nspecies", "Ntheta", "Nzeta", "Nxi", "NL", "Nx", "theta", "zeta", "x", "Nxi_for_x",
    "geometryScheme", "Delta", "alpha", "nu_n", "Er", "dPhiHatdpsiHat", "dPhiHatdpsiN",
    "dPhiHatdrHat", "dPhiHatdrN", "psiAHat", "aHat", "psiHat", "psiN", "rHat", "rN",
    "inputRadialCoordinate", "inputRadialCoordinateForGradients", "coordinateSystem",
    "rippleScale", "gpsiHatpsiHat", "diotadpsiHat", "EParallelHat", "collisionOperator",
    "constraintScheme", "RHSMode", "includeXDotTerm", "includeElectricFieldTermInXiDot",
    "useDKESExBDrift", "force0RadialCurrentInEquilibrium", "includePhi1",
    "includePhi1InKineticEquation", "includePhi1InCollisionOperator",
    "includeTemperatureEquilibrationTerm", "include_fDivVE_Term", "withAdiabatic",
    "withNBIspec", "NPeriods", "B0OverBBar", "iota", "GHat", "IHat", "VPrimeHat",
    "FSABHat2", "Zs", "mHats", "THats", "nHats", "dnHatdrHat", "dnHatdrN", "dnHatdpsiN",
    "dnHatdpsiHat", "dTHatdrHat", "dTHatdrN", "dTHatdpsiN", "dTHatdpsiHat", "DHat", "BHat",
    "dBHatdpsiHat", "dBHatdtheta", "dBHatdzeta", "BDotCurlB", "BHat_sub_psi",
    "dBHat_sub_psi_dtheta", "dBHat_sub_psi_dzeta", "BHat_sub_theta",
    "dBHat_sub_theta_dpsiHat", "dBHat_sub_theta_dzeta", "BHat_sub_zeta",
    "dBHat_sub_zeta_dpsiHat", "dBHat_sub_zeta_dtheta", "BHat_sup_theta",
    "dBHat_sup_theta_dpsiHat", "dBHat_sup_theta_dzeta", "BHat_sup_zeta",
    "dBHat_sup_zeta_dpsiHat", "dBHat_sup_zeta_dtheta",
]  # fmt: skip

# Kinetic transport outputs of the full canonical solve (state -> moments).
_TRANSPORT_KEYS = [
    "FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat",
    "momentumFlux_vm_psiHat", "particleFlux_vm0_psiHat", "heatFlux_vm0_psiHat",
    "NTV", "FSABFlow_vs_x",
]  # fmt: skip

SCHEMES = ("cs3", "cs4")


def _deck(cs: str) -> Path:
    return REF / f"fp_1species_FPCollisions_noEr_tiny_{cs}.input.namelist"


def _golden(cs: str) -> Path:
    return REF / f"fp_1species_FPCollisions_noEr_tiny_{cs}.sfincsOutput.h5"


@pytest.mark.parametrize("cs", SCHEMES)
def test_constraintscheme34_matches_fortran_fixture(cs: str, tmp_path: Path) -> None:
    from dkx.run import run_profile

    golden = _golden(cs)
    assert golden.exists(), f"Missing Fortran fixture: {golden}"

    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(_deck(cs), out_path=out, emit=None)
    assert run.solve_result.converged
    assert run.output_path is not None

    a = read_sfincs_h5(out)
    b = read_sfincs_h5(golden)
    # The canonical writer embeds the deck verbatim, exactly as Fortran does.
    assert a["input.namelist"] == b["input.namelist"]

    # Geometry + scalars: strict rounding-level parity.
    strict = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_STRICT_KEYS, rtol=0, atol=1e-12)
    bad = [r.key for r in strict if not r.ok]
    assert not bad, f"Mismatched strict keys: {bad}"

    # Kinetic transport / moments from the solved state (both sides direct-solve
    # this tiny FP system, so parity is ~1e-13 here).
    transport = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_TRANSPORT_KEYS, rtol=0, atol=1e-8)
    bad_t = [(r.key, r.max_abs) for r in transport if not r.ok]
    assert not bad_t, f"Mismatched transport keys: {bad_t}"

    # uHat (Shaing PS-flow field): looser 1e-8 gate like the sibling scheme tests.
    uhat = compare_sfincs_outputs(a_path=out, b_path=golden, keys=["uHat"], rtol=0, atol=1e-8)
    bad_u = [(r.key, r.max_abs) for r in uhat if not r.ok]
    assert not bad_u, f"Mismatched uHat: {bad_u}"
