"""End-to-end parity for magneticDriftScheme 2-9 (RHSMode=1, geometryScheme 11).

The ``magdrift_1species_tiny_scheme<N>`` decks are the tangential-magnetic-drift
fixture (W7-X standard Boozer ``.bc``, tiny grids) with ``magneticDriftScheme=N``
and PAS collisions (``nu_n = 8.4774e-3``) so the solve is well-conditioned.  Each
deck has a Fortran SFINCS v3 golden ``sfincsOutput.h5`` (single rank, MUMPS) and
a ``whichMatrix_1`` PETSc matrix; the matrix parity lives in
``tests/test_magnetic_drifts_parity.py`` and this module pins the canonical
:func:`sfincs_jax.run.run_profile` route end to end — the drift decks couple L±2
so they are not block-tridiagonal and tier-2 GCROT owns them (solverTolerance
1e-10 in the decks).

Mirrors the sibling ``test_output_h5_constraintscheme34_parity`` structure:
strict rounding-level parity for geometry/scalars (including the new
``BDotCurlB``/``gpsiHatpsiHat``/``diotadpsiHat`` drift-geometry datasets) and a
1e-8 absolute gate for the solution-derived transport moments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sfincs_jax.compare import compare_sfincs_outputs

REF = Path(__file__).parent / "ref"

SCHEMES = (2, 3, 4, 5, 6, 7, 8, 9)

# Geometry + scalar datasets compared at rounding level (the drift-relevant
# radial-derivative and curl datasets are refereed here so a geometry regression
# cannot hide behind the solve tolerance).
_STRICT_KEYS = [
    "Nspecies", "Ntheta", "Nzeta", "Nxi", "NL", "Nx", "theta", "zeta", "x", "Nxi_for_x",
    "geometryScheme", "Delta", "alpha", "nu_n", "Er", "dPhiHatdpsiHat", "psiAHat", "aHat",
    "psiHat", "psiN", "rHat", "rN", "inputRadialCoordinate",
    "inputRadialCoordinateForGradients", "coordinateSystem", "magneticDriftScheme",
    "collisionOperator", "constraintScheme", "RHSMode", "includeXDotTerm",
    "includeElectricFieldTermInXiDot", "useDKESExBDrift",
    "force0RadialCurrentInEquilibrium", "includePhi1", "NPeriods", "B0OverBBar", "iota",
    "GHat", "IHat", "VPrimeHat", "FSABHat2", "Zs", "mHats", "THats", "nHats", "DHat",
    "BHat", "dBHatdpsiHat", "dBHatdtheta", "dBHatdzeta", "BDotCurlB", "BHat_sub_psi",
    "dBHat_sub_psi_dtheta", "dBHat_sub_psi_dzeta", "BHat_sub_theta",
    "dBHat_sub_theta_dpsiHat", "dBHat_sub_theta_dzeta", "BHat_sub_zeta",
    "dBHat_sub_zeta_dpsiHat", "dBHat_sub_zeta_dtheta", "BHat_sup_theta",
    "dBHat_sup_theta_dpsiHat", "dBHat_sup_theta_dzeta", "BHat_sup_zeta",
    "dBHat_sup_zeta_dpsiHat", "dBHat_sup_zeta_dtheta", "diotadpsiHat",
]  # fmt: skip

# |grad psiHat|^2 involves the long R/Z/Dz harmonic reductions; compared at the
# sibling end-to-end gate (test_rhsmode1_write_output_end_to_end uses 5e-10).
_GPSIPSI_KEYS = ["gpsiHatpsiHat"]

# Kinetic transport outputs of the full canonical solve (state -> moments).
_TRANSPORT_KEYS = [
    "FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat",
    "momentumFlux_vm_psiHat", "particleFlux_vm0_psiHat", "heatFlux_vm0_psiHat",
    "NTV", "FSABFlow_vs_x",
]  # fmt: skip


def _deck(scheme: int) -> Path:
    return REF / f"magdrift_1species_tiny_scheme{scheme}.input.namelist"


def _golden(scheme: int) -> Path:
    return REF / f"magdrift_1species_tiny_scheme{scheme}.sfincsOutput.h5"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_magdrift_scheme_matches_fortran_fixture(
    scheme: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sfincs_jax.run import run_profile

    monkeypatch.setenv("SFINCS_JAX_EQUILIBRIA_DIRS", str(REF))
    golden = _golden(scheme)
    assert golden.exists(), f"Missing Fortran fixture: {golden}"

    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(_deck(scheme), out_path=out, emit=None)
    assert run.solve_result.converged
    assert run.output_path is not None

    # Geometry + scalars: strict rounding-level parity.
    strict = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_STRICT_KEYS, rtol=0, atol=1e-12)
    bad = [(r.key, r.max_abs) for r in strict if not r.ok]
    assert not bad, f"Mismatched strict keys: {bad}"

    gpsipsi = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_GPSIPSI_KEYS, rtol=0, atol=5e-10)
    bad_g = [(r.key, r.max_abs) for r in gpsipsi if not r.ok]
    assert not bad_g, f"Mismatched gpsiHatpsiHat: {bad_g}"

    # Kinetic transport / moments from the solved state.  Fortran direct-solves
    # (MUMPS) while the canonical route is tier-2 GCROT at solverTolerance
    # 1e-10; the measured absolute differences are <~2e-11 on these fixtures.
    transport = compare_sfincs_outputs(a_path=out, b_path=golden, keys=_TRANSPORT_KEYS, rtol=0, atol=1e-8)
    bad_t = [(r.key, r.max_abs) for r in transport if not r.ok]
    assert not bad_t, f"Mismatched transport keys: {bad_t}"
