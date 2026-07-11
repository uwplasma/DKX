"""Parity tests for geometryScheme=13 (namelist Boozer |B| spectrum).

geometryScheme=13 supplies the Boozer |B| spectrum directly in the namelist as
``boozer_bmnc(m,n)`` / ``boozer_bmns(m,n)`` 2-D arrays (geometry.F90 case 13, the
STELLOPT/BMNC optimization path).  These tests pin the canonical wiring in
:func:`sfincs_jax.drift_kinetic.kinetic_operator_from_namelist` (via
``_geometry_and_radial`` routing to
:meth:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_fourier`) against a
Fortran SFINCS v3 golden, mirroring the sibling ``test_output_h5_scheme*_parity``
tests.  The legacy ``outputs`` writer intentionally does not support scheme 13, so
the canonical side runs through :func:`sfincs_jax.run.run_profile`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5

REF = Path(__file__).parent / "ref"
DECK = REF / "pas_1species_PAS_noEr_tiny_scheme13.input.namelist"
GOLDEN = REF / "pas_1species_PAS_noEr_tiny_scheme13.sfincsOutput.h5"

# Geometry + scalar datasets compared bit-for-bit (the Boozer-spectrum convention
# gate: BHat, DHat, the co/contravariant B components, VPrimeHat, FSABHat2, the
# radial coordinates, and the flux functions all follow from the same analytic
# spectrum on the same theta/zeta grid, so they must match to rounding).
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


def _canonical_output(tmp_path: Path) -> Path:
    """Run the canonical RHSMode=1 solver on the scheme-13 deck and write its h5."""
    from sfincs_jax.run import run_profile

    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(DECK, out_path=out, emit=None)
    assert run.solve_result.converged
    assert run.output_path is not None
    return out


def test_output_scheme13_matches_fortran_fixture(tmp_path: Path) -> None:
    assert GOLDEN.exists(), f"Missing Fortran fixture: {GOLDEN}"
    out = _canonical_output(tmp_path)

    a = read_sfincs_h5(out)
    b = read_sfincs_h5(GOLDEN)
    # The canonical writer embeds the deck verbatim, exactly as Fortran does.
    assert a["input.namelist"] == b["input.namelist"]

    # Geometry + scalars: strict rounding-level parity (the convention gate).
    strict = compare_sfincs_outputs(a_path=out, b_path=GOLDEN, keys=_STRICT_KEYS, rtol=0, atol=1e-12)
    bad = [r.key for r in strict if not r.ok]
    assert not bad, f"Mismatched strict keys: {bad}"

    # Kinetic transport / moments from the solved state (Fortran uses a direct
    # solve; the canonical tier-1 route is also direct, so parity is ~1e-13 here).
    transport = compare_sfincs_outputs(a_path=out, b_path=GOLDEN, keys=_TRANSPORT_KEYS, rtol=0, atol=1e-8)
    bad_t = [r.key for r in transport if not r.ok]
    assert not bad_t, f"Mismatched transport keys: {bad_t}"

    # uHat is the Shaing PS-flow field; the canonical writer builds it with an FFT
    # magnetic-differential-equation solve while Fortran uses a harmonic quadrature,
    # so like the sibling scheme tests it is compared at the looser 1e-8 gate.
    uhat = compare_sfincs_outputs(a_path=out, b_path=GOLDEN, keys=["uHat"], rtol=0, atol=1e-8)
    bad_u = [(r.key, r.max_abs) for r in uhat if not r.ok]
    assert not bad_u, f"Mismatched uHat: {bad_u}"


def test_scheme13_deck_routes_through_canonical_stack() -> None:
    """The scheme-13 RHSMode=1 deck is owned by the canonical stack, not the legacy pipeline."""
    from sfincs_jax.cli import deck_requires_legacy_pipeline
    from sfincs_jax.inputs import read_sfincs_input

    assert deck_requires_legacy_pipeline(read_sfincs_input(DECK)) is None


def test_scheme13_from_namelist_matches_direct_from_fourier() -> None:
    """The namelist scheme-13 geometry equals a direct ``from_fourier`` build.

    Confirms ``_geometry_and_radial`` simply forwards the parsed ``boozer_bmnc``
    spectrum to :meth:`FluxSurfaceGeometry.from_fourier` with the namelist flux
    functions (no hidden transformation), matching BHat/DHat/FSABHat2 to rounding.
    """
    import jax.numpy as jnp

    from sfincs_jax.drift_kinetic import _geometry_and_radial, _get_int, _n_periods_from_namelist
    from sfincs_jax.inputs import read_sfincs_input
    from sfincs_jax.magnetic_geometry import FluxSurfaceGeometry
    from sfincs_jax.phase_space import make_grids

    raw = read_sfincs_input(DECK)
    res = raw.group("resolutionParameters")
    grids = make_grids(
        n_theta=_get_int(res, "Ntheta", 15),
        n_zeta=_get_int(res, "Nzeta", 15),
        n_xi=_get_int(res, "Nxi", 16),
        n_x=_get_int(res, "Nx", 5),
        n_l=_get_int(res, "NL", 4),
        n_periods=_n_periods_from_namelist(nml=raw),
        x_grid_scheme=5,
        n_xi_for_x_option=0,
    )
    geom, _radial = _geometry_and_radial(nml=raw, grids=grids)

    # Rebuild the same spectrum directly from the parsed boozer_bmnc map.
    spectrum = raw.indexed["geometryparameters"]["BOOZER_BMNC"]
    keys = sorted(spectrum)
    direct = FluxSurfaceGeometry.from_fourier(
        theta=grids.theta,
        zeta=grids.zeta,
        bmnc=jnp.asarray([float(spectrum[k]) for k in keys]),
        m=jnp.asarray([int(k[0]) for k in keys]),
        n=jnp.asarray([int(k[1]) for k in keys]),
        bmns=None,
        n_periods=int(raw.group("geometryParameters")["NPERIODS"]),
        iota=float(raw.group("geometryParameters")["IOTA"]),
        g_hat=float(raw.group("geometryParameters")["GHAT"]),
        i_hat=float(raw.group("geometryParameters")["IHAT"]),
    )

    np.testing.assert_allclose(np.asarray(geom.b_hat), np.asarray(direct.b_hat), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(geom.d_hat), np.asarray(direct.d_hat), rtol=0, atol=1e-12)
    tw, zw = grids.theta_weights, grids.zeta_weights
    fsa_geom = float(geom.fsab_hat2(theta_weights=tw, zeta_weights=zw))
    fsa_direct = float(direct.fsab_hat2(theta_weights=tw, zeta_weights=zw))
    assert abs(fsa_geom - fsa_direct) <= 1e-12
