"""Parity tests for non-stellarator-symmetric VMEC (geometryScheme=5, lasym=T).

VMEC writes the complementary-parity Fourier tables (``bmns``, ``gmns``,
``bsub{u,v}mns``, ``bsubsmnc``, ``bsup{u,v}mns``, ``rmns``, ``zmnc``, ``lmnc``)
for stellarator-asymmetric equilibria.  These tests pin the canonical wiring in
:func:`sfincs_jax.magnetic_geometry.read_vmec_wout` /
:meth:`sfincs_jax.magnetic_geometry.FluxSurfaceGeometry.from_vmec` (each cosine
sum gains its sine partner, and the sine-parity fields gain cosine partners)
against a Fortran SFINCS v3 golden, mirroring the sibling
``test_output_h5_scheme*_parity`` tests.  The legacy ``outputs`` writer does not
support lasym VMEC, so the canonical side runs through
:func:`sfincs_jax.run.run_profile`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5

REF = Path(__file__).parent / "ref"
DECK = REF / "nonstelsym_vmec_tiny_scheme5.input.namelist"
GOLDEN = REF / "nonstelsym_vmec_tiny_scheme5.sfincsOutput.h5"

# Geometry + scalar datasets compared bit-for-bit: every VMEC Fourier evaluation
# gains its complementary-parity sine/cosine term, so BHat, DHat, the
# co/contravariant B components and their derivatives, gpsiHatpsiHat, and the
# flux functions must all match the golden to rounding.
_STRICT_KEYS = [
    "Nspecies", "Ntheta", "Nzeta", "Nxi", "NL", "Nx", "theta", "zeta", "x",
    "geometryScheme", "Delta", "alpha", "nu_n", "Er", "psiAHat", "aHat", "psiHat",
    "psiN", "rHat", "rN", "inputRadialCoordinate", "rippleScale", "gpsiHatpsiHat",
    "collisionOperator", "constraintScheme", "RHSMode", "NPeriods", "iota",
    "VPrimeHat", "FSABHat2", "Zs", "mHats", "THats", "nHats", "DHat", "BHat",
    "dBHatdpsiHat", "dBHatdtheta", "dBHatdzeta", "BHat_sub_psi",
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
    """Run the canonical RHSMode=1 solver on the lasym scheme-5 deck; write its h5."""
    from sfincs_jax.run import run_profile

    out = tmp_path / "sfincsOutput.h5"
    run = run_profile(DECK, out_path=out, emit=None)
    assert run.solve_result.converged
    assert run.output_path is not None
    return out


def test_geometry_bhat_matches_fortran_golden() -> None:
    """Canonical BHat(theta,zeta) equals the Fortran golden to rounding.

    The direct geometry gate (before transport): a sign / parity / Nyquist bug in
    the stellarator-asymmetric sine terms would surface here first.
    """
    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
    from sfincs_jax.inputs import read_sfincs_input

    op = kinetic_operator_from_namelist(read_sfincs_input(DECK))
    golden = read_sfincs_h5(GOLDEN)
    # Canonical stores (theta, zeta); the Fortran h5 stores (zeta, theta).
    np.testing.assert_allclose(np.asarray(op.b_hat), np.asarray(golden["BHat"]).T, rtol=0, atol=1e-10)


def test_output_nonstelsym_vmec_scheme5_matches_fortran_fixture(tmp_path: Path) -> None:
    assert GOLDEN.exists(), f"Missing Fortran fixture: {GOLDEN}"
    out = _canonical_output(tmp_path)

    a = read_sfincs_h5(out)
    b = read_sfincs_h5(GOLDEN)
    # The canonical writer embeds the deck verbatim, exactly as Fortran does.
    assert a["input.namelist"] == b["input.namelist"]

    # Geometry + scalars: strict rounding-level parity (the sine-term convention gate).
    strict = compare_sfincs_outputs(a_path=out, b_path=GOLDEN, keys=_STRICT_KEYS, rtol=0, atol=1e-10)
    bad = [r.key for r in strict if not r.ok]
    assert not bad, f"Mismatched strict keys: {bad}"

    # Kinetic transport / moments from the solved state (both codes use a direct solve).
    transport = compare_sfincs_outputs(a_path=out, b_path=GOLDEN, keys=_TRANSPORT_KEYS, rtol=0, atol=1e-8)
    bad_t = [r.key for r in transport if not r.ok]
    assert not bad_t, f"Mismatched transport keys: {bad_t}"


def test_nonstelsym_vmec_deck_routes_through_canonical_stack() -> None:
    """The lasym scheme-5 RHSMode=1 deck is owned by the canonical stack, not the legacy pipeline."""
    from sfincs_jax.cli import deck_requires_legacy_pipeline
    from sfincs_jax.inputs import read_sfincs_input

    assert deck_requires_legacy_pipeline(read_sfincs_input(DECK)) is None
