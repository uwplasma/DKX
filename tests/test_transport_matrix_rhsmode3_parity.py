from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_vec
from sfincs_jax.problems.transport_diagnostics import v3_transport_matrix_from_state_vectors
from sfincs_jax.problems.transport_diagnostics import v3_transport_output_fields_vm_only
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist


RHS_MODE3_MONOENERGETIC_BASES = (
    "monoenergetic_PAS_tiny_scheme1",
    "monoenergetic_PAS_tiny_scheme11",
    "monoenergetic_PAS_tiny_scheme5_filtered",
)


def _scalar_h5(out: dict[str, object], key: str) -> float:
    return float(np.asarray(out[key], dtype=np.float64).reshape(-1)[0])


def _l12_l21_reciprocity_tolerance(base: str) -> float:
    # The filtered VMEC fixture is intentionally low resolution and has a looser
    # DKES-style reciprocity error than the direct geometry fixtures.
    return 8e-2 if base.endswith("scheme5_filtered") else 3e-3


@pytest.mark.parametrize(
    "base",
    RHS_MODE3_MONOENERGETIC_BASES,
)
def test_transport_matrix_rhsmode3_matches_fortran_output(base: str) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / f"{base}.input.namelist"
    out_path = here / "ref" / f"{base}.sfincsOutput.h5"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    assert int(op0.rhs_mode) == 3

    vec1 = here / "ref" / f"{base}.whichRHS1.stateVector.petscbin"
    vec2 = here / "ref" / f"{base}.whichRHS2.stateVector.petscbin"
    state_vecs = {
        1: jnp.asarray(read_petsc_vec(vec1).values),
        2: jnp.asarray(read_petsc_vec(vec2).values),
    }

    tm = np.asarray(v3_transport_matrix_from_state_vectors(op0=op0, geom=geom, state_vectors_by_rhs=state_vecs))
    out = read_sfincs_h5(out_path)
    tm_ref = np.asarray(out["transportMatrix"], dtype=np.float64)

    assert tm.shape == (2, 2)
    assert tm_ref.shape == (2, 2)
    # Fortran writes arrays in column-major order; as read by Python, the dataset appears transposed.
    np.testing.assert_allclose(tm.T, tm_ref, rtol=0, atol=2e-10)

    # Also validate the diagnostic fields used by upstream scan plotting scripts.
    # For these extensible diagnostic arrays, the Fortran output is already in (species, whichRHS) order as read by Python.
    pf_ref = np.asarray(out["particleFlux_vm_psiHat"], dtype=np.float64)
    hf_ref = np.asarray(out["heatFlux_vm_psiHat"], dtype=np.float64)
    fsab_ref = np.asarray(out["FSABFlow"], dtype=np.float64)

    # Compute from the solved state vectors:
    from sfincs_jax.problems.transport_diagnostics import v3_transport_diagnostics_vm_only

    d1 = v3_transport_diagnostics_vm_only(op0, x_full=state_vecs[1])
    d2 = v3_transport_diagnostics_vm_only(op0, x_full=state_vecs[2])
    pf = np.stack([np.asarray(d1.particle_flux_vm_psi_hat), np.asarray(d2.particle_flux_vm_psi_hat)], axis=1)
    hf = np.stack([np.asarray(d1.heat_flux_vm_psi_hat), np.asarray(d2.heat_flux_vm_psi_hat)], axis=1)
    fsab = np.stack([np.asarray(d1.fsab_flow), np.asarray(d2.fsab_flow)], axis=1)

    np.testing.assert_allclose(pf, pf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf, hf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(fsab, fsab_ref, rtol=0, atol=5e-10)

    # Validate additional fields expected by upstream scan plotting scripts.
    fields = v3_transport_output_fields_vm_only(op0=op0, state_vectors_by_rhs=state_vecs)
    for key in (
        "FSABjHat",
        "FSABjHatOverRootFSAB2",
        "FSABVelocityUsingFSADensity",
        "particleFluxBeforeSurfaceIntegral_vm",
        "heatFluxBeforeSurfaceIntegral_vm",
        "particleFlux_vm_psiHat_vs_x",
        "heatFlux_vm_psiHat_vs_x",
        "sources",
    ):
        np.testing.assert_allclose(np.asarray(fields[key]), np.asarray(out[key]), rtol=0, atol=5e-10)

    # Coordinate variants:
    psi_a_hat = float(np.asarray(out["psiAHat"]))
    a_hat = float(np.asarray(out["aHat"]))
    r_n = float(np.asarray(out["rN"]))
    ddpsiN2ddpsiHat = 1.0 / psi_a_hat
    ddrHat2ddpsiHat = a_hat / (2.0 * psi_a_hat * r_n)
    ddrN2ddpsiHat = 1.0 / (2.0 * psi_a_hat * r_n)

    pf = np.asarray(fields["particleFlux_vm_psiHat"], dtype=np.float64)
    hf = np.asarray(fields["heatFlux_vm_psiHat"], dtype=np.float64)
    np.testing.assert_allclose(pf * ddpsiN2ddpsiHat, np.asarray(out["particleFlux_vm_psiN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(pf * ddrHat2ddpsiHat, np.asarray(out["particleFlux_vm_rHat"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(pf * ddrN2ddpsiHat, np.asarray(out["particleFlux_vm_rN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddpsiN2ddpsiHat, np.asarray(out["heatFlux_vm_psiN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddrHat2ddpsiHat, np.asarray(out["heatFlux_vm_rHat"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddrN2ddpsiHat, np.asarray(out["heatFlux_vm_rN"]), rtol=0, atol=5e-10)


@pytest.mark.parametrize("base", RHS_MODE3_MONOENERGETIC_BASES)
def test_rhsmode3_monoenergetic_fixture_normalization_and_reciprocity(base: str) -> None:
    """Guard the v3 monoenergetic/DKES normalization contract without solving."""

    out_path = Path(__file__).parent / "ref" / f"{base}.sfincsOutput.h5"
    out = read_sfincs_h5(out_path)

    if "Nx" in out:
        assert int(np.asarray(out["Nx"]).reshape(-1)[0]) == 1
    if "x" in out:
        np.testing.assert_allclose(np.asarray(out["x"], dtype=np.float64), np.asarray([1.0]), rtol=0, atol=0)
    if "Nxi_for_x_option" in out:
        assert int(np.asarray(out["Nxi_for_x_option"]).reshape(-1)[0]) == 0

    nu_fields = ("nuPrime", "B0OverBBar", "GHat", "IHat", "iota", "nu_n")
    if all(key in out for key in nu_fields):
        denom = _scalar_h5(out, "GHat") + _scalar_h5(out, "iota") * _scalar_h5(out, "IHat")
        assert np.isfinite(denom)
        assert abs(denom) > 1e-30
        expected_nu_n = _scalar_h5(out, "nuPrime") * _scalar_h5(out, "B0OverBBar") / denom
        np.testing.assert_allclose(_scalar_h5(out, "nu_n"), expected_nu_n, rtol=2e-14, atol=1e-14)

    estar_fields = ("alpha", "Delta", "EStar", "iota", "B0OverBBar", "GHat", "dPhiHatdpsiHat")
    if all(key in out for key in estar_fields):
        g_hat = _scalar_h5(out, "GHat")
        assert np.isfinite(g_hat)
        assert abs(g_hat) > 1e-30
        expected_dphi = (
            2.0
            / (_scalar_h5(out, "alpha") * _scalar_h5(out, "Delta"))
            * _scalar_h5(out, "EStar")
            * _scalar_h5(out, "iota")
            * _scalar_h5(out, "B0OverBBar")
            / g_hat
        )
        np.testing.assert_allclose(_scalar_h5(out, "dPhiHatdpsiHat"), expected_dphi, rtol=2e-14, atol=1e-13)

    # Fortran HDF5 storage is column-major; transpose to mathematical row/column order before
    # checking DKES/monoenergetic Onsager reciprocity between L12 and L21.
    tm_math = np.asarray(out["transportMatrix"], dtype=np.float64).T
    assert tm_math.shape == (2, 2)
    l12 = float(tm_math[0, 1])
    l21 = float(tm_math[1, 0])
    rel_asymmetry = abs(l12 - l21) / max(abs(l12), abs(l21), np.finfo(np.float64).tiny)
    assert rel_asymmetry <= _l12_l21_reciprocity_tolerance(base)
