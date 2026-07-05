from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_vec
from sfincs_jax.problems.transport_diagnostics import v3_transport_output_fields_vm_only
from sfincs_jax.outputs.transport import TransportStreamingOutputAccumulator
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.problems.transport_solve import solve_v3_transport_matrix_linear_gmres
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist


def _reference_transport_case():
    here = Path(__file__).parent
    base = "transportMatrix_PAS_tiny_rhsMode2_scheme2"
    input_path = here / "ref" / f"{base}.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    state_vectors = {
        which_rhs: jnp.asarray(read_petsc_vec(here / "ref" / f"{base}.whichRHS{which_rhs}.stateVector.petscbin").values)
        for which_rhs in (1, 2, 3)
    }
    return nml, grids, geom, op0, state_vectors


def test_transport_streaming_outputs_match_batched_reference_fields() -> None:
    nml, grids, geom, op0, state_vectors = _reference_transport_case()
    accumulator = TransportStreamingOutputAccumulator.create(
        nml=nml,
        grids=grids,
        geom=geom,
        op0=op0,
        n_rhs=3,
        collect_full_output_fields=True,
    )
    for which_rhs, state_vector in state_vectors.items():
        accumulator.collect(which_rhs, state_vector)

    particle_flux, heat_flux, fsab_flow = accumulator.diagnostic_flux_arrays()
    fields = accumulator.output_fields()
    reference_fields = v3_transport_output_fields_vm_only(op0=op0, state_vectors_by_rhs=state_vectors)

    assert fields is not None
    np.testing.assert_allclose(np.asarray(particle_flux), np.asarray(reference_fields["particleFlux_vm_psiHat"]))
    np.testing.assert_allclose(np.asarray(heat_flux), np.asarray(reference_fields["heatFlux_vm_psiHat"]))
    np.testing.assert_allclose(np.asarray(fsab_flow), np.asarray(reference_fields["FSABFlow"]))
    for key in (
        "FSABFlow",
        "FSABFlow_vs_x",
        "FSABjHat",
        "FSABjHatOverRootFSAB2",
        "particleFlux_vm_psiHat",
        "heatFlux_vm_psiHat",
        "particleFlux_vm0_psiHat",
        "heatFlux_vm0_psiHat",
        "particleFluxBeforeSurfaceIntegral_vm",
        "heatFluxBeforeSurfaceIntegral_vm",
        "particleFlux_vm_psiHat_vs_x",
        "heatFlux_vm_psiHat_vs_x",
        "sources",
    ):
        np.testing.assert_allclose(
            np.asarray(fields[key]),
            np.asarray(reference_fields[key]),
            rtol=1e-12,
            atol=1e-18,
            err_msg=key,
        )


def test_transport_streaming_outputs_can_skip_full_field_buffers() -> None:
    nml, grids, geom, op0, state_vectors = _reference_transport_case()
    accumulator = TransportStreamingOutputAccumulator.create(
        nml=nml,
        grids=grids,
        geom=geom,
        op0=op0,
        n_rhs=3,
        collect_full_output_fields=False,
    )
    for which_rhs, state_vector in state_vectors.items():
        accumulator.collect(which_rhs, state_vector)

    particle_flux, heat_flux, fsab_flow = accumulator.diagnostic_flux_arrays()
    reference_fields = v3_transport_output_fields_vm_only(op0=op0, state_vectors_by_rhs=state_vectors)
    assert accumulator.output_fields() is None
    np.testing.assert_allclose(np.asarray(particle_flux), np.asarray(reference_fields["particleFlux_vm_psiHat"]))
    np.testing.assert_allclose(np.asarray(heat_flux), np.asarray(reference_fields["heatFlux_vm_psiHat"]))
    np.testing.assert_allclose(np.asarray(fsab_flow), np.asarray(reference_fields["FSABFlow"]))


def test_transport_driver_forced_streaming_outputs_match_batched_path(monkeypatch) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    nml = read_sfincs_input(input_path)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    streamed = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=True,
        force_stream_diagnostics=True,
    )
    batched = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=True,
        force_stream_diagnostics=False,
    )

    np.testing.assert_allclose(np.asarray(streamed.transport_matrix), np.asarray(batched.transport_matrix))
    assert streamed.transport_output_fields is not None
    reference_fields = v3_transport_output_fields_vm_only(
        op0=batched.op0,
        state_vectors_by_rhs=batched.state_vectors_by_rhs,
    )
    for key in ("FSABFlow", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "sources"):
        np.testing.assert_allclose(
            np.asarray(streamed.transport_output_fields[key]),
            np.asarray(reference_fields[key]),
            rtol=1e-10,
            atol=1e-12,
            err_msg=key,
        )
