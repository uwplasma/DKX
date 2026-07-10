"""Referee tests: ``sfincs_jax.moments`` vs the legacy diagnostics modules.

Every output of the consolidated module is asserted equal (1e-14 relative) to
the OLD implementation evaluated on the same solved state vectors, using the
recorded Fortran state vectors in ``tests/ref`` exactly as the existing parity
tests do (``test_transport_matrix_rhsmode3_parity.py`` et al.), so no linear
solves are needed:

- RHSMode=3 monoenergetic (``monoenergetic_PAS_tiny_scheme1``): transport
  matrix + vm flux moments + the RHSMode=2/3 output-field table.
- RHSMode=2 Onsager (``transportMatrix_PAS_tiny_rhsMode2_scheme11``): same.
- RHSMode=1 (``quick_2species_FPCollisions_noEr``, 2-species FP): the full
  per-species output table, NTV, classical fluxes.
- RHSMode=1 with Phi1 (``pas_1species_PAS_noEr_tiny_withPhi1_linear``): the
  Phi1-from-state table, vE flux family, vd/vd1 combinations, and the
  radial-coordinate flux variants.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import sfincs_jax.moments as mv
from sfincs_jax.constants import RadialCoordinates
from sfincs_jax.diagnostics import u_hat_np
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
from sfincs_jax.outputs.rhsmode1 import (
    write_rhsmode1_electric_drift_diagnostics_to_data,
    write_rhsmode1_ntv_diagnostics_to_data,
)
from sfincs_jax.outputs.transport import conversion_factors_to_from_dpsi_hat
from sfincs_jax.physics.classical_transport import classical_flux_v3
from sfincs_jax.problems import transport_diagnostics as td
from sfincs_jax.validation.fortran import read_petsc_vec

REF = Path(__file__).parent / "ref"
RTOL = 1e-14


def _assert_close(new, old, *, atol: float = 0.0) -> None:
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=RTOL, atol=atol)


def _setup(base: str):
    nml = read_sfincs_input(REF / f"{base}.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    return (
        nml,
        op,
        mv.StateLayout.from_operator(op),
        mv.VelocityGrid.from_operator(op),
        mv.FluxSurface.from_operator(op),
        mv.SpeciesParams.from_operator(op),
    )


def _vec(name: str) -> jnp.ndarray:
    return jnp.asarray(read_petsc_vec(REF / f"{name}.petscbin").values)


def _assert_vm_moments_match(new: mv.VmFluxMoments, old: td.V3TransportDiagnostics) -> None:
    for field in dataclasses.fields(old):
        _assert_close(getattr(new, field.name), getattr(old, field.name))


def _check_transport_case(base: str, *, rhs_mode: int) -> None:
    nml, op, layout, vgrid, surface, species = _setup(base)
    assert int(op.rhs_mode) == rhs_mode
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    n = td.transport_matrix_size_from_rhs_mode(rhs_mode)
    assert mv.transport_matrix_size(rhs_mode) == n
    states = {i: _vec(f"{base}.whichRHS{i}.stateVector") for i in range(1, n + 1)}

    # Per-whichRHS vm flux moments:
    for i in range(1, n + 1):
        old_diag = td.v3_transport_diagnostics_vm_only(op, x_full=states[i])
        new_diag = mv.vm_flux_moments(
            layout, vgrid, surface, species, states[i], delta=op.delta, alpha=op.alpha, phi1_hat=op.phi1_hat_base
        )
        _assert_vm_moments_match(new_diag, old_diag)

    # Transport matrix:
    old_tm = td.v3_transport_matrix_from_state_vectors(op0=op, geom=geom, state_vectors_by_rhs=states)
    new_tm = mv.transport_matrix_from_state_vectors(
        layout,
        vgrid,
        surface,
        species,
        states,
        rhs_mode=rhs_mode,
        delta=op.delta,
        alpha=op.alpha,
        g_hat=float(geom.g_hat),
        i_hat=float(geom.i_hat),
        iota=float(geom.iota),
        b0_over_bbar=float(geom.b0_over_bbar),
        phi1_hat=op.phi1_hat_base,
    )
    assert new_tm.shape == (n, n)
    _assert_close(new_tm, old_tm)

    # RHSMode=2/3 output-field table (Python-read h5 order):
    old_fields = td.v3_transport_output_fields_vm_only(op0=op, state_vectors_by_rhs=states, chunk_size=0)
    new_fields = mv.transport_moments_table(
        layout,
        vgrid,
        surface,
        species,
        states,
        rhs_mode=rhs_mode,
        delta=op.delta,
        alpha=op.alpha,
        phi1_hat=op.phi1_hat_base,
    )
    assert set(new_fields) == set(old_fields)
    for key in sorted(old_fields):
        _assert_close(new_fields[key], old_fields[key])


def test_rhsmode3_monoenergetic_transport_matrix_and_fields_match_old() -> None:
    _check_transport_case("monoenergetic_PAS_tiny_scheme1", rhs_mode=3)


def test_rhsmode2_transport_matrix_and_fields_match_old() -> None:
    _check_transport_case("transportMatrix_PAS_tiny_rhsMode2_scheme11", rhs_mode=2)


def test_rhsmode1_full_per_species_table_matches_old_quick2species() -> None:
    _, op, layout, vgrid, surface, species = _setup("quick_2species_FPCollisions_noEr")
    assert int(op.rhs_mode) == 1 and layout.n_species == 2
    x_full = _vec("quick_2species_FPCollisions_noEr.stateVector")

    old = td.v3_rhsmode1_output_fields_vm_only(op, x_full=x_full)
    new = mv.rhsmode1_moments(
        layout, vgrid, surface, species, x_full, delta=op.delta, alpha=op.alpha, phi1_hat=op.phi1_hat_base
    )
    assert set(new) == set(old)
    for key in sorted(old):
        _assert_close(new[key], old[key])

    # Moment identities as a sanity anchor (not just old-vs-new agreement):
    _assert_close(np.asarray(new["particleFlux_vm_psiHat_vs_x"]).sum(axis=0), new["particleFlux_vm_psiHat"], atol=1e-16)
    _assert_close(new["FSABjHat"], np.dot(np.asarray(species.z_s), np.asarray(new["FSABFlow"])))


def test_rhsmode1_phi1_from_state_table_matches_old() -> None:
    _, op, layout, vgrid, surface, species = _setup("pas_1species_PAS_noEr_tiny_withPhi1_linear")
    assert layout.include_phi1
    x_full = _vec("pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector")

    old = td.v3_rhsmode1_output_fields_vm_only_phi1(op, x_full=x_full)
    new = mv.rhsmode1_moments(
        layout, vgrid, surface, species, x_full, delta=op.delta, alpha=op.alpha, phi1_from_state=True
    )
    assert set(new) == set(old)
    for key in sorted(old):
        _assert_close(new[key], old[key])

    phi1 = layout.phi1_hat(x_full)
    np.testing.assert_allclose(
        np.asarray(phi1).reshape(-1),
        np.asarray(x_full[layout.f_size : layout.f_size + layout.n_theta * layout.n_zeta]),
        rtol=0,
        atol=0,
    )


def test_ntv_moments_match_old_writer_helper() -> None:
    nml, op, layout, vgrid, surface, species = _setup("quick_2species_FPCollisions_noEr")
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    u_hat = u_hat_np(grids=grids, geom=geom)
    x_full = _vec("quick_2species_FPCollisions_noEr.stateVector")

    data = {
        "geometryScheme": np.asarray(4),
        "BHat": np.asarray(op.b_hat),
        "dBHatdtheta": np.asarray(op.db_hat_dtheta),
        "dBHatdzeta": np.asarray(op.db_hat_dzeta),
        "uHat": u_hat,
        "FSABHat2": float(op.fsab_hat2),
        "GHat": float(geom.g_hat),
        "IHat": float(geom.i_hat),
        "iota": float(geom.iota),
    }
    write_rhsmode1_ntv_diagnostics_to_data(
        data=data, op=op, xs=[np.asarray(x_full)], x_stack=None, n_iter=1, fortran_h5_layout=lambda a: a
    )
    old_before_ztsn = data["NTVBeforeSurfaceIntegral"]  # (Z,T,S,1)
    old_ntv_sn = data["NTV"]  # (S,1)

    kernel = mv.ntv_kernel(surface, u_hat=u_hat, g_hat=float(geom.g_hat), i_hat=float(geom.i_hat), iota=float(geom.iota))
    new_before_stz, new_ntv = mv.ntv_moments(layout, vgrid, surface, species, x_full, kernel=kernel)
    _assert_close(np.transpose(np.asarray(new_before_stz), (2, 1, 0))[:, :, :, None], old_before_ztsn)
    _assert_close(np.asarray(new_ntv)[:, None], old_ntv_sn)


def test_classical_fluxes_match_old_with_and_without_phi1() -> None:
    _, op, layout, vgrid, surface, species = _setup("quick_2species_FPCollisions_noEr")
    gpsipsi = jnp.asarray(op.b_hat, dtype=jnp.float64) ** 2 * 0.37 + 0.05
    phi1 = 0.02 * (jnp.asarray(op.b_hat, dtype=jnp.float64) - jnp.mean(jnp.asarray(op.b_hat)))
    nu_n = 8.33e-3
    vp = mv.vprime_hat(surface)

    for use_phi1 in (False, True):
        old_pf, old_hf = classical_flux_v3(
            use_phi1=use_phi1,
            theta_weights=op.theta_weights,
            zeta_weights=op.zeta_weights,
            d_hat=op.d_hat,
            gpsipsi=gpsipsi,
            b_hat=op.b_hat,
            vprime_hat=vp,
            alpha=op.alpha,
            phi1_hat=phi1,
            delta=op.delta,
            nu_n=nu_n,
            z_s=op.z_s,
            m_hat=op.m_hat,
            t_hat=op.t_hat,
            n_hat=op.n_hat,
            dn_hat_dpsi_hat=op.dn_hat_dpsi_hat,
            dt_hat_dpsi_hat=op.dt_hat_dpsi_hat,
        )
        new_pf, new_hf = mv.classical_fluxes(
            use_phi1=use_phi1,
            surface=surface,
            species=species,
            gpsipsi=gpsipsi,
            phi1_hat=phi1,
            alpha=op.alpha,
            delta=op.delta,
            nu_n=nu_n,
            dn_hat_dpsi_hat=op.dn_hat_dpsi_hat,
            dt_hat_dpsi_hat=op.dt_hat_dpsi_hat,
        )
        _assert_close(new_pf, old_pf)
        _assert_close(new_hf, old_hf)


def _old_ve_block(op, x_full, phi1, dpt, dpz):
    """Literal transcription of the vE flux block of ``outputs/writer.py``
    (the only place the old code computes the vE family; lines ~1690-1796)."""
    from dataclasses import replace

    from sfincs_jax.problems.transport_diagnostics import f0_l0_v3_from_operator

    tw = jnp.asarray(op.theta_weights, dtype=jnp.float64)
    zw = jnp.asarray(op.zeta_weights, dtype=jnp.float64)
    w2d = (tw[:, None] * zw[None, :]).astype(jnp.float64)
    vprime_hat = jnp.sum(w2d / op.d_hat)
    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)
    x = jnp.asarray(op.x, dtype=jnp.float64)
    xw = jnp.asarray(op.x_weights, dtype=jnp.float64)
    w_pf_vE = xw * (x**2)
    w_hf_vE = xw * (x**4)
    w_mf_vE = xw * (x**3)
    pf_factor_vE = 2.0 * op.alpha * jnp.pi * op.delta * t_hat * sqrt_t / (vprime_hat * m_hat * sqrt_m)
    hf_factor_vE = op.alpha * jnp.pi * op.delta * (t_hat * t_hat) * sqrt_t / (vprime_hat * m_hat * sqrt_m)
    mf_factor_vE = 2.0 * op.alpha * jnp.pi * op.delta * (t_hat * t_hat) / (vprime_hat * m_hat)

    op_use = replace(op, phi1_hat_base=phi1)
    f_delta = x_full[: op.f_size].reshape(op.fblock.f_shape)
    f0_l0 = f0_l0_v3_from_operator(op_use)
    f_full_l0 = f_delta[:, :, 0, :, :] + f0_l0
    factor_vE = (op.b_hat_sub_theta * dpz - op.b_hat_sub_zeta * dpt) / (op.b_hat * op.b_hat)

    pf_before_vE = pf_factor_vE[:, None, None] * factor_vE[None, :, :] * jnp.einsum("x,sxtz->stz", w_pf_vE, f_full_l0)
    pf_before_vE0 = pf_factor_vE[:, None, None] * factor_vE[None, :, :] * jnp.einsum("x,sxtz->stz", w_pf_vE, f0_l0)
    hf_before_vE = hf_factor_vE[:, None, None] * factor_vE[None, :, :] * jnp.einsum("x,sxtz->stz", w_hf_vE, f_full_l0)
    hf_before_vE0 = hf_factor_vE[:, None, None] * factor_vE[None, :, :] * jnp.einsum("x,sxtz->stz", w_hf_vE, f0_l0)
    mf_before_vE = (
        (2.0 / 3.0)
        * mf_factor_vE[:, None, None]
        * factor_vE[None, :, :]
        * op.b_hat[None, :, :]
        * jnp.einsum("x,sxtz->stz", w_mf_vE, f_delta[:, :, 1, :, :])
    )
    return {
        "pf_before_vE": pf_before_vE,
        "pf_before_vE0": pf_before_vE0,
        "hf_before_vE": hf_before_vE,
        "hf_before_vE0": hf_before_vE0,
        "mf_before_vE": mf_before_vE,
        "pf_vE": jnp.einsum("tz,stz->s", w2d, pf_before_vE),
        "pf_vE0": jnp.einsum("tz,stz->s", w2d, pf_before_vE0),
        "hf_vE": jnp.einsum("tz,stz->s", w2d, hf_before_vE),
        "hf_vE0": jnp.einsum("tz,stz->s", w2d, hf_before_vE0),
        "mf_vE": jnp.einsum("tz,stz->s", w2d, mf_before_vE),
    }


def test_electric_drift_fluxes_and_vd_combinations_match_old() -> None:
    _, op, layout, vgrid, surface, species = _setup("pas_1species_PAS_noEr_tiny_withPhi1_linear")
    x_full = _vec("pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector")
    phi1 = layout.phi1_hat(x_full)
    # Synthetic Phi1 angular derivatives (formula-level referee; any arrays work):
    dpt = 0.7 * jnp.roll(phi1, 1, axis=0) - 0.7 * jnp.roll(phi1, -1, axis=0)
    dpz = 0.4 * jnp.roll(phi1, 1, axis=1) - 0.4 * jnp.roll(phi1, -1, axis=1)

    old = _old_ve_block(op, x_full, phi1, dpt, dpz)
    new = mv.electric_drift_flux_moments(
        layout,
        vgrid,
        surface,
        species,
        x_full,
        delta=op.delta,
        alpha=op.alpha,
        phi1_hat=phi1,
        dphi1_hat_dtheta=dpt,
        dphi1_hat_dzeta=dpz,
    )
    _assert_close(new.particle_flux_before_surface_integral_ve, old["pf_before_vE"])
    _assert_close(new.particle_flux_before_surface_integral_ve0, old["pf_before_vE0"])
    _assert_close(new.heat_flux_before_surface_integral_ve, old["hf_before_vE"])
    _assert_close(new.heat_flux_before_surface_integral_ve0, old["hf_before_vE0"])
    _assert_close(new.momentum_flux_before_surface_integral_ve, old["mf_before_vE"])
    np.testing.assert_allclose(np.asarray(new.momentum_flux_before_surface_integral_ve0), 0.0, atol=0.0)
    _assert_close(new.particle_flux_ve_psi_hat, old["pf_vE"])
    _assert_close(new.particle_flux_ve0_psi_hat, old["pf_vE0"])
    _assert_close(new.heat_flux_ve_psi_hat, old["hf_vE"])
    _assert_close(new.heat_flux_ve0_psi_hat, old["hf_vE0"])
    _assert_close(new.momentum_flux_ve_psi_hat, old["mf_vE"])

    # vd/vd1 combinations and coordinate variants against the OLD public helper:
    table = mv.rhsmode1_moments(
        layout, vgrid, surface, species, x_full, delta=op.delta, alpha=op.alpha, phi1_from_state=True
    )
    coords = RadialCoordinates(psi_a_hat=0.7, a_hat=1.3, r_n=0.6)
    conv = conversion_factors_to_from_dpsi_hat(psi_a_hat=0.7, a_hat=1.3, r_n=0.6)
    new_flux = {
        "particleFlux": (table["particleFlux_vm_psiHat"], new.particle_flux_ve0_psi_hat, new.particle_flux_ve_psi_hat),
        "heatFlux": (table["heatFlux_vm_psiHat"], new.heat_flux_ve0_psi_hat, new.heat_flux_ve_psi_hat),
        "momentumFlux": (
            table["momentumFlux_vm_psiHat"],
            new.momentum_flux_ve0_psi_hat,
            new.momentum_flux_ve_psi_hat,
        ),
    }
    data = {}
    for name, (vm, ve0, ve) in new_flux.items():
        data[f"{name}_vm_psiHat"] = np.asarray(vm)[:, None]
        data[f"{name}_vE0_psiHat"] = np.asarray(ve0)[:, None]
        data[f"{name}_vE_psiHat"] = np.asarray(ve)[:, None]
    write_rhsmode1_electric_drift_diagnostics_to_data(
        data=data,
        before_surface_integral_stz={},
        fluxes_s={},
        ntv_list=[np.zeros((layout.n_species,))],
        conversion_factors=conv,
        fortran_h5_layout=lambda a: a,
    )
    for name, (vm, ve0, ve) in new_flux.items():
        vd1, vd = mv.combined_drift_fluxes(flux_vm_psi_hat=vm, flux_ve0_psi_hat=ve0, flux_ve_psi_hat=ve)
        _assert_close(np.asarray(vd1)[:, None], data[f"{name}_vd1_psiHat"])
        _assert_close(np.asarray(vd)[:, None], data[f"{name}_vd_psiHat"])
        for label, variant in (("psiN", "psi_n"), ("rHat", "r_hat"), ("rN", "r_n")):
            variants = mv.flux_coordinate_variants(vd, coords)
            _assert_close(np.asarray(getattr(variants, variant))[:, None], data[f"{name}_vd_{label}"])
    hf_no_phi1 = mv.heat_flux_without_phi1(
        heat_flux_vm_psi_hat=new_flux["heatFlux"][0], heat_flux_ve0_psi_hat=new_flux["heatFlux"][1]
    )
    _assert_close(np.asarray(hf_no_phi1)[:, None], data["heatFlux_withoutPhi1_psiHat"])


def test_flux_coordinate_variants_match_old_conversion_factors() -> None:
    coords = RadialCoordinates(psi_a_hat=0.31, a_hat=1.9, r_n=0.55)
    conv = conversion_factors_to_from_dpsi_hat(psi_a_hat=0.31, a_hat=1.9, r_n=0.55)
    values = jnp.asarray([1.5, -2.5, 0.0])
    variants = mv.flux_coordinate_variants(values, coords)
    _assert_close(variants.psi_hat, values)
    _assert_close(variants.psi_n, np.asarray(values) * conv["ddpsiN2ddpsiHat"])
    _assert_close(variants.r_hat, np.asarray(values) * conv["ddrHat2ddpsiHat"])
    _assert_close(variants.r_n, np.asarray(values) * conv["ddrN2ddpsiHat"])
