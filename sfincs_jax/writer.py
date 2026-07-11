"""Canonical ``sfincsOutput`` writer for RHSMode=1/2/3 runs.

Fortran counterpart: ``writeHDF5Output.F90`` (dataset names, shapes, and the
Fortran column-major storage layout) plus the per-``whichRHS`` diagnostic
writes of ``diagnostics.F90``.  This module is the canonical-stack replacement
for the legacy ``outputs/writer.py`` + ``outputs/transport.py`` +
``outputs/rhsmode1.py`` pipeline for the supported case families: it consumes
only canonical objects (:mod:`sfincs_jax.inputs`,
:mod:`sfincs_jax.drift_kinetic`, :mod:`sfincs_jax.moments`,
:mod:`sfincs_jax.magnetic_geometry`) and writes the same datasets the legacy
writer emits for these modes.  Deferred families (Phi1, magnetic drifts,
export_f data arrays) are intentionally absent — the operator defers them.

Layout convention: like the legacy writer (``fortran_layout=True``),
grid/geometry/normalization fields are stored reversed-transposed (Fortran
column-major view of a ``(T, Z)`` array reads as ``(Z, T)`` in h5py), while the
per-``whichRHS`` diagnostic fields are stored directly in the Python-read
order of Fortran v3 files (``(Z, T, S, N)``, ``(X, S, N)``, ``(S, N)``).

Units/normalizations: every dataset is the dimensionless v3 "Hat" quantity;
see :mod:`sfincs_jax.moments` for the flux conventions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from sfincs_jax.constants import RadialCoordinates, nu_n_from_nu_prime
from sfincs_jax.drift_kinetic import KineticOperator, _resolve_equilibrium_path
from sfincs_jax.input_compat import (
    effective_psi_n_wish,
    infer_input_radial_coordinate_for_gradients,
)
from sfincs_jax.inputs import SfincsInput
from sfincs_jax.magnetic_geometry import (
    FluxSurfaceGeometry,
    _bracketing_surfaces,
    read_boozer_bc,
    read_vmec_wout,
    vmec_radial_interpolation,
)
from sfincs_jax.moments import (
    FluxSurface,
    SpeciesParams,
    StateLayout,
    VelocityGrid,
    classical_fluxes,
    flux_surface_b_integrals,
    ntv_kernel,
    ntv_moments,
    rhsmode1_moments,
    transport_moments_table,
)
from sfincs_jax.phase_space import Grids

__all__ = ["operator_containers", "write_profile_output", "write_transport_output"]

_I32 = np.int32


def _logical(value: bool) -> np.ndarray:
    """v3 integerToRepresentTrue/False encoding of a Fortran logical."""
    return np.asarray(1 if value else -1, dtype=_I32)


def _f64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _reversed_axes(arr: np.ndarray) -> np.ndarray:
    """Fortran column-major storage view of a row-major logical array."""
    if arr.ndim <= 1:
        return arr
    return np.ascontiguousarray(np.transpose(arr, tuple(reversed(range(arr.ndim)))))


def operator_containers(
    op: KineticOperator,
) -> Tuple[StateLayout, VelocityGrid, FluxSurface, SpeciesParams]:
    """The :mod:`sfincs_jax.moments` input containers of a canonical operator."""
    layout = StateLayout(
        n_species=op.n_species, n_x=op.n_x, n_xi=op.n_xi, n_theta=op.n_theta,
        n_zeta=op.n_zeta, include_phi1=bool(op.include_phi1), constraint_scheme=op.constraint_scheme,
    )  # fmt: skip
    import jax.numpy as jnp  # noqa: PLC0415

    vgrid = VelocityGrid(
        x=jnp.asarray(op.x), x_weights=jnp.asarray(op.x_weights),
        n_xi_for_x=jnp.asarray(op.n_xi_for_x, dtype=jnp.int32),
    )  # fmt: skip
    return layout, vgrid, FluxSurface.from_operator(op), SpeciesParams.from_operator(op)


def _u_hat(geom: FluxSurfaceGeometry) -> np.ndarray:
    """The v3 ``uHat`` field (geometry.F90 ``computeBIntegrals`` transcendental u).

    Solves ``(iota d/dtheta + d/dzeta) u = iota (G dh/dtheta + I dh/dzeta)`` with
    ``h = 1/BHat^2`` spectrally in the FFT basis of the (theta, zeta') grid.
    """
    b_hat = np.asarray(geom.b_hat, dtype=np.float64)
    n_theta, n_zeta = b_hat.shape
    f = np.fft.fft2(1.0 / (b_hat * b_hat))
    m = np.fft.fftfreq(n_theta, d=1.0 / n_theta)
    k = np.fft.fftfreq(n_zeta, d=1.0 / n_zeta)
    mm, kk = np.meshgrid(m, k, indexing="ij")
    n_eff = (-kk) * float(geom.n_periods)
    denom = n_eff - float(geom.iota) * mm
    numer = float(geom.iota) * (float(geom.g_hat) * mm + float(geom.i_hat) * n_eff)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(denom) < 1e-30, 0.0, numer / denom).astype(f.dtype)
    scale[0, 0] = 0.0
    return np.asarray(np.fft.ifft2(scale * f).real, dtype=np.float64)


def _geometry_extras(
    *, inp: SfincsInput, grids: Grids, geom: FluxSurfaceGeometry, radial: RadialCoordinates
) -> tuple[np.ndarray, float]:
    """``(gpsiHatpsiHat, diotadpsiHat)`` for the supported geometry schemes.

    Analytic schemes (1/2/4) store zeros, matching v3.  Boozer ``.bc`` and VMEC
    schemes recompute the |grad psiHat|^2 metric with the canonical geometry
    constructors and the iota radial derivative from the bracketing surfaces.
    """
    scheme = inp.geometry.geometry_scheme
    zeros = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    if scheme in {1, 2, 4}:
        return zeros, 0.0

    raw = inp.raw
    geom_params = raw.group("geometryParameters")
    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=raw, geom_params=geom_params)
        psi_n_wish = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        r_n_wish = math.sqrt(float(psi_n_wish))
        geom_g = FluxSurfaceGeometry.from_boozer(
            path, theta=grids.theta, zeta=grids.zeta, r_n_wish=r_n_wish,
            vmec_radial_option=inp.geometry.vmec_radial_option,
            geometry_scheme=scheme, compute_gpsipsi=True,
        )  # fmt: skip
        header, surfaces = read_boozer_bc(path, geometry_scheme=scheme)
        surf_old, surf_new = _bracketing_surfaces(surfaces, r_n_wish)
        delta_psi_hat = float(header.psi_a_hat) * (
            float(surf_new.r_n) ** 2 - float(surf_old.r_n) ** 2
        )
        # Toroidal-direction sign switch (iota -> -iota), as in v3 geometry.F90.
        diota = (-float(surf_new.iota) + float(surf_old.iota)) / delta_psi_hat
        return np.asarray(geom_g.gpsipsi, dtype=np.float64), float(diota)

    # geometryScheme = 5 (VMEC).
    path = _resolve_equilibrium_path(nml=raw, geom_params=geom_params, vmec=True)
    w = read_vmec_wout(path)
    psi_n_wish = effective_psi_n_wish(
        geom_params=geom_params, default_r_n=0.5,
        psi_a_hat=radial.psi_a_hat, a_hat=radial.a_hat,
    )  # fmt: skip
    geom_g = FluxSurfaceGeometry.from_vmec(
        w, theta=grids.theta, zeta=grids.zeta, psi_n_wish=float(psi_n_wish),
        vmec_radial_option=inp.geometry.vmec_radial_option,
        vmec_nyquist_option=inp.geometry.vmec_nyquist_option,
        min_bmn_to_load=inp.geometry.min_bmn_to_load,
        ripple_scale=inp.geometry.ripple_scale,
        helicity_n=inp.geometry.helicity_n, helicity_l=inp.geometry.helicity_l,
        compute_gpsipsi=True,
    )  # fmt: skip
    interp = vmec_radial_interpolation(
        w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=inp.geometry.vmec_radial_option
    )
    j0, j1 = interp.index_half
    diota = 0.0
    dpsi_n = float(interp.psi_n_half[j1 - 1] - interp.psi_n_half[j0 - 1])
    if dpsi_n != 0.0:
        diota = float(w.iotas[j1] - w.iotas[j0]) / dpsi_n / float(radial.psi_a_hat)
    return np.asarray(geom_g.gpsipsi, dtype=np.float64), float(diota)


def _effective_flux_functions(
    op: KineticOperator, geom: FluxSurfaceGeometry
) -> tuple[float, float, float]:
    """``(B0OverBBar, GHat, IHat)``; VMEC placeholders replaced by surface averages."""
    b0, g, i = float(geom.b0_over_bbar), float(geom.g_hat), float(geom.i_hat)
    if abs(g) < 1e-30 or abs(b0) < 1e-30:
        _, _, surface, _ = operator_containers(op)
        b0_j, g_j, i_j = flux_surface_b_integrals(surface)
        b0, g, i = float(b0_j), float(g_j), float(i_j)
    return b0, g, i


def _base_fields(
    *,
    inp: SfincsInput,
    op: KineticOperator,
    grids: Grids,
    geom: FluxSurfaceGeometry,
    radial: RadialCoordinates,
    gpsipsi: np.ndarray,
    diotadpsi_hat: float,
    n_rhs: int,
) -> Dict[str, np.ndarray]:
    """Non-iteration datasets: sizes, grids, options, geometry, normalization."""
    gen, geo, phys, res, other, pre = (
        inp.general, inp.geometry, inp.physics, inp.resolution, inp.other, inp.preconditioner
    )
    scheme = geo.geometry_scheme
    rhs_mode = gen.rhs_mode
    b0_eff, g_eff, i_eff = _effective_flux_functions(op, geom)
    iota = float(geom.iota)

    int_scalars = (
        ("Nspecies", op.n_species), ("Ntheta", op.n_theta), ("Nzeta", op.n_zeta),
        ("Nxi", op.n_xi), ("NL", res.n_l), ("Nx", op.n_x),
        ("geometryScheme", scheme),
        ("thetaDerivativeScheme", other.theta_derivative_scheme),
        ("zetaDerivativeScheme", other.zeta_derivative_scheme),
        ("ExBDerivativeSchemeTheta", other.exb_derivative_scheme_theta),
        ("ExBDerivativeSchemeZeta", other.exb_derivative_scheme_zeta),
        ("magneticDriftDerivativeScheme", other.magnetic_drift_derivative_scheme),
        ("xGridScheme", other.x_grid_scheme),
        ("Nxi_for_x_option", other.n_xi_for_x_option),
        ("collisionOperator", phys.collision_operator),
        ("magneticDriftScheme", phys.magnetic_drift_scheme),
        ("inputRadialCoordinate", geo.input_radial_coordinate),
        ("coordinateSystem", 2 if scheme == 5 else 1),
        ("integerToRepresentFalse", -1), ("integerToRepresentTrue", 1),
        ("RHSMode", rhs_mode), ("NIterations", n_rhs),
        ("xPotentialsGridScheme", other.x_potentials_grid_scheme),
        ("preconditioner_species", pre.preconditioner_species),
        ("preconditioner_x", pre.preconditioner_x),
        ("preconditioner_x_min_L", pre.preconditioner_x_min_l),
        ("preconditioner_xi", pre.preconditioner_xi),
        ("preconditioner_theta", pre.preconditioner_theta),
        ("preconditioner_zeta", pre.preconditioner_zeta),
        ("preconditioner_magnetic_drifts_max_L", pre.preconditioner_magnetic_drifts_max_l),
        ("constraintScheme", op.constraint_scheme),
    )  # fmt: skip
    logicals = (
        ("includeXDotTerm", phys.include_x_dot_term),
        ("includeElectricFieldTermInXiDot", phys.include_electric_field_term_in_xi_dot),
        ("useDKESExBDrift", phys.use_dkes_exb_drift),
        ("useIterativeLinearSolver", other.use_iterative_linear_solver),
        ("finished", True), ("pointAtX0", op.point_at_x0),
        ("force0RadialCurrentInEquilibrium", True),
        ("includePhi1", phys.include_phi1),
        ("includePhi1InCollisionOperator", phys.include_phi1_in_collision_operator),
        ("includePhi1InKineticEquation", phys.include_phi1_in_kinetic_equation),
        ("includeTemperatureEquilibrationTerm", phys.include_temperature_equilibration_term),
        ("include_fDivVE_Term", phys.include_f_div_ve_term),
        ("withAdiabatic", inp.species.with_adiabatic),
        ("withNBIspec", inp.species.with_nbi_spec),
        ("reusePreconditioner", pre.reuse_preconditioner),
        # export_f: only the two logical flags for non-export runs (this slice).
        ("export_full_f", bool(inp.export_f.get("EXPORT_FULL_F", False))),
        ("export_delta_f", bool(inp.export_f.get("EXPORT_DELTA_F", False))),
    )  # fmt: skip
    float_scalars = (
        ("solverTolerance", res.solver_tolerance), ("Delta", phys.delta),
        ("alpha", phys.alpha), ("psiAHat", radial.psi_a_hat), ("aHat", radial.a_hat),
        ("psiN", radial.psi_n), ("psiHat", radial.psi_hat), ("rN", radial.r_n),
        ("rHat", radial.r_hat), ("EParallelHat", phys.e_parallel_hat),
        ("rippleScale", geo.ripple_scale), ("xMax", res.x_max),
        ("xGrid_k", other.x_grid_k), ("NxPotentialsPerVth", res.n_x_potentials_per_vth),
    )  # fmt: skip
    out: Dict[str, np.ndarray] = {
        "theta": _f64(grids.theta),
        "zeta": _f64(grids.zeta),
        "x": _f64(grids.x),
        "Nxi_for_x": np.asarray(grids.n_xi_for_x, dtype=_I32),
    }
    out.update({key: np.asarray(val, dtype=_I32) for key, val in int_scalars})
    out.update({key: _logical(val) for key, val in logicals})
    out.update({key: _f64(val) for key, val in float_scalars})

    if scheme == 1:
        for key in ("epsilon_t", "epsilon_h", "epsilon_antisymm"):
            out[key] = _f64(getattr(geo, key))
        for key in ("helicity_l", "helicity_n", "helicity_antisymm_l", "helicity_antisymm_n"):
            out[key] = np.asarray(getattr(geo, key), dtype=_I32)

    # nu_n (monoenergetic runs overwrite it from nuPrime, sfincs_main.F90).
    if rhs_mode == 3:
        out["nuPrime"] = _f64(phys.nu_prime)
        out["EStar"] = _f64(phys.e_star)
        out["nu_n"] = _f64(
            nu_n_from_nu_prime(
                nu_prime=phys.nu_prime, b0_over_bbar=b0_eff, g_hat=g_eff, i_hat=i_eff, iota=iota
            )
        )
    else:
        out["nu_n"] = _f64(phys.nu_n)

    # dPhiHat/d* family in all four radial coordinates; Er = -dPhiHatdrHat.
    dphi = float(op.dphi_hat_dpsi_hat)
    out["dPhiHatdpsiHat"] = _f64(dphi)
    out["dPhiHatdpsiN"] = _f64(radial.d_dpsi_hat_to_d_dpsi_n * dphi)
    out["dPhiHatdrHat"] = _f64(radial.d_dpsi_hat_to_d_dr_hat * dphi)
    out["dPhiHatdrN"] = _f64(radial.d_dpsi_hat_to_d_dr_n * dphi)
    out["Er"] = _f64(-float(out["dPhiHatdrHat"]))

    raw = inp.raw
    out["inputRadialCoordinateForGradients"] = np.asarray(
        infer_input_radial_coordinate_for_gradients(
            geom_params=raw.group("geometryParameters"),
            species_params=raw.group("speciesParameters"),
            phys_params=raw.group("physicsParameters"),
            default=4,
        ),
        dtype=_I32,
    )

    # Species arrays and gradients in all four radial coordinates.
    out["Zs"] = _f64(op.z_s)
    out["mHats"] = _f64(op.m_hat)
    out["THats"] = _f64(op.t_hat)
    out["nHats"] = _f64(op.n_hat)
    for name, grad in (("dnHatd", op.dn_hat_dpsi_hat), ("dTHatd", op.dt_hat_dpsi_hat)):
        g = _f64(grad)
        out[f"{name}psiHat"] = g
        out[f"{name}psiN"] = g * radial.d_dpsi_hat_to_d_dpsi_n
        out[f"{name}rHat"] = g * radial.d_dpsi_hat_to_d_dr_hat
        out[f"{name}rN"] = g * radial.d_dpsi_hat_to_d_dr_n

    # Geometry scalars and arrays.
    out["NPeriods"] = np.asarray(geom.n_periods, dtype=_I32)
    out["B0OverBBar"] = _f64(b0_eff)
    out["GHat"] = _f64(g_eff)
    out["IHat"] = _f64(i_eff)
    out["iota"] = _f64(iota)
    w2d = np.asarray(grids.theta_weights)[:, None] * np.asarray(grids.zeta_weights)[None, :]
    d_hat = np.asarray(geom.d_hat, dtype=np.float64)
    out["VPrimeHat"] = _f64(np.sum(w2d / d_hat))
    out["FSABHat2"] = _f64(op.fsab_hat2)
    out["gpsiHatpsiHat"] = _f64(gpsipsi)
    out["diotadpsiHat"] = _f64(diotadpsi_hat)
    out["uHat"] = (
        np.zeros_like(d_hat) if scheme == 5 else _u_hat(geom)
    )  # v3's VMEC path does not populate uHat.
    out["BDotCurlB"] = _f64(
        d_hat
        * (
            np.asarray(geom.b_hat_sub_theta) * np.asarray(geom.db_hat_sub_psi_dzeta)
            - np.asarray(geom.b_hat_sub_theta) * np.asarray(geom.db_hat_sub_zeta_dpsi_hat)
            + np.asarray(geom.b_hat_sub_zeta) * np.asarray(geom.db_hat_sub_theta_dpsi_hat)
            - np.asarray(geom.b_hat_sub_zeta) * np.asarray(geom.db_hat_sub_psi_dtheta)
        )
    )
    for key, value in (
        ("DHat", geom.d_hat), ("BHat", geom.b_hat), ("dBHatdpsiHat", geom.db_hat_dpsi_hat),
        ("dBHatdtheta", geom.db_hat_dtheta), ("dBHatdzeta", geom.db_hat_dzeta),
        ("BHat_sub_psi", geom.b_hat_sub_psi),
        ("dBHat_sub_psi_dtheta", geom.db_hat_sub_psi_dtheta),
        ("dBHat_sub_psi_dzeta", geom.db_hat_sub_psi_dzeta),
        ("BHat_sub_theta", geom.b_hat_sub_theta),
        ("dBHat_sub_theta_dpsiHat", geom.db_hat_sub_theta_dpsi_hat),
        ("dBHat_sub_theta_dzeta", geom.db_hat_sub_theta_dzeta),
        ("BHat_sub_zeta", geom.b_hat_sub_zeta),
        ("dBHat_sub_zeta_dpsiHat", geom.db_hat_sub_zeta_dpsi_hat),
        ("dBHat_sub_zeta_dtheta", geom.db_hat_sub_zeta_dtheta),
        ("BHat_sup_theta", geom.b_hat_sup_theta),
        ("dBHat_sup_theta_dpsiHat", geom.db_hat_sup_theta_dpsi_hat),
        ("dBHat_sup_theta_dzeta", geom.db_hat_sup_theta_dzeta),
        ("BHat_sup_zeta", geom.b_hat_sup_zeta),
        ("dBHat_sup_zeta_dpsiHat", geom.db_hat_sup_zeta_dpsi_hat),
        ("dBHat_sup_zeta_dtheta", geom.db_hat_sup_zeta_dtheta),
    ):  # fmt: skip
        out[key] = _f64(value)

    # Classical fluxes at the input gradients (Phi1 = 0), classicalTransport.F90.
    _, _, surface, species = operator_containers(op)
    pf0, hf0 = classical_fluxes(
        use_phi1=False, surface=surface, species=species,
        gpsipsi=gpsipsi, phi1_hat=np.zeros_like(gpsipsi),
        alpha=phys.alpha, delta=phys.delta, nu_n=float(out["nu_n"]),
        dn_hat_dpsi_hat=op.dn_hat_dpsi_hat, dt_hat_dpsi_hat=op.dt_hat_dpsi_hat,
    )  # fmt: skip
    for tag, arr in (("Particle", _f64(pf0)), ("Heat", _f64(hf0))):
        # Fluxes projected on grad(y) scale by dy/dpsiHat (radialCoordinates.F90).
        out[f"classical{tag}FluxNoPhi1_psiHat"] = arr
        out[f"classical{tag}FluxNoPhi1_psiN"] = arr * radial.d_dpsi_n_to_d_dpsi_hat
        out[f"classical{tag}FluxNoPhi1_rHat"] = arr * radial.d_dr_hat_to_d_dpsi_hat
        out[f"classical{tag}FluxNoPhi1_rN"] = arr * radial.d_dr_n_to_d_dpsi_hat

    return out


def _iteration_fields(
    *,
    inp: SfincsInput,
    op: KineticOperator,
    geom: FluxSurfaceGeometry,
    radial: RadialCoordinates,
    gpsipsi: np.ndarray,
    u_hat: np.ndarray,
    b0_eff: float,
    g_eff: float,
    i_eff: float,
    nu_n_eff: float,
    state_vectors: np.ndarray,
    transport_matrix: np.ndarray,
    elapsed_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-``whichRHS`` diagnostic datasets, stored in Fortran-file read order."""
    import jax.numpy as jnp  # noqa: PLC0415

    layout, vgrid, surface, species = operator_containers(op)
    n = int(state_vectors.shape[0])
    stack = jnp.asarray(state_vectors, dtype=jnp.float64)

    out: Dict[str, np.ndarray] = {
        key: np.asarray(val, dtype=np.float64)
        for key, val in transport_moments_table(
            layout, vgrid, surface, species, stack,
            rhs_mode=inp.general.rhs_mode, delta=op.delta, alpha=op.alpha,
        ).items()  # fmt: skip
    }

    # NTV kernel (zero for VMEC scheme 5, where v3 does not populate uHat).
    if inp.geometry.geometry_scheme == 5:
        kernel = jnp.zeros_like(jnp.asarray(op.b_hat))
    else:
        kernel = ntv_kernel(surface, u_hat=u_hat, g_hat=g_eff, i_hat=i_eff, iota=float(geom.iota))

    per_rhs_ztsn = (
        "densityPerturbation", "pressurePerturbation", "pressureAnisotropy", "flow",
        "totalDensity", "totalPressure", "velocityUsingFSADensity",
        "velocityUsingTotalDensity", "MachUsingFSAThermalSpeed",
        "momentumFluxBeforeSurfaceIntegral_vm", "momentumFluxBeforeSurfaceIntegral_vm0",
        "momentumFluxBeforeSurfaceIntegral_vE", "momentumFluxBeforeSurfaceIntegral_vE0",
        "NTVBeforeSurfaceIntegral",
    )  # fmt: skip
    per_rhs_sn = (
        "FSADensityPerturbation", "FSAPressurePerturbation",
        "momentumFlux_vm_psiHat", "momentumFlux_vm0_psiHat", "NTV",
    )  # fmt: skip
    s, t, z = op.n_species, op.n_theta, op.n_zeta
    for key in per_rhs_ztsn:
        out[key] = np.zeros((z, t, s, n), dtype=np.float64)
    for key in per_rhs_sn:
        out[key] = np.zeros((s, n), dtype=np.float64)
    out["jHat"] = np.zeros((z, t, n), dtype=np.float64)
    classical_pf = np.zeros((s, n), dtype=np.float64)
    classical_hf = np.zeros((s, n), dtype=np.float64)

    for which_rhs in range(1, n + 1):
        j = which_rhs - 1
        x_full = stack[j]
        d = rhsmode1_moments(
            layout, vgrid, surface, species, x_full, delta=op.delta, alpha=op.alpha
        )
        ntv_before, ntv_s = ntv_moments(layout, vgrid, surface, species, x_full, kernel=kernel)
        d = dict(d)
        d["NTVBeforeSurfaceIntegral"] = ntv_before
        d["NTV"] = ntv_s
        for key in per_rhs_ztsn:
            out[key][:, :, :, j] = np.transpose(np.asarray(d[key], dtype=np.float64), (2, 1, 0))
        for key in per_rhs_sn:
            out[key][:, j] = np.asarray(d[key], dtype=np.float64)
        out["jHat"][:, :, j] = np.transpose(np.asarray(d["jHat"], dtype=np.float64), (1, 0))

        op_rhs = op._with_rhs_settings(which_rhs)
        pf_j, hf_j = classical_fluxes(
            use_phi1=False, surface=surface, species=species,
            gpsipsi=gpsipsi, phi1_hat=np.zeros_like(gpsipsi),
            alpha=op.alpha, delta=op.delta, nu_n=nu_n_eff,
            dn_hat_dpsi_hat=op_rhs.dn_hat_dpsi_hat, dt_hat_dpsi_hat=op_rhs.dt_hat_dpsi_hat,
        )  # fmt: skip
        classical_pf[:, j] = np.asarray(pf_j, dtype=np.float64)
        classical_hf[:, j] = np.asarray(hf_j, dtype=np.float64)

    out["classicalParticleFlux_psiHat"] = classical_pf
    out["classicalHeatFlux_psiHat"] = classical_hf

    # Radial-coordinate flux variants (radialCoordinates.F90 projections).
    _add_radial_flux_variants(out, radial)

    # Transport matrix is stored transposed (Fortran column-major layout).
    out["transportMatrix"] = np.asarray(transport_matrix, dtype=np.float64).T
    out["elapsed time (s)"] = np.asarray(elapsed_times, dtype=np.float64).reshape((n,))
    return out


# Bases expanded to the psiN/rHat/rN radial-coordinate flux variants
# (radialCoordinates.F90 projections), shared by RHSMode 1 and 2/3.
_RADIAL_VARIANT_BASES = (
    "particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "momentumFlux_vm_psiHat",
    "particleFlux_vm0_psiHat", "heatFlux_vm0_psiHat", "momentumFlux_vm0_psiHat",
    "classicalParticleFlux_psiHat", "classicalHeatFlux_psiHat",
)  # fmt: skip


def _add_radial_flux_variants(out: Dict[str, np.ndarray], radial: RadialCoordinates) -> None:
    for base in _RADIAL_VARIANT_BASES:
        arr = out[base]
        out[base.replace("_psiHat", "_psiN")] = arr * radial.d_dpsi_n_to_d_dpsi_hat
        out[base.replace("_psiHat", "_rHat")] = arr * radial.d_dr_hat_to_d_dpsi_hat
        out[base.replace("_psiHat", "_rN")] = arr * radial.d_dr_n_to_d_dpsi_hat


# RHSMode=1 moment-table keys by sfincsOutput.h5 storage layout: (S,T,Z)
# canonical arrays land as (Z,T,S,1), (S,) as (S,1), (X,S) as (X,S,1).
_RHSMODE1_ZTSN_KEYS = (
    "densityPerturbation", "pressurePerturbation", "pressureAnisotropy", "flow",
    "totalDensity", "totalPressure", "velocityUsingFSADensity",
    "velocityUsingTotalDensity", "MachUsingFSAThermalSpeed",
    "particleFluxBeforeSurfaceIntegral_vm", "particleFluxBeforeSurfaceIntegral_vm0",
    "particleFluxBeforeSurfaceIntegral_vE", "particleFluxBeforeSurfaceIntegral_vE0",
    "heatFluxBeforeSurfaceIntegral_vm", "heatFluxBeforeSurfaceIntegral_vm0",
    "heatFluxBeforeSurfaceIntegral_vE", "heatFluxBeforeSurfaceIntegral_vE0",
    "momentumFluxBeforeSurfaceIntegral_vm", "momentumFluxBeforeSurfaceIntegral_vm0",
    "momentumFluxBeforeSurfaceIntegral_vE", "momentumFluxBeforeSurfaceIntegral_vE0",
    "NTVBeforeSurfaceIntegral",
)  # fmt: skip
_RHSMODE1_SN_KEYS = (
    "FSADensityPerturbation", "FSAPressurePerturbation", "FSABFlow",
    "FSABVelocityUsingFSADensity", "FSABVelocityUsingFSADensityOverB0",
    "FSABVelocityUsingFSADensityOverRootFSAB2", "NTV",
    "particleFlux_vm_psiHat", "particleFlux_vm0_psiHat",
    "heatFlux_vm_psiHat", "heatFlux_vm0_psiHat",
    "momentumFlux_vm_psiHat", "momentumFlux_vm0_psiHat",
)  # fmt: skip
_RHSMODE1_SCALAR_KEYS = ("FSABjHat", "FSABjHatOverB0", "FSABjHatOverRootFSAB2")
_RHSMODE1_XSN_KEYS = ("FSABFlow_vs_x", "particleFlux_vm_psiHat_vs_x", "heatFlux_vm_psiHat_vs_x")


def _rhsmode1_iteration_fields(
    *,
    inp: SfincsInput,
    op: KineticOperator,
    geom: FluxSurfaceGeometry,
    radial: RadialCoordinates,
    gpsipsi: np.ndarray,
    u_hat: np.ndarray,
    g_eff: float,
    i_eff: float,
    nu_n_eff: float,
    state_vector: np.ndarray,
    elapsed_seconds: float,
) -> Dict[str, np.ndarray]:
    """RHSMode=1 solution-derived datasets, stored in Fortran-file read order.

    The single iteration axis (``NIterations=1``) is the trailing axis of every
    per-iteration dataset, matching ``writeHDF5Output.F90``.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    layout, vgrid, surface, species = operator_containers(op)
    x_full = jnp.asarray(state_vector, dtype=jnp.float64)
    d = dict(
        rhsmode1_moments(
            layout, vgrid, surface, species, x_full,
            delta=op.delta, alpha=op.alpha, phi1_from_state=bool(op.include_phi1),
        )  # fmt: skip
    )

    # NTV torque (zero for VMEC scheme 5, where v3 does not populate uHat).
    if inp.geometry.geometry_scheme == 5:
        kernel = jnp.zeros_like(jnp.asarray(op.b_hat))
    else:
        kernel = ntv_kernel(surface, u_hat=u_hat, g_hat=g_eff, i_hat=i_eff, iota=float(geom.iota))
    ntv_before, ntv_s = ntv_moments(layout, vgrid, surface, species, x_full, kernel=kernel)
    d["NTVBeforeSurfaceIntegral"] = ntv_before
    d["NTV"] = ntv_s

    out: Dict[str, np.ndarray] = {}
    for key in _RHSMODE1_ZTSN_KEYS:
        out[key] = np.transpose(np.asarray(d[key], dtype=np.float64), (2, 1, 0))[:, :, :, None]
    for key in _RHSMODE1_SN_KEYS:
        out[key] = np.asarray(d[key], dtype=np.float64)[:, None]
    for key in _RHSMODE1_SCALAR_KEYS:
        out[key] = np.asarray(d[key], dtype=np.float64).reshape((1,))
    for key in _RHSMODE1_XSN_KEYS:
        out[key] = np.asarray(d[key], dtype=np.float64)[:, :, None]
    out["jHat"] = np.transpose(np.asarray(d["jHat"], dtype=np.float64), (1, 0))[:, :, None]
    if "sources" in d:
        out["sources"] = np.asarray(d["sources"], dtype=np.float64)[:, :, None]

    # Phi1Hat(theta,zeta): stored (Nzeta, Ntheta, NIterations) as in
    # writeHDF5Output.F90.  The canonical Newton solve records the converged
    # state (NIterations=1); the legacy in-process writer stored every accepted
    # Newton iterate.  Phi1-dependent electric-drift (vE) and total-drift (vd)
    # flux families plus dPhi1Hat gradients are the writer-consolidation
    # follow-up (enumerated in tests/test_phi1.py::_KNOWN_MISSING_PHI1_H5).
    phi1_hat = layout.phi1_hat(x_full)
    if phi1_hat is not None:
        out["Phi1Hat"] = np.transpose(np.asarray(phi1_hat, dtype=np.float64), (1, 0))[:, :, None]

    # Classical fluxes at the run's actual gradients (classicalTransport.F90).
    pf, hf = classical_fluxes(
        use_phi1=False, surface=surface, species=species,
        gpsipsi=gpsipsi, phi1_hat=np.zeros_like(gpsipsi),
        alpha=op.alpha, delta=op.delta, nu_n=nu_n_eff,
        dn_hat_dpsi_hat=op.dn_hat_dpsi_hat, dt_hat_dpsi_hat=op.dt_hat_dpsi_hat,
    )  # fmt: skip
    out["classicalParticleFlux_psiHat"] = np.asarray(pf, dtype=np.float64)[:, None]
    out["classicalHeatFlux_psiHat"] = np.asarray(hf, dtype=np.float64)[:, None]

    _add_radial_flux_variants(out, radial)
    out["elapsed time (s)"] = np.asarray([elapsed_seconds], dtype=np.float64)
    return out


def _write_h5(path: Path, base: Dict[str, np.ndarray], iteration: Dict[str, np.ndarray], text: str) -> None:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "w") as f:
        for key, value in base.items():
            f.create_dataset(key, data=_reversed_axes(value))
        for key, value in iteration.items():
            f.create_dataset(key, data=value)
        f.create_dataset("input.namelist", data=text)


def _write_netcdf(path: Path, base: Dict[str, np.ndarray], iteration: Dict[str, np.ndarray], text: str) -> None:
    from netCDF4 import Dataset  # noqa: PLC0415

    with Dataset(path, "w") as f:
        f.setncattr("input_namelist", text)
        dims: Dict[int, str] = {}

        def _dims_for(shape: tuple[int, ...]) -> tuple[str, ...]:
            names = []
            for size in shape:
                if size not in dims:
                    name = f"dim_{size}"
                    f.createDimension(name, size)
                    dims[size] = name
                names.append(dims[size])
            return tuple(names)

        for key, value in {**{k: _reversed_axes(v) for k, v in base.items()}, **iteration}.items():
            var = f.createVariable(key.replace("/", "_").replace(" ", "_"), value.dtype, _dims_for(value.shape))
            if value.ndim == 0:
                var.assignValue(value)
            else:
                var[:] = value


def write_transport_output(
    *,
    path: str | Path,
    inp: SfincsInput,
    op: KineticOperator,
    grids: Grids,
    geom: FluxSurfaceGeometry,
    radial: RadialCoordinates,
    state_vectors: np.ndarray,
    transport_matrix: np.ndarray,
    elapsed_times: np.ndarray | None = None,
) -> Path:
    """Write the RHSMode=2/3 ``sfincsOutput`` file from canonical-stack objects.

    Args:
        path: output file; ``.h5``/``.hdf5`` (or no suffix) writes HDF5 and
            ``.nc``/``.netcdf`` writes NetCDF4.
        inp: the validated typed input (``inputs.load_sfincs_input``).
        op: the canonical operator the states were solved with.
        grids: phase-space grids (theta/zeta nodes are not stored on ``op``).
        geom: full flux-surface geometry (radial-derivative fields included).
        radial: radial-coordinate conversion factors for the selected surface.
        state_vectors: solved states, shape ``(n_rhs, total_size)``.
        transport_matrix: the RHSMode=2 (3x3) or RHSMode=3 (2x2) matrix.
        elapsed_times: per-``whichRHS`` wall-clock seconds (``elapsed time (s)``).

    Returns:
        The resolved output path.
    """
    path = Path(path)
    if inp.general.rhs_mode not in (2, 3):
        raise NotImplementedError("write_transport_output supports RHSMode 2 and 3 only.")
    state_vectors = np.asarray(state_vectors, dtype=np.float64)
    n_rhs = int(state_vectors.shape[0])
    if elapsed_times is None:
        elapsed_times = np.zeros((n_rhs,), dtype=np.float64)

    gpsipsi, diotadpsi_hat = _geometry_extras(inp=inp, grids=grids, geom=geom, radial=radial)
    base = _base_fields(
        inp=inp, op=op, grids=grids, geom=geom, radial=radial,
        gpsipsi=gpsipsi, diotadpsi_hat=diotadpsi_hat, n_rhs=n_rhs,
    )  # fmt: skip
    iteration = _iteration_fields(
        inp=inp, op=op, geom=geom, radial=radial, gpsipsi=gpsipsi,
        u_hat=base["uHat"], b0_eff=float(base["B0OverBBar"]),
        g_eff=float(base["GHat"]), i_eff=float(base["IHat"]),
        nu_n_eff=float(base["nu_n"]), state_vectors=state_vectors,
        transport_matrix=transport_matrix, elapsed_times=np.asarray(elapsed_times),
    )  # fmt: skip

    text = inp.raw.source_text if (inp.raw is not None and inp.raw.source_text is not None) else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".nc", ".netcdf"}:
        _write_netcdf(path, base, iteration, text)
    else:
        _write_h5(path, base, iteration, text)
    return path.resolve()


def write_profile_output(
    *,
    path: str | Path,
    inp: SfincsInput,
    op: KineticOperator,
    grids: Grids,
    geom: FluxSurfaceGeometry,
    radial: RadialCoordinates,
    state_vector: np.ndarray,
    elapsed_seconds: float = 0.0,
) -> Path:
    """Write the RHSMode=1 ``sfincsOutput`` file from canonical-stack objects.

    Same non-iteration machinery as :func:`write_transport_output`; the
    iteration datasets carry the full RHSMode=1 per-species moment table
    (:func:`sfincs_jax.moments.rhsmode1_moments`), NTV, and the classical
    fluxes at the run's gradients, all with a single trailing iteration axis
    (``NIterations=1``).  The deferred Phi1 / magnetic-drift / export_f data
    families are intentionally not written.

    Args:
        path: output file; ``.h5``/``.hdf5`` (or no suffix) writes HDF5 and
            ``.nc``/``.netcdf`` writes NetCDF4.
        inp: the validated typed input (``inputs.load_sfincs_input``).
        op: the canonical operator the state was solved with.
        grids: phase-space grids (theta/zeta nodes are not stored on ``op``).
        geom: full flux-surface geometry (radial-derivative fields included).
        radial: radial-coordinate conversion factors for the selected surface.
        state_vector: the solved state, shape ``(total_size,)``.
        elapsed_seconds: wall-clock seconds for ``elapsed time (s)``.

    Returns:
        The resolved output path.
    """
    path = Path(path)
    if inp.general.rhs_mode != 1:
        raise NotImplementedError("write_profile_output supports RHSMode=1 only.")
    state_vector = np.asarray(state_vector, dtype=np.float64).reshape((-1,))

    gpsipsi, diotadpsi_hat = _geometry_extras(inp=inp, grids=grids, geom=geom, radial=radial)
    base = _base_fields(
        inp=inp, op=op, grids=grids, geom=geom, radial=radial,
        gpsipsi=gpsipsi, diotadpsi_hat=diotadpsi_hat, n_rhs=1,
    )  # fmt: skip
    iteration = _rhsmode1_iteration_fields(
        inp=inp, op=op, geom=geom, radial=radial, gpsipsi=gpsipsi,
        u_hat=base["uHat"], g_eff=float(base["GHat"]), i_eff=float(base["IHat"]),
        nu_n_eff=float(base["nu_n"]), state_vector=state_vector,
        elapsed_seconds=elapsed_seconds,
    )  # fmt: skip

    text = inp.raw.source_text if (inp.raw is not None and inp.raw.source_text is not None) else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".nc", ".netcdf"}:
        _write_netcdf(path, base, iteration, text)
    else:
        _write_h5(path, base, iteration, text)
    return path.resolve()
