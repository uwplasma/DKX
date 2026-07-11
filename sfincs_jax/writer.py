"""Canonical ``sfincsOutput`` writer for RHSMode=1/2/3 runs.

Fortran counterpart: ``writeHDF5Output.F90`` (dataset names, shapes, and the
Fortran column-major storage layout) plus the per-``whichRHS`` diagnostic
writes of ``diagnostics.F90``.  This module is the canonical-stack replacement
for the legacy ``outputs/writer.py`` + ``outputs/transport.py`` +
``outputs/rhsmode1.py`` pipeline for the supported case families: it consumes
only canonical objects (:mod:`sfincs_jax.inputs`,
:mod:`sfincs_jax.drift_kinetic`, :mod:`sfincs_jax.moments`,
:mod:`sfincs_jax.magnetic_geometry`) and writes the same datasets the legacy
writer emits for these modes, including the ``export_f`` distribution-function
family (``full_f``/``delta_f`` on the user grids).  The remaining Phi1 electric
-drift flux families the legacy writer emits are the writer-consolidation
follow-up.

Layout convention: like the legacy writer (``fortran_layout=True``),
grid/geometry/normalization fields are stored reversed-transposed (Fortran
column-major view of a ``(T, Z)`` array reads as ``(Z, T)`` in h5py), while the
per-``whichRHS`` diagnostic fields are stored directly in the Python-read
order of Fortran v3 files (``(Z, T, S, N)``, ``(X, S, N)``, ``(S, N)``).

Units/normalizations: every dataset is the dimensionless v3 "Hat" quantity;
see :mod:`sfincs_jax.moments` for the flux conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    combined_drift_fluxes,
    electric_drift_flux_moments,
    flux_surface_b_integrals,
    heat_flux_without_phi1,
    ntv_kernel,
    ntv_moments,
    rhsmode1_moments,
    transport_moments_table,
)
from sfincs_jax.phase_space import Grids

__all__ = [
    "operator_containers",
    "write_profile_output",
    "write_run_solver_trace",
    "write_transport_output",
]

_I32 = np.int32


def _logical(value: bool) -> np.ndarray:
    """v3 integerToRepresentTrue/False encoding of a Fortran logical."""
    return np.asarray(1 if value else -1, dtype=_I32)


def _f64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _reversed_axes(arr: np.ndarray) -> np.ndarray:
    """Fortran column-major storage view of a row-major logical array."""
    if not isinstance(arr, np.ndarray) or arr.ndim <= 1:
        return arr
    return np.ascontiguousarray(np.transpose(arr, tuple(reversed(range(arr.ndim)))))


def _namelist_float(group: dict, key: str, default: float) -> float:
    """Case-insensitive namelist float read (list-valued entries take the first)."""
    value = group.get(key.upper(), default)
    if isinstance(value, list):
        value = value[0] if value else default
    return float(value)


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

    Analytic schemes (1/2/4/13) store zeros, matching v3.  Boozer ``.bc`` and
    VMEC schemes recompute the |grad psiHat|^2 metric with the canonical geometry
    constructors and the iota radial derivative from the bracketing surfaces.
    """
    scheme = inp.geometry.geometry_scheme
    zeros = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    if scheme in {1, 2, 4, 13}:
        # geometryScheme=13 is analytic (nearbyRadiiGiven=.false. in geometry.F90):
        # no nearby surfaces, so the |grad psiHat|^2 metric and diota/dpsiHat are 0.
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

    # Phi1 scalar metadata (writeHDF5Output.F90): v3 writes quasineutralityOption
    # and readExternalPhi1 for every Phi1 run; the adiabatic-species parameters
    # only when withAdiabatic is active.  readExternalPhi1 keeps the f-only linear
    # layout (op.include_phi1 is False) but is still a Phi1 run.
    if bool(op.include_phi1) or op.external_phi1_hat is not None:
        out["quasineutralityOption"] = np.asarray(int(op.quasineutrality_option), dtype=_I32)
        out["readExternalPhi1"] = _logical(op.external_phi1_hat is not None)
        if bool(op.with_adiabatic):
            spec = inp.raw.group("speciesParameters") if inp.raw is not None else {}
            out["adiabaticZ"] = _f64(float(op.adiabatic_z))
            out["adiabaticNHat"] = _f64(float(op.adiabatic_n_hat))
            out["adiabaticTHat"] = _f64(float(op.adiabatic_t_hat))
            # v3 default adiabaticMHat = m_e/m_p; not carried on the operator.
            out["adiabaticMHat"] = _f64(_namelist_float(spec, "adiabaticMHat", 5.446170214e-4))

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


def _add_radial_flux_variants(
    out: Dict[str, np.ndarray], radial: RadialCoordinates, bases: Tuple[str, ...] = _RADIAL_VARIANT_BASES
) -> None:
    for base in bases:
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


# Phi1 electric-drift (vE/vE0) and total-drift (vd/vd1) psiHat flux bases whose
# psiN/rHat/rN variants are the radialCoordinates.F90 projections (diagnostics.F90
# vE/vd family), plus the heat-only withoutPhi1 flux (heatFlux_vm + (5/3) heatFlux_vE0).
_PHI1_FLUX_VARIANT_BASES = tuple(
    f"{fam}Flux_{drift}_psiHat"
    for fam in ("particle", "heat", "momentum")
    for drift in ("vE", "vE0", "vd", "vd1")
) + ("heatFlux_withoutPhi1_psiHat",)


def _add_phi1_drift_output_fields(
    out: Dict[str, np.ndarray],
    *,
    op: KineticOperator,
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full: Any,
    phi1_hat: Any,
    moments: Dict[str, Any],
    radial: RadialCoordinates,
    include_lambda: bool = True,
) -> None:
    """Phi1-only electric-/total-drift flux families, dPhi1Hat gradients, lambda.

    Mirrors ``diagnostics.F90`` (and the legacy ``outputs/rhsmode1.py``
    ``write_rhsmode1_electric_drift_diagnostics_to_data`` template): the ExB
    (vE/vE0) flux moments come from :func:`electric_drift_flux_moments`, the
    total drift ``(vd1, vd) = (vm + vE0, vm + vE)`` from
    :func:`combined_drift_fluxes`, and ``heatFlux_withoutPhi1`` from
    :func:`heat_flux_without_phi1`.  The Phi1 gradients reuse the operator's
    theta/zeta differentiation matrices (the same ``ddtheta``/``ddzeta`` the DKE
    streaming term uses).  All per-iteration arrays carry the trailing
    ``NIterations=1`` axis in the ``sfincsOutput.h5`` read order.
    """
    phi1 = np.asarray(phi1_hat, dtype=np.float64)
    ddtheta = np.asarray(op.ddtheta, dtype=np.float64)
    ddzeta = np.asarray(op.ddzeta, dtype=np.float64)
    dphi1_dtheta = ddtheta @ phi1  # (T, Z)
    dphi1_dzeta = phi1 @ ddzeta.T  # (T, Z)
    out["dPhi1Hatdtheta"] = np.transpose(dphi1_dtheta, (1, 0))[:, :, None]
    out["dPhi1Hatdzeta"] = np.transpose(dphi1_dzeta, (1, 0))[:, :, None]

    ve = electric_drift_flux_moments(
        layout, vgrid, surface, species, x_full,
        delta=op.delta, alpha=op.alpha, phi1_hat=phi1_hat,
        dphi1_hat_dtheta=dphi1_dtheta, dphi1_hat_dzeta=dphi1_dzeta,
    )  # fmt: skip

    # Overwrite the vE/vE0 BeforeSurfaceIntegral placeholders (zeros for the
    # non-Phi1 moment table) with the ExB values, stored (Z, T, S, 1).
    for key, arr in (
        ("particleFluxBeforeSurfaceIntegral_vE", ve.particle_flux_before_surface_integral_ve),
        ("particleFluxBeforeSurfaceIntegral_vE0", ve.particle_flux_before_surface_integral_ve0),
        ("heatFluxBeforeSurfaceIntegral_vE", ve.heat_flux_before_surface_integral_ve),
        ("heatFluxBeforeSurfaceIntegral_vE0", ve.heat_flux_before_surface_integral_ve0),
        ("momentumFluxBeforeSurfaceIntegral_vE", ve.momentum_flux_before_surface_integral_ve),
        ("momentumFluxBeforeSurfaceIntegral_vE0", ve.momentum_flux_before_surface_integral_ve0),
    ):  # fmt: skip
        out[key] = np.transpose(np.asarray(arr, dtype=np.float64), (2, 1, 0))[:, :, :, None]

    # Surface-integrated vE/vE0 and total-drift vd/vd1 psiHat fluxes, stored (S, 1).
    for fam, ve_s, ve0_s in (
        ("particleFlux", ve.particle_flux_ve_psi_hat, ve.particle_flux_ve0_psi_hat),
        ("heatFlux", ve.heat_flux_ve_psi_hat, ve.heat_flux_ve0_psi_hat),
        ("momentumFlux", ve.momentum_flux_ve_psi_hat, ve.momentum_flux_ve0_psi_hat),
    ):
        vm_s = np.asarray(moments[f"{fam}_vm_psiHat"], dtype=np.float64)
        ve_a = np.asarray(ve_s, dtype=np.float64)
        ve0_a = np.asarray(ve0_s, dtype=np.float64)
        vd1, vd = combined_drift_fluxes(flux_vm_psi_hat=vm_s, flux_ve0_psi_hat=ve0_a, flux_ve_psi_hat=ve_a)
        out[f"{fam}_vE_psiHat"] = ve_a[:, None]
        out[f"{fam}_vE0_psiHat"] = ve0_a[:, None]
        out[f"{fam}_vd_psiHat"] = np.asarray(vd, dtype=np.float64)[:, None]
        out[f"{fam}_vd1_psiHat"] = np.asarray(vd1, dtype=np.float64)[:, None]

    hf_wo = heat_flux_without_phi1(
        heat_flux_vm_psi_hat=np.asarray(moments["heatFlux_vm_psiHat"], dtype=np.float64),
        heat_flux_ve0_psi_hat=np.asarray(ve.heat_flux_ve0_psi_hat, dtype=np.float64),
    )
    out["heatFlux_withoutPhi1_psiHat"] = np.asarray(hf_wo, dtype=np.float64)[:, None]

    _add_radial_flux_variants(out, radial, _PHI1_FLUX_VARIANT_BASES)

    # <Phi1>=0 Lagrange multiplier (state row after the Phi1(theta,zeta) block).
    # readExternalPhi1 has no lambda row (the field is fixed, not solved).
    if include_lambda:
        lam = float(np.asarray(x_full, dtype=np.float64)[layout.f_size + layout.n_theta * layout.n_zeta])
        out["lambda"] = np.asarray([lam], dtype=np.float64)


def _add_phi1_solver_metadata(
    base: Dict[str, np.ndarray],
    *,
    converged: bool | None,
    solver_method: str | None,
    solver_requested_method: str | None,
    residual_norm: float | None,
) -> None:
    """Phi1 Newton-solve metadata (writeHDF5Output.F90 nonlinear-run fields).

    ``didNonlinearCalculationConverge`` is the Newton convergence flag; the
    ``linearSolver*`` fields describe the inner linear solve.  The method names
    are stored as strings (skipped by the io name-map value comparison); the
    residual norm is the converged solve residual (near zero for both the
    canonical Newton and the legacy inner GMRES/dense solve).
    """
    if converged is not None:
        base["didNonlinearCalculationConverge"] = _logical(bool(converged))
    if solver_method is not None:
        base["linearSolverMethod"] = str(solver_method)
        base["linearSolverRequestedMethod"] = str(solver_requested_method or solver_method)
    if residual_norm is not None:
        base["linearSolverResidualNorm"] = _f64(float(residual_norm))


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
            phi1_hat=op.external_phi1_hat,
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
    # Newton iterate.  For Phi1 runs the electric-drift (vE/vE0) and total-drift
    # (vd/vd1) flux families, the heat withoutPhi1 flux, the dPhi1Hat gradients,
    # and the lambda multiplier are emitted here from the canonical moments
    # (diagnostics.F90 parity); the Phi1 scalar/solver metadata is added by
    # ``_base_fields``/``write_profile_output``.
    # For readExternalPhi1 the Phi1 field is fixed (not in the state); the
    # self-consistent path extracts it from the state.  Either way it drives the
    # same electric-/total-drift flux families.
    phi1_hat = op.external_phi1_hat if op.external_phi1_hat is not None else layout.phi1_hat(x_full)
    if phi1_hat is not None:
        out["Phi1Hat"] = np.transpose(np.asarray(phi1_hat, dtype=np.float64), (1, 0))[:, :, None]
        _add_phi1_drift_output_fields(
            out, op=op, layout=layout, vgrid=vgrid, surface=surface, species=species,
            x_full=x_full, phi1_hat=phi1_hat, moments=d, radial=radial,
            include_lambda=op.external_phi1_hat is None,
        )  # fmt: skip

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


def _write_npz(path: Path, base: Dict[str, np.ndarray], iteration: Dict[str, np.ndarray], text: str) -> None:
    """Write an uncompressed ``.npz`` archive with the SFINCS Fortran readback layout.

    Mirrors :func:`_write_h5`: ``base`` grid/geometry/normalization fields are
    stored reversed-transposed (Fortran column-major view), the per-``whichRHS``
    diagnostic fields are stored directly, and the ``input.namelist`` source
    text is stored as a 0-d string array.  This is the canonical-stack
    replacement for the legacy ``outputs.formats.write_sfincs_npz`` path so
    ``--out *.npz`` no longer needs the legacy writer.
    """
    payload: Dict[str, np.ndarray] = {}
    for key, value in base.items():
        arr = _reversed_axes(np.asarray(value))
        if arr.ndim > 0 and arr.dtype.kind != "O":
            arr = np.ascontiguousarray(arr)
        payload[key] = arr
    for key, value in iteration.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.dtype.kind != "O":
            arr = np.ascontiguousarray(arr)
        payload[key] = arr
    payload["input.namelist"] = np.asarray(text)
    np.savez(path, **payload)


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
            safe_key = key.replace("/", "_").replace(" ", "_")
            if isinstance(value, str) or (isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}):
                f.setncattr(safe_key, str(value))
                continue
            var = f.createVariable(safe_key, value.dtype, _dims_for(value.shape))
            if value.ndim == 0:
                var.assignValue(value)
            else:
                var[:] = value


def _write_output_by_suffix(
    path: Path, base: Dict[str, np.ndarray], iteration: Dict[str, np.ndarray], text: str
) -> None:
    """Dispatch to the h5/netcdf/npz writer selected by ``path``'s suffix."""
    suffix = path.suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        _write_netcdf(path, base, iteration, text)
    elif suffix == ".npz":
        _write_npz(path, base, iteration, text)
    else:
        _write_h5(path, base, iteration, text)


# ---------------------------------------------------------------------------
# export_f: distribution-function export on the user grids (export_f.F90).
#
# Ported from the legacy ``outputs.formats`` export_f mapping so the canonical
# writer emits the full_f/delta_f data family (and the export_f_* grid/option
# datasets) directly from the canonical solved state, without the legacy
# outputs pipeline.  The maps and Legendre/interpolation kernels are pure NumPy;
# the ``export_f_x_option=1`` interpolation reuses the canonical
# :func:`sfincs_jax.collisions.polynomial_interpolation_matrix_np`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExportFConfig:
    """Axis maps and metadata for Fortran-compatible ``export_f`` output."""

    export_full_f: bool
    export_delta_f: bool
    theta_option: int
    zeta_option: int
    x_option: int
    xi_option: int
    export_theta: np.ndarray
    export_zeta: np.ndarray
    export_x: np.ndarray
    export_xi: Optional[np.ndarray]
    n_export_theta: int
    n_export_zeta: int
    n_export_x: int
    n_export_xi: int
    map_theta: np.ndarray
    map_zeta: np.ndarray
    map_x: np.ndarray
    map_xi: np.ndarray


def _export_f_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _export_f_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def _export_f_1d(group: dict, key: str, *, default: float) -> np.ndarray:
    k = key.upper()
    if k not in group:
        return np.atleast_1d(np.asarray([default], dtype=np.float64))
    return np.atleast_1d(np.asarray(group[k], dtype=np.float64))


def _export_f_legendre_matrix(xi: np.ndarray, *, n_l: int) -> np.ndarray:
    """Evaluate Legendre polynomials ``P_0`` .. ``P_{n_l-1}`` at ``xi`` (rows=xi)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    if n_l < 1:
        raise ValueError("n_l must be >= 1")
    out = np.zeros((xi.size, n_l), dtype=np.float64)
    out[:, 0] = 1.0
    if n_l == 1:
        return out
    out[:, 1] = xi
    for ell in range(2, n_l):
        out[:, ell] = ((2 * ell - 1) * xi * out[:, ell - 1] - (ell - 1) * out[:, ell - 2]) / float(ell)
    return out


def _export_f_config(*, raw, grids: Grids, geom: FluxSurfaceGeometry) -> _ExportFConfig | None:
    """Build Fortran-compatible ``export_f`` axis maps (export_f.F90 semantics)."""
    export_f = raw.group("export_f")
    export_full_f = bool(export_f.get("EXPORT_FULL_F", False))
    export_delta_f = bool(export_f.get("EXPORT_DELTA_F", False))
    if not (export_full_f or export_delta_f):
        return None

    theta_option = _export_f_int(export_f, "EXPORT_F_THETA_OPTION", 2)
    zeta_option = _export_f_int(export_f, "EXPORT_F_ZETA_OPTION", 2)
    xi_option = _export_f_int(export_f, "EXPORT_F_XI_OPTION", 1)
    x_option = _export_f_int(export_f, "EXPORT_F_X_OPTION", 0)

    export_theta = _export_f_1d(export_f, "EXPORT_F_THETA", default=0.0)
    export_zeta = _export_f_1d(export_f, "EXPORT_F_ZETA", default=0.0)
    export_xi = _export_f_1d(export_f, "EXPORT_F_XI", default=0.0)
    export_x = _export_f_1d(export_f, "EXPORT_F_X", default=1.0)

    theta = np.asarray(grids.theta, dtype=np.float64)
    zeta = np.asarray(grids.zeta, dtype=np.float64)
    x = np.asarray(grids.x, dtype=np.float64)
    n_theta = int(theta.size)
    n_zeta = int(zeta.size)
    n_x = int(x.size)
    n_xi = int(grids.n_xi)

    if theta_option == 0:
        export_theta = theta.copy()
        map_theta = np.eye(n_theta, dtype=np.float64)
    elif theta_option == 1:
        export_theta = np.mod(export_theta, 2.0 * math.pi)
        map_theta = np.zeros((export_theta.size, n_theta), dtype=np.float64)
        for j, val in enumerate(export_theta):
            idx1 = int(math.floor(val * n_theta / (2.0 * math.pi))) + 1
            if idx1 < 1:
                raise ValueError(f"Invalid export_f_theta index for value {val}")
            if idx1 == n_theta + 1:
                idx1, idx2 = n_theta, 1
            elif idx1 == n_theta:
                idx2 = 1
            elif idx1 > n_theta + 1:
                raise ValueError(f"Invalid export_f_theta index for value {val}")
            else:
                idx2 = idx1 + 1
            weight1 = idx1 - val * n_theta / (2.0 * math.pi)
            map_theta[j, idx1 - 1] = weight1
            map_theta[j, idx2 - 1] = 1.0 - weight1
    elif theta_option == 2:
        export_theta = np.mod(export_theta, 2.0 * math.pi)
        include = np.zeros((n_theta,), dtype=bool)
        for val in export_theta:
            err = np.minimum.reduce(
                [(val - theta) ** 2, (val - theta - 2.0 * math.pi) ** 2, (val - theta + 2.0 * math.pi) ** 2]
            )
            include[int(np.argmin(err))] = True
        export_theta = theta[include].copy()
        map_theta = np.zeros((export_theta.size, n_theta), dtype=np.float64)
        for row_idx, j in enumerate(np.where(include)[0]):
            map_theta[row_idx, j] = 1.0
    else:
        raise ValueError("Invalid export_f_theta_option")

    if n_zeta == 1:
        export_zeta = np.asarray([0.0], dtype=np.float64)
        map_zeta = np.ones((1, 1), dtype=np.float64)
    else:
        zeta_period = 2.0 * math.pi / float(geom.n_periods)
        if zeta_option == 0:
            export_zeta = zeta.copy()
            map_zeta = np.eye(n_zeta, dtype=np.float64)
        elif zeta_option == 1:
            export_zeta = np.mod(export_zeta, zeta_period)
            map_zeta = np.zeros((export_zeta.size, n_zeta), dtype=np.float64)
            for j, val in enumerate(export_zeta):
                idx1 = int(math.floor(val * n_zeta / zeta_period)) + 1
                if idx1 < 1:
                    raise ValueError(f"Invalid export_f_zeta index for value {val}")
                if idx1 == n_zeta + 1:
                    idx1, idx2 = n_zeta, 1
                elif idx1 == n_zeta:
                    idx2 = 1
                elif idx1 > n_zeta + 1:
                    raise ValueError(f"Invalid export_f_zeta index for value {val}")
                else:
                    idx2 = idx1 + 1
                weight1 = idx1 - val * n_zeta / zeta_period
                map_zeta[j, idx1 - 1] = weight1
                map_zeta[j, idx2 - 1] = 1.0 - weight1
        elif zeta_option == 2:
            export_zeta = np.mod(export_zeta, zeta_period)
            include = np.zeros((n_zeta,), dtype=bool)
            for val in export_zeta:
                err = np.minimum.reduce(
                    [(val - zeta) ** 2, (val - zeta - zeta_period) ** 2, (val - zeta + zeta_period) ** 2]
                )
                include[int(np.argmin(err))] = True
            export_zeta = zeta[include].copy()
            map_zeta = np.zeros((export_zeta.size, n_zeta), dtype=np.float64)
            for row_idx, j in enumerate(np.where(include)[0]):
                map_zeta[row_idx, j] = 1.0
        else:
            raise ValueError("Invalid export_f_zeta_option")

    if x_option == 0:
        export_x = x.copy()
        map_x = np.eye(n_x, dtype=np.float64)
    elif x_option == 1:
        from sfincs_jax.collisions import polynomial_interpolation_matrix_np  # noqa: PLC0415

        other = raw.group("otherNumericalParameters")
        x_grid_scheme = _export_f_int(other, "XGRIDSCHEME", _export_f_int(other, "xGridScheme", 5))
        x_grid_k = _export_f_float(other, "xGrid_k", 0.0)
        if x_grid_scheme not in {1, 2, 5, 6}:
            raise NotImplementedError(
                f"export_f_x_option=1 is only implemented for xGridScheme in {{1,2,5,6}} (got {x_grid_scheme})."
            )
        alpxk = np.exp(-(x * x)) * (x**x_grid_k)
        alpx = np.exp(-(export_x * export_x)) * (export_x**x_grid_k)
        map_x = polynomial_interpolation_matrix_np(xk=x, x=export_x, alpxk=alpxk, alpx=alpx)
    elif x_option == 2:
        include = np.zeros((n_x,), dtype=bool)
        for val in export_x:
            include[int(np.argmin((val - x) ** 2))] = True
        export_x = x[include].copy()
        map_x = np.zeros((export_x.size, n_x), dtype=np.float64)
        for row_idx, j in enumerate(np.where(include)[0]):
            map_x[row_idx, j] = 1.0
    else:
        raise ValueError("Invalid export_f_x_option")

    if xi_option == 0:
        map_xi = np.eye(n_xi, dtype=np.float64)
        export_xi_out: Optional[np.ndarray] = None
        n_export_xi = n_xi
    elif xi_option == 1:
        map_xi = _export_f_legendre_matrix(export_xi, n_l=n_xi)
        export_xi_out = export_xi.copy()
        n_export_xi = int(export_xi.size)
    else:
        raise ValueError("Invalid export_f_xi_option")

    return _ExportFConfig(
        export_full_f=export_full_f, export_delta_f=export_delta_f,
        theta_option=int(theta_option), zeta_option=int(zeta_option),
        x_option=int(x_option), xi_option=int(xi_option),
        export_theta=np.asarray(export_theta, dtype=np.float64),
        export_zeta=np.asarray(export_zeta, dtype=np.float64),
        export_x=np.asarray(export_x, dtype=np.float64),
        export_xi=export_xi_out,
        n_export_theta=int(export_theta.size), n_export_zeta=int(export_zeta.size),
        n_export_x=int(export_x.size), n_export_xi=int(n_export_xi),
        map_theta=map_theta, map_zeta=map_zeta, map_x=map_x, map_xi=map_xi,
    )  # fmt: skip


def _apply_export_f_maps(f: np.ndarray, cfg: _ExportFConfig) -> np.ndarray:
    """Apply the ``export_f`` maps to ``(S, X, L, theta, zeta)`` -> ``(S, x, xi, theta, zeta)`` export axes."""
    f = np.asarray(f, dtype=np.float64)
    f = np.einsum("ax,sxltz->saltz", cfg.map_x, f, optimize=True)
    f = np.einsum("bl,saltz->sabtz", cfg.map_xi, f, optimize=True)
    f = np.einsum("ct,sabtz->sabcz", cfg.map_theta, f, optimize=True)
    f = np.einsum("dz,sabcz->sabcd", cfg.map_zeta, f, optimize=True)
    return f


def _f0_l0_maxwellian(op: KineticOperator) -> np.ndarray:
    """v3 ``f0`` at ``L=0`` for the export full_f base, shape ``(S, X, theta, zeta)``.

    Mirrors ``export_f.F90``/``transport_diagnostics.f0_l0_v3_from_operator``:
    a shifted Maxwellian ``n m/(pi T) sqrt(m/(pi T)) exp(-x^2) exp(-Z alpha Phi1 / T)``.
    """
    x = np.asarray(op.x, dtype=np.float64)
    expx2 = np.exp(-(x * x))
    z = np.asarray(op.z_s, dtype=np.float64)
    n_hat = np.asarray(op.n_hat, dtype=np.float64)
    t_hat = np.asarray(op.t_hat, dtype=np.float64)
    m_hat = np.asarray(op.m_hat, dtype=np.float64)
    pref = n_hat[:, None] * m_hat[:, None] / (np.pi * t_hat[:, None])
    pref = pref * np.sqrt(m_hat[:, None] / (np.pi * t_hat[:, None])) * expx2[None, :]  # (S, X)
    phi1_base = getattr(op, "phi1_hat_base", None)
    if phi1_base is None:
        phi1 = np.zeros((int(op.n_theta), int(op.n_zeta)), dtype=np.float64)
    else:
        phi1 = np.asarray(phi1_base, dtype=np.float64)
    exp_phi1 = np.exp(-(z[:, None, None] * float(op.alpha) / t_hat[:, None, None]) * phi1[None, :, :])  # (S, T, Z)
    return pref[:, :, None, None] * exp_phi1[:, None, :, :]  # (S, X, T, Z)


def _export_f_data(
    *, op: KineticOperator, state_vectors: List[np.ndarray], cfg: _ExportFConfig
) -> Dict[str, np.ndarray]:
    """Compute ``full_f``/``delta_f`` on the export grids, in the stored Fortran read order.

    The stored order is ``(x_export, xi_export, zeta_export, theta_export,
    species, iteration)`` (the canonical writer stores iteration datasets
    directly, so no additional layout reversal is applied here).
    """
    f_size = int(op.f_size)
    f_shape = tuple(op.f_shape)
    f0_l0 = _f0_l0_maxwellian(op) if cfg.export_full_f else None

    delta_list: List[np.ndarray] = []
    full_list: List[np.ndarray] = []
    for x_full in state_vectors:
        f_delta = np.asarray(x_full[:f_size], dtype=np.float64).reshape(f_shape)
        if cfg.export_delta_f:
            delta_np = _apply_export_f_maps(f_delta, cfg)
            delta_list.append(np.transpose(delta_np, (1, 2, 4, 3, 0)))
        if cfg.export_full_f:
            f_full = np.array(f_delta, dtype=np.float64, copy=True)
            f_full[:, :, 0, :, :] += f0_l0
            full_np = _apply_export_f_maps(f_full, cfg)
            full_list.append(np.transpose(full_np, (1, 2, 4, 3, 0)))

    out: Dict[str, np.ndarray] = {}
    if cfg.export_delta_f and delta_list:
        out["delta_f"] = np.ascontiguousarray(np.stack(delta_list, axis=-1))
    if cfg.export_full_f and full_list:
        out["full_f"] = np.ascontiguousarray(np.stack(full_list, axis=-1))
    return out


def _add_export_f_datasets(
    base: Dict[str, np.ndarray],
    iteration: Dict[str, np.ndarray],
    *,
    op: KineticOperator,
    state_vectors: List[np.ndarray],
    cfg: _ExportFConfig,
) -> None:
    """Add the export_f grid/option datasets to ``base`` and full_f/delta_f to ``iteration``."""
    base["export_f_theta_option"] = np.asarray(cfg.theta_option, dtype=_I32)
    base["export_f_zeta_option"] = np.asarray(cfg.zeta_option, dtype=_I32)
    base["export_f_x_option"] = np.asarray(cfg.x_option, dtype=_I32)
    base["export_f_xi_option"] = np.asarray(cfg.xi_option, dtype=_I32)
    base["export_f_theta"] = np.asarray(cfg.export_theta, dtype=np.float64)
    base["export_f_zeta"] = np.asarray(cfg.export_zeta, dtype=np.float64)
    base["export_f_x"] = np.asarray(cfg.export_x, dtype=np.float64)
    base["N_export_f_theta"] = np.asarray(cfg.n_export_theta, dtype=_I32)
    base["N_export_f_zeta"] = np.asarray(cfg.n_export_zeta, dtype=_I32)
    base["N_export_f_x"] = np.asarray(cfg.n_export_x, dtype=_I32)
    if cfg.export_xi is not None:
        base["export_f_xi"] = np.asarray(cfg.export_xi, dtype=np.float64)
        base["N_export_f_xi"] = np.asarray(cfg.n_export_xi, dtype=_I32)
    iteration.update(_export_f_data(op=op, state_vectors=state_vectors, cfg=cfg))


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

    export_cfg = _export_f_config(raw=inp.raw, grids=grids, geom=geom) if inp.raw is not None else None
    if export_cfg is not None:
        _add_export_f_datasets(
            base, iteration, op=op,
            state_vectors=[state_vectors[k] for k in range(n_rhs)], cfg=export_cfg,
        )  # fmt: skip

    text = inp.raw.source_text if (inp.raw is not None and inp.raw.source_text is not None) else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_output_by_suffix(path, base, iteration, text)
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
    converged: bool | None = None,
    solver_method: str | None = None,
    solver_requested_method: str | None = None,
    residual_norm: float | None = None,
) -> Path:
    """Write the RHSMode=1 ``sfincsOutput`` file from canonical-stack objects.

    Same non-iteration machinery as :func:`write_transport_output`; the
    iteration datasets carry the full RHSMode=1 per-species moment table
    (:func:`sfincs_jax.moments.rhsmode1_moments`), NTV, and the classical
    fluxes at the run's gradients, all with a single trailing iteration axis
    (``NIterations=1``).  When the deck requests ``export_f``, the ``full_f`` /
    ``delta_f`` distribution-function datasets are written on the user grids.

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
        converged: Phi1-run Newton convergence flag; when provided (Phi1 runs)
            emits ``didNonlinearCalculationConverge`` (writeHDF5Output.F90).
        solver_method / solver_requested_method: linear-solver method names
            emitted as ``linearSolverMethod`` / ``linearSolverRequestedMethod``.
        residual_norm: converged residual norm emitted as
            ``linearSolverResidualNorm``.

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
    if bool(op.include_phi1):
        _add_phi1_solver_metadata(
            base, converged=converged, solver_method=solver_method,
            solver_requested_method=solver_requested_method, residual_norm=residual_norm,
        )  # fmt: skip
    iteration = _rhsmode1_iteration_fields(
        inp=inp, op=op, geom=geom, radial=radial, gpsipsi=gpsipsi,
        u_hat=base["uHat"], g_eff=float(base["GHat"]), i_eff=float(base["IHat"]),
        nu_n_eff=float(base["nu_n"]), state_vector=state_vector,
        elapsed_seconds=elapsed_seconds,
    )  # fmt: skip

    export_cfg = _export_f_config(raw=inp.raw, grids=grids, geom=geom) if inp.raw is not None else None
    if export_cfg is not None:
        _add_export_f_datasets(
            base, iteration, op=op, state_vectors=[state_vector], cfg=export_cfg
        )

    text = inp.raw.source_text if (inp.raw is not None and inp.raw.source_text is not None) else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_output_by_suffix(path, base, iteration, text)
    return path.resolve()


def _active_size(op: KineticOperator) -> int:
    """Packed active-DOF count of the operator (``Nxi_for_x`` truncation applied)."""
    mask = op.active_dof_mask()
    if mask is None:
        return int(op.total_size)
    return int(np.asarray(mask, dtype=np.float64).sum())


def write_run_solver_trace(
    *,
    path: str | Path,
    inp: SfincsInput,
    op: KineticOperator,
    solve_result: Any,
    rhs_norm: float,
    solver_tol: float,
    selected_path: str,
    elapsed_seconds: float,
    input_namelist: str | Path | None = None,
    output_path: str | Path | None = None,
    compute_solution: bool,
    compute_transport_matrix: bool,
) -> Path:
    """Write a versioned solver-trace JSON sidecar from a canonical run.

    Emits the same :class:`sfincs_jax.solvers.diagnostics.SolverTrace` schema
    the legacy ``--solver-trace`` path produces, populated from the canonical
    :class:`sfincs_jax.solve.SolveResult` (method, residual norms, convergence,
    per-phase timings).  Solver-implementation fields (``solve_method``,
    per-phase timings, memory estimates) naturally differ from the retired
    GMRES pipeline; the parity-relevant fields (backend, ``rhs_mode``,
    ``selected_path``, ``geometry_scheme``, sizes, residual norm vs target,
    convergence) match.  Imported lazily so the canonical writer keeps no
    module-load dependency on the retained ``solvers`` package.
    """
    from sfincs_jax.solvers.diagnostics import SolverTrace, write_solver_trace_json  # noqa: PLC0415

    try:
        import jax  # noqa: PLC0415

        backend = str(jax.default_backend())
        device_count = len(jax.devices())
    except Exception:  # noqa: BLE001
        backend = "unknown"
        device_count = None

    residual_norms = np.atleast_1d(np.asarray(solve_result.residual_norms, dtype=np.float64))
    residual_norm = float(np.max(residual_norms)) if residual_norms.size else None
    residual_target = max(0.0, float(solver_tol) * float(rhs_norm))
    converged = bool(solve_result.converged)
    timings = dict(getattr(solve_result, "timings", {}) or {})
    iterations = getattr(solve_result, "iterations", None)

    trace = SolverTrace(
        backend=backend,
        rhs_mode=int(inp.general.rhs_mode),
        selected_path=selected_path,
        solve_method=str(getattr(solve_result, "method", "auto")),
        geometry_scheme=int(inp.geometry.geometry_scheme),
        collision_operator=str(int(inp.physics.collision_operator)),
        total_size=int(op.total_size),
        active_size=_active_size(op),
        device_count=device_count,
        residual_norm=residual_norm,
        residual_target=residual_target,
        converged=converged,
        elapsed_s=float(elapsed_seconds),
        setup_s=(float(timings["build"]) if "build" in timings else None),
        solve_s=(float(timings["solve"]) if "solve" in timings else None),
        matvec_count=(int(iterations) if iterations is not None else None),
        metadata={
            "input_namelist": str(Path(input_namelist).resolve()) if input_namelist else "",
            "output_path": str(Path(output_path).resolve()) if output_path is not None else "",
            "output_format": (Path(output_path).suffix.lstrip(".") if output_path is not None else ""),
            "compute_solution": bool(compute_solution),
            "compute_transport_matrix": bool(compute_transport_matrix),
            "differentiable": False,
            "converged": converged,
        },
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_solver_trace_json(path, trace)
    return path.resolve()
