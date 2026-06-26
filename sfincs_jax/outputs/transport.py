"""RHSMode=2/3 transport output-schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import os

import h5py
import jax.numpy as jnp
import numpy as np

from sfincs_jax.diagnostics import u_hat_np
from sfincs_jax.operators.profile_system import with_transport_rhs_settings
from sfincs_jax.problems.transport_matrix.diagnostics import (
    _flux_functions_from_op,
    transport_matrix_size_from_rhs_mode,
    v3_rhsmode1_output_fields_vm_only_jit,
    v3_transport_diagnostics_vm_only,
)

from .formats import fortran_h5_layout, to_numpy_for_h5


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _require_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"transport streaming output field {name!r} was not allocated")
    return value


@dataclass
class TransportStreamingOutputAccumulator:
    """Mutable streaming buffers for RHSMode=2/3 transport diagnostics.

    Arrays are stored in the Python-read axis order used by Fortran v3 HDF5
    outputs. The object is intentionally host-side state for non-differentiated
    diagnostics, not part of the JAX solve pytree.
    """

    op0: Any
    collect_full_output_fields: bool
    particle_flux_vm_psi_hat: np.ndarray
    heat_flux_vm_psi_hat: np.ndarray
    fsab_flow_diag: np.ndarray
    density_perturbation: np.ndarray | None = None
    pressure_perturbation: np.ndarray | None = None
    pressure_anisotropy: np.ndarray | None = None
    flow: np.ndarray | None = None
    total_density: np.ndarray | None = None
    total_pressure: np.ndarray | None = None
    velocity_using_fsa_density: np.ndarray | None = None
    velocity_using_total_density: np.ndarray | None = None
    mach_using_fsa_thermal_speed: np.ndarray | None = None
    j_hat: np.ndarray | None = None
    fsa_density_perturbation: np.ndarray | None = None
    fsa_pressure_perturbation: np.ndarray | None = None
    momentum_flux_before_vm: np.ndarray | None = None
    momentum_flux_before_vm0: np.ndarray | None = None
    momentum_flux_before_ve: np.ndarray | None = None
    momentum_flux_before_ve0: np.ndarray | None = None
    momentum_flux_vm_psi_hat: np.ndarray | None = None
    momentum_flux_vm0_psi_hat: np.ndarray | None = None
    ntv_before: np.ndarray | None = None
    ntv: np.ndarray | None = None
    particle_flux_before_vm: np.ndarray | None = None
    heat_flux_before_vm: np.ndarray | None = None
    particle_flux_before_vm0: np.ndarray | None = None
    heat_flux_before_vm0: np.ndarray | None = None
    particle_flux_before_ve: np.ndarray | None = None
    heat_flux_before_ve: np.ndarray | None = None
    particle_flux_before_ve0: np.ndarray | None = None
    heat_flux_before_ve0: np.ndarray | None = None
    particle_flux_vm0_psi_hat: np.ndarray | None = None
    heat_flux_vm0_psi_hat: np.ndarray | None = None
    particle_flux_vm_psi_hat_vs_x: np.ndarray | None = None
    heat_flux_vm_psi_hat_vs_x: np.ndarray | None = None
    fsab_flow: np.ndarray | None = None
    fsab_flow_vs_x: np.ndarray | None = None
    sources: np.ndarray | None = None
    compute_ntv: bool = False
    ntv_kernel: jnp.ndarray | None = None
    w2d_np: np.ndarray | None = None
    w_ntv: jnp.ndarray | None = None
    t_hat: jnp.ndarray | None = None
    sqrt_t: jnp.ndarray | None = None
    m_hat: jnp.ndarray | None = None
    sqrt_m: jnp.ndarray | None = None
    vprime_hat: jnp.ndarray | None = None
    b0_val: float | None = None
    fsab2_val: float | None = None
    n_hat_np: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        *,
        nml: Any,
        grids: Any,
        geom: Any,
        op0: Any,
        n_rhs: int,
        collect_full_output_fields: bool,
    ) -> "TransportStreamingOutputAccumulator":
        """Allocate streaming diagnostic buffers for a transport solve."""
        s = int(op0.n_species)
        t = int(op0.n_theta)
        z = int(op0.n_zeta)
        x = int(op0.n_x)
        n = int(n_rhs)
        accumulator = cls(
            op0=op0,
            collect_full_output_fields=bool(collect_full_output_fields),
            particle_flux_vm_psi_hat=_zeros((s, n)),
            heat_flux_vm_psi_hat=_zeros((s, n)),
            fsab_flow_diag=_zeros((s, n)),
        )
        if not collect_full_output_fields:
            return accumulator

        accumulator.density_perturbation = _zeros((z, t, s, n))
        accumulator.pressure_perturbation = _zeros((z, t, s, n))
        accumulator.pressure_anisotropy = _zeros((z, t, s, n))
        accumulator.flow = _zeros((z, t, s, n))
        accumulator.total_density = _zeros((z, t, s, n))
        accumulator.total_pressure = _zeros((z, t, s, n))
        accumulator.velocity_using_fsa_density = _zeros((z, t, s, n))
        accumulator.velocity_using_total_density = _zeros((z, t, s, n))
        accumulator.mach_using_fsa_thermal_speed = _zeros((z, t, s, n))
        accumulator.j_hat = _zeros((z, t, n))
        accumulator.fsa_density_perturbation = _zeros((s, n))
        accumulator.fsa_pressure_perturbation = _zeros((s, n))
        accumulator.momentum_flux_before_vm = _zeros((z, t, s, n))
        accumulator.momentum_flux_before_vm0 = _zeros((z, t, s, n))
        accumulator.momentum_flux_before_ve = _zeros((z, t, s, n))
        accumulator.momentum_flux_before_ve0 = _zeros((z, t, s, n))
        accumulator.momentum_flux_vm_psi_hat = _zeros((s, n))
        accumulator.momentum_flux_vm0_psi_hat = _zeros((s, n))
        accumulator.ntv_before = _zeros((z, t, s, n))
        accumulator.ntv = _zeros((s, n))
        accumulator.particle_flux_before_vm = _zeros((z, t, s, n))
        accumulator.heat_flux_before_vm = _zeros((z, t, s, n))
        accumulator.particle_flux_before_vm0 = _zeros((z, t, s, n))
        accumulator.heat_flux_before_vm0 = _zeros((z, t, s, n))
        accumulator.particle_flux_before_ve = _zeros((z, t, s, n))
        accumulator.heat_flux_before_ve = _zeros((z, t, s, n))
        accumulator.particle_flux_before_ve0 = _zeros((z, t, s, n))
        accumulator.heat_flux_before_ve0 = _zeros((z, t, s, n))
        accumulator.particle_flux_vm0_psi_hat = _zeros((s, n))
        accumulator.heat_flux_vm0_psi_hat = _zeros((s, n))
        accumulator.particle_flux_vm_psi_hat_vs_x = _zeros((x, s, n))
        accumulator.heat_flux_vm_psi_hat_vs_x = _zeros((x, s, n))
        accumulator.fsab_flow = _zeros((s, n))
        accumulator.fsab_flow_vs_x = _zeros((x, s, n))

        if int(op0.constraint_scheme) == 2:
            accumulator.sources = _zeros((x, s, n))
        elif int(op0.constraint_scheme) in {1, 3, 4}:
            accumulator.sources = _zeros((2, s, n))

        geom_params = nml.group("geometryParameters")
        geometry_scheme = int(geom_params.get("GEOMETRYSCHEME", geom_params.get("geometryScheme", -1)))
        accumulator.compute_ntv = geometry_scheme != 5
        if accumulator.compute_ntv:
            uhat_np = u_hat_np(grids=grids, geom=geom)
            uhat = jnp.asarray(uhat_np, dtype=jnp.float64)
            bh = jnp.asarray(op0.b_hat, dtype=jnp.float64)
            dbt = jnp.asarray(op0.db_hat_dtheta, dtype=jnp.float64)
            dbz = jnp.asarray(op0.db_hat_dzeta, dtype=jnp.float64)
            inv_fsa_b2 = 1.0 / jnp.asarray(op0.fsab_hat2, dtype=jnp.float64)
            ghat = jnp.asarray(float(geom.g_hat), dtype=jnp.float64)
            ihat = jnp.asarray(float(geom.i_hat), dtype=jnp.float64)
            iota = jnp.asarray(float(geom.iota), dtype=jnp.float64)
            accumulator.ntv_kernel = (2.0 / 5.0) / bh * (
                (uhat - ghat * inv_fsa_b2) * (iota * dbt + dbz)
                + iota * (1.0 / (bh * bh)) * (ghat * dbt - ihat * dbz)
            )
        else:
            accumulator.ntv_kernel = jnp.zeros_like(jnp.asarray(op0.b_hat, dtype=jnp.float64))

        w2d = jnp.asarray(op0.theta_weights, dtype=jnp.float64)[:, None] * jnp.asarray(
            op0.zeta_weights, dtype=jnp.float64
        )[None, :]
        accumulator.vprime_hat = jnp.sum(w2d / jnp.asarray(op0.d_hat, dtype=jnp.float64))
        x_grid = jnp.asarray(op0.x, dtype=jnp.float64)
        xw = jnp.asarray(op0.x_weights, dtype=jnp.float64)
        accumulator.w_ntv = xw * (x_grid**4)
        accumulator.t_hat = jnp.asarray(op0.t_hat, dtype=jnp.float64)
        accumulator.m_hat = jnp.asarray(op0.m_hat, dtype=jnp.float64)
        accumulator.sqrt_t = jnp.sqrt(accumulator.t_hat)
        accumulator.sqrt_m = jnp.sqrt(accumulator.m_hat)
        b0, _g, _i = _flux_functions_from_op(op0)
        fsab2 = jnp.asarray(op0.fsab_hat2, dtype=jnp.float64)
        accumulator.w2d_np = np.asarray(w2d, dtype=np.float64)
        accumulator.b0_val = float(np.asarray(b0, dtype=np.float64))
        accumulator.fsab2_val = float(np.asarray(fsab2, dtype=np.float64))
        accumulator.n_hat_np = np.asarray(op0.n_hat, dtype=np.float64)
        return accumulator

    def collect(self, which_rhs: int, x_full: jnp.ndarray) -> None:
        """Populate streaming diagnostics for one completed ``whichRHS`` solve."""
        op0 = self.op0
        j = int(which_rhs) - 1
        op_rhs = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))
        diag = v3_transport_diagnostics_vm_only(op_rhs, x_full=x_full)
        self.particle_flux_vm_psi_hat[:, j] = np.asarray(diag.particle_flux_vm_psi_hat, dtype=np.float64)
        self.heat_flux_vm_psi_hat[:, j] = np.asarray(diag.heat_flux_vm_psi_hat, dtype=np.float64)
        self.fsab_flow_diag[:, j] = np.asarray(diag.fsab_flow, dtype=np.float64)
        if not self.collect_full_output_fields:
            return

        d = v3_rhsmode1_output_fields_vm_only_jit(op_rhs, x_full=x_full)
        _require_array(self.density_perturbation, "densityPerturbation")[:, :, :, j] = np.asarray(
            jnp.transpose(d["densityPerturbation"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.pressure_perturbation, "pressurePerturbation")[:, :, :, j] = np.asarray(
            jnp.transpose(d["pressurePerturbation"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.pressure_anisotropy, "pressureAnisotropy")[:, :, :, j] = np.asarray(
            jnp.transpose(d["pressureAnisotropy"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.flow, "flow")[:, :, :, j] = np.asarray(
            jnp.transpose(d["flow"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.total_density, "totalDensity")[:, :, :, j] = np.asarray(
            jnp.transpose(d["totalDensity"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.total_pressure, "totalPressure")[:, :, :, j] = np.asarray(
            jnp.transpose(d["totalPressure"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.velocity_using_fsa_density, "velocityUsingFSADensity")[:, :, :, j] = np.asarray(
            jnp.transpose(d["velocityUsingFSADensity"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.velocity_using_total_density, "velocityUsingTotalDensity")[:, :, :, j] = np.asarray(
            jnp.transpose(d["velocityUsingTotalDensity"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.mach_using_fsa_thermal_speed, "MachUsingFSAThermalSpeed")[:, :, :, j] = np.asarray(
            jnp.transpose(d["MachUsingFSAThermalSpeed"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.j_hat, "jHat")[:, :, j] = np.asarray(jnp.transpose(d["jHat"], (1, 0)), dtype=np.float64)
        _require_array(self.fsa_density_perturbation, "FSADensityPerturbation")[:, j] = np.asarray(
            d["FSADensityPerturbation"], dtype=np.float64
        )
        _require_array(self.fsa_pressure_perturbation, "FSAPressurePerturbation")[:, j] = np.asarray(
            d["FSAPressurePerturbation"], dtype=np.float64
        )

        _require_array(self.momentum_flux_before_vm, "momentumFluxBeforeSurfaceIntegral_vm")[:, :, :, j] = np.asarray(
            jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vm"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.momentum_flux_before_vm0, "momentumFluxBeforeSurfaceIntegral_vm0")[:, :, :, j] = np.asarray(
            jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vm0"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.momentum_flux_before_ve, "momentumFluxBeforeSurfaceIntegral_vE")[:, :, :, j] = np.asarray(
            jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vE"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.momentum_flux_before_ve0, "momentumFluxBeforeSurfaceIntegral_vE0")[:, :, :, j] = np.asarray(
            jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vE0"], (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.momentum_flux_vm_psi_hat, "momentumFlux_vm_psiHat")[:, j] = np.asarray(
            d["momentumFlux_vm_psiHat"], dtype=np.float64
        )
        _require_array(self.momentum_flux_vm0_psi_hat, "momentumFlux_vm0_psiHat")[:, j] = np.asarray(
            d["momentumFlux_vm0_psiHat"], dtype=np.float64
        )

        _require_array(self.particle_flux_before_vm, "particleFluxBeforeSurfaceIntegral_vm")[:, :, :, j] = np.asarray(
            jnp.transpose(diag.particle_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.heat_flux_before_vm, "heatFluxBeforeSurfaceIntegral_vm")[:, :, :, j] = np.asarray(
            jnp.transpose(diag.heat_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.particle_flux_before_vm0, "particleFluxBeforeSurfaceIntegral_vm0")[:, :, :, j] = np.asarray(
            jnp.transpose(diag.particle_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.heat_flux_before_vm0, "heatFluxBeforeSurfaceIntegral_vm0")[:, :, :, j] = np.asarray(
            jnp.transpose(diag.heat_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64
        )
        _require_array(self.particle_flux_vm_psi_hat_vs_x, "particleFlux_vm_psiHat_vs_x")[:, :, j] = np.asarray(
            diag.particle_flux_vm_psi_hat_vs_x, dtype=np.float64
        )
        _require_array(self.heat_flux_vm_psi_hat_vs_x, "heatFlux_vm_psiHat_vs_x")[:, :, j] = np.asarray(
            diag.heat_flux_vm_psi_hat_vs_x, dtype=np.float64
        )
        _require_array(self.fsab_flow_vs_x, "FSABFlow_vs_x")[:, :, j] = np.asarray(
            diag.fsab_flow_vs_x, dtype=np.float64
        )
        _require_array(self.fsab_flow, "FSABFlow")[:, j] = np.asarray(diag.fsab_flow, dtype=np.float64)

        w2d_np = _require_array(self.w2d_np, "w2d_np")
        _require_array(self.particle_flux_vm0_psi_hat, "particleFlux_vm0_psiHat")[:, j] = np.einsum(
            "tz,stz->s",
            w2d_np,
            np.asarray(diag.particle_flux_before_surface_integral_vm0, dtype=np.float64),
        )
        _require_array(self.heat_flux_vm0_psi_hat, "heatFlux_vm0_psiHat")[:, j] = np.einsum(
            "tz,stz->s",
            w2d_np,
            np.asarray(diag.heat_flux_before_surface_integral_vm0, dtype=np.float64),
        )

        if self.compute_ntv and int(op0.n_xi) > 2:
            f_delta = np.asarray(x_full[: op0.f_size], dtype=np.float64).reshape(op0.fblock.f_shape)
            sum_ntv = np.einsum(
                "x,sxtz->stz",
                np.asarray(self.w_ntv, dtype=np.float64),
                f_delta[:, :, 2, :, :],
            )
            t_hat = np.asarray(self.t_hat, dtype=np.float64)
            sqrt_t = np.asarray(self.sqrt_t, dtype=np.float64)
            m_hat = np.asarray(self.m_hat, dtype=np.float64)
            sqrt_m = np.asarray(self.sqrt_m, dtype=np.float64)
            vprime_hat = float(np.asarray(self.vprime_hat, dtype=np.float64))
            ntv_prefactor = 4.0 * np.pi * t_hat**2 * sqrt_t / (m_hat * sqrt_m * vprime_hat)
            ntv_before_stz = (
                ntv_prefactor[:, None, None]
                * np.asarray(self.ntv_kernel, dtype=np.float64)[None, :, :]
                * sum_ntv
            )
            ntv_s = np.einsum("tz,stz->s", w2d_np, ntv_before_stz)
        else:
            ntv_before_stz = np.zeros((int(op0.n_species), int(op0.n_theta), int(op0.n_zeta)), dtype=np.float64)
            ntv_s = np.zeros((int(op0.n_species),), dtype=np.float64)
        _require_array(self.ntv, "NTV")[:, j] = ntv_s
        _require_array(self.ntv_before, "NTVBeforeSurfaceIntegral")[:, :, :, j] = np.asarray(
            np.transpose(ntv_before_stz, (2, 1, 0)), dtype=np.float64
        )

        if self.sources is not None:
            extra = np.asarray(x_full[op0.f_size + op0.phi1_size :], dtype=np.float64)
            if int(op0.constraint_scheme) == 2:
                src = extra.reshape((int(op0.n_species), int(op0.n_x))).T
            else:
                src = extra.reshape((int(op0.n_species), 2)).T
            self.sources[:, :, j] = src

    def diagnostic_flux_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return particle flux, heat flux, and FSABFlow diagnostics as JAX arrays."""
        return (
            jnp.asarray(self.particle_flux_vm_psi_hat, dtype=jnp.float64),
            jnp.asarray(self.heat_flux_vm_psi_hat, dtype=jnp.float64),
            jnp.asarray(self.fsab_flow_diag, dtype=jnp.float64),
        )

    def output_fields(self) -> dict[str, np.ndarray] | None:
        """Return full streamed output fields, or ``None`` when not requested."""
        if not self.collect_full_output_fields:
            return None
        fsab_flow = _require_array(self.fsab_flow, "FSABFlow")
        n_hat_np = _require_array(self.n_hat_np, "n_hat_np")
        if self.b0_val is None or self.fsab2_val is None:
            raise RuntimeError("transport streaming output normalization was not initialized")
        z_s_np = np.asarray(self.op0.z_s, dtype=np.float64)
        fsab_jhat = np.einsum("s,sn->n", z_s_np, fsab_flow)
        fields: dict[str, np.ndarray] = {
            "densityPerturbation": _require_array(self.density_perturbation, "densityPerturbation"),
            "pressurePerturbation": _require_array(self.pressure_perturbation, "pressurePerturbation"),
            "pressureAnisotropy": _require_array(self.pressure_anisotropy, "pressureAnisotropy"),
            "flow": _require_array(self.flow, "flow"),
            "totalDensity": _require_array(self.total_density, "totalDensity"),
            "totalPressure": _require_array(self.total_pressure, "totalPressure"),
            "velocityUsingFSADensity": _require_array(self.velocity_using_fsa_density, "velocityUsingFSADensity"),
            "velocityUsingTotalDensity": _require_array(
                self.velocity_using_total_density, "velocityUsingTotalDensity"
            ),
            "MachUsingFSAThermalSpeed": _require_array(
                self.mach_using_fsa_thermal_speed, "MachUsingFSAThermalSpeed"
            ),
            "jHat": _require_array(self.j_hat, "jHat"),
            "FSADensityPerturbation": _require_array(self.fsa_density_perturbation, "FSADensityPerturbation"),
            "FSAPressurePerturbation": _require_array(self.fsa_pressure_perturbation, "FSAPressurePerturbation"),
            "momentumFluxBeforeSurfaceIntegral_vm": _require_array(
                self.momentum_flux_before_vm, "momentumFluxBeforeSurfaceIntegral_vm"
            ),
            "momentumFluxBeforeSurfaceIntegral_vm0": _require_array(
                self.momentum_flux_before_vm0, "momentumFluxBeforeSurfaceIntegral_vm0"
            ),
            "momentumFluxBeforeSurfaceIntegral_vE": _require_array(
                self.momentum_flux_before_ve, "momentumFluxBeforeSurfaceIntegral_vE"
            ),
            "momentumFluxBeforeSurfaceIntegral_vE0": _require_array(
                self.momentum_flux_before_ve0, "momentumFluxBeforeSurfaceIntegral_vE0"
            ),
            "momentumFlux_vm_psiHat": _require_array(self.momentum_flux_vm_psi_hat, "momentumFlux_vm_psiHat"),
            "momentumFlux_vm0_psiHat": _require_array(self.momentum_flux_vm0_psi_hat, "momentumFlux_vm0_psiHat"),
            "NTVBeforeSurfaceIntegral": _require_array(self.ntv_before, "NTVBeforeSurfaceIntegral"),
            "NTV": _require_array(self.ntv, "NTV"),
            "FSABFlow": fsab_flow,
            "FSABFlow_vs_x": _require_array(self.fsab_flow_vs_x, "FSABFlow_vs_x"),
            "FSABVelocityUsingFSADensity": fsab_flow / n_hat_np[:, None],
            "FSABVelocityUsingFSADensityOverB0": (fsab_flow / n_hat_np[:, None]) / float(self.b0_val),
            "FSABVelocityUsingFSADensityOverRootFSAB2": (fsab_flow / n_hat_np[:, None])
            / np.sqrt(float(self.fsab2_val)),
            "FSABjHat": fsab_jhat,
            "FSABjHatOverB0": fsab_jhat / float(self.b0_val),
            "FSABjHatOverRootFSAB2": fsab_jhat / np.sqrt(float(self.fsab2_val)),
            "particleFlux_vm_psiHat": self.particle_flux_vm_psi_hat,
            "heatFlux_vm_psiHat": self.heat_flux_vm_psi_hat,
            "particleFlux_vm0_psiHat": _require_array(self.particle_flux_vm0_psi_hat, "particleFlux_vm0_psiHat"),
            "heatFlux_vm0_psiHat": _require_array(self.heat_flux_vm0_psi_hat, "heatFlux_vm0_psiHat"),
            "particleFluxBeforeSurfaceIntegral_vm": _require_array(
                self.particle_flux_before_vm, "particleFluxBeforeSurfaceIntegral_vm"
            ),
            "heatFluxBeforeSurfaceIntegral_vm": _require_array(
                self.heat_flux_before_vm, "heatFluxBeforeSurfaceIntegral_vm"
            ),
            "particleFluxBeforeSurfaceIntegral_vm0": _require_array(
                self.particle_flux_before_vm0, "particleFluxBeforeSurfaceIntegral_vm0"
            ),
            "heatFluxBeforeSurfaceIntegral_vm0": _require_array(
                self.heat_flux_before_vm0, "heatFluxBeforeSurfaceIntegral_vm0"
            ),
            "particleFluxBeforeSurfaceIntegral_vE": _require_array(
                self.particle_flux_before_ve, "particleFluxBeforeSurfaceIntegral_vE"
            ),
            "heatFluxBeforeSurfaceIntegral_vE": _require_array(
                self.heat_flux_before_ve, "heatFluxBeforeSurfaceIntegral_vE"
            ),
            "particleFluxBeforeSurfaceIntegral_vE0": _require_array(
                self.particle_flux_before_ve0, "particleFluxBeforeSurfaceIntegral_vE0"
            ),
            "heatFluxBeforeSurfaceIntegral_vE0": _require_array(
                self.heat_flux_before_ve0, "heatFluxBeforeSurfaceIntegral_vE0"
            ),
            "particleFlux_vm_psiHat_vs_x": _require_array(
                self.particle_flux_vm_psi_hat_vs_x, "particleFlux_vm_psiHat_vs_x"
            ),
            "heatFlux_vm_psiHat_vs_x": _require_array(
                self.heat_flux_vm_psi_hat_vs_x, "heatFlux_vm_psiHat_vs_x"
            ),
        }
        if self.sources is not None:
            fields["sources"] = self.sources
        return fields


def transport_solver_diagnostic_arrays(
    result: Any,
    n_rhs: int,
) -> dict[str, np.ndarray]:
    """Return absolute and relative transport residual diagnostics.

    The writer stores one residual, one RHS norm, and one relative residual per
    ``whichRHS`` solve, plus max summaries. Missing RHS entries are represented
    by ``NaN`` so partial debug artifacts remain explicit instead of silently
    looking converged.
    """

    residuals_by_rhs = getattr(result, "residual_norms_by_rhs", None) or {}
    rhs_norms_by_rhs = getattr(result, "rhs_norms_by_rhs", None) or {}
    residuals = np.asarray(
        [
            float(np.asarray(residuals_by_rhs.get(i, np.nan), dtype=np.float64))
            for i in range(1, int(n_rhs) + 1)
        ],
        dtype=np.float64,
    )
    rhs_norms = np.asarray(
        [
            float(np.asarray(rhs_norms_by_rhs.get(i, np.nan), dtype=np.float64))
            for i in range(1, int(n_rhs) + 1)
        ],
        dtype=np.float64,
    )
    rel = np.full_like(residuals, np.nan, dtype=np.float64)
    valid = np.isfinite(residuals) & np.isfinite(rhs_norms) & (rhs_norms > 0.0)
    rel[valid] = residuals[valid] / rhs_norms[valid]
    finite_residuals = residuals[np.isfinite(residuals)]
    finite_rel = rel[np.isfinite(rel)]
    return {
        "transportResidualNorms": residuals,
        "transportRhsNorms": rhs_norms,
        "transportRelativeResidualNorms": rel,
        "transportMaxResidualNorm": np.asarray(
            float(np.max(finite_residuals)) if finite_residuals.size else float("nan"),
            dtype=np.float64,
        ),
        "transportMaxRelativeResidualNorm": np.asarray(
            float(np.max(finite_rel)) if finite_rel.size else float("nan"),
            dtype=np.float64,
        ),
    }

def conversion_factors_to_from_dpsi_hat(*, psi_a_hat: float, a_hat: float, r_n: float) -> dict[str, float]:
    """Replicate v3 `radialCoordinates.setInputRadialCoordinate` derivative conversion factors."""
    psi_n = float(r_n) * float(r_n)
    root = float(np.sqrt(psi_n))
    ddpsi_n_to_ddpsi_hat = 1.0 / float(psi_a_hat)
    ddr_hat_to_ddpsi_hat = float(a_hat) / (2.0 * float(psi_a_hat) * root)
    ddr_n_to_ddpsi_hat = 1.0 / (2.0 * float(psi_a_hat) * root)

    ddpsi_hat_to_ddpsi_n = float(psi_a_hat)
    ddpsi_hat_to_ddr_hat = (2.0 * float(psi_a_hat) * root) / float(a_hat)
    ddpsi_hat_to_ddr_n = (2.0 * float(psi_a_hat) * root)

    return {
        "ddpsiN2ddpsiHat": ddpsi_n_to_ddpsi_hat,
        "ddrHat2ddpsiHat": ddr_hat_to_ddpsi_hat,
        "ddrN2ddpsiHat": ddr_n_to_ddpsi_hat,
        "ddpsiHat2ddpsiN": ddpsi_hat_to_ddpsi_n,
        "ddpsiHat2ddrHat": ddpsi_hat_to_ddr_hat,
        "ddpsiHat2ddrN": ddpsi_hat_to_ddr_n,
    }


def write_transport_h5_streaming(
    *,
    output_path: Path,
    data: dict[str, Any],
    input_namelist: Path,
    result: Any,
    nml: Any,
    fortran_layout: bool,
    overwrite: bool,
    emit: "Callable[[int, str], None] | None" = None,
) -> Path:
    """Stream RHSMode=2/3 transport diagnostics directly to H5 to reduce memory."""
    from ..physics.classical_transport import classical_flux_v3  # noqa: PLC0415

    op0 = result.op0
    n_rhs = transport_matrix_size_from_rhs_mode(int(op0.rhs_mode))
    z = int(op0.n_zeta)
    t = int(op0.n_theta)
    s = int(op0.n_species)
    x = int(op0.n_x)
    write_solver_diagnostics = os.environ.get("SFINCS_JAX_WRITE_SOLVER_DIAGNOSTICS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if len(result.state_vectors_by_rhs) < n_rhs:
        raise ValueError("Streaming transport H5 requires state vectors for every whichRHS.")

    # Transport-output field sets.
    ztsn_fields = (
        "densityPerturbation",
        "pressurePerturbation",
        "pressureAnisotropy",
        "flow",
        "totalDensity",
        "totalPressure",
        "velocityUsingFSADensity",
        "velocityUsingTotalDensity",
        "MachUsingFSAThermalSpeed",
        "momentumFluxBeforeSurfaceIntegral_vm",
        "momentumFluxBeforeSurfaceIntegral_vm0",
        "momentumFluxBeforeSurfaceIntegral_vE",
        "momentumFluxBeforeSurfaceIntegral_vE0",
        "particleFluxBeforeSurfaceIntegral_vm",
        "heatFluxBeforeSurfaceIntegral_vm",
        "particleFluxBeforeSurfaceIntegral_vm0",
        "heatFluxBeforeSurfaceIntegral_vm0",
        "particleFluxBeforeSurfaceIntegral_vE",
        "heatFluxBeforeSurfaceIntegral_vE",
        "particleFluxBeforeSurfaceIntegral_vE0",
        "heatFluxBeforeSurfaceIntegral_vE0",
        "NTVBeforeSurfaceIntegral",
    )
    ztn_fields = ("jHat",)
    xsn_fields = (
        "particleFlux_vm_psiHat_vs_x",
        "heatFlux_vm_psiHat_vs_x",
        "FSABFlow_vs_x",
    )

    flux_bases = (
        "particleFlux_vm_psiHat",
        "heatFlux_vm_psiHat",
        "momentumFlux_vm_psiHat",
        "particleFlux_vm0_psiHat",
        "heatFlux_vm0_psiHat",
        "momentumFlux_vm0_psiHat",
        "classicalParticleFlux_psiHat",
        "classicalHeatFlux_psiHat",
    )
    flux_variants: list[str] = []
    for base in flux_bases:
        flux_variants.append(base)
        flux_variants.append(base.replace("_psiHat", "_psiN"))
        flux_variants.append(base.replace("_psiHat", "_rHat"))
        flux_variants.append(base.replace("_psiHat", "_rN"))

    sn_fields = (
        "FSADensityPerturbation",
        "FSAPressurePerturbation",
        "NTV",
        "FSABFlow",
        "FSABVelocityUsingFSADensity",
        "FSABVelocityUsingFSADensityOverB0",
        "FSABVelocityUsingFSADensityOverRootFSAB2",
        *flux_variants,
    )
    n_fields = (
        "FSABjHat",
        "FSABjHatOverB0",
        "FSABjHatOverRootFSAB2",
    )

    constraint_scheme = int(np.asarray(data.get("constraintScheme", 0)).reshape(-1)[0])
    sources_shape: tuple[int, int] | None = None
    if constraint_scheme == 2:
        sources_shape = (x, s)
    elif constraint_scheme in {1, 3, 4}:
        sources_shape = (2, s)

    transport_keys = set(ztsn_fields) | set(ztn_fields) | set(xsn_fields) | set(sn_fields) | set(n_fields)
    transport_keys |= {"transportMatrix", "NIterations", "input.namelist", "elapsed time (s)"}
    if write_solver_diagnostics:
        transport_keys |= {
            "transportResidualNorms",
            "transportRhsNorms",
            "transportRelativeResidualNorms",
            "transportMaxResidualNorm",
            "transportMaxRelativeResidualNorm",
        }
    if sources_shape is not None:
        transport_keys.add("sources")

    # Prepare base data for streaming write.
    base_data: dict[str, Any] = {k: v for k, v in data.items() if k not in transport_keys}
    base_data["NIterations"] = np.asarray(n_rhs, dtype=np.int32)
    base_data["input.namelist"] = nml.source_text if nml.source_text is not None else input_namelist.read_text()

    if output_path.exists() and not overwrite:
        raise FileExistsError(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if emit is not None:
        emit(0, " Saving diagnostics to h5 file for iteration            1")

    def _shape_fortran(shape: tuple[int, ...]) -> tuple[int, ...]:
        # Transport fields are stored in "Python-read" order when fortran_layout=True,
        # mirroring the pre-transpose + write transpose in the non-streaming path.
        return tuple(reversed(shape)) if not fortran_layout else shape

    def _write_slice(dset: h5py.Dataset, j: int, arr: np.ndarray) -> None:
        arr_w = fortran_h5_layout(arr) if not fortran_layout else arr
        if fortran_layout:
            dset[..., j] = np.asarray(arr_w, dtype=np.float64)
        else:
            dset[j, ...] = np.asarray(arr_w, dtype=np.float64)

    with h5py.File(output_path, "w") as f:
        for k, v in base_data.items():
            if v is None:
                continue
            vv = to_numpy_for_h5(v)
            if fortran_layout:
                vv = fortran_h5_layout(vv)
            f.create_dataset(k, data=vv)

        dsets: dict[str, h5py.Dataset] = {}
        for name in ztsn_fields:
            dsets[name] = f.create_dataset(name, _shape_fortran((z, t, s, n_rhs)), dtype=np.float64)
        for name in ztn_fields:
            dsets[name] = f.create_dataset(name, _shape_fortran((z, t, n_rhs)), dtype=np.float64)
        for name in xsn_fields:
            dsets[name] = f.create_dataset(name, _shape_fortran((x, s, n_rhs)), dtype=np.float64)
        for name in sn_fields:
            dsets[name] = f.create_dataset(name, _shape_fortran((s, n_rhs)), dtype=np.float64)
        for name in n_fields:
            dsets[name] = f.create_dataset(name, _shape_fortran((n_rhs,)), dtype=np.float64)
        if sources_shape is not None:
            dsets["sources"] = f.create_dataset("sources", _shape_fortran((*sources_shape, n_rhs)), dtype=np.float64)

        # Small arrays accumulated across RHS for derived outputs.
        pf_vm = np.zeros((s, n_rhs), dtype=np.float64)
        hf_vm = np.zeros((s, n_rhs), dtype=np.float64)
        mf_vm = np.zeros((s, n_rhs), dtype=np.float64)
        pf_vm0 = np.zeros((s, n_rhs), dtype=np.float64)
        hf_vm0 = np.zeros((s, n_rhs), dtype=np.float64)
        mf_vm0 = np.zeros((s, n_rhs), dtype=np.float64)
        fsab_flow = np.zeros((s, n_rhs), dtype=np.float64)
        fsa_dens = np.zeros((s, n_rhs), dtype=np.float64)
        fsa_pres = np.zeros((s, n_rhs), dtype=np.float64)
        ntv_arr = np.zeros((s, n_rhs), dtype=np.float64)

        theta_w = np.asarray(op0.theta_weights, dtype=np.float64)
        zeta_w = np.asarray(op0.zeta_weights, dtype=np.float64)
        w2d = theta_w[:, None] * zeta_w[None, :]
        vprime_hat = float(np.sum(w2d / np.asarray(op0.d_hat, dtype=np.float64)))

        geometry_scheme = int(np.asarray(data["geometryScheme"]))
        compute_ntv = geometry_scheme != 5
        bh = np.asarray(data["BHat"], dtype=np.float64)
        if compute_ntv:
            dbt = np.asarray(data["dBHatdtheta"], dtype=np.float64)
            dbz = np.asarray(data["dBHatdzeta"], dtype=np.float64)
            uhat = np.asarray(data["uHat"], dtype=np.float64)
            inv_fsa_b2 = 1.0 / float(np.asarray(data["FSABHat2"], dtype=np.float64))
            ghat = float(np.asarray(data["GHat"], dtype=np.float64))
            ihat = float(np.asarray(data["IHat"], dtype=np.float64))
            iota = float(np.asarray(data["iota"], dtype=np.float64))
            ntv_kernel = (2.0 / 5.0) / bh * (
                (uhat - ghat * inv_fsa_b2) * (iota * dbt + dbz)
                + iota * (1.0 / (bh * bh)) * (ghat * dbt - ihat * dbz)
            )
        else:
            ntv_kernel = np.zeros_like(bh)

        x_grid = np.asarray(op0.x, dtype=np.float64)
        xw = np.asarray(op0.x_weights, dtype=np.float64)
        w_ntv = xw * (x_grid**4)
        z_s = np.asarray(op0.z_s, dtype=np.float64)
        t_hat = np.asarray(op0.t_hat, dtype=np.float64)
        m_hat = np.asarray(op0.m_hat, dtype=np.float64)
        sqrt_t = np.sqrt(t_hat)
        sqrt_m = np.sqrt(m_hat)

        zero_zts = np.zeros((z, t, s), dtype=np.float64)

        for which_rhs in range(1, n_rhs + 1):
            x_full = result.state_vectors_by_rhs.get(int(which_rhs))
            if x_full is None:
                raise ValueError(f"Missing state vector for which_rhs={which_rhs}.")
            j = int(which_rhs) - 1
            op_rhs = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))

            d = v3_rhsmode1_output_fields_vm_only_jit(op_rhs, x_full=x_full)
            diag = v3_transport_diagnostics_vm_only(op_rhs, x_full=x_full)

            dens = np.asarray(np.transpose(d["densityPerturbation"], (2, 1, 0)), dtype=np.float64)
            pres = np.asarray(np.transpose(d["pressurePerturbation"], (2, 1, 0)), dtype=np.float64)
            pres_aniso = np.asarray(np.transpose(d["pressureAnisotropy"], (2, 1, 0)), dtype=np.float64)
            flow = np.asarray(np.transpose(d["flow"], (2, 1, 0)), dtype=np.float64)
            total_dens = np.asarray(np.transpose(d["totalDensity"], (2, 1, 0)), dtype=np.float64)
            total_pres = np.asarray(np.transpose(d["totalPressure"], (2, 1, 0)), dtype=np.float64)
            vel_fsadens = np.asarray(np.transpose(d["velocityUsingFSADensity"], (2, 1, 0)), dtype=np.float64)
            vel_total = np.asarray(np.transpose(d["velocityUsingTotalDensity"], (2, 1, 0)), dtype=np.float64)
            mach = np.asarray(np.transpose(d["MachUsingFSAThermalSpeed"], (2, 1, 0)), dtype=np.float64)
            j_hat = np.asarray(np.transpose(d["jHat"], (1, 0)), dtype=np.float64)
            fsa_dens[:, j] = np.asarray(d["FSADensityPerturbation"], dtype=np.float64)
            fsa_pres[:, j] = np.asarray(d["FSAPressurePerturbation"], dtype=np.float64)

            mf_before_vm = np.asarray(np.transpose(d["momentumFluxBeforeSurfaceIntegral_vm"], (2, 1, 0)), dtype=np.float64)
            mf_before_vm0 = np.asarray(np.transpose(d["momentumFluxBeforeSurfaceIntegral_vm0"], (2, 1, 0)), dtype=np.float64)
            mf_before_vE = np.asarray(np.transpose(d["momentumFluxBeforeSurfaceIntegral_vE"], (2, 1, 0)), dtype=np.float64)
            mf_before_vE0 = np.asarray(np.transpose(d["momentumFluxBeforeSurfaceIntegral_vE0"], (2, 1, 0)), dtype=np.float64)
            mf_vm[:, j] = np.asarray(d["momentumFlux_vm_psiHat"], dtype=np.float64)
            mf_vm0[:, j] = np.asarray(d["momentumFlux_vm0_psiHat"], dtype=np.float64)

            pf_before_vm = np.asarray(np.transpose(diag.particle_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64)
            hf_before_vm = np.asarray(np.transpose(diag.heat_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64)
            pf_before_vm0 = np.asarray(np.transpose(diag.particle_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64)
            hf_before_vm0 = np.asarray(np.transpose(diag.heat_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64)
            pf_vs_x = np.asarray(diag.particle_flux_vm_psi_hat_vs_x, dtype=np.float64)
            hf_vs_x = np.asarray(diag.heat_flux_vm_psi_hat_vs_x, dtype=np.float64)
            flow_vs_x = np.asarray(diag.fsab_flow_vs_x, dtype=np.float64)

            pf_vm[:, j] = np.asarray(diag.particle_flux_vm_psi_hat, dtype=np.float64)
            hf_vm[:, j] = np.asarray(diag.heat_flux_vm_psi_hat, dtype=np.float64)
            fsab_flow[:, j] = np.asarray(diag.fsab_flow, dtype=np.float64)

            pf_vm0[:, j] = np.einsum("tz,stz->s", w2d, np.asarray(diag.particle_flux_before_surface_integral_vm0, dtype=np.float64))
            hf_vm0[:, j] = np.einsum("tz,stz->s", w2d, np.asarray(diag.heat_flux_before_surface_integral_vm0, dtype=np.float64))

            if compute_ntv and int(op0.n_xi) > 2:
                f_delta = np.asarray(x_full[: op0.f_size], dtype=np.float64).reshape(op0.fblock.f_shape)
                sum_ntv = np.einsum("x,sxtz->stz", w_ntv, f_delta[:, :, 2, :, :])
                ntv_before_stz = (
                    (4.0 * np.pi * (t_hat * t_hat) * sqrt_t / (m_hat * sqrt_m * vprime_hat))[:, None, None]
                    * ntv_kernel[None, :, :]
                    * sum_ntv
                )
                ntv_s = np.einsum("tz,stz->s", w2d, ntv_before_stz)
            else:
                ntv_before_stz = np.zeros((s, t, z), dtype=np.float64)
                ntv_s = np.zeros((s,), dtype=np.float64)
            ntv_arr[:, j] = ntv_s
            ntv_before = np.asarray(np.transpose(ntv_before_stz, (2, 1, 0)), dtype=np.float64)

            _write_slice(dsets["densityPerturbation"], j, dens)
            _write_slice(dsets["pressurePerturbation"], j, pres)
            _write_slice(dsets["pressureAnisotropy"], j, pres_aniso)
            _write_slice(dsets["flow"], j, flow)
            _write_slice(dsets["totalDensity"], j, total_dens)
            _write_slice(dsets["totalPressure"], j, total_pres)
            _write_slice(dsets["velocityUsingFSADensity"], j, vel_fsadens)
            _write_slice(dsets["velocityUsingTotalDensity"], j, vel_total)
            _write_slice(dsets["MachUsingFSAThermalSpeed"], j, mach)
            _write_slice(dsets["jHat"], j, j_hat)
            _write_slice(dsets["momentumFluxBeforeSurfaceIntegral_vm"], j, mf_before_vm)
            _write_slice(dsets["momentumFluxBeforeSurfaceIntegral_vm0"], j, mf_before_vm0)
            _write_slice(dsets["momentumFluxBeforeSurfaceIntegral_vE"], j, mf_before_vE)
            _write_slice(dsets["momentumFluxBeforeSurfaceIntegral_vE0"], j, mf_before_vE0)
            _write_slice(dsets["particleFluxBeforeSurfaceIntegral_vm"], j, pf_before_vm)
            _write_slice(dsets["heatFluxBeforeSurfaceIntegral_vm"], j, hf_before_vm)
            _write_slice(dsets["particleFluxBeforeSurfaceIntegral_vm0"], j, pf_before_vm0)
            _write_slice(dsets["heatFluxBeforeSurfaceIntegral_vm0"], j, hf_before_vm0)
            _write_slice(dsets["particleFluxBeforeSurfaceIntegral_vE"], j, zero_zts)
            _write_slice(dsets["heatFluxBeforeSurfaceIntegral_vE"], j, zero_zts)
            _write_slice(dsets["particleFluxBeforeSurfaceIntegral_vE0"], j, zero_zts)
            _write_slice(dsets["heatFluxBeforeSurfaceIntegral_vE0"], j, zero_zts)
            _write_slice(dsets["NTVBeforeSurfaceIntegral"], j, ntv_before)
            _write_slice(dsets["particleFlux_vm_psiHat_vs_x"], j, pf_vs_x)
            _write_slice(dsets["heatFlux_vm_psiHat_vs_x"], j, hf_vs_x)
            _write_slice(dsets["FSABFlow_vs_x"], j, flow_vs_x)

            if sources_shape is not None:
                extra = np.asarray(x_full[op0.f_size + op0.phi1_size :], dtype=np.float64)
                if constraint_scheme == 2:
                    src = extra.reshape((s, x)).T  # (X,S)
                else:
                    src = extra.reshape((s, 2)).T  # (2,S)
                _write_slice(dsets["sources"], j, src)

        # Write small arrays and derived flux variants.
        dsets["FSADensityPerturbation"][...] = fortran_h5_layout(fsa_dens) if not fortran_layout else fsa_dens
        dsets["FSAPressurePerturbation"][...] = fortran_h5_layout(fsa_pres) if not fortran_layout else fsa_pres
        dsets["momentumFlux_vm_psiHat"][...] = fortran_h5_layout(mf_vm) if not fortran_layout else mf_vm
        dsets["momentumFlux_vm0_psiHat"][...] = fortran_h5_layout(mf_vm0) if not fortran_layout else mf_vm0
        dsets["NTV"][...] = fortran_h5_layout(ntv_arr) if not fortran_layout else ntv_arr
        dsets["FSABFlow"][...] = fortran_h5_layout(fsab_flow) if not fortran_layout else fsab_flow

        n_hat = np.asarray(op0.n_hat, dtype=np.float64)
        fsab2 = float(np.asarray(op0.fsab_hat2, dtype=np.float64))
        b0, _g, _i = _flux_functions_from_op(op0)
        b0_val = float(np.asarray(b0, dtype=np.float64))

        fsab_vel = fsab_flow / n_hat[:, None]
        dsets["FSABVelocityUsingFSADensity"][...] = fortran_h5_layout(fsab_vel) if not fortran_layout else fsab_vel
        fsab_vel_b0 = fsab_vel / b0_val
        dsets["FSABVelocityUsingFSADensityOverB0"][...] = fortran_h5_layout(fsab_vel_b0) if not fortran_layout else fsab_vel_b0
        fsab_vel_root = fsab_vel / np.sqrt(fsab2)
        dsets["FSABVelocityUsingFSADensityOverRootFSAB2"][...] = (
            fortran_h5_layout(fsab_vel_root) if not fortran_layout else fsab_vel_root
        )

        fsab_jhat = np.einsum("s,sn->n", z_s, fsab_flow)
        dsets["FSABjHat"][...] = fortran_h5_layout(fsab_jhat) if not fortran_layout else fsab_jhat
        dsets["FSABjHatOverB0"][...] = fortran_h5_layout(fsab_jhat / b0_val) if not fortran_layout else fsab_jhat / b0_val
        dsets["FSABjHatOverRootFSAB2"][...] = (
            fortran_h5_layout(fsab_jhat / np.sqrt(fsab2)) if not fortran_layout else fsab_jhat / np.sqrt(fsab2)
        )

        dsets["particleFlux_vm_psiHat"][...] = fortran_h5_layout(pf_vm) if not fortran_layout else pf_vm
        dsets["heatFlux_vm_psiHat"][...] = fortran_h5_layout(hf_vm) if not fortran_layout else hf_vm
        dsets["particleFlux_vm0_psiHat"][...] = fortran_h5_layout(pf_vm0) if not fortran_layout else pf_vm0
        dsets["heatFlux_vm0_psiHat"][...] = fortran_h5_layout(hf_vm0) if not fortran_layout else hf_vm0

        # Classical fluxes per whichRHS.
        theta_w = np.asarray(op0.theta_weights, dtype=np.float64)
        zeta_w = np.asarray(op0.zeta_weights, dtype=np.float64)
        d_hat = np.asarray(op0.d_hat, dtype=np.float64)
        gpsipsi = np.asarray(data["gpsiHatpsiHat"], dtype=np.float64)
        b_hat = np.asarray(data["BHat"], dtype=np.float64)
        vprime_hat2 = np.asarray(data["VPrimeHat"], dtype=np.float64)
        alpha = np.asarray(data["alpha"], dtype=np.float64)
        delta = np.asarray(data["Delta"], dtype=np.float64)
        nu_n = np.asarray(data["nu_n"], dtype=np.float64)
        z_s = np.asarray(data["Zs"], dtype=np.float64)
        m_hat = np.asarray(data["mHats"], dtype=np.float64)
        t_hat = np.asarray(data["THats"], dtype=np.float64)
        n_hat = np.asarray(data["nHats"], dtype=np.float64)

        classical_pf = np.zeros((s, n_rhs), dtype=np.float64)
        classical_hf = np.zeros((s, n_rhs), dtype=np.float64)
        for which_rhs in range(1, n_rhs + 1):
            op_rhs = with_transport_rhs_settings(op0, which_rhs=which_rhs)
            pf_j, hf_j = classical_flux_v3(
                use_phi1=False,
                theta_weights=theta_w,
                zeta_weights=zeta_w,
                d_hat=d_hat,
                gpsipsi=gpsipsi,
                b_hat=b_hat,
                vprime_hat=vprime_hat2,
                alpha=alpha,
                phi1_hat=np.zeros_like(b_hat),
                delta=delta,
                nu_n=nu_n,
                z_s=z_s,
                m_hat=m_hat,
                t_hat=t_hat,
                n_hat=n_hat,
                dn_hat_dpsi_hat=np.asarray(op_rhs.dn_hat_dpsi_hat, dtype=np.float64),
                dt_hat_dpsi_hat=np.asarray(op_rhs.dt_hat_dpsi_hat, dtype=np.float64),
            )
            classical_pf[:, which_rhs - 1] = np.asarray(pf_j, dtype=np.float64)
            classical_hf[:, which_rhs - 1] = np.asarray(hf_j, dtype=np.float64)

        dsets["classicalParticleFlux_psiHat"][...] = (
            fortran_h5_layout(classical_pf) if not fortran_layout else classical_pf
        )
        dsets["classicalHeatFlux_psiHat"][...] = (
            fortran_h5_layout(classical_hf) if not fortran_layout else classical_hf
        )

        conv = conversion_factors_to_from_dpsi_hat(
            psi_a_hat=float(data["psiAHat"]),
            a_hat=float(data["aHat"]),
            r_n=float(data["rN"]),
        )
        for base, arr in (
            ("particleFlux_vm_psiHat", pf_vm),
            ("heatFlux_vm_psiHat", hf_vm),
            ("momentumFlux_vm_psiHat", mf_vm),
            ("particleFlux_vm0_psiHat", pf_vm0),
            ("heatFlux_vm0_psiHat", hf_vm0),
            ("momentumFlux_vm0_psiHat", mf_vm0),
            ("classicalParticleFlux_psiHat", classical_pf),
            ("classicalHeatFlux_psiHat", classical_hf),
        ):
            dsets[base.replace("_psiHat", "_psiN")][...] = fortran_h5_layout(arr * float(conv["ddpsiN2ddpsiHat"])) if not fortran_layout else arr * float(conv["ddpsiN2ddpsiHat"])
            dsets[base.replace("_psiHat", "_rHat")][...] = fortran_h5_layout(arr * float(conv["ddrHat2ddpsiHat"])) if not fortran_layout else arr * float(conv["ddrHat2ddpsiHat"])
            dsets[base.replace("_psiHat", "_rN")][...] = fortran_h5_layout(arr * float(conv["ddrN2ddpsiHat"])) if not fortran_layout else arr * float(conv["ddrN2ddpsiHat"])

        # Transport matrix + elapsed time
        tm = np.asarray(result.transport_matrix, dtype=np.float64)
        tm_out = tm.T if fortran_layout else tm
        f.create_dataset("transportMatrix", data=tm_out)
        elapsed = np.asarray(result.elapsed_time_s, dtype=np.float64)
        elapsed_out = fortran_h5_layout(elapsed) if not fortran_layout else elapsed
        f.create_dataset("elapsed time (s)", data=elapsed_out)
        if write_solver_diagnostics:
            for name, arr in transport_solver_diagnostic_arrays(result, n_rhs).items():
                f.create_dataset(name, data=arr)

    return output_path.resolve()



__all__ = [
    "TransportStreamingOutputAccumulator",
    "conversion_factors_to_from_dpsi_hat",
    "transport_solver_diagnostic_arrays",
    "write_transport_h5_streaming",
]
