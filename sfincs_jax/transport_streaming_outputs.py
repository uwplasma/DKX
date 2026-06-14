"""Streaming RHSMode=2/3 transport-output accumulation.

The transport solve can stream diagnostics one ``whichRHS`` at a time to avoid
keeping every state vector live for large cases.  This module owns the mutable
NumPy buffers used by that path so ``v3_driver.py`` can focus on solve
orchestration rather than output layout details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .diagnostics import u_hat_np
from .transport_matrix import (
    _flux_functions_from_op,
    v3_rhsmode1_output_fields_vm_only_jit,
    v3_transport_diagnostics_vm_only,
)
from .v3_system import with_transport_rhs_settings


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
    outputs.  The object is intentionally not a pytree: it is a host-side output
    accumulator for non-differentiated diagnostics, not part of the solve state.
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
            "velocityUsingFSADensity": _require_array(
                self.velocity_using_fsa_density, "velocityUsingFSADensity"
            ),
            "velocityUsingTotalDensity": _require_array(
                self.velocity_using_total_density, "velocityUsingTotalDensity"
            ),
            "MachUsingFSAThermalSpeed": _require_array(
                self.mach_using_fsa_thermal_speed, "MachUsingFSAThermalSpeed"
            ),
            "jHat": _require_array(self.j_hat, "jHat"),
            "FSADensityPerturbation": _require_array(
                self.fsa_density_perturbation, "FSADensityPerturbation"
            ),
            "FSAPressurePerturbation": _require_array(
                self.fsa_pressure_perturbation, "FSAPressurePerturbation"
            ),
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
            "momentumFlux_vm0_psiHat": _require_array(
                self.momentum_flux_vm0_psi_hat, "momentumFlux_vm0_psiHat"
            ),
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


__all__ = ["TransportStreamingOutputAccumulator"]
