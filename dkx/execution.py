"""Native Case normalization and execution without a namelist round-trip."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dkx.config import Case, CaseValidationError
from dkx.constants import RadialCoordinates
from dkx.result import Result


_ATOMIC_MASS_KG = 1.66053906892e-27
_PROTON_MASS_KG = 1.67262192369e-27


@dataclass(frozen=True)
class _GeometryState:
    """Geometry data and normalization loaded once for a native profile."""

    source: Any
    n_periods: int
    psi_a_hat: float
    a_hat: float


def _electric_field_kv_m_to_er_hat(value_kv_m: float) -> float:
    """Normalize ``Er`` using ``PhiHat=e*Phi/TBar`` and ``rHat=r/RBar``."""

    from dkx.units import ELEMENTARY_CHARGE, R_BAR, T_BAR  # noqa: PLC0415

    return float(value_kv_m) * 1000.0 * ELEMENTARY_CHARGE * R_BAR / T_BAR


def _unsupported(path: str, value: Any, expected: str, correction: str) -> None:
    raise CaseValidationError(path, value, expected, correction)


def _analytic_scheme(case: Case) -> int:
    token = str(case.geometry.file).lower().replace("-", "_")
    aliases = {
        "tokamak": 1,
        "lhd_standard": 2,
        "lhd_inward": 3,
        "w7x_standard": 4,
    }
    try:
        return aliases[token]
    except KeyError:
        _unsupported(
            "geometry.file",
            str(case.geometry.file),
            "one of tokamak, lhd_standard, lhd_inward, or w7x_standard for analytic geometry",
            "Use a named built-in analytic equilibrium; VMEC/Boozer native execution is the next route.",
        )
        raise AssertionError("unreachable")


def _validate_native_slice(case: Case) -> None:
    if case.run.workflow != "profile":
        _unsupported(
            "run.workflow",
            case.run.workflow,
            "profile",
            "Use workflow = 'profile' for this first native execution route.",
        )
    if case.geometry.format == "boozer":
        _unsupported(
            "geometry.format",
            case.geometry.format,
            "analytic or vmec",
            "Use a built-in analytic geometry or a VMEC wout file; native Boozer execution needs a dedicated reader.",
        )
    if case.geometry.format == "analytic":
        _analytic_scheme(case)
    if case.physics.magnetic_drifts != "dkes":
        _unsupported(
            "physics.magnetic_drifts",
            case.physics.magnetic_drifts,
            "dkes",
            "Set magnetic_drifts = 'dkes' for the supported local analytic route.",
        )
    if case.physics.phi1 != "off":
        _unsupported(
            "physics.phi1",
            case.physics.phi1,
            "off",
            "Set phi1 = 'off' for this route.",
        )
    if case.electric_field.mode != "prescribed":
        _unsupported(
            "electric_field.mode",
            case.electric_field.mode,
            "prescribed",
            "Supply value_kV_m and use prescribed mode; native root continuation is a separate workflow.",
        )
    if case.electric_field.value_kV_m is None:
        _unsupported(
            "electric_field.value_kV_m",
            None,
            "a finite electric field in kV/m",
            "Add value_kV_m = 0.0 for a zero-field calculation.",
        )
    if case.scan is not None:
        _unsupported(
            "scan",
            "configured",
            "no scan for dkx.run(case)",
            "Remove [scan] for a single run; dkx.scan will own resumable expansion.",
        )
    if case.convergence.enabled:
        _unsupported(
            "convergence.enabled",
            True,
            "false for dkx.run(case)",
            "Disable convergence refinement; dkx.converge will own rung expansion and certificates.",
        )
    if case.parallel.strategy == "batch" or case.parallel.shard:
        _unsupported(
            "parallel",
            {"strategy": case.parallel.strategy, "shard": case.parallel.shard},
            "strategy = 'auto' or 'serial' with no shard axes",
            "Remove explicit sharding for this sequential surface route.",
        )
    if case.solver.reuse != "auto":
        _unsupported(
            "solver.reuse",
            case.solver.reuse,
            "auto",
            "Use reuse = 'auto'; explicit reuse policy arrives with surface-state reuse.",
        )
    if len(case.geometry.surfaces) < 2:
        _unsupported(
            "geometry.surfaces",
            case.geometry.surfaces,
            "at least two surfaces",
            "Give at least two profile locations so density and temperature gradients are defined.",
        )
    if any(value <= 0.0 for value in case.geometry.surfaces) or any(
        right <= left
        for left, right in zip(case.geometry.surfaces, case.geometry.surfaces[1:])
    ):
        _unsupported(
            "geometry.surfaces",
            case.geometry.surfaces,
            "strictly increasing normalized toroidal flux values above zero",
            "Sort the surfaces and omit the magnetic axis, where the radial Jacobian is singular.",
        )


def _profile_matrix(case: Case, attribute: str) -> np.ndarray:
    values = np.asarray(
        [getattr(species, attribute) for species in case.species], dtype=np.float64
    )
    return values.T  # (surface, species)


def _profile_gradients(values: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    edge_order = 2 if r_hat.size >= 3 else 1
    return np.asarray(
        np.gradient(values, r_hat, axis=0, edge_order=edge_order), dtype=np.float64
    )


def _prepare_geometry(case: Case) -> _GeometryState:
    if case.geometry.format == "analytic":
        scheme = _analytic_scheme(case)
        psi_a_hat, a_hat = {
            1: (0.15596, 0.5585),
            2: (0.5585**2 / 2.0, 0.5585),
            3: (0.5400**2 / 2.0, 0.5400),
            4: (-0.384935, 0.5109),
        }[scheme]
        return _GeometryState(
            source=scheme,
            n_periods={1: 10, 2: 10, 3: 10, 4: 5}[scheme],
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
        )

    from dkx.magnetic_geometry import (  # noqa: PLC0415
        psi_a_hat_from_wout,
        read_vmec_wout,
    )

    wout = read_vmec_wout(case.geometry_path)
    return _GeometryState(
        source=wout,
        n_periods=int(wout.nfp),
        psi_a_hat=psi_a_hat_from_wout(wout),
        a_hat=float(wout.aminor_p),
    )


def _make_grids(case: Case, *, n_periods: int):
    from dkx.phase_space import make_grids  # noqa: PLC0415

    return make_grids(
        n_theta=case.resolution.theta,
        n_zeta=case.resolution.zeta,
        n_xi=case.resolution.pitch,
        n_x=case.resolution.speed,
        n_l=min(4, case.resolution.pitch),
        n_periods=n_periods,
        theta_derivative_scheme=2,
        zeta_derivative_scheme=2,
        magnetic_drift_derivative_scheme=3,
        x_grid_scheme=5,
        x_grid_k=0.0,
        x_max=5.0,
        x_dot_derivative_scheme=0,
        n_xi_for_x_option=1,
        monoenergetic=False,
    )


def _geometry_context(
    case: Case, grids, surface_index: int, geometry_state: _GeometryState
):
    from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: PLC0415

    psi_n = float(case.geometry.surfaces[surface_index])
    r_n = math.sqrt(psi_n)
    if case.geometry.format == "analytic":
        geometry = FluxSurfaceGeometry.from_scheme(
            int(geometry_state.source), theta=grids.theta, zeta=grids.zeta
        )
    else:
        geometry = FluxSurfaceGeometry.from_vmec(
            geometry_state.source,
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=psi_n,
            vmec_radial_option=0,
        )
    return geometry, RadialCoordinates(
        psi_a_hat=geometry_state.psi_a_hat,
        a_hat=geometry_state.a_hat,
        r_n=r_n,
    )


def _make_operator(
    case: Case,
    *,
    surface_index: int,
    n_hat: np.ndarray,
    t_hat: np.ndarray,
    dn_dr_hat: np.ndarray,
    dt_dr_hat: np.ndarray,
    grids,
    geometry_state: _GeometryState,
):
    import jax.numpy as jnp  # noqa: PLC0415

    from dkx.collisions import (  # noqa: PLC0415
        make_fokker_planck_v3_operator,
        make_pitch_angle_scattering_v3_operator,
    )
    from dkx.constants import DEFAULT_DELTA, DEFAULT_NU_N  # noqa: PLC0415
    from dkx.drift_kinetic import KineticOperator  # noqa: PLC0415
    from dkx.phase_space import (  # noqa: PLC0415
        legendre_coupling_lower,
        legendre_coupling_upper,
    )

    geometry, radial = _geometry_context(case, grids, surface_index, geometry_state)
    z_s = jnp.asarray([species.charge for species in case.species], dtype=jnp.float64)
    m_hat = jnp.asarray(
        [
            species.mass_amu * _ATOMIC_MASS_KG / _PROTON_MASS_KG
            for species in case.species
        ],
        dtype=jnp.float64,
    )
    n_surface = jnp.asarray(n_hat[surface_index], dtype=jnp.float64)
    t_surface = jnp.asarray(t_hat[surface_index], dtype=jnp.float64)
    dn_dpsi = jnp.asarray(
        radial.d_dr_hat_to_d_dpsi_hat * dn_dr_hat[surface_index], dtype=jnp.float64
    )
    dt_dpsi = jnp.asarray(
        radial.d_dr_hat_to_d_dpsi_hat * dt_dr_hat[surface_index], dtype=jnp.float64
    )
    pas = None
    fp = None
    if case.physics.collisions == "pitch_angle_scattering":
        pas = make_pitch_angle_scattering_v3_operator(
            x=grids.x,
            z_s=z_s,
            m_hats=m_hat,
            n_hats=n_surface,
            t_hats=t_surface,
            nu_n=DEFAULT_NU_N,
            n_xi_for_x=grids.n_xi_for_x,
            n_xi=grids.n_xi,
        )
        constraint_scheme = 2
    else:
        fp = make_fokker_planck_v3_operator(
            x=np.asarray(grids.x),
            x_weights=np.asarray(grids.x_weights),
            ddx=np.asarray(grids.ddx),
            d2dx2=np.asarray(grids.d2dx2),
            x_grid_k=0.0,
            z_s=np.asarray(z_s),
            m_hats=np.asarray(m_hat),
            n_hats=np.asarray(n_surface),
            t_hats=np.asarray(t_surface),
            nu_n=DEFAULT_NU_N,
            krook=0.0,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            n_xi_for_x=np.asarray(grids.n_xi_for_x),
            strict_parity=len(case.species) > 1,
        )
        constraint_scheme = 1

    er_hat = _electric_field_kv_m_to_er_hat(case.electric_field.value_kV_m)
    dphi = radial.d_dr_hat_to_d_dpsi_hat * (-er_hat)
    fsab_hat2 = geometry.fsab_hat2(
        theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights
    )
    op = KineticOperator(
        n_species=len(case.species),
        n_x=grids.n_x,
        n_xi=grids.n_xi,
        n_theta=grids.n_theta,
        n_zeta=grids.n_zeta,
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        use_dkes_exb=True,
        with_exb=er_hat != 0.0,
        with_er_xidot=False,
        with_er_xdot=False,
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        n_xi_for_x=grids.n_xi_for_x,
        xi_coupling_lower=jnp.asarray(legendre_coupling_lower(grids.n_xi)),
        xi_coupling_upper=jnp.asarray(legendre_coupling_upper(grids.n_xi)),
        b_hat=geometry.b_hat,
        db_hat_dtheta=geometry.db_hat_dtheta,
        db_hat_dzeta=geometry.db_hat_dzeta,
        d_hat=geometry.d_hat,
        b_hat_sup_theta=geometry.b_hat_sup_theta,
        b_hat_sup_zeta=geometry.b_hat_sup_zeta,
        b_hat_sub_theta=geometry.b_hat_sub_theta,
        b_hat_sub_zeta=geometry.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        z_s=z_s,
        m_hat=m_hat,
        t_hat=t_surface,
        n_hat=n_surface,
        dn_hat_dpsi_hat=dn_dpsi,
        dt_hat_dpsi_hat=dt_dpsi,
        alpha=jnp.asarray(1.0, dtype=jnp.float64),
        delta=jnp.asarray(DEFAULT_DELTA, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi, dtype=jnp.float64),
        e_parallel_hat=jnp.asarray(0.0, dtype=jnp.float64),
        e_parallel_hat_spec=jnp.zeros((len(case.species),), dtype=jnp.float64),
        pas=pas,
        fp=fp,
    )
    return op, grids, geometry, radial


def _route_name(method: str) -> str:
    return {
        "auto": "auto",
        "structured_direct": "block_tridiagonal",
        "recycled_krylov": "gmres",
        "sparse_direct_referee": "direct",
    }[method]


def _sha256(path_or_token: Path) -> str:
    if path_or_token.exists() and path_or_token.is_file():
        digest = hashlib.sha256()
        with path_or_token.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    return hashlib.sha256(str(path_or_token).encode("utf-8")).hexdigest()


def _peak_host_memory_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _total_host_memory_bytes() -> int:
    """Best-effort physical host memory for the case's fractional budget."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return max(_peak_host_memory_bytes(), 1024**3)


def run_case(case: Case, *, out: str | Path | None = None, emit=None) -> Result:
    """Execute the supported native profile route and return a native Result."""

    _validate_native_slice(case)
    from dkx import __version__  # noqa: PLC0415
    from dkx.run import profile_moments_from_operator  # noqa: PLC0415
    from dkx.solve import solve  # noqa: PLC0415
    from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    total_start = time.perf_counter()
    surfaces = np.asarray(case.geometry.surfaces, dtype=np.float64)
    r_n = np.sqrt(surfaces)
    geometry_state = _prepare_geometry(case)
    grids = _make_grids(case, n_periods=geometry_state.n_periods)
    r_hat = geometry_state.a_hat * r_n
    density_m3 = _profile_matrix(case, "density_m3")
    temperature_keV = _profile_matrix(case, "temperature_keV")
    n_hat = density_m3 / 1.0e20
    t_hat = temperature_keV
    dn_dr_hat = _profile_gradients(n_hat, r_hat)
    dt_dr_hat = _profile_gradients(t_hat, r_hat)

    shape = (surfaces.size, len(case.species))
    particle_flux = np.empty(shape, dtype=np.float64)
    heat_flux = np.empty(shape, dtype=np.float64)
    current = np.empty((surfaces.size,), dtype=np.float64)
    residuals = np.empty((surfaces.size,), dtype=np.float64)
    iterations = np.empty((surfaces.size,), dtype=np.int64)
    solve_seconds = np.empty((surfaces.size,), dtype=np.float64)
    retained_operator = None
    selected_routes: list[str] = []

    progress = emit if emit is not None else (print if case.run.progress else None)
    solved = None
    for index, surface in enumerate(surfaces):
        if progress is not None:
            progress(f"surface {index + 1}/{len(surfaces)}: psi_N={surface:.6g}")
        op, _grids, _geometry, radial = _make_operator(
            case,
            surface_index=index,
            n_hat=n_hat,
            t_hat=t_hat,
            dn_dr_hat=dn_dr_hat,
            dt_dr_hat=dt_dr_hat,
            grids=grids,
            geometry_state=geometry_state,
        )
        solve_start = time.perf_counter()
        solved = solve(
            op,
            op.rhs(),
            method=_route_name(case.solver.method),
            tol=case.solver.relative_tolerance,
            device=None if case.run.device == "auto" else case.run.device,
            tier1_memory_budget_gb=case.solver.memory_fraction
            * _total_host_memory_bytes()
            / (1024**3),
        )
        solve_seconds[index] = time.perf_counter() - solve_start
        if not solved.converged:
            raise RuntimeError(
                f"native profile solve did not converge at geometry.surfaces[{index}]={surface}; "
                f"route={solved.method}, residuals={np.asarray(solved.residual_norms)!r}"
            )
        state = np.asarray(solved.x, dtype=np.float64).reshape((-1,))
        moments = profile_moments_from_operator(op, state)
        radial_factor = radial.d_dpsi_hat_to_d_dr_hat
        particle_flux[index] = (
            np.asarray(moments["particleFlux_vm_psiHat"], dtype=np.float64)
            * radial_factor
            * PARTICLE_FLUX
        )
        heat_flux[index] = (
            np.asarray(moments["heatFlux_vm_psiHat"], dtype=np.float64)
            * radial_factor
            * HEAT_FLUX
        )
        current[index] = float(np.asarray(moments["FSABjHat"])) * PARALLEL_CURRENT
        norms = np.atleast_1d(np.asarray(solved.residual_norms, dtype=np.float64))
        residuals[index] = float(np.max(norms))
        iterations[index] = 0 if solved.iterations is None else int(solved.iterations)
        selected_routes.append(str(solved.method))
        retained_operator = op

    total_seconds = time.perf_counter() - total_start
    output_path = (
        Path(out).expanduser().resolve()
        if out is not None
        else (case.base_directory / case.output.file).resolve()
    )
    assert solved is not None
    device = solved.x.device
    arrays = {
        "surface": surfaces,
        "r_N": r_n,
        "species": np.asarray([species.name for species in case.species], dtype=object),
        "charge_e": np.asarray([species.charge for species in case.species]),
        "mass_amu": np.asarray([species.mass_amu for species in case.species]),
        "density_m3": density_m3,
        "temperature_keV": temperature_keV,
        "electric_field_kV_m": np.full(
            (surfaces.size,), case.electric_field.value_kV_m
        ),
        "particle_flux_m2_s": particle_flux,
        "heat_flux_W_m2": heat_flux,
        "parallel_current_A_T_m2": current,
        "primal_residual": residuals,
        "solver_iterations": iterations,
        "solve_time_s": solve_seconds,
    }
    dimensions = {
        "surface": ("surface",),
        "r_N": ("surface",),
        "species": ("species",),
        "charge_e": ("species",),
        "mass_amu": ("species",),
        "density_m3": ("surface", "species"),
        "temperature_keV": ("surface", "species"),
        "electric_field_kV_m": ("surface",),
        "particle_flux_m2_s": ("surface", "species"),
        "heat_flux_W_m2": ("surface", "species"),
        "parallel_current_A_T_m2": ("surface",),
        "primal_residual": ("surface",),
        "solver_iterations": ("surface",),
        "solve_time_s": ("surface",),
    }
    route_set = sorted(set(selected_routes))
    metadata = {
        "canonical_case": case.to_dict(),
        "converged": True,
        "solver_route": route_set[0] if len(route_set) == 1 else route_set,
        "route_reason": "selected from operator structure and requested native solver method",
        "residual_norm": float(np.max(residuals)),
        "iterations": int(np.sum(iterations)),
        "normalization": {
            "density_m3": 1.0e20,
            "temperature_keV": 1.0,
            "mass_kg": _PROTON_MASS_KG,
            "a_hat": geometry_state.a_hat,
            "psi_a_hat": geometry_state.psi_a_hat,
        },
        "geometry_sha256": _sha256(
            case.geometry_path if case.geometry.format == "vmec" else case.geometry.file
        ),
        "dkx_version": __version__,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "precision": case.run.precision,
        "device": f"{device.platform}:{device.device_kind}",
        "timings_s": {"solve": float(np.sum(solve_seconds)), "total": total_seconds},
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
    }
    result = Result(
        case_id=case.case_id,
        case_name=case.name,
        workflow=case.run.workflow,
        arrays=arrays,
        dimensions=dimensions,
        metadata=metadata,
        output_path=output_path,
        _runtime={"operator": retained_operator},
    )
    if out is not None:
        result.save(output_path)
    return result
