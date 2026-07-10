"""Optimization workflows, objectives, promotion gates, and evidence helpers.

This module owns the optimization-facing public workflow surface. Historical
``optimization_*`` submodules are kept as package-level compatibility aliases in
``sfincs_jax.workflows`` so existing imports continue to resolve without keeping
one implementation file per workflow stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..ambipolar import radial_current_from_output
from ..geometry.jax_adapters import boozer_bhat_from_spectrum
from ..io import localize_equilibrium_file_in_place, read_sfincs_h5
from ..namelist import read_sfincs_input
from ..validation.fortran import run_sfincs_fortran
from .scans import ScanResult, _er_scan_var_name, _patch_scalar_in_group


# optimization_objectives.py
@dataclass(frozen=True)
class AmbipolarRoot:
    """One radial-current zero from an electric-field scan."""

    er: float
    root_type: str
    bracket: tuple[float, float]
    slope: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "er": float(self.er),
            "root_type": self.root_type,
            "bracket": [float(self.bracket[0]), float(self.bracket[1])],
            "slope": float(self.slope),
        }


@dataclass(frozen=True)
class AmbipolarRootSummary:
    """Root-finding summary used by optimization gates."""

    roots: tuple[AmbipolarRoot, ...]
    er_min: float
    er_max: float
    radial_current_min: float
    radial_current_max: float
    bracketed: bool

    @property
    def has_electron_root(self) -> bool:
        return any(root.root_type == "electron" for root in self.roots)

    @property
    def electron_roots(self) -> tuple[AmbipolarRoot, ...]:
        return tuple(root for root in self.roots if root.root_type == "electron")

    def as_dict(self) -> dict[str, Any]:
        return {
            "roots": [root.as_dict() for root in self.roots],
            "er_min": float(self.er_min),
            "er_max": float(self.er_max),
            "radial_current_min": float(self.radial_current_min),
            "radial_current_max": float(self.radial_current_max),
            "bracketed": bool(self.bracketed),
            "has_electron_root": bool(self.has_electron_root),
        }


@dataclass(frozen=True)
class NeoclassicalObjectiveWeights:
    """Weights for the optimization-oriented neoclassical scalar."""

    bootstrap: float = 1.0
    electron_root: float = 1.0
    main_particle_flux: float = 1.0
    main_heat_flux: float = 1.0
    impurity_flux: float = 1.0
    qa_regularization: float = 1.0


def _as_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def find_ambipolar_roots(
    er_values: Any,
    radial_currents: Any,
    *,
    zero_tol: float = 1.0e-12,
) -> AmbipolarRootSummary:
    r"""Find ambipolar roots from a completed electric-field scan.

    The radial current convention is the same one used by
    :mod:`sfincs_jax.ambipolar`: :math:`j_r(E_r)=\sum_s Z_s\Gamma_s(E_r)`.
    Linear interpolation is used deliberately here so the gate has no SciPy
    dependency and remains cheap enough for tests and optimizer bookkeeping.
    """

    er = _as_float_array(er_values, name="er_values").reshape((-1,))
    current = _as_float_array(radial_currents, name="radial_currents").reshape((-1,))
    if er.size != current.size:
        raise ValueError(
            "er_values and radial_currents must have the same length; "
            f"got {er.size} and {current.size}"
        )
    if er.size < 2:
        raise ValueError("at least two electric-field samples are required")

    order = np.argsort(er)
    er = er[order]
    current = current[order]
    if np.any(np.diff(er) <= 0.0):
        raise ValueError("er_values must be distinct after sorting")

    roots: list[AmbipolarRoot] = []
    for idx in range(er.size - 1):
        e0 = float(er[idx])
        e1 = float(er[idx + 1])
        j0 = float(current[idx])
        j1 = float(current[idx + 1])
        slope = (j1 - j0) / (e1 - e0)
        if abs(j0) <= zero_tol:
            root_er = e0
        elif j0 * j1 > 0.0:
            continue
        elif abs(j1) <= zero_tol:
            root_er = e1
        else:
            root_er = e0 - j0 * (e1 - e0) / (j1 - j0)
        root_type = "electron" if root_er > zero_tol else "ion" if root_er < -zero_tol else "near_zero"
        if roots and abs(roots[-1].er - root_er) <= zero_tol:
            continue
        roots.append(
            AmbipolarRoot(
                er=float(root_er),
                root_type=root_type,
                bracket=(e0, e1),
                slope=float(slope),
            )
        )

    return AmbipolarRootSummary(
        roots=tuple(roots),
        er_min=float(er[0]),
        er_max=float(er[-1]),
        radial_current_min=float(np.min(current)),
        radial_current_max=float(np.max(current)),
        bracketed=bool(np.min(current) <= 0.0 <= np.max(current)),
    )


def electron_root_penalty(
    summary: AmbipolarRootSummary,
    *,
    min_positive_er: float = 0.0,
    min_abs_slope: float = 1.0e-12,
) -> float:
    """Return zero only when a resolved positive ambipolar root exists."""

    candidates = [
        root
        for root in summary.electron_roots
        if root.er > float(min_positive_er) and abs(root.slope) >= float(min_abs_slope)
    ]
    if candidates:
        return 0.0
    span = max(summary.er_max - summary.er_min, 1.0e-300)
    if summary.er_max <= min_positive_er:
        return 1.0 + float((min_positive_er - summary.er_max) / span)
    if not summary.bracketed:
        current_span = max(
            abs(summary.radial_current_min),
            abs(summary.radial_current_max),
            1.0e-300,
        )
        same_sign_offset = min(
            abs(summary.radial_current_min),
            abs(summary.radial_current_max),
        )
        return 1.0 + float(same_sign_offset / current_span)
    return 1.0


def bootstrap_current_objective(
    bootstrap_current: Any,
    *,
    normalizer: float = 1.0,
    surface_weights: Any | None = None,
) -> float:
    """Least-squares objective for small normalized bootstrap current."""

    current = _as_float_array(bootstrap_current, name="bootstrap_current").reshape((-1,))
    if normalizer <= 0.0:
        raise ValueError("normalizer must be positive")
    weights = (
        np.ones_like(current)
        if surface_weights is None
        else _as_float_array(surface_weights, name="surface_weights").reshape(current.shape)
    )
    scaled = current / float(normalizer)
    return float(np.sum(weights * scaled**2))


def flux_selectivity_objective(
    particle_flux: Any,
    heat_flux: Any,
    *,
    impurity_species_index: int,
    target_impurity_flux: float,
    main_particle_weight: float = 1.0,
    main_heat_weight: float = 1.0,
    impurity_weight: float = 1.0,
) -> dict[str, float]:
    """Penalize main-species transport while requiring outward impurity flux.

    ``particle_flux`` and ``heat_flux`` are arrays with species on the last axis.
    Positive impurity flux is treated as the desired outward direction.
    """

    gamma = _as_float_array(particle_flux, name="particle_flux")
    heat = _as_float_array(heat_flux, name="heat_flux")
    if gamma.shape != heat.shape:
        raise ValueError(f"particle_flux and heat_flux shapes differ: {gamma.shape} vs {heat.shape}")
    if gamma.ndim == 1:
        gamma = gamma[None, :]
        heat = heat[None, :]
    if gamma.ndim != 2:
        raise ValueError(f"flux arrays must be rank 1 or 2, got rank {gamma.ndim}")
    n_species = gamma.shape[1]
    impurity = int(impurity_species_index)
    if impurity < 0 or impurity >= n_species:
        raise ValueError(f"impurity_species_index={impurity} outside 0..{n_species - 1}")

    main_mask = np.ones((n_species,), dtype=bool)
    main_mask[impurity] = False
    main_particle = float(np.mean(gamma[:, main_mask] ** 2))
    main_heat = float(np.mean(heat[:, main_mask] ** 2))
    impurity_flux = gamma[:, impurity]
    impurity_shortfall = np.maximum(0.0, float(target_impurity_flux) - impurity_flux)
    impurity_penalty = float(np.mean(impurity_shortfall**2))
    total = (
        float(main_particle_weight) * main_particle
        + float(main_heat_weight) * main_heat
        + float(impurity_weight) * impurity_penalty
    )
    return {
        "total": float(total),
        "main_particle": main_particle,
        "main_heat": main_heat,
        "impurity_penalty": impurity_penalty,
        "mean_impurity_flux": float(np.mean(impurity_flux)),
    }


def kinetic_validation_gate(
    *,
    residual_norm: float | None,
    residual_target: float | None,
    cpu_gpu_relative_difference: float | None = None,
    max_cpu_gpu_relative_difference: float = 1.0e-7,
) -> dict[str, Any]:
    """Gate a high-fidelity SFINCS objective before it is trusted."""

    failures: list[str] = []
    if residual_norm is None or residual_target is None:
        failures.append("missing residual_norm or residual_target")
    elif float(residual_norm) > float(residual_target):
        failures.append(f"residual_norm={residual_norm:.3e} exceeds target={residual_target:.3e}")
    if cpu_gpu_relative_difference is not None and (
        float(cpu_gpu_relative_difference) > float(max_cpu_gpu_relative_difference)
    ):
        failures.append(
            "cpu_gpu_relative_difference="
            f"{cpu_gpu_relative_difference:.3e} exceeds {max_cpu_gpu_relative_difference:.3e}"
        )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "residual_norm": None if residual_norm is None else float(residual_norm),
        "residual_target": None if residual_target is None else float(residual_target),
        "cpu_gpu_relative_difference": (
            None
            if cpu_gpu_relative_difference is None
            else float(cpu_gpu_relative_difference)
        ),
        "max_cpu_gpu_relative_difference": float(max_cpu_gpu_relative_difference),
    }


def _smooth_hinge(shortfall: jnp.ndarray, softness: float) -> jnp.ndarray:
    beta = jnp.asarray(max(float(softness), 1.0e-12), dtype=shortfall.dtype)
    return beta * jnp.logaddexp(shortfall / beta, 0.0)


def qa_proxy_neoclassical_components(
    active_bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    b00: float = 1.0,
    target_impurity_flux: float = 0.035,
    target_electron_root_drive: float = 0.02,
    hinge_softness: float = 2.0e-3,
) -> dict[str, jnp.ndarray]:
    """Differentiable proxy components for QA neoclassical optimization.

    The active spectrum excludes the ``B00`` term, which is held fixed by
    default.  The returned terms are smooth functions of the Boozer spectrum and
    are meant for optimizer guidance.  Accepted designs should still be checked
    with high-fidelity SFINCS solves.
    """

    return symmetry_proxy_neoclassical_components(
        active_bmnc_b,
        ixm_b,
        ixn_b,
        theta=theta,
        zeta=zeta,
        b00=b00,
        target_impurity_flux=target_impurity_flux,
        target_electron_root_drive=target_electron_root_drive,
        hinge_softness=hinge_softness,
        symmetry="qa",
        nfp=1,
    )


def symmetry_proxy_neoclassical_components(
    active_bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    b00: float = 1.0,
    target_impurity_flux: float = 0.035,
    target_electron_root_drive: float = 0.02,
    hinge_softness: float = 2.0e-3,
    symmetry: str = "qa",
    nfp: int = 1,
) -> dict[str, jnp.ndarray]:
    """Differentiable QA/QI screening proxy for neoclassical objectives.

    This is a geometry-screening objective, not a kinetic SFINCS solve.  For
    ``symmetry="qa"`` the regularization penalizes non-axisymmetric Boozer
    field-strength content, matching the historical QA proxy.  For
    ``symmetry="qi"`` the regularization instead rewards a field strength whose
    dominant contours close poloidally by allowing ``m=0`` toroidal wells and
    penalizing poloidal ripple.  The QI model is deliberately conservative:
    accepted designs must still pass high-fidelity ``scan-er`` promotion before
    any electron-root claim.
    """

    active = jnp.asarray(active_bmnc_b)
    coeff = jnp.concatenate([jnp.asarray([b00], dtype=active.dtype), active])
    m_mode = jnp.asarray(ixm_b)
    n_mode = jnp.asarray(ixn_b)
    if coeff.shape[0] != m_mode.shape[0] or coeff.shape[0] != n_mode.shape[0]:
        raise ValueError("active_bmnc_b plus b00 must match ixm_b and ixn_b lengths")

    bhat = boozer_bhat_from_spectrum(
        theta,
        zeta,
        bmnc_b=coeff,
        ixm_b=m_mode,
        ixn_b=n_mode,
        normalize=True,
    )
    centered = bhat - jnp.mean(bhat)
    theta_delta = jnp.roll(bhat, -1, axis=0) - bhat
    zeta_delta = jnp.roll(bhat, -1, axis=1) - bhat
    variance = jnp.mean(centered**2)
    theta_roughness = jnp.mean(theta_delta**2)
    zeta_roughness = jnp.mean(zeta_delta**2)

    normalized_symmetry = str(symmetry).strip().lower().replace("-", "_")
    if normalized_symmetry not in {"qa", "qi"}:
        raise ValueError(f"symmetry must be 'qa' or 'qi', got {symmetry!r}")
    nfp_int = max(1, int(nfp))
    active_coeff = coeff / (jnp.abs(coeff[0]) + 1.0e-14)
    if normalized_symmetry == "qa":
        nonqa_mask = n_mode != 0
        symmetry_regularization = jnp.sum(jnp.where(nonqa_mask, active_coeff**2, 0.0))
        allowed_root_drive = zeta_roughness + 0.2 * symmetry_regularization
    else:
        field_periodic = jnp.equal(jnp.mod(jnp.abs(n_mode), nfp_int), 0)
        qi_allowed_mask = jnp.logical_and(m_mode == 0, field_periodic)
        symmetry_regularization = jnp.sum(jnp.where(qi_allowed_mask, 0.0, active_coeff**2))
        theta_mean = jnp.mean(bhat, axis=0)
        qi_well_variation = jnp.mean((theta_mean - jnp.mean(theta_mean)) ** 2)
        allowed_root_drive = qi_well_variation + 0.05 * zeta_roughness

    bootstrap_proxy = symmetry_regularization + 0.05 * zeta_roughness
    main_particle_proxy = variance + 0.25 * symmetry_regularization
    main_heat_proxy = 1.4 * variance + 0.15 * (theta_roughness + zeta_roughness)
    impurity_outward_proxy = symmetry_regularization + 0.5 * zeta_roughness
    impurity_penalty = _smooth_hinge(
        jnp.asarray(target_impurity_flux, dtype=active.dtype) - impurity_outward_proxy,
        hinge_softness,
    ) ** 2
    electron_root_drive = allowed_root_drive
    electron_root_penalty_proxy = _smooth_hinge(
        jnp.asarray(target_electron_root_drive, dtype=active.dtype) - electron_root_drive,
        hinge_softness,
    ) ** 2

    return {
        "bootstrap": bootstrap_proxy,
        "electron_root": electron_root_penalty_proxy,
        "main_particle_flux": main_particle_proxy,
        "main_heat_flux": main_heat_proxy,
        "impurity_flux": impurity_penalty,
        "qa_regularization": symmetry_regularization,
        "symmetry_regularization": symmetry_regularization,
        "electron_root_drive": electron_root_drive,
        "impurity_outward_proxy": impurity_outward_proxy,
        "field_variance": variance,
        "theta_roughness": theta_roughness,
        "zeta_roughness": zeta_roughness,
    }


def qa_proxy_neoclassical_objective(
    active_bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    weights: NeoclassicalObjectiveWeights | None = None,
    **component_kwargs: Any,
) -> jnp.ndarray:
    """Return the weighted differentiable QA neoclassical proxy objective."""

    w = NeoclassicalObjectiveWeights() if weights is None else weights
    components = qa_proxy_neoclassical_components(
        active_bmnc_b,
        ixm_b,
        ixn_b,
        theta=theta,
        zeta=zeta,
        **component_kwargs,
    )
    return (
        float(w.bootstrap) * components["bootstrap"]
        + float(w.electron_root) * components["electron_root"]
        + float(w.main_particle_flux) * components["main_particle_flux"]
        + float(w.main_heat_flux) * components["main_heat_flux"]
        + float(w.impurity_flux) * components["impurity_flux"]
        + float(w.qa_regularization) * components["qa_regularization"]
    )


def symmetry_proxy_neoclassical_objective(
    active_bmnc_b: Any,
    ixm_b: Any,
    ixn_b: Any,
    *,
    theta: Any,
    zeta: Any,
    weights: NeoclassicalObjectiveWeights | None = None,
    **component_kwargs: Any,
) -> jnp.ndarray:
    """Return the weighted QA/QI screening proxy objective."""

    w = NeoclassicalObjectiveWeights() if weights is None else weights
    components = symmetry_proxy_neoclassical_components(
        active_bmnc_b,
        ixm_b,
        ixn_b,
        theta=theta,
        zeta=zeta,
        **component_kwargs,
    )
    return (
        float(w.bootstrap) * components["bootstrap"]
        + float(w.electron_root) * components["electron_root"]
        + float(w.main_particle_flux) * components["main_particle_flux"]
        + float(w.main_heat_flux) * components["main_heat_flux"]
        + float(w.impurity_flux) * components["impurity_flux"]
        + float(w.qa_regularization) * components["qa_regularization"]
    )


def qa_proxy_gradient_gate(
    *,
    n_theta: int = 32,
    n_zeta: int = 24,
    finite_difference_step: float = 1.0e-5,
    rtol: float = 5.0e-4,
    atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Check autodiff and finite-difference agreement for the proxy objective."""

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(n_theta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, jnp.pi, int(n_zeta), endpoint=False, dtype=jnp.float64)
    active0 = jnp.asarray([0.080, 0.030, -0.024, 0.018, -0.010], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 1, 1, 2, 2, 3], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 0, 2, 0, -2, 4], dtype=jnp.int32)

    def objective(active: jnp.ndarray) -> jnp.ndarray:
        return qa_proxy_neoclassical_objective(
            active,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
        )

    value, grad = jax.value_and_grad(objective)(active0)
    eps = float(finite_difference_step)
    basis = jnp.eye(active0.size, dtype=active0.dtype)
    fd_grad = jax.vmap(
        lambda direction: (objective(active0 + eps * direction) - objective(active0 - eps * direction))
        / (2.0 * eps)
    )(basis)
    max_abs_error = float(jnp.max(jnp.abs(grad - fd_grad)))
    scale = float(jnp.max(jnp.abs(fd_grad)))
    tolerance = float(atol) + float(rtol) * scale
    passed = (
        bool(jnp.isfinite(value))
        and bool(jnp.all(jnp.isfinite(grad)))
        and float(jnp.linalg.norm(grad)) > 1.0e-10
        and max_abs_error <= tolerance
    )
    return {
        "status": "pass" if passed else "fail",
        "claim": "differentiable QA neoclassical proxy objective, not a kinetic SFINCS solve",
        "objective": float(value),
        "gradient_norm": float(jnp.linalg.norm(grad)),
        "max_gradient_abs_error": max_abs_error,
        "gradient_tolerance": tolerance,
        "finite_difference_step": eps,
        "grid_shape": {"n_theta": int(n_theta), "n_zeta": int(n_zeta)},
    }

# optimization_promotion.py
@dataclass(frozen=True)
class ScanPromotionRun:
    """One completed Er-scan point used in a promotion audit."""

    path: Path
    er: float
    radial_current: float
    bootstrap_current: float
    particle_flux: tuple[float, ...]
    heat_flux: tuple[float, ...]
    residual_norm: float | None
    residual_target: float | None
    residual_gate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "er": float(self.er),
            "radial_current": float(self.radial_current),
            "bootstrap_current": float(self.bootstrap_current),
            "particle_flux": [float(v) for v in self.particle_flux],
            "heat_flux": [float(v) for v in self.heat_flux],
            "residual_norm": None if self.residual_norm is None else float(self.residual_norm),
            "residual_target": None if self.residual_target is None else float(self.residual_target),
            "residual_gate": self.residual_gate,
        }


@dataclass(frozen=True)
class ScanPromotionSummary:
    """Machine-readable summary for a promoted optimization candidate."""

    scan_dir: Path
    runs: tuple[ScanPromotionRun, ...]
    selected_root: AmbipolarRoot | None
    bootstrap_objective: float
    flux_objective: dict[str, float] | None
    electron_root_penalty: float
    gate_status: str
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
            "scan_dir": str(self.scan_dir),
            "runs": [run.as_dict() for run in self.runs],
            "selected_root": None if self.selected_root is None else self.selected_root.as_dict(),
            "bootstrap_objective": float(self.bootstrap_objective),
            "flux_objective": self.flux_objective,
            "electron_root_penalty": float(self.electron_root_penalty),
            "gate_status": self.gate_status,
            "failures": list(self.failures),
            "claim_boundary": (
                "This audit evaluates completed sfincs_jax kinetic outputs. "
                "It promotes a proxy optimization candidate only if the scan is "
                "bracketed, residual-gated, and the requested observables are present."
            ),
        }


def _scalar(data: dict[str, Any], key: str, *, default: float | None = None) -> float | None:
    if key not in data:
        return default
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return default
    return float(arr.reshape((-1,))[-1])


def _last_species_values(data: dict[str, Any], key: str, *, n_species: int) -> np.ndarray:
    if key not in data:
        return np.zeros((int(n_species),), dtype=np.float64)
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim == 0:
        return np.full((int(n_species),), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        values = arr.reshape((-1,))
        if values.size == int(n_species):
            return values.astype(np.float64)
        return np.full((int(n_species),), float(values[-1]), dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == int(n_species):
            return arr[:, -1].astype(np.float64)
        if arr.shape[1] == int(n_species):
            return arr[-1, :].astype(np.float64)
    raise ValueError(f"{key} has unsupported shape {arr.shape} for n_species={n_species}")


def _output_paths(scan_dir: Path) -> list[Path]:
    direct = scan_dir / "sfincsOutput.h5"
    if direct.exists():
        return [direct]
    paths = sorted(path for path in scan_dir.glob("*/sfincsOutput.h5") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No sfincsOutput.h5 files found under {scan_dir}")
    return paths


def _read_run(path: Path, *, max_residual_ratio: float) -> ScanPromotionRun:
    data = read_sfincs_h5(path)
    er = _scalar(data, "Er")
    if er is None:
        raise KeyError(f"{path} is missing Er")
    n_species = int(_scalar(data, "Nspecies", default=1.0) or 1)
    radial_current = radial_current_from_output(data)
    bootstrap = _scalar(data, "FSABjHatOverRootFSAB2", default=None)
    if bootstrap is None:
        bootstrap = _scalar(data, "FSABjHat", default=0.0) or 0.0
    particle = _last_species_values(data, "particleFlux_vm_rHat", n_species=n_species)
    heat = _last_species_values(data, "heatFlux_vm_rHat", n_species=n_species)
    residual_norm = _scalar(data, "linearSolverResidualNorm", default=None)
    residual_target = _scalar(data, "linearSolverResidualTarget", default=None)
    residual_gate = kinetic_validation_gate(
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    if (
        residual_norm is not None
        and residual_target is not None
        and residual_target > 0.0
        and residual_norm <= float(max_residual_ratio) * residual_target
    ):
        residual_gate = {**residual_gate, "status": "pass", "failures": []}
    return ScanPromotionRun(
        path=path,
        er=float(er),
        radial_current=float(radial_current),
        bootstrap_current=float(bootstrap),
        particle_flux=tuple(float(v) for v in particle),
        heat_flux=tuple(float(v) for v in heat),
        residual_norm=residual_norm,
        residual_target=residual_target,
        residual_gate=residual_gate,
    )


def _interp_at_root(xs: np.ndarray, ys: np.ndarray, x0: float) -> np.ndarray:
    if ys.ndim == 1:
        return np.asarray(np.interp(float(x0), xs, ys), dtype=np.float64)
    cols = [np.interp(float(x0), xs, ys[:, i]) for i in range(ys.shape[1])]
    return np.asarray(cols, dtype=np.float64)


def evaluate_sfincs_scan_promotion(
    scan_dir: str | Path,
    *,
    require_electron_root: bool = False,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
    bootstrap_normalizer: float = 1.0,
    max_residual_ratio: float = 1.0,
    require_residuals: bool = True,
) -> ScanPromotionSummary:
    """Evaluate completed SFINCS outputs for optimization promotion.

    Parameters
    ----------
    scan_dir:
        Directory containing either one ``sfincsOutput.h5`` or subdirectories
        from ``sfincs_jax scan-er``.
    require_electron_root:
        If true, fail the promotion gate unless a resolved positive ambipolar
        root is present.
    impurity_species_index:
        Optional species index used for flux selectivity.  If omitted, the flux
        objective is not evaluated.
    target_impurity_flux:
        Outward impurity flux target for the flux-selectivity gate.
    bootstrap_normalizer:
        Scale for the bootstrap-current least-squares objective.
    max_residual_ratio:
        Accept residuals with ``residual_norm <= max_residual_ratio * target``.
    require_residuals:
        If true, missing residual diagnostics fail the promotion gate.
    """

    root_dir = Path(scan_dir).resolve()
    runs = tuple(
        sorted(
            (_read_run(path, max_residual_ratio=max_residual_ratio) for path in _output_paths(root_dir)),
            key=lambda run: run.er,
        )
    )
    er = np.asarray([run.er for run in runs], dtype=np.float64)
    radial_current = np.asarray([run.radial_current for run in runs], dtype=np.float64)
    root_summary = find_ambipolar_roots(er, radial_current)
    selected_root = None
    if require_electron_root and root_summary.electron_roots:
        selected_root = root_summary.electron_roots[0]
    elif root_summary.roots:
        selected_root = root_summary.roots[0]

    electron_penalty = electron_root_penalty(root_summary) if require_electron_root else 0.0
    bootstrap_values = np.asarray([run.bootstrap_current for run in runs], dtype=np.float64)
    bootstrap_objective = bootstrap_current_objective(
        bootstrap_values,
        normalizer=float(bootstrap_normalizer),
    )

    flux_objective = None
    if impurity_species_index is not None and selected_root is not None:
        particle = np.asarray([run.particle_flux for run in runs], dtype=np.float64)
        heat = np.asarray([run.heat_flux for run in runs], dtype=np.float64)
        particle_at_root = _interp_at_root(er, particle, selected_root.er)
        heat_at_root = _interp_at_root(er, heat, selected_root.er)
        flux_objective = flux_selectivity_objective(
            particle_at_root,
            heat_at_root,
            impurity_species_index=int(impurity_species_index),
            target_impurity_flux=float(target_impurity_flux),
        )

    failures: list[str] = []
    if require_electron_root and electron_penalty > 0.0:
        failures.append("required electron root was not found or was not resolved")
    if not root_summary.bracketed:
        failures.append("ambipolar radial-current scan is not bracketed")
    for run in runs:
        missing_residual = run.residual_norm is None or run.residual_target is None
        if require_residuals and missing_residual:
            failures.append(f"{run.path} is missing linear residual diagnostics")
        elif (not missing_residual) and run.residual_gate["status"] != "pass":
            failures.append(f"{run.path} failed residual gate")
    gate_status = "pass" if not failures else "fail"
    return ScanPromotionSummary(
        scan_dir=root_dir,
        runs=runs,
        selected_root=selected_root,
        bootstrap_objective=float(bootstrap_objective),
        flux_objective=flux_objective,
        electron_root_penalty=float(electron_penalty),
        gate_status=gate_status,
        failures=tuple(failures),
    )

# optimization_workflow.py
@dataclass(frozen=True)
class CandidateScanPlan:
    """Command plan for promoting one optimization candidate with ``scan-er``."""

    proxy_summary: Path
    input_namelist: Path
    out_dir: Path
    er_values: tuple[float, ...]
    compute_solution: bool
    compute_transport_matrix: bool
    jobs: int
    skip_existing: bool
    scan_command: tuple[str, ...]
    promotion_command: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_candidate_scan_plan",
            "proxy_summary": str(self.proxy_summary),
            "input_namelist": str(self.input_namelist),
            "out_dir": str(self.out_dir),
            "er_values": [float(value) for value in self.er_values],
            "compute_solution": bool(self.compute_solution),
            "compute_transport_matrix": bool(self.compute_transport_matrix),
            "jobs": int(self.jobs),
            "skip_existing": bool(self.skip_existing),
            "scan_command": list(self.scan_command),
            "scan_command_string": shlex.join(self.scan_command),
            "promotion_command": list(self.promotion_command),
            "promotion_command_string": shlex.join(self.promotion_command),
            "claim_boundary": (
                "This plan launches high-fidelity sfincs_jax scan-er runs for an "
                "accepted optimization candidate. Promotion still requires the "
                "scan outputs to pass residual, ambipolar-root, flux, and "
                "comparison gates."
            ),
        }


def load_proxy_summary(path: str | Path) -> dict[str, Any]:
    """Load a proxy optimization summary JSON."""

    proxy_path = Path(path).resolve()
    if not proxy_path.exists():
        raise FileNotFoundError(str(proxy_path))
    return json.loads(proxy_path.read_text(encoding="utf-8"))


def er_values_from_bounds(*, er_min: float, er_max: float, n: int) -> tuple[float, ...]:
    """Return deterministic Er scan values including both endpoints."""

    if int(n) < 2:
        raise ValueError("n must be >= 2")
    values = np.linspace(float(er_min), float(er_max), int(n), dtype=np.float64)
    return tuple(float(value) for value in values)


def build_candidate_scan_plan(
    *,
    proxy_summary: str | Path,
    input_namelist: str | Path,
    out_dir: str | Path,
    er_values: tuple[float, ...],
    compute_solution: bool = True,
    compute_transport_matrix: bool = False,
    jobs: int = 1,
    skip_existing: bool = True,
    promotion_stem: str = "candidate_promotion",
    require_electron_root: bool = True,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
) -> CandidateScanPlan:
    """Build scan and promotion commands for an accepted candidate."""

    proxy_path = Path(proxy_summary).resolve()
    input_path = Path(input_namelist).resolve()
    scan_dir = Path(out_dir).resolve()
    values = tuple(float(value) for value in er_values)
    if len(values) < 2:
        raise ValueError("at least two Er values are required")
    if int(jobs) < 1:
        raise ValueError("jobs must be >= 1")

    scan_cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "scan-er",
        "--input",
        str(input_path),
        "--out-dir",
        str(scan_dir),
        "--values",
        *[f"{value:.16g}" for value in values],
    ]
    if compute_solution:
        scan_cmd.append("--compute-solution")
    if compute_transport_matrix:
        scan_cmd.append("--compute-transport-matrix")
    if skip_existing:
        scan_cmd.append("--skip-existing")
    if int(jobs) > 1:
        scan_cmd.extend(["--jobs", str(int(jobs))])

    promotion_cmd = [
        sys.executable,
        "examples/optimization/evaluate_sfincs_jax_promotion_scan.py",
        "--scan-dir",
        str(scan_dir),
        "--out-dir",
        str(scan_dir / "promotion_audit"),
        "--stem",
        str(promotion_stem),
        "--target-impurity-flux",
        f"{float(target_impurity_flux):.16g}",
    ]
    if not require_electron_root:
        promotion_cmd.append("--allow-no-electron-root")
    if impurity_species_index is not None:
        promotion_cmd.extend(["--impurity-species-index", str(int(impurity_species_index))])

    return CandidateScanPlan(
        proxy_summary=proxy_path,
        input_namelist=input_path,
        out_dir=scan_dir,
        er_values=values,
        compute_solution=bool(compute_solution),
        compute_transport_matrix=bool(compute_transport_matrix),
        jobs=int(jobs),
        skip_existing=bool(skip_existing),
        scan_command=tuple(scan_cmd),
        promotion_command=tuple(promotion_cmd),
    )


def write_candidate_scan_plan(path: str | Path, plan: CandidateScanPlan, *, proxy_payload: dict[str, Any] | None = None) -> Path:
    """Write a scan plan and selected proxy metadata as JSON."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.as_dict()
    if proxy_payload is not None:
        payload["proxy_workflow"] = proxy_payload.get("workflow")
        payload["proxy_objective_preset"] = proxy_payload.get("objective_preset")
        payload["proxy_final_components"] = proxy_payload.get("final_components")
        payload["proxy_autodiff_gradient_gate"] = proxy_payload.get("autodiff_gradient_gate")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

# optimization_evidence.py
@dataclass(frozen=True)
class PromotionEvidenceLane:
    """One backend lane in a CPU/GPU/Fortran promotion campaign."""

    label: str
    backend: str
    scan_dir: Path
    promotion_dir: Path
    scan_command: tuple[str, ...] | None
    promotion_command: tuple[str, ...]
    env: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend,
            "scan_dir": str(self.scan_dir),
            "promotion_dir": str(self.promotion_dir),
            "scan_command": None if self.scan_command is None else list(self.scan_command),
            "scan_command_string": None
            if self.scan_command is None
            else shlex.join(self.scan_command),
            "promotion_command": list(self.promotion_command),
            "promotion_command_string": shlex.join(self.promotion_command),
            "env": dict(sorted(self.env.items())),
        }


@dataclass(frozen=True)
class PromotionEvidencePlan:
    """Serializable command plan for a promotion evidence campaign."""

    input_namelist: Path
    out_dir: Path
    er_values: tuple[float, ...]
    lanes: tuple[PromotionEvidenceLane, ...]
    comparison_command: tuple[str, ...] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_promotion_evidence_plan",
            "input_namelist": str(self.input_namelist),
            "out_dir": str(self.out_dir),
            "er_values": [float(value) for value in self.er_values],
            "lanes": [lane.as_dict() for lane in self.lanes],
            "comparison_command": None
            if self.comparison_command is None
            else list(self.comparison_command),
            "comparison_command_string": None
            if self.comparison_command is None
            else shlex.join(self.comparison_command),
            "claim_boundary": (
                "A campaign plan is execution provenance, not promotion evidence. "
                "Promotion requires completed scan outputs, passing residual and "
                "ambipolar gates, and passing backend/reference comparisons."
            ),
        }


def build_promotion_evidence_plan(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    er_values: tuple[float, ...],
    include_cpu: bool = True,
    include_gpu: bool = False,
    include_fortran: bool = False,
    fortran_exe: str | Path | None = None,
    gpu_device: str | None = None,
    jobs: int = 1,
    compute_solution: bool = True,
    compute_transport_matrix: bool = False,
    skip_existing: bool = True,
    require_electron_root: bool = True,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
    require_fortran_residuals: bool = False,
    promotion_stem: str = "candidate_promotion",
    compare_stem: str = "candidate_promotion_comparison",
) -> PromotionEvidencePlan:
    """Build commands for CPU/GPU/Fortran high-fidelity promotion evidence."""

    values = tuple(float(value) for value in er_values)
    if len(values) < 2:
        raise ValueError("at least two Er values are required")
    if int(jobs) < 1:
        raise ValueError("jobs must be >= 1")
    if not (include_cpu or include_gpu or include_fortran):
        raise ValueError("at least one backend lane must be requested")

    input_path = Path(input_namelist).resolve()
    root = Path(out_dir).resolve()
    lanes: list[PromotionEvidenceLane] = []

    if include_cpu:
        lanes.append(
            _build_jax_lane(
                label="cpu",
                input_path=input_path,
                out_dir=root,
                values=values,
                env={"JAX_PLATFORM_NAME": "cpu"},
                jobs=int(jobs),
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
                skip_existing=skip_existing,
                require_electron_root=require_electron_root,
                impurity_species_index=impurity_species_index,
                target_impurity_flux=target_impurity_flux,
                require_residuals=True,
                promotion_stem=promotion_stem,
            )
        )

    if include_gpu:
        env = {"JAX_PLATFORM_NAME": "gpu"}
        if gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        lanes.append(
            _build_jax_lane(
                label="gpu",
                input_path=input_path,
                out_dir=root,
                values=values,
                env=env,
                jobs=int(jobs),
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
                skip_existing=skip_existing,
                require_electron_root=require_electron_root,
                impurity_species_index=impurity_species_index,
                target_impurity_flux=target_impurity_flux,
                require_residuals=True,
                promotion_stem=promotion_stem,
            )
        )

    if include_fortran:
        env: dict[str, str] = {}
        if fortran_exe is not None:
            env["SFINCS_FORTRAN_EXE"] = str(Path(fortran_exe).expanduser())
        lane_dir = root / "fortran_v3_scan"
        lanes.append(
            PromotionEvidenceLane(
                label="fortran_v3",
                backend="fortran_v3",
                scan_dir=lane_dir,
                promotion_dir=root / "fortran_v3_promotion",
                scan_command=None,
                promotion_command=_promotion_command(
                    scan_dir=lane_dir,
                    promotion_dir=root / "fortran_v3_promotion",
                    promotion_stem=promotion_stem,
                    require_electron_root=require_electron_root,
                    impurity_species_index=impurity_species_index,
                    target_impurity_flux=target_impurity_flux,
                    require_residuals=bool(require_fortran_residuals),
                ),
                env=env,
            )
        )

    comparison_command = _comparison_command(
        lanes=tuple(lanes),
        out_dir=root / "comparison",
        stem=compare_stem,
        promotion_stem=promotion_stem,
        require_flux_objective=impurity_species_index is not None,
    )
    return PromotionEvidencePlan(
        input_namelist=input_path,
        out_dir=root,
        er_values=values,
        lanes=tuple(lanes),
        comparison_command=comparison_command,
    )


def write_promotion_evidence_plan(path: str | Path, plan: PromotionEvidencePlan) -> Path:
    """Write a promotion evidence plan to JSON."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def prepare_fortran_er_scan_inputs(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    values: tuple[float, ...],
) -> tuple[Path, ...]:
    """Write Fortran-v3 Er-scan input directories without executing solves."""

    input_path = Path(input_namelist).resolve()
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    template_txt = input_path.read_text(encoding="utf-8")
    nml = read_sfincs_input(input_path)
    var = _er_scan_var_name(nml=nml)
    vals = tuple(sorted((float(value) for value in values), reverse=True))

    scan_txt = template_txt
    if not scan_txt.endswith("\n"):
        scan_txt += "\n"
    scan_txt += f"!ss NErs = {len(vals)}\n"
    scan_txt += f"!ss {var}Min = {min(vals):.16g}\n"
    scan_txt += f"!ss {var}Max = {max(vals):.16g}\n"
    (root / "input.namelist").write_text(scan_txt, encoding="utf-8")

    input_paths: list[Path] = []
    for value in vals:
        run_dir = root / f"{var}{value:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        patched = _patch_scalar_in_group(
            txt=template_txt,
            group="physicsParameters",
            key=var,
            value=float(value),
        )
        run_input = run_dir / "input.namelist"
        run_input.write_text(patched, encoding="utf-8")
        localize_equilibrium_file_in_place(input_namelist=run_input, overwrite=False)
        input_paths.append(run_input)
    return tuple(input_paths)


def run_fortran_er_scan(
    *,
    input_namelist: str | Path,
    out_dir: str | Path,
    values: tuple[float, ...],
    exe: str | Path | None = None,
    timeout_s: float | None = None,
    skip_existing: bool = True,
    emit: Any | None = None,
) -> ScanResult:
    """Run the same Er scan with the external SFINCS Fortran v3 executable."""

    root = Path(out_dir).resolve()
    input_paths = prepare_fortran_er_scan_inputs(
        input_namelist=input_namelist,
        out_dir=root,
        values=values,
    )
    outputs: list[Path] = []
    run_dirs: list[Path] = []
    var = _er_scan_var_name(nml=read_sfincs_input(Path(input_namelist).resolve()))
    total = len(input_paths)
    t0 = time.perf_counter()
    for idx, run_input in enumerate(input_paths, start=1):
        run_dir = run_input.parent
        output_path = run_dir / "sfincsOutput.h5"
        run_dirs.append(run_dir)
        if bool(skip_existing) and output_path.exists():
            if emit is not None:
                emit(0, f"fortran-scan: progress {idx}/{total} {run_dir.name} reused existing output")
            outputs.append(output_path)
            continue
        if emit is not None:
            emit(0, f"fortran-scan: [{idx}/{total}] {run_dir.name}")
        run_sfincs_fortran(
            input_namelist=run_input,
            exe=None if exe is None else Path(exe).expanduser(),
            workdir=run_dir,
            timeout_s=timeout_s,
        )
        if not output_path.exists():
            raise FileNotFoundError(f"Fortran v3 did not create {output_path}")
        outputs.append(output_path)
        if emit is not None:
            elapsed = time.perf_counter() - t0
            remaining = total - idx
            avg = elapsed / max(idx, 1)
            emit(
                0,
                f"fortran-scan: progress {idx}/{total} {run_dir.name} "
                f"avg_point={avg:.1f}s est_remaining={avg * remaining:.1f}s",
            )
    vals = tuple(sorted((float(value) for value in values), reverse=True))
    return ScanResult(
        scan_dir=root,
        run_dirs=tuple(run_dirs),
        outputs=tuple(outputs),
        variable=var,
        values=vals,
    )


def _build_jax_lane(
    *,
    label: str,
    input_path: Path,
    out_dir: Path,
    values: tuple[float, ...],
    env: dict[str, str],
    jobs: int,
    compute_solution: bool,
    compute_transport_matrix: bool,
    skip_existing: bool,
    require_electron_root: bool,
    impurity_species_index: int | None,
    target_impurity_flux: float,
    require_residuals: bool,
    promotion_stem: str,
) -> PromotionEvidenceLane:
    scan_dir = out_dir / f"{label}_scan"
    promotion_dir = out_dir / f"{label}_promotion"
    scan_cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "scan-er",
        "--input",
        str(input_path),
        "--out-dir",
        str(scan_dir),
        "--values",
        *[f"{value:.16g}" for value in values],
    ]
    if compute_solution:
        scan_cmd.append("--compute-solution")
    if compute_transport_matrix:
        scan_cmd.append("--compute-transport-matrix")
    if skip_existing:
        scan_cmd.append("--skip-existing")
    if int(jobs) > 1:
        scan_cmd.extend(["--jobs", str(int(jobs))])
    return PromotionEvidenceLane(
        label=label,
        backend="sfincs_jax",
        scan_dir=scan_dir,
        promotion_dir=promotion_dir,
        scan_command=tuple(scan_cmd),
        promotion_command=_promotion_command(
            scan_dir=scan_dir,
            promotion_dir=promotion_dir,
            promotion_stem=promotion_stem,
            require_electron_root=require_electron_root,
            impurity_species_index=impurity_species_index,
            target_impurity_flux=target_impurity_flux,
            require_residuals=bool(require_residuals),
        ),
        env=dict(env),
    )


def _promotion_command(
    *,
    scan_dir: Path,
    promotion_dir: Path,
    promotion_stem: str,
    require_electron_root: bool,
    impurity_species_index: int | None,
    target_impurity_flux: float,
    require_residuals: bool,
) -> tuple[str, ...]:
    cmd = [
        sys.executable,
        "examples/optimization/evaluate_sfincs_jax_promotion_scan.py",
        "--scan-dir",
        str(scan_dir),
        "--out-dir",
        str(promotion_dir),
        "--stem",
        str(promotion_stem),
        "--target-impurity-flux",
        f"{float(target_impurity_flux):.16g}",
    ]
    if require_electron_root:
        cmd.append("--require-electron-root")
    else:
        cmd.append("--allow-no-electron-root")
    if impurity_species_index is not None:
        cmd.extend(["--impurity-species-index", str(int(impurity_species_index))])
    if not bool(require_residuals):
        cmd.append("--allow-missing-residuals")
    return tuple(cmd)


def _comparison_command(
    *,
    lanes: tuple[PromotionEvidenceLane, ...],
    out_dir: Path,
    stem: str,
    promotion_stem: str,
    require_flux_objective: bool,
) -> tuple[str, ...] | None:
    by_label = {lane.label: lane for lane in lanes}
    if "cpu" not in by_label or "gpu" not in by_label:
        return None
    cmd = [
        sys.executable,
        "examples/optimization/compare_sfincs_jax_promotion_runs.py",
        "--cpu",
        str(by_label["cpu"].promotion_dir / f"{promotion_stem}.json"),
        "--gpu",
        str(by_label["gpu"].promotion_dir / f"{promotion_stem}.json"),
        "--out-dir",
        str(out_dir),
        "--stem",
        str(stem),
    ]
    if "fortran_v3" in by_label:
        cmd.extend(
            [
                "--fortran",
                str(by_label["fortran_v3"].promotion_dir / f"{promotion_stem}.json"),
            ]
        )
    if not bool(require_flux_objective):
        cmd.append("--allow-missing-flux")
    return tuple(cmd)

# optimization_comparison.py
PromotionPayloadInput = Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class PromotionComparisonTolerances:
    """Numeric gates for optimization promotion comparisons."""

    selected_root_er_rtol: float = 1.0e-7
    selected_root_er_atol: float = 1.0e-10
    bootstrap_objective_rtol: float = 1.0e-6
    flux_objective_total_rtol: float = 1.0e-6


_MISSING = object()

_FIELD_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "selected_root_er": (
        ("selected_root", "er"),
        ("selected_root_er",),
        ("ambipolar_root", "er"),
        ("root", "er"),
    ),
    "bootstrap_objective": (
        ("bootstrap_objective",),
        ("objectives", "bootstrap_objective"),
        ("objectives", "bootstrap"),
        ("metrics", "bootstrap_objective", "value"),
    ),
    "flux_objective_total": (
        ("flux_objective", "total"),
        ("flux_objective_total",),
        ("objectives", "flux_objective_total"),
        ("objectives", "flux_total"),
        ("metrics", "flux_objective_total", "value"),
    ),
    "gate_status": (
        ("gate_status",),
        ("promotion_gate", "status"),
    ),
}


def load_promotion_payload(payload: PromotionPayloadInput) -> dict[str, Any]:
    """Return a promotion payload from a mapping, path, or raw JSON string."""

    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, Path):
        loaded = json.loads(payload.read_text(encoding="utf-8"))
    else:
        text = str(payload).strip()
        if text.startswith("{"):
            loaded = json.loads(text)
        else:
            loaded = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("promotion payload JSON must decode to an object")
    return loaded


def compare_promotion_pair(
    reference_payload: PromotionPayloadInput,
    candidate_payload: PromotionPayloadInput,
    *,
    reference_label: str,
    candidate_label: str,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare two optimization promotion summaries.

    The returned dictionary is intended to be JSON-serializable and stable for
    CI/reporting consumers: top-level ``status``, ``failures``, and ``metrics``
    carry all gate outcomes.
    """

    tol = PromotionComparisonTolerances() if tolerances is None else tolerances
    reference = load_promotion_payload(reference_payload)
    candidate = load_promotion_payload(candidate_payload)
    failures: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}

    metrics["gate_status"] = _compare_gate_status(
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        failures=failures,
        require_gate_pass=require_gate_pass,
    )
    metrics["selected_root_er"] = _compare_numeric_field(
        "selected_root_er",
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        rtol=tol.selected_root_er_rtol,
        atol=tol.selected_root_er_atol,
        failures=failures,
    )
    metrics["bootstrap_objective"] = _compare_numeric_field(
        "bootstrap_objective",
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        rtol=tol.bootstrap_objective_rtol,
        atol=0.0,
        failures=failures,
    )
    if require_flux_objective or _has_field(reference, "flux_objective_total") or _has_field(
        candidate,
        "flux_objective_total",
    ):
        metrics["flux_objective_total"] = _compare_numeric_field(
            "flux_objective_total",
            reference,
            candidate,
            reference_label=reference_label,
            candidate_label=candidate_label,
            rtol=tol.flux_objective_total_rtol,
            atol=0.0,
            failures=failures,
        )

    return {
        "comparison": f"{reference_label}_vs_{candidate_label}_promotion",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "metrics": metrics,
        "tolerances": asdict(tol),
        "reference_label": reference_label,
        "candidate_label": candidate_label,
    }


def compare_cpu_gpu_promotions(
    cpu_payload: PromotionPayloadInput,
    gpu_payload: PromotionPayloadInput,
    *,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare CPU and GPU promotion-summary payloads."""

    return compare_promotion_pair(
        cpu_payload,
        gpu_payload,
        reference_label="cpu",
        candidate_label="gpu",
        tolerances=tolerances,
        require_gate_pass=require_gate_pass,
        require_flux_objective=require_flux_objective,
    )


def compare_fortran_promotion(
    sfincs_jax_payload: PromotionPayloadInput,
    fortran_v3_payload: PromotionPayloadInput,
    *,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
    sfincs_jax_label: str = "sfincs_jax",
    fortran_label: str = "fortran_v3",
) -> dict[str, Any]:
    """Compare a SFINCS_JAX promotion payload with a Fortran-v3-derived one."""

    return compare_promotion_pair(
        sfincs_jax_payload,
        fortran_v3_payload,
        reference_label=sfincs_jax_label,
        candidate_label=fortran_label,
        tolerances=tolerances,
        require_gate_pass=require_gate_pass,
        require_flux_objective=require_flux_objective,
    )


def compare_optimization_promotions(
    cpu_payload: PromotionPayloadInput,
    gpu_payload: PromotionPayloadInput,
    *,
    fortran_v3_payload: PromotionPayloadInput | None = None,
    sfincs_jax_fortran_payload: PromotionPayloadInput | None = None,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare CPU/GPU promotions and optionally SFINCS_JAX against Fortran v3.

    When ``fortran_v3_payload`` is provided, the Fortran comparison uses
    ``sfincs_jax_fortran_payload`` as its JAX-side reference if supplied;
    otherwise the CPU payload is used.
    """

    tol = PromotionComparisonTolerances() if tolerances is None else tolerances
    comparisons = {
        "cpu_gpu": compare_cpu_gpu_promotions(
            cpu_payload,
            gpu_payload,
            tolerances=tol,
            require_gate_pass=require_gate_pass,
            require_flux_objective=require_flux_objective,
        )
    }
    if fortran_v3_payload is not None:
        jax_reference = (
            cpu_payload if sfincs_jax_fortran_payload is None else sfincs_jax_fortran_payload
        )
        comparisons["sfincs_jax_fortran_v3"] = compare_fortran_promotion(
            jax_reference,
            fortran_v3_payload,
            tolerances=tol,
            require_gate_pass=require_gate_pass,
            require_flux_objective=require_flux_objective,
        )

    failures: list[str] = []
    for comparison_name, comparison in comparisons.items():
        for failure in comparison["failures"]:
            failures.append(f"{comparison_name}: {failure}")

    return {
        "workflow": "sfincs_jax_optimization_promotion_comparison",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "comparisons": comparisons,
        "tolerances": asdict(tol),
    }


def _lookup(payload: Mapping[str, Any], field: str) -> Any:
    for path in _FIELD_PATHS[field]:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = _MISSING
                break
            current = current[key]
        if current is not _MISSING:
            return current
    return _MISSING


def _has_field(payload: Mapping[str, Any], field: str) -> bool:
    return _lookup(payload, field) is not _MISSING


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _relative_difference(reference: float, candidate: float) -> float:
    diff = abs(candidate - reference)
    scale = abs(reference)
    if scale == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / scale


def _compare_gate_status(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    failures: list[str],
    require_gate_pass: bool,
) -> dict[str, Any]:
    reference_status = _lookup(reference, "gate_status")
    candidate_status = _lookup(candidate, "gate_status")
    reference_text = None if reference_status is _MISSING else str(reference_status).lower()
    candidate_text = None if candidate_status is _MISSING else str(candidate_status).lower()
    metric = {
        "reference": reference_text,
        "candidate": candidate_text,
        "required": "pass" if require_gate_pass else None,
        "status": "pass",
    }
    if require_gate_pass:
        if reference_text != "pass":
            failures.append(f"{reference_label} gate_status is {reference_text!r}; expected 'pass'")
            metric["status"] = "fail"
        if candidate_text != "pass":
            failures.append(f"{candidate_label} gate_status is {candidate_text!r}; expected 'pass'")
            metric["status"] = "fail"
    return metric


def _compare_numeric_field(
    field: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    rtol: float,
    atol: float,
    failures: list[str],
) -> dict[str, Any]:
    reference_raw = _lookup(reference, field)
    candidate_raw = _lookup(candidate, field)
    reference_value = None if reference_raw is _MISSING else _finite_float(reference_raw)
    candidate_value = None if candidate_raw is _MISSING else _finite_float(candidate_raw)
    metric: dict[str, Any] = {
        "reference": reference_value,
        "candidate": candidate_value,
        "abs_diff": None,
        "rel_diff": None,
        "rtol": float(rtol),
        "atol": float(atol),
        "limit": None,
        "status": "fail",
    }
    if reference_value is None:
        failures.append(f"{reference_label} is missing finite {field}")
        return metric
    if candidate_value is None:
        failures.append(f"{candidate_label} is missing finite {field}")
        return metric

    abs_diff = abs(candidate_value - reference_value)
    rel_diff = _relative_difference(reference_value, candidate_value)
    limit = float(atol) + float(rtol) * abs(reference_value)
    passed = abs_diff <= limit
    metric.update(
        {
            "abs_diff": float(abs_diff),
            "rel_diff": float(rel_diff),
            "limit": float(limit),
            "status": "pass" if passed else "fail",
        }
    )
    if not passed:
        failures.append(
            f"{field} differs between {reference_label} and {candidate_label}: "
            f"abs_diff={abs_diff:.6e} exceeds limit={limit:.6e}"
        )
    return metric

# optimization_ladder.py
DEFAULT_PRODUCTION_FLOOR = {
    "Ntheta": 25,
    "Nzeta": 51,
    "Nxi": 100,
    "NL": 4,
    "Nx": 4,
}


def load_ladder_config(path: str | Path) -> dict[str, Any]:
    """Load a finite-beta promotion-ladder configuration JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ladder config JSON must decode to an object")
    return payload


def estimate_rhs1_active_size(
    *,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nx: int,
    n_species: int = 2,
    constraints: int = 4,
) -> int:
    """Estimate RHSMode=1 active unknowns for the finite-beta two-species deck."""

    return int(ntheta) * int(nzeta) * int(nxi) * int(nx) * int(n_species) + int(constraints)


def dense_matrix_gib(active_size: int, *, dtype_bytes: int = 8) -> float:
    """Return the dense matrix memory footprint in GiB."""

    return float(int(active_size) ** 2 * int(dtype_bytes) / (1024.0**3))


def evaluate_promotion_ladder(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    backend_root_atol: float = 1.0e-6,
    root_drift_atol: float = 2.0e-2,
    production_floor: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate a promoted optimization convergence ladder.

    The returned summary uses ``status='pass'`` only when all tiers pass their
    backend gates, the final tier reaches the production floor, and adjacent
    tier root drift is below ``root_drift_atol``.  Otherwise a clean but
    under-resolved ladder is reported as ``deferred`` rather than ``fail``.
    """

    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    floor = _normalized_resolution(production_floor or DEFAULT_PRODUCTION_FLOOR)
    raw_tiers = config.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("ladder config must contain a non-empty 'tiers' list")

    tiers: list[dict[str, Any]] = []
    failures: list[str] = []
    blockers: list[str] = []
    previous_root: float | None = None
    all_backend_clean = True
    all_converged = True

    for index, raw_tier in enumerate(raw_tiers):
        if not isinstance(raw_tier, Mapping):
            raise TypeError("each ladder tier must be an object")
        tier = _evaluate_tier(
            raw_tier,
            base_dir=root,
            production_floor=floor,
            previous_root=previous_root,
            backend_root_atol=float(backend_root_atol),
            root_drift_atol=float(root_drift_atol),
            is_baseline=index == 0,
        )
        tiers.append(tier)
        lane_failures = list(tier.get("failures", []))
        if lane_failures:
            failures.extend(str(item) for item in lane_failures)
            all_backend_clean = False
        if tier["convergence_gate"]["status"] == "fail":
            all_converged = False
            blockers.append(
                f"{tier['name']}: root drift {tier['convergence_gate']['root_drift_from_previous']:.6g} "
                f"exceeds {float(root_drift_atol):.6g}"
            )
        previous_root = _reference_root(tier)

    final_tier = tiers[-1]
    if not final_tier["production_floor_met"]:
        blockers.append(f"{final_tier['name']}: final tier does not meet production floor")

    if failures:
        status = "fail"
    elif final_tier["production_floor_met"] and all_backend_clean and all_converged:
        status = "pass"
    else:
        status = "deferred"

    return {
        "workflow": "sfincs_jax_finite_beta_electron_root_convergence_ladder",
        "status": status,
        "claim_boundary": (
            "This summary compares already-promoted finite-beta QA electron-root "
            "scans across resolution tiers. It is publication-grade only when "
            "the final tier reaches the declared production floor and the "
            "convergence/backend gates pass."
        ),
        "production_floor": floor,
        "tolerances": {
            "backend_root_atol": float(backend_root_atol),
            "root_drift_atol": float(root_drift_atol),
        },
        "tiers": tiers,
        "failures": failures,
        "blockers": blockers,
    }


def _evaluate_tier(
    raw_tier: Mapping[str, Any],
    *,
    base_dir: Path,
    production_floor: Mapping[str, int],
    previous_root: float | None,
    backend_root_atol: float,
    root_drift_atol: float,
    is_baseline: bool,
) -> dict[str, Any]:
    name = str(raw_tier.get("name") or f"tier_{id(raw_tier)}")
    resolution = _normalized_resolution(raw_tier.get("resolution", {}))
    n_species = int(raw_tier.get("n_species", 2))
    active_size = estimate_rhs1_active_size(
        ntheta=resolution["Ntheta"],
        nzeta=resolution["Nzeta"],
        nxi=resolution["Nxi"],
        nx=resolution["Nx"],
        n_species=n_species,
    )
    lanes = _load_lanes(raw_tier.get("promotions", {}), base_dir=base_dir)
    lane_summaries = {lane: _summarize_promotion_payload(payload) for lane, payload in lanes.items()}

    failures: list[str] = []
    for lane, payload in lane_summaries.items():
        if payload["gate_status"] != "pass":
            failures.append(f"{name}:{lane}: promotion gate did not pass")
        if payload["root_type"] != "electron":
            failures.append(f"{name}:{lane}: selected root is not an electron root")
        if not np.isfinite(float(payload["selected_root_er"])):
            failures.append(f"{name}:{lane}: selected root is not finite")

    backend_diffs: dict[str, float] = {}
    if "cpu" in lane_summaries and "gpu" in lane_summaries:
        backend_diffs["cpu_gpu"] = abs(
            float(lane_summaries["cpu"]["selected_root_er"]) - float(lane_summaries["gpu"]["selected_root_er"])
        )
    if "cpu" in lane_summaries and "fortran_v3" in lane_summaries:
        backend_diffs["cpu_fortran_v3"] = abs(
            float(lane_summaries["cpu"]["selected_root_er"])
            - float(lane_summaries["fortran_v3"]["selected_root_er"])
        )
    for pair, diff in backend_diffs.items():
        if diff > float(backend_root_atol):
            failures.append(f"{name}:{pair}: selected root diff {diff:.6g} exceeds {backend_root_atol:.6g}")

    reference_root = _reference_root_from_lanes(lane_summaries)
    if is_baseline or previous_root is None:
        convergence_gate = {
            "status": "baseline",
            "root_drift_from_previous": 0.0,
            "root_drift_atol": float(root_drift_atol),
        }
    else:
        drift = abs(float(reference_root) - float(previous_root))
        convergence_gate = {
            "status": "pass" if drift <= float(root_drift_atol) else "fail",
            "root_drift_from_previous": float(drift),
            "root_drift_atol": float(root_drift_atol),
        }

    return {
        "name": name,
        "resolution": resolution,
        "r_n": float(raw_tier.get("r_n", 0.5)),
        "n_species": n_species,
        "active_size_estimate": int(active_size),
        "dense_matrix_gib": dense_matrix_gib(active_size),
        "production_floor_met": _meets_floor(resolution, production_floor),
        "lanes": lane_summaries,
        "backend_root_diffs": backend_diffs,
        "convergence_gate": convergence_gate,
        "reference_root_er": float(reference_root),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def _normalized_resolution(raw: Mapping[str, Any]) -> dict[str, int]:
    aliases = {
        "Ntheta": ("Ntheta", "ntheta", "n_theta"),
        "Nzeta": ("Nzeta", "nzeta", "n_zeta"),
        "Nxi": ("Nxi", "nxi", "n_xi"),
        "NL": ("NL", "nl", "n_l"),
        "Nx": ("Nx", "nx", "n_x"),
    }
    out: dict[str, int] = {}
    for canonical, keys in aliases.items():
        value = None
        for key in keys:
            if key in raw:
                value = raw[key]
                break
        if value is None:
            if canonical == "NL":
                value = 4
            else:
                raise ValueError(f"resolution is missing {canonical}")
        out[canonical] = int(value)
    return out


def _load_lanes(raw: Any, *, base_dir: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("each ladder tier must define promotion JSON paths")
    lanes: dict[str, dict[str, Any]] = {}
    for lane, raw_path in raw.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base_dir / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must decode to an object")
        lanes[str(lane)] = payload
    return lanes


def _summarize_promotion_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected_root = payload.get("selected_root")
    if not isinstance(selected_root, Mapping):
        selected_root = {}
    flux = payload.get("flux_objective")
    if not isinstance(flux, Mapping):
        flux = {}
    return {
        "gate_status": str(payload.get("gate_status")),
        "selected_root_er": float(selected_root.get("er", np.nan)),
        "root_type": selected_root.get("root_type"),
        "root_bracket": list(selected_root.get("bracket", [])),
        "root_slope": float(selected_root.get("slope", np.nan)),
        "bootstrap_objective": float(payload.get("bootstrap_objective", np.nan)),
        "flux_objective_total": float(flux.get("total", np.nan)),
    }


def _reference_root_from_lanes(lanes: Mapping[str, Mapping[str, Any]]) -> float:
    if "cpu" in lanes:
        return float(lanes["cpu"]["selected_root_er"])
    first = next(iter(lanes.values()))
    return float(first["selected_root_er"])


def _reference_root(tier: Mapping[str, Any]) -> float:
    return float(tier["reference_root_er"])


def _meets_floor(resolution: Mapping[str, int], floor: Mapping[str, int]) -> bool:
    return all(int(resolution[key]) >= int(floor[key]) for key in ("Ntheta", "Nzeta", "Nxi", "NL", "Nx"))


__all__ = [
    "AmbipolarRoot",
    "AmbipolarRootSummary",
    "CandidateScanPlan",
    "DEFAULT_PRODUCTION_FLOOR",
    "NeoclassicalObjectiveWeights",
    "PromotionComparisonTolerances",
    "PromotionEvidenceLane",
    "PromotionEvidencePlan",
    "ScanPromotionRun",
    "ScanPromotionSummary",
    "bootstrap_current_objective",
    "build_candidate_scan_plan",
    "build_promotion_evidence_plan",
    "compare_cpu_gpu_promotions",
    "compare_fortran_promotion",
    "compare_optimization_promotions",
    "compare_promotion_pair",
    "dense_matrix_gib",
    "electron_root_penalty",
    "er_values_from_bounds",
    "estimate_rhs1_active_size",
    "evaluate_promotion_ladder",
    "evaluate_sfincs_scan_promotion",
    "find_ambipolar_roots",
    "flux_selectivity_objective",
    "kinetic_validation_gate",
    "load_ladder_config",
    "load_promotion_payload",
    "load_proxy_summary",
    "prepare_fortran_er_scan_inputs",
    "qa_proxy_gradient_gate",
    "qa_proxy_neoclassical_components",
    "qa_proxy_neoclassical_objective",
    "run_fortran_er_scan",
    "symmetry_proxy_neoclassical_components",
    "symmetry_proxy_neoclassical_objective",
    "write_candidate_scan_plan",
    "write_promotion_evidence_plan",
]
