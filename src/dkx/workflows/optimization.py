"""Optimization workflows, objectives, promotion gates, and evidence helpers.

This module owns the optimization-facing public workflow surface: objectives,
scan promotion and the re-exports of ``optimization_evidence``,
``optimization_comparison`` and ``optimization_ladder``. The older
``optimization_objectives``, ``optimization_promotion`` and
``optimization_workflow`` names are package-level aliases in ``dkx.workflows``.

A note on the word "tier", which means something different here than it used
to mean elsewhere in dkx. The solver tiers were renamed to route names
(structured direct, recycled Krylov, sparse direct) and survive only as
identifiers in :mod:`dkx.solve`. The ``tiers`` in this module are unrelated:
they are the resolution levels of a convergence ladder. ``"tiers"`` is a key
in the user's ladder config and in the emitted summary JSON, so it is a
compatibility surface -- do not rename it while retiring the solver spelling.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from ..runtime import configure as _configure_runtime

_configure_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from ..ambipolar import radial_current_from_output
from ..io import read_sfincs_h5
from .geometry_adapters import boozer_bhat_from_spectrum


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
    :mod:`dkx.ambipolar`: :math:`j_r(E_r)=\sum_s Z_s\Gamma_s(E_r)`.
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


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def kinetic_validation_gate(
    *,
    residual_norm: float | None,
    residual_target: float | None,
    cpu_gpu_relative_difference: float | None = None,
    max_cpu_gpu_relative_difference: float = 1.0e-7,
) -> dict[str, Any]:
    """Gate a high-fidelity SFINCS objective before it is trusted."""

    failures: list[str] = []
    residual_value = _finite_float(residual_norm)
    target_value = _finite_float(residual_target)
    comparison_value = _finite_float(cpu_gpu_relative_difference)
    comparison_limit = _finite_float(max_cpu_gpu_relative_difference)
    if residual_norm is None or residual_target is None:
        failures.append("missing residual_norm or residual_target")
    elif residual_value is None:
        failures.append("residual_norm must be finite")
    elif residual_value < 0.0:
        failures.append("residual_norm must be non-negative")
    elif target_value is None:
        failures.append("residual_target must be finite")
    elif target_value <= 0.0:
        failures.append("residual_target must be positive")
    elif residual_value > target_value:
        failures.append(
            f"residual_norm={residual_value:.3e} exceeds target={target_value:.3e}"
        )

    if comparison_limit is None or comparison_limit < 0.0:
        failures.append("max_cpu_gpu_relative_difference must be finite and non-negative")
    elif cpu_gpu_relative_difference is not None and comparison_value is None:
        failures.append("cpu_gpu_relative_difference must be finite")
    elif comparison_value is not None and comparison_value < 0.0:
        failures.append("cpu_gpu_relative_difference must be non-negative")
    elif comparison_value is not None and comparison_value > comparison_limit:
        failures.append(
            "cpu_gpu_relative_difference="
            f"{comparison_value:.3e} exceeds {comparison_limit:.3e}"
        )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "residual_norm": residual_value,
        "residual_target": target_value,
        "cpu_gpu_relative_difference": comparison_value,
        "max_cpu_gpu_relative_difference": comparison_limit,
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
            "workflow": "dkx_optimization_high_fidelity_promotion",
            "scan_dir": str(self.scan_dir),
            "runs": [run.as_dict() for run in self.runs],
            "selected_root": None if self.selected_root is None else self.selected_root.as_dict(),
            "bootstrap_objective": float(self.bootstrap_objective),
            "flux_objective": self.flux_objective,
            "electron_root_penalty": float(self.electron_root_penalty),
            "gate_status": self.gate_status,
            "failures": list(self.failures),
            "claim_boundary": (
                "This audit evaluates completed dkx kinetic outputs. "
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
        bootstrap = _scalar(data, "FSABjHat", default=None)
    if bootstrap is None:
        raise KeyError(f"{path} is missing FSABjHatOverRootFSAB2 and FSABjHat")
    if not isfinite(bootstrap):
        raise ValueError(f"{path} has a nonfinite bootstrap current")
    particle = _last_species_values(data, "particleFlux_vm_rHat", n_species=n_species)
    heat = _last_species_values(data, "heatFlux_vm_rHat", n_species=n_species)
    residual_norm = _scalar(data, "linearSolverResidualNorm", default=None)
    residual_target = _scalar(data, "linearSolverResidualTarget", default=None)
    residual_gate = kinetic_validation_gate(
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    residual_value = residual_gate["residual_norm"]
    target_value = residual_gate["residual_target"]
    if (
        residual_value is not None
        and residual_value >= 0.0
        and target_value is not None
        and target_value > 0.0
    ):
        residual_ratio = residual_value / target_value
        ratio_passed = residual_ratio <= float(max_residual_ratio)
        residual_gate = {
            **residual_gate,
            "status": "pass" if ratio_passed else "fail",
            "failures": (
                []
                if ratio_passed
                else [
                    f"residual ratio={residual_ratio:.3e} exceeds "
                    f"max_residual_ratio={float(max_residual_ratio):.3e}"
                ]
            ),
        }
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
        from ``dkx scan-er``.
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
    if not isfinite(float(max_residual_ratio)) or float(max_residual_ratio) <= 0.0:
        raise ValueError("max_residual_ratio must be finite and positive")
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
            "workflow": "dkx_optimization_candidate_scan_plan",
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
                "This plan launches high-fidelity dkx scan-er runs for an "
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
        "dkx",
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
        "examples/advanced/evaluate_dkx_promotion_scan.py",
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

from .optimization_comparison import (  # noqa: E402, F401
    _FIELD_PATHS,
    _MISSING,
    PromotionComparisonTolerances,
    PromotionPayloadInput,
    _compare_gate_status,
    _compare_numeric_field,
    _has_field,
    _lookup,
    _relative_difference,
    compare_cpu_gpu_promotions,
    compare_fortran_promotion,
    compare_optimization_promotions,
    compare_promotion_pair,
    load_promotion_payload,
)
from .optimization_evidence import (  # noqa: E402, F401
    PromotionEvidenceLane,
    PromotionEvidencePlan,
    _build_jax_lane,
    _comparison_command,
    _promotion_command,
    build_promotion_evidence_plan,
    prepare_fortran_er_scan_inputs,
    run_fortran_er_scan,
    write_promotion_evidence_plan,
)
from .optimization_ladder import (  # noqa: E402, F401
    DEFAULT_PRODUCTION_FLOOR,
    _evaluate_tier,
    _is_strict_refinement,
    _load_lanes,
    _meets_floor,
    _normalized_resolution,
    _reference_root,
    _reference_root_from_lanes,
    _summarize_promotion_payload,
    dense_matrix_gib,
    estimate_rhs1_active_size,
    evaluate_promotion_ladder,
    load_ladder_config,
)

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
