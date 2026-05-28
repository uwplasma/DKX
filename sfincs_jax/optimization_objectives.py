"""Optimization-oriented neoclassical objective helpers.

This module provides small, composable pieces for stellarator optimization
workflows that want to include ``sfincs_jax``-style transport information.  The
helpers are intentionally split into two classes:

* NumPy postprocessing gates for high-fidelity SFINCS outputs or Er scans.
* Pure-JAX proxy objectives for cheap, differentiable geometry optimization.

The proxy objective is not a kinetic solve.  It is useful inside an optimizer
loop, while the postprocessing gates are the evidence layer used to validate
accepted designs with actual SFINCS solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .jax_geometry_adapters import boozer_bhat_from_spectrum


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

    nonqa_mask = n_mode != 0
    active_coeff = coeff / (jnp.abs(coeff[0]) + 1.0e-14)
    nonqa_energy = jnp.sum(jnp.where(nonqa_mask, active_coeff**2, 0.0))

    bootstrap_proxy = nonqa_energy + 0.05 * zeta_roughness
    main_particle_proxy = variance + 0.25 * nonqa_energy
    main_heat_proxy = 1.4 * variance + 0.15 * (theta_roughness + zeta_roughness)
    impurity_outward_proxy = nonqa_energy + 0.5 * zeta_roughness
    impurity_penalty = _smooth_hinge(
        jnp.asarray(target_impurity_flux, dtype=active.dtype) - impurity_outward_proxy,
        hinge_softness,
    ) ** 2
    electron_root_drive = zeta_roughness + 0.2 * nonqa_energy
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
        "qa_regularization": nonqa_energy,
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


__all__ = [
    "AmbipolarRoot",
    "AmbipolarRootSummary",
    "NeoclassicalObjectiveWeights",
    "bootstrap_current_objective",
    "electron_root_penalty",
    "find_ambipolar_roots",
    "flux_selectivity_objective",
    "kinetic_validation_gate",
    "qa_proxy_gradient_gate",
    "qa_proxy_neoclassical_components",
    "qa_proxy_neoclassical_objective",
]
