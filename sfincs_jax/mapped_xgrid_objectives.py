"""Differentiable objective functions for mapped SFINCS speed grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.special import gammaln  # noqa: E402

from .adaptive_maps import MappedXGrid, RationalTailXMap, make_reference_eta_grid


Array = jax.Array


@dataclass(frozen=True)
class TransportMomentReport:
    """Diagnostics for a mapped speed-grid moment objective."""

    objective: Array
    moment_loss: Array
    regularization_loss: Array
    powers: Array
    moments: Array
    references: Array
    relative_errors: Array
    regularization: dict[str, Array]


def maxwellian_speed_moment(power: float | Array) -> Array:
    """Return ``int_0^inf x**power exp(-x**2) dx`` for ``power > -1``."""

    p = jnp.asarray(power, dtype=jnp.float64)
    return 0.5 * jnp.exp(gammaln(0.5 * (p + 1.0)))


def analytic_maxwellian_moments(powers: Sequence[float] | Array) -> Array:
    """Return analytic Maxwellian speed moments for a sequence of powers."""

    p = jnp.asarray(powers, dtype=jnp.float64)
    return maxwellian_speed_moment(p)


def mapped_maxwellian_moments(grid: MappedXGrid, powers: Sequence[float] | Array) -> Array:
    """Evaluate Maxwellian speed moments on a mapped ``x`` grid."""

    p = jnp.asarray(powers, dtype=jnp.float64)
    x_pow = grid.x[:, None] ** p[None, :]
    return jnp.sum(grid.x_weights[:, None] * x_pow * jnp.exp(-(grid.x[:, None] ** 2)), axis=0)


def relative_moment_errors(grid: MappedXGrid, powers: Sequence[float] | Array) -> Array:
    """Return relative errors against analytic Maxwellian speed moments."""

    moments = mapped_maxwellian_moments(grid, powers)
    refs = analytic_maxwellian_moments(powers)
    return (moments - refs) / refs


def transport_moment_report(
    grid: MappedXGrid,
    *,
    powers: Sequence[float] | Array = (2.0, 4.0, 6.0),
    moment_weights: Sequence[float] | Array | None = None,
    regularization_weights: Mapping[str, float] | None = None,
) -> TransportMomentReport:
    """Return a differentiable moment-matching objective and diagnostics.

    The moment powers default to low-order Maxwellian speed moments that enter
    transport-weighted velocity integrals. This is a proxy objective, not a
    replacement for solving the drift-kinetic system.
    """

    p = jnp.asarray(powers, dtype=jnp.float64)
    errors = relative_moment_errors(grid, p)
    if moment_weights is None:
        weights = jnp.ones_like(errors)
    else:
        weights = jnp.asarray(moment_weights, dtype=jnp.float64)
        if weights.shape != errors.shape:
            raise ValueError("moment_weights must have the same shape as powers")
    moment_loss = jnp.sum(weights * errors**2) / jnp.sum(weights)

    reg_loss = jnp.asarray(0.0, dtype=jnp.float64)
    if regularization_weights is not None:
        for name, weight in regularization_weights.items():
            if name not in grid.regularization:
                raise KeyError(f"Unknown mapped-grid regularization diagnostic {name!r}")
            reg_loss = reg_loss + jnp.asarray(weight, dtype=jnp.float64) * grid.regularization[name]

    return TransportMomentReport(
        objective=moment_loss + reg_loss,
        moment_loss=moment_loss,
        regularization_loss=reg_loss,
        powers=p,
        moments=mapped_maxwellian_moments(grid, p),
        references=analytic_maxwellian_moments(p),
        relative_errors=errors,
        regularization=grid.regularization,
    )


def transport_moment_objective(
    grid: MappedXGrid,
    *,
    powers: Sequence[float] | Array = (2.0, 4.0, 6.0),
    moment_weights: Sequence[float] | Array | None = None,
    regularization_weights: Mapping[str, float] | None = None,
) -> Array:
    """Return only the scalar mapped-grid transport moment objective."""

    return transport_moment_report(
        grid,
        powers=powers,
        moment_weights=moment_weights,
        regularization_weights=regularization_weights,
    ).objective


def rational_tail_transport_grid(
    n: int,
    log_length: float | Array,
    *,
    eta_kind: str = "gauss",
    eps: float = 1.0e-6,
    derivative: str = "barycentric",
) -> MappedXGrid:
    """Build a rational-tail mapped grid for transport moment objectives."""

    eta, eta_weights = make_reference_eta_grid(int(n), kind=eta_kind)
    return RationalTailXMap(eps=eps, derivative=derivative)(
        log_length,
        eta=eta,
        eta_weights=eta_weights,
    )


def brute_force_rational_tail_moment_baseline(
    n: int,
    *,
    log_length_values: Sequence[float] | Array | None = None,
    powers: Sequence[float] | Array = (2.0, 4.0, 6.0),
    regularization_weights: Mapping[str, float] | None = None,
) -> dict[str, Array]:
    """Tune the one-parameter rational-tail map by brute force.

    This provides a deterministic non-gradient baseline for later optimizer and
    implicit-solve studies.
    """

    if log_length_values is None:
        values = jnp.linspace(-3.0, 2.0, 81)
    else:
        values = jnp.asarray(log_length_values, dtype=jnp.float64)
    if values.ndim != 1:
        raise ValueError("log_length_values must be one-dimensional")

    objectives = []
    for value in np.asarray(values, dtype=np.float64):
        grid = rational_tail_transport_grid(int(n), float(value))
        objectives.append(
            transport_moment_objective(
                grid,
                powers=powers,
                regularization_weights=regularization_weights,
            )
        )
    objective_arr = jnp.asarray(objectives, dtype=jnp.float64)
    idx = int(jnp.argmin(objective_arr))
    return {
        "log_length": values[idx],
        "objective": objective_arr[idx],
        "objectives": objective_arr,
        "log_length_values": values,
    }
