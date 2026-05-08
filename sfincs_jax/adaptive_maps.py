from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MappedXGrid:
    """Differentiable mapped speed grid primitives.

    The weights stored here are plain ``dx`` quadrature weights. Maxwellian or
    SFINCS-v3 polynomial weights should remain part of the physical integrands
    unless a solver integration explicitly chooses otherwise.
    """

    eta: Array
    eta_weights: Array
    x: Array
    jac: Array
    x_weights: Array
    ddx: Array
    d2dx2: Array
    regularization: dict[str, Array]

    def tree_flatten(self):
        children = (
            self.eta,
            self.eta_weights,
            self.x,
            self.jac,
            self.x_weights,
            self.ddx,
            self.d2dx2,
            tuple(sorted(self.regularization.items())),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        eta, eta_weights, x, jac, x_weights, ddx, d2dx2, reg_items = children
        return cls(
            eta=eta,
            eta_weights=eta_weights,
            x=x,
            jac=jac,
            x_weights=x_weights,
            ddx=ddx,
            d2dx2=d2dx2,
            regularization=dict(reg_items),
        )


def make_reference_eta_grid(n: int, *, kind: str = "gauss") -> tuple[Array, Array]:
    """Return fixed reference nodes and weights on ``[0, 1]``.

    ``kind="gauss"`` uses Gauss-Legendre nodes. ``kind="uniform"`` uses cell
    centers with midpoint weights, which is useful for debugging cell-width maps.
    """

    n_int = int(n)
    if n_int < 2:
        raise ValueError("n must be at least 2")
    key = kind.strip().lower()
    if key in {"gauss", "legendre", "gauss-legendre"}:
        nodes, weights = np.polynomial.legendre.leggauss(n_int)
        eta = 0.5 * (nodes + 1.0)
        eta_weights = 0.5 * weights
    elif key in {"uniform", "midpoint", "cell"}:
        eta = (np.arange(n_int, dtype=np.float64) + 0.5) / float(n_int)
        eta_weights = np.full((n_int,), 1.0 / float(n_int), dtype=np.float64)
    else:
        raise ValueError("kind must be one of {'gauss', 'uniform'}")
    return jnp.asarray(eta, dtype=jnp.float64), jnp.asarray(eta_weights, dtype=jnp.float64)


def barycentric_diff_matrix(x: Array) -> Array:
    """Return the first derivative collocation matrix for arbitrary nodes."""

    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("x must contain at least two nodes")
    dx = x[:, None] - x[None, :]
    safe_dx = dx + jnp.eye(n, dtype=x.dtype)
    weights = 1.0 / jnp.prod(safe_dx, axis=1)
    ratio = weights[None, :] / weights[:, None]
    off_diag = ratio / safe_dx
    off_diag = off_diag * (1.0 - jnp.eye(n, dtype=x.dtype))
    diag = -jnp.sum(off_diag, axis=1)
    return off_diag + jnp.diag(diag)


def barycentric_diff_matrices(x: Array) -> tuple[Array, Array]:
    """Return first and second derivative matrices on nodes ``x``."""

    ddx = barycentric_diff_matrix(x)
    return ddx, ddx @ ddx


def chain_rule_diff_matrices(eta: Array, jac: Array, jac_eta: Array) -> tuple[Array, Array]:
    """Return mapped derivative matrices using ``d/dx = J^{-1} d/deta``."""

    eta = jnp.asarray(eta, dtype=jnp.float64)
    jac = jnp.asarray(jac, dtype=jnp.float64)
    jac_eta = jnp.asarray(jac_eta, dtype=jnp.float64)
    d_eta, d2_eta = barycentric_diff_matrices(eta)
    inv_j = 1.0 / jac
    ddx = jnp.diag(inv_j) @ d_eta
    d2dx2 = jnp.diag(inv_j * inv_j) @ d2_eta - jnp.diag(jac_eta * inv_j**3) @ d_eta
    return ddx, d2dx2


def mapped_grid_regularization(x: Array, jac: Array) -> dict[str, Array]:
    """Return smoothness and conditioning diagnostics for a mapped speed grid."""

    x = jnp.asarray(x, dtype=jnp.float64)
    jac = jnp.asarray(jac, dtype=jnp.float64)
    dx = jnp.diff(x)
    min_dx = jnp.min(dx)
    max_dx = jnp.max(dx)
    width_ratio = max_dx / jnp.maximum(min_dx, jnp.asarray(1.0e-300, dtype=x.dtype))
    log_dx = jnp.log(jnp.maximum(dx, jnp.asarray(1.0e-300, dtype=x.dtype)))
    if int(dx.shape[0]) >= 3:
        smoothness = jnp.sum(jnp.diff(log_dx, n=2) ** 2)
    else:
        smoothness = jnp.asarray(0.0, dtype=x.dtype)
    log_jac = jnp.log(jnp.maximum(jac, jnp.asarray(1.0e-300, dtype=x.dtype)))
    jac_roughness = jnp.sum(jnp.diff(log_jac) ** 2)
    tail_mass_proxy = jnp.exp(-(jnp.max(x) ** 2))
    return {
        "min_dx": min_dx,
        "width_ratio": width_ratio,
        "smoothness": smoothness,
        "jac_roughness": jac_roughness,
        "tail_mass_proxy": tail_mass_proxy,
    }


def _build_grid_from_x_jac(
    *,
    eta: Array,
    eta_weights: Array,
    x: Array,
    jac: Array,
    derivative: str,
    jac_eta: Array | None = None,
) -> MappedXGrid:
    derivative_key = derivative.strip().lower()
    if derivative_key == "barycentric":
        ddx, d2dx2 = barycentric_diff_matrices(x)
    elif derivative_key in {"chain", "chain-rule", "chain_rule"}:
        if jac_eta is None:
            d_eta = barycentric_diff_matrix(eta)
            jac_eta = d_eta @ jac
        ddx, d2dx2 = chain_rule_diff_matrices(eta, jac, jac_eta)
    else:
        raise ValueError("derivative must be 'barycentric' or 'chain-rule'")
    x_weights = eta_weights * jac
    return MappedXGrid(
        eta=eta,
        eta_weights=eta_weights,
        x=x,
        jac=jac,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        regularization=mapped_grid_regularization(x, jac),
    )


@dataclass(frozen=True)
class AffineXMap:
    """Finite-interval identity-style map ``x = x0 + scale * eta``."""

    x0: float = 0.0
    derivative: str = "barycentric"

    def __call__(
        self,
        log_scale: Array | float,
        *,
        eta: Array,
        eta_weights: Array,
    ) -> MappedXGrid:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta_weights = jnp.asarray(eta_weights, dtype=jnp.float64)
        scale = jnp.exp(jnp.asarray(log_scale, dtype=jnp.float64))
        x = jnp.asarray(self.x0, dtype=jnp.float64) + scale * eta
        jac = jnp.broadcast_to(scale, eta.shape)
        jac_eta = jnp.zeros_like(jac)
        return _build_grid_from_x_jac(
            eta=eta,
            eta_weights=eta_weights,
            x=x,
            jac=jac,
            jac_eta=jac_eta,
            derivative=self.derivative,
        )


@dataclass(frozen=True)
class RationalTailXMap:
    """Semi-infinite monotone map ``x = x0 + L eta / (1 - eta + eps)``."""

    x0: float = 0.0
    eps: float = 1.0e-6
    derivative: str = "barycentric"

    def __call__(
        self,
        log_length: Array | float,
        *,
        eta: Array,
        eta_weights: Array,
    ) -> MappedXGrid:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta_weights = jnp.asarray(eta_weights, dtype=jnp.float64)
        length = jnp.exp(jnp.asarray(log_length, dtype=jnp.float64))
        denom = 1.0 - eta + jnp.asarray(self.eps, dtype=jnp.float64)
        x = jnp.asarray(self.x0, dtype=jnp.float64) + length * eta / denom
        jac = length * (1.0 + jnp.asarray(self.eps, dtype=jnp.float64)) / (denom * denom)
        jac_eta = 2.0 * length * (1.0 + jnp.asarray(self.eps, dtype=jnp.float64)) / (denom**3)
        return _build_grid_from_x_jac(
            eta=eta,
            eta_weights=eta_weights,
            x=x,
            jac=jac,
            jac_eta=jac_eta,
            derivative=self.derivative,
        )


@dataclass(frozen=True)
class SoftplusCellXMap:
    """Positive-width cell-center speed grid with optional fixed total extent."""

    x_min: float = 0.0
    delta_min: float = 1.0e-8
    x_max: float | None = None
    derivative: str = "barycentric"

    def __call__(
        self,
        params: Array,
        *,
        eta: Array,
        eta_weights: Array,
    ) -> MappedXGrid:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta_weights = jnp.asarray(eta_weights, dtype=jnp.float64)
        params = jnp.asarray(params, dtype=jnp.float64)
        if params.ndim != 1:
            raise ValueError("params must be one-dimensional")
        if int(params.shape[0]) != int(eta.shape[0]):
            raise ValueError("params must have the same length as eta")
        widths = jnp.asarray(self.delta_min, dtype=jnp.float64) + jax.nn.softplus(params)
        if self.x_max is not None:
            total = jnp.asarray(self.x_max - self.x_min, dtype=jnp.float64)
            widths = widths * (total / jnp.sum(widths))
        left_edges = jnp.asarray(self.x_min, dtype=jnp.float64) + jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(widths[:-1])]
        )
        x = left_edges + 0.5 * widths
        # Convert cell widths into a positive nodal Jacobian estimate.
        jac = widths / eta_weights
        return _build_grid_from_x_jac(
            eta=eta,
            eta_weights=eta_weights,
            x=x,
            jac=jac,
            derivative=self.derivative,
        )


@dataclass(frozen=True)
class SplineDensityXMap:
    """Smooth monotone map from a positive polynomial density on reference nodes."""

    x0: float = 0.0
    density_floor: float = 1.0e-8
    derivative: str = "barycentric"

    def __call__(
        self,
        coeffs: Array,
        *,
        eta: Array,
        eta_weights: Array,
    ) -> MappedXGrid:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta_weights = jnp.asarray(eta_weights, dtype=jnp.float64)
        coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        if coeffs.ndim != 1:
            raise ValueError("coeffs must be one-dimensional")
        powers = jnp.arange(int(coeffs.shape[0]), dtype=jnp.float64)
        basis = eta[:, None] ** powers[None, :]
        raw_density = basis @ coeffs
        jac = jnp.asarray(self.density_floor, dtype=jnp.float64) + jax.nn.softplus(raw_density)

        if int(eta.shape[0]) < 2:
            raise ValueError("eta must have at least two nodes")
        increments = 0.5 * (jac[1:] + jac[:-1]) * jnp.diff(eta)
        x = jnp.asarray(self.x0, dtype=jnp.float64) + jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float64), jnp.cumsum(increments)]
        )
        d_eta = barycentric_diff_matrix(eta)
        jac_eta = d_eta @ jac
        return _build_grid_from_x_jac(
            eta=eta,
            eta_weights=eta_weights,
            x=x,
            jac=jac,
            jac_eta=jac_eta,
            derivative=self.derivative,
        )


def maxwellian_tail_integral_proxy(grid: MappedXGrid, *, power: float = 2.0) -> Array:
    """Return a simple mapped quadrature proxy for Maxwellian-weighted tails."""

    p = jnp.asarray(power, dtype=jnp.float64)
    return jnp.sum(grid.x_weights * (grid.x**p) * jnp.exp(-(grid.x**2)))
