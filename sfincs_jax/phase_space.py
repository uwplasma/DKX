"""Grids and discretization machinery for the SFINCS drift-kinetic equation.

This module consolidates every grid/quadrature/differentiation primitive that a
sfincs_jax solve needs, mirroring the grid-construction part of the Fortran v3
code base:

* ``createGrids.F90`` — overall grid assembly, ``Nxi_for_x`` ramps, and the
  Rosenbluth-potential grid sizing.
* ``uniformDiffMatrices.F90`` — uniform periodic grids and finite-difference /
  spectral-collocation differentiation matrices for theta and zeta.
* ``xGrid.F90`` — Gaussian quadrature nodes/weights for the weight
  ``exp(-x^2) * x^k`` on ``[0, inf)`` (Stieltjes procedure + Golub-Welsch),
  i.e. the speed grid of Landreman & Ernst, J. Comput. Phys. 243, 130 (2013).
* ``polynomialDiffMatrices.F90`` — collocation differentiation matrices on the
  (nonuniform) speed grid.
* ``polynomialInterpolationMatrix.F90`` — barycentric interpolation between
  species-specific speed grids.

The pitch (xi) coordinate is discretized with a Legendre-polynomial modal
expansion; the mode-coupling coefficients ``l/(2l-1)`` and ``(l+1)/(2l+3)`` and
the Lorentz pitch-angle-scattering eigenvalues ``l(l+1)`` used throughout
``populateMatrix.F90`` are provided here as the single source of truth.

This canonical module replaces ``sfincs_jax/grids.py``, ``sfincs_jax/discretization/xgrid.py``, and the grid
construction half of ``sfincs_jax/discretization/v3.py`` at the purge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

__all__ = [
    "Grids",
    "SpeedGrid",
    "chebyshev_grid",
    "legendre_coupling_lower",
    "legendre_coupling_upper",
    "lorentz_eigenvalues",
    "make_grids",
    "make_speed_grid",
    "n_xi_for_x_ramp",
    "polynomial_interpolation_matrix",
    "rosenbluth_potential_grid_size",
    "speed_grid_diff_matrices",
    "speed_weight",
    "uniform_aperiodic_diff_matrices",
    "uniform_periodic_diff_matrices",
    "xdot_diff_matrices",
]


# ----------------------------------------------------------------------------
# 1. Uniform periodic grids for theta and zeta (uniformDiffMatrices.F90)
# ----------------------------------------------------------------------------

#: Internal uniformDiffMatrices scheme numbers with at least one production
#: call site. All are periodic grids that include x_min but not x_max.
_PERIODIC_SCHEMES = frozenset({0, 10, 20, 80, 90, 100, 110, 120, 130})

#: thetaDerivativeScheme / zetaDerivativeScheme (namelist) -> internal scheme.
#: 0 = spectral collocation, 1 = 2nd-order centered FD, 2 = 4th-order centered FD.
_ANGLE_DERIVATIVE_SCHEME_MAP = {0: 20, 1: 0, 2: 10}

#: magneticDriftDerivativeScheme (namelist) -> (plus, minus) upwinded schemes.
_MAGNETIC_DRIFT_SCHEME_MAP = {
    1: (80, 90),
    2: (100, 110),
    3: (120, 130),
    -1: (90, 80),
    -2: (110, 100),
    -3: (130, 120),
}


def uniform_periodic_diff_matrices(
    *,
    n: int,
    x_min: float,
    x_max: float,
    scheme: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform periodic grid, quadrature weights, and differentiation matrices.

    Port of SFINCS v3 ``uniformDiffMatrices.F90`` restricted to the periodic
    schemes actually used by ``createGrids.F90`` for the theta and zeta
    coordinates. The grid includes ``x_min`` but not ``x_max`` and the
    quadrature weight of every node is the uniform spacing ``dx``.

    Args:
      n: Number of grid points (>= 2; some schemes require more).
      x_min: Left end of the periodic interval (included in the grid).
      x_max: Right end of the periodic interval (excluded from the grid).
      scheme: Differentiation scheme, one of:
        ``0`` 3-point centered FD (2nd order); ``10`` 5-point centered FD
        (4th order); ``20`` Fourier spectral collocation; ``80``/``90``
        left/right-biased 4-point upwind (2nd order); ``100``/``110``
        left/right-biased 5-point upwind (3rd order, first derivative only);
        ``120``/``130`` left/right-biased 6-point upwind (4th order, first
        derivative only). The upwinded pairs implement
        ``magneticDriftDerivativeScheme`` 1/2/3.

    Returns:
      Tuple ``(x, weights, ddx, d2dx2)`` of JAX float64 arrays with shapes
      ``(n,)``, ``(n,)``, ``(n, n)``, ``(n, n)``. For schemes 100-130 only the
      first-derivative matrix is populated (``d2dx2`` is zero), matching the
      Fortran code.
    """
    if scheme not in _PERIODIC_SCHEMES:
        raise ValueError(
            f"Invalid or unsupported scheme: {scheme} (supported: {sorted(_PERIODIC_SCHEMES)})"
        )
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if x_min >= x_max:
        raise ValueError(f"x_max must be > x_min, got x_min={x_min}, x_max={x_max}")

    x_min = float(x_min)
    x_max = float(x_max)
    dtype = np.float64

    x = x_min + (x_max - x_min) * (np.arange(n, dtype=dtype) / n)
    dx = float(x[1] - x[0])
    dx2 = dx * dx
    weights = np.full((n,), dx, dtype=dtype)

    ddx = np.zeros((n, n), dtype=dtype)
    d2dx2 = np.zeros((n, n), dtype=dtype)

    if scheme == 0:
        # 3-point centered stencil, periodic wrap at the endpoints.
        if n < 3:
            raise ValueError("n must be at least 3 for 3-point stencil schemes")
        for i in range(1, n - 1):
            ddx[i, i + 1] = 1.0 / (2 * dx)
            ddx[i, i - 1] = -1.0 / (2 * dx)
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2
        ddx[0, -1] = -1.0 / (2 * dx)
        ddx[0, 1] = 1.0 / (2 * dx)
        ddx[-1, 0] = 1.0 / (2 * dx)
        ddx[-1, -2] = -1.0 / (2 * dx)
        d2dx2[0, 0] = -2.0 / dx2
        d2dx2[-1, -1] = -2.0 / dx2
        d2dx2[0, -1] = 1.0 / dx2
        d2dx2[0, 1] = 1.0 / dx2
        d2dx2[-1, 0] = 1.0 / dx2
        d2dx2[-1, -2] = 1.0 / dx2

    elif scheme == 10:
        # 5-point centered stencil, periodic wrap at the two rows on each end.
        if n < 5:
            raise ValueError("n must be at least 5 for 5-point stencil schemes")
        for i in range(2, n - 2):
            ddx[i, i + 2] = -1.0 / (12 * dx)
            ddx[i, i + 1] = 2.0 / (3 * dx)
            ddx[i, i - 1] = -2.0 / (3 * dx)
            ddx[i, i - 2] = 1.0 / (12 * dx)
            d2dx2[i, i + 2] = -1.0 / (12 * dx2)
            d2dx2[i, i + 1] = 4.0 / (3 * dx2)
            d2dx2[i, i] = -5.0 / (2 * dx2)
            d2dx2[i, i - 1] = 4.0 / (3 * dx2)
            d2dx2[i, i - 2] = -1.0 / (12 * dx2)
        # Wrap-around terms:
        ddx[0, -1] = -(4.0 / 3) / (2 * dx)
        ddx[0, -2] = (1.0 / 6) / (2 * dx)
        ddx[1, -1] = (1.0 / 6) / (2 * dx)
        ddx[-1, 0] = (4.0 / 3) / (2 * dx)
        ddx[-1, 1] = -(1.0 / 6) / (2 * dx)
        ddx[-2, 0] = -(1.0 / 6) / (2 * dx)
        d2dx2[0, -1] = (4.0 / 3) / dx2
        d2dx2[0, -2] = -(1.0 / 12) / dx2
        d2dx2[1, -1] = -(1.0 / 12) / dx2
        d2dx2[-1, 0] = (4.0 / 3) / dx2
        d2dx2[-1, 1] = -(1.0 / 12) / dx2
        d2dx2[-2, 0] = -(1.0 / 12) / dx2
        # Remaining interior parts of the first/last two rows:
        ddx[0, 2] = -1.0 / (12 * dx)
        ddx[0, 1] = 2.0 / (3 * dx)
        d2dx2[0, 2] = -1.0 / (12 * dx2)
        d2dx2[0, 1] = 4.0 / (3 * dx2)
        d2dx2[0, 0] = -5.0 / (2 * dx2)
        ddx[1, 3] = -1.0 / (12 * dx)
        ddx[1, 2] = 2.0 / (3 * dx)
        ddx[1, 0] = -2.0 / (3 * dx)
        d2dx2[1, 3] = -1.0 / (12 * dx2)
        d2dx2[1, 2] = 4.0 / (3 * dx2)
        d2dx2[1, 1] = -5.0 / (2 * dx2)
        d2dx2[1, 0] = 4.0 / (3 * dx2)
        ddx[-1, -2] = -2.0 / (3 * dx)
        ddx[-1, -3] = 1.0 / (12 * dx)
        d2dx2[-1, -1] = -5.0 / (2 * dx2)
        d2dx2[-1, -2] = 4.0 / (3 * dx2)
        d2dx2[-1, -3] = -1.0 / (12 * dx2)
        ddx[-2, -1] = 2.0 / (3 * dx)
        ddx[-2, -3] = -2.0 / (3 * dx)
        ddx[-2, -4] = 1.0 / (12 * dx)
        d2dx2[-2, -1] = 4.0 / (3 * dx2)
        d2dx2[-2, -2] = -5.0 / (2 * dx2)
        d2dx2[-2, -3] = 4.0 / (3 * dx2)
        d2dx2[-2, -4] = -1.0 / (12 * dx2)

    elif scheme == 20:
        # Fourier spectral collocation (Trefethen, Spectral Methods in MATLAB).
        pi = math.pi
        h = 2 * pi / n
        n1 = int(math.floor((n - 1.0) / 2))
        n2 = int(math.ceil((n - 1.0) / 2))

        col1 = np.zeros((n,), dtype=dtype)
        if n % 2 == 0:
            topc = np.array([0.5 / math.tan(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype)
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = -topc[n1 - 1 :: -1]
        else:
            topc = np.array([0.5 / math.sin(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype)
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = topc[n1 - 1 :: -1]
        col1[1::2] *= -1
        col1 *= 2 * pi / (x_max - x_min)
        for i in range(n):
            ddx[i, i:] = -col1[: n - i]
            ddx[i, :i] = col1[i:0:-1]

        col1 = np.zeros((n,), dtype=dtype)
        if n % 2 == 0:
            col1[0] = -pi * pi / (3 * h * h) - 1.0 / 6
            topc = np.array(
                [-(0.5) / (math.sin(i * h / 2) ** 2) for i in range(1, n2 + 1)], dtype=dtype
            )
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = topc[n1 - 1 :: -1]
        else:
            col1[0] = -pi * pi / (3 * h * h) + 1.0 / 12
            topc = np.array(
                [-(0.5) / (math.sin(i * h / 2) * math.tan(i * h / 2)) for i in range(1, n2 + 1)],
                dtype=dtype,
            )
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = -topc[n1 - 1 :: -1]
        col1[1::2] *= -1
        col1 *= (2 * pi / (x_max - x_min)) ** 2
        for i in range(n):
            d2dx2[i, i:] = col1[: n - i]
            d2dx2[i, :i] = col1[i:0:-1]

    elif scheme in {80, 90}:
        if n < 5:
            raise ValueError("n must be at least 5 for 4 point stencil schemes")
        sign = 1.0 if scheme == 80 else -1.0
        for i in range(n):
            ddx[i, (i + int(sign)) % n] = sign / (3 * dx)
            ddx[i, i] = sign / (2 * dx)
            ddx[i, (i - int(sign)) % n] = -sign / dx
            ddx[i, (i - 2 * int(sign)) % n] = sign / (6 * dx)
            d2dx2[i, (i + 1) % n] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, (i - 1) % n] = 1.0 / dx2

    elif scheme in {100, 110}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 100, 110")
        sign = 1.0 if scheme == 100 else -1.0
        for i in range(n):
            ddx[i, (i + int(sign)) % n] = sign / (4 * dx)
            ddx[i, i] = sign * 5.0 / (6 * dx)
            ddx[i, (i - int(sign)) % n] = -sign * 3.0 / (2 * dx)
            ddx[i, (i - 2 * int(sign)) % n] = sign / (2 * dx)
            ddx[i, (i - 3 * int(sign)) % n] = -sign / (12 * dx)

    elif scheme in {120, 130}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 120, 130")
        sign = 1.0 if scheme == 120 else -1.0
        for i in range(n):
            ddx[i, (i + 2 * int(sign)) % n] = -sign / (20 * dx)
            ddx[i, (i + int(sign)) % n] = sign / (2 * dx)
            ddx[i, i] = sign / (3 * dx)
            ddx[i, (i - int(sign)) % n] = -sign / dx
            ddx[i, (i - 2 * int(sign)) % n] = sign / (4 * dx)
            ddx[i, (i - 3 * int(sign)) % n] = -sign / (30 * dx)

    return jnp.asarray(x), jnp.asarray(weights), jnp.asarray(ddx), jnp.asarray(d2dx2)


#: Internal uniformDiffMatrices scheme numbers for aperiodic grids with a node
#: at both x_min and x_max, as used by the speed (x) coordinate.
_APERIODIC_SCHEMES = frozenset({12, 32, 42, 52, 62, 82, 92, 102, 112})


def uniform_aperiodic_diff_matrices(
    *,
    n: int,
    x_min: float,
    x_max: float,
    scheme: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform aperiodic grid, trapezoid weights, and differentiation matrices.

    Port of SFINCS v3 ``uniformDiffMatrices.F90`` restricted to the aperiodic
    schemes used by ``createGrids.F90`` for the speed (x) coordinate:
    xGridScheme 3/4 (scheme 12) and the upwinded ``xDotDerivativeScheme``
    pairs. The grid includes both ``x_min`` and ``x_max`` and the trapezoid
    quadrature weight is ``dx`` (halved at the two endpoints).

    Args:
      n: Number of grid points.
      x_min: Left end of the interval (included in the grid).
      x_max: Right end of the interval (included in the grid).
      scheme: Differentiation scheme, one of:
        ``12`` 5-point centered FD (4th order, one-sided rows at the ends);
        ``32``/``42`` left/right 2-point upwind (top/bottom ddx row zero);
        ``52``/``62`` left/right 3-point upwind (2nd order);
        ``82``/``92`` left/right 4-point upwind (1 point on one side);
        ``102``/``112`` left/right 5-point upwind (1 point on one side,
        first derivative only).

    Returns:
      Tuple ``(x, weights, ddx, d2dx2)`` of float64 numpy arrays with shapes
      ``(n,)``, ``(n,)``, ``(n, n)``, ``(n, n)``. For schemes 102/112 only
      the first-derivative matrix is populated, matching the Fortran code.
    """
    if scheme not in _APERIODIC_SCHEMES:
        raise ValueError(
            f"Invalid or unsupported scheme: {scheme} (supported: {sorted(_APERIODIC_SCHEMES)})"
        )
    if x_min >= x_max:
        raise ValueError(f"x_max must be > x_min, got x_min={x_min}, x_max={x_max}")
    min_n = 3 if scheme in {32, 42, 52, 62} else 5
    if n < min_n:
        raise ValueError(f"n must be at least {min_n} for scheme {scheme}, got {n}")

    dtype = np.float64
    x = float(x_min) + (float(x_max) - float(x_min)) * (np.arange(n, dtype=dtype) / (n - 1))
    dx = float(x[1] - x[0])
    dx2 = dx * dx
    weights = np.full((n,), dx, dtype=dtype)
    weights[0] /= 2.0
    weights[-1] /= 2.0

    ddx = np.zeros((n, n), dtype=dtype)
    d2dx2 = np.zeros((n, n), dtype=dtype)

    if scheme == 12:
        # 5-point centered stencil; one-sided 5-point rows at the two ends.
        for i in range(2, n - 2):
            ddx[i, i + 2] = -1.0 / (6 * 2 * dx)
            ddx[i, i + 1] = 4.0 / (3 * 2 * dx)
            ddx[i, i - 1] = -4.0 / (3 * 2 * dx)
            ddx[i, i - 2] = 1.0 / (6 * 2 * dx)
            d2dx2[i, i + 2] = -1.0 / (12 * dx2)
            d2dx2[i, i + 1] = 4.0 / (3 * dx2)
            d2dx2[i, i] = -5.0 / (2 * dx2)
            d2dx2[i, i - 1] = 4.0 / (3 * dx2)
            d2dx2[i, i - 2] = -1.0 / (12 * dx2)
        ddx[0, 0] = -25.0 / (12 * dx)
        ddx[0, 1] = 4.0 / dx
        ddx[0, 2] = -3.0 / dx
        ddx[0, 3] = 4.0 / (3 * dx)
        ddx[0, 4] = -1.0 / (4 * dx)
        ddx[1, 0] = -1.0 / (4 * dx)
        ddx[1, 1] = -5.0 / (6 * dx)
        ddx[1, 2] = 3.0 / (2 * dx)
        ddx[1, 3] = -1.0 / (2 * dx)
        ddx[1, 4] = 1.0 / (12 * dx)
        ddx[-1, -1] = 25.0 / (12 * dx)
        ddx[-1, -2] = -4.0 / dx
        ddx[-1, -3] = 3.0 / dx
        ddx[-1, -4] = -4.0 / (3 * dx)
        ddx[-1, -5] = 1.0 / (4 * dx)
        ddx[-2, -1] = 1.0 / (4 * dx)
        ddx[-2, -2] = 5.0 / (6 * dx)
        ddx[-2, -3] = -3.0 / (2 * dx)
        ddx[-2, -4] = 1.0 / (2 * dx)
        ddx[-2, -5] = -1.0 / (12 * dx)
        d2dx2[0, 0] = 35.0 / (12 * dx2)
        d2dx2[0, 1] = -26.0 / (3 * dx2)
        d2dx2[0, 2] = 19.0 / (2 * dx2)
        d2dx2[0, 3] = -14.0 / (3 * dx2)
        d2dx2[0, 4] = 11.0 / (12 * dx2)
        d2dx2[1, 0] = 11.0 / (12 * dx2)
        d2dx2[1, 1] = -5.0 / (3 * dx2)
        d2dx2[1, 2] = 1.0 / (2 * dx2)
        d2dx2[1, 3] = 1.0 / (3 * dx2)
        d2dx2[1, 4] = -1.0 / (12 * dx2)
        d2dx2[-1, -1] = 35.0 / (12 * dx2)
        d2dx2[-1, -2] = -26.0 / (3 * dx2)
        d2dx2[-1, -3] = 19.0 / (2 * dx2)
        d2dx2[-1, -4] = -14.0 / (3 * dx2)
        d2dx2[-1, -5] = 11.0 / (12 * dx2)
        d2dx2[-2, -1] = 11.0 / (12 * dx2)
        d2dx2[-2, -2] = -5.0 / (3 * dx2)
        d2dx2[-2, -3] = 1.0 / (2 * dx2)
        d2dx2[-2, -4] = 1.0 / (3 * dx2)
        d2dx2[-2, -5] = -1.0 / (12 * dx2)

    elif scheme == 32:
        # 2-point ddx / 3-point d2dx2, upwinding left; top ddx row zero.
        for i in range(1, n):
            ddx[i, i] = 1.0 / dx
            ddx[i, i - 1] = -1.0 / dx
        for i in range(2, n):
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i - 1] = -2.0 / dx2
            d2dx2[i, i - 2] = 1.0 / dx2

    elif scheme == 42:
        # 2-point ddx / 3-point d2dx2, upwinding right; bottom ddx row zero.
        for i in range(n - 1):
            ddx[i, i] = -1.0 / dx
            ddx[i, i + 1] = 1.0 / dx
        for i in range(n - 2):
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i + 1] = -2.0 / dx2
            d2dx2[i, i + 2] = 1.0 / dx2

    elif scheme == 52:
        # 3-point stencils, upwinding left; 2-point second ddx row.
        for i in range(2, n):
            ddx[i, i] = 1.5 / dx
            ddx[i, i - 1] = -2.0 / dx
            ddx[i, i - 2] = 1.0 / (2 * dx)
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i - 1] = -2.0 / dx2
            d2dx2[i, i - 2] = 1.0 / dx2
        ddx[1, 0] = -1.0 / dx
        ddx[1, 1] = 1.0 / dx

    elif scheme == 62:
        # 3-point stencils, upwinding right; 2-point penultimate ddx row.
        for i in range(n - 2):
            ddx[i, i] = -1.5 / dx
            ddx[i, i + 1] = 2.0 / dx
            ddx[i, i + 2] = -1.0 / (2 * dx)
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i + 1] = -2.0 / dx2
            d2dx2[i, i + 2] = 1.0 / dx2
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx

    elif scheme == 82:
        # 4-point ddx (1 right + 2 left), upwinding left; top ddx row zero.
        for i in range(2, n - 1):
            ddx[i, i + 1] = 1.0 / (3 * dx)
            ddx[i, i] = 1.0 / (2 * dx)
            ddx[i, i - 1] = -1.0 / dx
            ddx[i, i - 2] = 1.0 / (6 * dx)
        for i in range(1, n - 1):
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2
        ddx[1, 1] = 1.0 / dx
        ddx[1, 0] = -1.0 / dx
        ddx[-1, -1] = 1.5 / dx
        ddx[-1, -2] = -2.0 / dx
        ddx[-1, -3] = 1.0 / (2 * dx)

    elif scheme == 92:
        # 4-point ddx (1 left + 2 right), upwinding right; bottom ddx row zero.
        for i in range(1, n - 2):
            ddx[i, i - 1] = -1.0 / (3 * dx)
            ddx[i, i] = -1.0 / (2 * dx)
            ddx[i, i + 1] = 1.0 / dx
            ddx[i, i + 2] = -1.0 / (6 * dx)
        for i in range(1, n - 1):
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -1.0 / (2 * dx)

    elif scheme == 102:
        # 5-point ddx (1 right + 3 left), upwinding left; ddx only.
        for i in range(3, n - 1):
            ddx[i, i + 1] = 1.0 / (4 * dx)
            ddx[i, i] = 5.0 / (6 * dx)
            ddx[i, i - 1] = -3.0 / (2 * dx)
            ddx[i, i - 2] = 1.0 / (2 * dx)
            ddx[i, i - 3] = -1.0 / (12 * dx)
        ddx[1, 1] = 1.0 / dx
        ddx[1, 0] = -1.0 / dx
        ddx[2, 3] = 1.0 / (3 * dx)
        ddx[2, 2] = 1.0 / (2 * dx)
        ddx[2, 1] = -1.0 / dx
        ddx[2, 0] = 1.0 / (6 * dx)
        ddx[-1, -1] = 5.0 / (6 * dx)
        ddx[-1, -2] = -3.0 / (2 * dx)
        ddx[-1, -3] = 1.0 / (2 * dx)
        ddx[-1, -4] = -1.0 / (12 * dx)

    elif scheme == 112:
        # 5-point ddx (1 left + 3 right), upwinding right; ddx only.
        for i in range(1, n - 3):
            ddx[i, i - 1] = -1.0 / (4 * dx)
            ddx[i, i] = -5.0 / (6 * dx)
            ddx[i, i + 1] = 3.0 / (2 * dx)
            ddx[i, i + 2] = -1.0 / (2 * dx)
            ddx[i, i + 3] = 1.0 / (12 * dx)
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -1.0 / (2 * dx)

    return x, weights, ddx, d2dx2


def chebyshev_grid(
    *, n: int, x_min: float, x_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev grid, Clenshaw-Curtis weights, and differentiation matrix.

    Port of v3 ``ChebyshevGrid.F90`` (itself based on Trefethen's ``cheb.m``),
    used by ``createGrids.F90`` for xGridScheme 7/8. The nodes run from
    ``x_min`` to ``x_max`` inclusive, clustered at both ends.

    Args:
      n: Number of grid points (>= 2).
      x_min: Minimum of the interval.
      x_max: Maximum of the interval.

    Returns:
      Tuple ``(x, weights, ddx)`` of float64 numpy arrays with shapes
      ``(n,)``, ``(n,)``, ``(n, n)``.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be > x_min, got x_min={x_min}, x_max={x_max}")
    dtype = np.float64
    pi = 3.1415926535897932384626433

    n1 = n - 1
    xc = np.cos(pi * np.arange(n, dtype=dtype) / n1)

    c = np.ones((n,), dtype=dtype)
    c[0] = 2.0
    c[-1] = 2.0
    c[1::2] *= -1.0

    dxm = xc[:, None] - xc[None, :]
    np.fill_diagonal(dxm, 1.0)
    d = (c[:, None] / c[None, :]) / dxm
    np.fill_diagonal(d, np.diag(d) - np.sum(d, axis=1))
    d = -d * 2.0 / (float(x_max) - float(x_min))

    x = (1.0 - xc) * (float(x_max) - float(x_min)) / 2.0 + float(x_min)

    # Clenshaw-Curtis weights via the real inverse DFT of the cosine moments.
    cw = np.zeros((n,), dtype=dtype)
    cw[0] = 2.0
    for i in range(2, n, 2):  # 1-based odd indices 3, 5, ...
        cw[i] = 2.0 / (1.0 - i * i)
    m = 2 * n - 2
    cc = np.zeros((m,), dtype=dtype)
    cc[:n] = cw
    cc[n:] = cw[n - 2 : 0 : -1]
    idx = np.arange(m, dtype=dtype)
    f = (np.cos(2.0 * pi * np.outer(idx, idx) / m) @ cc) / m
    weights = f[:n].copy()
    weights[0] = f[0] / 2.0
    weights[-1] = f[n - 1] / 2.0
    weights *= float(x_max) - float(x_min)

    return x, weights, d


# ----------------------------------------------------------------------------
# 2. Legendre pitch (xi) machinery (createGrids.F90 / populateMatrix.F90)
# ----------------------------------------------------------------------------


def legendre_coupling_lower(n_xi: int) -> np.ndarray:
    """Legendre streaming coupling ``l / (2l - 1)`` from mode l to l-1.

    From the recursion ``xi P_l = (l+1)/(2l+3) P_{l+1} + l/(2l-1) P_{l-1}``
    used by the parallel-streaming and mirror terms in ``populateMatrix.F90``.
    The l=0 entry is zero (no l=-1 mode).

    Args:
      n_xi: Number of Legendre modes (l = 0 .. n_xi - 1).

    Returns:
      Float64 array of shape ``(n_xi,)``.
    """
    ell = np.arange(int(n_xi), dtype=np.float64)
    return np.where(ell > 0, ell / (2.0 * ell - 1.0), 0.0)


def legendre_coupling_upper(n_xi: int) -> np.ndarray:
    """Legendre streaming coupling ``(l + 1) / (2l + 3)`` from mode l to l+1.

    See :func:`legendre_coupling_lower`.

    Args:
      n_xi: Number of Legendre modes (l = 0 .. n_xi - 1).

    Returns:
      Float64 array of shape ``(n_xi,)``.
    """
    ell = np.arange(int(n_xi), dtype=np.float64)
    return (ell + 1.0) / (2.0 * ell + 3.0)


def lorentz_eigenvalues(n_xi: int) -> np.ndarray:
    """Eigenvalues ``l (l + 1)`` of the Lorentz pitch-angle-scattering operator.

    The Legendre polynomials are eigenfunctions of the Lorentz operator
    ``L = (1/2) d/dxi [(1 - xi^2) d/dxi]`` with eigenvalues ``-l(l+1)/2``;
    SFINCS's collision operator (``populateMatrix.F90``) multiplies the
    deflection frequency by ``l(l+1)/2``, and this helper returns the
    ``l(l+1)`` factor.

    Args:
      n_xi: Number of Legendre modes (l = 0 .. n_xi - 1).

    Returns:
      Float64 array of shape ``(n_xi,)``.
    """
    ell = np.arange(int(n_xi), dtype=np.float64)
    return ell * (ell + 1.0)


def n_xi_for_x_ramp(
    *,
    x: np.ndarray,
    n_xi: int,
    n_l: int,
    option: int,
) -> np.ndarray:
    """Number of active Legendre modes retained at each speed-grid point.

    Port of the ``Nxi_for_x_option`` logic in ``createGrids.F90``: low-speed
    points need fewer pitch modes, so the number of retained modes ramps up
    with x. Option 0 keeps all ``n_xi`` modes everywhere.

    Args:
      x: Speed-grid nodes, shape ``(n_x,)``.
      n_xi: Maximum number of Legendre modes (namelist ``Nxi``).
      n_l: Number of Legendre modes in the Rosenbluth potentials (namelist
        ``NL``); acts as a floor.
      option: Ramp option 0, 1, 2, or 3 (namelist ``Nxi_for_x_option``).

    Returns:
      Integer array of shape ``(n_x,)``.
    """
    x_np = np.asarray(x, dtype=float)
    n_x = int(x_np.size)
    n_xi = int(n_xi)
    n_l = int(n_l)
    out = np.zeros((n_x,), dtype=int)
    if option == 0:
        out[:] = n_xi
    elif option == 1:
        for j in range(n_x):
            temp = n_xi * (0.1 + 0.9 * x_np[j] / 2.0)
            out[j] = max(4, n_l, min(int(temp), n_xi))
    elif option == 2:
        for j in range(n_x):
            temp = n_xi * (0.1 + 0.9 * ((x_np[j] / 2.0) ** 2))
            out[j] = max(4, n_l, min(int(temp), n_xi))
    elif option == 3:
        for j in range(n_x):
            temp = n_xi * (0.1 + 0.9 * x_np[j] / 2.0)
            out[j] = max(3, n_l, int(temp))
    else:
        raise ValueError(f"Invalid Nxi_for_x_option={option}")
    return out


# ----------------------------------------------------------------------------
# 3. Speed (x) grid: Landreman-Ernst polynomial collocation (xGrid.F90)
# ----------------------------------------------------------------------------


def speed_weight(x: np.ndarray, k: float) -> np.ndarray:
    """Weight function ``exp(-x^2) * x^k`` of the SFINCS speed grid (xGrid.F90)."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-(x * x)) * (x**k)


def _speed_weight_d1_over_weight(x: np.ndarray, k: float) -> np.ndarray:
    """(d/dx weight) / weight, matching v3 ``polynomialDiffMatrices.F90``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = 0.0
    out[~mask0] = k / x[~mask0] - 2.0 * x[~mask0]
    return out


def _speed_weight_d2_over_weight(x: np.ndarray, k: float) -> np.ndarray:
    """(d^2/dx^2 weight) / weight, matching v3 ``polynomialDiffMatrices.F90``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = -2.0
    out[~mask0] = k * (k - 1.0) / (x[~mask0] * x[~mask0]) - 2.0 * (2.0 * k + 1.0) + 4.0 * (
        x[~mask0] * x[~mask0]
    )
    return out


@dataclass(frozen=True)
class SpeedGrid:
    """Collocation nodes, weights, and recurrence data for the speed grid.

    Produced by :func:`make_speed_grid` (port of ``xGrid.F90:makeXGrid``, the
    grid of Landreman & Ernst, J. Comput. Phys. 243, 130 (2013)).

    Attributes:
      x: Collocation nodes on ``[0, inf)``, shape ``(n_x,)``.
      gaussian_weights: Gaussian quadrature weights for integrals of the form
        ``int_0^inf p(x) exp(-x^2) x^k dx``, shape ``(n_x,)``.
      poly_a: 1-based recurrence coefficients ``a`` (index 0 unused).
      poly_b: 1-based recurrence coefficients ``b`` (index 0 unused).
      poly_c: 1-based polynomial norms ``c`` (index 0 unused).
      k: Exponent in the weight ``exp(-x^2) x^k``.
      include_point_at_x0: Whether the lowest node is pinned at ``x0``
        (Gauss-Radau style; xGridScheme 2/6).
      x0: The pinned abscissa (normally 0).
    """

    x: np.ndarray
    gaussian_weights: np.ndarray
    poly_a: np.ndarray
    poly_b: np.ndarray
    poly_c: np.ndarray
    k: float
    include_point_at_x0: bool
    x0: float

    def dx_weights(self, k: float | None = None) -> np.ndarray:
        """Weights for plain ``dx`` integrals (Fortran divides by the weight)."""
        if k is None:
            k = self.k
        w = np.exp(-(self.x * self.x)) * (self.x**k)
        return self.gaussian_weights / w


def _integrate_split(f, *, finite_bound: float = 10.0) -> float:
    """Integrate f on [0, inf), semi-infinite part first, matching v3 ordering."""
    a2, _ = quad(f, finite_bound, np.inf, epsabs=0.0, epsrel=1e-13, limit=5000)
    a1, _ = quad(f, 0.0, finite_bound, epsabs=0.0, epsrel=1e-13, limit=5000)
    return float(a2 + a1)


def _evaluate_orthogonal_polynomial(x: float, *, j: int, a: np.ndarray, b: np.ndarray) -> float:
    """Evaluate p_j(x) by the 3-term recurrence (xGrid.F90:evaluatePolynomial)."""
    if j == 1:
        return 1.0
    pj_minus1 = 0.0
    pj = 1.0
    y = 0.0
    for ii in range(1, j):
        y = (x - float(a[ii])) * pj - float(b[ii]) * pj_minus1
        pj_minus1, pj = pj, y
    return float(y)


@lru_cache(maxsize=32)
def _make_speed_grid_cached(
    n_x: int, k: float, include_point_at_x0: bool, x0: float, finite_bound: float
) -> SpeedGrid:
    # Stieltjes procedure (Numerical Recipes 3rd ed., sections 4.6.2-4.6.3):
    # build the 3-term recurrence for polynomials orthogonal under the weight
    # exp(-x^2) x^k on [0, inf), then take nodes/weights from the Jacobi
    # matrix eigen-decomposition (Golub-Welsch).
    a = np.zeros((n_x + 1,), dtype=float)
    b = np.zeros((n_x + 2,), dtype=float)
    c = np.zeros((n_x + 1,), dtype=float)
    d = np.zeros((n_x + 1,), dtype=float)

    oldc = 1.0
    last_poly_x0 = 0.0
    penult_poly_x0 = 0.0

    for j in range(1, n_x + 1):

        def p(xx: float) -> float:
            return _evaluate_orthogonal_polynomial(xx, j=j, a=a, b=b)

        def integrand_c(xx: float) -> float:
            pj = p(xx)
            return pj * (math.exp(-(xx * xx)) * xx**k) * pj

        def integrand_d(xx: float) -> float:
            pj = p(xx)
            return xx * pj * (math.exp(-(xx * xx)) * xx**k) * pj

        c[j] = _integrate_split(integrand_c, finite_bound=finite_bound)
        d[j] = _integrate_split(integrand_d, finite_bound=finite_bound)

        b[j] = c[j] / oldc
        a[j] = d[j] / c[j]
        oldc = c[j]

        penult_poly_x0 = last_poly_x0
        last_poly_x0 = p(x0)

    if include_point_at_x0:
        # Gauss-Radau modification: pin the lowest abscissa at x0.
        a[n_x] = x0 - b[n_x] * penult_poly_x0 / last_poly_x0

    diag = a[1 : n_x + 1].copy()
    off = np.sqrt(b[2 : n_x + 1].copy())
    try:
        if include_point_at_x0:
            from scipy.linalg import eigh_tridiagonal

            abscissae, eigenvectors = eigh_tridiagonal(diag, off, lapack_driver="stevr")
        else:
            from scipy.linalg.lapack import dpteqr

            d_out, _e_out, z, info = dpteqr("I", diag.copy(), off.copy())
            if info != 0:
                raise RuntimeError(f"dpteqr failed with info={info}")
            # Fortran reverses order for DPTEQR.
            abscissae = d_out[::-1]
            eigenvectors = z[:, ::-1]
    except Exception:  # noqa: BLE001
        jmat = np.diag(diag)
        for i in range(n_x - 1):
            jmat[i, i + 1] = off[i]
            jmat[i + 1, i] = off[i]
        abscissae, eigenvectors = np.linalg.eigh(jmat)
    weights = c[1] * (eigenvectors[0, :] ** 2)

    if include_point_at_x0:
        # Match the Fortran behavior: force the smallest node to be exactly x0.
        abscissae = abscissae.copy()
        abscissae[0] = x0

    return SpeedGrid(
        x=abscissae,
        gaussian_weights=weights,
        poly_a=a,
        poly_b=b,
        poly_c=c,
        k=float(k),
        include_point_at_x0=bool(include_point_at_x0),
        x0=float(x0),
    )


def make_speed_grid(
    *,
    n_x: int,
    k: float = 0.0,
    include_point_at_x0: bool = False,
    x0: float = 0.0,
    finite_bound: float = 10.0,
) -> SpeedGrid:
    """Collocation nodes and quadrature weights for the SFINCS speed grid.

    Port of ``xGrid.F90:makeXGrid``: Gaussian quadrature for the weight
    ``exp(-x^2) * x^k`` on ``[0, inf)`` following Landreman & Ernst,
    J. Comput. Phys. 243, 130 (2013). This covers namelist ``xGridScheme``
    1/5 (``include_point_at_x0=False``, the default scheme 5) and 2/6
    (``include_point_at_x0=True``). The construction is static and cached;
    it is not part of the differentiable JAX graph.

    Args:
      n_x: Number of speed-grid points (>= 1).
      k: Weight exponent (namelist ``xGrid_k``; must be >= 0).
      include_point_at_x0: Pin the lowest node at ``x0`` (Gauss-Radau).
      x0: Pinned abscissa used when ``include_point_at_x0`` is set.
      finite_bound: Split point of the [0, inf) quadratures (Fortran uses 10).

    Returns:
      A :class:`SpeedGrid`.
    """
    if n_x < 1:
        raise ValueError(f"n_x must be >= 1, got {n_x}")
    if k < 0:
        raise ValueError("k must be >= 0 for the built-in SFINCS weight")
    return _make_speed_grid_cached(
        int(n_x), float(k), bool(include_point_at_x0), float(x0), float(finite_bound)
    )


def speed_grid_diff_matrices(x: np.ndarray, *, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Collocation differentiation matrices on the (nonuniform) speed grid.

    Port of v3 ``polynomialDiffMatrices.F90:makeXPolynomialDiffMatrices``:
    spectral differentiation for functions of the form
    ``f(x) = exp(-x^2) x^k * polynomial(x)`` collocated at the nodes ``x``.

    Args:
      x: Speed-grid nodes, shape ``(n_x,)``.
      k: Weight exponent (namelist ``xGrid_k``).

    Returns:
      Tuple ``(ddx, d2dx2)`` of float64 arrays with shape ``(n_x, n_x)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n < 1:
        raise ValueError("x must have at least one point")

    xx = np.broadcast_to(x[:, None], (n, n)).copy()
    dx = xx - xx.T
    np.fill_diagonal(dx, 1.0)

    c = np.prod(dx, axis=1)
    c = c * speed_weight(x, k)
    ccc = c[:, None] / c[None, :]

    z = 1.0 / dx
    np.fill_diagonal(z, 0.0)

    xxx = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n):
        if i + 1 < n:
            xxx[i:, i] = z[i, i + 1 :]
        if i > 0:
            xxx[:i, i] = z[i, :i]

    y = np.zeros((n, n), dtype=np.float64)
    y[0, :] = _speed_weight_d1_over_weight(x, k)
    for i in range(1, n):
        y[i, :] = y[i - 1, :] + xxx[i - 1, :]

    ddx = z * ccc
    np.fill_diagonal(ddx, y[-1, :])

    old_y = y
    y2 = np.zeros((n, n), dtype=np.float64)
    y2[0, :] = _speed_weight_d2_over_weight(x, k)
    for i in range(1, n):
        y2[i, :] = y2[i - 1, :] + 2.0 * old_y[i - 1, :] * xxx[i - 1, :]

    repmat_diag_ddx = np.broadcast_to(np.diag(ddx)[:, None], (n, n))
    d2dx2 = 2.0 * z * (ccc * repmat_diag_ddx - ddx)
    np.fill_diagonal(d2dx2, y2[-1, :])

    return ddx, d2dx2


def xdot_diff_matrices(
    *,
    x: np.ndarray,
    ddx: np.ndarray,
    k: float,
    scheme: int,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Upwinded ``(ddx_xDot_plus, ddx_xDot_minus)`` pair for the E_r xDot term.

    Port of the ``select case (xDotDerivativeScheme)`` block of
    ``createGrids.F90``: the d/dx matrices used by the collisionless d/dx term
    associated with E_r (``includeXDotTerm``), upwinded by the sign of the
    local ``xDotFactor``. Scheme 0 (both matrices equal to ``ddx``) is handled
    by the caller; schemes 1-10 are only valid for xGridScheme 3/4 (uniform
    speed grids, ``validateInput.F90``), while -2 and 11 work on any grid.
    Scheme -1 is not ported: the Fortran loop ``do i=i,Nx`` reads an undefined
    loop start, so its result is compiler-dependent garbage.

    Args:
      x: Speed-grid nodes, shape ``(n_x,)``.
      ddx: The main d/dx collocation matrix on ``x``.
      k: Weight exponent ``xGrid_k`` (used by the polynomial subsets of
        scheme -2).
      scheme: Namelist ``xDotDerivativeScheme`` (-2..11, excluding -1 and 0).
      x_max: Namelist ``xMax`` (the uniform-grid schemes rebuild matrices on
        ``n_x + 1`` points over ``[0, x_max]`` and truncate, exactly as
        ``createGrids.F90`` does).

    Returns:
      Tuple ``(plus, minus)`` of float64 arrays with shape ``(n_x, n_x)``.
    """
    x = np.asarray(x, dtype=np.float64)
    ddx = np.asarray(ddx, dtype=np.float64)
    n = int(x.size)

    def _uniform_pair(scheme_plus: int, scheme_minus: int) -> tuple[np.ndarray, np.ndarray]:
        _, _, ddx_plus1, _ = uniform_aperiodic_diff_matrices(
            n=n + 1, x_min=0.0, x_max=x_max, scheme=scheme_plus
        )
        _, _, ddx_minus1, _ = uniform_aperiodic_diff_matrices(
            n=n + 1, x_min=0.0, x_max=x_max, scheme=scheme_minus
        )
        return ddx_plus1[:n, :n].copy(), ddx_minus1[:n, :n].copy()

    def _shift_last_row(m: np.ndarray) -> None:
        # createGrids.F90: ddx_xDot_minus(Nx,i) = ddx_xDot_minus(Nx-1,i-1).
        m[n - 1, 1:] = m[n - 2, : n - 1]

    def _exp_similarity(m: np.ndarray) -> np.ndarray:
        # m -> expx2 m / expx2 with -2x added on the diagonal (d/dx acting on
        # f/exp(-x^2), i.e. the derivative of the non-Maxwellian part).
        expx2 = np.exp(-(x * x))
        out = expx2[:, None] * m / expx2[None, :]
        out[np.arange(n), np.arange(n)] -= 2.0 * x
        return out

    if scheme == -2:
        plus = np.zeros((n, n), dtype=np.float64)
        minus = np.zeros((n, n), dtype=np.float64)
        ddx_sub, _ = speed_grid_diff_matrices(x[: n - 1], k=k)
        plus[: n - 1, : n - 1] = ddx_sub
        ddx_sub, _ = speed_grid_diff_matrices(x[1:], k=k)
        minus[1:, 1:] = ddx_sub
        return plus, minus

    if scheme == -1:
        raise NotImplementedError(
            "xDotDerivativeScheme=-1 is not ported: the Fortran loop 'do i=i,Nx' in "
            "createGrids.F90 reads an undefined loop start, so its result is "
            "compiler-dependent."
        )

    if scheme == 1:
        return _uniform_pair(32, 42)

    if scheme in {2, 3}:
        plus, minus = _uniform_pair(52, 62)
        if scheme == 3:
            _shift_last_row(minus)
        return plus, minus

    if scheme in {4, 5, 7}:
        plus, minus = _uniform_pair(82, 92)
        if scheme == 5:
            plus[1, :] = ddx[1, :]
        if scheme in {5, 7}:
            _shift_last_row(minus)
        if scheme == 7:
            plus = _exp_similarity(plus)
            minus = _exp_similarity(minus)
        return plus, minus

    if scheme == 6:
        plus = _exp_similarity(ddx.copy())
        return plus, plus.copy()

    if scheme == 8:
        plus, minus = _uniform_pair(102, 112)
        # Two shifted bottom rows (5-point upwind stencil).
        minus[n - 1, 2:] = minus[n - 3, : n - 2]
        minus[n - 2, 2:] = minus[n - 3, 1 : n - 1]
        return plus, minus

    if scheme in {9, 10}:
        _, _, ddx_sub, _ = uniform_aperiodic_diff_matrices(
            n=n, x_min=0.0, x_max=float(x[-1]), scheme=12
        )
        if scheme == 9:
            return ddx_sub, ddx.copy()
        return ddx.copy(), ddx_sub

    if scheme == 11:
        # Tridiagonal FD matrix for the irregular (polynomial) grid
        # (createGrids.F90, notes 20170118-02).
        plus = np.zeros((n, n), dtype=np.float64)
        plus[0, 0] = -1.0 / (x[1] - x[0])
        plus[0, 1] = 1.0 / (x[1] - x[0])
        plus[n - 1, n - 2] = -1.0 / (x[n - 1] - x[n - 2])
        plus[n - 1, n - 1] = 1.0 / (x[n - 1] - x[n - 2])
        for j in range(1, n - 1):
            d32 = x[j + 1] - x[j]
            d21 = x[j] - x[j - 1]
            denominator = d21 * d32 * (d21 + d32)
            plus[j, j - 1] = -d32 * d32 / denominator
            plus[j, j] = (d32 * d32 - d21 * d21) / denominator
            plus[j, j + 1] = d21 * d21 / denominator
        return plus, plus.copy()

    raise ValueError(f"Invalid xDotDerivativeScheme={scheme}")


def polynomial_interpolation_matrix(
    *,
    xk: np.ndarray,
    x: np.ndarray,
    alpxk: np.ndarray,
    alpx: np.ndarray,
) -> np.ndarray:
    """Barycentric spectral interpolation matrix between speed grids.

    Port of v3 ``polynomialInterpolationMatrix.F90``, used by the
    Fokker-Planck collision operator to interpolate the distribution function
    between species-specific speed variables ``x_b = x sqrt(T_a m_b / T_b m_a)``.
    Explicit loops mirror the Fortran rounding order for strict parity.

    Args:
      xk: Source collocation nodes, shape ``(n,)``.
      x: Target evaluation points, shape ``(m,)``.
      alpxk: Weight-function values at ``xk`` (e.g. ``exp(-xk^2) xk^k``).
      alpx: Weight-function values at ``x``.

    Returns:
      Interpolation matrix of shape ``(m, n)``.
    """
    xk = np.asarray(xk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    alpxk = np.asarray(alpxk, dtype=np.float64)
    alpx = np.asarray(alpx, dtype=np.float64)
    n = int(xk.size)
    m = int(x.size)
    if alpxk.shape != (n,):
        raise ValueError(f"alpxk must have shape {(n,)}, got {alpxk.shape}")
    if alpx.shape != (m,):
        raise ValueError(f"alpx must have shape {(m,)}, got {alpx.shape}")

    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = xk[i] - xk[j]
    for i in range(n):
        d[i, i] = 1.0

    w = np.zeros((n,), dtype=np.float64)
    for j in range(n):
        prod = 1.0
        for i in range(n):
            prod *= d[i, j]
        w[j] = 1.0 / prod

    mat = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            dxv = x[i] - xk[j]
            if dxv == 0.0:
                dxv = 1e-15
            mat[i, j] = 1.0 / dxv

    for i in range(m):
        denom = 0.0
        for j in range(n):
            denom += mat[i, j] * w[j]
        factor = alpx[i] / denom
        for j in range(n):
            mat[i, j] *= factor

    for j in range(n):
        factor = w[j] / alpxk[j]
        for i in range(m):
            mat[i, j] *= factor
    return mat


def rosenbluth_potential_grid_size(
    *,
    x: np.ndarray,
    n_x: int,
    x_max: float = 5.0,
    n_x_potentials_per_vth: float = 40.0,
    x_potentials_grid_scheme: int = 2,
    monoenergetic: bool = False,
) -> int:
    """Number of points of the auxiliary Rosenbluth-potential speed grid.

    Port of the ``NxPotentials`` sizing rule in ``createGrids.F90``. The
    default namelist values (``xMax=5.0``, ``NxPotentialsPerVth=40.0``,
    ``xPotentialsGridScheme=2``) match ``readInput.F90``.

    Args:
      x: Speed-grid nodes, shape ``(n_x,)``.
      n_x: Number of speed-grid points (namelist ``Nx``).
      x_max: Namelist ``xMax``; the potentials grid extends to
        ``max(x[-1], x_max)``.
      n_x_potentials_per_vth: Namelist ``NxPotentialsPerVth``.
      x_potentials_grid_scheme: Namelist ``xPotentialsGridScheme``; schemes
        3/4 place the potentials on the collocation grid plus a point at
        ``x_max`` and therefore use ``n_x + 1`` points.
      monoenergetic: RHSMode=3 monoenergetic runs use a single point.

    Returns:
      ``NxPotentials`` as an int.
    """
    if monoenergetic:
        return 1
    if x_potentials_grid_scheme in {3, 4}:
        return int(n_x) + 1
    x_np = np.asarray(x, dtype=float)
    x_max_not_too_small = max(float(x_np[-1]), float(x_max))
    return int(math.ceil(x_max_not_too_small * float(n_x_potentials_per_vth)))


# ----------------------------------------------------------------------------
# 4. Grids container (createGrids.F90)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Grids:
    """All grids, weights, and differentiation matrices for one solve.

    Field-compatible with the transitional ``discretization.v3.V3Grids``
    container (a frozen dataclass, kept for drop-in replacement at the purge)
    plus the integer resolution parameters. Mirrors the outputs of
    ``createGrids.F90``.

    Attributes:
      theta: Poloidal angle nodes on ``[0, 2 pi)``, shape ``(n_theta,)``.
      zeta: Toroidal angle nodes on ``[0, 2 pi / n_periods)``, shape
        ``(n_zeta,)``.
      x: Speed nodes (normalized to the species thermal speed), ``(n_x,)``.
      theta_weights: Quadrature weights integrating to ``2 pi``.
      zeta_weights: Quadrature weights integrating to ``2 pi`` (the
        per-period weights times ``n_periods``).
      x_weights: Plain ``dx`` speed quadrature weights.
      ddtheta: d/dtheta differentiation matrix, ``(n_theta, n_theta)``.
      ddzeta: d/dzeta differentiation matrix, ``(n_zeta, n_zeta)``.
      ddx: d/dx collocation matrix, ``(n_x, n_x)``.
      d2dx2: d^2/dx^2 collocation matrix, ``(n_x, n_x)``.
      ddtheta_magdrift_plus: Upwinded d/dtheta for positive magnetic-drift
        velocities (``ddtheta_magneticDrift_plus`` in v3).
      ddtheta_magdrift_minus: Upwinded d/dtheta for negative drifts.
      ddzeta_magdrift_plus: Upwinded d/dzeta for positive drifts.
      ddzeta_magdrift_minus: Upwinded d/dzeta for negative drifts.
      ddx_xdot_plus: Upwinded d/dx for positive ``xDotFactor``
        (``ddx_xDot_plus`` in v3); ``None`` when ``xDotDerivativeScheme=0``
        (both directions then use the centered ``ddx``).
      ddx_xdot_minus: Upwinded d/dx for negative ``xDotFactor``; ``None``
        when ``xDotDerivativeScheme=0``.
      n_xi: Maximum number of Legendre pitch modes.
      n_l: Number of Legendre modes kept in the Rosenbluth potentials.
      n_xi_for_x: Active Legendre modes per speed point, ``(n_x,)`` int32.
      n_theta: Number of theta points (after the force-odd rule).
      n_zeta: Number of zeta points (after the force-odd rule).
      n_x: Number of speed points.
      n_periods: Number of toroidal field periods.
    """

    theta: jnp.ndarray
    zeta: jnp.ndarray
    x: jnp.ndarray

    theta_weights: jnp.ndarray
    zeta_weights: jnp.ndarray
    x_weights: jnp.ndarray

    ddtheta: jnp.ndarray
    ddzeta: jnp.ndarray
    ddx: jnp.ndarray
    d2dx2: jnp.ndarray
    ddtheta_magdrift_plus: jnp.ndarray
    ddtheta_magdrift_minus: jnp.ndarray
    ddzeta_magdrift_plus: jnp.ndarray
    ddzeta_magdrift_minus: jnp.ndarray

    n_xi: int
    n_l: int
    n_xi_for_x: jnp.ndarray

    n_theta: int
    n_zeta: int
    n_x: int
    n_periods: int

    ddx_xdot_plus: jnp.ndarray | None = None
    ddx_xdot_minus: jnp.ndarray | None = None

    @property
    def xi_coupling_lower(self) -> np.ndarray:
        """Legendre coupling ``l/(2l-1)`` for the retained modes."""
        return legendre_coupling_lower(self.n_xi)

    @property
    def xi_coupling_upper(self) -> np.ndarray:
        """Legendre coupling ``(l+1)/(2l+3)`` for the retained modes."""
        return legendre_coupling_upper(self.n_xi)

    @property
    def lorentz_eigenvalues(self) -> np.ndarray:
        """Lorentz pitch-angle-scattering eigenvalues ``l(l+1)``."""
        return lorentz_eigenvalues(self.n_xi)


def _upwinded_pair(
    *, n: int, x_max: float, magnetic_drift_derivative_scheme: int, ddx_centered: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the (plus, minus) upwinded periodic d/dx pair for one angle."""
    if magnetic_drift_derivative_scheme == 0:
        return ddx_centered, ddx_centered
    schemes = _MAGNETIC_DRIFT_SCHEME_MAP.get(magnetic_drift_derivative_scheme)
    if schemes is None:
        raise ValueError(
            f"Invalid magneticDriftDerivativeScheme={magnetic_drift_derivative_scheme}"
        )
    scheme_plus, scheme_minus = schemes
    _, _, ddx_plus, _ = uniform_periodic_diff_matrices(
        n=n, x_min=0.0, x_max=x_max, scheme=scheme_plus
    )
    _, _, ddx_minus, _ = uniform_periodic_diff_matrices(
        n=n, x_min=0.0, x_max=x_max, scheme=scheme_minus
    )
    return ddx_plus, ddx_minus


def make_grids(
    *,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    n_x: int,
    n_l: int,
    n_periods: int,
    theta_derivative_scheme: int = 2,
    zeta_derivative_scheme: int = 2,
    magnetic_drift_derivative_scheme: int = 3,
    x_grid_scheme: int = 5,
    x_grid_k: float = 0.0,
    x_max: float = 5.0,
    x_dot_derivative_scheme: int = 0,
    n_xi_for_x_option: int = 1,
    monoenergetic: bool = False,
) -> Grids:
    """Build all grids for one solve from the resolution parameters.

    Port of the grid-construction part of ``createGrids.F90`` with the same
    defaults as ``readInput.F90``. Geometry-dependent inputs are reduced to
    the single integer ``n_periods``; reading it from an equilibrium file is
    the job of the geometry/io layer.

    Args:
      n_theta: Requested poloidal resolution (forced odd, as in v3).
      n_zeta: Requested toroidal resolution (forced odd; 1 = axisymmetric).
      n_xi: Number of Legendre pitch modes.
      n_x: Number of speed-grid points.
      n_l: Number of Legendre modes in the Rosenbluth potentials.
      n_periods: Number of toroidal field periods of the device.
      theta_derivative_scheme: Namelist ``thetaDerivativeScheme``
        (0 spectral, 1 2nd-order FD, 2 4th-order FD).
      zeta_derivative_scheme: Namelist ``zetaDerivativeScheme`` (same options).
      magnetic_drift_derivative_scheme: Namelist
        ``magneticDriftDerivativeScheme`` (0 = centered; +/-1, +/-2, +/-3 =
        upwinded pairs of increasing order).
      x_grid_scheme: Namelist ``xGridScheme``; 1/5 are the Landreman-Ernst
        polynomial grid with no node at x=0 (5 is the v3 default), 2/6 pin a
        node at x=0 (Gauss-Radau), 3/4 are the uniform grid on
        ``[0, x_max]`` with ``n_x + 1`` points and the last dropped (they
        differ only in the interpolation scheme used by the Fokker-Planck
        potentials), 7 is the Chebyshev grid on ``[0, x_max]`` with the last
        of ``n_x + 1`` points dropped, and 8 is the ``n_x``-point Chebyshev
        grid including ``x_max``.
      x_grid_k: Namelist ``xGrid_k`` weight exponent.
      x_max: Namelist ``xMax``; the domain limit of the uniform/Chebyshev
        grids (schemes 3/4/7/8; unused by 1/2/5/6).
      x_dot_derivative_scheme: Namelist ``xDotDerivativeScheme``; 0 uses the
        centered ``ddx`` for the E_r xDot term in both upwind directions,
        nonzero values build the ``ddx_xdot_plus``/``ddx_xdot_minus`` pair
        (see :func:`xdot_diff_matrices`).
      n_xi_for_x_option: Namelist ``Nxi_for_x_option`` ramp (0-3).
      monoenergetic: RHSMode=3 handling from ``createGrids.F90`` /
        ``validateInput.F90``: forces ``n_x=1``, ``n_xi_for_x_option=0``, a
        single speed point at ``x=1`` with weight ``e``, and zero ddx/d2dx2.

    Returns:
      A :class:`Grids` container.
    """
    n_theta = int(n_theta)
    n_zeta = int(n_zeta)
    n_x = int(n_x)
    n_xi = int(n_xi)
    n_l = int(n_l)
    n_periods = int(n_periods)

    # v3 forces odd Ntheta/Nzeta so the grids contain no point conjugate to
    # the Nyquist mode.
    if n_theta % 2 == 0:
        n_theta += 1
    if n_zeta % 2 == 0:
        n_zeta += 1

    if monoenergetic:
        # v3 validateInput() hard-overrides for RHSMode=3.
        n_x = 1
        n_xi_for_x_option = 0

    # --- theta ---
    theta_scheme = _ANGLE_DERIVATIVE_SCHEME_MAP.get(theta_derivative_scheme)
    if theta_scheme is None:
        raise ValueError(f"Invalid thetaDerivativeScheme={theta_derivative_scheme}")
    theta, theta_weights, ddtheta, _ = uniform_periodic_diff_matrices(
        n=n_theta, x_min=0.0, x_max=2 * math.pi, scheme=theta_scheme
    )
    ddtheta_magdrift_plus, ddtheta_magdrift_minus = _upwinded_pair(
        n=n_theta,
        x_max=2 * math.pi,
        magnetic_drift_derivative_scheme=magnetic_drift_derivative_scheme,
        ddx_centered=ddtheta,
    )

    # --- zeta ---
    zeta_max = 2 * math.pi / n_periods
    zeta_scheme = _ANGLE_DERIVATIVE_SCHEME_MAP.get(zeta_derivative_scheme)
    if zeta_scheme is None:
        raise ValueError(f"Invalid zetaDerivativeScheme={zeta_derivative_scheme}")
    if n_zeta == 1:
        # Axisymmetric: single zeta point carrying the full 2 pi weight.
        zeta = jnp.asarray(np.array([0.0], dtype=np.float64))
        zeta_weights = jnp.asarray(np.array([2 * math.pi * n_periods], dtype=np.float64))
        ddzeta = jnp.zeros((1, 1), dtype=jnp.float64)
        ddzeta_magdrift_plus = ddzeta
        ddzeta_magdrift_minus = ddzeta
    else:
        zeta, zeta_weights, ddzeta, _ = uniform_periodic_diff_matrices(
            n=n_zeta, x_min=0.0, x_max=zeta_max, scheme=zeta_scheme
        )
        zeta_weights = zeta_weights * n_periods
        ddzeta_magdrift_plus, ddzeta_magdrift_minus = _upwinded_pair(
            n=n_zeta,
            x_max=zeta_max,
            magnetic_drift_derivative_scheme=magnetic_drift_derivative_scheme,
            ddx_centered=ddzeta,
        )

    # --- x (speed) ---
    if monoenergetic:
        # createGrids.F90 for RHSMode=3: x = 1, xWeights = exp(1), ddx = 0.
        x = jnp.asarray(np.full((n_x,), 1.0, dtype=np.float64))
        x_weights = jnp.asarray(np.full((n_x,), math.exp(1.0), dtype=np.float64))
        ddx = jnp.zeros((n_x, n_x), dtype=jnp.float64)
        d2dx2 = jnp.zeros((n_x, n_x), dtype=jnp.float64)
    elif x_grid_scheme in {1, 2, 5, 6}:
        include_x0 = x_grid_scheme in {2, 6}
        speed_grid = make_speed_grid(n_x=n_x, k=x_grid_k, include_point_at_x0=include_x0)
        x = jnp.asarray(speed_grid.x)
        x_weights = jnp.asarray(speed_grid.dx_weights(x_grid_k))
        ddx_np, d2dx2_np = speed_grid_diff_matrices(
            np.asarray(speed_grid.x, dtype=np.float64), k=x_grid_k
        )
        ddx = jnp.asarray(ddx_np)
        d2dx2 = jnp.asarray(d2dx2_np)
    elif x_grid_scheme in {3, 4}:
        # Uniform grid on [0, x_max] with n_x+1 points; the last (f=0 there)
        # is dropped. Schemes 3 and 4 differ only in xInterpolationScheme.
        x_p1, w_p1, ddx_p1, d2dx2_p1 = uniform_aperiodic_diff_matrices(
            n=n_x + 1, x_min=0.0, x_max=x_max, scheme=12
        )
        x_np = x_p1[:n_x].copy()
        x_np[0] = 0.0
        x = jnp.asarray(x_np)
        x_weights = jnp.asarray(w_p1[:n_x])
        ddx = jnp.asarray(ddx_p1[:n_x, :n_x])
        d2dx2 = jnp.asarray(d2dx2_p1[:n_x, :n_x])
    elif x_grid_scheme == 7:
        # Chebyshev grid on [0, x_max] with n_x+1 points, last dropped;
        # d2dx2 is the square of the full (n_x+1)-point ddx, then truncated.
        x_p1, w_p1, ddx_p1 = chebyshev_grid(n=n_x + 1, x_min=0.0, x_max=x_max)
        x_np = x_p1[:n_x].copy()
        x_np[0] = 0.0
        d2dx2_p1 = ddx_p1 @ ddx_p1
        x = jnp.asarray(x_np)
        x_weights = jnp.asarray(w_p1[:n_x])
        ddx = jnp.asarray(ddx_p1[:n_x, :n_x])
        d2dx2 = jnp.asarray(d2dx2_p1[:n_x, :n_x])
    elif x_grid_scheme == 8:
        # Chebyshev grid on [0, x_max] with n_x points including x_max.
        x_np, w_np, ddx_np = chebyshev_grid(n=n_x, x_min=0.0, x_max=x_max)
        x_np = x_np.copy()
        x_np[0] = 0.0
        x = jnp.asarray(x_np)
        x_weights = jnp.asarray(w_np)
        ddx = jnp.asarray(ddx_np)
        d2dx2 = jnp.asarray(ddx_np @ ddx_np)
    else:
        raise ValueError(f"Invalid xGridScheme={x_grid_scheme} (must be 1-8).")

    # --- upwinded d/dx pair for the E_r xDot term (createGrids.F90) ---
    x_dot_derivative_scheme = int(x_dot_derivative_scheme)
    if not -2 <= x_dot_derivative_scheme <= 11:
        raise ValueError("xDotDerivativeScheme must be between -2 and 11.")
    if (
        x_dot_derivative_scheme > 0
        and x_dot_derivative_scheme != 11
        and x_grid_scheme not in {3, 4}
    ):
        raise ValueError(
            "If xDotDerivativeScheme is >0 and not 11, then xGridScheme must be either 3 or 4."
        )
    ddx_xdot_plus = None
    ddx_xdot_minus = None
    if x_dot_derivative_scheme != 0 and not monoenergetic:
        plus_np, minus_np = xdot_diff_matrices(
            x=np.asarray(x, dtype=np.float64),
            ddx=np.asarray(ddx, dtype=np.float64),
            k=x_grid_k,
            scheme=x_dot_derivative_scheme,
            x_max=x_max,
        )
        ddx_xdot_plus = jnp.asarray(plus_np)
        ddx_xdot_minus = jnp.asarray(minus_np)

    # --- pitch ramp ---
    n_xi_for_x = n_xi_for_x_ramp(
        x=np.asarray(x, dtype=float), n_xi=n_xi, n_l=n_l, option=n_xi_for_x_option
    )

    return Grids(
        theta=theta,
        zeta=zeta,
        x=x,
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        x_weights=x_weights,
        ddtheta=ddtheta,
        ddzeta=ddzeta,
        ddx=ddx,
        d2dx2=d2dx2,
        ddtheta_magdrift_plus=ddtheta_magdrift_plus,
        ddtheta_magdrift_minus=ddtheta_magdrift_minus,
        ddzeta_magdrift_plus=ddzeta_magdrift_plus,
        ddzeta_magdrift_minus=ddzeta_magdrift_minus,
        n_xi=n_xi,
        n_l=n_l,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_x=n_x,
        n_periods=n_periods,
        ddx_xdot_plus=ddx_xdot_plus,
        ddx_xdot_minus=ddx_xdot_minus,
    )
