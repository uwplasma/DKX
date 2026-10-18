"""Grids and discretization machinery for the SFINCS drift-kinetic equation.

This module consolidates every grid/quadrature/differentiation primitive that a
dkx solve needs, mirroring the grid-construction part of the Fortran v3
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

This canonical module replaced the retired ``dkx/grids.py`` and the
grid-construction half of the retired ``discretization`` package; the
polynomial x-grid kernel the collision operators consume lives in
:mod:`dkx.xgrid`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .phase_space_angles import (  # noqa: E402, F401
    _ANGLE_DERIVATIVE_SCHEME_MAP,
    _APERIODIC_SCHEMES,
    _MAGNETIC_DRIFT_SCHEME_MAP,
    _PERIODIC_SCHEMES,
    _WIDENED_ANGLE_DERIVATIVE_SCHEME_MAP,
    _WIDENED_MAGNETIC_DRIFT_SCHEME_MAP,
    _WIDENED_UPWIND_SCHEME_INFO,
    _WIDENED_UPWIND_SCHEMES,
    _WIDENED_UPWIND_STENCILS,
    _periodic_stencil_rows,
    stencil_diagonal_dominance,
    uniform_aperiodic_diff_matrices,
    uniform_periodic_diff_matrices,
    widened_upwind_periodic_diff_matrix,
    widened_upwind_stencil,
)
from .phase_space_speed import (  # noqa: E402, F401
    SpeedGrid,
    _evaluate_orthogonal_polynomial,
    _integrate_split,
    _make_speed_grid_cached,
    _speed_weight_d1_over_weight,
    _speed_weight_d2_over_weight,
    make_speed_grid,
    polynomial_interpolation_matrix,
    rosenbluth_potential_grid_size,
    speed_grid_diff_matrices,
    speed_weight,
    xdot_diff_matrices,
)

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
    "stencil_diagonal_dominance",
    "uniform_aperiodic_diff_matrices",
    "uniform_periodic_diff_matrices",
    "widened_upwind_periodic_diff_matrix",
    "widened_upwind_stencil",
    "xdot_diff_matrices",
]


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

    Option 1 is the SFINCS *default*, so this ramp is what a deck that says
    nothing about ``Nxi_for_x_option`` actually gets.

    The reference is ``fortran/version3/createGrids.F90`` -- the variant this
    package ports and the one the ``version3`` binary is built from.  Other
    trees in the SFINCS repository (``fortran/Fourier`` in particular) carry a
    *different* ramp, and reading one of those instead is a live way to
    "fix" this into disagreement.

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
        # dkx extension (``version3/createGrids.F90`` stops at 2): the same
        # linear ramp, uncapped and with the lower floor.
        for j in range(n_x):
            temp = n_xi * (0.1 + 0.9 * x_np[j] / 2.0)
            out[j] = max(3, n_l, int(temp))
    else:
        raise ValueError(f"Invalid Nxi_for_x_option={option}")
    return out



# ----------------------------------------------------------------------------
# 4. Grids container (createGrids.F90)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Grids:
    """All grids, weights, and differentiation matrices for one solve.

    Field-compatible with the retired ``discretization.v3.V3Grids``
    container (a frozen dataclass)
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


def _angle_derivative_scheme(namelist_scheme: int) -> int | None:
    """thetaDerivativeScheme / zetaDerivativeScheme -> internal scheme, or None."""
    scheme = _ANGLE_DERIVATIVE_SCHEME_MAP.get(namelist_scheme)
    if scheme is not None:
        return scheme
    return _WIDENED_ANGLE_DERIVATIVE_SCHEME_MAP.get(namelist_scheme)


def _upwinded_pair(
    *,
    n: int,
    x_max: float,
    magnetic_drift_derivative_scheme: int,
    ddx_centered: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the (plus, minus) upwinded periodic d/dx pair for one angle."""
    if magnetic_drift_derivative_scheme == 0:
        return ddx_centered, ddx_centered
    schemes = _MAGNETIC_DRIFT_SCHEME_MAP.get(
        magnetic_drift_derivative_scheme
    ) or _WIDENED_MAGNETIC_DRIFT_SCHEME_MAP.get(magnetic_drift_derivative_scheme)
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
    n_xi_for_x_override: Sequence[int] | None = None,
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
        (0 spectral, 1 2nd-order FD, 2 4th-order FD). The dkx-only opt-in
        codes ``+/-103`` and ``+/-104`` select the 3rd- and 4th-order widened
        upwind stencils of :func:`widened_upwind_stencil`, with the sign giving
        the wind direction; they are not bit-parity with the Fortran code.
      zeta_derivative_scheme: Namelist ``zetaDerivativeScheme`` (same options).
      magnetic_drift_derivative_scheme: Namelist
        ``magneticDriftDerivativeScheme`` (0 = centered; +/-1, +/-2, +/-3 =
        upwinded pairs of increasing order; dkx-only ``+/-103`` and ``+/-104``
        = the widened upwind pairs of 3rd and 4th order).
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
      n_xi_for_x_override: Optional explicit active Legendre-mode count at
        every speed node. Values must be nondecreasing, lie between
        ``max(4, n_l)`` and ``n_xi``, and retain ``n_xi`` at the final node.
        This DKX evidence control replaces the ramp without changing the
        rectangular operator shape.
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
        if n_xi_for_x_override is not None:
            raise ValueError(
                "n_xi_for_x_override is unavailable for monoenergetic grids"
            )
        # v3 validateInput() hard-overrides for RHSMode=3.
        n_x = 1
        n_xi_for_x_option = 0

    # --- theta ---
    theta_scheme = _angle_derivative_scheme(theta_derivative_scheme)
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
    zeta_scheme = _angle_derivative_scheme(zeta_derivative_scheme)
    if zeta_scheme is None:
        raise ValueError(f"Invalid zetaDerivativeScheme={zeta_derivative_scheme}")
    if n_zeta == 1:
        # Axisymmetric: single zeta point carrying the full 2 pi weight.
        zeta = jnp.asarray(np.array([0.0], dtype=np.float64))
        zeta_weights = jnp.asarray(
            np.array([2 * math.pi * n_periods], dtype=np.float64)
        )
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
        speed_grid = make_speed_grid(
            n_x=n_x, k=x_grid_k, include_point_at_x0=include_x0
        )
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
    if n_xi_for_x_override is None:
        n_xi_for_x = n_xi_for_x_ramp(
            x=np.asarray(x, dtype=float),
            n_xi=n_xi,
            n_l=n_l,
            option=n_xi_for_x_option,
        )
    else:
        values = tuple(n_xi_for_x_override)
        if len(values) != n_x:
            raise ValueError(
                f"n_xi_for_x_override must have {n_x} entries, got {len(values)}"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("n_xi_for_x_override entries must be integers")
        floor = max(4, n_l)
        if any(value < floor or value > n_xi for value in values):
            raise ValueError(
                f"n_xi_for_x_override entries must lie from {floor} through {n_xi}"
            )
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError("n_xi_for_x_override must be nondecreasing")
        if values[-1] != n_xi:
            raise ValueError(
                f"n_xi_for_x_override must retain n_xi={n_xi} at the final speed node"
            )
        n_xi_for_x = np.asarray(values, dtype=int)

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
