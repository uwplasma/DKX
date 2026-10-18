"""Uniform periodic and aperiodic angle grids and widened upwind stencils."""

from __future__ import annotations

import math
from collections.abc import Sequence

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------
# 1. Uniform periodic grids for theta and zeta (uniformDiffMatrices.F90)
# ----------------------------------------------------------------------------

#: Internal uniformDiffMatrices scheme numbers with at least one production
#: call site. All are periodic grids that include x_min but not x_max.
#: Codes below 200 are the SFINCS v3 ``uniformDiffMatrices.F90`` numbering
#: (0-130, with the aperiodic variants at those values plus 2); the 200 block
#: holds the dkx-only widened upwind stencils, which upstream numbering can
#: never reach.
_PERIODIC_SCHEMES = frozenset(
    {0, 10, 20, 80, 90, 100, 110, 120, 130, 200, 210, 220, 230}
)

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

# ----------------------------------------------------------------------------
# 1a. Widened upwind stencils (dkx extension to uniformDiffMatrices.F90)
# ----------------------------------------------------------------------------

#: Widened upwind stencils for the first derivative on a uniform grid, given as
#: ``order -> (offsets, coefficients)`` in units of ``1/dx`` and oriented for a
#: **positive** wind (advection ``a df/dx`` with ``a > 0``, so the stencil leans
#: upstream toward negative offsets and the on-node coefficient is positive).
#:
#: Each entry is the solution of the standard Vandermonde moment conditions
#: ``sum_j c_j o_j^m = m! delta_{m1}`` for ``m = 0 .. N-1`` on ``N`` distinct
#: offsets, so an ``N``-point stencil is accurate to order ``N - 1``. The
#: offsets themselves were selected by exhaustive search over every ``N``-point
#: offset set that
#:
#: * lies in a window of width ``N + 1`` containing 0 (two grid cells wider than
#:   the compact ``N``-point stencil — the "widening"),
#: * is dissipative for a positive wind, i.e. the Fourier symbol
#:   ``chat(k) = sum_j c_j exp(i k o_j)`` satisfies ``Re chat(k) >= 0`` for all
#:   ``k`` in ``[-pi, pi]`` with strict positivity away from ``k = 0``, and
#: * maximizes the diagonal-dominance ratio
#:   ``|c_0| / sum_{j != 0} |c_j|`` (:func:`stencil_diagonal_dominance`).
#:
#: The last criterion is the whole point: a relaxation smoother or a simple
#: diagonal preconditioner on an advection-dominated operator only works when
#: the discrete derivative carries weight on the diagonal, and a centered
#: stencil carries exactly none (``c_0 = 0``). The measured ratios are
#: ``5/7 ~ 0.714`` (order 3) and ``13/21 ~ 0.619`` (order 4), against ``0`` for
#: both centered schemes and ``1/3``, ``5/14``, ``2/11`` for the compact
#: upwind-biased stencils of ``magneticDriftDerivativeScheme`` 1/2/3.
#:
#: On upwind-biased differencing for advection and the dissipativity criterion
#: see Fromm, J. Comput. Phys. 3, 176 (1968); Warming & Beam, AIAA J. 14, 1241
#: (1976); Tam & Webb, J. Comput. Phys. 107, 262 (1993); and Hirsch,
#: *Numerical Computation of Internal and External Flows*, 2nd ed. (2007),
#: ch. 8. Arbitrary-offset weights follow Fornberg, Math. Comput. 51, 699
#: (1988).
_WIDENED_UPWIND_STENCILS: dict[int, tuple[tuple[int, ...], tuple[float, ...]]] = {
    # 4 points, span 5, truncation error +1/4 dx^3 d4f/dx4, |c_0|/off = 5/7.
    3: ((-3, -1, 0, 2), (1.0 / 15.0, -1.0, 5.0 / 6.0, 1.0 / 10.0)),
    # 5 points, span 6, truncation error +1/5 dx^4 d5f/dx5, |c_0|/off = 13/21.
    4: (
        (-4, -3, -1, 0, 2),
        (-1.0 / 12.0, 4.0 / 15.0, -4.0 / 3.0, 13.0 / 12.0, 1.0 / 15.0),
    ),
}

#: Internal scheme code of each widened upwind stencil, as
#: ``order -> (plus, minus)`` where ``plus`` is the positive-wind orientation.
_WIDENED_UPWIND_SCHEMES: dict[int, tuple[int, int]] = {3: (200, 210), 4: (220, 230)}

#: Internal scheme code -> ``(order, wind_sign)``.
_WIDENED_UPWIND_SCHEME_INFO: dict[int, tuple[int, float]] = {
    scheme: (order, sign)
    for order, pair in _WIDENED_UPWIND_SCHEMES.items()
    for scheme, sign in zip(pair, (1.0, -1.0))
}

#: dkx-only ``thetaDerivativeScheme`` / ``zetaDerivativeScheme`` extension
#: codes -> internal scheme. SFINCS v3 numbers every namelist scheme knob with
#: small integers (0-2 for the angular schemes, -3..3 for
#: ``magneticDriftDerivativeScheme``, -2..11 for ``xDotDerivativeScheme``, 1-8
#: for ``xGridScheme``), so the 100 block is reserved for dkx extensions and
#: cannot collide with an upstream value. The sign selects the wind: ``+103``
#: is the 3rd-order widened upwind stencil for a positive wind and ``-103`` the
#: mirrored stencil for a negative wind, and likewise ``+/-104`` at 4th order.
#: These are not bit-parity with the Fortran code and are never a default.
_WIDENED_ANGLE_DERIVATIVE_SCHEME_MAP = {103: 200, -103: 210, 104: 220, -104: 230}

#: dkx-only ``magneticDriftDerivativeScheme`` extension codes -> (plus, minus),
#: with the same sign convention as the SFINCS +/-1, +/-2, +/-3 pairs (a
#: negative code swaps the two orientations).
_WIDENED_MAGNETIC_DRIFT_SCHEME_MAP = {
    103: (200, 210),
    -103: (210, 200),
    104: (220, 230),
    -104: (230, 220),
}


def widened_upwind_stencil(
    *, order: int, wind_sign: float = 1.0
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Offsets and ``1/dx`` coefficients of a widened upwind stencil.

    See :data:`_WIDENED_UPWIND_STENCILS` for the construction and the
    literature. The stencil is mirrored for a negative wind: the offsets are
    negated and the coefficients change sign, which is the exact reflection of
    the positive-wind stencil about the evaluation node and preserves both the
    order of accuracy and the diagonal weight.

    Args:
      order: Order of accuracy, 3 (4-point stencil) or 4 (5-point stencil).
      wind_sign: Sign of the local advection speed; positive selects the
        upstream-leaning (negative-offset) stencil, negative its mirror. Zero
        is rejected because it leaves the upwind direction undefined.

    Returns:
      Tuple ``(offsets, coefficients)`` sorted by increasing offset. Multiply
      the coefficients by ``1/dx`` to get the derivative weights.
    """
    stencil = _WIDENED_UPWIND_STENCILS.get(int(order))
    if stencil is None:
        raise ValueError(
            f"Invalid widened upwind order {order} (supported: "
            f"{sorted(_WIDENED_UPWIND_STENCILS)})"
        )
    offsets, coefficients = stencil
    wind_sign = float(wind_sign)
    if wind_sign == 0.0:
        raise ValueError(
            "wind_sign must be nonzero; the upwind direction is otherwise undefined"
        )
    if wind_sign > 0.0:
        return offsets, coefficients
    mirrored = sorted(zip((-o for o in offsets), (-c for c in coefficients)))
    return tuple(o for o, _ in mirrored), tuple(c for _, c in mirrored)


def stencil_diagonal_dominance(
    offsets: Sequence[int], coefficients: Sequence[float]
) -> float:
    """Diagonal-to-off-diagonal row-sum ratio ``|c_0| / sum_{j != 0} |c_j|``.

    The diagonal-dominance measure that decides whether a relaxation smoother
    or a diagonal preconditioner is useful on the advection operator ``a d/dx``
    built from this stencil: the row of the resulting matrix has diagonal
    ``a c_0 / dx`` and off-diagonal entries ``a c_j / dx``, so the ratio is
    independent of both the wind speed and the grid spacing. Every consistent
    first-derivative stencil satisfies ``sum_j c_j = 0``, which bounds the
    ratio by 1 (attained only by the two-point first-order upwind stencil).

    Args:
      offsets: Grid offsets of the stencil.
      coefficients: Matching coefficients in units of ``1/dx``.

    Returns:
      The ratio as a float; ``0.0`` for a stencil with no on-node coefficient
      (both centered schemes), ``inf`` for a stencil with no off-diagonal
      entries.
    """
    offsets = tuple(int(o) for o in offsets)
    coefficients = tuple(float(c) for c in coefficients)
    if len(offsets) != len(coefficients):
        raise ValueError("offsets and coefficients must have the same length")
    diagonal = sum(abs(c) for o, c in zip(offsets, coefficients) if o == 0)
    off_diagonal = sum(abs(c) for o, c in zip(offsets, coefficients) if o != 0)
    if off_diagonal == 0.0:
        return math.inf
    return diagonal / off_diagonal


def _periodic_stencil_rows(
    *, n: int, dx: float, offsets: Sequence[int], coefficients: Sequence[float]
) -> np.ndarray:
    """Circulant matrix of one stencil on a periodic n-point grid."""
    rows = np.arange(n)
    out = np.zeros((n, n), dtype=np.float64)
    for offset, coefficient in zip(offsets, coefficients):
        out[rows, (rows + int(offset)) % n] += float(coefficient) / dx
    return out


def widened_upwind_periodic_diff_matrix(
    *,
    n: int,
    x_min: float,
    x_max: float,
    order: int,
    wind_sign: float | Sequence[float] | np.ndarray,
) -> jnp.ndarray:
    """Wind-oriented widened upwind d/dx matrix on a uniform periodic grid.

    The orientation of every row is set by the sign of the local advection
    speed, which for the drift-kinetic streaming and drift terms is fixed by
    the field geometry. Pass a scalar when the wind has one sign over the whole
    coordinate, or an array of per-point signs when it does not.

    Args:
      n: Number of grid points; must exceed the stencil span so that no two
        offsets alias onto the same column.
      x_min: Left end of the periodic interval (included in the grid).
      x_max: Right end of the periodic interval (excluded from the grid).
      order: Order of accuracy, 3 or 4 (see :func:`widened_upwind_stencil`).
      wind_sign: Scalar sign of the advection speed, or an array of shape
        ``(n,)`` giving the sign at each grid point.

    Returns:
      A JAX float64 array of shape ``(n, n)``.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if x_min >= x_max:
        raise ValueError(f"x_max must be > x_min, got x_min={x_min}, x_max={x_max}")
    offsets, _ = widened_upwind_stencil(order=order)
    span = max(offsets) - min(offsets)
    if n <= span:
        raise ValueError(
            f"n must be greater than the stencil span {span} for the order-{order} "
            f"widened upwind stencil, got n={n}"
        )

    dx = (float(x_max) - float(x_min)) / n
    plus = _periodic_stencil_rows(
        n=n, dx=dx, offsets=offsets, coefficients=widened_upwind_stencil(order=order)[1]
    )
    signs = np.asarray(wind_sign, dtype=np.float64)
    if signs.ndim == 0 and float(signs) > 0.0:
        return jnp.asarray(plus)
    minus_offsets, minus_coefficients = widened_upwind_stencil(
        order=order, wind_sign=-1.0
    )
    minus = _periodic_stencil_rows(
        n=n, dx=dx, offsets=minus_offsets, coefficients=minus_coefficients
    )
    if signs.ndim == 0:
        return jnp.asarray(minus)
    if signs.shape != (n,):
        raise ValueError(
            f"wind_sign must be a scalar or have shape {(n,)}, got {signs.shape}"
        )
    if np.any(signs == 0.0):
        raise ValueError("wind_sign must be nonzero at every grid point")
    return jnp.asarray(np.where(signs[:, None] > 0.0, plus, minus))


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
        ``magneticDriftDerivativeScheme`` 1/2/3. The dkx-only 200 block adds
        the widened upwind stencils of :func:`widened_upwind_stencil`:
        ``200``/``210`` positive/negative-wind 4-point (3rd order) and
        ``220``/``230`` positive/negative-wind 5-point (4th order), first
        derivative only.

    Returns:
      Tuple ``(x, weights, ddx, d2dx2)`` of JAX float64 arrays with shapes
      ``(n,)``, ``(n,)``, ``(n, n)``, ``(n, n)``. For schemes 100-130 only the
      first-derivative matrix is populated (``d2dx2`` is zero), matching the
      Fortran code; the same holds for the widened 200 block.
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
            topc = np.array(
                [0.5 / math.tan(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype
            )
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = -topc[n1 - 1 :: -1]
        else:
            topc = np.array(
                [0.5 / math.sin(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype
            )
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
                [-(0.5) / (math.sin(i * h / 2) ** 2) for i in range(1, n2 + 1)],
                dtype=dtype,
            )
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = topc[n1 - 1 :: -1]
        else:
            col1[0] = -pi * pi / (3 * h * h) + 1.0 / 12
            topc = np.array(
                [
                    -(0.5) / (math.sin(i * h / 2) * math.tan(i * h / 2))
                    for i in range(1, n2 + 1)
                ],
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

    else:
        # dkx-only widened upwind stencils (200 block), first derivative only.
        order, wind_sign = _WIDENED_UPWIND_SCHEME_INFO[scheme]
        return (
            jnp.asarray(x),
            jnp.asarray(weights),
            widened_upwind_periodic_diff_matrix(
                n=n, x_min=x_min, x_max=x_max, order=order, wind_sign=wind_sign
            ),
            jnp.asarray(d2dx2),
        )

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
    x = float(x_min) + (float(x_max) - float(x_min)) * (
        np.arange(n, dtype=dtype) / (n - 1)
    )
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


