"""SFINCS-v3-compatible speed-grid quadrature and differentiation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def x_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """Weight function used by SFINCS v3 for the polynomial x grid."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-(x * x)) * (x**k)


def x_weight_d1_over_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """(d/dx weight) / weight, matching v3 `polynomialDiffMatrices.F90`."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = 0.0
    out[~mask0] = k / x[~mask0] - 2.0 * x[~mask0]
    return out


def x_weight_d2_over_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """(d^2/dx^2 weight) / weight, matching v3 `polynomialDiffMatrices.F90`."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = -2.0
    out[~mask0] = k * (k - 1.0) / (x[~mask0] * x[~mask0]) - 2.0 * (2.0 * k + 1.0) + 4.0 * (
        x[~mask0] * x[~mask0]
    )
    return out


def make_x_polynomial_diff_matrices(x: np.ndarray, *, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Port of v3 `makeXPolynomialDiffMatrices` (polynomialDiffMatrices.F90).

    Returns
    -------
    ddx, d2dx2:
      First and second derivative matrices on the (possibly nonuniform) x-grid.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n < 1:
        raise ValueError("x must have at least one point")

    xx = np.broadcast_to(x[:, None], (n, n)).copy()
    dx = xx - xx.T
    np.fill_diagonal(dx, 1.0)

    c = np.prod(dx, axis=1)
    c = c * x_weight_np(x, k)
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
    y[0, :] = x_weight_d1_over_weight_np(x, k)
    for i in range(1, n):
        y[i, :] = y[i - 1, :] + xxx[i - 1, :]

    ddx = z * ccc
    np.fill_diagonal(ddx, y[-1, :])

    old_y = y
    y2 = np.zeros((n, n), dtype=np.float64)
    y2[0, :] = x_weight_d2_over_weight_np(x, k)
    for i in range(1, n):
        y2[i, :] = y2[i - 1, :] + 2.0 * old_y[i - 1, :] * xxx[i - 1, :]

    repmat_diag_ddx = np.broadcast_to(np.diag(ddx)[:, None], (n, n))
    d2dx2 = 2.0 * z * (ccc * repmat_diag_ddx - ddx)
    np.fill_diagonal(d2dx2, y2[-1, :])

    return ddx, d2dx2

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def speed_recurrence(
    n: int, k: float, x0: float, finite_bound: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Stieltjes recurrence ``a, b, c`` (1-based) for the weight ``exp(-x^2) x^k``.

    The inner products are composite Gauss-Legendre sums on ``[0, inf)``: panels
    graded geometrically toward 0 (so non-integer ``k`` converges), uniform panels
    up to ``finite_bound`` and a tail to ``finite_bound + 16``.  Also returns the
    last two polynomial values at ``x0`` for the Gauss-Radau modification.
    """
    edges = np.concatenate([[0.0], finite_bound * 2.0 ** -np.arange(40, 3, -1),
                            np.linspace(finite_bound / 8, finite_bound, 17),
                            np.linspace(finite_bound, finite_bound + 16.0, 9)[1:]])  # fmt: skip
    lo, hi = edges[:-1, None], edges[1:, None]
    x = ((hi - lo) / 2 * _GL_NODES + (hi + lo) / 2).ravel()
    weight = ((hi - lo) / 2 * _GL_WEIGHTS).ravel() * np.exp(-(x * x)) * x**k
    a, b, c = np.zeros(n + 1), np.zeros(n + 2), np.zeros(n + 1)
    p, p_prev, oldc = np.ones_like(x), np.zeros_like(x), 1.0
    last, penult = 1.0, 0.0  # p_j(x0) and p_{j-1}(x0)
    for j in range(1, n + 1):
        if j > 1:
            p, p_prev = (x - a[j - 1]) * p - b[j - 1] * p_prev, p
            penult, last = last, (x0 - a[j - 1]) * last - b[j - 1] * penult
        c[j] = np.sum(weight * p * p)
        b[j] = c[j] / oldc
        a[j] = np.sum(weight * x * p * p) / c[j]
        oldc = c[j]
    return a, b, c, last, penult


@dataclass(frozen=True)
class XGrid:
    x: np.ndarray
    gaussian_weights: np.ndarray
    poly_a: np.ndarray  # (n+1,) 1-based recurrence "a" (a[0] unused)
    poly_b: np.ndarray  # (n+2,) 1-based recurrence "b" (b[0] unused)
    poly_c: np.ndarray  # (n+1,) 1-based norms "c" (c[0] unused)
    k: float
    include_point_at_x0: bool
    x0: float

    def __post_init__(self) -> None:
        # Instances are shared by a memoizing cache; read-only arrays stop one
        # caller's in-place edit from silently changing every later result.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def dx_weights(self, k: float | None = None) -> np.ndarray:
        """Return weights for plain `dx` integrals (Fortran divides by the weight function)."""
        if k is None:
            k = self.k
        w = np.exp(-(self.x * self.x)) * (self.x**k)
        return self.gaussian_weights / w


@lru_cache(maxsize=32)
def _make_x_grid_cached(
    *,
    n: int,
    k: float,
    include_point_at_x0: bool,
    x0: float,
    finite_bound: float,
) -> XGrid:
    return _make_x_grid_uncached(
        n=n,
        k=k,
        include_point_at_x0=include_point_at_x0,
        x0=x0,
        finite_bound=finite_bound,
    )


def make_x_grid(
    *,
    n: int,
    k: float = 0.0,
    include_point_at_x0: bool = False,
    x0: float = 0.0,
    finite_bound: float = 10.0,
) -> XGrid:
    """Port of SFINCS v3 `makeXGrid` for the built-in weight exp(-x^2)*x^k on [0, +inf).

    This routine is not performance-critical for the JAX solve (the grid is static). It is
    implemented primarily for parity with the Fortran v3 grids.
    """
    return _make_x_grid_cached(
        n=int(n),
        k=float(k),
        include_point_at_x0=bool(include_point_at_x0),
        x0=float(x0),
        finite_bound=float(finite_bound),
    )


def _make_x_grid_uncached(
    *,
    n: int,
    k: float = 0.0,
    include_point_at_x0: bool = False,
    x0: float = 0.0,
    finite_bound: float = 10.0,
) -> XGrid:
    """Uncached implementation of :func:`make_x_grid` (split out for memoization)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k < 0:
        raise ValueError("k must be >= 0 for the built-in SFINCS weight")

    a, b, c, last_poly_x0, penult_poly_x0 = speed_recurrence(n, k, x0, finite_bound)

    if include_point_at_x0:
        a[n] = x0 - b[n] * penult_poly_x0 / last_poly_x0

    # Jacobi matrix eigen-decomposition (Golub-Welsch).
    diag = a[1 : n + 1].copy()
    off = np.sqrt(b[2 : n + 1].copy())
    try:
        if include_point_at_x0:
            from scipy.linalg import eigh_tridiagonal

            abscissae, eigenvectors = eigh_tridiagonal(diag, off, lapack_driver="stevr")
        else:
            from scipy.linalg.lapack import dpteqr

            d_out, e_out, z, info = dpteqr("I", diag.copy(), off.copy())
            if info != 0:
                raise RuntimeError(f"dpteqr failed with info={info}")
            # Fortran reverses order for DPTEQR.
            abscissae = d_out[::-1]
            eigenvectors = z[:, ::-1]
    except Exception:  # noqa: BLE001
        jmat = np.diag(diag)
        for i in range(n - 1):
            jmat[i, i + 1] = off[i]
            jmat[i + 1, i] = off[i]
        abscissae, eigenvectors = np.linalg.eigh(jmat)
    weights = c[1] * (eigenvectors[0, :] ** 2)

    if include_point_at_x0:
        # Match the Fortran behavior: force the smallest node to be exactly x0 (typically 0).
        abscissae = abscissae.copy()
        abscissae[0] = x0

    return XGrid(
        x=abscissae,
        gaussian_weights=weights,
        poly_a=a,
        poly_b=b,
        poly_c=c,
        k=float(k),
        include_point_at_x0=bool(include_point_at_x0),
        x0=float(x0),
    )
