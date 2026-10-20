"""Speed (x) grid: Landreman-Ernst polynomial collocation and its derivative matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from .phase_space_angles import uniform_aperiodic_diff_matrices  # noqa: E402

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
    out[~mask0] = (
        k * (k - 1.0) / (x[~mask0] * x[~mask0])
        - 2.0 * (2.0 * k + 1.0)
        + 4.0 * (x[~mask0] * x[~mask0])
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

    def __post_init__(self) -> None:
        # Instances are shared by a memoizing cache; read-only arrays stop one
        # caller's in-place edit from silently changing every later result.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)


def _integrate_split(f, *, finite_bound: float = 10.0) -> float:
    """Integrate f on [0, inf), semi-infinite part first, matching v3 ordering."""
    a2, _ = quad(f, finite_bound, np.inf, epsabs=0.0, epsrel=1e-13, limit=5000)
    a1, _ = quad(f, 0.0, finite_bound, epsabs=0.0, epsrel=1e-13, limit=5000)
    return float(a2 + a1)


def _evaluate_orthogonal_polynomial(
    x: float, *, j: int, a: np.ndarray, b: np.ndarray
) -> float:
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


def speed_grid_diff_matrices(
    x: np.ndarray, *, k: float
) -> tuple[np.ndarray, np.ndarray]:
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

    def _uniform_pair(
        scheme_plus: int, scheme_minus: int
    ) -> tuple[np.ndarray, np.ndarray]:
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

