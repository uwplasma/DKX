"""Pitch-basis diagnostics: measurements behind the negative result in ``dkx.multigrid``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()


from .drift_kinetic import KineticOperator  # noqa: E402
from .multigrid_levels import simplified_operator  # noqa: E402

# =============================================================================
# Pitch-basis diagnostics: why no relaxation smooths in a Legendre-modal basis
# =============================================================================
#
# Everything below is *measurement*, not a solver path.  It exists so the
# negative result in this module's docstring -- that the multigrid family cannot
# be made to work on a Legendre-modal pitch discretization -- is reproducible
# from the repository rather than asserted, and so a future lane can re-measure
# it on its own deck before spending a week on the alternative.

# Backward-biased (wind blowing from smaller index) first-derivative stencils,
# as offsets in units of the grid spacing.  ``up*`` are the textbook one-sided
# and upwind-biased schemes; ``wide*`` skip near neighbours to buy diagonal
# dominance at the same formal order -- the same trade Leonard's QUICK family
# makes for boundedness (B. P. Leonard, Comput. Methods Appl. Mech. Eng. 19, 59
# (1979)).  ``ctr2`` is the centered scheme SFINCS and dkx actually use on the
# angles (``thetaDerivativeScheme`` 1), included as the zero-dissipation
# reference.
UPWIND_STENCILS: dict[str, tuple[int, ...]] = {
    "up1": (-1, 0),
    "up2": (-2, -1, 0),
    "up3": (-2, -1, 0, 1),
    "wide2": (-4, -1, 0),
    "wide4": (-4, -3, -1, 0, 2),
    "ctr2": (-1, 0, 1),
}

def _stencil_weights(offsets: tuple[int, ...], order: int = 1) -> np.ndarray:
    """Finite-difference weights of the ``order``-th derivative at ``offsets*h``."""
    nodes = np.asarray(offsets, dtype=np.float64)
    vander = np.vander(nodes, nodes.size, increasing=True).T
    rhs = np.zeros(nodes.size)
    rhs[order] = float(math.factorial(order))
    return np.linalg.solve(vander, rhs)

def stencil_matrices(
    n: int, h: float, name: str, *, periodic: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(backward-biased, forward-biased)`` first-derivative pair of a stencil.

    An upwinded advection ``w df/dx`` uses the backward-biased matrix wherever
    ``w > 0`` and the forward-biased one wherever ``w <= 0``.  ``periodic=False``
    reflects the stencil at the ends (an even extension), which is the right
    closure for the pitch angle: ``f`` is a function of ``xi = cos(alpha)`` and
    is therefore even about ``alpha = 0`` and ``alpha = pi``, where the mirror
    advection speed vanishes anyway.

    Args:
        n: number of grid points.
        h: uniform spacing.
        name: key of :data:`UPWIND_STENCILS`.
        periodic: wrap (angles) rather than reflect (pitch).

    Returns:
        Two ``(n, n)`` arrays.
    """
    offsets = UPWIND_STENCILS[name]

    def build(offs: tuple[int, ...]) -> np.ndarray:
        w = _stencil_weights(offs, 1) / h
        mat = np.zeros((n, n))
        for j, off in enumerate(offs):
            for i in range(n):
                k = i + off
                if periodic:
                    k %= n
                else:
                    if k < 0:
                        k = -1 - k
                    if k > n - 1:
                        k = 2 * n - 1 - k
                    k = min(max(k, 0), n - 1)
                mat[i, k] += w[j]
        return mat

    return build(offsets), build(tuple(-o for o in offsets)[::-1])

def line_diagonal_dominance(
    matrix: np.ndarray, shape: tuple[int, ...], axis: int
) -> tuple[float, float]:
    """``(min, median)`` diagonal dominance of the line blocks along ``axis``.

    An alternating line block-Jacobi smoother inverts, for each coordinate in
    turn, the block of ``matrix`` that runs along that coordinate with all the
    others frozen.  The relevant question about such a block is whether it is
    diagonally dominant, ``d = min_i |a_ii| / sum_{j != i} |a_ij|``: a damped
    Jacobi/block-Jacobi sweep is a smoother when ``d`` is order one and
    *amplifies* when ``d`` is near zero (Trottenberg et al., section 2.1;
    Brandt, Math. Comp. **31**, 333 (1977), section 3).

    Args:
        matrix: dense operator on the flattened ``shape`` grid.
        shape: grid shape the operator acts on.
        axis: which axis the line runs along.

    Returns:
        The minimum and median of ``d`` over all lines and rows.
    """
    n_dim = len(shape)
    m = shape[axis]
    rest = tuple(s for i, s in enumerate(shape) if i != axis)
    n_rest = int(np.prod(rest)) if rest else 1
    blocks = np.moveaxis(
        matrix.reshape(tuple(shape) + tuple(shape)), (axis, n_dim + axis), (0, 1)
    ).reshape((m, m, n_rest, n_rest))
    blocks = np.einsum("abrr->rab", blocks)
    diag = np.abs(np.einsum("rii->ri", blocks))
    off = np.abs(blocks).sum(axis=2) - diag
    finite = off > 1e-14
    ratio = np.where(finite, diag / np.where(finite, off, 1.0), np.inf)
    return float(ratio.min()), float(np.median(ratio))

def dense_simplified_block(
    op: KineticOperator, *, species: int = 0, speed: int = 0
) -> np.ndarray:
    """Dense ``(Nxi*Ntheta*Nzeta)`` square block of the simplified operator.

    The simplified operator (:func:`simplified_operator`) is uncoupled over
    ``(species, x)``, so one such block is the whole thing the classical Krylov
    preconditioner factors, for one speed node of one species.  Index order is
    ``(L, theta, zeta)`` with ``zeta`` fastest, matching
    :meth:`KineticOperator.to_block_tridiagonal`.

    Only for analysis and tests -- it materializes ``(Nxi*Ntheta*Nzeta)**2``
    entries.
    """
    blocks = simplified_operator(op).to_block_tridiagonal()
    lower = np.asarray(blocks.lower[:, species, speed])
    diag = np.asarray(blocks.diag[:, species, speed])
    upper = np.asarray(blocks.upper[:, species, speed])
    n_l, n_tz, _ = diag.shape
    out = np.zeros((n_l * n_tz, n_l * n_tz))
    for ell in range(n_l):
        row = slice(ell * n_tz, (ell + 1) * n_tz)
        out[row, row] = diag[ell]
        if ell >= 1:
            out[row, (ell - 1) * n_tz : ell * n_tz] = lower[ell]
        if ell + 1 < n_l:
            out[row, (ell + 1) * n_tz : (ell + 2) * n_tz] = upper[ell]
    return out

@dataclass(frozen=True)
class PitchCollocationSurrogate:
    """A finite-difference discretization of the simplified operator on a pitch grid.

    Attributes:
        matrix: dense ``(Nalpha*Ntheta*Nzeta)`` square operator, index order
            ``(alpha, theta, zeta)``.
        to_nodal: ``(Nalpha, Nxi)`` Legendre Vandermonde ``P_L(xi_k)`` -- turns
            dkx's Legendre coefficients into values on the pitch grid.
        to_modal: its pseudo-inverse, ``(Nxi, Nalpha)``.
        shape: ``(Nalpha, Ntheta, Nzeta)``.
        alpha: the pitch-angle nodes.
    """

    matrix: np.ndarray
    to_nodal: np.ndarray
    to_modal: np.ndarray
    shape: tuple[int, int, int]
    alpha: np.ndarray

    def nodal(self, modal: np.ndarray) -> np.ndarray:
        """Legendre coefficients ``(Nxi, T, Z)`` -> grid values ``(Nalpha, T, Z)``."""
        n_xi = self.to_nodal.shape[1]
        v = np.asarray(modal).reshape(n_xi, self.shape[1], self.shape[2])
        return np.einsum("al,ltz->atz", self.to_nodal, v).reshape(-1)

    def modal(self, nodal: np.ndarray) -> np.ndarray:
        """Grid values ``(Nalpha, T, Z)`` -> Legendre coefficients ``(Nxi, T, Z)``."""
        g = np.asarray(nodal).reshape(self.shape)
        return np.einsum("la,atz->ltz", self.to_modal, g).reshape(-1)

def _legendre_vandermonde(xi: np.ndarray, n_l: int) -> np.ndarray:
    out = np.zeros((xi.size, n_l))
    out[:, 0] = 1.0
    if n_l > 1:
        out[:, 1] = xi
    for ell in range(1, n_l - 1):
        out[:, ell + 1] = ((2 * ell + 1) * xi * out[:, ell] - ell * out[:, ell - 1]) / (ell + 1)
    return out

def _uniform_spacing(ddx: np.ndarray, n: int) -> float:
    """Recover the grid spacing of a centered periodic first-derivative matrix.

    ``uniform_periodic_diff_matrices`` builds circulant matrices; the centered
    schemes dkx uses have first-neighbour weight ``1/2`` (3-point) or ``2/3``
    (5-point), so the spacing follows from the ``+1`` entry.
    """
    if n == 1:
        return 1.0
    row = np.asarray(ddx)[0]
    n_nonzero = int(np.count_nonzero(np.abs(row) > 1e-13))
    if n_nonzero == 2:
        return 0.5 / float(row[1])
    if n_nonzero == 4:
        return (2.0 / 3.0) / float(row[1])
    raise ValueError(
        "the pitch-collocation surrogate needs a 3- or 5-point centered angular "
        f"scheme to recover the grid spacing from, got {n_nonzero} nonzero weights"
    )

def pitch_collocation_surrogate(
    op: KineticOperator,
    *,
    species: int = 0,
    speed: int = 0,
    n_alpha: int | None = None,
    angular_stencil: str = "up1",
    pitch_stencil: str | None = None,
) -> PitchCollocationSurrogate:
    r"""Discretize the *same continuum operator* on a pitch-angle collocation grid.

    Per ``(species, speed)`` the simplified operator is, in continuum form,

    .. math::

        K f = a \left[ \xi\, \mathbf{b}\cdot\nabla f
                       - \frac{1-\xi^2}{2}\,
                         (\mathbf{b}\cdot\nabla \ln B)\, \partial_\xi f \right]
              + w^{E}_\theta \partial_\theta f + w^{E}_\zeta \partial_\zeta f
              + C f ,

    with ``a = x sqrt(That/mHat)`` and ``C`` diagonal in the Legendre index.
    dkx discretizes it Legendre-*modally* in pitch; this builds the alternative
    in which pitch is a *grid*, uniform in ``alpha`` with ``xi = cos alpha``
    (half-index, so the ends where the mirror speed vanishes are not nodes), and
    the streaming/mirror advection is upwind-differenced in all three of
    ``(alpha, theta, zeta)``.  ``C`` is carried across exactly, as
    ``V diag(c_L) V^+`` with ``V`` the Legendre Vandermonde -- dense in pitch but
    tiny, and it keeps the comparison about the *advection* discretization
    alone.

    Coupled to dkx's Legendre state by ``V`` and ``V^+``, this is the classical
    "low-order finite-difference preconditioner for a spectral operator"
    construction (S. A. Orszag, J. Comput. Phys. **37**, 70 (1980); M. O. Deville
    & E. H. Mund, J. Comput. Phys. **60**, 517 (1985)).  What it is *for* here is
    the measurement in this module's docstring: it is the discretization in
    which an alternating line block-Jacobi smoother converges, and dkx's is not.

    Args:
        op: the full recycled-Krylov operator (only the simplified part is
            used).
        species: species index.
        speed: speed-node index.
        n_alpha: pitch-grid size; defaults to ``op.n_xi``.
        angular_stencil: key of :data:`UPWIND_STENCILS` for ``theta``/``zeta``.
        pitch_stencil: key for ``alpha``; defaults to ``angular_stencil``.

    Returns:
        A :class:`PitchCollocationSurrogate`.  Dense: only for analysis on small
        decks.
    """
    simplified = simplified_operator(op)
    n_theta, n_zeta, n_xi = op.n_theta, op.n_zeta, op.n_xi
    n_a = int(n_alpha or n_xi)

    a_scale = float(
        np.asarray(op.x)[speed]
        * np.sqrt(np.asarray(op.t_hat)[species] / np.asarray(op.m_hat)[species])
    )
    b_hat = np.asarray(op.b_hat)
    v_theta = np.asarray(op.b_hat_sup_theta) / b_hat
    v_zeta = np.asarray(op.b_hat_sup_zeta) / b_hat
    b_grad_ln_b = (
        np.asarray(op.b_hat_sup_theta) * np.asarray(op.db_hat_dtheta)
        + np.asarray(op.b_hat_sup_zeta) * np.asarray(op.db_hat_dzeta)
    ) / b_hat**2
    if op.with_exb:
        coef_theta, coef_zeta = (np.asarray(c) for c in op._exb_coefficients())
    else:
        coef_theta = np.zeros_like(b_hat)
        coef_zeta = np.zeros_like(b_hat)
    coll = np.asarray(simplified.pas.coef)[species, speed][:n_xi]

    alpha = np.pi * (2 * np.arange(n_a) + 1) / (2 * n_a)
    xi = np.cos(alpha)
    h_alpha = np.pi / n_a
    h_theta = 2.0 * np.pi / n_theta
    h_zeta = _uniform_spacing(np.asarray(op.ddzeta), n_zeta)

    # advection velocities; d(alpha)/ds follows from d(xi)/ds and xi = cos alpha
    w_alpha = a_scale * (np.sin(alpha)[:, None, None] / 2.0) * b_grad_ln_b[None]
    w_theta = a_scale * xi[:, None, None] * v_theta[None] + coef_theta[None]
    w_zeta = a_scale * xi[:, None, None] * v_zeta[None] + coef_zeta[None]

    pitch_stencil = pitch_stencil or angular_stencil
    pairs = {
        0: stencil_matrices(n_a, h_alpha, pitch_stencil, periodic=False),
        1: stencil_matrices(n_theta, h_theta, angular_stencil, periodic=True),
        2: stencil_matrices(n_zeta, h_zeta, angular_stencil, periodic=True),
    }
    eye = {0: np.eye(n_a), 1: np.eye(n_theta), 2: np.eye(n_zeta)}

    def lift(axis: int, mat: np.ndarray) -> np.ndarray:
        factors = [eye[0], eye[1], eye[2]]
        factors[axis] = mat
        return np.kron(factors[0], np.kron(factors[1], factors[2]))

    shape = (n_a, n_theta, n_zeta)
    size = n_a * n_theta * n_zeta
    matrix = np.zeros((size, size))
    for axis, field in ((0, w_alpha), (1, w_theta), (2, w_zeta)):
        flat = np.broadcast_to(field, shape).reshape(-1)
        back, fwd = pairs[axis]
        matrix += (flat * (flat > 0))[:, None] * lift(axis, back)
        matrix += (flat * (flat <= 0))[:, None] * lift(axis, fwd)

    vander = _legendre_vandermonde(xi, n_xi)
    pinv = np.linalg.pinv(vander)
    matrix += lift(0, vander @ np.diag(coll) @ pinv)
    return PitchCollocationSurrogate(
        matrix=matrix, to_nodal=vander, to_modal=pinv, shape=shape, alpha=alpha
    )

def line_smoother_spectral_radius(
    matrix: np.ndarray, shape: tuple[int, ...], *, omega: float = 1.0, floor: float = 1e-12
) -> float:
    """Spectral radius of an alternating line block-Jacobi error propagator.

    Builds ``S = prod_axes (I - omega M_axis^{-1} A)`` with ``M_axis`` the block
    diagonal of ``A`` whose blocks run along that axis -- one exact line solve
    per coordinate, applied multiplicatively, which is the standard remedy for
    an advection whose direction is not grid-aligned (Trottenberg et al.,
    sections 5.1 and 7.4) -- and returns ``rho(S)``.

    ``rho(S) < 1`` is necessary for *any* multigrid cycle built on this
    relaxation: a coarse-grid correction cannot rescue a divergent smoother.

    Dense, ``O(N^3)``: for analysis on small decks only.
    """
    size = matrix.shape[0]
    prop = np.eye(size)
    for axis in range(len(shape)):
        m = shape[axis]
        rest = tuple(s for i, s in enumerate(shape) if i != axis)
        n_rest = int(np.prod(rest)) if rest else 1
        blocks = np.moveaxis(
            matrix.reshape(tuple(shape) + tuple(shape)),
            (axis, len(shape) + axis),
            (0, 1),
        ).reshape((m, m, n_rest, n_rest))
        blocks = np.einsum("abrr->rab", blocks)
        blocks = blocks + floor * np.abs(blocks).sum(axis=(1, 2)).max() * np.eye(m)[None]
        rhs = (matrix @ prop).reshape(tuple(shape) + (size,))
        rhs = np.moveaxis(rhs, axis, 0).reshape(m, n_rest, size).transpose(1, 0, 2)
        step = np.linalg.solve(blocks, rhs).transpose(1, 0, 2)
        step = np.moveaxis(step.reshape((m,) + rest + (size,)), 0, axis)
        prop = prop - omega * step.reshape(size, size)
    return float(np.abs(np.linalg.eigvals(prop)).max())

