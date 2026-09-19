"""Assembling the kinetic operator as a sparse matrix, from products with it.

The operator is matrix-free, and the sparse direct route wants a matrix. Reading
one column per operator application costs ``total_size`` applications --
633,604 on the ``Nxi = 120, Nx = 16`` deck that motivated this -- which is why
:func:`dkx.solve.materialize_csr` is confined to small decks and the direct
route is guarded off above ``max_dense_size``.

The couplings are known, so the matrix can be recovered from far fewer products
(:func:`solvax.compression.matrix_from_products`). Per f-block row
``(s, x, l, theta, zeta)`` the operator reaches:

* ``|l' - l| <= 2`` at the same ``(s, x)`` and the angular stencil of
  ``ddtheta``/``ddzeta`` -- streaming, mirror, ExB and the magnetic drifts;
* every ``(s', x')`` at the same ``(theta, zeta)`` and ``|l' - l| <= 2`` -- the
  collision operator is dense in speed and couples species, and the ``E_r``
  speed derivative is dense in speed through ``ddx``.

That is about 200 entries per row rather than 633,604, and the number of
products is the number of column groups, which is near that bound.

The border is not recovered this way. Its rows are dense, and a dense row makes
every pair of columns conflict, which would force one product per column and
undo the method. :func:`dkx.coarse_precond._materialize_borders` already probes
``B`` and ``C`` exactly with ``extra_size`` products, so the bordered matrix is
assembled from those pieces around the recovered f-block.

The pattern must cover every coupling: an entry outside it lands on another
column of the same group and corrupts that one too. The recovery is therefore
checked against the operator on random vectors before it is returned, and the
check is not optional -- a wrong matrix factors successfully and answers a
different question.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dkx.drift_kinetic import KineticOperator

#: Legendre half-bandwidth the pattern assumes: streaming and mirror couple
#: ``l +- 1``, the magnetic drifts ``l +- 2``.
_L_BANDWIDTH = 2


@dataclass(frozen=True)
class AssembledOperator:
    """The bordered operator as CSR, with the evidence that it is the operator.

    Attributes:
        matrix: scipy CSR of the full bordered operator, ``total_size`` square.
        products: how many operator applications the recovery cost.
        relative_error: largest relative difference against the operator on
            random vectors; the recovery is rejected above ``tolerance``.
    """

    matrix: object
    products: int
    relative_error: float


def _angular_pattern(op: KineticOperator, sparse):
    """Nonzero pattern of one angular block: what ``theta``/``zeta`` reach."""
    eye_theta = sparse.identity(op.n_theta, format="csr", dtype=bool)
    eye_zeta = sparse.identity(op.n_zeta, format="csr", dtype=bool)
    eye = sparse.kron(eye_theta, eye_zeta, format="csr").astype(bool)
    # The derivatives act on one angle each; the block they build is their
    # Kronecker lift onto (theta, zeta).
    d_theta = sparse.kron(
        sparse.csr_matrix(np.asarray(op.ddtheta) != 0.0), eye_zeta, format="csr"
    ).astype(bool)
    d_zeta = sparse.kron(
        eye_theta, sparse.csr_matrix(np.asarray(op.ddzeta) != 0.0), format="csr"
    ).astype(bool)
    # Each term applies one derivative to the block; the union covers them all,
    # and squaring covers a term that composes two.
    return (eye + d_theta + d_zeta).astype(bool)


def _cyclic_radius(matrix) -> int:
    """How far a derivative reaches on its periodic grid, in points."""
    pattern = np.asarray(matrix) != 0.0
    n = pattern.shape[0]
    rows, columns = np.nonzero(pattern)
    distance = np.abs(rows - columns)
    return int(np.max(np.minimum(distance, n - distance)), ) if rows.size else 0


def _angular_stride(n: int, radius: int) -> int:
    """Smallest stride that divides the grid and clears twice the stencil.

    The grid is periodic, so members of a residue class are a whole stride
    apart only when the stride divides the grid; two columns share a row when
    they sit within twice the stencil radius of each other.
    """
    for stride in range(1, n + 1):
        if n % stride == 0 and stride > 2 * radius:
            return stride
    return n


def f_block_pattern(op: KineticOperator, *, l_bandwidth: int = _L_BANDWIDTH):
    """Sparsity pattern of the f-block, as a boolean CSR of ``(f_size, f_size)``.

    A superset: it is built from which indices each term can reach, not from
    values, so a coefficient that happens to vanish still occupies a slot.
    """
    import scipy.sparse as sparse  # noqa: PLC0415

    n_s, n_x, n_xi, n_theta, n_zeta = op.f_shape
    n_tz = n_theta * n_zeta
    angular = _angular_pattern(op, sparse)
    dense_tz = sparse.identity(n_tz, format="csr", dtype=bool)

    # Legendre couplings, as a banded (n_xi, n_xi) pattern.
    offsets = range(-l_bandwidth, l_bandwidth + 1)
    l_band = sparse.diags(
        [np.ones(n_xi - abs(k), dtype=bool) for k in offsets],
        list(offsets), format="csr", dtype=bool,
    )  # fmt: skip
    # Speed and species: dense, at the same angular point.
    sx = n_s * n_x
    sx_dense = sparse.csr_matrix(np.ones((sx, sx), dtype=bool))
    sx_eye = sparse.identity(sx, format="csr", dtype=bool)

    # (s, x) outer index, then l, then (theta, zeta): the state's own order.
    same_sx = sparse.kron(sx_eye, sparse.kron(l_band, angular, format="csr"), format="csr")
    across_sx = sparse.kron(sx_dense, sparse.kron(l_band, dense_tz, format="csr"), format="csr")
    pattern = (same_sx + across_sx).astype(bool).tocsr()
    pattern.eliminate_zeros()
    return pattern


def f_block_groups(op: KineticOperator, *, l_bandwidth: int = _L_BANDWIDTH) -> list[np.ndarray]:
    """Column groups for the f-block pattern, from the operator's own strides.

    A greedy colouring would have to walk the pattern's row density times its
    nonzero count, which is minutes of Python at the sizes this exists for. The
    strides give the same property directly: two columns of a group differ by
    more than the angular stencil, by more than the Legendre bandwidth, or sit
    at the same ``(theta, zeta)`` with different ``(s, x)`` -- which the collision
    coupling forbids, so ``(s, x)`` enters the group key.
    """
    n_s, n_x, n_xi, n_theta, n_zeta = op.f_shape
    stride_theta = _angular_stride(n_theta, _cyclic_radius(op.ddtheta))
    stride_zeta = _angular_stride(n_zeta, _cyclic_radius(op.ddzeta))
    # Legendre is a band, not a circle, so a stride past the bandwidth is enough.
    stride_l = min(2 * l_bandwidth + 1, n_xi)
    index = np.arange(op.f_size).reshape(n_s * n_x, n_xi, n_theta, n_zeta)
    groups: list[np.ndarray] = []
    for sx in range(n_s * n_x):
        for l0 in range(stride_l):
            for t0 in range(stride_theta):
                for z0 in range(stride_zeta):
                    block = index[sx, l0::stride_l, t0::stride_theta, z0::stride_zeta]
                    if block.size:
                        groups.append(block.reshape(-1))
    return groups


def assemble_operator(
    op: KineticOperator, *, tolerance: float = 1.0e-10, samples: int = 3
) -> AssembledOperator:
    """Recover the bordered operator as CSR and check it against the operator.

    Args:
        op: the kinetic operator.
        tolerance: largest relative difference accepted on random vectors.
        samples: how many random vectors the check uses.

    Returns:
        The assembled operator, its cost in products, and the check's result.

    Raises:
        RuntimeError: if the recovered matrix does not reproduce the operator,
            which means the pattern missed a coupling.
    """
    import scipy.sparse as sparse  # noqa: PLC0415
    from solvax.compression import matrix_from_products, verify_products  # noqa: PLC0415

    from dkx.coarse_precond import _materialize_borders  # noqa: PLC0415

    f_size, extra = op.f_size, op.extra_size
    shape = op.f_shape

    def apply_f(vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(op.apply_f(jnp.reshape(vector, shape)), (f_size,))

    apply_f = jax.jit(apply_f)
    pattern = f_block_pattern(op)
    groups = f_block_groups(op)
    block = matrix_from_products(apply_f, pattern, groups=groups)
    block.eliminate_zeros()

    if extra:
        b_cols, c_rows = _materialize_borders(op)
        matrix = sparse.bmat(
            [
                [block, sparse.csr_matrix(np.asarray(b_cols))],
                [sparse.csr_matrix(np.asarray(c_rows)), sparse.csr_matrix((extra, extra))],
            ],
            format="csr",
        )
        products = len(groups) + 2 * extra
    else:
        matrix, products = block, len(groups)

    error = verify_products(matrix, jax.jit(op.apply), samples=samples)
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"the assembled matrix does not reproduce the operator (relative "
            f"difference {error:.3e} > {tolerance:.1e}). The pattern in "
            f"dkx.assembly missed a coupling; widen it rather than raising the "
            f"tolerance, because the factorization of a matrix that is not the "
            f"operator succeeds and answers a different question."
        )
    return AssembledOperator(matrix=matrix, products=products, relative_error=float(error))
