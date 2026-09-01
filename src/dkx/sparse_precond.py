"""Sparse-elimination inverse of the SFINCS-simplified operator.

A preconditioner route of the recycled Krylov solve.

:func:`dkx.coarse_precond.build_coarse_preconditioner` inverts the simplified operator
*exactly* by a block-Thomas recursion over the Legendre index ``L`` whose
blocks are ``(Ntheta*Nzeta)`` square and **dense**.  The blocks are not dense in
the operator -- :meth:`KineticOperator.legendre_blocks` builds each one as

.. math::

    \\alpha(\\theta,\\zeta)\\, (D_\\theta \\otimes I)
    + \\beta(\\theta,\\zeta)\\, (I \\otimes D_\\zeta) + \\mathrm{diag},

with the 3- or 5-point centred stencils of ``createGrids.F90``, so about 9 of
1121 entries per row are nonzero on a ``19 x 59`` surface.  Eliminating ``L``
first is what fills them in: the Schur complement
``D_l - L_l D_{l-1}^{-1} U_{l-1}`` is dense even when every input block is
banded.  The cost of that choice is
``O(Nxi Nspecies Nx (Ntheta Nzeta)^3)`` work and
``O(Nxi Nspecies Nx (Ntheta Nzeta)^2)`` memory -- 845 Gflop and 16.9 GB of
bands on the ``sfincsPaperFigure3`` two-species deck.

SFINCS does not eliminate in that order.  It assembles the same simplified
operator as one sparse PETSc matrix and lets MUMPS/SuperLU_DIST choose a
fill-reducing ordering.  This module does the same thing in ``dkx``: assemble
the simplified operator in CSR from the coefficients ``legendre_blocks``
already uses, factor it on the host with SuperLU, and apply it through
``jax.pure_callback``.

Three facts make this admissible where a host callback would not otherwise be:

* The ``(species, x)`` subsystems are **uncoupled** in the simplified operator,
  so this is ``Nspecies * Nx`` independent factorizations of ``Nxi *
  Ntheta * Nzeta`` rows each, not one factorization of the whole system.
* A preconditioner is never differentiated.  The recycled Krylov implicit-diff
  wrapper differentiates the *solution*; the preconditioner enters only the forward and
  transposed linear solves, whose derivatives the implicit function theorem
  supplies.
* The route is opt-in (``solve(preconditioner="sparse")``) and refuses under
  ``jit``/``vmap``/``grad`` of the operator leaves, where the host assembly has
  no values to read.

What it does **not** change is the operator being inverted.  Every
simplification, floor, mask pin and ``l = 0`` null-space pin of
:func:`~dkx.coarse_precond.build_coarse_preconditioner` is reproduced here, so the two
routes are the same linear map up to factorization round-off --
``tests/test_sparse_precond.py`` pins that agreement.  The ``l = 0`` pin is the
one term that would destroy sparsity (it is a dense rank-one outer product on
the ``l = 0`` diagonal block), so it is applied by an exact Sherman-Morrison
correction around the factorization instead of being assembled into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from dkx.drift_kinetic import KineticOperator

__all__ = [
    "SparseSimplified",
    "build_sparse_f_inverse",
    "build_sparse_preconditioner",
    "simplified_subsystem_csr",
]


def _np(x) -> np.ndarray:
    """Host copy of an operator leaf, refusing tracers with a usable message."""
    if isinstance(x, jax.core.Tracer):
        raise NotImplementedError(
            "the sparse tier-2 preconditioner assembles its matrix on the host and "
            "cannot run with traced operator leaves (jit/vmap/grad over the "
            "operator); use preconditioner='coarse', which stays traceable."
        )
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class SparseSimplified:
    """Assembled subsystems plus the structural numbers worth reporting.

    ``matrices`` holds one CSR block per ``(species, x)`` subsystem in the flat
    ``s * n_x + x`` order the coarse route uses, each of size ``Nxi * Ntheta *
    Nzeta``.  ``pin`` holds the rank-one ``l = 0`` correction vectors
    ``(gamma * ones, c0)`` for that subsystem, or ``None`` where the adaptive
    pin switched itself off.
    """

    matrices: list  # list[scipy.sparse.csr_matrix]
    pin: list  # list[tuple[np.ndarray, np.ndarray] | None]
    n_xi: int
    n_tz: int

    @property
    def nnz(self) -> int:
        """Nonzeros across all subsystems, before factorization."""
        return int(sum(m.nnz for m in self.matrices))

    @property
    def dense_band_bytes(self) -> float:
        """What the dense block-Thomas bands of the same operator would cost."""
        return 3.0 * len(self.matrices) * self.n_xi * self.n_tz**2 * 8.0


def _tz_templates(op: KineticOperator):
    """``(A_theta, A_zeta, eye)`` sparse ``(TZ, TZ)`` angular templates.

    ``A_theta = kron(ddtheta, I_zeta)`` and ``A_zeta = kron(I_theta, ddzeta)``
    are exactly the matrices :meth:`KineticOperator.legendre_blocks` forms
    densely; taking them through ``scipy.sparse`` keeps only the stencil
    entries, which is the whole point of this module.
    """
    import scipy.sparse as sp  # lazy: matches the sparse direct optional-scipy policy

    d_theta = sp.csr_matrix(_np(op.ddtheta))
    d_zeta = sp.csr_matrix(_np(op.ddzeta))
    eye_t = sp.identity(op.n_theta, dtype=np.float64, format="csr")
    eye_z = sp.identity(op.n_zeta, dtype=np.float64, format="csr")
    return (
        sp.kron(d_theta, eye_z, format="csr"),
        sp.kron(eye_t, d_zeta, format="csr"),
        sp.identity(op.n_theta * op.n_zeta, dtype=np.float64, format="csr"),
    )


def _angular_pieces(op: KineticOperator):
    """Per-species streaming and mirror templates, and the ExB block.

    Mirrors :meth:`KineticOperator.legendre_blocks` term by term:
    ``stream`` carries the row-scaled ``d/dtheta`` and ``d/dzeta`` of parallel
    streaming, ``mirror`` is the diagonal mirror force, and ``exb`` is the
    ``E x B`` drift block, which is species-independent and diagonal in ``L``.
    """
    a_theta, a_zeta, eye = _tz_templates(op)
    import scipy.sparse as sp

    sqrt_t_over_m = np.sqrt(_np(op.t_hat) / _np(op.m_hat))  # (S,)
    b_hat = _np(op.b_hat)
    v_theta = (_np(op.b_hat_sup_theta) / b_hat).reshape(-1)  # (TZ,)
    v_zeta = (_np(op.b_hat_sup_zeta) / b_hat).reshape(-1)

    scale_theta = sp.diags(v_theta, format="csr")
    scale_zeta = sp.diags(v_zeta, format="csr")
    stream_1 = (scale_theta @ a_theta + scale_zeta @ a_zeta).tocsr()
    stream = [float(c) * stream_1 for c in sqrt_t_over_m]  # (S,) of (TZ,TZ)

    mirror_geom = _np(op.b_hat_sup_theta) * _np(op.db_hat_dtheta) + _np(
        op.b_hat_sup_zeta
    ) * _np(op.db_hat_dzeta)
    mirror_diag = -(mirror_geom / (2.0 * b_hat**2)).reshape(-1)  # (TZ,)
    mirror = [sp.diags(float(c) * mirror_diag, format="csr") for c in sqrt_t_over_m]

    if op.with_exb:
        denom = _np(op.fsab_hat2) if op.use_dkes_exb else b_hat**2
        factor = float(op.alpha) * float(op.delta) * 0.5 * float(
            op.dphi_hat_dpsi_hat_kinetic
        )
        d_hat = _np(op.d_hat)
        coef_theta = (factor * d_hat * _np(op.b_hat_sub_zeta) / denom).reshape(-1)
        coef_zeta = (-factor * d_hat * _np(op.b_hat_sub_theta) / denom).reshape(-1)
        exb = (
            sp.diags(coef_theta, format="csr") @ a_theta
            + sp.diags(coef_zeta, format="csr") @ a_zeta
        ).tocsr()
    else:
        exb = sp.csr_matrix((op.n_theta * op.n_zeta,) * 2, dtype=np.float64)
    return stream, mirror, exb, eye


def _diagonal_coefficients(op: KineticOperator) -> np.ndarray:
    """``(S, X, L)`` collision/floor diagonal the coarse route adds to each block.

    Reproduces :func:`dkx.coarse_precond.build_coarse_preconditioner`: pitch-angle
    scattering, the self-species x-diagonal reduction of a dense Fokker-Planck
    or improved-Sugama operator, and the probed ``includePhi1InCollisionOperator``
    diagonal, each masked by the ``Nxi_for_x`` truncation.
    """
    from dkx.coarse_precond import (  # noqa: PLC0415
        _collision_phi1_diagonal,
        _dense_collision_diagonal,
    )

    mask = _np(op._mask())  # (X, L)
    coef = np.zeros((op.n_species, op.n_x, op.n_xi), dtype=np.float64)
    if op.pas is not None:
        coef = coef + _np(op.pas.coef) * mask[None, :, :]
    for coll in (op.fp, op.sugama):
        if coll is not None:
            coef = coef + _np(_dense_collision_diagonal(coll.mat)) * mask[None, :, :]
    if op.fp_phi1 is not None:
        coef = coef + _np(_collision_phi1_diagonal(op)) * mask[None, :, :]
    return coef


def simplified_subsystem_csr(
    op: KineticOperator, s: int, ix: int, *, drop_l_coupling: bool = False
):
    """CSR of the simplified operator on one ``(species, x)`` subsystem.

    Rows and columns run in the ``(L, theta, zeta)`` order of the flat state,
    so the block is ``Nxi * Ntheta * Nzeta`` square.  The ``l = 0`` null-space
    pin is *not* included: it is dense, and :func:`build_sparse_f_inverse`
    applies it by Sherman-Morrison instead.
    """
    import scipy.sparse as sp

    stream, mirror, exb, eye = _angular_pieces(op)
    return _assemble(op, s, ix, stream, mirror, exb, eye, sp, drop_l_coupling)[0]


def _assemble(op, s, ix, stream, mirror, exb, eye, sp, drop_l_coupling):
    """Block-tridiagonal-in-L CSR assembly of one subsystem (shared inner loop).

    Returns ``(csr, band, l0_defect, scale)``.  The three scalars are the
    quantities :func:`dkx.coarse_precond.build_coarse_preconditioner` measures to size its
    regularization, computed at the same points in the assembly: ``band`` and
    ``l0_defect`` from the raw blocks, ``scale`` from the floored and mask-pinned
    diagonal.  Reproducing them here is what keeps the two routes the same map.
    """
    n_xi = op.n_xi
    mask = _np(op._mask())  # (X, L)
    x_val = float(_np(op.x)[ix])
    coef_lower = _np(op.xi_coupling_lower)
    coef_upper = _np(op.xi_coupling_upper)
    diag_coef = _diagonal_coefficients(op)[s, ix]  # (L,)

    # Invertibility floor and mask pins, exactly as the coarse route sizes them:
    # the floor is relative to the largest band entry of *this* subsystem.
    band = 0.0
    blocks: dict[tuple[int, int], object] = {}
    for ell in range(n_xi):
        row_mask = float(mask[ix, ell])
        diag = row_mask * exb + diag_coef[ell] * eye
        blocks[(ell, ell)] = diag.tocsr()
        if not drop_l_coupling and ell >= 1:
            c = float(coef_lower[ell])
            m = x_val * row_mask * float(mask[ix, ell - 1])
            blocks[(ell, ell - 1)] = (
                m * (c * stream[s] + (-c * (ell - 1.0)) * mirror[s])
            ).tocsr()
        if not drop_l_coupling and ell + 1 < n_xi:
            c = float(coef_upper[ell])
            m = x_val * row_mask * float(mask[ix, ell + 1])
            blocks[(ell, ell + 1)] = (
                m * (c * stream[s] + (c * (ell + 2.0)) * mirror[s])
            ).tocsr()
    for block in blocks.values():
        if block.nnz:
            band = max(band, float(np.abs(block.data).max()))
    band = band if band > 0.0 else 1.0

    # The constant-on-surface vector is the l=0 block's null vector, so its row
    # sums are the whole defect -- measured before the floor masks it.
    l0_defect = float(np.abs(np.asarray(blocks[(0, 0)].sum(axis=1)).reshape(-1)).max())

    from dkx.coarse_precond import _COARSE_DIAGONAL_FLOOR  # noqa: PLC0415

    floor = _COARSE_DIAGONAL_FLOOR * band
    diag_entries = []
    for ell in range(n_xi):
        shift = floor + (1.0 - float(mask[ix, ell]))  # mask pin -> identity rows
        blocks[(ell, ell)] = (blocks[(ell, ell)] + shift * eye).tocsr()
        diag_entries.append(blocks[(ell, ell)].diagonal())
    scale = float(np.mean(np.abs(np.concatenate(diag_entries))))
    scale = scale if scale > 0.0 else 1.0

    rows = [[blocks.get((r, c)) for c in range(n_xi)] for r in range(n_xi)]
    return sp.bmat(rows, format="csr"), band, l0_defect, scale


def assemble_simplified(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> SparseSimplified:
    """Assemble every ``(species, x)`` subsystem of the simplified operator.

    The rank-one ``l = 0`` pin is sized here with the same
    :func:`dkx.coarse_precond._l0_pin_gamma` the coarse route uses, but returned as its
    two vectors rather than added in: it is a dense outer product, and
    assembling it would fill the ``l = 0`` block completely.
    """
    import scipy.sparse as sp  # noqa: PLC0415

    from dkx.coarse_precond import _l0_pin_gamma  # noqa: PLC0415

    stream, mirror, exb, eye = _angular_pieces(op)
    matrices, bands, defects, scales = [], [], [], []
    for s in range(op.n_species):
        for ix in range(op.n_x):
            mat, band, defect, scale = _assemble(
                op, s, ix, stream, mirror, exb, eye, sp, drop_l_coupling
            )
            matrices.append(mat)
            bands.append(band)
            defects.append(defect)
            scales.append(scale)

    c0 = _np(op._fs_average_factor()).reshape(-1)  # (TZ,)
    gamma = _np(
        _l0_pin_gamma(
            jnp.asarray(np.asarray(defects)),
            jnp.asarray(np.asarray(bands)),
            jnp.asarray(np.asarray(scales)),
            jnp.asarray(c0),
        )
    ).reshape(-1)
    ones = np.ones((op.n_theta * op.n_zeta,), dtype=np.float64)
    pin = [None if g == 0.0 else (g * ones, c0) for g in gamma]
    return SparseSimplified(
        matrices=matrices,
        pin=pin,
        n_xi=op.n_xi,
        n_tz=op.n_theta * op.n_zeta,
    )


def build_sparse_f_inverse(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> tuple[Callable, Callable, SparseSimplified]:
    """``(a_inv, a_inv_t, assembled)`` — host sparse-LU inverse of the f-block.

    Each subsystem is factored once with SuperLU and applied through
    ``jax.pure_callback``; the rank-one ``l = 0`` pin is applied exactly by
    Sherman-Morrison around that factorization,

    ``(A + u v^T)^{-1} r = A^{-1} r - A^{-1} u (v^T A^{-1} r) / (1 + v^T A^{-1} u)``,

    which needs one extra triangular solve per subsystem, done once at build
    time for ``A^{-1} u``.
    """
    import scipy.sparse.linalg as spla  # noqa: PLC0415

    assembled = assemble_simplified(op, drop_l_coupling=drop_l_coupling)
    n_sub = len(assembled.matrices)
    width = assembled.n_xi * assembled.n_tz

    factors = [spla.splu(m.tocsc()) for m in assembled.matrices]
    # Sherman-Morrison carries, per subsystem and per direction.
    carry: list = []
    for lu, pin in zip(factors, assembled.pin):
        if pin is None:
            carry.append(None)
            continue
        u, v = pin
        u_full = np.zeros(width, dtype=np.float64)
        v_full = np.zeros(width, dtype=np.float64)
        u_full[: len(u)] = u  # the pin lives on the l = 0 block
        v_full[: len(v)] = v
        a_inv_u = lu.solve(u_full)
        at_inv_v = lu.solve(v_full, trans="T")
        carry.append((u_full, v_full, a_inv_u, at_inv_v))

    def _apply(v: np.ndarray, transpose: bool) -> np.ndarray:
        # ``v`` carries any number of leading batch axes (``schur_projected_precond``
        # vmaps the inverse over the border columns), so flatten them into one
        # right-hand-side axis and hand each subsystem all of its columns at once
        # -- one triangular sweep over many vectors, not one sweep each.
        a = np.asarray(v, dtype=np.float64)
        lead = a.shape[:-1]
        g = a.reshape(-1, n_sub, width)  # (R, subsystem, row)
        out = np.empty_like(g)
        trans = "T" if transpose else "N"
        for i, (lu, c) in enumerate(zip(factors, carry)):
            rhs = g[:, i, :].T  # (row, R)
            y = lu.solve(rhs, trans=trans)
            if c is not None:
                u_full, v_full, a_inv_u, at_inv_v = c
                if transpose:
                    # (A + u v^T)^T = A^T + v u^T
                    denom = 1.0 + float(u_full @ at_inv_v)
                    y = y - np.outer(at_inv_v, (u_full @ y) / denom)
                else:
                    denom = 1.0 + float(v_full @ a_inv_u)
                    y = y - np.outer(a_inv_u, (v_full @ y) / denom)
            out[:, i, :] = y.T
        return out.reshape(*lead, n_sub * width)

    def _make(transpose: bool) -> Callable:
        def apply(v: jnp.ndarray) -> jnp.ndarray:
            shape = jax.ShapeDtypeStruct(v.shape, jnp.float64)
            return jax.pure_callback(
                lambda a: _apply(a, transpose),
                shape,
                v,
                vmap_method="broadcast_all",
            )

        return apply

    return _make(False), _make(True), assembled


def build_sparse_preconditioner(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Drop-in sparse replacement for :func:`dkx.coarse_precond.build_coarse_preconditioner`.

    Same simplified operator, same regularization, same exact elimination of the
    bordered constraint / ``Phi1`` rows — only the inner f-block inverse changes
    from a dense block-Thomas factorization to a host sparse LU.
    """
    from dkx.coarse_precond import (  # noqa: PLC0415
        _materialize_borders,
        _materialize_full_border,
        schur_projected_precond,
    )

    a_inv, a_inv_t, _ = build_sparse_f_inverse(op, drop_l_coupling=drop_l_coupling)
    if op.include_phi1:
        b_cols, c_rows, d_block = _materialize_full_border(op)
        return (
            schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block),
            schur_projected_precond(a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T),
        )
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = _materialize_borders(op)
    return (
        schur_projected_precond(a_inv, b_cols, c_rows),
        schur_projected_precond(a_inv_t, c_rows.T, b_cols.T),
    )
