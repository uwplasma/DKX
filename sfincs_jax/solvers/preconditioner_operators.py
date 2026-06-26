"""Operator-shaping helpers for v3 preconditioners.

These functions build simplified ``V3FullSystemOperator`` variants used as
preconditioner matrices. They are intentionally pure transformations of the
operator dataclasses: no solve state, caches, environment variables, or sparse
factorization happens here. Keeping them outside ``v3_driver`` makes the
Fortran-reduced/PETSc-style ``Pmat`` behavior directly testable.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.operators.profile_response.system import V3FullSystemOperator

__all__ = [
    "block_diagonal_only",
    "diagonal_only",
    "_build_rhsmode1_preconditioner_operator_fortran_reduced",
    "_build_rhsmode1_preconditioner_operator_point",
    "_build_rhsmode1_preconditioner_operator_theta_dd",
    "_build_rhsmode1_preconditioner_operator_theta_line",
    "_build_rhsmode1_preconditioner_operator_zeta_dd",
    "_build_rhsmode1_preconditioner_operator_zeta_line",
    "_build_transport_preconditioner_operator_fortran_reduced",
    "_build_transport_preconditioner_operator_point",
]


def diagonal_only(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return a diagonal-only copy of a square matrix."""

    return jnp.diag(jnp.diag(matrix))


def block_diagonal_only(matrix: jnp.ndarray, block: int) -> jnp.ndarray:
    """Return a block-diagonal copy of a square matrix."""

    if int(block) <= 1:
        return diagonal_only(matrix)
    matrix_np = np.asarray(matrix, dtype=np.float64)
    n = int(matrix_np.shape[0])
    mask = np.zeros((n, n), dtype=bool)
    for start in range(0, n, int(block)):
        end = min(n, start + int(block))
        mask[start:end, start:end] = True
    matrix_np = np.where(mask, matrix_np, 0.0)
    return jnp.asarray(matrix_np, dtype=matrix.dtype)


_diag_only = diagonal_only
_block_diag_only = block_diagonal_only


def _build_rhsmode1_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for point-block preconditioning.

    This is the original cheap RHSMode=1 preconditioner: it retains local x/L
    couplings and collisions while dropping theta/zeta derivative couplings
    (streaming, ExB, and magnetic-drift derivatives) by diagonalizing the derivative
    matrices.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_fortran_reduced(
    op: V3FullSystemOperator,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
) -> V3FullSystemOperator:
    """Return a Fortran-v3-style reduced global RHSMode=1 preconditioner operator.

    SFINCS Fortran v3's default PETSc preconditioner is not a local point
    smoother: it keeps the angular streaming/drift derivatives and the global
    source/constraint rows, while simplifying selected radial, pitch-angle, and
    species couplings. This function provides the first SFINCS-JAX operator
    shaping for that route. It intentionally preserves theta/zeta coupling and
    only applies the x-diagonal simplification to terms that expose radial
    derivative matrices directly in the current JAX operator tree.

    ``preconditioner_species`` and ``preconditioner_x`` are applied to the full
    Fokker-Planck collision tensor when it is available. ``preconditioner_x_min_l``
    follows the Fortran rule: radial simplification is only applied to rows with
    Legendre index ``L >= preconditioner_x_min_L``. ``preconditioner_xi`` follows
    v3's matrix-0 rule by dropping the collisionless ``L±2`` pitch couplings
    while preserving diagonal-in-``L`` drift/Er terms and streaming ``L±1`` terms.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    fp = fblock.fp
    if fp is not None and hasattr(fp, "mat"):
        mat = jnp.asarray(fp.mat)
        if int(preconditioner_species) > 0 and mat.ndim == 5:
            species_eye = jnp.eye(int(op.n_species), dtype=mat.dtype)
            mat = mat * species_eye[:, :, None, None, None]
        if int(preconditioner_x) > 0 and mat.ndim == 5:
            n_x = int(op.n_x)
            row = jnp.arange(n_x)[:, None]
            col = jnp.arange(n_x)[None, :]
            if int(preconditioner_x) == 1:
                x_mask = row == col
            elif int(preconditioner_x) == 2:
                x_mask = col >= row
            elif int(preconditioner_x) in {3, 5}:
                x_mask = jnp.abs(row - col) <= 1
            elif int(preconditioner_x) == 4:
                x_mask = (col == row) | (col == row + 1)
            else:
                x_mask = row == col
            if int(preconditioner_x_min_l) > 0:
                ell = jnp.arange(int(mat.shape[2]), dtype=jnp.int32)
                l_gate = ell >= int(preconditioner_x_min_l)
                x_mask = jnp.where(l_gate[:, None, None], x_mask[None, :, :], True)
                mat = mat * x_mask[None, None, :, :, :]
            else:
                mat = mat * x_mask[None, None, None, :, :]
        fp = replace(fp, mat=mat)

    drop_l2 = int(preconditioner_xi) > 0

    def _maybe_drop_l2(term):
        if term is None or not drop_l2 or not hasattr(term, "drop_l2_couplings"):
            return term
        return replace(term, drop_l2_couplings=True)

    term_replacements = {
        "fp": fp,
    }
    for name in ("magdrift_theta", "magdrift_zeta", "magdrift_xidot", "er_xidot"):
        if hasattr(fblock, name):
            term_replacements[name] = _maybe_drop_l2(getattr(fblock, name))

    er_xdot = getattr(fblock, "er_xdot", None)
    if er_xdot is not None:
        replacements = {}
        if int(preconditioner_x) > 0 and int(preconditioner_x_min_l) <= 0:
            replacements["ddx_plus"] = _diag_only(er_xdot.ddx_plus)
            replacements["ddx_minus"] = _diag_only(er_xdot.ddx_minus)
        if drop_l2 and hasattr(er_xdot, "drop_l2_couplings"):
            replacements["drop_l2_couplings"] = True
        if replacements:
            er_xdot = replace(er_xdot, **replacements)
    if hasattr(fblock, "er_xdot"):
        term_replacements["er_xdot"] = er_xdot

    fblock_pc = replace(
        fblock,
        # Keep collisionless ddtheta/ddzeta, ExB, magnetic-drift theta/zeta,
        # collisions, source rows, and constraint rows globally coupled.
        **term_replacements,
    )
    return replace(op, fblock=fblock_pc)


def _build_transport_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified transport operator for point-block preconditioning.

    This mirrors `_build_rhsmode1_preconditioner_operator_point` but does not
    require RHSMode=1, since RHSMode=2/3 transport solves reuse the same operator
    structure with different right-hand sides.
    """
    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_transport_preconditioner_operator_fortran_reduced(
    op: V3FullSystemOperator,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
    keep_theta_zeta: bool = True,
) -> V3FullSystemOperator:
    """Return a Fortran-v3-style reduced transport preconditioner operator.

    SFINCS Fortran v3 uses the true matrix as ``Amat`` and a distinct
    ``whichMatrix=0`` reduced matrix as ``Pmat`` for PETSc.  This transport
    helper applies the same x/species/pitch simplifications as the RHSMode=1
    reduced operator but works for RHSMode=2/3.  By default it keeps theta/zeta
    derivative couplings, matching the Fortran v3 defaults
    ``preconditioner_theta=0`` and ``preconditioner_zeta=0``.  Set
    ``keep_theta_zeta=False`` only for a smaller diagnostic Pmat that explicitly
    drops the angular streaming/drift graph.
    """

    rhs_mode_original = int(op.rhs_mode)
    op_rhs1 = replace(op, rhs_mode=1)
    op_pc = _build_rhsmode1_preconditioner_operator_fortran_reduced(
        op_rhs1,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )
    op_pc = replace(op_pc, rhs_mode=rhs_mode_original)
    if not bool(keep_theta_zeta):
        op_pc = _build_transport_preconditioner_operator_point(op_pc)
    return op_pc


def _build_rhsmode1_preconditioner_operator_theta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for theta-line preconditioning.

    Keep full theta derivative couplings but drop zeta derivative couplings. This enables
    a significantly stronger preconditioner than point-block Jacobi, while remaining
    much cheaper than a full (theta,zeta)-coupled preconditioner.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = fblock.exb_theta
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = fblock.magdrift_theta
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_theta_dd(
    op: V3FullSystemOperator, *, block: int
) -> V3FullSystemOperator:
    """Return a theta-block domain-decomposition operator for preconditioning.

    This operator shaping is used by RHSMode=1 and RHSMode=2/3 transport solves.
    """

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_block_diag_only(fblock.collisionless.ddtheta, block),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = fblock.exb_theta
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = fblock.magdrift_theta
    if mag_theta is not None:
        mag_theta = replace(
            mag_theta,
            ddtheta_plus=_block_diag_only(mag_theta.ddtheta_plus, block),
            ddtheta_minus=_block_diag_only(mag_theta.ddtheta_minus, block),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_zeta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for zeta-line preconditioning.

    Keep full zeta derivative couplings but drop theta derivative couplings. This is the
    zeta-analog of `_build_rhsmode1_preconditioner_operator_theta_line`.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = fblock.exb_zeta
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = fblock.magdrift_zeta
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_zeta_dd(
    op: V3FullSystemOperator, *, block: int
) -> V3FullSystemOperator:
    """Return a zeta-block domain-decomposition operator for preconditioning.

    This operator shaping is used by RHSMode=1 and RHSMode=2/3 transport solves.
    """

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_block_diag_only(fblock.collisionless.ddzeta, block),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = fblock.exb_zeta
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = fblock.magdrift_zeta
    if mag_zeta is not None:
        mag_zeta = replace(
            mag_zeta,
            ddzeta_plus=_block_diag_only(mag_zeta.ddzeta_plus, block),
            ddzeta_minus=_block_diag_only(mag_zeta.ddzeta_minus, block),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)

