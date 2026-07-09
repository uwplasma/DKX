"""Direct Legendre block-tridiagonal solve for RHSMode=3 (monoenergetic) transport systems.

For RHSMode=3 the v3 kinetic operator acts on ``f(theta, zeta, xi)`` expanded in
Legendre modes ``l = 0..Nxi-1`` (``Nx = 1``, single species, pitch-angle-scattering
collisions, DKES ExB drift). Every term couples only ``l-1, l, l+1``:

- parallel streaming/mirror: factors ``l/(2l-1)`` and ``(l+1)/(2l+3)``,
- Lorentz pitch-angle scattering: diagonal ``nu * l(l+1)/2``,
- DKES ExB drift: diagonal in ``l``.

The full solver matrix is therefore exactly block tridiagonal over ``l`` with dense
``(Ntheta*Nzeta)^2`` blocks, bordered by the constraintScheme=2 machinery
(one source column ``B`` injected into the ``l=0`` rows and one constraint row ``C``
enforcing a vanishing flux-surface average of the ``l=0`` component) which removes
the null space of constants on the flux surface.

This module extracts the Legendre blocks by probing the *existing* matrix-free
``apply_v3_full_system_operator`` (so parity with the iterative path is by
construction), absorbs the bordering with an exact rank-one trick, and solves the
system directly with :func:`solvax.direct.block_thomas_factor` /
:func:`solvax.direct.block_thomas_solve` for all transport right-hand sides at once.

Null-space handling (exact rank-one absorbed bordering)
--------------------------------------------------------
The bordered system is::

    A f + B s = b_f,      C f = r_c,

where ``A`` (block tridiagonal) is singular: its null space is spanned by constants
on the surface in the ``l=0`` block. Define ``A~ = A + gamma * B C`` for any
``gamma != 0`` (a rank-one update confined to the ``l=0`` diagonal block, so ``A~``
stays block tridiagonal and is nonsingular whenever the bordered system is). Then::

    f = y - (s - gamma r_c) z,   y = A~^{-1} b_f,   z = A~^{-1} B,
    s = gamma r_c + (C y - r_c) / (C z),

which is algebraically exact for any ``gamma`` -- the arbitrary scale only affects
conditioning, never the solution. This reuses the operator's own constraint/source
machinery (``B`` and ``C`` are probed from the same matvec) instead of overwriting
matrix rows, so the computed state vector -- including the source unknown --
matches the existing path to solver precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# solvax is optional until its PyPI release (CI installs it from git): keep
# this module importable without it and raise a clear error on first use.
try:
    from solvax.direct import (
        BlockTridiagFactors,
        block_thomas_factor,
        block_thomas_solve,
        block_thomas_truncated_fn,
    )

    _SOLVAX_IMPORT_ERROR: BaseException | None = None
except ImportError as _solvax_exc:
    BlockTridiagFactors = None  # type: ignore[assignment, misc]
    block_thomas_factor = None  # type: ignore[assignment]
    block_thomas_solve = None  # type: ignore[assignment]
    block_thomas_truncated_fn = None  # type: ignore[assignment]
    _SOLVAX_IMPORT_ERROR = _solvax_exc


def _require_solvax() -> None:
    """Raise a clear error when the optional ``solvax`` dependency is missing."""
    if _SOLVAX_IMPORT_ERROR is not None:
        raise ImportError(
            "the RHSMode=3 block-Thomas direct path requires the optional "
            "'solvax' package (install with `pip install sfincs_jax[structured]` "
            "or from git: pip install git+https://github.com/uwplasma/SOLVAX)"
        ) from _SOLVAX_IMPORT_ERROR


EmitFn = Callable[[int, str], None]

_ENV_ENABLE = "SFINCS_JAX_RHS3_BLOCK_THOMAS"
_ENV_CHUNK = "SFINCS_JAX_RHS3_BLOCK_THOMAS_CHUNK"
_ENV_VALIDATE = "SFINCS_JAX_RHS3_BLOCK_THOMAS_VALIDATE"

_SOLVE_METHOD_ALIASES = frozenset({"block_tridiagonal", "block_thomas"})


class Rhs3BlockStructureError(RuntimeError):
    """Raised when the RHSMode=3 operator is not block tridiagonal over Legendre modes."""


def rhs3_block_tridiagonal_requested(*, solve_method: str, rhs_mode: int) -> bool:
    """Return True when the RHSMode=3 block-Thomas direct path is opted in.

    Opt-in is either ``solve_method="block_tridiagonal"`` or the environment
    variable ``SFINCS_JAX_RHS3_BLOCK_THOMAS=1``. The path only applies to
    RHSMode=3 (monoenergetic) systems.
    """
    if int(rhs_mode) != 3:
        return False
    if str(solve_method).strip().lower() in _SOLVE_METHOD_ALIASES:
        return True
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in {"1", "true", "yes", "on"}


def rhs3_block_tridiagonal_supported(op: Any) -> tuple[bool, str]:
    """Check whether ``op`` fits the monoenergetic Legendre block-tridiagonal layout."""
    if int(op.rhs_mode) != 3:
        return False, f"rhs_mode={int(op.rhs_mode)} != 3"
    if int(op.n_species) != 1 or int(op.n_x) != 1:
        return False, f"needs single species and Nx=1 (got S={int(op.n_species)}, X={int(op.n_x)})"
    if int(op.phi1_size) != 0:
        return False, "includePhi1 layouts are not supported"
    if int(op.extra_size) != 1:
        return False, f"expected exactly one constraint/source unknown (got {int(op.extra_size)})"
    if int(op.n_xi) < 2:
        return False, f"needs Nxi >= 2 (got {int(op.n_xi)})"
    return True, ""


@dataclass(frozen=True)
class Rhs3LegendreBlocks:
    """Dense Legendre bands plus bordering probed from the RHSMode=3 operator.

    Attributes:
        lower: sub-diagonal blocks ``L_l``, shape ``(Nxi, m, m)``; ``lower[0]`` unused.
        diag: diagonal blocks ``D_l``, shape ``(Nxi, m, m)``.
        upper: super-diagonal blocks ``U_l``, shape ``(Nxi, m, m)``; ``upper[-1]`` unused.
        source_col: source column ``B`` restricted to the ``l=0`` rows, shape ``(m,)``.
        constraint_row: constraint row ``C`` restricted to the ``l=0`` columns, shape ``(m,)``.
        m: spatial block size ``Ntheta * Nzeta``.
        n_xi: number of Legendre blocks.
        f_size: size of the distribution-function part of the state vector.
        total_size: full state-vector size (``f_size + 1``).
    """

    lower: jnp.ndarray
    diag: jnp.ndarray
    upper: jnp.ndarray
    source_col: jnp.ndarray
    constraint_row: jnp.ndarray
    m: int
    n_xi: int
    f_size: int
    total_size: int


def _resolve_column_chunk(m: int, column_chunk: int | None) -> int:
    if column_chunk is None:
        env = os.environ.get(_ENV_CHUNK, "").strip()
        try:
            column_chunk = int(env) if env else 0
        except ValueError:
            column_chunk = 0
    if int(column_chunk) <= 0:
        column_chunk = 128 if int(m) > 128 else int(m)
    return max(1, min(int(column_chunk), int(m)))


def extract_rhs3_legendre_blocks(
    op: Any,
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    column_chunk: int | None = None,
    validate: bool | None = None,
) -> Rhs3LegendreBlocks:
    """Probe the matrix-free RHSMode=3 operator into dense Legendre bands.

    Uses the mod-3 phase trick: applying the operator to unit columns placed in
    every third Legendre block resolves ``L``, ``D`` and ``U`` contributions
    unambiguously (a row block ``l`` only receives from column blocks
    ``l-1, l, l+1``). Three phases of ``m`` columns each therefore recover all
    bands with ``3 m + 1`` operator applications, batched in chunks.

    Args:
        op: a ``V3FullSystemOperator`` with ``rhs_mode == 3``.
        matvec: optional override for the operator application (defaults to the
            cached jitted full-system apply).
        column_chunk: batch width for the probing vmap (env
            ``SFINCS_JAX_RHS3_BLOCK_THOMAS_CHUNK``, default 128).
        validate: verify the block-tridiagonal reconstruction against one random
            matvec (env ``SFINCS_JAX_RHS3_BLOCK_THOMAS_VALIDATE``, default on).

    Raises:
        Rhs3BlockStructureError: if the layout is unsupported or the probed bands
            fail to reproduce the operator action.
    """
    ok, reason = rhs3_block_tridiagonal_supported(op)
    if not ok:
        raise Rhs3BlockStructureError(f"RHSMode=3 block-tridiagonal path unsupported: {reason}")

    if matvec is None:
        from sfincs_jax.operators.profile_system import apply_v3_full_system_operator_cached  # noqa: PLC0415

        def matvec(v: jnp.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator_cached(op, v)

    m = int(op.n_theta) * int(op.n_zeta)
    n_xi = int(op.n_xi)
    f_size = int(op.f_size)
    total_size = int(op.total_size)
    if f_size != n_xi * m or total_size != f_size + 1:
        raise Rhs3BlockStructureError(
            f"unexpected layout: f_size={f_size}, Nxi*m={n_xi * m}, total_size={total_size}"
        )

    chunk = _resolve_column_chunk(m, column_chunk)
    batched_apply = jax.jit(jax.vmap(matvec, in_axes=1, out_axes=1))

    lower = np.zeros((n_xi, m, m), dtype=np.float64)
    diag = np.zeros((n_xi, m, m), dtype=np.float64)
    upper = np.zeros((n_xi, m, m), dtype=np.float64)
    constraint_row = np.zeros((m,), dtype=np.float64)

    for phase in range(3):
        blocks_in_phase = list(range(phase, n_xi, 3))
        if not blocks_in_phase:
            continue
        for j0 in range(0, m, chunk):
            j1 = min(j0 + chunk, m)
            cols = j1 - j0
            basis = np.zeros((total_size, cols), dtype=np.float64)
            col_idx = np.arange(cols)
            for lc in blocks_in_phase:
                basis[lc * m + j0 + col_idx, col_idx] = 1.0
            y = np.asarray(batched_apply(jnp.asarray(basis)))
            for lc in blocks_in_phase:
                diag[lc, :, j0:j1] = y[lc * m : (lc + 1) * m, :]
                if lc >= 1:
                    upper[lc - 1, :, j0:j1] = y[(lc - 1) * m : lc * m, :]
                if lc + 1 < n_xi:
                    lower[lc + 1, :, j0:j1] = y[(lc + 1) * m : (lc + 2) * m, :]
            if phase == 0:
                constraint_row[j0:j1] = y[f_size, :]

    # Source column: the operator applied to the constraint/source unit vector.
    e_src = np.zeros((total_size,), dtype=np.float64)
    e_src[f_size] = 1.0
    b_full = np.asarray(matvec(jnp.asarray(e_src)))
    source_col = b_full[:m].copy()
    tail_norm = float(np.linalg.norm(b_full[m:]))
    src_norm = float(np.linalg.norm(source_col))
    if tail_norm > 1e-12 * max(src_norm, 1.0):
        raise Rhs3BlockStructureError(
            "source column has support outside the l=0 block "
            f"(|tail|={tail_norm:.3e}, |l=0|={src_norm:.3e})"
        )

    blocks = Rhs3LegendreBlocks(
        lower=jnp.asarray(lower),
        diag=jnp.asarray(diag),
        upper=jnp.asarray(upper),
        source_col=jnp.asarray(source_col),
        constraint_row=jnp.asarray(constraint_row),
        m=m,
        n_xi=n_xi,
        f_size=f_size,
        total_size=total_size,
    )

    if validate is None:
        validate = os.environ.get(_ENV_VALIDATE, "").strip().lower() not in {"0", "false", "no", "off"}
    if validate:
        _validate_block_reconstruction(blocks, matvec)
    return blocks


def _apply_blocks(blocks: Rhs3LegendreBlocks, x_full: jnp.ndarray) -> jnp.ndarray:
    """Apply the probed bordered block-tridiagonal operator to a full state vector."""
    x_f = x_full[: blocks.f_size].reshape(blocks.n_xi, blocks.m)
    s = x_full[blocks.f_size]
    y = jnp.einsum("kij,kj->ki", blocks.diag, x_f)
    y = y.at[1:].add(jnp.einsum("kij,kj->ki", blocks.lower[1:], x_f[:-1]))
    y = y.at[:-1].add(jnp.einsum("kij,kj->ki", blocks.upper[:-1], x_f[1:]))
    y = y.at[0].add(blocks.source_col * s)
    y_extra = jnp.dot(blocks.constraint_row, x_f[0])
    return jnp.concatenate([y.reshape(-1), y_extra[None]])


def _validate_block_reconstruction(
    blocks: Rhs3LegendreBlocks,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    rtol: float = 1e-10,
) -> None:
    """Verify the probed bands reproduce one random operator application.

    This catches any Legendre coupling beyond ``l +- 1`` (which would silently
    corrupt the mod-3 probing) at the cost of a single extra matvec.
    """
    rng = np.random.default_rng(20260708)
    x = jnp.asarray(rng.standard_normal(blocks.total_size))
    y_ref = matvec(x)
    y_blocks = _apply_blocks(blocks, x)
    err = float(jnp.linalg.norm(y_blocks - y_ref))
    scale = float(jnp.linalg.norm(y_ref))
    if not np.isfinite(err) or err > rtol * max(scale, 1.0):
        raise Rhs3BlockStructureError(
            "probed Legendre bands do not reproduce the operator action "
            f"(|err|={err:.3e}, |A x|={scale:.3e}); the operator is not "
            "block tridiagonal over l"
        )


@dataclass(frozen=True)
class Rhs3BlockThomasSolver:
    """Reusable factored solver for the RHSMode=3 bordered block-tridiagonal system."""

    blocks: Rhs3LegendreBlocks
    factors: BlockTridiagFactors
    z_col: jnp.ndarray  # A~^{-1} B, shape (Nxi, m)
    gamma: jnp.ndarray  # regularization scale (scalar)

    def solve(self, rhs_full: jnp.ndarray) -> jnp.ndarray:
        """Solve the bordered system for stacked full RHS vectors.

        Args:
            rhs_full: ``(total_size,)`` or ``(total_size, n_rhs)`` right-hand sides
                in the existing full state-vector layout (f part then the
                constraint-row entry).

        Returns:
            Solution(s) with the same shape, including the source unknown.
        """
        blocks = self.blocks
        squeeze = rhs_full.ndim == 1
        rhs2d = rhs_full[:, None] if squeeze else rhs_full
        n_rhs = rhs2d.shape[1]
        b_f = rhs2d[: blocks.f_size].reshape(blocks.n_xi, blocks.m, n_rhs)
        r_c = rhs2d[blocks.f_size]  # (n_rhs,)

        y = block_thomas_solve(self.factors, b_f)  # (Nxi, m, n_rhs)
        c_y = jnp.einsum("j,jr->r", blocks.constraint_row, y[0])
        c_z = jnp.dot(blocks.constraint_row, self.z_col[0])
        s = self.gamma * r_c + (c_y - r_c) / c_z
        f = y - (s - self.gamma * r_c)[None, None, :] * self.z_col[:, :, None]
        x = jnp.concatenate([f.reshape(blocks.f_size, n_rhs), s[None, :]], axis=0)
        return x[:, 0] if squeeze else x


def build_rhs3_block_thomas_solver(
    blocks: Rhs3LegendreBlocks,
    *,
    gamma: float | None = None,
) -> Rhs3BlockThomasSolver:
    """Factor the rank-one-regularized Legendre bands once for reuse across RHS.

    Args:
        blocks: probed bands and bordering from :func:`extract_rhs3_legendre_blocks`.
        gamma: regularization scale for ``A~ = A + gamma B C``. Any nonzero value
            is exact; the default scales the rank-one update to the mean magnitude
            of the diagonal blocks for good conditioning.
    """
    _require_solvax()
    if gamma is None:
        scale = float(jnp.mean(jnp.abs(jnp.diagonal(blocks.diag, axis1=1, axis2=2))))
        if scale == 0.0:
            scale = float(jnp.mean(jnp.abs(blocks.diag)))
        outer_max = float(jnp.max(jnp.abs(blocks.source_col))) * float(
            jnp.max(jnp.abs(blocks.constraint_row))
        )
        gamma = scale / outer_max if (outer_max > 0.0 and scale > 0.0) else 1.0
    gamma_arr = jnp.asarray(float(gamma), dtype=jnp.float64)

    diag_reg = blocks.diag.at[0].add(
        gamma_arr * jnp.outer(blocks.source_col, blocks.constraint_row)
    )
    factors = block_thomas_factor(blocks.lower, diag_reg, blocks.upper)

    b_rhs = jnp.zeros((blocks.n_xi, blocks.m), dtype=jnp.float64).at[0].set(blocks.source_col)
    z_col = block_thomas_solve(factors, b_rhs)
    return Rhs3BlockThomasSolver(blocks=blocks, factors=factors, z_col=z_col, gamma=gamma_arr)


def solve_rhs3_block_tridiagonal(
    *,
    op: Any,
    rhs_columns: jnp.ndarray,
    matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    column_chunk: int | None = None,
    validate: bool | None = None,
    gamma: float | None = None,
) -> jnp.ndarray:
    """One-shot direct solve of the RHSMode=3 system for stacked RHS columns.

    Convenience wrapper: probe blocks, factor once, solve all columns.

    Args:
        op: ``V3FullSystemOperator`` with ``rhs_mode == 3``.
        rhs_columns: ``(total_size,)`` or ``(total_size, n_rhs)``.

    Returns:
        Solution(s) with the same shape as ``rhs_columns``.
    """
    blocks = extract_rhs3_legendre_blocks(
        op, matvec=matvec, column_chunk=column_chunk, validate=validate
    )
    solver = build_rhs3_block_thomas_solver(blocks, gamma=gamma)
    return solver.solve(jnp.asarray(rhs_columns, dtype=jnp.float64))


def _transposed_batched_apply(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], total_size: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a column-batched application of the transposed operator ``A^T``."""
    primal = jax.ShapeDtypeStruct((int(total_size),), jnp.float64)

    def at_single(w: jnp.ndarray) -> jnp.ndarray:
        (out,) = jax.linear_transpose(matvec, primal)(w)
        return out

    return jax.vmap(at_single, in_axes=1, out_axes=1)


def solve_rhs3_block_tridiagonal_truncated(
    *,
    op: Any,
    rhs_columns: jnp.ndarray,
    keep_lowest: int = 3,
    matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    column_chunk: int | None = None,
    gamma: float | None = None,
) -> jnp.ndarray:
    """Memory-lean RHSMode=3 direct solve returning only the lowest Legendre blocks.

    For RHSMode=3 both transport drives have Legendre support ``l <= 2`` (density
    gradient: ``l = 0, 2``; E_parallel: ``l = 1``) and every transport-matrix
    moment (particle flux, heat flux, FSAB flow, sources) contracts the solution
    against weights supported on ``l <= 2`` only. This wraps
    :func:`solvax.direct.block_thomas_truncated_fn`: the Legendre blocks are
    probed on the fly inside the Schur sweeps (one row block at a time, via the
    transposed matvec), so the ``(Nxi, m, m)`` bands and their factors are never
    materialized -- peak memory is ``O(keep_lowest * m^2)`` plus one probing
    batch, independent of ``Nxi``.

    Trade-off: on-the-fly probing applies the (transposed) full operator to
    ``m`` columns per block, i.e. ``Nxi * m`` matvecs total versus ``3 m`` for
    the phase-trick band assembly used by :func:`extract_rhs3_legendre_blocks`,
    so this variant trades time for memory. Prefer it when the bands do not fit
    comfortably in memory (large ``Nxi * m^2``).

    Args:
        op: ``V3FullSystemOperator`` with ``rhs_mode == 3``.
        rhs_columns: ``(total_size,)`` or ``(total_size, n_rhs)`` right-hand
            sides whose f part vanishes for ``l >= keep_lowest``.
        keep_lowest: number of Legendre blocks of the solution to compute
            (default 3, sufficient for the transport matrix).
        matvec: optional override of the operator application.
        column_chunk: probing batch width (defaults as in
            :func:`extract_rhs3_legendre_blocks`).
        gamma: null-space regularization scale (see module docstring).

    Returns:
        Full-size solution columns with the same shape as ``rhs_columns``, where
        Legendre blocks ``l >= keep_lowest`` are ZERO-PADDED (not solved). The
        source unknown is exact. Valid only for contracting against moments
        supported on ``l < keep_lowest`` (e.g. the RHSMode=3 transport matrix).

    Raises:
        Rhs3BlockStructureError: if the layout is unsupported or the RHS has
            support at ``l >= keep_lowest``.
    """
    ok, reason = rhs3_block_tridiagonal_supported(op)
    if not ok:
        raise Rhs3BlockStructureError(f"RHSMode=3 block-tridiagonal path unsupported: {reason}")

    if matvec is None:
        from sfincs_jax.operators.profile_system import apply_v3_full_system_operator_cached  # noqa: PLC0415

        def matvec(v: jnp.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator_cached(op, v)

    m = int(op.n_theta) * int(op.n_zeta)
    n_xi = int(op.n_xi)
    f_size = int(op.f_size)
    total_size = int(op.total_size)
    keep = int(keep_lowest)
    if not 1 <= keep <= n_xi:
        raise Rhs3BlockStructureError(f"need 1 <= keep_lowest <= Nxi (got {keep}, Nxi={n_xi})")

    rhs_columns = jnp.asarray(rhs_columns, dtype=jnp.float64)
    squeeze = rhs_columns.ndim == 1
    rhs2d = rhs_columns[:, None] if squeeze else rhs_columns
    n_rhs = rhs2d.shape[1]
    rhs_f = rhs2d[:f_size].reshape(n_xi, m, n_rhs)
    r_c = rhs2d[f_size]  # (n_rhs,)
    tail_max = float(jnp.max(jnp.abs(rhs_f[keep:]))) if keep < n_xi else 0.0
    if tail_max != 0.0:
        raise Rhs3BlockStructureError(
            f"RHS has Legendre support at l >= {keep} (max |rhs| = {tail_max:.3e}); "
            "the truncated solve requires the RHS to vanish there"
        )

    chunk = _resolve_column_chunk(m, column_chunk)
    starts = list(range(0, m, chunk))
    at_batched = _transposed_batched_apply(matvec, total_size)
    batched_apply = jax.vmap(matvec, in_axes=1, out_axes=1)

    # Bordering pieces: B (source column, l=0 rows) and C (constraint row, l=0 cols).
    e_src = jnp.zeros((total_size,), dtype=jnp.float64).at[f_size].set(1.0)
    b0 = matvec(e_src)[:m]
    c0 = at_batched(e_src[:, None])[:m, 0]

    if gamma is None:
        # Probe the l=0 diagonal block once for a conditioning-friendly scale.
        # Note: diag(D_0) can be exactly zero (pure PAS has no l=0 diagonal
        # term), so fall back to the mean |entry| of the whole block.
        basis0 = jnp.zeros((total_size, m), dtype=jnp.float64).at[jnp.arange(m), jnp.arange(m)].set(1.0)
        d0_cols = batched_apply(basis0)[:m]
        scale = float(jnp.mean(jnp.abs(jnp.diagonal(d0_cols))))
        if scale == 0.0:
            scale = float(jnp.mean(jnp.abs(d0_cols)))
        outer_max = float(jnp.max(jnp.abs(b0))) * float(jnp.max(jnp.abs(c0)))
        gamma = scale / outer_max if (outer_max > 0.0 and scale > 0.0) else 1.0
    gamma_arr = jnp.asarray(float(gamma), dtype=jnp.float64)
    g_outer = gamma_arr * jnp.outer(b0, c0)

    def block_fn(k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Probe row block ``k``: columns of ``A^T`` give rows of ``A``."""
        start_l = jnp.clip((k - 1) * m, 0, f_size - m)
        start_d = k * m
        start_u = jnp.clip((k + 1) * m, 0, f_size - m)
        zero = jnp.zeros((), dtype=start_d.dtype)
        l_parts, d_parts, u_parts = [], [], []
        for j0 in starts:
            j1 = min(j0 + chunk, m)
            cols = j1 - j0
            rows = start_d + jnp.arange(j0, j1)
            basis = jnp.zeros((total_size, cols), dtype=jnp.float64)
            basis = basis.at[rows, jnp.arange(cols)].set(1.0)
            y = at_batched(basis)  # (total_size, cols) = A^T columns of row block k
            l_parts.append(jax.lax.dynamic_slice(y, (start_l.astype(zero.dtype), zero), (m, cols)))
            d_parts.append(jax.lax.dynamic_slice(y, (start_d, zero), (m, cols)))
            u_parts.append(jax.lax.dynamic_slice(y, (start_u.astype(zero.dtype), zero), (m, cols)))
        l_k = jnp.concatenate(l_parts, axis=1).T
        d_k = jnp.concatenate(d_parts, axis=1).T
        u_k = jnp.concatenate(u_parts, axis=1).T
        d_k = jnp.where(k == 0, d_k + g_outer, d_k)
        return l_k, d_k, u_k

    # Append the source column as one extra RHS; all columns vanish for l >= keep.
    rhs_low = jnp.concatenate(
        [
            rhs_f[:keep],
            jnp.zeros((keep, m, 1), dtype=jnp.float64).at[0, :, 0].set(b0),
        ],
        axis=2,
    )
    x_low = block_thomas_truncated_fn(block_fn, n_xi, rhs_low, keep)
    y_low, z_low = x_low[:, :, :n_rhs], x_low[:, :, n_rhs]

    c_y = jnp.einsum("j,jr->r", c0, y_low[0])
    c_z = jnp.dot(c0, z_low[0])
    s_shift = (c_y - r_c) / c_z
    s = gamma_arr * r_c + s_shift
    f_low = y_low - z_low[:, :, None] * s_shift[None, None, :]

    x = jnp.zeros((total_size, n_rhs), dtype=jnp.float64)
    x = x.at[: keep * m].set(f_low.reshape(keep * m, n_rhs))
    x = x.at[f_size].set(s)
    return x[:, 0] if squeeze else x


def solve_transport_block_tridiagonal_batch(
    *,
    context: Any,
    op_probe_ref: Any,
    tol: float,
    atol: float,
) -> bool:
    """Solve all RHSMode=3 transport drives with the solvax block-Thomas direct path.

    Mirrors ``solve_transport_dense_batch``: on success, stores state vectors,
    residual norms and solver bookkeeping into the mutable ``context``
    (a ``TransportDenseBatchContext``) and returns True. Returns False (leaving
    the context untouched) when the operator does not fit the monoenergetic
    block-tridiagonal layout, so the caller falls back to the standard path.
    """
    _require_solvax()
    from sfincs_jax.operators.profile_system import (  # noqa: PLC0415
        _operator_signature_cached,
        apply_v3_full_system_operator_cached,
    )
    from sfincs_jax.profiling import Timer  # noqa: PLC0415

    emit = context.emit
    ok, reason = rhs3_block_tridiagonal_supported(op_probe_ref)
    if not ok:
        if emit is not None:
            emit(1, f"solve_v3_transport_matrix_linear_gmres: block-Thomas path skipped ({reason})")
        return False
    sig_ref = _operator_signature_cached(op_probe_ref)
    for op_probe in context.op_matvec_by_index[1:]:
        if _operator_signature_cached(op_probe) != sig_ref:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: block-Thomas path skipped "
                    "(matvec operator varies across whichRHS)",
                )
            return False

    timer = Timer()
    if emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: solvax block-Thomas direct solve "
            f"across all whichRHS (Nxi={int(op_probe_ref.n_xi)} blocks of size "
            f"{int(op_probe_ref.n_theta) * int(op_probe_ref.n_zeta)})",
        )
    try:
        rhs_mat = jnp.stack([jnp.asarray(r, dtype=jnp.float64) for r in context.rhs_by_index], axis=1)
        x_mat = solve_rhs3_block_tridiagonal(op=op_probe_ref, rhs_columns=rhs_mat)
    except Rhs3BlockStructureError as exc:
        if emit is not None:
            emit(1, f"solve_v3_transport_matrix_linear_gmres: block-Thomas path skipped ({exc})")
        return False

    # True residuals through the original matrix-free operator.
    res_norms: list[float] = []
    for idx in range(x_mat.shape[1]):
        ax = apply_v3_full_system_operator_cached(op_probe_ref, x_mat[:, idx])
        res_norms.append(float(jnp.linalg.norm(ax - rhs_mat[:, idx])))
    for idx, which_rhs in enumerate(context.which_rhs_values):
        rhs_norm = float(context.rhs_norms[int(which_rhs)])
        target = max(float(atol), float(tol) * rhs_norm)
        if not np.isfinite(res_norms[idx]) or res_norms[idx] > max(target, 1e-8 * max(rhs_norm, 1.0)):
            if emit is not None:
                emit(
                    0,
                    "solve_v3_transport_matrix_linear_gmres: block-Thomas residual too large "
                    f"(whichRHS={int(which_rhs)} residual={res_norms[idx]:.3e} target={target:.3e}); "
                    "falling back to the standard path",
                )
            return False

    elapsed_each = float(timer.elapsed_s() / float(len(context.which_rhs_values)))
    for idx, which_rhs in enumerate(context.which_rhs_values):
        which = int(which_rhs)
        x_col = x_mat[:, idx]
        if context.store_state_vectors:
            context.state_vectors[which] = x_col
        if context.stream_diagnostics:
            if context.collect_transport_outputs is None:
                raise RuntimeError(
                    "block-Thomas streaming diagnostics requested without an output collector"
                )
            context.collect_transport_outputs(which, x_col)
        context.residual_norms[which] = jnp.asarray(res_norms[idx], dtype=jnp.float64)
        context.solver_kinds_by_rhs[which] = "block_tridiagonal"
        context.solve_methods_by_rhs[which] = "block_tridiagonal"
        context.elapsed_s[which - 1] = elapsed_each
        if emit is not None:
            rhs_norm = float(context.rhs_norms[which])
            rel = res_norms[idx] / rhs_norm if rhs_norm > 0.0 else float("nan")
            emit(
                0,
                f"whichRHS={which}: residual_norm={res_norms[idx]:.6e} "
                f"rhs_norm={rhs_norm:.6e} relative_residual={rel:.6e} "
                f"elapsed_s={elapsed_each:.3f}",
            )
    return True
