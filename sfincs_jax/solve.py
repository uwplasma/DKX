"""The plan-§2.3 three-tier auto-policy linear solver over a :class:`KineticOperator`.

This module is the Phase-3.3 solve track: given the consolidated v3
drift-kinetic operator (:mod:`sfincs_jax.drift_kinetic`) and one or more right-hand
sides, pick and run the cheapest adequate linear solver:

Tier 1 — structured direct (``solvax.direct`` block Thomas over Legendre modes)
    Available when :meth:`KineticOperator.to_block_tridiagonal` succeeds (the
    DKES-trajectory / pitch-angle-scattering family: streaming+mirror couple
    L±1, ExB and PAS are diagonal in L, no Er xDot/xiDot L±2 terms, no
    Fokker-Planck (species,x) coupling).  For that family the (species, x)
    axes are mutually uncoupled in the f-block and — for ``constraintScheme=2``
    — the bordered source/constraint machinery is diagonal over (species, x)
    too, so the full system splits into ``n_species * n_x`` independent
    block-tridiagonal systems of ``n_xi`` dense (Ntheta*Nzeta) blocks with a
    rank-one border each.  The border is absorbed exactly with the rank-one
    trick of ``sfincs_jax/solvers/block_tridiagonal_transport.py``
    (``A~ = A + gamma B C``; algebraically exact for any ``gamma != 0``) and
    the batch is solved by ``vmap``-ed ``solvax.block_thomas_factor`` /
    ``block_thomas_solve``.  Multi-RHS shares one elimination.

Tier 2 — preconditioned, recycled Krylov (``solvax.krylov.gcrot``)
    Matrix-free FGMRES+recycling on :meth:`KineticOperator.apply`,
    right-preconditioned by an exact tier-1 solve of the SFINCS-simplified
    coarse operator (the Fortran ``preconditionerOptions`` idiom):
    ``preconditioner_species=1`` (self-collisions only) and
    ``preconditioner_x=1`` (x-diagonal collisions) reduce Fokker-Planck to a
    PAS-like L-diagonal coefficient; the Er L±2 terms are dropped; optionally
    ``preconditioner_xi=1`` drops the L±1 streaming coupling.  The bordered
    constraint rows are eliminated exactly through
    ``solvax.operators.schur_projected_precond``.  The recycle pair (C, U) is
    returned for warm-starting continuation (Er scans, Newton steps).

Tier 3 — host sparse-direct fallback (``solvax.native.splu_solve``)
    Materializes the operator (vmapped unit vectors; guarded by
    ``max_dense_size``) into CSR and hands it to SuperLU on the host.
    Non-differentiable, non-jittable; prints a loud one-line notice.  Used on
    explicit request (``method="direct"``) or when tier 2 breaches its
    iteration cap under ``method="auto"``.

Differentiability: tiers 1 and 2 are wrapped with
``solvax.implicit.linear_solve`` (implicit function theorem via
``jax.lax.custom_linear_solve``) when ``differentiable=True``; the adjoint
costs one transposed solve which reuses the same tier-1 factors
(``block_thomas_solve(transpose=True)``) or a transposed-preconditioner
GCROT solve.  Tier 3 is a loud, non-differentiable escape hatch.

Fortran correspondence: ``solver.F90`` (KSP setup / preconditioner matrix
``whichMatrix=0``), ``preconditioner.F90`` (the ``preconditioner_*`` knobs),
and the PETSc ``Pmat`` idiom of production SFINCS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# solvax is optional until its PyPI release (CI installs it from git): keep
# this module importable without it and raise a clear error on first use.
try:  # noqa: E402
    from solvax.direct import (
        BlockTridiagFactors,
        block_thomas_factor,
        block_thomas_solve,
    )
    from solvax.implicit import linear_solve as solvax_linear_solve
    from solvax.krylov import gcrot
    from solvax.native import SpluFactorization
    from solvax.operators import schur_projected_precond

    _SOLVAX_IMPORT_ERROR: BaseException | None = None
except ImportError as _solvax_exc:
    BlockTridiagFactors = None  # type: ignore[assignment, misc]
    block_thomas_factor = None  # type: ignore[assignment]
    block_thomas_solve = None  # type: ignore[assignment]
    solvax_linear_solve = None  # type: ignore[assignment]
    gcrot = None  # type: ignore[assignment]
    SpluFactorization = None  # type: ignore[assignment, misc]
    schur_projected_precond = None  # type: ignore[assignment]
    _SOLVAX_IMPORT_ERROR = _solvax_exc

from sfincs_jax.drift_kinetic import KineticOperator  # noqa: E402


def _require_solvax() -> None:
    """Raise a clear error when the optional ``solvax`` dependency is missing."""
    if _SOLVAX_IMPORT_ERROR is not None:
        raise ImportError(
            "sfincs_jax.solve requires the optional 'solvax' package for its "
            "solver tiers (install with `pip install sfincs_jax[structured]` "
            "or from git: pip install git+https://github.com/uwplasma/SOLVAX)"
        ) from _SOLVAX_IMPORT_ERROR

__all__ = [
    "SolveResult",
    "Tier1Solver",
    "build_coarse_preconditioner",
    "build_tier1_solver",
    "materialize_dense",
    "solve",
    "tier1_available",
]


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class SolveResult:
    """Outcome of :func:`solve`.

    Attributes:
        x: solution state vector(s), same shape as the ``rhs`` passed in
            (``(n,)`` or ``(n, n_rhs)``).
        method: solver actually used: ``"block_tridiagonal"`` (tier 1),
            ``"gcrot"`` (tier 2), or ``"direct"`` (tier 3).
        iterations: total Krylov inner iterations across all right-hand sides
            (tier 2), else ``None``.
        residual_norms: true residual norms ``||b - A x||`` per right-hand
            side, shape ``(n_rhs,)`` (jnp array; traced under ``jax.grad``).
        converged: every residual below ``max(atol, tol * ||b||)``.  ``True``
            by construction for the direct tiers when residuals are finite.
        recycle: GCROT recycle pair ``(C, U)`` from the last right-hand side
            (tier 2), for warm-starting the next solve of a continuation.
        timings: wall-clock seconds per phase (``build``, ``solve``).
    """

    x: jnp.ndarray
    method: str
    iterations: int | None
    residual_norms: jnp.ndarray
    converged: bool
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None
    timings: dict[str, float]


def _as_columns(rhs: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs.ndim == 1:
        return rhs[:, None], True
    if rhs.ndim == 2:
        return rhs, False
    raise ValueError(f"rhs must be (n,) or (n, n_rhs); got shape {rhs.shape}")


def _is_traced(*arrays: Any) -> bool:
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


def _residual_norms(op: KineticOperator, x2d: jnp.ndarray, rhs2d: jnp.ndarray) -> jnp.ndarray:
    res = jax.vmap(op.apply, in_axes=1, out_axes=1)(x2d) - rhs2d
    return jnp.linalg.norm(res, axis=0)


def _converged_flag(
    res_norms: jnp.ndarray, rhs2d: jnp.ndarray, tol: float, atol: float
) -> bool:
    if _is_traced(res_norms):
        return True  # direct tiers under trace: exact up to factor accuracy
    rhs_norms = np.linalg.norm(np.asarray(rhs2d), axis=0)
    targets = np.maximum(atol, tol * rhs_norms)
    res = np.asarray(res_norms)
    return bool(np.all(np.isfinite(res)) and np.all(res <= np.maximum(targets, 1e-30)))


def _transposed_apply(op: KineticOperator) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """The transposed matvec ``w -> A^T w`` via ``jax.linear_transpose``."""
    primal = jax.ShapeDtypeStruct((op.total_size,), jnp.float64)

    def apply_t(w: jnp.ndarray) -> jnp.ndarray:
        (out,) = jax.linear_transpose(op.apply, primal)(w)
        return out

    return apply_t


# =============================================================================
# Tier 1 — structured direct (block Thomas over Legendre modes)
# =============================================================================


def tier1_available(op: KineticOperator) -> tuple[bool, str]:
    """Check whether the tier-1 structured direct path applies to ``op``.

    The decision is driven by the operator's own block extraction: if
    :meth:`KineticOperator.legendre_blocks` refuses (Er L±2 terms,
    Fokker-Planck collisions), tier 1 is off.  On top of that the bordered
    constraint machinery must be diagonal over (species, x)
    (``constraintScheme`` 0 or 2 without ``point_at_x0``) and every speed node
    must retain the full Legendre resolution (uniform ``Nxi_for_x``), so the
    per-(species, x) blocks are nonsingular after the rank-one border
    absorption.
    """
    try:
        op._check_block_extraction_supported()
    except NotImplementedError as exc:
        return False, str(exc)
    if op.constraint_scheme not in (0, 2):
        return False, (
            f"constraintScheme={op.constraint_scheme} borders couple speed nodes; "
            "only 0 and 2 keep the (species, x) block split exact"
        )
    if op.constraint_scheme == 2 and op.point_at_x0:
        return False, "point_at_x0 x-grids give the x=0 constraint row a different form"
    if int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi:
        return False, "non-uniform Nxi_for_x leaves zero rows in the truncated-L blocks"
    return True, ""


@dataclass(frozen=True)
class Tier1Solver:
    """Factored per-(species, x) bordered block-tridiagonal solver.

    Holds the batched block-Thomas factors of the rank-one-regularized
    Legendre bands ``A~ = A + gamma B C`` for every (species, x) subsystem,
    plus the presolved border columns ``z = A~^{-1} B`` (forward) and
    ``z_t = A~^{-T} C^T`` (transpose), so both the forward and the adjoint
    bordered solve reuse the same elimination.
    """

    op: KineticOperator
    factors: BlockTridiagFactors  # leading batch axis B = S*X
    z_fwd: jnp.ndarray  # (B, L, TZ)
    z_t: jnp.ndarray  # (B, L, TZ)
    gamma: jnp.ndarray  # (B,)
    b0: jnp.ndarray  # (TZ,) source column shape on the l=0 rows
    c0: jnp.ndarray  # (TZ,) constraint row (flux-surface-average weights)

    def solve(self, rhs: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
        """Solve ``K x = rhs`` (or ``K^T x = rhs``) for flat state vector(s).

        Args:
            rhs: ``(total_size,)`` or ``(total_size, n_rhs)``.
            transpose: solve the transposed bordered system, reusing the same
                factors via ``block_thomas_solve(transpose=True)``.

        Returns:
            Solution(s) with the same shape as ``rhs``.
        """
        op = self.op
        rhs2d, squeeze = _as_columns(rhs)
        n_rhs = rhs2d.shape[1]
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        batch = n_s * n_x
        n_tz = n_t * n_z

        # f part -> (B, L, TZ, n_rhs)
        b_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
        b_f = b_f.reshape(batch, n_xi, n_tz, n_rhs)

        solve_batched = jax.vmap(lambda f, r: block_thomas_solve(f, r, transpose=transpose))
        y = solve_batched(self.factors, b_f)  # (B, L, TZ, n_rhs)

        if op.constraint_scheme == 0:
            x = y.reshape(op.f_size, n_rhs)
            return x[:, 0] if squeeze else x

        # constraintScheme=2: one bordered unknown per (species, x).
        # Forward:  [[A, b0 e0], [c0^T e0^T, 0]];  transpose swaps b0 <-> c0.
        r_c = rhs2d[op.f_size :].reshape(batch, n_rhs)
        z = self.z_t if transpose else self.z_fwd
        w_row = self.b0 if transpose else self.c0  # constraint row of the (transposed) system
        c_y = jnp.einsum("j,bjr->br", w_row, y[:, 0])  # w·y[l=0], (B, n_rhs)
        c_z = jnp.einsum("j,bj->b", w_row, z[:, 0])  # (B,)
        s = self.gamma[:, None] * r_c + (c_y - r_c) / c_z[:, None]
        shift = s - self.gamma[:, None] * r_c  # (B, n_rhs)
        f = y - shift[:, None, None, :] * z[:, :, :, None]

        x = jnp.concatenate(
            [f.reshape(op.f_size, n_rhs), s.reshape(op.extra_size, n_rhs)], axis=0
        )
        return x[:, 0] if squeeze else x


def build_tier1_solver(op: KineticOperator) -> Tier1Solver:
    """Assemble and factor the tier-1 batched bordered block-tridiagonal solver.

    Uses the analytic (probing-free) :meth:`KineticOperator.to_block_tridiagonal`
    blocks — the replacement for the probing-based extraction in
    ``sfincs_jax/solvers/block_tridiagonal_transport.py`` — and absorbs the
    ``constraintScheme=2`` border with the exact rank-one trick
    ``A~ = A + gamma B C`` documented there.

    Raises:
        NotImplementedError: when :func:`tier1_available` says no.
    """
    _require_solvax()
    ok, reason = tier1_available(op)
    if not ok:
        raise NotImplementedError(f"tier-1 structured direct path unavailable: {reason}")

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    blocks = op.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower, diag, upper = (
        jnp.transpose(a, (1, 2, 0, 3, 4)).reshape(batch, n_xi, n_tz, n_tz) for a in blocks
    )

    b0 = jnp.ones((n_tz,), dtype=jnp.float64)  # source shape on the l=0 rows
    c0 = op._fs_average_factor().reshape(-1)  # flux-surface-average constraint row

    if op.constraint_scheme == 2:
        # Conditioning-friendly rank-one scale per (species, x): mean |diag entry|
        # of the bands over the max magnitude of the rank-one update.
        scale = jnp.mean(jnp.abs(jnp.diagonal(diag, axis1=2, axis2=3)), axis=(1, 2))
        scale = jnp.where(scale > 0.0, scale, jnp.mean(jnp.abs(diag), axis=(1, 2, 3)))
        outer_max = jnp.max(jnp.abs(b0)) * jnp.max(jnp.abs(c0))
        gamma = scale / outer_max
        diag = diag.at[:, 0].add(gamma[:, None, None] * jnp.outer(b0, c0)[None, :, :])
    else:
        gamma = jnp.ones((batch,), dtype=jnp.float64)

    factors = jax.vmap(block_thomas_factor)(lower, diag, upper)

    e0 = jnp.zeros((batch, n_xi, n_tz), dtype=jnp.float64)
    z_fwd = jax.vmap(block_thomas_solve)(factors, e0.at[:, 0, :].set(b0[None, :]))
    z_t = jax.vmap(lambda f, r: block_thomas_solve(f, r, transpose=True))(
        factors, e0.at[:, 0, :].set(c0[None, :])
    )
    return Tier1Solver(op=op, factors=factors, z_fwd=z_fwd, z_t=z_t, gamma=gamma, b0=b0, c0=c0)


# =============================================================================
# Tier 2 — coarse-operator preconditioner (Fortran preconditioner_* knobs)
# =============================================================================


def _fp_diagonal_coefficients(op: KineticOperator) -> jnp.ndarray:
    """(S, X, L) self-species, x-diagonal Fokker-Planck coefficients.

    This is the ``preconditioner_species=1`` + ``preconditioner_x=1``
    simplification: keep only ``mat[s, s, l, x, x]`` of the dense
    (species, x)-coupled collision blocks, which is PAS-like (diagonal in
    everything but L).
    """
    mat = op.fp.mat  # (S, S, L, X, X)
    coef = jnp.diagonal(mat, axis1=0, axis2=1)  # (L, X, X, S)
    coef = jnp.diagonal(coef, axis1=1, axis2=2)  # (L, S, X)
    return jnp.transpose(coef, (1, 2, 0))  # (S, X, L)


def _materialize_borders(op: KineticOperator) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact border columns ``B`` (f_size, extra) and rows ``C`` (extra, f_size).

    Probed from the operator itself (``extra_size`` matvecs + ``extra_size``
    transposed matvecs — cheap: the border is tiny).
    """
    n, fs, ex = op.total_size, op.f_size, op.extra_size
    basis = jnp.zeros((n, ex), dtype=jnp.float64)
    basis = basis.at[fs + jnp.arange(ex), jnp.arange(ex)].set(1.0)
    b_cols = jax.vmap(op.apply, in_axes=1, out_axes=1)(basis)[:fs]
    apply_t = _transposed_apply(op)
    c_rows = jax.vmap(apply_t, in_axes=1, out_axes=1)(basis)[:fs].T
    return b_cols, c_rows


def build_coarse_preconditioner(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Tier-1 exact solve of the SFINCS-simplified coarse operator, as a preconditioner.

    Mirrors the Fortran ``preconditionerOptions`` defaults: collisions become
    self-species and x-diagonal (Fokker-Planck reduces to its PAS-like
    diagonal), the Er L±2 xDot/xiDot terms are dropped, and (optionally, the
    ``preconditioner_xi=1`` knob) the L±1 streaming coupling is dropped too.
    The result is block-tridiagonal over L and uncoupled over (species, x), so
    one batched block-Thomas factorization inverts it exactly; the bordered
    constraint rows of the *full* operator are then eliminated exactly with
    ``solvax.operators.schur_projected_precond``.

    Returns:
        ``(precond, precond_t)`` — approximate inverses of ``K`` and ``K^T``
        on flat ``(total_size,)`` vectors, sharing one factorization.
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    stripped = replace(op, fp=None, with_er_xidot=False, with_er_xdot=False)
    blocks = stripped.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower, diag, upper = (jnp.transpose(a, (1, 2, 0, 3, 4)) for a in blocks)  # (S,X,L,TZ,TZ)

    eye = jnp.eye(n_tz, dtype=jnp.float64)
    mask = np.asarray(op._mask())  # (X, L)
    if op.fp is not None:
        coef = _fp_diagonal_coefficients(op) * jnp.asarray(mask)[None, :, :]  # (S, X, L)
        diag = diag + coef[:, :, :, None, None] * eye[None, None, None, :, :]
    if drop_l_coupling:
        lower = jnp.zeros_like(lower)
        upper = jnp.zeros_like(upper)

    # Masked (x, l) rows are identically zero in the operator: pin them with
    # the identity so the coarse factorization stays nonsingular.
    pin = jnp.asarray(1.0 - mask)  # (X, L)
    diag = diag + pin[None, :, :, None, None] * eye[None, None, None, :, :]

    # Pin the constant-on-surface null space of the l=0 block per (species, x)
    # (PAS has no l=0 collision diagonal; harmless when the block is regular).
    c0 = op._fs_average_factor().reshape(-1)
    ones = jnp.ones((n_tz,), dtype=jnp.float64)
    d4 = diag.reshape(batch, n_xi, n_tz, n_tz)
    scale = jnp.mean(jnp.abs(jnp.diagonal(d4, axis1=2, axis2=3)), axis=(1, 2))
    scale = jnp.where(scale > 0.0, scale, 1.0)
    gamma = scale / jnp.max(jnp.abs(c0))
    d4 = d4.at[:, 0].add(gamma[:, None, None] * jnp.outer(ones, c0)[None, :, :])

    factors = jax.vmap(block_thomas_factor)(
        lower.reshape(batch, n_xi, n_tz, n_tz), d4, upper.reshape(batch, n_xi, n_tz, n_tz)
    )

    def _a_inv(transpose: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def apply(v: jnp.ndarray) -> jnp.ndarray:
            g = v.reshape(batch, n_xi, n_tz)
            sol = jax.vmap(lambda f, r: block_thomas_solve(f, r, transpose=transpose))(
                factors, g
            )
            return sol.reshape(v.shape)

        return apply

    a_inv, a_inv_t = _a_inv(False), _a_inv(True)
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = _materialize_borders(op)
    precond = schur_projected_precond(a_inv, b_cols, c_rows)
    precond_t = schur_projected_precond(a_inv_t, c_rows.T, b_cols.T)
    return precond, precond_t


# =============================================================================
# Tier 3 — host sparse-direct fallback
# =============================================================================


def materialize_dense(op: KineticOperator, *, column_chunk: int = 1024) -> np.ndarray:
    """Materialize the full bordered operator as a dense numpy matrix.

    Applies the matrix-free operator to identity columns in vmapped chunks.
    Meant for tiny systems (tier-3 fallback and referee tests) — memory is
    ``O(total_size**2)``.
    """
    n = op.total_size
    batched = jax.jit(jax.vmap(op.apply, in_axes=1, out_axes=1))
    cols: list[np.ndarray] = []
    for j0 in range(0, n, column_chunk):
        j1 = min(j0 + column_chunk, n)
        basis = jnp.zeros((n, j1 - j0), dtype=jnp.float64)
        basis = basis.at[j0 + jnp.arange(j1 - j0), jnp.arange(j1 - j0)].set(1.0)
        cols.append(np.asarray(batched(basis)))
    return np.concatenate(cols, axis=1)


def _solve_tier3(
    op: KineticOperator, rhs2d: jnp.ndarray, *, tol: float, atol: float, max_dense_size: int
) -> SolveResult:
    _require_solvax()
    if _is_traced(rhs2d):
        raise RuntimeError(
            "tier-3 host sparse-direct solve is non-differentiable and cannot run "
            "under jit/vmap/grad; use method='block_tridiagonal' or 'gmres' with "
            "differentiable=True."
        )
    n = op.total_size
    if n > max_dense_size:
        raise RuntimeError(
            f"tier-3 dense materialization refused: total_size={n} > "
            f"max_dense_size={max_dense_size}; raise max_dense_size explicitly if "
            "you really want this."
        )
    print(
        f"[sfincs_jax.solve] tier-3 host sparse-direct solve (SuperLU, n={n}): "
        "non-differentiable fallback path."
    )
    import scipy.sparse as sp  # lazy: matches solvax.native's optional-scipy policy

    t0 = time.perf_counter()
    dense = materialize_dense(op)
    lu = SpluFactorization(sp.csr_matrix(dense))
    t1 = time.perf_counter()
    x2d = jnp.asarray(lu.solve(np.asarray(rhs2d)))
    if x2d.ndim == 1:
        x2d = x2d[:, None]
    t2 = time.perf_counter()
    res = _residual_norms(op, x2d, rhs2d)
    return SolveResult(
        x=x2d,
        method="direct",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


# =============================================================================
# Tier drivers
# =============================================================================


def _implicit_solve(
    op: KineticOperator,
    rhs_col: jnp.ndarray,
    fwd_solve: Callable[[jnp.ndarray], jnp.ndarray],
    t_solve: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """One differentiable column solve via ``solvax.implicit.linear_solve``.

    The single ``solver`` callable required by the API dispatches between the
    forward and transposed factorized solves by identity of the matvec it is
    handed (``linear_solve`` passes ``transpose_matvec`` through verbatim).
    """
    apply_t = _transposed_apply(op)

    def solver(mv: Callable, b: jnp.ndarray) -> jnp.ndarray:
        return t_solve(b) if mv is apply_t else fwd_solve(b)

    return solvax_linear_solve(op.apply, rhs_col, solver, transpose_matvec=apply_t)


def _solve_tier1(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    differentiable: bool,
) -> SolveResult:
    t0 = time.perf_counter()
    t1_solver = build_tier1_solver(op)
    t1 = time.perf_counter()
    if differentiable:
        cols = [
            _implicit_solve(
                op,
                rhs2d[:, j],
                lambda b: t1_solver.solve(b),
                lambda b: t1_solver.solve(b, transpose=True),
            )
            for j in range(rhs2d.shape[1])
        ]
        x2d = jnp.stack(cols, axis=1)
    else:
        x2d = t1_solver.solve(rhs2d)
    t2 = time.perf_counter()
    res = _residual_norms(op, x2d, rhs2d)
    return SolveResult(
        x=x2d,
        method="block_tridiagonal",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


def _solve_tier2(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    x0: jnp.ndarray | None,
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None,
    use_preconditioner: bool,
    drop_l_coupling_in_precond: bool,
    restart: int,
    recycle_dim: int,
    max_restarts: int,
    differentiable: bool,
) -> SolveResult:
    t0 = time.perf_counter()
    precond = precond_t = None
    if use_preconditioner:
        precond, precond_t = build_coarse_preconditioner(
            op, drop_l_coupling=drop_l_coupling_in_precond
        )
    t1 = time.perf_counter()

    x0_2d = None
    if x0 is not None:
        x0_2d, _ = _as_columns(x0)
        if x0_2d.shape != rhs2d.shape:
            raise ValueError(f"x0 shape {x0_2d.shape} must match rhs shape {rhs2d.shape}")

    apply_t = _transposed_apply(op)
    traced = _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    cols: list[jnp.ndarray] = []
    total_iters: int | None = 0
    converged = True
    res_norms: list[jnp.ndarray] = []
    for j in range(rhs2d.shape[1]):
        b = rhs2d[:, j]
        sol = gcrot(
            op.apply,
            b,
            x0=None if x0_2d is None else x0_2d[:, j],
            precond=precond,
            m=restart,
            k=recycle_dim,
            rtol=tol,
            atol=atol,
            max_restarts=max_restarts,
            recycle=recycle,
        )
        recycle = sol.recycle
        if traced:
            total_iters = None  # iteration counts are tracers under jit/grad
        else:
            total_iters += int(sol.iterations)
            converged = converged and bool(sol.converged)
        res_norms.append(sol.residual_norm)
        if differentiable:
            # Re-run under the implicit-function-theorem wrapper so gradients
            # flow (one extra solve; the adjoint uses the transposed
            # preconditioner and the same recycle-free GCROT).
            def fwd_solve(rhs_col: jnp.ndarray) -> jnp.ndarray:
                return gcrot(
                    op.apply, rhs_col, precond=precond, m=restart, k=recycle_dim,
                    rtol=tol, atol=atol, max_restarts=max_restarts,
                ).x

            def t_solve(rhs_col: jnp.ndarray) -> jnp.ndarray:
                return gcrot(
                    apply_t, rhs_col, precond=precond_t, m=restart, k=recycle_dim,
                    rtol=tol, atol=atol, max_restarts=max_restarts,
                ).x

            cols.append(_implicit_solve(op, b, fwd_solve, t_solve))
        else:
            cols.append(sol.x)
    t2 = time.perf_counter()
    return SolveResult(
        x=jnp.stack(cols, axis=1),
        method="gcrot",
        iterations=total_iters,
        residual_norms=jnp.stack(res_norms),
        converged=converged,
        recycle=recycle,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


# =============================================================================
# The auto-policy entry point
# =============================================================================


def solve(
    op: KineticOperator,
    rhs: jnp.ndarray,
    *,
    method: str = "auto",
    tol: float = 1e-10,
    atol: float = 0.0,
    x0: jnp.ndarray | None = None,
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    differentiable: bool = False,
    use_preconditioner: bool = True,
    drop_l_coupling_in_precond: bool = False,
    restart: int = 30,
    recycle_dim: int = 8,
    max_restarts: int = 200,
    max_dense_size: int = 8192,
) -> SolveResult:
    """Solve ``K x = rhs`` with the plan-§2.3 three-tier auto-policy.

    Policy (``method="auto"``):

    1. **tier 1** (``"block_tridiagonal"``) when :func:`tier1_available` —
       PAS/DKES family, exact direct solve, multi-RHS in one elimination;
    2. **tier 2** (``"gmres"``) otherwise — GCROT-recycled FGMRES on the
       matrix-free operator, right-preconditioned by an exact tier-1 solve of
       the Fortran-style simplified coarse operator;
    3. **tier 3** (``"direct"``) on explicit request, or automatically when
       tier 2 breaches its iteration cap — host SuperLU on the materialized
       matrix, non-differentiable, loud.

    Args:
        op: the kinetic operator (:class:`sfincs_jax.drift_kinetic.KineticOperator`).
        rhs: right-hand side(s), ``(total_size,)`` or ``(total_size, n_rhs)``
            — e.g. columns of :meth:`KineticOperator.rhs` for RHSMode 2/3.
        method: ``"auto"`` | ``"block_tridiagonal"`` | ``"gmres"`` |
            ``"direct"``.  Explicit tier requests raise if unsupported.
        tol: relative residual tolerance (on ``||rhs||``, per column).
        atol: absolute residual floor.
        x0: warm-start solution (tier 2), same shape as ``rhs``.
        recycle: GCROT recycle pair from a previous :class:`SolveResult`
            (tier 2 continuation warm start).
        differentiable: wrap the solution in
            ``solvax.implicit.linear_solve`` so ``jax.grad`` flows through
            (tiers 1/2; tier 3 refuses).  Tier 2 pays one extra solve.
        use_preconditioner: tier-2 coarse-operator preconditioner on/off.
        drop_l_coupling_in_precond: the Fortran ``preconditioner_xi=1`` knob
            (drop the L±1 streaming coupling in the coarse operator).
        restart: FGMRES cycle size ``m``.
        recycle_dim: GCROT recycle directions ``k``.
        max_restarts: tier-2 outer-cycle cap (the tier-3 trigger in auto).
        max_dense_size: tier-3 materialization guard.

    Returns:
        A :class:`SolveResult`; ``x`` matches the shape of ``rhs``.
    """
    _require_solvax()
    method = str(method).strip().lower()
    if method not in {"auto", "block_tridiagonal", "gmres", "direct"}:
        raise ValueError(f"unknown method {method!r}")
    rhs2d, squeeze = _as_columns(rhs)
    if rhs2d.shape[0] != op.total_size:
        raise ValueError(f"rhs has {rhs2d.shape[0]} rows; operator expects {op.total_size}")

    chosen = method
    if method == "auto":
        ok, _reason = tier1_available(op)
        chosen = "block_tridiagonal" if ok else "gmres"

    if chosen == "block_tridiagonal":
        result = _solve_tier1(op, rhs2d, tol=tol, atol=atol, differentiable=differentiable)
    elif chosen == "gmres":
        result = _solve_tier2(
            op,
            rhs2d,
            tol=tol,
            atol=atol,
            x0=x0,
            recycle=recycle,
            use_preconditioner=use_preconditioner,
            drop_l_coupling_in_precond=drop_l_coupling_in_precond,
            restart=restart,
            recycle_dim=recycle_dim,
            max_restarts=max_restarts,
            differentiable=differentiable,
        )
        if method == "auto" and not result.converged and not differentiable:
            print(
                "[sfincs_jax.solve] tier-2 Krylov breached its iteration cap "
                f"(iterations={result.iterations}); falling back to the tier-3 "
                "host direct solve."
            )
            result = _solve_tier3(
                op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size
            )
    else:  # direct
        if differentiable:
            raise RuntimeError("tier-3 (method='direct') is non-differentiable.")
        result = _solve_tier3(op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size)

    if squeeze:
        result = replace(result, x=result.x[:, 0])
    return result
