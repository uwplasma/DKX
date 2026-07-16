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
    trick ``A~ = A + gamma B C`` (algebraically exact for any ``gamma != 0``,
    inherited from the retired probing-based RHSMode=3 solver POC) and
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

import inspect
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.linalg import lu_factor, lu_solve  # noqa: E402

# solvax is a core dependency (installed automatically with sfincs_jax), but
# keep this module importable without it and raise a clear error on first use
# so broken/partial environments fail with an actionable message.
try:  # noqa: E402
    from solvax.direct import (
        BlockTridiagFactors,
        block_thomas_factor,
        block_thomas_solve,
        block_thomas_truncated_fn,
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
    block_thomas_truncated_fn = None  # type: ignore[assignment]
    solvax_linear_solve = None  # type: ignore[assignment]
    gcrot = None  # type: ignore[assignment]
    SpluFactorization = None  # type: ignore[assignment, misc]
    schur_projected_precond = None  # type: ignore[assignment]
    _SOLVAX_IMPORT_ERROR = _solvax_exc

from sfincs_jax.drift_kinetic import KineticOperator  # noqa: E402


# The nonzero-border-block ``d_block`` argument of
# :func:`solvax.operators.schur_projected_precond` was upstreamed to solvax
# (uwplasma/SOLVAX#20). Prefer it when the installed solvax exposes it; otherwise
# fall back to the local ``_bordered_schur_precond`` so this module also works
# against solvax releases that predate that argument.
_SCHUR_ACCEPTS_D_BLOCK = schur_projected_precond is not None and (
    "d_block" in inspect.signature(schur_projected_precond).parameters
)


def _require_solvax() -> None:
    """Raise a clear error when the ``solvax`` core dependency is missing."""
    if _SOLVAX_IMPORT_ERROR is not None:
        raise ImportError(
            "sfincs_jax.solve requires the 'solvax' package for its solver "
            "tiers. solvax is a core dependency: `pip install sfincs_jax` "
            "pulls it in automatically (the `sfincs_jax[structured]` extra is "
            "a no-op alias). To install it directly: `pip install solvax` or "
            "`pip install git+https://github.com/uwplasma/SOLVAX`."
        ) from _SOLVAX_IMPORT_ERROR

__all__ = [
    "SolveResult",
    "Tier1Solver",
    "build_coarse_preconditioner",
    "build_tier1_solver",
    "materialize_dense",
    "solve",
    "tier1_available",
    "tier1_full_band_bytes",
    "tier1_peak_memory_bytes",
]

# Default memory budget above which ``solve(method="auto")`` prefers the
# memory-lean truncated tier-1 kernel over the full-band factorization.  Chosen
# to match the validated HSX head-to-head benchmark
# (tools/benchmarks/tier1_hsx_head_to_head.py).  Overridable per call via the
# ``tier1_memory_budget_gb`` argument or the environment variable below.
_TIER1_BUDGET_GB_DEFAULT = 8.0
_TIER1_BUDGET_ENV = "SFINCS_TIER1_MEMORY_BUDGET_GB"

# RHSMode 1/2/3 drives (radial gradient on L=0,2; inductive E_parallel on L=1)
# and every RHSMode 1/2/3 output moment (fluxes, flows, sources, FSA
# constraints) live on the lowest three Legendre modes, so keeping three
# solution blocks is exact for the standard transport quantities.
_TIER1_KEEP_LOWEST_DEFAULT = 3


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
        timings: wall-clock seconds per phase (``build``, ``solve``).  Each
            phase ends with a ``jax.block_until_ready`` so the numbers are real
            device-compute time, not JAX async-dispatch latency (which would
            under-report by ~10x).  Under ``jit``/``grad`` the blocks are no-ops
            and the values are trace-time only.
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


def _residual_norms(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], x2d: jnp.ndarray, rhs2d: jnp.ndarray
) -> jnp.ndarray:
    res = jax.vmap(matvec, in_axes=1, out_axes=1)(x2d) - rhs2d
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


def _pinned_matvecs(
    op: KineticOperator,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Forward/transposed matvecs with the truncated ``Nxi_for_x`` DOFs pinned.

    Fortran v3 packs the ``(x, l >= Nxi_for_x(x))`` DOFs out of the matrix
    (``indices.F90`` packed indexing), so its matrix is nonsingular.  The
    rectangular jax layout keeps those DOFs as exact zero *rows* of
    :meth:`KineticOperator.apply` (with leaked nonzero *columns* from the
    x-dense Fokker-Planck blocks), i.e. the embedded operator is structurally
    singular and its transpose is inconsistent for generic adjoint cotangents
    — the root cause of the FP+constraintScheme=1 silently-wrong gradients.

    Pinning substitutes ``A_pinned = A M + (I - M)`` with ``M`` the
    active-DOF projector from :meth:`KineticOperator.active_dof_mask`:
    identical to ``A`` on the active subspace, identity on the truncated
    DOFs.  For the physical right-hand sides (zero on truncated DOFs, as
    :meth:`KineticOperator.rhs` guarantees) the solution is unchanged, and
    both ``A_pinned`` and ``A_pinned^T`` are nonsingular, so forward solves,
    transposed solves, and implicit-function-theorem adjoints are all
    well-posed.  This is exactly the packed Fortran system, extended with
    trivial identity equations on the DOFs Fortran does not carry.
    """
    apply_t_raw = _transposed_apply(op)
    mask = op.active_dof_mask()
    if mask is None:
        return op.apply, apply_t_raw

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        return op.apply(mask * v) + (1.0 - mask) * v

    def matvec_t(w: jnp.ndarray) -> jnp.ndarray:
        return mask * apply_t_raw(w) + (1.0 - mask) * w

    return matvec, matvec_t


def _convergence_guard(label: str) -> Callable[[jnp.ndarray, jnp.ndarray], None]:
    """Host callback that aborts loudly when a differentiable solve stalls.

    Used on both the forward and the adjoint (transposed) GCROT solves of the
    ``differentiable=True`` tier-2 path: a stalled Krylov solve there would
    otherwise return a silently wrong solution/gradient (the historical
    FP+constraintScheme=1 failure mode).  Runs at execution time via
    ``jax.debug.callback`` so it works under ``jit``/``grad`` tracing.
    """

    def guard(converged: jnp.ndarray, res_norm: jnp.ndarray) -> None:
        if not bool(np.asarray(converged)):
            raise RuntimeError(
                f"[sfincs_jax.solve] differentiable {label} GCROT solve failed to "
                f"converge (residual norm {float(np.asarray(res_norm)):.3e}). "
                "A stalled solve here silently corrupts gradients, so it aborts "
                "instead: the operator is likely singular (e.g. a physical null "
                "space the constraint scheme does not fix). Pass "
                "check_adjoint=False to bypass at your own risk."
            )

    return guard


# =============================================================================
# Tier 1 — structured direct (block Thomas over Legendre modes)
# =============================================================================


def tier1_available(op: KineticOperator) -> tuple[bool, str]:
    """Check whether the tier-1 structured direct family applies to ``op``.

    The decision is driven by the operator's own block extraction: if
    :meth:`KineticOperator.legendre_blocks` refuses (Er L±2 terms,
    Fokker-Planck collisions), tier 1 is off.  On top of that the bordered
    constraint machinery must be diagonal over (species, x)
    (``constraintScheme`` 0 or 2 without ``point_at_x0``).  Non-uniform
    ``Nxi_for_x`` (the production speed-dependent Legendre ramp) is accepted:
    every (species, x) subsystem is closed, so the truncated tier-1 kernel
    solves it with its own ``n_blocks = Nxi_for_x[ix]`` — exactly the packed
    Fortran system.  Only the full-band factorization
    (:func:`build_tier1_solver`) additionally requires uniform ``Nxi_for_x``;
    ramped decks always route through the truncated kernel.
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
    return True, ""


def _uniform_nxi_for_x(op: KineticOperator) -> bool:
    """Whether every speed node retains the full Legendre resolution."""
    return int(np.min(np.asarray(op.n_xi_for_x))) >= op.n_xi


# =============================================================================
# Tier 1 memory model and the full-vs-truncated route decision
# =============================================================================


def tier1_full_band_bytes(op: KineticOperator) -> float:
    """Bytes of the full tier-1 Legendre bands (``lower``/``diag``/``upper``).

    :func:`build_tier1_solver` materializes the three block-tridiagonal bands
    of :meth:`KineticOperator.to_block_tridiagonal`, each of shape
    ``(n_xi, n_species, n_x, m, m)`` with block dimension ``m = n_theta *
    n_zeta`` (the dense theta*zeta angular block per Legendre mode, per
    (species, x) subsystem), in float64::

        bytes = 3 * sum_x(Nxi_for_x) * n_species * (n_theta * n_zeta)**2 * 8

    The leading ``3`` counts ``lower``, ``diag`` and ``upper``; a subsystem at
    speed node ``ix`` carries only its own ``Nxi_for_x[ix]`` Legendre blocks
    (``sum_x(Nxi_for_x) = n_xi * n_x`` for uniform ``Nxi_for_x``).  This is
    the ~39 GB figure for the 744k-unknown uniform HSX case (n_theta=25,
    n_zeta=51, n_xi=100, n_x=5, n_species=2).
    """
    m = float(op.n_theta * op.n_zeta)
    n_blocks_total = float(np.sum(np.asarray(op.n_xi_for_x)))
    return 3.0 * n_blocks_total * float(op.n_species) * m * m * 8.0


def tier1_peak_memory_bytes(op: KineticOperator) -> float:
    """Peak-memory estimate of the full tier-1 factorization.

    Adds the block-Thomas LU factors and elimination temporaries on top of the
    three input bands (:func:`tier1_full_band_bytes`).  The
    ``BlockTridiagFactors`` store the per-block LU factors plus the two
    off-diagonal bands (~2x the band storage), and the vmapped sweep holds a
    few block temporaries live, so the peak is estimated at ``2.5x`` the band
    storage — the multiplier used by the validated HSX benchmark.
    """
    return 2.5 * tier1_full_band_bytes(op)


def _tier1_budget_bytes(budget_gb: float | None) -> tuple[float, float]:
    """Resolve the truncation budget (bytes, GB) from arg / env / default."""
    if budget_gb is None:
        env = os.environ.get(_TIER1_BUDGET_ENV)
        budget_gb = float(env) if env not in (None, "") else _TIER1_BUDGET_GB_DEFAULT
    return float(budget_gb) * 2.0**30, float(budget_gb)


def _truncation_supported(op: KineticOperator, keep: int) -> tuple[bool, str]:
    """Structural check that the truncated tier-1 kernel applies to ``op``.

    Assumes :func:`tier1_available` already passed (PAS/DKES family,
    constraintScheme in {0, 2}, no point_at_x0).  Additionally every closed
    (species, x) subsystem must retain at least ``keep`` Legendre blocks —
    the only ``Nxi_for_x`` requirement: a non-uniform ramp is solved exactly
    with ``n_blocks = Nxi_for_x[ix]`` per subsystem.
    """
    if op.constraint_scheme not in (0, 2):
        return False, f"constraintScheme={op.constraint_scheme} border couples Legendre modes"
    if op.point_at_x0:
        return False, "point_at_x0 x-grids are not handled by the truncated kernel"
    if keep > op.n_xi:
        return False, f"keep_lowest={keep} exceeds Nxi={op.n_xi}"
    if int(np.min(np.asarray(op.n_xi_for_x))) < keep:
        return False, f"min Nxi_for_x={int(np.min(np.asarray(op.n_xi_for_x)))} < keep_lowest={keep}"
    return True, ""


def _rhs_confined_to_lowest_blocks(
    op: KineticOperator, rhs2d: jnp.ndarray, keep: int
) -> bool | None:
    """Whether the RHS has Legendre support only on modes ``l < keep``.

    Returns ``None`` when ``rhs2d`` is a tracer (support cannot be read under
    jit/grad); callers then fall back to the structural ``rhs_mode`` guarantee.
    The truncated kernel computes exactly the lowest ``keep`` Legendre blocks
    and zero-pads the rest, so it is exact iff both the drive and the requested
    output moments live on ``l < keep`` — true for the RHSMode 1/2/3 transport
    drives and their fluxes/flows/sources, which touch only ``l <= 2``.
    """
    if _is_traced(rhs2d):
        return None
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    if keep >= n_xi:
        return True
    f = np.asarray(rhs2d)[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z, -1)
    return bool(np.max(np.abs(f[:, :, keep:])) == 0.0)


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
    blocks — the replacement for the retired probing-based RHSMode=3 solver POC
    — and absorbs the ``constraintScheme=2`` border with the exact rank-one
    trick ``A~ = A + gamma B C`` documented in the module docstring.

    Raises:
        NotImplementedError: when :func:`tier1_available` says no.
    """
    _require_solvax()
    ok, reason = tier1_available(op)
    if not ok:
        raise NotImplementedError(f"tier-1 structured direct path unavailable: {reason}")
    if not _uniform_nxi_for_x(op):
        raise NotImplementedError(
            "tier-1 full-band factorization requires uniform Nxi_for_x (the ramped "
            "bands carry singular zero rows on the truncated DOFs); ramped decks "
            "route through the truncated kernel (method='block_tridiagonal_truncated')"
        )

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


def _dense_collision_diagonal(mat: jnp.ndarray) -> jnp.ndarray:
    """(S, X, L) self-species, x-diagonal reduction of a dense collision block.

    ``mat`` is the ``(S, S, L, X, X)`` block layout shared by the Fokker-Planck
    (``op.fp.mat``, ``collisionOperator=1``) and improved-Sugama
    (``op.sugama.mat``, ``collisionOperator=3``) operators.  Keeping only
    ``mat[s, s, l, x, x]`` is the Fortran ``preconditioner_species=1`` +
    ``preconditioner_x=1`` simplification: it drops the cross-species and
    off-x-diagonal coupling — for the improved Sugama operator this discards the
    field-particle (momentum/energy-restoring) back-reaction entirely — leaving
    a PAS-like coefficient (diagonal in everything but L).  The dropped terms
    only degrade the *preconditioner*; the full operator GCROT solves keeps
    them, so the recycled Krylov iteration corrects the approximation.
    """
    coef = jnp.diagonal(mat, axis1=0, axis2=1)  # (L, X, X, S)
    coef = jnp.diagonal(coef, axis1=1, axis2=2)  # (L, S, X)
    return jnp.transpose(coef, (1, 2, 0))  # (S, X, L)


def _collision_phi1_diagonal(op: KineticOperator) -> jnp.ndarray:
    """(S, X, L) self-species, x-diagonal of the Phi1-in-collision operator.

    The ``includePhi1InCollisionOperator`` Fokker-Planck operator
    (``op.fp_phi1``, ``collisionOperator=0`` with poloidally varying densities)
    stores its coefficients as compact ``k_nu``/``k_cd``/``k_ce``/``k_rosen``
    kernels (not a dense ``(S,S,L,X,X)`` ``mat``), so its coarse diagonal cannot
    be sliced like :func:`_dense_collision_diagonal`.  It is however *diagonal in
    L and in ``(theta, zeta)``* (collisions are local in real space), so probing
    one constant-in-angle unit block per ``(species, x)`` and reading the
    angle-averaged self ``(s, x, l)`` response recovers the exact self-species
    x-diagonal ``preconditioner_species=1 + preconditioner_x=1`` reduction --
    the same PAS-like coefficient the ``op.fp``/``op.sugama`` branches take.  The
    densities are evaluated at ``Phi1=0`` (``n_pol=nHat``); the small Phi1 shift
    only perturbs the *preconditioner* diagonal, which GCROT corrects.
    """
    from sfincs_jax.collisions import apply_fokker_planck_v3_phi1  # noqa: PLC0415

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    ph = jnp.zeros((n_t, n_z), dtype=jnp.float64)
    k = n_s * n_x
    probes = jnp.eye(k, dtype=jnp.float64).reshape(k, n_s, n_x, 1, 1, 1) * jnp.ones(
        (1, 1, 1, n_xi, n_t, n_z), dtype=jnp.float64
    )
    y = jax.vmap(lambda f: apply_fokker_planck_v3_phi1(op.fp_phi1, f, phi1_hat=ph))(probes)
    factor = op._fs_average_factor()  # (T, Z)
    y_avg = jnp.einsum("tz,ksxltz->ksxl", factor, y) / jnp.sum(factor)
    y_avg = y_avg.reshape(n_s, n_x, n_s, n_x, n_xi)
    idx_s = jnp.arange(n_s)[:, None]
    idx_x = jnp.arange(n_x)[None, :]
    return y_avg[idx_s, idx_x, idx_s, idx_x, :]  # (S, X, L)


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


def _materialize_full_border(
    op: KineticOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Border columns ``B``, rows ``C``, and border-border block ``D`` of a Phi1 op.

    For a Phi1-augmented operator the whole block after the f-block --
    ``[ Phi1(theta,zeta) | lambda | sources ]`` of size ``p = phi1_size +
    extra_size`` -- is treated as the border of ``[[A, B], [C, D]]``.  Unlike
    the plain constraint border (``[[A, B], [C, 0]]``), ``D`` is *nonzero*: the
    quasineutrality rows carry the adiabatic ``Phi1`` diagonal and the ``+lambda``
    coupling, and the ``<Phi1>=0`` row couples ``Phi1`` (populateMatrix.F90 QN
    block).  All three pieces are probed exactly from the Jacobian JVP
    (:meth:`KineticOperator.apply`, which is ``d residual_phi1`` at
    ``phi1_lin_state``) and its transpose -- ``p`` forward + ``p`` transposed
    matvecs, cheap because the border (``~Ntheta*Nzeta``) is small.

    Returns ``(b_cols, c_rows, d_block)`` with shapes ``(f_size, p)``,
    ``(p, f_size)`` and ``(p, p)``.
    """
    n, fs = op.total_size, op.f_size
    p = n - fs
    basis = jnp.zeros((n, p), dtype=jnp.float64)
    basis = basis.at[fs + jnp.arange(p), jnp.arange(p)].set(1.0)
    applied = jax.vmap(op.apply, in_axes=1, out_axes=1)(basis)  # (n, p)
    b_cols = applied[:fs]  # f-rows response to the border columns
    d_block = applied[fs:]  # border-rows response to the border columns
    apply_t = _transposed_apply(op)
    c_rows = jax.vmap(apply_t, in_axes=1, out_axes=1)(basis)[:fs].T
    return b_cols, c_rows, d_block


def _bordered_schur_precond(
    a_inv: Callable[[jnp.ndarray], jnp.ndarray],
    b_cols: jnp.ndarray,
    c_rows: jnp.ndarray,
    d_block: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Preconditioner for ``[[A, B], [C, D]]`` from an approximate ``A^{-1}``.

    Generalizes ``solvax.operators.schur_projected_precond`` to a *nonzero*
    border-border block ``D`` (the Phi1 quasineutrality border; that function
    hard-codes ``D=0``).  Forms the small dense Schur complement
    ``S = D - C a_inv(B)`` once (``p`` coarse ``a_inv`` solves of the columns of
    ``B`` plus one ``p x p`` LU) and returns the exact inverse of the bordered
    system with ``A^{-1}`` replaced by ``a_inv`` throughout::

        y = S^{-1} (r_y - C a_inv(r_x)),    x = a_inv(r_x - B y).

    With ``a_inv`` exact the preconditioned operator is the identity; with the
    coarse (SFINCS-simplified, null-space-pinned) ``a_inv`` the border -- and in
    particular the Phi1/lambda coupling -- is still eliminated exactly through
    the projected Schur system, so a preconditioner built for the f-block ``A``
    preconditions the full Phi1-augmented system.  Reduces algebraically to
    ``schur_projected_precond`` when ``D=0`` (the constraint-only case).  Each
    application costs two ``a_inv`` calls plus one small LU triangular solve.
    """
    fs = c_rows.shape[1]
    ainv_b = jax.vmap(a_inv, in_axes=1, out_axes=1)(b_cols)  # (f_size, p)
    schur = d_block - c_rows @ ainv_b  # (p, p)
    schur_lu = lu_factor(schur)

    def precond(r: jnp.ndarray) -> jnp.ndarray:
        r_x, r_y = r[:fs], r[fs:]
        y = lu_solve(schur_lu, r_y - c_rows @ a_inv(r_x))
        x = a_inv(r_x - b_cols @ y)
        return jnp.concatenate([x, y])

    return precond


def build_coarse_preconditioner(
    op: KineticOperator, *, drop_l_coupling: bool = False
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Tier-1 exact solve of the SFINCS-simplified coarse operator, as a preconditioner.

    Mirrors the Fortran ``preconditionerOptions`` defaults: collisions become
    self-species and x-diagonal (the dense (species, x)-coupled Fokker-Planck
    ``op.fp`` and improved-Sugama ``op.sugama`` operators reduce to their
    PAS-like L-diagonal — for Sugama this drops the field-particle
    momentum/energy-restoring coupling, kept only in the full operator GCROT
    solves), the Er L±2 xDot/xiDot terms and the tangential magnetic-drift
    L±2 terms are dropped, and (optionally, the ``preconditioner_xi=1`` knob)
    the L±1 streaming coupling is dropped too.
    The result is block-tridiagonal over L and uncoupled over (species, x), so
    one batched block-Thomas factorization inverts it exactly; the bordered
    constraint rows of the *full* operator are then eliminated exactly with
    ``solvax.operators.schur_projected_precond``.

    When ``op.include_phi1`` the operator is the Jacobian of the nonlinear Phi1
    residual and its border is the whole quasineutrality block
    ``[Phi1(theta,zeta) | lambda | sources]`` with a *nonzero* border-border
    block ``D`` (the QN adiabatic Phi1 diagonal, the ``+lambda`` coupling, and
    the ``<Phi1>=0`` row).  That full border is eliminated exactly with the
    generalized bordered Schur complement (:func:`_bordered_schur_precond`) --
    the coarse f-block solve plus a dense ``~Ntheta*Nzeta`` Schur solve -- so
    the coarse preconditioner is Phi1-aware and the Newton inner Krylov solve
    converges in far fewer iterations (:func:`sfincs_jax.phi1.solve_phi1`).

    Returns:
        ``(precond, precond_t)`` — approximate inverses of ``K`` and ``K^T``
        on flat ``(total_size,)`` vectors, sharing one factorization.
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    stripped = replace(
        op, fp=None, sugama=None, fp_phi1=None, with_er_xidot=False, with_er_xdot=False,
        with_magnetic_drifts=False,
        external_phi1_hat=None, include_phi1=False, include_phi1_in_kinetic=False,
    )
    blocks = stripped.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower, diag, upper = (jnp.transpose(a, (1, 2, 0, 3, 4)) for a in blocks)  # (S,X,L,TZ,TZ)

    eye = jnp.eye(n_tz, dtype=jnp.float64)
    # Keep the Nxi_for_x truncation mask as a jnp array (no host materialization)
    # so the coarse preconditioner stays traceable when the operator leaves are
    # tracers (jit-over-leaves / vmap / the differentiable kernel).  The shape is
    # static; only the boolean pattern depends on the traced ``n_xi_for_x``.
    mask = op._mask()  # (X, L)
    # Add back the dense (species, x)-coupled collision operators — Fokker-Planck
    # (collisionOperator=1) or the improved Sugama model (collisionOperator=3) —
    # reduced to their PAS-like self-species x-diagonal.  For Sugama this drops
    # the field-particle momentum/energy-restoring coupling from the coarse
    # operator (kept only in the full operator GCROT solves); the two collision
    # models are mutually exclusive, so at most one branch fires.
    for coll in (op.fp, op.sugama):
        if coll is not None:
            coef = _dense_collision_diagonal(coll.mat) * mask[None, :, :]  # (S, X, L)
            diag = diag + coef[:, :, :, None, None] * eye[None, None, None, :, :]
    # includePhi1InCollisionOperator (op.fp_phi1): the poloidally varying FP
    # operator has no dense ``mat`` to slice, so probe its exact self-species
    # x-diagonal (L- and angle-diagonal) instead.  Without this the coarse
    # f-block of a Phi1-in-collision deck is collisionless and singular.
    if op.fp_phi1 is not None:
        coef = _collision_phi1_diagonal(op) * mask[None, :, :]  # (S, X, L)
        diag = diag + coef[:, :, :, None, None] * eye[None, None, None, :, :]
    if drop_l_coupling:
        lower = jnp.zeros_like(lower)
        upper = jnp.zeros_like(upper)

    # Invertibility floor.  A purely collisionless, drift-free coarse f-block
    # (``nu_n=0`` with ``Er=0`` -> no PAS/collision *and* no ExB diagonal) has
    # EXACTLY zero diagonal blocks: only streaming/mirror couple L, so the
    # block-Thomas factorization would divide by zero.  (The non-Phi1 tier-2
    # path never hit this -- collisionless PAS decks route to the tier-1 direct
    # solver -- but the Phi1 Newton inner solve forces the coarse preconditioner
    # for every deck.)  Add a small per-(species, x) diagonal floor scaled by the
    # band magnitude so every diagonal block is invertible; it is negligible
    # against a real collision/ExB diagonal (and GCROT corrects the coarse
    # operator regardless), and it degrades gracefully toward a well-scaled
    # identity when the f-block is genuinely singular.
    band = jnp.maximum(
        jnp.max(jnp.abs(diag), axis=(2, 3, 4)),
        jnp.maximum(jnp.max(jnp.abs(lower), axis=(2, 3, 4)), jnp.max(jnp.abs(upper), axis=(2, 3, 4))),
    )  # (S, X)
    band = jnp.where(band > 0.0, band, 1.0)
    # 1e-8 (relative to the band) is tiny enough to leave a real diagonal -- and
    # the tightly-clustered preconditioning it gives -- untouched, yet keeps the
    # all-zero-diagonal collisionless case out of an exact-zero pivot.
    diag = diag + (1e-8 * band)[:, :, None, None, None] * eye[None, None, None, :, :]

    # Masked (x, l) rows are identically zero in the operator: pin them with
    # the identity so the coarse factorization stays nonsingular.
    pin = 1.0 - mask  # (X, L)
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
    if op.include_phi1:
        # Phi1-augmented operator: the border is the whole quasineutrality block
        # ``[Phi1(theta,zeta) | lambda | sources]`` with a NONZERO border-border
        # block ``D`` (the QN rows carry the adiabatic Phi1 diagonal + the
        # ``+lambda`` coupling and the ``<Phi1>=0`` row couples Phi1), so the
        # constraint-only ``schur_projected_precond`` (which assumes ``D=0``)
        # does not apply.  Eliminate the full border exactly with the generalized
        # bordered Schur complement -- the coarse tier-1 f-block solve plus a
        # dense ``p x p`` (``p ~ Ntheta*Nzeta``) Schur solve over the Phi1/border
        # block.  The Phi1->f coupling (``B``), the QN-from-f rows (``C``) and the
        # Phi1/lambda border block (``D``) are all probed exactly from the
        # Jacobian JVP, so only the f-block is approximated (GCROT corrects it).
        b_cols, c_rows, d_block = _materialize_full_border(op)
        if _SCHUR_ACCEPTS_D_BLOCK:
            precond = schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block)
            precond_t = schur_projected_precond(
                a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T
            )
        else:
            precond = _bordered_schur_precond(a_inv, b_cols, c_rows, d_block)
            precond_t = _bordered_schur_precond(a_inv_t, c_rows.T, b_cols.T, d_block.T)
        return precond, precond_t
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = _materialize_borders(op)
    precond = schur_projected_precond(a_inv, b_cols, c_rows)
    precond_t = schur_projected_precond(a_inv_t, c_rows.T, b_cols.T)
    return precond, precond_t


# =============================================================================
# Tier 3 — host sparse-direct fallback
# =============================================================================


def materialize_dense(
    op: KineticOperator, *, column_chunk: int = 1024, pin_masked_dofs: bool = False
) -> np.ndarray:
    """Materialize the full bordered operator as a dense numpy matrix.

    Applies the matrix-free operator to identity columns in vmapped chunks.
    Meant for tiny systems (tier-3 fallback and referee tests) — memory is
    ``O(total_size**2)``.

    Args:
        op: the kinetic operator.
        column_chunk: identity columns per vmapped batch.
        pin_masked_dofs: materialize the pinned operator (identity rows and
            columns on the DOFs truncated by ``Nxi_for_x``; see
            :func:`_pinned_matvecs`) instead of the raw rectangular embedding,
            which has exact zero rows on those DOFs.
    """
    n = op.total_size
    apply = _pinned_matvecs(op)[0] if pin_masked_dofs else op.apply
    batched = jax.jit(jax.vmap(apply, in_axes=1, out_axes=1))
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
    dense = materialize_dense(op, pin_masked_dofs=True)
    lu = SpluFactorization(sp.csr_matrix(dense))
    t1 = time.perf_counter()
    x2d = jnp.asarray(lu.solve(np.asarray(rhs2d)))
    if x2d.ndim == 1:
        x2d = x2d[:, None]
    t2 = time.perf_counter()
    res = _residual_norms(_pinned_matvecs(op)[0], x2d, rhs2d)
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
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    matvec_t: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_col: jnp.ndarray,
    fwd_solve: Callable[[jnp.ndarray], jnp.ndarray],
    t_solve: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """One differentiable column solve via ``solvax.implicit.linear_solve``.

    The single ``solver`` callable required by the API dispatches between the
    forward and transposed factorized solves by identity of the matvec it is
    handed (``linear_solve`` passes ``transpose_matvec`` through verbatim).
    """

    def solver(mv: Callable, b: jnp.ndarray) -> jnp.ndarray:
        return t_solve(b) if mv is matvec_t else fwd_solve(b)

    return solvax_linear_solve(matvec, rhs_col, solver, transpose_matvec=matvec_t)


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
    # Force the async block-Thomas factorization to complete so the "build"
    # timing reflects real compute, not JAX dispatch latency.  We block on the
    # array fields (the Tier1Solver dataclass itself is not a pytree, so
    # block_until_ready would treat it as an opaque leaf); a no-op under
    # jit/grad tracing.
    jax.block_until_ready(
        (t1_solver.factors, t1_solver.z_fwd, t1_solver.z_t, t1_solver.gamma)
    )
    t1 = time.perf_counter()

    def _solve_refined(b: jnp.ndarray, *, transpose: bool = False) -> jnp.ndarray:
        """Factor solve plus one iterative-refinement step.

        The block-Thomas elimination is backward-stable but its rounding can
        leave the true relative residual a small multiple of eps above the
        strict production gate; one refinement pass
        ``x += solve(b - A x)`` (one extra apply and substitution on the
        existing factors) takes it to O(1e-16).
        """
        apply = _transposed_apply(op) if transpose else op.apply
        apply2d = apply if b.ndim == 1 else jax.vmap(apply, in_axes=1, out_axes=1)
        x = t1_solver.solve(b, transpose=transpose)
        return x + t1_solver.solve(b - apply2d(x), transpose=transpose)

    if differentiable:
        apply_t = _transposed_apply(op)
        cols = [
            _implicit_solve(
                op.apply,
                apply_t,
                rhs2d[:, j],
                lambda b: _solve_refined(b),
                lambda b: _solve_refined(b, transpose=True),
            )
            for j in range(rhs2d.shape[1])
        ]
        x2d = jnp.stack(cols, axis=1)
    else:
        x2d = _solve_refined(rhs2d)
    x2d = jax.block_until_ready(x2d)  # real solve compute, not just dispatch
    t2 = time.perf_counter()
    res = _residual_norms(op.apply, x2d, rhs2d)
    return SolveResult(
        x=x2d,
        method="block_tridiagonal",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


# =============================================================================
# Tier 1 (truncated) — memory-lean block Thomas over the lowest K Legendre modes
# =============================================================================


def _truncated_coefficients(op: KineticOperator) -> dict[str, jnp.ndarray]:
    """Compact per-term coefficient matrices for the on-the-fly Legendre blocks.

    Mirrors :meth:`KineticOperator.legendre_blocks` exactly (same analytic
    streaming/mirror/ExB/PAS coefficients), but keeps only the per-term factors
    so the ``(m, m)`` blocks can be assembled inside
    ``solvax.direct.block_thomas_truncated_fn`` without ever materializing the
    full ``(n_xi, ...)`` bands.  Everything here is a differentiable function of
    the operator pytree, so gradients flow to the physics inputs.
    """
    n_tz = op.n_theta * op.n_zeta
    eye_t = jnp.eye(op.n_theta, dtype=jnp.float64)
    eye_z = jnp.eye(op.n_zeta, dtype=jnp.float64)
    d_theta_tz = jnp.kron(op.ddtheta, eye_z)
    d_zeta_tz = jnp.kron(eye_t, op.ddzeta)

    sqrt_t_over_m = jnp.sqrt(op.t_hat / op.m_hat)  # (S,)
    v_theta = (op.b_hat_sup_theta / op.b_hat).reshape((-1,))
    v_zeta = (op.b_hat_sup_zeta / op.b_hat).reshape((-1,))
    stream = sqrt_t_over_m[:, None, None] * (
        v_theta[None, :, None] * d_theta_tz[None, :, :]
        + v_zeta[None, :, None] * d_zeta_tz[None, :, :]
    )  # (S, TZ, TZ)
    mirror_geom = op.b_hat_sup_theta * op.db_hat_dtheta + op.b_hat_sup_zeta * op.db_hat_dzeta
    mirror = -sqrt_t_over_m[:, None] * (mirror_geom / (2.0 * op.b_hat**2)).reshape((-1,))[None, :]
    if op.with_exb:
        coef_theta, coef_zeta = op._exb_coefficients()
        exb = (
            coef_theta.reshape((-1,))[:, None] * d_theta_tz
            + coef_zeta.reshape((-1,))[:, None] * d_zeta_tz
        )  # (TZ, TZ)
    else:
        exb = jnp.zeros((n_tz, n_tz), dtype=jnp.float64)

    b0 = jnp.ones((n_tz,), dtype=jnp.float64)
    c0 = op._fs_average_factor().reshape((-1,))
    pas_coef = op.pas.coef if op.pas is not None else jnp.zeros(
        (op.n_species, op.n_x, op.n_xi), dtype=jnp.float64
    )  # (S, X, L)
    # Conditioning-friendly rank-one scale per (S, X) (any nonzero value is
    # algebraically exact) — identical recipe to the benchmark's TruncatedTier1.
    exb_diag_mean = jnp.mean(jnp.abs(jnp.diagonal(exb)))
    scale = jnp.mean(jnp.abs(pas_coef), axis=2) + exb_diag_mean  # (S, X)
    scale = jnp.where(scale > 0.0, scale, 1.0)
    gamma = scale / jnp.max(jnp.abs(c0))  # (S, X)
    return {
        "stream": stream, "mirror": mirror, "exb": exb, "pas": pas_coef,
        "cl": op.xi_coupling_lower, "cu": op.xi_coupling_upper,
        "b0": b0, "c0": c0, "gamma": gamma,
    }  # fmt: skip


def _truncated_block_fn(
    coef: dict[str, jnp.ndarray],
    n_xi: int,
    stream: jnp.ndarray,
    mirror: jnp.ndarray,
    pas_row: jnp.ndarray,
    x_val: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    shift_border: bool,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Analytic ``(L_k, D_k, U_k)`` for one (species, x) subsystem — as legendre_blocks.

    With ``shift_border`` the rank-one border ``gamma * outer(b0, c0)`` is added
    to the ``l=0`` diagonal block (the exact ``A~ = A + gamma B C`` absorption);
    without it the raw physical blocks are returned (used for residual checks).
    """
    exb, b0, c0, cl, cu = coef["exb"], coef["b0"], coef["c0"], coef["cl"], coef["cu"]
    m = exb.shape[0]
    idx = jnp.arange(m)

    def block_fn(k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        kf = k.astype(jnp.float64)
        cl_k = jnp.take(cl, k)
        lower = (x_val * cl_k) * stream
        lower = lower.at[idx, idx].add((x_val * (-cl_k * (kf - 1.0))) * mirror)
        cu_k = jnp.take(cu, jnp.minimum(k, n_xi - 1))
        upper = (x_val * cu_k) * stream
        upper = upper.at[idx, idx].add((x_val * (cu_k * (kf + 2.0))) * mirror)
        diag = exb.at[idx, idx].add(jnp.take(pas_row, k))
        if shift_border:
            diag = jnp.where(k == 0, diag + gamma * jnp.outer(b0, c0), diag)
        return lower, diag, upper

    return block_fn


def _solve_tier1_truncated(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    keep: int,
    tol: float,
    atol: float,
) -> SolveResult:
    """Memory-lean tier-1 solve: only the lowest ``keep`` Legendre blocks.

    Assembles each ``(m, m)`` Legendre block on the fly inside
    ``solvax.direct.block_thomas_truncated_fn`` (peak memory ``O(keep * m^2)``
    per subsystem, independent of ``n_xi``), so the ~39 GB full-band storage is
    never allocated.  The ``constraintScheme=2`` border is absorbed with the
    same exact rank-one trick as :func:`build_tier1_solver`.  The lowest
    ``keep`` blocks (and the source unknowns) are exact; blocks ``l >= keep``
    are zero-padded — valid because the drive and all requested output moments
    live on ``l < keep`` (see :func:`_rhs_confined_to_lowest_blocks`).

    Non-uniform ``Nxi_for_x`` (the production speed-dependent Legendre ramp)
    is exact too: each (species, x) subsystem is closed, so it is eliminated
    with its own ``n_blocks = Nxi_for_x[ix]`` — precisely the packed Fortran
    discretization (``indices.F90``), whose truncated DOFs the zero-padded
    ``l >= keep`` tail already covers (``keep <= min Nxi_for_x`` is enforced
    by :func:`_truncation_supported`).

    Differentiability: the whole solve is a pure-JAX composition of
    ``block_thomas_truncated_fn`` sweeps, so ``jax.grad`` differentiates
    straight through it (the block-Thomas scan is short and reverse-mode
    cheap).  It is *not* wrapped in the full-operator implicit-function-theorem
    adjoint used by the full tier-1/tier-2 paths: this solve inverts the
    *reduced* Schur-complemented operator on the lowest ``keep`` blocks, not
    the full band, so a full-operator ``A^T`` adjoint would be inconsistent and
    silently corrupt gradients.  Direct autodiff is exact here.
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x
    n_rhs = rhs2d.shape[1]
    cs = op.constraint_scheme

    t0 = time.perf_counter()
    coef = _truncated_coefficients(op)

    # Per-subsystem inputs, flattened to batch index b = s * n_x + x (matching
    # the (S, X) reshape used by KineticOperator.apply / build_tier1_solver).
    stream_b = jnp.repeat(coef["stream"], n_x, axis=0)  # (B, TZ, TZ)
    mirror_b = jnp.repeat(coef["mirror"], n_x, axis=0)  # (B, TZ)
    pas_b = coef["pas"].reshape(batch, n_xi)  # (B, L)
    x_b = jnp.tile(op.x, n_s)  # (B,)
    gamma_b = coef["gamma"].reshape(batch)  # (B,)
    b0, c0 = coef["b0"], coef["c0"]

    rhs_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
    rhs_low_b = rhs_f[:, :, :keep].reshape(batch, keep, n_tz, n_rhs)  # (B, keep, TZ, R)
    if cs == 2:
        r_c_b = rhs2d[op.f_size :].reshape(batch, n_rhs)  # (B, R)
    else:
        r_c_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)

    def solve_one(inputs, n_blocks: int):
        stream, mirror, pas_row, x_val, gamma, rhs_low, r_c = inputs
        block_fn = _truncated_block_fn(
            coef, n_xi, stream, mirror, pas_row, x_val, gamma, shift_border=(cs == 2)
        )
        if cs == 2:
            z_col = jnp.zeros((keep, n_tz, 1), dtype=jnp.float64).at[0, :, 0].set(b0)
            rhs_stack = jnp.concatenate([rhs_low, z_col], axis=2)  # (keep, TZ, R+1)
            sol = block_thomas_truncated_fn(block_fn, n_blocks, rhs_stack, keep)
            y = sol[:, :, :n_rhs]  # (keep, TZ, R)
            z = sol[:, :, n_rhs]  # (keep, TZ)
            c_y0 = c0 @ y[0]  # (R,)
            c_z0 = c0 @ z[0]  # scalar
            shift = (c_y0 - r_c) / c_z0  # (R,)
            s = gamma * r_c + shift  # (R,)
            f_low = y - shift[None, None, :] * z[:, :, None]  # (keep, TZ, R)
            return f_low, s
        sol = block_thomas_truncated_fn(block_fn, n_blocks, rhs_low, keep)  # (keep, TZ, R)
        return sol, jnp.zeros((n_rhs,), dtype=jnp.float64)

    # lax.map processes one subsystem at a time (scan-based) so the peak stays
    # at a single subsystem's O(keep * m^2) working set, not B times it.
    if _uniform_nxi_for_x(op):
        f_low_b, s_b = jax.lax.map(
            lambda t: solve_one(t, n_xi),
            (stream_b, mirror_b, pas_b, x_b, gamma_b, rhs_low_b, r_c_b),
        )
    else:
        # Ramped Nxi_for_x: each (species, x) subsystem is closed, so it is
        # eliminated with its own static n_blocks = Nxi_for_x[ix] (the packed
        # Fortran discretization) — group per speed node, map over species.
        rhs_low_sx = rhs_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs)
        r_c_sx = r_c_b.reshape(n_s, n_x, n_rhs)
        f_parts: list[jnp.ndarray] = []
        s_parts: list[jnp.ndarray] = []
        for ix, nb in enumerate(int(v) for v in np.asarray(op.n_xi_for_x)):
            f_ix, s_ix = jax.lax.map(
                lambda t, nb=nb: solve_one(t, nb),
                (
                    coef["stream"], coef["mirror"], coef["pas"][:, ix],
                    jnp.broadcast_to(op.x[ix], (n_s,)), coef["gamma"][:, ix],
                    rhs_low_sx[:, ix], r_c_sx[:, ix],
                ),
            )  # fmt: skip
            f_parts.append(f_ix)
            s_parts.append(s_ix)
        f_low_b = jnp.stack(f_parts, axis=1).reshape(batch, keep, n_tz, n_rhs)
        s_b = jnp.stack(s_parts, axis=1).reshape(batch, n_rhs)
    # Force the async truncated block-Thomas sweep to complete so the timing is
    # real compute, not JAX dispatch latency (a no-op under jit/grad tracing).
    f_low_b, s_b = jax.block_until_ready((f_low_b, s_b))
    t1 = time.perf_counter()

    f_full = jnp.zeros((n_s, n_x, n_xi, n_tz, n_rhs), dtype=jnp.float64)
    f_full = f_full.at[:, :, :keep].set(f_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs))
    parts = [f_full.reshape(op.f_size, n_rhs)]
    if op.extra_size:
        parts.append(s_b.reshape(op.extra_size, n_rhs))
    x2d = jnp.concatenate(parts, axis=0)

    res = _truncated_partial_residual(op, coef, stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b, keep)
    x2d, res = jax.block_until_ready((x2d, res))  # real residual/assembly time, not dispatch
    t2 = time.perf_counter()
    return SolveResult(
        x=x2d,
        method="block_tridiagonal_truncated",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


def _truncated_partial_residual(
    op: KineticOperator,
    coef: dict[str, jnp.ndarray],
    stream_b: jnp.ndarray,
    mirror_b: jnp.ndarray,
    pas_b: jnp.ndarray,
    x_b: jnp.ndarray,
    f_low_b: jnp.ndarray,
    s_b: jnp.ndarray,
    rhs_low_b: jnp.ndarray,
    r_c_b: jnp.ndarray,
    keep: int,
) -> jnp.ndarray:
    """Residual over the rows fully determined by the computed lowest-K blocks.

    Legendre row ``l`` couples to columns ``l-1, l, l+1``, so rows
    ``l = 0 .. keep-2`` (plus the ``constraintScheme=2`` FSA rows) are entirely
    fixed by the ``keep`` computed blocks and must vanish to machine precision;
    row ``keep-1`` couples to the (deliberately unsolved) block ``keep`` and is
    excluded.  This mirrors ``TruncatedTier1.partial_residual`` and is the
    honest convergence signal for the truncated solve.  Returns per-column
    norms of shape ``(n_rhs,)``.
    """
    cs = op.constraint_scheme
    n_rhs = rhs_low_b.shape[-1]
    b0, c0 = coef["b0"], coef["c0"]

    def per_subsystem(inputs):
        stream, mirror, pas_row, x_val, f_low, s, rhs_low, r_c = inputs
        raw = _truncated_block_fn(
            coef, op.n_xi, stream, mirror, pas_row, x_val, 0.0, shift_border=False
        )
        acc = jnp.zeros((n_rhs,), dtype=jnp.float64)
        for ell in range(keep - 1):
            lo, di, up = raw(jnp.asarray(ell, dtype=jnp.int32))
            r = jnp.einsum("ij,jr->ir", di, f_low[ell]) - rhs_low[ell]
            if ell > 0:
                r = r + jnp.einsum("ij,jr->ir", lo, f_low[ell - 1])
            r = r + jnp.einsum("ij,jr->ir", up, f_low[ell + 1])
            if ell == 0 and cs == 2:
                r = r + b0[:, None] * s[None, :]
            acc = acc + jnp.sum(r * r, axis=0)
        if cs == 2:
            rc = (c0 @ f_low[0]) - r_c  # (R,)
            acc = acc + rc * rc
        return acc

    sq = jax.lax.map(
        per_subsystem, (stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b)
    )  # (B, R)
    return jnp.sqrt(jnp.sum(sq, axis=0))


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
    check_adjoint: bool,
) -> SolveResult:
    traced = _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    t0 = time.perf_counter()
    precond = precond_t = None
    if use_preconditioner:
        precond, precond_t = build_coarse_preconditioner(
            op, drop_l_coupling=drop_l_coupling_in_precond
        )
        # The preconditioner closure captures the async coarse block-Thomas
        # factorization; force it to complete (a zero probe) so the "build"
        # timing is real compute, not JAX dispatch latency.  Skipped under
        # jit/grad tracing, where block_until_ready is a no-op on tracers and
        # the probe would only add dead nodes to the trace.
        if not traced:
            jax.block_until_ready(precond(jnp.zeros((op.total_size,), dtype=jnp.float64)))
    t1 = time.perf_counter()

    x0_2d = None
    if x0 is not None:
        x0_2d, _ = _as_columns(x0)
        if x0_2d.shape != rhs2d.shape:
            raise ValueError(f"x0 shape {x0_2d.shape} must match rhs shape {rhs2d.shape}")

    # Pinned matvecs: identical to op.apply on the physical subspace, identity
    # on the Nxi_for_x-truncated DOFs, so the system (and in particular its
    # transpose, used by the differentiable adjoint) is nonsingular.
    matvec, matvec_t = _pinned_matvecs(op)
    cols: list[jnp.ndarray] = []
    total_iters: int | None = 0
    converged = True
    res_norms: list[jnp.ndarray] = []
    for j in range(rhs2d.shape[1]):
        b = rhs2d[:, j]
        sol = gcrot(
            matvec,
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
            # preconditioner and the same recycle-free GCROT).  With
            # check_adjoint on, both the forward and the adjoint solves abort
            # loudly on non-convergence instead of silently corrupting the
            # gradient (jax.debug.callback fires at execution time).
            def fwd_solve(rhs_col: jnp.ndarray) -> jnp.ndarray:
                s = gcrot(
                    matvec, rhs_col, precond=precond, m=restart, k=recycle_dim,
                    rtol=tol, atol=atol, max_restarts=max_restarts,
                )
                if check_adjoint:
                    jax.debug.callback(
                        _convergence_guard("forward"), s.converged, s.residual_norm
                    )
                return s.x

            def t_solve(rhs_col: jnp.ndarray) -> jnp.ndarray:
                s = gcrot(
                    matvec_t, rhs_col, precond=precond_t, m=restart, k=recycle_dim,
                    rtol=tol, atol=atol, max_restarts=max_restarts,
                )
                if check_adjoint:
                    jax.debug.callback(
                        _convergence_guard("adjoint (transposed)"),
                        s.converged,
                        s.residual_norm,
                    )
                return s.x

            cols.append(_implicit_solve(matvec, matvec_t, b, fwd_solve, t_solve))
        else:
            cols.append(sol.x)
    x_stacked = jax.block_until_ready(jnp.stack(cols, axis=1))  # real solve time, not dispatch
    t2 = time.perf_counter()
    return SolveResult(
        x=x_stacked,
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


def _auto_route(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    budget_gb: float | None,
    keep_lowest: int,
) -> str:
    """Pick the tier for ``method="auto"`` and print a Fortran-style one-liner."""
    ok, _reason = tier1_available(op)
    if not ok:
        return "gmres"

    peak = tier1_peak_memory_bytes(op)
    bands = tier1_full_band_bytes(op)
    budget_bytes, budget_gb_val = _tier1_budget_bytes(budget_gb)
    peak_gb = peak / 2.0**30
    uniform = _uniform_nxi_for_x(op)
    if uniform and peak <= budget_bytes:
        print(
            f"[sfincs_jax.solve] tier-1 route: full factorization; "
            f"peak estimate {peak_gb:.2f} GB <= budget {budget_gb_val:.1f} GB "
            f"(bands {bands / 2.0**30:.2f} GB x2.5)."
        )
        return "block_tridiagonal"

    keep = min(keep_lowest, op.n_xi)
    sup_ok, sup_reason = _truncation_supported(op, keep)
    rhs_ok = _rhs_confined_to_lowest_blocks(op, rhs2d, keep)
    # Under trace the RHS support is unreadable; trust the structural RHSMode
    # 1/2/3 guarantee (drives + moments on l <= 2).
    rhs_valid = rhs_ok if rhs_ok is not None else (int(op.rhs_mode) in (1, 2, 3))
    if sup_ok and rhs_valid:
        because = (
            "non-uniform Nxi_for_x (per-subsystem n_blocks = Nxi_for_x[ix]; "
            "the full bands do not support the ramp)"
            if not uniform
            else f"peak estimate {peak_gb:.2f} GB > budget {budget_gb_val:.1f} GB "
            f"(bands {bands / 2.0**30:.2f} GB x2.5)"
        )
        print(
            f"[sfincs_jax.solve] tier-1 route: truncated block-Thomas "
            f"(keep_lowest={keep}); {because}, "
            f"solving the lowest {keep} Legendre blocks."
        )
        return "block_tridiagonal_truncated"

    why = sup_reason if not sup_ok else "RHS/output needs Legendre modes l >= keep"
    blocker = (
        "non-uniform Nxi_for_x rules out the full bands"
        if not uniform
        else f"full-band estimate {peak_gb:.2f} GB > budget {budget_gb_val:.1f} GB"
    )
    print(
        f"[sfincs_jax.solve] tier-1 route: {blocker} but truncation is invalid "
        f"({why}); falling back to tier-2 GCROT."
    )
    return "gmres"


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
    check_adjoint: bool = True,
    use_preconditioner: bool = True,
    drop_l_coupling_in_precond: bool = False,
    restart: int = 30,
    recycle_dim: int = 8,
    max_restarts: int = 200,
    max_dense_size: int = 8192,
    tier1_memory_budget_gb: float | None = None,
    tier1_keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
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
        check_adjoint: (differentiable tier 2 only, default on) abort loudly
            — a ``RuntimeError`` raised from a ``jax.debug.callback`` at
            execution time — when the forward or the adjoint (transposed)
            GCROT solve fails to converge.  A stalled Krylov solve under the
            implicit-function-theorem wrapper otherwise returns silently
            wrong gradients; this is how the singular FP+constraintScheme=1
            embedding used to fail before truncated-DOF pinning (see below).

    Operators with a truncated Legendre resolution (non-uniform ``Nxi_for_x``)
    are structurally singular in the rectangular state layout: the truncated
    DOFs are exact zero rows of :meth:`KineticOperator.apply` (Fortran v3
    never carries them — packed indexing in ``indices.F90``).  Tiers 2 and 3
    therefore solve the *pinned* system ``(A M + I - M) x = rhs`` with ``M``
    the active-DOF projector (:meth:`KineticOperator.active_dof_mask`): it is
    nonsingular, agrees with ``A`` on the physical subspace, and forces
    ``x = rhs = 0`` on the truncated DOFs, so solutions, residuals, and
    implicit-function-theorem gradients all match the packed Fortran system.
    The truncated tier-1 kernel is consistent with the same pinning: it
    eliminates each closed (species, x) subsystem with its own
    ``n_blocks = Nxi_for_x[ix]`` (exactly the packed Fortran system) and
    zero-pads everything above, so ramped PAS/DKES decks route through it;
    only the full tier-1 factorization requires uniform ``Nxi_for_x``.
        use_preconditioner: tier-2 coarse-operator preconditioner on/off.
        drop_l_coupling_in_precond: the Fortran ``preconditioner_xi=1`` knob
            (drop the L±1 streaming coupling in the coarse operator).
        restart: FGMRES cycle size ``m``.
        recycle_dim: GCROT recycle directions ``k``.
        max_restarts: tier-2 outer-cycle cap (the tier-3 trigger in auto).
        max_dense_size: tier-3 materialization guard.
        tier1_memory_budget_gb: budget (GB) above which ``method="auto"``
            prefers the memory-lean truncated tier-1 kernel over the full-band
            factorization.  ``None`` reads the ``SFINCS_TIER1_MEMORY_BUDGET_GB``
            environment variable, else the 8 GB default.  The full-band peak is
            estimated by :func:`tier1_peak_memory_bytes`.
        tier1_keep_lowest: number of Legendre blocks the truncated tier-1
            kernel computes exactly (default 3 — the RHSMode 1/2/3 drives and
            output moments live on ``l <= 2``).

    Auto-policy tier-1 routing (``method="auto"``, :func:`tier1_available` true):

    ================================  =============================================
    condition                         route
    ================================  =============================================
    uniform, peak estimate <= budget  full ``"block_tridiagonal"`` (any output
                                      mode, multi-RHS factor reuse)
    ramped Nxi_for_x or estimate >    ``"block_tridiagonal_truncated"`` when the
    budget                            truncation is valid (lowest ``keep`` blocks
                                      only, ~O(keep m^2) memory; ramps solved with
                                      per-subsystem ``n_blocks = Nxi_for_x[ix]``)
    …and truncation invalid           ``"gcrot"`` tier 2, with a printed notice
                                      (high-l output the truncation cannot supply)
    ================================  =============================================

    "Truncation valid" means the operator admits the truncated kernel
    (:func:`_truncation_supported`) and the RHS support is confined to
    ``l < keep`` (:func:`_rhs_confined_to_lowest_blocks`; under jit/grad the
    structural ``rhs_mode in {1,2,3}`` guarantee is used).

    Returns:
        A :class:`SolveResult`; ``x`` matches the shape of ``rhs``.
    """
    _require_solvax()
    method = str(method).strip().lower()
    if method not in {
        "auto", "block_tridiagonal", "block_tridiagonal_truncated", "gmres", "direct"
    }:
        raise ValueError(f"unknown method {method!r}")
    if method == "block_tridiagonal_truncated":
        ok, reason = tier1_available(op)
        if not ok:
            raise NotImplementedError(f"tier-1 truncated path unavailable: {reason}")
        keep = min(tier1_keep_lowest, op.n_xi)
        sup_ok, sup_reason = _truncation_supported(op, keep)
        if not sup_ok:
            raise NotImplementedError(f"tier-1 truncated path unavailable: {sup_reason}")
    rhs2d, squeeze = _as_columns(rhs)
    if rhs2d.shape[0] != op.total_size:
        raise ValueError(f"rhs has {rhs2d.shape[0]} rows; operator expects {op.total_size}")

    chosen = method
    if method == "auto":
        chosen = _auto_route(op, rhs2d, tier1_memory_budget_gb, tier1_keep_lowest)

    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        if chosen == "block_tridiagonal":
            result = _solve_tier1(op, rhs2d, tol=tol, atol=atol, differentiable=differentiable)
        else:
            keep = min(tier1_keep_lowest, op.n_xi)
            result = _solve_tier1_truncated(op, rhs2d, keep=keep, tol=tol, atol=atol)
        if method == "auto" and not result.converged and not differentiable:
            # Structured elimination has no pivoting across blocks: on
            # near-singular systems (e.g. a nu_n=0 collisionless deck, whose
            # bordered constraint leaves the operator with condition numbers
            # ~1e18) its residual can miss the tolerance even though the
            # system is consistent.  Mirror the tier-2 -> tier-3 pattern and
            # fall through to the preconditioned Krylov tier.
            print(
                "[sfincs_jax.solve] tier-1 structured solve missed the "
                f"tolerance (residuals={np.asarray(result.residual_norms)}); "
                "falling back to the tier-2 Krylov solve."
            )
            chosen = "gmres"
    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        pass  # tier-1 result stands.
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
            check_adjoint=check_adjoint,
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
