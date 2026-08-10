"""The tier-2 coarse preconditioner, and the generated block rows it is built from.

:func:`build_coarse_preconditioner` inverts the SFINCS-simplified coarse
operator exactly and hands the result to :mod:`dkx.solve` as the tier-2
right-preconditioner.  It lives here rather than in ``dkx/solve.py`` because it
is a self-contained owner with two drop-in siblings --- :mod:`dkx.sparse_precond`
and :mod:`dkx.multigrid` --- that both reproduce its pins and both import the
pieces from this module.

The simplified operator is block-tridiagonal over Legendre modes and uncoupled
over ``(species, x)``, so one block-Thomas factorization per subsystem inverts
it.  What differs between the three routes below is only where the blocks are
kept.

**Dense bands.**  Materialize the three dense ``(Ntheta*Nzeta)`` bands and
factor them once.  Fastest to apply and the default wherever the bands fit.

**Reusable factors, Schur LU only.**  ``solvax.direct.block_thomas_factor_fn``
with ``store_offdiagonals=False`` eliminates the same pinned chain from the
generated rows of :func:`_coarse_subsystem_block_fn` and keeps only the Schur
LU, regenerating the off-diagonal blocks during each substitution sweep.  A
third of the band state, a sixth with a float32 LU.  The elimination still runs
exactly once, so the factors are *reusable*: a Krylov application performs
triangular solves and two block regenerations per row, and no factorization.
Exact, reusable and iteration-for-iteration identical to the dense route where
both fit.  Its storage claim holds only because the generators are rebuilt from
traced leaves inside the application: closed over, their ``(Ntheta*Nzeta)``
blocks become captured constants and the unstored bands reappear
(docs/performance.rst).

**Checkpointed, one-shot.**  ``solvax.direct.block_thomas_checkpointed_fn``
retains no band-sized state at all, at the cost of re-eliminating on every
application.  It is the route of last resort, for chains where even the Schur LU
does not fit.

The routing between them is by measured size, not preference
(:func:`_coarse_bands_fit`, :func:`_coarse_factors_fit`): the band and factor
sizes are exact allocations known from the grid before any work happens, which
is what lets an oversized deck be diverted rather than killed part way through.

The chain the three routes share is **singular without its pins**.  Its ``l = 0``
block annihilates a distribution constant on the flux surface, and the
``Nxi_for_x`` truncation leaves whole ``(x, l)`` rows identically zero.
:func:`_coarse_subsystem_block_fn` folds in all three regularizations --- the
``1e-8`` diagonal floor, identity rows on the truncated pairs, and the rank-one
``l = 0`` pin --- and every route must use it, because the floor alone still
returns ``nan``.

Fortran correspondence: ``preconditioner.F90`` (the ``preconditioner_*`` knobs)
and the PETSc ``Pmat`` idiom of production SFINCS.
"""

from __future__ import annotations

import functools
import math
import os
import subprocess
import warnings
from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp

# solvax is a core dependency (installed automatically with dkx), but keep this
# module importable without it and raise a clear error on first use so broken or
# partial environments fail with an actionable message.  This is the package's
# one solvax import guard; ``dkx.solve`` imports :func:`_require_solvax` from
# here rather than keeping a second copy of the same message.
try:
    from solvax.direct import (
        block_thomas_checkpointed_fn,
        block_thomas_factor,
        block_thomas_factor_fn,
        block_thomas_solve,
    )
    from solvax.operators import schur_projected_precond

    _SOLVAX_IMPORT_ERROR: BaseException | None = None
except ImportError as _solvax_exc:  # pragma: no cover - exercised by env, not tests
    block_thomas_checkpointed_fn = None  # type: ignore[assignment]
    block_thomas_factor = None  # type: ignore[assignment]
    block_thomas_factor_fn = None  # type: ignore[assignment]
    block_thomas_solve = None  # type: ignore[assignment]
    schur_projected_precond = None  # type: ignore[assignment]
    _SOLVAX_IMPORT_ERROR = _solvax_exc

from dkx.drift_kinetic import KineticOperator


def _require_solvax() -> None:
    """Raise a clear error when the ``solvax`` core dependency is missing."""
    if _SOLVAX_IMPORT_ERROR is not None:
        raise ImportError(
            "dkx requires the 'solvax' package for its solver "
            "tiers. solvax is a core dependency: `pip install dkx` "
            "pulls it in automatically (the `dkx[structured]` extra is "
            "a no-op alias). To install it directly: `pip install solvax` or "
            "`pip install git+https://github.com/uwplasma/SOLVAX`."
        ) from _SOLVAX_IMPORT_ERROR


def _transposed_apply(op: KineticOperator) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """The transposed matvec ``w -> A^T w`` via ``jax.linear_transpose``."""
    primal = jax.ShapeDtypeStruct((op.total_size,), jnp.float64)

    def apply_t(w: jnp.ndarray) -> jnp.ndarray:
        (out,) = jax.linear_transpose(op.apply, primal)(w)
        return out

    return apply_t

# Multiple of physical RAM the ``"coarse"`` tier-2 preconditioner's dense bands may
# claim before :func:`_coarse_bands_fit` sends the solve down the generated route.
# Set at 1.0 -- "the arrays alone do not fit in RAM" -- because that needs no tuning
# and the measured outcomes separate cleanly on either side: on the 2026-08-01
# upstream campaign (24 GB machine) every deck up to 16.9 GB of bands completed and
# every deck from 42.9 GB was killed.  The transient is charged separately, by
# _COARSE_BAND_RESIDENT_OVERHEAD, rather than by shrinking this fraction, so that
# the fraction stays a policy and the overhead stays a measurement.
_TIER2_GUARD_FRACTION = 1.0
_TIER2_GUARD_ENV = "DKX_TIER2_MEMORY_GUARD"
# Precision of the reusable Schur LU factors: "float64" (default) or "float32".
# See :func:`_coarse_factor_dtype` for why the cheaper one is not the default.
_COARSE_FACTOR_DTYPE_ENV = "DKX_COARSE_FACTOR_DTYPE"

def coarse_preconditioner_band_bytes(op: KineticOperator) -> float:
    """Bytes the ``"coarse"`` tier-2 preconditioner allocates for its bands.

    Three dense ``(Ntheta*Nzeta)`` blocks per ``(species, x, L)``.  This is an
    exact allocation size, not an estimate: the arrays are materialized before
    any factorization runs.
    """
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    return 3.0 * n_s * n_x * n_xi * (n_t * n_z) ** 2 * 8.0

def _host_memory_bytes() -> float | None:
    """Physical RAM, or ``None`` when it cannot be read on this platform."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        return float(pages) * float(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None

def _available_memory_bytes() -> float | None:
    """Memory the OS can hand out *now*, or ``None`` where it cannot be read.

    Physical RAM is the wrong number to size a preconditioner against, and the
    difference is not academic: ``filteredW7XNetCDF_2species_magneticDrifts_noEr``
    wants 14.3 GB of float64 factors, which passes ``14.3 <= 24.0`` on a 24 GB
    machine and then thrashes, because ~10 GB of that machine was already spoken
    for.  Measured, it ran **six hours** with its resident set oscillating between
    3.2 and 8.0 GB and 8.4 of 9.2 GB of swap in use, and produced nothing --- a
    worse outcome than the OOM kill this guard was written to prevent, because a
    kill at least ends.
    """
    try:  # Linux states it directly, and it is the honest number.
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:  # macOS: free + inactive + speculative are all reclaimable
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        page = float(os.sysconf("SC_PAGE_SIZE"))
        wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
        pages = sum(
            float(line.split()[-1].rstrip("."))
            for line in out.splitlines()
            if line.startswith(wanted)
        )
        return pages * page if pages else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None

#: Resident bytes the reusable route actually costs per byte of stored factors.
#: Measured on the same deck at float32: 7.2 GB of factors peaked at 8.87 GB RSS.
_COARSE_RESIDENT_OVERHEAD = 1.25
#: The same ratio for the *dense band* route, which is not the same number and
#: must not borrow the one above.  Measured on a 62 GB box: three runs of the
#: 42.9 GB-band decks each peaked at 58.5-58.9 GB, i.e. **1.37x**, and were
#: OOM-killed 43 s in because ``42.9 * 1.25 = 53.6`` had passed a ~54 GB budget.
#: An earlier 24 GB-box note put bands at 0.69x (16.9 GB peaking at 11.7 GB);
#: that is not reproducible where there is room to materialize, so the guard
#: takes the conservative measurement.  Bands are three arrays where the factors
#: are one, so a larger transient than the factor route is what to expect.
_COARSE_BAND_RESIDENT_OVERHEAD = 1.45

def _coarse_memory_budget() -> float | None:
    """What the preconditioner may claim: available memory, else physical RAM."""
    available = _available_memory_bytes()
    return available if available is not None else _host_memory_bytes()

def coarse_preconditioner_factor_bytes(op: KineticOperator, dtype=jnp.float64) -> float:
    """Bytes the reusable Schur-LU-only factors retain, over every subsystem at once.

    One dense ``(Ntheta*Nzeta)`` LU per ``(species, x, L)`` plus its pivots --- the
    two off-diagonal bands are not stored, because
    :func:`_coarse_subsystem_block_fn` regenerates them.  That is
    :func:`coarse_preconditioner_band_bytes` divided by three at float64, or by six
    at float32, plus the pivots both policies keep.

    Exact rather than estimated, for the same reason the band size is: the factors
    are allocated up front and held for the life of the preconditioner, because
    reuse across Krylov applications is the whole point of taking this route.
    """
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    blocks = float(n_s * n_x * n_xi)
    m = float(n_t * n_z)
    itemsize = float(jnp.dtype(dtype).itemsize)
    return blocks * (m * m * itemsize + m * 4.0)  # LU + int32 pivots

def _coarse_generated_peak_bytes(op: KineticOperator) -> float:
    """Bound on the dense-factor bytes the *generated* coarse route holds per subsystem.

    One Schur checkpoint per ``cs = ceil(sqrt(Nxi))`` rows plus one segment's
    recomputed factors: ``Nxi / cs + 3 cs`` blocks against the bands' ``3 Nxi``.
    """
    _n_s, _n_x, n_xi, n_t, n_z = op.f_shape
    cs = math.isqrt(n_xi - 1) + 1 if n_xi > 1 else 1
    return (-(-n_xi // cs) + 3 * cs) * (n_t * n_z) ** 2 * 8.0

def _coarse_bands_fit(op: KineticOperator) -> bool:
    """Whether the coarse preconditioner's dense bands fit this machine.

    Measured on the 2026-08-01 upstream campaign: five decks were killed by the OS
    55-90 s in, at 13-15 GB resident on a 24 GB machine, having produced nothing --
    the least debuggable failure there is, and avoidable because
    :func:`coarse_preconditioner_band_bytes` is exact up front.
    ``DKX_TIER2_MEMORY_GUARD=off`` forces the dense route anyway; so does an
    unreadable memory size, rather than demoting a working deck on a guess.

    The budget is *available* memory, not physical RAM: see
    :func:`_available_memory_bytes` for the six-hour thrash that measuring
    against physical RAM admits.
    """
    if os.environ.get(_TIER2_GUARD_ENV, "").strip().lower() in {"off", "0", "false"}:
        return True
    budget = _coarse_memory_budget()
    return budget is None or (
        coarse_preconditioner_band_bytes(op) * _COARSE_BAND_RESIDENT_OVERHEAD
        <= _TIER2_GUARD_FRACTION * budget
    )

def _coarse_factor_dtype() -> object:
    """Working precision of the reusable Schur LU factors.

    float64 by default.  Halving the factors buys nothing on a deck that already
    fits, and where it decides whether a deck fits at all the caller is better
    served by being told to set it than by having their preconditioner silently
    change precision under them.  A float32 Schur LU is exact enough to
    precondition with --- GCROT still reaches ``1e-10`` --- but it is not free in
    iterations (docs/performance.rst), so it is opt-in via
    ``DKX_COARSE_FACTOR_DTYPE=float32``.
    """
    name = os.environ.get(_COARSE_FACTOR_DTYPE_ENV, "").strip().lower()
    if name in {"", "float64", "f64", "double"}:
        return jnp.float64
    if name in {"float32", "f32", "single"}:
        return jnp.float32
    raise ValueError(
        f"{_COARSE_FACTOR_DTYPE_ENV}={name!r} is not a recognized precision; "
        f"expected 'float64' (default) or 'float32'"
    )

def _coarse_factors_fit(op: KineticOperator) -> bool:
    """Whether the reusable Schur-LU-only factors fit where the bands did not.

    Same guard, same fraction, one third of the arrays --- so a deck that misses
    the dense route by less than 3x lands here rather than on the one-shot
    checkpointed route, and keeps reusable factors.  That distinction is the
    whole cost model: a preconditioner is applied tens of times per solve, and
    only this side of the branch amortizes its elimination over them.

    Against *available* memory and with the measured resident overhead, because
    this is the branch where getting it wrong hurts most: the factors are held
    for the life of the preconditioner, so a set that does not fit does not get
    killed, it thrashes (:func:`_available_memory_bytes`).
    """
    if os.environ.get(_TIER2_GUARD_ENV, "").strip().lower() in {"off", "0", "false"}:
        return True
    budget = _coarse_memory_budget()
    return budget is None or (
        coarse_preconditioner_factor_bytes(op, _coarse_factor_dtype())
        * _COARSE_RESIDENT_OVERHEAD
        <= _TIER2_GUARD_FRACTION * budget
    )

def _coarse_route_preamble(op: KineticOperator) -> str:
    """What did not fit, and on what machine -- shared by both fallback messages."""
    total = _host_memory_bytes()
    return (
        f"the coarse tier-2 preconditioner's dense (Ntheta*Nzeta) bands would take "
        f"{coarse_preconditioner_band_bytes(op) / 2**30:.1f} GB ({op.n_theta}x{op.n_zeta} "
        f"angular grid, Nxi={op.n_xi}, {op.n_species} species, Nx={op.n_x}) on a machine "
        f"with {'unknown' if total is None else format(total / 2**30, '.1f')} GB of RAM, "
    )

def _coarse_other_routes_note() -> str:
    """The two tier-2 preconditioners that are still not a way out at this size."""
    return (
        "\nThe other tier-2 preconditioners are still not a way out at this size: "
        "'sparse' stores far less but was measured killed or timed out on all five "
        "decks this size (tools/benchmarks/tier2_sparse_vs_coarse.py), and 'multigrid' "
        "fits but does not reach tolerance on this physics (docs/performance.rst)."
    )

def _coarse_reusable_fallback_message(op: KineticOperator) -> str:
    """Say what the reusable-factor route costs before it starts costing it.

    This is the route that made the oversized decks runnable, so the message says
    what it retains and what it does *not* claim: the applications are more
    expensive than the dense route's, just not by the order of magnitude the
    one-shot route costs.
    """
    dtype = _coarse_factor_dtype()
    return (
        f"{_coarse_route_preamble(op)}"
        f"so the solve keeps only the Schur LU factors (solvax block_thomas_factor_fn "
        f"with store_offdiagonals=False, "
        f"{coarse_preconditioner_factor_bytes(op, dtype) / 2**30:.1f} GB at "
        f"{jnp.dtype(dtype).name}) and regenerates the off-diagonal blocks during each "
        f"substitution sweep.\nThe elimination still runs once and the factors are "
        f"reused across Krylov applications, so this is the dense route's cost model "
        f"with a third of its storage, not the one-shot fallback: an application pays "
        f"two block regenerations per row and no factorization."
        f"{_coarse_other_routes_note()}\nTo get the dense route back, reduce "
        f"Ntheta/Nzeta or Nxi, or run where the bands fit. DKX_TIER2_MEMORY_GUARD=off "
        f"allocates them here anyway; DKX_COARSE_FACTOR_DTYPE=float32 halves the "
        f"factors again, and on the deck this was measured on it was better on every "
        f"axis --- 14% fewer iterations, 12% less wall time (docs/performance.rst)."
    )

def _coarse_downgrade_hint(op: KineticOperator, dtype: object) -> str:
    """Lead with float32 when that is the difference between reusable and one-shot.

    Falling from reusable factors to the checkpointed route costs an order of
    magnitude per application, and here that fall is avoidable by halving the
    factors rather than by changing machine.  Saying so first matters because the
    alternative --- the message below --- describes a route the caller does not
    have to take.
    """
    if dtype is jnp.float32:
        return ""
    budget = _coarse_memory_budget()
    if budget is None:
        return ""
    small = coarse_preconditioner_factor_bytes(op, jnp.float32) * _COARSE_RESIDENT_OVERHEAD
    if small > _TIER2_GUARD_FRACTION * budget:
        return ""
    return (
        f"DKX_COARSE_FACTOR_DTYPE=float32 would keep the reusable-factor route on "
        f"this machine ({coarse_preconditioner_factor_bytes(op, jnp.float32) / 2**30:.1f} GB "
        f"of factors, ~{small / 2**30:.1f} GB resident, against "
        f"{_coarse_memory_budget() / 2**30:.1f} GB available), and it is very likely "
        f"what you want: on filteredW7XNetCDF_2species_magneticDrifts_noEr it "
        f"completed in 2 h 50 min at 1041 iterations to a residual of 1.6e-14, where "
        f"float64 factors did not fit and thrashed for six hours without finishing. "
        f"The route described next is the one-shot fallback, which is slower still.\n"
    )

def _coarse_generated_fallback_message(op: KineticOperator) -> str:
    """Say what the one-shot fallback costs before it starts costing it.

    Reached only when even the Schur LU alone does not fit, which is the one thing
    the reusable-factor route cannot do.  The claim to make is "completes", never
    "fast".  It still recommends neither ``preconditioner="sparse"``, killed on
    three of five decks and timed out on two, nor ``multigrid``, which fits without
    reaching tolerance (docs/performance.rst).
    """
    dtype = _coarse_factor_dtype()
    return (
        f"{_coarse_downgrade_hint(op, dtype)}"
        f"{_coarse_route_preamble(op)}"
        f"and even their Schur LU factors alone would take "
        f"{coarse_preconditioner_factor_bytes(op, dtype) / 2**30:.1f} GB, so the solve "
        f"falls back to generating each block row on demand (solvax "
        f"block_thomas_checkpointed_fn, at most "
        f"{_coarse_generated_peak_bytes(op) / 2**30:.2f} GB of dense factors per "
        f"(species, x) subsystem, materializing no band).\nExpect an order of magnitude "
        f"more time per Krylov application than the dense route (measured 1.46 s against "
        f"47 ms on geometryScheme4_2species_withEr_fullTrajectories): it returns a "
        f"solution rather than reusable factors, so each application repeats the "
        f"elimination. It completes; it is not fast."
        f"{_coarse_other_routes_note()}\nTo get a route that keeps its elimination, "
        f"set DKX_COARSE_FACTOR_DTYPE=float32 (halves the Schur LU), reduce "
        f"Ntheta/Nzeta or Nxi, or run where the bands fit. "
        f"DKX_TIER2_MEMORY_GUARD=off allocates them here anyway."
    )

#: Legendre rows that keep the magnetic drifts in the coarse operator, matching
#: Fortran's ``preconditioner_magnetic_drifts_max_L`` default
#: (``globalVariables.F90:212``; the loop bound at ``populateMatrix.F90`` 544/671).
#: Only the L-diagonal half is carried --- the L+-2 half would make the chain
#: pentadiagonal, which block-Thomas cannot factor.
_COARSE_DRIFT_MAX_L = 2

def _drift_diagonal_block(
    parts: dict[str, jnp.ndarray] | None, x2: jnp.ndarray, s: int | jnp.ndarray,
    ell: jnp.ndarray, n_tz: int,
) -> jnp.ndarray | None:  # fmt: skip
    """The magnetic-drift contribution to one coarse diagonal block ``D_(s,x,L)``.

    ``parts`` is :meth:`KineticOperator.magnetic_drift_diagonal_parts`, whose six
    matrices do not depend on ``L``; the ``L`` dependence is the scalar
    coefficients, so a block costs a few scaled adds rather than a stored array.
    Rows above :data:`_COARSE_DRIFT_MAX_L` get zero, which is what Fortran's
    ``preconditioner_magnetic_drifts_max_L`` does.

    ``ell`` may be traced, so the row cutoff is a ``where`` and not a branch.
    """
    if parts is None:
        return None
    c = KineticOperator.magnetic_drift_diagonal_coefficients(ell)
    keep = jnp.where(ell <= _COARSE_DRIFT_MAX_L, 1.0, 0.0)
    block = (
        c["c1"] * (parts["mt1"][s] + parts["mz1"][s])
        + c["c2"] * (parts["mt2"][s] + parts["mz2"][s])
        + c["c3"] * (parts["mt3"][s] + parts["mz3"][s])
    )
    idx = jnp.arange(n_tz)
    block = block.at[idx, idx].add(c["xi"] * parts["xi"][s])
    return (keep * x2) * block

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

def _truncated_blocks(
    params: tuple[jnp.ndarray, ...],
    k: jnp.ndarray,
    *,
    n_xi: int,
    shift_border: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Analytic ``(L_k, D_k, U_k)`` as a pure function of ``(params, k)``.

    ``params`` carries every differentiable array entering the blocks —
    ``(stream, mirror, pas_row, x_val, gamma, exb, b0, c0, cl, cu)`` — so the
    same function serves both the legacy closure form (via
    :func:`_truncated_block_fn`) and solvax's structure-preserving
    ``params``/``adjoint_window`` custom VJP, which requires the
    differentiable inputs to be explicit arguments rather than closed-over
    tracers.  Only the static ints/bools stay in the closure.
    """
    stream, mirror, pas_row, x_val, gamma, exb, b0, c0, cl, cu = params
    m = exb.shape[0]
    idx = jnp.arange(m)
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

def _truncated_params(
    coef: dict[str, jnp.ndarray],
    stream: jnp.ndarray,
    mirror: jnp.ndarray,
    pas_row: jnp.ndarray,
    x_val: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """The differentiable-parameter pytree consumed by :func:`_truncated_blocks`."""
    return (
        stream, mirror, pas_row, x_val, gamma,
        coef["exb"], coef["b0"], coef["c0"], coef["cl"], coef["cu"],
    )

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
    The block algebra lives in :func:`_truncated_blocks`; this wrapper only
    closes over the parameter pytree for the legacy index-only signature.
    """
    params = _truncated_params(coef, stream, mirror, pas_row, x_val, gamma)

    def block_fn(k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _truncated_blocks(params, k, n_xi=n_xi, shift_border=shift_border)

    return block_fn

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
    from dkx.collisions import apply_fokker_planck_v3_phi1  # noqa: PLC0415

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

# Invertibility floor of the coarse f-block, relative to the operator's own band
# magnitude.  1e-8 is tiny enough to leave a real diagonal -- and the tightly
# clustered preconditioning it gives -- untouched, yet keeps an all-zero-diagonal
# collisionless deck out of an exact-zero pivot.  Mirrored by
# ``dkx.multigrid._DIAGONAL_FLOOR``.
_COARSE_DIAGONAL_FLOOR = 1e-8
# Experiment knob for :func:`_l0_pin_gamma`: ``"never"``, ``"legacy"``, or a
# float overriding the relative level of the l=0 null-space pin.
_L0_PIN_ENV = "DKX_COARSE_L0_PIN"

def _l0_pin_gamma(
    defect: jnp.ndarray, band: jnp.ndarray, scale: jnp.ndarray, c0: jnp.ndarray
) -> jnp.ndarray:
    """``(S, X)`` weight of the rank-one ``ones (x) c0`` pin on the ``l = 0`` block.

    The simplified ``l = 0`` diagonal block ``A_0`` has one *known* null vector: a
    distribution constant on the flux surface, annihilated by streaming, by the
    mirror force, by ExB, and by pitch-angle scattering (``nu l(l+1)/2 = 0`` at
    ``l = 0``).  ``defect = max_i |sum_j (A_0)_ij|`` is exactly that vector's
    residual, so it measures how singular ``A_0`` really is in the one direction
    this pin removes -- and it is what makes the pin adaptive rather than
    unconditional.  With ``sum_j (ones (x) c0)_ij = sum(c0)`` the added row sum is
    ``gamma sum(c0)``, so ``gamma sum(c0) = max(0, floor * band - defect)`` tops
    the constant direction up to the isotropic floor's level where the block
    really is singular there and switches the pin off completely -- ``gamma = 0``
    -- as soon as the block's own ``l = 0`` diagonal exceeds it.  Sizing it by the
    floor rather than by ``scale`` (the mean ``|diagonal|`` over *all* ``L``, which
    the ``nu l(l+1)/2`` collision diagonal makes ~1e3 times larger at high ``l``)
    is what keeps the coarse operator close to the operator it preconditions.

    ``DKX_COARSE_L0_PIN`` overrides the floor for experiments: ``"never"`` disables
    the pin, ``"legacy"`` restores the unconditional full-strength pin, and a float
    sets the relative level directly.
    """
    mode = os.environ.get(_L0_PIN_ENV, "").strip().lower()
    if mode == "legacy":
        return jnp.broadcast_to(scale / jnp.max(jnp.abs(c0)), defect.shape)
    level = _COARSE_DIAGONAL_FLOOR if not mode else (0.0 if mode == "never" else float(mode))
    if level <= 0.0:
        return jnp.zeros_like(defect)
    return jnp.maximum(level * band - defect, 0.0) / jnp.sum(c0)

def _coarse_subsystem_block_fn(
    coef: dict[str, jnp.ndarray], n_xi: int, sub: tuple[jnp.ndarray, ...], *,
    drop_l_coupling: bool, floor: jnp.ndarray | None = None, gamma: jnp.ndarray | None = None,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:  # fmt: skip
    """``(L_j, D_j, U_j)`` of one coarse subsystem from a traced ``j``, pins folded in.

    ``sub`` is ``(stream, mirror, pas_row, x, mask_row, coll_row)``; ``floor`` and
    ``gamma`` add the three regularizations that make the chain invertible -- floor,
    identity rows on the ``(x, l)`` pairs ``Nxi_for_x`` truncates, rank-one pin of the
    ``l = 0`` null vector -- all load-bearing (floor alone leaves it singular and the
    solve returns ``nan``), and ``None`` gives the unregularized blocks that size them.
    ``j`` is traced, which is why :meth:`KineticOperator.legendre_blocks` cannot serve.
    """
    stream, mirror, pas_row, x_val, mask_row, coll_row, d1, d2, d3, dxi = sub
    params = _truncated_params(
        coef, stream, mirror, pas_row, x_val, jnp.asarray(0.0, dtype=jnp.float64)
    )
    idx = jnp.arange(int(stream.shape[0]))
    l0_pin = None if gamma is None else gamma * jnp.outer(coef["b0"], coef["c0"])

    def block_fn(j: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        lower, diag, upper = _truncated_blocks(params, j, n_xi=n_xi, shift_border=False)
        # ``legendre_blocks`` bakes the Nxi_for_x row/column masks into every block
        # and zeroes the off-diagonal blocks at the ends of the chain; do that here.
        m_j = jnp.take(mask_row, j)
        m_lo = jnp.where(j > 0, jnp.take(mask_row, jnp.maximum(j - 1, 0)), 0.0)
        m_up = jnp.where(j < n_xi - 1, jnp.take(mask_row, jnp.minimum(j + 1, n_xi - 1)), 0.0)
        lower, upper, diag = lower * (m_j * m_lo), upper * (m_j * m_up), diag * m_j
        if drop_l_coupling:
            lower, upper = jnp.zeros_like(lower), jnp.zeros_like(upper)
        added = jnp.take(coll_row, j) * m_j
        if floor is not None:
            added = added + floor + (1.0 - m_j)
        diag = diag.at[idx, idx].add(added)
        # Magnetic drifts, L-diagonal half only, up to _COARSE_DRIFT_MAX_L --- the
        # rows Fortran keeps.  d1/d2/d3/dxi already carry base, the geometric
        # factors and x^2; only the scalar L coefficients are left.
        dc = KineticOperator.magnetic_drift_diagonal_coefficients(j)
        keep = jnp.where(j <= _COARSE_DRIFT_MAX_L, 1.0, 0.0) * m_j
        diag = diag + keep * (dc["c1"] * d1 + dc["c2"] * d2 + dc["c3"] * d3)
        diag = diag.at[idx, idx].add(keep * dc["xi"] * dxi)
        if l0_pin is not None:
            diag = diag + jnp.where(j == 0, 1.0, 0.0) * l0_pin
        return lower, diag, upper

    return block_fn

def _strip_for_coarse(op: KineticOperator) -> KineticOperator:
    """The SFINCS-simplified operator the coarse preconditioner is built from.

    Everything that breaks the block-tridiagonal-in-``L`` structure is dropped:
    the dense collision operators (reduced to an x-diagonal separately), the
    ``E_r`` ``xDot``/``xiDot`` terms, the tangential magnetic drifts and the
    ``Phi1`` coupling.  GCROT corrects for all of it.
    """
    return replace(
        op, fp=None, sugama=None, fp_phi1=None, with_er_xidot=False, with_er_xdot=False,
        with_magnetic_drifts=False, external_phi1_hat=None, include_phi1=False,
        include_phi1_in_kinetic=False,
    )  # fmt: skip

def _coarse_pinned_block_fns(
    coef: dict[str, jnp.ndarray], n_xi: int, subs: list[tuple[jnp.ndarray, ...]],
    floor: jnp.ndarray, gamma: jnp.ndarray, drop_l_coupling: bool,
) -> list[Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:  # fmt: skip
    """The pinned row generators, rebuilt from whatever arrays are handed in.

    Split from :func:`_coarse_generated_block_data` so the generators can be
    rebuilt *inside* a jitted application from traced leaves; see
    :func:`build_coarse_preconditioner` for why that matters.
    """
    return [
        _coarse_subsystem_block_fn(
            coef, n_xi, sub, drop_l_coupling=drop_l_coupling,
            floor=floor[b], gamma=gamma[b],
        )  # fmt: skip
        for b, sub in enumerate(subs)
    ]

def _coarse_generated_block_data(
    op: KineticOperator, coef: dict[str, jnp.ndarray], mask: jnp.ndarray,
    coll_diag: jnp.ndarray | None, c0: jnp.ndarray, drop_l_coupling: bool,
) -> tuple[list[tuple[jnp.ndarray, ...]], jnp.ndarray, jnp.ndarray]:  # fmt: skip
    """``(subs, floor, gamma)`` --- every array the pinned row generators close over.

    Returned as data rather than baked into closures so that a caller who needs
    the generators on the far side of a jit boundary can pass these across as
    arguments and rebuild them there with :func:`_coarse_pinned_block_fns`.

    ``band``, the ``l = 0`` defect and the mean diagonal are reductions over ``L``, not
    storage, so they stream one live block at a time off the unregularized generator.
    """
    n_s, n_x, n_xi = op.n_species, op.n_x, op.n_xi
    batch = n_s * n_x
    mask_b = jnp.tile(mask, (n_s, 1))  # (B, L)
    zeros = jnp.zeros((batch, n_xi), dtype=jnp.float64)
    n_tz = op.n_theta * op.n_zeta
    parts = op.magnetic_drift_diagonal_parts()
    if parts is None:
        z_mat = jnp.zeros((batch, n_tz, n_tz), dtype=jnp.float64)
        d1 = d2 = d3 = z_mat
        dxi = jnp.zeros((batch, n_tz), dtype=jnp.float64)
    else:
        # Fold x^2 and the species index in here, so the generator carries three
        # (TZ, TZ) matrices per subsystem instead of the six per species plus a
        # per-row combination --- and so every leaf crosses the jit boundary as
        # data (see build_coarse_preconditioner on captured constants).
        x2 = jnp.tile(op.x * op.x, n_s)[:, None, None]  # (B,1,1)
        rep = lambda m: jnp.repeat(m, n_x, axis=0)  # noqa: E731  (S,..)->(B,..)
        d1 = x2 * rep(parts["mt1"] + parts["mz1"])
        d2 = x2 * rep(parts["mt2"] + parts["mz2"])
        d3 = x2 * rep(parts["mt3"] + parts["mz3"])
        dxi = x2[:, :, 0] * rep(parts["xi"])
    subs = list(zip(
        jnp.repeat(coef["stream"], n_x, axis=0), jnp.repeat(coef["mirror"], n_x, axis=0),
        coef["pas"].reshape(batch, n_xi), jnp.tile(op.x, n_s), mask_b,
        zeros if coll_diag is None else coll_diag.reshape(batch, n_xi),
        d1, d2, d3, dxi,
    ))  # fmt: skip
    rows = functools.partial(_coarse_subsystem_block_fn, coef, n_xi,
                             drop_l_coupling=drop_l_coupling)  # fmt: skip

    def stats(block_fn):
        def step(worst, j):
            lo, di, up = block_fn(j)
            return jnp.maximum(worst, jnp.max(jnp.abs(jnp.stack([lo, di, up])))), jnp.diagonal(di)

        start = jnp.asarray(0.0, dtype=jnp.float64)
        band, diagonals = jax.lax.scan(step, start, jnp.arange(n_xi, dtype=jnp.int32))
        diag0 = block_fn(jnp.asarray(0, dtype=jnp.int32))[1]
        return band, jnp.max(jnp.abs(jnp.sum(diag0, axis=-1))), diagonals

    raw = [stats(rows(sub)) for sub in subs]
    band = jnp.stack([s[0] for s in raw])  # (B,)
    band = jnp.where(band > 0.0, band, 1.0)
    floor = _COARSE_DIAGONAL_FLOOR * band  # (B,)
    diagonals = jnp.stack([s[2] for s in raw])  # (B, L, TZ)
    scale = jnp.mean(
        jnp.abs(diagonals + floor[:, None, None] + (1.0 - mask_b)[:, :, None]), axis=(1, 2)
    )
    gamma = _l0_pin_gamma(
        jnp.stack([s[1] for s in raw]).reshape(n_s, n_x), band.reshape(n_s, n_x),
        jnp.where(scale > 0.0, scale, 1.0).reshape(n_s, n_x), c0,
    ).reshape(-1)  # fmt: skip
    return subs, floor, gamma

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
    L±2 terms are dropped (Fortran's ``preconditioner_xi=1``, unconditional here);
    ``drop_l_coupling`` severs the dominant L±1 coupling too, a separate and far
    stronger cut measured at 6000 iterations to 0.77 against 19 with it kept.
    The result is block-tridiagonal over L and uncoupled over (species, x), so
    one batched block-Thomas factorization inverts it exactly; the bordered
    constraint rows of the *full* operator are then eliminated exactly with
    ``solvax.operators.schur_projected_precond``.

    When ``op.include_phi1`` the operator is the Jacobian of the nonlinear Phi1
    residual and its border is the whole quasineutrality block
    ``[Phi1(theta,zeta) | lambda | sources]`` with a *nonzero* border-border
    block ``D`` (the QN adiabatic Phi1 diagonal, the ``+lambda`` coupling, and
    the ``<Phi1>=0`` row).  That full border is eliminated exactly with the
    generalized bordered Schur complement --
    the coarse f-block solve plus a dense ``~Ntheta*Nzeta`` Schur solve -- so
    the coarse preconditioner is Phi1-aware and the Newton inner Krylov solve
    converges in far fewer iterations (:func:`dkx.phi1.solve_phi1`).

    Three routes to the same inverse, chosen by what fits (module docstring).  The
    default materializes the three dense ``(Ntheta*Nzeta)`` bands and factors them
    once, which is what makes the preconditioner cheap to apply.  Where they exceed
    RAM (:func:`_coarse_bands_fit`; 42.9-53.3 GB on five upstream decks) the same
    pinned operator is eliminated from the generated rows of
    :func:`_coarse_subsystem_block_fn` instead --- keeping only the Schur LU
    (:func:`_coarse_factors_fit`, a third of the bands, still reusable), or, where
    even that does not fit, re-eliminating on every application through
    ``solvax.direct.block_thomas_checkpointed_fn`` (docs/performance.rst, "Running
    the decks the bands do not fit").

    Every route uses the *same* pinned generator or its dense equivalent, because
    the coarse chain is singular without all three pins and a route that reproduced
    only two would return ``nan`` rather than a worse preconditioner.

    Returns:
        ``(precond, precond_t)`` — approximate inverses of ``K`` and ``K^T`` on flat
        ``(total_size,)`` vectors.  The dense and Schur-LU routes share one
        factorization between the forward and transposed applications; the one-shot
        route shares the pins and the row generator instead.
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    stripped = _strip_for_coarse(op)
    # Keep the Nxi_for_x truncation mask a jnp array (no host materialization) so the
    # coarse preconditioner stays traceable when the operator leaves are tracers
    # (jit-over-leaves / vmap / the differentiable kernel): the shape is static, only
    # the boolean pattern depends on the traced ``n_xi_for_x``.
    mask = op._mask()  # (X, L)
    # Fokker-Planck (collisionOperator=1) or improved Sugama (=3), reduced to its
    # PAS-like self-species x-diagonal -- for Sugama that drops the field-particle
    # coupling, kept only in the full operator GCROT solves.  op.fp_phi1 has no dense
    # mat to slice, so its diagonal is probed instead; without it a Phi1-in-collision
    # coarse f-block is singular.  None rather than zeros: on a 17 GB band an
    # all-zero add is a full-size pass for nothing.
    coll_diag = None  # (S, X, L)
    for coll in (op.fp, op.sugama):
        if coll is not None:
            coll_diag = _dense_collision_diagonal(coll.mat) * mask[None, :, :]
    if op.fp_phi1 is not None:
        coll_diag = _collision_phi1_diagonal(op) * mask[None, :, :]
    # The magnetic drifts SFINCS keeps in its preconditioner and DKX used to drop.
    # Only the L-diagonal half fits a block-tridiagonal chain, and it is enough:
    # on a reduced-resolution W7-X deck with the production physics (194,404
    # unknowns) this is 5838 -> 2163 GCROT iterations and 900 s -> 337 s
    # (tests/test_magnetic_drift_diagonal.py).
    drift_parts = op.magnetic_drift_diagonal_parts()
    c0 = op._fs_average_factor().reshape(-1)
    ones = jnp.ones((n_tz,), dtype=jnp.float64)

    if _coarse_bands_fit(op):
        blocks = stripped.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
        lower, diag, upper = (jnp.transpose(a, (1, 2, 0, 3, 4)) for a in blocks)  # (S,X,L,TZ,TZ)
        eye = jnp.eye(n_tz, dtype=jnp.float64)
        if coll_diag is not None:
            diag = diag + coll_diag[:, :, :, None, None] * eye[None, None, None, :, :]
        if drift_parts is not None:
            x2 = op.x * op.x
            drift = jnp.stack([
                jnp.stack([
                    jnp.stack([
                        _drift_diagonal_block(drift_parts, x2[ix], sp,
                                              jnp.asarray(ell, jnp.float64), n_tz)
                        for ell in range(n_xi)
                    ])
                    for ix in range(n_x)
                ])
                for sp in range(n_s)
            ])  # (S, X, L, TZ, TZ)
            diag = diag + drift
        if drop_l_coupling:
            lower, upper = jnp.zeros_like(lower), jnp.zeros_like(upper)

        # Invertibility floor.  A collisionless, drift-free coarse f-block (``nu_n=0``
        # with ``Er=0``) has EXACTLY zero diagonal blocks -- only streaming and mirror
        # couple L -- so block-Thomas would divide by zero, and the Phi1 Newton inner
        # solve forces this preconditioner for every deck.  A per-(species, x) floor
        # scaled by the band magnitude is negligible against a real collision/ExB
        # diagonal and degrades toward a well-scaled identity where it is not.
        band = jnp.maximum(
            jnp.max(jnp.abs(diag), axis=(2, 3, 4)),
            jnp.maximum(
                jnp.max(jnp.abs(lower), axis=(2, 3, 4)), jnp.max(jnp.abs(upper), axis=(2, 3, 4))
            ),
        )  # (S, X)
        band = jnp.where(band > 0.0, band, 1.0)
        # How singular the ``l = 0`` block is in the direction the rank-one pin below
        # removes, measured *before* the floor masks it: the constant-on-surface vector
        # is that block's exact null vector, so its row sums are the whole defect, and
        # flooring first makes every block look regular here (:func:`_l0_pin_gamma`).
        l0_defect = jnp.max(jnp.abs(jnp.sum(diag[:, :, 0], axis=-1)), axis=-1)  # (S, X)
        floor = _COARSE_DIAGONAL_FLOOR * band
        diag = diag + floor[:, :, None, None, None] * eye[None, None, None, :, :]

        # Masked (x, l) rows are identically zero in the operator: pin them with the
        # identity so the coarse factorization stays nonsingular.
        diag = diag + (1.0 - mask)[None, :, :, None, None] * eye[None, None, None, :, :]

        # Adaptive rank-one pin of the l=0 block's constant-on-surface null space,
        # per (species, x): tops that direction up to the floor's relative level where
        # the block really is singular there, off entirely where its own l=0 diagonal
        # exceeds it.  Pinning unconditionally instead cost GCROT 87 iterations against
        # 21 on the NCSX 11x21x41x5 ladder (:func:`_l0_pin_gamma`, docs/performance).
        d4 = diag.reshape(batch, n_xi, n_tz, n_tz)
        scale = jnp.mean(jnp.abs(jnp.diagonal(d4, axis1=2, axis2=3)), axis=(1, 2))
        scale = jnp.where(scale > 0.0, scale, 1.0)
        gamma = _l0_pin_gamma(l0_defect, band, scale.reshape(n_s, n_x), c0).reshape(-1)
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
    else:
        # The bands do not fit in RAM.  Both remaining routes eliminate the same
        # pinned chain from generated rows and materialize no band; they differ in
        # what survives the elimination, which is what decides whether the result
        # can be applied tens of times per solve or has to be rebuilt each time.
        stripped._check_block_extraction_supported()
        coef = _truncated_coefficients(stripped)
        gen_data = _coarse_generated_block_data(
            op, coef, mask, coll_diag, c0, drop_l_coupling
        )
        pinned = _coarse_pinned_block_fns(coef, n_xi, *gen_data, drop_l_coupling)
        reusable = _coarse_factors_fit(op)
        warnings.warn(
            _coarse_reusable_fallback_message(op) if reusable
            else _coarse_generated_fallback_message(op),
            RuntimeWarning, stacklevel=2,
        )  # fmt: skip

        if reusable:
            # Keep the Schur LU and nothing else: one (Nxi, TZ, TZ) array per
            # subsystem instead of three, with the off-diagonal blocks regenerated
            # from the same pinned generator inside each substitution sweep.  The
            # elimination runs here, once, and every later application is two
            # triangular solves and two block regenerations per row -- so unlike the
            # checkpointed route below these factors amortize over a Krylov solve.
            factor_dtype = _coarse_factor_dtype()
            factors = [
                block_thomas_factor_fn(
                    f, n_xi, factor_dtype=factor_dtype, store_offdiagonals=False
                )
                for f in pinned
            ]

            def _a_inv(transpose: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
                # Both the factors and the arrays the generators rebuild blocks
                # from are ARGUMENTS, never closed over: a concrete leaf reached
                # from inside a lowering is a compile-time constant of it, and
                # XLA holds a second copy.  The factors alone are not enough ---
                # the generator regenerating the unstored off-diagonals closes
                # over the (TZ, TZ) stream and exb blocks, and
                # GeneratedBlockTridiagFactors carries it as a *static* field, so
                # the bands this route avoided storing came back as constants:
                # "15.52GB total", OOM-killed at 640 s, on
                # filteredW7XNetCDF_2species_magneticDrifts_noEr.  Rebuilding the
                # generators here, from traced leaves, is what makes the storage
                # claim true (tests/test_coarse_precond_constants.py).
                @jax.jit
                def apply(facs: list, gen: tuple, v: jnp.ndarray) -> jnp.ndarray:
                    coef_t, subs_t, floor_t, gamma_t = gen
                    rows = _coarse_pinned_block_fns(
                        coef_t, n_xi, subs_t, floor_t, gamma_t, drop_l_coupling
                    )
                    # Serial over the 5-10 subsystems for the same reason as below.
                    g = v.reshape(batch, n_xi, n_tz)
                    return jnp.stack([
                        block_thomas_solve(replace(fac, block_fn=row), g[b], transpose=transpose)
                        for b, (fac, row) in enumerate(zip(facs, rows, strict=True))
                    ]).reshape(v.shape)

                return functools.partial(apply, factors, (coef, *gen_data))

        else:

            def _a_inv(transpose: bool) -> Callable[[jnp.ndarray], jnp.ndarray]:
                def apply(v: jnp.ndarray) -> jnp.ndarray:
                    # Serial over the 5-10 subsystems on purpose: vmap would broadcast
                    # the generator, which computes all (S, X) blocks internally, to
                    # (S, X, TZ, TZ) per lane -- exactly the storage this route avoids.
                    g = v.reshape(batch, n_xi, n_tz)
                    return jnp.stack([
                        block_thomas_checkpointed_fn(f, n_xi, g[b], transpose=transpose)
                        for b, f in enumerate(pinned)
                    ]).reshape(v.shape)

                # Compiled once and reused: an application traces n_xi generated block
                # rows per subsystem plus the operator ``custom_linear_solve`` holds for
                # its implicit VJP, and re-tracing that eagerly on every Krylov
                # application costs more than the elimination it wraps.
                return jax.jit(apply)

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
        precond = schur_projected_precond(a_inv, b_cols, c_rows, d_block=d_block)
        precond_t = schur_projected_precond(a_inv_t, c_rows.T, b_cols.T, d_block=d_block.T)
        return precond, precond_t
    if op.extra_size == 0:
        return a_inv, a_inv_t
    b_cols, c_rows = _materialize_borders(op)
    precond = schur_projected_precond(a_inv, b_cols, c_rows)
    precond_t = schur_projected_precond(a_inv_t, c_rows.T, b_cols.T)
    return precond, precond_t
