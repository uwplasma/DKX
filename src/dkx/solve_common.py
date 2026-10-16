"""Shared pieces of :func:`dkx.solve.solve`: results, residual guards, adjoint checks.

Every route (structured direct, recycled Krylov, sparse direct) returns a
:class:`SolveResult` and checks its original equations through the helpers here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
import itertools
import os
import threading

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:  # noqa: E402
    from solvax.implicit import linear_solve as solvax_linear_solve
except ImportError:
    solvax_linear_solve = None  # type: ignore[assignment, misc]

from dkx.coarse_precond import (  # noqa: E402
    _available_memory_bytes,
    _transposed_apply,
)
from dkx.drift_kinetic import (  # noqa: E402
    KineticOperator,
)
from dkx.structured_direct import (  # noqa: E402
    CoupledSolver,
    _is_traced,
)

# Recycled-Krylov restart policy, used when ``solve(restart=None)`` (the
# default).  The first cycles run at a short restart, where an easy deck
# converges and every Arnoldi step is cheap; a solve still unconverged after
# them continues from its iterate at the longest restart whose FGMRES basis fits
# the memory budget, capped at ``_AUTO_RESTART_CAP``, after two cycles of 100.
# A longer restart is not free: its basis is allocated whole, and every step
# orthogonalizes against all of it.  The cap is the measured
# one: on the HSX-like ``Nx`` ladder, restart 200 needs 2,788 iterations at
# ``Nx = 16`` where 1,000 needs 357 and 2,000 gives the same 357 at a higher
# cost per step (2026-09-23 restart-and-direct-reach study; SOLVAX then
# orthogonalized against the whole padded basis, since 0.28 only its filled
# rows).  The wide basis costs ``restart * total_size * itemsize`` bytes: DKX's
# own preconditioners are fixed linear maps, so the wide window runs GCROT with
# ``fixed_precond=True`` and stores only the Arnoldi basis ``V``, not ``Z``.
_AUTO_RESTART_PROBE = 30

_AUTO_RESTART_PROBE_CYCLES = 5

# Between the two, the pre-2.7 widened cycle: a deck that the earlier policy
# (five cycles of 30, then cycles of 100) converged within two cycles of 100
# takes exactly the same path, without paying the wide basis.
_AUTO_RESTART_MIDDLE = 100

_AUTO_RESTART_MIDDLE_CYCLES = 2

_AUTO_RESTART_CAP = 1000

# Share of the memory available when the solve widens that its basis may take,
# unless an explicit budget (argument or environment variable) is given.
_KRYLOV_BASIS_MEMORY_FRACTION = 0.25

# Used only where available memory cannot be read: the pre-2.7 256 MiB guard.
_KRYLOV_BASIS_FALLBACK_BYTES = 256 * 1024**2

_KRYLOV_BUDGET_ENV = "DKX_KRYLOV_MEMORY_BUDGET_GB"

# Defect-correction sweeps after the structured direct factor solve.  One
# reproduces the hand-rolled pass this replaced and already reaches O(1e-16)
# relative residual in float64; the knob exists because the float32-factor variant needs more.
_TIER1_REFINEMENT_SWEEPS = 1

# The speed-coupled elimination meets the electron-ion mass ratio in its
# pivots, and one substitution lands near 1e-6 on two-species decks; each sweep
# gains about two digits.
_COUPLED_REFINEMENT_SWEEPS = 3

def _refinement_sweeps(t1_solver: Any) -> int:
    return _COUPLED_REFINEMENT_SWEEPS if isinstance(t1_solver, CoupledSolver) else _TIER1_REFINEMENT_SWEEPS

# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class SolveResidualRecord:
    """One differentiable-solve residual measured on the host at execution time.

    Attributes:
        label: ``"forward"`` or ``"adjoint (transposed)"``.
        rhs_index: column of the right-hand side this solve belongs to.
        residual_norm: the *true* residual norm — ``||A x - b||`` for the
            forward solve, ``||A^T y - g||`` for the adjoint one, recomputed
            from the operator itself rather than read off the Krylov method's
            internal (recursively updated) estimate.
        rhs_norm: ``||b||`` (adjoint: ``||g||``, the cotangent).
        solution_norm: ``||x||`` (adjoint: ``||y||``).
        operator_norm: probe estimate of ``||A||`` (adjoint: ``||A^T||``),
            used only for a diagnostic backward-error scale.
        target: the residual the caller asked for, ``max(atol, tol*||b||)``.
        floor: capped diagnostic scale ``slack * eps * (||A|| ||x|| + ||b||)``.
            This estimate is neither a lower bound nor an acceptance threshold.
        limit: the accepted residual, ``factor * target``.
        within_tolerance: residual gate and finite norms, with entrywise zero
            defect required for homogeneous or zero-target equations.
    """

    label: str
    rhs_index: int
    residual_norm: float
    rhs_norm: float
    solution_norm: float
    operator_norm: float
    target: float
    floor: float
    limit: float
    within_tolerance: bool

    @property
    def relative_residual(self) -> float:
        """``residual_norm / ||b||`` (``inf`` for a zero right-hand side)."""
        if self.rhs_norm > 0.0:
            return self.residual_norm / self.rhs_norm
        return float("inf") if self.residual_norm > 0.0 else 0.0

@dataclass
class AdjointDiagnostics:
    """True residuals of the implicit-differentiation solves of one :func:`solve`.

    Callbacks update records when each solve executes, including during the
    backward pass. Synchronize device work/effects before inspecting them.
    Cached JIT executions reuse this container; only the latest record for
    each equation label and RHS column is retained, so long scans do not
    accumulate an unbounded history. Copy ``records`` after synchronization
    to retain a snapshot, or call :meth:`reset` before a new measurement.

    Attributes:
        tol, atol: the tolerances the solve was asked for.
        factor: the explicitly accepted slack over ``target``
            (``adjoint_residual_factor`` of :func:`solve`).
        checked: whether a failure raises (``check_adjoint`` of :func:`solve`).
        records: latest :class:`SolveResidualRecord` per equation/RHS key,
            in first-seen key order.
    """

    tol: float
    atol: float
    factor: float
    checked: bool
    records: list[SolveResidualRecord] = field(default_factory=list)

    def reset(self) -> None:
        """Drop every record (e.g. between two runs of a jitted function)."""
        self.records.clear()

    def _record(self, record: SolveResidualRecord) -> None:
        for i, old in enumerate(self.records):
            if (old.label, old.rhs_index) == (record.label, record.rhs_index):
                self.records[i] = record
                return
        self.records.append(record)

    def _select(self, label_prefix: str) -> list[SolveResidualRecord]:
        return [r for r in self.records if r.label.startswith(label_prefix)]

    @property
    def adjoint_records(self) -> list[SolveResidualRecord]:
        """Only the transposed (adjoint) solves."""
        return self._select("adjoint")

    @property
    def forward_records(self) -> list[SolveResidualRecord]:
        """Only the forward solves run under the implicit-diff wrapper."""
        return self._select("forward")

    @property
    def worst_relative_residual(self) -> float:
        """Largest ``||A^T y - g|| / ||g||`` over the adjoint solves.

        ``0.0`` when the backward pass has not run yet (no records).
        """
        rel = [r.relative_residual for r in self.adjoint_records]
        return max(rel) if rel else 0.0

    @property
    def converged(self) -> bool:
        """Whether every recorded solve landed inside its accepted residual."""
        return all(r.within_tolerance for r in self.records)

@dataclass(frozen=True)
class SolveResult:
    """Outcome of :func:`solve`.

    Attributes:
        x: solution state vector(s), same shape as the ``rhs`` passed in
            (``(n,)`` or ``(n, n_rhs)``).
        method: the implementation that ran --- ``"block_tridiagonal"``,
            ``"gcrot"``, or ``"direct"``.  For "what did it do", read
            :attr:`route` instead, which is ``"direct"`` or ``"iterative"``.
        iterations: total Krylov inner iterations across all right-hand sides
            (recycled Krylov), else ``None``.
        residual_norms: residual norms per right-hand side, shape ``(n_rhs,)``
            (jnp array; traced under ``jax.grad``). Complete states report the
            original ``||b - A x||``; moment-only truncated states report only
            the rows determined by the retained head. Certify full equations
            using complete recovery or an independent original residual.
        converged: whether residuals meet their per-column targets
            ``max(atol, tol * ||b||)``; a finite direct result alone does not
            imply convergence.
        recycle: GCROT recycle pair ``(C, U)`` from the last right-hand side
            (recycled Krylov), for warm-starting the next solve of a
            continuation.
        timings: wall-clock seconds per phase (``build``, ``solve``), or
            ``build_and_solve`` when generated implicit recovery cannot separate
            them. Each
            phase ends with a ``jax.block_until_ready`` so the numbers are real
            device-compute time, not JAX async-dispatch latency (which would
            under-report by ~10x).  Under ``jit``/``grad`` the blocks are no-ops
            and the values are trace-time only.
        adjoint: :class:`AdjointDiagnostics` of the implicit-differentiation
            solves (differentiable full-state structured and recycled Krylov, else ``None``).
            Empty when the result is returned: the records are appended at
            *execution* time, i.e. the forward ones when the value is computed
            and the adjoint ones when the backward pass runs, so inspect it
            after the enclosing ``jax.grad`` / ``jax.vjp`` call has
            completed.
        precond: the preconditioner pair this solve built, or ``None`` when the
            route did not need one. Pass it back as ``solve(..., precond=...)``
            to skip the rebuild on a nearby operator -- an ``Er`` scan, a Newton
            step, the next surface. Returning it rather than having the caller
            build one ahead of time is what makes reuse correct: the route is
            not known until the solve runs, so a caller who guesses builds a
            preconditioner the structured-direct route then never uses.
        factors: the factorization this solve built, on the two direct routes
            --- a :class:`Tier1Solver` (structured) or :class:`DirectFactors`
            (sparse), else ``None``.  Pass it back as ``solve(..., factors=...)``
            and the next solve of the same operator is a triangular solve:
            another right-hand side, or the adjoint with ``transpose=True``.
            On a *neighbouring* operator the stored factors are an approximate
            inverse, and the route says so by checking the original residual
            and refactorizing once when it misses (see :func:`solve`).
    """

    x: jnp.ndarray
    method: str
    iterations: int | None
    residual_norms: jnp.ndarray
    converged: bool
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None
    timings: dict[str, float]
    adjoint: AdjointDiagnostics | None = None
    precond: tuple[Callable, Callable] | None = None
    factors: Any = None

    @property
    def route(self) -> str:
        """``"direct"`` or ``"iterative"`` --- what the solve did, not how.

        ``method`` names the implementation that ran, which is what the solver
        trace and the benchmarks need.  A reader of a run wants the shorter
        answer, and "block_tridiagonal_truncated" is not it.
        """
        return "iterative" if self.method in {"gcrot", "gmres"} else "direct"

def _as_columns(rhs: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs.ndim == 1:
        return rhs[:, None], True
    if rhs.ndim == 2:
        return rhs, False
    raise ValueError(f"rhs must be (n,) or (n, n_rhs); got shape {rhs.shape}")

def _residual_norms(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], x2d: jnp.ndarray, rhs2d: jnp.ndarray
) -> jnp.ndarray:
    res = jax.vmap(matvec, in_axes=1, out_axes=1)(x2d) - rhs2d
    return jnp.linalg.norm(res, axis=0)

def _converged_flag(
    res_norms: jnp.ndarray, rhs2d: jnp.ndarray, tol: float, atol: float
) -> bool:
    if _is_traced(res_norms):
        # A Python result flag cannot depend on traced values. Differentiable
        # full-state routes enforce the same per-column gate at execution time
        # in ``_residual_guard``; this flag remains optimistic under tracing.
        return True
    rhs_norms = np.linalg.norm(np.asarray(rhs2d), axis=0)
    targets = np.maximum(atol, tol * rhs_norms)
    res = np.asarray(res_norms)
    return bool(
        np.all(np.isfinite(rhs_norms))
        and np.all(np.isfinite(res))
        and np.all(res >= 0.0)
        and np.all(res <= targets)
    )

def _say(emit: Callable[[str], None] | None, line: str) -> None:
    """Route a status line to the caller's sink, or nowhere when silenced."""
    if emit is not None:
        emit(line)

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

# Accepted slack over the residual the caller asked for, before a
# differentiable solve is declared failed.  The default of :func:`solve`'s
# ``adjoint_residual_factor``.
DEFAULT_ADJOINT_RESIDUAL_FACTOR = 1.0

# Diagnostic backward-error scale; it does not enlarge the acceptance gate.
_ADJOINT_FLOOR_SLACK = 32.0

# Retain the historical cap for comparable diagnostics, not for admission.
_ADJOINT_FLOOR_MAX_RELATIVE = 1e-6

# Rademacher probes used to estimate ``||A||`` for that floor.  Each costs one
# operator apply against the hundreds the Krylov solve itself runs.
_ADJOINT_NORM_PROBES = 2

_EPS64 = float(np.finfo(np.float64).eps)

def _operator_norm_estimate(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], n: int
) -> jnp.ndarray:
    """Probe estimate of ``||A||_2`` from a few Rademacher matvecs.

    ``max_i ||A u_i|| / ||u_i||`` over fixed pseudo-random sign vectors: a
    lower bound on the spectral norm, tight to within a small factor for the
    operators here, and used only to set the order of magnitude of the
    diagnostic backward-error scale in :func:`_residual_guard`.
    """
    key = jax.random.PRNGKey(0)
    est = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(_ADJOINT_NORM_PROBES):
        u = jax.random.rademacher(jax.random.fold_in(key, i), (n,), dtype=jnp.float64)
        est = jnp.maximum(est, jnp.linalg.norm(matvec(u)) / jnp.linalg.norm(u))
    return est

def _residual_guard(
    label: str,
    rhs_index: int,
    *,
    tol: float,
    atol: float,
    factor: float,
    raise_on_failure: bool,
    diagnostics: AdjointDiagnostics | None,
    solver_name: str = "GCROT",
) -> Callable[..., None]:
    """Host callback that records — and by default aborts on — a bad solve.

    Used on forward and transposed solves of the differentiable full-factor
    structured and recycled Krylov paths.  A stalled *adjoint* solve is
    the dangerous one: the forward solution stays good, nothing in the returned
    :class:`SolveResult` changes, and the implicit-function-theorem VJP hands
    back a wrong gradient (the singular Fokker-Planck +
    ``constraintScheme=1`` failure mode).  Runs at execution time via
    ``jax.debug.callback``, so it fires under ``jit``/``grad`` tracing and,
    for the adjoint, during the backward pass.

    The residual handed in is recomputed from the operator itself
    (``||A^T y - g||``), never the Krylov method's internal recursive
    estimate. Acceptance requires ``factor * max(atol, tol*||b||)``.
    The backward-error scale is recorded for diagnosis only: a small backward
    error does not establish the accuracy of an ill-conditioned adjoint or
    of the requested observable. Homogeneous equations require zero defect.
    """

    def guard(
        res_norm: jnp.ndarray,
        rhs_norm: jnp.ndarray,
        sol_norm: jnp.ndarray,
        op_norm: jnp.ndarray,
        zero_defect: jnp.ndarray | None = None,
    ) -> None:
        res = float(np.asarray(res_norm))
        b_norm = float(np.asarray(rhs_norm))
        x_norm = float(np.asarray(sol_norm))
        a_norm = float(np.asarray(op_norm))
        target = max(atol, tol * b_norm)
        floor = min(
            _ADJOINT_FLOOR_SLACK * _EPS64 * (a_norm * x_norm + b_norm),
            _ADJOINT_FLOOR_MAX_RELATIVE * b_norm,
        )
        limit = factor * target
        ok = bool(
            np.isfinite(res)
            and np.isfinite(x_norm)
            and np.isfinite(a_norm)
            and np.isfinite(b_norm)
            and res <= limit
            and ((b_norm > 0 and target > 0) or bool(res == 0 if zero_defect is None else zero_defect))
        )
        record = SolveResidualRecord(
            label=label,
            rhs_index=rhs_index,
            residual_norm=res,
            rhs_norm=b_norm,
            solution_norm=x_norm,
            operator_norm=a_norm,
            target=target,
            floor=floor,
            limit=limit,
            within_tolerance=ok,
        )
        if diagnostics is not None:
            diagnostics._record(record)
        if ok or not raise_on_failure:
            return
        rel = record.relative_residual
        formula = (
            "||A^T y - g|| / ||g||"
            if label.startswith("adjoint")
            else ("||A x - b|| / ||b||")
        )
        raise RuntimeError(
            f"[dkx.solve] differentiable {label} {solver_name} solve failed to "
            f"converge on right-hand side {rhs_index}: the true relative "
            f"residual {formula} is {rel:.3e} (residual {res:.3e}, "
            f"rhs norm {b_norm:.3e}, solution norm {x_norm:.3e}), above the accepted "
            f"{limit:.3e} = {factor:g} x requested {target:.3e}. "
            f"The diagnostic backward-error scale {floor:.3e} does not "
            "enlarge this gate; homogeneous equations require zero defect. Returning would "
            "leave the implicit-differentiation gradient uncertified, so the "
            "solve aborts instead.\n"
            "Possible causes include ill-conditioning, incomplete constraints, "
            "or insufficient convergence/refinement. A physical forward drive "
            "may be in the range of A even when a generic adjoint cotangent "
            "is not, so a converged forward solve does not certify the adjoint.\n"
            "Remedies: raise the resolution or pick a constraint scheme that "
            "regularizes this operator (SFINCS pairs constraintScheme=1 with "
            "the Fokker-Planck/Sugama operators and 2 with pitch-angle "
            "scattering); referee the case on a direct route "
            "(method='block_tridiagonal' where available, or method='direct' "
            "for a non-differentiable check of the forward solve); tighten "
            "tol and raise max_restarts; or, if you have verified the "
            "gradient against finite differences yourself, pass "
            "check_adjoint=False (or raise adjoint_residual_factor) and read "
            "SolveResult.adjoint for the residuals behind this decision."
        )

    return guard

def _guarded_solve(
    label: str,
    rhs_index: int,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_col: jnp.ndarray,
    x: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    factor: float,
    raise_on_failure: bool,
    diagnostics: AdjointDiagnostics | None,
    solver_name: str = "GCROT",
) -> jnp.ndarray:
    """Measure ``x``'s true residual against ``matvec`` and hand it to the guard.

    Returns ``x`` unchanged: the check observes, it never alters the solution,
    so forward answers are bit-identical with and without it.
    """
    residual = matvec(x) - rhs_col
    jax.debug.callback(
        _residual_guard(
            label,
            rhs_index,
            tol=tol,
            atol=atol,
            factor=factor,
            raise_on_failure=raise_on_failure,
            diagnostics=diagnostics,
            solver_name=solver_name,
        ),
        jnp.linalg.norm(residual),
        jnp.linalg.norm(rhs_col),
        jnp.linalg.norm(x),
        _operator_norm_estimate(matvec, rhs_col.shape[0]),
        jnp.all(residual == 0),
    )
    return x

def _krylov_basis_budget_bytes(budget_gb: float | None) -> float:
    """Bytes the widened FGMRES basis may take: argument, environment, or a
    quarter of the memory available now (the pre-2.7 256 MiB where that cannot
    be read)."""
    if budget_gb is None:
        env = os.environ.get(_KRYLOV_BUDGET_ENV)
        budget_gb = float(env) if env not in (None, "") else None
    if budget_gb is not None:
        if not np.isfinite(budget_gb) or budget_gb < 0.0:
            raise ValueError(
                f"krylov_memory_budget_gb must be finite and nonnegative; got {budget_gb!r}"
            )
        return float(budget_gb) * 2.0**30
    available = _available_memory_bytes()
    if available is None:
        return float(_KRYLOV_BASIS_FALLBACK_BYTES)
    return _KRYLOV_BASIS_MEMORY_FRACTION * float(available)

def _auto_wide_restart(
    total_size: int, itemsize: int, budget_gb: float | None
) -> int:
    """Longest restart, at most ``_AUTO_RESTART_CAP`` and ``total_size``, whose
    single-basis GCROT basis (``restart * total_size * itemsize`` bytes) fits the budget."""
    per_step = max(int(total_size), 1) * int(itemsize)
    fit = int(_krylov_basis_budget_bytes(budget_gb) // per_step)
    return max(0, min(_AUTO_RESTART_CAP, int(total_size), fit))

def _stop_gradient_tree(tree: Any) -> Any:
    """``stop_gradient`` on every traced inexact leaf of a pytree.

    Concrete leaves carry no derivative and are returned as they are, so an
    eager caller pays for the few leaves it differentiates, not for all of
    them.
    """

    def stop(leaf: Any) -> Any:
        if isinstance(leaf, jax.core.Tracer) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            return jax.lax.stop_gradient(leaf)
        return leaf

    return jax.tree_util.tree_map(stop, tree)

def _column_apply(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], v: jnp.ndarray
) -> jnp.ndarray:
    return matvec(v) if v.ndim == 1 else jax.vmap(matvec, in_axes=1, out_axes=1)(v)

def _operator_apply(op: KineticOperator, v: jnp.ndarray, transpose: bool) -> jnp.ndarray:
    return _column_apply(_pinned_matvecs(op)[1 if transpose else 0], v)

# Residual guards of compiled implicit solves, by token. The compiled program
# carries an integer token rather than the guard itself, so one executable
# serves every call whatever its tolerances and diagnostics sink; the callback
# looks the guard up when it fires, which for the adjoint is during the
# backward pass. Bounded: the oldest are dropped first, long after any
# backward pass that could still need them has run.
_GUARD_REGISTRY_SIZE = 1 << 14

_guard_registry: OrderedDict[int, Callable[[str, int], Callable[..., None]]] = OrderedDict()

_guard_tokens = itertools.count(1)

_guard_lock = threading.Lock()

def _register_guard(make_guard: Callable[[str, int], Callable[..., None]]) -> int:
    with _guard_lock:
        token = next(_guard_tokens) % (2**31 - 1)
        _guard_registry[token] = make_guard
        while len(_guard_registry) > _GUARD_REGISTRY_SIZE:
            _guard_registry.popitem(last=False)
    return token

def _dispatch_guard(label: str, rhs_index: int, token: Any, *stats: Any) -> None:
    make_guard = _guard_registry.get(int(np.asarray(token)))
    if make_guard is None:
        raise RuntimeError(
            "[dkx.solve] the residual guard of a differentiable structured "
            f"solve (token {int(np.asarray(token))}) is no longer registered; "
            "the gradient cannot be certified"
        )
    make_guard(label, rhs_index)(*stats)

# =============================================================================
# Route drivers
# =============================================================================


def _implicit_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    matvec_t: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_col: jnp.ndarray,
    fwd_solve: Callable[[jnp.ndarray], jnp.ndarray],
    t_solve: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    has_aux: bool = False,
):
    """One differentiable solve via ``solvax.implicit.linear_solve``.

    The single ``solver`` callable required by the API dispatches between the
    forward and transposed factorized solves by identity of the matvec it is
    handed (``linear_solve`` passes ``transpose_matvec`` through verbatim).

    With ``has_aux`` both callables return ``(x, aux)`` and the *forward*
    solve's ``aux`` comes back beside the solution.  That is how the recycled
    Krylov route reads its iteration count, residual and recycle pair off the
    one solve the wrapper runs, instead of running a second solve outside it
    just to see them.
    """

    def solver(mv: Callable, b: jnp.ndarray):
        return t_solve(b) if mv is matvec_t else fwd_solve(b)

    return solvax_linear_solve(
        matvec, rhs_col, solver, transpose_matvec=matvec_t, has_aux=has_aux
    )
