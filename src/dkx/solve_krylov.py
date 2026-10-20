"""Recycled Krylov route of :func:`dkx.solve.solve` (code name ``tier2``).

Preconditioner choice, GCROT with recycling, and the escalation to sparse
direct when the iteration cap is reached.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable
import os
import sys
import time

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:  # noqa: E402
    from solvax.krylov import gcrot
except ImportError:
    gcrot = None  # type: ignore[assignment, misc]

from dkx.coarse_precond import (  # noqa: E402
    _available_memory_bytes,
    _coarse_bands_fit,
    build_coarse_preconditioner,
)
from dkx.drift_kinetic import (  # noqa: E402
    KineticOperator,
)
from dkx.profiling import (  # noqa: E402
    make_emit,
    maybe_profiler,
)
from dkx.solve_common import (  # noqa: E402
    AdjointDiagnostics,
    DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    SolveResult,
    _AUTO_RESTART_MIDDLE,
    _AUTO_RESTART_MIDDLE_CYCLES,
    _AUTO_RESTART_PROBE_CYCLES,
    _KRYLOV_BUDGET_ENV,
    _as_columns,
    _auto_wide_restart,
    _guarded_solve,
    _implicit_solve,
    _krylov_basis_budget_bytes,
    _pinned_matvecs,
)
from dkx.solve_direct import (  # noqa: E402
    _solve_tier3,
    superlu_fits,
    superlu_peak_bytes,
)
from dkx.structured_direct import (  # noqa: E402
    _is_traced,
    coupled_available,
)

# Recycled Krylov preconditioner routes of ``solve(preconditioner=...)``.  All
# three invert the *same* SFINCS-simplified operator and differ only in how.
# ``"coarse"`` is the historical default (exact block-Thomas over L with dense
# ``Ntheta*Nzeta`` blocks); ``"multigrid"`` swaps its inner f-block inverse for
# the semicoarsened V-cycle of :mod:`dkx.multigrid`; ``"sparse"`` keeps the
# inverse exact but eliminates in a fill-reducing order on the host
# (:mod:`dkx.sparse_precond`), which is what the Fortran reference does.
# Legendre blocks the "coupled" preconditioner factors with full speed coupling.
_COUPLED_PRECOND_KEEP = 2

_COUPLED_PRECOND_F32 = True  # float32 dense L < keep factors too (same iterations on HSX FP)

_TIER2_PRECONDITIONERS = ("coarse", "coarse_triangle", "coupled", "multigrid", "sparse", "sparse_triangle", "none")

def _escalate_after_tier2_stall(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    stalled: SolveResult,
    tol: float,
    atol: float,
    x0,
    recycle,
    preconditioner: str,
    drop_l_coupling_in_precond: bool,
    restart: int,
    recycle_dim: int,
    max_restarts: int,
    check_adjoint: bool,
    adjoint_residual_factor: float,
    max_dense_size: int,
    direct_backend: str | None = None,
    direct_memory_budget_gb: float | None = None,
    auto_restart: bool = False,
    krylov_memory_budget_gb: float | None = None,
) -> SolveResult:
    """Try alternate preconditioners and a larger iteration budget at fixed physics.

    Return the first converged result. If every iterative attempt fails, use
    the host direct route only within its size guard; otherwise raise with the
    best attempted residual. A stall alone does not diagnose its cause.
    """
    attempts: list[tuple[str, SolveResult]] = [
        (f"{preconditioner} preconditioner", stalled)
    ]

    def _residual(result: SolveResult) -> float:
        norms = np.asarray(result.residual_norms)
        return float(norms[-1]) if norms.size else float("inf")

    print(
        f"[dkx.solve] recycled Krylov solve stalled with the {preconditioner} "
        f"preconditioner (iterations={stalled.iterations}, residual="
        f"{_residual(stalled):.3e}); escalating rather than giving up."
    )

    def _warm_start(result: SolveResult, fallback):
        """The stalled iterate, when starting a rung there beats starting over.

        Every rung solves the same equation, so an iterate the stalled solve
        reached is the work already paid for.  It is used only when it is
        finite and its residual improved on the right-hand side it started
        from: a diverged iterate would begin the next rung further away than
        the original guess.
        """
        if not np.isfinite(_residual(result)) or _residual(result) >= 1.0:
            return fallback
        x = np.asarray(result.x)
        if x.shape != rhs2d.shape or not np.all(np.isfinite(x)):
            return fallback
        return x

    warm = _warm_start(stalled, x0)

    # Rung 1: a preconditioner that can do what the stalled one could not.
    #
    # "coarse", "sparse" and "multigrid" are three inverses of one simplified
    # operator, so they differ in fill and cost, not in what they approximate.
    # When the stall comes from the Fokker-Planck speed coupling that
    # ``preconditioner_x = 1`` drops, none of them can fix it: on NCSX that
    # coupling is what makes GCROT iterations grow with Ntheta
    # (2026-09-14 ntheta-iteration-growth study).  Retaining its
    # upper triangle changes the operator being inverted and cuts iterations
    # 2.3-4.1x there, and it reuses the factors the coarse route already built,
    # so it is both the matched remedy and the cheapest rung to try
    # (2026-09-18 speed-triangle-back-substitution study).  It is
    # skipped where there is no dense collision operator to retain, such as
    # pitch-angle scattering, since there it repeats the stalled solve.
    #
    # "sparse" follows because it eliminates in a fill-reducing order where
    # "coarse" eliminates L first and fills the angular stencils in; with a
    # dense collision operator it keeps the same triangle (2.9-3.1x fewer
    # iterations on HSX Fokker-Planck decks, 2026-10-05 ladder of #303).
    kinds = ("sparse", "multigrid")
    if op.fp is not None or op.sugama is not None:
        kinds = ("coarse_triangle", "sparse_triangle", "multigrid")
    for kind in kinds:
        if kind == preconditioner:
            continue
        try:
            print(
                f"[dkx.solve]   retrying the iterative solve with the {kind} preconditioner ..."
            )
            candidate = _solve_tier2(
                op,
                rhs2d,
                tol=tol,
                atol=atol,
                x0=warm,
                recycle=recycle,
                preconditioner=kind,
                drop_l_coupling_in_precond=drop_l_coupling_in_precond,
                restart=restart,
                recycle_dim=recycle_dim,
                max_restarts=max_restarts,
                differentiable=False,
                check_adjoint=check_adjoint,
                adjoint_residual_factor=adjoint_residual_factor,
                auto_restart_recovery=auto_restart,
                krylov_memory_budget_gb=krylov_memory_budget_gb,
            )
        except (
            Exception
        ) as exc:  # a preconditioner that cannot build is a rung, not a failure
            print(f"[dkx.solve]   {kind} preconditioner unavailable: {exc}")
            continue
        attempts.append((f"{kind} preconditioner", candidate))
        if candidate.converged:
            print(f"[dkx.solve]   converged with the {kind} preconditioner.")
            return candidate

    # Rung 2: more iterations.  Cheap to ask for, and the remedy when the
    # preconditioner is adequate but the cap was simply too low for this Er.
    # Under the restart=None policy the budget is spent at the wide restart:
    # the measured stalls of this route are restart stagnation
    # (2026-09-23 restart-and-direct-reach study), which more
    # short cycles do not cure.
    widened = max(max_restarts * 4, max_restarts + 1)
    rung_restart = restart
    if auto_restart:
        wide = _auto_wide_restart(
            op.total_size, rhs2d.dtype.itemsize, krylov_memory_budget_gb
        )
        if wide > restart:
            rung_restart = wide
            widened = max(1, restart * widened // wide)
    best_label_so_far, best_so_far = max(attempts, key=lambda item: -_residual(item[1]))
    best_kind = best_label_so_far.split()[0]
    if best_kind not in _TIER2_PRECONDITIONERS:
        best_kind = preconditioner
    # Continue from whichever rung came closest, for the same reason.
    warm = _warm_start(best_so_far, warm)
    try:
        print(
            f"[dkx.solve]   retrying the iterative solve with the {best_kind} preconditioner "
            f"and {widened} restarts of {rung_restart} (was {max_restarts} of {restart}) ..."
        )
        candidate = _solve_tier2(
            op,
            rhs2d,
            tol=tol,
            atol=atol,
            x0=warm,
            recycle=recycle,
            preconditioner=best_kind,
            drop_l_coupling_in_precond=drop_l_coupling_in_precond,
            restart=rung_restart,
            recycle_dim=recycle_dim,
            max_restarts=widened,
            differentiable=False,
            check_adjoint=check_adjoint,
            adjoint_residual_factor=adjoint_residual_factor,
        )
        attempts.append((f"{best_kind} preconditioner, {widened} restarts", candidate))
        if candidate.converged:
            print("[dkx.solve]   converged with the widened iteration budget.")
            return candidate
    except Exception as exc:
        print(f"[dkx.solve]   widened retry failed: {exc}")

    # Rung 3: sparse direct.  An explicitly selected, budgeted MUMPS backend may use
    # grouped assembly; max_dense_size limits only the column-sampling fallback in that
    # case.  SuperLU above max_dense_size is admitted when its estimated peak fits the
    # memory available now (it is the answer of last resort, not a cheap one: minutes to
    # an hour at tens of thousands of unknowns).
    large_mumps_requested = (
        direct_backend == "mumps" and direct_memory_budget_gb is not None
    )
    large_superlu_fits = direct_backend in (None, "superlu") and superlu_fits(op.total_size)
    if op.total_size <= max_dense_size or large_mumps_requested or large_superlu_fits:
        print("[dkx.solve]   falling back to the sparse direct host solve ...")
        return _solve_tier3(
            op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size,
            direct_backend=direct_backend,
            direct_memory_budget_gb=direct_memory_budget_gb,
        )

    best_label, best = min(attempts, key=lambda item: _residual(item[1]))
    if direct_backend == "mumps" and direct_memory_budget_gb is None:
        direct_refusal = (
            "The large host direct fallback was not attempted because explicit "
            "MUMPS requires a positive direct_memory_budget_gb."
        )
    else:
        direct_refusal = (
            f"The default SuperLU host fallback is protected above "
            f"max_dense_size={max_dense_size} unless its estimated peak "
            f"({superlu_peak_bytes(op.total_size) / 2.0**30:.1f} GiB) fits the available "
            "memory; it was not attempted. Free memory, or select "
            "direct_backend='mumps' with a memory budget to admit grouped assembly."
        )
    raise RuntimeError(
        f"the linear solve did not converge at total_size={op.total_size}.\n"
        f"Tried: {'; '.join(label for label, _ in attempts)}.\n"
        f"Best final residual {_residual(best):.3e} ({best_label}), tolerance {tol:.1e}.\n"
        f"{direct_refusal}\n"
        "Nonconvergence alone does not identify under-resolution, an ill-conditioned "
        "operator, or a preconditioner problem, and does not establish that the "
        "physical model has no solution.\n"
        "Inspect the residual history and available memory; compare preconditioners "
        "at fixed physics. Then check Nxi, Nx, Ntheta and Nzeta convergence of "
        "the fluxes and current separately. A small linear residual does not "
        "establish resolution convergence. Keep Er and the requested tolerance "
        "fixed when comparing solvers.\n"
        "Pass solver=SolverOptions(...) to control the solver directly."
    )

def _auto_preconditioner(op: KineticOperator, budget_gb: float | None) -> str:
    """``"coupled"`` for Fokker-Planck/Sugama decks whose coupled build fits, else ``"coarse"``.

    With the E_r xiDot/xDot terms active and nonzero, a deck whose dense coarse bands fit
    takes ``"coarse_triangle"`` instead. ``coupled`` drops those terms from the operator it
    factors and, past its kept low modes, the speed coupling too; at |E_r| >= 15 on an HSX QHS
    full-FP deck (66,004 unknowns) it stalled after 768 to 1,262 iterations where
    ``coarse_triangle`` converged in 350 and 208 (65 and 52 s, against 141 and 258 s spent
    before the stall was declared). At |E_r| = 10 it is 22 s against 32 s; at E_r = 0 the
    terms vanish and ``coupled`` stays (7 s against 16 s).

    The coupled estimate is a whole-process peak, so it is held against
    ``krylov_memory_budget_gb`` (or ``DKX_KRYLOV_MEMORY_BUDGET_GB``) when given,
    and otherwise against half the memory available now: the program trades
    memory for speed (17 vs 46 iterations, 508 vs 1,100 s on HSX FP).
    """
    from dkx.structured_direct import coupled_precond_peak_bytes  # noqa: PLC0415

    if not isinstance(op, KineticOperator):  # stand-ins in policy tests
        return "coarse"
    base = replace(op, with_er_xidot=False, with_er_xdot=False, with_magnetic_drifts=False)
    if _is_traced(*jax.tree_util.tree_leaves(op)) or not coupled_available(base)[0]:
        return "coarse"
    if (
        (op.with_er_xidot or op.with_er_xdot)
        and float(jnp.abs(op.dphi_hat_dpsi_hat_kinetic)) != 0.0
        and _coarse_bands_fit(op)
    ):
        return "coarse_triangle"
    if budget_gb is not None or os.environ.get(_KRYLOV_BUDGET_ENV) not in (None, ""):
        budget = _krylov_basis_budget_bytes(budget_gb)
    else:
        available = _available_memory_bytes()
        budget = 0.5 * available if available is not None else 0.0
    return "coupled" if coupled_precond_peak_bytes(op, _COUPLED_PRECOND_KEEP) <= budget else "coarse"

def _resolve_preconditioner(
    preconditioner: str | None, use_preconditioner: bool
) -> str:
    """Normalize the recycled Krylov preconditioner request.

    ``preconditioner=None`` keeps the historical ``use_preconditioner`` boolean
    (``True`` -> the coarse block-Thomas preconditioner, ``False`` -> none), so
    every existing caller is byte-for-byte unaffected.
    """
    if preconditioner is None:
        return "coarse" if use_preconditioner else "none"
    name = str(preconditioner).strip().lower()
    if name not in _TIER2_PRECONDITIONERS:
        raise ValueError(
            f"unknown preconditioner {preconditioner!r}; expected one of "
            f"{sorted(_TIER2_PRECONDITIONERS)}"
        )
    return name

def build_tier2_preconditioner(
    op: KineticOperator, kind: str, *, drop_l_coupling: bool = False
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """``(precond, precond_t)`` for the requested Krylov preconditioner.

    ``"coarse"`` is :func:`build_coarse_preconditioner` (the exact block-Thomas
    factorization of the SFINCS-simplified operator); ``"coarse_triangle"`` is
    the same factorization with the collision operator's upper speed triangle
    retained, Fortran ``preconditioner_x = 2``, applied by back-substitution
    over ``x`` rather than one batched solve; ``"multigrid"`` is
    :func:`dkx.multigrid.build_multigrid_preconditioner`, which approximates
    the inverse of that *same* operator with a semicoarsened multigrid V-cycle
    and so is affordable where the cubic-in-``Ntheta*Nzeta`` factorization is
    not; ``"sparse"`` is :func:`dkx.sparse_precond.build_sparse_preconditioner`,
    which keeps the inverse exact but eliminates in a fill-reducing order on the
    host instead of eliminating ``L`` first, so the angular stencils stay sparse.
    ``"sparse_triangle"`` retains the same speed triangle as ``"coarse_triangle"``
    using these sparse factors. All routes eliminate the border identically, and
    none changes the solution: they change how fast the Krylov route gets
    there.
    """
    if kind == "coarse":
        return build_coarse_preconditioner(op, drop_l_coupling=drop_l_coupling)
    if kind == "coarse_triangle":
        return build_coarse_preconditioner(
            op, drop_l_coupling=drop_l_coupling, retain_speed_triangle=True
        )
    if kind == "coupled":
        from dkx.structured_direct import coupled_preconditioner  # noqa: PLC0415

        return coupled_preconditioner(op, _COUPLED_PRECOND_KEEP, _COUPLED_PRECOND_F32)
    if kind in ("sparse", "sparse_triangle"):
        from dkx.sparse_precond import build_sparse_preconditioner  # noqa: PLC0415

        return build_sparse_preconditioner(
            op, drop_l_coupling=drop_l_coupling,
            retain_speed_triangle=kind == "sparse_triangle",
        )
    from dkx.multigrid import build_multigrid_preconditioner  # noqa: PLC0415

    return build_multigrid_preconditioner(op, drop_l_coupling=drop_l_coupling)

def _solve_tier2(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    x0: jnp.ndarray | None,
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None,
    preconditioner: str,
    drop_l_coupling_in_precond: bool,
    restart: int,
    recycle_dim: int,
    max_restarts: int,
    differentiable: bool,
    check_adjoint: bool,
    adjoint_residual_factor: float = DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    prebuilt_precond: tuple[Callable, Callable] | None = None,
    auto_restart_recovery: bool = False,
    krylov_memory_budget_gb: float | None = None,
) -> SolveResult:
    if int(recycle_dim) < 1:
        # solvax's GCROT scatters the recycled subspace into a (n, k) array, so
        # k = 0 dies inside jax indexing with "index is out of bounds for axis 1
        # with size 0" -- a traceback naming neither the parameter nor DKX.
        # Plain restarted FGMRES is not reachable through this route; refusing
        # by name is better than a crash that looks like a library bug.
        raise ValueError(
            f"recycle_dim={recycle_dim}: the recycled Krylov route needs at least "
            "one recycled direction. Use recycle_dim >= 1, or method='direct' for "
            "a non-recycling solve."
        )
    traced = _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    profiler = None if traced else maybe_profiler(emit=make_emit(stream=sys.stderr))
    t0 = time.perf_counter()
    precond = precond_t = None
    # The escalation path deliberately does NOT pass a prebuilt pair: it is
    # searching over preconditioner kinds, so pinning one would defeat the
    # search it exists to perform.
    if prebuilt_precond is not None:
        # A caller that solves the same operator family repeatedly -- an Er
        # scan, a Newton loop, a batched sweep -- can build once and hand the
        # pair in. The build is a large fraction of a single call (39% on the
        # Sugama collisionOperator=3 deck), so paying it per point is the
        # single largest avoidable cost in a scan.
        precond, precond_t = prebuilt_precond
    elif preconditioner != "none":
        if profiler is not None:
            profiler.mark("preconditioner_build.start")
        precond, precond_t = build_tier2_preconditioner(
            op, preconditioner, drop_l_coupling=drop_l_coupling_in_precond
        )
        # Force asynchronous construction to complete so the "build" timing is
        # real compute, not JAX dispatch latency.  The coarse routes synchronize
        # on their own factors; a zero probe there compiled and discarded an
        # extra application.  Skipped under jit/grad tracing, where
        # block_until_ready is a no-op on tracers and the probe would only add
        # dead nodes to the trace.
        if not traced and preconditioner not in ("coarse", "coarse_triangle"):
            jax.block_until_ready(
                precond(jnp.zeros((op.total_size,), dtype=jnp.float64))
            )
        if profiler is not None:
            profiler.mark("preconditioner_build.complete")
    t1 = time.perf_counter()

    x0_2d = None
    if x0 is not None:
        x0_2d, _ = _as_columns(x0)
        if x0_2d.shape != rhs2d.shape:
            raise ValueError(
                f"x0 shape {x0_2d.shape} must match rhs shape {rhs2d.shape}"
            )

    # Pinned matvecs: identical to op.apply on the physical subspace, identity
    # on the Nxi_for_x-truncated DOFs, so the system (and in particular its
    # transpose, used by the differentiable adjoint) is nonsingular.
    matvec, matvec_t = _pinned_matvecs(op)
    diagnostics = (
        AdjointDiagnostics(
            tol=float(tol),
            atol=float(atol),
            factor=float(adjoint_residual_factor),
            checked=bool(check_adjoint),
        )
        if differentiable
        else None
    )
    # The widened windows of the host restart policy, sized once per call from
    # the memory available after the preconditioner is built, within the
    # original inner-step cap ``restart * max_restarts``.
    windows_after_probe: list[tuple[int, int]] = []
    if (
        auto_restart_recovery and not differentiable and not traced
        and prebuilt_precond is None and restart > 0
        and max_restarts > _AUTO_RESTART_PROBE_CYCLES
    ):
        wide = _auto_wide_restart(
            op.total_size, rhs2d.dtype.itemsize, krylov_memory_budget_gb
        )
        steps = restart * (max_restarts - _AUTO_RESTART_PROBE_CYCLES)
        if restart < _AUTO_RESTART_MIDDLE < wide:
            middle = min(_AUTO_RESTART_MIDDLE_CYCLES, steps // _AUTO_RESTART_MIDDLE)
            if middle > 0:
                windows_after_probe.append((_AUTO_RESTART_MIDDLE, middle))
                steps -= _AUTO_RESTART_MIDDLE * middle
        if wide > restart and steps // wide > 0:
            windows_after_probe.append((wide, steps // wide))
    # DKX's preconditioners are fixed linear maps, so the wide window (the one
    # the memory budget sizes) keeps one Krylov basis. Short windows and the
    # differentiable route keep Z: forming M^{-1}(V y) amplifies roundoff by
    # ||M^{-1}||, about 1e8 on the near-singular coarse chains, and there a
    # 30-step cycle diverged where the stored-Z cycle converged.
    # Caller-supplied preconditioners may be flexible and always keep Z.
    fixed_precond = prebuilt_precond is None
    cols: list[jnp.ndarray] = []
    total_iters: int | None = 0
    converged = True
    res_norms: list[jnp.ndarray] = []
    if profiler is not None:
        # The first GCROT call traces, compiles, and executes through JAX; this
        # boundary deliberately does not claim those costs can be split.
        profiler.mark("krylov_compile_and_execute.start")
    for j in range(rhs2d.shape[1]):
        b = rhs2d[:, j]
        if differentiable:
            # One forward solve, not two.  ``custom_linear_solve`` *defines*
            # the primal as ``solve(matvec, b)``, so the wrapper runs a GCROT
            # of its own no matter what; this used to sit on top of a plain
            # GCROT whose solution was then thrown away, and every
            # differentiable Krylov solve paid for both.  ``has_aux`` carries
            # the recycle pair, iteration count, convergence flag and residual
            # out of the wrapper's own solve instead, so there is nothing left
            # to recompute outside it.
            #
            # The wrapped solve takes no ``x0``: ``custom_linear_solve``
            # applies the same solver to *tangent* right-hand sides, for which
            # the primal's warm start is a bad initial guess.  The recycle pair
            # is right-hand-side independent -- a deflation subspace, with
            # ``A U`` recomputed and re-orthonormalized against the current
            # operator on entry -- so it is still threaded across columns.
            #
            # Both the forward and the adjoint (transposed) solves have their
            # *true* residual recomputed from the operator and recorded in
            # ``diagnostics``; with check_adjoint on, a miss aborts loudly
            # instead of silently corrupting the gradient
            # (jax.debug.callback fires at execution time, i.e. during the
            # backward pass for the adjoint).
            def _measured(
                label: str,
                mv: Callable[[jnp.ndarray], jnp.ndarray],
                pc: Callable[[jnp.ndarray], jnp.ndarray] | None,
                rhs_col: jnp.ndarray,
                warm: tuple[jnp.ndarray, jnp.ndarray] | None,
                index: int = j,
            ):
                s = gcrot(
                    mv,
                    rhs_col,
                    precond=pc,
                    m=restart,
                    k=recycle_dim,
                    rtol=tol,
                    atol=atol,
                    max_restarts=max_restarts,
                    recycle=warm,
                )
                measured = _guarded_solve(
                    label,
                    index,
                    mv,
                    rhs_col,
                    s.x,
                    tol=tol,
                    atol=atol,
                    factor=adjoint_residual_factor,
                    raise_on_failure=check_adjoint,
                    diagnostics=diagnostics,
                )
                return measured, (
                    s.recycle,
                    s.iterations,
                    s.converged,
                    s.residual_norm,
                )

            def fwd_solve(rhs_col: jnp.ndarray, warm=recycle):
                return _measured("forward", matvec, precond, rhs_col, warm)

            def t_solve(rhs_col: jnp.ndarray):
                # The adjoint keeps its own cold start: the forward pair is
                # built inside the wrapper's trace and cannot reach here.
                return _measured(
                    "adjoint (transposed)", matvec_t, precond_t, rhs_col, None
                )

            x_col, aux = _implicit_solve(
                matvec, matvec_t, b, fwd_solve, t_solve, has_aux=True
            )
            recycle, iterations, col_converged, residual_norm = aux
        else:
            # Host-controlled restart policy with factors built for this exact
            # operator: a few short cycles, then the longest restart whose basis
            # fits the memory budget, continuing from the iterate. Whole wide
            # cycles are reserved within the original inner-step cap;
            # caller-supplied (possibly stale) factors keep the explicit path.
            recover = bool(windows_after_probe)
            windows = (
                [(restart, _AUTO_RESTART_PROBE_CYCLES), *windows_after_probe]
                if recover else [(restart, max_restarts)]
            )
            guess = None if x0_2d is None else x0_2d[:, j]
            iterations = 0
            for window, cycles in windows:
                sol = gcrot(
                    matvec, b, x0=guess, precond=precond, m=window,
                    k=recycle_dim, rtol=tol, atol=atol,
                    max_restarts=cycles, recycle=recycle,
                    fixed_precond=fixed_precond and window > _AUTO_RESTART_MIDDLE,
                )
                iterations = iterations + sol.iterations
                recycle = sol.recycle
                if not recover or bool(sol.converged):
                    break
                finite = all(
                    np.all(np.isfinite(np.asarray(value)))
                    for value in jax.tree_util.tree_leaves((sol.x, sol.residual_norm, recycle))
                )
                guess = sol.x if finite else None
                recycle = recycle if finite else None
            x_col = sol.x
            col_converged, residual_norm = sol.converged, sol.residual_norm
        if traced:
            total_iters = None  # iteration counts are tracers under jit/grad
        else:
            total_iters += int(iterations)
            converged = converged and bool(col_converged)
        res_norms.append(residual_norm)
        cols.append(x_col)
    x_stacked = jax.block_until_ready(
        jnp.stack(cols, axis=1)
    )  # real solve time, not dispatch
    residual_norms = jnp.stack(res_norms)
    if profiler is not None:
        # Do not block on the result dataclass itself: synchronize only JAX
        # leaves already produced by GCROT, after the numerical calls finish.
        jax.block_until_ready(
            jax.tree_util.tree_leaves((x_stacked, residual_norms, recycle))
        )
        profiler.mark("krylov_compile_and_execute.complete")
    t2 = time.perf_counter()
    return SolveResult(
        x=x_stacked,
        method="gcrot",
        iterations=total_iters,
        residual_norms=residual_norms,
        converged=converged,
        recycle=recycle,
        timings={"build": t1 - t0, "solve": t2 - t1},
        adjoint=diagnostics,
        # Only the recycled-Krylov route builds one, so a caller that threads
        # this forward gets a preconditioner exactly when there is one to reuse
        # and None when the route was direct.
        precond=None if precond is None else (precond, precond_t),
    )
