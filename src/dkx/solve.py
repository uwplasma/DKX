"""Solver policy for a :class:`KineticOperator`.

Given the drift-kinetic operator (:mod:`dkx.drift_kinetic`) and one or more
right-hand sides, select a supported route and verify its original equations.

The three routes are named as a case file's ``[solver] method`` names them:
``structured_direct``, ``recycled_krylov``, ``sparse_direct_referee``.  Code
identifiers in this module still spell them with the retired tier numbering —
``tier1`` names belong to the structured direct route, ``tier2`` to recycled
Krylov, ``tier3`` to sparse direct — while the prose uses the route names.

Structured direct (``solvax.direct`` block Thomas over Legendre modes)
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
    Full Fokker-Planck and improved Sugama decks of the same trajectory
    family take the speed-coupled form of this route
    (:mod:`dkx.structured_direct`): one chain whose blocks couple every
    (species, x) pair, taken by ``auto`` when it fits the memory budget and
    its flop cap.

Recycled Krylov — preconditioned, with subspace recycling (``solvax.krylov.gcrot``)
    Matrix-free FGMRES+recycling on :meth:`KineticOperator.apply`,
    right-preconditioned by an exact structured direct solve of the
    SFINCS-simplified coarse operator (the Fortran ``preconditionerOptions``
    idiom): ``preconditioner_species=1`` (self-collisions only) and
    ``preconditioner_x=1`` (x-diagonal collisions) reduce Fokker-Planck to a
    PAS-like L-diagonal coefficient; the Er L±2 terms are dropped, which is
    Fortran's ``preconditioner_xi=1`` applied unconditionally.  The bordered
    constraint rows are eliminated exactly through
    ``solvax.operators.schur_projected_precond``.  The recycle pair (C, U) is
    returned for warm-starting continuation (Er scans, Newton steps); where its
    dense bands exceed RAM it eliminates a generated operator instead.  The
    preconditioner itself, its pins and the routing between its three storage
    policies live in :mod:`dkx.coarse_precond`.

Sparse direct — host fallback and cross-check (``solvax.native.SpluFactorization``)
    Assembles the operator into CSR and factors it on the host; small operators
    may instead use column sampling, limited by ``max_dense_size``. SuperLU remains
    the default; the optional MUMPS adapter is an explicit experimental choice.
    Non-differentiable, non-jittable; prints a loud one-line notice. Used on
    explicit request (``method="direct"``) or when recycled Krylov breaches
    its iteration cap under ``method="auto"``.

Differentiability: full-state structured and recycled Krylov routes use
``solvax.implicit.linear_solve`` when ``differentiable=True``. Full-factor
structured solves reuse factors for the transpose. Generated full recovery
retains Schur factors and refines with the original action when its storage
estimate fits the budget; otherwise it uses the generated RHS map's pullback,
including its setup. Partial generated
recovery retains its reduced-system derivative. Sparse direct refuses AD.

Full-state implicit routes recompute forward/transpose residuals from their
original physical actions and record bounded :class:`AdjointDiagnostics`.
Unless ``check_adjoint=False``, a missed requested tolerance (or explicitly
relaxed residual factor) raises at execution time. Padded coordinates cannot
enlarge the generated full route's physical adjoint tolerance.

Fortran correspondence: ``solver.F90`` (KSP setup / preconditioner matrix
``whichMatrix=0``), ``preconditioner.F90`` (the ``preconditioner_*`` knobs),
and the PETSc ``Pmat`` idiom of production SFINCS.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dkx import require_float64  # noqa: E402
from dkx.coarse_precond import (  # noqa: E402
    _require_solvax,
    build_coarse_preconditioner,
)
from dkx.drift_kinetic import KineticOperator  # noqa: E402
from dkx.structured_direct import (  # noqa: E402,F401  (re-exported: dkx.solve API)
    _TIER1_BUDGET_ENV,
    _TIER1_BUDGET_GB_DEFAULT,
    _TIER1_KEEP_LOWEST_DEFAULT,
    CoupledSolver,
    _coupled_route_fits,
    _is_traced,
    _resolve_subsystem_batch,
    _rhs_confined_to_lowest_blocks,
    _tier1_budget_bytes,
    _truncation_supported,
    _uniform_nxi_for_x,
    build_coupled_solver,
    coupled_available,
    coupled_flops,
    coupled_peak_memory_bytes,
    tier1_available,
    tier1_full_band_bytes,
    tier1_peak_memory_bytes,
    tier1_truncated_peak_memory_bytes,
    tier1_truncated_subsystem_width,
)

from dkx.solve_common import (  # noqa: E402,F401  (re-exported: dkx.solve API)
    AdjointDiagnostics,
    DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    SolveResidualRecord,
    SolveResult,
    _AUTO_RESTART_PROBE,
    _as_columns,
    _krylov_basis_budget_bytes,
)
from dkx.solve_structured import (  # noqa: E402,F401  (re-exported: dkx.solve API)
    Tier1Solver,
    _generated_full_factor_bytes,
    _solve_generated_full,
    _solve_tier1,
    _solve_tier1_truncated,
    build_structured_solver,
    build_tier1_solver,
    tier1_truncated_tail_blocks,
)
from dkx.solve_direct import (  # noqa: E402,F401  (re-exported: dkx.solve API)
    DirectFactors,
    _solve_tier3,
    materialize_csr,
    materialize_dense,
)
from dkx.solve_krylov import (  # noqa: E402,F401  (re-exported: dkx.solve API)
    _auto_preconditioner,
    _escalate_after_tier2_stall,
    _resolve_preconditioner,
    _solve_tier2,
    build_tier2_preconditioner,
)

__all__ = [
    "DirectFactors",
    "SolveResult",
    "Tier1Solver",
    "auto_solve_peak_memory_bytes",
    "build_coarse_preconditioner",
    "build_tier1_solver",
    "build_tier2_preconditioner",
    "materialize_csr",
    "materialize_dense",
    "solve",
    "tier1_available",
    "tier1_full_band_bytes",
    "tier1_peak_memory_bytes",
    "tier1_truncated_peak_memory_bytes",
    "tier1_truncated_subsystem_width",
    "tier1_truncated_tail_blocks",
]



# Size-aware device routing (``solve(device=...)``): on an accelerator-default
# host, ``device="auto"`` runs systems at or below these sizes on the host CPU
# instead.  Both thresholds default to 0 — auto-routing OFF — because the
# same-host measurements do not support a nonzero default (36-core Pop!_OS box
# with an RTX A4000, 2026-07-17, docs/benchmarks/performance.md "Same-host CPU/GPU
# crossover"): the GPU won every structured direct warm solve measured down to
# 6.5k DOFs (2.7x-39x) and every preconditioned recycled Krylov warm solve down
# to 2.8k DOFs (1.5x-2.7x).  The one CPU-wins case — the small unpreconditioned
# recycled Krylov loop of the Phi1 Newton solve (4.5k DOFs: warm 0.048 s CPU
# vs 0.159 s GPU) — did NOT recover its win under solve-level routing
# (0.12-0.13 s: the per-Newton-iteration residuals stay on the GPU and each
# routed solve pays device transfers plus a one-time CPU compile), so routing
# small Phi1 workloads is best done whole-process (``JAX_PLATFORMS=cpu``), not
# per solve.  The knob remains for hosts where the balance differs: set e.g.
# ``DKX_SOLVE_CPU_MAX_SIZE_TIER2=6000`` to route small recycled Krylov solves.
_SOLVE_DEVICE_ENV = "DKX_SOLVE_DEVICE"

_SOLVE_CPU_MAX_TIER1_ENV = "DKX_SOLVE_CPU_MAX_SIZE_TIER1"

_SOLVE_CPU_MAX_TIER2_ENV = "DKX_SOLVE_CPU_MAX_SIZE_TIER2"

_SOLVE_CPU_MAX_TIER1_DEFAULT = 0

_SOLVE_CPU_MAX_TIER2_DEFAULT = 0

# =============================================================================
# The auto-policy entry point
# =============================================================================


def _auto_route_structural(
    op: KineticOperator,
    budget_gb: float | None = None,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
) -> str:
    """The ``method="auto"`` structured route decided from operator structure.

    The RHS-free twin of :func:`_auto_route`, for callers that must predict the
    route without a right-hand side (the memory model behind
    :func:`auto_solve_peak_memory_bytes` and the batched-scan chunk sizing in
    :mod:`dkx.batch`).  The one RHS-dependent check —
    :func:`_rhs_confined_to_lowest_blocks` — is replaced by the structural
    RHSMode 1/2/3 guarantee (drives and output moments on ``l <= 2``), exactly
    the fallback :func:`_auto_route` itself uses when the RHS is traced.  Any
    change to the routing conditions must be applied to both functions.

    Returns ``"block_tridiagonal"``, ``"block_tridiagonal_truncated"``, or
    ``"gmres"``.
    """
    ok, _reason = tier1_available(op)
    if not ok:
        return "block_tridiagonal" if _coupled_route_fits(op, budget_gb) else "gmres"
    budget_bytes, _ = _tier1_budget_bytes(budget_gb)
    if _uniform_nxi_for_x(op) and tier1_peak_memory_bytes(op) <= budget_bytes:
        return "block_tridiagonal"
    keep = min(keep_lowest, op.n_xi)
    sup_ok, _sup_reason = _truncation_supported(op, keep)
    if sup_ok and int(op.rhs_mode) in (1, 2, 3):
        return "block_tridiagonal_truncated"
    return "gmres"

def auto_solve_peak_memory_bytes(
    op: KineticOperator,
    budget_gb: float | None = None,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
) -> float:
    """Peak-memory estimate of the solve ``method="auto"`` would run on ``op``.

    Follows the auto-router's own decision (:func:`_auto_route_structural`) so
    the estimate models the kernel that actually executes: a solve that routes
    to the truncated block-Thomas kernel is charged its truncated working set
    (:func:`tier1_truncated_peak_memory_bytes`) — the full Legendre bands are
    never allocated on that route, and charging their factorization peak
    overstates a ramped or budget-forced production solve by ~46x.  The
    full-band route keeps the factorization peak
    (:func:`tier1_peak_memory_bytes`); the recycled Krylov GCROT fallback keeps
    it too, as a deliberately conservative stand-in (its matvec working set is
    smaller, but it has no validated model of its own).
    """
    route = _auto_route_structural(op, budget_gb, keep_lowest)
    if route == "block_tridiagonal" and not tier1_available(op)[0]:
        return coupled_peak_memory_bytes(op)
    if route == "block_tridiagonal_truncated":
        peak = tier1_truncated_peak_memory_bytes(op, keep_lowest=keep_lowest)
        # Full-state AD can retain Schur factors across primal/transpose calls.
        # Include that optional state in batch sizing even for forward callers.
        if keep_lowest >= op.n_xi:
            peak += 2 * _generated_full_factor_bytes(op)
        return peak
    return tier1_peak_memory_bytes(op)

def _auto_route(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    budget_gb: float | None,
    keep_lowest: int,
    subsystem_batch: int | str = "auto",
    emit: Callable[[str], None] | None = print,
) -> str:
    """Pick the route for ``method="auto"`` and print a Fortran-style one-liner.

    Structural changes to the routing conditions here must be mirrored in
    :func:`_auto_route_structural` (the RHS-free twin used by the memory
    model).
    """
    ok, _reason = tier1_available(op)
    if not ok:
        if _coupled_route_fits(op, budget_gb, int(rhs2d.shape[1])):
            if emit is not None:
                emit(
                    "[dkx.solve] structured direct route: speed-coupled block "
                    f"elimination over L; peak estimate "
                    f"{coupled_peak_memory_bytes(op) / 2.0**30:.2f} GB, "
                    f"{coupled_flops(op) / 1e9:.0f} GFlop."
                )
            return "block_tridiagonal"
        return "gmres"

    peak = tier1_peak_memory_bytes(op)
    bands = tier1_full_band_bytes(op)
    budget_bytes, budget_gb_val = _tier1_budget_bytes(budget_gb)
    peak_gb = peak / 2.0**30
    uniform = _uniform_nxi_for_x(op)
    if uniform and peak <= budget_bytes:
        if emit is not None:
            emit(
                f"[dkx.solve] structured direct route: full factorization; "
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
        width = _resolve_subsystem_batch(op, subsystem_batch, keep)
        n_sub = max(1, int(op.n_species) * int(op.n_x))
        trunc_gb = (
            tier1_truncated_peak_memory_bytes(op, keep, subsystem_batch=width) / 2.0**30
        )
        if emit is not None:
            emit(
                f"[dkx.solve] memory-bounded structured direct route: "
                f"truncated block-Thomas (keep_lowest={keep}); {because}, "
                f"solving the lowest {keep} Legendre blocks "
                f"with subsystem batch width {width} of {n_sub} "
                f"(working-set estimate {trunc_gb:.2f} GB)."
            )
        return "block_tridiagonal_truncated"

    why = sup_reason if not sup_ok else "RHS/output needs Legendre modes l >= keep"
    blocker = (
        "non-uniform Nxi_for_x rules out the full bands"
        if not uniform
        else f"full-band estimate {peak_gb:.2f} GB > budget {budget_gb_val:.1f} GB"
    )
    if emit is not None:
        emit(
            f"[dkx.solve] structured direct route unavailable: {blocker} and "
            f"truncation is invalid ({why}); using the recycled Krylov solve."
        )
    return "gmres"

def _env_size(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default

def _cpu_device_or_none() -> "jax.Device | None":
    """The first host-CPU device, or ``None`` when the CPU backend is absent
    (e.g. ``JAX_PLATFORMS=cuda`` initializes only the CUDA platform)."""
    try:
        return jax.local_devices(backend="cpu")[0]
    except RuntimeError:
        return None

def _single_device_of(arr: jnp.ndarray) -> "jax.Device | None":
    """The unique device holding ``arr``, or ``None`` (sharded/unknown)."""
    try:
        devices = arr.devices()
    except Exception:
        return None
    if len(devices) != 1:
        return None
    return next(iter(devices))

def _resolve_solve_device(
    device: "str | jax.Device | None",
    chosen: str,
    op: KineticOperator,
    traced: bool,
) -> "jax.Device | None":
    """Map the ``solve(device=...)`` knob to a target device (or ``None``).

    ``None`` means "stay put" (no array movement).  Under jit/grad tracing the
    knob is inert: arrays cannot be moved mid-trace, and the enclosing ``jit``
    already pinned the computation's devices.
    """
    if traced or chosen == "direct":  # sparse direct is a host solve already
        return None
    if device is None:
        device = os.environ.get(_SOLVE_DEVICE_ENV, "").strip().lower() or "auto"
    if isinstance(device, jax.Device):
        return device
    device = str(device).strip().lower()
    if device == "default":
        return None
    backend = jax.default_backend()
    if device == "cpu":
        if backend == "cpu":
            return None
        cpu = _cpu_device_or_none()
        if cpu is None:
            raise ValueError(
                "solve(device='cpu') requested but no CPU backend is available "
                "(JAX_PLATFORMS excludes 'cpu'?)."
            )
        return cpu
    if device in ("gpu", "cuda", "tpu", "accelerator"):
        if backend == "cpu":
            raise ValueError(
                f"solve(device={device!r}) requested but the default JAX backend is CPU "
                "(no accelerator available)."
            )
        return None
    if device != "auto":
        raise ValueError(
            f"unknown solve device {device!r}; expected 'auto', 'default', 'cpu', "
            "'gpu', or a jax.Device"
        )
    if backend == "cpu":
        return None
    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        max_size = _env_size(_SOLVE_CPU_MAX_TIER1_ENV, _SOLVE_CPU_MAX_TIER1_DEFAULT)
    else:
        max_size = _env_size(_SOLVE_CPU_MAX_TIER2_ENV, _SOLVE_CPU_MAX_TIER2_DEFAULT)
    if int(op.total_size) <= max_size:
        cpu = _cpu_device_or_none()
        if cpu is None:  # e.g. JAX_PLATFORMS=cuda: no CPU backend to route to
            return None
        print(
            f"[dkx.solve] device route: total_size={int(op.total_size)} <= "
            f"{max_size} — running this "
            f"{'structured direct' if chosen.startswith('block') else 'recycled Krylov'} "
            f"solve on the host CPU (small solves are dispatch-bound on {backend}; "
            f"override with device='default' or {_SOLVE_DEVICE_ENV})."
        )
        return cpu
    return None

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
    adjoint_residual_factor: float = DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    use_preconditioner: bool = True,
    preconditioner: str | None = None,
    drop_l_coupling_in_precond: bool = False,
    restart: int | None = None,
    recycle_dim: int = 8,
    max_restarts: int = 200,
    max_dense_size: int = 8192,
    krylov_memory_budget_gb: float | None = None,
    tier1_memory_budget_gb: float | None = None,
    direct_backend: str | None = None,
    direct_memory_budget_gb: float | None = None,
    tier1_keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    subsystem_batch: int | str = "auto",
    tier1_adjoint_window: int | None = None,
    device: str | jax.Device | None = None,
    emit: Callable[[str], None] | None = print,
    precond: tuple[Callable, Callable] | None = None,
    factors: Any = None,
    transpose: bool = False,
) -> SolveResult:
    """Solve ``K x = rhs`` with the selected solver policy.

    Policy (``method="auto"``):

    1. **structured direct** (``"block_tridiagonal"``) when
       :func:`tier1_available` — PAS/DKES family, exact direct solve,
       multi-RHS in one elimination;
    2. **recycled Krylov** (``"gmres"``) otherwise — GCROT-recycled FGMRES on
       the matrix-free operator, right-preconditioned by an exact structured
       direct solve of the Fortran-style simplified coarse operator;
    3. **sparse direct** (``"direct"``) on explicit request, or automatically
       when recycled Krylov breaches its iteration cap — host factorization of
       the materialized matrix, non-differentiable, loud. SuperLU is the
       default; MUMPS is an explicit experimental option.

    Args:
        op: the kinetic operator (:class:`dkx.drift_kinetic.KineticOperator`).
        rhs: right-hand side(s), ``(total_size,)`` or ``(total_size, n_rhs)``
            — e.g. columns of :meth:`KineticOperator.rhs` for RHSMode 2/3.
        method: ``"auto"`` | ``"block_tridiagonal"`` | ``"gmres"`` |
            ``"direct"``.  Explicit route requests raise if unsupported.
        tol: relative residual tolerance (on ``||rhs||``, per column).
        atol: absolute residual floor.
        x0: warm-start solution (recycled Krylov), same shape as ``rhs``.
        recycle: GCROT recycle pair from a previous :class:`SolveResult`
            (recycled Krylov continuation warm start).
        differentiable: wrap the solution in
            ``solvax.implicit.linear_solve`` so ``jax.grad`` flows through
            (full-state structured and recycled Krylov; sparse direct refuses).
            Generated full recovery builds its transpose from the generated
            RHS pullback; partial recovery retains the reduced-system derivative.
        check_adjoint: (differentiable full-state structured and recycled Krylov,
            default on) abort
            loudly — a ``RuntimeError`` raised from a ``jax.debug.callback`` at
            execution time — when the forward or the adjoint (transposed)
            solve misses its residual.  A stalled *adjoint* solve is
            invisible from the outside: the forward solution and every field
            of the returned :class:`SolveResult` are fine and only the
            implicit-function-theorem gradient is wrong, which is how the
            singular Fokker-Planck + ``constraintScheme=1`` system corrupts
            an optimization.  Set ``False`` for the explicitly unchecked
            path; the residuals are still recorded in
            :attr:`SolveResult.adjoint`, so a caller that opts out can
            inspect them and decide.  The check recomputes the true residual
            from the operator (``||A^T y - g||``, one extra apply plus two
            norm probes per solve) rather than trusting the Krylov method's
            internal estimate, and it never touches the solution — forward
            answers are identical either way.
        adjoint_residual_factor: how far over its residual a differentiable
            solve may land before ``check_adjoint`` calls it a failure
            (default 1). The accepted residual is
            ``factor * max(atol, tol*||b||)``. The recorded backward-error
            scale never widens this gate. Homogeneous equations require
            entrywise zero defect. Raising the factor is an explicit
            relaxation, not an observable-accuracy certificate.


    Operators with a truncated Legendre resolution (non-uniform ``Nxi_for_x``)
    are structurally singular in the rectangular state layout: the truncated
    DOFs are exact zero rows of :meth:`KineticOperator.apply` (Fortran v3
    never carries them — packed indexing in ``indices.F90``).  The recycled
    Krylov and sparse direct routes therefore solve the *pinned* system
    ``(A M + I - M) x = rhs`` with ``M`` the active-DOF projector
    (:meth:`KineticOperator.active_dof_mask`): it is nonsingular, agrees with
    ``A`` on the physical subspace, and forces
    ``x = rhs = 0`` on the truncated DOFs, so solutions, residuals, and
    implicit-function-theorem gradients all match the packed Fortran system.
    The truncated structured direct kernel is consistent with the same
    pinning: it eliminates each closed (species, x) subsystem with its own
    ``n_blocks = Nxi_for_x[ix]`` (exactly the packed Fortran system) and
    zero-pads everything above, so ramped PAS/DKES decks route through it;
    only the full structured direct factorization requires uniform
    ``Nxi_for_x``.
        use_preconditioner: recycled-Krylov preconditioner on/off (legacy
            boolean; a non-``None`` ``preconditioner`` overrides it).
        preconditioner: which recycled-Krylov preconditioner to build —

            ``"coarse"``
                the classical one (:func:`build_coarse_preconditioner`): an
                exact batched block-Thomas factorization of the
                SFINCS-simplified operator, with dense ``Ntheta*Nzeta``
                blocks.  ``O(Nxi Nspecies Nx (Ntheta Nzeta)**3)`` time and
                ``O(Nxi Nspecies Nx (Ntheta Nzeta)**2)`` memory *per solve*.
            ``"multigrid"``
                the same simplified operator, inverted approximately by the
                semicoarsened geometric multigrid V-cycle of
                :mod:`dkx.multigrid`: linear in the grid size and therefore
                affordable at angular resolutions where the cubic
                factorization is not.  It buys that affordability at the cost
                of preconditioner quality — on the measured NCSX
                full-Fokker-Planck ladder it does *not* reach the Krylov
                tolerance, and ``docs/benchmarks/performance.md`` records both the table
                and the diagnosis — so it stays opt-in.
            ``"sparse"``
                the same simplified operator, inverted *exactly* but in a
                fill-reducing elimination order
                (:mod:`dkx.sparse_precond`): the angular blocks carry the 3- or
                5-point ``createGrids.F90`` stencils, and eliminating ``L``
                first is what fills them in, so ``dkx`` assembles the operator
                in CSR and factors it with host SuperLU — which is what the
                Fortran reference does with MUMPS.  Same map as ``"coarse"`` to
                factorization round-off; it cannot run with traced operator
                leaves, since the assembly reads values on the host.
            ``"none"``
                unpreconditioned GCROT.

            ``"coupled"``
                the speed-coupled elimination of
                :mod:`dkx.structured_direct`, exact at ``l < 2`` and
                ``(species, x)``-diagonal in the tail, for Fokker-Planck and
                Sugama decks (``E_r``/drift ``l +- 2`` terms left out).

            ``"sparse_triangle"``
                sparse LU with the self-species collision operator's upper
                speed triangle retained. Same map as ``"coarse_triangle"``;
                reads each existing sparse factor once by back-substitution.

            ``None`` (the default) keeps the historical behaviour selected by
            ``use_preconditioner``, so nothing changes silently.  A
            preconditioner cannot change the answer — only the iteration
            count and the wall time.
        drop_l_coupling_in_precond: sever the L±1 coupling in the coarse
            operator.  Not Fortran's ``preconditioner_xi``, which drops L±2,
            and expensive; see :func:`dkx.coarse_precond.build_coarse_preconditioner`.
        restart: FGMRES cycle size ``m``. ``None`` (the default) selects the
            memory-aware policy: cycles of 30. A host-controlled,
            non-differentiable ``method="auto"`` solve with freshly built
            factors runs five of them, then two cycles of 100, then continues
            from its iterate at the longest restart, at most 1,000 and at most
            ``total_size``, whose
            GCROT basis (``restart * total_size * 8`` bytes: DKX's fixed
            preconditioners let SOLVAX keep only ``V``) fits ``krylov_memory_budget_gb``.
            Probe and wide cycles together use at most ``30 * max_restarts``
            inner steps per RHS (the remainder rounded down to whole wide
            cycles), and the escalation ladder's larger-budget rung uses the
            same wide restart. An integer fixes the cycle size everywhere;
            explicit methods, traced and differentiable solves and
            caller-supplied preconditioners always use a fixed size.
        recycle_dim: GCROT recycle directions ``k``.
        max_restarts: recycled-Krylov outer-cycle cap (what makes ``auto``
            fall through to the sparse direct route).
        max_dense_size: sparse direct materialization guard.
        krylov_memory_budget_gb: memory (GB, ``2**30`` bytes) the widened
            FGMRES basis of the ``restart=None`` policy may take. ``None`` reads
            the ``DKX_KRYLOV_MEMORY_BUDGET_GB`` environment variable, else a
            quarter of the memory available when the solve widens (256 MiB
            where available memory cannot be read). It bounds the basis
            estimate, not the solver's total memory. Ignored with an integer
            ``restart``.
        tier1_memory_budget_gb: budget (GB) above which ``method="auto"``
            prefers the memory-lean truncated structured direct kernel over
            the full-band factorization.  ``None`` reads the ``DKX_TIER1_MEMORY_BUDGET_GB``
            environment variable, else the 8 GB default.  The full-band peak is
            estimated by :func:`tier1_peak_memory_bytes`.
        direct_backend: sparse-direct factor backend. ``None`` preserves the
            released default, SuperLU; ``"mumps"`` explicitly selects the
            optional SOLVAX MUMPS adapter. Stored factors retain their backend.
        direct_memory_budget_gb: process budget for an explicit MUMPS backend.
            DKX subtracts measured current process RSS (or a conservative
            process high-water fallback), projected COO, RHS-shaped buffers and
            runtime headroom, intersects that remainder with current host
            availability, and passes only the result to MUMPS. Ignored by SuperLU.
        tier1_keep_lowest: number of Legendre blocks the truncated structured
            direct kernel computes exactly (default 3 — the RHSMode 1/2/3 drives and
            output moments live on ``l <= 2``).
        subsystem_batch: how many of the ``B = n_species * n_x`` independent
            (species, x) subsystems the truncated structured direct kernel
            eliminates concurrently.  ``"auto"`` (default) is backend-aware: width 1 on
            the CPU backend (XLA:CPU runs batched LAPACK factor/solve calls
            serially per batch element, so wider sweeps measure
            neutral-to-slower — see :func:`_resolve_subsystem_batch`), and on
            accelerators the widest width whose modeled footprint
            (:func:`tier1_truncated_peak_memory_bytes`) fits the memory
            budget (:func:`tier1_truncated_subsystem_width`).  An integer
            fixes the width (clamped to ``[1, B]``; 1 is the fully serial
            minimum-memory sweep).  Any width computes identical
            per-subsystem arithmetic — the knob trades memory for batched
            parallel work.  Ignored by the non-truncated routes.
        tier1_adjoint_window: opt-in bounded reverse mode for the truncated
            structured direct kernel.  ``None`` (default)
            keeps the taped gradient — bit-identical behavior to previous
            releases.  An integer ``w`` selects solvax's structure-preserving
            custom VJP: ``jax.grad`` through the truncated solve then runs at
            ``O((keep + w) m^2)`` memory per (species, x) subsystem instead of
            taping the full ``Nxi`` sweep, with an exact right-hand-side
            gradient and ``O(rho^{2w})`` coefficient-gradient error;
            ``w >= Nxi`` reproduces the taped gradient exactly.

            **Reverse mode only.**  The bounded path is a ``jax.custom_vjp``,
            and JAX cannot push forward-mode autodiff through one, so
            ``jax.jacfwd`` and ``jax.jvp`` over a solve that sets this option
            raise ``TypeError: can't apply forward-mode autodiff (jvp) to a
            custom_vjp function`` rather than falling back.  That matters here
            because :func:`dkx.sensitivity.jvp_flux` is forward mode: a
            sensitivity study that mixes the two must leave
            ``tier1_adjoint_window=None`` on the forward-mode calls.  Choosing
            the window is therefore a per-call-site decision, not a global
            switch.  See :func:`_solve_tier1_truncated`.
        device: where to run the solve.  ``"cpu"``/``"gpu"`` force a backend
            and a ``jax.Device`` pins the solve to that device: inputs are
            moved with ``jax.device_put`` and the solution is returned on
            the device that held ``rhs``.  ``"auto"`` (the default, also
            read from the ``DKX_SOLVE_DEVICE`` environment variable)
            additionally routes solves at or below the
            ``DKX_SOLVE_CPU_MAX_SIZE_TIER1`` / ``_TIER2`` thresholds
            to the host CPU on accelerator-default hosts — but both
            thresholds default to 0 (no routing), because the same-host
            measurements in docs/benchmarks/performance.md found the GPU faster at
            every practical size; the knobs exist for hosts where that
            balance differs.  ``"default"`` disables all movement.  Under
            ``jit``/``grad`` tracing the knob is inert (arrays cannot move
            mid-trace), so jitted callers are unaffected.
        emit: route-status sink. The default prints one physical route
            description; ``None`` suppresses routine route presentation while
            retaining exceptions and convergence failures.
        factors: a factorization a previous solve of this operator returned in
            :attr:`SolveResult.factors` --- a :class:`Tier1Solver` on the
            structured direct route, a :class:`DirectFactors` on the sparse
            one.  The route then skips building one and the solve is a
            triangular substitution, which is what makes a transport matrix,
            an adjoint and a line search cost solves rather than
            factorizations.  Only the two direct routes take it; the recycled
            Krylov route's reusable object is its preconditioner
            (``precond``), which is a different thing and is reused on
            different terms.

            Factors of a *different* operator are accepted deliberately: they
            are then an approximate inverse, and the route corrects the defect
            against the operator actually passed in.  What makes that bounded
            rather than a gamble is the check that follows: the original
            residual is recomputed, and a solve that misses its tolerance
            factorizes once and repeats, at a cost of one factorization and a
            few applies.  Stale factors have been measured to cost a Krylov
            solve 6,000 iterations before it gave up
            (2026-09-07 recycling-and-preconditioner-reuse study);
            this route cannot, because it never iterates.  Reuse across
            operators is a host decision --- nothing here refreshes on its own,
            and no default changes.
        transpose: solve ``A^T x = rhs`` instead of ``A x = rhs``, on the two
            direct routes, from the same factors.  This is the adjoint of
            §5.2: with ``factors`` from the primal, a gradient's linear algebra
            is one substitution rather than a second factorization.  Not
            available with ``differentiable=True`` --- a differentiable solve
            builds its own transposed solve through
            ``solvax.implicit.linear_solve`` and the two must not be nested.

    Auto-policy structured routing (``method="auto"``, when
    :func:`tier1_available` is true):

    ================================  =============================================
    condition                         route
    ================================  =============================================
    uniform, peak estimate <= budget  full ``"block_tridiagonal"`` (any output
                                      mode, multi-RHS factor reuse)
    ramped Nxi_for_x or estimate >    ``"block_tridiagonal_truncated"`` when the
    budget                            truncation is valid (lowest ``keep`` blocks
                                      only, ~O(keep m^2) memory; ramps solved with
                                      per-subsystem ``n_blocks = Nxi_for_x[ix]``)
    …and truncation invalid           ``"gcrot"`` recycled Krylov, with a
                                      printed notice (high-l output the
                                      truncation cannot supply)
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
    # "direct" and "iterative" are the names to reach for: they say what the
    # route does rather than how it is built.  The implementation names stay
    # accepted because the benchmarks and the solver trace speak them.
    method = {"iterative": "gmres"}.get(method, method)
    if method not in {
        "auto",
        "block_tridiagonal",
        "block_tridiagonal_truncated",
        "gmres",
        "direct",
    }:
        raise ValueError(
            f"unknown method {method!r}; use 'auto' (default), 'direct', or 'iterative'"
        )
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and nonnegative; got {tol!r}")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be finite and nonnegative; got {atol!r}")
    if direct_backend is not None:
        direct_backend = str(direct_backend).strip().lower()
        if direct_backend not in {"superlu", "mumps"}:
            raise ValueError("direct_backend must be None, 'superlu', or 'mumps'")
    auto_restart = restart is None
    if auto_restart:
        restart = _AUTO_RESTART_PROBE
    elif isinstance(restart, (bool, np.bool_)) or int(restart) != restart or restart < 1:
        raise ValueError(f"restart must be None or a positive integer; got {restart!r}")
    restart = int(restart)
    if krylov_memory_budget_gb is not None:
        _krylov_basis_budget_bytes(krylov_memory_budget_gb)  # validate early
    if direct_memory_budget_gb is not None:
        if (isinstance(direct_memory_budget_gb, (bool, np.bool_))
                or not np.isfinite(direct_memory_budget_gb)
                or direct_memory_budget_gb <= 0.0):
            raise ValueError("direct_memory_budget_gb must be finite and positive")
    if method == "block_tridiagonal_truncated":
        ok, reason = tier1_available(op)
        if not ok:
            raise NotImplementedError(
                f"truncated structured direct route unavailable: {reason}"
            )
        keep = min(tier1_keep_lowest, op.n_xi)
        sup_ok, sup_reason = _truncation_supported(op, keep)
        if not sup_ok:
            raise NotImplementedError(
                f"truncated structured direct route unavailable: {sup_reason}"
            )
    if transpose and differentiable:
        raise ValueError(
            "transpose=True is the explicit adjoint on stored factors; a "
            "differentiable solve already builds its own transposed solve, so "
            "pass one or the other"
        )
    if (factors is not None or transpose) and method in {"gmres", "iterative"}:
        raise ValueError(
            "stored factors and the explicit transpose belong to the direct "
            "routes; the recycled Krylov route reuses a preconditioner "
            "instead (precond=)"
        )
    require_float64()
    rhs2d, squeeze = _as_columns(rhs)
    if rhs2d.shape[0] != op.total_size:
        raise ValueError(
            f"rhs has {rhs2d.shape[0]} rows; operator expects {op.total_size}"
        )

    chosen = method
    if method == "auto" and factors is not None:
        # A factorization in hand names its own route: the policy that would
        # otherwise choose one cannot pick a route that then ignores it.
        chosen = (
            "block_tridiagonal"
            if isinstance(factors, (Tier1Solver, CoupledSolver))
            else "direct"
        )
    elif method == "auto":
        chosen = _auto_route(
            op,
            rhs2d,
            tier1_memory_budget_gb,
            tier1_keep_lowest,
            subsystem_batch,
            emit,
        )
    if factors is not None and not isinstance(
        factors, (Tier1Solver, CoupledSolver) if chosen == "block_tridiagonal" else DirectFactors
    ):
        raise TypeError(
            f"the {chosen!r} route factors a "
            f"{'Tier1Solver' if chosen == 'block_tridiagonal' else 'DirectFactors'}"
            f"; it was handed a {type(factors).__name__}"
        )
    if transpose and chosen not in ("block_tridiagonal", "direct"):
        raise NotImplementedError(
            f"the {chosen!r} route has no transposed solve on stored factors"
        )

    target_device = _resolve_solve_device(
        device, chosen, op, _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    )
    home_device: jax.Device | None = None
    if target_device is not None:
        home = _single_device_of(rhs2d)
        if home is not None and home != target_device:
            home_device = home
            op = jax.device_put(op, target_device)
            rhs2d = jax.device_put(rhs2d, target_device)
            if x0 is not None:
                x0 = jax.device_put(x0, target_device)
            if recycle is not None:
                recycle = jax.device_put(recycle, target_device)

    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        if chosen == "block_tridiagonal":
            result = _solve_tier1(
                op, rhs2d, tol=tol, atol=atol, differentiable=differentiable,
                check_adjoint=check_adjoint, adjoint_residual_factor=adjoint_residual_factor,
                factors=factors, transpose=transpose, emit=emit,
            )
        else:
            keep = min(tier1_keep_lowest, op.n_xi)
            if differentiable and keep == op.n_xi:
                result = _solve_generated_full(
                    op, rhs2d, tol=tol, atol=atol, subsystem_batch=subsystem_batch,
                    adjoint_window=tier1_adjoint_window, check_adjoint=check_adjoint,
                    adjoint_residual_factor=adjoint_residual_factor,
                    memory_budget_gb=tier1_memory_budget_gb,
                )
            else:
                result = _solve_tier1_truncated(
                    op, rhs2d, keep=keep, tol=tol, atol=atol,
                    subsystem_batch=subsystem_batch, adjoint_window=tier1_adjoint_window,
                )
                if not (result.converged or differentiable or keep == op.n_xi):
                    # One defect correction before the dense Krylov fallback: at low
                    # collisionality ||A|| ||x|| >> ||b|| leaves the elimination's
                    # rounding just above tol. Rows l < keep-1 see only solved blocks,
                    # so their residual is exact; row keep-1 belongs to the reduced system.
                    def low_residual(x):
                        r = rhs2d - jax.vmap(op.apply, 1, 1)(x)
                        f = r[: op.f_size].reshape(*op.f_shape, -1).at[:, :, keep - 1 :].set(0.0)
                        return r.at[: op.f_size].set(f.reshape(op.f_size, -1))
                    x = result.x + _solve_tier1_truncated(
                        op, low_residual(result.x), keep=keep, tol=tol, atol=atol,
                        subsystem_batch=subsystem_batch).x
                    res = jnp.linalg.norm(low_residual(x), axis=0)
                    result = replace(result, x=x, residual_norms=res,
                                     converged=_converged_flag(res, rhs2d, tol, atol))
        if method == "auto" and not result.converged and not differentiable:
            # Structured elimination has no pivoting across blocks: on
            # near-singular systems (e.g. a nu_n=0 collisionless deck, whose
            # bordered constraint leaves the operator with condition numbers
            # ~1e18) its residual can miss the tolerance even though the
            # system is consistent.  Mirror the recycled-Krylov -> sparse-direct
            # pattern and fall through to the preconditioned Krylov route.
            print(
                "[dkx.solve] the structured direct solve missed the "
                f"tolerance (residuals={np.asarray(result.residual_norms)}); "
                "falling back to the recycled Krylov route."
            )
            chosen = "gmres"
    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        pass  # structured direct result stands.
    elif chosen == "gmres":
        result = _solve_tier2(
            op,
            rhs2d,
            tol=tol,
            atol=atol,
            x0=x0,
            recycle=recycle,
            preconditioner=(
                _auto_preconditioner(op, krylov_memory_budget_gb)
                if method == "auto" and preconditioner is None and use_preconditioner
                and not differentiable and precond is None
                else _resolve_preconditioner(preconditioner, use_preconditioner)
            ),
            prebuilt_precond=precond,
            auto_restart_recovery=method == "auto" and not differentiable and auto_restart,
            krylov_memory_budget_gb=krylov_memory_budget_gb,
            drop_l_coupling_in_precond=drop_l_coupling_in_precond,
            restart=restart,
            recycle_dim=recycle_dim,
            max_restarts=max_restarts,
            differentiable=differentiable,
            check_adjoint=check_adjoint,
            adjoint_residual_factor=adjoint_residual_factor,
        )
        if method == "auto" and not result.converged and not differentiable:
            requested = _resolve_preconditioner(preconditioner, use_preconditioner)
            result = _escalate_after_tier2_stall(
                op,
                rhs2d,
                stalled=result,
                tol=tol,
                atol=atol,
                x0=x0,
                recycle=recycle,
                preconditioner=requested,
                drop_l_coupling_in_precond=drop_l_coupling_in_precond,
                restart=restart,
                recycle_dim=recycle_dim,
                max_restarts=max_restarts,
                check_adjoint=check_adjoint,
                adjoint_residual_factor=adjoint_residual_factor,
                max_dense_size=max_dense_size,
                direct_backend=direct_backend,
                direct_memory_budget_gb=direct_memory_budget_gb,
                auto_restart=auto_restart,
                krylov_memory_budget_gb=krylov_memory_budget_gb,
            )
    else:  # direct
        if differentiable:
            raise RuntimeError(
                "the sparse direct route (method='direct') is non-differentiable."
            )
        result = _solve_tier3(
            op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size,
            direct_backend=direct_backend,
            direct_memory_budget_gb=direct_memory_budget_gb,
            factors=factors, transpose=transpose, emit=emit,
        )

    if home_device is not None:
        # Return the solution (and warm-start state) on the device that held
        # the inputs, so downstream pipelines are unaffected by the routing.
        result = replace(
            result,
            x=jax.device_put(result.x, home_device),
            residual_norms=jax.device_put(result.residual_norms, home_device),
            recycle=(
                None
                if result.recycle is None
                else jax.device_put(result.recycle, home_device)
            ),
        )

    if squeeze:
        result = replace(result, x=result.x[:, 0])
    return result
