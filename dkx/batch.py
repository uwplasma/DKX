"""Batched multi-surface / multi-``E_r`` kinetic solves (``jax.vmap`` productized).

Single ``dkx`` solves are CPU/GPU parity on the direct tier; the GPU win
comes from **batching** many solves that share a discretization but differ in
their physics leaves.  Because :class:`dkx.drift_kinetic.KineticOperator`
is a registered, jit-safe pytree, ``jax.vmap`` maps cleanly over its varying
leaves, and the implicit-differentiation solve composes with the batch axis.

This module turns that into a first-class API around the two canonical batch
axes:

* an **``E_r`` scan** — a vector of radial-electric-field values on one geometry
  (:func:`batched_er_scan`); only the two ExB/Er drive scalars vary; and
* a **surface scan** — a batch of geometries sharing grids/layout
  (:func:`batched_surface_scan`); the geometry, species, collision and drive
  leaves vary while the discretization leaves stay shared.

Both are thin builders over the primitive :func:`batched_solve`, which
``jax.vmap``s a solve-plus-moments over an explicit mapping of the varying
operator leaves.  Peak memory is bounded automatically: the per-solve footprint
comes from the tier-1 memory model in :mod:`dkx.solve`
(:func:`dkx.solve.tier1_peak_memory_bytes`), the memory budget from the
device/host, and the batch is processed in ``jax.lax.map`` chunks of the
computed size so only one chunk's intermediates are ever live.

Design constraints honoured here:

* reuse :func:`dkx.solve.solve` and the operator read-only (no new solver
  code, no operator surgery beyond ``dataclasses.replace`` of leaves);
* differentiable end to end (``jax.grad`` of a scalar over the batch flows
  through the implicit solve) and jit-safe (``jax.jit(batched_solve)`` compiles);
* automatic memory budgeting with only a simple optional ``max_batch`` /
  ``memory_budget_gb`` override — no advanced user knobs, no env-var switches.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from .drift_kinetic import KineticOperator
from .solve import solve, tier1_peak_memory_bytes

# ---------------------------------------------------------------------------
# Leaf classification and memory-budget defaults
# ---------------------------------------------------------------------------

# Grid / derivative-matrix leaves that define the discretization: they set the
# per-solve memory footprint and are read host-side by the ``solve`` auto-router
# (``tier1_peak_memory_bytes`` / ``_auto_route``).  They MUST stay concrete
# (shared, un-vmapped) so the router and the footprint estimate see real values,
# so batching any of them is rejected and a surface scan keeps them from the
# template operator (and requires every surface to agree on them).
_DISCRETIZATION_FIELDS = frozenset(
    {
        "x",
        "x_weights",
        "ddx",
        "ddtheta",
        "ddzeta",
        "theta_weights",
        "zeta_weights",
        "n_xi_for_x",
        "xi_coupling_lower",
        "xi_coupling_upper",
        "ddx_xdot_plus",
        "ddx_xdot_minus",
        "ddtheta_magdrift_plus",
        "ddtheta_magdrift_minus",
        "ddzeta_magdrift_plus",
        "ddzeta_magdrift_minus",
    }
)

# Fraction of the resolved device/host memory used as the default budget: the
# tier-1 footprint is an estimate and the runtime keeps other buffers live, so a
# margin below the hard limit keeps the auto-chunked batch safely resident.
_DEFAULT_BUDGET_FRACTION = 0.8

# Budget floor when neither the device nor the host expose a memory size.
_FALLBACK_BUDGET_GB = 8.0

# Footprint floor: tiny reduced operators estimate at a few MB, and dividing the
# budget by an unrealistically small footprint would pick an enormous chunk that
# ignores the runtime's fixed overhead.  Clamp so the chunk stays sane.
_MIN_FOOTPRINT_BYTES = 64.0 * 2.0**20  # 64 MB

_BYTES_PER_GB = 2.0**30


# ---------------------------------------------------------------------------
# Result container (registered pytree so ``jax.jit(batched_solve)`` may return it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchedSolveResult:
    """Outcome of a batched solve.

    Attributes:
        states: solved state vectors, shape ``(batch, total_size)``.
        moments: the per-element RHSMode-1 moment table
            (:func:`dkx.run.profile_moments_from_operator`), every entry
            with a leading batch axis, e.g. ``particleFlux_vm_psiHat`` shape
            ``(batch, n_species)`` and ``FSABjHat`` shape ``(batch,)``.
        chunk_size: the memory-budgeted ``jax.lax.map`` chunk size actually used.
        n_chunks: number of chunks the batch was processed in
            (``ceil(batch / chunk_size)``).
        method: the ``solve_method`` requested for every element.
        radial_current: ``J_r = sum_a Z_a Gamma_a`` per element, shape
            ``(batch,)`` — populated by :func:`batched_er_scan`, ``None``
            otherwise.
    """

    states: jnp.ndarray
    moments: Mapping[str, jnp.ndarray]
    chunk_size: int
    n_chunks: int
    method: str
    radial_current: jnp.ndarray | None = None


# ``states``/``moments``/``radial_current`` are the traced arrays; the chunking
# metadata and method string are static aux so the whole result is a valid
# ``jax.jit`` return value.
jax.tree_util.register_dataclass(
    BatchedSolveResult,
    data_fields=["states", "moments", "radial_current"],
    meta_fields=["chunk_size", "n_chunks", "method"],
)


# ---------------------------------------------------------------------------
# Memory model: per-solve footprint, budget resolution, and chunk size
# ---------------------------------------------------------------------------


def solve_footprint_bytes(op: KineticOperator) -> float:
    """Estimated peak bytes of one solve of ``op`` (tier-1 memory model).

    Reuses :func:`dkx.solve.tier1_peak_memory_bytes` — the full-band
    factorization peak the tier-1 auto-route sizes against — as a conservative
    per-solve upper bound, adding the solved state vector, and clamped to a
    small floor so tiny reduced operators do not imply an unrealistically large
    chunk (:data:`_MIN_FOOTPRINT_BYTES`).
    """
    factor_bytes = float(tier1_peak_memory_bytes(op))
    state_bytes = float(op.total_size) * 8.0
    return max(factor_bytes + state_bytes, _MIN_FOOTPRINT_BYTES)


def _device_memory_bytes() -> float | None:
    """Device memory limit (bytes) if the backend exposes it, else ``None``."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 - backends without memory_stats
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    return float(limit) if limit else None


def _host_memory_bytes() -> float | None:
    """Total physical host RAM (bytes) via ``os.sysconf``, else ``None``."""
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def resolve_memory_budget_bytes(memory_budget_gb: float | None = None) -> float:
    """Resolve the batch memory budget in bytes.

    An explicit ``memory_budget_gb`` is taken verbatim; otherwise a fraction
    (:data:`_DEFAULT_BUDGET_FRACTION`) of the device memory limit is used, then
    the host RAM, then a fixed fallback (:data:`_FALLBACK_BUDGET_GB`).
    """
    if memory_budget_gb is not None:
        return float(memory_budget_gb) * _BYTES_PER_GB
    total = _device_memory_bytes()
    if total is None:
        total = _host_memory_bytes()
    if total is None:
        total = _FALLBACK_BUDGET_GB * _BYTES_PER_GB
    return _DEFAULT_BUDGET_FRACTION * total


def auto_chunk_size(
    op: KineticOperator,
    batch: int,
    *,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
) -> int:
    """Largest chunk whose peak stays within the memory budget (>= 1, <= batch).

    ``chunk = floor(budget / per_solve_footprint)`` from
    :func:`resolve_memory_budget_bytes` and :func:`solve_footprint_bytes`, then
    clamped by the optional ``max_batch`` override and the batch length.
    """
    if batch <= 0:
        raise ValueError("batch must be a positive integer.")
    budget = resolve_memory_budget_bytes(memory_budget_gb)
    per_solve = solve_footprint_bytes(op)
    chunk = int(budget // per_solve)
    chunk = max(1, chunk)
    if max_batch is not None:
        if int(max_batch) < 1:
            raise ValueError("max_batch must be a positive integer.")
        chunk = min(chunk, int(max_batch))
    return min(chunk, int(batch))


# ---------------------------------------------------------------------------
# Batch-leaf validation / helpers
# ---------------------------------------------------------------------------


def _batch_size_of(batch_leaves: Mapping[str, Any]) -> int:
    """The shared leading-axis length of every leaf in ``batch_leaves``."""
    leaves = jax.tree_util.tree_leaves(batch_leaves)
    if not leaves:
        raise ValueError("batch_leaves is empty; nothing to batch over.")
    sizes = {int(jnp.asarray(leaf).shape[0]) for leaf in leaves if jnp.ndim(leaf) >= 1}
    if not sizes:
        raise ValueError("batch_leaves has no array leaves with a batch axis.")
    if len(sizes) != 1:
        raise ValueError(
            f"batch_leaves must share one leading batch axis; got sizes {sorted(sizes)}."
        )
    return sizes.pop()


def _validate_batch_leaves(op: KineticOperator, batch_leaves: Mapping[str, Any]) -> None:
    """Reject unknown fields and any discretization leaf (must stay shared)."""
    if not isinstance(batch_leaves, Mapping):
        raise TypeError("batch_leaves must be a mapping of operator field name -> array.")
    valid = set(KineticOperator._CHILD_FIELDS)
    for name in batch_leaves:
        if name not in valid:
            raise ValueError(
                f"batch_leaf {name!r} is not a batchable KineticOperator leaf; "
                f"valid leaves are {sorted(valid - _DISCRETIZATION_FIELDS)}."
            )
        if name in _DISCRETIZATION_FIELDS:
            raise ValueError(
                f"batch_leaf {name!r} is a discretization leaf and must stay shared "
                "across the batch (it sets the per-solve footprint and the solver "
                "route); batch geometries/species/drives, not grids."
            )


# ---------------------------------------------------------------------------
# The primitive: vmap a solve-plus-moments over varying operator leaves
# ---------------------------------------------------------------------------


def batched_solve(
    op: KineticOperator,
    batch_leaves: Mapping[str, Any],
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    ntv_kernel_tz: Any | None = None,
) -> BatchedSolveResult:
    """Solve a batch of kinetic problems sharing ``op``'s discretization.

    Every batch element is ``op`` with the leaves in ``batch_leaves`` replaced
    by their per-element slice (``dataclasses.replace``); the element is solved
    with :func:`dkx.solve.solve` and reduced to the RHSMode-1 moment
    table (:func:`dkx.run.profile_moments_from_operator`).  The map is a
    ``jax.vmap`` executed in memory-budgeted ``jax.lax.map`` chunks, so it is
    differentiable (the implicit solve composes with the batch axis), jit-safe,
    and bounded in peak memory.

    Only ``op``'s varying leaves are mapped; every other leaf — crucially the
    discretization grids — is closed over and stays concrete, which keeps the
    host-side solver auto-route and the footprint estimate well defined.

    Args:
        op: the base operator (defines grids, layout, and the shared leaves).
        batch_leaves: mapping ``field name -> batched value`` for the leaves
            that vary; each value carries a leading batch axis of the same
            length.  Field names must be batchable
            :class:`~dkx.drift_kinetic.KineticOperator` leaves (no
            discretization leaf; see :data:`_DISCRETIZATION_FIELDS`).
        solve_method: forwarded to :func:`dkx.solve.solve` (``"auto"``
            routes host-side on the shared ``op`` — well defined because the
            discretization is concrete).
        tol: relative residual tolerance forwarded to the solve.
        differentiable: use the implicit-differentiation solve so ``jax.grad``
            of a scalar over the batch flows through each element.
        max_batch: optional hard cap on the chunk size.
        memory_budget_gb: optional memory budget override (GB); defaults to a
            fraction of the device/host memory (:func:`resolve_memory_budget_bytes`).
        ntv_kernel_tz: optional NTV geometric kernel forwarded to the moment
            table (see :func:`dkx.run.profile_moments_from_operator`).

    Returns:
        A :class:`BatchedSolveResult` with batched ``states`` and ``moments``.
    """
    from .run import profile_moments_from_operator  # noqa: PLC0415 - heavy run stack

    _validate_batch_leaves(op, batch_leaves)
    batch = _batch_size_of(batch_leaves)
    chunk = auto_chunk_size(
        op, batch, max_batch=max_batch, memory_budget_gb=memory_budget_gb
    )

    def solve_one(leaves: Mapping[str, Any]):
        op_i = dataclasses.replace(op, **leaves)
        result = solve(
            op_i,
            op_i.rhs(),
            method=solve_method,
            tol=tol,
            differentiable=differentiable,
        )
        state = jnp.reshape(result.x, (-1,))
        moments = profile_moments_from_operator(op_i, state, ntv_kernel_tz=ntv_kernel_tz)
        return state, dict(moments)

    states, moments = jax.lax.map(solve_one, dict(batch_leaves), batch_size=chunk)
    n_chunks = -(-batch // chunk)
    return BatchedSolveResult(
        states=states,
        moments=dict(moments),
        chunk_size=int(chunk),
        n_chunks=int(n_chunks),
        method=str(solve_method),
    )


# ---------------------------------------------------------------------------
# Canonical axis 1: an E_r scan on one geometry
# ---------------------------------------------------------------------------


def batched_er_scan(
    problem: Any,
    er_values: Any,
    *,
    solve_method: str | None = None,
    tol: float | None = None,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
) -> BatchedSolveResult:
    """Batched radial current / moments over a vector of ``E_r`` on one geometry.

    Each ``E_r`` sets both ExB/Er drive scalars to ``dphi_per_er * E_r`` (the
    ``ambipolarSolver.F90`` ``updateEr`` relation, exactly
    :func:`dkx.er.operator_at_er`); every other leaf — geometry, species,
    collisions, grids — is shared, so the scan is one vmapped solve over the
    single varying drive.  The result's ``radial_current`` is
    ``J_r = sum_a Z_a Gamma_a`` per ``E_r``.

    Args:
        problem: a prepared :class:`dkx.er.ErProblem`
            (:func:`dkx.er.prepare`) — carries the base operator with the
            ExB/Er term flags active, the ``dphi_per_er`` conversion, and ``z_s``.
        er_values: the ``E_r`` scan values, shape ``(batch,)``.
        solve_method, tol: forwarded to the solve (default to the problem's).
        differentiable: differentiable implicit solves (for ``jax.grad``).
        max_batch, memory_budget_gb: memory-budgeting overrides
            (:func:`auto_chunk_size`).

    Returns:
        A :class:`BatchedSolveResult` with ``radial_current`` populated.
    """
    er = jnp.asarray(er_values, dtype=jnp.float64).reshape((-1,))
    dphi = jnp.asarray(problem.dphi_per_er, dtype=jnp.float64) * er
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}

    result = batched_solve(
        problem.operator,
        batch_leaves,
        solve_method=solve_method or problem.solve_method,
        tol=tol if tol is not None else problem.tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
    )
    z_s = jnp.asarray(problem.z_s, dtype=jnp.float64)
    gamma = jnp.asarray(result.moments["particleFlux_vm_psiHat"])  # (batch, n_species)
    j_r = gamma @ z_s  # (batch,)
    return dataclasses.replace(result, radial_current=j_r)


# ---------------------------------------------------------------------------
# Canonical axis 2: a batch of flux surfaces sharing discretization
# ---------------------------------------------------------------------------


def _shared_structure(op: KineticOperator) -> Any:
    """The pytree structure (aux + None layout) that batched surfaces must share."""
    return jax.tree_util.tree_structure(op)


def _stack_varying_leaves(operators: Sequence[KineticOperator]) -> dict[str, Any]:
    """Stack every non-discretization, non-``None`` child leaf across surfaces.

    Discretization leaves stay shared (kept from the template and validated
    equal by :func:`_check_shared_discretization`); ``None`` leaves are shared
    structurally.  Every other leaf — geometry, species, drive scalars, and the
    collision sub-pytrees — is stacked with a leading batch axis, tree-aware so
    the ``pas``/``fp`` operator pytrees stack leaf by leaf.
    """
    template = operators[0]
    batch_leaves: dict[str, Any] = {}
    for name in KineticOperator._CHILD_FIELDS:
        if name in _DISCRETIZATION_FIELDS or getattr(template, name) is None:
            continue
        values = [getattr(o, name) for o in operators]
        batch_leaves[name] = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *values
        )
    return batch_leaves


def _check_shared_discretization(operators: Sequence[KineticOperator]) -> None:
    """Require every surface to share structure and discretization leaves."""
    import numpy as np  # noqa: PLC0415 - host-only equality check

    template = operators[0]
    ref_struct = _shared_structure(template)
    for idx, op in enumerate(operators[1:], start=1):
        if _shared_structure(op) != ref_struct:
            raise ValueError(
                f"surface {idx} has a different operator structure (aux flags / "
                "None layout) than surface 0; a batch must share discretization "
                "and configuration."
            )
        for name in _DISCRETIZATION_FIELDS:
            ref = getattr(template, name)
            if ref is None:
                continue
            if not np.array_equal(np.asarray(ref), np.asarray(getattr(op, name))):
                raise ValueError(
                    f"surface {idx} differs from surface 0 in discretization leaf "
                    f"{name!r}; batched surfaces must share grids/derivative "
                    "matrices (only geometry/species/drive leaves may vary)."
                )


def batched_surface_scan(
    operators: Sequence[KineticOperator],
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    ntv_kernel_tz: Any | None = None,
) -> BatchedSolveResult:
    """Batched solve / moments over a sequence of flux-surface operators.

    The surfaces share discretization (grids, derivative matrices, layout, and
    option flags) but differ in geometry, species, collision and drive leaves.
    Those varying leaves are stacked (:func:`_stack_varying_leaves`) and mapped
    with :func:`batched_solve`; the shared discretization is validated equal
    across surfaces (:func:`_check_shared_discretization`).

    Args:
        operators: the per-surface
            :class:`~dkx.drift_kinetic.KineticOperator` objects (e.g. one
            per radial location, each built from that surface's geometry).
        solve_method, tol, differentiable: forwarded to :func:`batched_solve`.
        max_batch, memory_budget_gb: memory-budgeting overrides.
        ntv_kernel_tz: optional NTV kernel forwarded to the moment table.

    Returns:
        A :class:`BatchedSolveResult` with batched ``states`` and ``moments``.
    """
    ops = list(operators)
    if not ops:
        raise ValueError("batched_surface_scan requires at least one operator.")
    _check_shared_discretization(ops)
    batch_leaves = _stack_varying_leaves(ops)
    return batched_solve(
        ops[0],
        batch_leaves,
        solve_method=solve_method,
        tol=tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
        ntv_kernel_tz=ntv_kernel_tz,
    )


__all__ = [
    "BatchedSolveResult",
    "auto_chunk_size",
    "batched_er_scan",
    "batched_solve",
    "batched_surface_scan",
    "resolve_memory_budget_bytes",
    "solve_footprint_bytes",
]
