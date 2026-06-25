"""Device-compatible residual deflation for RHSMode=1 QI hard seeds.

The QI hard-seed failures seen so far are not storage failures: the main
missing piece is a small residual-reducing subspace that captures global
coupling left by the local x/species smoother.  This module implements a
bounded, JAX-array deflation primitive inspired by recycling/deflated Krylov
methods:

``M^{-1} r = S^{-1} r + Z c``

where ``S^{-1}`` is the local smoother, ``Z`` is a rank-gated basis generated
from the preconditioned residual Krylov sequence, and ``c`` solves

``min ||A Z c - (r - A S^{-1} r)||``.

The builder is intentionally driver-independent.  A future driver hook can
close over the returned JAX arrays inside a device Krylov solve, while the
fail-closed probe here ensures no candidate is promoted unless the true
physical residual decreases by the requested margin.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.qi.coarse import RHS1QICoarseBasis, orthonormalize_rhs1_qi_coarse_basis

ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QIDeflationMetadata:
    """Diagnostics for a residual-deflated QI preconditioner candidate."""

    total_size: int
    candidate_count: int
    rank: int
    discarded_count: int
    requested_krylov_depth: int
    basis_shape: tuple[int, int]
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    stable_rank: float
    condition_estimate: float
    min_singular_value: float
    max_singular_value: float
    regularization_rcond: float
    damping: float
    correction_cycles: int
    composition: str
    accepted_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    device_resident: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly metadata for solver traces."""

        return {
            "total_size": int(self.total_size),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "discarded_count": int(self.discarded_count),
            "requested_krylov_depth": int(self.requested_krylov_depth),
            "basis_shape": tuple(int(v) for v in self.basis_shape),
            "operator_on_basis_shape": tuple(int(v) for v in self.operator_on_basis_shape),
            "operator_on_basis_norm": float(self.operator_on_basis_norm),
            "stable_rank": float(self.stable_rank),
            "condition_estimate": float(self.condition_estimate),
            "min_singular_value": float(self.min_singular_value),
            "max_singular_value": float(self.max_singular_value),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
            "correction_cycles": int(self.correction_cycles),
            "composition": self.composition,
            "accepted_labels": self.accepted_labels,
            "candidate_labels": self.candidate_labels,
            "device_resident": bool(self.device_resident),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIDeflatedPreconditioner:
    """Reusable local-smoother plus deflated coarse-correction action."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    metadata: RHS1QIDeflationMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Return least-squares coarse coefficients for a residual."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        return _regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def _apply_one_cycle(self, residual: ArrayLike) -> ArrayLike:
        """Apply one fixed-basis local-plus-deflated correction cycle."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return float(self.metadata.damping) * local
        if self.metadata.composition == "multiplicative":
            coarse_input = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        else:
            coarse_input = residual_vec
        coefficients = self.solve_coarse(coarse_input)
        coarse = self.basis.vectors @ coefficients
        return float(self.metadata.damping) * (local + coarse)

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply bounded residual-propagation deflation cycles.

        With a fixed basis this remains a linear stationary-polynomial
        preconditioner:

        ``z_{k+1} = z_k + M_0^{-1}(r - A z_k)``.

        The extra cycles are intended for QI hard seeds where one local/coarse
        correction only damps a slow global mode weakly.
        """

        residual_vec = jnp.asarray(residual).reshape((-1,))
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        for _ in range(max(1, int(self.metadata.correction_cycles))):
            step = self._apply_one_cycle(remaining)
            correction = correction + step
            remaining = remaining - jnp.asarray(self.operator(step)).reshape((-1,))
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov preconditioner hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIDeflationProbe:
    """True-residual acceptance result for a deflated QI candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIDeflationMetadata
    seed_solver: str = "linear_apply"
    cycle_residual_history: tuple[float, ...] = ()
    cycle_coefficients: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
            "seed_solver": self.seed_solver,
            "cycle_residual_history": tuple(float(v) for v in self.cycle_residual_history),
            "cycle_coefficients": tuple(float(v) for v in self.cycle_coefficients),
        }


def _normalize_composition(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"add", "additive"}:
        return "additive"
    if normalized in {"mult", "multiplicative", "field_split", "schur"}:
        return "multiplicative"
    raise ValueError("composition must be 'additive' or 'multiplicative'")


def _append_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike,
    label: str,
    *,
    total_size: int,
) -> None:
    vector = jnp.asarray(values).reshape((-1,))
    if int(vector.shape[0]) != int(total_size):
        raise ValueError(f"candidate {label!r} has length {vector.shape[0]}, expected {total_size}")
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector / norm)
    labels.append(str(label))


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2 or int(q.shape[1]) == 0:
        return jnp.zeros_like(q)
    columns = [jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))]
    return jnp.stack(columns, axis=1)


def _regularized_action_least_squares(a: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a_arr = jnp.asarray(a)
    rhs_arr = jnp.asarray(rhs).reshape((-1,))
    if a_arr.ndim != 2:
        raise ValueError("least-squares operator must be a matrix")
    if int(a_arr.shape[1]) == 0:
        return jnp.zeros((0,), dtype=rhs_arr.dtype)
    gram = jnp.conjugate(a_arr).T @ a_arr
    normal_rhs = jnp.conjugate(a_arr).T @ rhs_arr
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(float(rcond), 1.0e-14), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _singular_diagnostics(operator_on_basis: ArrayLike) -> tuple[float, float, float, float]:
    a = jnp.asarray(operator_on_basis)
    if a.ndim != 2 or int(a.shape[1]) == 0:
        return 0.0, 0.0, 0.0, 0.0
    singular_values = np.asarray(jnp.linalg.svd(a, compute_uv=False), dtype=np.float64)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return 0.0, 0.0, 0.0, 0.0
    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))
    frob_sq = float(np.sum(singular_values * singular_values))
    stable_rank = frob_sq / max(max_sv * max_sv, 1.0e-300)
    condition = max_sv / max(min_sv, 1.0e-300)
    return min_sv, max_sv, stable_rank, condition


def build_rhs1_qi_residual_deflated_preconditioner(
    *,
    operator: LinearOperator,
    local_smoother: LinearOperator,
    residual_seed: ArrayLike,
    extra_directions: Sequence[tuple[str, ArrayLike]] = (),
    krylov_depth: int = 4,
    max_rank: int = 16,
    regularization_rcond: float = 1.0e-12,
    basis_rtol: float = 1.0e-10,
    basis_atol: float = 0.0,
    damping: float = 1.0,
    correction_cycles: int = 1,
    composition: str = "multiplicative",
    include_raw_residual: bool = False,
) -> RHS1QIDeflatedPreconditioner:
    """Build a residual-deflated QI preconditioner from a current residual.

    The correction basis is generated from ``S^{-1} r`` and the bounded
    preconditioned Krylov sequence ``(S^{-1} A)^k S^{-1} r``.  Optional
    ``extra_directions`` may add physics-informed block-Schur vectors from
    another builder; all directions are normalized before rank gating to avoid
    large-norm adaptive vectors hiding useful low-norm modes.
    """

    residual = jnp.asarray(residual_seed).reshape((-1,))
    total_size = int(residual.shape[0])
    if total_size <= 0:
        raise ValueError("residual_seed must be non-empty")
    composition_use = _normalize_composition(composition)
    depth_use = max(0, int(krylov_depth))
    max_rank_use = max(0, int(max_rank))
    cycles_use = max(1, int(correction_cycles))
    columns: list[ArrayLike] = []
    labels: list[str] = []

    if bool(include_raw_residual):
        _append_candidate(columns, labels, residual, "residual", total_size=total_size)

    z = jnp.asarray(local_smoother(residual)).reshape((-1,))
    _append_candidate(columns, labels, z, "local_smoother(residual)", total_size=total_size)
    for depth in range(depth_use):
        az = jnp.asarray(operator(z)).reshape((-1,))
        z = jnp.asarray(local_smoother(az)).reshape((-1,))
        _append_candidate(
            columns,
            labels,
            z,
            f"preconditioned_krylov:{depth + 1}",
            total_size=total_size,
        )

    for label, direction in extra_directions:
        _append_candidate(columns, labels, direction, f"extra:{label}", total_size=total_size)

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=residual.dtype)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(basis_rtol),
        atol=float(basis_atol),
        max_rank=max_rank_use,
    )
    operator_on_basis = _apply_operator_to_basis(operator, basis.vectors)
    op_norm = float(jnp.linalg.norm(operator_on_basis)) if int(basis.metadata.rank) > 0 else 0.0
    min_sv, max_sv, stable_rank, condition = _singular_diagnostics(operator_on_basis)
    metadata = RHS1QIDeflationMetadata(
        total_size=total_size,
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        discarded_count=int(basis.metadata.discarded_count),
        requested_krylov_depth=depth_use,
        basis_shape=tuple(int(v) for v in basis.vectors.shape),
        operator_on_basis_shape=tuple(int(v) for v in operator_on_basis.shape),
        operator_on_basis_norm=op_norm,
        stable_rank=stable_rank,
        condition_estimate=condition,
        min_singular_value=min_sv,
        max_singular_value=max_sv,
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
        correction_cycles=cycles_use,
        composition=composition_use,
        accepted_labels=tuple(basis.metadata.accepted_labels),
        candidate_labels=tuple(basis.metadata.candidate_labels),
        device_resident=True,
        reason="built" if int(basis.metadata.rank) > 0 else "empty_rank",
    )
    return RHS1QIDeflatedPreconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
        operator_on_basis=operator_on_basis,
        metadata=metadata,
    )


def probe_rhs1_qi_deflated_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QIDeflatedPreconditioner,
    min_relative_improvement: float = 0.0,
) -> tuple[ArrayLike, RHS1QIDeflationProbe]:
    """Apply one deflated correction and accept only true-residual improvement."""

    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(x_candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = (
        residual_after_norm / residual_before_norm if residual_before_norm > 0.0 else None
    )
    required = residual_before_norm * max(0.0, 1.0 - float(min_relative_improvement))
    accepted = bool(np.isfinite(residual_after_norm) and residual_after_norm < required)
    probe = RHS1QIDeflationProbe(
        accepted=accepted,
        reason="residual_reduced" if accepted else "residual_not_reduced",
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=preconditioner.metadata,
    )
    return x_candidate if accepted else x_initial, probe


def probe_rhs1_qi_deflated_minres_seed(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QIDeflatedPreconditioner,
    cycles: int,
    min_relative_improvement: float = 0.0,
    regularization_rcond: float | None = None,
) -> tuple[ArrayLike, RHS1QIDeflationProbe]:
    """Build a minimum-residual seed from fixed-basis deflation cycles.

    The generated columns are the stationary residual-propagation corrections

    ``z_k = M_0^{-1} r_k`` and ``r_{k+1} = r_k - A z_k``.

    Rather than accepting the raw accumulated stationary iteration, this helper
    solves the small problem ``min ||A Z c - r||``. This is a seed-only
    GCRO/GMRES-like acceleration for hard QI cases; it is deliberately separate
    from ``as_preconditioner()`` because the optimized coefficients depend on
    the current residual.
    """

    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    max_cycles = max(1, int(cycles))
    correction_columns: list[ArrayLike] = []
    action_columns: list[ArrayLike] = []
    residual_history: list[float] = [residual_before_norm]
    remaining = residual_before
    for _ in range(max_cycles):
        step = jnp.asarray(preconditioner._apply_one_cycle(remaining)).reshape((-1,))
        action = jnp.asarray(operator(step)).reshape((-1,))
        step_norm = float(jnp.linalg.norm(step))
        action_norm = float(jnp.linalg.norm(action))
        if (not np.isfinite(step_norm)) or (not np.isfinite(action_norm)) or step_norm <= 0.0 or action_norm <= 0.0:
            break
        correction_columns.append(step)
        action_columns.append(action)
        remaining = remaining - action
        residual_history.append(float(jnp.linalg.norm(remaining)))

    if not correction_columns:
        probe = RHS1QIDeflationProbe(
            accepted=False,
            reason="empty_minres_seed",
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0 if residual_before_norm > 0.0 else None,
            metadata=preconditioner.metadata,
            seed_solver="cycle_minres",
            cycle_residual_history=tuple(residual_history),
        )
        return x_initial, probe

    z_basis = jnp.stack(tuple(correction_columns), axis=1)
    az_basis = jnp.stack(tuple(action_columns), axis=1)
    rcond = (
        float(preconditioner.metadata.regularization_rcond)
        if regularization_rcond is None
        else float(regularization_rcond)
    )
    coefficients = _regularized_action_least_squares(az_basis, residual_before, rcond=rcond)
    dx = z_basis @ coefficients
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(x_candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = (
        residual_after_norm / residual_before_norm if residual_before_norm > 0.0 else None
    )
    required = residual_before_norm * max(0.0, 1.0 - float(min_relative_improvement))
    accepted = bool(np.isfinite(residual_after_norm) and residual_after_norm < required)
    probe = RHS1QIDeflationProbe(
        accepted=accepted,
        reason="residual_reduced" if accepted else "residual_not_reduced",
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=preconditioner.metadata,
        seed_solver="cycle_minres",
        cycle_residual_history=tuple(residual_history),
        cycle_coefficients=tuple(float(v) for v in np.asarray(coefficients, dtype=np.float64).reshape((-1,))),
    )
    return x_candidate if accepted else x_initial, probe


__all__ = [
    "RHS1QIDeflatedPreconditioner",
    "RHS1QIDeflationMetadata",
    "RHS1QIDeflationProbe",
    "build_rhs1_qi_residual_deflated_preconditioner",
    "probe_rhs1_qi_deflated_correction",
    "probe_rhs1_qi_deflated_minres_seed",
]
