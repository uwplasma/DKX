"""RHSMode=1 device-QI block-Schur/angular/radial coarse preconditioner.

This module is intentionally driver-independent.  It builds a bounded,
structure-informed coarse space and a pure JAX preconditioner action that can be
probed against the true residual before a future driver integration accepts it.
The action is a local smoother plus a least-squares coarse Schur correction:

``M^{-1} r = S^{-1} r + Q c,  c = argmin ||A Q c - (r - A S^{-1} r)||``.

The fail-closed probe returns the original iterate unless the measured residual
strictly decreases by the requested margin.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from sfincs_jax.solvers.preconditioners.qi.coarse import (
    RHS1QICoarseBasis,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)

ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QIBlockSchurMetadata:
    """Build diagnostics for a block-Schur/angular/radial QI candidate."""

    total_size: int
    candidate_count: int
    rank: int
    numerical_rank: int
    discarded_count: int
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    stable_rank: float
    condition_estimate: float
    min_singular_value: float
    max_singular_value: float
    singular_threshold: float
    regularization_rcond: float
    damping: float
    accepted_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for solver traces."""

        return {
            "total_size": int(self.total_size),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "numerical_rank": int(self.numerical_rank),
            "discarded_count": int(self.discarded_count),
            "operator_on_basis_shape": tuple(int(v) for v in self.operator_on_basis_shape),
            "operator_on_basis_norm": float(self.operator_on_basis_norm),
            "coarse_operator_shape": tuple(int(v) for v in self.coarse_operator_shape),
            "coarse_operator_norm": float(self.coarse_operator_norm),
            "stable_rank": float(self.stable_rank),
            "condition_estimate": float(self.condition_estimate),
            "min_singular_value": float(self.min_singular_value),
            "max_singular_value": float(self.max_singular_value),
            "singular_threshold": float(self.singular_threshold),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
            "accepted_labels": self.accepted_labels,
            "candidate_labels": self.candidate_labels,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIBlockSchurPreconditioner:
    """Reusable JAX-compatible RHSMode=1 QI preconditioner candidate."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    metadata: RHS1QIBlockSchurMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Return the coarse coefficients for a residual vector."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        return _regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one local-plus-coarse block-Schur correction."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return float(self.metadata.damping) * local
        remaining = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        coarse = self.basis.vectors @ self.solve_coarse(remaining)
        return float(self.metadata.damping) * (local + coarse)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return the preconditioner action for Krylov hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIBlockSchurProbe:
    """True-residual acceptance result for a block-Schur QI candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIBlockSchurMetadata

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _append(columns: list[ArrayLike], labels: list[str], values: ArrayLike | None, label: str) -> None:
    if values is not None:
        columns.append(jnp.asarray(values).reshape((-1,)))
        labels.append(str(label))


def _block_offsets(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    offsets = [0]
    for size in layout.block_sizes:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _centered_unit_weights(values: Sequence[float]) -> tuple[float, ...] | None:
    raw = tuple(float(value) for value in values)
    if not raw:
        return None
    mean = sum(raw) / float(len(raw))
    centered = tuple(value - mean for value in raw)
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple(value / scale for value in centered)


def _centered_power_weights(values: Sequence[float], degree: int) -> tuple[float, ...] | None:
    linear = _centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple(value ** int(degree) for value in linear)
    return _centered_unit_weights(powered)


def _block_weighted_constant(
    layout: RHS1QICoarseBlockLayout,
    weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    if not any(float(value) != 0.0 for value in weights):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _weights_for_blocks(
    n_blocks: int,
    weighted_blocks: Sequence[tuple[int, float]],
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _angular_specs(
    layout: RHS1QICoarseBlockLayout,
    *,
    max_angular_mode: int,
    dtype: Any,
) -> tuple[tuple[str, ArrayLike], ...]:
    max_mode = max(0, int(max_angular_mode))
    if max_mode <= 0:
        return ()
    theta = jnp.arange(int(layout.n_theta), dtype=dtype)
    zeta = jnp.arange(int(layout.n_zeta), dtype=dtype)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    specs: list[tuple[str, ArrayLike]] = []
    for mode in range(1, max_mode + 1):
        if int(layout.n_theta) > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if int(layout.n_zeta) > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if int(layout.n_theta) > 1 and int(layout.n_zeta) > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"mixed_cos_plus{mode}", jnp.cos(theta_phase + zeta_phase)))
            specs.append((f"mixed_sin_plus{mode}", jnp.sin(theta_phase + zeta_phase)))
            specs.append((f"mixed_cos_minus{mode}", jnp.cos(theta_phase - zeta_phase)))
            specs.append((f"mixed_sin_minus{mode}", jnp.sin(theta_phase - zeta_phase)))
    return tuple(specs)


def _angular_candidate(
    layout: RHS1QICoarseBlockLayout,
    angular_values: ArrayLike,
    weights: Sequence[float] | None,
    dtype: Any,
) -> ArrayLike | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 1:
        return None
    if weights is not None and len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    if weights is not None and not any(float(value) != 0.0 for value in weights):
        return None

    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for block_index, block_size in enumerate(layout.block_sizes):
        if int(block_size) % n_angular != 0:
            return None
        repeats = int(block_size) // n_angular
        weight = 1.0 if weights is None else float(weights[block_index])
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(weight * jnp.tile(angular, repeats))
    return values


def build_rhs1_qi_block_schur_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    max_candidates: int = 128,
    max_angular_mode: int = 2,
    max_radial_degree: int = 2,
    include_global: bool = True,
    include_radial: bool = True,
    include_angular: bool = True,
    include_radial_angular: bool = True,
    include_block_schur: bool = True,
    include_block_schur_angular: bool = True,
    include_blocks: bool = True,
    dtype: Any = jnp.float64,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build deterministic block-Schur/angular/radial coarse candidates."""

    limit = max(0, int(max_candidates))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    n_blocks = len(layout.block_sizes)
    block_x = tuple(float(value) for value in (layout.block_x or tuple(range(n_blocks))))

    def has_room() -> bool:
        return len(columns) < limit

    def add(values: ArrayLike | None, label: str) -> None:
        if has_room():
            _append(columns, labels, values, label)

    if limit <= 0:
        return _empty_matrix(layout.total_size, dtype), ()

    angular_specs = _angular_specs(layout, max_angular_mode=max_angular_mode, dtype=dtype)
    radial_weights: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(max_radial_degree)) + 1):
        weights = _centered_power_weights(block_x, degree)
        if weights is not None:
            radial_weights.append((f"radial:p{degree}", weights))

    if include_global:
        add(jnp.ones((layout.total_size,), dtype=dtype), "global")

    if include_radial:
        for label, weights in radial_weights:
            add(_block_weighted_constant(layout, weights, dtype), label)

    if include_angular:
        for label, values in angular_specs:
            add(_angular_candidate(layout, values, None, dtype), f"angular:{label}")

    if include_radial_angular:
        for radial_label, weights in radial_weights:
            for angular_label, values in angular_specs:
                add(
                    _angular_candidate(layout, values, weights, dtype),
                    f"{radial_label}*angular:{angular_label}",
                )

    if include_block_schur or include_block_schur_angular:
        x_to_blocks: dict[int, list[int]] = {}
        for block_index, x_value in enumerate(layout.block_x or tuple(range(n_blocks))):
            x_to_blocks.setdefault(int(x_value), []).append(block_index)
        x_values = sorted(x_to_blocks)
        schur_weights: list[tuple[str, tuple[float, ...]]] = []
        if include_block_schur:
            for left_x, right_x in zip(x_values, x_values[1:], strict=False):
                weighted = tuple((block, -1.0) for block in x_to_blocks[left_x]) + tuple(
                    (block, 1.0) for block in x_to_blocks[right_x]
                )
                weights = _weights_for_blocks(n_blocks, weighted)
                schur_weights.append((f"schur:x_diff:{left_x}->{right_x}", weights))
                add(_block_weighted_constant(layout, weights, dtype), schur_weights[-1][0])
            for left_x, center_x, right_x in zip(x_values, x_values[1:], x_values[2:], strict=False):
                weighted = (
                    tuple((block, 1.0) for block in x_to_blocks[left_x])
                    + tuple((block, -2.0) for block in x_to_blocks[center_x])
                    + tuple((block, 1.0) for block in x_to_blocks[right_x])
                )
                weights = _weights_for_blocks(n_blocks, weighted)
                schur_weights.append((f"schur:x_curve:{left_x},{center_x},{right_x}", weights))
                add(_block_weighted_constant(layout, weights, dtype), schur_weights[-1][0])

        if include_block_schur_angular:
            for schur_label, weights in schur_weights:
                for angular_label, values in angular_specs:
                    add(
                        _angular_candidate(layout, values, weights, dtype),
                        f"{schur_label}*angular:{angular_label}",
                    )

    if include_blocks:
        for block_index in range(n_blocks):
            weights = _weights_for_blocks(n_blocks, ((block_index, 1.0),))
            add(_block_weighted_constant(layout, weights, dtype), f"block:{block_index}")

    if not columns:
        return _empty_matrix(layout.total_size, dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def build_rhs1_qi_block_schur_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    max_rank: int = 48,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QICoarseBasis:
    """Build and rank-gate the block-Schur/angular/radial coarse basis."""

    candidates, labels = build_rhs1_qi_block_schur_candidates(
        layout,
        dtype=dtype,
        **candidate_options,
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=rtol,
        atol=atol,
        max_rank=max(0, int(max_rank)),
    )


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if int(q.shape[1]) == 0:
        return _empty_matrix(int(q.shape[0]), q.dtype)
    return jnp.stack(tuple(jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))), axis=1)


def _regularized_action_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _conditioning_metadata(
    operator_on_basis: ArrayLike,
    *,
    rcond: float,
    atol: float,
) -> tuple[int, float, float, float, float, float]:
    aq = jnp.asarray(operator_on_basis)
    if aq.size == 0 or int(aq.shape[1]) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    singular_values = jnp.linalg.svd(aq, compute_uv=False)
    max_sv = float(jnp.max(singular_values))
    min_sv = float(jnp.min(singular_values))
    threshold = max(float(atol), max_sv * max(0.0, float(rcond)))
    numerical_rank = int(jnp.sum(singular_values > threshold))
    frob_sq = float(jnp.sum(singular_values * singular_values))
    stable_rank = 0.0 if max_sv <= 0.0 else frob_sq / (max_sv * max_sv)
    denom = max(min_sv, threshold, 1.0e-300)
    condition = 0.0 if max_sv <= 0.0 else max_sv / denom
    return numerical_rank, stable_rank, condition, min_sv, max_sv, threshold


def build_rhs1_qi_block_schur_preconditioner(
    *,
    operator: LinearOperator,
    layout: RHS1QICoarseBlockLayout | None = None,
    basis: RHS1QICoarseBasis | None = None,
    local_smoother: LinearOperator | None = None,
    regularization_rcond: float = 1.0e-12,
    conditioning_atol: float = 0.0,
    damping: float = 1.0,
    **basis_options: Any,
) -> RHS1QIBlockSchurPreconditioner:
    """Build a reusable block-Schur/angular/radial QI preconditioner."""

    if basis is None:
        if layout is None:
            raise ValueError("either basis or layout must be provided")
        basis = build_rhs1_qi_block_schur_basis(layout, **basis_options)
    q = jnp.asarray(basis.vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")

    smoother = (lambda residual: jnp.zeros_like(jnp.asarray(residual).reshape((-1,)))) if local_smoother is None else local_smoother
    rank = int(q.shape[1])
    if rank <= 0:
        operator_on_basis = _empty_matrix(int(q.shape[0]), q.dtype)
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        operator_on_basis = _apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis

    numerical_rank, stable_rank, condition, min_sv, max_sv, threshold = _conditioning_metadata(
        operator_on_basis,
        rcond=float(regularization_rcond),
        atol=float(conditioning_atol),
    )
    reason = "built" if rank > 0 and numerical_rank > 0 else "empty_or_rank_deficient"
    metadata = RHS1QIBlockSchurMetadata(
        total_size=int(q.shape[0]),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        operator_on_basis_shape=tuple(int(v) for v in operator_on_basis.shape),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis)) if rank > 0 else 0.0,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0,
        stable_rank=float(stable_rank),
        condition_estimate=float(condition),
        min_singular_value=float(min_sv),
        max_singular_value=float(max_sv),
        singular_threshold=float(threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
        accepted_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
        candidate_labels=tuple(str(label) for label in basis.metadata.candidate_labels),
        reason=reason,
    )
    return RHS1QIBlockSchurPreconditioner(
        operator=operator,
        local_smoother=smoother,
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )


def probe_rhs1_qi_block_schur_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QIBlockSchurPreconditioner,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> tuple[ArrayLike, RHS1QIBlockSchurProbe]:
    """Apply one correction and accept it only if the true residual improves."""

    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIBlockSchurProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=preconditioner.metadata,
        )
        return x_initial, probe
    if int(preconditioner.metadata.rank) <= 0:
        probe = RHS1QIBlockSchurProbe(
            accepted=False,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            metadata=preconditioner.metadata,
        )
        return x_initial, probe

    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(float(acceptance_atol), residual_before_norm * max(0.0, float(min_relative_improvement)))
    finite = bool(jnp.isfinite(jnp.asarray(residual_after_norm)))
    accepted = finite and residual_after_norm < residual_before_norm - required_drop
    if accepted:
        reason = "residual_reduced"
    elif not finite:
        reason = "nonfinite_candidate"
    else:
        reason = "not_reduced"
    probe = RHS1QIBlockSchurProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm if finite else residual_before_norm,
        improvement_ratio=float(improvement_ratio) if finite else None,
        metadata=preconditioner.metadata,
    )
    return candidate if accepted else x_initial, probe


__all__ = [
    "RHS1QIBlockSchurMetadata",
    "RHS1QIBlockSchurPreconditioner",
    "RHS1QIBlockSchurProbe",
    "build_rhs1_qi_block_schur_basis",
    "build_rhs1_qi_block_schur_candidates",
    "build_rhs1_qi_block_schur_preconditioner",
    "probe_rhs1_qi_block_schur_correction",
]
